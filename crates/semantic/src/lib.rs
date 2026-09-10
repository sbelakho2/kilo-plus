//! Faktor semantic-provider abstraction (audit 48-54/58/59).
//!
//! A semantic provider is optional acceleration: coverage, context, deltas,
//! affected sets, verification metadata and constrained edit proposals. The
//! runtime never requires one — [`registry::SemanticProviderRegistry`]
//! selects providers by [`types::SemanticCapabilities`] only and falls back
//! to [`fallback::GenericSemanticFallback`], which composes the existing
//! index/LSP/tree-sitter/git/verification metadata seams through in-crate
//! trait hook points (no cross-crate dependency).
//!
//! Non-negotiables enforced here, each with adversarial tests:
//!
//! - **Provider output is DATA.** Envelopes are validated against provenance
//!   rules from `faktor-evidence`; `UserPolicy` authority can never be
//!   laundered through a provider payload and instruction-like text stays
//!   data.
//! - **Every response is bounded and validated.** Wrong workspace, stale
//!   snapshot, unknown schema, oversized payload, duplicate entity ids and
//!   out-of-workspace refs are typed errors.
//! - **Calls are guarded.** Cancellation and deadlines are observed before
//!   the provider is polled; panics become typed provider-crash errors;
//!   provider failures degrade to the fallback, caller cancellation never
//!   does.
//! - **Cache is bounded and delta-scoped.** Keys are
//!   `{workspace, source revision, provider id, provider version, schema
//!   version}`; an unchanged re-request never calls the provider; a delta
//!   invalidates only entries covering changed entity ids.
//! - **The crate only proposes edits.** Constrained edit proposals carry
//!   expected source hashes and validated workspace-relative paths; Faktor
//!   applies them through its own machinery. No filesystem API is used
//!   anywhere in this crate (locked by a source-scan test).

pub mod cache;
pub mod fallback;
pub mod registry;
pub mod risk;
pub mod types;

pub use cache::{
    CacheEntry, SemanticCache, SemanticCacheEntryKey, SemanticCacheKey, SemanticCached,
    SemanticCachedProvider, SemanticInvalidation, SemanticSnapshotRef,
};
pub use fallback::{
    GenericSemanticFallback, GitMetadata, IndexMetadata, LspMetadata, ProposedPatch,
    SemanticEditRequest, SemanticFileEdit, SourceHashView, TreeSitterMetadata,
    VerificationMetadata, GENERIC_FALLBACK_ID,
};
pub use registry::{guard_call, GuardedCall, SemanticProviderRegistry, SemanticSelection};
pub use risk::{capability_intersection, RiskPolicy};
pub use types::{
    AffectedRequest, AffectedSet, BoxFuture, RiskLevel, SemanticCall, SemanticCapabilities,
    SemanticCheck, SemanticContextItem, SemanticContextPack, SemanticContextRequest, SemanticDelta,
    SemanticDeltaChange, SemanticDeltaKind, SemanticDeltaRequest, SemanticEntityId,
    SemanticEntityRef, SemanticEnvelope, SemanticError, SemanticExpectation,
    SemanticExplainRequest, SemanticExplanation, SemanticOp, SemanticPayload, SemanticProvider,
    SemanticProviderId, SemanticResponseCaps, SemanticRisk, SemanticSnapshot, SemanticSnapshotId,
    SemanticSnapshotRequest, SemanticVerification, SemanticVerifyRequest, WorkspacePath,
    DEFAULT_MAX_PAYLOAD_BYTES, MAX_ENTITY_ID_BYTES, MAX_ENTITY_REFS, MAX_PATH_BYTES,
    MAX_PROVIDER_ID_BYTES, MAX_QUERY_BYTES, SEMANTIC_SCHEMA_VERSION,
};

#[cfg(test)]
pub(crate) mod test_support {
    use std::future::Future;
    use std::task::{Context, Poll, Waker};

    use faktor_core::{CancellationToken, OpId, SessionId, WorkspaceId};

    use crate::types::{
        SemanticCall, SemanticEntityId, SemanticEntityRef, SemanticProviderId, SemanticSnapshotId,
        WorkspacePath, SEMANTIC_SCHEMA_VERSION,
    };

    pub(crate) const TEST_PROVIDER_ID: &str = "test-provider";

    /// Minimal single-poll executor used by unit tests (the crate has no
    /// async runtime by design; all test futures complete on first poll or
    /// are explicitly polled again after a state change).
    pub(crate) fn block_on<F: Future>(future: F) -> F::Output {
        let mut future = Box::pin(future);
        let mut cx = Context::from_waker(Waker::noop());
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(value) => value,
            Poll::Pending => panic!("future parked without a ready executor"),
        }
    }

    pub(crate) fn provider_id(name: &str) -> SemanticProviderId {
        SemanticProviderId::parse(name).expect("test provider id is valid")
    }

    pub(crate) fn call() -> SemanticCall {
        SemanticCall::new(
            OpId::new(1),
            SessionId::new(1),
            WorkspaceId::new(1),
            0,
            CancellationToken::new(),
        )
    }

    pub(crate) fn snapshot(workspace: WorkspaceId, revision: &str) -> SemanticSnapshotId {
        SemanticSnapshotId::derive(
            workspace,
            revision,
            &provider_id(TEST_PROVIDER_ID),
            1,
            SEMANTIC_SCHEMA_VERSION,
        )
    }

    pub(crate) fn entity(path: &str, id: &str) -> SemanticEntityRef {
        SemanticEntityRef::new(
            WorkspaceId::new(1),
            WorkspacePath::parse(path).expect("test path is valid"),
            SemanticEntityId::parse(id).expect("test entity id is valid"),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    /// Tokens that would mark a production file as touching the filesystem
    /// or spawning processes. The crate proposes; Faktor applies.
    const FORBIDDEN: &[&str] = &[
        "std::fs",
        "faktor_fs",
        "OpenOptions",
        "File::create",
        "File::open",
        "fs::write",
        "fs::remove",
        "remove_file",
        "create_dir",
        "write_all",
        "tempfile",
        "Command::new",
        "std::process",
    ];

    #[test]
    fn production_sources_never_touch_the_filesystem() {
        let src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut scanned = 0usize;
        for entry in std::fs::read_dir(&src_dir).expect("src dir exists") {
            let path = entry.expect("readable dir entry").path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("source is readable");
            // Everything from the first test-gated item onward is test-only;
            // production code above it is what gets certified.
            let production = match text.find("#[cfg(test)]") {
                Some(index) => &text[..index],
                None => text.as_str(),
            };
            for token in FORBIDDEN {
                assert!(
                    !production.contains(token),
                    "{} production code contains forbidden token {token:?}",
                    path.display()
                );
            }
            scanned += 1;
        }
        assert!(
            scanned >= 6,
            "source scan must cover every module, scanned {scanned}"
        );
    }
}
