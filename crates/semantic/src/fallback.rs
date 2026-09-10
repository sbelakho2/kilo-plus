//! Generic semantic fallback and constrained edit proposals (audit
//! 48-54/58/59).
//!
//! The fallback is the in-crate stub that composes the metadata seams the
//! runtime already has — index, language-server, tree-sitter, git and
//! verification adapters — **without depending on any of those crates**.
//! Each seam is a trait; downstream crates register `Arc<dyn ...>`
//! implementations and the fallback merges them. When a seam is absent the
//! fallback still answers with honest degraded/empty payloads, so ordinary
//! operation never fails solely because no semantic provider is installed.
//!
//! This module also carries the constrained-edit proposal types. A provider
//! may propose a [`ProposedPatch`]; Faktor validates it against workspace,
//! snapshot, path and source-hash rules and then applies it through its own
//! edit/FS machinery. The crate only proposes: there is no apply function
//! here, and no filesystem API is used anywhere in this crate.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use faktor_core::{Clock, FileHash, SystemClock, WorkspaceId};

use crate::types::{
    AffectedRequest, AffectedSet, BoxFuture, SemanticCall, SemanticCapabilities, SemanticCheck,
    SemanticContextItem, SemanticContextPack, SemanticContextRequest, SemanticDelta,
    SemanticDeltaChange, SemanticDeltaKind, SemanticDeltaRequest, SemanticEntityRef,
    SemanticEnvelope, SemanticError, SemanticExpectation, SemanticExplainRequest,
    SemanticExplanation, SemanticPayload, SemanticProvider, SemanticProviderId,
    SemanticResponseCaps, SemanticSnapshot, SemanticSnapshotId, SemanticSnapshotRequest,
    SemanticVerification, SemanticVerifyRequest, WorkspacePath,
};

/// The fallback's provider id. Not a language and not a vendor.
pub const GENERIC_FALLBACK_ID: &str = "generic-fallback";

/// Index metadata seam: entities, revisions, impacted lookups.
pub trait IndexMetadata: Send + Sync {
    /// Current source revision the index was built from, when known.
    fn snapshot_revision(&self, _workspace: WorkspaceId) -> Option<String> {
        None
    }

    fn entity_count(&self, _workspace: WorkspaceId) -> u64 {
        0
    }

    fn context_items(
        &self,
        _workspace: WorkspaceId,
        _query: &str,
        _limit: usize,
    ) -> Vec<SemanticContextItem> {
        Vec::new()
    }

    fn impacted_entities(
        &self,
        _workspace: WorkspaceId,
        _changed: &[SemanticEntityRef],
    ) -> Vec<SemanticEntityRef> {
        Vec::new()
    }
}

/// Language-server metadata seam: definitions/references for an entity.
pub trait LspMetadata: Send + Sync {
    fn definitions(&self, _entity: &SemanticEntityRef) -> Vec<SemanticEntityRef> {
        Vec::new()
    }
}

/// Tree-sitter metadata seam: structural outline of one file.
pub trait TreeSitterMetadata: Send + Sync {
    fn outline(&self, _path: &WorkspacePath) -> Vec<SemanticEntityRef> {
        Vec::new()
    }
}

/// Git metadata seam: entities changed between two revisions.
pub trait GitMetadata: Send + Sync {
    fn changed_entities(
        &self,
        _workspace: WorkspaceId,
        _from_revision: &str,
        _to_revision: &str,
    ) -> Vec<SemanticEntityRef> {
        Vec::new()
    }
}

/// Verification metadata seam: checks relevant to a workspace.
pub trait VerificationMetadata: Send + Sync {
    fn checks(&self, _workspace: WorkspaceId) -> Vec<SemanticCheck> {
        Vec::new()
    }
}

/// The in-crate fallback provider. All seams are optional; the fallback
/// answers every operation regardless.
pub struct GenericSemanticFallback {
    index: Option<Arc<dyn IndexMetadata>>,
    lsp: Option<Arc<dyn LspMetadata>>,
    tree_sitter: Option<Arc<dyn TreeSitterMetadata>>,
    git: Option<Arc<dyn GitMetadata>>,
    verification: Option<Arc<dyn VerificationMetadata>>,
    clock: Arc<dyn Clock>,
}

impl Default for GenericSemanticFallback {
    fn default() -> Self {
        Self {
            index: None,
            lsp: None,
            tree_sitter: None,
            git: None,
            verification: None,
            clock: Arc::new(SystemClock),
        }
    }
}

impl GenericSemanticFallback {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_index(mut self, index: Arc<dyn IndexMetadata>) -> Self {
        self.index = Some(index);
        self
    }

    pub fn with_lsp(mut self, lsp: Arc<dyn LspMetadata>) -> Self {
        self.lsp = Some(lsp);
        self
    }

    pub fn with_tree_sitter(mut self, tree_sitter: Arc<dyn TreeSitterMetadata>) -> Self {
        self.tree_sitter = Some(tree_sitter);
        self
    }

    pub fn with_git(mut self, git: Arc<dyn GitMetadata>) -> Self {
        self.git = Some(git);
        self
    }

    pub fn with_verification(mut self, verification: Arc<dyn VerificationMetadata>) -> Self {
        self.verification = Some(verification);
        self
    }

    pub fn with_clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = clock;
        self
    }

    fn provider_id(&self) -> SemanticProviderId {
        SemanticProviderId::parse(GENERIC_FALLBACK_ID)
            .expect("static generic fallback provider id is valid")
    }

    fn now_ms(&self) -> i64 {
        self.clock.now_ms()
    }

    fn envelope<T: SemanticPayload>(
        &self,
        workspace: WorkspaceId,
        snapshot_id: SemanticSnapshotId,
        payload: T,
    ) -> SemanticEnvelope<T> {
        SemanticEnvelope::new(
            self.provider_id(),
            self.version(),
            workspace,
            snapshot_id,
            self.now_ms(),
            payload,
        )
    }
}

fn revision_tree_hash(workspace: WorkspaceId, revision: &str) -> FileHash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&workspace.raw().to_le_bytes());
    hasher.update(&(revision.len() as u64).to_le_bytes());
    hasher.update(revision.as_bytes());
    FileHash::from(*hasher.finalize().as_bytes())
}

impl SemanticProvider for GenericSemanticFallback {
    fn id(&self) -> SemanticProviderId {
        self.provider_id()
    }

    fn version(&self) -> u32 {
        1
    }

    fn capabilities(&self) -> SemanticCapabilities {
        SemanticCapabilities::ALL
    }

    fn snapshot(
        &self,
        request: SemanticSnapshotRequest,
    ) -> BoxFuture<'_, Result<SemanticEnvelope<SemanticSnapshot>, SemanticError>> {
        let revision = self
            .index
            .as_ref()
            .and_then(|index| index.snapshot_revision(request.workspace))
            .unwrap_or_else(|| request.source_revision.clone());
        let entity_count = self
            .index
            .as_ref()
            .map_or(0, |index| index.entity_count(request.workspace));
        let snapshot_id = SemanticSnapshotId::derive(
            request.workspace,
            &revision,
            &self.provider_id(),
            self.version(),
            SemanticEnvelope::<SemanticSnapshot>::SCHEMA_VERSION,
        );
        let payload = SemanticSnapshot {
            source_revision: revision.clone(),
            tree_hash: revision_tree_hash(request.workspace, &revision),
            entity_count,
            created_ms: self.now_ms(),
        };
        let envelope = self.envelope(request.workspace, snapshot_id, payload);
        Box::pin(async move { Ok(envelope) })
    }

    fn context(
        &self,
        request: SemanticContextRequest,
    ) -> BoxFuture<'_, Result<SemanticEnvelope<SemanticContextPack>, SemanticError>> {
        let degraded = self.index.is_none();
        let candidates = self.index.as_ref().map_or_else(Vec::new, |index| {
            index.context_items(request.workspace, &request.query, request.max_items)
        });
        let mut kept = Vec::new();
        let mut total_bytes = 0usize;
        let mut truncated = false;
        for item in candidates {
            if kept.len() >= request.max_items {
                truncated = true;
                break;
            }
            let item_bytes = item.excerpt.len();
            if total_bytes.saturating_add(item_bytes) > request.max_bytes {
                truncated = true;
                break;
            }
            total_bytes += item_bytes;
            kept.push(item);
        }
        let payload = SemanticContextPack {
            items: kept,
            truncated,
            total_bytes,
            degraded,
        };
        let envelope = self.envelope(request.workspace, request.snapshot_id, payload);
        Box::pin(async move { Ok(envelope) })
    }

    fn delta(
        &self,
        request: SemanticDeltaRequest,
    ) -> BoxFuture<'_, Result<SemanticEnvelope<SemanticDelta>, SemanticError>> {
        let changed = self.git.as_ref().map_or_else(Vec::new, |git| {
            git.changed_entities(
                request.workspace,
                &request.from_source_revision,
                &request.to_source_revision,
            )
        });
        let changes = changed
            .into_iter()
            .map(|entity| SemanticDeltaChange {
                entity,
                kind: SemanticDeltaKind::Modified,
                old_hash: None,
                new_hash: None,
            })
            .collect();
        let to_snapshot = SemanticSnapshotId::derive(
            request.workspace,
            &request.to_source_revision,
            &self.provider_id(),
            self.version(),
            SemanticEnvelope::<SemanticDelta>::SCHEMA_VERSION,
        );
        let payload = SemanticDelta {
            workspace: request.workspace,
            from_snapshot: request.from_snapshot,
            to_snapshot,
            changes,
            degraded: self.git.is_none(),
        };
        let envelope = self.envelope(request.workspace, to_snapshot, payload);
        Box::pin(async move { Ok(envelope) })
    }

    fn affected(
        &self,
        request: AffectedRequest,
    ) -> BoxFuture<'_, Result<SemanticEnvelope<AffectedSet>, SemanticError>> {
        let mut found = BTreeSet::new();
        if let Some(index) = &self.index {
            found.extend(index.impacted_entities(request.workspace, &request.changed));
        }
        if let Some(lsp) = &self.lsp {
            for changed in &request.changed {
                found.extend(lsp.definitions(changed));
            }
        }
        if let Some(tree_sitter) = &self.tree_sitter {
            for changed in &request.changed {
                found.extend(tree_sitter.outline(&changed.path));
            }
        }
        let payload = AffectedSet {
            affected: found.into_iter().collect(),
            tests: Vec::new(),
            degraded: self.index.is_none() && self.lsp.is_none(),
        };
        let envelope = self.envelope(request.workspace, request.snapshot_id, payload);
        Box::pin(async move { Ok(envelope) })
    }

    fn verify(
        &self,
        request: SemanticVerifyRequest,
    ) -> BoxFuture<'_, Result<SemanticEnvelope<SemanticVerification>, SemanticError>> {
        let checks = self
            .verification
            .as_ref()
            .map_or_else(Vec::new, |verification| {
                verification.checks(request.workspace)
            });
        // No checks means we do not know: report the gap, never a fake pass.
        let (passed, degraded) = if checks.is_empty() {
            (false, true)
        } else {
            (checks.iter().all(|check| check.passed), false)
        };
        let payload = SemanticVerification {
            passed,
            degraded,
            checks,
        };
        let envelope = self.envelope(request.workspace, request.snapshot_id, payload);
        Box::pin(async move { Ok(envelope) })
    }

    fn explain(
        &self,
        request: SemanticExplainRequest,
    ) -> BoxFuture<'_, Result<SemanticEnvelope<SemanticExplanation>, SemanticError>> {
        let citations = self
            .lsp
            .as_ref()
            .map_or_else(Vec::new, |lsp| lsp.definitions(&request.entity));
        let text = format!(
            "entity {}: no provider explanation available; question recorded as data ({} chars)",
            request.entity.entity_id,
            request.question.len()
        );
        let payload = SemanticExplanation {
            entity: request.entity,
            text,
            citations,
        };
        let envelope = self.envelope(request.workspace, request.snapshot_id, payload);
        Box::pin(async move { Ok(envelope) })
    }
}

/// A request for a constrained semantic edit proposal.
#[derive(Debug, Clone)]
pub struct SemanticEditRequest {
    pub call: SemanticCall,
    pub workspace: WorkspaceId,
    pub snapshot_id: SemanticSnapshotId,
    pub entity: SemanticEntityRef,
    pub intent: String,
}

/// One proposed file replacement. The path type already refuses absolute
/// paths and `..`; validation re-checks defense in depth.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SemanticFileEdit {
    pub path: WorkspacePath,
    pub replacement: String,
}

/// A constrained edit proposal. The provider proposes; Faktor validates and
/// applies through its own machinery. There is deliberately no `apply` here.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ProposedPatch {
    pub workspace: WorkspaceId,
    pub snapshot_id: SemanticSnapshotId,
    pub file_edits: Vec<SemanticFileEdit>,
    /// Expected current content hash per workspace-relative path.
    pub expected_source_hashes: BTreeMap<String, FileHash>,
    pub semantic_delta: SemanticDelta,
}

/// Read-only view of current source hashes, supplied by the Faktor-side
/// applier. This crate never reads files itself.
pub trait SourceHashView {
    fn current_hash(&self, path: &WorkspacePath) -> Option<FileHash>;
}

impl ProposedPatch {
    /// Validate a proposal BEFORE any apply stage: workspace/snapshot
    /// identity, path safety and uniqueness, complete expected-hash coverage,
    /// freshness against the applier's hash view, delta refs inside the
    /// workspace, and payload bounds.
    pub fn validate(
        &self,
        expected_workspace: WorkspaceId,
        expected_snapshot: SemanticSnapshotId,
        view: &dyn SourceHashView,
        caps: &SemanticResponseCaps,
    ) -> Result<(), SemanticError> {
        if self.workspace != expected_workspace {
            return Err(SemanticError::WorkspaceMismatch {
                expected: expected_workspace,
                actual: self.workspace,
            });
        }
        if self.snapshot_id != expected_snapshot {
            return Err(SemanticError::SnapshotMismatch {
                expected: expected_snapshot,
                actual: self.snapshot_id,
            });
        }
        if self.semantic_delta.workspace != expected_workspace {
            return Err(SemanticError::WorkspaceMismatch {
                expected: expected_workspace,
                actual: self.semantic_delta.workspace,
            });
        }
        let expectation = SemanticExpectation::new(expected_workspace, expected_snapshot);
        if self.file_edits.is_empty() {
            return Err(SemanticError::Refused(
                "proposed patch carries no file edits".to_string(),
            ));
        }

        let mut seen_paths = BTreeSet::new();
        let mut total_bytes = 0usize;
        for edit in &self.file_edits {
            edit.path.ensure_safe()?;
            if !seen_paths.insert(edit.path.as_str()) {
                return Err(SemanticError::DuplicateEntity(edit.path.to_string()));
            }
            let Some(expected) = self.expected_source_hashes.get(edit.path.as_str()) else {
                return Err(SemanticError::Malformed(format!(
                    "no expected source hash for edit path {}",
                    edit.path
                )));
            };
            let actual = view.current_hash(&edit.path);
            if actual.as_ref() != Some(expected) {
                return Err(SemanticError::StaleHash {
                    path: edit.path.to_string(),
                    expected: *expected,
                    actual,
                });
            }
            total_bytes = total_bytes.saturating_add(edit.replacement.len());
        }

        // Hash-map keys are attacker-controlled strings: they parse through
        // the same path grammar, so `../../outside` is refused here too.
        for key in self.expected_source_hashes.keys() {
            let path = WorkspacePath::parse(key)?;
            if !seen_paths.contains(path.as_str()) {
                return Err(SemanticError::Malformed(format!(
                    "expected hash for {key:?} has no matching file edit"
                )));
            }
        }

        // Delta refs must live in the expected workspace and be duplicate
        // free, exactly like any other semantic payload.
        let mut seen_ids = BTreeSet::new();
        for change in &self.semantic_delta.changes {
            change.entity.validate_for(expectation.workspace)?;
            if !seen_ids.insert(change.entity.entity_id.clone()) {
                return Err(SemanticError::DuplicateEntity(
                    change.entity.entity_id.to_string(),
                ));
            }
        }

        total_bytes = total_bytes.saturating_add(self.semantic_delta.byte_len()?);
        if total_bytes > caps.max_payload_bytes {
            return Err(SemanticError::Oversized {
                max: caps.max_payload_bytes,
                actual: total_bytes,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{block_on, call, entity, snapshot};
    use crate::types::{SemanticContextRequest, SemanticEntityId, WorkspacePath};
    use std::collections::BTreeMap;

    struct TestIndex {
        revision: String,
        items: Vec<SemanticContextItem>,
    }

    impl IndexMetadata for TestIndex {
        fn snapshot_revision(&self, _workspace: WorkspaceId) -> Option<String> {
            Some(self.revision.clone())
        }

        fn entity_count(&self, _workspace: WorkspaceId) -> u64 {
            self.items.len() as u64
        }

        fn context_items(
            &self,
            _workspace: WorkspaceId,
            _query: &str,
            limit: usize,
        ) -> Vec<SemanticContextItem> {
            self.items
                .iter()
                .take(limit.saturating_add(10))
                .cloned()
                .collect()
        }
    }

    struct MapHashes(BTreeMap<String, FileHash>);

    impl SourceHashView for MapHashes {
        fn current_hash(&self, path: &WorkspacePath) -> Option<FileHash> {
            self.0.get(path.as_str()).copied()
        }
    }

    fn context_request() -> SemanticContextRequest {
        SemanticContextRequest {
            call: call(),
            workspace: WorkspaceId::new(1),
            source_revision: "rev-1".to_string(),
            snapshot_id: snapshot(WorkspaceId::new(1), "rev-1"),
            query: "q".to_string(),
            max_items: 8,
            max_bytes: 4096,
        }
    }

    fn item(path: &str, id: &str, excerpt: &str) -> SemanticContextItem {
        SemanticContextItem {
            entity: entity(path, id),
            relevance_bps: 9000,
            excerpt: excerpt.to_string(),
        }
    }

    fn hash(byte: u8) -> FileHash {
        FileHash::from([byte; 32])
    }

    fn proposal(edit_path: &str, expected: Option<FileHash>) -> ProposedPatch {
        let file_edits = match WorkspacePath::parse(edit_path) {
            Ok(path) => vec![SemanticFileEdit {
                path,
                replacement: "fn main() {}".to_string(),
            }],
            Err(_) => Vec::new(),
        };
        let mut expected_source_hashes = BTreeMap::new();
        if let Some(hash) = expected {
            expected_source_hashes.insert(edit_path.to_string(), hash);
        }
        let workspace = WorkspaceId::new(1);
        ProposedPatch {
            workspace,
            snapshot_id: snapshot(workspace, "rev-1"),
            file_edits,
            expected_source_hashes,
            semantic_delta: SemanticDelta {
                workspace,
                from_snapshot: snapshot(workspace, "rev-1"),
                to_snapshot: snapshot(workspace, "rev-2"),
                changes: Vec::new(),
                degraded: true,
            },
        }
    }

    #[test]
    fn context_without_seams_is_degraded_empty_but_ok() {
        let fallback = GenericSemanticFallback::default();
        let envelope = block_on(fallback.context(context_request())).unwrap();
        assert!(envelope.payload.degraded);
        assert!(envelope.payload.items.is_empty());
        assert!(!envelope.payload.truncated);
        // Provenance is provider DATA, never instruction authority.
        assert!(envelope.assert_data_only().is_ok());
    }

    #[test]
    fn index_seam_feeds_bounded_context() {
        let index = Arc::new(TestIndex {
            revision: "rev-1".to_string(),
            items: vec![
                item("src/a.rs", "a", "aaaa"),
                item("src/b.rs", "b", "bbbb"),
                item("src/c.rs", "c", "cccc"),
            ],
        });
        let fallback = GenericSemanticFallback::default().with_index(index);
        let mut request = context_request();
        request.max_items = 2;
        let envelope = block_on(fallback.context(request)).unwrap();
        assert!(!envelope.payload.degraded);
        assert_eq!(envelope.payload.items.len(), 2);
        assert!(envelope.payload.truncated);

        // Byte bound truncates before the item that would exceed it.
        let index = Arc::new(TestIndex {
            revision: "rev-1".to_string(),
            items: vec![item("src/a.rs", "a", "aaaa"), item("src/b.rs", "b", "bbbb")],
        });
        let fallback = GenericSemanticFallback::default().with_index(index);
        let mut request = context_request();
        request.max_bytes = 6;
        let envelope = block_on(fallback.context(request)).unwrap();
        assert_eq!(envelope.payload.items.len(), 1);
        assert!(envelope.payload.truncated);

        // Snapshot picks up the index revision and entity count.
        let snapshot_env = block_on(fallback.snapshot(SemanticSnapshotRequest {
            call: call(),
            workspace: WorkspaceId::new(1),
            source_revision: "ignored".to_string(),
        }))
        .unwrap();
        assert_eq!(snapshot_env.payload.source_revision, "rev-1");
        assert_eq!(snapshot_env.payload.entity_count, 2);
    }

    #[test]
    fn verification_without_metadata_reports_gap_not_pass() {
        let fallback = GenericSemanticFallback::default();
        let envelope = block_on(fallback.verify(SemanticVerifyRequest {
            call: call(),
            workspace: WorkspaceId::new(1),
            snapshot_id: snapshot(WorkspaceId::new(1), "rev-1"),
            entity: entity("src/lib.rs", "lib"),
            claim: "compiles".to_string(),
        }))
        .unwrap();
        assert!(!envelope.payload.passed);
        assert!(envelope.payload.degraded);
        assert!(envelope.payload.checks.is_empty());
    }

    #[test]
    fn proposal_validation_rejects_traversal_and_stale_hashes() {
        let caps = SemanticResponseCaps::default();
        let workspace = WorkspaceId::new(1);
        let snapshot_id = snapshot(workspace, "rev-1");

        // Path traversal is refused at the type boundary.
        match WorkspacePath::parse("../../outside") {
            Err(SemanticError::PathTraversal(_)) => {}
            other => panic!("expected PathTraversal, got {other:?}"),
        }
        // And again when a hash-map key carries the traversal string.
        let mut hostile = proposal("src/lib.rs", Some(hash(1)));
        hostile
            .expected_source_hashes
            .insert("../../outside".to_string(), hash(1));
        let view = MapHashes(BTreeMap::from([("src/lib.rs".to_string(), hash(1))]));
        match hostile.validate(workspace, snapshot_id, &view, &caps) {
            Err(SemanticError::PathTraversal(_)) => {}
            other => panic!("expected PathTraversal from hash key, got {other:?}"),
        }

        // Stale hash: view holds different content.
        let stale = proposal("src/lib.rs", Some(hash(1)));
        let view = MapHashes(BTreeMap::from([("src/lib.rs".to_string(), hash(2))]));
        match stale.validate(workspace, snapshot_id, &view, &caps) {
            Err(SemanticError::StaleHash {
                path,
                actual: Some(actual),
                ..
            }) => {
                assert_eq!(path, "src/lib.rs");
                assert_eq!(actual, hash(2));
            }
            other => panic!("expected StaleHash, got {other:?}"),
        }

        // Missing file: also stale, with actual None.
        let missing = proposal("src/lib.rs", Some(hash(1)));
        let view = MapHashes(BTreeMap::new());
        match missing.validate(workspace, snapshot_id, &view, &caps) {
            Err(SemanticError::StaleHash { actual: None, .. }) => {}
            other => panic!("expected StaleHash(missing), got {other:?}"),
        }

        // Missing expected hash for an edit is malformed, never applied.
        let no_hash = proposal("src/lib.rs", None);
        let view = MapHashes(BTreeMap::from([("src/lib.rs".to_string(), hash(1))]));
        assert!(matches!(
            no_hash.validate(workspace, snapshot_id, &view, &caps),
            Err(SemanticError::Malformed(_))
        ));
    }

    #[test]
    fn proposal_validation_accepts_fresh_hash_and_rejects_bad_identity() {
        let caps = SemanticResponseCaps::default();
        let workspace = WorkspaceId::new(1);
        let snapshot_id = snapshot(workspace, "rev-1");
        let patch = proposal("src/lib.rs", Some(hash(7)));
        let view = MapHashes(BTreeMap::from([("src/lib.rs".to_string(), hash(7))]));
        assert!(patch.validate(workspace, snapshot_id, &view, &caps).is_ok());

        // Wrong workspace / stale snapshot are typed and precede hash checks.
        let view = MapHashes(BTreeMap::from([("src/lib.rs".to_string(), hash(7))]));
        assert!(matches!(
            patch.validate(WorkspaceId::new(2), snapshot_id, &view, &caps),
            Err(SemanticError::WorkspaceMismatch { .. })
        ));
        assert!(matches!(
            patch.validate(workspace, snapshot(workspace, "rev-old"), &view, &caps),
            Err(SemanticError::SnapshotMismatch { .. })
        ));

        // Delta refs outside the workspace are refused.
        let mut outside = patch.clone();
        outside.semantic_delta.changes.push(SemanticDeltaChange {
            entity: SemanticEntityRef::new(
                WorkspaceId::new(9),
                WorkspacePath::parse("src/lib.rs").unwrap(),
                SemanticEntityId::parse("ghost").unwrap(),
            ),
            kind: SemanticDeltaKind::Modified,
            old_hash: None,
            new_hash: None,
        });
        assert!(matches!(
            outside.validate(workspace, snapshot_id, &view, &caps),
            Err(SemanticError::InvalidEntityRef(_))
        ));

        // Duplicate delta entity ids are refused.
        let mut duplicate = patch;
        duplicate.semantic_delta.changes.push(SemanticDeltaChange {
            entity: entity("src/lib.rs", "dup"),
            kind: SemanticDeltaKind::Added,
            old_hash: None,
            new_hash: None,
        });
        duplicate.semantic_delta.changes.push(SemanticDeltaChange {
            entity: entity("src/other.rs", "dup"),
            kind: SemanticDeltaKind::Removed,
            old_hash: None,
            new_hash: None,
        });
        assert!(matches!(
            duplicate.validate(workspace, snapshot_id, &view, &caps),
            Err(SemanticError::DuplicateEntity(_))
        ));
    }

    #[test]
    fn proposal_validation_rejects_oversize_payload() {
        let workspace = WorkspaceId::new(1);
        let snapshot_id = snapshot(workspace, "rev-1");
        let mut patch = proposal("src/lib.rs", Some(hash(1)));
        patch.file_edits[0].replacement = "x".repeat(4096);
        let view = MapHashes(BTreeMap::from([("src/lib.rs".to_string(), hash(1))]));
        let caps = SemanticResponseCaps::new(1, 64, 16);
        assert!(matches!(
            patch.validate(workspace, snapshot_id, &view, &caps),
            Err(SemanticError::Oversized { max: 64, .. })
        ));
    }

    #[test]
    fn edit_request_carries_intent_and_identity() {
        let request = SemanticEditRequest {
            call: call(),
            workspace: WorkspaceId::new(1),
            snapshot_id: snapshot(WorkspaceId::new(1), "rev-1"),
            entity: entity("src/lib.rs", "lib"),
            intent: "rename".to_string(),
        };
        assert_eq!(request.intent, "rename");
        assert_eq!(request.workspace, WorkspaceId::new(1));
        assert_eq!(request.entity.entity_id.as_str(), "lib");
    }
}
