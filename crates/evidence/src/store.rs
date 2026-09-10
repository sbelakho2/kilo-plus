//! In-memory evidence storage with scoped reads and bounded backing retention.
//!
//! Two invariants are structural in this module:
//!
//! 1. **Knowing an id is not authorization.** [`EvidenceStore::get_scoped`]
//!    compares the caller's session/workspace (and task, when both sides name
//!    one) against the envelope and returns [`EvidenceError::AccessDenied`] on
//!    any mismatch — even when the caller presents a valid [`EvidenceId`].
//! 2. **Backing is optional and bounded.** Bytes are retained only for
//!    reversible/aggressive evidence whose capture is complete and whose
//!    length fits the store's `backing_cap`; otherwise the envelope (including
//!    its `backing_hash`) is kept and only the bytes are dropped. Reading
//!    dropped backing fails loudly, never by substitution.

use std::collections::HashMap;

use crate::types::{
    BackingCompleteness, Compressibility, EvidenceEnvelope, EvidenceError, EvidenceId,
};

/// The identity a reader presents. A read is permitted only when session and
/// workspace match the stored envelope; task ids are compared only when both
/// the envelope and the context name one. A task-less envelope stays visible to
/// task-scoped readers, and a task-less reader sees every task in its
/// session/workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvidenceAccessContext {
    pub session_id: u64,
    pub workspace_id: u64,
    pub task_id: Option<u64>,
}

impl EvidenceAccessContext {
    pub const fn new(session_id: u64, workspace_id: u64, task_id: Option<u64>) -> Self {
        Self {
            session_id,
            workspace_id,
            task_id,
        }
    }
}

/// One stored envelope plus the backing bytes the store chose to retain.
/// `backing == None` means the bytes were never provided or were dropped by
/// the store policy; the envelope and its `backing_hash` remain authoritative.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredEvidence {
    pub envelope: EvidenceEnvelope,
    pub backing: Option<Vec<u8>>,
}

impl StoredEvidence {
    /// Length of the retained backing, or `None` when no backing is retained.
    pub fn backing_len(&self) -> Option<usize> {
        self.backing.as_ref().map(Vec::len)
    }
}

/// Evidence storage contract. `insert` never overwrites: re-inserting a live
/// id is refused, so a replay can never silently replace evidence bytes.
pub trait EvidenceStore {
    fn insert(
        &mut self,
        env: EvidenceEnvelope,
        backing: Option<Vec<u8>>,
    ) -> Result<(), EvidenceError>;

    /// Scope-checked read. The default implementation performs the scope
    /// comparison on top of [`EvidenceStore::get`] so every store enforces the
    /// same rule.
    fn get_scoped(
        &self,
        id: EvidenceId,
        ctx: &EvidenceAccessContext,
    ) -> Result<&StoredEvidence, EvidenceError> {
        let stored = self
            .get(id)
            .ok_or_else(|| EvidenceError::Malformed(format!("unknown evidence id {id}")))?;
        ensure_scope(&stored.envelope, ctx)?;
        Ok(stored)
    }

    /// Raw, unscoped lookup used by storage internals. Callers acting for a
    /// session MUST use [`EvidenceStore::get_scoped`].
    fn get(&self, id: EvidenceId) -> Option<&StoredEvidence>;
}

fn ensure_scope(env: &EvidenceEnvelope, ctx: &EvidenceAccessContext) -> Result<(), EvidenceError> {
    let session_ok = env.session_id.raw() == ctx.session_id;
    let workspace_ok = env.workspace_id.raw() == ctx.workspace_id;
    let task_ok = match (env.task_id, ctx.task_id) {
        (Some(env_task), Some(ctx_task)) => env_task == ctx_task,
        _ => true,
    };
    if session_ok && workspace_ok && task_ok {
        return Ok(());
    }
    Err(EvidenceError::AccessDenied(format!(
        "evidence {} belongs to session {} workspace {}; caller is session {} workspace {}",
        env.id, env.session_id, env.workspace_id, ctx.session_id, ctx.workspace_id,
    )))
}

/// In-memory [`EvidenceStore`] with a hard per-envelope backing cap.
#[derive(Debug)]
pub struct MemoryEvidenceStore {
    entries: HashMap<EvidenceId, StoredEvidence>,
    backing_cap: usize,
}

impl MemoryEvidenceStore {
    /// Create a store that retains backing bytes only up to `backing_cap` per
    /// envelope. A cap of `0` keeps every envelope and drops all non-empty
    /// backing.
    pub fn new(backing_cap: usize) -> Self {
        Self {
            entries: HashMap::new(),
            backing_cap,
        }
    }

    pub const fn backing_cap(&self) -> usize {
        self.backing_cap
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn contains(&self, id: EvidenceId) -> bool {
        self.entries.contains_key(&id)
    }

    /// Backing is retained only when it can actually be re-expanded: the
    /// envelope claims `Reversible`/`Aggressive`, the capture is `Complete`,
    /// and the bytes fit the cap. Every other case drops the bytes (never the
    /// envelope, never the recorded `backing_hash`).
    fn retain_backing(
        env: &EvidenceEnvelope,
        backing: Option<Vec<u8>>,
        cap: usize,
    ) -> Option<Vec<u8>> {
        let bytes = backing?;
        let compactable = matches!(
            env.compressibility,
            Compressibility::Reversible | Compressibility::Aggressive
        );
        if compactable
            && env.backing_completeness == BackingCompleteness::Complete
            && bytes.len() <= cap
        {
            Some(bytes)
        } else {
            None
        }
    }
}

impl EvidenceStore for MemoryEvidenceStore {
    fn insert(
        &mut self,
        env: EvidenceEnvelope,
        backing: Option<Vec<u8>>,
    ) -> Result<(), EvidenceError> {
        if self.entries.contains_key(&env.id) {
            return Err(EvidenceError::Refused(format!(
                "evidence {} already exists; refusing to overwrite stored evidence",
                env.id
            )));
        }
        let backing = Self::retain_backing(&env, backing, self.backing_cap);
        self.entries.insert(
            env.id,
            StoredEvidence {
                envelope: env,
                backing,
            },
        );
        Ok(())
    }

    fn get(&self, id: EvidenceId) -> Option<&StoredEvidence> {
        self.entries.get(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        CompactRepresentation, CompressionRecord, EvidenceKind, ProvenanceSet, ProvenanceSource,
        RetrievalPolicy,
    };
    use faktor_core::{SessionId, WorkspaceId};

    fn envelope(
        id: u64,
        session: u64,
        workspace: u64,
        task: Option<u64>,
        compressibility: Compressibility,
        completeness: BackingCompleteness,
    ) -> EvidenceEnvelope {
        let backing_hash = matches!(
            compressibility,
            Compressibility::Reversible | Compressibility::Aggressive
        )
        .then_some([3u8; 32]);
        EvidenceEnvelope::new(
            EvidenceId(id),
            EvidenceKind::ProcessLog,
            SessionId::new(session),
            WorkspaceId::new(workspace),
            task,
            None,
            ProvenanceSet::new([ProvenanceSource::Tool]),
            compressibility,
            CompactRepresentation {
                grammar: "log-v1".to_string(),
                body: "bounded compact body".to_string(),
            },
            backing_hash,
            completeness,
            CompressionRecord::identity(17),
            RetrievalPolicy::new(true, true, 64),
        )
        .unwrap()
    }

    #[test]
    fn scope_is_checked_even_when_the_id_is_known() {
        let mut store = MemoryEvidenceStore::new(1024);
        store
            .insert(
                envelope(
                    7,
                    1,
                    2,
                    Some(3),
                    Compressibility::Aggressive,
                    BackingCompleteness::Complete,
                ),
                Some(vec![1, 2, 3]),
            )
            .unwrap();

        let owner = EvidenceAccessContext::new(1, 2, Some(3));
        assert_eq!(
            store
                .get_scoped(EvidenceId(7), &owner)
                .unwrap()
                .backing_len(),
            Some(3)
        );

        // Same session, different workspace: denied despite knowing id 7.
        let err = store
            .get_scoped(EvidenceId(7), &EvidenceAccessContext::new(1, 9, Some(3)))
            .unwrap_err();
        assert!(matches!(err, EvidenceError::AccessDenied(_)), "{err:?}");

        // Same session+workspace, different task: denied.
        let err = store
            .get_scoped(EvidenceId(7), &EvidenceAccessContext::new(1, 2, Some(4)))
            .unwrap_err();
        assert!(matches!(err, EvidenceError::AccessDenied(_)), "{err:?}");

        // Different session: denied.
        let err = store
            .get_scoped(EvidenceId(7), &EvidenceAccessContext::new(8, 2, Some(3)))
            .unwrap_err();
        assert!(matches!(err, EvidenceError::AccessDenied(_)), "{err:?}");

        // Task is only compared when both sides name one: a task-less reader
        // sees the tasked envelope.
        let taskless = EvidenceAccessContext::new(1, 2, None);
        assert!(store.get_scoped(EvidenceId(7), &taskless).is_ok());

        // A task-scoped reader sees task-less evidence.
        store
            .insert(
                envelope(
                    8,
                    1,
                    2,
                    None,
                    Compressibility::Aggressive,
                    BackingCompleteness::Complete,
                ),
                None,
            )
            .unwrap();
        assert!(store.get_scoped(EvidenceId(8), &owner).is_ok());

        // `get` is raw and unscoped; unknown ids are None, not an error.
        assert!(store.get(EvidenceId(7)).is_some());
        assert!(store.get(EvidenceId(999)).is_none());
        assert_eq!(store.len(), 2);
        assert!(!store.is_empty());
        assert!(store.contains(EvidenceId(8)));
    }

    #[test]
    fn unknown_scoped_id_is_malformed_not_a_denial() {
        let store = MemoryEvidenceStore::new(8);
        let err = store
            .get_scoped(EvidenceId(1), &EvidenceAccessContext::new(1, 2, None))
            .unwrap_err();
        assert!(matches!(err, EvidenceError::Malformed(_)), "{err:?}");
        assert!(err.to_string().contains('1'), "{err}");
    }

    #[test]
    fn backing_cap_keeps_envelope_and_drops_oversized_bytes() {
        let mut store = MemoryEvidenceStore::new(4);

        // Exactly at the cap is retained.
        store
            .insert(
                envelope(
                    1,
                    1,
                    2,
                    None,
                    Compressibility::Aggressive,
                    BackingCompleteness::Complete,
                ),
                Some(vec![0u8; 4]),
            )
            .unwrap();
        assert_eq!(
            store.get(EvidenceId(1)).unwrap().backing.as_deref(),
            Some(&[0u8; 4][..])
        );

        // Over the cap: envelope kept, bytes dropped, hash still recorded.
        let env = envelope(
            2,
            1,
            2,
            None,
            Compressibility::Reversible,
            BackingCompleteness::Complete,
        );
        assert!(env.backing_hash.is_some());
        store.insert(env, Some(vec![0u8; 5])).unwrap();
        let stored = store.get(EvidenceId(2)).unwrap();
        assert!(
            stored.backing.is_none(),
            "oversized backing must be dropped"
        );
        assert!(stored.envelope.backing_hash.is_some(), "hash must survive");
        assert_eq!(stored.envelope.id, EvidenceId(2));

        // Never/LosslessOnly evidence never retains bytes.
        store
            .insert(
                envelope(
                    3,
                    1,
                    2,
                    None,
                    Compressibility::LosslessOnly,
                    BackingCompleteness::Complete,
                ),
                Some(vec![1]),
            )
            .unwrap();
        assert!(store.get(EvidenceId(3)).unwrap().backing.is_none());
        store
            .insert(
                envelope(
                    4,
                    1,
                    2,
                    None,
                    Compressibility::Never,
                    BackingCompleteness::Complete,
                ),
                Some(vec![1]),
            )
            .unwrap();
        assert!(store.get(EvidenceId(4)).unwrap().backing.is_none());

        // Truncated capture never retains bytes even when aggressive + small.
        store
            .insert(
                envelope(
                    5,
                    1,
                    2,
                    None,
                    Compressibility::Aggressive,
                    BackingCompleteness::Truncated,
                ),
                Some(vec![1]),
            )
            .unwrap();
        assert!(store.get(EvidenceId(5)).unwrap().backing.is_none());

        // No backing provided stays None.
        store
            .insert(
                envelope(
                    6,
                    1,
                    2,
                    None,
                    Compressibility::Aggressive,
                    BackingCompleteness::Complete,
                ),
                None,
            )
            .unwrap();
        assert!(store.get(EvidenceId(6)).unwrap().backing.is_none());

        // Cap zero retains only the empty backing (0 bytes fit by definition).
        let mut zero = MemoryEvidenceStore::new(0);
        zero.insert(
            envelope(
                1,
                1,
                2,
                None,
                Compressibility::Aggressive,
                BackingCompleteness::Complete,
            ),
            Some(vec![1]),
        )
        .unwrap();
        assert!(zero.get(EvidenceId(1)).unwrap().backing.is_none());
        zero.insert(
            envelope(
                2,
                1,
                2,
                None,
                Compressibility::Aggressive,
                BackingCompleteness::Complete,
            ),
            Some(Vec::new()),
        )
        .unwrap();
        assert_eq!(
            zero.get(EvidenceId(2)).unwrap().backing.as_deref(),
            Some(&[][..])
        );
        assert_eq!(zero.backing_cap(), 0);
    }

    #[test]
    fn duplicate_insert_is_refused_never_a_silent_overwrite() {
        let mut store = MemoryEvidenceStore::new(16);
        store
            .insert(
                envelope(
                    1,
                    1,
                    2,
                    None,
                    Compressibility::Aggressive,
                    BackingCompleteness::Complete,
                ),
                Some(b"first".to_vec()),
            )
            .unwrap();
        let err = store
            .insert(
                envelope(
                    1,
                    1,
                    2,
                    None,
                    Compressibility::Aggressive,
                    BackingCompleteness::Complete,
                ),
                Some(b"second".to_vec()),
            )
            .unwrap_err();
        assert!(matches!(err, EvidenceError::Refused(_)), "{err:?}");
        assert_eq!(
            store.get(EvidenceId(1)).unwrap().backing.as_deref(),
            Some(&b"first"[..]),
            "the original bytes must survive a duplicate insert"
        );
        assert_eq!(store.len(), 1);
    }
}
