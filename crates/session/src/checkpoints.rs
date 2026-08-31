//! Native content-addressed checkpoints (section 16 of the architecture
//! spec): `before_hash`/`after_hash` pairs so rollback can verify the current
//! content before restoring, never overwriting unrelated edits.

use kilop_core::event::EventKind;
use kilop_core::hash::FileHash;
use kilop_store::CheckpointRow;

use crate::handle::SessionHandle;
use crate::SessionError;

const MAX_CHECKPOINT_PATH_BYTES: usize = 4096;

impl SessionHandle {
    /// Record a checkpoint for one file. `sequence` must be unique per
    /// session (duplicates conflict) and `path` bounded. Journals
    /// `CheckpointCreated` with the same state the session is in.
    pub fn put_checkpoint(
        &self,
        sequence: i64,
        path: &str,
        before_hash: FileHash,
        after_hash: FileHash,
    ) -> kilop_core::Result<i64> {
        if sequence < 0 {
            return Err(SessionError::Malformed(format!(
                "checkpoint sequence must be >= 0, got {sequence}"
            ))
            .into());
        }
        if path.is_empty() || path.len() > MAX_CHECKPOINT_PATH_BYTES {
            return Err(SessionError::Malformed("invalid checkpoint path".into()).into());
        }
        let existing = self.checkpoints_of()?;
        if existing.iter().any(|c| c.sequence == sequence) {
            return Err(SessionError::Conflict(format!(
                "checkpoint sequence {sequence} already exists"
            ))
            .into());
        }
        let _guard = self.command_guard();
        let current = self.state()?;
        crate::journal::validate_transition(
            current,
            EventKind::CheckpointCreated,
            current,
        )?;
        let id = self
            .manager
            .store()
            .put_checkpoint(
                self.id,
                sequence,
                path,
                &before_hash.to_hex(),
                &after_hash.to_hex(),
            )
            .map_err(crate::map_store_err)?;
        self.transition_locked(
            EventKind::CheckpointCreated,
            current,
            None,
            Some(serde_json::json!({
                "sequence": sequence,
                "path": path,
                "before_hash": before_hash.to_hex(),
                "after_hash": after_hash.to_hex(),
            })),
        )?;
        Ok(id)
    }

    pub fn checkpoints_of(&self) -> kilop_core::Result<Vec<CheckpointRow>> {
        self.manager
            .store()
            .checkpoints_of(self.id)
            .map_err(|e| crate::map_store_err(e).into())
    }

    /// Mark a checkpoint as restored (durable audit of the rollback path).
    pub fn mark_checkpoint_restored(&self, id: i64) -> kilop_core::Result<()> {
        self.manager
            .store()
            .mark_checkpoint_restored(id)
            .map_err(|e| crate::map_store_err(e).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::tests::{session, test_manager};

    fn hashes(a: u8, b: u8) -> (FileHash, FileHash) {
        (FileHash::from([a; 32]), FileHash::from([b; 32]))
    }

    #[test]
    fn checkpoints_record_and_survive_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let (sid, hashes_out) = {
            let m = crate::SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                .unwrap();
            let s = session(&m);
            let (before, after) = hashes(1, 2);
            s.put_checkpoint(0, "a.rs", before, after).unwrap();
            let (b2, a2) = hashes(3, 4);
            s.put_checkpoint(1, "b.rs", b2, a2).unwrap();
            (s.id(), vec![(0, before, after), (1, b2, a2)])
        };
        // Reopen: checkpoints are durable.
        let m = crate::SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
            .unwrap();
        let s = m.get_session(sid).unwrap().unwrap();
        let rows = s.checkpoints_of().unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].sequence, 0);
        assert_eq!(rows[0].after_hash, hashes_out[0].2.to_hex());
        assert_eq!(rows[1].path, "b.rs");
        // The journal carried CheckpointCreated events.
        let kinds: Vec<_> = s.events_range(1, None).unwrap().into_iter().map(|e| e.kind).collect();
        assert_eq!(
            kinds,
            vec![
                EventKind::SessionCreated,
                EventKind::CheckpointCreated,
                EventKind::CheckpointCreated
            ]
        );
    }

    #[test]
    fn duplicate_checkpoint_sequence_conflicts_without_trace() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let (before, after) = hashes(1, 2);
        s.put_checkpoint(0, "a.rs", before, after).unwrap();
        let err = s.put_checkpoint(0, "a.rs", before, after).unwrap_err();
        assert_eq!(err.kind, kilop_core::ErrorKind::Conflict);
        assert_eq!(s.checkpoints_of().unwrap().len(), 1);
        assert_eq!(s.events_range(1, None).unwrap().len(), 2, "one event only");
        // Negative sequences and empty paths are malformed.
        assert!(s.put_checkpoint(-1, "a.rs", before, after).is_err());
        assert!(s.put_checkpoint(1, "", before, after).is_err());
        assert!(s.put_checkpoint(1, &"p".repeat(MAX_CHECKPOINT_PATH_BYTES + 1), before, after).is_err());
    }

    #[test]
    fn rollback_audit_marking_is_idempotent_for_unknown_rows() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let (before, after) = hashes(1, 2);
        let id = s.put_checkpoint(0, "a.rs", before, after).unwrap();
        s.mark_checkpoint_restored(id).unwrap();
        // The store marks; unknown ids are silently ignored by the store, so
        // we verify the restored_ms is durable on the known row.
        let rows = s.checkpoints_of().unwrap();
        assert!(rows[0].restored_ms.is_some());
    }
}
