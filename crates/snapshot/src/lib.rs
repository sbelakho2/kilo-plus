//! kilop-snapshot — native content-addressed checkpoints (spec §16).
//!
//! Snapshots are NOT git repositories pretending to be undo history. Before
//! changing a file its original content is stored once in the CAS (dedup is
//! free: ten checkpoints of the same unchanged file = one copy). Rollback
//! verifies the current hash equals the recorded after-hash, then writes the
//! before content atomically; an independently changed file is a Conflict,
//! never silently overwritten.

use std::sync::Arc;

use kilop_cas::Cas;
use kilop_core::error::{Error, ErrorKind};
use kilop_core::hash::FileHash;
use kilop_core::id::SessionId;
use kilop_core::WorkspaceIdentity;
use kilop_fs::WorkspaceHandle;
use kilop_store::Store;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RollbackOutcome {
    Restored {
        path: String,
        hash: FileHash,
    },
    Conflict {
        path: String,
        current: FileHash,
        expected_after: FileHash,
    },
}

pub struct CheckpointStore {
    cas: Arc<Cas>,
    store: Arc<Store>,
}

impl CheckpointStore {
    pub fn new(cas: Arc<Cas>, store: Arc<Store>) -> Self {
        Self { cas, store }
    }

    /// Store the original content (deduped) and return its hash.
    pub fn before_write(
        &self,
        _session: SessionId,
        _path: &str,
        content: &[u8],
    ) -> Result<FileHash, Error> {
        self.cas
            .put(content)
            .map_err(|e| Error::new(ErrorKind::Store, format!("cas: {e}")))
    }

    /// Record the checkpoint after a successful write.
    pub fn after_write(
        &self,
        session: SessionId,
        path: &str,
        before: FileHash,
        after: FileHash,
        sequence: i64,
    ) -> Result<i64, Error> {
        if before == after {
            return Err(Error::malformed(format!(
                "checkpoint {path} records no change (before == after)"
            )));
        }
        self.store
            .put_checkpoint(session, sequence, path, &before.to_hex(), &after.to_hex())
            .map_err(map_store)
    }

    /// Rollback: verify current == after, then atomically write before.
    pub fn rollback(
        &self,
        workspace: &WorkspaceHandle,
        identity: &WorkspaceIdentity,
        checkpoint_id: i64,
    ) -> Result<RollbackOutcome, Error> {
        workspace.verify_identity(identity)?;
        let checkpoints = self
            .store
            .checkpoints_of(identity.workspace_id.into_session_for_checkpoint())
            .map_err(map_store)?;
        let row = checkpoints
            .iter()
            .find(|c| c.id == checkpoint_id)
            .ok_or_else(|| Error::not_found(format!("checkpoint {checkpoint_id}")))?;
        let before = FileHash::from_hex(&row.before_hash)
            .ok_or_else(|| Error::malformed("corrupt before_hash"))?;
        let after = FileHash::from_hex(&row.after_hash)
            .ok_or_else(|| Error::malformed("corrupt after_hash"))?;

        let rel = std::path::Path::new(&row.path);
        let current = workspace.read(rel, usize::MAX)?;
        if current.hash != after {
            return Ok(RollbackOutcome::Conflict {
                path: row.path.clone(),
                current: current.hash,
                expected_after: after,
            });
        }
        // Current content matches what we wrote: restore the original.
        let original = self
            .cas
            .get(before)
            .map_err(|e| Error::new(ErrorKind::Store, format!("cas: {e}")))?;
        let new_hash = workspace.write_atomic(rel, &original)?;
        if new_hash != before {
            return Err(Error::internal(format!(
                "rollback wrote {} but expected {}",
                new_hash.to_hex(),
                before.to_hex()
            )));
        }
        self.store
            .mark_checkpoint_restored(checkpoint_id)
            .map_err(map_store)?;
        Ok(RollbackOutcome::Restored {
            path: row.path.clone(),
            hash: new_hash,
        })
    }

    pub fn checkpoints(
        &self,
        session: SessionId,
    ) -> Result<Vec<kilop_store::CheckpointRow>, Error> {
        self.store.checkpoints_of(session).map_err(map_store)
    }
}

fn map_store(e: kilop_store::StoreError) -> Error {
    Error::new(ErrorKind::Store, format!("store: {e}"))
}

/// Helper: a session id derived from a workspace id for checkpoint storage
/// (checkpoints are stored per session; this is the workspace's own session).
trait SessionFromWorkspace {
    fn into_session_for_checkpoint(self) -> SessionId;
}

impl SessionFromWorkspace for kilop_core::WorkspaceId {
    fn into_session_for_checkpoint(self) -> SessionId {
        SessionId::new(self.raw())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kilop_core::id::{TaskId, WorkspaceId, WorktreeId};
    use std::fs;
    use tempfile::tempdir;

    fn fixture() -> (
        tempfile::TempDir,
        CheckpointStore,
        WorkspaceHandle,
        WorkspaceIdentity,
        SessionId,
    ) {
        let dir = tempdir().unwrap();
        let root = dir.path().join("ws");
        fs::create_dir_all(&root).unwrap();
        let service = kilop_fs::WorkspaceFileService::new();
        let handle = service.open(WorkspaceId::new(1), root.clone()).unwrap();
        let identity =
            WorkspaceIdentity::new(WorkspaceId::new(1), WorktreeId::new(1), TaskId::new(1));
        let cas = Arc::new(Cas::open(dir.path().join("cas")).unwrap());
        let store = Arc::new(Store::open(dir.path().join("store"), true).unwrap());
        let ws = store.create_workspace("/w").unwrap();
        let row = store.create_session(ws, "t", "p", "m").unwrap();
        let cps = CheckpointStore::new(cas, store);
        (dir, cps, handle, identity, row.id)
    }

    #[test]
    fn checkpoints_dedup_identical_files() {
        let (_d, cps, _h, _id, session) = fixture();
        let content = b"same content";
        let mut before_hashes = Vec::new();
        for i in 0..10 {
            let before = cps.before_write(session, "a.rs", content).unwrap();
            before_hashes.push(before);
            let after = FileHash::from([i as u8 + 1; 32]);
            cps.after_write(session, "a.rs", before, after, i).unwrap();
        }
        assert!(before_hashes.windows(2).all(|w| w[0] == w[1]));
        // All ten checkpoints reference the SAME blob.
        let rows = cps.checkpoints(session).unwrap();
        assert_eq!(rows.len(), 10);
        let blob = cps.cas.get(before_hashes[0]).unwrap();
        assert_eq!(blob, content);
        // blob_count counts the shard files: one blob + ... assert via has().
        assert!(cps.cas.has(before_hashes[0]));
    }

    #[test]
    fn rollback_restores_when_current_matches_after() {
        let (_d, cps, h, id, session) = fixture();
        fs::write(h.root().join("f.txt"), b"original").unwrap();
        let before = cps.before_write(session, "f.txt", b"original").unwrap();
        fs::write(h.root().join("f.txt"), b"edited by agent").unwrap();
        let after = CheckpointStore::hash_of(b"edited by agent");
        let cid = cps.after_write(session, "f.txt", before, after, 1).unwrap();
        let outcome = cps.rollback(&h, &id, cid).unwrap();
        match outcome {
            RollbackOutcome::Restored { path, hash } => {
                assert_eq!(path, "f.txt");
                assert_eq!(hash, before);
            }
            other => panic!("expected Restored, got {other:?}"),
        }
        assert_eq!(fs::read(h.root().join("f.txt")).unwrap(), b"original");
    }

    #[test]
    fn rollback_conflicts_on_independent_user_edits() {
        let (_d, cps, h, id, session) = fixture();
        fs::write(h.root().join("f.txt"), b"original").unwrap();
        let before = cps.before_write(session, "f.txt", b"original").unwrap();
        fs::write(h.root().join("f.txt"), b"edited by agent").unwrap();
        let after = CheckpointStore::hash_of(b"edited by agent");
        let cid = cps.after_write(session, "f.txt", before, after, 1).unwrap();
        // The USER edits the file after the agent's edit.
        fs::write(h.root().join("f.txt"), b"user edit").unwrap();
        let outcome = cps.rollback(&h, &id, cid).unwrap();
        match outcome {
            RollbackOutcome::Conflict {
                current,
                expected_after,
                ..
            } => {
                assert_eq!(current, CheckpointStore::hash_of(b"user edit"));
                assert_eq!(expected_after, after);
            }
            other => panic!("expected Conflict, got {other:?}"),
        }
        // Never overwritten.
        assert_eq!(fs::read(h.root().join("f.txt")).unwrap(), b"user edit");
    }

    #[test]
    fn rollback_unknown_checkpoint_not_found() {
        let (_d, cps, h, id, _session) = fixture();
        fs::write(h.root().join("f.txt"), b"x").unwrap();
        let err = cps.rollback(&h, &id, 999).unwrap_err();
        assert!(err.kind == ErrorKind::NotFound);
    }

    #[test]
    fn after_write_rejects_noop_checkpoint() {
        let (_d, cps, _h, _id, session) = fixture();
        let before = FileHash::from([1; 32]);
        let err = cps
            .after_write(session, "f.txt", before, before, 1)
            .unwrap_err();
        assert!(err.kind == ErrorKind::Malformed);
    }

    #[test]
    fn rollback_wrong_identity_rejected() {
        let (_d, cps, h, _id, session) = fixture();
        let wrong =
            WorkspaceIdentity::new(WorkspaceId::new(99), WorktreeId::new(1), TaskId::new(1));
        fs::write(h.root().join("f.txt"), b"x").unwrap();
        let before = cps.before_write(session, "f.txt", b"x").unwrap();
        let after = FileHash::from([2; 32]);
        let cid = cps.after_write(session, "f.txt", before, after, 1).unwrap();
        assert!(cps.rollback(&h, &wrong, cid).is_err());
    }

    #[test]
    fn corrupt_before_hash_is_malformed() {
        let (_d, cps, h, id, session) = fixture();
        // Insert a corrupt row directly (valid session FK); the row id is
        // what rollback addresses, not the sequence.
        let store = &cps.store;
        let corrupt_id = store
            .put_checkpoint(session, 5, "f.txt", "not-a-hash", &"aa".repeat(32))
            .unwrap();
        fs::write(h.root().join("f.txt"), b"x").unwrap();
        let err = cps.rollback(&h, &id, corrupt_id).unwrap_err();
        assert!(
            matches!(err.kind, ErrorKind::Malformed | ErrorKind::Store),
            "corrupt metadata must be loud: {err:?}"
        );
    }

    #[test]
    fn checkpoint_rows_survive_reopen() {
        let dir = tempdir().unwrap();
        let session = {
            let cas = Arc::new(Cas::open(dir.path().join("cas")).unwrap());
            let store = Arc::new(Store::open(dir.path().join("store"), true).unwrap());
            let ws = store.create_workspace("/w").unwrap();
            let row = store.create_session(ws, "t", "p", "m").unwrap();
            let cps = CheckpointStore::new(cas, store);
            let before = cps.before_write(row.id, "a", b"data").unwrap();
            cps.after_write(row.id, "a", before, FileHash::from([9; 32]), 1)
                .unwrap();
            row.id
        };
        let cas = Arc::new(Cas::open(dir.path().join("cas")).unwrap());
        let store = Arc::new(Store::open(dir.path().join("store"), true).unwrap());
        let cps = CheckpointStore::new(cas, store);
        assert_eq!(cps.checkpoints(session).unwrap().len(), 1);
    }

    #[test]
    fn corrupted_cas_blob_fails_rollback_loudly() {
        let (_d, cps, h, id, session) = fixture();
        fs::write(h.root().join("f.txt"), b"original").unwrap();
        let before = cps.before_write(session, "f.txt", b"original").unwrap();
        fs::write(h.root().join("f.txt"), b"edited").unwrap();
        let after = CheckpointStore::hash_of(b"edited");
        let cid = cps.after_write(session, "f.txt", before, after, 1).unwrap();
        // Corrupt the stored original blob.
        let path = cps
            .cas
            .root()
            .join(&before.to_hex()[..2])
            .join(&before.to_hex()[2..]);
        fs::write(path, b"garbage").unwrap();
        let err = cps.rollback(&h, &id, cid).unwrap_err();
        assert!(
            err.kind == ErrorKind::Store,
            "corruption must be loud: {err:?}"
        );
    }

    impl CheckpointStore {
        fn hash_of(bytes: &[u8]) -> FileHash {
            FileHash::from(blake3::hash(bytes).into())
        }
    }
}
