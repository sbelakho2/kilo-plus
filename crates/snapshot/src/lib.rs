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

/// One side of a line diff: a context line (' '), a removed line ('-') or an
/// added line ('+'). Deterministic, LCS-free, bounded by [`DIFF_MAX_LINES`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffLine {
    Context(String),
    Removed(String),
    Added(String),
}

impl DiffLine {
    /// The unified-diff rendering: `' '`/`'-'`/`'+'` prefix + line.
    pub fn render(&self) -> String {
        match self {
            DiffLine::Context(l) => format!(" {l}"),
            DiffLine::Removed(l) => format!("-{l}"),
            DiffLine::Added(l) => format!("+{l}"),
        }
    }
}

/// The result of diffing the latest checkpoint: the file path and the
/// unified-diff text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffResult {
    pub path: String,
    pub diff: String,
}

/// Context lines shown around a change.
const DIFF_CONTEXT_LINES: usize = 3;
/// Hard bound on a diff: never stream more than this many lines to the wire.
pub const DIFF_MAX_LINES: usize = 2000;

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

    /// Record the checkpoint after a successful write. The after-content
    /// itself is stored in the CAS (deduped) so unrevert (redo) and diff can
    /// reconstruct exactly what the edit wrote; a content/hash mismatch is
    /// loud, never silently recorded.
    pub fn after_write(
        &self,
        session: SessionId,
        path: &str,
        before: FileHash,
        after: FileHash,
        sequence: i64,
        after_content: &[u8],
    ) -> Result<i64, Error> {
        if before == after {
            return Err(Error::malformed(format!(
                "checkpoint {path} records no change (before == after)"
            )));
        }
        let after_cas = self
            .cas
            .put(after_content)
            .map_err(|e| Error::new(ErrorKind::Store, format!("cas: {e}")))?;
        if after_cas != after {
            return Err(Error::internal(format!(
                "after_write recorded {} but content hashes to {}",
                after.to_hex(),
                after_cas.to_hex()
            )));
        }
        self.store
            .put_checkpoint(
                session,
                sequence,
                path,
                &before.to_hex(),
                &after.to_hex(),
                Some(&after_cas.to_hex()),
            )
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
        let row = self.checkpoint_row(identity, checkpoint_id)?;
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

    /// Redo (unrevert): verify current == before, then atomically write the
    /// checkpoint's after content back. Checkpoints recorded before v3 (no
    /// after blob in the CAS) are refused honestly with a Conflict.
    pub fn redo(
        &self,
        workspace: &WorkspaceHandle,
        identity: &WorkspaceIdentity,
        checkpoint_id: i64,
    ) -> Result<RollbackOutcome, Error> {
        workspace.verify_identity(identity)?;
        let row = self.checkpoint_row(identity, checkpoint_id)?;
        let before = FileHash::from_hex(&row.before_hash)
            .ok_or_else(|| Error::malformed("corrupt before_hash"))?;
        let after = FileHash::from_hex(&row.after_hash)
            .ok_or_else(|| Error::malformed("corrupt after_hash"))?;

        let rel = std::path::Path::new(&row.path);
        let current = workspace.read(rel, usize::MAX)?;
        if current.hash != before {
            return Ok(RollbackOutcome::Conflict {
                path: row.path.clone(),
                current: current.hash,
                expected_after: before,
            });
        }
        let Some(after_cas_raw) = row.after_cas_hash.as_deref() else {
            return Err(Error::new(
                ErrorKind::Conflict,
                format!(
                    "after-content unavailable for checkpoint {} (recorded before after-blob storage)",
                    row.id
                ),
            ));
        };
        let after_cas = FileHash::from_hex(after_cas_raw)
            .ok_or_else(|| Error::malformed("corrupt after_cas_hash"))?;
        let after_bytes = self
            .cas
            .get(after_cas)
            .map_err(|e| Error::new(ErrorKind::Store, format!("cas: {e}")))?;
        let new_hash = workspace.write_atomic(rel, &after_bytes)?;
        if new_hash != after {
            return Err(Error::internal(format!(
                "redo wrote {} but expected {}",
                new_hash.to_hex(),
                after.to_hex()
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

    /// Unified line diff of the latest checkpoint's before/after contents.
    /// `Ok(None)` when the session has no checkpoints. Pre-v3 checkpoints
    /// (no after blob) are refused honestly with a Conflict.
    pub fn diff_latest(
        &self,
        workspace: &WorkspaceHandle,
        identity: &WorkspaceIdentity,
    ) -> Result<Option<DiffResult>, Error> {
        workspace.verify_identity(identity)?;
        let checkpoints = self
            .store
            .checkpoints_of(identity.workspace_id.into_session_for_checkpoint())
            .map_err(map_store)?;
        let Some(row) = checkpoints.iter().max_by_key(|c| c.sequence) else {
            return Ok(None);
        };
        let before = FileHash::from_hex(&row.before_hash)
            .ok_or_else(|| Error::malformed("corrupt before_hash"))?;
        let Some(after_cas_raw) = row.after_cas_hash.as_deref() else {
            return Err(Error::new(
                ErrorKind::Conflict,
                format!(
                    "after-content unavailable for checkpoint {} (recorded before after-blob storage)",
                    row.id
                ),
            ));
        };
        let after_cas = FileHash::from_hex(after_cas_raw)
            .ok_or_else(|| Error::malformed("corrupt after_cas_hash"))?;
        let before_bytes = self
            .cas
            .get(before)
            .map_err(|e| Error::new(ErrorKind::Store, format!("cas: {e}")))?;
        let after_bytes = self
            .cas
            .get(after_cas)
            .map_err(|e| Error::new(ErrorKind::Store, format!("cas: {e}")))?;
        let diff = diff_lines(&before_bytes, &after_bytes)
            .iter()
            .map(DiffLine::render)
            .collect::<Vec<_>>()
            .join("\n");
        Ok(Some(DiffResult {
            path: row.path.clone(),
            diff,
        }))
    }

    pub fn checkpoints(
        &self,
        session: SessionId,
    ) -> Result<Vec<kilop_store::CheckpointRow>, Error> {
        self.store.checkpoints_of(session).map_err(map_store)
    }

    fn checkpoint_row(
        &self,
        identity: &WorkspaceIdentity,
        checkpoint_id: i64,
    ) -> Result<kilop_store::CheckpointRow, Error> {
        let checkpoints = self
            .store
            .checkpoints_of(identity.workspace_id.into_session_for_checkpoint())
            .map_err(map_store)?;
        checkpoints
            .iter()
            .find(|c| c.id == checkpoint_id)
            .cloned()
            .ok_or_else(|| Error::not_found(format!("checkpoint {checkpoint_id}")))
    }
}

/// A deterministic, LCS-free line diff: common prefix, then a changed middle
/// (all removals then all additions), then common suffix, with 3 lines of
/// context around the change. Bounded to [`DIFF_MAX_LINES`] lines.
pub fn diff_lines(before: &[u8], after: &[u8]) -> Vec<DiffLine> {
    let before_lines = split_lines(before);
    let after_lines = split_lines(after);
    let mut lines = Vec::new();
    if before_lines == after_lines {
        for l in &before_lines {
            lines.push(DiffLine::Context(l.clone()));
        }
        return bound_diff(lines);
    }
    let prefix = before_lines
        .iter()
        .zip(&after_lines)
        .take_while(|(a, b)| a == b)
        .count();
    let mut suffix = 0usize;
    while suffix < before_lines.len() - prefix
        && suffix < after_lines.len() - prefix
        && before_lines[before_lines.len() - 1 - suffix]
            == after_lines[after_lines.len() - 1 - suffix]
    {
        suffix += 1;
    }
    let ctx_start = prefix.saturating_sub(DIFF_CONTEXT_LINES);
    for l in &before_lines[ctx_start..prefix] {
        lines.push(DiffLine::Context(l.clone()));
    }
    for l in &before_lines[prefix..before_lines.len() - suffix] {
        lines.push(DiffLine::Removed(l.clone()));
    }
    for l in &after_lines[prefix..after_lines.len() - suffix] {
        lines.push(DiffLine::Added(l.clone()));
    }
    let ctx_end = (after_lines.len() - suffix + DIFF_CONTEXT_LINES).min(after_lines.len());
    for l in &after_lines[after_lines.len() - suffix..ctx_end] {
        lines.push(DiffLine::Context(l.clone()));
    }
    bound_diff(lines)
}

fn bound_diff(mut lines: Vec<DiffLine>) -> Vec<DiffLine> {
    if lines.len() <= DIFF_MAX_LINES {
        return lines;
    }
    lines.truncate(DIFF_MAX_LINES - 1);
    lines.push(DiffLine::Context(format!(
        "… diff truncated: more than {DIFF_MAX_LINES} lines"
    )));
    lines
}

fn split_lines(bytes: &[u8]) -> Vec<String> {
    let text = String::from_utf8_lossy(bytes);
    let mut lines: Vec<String> = text.split('\n').map(|l| l.to_string()).collect();
    if lines.last().is_some_and(|l| l.is_empty()) {
        lines.pop();
    }
    lines
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
            let after_content = vec![i as u8 + 1; 32];
            let after = CheckpointStore::hash_of(&after_content);
            cps.after_write(session, "a.rs", before, after, i, &after_content)
                .unwrap();
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
    fn after_write_stores_after_blob_and_dedups() {
        let (_d, cps, _h, _id, session) = fixture();
        let before = cps.before_write(session, "f.txt", b"original").unwrap();
        let after_content = b"the exact edited bytes";
        let after = CheckpointStore::hash_of(after_content);
        let c1 = cps
            .after_write(session, "f.txt", before, after, 1, after_content)
            .unwrap();
        // Same after content, different path: the CAS must dedup to one blob.
        let before2 = cps
            .before_write(session, "g.txt", b"other original")
            .unwrap();
        let c2 = cps
            .after_write(session, "g.txt", before2, after, 2, after_content)
            .unwrap();
        let rows = cps.checkpoints(session).unwrap();
        assert_eq!(
            rows[0].after_cas_hash.as_deref(),
            Some(after.to_hex().as_str())
        );
        assert_eq!(
            rows[1].after_cas_hash.as_deref(),
            Some(after.to_hex().as_str())
        );
        // Exactly ONE stored after blob: three distinct contents total (two
        // before blobs + one shared after blob) — the second after_write must
        // have deduped, not written a second copy.
        assert_eq!(cps.cas.blob_count(), 3, "after blob must dedup to one copy");
        assert_eq!(cps.cas.get(after).unwrap(), after_content);
        assert_ne!(c1, c2);
    }

    #[test]
    fn after_write_hash_mismatch_is_loud() {
        let (_d, cps, _h, _id, session) = fixture();
        let before = cps.before_write(session, "f.txt", b"original").unwrap();
        // The caller claims a hash that does not match the content: the
        // checkpoint must be refused, never recorded with a lying hash.
        let wrong_after = FileHash::from([9; 32]);
        let err = cps
            .after_write(session, "f.txt", before, wrong_after, 1, b"different")
            .unwrap_err();
        assert!(
            err.kind == ErrorKind::Internal,
            "hash mismatch must be loud: {err:?}"
        );
        assert!(cps.checkpoints(session).unwrap().is_empty());
    }

    #[test]
    fn rollback_restores_when_current_matches_after() {
        let (_d, cps, h, id, session) = fixture();
        fs::write(h.root().join("f.txt"), b"original").unwrap();
        let before = cps.before_write(session, "f.txt", b"original").unwrap();
        fs::write(h.root().join("f.txt"), b"edited by agent").unwrap();
        let after_content = b"edited by agent";
        let after = CheckpointStore::hash_of(after_content);
        let cid = cps
            .after_write(session, "f.txt", before, after, 1, after_content)
            .unwrap();
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
        let after_content = b"edited by agent";
        let after = CheckpointStore::hash_of(after_content);
        let cid = cps
            .after_write(session, "f.txt", before, after, 1, after_content)
            .unwrap();
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
    fn redo_restores_after_state() {
        let (_d, cps, h, id, session) = fixture();
        fs::write(h.root().join("f.txt"), b"original").unwrap();
        let before = cps.before_write(session, "f.txt", b"original").unwrap();
        fs::write(h.root().join("f.txt"), b"edited by agent").unwrap();
        let after_content = b"edited by agent";
        let after = CheckpointStore::hash_of(after_content);
        let cid = cps
            .after_write(session, "f.txt", before, after, 1, after_content)
            .unwrap();
        // Rollback to the pre-edit state, then redo back to the after state.
        let r = cps.rollback(&h, &id, cid).unwrap();
        assert!(matches!(r, RollbackOutcome::Restored { .. }));
        assert_eq!(fs::read(h.root().join("f.txt")).unwrap(), b"original");
        let outcome = cps.redo(&h, &id, cid).unwrap();
        match outcome {
            RollbackOutcome::Restored { path, hash } => {
                assert_eq!(path, "f.txt");
                assert_eq!(hash, after);
            }
            other => panic!("expected Restored, got {other:?}"),
        }
        assert_eq!(
            fs::read(h.root().join("f.txt")).unwrap(),
            b"edited by agent"
        );
        // The audit trail marks the checkpoint restored.
        assert!(cps.checkpoints(session).unwrap()[0].restored_ms.is_some());
    }

    #[test]
    fn redo_conflicts_on_independent_edit() {
        let (_d, cps, h, id, session) = fixture();
        fs::write(h.root().join("f.txt"), b"original").unwrap();
        let before = cps.before_write(session, "f.txt", b"original").unwrap();
        fs::write(h.root().join("f.txt"), b"edited").unwrap();
        let after_content = b"edited";
        let after = CheckpointStore::hash_of(after_content);
        let cid = cps
            .after_write(session, "f.txt", before, after, 1, after_content)
            .unwrap();
        // Rollback happened, then the USER edits again: redo must conflict,
        // never clobber the user's content.
        cps.rollback(&h, &id, cid).unwrap();
        fs::write(h.root().join("f.txt"), b"user took over").unwrap();
        let outcome = cps.redo(&h, &id, cid).unwrap();
        match outcome {
            RollbackOutcome::Conflict {
                current,
                expected_after,
                ..
            } => {
                assert_eq!(current, CheckpointStore::hash_of(b"user took over"));
                assert_eq!(expected_after, before);
            }
            other => panic!("expected Conflict, got {other:?}"),
        }
        assert_eq!(fs::read(h.root().join("f.txt")).unwrap(), b"user took over");
    }

    #[test]
    fn redo_unknown_checkpoint_not_found() {
        let (_d, cps, h, id, _session) = fixture();
        fs::write(h.root().join("f.txt"), b"x").unwrap();
        let err = cps.redo(&h, &id, 999).unwrap_err();
        assert!(err.kind == ErrorKind::NotFound);
    }

    #[test]
    fn redo_pre_v3_row_refused_honestly() {
        let (_d, cps, h, id, session) = fixture();
        // A checkpoint row recorded before after-blob storage: real before
        // content on disk (so the conflict guard passes), but the store row
        // has no after blob — redo cannot reconstruct the after content.
        fs::write(h.root().join("f.txt"), b"original").unwrap();
        let before = cps.before_write(session, "f.txt", b"original").unwrap();
        let row_id = cps
            .store
            .put_checkpoint(
                session,
                1,
                "f.txt",
                &before.to_hex(),
                &"aa".repeat(32),
                None,
            )
            .unwrap();
        let err = cps.redo(&h, &id, row_id).unwrap_err();
        assert!(
            err.kind == ErrorKind::Conflict && err.message.contains("after-content unavailable"),
            "pre-v3 rows must be refused honestly: {err:?}"
        );
        assert_eq!(fs::read(h.root().join("f.txt")).unwrap(), b"original");
    }

    #[test]
    fn after_write_rejects_noop_checkpoint() {
        let (_d, cps, _h, _id, session) = fixture();
        let before = FileHash::from([1; 32]);
        let err = cps
            .after_write(session, "f.txt", before, before, 1, b"x")
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
        let after_content = b"y";
        let after = CheckpointStore::hash_of(after_content);
        let cid = cps
            .after_write(session, "f.txt", before, after, 1, after_content)
            .unwrap();
        assert!(cps.rollback(&h, &wrong, cid).is_err());
    }

    #[test]
    fn corrupt_before_hash_is_malformed() {
        let (_d, cps, h, id, session) = fixture();
        // Insert a corrupt row directly (valid session FK); the row id is
        // what rollback addresses, not the sequence.
        let store = &cps.store;
        let corrupt_id = store
            .put_checkpoint(session, 5, "f.txt", "not-a-hash", &"aa".repeat(32), None)
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
            let after_content = b"after-data";
            cps.after_write(
                row.id,
                "a",
                before,
                CheckpointStore::hash_of(after_content),
                1,
                after_content,
            )
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
        let after_content = b"edited";
        let after = CheckpointStore::hash_of(after_content);
        let cid = cps
            .after_write(session, "f.txt", before, after, 1, after_content)
            .unwrap();
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

    #[test]
    fn diff_latest_produces_lines_with_context() {
        let (_d, cps, h, id, session) = fixture();
        let before_text = "line1\nline2\nline3\nline4\nold\nline6\nline7\nline8\nline9\n";
        let after_text = "line1\nline2\nline3\nline4\nnew\nline6\nline7\nline8\nline9\n";
        fs::write(h.root().join("f.txt"), before_text).unwrap();
        let before = cps
            .before_write(session, "f.txt", before_text.as_bytes())
            .unwrap();
        fs::write(h.root().join("f.txt"), after_text).unwrap();
        let cid = cps
            .after_write(
                session,
                "f.txt",
                before,
                CheckpointStore::hash_of(after_text.as_bytes()),
                1,
                after_text.as_bytes(),
            )
            .unwrap();
        let _ = cid;
        let result = cps.diff_latest(&h, &id).unwrap().unwrap();
        assert_eq!(result.path, "f.txt");
        let lines: Vec<&str> = result.diff.lines().collect();
        // A removal and an addition for the changed middle line.
        assert!(lines.contains(&"-old"), "removal missing: {result:?}");
        assert!(lines.contains(&"+new"), "addition missing: {result:?}");
        // Context lines around the change: exactly 3 before and 3 after.
        let context = lines.iter().filter(|l| l.starts_with(' ')).count();
        assert_eq!(context, 6, "exactly 3+3 context lines: {result:?}");
        // Deterministic boundaries: the change sits at line 5 of 9, so the
        // context window is lines 2-4 and 6-8 — never line1/line9.
        assert!(lines.iter().any(|l| l == &" line2"));
        assert!(lines.iter().any(|l| l == &" line8"));
        assert!(!lines.iter().any(|l| l == &" line1"));
        assert!(!lines.iter().any(|l| l == &" line9"));
    }

    #[test]
    fn diff_latest_none_when_no_checkpoints() {
        let (_d, cps, h, id, _session) = fixture();
        assert!(cps.diff_latest(&h, &id).unwrap().is_none());
    }

    #[test]
    fn diff_latest_bounded_to_2000_lines() {
        let (_d, cps, h, id, session) = fixture();
        let before_text = (0..5000)
            .map(|i| format!("before line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let after_text = (0..5000)
            .map(|i| format!("after line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(h.root().join("big.txt"), &before_text).unwrap();
        let before = cps
            .before_write(session, "big.txt", before_text.as_bytes())
            .unwrap();
        let cid = cps
            .after_write(
                session,
                "big.txt",
                before,
                CheckpointStore::hash_of(after_text.as_bytes()),
                1,
                after_text.as_bytes(),
            )
            .unwrap();
        let _ = cid;
        let result = cps.diff_latest(&h, &id).unwrap().unwrap();
        let lines: Vec<&str> = result.diff.lines().collect();
        assert!(
            lines.len() <= DIFF_MAX_LINES,
            "diff must be bounded: {} lines",
            lines.len()
        );
        assert!(
            lines.last().unwrap().contains("truncated"),
            "truncation marker missing: {}",
            result.diff
        );
    }

    #[test]
    fn diff_latest_pre_v3_row_refused_honestly() {
        let (_d, cps, h, id, session) = fixture();
        cps.store
            .put_checkpoint(
                session,
                1,
                "f.txt",
                &"bb".repeat(32),
                &"aa".repeat(32),
                None,
            )
            .unwrap();
        let err = cps.diff_latest(&h, &id).unwrap_err();
        assert!(
            err.kind == ErrorKind::Conflict && err.message.contains("after-content unavailable"),
            "pre-v3 rows must be refused honestly: {err:?}"
        );
    }

    impl CheckpointStore {
        fn hash_of(bytes: &[u8]) -> FileHash {
            FileHash::from(blake3::hash(bytes).into())
        }
    }
}
