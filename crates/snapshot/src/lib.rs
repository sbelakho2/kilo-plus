//! faktor-snapshot — native content-addressed checkpoints (spec §16).
//!
//! Snapshots are NOT git repositories pretending to be undo history. Before
//! changing a file its original content is stored once in the CAS (dedup is
//! free: ten checkpoints of the same unchanged file = one copy). Rollback
//! verifies the current file state equals the recorded after-state, then
//! restores the before-state atomically; an independently changed file is a
//! Conflict, never silently overwritten.
//!
//! File states carry EXISTENCE, not just hashes: a hash alone cannot
//! distinguish a missing file from an empty one (both sides of a
//! missing→empty write hash to blake3("")), so the old hash-only checkpoints
//! skipped empty-file creation entirely and rolled a missing→content write
//! back to an empty file instead of deleting it.

use std::path::Path;
use std::sync::Arc;

use faktor_cas::Cas;
use faktor_core::error::{Error, ErrorKind};
use faktor_core::hash::FileHash;
use faktor_core::id::SessionId;
use faktor_core::WorkspaceIdentity;
use faktor_fs::WorkspaceHandle;
use faktor_store::Store;

/// The existence-bearing state of one file at checkpoint time. `exists=true`
/// always comes with the content hash; `exists=false` (a missing file) has
/// no content, so no hash. Rollback, unrevert and diff all reason over this
/// state, never over a bare hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileState {
    pub exists: bool,
    pub hash: Option<FileHash>,
}

impl FileState {
    pub const fn missing() -> Self {
        Self {
            exists: false,
            hash: None,
        }
    }

    pub fn existing(hash: FileHash) -> Self {
        Self {
            exists: true,
            hash: Some(hash),
        }
    }

    /// Disk truth at `rel` inside `workspace`: a missing file is a state,
    /// never an error. Resolution goes through the workspace handle
    /// (canonical root, traversal/symlink rejection), so a hostile path
    /// surfaces as a Permission error instead of touching the disk.
    pub fn probe(workspace: &WorkspaceHandle, rel: &Path) -> Result<Self, Error> {
        if !workspace.exists(rel) {
            return Ok(Self::missing());
        }
        let data = workspace.read(rel, usize::MAX)?;
        Ok(Self::existing(data.hash))
    }
}

/// Diff status derived from a checkpoint's before→after state transition:
/// exists false→true is Added, true→false is Deleted, true→true with
/// different content is Modified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeStatus {
    Added,
    Deleted,
    Modified,
}

impl ChangeStatus {
    pub fn from_transition(before: FileState, after: FileState) -> Option<Self> {
        match (before.exists, after.exists) {
            (false, true) => Some(Self::Added),
            (true, false) => Some(Self::Deleted),
            (true, true) if before.hash != after.hash => Some(Self::Modified),
            // Equal states: no transition. record_change refuses such rows;
            // raw rows that slipped in have nothing to report.
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Added => "added",
            Self::Deleted => "deleted",
            Self::Modified => "modified",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RollbackOutcome {
    Restored {
        path: String,
        /// Hash of the restored content; `None` when the restored state is a
        /// deleted file (the file was removed, nothing to hash).
        hash: Option<FileHash>,
    },
    Conflict {
        path: String,
        /// The file's state on disk right now (a missing file is a state).
        current: FileState,
        /// The recorded after-state the current state was expected to equal.
        expected_after: FileState,
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

/// The result of diffing the latest checkpoint: the file path, the
/// unified-diff text and the change status derived from the recorded state
/// transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffResult {
    pub path: String,
    pub diff: String,
    pub status: ChangeStatus,
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

    /// Record an EXISTENCE-BEARING checkpoint (P0). This is the one true row
    /// creator: both sides are [`FileState`]s, so a missing→empty write is a
    /// real transition (before {exists:false}, after {exists:true,
    /// empty-hash}) and IS recorded, and a rollback can distinguish "delete
    /// the file" from "write the empty content".
    ///
    /// Content rules:
    /// - a side with `exists=true` needs its bytes: pass them in
    ///   `before_content`/`after_content` (the blob is CAS-stored and the
    ///   hash verified loudly), or — for the before side only — `None` when
    ///   the bytes were already CAS-stored by an earlier `before_write`.
    /// - a side with `exists=false` must NOT pass bytes.
    ///
    /// The per-session sequence is ALLOCATED by the store in the same
    /// transaction as the insert (P1): two concurrent writers can never both
    /// receive the same sequence. Returns the checkpoint row id.
    pub fn record_change(
        &self,
        session: SessionId,
        path: &str,
        before: FileState,
        before_content: Option<&[u8]>,
        after: FileState,
        after_content: Option<&[u8]>,
    ) -> Result<i64, Error> {
        check_state(before, "before")?;
        check_state(after, "after")?;
        if before == after {
            return Err(Error::malformed(format!(
                "checkpoint {path} records no change (before == after)"
            )));
        }
        if before_content.is_some() && !before.exists {
            return Err(Error::malformed(format!(
                "checkpoint {path}: before content given for a missing file"
            )));
        }
        if after_content.is_none() && after.exists {
            return Err(Error::malformed(format!(
                "checkpoint {path}: after content required when the after state exists"
            )));
        }
        if after_content.is_some() && !after.exists {
            return Err(Error::malformed(format!(
                "checkpoint {path}: after content given for a deleted file"
            )));
        }
        if let Some(bytes) = before_content {
            let actual = self
                .cas
                .put(bytes)
                .map_err(|e| Error::new(ErrorKind::Store, format!("cas: {e}")))?;
            if Some(actual) != before.hash {
                return Err(Error::internal(format!(
                    "checkpoint {path} records before {} but content hashes to {}",
                    before.hash.unwrap_or(actual).to_hex(),
                    actual.to_hex()
                )));
            }
        } else if before.exists {
            // The caller CAS-stored the before blob earlier (before_write);
            // prove it is really there rather than recording a hash that
            // rollback could never fetch.
            if !self.cas.has(before.hash.unwrap_or_default()) {
                return Err(Error::internal(format!(
                    "checkpoint {path}: before content missing from the CAS"
                )));
            }
        }
        let after_cas = match after_content {
            Some(bytes) => {
                let actual = self
                    .cas
                    .put(bytes)
                    .map_err(|e| Error::new(ErrorKind::Store, format!("cas: {e}")))?;
                if Some(actual) != after.hash {
                    return Err(Error::internal(format!(
                        "checkpoint {path} records after {} but content hashes to {}",
                        after.hash.unwrap_or(actual).to_hex(),
                        actual.to_hex()
                    )));
                }
                Some(actual.to_hex())
            }
            None => None,
        };
        let (id, _allocated_sequence) = self
            .store
            .insert_checkpoint(
                session,
                path,
                before.exists,
                &side_hash(before),
                after.exists,
                &side_hash(after),
                after_cas.as_deref(),
            )
            .map_err(map_store)?;
        Ok(id)
    }

    /// Record the checkpoint after a successful write. The after-content
    /// itself is stored in the CAS (deduped) so unrevert (redo) and diff can
    /// reconstruct exactly what the edit wrote; a content/hash mismatch is
    /// loud, never silently recorded.
    ///
    /// Both sides are existing files (the hash-only form cannot express
    /// existence); use [`CheckpointStore::record_change`] with [`FileState`]s
    /// for creation/deletion/truncation semantics.
    ///
    /// NOTE on `sequence`: it is a compatibility vestige and is NEVER
    /// honored. The checkpoint numbering race was exactly callers deriving
    /// `rows.len()+1` outside the store; the sequence is now allocated
    /// atomically by the store in the same transaction as the insert.
    pub fn after_write(
        &self,
        session: SessionId,
        path: &str,
        before: FileHash,
        after: FileHash,
        _sequence: i64,
        after_content: &[u8],
    ) -> Result<i64, Error> {
        self.record_change(
            session,
            path,
            FileState::existing(before),
            None, // before_write already stored the original content
            FileState::existing(after),
            Some(after_content),
        )
    }

    /// Rollback: verify the current file state equals the recorded
    /// after-state, then restore the before-state. "Restore" means deleting
    /// the file when the before-state is missing and writing the CAS content
    /// back when it existed — an empty-but-existing before writes the empty
    /// content, a missing before deletes. Checkpoints are looked up under the
    /// REAL session id (a session whose id differs from its workspace's id
    /// must still roll back). All file access goes through the workspace
    /// handle (canonical root, traversal/symlink rejection), so a hostile
    /// stored path can neither read nor write outside the workspace.
    pub fn rollback(
        &self,
        workspace: &WorkspaceHandle,
        identity: &WorkspaceIdentity,
        session: SessionId,
        checkpoint_id: i64,
    ) -> Result<RollbackOutcome, Error> {
        workspace.verify_identity(identity)?;
        let row = self.checkpoint_row(session, checkpoint_id)?;
        let before = side_state(&row, Side::Before)?;
        let after = side_state(&row, Side::After)?;

        let rel = std::path::Path::new(&row.path);
        let current = FileState::probe(workspace, rel)?;
        if current != after {
            return Ok(RollbackOutcome::Conflict {
                path: row.path.clone(),
                current,
                expected_after: after,
            });
        }
        // Current state matches what we wrote: restore the original state.
        let outcome = match before {
            FileState {
                exists: true,
                hash: Some(before_hash),
            } => {
                let original = self
                    .cas
                    .get(before_hash)
                    .map_err(|e| Error::new(ErrorKind::Store, format!("cas: {e}")))?;
                let new_hash = workspace.write_atomic(rel, &original)?;
                if new_hash != before_hash {
                    return Err(Error::internal(format!(
                        "rollback wrote {} but expected {}",
                        new_hash.to_hex(),
                        before_hash.to_hex()
                    )));
                }
                RollbackOutcome::Restored {
                    path: row.path.clone(),
                    hash: Some(new_hash),
                }
            }
            // The file did not exist before the write: rollback DELETES it.
            FileState { exists: false, .. } => {
                self.delete_through_workspace(workspace, rel)?;
                RollbackOutcome::Restored {
                    path: row.path.clone(),
                    hash: None,
                }
            }
            // Unreachable: record_change/check_state keep states consistent.
            other => return Err(Error::malformed(format!("corrupt before state: {other:?}"))),
        };
        self.store
            .mark_checkpoint_restored(checkpoint_id)
            .map_err(map_store)?;
        Ok(outcome)
    }

    /// Delete `rel` via the workspace handle's canonical resolution. The
    /// resolved path is guaranteed inside the workspace root, so a stored
    /// row path like `../escape` fails here with Permission before any file
    /// is touched. Deleting an already-missing file is success (the goal
    /// state is deletion).
    fn delete_through_workspace(
        &self,
        workspace: &WorkspaceHandle,
        rel: &Path,
    ) -> Result<(), Error> {
        let resolved = workspace.resolve(rel)?;
        match std::fs::remove_file(&resolved) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(Error::new(
                ErrorKind::Store,
                format!("delete {}: {e}", resolved.display()),
            )),
        }
    }

    /// Redo (unrevert): verify the current file state equals the before-state
    /// (i.e. the rollback really happened), then restore the after-state —
    /// writing the checkpoint's after content back, or DELETING the file when
    /// the after-state is missing. Checkpoints recorded before the after-blob
    /// column existed are refused honestly with a Conflict.
    pub fn redo(
        &self,
        workspace: &WorkspaceHandle,
        identity: &WorkspaceIdentity,
        session: SessionId,
        checkpoint_id: i64,
    ) -> Result<RollbackOutcome, Error> {
        workspace.verify_identity(identity)?;
        let row = self.checkpoint_row(session, checkpoint_id)?;
        let before = side_state(&row, Side::Before)?;
        let after = side_state(&row, Side::After)?;

        let rel = std::path::Path::new(&row.path);
        let current = FileState::probe(workspace, rel)?;
        if current != before {
            return Ok(RollbackOutcome::Conflict {
                path: row.path.clone(),
                current,
                expected_after: before,
            });
        }
        let outcome = match after {
            // The edit's after-state is MISSING: unrevert deletes the file
            // the rollback recreated. No after blob exists — that is the
            // state, not a pre-v3 row.
            FileState { exists: false, .. } => {
                self.delete_through_workspace(workspace, rel)?;
                RollbackOutcome::Restored {
                    path: row.path.clone(),
                    hash: None,
                }
            }
            FileState {
                exists: true,
                hash: Some(after_hash),
            } => {
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
                if new_hash != after_hash {
                    return Err(Error::internal(format!(
                        "redo wrote {} but expected {}",
                        new_hash.to_hex(),
                        after_hash.to_hex()
                    )));
                }
                RollbackOutcome::Restored {
                    path: row.path.clone(),
                    hash: Some(new_hash),
                }
            }
            // Unreachable: record_change/check_state keep states consistent.
            other => return Err(Error::malformed(format!("corrupt after state: {other:?}"))),
        };
        // Redo undoes the rollback: the restored marker is CLEARED (audit
        // round 5 — a row must not read as restored after an unrevert).
        self.store
            .clear_checkpoint_restored(checkpoint_id)
            .map_err(map_store)?;
        Ok(outcome)
    }

    /// Unified line diff of the latest checkpoint's before/after states, with
    /// the status derived from the recorded transition. `Ok(None)` when the
    /// session has no checkpoints. Pre-v3 checkpoints on an existing after
    /// side (no after blob) are refused honestly with a Conflict.
    pub fn diff_latest(
        &self,
        workspace: &WorkspaceHandle,
        identity: &WorkspaceIdentity,
        session: SessionId,
    ) -> Result<Option<DiffResult>, Error> {
        workspace.verify_identity(identity)?;
        let checkpoints = self.store.checkpoints_of(session).map_err(map_store)?;
        let Some(row) = checkpoints.iter().max_by_key(|c| c.sequence) else {
            return Ok(None);
        };
        let before_state = side_state(row, Side::Before)?;
        let after_state = side_state(row, Side::After)?;
        // Resolve the after side FIRST: an existing after state without its
        // CAS blob (pre-v3 rows) is refused honestly before any other work.
        let after_bytes = match after_state {
            // Deletion: the after side has no content.
            FileState { exists: false, .. } => Vec::new(),
            FileState { exists: true, .. } => {
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
                self.cas
                    .get(after_cas)
                    .map_err(|e| Error::new(ErrorKind::Store, format!("cas: {e}")))?
            }
        };
        let before_bytes = match before_state {
            // Creation: there was no before content to diff against.
            FileState { exists: false, .. } => Vec::new(),
            FileState {
                exists: true,
                hash: Some(h),
            } => self
                .cas
                .get(h)
                .map_err(|e| Error::new(ErrorKind::Store, format!("cas: {e}")))?,
            other => return Err(Error::malformed(format!("corrupt before state: {other:?}"))),
        };
        let diff = diff_lines(&before_bytes, &after_bytes)
            .iter()
            .map(DiffLine::render)
            .collect::<Vec<_>>()
            .join("\n");
        Ok(Some(DiffResult {
            path: row.path.clone(),
            diff,
            // Raw rows could theoretically hold equal states; there is no
            // transition to report for them.
            status: ChangeStatus::from_transition(before_state, after_state)
                .unwrap_or(ChangeStatus::Modified),
        }))
    }

    pub fn checkpoints(
        &self,
        session: SessionId,
    ) -> Result<Vec<faktor_store::CheckpointRow>, Error> {
        self.store.checkpoints_of(session).map_err(map_store)
    }

    fn checkpoint_row(
        &self,
        session: SessionId,
        checkpoint_id: i64,
    ) -> Result<faktor_store::CheckpointRow, Error> {
        let checkpoints = self.store.checkpoints_of(session).map_err(map_store)?;
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
/// Line diff of `before` → `after` (audit round 9): the bounded Myers
/// engine from faktor-edit produces per-change hunks, so two distant edits
/// never collapse into one giant replacement. Coarse/prefix-suffix
/// fallbacks (hostile inputs, budget exhaustion) delegate to the legacy
/// algorithm below; identical files short-circuit.
pub fn diff_lines(before: &[u8], after: &[u8]) -> Vec<DiffLine> {
    if before == after {
        return bound_diff(
            split_lines(before)
                .into_iter()
                .map(DiffLine::Context)
                .collect(),
        );
    }
    let outcome = faktor_edit::diff::diff_hunks(before, after);
    if outcome.mode == faktor_edit::diff::DiffMode::Myers {
        let mut lines: Vec<DiffLine> = Vec::new();
        for hunk in &outcome.hunks {
            for dl in &hunk.lines {
                lines.push(match dl {
                    faktor_edit::diff::DiffLine::Context(l) => DiffLine::Context(l.clone()),
                    faktor_edit::diff::DiffLine::Removed(l) => DiffLine::Removed(l.clone()),
                    faktor_edit::diff::DiffLine::Added(l) => DiffLine::Added(l.clone()),
                });
            }
        }
        return bound_diff(lines);
    }
    legacy_prefix_suffix_diff(before, after)
}

fn legacy_prefix_suffix_diff(before: &[u8], after: &[u8]) -> Vec<DiffLine> {
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

fn map_store(e: faktor_store::StoreError) -> Error {
    Error::new(ErrorKind::Store, format!("store: {e}"))
}

/// FileState invariant: exists ⇔ a content hash is present.
fn check_state(state: FileState, side: &str) -> Result<(), Error> {
    if state.exists == state.hash.is_some() {
        Ok(())
    } else {
        Err(Error::malformed(format!(
            "inconsistent {side} FileState: exists={} hash={:?}",
            state.exists,
            state.hash.map(|h| h.to_hex())
        )))
    }
}

/// Hex of a side's hash; a side that does not exist has no content, so the
/// store row carries the empty-string sentinel in its hash column.
fn side_hash(state: FileState) -> String {
    state.hash.map(|h| h.to_hex()).unwrap_or_default()
}

#[derive(Debug, Clone, Copy)]
enum Side {
    Before,
    After,
}

impl Side {
    fn name(self) -> &'static str {
        match self {
            Side::Before => "before",
            Side::After => "after",
        }
    }
}

/// Reconstruct one side's FileState from a store row. Backward
/// compatibility: pre-v6 rows carry no existence marker and read as
/// exists:true — old rows were only recorded for real files, so "hash
/// present with no existence marker means exists:true". A missing side has
/// the empty-string hash sentinel and must not be parsed as a hash.
fn side_state(row: &faktor_store::CheckpointRow, side: Side) -> Result<FileState, Error> {
    let (exists, hash_raw) = match side {
        Side::Before => (row.before_exists, row.before_hash.as_str()),
        Side::After => (row.after_exists, row.after_hash.as_str()),
    };
    if !exists {
        return Ok(FileState::missing());
    }
    let hash = FileHash::from_hex(hash_raw).ok_or_else(|| {
        Error::malformed(format!(
            "corrupt {}_hash: {hash_raw:?} (exists=true requires a real hash)",
            side.name()
        ))
    })?;
    Ok(FileState::existing(hash))
}

#[cfg(test)]
mod tests {
    use super::*;
    use faktor_core::id::{TaskId, WorkspaceId, WorktreeId};
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
        let service = faktor_fs::WorkspaceFileService::new();
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
        let outcome = cps.rollback(&h, &id, session, cid).unwrap();
        match outcome {
            RollbackOutcome::Restored { path, hash } => {
                assert_eq!(path, "f.txt");
                assert_eq!(hash, Some(before));
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
        let outcome = cps.rollback(&h, &id, session, cid).unwrap();
        match outcome {
            RollbackOutcome::Conflict {
                current,
                expected_after,
                ..
            } => {
                assert_eq!(
                    current,
                    FileState::existing(CheckpointStore::hash_of(b"user edit"))
                );
                assert_eq!(expected_after, FileState::existing(after));
            }
            other => panic!("expected Conflict, got {other:?}"),
        }
        // Never overwritten.
        assert_eq!(fs::read(h.root().join("f.txt")).unwrap(), b"user edit");
    }

    #[test]
    fn rollback_unknown_checkpoint_not_found() {
        let (_d, cps, h, id, session) = fixture();
        fs::write(h.root().join("f.txt"), b"x").unwrap();
        let err = cps.rollback(&h, &id, session, 999).unwrap_err();
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
        let r = cps.rollback(&h, &id, session, cid).unwrap();
        assert!(matches!(r, RollbackOutcome::Restored { .. }));
        assert_eq!(fs::read(h.root().join("f.txt")).unwrap(), b"original");
        let outcome = cps.redo(&h, &id, session, cid).unwrap();
        match outcome {
            RollbackOutcome::Restored { path, hash } => {
                assert_eq!(path, "f.txt");
                assert_eq!(hash, Some(after));
            }
            other => panic!("expected Restored, got {other:?}"),
        }
        assert_eq!(
            fs::read(h.root().join("f.txt")).unwrap(),
            b"edited by agent"
        );
        // Redo (unrevert) undoes the rollback: the restored marker must be
        // CLEARED, not left set (audit round 5).
        assert!(
            cps.checkpoints(session).unwrap()[0].restored_ms.is_none(),
            "redo must clear the restored marker"
        );
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
        cps.rollback(&h, &id, session, cid).unwrap();
        fs::write(h.root().join("f.txt"), b"user took over").unwrap();
        let outcome = cps.redo(&h, &id, session, cid).unwrap();
        match outcome {
            RollbackOutcome::Conflict {
                current,
                expected_after,
                ..
            } => {
                assert_eq!(
                    current,
                    FileState::existing(CheckpointStore::hash_of(b"user took over"))
                );
                assert_eq!(expected_after, FileState::existing(before));
            }
            other => panic!("expected Conflict, got {other:?}"),
        }
        assert_eq!(fs::read(h.root().join("f.txt")).unwrap(), b"user took over");
    }

    #[test]
    fn redo_unknown_checkpoint_not_found() {
        let (_d, cps, h, id, session) = fixture();
        fs::write(h.root().join("f.txt"), b"x").unwrap();
        let err = cps.redo(&h, &id, session, 999).unwrap_err();
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
        let err = cps.redo(&h, &id, session, row_id).unwrap_err();
        assert!(
            err.kind == ErrorKind::Conflict && err.message.contains("after-content unavailable"),
            "pre-v3 rows must be refused honestly: {err:?}"
        );
        assert_eq!(fs::read(h.root().join("f.txt")).unwrap(), b"original");
    }

    #[test]
    fn rollback_uses_the_real_session_not_workspace_derived() {
        let (_d, cps, h, id, _fixture_session) = fixture();
        // Session ids are auto-increment per store. The fixture already
        // created session 1; three more dummies make the next session's id
        // 5 — different from the workspace's own id 1. The seam being fixed
        // looked checkpoints up under `workspace_id.into_session()` (id 1),
        // so session 5's rollback would find nothing or the wrong rows.
        let store = cps.store.clone();
        for i in 0..3 {
            store
                .create_session(id.workspace_id, &format!("dummy {i}"), "p", "m")
                .unwrap();
        }
        let row = store
            .create_session(id.workspace_id, "real", "p", "m")
            .unwrap();
        let session = row.id;
        assert_eq!(session.raw(), 5, "session id must differ from workspace id");
        assert_ne!(session.raw(), id.workspace_id.raw());

        fs::write(h.root().join("f.txt"), b"original").unwrap();
        let before = cps.before_write(session, "f.txt", b"original").unwrap();
        fs::write(h.root().join("f.txt"), b"edited").unwrap();
        let after_content = b"edited";
        let after = CheckpointStore::hash_of(after_content);
        let cid = cps
            .after_write(session, "f.txt", before, after, 1, after_content)
            .unwrap();

        // The REAL session rolls back.
        let outcome = cps.rollback(&h, &id, session, cid).unwrap();
        assert!(matches!(outcome, RollbackOutcome::Restored { .. }));
        assert_eq!(fs::read(h.root().join("f.txt")).unwrap(), b"original");
        // A session whose id equals the workspace's id finds nothing: the
        // checkpoint lives under session 5, never under session 1.
        let workspace_derived = SessionId::new(id.workspace_id.raw());
        let err = cps.rollback(&h, &id, workspace_derived, cid).unwrap_err();
        assert!(
            err.kind == ErrorKind::NotFound,
            "workspace-derived lookup must not find session 5's checkpoint: {err:?}"
        );
    }

    #[test]
    fn redo_and_diff_latest_use_the_real_session() {
        let (_d, cps, h, id, _fixture_session) = fixture();
        let store = cps.store.clone();
        for i in 0..3 {
            store
                .create_session(id.workspace_id, &format!("dummy {i}"), "p", "m")
                .unwrap();
        }
        let row = store
            .create_session(id.workspace_id, "real", "p", "m")
            .unwrap();
        let session = row.id;
        assert_eq!(session.raw(), 5, "session id must differ from workspace id");
        let workspace_derived = SessionId::new(id.workspace_id.raw());

        fs::write(h.root().join("f.txt"), b"original").unwrap();
        let before = cps.before_write(session, "f.txt", b"original").unwrap();
        fs::write(h.root().join("f.txt"), b"edited").unwrap();
        let after_content = b"edited";
        let after = CheckpointStore::hash_of(after_content);
        let cid = cps
            .after_write(session, "f.txt", before, after, 1, after_content)
            .unwrap();

        // diff_latest under the real session sees the checkpoint; under the
        // workspace-derived id there are none.
        let result = cps.diff_latest(&h, &id, session).unwrap().unwrap();
        assert_eq!(result.path, "f.txt");
        assert!(result.diff.contains("+edited"), "{result:?}");
        assert!(
            cps.diff_latest(&h, &id, workspace_derived)
                .unwrap()
                .is_none(),
            "workspace-derived diff must not see session 5's checkpoint"
        );

        // redo (unrevert) under the real session: rollback then restore.
        let r = cps.rollback(&h, &id, session, cid).unwrap();
        assert!(matches!(r, RollbackOutcome::Restored { .. }));
        assert_eq!(fs::read(h.root().join("f.txt")).unwrap(), b"original");
        let outcome = cps.redo(&h, &id, session, cid).unwrap();
        assert!(matches!(outcome, RollbackOutcome::Restored { .. }));
        assert_eq!(fs::read(h.root().join("f.txt")).unwrap(), b"edited");
        // redo under the workspace-derived session finds nothing.
        let err = cps.redo(&h, &id, workspace_derived, cid).unwrap_err();
        assert!(
            err.kind == ErrorKind::NotFound,
            "workspace-derived redo must not find session 5's checkpoint: {err:?}"
        );
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
        assert!(cps.rollback(&h, &wrong, session, cid).is_err());
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
        let err = cps.rollback(&h, &id, session, corrupt_id).unwrap_err();
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
        let err = cps.rollback(&h, &id, session, cid).unwrap_err();
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
        let result = cps.diff_latest(&h, &id, session).unwrap().unwrap();
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
        let (_d, cps, h, id, session) = fixture();
        assert!(cps.diff_latest(&h, &id, session).unwrap().is_none());
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
        let result = cps.diff_latest(&h, &id, session).unwrap().unwrap();
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
        let err = cps.diff_latest(&h, &id, session).unwrap_err();
        assert!(
            err.kind == ErrorKind::Conflict && err.message.contains("after-content unavailable"),
            "pre-v3 rows must be refused honestly: {err:?}"
        );
    }

    #[test]
    fn empty_file_creation_records_checkpoint_and_rollback_deletes() {
        // THE P0 BUG: hash("")==hash("") made empty-file creation look like a
        // no-op, so the checkpoint was skipped and the empty file could never
        // be undone. With existence-bearing states, before {exists:false} vs
        // after {exists:true, empty-hash} is a REAL transition and IS
        // recorded; rollback deletes the deliberately created empty file.
        let (_d, cps, h, id, session) = fixture();
        let rel = Path::new("empty.txt");
        assert!(!h.exists(rel));
        let empty_hash = CheckpointStore::hash_of(b"");
        let cid = cps
            .record_change(
                session,
                "empty.txt",
                FileState::missing(),
                None,
                FileState::existing(empty_hash),
                Some(b""),
            )
            .unwrap();
        let rows = cps.checkpoints(session).unwrap();
        assert_eq!(
            rows.len(),
            1,
            "missing->empty must NOT be skipped as a no-op"
        );
        assert!(!rows[0].before_exists);
        assert_eq!(
            rows[0].before_hash, "",
            "missing side carries the empty sentinel"
        );
        assert!(rows[0].after_exists);
        assert_eq!(rows[0].after_hash, empty_hash.to_hex());

        // The tool's effect: the empty file now exists on disk.
        fs::write(h.root().join("empty.txt"), b"").unwrap();
        let outcome = cps.rollback(&h, &id, session, cid).unwrap();
        match outcome {
            RollbackOutcome::Restored { hash, .. } => {
                assert_eq!(hash, None, "deleting a file restores no content hash")
            }
            other => panic!("expected Restored, got {other:?}"),
        }
        assert!(
            !h.exists(rel),
            "rollback must DELETE the deliberately created empty file"
        );
        assert!(cps.checkpoints(session).unwrap()[0].restored_ms.is_some());
        // Redo (unrevert) recreates the empty file: it EXISTS with zero bytes.
        let outcome = cps.redo(&h, &id, session, cid).unwrap();
        assert!(matches!(outcome, RollbackOutcome::Restored { .. }));
        assert!(h.exists(rel), "redo must recreate the empty file");
        assert_eq!(fs::read(h.root().join("empty.txt")).unwrap(), b"");
        assert!(cps.checkpoints(session).unwrap()[0].restored_ms.is_none());
    }

    #[test]
    fn missing_to_content_rollback_deletes_the_file() {
        let (_d, cps, h, id, session) = fixture();
        let rel = Path::new("new.rs");
        let content = b"brand new non-empty file";
        let cid = cps
            .record_change(
                session,
                "new.rs",
                FileState::missing(),
                None,
                FileState::existing(CheckpointStore::hash_of(content)),
                Some(content),
            )
            .unwrap();
        fs::write(h.root().join("new.rs"), content).unwrap();
        assert!(h.exists(rel));
        let outcome = cps.rollback(&h, &id, session, cid).unwrap();
        assert!(
            matches!(outcome, RollbackOutcome::Restored { hash: None, .. }),
            "rollback of a creation deletes: {outcome:?}"
        );
        assert!(
            !h.exists(rel),
            "rollback must delete a file that did not exist before"
        );
    }

    #[test]
    fn content_to_missing_rollback_recreates_exact_bytes() {
        let (_d, cps, h, id, session) = fixture();
        let original = b"exact bytes to bring back, \x00\x01\x02 and more";
        fs::write(h.root().join("del.txt"), original).unwrap();
        let before_hash = CheckpointStore::hash_of(original);
        let cid = cps
            .record_change(
                session,
                "del.txt",
                FileState::existing(before_hash),
                Some(original),
                FileState::missing(),
                None,
            )
            .unwrap();
        // The delete tool's effect: file gone, which matches the after state.
        fs::remove_file(h.root().join("del.txt")).unwrap();
        assert!(!h.exists(Path::new("del.txt")));
        let outcome = cps.rollback(&h, &id, session, cid).unwrap();
        match outcome {
            RollbackOutcome::Restored { hash, .. } => {
                assert_eq!(hash, Some(before_hash));
            }
            other => panic!("expected Restored, got {other:?}"),
        }
        assert_eq!(
            fs::read(h.root().join("del.txt")).unwrap(),
            original,
            "rollback must recreate the file with its exact original bytes"
        );
        // Redo re-deletes (the after state is missing).
        let outcome = cps.redo(&h, &id, session, cid).unwrap();
        assert!(matches!(
            outcome,
            RollbackOutcome::Restored { hash: None, .. }
        ));
        assert!(!h.exists(Path::new("del.txt")));
    }

    #[test]
    fn empty_file_to_content_rollback_restores_the_empty_file() {
        let (_d, cps, h, id, session) = fixture();
        let empty_hash = CheckpointStore::hash_of(b"");
        let content = b"the empty file grows content";
        let cid = cps
            .record_change(
                session,
                "grew.txt",
                FileState::existing(empty_hash),
                Some(b""),
                FileState::existing(CheckpointStore::hash_of(content)),
                Some(content),
            )
            .unwrap();
        fs::write(h.root().join("grew.txt"), content).unwrap();
        let outcome = cps.rollback(&h, &id, session, cid).unwrap();
        match outcome {
            RollbackOutcome::Restored { hash, .. } => {
                assert_eq!(hash, Some(empty_hash));
            }
            other => panic!("expected Restored, got {other:?}"),
        }
        assert!(
            h.exists(Path::new("grew.txt")),
            "rollback must restore the EMPTY file as an existing zero-byte file"
        );
        assert_eq!(fs::read(h.root().join("grew.txt")).unwrap(), b"");
    }

    #[test]
    fn truncate_to_empty_rollback_restores_original_content() {
        let (_d, cps, h, id, session) = fixture();
        let original = b"original content that must survive a truncate";
        fs::write(h.root().join("t.txt"), original).unwrap();
        let empty_hash = CheckpointStore::hash_of(b"");
        let cid = cps
            .record_change(
                session,
                "t.txt",
                FileState::existing(CheckpointStore::hash_of(original)),
                Some(original),
                FileState::existing(empty_hash),
                Some(b""),
            )
            .unwrap();
        // Truncate: the file exists but is now empty (NOT missing).
        fs::write(h.root().join("t.txt"), b"").unwrap();
        let outcome = cps.rollback(&h, &id, session, cid).unwrap();
        assert!(matches!(
            outcome,
            RollbackOutcome::Restored { hash: Some(_), .. }
        ));
        assert_eq!(
            fs::read(h.root().join("t.txt")).unwrap(),
            original,
            "rollback must restore the pre-truncate content"
        );
    }

    #[test]
    fn rollback_conflicts_on_existence_mismatch_never_clobbers() {
        // Existence is part of the verified state, both directions.
        let (_d, cps, h, id, session) = fixture();
        // (a) after-state says the file EXISTS; the user deleted it instead.
        let content = b"agent content";
        let cid = cps
            .record_change(
                session,
                "a.txt",
                FileState::missing(),
                None,
                FileState::existing(CheckpointStore::hash_of(content)),
                Some(content),
            )
            .unwrap();
        fs::write(h.root().join("a.txt"), content).unwrap();
        fs::remove_file(h.root().join("a.txt")).unwrap(); // user deletes
        let outcome = cps.rollback(&h, &id, session, cid).unwrap();
        match outcome {
            RollbackOutcome::Conflict {
                current,
                expected_after,
                ..
            } => {
                assert_eq!(current, FileState::missing());
                assert!(expected_after.exists);
            }
            other => panic!("missing current vs existing after must conflict: {other:?}"),
        }
        assert!(!h.exists(Path::new("a.txt")), "conflict must never write");
        // (b) after-state says MISSING; the user recreated the file.
        let original = b"deleted by the agent";
        fs::write(h.root().join("b.txt"), original).unwrap();
        let cid2 = cps
            .record_change(
                session,
                "b.txt",
                FileState::existing(CheckpointStore::hash_of(original)),
                Some(original),
                FileState::missing(),
                None,
            )
            .unwrap();
        fs::remove_file(h.root().join("b.txt")).unwrap(); // agent delete effect
        fs::write(h.root().join("b.txt"), b"user recreated").unwrap();
        let outcome = cps.rollback(&h, &id, session, cid2).unwrap();
        match outcome {
            RollbackOutcome::Conflict { current, .. } => {
                assert!(
                    current.exists,
                    "user's recreation is a state, not emptiness"
                );
            }
            other => panic!("recreated file vs missing after must conflict: {other:?}"),
        }
        assert_eq!(
            fs::read(h.root().join("b.txt")).unwrap(),
            b"user recreated",
            "conflict must never overwrite"
        );
    }

    #[test]
    fn diff_status_maps_state_transitions() {
        // Pure mapping across every transition (the audit's projection).
        let h1 = FileHash::from([1; 32]);
        let h2 = FileHash::from([2; 32]);
        assert_eq!(
            ChangeStatus::from_transition(FileState::missing(), FileState::existing(h1)),
            Some(ChangeStatus::Added)
        );
        assert_eq!(
            ChangeStatus::from_transition(FileState::existing(h1), FileState::missing()),
            Some(ChangeStatus::Deleted)
        );
        assert_eq!(
            ChangeStatus::from_transition(FileState::existing(h1), FileState::existing(h2)),
            Some(ChangeStatus::Modified)
        );
        assert_eq!(
            ChangeStatus::from_transition(FileState::existing(h1), FileState::existing(h1)),
            None,
            "identical states have no diff status"
        );
        assert_eq!(
            ChangeStatus::from_transition(FileState::missing(), FileState::missing()),
            None
        );

        // The recorded rows derive the same statuses end to end.
        let (_d, cps, h, id, session) = fixture();
        let rel = Path::new("s.txt");
        fs::write(h.root().join(rel), b"one").unwrap();
        cps.record_change(
            session,
            "s.txt",
            FileState::missing(),
            None,
            FileState::existing(CheckpointStore::hash_of(b"one")),
            Some(b"one"),
        )
        .unwrap();
        fs::write(h.root().join(rel), b"two").unwrap();
        cps.record_change(
            session,
            "s.txt",
            FileState::existing(CheckpointStore::hash_of(b"one")),
            Some(b"one"),
            FileState::existing(CheckpointStore::hash_of(b"two")),
            Some(b"two"),
        )
        .unwrap();
        fs::remove_file(h.root().join(rel)).unwrap();
        cps.record_change(
            session,
            "s.txt",
            FileState::existing(CheckpointStore::hash_of(b"two")),
            Some(b"two"),
            FileState::missing(),
            None,
        )
        .unwrap();
        let result = cps.diff_latest(&h, &id, session).unwrap().unwrap();
        assert_eq!(result.status, ChangeStatus::Deleted);
        assert_eq!(result.path, "s.txt");
        assert!(result.diff.contains("-two"), "{:?}", result.diff);
    }

    #[test]
    fn checkpoint_sequence_race_never_allocates_twice() {
        // P1: two writers racing to checkpoint must receive distinct
        // sequences even when both PASS THE SAME stale guess (the old
        // rows.len()+1 derivation). The store allocates atomically.
        let (_d, cps, _h, _id, session) = fixture();
        let cps = Arc::new(cps);
        let mut handles = Vec::new();
        for t in 0..2 {
            let cps = cps.clone();
            handles.push(std::thread::spawn(move || {
                for i in 0..25 {
                    let path = format!("w{t}-{i}.rs");
                    let before = cps.before_write(session, &path, b"x").unwrap();
                    let after_content = format!("content-{t}-{i}");
                    // Every call passes the SAME sequence: only the store's
                    // atomic allocation can keep the numbers distinct.
                    cps.after_write(
                        session,
                        &path,
                        before,
                        CheckpointStore::hash_of(after_content.as_bytes()),
                        9999,
                        after_content.as_bytes(),
                    )
                    .unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let rows = cps.checkpoints(session).unwrap();
        assert_eq!(rows.len(), 50, "every racing checkpoint must land");
        let mut seqs: Vec<i64> = rows.iter().map(|c| c.sequence).collect();
        seqs.sort_unstable();
        for (i, seq) in seqs.iter().enumerate() {
            assert_eq!(
                *seq,
                (i + 1) as i64,
                "sequences must be distinct and gapless"
            );
        }
        let unique: std::collections::HashSet<i64> = seqs.iter().copied().collect();
        assert_eq!(unique.len(), 50, "two writers must never both receive N+1");
    }

    #[test]
    fn rollback_refuses_hostile_relative_path() {
        // A checkpoint row whose path escapes the workspace root must never
        // make rollback read or write outside it: resolution through the
        // workspace handle (canonical root + traversal rejection) refuses.
        let (dir, cps, h, id, session) = fixture();
        let outside = dir.path().join("escape-target.txt");
        fs::write(&outside, b"outside secret").unwrap();
        // The deletion checkpoint (before exists, after missing) is the
        // dangerous shape: restoring it WRITES through the recorded path.
        let before_hash = CheckpointStore::hash_of(b"would-be content");
        cps.before_write(session, "escape.txt", b"would-be content")
            .unwrap();
        cps.store
            .insert_checkpoint(
                session,
                "../escape-target.txt",
                true,
                &before_hash.to_hex(),
                false,
                "",
                None,
            )
            .unwrap();
        // The hostile path resolves to nothing inside the workspace: the
        // current state probe sees "missing", which matches the after state.
        let row_id = cps.checkpoints(session).unwrap()[0].id;
        let err = cps.rollback(&h, &id, session, row_id).unwrap_err();
        assert!(
            err.kind == ErrorKind::Permission,
            "hostile relative path must be refused by resolution, got: {err:?}"
        );
        assert_eq!(
            fs::read(&outside).unwrap(),
            b"outside secret",
            "rollback must never touch a file outside the workspace"
        );
        assert!(!h.exists(Path::new("escape-target.txt")));
    }

    #[test]
    fn record_change_rejects_inconsistent_states() {
        let (_d, cps, _h, _id, session) = fixture();
        // exists without a hash.
        let bad = FileState {
            exists: true,
            hash: None,
        };
        let err = cps
            .record_change(
                session,
                "f.txt",
                bad,
                Some(b"x"),
                FileState::existing(CheckpointStore::hash_of(b"y")),
                Some(b"y"),
            )
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Malformed);
        // Hash declared but the bytes hash to something else: loud, never
        // recorded with a lying hash.
        let err = cps
            .record_change(
                session,
                "f.txt",
                FileState::missing(),
                None,
                FileState::existing(FileHash::from([9; 32])),
                Some(b"different"),
            )
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Internal);
        // Content on a missing side and equal states are malformed too.
        let err = cps
            .record_change(
                session,
                "f.txt",
                FileState::missing(),
                Some(b"x"),
                FileState::existing(CheckpointStore::hash_of(b"y")),
                Some(b"y"),
            )
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Malformed);
        let h = CheckpointStore::hash_of(b"same");
        let err = cps
            .record_change(
                session,
                "f.txt",
                FileState::existing(h),
                Some(b"same"),
                FileState::existing(h),
                Some(b"same"),
            )
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Malformed);
        assert!(cps.checkpoints(session).unwrap().is_empty());
    }

    impl CheckpointStore {
        fn hash_of(bytes: &[u8]) -> FileHash {
            FileHash::from(blake3::hash(bytes).into())
        }
    }

    #[test]
    fn distant_edits_never_collapse_into_one_giant_replacement() {
        // Audit round 9: with changes at line 20 and line 900 of a 1000-line
        // file, the diff must NOT paint lines 20-900 as one replacement.
        let mut before = String::new();
        let mut after = String::new();
        for i in 1..=1000 {
            let a = format!("line {i:04}");
            let b = if i == 20 {
                "CHANGED TWENTY".to_string()
            } else if i == 900 {
                "CHANGED NINE HUNDRED".to_string()
            } else {
                a.clone()
            };
            before.push_str(&a);
            before.push('\n');
            after.push_str(&b);
            after.push('\n');
        }
        let lines = diff_lines(before.as_bytes(), after.as_bytes());
        let removed = lines
            .iter()
            .filter(|l| matches!(l, DiffLine::Removed(_)))
            .count();
        let added = lines
            .iter()
            .filter(|l| matches!(l, DiffLine::Added(_)))
            .count();
        // Two one-line edits: exactly two removed + two added, not 881.
        assert_eq!(
            removed, 2,
            "only the changed old lines are removed: {removed}"
        );
        assert_eq!(added, 2, "only the changed new lines are added: {added}");
        let rendered = lines.iter().map(|l| l.render()).collect::<String>();
        assert!(rendered.contains("CHANGED TWENTY"));
        assert!(rendered.contains("CHANGED NINE HUNDRED"));
        // Context padding around the two hunks, no giant middle blob.
        assert!(
            !rendered.contains("line 0021\n-line 0022"),
            "no middle blob"
        );
    }
}
