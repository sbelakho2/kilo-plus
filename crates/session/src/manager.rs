//! The session manager: opens the durable store + CAS, owns the per-session
//! resource registries, and is the factory for `SessionHandle`s.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use kilop_cas::Cas;
use kilop_core::id::{OpId, SessionId, WorkspaceId};
use kilop_core::time::{Clock, SystemClock};
use kilop_store::Store;

use crate::artifacts::ArtifactSizes;
use crate::handle::SessionHandle;
use crate::ops::OpRegistry;
use crate::process::ProcessRegistry;
use crate::recovery::SystemFileHasher;
use crate::SessionError;

/// Per-session in-memory resources shared by every handle to the same session,
/// so cancellation and process ownership are global per session, not per
/// handle clone.
#[derive(Debug)]
pub(crate) struct SessionResources {
    pub(crate) ops: OpRegistry,
    pub(crate) processes: ProcessRegistry,
    /// Serializes read-validate-append transition sequences per session so
    /// concurrent callers cannot both validate against the same state.
    pub(crate) command_lock: std::sync::Mutex<()>,
}

/// The entry point of `kilop-session`. One manager per daemon data root.
pub struct SessionManager {
    store: Arc<Store>,
    cas: Arc<Cas>,
    clock: Arc<dyn Clock>,
    op_counter: AtomicU64,
    resources: Mutex<HashMap<SessionId, Arc<SessionResources>>>,
    system_hasher: Arc<SystemFileHasher>,
    pub(crate) artifact_sizes: ArtifactSizes,
}

impl std::fmt::Debug for SessionManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionManager").finish_non_exhaustive()
    }
}

impl SessionManager {
    /// Open (creating if needed) the SQLite store at `root` and the CAS at
    /// `cas_root`. `integrity_check: true` refuses to open a corrupt store.
    pub fn open(
        root: impl Into<PathBuf>,
        cas_root: impl Into<PathBuf>,
        integrity_check: bool,
    ) -> kilop_core::Result<Arc<SessionManager>> {
        Self::open_with_clock(root, cas_root, integrity_check, Arc::new(SystemClock))
    }

    /// Same as [`SessionManager::open`] with an injectable clock (tests).
    pub fn open_with_clock(
        root: impl Into<PathBuf>,
        cas_root: impl Into<PathBuf>,
        integrity_check: bool,
        clock: Arc<dyn Clock>,
    ) -> kilop_core::Result<Arc<SessionManager>> {
        let store = Arc::new(Store::open(root, integrity_check).map_err(SessionError::from)?);
        let cas_root = cas_root.into();
        let cas = Cas::open(cas_root).map_err(SessionError::from)?;
        let cas = Arc::new(cas);
        let system_hasher = Arc::new(SystemFileHasher::new(cas.clone()));
        Ok(Arc::new(SessionManager {
            store,
            cas,
            clock,
            op_counter: AtomicU64::new(1),
            resources: Mutex::new(HashMap::new()),
            system_hasher,
            artifact_sizes: ArtifactSizes::default(),
        }))
    }

    /// The durable store, shared with snapshot/redo consumers (the wire
    /// revert/unrevert/diff surface builds its checkpoint store over the
    /// same `Arc<Store>` so both sides see the same rows).
    pub fn store(&self) -> Arc<Store> {
        self.store.clone()
    }

    /// The content-addressed blob store, shared with snapshot consumers.
    pub fn cas(&self) -> Arc<Cas> {
        self.cas.clone()
    }

    /// Current wall-clock time (milliseconds since the epoch).
    pub fn now_ms(&self) -> i64 {
        self.clock.now_ms()
    }

    /// A fresh, non-zero, process-unique operation id. Uniqueness comes from a
    /// per-manager counter mixed with the clock; zero is contractually never
    /// returned.
    /// `doctor`-style health report: store diagnostics + recovery scan.
    pub fn integrity_report(&self) -> kilop_core::Result<serde_json::Value> {
        let diagnostics = self.store().diagnostics().map_err(crate::map_store_err)?;
        let pending = self
            .store()
            .pending_tool_runs(SessionId::new(1))
            .unwrap_or_default()
            .len();
        let mut v = diagnostics.as_object().cloned().unwrap_or_default();
        v.insert("orphaned_runs".into(), serde_json::json!(pending));
        Ok(serde_json::Value::Object(v))
    }

    pub fn next_op_id(&self) -> OpId {
        let seed = self.now_ms() as u64;
        loop {
            let n = self.op_counter.fetch_add(1, Ordering::Relaxed);
            let raw = seed.wrapping_add(n);
            if raw != 0 {
                return OpId::new(raw);
            }
        }
    }

    fn resources(&self, id: SessionId) -> Arc<SessionResources> {
        let mut map = self.resources.lock().expect("session resources poisoned");
        map.entry(id)
            .or_insert_with(|| {
                Arc::new(SessionResources {
                    ops: OpRegistry::default(),
                    processes: ProcessRegistry::default(),
                    command_lock: std::sync::Mutex::new(()),
                })
            })
            .clone()
    }

    // ---------------------------------------------------------------- workspaces

    pub fn create_workspace(&self, root: &str) -> kilop_core::Result<WorkspaceId> {
        self.store
            .create_workspace(root)
            .map_err(|e| crate::map_store_err(e).into())
    }

    // ---------------------------------------------------------------- worktrees

    pub fn put_worktree(
        &self,
        ws: WorkspaceId,
        path: &str,
        branch: &str,
    ) -> kilop_core::Result<i64> {
        if path.is_empty() || branch.is_empty() {
            return Err(SessionError::Malformed(
                "worktree path and branch must be non-empty".into(),
            )
            .into());
        }
        self.store
            .put_worktree(ws, path, branch)
            .map_err(|e| crate::map_store_err(e).into())
    }

    pub fn worktrees_of(
        &self,
        ws: WorkspaceId,
    ) -> kilop_core::Result<Vec<kilop_store::WorktreeRow>> {
        self.store
            .worktrees_of(ws)
            .map_err(|e| crate::map_store_err(e).into())
    }

    pub fn remove_worktree(&self, path: &str) -> kilop_core::Result<()> {
        self.store
            .remove_worktree(path)
            .map_err(|e| crate::map_store_err(e).into())
    }

    // ---------------------------------------------------------------- sessions

    /// Create a session; returns a handle wired to the manager's shared
    /// per-session resources.
    pub fn create_session(
        self: &Arc<Self>,
        ws: WorkspaceId,
        title: &str,
        provider: &str,
        model: &str,
    ) -> kilop_core::Result<SessionHandle> {
        if title.len() > 4096 || provider.len() > 256 || model.len() > 256 {
            return Err(SessionError::Oversized(
                "session title/provider/model exceed bounds".into(),
            )
            .into());
        }
        let row = self
            .store
            .create_session(ws, title, provider, model)
            .map_err(crate::map_store_err)?;
        Ok(SessionHandle::new(
            self.clone(),
            row.id,
            self.resources(row.id),
            self.system_hasher.clone(),
        ))
    }

    pub fn get_session(
        self: &Arc<Self>,
        id: SessionId,
    ) -> kilop_core::Result<Option<SessionHandle>> {
        match self.store.get_session(id).map_err(crate::map_store_err)? {
            Some(_row) => Ok(Some(SessionHandle::new(
                self.clone(),
                id,
                self.resources(id),
                self.system_hasher.clone(),
            ))),
            None => Ok(None),
        }
    }

    pub fn list_sessions(
        self: &Arc<Self>,
        ws: Option<WorkspaceId>,
    ) -> kilop_core::Result<Vec<SessionHandle>> {
        let rows = self.store.list_sessions(ws).map_err(crate::map_store_err)?;
        Ok(rows
            .into_iter()
            .map(|r| {
                SessionHandle::new(
                    self.clone(),
                    r.id,
                    self.resources(r.id),
                    self.system_hasher.clone(),
                )
            })
            .collect())
    }

    /// Crash-recovery sweep over every session: each handle's journal is
    /// reconstructed and unfinished tool runs are resolved. Idempotent.
    pub fn recover_all_sessions(
        self: &Arc<Self>,
    ) -> kilop_core::Result<Vec<crate::recovery::RecoveryReport>> {
        let handles = self.list_sessions(None)?;
        let mut reports = Vec::with_capacity(handles.len());
        for h in handles {
            reports.push(h.recover_all()?);
        }
        Ok(reports)
    }

    /// Fork `source` into a NEW session: same workspace/provider/model, the
    /// given title, and a durable copy of every message row and its parts,
    /// ascending by sequence. The copy is made row-by-row through the store
    /// (pages of at most [`FORK_PAGE_SIZE`]), so a crash mid-fork leaves the
    /// fork partially copied — never a torn source (the source is only
    /// read). The fork starts `Idle` with its own `SessionCreated` journal
    /// event; it is fully independent afterwards.
    pub fn fork_session(
        self: &Arc<Self>,
        source: SessionId,
        title: &str,
    ) -> kilop_core::Result<SessionHandle> {
        let src_row = match self
            .store
            .get_session(source)
            .map_err(crate::map_store_err)?
        {
            Some(r) => r,
            None => {
                return Err(SessionError::NotFound(format!("session {source}")).into());
            }
        };
        let fork = self.create_session(
            src_row.workspace_id,
            title,
            &src_row.provider,
            &src_row.model,
        )?;
        // Walk the source history newest-first in bounded pages (paging is
        // fundamental), then copy ascending so the fork's seqs stay
        // contiguous and its `proposed_message_seq` keeps working.
        let mut newest_first: Vec<kilop_store::MessageRow> = Vec::new();
        let mut cursor: Option<i64> = None;
        loop {
            let page = self
                .store
                .messages_before(source, cursor, FORK_PAGE_SIZE)
                .map_err(crate::map_store_err)?;
            if page.is_empty() {
                break;
            }
            let exhausted = page.len() < FORK_PAGE_SIZE as usize;
            newest_first.extend(page);
            if exhausted {
                break;
            }
            cursor = newest_first.last().map(|r| r.seq);
        }
        for m in newest_first.into_iter().rev() {
            let new_message_id = self
                .store
                .put_message(fork.id(), m.seq, &m.role, m.data.clone())
                .map_err(crate::map_store_err)?;
            for p in self.store.parts_of(m.id).map_err(crate::map_store_err)? {
                // Part payloads were validated when the source row was
                // written; the copy preserves them verbatim.
                self.store
                    .put_part(new_message_id, &p.kind, p.data.clone())
                    .map_err(crate::map_store_err)?;
            }
        }
        Ok(fork)
    }

    /// Delete a session durably: refuses while a turn record is active or
    /// the turn machine is mid-turn, cancels any lingering queued prompt
    /// rows, ends the session (durable `SessionEnded` journal event +
    /// `lifecycle = Closed`), and closes the per-session in-process
    /// registries (handles). The store keeps the session row (the durable
    /// Closed marker is the tombstone; a row-drop API does not exist in this
    /// slice of the workspace) — a deleted session reads as
    /// `Completed`/`Closed` forever after, and prompts are refused.
    pub fn delete_session(self: &Arc<Self>, id: SessionId) -> kilop_core::Result<()> {
        let handle = match self.get_session(id)? {
            Some(h) => h,
            None => return Err(SessionError::NotFound(format!("session {id}")).into()),
        };
        let row = handle.row()?;
        if let Some(record) = self
            .store
            .active_turn_record(id)
            .map_err(crate::map_store_err)?
        {
            return Err(SessionError::Conflict(format!(
                "session {id} is mid-turn (active turn record {}); refuse to delete",
                record.turn_op_id
            ))
            .into());
        }
        if row.state.is_active() {
            return Err(SessionError::Conflict(format!(
                "session {id} is mid-turn ({:?}); refuse to delete",
                row.state
            ))
            .into());
        }
        if row.lifecycle.is_terminal() {
            return Err(SessionError::Conflict(format!(
                "session {id} is already {:?}; nothing to delete",
                row.lifecycle
            ))
            .into());
        }
        // A session whose turn machine failed recoverably must be reset to
        // Idle before the SessionEnded transition (FailedRecoverable may not
        // jump to Completed).
        if row.state == kilop_core::state::AgentState::FailedRecoverable {
            handle.reset()?;
        }
        // Hygiene: any lingering queued-prompt rows are cancelled durably so
        // the deleted session never admits them.
        let queued = self.store.queue_op_ids(id).map_err(crate::map_store_err)?;
        if !queued.is_empty() {
            self.store
                .cancel_queued_ops(id, &queued)
                .map_err(crate::map_store_err)?;
        }
        handle.end_session()?;
        // Close handles: drop the per-session registries (ops/processes/
        // locks) so nothing references the session in-process any more.
        self.resources
            .lock()
            .expect("session resources poisoned")
            .remove(&id);
        Ok(())
    }
}

/// One fork copy page (bounded everything: the walk never materializes more
/// than one page of source rows at a time).
const FORK_PAGE_SIZE: u64 = 500;

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_manager() -> (tempfile::TempDir, Arc<SessionManager>) {
        let dir = tempfile::tempdir().unwrap();
        let m =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        (dir, m)
    }

    #[test]
    fn op_id_factory_never_zero_and_unique() {
        let (_d, m) = tmp_manager();
        let mut seen = std::collections::HashSet::new();
        for _ in 0..10_000 {
            let id = m.next_op_id();
            assert_ne!(id.raw(), 0, "zero is contractually impossible");
            assert!(seen.insert(id.raw()), "op ids must be unique");
        }
    }

    #[test]
    fn worktree_crud_roundtrip_and_idempotent_put() {
        let (_d, m) = tmp_manager();
        let ws = m.create_workspace("/ws").unwrap();
        let a = m.put_worktree(ws, "/ws/wt1", "feat/x").unwrap();
        let b = m.put_worktree(ws, "/ws/wt1", "feat/x").unwrap();
        assert_eq!(a, b, "INSERT OR IGNORE must be idempotent");
        let rows = m.worktrees_of(ws).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].branch, "feat/x");
        assert!(rows[0].active);
        m.remove_worktree("/ws/wt1").unwrap();
        assert!(m.worktrees_of(ws).unwrap().is_empty());
        // Malformed inputs are rejected before touching the store.
        assert!(m.put_worktree(ws, "", "b").is_err());
        assert!(m.put_worktree(ws, "/p", "").is_err());
    }

    #[test]
    fn session_row_carries_explicit_workspace_identity() {
        // Every session row carries its WorkspaceId explicitly; there is no
        // implicit "current directory" anywhere.
        let (_d, m) = tmp_manager();
        let ws = m.create_workspace("/root").unwrap();
        let s = m.create_session(ws, "t", "ollama", "qwen3.8").unwrap();
        let row = s.row().unwrap();
        assert_eq!(row.workspace_id, ws);
        assert_eq!(s.id(), row.id);
    }

    #[test]
    fn create_session_rejects_oversized_metadata() {
        let (_d, m) = tmp_manager();
        let ws = m.create_workspace("/w").unwrap();
        let huge = "x".repeat(5000);
        assert!(m.create_session(ws, &huge, "p", "m").is_err());
        assert!(m.create_session(ws, "t", &huge, "m").is_err());
        assert!(m.create_session(ws, "t", "p", &huge).is_err());
        // Nothing was written.
        assert!(m.list_sessions(None).unwrap().is_empty());
    }

    #[test]
    fn fork_session_copies_messages_and_parts_in_order() {
        let (_d, m) = tmp_manager();
        let ws = m.create_workspace("/root").unwrap();
        let s = m.create_session(ws, "orig", "ollama", "qwen3.8").unwrap();
        // Two messages with parts, plus a tool_call/tool_result pair whose
        // call ids must survive verbatim.
        let mid1 = s
            .put_message(1, "user", serde_json::json!({"text": "hi"}))
            .unwrap();
        s.put_text_part(mid1, "hi there").unwrap();
        let mid2 = s
            .put_message(2, "assistant", serde_json::json!({"parts": []}))
            .unwrap();
        s.put_tool_call_part(mid2, "c1", "echo", serde_json::json!({"x": 1}), "completed")
            .unwrap();
        let fork = m.fork_session(s.id(), "orig (fork)").unwrap();
        assert_ne!(fork.id(), s.id());
        let fork_row = fork.row().unwrap();
        assert_eq!(fork_row.title, "orig (fork)");
        assert_eq!(fork_row.workspace_id, ws);
        assert_eq!(fork_row.provider, "ollama");
        assert_eq!(fork_row.model, "qwen3.8");
        // Rows and parts copied in order, tool call ids intact.
        let source_page = s.messages_page(None, 100).unwrap();
        let fork_page = fork.messages_page(None, 100).unwrap();
        assert_eq!(source_page.messages.len(), 2);
        assert_eq!(fork_page.messages.len(), 2);
        for (a, b) in source_page.messages.iter().zip(&fork_page.messages) {
            assert_eq!(a.role, b.role);
            assert_eq!(a.seq, b.seq, "seqs copy in order");
            assert_eq!(a.parts, b.parts, "parts copy verbatim");
        }
        assert_eq!(fork.proposed_message_seq().unwrap(), 3);
        // The fork is independent: new rows on the source never appear.
        let mid3 = s
            .put_message(3, "user", serde_json::json!({"text": "later"}))
            .unwrap();
        s.put_text_part(mid3, "later text").unwrap();
        assert_eq!(fork.message_count().unwrap(), 2);
        // Unknown source sessions are not found.
        assert!(m
            .fork_session(kilop_core::id::SessionId::new(9999), "x")
            .is_err());
    }

    #[test]
    fn fork_survives_reopen_with_durable_rows() {
        let dir = tempfile::tempdir().unwrap();
        let (sid, fork_id) = {
            let m = SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                .unwrap();
            let ws = m.create_workspace("/w").unwrap();
            let s = m.create_session(ws, "t", "p", "m").unwrap();
            s.put_message(1, "user", serde_json::json!({"text": "a"}))
                .unwrap();
            (s.id(), m.fork_session(s.id(), "t (fork)").unwrap().id())
        };
        let m =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let src = m.get_session(sid).unwrap().unwrap();
        let fork = m.get_session(fork_id).unwrap().unwrap();
        assert_eq!(src.message_count().unwrap(), 1);
        assert_eq!(fork.message_count().unwrap(), 1);
        assert_eq!(
            fork.messages_page(None, 10).unwrap().messages[0]
                .parts
                .len(),
            0,
            "the durable copy carries the same (part-less) row"
        );
    }

    #[test]
    fn delete_session_refuses_mid_turn_and_ends_durably() {
        let dir = tempfile::tempdir().unwrap();
        let m =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let ws = m.create_workspace("/w").unwrap();
        let busy = m.create_session(ws, "busy", "p", "m").unwrap();
        busy.submit_prompt("first", &[]).unwrap();
        assert!(busy.state().unwrap().is_active());
        let err = m.delete_session(busy.id()).unwrap_err();
        assert_eq!(err.kind, kilop_core::error::ErrorKind::Conflict);
        assert!(err.message.contains("mid-turn"), "{}", err.message);
        assert!(
            !busy.state().unwrap().is_terminal(),
            "refused delete leaves no trace"
        );
        // The state machine can still be cancelled/ended afterwards.
        busy.append_event(
            kilop_core::event::EventKind::RecoveryApplied,
            kilop_core::state::AgentState::Idle,
            None,
            None,
        )
        .unwrap_err(); // Preparing may not jump to Idle
        let _ = busy.abort(None).unwrap();

        // An idle session deletes: durable Closed + resources dropped.
        let s = m.create_session(ws, "gone", "p", "m").unwrap();
        s.put_message(1, "user", serde_json::json!({"text": "x"}))
            .unwrap();
        m.delete_session(s.id()).unwrap();
        let row = s.row().unwrap();
        assert!(row.lifecycle.is_terminal());
        assert_eq!(row.state, kilop_core::state::AgentState::Completed);
        // Prompts are refused on the deleted session.
        assert!(s.submit_prompt("nope", &[]).is_err());
        // Double delete conflicts.
        assert!(m.delete_session(s.id()).is_err());
        // Unknown session → not found.
        assert_eq!(
            m.delete_session(kilop_core::id::SessionId::new(9999))
                .unwrap_err()
                .kind,
            kilop_core::error::ErrorKind::NotFound
        );
        // The tombstone survives a manager reopen.
        drop(m);
        let m2 =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let row = m2.get_session(s.id()).unwrap().unwrap().row().unwrap();
        assert!(row.lifecycle.is_terminal(), "Closed survives reopen");
    }
}
