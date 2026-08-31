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
    store: Store,
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
        let store = Store::open(root, integrity_check).map_err(SessionError::from)?;
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

    pub(crate) fn store(&self) -> &Store {
        &self.store
    }

    pub(crate) fn cas(&self) -> &Cas {
        &self.cas
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
        let diagnostics = self
            .store()
            .diagnostics()
            .map_err(crate::map_store_err)?;
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
            return Err(SessionError::Malformed("worktree path and branch must be non-empty".into())
                .into());
        }
        self.store
            .put_worktree(ws, path, branch)
            .map_err(|e| crate::map_store_err(e).into())
    }

    pub fn worktrees_of(&self, ws: WorkspaceId) -> kilop_core::Result<Vec<kilop_store::WorktreeRow>> {
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

    pub fn get_session(self: &Arc<Self>, id: SessionId) -> kilop_core::Result<Option<SessionHandle>> {
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
        let rows = self
            .store
            .list_sessions(ws)
            .map_err(crate::map_store_err)?;
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_manager() -> (tempfile::TempDir, Arc<SessionManager>) {
        let dir = tempfile::tempdir().unwrap();
        let m = SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
            .unwrap();
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
}
