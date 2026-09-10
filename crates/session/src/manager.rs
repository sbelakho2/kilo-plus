//! The session manager: opens the durable store + CAS, owns the per-session
//! resource registries, and is the factory for `SessionHandle`s.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

use faktor_cas::Cas;
use faktor_core::id::{OpId, SessionId, TaskId, WorkspaceId};
use faktor_core::time::{Clock, SystemClock};
use faktor_store::Store;

use crate::artifacts::ArtifactSizes;
use crate::budget::BudgetAuthority as _;
use crate::handle::SessionHandle;
use crate::ops::OpRegistry;
use crate::process::ProcessRegistry;
use crate::read_service::DbReadKind;
use crate::recovery::SystemFileHasher;
use crate::{SessionError, DEFAULT_TURN_BUDGET_MS};

// ------------------------------------------------------------- shadow roots
//
// P0-48 shadow mutation registry (additive): ONE durable "active shadow" row
// per session, stored in the session's memory-fact space under kind
// [`SHADOW_ROW_KIND`] / key [`SHADOW_ROW_KEY`]. The row survives manager
// reopens, so a crashed daemon's shadow directories stay discoverable and
// cleanable (zero-orphan recovery). The row is written by the daemon's
// `ShadowRoots` service (orchestrator crate); this module only defines the
// shape and the re-pointing query [`SessionManager::active_root`], which
// root-resolving consumers consult in preference over the stored workspace
// root while a session runs shadowed.

/// Durable row kind of one session's active shadow.
pub const SHADOW_ROW_KIND: &str = "shadow_root";
/// Durable row key (one active shadow per session).
pub const SHADOW_ROW_KEY: &str = "active";
/// Bound of one stored shadow root path (rows must stay far below the
/// memory-fact value cap of 4096 bytes).
pub const SHADOW_PATH_MAX_BYTES: usize = 1024;
/// Bound of one stored shadow id.
pub const SHADOW_ID_MAX_BYTES: usize = 64;

/// Lifecycle of one shadow (written by the ShadowRoots service; the session
/// crate stores and reports it opaquely so reopen-safe recovery sees a typed
/// state, never a guessed one).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShadowRowState {
    /// The shadow exists on disk and may be the mutation target of a live
    /// or crashed drive.
    Active,
    /// A prior integration attempt surfaced conflicts; the shadow is
    /// retained and the durable conflict list must be resolved first.
    IntegrationBlocked,
    /// The shadow was integrated (or is a no-op); its directory is gone.
    Integrated,
    /// The shadow was discarded; its directory is gone.
    Discarded,
}

impl ShadowRowState {
    /// States in which the shadow root is a LIVE mutation/integration
    /// target ([`SessionManager::active_root`] reports it).
    pub fn is_live(self) -> bool {
        matches!(
            self,
            ShadowRowState::Active | ShadowRowState::IntegrationBlocked
        )
    }
}

/// The durable active-shadow row of ONE session (P0-48). Paths are stored as
/// strings (serde-friendly); every bound is enforced at write time.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowRow {
    pub session_id: u64,
    pub shadow_id: String,
    /// The USER checkout the shadow was copied from (the integration
    /// target of `commit_back`).
    pub base_root: String,
    /// The shadow's own root (daemon data dir; never inside the checkout).
    pub root: String,
    pub state: ShadowRowState,
    /// Files/bytes of the base copy (bounded summary of the base manifest).
    pub base_entries: u64,
    pub base_bytes: u64,
    pub created_ms: i64,
}

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

/// The in-memory half of the durable op-id allocator: one reserved range
/// `[next, next + remaining)` from the store's global sequence, handed out
/// one id at a time. `remaining == 0` triggers the next store reservation.
/// Ids are NEVER minted from the clock or a bare process counter, so
/// restart-in-the-same-millisecond and backward-clock jumps cannot reuse an
/// id; a crash only wastes the unreserved tail of the cached range (a gap,
/// never a duplicate).
#[derive(Debug, Default)]
struct OpIdRange {
    next: u64,
    remaining: u64,
}

/// The entry point of `faktor-session`. One manager per daemon data root.
pub struct SessionManager {
    store: Arc<Store>,
    cas: Arc<Cas>,
    clock: Arc<dyn Clock>,
    /// The async append actor over the SAME `Arc<Store>` (audit 42): hot
    /// append paths run through it; every other call stays direct sync.
    actor: Arc<crate::actor::DbActor>,
    /// The bounded async SQLite READ pool (audit 13): the async wrappers
    /// below submit the turn-path reads here, so their SQLite I/O + JSON
    /// decode never run on a Tokio worker. Lazy: no thread exists until the
    /// first bounded read is awaited.
    reads: crate::read_service::DbReadService,
    op_ids: Mutex<OpIdRange>,
    resources: Mutex<HashMap<SessionId, Arc<SessionResources>>>,
    system_hasher: Arc<SystemFileHasher>,
    pub(crate) artifact_sizes: ArtifactSizes,
    /// Layered lifetime bound (audit 26): wall-clock budget of ONE logical
    /// turn — the prompt operation's deadline and the runtime's per-turn
    /// slice ceiling. Default [`DEFAULT_TURN_BUDGET_MS`] (30 min), NOT 24h;
    /// the TASK's lifetime is bounded by its budget, never by a single
    /// future. `0` disables the wall-clock turn bound.
    turn_budget_ms: AtomicU64,
}

impl std::fmt::Debug for SessionManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionManager").finish_non_exhaustive()
    }
}

impl SessionManager {
    /// Open (creating if needed) the SQLite store at `root` and the CAS at
    /// `cas_root`. `integrity_check: true` refuses to open a corrupt store
    /// (full `PRAGMA integrity_check` scan).
    pub fn open(
        root: impl Into<PathBuf>,
        cas_root: impl Into<PathBuf>,
        integrity_check: bool,
    ) -> faktor_core::Result<Arc<SessionManager>> {
        Self::open_with_clock(root, cas_root, integrity_check, Arc::new(SystemClock))
    }

    /// Same as [`SessionManager::open`] with an injectable clock (tests).
    pub fn open_with_clock(
        root: impl Into<PathBuf>,
        cas_root: impl Into<PathBuf>,
        integrity_check: bool,
        clock: Arc<dyn Clock>,
    ) -> faktor_core::Result<Arc<SessionManager>> {
        let store = Arc::new(Store::open(root, integrity_check).map_err(SessionError::from)?);
        Self::assemble(store, cas_root, clock)
    }

    /// Fast normal-start open (production `serve`, plain `doctor`): the
    /// bounded quick path — WAL recovery, migrations and `PRAGMA
    /// quick_check` — never the full integrity scan. The deep scan belongs
    /// to `doctor --deep` and crash forensics. Purely additive: the
    /// full-check [`SessionManager::open`] keeps its behavior for tests and
    /// tooling.
    pub fn open_quick(
        root: impl Into<PathBuf>,
        cas_root: impl Into<PathBuf>,
    ) -> faktor_core::Result<Arc<SessionManager>> {
        Self::open_quick_with_clock(root, cas_root, Arc::new(SystemClock))
    }

    /// Same as [`SessionManager::open_quick`] with an injectable clock.
    pub fn open_quick_with_clock(
        root: impl Into<PathBuf>,
        cas_root: impl Into<PathBuf>,
        clock: Arc<dyn Clock>,
    ) -> faktor_core::Result<Arc<SessionManager>> {
        let store = Arc::new(Store::open_fast(root).map_err(SessionError::from)?);
        Self::assemble(store, cas_root, clock)
    }

    /// The shared open tail: once the store is open (full or fast), the CAS,
    /// system hasher and per-session registries are wired identically.
    fn assemble(
        store: Arc<Store>,
        cas_root: impl Into<PathBuf>,
        clock: Arc<dyn Clock>,
    ) -> faktor_core::Result<Arc<SessionManager>> {
        let cas_root = cas_root.into();
        let cas = Cas::open(cas_root).map_err(SessionError::from)?;
        let cas = Arc::new(cas);
        let system_hasher = Arc::new(SystemFileHasher::new(cas.clone()));
        let actor = crate::actor::DbActor::spawn(store.clone(), Default::default());
        let reads = crate::read_service::DbReadService::spawn(store.clone(), Default::default());
        Ok(Arc::new(SessionManager {
            store,
            cas,
            clock,
            actor,
            reads,
            op_ids: Mutex::new(OpIdRange::default()),
            resources: Mutex::new(HashMap::new()),
            system_hasher,
            artifact_sizes: ArtifactSizes::default(),
            turn_budget_ms: AtomicU64::new(DEFAULT_TURN_BUDGET_MS),
        }))
    }

    /// Wall-clock budget of ONE logical turn in ms (see the field docs).
    pub fn set_turn_budget_ms(&self, ms: u64) {
        self.turn_budget_ms
            .store(ms, std::sync::atomic::Ordering::Relaxed);
    }

    /// The configured per-turn wall-clock budget; `0` means unbounded.
    pub fn turn_budget_ms(&self) -> u64 {
        self.turn_budget_ms
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The durable store, shared with snapshot/redo consumers (the wire
    /// revert/unrevert/diff surface builds its checkpoint store over the
    /// same `Arc<Store>` so both sides see the same rows).
    pub fn store(&self) -> Arc<Store> {
        self.store.clone()
    }

    /// The async append actor over the SAME store (audit 42): hot append
    /// paths (message / part / journal event / usage settlement) run through
    /// it; all other store calls stay direct on [`SessionManager::store`].
    pub fn actor(&self) -> Arc<crate::actor::DbActor> {
        self.actor.clone()
    }

    /// Tune the actor's config (tests): takes effect when the actor starts
    /// or respawns.
    pub fn set_actor_config(&self, cfg: crate::actor::DbActorConfig) {
        self.actor.set_config_for_test(cfg);
    }

    /// The bounded async read pool (audit 13): the async wrappers below
    /// submit through it. SQLite I/O + JSON decode of the listed turn-path
    /// reads runs on the pool's 2-4 hard-capped worker threads — never on
    /// a Tokio worker, never via unrestricted `spawn_blocking`.
    pub fn read_service(&self) -> &crate::read_service::DbReadService {
        &self.reads
    }

    // ------------------------------------------------------- bounded reads
    // (audit 13/32) Async twins of the synchronous turn-path store reads the
    // agent runtime used to run inline on Tokio workers. Each wrapper
    // submits the IDENTICAL sync read to the bounded pool and awaits the
    // worker's result, so result parity with the sync calls is by
    // construction. A closed/drained pool (daemon shutdown) fails reads
    // fast with a typed error — never a hang and never an inline fallback
    // that would silently reintroduce worker-thread SQLite I/O.

    /// Bounded newest-first conversation window (audit 29 semantics of
    /// `SessionHandle::messages_backwards_bounded`), read off the pool.
    pub async fn messages_backwards_bounded(
        &self,
        session: SessionId,
        before: Option<u64>,
        max_messages: u64,
        max_bytes: u64,
    ) -> faktor_core::Result<Vec<faktor_store::MessageRow>> {
        self.reads
            .submit_tagged(DbReadKind::History, move |store| {
                store
                    .messages_backwards_bounded(session, before, max_messages, max_bytes)
                    .map_err(crate::map_store_err)
                    .map_err(Into::into)
            })
            .await?
    }

    /// One durable task row of `session` (`SessionHandle::get_task`
    /// semantics), read off the pool.
    pub async fn task(
        &self,
        session: SessionId,
        task_id: TaskId,
    ) -> faktor_core::Result<Option<crate::task::Task>> {
        self.reads
            .submit_tagged(DbReadKind::Task, move |store| {
                store
                    .get_task(session, task_id)
                    .map_err(crate::map_store_err)
                    .map(|row| row.map(crate::task::Task::from))
                    .map_err(Into::into)
            })
            .await?
    }

    /// The task's durable monetary picture (`DurableBudgetLedger::
    /// session_budget_view` semantics), read off the pool. A failed read is
    /// a typed [`crate::budget::BudgetError`] — the manager NEVER synthesizes
    /// an unlimited-zero view from a failure.
    pub async fn budget_view(
        self: &Arc<Self>,
        session: SessionId,
        task_id: TaskId,
    ) -> Result<crate::budget::BudgetView, crate::budget::BudgetError> {
        let manager = self.clone();
        self.reads
            .submit_tagged(DbReadKind::Budget, move |_store| {
                crate::budget::DurableBudgetLedger::new(manager)
                    .session_budget_view(session, task_id)
            })
            .await
            .map_err(|e| crate::budget::BudgetError::Unavailable {
                cap: crate::budget::BudgetCapEvidence::Unknown,
                reason: format!("budget read pool: {e}"),
            })?
    }

    /// The UI-facing budget state ([`crate::budget::BudgetState`]): an
    /// honest `Known` view or a typed `Unavailable { reason }` — never a
    /// synthesized unlimited/zero view.
    pub async fn budget_state(
        self: &Arc<Self>,
        session: SessionId,
        task_id: TaskId,
    ) -> crate::budget::BudgetState {
        crate::budget::BudgetState::from(self.budget_view(session, task_id).await)
    }

    /// Every durable verification record of `task_id`
    /// (`SessionHandle::list_verification_records` semantics — records are
    /// keyed by the numeric task id), read off the pool. The v20 evidence
    /// columns (environment fingerprint, candidate-proof reference) parse
    /// exactly like the handle's synchronous read, so both surfaces return
    /// byte-identical records.
    pub async fn verification_records(
        &self,
        _session: SessionId,
        task_id: TaskId,
    ) -> faktor_core::Result<Vec<crate::task::VerificationRecord>> {
        self.reads
            .submit_tagged(DbReadKind::Verification, move |store| {
                let rows = match store.verification_record_list_by_task_with_evidence(task_id) {
                    Ok(rows) => rows,
                    Err(e) => return Err(crate::task::TaskError::from(e).into()),
                };
                rows.into_iter()
                    .map(|(row, fingerprint_json, candidate_json)| {
                        crate::task::verification_record_from_row_with_evidence(
                            row,
                            fingerprint_json,
                            candidate_json,
                        )
                    })
                    .collect::<Result<Vec<_>, crate::task::TaskError>>()
                    .map_err(Into::into)
            })
            .await?
    }

    /// The session's durable prefix observations, oldest call first
    /// (`Store::provider_call_prefix_rows` — the consult feed of cache
    /// economics), read off the pool.
    pub async fn provider_prefix_history(
        &self,
        session: SessionId,
    ) -> faktor_core::Result<Vec<faktor_store::ProviderCallPrefixRow>> {
        self.reads
            .submit_tagged(DbReadKind::Prefix, move |store| {
                store
                    .provider_call_prefix_rows(session)
                    .map_err(crate::map_store_err)
                    .map_err(Into::into)
            })
            .await?
    }

    /// One deterministic page of memory facts (`SessionHandle::
    /// memory_facts_page` semantics: bounded page + probe + exact count),
    /// read off the pool.
    pub async fn memory_page(
        &self,
        session: SessionId,
        after: Option<(i64, String, String)>,
        limit: i64,
    ) -> faktor_core::Result<crate::memory::MemoryFactsPage> {
        self.reads
            .submit_tagged(DbReadKind::Memory, move |store| {
                let limit = limit.clamp(1, crate::memory::MAX_FACT_PAGE_SIZE);
                let (rows, has_more) = store
                    .memory_facts_page(session, after.as_ref(), limit as u64)
                    .map_err(crate::map_store_err)?;
                let cursor = if has_more {
                    rows.last()
                        .map(|r| (r.updated_ms, r.kind.clone(), r.key.clone()))
                } else {
                    None
                };
                let total_estimate = store
                    .memory_fact_count(session)
                    .map_err(crate::map_store_err)?;
                Ok(crate::memory::MemoryFactsPage {
                    facts: rows.into_iter().map(|r| (r.kind, r.key, r.value)).collect(),
                    size: limit,
                    cursor,
                    has_more,
                    total_estimate,
                })
            })
            .await?
    }

    /// The content-addressed blob store, shared with snapshot consumers.
    pub fn cas(&self) -> Arc<Cas> {
        self.cas.clone()
    }

    /// Current wall-clock time (milliseconds since the epoch).
    pub fn now_ms(&self) -> i64 {
        self.clock.now_ms()
    }

    /// A fresh, non-zero, PROCESS-INDEPENDENT operation id. Ids come from
    /// the store's durable global sequence in reserved ranges of
    /// [`OP_ID_RANGE`], so they stay unique and strictly increasing ACROSS
    /// daemon restarts — even when a restart lands in the same millisecond
    /// or the wall clock jumped backwards. Zero is contractually never
    /// returned.
    pub fn next_op_id(&self) -> OpId {
        let mut cache = self.op_ids.lock().expect("op id cache poisoned");
        if cache.remaining == 0 {
            // The sequence is GLOBAL (one store row shared by every
            // session), so no session id is semantically meaningful here;
            // the placeholder only satisfies the allocator's future
            // per-session-scope signature. A refill failure means the
            // durable sequence cannot advance — minting from the clock
            // would reintroduce exactly the collision this fixes, so a
            // loud abort is the only safe behavior.
            let (start, granted) = self
                .store
                .alloc_op_ids(SessionId::new(1), OP_ID_RANGE)
                .unwrap_or_else(|e| panic!("op-id sequence refill failed: {e}"));
            cache.next = start;
            cache.remaining = granted;
        }
        let raw = cache.next;
        cache.next += 1;
        cache.remaining -= 1;
        OpId::new(raw)
    }

    /// `doctor`-style health report: store diagnostics + recovery scan.
    pub fn integrity_report(&self) -> faktor_core::Result<serde_json::Value> {
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

    pub fn create_workspace(&self, root: &str) -> faktor_core::Result<WorkspaceId> {
        self.store
            .create_workspace(root)
            .map_err(|e| crate::map_store_err(e).into())
    }

    /// The DURABLE filesystem root of `ws` (P0-32): the feed for
    /// per-workspace repository instructions. `None` when the workspace row
    /// does not exist or carries no root; a store failure is a typed error
    /// — never a guessed root, never the process CWD.
    pub fn workspace_root(&self, ws: WorkspaceId) -> faktor_core::Result<Option<PathBuf>> {
        self.store
            .workspace_root(ws)
            .map(|r| r.map(PathBuf::from))
            .map_err(|e| crate::map_store_err(e).into())
    }

    // --------------------------------------------------- shadow registry (P0-48)

    /// Read the session's durable active-shadow row (kind
    /// [`SHADOW_ROW_KIND`]). `Ok(None)` when no shadow was ever begun on
    /// this session; a hostile stored value is a loud error, never a
    /// guessed row. Reopen-safe: the row is a store fact, not memory.
    pub fn shadow_row(&self, session: SessionId) -> faktor_core::Result<Option<ShadowRow>> {
        let facts = self
            .store
            .memory_facts(session)
            .map_err(|e| -> faktor_core::Error { crate::map_store_err(e).into() })?;
        for (kind, key, value) in facts {
            if kind == SHADOW_ROW_KIND && key == SHADOW_ROW_KEY {
                let row: ShadowRow = serde_json::from_str(&value).map_err(|e| {
                    faktor_core::Error::internal(format!(
                        "shadow row of session {session} is corrupt: {e}"
                    ))
                })?;
                if row.session_id != session.raw() {
                    return Err(faktor_core::Error::internal(format!(
                        "shadow row of session {session} names session {}",
                        row.session_id
                    )));
                }
                return Ok(Some(row));
            }
        }
        Ok(None)
    }

    /// Write (upsert) the session's active-shadow row. Every stored field is
    /// bounded before the write (typed Oversized otherwise); an unknown
    /// session still stores the row (the row is the recovery record — the
    /// service validates session existence at begin).
    pub fn put_shadow_row(&self, session: SessionId, row: &ShadowRow) -> faktor_core::Result<()> {
        if row.shadow_id.is_empty() || row.shadow_id.len() > SHADOW_ID_MAX_BYTES {
            return Err(SessionError::Oversized(format!(
                "shadow id must be 1..={SHADOW_ID_MAX_BYTES} bytes"
            ))
            .into());
        }
        if row.base_root.is_empty()
            || row.base_root.len() > SHADOW_PATH_MAX_BYTES
            || row.root.is_empty()
            || row.root.len() > SHADOW_PATH_MAX_BYTES
        {
            return Err(
                SessionError::Oversized("shadow paths must be 1..=1024 bytes".into()).into(),
            );
        }
        let mut stored = row.clone();
        stored.session_id = session.raw();
        let value = serde_json::to_string(&stored)
            .map_err(|e| SessionError::Internal(format!("shadow row serialization: {e}")))?;
        if value.len() > 4000 {
            return Err(SessionError::Oversized(format!(
                "shadow row of {} bytes exceeds the 4096-byte fact cap",
                value.len()
            ))
            .into());
        }
        self.store
            .upsert_memory_fact(session, SHADOW_ROW_KIND, SHADOW_ROW_KEY, &value)
            .map_err(|e| -> faktor_core::Error { crate::map_store_err(e).into() })
    }

    /// Root re-pointing query (P0-48): the workspace root a shadowed session
    /// must resolve files against. `Some(shadow root)` when the session
    /// carries a durable shadow row in a LIVE state (Active /
    /// IntegrationBlocked) — preferred over the stored workspace root by
    /// every consumer that resolves a session's files (tool contexts,
    /// instructions resolver, evidence/index). `None` = the session runs
    /// un-shadowed and consumers use the stored root as today. The
    /// filesystem presence of the returned root is NOT re-verified here
    /// (openers do that); a row whose directory was cleaned up transitions
    /// through the service's reconcile pass.
    pub fn active_root(&self, session: SessionId) -> faktor_core::Result<Option<PathBuf>> {
        Ok(self
            .shadow_row(session)?
            .filter(|row| row.state.is_live())
            .map(|row| PathBuf::from(row.root)))
    }

    /// The ONE root every session-scoped filesystem consumer resolves
    /// against (P0-48 root re-pointing): `Some(live shadow root)` while the
    /// session carries a durable shadow row in a LIVE state
    /// ([`SessionManager::active_root`] — the shadow mutation target), else
    /// the durable root of the session's workspace row (today's behavior,
    /// byte-identical). `Ok(None)` when the session or its workspace row is
    /// unknown — consumers keep their documented empty/missing-root
    /// semantics. A corrupt shadow row or a store failure is a loud error,
    /// never a guessed root.
    ///
    /// The five agent-crate root consumers (tool batches, write-verify,
    /// replay, turn verification, repo knowledge) and the session-scoped
    /// daemon consumers (evidence, snapshot revert targets) all resolve
    /// through this ONE helper, so a shadowed drive re-points every file
    /// consumer at the same root.
    pub fn resolve_workspace_root(
        &self,
        session: SessionId,
    ) -> faktor_core::Result<Option<PathBuf>> {
        if let Some(shadow) = self.active_root(session)? {
            return Ok(Some(shadow));
        }
        match self
            .store
            .get_session(session)
            .map_err(crate::map_store_err)?
        {
            Some(row) => self.workspace_root(row.workspace_id),
            None => Ok(None),
        }
    }

    /// The live shadow root re-pointing a WORKSPACE-scoped consumer (the
    /// per-workspace instruction resolver, the index attach): `Some(root)`
    /// exactly when ONE session of the workspace carries a live shadow row
    /// — the daemon's single-shadowed-drive shape. `Ok(None)` with no live
    /// shadow (the stored workspace root stays authoritative). MORE than
    /// one live shadow on one workspace (never produced by the executor's
    /// per-session begin/settle discipline; only hostile/crash residue) is
    /// a loud error — the caller degrades to the stored root, never a
    /// guessed one. Reopen-safe: everything is read from durable rows.
    pub fn live_workspace_shadow_root(
        &self,
        ws: WorkspaceId,
    ) -> faktor_core::Result<Option<PathBuf>> {
        let sessions = self
            .store
            .list_sessions(Some(ws))
            .map_err(crate::map_store_err)?;
        let mut found: Option<PathBuf> = None;
        for row in sessions {
            let shadow = self.active_root(row.id)?;
            if shadow.is_some() {
                if found.is_some() {
                    return Err(faktor_core::Error::new(
                        faktor_core::ErrorKind::Conflict,
                        format!(
                            "workspace {ws} carries more than one live shadow; deterministic root resolution is impossible"
                        ),
                    ));
                }
                found = shadow;
            }
        }
        Ok(found)
    }

    // ---------------------------------------------------------------- worktrees

    pub fn put_worktree(
        &self,
        ws: WorkspaceId,
        path: &str,
        branch: &str,
    ) -> faktor_core::Result<i64> {
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
    ) -> faktor_core::Result<Vec<faktor_store::WorktreeRow>> {
        self.store
            .worktrees_of(ws)
            .map_err(|e| crate::map_store_err(e).into())
    }

    pub fn remove_worktree(&self, path: &str) -> faktor_core::Result<()> {
        self.store
            .remove_worktree(path)
            .map_err(|e| crate::map_store_err(e).into())
    }

    // ---------------------------------------------------------------- sessions

    #[allow(clippy::too_many_arguments)]
    pub fn create_child_session(
        self: &Arc<Self>,
        parent: SessionId,
        ws: WorkspaceId,
        worktree_id: faktor_core::WorktreeId,
        task_id: faktor_core::TaskId,
        provider: &str,
        model: &str,
        title: &str,
        ownership: crate::child::ChildOwnership,
    ) -> faktor_core::Result<SessionHandle> {
        if self
            .store
            .get_session(parent)
            .map_err(crate::map_store_err)?
            .is_none()
        {
            return Err(SessionError::NotFound(format!(
                "parent session {parent} of the requested child does not exist"
            ))
            .into());
        }
        let child = self.create_session(ws, title, provider, model)?;
        self.adopt_identity(child.id(), worktree_id, task_id)?;
        let id = crate::child::ChildIdentity {
            parent_session_id: parent,
            workspace_id: ws.raw(),
            worktree_id: worktree_id.raw(),
            item_id: String::new(),
            task_goal: title.to_string(),
            operation_id: 0,
            ownership,
            model: String::new(),
            created_ms: self.now_ms(),
        };
        child.orchestrator_child_identity_put(&id)?;
        Ok(child)
    }

    /// Create a session; returns a handle wired to the manager's shared
    /// per-session resources.
    ///
    /// The session's WORKTREE/TASK identity defaults to the documented
    /// standalone 1/1 (no worktree or task adopted). WorktreeManager-created
    /// worktrees adopt their sessions with [`SessionManager::adopt_identity`]
    /// so tool calls and crash recovery carry the REAL worktree/task ids.
    pub fn create_session(
        self: &Arc<Self>,
        ws: WorkspaceId,
        title: &str,
        provider: &str,
        model: &str,
    ) -> faktor_core::Result<SessionHandle> {
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

    /// Durably adopt a worktree/task identity for an EXISTING session
    /// (v8): the session row then carries the real ids, and every
    /// subsequent tool call builds its `WorkspaceIdentity` from the row.
    /// The worktree/task ids must be non-zero (the typed constructors
    /// enforce that); the session must exist (loud otherwise). This is
    /// identity bookkeeping, not a turn transition: the journal is
    /// intentionally untouched.
    pub fn adopt_identity(
        self: &Arc<Self>,
        session: SessionId,
        worktree_id: faktor_core::WorktreeId,
        task_id: faktor_core::TaskId,
    ) -> faktor_core::Result<()> {
        self.store
            .adopt_session_identity(session, worktree_id, task_id)
            .map_err(crate::map_store_err)?;
        Ok(())
    }

    pub fn get_session(
        self: &Arc<Self>,
        id: SessionId,
    ) -> faktor_core::Result<Option<SessionHandle>> {
        match self.store.get_session(id).map_err(crate::map_store_err)? {
            Some(_row) => {
                let handle = SessionHandle::new(
                    self.clone(),
                    id,
                    self.resources(id),
                    self.system_hasher.clone(),
                );
                // Open-time typed-ledger verification (audit 27): an entry
                // that fails its schema decode fails the session open
                // loudly — never a silent drop.
                handle.ledger_verify_open()?;
                Ok(Some(handle))
            }
            None => Ok(None),
        }
    }

    pub fn list_sessions(
        self: &Arc<Self>,
        ws: Option<WorkspaceId>,
    ) -> faktor_core::Result<Vec<SessionHandle>> {
        let rows = self.store.list_sessions(ws).map_err(crate::map_store_err)?;
        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            let handle = SessionHandle::new(
                self.clone(),
                r.id,
                self.resources(r.id),
                self.system_hasher.clone(),
            );
            // Open-time typed-ledger verification (audit 27): a corrupt
            // typed entry fails the session open loudly.
            handle.ledger_verify_open()?;
            out.push(handle);
        }
        Ok(out)
    }

    /// Crash-recovery sweep over every session: each handle's journal is
    /// reconstructed and unfinished tool runs are resolved. Idempotent.
    pub fn recover_all_sessions(
        self: &Arc<Self>,
    ) -> faktor_core::Result<Vec<crate::recovery::RecoveryReport>> {
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
    ) -> faktor_core::Result<SessionHandle> {
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
        let mut newest_first: Vec<faktor_store::MessageRow> = Vec::new();
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
    pub fn delete_session(self: &Arc<Self>, id: SessionId) -> faktor_core::Result<()> {
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
        if row.state == faktor_core::state::AgentState::FailedRecoverable {
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

/// Ids reserved from the durable sequence per refill. Matches the store's
/// seed alignment (1024), so every refill starts on a quantum boundary.
/// A restart wastes at most `OP_ID_RANGE - 1` unreserved ids — a gap, never
/// a collision.
const OP_ID_RANGE: u64 = 1024;

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
    fn shadow_registry_row_roundtrips_bounds_and_reopen() {
        // P0-48: the active-shadow row is a durable session fact (survives a
        // full reopen), hostile/oversized values are rejected before the
        // write, and active_root reports the shadow root ONLY while the row
        // is in a live state (preferred over the stored workspace root).
        let dir = tempfile::tempdir().unwrap();
        let (_m, ws, session, row) = {
            let m = SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                .unwrap();
            let ws = m.create_workspace("/user/checkout").unwrap();
            let session = m.create_session(ws, "shadowed", "p", "m").unwrap().id();
            let row = ShadowRow {
                session_id: session.raw(),
                shadow_id: "sh-0000000000000001".into(),
                base_root: "/user/checkout".into(),
                root: "/data/shadows/1/sh-0000000000000001".into(),
                state: ShadowRowState::Active,
                base_entries: 3,
                base_bytes: 42,
                created_ms: 1,
            };
            m.put_shadow_row(session, &row).unwrap();
            // Read-back is exact.
            assert_eq!(m.shadow_row(session).unwrap(), Some(row.clone()));
            // active_root reports the live shadow root — preferred over the
            // stored workspace root.
            assert_eq!(
                m.active_root(session).unwrap(),
                Some(PathBuf::from("/data/shadows/1/sh-0000000000000001"))
            );
            (m, ws, session, row)
        };
        // Reopen: the row is a store fact, never memory.
        let m =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let stored = m.shadow_row(session).unwrap().expect("row survives reopen");
        assert_eq!(stored, row);
        assert_eq!(
            m.active_root(session).unwrap(),
            Some(PathBuf::from("/data/shadows/1/sh-0000000000000001"))
        );
        assert_eq!(
            m.workspace_root(ws).unwrap(),
            Some(PathBuf::from("/user/checkout"))
        );
        // A non-live row (integrated/discarded) stops re-pointing.
        let mut done = stored.clone();
        done.state = ShadowRowState::Integrated;
        m.put_shadow_row(session, &done).unwrap();
        assert_eq!(m.active_root(session).unwrap(), None);
        assert_eq!(
            m.shadow_row(session).unwrap().unwrap().state,
            ShadowRowState::Integrated
        );
        // Hostile/bound-broken rows are refused before the write.
        for bad in [
            ShadowRow {
                shadow_id: "".into(),
                ..row.clone()
            },
            ShadowRow {
                shadow_id: "x".repeat(SHADOW_ID_MAX_BYTES + 1),
                ..row.clone()
            },
            ShadowRow {
                base_root: "".into(),
                ..row.clone()
            },
            ShadowRow {
                root: "x".repeat(SHADOW_PATH_MAX_BYTES + 1),
                ..row.clone()
            },
        ] {
            assert!(m.put_shadow_row(session, &bad).is_err(), "{bad:?}");
        }
        // A hostile stored value is a loud error, never a guessed row.
        m.store()
            .upsert_memory_fact(
                session,
                SHADOW_ROW_KIND,
                SHADOW_ROW_KEY,
                "{not a shadow row",
            )
            .unwrap();
        let err = m.shadow_row(session).unwrap_err();
        assert!(err.message.contains("corrupt"), "{err}");
        assert!(m.active_root(session).is_err());
    }

    #[test]
    fn resolve_workspace_root_repoints_only_while_a_shadow_is_live() {
        // P0-48 root re-pointing: resolve_workspace_root returns the LIVE
        // shadow root of the session when one exists, else the stored
        // workspace root byte-identically — a plain session (shadow only
        // reachable through the TaskExecutor by construction) never reports
        // a shadow, and a retired shadow stops re-pointing.
        let dir = tempfile::tempdir().unwrap();
        let (_m, ws, session) = {
            let m = SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                .unwrap();
            let ws = m.create_workspace("/user/checkout").unwrap();
            let session = m.create_session(ws, "shadowed", "p", "m").unwrap().id();
            // (f): a direct session drive carries no shadow row — active_root
            // is None and resolve_workspace_root is the stored root, exactly
            // today's behavior.
            assert_eq!(m.active_root(session).unwrap(), None);
            assert_eq!(
                m.resolve_workspace_root(session).unwrap(),
                Some(PathBuf::from("/user/checkout"))
            );
            // Unknown sessions: None, never a guessed root.
            assert_eq!(
                m.resolve_workspace_root(SessionId::new(9999)).unwrap(),
                None
            );
            let row = ShadowRow {
                session_id: session.raw(),
                shadow_id: "sh-0000000000000001".into(),
                base_root: "/user/checkout".into(),
                root: "/data/shadows/1/sh-0000000000000001".into(),
                state: ShadowRowState::Active,
                base_entries: 3,
                base_bytes: 42,
                created_ms: 1,
            };
            m.put_shadow_row(session, &row).unwrap();
            assert_eq!(
                m.resolve_workspace_root(session).unwrap(),
                Some(PathBuf::from("/data/shadows/1/sh-0000000000000001")),
                "the live shadow overrides the stored root"
            );
            assert_eq!(
                m.resolve_workspace_root(SessionId::new(9999)).unwrap(),
                None,
                "an unknown session resolves nothing even with the shadow row live"
            );
            // A retired shadow (Integrated) stops re-pointing: the stored
            // root is authoritative again.
            let mut done = row.clone();
            done.state = ShadowRowState::Integrated;
            m.put_shadow_row(session, &done).unwrap();
            assert_eq!(
                m.resolve_workspace_root(session).unwrap(),
                Some(PathBuf::from("/user/checkout"))
            );
            // IntegrationBlocked keeps re-pointing (the shadow is retained
            // and remains the mutation target until the conflict resolves).
            let mut blocked = row.clone();
            blocked.state = ShadowRowState::IntegrationBlocked;
            m.put_shadow_row(session, &blocked).unwrap();
            assert_eq!(
                m.resolve_workspace_root(session).unwrap(),
                Some(PathBuf::from("/data/shadows/1/sh-0000000000000001"))
            );
            // A corrupt shadow row is a loud error, never a guessed root.
            m.store()
                .upsert_memory_fact(
                    session,
                    SHADOW_ROW_KIND,
                    SHADOW_ROW_KEY,
                    "{not a shadow row",
                )
                .unwrap();
            assert!(m.resolve_workspace_root(session).is_err());
            m.put_shadow_row(session, &row).unwrap();
            (m, ws, session)
        };
        // The row survives a full reopen: re-pointing is durable.
        let m =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        assert_eq!(
            m.resolve_workspace_root(session).unwrap(),
            Some(PathBuf::from("/data/shadows/1/sh-0000000000000001"))
        );
        assert_eq!(
            m.workspace_root(ws).unwrap(),
            Some(PathBuf::from("/user/checkout")),
            "the workspace row itself never moves"
        );
    }

    #[test]
    fn live_workspace_shadow_root_resolves_only_a_single_live_shadow() {
        // Workspace-scoped re-pointing (instruction resolver / index
        // attach): exactly one session of the workspace with a live shadow
        // re-points the workspace; zero keeps the stored root; two live
        // shadows (hostile residue) are a loud error, never a guess.
        let (_d, m) = tmp_manager();
        let ws = m.create_workspace("/user/checkout").unwrap();
        let a = m.create_session(ws, "a", "p", "m").unwrap().id();
        assert_eq!(m.live_workspace_shadow_root(ws).unwrap(), None);
        assert_eq!(
            m.live_workspace_shadow_root(WorkspaceId::new(999)).unwrap(),
            None,
            "unknown workspaces have no live shadow and no error"
        );
        let row_of = |session: SessionId, root: &str, state: ShadowRowState| ShadowRow {
            session_id: session.raw(),
            shadow_id: format!("sh-{:016x}", session.raw()),
            base_root: "/user/checkout".into(),
            root: root.into(),
            state,
            base_entries: 1,
            base_bytes: 1,
            created_ms: 1,
        };
        // One live shadow of session `a`: the workspace re-points.
        m.put_shadow_row(
            a,
            &row_of(a, "/data/shadows/1/sh-a", ShadowRowState::Active),
        )
        .unwrap();
        assert_eq!(
            m.live_workspace_shadow_root(ws).unwrap(),
            Some(PathBuf::from("/data/shadows/1/sh-a"))
        );
        // A retired shadow on a second session does not create ambiguity.
        let b = m.create_session(ws, "b", "p", "m").unwrap().id();
        m.put_shadow_row(
            b,
            &row_of(b, "/data/shadows/1/sh-b", ShadowRowState::Discarded),
        )
        .unwrap();
        assert_eq!(
            m.live_workspace_shadow_root(ws).unwrap(),
            Some(PathBuf::from("/data/shadows/1/sh-a"))
        );
        // A second LIVE shadow is ambiguous: loud error, never a guess.
        m.put_shadow_row(
            b,
            &row_of(
                b,
                "/data/shadows/1/sh-b",
                ShadowRowState::IntegrationBlocked,
            ),
        )
        .unwrap();
        let err = m.live_workspace_shadow_root(ws).unwrap_err();
        assert!(err.message.contains("more than one live shadow"), "{err}");
        // Retire the first: the second (IntegrationBlocked) re-points alone.
        m.put_shadow_row(
            a,
            &row_of(a, "/data/shadows/1/sh-a", ShadowRowState::Integrated),
        )
        .unwrap();
        assert_eq!(
            m.live_workspace_shadow_root(ws).unwrap(),
            Some(PathBuf::from("/data/shadows/1/sh-b"))
        );
        // A corrupt row anywhere is a loud error.
        m.put_shadow_row(
            a,
            &row_of(a, "/data/shadows/1/sh-a", ShadowRowState::Active),
        )
        .unwrap();
        m.store()
            .upsert_memory_fact(a, SHADOW_ROW_KIND, SHADOW_ROW_KEY, "{broken")
            .unwrap();
        assert!(m.live_workspace_shadow_root(ws).is_err());
    }

    #[test]
    fn workspace_root_resolves_only_durable_rows() {
        // P0-32: root resolution reads the durable workspace table only —
        // an unknown workspace carries no root (None, never an error) and
        // the value survives a full reopen (it is a store row, not memory).
        let dir = tempfile::tempdir().unwrap();
        let ws = {
            let m = SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                .unwrap();
            let ws = m.create_workspace("/durable/root").unwrap();
            assert_eq!(
                m.workspace_root(ws).unwrap(),
                Some(PathBuf::from("/durable/root"))
            );
            // Unknown workspace: None, not an error and not a guessed root.
            assert_eq!(m.workspace_root(WorkspaceId::new(999)).unwrap(), None);
            ws
        };
        let m =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        assert_eq!(
            m.workspace_root(ws).unwrap(),
            Some(PathBuf::from("/durable/root")),
            "the durable root survives a reopen"
        );
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
    fn session_identity_defaults_standalone_and_adoption_is_durable() {
        // v8 identity plumbing: create_session keeps its signature and
        // defaults to the DOCUMENTED standalone worktree 1/task 1; adoption
        // moves the durable row onto a real worktree/task and survives a
        // full manager reopen.
        let dir = tempfile::tempdir().unwrap();
        let (_, ws) = {
            let m = SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                .unwrap();
            let ws = m.create_workspace("/root").unwrap();
            let s = m.create_session(ws, "t", "ollama", "qwen3.8").unwrap();
            assert_eq!(s.id().raw(), 1, "first session row id");
            // Standalone default (documented): 1/1 — never a fake identity.
            let identity = s.identity().unwrap();
            assert_eq!(
                identity,
                faktor_core::WorkspaceIdentity::new(
                    ws,
                    faktor_core::WorktreeId::new(1),
                    faktor_core::TaskId::new(1)
                )
            );
            assert_eq!(
                s.row().unwrap().worktree_id,
                faktor_core::WorktreeId::new(1),
                "durable row carries the standalone worktree default"
            );
            // Adoption persists durably on the row.
            m.adopt_identity(
                s.id(),
                faktor_core::WorktreeId::new(7),
                faktor_core::TaskId::new(9),
            )
            .unwrap();
            assert_eq!(
                s.identity().unwrap().worktree_id,
                faktor_core::WorktreeId::new(7)
            );
            assert_eq!(s.identity().unwrap().task_id, faktor_core::TaskId::new(9));
            // Adoption of a second session is independent (no cross-talk).
            let s2 = m.create_session(ws, "t2", "p", "m").unwrap();
            assert_eq!(
                s2.identity().unwrap().worktree_id,
                faktor_core::WorktreeId::new(1)
            );
            // Unknown sessions are loud, never silent.
            assert!(m
                .adopt_identity(
                    faktor_core::id::SessionId::new(9999),
                    faktor_core::WorktreeId::new(2),
                    faktor_core::TaskId::new(2)
                )
                .is_err());
            (m, ws)
        };
        // Reopen: the adopted ids are durable.
        let m =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let s = m
            .get_session(faktor_core::id::SessionId::new(1))
            .unwrap()
            .unwrap();
        assert_eq!(
            s.identity().unwrap().worktree_id,
            faktor_core::WorktreeId::new(7),
            "adoption survives reopen"
        );
        assert_eq!(s.identity().unwrap().task_id, faktor_core::TaskId::new(9));
        assert_eq!(s.identity().unwrap().workspace_id, ws);
        // list_sessions rows carry the same columns.
        let handles = m.list_sessions(Some(ws)).unwrap();
        assert_eq!(handles.len(), 2);
        assert!(handles
            .iter()
            .any(|h| h.identity().unwrap().worktree_id == faktor_core::WorktreeId::new(7)));
        drop(m);
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
            .fork_session(faktor_core::id::SessionId::new(9999), "x")
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
        assert_eq!(err.kind, faktor_core::error::ErrorKind::Conflict);
        assert!(err.message.contains("mid-turn"), "{}", err.message);
        assert!(
            !busy.state().unwrap().is_terminal(),
            "refused delete leaves no trace"
        );
        // The state machine can still be cancelled/ended afterwards.
        busy.append_event(
            faktor_core::event::EventKind::RecoveryApplied,
            faktor_core::state::AgentState::Idle,
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
        assert_eq!(row.state, faktor_core::state::AgentState::Completed);
        // Prompts are refused on the deleted session.
        assert!(s.submit_prompt("nope", &[]).is_err());
        // Double delete conflicts.
        assert!(m.delete_session(s.id()).is_err());
        // Unknown session → not found.
        assert_eq!(
            m.delete_session(faktor_core::id::SessionId::new(9999))
                .unwrap_err()
                .kind,
            faktor_core::error::ErrorKind::NotFound
        );
        // The tombstone survives a manager reopen.
        drop(m);
        let m2 =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let row = m2.get_session(s.id).unwrap().unwrap().row().unwrap();
        assert!(row.lifecycle.is_terminal(), "Closed survives reopen");
    }

    #[test]
    fn same_millisecond_restart_never_collides() {
        // The old scheme (now_ms + in-memory counter) collided when a
        // restart landed in the same millisecond or the clock ticked
        // backwards. The durable sequence is clock-independent: two managers
        // opened over the SAME store dir back-to-back (10 rounds to catch
        // any millisecond-boundary jitter) must mint disjoint ids, with the
        // second manager's ids strictly above the first's — no sleeps, no
        // clock freezing needed.
        let dir = tempfile::tempdir().unwrap();
        for _round in 0..10 {
            let first =
                SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                    .unwrap();
            let second =
                SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                    .unwrap();
            let a: Vec<u64> = (0..1000).map(|_| first.next_op_id().raw()).collect();
            let b: Vec<u64> = (0..1000).map(|_| second.next_op_id().raw()).collect();
            let mut seen = std::collections::HashSet::new();
            for raw in a.iter().chain(&b) {
                assert!(seen.insert(*raw), "op id {raw} reused across restarts");
            }
            assert!(
                b[0] > a[999],
                "second manager's ids must continue above the first's: {} vs {}",
                b[0],
                a[999]
            );
            drop(second);
            drop(first);
        }
    }

    #[test]
    fn backward_clock_jump_cannot_reuse_ids() {
        // Force the durable sequence far into the future — the equivalent of
        // the wall clock running years ahead before a regression — then
        // reopen: minted ids continue PAST the future mark instead of
        // wrapping back into the recently-used space.
        let dir = tempfile::tempdir().unwrap();
        let (jump_to, max_before) = {
            let m = SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                .unwrap();
            let before: Vec<u64> = (0..300).map(|_| m.next_op_id().raw()).collect();
            let hw = m.store().op_id_seq_high_water().unwrap();
            let (start, granted) = m
                .store()
                .alloc_op_ids(faktor_core::id::SessionId::new(1), 1u64 << 62)
                .unwrap();
            assert_eq!(start, hw, "the jump starts at the durable high-water");
            drop(m);
            (
                start + granted,
                *before.iter().max().expect("300 ids were minted"),
            )
        };
        let m =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let after: Vec<u64> = (0..1000).map(|_| m.next_op_id().raw()).collect();
        assert!(
            after.windows(2).all(|w| w[0] < w[1]),
            "ids stay strictly increasing after the jump"
        );
        assert_eq!(after[0], jump_to, "ids resume exactly past the jump");
        assert!(
            after[0] > max_before,
            "a backward clock can never collide with minted ids"
        );
    }

    #[test]
    fn concurrent_op_id_allocation_is_collision_free() {
        // 8 threads x 200 allocations through ONE manager: the cached range
        // refill is mutex-serialized and every id comes from the durable
        // sequence, so nothing is ever handed out twice.
        let (_d, m) = tmp_manager();
        let mut handles = Vec::new();
        for _ in 0..8 {
            let m = m.clone();
            handles.push(std::thread::spawn(move || {
                (0..200).map(|_| m.next_op_id().raw()).collect::<Vec<u64>>()
            }));
        }
        let mut seen = std::collections::HashSet::new();
        for h in handles {
            for raw in h.join().unwrap() {
                assert_ne!(raw, 0, "zero is contractually impossible");
                assert!(seen.insert(raw), "op id {raw} handed out twice");
            }
        }
        assert_eq!(seen.len(), 1600, "every allocation was distinct");
    }

    #[test]
    fn op_ids_remain_global_and_monotonic_across_sessions() {
        // The durable sequence is ONE global row shared by every session:
        // allocations interleaved across two live sessions stay unique and
        // strictly increasing. A per-session time+counter scheme would give
        // neither cross-session ordering nor cross-restart uniqueness.
        let (_d, m) = tmp_manager();
        let ws = m.create_workspace("/w").unwrap();
        let s1 = m.create_session(ws, "a", "p", "m").unwrap();
        let s2 = m.create_session(ws, "b", "p", "m").unwrap();
        let mut last = 0u64;
        let mut seen = std::collections::HashSet::new();
        for i in 0..4000 {
            let _alternating = if i % 2 == 0 { s1.id() } else { s2.id() };
            let raw = m.next_op_id().raw();
            assert!(raw > last, "ids strictly increase across sessions");
            last = raw;
            assert!(seen.insert(raw), "op id {raw} reused");
        }
    }
}
