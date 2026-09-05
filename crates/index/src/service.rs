//! The real repository [`IndexService`] (audits 30/64): a persistent,
//! generation-addressed, per-workspace repository index with a durable
//! [`WorkspaceIndexState`] machine in the store.
//!
//! # Never block the first prompt
//!
//! [`IndexService::view`] is instant: it returns the newest PUBLISHED
//! generation's content when one exists (`None` while a first build is in
//! flight or while persisted content is still reloading). The agent runtime
//! keeps the bounded evidence scan as the fallback until a `Ready`
//! generation exists for the workspace, so "first user prompt -> wait for a
//! complete repo indexing" is never the normal path.
//! [`IndexService::ensure_ready`] is the only blocking entry point (tests /
//! retry paths, always with an explicit deadline).
//!
//! # Off-lock generation swap
//!
//! A build scans the workspace into a private in-memory index, fsyncs it to
//! `<data_root>/scratch/<ws>/`, then publishes: the durable CAS
//! `Building{g} -> Ready{g}`, followed by ONE rename into
//! `<data_root>/generations/<ws>/gen-<g>.json`. Readers hold an immutable
//! generation snapshot (`Arc`); a read issued at generation `g` can never
//! observe a partial `g+1` — the file only ever appears complete and the
//! in-memory content swaps only after the publish. Exactly one builder wins
//! the publish CAS per generation; losers discard their scratch.
//!
//! # Restart and pruning
//!
//! `Ready{g}` restarts as `Ready{g}` (content reloaded from `gen-g.json`);
//! `Building`/`Dirty` resume building the SAME target; `Failed` waits for
//! the next change or an explicit retry. Publishing `N+1` prunes generation
//! files `<= N-1`, so at most [`KEEP_GENERATIONS`] generations are on disk.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(test)]
use std::sync::OnceLock;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use faktor_core::id::WorkspaceId;
use faktor_fs::atomic::fsync_parent;
use faktor_fs::{FsEventKind, WorkspaceFileService, WorkspaceHandle};
use faktor_store::Store;

use crate::generation::{FingerprintEntry, GenerationFile};
use crate::state::{
    PersistedIndexState, StateError, WorkspaceIndexState, JOURNAL_BUILDING, JOURNAL_CORRUPT,
    JOURNAL_DIRTY, JOURNAL_FAILED, JOURNAL_NOT_STARTED, JOURNAL_READY, JOURNAL_RESUME,
    JOURNAL_TORN_READY,
};
use crate::WorkspaceIndex;

/// Generations kept on disk: after `Ready{N}` publishes, everything
/// `<= N - KEEP_GENERATIONS` is pruned (so `N-1` and `N` remain — exactly
/// two generations once `N >= 2`).
pub const KEEP_GENERATIONS: u64 = 2;

/// Directories never walked by a build or fingerprint (vcs metadata,
/// dependency trees).
pub const SKIP_DIRS: &[&str] = &[
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "target",
    ".venv",
    "dist",
];

/// Content-scan caps: a workspace larger than this is PARTIALLY indexed
/// (deterministic walk order) — evidence stays bounded even for hostile
/// repos. Mirrors the bounded evidence scan's budget (cli `RepoEvidence`).
pub const SCAN_MAX_FILES: usize = 4_000;
pub const SCAN_MAX_DIRS: usize = 16_000;
pub const SCAN_MAX_BYTES: usize = 64 * 1024 * 1024;
pub const SCAN_MAX_FILE_BYTES: u64 = 1_000_000;

/// Stat-only fingerprint walk caps (looser than the content caps: changes
/// to big/binary files must still dirty the generation).
const FP_MAX_DIRS: usize = 16_000;
const FP_MAX_FILES: usize = 1_000_000;
const FP_PER_DIR: usize = 50_000;

/// Maximum generation-file bytes read back at load (a hostile or corrupt
/// file must not materialize unboundedly into RAM).
const MAX_GENERATION_FILE_BYTES: u64 = 512 * 1024 * 1024;

/// Tuning knobs of one service (see [`IndexService::set_config`]).
#[derive(Debug, Clone, Copy)]
pub struct ServiceConfig {
    /// Poll cadence of the background reconciliation worker.
    pub poll: Duration,
    /// Minimum interval between full fingerprint reconciliations of an
    /// idle ready workspace (heals lost watcher events).
    pub fingerprint_interval: Duration,
    /// Lease after which an in-process build is assumed dead (crash) and
    /// reclaimable by another builder.
    pub build_lease: Duration,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            poll: Duration::from_millis(200),
            fingerprint_interval: DEFAULT_FINGERPRINT_INTERVAL,
            build_lease: Duration::from_secs(10 * 60),
        }
    }
}

/// Typed service errors. Every error is advisory to the caller: the runtime
/// falls back to the bounded scan; the machine records durable failures.
#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error("index store error: {0}")]
    Store(#[from] faktor_store::StoreError),
    #[error("workspace {0} is not a known workspace: {1}")]
    UnknownWorkspace(u64, String),
    #[error("index state machine error: {0}")]
    State(#[from] StateError),
    #[error("workspace {workspace} index not ready within deadline")]
    Deadline { workspace: u64 },
    #[error("index io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("index build for workspace {workspace} failed: {message}")]
    BuildFailed { workspace: u64, message: String },
    #[error("corrupt generation data for workspace {workspace}: {message}")]
    CorruptGeneration { workspace: u64, message: String },
}

/// One immutable published-generation snapshot. Holding an [`IndexView`]
/// pins generation `g`: the content is an `Arc` that outlives any later
/// swap, so reads issued on it never observe a partial `g+1`.
#[derive(Clone)]
pub struct IndexView {
    workspace: WorkspaceId,
    generation: u64,
    index: Arc<Mutex<WorkspaceIndex>>,
}

impl IndexView {
    pub fn workspace(&self) -> WorkspaceId {
        self.workspace
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// The immutable content of this generation, in the shape the search
    /// layer consumes (`SearchService::new`).
    pub fn index(&self) -> Arc<Mutex<WorkspaceIndex>> {
        self.index.clone()
    }
}

impl std::fmt::Debug for IndexView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexView")
            .field("workspace", &self.workspace)
            .field("generation", &self.generation)
            .finish_non_exhaustive()
    }
}

/// Per-workspace live record (mirror of the durable row + runtime state).
struct LiveWs {
    /// Mirror of the store row's state.
    state: WorkspaceIndexState,
    /// Mirror of the store row's numeric generation.
    row_generation: u64,
    /// Byte-exact `state_json` of the current store row (CAS equality).
    state_json: String,
    root: Option<PathBuf>,
    handle: Option<Arc<WorkspaceHandle>>,
    /// Published content (always for the newest published generation).
    content: Option<Arc<Mutex<WorkspaceIndex>>>,
    content_generation: u64,
    /// Fingerprint stored in the published generation file.
    last_fingerprint: Option<Vec<FingerprintEntry>>,
    /// A build is in flight IN THIS PROCESS (lease-guarded).
    building: bool,
    building_since: Option<Instant>,
    /// Events/retries arrived since the last build started (coalesced).
    pending: bool,
    /// The published generation's content is not loaded yet.
    pending_load: bool,
    last_fp_check: Instant,
}

struct Inner {
    store: Arc<Store>,
    data_root: PathBuf,
    fs: Arc<WorkspaceFileService>,
    cfg: Mutex<ServiceConfig>,
    live: Mutex<HashMap<WorkspaceId, LiveWs>>,
    notify: tokio::sync::Notify,
    worker_started: AtomicBool,
}

/// The repository index service: durable state machine + generation store +
/// watcher-driven reconciliation worker.
pub struct IndexService {
    inner: Arc<Inner>,
}

/// Outcome of a content scan (see [`scan_workspace`]).
struct ScanOutcome {
    index: WorkspaceIndex,
    fingerprint: Vec<FingerprintEntry>,
    /// True when the filesystem changed while the content was read (the
    /// build publishes anyway; a follow-up rebuild is scheduled).
    churned: bool,
}

/// Build/crash seam hook for adversarial tests, fired at named points of
/// the publish pipeline. A hook may panic to simulate a builder killed
/// mid-build (the durable state stays `Building`; content is never
/// published; no torn reads are possible).
#[cfg(test)]
type SeamHook = Box<dyn Fn(u64, &'static str) + Send>;
#[cfg(test)]
static SEAM: OnceLock<Mutex<Option<SeamHook>>> = OnceLock::new();

#[cfg(test)]
fn fire_seam(ws: u64, point: &'static str) {
    if let Some(lock) = SEAM.get() {
        // Take the hook OUT of the mutex before invoking: a hook that
        // panics (the crash simulation) must never poison the seam lock.
        let hook = lock.lock().expect("seam poisoned").take();
        if let Some(hook) = hook {
            hook(ws, point);
        }
    }
}

#[cfg(not(test))]
fn fire_seam(_ws: u64, _point: &'static str) {}

#[cfg(test)]
pub(crate) fn install_seam(hook: SeamHook) {
    let lock = SEAM.get_or_init(|| Mutex::new(None));
    *lock.lock().expect("seam poisoned") = Some(hook);
}

#[cfg(test)]
pub(crate) fn clear_seam() {
    if let Some(lock) = SEAM.get() {
        *lock.lock().expect("seam poisoned") = None;
    }
}

impl IndexService {
    /// Open (creating when needed) an index service rooted at `data_root`,
    /// persisted through `store`, watching workspaces via `fs`.
    pub fn open(
        store: Arc<Store>,
        data_root: PathBuf,
        fs: Arc<WorkspaceFileService>,
    ) -> Result<Arc<Self>, IndexError> {
        fs::create_dir_all(data_root.join("generations"))?;
        fs::create_dir_all(data_root.join("scratch"))?;
        Ok(Arc::new(Self {
            inner: Arc::new(Inner {
                store,
                data_root,
                fs,
                cfg: Mutex::new(ServiceConfig::default()),
                live: Mutex::new(HashMap::new()),
                notify: tokio::sync::Notify::new(),
                worker_started: AtomicBool::new(false),
            }),
        }))
    }

    /// Tune the reconciliation knobs (operator/test surface).
    pub fn set_config(&self, cfg: ServiceConfig) {
        *self.inner.cfg.lock().expect("cfg poisoned") = cfg;
    }

    fn cfg_of(inner: &Inner) -> ServiceConfig {
        *inner.cfg.lock().expect("cfg poisoned")
    }

    /// Start the single background reconciliation worker of this service
    /// (idempotent). Requires a tokio runtime context; when none exists the
    /// service keeps working synchronously through
    /// [`IndexService::ensure_ready`] / [`IndexService::reconcile_now`].
    pub fn spawn_worker(&self) -> bool {
        let inner = self.inner.clone();
        if inner
            .worker_started
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return true;
        }
        if tokio::runtime::Handle::try_current().is_err() {
            inner.worker_started.store(false, Ordering::Release);
            return false;
        }
        let weak = Arc::downgrade(&inner);
        tokio::spawn(async move {
            worker_loop(weak).await;
        });
        true
    }

    /// Register a workspace: mirror its durable state row, open its watcher
    /// handle, and kick the reconciliation worker. Idempotent and cheap;
    /// NEVER blocks on a build. Failures (unknown/unresolvable workspace)
    /// are errors the caller turns into fallback behavior.
    pub fn attach(&self, workspace: WorkspaceId) -> Result<(), IndexError> {
        {
            let live = self.inner.live.lock().expect("live poisoned");
            if live.contains_key(&workspace) {
                drop(live);
                self.inner.notify.notify_one();
                self.spawn_worker_if_possible();
                return Ok(());
            }
        }
        // Fresh workspace: resolve root + watcher handle.
        let root_s = self
            .inner
            .store
            .workspace_root(workspace)?
            .ok_or_else(|| IndexError::UnknownWorkspace(workspace.raw(), "no row".into()))?;
        let root = PathBuf::from(root_s);
        let handle = match self.inner.fs.open(workspace, root.clone()) {
            Ok(h) => Some(Arc::new(h)),
            Err(e) => {
                tracing::warn!(
                    workspace = workspace.raw(),
                    "index attach: watcher open failed ({}); fingerprint reconciliation only",
                    e
                );
                None
            }
        };
        // Durable row mirror; corrupt payloads fail OPEN as a durable
        // Failed row (never a silent NotStarted default).
        let (mut state, mut row_generation, mut state_json) =
            match self.inner.store.index_state_get(workspace)? {
                Some(row) => {
                    match PersistedIndexState::parse(row.state_json.clone(), row.generation) {
                        Ok(p) => (p.state, p.row_generation, p.state_json),
                        Err(e) => {
                            let msg = format!("corrupt persisted index state: {e}");
                            tracing::error!(workspace = workspace.raw(), "{msg}");
                            let failed = WorkspaceIndexState::Failed {
                                message: truncate(&msg, 256),
                            };
                            let failed_json = failed.to_row_json();
                            self.inner.store.index_state_put(
                                workspace,
                                &failed_json,
                                row.generation.max(0),
                                JOURNAL_CORRUPT,
                            )?;
                            (failed, row.generation.max(0) as u64, failed_json)
                        }
                    }
                }
                None => {
                    // Seed the durable row so CAS transitions have a target.
                    let ns = WorkspaceIndexState::NotStarted;
                    self.inner.store.index_state_put(
                        workspace,
                        &ns.to_row_json(),
                        0,
                        JOURNAL_NOT_STARTED,
                    )?;
                    (ns.clone(), 0, ns.to_row_json())
                }
            };
        // Torn-publish recovery: the row says Ready{g} but the generation
        // file is absent (crash between the publish CAS and the rename):
        // heal durably through the machine (Ready -> Dirty -> rebuild).
        if let WorkspaceIndexState::Ready { generation } = &state {
            let gen = *generation;
            if !generation_file_path(&self.inner.data_root, workspace, gen).exists() {
                tracing::warn!(
                    workspace = workspace.raw(),
                    generation = gen,
                    "torn publish heal: ready generation file missing"
                );
                let dirty = WorkspaceIndexState::Dirty { generation: gen };
                state.check_transition(row_generation, &dirty, gen)?;
                let ok = self.inner.store.index_state_cas(
                    workspace,
                    &state_json,
                    row_generation as i64,
                    &dirty.to_row_json(),
                    gen as i64,
                    JOURNAL_TORN_READY,
                )?;
                if ok {
                    state = dirty;
                    state_json = state.to_row_json();
                    row_generation = gen;
                }
            }
        }
        let pending = !matches!(
            state,
            WorkspaceIndexState::Ready { .. } | WorkspaceIndexState::Failed { .. }
        );
        let pending_load = matches!(
            state,
            WorkspaceIndexState::Ready { .. } | WorkspaceIndexState::Dirty { .. }
        );
        let cfg = Self::cfg_of(&self.inner);
        {
            let mut live = self.inner.live.lock().expect("live poisoned");
            live.insert(
                workspace,
                LiveWs {
                    state,
                    row_generation,
                    state_json,
                    root: Some(root),
                    handle,
                    content: None,
                    content_generation: 0,
                    last_fingerprint: None,
                    building: false,
                    building_since: None,
                    pending,
                    pending_load,
                    last_fp_check: Instant::now() - cfg.fingerprint_interval,
                },
            );
        }
        self.inner.notify.notify_one();
        self.spawn_worker_if_possible();
        Ok(())
    }

    fn spawn_worker_if_possible(&self) {
        let _ = self.spawn_worker();
    }

    /// Immediate view of the newest PUBLISHED generation of `workspace`.
    /// `None` while the first build is in flight (no Ready generation yet),
    /// or while the persisted Ready content is still being reloaded. Never
    /// blocks and never triggers work.
    pub fn view(&self, workspace: WorkspaceId) -> Option<IndexView> {
        let live = self.inner.live.lock().expect("live poisoned");
        let l = live.get(&workspace)?;
        if matches!(l.state, WorkspaceIndexState::NotStarted) || l.pending_load {
            return None;
        }
        let index = l.content.as_ref()?;
        Some(IndexView {
            workspace,
            generation: l.content_generation,
            index: index.clone(),
        })
    }

    /// Probe the mirrored durable state (tests/observability).
    pub fn state(&self, workspace: WorkspaceId) -> Option<(WorkspaceIndexState, u64)> {
        let live = self.inner.live.lock().expect("live poisoned");
        live.get(&workspace)
            .map(|l| (l.state.clone(), l.row_generation))
    }

    /// Explicit retry (e.g. after [`WorkspaceIndexState::Failed`]) or event
    /// kick: marks the workspace pending and wakes the worker. Never blocks.
    pub fn request_build(&self, workspace: WorkspaceId) -> Result<(), IndexError> {
        self.attach(workspace)?;
        {
            let mut live = self.inner.live.lock().expect("live poisoned");
            if let Some(l) = live.get_mut(&workspace) {
                l.pending = true;
            }
        }
        self.inner.notify.notify_one();
        self.spawn_worker_if_possible();
        Ok(())
    }

    /// Blocking readiness: waits until the workspace's machine is at rest
    /// on a PUBLISHED generation (`Ready`, no pending rebuild) and returns
    /// that view — driving builds synchronously when no worker is running
    /// (tests, retry paths). A stale-but-readable generation is served
    /// while a rebuild is in flight only after the deadline passes (the
    /// caller then keeps the newest published snapshot). The runtime NEVER
    /// calls this on the first-prompt path — it uses
    /// [`IndexService::view`] + attach only.
    pub fn ensure_ready(
        &self,
        workspace: WorkspaceId,
        deadline: Instant,
    ) -> Result<IndexView, IndexError> {
        self.attach(workspace)?;
        let mut stale_view: Option<IndexView> = None;
        loop {
            if self.machine_at_rest(workspace) {
                if let Some(view) = self.view(workspace) {
                    return Ok(view);
                }
            } else if stale_view.is_none() {
                stale_view = self.view(workspace);
            }
            if Instant::now() >= deadline {
                if let Some(view) = stale_view.or_else(|| self.view(workspace)) {
                    return Ok(view); // serve the newest published snapshot
                }
                return Err(IndexError::Deadline {
                    workspace: workspace.raw(),
                });
            }
            // One reconcile pass may claim and run a full build inline.
            let _ = self.reconcile_now(workspace);
            std::thread::sleep(Duration::from_millis(15));
        }
    }

    /// True when the workspace has no pending rebuild: durable state is
    /// Ready (or Failed with no retry pending) and the published content
    /// for that state is loaded.
    fn machine_at_rest(&self, workspace: WorkspaceId) -> bool {
        let live = self.inner.live.lock().expect("live poisoned");
        match live.get(&workspace) {
            Some(l) => {
                let ready_state = match &l.state {
                    WorkspaceIndexState::Ready { .. } => true,
                    WorkspaceIndexState::Failed { .. } => !l.pending,
                    _ => false,
                };
                ready_state
                    && !l.pending
                    && !l.pending_load
                    && !l.building
                    && l.content.is_some()
                    && l.content_generation == l.state.generation().unwrap_or(u64::MAX)
            }
            None => false,
        }
    }

    /// One synchronous reconciliation pass over one workspace: drain
    /// watcher events, mark dirty durably, load published content, claim
    /// and run at most one build. The worker calls this on the blocking
    /// pool; synchronous callers (ensure_ready) call it directly.
    pub fn reconcile_now(&self, workspace: WorkspaceId) -> Result<(), IndexError> {
        // 1. Drain this workspace's watcher channel (lossy by design; all
        // events coalesce into ONE pending flag — a 1000-event storm is one
        // dirty mark and one rebuild).
        {
            let mut live = self.inner.live.lock().expect("live poisoned");
            if let Some(l) = live.get_mut(&workspace) {
                if let Some(handle) = &l.handle {
                    let mut rx = handle.events().lock().expect("events poisoned");
                    let mut saw = false;
                    while let Ok(ev) = rx.try_recv() {
                        if ev.workspace_id == workspace
                            && matches!(
                                ev.kind,
                                FsEventKind::Created | FsEventKind::Modified | FsEventKind::Removed
                            )
                        {
                            saw = true;
                        }
                    }
                    if saw {
                        l.pending = true;
                    }
                }
            }
        }
        // 2. Machine steps (bounded loop; at most one build per call).
        for _ in 0..8 {
            let next = self.decide_next(workspace)?;
            match next {
                Next::Idle => return Ok(()),
                Next::Load => self.load_content(workspace)?,
                Next::MarkDirty => self.mark_dirty(workspace)?,
                Next::Claim { target, kind } => {
                    if self.claim_build(workspace, target, kind)? {
                        // One claim per reconcile call: run the build
                        // inline (blocking by design; the worker wraps the
                        // whole pass in spawn_blocking).
                        self.run_build(workspace, target)?;
                        return Ok(());
                    }
                    // A concurrent builder claimed (or a restart advanced)
                    // the row: re-read the durable truth before deciding.
                    self.refresh_mirror(workspace)?;
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    /// Decide the next machine step from the mirrored state.
    fn decide_next(&self, workspace: WorkspaceId) -> Result<Next, IndexError> {
        let cfg = Self::cfg_of(&self.inner);
        let mut live = self.inner.live.lock().expect("live poisoned");
        let Some(l) = live.get_mut(&workspace) else {
            return Ok(Next::Idle);
        };
        let state = l.state.clone();
        let row_generation = l.row_generation;
        let claimable = {
            let active = l.building
                && l.building_since
                    .map(|t| t.elapsed() < cfg.build_lease)
                    .unwrap_or(false);
            !active
        };
        let now = Instant::now();
        match &state {
            WorkspaceIndexState::NotStarted => {
                if claimable {
                    Ok(Next::Claim {
                        target: 1,
                        kind: JOURNAL_BUILDING,
                    })
                } else {
                    Ok(Next::Idle)
                }
            }
            WorkspaceIndexState::Building { .. } => {
                if claimable {
                    // Durable Building with no live builder: crash residue,
                    // resume the SAME generation the row names.
                    Ok(Next::Claim {
                        target: row_generation,
                        kind: JOURNAL_RESUME,
                    })
                } else {
                    Ok(Next::Idle)
                }
            }
            WorkspaceIndexState::Dirty { .. } => {
                let target = state.next_build_target(row_generation)?;
                if claimable {
                    Ok(Next::Claim {
                        target,
                        kind: JOURNAL_BUILDING,
                    })
                } else {
                    Ok(Next::Idle)
                }
            }
            WorkspaceIndexState::Failed { .. } => {
                if l.pending && claimable {
                    let target = state.next_build_target(row_generation)?;
                    Ok(Next::Claim {
                        target,
                        kind: JOURNAL_BUILDING,
                    })
                } else {
                    Ok(Next::Idle)
                }
            }
            WorkspaceIndexState::Ready { generation } => {
                let gen = *generation;
                if l.pending_load || l.content_generation != gen || l.content.is_none() {
                    l.pending_load = true;
                    Ok(Next::Load)
                } else if l.pending
                    || now.duration_since(l.last_fp_check) >= cfg.fingerprint_interval
                {
                    // Every rebuild decision is fingerprint-VERIFIED: a
                    // pending watcher event (or the periodic reconciliation
                    // that heals events the lossy fs channel dropped) only
                    // materializes into a durable Dirty when the disk really
                    // differs from the published generation. Duplicate or
                    // late events for an already-built state are dropped —
                    // a watcher storm never churns generations.
                    l.last_fp_check = now;
                    let was_pending = l.pending;
                    let stored = l.last_fingerprint.clone();
                    let root = l.root.clone();
                    drop(live);
                    let mismatch = match (stored, root) {
                        (Some(stored), Some(root)) => {
                            fingerprint(&root).map(|fp| fp != stored).unwrap_or(false)
                        }
                        _ => false,
                    };
                    let mut live = self.inner.live.lock().expect("live poisoned");
                    let Some(l) = live.get_mut(&workspace) else {
                        return Ok(Next::Idle);
                    };
                    if mismatch {
                        tracing::info!(
                            workspace = workspace.raw(),
                            "filesystem differs from the published generation; marking dirty"
                        );
                        l.pending = true;
                        Ok(Next::MarkDirty)
                    } else {
                        l.pending = false;
                        if was_pending {
                            tracing::debug!(
                                workspace = workspace.raw(),
                                "stale watcher event for an already-built state; dropped"
                            );
                        }
                        Ok(Next::Idle)
                    }
                } else {
                    Ok(Next::Idle)
                }
            }
        }
    }

    /// Re-read the durable row into the mirror (a CAS lost the race, or an
    /// external writer advanced the row).
    fn refresh_mirror(&self, workspace: WorkspaceId) -> Result<(), IndexError> {
        let row = self.inner.store.index_state_get(workspace)?;
        let mut live = self.inner.live.lock().expect("live poisoned");
        let Some(l) = live.get_mut(&workspace) else {
            return Ok(());
        };
        match row {
            Some(row) => match PersistedIndexState::parse(row.state_json.clone(), row.generation) {
                Ok(p) => refresh_mirror_into(l, p),
                Err(_) => {
                    // Corrupt row from an external writer: fail open loudly.
                    let failed = WorkspaceIndexState::Failed {
                        message: "corrupt persisted index state".into(),
                    };
                    let failed_json = failed.to_row_json();
                    let _ = self.inner.store.index_state_put(
                        workspace,
                        &failed_json,
                        row.generation.max(0),
                        JOURNAL_CORRUPT,
                    );
                    refresh_mirror_into(
                        l,
                        PersistedIndexState {
                            state: failed,
                            row_generation: row.generation.max(0) as u64,
                            state_json: failed_json,
                        },
                    );
                }
            },
            None => {
                l.state = WorkspaceIndexState::NotStarted;
                l.row_generation = 0;
                l.state_json = l.state.to_row_json();
                l.pending = true;
                l.pending_load = false;
                l.content = None;
            }
        }
        Ok(())
    }

    /// Load the published generation's content from its durable file.
    fn load_content(&self, workspace: WorkspaceId) -> Result<(), IndexError> {
        let (state, state_json, row_generation) = {
            let live = self.inner.live.lock().expect("live poisoned");
            let l = live.get(&workspace).expect("live ws present");
            (l.state.clone(), l.state_json.clone(), l.row_generation)
        };
        let gen = match &state {
            WorkspaceIndexState::Ready { generation }
            | WorkspaceIndexState::Dirty { generation } => *generation,
            _ => return Ok(()),
        };
        let path = generation_file_path(&self.inner.data_root, workspace, gen);
        let loaded = match read_generation_file(&path) {
            Ok(file) => {
                if file.workspace != workspace.raw() || file.generation != gen {
                    Err(IndexError::CorruptGeneration {
                        workspace: workspace.raw(),
                        message: format!(
                            "envelope claims workspace {}/gen {}, expected {}/{gen}",
                            file.workspace, file.generation, workspace
                        ),
                    })
                } else {
                    file.materialize()
                        .map(|idx| (idx, file.fingerprint))
                        .map_err(|e| IndexError::CorruptGeneration {
                            workspace: workspace.raw(),
                            message: e,
                        })
                }
            }
            Err(e) => Err(e),
        };
        let mut live = self.inner.live.lock().expect("live poisoned");
        let Some(l) = live.get_mut(&workspace) else {
            return Ok(());
        };
        match loaded {
            Ok((idx, fingerprint)) => {
                l.content = Some(Arc::new(Mutex::new(idx)));
                l.content_generation = gen;
                l.last_fingerprint = Some(fingerprint);
                l.pending_load = false;
                drop(live);
                prune_generations(&self.inner.data_root, workspace, gen);
                Ok(())
            }
            Err(e) => {
                // Missing/corrupt file under a published generation: heal
                // through the machine — Ready -> Dirty -> rebuild. The
                // journal names the tear loudly; the row never silently
                // falls back to "no index".
                tracing::error!(
                    workspace = workspace.raw(),
                    generation = gen,
                    "published generation unreadable: {e}"
                );
                let dirty = WorkspaceIndexState::Dirty { generation: gen };
                let ok = self.inner.store.index_state_cas(
                    workspace,
                    &state_json,
                    row_generation as i64,
                    &dirty.to_row_json(),
                    gen as i64,
                    JOURNAL_TORN_READY,
                )?;
                if ok {
                    l.state = dirty;
                    l.state_json = l.state.to_row_json();
                    l.pending = true;
                    l.pending_load = false;
                }
                Ok(())
            }
        }
    }

    /// Durable `Ready { g } -> Dirty { g }` (the watcher-event hop).
    fn mark_dirty(&self, workspace: WorkspaceId) -> Result<(), IndexError> {
        let (state, state_json, row_generation) = {
            let live = self.inner.live.lock().expect("live poisoned");
            let l = live.get(&workspace).expect("live ws present");
            (l.state.clone(), l.state_json.clone(), l.row_generation)
        };
        let WorkspaceIndexState::Ready { generation } = state else {
            return Ok(());
        };
        let dirty = WorkspaceIndexState::Dirty { generation };
        state.check_transition(row_generation, &dirty, generation)?;
        let ok = self.inner.store.index_state_cas(
            workspace,
            &state_json,
            row_generation as i64,
            &dirty.to_row_json(),
            generation as i64,
            JOURNAL_DIRTY,
        )?;
        let mut live = self.inner.live.lock().expect("live poisoned");
        let Some(l) = live.get_mut(&workspace) else {
            return Ok(());
        };
        if ok {
            l.state = dirty;
            l.state_json = l.state.to_row_json();
        } else {
            // Lost to a concurrent writer: re-read before deciding again.
            l.pending = true;
        }
        Ok(())
    }

    /// CAS into `Building { target }` (the claim). Exactly one claimant
    /// wins per generation; losers refresh their mirror and wait.
    fn claim_build(
        &self,
        workspace: WorkspaceId,
        target: u64,
        kind: &'static str,
    ) -> Result<bool, IndexError> {
        let cfg = Self::cfg_of(&self.inner);
        let (state, state_json, row_generation) = {
            let mut live = self.inner.live.lock().expect("live poisoned");
            let stale = live
                .get(&workspace)
                .map(|l| {
                    l.building
                        && l.building_since
                            .map(|t| t.elapsed() >= cfg.build_lease)
                            .unwrap_or(false)
                })
                .unwrap_or(false);
            if stale {
                let l = live.get_mut(&workspace).expect("live ws present");
                l.building = false; // stale lease: crash reclaim
                l.building_since = None;
            }
            let l = live.get(&workspace).expect("live ws present");
            if l.building {
                return Ok(false);
            }
            (l.state.clone(), l.state_json.clone(), l.row_generation)
        };
        let building = WorkspaceIndexState::Building { generation: target };
        let resume = matches!(state, WorkspaceIndexState::Building { .. });
        let journal_kind = if resume || kind == JOURNAL_RESUME {
            JOURNAL_RESUME
        } else {
            JOURNAL_BUILDING
        };
        state.check_transition(row_generation, &building, target)?;
        let ok = self.inner.store.index_state_cas(
            workspace,
            &state_json,
            row_generation as i64,
            &building.to_row_json(),
            target as i64,
            journal_kind,
        )?;
        if ok {
            let mut live = self.inner.live.lock().expect("live poisoned");
            if let Some(l) = live.get_mut(&workspace) {
                l.state = building;
                l.state_json = l.state.to_row_json();
                l.row_generation = target;
                l.building = true;
                l.building_since = Some(Instant::now());
                l.pending = false;
            }
        }
        Ok(ok)
    }

    /// Run the whole build pipeline for a claimed generation: scan, scratch
    /// write + fsync, [seam], publish CAS, rename, in-memory swap, prune.
    ///
    /// Failure modes:
    /// - genuine build error -> durable `Building -> Failed` (message);
    /// - panic/unwind mid-pipeline (crash) -> durable state stays
    ///   `Building { target }`; the next attach/reclaim resumes the SAME
    ///   target and no reader ever saw a torn generation;
    /// - publish CAS lost -> another builder won; scratch discarded.
    fn run_build(&self, workspace: WorkspaceId, target: u64) -> Result<(), IndexError> {
        let ws_raw = workspace.raw();
        let (expected_json, root) = {
            let live = self.inner.live.lock().expect("live poisoned");
            let l = live.get(&workspace).expect("live ws present");
            (l.state_json.clone(), l.root.clone())
        };
        let Some(root) = root else {
            return self.fail_build(
                workspace,
                target,
                &expected_json,
                "no workspace root".into(),
            );
        };
        let scan = match scan_workspace(workspace, &root) {
            Ok(outcome) => outcome,
            Err(e) => {
                eprintln!("[run_build] scan error: {e}");
                return self.fail_build(workspace, target, &expected_json, e);
            }
        };
        let envelope = GenerationFile::capture(ws_raw, target, &scan.index, scan.fingerprint);
        let bytes = envelope.to_bytes().map_err(|e| IndexError::BuildFailed {
            workspace: ws_raw,
            message: e,
        })?;
        // Scratch write + fsync; the single rename later makes it visible.
        let scratch = write_scratch(&self.inner.data_root, workspace, target, &bytes)?;
        // Crash seam: a test hook may kill this builder right here. The
        // durable state is still Building{target} and the content was never
        // made visible (no rename), so no reader can observe a torn
        // generation and the next build resumes the same target.
        fire_seam(ws_raw, "after_scratch");
        // The publish directory must exist before the CAS: the rename that
        // follows can then only fail for real IO reasons (which the torn
        // publish heal on the next reconcile recovers).
        fs::create_dir_all(generation_dir(&self.inner.data_root, workspace))?;
        // Publish CAS: only ONE builder wins per generation.
        let ready = WorkspaceIndexState::Ready { generation: target };
        let ok = self.inner.store.index_state_cas(
            workspace,
            &expected_json,
            target as i64,
            &ready.to_row_json(),
            target as i64,
            JOURNAL_READY,
        )?;
        if !ok {
            // Another builder published this generation (crash-resume race):
            // discard our scratch and refresh the mirror.
            let _ = fs::remove_file(&scratch);
            let mut live = self.inner.live.lock().expect("live poisoned");
            if let Some(l) = live.get_mut(&workspace) {
                l.building = false;
                l.building_since = None;
            }
            return Ok(());
        }
        // The single rename that makes the generation visible.
        let gen_path = generation_file_path(&self.inner.data_root, workspace, target);
        if let Err(e) = fs::rename(&scratch, &gen_path) {
            // The row already says Ready{target} but the file is missing
            // (torn publish): clear the in-process lease and refresh the
            // mirror so the heal path (Ready -> Dirty -> rebuild) takes
            // over on the next reconcile.
            let mut live = self.inner.live.lock().expect("live poisoned");
            if let Some(l) = live.get_mut(&workspace) {
                l.building = false;
                l.building_since = None;
            }
            return Err(IndexError::Io(e));
        }
        if let Some(parent) = gen_path.parent() {
            fsync_parent(parent);
        }
        if let Some(parent) = scratch.parent() {
            fsync_parent(parent);
        }
        {
            let mut live = self.inner.live.lock().expect("live poisoned");
            if let Some(l) = live.get_mut(&workspace) {
                l.state = ready;
                l.state_json = l.state.to_row_json();
                l.row_generation = target;
                l.content = Some(Arc::new(Mutex::new(scan.index)));
                l.content_generation = target;
                l.last_fingerprint = Some(envelope.fingerprint.clone());
                l.pending_load = false;
                l.building = false;
                l.building_since = None;
                // The filesystem churned while we scanned: schedule the
                // follow-up rebuild only AFTER this publish (a reader of g+1
                // never sees a mix; g+2 is rebuilt from scratch).
                if scan.churned {
                    l.pending = true;
                }
            }
        }
        prune_generations(&self.inner.data_root, workspace, target);
        tracing::info!(
            workspace = ws_raw,
            generation = target,
            "index generation published"
        );
        Ok(())
    }

    /// Journal a genuine build failure: `Building { target } -> Failed`.
    /// The message is durable; retry happens on the next change or on an
    /// explicit [`IndexService::request_build`] — never on a timer.
    fn fail_build(
        &self,
        workspace: WorkspaceId,
        target: u64,
        expected_json: &str,
        message: String,
    ) -> Result<(), IndexError> {
        tracing::error!(
            workspace = workspace.raw(),
            generation = target,
            "index build failed: {message}"
        );
        let failed = WorkspaceIndexState::Failed {
            message: truncate(&message, 512),
        };
        let ok = self.inner.store.index_state_cas(
            workspace,
            expected_json,
            target as i64,
            &failed.to_row_json(),
            target as i64,
            JOURNAL_FAILED,
        )?;
        let mut live = self.inner.live.lock().expect("live poisoned");
        if let Some(l) = live.get_mut(&workspace) {
            if ok {
                l.state = failed;
                l.state_json = l.state.to_row_json();
                l.row_generation = target;
            }
            l.building = false;
            l.building_since = None;
        }
        Ok(())
    }
}

/// What the reconciler should do next for one workspace.
enum Next {
    Idle,
    /// Load the published generation's content from its durable file.
    Load,
    /// Mark the published generation dirty durably (`Ready -> Dirty`).
    MarkDirty,
    /// Claim a build into `Building { target }` and run it.
    Claim {
        target: u64,
        kind: &'static str,
    },
}

fn refresh_mirror_into(l: &mut LiveWs, p: PersistedIndexState) {
    let keep_content = match (&p.state, l.content_generation) {
        (
            WorkspaceIndexState::Ready { generation } | WorkspaceIndexState::Dirty { generation },
            cg,
        ) => *generation == cg && l.content.is_some(),
        _ => false,
    };
    if !keep_content {
        l.content = None;
        l.last_fingerprint = None;
    }
    l.state = p.state;
    l.state_json = p.state_json;
    l.row_generation = p.row_generation;
    l.building = false;
    l.building_since = None;
    l.pending_load = matches!(l.state, WorkspaceIndexState::Ready { .. }) && !keep_content;
    l.pending = matches!(
        l.state,
        WorkspaceIndexState::NotStarted
            | WorkspaceIndexState::Building { .. }
            | WorkspaceIndexState::Dirty { .. }
    );
    l.last_fp_check = Instant::now() - DEFAULT_FINGERPRINT_INTERVAL;
}

/// Default fingerprint reconciliation interval (see [`ServiceConfig`]).
const DEFAULT_FINGERPRINT_INTERVAL: Duration = Duration::from_secs(30);

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let mut end = max;
        while !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}…", &s[..end])
    }
}

// ------------------------------------------------------------------ worker

/// The background reconciliation worker: ONE task per service. Runs bounded
/// synchronous passes on the blocking pool; waits for kicks or the poll
/// cadence between passes. The worker holds only a `Weak` — when the
/// service is dropped the task exits on the next upgrade.
async fn worker_loop(weak: std::sync::Weak<Inner>) {
    loop {
        let Some(inner) = weak.upgrade() else {
            return;
        };
        let cfg = IndexService::cfg_of(&inner);
        let service = IndexService {
            inner: inner.clone(),
        };
        let again = tokio::task::spawn_blocking(move || service.run_pass())
            .await
            .unwrap_or(true);
        if again {
            continue;
        }
        tokio::select! {
            _ = inner.notify.notified() => {}
            _ = tokio::time::sleep(cfg.poll) => {}
        }
    }
}

impl IndexService {
    /// One blocking pass over every attached workspace, executed by the
    /// worker under `spawn_blocking` (and reusable synchronously). Returns
    /// true when more work is immediately claimable.
    fn run_pass(&self) -> bool {
        let workspaces: Vec<WorkspaceId> = self
            .inner
            .live
            .lock()
            .expect("live poisoned")
            .keys()
            .copied()
            .collect();
        let mut work_remains = false;
        for ws in workspaces {
            if let Err(e) = self.reconcile_now(ws) {
                tracing::warn!(workspace = ws.raw(), "reconcile pass error: {e}");
                continue;
            }
            let live = self.inner.live.lock().expect("live poisoned");
            if live
                .get(&ws)
                .map(|l| l.pending || l.pending_load)
                .unwrap_or(false)
            {
                work_remains = true;
            }
        }
        work_remains
    }
}

// ------------------------------------------------------------------ layout

/// Durable published-generation file of one workspace.
fn generation_dir(data_root: &Path, ws: WorkspaceId) -> PathBuf {
    data_root.join("generations").join(ws.raw().to_string())
}

/// Scratch area builders write (and fsync) into before the single rename.
fn scratch_dir(data_root: &Path, ws: WorkspaceId) -> PathBuf {
    data_root.join("scratch").join(ws.raw().to_string())
}

fn generation_file_path(data_root: &Path, ws: WorkspaceId, generation: u64) -> PathBuf {
    generation_dir(data_root, ws).join(format!("gen-{generation}.json"))
}

static SCRATCH_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Write + fsync a generation payload to a uniquely named scratch file.
/// Returns the scratch path; the caller renames it to the generation file
/// (the single atomic swap that makes the generation visible).
fn write_scratch(
    data_root: &Path,
    ws: WorkspaceId,
    generation: u64,
    bytes: &[u8],
) -> Result<PathBuf, IndexError> {
    let dir = scratch_dir(data_root, ws);
    fs::create_dir_all(&dir)?;
    let nonce = SCRATCH_SEQ.fetch_add(1, Ordering::Relaxed);
    let path = dir.join(format!(
        "gen-{generation}-{}-{nonce}.tmp",
        std::process::id()
    ));
    let mut f = fs::File::create(&path)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    Ok(path)
}

/// Read + decode a generation file (bounded read: oversized hostile files
/// are a loud corruption, never a RAM flood).
fn read_generation_file(path: &Path) -> Result<GenerationFile, IndexError> {
    let meta = fs::metadata(path).map_err(|e| IndexError::CorruptGeneration {
        workspace: 0,
        message: format!("generation file {} unreadable: {e}", path.display()),
    })?;
    if meta.len() > MAX_GENERATION_FILE_BYTES {
        return Err(IndexError::CorruptGeneration {
            workspace: 0,
            message: format!(
                "generation file {} is {} bytes (cap {MAX_GENERATION_FILE_BYTES})",
                path.display(),
                meta.len()
            ),
        });
    }
    let bytes = fs::read(path).map_err(|e| IndexError::CorruptGeneration {
        workspace: 0,
        message: format!("generation file {} unreadable: {e}", path.display()),
    })?;
    GenerationFile::from_bytes(&bytes).map_err(|e| IndexError::CorruptGeneration {
        workspace: 0,
        message: format!("{}: {e}", path.display()),
    })
}

/// Bound disk: after `newest` publishes, remove generation files (and stale
/// scratch files) whose generation is at least `KEEP_GENERATIONS` older.
/// Best effort — errors are logged, never fatal.
fn prune_generations(data_root: &Path, ws: WorkspaceId, newest: u64) {
    if newest < KEEP_GENERATIONS {
        return;
    }
    let floor = newest - KEEP_GENERATIONS;
    for dir in [generation_dir(data_root, ws), scratch_dir(data_root, ws)] {
        let Ok(rd) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in rd.flatten() {
            let Ok(name) = entry.file_name().into_string() else {
                continue;
            };
            // gen-<g>.json | gen-<g>-<pid>-<nonce>.tmp — the generation is
            // the number between "gen-" and the next "-" or ".".
            let g = name
                .strip_prefix("gen-")
                .and_then(|n| n.split(['-', '.']).next())
                .and_then(|n| n.parse::<u64>().ok());
            if let Some(g) = g {
                if g <= floor {
                    if let Err(e) = fs::remove_file(entry.path()) {
                        tracing::debug!("prune {} failed: {e}", entry.path().display());
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------------ scanning

/// Deterministic bounded walk of a workspace root (skips
/// [`SKIP_DIRS`], symlinks, special files, binary and oversized files).
/// Reads at most `SCAN_MAX_FILES` files and `SCAN_MAX_BYTES` in total, so
/// a hostile repo is PARTIALLY indexed — never an unbounded build. Mirrors
/// the bounded evidence scan's budget so both evidence paths agree.
fn scan_workspace(ws: WorkspaceId, root: &Path) -> Result<ScanOutcome, String> {
    let before = fingerprint(root)?;
    let mut index = WorkspaceIndex::new();
    let mut files_scanned = 0usize;
    let mut dirs_visited = 0usize;
    let mut bytes_indexed = 0usize;
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        dirs_visited += 1;
        if dirs_visited > SCAN_MAX_DIRS {
            break;
        }
        let entries = fs::read_dir(&dir).map_err(|e| format!("read_dir {}: {e}", dir.display()))?;
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            match entry.file_type() {
                Ok(t) if t.is_dir() => {
                    if !SKIP_DIRS.contains(&name.as_str()) {
                        stack.push(path);
                    }
                }
                Ok(t) if t.is_file() => {
                    if files_scanned >= SCAN_MAX_FILES {
                        break;
                    }
                    let meta = match fs::metadata(&path) {
                        Ok(m) => m,
                        Err(_) => continue,
                    };
                    if meta.len() > SCAN_MAX_FILE_BYTES {
                        continue;
                    }
                    if bytes_indexed >= SCAN_MAX_BYTES {
                        break;
                    }
                    let bytes = match fs::read(&path) {
                        Ok(b) => b,
                        Err(_) => continue,
                    };
                    // Binary sniff: a NUL in the first 8 KiB means not text.
                    if bytes.iter().take(8192).any(|b| *b == 0) {
                        continue;
                    }
                    let rel = path
                        .strip_prefix(root)
                        .map_err(|_| format!("{} escapes the workspace root", path.display()))?;
                    let rel_str = rel
                        .components()
                        .map(|c| c.as_os_str().to_string_lossy())
                        .collect::<Vec<_>>()
                        .join("/");
                    let modified_ms = meta
                        .modified()
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_millis() as i64)
                        .unwrap_or(0);
                    index
                        .index_file(ws, Path::new(&rel_str), &bytes, modified_ms)
                        .map_err(|e| format!("index {rel_str}: {e}"))?;
                    files_scanned += 1;
                    bytes_indexed = bytes_indexed.saturating_add(bytes.len());
                }
                // Symlinks and special files are never indexed.
                _ => {}
            }
        }
    }
    let after = fingerprint(root)?;
    let churned = after != before;
    Ok(ScanOutcome {
        index,
        fingerprint: after,
        churned,
    })
}

/// Stat-only fingerprint of every regular file under `root` (sorted,
/// bounded). Compared entry-wise: any difference (add/remove/size/mtime)
/// means the published generation is stale.
pub fn fingerprint(root: &Path) -> Result<Vec<FingerprintEntry>, String> {
    let mut out = Vec::new();
    let mut dirs_visited = 0usize;
    let mut files = 0usize;
    let mut stack = vec![(root.to_path_buf(), Vec::<String>::new())];
    while let Some((dir, rel_parts)) = stack.pop() {
        dirs_visited += 1;
        if dirs_visited > FP_MAX_DIRS {
            break;
        }
        let entries = fs::read_dir(&dir).map_err(|e| format!("read_dir {}: {e}", dir.display()))?;
        let mut files_here = 0usize;
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let Ok(ft) = entry.file_type() else {
                continue;
            };
            if ft.is_dir() {
                if !SKIP_DIRS.contains(&name.as_str()) {
                    let mut parts = rel_parts.clone();
                    parts.push(name.clone());
                    stack.push((entry.path(), parts));
                }
            } else if ft.is_file() {
                if files >= FP_MAX_FILES {
                    break;
                }
                files_here += 1;
                if files_here > FP_PER_DIR {
                    break;
                }
                let Ok(meta) = entry.metadata() else {
                    continue;
                };
                let mut parts = rel_parts.clone();
                parts.push(name);
                out.push(FingerprintEntry {
                    path: parts.join("/"),
                    size: meta.len(),
                    modified_ms: meta
                        .modified()
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_millis() as i64)
                        .unwrap_or(0),
                });
                files += 1;
            }
        }
    }
    out.sort();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::WorkspaceIndexState as St;
    use tempfile::TempDir;

    /// The crash seam is a process-global test hook; the harness runs tests
    /// in parallel, so every service test serializes on this lock (the
    /// seam is only ever installed while the crash test holds it).
    fn serial() -> std::sync::MutexGuard<'static, ()> {
        static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
        SERIAL.lock().expect("serial lock poisoned")
    }

    const DEADLINE: Duration = Duration::from_secs(30);

    struct Env {
        _dir: TempDir,
        repo: PathBuf,
        store_root: PathBuf,
        data_root: PathBuf,
    }

    fn env() -> Env {
        let dir = TempDir::new().unwrap();
        let repo = dir.path().join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        let store_root = dir.path().join("store");
        let data_root = dir.path().join("index_data");
        Env {
            _dir: dir,
            repo,
            store_root,
            data_root,
        }
    }

    fn write(repo: &Path, rel: &str, content: &str) {
        let p = repo.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, content).unwrap();
    }

    fn fast_cfg() -> ServiceConfig {
        ServiceConfig {
            poll: Duration::from_millis(25),
            fingerprint_interval: Duration::from_millis(50),
            build_lease: Duration::from_millis(200),
        }
    }

    fn cfg_poll_only() -> ServiceConfig {
        // fingerprint reconciliation disabled for watcher e2e determinism
        ServiceConfig {
            poll: Duration::from_millis(25),
            fingerprint_interval: Duration::from_secs(3600),
            build_lease: Duration::from_millis(200),
        }
    }

    /// Fresh store + service on the env (a "daemon restart").
    fn restart(
        env: &Env,
        fs: Arc<WorkspaceFileService>,
        cfg: Option<ServiceConfig>,
    ) -> (Arc<Store>, Arc<IndexService>, WorkspaceId) {
        let store = Arc::new(Store::open(&env.store_root, true).unwrap());
        let ws = store.create_workspace(env.repo.to_str().unwrap()).unwrap();
        let svc = IndexService::open(store.clone(), env.data_root.clone(), fs).unwrap();
        if let Some(c) = cfg {
            svc.set_config(c);
        }
        (store, svc, ws)
    }

    fn first_fixture() -> (Env, Arc<Store>, Arc<IndexService>, WorkspaceId) {
        let env = env();
        write(
            &env.repo,
            "src/lib.rs",
            "pub fn alpha() -> i64 { 1 }\npub struct Beta {}\n",
        );
        write(&env.repo, "src/util.py", "def gamma():\n    return 1\n");
        write(&env.repo, "AGENTS.md", "# rules\n");
        write(&env.repo, "target/junk.rs", "fn junk() {}");
        let fs = faktor_fs::WorkspaceFileService::new();
        let (store, svc, ws) = restart(&env, fs, None);
        svc.set_config(fast_cfg());
        (env, store, svc, ws)
    }

    /// Build to `want_gen` and assert the published content serves lookups.
    fn pub_view_asserts(svc: &IndexService, ws: WorkspaceId, want_gen: u64) {
        let view = svc.ensure_ready(ws, Instant::now() + DEADLINE).unwrap();
        assert_eq!(view.generation(), want_gen);
        let arc = view.index();
        let idx = arc.lock().unwrap();
        assert!(!idx.files_for_token(ws, "alpha", 10).is_empty());
        let syms = idx.symbol_lookup(ws, "Beta", 10);
        assert_eq!(syms[0].1.name, "Beta");
        assert!(idx.file_paths(ws).iter().any(|p| p == "src/lib.rs"));
        assert!(
            !idx.file_paths(ws).iter().any(|p| p.contains("junk")),
            "skipped dirs never indexed"
        );
    }

    fn journal_counts(store: &Store, ws: WorkspaceId) -> Vec<(String, i64)> {
        store
            .index_state_log(ws, 100_000)
            .unwrap()
            .into_iter()
            .map(|r| (r.kind, r.generation))
            .collect()
    }

    // ------------------------------------------------------------- happy path

    #[test]
    fn build_serves_and_dirty_rebuilds() {
        let _serial = serial();
        let (env, store, svc, ws) = first_fixture();
        pub_view_asserts(&svc, ws, 1);
        let (state, gen) = svc.state(ws).unwrap();
        assert_eq!((state, gen), (St::Ready { generation: 1 }, 1));
        // Watcher-event semantics: change the tree, request a rebuild.
        write(&env.repo, "src/lib.rs", "pub fn delta() -> i64 { 2 }\n");
        std::thread::sleep(Duration::from_millis(60)); // mtime resolution
        svc.request_build(ws).unwrap();
        let view = svc.ensure_ready(ws, Instant::now() + DEADLINE).unwrap();
        assert_eq!(view.generation(), 2);
        let arc = view.index();
        let idx = arc.lock().unwrap();
        assert!(!idx.symbol_lookup(ws, "delta", 10).is_empty());
        assert!(idx.symbol_lookup(ws, "alpha", 10).is_empty());
        drop(idx);
        let log = journal_counts(&store, ws);
        assert_eq!(
            log.iter().filter(|(k, _)| k == "building").count(),
            2,
            "{log:?}"
        );
        assert_eq!(
            log.iter()
                .filter(|(k, g)| k == "building" && *g == 2)
                .count(),
            1,
            "{log:?}"
        );
        assert_eq!(
            log.iter().filter(|(k, _)| k == "ready").count(),
            2,
            "{log:?}"
        );
        assert_eq!(
            log.iter().filter(|(k, _)| k == "dirty").count(),
            1,
            "{log:?}"
        );
        // Durable row + mirror agree.
        let (state, gen) = svc.state(ws).unwrap();
        assert_eq!((state, gen), (St::Ready { generation: 2 }, 2));
    }

    #[test]
    fn view_is_none_while_first_build_has_no_ready_generation() {
        let _serial = serial();
        let (env, _store, svc, ws) = first_fixture();
        let _ = &svc;
        let _ = &ws;
        // A second service has never built this workspace: attach leaves it
        // NotStarted and view() must be None (never torn/partial) until a
        // Ready generation exists.
        let fs = faktor_fs::WorkspaceFileService::new();
        let (_, svc2, ws2) = restart(&env, fs, Some(fast_cfg()));
        svc2.attach(ws2).unwrap();
        assert!(svc2.view(ws2).is_none());
        let (state, _) = svc2.state(ws2).unwrap();
        assert_eq!(state, St::NotStarted);
        pub_view_asserts(&svc2, ws2, 1);
    }

    // ------------------------------------------------------ (a) crash mid-build

    #[test]
    fn crash_mid_build_leaves_durable_building_and_never_torn() {
        let _serial = serial();
        let (env, store, svc, ws) = first_fixture();
        pub_view_asserts(&svc, ws, 1);
        write(&env.repo, "extra.rs", "pub fn extra_fn() {}\n");
        std::thread::sleep(Duration::from_millis(60));

        let reader_stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let reader = {
            let svc = svc.clone();
            let stop = reader_stop.clone();
            std::thread::spawn(move || {
                let mut last_gen = 0u64;
                while !stop.load(std::sync::atomic::Ordering::SeqCst) {
                    if let Some(view) = svc.view(ws) {
                        let g = view.generation();
                        assert!(
                            g >= last_gen,
                            "generations must be monotone, saw {g} after {last_gen}"
                        );
                        last_gen = g;
                        let arc = view.index();
                        let idx = arc.lock().unwrap();
                        let paths = idx.file_paths(ws);
                        // Torn-read guard: published content is COMPLETE for
                        // its generation. gen-1 never carries extra.rs;
                        // gen-2 always does.
                        let has_extra = paths.iter().any(|p| p == "extra.rs");
                        assert!(
                            !(has_extra && g == 1),
                            "gen 1 must never observe gen 2 rows"
                        );
                        assert!(has_extra || g == 1, "gen {g} must be complete: {paths:?}");
                    }
                    std::thread::sleep(Duration::from_millis(2));
                }
            })
        };

        // 50 crash rounds. Each round: fresh service, seam armed; the
        // rebuild toward gen 2 panics between scratch write and publish.
        let mut crash_rounds = 0u64;
        for _ in 0..50 {
            install_seam(Box::new(move |w, point| {
                if point == "after_scratch" {
                    // Target only THIS round's crash; rounds are sequential
                    // so the hook is cleared right after each attempt.
                    let _ = w;
                    panic!("simulated builder crash between scratch and swap");
                }
            }));
            let fs = faktor_fs::WorkspaceFileService::new();
            let (store2, svc2, ws2) = restart(&env, fs, Some(fast_cfg()));
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                svc2.ensure_ready(ws2, Instant::now() + DEADLINE)
            }));
            clear_seam(); // never leak the crash hook into other tests
            drop(store2);
            match outcome {
                Err(_) => {
                    crash_rounds += 1;
                    // Crash residue: durable + mirrored Building{2}.
                    let (state, gen) = svc2.state(ws2).unwrap();
                    assert_eq!(gen, 2);
                    assert!(matches!(state, St::Building { generation: 2 }), "{state:?}");
                    // Only complete generations are ever visible: gen 1 with
                    // NO extra.rs, or nothing at all.
                    if let Some(view) = svc2.view(ws2) {
                        assert_eq!(view.generation(), 1);
                        let arc = view.index();
                        let idx = arc.lock().unwrap();
                        assert!(
                            !idx.file_paths(ws2).iter().any(|p| p == "extra.rs"),
                            "no gen-2 rows before the swap"
                        );
                    }
                }
                Ok(view) => {
                    // Impossible while the seam is armed (each round crashes
                    // before the publish); kept as a safety net.
                    assert_eq!(view.unwrap().generation(), 2);
                    break;
                }
            }
            drop(svc2);
        }
        assert_eq!(crash_rounds, 50, "every armed round must crash the builder");

        // Final restart WITHOUT the seam completes at exactly gen 2 and the
        // journal shows resume rows for the crashed attempts.
        clear_seam();
        let fs = faktor_fs::WorkspaceFileService::new();
        let (store3, svc3, ws3) = restart(&env, fs, Some(fast_cfg()));
        pub_view_asserts(&svc3, ws3, 2);
        let (state, gen) = svc3.state(ws3).unwrap();
        assert_eq!((state, gen), (St::Ready { generation: 2 }, 2));
        let log = journal_counts(&store3, ws3);
        assert_eq!(
            log.iter()
                .filter(|(k, g)| k == "building" && *g == 2)
                .count(),
            1,
            "exactly ONE first claim of gen 2: {log:?}"
        );
        // Rounds 2..50 resumed the SAME crashed generation AND the final
        // clean restart resumed it once more (50 resume rows total), then
        // the publish succeeded: one resume per crashed attempt, and the
        // crashed target is never renumbered.
        assert_eq!(
            log.iter().filter(|(k, g)| k == "resume" && *g == 2).count(),
            50,
            "each crashed restart must journal a resume of gen 2: {log:?}"
        );
        assert_eq!(
            log.iter().filter(|(k, g)| k == "ready" && *g == 2).count(),
            1,
            "gen 2 published exactly once: {log:?}"
        );
        assert_eq!(
            log.iter().filter(|(k, g)| k == "ready" && *g == 1).count(),
            1,
            "gen 1 published exactly once: {log:?}"
        );
        reader_stop.store(true, std::sync::atomic::Ordering::SeqCst);
        reader.join().unwrap();
        drop(store);
        drop(svc3);
    }

    // ------------------------------------------------------ (b) corrupt JSON

    #[test]
    fn corrupt_persisted_state_fails_open_as_failed_not_silent_default() {
        let _serial = serial();
        let (env, store, svc, ws) = first_fixture();
        pub_view_asserts(&svc, ws, 1);
        drop(svc);
        // Adversary corrupts the state row directly in the store.
        store
            .index_state_put(ws, "{ this is not json !!!", 3, "corrupt")
            .unwrap();
        let fs = faktor_fs::WorkspaceFileService::new();
        let (store2, svc2, ws2) = restart(&env, fs, Some(fast_cfg()));
        svc2.attach(ws2).unwrap();
        let (state, gen) = svc2.state(ws2).unwrap();
        assert!(
            matches!(state, St::Failed { .. }),
            "corrupt row must fail OPEN as durable Failed, got {state:?}"
        );
        assert_eq!(gen, 3);
        assert!(svc2.view(ws2).is_none());
        // Loud journal entry + explicit retry recovers (retry targets the
        // corrupt row's generation 3 — no renumbering).
        let log = journal_counts(&store2, ws2);
        assert_eq!(log[0].0, "corrupt", "{log:?}");
        svc2.request_build(ws2).unwrap();
        let view = svc2.ensure_ready(ws2, Instant::now() + DEADLINE).unwrap();
        assert_eq!(view.generation(), 3);
        let (state, gen) = svc2.state(ws2).unwrap();
        assert_eq!((state, gen), (St::Ready { generation: 3 }, 3));
        // The gen-1 file from before the corruption was pruned when gen 3
        // published (keep 2: gen 2 is missing, so gen 1 goes).
        let gen_dir = generation_dir(&env.data_root, ws2);
        let gens: Vec<String> = std::fs::read_dir(&gen_dir)
            .map(|rd| {
                rd.flatten()
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();
        assert!(
            gens.iter().any(|g| g == "gen-3.json"),
            "gen-3 published: {gens:?}"
        );
    }

    // ------------------------------------------------------ (c) event storm

    #[test]
    fn thousand_event_storm_coalesces_to_one_dirty_and_one_rebuild() {
        let _serial = serial();
        let (env, store, svc, ws) = first_fixture();
        pub_view_asserts(&svc, ws, 1);
        for i in 0..1000 {
            write(&env.repo, &format!("f{i:04}.rs"), "pub fn storm_fn() {}\n");
        }
        std::thread::sleep(Duration::from_millis(80));
        for _ in 0..1000 {
            svc.request_build(ws).unwrap();
        }
        // Settle at gen 2 through sync reconciles.
        let deadline = Instant::now() + DEADLINE;
        while svc.view(ws).map(|v| v.generation()) != Some(2) && Instant::now() < deadline {
            svc.reconcile_now(ws).unwrap();
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(svc.view(ws).unwrap().generation(), 2);
        let view = svc.view(ws).unwrap();
        let arc = view.index();
        let idx = arc.lock().unwrap();
        // 3 fixture files (incl. AGENTS.md) + 1000 storm files.
        assert_eq!(idx.file_count(ws), 1003, "storm files all indexed once");
        drop(idx);
        drop(view);
        let log = journal_counts(&store, ws);
        assert_eq!(
            log.iter().filter(|(k, _)| k == "dirty").count(),
            1,
            "1000 events must coalesce into ONE durable dirty mark: {log:?}"
        );
        assert_eq!(
            log.iter()
                .filter(|(k, g)| k == "building" && *g == 2)
                .count(),
            1,
            "1000 events must coalesce into ONE rebuild: {log:?}"
        );
    }

    // ------------------------------------------------------ (d) builder race

    #[test]
    fn two_racing_builders_swap_once_and_generation_increments_once() {
        let _serial = serial();
        let (env, store, svc, ws) = first_fixture();
        pub_view_asserts(&svc, ws, 1);
        write(&env.repo, "race.rs", "pub fn race_fn() {}\n");
        std::thread::sleep(Duration::from_millis(60));
        svc.request_build(ws).unwrap();
        let svc = Arc::new(svc.clone());
        let mut handles = Vec::new();
        for _ in 0..2 {
            let svc = svc.clone();
            handles.push(std::thread::spawn(move || {
                svc.ensure_ready(ws, Instant::now() + DEADLINE)
                    .unwrap()
                    .generation()
            }));
        }
        let gens: Vec<u64> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(gens, vec![2, 2], "both callers see the SAME new generation");
        let log = journal_counts(&store, ws);
        assert_eq!(
            log.iter()
                .filter(|(k, g)| k == "building" && *g == 2)
                .count(),
            1,
            "exactly ONE builder wins the gen-2 claim: {log:?}"
        );
        assert_eq!(
            log.iter().filter(|(k, g)| k == "ready" && *g == 2).count(),
            1,
            "{log:?}"
        );
    }

    // ------------------------------------------------------ (e) 10 restarts

    #[test]
    fn ten_restarts_keep_generation_monotone_and_ready_persisted() {
        let _serial = serial();
        let (env, _store, svc, ws) = first_fixture();
        pub_view_asserts(&svc, ws, 1);
        let fs = faktor_fs::WorkspaceFileService::new();
        drop(svc);
        let mut last_gen = 1u64;
        for round in 2..=10u64 {
            // The tree changes while NO service is alive. A restart must
            // detect it (fingerprint reconciliation), rebuild exactly one
            // generation, and stay Ready.
            write(
                &env.repo,
                &format!("gen{round}.rs"),
                &format!("pub fn gen{round}() {{}}\n"),
            );
            std::thread::sleep(Duration::from_millis(25));
            let (store_r, svc_r, ws_r) = restart(&env, fs.clone(), Some(fast_cfg()));
            let view = svc_r.ensure_ready(ws_r, Instant::now() + DEADLINE).unwrap();
            assert_eq!(
                view.generation(),
                round,
                "rebuild after restart bumps generation by exactly 1"
            );
            assert!(view.generation() > last_gen);
            last_gen = view.generation();
            let (state, gen) = svc_r.state(ws_r).unwrap();
            assert_eq!((state, gen), (St::Ready { generation: round }, round));
            let arc = view.index();
            let idx = arc.lock().unwrap();
            assert!(!idx
                .symbol_lookup(ws_r, &format!("gen{round}"), 10)
                .is_empty());
            drop(idx);
            drop(view);
            // Restart again WITHOUT changes: Ready stays Ready (no rebuild).
            drop(store_r);
            drop(svc_r);
            let (store_r2, svc_r2, ws_r2) = restart(&env, fs.clone(), Some(fast_cfg()));
            let view = svc_r2
                .ensure_ready(ws_r2, Instant::now() + DEADLINE)
                .unwrap();
            assert_eq!(
                view.generation(),
                round,
                "Ready must stay Ready across a clean restart (no rebuild)"
            );
            let (state, _) = svc_r2.state(ws_r2).unwrap();
            assert_eq!(state, St::Ready { generation: round });
            drop(view);
            // Generation files on disk agree with the durable row.
            let file = generation_file_path(&env.data_root, ws_r2, round);
            assert!(file.exists(), "gen {round} file must be durable");
            drop(store_r2);
            drop(svc_r2);
        }
        assert_eq!(last_gen, 10);
    }

    // ------------------------------------------------------ (f) pruning

    #[test]
    fn old_generation_pruned_exactly_at_ready_n_plus_2() {
        let _serial = serial();
        let (env, _store, svc, ws) = first_fixture();
        pub_view_asserts(&svc, ws, 1);
        let gen_dir = generation_dir(&env.data_root, ws);
        let list_gens = || -> Vec<u64> {
            let mut v: Vec<u64> = std::fs::read_dir(&gen_dir)
                .unwrap()
                .flatten()
                .filter_map(|e| {
                    let n = e.file_name().to_string_lossy().into_owned();
                    n.strip_prefix("gen-")
                        .and_then(|n| n.strip_suffix(".json"))
                        .and_then(|n| n.parse().ok())
                })
                .collect();
            v.sort();
            v
        };
        assert_eq!(list_gens(), vec![1], "gen 1 file present at Ready(1)");
        // Change -> Ready(2): data for gen 1 must STILL be present.
        write(&env.repo, "two.rs", "pub fn two() {}\n");
        std::thread::sleep(Duration::from_millis(60));
        svc.request_build(ws).unwrap();
        pub_view_asserts(&svc, ws, 2);
        assert_eq!(
            list_gens(),
            vec![1, 2],
            "gen 1 data still present at Ready(2) (keep 2 generations)"
        );
        // Change -> Ready(3): gen 1 pruned exactly now.
        write(&env.repo, "three.rs", "pub fn three() {}\n");
        std::thread::sleep(Duration::from_millis(60));
        svc.request_build(ws).unwrap();
        pub_view_asserts(&svc, ws, 3);
        assert_eq!(
            list_gens(),
            vec![2, 3],
            "gen 1 pruned exactly at Ready(3); gen 2 kept"
        );
        // No scratch leftovers after clean builds.
        let scratch = scratch_dir(&env.data_root, ws);
        let leftovers: Vec<String> = std::fs::read_dir(&scratch)
            .map(|rd| {
                rd.flatten()
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();
        assert!(leftovers.is_empty(), "scratch leaked: {leftovers:?}");
    }

    // ------------------------------------------------------ watcher e2e

    #[test]
    fn watcher_event_drives_dirty_to_ready_via_worker() {
        let _serial = serial();
        // Explicit runtime (block_on, no .await in this fn): the serial
        // std-lock guard must never cross an await point.
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let env = env();
            write(&env.repo, "a.rs", "pub fn watcher_fn() {}\n");
            let fs = faktor_fs::WorkspaceFileService::new();
            let (store, svc, ws) = restart(&env, fs, None);
            svc.set_config(cfg_poll_only());
            // Attach: the worker builds gen 1 (a.rs only) in the background.
            svc.attach(ws).unwrap();
            let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
            loop {
                if matches!(svc.state(ws), Some((St::Ready { generation: 1 }, 1))) {
                    break;
                }
                if tokio::time::Instant::now() >= deadline {
                    panic!("initial build never reached Ready(1)");
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            // Real fs watcher event: the worker marks dirty (durable, in the
            // journal) and rebuilds to gen 2. The Dirty state itself is
            // transient (dirty -> claim -> build happen inside one reconcile
            // pass), so the assertion is journal-based.
            write(&env.repo, "b.rs", "pub fn second_fn() {}\n");
            loop {
                if let Some((St::Ready { generation }, _)) = svc.state(ws) {
                    if generation >= 2 {
                        break;
                    }
                }
                if tokio::time::Instant::now() >= deadline {
                    panic!("watcher-driven rebuild never reached Ready(2)");
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            // Settle: with no further changes the state STAYS ready — the
            // worker does not rebuild on a loop.
            tokio::time::sleep(Duration::from_millis(600)).await;
            let (state, gen) = svc.state(ws).unwrap();
            assert_eq!((state, gen), (St::Ready { generation: 2 }, 2));
            // Journal: the watcher change produced a DURABLE dirty mark and
            // exactly one claim+publish of gen 2 (fs events coalesce; no
            // duplicate generations).
            let log = journal_counts(&store, ws);
            assert!(
                log.iter().any(|(k, g)| k == "dirty" && *g == 1),
                "the watcher event must be journaled as a durable Dirty: {log:?}"
            );
            let building2 = log
                .iter()
                .filter(|(k, g)| *k == "building" && *g == 2)
                .count();
            let resume2 = log
                .iter()
                .filter(|(k, g)| *k == "resume" && *g == 2)
                .count();
            let ready2 = log.iter().filter(|(k, g)| *k == "ready" && *g == 2).count();
            assert_eq!(ready2, 1, "one publish of gen 2: {log:?}");
            assert!(building2 + resume2 <= 1, "one claim of gen 2: {log:?}");
            drop(svc);
        });
    }

    // ------------------------------------------------------ deadline / failure

    #[test]
    fn ensure_ready_honors_deadline_when_workspace_unreadable() {
        let _serial = serial();
        let env = env();
        write(&env.repo, "gone.rs", "pub fn g() {}\n");
        let fs = faktor_fs::WorkspaceFileService::new();
        let (store, svc, ws) = restart(&env, fs, None);
        // Make the workspace unreadable for builds: move the repo away.
        let moved = env._dir.path().join("moved");
        std::fs::rename(&env.repo, &moved).unwrap();
        svc.set_config(fast_cfg());
        svc.attach(ws).unwrap();
        let err = svc.ensure_ready(ws, Instant::now() + Duration::from_millis(600));
        assert!(matches!(err, Err(IndexError::Deadline { .. })), "{err:?}");
        // The failure was recorded durably as Failed, not swallowed.
        let (state, _) = svc.state(ws).unwrap();
        assert!(
            matches!(state, St::Failed { .. }),
            "unreadable workspace must land in durable Failed: {state:?}"
        );
        drop(store);
        // Explicit retry recovers once the dir is back.
        std::fs::rename(&moved, &env.repo).unwrap();
        svc.request_build(ws).unwrap();
        let view = svc.ensure_ready(ws, Instant::now() + DEADLINE).unwrap();
        assert_eq!(view.generation(), 1);
    }

    #[test]
    fn scratch_orphans_never_become_generations() {
        let _serial = serial();
        // A hostile/crashed writer drops junk into the scratch dir: the
        // service must never load it as a generation, and prune must
        // eventually remove it.
        let (env, _store, svc, ws) = first_fixture();
        pub_view_asserts(&svc, ws, 1);
        let scratch = scratch_dir(&env.data_root, ws);
        std::fs::create_dir_all(&scratch).unwrap();
        std::fs::write(scratch.join("gen-1-9999-1.tmp"), b"partial garbage").unwrap();
        std::fs::write(scratch.join("gen-77-9999-1.tmp"), b"partial garbage").unwrap();
        // gen-77-*.tmp is newer than any published gen: prune must leave it
        // (it belongs to an in-flight target) until that target is passed.
        write(&env.repo, "next.rs", "pub fn next_fn() {}\n");
        std::thread::sleep(Duration::from_millis(60));
        svc.request_build(ws).unwrap();
        pub_view_asserts(&svc, ws, 2);
        // gen-1 scratch pruned at Ready(2)? floor = 2-2 = 0 -> nothing
        // pruned yet; but after Ready(3) both gen-1 leftovers go.
        write(&env.repo, "next2.rs", "pub fn next2_fn() {}\n");
        std::thread::sleep(Duration::from_millis(60));
        svc.request_build(ws).unwrap();
        pub_view_asserts(&svc, ws, 3);
        let leftovers: Vec<String> = std::fs::read_dir(&scratch)
            .map(|rd| {
                rd.flatten()
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();
        assert!(
            !leftovers.iter().any(|n| n.starts_with("gen-1-")),
            "stale gen-1 scratch pruned: {leftovers:?}"
        );
        // A scratch orphan of a FUTURE generation (77) is kept (a real
        // builder may legitimately hold scratch for newest+1) but it can
        // never become a generation: no gen-77.json exists and the view is
        // untouched.
        let gens = generation_dir(&env.data_root, ws);
        let published: Vec<String> = std::fs::read_dir(&gens)
            .map(|rd| {
                rd.flatten()
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();
        assert!(
            !published.iter().any(|n| n.starts_with("gen-77")),
            "scratch must never be published as a generation: {published:?}"
        );
        assert!(svc.view(ws).unwrap().generation() == 3);
    }
}
