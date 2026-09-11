//! faktor-scheduler — tool/subagent concurrency as a dependency DAG with
//! resource-class budgets, state-aware retries with jitter, and circuit
//! breakers. Independent reads/subagents run concurrently; edits touching
//! overlapping ownership sets do not.
//!
//! Scheduling is event-driven: a task starts only when every dependency
//! satisfies its edge policy (see `DependencyPolicy`) AND it holds a resource
//! permit (permit-before-spawn). Completion of any task immediately frees its
//! permit and decrements its dependents, so a long task never gates short
//! tasks that became ready after it.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicI64, AtomicU32, AtomicU8, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use faktor_core::error::{Error, ErrorKind};
use faktor_core::id::{OpId, SessionId};
use faktor_core::op::OpMeta;
use faktor_core::resource::{ResourceClass, ResourceGauge, ResourceLimits};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OwnershipSet(Vec<String>);

impl OwnershipSet {
    pub fn new(paths: impl IntoIterator<Item = String>) -> Self {
        let mut v: Vec<String> = paths.into_iter().collect();
        v.sort();
        v.dedup();
        Self(v)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The canonical sorted path list (for ledger/journal bookkeeping).
    pub fn entries(&self) -> &[String] {
        &self.0
    }

    /// Resolve every entry against `base` (real `canonicalize` when the path
    /// exists, lexical join otherwise) so spellings like `src/../src/a.rs`
    /// collapse to one canonical path. Entries with a trailing `/` stay
    /// directory markers and keep their directory semantics.
    pub fn canonicalized(&self, base: &Path) -> Self {
        let base = std::fs::canonicalize(base).unwrap_or_else(|_| base.to_path_buf());
        let mut v: Vec<String> = Vec::new();
        for p in &self.0 {
            let is_dir = p.ends_with('/');
            let trimmed = p.trim_end_matches('/');
            if trimmed.is_empty() {
                continue;
            }
            let joined = base.join(trimmed);
            let canon = std::fs::canonicalize(&joined).unwrap_or(joined);
            let mut s = canon.to_string_lossy().to_string();
            if is_dir {
                s.push('/');
            }
            v.push(s);
            // keep the raw form; comparison normalizes per-platform in
            // `path_overlaps` (Windows: verbatim/case/separators).
        }
        v.sort();
        v.dedup();
        Self(v)
    }

    /// True when both sets touch any common path (edits must serialize).
    /// A trailing `/` marks a directory entry: it overlaps every path under
    /// it at a component boundary (`src/` covers `src/a.rs`, not `src2/a.rs`).
    /// Entries without a trailing slash are files and overlap only exact
    /// matches.
    pub fn overlaps(&self, other: &OwnershipSet) -> bool {
        self.0
            .iter()
            .any(|a| other.0.iter().any(|b| path_overlaps(a, b)))
    }
}

/// Normalize an ownership path for comparison: strip Windows verbatim/UNC
/// prefixes, fold separators to `/`, and case-fold on Windows (NTFS is
/// case-insensitive; two differently-cased spellings of one path are the
/// same file). Unix behavior is unchanged (no folding, identity).
fn normalize_owned_path(s: &str) -> String {
    let mut t = s.to_string();
    if cfg!(windows) {
        if let Some(rest) = t.strip_prefix("\\\\?\\UNC\\") {
            t = format!("//{rest}");
        } else if let Some(rest) = t.strip_prefix("\\\\?\\") {
            t = rest.to_string();
        }
        t = t.replace('\\', "/");
        t = t.to_ascii_lowercase();
    }
    t
}

fn path_overlaps(a: &str, b: &str) -> bool {
    let (a, b) = (normalize_owned_path(a), normalize_owned_path(b));
    if a == b {
        return true;
    }
    let (a, b) = (a.as_str(), b.as_str());
    let a_dir = a.ends_with('/');
    let b_dir = b.ends_with('/');
    if a_dir
        && b.trim_end_matches('/')
            .starts_with(&format!("{}/", a.trim_end_matches('/')))
    {
        return true;
    }
    if b_dir
        && a.trim_end_matches('/')
            .starts_with(&format!("{}/", b.trim_end_matches('/')))
    {
        return true;
    }
    false
}

/// Two registrations are "the same op" when every schedulable dimension
/// matches: identity, session, resource class, ownership sets and edges.
/// The runnable closure itself is excluded — re-registering a build with a
/// fresh closure but identical semantics is idempotent.
fn same_registration(a: &ScheduledOp, b: &ScheduledOp) -> bool {
    a.meta.operation_id == b.meta.operation_id
        && a.meta.session_id == b.meta.session_id
        && a.resources.class == b.resources.class
        && a.reads == b.reads
        && a.writes == b.writes
        && a.dependencies == b.dependencies
}

/// Which resource class an operation draws from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceRequest {
    pub class: ResourceClass,
}

/// Per-edge dependency semantics: which upstream outcomes satisfy the edge
/// and therefore release the dependent's pending-count.
///
/// A state is TERMINAL when the upstream can make no further execution
/// progress: `Done`, `Failed`, `Cancelled`, and `Blocked` (a blocked task
/// can never run). Whether the upstream "executed" is irrelevant to the
/// edge — a task that can never run is just as final as one that failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DependencyPolicy {
    /// DEFAULT: the dependent runs only if the upstream ended `Done`.
    /// An upstream that `Failed`, was `Cancelled`, or was itself `Blocked`
    /// makes this edge dead: the dependent can never run and is marked
    /// `Blocked` (transitively through its own `Success` edges).
    Success,
    /// The dependent runs after the upstream reaches ANY terminal state:
    /// `Done | Failed | Cancelled | Blocked`. A `Blocked` upstream DOES
    /// satisfy this edge (it can never run, so waiting for it is a false
    /// deadlock). A `Terminal`-edge dependent behind a *chain* of blocked
    /// tasks is released when the dependency graph is next rebuilt from
    /// live state (a blocked task produces no terminal event of its own).
    Terminal,
    /// Cleanup/finalizer edge — defined by PURPOSE, not by a different
    /// state set: use this ONLY for teardown that must fire once its
    /// upstream settles, no matter how it settled (including a `Blocked`
    /// upstream that never ran). Satisfaction is the same terminal set as
    /// `Terminal`; as the sole cleanup policy it is additionally released
    /// through a transitively blocked chain immediately at block time.
    Always,
}

/// One schedulable operation. The whole OpMeta envelope travels with the op:
/// identity, deadline, retry policy, cancellation and crash recovery are one
/// object — the scheduler never builds a second identity.
#[derive(Clone)]
pub struct ScheduledOp {
    pub meta: OpMeta,
    pub resources: ResourceRequest,
    /// Files this task reads (for dependency overlap analysis).
    pub reads: OwnershipSet,
    /// Files this task writes (edits with overlapping writes serialize).
    pub writes: OwnershipSet,
    /// Dependencies, each with the edge policy that gates this task.
    /// The default (and the historical `depends_on` semantics) is
    /// `DependencyPolicy::Success`.
    pub dependencies: Vec<(OpId, DependencyPolicy)>,
    pub run: OpFn,
}

/// The work itself. Kept boxed and pinned so the scheduler stays
/// tool-agnostic and timeout-able without Unpin gymnastics.
pub type OpFn = Arc<
    dyn Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), Error>> + Send>>
        + Send
        + Sync,
>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    Pending,
    Running,
    Done,
    Failed,
    Cancelled,
    /// A `Success` edge on this task ended not-`Done`: it can never run.
    /// Terminal for scheduling purposes; reported by `statuses()`.
    Blocked,
}

/// Distinct from `Error`: signals that a resource budget had no free slot
/// right now and the caller should retry the task later (never a failure).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetBusy(pub ResourceClass);

/// Execution outcome: an ordinary task error or a budget-busy signal.
#[derive(Debug, Clone)]
pub enum ExecuteError {
    Busy(BudgetBusy),
    Err(Error),
}

impl From<Error> for ExecuteError {
    fn from(e: Error) -> Self {
        ExecuteError::Err(e)
    }
}

impl std::fmt::Display for ExecuteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecuteError::Busy(b) => write!(f, "budget busy: {:?}", b.0),
            ExecuteError::Err(e) => write!(f, "{e}"),
        }
    }
}

#[derive(Clone)]
struct TaskState {
    op: ScheduledOp,
    status: TaskStatus,
    error: Option<String>,
    start_ms: i64,
    end_ms: Option<i64>,
    /// Dependencies whose edge is not yet satisfied: their upstream is
    /// still live (not terminal), or the satisfaction waits for a graph
    /// rebuild (a `Terminal`-edge dependent behind a blocked chain).
    remaining: usize,
    /// Dead `Success`-policy edges whose upstream ended not-`Done`: this
    /// task can never run. `blocked > 0` ⇒ status is `Blocked`.
    blocked: usize,
    /// Tasks that wait on this one, with the policy of each edge.
    dependents: Vec<(OpId, DependencyPolicy)>,
}

/// Does `upstream` satisfy an edge with `policy`?
fn edge_satisfied(policy: DependencyPolicy, upstream: TaskStatus) -> bool {
    match policy {
        DependencyPolicy::Success => upstream == TaskStatus::Done,
        DependencyPolicy::Terminal => matches!(
            upstream,
            TaskStatus::Done
                | TaskStatus::Failed
                | TaskStatus::Cancelled
                // A Blocked upstream is terminally unavailable: treating it
                // as satisfied prevents a false deadlock (audit round 5).
                | TaskStatus::Blocked
        ),
        DependencyPolicy::Always => matches!(
            upstream,
            TaskStatus::Done | TaskStatus::Failed | TaskStatus::Cancelled | TaskStatus::Blocked
        ),
    }
}

/// A `Success` edge is dead — its dependent can never run — when the
/// upstream reached a terminal state other than `Done`.
fn success_edge_dead(upstream: TaskStatus) -> bool {
    matches!(
        upstream,
        TaskStatus::Failed | TaskStatus::Cancelled | TaskStatus::Blocked
    )
}

/// Terminal exactly-once (audit 79-80): a task in a terminal state must
/// never receive a second terminal transition. Every transition API is
/// guarded by this predicate so complete/cancel/fail can never silently
/// flip or resurrect a terminal op.
fn status_is_terminal(status: TaskStatus) -> bool {
    matches!(
        status,
        TaskStatus::Done | TaskStatus::Failed | TaskStatus::Cancelled | TaskStatus::Blocked
    )
}

/// A task became terminal (`Done`/`Failed`/`Cancelled`/`Blocked`) — the
/// upstream can make no further execution progress — so notify every
/// dependent according to its edge policy: satisfied edges (`Terminal` and
/// `Always` are both satisfied by a `Blocked` upstream; the upstream is
/// terminal either way) decrement the pending-count, dead `Success` edges
/// mark the dependent `Blocked` and propagate that transitively. `Always`
/// edges through a blocked chain still fire immediately (cleanup); a
/// `Terminal`-edge dependent behind a blocked chain stays pending until the
/// graph is rebuilt from live state, which also releases it (a blocked task
/// never produces a terminal event of its own, so this rebuild is the only
/// release path for it).
fn terminalize(guard: &mut std::sync::MutexGuard<'_, Inner>, upstream_id: OpId) {
    // Ownership is released the moment the op reaches any terminal outcome.
    guard.running.remove(&upstream_id);
    let upstream = guard.tasks[&upstream_id].status;
    let dependents: Vec<(OpId, DependencyPolicy)> = guard
        .tasks
        .get_mut(&upstream_id)
        .map(|t| std::mem::take(&mut t.dependents))
        .unwrap_or_default();
    let mut worklist: Vec<OpId> = Vec::new();
    for (dep_id, policy) in dependents {
        if edge_satisfied(policy, upstream) {
            if let Some(t) = guard.tasks.get_mut(&dep_id) {
                t.remaining = t.remaining.saturating_sub(1);
                if t.remaining == 0 && t.blocked == 0 && t.status == TaskStatus::Pending {
                    guard.ready.push_back(dep_id);
                }
            }
        } else if policy == DependencyPolicy::Success {
            if let Some(t) = guard.tasks.get_mut(&dep_id) {
                t.blocked += 1;
                if t.status == TaskStatus::Pending {
                    t.status = TaskStatus::Blocked;
                    worklist.push(dep_id);
                }
            }
        }
    }
    // Transitive blocking: a newly blocked task blocks its Success-edge
    // dependents, fires its Always-edge dependents, and leaves Terminal-edge
    // dependents pending.
    while let Some(id) = worklist.pop() {
        let subs: Vec<(OpId, DependencyPolicy)> = guard
            .tasks
            .get(&id)
            .map(|t| t.dependents.clone())
            .unwrap_or_default();
        for (dep_id, policy) in subs {
            match policy {
                DependencyPolicy::Always => {
                    if let Some(t) = guard.tasks.get_mut(&dep_id) {
                        t.remaining = t.remaining.saturating_sub(1);
                        if t.remaining == 0 && t.blocked == 0 && t.status == TaskStatus::Pending {
                            guard.ready.push_back(dep_id);
                        }
                    }
                }
                DependencyPolicy::Success => {
                    if let Some(t) = guard.tasks.get_mut(&dep_id) {
                        t.blocked += 1;
                        if t.status == TaskStatus::Pending {
                            t.status = TaskStatus::Blocked;
                            worklist.push(dep_id);
                        }
                    }
                }
                DependencyPolicy::Terminal => {}
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// Why a call was denied by a circuit breaker. The classification is
/// `CircuitOpen` — NEVER `Deadlock`: an open circuit is a resource-health
/// signal (provider/model, MCP server, host), while a deadlock is a
/// scheduler-graph problem with its own recovery machinery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakerStatus {
    /// The circuit for the resource is open (cooldown not yet elapsed), or
    /// a half-open probe is already claimed by another caller.
    CircuitOpen,
}

/// Default breaker policy for resources with no explicit configuration.
pub const DEFAULT_BREAKER_FAILURE_THRESHOLD: u32 = 4;
pub const DEFAULT_BREAKER_COOLDOWN_MS: u64 = 5_000;

/// Hard structural bounds for ONE DAG. Hostile submissions are rejected at
/// submit time — they can never explode the scheduler's memory or topology.
pub const MAX_TASKS_PER_DAG: usize = 1024;
pub const MAX_DEPENDENCIES_PER_TASK: usize = 64;
pub const MAX_OWNERSHIP_PATHS: usize = 512;

/// One circuit breaker scoped to ONE resource. The caller chooses the
/// resource string — provider/model endpoint (`anthropic:opus`), MCP server
/// name, tool kind, or host — so a degraded resource trips its own breaker
/// without poisoning unrelated work. Never key by (session, operation): a
/// per-operation breaker forgets every previous failure, so a storm of
/// distinct operations against one broken resource never trips.
///
/// Behavior: opens after `failure_threshold` consecutive failures; after
/// `cooldown_ms` the circuit decays by admitting EXACTLY ONE probe
/// atomically (AtomicU8 CAS Open→HalfOpen — the winner runs, all other
/// concurrent callers are denied); probe success closes the circuit, probe
/// failure re-opens it for a fresh cooldown.
#[derive(Debug)]
pub struct CircuitBreaker {
    failure_threshold: u32,
    cooldown_ms: u64,
    state: AtomicU8,
    failures: AtomicU32,
    opened_at_ms: AtomicI64,
}

const STATE_CLOSED: u8 = 0;
const STATE_OPEN: u8 = 1;
const STATE_HALF_OPEN: u8 = 2;

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, cooldown_ms: u64) -> Self {
        Self {
            failure_threshold: failure_threshold.max(1),
            cooldown_ms,
            state: AtomicU8::new(STATE_CLOSED),
            failures: AtomicU32::new(0),
            opened_at_ms: AtomicI64::new(0),
        }
    }

    pub fn state(&self) -> CircuitState {
        match self.state.load(Ordering::SeqCst) {
            STATE_CLOSED => CircuitState::Closed,
            STATE_OPEN => CircuitState::Open,
            _ => CircuitState::HalfOpen,
        }
    }

    /// Consecutive failures recorded while closed (diagnostics).
    pub fn failures(&self) -> u32 {
        self.failures.load(Ordering::SeqCst)
    }

    /// May a call against this resource proceed?
    ///
    /// - `Closed` → `Ok(())`.
    /// - `Open` with the cooldown elapsed → atomically claim the single
    ///   half-open probe (CAS `Open → HalfOpen`): the winning caller gets
    ///   `Ok(())` and runs the probe; every other concurrent caller is
    ///   denied.
    /// - `Open` in cooldown, or `HalfOpen` (probe already claimed) →
    ///   `Err(BreakerStatus::CircuitOpen)`.
    pub fn allow(&self, now_ms: i64) -> Result<(), BreakerStatus> {
        loop {
            match self.state.load(Ordering::SeqCst) {
                STATE_CLOSED => return Ok(()),
                STATE_HALF_OPEN => return Err(BreakerStatus::CircuitOpen),
                _ => {
                    let opened = self.opened_at_ms.load(Ordering::SeqCst);
                    if now_ms.saturating_sub(opened) < self.cooldown_ms as i64 {
                        return Err(BreakerStatus::CircuitOpen);
                    }
                    if self
                        .state
                        .compare_exchange(
                            STATE_OPEN,
                            STATE_HALF_OPEN,
                            Ordering::SeqCst,
                            Ordering::SeqCst,
                        )
                        .is_ok()
                    {
                        return Ok(()); // this caller owns the single probe
                    }
                    continue; // another caller claimed the probe first
                }
            }
        }
    }

    /// A successful call — or a successful probe — closes the circuit and
    /// resets the failure streak.
    pub fn record_success(&self) {
        self.failures.store(0, Ordering::SeqCst);
        self.state.store(STATE_CLOSED, Ordering::SeqCst);
    }

    /// A failed call. `Closed` → count toward the threshold; at the
    /// threshold the circuit opens. `HalfOpen` → the probe failed: re-open
    /// for a fresh cooldown. `Open` → stays open (cooldown from the first
    /// trip).
    pub fn record_failure(&self, now_ms: i64) {
        match self.state() {
            CircuitState::Closed => {
                let seen = self.failures.fetch_add(1, Ordering::SeqCst) + 1;
                if seen >= self.failure_threshold {
                    self.state.store(STATE_OPEN, Ordering::SeqCst);
                    self.opened_at_ms.store(now_ms, Ordering::SeqCst);
                }
            }
            CircuitState::HalfOpen => {
                self.state.store(STATE_OPEN, Ordering::SeqCst);
                self.opened_at_ms.store(now_ms, Ordering::SeqCst);
            }
            CircuitState::Open => {}
        }
    }

    /// Force the circuit open now (host/endpoint reported dead out of band).
    pub fn open(&self, now_ms: i64) {
        self.failures
            .store(self.failure_threshold, Ordering::SeqCst);
        self.state.store(STATE_OPEN, Ordering::SeqCst);
        self.opened_at_ms.store(now_ms, Ordering::SeqCst);
    }
}

/// The identity of one external resource a call can fail against. Circuit
/// breakers are keyed by THIS, never by `(session, resource class)`: a
/// session- and class-scoped breaker cannot trip across sessions (one
/// broken provider must refuse every session's traffic at once) and it
/// conflates independent resources that merely share a class (two models of
/// one provider would poison each other).
///
/// Variants mirror the callable resources the runtime knows: a provider
/// deployment (`instance` = the deployment/alias owning the key, `model`,
/// `endpoint` = the API endpoint), an MCP server, a tool kind, or a host.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize)]
pub enum ResourceKey {
    Provider {
        instance: String,
        model: String,
        endpoint: String,
    },
    Mcp {
        server: String,
    },
    Tool {
        name: String,
    },
    Host {
        scheme: String,
        host: String,
        port: u16,
    },
}

impl std::fmt::Display for ResourceKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResourceKey::Provider {
                instance,
                model,
                endpoint,
            } => write!(f, "provider:{instance}:{model}:{endpoint}"),
            ResourceKey::Mcp { server } => write!(f, "mcp:{server}"),
            ResourceKey::Tool { name } => write!(f, "tool:{name}"),
            ResourceKey::Host { scheme, host, port } => write!(f, "host:{scheme}://{host}:{port}"),
        }
    }
}

/// Default key cap of a [`CircuitBoard`] (configurable via
/// [`CircuitBoard::with_max_keys`]): the board never grows unbounded, and
/// the least recently consulted key is evicted (LRU) past the cap.
pub const DEFAULT_CIRCUIT_BOARD_MAX_KEYS: usize = 4096;

/// Resource-keyed circuit breakers — one board is the DAEMON-LEVEL object:
/// cloneable and shareable across every session's [`Scheduler`], so one
/// broken provider trips the SAME breaker for all sessions and healthy
/// resources are never poisoned by an unrelated one. Keys are chosen by the
/// CALLER from [`ResourceKey`] (provider/model/endpoint, MCP server, tool
/// kind, host) — never derived from `(session, resource class)`, which
/// cannot express what actually failed and lets resources of one class
/// interfere with each other.
///
/// Bounded: at most `max_keys` breakers are cached. The least recently
/// consulted key is evicted (LRU, recency = last [`CircuitBoard::breaker`]
/// touch); an evicted key simply gets a fresh closed breaker on its next
/// use. Breaker state itself (`CircuitBreaker`) is per-key atomic state
/// consulted without the board lock, so board traffic contends only on key
/// lookup/creation.
#[derive(Debug, Clone)]
pub struct CircuitBoard {
    inner: Arc<BoardInner>,
}

#[derive(Debug)]
struct BoardInner {
    state: Mutex<BoardState>,
}

#[derive(Debug)]
struct BoardState {
    max_keys: usize,
    /// Monotonic recency tick; `last_used` = tick of the most recent
    /// [`CircuitBoard::breaker`] touch.
    tick: u64,
    map: HashMap<ResourceKey, (Arc<CircuitBreaker>, u64)>,
}

impl Default for CircuitBoard {
    fn default() -> Self {
        Self::new()
    }
}

impl CircuitBoard {
    pub fn new() -> Self {
        Self::with_max_keys(DEFAULT_CIRCUIT_BOARD_MAX_KEYS)
    }

    /// A board bounded to `max_keys` breakers (never unbounded; `0` is
    /// clamped to 1 so the board always admits at least its newest key).
    pub fn with_max_keys(max_keys: usize) -> Self {
        Self {
            inner: Arc::new(BoardInner {
                state: Mutex::new(BoardState {
                    max_keys: max_keys.max(1),
                    tick: 0,
                    map: HashMap::new(),
                }),
            }),
        }
    }

    /// The breaker for `resource`, created on first use. Every consult
    /// refreshes the key's LRU recency; inserting a new key past the cap
    /// evicts the least recently consulted key first.
    pub fn breaker(&self, resource: &ResourceKey) -> Arc<CircuitBreaker> {
        let mut guard = self.inner.state.lock().unwrap();
        guard.tick += 1;
        let tick = guard.tick;
        if let Some((breaker, last_used)) = guard.map.get_mut(resource) {
            *last_used = tick;
            return breaker.clone();
        }
        if guard.map.len() >= guard.max_keys {
            // LRU eviction: drop the least recently consulted key. A
            // dropped breaker's state is gone with it — the key re-enters
            // closed on its next use.
            let victim = guard
                .map
                .iter()
                .min_by_key(|(_, (_, last_used))| *last_used)
                .map(|(k, _)| k.clone());
            if let Some(victim) = victim {
                guard.map.remove(&victim);
            }
        }
        let breaker = Arc::new(CircuitBreaker::new(
            DEFAULT_BREAKER_FAILURE_THRESHOLD,
            DEFAULT_BREAKER_COOLDOWN_MS,
        ));
        guard.map.insert(resource.clone(), (breaker.clone(), tick));
        breaker
    }

    /// May a call against `resource` proceed? See [`CircuitBreaker::allow`].
    pub fn allow(&self, resource: &ResourceKey, now_ms: i64) -> Result<(), BreakerStatus> {
        self.breaker(resource).allow(now_ms)
    }

    /// Force-open the breaker for `resource` (host/endpoint reported dead).
    pub fn open(&self, resource: &ResourceKey, now_ms: i64) {
        self.breaker(resource).open(now_ms);
    }

    pub fn record_success(&self, resource: &ResourceKey) {
        self.breaker(resource).record_success();
    }

    pub fn record_failure(&self, resource: &ResourceKey, now_ms: i64) {
        self.breaker(resource).record_failure(now_ms);
    }

    pub fn state(&self, resource: &ResourceKey) -> CircuitState {
        self.breaker(resource).state()
    }

    /// Number of breakers currently cached (bounded by `max_keys`).
    pub fn len(&self) -> usize {
        self.inner.state.lock().unwrap().map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The configured cap on cached breakers.
    pub fn max_keys(&self) -> usize {
        self.inner.state.lock().unwrap().max_keys
    }
}

#[derive(Default)]
struct Inner {
    tasks: HashMap<OpId, TaskState>,
    gauge: ResourceGauge,
    /// FIFO of tasks whose dependencies are satisfied and that still need a
    /// permit. Tasks deferred for a busy budget cycle back to the tail.
    ready: VecDeque<OpId>,
    /// Ops currently executing (ownership sets), for overlap serialization:
    /// a ready op whose writes overlap a running op's reads/writes is
    /// deferred until that op completes (audit round 5).
    running: HashMap<OpId, (OwnershipSet, OwnershipSet)>,
}

/// A scheduler for one session. Clonable handle; all execution goes through
/// shared state, so concurrent task execution and status reads are safe.
#[derive(Clone)]
pub struct Scheduler {
    session_id: SessionId,
    limits: Arc<ResourceLimits>,
    inner: Arc<Mutex<Inner>>,
    /// The daemon-level circuit board this scheduler executes against:
    /// share ONE board across schedulers (sessions) so a broken provider
    /// trips for every session at once. Breaker keys are
    /// [`ResourceKey`]s chosen by the caller for what ACTUALLY fails
    /// (provider/model/endpoint, MCP server, tool kind, host) — never
    /// derived from `(session, resource class)`, which cannot express the
    /// failing resource and lets resources of one class poison each other.
    /// Resource-CLASS LIMITS (concurrency ceilings) stay the separate
    /// `ResourceLimits`/`ResourceGauge` machinery, untouched by breakers.
    circuits: CircuitBoard,
    clock: Arc<dyn faktor_core::time::Clock>,
    /// Bounded record of registrations rejected through the legacy
    /// [`Scheduler::submit`] compat entry point, so a rejection is
    /// observable even by callers that cannot propagate the error.
    rejections: Arc<Mutex<VecDeque<(OpId, String)>>>,
}

/// How many rejections the legacy [`Scheduler::submit`] shim keeps for
/// [`Scheduler::rejections`] (bounded; the oldest is dropped past this).
const MAX_RECORDED_REJECTIONS: usize = 64;

impl Scheduler {
    pub fn new(session_id: SessionId, clock: Arc<dyn faktor_core::time::Clock>) -> Self {
        Self {
            session_id,
            limits: Arc::new(ResourceLimits::default()),
            inner: Arc::new(Mutex::new(Inner::default())),
            circuits: CircuitBoard::new(),
            clock,
            rejections: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    pub fn with_limits(self, limits: ResourceLimits) -> Self {
        Self {
            limits: Arc::new(limits),
            ..self
        }
    }

    /// Execute against a SHARED, daemon-level [`CircuitBoard`]. Two
    /// schedulers (two sessions) given the same board trip the same
    /// breaker for the same [`ResourceKey`]: session A breaking provider P
    /// refuses session B's P traffic too, while healthy resources proceed.
    pub fn with_circuits(self, circuits: CircuitBoard) -> Self {
        Self { circuits, ..self }
    }

    /// The session this scheduler serves. Circuit-breaker state is
    /// deliberately NOT session-scoped: it lives on the shared daemon
    /// board (see [`Scheduler::circuits`]).
    pub fn session_id(&self) -> SessionId {
        self.session_id
    }

    /// The resource-keyed circuit board this scheduler executes against.
    /// Keys are the failing [`ResourceKey`]s (provider/model/endpoint, MCP
    /// server, tool kind, host), never a fresh operation id — a
    /// per-operation breaker forgets every previous failure.
    pub fn circuits(&self) -> &CircuitBoard {
        &self.circuits
    }

    /// Submit one op through the legacy compat entry point.
    ///
    /// KEPT ONLY because an out-of-scope caller still uses the historical
    /// infallible signature (crates/agent/src/runtime.rs:3571); that call
    /// site is scheduled for migration to [`Scheduler::try_submit`] in the
    /// next wave. The silent-failure defect is gone: a rejected
    /// registration is (1) logged at error severity with the op id and
    /// reason and (2) recorded — bounded — for
    /// [`Scheduler::rejections`], so a dropped tool execution is never
    /// invisible. Do NOT use this entry point in new code; call
    /// `try_submit` and propagate the [`ErrorKind::Conflict`] / bounds
    /// errors explicitly.
    pub fn submit(&self, op: ScheduledOp) {
        let id = op.meta.operation_id;
        if let Err(e) = self.try_submit(op) {
            tracing::error!(op = %id, %e, "scheduled op rejected via legacy submit()");
            let mut ring = self.rejections.lock().unwrap();
            if ring.len() == MAX_RECORDED_REJECTIONS {
                ring.pop_front();
            }
            ring.push_back((id, e.message));
        }
    }

    /// Rejections recorded by the legacy [`Scheduler::submit`] compat shim
    /// (bounded to the most recent [`MAX_RECORDED_REJECTIONS`]), newest
    /// last. Production code that calls `try_submit` and propagates errors
    /// never produces entries here.
    pub fn rejections(&self) -> Vec<(OpId, String)> {
        self.rejections.lock().unwrap().iter().cloned().collect()
    }

    /// The audited submission API. Conflict semantics: registering the SAME
    /// op twice with an identical payload is idempotent and returns
    /// `Ok(())`; reusing an existing op id with a DIFFERENT payload is a
    /// [`ErrorKind::Conflict`] error and never overwrites the first
    /// registration. Structural bounds (DAG size, per-task dependencies,
    /// ownership paths) are enforced here too — hostile DAGs cannot
    /// explode the scheduler.
    pub fn try_submit(&self, op: ScheduledOp) -> Result<(), Error> {
        let op_len = op.dependencies.len();
        if op_len > MAX_DEPENDENCIES_PER_TASK {
            return Err(Error::oversized(format!(
                "op {} declares {op_len} dependencies; cap is {MAX_DEPENDENCIES_PER_TASK}",
                op.meta.operation_id
            )));
        }
        let paths = op.reads.entries().len() + op.writes.entries().len();
        if paths > MAX_OWNERSHIP_PATHS {
            return Err(Error::oversized(format!(
                "op {} declares {paths} ownership paths; cap is {MAX_OWNERSHIP_PATHS}",
                op.meta.operation_id
            )));
        }
        let id = op.meta.operation_id;
        let mut guard = self.inner.lock().unwrap();
        if let Some(existing) = guard.tasks.get(&id) {
            if same_registration(&existing.op, &op) {
                // Idempotent re-registration of an identical op is safe:
                // the task is already registered with this exact payload.
                return Ok(());
            }
            return Err(Error::conflict(format!(
                "op {id} already submitted with a different payload"
            )));
        }
        if guard.tasks.len() >= MAX_TASKS_PER_DAG {
            return Err(Error::oversized(format!(
                "dag already holds {} tasks; cap is {MAX_TASKS_PER_DAG}",
                guard.tasks.len()
            )));
        }
        let mut remaining = 0usize;
        let mut blocked = 0usize;
        for (dep_id, policy) in &op.dependencies {
            let dep_status = guard.tasks.get(dep_id).map(|t| t.status);
            match dep_status {
                None => remaining += 1, // missing dep; validate() reports it
                Some(status) => {
                    if edge_satisfied(*policy, status) {
                        // Edge already satisfied; not counted anywhere.
                    } else if *policy == DependencyPolicy::Success && success_edge_dead(status) {
                        blocked += 1; // dead edge: this task can never run
                    } else {
                        remaining += 1;
                        guard
                            .tasks
                            .get_mut(dep_id)
                            .unwrap()
                            .dependents
                            .push((id, *policy));
                    }
                }
            }
        }
        let status = if blocked > 0 {
            TaskStatus::Blocked
        } else {
            TaskStatus::Pending
        };
        guard.tasks.insert(
            id,
            TaskState {
                op,
                status,
                error: None,
                start_ms: 0,
                end_ms: None,
                remaining,
                blocked,
                dependents: Vec::new(),
            },
        );
        if remaining == 0 && blocked == 0 && status == TaskStatus::Pending {
            guard.ready.push_back(id);
        }
        Ok(())
    }

    pub fn status(&self, id: OpId) -> Option<TaskStatus> {
        self.inner.lock().unwrap().tasks.get(&id).map(|t| t.status)
    }

    /// Validate the DAG before running: structural bounds (hostile DAGs),
    /// unknown dependencies and cycles are loud errors, never silent
    /// deadlocks. Per-op bounds are enforced at submit time as well.
    pub fn validate(&self) -> Result<(), Error> {
        let tasks = &self.inner.lock().unwrap().tasks;
        if tasks.len() > MAX_TASKS_PER_DAG {
            return Err(Error::oversized(format!(
                "dag holds {} tasks; cap is {MAX_TASKS_PER_DAG}",
                tasks.len()
            )));
        }
        for (id, t) in tasks {
            let deps = t.op.dependencies.len();
            if deps > MAX_DEPENDENCIES_PER_TASK {
                return Err(Error::oversized(format!(
                    "task {id} declares {deps} dependencies; cap is {MAX_DEPENDENCIES_PER_TASK}"
                )));
            }
            let paths = t.op.reads.entries().len() + t.op.writes.entries().len();
            if paths > MAX_OWNERSHIP_PATHS {
                return Err(Error::oversized(format!(
                    "task {id} declares {paths} ownership paths; cap is {MAX_OWNERSHIP_PATHS}"
                )));
            }
        }
        let mut visiting: HashSet<OpId> = HashSet::new();
        let mut visited: HashSet<OpId> = HashSet::new();
        for id in tasks.keys() {
            visit_dag(*id, tasks, &mut visiting, &mut visited)?;
        }
        Ok(())
    }

    /// Run the whole DAG to completion. Event-driven, no wave barrier: the
    /// ready queue is drained as permits free, and dependents are scheduled
    /// the instant their last dependency completes. Returns the ids that
    /// finished Done.
    pub async fn run_to_completion(&self) -> Result<Vec<OpId>, Error> {
        self.validate()?;
        self.rebuild_graph();
        let mut set: tokio::task::JoinSet<OpId> = tokio::task::JoinSet::new();
        let mut rebuilt_once = false;
        loop {
            self.schedule_ready(&mut set);
            if set.is_empty() {
                let any_pending = self
                    .inner
                    .lock()
                    .unwrap()
                    .tasks
                    .values()
                    .any(|t| t.status == TaskStatus::Pending);
                if !any_pending {
                    break;
                }
                if !rebuilt_once {
                    // Late submits or stale graph state: rebuild once from
                    // live state before declaring a deadlock.
                    rebuilt_once = true;
                    self.rebuild_graph();
                    continue;
                }
                let ready_stuck = self
                    .inner
                    .lock()
                    .unwrap()
                    .tasks
                    .values()
                    .any(|t| t.status == TaskStatus::Pending && t.remaining == 0);
                if ready_stuck {
                    return Err(Error::new(
                        ErrorKind::Deadlock,
                        "resource budgets prevent any ready task from starting",
                    ));
                }
                return Err(Error::new(
                    ErrorKind::Deadlock,
                    "no ready tasks and work remains; cycle or unscheduled dependency",
                ));
            }
            // Wait for ANY task to finish, then re-drain the ready queue:
            // short tasks that became ready meanwhile start before long ones
            // that are still running.
            let joined = set.join_next().await;
            match joined {
                Some(Ok(id)) => self.on_complete(id),
                Some(Err(_)) => {
                    return Err(Error::new(
                        ErrorKind::Internal,
                        "scheduler task failed unexpectedly",
                    ))
                }
                None => unreachable!("JoinSet was non-empty"),
            }
        }
        let mut done: Vec<OpId> = {
            let guard = self.inner.lock().unwrap();
            guard
                .tasks
                .iter()
                .filter(|(_, t)| t.status == TaskStatus::Done)
                .map(|(id, _)| *id)
                .collect()
        };
        done.sort();
        Ok(done)
    }

    /// Drain the ready queue: tasks whose dependencies are satisfied and
    /// that can acquire a resource permit are spawned into `set`; the permit
    /// is held BEFORE spawn and released only after the task completes.
    fn schedule_ready(&self, set: &mut tokio::task::JoinSet<OpId>) {
        loop {
            let mut to_spawn: Vec<(OpId, ScheduledOp)> = Vec::new();
            {
                let mut guard = self.inner.lock().unwrap();
                let n = guard.ready.len();
                let mut deferred = 0usize;
                while deferred < n && !guard.ready.is_empty() {
                    let id = guard.ready.pop_front().unwrap();
                    let class = {
                        let t = guard.tasks.get_mut(&id).unwrap();
                        if t.status != TaskStatus::Pending || t.remaining != 0 {
                            None // stale queue entry; drop it
                        } else {
                            Some(t.op.resources.class)
                        }
                    };
                    let Some(class) = class else { continue };
                    // Ownership-overlap serialization (audit round 5).
                    // Symmetric about overlapping access: a ready op is
                    // deferred when its WRITES overlap a running op's reads
                    // OR writes, or its READS overlap a running op's writes.
                    // Reads-vs-reads and disjoint writes stay concurrent.
                    // (An asymmetric check would let a writer start, then a
                    // reader run over it concurrently.)
                    let t = guard.tasks.get(&id).unwrap();
                    let conflicts = guard.running.values().any(|(rr, ww)| {
                        t.op.writes.overlaps(ww)
                            || t.op.writes.overlaps(rr)
                            || t.op.reads.overlaps(ww)
                    });
                    if conflicts {
                        guard.ready.push_back(id); // retry when a running op completes
                        deferred += 1;
                        continue;
                    }
                    if guard.gauge.try_acquire(class, &self.limits).is_err() {
                        guard.ready.push_back(id); // retry when a permit frees
                        deferred += 1;
                        continue;
                    }
                    let reads;
                    let writes;
                    {
                        let t = guard.tasks.get_mut(&id).unwrap();
                        t.status = TaskStatus::Running;
                        t.start_ms = self.clock.now_ms();
                        reads = t.op.reads.clone();
                        writes = t.op.writes.clone();
                    }
                    guard.running.insert(id, (reads, writes));
                    to_spawn.push((id, guard.tasks[&id].op.clone()));
                }
            }
            if to_spawn.is_empty() {
                return;
            }
            for (id, op) in to_spawn {
                let sched = self.clone();
                set.spawn(async move {
                    // A panicking runnable must not wedge the budget: catch
                    // it here, mark the task failed, and still release.
                    // DAG tasks carry no ResourceKey, so they run without a
                    // circuit breaker (see `execute_inner`).
                    let result = futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(
                        sched.execute_inner(id, op, None),
                    ))
                    .await;
                    if result.is_err() {
                        sched.mark(id, TaskStatus::Failed, Some("runnable panicked".into()));
                    }
                    id
                });
            }
        }
    }

    /// A task completed (any outcome): free its permit and notify every
    /// dependent — satisfied edges decrement the pending-count (dependents
    /// that reach zero join the ready queue immediately), dead `Success`
    /// edges block the dependent transitively.
    fn on_complete(&self, id: OpId) {
        let mut guard = self.inner.lock().unwrap();
        let class = guard.tasks[&id].op.resources.class;
        guard.gauge.release(class);
        terminalize(&mut guard, id);
    }

    /// Recompute the dependency graph from live state (dependents, unmet
    /// counts, blocked marks, ready queue). Statuses are snapshotted first so
    /// the result is independent of iteration order; tasks that become
    /// `Blocked` here propagate to their own dependents afterwards.
    fn rebuild_graph(&self) {
        let mut guard = self.inner.lock().unwrap();
        guard.ready.clear();
        let statuses: HashMap<OpId, TaskStatus> =
            guard.tasks.iter().map(|(id, t)| (*id, t.status)).collect();
        for t in guard.tasks.values_mut() {
            t.remaining = 0;
            t.blocked = 0;
            t.dependents.clear();
        }
        let ids: Vec<OpId> = guard.tasks.keys().copied().collect();
        let mut newly_blocked: Vec<OpId> = Vec::new();
        for id in &ids {
            let deps: Vec<(OpId, DependencyPolicy)> = guard.tasks[id].op.dependencies.clone();
            let mut remaining = 0usize;
            let mut blocked = 0usize;
            for (dep_id, policy) in deps {
                let dep_status = statuses[&dep_id];
                if edge_satisfied(policy, dep_status) {
                    continue;
                }
                if policy == DependencyPolicy::Success && success_edge_dead(dep_status) {
                    blocked += 1;
                } else {
                    remaining += 1;
                    guard
                        .tasks
                        .get_mut(&dep_id)
                        .unwrap()
                        .dependents
                        .push((*id, policy));
                }
            }
            let t = guard.tasks.get_mut(id).unwrap();
            t.remaining = remaining;
            t.blocked = blocked;
            if blocked > 0 && t.status == TaskStatus::Pending {
                t.status = TaskStatus::Blocked;
                newly_blocked.push(*id);
            }
        }
        for id in newly_blocked {
            terminalize(&mut guard, id);
        }
        for id in &ids {
            let t = &guard.tasks[id];
            if t.status == TaskStatus::Pending && t.remaining == 0 && t.blocked == 0 {
                guard.ready.push_back(*id);
            }
        }
    }

    /// Execute one caller-declared RESOURCE call (`resource` names the
    /// failing provider/model/endpoint, MCP server, tool kind, or host the
    /// call runs against) with retry-with-jitter, deadline, cancellation,
    /// and circuit breaking on that resource's breaker. Acquires the
    /// resource-class slot first (or returns `BudgetBusy`), updates shared
    /// state, and always releases the slot.
    ///
    /// Circuit semantics: breaker state is SHARED by [`ResourceKey`] across
    /// every scheduler that holds the same [`CircuitBoard`] — one broken
    /// provider refuses fast for every session, and independent resources
    /// of the same class never share a breaker. DAG-managed tasks
    /// (`run_to_completion`) carry no resource identity and run without a
    /// breaker; the runtime gates those through the board itself.
    ///
    /// Terminal exactly-once (audit 79-80): when `id` is already registered
    /// AND terminal, the execution is rejected with a typed conflict error —
    /// never a silent second terminal event. Unregistered ids keep the
    /// standalone-execution contract (a caller-owned op outside the DAG).
    pub async fn execute(
        &self,
        id: OpId,
        op: ScheduledOp,
        resource: &ResourceKey,
    ) -> Result<(), ExecuteError> {
        let terminal = self
            .inner
            .lock()
            .unwrap()
            .tasks
            .get(&id)
            .map(|t| status_is_terminal(t.status))
            .unwrap_or(false);
        if terminal {
            let msg =
                format!("op {id} is already terminal; refusing to execute a second terminal event");
            tracing::warn!(%msg, "execute on terminal op rejected");
            return Err(ExecuteError::Err(Error::conflict(msg)));
        }
        let class = op.resources.class;
        let acquired = {
            let mut guard = self.inner.lock().unwrap();
            guard.gauge.try_acquire(class, &self.limits).is_ok()
        };
        if !acquired {
            return Err(ExecuteError::Busy(BudgetBusy(class)));
        }
        let breaker = self.circuits.breaker(resource);
        let result = self.execute_inner(id, op, Some(breaker.as_ref())).await;
        self.release(class);
        result
    }

    /// The actual execution loop. Assumes the budget slot is already held.
    /// Circuit breaking (when `breaker` is supplied) is scoped to the
    /// caller-declared [`ResourceKey`] of the failing resource — the SAME
    /// breaker every session with the shared daemon board consults. DAG
    /// executions pass `None`: a DAG task carries no resource identity, so
    /// it can neither claim a key nor poison one (the runtime gates DAG
    /// tool calls through the board directly, before submitting them).
    async fn execute_inner(
        &self,
        id: OpId,
        op: ScheduledOp,
        breaker: Option<&CircuitBreaker>,
    ) -> Result<(), ExecuteError> {
        let mut attempt = 0u32;
        loop {
            let now = self.clock.now_ms();
            // Cancellation and deadline come from the op's own metadata.
            if op.meta.cancellation.is_cancelled() {
                self.mark(id, TaskStatus::Cancelled, None);
                return Ok(());
            }
            if op.meta.deadline.is_expired(now) {
                let msg = format!("op {id} deadline exceeded");
                self.mark(id, TaskStatus::Failed, Some(msg.clone()));
                return Err(ExecuteError::Err(Error::timeout(msg)));
            }
            if let Some(breaker) = breaker {
                if breaker.allow(now).is_err() {
                    // BreakerStatus::CircuitOpen — a resource-health denial,
                    // NEVER a Deadlock classification.
                    let msg = "circuit open for this resource, not attempting".to_string();
                    self.mark(id, TaskStatus::Failed, Some(msg.clone()));
                    return Err(ExecuteError::Err(Error::new(ErrorKind::Internal, msg)));
                }
            }
            let remaining_ms = (op.meta.deadline.at_ms() - now).max(1) as u64;
            let run = op.run.clone();
            let fut = run();
            let result = tokio::time::timeout(Duration::from_millis(remaining_ms), fut).await;
            match result {
                Ok(Ok(())) => {
                    if op.meta.cancellation.is_cancelled() {
                        self.mark(id, TaskStatus::Cancelled, None);
                        return Ok(());
                    }
                    self.mark(id, TaskStatus::Done, None);
                    if let Some(breaker) = breaker {
                        breaker.record_success();
                    }
                    return Ok(());
                }
                Ok(Err(e)) => {
                    if op.meta.cancellation.is_cancelled() {
                        self.mark(id, TaskStatus::Cancelled, None);
                        return Ok(());
                    }
                    attempt += 1;
                    if let Some(breaker) = breaker {
                        breaker.record_failure(self.clock.now_ms());
                    }
                    let retryable =
                        e.retryable && op.meta.retry_policy.should_retry(attempt - 1, true, false);
                    if !retryable {
                        self.mark(id, TaskStatus::Failed, Some(e.message.clone()));
                        return Err(ExecuteError::Err(e));
                    }
                    let delay = op.meta.retry_policy.next_delay(attempt - 1);
                    tokio::time::sleep(delay).await;
                }
                Err(_elapsed) => {
                    if let Some(breaker) = breaker {
                        breaker.record_failure(self.clock.now_ms());
                    }
                    let msg = format!("op {id} deadline exceeded");
                    self.mark(id, TaskStatus::Failed, Some(msg.clone()));
                    return Err(ExecuteError::Err(Error::timeout(msg)));
                }
            }
        }
    }

    /// Cancel a task: the op's own cancellation token fires (so a running
    /// runnable that polls it aborts promptly) and pending tasks flip to
    /// Cancelled immediately, notifying their dependents (Success-edge
    /// dependents block, Terminal/Always-edge dependents may run).
    ///
    /// Legacy infallible entry point: delegates to
    /// [`Scheduler::try_cancel`]; a rejection (unknown op, or an
    /// already-terminal op — cancellation is exactly-once and must never
    /// fire a second terminal event) is logged, never silent, never a
    /// state flip.
    pub fn cancel(&self, id: OpId) {
        if let Err(e) = self.try_cancel(id) {
            tracing::warn!(%e, "cancel rejected");
        }
    }

    /// The audited cancellation entry point (model-checked, audit 79-80):
    ///
    /// - unknown op → [`ErrorKind::NotFound`];
    /// - already-terminal op (`Done`/`Failed`/`Cancelled`/`Blocked`) →
    ///   [`ErrorKind::Conflict`]: a terminal state is exactly-once and a
    ///   second terminal event is rejected, never a silent no-op that
    ///   resurrects or re-flips the state;
    /// - `Pending` → the task flips to `Cancelled` immediately and its
    ///   dependents are notified;
    /// - `Running` → the cancellation token fires; the running op observes
    ///   it at its next poll point and terminalizes `Cancelled` exactly
    ///   once.
    pub fn try_cancel(&self, id: OpId) -> Result<(), Error> {
        let mut guard = self.inner.lock().unwrap();
        let Some(t) = guard.tasks.get(&id) else {
            return Err(Error::not_found(format!("op {id} is not registered")));
        };
        if status_is_terminal(t.status) {
            return Err(Error::conflict(format!(
                "op {id} is already terminal in state {:?}; cancellation is \
                 exactly-once and cannot fire a second terminal event",
                t.status
            )));
        }
        let t = guard.tasks.get_mut(&id).unwrap();
        t.op.meta.cancellation.cancel();
        if t.status == TaskStatus::Pending {
            t.status = TaskStatus::Cancelled;
            terminalize(&mut guard, id);
        }
        Ok(())
    }

    pub fn statuses(&self) -> Vec<(OpId, TaskStatus)> {
        let mut v: Vec<_> = self
            .inner
            .lock()
            .unwrap()
            .tasks
            .iter()
            .map(|(id, t)| (*id, t.status))
            .collect();
        v.sort_by_key(|(id, _)| *id);
        v
    }

    /// Record a terminal (or Running-phase) state. Terminal exactly-once
    /// (audit 79-80): an already-terminal task is never overwritten — a
    /// second terminal event is refused loudly instead of silently flipping
    /// the state (duplicate completion, cancel-after-done resurrecting a
    /// live phase, ...). Non-terminal phase writes (Pending→Running phase
    /// bookkeeping is done in `schedule_ready`; error/end-time updates on a
    /// live task) are unaffected.
    fn mark(&self, id: OpId, status: TaskStatus, error: Option<String>) {
        let mut guard = self.inner.lock().unwrap();
        if let Some(t) = guard.tasks.get_mut(&id) {
            if status_is_terminal(t.status) {
                tracing::warn!(
                    op = %id,
                    from = ?t.status,
                    to = ?status,
                    "refusing a second terminal transition (terminal exactly-once)"
                );
                return;
            }
            t.status = status;
            t.error = error;
            t.end_ms = Some(self.clock.now_ms());
        }
    }

    fn release(&self, class: ResourceClass) {
        let mut guard = self.inner.lock().unwrap();
        guard.gauge.release(class);
    }
}

fn visit_dag(
    id: OpId,
    tasks: &HashMap<OpId, TaskState>,
    visiting: &mut HashSet<OpId>,
    visited: &mut HashSet<OpId>,
) -> Result<(), Error> {
    if visited.contains(&id) {
        return Ok(());
    }
    if visiting.contains(&id) {
        return Err(Error::new(
            ErrorKind::Deadlock,
            format!("dependency cycle detected at task {id}"),
        ));
    }
    visiting.insert(id);
    let task = tasks
        .get(&id)
        .ok_or_else(|| Error::not_found(format!("task {id} not found")))?;
    for (dep, _policy) in &task.op.dependencies {
        if !tasks.contains_key(dep) {
            return Err(Error::new(
                ErrorKind::NotFound,
                format!("task {id} depends on missing task {dep}"),
            ));
        }
        visit_dag(*dep, tasks, visiting, visited)?;
    }
    visiting.remove(&id);
    visited.insert(id);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use faktor_core::cancellation::CancellationToken;
    use faktor_core::hash::FileHash;
    use faktor_core::op::RecoveryStrategy;
    use faktor_core::retry::{RetryClass, RetryPolicy};
    use faktor_core::time::{Clock, Deadline, SystemClock};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    const FAR: i64 = i64::MAX / 2;

    /// Tests submit only ops that must register; a submit that fails here
    /// is a test bug, not a runtime path.
    fn submit(s: &Scheduler, op: ScheduledOp) {
        s.try_submit(op)
            .unwrap_or_else(|e| panic!("submit failed: {e}"));
    }

    /// A resource key for tests that execute a call without asserting on
    /// the resource identity itself.
    fn test_key() -> ResourceKey {
        ResourceKey::Tool {
            name: "test-tool".into(),
        }
    }

    fn task(
        id: u64,
        deps: Vec<u64>,
        class: ResourceClass,
        work_ms: u64,
        flag: Arc<AtomicUsize>,
    ) -> ScheduledOp {
        ScheduledOp {
            meta: OpMeta::new(
                OpId::new(id),
                SessionId::new(1),
                Deadline::at(FAR),
                RetryPolicy::default(),
                CancellationToken::new(),
                RecoveryStrategy::None,
                0,
            ),
            resources: ResourceRequest { class },
            reads: OwnershipSet::new([]),
            writes: OwnershipSet::new([]),
            dependencies: deps
                .into_iter()
                .map(|d| (OpId::new(d), DependencyPolicy::Success))
                .collect(),
            run: Arc::new(move || {
                let f = flag.clone();
                Box::pin(async move {
                    tokio::time::sleep(Duration::from_millis(work_ms)).await;
                    f.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                })
            }),
        }
    }

    fn err_task(id: u64, deps: Vec<u64>) -> ScheduledOp {
        ScheduledOp {
            meta: OpMeta::new(
                OpId::new(id),
                SessionId::new(1),
                Deadline::at(FAR),
                RetryPolicy::default(),
                CancellationToken::new(),
                RecoveryStrategy::None,
                0,
            ),
            resources: ResourceRequest {
                class: ResourceClass::Cpu,
            },
            reads: OwnershipSet::new([]),
            writes: OwnershipSet::new([]),
            dependencies: deps
                .into_iter()
                .map(|d| (OpId::new(d), DependencyPolicy::Success))
                .collect(),
            run: Arc::new(|| Box::pin(async { Err(Error::internal("boom")) })),
        }
    }

    /// A task with explicit per-edge policies, for the policy tests.
    fn policy_task(
        id: u64,
        deps: Vec<(u64, DependencyPolicy)>,
        class: ResourceClass,
        work_ms: u64,
        flag: Arc<AtomicUsize>,
    ) -> ScheduledOp {
        ScheduledOp {
            meta: OpMeta::new(
                OpId::new(id),
                SessionId::new(1),
                Deadline::at(FAR),
                RetryPolicy::default(),
                CancellationToken::new(),
                RecoveryStrategy::None,
                0,
            ),
            resources: ResourceRequest { class },
            reads: OwnershipSet::new([]),
            writes: OwnershipSet::new([]),
            dependencies: deps.into_iter().map(|(d, p)| (OpId::new(d), p)).collect(),
            run: Arc::new(move || {
                let f = flag.clone();
                Box::pin(async move {
                    tokio::time::sleep(Duration::from_millis(work_ms)).await;
                    f.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                })
            }),
        }
    }

    #[tokio::test]
    async fn dag_runs_dependencies_before_dependents() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let counter = Arc::new(AtomicUsize::new(0));
        submit(&s, task(1, vec![], ResourceClass::Cpu, 5, counter.clone()));
        submit(&s, task(2, vec![], ResourceClass::Cpu, 5, counter.clone()));
        submit(
            &s,
            task(3, vec![1, 2], ResourceClass::Cpu, 5, counter.clone()),
        );
        let done = s.run_to_completion().await.unwrap();
        assert_eq!(done.len(), 3);
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn overlapping_writes_serialize() {
        let a = OwnershipSet::new(["src/a.rs".to_string()]);
        let b = OwnershipSet::new(["src/a.rs".to_string()]);
        assert!(a.overlaps(&b));
        let c = OwnershipSet::new(["src/b.rs".to_string()]);
        assert!(!a.overlaps(&c));
        assert!(OwnershipSet::new([] as [String; 0]).is_empty());
        assert!(!OwnershipSet::new([] as [String; 0]).overlaps(&a));
    }

    #[test]
    fn directory_overlap() {
        let src = OwnershipSet::new(["src/".into()]);
        let a = OwnershipSet::new(["src/a.rs".into()]);
        let nested = OwnershipSet::new(["src/sub/b.rs".into()]);
        let lib = OwnershipSet::new(["lib/".into()]);
        let sibling = OwnershipSet::new(["src2/a.rs".into()]);
        let sub = OwnershipSet::new(["src/sub/".into()]);

        assert!(src.overlaps(&a), "dir must overlap a file below it");
        assert!(a.overlaps(&src), "overlap is symmetric");
        assert!(src.overlaps(&nested));
        assert!(nested.overlaps(&src));
        assert!(!src.overlaps(&lib));
        assert!(!lib.overlaps(&src));
        assert!(
            !src.overlaps(&sibling),
            "component boundary: src2 is not under src/"
        );
        assert!(!sibling.overlaps(&src));
        assert!(a.overlaps(&OwnershipSet::new(["src/a.rs".into()])));
        assert!(!a.overlaps(&OwnershipSet::new(["src/a.rs2".into()])));
        assert!(!a.overlaps(&OwnershipSet::new(["src/a.rs.bak".into()])));
        assert!(src.overlaps(&src));
        assert!(sub.overlaps(&nested));
        assert!(src.overlaps(&sub), "parent dir covers child dir");
        // A plain entry without a trailing slash is a file, not a directory.
        assert!(!OwnershipSet::new(["src".into()]).overlaps(&a));
        assert!(!OwnershipSet::new(["src".into()]).overlaps(&src));
        // Empty sets never overlap.
        assert!(!OwnershipSet::new([] as [String; 0]).overlaps(&src));
        assert!(
            !OwnershipSet::new([] as [String; 0]).overlaps(&OwnershipSet::new([] as [String; 0]))
        );
    }

    #[test]
    fn canonicalized_overlap() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path();
        std::fs::create_dir_all(base.join("src/sub")).unwrap();
        std::fs::create_dir_all(base.join("lib")).unwrap();
        std::fs::write(base.join("src/a.rs"), "a").unwrap();
        std::fs::write(base.join("src/sub/b.rs"), "b").unwrap();
        std::fs::write(base.join("lib/c.rs"), "c").unwrap();

        let src = OwnershipSet::new(["src/".into()]).canonicalized(base);
        let a = OwnershipSet::new(["src/a.rs".into()]).canonicalized(base);
        let nested = OwnershipSet::new(["src/sub/b.rs".into()]).canonicalized(base);
        let lib = OwnershipSet::new(["lib/".into()]).canonicalized(base);

        assert!(
            src.overlaps(&a),
            "canonicalized dir must cover files below it"
        );
        assert!(src.overlaps(&nested));
        assert!(
            !src.overlaps(&lib),
            "canonicalized dir must not cover sibling dirs"
        );
        assert!(a.overlaps(&OwnershipSet::new(["src/a.rs".into()]).canonicalized(base)));

        // Identical files spelled differently collapse to one canonical path.
        let dup =
            OwnershipSet::new(["src/a.rs".into(), "src/../src/a.rs".into()]).canonicalized(base);
        assert_eq!(
            dup.0.len(),
            1,
            "canonicalization must dedup spellings: {:?}",
            dup.0
        );

        // A path that does not exist falls back lexically and never panics.
        let ghost = OwnershipSet::new(["nope/x.rs".into()]).canonicalized(base);
        assert!(!ghost.overlaps(&a));

        // Directory markers survive canonicalization.
        let mixed = OwnershipSet::new(["src/".into(), "src/a.rs".into()]).canonicalized(base);
        assert!(mixed.overlaps(&nested));
    }

    #[tokio::test]
    async fn cycle_detected_before_any_run() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let counter = Arc::new(AtomicUsize::new(0));
        submit(&s, task(1, vec![2], ResourceClass::Cpu, 1, counter.clone()));
        submit(&s, task(2, vec![1], ResourceClass::Cpu, 1, counter.clone()));
        let err = s.run_to_completion().await.unwrap_err();
        assert!(err.kind == ErrorKind::Deadlock);
        assert_eq!(
            counter.load(Ordering::SeqCst),
            0,
            "cycle must prevent all work"
        );
    }

    #[tokio::test]
    async fn missing_dependency_rejected() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        submit(
            &s,
            task(
                1,
                vec![99],
                ResourceClass::Cpu,
                1,
                Arc::new(AtomicUsize::new(0)),
            ),
        );
        let err = s.run_to_completion().await.unwrap_err();
        assert!(err.kind == ErrorKind::NotFound);
    }

    #[tokio::test]
    async fn task_failure_does_not_block_other_branches() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let counter = Arc::new(AtomicUsize::new(0));
        submit(&s, err_task(1, vec![]));
        submit(&s, task(2, vec![], ResourceClass::Cpu, 1, counter.clone()));
        let result = s.run_to_completion().await;
        assert!(result.is_ok());
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert_eq!(s.status(OpId::new(1)), Some(TaskStatus::Failed));
        assert_eq!(s.status(OpId::new(2)), Some(TaskStatus::Done));
    }

    /// A failed upstream must block a `Success`-edge dependent forever, while
    /// an independent branch keeps running. The dependent's runnable must
    /// never execute (counter stays 0).
    #[tokio::test]
    async fn failed_upstream_blocks_success_dependent() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let b_runs = Arc::new(AtomicUsize::new(0));
        let c_runs = Arc::new(AtomicUsize::new(0));
        submit(&s, err_task(1, vec![]));
        submit(&s, task(2, vec![1], ResourceClass::Cpu, 1, b_runs.clone()));
        submit(&s, task(3, vec![], ResourceClass::Cpu, 1, c_runs.clone()));
        let result = s.run_to_completion().await;
        assert!(result.is_ok());
        assert_eq!(s.status(OpId::new(1)), Some(TaskStatus::Failed));
        assert_eq!(s.status(OpId::new(2)), Some(TaskStatus::Blocked));
        assert_eq!(s.status(OpId::new(3)), Some(TaskStatus::Done));
        assert_eq!(
            b_runs.load(Ordering::SeqCst),
            0,
            "blocked dependent must never execute its runnable"
        );
        assert_eq!(
            c_runs.load(Ordering::SeqCst),
            1,
            "independent branch must run to completion"
        );
    }

    /// Blocking is transitive: C's `Success` edge on the blocked B blocks C
    /// even though B never ran (B produced no failure event of its own).
    #[tokio::test]
    async fn failed_upstream_transitive_block() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let b_runs = Arc::new(AtomicUsize::new(0));
        let c_runs = Arc::new(AtomicUsize::new(0));
        submit(&s, err_task(1, vec![]));
        submit(&s, task(2, vec![1], ResourceClass::Cpu, 1, b_runs.clone()));
        submit(&s, task(3, vec![2], ResourceClass::Cpu, 1, c_runs.clone()));
        let result = s.run_to_completion().await;
        assert!(result.is_ok());
        assert_eq!(s.status(OpId::new(1)), Some(TaskStatus::Failed));
        assert_eq!(s.status(OpId::new(2)), Some(TaskStatus::Blocked));
        assert_eq!(s.status(OpId::new(3)), Some(TaskStatus::Blocked));
        assert_eq!(b_runs.load(Ordering::SeqCst), 0);
        assert_eq!(
            c_runs.load(Ordering::SeqCst),
            0,
            "transitively blocked task must never execute"
        );
    }

    /// A `Terminal` edge is satisfied by a failed upstream: B must run after
    /// A fails.
    #[tokio::test]
    async fn terminal_policy_runs_after_failure() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let b_runs = Arc::new(AtomicUsize::new(0));
        submit(&s, err_task(1, vec![]));
        submit(
            &s,
            policy_task(
                2,
                vec![(1, DependencyPolicy::Terminal)],
                ResourceClass::Cpu,
                1,
                b_runs.clone(),
            ),
        );
        let result = s.run_to_completion().await;
        assert!(result.is_ok());
        assert_eq!(s.status(OpId::new(1)), Some(TaskStatus::Failed));
        assert_eq!(s.status(OpId::new(2)), Some(TaskStatus::Done));
        assert_eq!(
            b_runs.load(Ordering::SeqCst),
            1,
            "terminal-edge dependent must run after upstream failure"
        );
    }

    /// Cleanup semantics: an `Always` edge on a blocked task still fires,
    /// even though every `Success` edge in the main path is dead.
    #[tokio::test]
    async fn always_policy_cleanup_runs_even_when_blocked() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let b_runs = Arc::new(AtomicUsize::new(0));
        let c_runs = Arc::new(AtomicUsize::new(0));
        submit(&s, err_task(1, vec![]));
        submit(&s, task(2, vec![1], ResourceClass::Cpu, 1, b_runs.clone()));
        submit(
            &s,
            policy_task(
                3,
                vec![(2, DependencyPolicy::Always)],
                ResourceClass::Cpu,
                1,
                c_runs.clone(),
            ),
        );
        let result = s.run_to_completion().await;
        assert!(result.is_ok());
        assert_eq!(s.status(OpId::new(1)), Some(TaskStatus::Failed));
        assert_eq!(s.status(OpId::new(2)), Some(TaskStatus::Blocked));
        assert_eq!(s.status(OpId::new(3)), Some(TaskStatus::Done));
        assert_eq!(b_runs.load(Ordering::SeqCst), 0);
        assert_eq!(
            c_runs.load(Ordering::SeqCst),
            1,
            "always-policy cleanup must run even off a blocked task"
        );
    }

    /// Cancelling an upstream is a dead `Success` edge: the dependent blocks.
    #[tokio::test]
    async fn cancel_upstream_blocks_success_dependent() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let b_runs = Arc::new(AtomicUsize::new(0));
        submit(
            &s,
            task(
                1,
                vec![],
                ResourceClass::Cpu,
                1,
                Arc::new(AtomicUsize::new(0)),
            ),
        );
        submit(&s, task(2, vec![1], ResourceClass::Cpu, 1, b_runs.clone()));
        s.cancel(OpId::new(1));
        let result = s.run_to_completion().await;
        assert!(result.is_ok());
        assert_eq!(s.status(OpId::new(1)), Some(TaskStatus::Cancelled));
        assert_eq!(s.status(OpId::new(2)), Some(TaskStatus::Blocked));
        assert_eq!(
            b_runs.load(Ordering::SeqCst),
            0,
            "dependent of cancelled upstream must never run"
        );
    }

    /// The default policy is Success: an explicit `(a, Success)` tuple behaves
    /// exactly like the historical `depends_on` on the success path.
    #[tokio::test]
    async fn success_policy_is_the_default() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let counter = Arc::new(AtomicUsize::new(0));
        let op_a = ScheduledOp {
            dependencies: vec![],
            ..task(1, vec![], ResourceClass::Cpu, 1, counter.clone())
        };
        let op_b = ScheduledOp {
            dependencies: vec![(OpId::new(1), DependencyPolicy::Success)],
            ..task(2, vec![], ResourceClass::Cpu, 1, counter.clone())
        };
        submit(&s, op_a);
        submit(&s, op_b);
        let done = s.run_to_completion().await.unwrap();
        assert_eq!(done.len(), 2);
        assert_eq!(s.status(OpId::new(1)), Some(TaskStatus::Done));
        assert_eq!(s.status(OpId::new(2)), Some(TaskStatus::Done));
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    /// A blocked path is a defined outcome, not a deadlock: the run returns
    /// Ok and reports the blocked task, even when nothing else runs.
    #[tokio::test]
    async fn blocked_is_not_a_deadlock() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let b_runs = Arc::new(AtomicUsize::new(0));
        submit(&s, err_task(1, vec![]));
        submit(&s, task(2, vec![1], ResourceClass::Cpu, 1, b_runs.clone()));
        let done = s
            .run_to_completion()
            .await
            .expect("blocked is not a deadlock");
        assert!(
            !done.contains(&OpId::new(2)),
            "blocked task must not be reported Done"
        );
        assert_eq!(s.status(OpId::new(1)), Some(TaskStatus::Failed));
        assert_eq!(s.status(OpId::new(2)), Some(TaskStatus::Blocked));
        assert_eq!(b_runs.load(Ordering::SeqCst), 0);
        assert!(s.statuses().contains(&(OpId::new(2), TaskStatus::Blocked)));
    }

    #[tokio::test]
    async fn deadline_exceeded_is_timeout_error() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let mut t = task(
            1,
            vec![],
            ResourceClass::Cpu,
            50_000,
            Arc::new(AtomicUsize::new(0)),
        );
        t.meta.deadline = Deadline::at(SystemClock.now_ms() + 20);
        submit(&s, t);
        let spec = {
            let guard = s.inner.lock().unwrap();
            guard.tasks[&OpId::new(1)].op.clone()
        };
        let err = s
            .execute(OpId::new(1), spec, &test_key())
            .await
            .unwrap_err();
        assert!(matches!(err, ExecuteError::Err(e) if e.kind == ErrorKind::Timeout));
        assert_eq!(s.status(OpId::new(1)), Some(TaskStatus::Failed));
    }

    #[tokio::test]
    async fn retry_with_jitter_bounded() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let attempts = Arc::new(AtomicUsize::new(0));
        let spec = ScheduledOp {
            meta: OpMeta::new(
                OpId::new(1),
                SessionId::new(1),
                Deadline::at(FAR),
                RetryPolicy {
                    max_attempts: 4,
                    base_delay_ms: 1,
                    max_delay_ms: 5,
                    jitter: 0.0,
                    class: RetryClass::Always,
                },
                CancellationToken::new(),
                RecoveryStrategy::None,
                0,
            ),
            resources: ResourceRequest {
                class: ResourceClass::Network,
            },
            reads: OwnershipSet::new([]),
            writes: OwnershipSet::new([]),
            dependencies: vec![],
            run: {
                let a = attempts.clone();
                Arc::new(move || {
                    let a = a.clone();
                    Box::pin(async move {
                        if a.fetch_add(1, Ordering::SeqCst) < 2 {
                            Err(Error::new(ErrorKind::Network, "flaky"))
                        } else {
                            Ok(())
                        }
                    })
                })
            },
        };
        submit(&s, spec.clone());
        s.execute(OpId::new(1), spec, &test_key()).await.unwrap();
        assert_eq!(
            attempts.load(Ordering::SeqCst),
            3,
            "2 failures then success = 3 attempts"
        );
        assert_eq!(s.status(OpId::new(1)), Some(TaskStatus::Done));
    }

    #[tokio::test]
    async fn non_retryable_failure_stops_immediately() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let attempts = Arc::new(AtomicUsize::new(0));
        let spec = ScheduledOp {
            meta: OpMeta::new(
                OpId::new(1),
                SessionId::new(1),
                Deadline::at(FAR),
                RetryPolicy {
                    max_attempts: 10,
                    base_delay_ms: 1,
                    max_delay_ms: 5,
                    jitter: 0.0,
                    class: RetryClass::Always,
                },
                CancellationToken::new(),
                RecoveryStrategy::None,
                0,
            ),
            resources: ResourceRequest {
                class: ResourceClass::Cpu,
            },
            reads: OwnershipSet::new([]),
            writes: OwnershipSet::new([]),
            dependencies: vec![],
            run: {
                let a = attempts.clone();
                Arc::new(move || {
                    let a = a.clone();
                    Box::pin(async move {
                        a.fetch_add(1, Ordering::SeqCst);
                        Err(Error::new(ErrorKind::Conflict, "stale patch"))
                    })
                })
            },
        };
        submit(&s, spec.clone());
        let err = s
            .execute(OpId::new(1), spec, &test_key())
            .await
            .unwrap_err();
        assert!(matches!(err, ExecuteError::Err(e) if e.kind == ErrorKind::Conflict));
        assert_eq!(
            attempts.load(Ordering::SeqCst),
            1,
            "conflict is never retried"
        );
    }

    #[test]
    fn circuit_breaker_opens_probes_once_and_recovers() {
        let cb = CircuitBreaker::new(3, 1000);
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.allow(0).is_ok());
        cb.record_failure(1);
        cb.record_failure(2);
        assert_eq!(cb.state(), CircuitState::Closed, "threshold is 3");
        assert_eq!(cb.failures(), 2);
        cb.record_failure(3);
        assert_eq!(cb.state(), CircuitState::Open);
        assert_eq!(
            cb.allow(4),
            Err(BreakerStatus::CircuitOpen),
            "cooldown not elapsed"
        );
        // Cooldown elapsed: the breaker decays by admitting exactly ONE
        // probe; a second concurrent caller is denied.
        assert!(cb.allow(2001).is_ok(), "cooldown elapsed admits a probe");
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        assert_eq!(
            cb.allow(2002),
            Err(BreakerStatus::CircuitOpen),
            "only one probe may run at a time"
        );
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.failures(), 0);
        // Re-open and fail the probe: circuit re-opens for a fresh cooldown.
        cb.record_failure(10);
        cb.record_failure(11);
        cb.record_failure(12);
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(
            cb.allow(2000).is_ok(),
            "fresh cooldown elapsed admits probe"
        );
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        cb.record_failure(13);
        assert_eq!(
            cb.state(),
            CircuitState::Open,
            "half-open failure re-opens the circuit"
        );
        assert_eq!(cb.allow(13), Err(BreakerStatus::CircuitOpen));
        assert!(
            cb.allow(13 + 1000 + 1).is_ok(),
            "a re-opened breaker must admit a new probe after its cooldown"
        );
        // A record_failure on an already-open breaker never moves it.
        cb.record_failure(15);
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn circuit_breaker_force_open_denies_until_probe() {
        let cb = CircuitBreaker::new(4, 500);
        assert!(cb.allow(0).is_ok());
        cb.open(0);
        assert_eq!(cb.state(), CircuitState::Open);
        assert_eq!(cb.allow(100), Err(BreakerStatus::CircuitOpen));
        assert!(cb.allow(501).is_ok(), "forced-open breaker still probes");
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn circuit_board_keys_by_resource_not_operation() {
        // Audit: breakers are scoped to the failing RESOURCE, keyed by a
        // typed [`ResourceKey`] (provider/model/endpoint, MCP server, tool
        // kind, host). A burst of distinct operations against one resource
        // shares a breaker, while a healthy resource is never poisoned by
        // an unrelated one — and different resources of one CLASS never
        // share a breaker.
        let board = CircuitBoard::new();
        let res_bad = ResourceKey::Provider {
            instance: "anthropic".into(),
            model: "opus".into(),
            endpoint: "https://api.anthropic.com/v1".into(),
        };
        let res_ok = ResourceKey::Mcp {
            server: "filesystem".into(),
        };
        assert!(board.allow(&res_bad, 0).is_ok());
        // Four DIFFERENT failing operations against the same resource.
        for op in 0..4u32 {
            let now = 1 + i64::from(op);
            board.record_failure(&res_bad, now);
        }
        assert_eq!(board.state(&res_bad), CircuitState::Open);
        assert_eq!(board.allow(&res_bad, 2), Err(BreakerStatus::CircuitOpen));
        assert_eq!(
            board.state(&res_ok),
            CircuitState::Closed,
            "an unrelated resource must stay healthy"
        );
        assert!(board.allow(&res_ok, 2).is_ok());
        // Cooldown decay admits a probe; probe success heals the resource.
        // (Opened at t=4 with the default 5000ms cooldown.)
        assert!(board.allow(&res_bad, 5_004).is_ok());
        board.record_success(&res_bad);
        assert_eq!(board.state(&res_bad), CircuitState::Closed);
        // Force-open API (runtime sees a dead host out of band).
        board.open(&res_ok, 0);
        assert_eq!(board.allow(&res_ok, 100), Err(BreakerStatus::CircuitOpen));
        assert_eq!(
            board.state(&res_ok),
            CircuitState::Open,
            "board is per-key: one breaker per resource"
        );
        // Resource keys compare/order deterministically (BTree/Map keys):
        // Provider < Mcp by variant order, then lexicographically by field.
        assert!(res_bad < res_ok);
        assert_eq!(
            format!("{res_bad}"),
            "provider:anthropic:opus:https://api.anthropic.com/v1"
        );
    }

    #[test]
    fn circuit_board_is_bounded_and_evicts_least_recently_used() {
        // (d) Adversarial: the board is NEVER unbounded. 10_000 distinct
        // keys against a cap of 1000 pin `len` at the cap; the least
        // recently consulted key is evicted even when its breaker is OPEN
        // (evicted state is gone — the key re-enters closed), while a key
        // kept hot survives the whole storm with its state intact.
        let board = CircuitBoard::with_max_keys(1000);
        assert_eq!(board.max_keys(), 1000);
        let hot = ResourceKey::Host {
            scheme: "https".into(),
            host: "hot.example".into(),
            port: 443,
        };
        let cold = ResourceKey::Host {
            scheme: "https".into(),
            host: "cold.example".into(),
            port: 443,
        };
        for i in 0..4 {
            board.record_failure(&hot, i as i64);
            board.record_failure(&cold, i as i64);
        }
        assert_eq!(board.state(&hot), CircuitState::Open);
        assert_eq!(board.state(&cold), CircuitState::Open);
        for i in 0..10_000u32 {
            if i % 50 == 0 {
                // Keep `hot` recently used; the consult must keep seeing
                // the OPEN state it had before the storm (no corruption).
                assert_eq!(
                    board.state(&hot),
                    CircuitState::Open,
                    "hot key must not be evicted mid-storm"
                );
            }
            let k = ResourceKey::Provider {
                instance: format!("inst-{i}"),
                model: "m".into(),
                endpoint: format!("https://endpoint/{i}"),
            };
            assert!(
                board.allow(&k, 0).is_ok(),
                "a fresh key always starts closed"
            );
        }
        assert_eq!(
            board.len(),
            1000,
            "board must never grow past max_keys under a 10k-key storm"
        );
        assert_eq!(
            board.state(&hot),
            CircuitState::Open,
            "the hot key survived eviction (true LRU)"
        );
        assert_eq!(
            board.state(&cold),
            CircuitState::Closed,
            "the least-recently-used key was evicted while OPEN and \
             re-enters with a fresh closed breaker"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn half_open_admits_exactly_one_probe_under_contention() {
        // AtomicU8 CAS: when the cooldown elapses, N racing callers must
        // yield EXACTLY ONE probe winner.
        let cb = Arc::new(CircuitBreaker::new(1, 1000));
        cb.record_failure(0); // threshold 1 -> open
        assert_eq!(cb.state(), CircuitState::Open);
        let n = 32;
        let mut set = tokio::task::JoinSet::new();
        for _ in 0..n {
            let cb = cb.clone();
            set.spawn(async move { cb.allow(2_000).is_ok() });
        }
        let mut winners = 0usize;
        while let Some(r) = set.join_next().await {
            winners += usize::from(r.unwrap());
        }
        assert_eq!(winners, 1, "exactly one probe may win the CAS race");
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        // The winner's probe succeeded -> closed again.
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[tokio::test]
    async fn circuit_open_denial_is_internal_not_deadlock() {
        // A resource storm trips the shared breaker mid-retry; the task is
        // marked Failed with a circuit-open signal — never a Deadlock
        // classification (deadlock drives scheduler-level recovery).
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let attempts = Arc::new(AtomicUsize::new(0));
        let key = ResourceKey::Host {
            scheme: "https".into(),
            host: "api.anthropic.com".into(),
            port: 443,
        };
        let spec = ScheduledOp {
            meta: OpMeta::new(
                OpId::new(1),
                SessionId::new(1),
                Deadline::at(FAR),
                RetryPolicy {
                    max_attempts: 100,
                    base_delay_ms: 1,
                    max_delay_ms: 2,
                    jitter: 0.0,
                    class: RetryClass::Always,
                },
                CancellationToken::new(),
                RecoveryStrategy::None,
                0,
            ),
            resources: ResourceRequest {
                class: ResourceClass::Network,
            },
            reads: OwnershipSet::new([]),
            writes: OwnershipSet::new([]),
            dependencies: vec![],
            run: {
                let a = attempts.clone();
                Arc::new(move || {
                    let a = a.clone();
                    Box::pin(async move {
                        a.fetch_add(1, Ordering::SeqCst);
                        Err(Error::new(ErrorKind::Network, "flaky"))
                    })
                })
            },
        };
        submit(&s, spec.clone());
        let err = s.execute(OpId::new(1), spec, &key).await.unwrap_err();
        let e = match err {
            ExecuteError::Err(e) => e,
            ExecuteError::Busy(_) => panic!("budget must not be busy"),
        };
        assert_eq!(
            e.kind,
            ErrorKind::Internal,
            "circuit open is not a deadlock"
        );
        assert!(e.message.contains("circuit open"), "message: {}", e.message);
        assert_eq!(
            attempts.load(Ordering::SeqCst),
            DEFAULT_BREAKER_FAILURE_THRESHOLD as usize,
            "attempts must stop at the breaker threshold"
        );
        assert_eq!(s.status(OpId::new(1)), Some(TaskStatus::Failed));
        assert_eq!(s.circuits().state(&key), CircuitState::Open);
    }

    /// A Network-class op that fails with a retryable error on EVERY
    /// attempt (breaker-storm fodder).
    fn storm_op(id: u64, attempts: Arc<AtomicUsize>) -> ScheduledOp {
        ScheduledOp {
            meta: OpMeta::new(
                OpId::new(id),
                SessionId::new(1),
                Deadline::at(FAR),
                RetryPolicy {
                    max_attempts: 100,
                    base_delay_ms: 1,
                    max_delay_ms: 2,
                    jitter: 0.0,
                    class: RetryClass::Always,
                },
                CancellationToken::new(),
                RecoveryStrategy::None,
                0,
            ),
            resources: ResourceRequest {
                class: ResourceClass::Network,
            },
            reads: OwnershipSet::new([]),
            writes: OwnershipSet::new([]),
            dependencies: vec![],
            run: Arc::new(move || {
                let a = attempts.clone();
                Box::pin(async move {
                    a.fetch_add(1, Ordering::SeqCst);
                    Err(Error::new(ErrorKind::Network, "flaky"))
                })
            }),
        }
    }

    /// A Network-class op that succeeds on its first attempt.
    fn ok_op(id: u64, runs: Arc<AtomicUsize>) -> ScheduledOp {
        ScheduledOp {
            meta: OpMeta::new(
                OpId::new(id),
                SessionId::new(1),
                Deadline::at(FAR),
                RetryPolicy::default(),
                CancellationToken::new(),
                RecoveryStrategy::None,
                0,
            ),
            resources: ResourceRequest {
                class: ResourceClass::Network,
            },
            reads: OwnershipSet::new([]),
            writes: OwnershipSet::new([]),
            dependencies: vec![],
            run: Arc::new(move || {
                let r = runs.clone();
                Box::pin(async move {
                    r.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                })
            }),
        }
    }

    fn provider_key(model: &str, endpoint: &str) -> ResourceKey {
        ResourceKey::Provider {
            instance: "anthropic".into(),
            model: model.into(),
            endpoint: endpoint.into(),
        }
    }

    /// Assert `execute` ended with a circuit-open denial (Internal kind,
    /// message names the circuit) after exactly `attempts` runnable starts.
    fn assert_circuit_denied(
        err: ExecuteError,
        attempts: &Arc<AtomicUsize>,
        expected_attempts: usize,
    ) {
        let e = match err {
            ExecuteError::Err(e) => e,
            ExecuteError::Busy(_) => panic!("budget must not be busy"),
        };
        assert_eq!(
            e.kind,
            ErrorKind::Internal,
            "circuit open is a resource-health denial, never a Deadlock"
        );
        assert!(e.message.contains("circuit open"), "message: {}", e.message);
        assert_eq!(
            attempts.load(Ordering::SeqCst),
            expected_attempts,
            "runnable starts must stop exactly at the breaker gate"
        );
    }

    #[tokio::test]
    async fn one_broken_provider_refuses_other_sessions_while_healthy_resources_proceed() {
        // (a) Adversarial: the daemon-level board is SHARED across
        // schedulers/sessions. Session A trips provider P open; session
        // B's request against P is refused BEFORE its runnable starts
        // (zero attempts — fast fail, no wasted work), while session B's
        // request against healthy provider Q — same resource class —
        // proceeds to completion.
        let board = CircuitBoard::new();
        let p = provider_key("opus", "https://api.anthropic.com/v1");
        let q = ResourceKey::Mcp {
            server: "filesystem".into(),
        };
        let sched_a =
            Scheduler::new(SessionId::new(1), Arc::new(SystemClock)).with_circuits(board.clone());
        let sched_b =
            Scheduler::new(SessionId::new(2), Arc::new(SystemClock)).with_circuits(board.clone());

        let storm = Arc::new(AtomicUsize::new(0));
        submit(&sched_a, storm_op(1, storm.clone()));
        let err = sched_a
            .execute(OpId::new(1), storm_op(1, storm.clone()), &p)
            .await
            .expect_err("session A must fail once provider P trips");
        assert_circuit_denied(err, &storm, DEFAULT_BREAKER_FAILURE_THRESHOLD as usize);
        assert_eq!(sched_a.status(OpId::new(1)), Some(TaskStatus::Failed));

        // Session B, same broken provider P: refused fast, runnable never
        // runs (the breaker is open; cooldown has not elapsed).
        let denied = Arc::new(AtomicUsize::new(0));
        submit(&sched_b, storm_op(2, denied.clone()));
        let err = sched_b
            .execute(OpId::new(2), storm_op(2, denied.clone()), &p)
            .await
            .expect_err("session B must be refused for the broken provider");
        assert_circuit_denied(err, &denied, 0);
        assert_eq!(sched_b.status(OpId::new(2)), Some(TaskStatus::Failed));

        // Session B, healthy resource Q of the SAME class (Network):
        // unaffected by P's breaker.
        let healthy = Arc::new(AtomicUsize::new(0));
        submit(&sched_b, ok_op(3, healthy.clone()));
        sched_b
            .execute(OpId::new(3), ok_op(3, healthy.clone()), &q)
            .await
            .expect("healthy resource must proceed while P is open");
        assert_eq!(healthy.load(Ordering::SeqCst), 1);
        assert_eq!(sched_b.status(OpId::new(3)), Some(TaskStatus::Done));
        assert_eq!(board.state(&p), CircuitState::Open);
        assert_eq!(board.state(&q), CircuitState::Closed);
    }

    #[tokio::test]
    async fn provider_models_have_independent_breakers() {
        // (b) Adversarial: two MODELS of the same provider (same instance,
        // same endpoint) do not share a breaker — M1 tripping open must not
        // block M2.
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let m1 = provider_key("opus", "https://api.anthropic.com/v1");
        let m2 = provider_key("sonnet", "https://api.anthropic.com/v1");
        let storm = Arc::new(AtomicUsize::new(0));
        submit(&s, storm_op(1, storm.clone()));
        let err = s
            .execute(OpId::new(1), storm_op(1, storm.clone()), &m1)
            .await
            .expect_err("M1 storm must trip M1");
        assert_circuit_denied(err, &storm, DEFAULT_BREAKER_FAILURE_THRESHOLD as usize);
        assert_eq!(s.circuits().state(&m1), CircuitState::Open);
        // M2 — different model, same provider/endpoint — must be unaffected.
        let ok_runs = Arc::new(AtomicUsize::new(0));
        submit(&s, ok_op(2, ok_runs.clone()));
        s.execute(OpId::new(2), ok_op(2, ok_runs.clone()), &m2)
            .await
            .expect("M2 must run while M1 is open");
        assert_eq!(ok_runs.load(Ordering::SeqCst), 1);
        assert_eq!(s.status(OpId::new(2)), Some(TaskStatus::Done));
        assert_eq!(s.circuits().state(&m2), CircuitState::Closed);
    }

    #[tokio::test]
    async fn endpoints_of_one_provider_model_have_independent_breakers() {
        // (c) Adversarial: the same provider instance+model on two
        // ENDPOINTS has independent breakers (one endpoint's outage must
        // not blacklist the other), and both stay independent although the
        // ops share a resource class (Network).
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let e1 = provider_key("opus", "https://us.api.anthropic.com/v1");
        let e2 = provider_key("opus", "https://eu.api.anthropic.com/v1");
        let storm = Arc::new(AtomicUsize::new(0));
        submit(&s, storm_op(1, storm.clone()));
        let err = s
            .execute(OpId::new(1), storm_op(1, storm.clone()), &e1)
            .await
            .expect_err("endpoint 1 storm must trip endpoint 1");
        assert_circuit_denied(err, &storm, DEFAULT_BREAKER_FAILURE_THRESHOLD as usize);
        assert_eq!(s.circuits().state(&e1), CircuitState::Open);
        let ok_runs = Arc::new(AtomicUsize::new(0));
        submit(&s, ok_op(2, ok_runs.clone()));
        s.execute(OpId::new(2), ok_op(2, ok_runs.clone()), &e2)
            .await
            .expect("endpoint 2 must run while endpoint 1 is open");
        assert_eq!(ok_runs.load(Ordering::SeqCst), 1);
        assert_eq!(s.status(OpId::new(2)), Some(TaskStatus::Done));
        assert_eq!(s.circuits().state(&e2), CircuitState::Closed);
    }

    #[tokio::test]
    async fn tool_kinds_of_one_class_do_not_interfere() {
        // Two tool kinds sharing a resource class never share a breaker:
        // the OLD (session, resource-class) key would have tripped both.
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let t1 = ResourceKey::Tool {
            name: "edit".into(),
        };
        let t2 = ResourceKey::Tool {
            name: "search".into(),
        };
        let storm = Arc::new(AtomicUsize::new(0));
        submit(&s, storm_op(1, storm.clone()));
        let err = s
            .execute(OpId::new(1), storm_op(1, storm.clone()), &t1)
            .await
            .expect_err("tool-1 storm must trip tool 1");
        assert_circuit_denied(err, &storm, DEFAULT_BREAKER_FAILURE_THRESHOLD as usize);
        let ok_runs = Arc::new(AtomicUsize::new(0));
        submit(&s, ok_op(2, ok_runs.clone()));
        s.execute(OpId::new(2), ok_op(2, ok_runs.clone()), &t2)
            .await
            .expect("tool 2 must run while tool 1 is open");
        assert_eq!(ok_runs.load(Ordering::SeqCst), 1);
        assert_eq!(s.circuits().state(&t2), CircuitState::Closed);
    }

    #[tokio::test]
    async fn resource_budget_prevents_starvation() {
        let mut limits = ResourceLimits::default();
        limits.limits.insert(ResourceClass::Indexing, 1);
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock)).with_limits(limits);
        let counter = Arc::new(AtomicUsize::new(0));
        for i in 1..=5 {
            submit(
                &s,
                task(i, vec![], ResourceClass::Indexing, 1, counter.clone()),
            );
        }
        submit(
            &s,
            task(9, vec![], ResourceClass::Model, 1, counter.clone()),
        );
        s.run_to_completion().await.unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 6, "nothing starved");
    }

    #[tokio::test]
    async fn cancel_pending_task() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        submit(
            &s,
            task(
                1,
                vec![],
                ResourceClass::Cpu,
                1,
                Arc::new(AtomicUsize::new(0)),
            ),
        );
        s.cancel(OpId::new(1));
        assert_eq!(s.status(OpId::new(1)), Some(TaskStatus::Cancelled));
    }

    #[test]
    fn ownership_set_normalization() {
        let a = OwnershipSet::new(["b".into(), "a".into(), "b".into()]);
        assert_eq!(a.0, vec!["a".to_string(), "b".to_string()]);
    }

    #[tokio::test]
    async fn many_independent_tasks_run_concurrently() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let counter = Arc::new(AtomicUsize::new(0));
        for i in 1..=32 {
            submit(
                &s,
                task(i, vec![], ResourceClass::DiskRead, 30, counter.clone()),
            );
        }
        let t0 = std::time::Instant::now();
        s.run_to_completion().await.unwrap();
        let elapsed = t0.elapsed();
        assert_eq!(counter.load(Ordering::SeqCst), 32);
        // 32 tasks × 30ms with DiskRead budget 16 → ~2 waves ≈ 60ms+.
        assert!(elapsed < Duration::from_millis(400), "took {elapsed:?}");
    }

    #[tokio::test]
    async fn concurrent_status_reads_are_safe() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        for i in 1..=4 {
            submit(
                &s,
                task(
                    i,
                    vec![],
                    ResourceClass::Cpu,
                    2,
                    Arc::new(AtomicUsize::new(0)),
                ),
            );
        }
        let s2 = s.clone();
        let reader = tokio::spawn(async move {
            for _ in 0..200 {
                let _ = s2.statuses();
            }
        });
        s.run_to_completion().await.unwrap();
        reader.await.unwrap();
        assert_eq!(s.statuses().len(), 4);
        assert!(s.statuses().iter().all(|(_, st)| *st == TaskStatus::Done));
    }

    /// A herd of 1024 queued tasks (the MAX_TASKS_PER_DAG bound) on a budget
    /// of 16 must never run more than 16 concurrently: the permit is
    /// acquired before the task is spawned, so the runnable's own in-flight
    /// counter is the proof.
    #[tokio::test(flavor = "multi_thread", worker_threads = 32)]
    async fn permit_before_spawn_prevents_herd() {
        let mut limits = ResourceLimits::default();
        limits.limits.insert(ResourceClass::DiskRead, 16);
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock)).with_limits(limits);
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let completed = Arc::new(AtomicUsize::new(0));
        for i in 1..=MAX_TASKS_PER_DAG {
            let i = i as u64;
            let active = active.clone();
            let max_active = max_active.clone();
            let completed = completed.clone();
            let op = ScheduledOp {
                meta: OpMeta::new(
                    OpId::new(i),
                    SessionId::new(1),
                    Deadline::at(FAR),
                    RetryPolicy::default(),
                    CancellationToken::new(),
                    RecoveryStrategy::None,
                    0,
                ),
                resources: ResourceRequest {
                    class: ResourceClass::DiskRead,
                },
                reads: OwnershipSet::new([]),
                writes: OwnershipSet::new([]),
                dependencies: vec![],
                run: Arc::new(move || {
                    let active = active.clone();
                    let max_active = max_active.clone();
                    let completed = completed.clone();
                    Box::pin(async move {
                        let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                        max_active.fetch_max(now, Ordering::SeqCst);
                        tokio::time::sleep(Duration::from_millis(2)).await;
                        active.fetch_sub(1, Ordering::SeqCst);
                        completed.fetch_add(1, Ordering::SeqCst);
                        Ok(())
                    })
                }),
            };
            submit(&s, op);
        }
        s.run_to_completion().await.unwrap();
        assert_eq!(
            completed.load(Ordering::SeqCst),
            MAX_TASKS_PER_DAG,
            "every queued task must run"
        );
        let peak = max_active.load(Ordering::SeqCst);
        assert!(
            peak <= 16,
            "herd: {peak} tasks ran concurrently with budget 16"
        );
    }

    /// No wave barrier: B (1ms) becomes ready while A (200ms) is still
    /// running; C (depends on B) must start while A is still in flight. A
    /// wave scheduler would defer C until A completes.
    #[tokio::test]
    async fn event_driven_no_wave_barrier() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let c_saw_a_running = Arc::new(AtomicBool::new(false));

        let op_a = ScheduledOp {
            meta: OpMeta::new(
                OpId::new(1),
                SessionId::new(1),
                Deadline::at(FAR),
                RetryPolicy::default(),
                CancellationToken::new(),
                RecoveryStrategy::None,
                0,
            ),
            resources: ResourceRequest {
                class: ResourceClass::Cpu,
            },
            reads: OwnershipSet::new([]),
            writes: OwnershipSet::new([]),
            dependencies: vec![],
            run: Arc::new(|| {
                Box::pin(async {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    Ok(())
                })
            }),
        };
        let op_b = ScheduledOp {
            meta: OpMeta::new(
                OpId::new(2),
                SessionId::new(1),
                Deadline::at(FAR),
                RetryPolicy::default(),
                CancellationToken::new(),
                RecoveryStrategy::None,
                0,
            ),
            resources: ResourceRequest {
                class: ResourceClass::Cpu,
            },
            reads: OwnershipSet::new([]),
            writes: OwnershipSet::new([]),
            dependencies: vec![],
            run: Arc::new(|| {
                Box::pin(async {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                    Ok(())
                })
            }),
        };
        let s_for_c = s.clone();
        let c_saw = c_saw_a_running.clone();
        let op_c = ScheduledOp {
            meta: OpMeta::new(
                OpId::new(3),
                SessionId::new(1),
                Deadline::at(FAR),
                RetryPolicy::default(),
                CancellationToken::new(),
                RecoveryStrategy::None,
                0,
            ),
            resources: ResourceRequest {
                class: ResourceClass::Cpu,
            },
            reads: OwnershipSet::new([]),
            writes: OwnershipSet::new([]),
            dependencies: vec![(OpId::new(2), DependencyPolicy::Success)],
            run: Arc::new(move || {
                let s = s_for_c.clone();
                let flag = c_saw.clone();
                Box::pin(async move {
                    if s.status(OpId::new(1)) == Some(TaskStatus::Running) {
                        flag.store(true, Ordering::SeqCst);
                    }
                    tokio::time::sleep(Duration::from_millis(1)).await;
                    Ok(())
                })
            }),
        };
        submit(&s, op_a);
        submit(&s, op_b);
        submit(&s, op_c);
        s.run_to_completion().await.unwrap();
        assert!(
            c_saw_a_running.load(Ordering::SeqCst),
            "C must start while A is still running; a wave barrier would defer C until A completes"
        );
    }

    /// Cancellation flows from OpMeta: Scheduler::cancel fires the op's own
    /// token, and a runnable that polls it aborts promptly instead of running
    /// its full (near-infinite) duration.
    #[tokio::test]
    async fn cancellation_from_op_meta() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let token = CancellationToken::new();
        let started = Arc::new(AtomicBool::new(false));
        let run_ms = Arc::new(AtomicUsize::new(0));
        let token_for_run = token.clone();
        let started_c = started.clone();
        let run_ms_c = run_ms.clone();
        let op = ScheduledOp {
            meta: OpMeta::new(
                OpId::new(1),
                SessionId::new(1),
                Deadline::at(FAR),
                RetryPolicy::default(),
                token,
                RecoveryStrategy::None,
                0,
            ),
            resources: ResourceRequest {
                class: ResourceClass::Cpu,
            },
            reads: OwnershipSet::new([]),
            writes: OwnershipSet::new([]),
            dependencies: vec![],
            run: Arc::new(move || {
                let started = started_c.clone();
                let run_ms = run_ms_c.clone();
                let token = token_for_run.clone();
                Box::pin(async move {
                    let t0 = std::time::Instant::now();
                    started.store(true, Ordering::SeqCst);
                    while !token.is_cancelled() {
                        tokio::time::sleep(Duration::from_millis(1)).await;
                    }
                    run_ms.store(t0.elapsed().as_millis() as usize, Ordering::SeqCst);
                    Ok(())
                })
            }),
        };
        submit(&s, op);
        let s2 = s.clone();
        let canceller = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            s2.cancel(OpId::new(1));
        });
        let t0 = std::time::Instant::now();
        s.run_to_completion().await.unwrap();
        let elapsed = t0.elapsed();
        canceller.await.unwrap();
        assert!(
            started.load(Ordering::SeqCst),
            "runnable must start before cancellation"
        );
        assert_eq!(s.status(OpId::new(1)), Some(TaskStatus::Cancelled));
        let ms = run_ms.load(Ordering::SeqCst);
        assert!(
            ms > 0 && ms < 1000,
            "runnable must exit promptly after cancel, ran {ms}ms"
        );
        assert!(
            elapsed < Duration::from_millis(2000),
            "cancellation must end the run promptly, took {elapsed:?}"
        );
    }

    /// Recovery strategy is part of the op envelope and must survive the run
    /// untouched — success, failure, and (by construction) cancellation.
    #[tokio::test]
    async fn recovery_strategy_carries_through() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let recovery = RecoveryStrategy::VerifyHash {
            path: "/tmp/owned.rs".into(),
            expected: FileHash::from([7u8; 32]),
        };
        let mut op = task(
            1,
            vec![],
            ResourceClass::Cpu,
            1,
            Arc::new(AtomicUsize::new(0)),
        );
        op.meta.recovery = recovery.clone();
        submit(&s, op);
        s.run_to_completion().await.unwrap();
        {
            let guard = s.inner.lock().unwrap();
            let stored = &guard.tasks[&OpId::new(1)].op.meta;
            assert_eq!(
                stored.recovery, recovery,
                "successful run must not clobber recovery"
            );
            assert_eq!(stored.operation_id, OpId::new(1));
            assert_eq!(stored.session_id, SessionId::new(1));
        }

        // A failing op keeps its recovery too (crash/failure must not clobber it).
        let mut failing = err_task(2, vec![]);
        failing.meta.recovery = RecoveryStrategy::MarkUnknown;
        submit(&s, failing);
        s.run_to_completion().await.unwrap();
        {
            let guard = s.inner.lock().unwrap();
            assert_eq!(
                guard.tasks[&OpId::new(2)].op.meta.recovery,
                RecoveryStrategy::MarkUnknown
            );
            assert_eq!(guard.tasks[&OpId::new(2)].status, TaskStatus::Failed);
        }

        // Cancellation preserves it too.
        let mut cancelled = task(
            3,
            vec![],
            ResourceClass::Cpu,
            1,
            Arc::new(AtomicUsize::new(0)),
        );
        cancelled.meta.recovery = RecoveryStrategy::Manual;
        submit(&s, cancelled);
        s.cancel(OpId::new(3));
        s.run_to_completion().await.unwrap();
        {
            let guard = s.inner.lock().unwrap();
            assert_eq!(
                guard.tasks[&OpId::new(3)].op.meta.recovery,
                RecoveryStrategy::Manual
            );
            assert_eq!(guard.tasks[&OpId::new(3)].status, TaskStatus::Cancelled);
        }
    }

    /// A runnable that panics must fail its task, release its permit, and
    /// never wedge the budget for other tasks of the same class.
    #[tokio::test]
    async fn panicking_runnable_does_not_wedge_budget() {
        let mut limits = ResourceLimits::default();
        limits.limits.insert(ResourceClass::Indexing, 1);
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock)).with_limits(limits);
        let ran = Arc::new(AtomicUsize::new(0));
        let mut panicking = task(1, vec![], ResourceClass::Indexing, 1, ran.clone());
        panicking.run = Arc::new(|| Box::pin(async move { panic!("runnable exploded") }));
        let healthy = task(2, vec![], ResourceClass::Indexing, 1, ran.clone());
        submit(&s, panicking);
        submit(&s, healthy);
        let done = s.run_to_completion().await.unwrap();
        assert_eq!(
            s.status(OpId::new(1)),
            Some(TaskStatus::Failed),
            "panic must fail the task"
        );
        assert_eq!(s.status(OpId::new(2)), Some(TaskStatus::Done));
        assert_eq!(
            ran.load(Ordering::SeqCst),
            1,
            "healthy task must run on the freed permit"
        );
        assert!(done.contains(&OpId::new(2)));
    }

    // ---- audit round 5: ownership enforcement + Terminal-accepts-Blocked ----

    fn op_meta(id: u64) -> OpMeta {
        OpMeta::new(
            OpId::new(id),
            SessionId::new(1),
            faktor_core::time::Deadline::at(now_ms() + 60_000),
            RetryPolicy::default(),
            CancellationToken::new(),
            RecoveryStrategy::None,
            now_ms(),
        )
    }

    fn owned_op(
        id: u64,
        class: ResourceClass,
        reads: Vec<&str>,
        writes: Vec<&str>,
        ms: u64,
        active: Arc<AtomicUsize>,
        peak: Arc<AtomicUsize>,
    ) -> ScheduledOp {
        ScheduledOp {
            meta: op_meta(id),
            resources: ResourceRequest { class },
            reads: OwnershipSet::new(reads.into_iter().map(|s| s.to_string())),
            writes: OwnershipSet::new(writes.into_iter().map(|s| s.to_string())),
            dependencies: vec![],
            run: Arc::new(move || {
                let active = active.clone();
                let peak = peak.clone();
                Box::pin(async move {
                    let cur = active.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(cur, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(ms)).await;
                    active.fetch_sub(1, Ordering::SeqCst);
                    Ok(())
                })
            }),
        }
    }

    fn now_ms() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0)
    }

    #[tokio::test]
    async fn overlapping_writes_actually_serialize_while_running() {
        // Audit round 5: ownership is now ENFORCED against running ops.
        // Two ready ops writing the same file must never run concurrently;
        // disjoint writes may.
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        submit(
            &s,
            owned_op(
                1,
                ResourceClass::DiskWrite,
                vec![],
                vec!["src/a.rs"],
                80,
                active.clone(),
                peak.clone(),
            ),
        );
        submit(
            &s,
            owned_op(
                2,
                ResourceClass::DiskWrite,
                vec![],
                vec!["src/a.rs"],
                80,
                active.clone(),
                peak.clone(),
            ),
        );
        submit(
            &s,
            owned_op(
                3,
                ResourceClass::DiskWrite,
                vec![],
                vec!["src/b.rs"],
                80,
                active.clone(),
                peak.clone(),
            ),
        );
        s.run_to_completion().await.unwrap();
        let overall = peak.load(Ordering::SeqCst);
        assert!(
            overall <= 2,
            "overlapping writers ran concurrently: peak {overall}"
        );
        assert_eq!(s.status(OpId::new(1)), Some(TaskStatus::Done));
        assert_eq!(s.status(OpId::new(2)), Some(TaskStatus::Done));
        assert_eq!(s.status(OpId::new(3)), Some(TaskStatus::Done));
    }

    #[tokio::test]
    async fn read_vs_write_overlap_serializes() {
        // A read of a.rs while B writes a.rs must serialize (peak 1).
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        submit(
            &s,
            owned_op(
                1,
                ResourceClass::DiskRead,
                vec!["src/a.rs"],
                vec![],
                60,
                active.clone(),
                peak.clone(),
            ),
        );
        submit(
            &s,
            owned_op(
                2,
                ResourceClass::DiskWrite,
                vec![],
                vec!["src/a.rs"],
                60,
                active.clone(),
                peak.clone(),
            ),
        );
        s.run_to_completion().await.unwrap();
        assert_eq!(
            peak.load(Ordering::SeqCst),
            1,
            "read vs write must serialize"
        );
    }

    #[tokio::test]
    async fn directory_write_overlap_is_serialized() {
        // write src/ + write src/x/y.rs → serialize (directory ownership).
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        submit(
            &s,
            owned_op(
                1,
                ResourceClass::DiskWrite,
                vec![],
                vec!["src/"],
                60,
                active.clone(),
                peak.clone(),
            ),
        );
        submit(
            &s,
            owned_op(
                2,
                ResourceClass::DiskWrite,
                vec![],
                vec!["src/x/y.rs"],
                60,
                active.clone(),
                peak.clone(),
            ),
        );
        s.run_to_completion().await.unwrap();
        assert_eq!(
            peak.load(Ordering::SeqCst),
            1,
            "directory write overlap must serialize"
        );
    }

    #[tokio::test]
    async fn disjoint_writes_stay_concurrent() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        submit(
            &s,
            owned_op(
                1,
                ResourceClass::DiskWrite,
                vec![],
                vec!["a.rs"],
                100,
                active.clone(),
                peak.clone(),
            ),
        );
        submit(
            &s,
            owned_op(
                2,
                ResourceClass::DiskWrite,
                vec![],
                vec!["b.rs"],
                100,
                active.clone(),
                peak.clone(),
            ),
        );
        s.run_to_completion().await.unwrap();
        assert_eq!(
            peak.load(Ordering::SeqCst),
            2,
            "disjoint writes must stay concurrent"
        );
    }

    #[tokio::test]
    async fn terminal_policy_accepts_blocked_upstream_no_false_deadlock() {
        // A fails → B (Success) Blocked → C (Terminal edge on B) must RUN,
        // and run_to_completion must NOT report a deadlock.
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let counter = Arc::new(AtomicUsize::new(0));
        let c2 = counter.clone();
        let c3 = counter.clone();
        submit(
            &s,
            ScheduledOp {
                meta: op_meta(1),
                resources: ResourceRequest {
                    class: ResourceClass::Cpu,
                },
                reads: OwnershipSet::new([]),
                writes: OwnershipSet::new([]),
                dependencies: vec![],
                run: Arc::new(|| Box::pin(async { Err(Error::internal("boom")) })),
            },
        );
        submit(
            &s,
            ScheduledOp {
                meta: op_meta(2),
                resources: ResourceRequest {
                    class: ResourceClass::Cpu,
                },
                reads: OwnershipSet::new([]),
                writes: OwnershipSet::new([]),
                dependencies: vec![(OpId::new(1), DependencyPolicy::Success)],
                run: Arc::new(move || {
                    let c = c2.clone();
                    Box::pin(async move {
                        c.fetch_add(1, Ordering::SeqCst);
                        Ok(())
                    })
                }),
            },
        );
        submit(
            &s,
            ScheduledOp {
                meta: op_meta(3),
                resources: ResourceRequest {
                    class: ResourceClass::Cpu,
                },
                reads: OwnershipSet::new([]),
                writes: OwnershipSet::new([]),
                dependencies: vec![(OpId::new(2), DependencyPolicy::Terminal)],
                run: Arc::new(move || {
                    let c = c3.clone();
                    Box::pin(async move {
                        c.fetch_add(1, Ordering::SeqCst);
                        Ok(())
                    })
                }),
            },
        );
        let done = s.run_to_completion().await.unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1, "only C runs");
        assert_eq!(s.status(OpId::new(2)), Some(TaskStatus::Blocked));
        assert_eq!(s.status(OpId::new(3)), Some(TaskStatus::Done));
        assert!(done.contains(&OpId::new(3)));
    }

    /// P0-19: locks the DOCUMENTED definitions to the executable ones.
    /// Terminal = the upstream can make no further execution progress
    /// (`Done | Failed | Cancelled | Blocked` — a blocked task can never
    /// run, so a `Terminal` edge on it must not deadlock); Always = the
    /// CLEANUP edge, defined by purpose (teardown fires once its upstream
    /// settles, however it settled), satisfied by the same terminal set and
    /// released through a blocked chain immediately.
    #[tokio::test]
    async fn terminal_includes_blocked_and_always_is_cleanup() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let runs = Arc::new(AtomicUsize::new(0));
        // A fails. B waits on A with a Success edge → B blocks (dead path).
        submit(
            &s,
            ScheduledOp {
                meta: op_meta(1),
                resources: ResourceRequest {
                    class: ResourceClass::Cpu,
                },
                reads: OwnershipSet::new([]),
                writes: OwnershipSet::new([]),
                dependencies: vec![],
                run: Arc::new(|| Box::pin(async { Err(Error::internal("boom")) })),
            },
        );
        let b = runs.clone();
        submit(
            &s,
            ScheduledOp {
                meta: op_meta(2),
                resources: ResourceRequest {
                    class: ResourceClass::Cpu,
                },
                reads: OwnershipSet::new([]),
                writes: OwnershipSet::new([]),
                dependencies: vec![(OpId::new(1), DependencyPolicy::Success)],
                run: Arc::new(move || {
                    let b = b.clone();
                    Box::pin(async move {
                        b.fetch_add(1, Ordering::SeqCst);
                        Ok(())
                    })
                }),
            },
        );
        // C: Terminal edge on the BLOCKED B — B is terminal (no further
        // progress possible) so C must run, never a deadlock.
        let c = runs.clone();
        submit(
            &s,
            ScheduledOp {
                meta: op_meta(3),
                resources: ResourceRequest {
                    class: ResourceClass::Cpu,
                },
                reads: OwnershipSet::new([]),
                writes: OwnershipSet::new([]),
                dependencies: vec![(OpId::new(2), DependencyPolicy::Terminal)],
                run: Arc::new(move || {
                    let c = c.clone();
                    Box::pin(async move {
                        c.fetch_add(1, Ordering::SeqCst);
                        Ok(())
                    })
                }),
            },
        );
        // D: Always cleanup edge on the same blocked B — fires too.
        let d = runs.clone();
        submit(
            &s,
            ScheduledOp {
                meta: op_meta(4),
                resources: ResourceRequest {
                    class: ResourceClass::Cpu,
                },
                reads: OwnershipSet::new([]),
                writes: OwnershipSet::new([]),
                dependencies: vec![(OpId::new(2), DependencyPolicy::Always)],
                run: Arc::new(move || {
                    let d = d.clone();
                    Box::pin(async move {
                        d.fetch_add(1, Ordering::SeqCst);
                        Ok(())
                    })
                }),
            },
        );
        // E: Success edge on B stays blocked forever (never runs).
        let e = runs.clone();
        submit(
            &s,
            ScheduledOp {
                meta: op_meta(5),
                resources: ResourceRequest {
                    class: ResourceClass::Cpu,
                },
                reads: OwnershipSet::new([]),
                writes: OwnershipSet::new([]),
                dependencies: vec![(OpId::new(2), DependencyPolicy::Success)],
                run: Arc::new(move || {
                    let e = e.clone();
                    Box::pin(async move {
                        e.fetch_add(1, Ordering::SeqCst);
                        Ok(())
                    })
                }),
            },
        );
        let done = s.run_to_completion().await.expect("no false deadlock");
        assert_eq!(s.status(OpId::new(1)), Some(TaskStatus::Failed));
        assert_eq!(s.status(OpId::new(2)), Some(TaskStatus::Blocked));
        assert_eq!(s.status(OpId::new(3)), Some(TaskStatus::Done));
        assert_eq!(s.status(OpId::new(4)), Some(TaskStatus::Done));
        assert_eq!(s.status(OpId::new(5)), Some(TaskStatus::Blocked));
        assert_eq!(runs.load(Ordering::SeqCst), 2, "C and D run exactly once");
        assert!(done.contains(&OpId::new(3)) && done.contains(&OpId::new(4)));
    }

    // ---- audit round 18: duplicate submissions are conflicts, never
    // ---- silent overwrites; hostile DAGs are bounded at submit time. ----

    #[tokio::test]
    async fn duplicate_submit_identical_payload_is_idempotent() {
        // Re-registering the EXACT same op is safe (idempotent re-insert);
        // the scheduler keeps one registration and the op still runs once.
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let counter = Arc::new(AtomicUsize::new(0));
        let op = task(1, vec![], ResourceClass::Cpu, 0, counter.clone());
        assert!(
            s.try_submit(op.clone()).is_ok(),
            "identical re-registration ok"
        );
        assert!(
            s.try_submit(op.clone()).is_ok(),
            "identical re-registration ok"
        );
        assert_eq!(s.statuses().len(), 1, "still exactly one registration");
        let done = s.run_to_completion().await.unwrap();
        assert!(done.contains(&OpId::new(1)));
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "an idempotent re-registration must not run the op twice"
        );
    }

    #[test]
    fn duplicate_submit_different_payload_is_conflict() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let counter = Arc::new(AtomicUsize::new(0));
        let op_a = task(1, vec![], ResourceClass::Cpu, 0, counter.clone());
        assert!(s.try_submit(op_a).is_ok());
        // Same id, DIFFERENT payload (different resource class): conflict,
        // and the FIRST registration survives untouched.
        let op_b = task(1, vec![], ResourceClass::Network, 0, counter.clone());
        let err = s.try_submit(op_b).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Conflict);
        assert!(err.message.contains("different payload"));
        // ...same for a different ownership set and a different edge list.
        let mut op_c = task(1, vec![], ResourceClass::Cpu, 0, counter.clone());
        op_c.writes = OwnershipSet::new(["src/a.rs".to_string()]);
        assert_eq!(s.try_submit(op_c).unwrap_err().kind, ErrorKind::Conflict);
        let op_d = task(1, vec![2], ResourceClass::Cpu, 0, counter.clone());
        assert_eq!(s.try_submit(op_d).unwrap_err().kind, ErrorKind::Conflict);
        assert_eq!(s.statuses().len(), 1, "the original op is never replaced");
        assert_eq!(s.status(OpId::new(1)), Some(TaskStatus::Pending));
        // The legacy infallible submit() shim keeps registration semantics
        // for its out-of-scope caller: an identical re-registration stays a
        // no-op, and a conflicting payload is refused with the FIRST
        // registration untouched — never a silent overwrite. The P0-17
        // audit fix: the rejection is never silently dropped — it is
        // recorded (bounded) and observable via `rejections()`, so a
        // dropped registration cannot hide.
        let identical = task(1, vec![], ResourceClass::Cpu, 0, counter.clone());
        s.submit(identical);
        assert_eq!(s.statuses().len(), 1);
        assert!(
            s.rejections().is_empty(),
            "an idempotent re-registration is not a rejection"
        );
        let conflicting = task(1, vec![], ResourceClass::DiskRead, 0, counter.clone());
        s.submit(conflicting);
        assert_eq!(s.statuses().len(), 1, "the original op is never replaced");
        assert_eq!(s.status(OpId::new(1)), Some(TaskStatus::Pending));
        let rejected = s.rejections();
        assert_eq!(rejected.len(), 1, "the shim must record the rejection");
        assert_eq!(rejected[0].0, OpId::new(1));
        assert!(
            rejected[0].1.contains("different payload"),
            "reason: {}",
            rejected[0].1
        );
    }

    #[test]
    fn submit_rejects_oversized_dependency_fan() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let counter = Arc::new(AtomicUsize::new(0));
        let fan = task(
            1,
            (1..=64).collect(),
            ResourceClass::Cpu,
            0,
            counter.clone(),
        );
        assert!(s.try_submit(fan).is_ok(), "64 dependencies are allowed");
        let too_fan = task(
            2,
            (1..=MAX_DEPENDENCIES_PER_TASK as u64 + 1).collect(),
            ResourceClass::Cpu,
            0,
            counter.clone(),
        );
        let err = s.try_submit(too_fan).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Oversized);
        assert_eq!(s.statuses().len(), 1, "oversized op never registered");
    }

    #[test]
    fn submit_rejects_oversized_ownership_paths() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let counter = Arc::new(AtomicUsize::new(0));
        let writes = (0..MAX_OWNERSHIP_PATHS)
            .map(|i| format!("src/{i}.rs"))
            .collect::<Vec<_>>();
        let mut ok = task(1, vec![], ResourceClass::Cpu, 0, counter.clone());
        ok.writes = OwnershipSet::new(writes.clone());
        assert!(s.try_submit(ok).is_ok(), "512 ownership paths are allowed");
        let mut big = task(2, vec![], ResourceClass::Cpu, 0, counter.clone());
        big.reads = OwnershipSet::new(
            writes
                .iter()
                .take(MAX_OWNERSHIP_PATHS - 100)
                .cloned()
                .collect::<Vec<_>>(),
        );
        big.writes = OwnershipSet::new(writes.iter().skip(100).cloned().collect::<Vec<_>>());
        // reads + writes combined now exceed the per-op ownership cap.
        assert!(big.reads.entries().len() + big.writes.entries().len() > MAX_OWNERSHIP_PATHS);
        let err = s.try_submit(big).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Oversized);
        assert_eq!(s.statuses().len(), 1);
    }

    #[test]
    fn submit_rejects_dags_over_the_task_cap() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let counter = Arc::new(AtomicUsize::new(0));
        for i in 1..=MAX_TASKS_PER_DAG as u64 {
            s.try_submit(task(i, vec![], ResourceClass::Cpu, 0, counter.clone()))
                .expect("dag cap permits exactly MAX_TASKS_PER_DAG registrations");
        }
        let extra = task(
            MAX_TASKS_PER_DAG as u64 + 1,
            vec![],
            ResourceClass::Cpu,
            0,
            counter.clone(),
        );
        let err = s.try_submit(extra).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Oversized);
        assert_eq!(s.statuses().len(), MAX_TASKS_PER_DAG);
        // The whole-DAG validate() reports the same bound.
        s.validate().expect("at-cap dag still validates");
    }

    #[test]
    fn whole_dag_validate_enforces_bounds() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let counter = Arc::new(AtomicUsize::new(0));
        for i in 1..=MAX_TASKS_PER_DAG as u64 {
            s.try_submit(task(i, vec![], ResourceClass::Cpu, 0, counter.clone()))
                .unwrap();
        }
        assert!(s.validate().is_ok());
        // A 1025th submit is already rejected at the door...
        s.try_submit(task(
            MAX_TASKS_PER_DAG as u64 + 1,
            vec![],
            ResourceClass::Cpu,
            0,
            counter.clone(),
        ))
        .unwrap_err();
        assert!(s.validate().is_ok(), "rejected submits leave a valid DAG");
        // ...and validate() is the last line of defense against corrupted
        // state: inject one task past the cap directly and it must fire.
        {
            let mut guard = s.inner.lock().unwrap();
            let op = task(
                MAX_TASKS_PER_DAG as u64 + 1,
                vec![],
                ResourceClass::Cpu,
                0,
                counter.clone(),
            );
            let id = op.meta.operation_id;
            guard.tasks.insert(
                id,
                TaskState {
                    op,
                    status: TaskStatus::Pending,
                    error: None,
                    start_ms: 0,
                    end_ms: None,
                    remaining: 0,
                    blocked: 0,
                    dependents: Vec::new(),
                },
            );
        }
        let err = s.validate().unwrap_err();
        assert_eq!(err.kind, ErrorKind::Oversized);
        // The cycle detection in validate() is unchanged and still fires.
        let s2 = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        s2.try_submit(task(1, vec![2], ResourceClass::Cpu, 0, counter.clone()))
            .unwrap();
        s2.try_submit(task(2, vec![1], ResourceClass::Cpu, 0, counter.clone()))
            .unwrap();
        assert_eq!(s2.validate().unwrap_err().kind, ErrorKind::Deadlock);
    }
}

/// Model-checking property tests (audit 79-80): deterministic seeded traces
/// over the real scheduler API (submit/conflict, cancellation, terminal
/// exactly-once, dependency ordering) with an in-test mirror asserted equal
/// to the scheduler at every step, plus a crash/recovery/epoch driver.
#[cfg(test)]
mod modelcheck;
