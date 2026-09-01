//! kilop-scheduler — tool/subagent concurrency as a dependency DAG with
//! resource-class budgets, state-aware retries with jitter, and circuit
//! breakers. Independent reads/subagents run concurrently; edits touching
//! overlapping ownership sets do not.
//!
//! Scheduling is event-driven: a task starts only when its dependencies are
//! terminal AND it holds a resource permit (permit-before-spawn). Completion
//! of any task immediately frees its permit and decrements its dependents,
//! so a long task never gates short tasks that became ready after it.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use kilop_core::error::{Error, ErrorKind};
use kilop_core::id::{OpId, SessionId};
use kilop_core::op::OpMeta;
use kilop_core::resource::{ResourceClass, ResourceGauge, ResourceLimits};

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

fn path_overlaps(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
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

/// Which resource class an operation draws from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceRequest {
    pub class: ResourceClass,
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
    /// Dependencies: this task runs only after these complete.
    pub depends_on: Vec<OpId>,
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
    /// Unmet dependencies (terminal deps do not count).
    remaining: usize,
    /// Tasks that wait on this one.
    dependents: Vec<OpId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// Circuit breaker keyed by (session, operation). Opens after
/// `failure_threshold` consecutive failures; a probe is allowed after
/// cooldown; success closes it.
#[derive(Debug)]
pub struct CircuitBreaker {
    failure_threshold: u32,
    cooldown_ms: u64,
    failures: u32,
    state: CircuitState,
    opened_at_ms: i64,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, cooldown_ms: u64) -> Self {
        Self {
            failure_threshold: failure_threshold.max(1),
            cooldown_ms,
            failures: 0,
            state: CircuitState::Closed,
            opened_at_ms: 0,
        }
    }

    pub fn state(&self) -> CircuitState {
        self.state
    }

    /// May this call proceed? (Open + cooldown elapsed = allow one probe.)
    pub fn allow(&self, now_ms: i64) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                now_ms.saturating_sub(self.opened_at_ms) >= self.cooldown_ms as i64
            }
            CircuitState::HalfOpen => true,
        }
    }

    pub fn record_success(&mut self) {
        self.failures = 0;
        self.state = CircuitState::Closed;
    }

    pub fn record_failure(&mut self, now_ms: i64) {
        self.failures += 1;
        match self.state {
            CircuitState::HalfOpen => {
                self.state = CircuitState::Open;
                self.opened_at_ms = now_ms;
            }
            CircuitState::Closed if self.failures >= self.failure_threshold => {
                self.state = CircuitState::Open;
                self.opened_at_ms = now_ms;
            }
            _ => {}
        }
    }

    /// Transition Open → HalfOpen for a probe. Returns false if not Open.
    pub fn half_open_probe(&mut self) -> bool {
        if self.state == CircuitState::Open {
            self.state = CircuitState::HalfOpen;
            true
        } else {
            false
        }
    }
}

#[derive(Default)]
struct Inner {
    tasks: HashMap<OpId, TaskState>,
    gauge: ResourceGauge,
    circuits: HashMap<String, CircuitBreaker>,
    /// FIFO of tasks whose dependencies are satisfied and that still need a
    /// permit. Tasks deferred for a busy budget cycle back to the tail.
    ready: VecDeque<OpId>,
}

/// A scheduler for one session. Clonable handle; all execution goes through
/// shared state, so concurrent task execution and status reads are safe.
#[derive(Clone)]
pub struct Scheduler {
    session_id: SessionId,
    limits: Arc<ResourceLimits>,
    inner: Arc<Mutex<Inner>>,
    clock: Arc<dyn kilop_core::time::Clock>,
}

impl Scheduler {
    pub fn new(session_id: SessionId, clock: Arc<dyn kilop_core::time::Clock>) -> Self {
        Self {
            session_id,
            limits: Arc::new(ResourceLimits::default()),
            inner: Arc::new(Mutex::new(Inner::default())),
            clock,
        }
    }

    pub fn with_limits(self, limits: ResourceLimits) -> Self {
        Self {
            limits: Arc::new(limits),
            ..self
        }
    }

    pub fn submit(&self, op: ScheduledOp) {
        let mut guard = self.inner.lock().unwrap();
        let id = op.meta.operation_id;
        let mut remaining = op.depends_on.len();
        for dep in &op.depends_on {
            if let Some(t) = guard.tasks.get_mut(dep) {
                if matches!(
                    t.status,
                    TaskStatus::Done | TaskStatus::Failed | TaskStatus::Cancelled
                ) {
                    remaining -= 1;
                } else {
                    t.dependents.push(id);
                }
            }
        }
        guard.tasks.insert(
            id,
            TaskState {
                op,
                status: TaskStatus::Pending,
                error: None,
                start_ms: 0,
                end_ms: None,
                remaining,
                dependents: Vec::new(),
            },
        );
        if remaining == 0 {
            guard.ready.push_back(id);
        }
    }

    pub fn status(&self, id: OpId) -> Option<TaskStatus> {
        self.inner.lock().unwrap().tasks.get(&id).map(|t| t.status)
    }

    /// Validate the DAG before running: unknown dependencies and cycles are
    /// loud errors, never silent deadlocks.
    pub fn validate(&self) -> Result<(), Error> {
        let tasks = &self.inner.lock().unwrap().tasks;
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
                    if guard.gauge.try_acquire(class, &self.limits).is_err() {
                        guard.ready.push_back(id); // retry when a permit frees
                        deferred += 1;
                        continue;
                    }
                    let t = guard.tasks.get_mut(&id).unwrap();
                    t.status = TaskStatus::Running;
                    t.start_ms = self.clock.now_ms();
                    to_spawn.push((id, t.op.clone()));
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
                    let result = futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(
                        sched.execute_inner(id, op),
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

    /// A task completed (any outcome): free its permit and decrement every
    /// dependent; dependents that reach zero unmet deps join the ready queue
    /// immediately.
    fn on_complete(&self, id: OpId) {
        let mut guard = self.inner.lock().unwrap();
        let class = guard.tasks[&id].op.resources.class;
        guard.gauge.release(class);
        let dependents = guard
            .tasks
            .get_mut(&id)
            .map(|t| std::mem::take(&mut t.dependents))
            .unwrap_or_default();
        for dep in dependents {
            if let Some(t) = guard.tasks.get_mut(&dep) {
                t.remaining = t.remaining.saturating_sub(1);
                if t.remaining == 0 && t.status == TaskStatus::Pending {
                    guard.ready.push_back(dep);
                }
            }
        }
    }

    /// Recompute the dependency graph from live state (dependents, unmet
    /// counts, ready queue). Terminal dependencies satisfy the DAG edge.
    fn rebuild_graph(&self) {
        let mut guard = self.inner.lock().unwrap();
        guard.ready.clear();
        let ids: Vec<OpId> = guard.tasks.keys().copied().collect();
        for t in guard.tasks.values_mut() {
            t.remaining = 0;
            t.dependents.clear();
        }
        for id in ids {
            let deps: Vec<OpId> = guard.tasks[&id].op.depends_on.clone();
            let mut remaining = 0usize;
            for dep in deps {
                let dep_terminal = matches!(
                    guard.tasks[&dep].status,
                    TaskStatus::Done | TaskStatus::Failed | TaskStatus::Cancelled
                );
                if !dep_terminal {
                    remaining += 1;
                    guard.tasks.get_mut(&dep).unwrap().dependents.push(id);
                }
            }
            let t = guard.tasks.get_mut(&id).unwrap();
            t.remaining = remaining;
            if remaining == 0 && t.status == TaskStatus::Pending {
                guard.ready.push_back(id);
            }
        }
    }

    /// Execute one task with retry-with-jitter, deadline, cancellation, and
    /// circuit breaking. Acquires the resource slot first (or returns
    /// `BudgetBusy`), updates shared state, and always releases the slot.
    pub async fn execute(&self, id: OpId, op: ScheduledOp) -> Result<(), ExecuteError> {
        let class = op.resources.class;
        let acquired = {
            let mut guard = self.inner.lock().unwrap();
            guard.gauge.try_acquire(class, &self.limits).is_ok()
        };
        if !acquired {
            return Err(ExecuteError::Busy(BudgetBusy(class)));
        }
        let result = self.execute_inner(id, op).await;
        self.release(class);
        result
    }

    /// The actual execution loop. Assumes the budget slot is already held.
    async fn execute_inner(&self, id: OpId, op: ScheduledOp) -> Result<(), ExecuteError> {
        let breaker_key = format!("{}:{}", self.session_id, id);
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
            let allow = self
                .with_circuit(&breaker_key, |cb| cb.allow(now))
                .unwrap_or(true);
            if !allow {
                self.mark(
                    id,
                    TaskStatus::Failed,
                    Some("circuit open, not attempting".into()),
                );
                return Err(ExecuteError::Err(Error::new(
                    ErrorKind::Deadlock,
                    "circuit open, not attempting",
                )));
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
                    self.with_circuit(&breaker_key, |cb| {
                        cb.record_success();
                        false
                    });
                    return Ok(());
                }
                Ok(Err(e)) => {
                    if op.meta.cancellation.is_cancelled() {
                        self.mark(id, TaskStatus::Cancelled, None);
                        return Ok(());
                    }
                    attempt += 1;
                    self.with_circuit(&breaker_key, |cb| {
                        cb.record_failure(self.clock.now_ms());
                        false
                    });
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
                    self.with_circuit(&breaker_key, |cb| {
                        cb.record_failure(self.clock.now_ms());
                        false
                    });
                    let msg = format!("op {id} deadline exceeded");
                    self.mark(id, TaskStatus::Failed, Some(msg.clone()));
                    return Err(ExecuteError::Err(Error::timeout(msg)));
                }
            }
        }
    }

    /// Cancel a task: the op's own cancellation token fires (so a running
    /// runnable that polls it aborts promptly) and pending tasks flip to
    /// Cancelled immediately.
    pub fn cancel(&self, id: OpId) {
        let mut guard = self.inner.lock().unwrap();
        if let Some(t) = guard.tasks.get_mut(&id) {
            t.op.meta.cancellation.cancel();
            if t.status == TaskStatus::Pending {
                t.status = TaskStatus::Cancelled;
            }
        }
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

    fn mark(&self, id: OpId, status: TaskStatus, error: Option<String>) {
        let mut guard = self.inner.lock().unwrap();
        if let Some(t) = guard.tasks.get_mut(&id) {
            t.status = status;
            t.error = error;
            t.end_ms = Some(self.clock.now_ms());
        }
    }

    fn release(&self, class: ResourceClass) {
        let mut guard = self.inner.lock().unwrap();
        guard.gauge.release(class);
    }

    fn with_circuit<T>(&self, key: &str, f: impl FnOnce(&mut CircuitBreaker) -> T) -> Option<T> {
        let mut guard = self.inner.lock().unwrap();
        guard
            .circuits
            .entry(key.to_string())
            .or_insert_with(|| CircuitBreaker::new(4, 5_000));
        f(guard.circuits.get_mut(key).unwrap()).into()
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
    for dep in &task.op.depends_on {
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
    use kilop_core::cancellation::CancellationToken;
    use kilop_core::hash::FileHash;
    use kilop_core::op::RecoveryStrategy;
    use kilop_core::retry::{RetryClass, RetryPolicy};
    use kilop_core::time::{Clock, Deadline, SystemClock};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    const FAR: i64 = i64::MAX / 2;

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
            depends_on: deps.into_iter().map(OpId::new).collect(),
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
            depends_on: deps.into_iter().map(OpId::new).collect(),
            run: Arc::new(|| Box::pin(async { Err(Error::internal("boom")) })),
        }
    }

    #[tokio::test]
    async fn dag_runs_dependencies_before_dependents() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let counter = Arc::new(AtomicUsize::new(0));
        s.submit(task(1, vec![], ResourceClass::Cpu, 5, counter.clone()));
        s.submit(task(2, vec![], ResourceClass::Cpu, 5, counter.clone()));
        s.submit(task(3, vec![1, 2], ResourceClass::Cpu, 5, counter.clone()));
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
        s.submit(task(1, vec![2], ResourceClass::Cpu, 1, counter.clone()));
        s.submit(task(2, vec![1], ResourceClass::Cpu, 1, counter.clone()));
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
        s.submit(task(
            1,
            vec![99],
            ResourceClass::Cpu,
            1,
            Arc::new(AtomicUsize::new(0)),
        ));
        let err = s.run_to_completion().await.unwrap_err();
        assert!(err.kind == ErrorKind::NotFound);
    }

    #[tokio::test]
    async fn task_failure_does_not_block_other_branches() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let counter = Arc::new(AtomicUsize::new(0));
        s.submit(err_task(1, vec![]));
        s.submit(task(2, vec![], ResourceClass::Cpu, 1, counter.clone()));
        let result = s.run_to_completion().await;
        assert!(result.is_ok());
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert_eq!(s.status(OpId::new(1)), Some(TaskStatus::Failed));
        assert_eq!(s.status(OpId::new(2)), Some(TaskStatus::Done));
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
        s.submit(t);
        let spec = {
            let guard = s.inner.lock().unwrap();
            guard.tasks[&OpId::new(1)].op.clone()
        };
        let err = s.execute(OpId::new(1), spec).await.unwrap_err();
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
            depends_on: vec![],
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
        s.submit(spec.clone());
        s.execute(OpId::new(1), spec).await.unwrap();
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
            depends_on: vec![],
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
        s.submit(spec.clone());
        let err = s.execute(OpId::new(1), spec).await.unwrap_err();
        assert!(matches!(err, ExecuteError::Err(e) if e.kind == ErrorKind::Conflict));
        assert_eq!(
            attempts.load(Ordering::SeqCst),
            1,
            "conflict is never retried"
        );
    }

    #[test]
    fn circuit_breaker_opens_and_cooldowns() {
        let mut cb = CircuitBreaker::new(3, 1000);
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.allow(0));
        cb.record_failure(1);
        cb.record_failure(2);
        assert_eq!(cb.state(), CircuitState::Closed, "threshold is 3");
        cb.record_failure(3);
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.allow(4), "cooldown not elapsed");
        assert!(cb.allow(1000 + 1001), "cooldown elapsed allows a probe");
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
        cb.record_failure(10);
        cb.record_failure(11);
        cb.record_failure(12);
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(cb.half_open_probe());
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        cb.record_failure(13);
        assert_eq!(cb.state(), CircuitState::Open, "half-open failure re-opens");
        assert!(
            cb.half_open_probe(),
            "a re-opened breaker must admit a new probe"
        );
        assert_eq!(cb.state(), CircuitState::HalfOpen);
    }

    #[tokio::test]
    async fn resource_budget_prevents_starvation() {
        let mut limits = ResourceLimits::default();
        limits.limits.insert(ResourceClass::Indexing, 1);
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock)).with_limits(limits);
        let counter = Arc::new(AtomicUsize::new(0));
        for i in 1..=5 {
            s.submit(task(i, vec![], ResourceClass::Indexing, 1, counter.clone()));
        }
        s.submit(task(9, vec![], ResourceClass::Model, 1, counter.clone()));
        s.run_to_completion().await.unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 6, "nothing starved");
    }

    #[tokio::test]
    async fn cancel_pending_task() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        s.submit(task(
            1,
            vec![],
            ResourceClass::Cpu,
            1,
            Arc::new(AtomicUsize::new(0)),
        ));
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
            s.submit(task(
                i,
                vec![],
                ResourceClass::DiskRead,
                30,
                counter.clone(),
            ));
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
            s.submit(task(
                i,
                vec![],
                ResourceClass::Cpu,
                2,
                Arc::new(AtomicUsize::new(0)),
            ));
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

    /// A herd of 2000 queued tasks on a budget of 16 must never run more
    /// than 16 concurrently: the permit is acquired before the task is
    /// spawned, so the runnable's own in-flight counter is the proof.
    #[tokio::test(flavor = "multi_thread", worker_threads = 32)]
    async fn permit_before_spawn_prevents_herd() {
        let mut limits = ResourceLimits::default();
        limits.limits.insert(ResourceClass::DiskRead, 16);
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock)).with_limits(limits);
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let completed = Arc::new(AtomicUsize::new(0));
        for i in 1..=2000 {
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
                depends_on: vec![],
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
            s.submit(op);
        }
        s.run_to_completion().await.unwrap();
        assert_eq!(
            completed.load(Ordering::SeqCst),
            2000,
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
            depends_on: vec![],
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
            depends_on: vec![],
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
            depends_on: vec![OpId::new(2)],
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
        s.submit(op_a);
        s.submit(op_b);
        s.submit(op_c);
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
            depends_on: vec![],
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
        s.submit(op);
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
        s.submit(op);
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
        s.submit(failing);
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
        s.submit(cancelled);
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
        s.submit(panicking);
        s.submit(healthy);
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
}
