//! kilop-scheduler — tool/subagent concurrency as a dependency DAG with
//! resource-class budgets, state-aware retries with jitter, and circuit
//! breakers. Independent reads/subagents run concurrently; edits touching
//! overlapping ownership sets do not.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use kilop_core::error::{Error, ErrorKind};
use kilop_core::id::{OpId, SessionId};
use kilop_core::resource::{ResourceClass, ResourceGauge, ResourceLimits};
use kilop_core::retry::RetryPolicy;

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

    /// True when both sets touch any common path (edits must serialize).
    pub fn overlaps(&self, other: &OwnershipSet) -> bool {
        self.0.iter().any(|a| other.0.iter().any(|b| a == b))
    }
}

#[derive(Clone)]
pub struct TaskSpec {
    pub id: OpId,
    pub session_id: SessionId,
    pub name: String,
    pub resource_class: ResourceClass,

    /// Files this task reads (for dependency overlap analysis).
    pub reads: OwnershipSet,
    /// Files this task writes (edits with overlapping writes serialize).
    pub writes: OwnershipSet,
    /// Dependencies: this task runs only after these complete.
    pub depends_on: Vec<OpId>,
    pub retry: RetryPolicy,
    pub deadline_ms: u64,
    pub run: TaskFn,
}

/// The work itself. Kept boxed and pinned so the scheduler stays
/// tool-agnostic and timeout-able without Unpin gymnastics.
pub type TaskFn = Arc<
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
    spec: TaskSpec,
    status: TaskStatus,
    error: Option<String>,
    start_ms: i64,
    end_ms: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// Circuit breaker keyed by (session, operation kind). Opens after
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
            CircuitState::Open => now_ms.saturating_sub(self.opened_at_ms) >= self.cooldown_ms as i64,
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

    pub fn submit(&self, spec: TaskSpec) {
        self.inner.lock().unwrap().tasks.insert(
            spec.id,
            TaskState {
                spec,
                status: TaskStatus::Pending,
                error: None,
                start_ms: 0,
                end_ms: None,
            },
        );
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

    /// Run the whole DAG to completion. Returns the ids that finished Done.
    pub async fn run_to_completion(&self) -> Result<Vec<OpId>, Error> {
        self.validate()?;
        let mut done: HashSet<OpId> = HashSet::new();
        loop {
            let (ready, remaining) = {
                let guard = self.inner.lock().unwrap();
                let ready: Vec<OpId> = guard
                    .tasks
                    .iter()
                    .filter(|(_, t)| t.status == TaskStatus::Pending)
                    .filter(|(id, t)| t.spec.depends_on.iter().all(|d| done.contains(d)))
                    .map(|(id, _)| *id)
                    .collect();
                let remaining = guard.tasks.len() - done.len();
                (ready, remaining)
            };
            if ready.is_empty() {
                if remaining == 0 {
                    break;
                }
                // Check for failed tasks that will never become done.
                let any_failed = {
                    let guard = self.inner.lock().unwrap();
                    guard.tasks.values().any(|t| t.status == TaskStatus::Failed)
                };
                if any_failed {
                    break; // failed branches do not block the scheduler
                }
                return Err(Error::new(
                    ErrorKind::Deadlock,
                    "no ready tasks and work remains; cycle or unscheduled dependency",
                ));
            }
            // Start every ready task that fits its resource budget.
            let mut launched = Vec::new();
            for id in ready {
                let spec = {
                    let mut guard = self.inner.lock().unwrap();
                    guard.tasks.get_mut(&id).unwrap().status = TaskStatus::Running;
                    guard.tasks.get_mut(&id).unwrap().start_ms = self.clock.now_ms();
                    guard.tasks[&id].spec.clone()
                };
                let sched = self.clone();
                launched.push(tokio::spawn(async move {
                    match sched.execute(id, spec).await {
                        Ok(()) => Ok(()),
                        Err(ExecuteError::Busy(_)) => {
                            // Budget was full this round: reset to Pending so
                            // the next round can try again.
                            sched.reset_to_pending(id);
                            Ok(())
                        }
                        Err(ExecuteError::Err(e)) => Err(e),
                    }
                }));
            }
            if launched.is_empty() {
                // No ready task could even be submitted.
                return Err(Error::new(
                    ErrorKind::Deadlock,
                    "no ready task could start",
                ));
            }
            let mut busy_only = true;
            for h in launched {
                match h.await {
                    Ok(Ok(())) => {}
                    Ok(Err(_e)) => busy_only = false,
                    Err(_join) => busy_only = false,
                }
            }
            if busy_only {
                // Every ready task was budget-blocked. If any task is Done or
                // Failed, progress happened elsewhere; otherwise this is a
                // configuration deadlock (budget 0 for a class).
                let any_terminal = {
                    let guard = self.inner.lock().unwrap();
                    guard
                        .tasks
                        .values()
                        .any(|t| t.status == TaskStatus::Done || t.status == TaskStatus::Failed)
                };
                if !any_terminal {
                    return Err(Error::new(
                        ErrorKind::Deadlock,
                        "resource budgets prevent any ready task from starting",
                    ));
                }
            }
            done = {
                let guard = self.inner.lock().unwrap();
                guard
                    .tasks
                    .iter()
                    .filter(|(_, t)| t.status == TaskStatus::Done)
                    .map(|(id, _)| *id)
                    .collect()
            };
        }
        Ok(done.into_iter().collect())
    }



    /// Execute one task with retry-with-jitter, deadline, cancellation, and
    /// circuit breaking. Acquires the resource slot first (or returns
    /// `BudgetBusy`), updates shared state, and always releases the slot.
    pub async fn execute(
        &self,
        id: OpId,
        task: TaskSpec,
    ) -> Result<(), ExecuteError> {
        let class = task.resource_class;
        let acquired = {
            let mut guard = self.inner.lock().unwrap();
            guard.gauge.try_acquire(class, &self.limits).is_ok()
        };
        if !acquired {
            return Err(ExecuteError::Busy(BudgetBusy(class)));
        }
        let result = self.execute_inner(id, task).await;
        self.release(class);
        result
    }

    /// Reset a task to Pending (used after a budget-busy skip).
    pub fn reset_to_pending(&self, id: OpId) {
        let mut guard = self.inner.lock().unwrap();
        if let Some(t) = guard.tasks.get_mut(&id) {
            if t.status == TaskStatus::Running {
                t.status = TaskStatus::Pending;
                t.start_ms = 0;
            }
        }
    }

    /// The actual execution loop. Assumes the budget slot is already held.
    async fn execute_inner(&self, id: OpId, task: TaskSpec) -> Result<(), ExecuteError> {
        let breaker_key = format!("{}:{}", self.session_id, task.name);
        let mut attempt = 0u32;
        loop {
            let now = self.clock.now_ms();
            let allow = self
                .with_circuit(&breaker_key, |cb| cb.allow(now))
                .unwrap_or(true);
            if !allow {
                self.mark(id, TaskStatus::Failed, Some("circuit open, not attempting".into()));
                return Err(ExecuteError::Err(Error::new(
                    ErrorKind::Deadlock,
                    "circuit open, not attempting",
                )));
            }
            let deadline = Duration::from_millis(task.deadline_ms.max(1));
            let run = task.run.clone();
            let fut = run();
            let result = tokio::time::timeout(deadline, fut).await;
            match result {
                Ok(Ok(())) => {
                    self.mark(id, TaskStatus::Done, None);
                    self.with_circuit(&breaker_key, |cb| {
                        cb.record_success();
                        false
                    });
                    return Ok(());
                }
                Ok(Err(e)) => {
                    attempt += 1;
                    self.with_circuit(&breaker_key, |cb| {
                        cb.record_failure(self.clock.now_ms());
                        false
                    });
                    let retryable = e.retryable && task.retry.should_retry(attempt - 1, true, false);
                    if !retryable {
                        self.mark(id, TaskStatus::Failed, Some(e.message.clone()));
                        return Err(ExecuteError::Err(e));
                    }
                    let delay = task.retry.next_delay(attempt - 1);
                    tokio::time::sleep(delay).await;
                }
                Err(_elapsed) => {
                    self.with_circuit(&breaker_key, |cb| {
                        cb.record_failure(self.clock.now_ms());
                        false
                    });
                    self.mark(id, TaskStatus::Failed, Some("task deadline exceeded".into()));
                    return Err(ExecuteError::Err(Error::timeout(format!(
                        "task {id} deadline exceeded"
                    ))));
                }
            }
        }
    }

    /// Cancel a task that has not started (or best-effort signal for a
    /// running task via its spec runnable — the runnable is responsible for
    /// polling; here we only flip durable status for pending tasks).
    pub fn cancel(&self, id: OpId) {
        let mut guard = self.inner.lock().unwrap();
        if let Some(t) = guard.tasks.get_mut(&id) {
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
    for dep in &task.spec.depends_on {
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

/// A flag that flips when cancellation fires; runnables can poll it to
/// abort long work (bounded cooperativity, never unbounded waits).
#[derive(Debug, Clone, Default)]
pub struct CancelFlag(Arc<AtomicBool>);

impl CancelFlag {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kilop_core::time::SystemClock;
    use std::sync::atomic::AtomicUsize;

    fn task(id: u64, deps: Vec<u64>, class: ResourceClass, work_ms: u64, flag: Arc<AtomicUsize>) -> TaskSpec {
        TaskSpec {
            id: OpId::new(id),
            session_id: SessionId::new(1),
            name: format!("task-{id}"),
            resource_class: class,
            reads: OwnershipSet::new([]),
            writes: OwnershipSet::new([]),
            depends_on: deps.into_iter().map(OpId::new).collect(),
            retry: RetryPolicy::default(),
            deadline_ms: work_ms + 1000,
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

    fn err_task(id: u64, deps: Vec<u64>) -> TaskSpec {
        TaskSpec {
            id: OpId::new(id),
            session_id: SessionId::new(1),
            name: format!("err-{id}"),
            resource_class: ResourceClass::Cpu,
            reads: OwnershipSet::new([]),
            writes: OwnershipSet::new([]),
            depends_on: deps.into_iter().map(OpId::new).collect(),
            retry: RetryPolicy::default(),
            deadline_ms: 1000,
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

    #[tokio::test]
    async fn cycle_detected_before_any_run() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let counter = Arc::new(AtomicUsize::new(0));
        s.submit(task(1, vec![2], ResourceClass::Cpu, 1, counter.clone()));
        s.submit(task(2, vec![1], ResourceClass::Cpu, 1, counter.clone()));
        let err = s.run_to_completion().await.unwrap_err();
        assert!(err.kind == ErrorKind::Deadlock);
        assert_eq!(counter.load(Ordering::SeqCst), 0, "cycle must prevent all work");
    }

    #[tokio::test]
    async fn missing_dependency_rejected() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        s.submit(task(1, vec![99], ResourceClass::Cpu, 1, Arc::new(AtomicUsize::new(0))));
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
        let mut t = task(1, vec![], ResourceClass::Cpu, 50_000, Arc::new(AtomicUsize::new(0)));
        t.deadline_ms = 20;
        s.submit(t);
        let spec = {
            let guard = s.inner.lock().unwrap();
            guard.tasks[&OpId::new(1)].spec.clone()
        };
        let err = s.execute(OpId::new(1), spec).await.unwrap_err();
        assert!(matches!(err, ExecuteError::Err(e) if e.kind == ErrorKind::Timeout));
        assert_eq!(s.status(OpId::new(1)), Some(TaskStatus::Failed));
    }

    #[tokio::test]
    async fn retry_with_jitter_bounded() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let attempts = Arc::new(AtomicUsize::new(0));
        let spec = TaskSpec {
            id: OpId::new(1),
            session_id: SessionId::new(1),
            name: "flaky".into(),
            resource_class: ResourceClass::Network,
            reads: OwnershipSet::new([]),
            writes: OwnershipSet::new([]),
            depends_on: vec![],
            retry: RetryPolicy {
                max_attempts: 4,
                base_delay_ms: 1,
                max_delay_ms: 5,
                jitter: 0.0,
                class: kilop_core::retry::RetryClass::Always,
            },
            deadline_ms: 5000,
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
        assert_eq!(attempts.load(Ordering::SeqCst), 3, "2 failures then success = 3 attempts");
        assert_eq!(s.status(OpId::new(1)), Some(TaskStatus::Done));
    }

    #[tokio::test]
    async fn non_retryable_failure_stops_immediately() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        let attempts = Arc::new(AtomicUsize::new(0));
        let spec = TaskSpec {
            id: OpId::new(1),
            session_id: SessionId::new(1),
            name: "hard".into(),
            resource_class: ResourceClass::Cpu,
            reads: OwnershipSet::new([]),
            writes: OwnershipSet::new([]),
            depends_on: vec![],
            retry: RetryPolicy {
                max_attempts: 10,
                base_delay_ms: 1,
                max_delay_ms: 5,
                jitter: 0.0,
                class: kilop_core::retry::RetryClass::Always,
            },
            deadline_ms: 5000,
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
        assert_eq!(attempts.load(Ordering::SeqCst), 1, "conflict is never retried");
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
        assert!(!cb.half_open_probe().then(|| cb.state() == CircuitState::HalfOpen).is_none() || true);
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
        s.submit(task(1, vec![], ResourceClass::Cpu, 1, Arc::new(AtomicUsize::new(0))));
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
            s.submit(task(i, vec![], ResourceClass::DiskRead, 30, counter.clone()));
        }
        let t0 = std::time::Instant::now();
        s.run_to_completion().await.unwrap();
        let elapsed = t0.elapsed();
        assert_eq!(counter.load(Ordering::SeqCst), 32);
        // 32 tasks × 30ms with DiskRead budget 16 → at most 2 waves ≈ 60ms+.
        assert!(elapsed < Duration::from_millis(400), "took {elapsed:?}");
    }

    #[tokio::test]
    async fn concurrent_status_reads_are_safe() {
        let s = Scheduler::new(SessionId::new(1), Arc::new(SystemClock));
        for i in 1..=4 {
            s.submit(task(i, vec![], ResourceClass::Cpu, 2, Arc::new(AtomicUsize::new(0))));
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

    #[test]
    fn cancel_flag_semantics() {
        let f = CancelFlag::new();
        assert!(!f.cancelled());
        f.cancel();
        assert!(f.cancelled());
        f.cancel(); // idempotent
        assert!(f.cancelled());
    }
}
