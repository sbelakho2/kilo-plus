//! `DbReadService`: the bounded async SQLite read pool (audit 13/32).
//!
//! `DbActor` (audit 42) moved the hot WRITE paths off Tokio workers; the
//! synchronous bounded reads (conversation windows, budget views,
//! verification records, prefix rows, memory pages — SQLite I/O + JSON
//! decode) still ran inline on Tokio workers. This module gives those reads
//! the same discipline, with the audit's explicit requirement: a BOUNDED
//! worker pool (2-4 `std::thread`s, hard-capped by
//! [`MAX_READ_WORKERS`], configurable) — never unrestricted
//! `spawn_blocking` everywhere.
//!
//! Shape:
//!
//! - A small pool of dedicated **`std::thread`s** (never tokio tasks)
//!   executes the store reads. The threads block only on std primitives
//!   (a mutex+condvar work queue), so they need no runtime.
//! - Outstanding reads are bounded by a **`tokio::sync::Semaphore`** of
//!   `DbReadServiceConfig::capacity` permits: a caller acquires one permit
//!   before enqueueing (awaiting capacity = genuine backpressure) and the
//!   executing worker releases it when the read completes. A saturated
//!   service therefore holds at most `capacity` queued+executing reads —
//!   never unbounded buffering, no OOM. Concurrent EXECUTION is bounded by
//!   the worker count (a worker runs one read at a time).
//! - Workers are spawned LAZILY on the first submit (a manager that never
//!   awaits a bounded read pays nothing) and exit by themselves once the
//!   queue is closed AND drained: every queued read completes first, then
//!   the threads end (drop-the-last-handle semantics identical to
//!   `DbActor`).
//! - A panicking store read is caught per job: its caller receives a typed
//!   error (never a hang) and the worker survives to serve the next read.
//! - Instrumentation, not inference ([`DbReadStats`]): concurrent-execution
//!   high-water (`max_active`) and queue-occupancy high-water
//!   (`max_queue_depth`). The bounded-concurrency and overload gates assert
//!   against these.
//!
//! The async surface is generic ([`DbReadService::submit`]); the
//! `SessionManager` read wrappers (`manager.rs`) submit the SAME sync store
//! reads the turn machinery used to run inline, so result parity is by
//! construction.

use std::any::Any;
use std::collections::VecDeque;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use faktor_core::Error;
use faktor_store::Store;
use tokio::sync::{oneshot, OwnedSemaphorePermit, Semaphore};

/// Hard cap of the worker pool. The audit's bound is 2-4 workers;
/// [`DbReadServiceConfig::workers`] is clamped into `2..=MAX_READ_WORKERS`.
pub const MAX_READ_WORKERS: usize = 4;
/// Default worker count (the audit's upper bound: four parallel reads).
pub const DEFAULT_READ_WORKERS: usize = 4;
/// Default bounded permit count (queued + executing reads; see
/// [`DbReadServiceConfig::capacity`]).
pub const DEFAULT_READ_QUEUE_CAPACITY: usize = 1024;

/// Thread-name prefix of the read pool (observable in crash forensics).
const READ_THREAD_NAME: &str = "faktor-db-read";

/// One queued read job: the store read to execute, its reply channel, and
/// the capacity permit that is released when the read completes. The boxed
/// payload is type-erased at the submit boundary; a worker never
/// interprets it, so a mismatch is impossible except for a bug in the
/// submit/downcast pair (which fails loudly, never silently).
struct Job {
    reply: oneshot::Sender<std::result::Result<Box<dyn Any + Send + 'static>, String>>,
    #[allow(clippy::type_complexity)]
    run: Box<dyn FnOnce(&Store) -> Box<dyn Any + Send + 'static> + Send + 'static>,
    _permit: OwnedSemaphorePermit,
}

/// Tuning knobs of a [`DbReadService`].
#[derive(Debug, Clone)]
pub struct DbReadServiceConfig {
    /// Worker threads of the pool, clamped into `2..=MAX_READ_WORKERS`
    /// (default [`DEFAULT_READ_WORKERS`]). The clamp is a HARD cap: a
    /// caller requesting more workers gets `MAX_READ_WORKERS`, never an
    /// unbounded pool.
    pub workers: usize,
    /// Bounded permit count of the service (default
    /// [`DEFAULT_READ_QUEUE_CAPACITY`]): at most this many reads are
    /// queued or executing at any instant. Callers await capacity when it
    /// is exhausted — genuine backpressure, never unbounded growth.
    pub capacity: usize,
    /// Test seam: sleep this long before executing every read to make
    /// concurrency/backpressure deterministic. `None` in production.
    #[doc(hidden)]
    pub pre_read_delay: Option<Duration>,
}

impl Default for DbReadServiceConfig {
    fn default() -> Self {
        Self {
            workers: DEFAULT_READ_WORKERS,
            capacity: DEFAULT_READ_QUEUE_CAPACITY,
            pre_read_delay: None,
        }
    }
}

impl DbReadServiceConfig {
    fn clamped(&self) -> Self {
        Self {
            workers: self.workers.clamp(2, MAX_READ_WORKERS),
            capacity: self.capacity.max(1),
            pre_read_delay: self.pre_read_delay,
        }
    }
}

/// The tagged class of one bounded read (audit 13: the DB-read-pool
/// tripwire needs per-capability counts, not just a total). Every manager
/// read wrapper submits under its own class; [`DbReadService::submit`] (the
/// generic seam) tags [`DbReadKind::Other`]. The classes are the turn
/// machinery's real read set: conversation history, task rows, budget
/// views, prefix observations, verification records and memory pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DbReadKind {
    History,
    Task,
    Budget,
    Prefix,
    Verification,
    Memory,
    Other,
}

impl DbReadKind {
    /// Every tag, in stable index order (the stats array order).
    pub const ALL: [DbReadKind; 7] = [
        DbReadKind::History,
        DbReadKind::Task,
        DbReadKind::Budget,
        DbReadKind::Prefix,
        DbReadKind::Verification,
        DbReadKind::Memory,
        DbReadKind::Other,
    ];

    /// Stable array index of this tag.
    pub const fn index(self) -> usize {
        match self {
            DbReadKind::History => 0,
            DbReadKind::Task => 1,
            DbReadKind::Budget => 2,
            DbReadKind::Prefix => 3,
            DbReadKind::Verification => 4,
            DbReadKind::Memory => 5,
            DbReadKind::Other => 6,
        }
    }

    /// Stable human/telemetry label (never the Debug spelling).
    pub const fn label(self) -> &'static str {
        match self {
            DbReadKind::History => "history",
            DbReadKind::Task => "task",
            DbReadKind::Budget => "budget",
            DbReadKind::Prefix => "prefix",
            DbReadKind::Verification => "verification",
            DbReadKind::Memory => "memory",
            DbReadKind::Other => "other",
        }
    }
}

/// Instrumentation snapshot of a [`DbReadService`] (audit gate 13:
/// instrumented, never inferred).
#[derive(Debug, Clone, Default)]
pub struct DbReadStats {
    /// The configured (clamped) worker count.
    pub workers: usize,
    /// Reads submitted by callers.
    pub enqueued: u64,
    /// Reads that ran to completion on a worker (Ok or error reply).
    pub completed: u64,
    /// High-water of concurrently executing reads. Never exceeds
    /// `workers` (by construction) — the bounded-concurrency gate.
    pub max_active: u64,
    /// High-water of the shared work queue (occupancy after every
    /// enqueue). Never exceeds `capacity` (by construction) — the
    /// no-OOM gate.
    pub max_queue_depth: u64,
    /// Per-tag submitted counts, indexed by [`DbReadKind::index`]: the
    /// tagged runtime tripwire (`enqueued > 0` plus one count per
    /// capability the read pool serves).
    pub tagged: [u64; DbReadKind::ALL.len()],
}

impl DbReadStats {
    /// Submitted count of one tag.
    pub fn kind(&self, kind: DbReadKind) -> u64 {
        self.tagged.get(kind.index()).copied().unwrap_or(0)
    }

    /// Submitted count by stable label.
    pub fn kind_label(&self, label: &str) -> u64 {
        self.tagged_by_kind()
            .find(|(k, _)| k.label() == label)
            .map(|(_, n)| n)
            .unwrap_or(0)
    }

    fn tagged_by_kind(&self) -> impl Iterator<Item = (DbReadKind, u64)> + '_ {
        DbReadKind::ALL.iter().map(|k| (*k, self.kind(*k)))
    }
}

/// Shared state between the service handle and the worker threads.
struct ReadShared {
    store: Arc<Store>,
    cfg: DbReadServiceConfig,
    /// Serializes the lazy one-time pool start.
    start_lock: Mutex<()>,
    /// Backpressure limiter: one permit per queued+executing read.
    semaphore: Arc<Semaphore>,
    /// The shared FIFO work queue + its wakeup condvar.
    queue: Mutex<VecDeque<Job>>,
    wake: Condvar,
    /// Set when the service is closed: no new submissions, workers drain
    /// the queue and exit.
    closed: AtomicBool,
    /// Live worker threads (spawned lazily, decremented on exit).
    running: AtomicU64,
    /// Condvar/flag pair: last-worker-exit signal, waitable from async
    /// tests.
    exit: ExitFlag,
    stats: StatsCore,
}

/// Condvar-backed last-worker-exit signal.
#[derive(Default)]
struct ExitFlag {
    state: Mutex<bool>,
    cv: Condvar,
}

impl ExitFlag {
    fn notify_exited(&self) {
        *self.state.lock().unwrap_or_else(|p| p.into_inner()) = true;
        self.cv.notify_all();
    }
}

#[derive(Default)]
struct StatsCore {
    enqueued: AtomicU64,
    completed: AtomicU64,
    active: AtomicU64,
    max_active: AtomicU64,
    max_queue_depth: AtomicU64,
    tagged: [AtomicU64; DbReadKind::ALL.len()],
}

impl StatsCore {
    fn snapshot(&self) -> DbReadStats {
        let tagged = std::array::from_fn(|i| self.tagged[i].load(Ordering::Relaxed));
        DbReadStats {
            workers: 0,
            enqueued: self.enqueued.load(Ordering::Relaxed),
            completed: self.completed.load(Ordering::Relaxed),
            max_active: self.max_active.load(Ordering::Relaxed),
            max_queue_depth: self.max_queue_depth.load(Ordering::Relaxed),
            tagged,
        }
    }
}

/// The bounded read pool of one store. Cheap to build: no thread exists
/// until the first async submit needs one. Dropping the last handle closes
/// the service; workers drain every queued read and then exit.
pub struct DbReadService {
    shared: Arc<ReadShared>,
}

impl std::fmt::Debug for DbReadService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DbReadService")
            .field("workers", &self.stats().workers)
            .field("capacity", &self.shared.cfg.capacity)
            .finish_non_exhaustive()
    }
}

impl DbReadService {
    /// Create the service for `store` (lazy: no thread is spawned until the
    /// first async submit; outstanding reads are bounded by `cfg.capacity`
    /// and the pool by `cfg.workers`, hard-capped).
    pub fn spawn(store: Arc<Store>, cfg: DbReadServiceConfig) -> Self {
        let cfg = cfg.clamped();
        let capacity = cfg.capacity;
        Self {
            shared: Arc::new(ReadShared {
                store,
                cfg,
                semaphore: Arc::new(Semaphore::new(capacity)),
                queue: Mutex::new(VecDeque::new()),
                wake: Condvar::new(),
                closed: AtomicBool::new(false),
                running: AtomicU64::new(0),
                exit: ExitFlag::default(),
                stats: StatsCore::default(),
                start_lock: Mutex::new(()),
            }),
        }
    }

    /// Instrumentation snapshot (see [`DbReadStats`]).
    pub fn stats(&self) -> DbReadStats {
        let mut s = self.shared.stats.snapshot();
        s.workers = self.shared.cfg.workers;
        s
    }

    /// The shared store reads execute against.
    pub fn store(&self) -> Arc<Store> {
        self.shared.store.clone()
    }

    /// Execute ONE store read on the bounded pool and await its result.
    ///
    /// `op` runs on a pool thread — the same sync read a caller used to
    /// run inline on its tokio worker. Backpressure: when `capacity`
    /// reads are already queued or executing, the caller awaits a permit
    /// (never unbounded buffering). The read's result is delivered
    /// verbatim; a panicking read or a closed service surfaces as a typed
    /// error, never a hang and never a silent inline fallback.
    pub async fn submit<T: Send + 'static>(
        &self,
        op: impl FnOnce(&Store) -> T + Send + 'static,
    ) -> faktor_core::Result<T> {
        self.submit_tagged(DbReadKind::Other, op).await
    }

    /// [`DbReadService::submit`] with the read's capability tag: the
    /// manager's read wrappers submit under [`DbReadKind::History`],
    /// `Task`, `Budget`, `Prefix`, `Verification` and `Memory` so the
    /// runtime tripwire can prove a full turn's reads were served by the
    /// pool with per-capability counts. Tagging never changes execution,
    /// ordering or bounds.
    pub async fn submit_tagged<T: Send + 'static>(
        &self,
        kind: DbReadKind,
        op: impl FnOnce(&Store) -> T + Send + 'static,
    ) -> faktor_core::Result<T> {
        self.start()?;
        // Backpressure: at most `capacity` reads are queued or executing.
        let permit = Arc::clone(&self.shared.semaphore)
            .acquire_owned()
            .await
            .map_err(|_| Error::internal("db read service closed while awaiting capacity"))?;
        let (reply_tx, reply_rx) = oneshot::channel();
        let job = Job {
            reply: reply_tx,
            run: Box::new(move |store| -> Box<dyn Any + Send + 'static> { Box::new(op(store)) }),
            _permit: permit,
        };
        {
            let mut queue = self.shared.queue.lock().unwrap_or_else(|p| p.into_inner());
            if self.shared.closed.load(Ordering::Relaxed) {
                // Closed between the permit and the enqueue: release and
                // fail fast (a drained pool never accepts new reads).
                drop(queue);
                return Err(Error::internal(
                    "db read service is closed (shutdown); reads fail fast",
                ));
            }
            queue.push_back(job);
            self.shared
                .stats
                .max_queue_depth
                .fetch_max(queue.len() as u64, Ordering::Relaxed);
            self.shared.stats.enqueued.fetch_add(1, Ordering::Relaxed);
            if let Some(counter) = self.shared.stats.tagged.get(kind.index()) {
                counter.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.shared.wake.notify_one();
        let reply = reply_rx
            .await
            .map_err(|_| Error::internal("db read worker died before replying"))?;
        match reply {
            Ok(boxed) => boxed
                .downcast::<T>()
                .map(|b| *b)
                .map_err(|_| Error::internal("db read service type mismatch on a reply")),
            Err(message) => Err(Error::internal(message)),
        }
    }

    /// Lazy pool start: spawn the worker threads exactly once, on the
    /// first submit. Returns an error when the pool is closed or could not
    /// start.
    fn start(&self) -> faktor_core::Result<()> {
        if self.shared.closed.load(Ordering::Relaxed) {
            return Err(Error::internal(
                "db read service is closed (shutdown); reads fail fast",
            ));
        }
        let _lock = self
            .shared
            .start_lock
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        if self.shared.closed.load(Ordering::Relaxed) {
            return Err(Error::internal(
                "db read service is closed (shutdown); reads fail fast",
            ));
        }
        if self.shared.running.load(Ordering::Relaxed) > 0 {
            return Ok(());
        }
        let workers = self.shared.cfg.workers;
        let mut spawned = 0usize;
        for i in 0..workers {
            let shared = self.shared.clone();
            let name = format!("{READ_THREAD_NAME}-{i}");
            if std::thread::Builder::new()
                .name(name)
                .spawn(move || worker_main(shared))
                .is_ok()
            {
                spawned += 1;
            }
        }
        if spawned == 0 {
            return Err(Error::internal(
                "db read service could not start its worker pool",
            ));
        }
        self.shared.running.store(spawned as u64, Ordering::SeqCst);
        Ok(())
    }

    /// Close the queue and wait (bounded by `timeout`) until every worker
    /// has drained every queued read, replied, and exited. Returns whether
    /// the pool actually drained and exited. Idempotent; `true` when no
    /// pool was ever started. Reads submitted after a successful shutdown
    /// fail fast.
    pub async fn shutdown(&self, timeout: Duration) -> bool {
        self.close();
        let shared = self.shared.clone();
        tokio::task::spawn_blocking(move || {
            let mut state = shared.exit.state.lock().unwrap_or_else(|p| p.into_inner());
            if *state || shared.running.load(Ordering::Relaxed) == 0 {
                return true;
            }
            let deadline = Instant::now() + timeout;
            while !*state {
                let now = Instant::now();
                if now >= deadline {
                    return false;
                }
                let (guard, _) = shared
                    .exit
                    .cv
                    .wait_timeout(state, deadline - now)
                    .unwrap_or_else(|p| p.into_inner());
                state = guard;
            }
            true
        })
        .await
        .unwrap_or(false)
    }

    /// Whether the service is closed (shutdown or last-handle drop).
    pub fn is_closed(&self) -> bool {
        self.shared.closed.load(Ordering::Relaxed)
    }

    /// Mark the service closed and wake the workers: every queued read
    /// drains, then the threads exit on their own.
    fn close(&self) {
        self.shared.closed.store(true, Ordering::SeqCst);
        self.shared.wake.notify_all();
    }
}

impl Drop for DbReadService {
    fn drop(&mut self) {
        // Break the worker->shared cycle: closing the service lets the
        // workers drain and exit, releasing their Arc<ReadShared>.
        self.close();
    }
}

/// Worker thread main: pull reads in FIFO order and execute them on the
/// shared store. Exits (after draining) once the service is closed.
fn worker_main(shared: Arc<ReadShared>) {
    let delay = shared.cfg.pre_read_delay;
    loop {
        let job = {
            let mut queue = shared.queue.lock().unwrap_or_else(|p| p.into_inner());
            loop {
                if let Some(job) = queue.pop_front() {
                    break job;
                }
                if shared.closed.load(Ordering::Relaxed) {
                    if shared.running.fetch_sub(1, Ordering::SeqCst) <= 1 {
                        shared.exit.notify_exited();
                    }
                    return;
                }
                queue = shared.wake.wait(queue).unwrap_or_else(|p| p.into_inner());
            }
        };
        let Job {
            reply,
            run,
            _permit,
        } = job;
        let active = shared.stats.active.fetch_add(1, Ordering::Relaxed) + 1;
        shared.stats.max_active.fetch_max(active, Ordering::Relaxed);
        if let Some(delay) = delay {
            std::thread::sleep(delay);
        }
        // Panic isolation: a hostile store read must error its caller,
        // never hang it and never kill the pool.
        let outcome = catch_unwind(AssertUnwindSafe(|| run(&shared.store)));
        let _ = reply.send(match outcome {
            Ok(value) => Ok(value),
            Err(_) => Err("db read worker panicked inside a store read".to_string()),
        });
        shared.stats.active.fetch_sub(1, Ordering::Relaxed);
        shared.stats.completed.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::{BudgetAuthority as _, DurableBudgetLedger};
    use crate::handle::tests::{session, test_manager};
    use crate::task::Task;
    use faktor_core::id::{OpId, TaskId};
    use faktor_core::state::{TaskState, VerificationStatus};
    use faktor_store::Store;

    /// One deterministic burst helper: `count` concurrent submits of a tiny
    /// store read (the shared-store `message_count` probe is a real SQLite
    /// read, decoded on the worker).
    async fn burst(service: &Arc<DbReadService>, count: usize) -> Vec<faktor_core::Result<i64>> {
        let mut tasks = Vec::with_capacity(count);
        for _ in 0..count {
            let svc = Arc::clone(service);
            tasks.push(tokio::spawn(async move {
                svc.submit(move |store| store.message_count(faktor_core::id::SessionId::new(1)))
                    .await
            }));
        }
        let mut out = Vec::with_capacity(count);
        for t in tasks {
            match t.await.expect("burst task panicked") {
                Ok(Ok(count)) => out.push(Ok(count)),
                Ok(Err(e)) => out.push(Err(crate::map_store_err(e).into())),
                Err(e) => out.push(Err(e)),
            }
        }
        out
    }

    /// MessageRow has no PartialEq; compare the durable projection fields
    /// the wrapper parity actually promises.
    fn rows_key(rows: &[faktor_store::MessageRow]) -> Vec<(i64, i64, String, serde_json::Value)> {
        rows.iter()
            .map(|r| (r.id, r.seq, r.role.clone(), r.data.clone()))
            .collect()
    }

    #[tokio::test]
    async fn async_read_wrappers_parity_with_the_sync_reads() {
        // (a) correctness parity: every async SessionManager wrapper returns
        // results identical to the synchronous read it replaces. The sync
        // twins are the exact calls the turn machinery ran inline.
        let (_d, m) = test_manager();
        let s = session(&m);
        let sid = s.id();

        // --- messages_backwards_bounded --------------------------------------------------
        for seq in 1..=7i64 {
            s.put_message(
                seq,
                "user",
                serde_json::json!({ "text": format!("m{seq}") }),
            )
            .unwrap();
        }
        let sync = s.messages_backwards_bounded(None, 4, u64::MAX).unwrap();
        let async_rows = m
            .messages_backwards_bounded(sid, None, 4, u64::MAX)
            .await
            .unwrap();
        assert_eq!(
            rows_key(&async_rows),
            rows_key(&sync),
            "windowed history parity"
        );
        let sync = s.messages_backwards_bounded(Some(6), 10, 64).unwrap();
        let async_rows = m
            .messages_backwards_bounded(sid, Some(6), 10, 64)
            .await
            .unwrap();
        assert_eq!(
            rows_key(&async_rows),
            rows_key(&sync),
            "byte-bound + cursor parity"
        );
        let sync = s.messages_backwards_bounded(None, 0, u64::MAX).unwrap();
        let async_rows = m
            .messages_backwards_bounded(sid, None, 0, u64::MAX)
            .await
            .unwrap();
        assert_eq!(rows_key(&async_rows), rows_key(&sync), "zero-cap parity");
        assert!(async_rows.is_empty());

        // --- task -------------------------------------------------------------------------
        let tid = s.task_id().unwrap();
        let task = Task {
            task_id: tid,
            session_id: sid,
            goal: "parity goal".into(),
            acceptance_criteria: vec![],
            plan: vec![],
            budget: crate::TaskBudget::default(),
            state: TaskState::Pending,
            created_ms: 1,
            updated_ms: 1,
        };
        let created = s.create_task(task).unwrap();
        assert_eq!(s.get_task(tid).unwrap().as_ref(), Some(&created));
        assert_eq!(
            m.task(sid, tid).await.unwrap().as_ref(),
            Some(&created),
            "task-row parity"
        );
        assert_eq!(
            m.task(sid, TaskId::new(9999)).await.unwrap(),
            s.get_task(TaskId::new(9999)).unwrap(),
            "missing task parity (None both sides)"
        );

        // --- budget_view ------------------------------------------------------------------
        let ledger = DurableBudgetLedger::new(m.clone());
        // No money yet: unlimited-zero view, identical on both surfaces.
        assert_eq!(
            m.budget_view(sid, tid).await,
            ledger.session_budget_view(sid, tid)
        );
        ledger.set_task_max_cost(sid, tid, Some(1_000_000)).unwrap();
        let reserved = ledger
            .reserve(sid, tid, OpId::new(31), 500_000, None)
            .await
            .unwrap();
        assert_eq!(
            m.budget_view(sid, tid).await,
            ledger.session_budget_view(sid, tid)
        );
        let view = m.budget_view(sid, tid).await.unwrap();
        assert_eq!(view.max_cost_micro, Some(1_000_000));
        assert_eq!(view.open_reserved_micro, 500_000);
        assert_eq!(view.open_reservations, 1);
        ledger
            .settle_usage(sid, reserved, 10, 0, 0, 20, Some(12), None)
            .await
            .unwrap();
        assert_eq!(
            m.budget_view(sid, tid).await,
            ledger.session_budget_view(sid, tid),
            "budget parity after settle"
        );
        assert_eq!(m.budget_view(sid, tid).await.unwrap().settled_count, 1);

        // --- verification_records ----------------------------------------------------------
        assert_eq!(
            m.verification_records(sid, tid).await.unwrap(),
            s.list_verification_records(tid).unwrap(),
            "empty record list parity"
        );
        s.create_verification_record(
            tid,
            None,
            vec![],
            vec![],
            vec![],
            vec![],
            None,
            VerificationStatus::Passed,
            1,
        )
        .unwrap();
        assert_eq!(
            m.verification_records(sid, tid).await.unwrap(),
            s.list_verification_records(tid).unwrap(),
            "record list parity"
        );
        assert_eq!(
            m.verification_records(sid, TaskId::new(77)).await.unwrap(),
            s.list_verification_records(TaskId::new(77)).unwrap(),
            "foreign-task record list parity (both empty)"
        );

        // --- provider_prefix_history -------------------------------------------------------
        let h1 = [7u8; 32];
        s.settle_usage_with_prefix(
            OpId::new(41),
            "p",
            "m",
            "completed",
            Some(1),
            Some(1),
            None,
            Some(h1),
            Some(10),
        )
        .unwrap();
        let h2 = [9u8; 32];
        s.settle_usage_with_prefix(
            OpId::new(42),
            "p",
            "m",
            "completed",
            Some(2),
            Some(1),
            None,
            Some(h2),
            Some(20),
        )
        .unwrap();
        assert_eq!(
            m.provider_prefix_history(sid).await.unwrap(),
            m.store().provider_call_prefix_rows(sid).unwrap(),
            "prefix-history parity (oldest call first)"
        );

        // --- memory_page -------------------------------------------------------------------
        for i in 0..7i64 {
            s.upsert_memory_fact("pref", &format!("k{i}"), &format!("v{i}"))
                .unwrap();
        }
        let sync_page = s.memory_facts_page(None, 3).unwrap();
        let async_page = m.memory_page(sid, None, 3).await.unwrap();
        assert_eq!(async_page, sync_page, "memory page parity (first page)");
        let cursor = sync_page.cursor;
        let sync_page2 = s.memory_facts_page(cursor.as_ref(), 3).unwrap();
        let async_page2 = m.memory_page(sid, cursor.clone(), 3).await.unwrap();
        assert_eq!(async_page2, sync_page2, "memory page parity (cursor page)");
        assert_eq!(async_page2.total_estimate, sync_page2.total_estimate);
        let sync_last = s.memory_facts_page(async_page2.cursor.as_ref(), 3).unwrap();
        let async_last = m
            .memory_page(sid, async_page2.cursor.clone(), 3)
            .await
            .unwrap();
        assert_eq!(async_last, sync_last);
        assert!(!async_last.has_more, "final page parity");
    }

    #[tokio::test]
    async fn two_hundred_concurrent_reads_never_exceed_the_worker_cap() {
        // (b) bounded concurrency: 200 concurrent async reads complete and
        // the instrumented concurrent-execution high-water never exceeds
        // the configured worker count (2). The delay seam forces overlap.
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(Store::open(dir.path().join("store"), false).unwrap());
        let service = Arc::new(DbReadService::spawn(
            store,
            DbReadServiceConfig {
                workers: 2,
                capacity: 32,
                pre_read_delay: Some(Duration::from_millis(2)),
            },
        ));
        let results = tokio::time::timeout(Duration::from_secs(60), burst(&service, 200))
            .await
            .expect("200 concurrent reads must complete");
        assert_eq!(results.len(), 200);
        assert!(
            results.iter().all(|r| r.is_ok()),
            "every concurrent read must succeed"
        );
        let stats = service.stats();
        assert_eq!(stats.enqueued, 200);
        assert_eq!(stats.completed, 200, "none lost");
        assert!(
            stats.max_active <= service.shared.cfg.workers as u64,
            "concurrent executions must never exceed the worker cap: {stats:?}"
        );
        assert_eq!(
            stats.max_active, 2,
            "the delay seam must force real overlap: {stats:?}"
        );
    }

    #[tokio::test]
    async fn overload_burst_backpressures_within_the_bounded_queue_and_completes() {
        // (c) overload: a burst far beyond capacity backpressures (the
        // shared queue never exceeds its bound — no OOM), loses nothing
        // and completes.
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(Store::open(dir.path().join("store"), false).unwrap());
        let service = Arc::new(DbReadService::spawn(
            store,
            DbReadServiceConfig {
                workers: 2,
                capacity: 8,
                pre_read_delay: Some(Duration::from_millis(4)),
            },
        ));
        let started = Instant::now();
        let results = tokio::time::timeout(Duration::from_secs(60), burst(&service, 200))
            .await
            .expect("an overloaded pool must still complete");
        assert_eq!(results.len(), 200);
        assert!(
            results.iter().all(|r| r.is_ok()),
            "every overloaded read must succeed"
        );
        let stats = service.stats();
        assert_eq!(stats.enqueued, 200);
        assert_eq!(stats.completed, 200, "no read lost under overload");
        assert!(
            stats.max_queue_depth <= 8,
            "the bounded queue must never exceed its capacity: {stats:?}"
        );
        assert!(
            stats.max_active <= 2,
            "execution stays within the worker pool: {stats:?}"
        );
        // Backpressure observable: 200 reads of 4 ms each on 2 workers need
        // at least ~400 ms of worker time; finishing far sooner would mean
        // the burst buffered without bound instead of awaiting capacity.
        assert!(
            started.elapsed() >= Duration::from_millis(300),
            "the burst must have waited for capacity, not buffered without bound"
        );
    }

    #[tokio::test]
    async fn shutdown_drains_pending_reads_and_later_submits_fail_fast() {
        // (e) shutdown: every queued read drains (replies, never lost), the
        // workers exit, and reads submitted afterwards fail fast — a closed
        // pool never accepts new work and never hangs.
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(Store::open(dir.path().join("store"), false).unwrap());
        let service = Arc::new(DbReadService::spawn(
            store,
            DbReadServiceConfig {
                workers: 2,
                capacity: 64,
                pre_read_delay: Some(Duration::from_millis(2)),
            },
        ));
        let mut pending = Vec::new();
        for _ in 0..40 {
            let svc = Arc::clone(&service);
            pending.push(tokio::spawn(async move {
                svc.submit(|store| store.message_count(faktor_core::id::SessionId::new(1)))
                    .await
            }));
        }
        // Every read must be QUEUED (in flight or waiting on a worker)
        // before the shutdown: poll the enqueue counter, bounded.
        let deadline = Instant::now() + Duration::from_secs(10);
        while service.stats().enqueued < 40 {
            assert!(
                Instant::now() < deadline,
                "all 40 reads must enqueue before shutdown"
            );
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        assert!(
            service.shutdown(Duration::from_secs(30)).await,
            "the pool must drain every pending read and exit"
        );
        for p in pending {
            p.await
                .expect("pending reader panicked")
                .expect("drained read must succeed")
                .expect("drained store read must succeed");
        }
        let stats = service.stats();
        assert_eq!(stats.enqueued, 40);
        assert_eq!(stats.completed, 40, "shutdown drained every pending read");
        assert!(service.is_closed());
        let late = service
            .submit(|store| store.message_count(faktor_core::id::SessionId::new(1)))
            .await;
        assert!(late.is_err(), "a closed service fails reads fast: {late:?}");
    }

    #[tokio::test]
    async fn panicking_read_errors_its_caller_and_the_pool_survives() {
        // Adversarial: a hostile read (panic inside the closure) must reply
        // a typed error — never hang, never kill the pool; later reads
        // keep succeeding on the same workers.
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(Store::open(dir.path().join("store"), false).unwrap());
        let service = Arc::new(DbReadService::spawn(store, DbReadServiceConfig::default()));
        let poisoned = service
            .submit(|_store| -> i64 { panic!("hostile read") })
            .await;
        assert!(poisoned.is_err());
        let message = poisoned.unwrap_err().message;
        assert!(message.contains("panicked"), "{message}");
        let healthy = service
            .submit(|store| store.message_count(faktor_core::id::SessionId::new(1)))
            .await;
        assert_eq!(
            healthy.unwrap().unwrap(),
            0,
            "the pool survives a panicking read"
        );
        let stats = service.stats();
        assert_eq!(stats.completed, 2);
        assert!(stats.max_active <= service.shared.cfg.workers as u64);
    }

    #[test]
    fn worker_config_is_hard_capped_to_four() {
        // The audit's pool bound: requests above MAX_READ_WORKERS are
        // clamped, never honored.
        let config = DbReadServiceConfig {
            workers: 64,
            capacity: 2,
            ..Default::default()
        }
        .clamped();
        assert_eq!(config.workers, MAX_READ_WORKERS);
        let config = DbReadServiceConfig {
            workers: 1,
            capacity: 2,
            ..Default::default()
        }
        .clamped();
        assert_eq!(config.workers, 2, "the floor is the audit's 2 workers");
    }

    #[tokio::test]
    async fn tagged_submits_land_in_their_capability_counters() {
        // Audit gate 13: the stats are tagged by capability so the runtime
        // tripwire can assert a full turn's history/task/budget/prefix/
        // verification/memory reads all ran through this pool. The generic
        // `submit` seam stays available and lands under `Other`.
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(Store::open(dir.path().join("store"), false).unwrap());
        let service = DbReadService::spawn(store, DbReadServiceConfig::default());
        for kind in DbReadKind::ALL {
            service
                .submit_tagged(kind, |store| {
                    store.message_count(faktor_core::id::SessionId::new(1))
                })
                .await
                .unwrap()
                .unwrap();
        }
        // The generic seam is the untagged compatibility path.
        service
            .submit(|store| store.message_count(faktor_core::id::SessionId::new(1)))
            .await
            .unwrap()
            .unwrap();
        let stats = service.stats();
        assert_eq!(stats.enqueued, 8);
        for kind in DbReadKind::ALL {
            let expected = if kind == DbReadKind::Other { 2 } else { 1 };
            assert_eq!(
                stats.kind(kind),
                expected,
                "{} must be tagged exactly once: {stats:?}",
                kind.label()
            );
            assert_eq!(stats.kind_label(kind.label()), expected);
        }
        // A forged label never fabricates a count.
        assert_eq!(stats.kind_label("forged"), 0);
        let sum: u64 = DbReadKind::ALL.iter().map(|k| stats.kind(*k)).sum();
        assert_eq!(sum, stats.enqueued, "every submitted read is tagged");
    }
}
