//! `DbActor`: the async SQLite append actor (audit 42).
//!
//! Tokio workers used to run multi-ms synchronous SQLite commits inline on
//! the message/part/journal/usage hot paths. This module moves those writes
//! off every worker:
//!
//! - A dedicated **`std::thread`** (never a tokio task) executes the store
//!   work. It runs only blocking std primitives (`std::sync::mpsc` +
//!   `recv_timeout`) so it needs no runtime.
//! - Callers (async code on tokio workers) enqueue into a **bounded tokio
//!   channel** (`DbActorConfig::capacity`, default 1024). A full channel
//!   makes senders await — genuine backpressure, never unbounded growth.
//! - A small **bridge task** (spawned lazily on the caller's runtime)
//!   coalesces envelopes into batches (at most `max_batch` writes, or those
//!   that arrived within `flush_tick`) and hands each batch to the actor
//!   thread over a *rendezvous* std channel (capacity 0): a batch is always
//!   either in the bounded tokio stage, in the bridge's hand, or in the
//!   actor's current batch — never parked in an unbounded queue and never
//!   orphaned by an actor death.
//! - The actor executes each batch as ONE store transaction with ONE commit
//!   fsync ([`Store::batch_hot_writes`]) and then replies to every caller.
//!   A reply therefore means "durable".
//! - **Checkpointing is maintenance, never interactive work**: the store
//!   disables `wal_autocheckpoint`, so SQLite's automatic checkpoint — which
//!   used to execute inside the committing statement and could exceed 5 ms
//!   once the WAL grew past its watermark — never runs on the batch path.
//!   The actor instead runs a bounded PASSIVE checkpoint
//!   ([`Store::wal_checkpoint_passive`]) on its idle flush tick, after the
//!   queue drained. Maintenance segments are typed separately and excluded
//!   BY TYPE from the 5 ms interactive gate: no tolerance is applied, a slow
//!   checkpoint cannot be mistaken for hot-path SQL work, and a fast one
//!   cannot dilute the gate.
//! - **Instrumentation, not inference**: every synchronous store segment is
//!   timed inside the actor (`worker_blocked_over_5ms` counts interactive
//!   segments over 5 ms; maintenance is reported separately) and every
//!   caller-side send→reply wait is sampled (`p95_wait_us`, `max_wait_us`).
//!   See [`DbActorStats`].
//!
//! Failure handling:
//! - When the actor thread dies (test seam, or a panic while the batch is in
//!   hand) every in-flight caller receives an error — never a hang — and the
//!   bridge spawns a replacement actor thread over the SAME shared [`Store`]
//!   (the last durable position is re-derived from the store itself by the
//!   replacement's transactions).
//! - A panic *inside* the store segment poisons the store writer lock, which
//!   is fatal for every writer (actor or direct): the actor marks itself
//!   permanently failed and all later calls fail fast instead of spawning a
//!   doomed replacement.
//!
//! The async surface mirrors the store's hot call surface ONLY (see
//! [`StoreHandle`]): message append, part append, journal event, and usage
//! settlement. Every other store call stays direct and synchronous on the
//! shared [`Store`]; see [`Store::direct`] in `faktor-store`.

use std::collections::VecDeque;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use faktor_core::event::EventKind;
use faktor_core::id::{EventSeq, OpId, SessionId};
use faktor_core::state::AgentState;
use faktor_store::{HotWrite, HotWriteOutcome, Store, StoreError, StoreResult};
use tokio::sync::{mpsc, oneshot};

/// Bounded wait-sample ring: caller-side queue waits are never accumulated
/// without limit (bounded everything).
const WAIT_RING_CAP: usize = 8192;

/// Thread name of the actor thread (observable in crash forensics).
const ACTOR_THREAD_NAME: &str = "faktor-db-actor";

/// One queued write and its reply channel.
struct Envelope {
    op: ActorOp,
    reply: oneshot::Sender<StoreResult<ActorOutcome>>,
}

/// The four hot write shapes the actor executes (message append / part
/// append / journal event / usage settlement).
#[derive(Debug, Clone)]
pub(crate) enum ActorOp {
    AppendEvent {
        session_id: SessionId,
        op_id: Option<OpId>,
        kind: EventKind,
        state: AgentState,
        ts_ms: i64,
        payload: Option<serde_json::Value>,
    },
    PutMessage {
        session_id: SessionId,
        seq: i64,
        role: String,
        data: serde_json::Value,
    },
    PutPart {
        message_id: i64,
        kind: String,
        data: serde_json::Value,
    },
    SettleUsage {
        session_id: SessionId,
        op_id: OpId,
        provider: String,
        model: String,
        status: String,
        tokens_in: Option<u64>,
        tokens_out: Option<u64>,
        error: Option<String>,
    },
}

impl ActorOp {
    fn to_hot_write(&self) -> HotWrite {
        match self {
            ActorOp::AppendEvent {
                session_id,
                op_id,
                kind,
                state,
                ts_ms,
                payload,
            } => HotWrite::AppendEvent {
                session_id: *session_id,
                op_id: *op_id,
                kind: *kind,
                state: *state,
                ts_ms: *ts_ms,
                payload: payload.clone(),
                // Every append stamps the writer's current payload schema
                // version (audits 71-72): readers refuse unknown versions.
                payload_ver: crate::payload::PAYLOAD_SCHEMA_V,
            },
            ActorOp::PutMessage {
                session_id,
                seq,
                role,
                data,
            } => HotWrite::PutMessage {
                session_id: *session_id,
                seq: *seq,
                role: role.clone(),
                data: data.clone(),
            },
            ActorOp::PutPart {
                message_id,
                kind,
                data,
            } => HotWrite::PutPart {
                message_id: *message_id,
                kind: kind.clone(),
                data: data.clone(),
            },
            ActorOp::SettleUsage {
                session_id,
                op_id,
                provider,
                model,
                status,
                tokens_in,
                tokens_out,
                error,
            } => HotWrite::RecordProviderCall {
                session_id: *session_id,
                op_id: *op_id,
                provider: provider.clone(),
                model: model.clone(),
                status: status.clone(),
                tokens_in: *tokens_in,
                tokens_out: *tokens_out,
                error: error.clone(),
            },
        }
    }
}

/// Per-op result of an actor call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ActorOutcome {
    /// The journal event's gapless per-session sequence.
    EventSeq(EventSeq),
    /// The inserted row id (message / part / provider-call).
    RowId(i64),
}

/// Tuning knobs of a [`DbActor`].
#[derive(Debug, Clone)]
pub struct DbActorConfig {
    /// Bounded caller-side queue capacity (default 1024). Senders await when
    /// the queue is full; it never grows beyond this.
    pub capacity: usize,
    /// At most this many queued writes execute per transaction (default 32).
    pub max_batch: usize,
    /// Idle flush tick: a partial batch waits at most this long after its
    /// first write arrives before its transaction runs (default 2 ms).
    pub flush_tick: Duration,
    /// Test seam: sleep this long before every batch to make backpressure
    /// deterministic. `None` in production.
    #[doc(hidden)]
    pub pre_batch_delay: Option<Duration>,
    /// Test seam: panic the actor thread (after replying an error to its
    /// in-flight callers) when the executed-batch counter reaches this
    /// value. `None` (default) disables the seam.
    #[doc(hidden)]
    pub panic_after_batches: Option<u64>,
    /// Test seam: sleep this long inside each maintenance segment before
    /// checkpointing — simulates a checkpoint-heavy store (a huge WAL /
    /// slow disk) deterministically. `None` in production.
    #[doc(hidden)]
    pub maintenance_delay: Option<Duration>,
    /// Test seam: force maintenance checkpoints to fail with this message
    /// (typed error path) instead of touching the store. `None` in
    /// production.
    #[doc(hidden)]
    pub maintenance_fail_for_test: Option<String>,
}

impl Default for DbActorConfig {
    fn default() -> Self {
        Self {
            capacity: 1024,
            max_batch: 32,
            flush_tick: Duration::from_millis(2),
            pre_batch_delay: None,
            panic_after_batches: None,
            maintenance_delay: None,
            maintenance_fail_for_test: None,
        }
    }
}

/// Instrumentation snapshot of a [`DbActor`] (audit gate 42: instrumented,
/// never inferred).
#[derive(Debug, Clone, Default)]
pub struct DbActorStats {
    /// Writes enqueued by callers (successful tokio sends).
    pub enqueued: u64,
    /// Writes the actor replied to (Ok or Err). A drained actor has
    /// `completed == enqueued`.
    pub completed: u64,
    /// Store transactions executed (batches).
    pub batches: u64,
    /// Caller-side queue wait: send→reply p95 in microseconds.
    pub p95_wait_us: u64,
    /// Caller-side queue wait: send→reply maximum in microseconds.
    pub max_wait_us: u64,
    /// Longest INTERACTIVE synchronous store segment inside the actor,
    /// microseconds (SQL work + commit fsync). Maintenance is a separate
    /// type and never lands here.
    pub max_block_us: u64,
    /// Audit gate: count of INTERACTIVE synchronous store segments over
    /// 5 ms. Maintenance segments cannot increment this by construction.
    pub worker_blocked_over_5ms: u64,
    /// Maintenance WAL checkpoints completed (idle flush tick).
    pub maintenance_checkpoints: u64,
    /// Maintenance WAL checkpoints that failed. Failures are logged
    /// (`tracing::warn!`) and never fatal: the next batch still runs.
    pub maintenance_errors: u64,
    /// Longest maintenance segment, microseconds. Measured like the
    /// interactive segment but tracked in its OWN counter: the 5 ms gate
    /// excludes it by type, not by tolerance.
    pub max_maintenance_us: u64,
    /// Text of the last maintenance failure, when one occurred.
    pub last_maintenance_error: Option<String>,
    /// High-water of the bounded caller queue (occupancy + in-bridge batch;
    /// never exceeds `DbActorConfig::capacity`).
    pub max_queue_depth: u64,
}

/// Shared state between the bridge, the actor thread and the handles.
struct ActorShared {
    store: Arc<Store>,
    cfg: Mutex<DbActorConfig>,
    stats: StatsCore,
    /// The bounded caller-side channel sender (lazily created on first use).
    tx: Mutex<Option<mpsc::Sender<Envelope>>>,
    /// The bridge task handle (lazily spawned on first use).
    bridge: Mutex<Option<tokio::task::JoinHandle<()>>>,
    /// Set when a store-segment panic poisoned the writer lock: fatal.
    fatal: AtomicBool,
    /// Panic seam (batch target), cleared on actor respawn.
    panic_after_batches: AtomicU64,
    /// Executed batches so far (monotonic across respawns).
    batches_done: AtomicU64,
    /// `batches_done` observed at the last maintenance attempt: a FAILED
    /// attempt is retried only after new batch work arrived (never at
    /// flush-tick frequency, so a persistent error cannot spam the log).
    maintenance_batches: AtomicU64,
    /// Whether the last maintenance attempt failed.
    maintenance_failed: AtomicBool,
    /// Actor thread exit signal (drained / died), waitable from async tests.
    exit: ExitSignal,
}

/// Condvar-backed actor-exit signal.
#[derive(Default)]
struct ExitSignal {
    state: Mutex<bool>,
    cv: Condvar,
}

impl ExitSignal {
    fn notify_exited(&self) {
        *self.state.lock().unwrap_or_else(|p| p.into_inner()) = true;
        self.cv.notify_all();
    }
}

/// Atomic counters + bounded wait-sample ring.
///
/// The ring is a `VecDeque` (P0-44): the old `Vec` + `remove(0)` shifted
/// every element on every eviction — O(n) per sample, O(n²) over a full
/// ring — so sustained sampling (e.g. a burst of 10k appends filling the
/// 8192-entry ring) degenerated quadratically. `VecDeque::pop_front` +
/// `push_back` keeps every sample O(1); p95/max semantics are unchanged
/// (the ring still holds the most recent `WAIT_RING_CAP` samples).
#[derive(Default)]
struct StatsCore {
    enqueued: AtomicU64,
    completed: AtomicU64,
    batches: AtomicU64,
    queue_depth_high: AtomicU64,
    max_block_us: AtomicU64,
    worker_blocked_over_5ms: AtomicU64,
    maintenance_checkpoints: AtomicU64,
    maintenance_errors: AtomicU64,
    max_maintenance_us: AtomicU64,
    last_maintenance_error: Mutex<Option<String>>,
    waits: Mutex<VecDeque<u32>>,
    max_wait_us: AtomicU64,
}

/// One timed segment of the actor thread, typed so the 5 ms gate applies to
/// exactly one kind of work: `Interactive` hot-path SQL. Maintenance is a
/// DIFFERENT variant, so a slow WAL checkpoint can never be counted as
/// interactive blockage (and never diluted into the interactive maximum) —
/// the exclusion is structural, not a tolerance.
enum Segment {
    /// Interactive store work for one batch: SQLite statements up to the
    /// commit, timed as consumed thread CPU.
    Interactive { work: Duration, total: Duration },
    /// Idle-tick WAL checkpoint maintenance (or its test-slowed equivalent).
    Maintenance {
        elapsed: Duration,
        error: Option<String>,
    },
}

impl StatsCore {
    fn record_wait(&self, wait: Duration) {
        let us = wait.as_micros().min(u32::MAX as u128) as u32;
        self.max_wait_us.fetch_max(us as u64, Ordering::Relaxed);
        let mut ring = self.waits.lock().unwrap_or_else(|p| p.into_inner());
        if ring.len() == WAIT_RING_CAP {
            ring.pop_front();
        }
        ring.push_back(us);
    }

    /// Observe the bounded-queue occupancy (bridge side).
    fn observe_queue_depth(&self, depth: u64) {
        self.queue_depth_high.fetch_max(depth, Ordering::Relaxed);
    }

    fn record_segment(&self, segment: Segment) {
        match segment {
            Segment::Interactive { work, total } => {
                // max_block_us reports the FULL interactive synchronous
                // segment (SQL work + commit fsync); the >5 ms gate counts
                // the SQL WORK portion only.
                let total_us = total.as_micros().min(u64::MAX as u128) as u64;
                self.max_block_us.fetch_max(total_us, Ordering::Relaxed);
                let work_us = work.as_micros().min(u64::MAX as u128) as u64;
                if work_us > 5_000 {
                    self.worker_blocked_over_5ms.fetch_add(1, Ordering::Relaxed);
                }
            }
            Segment::Maintenance { elapsed, error } => {
                let us = elapsed.as_micros().min(u64::MAX as u128) as u64;
                self.max_maintenance_us.fetch_max(us, Ordering::Relaxed);
                match error {
                    None => {
                        self.maintenance_checkpoints.fetch_add(1, Ordering::Relaxed);
                    }
                    Some(message) => {
                        self.maintenance_errors.fetch_add(1, Ordering::Relaxed);
                        *self
                            .last_maintenance_error
                            .lock()
                            .unwrap_or_else(|p| p.into_inner()) = Some(message);
                    }
                }
            }
        }
    }

    fn snapshot(&self) -> DbActorStats {
        let mut ring: Vec<u32> = self
            .waits
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .copied()
            .collect();
        ring.sort_unstable();
        let p95 = if ring.is_empty() {
            0
        } else {
            let idx = ((ring.len() as f64) * 0.95).ceil() as usize;
            ring[idx.min(ring.len()) - 1] as u64
        };
        DbActorStats {
            enqueued: self.enqueued.load(Ordering::Relaxed),
            completed: self.completed.load(Ordering::Relaxed),
            batches: self.batches.load(Ordering::Relaxed),
            p95_wait_us: p95,
            max_wait_us: self.max_wait_us.load(Ordering::Relaxed),
            max_block_us: self.max_block_us.load(Ordering::Relaxed),
            worker_blocked_over_5ms: self.worker_blocked_over_5ms.load(Ordering::Relaxed),
            maintenance_checkpoints: self.maintenance_checkpoints.load(Ordering::Relaxed),
            maintenance_errors: self.maintenance_errors.load(Ordering::Relaxed),
            max_maintenance_us: self.max_maintenance_us.load(Ordering::Relaxed),
            last_maintenance_error: self
                .last_maintenance_error
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .clone(),
            max_queue_depth: self.queue_depth_high.load(Ordering::Relaxed),
        }
    }
}

/// The actor: a dedicated std thread owning the hot writes of one store.
/// Cheap to build — no thread and no bridge exist until the first async call
/// needs one.
pub struct DbActor {
    shared: Arc<ActorShared>,
}

impl std::fmt::Debug for DbActor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DbActor").finish_non_exhaustive()
    }
}

impl DbActor {
    /// Create the actor for `store` (lazy: no thread is spawned until the
    /// first async call; the caller queue is bounded by `cfg.capacity`).
    pub fn spawn(store: Arc<Store>, cfg: DbActorConfig) -> Arc<Self> {
        Arc::new(Self {
            shared: Arc::new(ActorShared {
                store,
                cfg: Mutex::new(cfg),
                stats: StatsCore::default(),
                tx: Mutex::new(None),
                bridge: Mutex::new(None),
                fatal: AtomicBool::new(false),
                panic_after_batches: AtomicU64::new(0),
                batches_done: AtomicU64::new(0),
                maintenance_batches: AtomicU64::new(0),
                maintenance_failed: AtomicBool::new(false),
                exit: ExitSignal::default(),
            }),
        })
    }

    /// Replace the tuning knobs (tests): applies to the next actor start or
    /// respawn, not to a thread already running.
    #[doc(hidden)]
    pub fn set_config_for_test(&self, cfg: DbActorConfig) {
        *self.shared.cfg.lock().unwrap_or_else(|p| p.into_inner()) = cfg;
    }

    fn cfg(&self) -> DbActorConfig {
        self.shared
            .cfg
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// The cloneable async write surface.
    pub fn handle(self: &Arc<Self>) -> StoreHandle {
        StoreHandle { db: self.clone() }
    }

    /// Instrumentation snapshot (see [`DbActorStats`]).
    pub fn stats(&self) -> DbActorStats {
        self.shared.stats.snapshot()
    }

    /// The shared store the actor writes through (read paths stay direct).
    pub fn store(&self) -> Arc<Store> {
        self.shared.store.clone()
    }

    /// Test seam: panic the actor thread (replying errors to its in-flight
    /// callers) when the executed-batch counter reaches `target`; `None`
    /// disarms. Cleared automatically when a replacement actor is spawned.
    #[doc(hidden)]
    pub fn set_panic_after_batches(&self, target: Option<u64>) {
        self.shared
            .panic_after_batches
            .store(target.unwrap_or(0), Ordering::SeqCst);
    }

    /// Close the caller queue and wait (bounded by `timeout`) until the
    /// actor has drained every enqueued write, replied, and exited. Returns
    /// whether the actor actually exited. Idempotent; `true` when no actor
    /// was ever started.
    pub async fn shutdown(&self, timeout: Duration) -> bool {
        {
            let mut tx = self.shared.tx.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(tx) = tx.take() {
                drop(tx);
            }
        }
        let shared = self.shared.clone();
        tokio::task::spawn_blocking(move || {
            let mut state = shared.exit.state.lock().unwrap_or_else(|p| p.into_inner());
            if *state {
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

    /// Whether a fatal store fault (poisoned writer lock) permanently failed
    /// the actor.
    pub fn is_fatal(&self) -> bool {
        self.shared.fatal.load(Ordering::Relaxed)
    }

    /// Test hook: enqueue `op` and drop the reply receiver immediately —
    /// the caller never observes the ack, the actor still executes it.
    /// Returns false when there is no queue to enqueue into.
    #[cfg(test)]
    pub(crate) async fn enqueue_unawaited(&self, op: ActorOp) -> bool {
        let Some(tx) = self.ensure_started().await else {
            return false;
        };
        let (reply_tx, _reply_rx) = oneshot::channel();
        let ok = tx
            .send(Envelope {
                op,
                reply: reply_tx,
            })
            .await
            .is_ok();
        if ok {
            self.shared.stats.enqueued.fetch_add(1, Ordering::Relaxed);
        }
        ok
    }

    /// Lazy bridge start. Returns the bounded sender, or `None` when no
    /// tokio runtime is reachable (the caller must fall back to a direct
    /// synchronous store call) or the actor is permanently failed.
    async fn ensure_started(&self) -> Option<mpsc::Sender<Envelope>> {
        {
            let tx = self.shared.tx.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(tx) = tx.as_ref() {
                return Some(tx.clone());
            }
            if self.shared.fatal.load(Ordering::Relaxed) {
                return None;
            }
        }
        // No bridge yet: spawn one on the current runtime. Building the
        // channel needs no runtime; the tokio task does.
        let handle = tokio::runtime::Handle::try_current().ok()?;
        let cfg = self.cfg();
        let (tx, rx) = mpsc::channel::<Envelope>(cfg.capacity);
        let shared = self.shared.clone();
        let mut guard = self.shared.tx.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(existing) = guard.as_ref() {
            return Some(existing.clone());
        }
        *guard = Some(tx.clone());
        let bridge = handle.spawn(bridge_main(shared, rx));
        *self.shared.bridge.lock().unwrap_or_else(|p| p.into_inner()) = Some(bridge);
        Some(tx)
    }

    /// Execute one op directly and synchronously on the CALLER'S thread.
    /// Only used when no tokio runtime is reachable (documented degrade:
    /// identical store semantics to the pre-actor surface, blocking the
    /// caller).
    fn direct(&self, op: ActorOp) -> StoreResult<ActorOutcome> {
        let store = &*self.shared.store;
        match op {
            ActorOp::AppendEvent {
                session_id,
                op_id,
                kind,
                state,
                ts_ms,
                payload,
            } => store
                .append_event(session_id, op_id, kind, state, ts_ms, payload)
                .map(ActorOutcome::EventSeq),
            ActorOp::PutMessage {
                session_id,
                seq,
                role,
                data,
            } => store
                .put_message(session_id, seq, &role, data)
                .map(ActorOutcome::RowId),
            ActorOp::PutPart {
                message_id,
                kind,
                data,
            } => store
                .put_part(message_id, &kind, data)
                .map(ActorOutcome::RowId),
            ActorOp::SettleUsage {
                session_id,
                op_id,
                provider,
                model,
                status,
                tokens_in,
                tokens_out,
                error,
            } => store
                .record_provider_call(
                    session_id,
                    op_id,
                    &provider,
                    &model,
                    &status,
                    tokens_in,
                    tokens_out,
                    error.as_deref(),
                )
                .map(ActorOutcome::RowId),
        }
    }

    async fn call(&self, op: ActorOp) -> StoreResult<ActorOutcome> {
        if self.shared.fatal.load(Ordering::Relaxed) {
            return Err(StoreError::Migration(
                "db actor permanently failed (store writer poisoned)".into(),
            ));
        }
        let Some(tx) = self.ensure_started().await else {
            // No runtime reachable: synchronous degrade keeps callers
            // working with identical store semantics.
            return self.direct(op);
        };
        let (reply_tx, reply_rx) = oneshot::channel();
        tx.send(Envelope {
            op,
            reply: reply_tx,
        })
        .await
        .map_err(|_| StoreError::Migration("db actor queue closed (runtime gone?)".into()))?;
        self.shared.stats.enqueued.fetch_add(1, Ordering::Relaxed);
        let started = Instant::now();
        let out = reply_rx
            .await
            .map_err(|_| StoreError::Migration("db actor dropped an in-flight request".into()))?;
        self.shared.stats.record_wait(started.elapsed());
        out
    }
}

/// Bridge task: coalesce envelopes from the bounded tokio channel into
/// batches (arrival order, at most `max_batch`, flushed by `flush_tick`) and
/// hand each batch to the actor thread over a rendezvous std channel.
/// Supervises actor-thread lifetime: on a death it replies nothing itself
/// (the dying actor already replied to its in-flight batch) and spawns a
/// replacement from the same shared store.
async fn bridge_main(shared: Arc<ActorShared>, mut rx: mpsc::Receiver<Envelope>) {
    let cfg = shared.cfg.lock().unwrap_or_else(|p| p.into_inner()).clone();
    // Rendezvous (capacity 0): a batch is handed to the actor only when the
    // actor is parked in recv, so nothing is queued on the std side and
    // nothing can be orphaned by an actor death.
    let mut relay: Option<std::sync::mpsc::SyncSender<Vec<Envelope>>> = None;

    loop {
        if relay.is_none() {
            relay = Some(spawn_actor_thread(&shared));
        }

        // Wait for the first write of the next batch (or close).
        let first = match rx.recv().await {
            Some(e) => e,
            None => {
                // All caller handles are gone. Deliver nothing more (there
                // is nothing pending) and let the actor exit after its
                // current batch.
                if let Some(tx) = relay.take() {
                    drop(tx);
                }
                break;
            }
        };
        let mut batch = vec![first];
        // Coalesce: fill up to max_batch, bounded by the flush tick.
        if batch.len() < cfg.max_batch {
            let tick = tokio::time::sleep(cfg.flush_tick);
            tokio::pin!(tick);
            loop {
                tokio::select! {
                    biased;
                    env = rx.recv() => match env {
                        Some(e) => {
                            batch.push(e);
                            if batch.len() == cfg.max_batch {
                                break;
                            }
                        }
                        None => break,
                    },
                    _ = &mut tick => break,
                }
            }
        }
        // Observe bounded queue occupancy for stats (occupancy + this batch).
        shared
            .stats
            .observe_queue_depth(rx.len() as u64 + batch.len() as u64);
        handoff_batch(&shared, &mut relay, batch).await;
    }
}

/// Deliver one batch to the actor thread, respawning the thread if it died.
async fn handoff_batch(
    shared: &Arc<ActorShared>,
    relay: &mut Option<std::sync::mpsc::SyncSender<Vec<Envelope>>>,
    mut batch: Vec<Envelope>,
) {
    loop {
        let tx = match relay.as_ref() {
            Some(tx) => tx,
            None => {
                // Death between batches: spawn a replacement and retry.
                *relay = Some(spawn_actor_thread(shared));
                relay.as_ref().expect("just spawned")
            }
        };
        match tx.try_send(batch) {
            Ok(()) => return,
            Err(std::sync::mpsc::TrySendError::Full(pending)) => {
                batch = pending;
                // Actor busy executing: brief non-blocking backoff; the
                // bounded tokio stage absorbs the pressure.
                tokio::time::sleep(Duration::from_micros(100)).await;
            }
            Err(std::sync::mpsc::TrySendError::Disconnected(pending)) => {
                batch = pending;
                *relay = None;
            }
        }
    }
}

fn spawn_actor_thread(shared: &Arc<ActorShared>) -> std::sync::mpsc::SyncSender<Vec<Envelope>> {
    let (tx, rx_std) = std::sync::mpsc::sync_channel(0);
    let thread_shared = shared.clone();
    let cfg = shared.cfg.lock().unwrap_or_else(|p| p.into_inner()).clone();
    let _ = std::thread::Builder::new()
        .name(ACTOR_THREAD_NAME.to_string())
        .spawn(move || actor_main(thread_shared, cfg, rx_std));
    // One-shot panic seams can never re-fire: the seam target is compared
    // against the cumulative executed-batch counter, which keeps advancing
    // across respawns.
    tx
}

/// Actor thread main: pull batches in arrival order and execute them.
fn actor_main(
    shared: Arc<ActorShared>,
    cfg: DbActorConfig,
    rx: std::sync::mpsc::Receiver<Vec<Envelope>>,
) {
    // Panic safety net: whatever batch was in flight when a panic unwound
    // must have its callers replied to (never a hang).
    let in_flight: Arc<Mutex<Vec<oneshot::Sender<StoreResult<ActorOutcome>>>>> =
        Arc::new(Mutex::new(Vec::new()));
    let unwind_result = catch_unwind(AssertUnwindSafe(|| {
        actor_loop(shared.clone(), cfg, &rx, &in_flight);
    }));
    if unwind_result.is_err() {
        fail_in_flight(&in_flight, "db actor thread panicked");
    }
    shared.exit.notify_exited();
}

fn actor_loop(
    shared: Arc<ActorShared>,
    cfg: DbActorConfig,
    rx: &std::sync::mpsc::Receiver<Vec<Envelope>>,
    in_flight: &Arc<Mutex<Vec<oneshot::Sender<StoreResult<ActorOutcome>>>>>,
) {
    loop {
        match rx.recv_timeout(cfg.flush_tick) {
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                // Queue drained for a full flush tick: fold the WAL with a
                // bounded PASSIVE checkpoint. Maintenance is typed
                // (`Segment::Maintenance`) and runs OUTSIDE every measured
                // interactive segment; SQLite's own auto-checkpoint is
                // disabled, so no checkpoint work can land inside a batch.
                run_maintenance(&shared, &cfg);
                continue;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                // Bridge gone; recv_timeout only reports Disconnected once
                // the queue is drained, so every delivered batch ran.
                return;
            }
            Ok(mut batch) => {
                if execute_batch(&shared, &cfg, &mut batch, in_flight) {
                    // Actor death (test seam or poisoned store): reply the
                    // current batch with an error and exit; the bridge
                    // spawns a replacement.
                    fail_batch(&mut batch, "db actor died mid-batch");
                    return;
                }
            }
        }
    }
}

/// Idle-tick maintenance: run a bounded PASSIVE WAL checkpoint while the
/// queue is drained, recorded as a typed `Segment::Maintenance` (never the
/// interactive 5 ms gate). Failures are logged (`tracing::warn!`) and
/// counted, never fatal: the actor keeps serving.
fn run_maintenance(shared: &Arc<ActorShared>, cfg: &DbActorConfig) {
    // Maintenance runs on EVERY drained tick: it is also the checkpoint
    // driver for frames appended by direct (non-actor) writers, which the
    // actor cannot count. Only a FAILED attempt waits for new batch work
    // before retrying, so a persistent error cannot spam the log at
    // flush-tick frequency.
    let done = shared.batches_done.load(Ordering::Relaxed);
    if shared.maintenance_failed.load(Ordering::Relaxed)
        && done == shared.maintenance_batches.load(Ordering::Relaxed)
    {
        return;
    }
    shared.maintenance_batches.store(done, Ordering::Relaxed);
    // The seam delay models checkpoint-heavy work INSIDE the typed
    // maintenance region (it must never touch the interactive timers).
    let t0 = Instant::now();
    if let Some(delay) = cfg.maintenance_delay {
        std::thread::sleep(delay);
    }
    let result = catch_unwind(AssertUnwindSafe(|| match &cfg.maintenance_fail_for_test {
        Some(message) => Err(StoreError::Migration(format!(
            "wal checkpoint maintenance (test seam): {message}"
        ))),
        None => shared.store.wal_checkpoint_passive(),
    }));
    let elapsed = t0.elapsed();
    let (error, failed) = match result {
        Ok(Ok(())) => (None, false),
        Ok(Err(e)) => {
            let message = e.to_string();
            tracing::warn!(error = %message, "db actor wal checkpoint (maintenance) failed; non-fatal");
            (Some(message), true)
        }
        Err(_) => {
            let message = "db actor wal checkpoint (maintenance) panicked; caught, non-fatal";
            tracing::warn!("{message}");
            (Some(message.to_string()), true)
        }
    };
    shared.maintenance_failed.store(failed, Ordering::Relaxed);
    shared
        .stats
        .record_segment(Segment::Maintenance { elapsed, error });
}

/// Consumed CPU time of the calling thread, when the platform clock is
/// available (unix).
///
/// The audit gate times store WORK, not scheduling: under machine-wide load
/// (other test binaries / heavy fixtures sharing the box) a preempted actor
/// thread accumulates arbitrary WALL time inside the timed region without
/// executing any SQLite work, which made `worker_blocked_over_5ms` flaky.
/// Thread CPU time is invariant to descheduling and to the deliberate commit
/// fsync wait, so the gate keeps its documented meaning ("synchronous store
/// work segment"). `None` on platforms without the clock: callers fall back
/// to the store's own wall-clock `BatchTiming` split.
#[cfg(unix)]
fn thread_cpu_time() -> Option<Duration> {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a live, properly aligned `timespec`; the clock id is a
    // constant valid for thread-CPU accounting on every unix target.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    if rc != 0 {
        return None;
    }
    Some(Duration::new(
        ts.tv_sec.max(0) as u64,
        ts.tv_nsec.max(0) as u32,
    ))
}

#[cfg(not(unix))]
fn thread_cpu_time() -> Option<Duration> {
    None
}

/// Returns `true` when the actor must die after this batch.
fn execute_batch(
    shared: &Arc<ActorShared>,
    cfg: &DbActorConfig,
    batch: &mut Vec<Envelope>,
    in_flight: &Arc<Mutex<Vec<oneshot::Sender<StoreResult<ActorOutcome>>>>>,
) -> bool {
    if batch.is_empty() {
        return false;
    }
    let idx = shared.batches_done.fetch_add(1, Ordering::SeqCst);
    let seam = shared.panic_after_batches.load(Ordering::SeqCst);
    if seam != 0 && idx + 1 == seam {
        // Test seam: die after replying to this in-flight batch (callers
        // see a store error, never a hang); the bridge respawns the actor.
        shared
            .stats
            .completed
            .fetch_add(batch.len() as u64, Ordering::Relaxed);
        return true;
    }
    if let Some(delay) = cfg.pre_batch_delay {
        std::thread::sleep(delay);
    }
    let op_count = batch.len();
    let ops: Vec<HotWrite> = batch.iter().map(|e| e.op.to_hot_write()).collect();

    // Register the reply senders so ANY unwind (including a store-side
    // panic) can fail them before the thread exits.
    {
        let mut guard = in_flight.lock().unwrap_or_else(|p| p.into_inner());
        for e in batch.drain(..) {
            guard.push(e.reply);
        }
    }

    // The timed synchronous store segment (audit gate: instrumented). The
    // store splits SQL work from the deliberate commit fsync: the >5 ms
    // gate counts the WORK segment (SQLite work that used to block Tokio
    // workers), while the fsync wait surfaces as caller-side queue latency.
    let cpu_start = thread_cpu_time();
    let t0 = Instant::now();
    let result = catch_unwind(AssertUnwindSafe(|| shared.store.batch_hot_writes(&ops)));
    let total = t0.elapsed();
    let work = match cpu_start.zip(thread_cpu_time()) {
        // Preferred: consumed CPU (descheduling- and fsync-invariant).
        Some((start, end)) => end.saturating_sub(start),
        // Fallback: the store's own wall-clock SQL-work split.
        None => match &result {
            Ok(Ok((_, timing))) => Duration::from_micros(timing.work_us),
            _ => total,
        },
    };
    shared
        .stats
        .record_segment(Segment::Interactive { work, total });

    let mut guard = in_flight.lock().unwrap_or_else(|p| p.into_inner());
    match result {
        Ok(Ok((outcomes, _timing))) => {
            // The group committed as ONE fsynced transaction: reply per
            // write, preserving per-write errors from the store.
            for (sender, outcome) in guard.drain(..).zip(outcomes) {
                let mapped = outcome.map(|o| match o {
                    HotWriteOutcome::EventSeq(seq) => ActorOutcome::EventSeq(seq),
                    HotWriteOutcome::RowId(id) => ActorOutcome::RowId(id),
                });
                let _ = sender.send(mapped);
            }
        }
        Ok(Err(e)) => {
            // Whole-group infrastructure failure (busy timeout / commit
            // error): every write of the batch failed as one transaction.
            let message = format!("db actor batch failed: {e}");
            for sender in guard.drain(..) {
                let _ = sender.send(Err(StoreError::Migration(message.clone())));
            }
        }
        Err(_) => {
            // A store panic poisoned the writer lock: fatal for every later
            // write (direct or actor), so stop respawning doomed threads.
            shared.fatal.store(true, Ordering::SeqCst);
            for sender in guard.drain(..) {
                let _ = sender.send(Err(StoreError::Migration(
                    "db actor store segment panicked; actor permanently failed".into(),
                )));
            }
            shared
                .stats
                .completed
                .fetch_add(op_count as u64, Ordering::Relaxed);
            return true;
        }
    }
    shared
        .stats
        .completed
        .fetch_add(op_count as u64, Ordering::Relaxed);
    shared.stats.batches.fetch_add(1, Ordering::Relaxed);
    false
}

fn fail_batch(batch: &mut Vec<Envelope>, why: &str) {
    for e in batch.drain(..) {
        let _ = e.reply.send(Err(StoreError::Migration(why.to_string())));
    }
}

fn fail_in_flight(
    in_flight: &Arc<Mutex<Vec<oneshot::Sender<StoreResult<ActorOutcome>>>>>,
    why: &str,
) {
    let mut guard = in_flight.lock().unwrap_or_else(|p| p.into_inner());
    for sender in guard.drain(..) {
        let _ = sender.send(Err(StoreError::Migration(why.to_string())));
    }
}

/// Cloneable async surface mirroring the store's hot call surface ONLY.
/// All other store calls stay direct and synchronous on the shared
/// [`Store`] (see `Store::direct` in faktor-store).
#[derive(Clone)]
pub struct StoreHandle {
    db: Arc<DbActor>,
}

impl StoreHandle {
    /// Instrumentation of the underlying actor.
    pub fn stats(&self) -> DbActorStats {
        self.db.stats()
    }

    /// Append one message row. Awaits the fsynced store response.
    pub async fn append_message(
        &self,
        session_id: SessionId,
        seq: i64,
        role: &str,
        data: serde_json::Value,
    ) -> StoreResult<i64> {
        let out = self
            .db
            .call(ActorOp::PutMessage {
                session_id,
                seq,
                role: role.to_string(),
                data,
            })
            .await?;
        match out {
            ActorOutcome::RowId(id) => Ok(id),
            other => Err(StoreError::Migration(format!(
                "db actor returned {other:?} for a message append"
            ))),
        }
    }

    /// Append one part row to `message_id`. Awaits the fsynced store
    /// response.
    pub async fn append_part(
        &self,
        message_id: i64,
        kind: &str,
        data: serde_json::Value,
    ) -> StoreResult<i64> {
        let out = self
            .db
            .call(ActorOp::PutPart {
                message_id,
                kind: kind.to_string(),
                data,
            })
            .await?;
        match out {
            ActorOutcome::RowId(id) => Ok(id),
            other => Err(StoreError::Migration(format!(
                "db actor returned {other:?} for a part append"
            ))),
        }
    }

    /// Append one journal event with the next gapless per-session sequence.
    /// Awaits the fsynced store response.
    #[allow(clippy::too_many_arguments)]
    pub async fn append_journal_event(
        &self,
        session_id: SessionId,
        op_id: Option<OpId>,
        kind: EventKind,
        state: AgentState,
        ts_ms: i64,
        payload: Option<serde_json::Value>,
    ) -> StoreResult<EventSeq> {
        let out = self
            .db
            .call(ActorOp::AppendEvent {
                session_id,
                op_id,
                kind,
                state,
                ts_ms,
                payload,
            })
            .await?;
        match out {
            ActorOutcome::EventSeq(seq) => Ok(seq),
            other => Err(StoreError::Migration(format!(
                "db actor returned {other:?} for a journal append"
            ))),
        }
    }

    /// Settle one provider usage frame into the durable provider-call rows
    /// (tokens in/out, status, error). Awaits the fsynced store response.
    #[allow(clippy::too_many_arguments)]
    pub async fn settle_usage(
        &self,
        session_id: SessionId,
        op_id: OpId,
        provider: &str,
        model: &str,
        status: &str,
        tokens_in: Option<u64>,
        tokens_out: Option<u64>,
        error: Option<&str>,
    ) -> StoreResult<i64> {
        let out = self
            .db
            .call(ActorOp::SettleUsage {
                session_id,
                op_id,
                provider: provider.to_string(),
                model: model.to_string(),
                status: status.to_string(),
                tokens_in,
                tokens_out,
                error: error.map(|e| e.to_string()),
            })
            .await?;
        match out {
            ActorOutcome::RowId(id) => Ok(id),
            other => Err(StoreError::Migration(format!(
                "db actor returned {other:?} for a usage settlement"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faktor_core::event::EventKind;
    use faktor_core::id::OpId;
    use faktor_core::state::AgentState;
    use faktor_store::{MessageRow, Store};

    fn tmp_actor(cfg: DbActorConfig) -> (tempfile::TempDir, Arc<Store>, Arc<DbActor>) {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(Store::open(dir.path().join("store"), false).unwrap());
        let actor = DbActor::spawn(store.clone(), cfg);
        (dir, store, actor)
    }

    fn new_session(store: &Store) -> SessionId {
        let ws = store.create_workspace("/w").unwrap();
        store.create_session(ws, "t", "p", "m").unwrap().id
    }

    fn msg_rows(store: &Store, sid: SessionId) -> Vec<MessageRow> {
        let mut out = Vec::new();
        let mut cursor = None;
        loop {
            let page = store.messages_before(sid, cursor, 200).unwrap();
            if page.is_empty() {
                break;
            }
            cursor = page.last().map(|m| m.seq);
            out.extend(page);
        }
        out.sort_by_key(|m| m.seq);
        out
    }

    fn event_rows(store: &Store, sid: SessionId) -> Vec<faktor_core::event::Event> {
        store.events_range(sid, 1, None).unwrap()
    }

    /// Heavyweight actor tests serialize on one tokio mutex: FULL-sync batch
    /// storms must never overlap each other inside one test binary (their
    /// fsync latency spikes would trip the 5 ms instrumentation gate).
    async fn serial_lock() -> tokio::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::OnceLock<tokio::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await
    }

    /// One production-shaped burst: `producers` concurrent tasks each
    /// appending `per_producer` messages on `sid`, with `base` offsetting the
    /// sequence numbers (so a second burst can continue the same session).
    async fn append_burst(
        handle: &StoreHandle,
        sid: SessionId,
        producers: i64,
        per_producer: i64,
        base: i64,
    ) {
        let mut senders = Vec::new();
        for p in 0..producers {
            let h = handle.clone();
            senders.push(tokio::spawn(async move {
                for i in 0..per_producer {
                    let seq = base + p * per_producer + i + 1;
                    h.append_message(sid, seq, "assistant", serde_json::json!({ "i": seq }))
                        .await
                        .expect("append ok");
                }
            }));
        }
        for p in senders {
            tokio::time::timeout(Duration::from_secs(60), p)
                .await
                .expect("burst must finish")
                .expect("no panic");
        }
    }

    /// Poll `actor.stats()` until `pred` holds; panic with the last snapshot
    /// after `timeout` (the actor thread is independent of the runtime, so
    /// polling with a sleep is deterministic enough and never hangs).
    async fn wait_for_stats(
        actor: &Arc<DbActor>,
        what: &str,
        timeout: Duration,
        pred: impl Fn(&DbActorStats) -> bool,
    ) {
        let deadline = Instant::now() + timeout;
        while !pred(&actor.stats()) {
            assert!(
                Instant::now() < deadline,
                "timed out waiting for {what}: {:?}",
                actor.stats()
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    #[tokio::test]
    async fn burst_of_10k_appends_cap_32_loses_nothing_and_stays_bounded() {
        let _serial = serial_lock().await;
        // (a) 10k appends over 8 sessions, channel capacity 32: none lost,
        // per-session order preserved, queue depth never exceeds the cap,
        // backpressure visible, no OOM.
        let (_d, store, actor) = tmp_actor(DbActorConfig {
            capacity: 32,
            max_batch: 16,
            flush_tick: Duration::from_millis(1),
            ..Default::default()
        });
        let handle = actor.handle();
        let sessions: Vec<SessionId> = (0..8).map(|_| new_session(&store)).collect();
        let per_session = 1250usize;
        let mut producers = Vec::new();
        for sid in sessions.clone() {
            let h = handle.clone();
            producers.push(tokio::spawn(async move {
                for seq in 1..=per_session as i64 {
                    h.append_message(sid, seq, "assistant", serde_json::json!({ "i": seq }))
                        .await
                        .expect("append must not fail under burst");
                }
            }));
        }
        for p in producers {
            tokio::time::timeout(Duration::from_secs(120), p)
                .await
                .expect("burst producers must finish")
                .expect("no producer task panic");
        }
        let stats = actor.stats();
        assert_eq!(stats.enqueued, 10_000);
        assert_eq!(stats.completed, 10_000, "none lost");
        assert!(stats.batches >= 10_000 / 16, "batching must coalesce");
        // Per-session order: each session holds exactly its own 1..=N.
        for sid in &sessions {
            let rows = msg_rows(&store, *sid);
            assert_eq!(rows.len(), per_session, "session {sid} lost rows");
            for (i, m) in rows.iter().enumerate() {
                assert_eq!(m.seq, i as i64 + 1, "per-session order broken");
            }
        }
        // Bounded queue: the caller-visible stage never exceeds the cap.
        assert!(
            stats.max_queue_depth <= 32,
            "queue depth {max} must never exceed capacity 32",
            max = stats.max_queue_depth
        );
        // Backpressure visible: callers waited for capacity/acks.
        assert!(stats.max_wait_us > 0, "burst must observe queue waits");
        assert!(stats.p95_wait_us <= stats.max_wait_us);
    }

    #[tokio::test]
    async fn backpressure_is_deterministic_when_the_actor_is_slowed() {
        let _serial = serial_lock().await;
        // A deliberately slow actor (1 ms pre-batch delay, max_batch 1)
        // must make callers WAIT (backpressure), not buffer without bound.
        let (_d, store, actor) = tmp_actor(DbActorConfig {
            capacity: 8,
            max_batch: 1,
            flush_tick: Duration::from_millis(1),
            pre_batch_delay: Some(Duration::from_millis(1)),
            ..Default::default()
        });
        let handle = actor.handle();
        let sid = new_session(&store);
        let mut senders = Vec::new();
        for p in 0..16i64 {
            let h = handle.clone();
            senders.push(tokio::spawn(async move {
                for i in 0..25i64 {
                    let seq = p * 25 + i + 1;
                    h.append_message(sid, seq, "assistant", serde_json::json!({ "i": seq }))
                        .await
                        .expect("append must succeed");
                }
            }));
        }
        for p in senders {
            tokio::time::timeout(Duration::from_secs(60), p)
                .await
                .expect("slowed actor must still finish")
                .expect("no panic");
        }
        let stats = actor.stats();
        assert_eq!(stats.completed, 16 * 25);
        assert!(stats.max_queue_depth <= 8, "bounded stage only");
        assert!(
            stats.max_wait_us >= 500,
            "callers must visibly wait when the actor is slowed: {max}",
            max = stats.max_wait_us
        );
    }

    #[tokio::test]
    async fn actor_killed_mid_batch_errors_inflight_and_respawns() {
        let _serial = serial_lock().await;
        // (b) Seam: the actor thread dies mid-batch. In-flight callers get
        // an error (none hang: whole producer is timeout-wrapped), the
        // pre-crash appends are durable, and a replacement actor thread
        // serves subsequent appends.
        let (_d, store, actor) = tmp_actor(DbActorConfig {
            capacity: 64,
            max_batch: 8,
            flush_tick: Duration::from_millis(2),
            ..Default::default()
        });
        let handle = actor.handle();
        let sid = new_session(&store);
        // Arm the seam: batch #10 dies (each batch holds <= 8 writes).
        actor.set_panic_after_batches(Some(10));
        let mut producers = Vec::new();
        for p in 0..8i64 {
            let h = handle.clone();
            producers.push(tokio::spawn(async move {
                let mut results = Vec::new();
                for i in 0..60i64 {
                    let seq = p * 60 + i + 1;
                    let r = h
                        .append_message(sid, seq, "assistant", serde_json::json!({ "i": seq }))
                        .await;
                    results.push((seq, r.is_ok()));
                }
                results
            }));
        }
        let joined = tokio::time::timeout(Duration::from_secs(120), async {
            let mut all = Vec::new();
            for p in producers {
                all.extend(p.await.expect("producer panicked"));
            }
            all
        })
        .await
        .expect("in-flight callers must never hang when the actor dies");
        let ok = joined.iter().filter(|(_, ok)| *ok).count();
        let err = joined.iter().filter(|(_, ok)| !*ok).count();
        assert_eq!(ok + err, 8 * 60, "every caller got a reply");
        assert!(
            (1..=8).contains(&err),
            "only the dying batch's writes error (<= max_batch), got {err}"
        );
        assert!(
            ok >= 8 * 60 - 8,
            "everything else (incl. post-respawn) lands: {ok}"
        );
        let stats = actor.stats();
        assert_eq!(stats.completed, 8 * 60, "all replied (ok or error)");
        // The replacement actor served subsequent appends: seqs up to 480.
        let rows = msg_rows(&store, sid);
        assert_eq!(rows.len(), ok, "only acked rows are durable");
        assert_eq!(
            rows.last().unwrap().seq,
            8 * 60,
            "post-respawn appends land"
        );
    }

    #[tokio::test]
    async fn acked_appends_survive_reopen_fsync_before_ack() {
        let _serial = serial_lock().await;
        // (c) Every ACKED append is durable: after the actor drains and the
        // process is "killed" (fresh open_fast over the same root), all
        // acked message rows and journal events are present.
        let dir = tempfile::tempdir().unwrap();
        let sid;
        let total = 300u64;
        {
            let store = Arc::new(Store::open(dir.path().join("store"), false).unwrap());
            let actor = DbActor::spawn(
                store.clone(),
                DbActorConfig {
                    max_batch: 8,
                    ..Default::default()
                },
            );
            let handle = actor.handle();
            let ws = store.create_workspace("/w").unwrap();
            let row = store.create_session(ws, "t", "p", "m").unwrap();
            sid = row.id;
            for seq in 1..=total as i64 {
                let ev = handle
                    .append_journal_event(
                        sid,
                        Some(OpId::new(seq as u64)),
                        EventKind::ModelStarted,
                        AgentState::Streaming,
                        seq,
                        None,
                    )
                    .await
                    .expect("event ack");
                assert_eq!(ev.raw(), (seq + 1) as u64, "seed is seq 1");
                handle
                    .append_message(sid, seq, "assistant", serde_json::json!({}))
                    .await
                    .expect("message ack");
            }
            assert!(actor.shutdown(Duration::from_secs(30)).await);
        }
        let reopened = Store::open_fast(dir.path().join("store")).unwrap();
        assert_eq!(
            reopened.message_count(sid).unwrap(),
            total as i64,
            "every acked message survived the simulated kill"
        );
        assert_eq!(
            reopened.last_event_seq(sid).unwrap().unwrap().raw(),
            total + 1,
            "every acked journal event survived"
        );
    }

    #[tokio::test]
    async fn five_ms_gate_reports_zero_blocked_segments_under_load() {
        let _serial = serial_lock().await;
        // (d) Audit gate: a 2048-append burst (64 producers x 32) must show
        // ZERO interactive store WORK segments over 5 ms — batching keeps
        // each segment tiny (instrumented, not inferred). SQLite's automatic
        // checkpoint is disabled at the store and checkpointing runs only as
        // typed idle-tick maintenance, so WAL growth can no longer leak into
        // a measured interactive segment. The gate keeps its teeth: ONE
        // attempt, no retry. (The old 20-round retry could not have helped
        // anyway: `worker_blocked_over_5ms` is a cumulative counter that only
        // increments, so once a round tripped it no later round could pass.)
        let (_d, store, actor) = tmp_actor(DbActorConfig {
            capacity: 2048,
            max_batch: 32,
            flush_tick: Duration::from_millis(1),
            ..Default::default()
        });
        let handle = actor.handle();
        let sid = new_session(&store);
        append_burst(&handle, sid, 64, 32, 0).await;
        let stats = actor.stats();
        assert_eq!(stats.completed, 2048, "burst lost appends");
        assert_eq!(
            stats.worker_blocked_over_5ms, 0,
            "interactive store segments must stay under 5 ms: {stats:?}"
        );
        assert!(stats.max_block_us > 0, "interactive segments instrumented");
        assert_eq!(store.message_count(sid).unwrap(), 2048);
    }

    #[tokio::test]
    async fn slow_maintenance_is_excluded_by_type_and_gate_stays_zero() {
        let _serial = serial_lock().await;
        // Checkpoint-heavy store: maintenance is artificially slowed far past
        // 5 ms (the deterministic stand-in for a huge WAL on a slow disk).
        // The interactive gate must stay at ZERO because maintenance is a
        // different segment TYPE, while the slow checkpoint is fully visible
        // in its OWN timer — proving the exclusion is structural, not a
        // tolerance.
        let (_d, store, actor) = tmp_actor(DbActorConfig {
            capacity: 2048,
            max_batch: 32,
            flush_tick: Duration::from_millis(1),
            maintenance_delay: Some(Duration::from_millis(25)),
            ..Default::default()
        });
        let handle = actor.handle();
        let sid = new_session(&store);
        // First half of the burst, then wait for the slowed maintenance.
        append_burst(&handle, sid, 64, 16, 0).await;
        wait_for_stats(
            &actor,
            "the slowed maintenance checkpoint",
            Duration::from_secs(15),
            |s| s.maintenance_checkpoints >= 1,
        )
        .await;
        // The WAL fold itself is certified by the faktor-store maintenance
        // tests; here the second half runs only after at least one >5 ms
        // maintenance segment was recorded, and the interactive gate spans
        // all 2048 appends.
        append_burst(&handle, sid, 64, 16, 1024).await;
        let stats = actor.stats();
        assert_eq!(stats.completed, 2048, "whole burst landed");
        assert!(stats.maintenance_checkpoints >= 1, "maintenance ran");
        assert!(
            stats.max_maintenance_us >= 25_000,
            "the slowed checkpoint is fully measured, not hidden: {stats:?}"
        );
        assert_eq!(
            stats.worker_blocked_over_5ms, 0,
            "maintenance is excluded by type; no interactive segment may trip: {stats:?}"
        );
        assert!(stats.max_block_us > 0, "interactive segments instrumented");
        assert_eq!(store.message_count(sid).unwrap(), 2048);
    }

    #[tokio::test]
    async fn maintenance_failure_is_non_fatal_logged_and_recorded() {
        let _serial = serial_lock().await;
        // Maintenance is best-effort: a checkpoint failure (typed store
        // error) is logged (`tracing::warn!`), counted and surfaced — it
        // must never kill the actor or fail later writes.
        let (_d, store, actor) = tmp_actor(DbActorConfig {
            capacity: 32,
            max_batch: 16,
            flush_tick: Duration::from_millis(1),
            maintenance_fail_for_test: Some("injected checkpoint failure".into()),
            ..Default::default()
        });
        let handle = actor.handle();
        let sid = new_session(&store);
        append_burst(&handle, sid, 2, 8, 0).await;
        wait_for_stats(
            &actor,
            "the injected maintenance failure",
            Duration::from_secs(10),
            |s| s.maintenance_errors >= 1,
        )
        .await;
        let stats = actor.stats();
        assert!(
            stats
                .last_maintenance_error
                .as_deref()
                .unwrap_or_default()
                .contains("injected checkpoint failure"),
            "the typed maintenance error is surfaced for logging: {stats:?}"
        );
        assert!(!actor.is_fatal(), "maintenance failure must not be fatal");
        // The actor keeps serving after the failed maintenance.
        handle
            .append_message(sid, 17, "assistant", serde_json::json!({ "i": 17 }))
            .await
            .expect("actor must keep serving after a maintenance failure");
        assert_eq!(actor.stats().completed, 17);
        assert_eq!(store.message_count(sid).unwrap(), 17);
        assert_eq!(
            actor.stats().worker_blocked_over_5ms,
            0,
            "failed maintenance never trips the interactive gate"
        );
    }

    #[tokio::test]
    async fn drop_all_handles_drains_pending_then_exits() {
        let _serial = serial_lock().await;
        // (e) Shutdown: dropping every handle lets the actor drain ALL
        // pending batches (even unawaited ones), flush, then exit.
        let (_d, store, actor) = tmp_actor(DbActorConfig {
            capacity: 32,
            max_batch: 16,
            flush_tick: Duration::from_millis(1),
            ..Default::default()
        });
        let handle = actor.handle();
        let sid = new_session(&store);
        for seq in 1..=150i64 {
            handle
                .append_message(sid, seq, "assistant", serde_json::json!({}))
                .await
                .expect("awaited append");
        }
        // Fire-and-forget: enqueue 150 more, never observing their acks
        // (the actor must still drain and execute them after shutdown).
        for seq in 151..=300i64 {
            assert!(
                actor
                    .enqueue_unawaited(ActorOp::PutMessage {
                        session_id: sid,
                        seq,
                        role: "assistant".into(),
                        data: serde_json::json!({}),
                    })
                    .await,
                "enqueue into the bounded queue must succeed"
            );
        }
        // Drop the LAST handle: queue closes, the actor drains & exits.
        drop(handle);
        assert!(
            actor.shutdown(Duration::from_secs(30)).await,
            "actor must drain and exit after all handles dropped"
        );
        assert_eq!(actor.stats().completed, 300, "pending work drained");
        let rows = msg_rows(&store, sid);
        assert_eq!(rows.len(), 300, "unawaited appends were still flushed");
    }

    #[tokio::test]
    async fn mixed_usage_and_journal_streams_preserve_causal_order() {
        let _serial = serial_lock().await;
        // (f) Usage settlement + journal event + message interleavings keep
        // per-session causal order: a settlement (or message) referencing
        // journal seq N is only enqueued after N was ACKED, so at every
        // observed point the durable journal leads and gaplessness holds.
        let (_d, store, actor) = tmp_actor(DbActorConfig {
            capacity: 32,
            max_batch: 8,
            flush_tick: Duration::from_millis(1),
            ..Default::default()
        });
        let handle = actor.handle();
        let sessions: Vec<SessionId> = (0..4).map(|_| new_session(&store)).collect();
        let mut producers = Vec::new();
        for (i, s) in sessions.iter().enumerate() {
            let h = handle.clone();
            let sid = *s;
            let op = OpId::new(1000 + i as u64);
            producers.push(tokio::spawn(async move {
                let mut ev_seqs = Vec::new();
                for seq in 1..=100i64 {
                    let ev = h
                        .append_journal_event(
                            sid,
                            Some(op),
                            EventKind::ModelStarted,
                            AgentState::Streaming,
                            seq,
                            Some(serde_json::json!({ "n": seq })),
                        )
                        .await
                        .expect("event ack");
                    ev_seqs.push(ev.raw());
                    let mid = h
                        .append_message(sid, seq, "assistant", serde_json::json!({}))
                        .await
                        .expect("message ack");
                    h.append_part(mid, "text", serde_json::json!({ "text": seq }))
                        .await
                        .expect("part ack");
                    h.settle_usage(
                        sid,
                        op,
                        "ollama",
                        "qwen3.8",
                        "completed",
                        Some(seq as u64),
                        Some(1),
                        None,
                    )
                    .await
                    .expect("usage ack");
                }
                ev_seqs
            }));
        }
        for p in producers {
            let seqs = tokio::time::timeout(Duration::from_secs(60), p)
                .await
                .expect("mixed producer must finish")
                .expect("no panic");
            assert_eq!(seqs, (2..=101).collect::<Vec<u64>>(), "gapless per session");
        }
        for s in &sessions {
            let events = event_rows(&store, *s);
            assert_eq!(events.len(), 101, "seed + 100");
            let messages = msg_rows(&store, *s);
            assert_eq!(messages.len(), 100);
            assert_eq!(messages[0].seq, 1);
            assert_eq!(messages[99].seq, 100);
            // Every message references an event seq that is durably present
            // (the message was enqueued only after the event ack).
            assert!(events.iter().any(|e| e.seq.raw() == 100));
            // Usage rows all landed (100 settlements of seq + 1 tokens each).
            assert_eq!(
                store.session_usage_tokens(*s).unwrap(),
                100 * 101 / 2 + 100,
                "every usage settlement is durable"
            );
        }
    }

    #[tokio::test]
    async fn manager_handle_async_twins_route_through_the_actor() {
        // Integration: the SessionHandle async twins (message append / part
        // append) run through the manager's DbActor with the SAME validation
        // as the sync twins, durably.
        let dir = tempfile::tempdir().unwrap();
        let manager =
            crate::SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                .unwrap();
        let ws = manager.create_workspace("/w").unwrap();
        let handle = manager
            .create_session(ws, "t", "ollama", "qwen3.8")
            .unwrap();
        let mid = handle
            .append_message(1, "assistant", serde_json::json!({ "parts": [] }))
            .await
            .unwrap();
        handle.append_text_part(mid, "hello world").await.unwrap();
        handle
            .append_tool_result_part(
                mid,
                "call_1",
                &faktor_protocol::v756::ToolResultBody {
                    excerpt: "ok".into(),
                    exit_code: Some(0),
                    artifact: None,
                    slice_hint: None,
                },
            )
            .await
            .unwrap();
        handle
            .settle_usage(
                OpId::new(9),
                "ollama",
                "qwen3.8",
                "completed",
                Some(10),
                Some(2),
                None,
            )
            .await
            .unwrap();
        // Durable & coherent via the DIRECT surface (reads stay direct).
        assert_eq!(handle.message_count().unwrap(), 1);
        assert_eq!(handle.parts_of(mid).unwrap().len(), 2);
        assert_eq!(
            handle.last_event_seq().unwrap().unwrap().raw(),
            1,
            "journal untouched by the smoke writes"
        );
        let stats = manager.actor().stats();
        assert!(
            stats.enqueued >= 4,
            "all four hot writes went through the actor"
        );
        assert_eq!(stats.completed, stats.enqueued);
        assert_eq!(
            manager.store().session_usage_tokens(handle.id()).unwrap(),
            12
        );
    }

    // ---- P0-44: the wait-sample ring must not shift on eviction ----

    /// Exact p95/max semantics over a known distribution that fills the
    /// ring exactly: samples 1..=8192 µs → sorted p95 index
    /// ceil(8192·0.95) = 7783 → p95 == 7783, max == 8192.
    #[test]
    fn wait_ring_p95_matches_known_distribution_at_capacity() {
        let core = StatsCore::default();
        for i in 1..=WAIT_RING_CAP as u32 {
            core.record_wait(Duration::from_micros(u64::from(i)));
        }
        let stats = core.snapshot();
        assert_eq!(stats.p95_wait_us, 7783);
        assert_eq!(stats.max_wait_us, WAIT_RING_CAP as u64);
        assert!(stats.p95_wait_us <= stats.max_wait_us);
    }

    /// Overflow keeps the MOST RECENT WAIT_RING_CAP samples: after 10_000
    /// increasing samples 1..=10_000 the retained window is 1809..=10_000,
    /// so p95 (the 7783rd smallest of 8192) == 1809 + 7782 == 9591 and the
    /// evicted minimum (1) is gone from the percentile.
    #[test]
    fn wait_ring_evicts_oldest_and_keeps_recent_window() {
        let core = StatsCore::default();
        for i in 1..=10_000u32 {
            core.record_wait(Duration::from_micros(u64::from(i)));
        }
        let stats = core.snapshot();
        assert_eq!(stats.max_wait_us, 10_000);
        assert_eq!(
            stats.p95_wait_us, 9591,
            "p95 must be computed over the retained most-recent window only"
        );
        // A single sample below the retained minimum must not move p95.
        core.record_wait(Duration::from_micros(1));
        let stats = core.snapshot();
        assert_eq!(stats.p95_wait_us, 9591, "evicted samples stay evicted");
        assert_eq!(
            stats.max_wait_us, 10_000,
            "max is a high-water, not a window stat"
        );
    }

    /// Adversarial perf regression: 1M samples over a full ring must stay
    /// linear. The OLD Vec+remove(0) implementation shifted up to 8191
    /// elements per sample (~4×10⁹ moves here); VecDeque pop_front/push_back
    /// is O(1) per sample. The bound is generous for shared CI boxes.
    #[test]
    fn wait_ring_million_samples_does_not_shift() {
        let core = StatsCore::default();
        let t0 = Instant::now();
        for i in 1..=1_000_000u32 {
            core.record_wait(Duration::from_micros(1 + u64::from(i % 100)));
        }
        let elapsed = t0.elapsed();
        assert!(
            elapsed < Duration::from_secs(3),
            "1M ring samples took {elapsed:?} — appends must not shift"
        );
        let stats = core.snapshot();
        assert!(
            (1..=100).contains(&stats.p95_wait_us),
            "p95 over a uniform 1..=100 µs window: {}",
            stats.p95_wait_us
        );
        assert_eq!(stats.max_wait_us, 100);
    }
}
