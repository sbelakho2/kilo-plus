//! Model-checking property tests over the REAL scheduler API (audit 79-80).
//!
//! Two deterministic trace drivers iterate thousands of seeded traces built
//! from a tiny inline LCG (no rand/proptest: workspace deps are frozen):
//!
//! 1. `seq_driver` — sequential traces over the full public surface:
//!    submit / duplicate-submit / submit-after-terminal / cancel / tick
//!    (mock clock; deadline expiry) / run-to-completion, with an in-test
//!    mirror of the expected per-op state machine asserted equal to the
//!    scheduler after EVERY step and at the end.
//! 2. `storm_driver` — adversarial traces: cancellation storms over running
//!    and pending ops in a live DAG, mid-run cancellation, crash mid-flight
//!    with journal-based recovery, and epoch-scoped op ids.
//!
//! Invariants (asserted every step / at the end):
//!   (a) terminal is exactly-once: a terminal op rejects complete/cancel/
//!       fail with a typed error (`try_cancel` → `Conflict`, `execute` →
//!       `Conflict`, `mark` refuses a second terminal write) — no silent
//!       state flip, no duplicate terminal event;
//!   (b) cancellation only kills running-or-pending ops and never
//!       resurrects a terminal op;
//!   (c) duplicate submit of a live op id → `Conflict` and the FIRST
//!       registration's metadata (class/deps/outcome) is unchanged
//!       (identical re-registration is idempotent and also unchanged);
//!   (d) an op with unmet dependencies never runs: every runnable snapshots
//!       its dependencies' states at run start and every snapshot is
//!       terminal;
//!   (e) exactly-once replay: every op that reached a terminal state has
//!       exactly ONE terminal journal record, and replaying the journal
//!       reconstructs the identical final state. The scheduler itself does
//!       not journal (it is an in-memory DAG executor), so the driver
//!       maintains the journal the RUNTIME's durable layer would write (the
//!       store's per-session event journal is the production oracle; the
//!       scheduler crate cannot depend on the store, so the mirror journal
//!       stands in and is documented as such); crash recovery re-drives
//!       ONLY ops without a terminal journal record.
//!
//! Any divergence between the mirror and the real scheduler is a scheduler
//! bug and fails with the trace seed so it can be replayed exactly; the
//! regression is then named for the seed.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};

use faktor_core::cancellation::CancellationToken;
use faktor_core::error::{Error, ErrorKind};
use faktor_core::id::{OpId, SessionId};
use faktor_core::op::{OpMeta, RecoveryStrategy};
use faktor_core::resource::ResourceClass;
use faktor_core::retry::RetryPolicy;
use faktor_core::time::{Clock, Deadline, TestClock};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use crate::{DependencyPolicy, OwnershipSet, ResourceRequest, ScheduledOp, Scheduler, TaskStatus};

// ---------------------------------------------------------------------------
// Deterministic LCG + trace context
// ---------------------------------------------------------------------------

/// Tiny seeded LCG (Numerical-Recipes constants). Deterministic across
/// platforms and runs: u64 wrapping arithmetic only.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1) | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // High bits of a plain LCG are the least correlated.
        (self.0 >> 33) as u32 as u64
    }

    /// Uniform in 0..n (n > 0).
    fn below(&mut self, n: u64) -> u64 {
        debug_assert!(n > 0);
        self.next() % n
    }

    /// True with probability percent/100.
    fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }
}

#[derive(Clone, Copy)]
struct Ctx {
    seed: u64,
    trace: u64,
    step: u32,
}

impl Ctx {
    fn msg(self, what: impl std::fmt::Display) -> String {
        format!(
            "{what} (seed={:#x} trace={} step={})",
            self.seed, self.trace, self.step
        )
    }
}

macro_rules! mc_assert {
    ($cond:expr, $ctx:expr, $what:expr) => {
        if !($cond) {
            panic!("{}", $ctx.msg($what));
        }
    };
}

// ---------------------------------------------------------------------------
// Payload flavors
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Flavor {
    /// Runnable returns Ok → Done.
    Ok,
    /// Runnable returns a hard (non-retryable) error → Failed.
    HardErr,
    /// Runnable panics → Failed via the scheduler's panic wrapper.
    Panic,
    /// The op carries a near deadline; when the clock passes it before the
    /// op runs, the scheduler fails it with Timeout BEFORE the runnable
    /// runs (the runnable never executes). Otherwise behaves like `Ok`.
    Deadline,
}

/// The registration metadata the scheduler actually compares on re-submit
/// (operation/session identity, resource class, reads/writes, deps — the
/// runnable closure and envelope fields like deadline/retry are excluded by
/// `same_registration`). Deps are sorted+deduped so payloads are
/// deterministic.
#[derive(Clone, PartialEq, Eq, Debug)]
struct Spec {
    class: ResourceClass,
    deps: Vec<u64>,
}

#[derive(Clone, Debug)]
struct OpModel {
    spec: Spec,
    flavor: Flavor,
    deadline_at: i64,
    status: TaskStatus,
    /// Executions the mirror has observed.
    runs: u64,
    /// Ops registered on this one through a Success edge while it was live.
    dependents: Vec<u64>,
    /// Terminal exactly-once: set at most once.
    terminal_event: Option<TaskStatus>,
}

fn terminal_status(s: TaskStatus) -> bool {
    matches!(
        s,
        TaskStatus::Done | TaskStatus::Failed | TaskStatus::Cancelled | TaskStatus::Blocked
    )
}

fn dep_satisfied(dep_status: TaskStatus) -> bool {
    dep_status == TaskStatus::Done
}

fn dep_dead(dep_status: TaskStatus) -> bool {
    matches!(
        dep_status,
        TaskStatus::Failed | TaskStatus::Cancelled | TaskStatus::Blocked
    )
}

/// One terminal record in the durable journal the trace maintains (the
/// production runtime writes the same facts into the store's event journal).
type Journal = Vec<(u64, TaskStatus)>;

#[derive(Clone)]
struct Model {
    ops: HashMap<u64, OpModel>,
    journal: Journal,
}

impl Model {
    fn new() -> Self {
        Self {
            ops: HashMap::new(),
            journal: Vec::new(),
        }
    }

    /// Record a terminal transition. Exactly-once: every op has at most one
    /// terminal journal record; a second one is a hard model failure.
    fn terminalize(&mut self, id: u64, status: TaskStatus, ctx: Ctx) {
        mc_assert!(
            terminal_status(status),
            ctx,
            "mirror: non-terminal journal write"
        );
        let op = self
            .ops
            .get_mut(&id)
            .unwrap_or_else(|| panic!("{}", ctx.msg("mirror: terminalize of unknown op")));
        mc_assert!(
            op.terminal_event.is_none(),
            ctx,
            "mirror: duplicate terminal event"
        );
        mc_assert!(
            !terminal_status(op.status),
            ctx,
            "mirror: terminal op changed state again"
        );
        op.terminal_event = Some(status);
        op.status = status;
        self.journal.push((id, status));
        // Blocking cascade: a NON-Done terminal op blocks its Success
        // dependents transitively (a Done upstream satisfies the edge — its
        // dependents just become runnable). Mirror of `terminalize`.
        if status != TaskStatus::Done {
            let dependents = std::mem::take(&mut op.dependents);
            for d in dependents {
                if self.ops.get(&d).map(|o| o.status) == Some(TaskStatus::Pending) {
                    self.block(d, ctx);
                }
            }
        }
    }

    /// Block `d` (Pending) and propagate through Success edges. A terminal
    /// dependent is never touched (no resurrection).
    fn block(&mut self, d: u64, _ctx: Ctx) {
        let status = self.ops.get(&d).map(|o| o.status);
        if status != Some(TaskStatus::Pending) {
            return;
        }
        let dependents = {
            let op = self.ops.get_mut(&d).unwrap();
            op.terminal_event = Some(TaskStatus::Blocked);
            op.status = TaskStatus::Blocked;
            self.journal.push((d, TaskStatus::Blocked));
            std::mem::take(&mut op.dependents)
        };
        for dd in dependents {
            if self.ops.get(&dd).map(|o| o.status) == Some(TaskStatus::Pending) {
                self.block(dd, _ctx);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Mirror-side transitions (replicate the scheduler's own semantics)
// ---------------------------------------------------------------------------

#[derive(PartialEq, Eq, Debug)]
enum SubmitOutcome {
    Fresh,
    Idempotent,
    Conflict,
}

#[derive(PartialEq, Eq, Debug)]
enum CancelOutcome {
    NotFound,
    AlreadyTerminal(TaskStatus),
    FlippedCancelled,
    TokenFired,
}

fn mirror_submit(
    m: &mut Model,
    id: u64,
    spec: &Spec,
    flavor: Flavor,
    deadline_at: i64,
) -> SubmitOutcome {
    if m.ops.contains_key(&id) {
        if m.ops[&id].spec == *spec {
            return SubmitOutcome::Idempotent;
        }
        return SubmitOutcome::Conflict;
    }
    // Fresh registration: replicate try_submit's dep bookkeeping.
    let mut dependents_push = Vec::new();
    let mut blocked = false;
    for dep in &spec.deps {
        let dep_status = m.ops.get(dep).map(|o| o.status);
        match dep_status {
            Some(s) if dep_satisfied(s) => {}
            Some(s) if dep_dead(s) => blocked = true,
            _ => dependents_push.push(*dep),
        }
    }
    let status = if blocked {
        TaskStatus::Blocked
    } else {
        TaskStatus::Pending
    };
    let mut op = OpModel {
        spec: spec.clone(),
        flavor,
        deadline_at,
        status,
        runs: 0,
        dependents: Vec::new(),
        terminal_event: None,
    };
    // An op that can never run is terminal from its registration moment.
    if status == TaskStatus::Blocked {
        op.terminal_event = Some(TaskStatus::Blocked);
        m.journal.push((id, TaskStatus::Blocked));
    }
    for dep in dependents_push {
        m.ops.get_mut(&dep).unwrap().dependents.push(id);
    }
    m.ops.insert(id, op);
    SubmitOutcome::Fresh
}

fn mirror_cancel(m: &mut Model, id: u64, ctx: Ctx) -> CancelOutcome {
    let status = m.ops.get(&id).map(|o| o.status);
    let Some(status) = status else {
        return CancelOutcome::NotFound;
    };
    if terminal_status(status) {
        return CancelOutcome::AlreadyTerminal(status);
    }
    let running = status == TaskStatus::Running;
    m.terminalize(id, TaskStatus::Cancelled, ctx);
    if running {
        CancelOutcome::TokenFired
    } else {
        CancelOutcome::FlippedCancelled
    }
}

/// One run-to-completion in the mirror: every pending op whose dependencies
/// are all Done executes (in deterministic sorted order) until no pending op
/// remains. Outcomes come from the scripted flavor and the mock clock.
fn mirror_run(m: &mut Model, now: i64, ctx: Ctx) {
    loop {
        let mut runnable: Vec<u64> = m
            .ops
            .iter()
            .filter(|(_, o)| {
                o.status == TaskStatus::Pending
                    && o.spec
                        .deps
                        .iter()
                        .all(|d| dep_satisfied(m.ops.get(d).unwrap().status))
            })
            .map(|(id, _)| *id)
            .collect();
        if runnable.is_empty() {
            return;
        }
        runnable.sort_unstable();
        for id in runnable {
            mirror_execute(m, id, now, ctx);
        }
    }
}

fn mirror_execute(m: &mut Model, id: u64, now: i64, ctx: Ctx) {
    let (flavor, deadline_at) = {
        let op = m.ops.get_mut(&id).unwrap();
        mc_assert!(
            op.status == TaskStatus::Pending,
            ctx,
            "mirror: executed an op that was not pending"
        );
        (op.flavor, op.deadline_at)
    };
    // Deadline expiry is checked by the scheduler BEFORE the runnable runs:
    // the runnable never executes; the op fails with Timeout.
    if flavor == Flavor::Deadline && now >= deadline_at {
        m.terminalize(id, TaskStatus::Failed, ctx);
        return;
    }
    m.ops.get_mut(&id).unwrap().runs += 1;
    let outcome = match flavor {
        Flavor::Ok | Flavor::Deadline => TaskStatus::Done,
        Flavor::HardErr | Flavor::Panic => TaskStatus::Failed,
    };
    m.terminalize(id, outcome, ctx);
}

// ---------------------------------------------------------------------------
// Real-side helpers
// ---------------------------------------------------------------------------

fn real_status_map(s: &Scheduler) -> HashMap<u64, TaskStatus> {
    s.statuses()
        .into_iter()
        .map(|(o, st)| (o.raw(), st))
        .collect()
}

fn assert_same_state(s: &Scheduler, m: &Model, ctx: Ctx) {
    let real = real_status_map(s);
    let expect: HashMap<u64, TaskStatus> = m.ops.iter().map(|(id, o)| (*id, o.status)).collect();
    if real != expect || real.len() != m.ops.len() {
        let mut keys: Vec<u64> = real.keys().chain(expect.keys()).copied().collect();
        keys.sort_unstable();
        keys.dedup();
        let mut detail = String::new();
        for k in keys {
            let r = real.get(&k).map(|s| format!("{s:?}"));
            let e = expect.get(&k).map(|s| format!("{s:?}"));
            let mark = if r == e { " " } else { "*" };
            detail.push_str(&format!(
                "{mark} op {k}: real={} mirror={}\n",
                r.unwrap_or_else(|| "-".into()),
                e.unwrap_or_else(|| "-".into())
            ));
        }
        panic!(
            "{}",
            ctx.msg(format!("mirror/real state divergence:\n{detail}"))
        );
    }
}

/// Journal replay oracle: every terminal op has exactly one terminal record,
/// the record's kind matches the final state, and replay reconstructs the
/// identical terminal map.
fn assert_model_replay(model: &Model, ctx: Ctx) {
    let mut seen: HashMap<u64, usize> = HashMap::new();
    for (id, _) in &model.journal {
        *seen.entry(*id).or_insert(0) += 1;
    }
    for (id, op) in &model.ops {
        if let Some(ev) = op.terminal_event {
            mc_assert!(
                seen.get(id) == Some(&1),
                ctx,
                "terminal op must have exactly one journal record"
            );
            mc_assert!(ev == op.status, ctx, "journal kind != final status");
        }
    }
    for (id, count) in &seen {
        mc_assert!(
            model.ops[id].terminal_event.is_some(),
            ctx,
            "journal holds an op that is not terminal"
        );
        mc_assert!(*count == 1, ctx, "duplicate terminal journal record");
    }
}

// ---------------------------------------------------------------------------
// Payload builders
// ---------------------------------------------------------------------------

fn empty_ownership() -> OwnershipSet {
    OwnershipSet::new([] as [String; 0])
}

fn classes() -> &'static [ResourceClass] {
    &ResourceClass::ALL[..]
}

/// Sequential (non-blocking) payload: a scripted immediate outcome.
fn seq_payload(
    id: u64,
    session: SessionId,
    spec: &Spec,
    flavor: Flavor,
    deadline_at: i64,
    counters: Arc<Vec<AtomicU64>>,
) -> ScheduledOp {
    let deadline = if flavor == Flavor::Deadline {
        Deadline::at(deadline_at)
    } else {
        Deadline::at(i64::MAX / 2)
    };
    let deps = spec
        .deps
        .iter()
        .map(|d| (OpId::new(*d), DependencyPolicy::Success))
        .collect();
    let class = spec.class;
    ScheduledOp {
        meta: OpMeta::new(
            OpId::new(id),
            session,
            deadline,
            RetryPolicy::default(),
            CancellationToken::new(),
            RecoveryStrategy::None,
            0,
        ),
        resources: ResourceRequest { class },
        reads: empty_ownership(),
        writes: empty_ownership(),
        dependencies: deps,
        run: Arc::new(move || {
            let counters = counters.clone();
            Box::pin(async move {
                counters[id as usize].fetch_add(1, AtomicOrdering::SeqCst);
                match flavor {
                    Flavor::Ok | Flavor::Deadline => Ok(()),
                    Flavor::HardErr => Err(Error::internal("scripted hard failure")),
                    Flavor::Panic => panic!("modelcheck scripted panic"),
                }
            })
        }),
    }
}

/// Blocking payload for the storm driver. The runnable observes the SAME
/// cancellation token that travels in the registration (both come from the
/// one payload), so `try_cancel` firing it resolves the runnable's select.
/// The execution counter bumps BEFORE the started signal so an observed
/// start always implies a counted execution.
fn storm_payload(
    id: u64,
    session: SessionId,
    sched: Scheduler,
    spec: Spec,
    counters: Arc<Vec<AtomicU64>>,
    started_tx: UnboundedSender<(u64, Vec<(u64, TaskStatus)>)>,
    rel_rx: UnboundedReceiver<()>,
) -> ScheduledOp {
    let token = CancellationToken::new();
    let deps: Vec<(OpId, DependencyPolicy)> = spec
        .deps
        .iter()
        .map(|d| (OpId::new(*d), DependencyPolicy::Success))
        .collect();
    let dep_ids = spec.deps.clone();
    let class = spec.class;
    let rel_rx = Arc::new(Mutex::new(Some(rel_rx)));
    ScheduledOp {
        meta: OpMeta::new(
            OpId::new(id),
            session,
            Deadline::at(i64::MAX / 2),
            RetryPolicy::default(),
            token.clone(),
            RecoveryStrategy::None,
            0,
        ),
        resources: ResourceRequest { class },
        reads: empty_ownership(),
        writes: empty_ownership(),
        dependencies: deps,
        run: Arc::new(move || {
            let sched = sched.clone();
            let counters = counters.clone();
            let started_tx = started_tx.clone();
            let dep_ids = dep_ids.clone();
            let token = token.clone();
            let rel_rx = rel_rx.clone();
            Box::pin(async move {
                counters[id as usize].fetch_add(1, AtomicOrdering::SeqCst);
                let snap = dep_ids
                    .iter()
                    .map(|d| {
                        let st = sched.status(OpId::new(*d)).expect("dependency registered");
                        (*d, st)
                    })
                    .collect();
                let _ = started_tx.send((id, snap));
                let mut rx = rel_rx.lock().unwrap().take().expect("runnable ran twice");
                tokio::select! {
                    _ = rx.recv() => Ok(()),
                    _ = token.cancelled() => Err(Error::cancelled()),
                }
            })
        }),
    }
}

// ---------------------------------------------------------------------------
// Driver 1: sequential model check
// ---------------------------------------------------------------------------

const SEQ_TRACES: u64 = 5000;
const MAX_OPS: u64 = 12;
const UNIVERSE: u64 = MAX_OPS + 3;

fn seq_driver(seed: u64) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    rt.block_on(seq_driver_async(seed));
}

async fn seq_driver_async(seed: u64) {
    for trace in 0..SEQ_TRACES {
        let dbg_trace = std::env::var("MC_TRACE")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            == Some(trace);
        let session = SessionId::new(1 + (seed.wrapping_mul(31) + trace) % 1_000_000);
        let clock = Arc::new(TestClock::new(1_000_000));
        let sched = Scheduler::new(session, clock.clone());
        let mut model = Model::new();
        let mut lcg = Lcg::new(seed ^ trace.wrapping_mul(0x9E37_79B9));
        let counters = Arc::new(
            (0..=UNIVERSE)
                .map(|_| AtomicU64::new(0))
                .collect::<Vec<_>>(),
        );

        let mut next_id = 1u64;
        let steps = 24 + lcg.below(10);
        let mut step = 0u32;
        while u64::from(step) < steps {
            let ctx = Ctx { seed, trace, step };
            let pick = lcg.below(100);
            if dbg_trace {
                eprintln!(
                    "-- trace {trace} step {step} pick={pick} clock={}",
                    clock.now_ms()
                );
            }
            match pick {
                // --- submit a fresh op -----------------------------------
                0..45 => {
                    if model.ops.len() as u64 >= MAX_OPS {
                        step += 1;
                        continue;
                    }
                    let id = next_id;
                    next_id += 1;
                    let class = classes()[lcg.below(classes().len() as u64) as usize];
                    // Deps: up to 3 previously registered ids (acyclic by
                    // construction). Deps may be terminal or still pending.
                    let mut dep_pool: Vec<u64> = model.ops.keys().copied().collect();
                    dep_pool.sort_unstable();
                    let mut deps = Vec::new();
                    let n_deps = lcg.below(4);
                    for _ in 0..n_deps {
                        if dep_pool.is_empty() {
                            break;
                        }
                        let di = lcg.below(dep_pool.len() as u64) as usize;
                        let d = dep_pool.swap_remove(di);
                        if !deps.contains(&d) {
                            deps.push(d);
                        }
                    }
                    deps.sort_unstable();
                    let spec = build_spec(class, &deps);
                    let flavor = match lcg.below(100) {
                        0..60 => Flavor::Ok,
                        60..82 => Flavor::HardErr,
                        82..92 => Flavor::Panic,
                        _ => Flavor::Deadline,
                    };
                    let deadline_at = clock.now_ms() + 200 + lcg.below(4000) as i64;
                    let expected = mirror_submit(&mut model, id, &spec, flavor, deadline_at);
                    mc_assert!(
                        matches!(expected, SubmitOutcome::Fresh),
                        ctx,
                        "fresh submit not fresh in mirror"
                    );
                    let payload =
                        seq_payload(id, session, &spec, flavor, deadline_at, counters.clone());
                    let real = sched.try_submit(payload);
                    mc_assert!(
                        real.is_ok(),
                        ctx,
                        "fresh submit rejected by the real scheduler"
                    );
                }
                // --- duplicate submit: identical payload -----------------
                45..55 => {
                    let Some(id) = pick_registered(&mut lcg, &model, false) else {
                        step += 1;
                        continue;
                    };
                    let spec = model.ops[&id].spec.clone();
                    let flavor = model.ops[&id].flavor;
                    let deadline_at = model.ops[&id].deadline_at;
                    let expected = mirror_submit(&mut model, id, &spec, flavor, deadline_at);
                    mc_assert!(
                        matches!(expected, SubmitOutcome::Idempotent),
                        ctx,
                        "duplicate of an existing op must be idempotent in mirror"
                    );
                    let payload =
                        seq_payload(id, session, &spec, flavor, deadline_at, counters.clone());
                    let real = sched.try_submit(payload);
                    mc_assert!(
                        real.is_ok(),
                        ctx,
                        "identical re-registration must be idempotent in the real scheduler"
                    );
                }
                // --- duplicate submit: conflicting payload ---------------
                55..64 => {
                    let Some(id) = pick_registered(&mut lcg, &model, false) else {
                        step += 1;
                        continue;
                    };
                    let orig = model.ops[&id].spec.clone();
                    let spec = conflicting_spec(&mut lcg, &orig, &model);
                    let expected = mirror_submit(&mut model, id, &spec, Flavor::Ok, i64::MAX / 2);
                    mc_assert!(
                        matches!(expected, SubmitOutcome::Conflict),
                        ctx,
                        "conflicting payload must conflict in mirror"
                    );
                    let payload = seq_payload(
                        id,
                        session,
                        &spec,
                        Flavor::Ok,
                        i64::MAX / 2,
                        counters.clone(),
                    );
                    let real = sched.try_submit(payload);
                    mc_assert!(
                        real.is_err() && real.unwrap_err().kind == ErrorKind::Conflict,
                        ctx,
                        "conflicting re-submit must be rejected with Conflict"
                    );
                    // (c) the first registration's metadata is unchanged.
                    mc_assert!(
                        model.ops[&id].spec == orig,
                        ctx,
                        "mirror lost the first registration"
                    );
                }
                // --- submit-after-terminal -------------------------------
                64..76 => {
                    let Some(id) = pick_registered(&mut lcg, &model, true) else {
                        step += 1;
                        continue;
                    };
                    if !terminal_status(model.ops[&id].status) {
                        step += 1;
                        continue;
                    }
                    let orig = model.ops[&id].spec.clone();
                    let flavor = model.ops[&id].flavor;
                    let deadline_at = model.ops[&id].deadline_at;
                    // Half the time: identical payload (idempotent — the
                    // terminal state is untouched, never re-run). Half:
                    // conflicting payload → Conflict.
                    let (spec, expect_conflict) = if lcg.chance(50) {
                        (orig.clone(), false)
                    } else {
                        (conflicting_spec(&mut lcg, &orig, &model), true)
                    };
                    let before = model.ops[&id].status;
                    let expected = mirror_submit(&mut model, id, &spec, flavor, deadline_at);
                    let payload =
                        seq_payload(id, session, &spec, flavor, deadline_at, counters.clone());
                    let real = sched.try_submit(payload);
                    if expect_conflict {
                        mc_assert!(
                            matches!(expected, SubmitOutcome::Conflict)
                                && real.is_err()
                                && real.unwrap_err().kind == ErrorKind::Conflict,
                            ctx,
                            "conflicting submit-after-terminal must Conflict"
                        );
                    } else {
                        mc_assert!(
                            matches!(expected, SubmitOutcome::Idempotent) && real.is_ok(),
                            ctx,
                            "identical submit-after-terminal must be idempotent"
                        );
                    }
                    mc_assert!(
                        model.ops[&id].status == before,
                        ctx,
                        "submit-after-terminal changed the terminal state"
                    );
                    mc_assert!(
                        model.ops[&id].terminal_event.is_some(),
                        ctx,
                        "terminal op lost its journal record"
                    );
                }
                // --- cancel ----------------------------------------------
                76..88 => {
                    let id = pick_cancel_target(&mut lcg, &model);
                    let before_status = model.ops.get(&id).map(|o| o.status);
                    let expected = mirror_cancel(&mut model, id, ctx);
                    let real = sched.try_cancel(OpId::new(id));
                    match expected {
                        CancelOutcome::NotFound => {
                            mc_assert!(
                                real.is_err() && real.unwrap_err().kind == ErrorKind::NotFound,
                                ctx,
                                "cancel of an unknown op must be NotFound"
                            );
                        }
                        CancelOutcome::AlreadyTerminal(st) => {
                            mc_assert!(
                                real.is_err() && real.unwrap_err().kind == ErrorKind::Conflict,
                                ctx,
                                "cancel of a terminal op must be a typed Conflict"
                            );
                            mc_assert!(
                                before_status == Some(st) && model.ops[&id].status == st,
                                ctx,
                                "cancel changed a terminal op"
                            );
                        }
                        CancelOutcome::FlippedCancelled => {
                            mc_assert!(real.is_ok(), ctx, "cancel of a pending op must succeed");
                            mc_assert!(
                                before_status == Some(TaskStatus::Pending)
                                    && model.ops[&id].status == TaskStatus::Cancelled,
                                ctx,
                                "pending cancel did not flip to Cancelled"
                            );
                        }
                        CancelOutcome::TokenFired => {
                            panic!("{}", ctx.msg("seq driver cannot cancel a running op"))
                        }
                    }
                }
                // --- tick the mock clock ---------------------------------
                88..94 => {
                    clock.advance(1 + lcg.below(4000) as i64);
                }
                // --- run the DAG to a fixed point ------------------------
                _ => {
                    let had_pending = model.ops.values().any(|o| o.status == TaskStatus::Pending);
                    if had_pending {
                        mirror_run(&mut model, clock.now_ms(), ctx);
                        let res = sched.run_to_completion().await;
                        mc_assert!(res.is_ok(), ctx, "run_to_completion failed on a valid DAG");
                    }
                }
            }
            assert_same_state(&sched, &model, ctx);
            if dbg_trace {
                let mut ids: Vec<u64> = model.ops.keys().copied().collect();
                ids.sort_unstable();
                eprintln!("trace {trace} step {step}:");
                for id in ids {
                    let op = &model.ops[&id];
                    let real = sched.status(OpId::new(id)).map(|s| format!("{s:?}"));
                    eprintln!(
                        "  op {id} flavor={:?} deps={:?} mirror={:?} real={} runs={}",
                        op.flavor,
                        op.spec.deps,
                        op.status,
                        real.unwrap_or_else(|| "-".into()),
                        op.runs
                    );
                }
            }
            step += 1;
        }

        // ---------------------------------------------------- end of trace
        let ctx = Ctx {
            seed,
            trace,
            step: u32::MAX,
        };
        let now = clock.now_ms();
        assert_same_state(&sched, &model, ctx);
        for (id, op) in &model.ops {
            let real_count = counters[*id as usize].load(AtomicOrdering::SeqCst);
            mc_assert!(
                real_count == op.runs,
                ctx,
                "execution count diverges from the mirror"
            );
            mc_assert!(
                op.runs <= 1,
                ctx,
                "an op executed more than once in one epoch"
            );
            match op.status {
                // A Done op ran its runnable exactly once.
                TaskStatus::Done => mc_assert!(op.runs == 1, ctx, "Done op never ran"),
                // HardErr/Panic ran once; a deadline-expired op fails BEFORE
                // its runnable runs (runs == 0), exactly like the scheduler.
                TaskStatus::Failed => {
                    if op.flavor == Flavor::Deadline && now >= op.deadline_at {
                        mc_assert!(op.runs == 0, ctx, "deadline-failed op executed");
                    } else {
                        mc_assert!(op.runs == 1, ctx, "Failed op never ran");
                    }
                }
                // Canceled while pending: never ran. (The seq driver cannot
                // cancel a running op.)
                TaskStatus::Cancelled => mc_assert!(op.runs == 0, ctx, "Cancelled op ran"),
                TaskStatus::Blocked => mc_assert!(op.runs == 0, ctx, "Blocked op ran"),
                TaskStatus::Pending | TaskStatus::Running => {
                    mc_assert!(op.runs == 0, ctx, "live op executed")
                }
            }
        }
        assert_model_replay(&model, ctx);
    }
}

fn pick_registered(lcg: &mut Lcg, model: &Model, terminal_only: bool) -> Option<u64> {
    let mut pool: Vec<u64> = model
        .ops
        .iter()
        .filter(|(_, o)| !terminal_only || terminal_status(o.status))
        .map(|(id, _)| *id)
        .collect();
    if pool.is_empty() {
        return None;
    }
    pool.sort_unstable();
    Some(pool[lcg.below(pool.len() as u64) as usize])
}

/// A payload that differs from `orig` in a scheduler-compared dimension.
fn conflicting_spec(lcg: &mut Lcg, orig: &Spec, model: &Model) -> Spec {
    if lcg.chance(50) {
        let mut class = orig.class;
        while class == orig.class {
            class = classes()[lcg.below(classes().len() as u64) as usize];
        }
        build_spec(class, &orig.deps)
    } else {
        let mut pool: Vec<u64> = model
            .ops
            .keys()
            .copied()
            .filter(|d| !orig.deps.contains(d))
            .collect();
        pool.sort_unstable();
        let mut deps = orig.deps.clone();
        if pool.is_empty() {
            if deps.is_empty() {
                let mut class = orig.class;
                while class == orig.class {
                    class = classes()[lcg.below(classes().len() as u64) as usize];
                }
                return build_spec(class, &deps);
            }
            deps.pop();
        } else {
            deps.push(pool[lcg.below(pool.len() as u64) as usize]);
            deps.sort_unstable();
        }
        build_spec(orig.class, &deps)
    }
}

/// Cancel target: mostly registered ops; sometimes an unregistered id
/// (NotFound coverage).
fn pick_cancel_target(lcg: &mut Lcg, model: &Model) -> u64 {
    if !model.ops.is_empty() && !lcg.chance(25) {
        let mut pool: Vec<u64> = model.ops.keys().copied().collect();
        pool.sort_unstable();
        return pool[lcg.below(pool.len() as u64) as usize];
    }
    1 + lcg.below(UNIVERSE)
}

fn build_spec(class: ResourceClass, deps: &[u64]) -> Spec {
    Spec {
        class,
        deps: deps.to_vec(),
    }
}

// ---------------------------------------------------------------------------
// Driver 2: cancellation storms, crash + journal recovery
// ---------------------------------------------------------------------------

const STORM_TRACES: u64 = 5000;

/// Deterministic multi-chain DAG specs: ops 1..k across two chains (odd ids
/// chain 1, even ids chain 2) so several ops run concurrently and mid-run
/// cancellation/crash is reachable. `Model`-class ops also exercise the
/// budget-1 deferral path.
fn storm_specs(k: u64, lcg: &mut Lcg) -> Vec<(u64, Spec)> {
    (1..=k)
        .map(|id| {
            let parent = if id <= 2 { vec![] } else { vec![id - 2] };
            let class = if lcg.chance(30) {
                ResourceClass::Model
            } else {
                ResourceClass::DiskRead
            };
            (id, build_spec(class, &parent))
        })
        .collect()
}

async fn storm_driver_async(seed: u64) {
    let stats = std::env::var("MC_STATS").is_ok();
    let (mut n_crashed, mut n_ops_total, mut n_decided_total, mut n_runs_total) =
        (0u64, 0u64, 0u64, 0u64);
    let (mut stat_p1, mut stat_t1, mut stat_r1) = (0u64, 0u64, 0u64);
    for trace in 0..STORM_TRACES {
        let session = SessionId::new(1 + (seed.wrapping_mul(17) + trace) % 1_000_000);
        let clock = Arc::new(TestClock::new(2_000_000));
        let mut lcg = Lcg::new(seed ^ trace.wrapping_mul(0x85EB_CA6B));
        let ctx = Ctx {
            seed,
            trace,
            step: 0,
        };

        let ops = storm_specs(3 + lcg.below(5), &mut lcg);
        let k = ops.len() as u64;
        n_ops_total += k;
        let counters = Arc::new((0..=k + 2).map(|_| AtomicU64::new(0)).collect::<Vec<_>>());

        // Mirror of phase A's registrations (fresh, single epoch).
        let mut model = Model::new();
        for (id, spec) in &ops {
            let expected = mirror_submit(&mut model, *id, spec, Flavor::Ok, i64::MAX / 2);
            mc_assert!(
                matches!(expected, SubmitOutcome::Fresh),
                ctx,
                "storm submit not fresh"
            );
        }
        let mut phase = PhaseDriver::new(session, clock.clone(), ops, counters.clone(), model);
        let (mut model, crashed) = phase.drive_crashable(ctx, &mut lcg).await;
        n_decided_total += phase.decided.len() as u64;
        n_runs_total += phase.started_log.len() as u64;
        stat_p1 += phase.stat_parks;
        stat_t1 += phase.stat_tops;
        stat_r1 += phase.stat_rolls;
        if crashed {
            n_crashed += 1;
        }

        // Invariant (d): every observed run start had all dependencies
        // terminal at that moment (snapshot read from the real scheduler).
        let starts = std::mem::take(&mut phase.started_log);
        for (_id, snap) in starts {
            for (_dep, dep_status) in snap {
                mc_assert!(
                    terminal_status(dep_status),
                    ctx,
                    "op ran before its dependency was terminal"
                );
            }
        }

        if !crashed {
            // Every op terminal, mirror == real, replay exactly-once.
            for (id, op) in &model.ops {
                let real_count = counters[*id as usize].load(AtomicOrdering::SeqCst);
                mc_assert!(real_count == op.runs, ctx, "storm count diverges");
            }
            assert_model_replay(&model, ctx);
            continue;
        }

        // ---- crash: recover from the journal ----------------------------
        // Re-drive ONLY ops with no terminal journal record; dependency
        // edges onto durably-terminal ops are dropped (durably satisfied).
        // Execution counts carry over: an op that was mid-flight at the
        // crash already executed once; recovery re-drives it (its effect is
        // UNKNOWN — the durable layer must verify; the scheduler only
        // re-drives, which is the recovery contract).
        let live_ids: Vec<u64> = {
            let mut v: Vec<u64> = model
                .ops
                .iter()
                .filter(|(_, o)| o.terminal_event.is_none())
                .map(|(id, _)| *id)
                .collect();
            v.sort_unstable();
            v
        };
        mc_assert!(!live_ids.is_empty(), ctx, "crash left nothing to recover");
        let durable_runs: HashMap<u64, u64> = model
            .ops
            .iter()
            .filter(|(_, o)| o.terminal_event.is_some())
            .map(|(id, o)| (*id, o.runs))
            .collect();
        let live: HashSet<u64> = live_ids.iter().copied().collect();

        // Mirror state for the recovery epoch: fresh registrations over the
        // live ops (deps restricted to live ids), counts carried over, and
        // the pre-crash journal preserved.
        let pre_crash_journal = model.journal.clone();
        let pre_crash_ops = model.ops.clone();
        model.ops = HashMap::new();
        model.journal = pre_crash_journal;
        let mut ops2 = Vec::new();
        for id in &live_ids {
            let orig = pre_crash_ops[id].clone();
            let mut spec = orig.spec.clone();
            spec.deps.retain(|d| live.contains(d));
            ops2.push((*id, spec.clone()));
            let op = OpModel {
                spec,
                flavor: orig.flavor,
                deadline_at: orig.deadline_at,
                status: TaskStatus::Pending,
                runs: orig.runs,
                dependents: Vec::new(),
                terminal_event: None,
            };
            for dep in &op.spec.deps {
                model.ops.get_mut(dep).unwrap().dependents.push(*id);
            }
            model.ops.insert(*id, op);
        }

        let mut phase2 = PhaseDriver::new(session, clock.clone(), ops2, counters.clone(), model);
        let (mut model, crashed2) = phase2.drive_plain(ctx, &mut lcg).await;
        n_decided_total += phase2.decided.len() as u64;
        n_runs_total += phase2.started_log.len() as u64;
        stat_p1 += phase2.stat_parks;
        stat_t1 += phase2.stat_tops;
        stat_r1 += phase2.stat_rolls;
        mc_assert!(!crashed2, ctx, "recovery phase crashed again");
        // Re-admit the durably-terminal ops into the mirror model (they are
        // not registered in the recovery scheduler, but they own journal
        // records and final terminal states for the replay oracle).
        for (id, op) in &pre_crash_ops {
            if op.terminal_event.is_some() {
                model.ops.insert(*id, op.clone());
            }
        }

        for (id, op) in &model.ops {
            let real_count = counters[*id as usize].load(AtomicOrdering::SeqCst);
            mc_assert!(real_count == op.runs, ctx, "post-recovery count diverges");
            if let Some(pre) = durable_runs.get(id) {
                mc_assert!(
                    real_count == *pre && op.runs == *pre,
                    ctx,
                    "durable-complete op was re-executed after the crash"
                );
            } else {
                mc_assert!(
                    real_count <= 2 && op.runs <= 2,
                    ctx,
                    "recovered op executed more than twice across the epoch"
                );
            }
        }
        assert_model_replay(&model, ctx);
    }
    if stats {
        eprintln!(
            "storm stats seed={seed:#x}: crashed={n_crashed}/5000 ops={n_ops_total} decided={n_decided_total} started={n_runs_total} parks={stat_p1} tops={stat_t1} rolls={stat_r1}"
        );
    }
}

fn storm_driver(seed: u64) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    rt.block_on(storm_driver_async(seed));
}

/// One scheduler phase over real submit/run machinery: the (id, spec) ops
/// are submitted through `try_submit`, the DAG runs as a spawned task, and
/// the driver reacts to `started` events with release / cancel / park
/// decisions. Each op receives at most one of release or cancel, so every
/// outcome is deterministic. Cancellation storms sweep all ops every other
/// iteration and assert the typed contract against each op's live state.
struct PhaseDriver {
    sched: Scheduler,
    model: Model,
    started_log: Vec<(u64, Vec<(u64, TaskStatus)>)>,
    rel_senders: HashMap<u64, UnboundedSender<()>>,
    started_rx: UnboundedReceiver<(u64, Vec<(u64, TaskStatus)>)>,
    /// Ops seen started but not yet released/cancelled.
    parked: Vec<u64>,
    /// Ops whose fate is sealed (released or cancelled).
    decided: HashSet<u64>,
    /// Diagnostics (env MC_STATS): parks / parked tops / crash rolls.
    pub stat_parks: u64,
    pub stat_tops: u64,
    pub stat_rolls: u64,
}

impl PhaseDriver {
    fn new(
        session: SessionId,
        clock: Arc<TestClock>,
        ops: Vec<(u64, Spec)>,
        counters: Arc<Vec<AtomicU64>>,
        model: Model,
    ) -> Self {
        let sched = Scheduler::new(session, clock);
        let (started_tx, started_rx) = tokio::sync::mpsc::unbounded_channel();
        let mut rel_senders = HashMap::new();
        for (id, spec) in &ops {
            let (rel_tx, rel_rx) = tokio::sync::mpsc::unbounded_channel();
            let payload = storm_payload(
                *id,
                session,
                sched.clone(),
                spec.clone(),
                counters.clone(),
                started_tx.clone(),
                rel_rx,
            );
            sched
                .try_submit(payload)
                .expect("storm submit must succeed");
            rel_senders.insert(*id, rel_tx);
        }
        assert_eq!(
            model.ops.len(),
            ops.len(),
            "mirror ops must match the phase's registrations"
        );
        for (id, _) in &ops {
            assert!(model.ops.contains_key(id), "mirror missing op {id}");
        }
        Self {
            sched,
            model,
            started_log: Vec::new(),
            rel_senders,
            started_rx,
            parked: Vec::new(),
            decided: HashSet::new(),
            stat_parks: 0,
            stat_tops: 0,
            stat_rolls: 0,
        }
    }

    /// Drive and maybe crash mid-flight. Returns (final mirror, crashed?).
    /// Drive with mid-flight crashes enabled (phase A).
    async fn drive_crashable(&mut self, ctx: Ctx, lcg: &mut Lcg) -> (Model, bool) {
        self.drive_inner(ctx, true, lcg).await
    }

    /// Drive without crashes (recovery phases must always complete).
    async fn drive_plain(&mut self, ctx: Ctx, lcg: &mut Lcg) -> (Model, bool) {
        self.drive_inner(ctx, false, lcg).await
    }

    async fn drive_inner(&mut self, ctx: Ctx, want_crash: bool, lcg: &mut Lcg) -> (Model, bool) {
        let mut crashed = false;
        let mut run_handle = tokio::spawn({
            let sched = self.sched.clone();
            async move { sched.run_to_completion().await }
        });

        let mut storm_counter = 0u32;
        let storm_every = 2u32;

        loop {
            if crashed {
                break;
            }
            // ---- act on a parked op (never block while parked) ----------
            if !self.parked.is_empty() {
                self.stat_tops += 1;
                // Crash mid-flight over a parked (running) op.
                let crash_roll = want_crash && lcg.chance(60);
                if crash_roll {
                    self.stat_rolls += 1;
                }
                if crash_roll {
                    // Crash mid-flight over a SETTLED barrier: cancel one
                    // parked (running) op, drain until mirror == real, then
                    // abort the run. Ops still running at the barrier have
                    // no terminal journal record — recovery re-drives them.
                    crashed = true;
                    let target = self.parked[0];
                    self.cancel_live(target, ctx).await;
                    self.settle(ctx).await;
                    assert_same_state(&self.sched, &self.model, ctx);
                    run_handle.abort();
                    let _ = run_handle.await;
                    // Drain residual started signals the aborted tasks had
                    // already sent (their executions happened).
                    self.drain_started();
                    for _ in 0..4 {
                        tokio::task::yield_now().await;
                    }
                    break;
                }
                if lcg.chance(70) {
                    let target = self.parked[0];
                    if lcg.chance(55) {
                        self.release(target, ctx).await;
                    } else {
                        self.cancel_live(target, ctx).await;
                    }
                    continue;
                }
            }

            // ---- cancellation storm over every op -----------------------
            storm_counter += 1;
            if storm_counter.is_multiple_of(storm_every) {
                self.storm_sweep(ctx).await;
            }

            // ---- await the next run start (or completion) ---------------
            let started = tokio::select! {
                started = self.started_rx.recv() => started,
                _ = &mut run_handle => None,
            };
            let Some((id, snap)) = started else {
                break; // run completed
            };
            let decided = self.apply_started(id, snap);
            if decided {
                continue; // storm already sealed this op's fate
            }
            match lcg.below(100) {
                0..60 => self.release(id, ctx).await,
                60..80 => self.cancel_live(id, ctx).await,
                _ => {
                    self.parked.push(id);
                    self.stat_parks += 1;
                }
            }
        }

        // ---- the run completed (or was aborted): settle and verify ------
        if !crashed {
            self.drain_started();
            self.settle(ctx).await;
            assert_same_state(&self.sched, &self.model, ctx);
            for (id, op) in &self.model.ops {
                mc_assert!(
                    terminal_status(op.status),
                    ctx,
                    "storm run ended with a live op"
                );
                mc_assert!(
                    self.sched.status(OpId::new(*id)) == Some(op.status),
                    ctx,
                    "storm end status mismatch"
                );
            }
            mc_assert!(
                self.parked.is_empty(),
                ctx,
                "run completed while ops were parked"
            );
        }
        (self.model.clone(), crashed)
    }

    /// Apply one observed run start to the mirror: a Pending op becomes
    /// Running; an op the storm already cancelled stays terminal. Either
    /// way the execution is counted (the real runnable bumped the counter
    /// before signalling). Returns whether the op's fate was already
    /// decided.
    fn apply_started(&mut self, id: u64, snap: Vec<(u64, TaskStatus)>) -> bool {
        let op = self.model.ops.get_mut(&id).unwrap();
        match op.status {
            TaskStatus::Pending => {
                op.status = TaskStatus::Running;
                op.runs += 1;
            }
            _ => {
                op.runs += 1;
            }
        }
        self.started_log.push((id, snap));
        self.decided.contains(&id)
    }

    /// Drain every queued started signal (mirror bookkeeping only — no fate
    /// decisions: the run is over or about to be aborted, and undecided
    /// running ops are exactly the recovery set).
    fn drain_started(&mut self) {
        while let Ok((id, snap)) = self.started_rx.try_recv() {
            self.apply_started(id, snap);
        }
    }

    /// Sweep-cancel every undecided LIVE op (the "cancel everything every
    /// other step" storm). Mirror-terminal undecided ops (blocked
    /// dependents) are asserted at settle points, where the real state is
    /// equal to the mirror.
    async fn storm_sweep(&mut self, ctx: Ctx) {
        let mut ids_sorted: Vec<u64> = self.rel_senders.keys().copied().collect();
        ids_sorted.sort_unstable();
        let mut cancelled_running = false;
        for id in ids_sorted {
            if self.decided.contains(&id) {
                continue;
            }
            if terminal_status(self.model.ops[&id].status) {
                continue; // blocked dependents: asserted at settle points
            }
            let mirror_was_running = self.model.ops[&id].status == TaskStatus::Running;
            let real_before = self.sched.status(OpId::new(id));
            let real_was_running = real_before == Some(TaskStatus::Running);
            let real_result = self.sched.try_cancel(OpId::new(id));
            mc_assert!(
                real_result.is_ok(),
                ctx,
                "storm: cancel of a live op must succeed"
            );
            let outcome = mirror_cancel(&mut self.model, id, ctx);
            mc_assert!(
                matches!(
                    outcome,
                    CancelOutcome::FlippedCancelled | CancelOutcome::TokenFired
                ),
                ctx,
                "storm: mirror cancel outcome mismatch"
            );
            if mirror_was_running || real_was_running {
                cancelled_running = true;
            }
            self.decided.insert(id);
            self.parked.retain(|p| *p != id);
        }
        if cancelled_running {
            self.settle(ctx).await;
        }
    }

    /// Release a running op: the runnable returns Ok; the op completes Done
    /// (unless it was already cancelled — releases never override a
    /// cancellation).
    async fn release(&mut self, id: u64, ctx: Ctx) {
        if let Some(tx) = self.rel_senders.get(&id) {
            let _ = tx.send(());
        }
        if self.model.ops[&id].status == TaskStatus::Running {
            self.model.terminalize(id, TaskStatus::Done, ctx);
        }
        self.decided.insert(id);
        self.parked.retain(|p| *p != id);
    }

    /// Cancel a live (Pending or Running) op, asserting the real typed
    /// contract.
    async fn cancel_live(&mut self, id: u64, ctx: Ctx) {
        let real_before = self.sched.status(OpId::new(id));
        let mirror_before = self.model.ops[&id].status;
        mc_assert!(
            !terminal_status(mirror_before),
            ctx,
            "cancel_live on a mirror-terminal op"
        );
        let real_result = self.sched.try_cancel(OpId::new(id));
        match real_before {
            Some(st) if terminal_status(st) => {
                mc_assert!(
                    real_result.is_err()
                        && real_result.as_ref().unwrap_err().kind == ErrorKind::Conflict,
                    ctx,
                    "cancel of a terminal op must be rejected"
                );
                mc_assert!(
                    terminal_status(mirror_before),
                    ctx,
                    "real terminal op is not terminal in the mirror"
                );
                return;
            }
            Some(_) => {
                mc_assert!(real_result.is_ok(), ctx, "cancel of a live op failed");
            }
            None => unreachable!("registered op vanished"),
        }
        let expected = mirror_cancel(&mut self.model, id, ctx);
        mc_assert!(
            matches!(
                expected,
                CancelOutcome::FlippedCancelled | CancelOutcome::TokenFired
            ),
            ctx,
            "mirror cancel outcome mismatch"
        );
        self.decided.insert(id);
        self.parked.retain(|p| *p != id);
    }

    /// Yield (and drain started signals) until the real scheduler state
    /// equals the mirror — async flips settle deterministically, no timers —
    /// then assert invariant (b) across the terminal set: cancellation of
    /// any terminal op is a typed Conflict and never changes the state.
    async fn settle(&mut self, ctx: Ctx) {
        let mut spins = 0u32;
        loop {
            self.drain_started();
            if real_status_map(&self.sched) == mirror_map(&self.model) {
                break;
            }
            tokio::task::yield_now().await;
            spins += 1;
            mc_assert!(spins < 1_000_000, ctx, "mirror/real never settled");
        }
        for (id, op) in &self.model.ops {
            if !terminal_status(op.status) {
                continue;
            }
            let real_result = self.sched.try_cancel(OpId::new(*id));
            mc_assert!(
                real_result.is_err()
                    && real_result.as_ref().unwrap_err().kind == ErrorKind::Conflict,
                ctx,
                "cancel of a terminal op must be a typed Conflict"
            );
            mc_assert!(
                self.sched.status(OpId::new(*id)) == Some(op.status),
                ctx,
                "rejected cancel changed a terminal op"
            );
        }
    }
}

fn mirror_map(m: &Model) -> HashMap<u64, TaskStatus> {
    m.ops.iter().map(|(id, o)| (*id, o.status)).collect()
}

// ---------------------------------------------------------------------------
// Small deterministic unit checks + test entry points
// ---------------------------------------------------------------------------

#[test]
fn lcg_is_deterministic() {
    let mut a = Lcg::new(42);
    let mut b = Lcg::new(42);
    for _ in 0..100 {
        assert_eq!(a.next(), b.next());
    }
    let mut c = Lcg::new(43);
    assert_ne!(a.next(), c.next());
}

#[test]
fn typed_cancel_matrix() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    rt.block_on(async {
        let session = SessionId::new(7);
        let clock = Arc::new(TestClock::new(1000));
        let s = Scheduler::new(session, clock.clone());
        let counters = Arc::new((0..=6u64).map(|_| AtomicU64::new(0)).collect::<Vec<_>>());
        let spec = build_spec(ResourceClass::Cpu, &[]);
        let payload = seq_payload(
            1,
            session,
            &spec,
            Flavor::Ok,
            i64::MAX / 2,
            counters.clone(),
        );
        s.try_submit(payload).unwrap();
        // Unknown op → NotFound.
        let err = s.try_cancel(OpId::new(99)).unwrap_err();
        assert_eq!(err.kind, ErrorKind::NotFound);
        // Pending → flips to Cancelled.
        s.try_cancel(OpId::new(1)).unwrap();
        assert_eq!(s.status(OpId::new(1)), Some(TaskStatus::Cancelled));
        // Terminal → typed Conflict; state untouched, no resurrection.
        let err = s.try_cancel(OpId::new(1)).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Conflict);
        assert_eq!(s.status(OpId::new(1)), Some(TaskStatus::Cancelled));
        // submit-after-terminal with an identical payload is idempotent and
        // does not re-run or flip the state.
        let payload = seq_payload(
            1,
            session,
            &spec,
            Flavor::Ok,
            i64::MAX / 2,
            counters.clone(),
        );
        s.try_submit(payload).unwrap();
        assert_eq!(s.status(OpId::new(1)), Some(TaskStatus::Cancelled));
        // A conflicting payload after terminal is a Conflict.
        let other_spec = build_spec(ResourceClass::Model, &[]);
        let other = seq_payload(
            1,
            session,
            &other_spec,
            Flavor::Ok,
            i64::MAX / 2,
            counters.clone(),
        );
        let err = s.try_submit(other).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Conflict);
        assert_eq!(s.status(OpId::new(1)), Some(TaskStatus::Cancelled));
        // execute() on the terminal op is a typed rejection (no rerun).
        let payload = seq_payload(
            1,
            session,
            &spec,
            Flavor::Ok,
            i64::MAX / 2,
            counters.clone(),
        );
        let err = s.execute(OpId::new(1), payload).await.unwrap_err();
        assert!(matches!(err, crate::ExecuteError::Err(e) if e.kind == ErrorKind::Conflict));
        assert_eq!(counters[1].load(AtomicOrdering::SeqCst), 0);
    });
}

/// Epoch-scoped ids never collide (two epochs of the same session with
/// allocator-disjoint id ranges), and the journal oracle is the ONLY layer
/// that catches a recovery bug re-submitting a durable-complete op id from
/// an old epoch — the in-memory scheduler is fresh and cannot remember the
/// previous epoch (the exactly-once boundary is documented: ids come from
/// the store's durable op_id_seq).
#[test]
fn epoch_ids_do_not_collide_journal_oracle_catches_recovery_bug() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let session = SessionId::new(9);
        let counters = Arc::new((0..=6u64).map(|_| AtomicU64::new(0)).collect::<Vec<_>>());

        // ---- epoch A: op 1 completes durably ----------------------------
        let clock = Arc::new(TestClock::new(1000));
        let sched_a = Scheduler::new(session, clock.clone());
        let spec = build_spec(ResourceClass::Cpu, &[]);
        let payload = seq_payload(
            1,
            session,
            &spec,
            Flavor::Ok,
            i64::MAX / 2,
            counters.clone(),
        );
        sched_a.try_submit(payload).unwrap();
        sched_a.run_to_completion().await.unwrap();
        assert_eq!(sched_a.status(OpId::new(1)), Some(TaskStatus::Done));
        let mut journal: Journal = vec![(1, TaskStatus::Done)];

        // ---- epoch B: fresh scheduler; ids continue (op_id_seq) ---------
        let sched_b = Scheduler::new(session, clock.clone());
        // The second epoch cannot reference epoch A's terminal state as
        // running or pending: the scheduler instance is fresh.
        assert_eq!(sched_b.status(OpId::new(1)), None);
        assert_eq!(
            sched_b.try_cancel(OpId::new(1)).unwrap_err().kind,
            ErrorKind::NotFound
        );
        let payload = seq_payload(
            2,
            session,
            &build_spec(ResourceClass::Cpu, &[]),
            Flavor::Ok,
            i64::MAX / 2,
            counters.clone(),
        );
        sched_b.try_submit(payload).unwrap();
        sched_b.run_to_completion().await.unwrap();
        assert_eq!(counters[2].load(AtomicOrdering::SeqCst), 1);
        journal.push((2, TaskStatus::Done));

        // ---- adversarial: a recovery bug re-submits epoch A's id ---------
        // The fresh scheduler ACCEPTS it (it has no epoch memory) and
        // re-executes the op: a second terminal event that ONLY the durable
        // journal oracle can catch. This documents that exactly-once for
        // epoch-scoped ids lives in the durable op_id_seq/journal layer,
        // never in the in-memory scheduler.
        let payload = seq_payload(
            1,
            session,
            &spec,
            Flavor::Ok,
            i64::MAX / 2,
            counters.clone(),
        );
        sched_b.try_submit(payload).unwrap(); // accepted: no epoch memory
        sched_b.run_to_completion().await.unwrap();
        assert_eq!(
            counters[1].load(AtomicOrdering::SeqCst),
            2,
            "the scheduler re-ran the old-epoch op"
        );
        journal.push((1, TaskStatus::Done));
        let dupes = journal.iter().filter(|(id, _)| *id == 1).count();
        assert_eq!(
            dupes, 2,
            "the journal oracle flags the duplicate terminal record"
        );
    });
}

/// Driver 1 (sequential). Multiple seeds broaden coverage cheaply; every
/// seed is deterministic and replayable.
#[test]
fn modelcheck_seed_0x5eed() {
    seq_driver(0x5EED_0000_0000_0001);
}

#[test]
fn modelcheck_seed_0xbeef() {
    seq_driver(0xBEEF_0000_0000_0002);
}

#[test]
fn modelcheck_seed_0xcafe() {
    seq_driver(0xCAFE_0000_0000_0003);
}

/// Driver 2: cancellation storms, mid-run cancel, crash + recovery.
#[test]
fn storm_driver_seed_0x5eed() {
    storm_driver(0x5EED_0000_0000_1001);
}

#[test]
fn storm_driver_seed_0xbeef() {
    storm_driver(0xBEEF_0000_0000_1002);
}
