//! Campaign (d): scheduler run crashed mid-DAG vs the reference terminal
//! set (P0-76, 300 seeds).
//!
//! A six-op `Success` chain (`op i` depends on `op i-1`) runs on the REAL
//! scheduler API (`try_submit` + `run_to_completion`). Each runnable
//! signals `started` and then parks until the driver releases it, so the
//! driver observes every execution start and controls every terminal
//! transition — exactly like the scheduler modelcheck storm driver. The
//! crash = the driver ABORTS the executor mid-flight and drops the
//! scheduler (a process death of the in-memory DAG executor); the runnable
//! that was parked has EXECUTED ONCE with no terminal record.
//!
//! Terminal facts are journaled into a REAL `faktor-store` typed ledger —
//! the production oracle the scheduler crate itself cannot depend on
//! (modelcheck's mirror journal documents the same stand-in role). A
//! terminal record is only written once the op is provably terminal (the
//! next op of the chain started, or the run completed). The recovery epoch
//! reopens the store, re-registers exactly the ops WITHOUT a terminal
//! record (dependencies restricted to live ops) and re-drives them.
//!
//! Boundaries: crash before the executor ever starts, or crash with op
//! `k+1` parked mid-flight after ops 1..=k are durably terminal, k = 0..5.
//! Certification per (seed, boundary): the recovered store ledger EQUALS
//! the reference ledger (every op exactly one terminal record, same order,
//! same payloads) and the recovered terminal map equals the reference
//! (all Done). Physical counts are asserted inside the run: ops with a
//! durable record executed EXACTLY once; the parked mid-flight op exactly
//! twice (its second execution is the recovery re-drive of a non-durable
//! op); every other op once — no op ever executes twice unless it had no
//! durable terminal record.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use faktor_core::cancellation::CancellationToken;
use faktor_core::error::Error;
use faktor_core::id::{OpId, SessionId};
use faktor_core::op::{OpMeta, RecoveryStrategy};
use faktor_core::resource::ResourceClass;
use faktor_core::retry::RetryPolicy;
use faktor_core::time::{Deadline, TestClock};
use faktor_scheduler::{
    DependencyPolicy, OwnershipSet, ResourceRequest, ScheduledOp, Scheduler, TaskStatus,
};
use faktor_store::Store;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use super::{check_equals, BoundarySpec, Campaign, CrashClass, Lcg, WorldState};

/// Seeds of the full [fault]-gated campaign.
pub const FULL_SEEDS: u64 = 300;
/// Seeds of the normal-mode smoke run.
pub const SMOKE_SEEDS: u64 = 3;

const N_OPS: usize = 6;

fn ops() -> Vec<u64> {
    (1..=N_OPS as u64).collect()
}

fn content_mix(seed: u64) -> Lcg {
    Lcg::new(seed ^ 0xD06_F00D)
}

// ---------------------------------------------------------------------------
// DAG runnable: bumps its physical counter, snapshots its dependencies,
// signals `started`, then parks until released (or the token cancels).
// ---------------------------------------------------------------------------

fn op_fn(
    id: u64,
    sched: Scheduler,
    deps: Vec<u64>,
    counters: Arc<Vec<AtomicU64>>,
    started_tx: UnboundedSender<(u64, Vec<(u64, TaskStatus)>)>,
    rel_rx: UnboundedReceiver<()>,
) -> ScheduledOp {
    let token = CancellationToken::new();
    let dependencies: Vec<(OpId, DependencyPolicy)> = deps
        .iter()
        .map(|d| (OpId::new(*d), DependencyPolicy::Success))
        .collect();
    let dep_ids = deps;
    let class = match id % 3 {
        0 => ResourceClass::Cpu,
        1 => ResourceClass::DiskRead,
        _ => ResourceClass::Model,
    };
    let rel_rx = Arc::new(std::sync::Mutex::new(Some(rel_rx)));
    ScheduledOp {
        meta: OpMeta::new(
            OpId::new(id),
            sched.session_id(),
            Deadline::at(i64::MAX / 2),
            RetryPolicy::default(),
            token.clone(),
            RecoveryStrategy::None,
            0,
        ),
        resources: ResourceRequest { class },
        reads: OwnershipSet::new([] as [String; 0]),
        writes: OwnershipSet::new([] as [String; 0]),
        dependencies,
        run: Arc::new(move || {
            let sched = sched.clone();
            let counters = counters.clone();
            let started_tx = started_tx.clone();
            let dep_ids = dep_ids.clone();
            let token = token.clone();
            let rel_rx = rel_rx.clone();
            Box::pin(async move {
                counters[id as usize].fetch_add(1, Ordering::SeqCst);
                let snap = dep_ids
                    .iter()
                    .map(|d| {
                        let st = sched.status(OpId::new(*d)).expect("dependency registered");
                        (*d, st)
                    })
                    .collect();
                let _ = started_tx.send((id, snap));
                let mut rx = rel_rx
                    .lock()
                    .expect("poisoned")
                    .take()
                    .expect("runnable ran twice");
                tokio::select! {
                    _ = rx.recv() => Ok(()),
                    _ = token.cancelled() => Err(Error::cancelled()),
                }
            })
        }),
    }
}

// ---------------------------------------------------------------------------
// Terminal journal over a real store (the production oracle)
// ---------------------------------------------------------------------------

fn open_store(root: &std::path::Path, seed: u64) -> (Store, SessionId) {
    let store = Store::open(root.join("store"), true).expect("store opens");
    let ws = store.create_workspace("/w").expect("workspace");
    let row = store
        .create_session(ws, &format!("dag{seed}"), "sched", "fault")
        .expect("session");
    (store, row.id)
}

/// One durable terminal fact: (op, Done). The payload is seed-derived and
/// IDENTICAL between the reference run and the recovery epoch — a fact
/// about the op, never about the physical attempt.
fn journal_terminal(store: &Store, session: SessionId, seed: u64, op: u64) {
    let mut mix = content_mix(seed);
    let payload = serde_json::json!({
        "op": op,
        "status": "done",
        "salt": mix.next_u64(),
    });
    store
        .append_ledger_entry(session, "dag_terminal", 1, payload)
        .expect("terminal record");
}

fn durable_terminal_ops(store: &Store, session: SessionId) -> Vec<u64> {
    store
        .ledger_entries(session, None, u64::MAX)
        .expect("read ledger")
        .into_iter()
        .map(|le| {
            le.payload["op"]
                .as_u64()
                .expect("terminal payload carries the op id")
        })
        .collect()
}

// ---------------------------------------------------------------------------
// One DAG phase over the real scheduler. `park`: when `Some(id)` the phase
// crashes (aborts the executor) with op `id` parked mid-flight after its
// start was observed. Journaling is deferred until an op is PROVABLY
// terminal (the next chain op started, or the run completed), so a durable
// record never lies about a still-running op.
// ---------------------------------------------------------------------------

struct PhaseOutcome {
    /// Physical execution count of every op (index = op id, 0 unused).
    counts: Vec<u64>,
    /// Scheduler statuses at phase end.
    statuses: Vec<(u64, TaskStatus)>,
    /// The parked mid-flight op of a crashed phase (`Some(0)` = crashed
    /// before the executor ever started; `None` = clean completion).
    parked: Option<u64>,
}

// Test-driver helper: every parameter is one phase-execution knob the
// campaign must inject (seed, session, clock, ops, dependency fn,
// counters, crash park point, store) — a context struct would hide the
// injection points this campaign deliberately names.
#[allow(clippy::too_many_arguments)]
async fn run_phase(
    seed: u64,
    session: SessionId,
    clock: Arc<TestClock>,
    ops: Vec<u64>,
    deps_of: &dyn Fn(u64) -> Vec<u64>,
    counters: Arc<Vec<AtomicU64>>,
    park: Option<u64>,
    store: &Store,
) -> Result<PhaseOutcome, String> {
    let sched = Scheduler::new(session, clock);
    let (started_tx, mut started_rx) = tokio::sync::mpsc::unbounded_channel();
    let mut rel_senders = std::collections::HashMap::new();
    for &id in &ops {
        let (rel_tx, rel_rx) = tokio::sync::mpsc::unbounded_channel();
        let deps = deps_of(id);
        let payload = op_fn(
            id,
            sched.clone(),
            deps,
            counters.clone(),
            started_tx.clone(),
            rel_rx,
        );
        sched
            .try_submit(payload)
            .map_err(|e| format!("submit op {id} failed: {e}"))?;
        rel_senders.insert(id, rel_tx);
    }
    // The driver must not hold a sender: recv() may then return None when
    // the executor finishes (all runnable tasks dropped their clones).
    drop(started_tx);

    let handle = tokio::spawn({
        let sched = sched.clone();
        async move { sched.run_to_completion().await }
    });

    // Journaling is deferred until an op is PROVABLY terminal (the next op
    // of the chain started — which happens only after the terminal
    // transition — or the run completed), so a durable terminal record
    // never lies about a still-running op.
    let mut pending: Vec<u64> = Vec::new();
    let mut parked_op: Option<u64> = None;
    let mut expected_next = ops[0];
    let mut released = 0u64;

    loop {
        // Executor ended before every op started: an error unless the
        // phase's ops all started already.
        let Some((id, _snap)) = started_rx.recv().await else {
            if released == ops.len() as u64 {
                break;
            }
            return Err("executor ended before every op started".into());
        };
        if id != expected_next {
            return Err(format!(
                "DAG start order violated: expected op {expected_next}, saw {id}"
            ));
        }
        expected_next = if id == *ops.last().expect("ops non-empty") {
            u64::MAX
        } else {
            id + 1
        };
        // The previously released op became provably terminal.
        for done in pending.drain(..) {
            journal_terminal(store, session, seed, done);
        }
        if park == Some(id) {
            parked_op = park;
            break;
        }
        if let Some(tx) = rel_senders.get(&id) {
            let _ = tx.send(());
            pending.push(id);
            released += 1;
        }
        if released == ops.len() as u64 {
            // Every op of this phase was released; the executor finishes
            // them (the runnables' own sender clones keep the channel
            // open, so completion is observed through the run handle).
            break;
        }
    }

    if parked_op.is_some() {
        // The crash: abort the executor mid-flight, drop the scheduler.
        handle.abort();
        let _ = handle.await;
        // The released op right before the park is terminal (the parked op
        // started after that terminal transition) — journal it.
        for done in pending.drain(..) {
            journal_terminal(store, session, seed, done);
        }
    } else {
        let result = handle
            .await
            .map_err(|e| format!("executor task failed: {e}"))?;
        result.map_err(|e| format!("run_to_completion failed: {e}"))?;
        // Every released op is terminal: the run completed.
        for done in pending.drain(..) {
            journal_terminal(store, session, seed, done);
        }
    }

    let counts = counters
        .iter()
        .map(|c| c.load(Ordering::SeqCst))
        .collect::<Vec<_>>();
    let statuses = sched
        .statuses()
        .into_iter()
        .map(|(o, st)| (o.raw(), st))
        .collect::<Vec<_>>();
    drop(sched);
    Ok(PhaseOutcome {
        counts,
        statuses,
        parked: parked_op,
    })
}

fn deps_of(id: u64) -> Vec<u64> {
    if id == 1 {
        Vec::new()
    } else {
        vec![id - 1]
    }
}

/// Recovery-epoch dependency restriction: edges onto durably-terminal
/// (Done) ops are dropped — the live sub-DAG stands alone.
fn live_deps_of(id: u64, live: &[u64]) -> Vec<u64> {
    deps_of(id)
        .into_iter()
        .filter(|d| live.contains(d))
        .collect()
}

// ---------------------------------------------------------------------------
// Full run: phase 1 (reference or crashed) + recovery epoch when crashed
// ---------------------------------------------------------------------------

struct RunOutcome {
    /// The store's tempdir: kept alive as long as the store is readable.
    _dir: tempfile::TempDir,
    store: Store,
    session: SessionId,
    phase1: PhaseOutcome,
    phase2: Option<PhaseOutcome>,
    /// Ops durably terminal BEFORE the crash (the recovery set decision;
    /// for the reference run this is every op at the end).
    durable_pre_crash: Vec<u64>,
}

fn run_dag(seed: u64, park: Option<u64>) -> Result<RunOutcome, String> {
    let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
    let root = dir.path().to_path_buf();
    let clock = Arc::new(TestClock::new(1_000_000 + (seed % 1_000_000) as i64));
    let counters = Arc::new((0..=N_OPS).map(|_| AtomicU64::new(0)).collect::<Vec<_>>());
    let all = ops();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    match park {
        None => rt.block_on(async move {
            let (store, session) = open_store(&root, seed);
            let phase1 =
                run_phase(seed, session, clock, all, &deps_of, counters, None, &store).await?;
            Ok(RunOutcome {
                _dir: dir,
                store,
                session,
                phase1,
                phase2: None,
                durable_pre_crash: ops(),
            })
        }),
        Some(p) => rt.block_on(async move {
            let (store, session) = open_store(&root, seed);
            let phase1 = run_phase(
                seed,
                session,
                clock.clone(),
                all.clone(),
                &deps_of,
                counters.clone(),
                Some(p),
                &store,
            )
            .await?;
            // Process death of the whole daemon: drop the store and reopen
            // from disk; the durable records drive the recovery set.
            drop(store);
            let (store2, session2) = open_store(&root, seed);
            let durable = durable_terminal_ops(&store2, session2);
            let live: Vec<u64> = all
                .iter()
                .copied()
                .filter(|id| !durable.contains(id))
                .collect();
            if live.is_empty() {
                return Err("a crash must leave at least the parked op to recover".into());
            }
            let live_for_deps = live.clone();
            let phase2 = run_phase(
                seed,
                session2,
                clock,
                live,
                &|id| live_deps_of(id, &live_for_deps),
                counters,
                None,
                &store2,
            )
            .await?;
            Ok(RunOutcome {
                _dir: dir,
                store: store2,
                session: session2,
                phase1,
                phase2: Some(phase2),
                durable_pre_crash: durable,
            })
        }),
    }
}

// ---------------------------------------------------------------------------
// Canonical world state: the store ledger plus the full terminal map,
// reconstructed from the durable records (reference and recovered stores
// carry the SAME facts; physical counts are asserted inside the run).
// ---------------------------------------------------------------------------

fn dump_world(store: &Store, session: SessionId) -> WorldState {
    let mut lines = Vec::new();
    for le in store
        .ledger_entries(session, None, u64::MAX)
        .expect("ledger")
    {
        lines.push(format!("le:{}:{}:{}", le.seq, le.entry_type, le.payload));
    }
    let mut ops_done = durable_terminal_ops(store, session);
    ops_done.sort_unstable();
    for op in ops_done {
        lines.push(format!("status:op{op}:Done"));
    }
    WorldState { lines }
}

fn counts_of(out: &RunOutcome) -> Vec<u64> {
    // The counters are SHARED across phases (a physical-execution counter
    // of the epoch, like the modelcheck driver's), so phase 2's snapshot
    // is already cumulative.
    match &out.phase2 {
        Some(p2) => p2.counts.clone(),
        None => out.phase1.counts.clone(),
    }
}

fn verify_phase_counts(out: &RunOutcome, boundary: &BoundarySpec) -> Result<(), String> {
    let total = counts_of(out);
    let durable_pre = &out.durable_pre_crash;
    for id in ops() {
        let ran = total[id as usize];
        if durable_pre.contains(&id) {
            if ran != 1 {
                return Err(format!(
                    "durable-complete op {id} was re-executed after the crash (ran {ran} times)"
                ));
            }
        } else if ran > 2 {
            return Err(format!(
                "recovered op {id} executed more than twice across the epoch (ran {ran})"
            ));
        }
    }
    match boundary.name {
        "dag.pre_execution" => {
            if total.iter().skip(1).any(|c| *c != 1) {
                return Err(format!(
                    "pre-execution crash must re-run every op exactly once: {total:?}"
                ));
            }
        }
        name if name.starts_with("dag.parked") => {
            let parked = boundary_index(name)? as u64 + 1;
            if durable_pre.contains(&parked) {
                return Err(format!(
                    "the parked op {parked} must NOT be durably complete pre-crash"
                ));
            }
            if total[parked as usize] != 2 {
                return Err(format!(
                    "the parked mid-flight op {parked} must execute exactly twice (its recovery re-drive): {total:?}"
                ));
            }
            if let Some(observed) = out.phase1.parked {
                if observed != parked {
                    return Err(format!("phase parked op {observed}, expected {parked}"));
                }
            }
        }
        other => return Err(format!("unknown boundary {other:?}")),
    }
    // Terminal maps: the crashed phase left the parked op Running, every
    // released op Done, and un-started ops Pending; the recovery phase
    // completed with every live op Done.
    if let Some(p1parked) = out.phase1.parked {
        for (op, st) in &out.phase1.statuses {
            let expected = if *op < p1parked {
                TaskStatus::Done
            } else if *op == p1parked {
                TaskStatus::Running
            } else {
                TaskStatus::Pending
            };
            if *st != expected {
                return Err(format!(
                    "the crashed phase left op {op} at {st:?}, expected {expected:?}"
                ));
            }
        }
    }
    if let Some(p2) = &out.phase2 {
        for (op, st) in &p2.statuses {
            if !matches!(st, TaskStatus::Done) {
                return Err(format!(
                    "the recovery phase left op {op} at {st:?}, expected Done"
                ));
            }
        }
    }
    Ok(())
}

fn reference(seed: u64) -> Result<WorldState, String> {
    let out = run_dag(seed, None)?;
    verify_phase_counts(
        &out,
        &BoundarySpec {
            name: "dag.pre_execution",
            class: CrashClass::FullyCommitted,
        },
    )?;
    let world = dump_world(&out.store, out.session);
    if world.lines.len() != 2 * N_OPS {
        return Err(format!(
            "reference world must hold {N_OPS} records + statuses: {world:?}"
        ));
    }
    // The reference scheduler itself ended with every op terminal-Done.
    for (op, st) in &out.phase1.statuses {
        if !matches!(st, TaskStatus::Done) {
            return Err(format!("reference run left op {op} at {st:?}"));
        }
    }
    Ok(world)
}

fn crash_run(seed: u64, boundary: &BoundarySpec) -> Result<WorldState, String> {
    let park = match boundary.name {
        "dag.pre_execution" => Some(0),
        name if name.starts_with("dag.parked#") => Some(boundary_index(name)? as u64 + 1),
        other => return Err(format!("unknown boundary {other:?}")),
    };
    // The pre-execution boundary parks BEFORE the first op starts: park
    // target 1, but the phase must crash before op 1 ever starts. Implement
    // by parking the run handle before any op can be observed.
    let out = if park == Some(0) {
        // Crash before the executor ever runs: phase 1 spawns nothing.
        let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
        let root = dir.path().to_path_buf();
        let (store, session) = open_store(&root, seed);
        let clock = Arc::new(TestClock::new(1_000_000 + (seed % 1_000_000) as i64));
        let counters = Arc::new((0..=N_OPS).map(|_| AtomicU64::new(0)).collect::<Vec<_>>());
        let all = ops();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| e.to_string())?;
        rt.block_on(async move {
            let store_ref = &store;
            let session_ref = session;
            let counters_ref = counters.clone();
            // Phase 1: ops registered, executor never spawned, then the
            // process dies: nothing ran, nothing was journaled.
            let sched = Scheduler::new(session_ref, clock.clone());
            for id in all.clone() {
                let (rel_tx, rel_rx) = tokio::sync::mpsc::unbounded_channel();
                let deps = deps_of(id);
                let payload = op_fn(
                    id,
                    sched.clone(),
                    deps,
                    counters_ref.clone(),
                    tokio::sync::mpsc::unbounded_channel().0,
                    rel_rx,
                );
                sched
                    .try_submit(payload)
                    .map_err(|e| format!("submit op {id} failed: {e}"))?;
                drop(rel_tx);
            }
            drop(sched);
            let phase1 = PhaseOutcome {
                counts: counters_ref
                    .iter()
                    .map(|c| c.load(Ordering::SeqCst))
                    .collect(),
                statuses: Vec::new(),
                parked: Some(0),
            };
            let durable = durable_terminal_ops(store_ref, session_ref);
            if !durable.is_empty() {
                return Err("pre-execution crash must not have journaled".into());
            }
            let phase2 = run_phase(
                seed,
                session_ref,
                clock,
                all,
                &deps_of,
                counters_ref,
                None,
                store_ref,
            )
            .await?;
            Ok(RunOutcome {
                _dir: dir,
                store,
                session: session_ref,
                phase1,
                phase2: Some(phase2),
                durable_pre_crash: Vec::new(),
            })
        })
    } else {
        run_dag(seed, park)
    }?;

    verify_phase_counts(&out, boundary)?;
    let world = dump_world(&out.store, out.session);
    Ok(world)
}

fn boundary_index(name: &str) -> Result<usize, String> {
    name.rsplit('#')
        .next()
        .ok_or_else(|| format!("boundary {name} has no index"))?
        .parse::<usize>()
        .map_err(|e| format!("bad boundary index in {name}: {e}"))
}

static BOUNDARY_SPECS: &[BoundarySpec] = &[
    BoundarySpec {
        name: "dag.pre_execution",
        class: CrashClass::FullyCommitted,
    },
    BoundarySpec {
        name: "dag.parked#0",
        class: CrashClass::FullyCommitted,
    },
    BoundarySpec {
        name: "dag.parked#1",
        class: CrashClass::FullyCommitted,
    },
    BoundarySpec {
        name: "dag.parked#2",
        class: CrashClass::FullyCommitted,
    },
    BoundarySpec {
        name: "dag.parked#3",
        class: CrashClass::FullyCommitted,
    },
    BoundarySpec {
        name: "dag.parked#4",
        class: CrashClass::FullyCommitted,
    },
    BoundarySpec {
        name: "dag.parked#5",
        class: CrashClass::FullyCommitted,
    },
];

pub fn campaign() -> Campaign {
    Campaign {
        name: "scheduler-dag-crash-mid-dag",
        boundaries: BOUNDARY_SPECS,
        reference,
        crash_run,
        check: check_equals,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn smoke_scheduler_dag_crash() {
    let c = campaign();
    let checks = super::run_campaign(&c, SMOKE_SEEDS).expect("smoke must pass");
    assert_eq!(checks, SMOKE_SEEDS * c.boundaries.len() as u64);
}

#[test]
#[ignore = "[fault] scheduler run crash mid-DAG vs reference terminal set, 300 seeds"]
fn full_scheduler_dag_crash() {
    let c = campaign();
    let checks = super::run_campaign(&c, FULL_SEEDS).expect("full campaign must pass");
    assert_eq!(checks, FULL_SEEDS * c.boundaries.len() as u64);
}
