//! Campaign (a): store/journal append + compaction crash at every
//! durability boundary (P0-76, 500 seeds).
//!
//! Rigid per-seed op recipe over ONE session of a real `faktor-store`:
//!
//! ```text
//! setup  create_workspace + create_session (journals SessionCreated, seq 1)
//! op 0   actor flush 1: batch_hot_writes [AppendEvent, PutMessage]
//! op 1   actor flush 2: batch_hot_writes [AppendEvent, PutPart, AppendEvent]
//! op 2   single append_event_v
//! op 3..5  typed-ledger appends (ledger_entry seq 1..3)
//! op 6   compaction fold: compact_ledger(below = 3) — one transaction that
//!        deletes ledger seq 1..2 and rewrites the head checkpoint
//! op 7   standalone head refresh: put_ledger_head
//! ```
//!
//! Boundaries are exactly the durability crossings the store seam declares:
//! crash inside a group after a write (the WHOLE group must roll back),
//! crash before/after each transactional COMMIT, crash mid-fold between the
//! compaction DELETE and the head rewrite, crash before/after the
//! standalone head write. Content (kinds, payloads, ids, sizes) is
//! seed-derived and hostile-inclusive (empty payloads, `u64::MAX` op ids,
//! multi-KiB payloads). Each op's content derives from a per-op LCG, so a
//! recovery replay from any durable cursor reproduces the reference bytes
//! exactly.
//!
//! Certification per (seed, boundary):
//! 1. reopen the crashed store and prove the DURABLE world equals the
//!    expected prefix world of the completed ops — exactly the declared
//!    class (`PreOp` = the crash rolled the in-flight op back;
//!    `FullyCommitted` = the crashed op's COMMIT landed);
//! 2. replay the op tail on the reopened store (the recovery contract: the
//!    daemon re-issues from the durable cursor);
//! 3. the recovered world EQUALS the uninterrupted reference world.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;

use faktor_core::event::EventKind;
use faktor_core::id::{OpId, SessionId};
use faktor_core::state::AgentState;
use faktor_store::{CrashArm, HotWrite, HotWriteOutcome, Store};

use super::{check_equals, BoundarySpec, Campaign, CrashClass, Lcg, WorldState};

/// Seeds of the full [fault]-gated campaign.
pub const FULL_SEEDS: u64 = 500;
/// Seeds of the normal-mode smoke run.
pub const SMOKE_SEEDS: u64 = 3;

const OPS: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    Flush1 = 0,
    Flush2 = 1,
    AppendEv = 2,
    Ledger1 = 3,
    Ledger2 = 4,
    Ledger3 = 5,
    Compact = 6,
    Head = 7,
}

// ---------------------------------------------------------------------------
// Seed-derived content. Every op derives its content from a FRESH per-op
// LCG (keyed on seed and op index), so executing an op after a crash-replay
// reproduces byte-identical content without any stream skipping.
// ---------------------------------------------------------------------------

fn op_lcg(seed: u64, op: usize) -> Lcg {
    Lcg::new(seed ^ (op as u64).wrapping_mul(0x9E37_79B9_7F4A_C15F))
}

fn pick<'a, T>(lcg: &mut Lcg, choices: &'a [T]) -> &'a T {
    &choices[lcg.below(choices.len() as u64) as usize]
}

const KINDS: &[EventKind] = &[
    EventKind::PromptReceived,
    EventKind::ContextPrepared,
    EventKind::ModelStarted,
    EventKind::ModelChunkReceived,
    EventKind::ToolRequested,
    EventKind::ToolStarted,
    EventKind::ToolCompleted,
    EventKind::CheckpointCreated,
    EventKind::ContextCompacted,
    EventKind::TurnCompleted,
    EventKind::PermissionGranted,
    EventKind::Failed,
    EventKind::CrashDetected,
    EventKind::RecoveryApplied,
    EventKind::SessionEnded,
];

const STATES: &[AgentState] = &[
    AgentState::Idle,
    AgentState::BuildingContext,
    AgentState::WaitingForModel,
    AgentState::Streaming,
    AgentState::ExecutingTool,
    AgentState::ReadyForNextTurn,
    AgentState::Completed,
    AgentState::NeedsUserInput,
    AgentState::Suspended,
];

fn hostile_payload(lcg: &mut Lcg, op: usize) -> Option<serde_json::Value> {
    match lcg.below(7) {
        0 => None,
        1 => Some(serde_json::json!({ "op": op, "n": lcg.next_u64() })),
        2 => Some(serde_json::json!([lcg.next_u64(), i64::MIN, i64::MAX])),
        3 => Some(serde_json::json!({})),
        4 => Some(serde_json::json!(i64::MAX)),
        5 => Some(serde_json::json!(null)),
        _ => Some(serde_json::json!({
            "blob": "x".repeat(1 + lcg.below(3000) as usize),
            "n": lcg.next_u64(),
        })),
    }
}

fn op_id(lcg: &mut Lcg) -> Option<OpId> {
    match lcg.below(4) {
        0 => None,
        1 => Some(OpId::new(1)),
        _ => Some(OpId::new(u64::MAX - lcg.below(3))),
    }
}

fn text(lcg: &mut Lcg, max: usize) -> String {
    let n = 1 + lcg.below(max as u64) as usize;
    let mut s = format!("seed{:x}", lcg.next_u64());
    while s.len() < n {
        s.push('c');
    }
    s
}

fn ts_ms(lcg: &mut Lcg, i: usize) -> i64 {
    1_700_000_000_000 + (i as i64) * 1_000_000 + (lcg.below(1000) as i64)
}

// ---------------------------------------------------------------------------
// Setup + op execution against a real store
// ---------------------------------------------------------------------------

fn setup(root: &Path, seed: u64) -> (Store, SessionId) {
    let store = Store::open(root.join("store"), true).expect("store opens");
    let ws = store.create_workspace("/w").expect("workspace");
    let row = store
        .create_session(ws, &format!("s{seed}"), "p", "m")
        .expect("session");
    (store, row.id)
}

fn exec_op(store: &Store, session: SessionId, seed: u64, op: Op) {
    let i = op as usize;
    let mut lcg = op_lcg(seed, i);
    match op {
        Op::Flush1 => {
            let ts = ts_ms(&mut lcg, i);
            let w1 = HotWrite::AppendEvent {
                session_id: session,
                op_id: op_id(&mut lcg),
                kind: *pick(&mut lcg, KINDS),
                state: *pick(&mut lcg, STATES),
                ts_ms: ts,
                payload: hostile_payload(&mut lcg, i),
                payload_ver: 1,
            };
            let w2 = HotWrite::PutMessage {
                session_id: session,
                seq: 1,
                role: "user".into(),
                data: serde_json::json!({ "m": seed, "t": text(&mut lcg, 2000) }),
            };
            let (outcomes, _) = store.batch_hot_writes(&[w1, w2]).expect("flush1");
            assert_eq!(outcomes.len(), 2, "flush1 outcome count");
            assert!(matches!(outcomes[0], Ok(HotWriteOutcome::EventSeq(_))));
            assert!(matches!(outcomes[1], Ok(HotWriteOutcome::RowId(1))));
        }
        Op::Flush2 => {
            let ts = ts_ms(&mut lcg, i);
            let w1 = HotWrite::AppendEvent {
                session_id: session,
                op_id: op_id(&mut lcg),
                kind: *pick(&mut lcg, KINDS),
                state: *pick(&mut lcg, STATES),
                ts_ms: ts,
                payload: hostile_payload(&mut lcg, i),
                payload_ver: 1,
            };
            // PutPart references the message row the FIRST flush created
            // (rowid 1, deterministic: op 1 only runs after op 0 committed
            // or was replayed identically).
            let w2 = HotWrite::PutPart {
                message_id: 1,
                kind: "text".into(),
                data: serde_json::json!({ "p": text(&mut lcg, 800) }),
            };
            let w3 = HotWrite::AppendEvent {
                session_id: session,
                op_id: op_id(&mut lcg),
                kind: *pick(&mut lcg, KINDS),
                state: *pick(&mut lcg, STATES),
                ts_ms: ts + 1,
                payload: hostile_payload(&mut lcg, i),
                payload_ver: 1,
            };
            let (outcomes, _) = store.batch_hot_writes(&[w1, w2, w3]).expect("flush2");
            assert_eq!(outcomes.len(), 3, "flush2 outcome count");
            assert!(matches!(outcomes[1], Ok(HotWriteOutcome::RowId(1))));
        }
        Op::AppendEv => {
            store
                .append_event_v(
                    session,
                    op_id(&mut lcg),
                    *pick(&mut lcg, KINDS),
                    *pick(&mut lcg, STATES),
                    ts_ms(&mut lcg, i),
                    hostile_payload(&mut lcg, i),
                    1,
                )
                .expect("append_event_v");
        }
        Op::Ledger1 => {
            let payload =
                hostile_payload(&mut lcg, i).unwrap_or_else(|| serde_json::json!({ "goal": seed }));
            store
                .append_ledger_entry(session, "goal_set", 1, payload)
                .expect("ledger1");
        }
        Op::Ledger2 => {
            let payload =
                hostile_payload(&mut lcg, i).unwrap_or_else(|| serde_json::json!({ "d": seed }));
            store
                .append_ledger_entry(session, "decision", 1, payload)
                .expect("ledger2");
        }
        Op::Ledger3 => {
            let payload =
                hostile_payload(&mut lcg, i).unwrap_or_else(|| serde_json::json!({ "b": seed }));
            store
                .append_ledger_entry(session, "blocker_opened", 1, payload)
                .expect("ledger3");
        }
        Op::Compact => {
            // Fold of ledger seq 1..=2 into the head checkpoint: deletes
            // exactly the two older entries.
            let fold = serde_json::json!({ "fold": 2, "seed": seed });
            let deleted = store
                .compact_ledger(session, 3, &[], fold, 2, 1)
                .expect("compact");
            assert_eq!(deleted, 2, "compaction deletes exactly seq 1..2");
        }
        Op::Head => {
            let head = serde_json::json!({ "fold": 3, "seed": seed });
            store.put_ledger_head(session, head, 3, 1).expect("head");
        }
    }
}

fn op_of(idx: usize) -> Op {
    match idx {
        0 => Op::Flush1,
        1 => Op::Flush2,
        2 => Op::AppendEv,
        3 => Op::Ledger1,
        4 => Op::Ledger2,
        5 => Op::Ledger3,
        6 => Op::Compact,
        7 => Op::Head,
        _ => panic!("op index {idx} out of the rigid recipe"),
    }
}

fn run_ops(store: &Store, session: SessionId, seed: u64, from: usize, to: usize) {
    for idx in from..to {
        exec_op(store, session, seed, op_of(idx));
    }
}

// ---------------------------------------------------------------------------
// Canonical world-state dump. Volatile wall-clock columns (created_ms /
// updated_ms) and event timestamps are outside the comparison scope; every
// semantic column is compared exactly.
// ---------------------------------------------------------------------------

fn dump_world(store: &Store, session: SessionId) -> WorldState {
    let mut lines = Vec::new();
    for (ev, ver) in store.events_versioned_range(session, 0, None).unwrap() {
        lines.push(format!(
            "ev:{}:{}:{}:{}:{}:{}",
            ev.seq.raw(),
            kind_tag(ev.kind),
            state_tag(&ev.state),
            ev.op_id.map(|o| o.raw()).unwrap_or(0),
            ver,
            ev.payload.map(|p| p.to_string()).unwrap_or_default()
        ));
    }
    let mut msgs = store.messages_before(session, None, u64::MAX).unwrap();
    msgs.sort_by_key(|m| m.seq);
    for m in msgs {
        lines.push(format!("msg:{}:{}:{}", m.id, m.seq, m.data));
        let mut parts = store.parts_of(m.id).unwrap();
        parts.sort_by_key(|p| p.id);
        for p in parts {
            lines.push(format!("part:{}:{}:{}", p.message_id, p.kind, p.data));
        }
    }
    for le in store.ledger_entries(session, None, u64::MAX).unwrap() {
        lines.push(format!(
            "le:{}:{}:{}:{}",
            le.seq, le.entry_type, le.schema_ver, le.payload
        ));
    }
    match store.ledger_head(session).unwrap() {
        Some(h) => lines.push(format!(
            "head:{}:{}:{}",
            h.checkpoint_seq, h.schema_ver, h.head_json
        )),
        None => lines.push("head:-".into()),
    }
    WorldState { lines }
}

fn kind_tag(k: EventKind) -> &'static str {
    match k {
        EventKind::SessionCreated => "SessionCreated",
        EventKind::PromptReceived => "PromptReceived",
        EventKind::ContextPrepared => "ContextPrepared",
        EventKind::ModelStarted => "ModelStarted",
        EventKind::ModelChunkReceived => "ModelChunkReceived",
        EventKind::ToolRequested => "ToolRequested",
        EventKind::ToolStarted => "ToolStarted",
        EventKind::FileChanged => "FileChanged",
        EventKind::ToolCompleted => "ToolCompleted",
        EventKind::ToolCancelled => "ToolCancelled",
        EventKind::CheckpointCreated => "CheckpointCreated",
        EventKind::ContextCompacted => "ContextCompacted",
        EventKind::CompactRejected => "CompactRejected",
        EventKind::SubagentStarted => "SubagentStarted",
        EventKind::SubagentCompleted => "SubagentCompleted",
        EventKind::TurnCompleted => "TurnCompleted",
        EventKind::PermissionGranted => "PermissionGranted",
        EventKind::PermissionDenied => "PermissionDenied",
        EventKind::PromptAdmitted => "PromptAdmitted",
        EventKind::PhaseChanged => "PhaseChanged",
        EventKind::ReplayStarted => "ReplayStarted",
        EventKind::CrashDetected => "CrashDetected",
        EventKind::RecoveryApplied => "RecoveryApplied",
        EventKind::SessionEnded => "SessionEnded",
        EventKind::Suspended => "Suspended",
        EventKind::Resumed => "Resumed",
        EventKind::Failed => "Failed",
    }
}

fn state_tag(s: &AgentState) -> &'static str {
    match s {
        AgentState::Idle => "Idle",
        AgentState::Preparing => "Preparing",
        AgentState::BuildingContext => "BuildingContext",
        AgentState::WaitingForModel => "WaitingForModel",
        AgentState::Streaming => "Streaming",
        AgentState::ToolRequested => "ToolRequested",
        AgentState::WaitingForPermission => "WaitingForPermission",
        AgentState::ExecutingTool => "ExecutingTool",
        AgentState::Validating => "Validating",
        AgentState::UpdatingMemory => "UpdatingMemory",
        AgentState::ReadyForNextTurn => "ReadyForNextTurn",
        AgentState::Completed => "Completed",
        AgentState::Cancelled => "Cancelled",
        AgentState::FailedRecoverable => "FailedRecoverable",
        AgentState::FailedPermanent => "FailedPermanent",
        AgentState::NeedsUserInput => "NeedsUserInput",
        AgentState::Suspended => "Suspended",
    }
}

// ---------------------------------------------------------------------------
// The campaign
// ---------------------------------------------------------------------------

/// (name, crashed op index, class, seam point, seam ordinal)
const BOUNDARIES: &[(&str, usize, CrashClass, &str, u64)] = &[
    (
        "flush1.precommit",
        0,
        CrashClass::PreOp,
        "flush_precommit",
        0,
    ),
    (
        "flush1.committed",
        0,
        CrashClass::FullyCommitted,
        "flush_committed",
        0,
    ),
    // Mid-flush-2: crash after its SECOND write executed — the whole group
    // must roll back. Crossing ordinal 3 = flush1's writes (0..1) then
    // flush2's first two writes (2..3).
    ("flush2.midwrite", 1, CrashClass::PreOp, "flush_progress", 3),
    (
        "flush2.precommit",
        1,
        CrashClass::PreOp,
        "flush_precommit",
        1,
    ),
    (
        "flush2.committed",
        1,
        CrashClass::FullyCommitted,
        "flush_committed",
        1,
    ),
    ("ev4.precommit", 2, CrashClass::PreOp, "ev_precommit", 0),
    (
        "ev4.committed",
        2,
        CrashClass::FullyCommitted,
        "ev_committed",
        0,
    ),
    ("le1.precommit", 3, CrashClass::PreOp, "le_precommit", 0),
    (
        "le1.committed",
        3,
        CrashClass::FullyCommitted,
        "le_committed",
        0,
    ),
    ("le2.precommit", 4, CrashClass::PreOp, "le_precommit", 1),
    (
        "le2.committed",
        4,
        CrashClass::FullyCommitted,
        "le_committed",
        1,
    ),
    ("le3.precommit", 5, CrashClass::PreOp, "le_precommit", 2),
    (
        "le3.committed",
        5,
        CrashClass::FullyCommitted,
        "le_committed",
        2,
    ),
    ("compact.fold", 6, CrashClass::PreOp, "compact_fold", 0),
    (
        "compact.precommit",
        6,
        CrashClass::PreOp,
        "compact_precommit",
        0,
    ),
    (
        "compact.committed",
        6,
        CrashClass::FullyCommitted,
        "compact_committed",
        0,
    ),
    ("head.precommit", 7, CrashClass::PreOp, "head_prewrite", 0),
    (
        "head.committed",
        7,
        CrashClass::FullyCommitted,
        "head_written",
        0,
    ),
];

static BOUNDARY_SPECS: &[BoundarySpec] = &[
    BoundarySpec {
        name: "flush1.precommit",
        class: CrashClass::PreOp,
    },
    BoundarySpec {
        name: "flush1.committed",
        class: CrashClass::FullyCommitted,
    },
    BoundarySpec {
        name: "flush2.midwrite",
        class: CrashClass::PreOp,
    },
    BoundarySpec {
        name: "flush2.precommit",
        class: CrashClass::PreOp,
    },
    BoundarySpec {
        name: "flush2.committed",
        class: CrashClass::FullyCommitted,
    },
    BoundarySpec {
        name: "ev4.precommit",
        class: CrashClass::PreOp,
    },
    BoundarySpec {
        name: "ev4.committed",
        class: CrashClass::FullyCommitted,
    },
    BoundarySpec {
        name: "le1.precommit",
        class: CrashClass::PreOp,
    },
    BoundarySpec {
        name: "le1.committed",
        class: CrashClass::FullyCommitted,
    },
    BoundarySpec {
        name: "le2.precommit",
        class: CrashClass::PreOp,
    },
    BoundarySpec {
        name: "le2.committed",
        class: CrashClass::FullyCommitted,
    },
    BoundarySpec {
        name: "le3.precommit",
        class: CrashClass::PreOp,
    },
    BoundarySpec {
        name: "le3.committed",
        class: CrashClass::FullyCommitted,
    },
    BoundarySpec {
        name: "compact.fold",
        class: CrashClass::PreOp,
    },
    BoundarySpec {
        name: "compact.precommit",
        class: CrashClass::PreOp,
    },
    BoundarySpec {
        name: "compact.committed",
        class: CrashClass::FullyCommitted,
    },
    BoundarySpec {
        name: "head.precommit",
        class: CrashClass::PreOp,
    },
    BoundarySpec {
        name: "head.committed",
        class: CrashClass::FullyCommitted,
    },
];

fn reference(seed: u64) -> Result<WorldState, String> {
    let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
    let (store, session) = setup(dir.path(), seed);
    run_ops(&store, session, seed, 0, OPS);
    let world = dump_world(&store, session);
    drop(store);
    Ok(world)
}

/// Durable prefix oracle: the expected world after exactly `completed` ops.
/// A throwaway store run is the cheapest faithful oracle (tiny prefix).
fn prefix_world(seed: u64, completed: usize) -> WorldState {
    let dir = tempfile::tempdir().expect("tempdir");
    let (store, session) = setup(dir.path(), seed);
    run_ops(&store, session, seed, 0, completed);
    let world = dump_world(&store, session);
    drop(store);
    world
}

fn crash_run(seed: u64, boundary: &BoundarySpec) -> Result<WorldState, String> {
    let (_, crashed_op, class, point, ordinal) = BOUNDARIES
        .iter()
        .find(|(name, ..)| *name == boundary.name)
        .ok_or_else(|| format!("unknown boundary {:?}", boundary.name))?;
    let dir = tempfile::tempdir().map_err(|e| e.to_string())?;

    // --- crashed run: fresh store, armed seam, ops through the crashed op.
    let caught = catch_unwind(AssertUnwindSafe(|| {
        let (store, session) = setup(dir.path(), seed);
        store.crash_arm(CrashArm {
            point,
            ordinal: *ordinal,
        });
        run_ops(&store, session, seed, 0, crashed_op + 1);
    }));
    super::expect_crash_fired(caught)?;

    // --- reopen: the durable world must be EXACTLY the expected prefix
    // (the declared class of the boundary), never a torn residue.
    let reopened = Store::open(dir.path().join("store"), true).map_err(|e| e.to_string())?;
    let sessions = reopened.list_sessions(None).map_err(|e| e.to_string())?;
    assert_eq!(sessions.len(), 1, "recipe creates exactly one session");
    let session = sessions[0].id;
    let durable = dump_world(&reopened, session);
    let (completed, replay_from) = match class {
        CrashClass::PreOp => (*crashed_op, *crashed_op),
        CrashClass::FullyCommitted => (crashed_op + 1, crashed_op + 1),
        CrashClass::Ambiguous => unreachable!("campaign (a) declares no ambiguous boundary"),
    };
    let expected = prefix_world(seed, completed);
    if durable != expected {
        return Err(format!(
            "durable world after crash diverges from the declared {} class ({completed} completed ops):\n{}",
            class_name(class),
            durable.diff(&expected)
        ));
    }

    // --- recovery contract: replay the tail from the durable cursor.
    run_ops(&reopened, session, seed, replay_from, OPS);
    let recovered = dump_world(&reopened, session);
    Ok(recovered)
}

fn class_name(c: &CrashClass) -> &'static str {
    match c {
        CrashClass::PreOp => "PreOp",
        CrashClass::FullyCommitted => "FullyCommitted",
        CrashClass::Ambiguous => "Ambiguous",
    }
}

pub fn campaign() -> Campaign {
    Campaign {
        name: "store-journal-append-compaction",
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
fn smoke_store_journal_boundaries() {
    let c = campaign();
    let checks = super::run_campaign(&c, SMOKE_SEEDS).expect("smoke must pass");
    assert_eq!(checks, SMOKE_SEEDS * c.boundaries.len() as u64);
}

#[test]
#[ignore = "[fault] store/journal append+compaction crash at every durability boundary, 500 seeds"]
fn full_store_journal_boundaries() {
    let c = campaign();
    let checks = super::run_campaign(&c, FULL_SEEDS).expect("full campaign must pass");
    assert_eq!(checks, FULL_SEEDS * c.boundaries.len() as u64);
}
