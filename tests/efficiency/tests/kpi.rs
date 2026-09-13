//! KPI unit tests on SYNTHETIC durable rows: known row values -> exact
//! derive output. Every scenario is adversarial in the audit sense: multiple
//! attempts of one logical call, prefix-only rows that must not double count,
//! failed + reverted edits, failure records, human steering, truncated
//! bounded reads, corrupt durable shapes, and reopen stability.

use faktor_core::event::EventKind;
use faktor_core::id::{OpId, TaskId, TaskRevision, VerificationRecordId, WorktreeId};
use faktor_core::op::ModelCallAttempt;
use faktor_core::state::{AgentState, CriterionVerification, TaskState, VerificationStatus};
use faktor_store::{Store, TaskRow, VerificationRecordRow};
use faktor_tests_efficiency::*;
use serde_json::json;
use tempfile::tempdir;

struct Fixture {
    dir: tempfile::TempDir,
    store: Store,
    workspace: faktor_core::id::WorkspaceId,
    session: faktor_core::id::SessionId,
    task: TaskId,
}

impl Fixture {
    fn derive(&self) -> Result<TaskEfficiencyMetrics, EfficiencyError> {
        TaskEfficiencyMetrics::derive(&self.store, self.session, self.task)
    }
}

fn fixture(criteria: &[&str]) -> Fixture {
    let dir = tempdir().unwrap();
    let store = Store::open_fast(dir.path()).unwrap();
    let workspace = store.create_workspace("/ws").unwrap();
    let session = store.create_session(workspace, "kpi", "p", "m").unwrap();
    let task = TaskId::new(3);
    store
        .adopt_session_identity(session.id, WorktreeId::new(7), task)
        .unwrap();
    store
        .upsert_task(&TaskRow {
            task_id: task,
            session_id: session.id,
            goal: "kpi fixture".into(),
            acceptance_criteria: criteria.iter().map(|c| (*c).to_string()).collect(),
            plan: Vec::new(),
            attachments: Vec::new(),
            max_tokens: None,
            max_turns: None,
            spent_tokens: 0,
            spent_turns: 0,
            state: TaskState::Running,
            revision: TaskRevision::new(1),
            created_ms: 1_000_000,
            updated_ms: 1_000_000,
        })
        .unwrap();
    Fixture {
        dir,
        store,
        workspace,
        session: session.id,
        task,
    }
}

fn move_task_state(f: &Fixture, state: TaskState, updated_ms: i64) {
    let mut row = f.store.get_task(f.session, f.task).unwrap().unwrap();
    row.state = state;
    row.updated_ms = updated_ms;
    f.store.upsert_task(&row).unwrap();
}

fn complete_verified(f: &Fixture, now: i64) {
    move_task_state(f, TaskState::NeedsVerification, now - 2);
    move_task_state(f, TaskState::Verifying, now - 1);
    let task = f.store.get_task(f.session, f.task).unwrap().unwrap();
    let record = VerificationRecordRow {
        id: VerificationRecordId::new(1),
        task_id: f.task,
        revision: task.revision,
        workspace_id: f.workspace,
        worktree_id: WorktreeId::new(7),
        tree_hash: Some("tree".into()),
        criteria: task
            .acceptance_criteria
            .iter()
            .map(|key| CriterionVerification {
                criterion_key: key.clone(),
                passed: true,
                evidence: Some("fixture".into()),
            })
            .collect(),
        checks: Vec::new(),
        changed_files: Vec::new(),
        unrelated_changes: Vec::new(),
        reviewer: None,
        status: VerificationStatus::Running,
        started_ms: now - 1,
        completed_ms: None,
    };
    let record_id = f.store.verification_record_put(&record).unwrap();
    f.store
        .verification_record_finalize(record_id, VerificationStatus::Passed, now)
        .unwrap()
        .unwrap();
    f.store
        .task_complete_verified(f.session, f.task, task.revision, record_id, now)
        .unwrap()
        .unwrap();
}

/// One attempt-keyed call: reserve -> dispatch -> usage row -> settle.
fn attempt_call(
    f: &Fixture,
    logical: u64,
    attempt: u64,
    ordinal: u32,
    input: u64,
    output: u64,
    cost: u64,
) {
    let attempt = ModelCallAttempt::new(OpId::new(logical), OpId::new(attempt), ordinal).unwrap();
    let reservation = match f
        .store
        .cost_reserve_attempt(f.session, f.task, &attempt, cost, 1_000_100, None)
        .unwrap()
    {
        faktor_store::CostReserveOutcome::Granted(id) => id,
        other => panic!("reserve must be granted: {other:?}"),
    };
    f.store
        .cost_mark_dispatched(reservation, 1_000_101)
        .unwrap();
    f.store
        .record_provider_call_attempt(
            f.session,
            &attempt,
            Some(reservation),
            "p",
            "m",
            "completed",
            Some(input),
            Some(output),
            None,
        )
        .unwrap();
    f.store
        .cost_settle(reservation, cost, Some(cost), None, None, 1_000_102)
        .unwrap();
}

/// One legacy logical-op call: reserve -> dispatch -> usage row -> settle.
fn legacy_call(f: &Fixture, logical: u64, input: u64, output: u64, cost: u64) {
    let op = OpId::new(logical);
    let reservation = match f
        .store
        .cost_reserve(f.session, f.task, op, cost, 1_000_100)
        .unwrap()
    {
        faktor_store::CostReserveOutcome::Granted(id) => id,
        other => panic!("reserve must be granted: {other:?}"),
    };
    f.store
        .cost_mark_dispatched(reservation, 1_000_101)
        .unwrap();
    f.store
        .record_provider_call(
            f.session,
            op,
            "p",
            "m",
            "completed",
            Some(input),
            Some(output),
            None,
        )
        .unwrap();
    f.store
        .cost_settle(reservation, cost, Some(cost), None, None, 1_000_102)
        .unwrap();
}

fn prefix_observation(f: &Fixture, logical: u64, tokens: u64, stability: f64) {
    f.store
        .record_provider_call_with_prefix(
            f.session,
            OpId::new(logical),
            "p",
            "m",
            "completed",
            None,
            None,
            None,
            Some([7u8; 32]),
            Some(tokens),
            Some(stability),
        )
        .unwrap();
}

fn evidence_observation(
    f: &Fixture,
    ts: i64,
    original_bytes: u64,
    sent_tokens: u64,
    retrievals: u64,
) {
    f.store
        .append_event_v(
            f.session,
            None,
            EventKind::ContextPrepared,
            AgentState::BuildingContext,
            ts,
            Some(json!({
                EVIDENCE_OBSERVATION_KEY: {
                    "original_bytes": original_bytes,
                    "sent_tokens": sent_tokens,
                    "retrievals": retrievals
                }
            })),
            1,
        )
        .unwrap();
}

#[test]
fn derive_sums_every_durable_field_exactly() {
    let f = fixture(&["crit-a", "crit-b"]);

    // Two calls (one attempt-keyed, one legacy), each with a prefix row.
    attempt_call(&f, 10, 11, 0, 1000, 200, 400);
    legacy_call(&f, 12, 300, 50, 100);
    prefix_observation(&f, 10, 600, 1.0); // attributed cache 600
    prefix_observation(&f, 12, 100, 0.5); // attributed cache 50

    // Edits: one conflicted per-file write, two paths rolled back.
    f.store
        .append_ledger_entry(
            f.session,
            "edit_txn_progress",
            1,
            json!({"kind":"edit_txn_progress","txn_id":7,"seq":0,"path":"src/a.rs","outcome":"conflicted"}),
        )
        .unwrap();
    f.store
        .append_ledger_entry(
            f.session,
            "edit_txn_rolled_back",
            1,
            json!({"kind":"edit_txn_rolled_back","txn_id":7,"rolled_back":["src/a.rs","src/b.rs"],"rollback_conflicts":[]}),
        )
        .unwrap();
    // One repair turn.
    f.store
        .append_ledger_entry(
            f.session,
            "failure_recorded",
            1,
            json!({"kind":"failure_recorded","failure":"test failed"}),
        )
        .unwrap();

    // Two prompt admissions => one human intervention.
    f.store
        .append_event(
            f.session,
            None,
            EventKind::PromptAdmitted,
            AgentState::Idle,
            2_000,
            None,
        )
        .unwrap();
    f.store
        .append_event(
            f.session,
            None,
            EventKind::PromptAdmitted,
            AgentState::Idle,
            2_001,
            None,
        )
        .unwrap();

    // Evidence observations from two retrievals.
    evidence_observation(&f, 1_900, 1_000, 200, 2);
    evidence_observation(&f, 1_901, 1_500, 300, 3);

    complete_verified(&f, 1_002_500);

    let metrics = f.derive().unwrap();
    assert_eq!(metrics.input_tokens_total, 1_300);
    assert_eq!(metrics.input_tokens_cached, 650);
    assert_eq!(metrics.input_tokens_noncached, 650);
    assert_eq!(metrics.output_tokens, 250);
    assert_eq!(metrics.model_cost_micro, 500);
    assert_eq!(metrics.evidence_original_bytes, 2_500);
    assert_eq!(metrics.evidence_sent_tokens, 500);
    assert_eq!(metrics.evidence_retrieval_count, 5);
    assert_eq!(metrics.model_calls, 2, "prefix-only rows are not calls");
    assert_eq!(metrics.repair_turns, 1);
    assert_eq!(metrics.failed_edits, 1);
    assert_eq!(metrics.reverted_edits, 2);
    assert_eq!(metrics.human_interventions, 1);
    assert_eq!(metrics.wall_ms, 2_500);
    assert!(metrics.verified);

    // The reservation-derived cost agrees with the durable materialized fold.
    let cost = f.store.cost_task_row(f.session, f.task).unwrap().unwrap();
    assert_eq!(metrics.model_cost_micro, cost.spent_cost_micro);

    // Reopen stability: the same store bytes derive the same KPIs. The
    // tempdir is kept alive while only the Store handle is dropped.
    let Fixture {
        dir,
        store,
        session,
        task,
        workspace: _,
    } = f;
    drop(store);
    let reopened = Store::open_fast(dir.path()).unwrap();
    let again = TaskEfficiencyMetrics::derive(&reopened, session, task).unwrap();
    assert_eq!(metrics, again, "reopen must not move a durable KPI");
}

#[test]
fn multiple_attempts_of_one_logical_call_count_and_sum_once_each() {
    let f = fixture(&["crit-a"]);
    // Attempt 0 fails with no usage counters; attempt 1 succeeds. Both key
    // the same logical op but different attempt ids and reservations.
    attempt_call(&f, 20, 21, 0, 0, 0, 0);
    attempt_call(&f, 20, 22, 1, 700, 90, 250);
    let metrics = f.derive().unwrap();
    assert_eq!(metrics.model_calls, 2);
    assert_eq!(metrics.input_tokens_total, 700);
    assert_eq!(metrics.output_tokens, 90);
    assert_eq!(metrics.model_cost_micro, 250);

    // Truncation is loud, never a partial total.
    let (page, truncated) = f
        .store
        .provider_call_task_rows(f.session, f.task, 1)
        .unwrap();
    assert_eq!(page.len(), 1);
    assert!(truncated);
}

#[test]
fn pending_and_unknown_spend_never_fold_into_the_cost_kpi() {
    let f = fixture(&["crit-a"]);
    // A dispatched-but-uncertain reservation (provider may have billed, but
    // nothing was folded into the task's spend).
    let reservation = match f
        .store
        .cost_reserve(f.session, f.task, OpId::new(30), 900, 1_000_100)
        .unwrap()
    {
        faktor_store::CostReserveOutcome::Granted(id) => id,
        other => panic!("{other:?}"),
    };
    f.store
        .cost_mark_dispatched(reservation, 1_000_101)
        .unwrap();
    f.store
        .cost_mark_uncertain(reservation, "stream_failed", None, 1_000_102)
        .unwrap();
    let metrics = f.derive().unwrap();
    assert_eq!(metrics.model_cost_micro, 0, "uncertain is not folded spend");
    assert_eq!(
        f.store
            .cost_task_row(f.session, f.task)
            .unwrap()
            .unwrap()
            .spent_cost_micro,
        0
    );
}

#[test]
fn derive_refuses_task_mismatch_and_corrupt_shapes() {
    let f = fixture(&["crit-a"]);

    // Session scoped to another task: refuse, never guess attribution.
    let wrong = TaskId::new(9);
    f.store
        .adopt_session_identity(f.session, WorktreeId::new(7), wrong)
        .unwrap();
    match TaskEfficiencyMetrics::derive(&f.store, f.session, f.task) {
        Err(EfficiencyError::TaskSessionMismatch { .. }) => {}
        other => panic!("expected TaskSessionMismatch, got {other:?}"),
    }
    f.store
        .adopt_session_identity(f.session, WorktreeId::new(7), f.task)
        .unwrap();

    // A malformed evidence observation is loud.
    f.store
        .append_event_v(
            f.session,
            None,
            EventKind::ContextPrepared,
            AgentState::BuildingContext,
            3_000,
            Some(json!({ EVIDENCE_OBSERVATION_KEY: { "original_bytes": "a lot" } })),
            1,
        )
        .unwrap();
    match f.derive() {
        Err(EfficiencyError::Malformed(message)) => {
            assert!(message.contains(EVIDENCE_OBSERVATION_KEY), "{message}")
        }
        other => panic!("expected Malformed evidence observation, got {other:?}"),
    }

    // A corrupt prefix stability injected behind the API's back is loud.
    attempt_call(&f, 40, 41, 0, 100, 10, 10);
    prefix_observation(&f, 40, 50, 1.0);
    f.store
        .sql_execute(
            "UPDATE provider_call SET prefix_stability = 2.0 WHERE prompt_prefix_hash IS NOT NULL",
        )
        .unwrap();
    match f.derive() {
        Err(EfficiencyError::Store(faktor_store::StoreError::Malformed(message))) => {
            assert!(message.contains("prefix_stability"), "{message}")
        }
        other => panic!("expected Malformed prefix stability, got {other:?}"),
    }
}

#[test]
fn unverified_control_derives_unverified_and_wall_is_zero_clamped() {
    let f = fixture(&["crit-a"]);
    // Move the task's updated_ms behind created_ms via a raw durable update:
    // wall_ms must clamp at 0, never go negative.
    f.store
        .sql_execute(&format!(
            "UPDATE task SET updated_ms = created_ms - 500 WHERE session_id = {}",
            f.session.raw()
        ))
        .unwrap();
    let metrics = f.derive().unwrap();
    assert_eq!(metrics.wall_ms, 0);
    assert!(!metrics.verified);
}
