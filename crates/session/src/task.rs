//! The first-class durable Task object (audit 25) + the locked-down task
//! state machine (audit P0-7) + the first-class VerificationRecord (audit
//! P0-8).
//!
//! A `Task` is the durable object a session works on: a goal, the
//! acceptance criteria derived from that goal (goal + project checks,
//! seeded once), an append-only ordered step plan, and a durable budget
//! envelope (`max_tokens`/`max_turns` vs crash-safe `spent_tokens`/
//! `spent_turns`). It lives in typed store rows (`task`, schema v10) that
//! survive IDE close, daemon restart, OS restart, provider switch and
//! context compaction — compaction never rewrites them and they are never
//! FIFO-evicted.
//!
//! # Revisions (schema v14)
//!
//! Every effective state/criteria/plan/budget mutation bumps the row's
//! monotonic `revision` exactly once, in the same transaction as the
//! mutation. The revision is the row's optimistic-lock token: the
//! transition and completion APIs take the caller's `expected_revision`
//! and refuse with a typed error when the row moved on.
//!
//! # State machine enforcement (audit P0-7)
//!
//! `VerifiedComplete` (and the states that lead to it, `NeedsVerification`
//! and `Verifying`) can NEVER be assigned through a generic patch:
//! [`SessionHandle::update_task`] rejects any patch carrying a
//! completion-relevant state with [`TaskError::CompletionStateViaPatch`]
//! and any patch that would jump a non-completion machine edge with
//! [`TaskError::IllegalTransition`]. The legal producers are exactly:
//!
//! - [`SessionHandle::transition_task`] — one named [`TaskTransition`]
//!   edge; `VerifiedComplete` has no edge and is unreachable here;
//! - [`SessionHandle::complete_verified_task`] — the ONLY path to
//!   `VerifiedComplete`, validated in ONE store transaction against a
//!   passing [`VerificationRecord`] that certifies the task's current
//!   revision and covers every current acceptance criterion.
//!
//! The store backstops the same rule at the row level: a raw
//! `task` upsert that would move a row INTO a completion-relevant state is
//! refused unless the machine allows the edge (or the row already holds
//! that state), so no generic write — API or raw — can mint completion.
//!
//! Bounds are enforced HERE, before any write, and oversized input is
//! REJECTED with an error — never silently truncated: goal <=
//! [`MAX_TASK_GOAL_BYTES`] bytes, criteria <= [`MAX_TASK_CRITERIA`] entries
//! (each <= [`MAX_TASK_CRITERION_BYTES`]), plan <= [`MAX_TASK_PLAN_STEPS`]
//! steps (each <= [`MAX_TASK_STEP_BYTES`]). A patch with a `None` field
//! keeps the row's current value, so updates are read-modify-write safe
//! under the session command lock.
//!
//! # Error typing
//!
//! The task operations that can fail with machine/proof rejections return
//! the crate's typed [`TaskError`] (an audit requirement: distinct typed
//! causes, never prose-only). The legacy `create_task` keeps returning
//! [`faktor_core::Error`] because the agent runtime performs a direct
//! `return` of its mapped result; its rejections carry stable
//! `create_task refused: ...` messages. Every `TaskError` converts into
//! [`faktor_core::Error`], so `?` interop with core-`Result` callers is
//! unchanged.

use faktor_core::id::{
    SessionId, TaskId, TaskRevision, VerificationRecordId, WorkspaceId, WorktreeId,
};
use faktor_core::state::{
    CheckExecution, CriterionVerification, FileStateEvidence, TaskState, TaskTransition,
    VerificationStatus,
};

use crate::handle::SessionHandle;
use crate::SessionError;

/// Hard bound on one task goal (UTF-8 bytes).
pub const MAX_TASK_GOAL_BYTES: usize = 16 * 1024;
/// Hard bound on the acceptance-criteria entry count.
pub const MAX_TASK_CRITERIA: usize = 32;
/// Hard bound on ONE criterion (mirrors the memory-fact value cap).
pub const MAX_TASK_CRITERION_BYTES: usize = 3000;
/// Hard bound on the append-only plan's step count.
pub const MAX_TASK_PLAN_STEPS: usize = 256;
/// Hard bound on ONE plan step.
pub const MAX_TASK_STEP_BYTES: usize = 3000;

// ------------------------------------------------------ record bounds (P0-8)

/// Hard bound on the number of criterion verdicts in one record.
pub const MAX_VERIFICATION_RECORD_CRITERIA: usize = 64;
/// Serialized bound of the record's criteria JSON (128 KiB).
pub const MAX_VERIFICATION_CRITERIA_JSON_BYTES: usize = 128 * 1024;
/// Hard bound on the executed checks in one record.
pub const MAX_VERIFICATION_RECORD_CHECKS: usize = 256;
/// Serialized bound of the record's checks JSON (256 KiB).
pub const MAX_VERIFICATION_CHECKS_JSON_BYTES: usize = 256 * 1024;
/// Hard bound on the changed-file evidence entries in one record.
pub const MAX_VERIFICATION_CHANGED_FILES: usize = 4096;
/// Serialized bound of the record's changed-files JSON.
pub const MAX_VERIFICATION_CHANGED_FILES_JSON_BYTES: usize = 128 * 1024;
/// Hard bound on the unrelated-change path entries in one record.
pub const MAX_VERIFICATION_UNRELATED_CHANGES: usize = 4096;
/// Serialized bound of the record's unrelated-changes JSON.
pub const MAX_VERIFICATION_UNRELATED_JSON_BYTES: usize = 128 * 1024;
/// Serialized bound of the opaque reviewer JSON.
pub const MAX_VERIFICATION_REVIEWER_JSON_BYTES: usize = 16 * 1024;
/// Bound on the tree-hash hex text.
pub const MAX_VERIFICATION_TREE_HASH_BYTES: usize = 128;
/// Bound on one criterion key (equal texts live on the task row at
/// <= MAX_TASK_CRITERION_BYTES; the bound must not be tighter than that).
pub const MAX_VERIFICATION_CRITERION_KEY_BYTES: usize = MAX_TASK_CRITERION_BYTES;
/// Bound on one criterion's evidence prose.
pub const MAX_VERIFICATION_EVIDENCE_BYTES: usize = 4096;
/// Bound on one check name.
pub const MAX_VERIFICATION_CHECK_NAME_BYTES: usize = 2048;
/// Bound on one check program.
pub const MAX_VERIFICATION_PROGRAM_BYTES: usize = 4096;
/// Bound on the per-argument count of one check.
pub const MAX_VERIFICATION_CHECK_ARGS: usize = 32;
/// Bound on ONE check argument.
pub const MAX_VERIFICATION_CHECK_ARG_BYTES: usize = 1024;
/// Bound on one check category.
pub const MAX_VERIFICATION_CATEGORY_BYTES: usize = 128;
/// Bound on one check summary prose.
pub const MAX_VERIFICATION_SUMMARY_BYTES: usize = 8192;
/// Bound on one file path (mirrors the message-payload path bound).
pub const MAX_VERIFICATION_PATH_BYTES: usize = 4096;
/// Bound on one file digest hex text.
pub const MAX_VERIFICATION_DIGEST_BYTES: usize = 128;

/// The durable budget envelope of a Task. `None` max fields mean unlimited;
/// `spent_*` fields grow monotonically from durable sources (provider-call
/// rows + `turn_completed` journal events), so spend survives crashes.
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct TaskBudget {
    pub max_tokens: Option<u64>,
    pub max_turns: Option<u32>,
    pub spent_tokens: u64,
    pub spent_turns: u32,
}

/// The durable Task object (audit 25). One row per `(session_id, task_id)`;
/// `task_id` is the session's adopted durable task identity.
///
/// The row's `revision` counter is NOT a field of this struct (the agent
/// runtime constructs `Task` literals by field, and adding a field would
/// break every literal site in a crate this wave may not touch); it rides
/// the store row and is read through
/// [`SessionHandle::task_revision`].
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Task {
    pub task_id: TaskId,
    pub session_id: SessionId,
    pub goal: String,
    /// Goal + project-derived required checks, seeded once when first seen.
    pub acceptance_criteria: Vec<String>,
    /// Ordered steps; append-only durable.
    pub plan: Vec<String>,
    pub budget: TaskBudget,
    pub state: TaskState,
    pub created_ms: i64,
    pub updated_ms: i64,
}

impl Default for Task {
    fn default() -> Self {
        Self {
            task_id: TaskId::new(1),
            session_id: SessionId::new(1),
            goal: String::new(),
            acceptance_criteria: Vec::new(),
            plan: Vec::new(),
            budget: TaskBudget::default(),
            state: TaskState::Pending,
            created_ms: 0,
            updated_ms: 0,
        }
    }
}

impl From<faktor_store::TaskRow> for Task {
    fn from(r: faktor_store::TaskRow) -> Self {
        Self {
            task_id: r.task_id,
            session_id: r.session_id,
            goal: r.goal,
            acceptance_criteria: r.acceptance_criteria,
            plan: r.plan,
            budget: TaskBudget {
                max_tokens: r.max_tokens,
                max_turns: r.max_turns,
                spent_tokens: r.spent_tokens,
                spent_turns: r.spent_turns,
            },
            state: r.state,
            created_ms: r.created_ms,
            updated_ms: r.updated_ms,
        }
    }
}

/// Build a store row for a caller-constructed revision (private: the
/// revision is never derived from a `Task`, which deliberately does not
/// carry one).
fn task_row(task: Task, revision: TaskRevision) -> faktor_store::TaskRow {
    faktor_store::TaskRow {
        task_id: task.task_id,
        session_id: task.session_id,
        goal: task.goal,
        acceptance_criteria: task.acceptance_criteria,
        plan: task.plan,
        max_tokens: task.budget.max_tokens,
        max_turns: task.budget.max_turns,
        spent_tokens: task.budget.spent_tokens,
        spent_turns: task.budget.spent_turns,
        state: task.state,
        revision,
        created_ms: task.created_ms,
        updated_ms: task.updated_ms,
    }
}

/// The session-facing view of one durable verification record (audit P0-8).
/// Records are immutable after creation except the single CAS finalize
/// `Running -> Passed|Failed` ([`SessionHandle::finalize_verification_record`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationRecord {
    pub record_id: VerificationRecordId,
    pub task_id: TaskId,
    /// The task revision this record certifies (its creation-time task
    /// revision). Completion requires the task to STILL be at this revision.
    pub revision: TaskRevision,
    pub workspace_id: WorkspaceId,
    pub worktree_id: WorktreeId,
    pub tree_hash: Option<String>,
    pub criteria: Vec<CriterionVerification>,
    pub checks: Vec<CheckExecution>,
    pub changed_files: Vec<FileStateEvidence>,
    pub unrelated_changes: Vec<String>,
    pub reviewer: Option<serde_json::Value>,
    pub status: VerificationStatus,
    pub started_ms: i64,
    pub completed_ms: Option<i64>,
}

impl From<faktor_store::VerificationRecordRow> for VerificationRecord {
    fn from(r: faktor_store::VerificationRecordRow) -> Self {
        Self {
            record_id: r.id,
            task_id: r.task_id,
            revision: r.revision,
            workspace_id: r.workspace_id,
            worktree_id: r.worktree_id,
            tree_hash: r.tree_hash,
            criteria: r.criteria,
            checks: r.checks,
            changed_files: r.changed_files,
            unrelated_changes: r.unrelated_changes,
            reviewer: r.reviewer,
            status: r.status,
            started_ms: r.started_ms,
            completed_ms: r.completed_ms,
        }
    }
}

/// Typed failure of the task machine / completion-proof operations
/// (audits P0-7/P0-8). Every rejection cause is its own variant, so
/// callers distinguish a missing record from a wrong-revision record from
/// an uncovered criterion without parsing prose. Conversions into
/// [`faktor_core::Error`] keep the crate's `?` interop with core-`Result`
/// callers (the agent runtime) intact.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TaskError {
    #[error("task {0} not found")]
    NotFound(TaskId),
    #[error("task {task_id} already exists: a task row is created once and mutated through update_task/transition_task/complete_verified_task, never recreated")]
    AlreadyExists { task_id: TaskId },
    #[error("create_task refused: state {state:?} cannot seed a task; only Pending, Planning or Running may (VerifiedComplete requires a passing verification record via complete_verified_task)")]
    IllegalCreateState { state: TaskState },
    #[error("illegal task state transition {from:?} -> {to:?}: {detail}")]
    IllegalTransition {
        from: TaskState,
        to: TaskState,
        detail: String,
    },
    #[error("completion-relevant task state {state:?} cannot be assigned through update_task; use transition_task (NeedsVerification/Verifying) or complete_verified_task (VerifiedComplete)")]
    CompletionStateViaPatch { state: TaskState },
    #[error(
        "task {task_id} is terminal ({state:?}): its row is frozen, no further mutation is legal"
    )]
    TerminalTask { task_id: TaskId, state: TaskState },
    #[error("task {task_id} revision mismatch: expected {expected}, actual {actual} (re-read the row and retry with the current revision)")]
    RevisionMismatch {
        task_id: TaskId,
        expected: TaskRevision,
        actual: TaskRevision,
    },
    #[error("task is not Verifying (actual state {actual:?}); VerifiedComplete requires Verifying plus a passing record via complete_verified_task")]
    NotVerifying { actual: TaskState },
    #[error("verification record {0} does not exist")]
    RecordNotFound(VerificationRecordId),
    #[error(
        "verification record {record} certifies task {record_task}, not task {requested_task}"
    )]
    RecordWrongTask {
        record: VerificationRecordId,
        record_task: TaskId,
        requested_task: TaskId,
    },
    #[error("verification record {record} certifies task revision {record_revision}, but the task is at revision {expected}")]
    RecordWrongRevision {
        record: VerificationRecordId,
        record_revision: TaskRevision,
        expected: TaskRevision,
    },
    #[error(
        "verification record {record} has status {status:?}; only Passed certifies completion"
    )]
    RecordNotPassed {
        record: VerificationRecordId,
        status: VerificationStatus,
    },
    #[error("verification record {record} does not cover {missing:?} (present with passed=true is required for every current acceptance criterion)")]
    CriteriaNotCovered {
        record: VerificationRecordId,
        missing: Vec<String>,
    },
    #[error("verification record {record} was certified against worktree {record_workspace}/{record_worktree}; the task's base worktree is {task_workspace}/{task_worktree}")]
    WorktreeMismatch {
        record: VerificationRecordId,
        record_workspace: WorkspaceId,
        record_worktree: WorktreeId,
        task_workspace: WorkspaceId,
        task_worktree: WorktreeId,
    },
    #[error("verification record {record} cannot be finalized: it is {current:?}; only a Running record finalizes, exactly once")]
    RecordNotFinalizable {
        record: VerificationRecordId,
        current: VerificationStatus,
    },
    #[error("input exceeds bound: {0}")]
    Oversized(String),
    #[error("malformed input: {0}")]
    Malformed(String),
    #[error("store failure: {0}")]
    Store(String),
}

impl From<faktor_store::StoreError> for TaskError {
    fn from(e: faktor_store::StoreError) -> Self {
        TaskError::Store(e.to_string())
    }
}

impl From<TaskError> for SessionError {
    fn from(e: TaskError) -> Self {
        match e {
            TaskError::NotFound(m) => SessionError::NotFound(format!("task {m}")),
            TaskError::Store(m) => SessionError::Store(faktor_store::StoreError::Conflict(m)),
            TaskError::Oversized(m) => SessionError::Oversized(m),
            TaskError::Malformed(m) => SessionError::Malformed(m),
            other => SessionError::Conflict(other.to_string()),
        }
    }
}

impl From<TaskError> for faktor_core::Error {
    fn from(e: TaskError) -> Self {
        SessionError::from(e).into()
    }
}

/// A bounded update over an existing durable Task. Every field is optional:
/// `None` keeps the row's current value, so `update_task` never clobbers a
/// field its caller did not intend to change (e.g. the runtime preserves a
/// caller-set budget when it patches the gate state).
///
/// `state` assignments are machine-checked (audit P0-7): completion-relevant
/// states (`NeedsVerification`/`Verifying`/`VerifiedComplete`) are rejected
/// with [`TaskError::CompletionStateViaPatch`], and any non-completion
/// assignment that is not a legal edge from the row's current state is
/// rejected with [`TaskError::IllegalTransition`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TaskPatch {
    pub goal: Option<String>,
    pub acceptance_criteria: Option<Vec<String>>,
    pub plan: Option<Vec<String>>,
    pub budget: Option<TaskBudget>,
    pub state: Option<TaskState>,
}

impl SessionHandle {
    /// The session's durable task identity (the adopted `task_id` on the
    /// session row; standalone sessions default to 1).
    pub fn task_id(&self) -> faktor_core::Result<TaskId> {
        Ok(self.row()?.task_id)
    }

    /// Create ONE durable task row (audit P0-7). Creation is the ONLY place
    /// a row may seed with `Pending`/`Planning`/`Running`; every other
    /// state is refused (creating a "verified" task would mint completion
    /// proof from nothing), and an existing row for `(session_id, task_id)`
    /// is refused — a task row is created once and mutated afterwards.
    /// Fresh rows start at revision 1. Oversized fields are rejected with
    /// an error BEFORE any write — never truncated silently.
    pub fn create_task(&self, task: Task) -> faktor_core::Result<Task> {
        if task.task_id.raw() == 0 || task.session_id.raw() == 0 {
            return Err(
                SessionError::Malformed("task_id and session_id must be non-zero".into()).into(),
            );
        }
        if !task.state.is_creatable() {
            return Err(faktor_core::Error::new(
                faktor_core::ErrorKind::Conflict,
                format!(
                    "create_task refused: state {:?} cannot seed a task; only Pending, Planning or Running may (VerifiedComplete requires a passing verification record via complete_verified_task)",
                    task.state
                ),
            ));
        }
        validate_task_fields(&task)?;
        if task.created_ms == 0 {
            return Err(SessionError::Malformed("created_ms must be set".into()).into());
        }
        let _guard = self.command_guard();
        let store = self.manager.store();
        if store
            .get_task(self.id, task.task_id)
            .map_err(crate::map_store_err)?
            .is_some()
        {
            return Err(faktor_core::Error::new(
                faktor_core::ErrorKind::Conflict,
                format!(
                    "create_task refused: task {} already exists for this session; a task row is created once and mutated through update_task/transition_task/complete_verified_task, never recreated",
                    task.task_id
                ),
            ));
        }
        store
            .upsert_task(&task_row(task.clone(), TaskRevision::new(1)))
            .map_err(crate::map_store_err)?;
        Ok(task)
    }

    /// Patch ONE durable task row under the task state machine (audit
    /// P0-7). Fields that validate and are present are applied; the other
    /// fields keep their current values; `created_ms` is preserved by
    /// construction.
    ///
    /// Enforcement, in order:
    /// 1. a patch whose `state` is completion-relevant
    ///    (`NeedsVerification`/`Verifying`/`VerifiedComplete`) is rejected
    ///    with [`TaskError::CompletionStateViaPatch`] — those states are
    ///    produced only by `transition_task`/`complete_verified_task`;
    /// 2. a patch whose `state` would jump an edge the machine does not
    ///    allow from the row's current state is rejected with
    ///    [`TaskError::IllegalTransition`] (self-assignment is an
    ///    idempotent no-op);
    /// 3. a patch that effectively changes the content of a TERMINAL row
    ///    (VerifiedComplete/Failed/Cancelled) is rejected with
    ///    [`TaskError::TerminalTask`].
    ///
    /// Every effective change bumps the row revision exactly once. A no-op
    /// patch (nothing actually changes — e.g. a spend heal that found
    /// nothing to heal) writes nothing and does not bump.
    pub fn update_task(&self, task_id: TaskId, patch: TaskPatch) -> Result<Task, TaskError> {
        if task_id.raw() == 0 {
            return Err(TaskError::Malformed("task_id must be non-zero".into()));
        }
        let _guard = self.command_guard();
        let store = self.manager.store();
        let row = store
            .get_task(self.id, task_id)?
            .ok_or(TaskError::NotFound(task_id))?;
        let current = Task::from(row.clone());
        let mut next = current.clone();
        if let Some(goal) = patch.goal {
            next.goal = goal;
        }
        if let Some(criteria) = patch.acceptance_criteria {
            next.acceptance_criteria = criteria;
        }
        if let Some(plan) = patch.plan {
            next.plan = plan;
        }
        if let Some(budget) = patch.budget {
            // The durable budget cap: spent counters only ever move
            // forward. A patch that would rewind spend is a corruption sign
            // (two writers cannot both gate on a rewindable counter).
            next.budget.spent_tokens = budget.spent_tokens.max(next.budget.spent_tokens);
            next.budget.spent_turns = budget.spent_turns.max(next.budget.spent_turns);
            next.budget.max_tokens = budget.max_tokens;
            next.budget.max_turns = budget.max_turns;
        }
        if let Some(state) = patch.state {
            if state.is_completion_relevant() {
                return Err(TaskError::CompletionStateViaPatch { state });
            }
            if state != row.state && !row.state.allowed_transitions().contains(&state) {
                return Err(TaskError::IllegalTransition {
                    from: row.state,
                    to: state,
                    detail: "update_task only drives ordinary machine edges; completion states (NeedsVerification/Verifying/VerifiedComplete) require transition_task/complete_verified_task".into(),
                });
            }
            next.state = state;
        }
        validate_task_fields(&next)?;
        if next == current {
            // Idempotent no-op (replay / heal): nothing to bump, no write.
            return Ok(current);
        }
        if row.state.is_terminal() {
            return Err(TaskError::TerminalTask {
                task_id,
                state: row.state,
            });
        }
        let revision = row
            .revision
            .checked_next()
            .ok_or_else(|| TaskError::Malformed("task revision overflow".into()))?;
        let mut out = task_row(next, revision);
        out.updated_ms = self.manager.now_ms();
        store.upsert_task(&out)?;
        Ok(Task::from(out))
    }

    /// Drive ONE legal machine edge (audit P0-7) — the single chokepoint
    /// for `Pending`/`Planning`/`Running`/`Waiting`/`Blocked`/`Verifying`
    /// state changes plus cancellation. `expected_revision` must equal the
    /// row's current revision (typed [`TaskError::RevisionMismatch`]
    /// otherwise); the edge must be legal from the row's current state
    /// (typed [`TaskError::IllegalTransition`]). Success bumps the revision
    /// exactly once. `proof` is refused here with a typed error: only
    /// [`SessionHandle::complete_verified_task`] consumes a verification
    /// record, and `VerifiedComplete` has no `TaskTransition` edge at all.
    pub fn transition_task(
        &self,
        task_id: TaskId,
        expected_revision: TaskRevision,
        transition: TaskTransition,
        proof: Option<VerificationRecordId>,
    ) -> Result<Task, TaskError> {
        if task_id.raw() == 0 {
            return Err(TaskError::Malformed("task_id must be non-zero".into()));
        }
        if let Some(proof) = proof {
            return Err(TaskError::Malformed(format!(
                "transition_task does not consume a proof ({proof}); a completion proof belongs to complete_verified_task, and VerifiedComplete is unreachable by transition"
            )));
        }
        let _guard = self.command_guard();
        let store = self.manager.store();
        let row = store
            .get_task(self.id, task_id)?
            .ok_or(TaskError::NotFound(task_id))?;
        if row.revision != expected_revision {
            return Err(TaskError::RevisionMismatch {
                task_id,
                expected: expected_revision,
                actual: row.revision,
            });
        }
        if !transition.legal_from(row.state) {
            return Err(TaskError::IllegalTransition {
                from: row.state,
                to: transition.to_state(),
                detail: format!(
                    "transition_task({transition:?}): the task is {:?}; re-read the row and pick a legal edge",
                    row.state
                ),
            });
        }
        let revision = expected_revision
            .checked_next()
            .ok_or_else(|| TaskError::Malformed("task revision overflow".into()))?;
        let mut out = task_row(Task::from(row), revision);
        out.state = transition.to_state();
        out.updated_ms = self.manager.now_ms();
        store.upsert_task(&out)?;
        Ok(Task::from(out))
    }

    /// Complete a verified task (audit P0-7/P0-8) — the ONLY path to
    /// `VerifiedComplete`. The whole proof validation runs inside ONE store
    /// transaction: (a) the task is `Verifying` (a `NeedsVerification` task
    /// must transition to `Verifying` first — completion never skips the
    /// verifier), (b) the record exists, (c) it certifies THIS task,
    /// (d) it certifies exactly `expected_revision` == the task's current
    /// revision, (e) its status is `Passed`, (f) it covers every current
    /// acceptance criterion of the task (extra record criteria are fine),
    /// (g) its workspace/worktree equal the task's current base worktree.
    /// Success writes `VerifiedComplete` and bumps the revision exactly
    /// once. Every rejection is a distinct typed [`TaskError`] and leaves
    /// the task row untouched.
    pub fn complete_verified_task(
        &self,
        task_id: TaskId,
        expected_revision: TaskRevision,
        proof: VerificationRecordId,
    ) -> Result<Task, TaskError> {
        if task_id.raw() == 0 {
            return Err(TaskError::Malformed("task_id must be non-zero".into()));
        }
        let _guard = self.command_guard();
        let outcome = self.manager.store().task_complete_verified(
            self.id,
            task_id,
            expected_revision,
            proof,
            self.manager.now_ms(),
        )?;
        match outcome {
            Ok(row) => Ok(Task::from(row)),
            Err(refusal) => Err(match refusal {
                faktor_store::TaskCompletionRefusal::TaskMissing { .. } => {
                    TaskError::NotFound(task_id)
                }
                faktor_store::TaskCompletionRefusal::RevisionMismatch { expected, actual } => {
                    TaskError::RevisionMismatch {
                        task_id,
                        expected,
                        actual,
                    }
                }
                faktor_store::TaskCompletionRefusal::NotVerifying { actual } => {
                    TaskError::NotVerifying { actual }
                }
                faktor_store::TaskCompletionRefusal::RecordMissing { record_id } => {
                    TaskError::RecordNotFound(record_id)
                }
                faktor_store::TaskCompletionRefusal::RecordWrongTask {
                    record_id,
                    record_task,
                    ..
                } => TaskError::RecordWrongTask {
                    record: record_id,
                    record_task,
                    requested_task: task_id,
                },
                faktor_store::TaskCompletionRefusal::RecordWrongRevision {
                    record_id,
                    record_revision,
                    ..
                } => TaskError::RecordWrongRevision {
                    record: record_id,
                    record_revision,
                    expected: expected_revision,
                },
                faktor_store::TaskCompletionRefusal::RecordNotPassed { record_id, status } => {
                    TaskError::RecordNotPassed {
                        record: record_id,
                        status,
                    }
                }
                faktor_store::TaskCompletionRefusal::CriteriaNotCovered { record_id, missing } => {
                    TaskError::CriteriaNotCovered {
                        record: record_id,
                        missing,
                    }
                }
                faktor_store::TaskCompletionRefusal::WorktreeMismatch {
                    record_id,
                    record_workspace,
                    record_worktree,
                    task_workspace,
                    task_worktree,
                } => TaskError::WorktreeMismatch {
                    record: record_id,
                    record_workspace,
                    record_worktree,
                    task_workspace,
                    task_worktree,
                },
            }),
        }
    }

    /// The row's current revision — the `expected_revision` a transition
    /// or completion must be called with.
    pub fn task_revision(&self, task_id: TaskId) -> Result<TaskRevision, TaskError> {
        let row = self
            .manager
            .store()
            .get_task(self.id, task_id)?
            .ok_or(TaskError::NotFound(task_id))?;
        Ok(row.revision)
    }

    /// The durable task row identified by `task_id` (session-scoped).
    pub fn get_task(&self, task_id: TaskId) -> faktor_core::Result<Option<Task>> {
        self.manager
            .store()
            .get_task(self.id, task_id)
            .map_err(|e| crate::map_store_err(e).into())
            .map(|r| r.map(Task::from))
    }

    /// Every durable task row of this session (oldest-created first).
    pub fn list_tasks(&self) -> faktor_core::Result<Vec<Task>> {
        self.manager
            .store()
            .list_tasks(self.id)
            .map_err(|e| crate::map_store_err(e).into())
            .map(|rows| rows.into_iter().map(Task::from).collect())
    }

    /// Crash-safe token spend of the session: the durable sum of every
    /// recorded provider call (input + output tokens).
    pub fn spent_tokens(&self) -> faktor_core::Result<u64> {
        self.manager
            .store()
            .session_usage_tokens(self.id)
            .map_err(|e| crate::map_store_err(e).into())
    }

    /// Crash-safe logical-turn count of the session: durable
    /// `turn_completed` journal events.
    pub fn spent_turns(&self) -> faktor_core::Result<u64> {
        self.manager
            .store()
            .turn_completed_count(self.id)
            .map_err(|e| crate::map_store_err(e).into())
    }

    /// The configured wall-clock budget of one logical turn (ms; 0 =
    /// unbounded). The runtime caps every turn slice with this value.
    pub fn turn_budget_ms(&self) -> u64 {
        self.manager.turn_budget_ms()
    }

    // -------------------------------------------------- verification records

    /// Create one durable verification record certifying `task_id` at its
    /// CURRENT revision (audit P0-8). The record is written once and is
    /// immutable afterwards except the single CAS finalize
    /// (`Running -> Passed|Failed`); a record created directly as `Passed`
    /// is complete from birth. Every bound is enforced BEFORE the write —
    /// oversized criteria/checks/evidence JSON is rejected with a typed
    /// [`TaskError::Oversized`], never truncated.
    #[allow(clippy::too_many_arguments)]
    pub fn create_verification_record(
        &self,
        task_id: TaskId,
        tree_hash: Option<String>,
        criteria: Vec<CriterionVerification>,
        checks: Vec<CheckExecution>,
        changed_files: Vec<FileStateEvidence>,
        unrelated_changes: Vec<String>,
        reviewer: Option<serde_json::Value>,
        status: VerificationStatus,
        started_ms: i64,
    ) -> Result<VerificationRecordId, TaskError> {
        if task_id.raw() == 0 {
            return Err(TaskError::Malformed("task_id must be non-zero".into()));
        }
        validate_verification_record(
            &criteria,
            &checks,
            &changed_files,
            &unrelated_changes,
            reviewer.as_ref(),
            tree_hash.as_deref(),
        )?;
        let _guard = self.command_guard();
        let store = self.manager.store();
        let task = store
            .get_task(self.id, task_id)?
            .ok_or(TaskError::NotFound(task_id))?;
        let session = store
            .get_session(self.id)?
            .ok_or(TaskError::NotFound(task_id))?;
        let rec = faktor_store::VerificationRecordRow {
            id: VerificationRecordId::new(1), // ignored by put; a fresh id is minted
            task_id,
            revision: task.revision,
            workspace_id: session.workspace_id,
            worktree_id: session.worktree_id,
            tree_hash,
            criteria,
            checks,
            changed_files,
            unrelated_changes,
            reviewer,
            status,
            started_ms,
            completed_ms: None,
        };
        Ok(store.verification_record_put(&rec)?)
    }

    /// One verification record by id, or `None`.
    pub fn get_verification_record(
        &self,
        record_id: VerificationRecordId,
    ) -> Result<Option<VerificationRecord>, TaskError> {
        self.manager
            .store()
            .verification_record_get(record_id)
            .map(|r| r.map(VerificationRecord::from))
            .map_err(TaskError::from)
    }

    /// Every verification record of `task_id`, in deterministic creation
    /// order (record id ascending).
    ///
    /// NOTE: `verification_record` rows are keyed by the NUMERIC task id
    /// (the approved schema carries no session column), so records of a
    /// standalone session (task id 1) of another session in the same
    /// workspace are visible here. Completion itself is protected by the
    /// record's workspace/worktree content check against the completing
    /// session's base worktree.
    pub fn list_verification_records(
        &self,
        task_id: TaskId,
    ) -> Result<Vec<VerificationRecord>, TaskError> {
        self.manager
            .store()
            .verification_record_list_by_task(task_id)
            .map(|rows| rows.into_iter().map(VerificationRecord::from).collect())
            .map_err(TaskError::from)
    }

    /// The record's single allowed status write (audit P0-8): a CAS from
    /// `Running` to `Passed` or `Failed`, written exactly once. A second
    /// completion attempt on an already-final record is a typed
    /// [`TaskError::RecordNotFinalizable`] carrying the current status.
    pub fn finalize_verification_record(
        &self,
        record_id: VerificationRecordId,
        status: VerificationStatus,
        completed_ms: i64,
    ) -> Result<(), TaskError> {
        if !matches!(
            status,
            VerificationStatus::Passed | VerificationStatus::Failed
        ) {
            return Err(TaskError::Malformed(format!(
                "finalize status must be Passed or Failed, got {status:?}"
            )));
        }
        let _guard = self.command_guard();
        match self
            .manager
            .store()
            .verification_record_finalize(record_id, status, completed_ms)?
        {
            Ok(()) => Ok(()),
            Err(faktor_store::RecordFinalizeRefusal::Missing { .. }) => {
                Err(TaskError::RecordNotFound(record_id))
            }
            Err(faktor_store::RecordFinalizeRefusal::NotRunning { record_id, current }) => {
                Err(TaskError::RecordNotFinalizable {
                    record: record_id,
                    current,
                })
            }
        }
    }
}

fn validate_task_fields(t: &Task) -> Result<(), TaskError> {
    if t.goal.len() > MAX_TASK_GOAL_BYTES {
        return Err(TaskError::Oversized(format!(
            "task goal of {} bytes exceeds MAX_TASK_GOAL_BYTES ({MAX_TASK_GOAL_BYTES})",
            t.goal.len()
        )));
    }
    if t.acceptance_criteria.len() > MAX_TASK_CRITERIA {
        return Err(TaskError::Oversized(format!(
            "{} acceptance criteria exceed MAX_TASK_CRITERIA ({MAX_TASK_CRITERIA})",
            t.acceptance_criteria.len()
        )));
    }
    for c in &t.acceptance_criteria {
        if c.len() > MAX_TASK_CRITERION_BYTES {
            return Err(TaskError::Oversized(format!(
                "a criterion of {} bytes exceeds MAX_TASK_CRITERION_BYTES ({MAX_TASK_CRITERION_BYTES})",
                c.len()
            )));
        }
    }
    if t.plan.len() > MAX_TASK_PLAN_STEPS {
        return Err(TaskError::Oversized(format!(
            "{} plan steps exceed MAX_TASK_PLAN_STEPS ({MAX_TASK_PLAN_STEPS})",
            t.plan.len()
        )));
    }
    for s in &t.plan {
        if s.len() > MAX_TASK_STEP_BYTES {
            return Err(TaskError::Oversized(format!(
                "a plan step of {} bytes exceeds MAX_TASK_STEP_BYTES ({MAX_TASK_STEP_BYTES})",
                s.len()
            )));
        }
    }
    Ok(())
}

/// Bounded-field contract of one verification record (audit P0-8): every
/// bound is enforced before ANY write; oversized input is rejected with a
/// typed error, never truncated.
#[allow(clippy::too_many_lines)]
fn validate_verification_record(
    criteria: &[CriterionVerification],
    checks: &[CheckExecution],
    changed_files: &[FileStateEvidence],
    unrelated_changes: &[String],
    reviewer: Option<&serde_json::Value>,
    tree_hash: Option<&str>,
) -> Result<(), TaskError> {
    let reject = |what: &str| TaskError::Oversized(what.to_string());
    let malformed = |what: String| TaskError::Malformed(what);
    if criteria.len() > MAX_VERIFICATION_RECORD_CRITERIA {
        return Err(reject(&format!(
            "{} criterion verdicts exceed MAX_VERIFICATION_RECORD_CRITERIA ({MAX_VERIFICATION_RECORD_CRITERIA})",
            criteria.len()
        )));
    }
    for c in criteria {
        if c.criterion_key.is_empty() {
            return Err(malformed(
                "a criterion verdict has an empty criterion_key".into(),
            ));
        }
        if c.criterion_key.len() > MAX_VERIFICATION_CRITERION_KEY_BYTES {
            return Err(reject(&format!(
                "a criterion key of {} bytes exceeds MAX_VERIFICATION_CRITERION_KEY_BYTES ({MAX_VERIFICATION_CRITERION_KEY_BYTES})",
                c.criterion_key.len()
            )));
        }
        if let Some(evidence) = &c.evidence {
            if evidence.len() > MAX_VERIFICATION_EVIDENCE_BYTES {
                return Err(reject(&format!(
                    "criterion evidence of {} bytes exceeds MAX_VERIFICATION_EVIDENCE_BYTES ({MAX_VERIFICATION_EVIDENCE_BYTES})",
                    evidence.len()
                )));
            }
        }
    }
    let criteria_json =
        serde_json::to_vec(criteria).map_err(|e| malformed(format!("criteria json: {e}")))?;
    if criteria_json.len() > MAX_VERIFICATION_CRITERIA_JSON_BYTES {
        return Err(reject(&format!(
            "criteria JSON of {} bytes exceeds MAX_VERIFICATION_CRITERIA_JSON_BYTES ({MAX_VERIFICATION_CRITERIA_JSON_BYTES})",
            criteria_json.len()
        )));
    }
    if checks.len() > MAX_VERIFICATION_RECORD_CHECKS {
        return Err(reject(&format!(
            "{} checks exceed MAX_VERIFICATION_RECORD_CHECKS ({MAX_VERIFICATION_RECORD_CHECKS})",
            checks.len()
        )));
    }
    for ch in checks {
        if ch.check.is_empty() || ch.check.len() > MAX_VERIFICATION_CHECK_NAME_BYTES {
            return Err(reject(&format!(
                "check name {} exceeds MAX_VERIFICATION_CHECK_NAME_BYTES ({MAX_VERIFICATION_CHECK_NAME_BYTES})",
                ch.check.len()
            )));
        }
        if ch.program.is_empty() || ch.program.len() > MAX_VERIFICATION_PROGRAM_BYTES {
            return Err(reject(&format!(
                "check program {} exceeds MAX_VERIFICATION_PROGRAM_BYTES ({MAX_VERIFICATION_PROGRAM_BYTES})",
                ch.program.len()
            )));
        }
        if ch.args.len() > MAX_VERIFICATION_CHECK_ARGS {
            return Err(reject(&format!(
                "{} check args exceed MAX_VERIFICATION_CHECK_ARGS ({MAX_VERIFICATION_CHECK_ARGS})",
                ch.args.len()
            )));
        }
        for a in &ch.args {
            if a.len() > MAX_VERIFICATION_CHECK_ARG_BYTES {
                return Err(reject(&format!(
                    "a check arg of {} bytes exceeds MAX_VERIFICATION_CHECK_ARG_BYTES ({MAX_VERIFICATION_CHECK_ARG_BYTES})",
                    a.len()
                )));
            }
        }
        if ch.category.len() > MAX_VERIFICATION_CATEGORY_BYTES {
            return Err(reject(&format!(
                "check category {} exceeds MAX_VERIFICATION_CATEGORY_BYTES ({MAX_VERIFICATION_CATEGORY_BYTES})",
                ch.category.len()
            )));
        }
        if let Some(summary) = &ch.summary {
            if summary.len() > MAX_VERIFICATION_SUMMARY_BYTES {
                return Err(reject(&format!(
                    "check summary of {} bytes exceeds MAX_VERIFICATION_SUMMARY_BYTES ({MAX_VERIFICATION_SUMMARY_BYTES})",
                    summary.len()
                )));
            }
        }
    }
    let checks_json =
        serde_json::to_vec(checks).map_err(|e| malformed(format!("checks json: {e}")))?;
    if checks_json.len() > MAX_VERIFICATION_CHECKS_JSON_BYTES {
        return Err(reject(&format!(
            "checks JSON of {} bytes exceeds MAX_VERIFICATION_CHECKS_JSON_BYTES ({MAX_VERIFICATION_CHECKS_JSON_BYTES})",
            checks_json.len()
        )));
    }
    if changed_files.len() > MAX_VERIFICATION_CHANGED_FILES {
        return Err(reject(&format!(
            "{} changed files exceed MAX_VERIFICATION_CHANGED_FILES ({MAX_VERIFICATION_CHANGED_FILES})",
            changed_files.len()
        )));
    }
    for f in changed_files {
        if f.path.is_empty() || f.path.len() > MAX_VERIFICATION_PATH_BYTES {
            return Err(reject(&format!(
                "file path of {} bytes exceeds MAX_VERIFICATION_PATH_BYTES ({MAX_VERIFICATION_PATH_BYTES})",
                f.path.len()
            )));
        }
        if f.digest_hex.is_empty()
            || f.digest_hex.len() > MAX_VERIFICATION_DIGEST_BYTES
            || !f.digest_hex.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err(malformed(format!(
                "file {} digest {:?} must be non-empty lowercase/uppercase hex of at most {MAX_VERIFICATION_DIGEST_BYTES} chars",
                f.path, f.digest_hex
            )));
        }
    }
    let files_json = serde_json::to_vec(changed_files)
        .map_err(|e| malformed(format!("changed_files json: {e}")))?;
    if files_json.len() > MAX_VERIFICATION_CHANGED_FILES_JSON_BYTES {
        return Err(reject(&format!(
            "changed_files JSON of {} bytes exceeds MAX_VERIFICATION_CHANGED_FILES_JSON_BYTES ({MAX_VERIFICATION_CHANGED_FILES_JSON_BYTES})",
            files_json.len()
        )));
    }
    if unrelated_changes.len() > MAX_VERIFICATION_UNRELATED_CHANGES {
        return Err(reject(&format!(
            "{} unrelated changes exceed MAX_VERIFICATION_UNRELATED_CHANGES ({MAX_VERIFICATION_UNRELATED_CHANGES})",
            unrelated_changes.len()
        )));
    }
    for u in unrelated_changes {
        if u.is_empty() || u.len() > MAX_VERIFICATION_PATH_BYTES {
            return Err(reject(&format!(
                "an unrelated-change path of {} bytes exceeds MAX_VERIFICATION_PATH_BYTES ({MAX_VERIFICATION_PATH_BYTES})",
                u.len()
            )));
        }
    }
    let unrelated_json = serde_json::to_vec(unrelated_changes)
        .map_err(|e| malformed(format!("unrelated_changes json: {e}")))?;
    if unrelated_json.len() > MAX_VERIFICATION_UNRELATED_JSON_BYTES {
        return Err(reject(&format!(
            "unrelated_changes JSON of {} bytes exceeds MAX_VERIFICATION_UNRELATED_JSON_BYTES ({MAX_VERIFICATION_UNRELATED_JSON_BYTES})",
            unrelated_json.len()
        )));
    }
    if let Some(reviewer) = reviewer {
        let len = serde_json::to_vec(reviewer)
            .map_err(|e| malformed(format!("reviewer json: {e}")))?
            .len();
        if len > MAX_VERIFICATION_REVIEWER_JSON_BYTES {
            return Err(reject(&format!(
                "reviewer JSON of {len} bytes exceeds MAX_VERIFICATION_REVIEWER_JSON_BYTES ({MAX_VERIFICATION_REVIEWER_JSON_BYTES})"
            )));
        }
    }
    if let Some(hash) = tree_hash {
        if hash.is_empty()
            || hash.len() > MAX_VERIFICATION_TREE_HASH_BYTES
            || !hash.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err(malformed(format!(
                "tree_hash {:?} must be non-empty hex of at most {MAX_VERIFICATION_TREE_HASH_BYTES} chars",
                hash
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::tests::{session, test_manager};
    use faktor_core::ErrorKind;
    use std::sync::Arc;
    use std::thread;

    fn task(s: &SessionHandle) -> Task {
        Task {
            task_id: s.task_id().unwrap(),
            session_id: s.id,
            goal: "implement durable tasks".into(),
            acceptance_criteria: vec!["goal: implement durable tasks".into()],
            plan: vec!["schema".into(), "repo".into()],
            budget: TaskBudget {
                max_tokens: Some(100_000),
                max_turns: Some(10),
                spent_tokens: 0,
                spent_turns: 0,
            },
            state: TaskState::Pending,
            created_ms: 1,
            updated_ms: 1,
        }
    }

    fn criteria_task(s: &SessionHandle, task_id: TaskId, criteria: Vec<String>) -> Task {
        Task {
            task_id,
            session_id: s.id,
            goal: "gated goal".into(),
            acceptance_criteria: criteria,
            plan: vec![],
            budget: TaskBudget::default(),
            state: TaskState::Pending,
            created_ms: 1,
            updated_ms: 1,
        }
    }

    /// Drive a task Pending -> Running -> NeedsVerification -> Verifying
    /// through legal transitions; returns the revision at Verifying.
    fn drive_to_verifying(s: &SessionHandle, task_id: TaskId) -> TaskRevision {
        let r1 = s.task_revision(task_id).unwrap();
        s.transition_task(task_id, r1, TaskTransition::StartRunning, None)
            .unwrap();
        let r2 = s.task_revision(task_id).unwrap();
        s.transition_task(task_id, r2, TaskTransition::RequestVerification, None)
            .unwrap();
        let r3 = s.task_revision(task_id).unwrap();
        s.transition_task(task_id, r3, TaskTransition::StartVerification, None)
            .unwrap();
        s.task_revision(task_id).unwrap()
    }

    fn passed_record(
        s: &SessionHandle,
        task_id: TaskId,
        criteria: &[String],
    ) -> VerificationRecordId {
        let criteria: Vec<CriterionVerification> = criteria
            .iter()
            .map(|c| CriterionVerification {
                criterion_key: c.clone(),
                passed: true,
                evidence: Some("exit 0".into()),
            })
            .collect();
        s.create_verification_record(
            task_id,
            None,
            criteria,
            vec![],
            vec![],
            vec![],
            None,
            VerificationStatus::Passed,
            1,
        )
        .unwrap()
    }

    #[test]
    fn task_create_get_update_list_roundtrip() {
        let (_d, m) = test_manager();
        let s = session(&m);
        assert!(s.list_tasks().unwrap().is_empty());
        let t = task(&s);
        let created = s.create_task(t.clone()).unwrap();
        assert_eq!(created.state, TaskState::Pending);
        assert_eq!(s.get_task(created.task_id).unwrap(), Some(created.clone()));
        assert_eq!(
            s.task_revision(created.task_id).unwrap(),
            TaskRevision::new(1)
        );
        // Patch: state + spend move forward; untouched fields survive; the
        // effective change bumps the revision exactly once.
        let patched = s
            .update_task(
                created.task_id,
                TaskPatch {
                    state: Some(TaskState::Running),
                    budget: Some(TaskBudget {
                        max_tokens: Some(100_000),
                        max_turns: Some(10),
                        spent_tokens: 40,
                        spent_turns: 1,
                    }),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(patched.state, TaskState::Running);
        assert_eq!(patched.budget.spent_tokens, 40);
        assert_eq!(patched.goal, created.goal, "unpatched fields survive");
        assert_eq!(patched.created_ms, created.created_ms);
        assert_eq!(
            s.task_revision(created.task_id).unwrap(),
            TaskRevision::new(2)
        );
        assert_eq!(s.get_task(created.task_id).unwrap(), Some(patched.clone()));
        assert_eq!(s.list_tasks().unwrap().len(), 1);
    }

    #[test]
    fn update_on_missing_task_is_not_found() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let err = s
            .update_task(TaskId::new(999), TaskPatch::default())
            .unwrap_err();
        assert_eq!(err, TaskError::NotFound(TaskId::new(999)));
    }

    #[test]
    fn oversized_goal_criteria_and_plan_are_rejected_never_truncated() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let mut t = task(&s);
        t.goal = "g".repeat(MAX_TASK_GOAL_BYTES + 1);
        assert!(matches!(
            s.create_task(t.clone()).unwrap_err().kind,
            ErrorKind::Oversized
        ));
        t.goal = "ok".into();
        t.acceptance_criteria = (0..=MAX_TASK_CRITERIA)
            .map(|i| format!("criterion {i}"))
            .collect();
        assert!(matches!(
            s.create_task(t.clone()).unwrap_err().kind,
            ErrorKind::Oversized
        ));
        t.acceptance_criteria = vec!["c".repeat(MAX_TASK_CRITERION_BYTES + 1)];
        assert!(matches!(
            s.create_task(t.clone()).unwrap_err().kind,
            ErrorKind::Oversized
        ));
        t.acceptance_criteria = vec!["c".into()];
        t.plan = (0..=MAX_TASK_PLAN_STEPS)
            .map(|i| format!("step {i}"))
            .collect();
        assert!(matches!(
            s.create_task(t.clone()).unwrap_err().kind,
            ErrorKind::Oversized
        ));
        // Rejection leaves NO trace: the store stays empty, and the
        // rejected values were never silently truncated into the row.
        assert!(s.list_tasks().unwrap().is_empty());
    }

    #[test]
    fn oversized_patch_fields_are_rejected() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let created = s.create_task(task(&s)).unwrap();
        let err = s
            .update_task(
                created.task_id,
                TaskPatch {
                    goal: Some("x".repeat(MAX_TASK_GOAL_BYTES + 1)),
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(matches!(err, TaskError::Oversized(_)));
        let row = s.get_task(created.task_id).unwrap().unwrap();
        assert_eq!(row.goal, created.goal, "rejected patch left no trace");
        let err = s
            .update_task(
                created.task_id,
                TaskPatch {
                    plan: Some((0..=MAX_TASK_PLAN_STEPS).map(|i| format!("s{i}")).collect()),
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(matches!(err, TaskError::Oversized(_)));
        // A no-op patch (nothing changes) writes nothing and bumps nothing.
        let rev_before = s.task_revision(created.task_id).unwrap();
        let noop = s
            .update_task(created.task_id, TaskPatch::default())
            .unwrap();
        assert_eq!(noop, created);
        assert_eq!(s.task_revision(created.task_id).unwrap(), rev_before);
    }

    #[test]
    fn budget_spend_only_moves_forward() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let created = s.create_task(task(&s)).unwrap();
        let bump = |spent_tokens: u64| TaskPatch {
            budget: Some(TaskBudget {
                max_tokens: Some(100_000),
                max_turns: Some(10),
                spent_tokens,
                spent_turns: 3,
            }),
            ..Default::default()
        };
        let p1 = s.update_task(created.task_id, bump(50)).unwrap();
        assert_eq!(p1.budget.spent_tokens, 50);
        // A rewind attempt is refused by construction: the effective content
        // is unchanged, so NOTHING is written and the revision does not move.
        let rev = s.task_revision(created.task_id).unwrap();
        let p2 = s.update_task(created.task_id, bump(10)).unwrap();
        assert_eq!(p2.budget.spent_tokens, 50, "spend is monotone");
        assert_eq!(s.task_revision(created.task_id).unwrap(), rev);
        // max fields DO update (they are not counters) and bump.
        let p3 = s
            .update_task(
                created.task_id,
                TaskPatch {
                    budget: Some(TaskBudget {
                        max_tokens: Some(5),
                        max_turns: Some(1),
                        spent_tokens: 0,
                        spent_turns: 0,
                    }),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(p3.budget.max_tokens, Some(5));
        assert_eq!(p3.budget.spent_tokens, 50);
        assert_eq!(
            s.task_revision(created.task_id).unwrap(),
            TaskRevision::new(3)
        );
    }

    #[test]
    fn tasks_are_session_scoped() {
        let (_d, m) = test_manager();
        let s1 = session(&m);
        let s2 = {
            let ws = m.create_workspace("/w2").unwrap();
            m.create_session(ws, "t2", "p", "m").unwrap()
        };
        let t1 = s1.create_task(task(&s1)).unwrap();
        assert!(s2.list_tasks().unwrap().is_empty());
        assert!(s2.get_task(t1.task_id).unwrap().is_none());
    }

    // (a) the direct patch cannot reach completion states or jump edges.
    #[test]
    fn patch_cannot_reach_completion_states_or_jump_edges() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let created = s.create_task(task(&s)).unwrap();
        for forbidden in [
            TaskState::VerifiedComplete,
            TaskState::Verifying,
            TaskState::NeedsVerification,
        ] {
            let err = s
                .update_task(
                    created.task_id,
                    TaskPatch {
                        state: Some(forbidden),
                        ..Default::default()
                    },
                )
                .unwrap_err();
            assert_eq!(
                err,
                TaskError::CompletionStateViaPatch { state: forbidden },
                "{forbidden:?} must be patch-unreachable"
            );
        }
        let row = s.get_task(created.task_id).unwrap().unwrap();
        assert_eq!(row.state, TaskState::Pending, "state unchanged");
        assert_eq!(
            s.task_revision(created.task_id).unwrap(),
            TaskRevision::new(1)
        );
        // A non-completion edge the machine forbids (Pending -> Failed).
        let err = s
            .update_task(
                created.task_id,
                TaskPatch {
                    state: Some(TaskState::Failed),
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(matches!(
            err,
            TaskError::IllegalTransition {
                from: TaskState::Pending,
                to: TaskState::Failed,
                ..
            }
        ));
        assert_eq!(
            s.task_revision(created.task_id).unwrap(),
            TaskRevision::new(1)
        );
        // Legal ordinary edges apply and bump exactly once.
        s.update_task(
            created.task_id,
            TaskPatch {
                state: Some(TaskState::Running),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            s.task_revision(created.task_id).unwrap(),
            TaskRevision::new(2)
        );
    }

    // (b) create_task admits only Pending/Planning/Running and never
    // recreates an existing row.
    #[test]
    fn create_task_rejects_uncreatable_states_and_recreation() {
        let (_d, m) = test_manager();
        let s = session(&m);
        for forbidden in [
            TaskState::VerifiedComplete,
            TaskState::Verifying,
            TaskState::NeedsVerification,
            TaskState::Failed,
            TaskState::Cancelled,
            TaskState::Waiting,
            TaskState::Blocked,
        ] {
            let mut t = task(&s);
            t.state = forbidden;
            let err = s.create_task(t.clone()).unwrap_err();
            assert_eq!(err.kind, ErrorKind::Conflict);
            assert!(
                err.message.contains("create_task refused"),
                "{forbidden:?}: {}",
                err.message
            );
        }
        assert!(
            s.list_tasks().unwrap().is_empty(),
            "no trace of refused creates"
        );
        let t = task(&s);
        let created = s.create_task(t.clone()).unwrap();
        assert_eq!(created.state, TaskState::Pending);
        let again = s.create_task(t.clone()).unwrap_err();
        assert_eq!(again.kind, ErrorKind::Conflict);
        assert!(again.message.contains("already exists"));
        assert_eq!(s.list_tasks().unwrap().len(), 1);
        assert_eq!(
            s.task_revision(created.task_id).unwrap(),
            TaskRevision::new(1)
        );
    }

    // (c)+(d) the completion transaction: typed refusals for every broken
    // proof facet, state and revision untouched, happy path bumps exactly
    // once, and VerifiedComplete is terminal.
    #[test]
    fn completion_requires_proof_records_revision_criteria_and_worktree() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let main = s
            .create_task(criteria_task(
                &s,
                s.task_id().unwrap(),
                vec!["c1".into(), "c2".into()],
            ))
            .unwrap();
        let t1 = main.task_id;
        let rev_at_verifying = drive_to_verifying(&s, t1);
        let record = passed_record(&s, t1, &["c1".into(), "c2".into()]);
        let done = s
            .complete_verified_task(t1, rev_at_verifying, record)
            .unwrap();
        assert_eq!(done.state, TaskState::VerifiedComplete);
        assert_eq!(
            s.task_revision(t1).unwrap(),
            rev_at_verifying.checked_next().unwrap(),
            "revision bumped exactly once"
        );
        // VerifiedComplete is terminal: content edits are frozen, no
        // transition is legal, a second completion refuses.
        let frozen = s
            .update_task(
                t1,
                TaskPatch {
                    goal: Some("tamper".into()),
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert_eq!(
            frozen,
            TaskError::TerminalTask {
                task_id: t1,
                state: TaskState::VerifiedComplete
            }
        );
        let rev_done = s.task_revision(t1).unwrap();
        let frozen2 = s
            .transition_task(t1, rev_done, TaskTransition::Cancel, None)
            .unwrap_err();
        assert!(matches!(frozen2, TaskError::IllegalTransition { .. }));
        let frozen3 = s.complete_verified_task(t1, rev_done, record).unwrap_err();
        assert_eq!(
            frozen3,
            TaskError::NotVerifying {
                actual: TaskState::VerifiedComplete
            }
        );
        assert_eq!(s.task_revision(t1).unwrap(), rev_done, "no double bump");

        let keep_verifying = |s: &SessionHandle, id: TaskId, rev: TaskRevision| {
            let row = s.get_task(id).unwrap().unwrap();
            assert_eq!(row.state, TaskState::Verifying);
            assert_eq!(s.task_revision(id).unwrap(), rev, "refusal must not bump");
        };

        // Missing record.
        let t2 = s
            .create_task(criteria_task(&s, TaskId::new(2), vec!["c1".into()]))
            .unwrap();
        let r2 = drive_to_verifying(&s, t2.task_id);
        let err = s
            .complete_verified_task(t2.task_id, r2, VerificationRecordId::new(42_000))
            .unwrap_err();
        assert_eq!(
            err,
            TaskError::RecordNotFound(VerificationRecordId::new(42_000))
        );
        keep_verifying(&s, t2.task_id, r2);
        // Wrong task: the record certifies task 1, not task 2.
        let err = s
            .complete_verified_task(t2.task_id, r2, record)
            .unwrap_err();
        assert_eq!(
            err,
            TaskError::RecordWrongTask {
                record,
                record_task: t1,
                requested_task: t2.task_id
            }
        );
        keep_verifying(&s, t2.task_id, r2);

        // Wrong revision: a record certifies the revision before a legal
        // content bump (budget while Verifying) and cannot complete the
        // moved task.
        let t3 = s
            .create_task(criteria_task(&s, TaskId::new(3), vec!["c1".into()]))
            .unwrap();
        let r3 = drive_to_verifying(&s, t3.task_id);
        let rec3 = passed_record(&s, t3.task_id, &["c1".into()]);
        s.update_task(
            t3.task_id,
            TaskPatch {
                budget: Some(TaskBudget {
                    max_tokens: Some(1),
                    max_turns: None,
                    spent_tokens: 0,
                    spent_turns: 0,
                }),
                ..Default::default()
            },
        )
        .unwrap();
        let r3_after = s.task_revision(t3.task_id).unwrap();
        assert_ne!(r3_after, r3);
        let err = s
            .complete_verified_task(t3.task_id, r3_after, rec3)
            .unwrap_err();
        assert_eq!(
            err,
            TaskError::RecordWrongRevision {
                record: rec3,
                record_revision: r3,
                expected: r3_after
            }
        );
        keep_verifying(&s, t3.task_id, r3_after);
        // The STALE expected revision reports the task-side mismatch first.
        let err = s.complete_verified_task(t3.task_id, r3, rec3).unwrap_err();
        assert_eq!(
            err,
            TaskError::RevisionMismatch {
                task_id: t3.task_id,
                expected: r3,
                actual: r3_after
            }
        );
        keep_verifying(&s, t3.task_id, r3_after);

        // Record status Failed refuses with full coverage.
        let t4 = s
            .create_task(criteria_task(&s, TaskId::new(4), vec!["c1".into()]))
            .unwrap();
        let r4 = drive_to_verifying(&s, t4.task_id);
        let failed_rec = s
            .create_verification_record(
                t4.task_id,
                None,
                vec![CriterionVerification {
                    criterion_key: "c1".into(),
                    passed: true,
                    evidence: None,
                }],
                vec![],
                vec![],
                vec![],
                None,
                VerificationStatus::Failed,
                1,
            )
            .unwrap();
        let err = s
            .complete_verified_task(t4.task_id, r4, failed_rec)
            .unwrap_err();
        assert_eq!(
            err,
            TaskError::RecordNotPassed {
                record: failed_rec,
                status: VerificationStatus::Failed
            }
        );
        keep_verifying(&s, t4.task_id, r4);

        // A record missing one current criterion refuses with the missing
        // list; full coverage completes; a passed=false entry is NOT
        // coverage.
        let t5 = s
            .create_task(criteria_task(
                &s,
                TaskId::new(5),
                vec!["c1".into(), "c2".into()],
            ))
            .unwrap();
        let r5 = drive_to_verifying(&s, t5.task_id);
        let partial = passed_record(&s, t5.task_id, &["c1".into()]);
        let err = s
            .complete_verified_task(t5.task_id, r5, partial)
            .unwrap_err();
        assert_eq!(
            err,
            TaskError::CriteriaNotCovered {
                record: partial,
                missing: vec!["c2".to_string()]
            }
        );
        keep_verifying(&s, t5.task_id, r5);
        let full = passed_record(&s, t5.task_id, &["c1".into(), "c2".into()]);
        let done5 = s.complete_verified_task(t5.task_id, r5, full).unwrap();
        assert_eq!(done5.state, TaskState::VerifiedComplete);
        let t6 = s
            .create_task(criteria_task(&s, TaskId::new(6), vec!["c1".into()]))
            .unwrap();
        let r6 = drive_to_verifying(&s, t6.task_id);
        let lying = s
            .create_verification_record(
                t6.task_id,
                None,
                vec![CriterionVerification {
                    criterion_key: "c1".into(),
                    passed: false,
                    evidence: None,
                }],
                vec![],
                vec![],
                vec![],
                None,
                VerificationStatus::Passed,
                1,
            )
            .unwrap();
        let err = s.complete_verified_task(t6.task_id, r6, lying).unwrap_err();
        assert_eq!(
            err,
            TaskError::CriteriaNotCovered {
                record: lying,
                missing: vec!["c1".to_string()]
            }
        );
        keep_verifying(&s, t6.task_id, r6);

        // (g) worktree mismatch: the record's base worktree must equal the
        // completing session's. Standalone sessions share the numeric task
        // id 1 but live in different workspaces: a record certified in /wb
        // cannot complete task 1 of a session in /wc.
        let wb = m.create_workspace("/wb").unwrap();
        let sb = m.create_session(wb, "b", "p", "m").unwrap();
        let tb = sb
            .create_task(criteria_task(&sb, sb.task_id().unwrap(), vec!["c1".into()]))
            .unwrap();
        let rb = drive_to_verifying(&sb, tb.task_id);
        let rec_b = passed_record(&sb, tb.task_id, &["c1".into()]);
        let wc = m.create_workspace("/wc").unwrap();
        let sc = m.create_session(wc, "c", "p", "m").unwrap();
        assert_eq!(
            sc.task_id().unwrap(),
            tb.task_id,
            "both standalone at task 1"
        );
        let tc = sc
            .create_task(criteria_task(&sc, sc.task_id().unwrap(), vec!["c1".into()]))
            .unwrap();
        let rc = drive_to_verifying(&sc, tc.task_id);
        let err = sc
            .complete_verified_task(tc.task_id, rc, rec_b)
            .unwrap_err();
        assert!(matches!(err, TaskError::WorktreeMismatch { .. }));
        assert_eq!(sc.task_revision(tc.task_id).unwrap(), rc);
        // And the same-workspace record completes it.
        let rec_c = passed_record(&sc, tc.task_id, &["c1".into()]);
        let done_c = sc.complete_verified_task(tc.task_id, rc, rec_c).unwrap();
        assert_eq!(done_c.state, TaskState::VerifiedComplete);
        let _ = (rb, wb, sb, tb);
    }

    #[test]
    fn transition_task_is_the_only_way_into_completion_relevant_states() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let created = s.create_task(task(&s)).unwrap();
        let t = created.task_id;
        // Wait on a Pending task: illegal edge.
        let err = s
            .transition_task(t, TaskRevision::new(1), TaskTransition::Wait, None)
            .unwrap_err();
        assert!(matches!(err, TaskError::IllegalTransition { .. }));
        // A stale revision refuses even for a legal edge.
        let err = s
            .transition_task(t, TaskRevision::new(9), TaskTransition::StartRunning, None)
            .unwrap_err();
        assert_eq!(
            err,
            TaskError::RevisionMismatch {
                task_id: t,
                expected: TaskRevision::new(9),
                actual: TaskRevision::new(1)
            }
        );
        // Proofs do not ride transitions.
        let err = s
            .transition_task(
                t,
                TaskRevision::new(1),
                TaskTransition::StartRunning,
                Some(VerificationRecordId::new(7)),
            )
            .unwrap_err();
        assert!(matches!(err, TaskError::Malformed(_)));
        // The full chain into Verifying bumps revision every hop.
        let r1 = s.task_revision(t).unwrap();
        s.transition_task(t, r1, TaskTransition::StartRunning, None)
            .unwrap();
        let r2 = s.task_revision(t).unwrap();
        s.transition_task(t, r2, TaskTransition::Wait, None)
            .unwrap();
        let r3 = s.task_revision(t).unwrap();
        s.transition_task(t, r3, TaskTransition::ResumeFromWaiting, None)
            .unwrap();
        let r4 = s.task_revision(t).unwrap();
        s.transition_task(t, r4, TaskTransition::RequestVerification, None)
            .unwrap();
        let r5 = s.task_revision(t).unwrap();
        s.transition_task(t, r5, TaskTransition::StartVerification, None)
            .unwrap();
        let r6 = s.task_revision(t).unwrap();
        assert_eq!(r6, TaskRevision::new(6));
        assert_eq!(s.get_task(t).unwrap().unwrap().state, TaskState::Verifying);
        // Reverify and fail edges from Verifying.
        s.transition_task(t, r6, TaskTransition::Reverify, None)
            .unwrap();
        let r7 = s.task_revision(t).unwrap();
        assert_eq!(
            s.get_task(t).unwrap().unwrap().state,
            TaskState::NeedsVerification
        );
        s.transition_task(t, r7, TaskTransition::StartVerification, None)
            .unwrap();
        let r8 = s.task_revision(t).unwrap();
        s.transition_task(t, r8, TaskTransition::FailFromVerifying, None)
            .unwrap();
        assert_eq!(s.get_task(t).unwrap().unwrap().state, TaskState::Failed);
        // Failed is terminal: Cancel is illegal.
        let r9 = s.task_revision(t).unwrap();
        let err = s
            .transition_task(t, r9, TaskTransition::Cancel, None)
            .unwrap_err();
        assert!(matches!(err, TaskError::IllegalTransition { .. }));
        // Cancellation from a non-terminal state is legal.
        let t2 = s
            .create_task(criteria_task(&s, TaskId::new(7), vec![]))
            .unwrap();
        let c = s
            .transition_task(
                t2.task_id,
                TaskRevision::new(1),
                TaskTransition::Cancel,
                None,
            )
            .unwrap();
        assert_eq!(c.state, TaskState::Cancelled);
    }

    // (e) racing completions and finalizes: exactly one winner each.
    #[test]
    fn concurrent_completions_and_finalizes_win_exactly_once() {
        let (_d, m) = test_manager();
        let s = Arc::new(session(&m));
        let t = s
            .create_task(criteria_task(&s, s.task_id().unwrap(), vec!["c1".into()]))
            .unwrap();
        let tid = t.task_id;
        let rev = drive_to_verifying(&s, tid);
        let rec = passed_record(&s, tid, &["c1".into()]);
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let mut handles = Vec::new();
        for _ in 0..2 {
            let s = s.clone();
            let barrier = barrier.clone();
            handles.push(thread::spawn(move || {
                barrier.wait();
                s.complete_verified_task(tid, rev, rec)
            }));
        }
        barrier.wait();
        let results: Vec<Result<Task, TaskError>> =
            handles.into_iter().map(|h| h.join().unwrap()).collect();
        let wins = results.iter().filter(|r| r.is_ok()).count();
        assert_eq!(wins, 1, "exactly one completion winner: {results:?}");
        for r in &results {
            if let Err(e) = r {
                assert!(
                    matches!(
                        e,
                        TaskError::NotVerifying { .. } | TaskError::RevisionMismatch { .. }
                    ),
                    "loser must fail typed, got {e:?}"
                );
            }
        }
        assert_eq!(s.task_revision(tid).unwrap(), rev.checked_next().unwrap());
        assert_eq!(
            s.get_task(tid).unwrap().unwrap().state,
            TaskState::VerifiedComplete
        );

        // Record-finalize CAS: a Running record finalizes exactly once.
        let t2 = s
            .create_task(criteria_task(&s, TaskId::new(2), vec![]))
            .unwrap();
        let running = s
            .create_verification_record(
                t2.task_id,
                None,
                vec![],
                vec![],
                vec![],
                vec![],
                None,
                VerificationStatus::Running,
                1,
            )
            .unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let mut handles = Vec::new();
        for _ in 0..2 {
            let s = s.clone();
            let barrier = barrier.clone();
            handles.push(thread::spawn(move || {
                barrier.wait();
                s.finalize_verification_record(running, VerificationStatus::Passed, 99)
            }));
        }
        barrier.wait();
        let results: Vec<Result<(), TaskError>> =
            handles.into_iter().map(|h| h.join().unwrap()).collect();
        let wins = results.iter().filter(|r| r.is_ok()).count();
        assert_eq!(wins, 1, "exactly one finalize winner: {results:?}");
        for r in &results {
            if let Err(e) = r {
                assert!(matches!(
                    e,
                    TaskError::RecordNotFinalizable {
                        current: VerificationStatus::Passed,
                        ..
                    }
                ));
            }
        }
        assert_eq!(
            s.get_verification_record(running).unwrap().unwrap().status,
            VerificationStatus::Passed
        );
    }

    // (f)+(i) crash between record creation and completion: reopen shows a
    // consistent store; the completion applies exactly once afterwards, and
    // records survive.
    #[test]
    fn records_and_state_survive_reopen_and_completion_after_crash() {
        let (dir, m) = test_manager();
        let s = session(&m);
        let sid = s.id;
        let t = s
            .create_task(criteria_task(&s, s.task_id().unwrap(), vec!["c1".into()]))
            .unwrap();
        let tid = t.task_id;
        let rev = drive_to_verifying(&s, tid);
        let rec = passed_record(&s, tid, &["c1".into()]);
        let listed = s.list_verification_records(tid).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].record_id, rec);
        assert_eq!(listed[0].revision, rev);
        // Crash (drop the manager) before the completion transaction.
        drop(s);
        drop(m);
        let m2 =
            crate::SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                .unwrap();
        let s2 = m2.get_session(sid).unwrap().unwrap();
        let row = s2.get_task(tid).unwrap().unwrap();
        assert_eq!(row.state, TaskState::Verifying, "crashed mid-verification");
        assert_eq!(s2.task_revision(tid).unwrap(), rev);
        assert_eq!(
            s2.get_verification_record(rec).unwrap().unwrap().revision,
            rev,
            "the record survived the crash"
        );
        // Reopen again after a second crash between record and completion…
        drop(s2);
        drop(m2);
        let m3 =
            crate::SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                .unwrap();
        let s3 = m3.get_session(sid).unwrap().unwrap();
        let done = s3.complete_verified_task(tid, rev, rec).unwrap();
        assert_eq!(done.state, TaskState::VerifiedComplete);
        assert_eq!(s3.task_revision(tid).unwrap(), rev.checked_next().unwrap());
        // Repeat completion after ANOTHER reopen refuses and does not bump.
        drop(s3);
        drop(m3);
        let m4 =
            crate::SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                .unwrap();
        let s4 = m4.get_session(sid).unwrap().unwrap();
        let err = s4
            .complete_verified_task(tid, rev.checked_next().unwrap(), rec)
            .unwrap_err();
        assert_eq!(
            err,
            TaskError::NotVerifying {
                actual: TaskState::VerifiedComplete
            }
        );
        assert_eq!(s4.task_revision(tid).unwrap(), rev.checked_next().unwrap());
        let frozen = s4
            .update_task(
                tid,
                TaskPatch {
                    goal: Some("tamper".into()),
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert_eq!(
            frozen,
            TaskError::TerminalTask {
                task_id: tid,
                state: TaskState::VerifiedComplete
            }
        );
        let records = s4.list_verification_records(tid).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].record_id, rec);
    }

    // (g) oversized record JSON is rejected with a typed error and NO row
    // is written (never truncated).
    #[test]
    fn oversized_record_json_is_rejected_typed_and_never_truncated() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let created = s.create_task(task(&s)).unwrap();
        let tid = created.task_id;
        // criteria_json over 128 KiB: 64 keys near the per-key cap.
        let huge_criteria: Vec<CriterionVerification> = (0..MAX_VERIFICATION_RECORD_CRITERIA)
            .map(|i| CriterionVerification {
                criterion_key: format!("{i:03}").repeat(MAX_VERIFICATION_CRITERION_KEY_BYTES / 4),
                passed: true,
                evidence: None,
            })
            .collect();
        assert!(
            serde_json::to_vec(&huge_criteria).unwrap().len()
                > MAX_VERIFICATION_CRITERIA_JSON_BYTES
        );
        let err = s
            .create_verification_record(
                tid,
                None,
                huge_criteria,
                vec![],
                vec![],
                vec![],
                None,
                VerificationStatus::Passed,
                1,
            )
            .unwrap_err();
        assert!(matches!(err, TaskError::Oversized(_)));
        // checks_json over 256 KiB.
        let huge_checks: Vec<CheckExecution> = (0..MAX_VERIFICATION_RECORD_CHECKS)
            .map(|i| CheckExecution {
                check: format!("check {i}").repeat(MAX_VERIFICATION_CHECK_NAME_BYTES / 10),
                program: "cargo".into(),
                args: vec![],
                category: "required".into(),
                required: true,
                status: VerificationStatus::Passed,
                started_ms: 1,
                finished_ms: Some(2),
                exit: Some(0),
                summary: Some("s".repeat(MAX_VERIFICATION_SUMMARY_BYTES)),
            })
            .collect();
        assert!(
            serde_json::to_vec(&huge_checks).unwrap().len() > MAX_VERIFICATION_CHECKS_JSON_BYTES
        );
        let err = s
            .create_verification_record(
                tid,
                None,
                vec![],
                huge_checks,
                vec![],
                vec![],
                None,
                VerificationStatus::Passed,
                1,
            )
            .unwrap_err();
        assert!(matches!(err, TaskError::Oversized(_)));
        // A hostile non-hex digest is malformed (typed), a giant evidence is
        // oversized.
        let err = s
            .create_verification_record(
                tid,
                None,
                vec![CriterionVerification {
                    criterion_key: "c1".into(),
                    passed: true,
                    evidence: None,
                }],
                vec![],
                vec![FileStateEvidence {
                    path: "src/main.rs".into(),
                    digest_hex: "not-hex!!".into(),
                    size: 1,
                }],
                vec![],
                None,
                VerificationStatus::Passed,
                1,
            )
            .unwrap_err();
        assert!(matches!(err, TaskError::Malformed(_)));
        let err = s
            .create_verification_record(
                tid,
                None,
                vec![CriterionVerification {
                    criterion_key: "c1".into(),
                    passed: true,
                    evidence: Some("e".repeat(MAX_VERIFICATION_EVIDENCE_BYTES + 1)),
                }],
                vec![],
                vec![],
                vec![],
                None,
                VerificationStatus::Passed,
                1,
            )
            .unwrap_err();
        assert!(matches!(err, TaskError::Oversized(_)));
        assert!(
            s.list_verification_records(tid).unwrap().is_empty(),
            "refused records left no trace"
        );
    }

    // (h) revisions are strictly monotone under 20 mixed updates — no
    // reuse, no backwards gap — also across a reopen.
    #[test]
    fn revisions_stay_monotone_under_mixed_updates_and_reopen() {
        let (dir, m) = test_manager();
        let s = session(&m);
        let sid = s.id;
        let created = s.create_task(task(&s)).unwrap();
        let tid = created.task_id;
        // Start the machine so the Wait/Resume round trips are legal.
        let r0 = s.task_revision(tid).unwrap();
        s.transition_task(tid, r0, TaskTransition::StartRunning, None)
            .unwrap();
        let mut last = s.task_revision(tid).unwrap();
        let mut seen = vec![last];
        for i in 0..20u64 {
            if i % 3 == 0 {
                s.update_task(
                    tid,
                    TaskPatch {
                        goal: Some(format!("goal iteration {i}")),
                        ..Default::default()
                    },
                )
                .unwrap();
            } else if i % 3 == 1 {
                s.update_task(
                    tid,
                    TaskPatch {
                        plan: Some(vec![format!("step {i}")]),
                        budget: Some(TaskBudget {
                            max_tokens: Some(i),
                            max_turns: Some(i as u32),
                            spent_tokens: 0,
                            spent_turns: 0,
                        }),
                        ..Default::default()
                    },
                )
                .unwrap();
            } else {
                // Ordinary state machine round trip Running -> Waiting ->
                // Running.
                let r = s.task_revision(tid).unwrap();
                s.transition_task(tid, r, TaskTransition::Wait, None)
                    .unwrap();
                let r = s.task_revision(tid).unwrap();
                s.transition_task(tid, r, TaskTransition::ResumeFromWaiting, None)
                    .unwrap();
            }
            let now = s.task_revision(tid).unwrap();
            assert!(
                now > last,
                "revision must strictly increase: {now:?} after {last:?}"
            );
            assert!(
                !seen.contains(&now),
                "revision {now:?} reused after {seen:?}"
            );
            seen.push(now);
            last = now;
        }
        assert!(last.raw() >= 2 + 20, "each iteration bumped at least once");
        // No-op patches do not move the revision.
        let before = last;
        s.update_task(tid, TaskPatch::default()).unwrap();
        assert_eq!(s.task_revision(tid).unwrap(), before);
        // Reopen: the sequence continues, never resetting or reusing.
        drop(s);
        drop(m);
        let m2 =
            crate::SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                .unwrap();
        let s2 = m2.get_session(sid).unwrap().unwrap();
        let after_reopen = s2.task_revision(tid).unwrap();
        assert_eq!(after_reopen, before, "revision survives reopen untouched");
        s2.update_task(
            tid,
            TaskPatch {
                goal: Some("after reopen".into()),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            s2.task_revision(tid).unwrap(),
            after_reopen.checked_next().unwrap(),
            "no reset, no reuse after reopen"
        );
    }

    // (j) record listing is deterministic (creation order) across calls.
    #[test]
    fn record_listing_is_deterministic() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let created = s.create_task(task(&s)).unwrap();
        let tid = created.task_id;
        let r1 = passed_record(&s, tid, &[]);
        let r2 = passed_record(&s, tid, &[]);
        let r3 = passed_record(&s, tid, &[]);
        let first = s.list_verification_records(tid).unwrap();
        let ids: Vec<VerificationRecordId> = first.iter().map(|r| r.record_id).collect();
        assert_eq!(ids, vec![r1, r2, r3], "creation order");
        assert_eq!(s.list_verification_records(tid).unwrap(), first, "stable");
        // Records of another task do not leak into this task's list.
        let t2 = s
            .create_task(criteria_task(&s, TaskId::new(2), vec![]))
            .unwrap();
        assert!(s.list_verification_records(t2.task_id).unwrap().is_empty());
    }
}
