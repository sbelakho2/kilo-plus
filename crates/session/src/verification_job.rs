//! Durable verification attempts and jobs (audit P0-5/P0-6/26/27; schema v22).
//!
//! One `VerificationAttempt` row = ONE verification attempt, keyed
//! `(session_id, task_id, attempt_op_id)`. Every required check of the
//! attempt — inline AND background — is a `verification_job` row keyed
//! `(session_id, task_id, attempt_op_id, check_id)`: inline checks carry
//! their terminal outcome from birth, background checks walk
//! `Queued -> Running -> Passed|Failed|Unavailable|Cancelled` via guarded
//! CAS transitions. The executed outcome JSON of a background check lands in
//! `verification_job_result` (same key) exactly once. Changed files live in
//! `verification_attempt_changed_file`, keyed `(..., ordinal)`.
//!
//! The attempt-keyed identity is what makes attempt isolation structural:
//! a result for attempt N is a DIFFERENT set of rows from attempt N+1, and
//! the store refuses to mutate attempt N once N+1 exists (typed
//! `Superseded`), so a late result can never contaminate the newer attempt.
//! The whole begin is ONE transaction, so a crash can never leave torn job
//! rows: recovery only ever re-queues a `Running` row whose executor died.
//!
//! Caps mirror the `VerificationRecord` contract: changed files <= 4096,
//! required checks <= 256, bounded program/argv, bounded spec/result JSON.
//! Oversized or malformed input is rejected with typed errors before any
//! write; a hostile/corrupt row is a LOUD typed decode error on every read
//! path — never a silent drop.

use serde::{Deserialize, Serialize};

use faktor_core::state::EnvironmentFingerprint;
use faktor_store::{
    StoreError, VerificationAttemptRow, VerificationAttemptView, VerificationJobRefusal,
    VerificationJobRow,
};

use crate::handle::SessionHandle;
use crate::SessionError;

// ---------------------------------------------------------------- bounds

/// Hard bound on one stored job check-id (row keys embed it).
pub const MAX_VERIFICATION_JOB_CHECK_ID_BYTES: usize = 128;
/// Hard bound on one stored job kind tag (`compile`/`test`/`lint`).
pub const MAX_VERIFICATION_JOB_KIND_BYTES: usize = 16;
/// Hard bound on one stored job canonical command text.
pub const MAX_VERIFICATION_JOB_COMMAND_BYTES: usize = 512;
/// Hard bound on the typed spec JSON of one job (the `CheckSpec` serde form).
pub const MAX_VERIFICATION_JOB_SPEC_JSON_BYTES: usize = 64 * 1024;
/// Hard bound on the typed result JSON of one job.
pub const MAX_VERIFICATION_JOB_RESULT_JSON_BYTES: usize = 64 * 1024;
/// Hard bound on one job's execution budget in ms.
pub const MAX_VERIFICATION_JOB_BUDGET_MS: u64 = 3_600_000;
/// Hard bound on the durable note of one job/attempt transition.
pub const MAX_VERIFICATION_JOB_NOTE_BYTES: usize = 512;
/// Changed files an attempt may certify (mirrors
/// `MAX_VERIFICATION_CHANGED_FILES`).
pub const MAX_VERIFICATION_ATTEMPT_CHANGED: usize = 4096;
/// One changed-file path bound (mirrors `MAX_VERIFICATION_PATH_BYTES`).
pub const MAX_VERIFICATION_ATTEMPT_PATH_BYTES: usize = 4096;
/// Background checks one attempt may enqueue (also the cap on the ordered
/// required-checks list — inline and background together; mirrors
/// `MAX_VERIFICATION_RECORD_CHECKS`).
pub const MAX_VERIFICATION_ATTEMPT_JOBS: usize = 256;
/// Workspace-root bound of one attempt.
pub const MAX_VERIFICATION_JOB_ROOT_BYTES: usize = 4096;
/// One check program bound (mirrors `MAX_VERIFICATION_PROGRAM_BYTES`).
pub const MAX_VERIFICATION_JOB_PROGRAM_BYTES: usize = 4096;
/// Per-argument count bound (mirrors `MAX_VERIFICATION_CHECK_ARGS`).
pub const MAX_VERIFICATION_JOB_ARGS: usize = 32;
/// One argument bound (mirrors `MAX_VERIFICATION_CHECK_ARG_BYTES`).
pub const MAX_VERIFICATION_JOB_ARG_BYTES: usize = 1024;
/// One environment-fingerprint JSON column bound.
pub const MAX_VERIFICATION_JOB_FINGERPRINT_JSON_BYTES: usize = 64 * 1024;

// ---------------------------------------------------------------- types

/// Lifecycle state of one durable verification job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationJobState {
    /// Durable, waiting for an executor claim.
    Queued,
    /// An executor claimed it (CAS) and the check process is live.
    Running,
    /// The check ran to completion and passed (exit 0).
    Passed,
    /// The check ran to completion and failed (non-zero exit).
    Failed,
    /// The check could not produce a verdict (spawn refused, killed by
    /// deadline/cancellation, infra error). NEVER a failure of the code
    /// under check.
    Unavailable,
    /// The job will never run: superseded by a newer attempt, or the
    /// attempt was aborted by an inline failure. Typed note carries why.
    Cancelled,
}

impl VerificationJobState {
    /// True while an executor may still settle the job.
    pub fn is_open(self) -> bool {
        matches!(self, Self::Queued | Self::Running)
    }

    pub fn is_terminal(self) -> bool {
        !self.is_open()
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Passed => "passed",
            Self::Failed => "failed",
            Self::Unavailable => "unavailable",
            Self::Cancelled => "cancelled",
        }
    }

    fn from_str(raw: &str) -> Result<Self, SessionError> {
        match raw {
            "queued" => Ok(Self::Queued),
            "running" => Ok(Self::Running),
            "passed" => Ok(Self::Passed),
            "failed" => Ok(Self::Failed),
            "unavailable" => Ok(Self::Unavailable),
            "cancelled" => Ok(Self::Cancelled),
            other => Err(SessionError::Malformed(format!(
                "verification job row carries unknown state {other:?}"
            ))),
        }
    }
}

/// The static definition of one background check inside an attempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerificationJobInput {
    /// Stable check id (the derived check's id).
    pub check_id: String,
    /// `compile` | `test` | `lint` (the record category tag).
    pub kind: String,
    /// Canonical command text (`program arg...`).
    pub command: String,
    /// Bounded argv identity of the check (the lossy UTF-8 view of the
    /// typed spec's `program`). Execution re-parses `spec_json`, so a
    /// non-UTF8 argument is never lost to this field.
    pub program: String,
    /// Bounded argv identity (lossy UTF-8 view, same contract as
    /// `program`).
    pub args: Vec<String>,
    /// The typed spec as serde JSON (opaque to this crate; parsed by the
    /// agent executor when the job runs).
    pub spec_json: String,
    /// Wall deadline of one execution of this job, in ms.
    pub budget_ms: u64,
}

/// One durable verification job row (a required check of one attempt).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationJob {
    /// `"vj:{task_id}:{attempt_op}:{check_id}"` — stable identity.
    pub id: String,
    pub task_id: u64,
    /// The task revision the job's attempt certified at enqueue.
    pub task_revision: u64,
    /// The durable workspace root the check runs in (never the daemon cwd).
    pub workspace_root: String,
    pub check_id: String,
    pub kind: String,
    pub command: String,
    /// The typed spec as serde JSON (empty for inline checks).
    pub spec_json: String,
    pub budget_ms: u64,
    /// The attempt (enqueueing turn op) this row belongs to.
    pub attempt_op: u64,
    /// Derivation order of the check inside the attempt.
    pub ordinal: u32,
    pub state: VerificationJobState,
    /// `Some(outcome)` for an INLINE check (terminal from birth); `None` for
    /// a background job.
    pub inline: Option<VerificationInlineStatus>,
    /// Typed note: recovery/abort/supersede explanations.
    pub note: Option<String>,
    /// The op that claimed this job (the executor run), while Running.
    pub op_id: Option<u64>,
    /// The typed outcome as serde JSON (`CheckOutcome`), when terminal by
    /// execution (Passed/Failed/Unavailable).
    pub result_json: Option<String>,
    /// The bounded environment fingerprint the attempt's jobs ran under.
    /// `None` on legacy rows that predate the field — an honest absence.
    pub environment_fingerprint: Option<EnvironmentFingerprint>,
    pub created_ms: i64,
    pub updated_ms: i64,
    pub finished_ms: Option<i64>,
}

/// How one INLINE required check of an attempt ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationInlineStatus {
    Passed,
    Failed,
    /// Ran inline but produced no verdict (deadline/cancellation kill,
    /// program missing, infra): never a failure of the code under check.
    Unavailable,
}

impl VerificationInlineStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Passed => "passed",
            Self::Failed => "failed",
            Self::Unavailable => "unavailable",
        }
    }

    fn from_str(raw: &str) -> Result<Self, SessionError> {
        match raw {
            "passed" => Ok(Self::Passed),
            "failed" => Ok(Self::Failed),
            "unavailable" => Ok(Self::Unavailable),
            other => Err(SessionError::Malformed(format!(
                "verification job row carries unknown inline status {other:?}"
            ))),
        }
    }

    fn state(self) -> VerificationJobState {
        match self {
            Self::Passed => VerificationJobState::Passed,
            Self::Failed => VerificationJobState::Failed,
            Self::Unavailable => VerificationJobState::Unavailable,
        }
    }
}

/// One required check of an attempt, IN DERIVATION ORDER: inline checks
/// carry their outcome, background checks carry `inline = None` (the
/// check's spec lives in its job row).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerificationAttemptCheck {
    pub check_id: String,
    /// Canonical command text (`program arg...`) of the required check.
    pub command: String,
    /// `Some(outcome)` = the check ran INLINE in the enqueueing turn;
    /// `None` = the check was enqueued as a background job row of this
    /// attempt.
    pub inline: Option<VerificationInlineStatus>,
}

/// One durable verification attempt record: everything a later settlement
/// needs to rebuild the attempt's proof from its rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationAttempt {
    pub task_id: u64,
    /// The attempt's identity: the op id of the enqueueing turn.
    pub op_id: u64,
    /// The task revision the attempt certified at enqueue.
    pub task_revision: u64,
    pub workspace_root: String,
    /// Files changed by the enqueueing turn (the evidence the attempt
    /// certifies), in derivation order.
    pub changed: Vec<String>,
    /// The required checks in derivation order (inline outcomes inline).
    pub checks: Vec<VerificationAttemptCheck>,
    /// The bounded environment fingerprint observed when the attempt was
    /// enqueued. `None` on legacy rows.
    pub environment_fingerprint: Option<EnvironmentFingerprint>,
    pub created_ms: i64,
}

/// Report of one honest recovery sweep.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VerificationJobRecoveryReport {
    /// `Running` rows re-queued with a typed note (the previous executor
    /// died mid-check).
    pub requeued: usize,
    /// Open rows orphaned to `Unavailable` (their attempt record is
    /// missing — only possible on a hand-corrupted database).
    pub orphaned: usize,
}

// ---------------------------------------------------------- conversions

fn job_store_err(e: StoreError) -> SessionError {
    match e {
        StoreError::Oversized(m) => SessionError::Oversized(m),
        StoreError::Malformed(m) => SessionError::Malformed(m),
        StoreError::Corrupt(v) => SessionError::Malformed(format!(
            "corrupt verification job store row: {}",
            v.join("; ")
        )),
        other => crate::map_store_err(other),
    }
}

fn malformed_row(what: &str, detail: &str) -> SessionError {
    SessionError::Malformed(format!("verification job {what} row is corrupt: {detail}"))
}

fn check_text(value: &str, what: &str, max: usize) -> Result<(), SessionError> {
    if value.is_empty() {
        return Err(SessionError::Malformed(format!(
            "verification job {what} must be non-empty"
        )));
    }
    if value.len() > max {
        return Err(SessionError::Oversized(format!(
            "verification job {what} of {} bytes exceeds {max}",
            value.len()
        )));
    }
    Ok(())
}

fn bounds_check_job_input(job: &VerificationJobInput) -> Result<(), SessionError> {
    check_text(
        &job.check_id,
        "check_id",
        MAX_VERIFICATION_JOB_CHECK_ID_BYTES,
    )?;
    check_text(&job.kind, "kind", MAX_VERIFICATION_JOB_KIND_BYTES)?;
    check_text(&job.command, "command", MAX_VERIFICATION_JOB_COMMAND_BYTES)?;
    if job.program.len() > MAX_VERIFICATION_JOB_PROGRAM_BYTES {
        return Err(SessionError::Oversized(format!(
            "verification job '{}' program of {} bytes exceeds {MAX_VERIFICATION_JOB_PROGRAM_BYTES}",
            job.check_id,
            job.program.len()
        )));
    }
    if job.args.len() > MAX_VERIFICATION_JOB_ARGS {
        return Err(SessionError::Oversized(format!(
            "verification job '{}' carries {} args (cap {MAX_VERIFICATION_JOB_ARGS})",
            job.check_id,
            job.args.len()
        )));
    }
    for arg in &job.args {
        if arg.len() > MAX_VERIFICATION_JOB_ARG_BYTES {
            return Err(SessionError::Oversized(format!(
                "verification job '{}' arg of {} bytes exceeds {MAX_VERIFICATION_JOB_ARG_BYTES}",
                job.check_id,
                arg.len()
            )));
        }
    }
    check_text(
        &job.spec_json,
        "spec_json",
        MAX_VERIFICATION_JOB_SPEC_JSON_BYTES,
    )?;
    if job.budget_ms == 0 || job.budget_ms > MAX_VERIFICATION_JOB_BUDGET_MS {
        return Err(SessionError::Oversized(format!(
            "verification job budget_ms {} outside 1..={MAX_VERIFICATION_JOB_BUDGET_MS}",
            job.budget_ms
        )));
    }
    Ok(())
}

fn bounds_check_attempt_check(run: &VerificationAttemptCheck) -> Result<(), SessionError> {
    check_text(
        &run.check_id,
        "check_id",
        MAX_VERIFICATION_JOB_CHECK_ID_BYTES,
    )?;
    check_text(&run.command, "command", MAX_VERIFICATION_JOB_COMMAND_BYTES)
}

/// The bounded `(program, args_json)` identity of one background spec. The
/// spec is opaque JSON to this crate, but `program`/`args` are required
/// bounded fields (mirrors the record's bounded argv contract).
fn parse_fingerprint(
    raw: Option<String>,
    what: &str,
) -> Result<Option<EnvironmentFingerprint>, SessionError> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    if raw.len() > MAX_VERIFICATION_JOB_FINGERPRINT_JSON_BYTES {
        return Err(SessionError::Oversized(format!(
            "verification job {what} fingerprint column of {} bytes exceeds {MAX_VERIFICATION_JOB_FINGERPRINT_JSON_BYTES}",
            raw.len()
        )));
    }
    serde_json::from_str(&raw).map(Some).map_err(|e| {
        malformed_row(
            what,
            &format!("undecodable environment fingerprint json: {e}"),
        )
    })
}

fn job_from_row(row: VerificationJobRow) -> Result<VerificationJob, SessionError> {
    let state = VerificationJobState::from_str(&row.state)?;
    let inline = match row.inline_status.as_deref() {
        Some(raw) => Some(VerificationInlineStatus::from_str(raw)?),
        None => None,
    };
    let environment_fingerprint =
        parse_fingerprint(row.environment_fingerprint_json.clone(), &row.check_id)?;
    Ok(VerificationJob {
        id: format!(
            "vj:{}:{}:{}",
            row.task_id.raw(),
            row.attempt_op_id,
            row.check_id
        ),
        task_id: row.task_id.raw(),
        task_revision: row.task_revision.raw(),
        workspace_root: row.workspace_root,
        check_id: row.check_id,
        kind: row.kind,
        command: row.command,
        spec_json: row.spec_json.unwrap_or_default(),
        budget_ms: row.budget_ms,
        attempt_op: row.attempt_op_id,
        ordinal: row.ordinal,
        state,
        inline,
        note: row.note,
        op_id: row.op_id,
        result_json: row.result_json,
        environment_fingerprint,
        created_ms: row.created_ms,
        updated_ms: row.updated_ms,
        finished_ms: row.finished_ms,
    })
}

fn attempt_from_view(view: VerificationAttemptView) -> Result<VerificationAttempt, SessionError> {
    let mut checks: Vec<VerificationAttemptCheck> = Vec::with_capacity(view.checks.len());
    for row in &view.checks {
        let inline = match row.inline_status.as_deref() {
            Some(raw) => Some(VerificationInlineStatus::from_str(raw)?),
            None => None,
        };
        checks.push(VerificationAttemptCheck {
            check_id: row.check_id.clone(),
            command: row.command.clone(),
            inline,
        });
    }
    let environment_fingerprint =
        parse_fingerprint(view.attempt.environment_fingerprint_json.clone(), "attempt")?;
    Ok(VerificationAttempt {
        task_id: view.attempt.task_id.raw(),
        op_id: view.attempt.attempt_op_id,
        task_revision: view.attempt.task_revision.raw(),
        workspace_root: view.attempt.workspace_root,
        changed: view.changed,
        checks,
        environment_fingerprint,
        created_ms: view.attempt.created_ms,
    })
}

fn refusal_error(refusal: VerificationJobRefusal) -> SessionError {
    match refusal {
        VerificationJobRefusal::Missing { check_id } => {
            SessionError::NotFound(format!("verification job '{check_id}'"))
        }
        VerificationJobRefusal::NotOpen { check_id, state } => SessionError::Conflict(format!(
            "job '{check_id}' is {state}, not in the required source state — a job transitions \
             exactly once"
        )),
        VerificationJobRefusal::Superseded {
            attempt_op_id,
            newest_attempt_op_id,
        } => SessionError::Conflict(format!(
            "attempt {attempt_op_id} is superseded by attempt {newest_attempt_op_id}: its rows \
             are frozen and can never be mutated or late-resolved"
        )),
        VerificationJobRefusal::ResultExists { check_id } => SessionError::Conflict(format!(
            "job '{check_id}' already has a durable result — a result is recorded exactly once"
        )),
    }
}

// ------------------------------------------------------------- the surface

impl SessionHandle {
    /// Begin one verification attempt durably (audit P0-5/26): the attempt
    /// row, its changed-file rows and one row per required check (inline
    /// outcomes and background job definitions) in ONE store transaction.
    ///
    /// Refusals (all typed, before any write):
    /// - the attempt record for `op_id` already exists → idempotent `Ok`
    ///   (a crashed begin may be retried with the same op);
    /// - an OPEN job row for one of the checks already belongs to a
    ///   DIFFERENT attempt → [`SessionError::Conflict`] (the caller must
    ///   cancel/supersede that attempt first — an open job is never
    ///   silently replaced);
    /// - any bound violation → [`SessionError::Oversized`]/`Malformed`.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_verification_attempt(
        &self,
        task_id: u64,
        task_revision: u64,
        op_id: u64,
        workspace_root: &str,
        changed: &[String],
        checks: &[VerificationAttemptCheck],
        jobs: &[VerificationJobInput],
    ) -> Result<(), SessionError> {
        self.begin_verification_attempt_with_fingerprint(
            task_id,
            task_revision,
            op_id,
            workspace_root,
            changed,
            checks,
            jobs,
            None,
        )
    }

    /// Additive v2 twin of [`SessionHandle::begin_verification_attempt`]
    /// (audits 94/116/117): the enqueue carries the bounded environment
    /// fingerprint observed when the attempt began, stamped onto the attempt
    /// row AND every check row so a job settled after a restart still knows
    /// the environment it was enqueued under. `None` behaves
    /// byte-identically to the legacy method.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_verification_attempt_with_fingerprint(
        &self,
        task_id: u64,
        task_revision: u64,
        op_id: u64,
        workspace_root: &str,
        changed: &[String],
        checks: &[VerificationAttemptCheck],
        jobs: &[VerificationJobInput],
        environment_fingerprint: Option<EnvironmentFingerprint>,
    ) -> Result<(), SessionError> {
        if let Some(fp) = &environment_fingerprint {
            fp.validate().map_err(|violations| {
                let oversized = violations.iter().any(|v| v.oversized);
                let detail = violations
                    .iter()
                    .map(|v| format!("{}: {}", v.field, v.detail))
                    .collect::<Vec<_>>()
                    .join("; ");
                if oversized {
                    SessionError::Oversized(format!("verification fingerprint: {detail}"))
                } else {
                    SessionError::Malformed(format!("verification fingerprint: {detail}"))
                }
            })?;
        }
        if task_id == 0 || task_revision == 0 || op_id == 0 {
            return Err(SessionError::Malformed(
                "task_id/task_revision/op_id must be non-zero".into(),
            ));
        }
        check_text(
            workspace_root,
            "workspace_root",
            MAX_VERIFICATION_JOB_ROOT_BYTES,
        )?;
        if changed.len() > MAX_VERIFICATION_ATTEMPT_CHANGED {
            return Err(SessionError::Oversized(format!(
                "verification attempt certifies {} changed files (cap {MAX_VERIFICATION_ATTEMPT_CHANGED})",
                changed.len()
            )));
        }
        for p in changed {
            check_text(p, "changed path", MAX_VERIFICATION_ATTEMPT_PATH_BYTES)?;
        }
        if checks.is_empty() || checks.len() > MAX_VERIFICATION_ATTEMPT_JOBS {
            return Err(SessionError::Oversized(format!(
                "verification attempt carries {} required checks (1..={MAX_VERIFICATION_ATTEMPT_JOBS})",
                checks.len()
            )));
        }
        for run in checks {
            bounds_check_attempt_check(run)?;
        }
        if jobs.is_empty() {
            return Err(SessionError::Malformed(
                "a verification attempt must enqueue at least one background job".into(),
            ));
        }
        if jobs.len() > MAX_VERIFICATION_ATTEMPT_JOBS {
            return Err(SessionError::Oversized(format!(
                "verification attempt enqueues {} jobs (cap {MAX_VERIFICATION_ATTEMPT_JOBS})",
                jobs.len()
            )));
        }
        for job in jobs {
            bounds_check_job_input(job)?;
        }
        // Every background job must be declared exactly once in the ordered
        // checks, and every ordered background check must carry a job.
        let mut seen_jobs = std::collections::HashSet::new();
        for job in jobs {
            let declared = checks
                .iter()
                .filter(|c| c.inline.is_none() && c.check_id == job.check_id)
                .count();
            if declared != 1 {
                return Err(SessionError::Malformed(format!(
                    "job '{}' is declared {} times as a background check (must be exactly once)",
                    job.check_id, declared
                )));
            }
            if !seen_jobs.insert(job.check_id.as_str()) {
                return Err(SessionError::Malformed(format!(
                    "job '{}' is enqueued twice in one attempt",
                    job.check_id
                )));
            }
        }
        for c in checks {
            if c.inline.is_none() && !seen_jobs.contains(c.check_id.as_str()) {
                return Err(SessionError::Malformed(format!(
                    "background check '{}' has no job row definition",
                    c.check_id
                )));
            }
        }
        let fingerprint_json = match &environment_fingerprint {
            Some(fp) => Some(
                serde_json::to_string(fp)
                    .map_err(|e| SessionError::Internal(format!("fingerprint json: {e}")))?,
            ),
            None => None,
        };
        let now = self.now_ms();
        let mut check_rows: Vec<VerificationJobRow> = Vec::with_capacity(checks.len());
        for (ordinal, run) in checks.iter().enumerate() {
            let ordinal = u32::try_from(ordinal).unwrap_or(u32::MAX);
            let job = jobs.iter().find(|j| j.check_id == run.check_id);
            let (kind, spec_json, program, args_json, budget_ms, inline, state) = match run.inline {
                Some(inline) => (
                    String::new(),
                    None,
                    String::new(),
                    "[]".to_string(),
                    0u64,
                    Some(inline.as_str().to_string()),
                    inline.state().as_str().to_string(),
                ),
                None => {
                    let job = job.expect("every background check has a job (validated above)");
                    let args_json = serde_json::to_string(&job.args)
                        .map_err(|e| SessionError::Internal(format!("job args json: {e}")))?;
                    (
                        job.kind.clone(),
                        Some(job.spec_json.clone()),
                        job.program.clone(),
                        args_json,
                        job.budget_ms,
                        None,
                        VerificationJobState::Queued.as_str().to_string(),
                    )
                }
            };
            check_rows.push(VerificationJobRow {
                session_id: self.id,
                task_id: faktor_core::id::TaskId::new(task_id),
                attempt_op_id: op_id,
                check_id: run.check_id.clone(),
                ordinal,
                task_revision: faktor_core::id::TaskRevision::new(task_revision),
                workspace_root: workspace_root.to_string(),
                kind,
                command: run.command.clone(),
                program,
                args_json,
                spec_json,
                budget_ms,
                inline_status: inline,
                state,
                result_json: None,
                note: None,
                op_id: None,
                environment_fingerprint_json: fingerprint_json.clone(),
                created_ms: now,
                updated_ms: now,
                finished_ms: None,
            });
        }
        let attempt = VerificationAttemptRow {
            session_id: self.id,
            task_id: faktor_core::id::TaskId::new(task_id),
            attempt_op_id: op_id,
            task_revision: faktor_core::id::TaskRevision::new(task_revision),
            workspace_root: workspace_root.to_string(),
            environment_fingerprint_json: fingerprint_json,
            created_ms: now,
        };
        self.manager
            .store()
            .verification_attempt_begin(&attempt, changed, &check_rows)
            .map_err(job_store_err)?;
        Ok(())
    }

    /// Cancel every OPEN job of `attempt_op` to `Cancelled` with a typed
    /// note (supersede by a newer attempt, or abort after an inline
    /// required check failed). Returns the number of rows cancelled. A
    /// cancelled job is terminal and can never certify completion.
    pub fn cancel_verification_attempt(
        &self,
        task_id: u64,
        attempt_op: u64,
        reason: &str,
    ) -> Result<usize, SessionError> {
        check_text(reason, "cancel note", MAX_VERIFICATION_JOB_NOTE_BYTES)?;
        let task_id = faktor_core::id::TaskId::new(task_id);
        let cancelled = self
            .manager
            .store()
            .verification_attempt_cancel(self.id, task_id, attempt_op, reason, self.now_ms())
            .map_err(job_store_err)?;
        Ok(usize::try_from(cancelled).unwrap_or(usize::MAX))
    }

    /// The NEWEST attempt record of `task_id`, or `None`. "Newest" is the
    /// highest attempt op (session op ids are monotonic).
    pub fn current_verification_attempt(
        &self,
        task_id: u64,
    ) -> Result<Option<VerificationAttempt>, SessionError> {
        let task_id = faktor_core::id::TaskId::new(task_id);
        let view = self
            .manager
            .store()
            .verification_attempt_current(self.id, task_id)
            .map_err(job_store_err)?;
        view.map(attempt_from_view).transpose()
    }

    /// The attempt record of `task_id` for a specific op (settlement reads
    /// the exact attempt its job rows belong to).
    pub fn verification_attempt(
        &self,
        task_id: u64,
        attempt_op: u64,
    ) -> Result<Option<VerificationAttempt>, SessionError> {
        let task_id = faktor_core::id::TaskId::new(task_id);
        let view = self
            .manager
            .store()
            .verification_attempt_get(self.id, task_id, attempt_op)
            .map_err(job_store_err)?;
        view.map(attempt_from_view).transpose()
    }

    /// Every OPEN (Queued|Running) background job row of `task_id`,
    /// deterministic (task, check-id) order.
    pub fn open_verification_jobs(
        &self,
        task_id: u64,
    ) -> Result<Vec<VerificationJob>, SessionError> {
        let task_id = faktor_core::id::TaskId::new(task_id);
        self.manager
            .store()
            .verification_jobs_open(self.id, task_id)
            .map_err(job_store_err)?
            .into_iter()
            .map(job_from_row)
            .collect()
    }

    /// Every background job row of one attempt (any state), deterministic
    /// derivation order. Inline checks are part of the attempt record
    /// ([`SessionHandle::verification_attempt`]) and never appear here.
    pub fn verification_attempt_jobs(
        &self,
        task_id: u64,
        attempt_op: u64,
    ) -> Result<Vec<VerificationJob>, SessionError> {
        let task_id = faktor_core::id::TaskId::new(task_id);
        self.manager
            .store()
            .verification_jobs_for_attempt(self.id, task_id, attempt_op)
            .map_err(job_store_err)?
            .into_iter()
            .map(job_from_row)
            .collect()
    }

    /// Claim one `Queued` job of `attempt_op` for execution: the guarded
    /// CAS to `Running` (with the executor's op attached). A row that is not
    /// `Queued`, belongs to another attempt, or whose attempt was superseded
    /// refuses with a typed error — a job is executed at most once per claim.
    pub fn claim_verification_job(
        &self,
        task_id: u64,
        check_id: &str,
        attempt_op: u64,
        op_id: u64,
    ) -> Result<VerificationJob, SessionError> {
        check_text(check_id, "check_id", MAX_VERIFICATION_JOB_CHECK_ID_BYTES)?;
        if op_id == 0 {
            return Err(SessionError::Malformed("op_id must be non-zero".into()));
        }
        let task_id = faktor_core::id::TaskId::new(task_id);
        let outcome = self
            .manager
            .store()
            .verification_job_claim(self.id, task_id, attempt_op, check_id, op_id, self.now_ms())
            .map_err(job_store_err)?;
        match outcome {
            Ok(row) => job_from_row(row),
            Err(refusal) => Err(refusal_error(refusal)),
        }
    }

    /// Resolve one `Running` job to its terminal state (CAS): `Passed`,
    /// `Failed`, `Unavailable` (the typed outcome rides `result_json`) or
    /// `Cancelled`. Once a NEWER attempt exists, a late resolution of this
    /// attempt is a typed refusal — attempt N is frozen.
    pub fn resolve_verification_job(
        &self,
        task_id: u64,
        check_id: &str,
        attempt_op: u64,
        state: VerificationJobState,
        note: Option<String>,
        result_json: Option<String>,
    ) -> Result<VerificationJob, SessionError> {
        if !state.is_terminal() {
            return Err(SessionError::Malformed(format!(
                "resolve state {state:?} is not terminal"
            )));
        }
        check_text(check_id, "check_id", MAX_VERIFICATION_JOB_CHECK_ID_BYTES)?;
        if let Some(result) = &result_json {
            check_text(
                result,
                "result_json",
                MAX_VERIFICATION_JOB_RESULT_JSON_BYTES,
            )?;
        }
        if let Some(note) = &note {
            check_text(note, "note", MAX_VERIFICATION_JOB_NOTE_BYTES)?;
        }
        let task_id = faktor_core::id::TaskId::new(task_id);
        let outcome = self
            .manager
            .store()
            .verification_job_resolve(
                self.id,
                task_id,
                attempt_op,
                check_id,
                state.as_str(),
                note.as_deref(),
                result_json.as_deref(),
                self.now_ms(),
            )
            .map_err(job_store_err)?;
        match outcome {
            Ok(row) => job_from_row(row),
            Err(refusal) => Err(refusal_error(refusal)),
        }
    }

    /// Honest post-restart recovery (audit P0-5/26): every `Running` row —
    /// the previous process died while its executor was mid-check — is
    /// re-queued with a typed note (the check never certified anything, so
    /// re-running it deterministically is the only honest path to a
    /// verdict). Open rows whose attempt record is missing (only possible on
    /// a hand-corrupted database) are orphaned to `Unavailable` — they can
    /// never settle a claim. Idempotent; `Queued` rows of a vanished process
    /// are untouched (they were never claimed).
    pub fn recover_verification_jobs_after_restart(
        &self,
    ) -> Result<VerificationJobRecoveryReport, SessionError> {
        let recovery = self
            .manager
            .store()
            .verification_jobs_requeue_running(self.id, self.now_ms())
            .map_err(job_store_err)?;
        Ok(VerificationJobRecoveryReport {
            requeued: usize::try_from(recovery.requeued).unwrap_or(usize::MAX),
            orphaned: usize::try_from(recovery.orphaned).unwrap_or(usize::MAX),
        })
    }
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::tests::{session, test_manager};
    use crate::SessionManager;
    use std::sync::Arc;

    const ROOT: &str = "/tmp/session-job-root";
    const REV: u64 = 3;
    const TASK: u64 = 1;

    fn spec(id: &str, program: &str, args: &[&str]) -> String {
        serde_json::json!({
            "id": id,
            "kind": "compile",
            "category": "full",
            "program": program,
            "args": args,
            "cwd_rel": ".",
            "affects": [],
            "required": true,
        })
        .to_string()
    }

    fn job_input(id: &str) -> VerificationJobInput {
        VerificationJobInput {
            check_id: id.into(),
            kind: "test".into(),
            command: format!("ctest {id}"),
            program: "ctest".into(),
            args: Vec::new(),
            spec_json: spec(id, "ctest", &[]),
            budget_ms: 600_000,
        }
    }

    fn inline_run(id: &str, status: VerificationInlineStatus) -> VerificationAttemptCheck {
        VerificationAttemptCheck {
            check_id: id.into(),
            command: format!("make {id}"),
            inline: Some(status),
        }
    }

    fn inline_pass(id: &str) -> VerificationAttemptCheck {
        inline_run(id, VerificationInlineStatus::Passed)
    }

    fn job_check(id: &str) -> VerificationAttemptCheck {
        VerificationAttemptCheck {
            check_id: id.into(),
            command: format!("ctest {id}"),
            inline: None,
        }
    }

    fn outcome_json(status: &str, exit: i64) -> String {
        serde_json::json!({
            "status": status,
            "exit": exit,
            "started_ms": 1,
            "finished_ms": 2,
            "summary": null,
            "truncated": false,
        })
        .to_string()
    }

    #[test]
    fn begin_creates_queued_jobs_and_attempt_record() {
        let (_dir, m) = test_manager();
        let s = session(&m);
        let op = 10u64;
        s.begin_verification_attempt(
            TASK,
            REV,
            op,
            ROOT,
            &["src/a.c".into()],
            &[
                inline_pass("make_build"),
                job_check("make_test"),
                job_check("make_check"),
            ],
            &[job_input("make_test"), job_input("make_check")],
        )
        .unwrap();
        let jobs = s.open_verification_jobs(TASK).unwrap();
        assert_eq!(jobs.len(), 2);
        for j in &jobs {
            assert_eq!(j.state, VerificationJobState::Queued);
            assert_eq!(j.inline, None);
            assert_eq!(j.attempt_op, op);
            assert_eq!(j.task_revision, REV);
            assert_eq!(j.workspace_root, ROOT);
            assert_eq!(j.op_id, None);
        }
        let attempt = s.current_verification_attempt(TASK).unwrap().unwrap();
        assert_eq!(attempt.op_id, op);
        assert_eq!(attempt.changed, vec!["src/a.c".to_string()]);
        assert_eq!(attempt.checks.len(), 3, "derivation order preserved");
        assert_eq!(attempt.checks[0].check_id, "make_build");
        assert_eq!(
            attempt.checks[0].inline,
            Some(VerificationInlineStatus::Passed)
        );
        assert_eq!(attempt.checks[1].check_id, "make_test");
        assert_eq!(attempt.checks[1].inline, None);
        // Inline checks are terminal rows of the attempt, not open jobs.
        let all = s.verification_attempt_jobs(TASK, op).unwrap();
        assert_eq!(all.len(), 2, "only background jobs are job rows here");
    }

    #[test]
    fn job_lifecycle_is_a_guarded_cas() {
        let (_dir, m) = test_manager();
        let s = session(&m);
        let op = 11u64;
        s.begin_verification_attempt(
            TASK,
            REV,
            op,
            ROOT,
            &[],
            &[job_check("make_test")],
            &[job_input("make_test")],
        )
        .unwrap();
        // Double claim refuses: exactly one executor per job.
        let claimed = s.claim_verification_job(TASK, "make_test", op, 99).unwrap();
        assert_eq!(claimed.state, VerificationJobState::Running);
        assert_eq!(claimed.op_id, Some(99));
        assert!(matches!(
            s.claim_verification_job(TASK, "make_test", op, 100),
            Err(SessionError::Conflict(_))
        ));
        // Resolve from a non-Running row refuses (resolve once).
        let done = s
            .resolve_verification_job(
                TASK,
                "make_test",
                op,
                VerificationJobState::Failed,
                None,
                Some(outcome_json("failed", 7)),
            )
            .unwrap();
        assert_eq!(done.state, VerificationJobState::Failed);
        assert!(done.finished_ms.is_some());
        assert_eq!(
            done.result_json.as_deref(),
            Some(outcome_json("failed", 7).as_str())
        );
        assert!(matches!(
            s.resolve_verification_job(
                TASK,
                "make_test",
                op,
                VerificationJobState::Failed,
                None,
                None
            ),
            Err(SessionError::Conflict(_))
        ));
        assert!(s.open_verification_jobs(TASK).unwrap().is_empty());
        // A resolve to a non-terminal state is malformed.
        assert!(matches!(
            s.resolve_verification_job(
                TASK,
                "make_test",
                op,
                VerificationJobState::Running,
                None,
                None
            ),
            Err(SessionError::Malformed(_))
        ));
    }

    #[test]
    fn open_job_of_another_attempt_refuses_begin_never_silently_replaced() {
        let (_dir, m) = test_manager();
        let s = session(&m);
        let begin = |op: u64| {
            s.begin_verification_attempt(
                TASK,
                REV,
                op,
                ROOT,
                &[],
                &[job_check("make_test")],
                &[job_input("make_test")],
            )
        };
        begin(20).unwrap();
        // A second attempt (newer op) with an overlapping open check refuses:
        // supersede first.
        let err = begin(21).unwrap_err();
        let conflict = match &err {
            SessionError::Conflict(msg) => msg.clone(),
            other => panic!("expected Conflict, got {other:?}"),
        };
        assert!(conflict.contains("supersede"), "{conflict}");
        // Supersede: the old attempt's open rows cancel with a typed note,
        // then the new attempt may begin.
        let cancelled = s
            .cancel_verification_attempt(TASK, 20, "superseded by attempt 21")
            .unwrap();
        assert_eq!(cancelled, 1);
        let jobs = s.verification_attempt_jobs(TASK, 20).unwrap();
        assert_eq!(jobs[0].state, VerificationJobState::Cancelled);
        assert!(jobs[0].note.as_deref().unwrap().contains("superseded"));
        begin(21).unwrap();
        let current = s.current_verification_attempt(TASK).unwrap().unwrap();
        assert_eq!(current.op_id, 21);
        let rows = s.verification_attempt_jobs(TASK, 21).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].state, VerificationJobState::Queued);
    }

    #[test]
    fn attempt_begin_is_idempotent_for_the_same_op() {
        let (_dir, m) = test_manager();
        let s = session(&m);
        s.begin_verification_attempt(
            TASK,
            REV,
            30,
            ROOT,
            &[],
            &[job_check("make_test")],
            &[job_input("make_test")],
        )
        .unwrap();
        s.begin_verification_attempt(
            TASK,
            REV,
            30,
            ROOT,
            &[],
            &[job_check("make_test")],
            &[job_input("make_test")],
        )
        .unwrap();
        assert_eq!(s.open_verification_jobs(TASK).unwrap().len(), 1);
        let attempt = s.current_verification_attempt(TASK).unwrap().unwrap();
        assert_eq!(attempt.changed.len(), 0);
        assert_eq!(attempt.checks.len(), 1);
    }

    #[test]
    fn hostile_input_is_rejected_never_truncated() {
        let (_dir, m) = test_manager();
        let s = session(&m);
        // Oversized spec JSON: typed rejection before any write.
        let mut job = job_input("make_test");
        job.spec_json = "x".repeat(MAX_VERIFICATION_JOB_SPEC_JSON_BYTES + 1);
        assert!(matches!(
            s.begin_verification_attempt(
                TASK,
                REV,
                40,
                ROOT,
                &[],
                &[job_check("make_test")],
                &[job.clone()]
            ),
            Err(SessionError::Oversized(_))
        ));
        // Oversized changed-file set (4097 > the record-aligned cap).
        let changed: Vec<String> = (0..=MAX_VERIFICATION_ATTEMPT_CHANGED)
            .map(|i| format!("f{i}"))
            .collect();
        assert!(matches!(
            s.begin_verification_attempt(
                TASK,
                REV,
                40,
                ROOT,
                &changed,
                &[job_check("x")],
                &[job_input("x")]
            ),
            Err(SessionError::Oversized(_))
        ));
        // Over-cap check count (257 > 256).
        let many_checks: Vec<VerificationAttemptCheck> = (0..=MAX_VERIFICATION_ATTEMPT_JOBS)
            .map(|i| job_check(&format!("c{i}")))
            .collect();
        let many_jobs: Vec<VerificationJobInput> = (0..=MAX_VERIFICATION_ATTEMPT_JOBS)
            .map(|i| job_input(&format!("c{i}")))
            .collect();
        assert!(matches!(
            s.begin_verification_attempt(TASK, REV, 40, ROOT, &[], &many_checks, &many_jobs),
            Err(SessionError::Oversized(_))
        ));
        // Bounded argv: 33 args and one oversized arg refuse before any write.
        let mut wide = job_input("make_test");
        wide.args = (0..=MAX_VERIFICATION_JOB_ARGS)
            .map(|i| format!("a{i}"))
            .collect();
        assert!(matches!(
            s.begin_verification_attempt(
                TASK,
                REV,
                40,
                ROOT,
                &[],
                &[job_check("make_test")],
                &[wide]
            ),
            Err(SessionError::Oversized(_))
        ));
        let mut long = job_input("make_test");
        long.args = vec!["x".repeat(MAX_VERIFICATION_JOB_ARG_BYTES + 1)];
        assert!(matches!(
            s.begin_verification_attempt(
                TASK,
                REV,
                40,
                ROOT,
                &[],
                &[job_check("make_test")],
                &[long]
            ),
            Err(SessionError::Oversized(_))
        ));
        // Empty check id is malformed.
        let mut bad = job_input("make_test");
        bad.check_id.clear();
        assert!(matches!(
            s.begin_verification_attempt(
                TASK,
                REV,
                40,
                ROOT,
                &[],
                &[job_check("make_test")],
                &[bad]
            ),
            Err(SessionError::Malformed(_))
        ));
        // A zero budget is rejected.
        let mut zero = job_input("make_test");
        zero.budget_ms = 0;
        assert!(matches!(
            s.begin_verification_attempt(
                TASK,
                REV,
                40,
                ROOT,
                &[],
                &[job_check("make_test")],
                &[zero]
            ),
            Err(SessionError::Oversized(_))
        ));
        // A job that no ordered check declares is malformed (never written).
        assert!(matches!(
            s.begin_verification_attempt(
                TASK,
                REV,
                40,
                ROOT,
                &[],
                &[job_check("declared")],
                &[job_input("make_test"), job_input("undeclared")]
            ),
            Err(SessionError::Malformed(_))
        ));
        assert!(s.open_verification_jobs(TASK).unwrap().is_empty());
        assert!(s.current_verification_attempt(TASK).unwrap().is_none());
        // An opaque spec body is accepted (the executor parses it when the
        // job runs and resolves Unavailable if it is undecodable — the
        // session never second-guesses the typed spec).
        let mut opaque = job_input("make_test");
        opaque.spec_json = "not json".into();
        s.begin_verification_attempt(
            TASK,
            REV,
            41,
            ROOT,
            &[],
            &[job_check("make_test")],
            &[opaque],
        )
        .unwrap();
        assert_eq!(s.open_verification_jobs(TASK).unwrap().len(), 1);
        assert_eq!(
            s.current_verification_attempt(TASK).unwrap().unwrap().op_id,
            41
        );
    }

    #[test]
    fn recover_requeues_running_rows_honestly_never_drops() {
        let (_dir, m) = test_manager();
        let s = session(&m);
        let op = 50u64;
        s.begin_verification_attempt(
            TASK,
            REV,
            op,
            ROOT,
            &[],
            &[job_check("make_test"), job_check("make_check")],
            &[job_input("make_test"), job_input("make_check")],
        )
        .unwrap();
        // The executor died mid-job: one row is Running (claimed, never
        // resolved), one Queued (never claimed).
        s.claim_verification_job(TASK, "make_test", op, 77).unwrap();
        let report = s.recover_verification_jobs_after_restart().unwrap();
        assert_eq!(report.requeued, 1);
        assert_eq!(report.orphaned, 0);
        let jobs = s.open_verification_jobs(TASK).unwrap();
        assert_eq!(jobs.len(), 2, "nothing is dropped or made terminal");
        let requeued = jobs.iter().find(|j| j.check_id == "make_test").unwrap();
        assert_eq!(requeued.state, VerificationJobState::Queued);
        assert_eq!(requeued.op_id, None);
        assert!(
            requeued.note.as_deref().unwrap().contains("re-queued"),
            "{requeued:?}"
        );
        // The re-queued row can be claimed and resolved again (the audit's
        // "re-run resolves").
        let claimed = s.claim_verification_job(TASK, "make_test", op, 78).unwrap();
        assert_eq!(claimed.state, VerificationJobState::Running);
        let done = s
            .resolve_verification_job(
                TASK,
                "make_test",
                op,
                VerificationJobState::Passed,
                None,
                Some(outcome_json("passed", 0)),
            )
            .unwrap();
        assert_eq!(done.state, VerificationJobState::Passed);
    }

    #[test]
    fn recovery_survives_a_real_reopen_of_the_session_layer() {
        // Adversarial (audit P0-5/26 restart honesty): rows left `Running`
        // by a vanished process must NOT be silently terminal after the
        // session layer reopens. The task row is untouched and the job is
        // honestly re-queued.
        let dir = tempfile::tempdir().unwrap();
        let m1 = Arc::new(
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap(),
        );
        let ws = m1.create_workspace(ROOT).unwrap();
        let sid = m1.create_session(ws, "jobs", "fake", "m").unwrap().id();
        let s1 = m1.get_session(sid).unwrap().unwrap();
        let op = 60u64;
        s1.begin_verification_attempt(
            TASK,
            REV,
            op,
            ROOT,
            &[],
            &[job_check("make_test")],
            &[job_input("make_test")],
        )
        .unwrap();
        // The process "dies" mid-job: the row is claimed (Running) and the
        // handle/manager are dropped without a resolve.
        s1.claim_verification_job(TASK, "make_test", op, 88)
            .unwrap();
        drop(s1);
        drop(m1);
        // Reopen the session layer and recover honestly.
        let m2 = Arc::new(
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap(),
        );
        let s2 = m2.get_session(sid).unwrap().unwrap();
        assert!(
            s2.current_verification_attempt(TASK).unwrap().is_some(),
            "the attempt record survives the reopen"
        );
        let report = s2.recover_verification_jobs_after_restart().unwrap();
        assert_eq!(report.requeued, 1);
        let jobs = s2.open_verification_jobs(TASK).unwrap();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].state, VerificationJobState::Queued);
        assert!(jobs[0].note.as_deref().unwrap().contains("restart"));
    }

    /// 100 changed files, 20 required checks (5 inline, 15 background),
    /// killed and reopened between transitions: every row survives and the
    /// attempt reconstructs exactly.
    #[test]
    fn attempts_survive_reopen_at_every_transition_and_reconstruct_exactly() {
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("store");
        let cas = dir.path().join("cas");
        let m1 = Arc::new(SessionManager::open(&store, &cas, true).unwrap());
        let ws = m1.create_workspace(ROOT).unwrap();
        let sid = m1.create_session(ws, "jobs-big", "fake", "m").unwrap().id();

        let changed: Vec<String> = (0..100).map(|i| format!("src/file{i:03}.rs")).collect();
        let mut checks: Vec<VerificationAttemptCheck> = Vec::new();
        for i in 0..5 {
            let status = match i {
                0 => VerificationInlineStatus::Failed,
                4 => VerificationInlineStatus::Unavailable,
                _ => VerificationInlineStatus::Passed,
            };
            checks.push(inline_run(&format!("inline_{i}"), status));
        }
        let mut jobs: Vec<VerificationJobInput> = Vec::new();
        for i in 0..15 {
            checks.push(job_check(&format!("bg_{i}")));
            jobs.push(job_input(&format!("bg_{i}")));
        }
        let op = 100u64;
        {
            let s1 = m1.get_session(sid).unwrap().unwrap();
            s1.begin_verification_attempt(TASK, REV, op, ROOT, &changed, &checks, &jobs)
                .unwrap();
            assert_eq!(s1.open_verification_jobs(TASK).unwrap().len(), 15);
        }
        drop(m1);
        // Reopen: the whole attempt (100 files, 20 checks) is durable.
        let m2 = Arc::new(SessionManager::open(&store, &cas, true).unwrap());
        {
            let s2 = m2.get_session(sid).unwrap().unwrap();
            let attempt = s2.current_verification_attempt(TASK).unwrap().unwrap();
            assert_eq!(attempt.op_id, op);
            assert_eq!(attempt.changed, changed, "every changed file survives");
            assert_eq!(attempt.checks.len(), 20, "every required check survives");
            assert_eq!(
                attempt.checks[0].inline,
                Some(VerificationInlineStatus::Failed)
            );
            assert_eq!(
                attempt.checks[4].inline,
                Some(VerificationInlineStatus::Unavailable)
            );
            assert!(attempt.checks[5].inline.is_none());
            assert_eq!(s2.open_verification_jobs(TASK).unwrap().len(), 15);
        }
        drop(m2);
        // Settle every background job, killing the layer after each one:
        // every transition is durable and no result is lost.
        for i in 0..15u64 {
            let m = Arc::new(SessionManager::open(&store, &cas, true).unwrap());
            let s = m.get_session(sid).unwrap().unwrap();
            let check = format!("bg_{i}");
            let claimed = s
                .claim_verification_job(TASK, &check, op, 1_000 + i)
                .unwrap();
            assert_eq!(claimed.state, VerificationJobState::Running);
            drop(s);
            drop(m);
            let m = Arc::new(SessionManager::open(&store, &cas, true).unwrap());
            let s = m.get_session(sid).unwrap().unwrap();
            let running = s.open_verification_jobs(TASK).unwrap();
            assert!(
                running.iter().any(|j| j.check_id == check),
                "the claim survives the reopen"
            );
            let status = if i % 3 == 0 { "failed" } else { "passed" };
            let exit = if status == "failed" { 3 } else { 0 };
            let done = s
                .resolve_verification_job(
                    TASK,
                    &check,
                    op,
                    if status == "failed" {
                        VerificationJobState::Failed
                    } else {
                        VerificationJobState::Passed
                    },
                    None,
                    Some(outcome_json(status, exit)),
                )
                .unwrap();
            assert_eq!(
                done.state == VerificationJobState::Passed,
                status == "passed"
            );
            drop(s);
            drop(m);
        }
        // Final reconstruction: the exact attempt + all settled job rows.
        let m3 = Arc::new(SessionManager::open(&store, &cas, true).unwrap());
        let s3 = m3.get_session(sid).unwrap().unwrap();
        let attempt = s3.current_verification_attempt(TASK).unwrap().unwrap();
        assert_eq!(attempt.changed, changed);
        assert_eq!(attempt.checks.len(), 20);
        assert_eq!(attempt.checks.len(), {
            let inline = attempt.checks.iter().filter(|c| c.inline.is_some()).count();
            assert_eq!(inline, 5);
            inline + 15
        });
        let settled = s3.verification_attempt_jobs(TASK, op).unwrap();
        assert_eq!(settled.len(), 15);
        for (i, row) in settled.iter().enumerate() {
            assert_eq!(row.ordinal, 5 + i as u32, "derivation order survives");
            let expect = if (i as u64).is_multiple_of(3) {
                VerificationJobState::Failed
            } else {
                VerificationJobState::Passed
            };
            assert_eq!(row.state, expect, "job {i} state survives");
            assert!(row.result_json.is_some(), "job {i} result survives");
        }
        assert!(s3.open_verification_jobs(TASK).unwrap().is_empty());
    }

    /// A late result for attempt N is typed-rejected once attempt N+1
    /// exists: attempt N is frozen, its rows are untouched, and the newer
    /// attempt's jobs are never mutated by it.
    #[test]
    fn late_result_for_attempt_n_is_rejected_after_n_plus_one() {
        let (_dir, m) = test_manager();
        let s = session(&m);
        // Attempt N: one job, claimed and left Running (a live executor).
        s.begin_verification_attempt(
            TASK,
            REV,
            10,
            ROOT,
            &[],
            &[job_check("make_test")],
            &[job_input("make_test")],
        )
        .unwrap();
        s.claim_verification_job(TASK, "make_test", 10, 1).unwrap();
        // Attempt N+1 for a DIFFERENT check (no open-job conflict): the
        // moment it commits, attempt N is superseded.
        s.begin_verification_attempt(
            TASK,
            REV,
            11,
            ROOT,
            &[],
            &[job_check("make_check")],
            &[job_input("make_check")],
        )
        .unwrap();
        let late = s
            .resolve_verification_job(
                TASK,
                "make_test",
                10,
                VerificationJobState::Passed,
                None,
                Some(outcome_json("passed", 0)),
            )
            .unwrap_err();
        match &late {
            SessionError::Conflict(msg) => {
                assert!(
                    msg.contains("superseded") && msg.contains("10") && msg.contains("11"),
                    "{msg}"
                );
            }
            other => panic!("expected superseded Conflict, got {other:?}"),
        }
        // The late result left attempt N untouched (still Running)...
        let old = s.verification_attempt_jobs(TASK, 10).unwrap();
        assert_eq!(old[0].state, VerificationJobState::Running);
        assert!(old[0].result_json.is_none());
        // ... and a late CLAIM of N is equally frozen.
        assert!(matches!(
            s.claim_verification_job(TASK, "make_test", 10, 2),
            Err(SessionError::Conflict(_))
        ));
        // The current attempt is N+1, still queued, never touched by the
        // late attempt-N write.
        let current = s.current_verification_attempt(TASK).unwrap().unwrap();
        assert_eq!(current.op_id, 11);
        let new_jobs = s.verification_attempt_jobs(TASK, 11).unwrap();
        assert_eq!(new_jobs[0].state, VerificationJobState::Queued);
    }

    fn fingerprint_fixture() -> EnvironmentFingerprint {
        EnvironmentFingerprint {
            platform: "macos".into(),
            arch: "aarch64".into(),
            toolchain_versions: vec![faktor_core::state::ToolVersion {
                tool: "faktor-agent".into(),
                version: "0.1.0".into(),
            }],
            manifest_hashes: vec![faktor_core::state::FingerprintFileHash {
                path: "Cargo.toml".into(),
                digest_hex: "ab".repeat(32),
            }],
            lockfile_hashes: vec![],
            instruction_epoch: Some(3),
            base_tree_hash: None,
            task_contract_hash: "ef".repeat(32),
            check_argv_cwd_env_hash: "12".repeat(32),
            verification_impl_version: "faktor-agent/0.1.0".into(),
        }
    }

    #[test]
    fn fingerprint_rides_attempt_and_job_rows_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let m1 = Arc::new(
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap(),
        );
        let ws = m1.create_workspace(ROOT).unwrap();
        let sid = m1
            .create_session(ws, "fingerprint", "fake", "m")
            .unwrap()
            .id();
        let s1 = m1.get_session(sid).unwrap().unwrap();
        let fp = fingerprint_fixture();
        let op = 90u64;
        s1.begin_verification_attempt_with_fingerprint(
            TASK,
            REV,
            op,
            ROOT,
            &[],
            &[job_check("make_test")],
            &[job_input("make_test")],
            Some(fp.clone()),
        )
        .unwrap();
        let attempt = s1.current_verification_attempt(TASK).unwrap().unwrap();
        assert_eq!(attempt.environment_fingerprint.as_ref(), Some(&fp));
        let jobs = s1.open_verification_jobs(TASK).unwrap();
        assert_eq!(jobs[0].environment_fingerprint.as_ref(), Some(&fp));
        drop(s1);
        drop(m1);
        let m2 = Arc::new(
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap(),
        );
        let s2 = m2.get_session(sid).unwrap().unwrap();
        let attempt = s2.current_verification_attempt(TASK).unwrap().unwrap();
        assert_eq!(
            attempt.environment_fingerprint.as_ref(),
            Some(&fp),
            "the attempt fingerprint survives the reopen"
        );
        let jobs = s2.open_verification_jobs(TASK).unwrap();
        assert_eq!(jobs[0].environment_fingerprint.as_ref(), Some(&fp));
        // A fingerprint-free attempt reads an honest absence, never a
        // retro-fitted value.
        s2.begin_verification_attempt(
            TASK,
            REV + 1,
            91,
            ROOT,
            &[],
            &[job_check("make_check")],
            &[job_input("make_check")],
        )
        .unwrap();
        let plain = s2.verification_attempt(TASK, 91).unwrap().unwrap();
        assert!(plain.environment_fingerprint.is_none());
    }
}
