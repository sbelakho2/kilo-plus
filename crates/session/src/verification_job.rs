//! Durable background verification jobs (audit P0-5/P0-6/26/27).
//!
//! One `VerificationJob` row = ONE required check a verification attempt
//! could not run inline: the typed (program, argv) [`CheckSpec`] is stored
//! verbatim (opaque JSON — this crate is dependency-free of `faktor-verify`,
//! so the spec travels as bounded `spec_json`), the row walks
//! `Queued -> Running -> Passed|Failed|Unavailable|Cancelled`, and every
//! transition is a guarded CAS so a job settles exactly once.
//!
//! Storage: the compaction-proof durable `memory_fact` rows (the wave-9
//! pattern the task_state/verification rows already use) under the kinds
//! `verification_job` (one row per check) and `verification_attempt` (one
//! row per attempt: the derivation-ordered required checks, the changed
//! files the checks certified and the inline-run outcomes — everything a
//! later completion needs to rebuild the attempt's proof). Additive reuse:
//! no schema migration; the rows are plain durable facts compaction never
//! rewrites.
//!
//! Honest recovery ([`SessionHandle::recover_verification_jobs_after_restart`]):
//! a row the previous process left `Running` (its executor died mid-check)
//! is re-queued with a typed note — never silently terminal, and never able
//! to certify completion from a vanished process. A `Queued` row of a
//! vanished process simply stays queued (it was never claimed). Job rows
//! whose attempt record is missing (a crash inside `begin_verification_attempt`)
//! are orphaned to `Unavailable` with a typed note — they can never settle
//! anything.
//!
//! Everything is synchronous and bounded: values ride the 4096-byte durable
//! fact cap; oversized or malformed input is rejected with typed errors
//! before any write; a hostile/corrupt row under our kinds is a LOUD typed
//! decode error on every read path — never a silent drop.

use serde::{Deserialize, Serialize};

use crate::handle::SessionHandle;
use crate::SessionError;

// ---------------------------------------------------------------- bounds

/// Hard bound on one stored job check-id (facts keys embed it).
pub const MAX_VERIFICATION_JOB_CHECK_ID_BYTES: usize = 128;
/// Hard bound on one stored job kind tag (`compile`/`test`/`lint`).
pub const MAX_VERIFICATION_JOB_KIND_BYTES: usize = 16;
/// Hard bound on one stored job canonical command text.
pub const MAX_VERIFICATION_JOB_COMMAND_BYTES: usize = 512;
/// Hard bound on the typed spec JSON of one job (the `CheckSpec` serde form;
/// program + argv must fit — hostile oversized args are rejected, never
/// truncated).
pub const MAX_VERIFICATION_JOB_SPEC_JSON_BYTES: usize = 1800;
/// Hard bound on the typed result JSON of one job (the `CheckOutcome` serde
/// form: status/exit/timestamps and the bounded output tail).
pub const MAX_VERIFICATION_JOB_RESULT_JSON_BYTES: usize = 1024;
/// Hard bound on one job's execution budget in ms.
pub const MAX_VERIFICATION_JOB_BUDGET_MS: u64 = 3_600_000;
/// Hard bound on the durable note of one job/attempt transition.
pub const MAX_VERIFICATION_JOB_NOTE_BYTES: usize = 512;
/// Changed files an attempt may certify.
pub const MAX_VERIFICATION_ATTEMPT_CHANGED: usize = 16;
/// One changed-file path bound (mirrors the durable path caps).
pub const MAX_VERIFICATION_ATTEMPT_PATH_BYTES: usize = 200;
/// Background checks one attempt may enqueue (also the cap on the ordered
/// required-checks list — inline and background together).
pub const MAX_VERIFICATION_ATTEMPT_JOBS: usize = 8;
/// Workspace-root bound of one attempt.
pub const MAX_VERIFICATION_JOB_ROOT_BYTES: usize = 4096;

/// The durable fact-value ceiling (kind `memory_fact` values are capped at
/// 4096 bytes by the memory module; every row we write honors it).
const MAX_JOB_ROW_VALUE_BYTES: usize = 4096;

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
    /// The typed spec as serde JSON (opaque to this crate; parsed by the
    /// agent executor that runs the job).
    pub spec_json: String,
    /// Wall deadline of one execution of this job, in ms.
    pub budget_ms: u64,
}

/// One durable verification job row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationJob {
    /// `"{task_id}:{check_id}"` — stable identity (the durable fact key).
    pub id: String,
    pub task_id: u64,
    /// The task revision the job's attempt certified at enqueue.
    pub task_revision: u64,
    /// The durable workspace root the check runs in (never the daemon cwd).
    pub workspace_root: String,
    pub check_id: String,
    pub kind: String,
    pub command: String,
    /// The typed spec as serde JSON (opaque here).
    pub spec_json: String,
    pub budget_ms: u64,
    /// The attempt (enqueueing turn op) this row belongs to.
    pub attempt_op: u64,
    pub state: VerificationJobState,
    /// Typed note: recovery/abort/supersede explanations.
    pub note: Option<String>,
    /// The op that claimed this job (the executor run), while Running.
    pub op_id: Option<u64>,
    /// The typed outcome as serde JSON (`CheckOutcome`), when terminal by
    /// execution (Passed/Failed/Unavailable).
    pub result_json: Option<String>,
    pub created_ms: i64,
    pub updated_ms: i64,
    pub finished_ms: Option<i64>,
}

/// How one INLINE required check of an attempt ended (the attempt record
/// carries the outcome so a later settlement can rebuild the full result
/// set from durable rows — inline checks ran in the enqueueing turn and
/// their outcomes must survive it).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationInlineStatus {
    Passed,
    Failed,
    /// Ran inline but produced no verdict (deadline/cancellation kill,
    /// program missing, infra): never a failure of the code under check.
    Unavailable,
}

/// One required check of an attempt, IN DERIVATION ORDER: inline checks
/// carry their outcome, background checks carry `inline = None` (the
/// check's spec lives in its job row). The order is what later completions
/// need: acceptance-criteria entries and the criteria-verdict mapping are
/// index-aligned with this list.
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
/// needs to rebuild the attempt's proof from its job rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationAttempt {
    pub task_id: u64,
    /// The attempt's identity: the op id of the enqueueing turn.
    pub op_id: u64,
    /// The task revision the attempt certified at enqueue.
    pub task_revision: u64,
    pub workspace_root: String,
    /// Files changed by the enqueueing turn (the evidence the attempt
    /// certifies).
    pub changed: Vec<String>,
    /// The required checks in derivation order (inline outcomes inline).
    pub checks: Vec<VerificationAttemptCheck>,
    pub created_ms: i64,
}

/// Report of one honest recovery sweep.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VerificationJobRecoveryReport {
    /// `Running` rows re-queued with a typed note (the previous executor
    /// died mid-check).
    pub requeued: usize,
    /// Open rows orphaned to `Unavailable` (their attempt record is
    /// missing — a crash inside the attempt begin).
    pub orphaned: usize,
}

// ------------------------------------------------------------- row layout
//
// memory_fact rows written by this module:
//   kind "verification_job",    key "vj:{task_id}:{check_id}"
//   kind "verification_attempt", key "va:{task_id}:{op_id}"
// Both values are versioned JSON (`schema_ver: 1`); a row that fails its
// schema decode is a loud Malformed error on every read — never a silent
// drop and never a guess.

const JOB_KIND: &str = "verification_job";
const ATTEMPT_KIND: &str = "verification_attempt";
const SCHEMA_VER: i64 = 1;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct JobRowValue {
    schema_ver: i64,
    attempt_op: u64,
    task_id: u64,
    task_revision: u64,
    workspace_root: String,
    check_id: String,
    kind: String,
    command: String,
    spec_json: String,
    budget_ms: u64,
    state: VerificationJobState,
    note: Option<String>,
    op_id: Option<u64>,
    result_json: Option<String>,
    created_ms: i64,
    updated_ms: i64,
    finished_ms: Option<i64>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AttemptRowValue {
    schema_ver: i64,
    task_id: u64,
    op_id: u64,
    task_revision: u64,
    workspace_root: String,
    changed: Vec<String>,
    checks: Vec<VerificationAttemptCheck>,
    created_ms: i64,
}

fn job_key(task_id: u64, check_id: &str) -> String {
    format!("vj:{task_id}:{check_id}")
}

fn attempt_key(task_id: u64, op_id: u64) -> String {
    format!("va:{task_id}:{op_id}")
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

/// Decode one row value; hostile/foreign shapes are loud.
fn decode_job_value(key: &str, value: &str) -> Result<JobRowValue, SessionError> {
    let v: JobRowValue = serde_json::from_str(value)
        .map_err(|e| malformed_row(key, &format!("undecodable json: {e}")))?;
    if v.schema_ver != SCHEMA_VER {
        return Err(malformed_row(
            key,
            &format!(
                "unknown schema version {} (this reader understands v{SCHEMA_VER})",
                v.schema_ver
            ),
        ));
    }
    Ok(v)
}

fn decode_attempt_value(key: &str, value: &str) -> Result<AttemptRowValue, SessionError> {
    let v: AttemptRowValue = serde_json::from_str(value)
        .map_err(|e| malformed_row(key, &format!("undecodable json: {e}")))?;
    if v.schema_ver != SCHEMA_VER {
        return Err(malformed_row(
            key,
            &format!(
                "unknown schema version {} (this reader understands v{SCHEMA_VER})",
                v.schema_ver
            ),
        ));
    }
    Ok(v)
}

fn project_job(row: JobRowValue) -> VerificationJob {
    VerificationJob {
        id: job_key(row.task_id, &row.check_id),
        task_id: row.task_id,
        task_revision: row.task_revision,
        workspace_root: row.workspace_root,
        check_id: row.check_id,
        kind: row.kind,
        command: row.command,
        spec_json: row.spec_json,
        budget_ms: row.budget_ms,
        attempt_op: row.attempt_op,
        state: row.state,
        note: row.note,
        op_id: row.op_id,
        result_json: row.result_json,
        created_ms: row.created_ms,
        updated_ms: row.updated_ms,
        finished_ms: row.finished_ms,
    }
}

fn project_attempt(row: AttemptRowValue) -> VerificationAttempt {
    VerificationAttempt {
        task_id: row.task_id,
        op_id: row.op_id,
        task_revision: row.task_revision,
        workspace_root: row.workspace_root,
        changed: row.changed,
        checks: row.checks,
        created_ms: row.created_ms,
    }
}

// ------------------------------------------------------------- the surface

impl SessionHandle {
    /// Begin one verification attempt durably (audit P0-5/26): the attempt
    /// record (changed files, derivation-ordered required checks with their
    /// inline outcomes and background job definitions) plus one `Queued` job
    /// row per background check. Job rows are written first and the attempt
    /// record LAST — the record is the commit point, so a crash inside the
    /// begin leaves only orphaned job rows that recovery honestly resolves.
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
            seen_jobs.insert(job.check_id.as_str());
        }
        for c in checks {
            if c.inline.is_none() && !seen_jobs.contains(c.check_id.as_str()) {
                return Err(SessionError::Malformed(format!(
                    "background check '{}' has no job row definition",
                    c.check_id
                )));
            }
        }
        // The attempt value must fit the durable fact cap before any write.
        let attempt_value = AttemptRowValue {
            schema_ver: SCHEMA_VER,
            task_id,
            op_id,
            task_revision,
            workspace_root: workspace_root.to_string(),
            changed: changed.to_vec(),
            checks: checks.to_vec(),
            created_ms: self.now_ms(),
        };
        let attempt_text = serde_json::to_string(&attempt_value)
            .map_err(|e| SessionError::Internal(format!("attempt json: {e}")))?;
        if attempt_text.len() > MAX_JOB_ROW_VALUE_BYTES {
            return Err(SessionError::Oversized(format!(
                "verification attempt record of {} bytes exceeds the {MAX_JOB_ROW_VALUE_BYTES}-byte durable row cap",
                attempt_text.len()
            )));
        }

        let _guard = self.command_guard();
        let now = self.now_ms();
        let facts = self.facts_of_kinds(&[JOB_KIND, ATTEMPT_KIND])?;
        if facts
            .iter()
            .any(|(kind, key, _)| kind == ATTEMPT_KIND && *key == attempt_key(task_id, op_id))
        {
            return Ok(()); // idempotent retry of a crashed begin
        }
        // Job rows (Queued), keyed per check; an OPEN row of ANOTHER attempt
        // under the same check refuses — never a silent replacement.
        for job in jobs {
            let key = job_key(task_id, &job.check_id);
            let prior = facts.iter().find(|(k, kk, _)| k == JOB_KIND && kk == &key);
            if let Some((_, _, value)) = prior {
                let prior = decode_job_value(&key, value)?;
                if prior.attempt_op != op_id && prior.state.is_open() {
                    return Err(SessionError::Conflict(format!(
                        "check '{}' has an open job of attempt {}; supersede that attempt before \
                         beginning attempt {op_id}",
                        job.check_id, prior.attempt_op
                    )));
                }
            }
            let row = JobRowValue {
                schema_ver: SCHEMA_VER,
                attempt_op: op_id,
                task_id,
                task_revision,
                workspace_root: workspace_root.to_string(),
                check_id: job.check_id.clone(),
                kind: job.kind.clone(),
                command: job.command.clone(),
                spec_json: job.spec_json.clone(),
                budget_ms: job.budget_ms,
                state: VerificationJobState::Queued,
                note: None,
                op_id: None,
                result_json: None,
                created_ms: now,
                updated_ms: now,
                finished_ms: None,
            };
            self.put_fact(JOB_KIND, &key, &row)?;
        }
        // Commit point: the attempt record.
        self.upsert_fact(ATTEMPT_KIND, &attempt_key(task_id, op_id), &attempt_text)?;
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
        let _guard = self.command_guard();
        let mut cancelled = 0usize;
        for (key, value) in self.rows_of_kind(JOB_KIND)? {
            let mut row = decode_job_value(&key, &value)?;
            if row.task_id != task_id || row.attempt_op != attempt_op || !row.state.is_open() {
                continue;
            }
            row.state = VerificationJobState::Cancelled;
            row.note = Some(reason.to_string());
            row.updated_ms = self.now_ms();
            row.finished_ms = Some(row.updated_ms);
            self.put_fact(JOB_KIND, &key, &row)?;
            cancelled += 1;
        }
        Ok(cancelled)
    }

    /// The NEWEST attempt record of `task_id`, or `None`. "Newest" is the
    /// highest attempt op (session op ids are monotonic). Open job rows of
    /// older attempts can only exist as crash residue (a newer attempt
    /// supersedes explicitly), so callers settle the newest record.
    pub fn current_verification_attempt(
        &self,
        task_id: u64,
    ) -> Result<Option<VerificationAttempt>, SessionError> {
        let mut newest: Option<(u64, VerificationAttempt)> = None;
        for (key, value) in self.rows_of_kind(ATTEMPT_KIND)? {
            let row = decode_attempt_value(&key, &value)?;
            if row.task_id != task_id {
                continue;
            }
            if newest.as_ref().is_none_or(|(op, _)| row.op_id > *op) {
                newest = Some((row.op_id, project_attempt(row)));
            }
        }
        Ok(newest.map(|(_, a)| a))
    }

    /// The attempt record of `task_id` for a specific op (settlement reads
    /// the exact attempt its job rows belong to).
    pub fn verification_attempt(
        &self,
        task_id: u64,
        attempt_op: u64,
    ) -> Result<Option<VerificationAttempt>, SessionError> {
        let key = attempt_key(task_id, attempt_op);
        let value = self.fact_value(ATTEMPT_KIND, &key)?;
        match value {
            Some(v) => Ok(Some(project_attempt(decode_attempt_value(&key, &v)?))),
            None => Ok(None),
        }
    }

    /// Every OPEN (Queued|Running) job row of `task_id`, deterministic
    /// (task, check-id) order.
    pub fn open_verification_jobs(
        &self,
        task_id: u64,
    ) -> Result<Vec<VerificationJob>, SessionError> {
        let mut out = Vec::new();
        for (key, value) in self.rows_of_kind(JOB_KIND)? {
            let row = decode_job_value(&key, &value)?;
            if row.task_id == task_id && row.state.is_open() {
                out.push(project_job(row));
            }
        }
        out.sort_by(|a, b| a.check_id.cmp(&b.check_id));
        Ok(out)
    }

    /// Every job row of one attempt (any state), deterministic check-id
    /// order. Rows of superseded attempts that linger under checks the new
    /// attempt does not re-derive stay readable here (they are terminal).
    pub fn verification_attempt_jobs(
        &self,
        task_id: u64,
        attempt_op: u64,
    ) -> Result<Vec<VerificationJob>, SessionError> {
        let mut out = Vec::new();
        for (key, value) in self.rows_of_kind(JOB_KIND)? {
            let row = decode_job_value(&key, &value)?;
            if row.task_id == task_id && row.attempt_op == attempt_op {
                out.push(project_job(row));
            }
        }
        out.sort_by(|a, b| a.check_id.cmp(&b.check_id));
        Ok(out)
    }

    /// Claim one `Queued` job of `attempt_op` for execution: the guarded
    /// CAS to `Running` (with the executor's op attached). A row that is not
    /// `Queued`, belongs to another attempt, or no longer exists refuses
    /// with a typed error — a job is executed at most once per claim.
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
        let _guard = self.command_guard();
        let key = job_key(task_id, check_id);
        let value = self
            .fact_value(JOB_KIND, &key)?
            .ok_or_else(|| SessionError::NotFound(format!("verification job '{check_id}'")))?;
        let mut row = decode_job_value(&key, &value)?;
        if row.attempt_op != attempt_op {
            return Err(SessionError::Conflict(format!(
                "job '{check_id}' belongs to attempt {}, not {attempt_op}",
                row.attempt_op
            )));
        }
        if row.state != VerificationJobState::Queued {
            return Err(SessionError::Conflict(format!(
                "job '{check_id}' is {:?}, not Queued — a job is claimed exactly once per attempt",
                row.state
            )));
        }
        row.state = VerificationJobState::Running;
        row.op_id = Some(op_id);
        row.updated_ms = self.now_ms();
        self.put_fact(JOB_KIND, &key, &row)?;
        Ok(project_job(row))
    }

    /// Resolve one `Running` job to its terminal state (CAS): `Passed`,
    /// `Failed`, `Unavailable` (the typed outcome rides `result_json`), or
    /// `Cancelled`. The row must exist, belong to `attempt_op` and be
    /// `Running` — anything else is a typed refusal (never a blind write).
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
        let _guard = self.command_guard();
        let key = job_key(task_id, check_id);
        let value = self
            .fact_value(JOB_KIND, &key)?
            .ok_or_else(|| SessionError::NotFound(format!("verification job '{check_id}'")))?;
        let mut row = decode_job_value(&key, &value)?;
        if row.attempt_op != attempt_op {
            return Err(SessionError::Conflict(format!(
                "job '{check_id}' belongs to attempt {}, not {attempt_op}",
                row.attempt_op
            )));
        }
        if row.state != VerificationJobState::Running {
            return Err(SessionError::Conflict(format!(
                "job '{check_id}' is {:?}, not Running — a job resolves exactly once",
                row.state
            )));
        }
        row.state = state;
        row.note = note;
        row.result_json = result_json;
        row.updated_ms = self.now_ms();
        row.finished_ms = Some(row.updated_ms);
        self.put_fact(JOB_KIND, &key, &row)?;
        Ok(project_job(row))
    }

    /// Honest post-restart recovery (audit P0-5/26): every `Running` row —
    /// the previous process died while its executor was mid-check — is
    /// re-queued with a typed note (the check never certified anything, so
    /// re-running it deterministically is the only honest path to a
    /// verdict). Open rows whose attempt record is missing (a crash inside
    /// the attempt begin) are orphaned to `Unavailable` with a typed note —
    /// they can never settle a claim. Idempotent; `Queued` rows of a
    /// vanished process are untouched (they were never claimed).
    pub fn recover_verification_jobs_after_restart(
        &self,
    ) -> Result<VerificationJobRecoveryReport, SessionError> {
        let _guard = self.command_guard();
        let mut report = VerificationJobRecoveryReport::default();
        let job_rows: Vec<(String, JobRowValue)> = self
            .rows_of_kind(JOB_KIND)?
            .into_iter()
            .map(|(key, value)| decode_job_value(&key, &value).map(|row| (key.clone(), row)))
            .collect::<Result<_, _>>()?;
        // Attempt existence probe (one scan; small fact sets).
        let attempt_keys: Vec<String> = self
            .rows_of_kind(ATTEMPT_KIND)?
            .into_iter()
            .map(|(k, _)| k)
            .collect();
        for (key, mut row) in job_rows {
            if !row.state.is_open() {
                continue;
            }
            let has_attempt = attempt_keys.contains(&attempt_key(row.task_id, row.attempt_op));
            if !has_attempt {
                row.state = VerificationJobState::Unavailable;
                row.note = Some(
                    "orphaned job: its attempt record is missing (a crash inside the attempt \
                     begin); never certified"
                        .into(),
                );
                row.updated_ms = self.now_ms();
                row.finished_ms = Some(row.updated_ms);
                self.put_fact(JOB_KIND, &key, &row)?;
                report.orphaned += 1;
                continue;
            }
            if row.state == VerificationJobState::Running {
                row.state = VerificationJobState::Queued;
                row.note = Some(
                    "re-queued after a restart: the previous executor died mid-check and never \
                     produced a verdict; the check re-runs deterministically"
                        .into(),
                );
                row.updated_ms = self.now_ms();
                row.finished_ms = None;
                row.op_id = None;
                self.put_fact(JOB_KIND, &key, &row)?;
                report.requeued += 1;
            }
        }
        Ok(report)
    }

    // ------------------------------------------------------ private readers

    fn fact_value(&self, kind: &str, key: &str) -> Result<Option<String>, SessionError> {
        Ok(self
            .manager
            .store()
            .memory_facts(self.id)
            .map_err(crate::map_store_err)?
            .into_iter()
            .find(|(k, kk, _)| k == kind && kk == key)
            .map(|(_, _, v)| v))
    }

    fn facts_of_kinds(
        &self,
        kinds: &[&str],
    ) -> Result<Vec<(String, String, String)>, SessionError> {
        Ok(self
            .manager
            .store()
            .memory_facts(self.id)
            .map_err(crate::map_store_err)?
            .into_iter()
            .filter(|(k, _, _)| kinds.contains(&k.as_str()))
            .collect())
    }

    fn rows_of_kind(&self, kind: &str) -> Result<Vec<(String, String)>, SessionError> {
        Ok(self
            .manager
            .store()
            .memory_facts(self.id)
            .map_err(crate::map_store_err)?
            .into_iter()
            .filter(|(k, _, _)| k == kind)
            .map(|(_, key, value)| (key, value))
            .collect())
    }

    fn put_fact<T: serde::Serialize>(
        &self,
        kind: &str,
        key: &str,
        row: &T,
    ) -> Result<(), SessionError> {
        let text = serde_json::to_string(row)
            .map_err(|e| SessionError::Internal(format!("{kind} json: {e}")))?;
        self.upsert_fact(kind, key, &text)
    }

    fn upsert_fact(&self, kind: &str, key: &str, text: &str) -> Result<(), SessionError> {
        if text.len() > MAX_JOB_ROW_VALUE_BYTES {
            return Err(SessionError::Oversized(format!(
                "{kind} row of {} bytes exceeds the {MAX_JOB_ROW_VALUE_BYTES}-byte durable cap",
                text.len()
            )));
        }
        self.manager
            .store()
            .upsert_memory_fact(self.id, kind, key, text)
            .map_err(crate::map_store_err)
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
        let outcome = serde_json::json!({
            "status": "failed",
            "exit": 7,
            "started_ms": 1,
            "finished_ms": 2,
            "summary": "boom",
            "truncated": false,
        })
        .to_string();
        let done = s
            .resolve_verification_job(
                TASK,
                "make_test",
                op,
                VerificationJobState::Failed,
                None,
                Some(outcome),
            )
            .unwrap();
        assert_eq!(done.state, VerificationJobState::Failed);
        assert!(done.finished_ms.is_some());
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
        // Oversized changed-file set.
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
        let outcome = serde_json::json!({
            "status": "passed",
            "exit": 0,
            "started_ms": 1,
            "finished_ms": 2,
            "summary": null,
            "truncated": false,
        })
        .to_string();
        let done = s
            .resolve_verification_job(
                TASK,
                "make_test",
                op,
                VerificationJobState::Passed,
                None,
                Some(outcome),
            )
            .unwrap();
        assert_eq!(done.state, VerificationJobState::Passed);
    }

    #[test]
    fn recovery_survives_a_real_reopen_of_the_session_layer() {
        // Adversarial (audit P0-5/26 restart honesty): rows left `Running`
        // by a vanished process must NOT be silently terminal after the
        // session layer reopens. The task row is untouched (never
        // VerifiedComplete from a vanished process) and the job is honestly
        // re-queued.
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

    #[test]
    fn orphaned_job_rows_without_an_attempt_never_settle() {
        // A crash inside begin_verification_attempt (job rows written, the
        // attempt record not yet): recovery orphans the open rows to
        // Unavailable with a typed note — they can never settle a claim.
        let (_dir, m) = test_manager();
        let s = session(&m);
        let op = 70u64;
        // Simulate the torn begin: write the job row directly under the
        // documented row layout, no attempt record.
        s.upsert_fact(
            JOB_KIND,
            &job_key(TASK, "torn"),
            &serde_json::json!({
                "schema_ver": 1,
                "attempt_op": op,
                "task_id": TASK,
                "task_revision": REV,
                "workspace_root": ROOT,
                "check_id": "torn",
                "kind": "test",
                "command": "ctest torn",
                "spec_json": spec("torn", "ctest", &[]),
                "budget_ms": 600_000,
                "state": "running",
                "note": null,
                "op_id": 5,
                "result_json": null,
                "created_ms": 1,
                "updated_ms": 1,
                "finished_ms": null,
            })
            .to_string(),
        )
        .unwrap();
        let report = s.recover_verification_jobs_after_restart().unwrap();
        assert_eq!(report.requeued, 0);
        assert_eq!(report.orphaned, 1);
        let rows = s.verification_attempt_jobs(TASK, op).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].state, VerificationJobState::Unavailable);
        assert!(
            rows[0].note.as_deref().unwrap().contains("orphaned"),
            "{rows:?}"
        );
        assert!(s.open_verification_jobs(TASK).unwrap().is_empty());
    }

    #[test]
    fn hostile_row_values_are_loud_errors_never_silent_drops() {
        let (_dir, m) = test_manager();
        let s = session(&m);
        // A hostile/corrupt row under the job kind must surface loudly.
        s.manager
            .store()
            .upsert_memory_fact(s.id, JOB_KIND, &job_key(TASK, "evil"), "not json")
            .unwrap();
        assert!(matches!(
            s.open_verification_jobs(TASK),
            Err(SessionError::Malformed(_))
        ));
        assert!(matches!(
            s.recover_verification_jobs_after_restart(),
            Err(SessionError::Malformed(_))
        ));
        // A future schema version is equally loud.
        s.manager
            .store()
            .upsert_memory_fact(
                s.id,
                JOB_KIND,
                &job_key(TASK, "future"),
                &serde_json::json!({"schema_ver": 99}).to_string(),
            )
            .unwrap();
        assert!(matches!(
            s.open_verification_jobs(TASK),
            Err(SessionError::Malformed(_))
        ));
    }
}
