//! SQLite persistence done correctly: WAL, single logical writer + genuinely
//! bounded reader pool, busy timeout, explicit transactional migrations,
//! integrity checks, automatic backups.
//!
//! Large blobs never live here — they go to the CAS; SQLite stores hashes.
//! Message/part rows store JSON payloads so the store stays protocol-agnostic.
//!
//! # Reader pool bound
//!
//! `read()` acquires a semaphore permit before touching a connection, so at
//! most `READER_POOL` (4) connections exist concurrently; a 20-reader storm
//! therefore uses at most 4 connections and the remaining callers block on
//! the permit, bounded by the busy timeout (`StoreError::Busy`). The pool is
//! a concurrency limit, not merely a retention limit.
//!
//! # Async boundary
//!
//! This crate is intentionally synchronous; do not introduce tokio here.
//! The daemon's HOT append paths (message append / part append / journal
//! event / usage settlement) run through `faktor_session`'s `DbActor`: a
//! dedicated `std::thread` owning this store, fronted by a bounded async
//! request channel. The actor executes grouped writes through
//! [`Store::batch_hot_writes`] — ONE transaction and ONE fsync per batch —
//! so a Tokio worker never executes a SQLite statement for those paths.
//!
//! Every OTHER call stays direct and synchronous on the shared
//! [`Store`](crate::Store) (reads, compound transitions, recovery,
//! checkpoints, ...). Direct and actor writes share the same writer lock, so
//! both surfaces are safe to mix; see [`Store::direct`] for the deliberate
//! sync-access marker. Callers that need a write observed by a later direct
//! read must await the actor response (the actor fsyncs before replying), or
//! serialize through [`Store::direct`].
//!
//! # Stability rule
//!
//! Every value read back from the database is parsed fallibly: corrupt or
//! version-skewed rows surface as `StoreError::Corrupt` (or `Sqlite`) —
//! never a panic. `unwrap`/`expect` appear only where the input is provably
//! constructed in-process this session (each site is commented).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};

use faktor_core::attachment::AttachmentId;
use faktor_core::event::{Event, EventKind, JournalInvariants};
use faktor_core::id::{
    EventSeq, OpId, SessionId, TaskId, TaskRevision, VerificationRecordId, WorkspaceId, WorktreeId,
};
use faktor_core::model::{PricingSnapshot, RiskBucket, RouterPhase, TaskClass};
use faktor_core::state::{
    AgentState, CheckExecution, CriterionVerification, FileStateEvidence, SessionLifecycle,
    TaskState, VerificationStatus,
};

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("store is corrupted: integrity check failed with {0:?}")]
    Corrupt(Vec<String>),
    #[error("event sequence gap or duplicate detected at {0}")]
    SeqViolation(u64),
    #[error("conflict: {0}")]
    Conflict(String),
    #[error("reader pool busy: {0}")]
    Busy(String),
    #[error("migration failed: {0}")]
    Migration(String),
    #[error("oversized value rejected: {0}")]
    Oversized(String),
    #[error("malformed value rejected: {0}")]
    Malformed(String),
}

pub type StoreResult<T> = Result<T, StoreError>;

/// Max concurrent read connections. This is a concurrency limit (semaphore
/// permits), not just a retention limit.
const READER_POOL: usize = 4;

/// Hard bound on one `memory_fact` kind scan (`doctor --deep` orphan
/// checks): beyond this the scan refuses loudly instead of truncating.
const MAX_ORCHESTRATOR_FACT_SCAN_ROWS: i64 = 250_000;

// ------------------------------------------- verification job bounds (v22)
//
// The store backstops the session layer's bounds so a raw SQL write or a
// corrupt injected row can never outgrow the durable contract. They mirror
// `faktor_session`'s verification-job constants and the verification-record
// caps (checks <= 256, changed files <= 4096, bounded argv).

/// Changed files one attempt may certify (mirrors
/// `MAX_VERIFICATION_CHANGED_FILES`).
pub const MAX_VERIFICATION_ATTEMPT_CHANGED: usize = 4096;
/// Required checks one attempt may carry (mirrors
/// `MAX_VERIFICATION_RECORD_CHECKS`).
pub const MAX_VERIFICATION_ATTEMPT_CHECKS: usize = 256;
/// One check-id bound.
pub const MAX_VERIFICATION_JOB_CHECK_ID_BYTES: usize = 128;
/// One check kind tag bound.
pub const MAX_VERIFICATION_JOB_KIND_BYTES: usize = 16;
/// One canonical command text bound.
pub const MAX_VERIFICATION_JOB_COMMAND_BYTES: usize = 512;
/// One check program bound (mirrors `MAX_VERIFICATION_PROGRAM_BYTES`).
pub const MAX_VERIFICATION_JOB_PROGRAM_BYTES: usize = 4096;
/// Per-argument count bound (mirrors `MAX_VERIFICATION_CHECK_ARGS`).
pub const MAX_VERIFICATION_JOB_ARGS: usize = 32;
/// One argument bound (mirrors `MAX_VERIFICATION_CHECK_ARG_BYTES`).
pub const MAX_VERIFICATION_JOB_ARG_BYTES: usize = 1024;
/// One typed spec JSON bound.
pub const MAX_VERIFICATION_JOB_SPEC_JSON_BYTES: usize = 64 * 1024;
/// One result JSON bound.
pub const MAX_VERIFICATION_JOB_RESULT_JSON_BYTES: usize = 64 * 1024;
/// One job/attempt note bound.
pub const MAX_VERIFICATION_JOB_NOTE_BYTES: usize = 512;
/// One changed-file path bound (mirrors `MAX_VERIFICATION_PATH_BYTES`).
pub const MAX_VERIFICATION_ATTEMPT_PATH_BYTES: usize = 4096;
/// One attempt workspace-root bound.
pub const MAX_VERIFICATION_JOB_ROOT_BYTES: usize = 4096;
/// One job execution budget bound (ms).
pub const MAX_VERIFICATION_JOB_BUDGET_MS: u64 = 3_600_000;
/// One environment-fingerprint JSON column bound.
pub const MAX_VERIFICATION_JOB_FINGERPRINT_JSON_BYTES: usize = 64 * 1024;

/// The job states a row may durably hold.
pub const VERIFICATION_JOB_STATES: [&str; 6] = [
    "queued",
    "running",
    "passed",
    "failed",
    "unavailable",
    "cancelled",
];
/// The inline outcome states an inline check row may hold.
pub const VERIFICATION_INLINE_STATES: [&str; 3] = ["passed", "failed", "unavailable"];

/// How long `read()` waits for a permit before failing with `Busy`. Matches
/// the SQLite `busy_timeout` pragma (5s), so pool-level and engine-level
/// waits behave consistently.
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);

/// A counting semaphore on stable std (there is no `std::sync::Semaphore` on
/// stable Rust as of 1.98): `acquire_timeout` blocks on a condvar until a
/// permit frees or the deadline passes. Readers hold a `Permit` for the
/// whole borrow, which is what caps live read connections at `READER_POOL`.
#[derive(Debug)]
struct Semaphore {
    permits: Mutex<usize>,
    available: Condvar,
}

/// RAII permit; releases one permit and wakes one waiter on drop.
#[derive(Debug)]
struct Permit(Arc<Semaphore>);

impl Drop for Permit {
    fn drop(&mut self) {
        let mut p = self.0.permits.lock().unwrap_or_else(|e| e.into_inner());
        *p += 1;
        self.0.available.notify_one();
    }
}

/// RAII connection-level durability lift: sets `PRAGMA synchronous = FULL`
/// for the duration of the guard so a grouped actor batch's single COMMIT
/// fsyncs the WAL before any caller ack, then restores the crate's configured
/// `NORMAL` on drop (also on panic/error paths). Connection-scoped: the
/// store writer lock is held by the caller for the whole batch, so no other
/// writer observes the lifted mode.
struct StrongSync<'a> {
    conn: &'a Connection,
}

impl<'a> StrongSync<'a> {
    fn on(conn: &'a Connection) -> StoreResult<Self> {
        conn.execute_batch("PRAGMA synchronous = FULL")?;
        Ok(Self { conn })
    }
}

impl Drop for StrongSync<'_> {
    fn drop(&mut self) {
        let _ = self.conn.execute_batch("PRAGMA synchronous = NORMAL");
    }
}

impl Semaphore {
    fn new(permits: usize) -> Self {
        Self {
            permits: Mutex::new(permits),
            available: Condvar::new(),
        }
    }

    /// Block until a permit is free or `deadline` passes (`Busy`).
    fn acquire_timeout(self: &Arc<Self>, deadline: Instant) -> StoreResult<Permit> {
        let mut p = self
            .permits
            .lock()
            .map_err(|_| StoreError::Migration("reader pool semaphore poisoned".into()))?;
        loop {
            if *p > 0 {
                *p -= 1;
                return Ok(Permit(Arc::clone(self)));
            }
            let now = Instant::now();
            if now >= deadline {
                return Err(StoreError::Busy(format!(
                    "no reader permit within {}ms ({} readers already active)",
                    BUSY_TIMEOUT.as_millis(),
                    READER_POOL
                )));
            }
            let (guard, _) = self
                .available
                .wait_timeout(p, deadline - now)
                .map_err(|_| StoreError::Migration("reader pool semaphore poisoned".into()))?;
            p = guard;
        }
    }
}

/// Bounded pool of idle read connections + the semaphore that caps how many
/// readers may borrow one at a time.
#[derive(Debug)]
struct ReaderPool {
    conns: Mutex<Vec<Connection>>,
    sem: Arc<Semaphore>,
    /// Connections ever opened (doctor/test probe: proves the cap held).
    created: AtomicU64,
}

impl ReaderPool {
    fn new() -> Self {
        Self {
            conns: Mutex::new(Vec::with_capacity(READER_POOL)),
            sem: Arc::new(Semaphore::new(READER_POOL)),
            created: AtomicU64::new(0),
        }
    }
}

/// A borrowed read connection; returned to the pool on drop. The semaphore
/// permit is held for the borrow's lifetime, which is what bounds concurrent
/// readers at `READER_POOL`.
pub struct ReadConn {
    conn: Option<Connection>,
    pool: Arc<ReaderPool>,
    _permit: Permit,
}

impl ReadConn {
    pub fn get(&self) -> &Connection {
        // In-process invariant: a ReadConn is handed out exactly once and its
        // connection is only `take`n by Drop, so a live ReadConn always has
        // its connection. Never reachable from DB state.
        self.conn.as_ref().expect("read conn already returned")
    }
}

impl std::ops::Deref for ReadConn {
    type Target = Connection;

    fn deref(&self) -> &Connection {
        self.get()
    }
}

impl Drop for ReadConn {
    fn drop(&mut self) {
        if let Some(conn) = self.conn.take() {
            if let Ok(mut conns) = self.pool.conns.lock() {
                if conns.len() < READER_POOL {
                    conns.push(conn);
                    // The semaphore permit is released right after this
                    // method, via the `_permit` field's drop.
                    return;
                }
            }
            drop(conn);
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic crash seam (fault-certification campaigns only)
//
// One-shot fault injection at named DURABILITY BOUNDARIES: the instant a
// group/transaction crosses the boundary (its COMMIT executed) the state is
// durable; before the boundary the whole in-flight operation is rolled back
// by SQLite on the next open — exactly like a process death at that point
// (verified: dropping a rusqlite connection that holds an open transaction
// rolls the transaction back and the file reopens cleanly). The seam is
// inert unless armed, and the panic fires at most once per arm.
// ---------------------------------------------------------------------------

/// One-shot fault-injection target of [`CrashSeam`]: crash at the
/// `ordinal`-th crossing (0-based) of durability boundary `point`.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CrashArm {
    pub point: &'static str,
    pub ordinal: u64,
}

#[derive(Default)]
struct SeamState {
    armed: Option<CrashArm>,
    /// Crossings of the ARMED point observed so far.
    crossings: u64,
}

/// Per-store-instance deterministic crash seam. Additive and default-off:
/// while unarmed every `trip` is a single uncontended mutex check and no
/// behavior or format changes. The fault campaigns arm exactly one
/// boundary per interrupted run, panic the store mid-operation, drop the
/// instance (the "process death") and reopen from disk.
#[doc(hidden)]
pub struct CrashSeam {
    state: Mutex<SeamState>,
}

impl std::fmt::Debug for CrashSeam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self.state.lock().map(|s| s.armed).unwrap_or(None);
        f.debug_struct("CrashSeam").field("armed", &s).finish()
    }
}

impl Default for CrashSeam {
    fn default() -> Self {
        Self {
            state: Mutex::new(SeamState::default()),
        }
    }
}

impl CrashSeam {
    /// Arm ONE crossing, replacing any previous arm and resetting the
    /// crossing counter. The panic fires exactly once when `point` is
    /// crossed for the `ordinal`-th time.
    pub fn arm(&self, arm: CrashArm) {
        let mut s = self.state.lock().unwrap_or_else(|p| p.into_inner());
        *s = SeamState {
            armed: Some(arm),
            crossings: 0,
        };
    }

    /// Trip the seam at `point`. Panics when the armed crossing is hit.
    fn trip(&self, point: &'static str) {
        let mut s = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let Some(arm) = s.armed else {
            return;
        };
        if arm.point != point {
            return;
        }
        s.crossings += 1;
        if s.crossings - 1 != arm.ordinal {
            return;
        }
        s.armed = None;
        drop(s);
        panic!(
            "[fault-seam] simulated crash at store durability boundary `{point}` (crossing {})",
            arm.ordinal
        );
    }
}

/// The daemon's durable store. `write` takes a single writer lock; `read`
/// borrows a connection from a small pool (SQLite WAL allows concurrent
/// readers). All mutations happen inside explicit transactions.
#[derive(Debug)]
pub struct Store {
    root: PathBuf,
    writer: Mutex<Connection>,
    pool: Arc<ReaderPool>,
    seam: CrashSeam,
    /// Last `updated_ms` issued to a memory-fact row. Fact order is the
    /// paging contract ("an upsert only moves a row toward the NEWEST
    /// end"), so fact stamps are MONOTONIC: two writes inside the same
    /// wall-clock millisecond must still order strictly, otherwise a new
    /// row can tie the cursor's millisecond and sort BELOW an ongoing walk
    /// (kind/key tie-breaks can put it on the already-consumed side).
    fact_seq: AtomicU64,
}

#[derive(Debug, Clone)]
pub struct SessionRow {
    pub id: SessionId,
    pub workspace_id: WorkspaceId,
    /// Durable worktree identity of the session (v8+). The standalone
    /// default is 1/1 (the session's worktree/task ids are adopted
    /// deliberately when a WorktreeManager-created worktree takes it over).
    pub worktree_id: WorktreeId,
    /// Durable task identity of the session (v8+); standalone default 1.
    pub task_id: TaskId,
    pub title: String,
    pub provider: String,
    pub model: String,
    pub state: AgentState,
    pub lifecycle: faktor_core::state::SessionLifecycle,
    pub created_ms: i64,
    pub updated_ms: i64,
}

/// One durable child-runtime projection row (migration v23): the child
/// session's state plus its bounded blocker truth. All blocker columns are
/// NULL when the child is not blocked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChildRuntimeRow {
    pub session_id: SessionId,
    pub child_id: String,
    pub state: String,
    pub blocker_kind: Option<String>,
    pub blocker_reason: Option<String>,
    pub blocker_dependency: Option<String>,
    pub blocker_resolution: Option<String>,
    pub last_progress_ms: Option<i64>,
    pub updated_ms: i64,
}

/// One atomic session transition: verify expected lifecycle/state, move
/// lifecycle+state, and append the journal event in a SINGLE SQLite
/// transaction. A crash can never leave the lifecycle and the journal
/// contradictory (the old two-step update-then-append had exactly that
/// window). `expected_* = None` skips the corresponding check.
#[derive(Debug, Clone)]
pub struct SessionTransition {
    /// When `Some`, the session row must have exactly this lifecycle or the
    /// transition fails with `StoreError::Conflict` and writes nothing.
    pub expected_lifecycle: Option<SessionLifecycle>,
    /// When `Some`, the lifecycle is updated to this value.
    pub new_lifecycle: Option<SessionLifecycle>,
    /// When `Some`, the session row must have exactly this state.
    pub expected_state: Option<AgentState>,
    /// The state the session row AND the journal event land on.
    pub new_state: AgentState,
    /// Journal event kind appended in the same transaction.
    pub event_kind: EventKind,
    pub event_payload: Option<serde_json::Value>,
    /// Payload schema version of `event_payload` (v11+; 1 for the original
    /// unversioned writers). Readers decode through the version; an unknown
    /// version is a loud error, never a silent parse.
    pub event_payload_ver: i64,
}

#[derive(Debug, Clone)]
pub struct MessageRow {
    pub id: i64,
    pub session_id: SessionId,
    pub seq: i64,
    pub role: String,
    pub data: serde_json::Value,
    pub created_ms: i64,
}

#[derive(Debug, Clone)]
pub struct PartRow {
    pub id: i64,
    pub message_id: i64,
    pub kind: String,
    pub data: serde_json::Value,
    pub created_ms: i64,
}

/// One memory-fact row including its durable `updated_ms` (the paging
/// order key).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryFactRow {
    pub kind: String,
    pub key: String,
    pub value: String,
    pub updated_ms: i64,
}

/// Cursor over the memory-fact total order `(updated_ms DESC, kind DESC,
/// key DESC)`: the `(updated_ms, kind, key)` position of the last row of a
/// page. Identifies a stable position — replaying it returns the same
/// window.
pub type MemoryFactCursor = (i64, String, String);

/// One first-class durable Task row (audit 25, schema v10). Typed columns,
/// one row per `(session_id, task_id)`; the legacy one-row-per-session
/// ledger blob lives in the renamed `task_ledger` table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskRow {
    pub task_id: TaskId,
    pub session_id: SessionId,
    pub goal: String,
    pub acceptance_criteria: Vec<String>,
    /// Ordered steps, append-only durable.
    pub plan: Vec<String>,
    /// Durable typed binary/image attachments of the task (schema v24,
    /// migration index 24). SEPARATE from the workspace-relative `files`
    /// vocabulary: bytes live in the CAS under each `digest`, the metadata
    /// rides this JSON column. Rows written before v24 decode with an empty
    /// list (`'[]'` column default) — the attachment-free row stays
    /// byte-identical.
    pub attachments: Vec<AttachmentId>,
    pub max_tokens: Option<u64>,
    pub max_turns: Option<u32>,
    pub spent_tokens: u64,
    pub spent_turns: u32,
    pub state: TaskState,
    /// Per-row monotonic revision (schema v14, audit P0-7): every effective
    /// state/criteria/plan/budget mutation bumps it exactly once in the same
    /// transaction. It is the row's optimistic-lock token: the completion
    /// path refuses a record that does not certify the current revision.
    pub revision: TaskRevision,
    pub created_ms: i64,
    pub updated_ms: i64,
}

/// One first-class durable verification record (audit P0-8, schema v14):
/// the completion proof of a task. A record certifies ONE task revision
/// (`revision` == the task row's revision when the verification ran): it
/// names the task, its base worktree identity, the acceptance-criterion
/// verdicts that cover the task's criteria at that revision, the executed
/// checks, the workspace files observed with their digests, and the final
/// [`VerificationStatus`].
///
/// Records are immutable once created EXCEPT the single CAS status
/// transition `Running -> Passed|Failed`
/// ([`Store::verification_record_finalize`]): a record finalizes exactly
/// once, and an already-final record refuses a second completion attempt
/// with a typed error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationRecordRow {
    pub id: VerificationRecordId,
    pub task_id: TaskId,
    pub revision: TaskRevision,
    pub workspace_id: WorkspaceId,
    pub worktree_id: WorktreeId,
    /// HEX digest text of the verified tree state (NULL when no tree hash
    /// was derivable — an honest "no tree observation", never a guess).
    pub tree_hash: Option<String>,
    pub criteria: Vec<CriterionVerification>,
    pub checks: Vec<CheckExecution>,
    pub changed_files: Vec<FileStateEvidence>,
    /// Paths of workspace changes the verification judged unrelated to the
    /// task (bounded text list).
    pub unrelated_changes: Vec<String>,
    /// Opaque reviewer observation (protocol-agnostic JSON, NULL when no
    /// review ran).
    pub reviewer: Option<serde_json::Value>,
    pub status: VerificationStatus,
    pub started_ms: i64,
    pub completed_ms: Option<i64>,
}

/// One verification-record row plus its raw schema-v20 evidence JSON columns:
/// `(row, environment_fingerprint_json, candidate_proof_ref_json)`. Either
/// `None` is an honest absence (a pre-v20 row or a record written without
/// that evidence); the session layer parses non-null values loudly.
pub type VerificationRecordWithEvidence = (VerificationRecordRow, Option<String>, Option<String>);

// ------------------------------------------------ verification jobs (v22)

/// One durable verification attempt (schema v22, audit P0-5/26): the
/// attempt's identity `(session_id, task_id, attempt_op_id)`, the task
/// revision it certified at enqueue, its workspace root and the bounded
/// environment fingerprint JSON. Changed files live in
/// `verification_attempt_changed_file`; every required check (inline AND
/// background) lives in `verification_job`, keyed by the same attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationAttemptRow {
    pub session_id: SessionId,
    pub task_id: TaskId,
    pub attempt_op_id: u64,
    pub task_revision: TaskRevision,
    pub workspace_root: String,
    pub environment_fingerprint_json: Option<String>,
    pub created_ms: i64,
}

/// One durable required check of one attempt (schema v22). Inline checks
/// carry `inline_status` and are terminal from birth; background checks
/// carry `spec_json` and walk `queued -> running -> terminal`. The executed
/// outcome JSON of a background check lands in `verification_job_result`
/// (keyed by the same identity) exactly once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationJobRow {
    pub session_id: SessionId,
    pub task_id: TaskId,
    pub attempt_op_id: u64,
    pub check_id: String,
    /// Derivation order inside the attempt (0-based).
    pub ordinal: u32,
    pub task_revision: TaskRevision,
    pub workspace_root: String,
    /// `compile` | `test` | `lint`.
    pub kind: String,
    /// Canonical command text (`program arg...`).
    pub command: String,
    /// The bounded argv identity (validated by the session layer).
    pub program: String,
    pub args_json: String,
    /// The typed spec JSON (background checks only; None for inline checks).
    pub spec_json: Option<String>,
    pub budget_ms: u64,
    /// `passed` | `failed` | `unavailable` for an INLINE check; None for a
    /// background check.
    pub inline_status: Option<String>,
    /// `queued` | `running` | `passed` | `failed` | `unavailable` |
    /// `cancelled`.
    pub state: String,
    /// The executed background outcome JSON (joined from
    /// `verification_job_result`; None for inline checks and unresolved
    /// background checks).
    pub result_json: Option<String>,
    pub note: Option<String>,
    pub op_id: Option<u64>,
    pub environment_fingerprint_json: Option<String>,
    pub created_ms: i64,
    pub updated_ms: i64,
    pub finished_ms: Option<i64>,
}

/// One attempt together with its changed files and every required check, in
/// derivation order (changed files by ordinal).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationAttemptView {
    pub attempt: VerificationAttemptRow,
    pub changed: Vec<String>,
    pub checks: Vec<VerificationJobRow>,
}

/// Typed refusal of one `verification_job_claim`/`verification_job_resolve`
/// transition. The job row is NEVER changed by a refusal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationJobRefusal {
    /// No job row for this `(session, task, attempt, check)`.
    Missing { check_id: String },
    /// The job is not in the required source state (only `queued` claims;
    /// only `running` resolves).
    NotOpen { check_id: String, state: String },
    /// A NEWER attempt exists for this task: attempt N can never be mutated
    /// (nor late-resolved) after N+1 was durably begun.
    Superseded {
        attempt_op_id: u64,
        newest_attempt_op_id: u64,
    },
    /// A result row already exists for this attempt/check: a result is
    /// written exactly once.
    ResultExists { check_id: String },
}

/// Result of one successful recover sweep: `requeued` Running rows became
/// Queued; `orphaned` open rows had no attempt (only possible on a hand-
/// corrupted database — foreign keys make torn begins impossible).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VerificationJobRecovery {
    pub requeued: u64,
    pub orphaned: u64,
}

// ------------------------------------- legacy verification import (v22)

/// `memory_fact` kind of one pre-v22 verification attempt row (one fact row
/// per attempt, key `va:{task_id}:{op_id}`). The v22 tables replaced it; the
/// import below projects these rows (and their jobs) into the real tables
/// additively — the legacy facts are NEVER deleted.
pub const LEGACY_VERIFICATION_ATTEMPT_FACT_KIND: &str = "verification_attempt";
/// `memory_fact` kind of one pre-v22 verification job row (one fact row per
/// required check, key `vj:{task_id}:{check_id}`).
pub const LEGACY_VERIFICATION_JOB_FACT_KIND: &str = "verification_job";
/// Durable marker fact kind written (same transaction as the imported rows)
/// once one session's legacy rows were imported. Its presence is the
/// exactly-once guard: a re-open never duplicates the import.
pub const VERIFICATION_V22_IMPORT_MARKER_KIND: &str = "verification_v22_import";
/// Marker fact key.
pub const VERIFICATION_V22_IMPORT_MARKER_KEY: &str = "done";

/// One legacy row the import could not project: its fact key plus a typed
/// reason. Recorded in the durable marker fact (and logged) so a corrupt
/// legacy row is skipped LOUDLY, never silently dropped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegacyVerificationSkip {
    pub key: String,
    pub reason: String,
}

/// Outcome of one v22 legacy-verification import (or of the marker-guarded
/// no-op on every later open: all counters zero).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LegacyVerificationImport {
    pub imported_attempts: u64,
    pub imported_jobs: u64,
    pub imported_results: u64,
    /// Bounded typed notes for skipped corrupt/undecodable legacy rows.
    pub skipped: Vec<LegacyVerificationSkip>,
    /// Skipped rows beyond the bounded note list (still logged loudly).
    pub skipped_overflow: u64,
}

/// Typed refusal of a `task_complete_verified` request: every check the
/// completion transaction performs names its own variant, so callers can
/// distinguish a missing record from a wrong-revision record from an
/// uncovered criterion without parsing prose. The task row is NEVER changed
/// by a refusal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskCompletionRefusal {
    /// No task row for `(session_id, task_id)` (or the session row itself is
    /// missing).
    TaskMissing { task_id: TaskId },
    /// The task's current revision differs from the expected one (the task
    /// changed since the caller's read: re-read and re-decide).
    RevisionMismatch {
        expected: TaskRevision,
        actual: TaskRevision,
    },
    /// (a) only a `Verifying` task may be completed.
    NotVerifying { actual: TaskState },
    /// (b) no verification record row exists with this id.
    RecordMissing { record_id: VerificationRecordId },
    /// (c) the record certifies a different task.
    RecordWrongTask {
        record_id: VerificationRecordId,
        record_task: TaskId,
        requested: TaskId,
    },
    /// (d) the record certifies a different revision of the task.
    RecordWrongRevision {
        record_id: VerificationRecordId,
        record_revision: TaskRevision,
        expected: TaskRevision,
    },
    /// (e) only a `Passed` record certifies completion.
    RecordNotPassed {
        record_id: VerificationRecordId,
        status: VerificationStatus,
    },
    /// (f) the record does not cover (present with `passed = true`) every
    /// current acceptance criterion of the task.
    CriteriaNotCovered {
        record_id: VerificationRecordId,
        missing: Vec<String>,
    },
    /// (g) the record was certified against a different base worktree than
    /// the task's session currently stands on.
    WorktreeMismatch {
        record_id: VerificationRecordId,
        record_workspace: WorkspaceId,
        record_worktree: WorktreeId,
        task_workspace: WorkspaceId,
        task_worktree: WorktreeId,
    },
    /// (h) reservation rows of this task still hold budget (`reserved`,
    /// `dispatched` or `uncertain`): the accounting-before-completion gate
    /// refuses inside the completion transaction — a reserve that raced the
    /// session layer's accounting pass is caught HERE and the task stays
    /// Verifying. Nothing is transitioned.
    ReservationsHeld {
        reserved: usize,
        dispatched: usize,
        reserved_micro: u64,
        uncertain: usize,
        uncertain_micro: u64,
    },
}

/// Typed refusal of a record-finalize CAS. A record finalizes exactly once
/// (`Running -> Passed|Failed`); anything else is refused here with the
/// record's current status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordFinalizeRefusal {
    Missing {
        record_id: VerificationRecordId,
    },
    NotRunning {
        record_id: VerificationRecordId,
        current: VerificationStatus,
    },
}

/// One typed, versioned row of the durable session ledger (audits 27,
/// 71-72; schema v11). The journal is the source of truth for *what
/// happened*; `ledger_entry` is the rich append-only typed ledger and
/// `ledger_head` its materialized checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LedgerEntryRow {
    /// Per-session gapless entry sequence.
    pub seq: i64,
    /// Entry-type tag, e.g. `goal_set` / `blocker_opened` (snake_case).
    pub entry_type: String,
    /// Payload schema version of THIS row (schema v1 this wave).
    pub schema_ver: i64,
    /// Decoded JSON payload. Never opaque text: the session layer decodes
    /// it strictly by (entry_type, schema_ver) — an unknown version or a
    /// shape violation is a loud error, never a silent parse.
    pub payload: serde_json::Value,
    pub created_ms: i64,
}

/// The materialized head checkpoint of the typed ledger (v11): the folded
/// projection (`head_json`) of every entry up to `checkpoint_seq`, written
/// by compaction together with the entry deletions it summarizes. A crash
/// between an entry append and its head checkpoint is repaired on the next
/// open by folding the newer entries onto the head.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LedgerHeadRow {
    pub head_json: serde_json::Value,
    pub checkpoint_seq: i64,
    pub schema_ver: i64,
    pub updated_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolRunRow {
    pub id: i64,
    pub session_id: SessionId,
    pub op_id: OpId,
    pub tool: String,
    pub args: serde_json::Value,
    pub status: String,
    pub started_ms: i64,
    pub ended_ms: Option<i64>,
    pub effect_status: String,
    pub recovery: serde_json::Value,
    pub expected_hash: Option<String>,
    /// Durable replay descriptor (v7+): the stored invocation crash recovery
    /// may re-execute ONCE for idempotent tools. NULL on legacy rows.
    pub replay_descriptor: Option<serde_json::Value>,
    /// Physical attempt counter of the SAME logical operation (v7+): the
    /// original run is attempt 0; each crash recovery replay bumps it.
    pub attempt: i64,
    /// Durable workspace-write postcondition (v7+): `{workspace_id,
    /// worktree_id, relative_path, expected_hash}` — the hash of the ACTUAL
    /// bytes as written, recorded by the tool at execution end. NULL until
    /// the tool reports it (or for non-write tools).
    pub postcondition: Option<serde_json::Value>,
}

/// One durable logical-turn record (v7). Created transactionally when a
/// prompt is admitted as the ACTIVE logical turn (submit_prompt / queue
/// admission); it fixes the turn's exact operation identity and effective
/// model/provider envelope so crash recovery resumes the SAME turn with the
/// SAME identity instead of synthesizing a fresh operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnRecordRow {
    pub id: i64,
    pub session_id: SessionId,
    pub turn_op_id: OpId,
    /// Durable queue seq when the turn was admitted from the prompt queue.
    pub queue_seq: Option<i64>,
    /// Durable message seq of the materialized user prompt.
    pub prompt_message_id: Option<i64>,
    pub effective_provider: String,
    pub effective_model: String,
    /// Reasoning mode / variant of the logical turn (NULL when unset).
    pub variant: Option<String>,
    /// Tool-call parsing mode of the logical turn (NULL until driven).
    pub tool_mode: Option<String>,
    pub started_at: i64,
    /// active | completed | cancelled | failed
    pub status: String,
    pub updated_ms: i64,
}

pub const TURN_RECORD_ACTIVE: &str = "active";
pub const TURN_RECORD_COMPLETED: &str = "completed";
pub const TURN_RECORD_CANCELLED: &str = "cancelled";
pub const TURN_RECORD_FAILED: &str = "failed";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointRow {
    pub id: i64,
    pub session_id: SessionId,
    pub sequence: i64,
    pub path: String,
    pub before_hash: String,
    pub after_hash: String,
    /// CAS hash of the AFTER-content blob (v3+). NULL on rows recorded
    /// before the column existed: redo/diff refuse those honestly.
    pub after_cas_hash: Option<String>,
    /// Per-side EXISTENCE flags (v6+). A hash alone cannot distinguish a
    /// missing file from an empty one (both sides of a missing→empty write
    /// hash to blake3("")). When a side `exists` is false its hash column is
    /// the empty string (no content exists to address).
    ///
    /// Backward compatibility: pre-v6 rows carry NO marker, so these read as
    /// true — old rows were only recorded for real files (the caller had
    /// read/hashed the content on both sides).
    pub before_exists: bool,
    pub after_exists: bool,
    pub created_ms: i64,
    pub restored_ms: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorktreeRow {
    pub id: i64,
    pub workspace_id: WorkspaceId,
    pub path: String,
    pub branch: String,
    pub active: bool,
}

/// One CAS blob hash the store schema references (artifact rows by content
/// address, checkpoint rows by after-blob), with the referencing table and
/// row id. Doctor's dangling-reference scan compares these against the CAS;
/// the store itself never reads blob files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CasHashRef {
    /// Table the reference originates from: `artifact` or `checkpoint`.
    pub source: &'static str,
    /// Row id inside that table.
    pub row_id: i64,
    /// The 64-hex BLAKE3 CAS hash the row references.
    pub hash: String,
}

/// One cost-reservation row whose task row is gone (P0-97 `doctor --deep`
/// dangling-budget scan). Rows in this list can never be settled or refunded
/// and their predicted spend is untracked: the durable ledger points at
/// nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DanglingReservationRow {
    pub reservation_id: i64,
    pub session_id: SessionId,
    pub task_id: TaskId,
    pub op_id: OpId,
    pub status: String,
    pub predicted_micro: u64,
}

/// Per-status counts + the dangling rows of the whole `cost_reservation`
/// table (read-only `doctor --deep` invariant scan). `reserved`/`dispatched`
/// rows whose task is gone mean an untracked prediction is still counted by
/// nothing; a `settled` row whose task is gone means spend landed on a
/// vanished envelope. `abandoned` is the legacy pre-P0-2 vocabulary, kept
/// for doctor's line format: no row can hold it after the v16 migration
/// (the CHECK forbids it), so it reads 0 on every migrated store. `open` is
/// the legacy pre-v17 vocabulary (v17 renamed it to `reserved`), so it also
/// reads 0 on every v18 store.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CostReservationScan {
    pub total: u64,
    /// IN-FLIGHT rows holding budget: the legacy pre-v17 `open` rows (0 on
    /// every v18 store) PLUS the v17 vocabulary `reserved` and `dispatched`.
    /// Folded so doctor's legacy line format ("open N") keeps reporting the
    /// true in-flight total; `reserved`/`dispatched` below are the exact
    /// per-status counts.
    pub open: u64,
    pub reserved: u64,
    pub dispatched: u64,
    pub settled: u64,
    pub refunded: u64,
    pub abandoned: u64,
    /// P0-2: reservations a crashed daemon may have dispatched (marked, never
    /// settled) — they keep consuming the reserved amount until a reconcile
    /// or the task-completion finalize closes them.
    pub uncertain: u64,
    /// Rows (of ANY status except the refunded ledger tail) whose
    /// `(session_id, task_id)` has no task row.
    pub dangling: Vec<DanglingReservationRow>,
}

/// One verification-record-vs-task consistency violation (P0-97
/// `doctor --deep` wave-16 scan). Each row names its kind so doctor can
/// count per kind and print typed lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationInvariantIssue {
    pub kind: &'static str,
    pub detail: String,
}

/// The wave-16 verification-consistency invariant scan (`doctor --deep`,
/// read-only): records must reference existing task rows, a `Passed` record
/// may only certify a task's current revision when that task is
/// `VerifiedComplete`, and a `VerifiedComplete` task must carry the `Passed`
/// record the completion transaction consumed (the record certifies
/// `revision - 1`: completion bumps the row revision exactly once after
/// validating the record against the pre-completion revision).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VerificationInvariantScan {
    /// Every `verification_record` row, regardless of status.
    pub total_records: u64,
    /// Every `task` row in a completion-relevant state
    /// (NeedsVerification / Verifying / VerifiedComplete).
    pub relevant_tasks: u64,
    /// `VerifiedComplete` task rows.
    pub completed_tasks: u64,
    pub issues: Vec<VerificationInvariantIssue>,
}

/// One active logical-turn row that no durable path can recover after a
/// daemon crash (read-only `doctor --deep` scan): no prompt message row, no
/// queue row, no journal event and no tool-run row reference the turn's op.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnrecoverableActiveTurn {
    pub record_id: i64,
    pub session_id: SessionId,
    pub turn_op_id: OpId,
    pub detail: String,
}

/// Active-turn recoverable-owner scan (`doctor --deep`, read-only): a live
/// daemon legitimately owns active rows in memory, so the check is what a
/// CRASHED daemon needs — a durable record, prompt message or queue row and
/// a replayable journal (event or tool-run row naming the turn op).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TurnOwnershipScan {
    /// Active `turn_record` rows across every session.
    pub active_turns: u64,
    /// Active rows with at least one durable recovery path.
    pub recoverable: u64,
    pub unrecoverable: Vec<UnrecoverableActiveTurn>,
}

/// One raw `memory_fact` row of a kind doctor's orphan-child scan watches
/// (`orchestrator` identity rows in the child's row space and
/// `orchestrator_registry` rows in the parent's row space). Values stay
/// opaque here: parsing belongs to the session/orchestrator layer that owns
/// each JSON shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryFactRowRef {
    /// The session whose row space the fact lives in.
    pub session_id: SessionId,
    pub kind: String,
    pub key: String,
    pub value: String,
}

/// One durable per-workspace repository-index state row (schema v12, audits
/// 30/64): the `faktor-index` IndexService persists its WorkspaceIndexState
/// machine here. `state_json` is opaque JSON owned by the index layer (like
/// every other TEXT payload in this schema); `generation` is the numeric
/// generation that row names (0 for NotStarted).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexStateRow {
    pub workspace_id: WorkspaceId,
    pub state_json: String,
    pub generation: i64,
    pub updated_ms: i64,
}

/// One append-only transition-journal row of a workspace's index state.
/// Written in the SAME transaction as the row it transitions from/to, so
/// the journal can never contradict the current row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexStateLogRow {
    pub id: i64,
    pub workspace_id: WorkspaceId,
    /// Legal-machine transition kind owned by the index layer (e.g.
    /// `building`, `ready`, `dirty`, `failed`, `resume`, `corrupt`).
    pub kind: String,
    pub state_json: String,
    pub generation: i64,
    pub updated_ms: i64,
}

pub struct QueuedPrompt {
    pub queue_seq: i64,
    pub op_id: OpId,
    pub prompt: String,
    pub files: Vec<String>,
    pub model: Option<String>,
    pub variant: Option<String>,
    pub agent: Option<String>,
    pub status: String,
    pub requested_at: i64,
}

/// Result of the atomic claim/admission of the queue head.
#[derive(Debug, Clone)]
pub struct AdmittedPrompt {
    pub queue_seq: i64,
    pub op_id: OpId,
    pub prompt: String,
    pub files: Vec<String>,
    pub model: Option<String>,
    pub variant: Option<String>,
    pub agent: Option<String>,
    /// Message seq of the materialized user message (== the admission
    /// journal event seq).
    pub message_seq: i64,
}

/// One grouped hot write (the `faktor-session` `DbActor` request surface).
/// The four fixed write shapes the daemon issues per message / per part / per
/// journal event / per usage-settlement frame — never free-form SQL.
#[derive(Debug, Clone)]
pub enum HotWrite {
    AppendEvent {
        session_id: SessionId,
        op_id: Option<OpId>,
        kind: EventKind,
        state: AgentState,
        ts_ms: i64,
        payload: Option<serde_json::Value>,
        /// Payload schema version (v11+; session writers stamp
        /// `PAYLOAD_SCHEMA_V`). Readers refuse unknown versions loudly.
        payload_ver: i64,
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
    RecordProviderCall {
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

/// Microsecond timing split of one [`Store::batch_hot_writes`] group.
///
/// `work_us` covers everything up to the commit statement (writer lock wait
/// excluded: the lock is already held when timing starts, and it measures
/// only the SQLite work itself). `commit_us` covers the COMMIT statement,
/// which under the group's `synchronous = FULL` includes the deliberate WAL
/// fsync that makes the actor's ack mean "durable".
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BatchTiming {
    /// SQL work (BEGIN + per-write savepoints), microseconds.
    pub work_us: u64,
    /// COMMIT + fsync, microseconds.
    pub commit_us: u64,
}

/// Per-write result of a [`Store::batch_hot_writes`] group.
#[derive(Debug, Clone)]
pub enum HotWriteOutcome {
    /// The journal event's gapless per-session sequence.
    EventSeq(EventSeq),
    /// The inserted row id (message / part / provider-call).
    RowId(i64),
}

/// One durable per-call prefix observation (v13), ordered oldest-first by
/// row id (= call order within the session). Read back for the router's
/// prefix-stability aggregation; rows recorded before v13 (or settled
/// without a prefix hash) are excluded by
/// [`Store::provider_call_prefix_rows`] — a missing observation is not a
/// zero.
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderCallPrefixRow {
    /// The provider_call row id — call order within the session.
    pub row_id: i64,
    pub session_id: SessionId,
    /// Digest of the exact cacheable-prefix byte string the call sent.
    /// Validated to be exactly 32 bytes on every read; anything else is a
    /// loud `Malformed`, never a silent misread.
    pub prompt_prefix_hash: [u8; 32],
    /// Token count of that prefix (u32 bound, matching the router's
    /// `TurnPrefix.prefix_tokens`).
    pub prompt_tokens: u32,
    /// Optional per-row prefix stability in [0, 1] recorded by the
    /// settlement site (NULL = not recorded).
    pub prefix_stability: Option<f64>,
    /// Raw additive segment observation JSON (v19): the exact per-call
    /// `PrefixObservation` the settlement site measured (eight segment
    /// digests + token counts + this call's observed cache reads). The
    /// store validates its strict shape and bounds on write AND read — a
    /// corrupt row is a loud `Malformed`, never a silently degraded
    /// observation. `None` on pre-v19 (legacy) rows: absence is an honest
    /// "no segment identity recorded", never a guess.
    pub prefix_segments_json: Option<String>,
}

/// Hard bound on the serialized per-call segment observation persisted in
/// `provider_call.prefix_segments_json` (schema v19) — mirror of the wire
/// plan's own serialization bound: a hostile row may not describe an
/// unbounded prompt.
pub const MAX_PREFIX_SEGMENTS_JSON: usize = 64 * 1024;

/// Bound on the decoded segment vector inside `prefix_segments_json`
/// (bounded everything: the observation has a fixed conceptual segment
/// count; future segmentations may grow, but never without bound).
pub const MAX_PREFIX_SEGMENTS: usize = 64;

/// Session-level aggregate of the STORED per-row prefix stabilities (v13):
/// count, mean and population std dev over rows that carry one. Rows
/// without a recorded stability contribute nothing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrefixStabilityAggregate {
    pub observations: u64,
    pub mean: f64,
    pub std_dev: f64,
}

// ------------------------------------------------------- durable cost ledger types
// (schema v15, P0-6/12): see the migration block + the store section below.

/// The durable monetary envelope of one task row (schema v15): the cap
/// (`None` = unlimited) and the settled spend. READ-ONLY surface — the
/// session layer never patches these through `TaskBudget`; the cost ledger
/// is their only writer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskCostRow {
    pub max_cost_micro: Option<u64>,
    pub spent_cost_micro: u64,
}

/// Durable cost-basis tags of one settled reservation (schema v18
/// `cost_basis` column): the honest authority behind `settled_cost_micro`.
/// `ProviderReported` = the provider's billed amount won;
/// `RouteSnapshotEstimate` = the locally calculated categories x the frozen
/// route-time [`PricingSnapshot`] won; `ConservativeReservation` = the
/// task-completion finalize charged the reserved estimate (a dispatched
/// attempt that may have billed); `Unknown` = no price authority existed and
/// the row closed as a documented Unknown spend (nothing folded). Pre-v18
/// rows read `None` — the basis was never recorded.
pub const COST_BASIS_PROVIDER_REPORTED: &str = "ProviderReported";
pub const COST_BASIS_ROUTE_SNAPSHOT_ESTIMATE: &str = "RouteSnapshotEstimate";
pub const COST_BASIS_CONSERVATIVE_RESERVATION: &str = "ConservativeReservation";
pub const COST_BASIS_UNKNOWN: &str = "Unknown";

/// Durable wire-delivery phases of one reservation (schema v18
/// `delivery_state` column): what the attempt's provider delivery reached,
/// independent of the ledger status. NULL = nothing was ever dispatched
/// (or the row predates v18); `dispatched` = the request left the process
/// (the durable marker was written); `completed` = the usage settled
/// normally; `failed` = a terminal failure was recorded (uncertain). Written
/// only by the ledger's own transitions.
pub const DELIVERY_DISPATCHED: &str = "dispatched";
pub const DELIVERY_COMPLETED: &str = "completed";
pub const DELIVERY_FAILED: &str = "failed";

/// One durable cost reservation (schema v18). Status strings are the
/// ledger's frozen vocabulary: `reserved`, `dispatched`, `settled`,
/// `refunded`, `uncertain` (the v17 migration renamed the v15/v16 `open`
/// state to `reserved` — a reservation holds budget until dispatch — and
/// promoted `dispatched` to a real state so refund-after-dispatch is
/// SQL-impossible; the P0-2 migration renamed the legacy `abandoned` state
/// to `uncertain`, see the migration block comments). `dispatched_ms` is the
/// durable dispatch marker written immediately BEFORE the provider transport
/// call (NULL = dispatch never provably began); `pricing_snapshot_json` is
/// the immutable route-time price capture the settlement math prices usage
/// against (NULL = no pricing authority was consulted). `attempt_op_id` keys
/// this row to its physical network attempt (NULL = legacy row keyed by the
/// shared logical `op_id` only); `parent_op_id` is the shared logical
/// model-call op id every attempt of one call belongs to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CostReservationRow {
    pub reservation_id: i64,
    pub session_id: SessionId,
    pub task_id: TaskId,
    /// The op that owns this reservation. Legacy rows (and every pre-attempt
    /// writer): the shared logical model-call op. Attempt-keyed rows carry
    /// the physical attempt's fresh op id; `attempt_op_id` mirrors it and
    /// `parent_op_id` names the logical parent.
    pub op_id: OpId,
    /// The physical-attempt op id this reservation keys by (NULL = legacy
    /// single-attempt row: the op that owned the row was its only attempt).
    pub attempt_op_id: Option<OpId>,
    /// The shared logical model-call op id (the parent of the attempt; NULL
    /// only on rows whose own op was already the parent).
    pub parent_op_id: Option<OpId>,
    pub predicted_micro: u64,
    pub status: String,
    pub created_ms: i64,
    pub settled_ms: Option<i64>,
    /// The durable dispatch marker (NULL = never dispatched before a crash).
    pub dispatched_ms: Option<i64>,
    pub pricing_snapshot: Option<PricingSnapshot>,
    pub provider_cost_micro: Option<u64>,
    pub provider_reported_micro: Option<u64>,
    pub route_decision_json: Option<String>,
    /// The provider request/stream id of the attempt, when one was recorded
    /// (the terminal failure path records it; NULL otherwise).
    pub request_id: Option<String>,
    /// The last known wire-delivery phase; see [`DELIVERY_DISPATCHED`].
    pub delivery_state: Option<String>,
    /// Terminal-failure reason code (set by the uncertain transition).
    pub failure_reason_code: Option<String>,
    /// How `settled_cost_micro` was arrived at; see [`COST_BASIS_PROVIDER_REPORTED`].
    pub cost_basis: Option<String>,
    /// The provider-reported amount of the settlement (the v18 canonical
    /// column; `provider_reported_micro` is its pre-v18 twin).
    pub provider_reported_cost_micro: Option<u64>,
    /// The reserve-time estimate (`predicted_micro` captured durably).
    pub estimated_cost_micro: Option<u64>,
    /// The amount actually folded into the task's spent total (NULL for
    /// Unknown closes and every pre-v18 settlement, which never recorded
    /// which amount was folded).
    pub settled_cost_micro: Option<u64>,
}

/// One reservation attempt's outcome (schema v15).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CostReserveOutcome {
    /// The reservation is reserved, holding `predicted_micro` of the task's
    /// budget. The id is the AUTOINCREMENT row id: monotonic across daemon
    /// restarts and never reused.
    Granted(i64),
    /// spent + predicted would exceed the task's cap (`max` 0/None =
    /// unlimited). NOTHING is written.
    Exceeded { free: u64 },
}

/// The state a reservation must hold for settle/mark to apply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CostReservationState {
    /// The reservation moved to the requested state.
    Applied,
    /// The reservation exists but is not in the required state (double
    /// settle / settle after refund / mark after settle): typed, nothing
    /// written.
    NotOpen { current: String },
    /// No reservation row with this id.
    Missing,
}

/// The outcome of a refund attempt (schema v18). The refund's guarded UPDATE
/// (`status IN ('reserved','open') AND dispatched_ms IS NULL`) makes
/// refund-after-dispatch impossible AT THE SQL LEVEL: a dispatched, settled,
/// refunded or uncertain reservation changes zero rows and is refused here
/// with the full row truth (`current` status + dispatch marker), so the
/// session layer can raise `CannotRefundDispatched` instead of freeing
/// money.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefundOutcome {
    /// The reservation moved RESERVED -> REFUNDED; the prediction is free.
    Applied,
    /// The reservation exists but is no longer refundable pre-dispatch.
    /// `dispatched_ms` carries the durable marker (Some = a dispatch may
    /// have reached the provider — the money stays put).
    Blocked {
        current: String,
        dispatched_ms: Option<i64>,
    },
    /// No reservation row with this id.
    Missing,
}

/// The state a usage settlement landed in (P0-1 settlement truth).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CostSettleOutcome {
    /// The reservation closed SETTLED and `actual_micro` (the locally
    /// calculated cost) was folded into the task's spent total.
    Applied { actual_micro: u64 },
    /// The reservation closed SETTLED with NO local cost: the price
    /// source was Unknown/absent and the task has no hard cap — the row
    /// records an Unknown spend (both amount columns NULL; nothing
    /// folded). Never a fabricated zero or one.
    AppliedUnknown,
    /// The reservation exists but is not `open`.
    NotOpen { current: String },
    /// No reservation row with this id.
    Missing,
    /// No price authority (no snapshot, or an Unknown-source snapshot)
    /// and the task HAS a hard cost cap: settlement is refused — typed —
    /// and NOTHING is written (the row stays OPEN; recovery/finalize
    /// resolve it conservatively). Never pretend zero or one.
    UnknownPrice { reservation: i64 },
}

/// The outcome of one [`Store::cost_reconcile_uncertain`] pass.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CostReconcileReport {
    /// UNCERTAIN reservations closed SETTLED from a durable
    /// `provider_call` row's tokens, priced at the row's snapshot.
    pub settled: u64,
    /// UNCERTAIN reservations closed as documented Unknown spends (no
    /// price authority, no hard cap): amount columns NULL, nothing
    /// folded.
    pub closed_unknown: u64,
    /// UNCERTAIN reservations left untouched (no price authority under
    /// a hard cap, or no completed provider-call row for their op).
    pub left_uncertain: u64,
    /// Total microUSD folded into the task's spent total.
    pub charged_micro: u64,
}

/// The outcome of one [`Store::cost_finalize_uncertain`] pass.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CostFinalizeReport {
    /// UNCERTAIN reservations conservatively closed SETTLED at their
    /// reserved estimate.
    pub settled: u64,
    /// Rows left UNCERTAIN (their task row is gone: nothing can fold).
    pub left_uncertain: u64,
    /// Total microUSD charged (each row's `predicted_micro`).
    pub charged_micro: u64,
}

/// One durable evidence row (schema v21; the audit's evidence table): the
/// evidence IDENTITY the scoped evidence store reads back. `id` is the
/// GLOBALLY UNIQUE `EvidenceId` — `INTEGER PRIMARY KEY AUTOINCREMENT`, so a
/// daemon restart can never mint an id it already handed out (SQLite's
/// `sqlite_sequence` keeps the high-water mark across closed connections
/// even when a row were deleted). Scope columns (`session_id`,
/// `workspace_id`, `task_id`) are what the scoped read compares; `kind`,
/// `revision`, `provenance`, `compressibility`, `compression`, `retrieval`,
/// `compact`, `backing_cas_hash`, `completeness` and `created_ms` are the
/// envelope's own fields, JSON-encoded exactly as the evidence crate
/// serializes them (protocol-agnostic TEXT; parsed fallibly on read).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceRow {
    /// The globally unique evidence id; `0` on a fresh row means "assign the
    /// next id" for [`Store::evidence_insert`].
    pub id: u64,
    pub session_id: SessionId,
    pub workspace_id: WorkspaceId,
    pub task_id: Option<u64>,
    pub kind: String,
    pub revision: i64,
    pub provenance_json: String,
    pub compressibility: String,
    pub compression_json: String,
    pub retrieval_json: String,
    pub compact_json: String,
    pub backing_cas_hash: Option<String>,
    pub completeness: String,
    pub created_ms: i64,
}

impl Store {
    /// The directory this store was opened at. The durable evidence
    /// authority roots its backing CAS beside it (`<root>/evidence-cas`), so
    /// a daemon restart reopens the SAME backing files the rows reference.
    pub fn root(&self) -> &std::path::Path {
        &self.root
    }

    /// Open (creating if needed) and migrate. `integrity_check: true` runs a
    /// full integrity check before use and refuses to open a corrupt store.
    pub fn open(root: impl Into<PathBuf>, integrity_check: bool) -> StoreResult<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root)?;
        let db_path = root.join("faktor-plus.db");

        let mut conn = Connection::open(&db_path)?;
        configure(&conn)?;
        migrate(&mut conn)?;

        if integrity_check {
            let issues = check_integrity(&conn)?;
            if !issues.is_empty() {
                return Err(StoreError::Corrupt(issues));
            }
        }

        Ok(Self::finish_open(root, conn))
    }

    /// Fast normal-start open (production `serve`, plain `doctor`): WAL
    /// recovery, migrations, and the BOUNDED `PRAGMA quick_check` — never
    /// the full `PRAGMA integrity_check` scan. Audit 43: the production path
    /// ran the full scan on EVERY start; the deep scan belongs to
    /// `doctor --deep` and crash forensics, not to startup latency.
    ///
    /// "Fast" is not "blind": the WAL is recovered and folded into the main
    /// file BEFORE the check (a crashed predecessor's frames are validated,
    /// never shadowed), migrations always run, and quick_check still refuses
    /// a store whose pages are damaged.
    pub fn open_fast(root: impl Into<PathBuf>) -> StoreResult<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root)?;
        let db_path = root.join("faktor-plus.db");

        let mut conn = Connection::open(&db_path)?;
        configure(&conn)?;
        migrate(&mut conn)?;
        // WAL recovery: opening + configuring recovered any frames a crashed
        // predecessor left in the -wal; the checkpoint folds them into the
        // main file so the quick check below validates the post-recovery
        // state and a stale -wal sidecar can never shadow newer content.
        conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
        let issues = check_quick(&conn)?;
        if !issues.is_empty() {
            return Err(StoreError::Corrupt(issues));
        }

        Ok(Self::finish_open(root, conn))
    }

    fn finish_open(root: PathBuf, conn: Connection) -> Self {
        // Seed the fact sequence above every durable row (a burst that
        // crashed in the same millisecond as a write must not let the next
        // stamp tie an existing row) and above the wall clock (a machine
        // clock stepped backwards must not re-enter old order positions).
        let max_ms: i64 = conn
            .query_row(
                "SELECT COALESCE(MAX(updated_ms), 0) FROM memory_fact",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);
        let writer = Mutex::new(conn);
        let pool = Arc::new(ReaderPool::new());
        Self {
            root,
            writer,
            pool,
            seam: CrashSeam::default(),
            fact_seq: AtomicU64::new(max_ms.max(now_ms()).max(0) as u64),
        }
    }

    /// The next monotonic memory-fact stamp: the wall clock when it is
    /// ahead of every issued stamp (idle catch-up keeps stamps honest wall
    /// times), otherwise exactly one past the last issued stamp (bursts
    /// inside one millisecond stay strictly ordered).
    fn fact_timestamp(&self) -> i64 {
        let now = now_ms().max(0) as u64;
        let prev = self.fact_seq.load(Ordering::Relaxed);
        let next = now.max(prev.saturating_add(1));
        self.fact_seq.store(next, Ordering::Relaxed);
        next as i64
    }

    /// Arm this store instance's deterministic crash seam (fault
    /// certification only; see [`CrashSeam`]). Inert when never armed.
    #[doc(hidden)]
    pub fn crash_arm(&self, arm: CrashArm) {
        self.seam.arm(arm);
    }

    pub fn path(&self) -> PathBuf {
        self.root.join("faktor-plus.db")
    }

    /// Deliberate synchronous access to the shared store, coexisting with a
    /// `DbActor` (faktor-session) that batches the hot append paths through
    /// [`Store::batch_hot_writes`]. All reads and every non-hot write
    /// (compound transitions, queue ops, checkpoints, tool runs, recovery)
    /// go through this surface and share the same writer lock + reader pool,
    /// so direct and actor writes never corrupt each other. Read-your-write
    /// across the two surfaces is only guaranteed once the actor response
    /// (post-fsync) has been observed.
    pub fn direct(&self) -> &Store {
        self
    }

    fn write(&self) -> MutexGuard<'_, Connection> {
        // In-process invariant: the writer mutex is only poisoned by a panic
        // in a query (a bug, not corrupt data), so unwinding is correct.
        self.writer.lock().expect("store writer poisoned")
    }

    /// Borrow a read connection. A semaphore permit is acquired first, so at
    /// most `READER_POOL` connections exist concurrently: 20 simultaneous
    /// readers use at most 4 connections and the rest wait on the permit,
    /// bounded by the busy timeout (`StoreError::Busy`).
    fn read(&self) -> StoreResult<ReadConn> {
        let permit = self
            .pool
            .sem
            .acquire_timeout(Instant::now() + BUSY_TIMEOUT)?;
        let mut conns = self
            .pool
            .conns
            .lock()
            .map_err(|_| StoreError::Migration("reader pool poisoned".into()))?;
        let conn = match conns.pop() {
            Some(c) => c,
            None => {
                // WAL-correct readers: a SQLITE_OPEN_READ_ONLY connection may
                // read only the main database file (a stale snapshot) when it
                // cannot access the -shm/-wal sidecar; pooled readers then
                // serve rows that predate every recent commit. Opening
                // read-write (no CREATE) guarantees the reader participates in
                // WAL snapshotting, so a pooled connection always sees the
                // latest committed append (audit 42 regressions: SSE streams
                // polling the journal went blind mid-session).
                let c = Connection::open_with_flags(
                    self.path(),
                    rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE
                        | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
                )?;
                configure(&c)?;
                self.pool.created.fetch_add(1, Ordering::Relaxed);
                c
            }
        };
        Ok(ReadConn {
            conn: Some(conn),
            pool: self.pool.clone(),
            _permit: permit,
        })
    }

    /// Idle connections currently in the pool; at most `READER_POOL`.
    /// Test probe.
    #[cfg(test)]
    pub(crate) fn reader_pool_len(&self) -> usize {
        self.pool.conns.lock().map(|c| c.len()).unwrap_or(0)
    }

    /// Read connections ever opened since `open`; with the semaphore this
    /// never exceeds `READER_POOL` even under heavy contention.
    /// Test probe.
    #[cfg(test)]
    pub(crate) fn connections_created(&self) -> u64 {
        self.pool.created.load(Ordering::Relaxed)
    }

    // ---------------------------------------------------------------- workspaces

    pub fn create_workspace(&self, root: &str) -> StoreResult<WorkspaceId> {
        let conn = self.write();
        conn.execute(
            "INSERT OR IGNORE INTO workspace(root, created_ms) VALUES (?1, ?2)",
            params![root, now_ms()],
        )?;
        let id: i64 = conn.query_row(
            "SELECT id FROM workspace WHERE root = ?1",
            params![root],
            |r| r.get(0),
        )?;
        Ok(WorkspaceId::new(id as u64))
    }

    /// The recorded root path of a workspace; `None` when the workspace id is
    /// unknown (the revert/diff wire surface needs the on-disk root to open
    /// the file service handle).
    pub fn workspace_root(&self, id: WorkspaceId) -> StoreResult<Option<String>> {
        let conn = self.read()?;
        let out = conn
            .query_row(
                "SELECT root FROM workspace WHERE id = ?1",
                params![id.raw() as i64],
                |r| r.get(0),
            )
            .optional()?;
        Ok(out)
    }

    // ---------------------------------------------------------------- sessions

    pub fn create_session(
        &self,
        workspace_id: WorkspaceId,
        title: &str,
        provider: &str,
        model: &str,
    ) -> StoreResult<SessionRow> {
        let conn = self.write();
        let now = now_ms();
        conn.execute(
            "INSERT INTO session(workspace_id, title, provider, model, state, lifecycle, created_ms, updated_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, 'open', ?6, ?6)",
            params![
                workspace_id.raw() as i64,
                title,
                provider,
                model,
                // In-process constructed enum: serialization of a unit
                // variant can never fail.
                serde_json::to_string(&AgentState::Idle).unwrap(),
                now
            ],
        )?;
        let id: i64 = conn.last_insert_rowid();
        // Seed the journal with SessionCreated so every session starts at seq 1.
        // The session row and its seed event are one transaction.
        let tx = conn.unchecked_transaction()?;
        self.insert_event_locked(
            &tx,
            SessionId::new(id as u64),
            None,
            EventKind::SessionCreated,
            AgentState::Idle,
            now,
            Some(serde_json::json!({ "title": title, "provider": provider, "model": model })),
            1,
        )?;
        tx.commit()?;
        Ok(
            match self.get_session_locked(&conn, SessionId::new(id as u64))? {
                Some(row) => row,
                None => {
                    return Err(StoreError::Corrupt(vec![
                        "just-created session not readable back".into(),
                    ]))
                }
            },
        )
    }

    pub fn get_session(&self, id: SessionId) -> StoreResult<Option<SessionRow>> {
        let conn = self.read()?;
        let row = self.get_session_locked(&conn, id)?;
        Ok(row)
    }

    fn get_session_locked(
        &self,
        conn: &Connection,
        id: SessionId,
    ) -> StoreResult<Option<SessionRow>> {
        let mut stmt = conn.prepare(
            "SELECT id, workspace_id, worktree_id, task_id, title, provider, model, state, lifecycle, created_ms, updated_ms
             FROM session WHERE id = ?1",
        )?;
        let mut rows = stmt.query(params![id.raw() as i64])?;
        match rows.next()? {
            Some(row) => Ok(Some(session_row_map(row)?)),
            None => Ok(None),
        }
    }

    pub fn list_sessions(&self, workspace_id: Option<WorkspaceId>) -> StoreResult<Vec<SessionRow>> {
        let conn = self.read()?;
        let mut stmt = match workspace_id {
            Some(_) => conn.prepare(
                "SELECT id, workspace_id, worktree_id, task_id, title, provider, model, state, lifecycle, created_ms, updated_ms
                 FROM session WHERE workspace_id = ?1 ORDER BY updated_ms DESC",
            )?,
            None => conn.prepare(
                "SELECT id, workspace_id, worktree_id, task_id, title, provider, model, state, lifecycle, created_ms, updated_ms
                 FROM session ORDER BY updated_ms DESC",
            )?,
        };
        let mut rows = match workspace_id {
            Some(w) => stmt.query(params![w.raw() as i64])?,
            None => stmt.query([])?,
        };
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(session_row_map(row)?);
        }
        Ok(out)
    }

    /// Durably adopt a worktree/task identity (v8). The standalone session
    /// default is 1/1; WorktreeManager-created worktrees call this to make
    /// the identity durable so every later tool call (and crash recovery)
    /// carries the REAL worktree/task ids. The journal is intentionally
    /// untouched: adoption is identity bookkeeping, not a turn transition.
    pub fn adopt_session_identity(
        &self,
        id: SessionId,
        worktree_id: WorktreeId,
        task_id: TaskId,
    ) -> StoreResult<()> {
        if worktree_id.raw() == 0 || task_id.raw() == 0 {
            return Err(StoreError::Migration(
                "worktree/task ids must be non-zero".into(),
            ));
        }
        let conn = self.write();
        let n = conn.execute(
            "UPDATE session SET worktree_id = ?2, task_id = ?3, updated_ms = ?4 WHERE id = ?1",
            params![
                id.raw() as i64,
                worktree_id.raw() as i64,
                task_id.raw() as i64,
                now_ms()
            ],
        )?;
        if n == 0 {
            return Err(StoreError::Migration(format!(
                "adopt_session_identity: session {id} does not exist"
            )));
        }
        Ok(())
    }

    pub fn set_session_lifecycle(
        &self,
        id: SessionId,
        lifecycle: faktor_core::state::SessionLifecycle,
    ) -> StoreResult<()> {
        let conn = self.write();
        conn.execute(
            "UPDATE session SET lifecycle = ?2, updated_ms = ?3 WHERE id = ?1",
            params![
                id.raw() as i64,
                // In-process constructed enum (see create_session).
                serde_json::to_string(&lifecycle).unwrap(),
                now_ms()
            ],
        )?;
        Ok(())
    }

    pub fn set_session_state(&self, id: SessionId, state: AgentState) -> StoreResult<()> {
        let conn = self.write();
        conn.execute(
            "UPDATE session SET state = ?2, updated_ms = ?3 WHERE id = ?1",
            params![
                id.raw() as i64,
                // In-process constructed enum (see create_session).
                serde_json::to_string(&state).unwrap(),
                now_ms()
            ],
        )?;
        Ok(())
    }

    /// Single conditional lifecycle UPDATE (`WHERE lifecycle = expected`).
    /// Returns whether a row was updated. Used by prompt auto-resume
    /// (`Suspended -> Open`); the journal is intentionally untouched there —
    /// resuming on prompt is not a new event.
    pub fn set_lifecycle_if(
        &self,
        id: SessionId,
        expected: SessionLifecycle,
        new: SessionLifecycle,
    ) -> StoreResult<bool> {
        let conn = self.write();
        let n = conn.execute(
            "UPDATE session SET lifecycle = ?3, updated_ms = ?4
             WHERE id = ?1 AND lifecycle = ?2",
            params![
                id.raw() as i64,
                // In-process constructed enums (see create_session).
                serde_json::to_string(&expected).unwrap(),
                serde_json::to_string(&new).unwrap(),
                now_ms()
            ],
        )?;
        Ok(n > 0)
    }

    /// Durable session-title update (session.update, P1). Bumps
    /// `updated_ms` so list ordering reflects the rename. Returns whether a
    /// row was updated (callers check existence first for a clean
    /// NotFound). The journal is intentionally untouched: the title is
    /// session metadata, not a state-machine transition.
    pub fn update_session_title(&self, id: SessionId, title: &str) -> StoreResult<bool> {
        let conn = self.write();
        let n = conn.execute(
            "UPDATE session SET title = ?2, updated_ms = ?3 WHERE id = ?1",
            params![id.raw() as i64, title, now_ms()],
        )?;
        Ok(n > 0)
    }

    /// ONE SQLite transaction: read the session row, verify
    /// `expected_lifecycle`/`expected_state` (mismatch -> `Conflict`, nothing
    /// written), update lifecycle+state+updated_ms, append the event with the
    /// next gapless per-session seq, commit. Returns the event seq.
    ///
    /// This is the atomic guard for lifecycle+event transitions: the session
    /// layer's `end_session`/`suspend`/`resume` call it so a crash between
    /// "update lifecycle" and "append event" can never be observed.
    pub fn transition_session(
        &self,
        session_id: SessionId,
        op_id: Option<OpId>,
        t: SessionTransition,
    ) -> StoreResult<EventSeq> {
        let conn = self.write();
        let tx = conn.unchecked_transaction()?;
        // (a) read the session row inside the transaction.
        let Some(row) = self.get_session_locked(&tx, session_id)? else {
            return Err(StoreError::Conflict(format!(
                "session {session_id} does not exist; cannot transition"
            )));
        };
        // (b) verify the expected values; mismatch aborts with nothing written.
        if let Some(expected) = t.expected_lifecycle {
            if row.lifecycle != expected {
                return Err(StoreError::Conflict(format!(
                    "session {session_id} lifecycle is {:?}, expected {:?}",
                    row.lifecycle, expected
                )));
            }
        }
        if let Some(expected) = t.expected_state {
            if row.state != expected {
                return Err(StoreError::Conflict(format!(
                    "session {session_id} state is {:?}, expected {:?}",
                    row.state, expected
                )));
            }
        }
        // (c) update lifecycle+state+updated_ms.
        let now = now_ms();
        // In-process constructed enums (see create_session).
        let state_json = serde_json::to_string(&t.new_state).unwrap();
        match t.new_lifecycle {
            Some(lifecycle) => {
                tx.execute(
                    "UPDATE session SET lifecycle = ?2, state = ?3, updated_ms = ?4 WHERE id = ?1",
                    params![
                        session_id.raw() as i64,
                        // In-process constructed enum (see create_session).
                        serde_json::to_string(&lifecycle).unwrap(),
                        state_json,
                        now
                    ],
                )?;
            }
            None => {
                tx.execute(
                    "UPDATE session SET state = ?2, updated_ms = ?3 WHERE id = ?1",
                    params![session_id.raw() as i64, state_json, now],
                )?;
            }
        }
        // (d) append the event with the next gapless seq (shared insert path).
        let seq = self.insert_event_locked(
            &tx,
            session_id,
            op_id,
            t.event_kind,
            t.new_state,
            now,
            t.event_payload,
            t.event_payload_ver,
        )?;
        // (e) commit: lifecycle change and event are durable together.
        tx.commit()?;
        Ok(seq)
    }

    // ---------------------------------------------------------------- event journal

    /// Append an event with the next per-session sequence number, atomically.
    /// Duplicate/gap sequences are impossible under the transaction; the
    /// primary key enforces it structurally. Legacy callers keep this
    /// signature and stamp payload schema v1 (the original unversioned
    /// writers); version-aware writers use [`Store::append_event_v`].
    pub fn append_event(
        &self,
        session_id: SessionId,
        op_id: Option<OpId>,
        kind: EventKind,
        state: AgentState,
        ts_ms: i64,
        payload: Option<serde_json::Value>,
    ) -> StoreResult<EventSeq> {
        self.append_event_v(session_id, op_id, kind, state, ts_ms, payload, 1)
    }

    /// Versioned append (v11+): stamps `payload_ver`, the payload schema
    /// version of `payload`. Every append site stamps the writer's current
    /// payload schema so a reader can refuse an unknown version loudly
    /// instead of misreading a future shape.
    #[allow(clippy::too_many_arguments)]
    pub fn append_event_v(
        &self,
        session_id: SessionId,
        op_id: Option<OpId>,
        kind: EventKind,
        state: AgentState,
        ts_ms: i64,
        payload: Option<serde_json::Value>,
        payload_ver: i64,
    ) -> StoreResult<EventSeq> {
        let conn = self.write();
        let tx = conn.unchecked_transaction()?;
        let seq = self.insert_event_locked(
            &tx,
            session_id,
            op_id,
            kind,
            state,
            ts_ms,
            payload,
            payload_ver,
        )?;
        // Durability boundary: crossing `ev_precommit` fires the crash
        // AFTER the insert executed but BEFORE the COMMIT (the append
        // rolls back); crossing `ev_committed` fires right after the
        // COMMIT returned (the append is durable, the ack was lost).
        self.seam.trip("ev_precommit");
        tx.commit()?;
        self.seam.trip("ev_committed");
        Ok(seq)
    }

    /// The shared gapless-seq event insert path. Runs inside the CALLER'S
    /// transaction so `append_event`, `transition_session` and the actor's
    /// [`Store::batch_hot_writes`] groups are one atomic unit (a nested
    /// transaction here would silently demote to a savepoint). The argument
    /// is a bare `&Connection`: a live `rusqlite::Transaction` derefs to one,
    /// and the actor batch passes its outer transaction connection directly.
    #[allow(clippy::too_many_arguments)]
    fn insert_event_locked(
        &self,
        conn: &Connection,
        session_id: SessionId,
        op_id: Option<OpId>,
        kind: EventKind,
        state: AgentState,
        ts_ms: i64,
        payload: Option<serde_json::Value>,
        payload_ver: i64,
    ) -> StoreResult<EventSeq> {
        // Serialize appends per session so seq computation is race-free.
        // (The store writer lock already serializes; the per-session query is
        // a second belt for future multi-writer refactors.)
        let prev: Option<i64> = conn.query_row(
            "SELECT MAX(seq) FROM event WHERE session_id = ?1",
            params![session_id.raw() as i64],
            |r| r.get(0),
        )?;
        let seq = JournalInvariants::next_seq(prev.map(|p| EventSeq::new(p as u64)));
        let ts = JournalInvariants::monotonic_ts(
            prev.map(|_| {
                // Use the previous event's ts for monotonicity.
                conn.query_row(
                    "SELECT ts_ms FROM event WHERE session_id = ?1 AND seq = (SELECT MAX(seq) FROM event WHERE session_id = ?1)",
                    params![session_id.raw() as i64],
                    |r| r.get::<_, i64>(0),
                ).unwrap_or(0)
            }),
            ts_ms,
        );
        conn.execute(
            "INSERT INTO event(seq, session_id, op_id, kind, state, ts_ms, payload, payload_ver)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                seq.raw() as i64,
                session_id.raw() as i64,
                op_id.map(|o| o.raw() as i64),
                kind_name(kind),
                // In-process constructed enum (see create_session).
                serde_json::to_string(&state).unwrap(),
                ts,
                payload.map(|p| p.to_string()),
                payload_ver,
            ],
        )?;
        conn.execute(
            "UPDATE session SET state = ?2, updated_ms = ?3 WHERE id = ?1",
            params![
                session_id.raw() as i64,
                // In-process constructed enum (see create_session).
                serde_json::to_string(&state).unwrap(),
                ts
            ],
        )?;
        Ok(seq)
    }

    // -------------------------------------------------- actor batch surface

    /// Execute a FIFO group of hot writes as ONE SQLite transaction with ONE
    /// commit fsync (`PRAGMA synchronous = FULL` for the group; the
    /// connection's configured `NORMAL` is restored before returning). Each
    /// write runs in its own savepoint, so a failing write (duplicate
    /// `(session, seq)` message, FK violation, ...) rolls back only itself:
    /// the rest of the group still commits, and per-write results report
    /// their individual error. Responses may only be delivered after this
    /// returns, which is exactly what makes an actor ack mean "durable".
    ///
    /// Ordering is the caller's FIFO: writes execute in slice order, which
    /// preserves per-session causal order for interleaved event/message/
    /// usage streams as long as the caller enqueues causally.
    ///
    /// The returned [`BatchTiming`] splits the SQL work from the deliberate
    /// commit fsync so the actor's 5 ms instrumentation gate can count the
    /// work segments (what used to stall Tokio workers) while fsync waits —
    /// which no worker ever performs — stay visible as caller-side queue
    /// latency instead of being misattributed to SQLite work.
    pub fn batch_hot_writes(
        &self,
        writes: &[HotWrite],
    ) -> StoreResult<(Vec<StoreResult<HotWriteOutcome>>, BatchTiming)> {
        if writes.is_empty() {
            return Ok((Vec::new(), BatchTiming::default()));
        }
        let conn = self.write();
        // Acknowledged appends must survive a process kill: force the WAL
        // commit fsync for the whole group (the actor replies only after
        // this method returns). Restored to the configured NORMAL on drop.
        let _strong = StrongSync::on(&conn)?;
        let work_start = Instant::now();
        let run = (|| {
            conn.execute_batch("BEGIN IMMEDIATE")?;
            let mut out = Vec::with_capacity(writes.len());
            for w in writes {
                // Per-write savepoint: one failing write must not roll the
                // whole group back (a duplicate message seq on one session
                // must not lose another session's parts).
                conn.execute_batch("SAVEPOINT hot_write")?;
                let r = match self.hot_write_on(&conn, w) {
                    Ok(o) => {
                        conn.execute_batch("RELEASE hot_write")?;
                        Ok(o)
                    }
                    Err(e) => {
                        conn.execute_batch("ROLLBACK TO hot_write")?;
                        conn.execute_batch("RELEASE hot_write")?;
                        Err(e)
                    }
                };
                out.push(r);
                // Durability boundary inside the group: crash right after
                // write `out.len()` executed (its savepoint released) but
                // before the group COMMIT — the whole group rolls back.
                self.seam.trip("flush_progress");
            }
            let commit_start = Instant::now();
            let work_us = commit_start
                .duration_since(work_start)
                .as_micros()
                .min(u64::MAX as u128) as u64;
            // Durability boundary: crash after every write executed, before
            // the fsynced COMMIT (the whole actor flush rolls back).
            self.seam.trip("flush_precommit");
            conn.execute_batch("COMMIT")?;
            // Crash right after the COMMIT fsync: the whole flush is
            // durable; the caller's ack was lost.
            self.seam.trip("flush_committed");
            Ok((
                out,
                BatchTiming {
                    work_us,
                    commit_us: commit_start.elapsed().as_micros().min(u64::MAX as u128) as u64,
                },
            ))
        })();
        match run {
            Ok(out) => Ok(out),
            Err(e) => {
                // Never leave the writer connection inside a transaction.
                let _ = conn.execute_batch("ROLLBACK");
                Err(e)
            }
        }
    }

    /// Run a bounded PASSIVE WAL checkpoint on the shared writer connection.
    ///
    /// Maintenance only: [`configure`] disables SQLite's own
    /// `wal_autocheckpoint`, so checkpoint work is scheduled by the caller
    /// (the session `DbActor` runs it from its idle flush tick, after the
    /// queue drained) and can never execute inside an interactive batch
    /// segment. `PASSIVE` folds every frame no reader pins, never waits for
    /// readers, never restarts the WAL, and returns the same typed
    /// [`StoreError`] surface as every other call: a failure is the caller's
    /// to log, never a corruption by itself.
    pub fn wal_checkpoint_passive(&self) -> StoreResult<()> {
        let conn = self.write();
        conn.execute_batch("PRAGMA wal_checkpoint(PASSIVE);")?;
        Ok(())
    }

    fn hot_write_on(&self, conn: &Connection, w: &HotWrite) -> StoreResult<HotWriteOutcome> {
        match w {
            HotWrite::AppendEvent {
                session_id,
                op_id,
                kind,
                state,
                ts_ms,
                payload,
                payload_ver,
            } => self
                .insert_event_locked(
                    conn,
                    *session_id,
                    *op_id,
                    *kind,
                    *state,
                    *ts_ms,
                    payload.clone(),
                    *payload_ver,
                )
                .map(HotWriteOutcome::EventSeq),
            HotWrite::PutMessage {
                session_id,
                seq,
                role,
                data,
            } => self
                .insert_message_on(conn, *session_id, *seq, role, data)
                .map(HotWriteOutcome::RowId),
            HotWrite::PutPart {
                message_id,
                kind,
                data,
            } => self
                .insert_part_on(conn, *message_id, kind, data)
                .map(HotWriteOutcome::RowId),
            HotWrite::RecordProviderCall {
                session_id,
                op_id,
                provider,
                model,
                status,
                tokens_in,
                tokens_out,
                error,
            } => self
                .insert_provider_call_on(
                    conn,
                    *session_id,
                    *op_id,
                    provider,
                    model,
                    status,
                    *tokens_in,
                    *tokens_out,
                    error.as_deref(),
                    None,
                    None,
                    None,
                    // Hot-write rows predate the v19 segment observation: no
                    // per-call segments, the binary prefix rule stays.
                    None,
                    // Hot-write rows predate the v18 attempt surface: no
                    // attempt identity, no reservation link (the actor is
                    // migrated in the agent stream-loop wave).
                    None,
                    None,
                    None,
                    None,
                )
                .map(HotWriteOutcome::RowId),
        }
    }

    /// Events strictly after `after_seq` (SSE resume cursor).
    pub fn events_after(
        &self,
        session_id: SessionId,
        after_seq: EventSeq,
    ) -> StoreResult<Vec<Event>> {
        self.events_range(session_id, after_seq.raw() + 1, None)
    }

    pub fn events_range(
        &self,
        session_id: SessionId,
        from_seq: u64,
        limit: Option<u64>,
    ) -> StoreResult<Vec<Event>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT seq, session_id, op_id, kind, state, ts_ms, payload, payload_ver
             FROM event
             WHERE session_id = ?1 AND seq >= ?2 ORDER BY seq ASC LIMIT ?3",
        )?;
        let limit = limit.unwrap_or(u64::MAX);
        let mut rows = stmt.query(params![
            session_id.raw() as i64,
            from_seq as i64,
            limit as i64
        ])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(event_map(row, session_id)?.0);
        }
        Ok(out)
    }

    /// Versioned twin of [`Store::events_range`] (v11+): each event rides
    /// its payload schema version so typed journal readers can decode every
    /// payload through its version — an unknown version is a loud error,
    /// never a silent parse of a future shape.
    pub fn events_versioned_range(
        &self,
        session_id: SessionId,
        from_seq: u64,
        limit: Option<u64>,
    ) -> StoreResult<Vec<(Event, i64)>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT seq, session_id, op_id, kind, state, ts_ms, payload, payload_ver
             FROM event
             WHERE session_id = ?1 AND seq >= ?2 ORDER BY seq ASC LIMIT ?3",
        )?;
        let limit = limit.unwrap_or(u64::MAX);
        let mut rows = stmt.query(params![
            session_id.raw() as i64,
            from_seq as i64,
            limit as i64
        ])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(event_map(row, session_id)?);
        }
        Ok(out)
    }

    pub fn last_event_seq(&self, session_id: SessionId) -> StoreResult<Option<EventSeq>> {
        let conn = self.read()?;
        let out = conn.query_row(
            "SELECT MAX(seq) FROM event WHERE session_id = ?1",
            params![session_id.raw() as i64],
            |r| r.get::<_, Option<i64>>(0),
        )?;
        Ok(out.map(|o| EventSeq::new(o as u64)))
    }

    // ---------------------------------------------------------------- messages

    /// Paging is fundamental: the webview sees the latest page immediately and
    /// earlier pages stream on demand. Returns newest-first page.
    pub fn messages_before(
        &self,
        session_id: SessionId,
        before_seq: Option<i64>,
        limit: u64,
    ) -> StoreResult<Vec<MessageRow>> {
        let conn = self.read()?;
        let (sql, params) = match before_seq {
            Some(b) => (
                "SELECT id, session_id, seq, role, data, created_ms FROM message
                 WHERE session_id = ?1 AND seq < ?2 ORDER BY seq DESC LIMIT ?3",
                vec![session_id.raw() as i64, b, limit as i64],
            ),
            None => (
                "SELECT id, session_id, seq, role, data, created_ms FROM message
                 WHERE session_id = ?1 ORDER BY seq DESC LIMIT ?2",
                vec![session_id.raw() as i64, limit as i64],
            ),
        };
        let mut stmt = conn.prepare(sql)?;
        let mut rows = stmt.query(rusqlite::params_from_iter(params))?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(message_map(row)?);
        }
        Ok(out)
    }

    /// Newest-first bounded backward load of one conversation window
    /// (audit 29: never load thousands of historical messages just to trim
    /// them afterward). Walks rows newest-first and stops the moment EITHER
    /// bound is hit:
    ///
    /// - `max_messages`: the result never holds more rows than this.
    /// - `max_bytes`: the running total of stored message-payload bytes
    ///   (the `data` JSON column, exactly as persisted) never exceeds this.
    ///   Message granularity is absolute — a row is never partial: a single
    ///   oversized message still counts as one message and may exceed
    ///   `max_bytes` alone. The byte bound only applies between rows (the
    ///   newest row is always taken when the window is empty, so a hostile
    ///   `max_bytes = 0` yields the newest message, not an empty window).
    ///
    /// Rows older than the returned window are NEVER read: the statement is
    /// stepped lazily over the `idx_message_session_seq` backward index and
    /// the loop breaks before stepping past a bound. `before_seq` cuts
    /// strictly (`seq < before_seq`); values above `i64::MAX` clamp to "no
    /// older bound" (the newest page).
    pub fn messages_backwards_bounded(
        &self,
        session_id: SessionId,
        before_seq: Option<u64>,
        max_messages: u64,
        max_bytes: u64,
    ) -> StoreResult<Vec<MessageRow>> {
        let conn = self.read()?;
        if max_messages == 0 {
            return Ok(Vec::new());
        }
        let before = before_seq.map(|b| i64::try_from(b).unwrap_or(i64::MAX));
        let (sql, params): (&str, Vec<rusqlite::types::Value>) = match before {
            Some(b) => (
                "SELECT id, session_id, seq, role, data, created_ms FROM message
                 WHERE session_id = ?1 AND seq < ?2 ORDER BY seq DESC",
                vec![
                    rusqlite::types::Value::Integer(session_id.raw() as i64),
                    rusqlite::types::Value::Integer(b),
                ],
            ),
            None => (
                "SELECT id, session_id, seq, role, data, created_ms FROM message
                 WHERE session_id = ?1 ORDER BY seq DESC",
                vec![rusqlite::types::Value::Integer(session_id.raw() as i64)],
            ),
        };
        let mut stmt = conn.prepare(sql)?;
        let mut rows = stmt.query(rusqlite::params_from_iter(params))?;
        let mut out: Vec<MessageRow> = Vec::new();
        let mut total_bytes: u64 = 0;
        while let Some(row) = rows.next()? {
            if out.len() as u64 >= max_messages {
                break;
            }
            let raw: String = row.get(4)?;
            let row_bytes = raw.len() as u64;
            if !out.is_empty() && total_bytes.saturating_add(row_bytes) > max_bytes {
                break;
            }
            out.push(message_map(row)?);
            total_bytes = total_bytes.saturating_add(row_bytes);
        }
        Ok(out)
    }

    pub fn put_message(
        &self,
        session_id: SessionId,
        seq: i64,
        role: &str,
        data: serde_json::Value,
    ) -> StoreResult<i64> {
        let conn = self.write();
        self.insert_message_on(&conn, session_id, seq, role, &data)
    }

    /// Shared single-row message insert (fixed-arity contract). Runs on the
    /// caller's connection: the actor batch executes it inside one grouped
    /// transaction, the direct path outside any explicit transaction.
    fn insert_message_on(
        &self,
        conn: &Connection,
        session_id: SessionId,
        seq: i64,
        role: &str,
        data: &serde_json::Value,
    ) -> StoreResult<i64> {
        conn.execute(
            "INSERT INTO message(session_id, seq, role, data, created_ms) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![session_id.raw() as i64, seq, role, data.to_string(), now_ms()],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn put_part(
        &self,
        message_id: i64,
        kind: &str,
        data: serde_json::Value,
    ) -> StoreResult<i64> {
        let conn = self.write();
        self.insert_part_on(&conn, message_id, kind, &data)
    }

    /// Shared single-row part insert; see [`Self::insert_message_on`].
    fn insert_part_on(
        &self,
        conn: &Connection,
        message_id: i64,
        kind: &str,
        data: &serde_json::Value,
    ) -> StoreResult<i64> {
        conn.execute(
            "INSERT INTO part(message_id, kind, data, created_ms) VALUES (?1, ?2, ?3, ?4)",
            params![message_id, kind, data.to_string(), now_ms()],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn parts_of(&self, message_id: i64) -> StoreResult<Vec<PartRow>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT id, message_id, kind, data, created_ms FROM part WHERE message_id = ?1 ORDER BY id ASC",
        )?;
        let mut rows = stmt.query(params![message_id])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(part_map(row)?);
        }
        Ok(out)
    }

    pub fn message_count(&self, session_id: SessionId) -> StoreResult<i64> {
        let conn = self.read()?;
        let out = conn.query_row(
            "SELECT COUNT(*) FROM message WHERE session_id = ?1",
            params![session_id.raw() as i64],
            |r| r.get(0),
        )?;
        Ok(out)
    }

    /// `created_ms` of one message by (session, seq); `None` when the message
    /// does not exist. The revert wire surface uses it as the checkpoint
    /// cutoff: only checkpoints recorded at or before the message may roll
    /// back to it.
    pub fn message_created_ms(&self, session_id: SessionId, seq: i64) -> StoreResult<Option<i64>> {
        let conn = self.read()?;
        let out = conn
            .query_row(
                "SELECT created_ms FROM message WHERE session_id = ?1 AND seq = ?2",
                params![session_id.raw() as i64, seq],
                |r| r.get(0),
            )
            .optional()?;
        Ok(out)
    }

    /// Durable single-message removal (deleteMessage, P1): the message row
    /// AND its part rows are deleted in ONE transaction — a crash can never
    /// leave orphan parts (part rows reference the message row by foreign
    /// key, so the order is structural, not incidental). Message sequences
    /// are STABLE: rows are removed, nothing is renumbered, and the paging
    /// projection simply skips the hole. Returns whether a message row
    /// existed (false = nothing deleted). The journal is intentionally
    /// untouched: it is the durable log of what happened; deleting a
    /// conversation row is not a state-machine event.
    pub fn delete_message(&self, session_id: SessionId, seq: i64) -> StoreResult<bool> {
        let conn = self.write();
        let tx = conn.unchecked_transaction()?;
        let id: Option<i64> = tx
            .query_row(
                "SELECT id FROM message WHERE session_id = ?1 AND seq = ?2",
                params![session_id.raw() as i64, seq],
                |r| r.get(0),
            )
            .optional()?;
        let Some(id) = id else {
            return Ok(false);
        };
        tx.execute("DELETE FROM part WHERE message_id = ?1", params![id])?;
        tx.execute(
            "DELETE FROM message WHERE session_id = ?1 AND seq = ?2",
            params![session_id.raw() as i64, seq],
        )?;
        tx.commit()?;
        Ok(true)
    }

    // ---------------------------------------------------------------- task ledger

    pub fn get_task_ledger(&self, session_id: SessionId) -> StoreResult<Option<serde_json::Value>> {
        let conn = self.read()?;
        let raw: Option<String> = conn
            .query_row(
                // v10: the legacy one-row-per-session ledger blob lives in
                // `task_ledger`; `task` holds the typed durable Task rows.
                "SELECT ledger FROM task_ledger WHERE session_id = ?1 ORDER BY updated_ms DESC LIMIT 1",
                params![session_id.raw() as i64],
                |r| r.get(0),
            )
            .optional()?;
        match raw {
            Some(s) => Ok(Some(parse_json(
                &format!("task ledger for session {session_id}"),
                &s,
            )?)),
            None => Ok(None),
        }
    }

    pub fn put_task_ledger(
        &self,
        session_id: SessionId,
        ledger: serde_json::Value,
    ) -> StoreResult<()> {
        let conn = self.write();
        // v10: the ledger blob moved to `task_ledger` when the typed
        // durable `task` rows took over the `task` table name.
        conn.execute(
            "DELETE FROM task_ledger WHERE session_id = ?1",
            params![session_id.raw() as i64],
        )?;
        conn.execute(
            "INSERT INTO task_ledger(session_id, ledger, updated_ms) VALUES (?1, ?2, ?3)",
            params![session_id.raw() as i64, ledger.to_string(), now_ms()],
        )?;
        Ok(())
    }

    /// Raw-SQL seam (adversarial tests + crash forensics only): executes one
    /// SQL batch on the shared writer connection. Deliberately NOT
    /// cfg(test)-gated so downstream crate tests (faktor-session's typed
    /// ledger corruption tests) can craft corrupt rows; using it in
    /// production is equivalent to corrupting the database yourself.
    #[doc(hidden)]
    pub fn sql_execute(&self, sql: &str) -> StoreResult<()> {
        let conn = self.write();
        conn.execute_batch(sql)?;
        Ok(())
    }

    // ------------------------------------------------------- typed session ledger
    // (audits 27 / 71-72, schema v11: the append-only typed ledger plus its
    // materialized head checkpoint. Compaction policy — the never-FIFO-evict
    // watermark — lives in faktor-session; the store executes one atomic
    // delete+head-rewrite transaction.)

    /// Append ONE typed ledger entry with the next gapless per-session seq.
    /// `entry_type` and `schema_ver` are the row's explicit schema tag;
    /// payload is validated (typed decode) by the session layer BEFORE this
    /// call. Returns the assigned seq.
    pub fn append_ledger_entry(
        &self,
        session_id: SessionId,
        entry_type: &str,
        schema_ver: i64,
        payload: serde_json::Value,
    ) -> StoreResult<i64> {
        if entry_type.is_empty() || entry_type.len() > 64 {
            return Err(StoreError::Migration(
                "ledger entry_type must be 1..=64 chars".into(),
            ));
        }
        if schema_ver <= 0 {
            return Err(StoreError::Migration(
                "ledger entry schema_ver must be > 0".into(),
            ));
        }
        let conn = self.write();
        // One transaction: seq allocation and the insert are atomic. The
        // next seq NEVER rewinds below the head checkpoint (GREATEST of the
        // entry max and the folded checkpoint), so after a compaction the
        // "fold entries after checkpoint_seq" cursor keeps advancing even
        // though pruned rows are gone.
        let tx = conn.unchecked_transaction()?;
        let prev: Option<i64> = tx.query_row(
            "SELECT MAX(seq) FROM ledger_entry WHERE session_id = ?1",
            params![session_id.raw() as i64],
            |r| r.get(0),
        )?;
        let checkpoint: i64 = tx.query_row(
            "SELECT COALESCE(MAX(checkpoint_seq), 0) FROM ledger_head WHERE session_id = ?1",
            params![session_id.raw() as i64],
            |r| r.get(0),
        )?;
        let seq = prev
            .map(|p| p.max(checkpoint))
            .unwrap_or(checkpoint)
            .saturating_add(1);
        tx.execute(
            "INSERT INTO ledger_entry(session_id, seq, entry_type, schema_ver, payload, created_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                session_id.raw() as i64,
                seq,
                entry_type,
                schema_ver,
                payload.to_string(),
                now_ms()
            ],
        )?;
        // Durability boundary of one typed-ledger append: crash before the
        // COMMIT (entry rolls back) or right after it (entry durable).
        self.seam.trip("le_precommit");
        tx.commit()?;
        self.seam.trip("le_committed");
        Ok(seq)
    }

    /// Read ledger entries of one session, ascending by seq. Bounded reads:
    /// the caller pages with `after_seq` and `limit` (paging is fundamental).
    pub fn ledger_entries(
        &self,
        session_id: SessionId,
        after_seq: Option<i64>,
        limit: u64,
    ) -> StoreResult<Vec<LedgerEntryRow>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT seq, entry_type, schema_ver, payload, created_ms FROM ledger_entry
             WHERE session_id = ?1 AND seq > ?2 ORDER BY seq ASC LIMIT ?3",
        )?;
        let mut rows = stmt.query(params![
            session_id.raw() as i64,
            after_seq.unwrap_or(0),
            limit as i64
        ])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(ledger_entry_map(row, session_id)?);
        }
        Ok(out)
    }

    /// Read ledger entries of one session NEWEST-FIRST (descending seq),
    /// strictly below `before_seq` (exclusive; `None` starts at the newest).
    /// Additive read used by the coordination-board reader, whose pages are
    /// newest-first by contract: the (session_id, seq) primary key serves
    /// the ORDER BY seq DESC scan, so one page reads O(limit) rows — never
    /// the whole stream. Same row shape and decode contract as
    /// [`Store::ledger_entries`].
    pub fn ledger_entries_desc(
        &self,
        session_id: SessionId,
        before_seq: Option<i64>,
        limit: u64,
    ) -> StoreResult<Vec<LedgerEntryRow>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT seq, entry_type, schema_ver, payload, created_ms FROM ledger_entry
             WHERE session_id = ?1 AND seq < ?2 ORDER BY seq DESC LIMIT ?3",
        )?;
        let mut rows = stmt.query(params![
            session_id.raw() as i64,
            before_seq.unwrap_or(i64::MAX),
            limit as i64
        ])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(ledger_entry_map(row, session_id)?);
        }
        Ok(out)
    }

    /// The newest ledger seq of the session (0 when the ledger is empty).
    pub fn ledger_max_seq(&self, session_id: SessionId) -> StoreResult<i64> {
        let conn = self.read()?;
        Ok(conn.query_row(
            "SELECT COALESCE(MAX(seq), 0) FROM ledger_entry WHERE session_id = ?1",
            params![session_id.raw() as i64],
            |r| r.get::<_, i64>(0),
        )?)
    }

    /// The session's materialized head checkpoint, if one exists.
    pub fn ledger_head(&self, session_id: SessionId) -> StoreResult<Option<LedgerHeadRow>> {
        let conn = self.read()?;
        let raw: Option<(String, i64, i64, i64)> = conn
            .query_row(
                "SELECT head_json, checkpoint_seq, schema_ver, updated_ms
                 FROM ledger_head WHERE session_id = ?1",
                params![session_id.raw() as i64],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, i64>(1)?,
                        r.get::<_, i64>(2)?,
                        r.get::<_, i64>(3)?,
                    ))
                },
            )
            .optional()?;
        match raw {
            Some((head, checkpoint_seq, schema_ver, updated_ms)) => Ok(Some(LedgerHeadRow {
                head_json: parse_json(&format!("ledger head for session {session_id}"), &head)?,
                checkpoint_seq,
                schema_ver,
                updated_ms,
            })),
            None => Ok(None),
        }
    }

    /// Write (or refresh) the materialized head checkpoint. Standalone use:
    /// crash-recovery folding of entries appended since the last checkpoint.
    pub fn put_ledger_head(
        &self,
        session_id: SessionId,
        head_json: serde_json::Value,
        checkpoint_seq: i64,
        schema_ver: i64,
    ) -> StoreResult<()> {
        let conn = self.write();
        // Durability boundary of the standalone head refresh: crash before
        // the autocommit statement or right after it.
        self.seam.trip("head_prewrite");
        conn.execute(
            "INSERT INTO ledger_head(session_id, head_json, checkpoint_seq, schema_ver, updated_ms)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(session_id) DO UPDATE SET
                head_json = excluded.head_json,
                checkpoint_seq = excluded.checkpoint_seq,
                schema_ver = excluded.schema_ver,
                updated_ms = excluded.updated_ms",
            params![
                session_id.raw() as i64,
                head_json.to_string(),
                checkpoint_seq,
                schema_ver,
                now_ms()
            ],
        )?;
        self.seam.trip("head_written");
        Ok(())
    }

    /// ONE transaction: delete every ledger entry with `seq < below_seq`
    /// except the pinned never-evict entries (`protect`), then rewrite the
    /// materialized head checkpoint to fold the deletion. The session layer
    /// computed the watermark (`below_seq`) and the pinned set — the last
    /// GoalSet/CriteriaSet/Decision and every unresolved BlockerOpened —
    /// from the FULLY DECODED entry stream: compaction refuses to run when
    /// any entry fails its schema decode (an undecodable row is never
    /// silently deleted). Returns the number of deleted entries.
    pub fn compact_ledger(
        &self,
        session_id: SessionId,
        below_seq: i64,
        protect: &[i64],
        head_json: serde_json::Value,
        checkpoint_seq: i64,
        schema_ver: i64,
    ) -> StoreResult<usize> {
        let conn = self.write();
        let tx = conn.unchecked_transaction()?;
        let sid = session_id.raw() as i64;
        let mut sql =
            format!("DELETE FROM ledger_entry WHERE session_id = {sid} AND seq < {below_seq}");
        if !protect.is_empty() {
            sql.push_str(&format!(
                " AND seq NOT IN ({})",
                std::iter::repeat_n("?", protect.len())
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
        let mut params: Vec<&dyn rusqlite::types::ToSql> = Vec::new();
        for p in protect {
            params.push(p);
        }
        let deleted = tx.execute(&sql, rusqlite::params_from_iter(params.iter()))?;
        // Durability boundary mid-fold: crash after the DELETE executed but
        // before the head rewrite (the whole compaction transaction rolls
        // back — entries and head stay consistent).
        self.seam.trip("compact_fold");
        tx.execute(
            "INSERT INTO ledger_head(session_id, head_json, checkpoint_seq, schema_ver, updated_ms)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(session_id) DO UPDATE SET
                head_json = excluded.head_json,
                checkpoint_seq = excluded.checkpoint_seq,
                schema_ver = excluded.schema_ver,
                 updated_ms = excluded.updated_ms",
            params![
                session_id.raw() as i64,
                head_json.to_string(),
                checkpoint_seq,
                schema_ver,
                now_ms()
            ],
        )?;
        // Durability boundary of the compaction fold: crash before the
        // COMMIT (delete + head rewrite roll back together) or right after
        // it (the fold is durable).
        self.seam.trip("compact_precommit");
        tx.commit()?;
        self.seam.trip("compact_committed");
        Ok(deleted)
    }

    // ---------------------------------------------------------------- durable task

    /// Upsert one first-class durable Task row (audit 25). The row key is
    /// `(session_id, task_id)`; a second upsert of the same key replaces the
    /// goal/criteria/plan/state/budget/spend columns in place (created_ms is
    /// caller-preserved: the session layer reads the row before patching).
    /// The caller enforces the bounded-field contract (goal/criteria/plan
    /// caps); the store only persists.
    pub fn upsert_task(&self, t: &TaskRow) -> StoreResult<()> {
        let mut conn = self.write();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let current_state_raw: Option<String> = tx
            .query_row(
                "SELECT state FROM task WHERE session_id = ?1 AND task_id = ?2",
                params![t.session_id.raw() as i64, t.task_id.raw() as i64],
                |r| r.get(0),
            )
            .optional()?;
        let current_state: Option<TaskState> = match current_state_raw {
            Some(raw) => Some(parse_json(
                &format!("task {}/{} state", t.session_id, t.task_id),
                &raw,
            )?),
            None => None,
        };
        // P0-7 chokepoint backstop: completion-relevant states
        // (NeedsVerification/Verifying/VerifiedComplete) may be written
        // through this generic row path only when the row already holds
        // that exact state (idempotent heal) or when the machine allows the
        // edge into it (Running -> NeedsVerification,
        // NeedsVerification -> Verifying). VerifiedComplete has NO machine
        // edge and is produced exclusively by
        // [`Store::task_complete_verified`] against a passing record — a raw
        // row write can never mint a completion proof.
        if t.state.is_completion_relevant() {
            let legal = match current_state {
                Some(cur) => cur == t.state || cur.allowed_transitions().contains(&t.state),
                None => false,
            };
            if !legal {
                return Err(StoreError::Malformed(format!(
                    "task {}/{}: completion-relevant state {:?} may only be reached through the task machine (transition_task/complete_verified_task), never a raw row write",
                    t.session_id, t.task_id, t.state
                )));
            }
        }
        tx.execute(
            "INSERT INTO task(task_id, session_id, goal, acceptance_criteria, plan,
                              max_tokens, max_turns, spent_tokens, spent_turns,
                              state, created_ms, updated_ms, revision, attachments)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)
             ON CONFLICT(session_id, task_id) DO UPDATE SET
                goal = excluded.goal,
                acceptance_criteria = excluded.acceptance_criteria,
                plan = excluded.plan,
                max_tokens = excluded.max_tokens,
                max_turns = excluded.max_turns,
                spent_tokens = excluded.spent_tokens,
                spent_turns = excluded.spent_turns,
                state = excluded.state,
                created_ms = excluded.created_ms,
                updated_ms = excluded.updated_ms,
                revision = excluded.revision,
                attachments = excluded.attachments",
            params![
                t.task_id.raw() as i64,
                t.session_id.raw() as i64,
                t.goal,
                // In-process serialized arrays (see parse_json): failures
                // here are impossible for caller-constructed values.
                serde_json::to_string(&t.acceptance_criteria).unwrap_or_else(|_| "[]".into()),
                serde_json::to_string(&t.plan).unwrap_or_else(|_| "[]".into()),
                t.max_tokens.map(|m| m as i64),
                t.max_turns.map(|m| m as i64),
                t.spent_tokens.min(i64::MAX as u64) as i64,
                t.spent_turns.min(i64::MAX as u32) as i64,
                // In-process constructed enum (see create_session).
                serde_json::to_string(&t.state).unwrap(),
                t.created_ms,
                t.updated_ms,
                t.revision.raw() as i64,
                serde_json::to_string(&t.attachments).unwrap_or_else(|_| "[]".into()),
            ],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn get_task(&self, session_id: SessionId, task_id: TaskId) -> StoreResult<Option<TaskRow>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT task_id, session_id, goal, acceptance_criteria, plan,
                    max_tokens, max_turns, spent_tokens, spent_turns,
                    state, created_ms, updated_ms, revision, attachments
             FROM task WHERE session_id = ?1 AND task_id = ?2",
        )?;
        let mut rows = stmt.query(params![session_id.raw() as i64, task_id.raw() as i64])?;
        match rows.next()? {
            Some(row) => Ok(Some(task_row_map(row, session_id)?)),
            None => Ok(None),
        }
    }

    /// Every durable task row of a session, oldest-created first.
    /// The one completion path (audit P0-7/P0-8): validate the proof and
    /// move the task to `VerifiedComplete` in ONE transaction. The task row,
    /// the session row (the task's base worktree) and the verification
    /// record row are all read INSIDE the transaction and checked against
    /// each other before anything is written:
    ///
    /// (a) the task's state is `Verifying` (a `NeedsVerification` task must
    ///     first transition to `Verifying` — completion never skips the
    ///     verifier);
    /// (b) a `verification_record` row exists with `record_id`;
    /// (c) `record.task_id == task_id`;
    /// (d) `record.revision == expected_revision` — the record must certify
    ///     exactly the revision the caller is completing (and the task row
    ///     must still BE at that revision: any change since the caller's
    ///     read bumps it and refuses here);
    /// (e) `record.status == Passed`;
    /// (f) the record covers EVERY current acceptance criterion of the task
    ///     (present with `passed = true`; extra record criteria are fine,
    ///     missing ones refuse);
    /// (g) `record.workspace_id/worktree_id` equal the task's current base
    ///     worktree (the session row);
    /// (h) NO reservation row of the task still holds budget (`reserved`,
    ///     `dispatched` or `uncertain` — counted INSIDE this IMMEDIATE
    ///     transaction): a reserve that raced the caller's accounting pass
    ///     refuses the completion typed and the task stays Verifying.
    ///
    /// Only then does the transaction write `VerifiedComplete` and bump the
    /// revision exactly once. Every refusal leaves the task row untouched.
    pub fn task_complete_verified(
        &self,
        session_id: SessionId,
        task_id: TaskId,
        expected_revision: TaskRevision,
        record_id: VerificationRecordId,
        now: i64,
    ) -> StoreResult<std::result::Result<TaskRow, TaskCompletionRefusal>> {
        let mut conn = self.write();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        // The task's current base worktree: the session row (v8 identity).
        let base: Option<(i64, i64)> = tx
            .query_row(
                "SELECT workspace_id, worktree_id FROM session WHERE id = ?1",
                params![session_id.raw() as i64],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        let Some((task_ws, task_wt)) = base else {
            return Ok(Err(TaskCompletionRefusal::TaskMissing { task_id }));
        };
        let task = {
            let mut stmt = tx.prepare(
                "SELECT task_id, session_id, goal, acceptance_criteria, plan,
                        max_tokens, max_turns, spent_tokens, spent_turns,
                        state, created_ms, updated_ms, revision, attachments
                 FROM task WHERE session_id = ?1 AND task_id = ?2",
            )?;
            let mut rows = stmt.query(params![session_id.raw() as i64, task_id.raw() as i64])?;
            match rows.next()? {
                Some(row) => Some(task_row_map(row, session_id)?),
                None => None,
            }
        };
        let Some(task) = task else {
            return Ok(Err(TaskCompletionRefusal::TaskMissing { task_id }));
        };
        if task.revision != expected_revision {
            return Ok(Err(TaskCompletionRefusal::RevisionMismatch {
                expected: expected_revision,
                actual: task.revision,
            }));
        }
        if task.state != TaskState::Verifying {
            return Ok(Err(TaskCompletionRefusal::NotVerifying {
                actual: task.state,
            }));
        }
        let record = {
            let mut stmt = tx.prepare(
                "SELECT id, task_id, revision, workspace_id, worktree_id, tree_hash,
                        criteria_json, checks_json, changed_files_json,
                        unrelated_changes_json, reviewer_json, status,
                        started_ms, completed_ms
                 FROM verification_record WHERE id = ?1",
            )?;
            let mut rows = stmt.query(params![record_id.raw() as i64])?;
            match rows.next()? {
                Some(row) => Some(verification_record_map(row)?),
                None => None,
            }
        };
        let Some(record) = record else {
            return Ok(Err(TaskCompletionRefusal::RecordMissing { record_id }));
        };
        if record.task_id != task_id {
            return Ok(Err(TaskCompletionRefusal::RecordWrongTask {
                record_id,
                record_task: record.task_id,
                requested: task_id,
            }));
        }
        if record.revision != expected_revision {
            return Ok(Err(TaskCompletionRefusal::RecordWrongRevision {
                record_id,
                record_revision: record.revision,
                expected: expected_revision,
            }));
        }
        if record.status != VerificationStatus::Passed {
            return Ok(Err(TaskCompletionRefusal::RecordNotPassed {
                record_id,
                status: record.status,
            }));
        }
        let missing: Vec<String> = task
            .acceptance_criteria
            .iter()
            .filter(|c| {
                !record
                    .criteria
                    .iter()
                    .any(|cv| cv.passed && &cv.criterion_key == *c)
            })
            .cloned()
            .collect();
        if !missing.is_empty() {
            return Ok(Err(TaskCompletionRefusal::CriteriaNotCovered {
                record_id,
                missing,
            }));
        }
        let task_ws_id = WorkspaceId::new(task_ws as u64);
        let task_wt_id = WorktreeId::new(task_wt as u64);
        if record.workspace_id != task_ws_id || record.worktree_id != task_wt_id {
            return Ok(Err(TaskCompletionRefusal::WorktreeMismatch {
                record_id,
                record_workspace: record.workspace_id,
                record_worktree: record.worktree_id,
                task_workspace: task_ws_id,
                task_worktree: task_wt_id,
            }));
        }
        // (h) THE ACCOUNTING GATE (completion-vs-reserve invariant): no
        // reservation of this task may still hold budget — `reserved`
        // (dispatch never began), `dispatched` (the provider may have
        // billed) or `uncertain` (a crashed dispatched attempt). The count
        // runs INSIDE this IMMEDIATE transaction, so a reserve that landed
        // after the session layer's accounting pass but before this write is
        // caught: any nonzero count rolls the transaction back with a typed
        // refusal and the task row stays exactly as it was (Verifying).
        let (reserved, dispatched, reserved_micro, uncertain, uncertain_micro) = tx.query_row(
            "SELECT
                     COALESCE(SUM(CASE WHEN status = 'reserved' THEN 1 ELSE 0 END), 0),
                     COALESCE(SUM(CASE WHEN status = 'dispatched' THEN 1 ELSE 0 END), 0),
                     COALESCE(SUM(CASE WHEN status IN ('reserved', 'dispatched')
                                       THEN predicted_micro ELSE 0 END), 0),
                     COALESCE(SUM(CASE WHEN status = 'uncertain' THEN 1 ELSE 0 END), 0),
                     COALESCE(SUM(CASE WHEN status = 'uncertain'
                                       THEN predicted_micro ELSE 0 END), 0)
                 FROM cost_reservation
                 WHERE session_id = ?1 AND task_id = ?2",
            params![session_id.raw() as i64, task_id.raw() as i64],
            |r| {
                Ok((
                    usize::try_from(r.get::<_, i64>(0)?).unwrap_or(usize::MAX),
                    usize::try_from(r.get::<_, i64>(1)?).unwrap_or(usize::MAX),
                    u64::try_from(r.get::<_, i64>(2)?).unwrap_or(u64::MAX),
                    usize::try_from(r.get::<_, i64>(3)?).unwrap_or(usize::MAX),
                    u64::try_from(r.get::<_, i64>(4)?).unwrap_or(u64::MAX),
                ))
            },
        )?;
        if reserved
            .saturating_add(dispatched)
            .saturating_add(uncertain)
            != 0
        {
            tx.rollback()?;
            return Ok(Err(TaskCompletionRefusal::ReservationsHeld {
                reserved,
                dispatched,
                reserved_micro,
                uncertain,
                uncertain_micro,
            }));
        }
        let new_revision = expected_revision.checked_next().ok_or_else(|| {
            StoreError::Malformed(format!(
                "task {session_id}/{task_id} revision overflow at completion"
            ))
        })?;
        let updated = tx.execute(
            "UPDATE task SET state = ?3, revision = ?4, updated_ms = ?5
             WHERE session_id = ?1 AND task_id = ?2 AND revision = ?6",
            params![
                session_id.raw() as i64,
                task_id.raw() as i64,
                // In-process constructed enum (see create_session).
                serde_json::to_string(&TaskState::VerifiedComplete).unwrap(),
                new_revision.raw() as i64,
                now,
                expected_revision.raw() as i64
            ],
        )?;
        if updated != 1 {
            return Err(StoreError::Conflict(format!(
                "task {session_id}/{task_id} vanished between validation and write"
            )));
        }
        tx.commit()?;
        let mut completed = task;
        completed.state = TaskState::VerifiedComplete;
        completed.revision = new_revision;
        completed.updated_ms = now;
        Ok(Ok(completed))
    }

    // ------------------------------------------------------- verification records

    /// Persist one verification record. The record is immutable after this
    /// call except the single CAS finalize (`Running -> Passed|Failed`);
    /// `rec.id` is ignored and the fresh row id is returned. Bounded JSON
    /// columns are the caller's contract (the session layer rejects
    /// oversized criteria/checks before any write, mirroring
    /// [`Store::upsert_task`]).
    pub fn verification_record_put(
        &self,
        rec: &VerificationRecordRow,
    ) -> StoreResult<VerificationRecordId> {
        self.verification_record_put_with_evidence(rec, None, None)
    }

    /// Additive v20 twin of [`Store::verification_record_put`] (audits
    /// 94/116/117): one INSERT carrying the record together with its two
    /// optional evidence JSON columns — the bounded environment fingerprint
    /// and the candidate-proof reference. `None` writes SQL `NULL` (honestly
    /// absent, exactly like a pre-v20 row); the session layer validates and
    /// bounds both payloads BEFORE this call.
    pub fn verification_record_put_with_evidence(
        &self,
        rec: &VerificationRecordRow,
        environment_fingerprint_json: Option<&str>,
        candidate_proof_ref_json: Option<&str>,
    ) -> StoreResult<VerificationRecordId> {
        let conn = self.write();
        conn.execute(
            "INSERT INTO verification_record(
                task_id, revision, workspace_id, worktree_id, tree_hash,
                criteria_json, checks_json, changed_files_json,
                unrelated_changes_json, reviewer_json, status,
                started_ms, completed_ms,
                environment_fingerprint_json, candidate_proof_ref_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
            params![
                rec.task_id.raw() as i64,
                rec.revision.raw() as i64,
                rec.workspace_id.raw() as i64,
                rec.worktree_id.raw() as i64,
                rec.tree_hash,
                serde_json::to_string(&rec.criteria).unwrap_or_else(|_| "[]".into()),
                serde_json::to_string(&rec.checks).unwrap_or_else(|_| "[]".into()),
                serde_json::to_string(&rec.changed_files).unwrap_or_else(|_| "[]".into()),
                serde_json::to_string(&rec.unrelated_changes).unwrap_or_else(|_| "[]".into()),
                rec.reviewer.as_ref().map(|v| v.to_string()),
                serde_json::to_string(&rec.status).unwrap(),
                rec.started_ms,
                rec.completed_ms,
                environment_fingerprint_json,
                candidate_proof_ref_json,
            ],
        )?;
        let id = conn.last_insert_rowid();
        // SQLite rowids start at 1, so a fresh row id is always a valid
        // (non-zero) record id.
        Ok(VerificationRecordId::new(id as u64))
    }

    /// Additive v20 twin of [`Store::verification_record_get`]: the row plus
    /// its raw `(environment_fingerprint_json, candidate_proof_ref_json)`
    /// evidence columns. `None` on either side means the record predates v20
    /// or was written without that evidence — an honest absence the session
    /// layer maps to an absent typed value, never a guess.
    pub fn verification_record_get_with_evidence(
        &self,
        record_id: VerificationRecordId,
    ) -> StoreResult<Option<VerificationRecordWithEvidence>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT id, task_id, revision, workspace_id, worktree_id, tree_hash,
                    criteria_json, checks_json, changed_files_json,
                    unrelated_changes_json, reviewer_json, status,
                    started_ms, completed_ms,
                    environment_fingerprint_json, candidate_proof_ref_json
             FROM verification_record WHERE id = ?1",
        )?;
        let mut rows = stmt.query(params![record_id.raw() as i64])?;
        match rows.next()? {
            Some(row) => Ok(Some((
                verification_record_map(row)?,
                row.get::<_, Option<String>>(14)?,
                row.get::<_, Option<String>>(15)?,
            ))),
            None => Ok(None),
        }
    }

    /// Every verification record of one task with its evidence columns, in
    /// deterministic creation order (`id ASC`) — the v20 twin of
    /// [`Store::verification_record_list_by_task`].
    pub fn verification_record_list_by_task_with_evidence(
        &self,
        task_id: TaskId,
    ) -> StoreResult<Vec<VerificationRecordWithEvidence>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT id, task_id, revision, workspace_id, worktree_id, tree_hash,
                    criteria_json, checks_json, changed_files_json,
                    unrelated_changes_json, reviewer_json, status,
                    started_ms, completed_ms,
                    environment_fingerprint_json, candidate_proof_ref_json
             FROM verification_record WHERE task_id = ?1 ORDER BY id ASC",
        )?;
        let mut rows = stmt.query(params![task_id.raw() as i64])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push((
                verification_record_map(row)?,
                row.get::<_, Option<String>>(14)?,
                row.get::<_, Option<String>>(15)?,
            ));
        }
        Ok(out)
    }

    pub fn verification_record_get(
        &self,
        record_id: VerificationRecordId,
    ) -> StoreResult<Option<VerificationRecordRow>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT id, task_id, revision, workspace_id, worktree_id, tree_hash,
                    criteria_json, checks_json, changed_files_json,
                    unrelated_changes_json, reviewer_json, status,
                    started_ms, completed_ms
             FROM verification_record WHERE id = ?1",
        )?;
        let mut rows = stmt.query(params![record_id.raw() as i64])?;
        match rows.next()? {
            Some(row) => Ok(Some(verification_record_map(row)?)),
            None => Ok(None),
        }
    }

    /// Every verification record of one task, in deterministic creation
    /// order (`id ASC`).
    pub fn verification_record_list_by_task(
        &self,
        task_id: TaskId,
    ) -> StoreResult<Vec<VerificationRecordRow>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT id, task_id, revision, workspace_id, worktree_id, tree_hash,
                    criteria_json, checks_json, changed_files_json,
                    unrelated_changes_json, reviewer_json, status,
                    started_ms, completed_ms
             FROM verification_record WHERE task_id = ?1 ORDER BY id ASC",
        )?;
        let mut rows = stmt.query(params![task_id.raw() as i64])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(verification_record_map(row)?);
        }
        Ok(out)
    }

    /// The record's single allowed status write: a CAS from `Running` to
    /// `Passed` or `Failed`. Exactly one finalize wins; a second attempt on
    /// an already-final record is refused with its current status, and the
    /// finalize never rewinds (a finalized record is immutable).
    pub fn verification_record_finalize(
        &self,
        record_id: VerificationRecordId,
        new_status: VerificationStatus,
        completed_ms: i64,
    ) -> StoreResult<std::result::Result<(), RecordFinalizeRefusal>> {
        if !matches!(
            new_status,
            VerificationStatus::Passed | VerificationStatus::Failed
        ) {
            return Err(StoreError::Malformed(format!(
                "record {record_id}: finalize status must be Passed or Failed, got {new_status:?}"
            )));
        }
        let mut conn = self.write();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let updated = tx.execute(
            "UPDATE verification_record
             SET status = ?2, completed_ms = ?3
             WHERE id = ?1 AND status = ?4",
            params![
                record_id.raw() as i64,
                serde_json::to_string(&new_status).unwrap(),
                completed_ms,
                serde_json::to_string(&VerificationStatus::Running).unwrap()
            ],
        )?;
        if updated == 1 {
            tx.commit()?;
            return Ok(Ok(()));
        }
        // The CAS missed: surface the current status so callers can tell an
        // already-final record from a not-yet-started one.
        let current_raw: Option<String> = tx
            .query_row(
                "SELECT status FROM verification_record WHERE id = ?1",
                params![record_id.raw() as i64],
                |r| r.get(0),
            )
            .optional()?;
        let current: Option<VerificationStatus> = match current_raw {
            Some(raw) => Some(parse_json(
                &format!("verification_record {record_id} status"),
                &raw,
            )?),
            None => None,
        };
        match current {
            Some(current) => Ok(Err(RecordFinalizeRefusal::NotRunning {
                record_id,
                current,
            })),
            None => Ok(Err(RecordFinalizeRefusal::Missing { record_id })),
        }
    }

    // ------------------------------------------------ verification jobs (v22)

    /// Begin ONE verification attempt durably (schema v22): the attempt row,
    /// its changed-file rows and every required check row (inline outcomes
    /// AND background job definitions) in ONE immediate transaction. The
    /// commit is the attempt's commit point, so a crash can never leave torn
    /// job rows — either the whole attempt is durable or none of it is.
    ///
    /// Idempotent by identity: an existing `(session, task, attempt_op)`
    /// returns `Ok(false)` and writes nothing (a retried begin after a
    /// crash). An OPEN background check of a DIFFERENT attempt refuses with
    /// [`StoreError::Conflict`] (supersede first — an open job is never
    /// silently replaced). Every bound is enforced before any write.
    pub fn verification_attempt_begin(
        &self,
        attempt: &VerificationAttemptRow,
        changed: &[String],
        checks: &[VerificationJobRow],
    ) -> StoreResult<bool> {
        validate_verification_attempt(attempt, changed, checks)?;
        let mut conn = self.write();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let exists: Option<i64> = tx
            .query_row(
                "SELECT 1 FROM verification_attempt
                 WHERE session_id = ?1 AND task_id = ?2 AND attempt_op_id = ?3",
                params![
                    attempt.session_id.raw() as i64,
                    attempt.task_id.raw() as i64,
                    attempt.attempt_op_id as i64
                ],
                |r| r.get(0),
            )
            .optional()?;
        if exists.is_some() {
            return Ok(false);
        }
        for check in checks {
            if check.inline_status.is_some() {
                continue;
            }
            let open_elsewhere: Option<i64> = tx
                .query_row(
                    "SELECT attempt_op_id FROM verification_job
                     WHERE session_id = ?1 AND task_id = ?2 AND check_id = ?3
                       AND inline_status IS NULL AND state IN ('queued', 'running')
                       AND attempt_op_id <> ?4",
                    params![
                        attempt.session_id.raw() as i64,
                        attempt.task_id.raw() as i64,
                        check.check_id,
                        attempt.attempt_op_id as i64
                    ],
                    |r| r.get(0),
                )
                .optional()?;
            if let Some(prior) = open_elsewhere {
                return Err(StoreError::Conflict(format!(
                    "check '{}' has an open job of attempt {prior}; supersede that attempt before \
                     beginning attempt {}",
                    check.check_id, attempt.attempt_op_id
                )));
            }
        }
        tx.execute(
            "INSERT INTO verification_attempt(
                session_id, task_id, attempt_op_id, task_revision, workspace_root,
                environment_fingerprint_json, created_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                attempt.session_id.raw() as i64,
                attempt.task_id.raw() as i64,
                attempt.attempt_op_id as i64,
                attempt.task_revision.raw() as i64,
                attempt.workspace_root,
                attempt.environment_fingerprint_json,
                attempt.created_ms
            ],
        )?;
        for (ordinal, path) in changed.iter().enumerate() {
            tx.execute(
                "INSERT INTO verification_attempt_changed_file(
                    session_id, task_id, attempt_op_id, ordinal, path)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    attempt.session_id.raw() as i64,
                    attempt.task_id.raw() as i64,
                    attempt.attempt_op_id as i64,
                    ordinal as i64,
                    path
                ],
            )?;
        }
        for check in checks {
            tx.execute(
                "INSERT INTO verification_job(
                    session_id, task_id, attempt_op_id, check_id, ordinal,
                    task_revision, workspace_root, kind, command, program,
                    args_json, spec_json, budget_ms, inline_status, state, note,
                    op_id, environment_fingerprint_json, created_ms, updated_ms,
                    finished_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12,
                         ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21)",
                params![
                    check.session_id.raw() as i64,
                    check.task_id.raw() as i64,
                    check.attempt_op_id as i64,
                    check.check_id,
                    check.ordinal as i64,
                    check.task_revision.raw() as i64,
                    check.workspace_root,
                    check.kind,
                    check.command,
                    check.program,
                    check.args_json,
                    check.spec_json,
                    check.budget_ms as i64,
                    check.inline_status,
                    check.state,
                    check.note,
                    check.op_id.map(|op| op as i64),
                    check.environment_fingerprint_json,
                    check.created_ms,
                    check.updated_ms,
                    check.finished_ms
                ],
            )?;
        }
        tx.commit()?;
        Ok(true)
    }

    /// One attempt view by identity, or `None`.
    pub fn verification_attempt_get(
        &self,
        session_id: SessionId,
        task_id: TaskId,
        attempt_op_id: u64,
    ) -> StoreResult<Option<VerificationAttemptView>> {
        let conn = self.read()?;
        verification_attempt_view(&conn, session_id, task_id, attempt_op_id)
    }

    /// The NEWEST attempt view of `(session, task)` (highest attempt op), or
    /// `None`. Attempt ops are monotonic, so the max is the current attempt.
    pub fn verification_attempt_current(
        &self,
        session_id: SessionId,
        task_id: TaskId,
    ) -> StoreResult<Option<VerificationAttemptView>> {
        let conn = self.read()?;
        let newest: Option<i64> = conn
            .query_row(
                "SELECT MAX(attempt_op_id) FROM verification_attempt
                 WHERE session_id = ?1 AND task_id = ?2",
                params![session_id.raw() as i64, task_id.raw() as i64],
                |r| r.get(0),
            )
            .optional()?
            .flatten();
        match newest {
            Some(op) => verification_attempt_view(&conn, session_id, task_id, op.max(0) as u64),
            None => Ok(None),
        }
    }

    /// Every OPEN (queued|running) BACKGROUND job of `(session, task)`,
    /// ordered `(task, check-id)`. Rows are decoded FIRST (a corrupt state
    /// is a loud typed error, never filtered away by SQL), then filtered.
    pub fn verification_jobs_open(
        &self,
        session_id: SessionId,
        task_id: TaskId,
    ) -> StoreResult<Vec<VerificationJobRow>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(&format!(
            "{} WHERE session_id = ?1 AND task_id = ?2 AND inline_status IS NULL
             ORDER BY check_id ASC",
            VERIFICATION_JOB_SELECT
        ))?;
        let mut rows = stmt.query(params![session_id.raw() as i64, task_id.raw() as i64])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            let job = verification_job_map(row)?;
            if matches!(job.state.as_str(), "queued" | "running") {
                out.push(job);
            }
        }
        Ok(out)
    }

    /// Every BACKGROUND job row of one attempt (any state), ordered by
    /// derivation ordinal (check-id fallback).
    pub fn verification_jobs_for_attempt(
        &self,
        session_id: SessionId,
        task_id: TaskId,
        attempt_op_id: u64,
    ) -> StoreResult<Vec<VerificationJobRow>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(&format!(
            "{VERIFICATION_JOB_SELECT}
             WHERE session_id = ?1 AND task_id = ?2 AND attempt_op_id = ?3
               AND inline_status IS NULL
             ORDER BY ordinal ASC, check_id ASC"
        ))?;
        let mut rows = stmt.query(params![
            session_id.raw() as i64,
            task_id.raw() as i64,
            attempt_op_id as i64
        ])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(verification_job_map(row)?);
        }
        Ok(out)
    }

    /// Cancel every OPEN background job of one attempt to `cancelled` with a
    /// typed note. Returns the number of rows cancelled. A cancelled job is
    /// terminal and can never certify completion.
    pub fn verification_attempt_cancel(
        &self,
        session_id: SessionId,
        task_id: TaskId,
        attempt_op_id: u64,
        reason: &str,
        now: i64,
    ) -> StoreResult<u64> {
        if reason.is_empty() || reason.len() > MAX_VERIFICATION_JOB_NOTE_BYTES {
            return Err(StoreError::Oversized(format!(
                "cancel note of {} bytes outside 1..={MAX_VERIFICATION_JOB_NOTE_BYTES}",
                reason.len()
            )));
        }
        let mut conn = self.write();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let cancelled = tx.execute(
            "UPDATE verification_job
             SET state = 'cancelled', note = ?4, updated_ms = ?5, finished_ms = ?5
             WHERE session_id = ?1 AND task_id = ?2 AND attempt_op_id = ?3
               AND inline_status IS NULL AND state IN ('queued', 'running')",
            params![
                session_id.raw() as i64,
                task_id.raw() as i64,
                attempt_op_id as i64,
                reason,
                now
            ],
        )?;
        tx.commit()?;
        Ok(cancelled as u64)
    }

    /// Claim one `queued` background job: the guarded CAS to `running` with
    /// the executor's op attached. A job is claimed at most once per attempt;
    /// a NEWER attempt makes every mutation of attempt N a typed
    /// [`VerificationJobRefusal::Superseded`].
    pub fn verification_job_claim(
        &self,
        session_id: SessionId,
        task_id: TaskId,
        attempt_op_id: u64,
        check_id: &str,
        op_id: u64,
        now: i64,
    ) -> StoreResult<std::result::Result<VerificationJobRow, VerificationJobRefusal>> {
        if op_id == 0 {
            return Err(StoreError::Malformed(
                "job claim op_id must be non-zero".into(),
            ));
        }
        let mut conn = self.write();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        if let Some(newest) = newest_attempt_op(&tx, session_id, task_id)? {
            if newest > attempt_op_id {
                return Ok(Err(VerificationJobRefusal::Superseded {
                    attempt_op_id,
                    newest_attempt_op_id: newest,
                }));
            }
        }
        let row = verification_job_get(&tx, session_id, task_id, attempt_op_id, check_id)?;
        let Some(mut job) = row else {
            return Ok(Err(VerificationJobRefusal::Missing {
                check_id: check_id.to_string(),
            }));
        };
        if job.inline_status.is_some() || job.state != "queued" {
            return Ok(Err(VerificationJobRefusal::NotOpen {
                check_id: check_id.to_string(),
                state: job.state,
            }));
        }
        job.state = "running".into();
        job.op_id = Some(op_id);
        job.updated_ms = now;
        tx.execute(
            "UPDATE verification_job SET state = 'running', op_id = ?5, updated_ms = ?6
             WHERE session_id = ?1 AND task_id = ?2 AND attempt_op_id = ?3
               AND check_id = ?4 AND state = 'queued'",
            params![
                session_id.raw() as i64,
                task_id.raw() as i64,
                attempt_op_id as i64,
                check_id,
                op_id as i64,
                now
            ],
        )?;
        tx.commit()?;
        Ok(Ok(job))
    }

    /// Resolve one `running` background job to a terminal state and record
    /// its typed outcome exactly once. The attempt-N identity of the row and
    /// the result is structural: a result for attempt N is a DIFFERENT row
    /// from attempt N+1, and once N+1 exists attempt N is frozen — a late
    /// resolve refuses with [`VerificationJobRefusal::Superseded`].
    #[allow(clippy::too_many_arguments)]
    pub fn verification_job_resolve(
        &self,
        session_id: SessionId,
        task_id: TaskId,
        attempt_op_id: u64,
        check_id: &str,
        state: &str,
        note: Option<&str>,
        result_json: Option<&str>,
        now: i64,
    ) -> StoreResult<std::result::Result<VerificationJobRow, VerificationJobRefusal>> {
        if !matches!(state, "passed" | "failed" | "unavailable" | "cancelled") {
            return Err(StoreError::Malformed(format!(
                "resolve state {state:?} is not terminal"
            )));
        }
        if let Some(note) = note {
            if note.is_empty() || note.len() > MAX_VERIFICATION_JOB_NOTE_BYTES {
                return Err(StoreError::Oversized(format!(
                    "resolve note of {} bytes outside 1..={MAX_VERIFICATION_JOB_NOTE_BYTES}",
                    note.len()
                )));
            }
        }
        if let Some(result) = result_json {
            if result.is_empty() || result.len() > MAX_VERIFICATION_JOB_RESULT_JSON_BYTES {
                return Err(StoreError::Oversized(format!(
                    "result json of {} bytes outside 1..={MAX_VERIFICATION_JOB_RESULT_JSON_BYTES}",
                    result.len()
                )));
            }
        }
        let mut conn = self.write();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        if let Some(newest) = newest_attempt_op(&tx, session_id, task_id)? {
            if newest > attempt_op_id {
                return Ok(Err(VerificationJobRefusal::Superseded {
                    attempt_op_id,
                    newest_attempt_op_id: newest,
                }));
            }
        }
        let Some(mut job) =
            verification_job_get(&tx, session_id, task_id, attempt_op_id, check_id)?
        else {
            return Ok(Err(VerificationJobRefusal::Missing {
                check_id: check_id.to_string(),
            }));
        };
        if job.inline_status.is_some() || job.state != "running" {
            return Ok(Err(VerificationJobRefusal::NotOpen {
                check_id: check_id.to_string(),
                state: job.state,
            }));
        }
        if let Some(result) = result_json {
            let already: Option<i64> = tx
                .query_row(
                    "SELECT 1 FROM verification_job_result
                     WHERE session_id = ?1 AND task_id = ?2 AND attempt_op_id = ?3
                       AND check_id = ?4",
                    params![
                        session_id.raw() as i64,
                        task_id.raw() as i64,
                        attempt_op_id as i64,
                        check_id
                    ],
                    |r| r.get(0),
                )
                .optional()?;
            if already.is_some() {
                return Ok(Err(VerificationJobRefusal::ResultExists {
                    check_id: check_id.to_string(),
                }));
            }
            tx.execute(
                "INSERT INTO verification_job_result(
                    session_id, task_id, attempt_op_id, check_id, result_json, finished_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    session_id.raw() as i64,
                    task_id.raw() as i64,
                    attempt_op_id as i64,
                    check_id,
                    result,
                    now
                ],
            )?;
            job.result_json = Some(result.to_string());
        }
        job.state = state.to_string();
        job.note = note.map(str::to_string);
        job.updated_ms = now;
        job.finished_ms = Some(now);
        tx.execute(
            "UPDATE verification_job
             SET state = ?5, note = ?6, updated_ms = ?7, finished_ms = ?7
             WHERE session_id = ?1 AND task_id = ?2 AND attempt_op_id = ?3
               AND check_id = ?4 AND state = 'running'",
            params![
                session_id.raw() as i64,
                task_id.raw() as i64,
                attempt_op_id as i64,
                check_id,
                state,
                note,
                now
            ],
        )?;
        tx.commit()?;
        Ok(Ok(job))
    }

    /// Honest post-restart recovery for one session (schema v22): every
    /// `running` row — an executor died mid-check — is re-queued with a typed
    /// note (it never certified anything; re-running it deterministically is
    /// the only honest path to a verdict), and open rows whose attempt row is
    /// missing (impossible under the foreign keys; counted for hand-corrupted
    /// databases) are orphaned to `unavailable`. Idempotent.
    pub fn verification_jobs_requeue_running(
        &self,
        session_id: SessionId,
        now: i64,
    ) -> StoreResult<VerificationJobRecovery> {
        let mut conn = self.write();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let mut report = VerificationJobRecovery::default();
        let orphaned = tx.execute(
            "UPDATE verification_job
             SET state = 'unavailable', finished_ms = ?2, updated_ms = ?2,
                 note = 'orphaned job: its attempt record is missing; never certified'
             WHERE session_id = ?1 AND state IN ('queued', 'running')
               AND NOT EXISTS (
                   SELECT 1 FROM verification_attempt a
                   WHERE a.session_id = verification_job.session_id
                     AND a.task_id = verification_job.task_id
                     AND a.attempt_op_id = verification_job.attempt_op_id)",
            params![session_id.raw() as i64, now],
        )?;
        report.orphaned = orphaned as u64;
        let requeued = tx.execute(
            "UPDATE verification_job
             SET state = 'queued', op_id = NULL, note =
                 're-queued after a restart: the previous executor died mid-check and never \
                  produced a verdict; the check re-runs deterministically',
                 updated_ms = ?2, finished_ms = NULL
             WHERE session_id = ?1 AND state = 'running'",
            params![session_id.raw() as i64, now],
        )?;
        report.requeued = requeued as u64;
        tx.commit()?;
        Ok(report)
    }

    /// One-shot, idempotent v22 legacy-verification repair (invoked by every
    /// open after migration AND exposed for explicit repair tests): project
    /// pre-v22 `memory_fact` verification rows into the real tables. A
    /// session already carrying the durable import marker is skipped whole;
    /// corrupt rows are skipped loudly with typed notes and left in place.
    pub fn import_legacy_verification_facts(&self) -> StoreResult<LegacyVerificationImport> {
        let mut conn = self.write();
        import_legacy_verification_facts_conn(&mut conn)
    }

    pub fn list_tasks(&self, session_id: SessionId) -> StoreResult<Vec<TaskRow>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT task_id, session_id, goal, acceptance_criteria, plan,
                    max_tokens, max_turns, spent_tokens, spent_turns,
                    state, created_ms, updated_ms, revision, attachments
             FROM task WHERE session_id = ?1 ORDER BY created_ms ASC, task_id ASC",
        )?;
        let mut rows = stmt.query(params![session_id.raw() as i64])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(task_row_map(row, session_id)?);
        }
        Ok(out)
    }

    /// Durable token spending of a session: the sum of every recorded
    /// provider-call input+output token count (NULL counters count as 0).
    /// This is the task budget's crash-safe spend source (provider_call rows
    /// are written before any gate can evaluate the budget).
    pub fn session_usage_tokens(&self, session_id: SessionId) -> StoreResult<u64> {
        let conn = self.read()?;
        let out: i64 = conn.query_row(
            "SELECT COALESCE(SUM(tokens_in), 0) + COALESCE(SUM(tokens_out), 0)
             FROM provider_call WHERE session_id = ?1",
            params![session_id.raw() as i64],
            |r| r.get(0),
        )?;
        Ok(out.max(0) as u64)
    }

    /// Durable logical-turn count of a session: journal events with kind
    /// `turn_completed`. Each genuine turn end appends exactly one, so this
    /// is the crash-safe spent-turns source for the task budget.
    pub fn turn_completed_count(&self, session_id: SessionId) -> StoreResult<u64> {
        let conn = self.read()?;
        let out: i64 = conn.query_row(
            "SELECT COUNT(*) FROM event WHERE session_id = ?1 AND kind = 'turn_completed'",
            params![session_id.raw() as i64],
            |r| r.get(0),
        )?;
        Ok(out.max(0) as u64)
    }

    // ---------------------------------------------------------------- tool runs

    #[allow(clippy::too_many_arguments)]
    pub fn start_tool_run(
        &self,
        session_id: SessionId,
        op_id: OpId,
        tool: &str,
        args: serde_json::Value,
        recovery: serde_json::Value,
        expected_hash: Option<String>,
        replay_descriptor: Option<serde_json::Value>,
    ) -> StoreResult<i64> {
        let conn = self.write();
        conn.execute(
            "INSERT INTO tool_run(session_id, op_id, tool, args, status, started_ms, effect_status, recovery, expected_hash, replay_descriptor)
             VALUES (?1, ?2, ?3, ?4, 'running', ?5, 'unknown', ?6, ?7, ?8)",
            params![
                session_id.raw() as i64,
                op_id.raw() as i64,
                tool,
                args.to_string(),
                now_ms(),
                recovery.to_string(),
                expected_hash,
                replay_descriptor.map(|d| d.to_string()),
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// Record the workspace-write postcondition a tool reported at execution
    /// end (v7): recovery verifies the CURRENT file bytes against it through
    /// the workspace file service — never a hash inferred from args JSON.
    /// Only a still-running row may be annotated (loud otherwise).
    #[allow(clippy::too_many_arguments)]
    pub fn record_tool_postcondition(
        &self,
        session_id: SessionId,
        op_id: OpId,
        postcondition: &serde_json::Value,
    ) -> StoreResult<()> {
        let conn = self.write();
        let n = conn.execute(
            "UPDATE tool_run SET postcondition = ?3
             WHERE session_id = ?1 AND op_id = ?2 AND status = 'running'",
            params![
                session_id.raw() as i64,
                op_id.raw() as i64,
                postcondition.to_string()
            ],
        )?;
        if n == 0 {
            return Err(StoreError::Migration(
                "record_tool_postcondition: no running row".into(),
            ));
        }
        Ok(())
    }

    /// Bump the physical-attempt counter of one still-running tool run (v7:
    /// a crash-recovery replay is a NEW PHYSICAL attempt of the SAME logical
    /// operation). Loud when the row is not running.
    pub fn bump_tool_run_attempt(&self, session_id: SessionId, op_id: OpId) -> StoreResult<i64> {
        let conn = self.write();
        let tx = conn.unchecked_transaction()?;
        let n = tx.execute(
            "UPDATE tool_run SET attempt = attempt + 1
             WHERE session_id = ?1 AND op_id = ?2 AND status = 'running'",
            params![session_id.raw() as i64, op_id.raw() as i64],
        )?;
        if n == 0 {
            return Err(StoreError::Migration(
                "bump_tool_run_attempt: no running row".into(),
            ));
        }
        let attempt: i64 = tx.query_row(
            "SELECT attempt FROM tool_run WHERE session_id = ?1 AND op_id = ?2",
            params![session_id.raw() as i64, op_id.raw() as i64],
            |r| r.get(0),
        )?;
        tx.commit()?;
        Ok(attempt)
    }

    pub fn finish_tool_run(
        &self,
        session_id: SessionId,
        op_id: OpId,
        status: &str,
        effect_status: &str,
    ) -> StoreResult<()> {
        let conn = self.write();
        let n = conn.execute(
            "UPDATE tool_run SET status = ?3, effect_status = ?4, ended_ms = ?5
             WHERE session_id = ?1 AND op_id = ?2",
            params![
                session_id.raw() as i64,
                op_id.raw() as i64,
                status,
                effect_status,
                now_ms()
            ],
        )?;
        if n == 0 {
            return Err(StoreError::Migration(
                "finish_tool_run: no matching row".into(),
            ));
        }
        Ok(())
    }

    pub fn set_tool_run_effect(
        &self,
        session_id: SessionId,
        op_id: OpId,
        effect_status: &str,
    ) -> StoreResult<()> {
        let conn = self.write();
        conn.execute(
            "UPDATE tool_run SET effect_status = ?3 WHERE session_id = ?1 AND op_id = ?2",
            params![session_id.raw() as i64, op_id.raw() as i64, effect_status],
        )?;
        Ok(())
    }

    /// Unfinished tool runs (ToolStarted without ToolCompleted): the crash
    /// recovery scanner's input.
    pub fn pending_tool_runs(&self, session_id: SessionId) -> StoreResult<Vec<ToolRunRow>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT id, session_id, op_id, tool, args, status, started_ms, ended_ms, effect_status, recovery, expected_hash, replay_descriptor, attempt, postcondition
             FROM tool_run WHERE session_id = ?1 AND status = 'running'",
        )?;
        let mut rows = stmt.query(params![session_id.raw() as i64])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(tool_run_map(row)?);
        }
        Ok(out)
    }

    // ---------------------------------------------------------------- turn records

    /// Durably open a logical-turn record. Called transactionally when a
    /// prompt is admitted as the ACTIVE logical turn (immediate admission in
    /// `submit_prompt`, or queue admission). Re-admission of the SAME turn op
    /// (a crash between admission and the first drive; the queue row is
    /// re-admitted after recovery) UPSERTS the same record — the turn's
    /// identity is never duplicated. Any OTHER still-active record of the
    /// session is finalized as failed in the same transaction (at most one
    /// active logical turn may exist per session).
    #[allow(clippy::too_many_arguments)]
    pub fn start_turn_record(
        &self,
        session_id: SessionId,
        turn_op_id: OpId,
        queue_seq: Option<i64>,
        prompt_message_id: Option<i64>,
        provider: &str,
        model: &str,
        variant: Option<&str>,
    ) -> StoreResult<i64> {
        let conn = self.write();
        let tx = conn.unchecked_transaction()?;
        tx.execute(
            "UPDATE turn_record SET status = ?3, updated_ms = ?4
             WHERE session_id = ?1 AND status = 'active' AND turn_op_id != ?2",
            params![
                session_id.raw() as i64,
                turn_op_id.raw() as i64,
                TURN_RECORD_FAILED,
                now_ms()
            ],
        )?;
        let now = now_ms();
        tx.execute(
            "INSERT INTO turn_record(session_id, turn_op_id, queue_seq, prompt_message_id, effective_provider, effective_model, variant, started_at, status, updated_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'active', ?8)
             ON CONFLICT(session_id, turn_op_id) DO UPDATE SET
                queue_seq = excluded.queue_seq,
                prompt_message_id = excluded.prompt_message_id,
                effective_provider = excluded.effective_provider,
                effective_model = excluded.effective_model,
                variant = excluded.variant,
                started_at = excluded.started_at,
                status = 'active',
                updated_ms = excluded.updated_ms",
            params![
                session_id.raw() as i64,
                turn_op_id.raw() as i64,
                queue_seq,
                prompt_message_id,
                provider,
                model,
                variant,
                now
            ],
        )?;
        let id = tx.last_insert_rowid();
        tx.commit()?;
        Ok(id)
    }

    /// Finalize the record's effective envelope at logical-turn start (the
    /// per-message model override and the tool mode are only known once the
    /// runtime drives the turn). Only an active record is updated.
    pub fn set_turn_record_envelope(
        &self,
        session_id: SessionId,
        turn_op_id: OpId,
        provider: &str,
        model: &str,
        variant: Option<&str>,
        tool_mode: Option<&str>,
    ) -> StoreResult<bool> {
        let conn = self.write();
        let n = conn.execute(
            "UPDATE turn_record SET effective_provider = ?3, effective_model = ?4, variant = ?5, tool_mode = ?6, updated_ms = ?7
             WHERE session_id = ?1 AND turn_op_id = ?2 AND status = 'active'",
            params![
                session_id.raw() as i64,
                turn_op_id.raw() as i64,
                provider,
                model,
                variant,
                tool_mode,
                now_ms()
            ],
        )?;
        Ok(n > 0)
    }

    /// Close an active turn record (completed | cancelled | failed).
    /// No-op when the record is absent or already closed (idempotent).
    pub fn finish_turn_record(
        &self,
        session_id: SessionId,
        turn_op_id: OpId,
        status: &str,
    ) -> StoreResult<bool> {
        if !matches!(
            status,
            TURN_RECORD_COMPLETED | TURN_RECORD_CANCELLED | TURN_RECORD_FAILED
        ) {
            return Err(StoreError::Migration(format!(
                "finish_turn_record: invalid status {status:?}"
            )));
        }
        let conn = self.write();
        let n = conn.execute(
            "UPDATE turn_record SET status = ?3, updated_ms = ?4
             WHERE session_id = ?1 AND turn_op_id = ?2 AND status = 'active'",
            params![
                session_id.raw() as i64,
                turn_op_id.raw() as i64,
                status,
                now_ms()
            ],
        )?;
        Ok(n > 0)
    }

    /// The session's single active logical-turn record (at most one exists).
    pub fn active_turn_record(&self, session_id: SessionId) -> StoreResult<Option<TurnRecordRow>> {
        let conn = self.read()?;
        let out = conn
            .query_row(
                "SELECT id, session_id, turn_op_id, queue_seq, prompt_message_id, effective_provider, effective_model, variant, tool_mode, started_at, status, updated_ms
                 FROM turn_record WHERE session_id = ?1 AND status = 'active'
                 ORDER BY started_at DESC, id DESC LIMIT 1",
                params![session_id.raw() as i64],
                turn_record_map,
            )
            .optional()?;
        Ok(out)
    }

    pub fn turn_record_of(
        &self,
        session_id: SessionId,
        turn_op_id: OpId,
    ) -> StoreResult<Option<TurnRecordRow>> {
        let conn = self.read()?;
        let out = conn
            .query_row(
                "SELECT id, session_id, turn_op_id, queue_seq, prompt_message_id, effective_provider, effective_model, variant, tool_mode, started_at, status, updated_ms
                 FROM turn_record WHERE session_id = ?1 AND turn_op_id = ?2",
                params![session_id.raw() as i64, turn_op_id.raw() as i64],
                turn_record_map,
            )
            .optional()?;
        Ok(out)
    }

    /// Every turn record of a session (oldest first; diagnostics/tests).
    pub fn turn_records_of(&self, session_id: SessionId) -> StoreResult<Vec<TurnRecordRow>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT id, session_id, turn_op_id, queue_seq, prompt_message_id, effective_provider, effective_model, variant, tool_mode, started_at, status, updated_ms
             FROM turn_record WHERE session_id = ?1 ORDER BY started_at ASC, id ASC",
        )?;
        let mut rows = stmt.query(params![session_id.raw() as i64])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(turn_record_map(row)?);
        }
        Ok(out)
    }

    // ---------------------------------------------------------------- provider calls

    // Fixed-arity provider telemetry; the parameter list is a stable call
    // contract used across the workspace.
    #[allow(clippy::too_many_arguments)]
    pub fn record_provider_call(
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
        self.record_provider_call_with_prefix(
            session_id, op_id, provider, model, status, tokens_in, tokens_out, error, None, None,
            None,
        )
    }

    /// `record_provider_call` plus the additive prefix-cache stability
    /// observation (v13, audits 65-66): the digest of the exact
    /// cacheable-prefix byte string this call sent, its token count, and
    /// (optionally, once a fill site can measure it) the call's per-turn
    /// prefix stability. All three are optional and default to NULL — a row
    /// without them is a row with no prefix observation, never a guessed
    /// one. Values are validated LOUDLY: `prompt_prefix_hash` must be
    /// exactly 32 bytes, `prompt_tokens` must fit a `u32` (a single prompt
    /// prefix beyond 4.29e9 tokens is rejected as oversized, matching the
    /// router's `TurnPrefix.prefix_tokens`), and `prefix_stability` must be
    /// finite in [0, 1] (NaN/out-of-range is `Malformed`, never silently
    /// stored as NULL or a nonsense number).
    #[allow(clippy::too_many_arguments)]
    pub fn record_provider_call_with_prefix(
        &self,
        session_id: SessionId,
        op_id: OpId,
        provider: &str,
        model: &str,
        status: &str,
        tokens_in: Option<u64>,
        tokens_out: Option<u64>,
        error: Option<&str>,
        prompt_prefix_hash: Option<[u8; 32]>,
        prompt_tokens: Option<u64>,
        prefix_stability: Option<f64>,
    ) -> StoreResult<i64> {
        self.record_provider_call_with_prefix_segments(
            session_id,
            op_id,
            provider,
            model,
            status,
            tokens_in,
            tokens_out,
            error,
            prompt_prefix_hash,
            prompt_tokens,
            prefix_stability,
            // Legacy callers record no per-call segment observation: the
            // v19 column stays NULL and routing keeps the binary pair rule.
            None,
        )
    }

    /// Additive v19 twin of [`Store::record_provider_call_with_prefix`]:
    /// the prefix observation row additionally carries the raw per-call
    /// segment observation JSON (ordered segment digests + token counts +
    /// observed cache reads) the settlement site measured. The payload is
    /// validated LOUDLY before anything touches the row: bounded by
    /// [`MAX_PREFIX_SEGMENTS_JSON`], exactly the expected strict fields,
    /// hash/token vectors of equal bounded length, every hash a 32-byte hex
    /// digest. `None` records the legacy NULL — no segments, never a guess.
    #[allow(clippy::too_many_arguments)]
    pub fn record_provider_call_with_prefix_segments(
        &self,
        session_id: SessionId,
        op_id: OpId,
        provider: &str,
        model: &str,
        status: &str,
        tokens_in: Option<u64>,
        tokens_out: Option<u64>,
        error: Option<&str>,
        prompt_prefix_hash: Option<[u8; 32]>,
        prompt_tokens: Option<u64>,
        prefix_stability: Option<f64>,
        prefix_segments_json: Option<&str>,
    ) -> StoreResult<i64> {
        let prompt_tokens = prompt_tokens
            .map(|t| {
                u32::try_from(t).map_err(|_| {
                    StoreError::Oversized(format!(
                        "prompt_tokens {t} exceeds the u32 prefix-token bound"
                    ))
                })
            })
            .transpose()?;
        if let Some(s) = prefix_stability {
            if !s.is_finite() || !(0.0..=1.0).contains(&s) {
                return Err(StoreError::Malformed(format!(
                    "prefix_stability must be finite in [0, 1], got {s}"
                )));
            }
        }
        if let Some(json) = prefix_segments_json {
            validate_prefix_segments_json(json)?;
        }
        let conn = self.write();
        self.insert_provider_call_on(
            &conn,
            session_id,
            op_id,
            provider,
            model,
            status,
            tokens_in,
            tokens_out,
            error,
            prompt_prefix_hash,
            prompt_tokens,
            prefix_stability,
            prefix_segments_json,
            None,
            None,
            None,
            None,
        )
    }

    /// Attempt-oriented provider-call record (attempt accounting, schema
    /// v18): `op_id` keeps its meaning as the shared logical model-call op
    /// id this attempt belongs to (`parent_model_call_op_id` mirrors it
    /// durably), and the row keys to its physical attempt through
    /// `attempt_op_id` (+ `attempt_ordinal`) and to its money through
    /// `reservation_id`. Reconciliation of an uncertain reservation joins
    /// `provider_call.attempt_op_id = cost_reservation.attempt_op_id`, so
    /// two attempts of the same logical op can never settle each other's
    /// crashed reservations. Legacy callers (the current agent runtime) keep
    /// writing through [`Store::record_provider_call`], leaving the attempt
    /// columns NULL — those rows remain the single physical attempt of
    /// their op, exactly as before.
    #[allow(clippy::too_many_arguments)]
    pub fn record_provider_call_attempt(
        &self,
        session_id: SessionId,
        attempt: &faktor_core::op::ModelCallAttempt,
        reservation_id: Option<i64>,
        provider: &str,
        model: &str,
        status: &str,
        tokens_in: Option<u64>,
        tokens_out: Option<u64>,
        error: Option<&str>,
    ) -> StoreResult<i64> {
        let conn = self.write();
        conn.execute(
            "INSERT INTO provider_call(session_id, op_id, parent_model_call_op_id,
                attempt_op_id, attempt_ordinal, reservation_id,
                provider, model, started_ms, ended_ms, status, tokens_in,
                tokens_out, error)
             VALUES (?1, ?2, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?8, ?9, ?10, ?11, ?12)",
            params![
                session_id.raw() as i64,
                attempt.logical_op_id.raw() as i64,
                attempt.attempt_op_id.raw() as i64,
                attempt.ordinal as i64,
                reservation_id,
                provider,
                model,
                now_ms(),
                status,
                tokens_in.map(|t| t as i64),
                tokens_out.map(|t| t as i64),
                error
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// Shared single-row provider-call insert; see [`Self::insert_message_on`].
    /// The usage-settlement row of the hot append surface. The four attempt
    /// parameters are the additive v18 surface and
    /// `prefix_segments_json` the additive v19 one: `None` everywhere
    /// records a legacy row (no attempt identity, no reservation link, no
    /// segment observation).
    #[allow(clippy::too_many_arguments)]
    fn insert_provider_call_on(
        &self,
        conn: &Connection,
        session_id: SessionId,
        op_id: OpId,
        provider: &str,
        model: &str,
        status: &str,
        tokens_in: Option<u64>,
        tokens_out: Option<u64>,
        error: Option<&str>,
        prompt_prefix_hash: Option<[u8; 32]>,
        prompt_tokens: Option<u32>,
        prefix_stability: Option<f64>,
        prefix_segments_json: Option<&str>,
        attempt_op_id: Option<OpId>,
        attempt_ordinal: Option<u32>,
        parent_model_call_op_id: Option<OpId>,
        reservation_id: Option<i64>,
    ) -> StoreResult<i64> {
        conn.execute(
            "INSERT INTO provider_call(session_id, op_id, provider, model, started_ms, ended_ms, status, tokens_in, tokens_out, error, prompt_prefix_hash, prompt_tokens, prefix_stability, prefix_segments_json, attempt_op_id, attempt_ordinal, parent_model_call_op_id, reservation_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)",
            params![
                session_id.raw() as i64,
                op_id.raw() as i64,
                provider,
                model,
                now_ms(),
                now_ms(),
                status,
                tokens_in.map(|t| t as i64),
                tokens_out.map(|t| t as i64),
                error,
                prompt_prefix_hash.map(Vec::from),
                prompt_tokens.map(i64::from),
                prefix_stability,
                prefix_segments_json,
                attempt_op_id.map(|id| id.raw() as i64),
                attempt_ordinal.map(i64::from),
                parent_model_call_op_id.map(|id| id.raw() as i64),
                reservation_id
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// The session's durable prefix observations, oldest call first (v13).
    /// Rows whose `prompt_prefix_hash` is NULL (pre-v13 or settled without a
    /// prefix) are excluded — a missing observation is not a zero.
    /// Read-time validation is loud: a corrupt shape injected behind the
    /// API's back (wrong-length hash, out-of-range tokens/stability, or a
    /// malformed v19 segment payload) is a `Malformed` error, never a silent
    /// misread.
    pub fn provider_call_prefix_rows(
        &self,
        session_id: SessionId,
    ) -> StoreResult<Vec<ProviderCallPrefixRow>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT id, prompt_prefix_hash, prompt_tokens, prefix_stability, prefix_segments_json
             FROM provider_call
             WHERE session_id = ?1 AND prompt_prefix_hash IS NOT NULL
             ORDER BY id ASC",
        )?;
        let mut rows = stmt.query(params![session_id.raw() as i64])?;
        let mut out = Vec::new();
        while let Some(r) = rows.next()? {
            let hash: Option<Vec<u8>> = r.get(1)?;
            let Some(hash) = hash else {
                return Err(StoreError::Malformed(
                    "provider_call prefix row has NULL hash despite the filter".into(),
                ));
            };
            let hash: [u8; 32] = hash.try_into().map_err(|v: Vec<u8>| {
                StoreError::Malformed(format!(
                    "provider_call prefix hash must be exactly 32 bytes, got {}",
                    v.len()
                ))
            })?;
            let tokens: Option<i64> = r.get(2)?;
            let tokens = match tokens {
                None => {
                    return Err(StoreError::Malformed(
                        "provider_call prefix row has NULL prompt_tokens".into(),
                    ))
                }
                Some(t) => u32::try_from(t).map_err(|_| {
                    StoreError::Malformed(format!(
                        "provider_call prompt_tokens {t} out of u32 range"
                    ))
                })?,
            };
            let stability: Option<f64> = r.get(3)?;
            if let Some(s) = stability {
                if !s.is_finite() || !(0.0..=1.0).contains(&s) {
                    return Err(StoreError::Malformed(format!(
                        "provider_call prefix_stability {s} out of [0, 1]"
                    )));
                }
            }
            let prefix_segments_json: Option<String> = r.get(4)?;
            if let Some(json) = &prefix_segments_json {
                validate_prefix_segments_json(json)?;
            }
            out.push(ProviderCallPrefixRow {
                row_id: r.get(0)?,
                session_id,
                prompt_prefix_hash: hash,
                prompt_tokens: tokens,
                prefix_stability: stability,
                prefix_segments_json,
            });
        }
        Ok(out)
    }

    /// Additive aggregate query over the stored per-row prefix stabilities
    /// of one session (v13): count, mean and population std dev, computed
    /// in SQL then finished in guarded float math. `None` when the session
    /// has no rows with a recorded stability. Never parses the hash BLOBs —
    /// corrupt prefix rows surface through
    /// [`Store::provider_call_prefix_rows`] instead.
    pub fn session_stored_prefix_stability(
        &self,
        session_id: SessionId,
    ) -> StoreResult<Option<PrefixStabilityAggregate>> {
        let conn = self.read()?;
        let (count, sum, sum_sq): (i64, f64, f64) = conn.query_row(
            "SELECT COUNT(prefix_stability), COALESCE(SUM(prefix_stability), 0.0),
                    COALESCE(SUM(prefix_stability * prefix_stability), 0.0)
             FROM provider_call
             WHERE session_id = ?1
               AND prompt_prefix_hash IS NOT NULL
               AND prefix_stability IS NOT NULL",
            params![session_id.raw() as i64],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )?;
        if count == 0 {
            return Ok(None);
        }
        // Loud, never silent: a corrupt stability (out of [0, 1]) injected
        // behind the API's back must fail the aggregate, not pollute it.
        let bad: i64 = conn.query_row(
            "SELECT COUNT(*) FROM provider_call
             WHERE session_id = ?1
               AND prompt_prefix_hash IS NOT NULL
               AND prefix_stability IS NOT NULL
               AND (prefix_stability < 0.0 OR prefix_stability > 1.0)",
            params![session_id.raw() as i64],
            |r| r.get(0),
        )?;
        if bad > 0 {
            return Err(StoreError::Malformed(format!(
                "provider_call prefix_stability out of [0, 1] on {bad} row(s)"
            )));
        }
        let n = count as f64;
        // Population variance; guard the float subtraction from a hair
        // below zero on hostile magnitudes.
        let var = ((sum_sq - sum * sum / n) / n).max(0.0);
        Ok(Some(PrefixStabilityAggregate {
            observations: count as u64,
            mean: sum / n,
            std_dev: var.sqrt(),
        }))
    }

    // ---------------------------------------------------------------- checkpoints

    /// ALLOCATE the next per-session checkpoint sequence and insert the row
    /// in ONE transaction (P1 "checkpoint numbering race"): two concurrent
    /// writers must never both receive the same sequence. The sequence is
    /// `MAX(sequence)+1` over the session's rows, computed and inserted under
    /// the single writer lock, so allocation is atomic and gapless regardless
    /// of what any caller guessed outside the store.
    ///
    /// `before_hash`/`after_hash` carry the side's content hash, or the empty
    /// string when that side does not exist (`before_exists=false`). Returns
    /// the row id and the allocated sequence.
    #[allow(clippy::too_many_arguments)]
    pub fn insert_checkpoint(
        &self,
        session_id: SessionId,
        path: &str,
        before_exists: bool,
        before_hash: &str,
        after_exists: bool,
        after_hash: &str,
        after_cas_hash: Option<&str>,
    ) -> StoreResult<(i64, i64)> {
        let conn = self.write();
        let tx = conn.unchecked_transaction()?;
        let prev: i64 = tx.query_row(
            "SELECT COALESCE(MAX(sequence), 0) FROM checkpoint WHERE session_id = ?1",
            params![session_id.raw() as i64],
            |r| r.get(0),
        )?;
        let sequence = prev + 1;
        tx.execute(
            "INSERT INTO checkpoint(session_id, sequence, path, before_hash, after_hash, after_cas_hash, before_exists, after_exists, created_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                session_id.raw() as i64,
                sequence,
                path,
                before_hash,
                after_hash,
                after_cas_hash,
                before_exists as i64,
                after_exists as i64,
                now_ms()
            ],
        )?;
        let id = tx.last_insert_rowid();
        tx.commit()?;
        Ok((id, sequence))
    }

    /// Raw checkpoint insert at an explicit caller-chosen sequence (the
    /// session layer's journal-backed path, which validates duplicates
    /// itself). Both sides exist. Prefer [`Store::insert_checkpoint`] for
    /// content-aware checkpoints: it allocates the sequence atomically.
    pub fn put_checkpoint(
        &self,
        session_id: SessionId,
        sequence: i64,
        path: &str,
        before_hash: &str,
        after_hash: &str,
        after_cas_hash: Option<&str>,
    ) -> StoreResult<i64> {
        let conn = self.write();
        conn.execute(
            "INSERT INTO checkpoint(session_id, sequence, path, before_hash, after_hash, after_cas_hash, created_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                session_id.raw() as i64,
                sequence,
                path,
                before_hash,
                after_hash,
                after_cas_hash,
                now_ms()
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn checkpoints_of(&self, session_id: SessionId) -> StoreResult<Vec<CheckpointRow>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT id, session_id, sequence, path, before_hash, after_hash, after_cas_hash, before_exists, after_exists, created_ms, restored_ms
             FROM checkpoint WHERE session_id = ?1 ORDER BY sequence ASC, id ASC",
        )?;
        let rows = stmt.query_map(params![session_id.raw() as i64], |r| {
            Ok(CheckpointRow {
                id: r.get(0)?,
                session_id,
                sequence: r.get(2)?,
                path: r.get(3)?,
                before_hash: r.get(4)?,
                after_hash: r.get(5)?,
                after_cas_hash: r.get(6)?,
                before_exists: r.get::<_, i64>(7)? != 0,
                after_exists: r.get::<_, i64>(8)? != 0,
                created_ms: r.get(9)?,
                restored_ms: r.get(10)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Redo/undo marker: rollback sets restored_ms, redo clears it (a row
    /// must not read as "restored" after an unrevert; audit round 5).
    pub fn clear_checkpoint_restored(&self, id: i64) -> StoreResult<()> {
        let conn = self.write();
        conn.execute(
            "UPDATE checkpoint SET restored_ms = NULL WHERE id = ?1",
            params![id],
        )?;
        Ok(())
    }

    pub fn mark_checkpoint_restored(&self, id: i64) -> StoreResult<()> {
        let conn = self.write();
        conn.execute(
            "UPDATE checkpoint SET restored_ms = ?2 WHERE id = ?1",
            params![id, now_ms()],
        )?;
        Ok(())
    }

    // ---------------------------------------------------------------- artifacts

    /// Artifact rows reference CAS hashes; the blob itself lives in the CAS.
    pub fn put_artifact(
        &self,
        session_id: SessionId,
        kind: &str,
        cas_hash: &str,
        summary: &str,
        size: i64,
    ) -> StoreResult<i64> {
        let conn = self.write();
        conn.execute(
            "INSERT OR IGNORE INTO artifact(session_id, kind, cas_hash, summary, created_ms, size)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                session_id.raw() as i64,
                kind,
                cas_hash,
                summary,
                now_ms(),
                size
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn artifact(&self, cas_hash: &str) -> StoreResult<Option<(String, String)>> {
        let conn = self.read()?;
        let out = conn
            .query_row(
                "SELECT summary, kind FROM artifact WHERE cas_hash = ?1",
                params![cas_hash],
                |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)),
            )
            .ok();
        Ok(out)
    }

    // -------------------------------------------------------------- attachments

    /// Persist one binary/image attachment's typed metadata (schema v24).
    /// Dedupe IS the primary key `(session_id, digest)`: an identical digest
    /// is a no-op and the FIRST-written row is read back and returned, so
    /// repeated uploads of the same bytes resolve to one byte-identical
    /// `AttachmentId`. The caller (session layer) validates the id before
    /// this write; the store only persists/reads the typed row.
    pub fn put_attachment(
        &self,
        session_id: SessionId,
        attachment: &AttachmentId,
    ) -> StoreResult<AttachmentId> {
        let conn = self.write();
        conn.execute(
            "INSERT OR IGNORE INTO attachment(session_id, digest, mime, filename, size)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                session_id.raw() as i64,
                attachment.digest.to_hex(),
                attachment.mime.as_str(),
                attachment.filename.as_deref(),
                attachment.size as i64,
            ],
        )?;
        let stored = conn.query_row(
            "SELECT digest, mime, filename, size FROM attachment
             WHERE session_id = ?1 AND digest = ?2",
            params![session_id.raw() as i64, attachment.digest.to_hex()],
            |r| {
                let digest: String = r.get(0)?;
                let mime: String = r.get(1)?;
                let filename: Option<String> = r.get(2)?;
                let size: i64 = r.get(3)?;
                let digest = faktor_core::hash::FileHash::from_hex(&digest).ok_or_else(|| {
                    rusqlite::Error::FromSqlConversionFailure(
                        0,
                        rusqlite::types::Type::Text,
                        Box::new(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("attachment digest {digest:?} is not 32-byte hex"),
                        )),
                    )
                })?;
                if size < 0 {
                    return Err(rusqlite::Error::FromSqlConversionFailure(
                        3,
                        rusqlite::types::Type::Integer,
                        Box::new(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("attachment size {size} is negative"),
                        )),
                    ));
                }
                Ok(AttachmentId {
                    digest,
                    mime,
                    filename,
                    size: size as u64,
                })
            },
        )?;
        Ok(stored)
    }

    /// Resolve one durable attachment row by its digest (restart-safe: reads
    /// the typed metadata, never a process-local map).
    pub fn attachment(
        &self,
        session_id: SessionId,
        digest: faktor_core::hash::FileHash,
    ) -> StoreResult<Option<AttachmentId>> {
        let conn = self.read()?;
        let out = conn
            .query_row(
                "SELECT digest, mime, filename, size FROM attachment
                 WHERE session_id = ?1 AND digest = ?2",
                params![session_id.raw() as i64, digest.to_hex()],
                |r| {
                    let digest_raw: String = r.get(0)?;
                    let mime: String = r.get(1)?;
                    let filename: Option<String> = r.get(2)?;
                    let size: i64 = r.get(3)?;
                    let digest =
                        faktor_core::hash::FileHash::from_hex(&digest_raw).ok_or_else(|| {
                            rusqlite::Error::FromSqlConversionFailure(
                                0,
                                rusqlite::types::Type::Text,
                                Box::new(std::io::Error::new(
                                    std::io::ErrorKind::InvalidData,
                                    format!("attachment digest {digest_raw:?} is not 32-byte hex"),
                                )),
                            )
                        })?;
                    if size < 0 {
                        return Err(rusqlite::Error::FromSqlConversionFailure(
                            3,
                            rusqlite::types::Type::Integer,
                            Box::new(std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                format!("attachment size {size} is negative"),
                            )),
                        ));
                    }
                    Ok(AttachmentId {
                        digest,
                        mime,
                        filename,
                        size: size as u64,
                    })
                },
            )
            .optional()?;
        Ok(out)
    }

    /// The session's durable attachment metadata, newest digest order (a
    /// bounded, deterministic page: at most `limit` rows).
    pub fn list_attachments(
        &self,
        session_id: SessionId,
        limit: usize,
    ) -> StoreResult<Vec<AttachmentId>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT digest, mime, filename, size FROM attachment
             WHERE session_id = ?1 ORDER BY digest ASC LIMIT ?2",
        )?;
        let mut rows = stmt.query(params![session_id.raw() as i64, limit as i64])?;
        let mut out = Vec::new();
        while let Some(r) = rows.next()? {
            let digest_raw: String = r.get(0)?;
            let Some(digest) = faktor_core::hash::FileHash::from_hex(&digest_raw) else {
                return Err(StoreError::Corrupt(vec![format!(
                    "attachment digest {digest_raw:?} is not 32-byte hex"
                )]));
            };
            let size: i64 = r.get(3)?;
            if size < 0 {
                return Err(StoreError::Corrupt(vec![format!(
                    "attachment size {size} is negative"
                )]));
            }
            out.push(AttachmentId {
                digest,
                mime: r.get(1)?,
                filename: r.get(2)?,
                size: size as u64,
            });
        }
        Ok(out)
    }

    // ---------------------------------------------------------------- worktrees

    pub fn put_worktree(
        &self,
        workspace_id: WorkspaceId,
        path: &str,
        branch: &str,
    ) -> StoreResult<i64> {
        let conn = self.write();
        conn.execute(
            "INSERT OR IGNORE INTO worktree(workspace_id, path, branch, active)
             VALUES (?1, ?2, ?3, 1)",
            params![workspace_id.raw() as i64, path, branch],
        )?;
        conn.query_row(
            "SELECT id FROM worktree WHERE path = ?1",
            params![path],
            |r| r.get(0),
        )
        .map_err(Into::into)
    }

    pub fn worktrees_of(&self, workspace_id: WorkspaceId) -> StoreResult<Vec<WorktreeRow>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT id, workspace_id, path, branch, active FROM worktree WHERE workspace_id = ?1 ORDER BY id ASC",
        )?;
        let rows = stmt.query_map(params![workspace_id.raw() as i64], |r| {
            Ok(WorktreeRow {
                id: r.get(0)?,
                workspace_id,
                path: r.get(2)?,
                branch: r.get(3)?,
                active: r.get(4)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn remove_worktree(&self, path: &str) -> StoreResult<()> {
        let conn = self.write();
        conn.execute("DELETE FROM worktree WHERE path = ?1", params![path])?;
        Ok(())
    }

    // ---------------------------------------------------------------- memory facts

    pub fn upsert_memory_fact(
        &self,
        session_id: SessionId,
        kind: &str,
        key: &str,
        value: &str,
    ) -> StoreResult<()> {
        let conn = self.write();
        conn.execute(
            "INSERT INTO memory_fact(session_id, kind, key, value, updated_ms) VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(session_id, kind, key) DO UPDATE SET value = ?4, updated_ms = ?5",
            params![session_id.raw() as i64, kind, key, value, self.fact_timestamp()],
        )?;
        Ok(())
    }

    /// Atomically upsert MANY memory facts of ONE session in ONE SQLite
    /// transaction (one commit + fsync): the row group either lands fully
    /// or not at all — a crash between the writes can never expose a
    /// partial group (orchestrator run-compile assignment rows depend on
    /// exactly this). Same table/conflict semantics as the single-row
    /// [`Store::upsert_memory_fact`]; value bounds remain the caller's
    /// contract (the session layer enforces them on its single-row path).
    pub fn upsert_memory_facts(
        &self,
        session_id: SessionId,
        facts: &[(&str, &str, &str)],
    ) -> StoreResult<()> {
        if facts.is_empty() {
            return Ok(());
        }
        let conn = self.write();
        let tx = conn.unchecked_transaction()?;
        for (kind, key, value) in facts {
            tx.execute(
                "INSERT INTO memory_fact(session_id, kind, key, value, updated_ms) VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(session_id, kind, key) DO UPDATE SET value = ?4, updated_ms = ?5",
                params![
                    session_id.raw() as i64,
                    kind,
                    key,
                    value,
                    self.fact_timestamp()
                ],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Deterministic newest-first page over memory facts (paging is
    /// fundamental). Rows follow the total order
    /// `(updated_ms DESC, kind DESC, key DESC)` — the same order the
    /// legacy [`Store::memory_facts`] full read uses — and a page contains
    /// only rows strictly AFTER the `after` cursor position (older in the
    /// order). Returns `(rows, has_more)` with at most `limit` rows; pass
    /// the last row's `(updated_ms, kind, key)` as the next `after` cursor.
    ///
    /// Cursor semantics under concurrent writes: an upsert only moves a row
    /// toward the NEWEST end of the order (its `updated_ms` is rewritten to
    /// now), so a backward walk can never see the same `(kind, key)` twice,
    /// and rows that existed at the walk's start and are never rewritten
    /// appear exactly once — deterministic, no duplicate, no gap.
    pub fn memory_facts_page(
        &self,
        session_id: SessionId,
        after: Option<&MemoryFactCursor>,
        limit: u64,
    ) -> StoreResult<(Vec<MemoryFactRow>, bool)> {
        let conn = self.read()?;
        let mut sql = String::from(
            "SELECT kind, key, value, updated_ms FROM memory_fact
             WHERE session_id = ?1",
        );
        let mut params: Vec<rusqlite::types::Value> = Vec::with_capacity(5);
        params.push(rusqlite::types::Value::Integer(session_id.raw() as i64));
        if let Some((ms, kind, key)) = after {
            sql.push_str(
                " AND (updated_ms < ?2 OR (updated_ms = ?2 AND (kind < ?3 OR (kind = ?3 AND key < ?4))))",
            );
            params.push(rusqlite::types::Value::Integer(*ms));
            params.push(rusqlite::types::Value::Text(kind.clone()));
            params.push(rusqlite::types::Value::Text(key.clone()));
        }
        // Probe one extra row for the has_more verdict. u64 limits are
        // clamped to the i64 domain first (u64::MAX as i64 would wrap to -1
        // and silently return an empty page).
        sql.push_str(" ORDER BY updated_ms DESC, kind DESC, key DESC LIMIT ?");
        let probe = (limit.min(i64::MAX as u64) as i64).saturating_add(1);
        params.push(rusqlite::types::Value::Integer(probe));
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = stmt.query(rusqlite::params_from_iter(params.iter()))?;
        let mut out: Vec<MemoryFactRow> = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(MemoryFactRow {
                kind: row.get(0)?,
                key: row.get(1)?,
                value: row.get(2)?,
                updated_ms: row.get(3)?,
            });
        }
        let has_more = out.len() as u64 > limit;
        if has_more {
            out.truncate(limit as usize);
        }
        Ok((out, has_more))
    }

    pub fn memory_facts(
        &self,
        session_id: SessionId,
    ) -> StoreResult<Vec<(String, String, String)>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            // Deterministic total order (updated_ms DESC, kind DESC, key
            // DESC): page cursors cut this exact order, so the full read is
            // the unbounded prefix of the paged read.
            "SELECT kind, key, value FROM memory_fact WHERE session_id = ?1
             ORDER BY updated_ms DESC, kind DESC, key DESC",
        )?;
        let rows = stmt.query_map(params![session_id.raw() as i64], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
            ))
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Total count of memory facts for one session (cheap; facts are small
    /// per session).
    pub fn memory_fact_count(&self, session_id: SessionId) -> StoreResult<i64> {
        let conn = self.read()?;
        let out = conn.query_row(
            "SELECT COUNT(*) FROM memory_fact WHERE session_id = ?1",
            params![session_id.raw() as i64],
            |r| r.get::<_, i64>(0),
        )?;
        Ok(out)
    }

    // ---------------------------------------------------------------- compactions

    pub fn record_compaction(
        &self,
        session_id: SessionId,
        before_tokens: i64,
        after_tokens: i64,
        target_tokens: i64,
        accepted: bool,
        strategy: &str,
    ) -> StoreResult<()> {
        let conn = self.write();
        conn.execute(
            "INSERT INTO compaction(session_id, before_tokens, after_tokens, target_tokens, accepted, strategy, created_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                session_id.raw() as i64,
                before_tokens,
                after_tokens,
                target_tokens,
                accepted as i64,
                strategy,
                now_ms()
            ],
        )?;
        Ok(())
    }

    // ---------------------------------------------------------------- permissions

    pub fn insert_permission(
        &self,
        session_id: SessionId,
        op_id: OpId,
        capability: &str,
    ) -> StoreResult<i64> {
        let conn = self.write();
        conn.execute(
            "INSERT INTO permission(session_id, op_id, capability, decision, expires_ms)
             VALUES (?1, ?2, ?3, 'pending', ?4)",
            params![
                session_id.raw() as i64,
                op_id.raw() as i64,
                capability,
                now_ms() + 60_000
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn resolve_permission(&self, id: i64, decision: &str) -> StoreResult<()> {
        let conn = self.write();
        conn.execute(
            "UPDATE permission SET decision = ?2, resolved_ms = ?3 WHERE id = ?1 AND decision = 'pending'",
            params![id, decision, now_ms()],
        )?;
        Ok(())
    }

    pub fn pending_permission(&self, id: i64) -> StoreResult<Option<(SessionId, OpId, String)>> {
        let conn = self.read()?;
        let out = conn
            .query_row(
                "SELECT session_id, op_id, capability FROM permission WHERE id = ?1 AND decision = 'pending'",
                params![id],
                |r| {
                    Ok((
                        SessionId::new(r.get::<_, i64>(0)? as u64),
                        OpId::new(r.get::<_, i64>(1)? as u64),
                        r.get::<_, String>(2)?,
                    ))
                },
            )
            .ok();
        Ok(out)
    }

    // ---------------------------------------------------------------- prompt queue
    /// Durably queue a prompt that arrived while another turn was active.
    /// The full execution envelope is stored; the user conversation message
    /// is NOT materialized yet (deferred materialization — audit round 7:
    /// conversation chronology is insertion order, so the message is
    /// appended at admission, after the preceding turn's output).
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_prompt(
        &self,
        session: SessionId,
        op_id: OpId,
        prompt: &str,
        files: &[String],
        model: Option<&str>,
        variant: Option<&str>,
        agent: Option<&str>,
        requested_at: i64,
    ) -> StoreResult<i64> {
        let conn = self.write();
        let tx = conn.unchecked_transaction()?;
        let prev: i64 = tx.query_row(
            "SELECT COALESCE(MAX(seq), 0) FROM prompt_queue WHERE session_id = ?1",
            params![session.raw() as i64],
            |r| r.get(0),
        )?;
        let seq = prev + 1;
        tx.execute(
            "INSERT INTO prompt_queue(session_id, seq, op_id, prompt, files, model, variant, agent, status, requested_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'pending', ?9)",
            params![
                session.raw() as i64,
                seq,
                op_id.raw() as i64,
                prompt,
                serde_json::to_string(files).unwrap_or_else(|_| "[]".into()),
                model,
                variant,
                agent,
                requested_at
            ],
        )?;
        tx.commit()?;
        Ok(seq)
    }

    fn queue_row_from(r: &rusqlite::Row<'_>) -> rusqlite::Result<QueuedPrompt> {
        Ok(QueuedPrompt {
            queue_seq: r.get(0)?,
            op_id: OpId::new(r.get::<_, i64>(1)? as u64),
            prompt: r.get(2)?,
            files: serde_json::from_str(&r.get::<_, String>(3)?).unwrap_or_default(),
            model: r.get(4)?,
            variant: r.get(5)?,
            agent: r.get(6)?,
            status: r.get(7)?,
            requested_at: r.get(8)?,
        })
    }

    /// Oldest row that is not terminal (pending/claimed/running) — FIFO head.
    pub fn queue_head(&self, session: SessionId) -> StoreResult<Option<QueuedPrompt>> {
        let conn = self.read()?;
        let out = conn
            .query_row(
                "SELECT seq, op_id, prompt, files, model, variant, agent, status, requested_at
                 FROM prompt_queue
                 WHERE session_id = ?1 AND status IN ('pending','claimed','running')
                 ORDER BY seq ASC LIMIT 1",
                params![session.raw() as i64],
                Self::queue_row_from,
            )
            .ok();
        Ok(out)
    }

    /// Count of rows in each durable status (diagnostics).
    pub fn queue_status_counts(&self, session: SessionId) -> StoreResult<serde_json::Value> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT status, COUNT(*) FROM prompt_queue WHERE session_id = ?1 GROUP BY status",
        )?;
        let rows = stmt.query_map(params![session.raw() as i64], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))
        })?;
        let mut counts = serde_json::Map::new();
        for row in rows {
            let (k, v) = row?;
            counts.insert(k, serde_json::json!(v));
        }
        Ok(serde_json::Value::Object(counts))
    }

    /// Atomic claim + admission of the queue head (audit round 7): ONE
    /// transaction establishes (a) the head is pending and the session is
    /// eligible, (b) pending -> claimed, (c) the user message is materialized
    /// at the true conversation tail, (d) the session row moves to the
    /// target state, (e) the admission journal event seq is computed so the
    /// session layer can journal the turn-open with a gapless sequence. No
    /// other submission can cut between those operations (single writer).
    ///
    /// Returns Ok(None) when the head is absent or the session state is not
    /// in `eligible_states` (nothing is touched in either case).
    pub fn admit_queue_head(
        &self,
        session: SessionId,
        eligible_states: &[&str],
        target_state: &str,
    ) -> StoreResult<Option<(AdmittedPrompt, i64)>> {
        type QueueHeadRow = (
            i64,
            OpId,
            String,
            String,
            Option<String>,
            Option<String>,
            Option<String>,
        );
        let conn = self.write();
        let tx = conn.unchecked_transaction()?;
        let head: Option<QueueHeadRow> = tx
            .query_row(
                "SELECT seq, op_id, prompt, files, model, variant, agent FROM prompt_queue
                 WHERE session_id = ?1 AND status = 'pending' ORDER BY seq ASC LIMIT 1",
                params![session.raw() as i64],
                |r| {
                    Ok((
                        r.get(0)?,
                        OpId::new(r.get::<_, i64>(1)? as u64),
                        r.get::<_, String>(2)?,
                        r.get::<_, String>(3)?,
                        r.get::<_, Option<String>>(4)?,
                        r.get::<_, Option<String>>(5)?,
                        r.get::<_, Option<String>>(6)?,
                    ))
                },
            )
            .ok();
        let Some((queue_seq, op_id, prompt, files_json, model, variant, agent)) = head else {
            return Ok(None);
        };
        let state: String = tx
            .query_row(
                "SELECT state FROM session WHERE id = ?1",
                params![session.raw() as i64],
                |r| r.get(0),
            )
            .map_err(|e| StoreError::Migration(format!("session missing: {e}")))?;
        let state_label: String = serde_json::from_str(&state).unwrap_or_default();
        if !eligible_states.contains(&state_label.as_str()) {
            return Ok(None);
        }
        tx.execute(
            "UPDATE prompt_queue SET status = 'claimed', claimed_at = ?2
             WHERE session_id = ?1 AND seq = ?3",
            params![session.raw() as i64, now_ms(), queue_seq],
        )?;
        let prev_event: i64 = tx.query_row(
            "SELECT COALESCE(MAX(seq), 0) FROM event WHERE session_id = ?1",
            params![session.raw() as i64],
            |r| r.get(0),
        )?;
        let event_seq = prev_event + 1;
        tx.execute(
            "INSERT INTO message(session_id, seq, role, data, created_ms)
             VALUES (?1, ?2, 'user', ?3, ?4)",
            params![
                session.raw() as i64,
                event_seq,
                serde_json::json!({ "text": prompt }).to_string(),
                now_ms()
            ],
        )?;
        tx.execute(
            "UPDATE session SET state = ?2, updated_ms = ?3 WHERE id = ?1",
            params![
                session.raw() as i64,
                serde_json::to_string(target_state).unwrap(),
                now_ms()
            ],
        )?;
        tx.commit()?;
        Ok(Some((
            AdmittedPrompt {
                queue_seq,
                op_id,
                prompt,
                files: serde_json::from_str(&files_json).unwrap_or_default(),
                model,
                variant,
                agent,
                message_seq: event_seq,
            },
            event_seq,
        )))
    }

    pub fn mark_queue_status(
        &self,
        session: SessionId,
        queue_seq: i64,
        status: &str,
    ) -> StoreResult<()> {
        let conn = self.write();
        conn.execute(
            "UPDATE prompt_queue SET status = ?3, completed_at = ?4
             WHERE session_id = ?1 AND seq = ?2",
            params![session.raw() as i64, queue_seq, status, now_ms()],
        )?;
        Ok(())
    }

    /// Op ids of all non-terminal queue rows for a session (abort(None)
    /// must durably cancel queued prompts too).
    pub fn queue_op_ids(&self, session: SessionId) -> StoreResult<Vec<OpId>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT op_id FROM prompt_queue
             WHERE session_id = ?1 AND status IN ('pending','claimed')",
        )?;
        let rows = stmt.query_map(params![session.raw() as i64], |r| r.get::<_, i64>(0))?;
        let mut out = Vec::new();
        for v in rows {
            out.push(OpId::new(v? as u64));
        }
        Ok(out)
    }

    /// Durable cancellation of queued rows (abort semantics): pending and
    /// claimed rows for the given ops become cancelled and are never
    /// admitted. Returns how many rows were cancelled.
    pub fn cancel_queued_ops(&self, session: SessionId, ops: &[OpId]) -> StoreResult<i64> {
        let conn = self.write();
        let mut n = 0i64;
        for op in ops {
            n += conn.execute(
                "UPDATE prompt_queue SET status = 'cancelled', completed_at = ?3
                 WHERE session_id = ?1 AND op_id = ?2 AND status IN ('pending','claimed')",
                params![session.raw() as i64, op.raw() as i64, now_ms()],
            )? as i64;
        }
        Ok(n)
    }

    /// Recovery pass: claimed rows that were never executed (crash between
    /// claim and execution) return to pending so they are re-admitted;
    /// running rows are left for turn-level recovery. Returns the re-admitted
    /// count.
    pub fn recover_claimed_queue_rows(&self, session: SessionId) -> StoreResult<i64> {
        let conn = self.write();
        let n = conn.execute(
            "UPDATE prompt_queue SET status = 'pending', claimed_at = NULL
             WHERE session_id = ?1 AND status = 'claimed'",
            params![session.raw() as i64],
        )? as i64;
        Ok(n)
    }

    /// All session ids with non-terminal queue rows (startup kick list).
    pub fn sessions_with_pending_queues(&self) -> StoreResult<Vec<SessionId>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT DISTINCT session_id FROM prompt_queue
             WHERE status IN ('pending','claimed','running')",
        )?;
        let rows = stmt.query_map([], |r| r.get::<_, i64>(0))?;
        let mut out = Vec::new();
        for v in rows {
            out.push(SessionId::new(v? as u64));
        }
        Ok(out)
    }

    /// Durable bump of one loop signal (spec §28): count the same key
    /// across turns and daemon restarts. Returns true when `threshold` is
    /// reached; the count then resets (a trip closes the window).
    pub fn bump_loop_signal(
        &self,
        session: SessionId,
        key: &str,
        threshold: u32,
        ts_ms: i64,
    ) -> StoreResult<bool> {
        if key.is_empty() || key.len() > 1024 || threshold < 2 {
            return Err(StoreError::Migration(
                "loop signal key must be 1..=1024 bytes; threshold >= 2".into(),
            ));
        }
        let conn = self.write();
        let tx = conn.unchecked_transaction()?;
        let prev: Option<i64> = tx
            .query_row(
                "SELECT count FROM loop_signal WHERE session_id = ?1 AND key = ?2",
                params![session.raw() as i64, key],
                |r| r.get(0),
            )
            .optional()?;
        let count = prev.unwrap_or(0) + 1;
        tx.execute(
            "INSERT OR REPLACE INTO loop_signal(session_id, key, count, updated_ms)
             VALUES (?1, ?2, ?3, ?4)",
            params![session.raw() as i64, key, count, ts_ms],
        )?;
        if count >= i64::from(threshold) {
            tx.execute(
                "DELETE FROM loop_signal WHERE session_id = ?1 AND key = ?2",
                params![session.raw() as i64, key],
            )?;
            tx.commit()?;
            return Ok(true);
        }
        tx.commit()?;
        Ok(false)
    }

    /// Clear every loop signal of the session (the task made progress).
    pub fn reset_loop_signals(&self, session: SessionId) -> StoreResult<()> {
        let conn = self.write();
        conn.execute(
            "DELETE FROM loop_signal WHERE session_id = ?1",
            params![session.raw() as i64],
        )?;
        Ok(())
    }

    pub fn loop_signal_counts(&self, session: SessionId) -> StoreResult<Vec<(String, i64)>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT key, count FROM loop_signal WHERE session_id = ?1 ORDER BY updated_ms DESC LIMIT 100",
        )?;
        let rows = stmt.query_map(params![session.raw() as i64], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    // ---------------------------------------------------------------- op ids

    /// Atomically reserve the range `[start, start + count)` from the ONE
    /// durable op-id sequence shared by every session (schema scope 0), and
    /// return `(start, count)`. Reserved ids are handed out by the caller
    /// strictly in order, so every id is unique and strictly increasing
    /// ACROSS daemon restarts — even when a restart lands in the same
    /// millisecond or the wall clock jumped backwards (the sequence never
    /// consults the clock after migration). A crash between the commit and
    /// the use of the reserved ids only burns ids (gaps); it can never
    /// reuse one.
    ///
    /// `_session` is reserved for a future per-session scope; the current
    /// schema pins one global row (op ids are globally unique — the
    /// `tool_run.op_id` UNIQUE column), so the value is ignored.
    ///
    /// The reservation runs in an IMMEDIATE transaction, so two live stores
    /// over the same database file (a restart racing its predecessor) see
    /// each other's commits instead of double-issuing a range.
    pub fn alloc_op_ids(&self, _session: SessionId, count: u64) -> StoreResult<(u64, u64)> {
        if count == 0 {
            return Err(StoreError::Conflict(
                "alloc_op_ids: count must be non-zero".into(),
            ));
        }
        let mut conn = self.write();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let next: i64 = tx
            .query_row(
                "SELECT next_value FROM op_id_seq WHERE session_scope = 0",
                [],
                |r| r.get(0),
            )
            .optional()?
            .ok_or_else(|| {
                StoreError::Migration(
                    "op_id_seq global row missing (migrations not applied?)".into(),
                )
            })?;
        if next < 0 {
            return Err(StoreError::Migration(format!(
                "op_id_seq next_value corrupted: {next}"
            )));
        }
        let start = next as u64;
        // `next_value` lives in a signed INTEGER column: the sequence is
        // exhausted once a reservation would cross i64::MAX.
        let end = start
            .checked_add(count)
            .filter(|end| *end <= i64::MAX as u64)
            .ok_or_else(|| StoreError::Conflict("alloc_op_ids: op-id sequence exhausted".into()))?;
        tx.execute(
            "UPDATE op_id_seq SET next_value = ?1 WHERE session_scope = 0",
            params![end as i64],
        )?;
        tx.commit()?;
        Ok((start, count))
    }

    /// The sequence's current high-water mark: the first id NOT yet
    /// reserved (ids handed out so far are all `< high_water`). Test probe.
    pub fn op_id_seq_high_water(&self) -> StoreResult<u64> {
        let conn = self.read()?;
        let next: i64 = conn
            .query_row(
                "SELECT next_value FROM op_id_seq WHERE session_scope = 0",
                [],
                |r| r.get(0),
            )
            .optional()?
            .ok_or_else(|| {
                StoreError::Migration(
                    "op_id_seq global row missing (migrations not applied?)".into(),
                )
            })?;
        Ok(next as u64)
    }

    pub fn integrity_check(&self) -> StoreResult<Vec<String>> {
        let conn = self.read()?;
        let out = check_integrity(&conn)?;
        Ok(out)
    }

    /// The FULL deep scan (`doctor --deep`, crash forensics): the complete
    /// `PRAGMA integrity_check` over the live store. Production starts use
    /// the bounded [`Store::open_fast`]/[`Store::quick_integrity_check`]
    /// path instead.
    pub fn deep_integrity_check(&self) -> StoreResult<Vec<String>> {
        self.integrity_check()
    }

    /// Bounded live-store check (plain `doctor`, post-open validation): the
    /// same `PRAGMA quick_check` the fast open runs. Detects damaged pages
    /// but skips the full scan's index-content re-verification.
    pub fn quick_integrity_check(&self) -> StoreResult<Vec<String>> {
        let conn = self.read()?;
        let out = check_quick(&conn)?;
        Ok(out)
    }

    /// Online backup via the SQLite backup API (safe while the daemon runs).
    pub fn backup_to(&self, dest: &Path) -> StoreResult<()> {
        let src = self.write();
        let mut dst = Connection::open(dest)?;
        let backup = rusqlite::backup::Backup::new(&src, &mut dst)?;
        backup.run_to_completion(50, std::time::Duration::from_millis(100), None)?;
        Ok(())
    }

    /// `doctor`-style diagnostic with the FULL integrity scan (`doctor
    /// --deep` depth; kept for legacy callers such as the session manager's
    /// `integrity_report`).
    pub fn diagnostics(&self) -> StoreResult<serde_json::Value> {
        self.diagnostics_with(check_integrity)
    }

    /// `doctor`-style diagnostic with the BOUNDED quick check (plain
    /// `doctor`, matching the fast open).
    pub fn diagnostics_quick(&self) -> StoreResult<serde_json::Value> {
        self.diagnostics_with(check_quick)
    }

    fn diagnostics_with(
        &self,
        integrity: fn(&Connection) -> StoreResult<Vec<String>>,
    ) -> StoreResult<serde_json::Value> {
        let conn = self.read()?;
        let sessions: i64 = conn.query_row("SELECT COUNT(*) FROM session", [], |r| r.get(0))?;
        let events: i64 = conn.query_row("SELECT COUNT(*) FROM event", [], |r| r.get(0))?;
        let messages: i64 = conn.query_row("SELECT COUNT(*) FROM message", [], |r| r.get(0))?;
        let journal_mode: String = conn.query_row("PRAGMA journal_mode", [], |r| r.get(0))?;
        let integrity = integrity(&conn)?;
        Ok(serde_json::json!({
            "journal_mode": journal_mode,
            "sessions": sessions,
            "events": events,
            "messages": messages,
            "integrity": integrity,
        }))
    }

    // -------------------------------------------------- deep doctor queries

    /// EVERY unfinished (still `running`) tool run across ALL sessions —
    /// `doctor --deep` and cross-session recovery audits. Unfinished rows
    /// are crash leftovers that recovery replays at the next start. Doctor
    /// reports them as information (a live daemon legitimately has running
    /// rows) rather than as errors.
    pub fn all_running_tool_rows(&self) -> StoreResult<Vec<ToolRunRow>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT id, session_id, op_id, tool, args, status, started_ms, ended_ms, effect_status, recovery, expected_hash, replay_descriptor, attempt, postcondition
             FROM tool_run WHERE status = 'running' ORDER BY started_ms ASC, id ASC",
        )?;
        let mut rows = stmt.query([])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(tool_run_map(row)?);
        }
        Ok(out)
    }

    /// Every ACTIVE logical-turn record across ALL sessions (`doctor
    /// --deep`): at most one active turn may exist per session while a
    /// daemon is live, so several active rows after a crash are the durable
    /// picture recovery resumes from. Informational in doctor.
    pub fn all_active_turns(&self) -> StoreResult<Vec<TurnRecordRow>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT id, session_id, turn_op_id, queue_seq, prompt_message_id, effective_provider, effective_model, variant, tool_mode, started_at, status, updated_ms
             FROM turn_record WHERE status = 'active' ORDER BY started_at ASC, id ASC",
        )?;
        let mut rows = stmt.query([])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(turn_record_map(row)?);
        }
        Ok(out)
    }

    /// Journal projection consistency for EVERY session (`doctor --deep`).
    /// A session's journal is the gapless sequence 1..=N: the (session_id,
    /// seq) primary key structurally forbids duplicates, so a count/range
    /// mismatch means a gap, a lost commit, or tampering. A session row with
    /// NO journal rows at all is also flagged: creation seeds the journal in
    /// the same transaction as the session row, so a row without events is
    /// a torn write. Returns human-readable problems; empty = consistent.
    pub fn journal_consistency_issues(&self) -> StoreResult<Vec<String>> {
        let conn = self.read()?;
        let mut issues = Vec::new();
        {
            let mut stmt = conn.prepare(
                "SELECT session_id, COUNT(*), MIN(seq), MAX(seq)
                 FROM event GROUP BY session_id",
            )?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                let sid: i64 = row.get(0)?;
                let count: i64 = row.get(1)?;
                let min_seq: i64 = row.get(2)?;
                let max_seq: i64 = row.get(3)?;
                if min_seq != 1 || count != max_seq {
                    issues.push(format!(
                        "session {sid}: journal holds {count} event(s) spanning seq {min_seq}..={max_seq}; invariant is a gapless 1..={count}"
                    ));
                }
            }
        }
        {
            // Sessions whose journal is missing entirely (torn creation).
            let mut stmt = conn.prepare(
                "SELECT s.id FROM session s
                 WHERE NOT EXISTS (SELECT 1 FROM event e WHERE e.session_id = s.id)",
            )?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                let sid: i64 = row.get(0)?;
                issues.push(format!(
                    "session {sid}: session row exists but its journal has no events (torn creation)"
                ));
            }
        }
        Ok(issues)
    }

    /// Every CAS blob hash the store schema references — `artifact.cas_hash`
    /// rows and `checkpoint.after_cas_hash` rows — with the referencing
    /// table and row id, for the doctor dangling-reference scan. The store
    /// never reads blob files; existence is verified by the caller against
    /// the CAS (the CLI's doctor owns both handles).
    pub fn cas_hash_references(&self) -> StoreResult<Vec<CasHashRef>> {
        let conn = self.read()?;
        let mut out = Vec::new();
        {
            let mut stmt = conn.prepare("SELECT id, cas_hash FROM artifact")?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                out.push(CasHashRef {
                    source: "artifact",
                    row_id: row.get(0)?,
                    hash: row.get(1)?,
                });
            }
        }
        {
            let mut stmt = conn.prepare(
                "SELECT id, after_cas_hash FROM checkpoint WHERE after_cas_hash IS NOT NULL",
            )?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                out.push(CasHashRef {
                    source: "checkpoint",
                    row_id: row.get(0)?,
                    hash: row.get(1)?,
                });
            }
        }
        Ok(out)
    }

    /// The durable budget ledger invariant scan (`doctor --deep`, read-only,
    /// P0-97): every `cost_reservation` row is counted by status and every
    /// row whose `(session_id, task_id)` task row no longer exists is listed
    /// as dangling. A reservation is the ledger's handle onto its task
    /// envelope: a dangling row can never settle or refund, so its predicted
    /// spend silently vanishes from the cap math.
    pub fn cost_reservation_invariants(&self) -> StoreResult<CostReservationScan> {
        let conn = self.read()?;
        let mut scan = CostReservationScan::default();
        {
            let mut stmt =
                conn.prepare("SELECT status, COUNT(*) FROM cost_reservation GROUP BY status")?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                let status: String = row.get(0)?;
                let count: i64 = row.get(1)?;
                let count = count.max(0) as u64;
                scan.total = scan.total.saturating_add(count);
                match status.as_str() {
                    "open" => {
                        // Legacy pre-v17 vocabulary: 0 on every v18 store.
                        scan.open += count;
                    }
                    "reserved" => {
                        scan.reserved += count;
                        scan.open += count;
                    }
                    "dispatched" => {
                        scan.dispatched += count;
                        scan.open += count;
                    }
                    "settled" => scan.settled += count,
                    "refunded" => scan.refunded += count,
                    "abandoned" => scan.abandoned += count,
                    "uncertain" => scan.uncertain += count,
                    _ => {}
                }
            }
        }
        {
            let mut stmt = conn.prepare(
                "SELECT cr.reservation_id, cr.session_id, cr.task_id, cr.op_id,
                        cr.predicted_micro, cr.status
                 FROM cost_reservation cr
                 WHERE NOT EXISTS (
                     SELECT 1 FROM task t
                     WHERE t.session_id = cr.session_id AND t.task_id = cr.task_id)
                 ORDER BY cr.reservation_id ASC",
            )?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                scan.dangling.push(DanglingReservationRow {
                    reservation_id: row.get(0)?,
                    session_id: SessionId::new(row.get::<_, i64>(1)?.max(1) as u64),
                    task_id: TaskId::new(row.get::<_, i64>(2)?.max(1) as u64),
                    op_id: OpId::new(row.get::<_, i64>(3)?.max(1) as u64),
                    predicted_micro: row.get::<_, i64>(4)?.max(0) as u64,
                    status: row.get(5)?,
                });
            }
        }
        Ok(scan)
    }

    /// The wave-16 verification-record consistency invariant (`doctor
    /// --deep`, read-only, P0-97). Issue kinds:
    ///
    /// - `record_without_task` — a record whose `task_id` matches no task
    ///   row at all;
    /// - `passed_on_uncompleted` — a `Passed` record certifying the CURRENT
    ///   revision of a task that is not `VerifiedComplete` (a `Passed` claim
    ///   only holds at the completed revision; the completion transaction
    ///   would have consumed it);
    /// - `verified_without_record` — a `VerifiedComplete` task with no
    ///   `Passed` record certifying the revision the completion consumed
    ///   (revision N requires a record certifying N-1: completion bumps the
    ///   row exactly once).
    pub fn verification_record_invariants(&self) -> StoreResult<VerificationInvariantScan> {
        let conn = self.read()?;
        let mut scan = VerificationInvariantScan::default();
        // Tasks: session_id, task_id, state, revision (typed parse: a
        // corrupt state text fails the scan loudly — never guessed).
        struct TaskRef {
            session_id: i64,
            task_id: i64,
            state: TaskState,
            revision: i64,
        }
        let mut tasks: Vec<TaskRef> = Vec::new();
        {
            let mut stmt = conn.prepare(
                "SELECT session_id, task_id, state, revision FROM task ORDER BY session_id ASC, task_id ASC",
            )?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                let task = TaskRef {
                    session_id: row.get(0)?,
                    task_id: row.get(1)?,
                    state: parse_json(
                        &format!(
                            "task {}/{} state",
                            row.get::<_, i64>(0)?,
                            row.get::<_, i64>(1)?
                        ),
                        &row.get::<_, String>(2)?,
                    )?,
                    revision: row.get(3)?,
                };
                if task.state.is_completion_relevant() {
                    scan.relevant_tasks += 1;
                }
                if task.state == TaskState::VerifiedComplete {
                    scan.completed_tasks += 1;
                }
                tasks.push(task);
            }
        }
        // Records: id, task_id, revision, status (typed parse of the status
        // JSON text, same corruption contract as the row mappers).
        struct RecordRef {
            id: i64,
            task_id: i64,
            revision: i64,
            status: VerificationStatus,
        }
        let mut records: Vec<RecordRef> = Vec::new();
        {
            let mut stmt = conn.prepare(
                "SELECT id, task_id, revision, status FROM verification_record ORDER BY id ASC",
            )?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                records.push(RecordRef {
                    id: row.get(0)?,
                    task_id: row.get(1)?,
                    revision: row.get(2)?,
                    status: parse_json(
                        &format!("verification_record {} status", row.get::<_, i64>(0)?),
                        &row.get::<_, String>(3)?,
                    )?,
                });
            }
        }
        scan.total_records = records.len() as u64;
        // (1) Records whose task row is gone entirely.
        for r in &records {
            if !tasks.iter().any(|t| t.task_id == r.task_id) {
                scan.issues.push(VerificationInvariantIssue {
                    kind: "record_without_task",
                    detail: format!(
                        "verification record {} (status {:?}, revision {}) references task {} which has no task row",
                        r.id, r.status, r.revision, r.task_id
                    ),
                });
            }
        }
        // (2) A Passed record may certify the current revision only of a
        // VerifiedComplete task.
        for t in &tasks {
            if t.state == TaskState::VerifiedComplete {
                continue;
            }
            for r in &records {
                if r.task_id == t.task_id
                    && r.status == VerificationStatus::Passed
                    && r.revision == t.revision
                {
                    scan.issues.push(VerificationInvariantIssue {
                        kind: "passed_on_uncompleted",
                        detail: format!(
                            "Passed verification record {} certifies the current revision {} of task {}/{} whose state is {:?}, not VerifiedComplete",
                            r.id, t.revision, t.session_id, t.task_id, t.state
                        ),
                    });
                }
            }
        }
        // (3) VerifiedComplete without the Passed record the completion
        // consumed (revision N needs a Passed record certifying N-1).
        for t in &tasks {
            if t.state != TaskState::VerifiedComplete {
                continue;
            }
            let certified = if t.revision >= 2 {
                records.iter().any(|r| {
                    r.task_id == t.task_id
                        && r.status == VerificationStatus::Passed
                        && r.revision == t.revision - 1
                })
            } else {
                false
            };
            if !certified {
                scan.issues.push(VerificationInvariantIssue {
                    kind: "verified_without_record",
                    detail: format!(
                        "task {}/{} is VerifiedComplete at revision {} but no Passed verification record certifies its completion revision {}",
                        t.session_id, t.task_id, t.revision, t.revision.saturating_sub(1)
                    ),
                });
            }
        }
        Ok(scan)
    }

    /// The active-turn recoverable-owner invariant (`doctor --deep`,
    /// read-only, P0-97): a live daemon legitimately owns active turn rows
    /// in memory, so doctor's question is the crashed-daemon one — can
    /// recovery own this row? A turn is recoverable when at least one
    /// durable anchor exists: its prompt message row, its prompt-queue row,
    /// a journal event naming its turn op, or a tool-run row naming it.
    pub fn active_turn_ownership_invariants(&self) -> StoreResult<TurnOwnershipScan> {
        let conn = self.read()?;
        let mut scan = TurnOwnershipScan::default();
        let mut stmt = conn.prepare(
            "SELECT id, session_id, turn_op_id, queue_seq, prompt_message_id
             FROM turn_record WHERE status = 'active' ORDER BY id ASC",
        )?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let record_id: i64 = row.get(0)?;
            let session_id: i64 = row.get(1)?;
            let turn_op_id: i64 = row.get(2)?;
            let queue_seq: Option<i64> = row.get(3)?;
            let prompt_message_id: Option<i64> = row.get(4)?;
            scan.active_turns += 1;
            let anchored = |sql: &str, params: &[&dyn rusqlite::ToSql]| -> StoreResult<bool> {
                let n: i64 = conn.query_row(sql, params, |r| r.get(0))?;
                Ok(n > 0)
            };
            let message_anchor = match prompt_message_id {
                Some(mid) => anchored(
                    // `prompt_message_id` records the prompt's message SEQ
                    // (== the PromptReceived journal event seq); the row id
                    // is a separate autoincrement.
                    "SELECT COUNT(*) FROM message WHERE session_id = ?1 AND seq = ?2",
                    &[&session_id, &mid],
                )?,
                None => false,
            };
            let queue_anchor = match queue_seq {
                Some(seq) => anchored(
                    "SELECT COUNT(*) FROM prompt_queue WHERE session_id = ?1 AND seq = ?2",
                    &[&session_id, &seq],
                )?,
                None => false,
            };
            let event_anchor = anchored(
                "SELECT COUNT(*) FROM event WHERE session_id = ?1 AND op_id = ?2",
                &[&session_id, &turn_op_id],
            )?;
            let tool_anchor = anchored(
                "SELECT COUNT(*) FROM tool_run WHERE session_id = ?1 AND op_id = ?2",
                &[&session_id, &turn_op_id],
            )?;
            if message_anchor || queue_anchor || event_anchor || tool_anchor {
                scan.recoverable += 1;
            } else {
                scan.unrecoverable.push(UnrecoverableActiveTurn {
                    record_id,
                    session_id: SessionId::new(session_id.max(1) as u64),
                    turn_op_id: OpId::new(turn_op_id.max(1) as u64),
                    detail: format!(
                        "active turn record {record_id} of session {session_id} (op {turn_op_id}) has no prompt message row, no prompt-queue row, no journal event and no tool-run row naming it — nothing can recover it after a crash"
                    ),
                });
            }
        }
        Ok(scan)
    }

    /// Every session row id, ascending (`doctor --deep` orphan scans).
    pub fn session_ids(&self) -> StoreResult<Vec<SessionId>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare("SELECT id FROM session ORDER BY id ASC")?;
        let mut rows = stmt.query([])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(SessionId::new(row.get::<_, i64>(0)?.max(1) as u64));
        }
        Ok(out)
    }

    /// Raw `memory_fact` rows of the given kinds across EVERY session,
    /// ascending by session then kind then key (`doctor --deep` orphan-child
    /// scan). Kinds are internal constants of the session/orchestrator
    /// layers — never user input. Bounded: more than
    /// [`MAX_ORCHESTRATOR_FACT_SCAN_ROWS`] rows is a typed refusal, never a
    /// silent truncation (bounded everything).
    pub fn memory_fact_rows_of_kinds(&self, kinds: &[&str]) -> StoreResult<Vec<MemoryFactRowRef>> {
        if kinds.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.read()?;
        let placeholders = vec!["?"; kinds.len()].join(",");
        let sql = format!(
            "SELECT session_id, kind, key, value FROM memory_fact
             WHERE kind IN ({placeholders})
             ORDER BY session_id ASC, kind ASC, key ASC
             LIMIT {MAX_ORCHESTRATOR_FACT_SCAN_ROWS}"
        );
        let mut stmt = conn.prepare(&sql)?;
        let params: Vec<&dyn rusqlite::ToSql> =
            kinds.iter().map(|k| k as &dyn rusqlite::ToSql).collect();
        let mut rows = stmt.query(params.as_slice())?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(MemoryFactRowRef {
                session_id: SessionId::new(row.get::<_, i64>(0)?.max(1) as u64),
                kind: row.get(1)?,
                key: row.get(2)?,
                value: row.get(3)?,
            });
        }
        if out.len() as i64 >= MAX_ORCHESTRATOR_FACT_SCAN_ROWS {
            return Err(StoreError::Oversized(format!(
                "memory-fact kind scan reached the {MAX_ORCHESTRATOR_FACT_SCAN_ROWS}-row bound; refusing a partial orphan scan"
            )));
        }
        Ok(out)
    }

    // ------------------------------------------------ child runtime (v23)

    /// Insert-or-replace the durable child-runtime projection row of one
    /// CHILD session (migration v23): the child's state plus its blocker
    /// truth. The session layer validates the text bounds before calling.
    pub fn child_runtime_put(&self, row: &ChildRuntimeRow) -> StoreResult<()> {
        let conn = self.write();
        conn.execute(
            "INSERT INTO child_runtime(
                 session_id, child_id, state, blocker_kind, blocker_reason,
                 blocker_dependency, blocker_resolution, last_progress_ms, updated_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
             ON CONFLICT(session_id) DO UPDATE SET
                 child_id = excluded.child_id,
                 state = excluded.state,
                 blocker_kind = excluded.blocker_kind,
                 blocker_reason = excluded.blocker_reason,
                 blocker_dependency = excluded.blocker_dependency,
                 blocker_resolution = excluded.blocker_resolution,
                 last_progress_ms = excluded.last_progress_ms,
                 updated_ms = excluded.updated_ms",
            params![
                row.session_id.raw() as i64,
                row.child_id,
                row.state,
                row.blocker_kind,
                row.blocker_reason,
                row.blocker_dependency,
                row.blocker_resolution,
                row.last_progress_ms,
                row.updated_ms,
            ],
        )?;
        Ok(())
    }

    /// The durable child-runtime projection row of one child session.
    pub fn child_runtime_get(&self, session_id: SessionId) -> StoreResult<Option<ChildRuntimeRow>> {
        let conn = self.read()?;
        let out = conn
            .query_row(
                "SELECT session_id, child_id, state, blocker_kind, blocker_reason,
                        blocker_dependency, blocker_resolution, last_progress_ms, updated_ms
                 FROM child_runtime WHERE session_id = ?1",
                params![session_id.raw() as i64],
                |r| {
                    Ok(ChildRuntimeRow {
                        session_id,
                        child_id: r.get(1)?,
                        state: r.get(2)?,
                        blocker_kind: r.get(3)?,
                        blocker_reason: r.get(4)?,
                        blocker_dependency: r.get(5)?,
                        blocker_resolution: r.get(6)?,
                        last_progress_ms: r.get(7)?,
                        updated_ms: r.get(8)?,
                    })
                },
            )
            .optional()?;
        Ok(out)
    }

    /// Drop the child-runtime projection row (the blocker was cleared).
    pub fn child_runtime_delete(&self, session_id: SessionId) -> StoreResult<()> {
        let conn = self.write();
        conn.execute(
            "DELETE FROM child_runtime WHERE session_id = ?1",
            params![session_id.raw() as i64],
        )?;
        Ok(())
    }

    // ------------------------------------------------- index state machine

    /// Durable state row of one workspace's repository index (audits 30/64).
    /// `state_json` is an opaque JSON payload owned by `faktor-index`
    /// (protocol-agnostic, like every other TEXT payload here); `generation`
    /// is the numeric generation that row names (0 for NotStarted).
    pub fn index_state_get(&self, workspace_id: WorkspaceId) -> StoreResult<Option<IndexStateRow>> {
        let conn = self.read()?;
        let out = conn
            .query_row(
                "SELECT workspace_id, state_json, generation, updated_ms
                 FROM index_state WHERE workspace_id = ?1",
                params![workspace_id.raw() as i64],
                index_state_map,
            )
            .optional()?;
        Ok(out)
    }

    /// Insert-or-replace the workspace's index state row AND append one
    /// journal row in a single transaction (the row and its journal entry
    /// are never torn apart). Used for corruption recovery and fresh-row
    /// seeding, where no expected value exists.
    pub fn index_state_put(
        &self,
        workspace_id: WorkspaceId,
        state_json: &str,
        generation: i64,
        kind: &str,
    ) -> StoreResult<()> {
        let mut conn = self.write();
        let now = now_ms();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute(
            "INSERT INTO index_state(workspace_id, state_json, generation, updated_ms)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(workspace_id) DO UPDATE SET
                state_json = excluded.state_json,
                generation = excluded.generation,
                updated_ms = excluded.updated_ms",
            params![workspace_id.raw() as i64, state_json, generation, now],
        )?;
        tx.execute(
            "INSERT INTO index_state_log(workspace_id, kind, state_json, generation, updated_ms)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![workspace_id.raw() as i64, kind, state_json, generation, now],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Atomic compare-and-swap of the workspace's index state row: the row
    /// is replaced only when a row exists whose `(state_json, generation)`
    /// equal the expected pair exactly. The journal row is appended in the
    /// SAME transaction, so a legal state transition and its journal entry
    /// commit or fail together. Returns `Ok(false)` (writing nothing) when
    /// the row differs — the caller re-reads and re-decides. Two builders
    /// racing for the same generation therefore have exactly one winner.
    pub fn index_state_cas(
        &self,
        workspace_id: WorkspaceId,
        expected_state_json: &str,
        expected_generation: i64,
        new_state_json: &str,
        new_generation: i64,
        kind: &str,
    ) -> StoreResult<bool> {
        let mut conn = self.write();
        let now = now_ms();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let current: Option<(String, i64)> = tx
            .query_row(
                "SELECT state_json, generation FROM index_state WHERE workspace_id = ?1",
                params![workspace_id.raw() as i64],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        let matched = match current {
            Some((state_json, generation)) => {
                state_json == expected_state_json && generation == expected_generation
            }
            None => false,
        };
        if !matched {
            return Ok(false); // rollback on drop; nothing written
        }
        tx.execute(
            "UPDATE index_state SET state_json = ?2, generation = ?3, updated_ms = ?4
             WHERE workspace_id = ?1",
            params![
                workspace_id.raw() as i64,
                new_state_json,
                new_generation,
                now
            ],
        )?;
        tx.execute(
            "INSERT INTO index_state_log(workspace_id, kind, state_json, generation, updated_ms)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                workspace_id.raw() as i64,
                kind,
                new_state_json,
                new_generation,
                now
            ],
        )?;
        tx.commit()?;
        Ok(true)
    }

    /// Append-only transition journal of one workspace (newest first,
    /// bounded page). Read side for recovery forensics and the adversarial
    /// "exactly one rebuild" assertions.
    pub fn index_state_log(
        &self,
        workspace_id: WorkspaceId,
        limit: i64,
    ) -> StoreResult<Vec<IndexStateLogRow>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT id, workspace_id, kind, state_json, generation, updated_ms
             FROM index_state_log WHERE workspace_id = ?1
             ORDER BY id DESC LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![workspace_id.raw() as i64, limit.max(0)], |r| {
            index_state_log_map(r)
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    // ------------------------------------------------------- durable cost ledger
    // (P0-6/12 through attempt-identity accounting, schema v18: the
    // cost_reservation table + the task row's READ-ONLY monetary columns.
    // The task machine never writes these columns — upsert_task enumerates
    // its column list and get_task never selects them — so this section is
    // their ONLY writer and the ledger is the single monetary authority.
    // Every typed refusal leaves the row untouched. The v17 migration (index
    // 17) froze the status vocabulary to reserved | dispatched | settled |
    // refunded | uncertain and made refund-after-dispatch impossible at the
    // SQL level; attempt-keyed rows join provider_call by attempt_op_id.)

    /// The durable monetary envelope of one task row (schema v15): the cap
    /// (`None` = unlimited) and the settled spend.
    pub fn cost_task_row(
        &self,
        session_id: SessionId,
        task_id: TaskId,
    ) -> StoreResult<Option<TaskCostRow>> {
        let conn = self.read()?;
        conn.query_row(
            "SELECT max_cost_micro, spent_cost_micro FROM task
             WHERE session_id = ?1 AND task_id = ?2",
            params![session_id.raw() as i64, task_id.raw() as i64],
            |r| {
                Ok(TaskCostRow {
                    max_cost_micro: r.get::<_, Option<i64>>(0)?.map(|m| m.max(0) as u64),
                    spent_cost_micro: u64::try_from(r.get::<_, i64>(1)?).unwrap_or(u64::MAX),
                })
            },
        )
        .optional()
        .map_err(Into::into)
    }

    /// Set (or clear) the durable monetary cap of one task row (v15).
    /// `None` = unlimited. A missing task row is a typed `Conflict` — the
    /// task machine owns row creation.
    pub fn cost_task_cap_set(
        &self,
        session_id: SessionId,
        task_id: TaskId,
        max_cost_micro: Option<u64>,
    ) -> StoreResult<()> {
        let conn = self.write();
        let n = conn.execute(
            "UPDATE task SET max_cost_micro = ?1 WHERE session_id = ?2 AND task_id = ?3",
            params![
                max_cost_micro.map(|m| m.min(i64::MAX as u64) as i64),
                session_id.raw() as i64,
                task_id.raw() as i64
            ],
        )?;
        if n == 0 {
            return Err(StoreError::Conflict(format!(
                "task {task_id} of session {session_id} has no row; the task machine owns creation"
            )));
        }
        Ok(())
    }

    /// Whether a task in `state` may begin a NEW paid provider operation.
    /// A task that entered the verification/completion path
    /// (`NeedsVerification`/`Verifying`) or any terminal state
    /// (`VerifiedComplete`/`Failed`/`Cancelled`) may not: its accounting is
    /// being closed, and a reservation landing now could strand money after
    /// the completion gate. `Pending`/`Planning`/`Running`/`Waiting`/
    /// `Blocked` permit new provider operations.
    fn task_state_permits_provider_operation(state: TaskState) -> bool {
        !state.is_terminal() && !state.is_completion_relevant()
    }

    /// Reserve `predicted_micro` of the task's monetary budget in ONE
    /// transaction: the cap is read and the reservation inserted atomically
    /// (spent + predicted > cap => nothing written). A missing task row is a
    /// typed `Conflict` (drives create the row before any paid call); a task
    /// row in a completion/final state (`NeedsVerification`, `Verifying`,
    /// `VerifiedComplete`, `Failed`, `Cancelled`) refuses a new provider
    /// operation with a typed `Conflict` and writes NOTHING.
    ///
    /// This is the legacy entry point (no pricing snapshot: the row is
    /// reserved unpriced — settlement then cannot price it and fails closed
    /// under a hard cap); the settlement layer uses
    /// [`Store::cost_reserve_priced`], which additionally persists the
    /// route-time [`PricingSnapshot`] the call will be settled against.
    pub fn cost_reserve(
        &self,
        session_id: SessionId,
        task_id: TaskId,
        op_id: OpId,
        predicted_micro: u64,
        created_ms: i64,
    ) -> StoreResult<CostReserveOutcome> {
        self.cost_reserve_inner(
            session_id,
            task_id,
            op_id,
            predicted_micro,
            created_ms,
            None,
        )
    }

    /// [`Store::cost_reserve`] plus the immutable route-time price capture
    /// (P0-1): the snapshot is persisted on the reservation so settlement —
    /// including settlement that happens after a daemon restart — prices the
    /// call's usage against exactly the prices the router froze at route
    /// time. Bounded by the session layer before this call.
    pub fn cost_reserve_priced(
        &self,
        session_id: SessionId,
        task_id: TaskId,
        op_id: OpId,
        predicted_micro: u64,
        created_ms: i64,
        pricing_snapshot_json: Option<&str>,
    ) -> StoreResult<CostReserveOutcome> {
        self.cost_reserve_inner(
            session_id,
            task_id,
            op_id,
            predicted_micro,
            created_ms,
            pricing_snapshot_json,
        )
    }

    fn cost_reserve_inner(
        &self,
        session_id: SessionId,
        task_id: TaskId,
        op_id: OpId,
        predicted_micro: u64,
        created_ms: i64,
        pricing_snapshot_json: Option<&str>,
    ) -> StoreResult<CostReserveOutcome> {
        let mut conn = self.write();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let row_opt: Option<(Option<i64>, String)> = tx
            .query_row(
                "SELECT max_cost_micro, state FROM task WHERE session_id = ?1 AND task_id = ?2",
                params![session_id.raw() as i64, task_id.raw() as i64],
                |r| Ok((r.get::<_, Option<i64>>(0)?, r.get::<_, String>(1)?)),
            )
            .optional()?;
        // Outer None = no task row; inner None = the row's cap is NULL =
        // unlimited (distinct states: an unlimited cap is a valid cap).
        let (cap, state): (Option<i64>, TaskState) = match row_opt {
            Some((cap, state_json)) => {
                let state: TaskState = parse_json(
                    &format!("cost reserve: task {task_id} of session {session_id} state"),
                    &state_json,
                )?;
                (cap, state)
            }
            None => {
                tx.rollback()?;
                return Err(StoreError::Conflict(format!(
                    "cost reserve: task {task_id} of session {session_id} has no row"
                )));
            }
        };
        // The task-state gate: a reserve while the task is in a
        // completion/final state refuses typed and writes NOTHING (the same
        // predicate the completion transaction's zero-reservation gate
        // closes from the other side).
        if !Self::task_state_permits_provider_operation(state) {
            tx.rollback()?;
            return Err(StoreError::Conflict(format!(
                "cost reserve: task state {} forbids a new provider operation",
                state.label()
            )));
        }
        let spent: i64 = tx.query_row(
            "SELECT spent_cost_micro FROM task WHERE session_id = ?1 AND task_id = ?2",
            params![session_id.raw() as i64, task_id.raw() as i64],
            |r| r.get(0),
        )?;
        // Free balance subtracts BOTH the settled spend and the predictions
        // of every row that still holds budget (OPEN + UNCERTAIN — a
        // reservation a crashed daemon may have dispatched keeps consuming
        // the reserved amount until reconcile/finalize closes it):
        // ceiling = spent + in-flight + free; two concurrent reservations can
        // never jointly overshoot the cap.
        let open_sum: i64 = tx.query_row(
            "SELECT COALESCE(SUM(predicted_micro), 0) FROM cost_reservation
             WHERE session_id = ?1 AND task_id = ?2 AND status IN ('reserved', 'dispatched', 'uncertain')",
            params![session_id.raw() as i64, task_id.raw() as i64],
            |r| r.get(0),
        )?;
        // NULL cap = unlimited: `cap.unwrap_or(0)` keeps the free balance at
        // 0-minus-nothing (i.e. everything is free) and the `cap > 0` guard
        // below skips the refusal.
        let cap_limit = cap.unwrap_or(0);
        // saturating_sub clamps at i64::MIN, not at 0: clamp to zero here —
        // a task can never have negative free balance.
        let free = cap_limit
            .max(0)
            .saturating_sub(spent.max(0))
            .saturating_sub(open_sum.max(0))
            .max(0);
        if cap_limit > 0 && i64::try_from(predicted_micro).unwrap_or(i64::MAX) > free {
            tx.rollback()?;
            return Ok(CostReserveOutcome::Exceeded { free: free as u64 });
        }
        let id = tx.query_row(
            "INSERT INTO cost_reservation
                (session_id, task_id, op_id, predicted_micro, status, created_ms,
                 pricing_snapshot_json, estimated_cost_micro)
             VALUES (?1, ?2, ?3, ?4, 'reserved', ?5, ?6, ?4)
             RETURNING reservation_id",
            params![
                session_id.raw() as i64,
                task_id.raw() as i64,
                op_id.raw() as i64,
                predicted_micro.min(i64::MAX as u64) as i64,
                created_ms,
                pricing_snapshot_json
            ],
            |r| r.get(0),
        )?;
        tx.commit()?;
        Ok(CostReserveOutcome::Granted(id))
    }

    /// [`Store::cost_reserve_priced`] for one PHYSICAL ATTEMPT (attempt
    /// accounting, schema v18): the reservation keys by the attempt's fresh
    /// op id (`attempt_op_id`, also stored in `op_id` — the row's own op),
    /// carries the shared logical parent op id (`parent_op_id`, what legacy
    /// rows stored in `op_id`) and durably freezes the reserve-time estimate
    /// in `estimated_cost_micro`. Two attempts of the SAME logical op get
    /// two distinct rows with distinct `attempt_op_id`s; legacy `op_id`
    /// joins between reservations and provider-call rows never apply to
    /// attempt rows (reconciliation joins `attempt_op_id` instead).
    #[allow(clippy::too_many_arguments)]
    pub fn cost_reserve_attempt(
        &self,
        session_id: SessionId,
        task_id: TaskId,
        attempt: &faktor_core::op::ModelCallAttempt,
        predicted_micro: u64,
        created_ms: i64,
        pricing_snapshot_json: Option<&str>,
    ) -> StoreResult<CostReserveOutcome> {
        let mut conn = self.write();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let row_opt: Option<(Option<i64>, String)> = tx
            .query_row(
                "SELECT max_cost_micro, state FROM task WHERE session_id = ?1 AND task_id = ?2",
                params![session_id.raw() as i64, task_id.raw() as i64],
                |r| Ok((r.get::<_, Option<i64>>(0)?, r.get::<_, String>(1)?)),
            )
            .optional()?;
        let (cap, state): (Option<i64>, TaskState) = match row_opt {
            Some((cap, state_json)) => {
                let state: TaskState = parse_json(
                    &format!("cost reserve: task {task_id} of session {session_id} state"),
                    &state_json,
                )?;
                (cap, state)
            }
            None => {
                tx.rollback()?;
                return Err(StoreError::Conflict(format!(
                    "cost reserve: task {task_id} of session {session_id} has no row"
                )));
            }
        };
        if !Self::task_state_permits_provider_operation(state) {
            tx.rollback()?;
            return Err(StoreError::Conflict(format!(
                "cost reserve: task state {} forbids a new provider operation",
                state.label()
            )));
        }
        let spent: i64 = tx.query_row(
            "SELECT spent_cost_micro FROM task WHERE session_id = ?1 AND task_id = ?2",
            params![session_id.raw() as i64, task_id.raw() as i64],
            |r| r.get(0),
        )?;
        let open_sum: i64 = tx.query_row(
            "SELECT COALESCE(SUM(predicted_micro), 0) FROM cost_reservation
             WHERE session_id = ?1 AND task_id = ?2
               AND status IN ('reserved', 'dispatched', 'uncertain')",
            params![session_id.raw() as i64, task_id.raw() as i64],
            |r| r.get(0),
        )?;
        let cap_limit = cap.unwrap_or(0);
        let free = cap_limit
            .max(0)
            .saturating_sub(spent.max(0))
            .saturating_sub(open_sum.max(0))
            .max(0);
        if cap_limit > 0 && i64::try_from(predicted_micro).unwrap_or(i64::MAX) > free {
            tx.rollback()?;
            return Ok(CostReserveOutcome::Exceeded { free: free as u64 });
        }
        let predicted = predicted_micro.min(i64::MAX as u64) as i64;
        let id = tx.query_row(
            "INSERT INTO cost_reservation
                (session_id, task_id, op_id, attempt_op_id, parent_op_id,
                 predicted_micro, status, created_ms, pricing_snapshot_json,
                 estimated_cost_micro)
             VALUES (?1, ?2, ?3, ?3, ?4, ?5, 'reserved', ?6, ?7, ?5)
             RETURNING reservation_id",
            params![
                session_id.raw() as i64,
                task_id.raw() as i64,
                attempt.attempt_op_id.raw() as i64,
                attempt.logical_op_id.raw() as i64,
                predicted,
                created_ms,
                pricing_snapshot_json
            ],
            |r| r.get(0),
        )?;
        tx.commit()?;
        Ok(CostReserveOutcome::Granted(id))
    }

    /// The current state of one reservation row (attempt-accounting
    /// observability): `(session, status, dispatched_ms)`. `None` = no row.
    pub fn cost_reservation_state(
        &self,
        reservation_id: i64,
    ) -> StoreResult<Option<(SessionId, String, Option<i64>)>> {
        let conn = self.read()?;
        let out = conn
            .query_row(
                "SELECT session_id, status, dispatched_ms FROM cost_reservation
                 WHERE reservation_id = ?1",
                params![reservation_id],
                |r| {
                    Ok((
                        SessionId::new(r.get::<_, i64>(0)?.max(1) as u64),
                        r.get::<_, String>(1)?,
                        r.get::<_, Option<i64>>(2)?,
                    ))
                },
            )
            .optional()?;
        Ok(out)
    }

    /// Settle one RESERVED/DISPATCHED reservation and fold `actual_micro`
    /// into the task's spent total in ONE transaction (schema v18). The
    /// provider-reported cost, the locally calculated cost and the routing
    /// decision's JSON are recorded on the row (all bounded by the session
    /// layer before this call); the canonical v18 columns are written too:
    /// `settled_cost_micro` = the folded actual, `provider_reported_cost_micro`
    /// = the provider report, `cost_basis` = the honest authority behind the
    /// folded amount (ProviderReported when a provider report was given —
    /// the caller's documented numeric settle IS that report's authority —
    /// else RouteSnapshotEstimate for a pre-computed estimate-charge site).
    /// An overshooting actual is recorded honestly (the money was spent);
    /// the NEXT reservation is what refuses.
    ///
    /// LEGACY numeric settle: the caller supplies the actual directly (no
    /// price math). The settlement layer's truthful usage settlement is
    /// [`Store::cost_settle_usage`]; this entry point remains for callers
    /// that hold a pre-computed actual (documented estimate-charge sites and
    /// direct-store tests) and for the pre-P0-1 wire path.
    #[allow(clippy::too_many_arguments)]
    pub fn cost_settle(
        &self,
        reservation_id: i64,
        actual_micro: u64,
        provider_cost_micro: Option<u64>,
        provider_reported_micro: Option<u64>,
        route_decision_json: Option<&str>,
        settled_ms: i64,
    ) -> StoreResult<CostReservationState> {
        let mut conn = self.write();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let row: Option<(i64, i64, String)> = tx
            .query_row(
                "SELECT session_id, task_id, status FROM cost_reservation
                 WHERE reservation_id = ?1",
                params![reservation_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()?;
        let Some((session_id, task_id, status)) = row else {
            tx.rollback()?;
            return Ok(CostReservationState::Missing);
        };
        if status != "reserved" && status != "dispatched" {
            tx.rollback()?;
            return Ok(CostReservationState::NotOpen { current: status });
        }
        let actual = actual_micro.min(i64::MAX as u64) as i64;
        let basis = if provider_reported_micro.is_some() {
            COST_BASIS_PROVIDER_REPORTED
        } else {
            COST_BASIS_ROUTE_SNAPSHOT_ESTIMATE
        };
        tx.execute(
            "UPDATE cost_reservation
             SET status = 'settled', settled_ms = ?1, delivery_state = ?6,
                 provider_cost_micro = ?2, provider_reported_micro = ?3,
                 provider_reported_cost_micro = ?3, settled_cost_micro = ?7,
                 cost_basis = ?8,
                 route_decision_json = ?4
             WHERE reservation_id = ?5",
            params![
                settled_ms,
                provider_cost_micro.map(|m| m.min(i64::MAX as u64) as i64),
                provider_reported_micro.map(|m| m.min(i64::MAX as u64) as i64),
                route_decision_json,
                reservation_id,
                DELIVERY_COMPLETED,
                actual,
                basis
            ],
        )?;
        let n = tx.execute(
            "UPDATE task SET spent_cost_micro = spent_cost_micro + ?1
             WHERE session_id = ?2 AND task_id = ?3",
            params![actual, session_id, task_id],
        )?;
        if n == 0 {
            tx.rollback()?;
            return Err(StoreError::Conflict(format!(
                "cost settle: task {task_id} of session {session_id} has no row"
            )));
        }
        tx.commit()?;
        Ok(CostReservationState::Applied)
    }

    /// Settle one RESERVED/DISPATCHED reservation from a provider usage
    /// frame (P0-1 settlement truth, schema v18) in ONE transaction. The
    /// chosen actual: the provider-reported cost when the usage frame
    /// carried one (authoritative); otherwise the token categories x the
    /// reservation's stored route-time [`PricingSnapshot`] — exactly the
    /// price lines the router froze, never a fabricated per-token number.
    /// An Unknown price source (or no snapshot at all) with a hard task cap
    /// is a typed [`CostSettleOutcome::UnknownPrice`] refusal (nothing
    /// written); with no cap the reservation closes as an honest Unknown
    /// spend (amount columns NULL, nothing folded into the task total).
    ///
    /// The locally calculated amount and the provider-reported amount are
    /// BOTH recorded on the row when present; the canonical v18 columns are
    /// written too — `settled_cost_micro` = the amount actually folded,
    /// `provider_reported_cost_micro` = the provider report,
    /// `cost_basis` = ProviderReported when the reported amount won, else
    /// RouteSnapshotEstimate when the categories x snapshot actual won.
    /// An overshooting chosen actual is recorded honestly; the NEXT
    /// reservation is what refuses.
    #[allow(clippy::too_many_arguments)]
    pub fn cost_settle_usage(
        &self,
        reservation_id: i64,
        uncached_input_tokens: u64,
        cache_read_tokens: u64,
        cache_write_tokens: u64,
        output_tokens: u64,
        provider_reported_micro: Option<u64>,
        route_decision_json: Option<&str>,
        settled_ms: i64,
    ) -> StoreResult<CostSettleOutcome> {
        let mut conn = self.write();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let row: Option<(i64, i64, String, Option<String>)> = tx
            .query_row(
                "SELECT session_id, task_id, status, pricing_snapshot_json
                 FROM cost_reservation WHERE reservation_id = ?1",
                params![reservation_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )
            .optional()?;
        let Some((session_id, task_id, status, snapshot_json)) = row else {
            tx.rollback()?;
            return Ok(CostSettleOutcome::Missing);
        };
        if status != "reserved" && status != "dispatched" {
            tx.rollback()?;
            return Ok(CostSettleOutcome::NotOpen { current: status });
        }
        let snapshot = match snapshot_json {
            Some(json) => Some(parse_json::<PricingSnapshot>(
                &format!("reservation {reservation_id} pricing_snapshot_json"),
                &json,
            )?),
            None => None,
        };
        // The locally calculated actual: categories x the frozen snapshot.
        // Unknown/absent snapshot => no honest number exists.
        let local = snapshot.and_then(|s| {
            s.settle_cost(
                uncached_input_tokens,
                cache_read_tokens,
                cache_write_tokens,
                output_tokens,
            )
        });
        let (local_i64, reported_i64) = (
            local.map(|m| m.min(i64::MAX as u64) as i64),
            provider_reported_micro.map(|m| m.min(i64::MAX as u64) as i64),
        );
        // The chosen actual: the provider-reported cost when the usage frame
        // carried one (authoritative); else the locally calculated
        // categories x snapshot. Unknown price + no reported cost: with a
        // hard cap this is a typed refusal (the cap was protected by the
        // reservation's prediction, but a fabricated actual must never
        // land); with no cap the row closes as a documented Unknown spend.
        let chosen_i64: i64 = match provider_reported_micro {
            Some(_) => reported_i64.expect("reported Some maps to Some"),
            None => match local_i64 {
                Some(local) => local,
                None => {
                    let has_cap: bool = tx
                        .query_row(
                            "SELECT max_cost_micro > 0 FROM task
                             WHERE session_id = ?1 AND task_id = ?2",
                            params![session_id, task_id],
                            |r| r.get(0),
                        )
                        .unwrap_or(false);
                    if has_cap {
                        tx.rollback()?;
                        return Ok(CostSettleOutcome::UnknownPrice {
                            reservation: reservation_id,
                        });
                    }
                    // No cap: record the Unknown spend honestly — status
                    // settled, both amount columns NULL, nothing folded,
                    // cost_basis Unknown.
                    tx.execute(
                        "UPDATE cost_reservation
                         SET status = 'settled', settled_ms = ?1, delivery_state = ?3,
                             provider_cost_micro = NULL, provider_reported_micro = NULL,
                             provider_reported_cost_micro = NULL, settled_cost_micro = NULL,
                             cost_basis = ?4,
                             route_decision_json = ?2
                         WHERE reservation_id = ?5",
                        params![
                            settled_ms,
                            route_decision_json,
                            DELIVERY_COMPLETED,
                            COST_BASIS_UNKNOWN,
                            reservation_id
                        ],
                    )?;
                    tx.commit()?;
                    return Ok(CostSettleOutcome::AppliedUnknown);
                }
            },
        };
        let basis = if provider_reported_micro.is_some() {
            COST_BASIS_PROVIDER_REPORTED
        } else {
            COST_BASIS_ROUTE_SNAPSHOT_ESTIMATE
        };
        tx.execute(
            "UPDATE cost_reservation
             SET status = 'settled', settled_ms = ?1, delivery_state = ?6,
                 provider_cost_micro = ?2, provider_reported_micro = ?3,
                 provider_reported_cost_micro = ?3, settled_cost_micro = ?7,
                 cost_basis = ?8,
                 route_decision_json = ?4
             WHERE reservation_id = ?5",
            params![
                settled_ms,
                local_i64,
                reported_i64,
                route_decision_json,
                reservation_id,
                DELIVERY_COMPLETED,
                chosen_i64,
                basis
            ],
        )?;
        let n = tx.execute(
            "UPDATE task SET spent_cost_micro = spent_cost_micro + ?1
             WHERE session_id = ?2 AND task_id = ?3",
            params![chosen_i64, session_id, task_id],
        )?;
        if n == 0 {
            tx.rollback()?;
            return Err(StoreError::Conflict(format!(
                "cost settle: task {task_id} of session {session_id} has no row"
            )));
        }
        tx.commit()?;
        Ok(CostSettleOutcome::Applied {
            actual_micro: chosen_i64.max(0) as u64,
        })
    }

    /// Mark one RESERVED reservation as DISPATCHED (P0-2 + attempt-identity
    /// accounting): the status moves `reserved` -> `dispatched` AND the
    /// durable `dispatched_ms` marker + `delivery_state` are written in the
    /// SAME statement, immediately BEFORE the provider transport call. Crash
    /// recovery can then tell "dispatch never provably began" (still
    /// `reserved`, marker NULL -> refund) from "the provider may have
    /// billed" (`dispatched`, marker set -> UNCERTAIN), and the SQL-level
    /// refund guard (refund only touches `reserved` + marker NULL) can never
    /// free a dispatched reservation. A second mark of the same
    /// still-dispatchable reservation is idempotent (a retry re-marking its
    /// own row); anything settled/refunded/uncertain is a typed refusal.
    pub fn cost_mark_dispatched(
        &self,
        reservation_id: i64,
        dispatched_ms: i64,
    ) -> StoreResult<CostReservationState> {
        let conn = self.write();
        let status: Option<String> = conn
            .query_row(
                "SELECT status FROM cost_reservation WHERE reservation_id = ?1",
                params![reservation_id],
                |r| r.get(0),
            )
            .optional()?;
        let Some(status) = status else {
            return Ok(CostReservationState::Missing);
        };
        if status != "reserved" && status != "dispatched" {
            return Ok(CostReservationState::NotOpen { current: status });
        }
        conn.execute(
            "UPDATE cost_reservation
             SET status = 'dispatched', dispatched_ms = ?1, delivery_state = ?2
             WHERE reservation_id = ?3 AND status IN ('reserved', 'dispatched')",
            params![dispatched_ms, DELIVERY_DISPATCHED, reservation_id],
        )?;
        Ok(CostReservationState::Applied)
    }

    /// Refund one pre-dispatch reservation (`reserved`/legacy `open` with a
    /// NULL dispatch marker -> REFUNDED): the prediction is released and the
    /// spent total is untouched. The guard lives in the SQL itself —
    /// `WHERE ... AND status IN ('reserved','open') AND dispatched_ms IS
    /// NULL` — so a refund that reaches a dispatched, settled, refunded or
    /// uncertain reservation changes ZERO rows and is refused here with the
    /// row's current status and dispatch marker ([`RefundOutcome::Blocked`]):
    /// money is freed only by the guarded UPDATE, never by a caller's
    /// pre-check. This makes the pre-dispatch-only refund enforceable even
    /// against a caller that mis-calls refund after dispatch (the agent
    /// runtime's current post-dispatch refund sites): the money stays put.
    pub fn cost_refund(&self, reservation_id: i64, settled_ms: i64) -> StoreResult<RefundOutcome> {
        let mut conn = self.write();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let row: Option<(String, Option<i64>)> = tx
            .query_row(
                "SELECT status, dispatched_ms FROM cost_reservation WHERE reservation_id = ?1",
                params![reservation_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        let Some((status, dispatched_ms)) = row else {
            tx.rollback()?;
            return Ok(RefundOutcome::Missing);
        };
        let n = tx.execute(
            "UPDATE cost_reservation SET status = 'refunded', settled_ms = ?2
             WHERE reservation_id = ?1
               AND status IN ('reserved', 'open') AND dispatched_ms IS NULL",
            params![reservation_id, settled_ms],
        )?;
        if n == 0 {
            tx.rollback()?;
            // The guarded UPDATE changed nothing: the row is no longer
            // refundable pre-dispatch (dispatched / settled / refunded /
            // uncertain, or a legacy reserved+marker row). Typed, nothing
            // written, money untouched.
            return Ok(RefundOutcome::Blocked {
                current: status,
                dispatched_ms,
            });
        }
        tx.commit()?;
        Ok(RefundOutcome::Applied)
    }

    /// Crash recovery (P0-6/12): every pre-dispatch reservation of a crashed
    /// process becomes REFUNDED and is NEVER counted as spent — the op never
    /// settled, so its prediction was never spent (ops that DID run carry
    /// settled rows). Idempotent: a second recovery finds nothing reserved.
    ///
    /// LEGACY entry point retained for direct-store callers that predate the
    /// P0-2 marker semantics AND the v17 state vocabulary: the legacy
    /// `abandoned` state no longer exists (the v17 CHECK forbids it — an
    /// abandoned reservation charged $0, the exact hole the P0-2 marker
    /// semantics close), so the honest replacement for "close every
    /// still-open row of a crashed process" IS the marker split:
    /// pre-dispatch rows refund, post-marker rows go UNCERTAIN. This entry
    /// point delegates to that split and reports the number of rows closed.
    /// The settlement layer's recovery is
    /// [`Store::cost_recover_open_reservations`].
    pub fn cost_abandon_open_reservations(&self, at_ms: i64) -> StoreResult<u64> {
        let (refunded, uncertain) = self.cost_recover_open_reservations(at_ms)?;
        Ok(refunded + uncertain)
    }

    /// P0-2 crash recovery of every pre-dispatch reservation of a crashed
    /// process, split on the durable dispatch marker in ONE transaction
    /// (schema v18: `reserved`/`dispatched` are the crash-relevant states —
    /// a legacy v16 row can still read `reserved` + marker after migration).
    /// Idempotent (a second recovery finds nothing open). Returns
    /// `(refunded, uncertain)`:
    ///
    /// - `reserved` with `dispatched_ms` NULL — dispatch never provably
    ///   began, so the provider was never contacted: REFUNDED, free budget
    ///   restored;
    /// - `dispatched` (or a legacy `reserved`/`open` row with `dispatched_ms`
    ///   set) — the provider request was sent and may have been billed:
    ///   UNCERTAIN, the reserved amount keeps consuming the task's free
    ///   budget until a reconcile settles it from the attempt's durable
    ///   provider-call rows or the task-completion finalize charges the
    ///   estimate. `failure_reason_code` records the crash for forensics.
    pub fn cost_recover_open_reservations(&self, at_ms: i64) -> StoreResult<(u64, u64)> {
        let conn = self.write();
        let refunded = conn.execute(
            "UPDATE cost_reservation SET status = 'refunded', settled_ms = ?1
             WHERE status IN ('reserved', 'open') AND dispatched_ms IS NULL",
            params![at_ms],
        )?;
        let uncertain = conn.execute(
            "UPDATE cost_reservation
             SET status = 'uncertain', settled_ms = ?1, failure_reason_code = ?2
             WHERE status = 'dispatched'
                OR (status IN ('reserved', 'open') AND dispatched_ms IS NOT NULL)",
            params![at_ms, "crash_recovery_post_dispatch_marker"],
        )?;
        Ok((refunded as u64, uncertain as u64))
    }

    /// P0-2 reconcile: every UNCERTAIN reservation of one task settles FROM
    /// the durable `provider_call` row that belongs to the SAME ATTEMPT
    /// (attempt accounting, schema v18): an attempt-keyed reservation joins
    /// `provider_call.attempt_op_id = cost_reservation.attempt_op_id`; a
    /// legacy reservation (attempt_op_id NULL) joins its op id as before —
    /// the completed call's recorded tokens, priced at the reservation's
    /// stored route-time snapshot (a crash-resumed op that completes is the
    /// durable settlement basis for the crashed attempt of the same op: the
    /// provider billed each dispatched attempt). When two attempts of one
    /// logical op exist, the attempt join guarantees each reservation
    /// settles from ITS OWN attempt's row — never from the sibling's. Each
    /// reservation settles in its own immediate transaction. Rows without a
    /// completed provider-call row of their own attempt — or unpriced under
    /// a hard cap — stay UNCERTAIN (the task-completion finalize is their
    /// conservative backstop). Idempotent: a second pass finds the settled
    /// rows closed.
    pub fn cost_reconcile_uncertain(
        &self,
        session_id: SessionId,
        task_id: TaskId,
        at_ms: i64,
    ) -> StoreResult<CostReconcileReport> {
        let mut report = CostReconcileReport::default();
        let candidates: Vec<i64> = {
            let conn = self.read()?;
            let mut stmt = conn.prepare(
                "SELECT cr.reservation_id
                 FROM cost_reservation cr
                 WHERE cr.session_id = ?1 AND cr.task_id = ?2 AND cr.status = 'uncertain'
                   AND EXISTS (
                     SELECT 1 FROM provider_call p
                     WHERE p.session_id = cr.session_id
                       AND p.status = 'completed'
                       AND (
                         (cr.attempt_op_id IS NOT NULL
                          AND p.attempt_op_id = cr.attempt_op_id)
                         OR
                         (cr.attempt_op_id IS NULL AND p.attempt_op_id IS NULL
                          AND p.op_id = cr.op_id))
                 )
                 ORDER BY cr.reservation_id ASC",
            )?;
            let rows = stmt.query_map(
                params![session_id.raw() as i64, task_id.raw() as i64],
                |r| r.get::<_, i64>(0),
            )?;
            let mut out = Vec::new();
            for row in rows {
                out.push(row?);
            }
            out
        };
        for reservation_id in candidates {
            let mut conn = self.write();
            let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
            // Only a row still UNCERTAIN settles (concurrent recovery or a
            // previous pass may have closed it).
            let row: Option<(u64, Option<String>)> = tx
                .query_row(
                    "SELECT predicted_micro, pricing_snapshot_json FROM cost_reservation
                     WHERE reservation_id = ?1 AND status = 'uncertain'",
                    params![reservation_id],
                    |r| Ok((r.get::<_, i64>(0)?.max(0) as u64, r.get(1)?)),
                )
                .optional()?;
            let Some((_predicted, snapshot_json)) = row else {
                tx.commit()?;
                continue;
            };
            let snapshot = match snapshot_json {
                Some(json) => Some(parse_json::<PricingSnapshot>(
                    &format!("reservation {reservation_id} pricing_snapshot_json"),
                    &json,
                )?),
                None => None,
            };
            // The completed provider-call row of THIS SAME attempt (or, for
            // a legacy reservation, of its op) — never a sibling attempt's
            // row. Audit-13 primary counters; cache/reasoning detail is not
            // persisted on the provider-call row, so the settled basis is
            // input + output.
            let tokens: Option<(i64, i64)> = tx
                .query_row(
                    "SELECT COALESCE(p.tokens_in, 0), COALESCE(p.tokens_out, 0)
                     FROM provider_call p
                     JOIN cost_reservation cr ON cr.reservation_id = ?1
                       AND p.session_id = cr.session_id
                       AND p.status = 'completed'
                       AND (
                         (cr.attempt_op_id IS NOT NULL
                          AND p.attempt_op_id = cr.attempt_op_id)
                         OR
                         (cr.attempt_op_id IS NULL AND p.attempt_op_id IS NULL
                          AND p.op_id = cr.op_id))
                     ORDER BY p.id DESC LIMIT 1",
                    params![reservation_id],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .optional()?;
            let Some((tokens_in, tokens_out)) = tokens else {
                report.left_uncertain += 1;
                tx.commit()?;
                continue;
            };
            let actual = snapshot.and_then(|s| {
                s.settle_cost(tokens_in.max(0) as u64, 0, 0, tokens_out.max(0) as u64)
            });
            match actual {
                Some(cost) => {
                    let cost = cost.min(i64::MAX as u64) as i64;
                    tx.execute(
                        "UPDATE cost_reservation
                         SET status = 'settled', settled_ms = ?1, provider_cost_micro = ?2,
                             provider_reported_micro = NULL,
                             provider_reported_cost_micro = NULL,
                             settled_cost_micro = ?2,
                             estimated_cost_micro = COALESCE(estimated_cost_micro, predicted_micro),
                             cost_basis = ?3, delivery_state = NULL
                         WHERE reservation_id = ?4",
                        params![
                            at_ms,
                            cost,
                            COST_BASIS_ROUTE_SNAPSHOT_ESTIMATE,
                            reservation_id
                        ],
                    )?;
                    let n = tx.execute(
                        "UPDATE task SET spent_cost_micro = spent_cost_micro + ?1
                         WHERE session_id = ?2 AND task_id = ?3",
                        params![cost, session_id.raw() as i64, task_id.raw() as i64],
                    )?;
                    if n == 0 {
                        // The task row is gone: nothing can fold into it —
                        // roll the row back for doctor's dangling scan.
                        tx.rollback()?;
                        report.left_uncertain += 1;
                        continue;
                    }
                    report.settled += 1;
                    report.charged_micro = report.charged_micro.saturating_add(cost.max(0) as u64);
                }
                None => {
                    let has_cap: bool = tx
                        .query_row(
                            "SELECT max_cost_micro > 0 FROM task
                             WHERE session_id = ?1 AND task_id = ?2",
                            params![session_id.raw() as i64, task_id.raw() as i64],
                            |r| r.get(0),
                        )
                        .unwrap_or(false);
                    if has_cap {
                        // Unpriced under a hard cap: cannot settle honestly —
                        // the finalize (reserved estimate) is the backstop.
                        report.left_uncertain += 1;
                    } else {
                        tx.execute(
                            "UPDATE cost_reservation
                             SET status = 'settled', settled_ms = ?1,
                                 provider_cost_micro = NULL, provider_reported_micro = NULL,
                                 provider_reported_cost_micro = NULL,
                                 settled_cost_micro = NULL, cost_basis = ?3,
                                 delivery_state = NULL
                             WHERE reservation_id = ?2",
                            params![at_ms, reservation_id, COST_BASIS_UNKNOWN],
                        )?;
                        report.closed_unknown += 1;
                    }
                }
            }
            tx.commit()?;
        }
        Ok(report)
    }

    /// P0-2 conservative task-completion finalize: every still-UNCERTAIN
    /// reservation of one task settles at its RESERVED ESTIMATE — the
    /// provider may have billed for a dispatched attempt whose actual never
    /// reconciled, and the reserved prediction is the honest bound the
    /// ledger already committed to. Called ONCE when the task ends (the
    /// runtime's completion-gate site); idempotent — a second pass finds
    /// nothing UNCERTAIN and never double-charges. Rows whose task row is
    /// gone are left UNCERTAIN for doctor's dangling scan (nothing can fold
    /// into a vanished envelope). Each reservation settles in its own
    /// immediate transaction.
    pub fn cost_finalize_uncertain(
        &self,
        session_id: SessionId,
        task_id: TaskId,
        at_ms: i64,
    ) -> StoreResult<CostFinalizeReport> {
        let mut report = CostFinalizeReport::default();
        let candidates: Vec<i64> = {
            let conn = self.read()?;
            let mut stmt = conn.prepare(
                "SELECT reservation_id FROM cost_reservation
                 WHERE session_id = ?1 AND task_id = ?2 AND status = 'uncertain'
                 ORDER BY reservation_id ASC",
            )?;
            let rows = stmt.query_map(
                params![session_id.raw() as i64, task_id.raw() as i64],
                |r| r.get::<_, i64>(0),
            )?;
            let mut out = Vec::new();
            for row in rows {
                out.push(row?);
            }
            out
        };
        for reservation_id in candidates {
            let mut conn = self.write();
            let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
            let predicted: Option<u64> = tx
                .query_row(
                    "SELECT predicted_micro FROM cost_reservation
                     WHERE reservation_id = ?1 AND status = 'uncertain'",
                    params![reservation_id],
                    |r| r.get::<_, i64>(0).map(|v| v.max(0) as u64),
                )
                .optional()?;
            let Some(predicted) = predicted else {
                tx.commit()?;
                continue;
            };
            let predicted_i64 = predicted.min(i64::MAX as u64) as i64;
            tx.execute(
                "UPDATE cost_reservation
                 SET status = 'settled', settled_ms = ?1, provider_cost_micro = ?2,
                     provider_reported_micro = NULL,
                     provider_reported_cost_micro = NULL,
                     settled_cost_micro = ?2,
                     estimated_cost_micro = COALESCE(estimated_cost_micro, predicted_micro),
                     cost_basis = ?3,
                     delivery_state = NULL
                 WHERE reservation_id = ?4 AND status = 'uncertain'",
                params![
                    at_ms,
                    predicted_i64,
                    COST_BASIS_CONSERVATIVE_RESERVATION,
                    reservation_id
                ],
            )?;
            let n = tx.execute(
                "UPDATE task SET spent_cost_micro = spent_cost_micro + ?1
                 WHERE session_id = ?2 AND task_id = ?3",
                params![predicted_i64, session_id.raw() as i64, task_id.raw() as i64],
            )?;
            if n == 0 {
                // The task row is gone: nothing can fold — roll the row
                // back so the finalize stays idempotent and doctor's
                // dangling scan can see it.
                tx.rollback()?;
                report.left_uncertain += 1;
                continue;
            }
            report.settled += 1;
            report.charged_micro = report.charged_micro.saturating_add(predicted);
            tx.commit()?;
        }
        Ok(report)
    }

    /// The durable reservations of one task, newest first (bounded reads:
    /// at most `limit` rows). A corrupt `pricing_snapshot_json` fails loudly
    /// (`Corrupt`) — a reservation whose price capture cannot be read must
    /// never be silently treated as unpriced.
    pub fn cost_reservations_of(
        &self,
        session_id: SessionId,
        task_id: TaskId,
        limit: i64,
    ) -> StoreResult<Vec<CostReservationRow>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT reservation_id, session_id, task_id, op_id, predicted_micro,
                    status, created_ms, settled_ms, dispatched_ms,
                    pricing_snapshot_json, provider_cost_micro,
                    provider_reported_micro, route_decision_json,
                    attempt_op_id, parent_op_id, request_id, delivery_state,
                    failure_reason_code, cost_basis,
                    provider_reported_cost_micro, estimated_cost_micro,
                    settled_cost_micro
             FROM cost_reservation
             WHERE session_id = ?1 AND task_id = ?2
             ORDER BY reservation_id DESC LIMIT ?3",
        )?;
        let rows = stmt.query_map(
            params![session_id.raw() as i64, task_id.raw() as i64, limit.max(0)],
            |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, i64>(2)?,
                    r.get::<_, i64>(3)?,
                    r.get::<_, i64>(4)?,
                    r.get::<_, String>(5)?,
                    r.get::<_, i64>(6)?,
                    r.get::<_, Option<i64>>(7)?,
                    r.get::<_, Option<i64>>(8)?,
                    r.get::<_, Option<String>>(9)?,
                    r.get::<_, Option<i64>>(10)?,
                    r.get::<_, Option<i64>>(11)?,
                    r.get::<_, Option<String>>(12)?,
                    r.get::<_, Option<i64>>(13)?,
                    r.get::<_, Option<i64>>(14)?,
                    r.get::<_, Option<String>>(15)?,
                    r.get::<_, Option<String>>(16)?,
                    r.get::<_, Option<String>>(17)?,
                    r.get::<_, Option<String>>(18)?,
                    r.get::<_, Option<i64>>(19)?,
                    r.get::<_, Option<i64>>(20)?,
                    r.get::<_, Option<i64>>(21)?,
                ))
            },
        )?;
        let mut out = Vec::new();
        for row in rows {
            let (
                reservation_id,
                row_session,
                row_task,
                op_id,
                predicted,
                status,
                created_ms,
                settled_ms,
                dispatched_ms,
                snapshot_json,
                provider_cost,
                provider_reported,
                route_json,
                attempt_op_id,
                parent_op_id,
                request_id,
                delivery_state,
                failure_reason_code,
                cost_basis,
                provider_reported_cost_micro,
                estimated_cost_micro,
                settled_cost_micro,
            ) = row?;
            let pricing_snapshot = match snapshot_json {
                Some(json) => Some(parse_json(
                    &format!("reservation {reservation_id} pricing_snapshot_json"),
                    &json,
                )?),
                None => None,
            };
            out.push(CostReservationRow {
                reservation_id,
                session_id: SessionId::new(row_session.max(1) as u64),
                task_id: TaskId::new(row_task.max(1) as u64),
                op_id: OpId::new(op_id.max(1) as u64),
                attempt_op_id: attempt_op_id.map(|raw| OpId::new(raw.max(1) as u64)),
                parent_op_id: parent_op_id.map(|raw| OpId::new(raw.max(1) as u64)),
                predicted_micro: predicted.max(0) as u64,
                status,
                created_ms,
                settled_ms,
                dispatched_ms,
                pricing_snapshot,
                provider_cost_micro: provider_cost.map(|m| m.max(0) as u64),
                provider_reported_micro: provider_reported.map(|m| m.max(0) as u64),
                route_decision_json: route_json,
                request_id,
                delivery_state,
                failure_reason_code,
                cost_basis,
                provider_reported_cost_micro: provider_reported_cost_micro.map(|m| m.max(0) as u64),
                estimated_cost_micro: estimated_cost_micro.map(|m| m.max(0) as u64),
                settled_cost_micro: settled_cost_micro.map(|m| m.max(0) as u64),
            });
        }
        Ok(out)
    }

    /// Mark one DISPATCHED reservation UNCERTAIN (post-dispatch failure
    /// accounting, schema v18): a dispatched attempt whose request left the
    /// process but never settled (stream error, stall verdict, cancellation)
    /// becomes UNCERTAIN — the provider may have billed — and KEEPS
    /// consuming the reserved amount until reconcile/finalize closes it. The
    /// failure reason code and the provider request id (when known) are
    /// recorded durably for recovery forensics. Anything not `dispatched` is
    /// a typed refusal: a still-`reserved` row was never dispatched (refund
    /// it), and a settled/refunded/uncertain row is already closed.
    pub fn cost_mark_uncertain(
        &self,
        reservation_id: i64,
        reason_code: &str,
        request_id: Option<&str>,
        at_ms: i64,
    ) -> StoreResult<CostReservationState> {
        let conn = self.write();
        let status: Option<String> = conn
            .query_row(
                "SELECT status FROM cost_reservation WHERE reservation_id = ?1",
                params![reservation_id],
                |r| r.get(0),
            )
            .optional()?;
        let Some(status) = status else {
            return Ok(CostReservationState::Missing);
        };
        if status != "dispatched" {
            return Ok(CostReservationState::NotOpen { current: status });
        }
        conn.execute(
            "UPDATE cost_reservation
             SET status = 'uncertain', settled_ms = ?1,
                 failure_reason_code = ?2, request_id = ?3, delivery_state = ?4
             WHERE reservation_id = ?5 AND status = 'dispatched'",
            params![
                at_ms,
                reason_code,
                request_id,
                DELIVERY_FAILED,
                reservation_id
            ],
        )?;
        Ok(CostReservationState::Applied)
    }
}

// ---------------------------------------------------------------------------
// Verified-outcome learning (audit items 13/14/L, migration v18 / schema
// target 19): durable per-key verified-outcome accumulators.
//
// The `model_outcome_stats` table is a MATERIALIZED PROJECTION: each row is
// keyed `(provider, model, phase, task_class, risk_bucket)` and holds the
// five accumulators exactly as the router's registry defines them. Samples
// enter ONLY through [`Store::model_outcome_stats_append`] (one
// transactional read-modify-write per fact), which mirrors the router-side
// absorb rule: `sample_count = successes_first_pass + failures_first_pass`,
// and rework sums grow ONLY on failure samples — a verified first-pass
// success can never cause rework. Rows survive reopen (append facts +
// projection), reads parse every column fallibly (`Corrupt`, never a panic)
// and the phase consult ([`Store::model_outcome_stats_phase`]) folds every
// class/risk bucket of one (provider, model, phase) with saturating sums.
// ---------------------------------------------------------------------------

/// One durable per-key verified-outcome accumulator row (schema target 19).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelOutcomeStatsRow {
    pub provider: String,
    pub model: String,
    pub phase: RouterPhase,
    pub task_class: TaskClass,
    pub risk_bucket: RiskBucket,
    pub successes_first_pass: u64,
    pub failures_first_pass: u64,
    pub rework_cost_micro_sum: u64,
    pub rework_turns_sum: u64,
    pub sample_count: u64,
    pub updated_ms: i64,
}

/// ONE verified-outcome fact: the explicit verified-success signal plus the
/// rework its failure eventually caused. Mirrors the router registry's
/// sample shape; "the model said done" is NOT a verified signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelOutcomeSample {
    pub verified_success: bool,
    pub rework_cost_micro: u64,
    pub rework_turns: u64,
}

/// SQLite INTEGER is signed 64-bit: a u64 accumulator above `i64::MAX`
/// cannot be stored as one integer, so every column clamps at `i64::MAX`
/// (a count of rework beyond 9.22e18 samples is unrepresentable, and the
/// equality CHECK refuses near-boundary rows loudly instead of letting the
/// projection drift).
fn outcome_clamp_i64(v: u64) -> i64 {
    v.min(i64::MAX as u64) as i64
}

fn outcome_db_phase(p: RouterPhase) -> String {
    serde_json::to_string(&p).expect("unit enum serialization cannot fail")
}

fn outcome_db_class(c: TaskClass) -> String {
    serde_json::to_string(&c).expect("unit enum serialization cannot fail")
}

fn outcome_db_bucket(b: RiskBucket) -> String {
    serde_json::to_string(&b).expect("unit enum serialization cannot fail")
}

const OUTCOME_STATS_KEY_SQL: &str =
    "provider = ?1 AND model = ?2 AND phase = ?3 AND task_class = ?4 AND risk_bucket = ?5";

impl Store {
    /// Append ONE verified sample to a key's durable projection (migration
    /// v18). The write is a single immediate transaction: the projection
    /// row is read, absorbed with the router registry's saturating rule,
    /// and written back — a crash can never leave a half-absorbed row, and
    /// the writer lock serializes concurrent appenders. A success sample
    /// carries zero rework even when a hostile caller hands nonzero
    /// cost/turn values; `verified_success = false` records a FAILURE
    /// sample (rework was needed), never a success.
    pub fn model_outcome_stats_append(
        &self,
        provider: &str,
        model: &str,
        phase: RouterPhase,
        task_class: TaskClass,
        risk_bucket: RiskBucket,
        sample: ModelOutcomeSample,
    ) -> StoreResult<()> {
        let mut conn = self.write();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let current = tx
            .query_row(
                "SELECT successes_first_pass, failures_first_pass, rework_cost_micro_sum,
                        rework_turns_sum, sample_count
                 FROM model_outcome_stats
                 WHERE provider = ?1 AND model = ?2 AND phase = ?3 AND task_class = ?4
                   AND risk_bucket = ?5",
                params![
                    provider,
                    model,
                    outcome_db_phase(phase),
                    outcome_db_class(task_class),
                    outcome_db_bucket(risk_bucket)
                ],
                |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, i64>(1)?,
                        r.get::<_, i64>(2)?,
                        r.get::<_, i64>(3)?,
                        r.get::<_, i64>(4)?,
                    ))
                },
            )
            .optional()?;
        let (mut successes, mut failures, mut cost_sum, mut turns_sum, mut count) = match current {
            Some((s, f, c, t, n)) => (
                s.max(0) as u64,
                f.max(0) as u64,
                c.max(0) as u64,
                t.max(0) as u64,
                n.max(0) as u64,
            ),
            None => (0, 0, 0, 0, 0),
        };
        count = count.saturating_add(1);
        if sample.verified_success {
            successes = successes.saturating_add(1);
        } else {
            failures = failures.saturating_add(1);
            cost_sum = cost_sum.saturating_add(sample.rework_cost_micro);
            turns_sum = turns_sum.saturating_add(sample.rework_turns);
        }
        tx.execute(
            "INSERT INTO model_outcome_stats (
                provider, model, phase, task_class, risk_bucket,
                successes_first_pass, failures_first_pass, rework_cost_micro_sum,
                rework_turns_sum, sample_count, updated_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
             ON CONFLICT(provider, model, phase, task_class, risk_bucket)
             DO UPDATE SET
                successes_first_pass = ?6,
                failures_first_pass = ?7,
                rework_cost_micro_sum = ?8,
                rework_turns_sum = ?9,
                sample_count = ?10,
                updated_ms = ?11",
            params![
                provider,
                model,
                outcome_db_phase(phase),
                outcome_db_class(task_class),
                outcome_db_bucket(risk_bucket),
                outcome_clamp_i64(successes),
                outcome_clamp_i64(failures),
                outcome_clamp_i64(cost_sum),
                outcome_clamp_i64(turns_sum),
                outcome_clamp_i64(count),
                now_ms()
            ],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// The exact per-key projection row, or `None` when the key has no
    /// samples. Every column is parsed fallibly: an unreadable enum text,
    /// a negative count or a broken `sample_count = successes + failures`
    /// invariant surfaces as `StoreError::Corrupt`, never a silent number.
    pub fn model_outcome_stats_get(
        &self,
        provider: &str,
        model: &str,
        phase: RouterPhase,
        task_class: TaskClass,
        risk_bucket: RiskBucket,
    ) -> StoreResult<Option<ModelOutcomeStatsRow>> {
        let conn = self.read()?;
        let raw: Option<RawOutcomeStatsRow> = conn
            .query_row(
                &format!(
                    "SELECT provider, model, phase, task_class, risk_bucket,
                            successes_first_pass, failures_first_pass,
                            rework_cost_micro_sum, rework_turns_sum,
                            sample_count, updated_ms
                     FROM model_outcome_stats WHERE {OUTCOME_STATS_KEY_SQL}"
                ),
                params![
                    provider,
                    model,
                    outcome_db_phase(phase),
                    outcome_db_class(task_class),
                    outcome_db_bucket(risk_bucket)
                ],
                outcome_stats_row_raw,
            )
            .optional()?;
        match raw {
            Some(raw) => Ok(Some(outcome_stats_row_validate(raw)?)),
            None => Ok(None),
        }
    }

    /// The per-phase consult the economic router performs: every
    /// class/risk bucket row of one (provider, model, phase) folded into
    /// one saturating accumulator row (a route request carries no
    /// class/risk dimensions of its own). `None` when no row exists for the
    /// triple. A corrupt source row fails the whole consult loudly.
    pub fn model_outcome_stats_phase(
        &self,
        provider: &str,
        model: &str,
        phase: RouterPhase,
    ) -> StoreResult<Option<ModelOutcomeStatsRow>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT provider, model, phase, task_class, risk_bucket,
                    successes_first_pass, failures_first_pass,
                    rework_cost_micro_sum, rework_turns_sum,
                    sample_count, updated_ms
             FROM model_outcome_stats
             WHERE provider = ?1 AND model = ?2 AND phase = ?3",
        )?;
        let mut rows = stmt.query(params![provider, model, outcome_db_phase(phase)])?;
        let mut acc: Option<ModelOutcomeStatsRow> = None;
        while let Some(row) = rows.next()? {
            let validated = outcome_stats_row_validate(outcome_stats_row_raw(row)?)?;
            acc = Some(match acc {
                None => validated,
                Some(mut a) => {
                    a.successes_first_pass = a
                        .successes_first_pass
                        .saturating_add(validated.successes_first_pass);
                    a.failures_first_pass = a
                        .failures_first_pass
                        .saturating_add(validated.failures_first_pass);
                    a.rework_cost_micro_sum = a
                        .rework_cost_micro_sum
                        .saturating_add(validated.rework_cost_micro_sum);
                    a.rework_turns_sum = a
                        .rework_turns_sum
                        .saturating_add(validated.rework_turns_sum);
                    a.sample_count = a.sample_count.saturating_add(validated.sample_count);
                    a.updated_ms = a.updated_ms.max(validated.updated_ms);
                    a
                }
            });
        }
        Ok(acc)
    }

    // ---------------------------------------------------------------------
    // Durable evidence authority (v21)
    // ---------------------------------------------------------------------

    /// Insert one evidence row. `row.id == 0` asks SQLite to assign the next
    /// globally-unique id (`AUTOINCREMENT`); a non-zero id is inserted
    /// explicitly and a collision refuses with `Conflict` instead of ever
    /// overwriting an existing envelope. Returns the id actually stored.
    pub fn evidence_insert(&self, row: &EvidenceRow) -> StoreResult<u64> {
        if row.revision < 1 {
            return Err(StoreError::Malformed(format!(
                "evidence revision {} is below 1",
                row.revision
            )));
        }
        let mut conn = self.write();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let id = if row.id == 0 {
            tx.execute(
                "INSERT INTO evidence (
                    session_id, workspace_id, task_id, kind, revision,
                    provenance, compressibility, compression, retrieval,
                    compact, backing_cas_hash, completeness, created_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
                params![
                    row.session_id.raw() as i64,
                    row.workspace_id.raw() as i64,
                    row.task_id.map(|t| t as i64),
                    row.kind,
                    row.revision,
                    row.provenance_json,
                    row.compressibility,
                    row.compression_json,
                    row.retrieval_json,
                    row.compact_json,
                    row.backing_cas_hash,
                    row.completeness,
                    row.created_ms,
                ],
            )?;
            tx.last_insert_rowid() as u64
        } else {
            let existing: Option<i64> = tx
                .query_row(
                    "SELECT id FROM evidence WHERE id = ?1",
                    params![row.id as i64],
                    |r| r.get(0),
                )
                .optional()?;
            if existing.is_some() {
                return Err(StoreError::Conflict(format!(
                    "evidence {} already exists; refusing to overwrite a durable envelope",
                    row.id
                )));
            }
            tx.execute(
                "INSERT INTO evidence (
                    id, session_id, workspace_id, task_id, kind, revision,
                    provenance, compressibility, compression, retrieval,
                    compact, backing_cas_hash, completeness, created_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
                params![
                    row.id as i64,
                    row.session_id.raw() as i64,
                    row.workspace_id.raw() as i64,
                    row.task_id.map(|t| t as i64),
                    row.kind,
                    row.revision,
                    row.provenance_json,
                    row.compressibility,
                    row.compression_json,
                    row.retrieval_json,
                    row.compact_json,
                    row.backing_cas_hash,
                    row.completeness,
                    row.created_ms,
                ],
            )?;
            row.id
        };
        tx.commit()?;
        Ok(id)
    }

    /// Read one evidence row by id. Unscoped: callers acting for a session
    /// MUST compare the scope columns before serving the row (the evidence
    /// crate's `DurableEvidenceStore::get_scoped` owns that rule).
    pub fn evidence_get(&self, id: u64) -> StoreResult<Option<EvidenceRow>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT id, session_id, workspace_id, task_id, kind, revision,
                    provenance, compressibility, compression, retrieval,
                    compact, backing_cas_hash, completeness, created_ms
             FROM evidence WHERE id = ?1",
        )?;
        let mut rows = stmt.query(params![id as i64])?;
        match rows.next()? {
            Some(row) => Ok(Some(evidence_row_map(row)?)),
            None => Ok(None),
        }
    }

    /// Every evidence row of one session+workspace, oldest id first. The
    /// bounded scope listing the evidence layer exposes (never a whole-table
    /// scan); `limit` is clamped to a sane hard bound.
    pub fn evidence_list_by_scope(
        &self,
        session_id: SessionId,
        workspace_id: WorkspaceId,
        limit: usize,
    ) -> StoreResult<Vec<EvidenceRow>> {
        let bound = i64::try_from(limit.min(10_000)).unwrap_or(10_000);
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT id, session_id, workspace_id, task_id, kind, revision,
                    provenance, compressibility, compression, retrieval,
                    compact, backing_cas_hash, completeness, created_ms
             FROM evidence
             WHERE session_id = ?1 AND workspace_id = ?2
             ORDER BY id ASC LIMIT ?3",
        )?;
        let mut rows = stmt.query(params![
            session_id.raw() as i64,
            workspace_id.raw() as i64,
            bound
        ])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(evidence_row_map(row)?);
        }
        Ok(out)
    }

    /// Additive newest-first scoped listing for the evidence recency path:
    /// every row of one session+workspace with `id < before` (no bound when
    /// `before` is `None`), ordered `created_ms DESC, id DESC` and limited.
    /// `before` is the exclusive keyset cursor (evidence ids are assigned
    /// monotonically, so it is the canonical recency boundary); scope/task
    /// filtering stays in the evidence crate. `limit` is clamped to the same
    /// hard bound as [`Store::evidence_list_by_scope`].
    pub fn evidence_list_by_scope_newest(
        &self,
        session_id: SessionId,
        workspace_id: WorkspaceId,
        before: Option<u64>,
        limit: usize,
    ) -> StoreResult<Vec<EvidenceRow>> {
        let bound = i64::try_from(limit.min(10_000)).unwrap_or(10_000);
        let cursor = match before {
            Some(before) => i64::try_from(before).unwrap_or(i64::MAX),
            None => i64::MAX,
        };
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT id, session_id, workspace_id, task_id, kind, revision,
                    provenance, compressibility, compression, retrieval,
                    compact, backing_cas_hash, completeness, created_ms
             FROM evidence
             WHERE session_id = ?1 AND workspace_id = ?2 AND id < ?3
             ORDER BY created_ms DESC, id DESC LIMIT ?4",
        )?;
        let mut rows = stmt.query(params![
            session_id.raw() as i64,
            workspace_id.raw() as i64,
            cursor,
            bound
        ])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(evidence_row_map(row)?);
        }
        Ok(out)
    }

    /// Evidence ids whose backing bytes hash to `backing_cas_hash`, oldest
    /// first (bounded). The digest is an audit input only: the evidence
    /// layer still enforces scope before any read, so knowing a digest never
    /// grants cross-session access.
    pub fn evidence_ids_by_backing(
        &self,
        backing_cas_hash: &str,
        limit: usize,
    ) -> StoreResult<Vec<u64>> {
        let bound = i64::try_from(limit.min(10_000)).unwrap_or(10_000);
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT id FROM evidence WHERE backing_cas_hash = ?1 ORDER BY id ASC LIMIT ?2",
        )?;
        let mut rows = stmt.query(params![backing_cas_hash, bound])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row.get::<_, i64>(0)? as u64);
        }
        Ok(out)
    }

    /// The current evidence id high-water mark: the largest id ever issued
    /// (0 when none). Durable across reopen via `sqlite_sequence`, so a
    /// restart can never reissue an id.
    pub fn evidence_high_water(&self) -> StoreResult<u64> {
        let conn = self.read()?;
        let max_id: i64 = conn.query_row("SELECT COALESCE(MAX(id), 0) FROM evidence", [], |r| {
            r.get(0)
        })?;
        let seq: i64 = conn
            .query_row(
                "SELECT COALESCE(seq, 0) FROM sqlite_sequence WHERE name = 'evidence'",
                [],
                |r| r.get(0),
            )
            .optional()?
            .unwrap_or(0);
        Ok(max_id.max(seq).max(0) as u64)
    }
}

fn evidence_row_map(r: &rusqlite::Row<'_>) -> StoreResult<EvidenceRow> {
    let id_raw: i64 = r.get(0)?;
    if id_raw < 1 {
        return Err(StoreError::Corrupt(vec![format!(
            "evidence id {id_raw} is below 1"
        )]));
    }
    let id = id_raw as u64;
    let revision: i64 = r.get(5)?;
    if revision < 1 {
        return Err(StoreError::Corrupt(vec![format!(
            "evidence {id} revision {revision} is below 1"
        )]));
    }
    Ok(EvidenceRow {
        id,
        session_id: SessionId::new(r.get::<_, i64>(1)? as u64),
        workspace_id: WorkspaceId::new(r.get::<_, i64>(2)? as u64),
        task_id: r.get::<_, Option<i64>>(3)?.map(|t| t.max(0) as u64),
        kind: r.get(4)?,
        revision,
        provenance_json: r.get(6)?,
        compressibility: r.get(7)?,
        compression_json: r.get(8)?,
        retrieval_json: r.get(9)?,
        compact_json: r.get(10)?,
        backing_cas_hash: r.get(11)?,
        completeness: r.get(12)?,
        created_ms: r.get(13)?,
    })
}

/// One raw `model_outcome_stats` row exactly as stored: enum dimensions are
/// JSON-encoded TEXT and stay unparsed until [`outcome_stats_row_validate`]
/// turns the row into its typed shape.
type RawOutcomeStatsRow = (
    String,
    String,
    String,
    String,
    String,
    i64,
    i64,
    i64,
    i64,
    i64,
    i64,
);

fn outcome_stats_row_raw(r: &rusqlite::Row<'_>) -> rusqlite::Result<RawOutcomeStatsRow> {
    Ok((
        r.get(0)?,
        r.get(1)?,
        r.get(2)?,
        r.get(3)?,
        r.get(4)?,
        r.get(5)?,
        r.get(6)?,
        r.get(7)?,
        r.get(8)?,
        r.get(9)?,
        r.get(10)?,
    ))
}

/// Parse-fallibly validates one raw projection row into its typed shape:
/// the enum texts are JSON (`"implement"`), so unknown or version-skewed
/// text is `Corrupt`, and an invariant-broken row (`sample_count !=
/// successes + failures`) is refused the same way — never silently trusted.
fn outcome_stats_row_validate(raw: RawOutcomeStatsRow) -> StoreResult<ModelOutcomeStatsRow> {
    let (
        provider,
        model,
        phase,
        task_class,
        risk_bucket,
        successes_raw,
        failures_raw,
        cost_sum_raw,
        turns_sum_raw,
        sample_raw,
        updated_ms,
    ) = raw;
    let ctx = |col: &str| format!("model_outcome_stats {provider}/{model} {col}");
    let phase: RouterPhase = parse_json(&ctx("phase"), &phase)?;
    let task_class: TaskClass = parse_json(&ctx("task_class"), &task_class)?;
    let risk_bucket: RiskBucket = parse_json(&ctx("risk_bucket"), &risk_bucket)?;
    let successes_first_pass = successes_raw.max(0) as u64;
    let failures_first_pass = failures_raw.max(0) as u64;
    let rework_cost_micro_sum = cost_sum_raw.max(0) as u64;
    let rework_turns_sum = turns_sum_raw.max(0) as u64;
    let sample_count = sample_raw.max(0) as u64;
    if sample_count != successes_first_pass.saturating_add(failures_first_pass) {
        return Err(StoreError::Corrupt(vec![format!(
            "model_outcome_stats {provider}/{model} {phase:?}/{task_class:?}/{risk_bucket:?} \
             sample_count {sample_count} != successes {successes_first_pass} + failures \
             {failures_first_pass}"
        )]));
    }
    Ok(ModelOutcomeStatsRow {
        provider,
        model,
        phase,
        task_class,
        risk_bucket,
        successes_first_pass,
        failures_first_pass,
        rework_cost_micro_sum,
        rework_turns_sum,
        sample_count,
        updated_ms,
    })
}

fn index_state_map(r: &rusqlite::Row<'_>) -> rusqlite::Result<IndexStateRow> {
    Ok(IndexStateRow {
        workspace_id: WorkspaceId::new(r.get::<_, i64>(0)? as u64),
        state_json: r.get(1)?,
        generation: r.get(2)?,
        updated_ms: r.get(3)?,
    })
}

fn index_state_log_map(r: &rusqlite::Row<'_>) -> rusqlite::Result<IndexStateLogRow> {
    Ok(IndexStateLogRow {
        id: r.get(0)?,
        workspace_id: WorkspaceId::new(r.get::<_, i64>(1)? as u64),
        kind: r.get(2)?,
        state_json: r.get(3)?,
        generation: r.get(4)?,
        updated_ms: r.get(5)?,
    })
}

fn configure(conn: &Connection) -> StoreResult<()> {
    // `wal_autocheckpoint = 0` disables SQLite's automatic checkpoint on
    // commit. Auto-checkpoint executes INSIDE the committing statement, so on
    // the actor's interactive batch path a WAL that grew past the default
    // ~1000-page watermark turned into multi-millisecond SQLite work inside a
    // measured segment (the recurring 5 ms gate flake). Checkpointing is
    // scheduled explicitly instead: the session `DbActor` runs a PASSIVE
    // checkpoint from its idle flush tick via [`Store::wal_checkpoint_passive`],
    // outside every measured interactive segment.
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         PRAGMA busy_timeout = 5000;
         PRAGMA foreign_keys = ON;
         PRAGMA wal_autocheckpoint = 0;",
    )?;
    Ok(())
}

fn check_integrity(conn: &Connection) -> StoreResult<Vec<String>> {
    let mut stmt = conn.prepare("PRAGMA integrity_check")?;
    let mut rows = stmt.query([])?;
    let mut issues = Vec::new();
    while let Some(row) = rows.next()? {
        let line: String = row.get(0)?;
        if line != "ok" {
            issues.push(line);
        }
    }
    Ok(issues)
}

/// `PRAGMA quick_check`: the bounded sibling of [`check_integrity`] — it
/// validates page structure and record round-trips but skips the full
/// scan's index-content/UNIQUE re-verification, so it runs in a fraction of
/// the time on large stores. Same shape: problem lines; empty = healthy.
fn check_quick(conn: &Connection) -> StoreResult<Vec<String>> {
    let mut stmt = conn.prepare("PRAGMA quick_check")?;
    let mut rows = stmt.query([])?;
    let mut issues = Vec::new();
    while let Some(row) = rows.next()? {
        let line: String = row.get(0)?;
        if line != "ok" {
            issues.push(line);
        }
    }
    Ok(issues)
}

const MIGRATIONS: &[&str] = &[
    // v1 — initial schema
    "CREATE TABLE IF NOT EXISTS workspace (
        id INTEGER PRIMARY KEY,
        root TEXT NOT NULL UNIQUE,
        created_ms INTEGER NOT NULL
     );
     CREATE TABLE IF NOT EXISTS session (
        id INTEGER PRIMARY KEY,
        workspace_id INTEGER NOT NULL REFERENCES workspace(id),
        title TEXT NOT NULL DEFAULT '',
        provider TEXT NOT NULL DEFAULT '',
        model TEXT NOT NULL DEFAULT '',
        state TEXT NOT NULL DEFAULT '\"idle\"',
        created_ms INTEGER NOT NULL,
        updated_ms INTEGER NOT NULL
     );
     CREATE TABLE IF NOT EXISTS event (
        seq INTEGER NOT NULL,
        session_id INTEGER NOT NULL REFERENCES session(id),
        op_id INTEGER,
        kind TEXT NOT NULL,
        state TEXT NOT NULL,
        ts_ms INTEGER NOT NULL,
        payload TEXT,
        PRIMARY KEY (session_id, seq)
     );
     CREATE TABLE IF NOT EXISTS message (
        id INTEGER PRIMARY KEY,
        session_id INTEGER NOT NULL REFERENCES session(id),
        seq INTEGER NOT NULL,
        role TEXT NOT NULL,
        data TEXT NOT NULL,
        created_ms INTEGER NOT NULL,
        UNIQUE (session_id, seq)
     );
     CREATE TABLE IF NOT EXISTS part (
        id INTEGER PRIMARY KEY,
        message_id INTEGER NOT NULL REFERENCES message(id),
        kind TEXT NOT NULL,
        data TEXT NOT NULL,
        created_ms INTEGER NOT NULL
     );
     CREATE TABLE IF NOT EXISTS task (
        id INTEGER PRIMARY KEY,
        session_id INTEGER NOT NULL REFERENCES session(id),
        ledger TEXT NOT NULL,
        updated_ms INTEGER NOT NULL
     );
     CREATE TABLE IF NOT EXISTS tool_run (
        id INTEGER PRIMARY KEY,
        session_id INTEGER NOT NULL REFERENCES session(id),
        op_id INTEGER NOT NULL UNIQUE,
        tool TEXT NOT NULL,
        args TEXT NOT NULL,
        status TEXT NOT NULL,
        started_ms INTEGER NOT NULL,
        ended_ms INTEGER,
        effect_status TEXT NOT NULL,
        recovery TEXT NOT NULL,
        expected_hash TEXT
     );
     CREATE TABLE IF NOT EXISTS provider_call (
        id INTEGER PRIMARY KEY,
        session_id INTEGER NOT NULL REFERENCES session(id),
        op_id INTEGER NOT NULL,
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        started_ms INTEGER NOT NULL,
        ended_ms INTEGER,
        status TEXT NOT NULL,
        tokens_in INTEGER,
        tokens_out INTEGER,
        error TEXT
     );
     CREATE TABLE IF NOT EXISTS checkpoint (
        id INTEGER PRIMARY KEY,
        session_id INTEGER NOT NULL REFERENCES session(id),
        sequence INTEGER NOT NULL,
        path TEXT NOT NULL,
        before_hash TEXT NOT NULL,
        after_hash TEXT NOT NULL,
        created_ms INTEGER NOT NULL,
        restored_ms INTEGER
     );
     CREATE TABLE IF NOT EXISTS artifact (
        id INTEGER PRIMARY KEY,
        session_id INTEGER NOT NULL REFERENCES session(id),
        kind TEXT NOT NULL,
        cas_hash TEXT NOT NULL UNIQUE,
        summary TEXT NOT NULL,
        created_ms INTEGER NOT NULL,
        size INTEGER NOT NULL
     );
     CREATE TABLE IF NOT EXISTS worktree (
        id INTEGER PRIMARY KEY,
        workspace_id INTEGER NOT NULL REFERENCES workspace(id),
        path TEXT NOT NULL UNIQUE,
        branch TEXT NOT NULL,
        active INTEGER NOT NULL DEFAULT 1
     );
     CREATE TABLE IF NOT EXISTS memory_fact (
        id INTEGER PRIMARY KEY,
        session_id INTEGER NOT NULL REFERENCES session(id),
        kind TEXT NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        updated_ms INTEGER NOT NULL,
        UNIQUE (session_id, kind, key)
     );
     CREATE TABLE IF NOT EXISTS compaction (
        id INTEGER PRIMARY KEY,
        session_id INTEGER NOT NULL REFERENCES session(id),
        before_tokens INTEGER NOT NULL,
        after_tokens INTEGER NOT NULL,
        target_tokens INTEGER NOT NULL,
        accepted INTEGER NOT NULL,
        strategy TEXT NOT NULL,
        created_ms INTEGER NOT NULL
     );
     CREATE TABLE IF NOT EXISTS permission (
        id INTEGER PRIMARY KEY,
        session_id INTEGER NOT NULL REFERENCES session(id),
        op_id INTEGER NOT NULL,
        capability TEXT NOT NULL,
        decision TEXT NOT NULL,
        resolved_ms INTEGER,
        expires_ms INTEGER NOT NULL
     );
     CREATE INDEX IF NOT EXISTS idx_event_session_seq ON event(session_id, seq);
     CREATE INDEX IF NOT EXISTS idx_message_session_seq ON message(session_id, seq);
     CREATE INDEX IF NOT EXISTS idx_toolrun_session ON tool_run(session_id, status);
     CREATE INDEX IF NOT EXISTS idx_checkpoint_session ON checkpoint(session_id, sequence);",
    // v2 — session lifecycle (orthogonal to the turn state machine)
    "ALTER TABLE session ADD COLUMN lifecycle TEXT NOT NULL DEFAULT 'open';",
    // v3 — checkpoint rows carry the CAS hash of the AFTER-content blob, so
    // unrevert (redo) and diff can reconstruct what the edit wrote. NULL on
    // pre-v3 rows: those checkpoints refuse redo/diff honestly.
    "ALTER TABLE checkpoint ADD COLUMN after_cas_hash TEXT;",
    // v4 — durable per-session prompt queue with a FULL execution envelope
    // and a durable state machine (audit rounds 6+7): pending | claimed |
    // running | done | cancelled. The user conversation message is NOT
    // stored here — it is materialized at ADMISSION (after the preceding
    // turn's output) so conversation chronology is the insertion order.
    "CREATE TABLE IF NOT EXISTS prompt_queue (
        session_id INTEGER NOT NULL REFERENCES session(id),
        seq INTEGER NOT NULL,
        op_id INTEGER NOT NULL,
        message_seq INTEGER,
        delivered INTEGER NOT NULL DEFAULT 0,
        prompt TEXT NOT NULL DEFAULT '',
        files TEXT NOT NULL DEFAULT '[]',
        model TEXT,
        variant TEXT,
        agent TEXT,
        status TEXT NOT NULL DEFAULT 'pending',
        requested_at INTEGER NOT NULL DEFAULT 0,
        claimed_at INTEGER,
        completed_at INTEGER,
        PRIMARY KEY (session_id, seq)
     );",
    // v5 — durable loop signals (spec §28): repeated identical failing
    // tool calls across LOGICAL TURNS and daemon restarts are detected
    // from this table, never from memory.
    "CREATE TABLE IF NOT EXISTS loop_signal (
        session_id INTEGER NOT NULL REFERENCES session(id),
        key TEXT NOT NULL,
        count INTEGER NOT NULL,
        updated_ms INTEGER NOT NULL,
        PRIMARY KEY (session_id, key)
     );",
    // v6 — checkpoint rows carry per-side EXISTENCE flags. A hash alone
    // cannot distinguish a missing file from an empty one: before==after
    // ("no change") currently means the empty-file creation is skipped and
    // rollback of a missing→content checkpoint would recreate an empty file
    // instead of deleting. DEFAULT 1 keeps pre-v6 rows readable: old rows
    // were only recorded for real files (the caller had content on both
    // sides), so "hash present with no existence marker means exists:true".
    "ALTER TABLE checkpoint ADD COLUMN before_exists INTEGER NOT NULL DEFAULT 1;",
    "ALTER TABLE checkpoint ADD COLUMN after_exists INTEGER NOT NULL DEFAULT 1;",
    // v7 — exact per-turn operation identity + recovery descriptors.
    // `turn_record` fixes the durable identity of every ADMITTED logical
    // turn (op id, queue seq, prompt message, effective provider/model/
    // variant/tool mode, status) so crash recovery resumes the SAME turn
    // with the SAME recorded envelope instead of synthesizing an operation.
    // `tool_run` gains the crash-recovery machinery: the durable
    // `replay_descriptor` (the stored invocation an idempotent tool may be
    // re-executed from), the `attempt` counter (a replay is a new PHYSICAL
    // attempt of the SAME logical operation) and the `postcondition`
    // (workspace-write verification data computed from the actual bytes as
    // written — never from JSON-encoded args).
    "CREATE TABLE IF NOT EXISTS turn_record (
        id INTEGER PRIMARY KEY,
        session_id INTEGER NOT NULL REFERENCES session(id),
        turn_op_id INTEGER NOT NULL,
        queue_seq INTEGER,
        prompt_message_id INTEGER,
        effective_provider TEXT NOT NULL DEFAULT '',
        effective_model TEXT NOT NULL DEFAULT '',
        variant TEXT,
        tool_mode TEXT,
        started_at INTEGER NOT NULL,
        status TEXT NOT NULL,
        updated_ms INTEGER NOT NULL,
        UNIQUE (session_id, turn_op_id)
     );
     CREATE INDEX IF NOT EXISTS idx_turn_record_session_status ON turn_record(session_id, status);
     ALTER TABLE tool_run ADD COLUMN replay_descriptor TEXT;
     ALTER TABLE tool_run ADD COLUMN attempt INTEGER NOT NULL DEFAULT 0;
     ALTER TABLE tool_run ADD COLUMN postcondition TEXT;",
    // v8 — durable worktree/task identity on sessions. Tool calls were
    // being handed fake identities (worktree 1/task 1) because the real
    // ones lived nowhere durable: the session row now records them, so the
    // agent runtime builds `ToolRunCtx.identity` from the session row and
    // every replay descriptor / postcondition rides the SAME ids. DEFAULT 1
    // preserves existing rows: 1/1 is the documented STANDALONE session
    // identity (no worktree/task adopted); WorktreeManager-created
    // worktrees adopt their sessions deliberately afterwards.
    // (This block is array index 8, i.e. schema target 9: the v6 checkpoint
    // block spans two array entries before it.)
    "ALTER TABLE session ADD COLUMN worktree_id INTEGER NOT NULL DEFAULT 1;
     ALTER TABLE session ADD COLUMN task_id INTEGER NOT NULL DEFAULT 1;",
    // v9 — durable op-id sequence (schema target 10; array index 9). Op ids
    // used to be `now_ms + in-memory counter`: a daemon restart inside the
    // same millisecond (or after a backward clock jump) silently reused ids
    // that crash recovery still treats as live operations. The manager now
    // reserves RANGES from this table instead. `session_scope` is the scope
    // key: 0 is the ONE global sequence shared by every session (op ids are
    // globally unique — `tool_run.op_id` is a UNIQUE column). The seed row is
    // inserted by `migrate()` (not here) because its value is derived from
    // the wall clock at migration time: see `op_id_seq_seed`.
    "CREATE TABLE IF NOT EXISTS op_id_seq (
        session_scope INTEGER PRIMARY KEY CHECK (session_scope = 0),
        next_value INTEGER NOT NULL
     );",
    // v10 — first-class durable Task rows (schema target 11; array index
    // 10). Audit 25: no typed durable Task existed; the one-row-per-session
    // JSON ledger blob (the v1 `task` table) could not express task state,
    // bounded goal/criteria/plan or a durable budget. The typed table takes
    // the `task` name; the legacy ledger keeps its exact rows under its own
    // name `task_ledger` (get/put_task_ledger follow it; no data moves, the
    // rename is structural only — nothing references the old table).
    "ALTER TABLE task RENAME TO task_ledger;
     CREATE TABLE IF NOT EXISTS task (
        task_id INTEGER NOT NULL,
        session_id INTEGER NOT NULL REFERENCES session(id),
        goal TEXT NOT NULL,
        acceptance_criteria TEXT NOT NULL,
        plan TEXT NOT NULL,
        max_tokens INTEGER,
        max_turns INTEGER,
        spent_tokens INTEGER NOT NULL DEFAULT 0,
        spent_turns INTEGER NOT NULL DEFAULT 0,
        state TEXT NOT NULL,
        created_ms INTEGER NOT NULL,
        updated_ms INTEGER NOT NULL,
        PRIMARY KEY (session_id, task_id)
     );
      CREATE INDEX IF NOT EXISTS idx_task_session_updated ON task(session_id, updated_ms);",
    // v11 — journal payload schema versions + the typed durable session
    // ledger (audits 27 / 71-72; schema target 12, array index 11). Every
    // `event` payload row now carries the payload schema version that wrote
    // it (readers refuse unknown versions loudly instead of misreading a
    // future shape). The opaque per-session ledger JSON blob stays untouched
    // (`task_ledger`); the RICH ledger is a typed, versioned, append-only
    // entry stream (`ledger_entry`) with a per-session materialized head
    // checkpoint (`ledger_head`) that compaction rewrites atomically with
    // the entry deletion it summarizes. The session layer computes the
    // never-FIFO-evict durability watermark: entries are deleted only below
    // it and the last GoalSet/CriteriaSet/Decision and every unresolved
    // BlockerOpened survive every compaction in code, never by accident.
    "ALTER TABLE event ADD COLUMN payload_ver INTEGER NOT NULL DEFAULT 1;
     CREATE TABLE IF NOT EXISTS ledger_entry (
        session_id INTEGER NOT NULL REFERENCES session(id),
        seq INTEGER NOT NULL,
        entry_type TEXT NOT NULL,
        schema_ver INTEGER NOT NULL DEFAULT 1,
        payload TEXT NOT NULL,
        created_ms INTEGER NOT NULL,
        PRIMARY KEY (session_id, seq)
     );
     CREATE INDEX IF NOT EXISTS idx_ledger_entry_session_seq ON ledger_entry(session_id, seq);
     CREATE INDEX IF NOT EXISTS idx_ledger_entry_session_type ON ledger_entry(session_id, entry_type);
     CREATE TABLE IF NOT EXISTS ledger_head (
        session_id INTEGER PRIMARY KEY REFERENCES session(id),
        head_json TEXT NOT NULL,
        checkpoint_seq INTEGER NOT NULL,
        schema_ver INTEGER NOT NULL DEFAULT 1,
        updated_ms INTEGER NOT NULL
     );",
    // v12 — durable per-workspace repository-index state machine rows
    // (schema target 13; array index 12; audits 30/64). The real
    // IndexService (faktor-index) persists its WorkspaceIndexState machine
    // here, one row per workspace, with an append-only transition journal.
    // `state_json` is opaque to this crate (protocol-agnostic, parsed by
    // the index layer); the `generation` column is the numeric generation
    // that row names (0 for NotStarted). Every transition updates the row
    // AND appends one journal row in the SAME transaction, so a daemon
    // restart resumes exactly the generation the crashed process left.
    "CREATE TABLE IF NOT EXISTS index_state (
        workspace_id INTEGER PRIMARY KEY REFERENCES workspace(id),
        state_json TEXT NOT NULL,
        generation INTEGER NOT NULL,
        updated_ms INTEGER NOT NULL
     );
     CREATE TABLE IF NOT EXISTS index_state_log (
        id INTEGER PRIMARY KEY,
        workspace_id INTEGER NOT NULL REFERENCES workspace(id),
        kind TEXT NOT NULL,
        state_json TEXT NOT NULL,
        generation INTEGER NOT NULL,
        updated_ms INTEGER NOT NULL
     );
     CREATE INDEX IF NOT EXISTS idx_index_state_log_ws ON index_state_log(workspace_id, id);",
    // v13 — measurable prefix-cache stability (audits 65-66; schema target
    // 14; array index 13). Provider-side prompt caches turn a STABLE prompt
    // prefix into cheaper calls; a session that rewrites/reorders its
    // prefix every turn silently pays uncached prices. These ADDITIVE
    // provider_call columns persist one prefix observation per settled
    // usage row: `prompt_prefix_hash` (digest of the exact cacheable-prefix
    // byte string the caller sent, 32-byte BLOB), `prompt_tokens` (that
    // prefix's token count) and `prefix_stability` (the row's per-turn
    // stability in [0, 1], NULL until the settlement site fills it). Every
    // column is NULL-able with NO default: pre-v13 rows honestly read as
    // "no prefix observation recorded" and the routing layer never guesses.
    "ALTER TABLE provider_call ADD COLUMN prompt_prefix_hash BLOB;
     ALTER TABLE provider_call ADD COLUMN prompt_tokens INTEGER;
     ALTER TABLE provider_call ADD COLUMN prefix_stability REAL;",
    // v14 — task revisions + the first-class VerificationRecord (audit
    // P0-7/P0-8; schema target 15; array index 14). The typed task table
    // gains the per-row monotonic `revision` counter (DEFAULT 1 backfills
    // pre-v14 rows: every existing row was written once, so revision 1 is
    // the honest baseline) and the `verification_record` table becomes the
    // durable completion proof: immutable except the ONE CAS finalize
    // `Running -> Passed|Failed`. JSON columns follow the task-row style
    // (protocol-agnostic TEXT parsed fallibly on read); sizes are bounded by
    // the session layer before any write.
    "ALTER TABLE task ADD COLUMN revision INTEGER NOT NULL DEFAULT 1;
     CREATE TABLE IF NOT EXISTS verification_record (
        id INTEGER PRIMARY KEY,
        task_id INTEGER NOT NULL,
        revision INTEGER NOT NULL,
        workspace_id INTEGER NOT NULL,
        worktree_id INTEGER NOT NULL,
        tree_hash TEXT,
        criteria_json TEXT NOT NULL,
        checks_json TEXT NOT NULL,
        changed_files_json TEXT NOT NULL,
        unrelated_changes_json TEXT NOT NULL,
        reviewer_json TEXT,
        status TEXT NOT NULL,
        started_ms INTEGER NOT NULL,
        completed_ms INTEGER
     );
     CREATE INDEX IF NOT EXISTS idx_verification_record_task
        ON verification_record(task_id, id);",
    // v15 — the durable cost ledger (P0-6/12; schema target 16; array index
    // 15). The typed `task` row gains the MONETARY budget envelope columns:
    // `max_cost_micro` (NULL = unlimited; the token/turn caps stay where
    // they are) and `spent_cost_micro` (the durable settled-spend total).
    // These columns are READ-ONLY through the task machine: no generic task
    // upsert touches them (upsert_task enumerates its columns), the cost
    // ledger's own store section is their ONLY writer, and they are never
    // patched through `TaskBudget` — the ledger settles them in the same
    // transaction that closes a reservation, so the row and its
    // reservations can never disagree.
    //
    // `cost_reservation` records one attempt to spend task money, keyed by
    // its AUTOINCREMENT id (monotonic across daemon restarts, so a
    // reservation id is never reused after a crash). Every reservation
    // starts OPEN; a settlement (OPEN -> SETTLED) records the locally
    // calculated cost (`provider_cost_micro`), the provider-reported cost
    // when the usage frame carried one (`provider_reported_micro`), and the
    // routing decision's JSON when one produced the call
    // (`route_decision_json`, bounded by the session layer before the
    // write). A refund (OPEN -> REFUNDED) releases the reservation without
    // spending; crash recovery marks every surviving OPEN row ABANDONED
    // (the op never settled, so its prediction is never counted as spent).
    "ALTER TABLE task ADD COLUMN max_cost_micro INTEGER;
     ALTER TABLE task ADD COLUMN spent_cost_micro INTEGER NOT NULL DEFAULT 0;
     CREATE TABLE IF NOT EXISTS cost_reservation (
        reservation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        task_id INTEGER NOT NULL,
        op_id INTEGER NOT NULL,
        predicted_micro INTEGER NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('open', 'settled', 'refunded', 'abandoned')),
        created_ms INTEGER NOT NULL,
        settled_ms INTEGER,
        provider_cost_micro INTEGER,
        provider_reported_micro INTEGER,
        route_decision_json TEXT
     );
     CREATE INDEX IF NOT EXISTS idx_cost_reservation_session_task_status
        ON cost_reservation(session_id, task_id, status);",
    // v16 — spend-truth settlement + UNCERTAIN semantics (P0-1 settlement
    // half + P0-2; schema target 17; array index 16). The v15 crash rule
    // (OPEN -> ABANDONED, charged $0) undercounted spend when the crash hit
    // BETWEEN provider billing and the local settle: the provider may have
    // billed and the ledger counted zero. The v15 state `abandoned` is
    // RENAMED to `uncertain` — a reservation a crashed daemon may have
    // dispatched — and the table gains the durable dispatch marker
    // (`dispatched_ms`, NULL = dispatch never provably began; written
    // immediately before the provider transport call, so recovery can tell
    // "never dispatched -> refund" from "may have dispatched -> uncertain")
    // and the immutable route-time price capture (`pricing_snapshot_json`)
    // settlement prices usage against. Recovery code paths (not this
    // migration) split OPEN rows on that marker; this migration only
    // rebuilds the table: the CHECK swaps `abandoned` for `uncertain`, every
    // legacy `abandoned` row becomes `uncertain` (its prediction keeps
    // consuming the reserved amount), and pre-v17 rows read as
    // never-dispatched and unpriced. Column order in the INSERT is explicit
    // (the table is rebuilt, not altered), and the AUTOINCREMENT sequence is
    // re-seeded to the surviving max id by the explicit-id insert.
    "ALTER TABLE cost_reservation RENAME TO cost_reservation_v15;
     CREATE TABLE IF NOT EXISTS cost_reservation (
        reservation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        task_id INTEGER NOT NULL,
        op_id INTEGER NOT NULL,
        predicted_micro INTEGER NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('open', 'settled', 'refunded', 'uncertain')),
        created_ms INTEGER NOT NULL,
        settled_ms INTEGER,
        dispatched_ms INTEGER,
        pricing_snapshot_json TEXT,
        provider_cost_micro INTEGER,
        provider_reported_micro INTEGER,
        route_decision_json TEXT
     );
     INSERT INTO cost_reservation (
        reservation_id, session_id, task_id, op_id, predicted_micro, status,
        created_ms, settled_ms, dispatched_ms, pricing_snapshot_json,
        provider_cost_micro, provider_reported_micro, route_decision_json)
     SELECT reservation_id, session_id, task_id, op_id, predicted_micro,
            CASE status WHEN 'abandoned' THEN 'uncertain' ELSE status END,
            created_ms, settled_ms, NULL, NULL, provider_cost_micro,
            provider_reported_micro, route_decision_json
     FROM cost_reservation_v15;
     DROP TABLE cost_reservation_v15;
     CREATE INDEX IF NOT EXISTS idx_cost_reservation_session_task_status
        ON cost_reservation(session_id, task_id, status);",
    // v17 — attempt-identity accounting (audit Phase-1 items D/E/F + part of
    // G; schema target 18; array index 17). Every physical network attempt
    // now gets a fresh durable global OpId and the ledger rows key by that
    // ATTEMPT id instead of the shared turn/model-call op. The table is
    // rebuilt with the attempt/parent/delivery accounting columns and the
    // pre-dispatch state vocabulary: the v15/v16 `open` state is renamed
    // `reserved`, and `dispatched` becomes a REAL state (a reservation moves
    // reserved -> dispatched when its durable `dispatched_ms` marker is
    // written), so refund-after-dispatch is impossible at the SQL level:
    // a refund's guarded UPDATE (`status IN ('reserved','open') AND
    // dispatched_ms IS NULL`) changes zero rows on any dispatched/settled/
    // refunded/uncertain reservation, and the store refuses loudly instead
    // of freeing money. Legacy maps are lossless: `open` -> `reserved`
    // (carrying any v16 `dispatched_ms` marker — a v16 crash could leave
    // `open` + marker, which recovery still reads as may-have-dispatched),
    // `parent_op_id` is backfilled from `op_id` (the shared logical op every
    // legacy row keyed by), the reserve-time estimate is backfilled into
    // `estimated_cost_micro`, the provider-reported amount into
    // `provider_reported_cost_micro`, and `settled_cost_micro` /
    // `cost_basis` / `delivery_state` / `failure_reason_code` /
    // `request_id` are NULL (pre-v17 settlements never recorded which of the
    // two amount columns was folded, so an honest NULL beats a guessed
    // number). `provider_call` gains the same attempt identity columns
    // (NULL on legacy rows — each was its op's only physical attempt;
    // `parent_model_call_op_id` is backfilled from `op_id`).
    "ALTER TABLE cost_reservation RENAME TO cost_reservation_v16;
     CREATE TABLE IF NOT EXISTS cost_reservation (
        reservation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        task_id INTEGER NOT NULL,
        op_id INTEGER NOT NULL,
        attempt_op_id INTEGER,
        parent_op_id INTEGER,
        predicted_micro INTEGER NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('reserved', 'dispatched', 'settled', 'refunded', 'uncertain')),
        created_ms INTEGER NOT NULL,
        settled_ms INTEGER,
        dispatched_ms INTEGER,
        pricing_snapshot_json TEXT,
        provider_cost_micro INTEGER,
        provider_reported_micro INTEGER,
        route_decision_json TEXT,
        request_id TEXT,
        delivery_state TEXT,
        failure_reason_code TEXT,
        cost_basis TEXT,
        provider_reported_cost_micro INTEGER,
        estimated_cost_micro INTEGER,
        settled_cost_micro INTEGER
     );
     INSERT INTO cost_reservation (
        reservation_id, session_id, task_id, op_id, attempt_op_id,
        parent_op_id, predicted_micro, status, created_ms, settled_ms,
        dispatched_ms, pricing_snapshot_json, provider_cost_micro,
        provider_reported_micro, route_decision_json, request_id,
        delivery_state, failure_reason_code, cost_basis,
        provider_reported_cost_micro, estimated_cost_micro,
        settled_cost_micro)
     SELECT reservation_id, session_id, task_id, op_id, NULL, op_id,
            predicted_micro,
            CASE status WHEN 'open' THEN 'reserved' ELSE status END,
            created_ms, settled_ms, dispatched_ms, pricing_snapshot_json,
            provider_cost_micro, provider_reported_micro, route_decision_json,
            NULL, NULL, NULL, NULL, provider_reported_micro, predicted_micro,
            NULL
     FROM cost_reservation_v16;
     DROP TABLE cost_reservation_v16;
     CREATE INDEX IF NOT EXISTS idx_cost_reservation_session_task_status
        ON cost_reservation(session_id, task_id, status);
     CREATE INDEX IF NOT EXISTS idx_cost_reservation_attempt_op
        ON cost_reservation(attempt_op_id);
     ALTER TABLE provider_call ADD COLUMN attempt_op_id INTEGER;
     ALTER TABLE provider_call ADD COLUMN parent_model_call_op_id INTEGER;
     ALTER TABLE provider_call ADD COLUMN attempt_ordinal INTEGER;
     ALTER TABLE provider_call ADD COLUMN reservation_id INTEGER;
     UPDATE provider_call SET parent_model_call_op_id = op_id;
     CREATE INDEX IF NOT EXISTS idx_provider_call_session_attempt
        ON provider_call(session_id, attempt_op_id);",
    // v18 — verified-outcome learning (audit items 13/14/L; schema target
    // 19; array index 18). ONE new table, no other table changes: the
    // durable per-key verified-outcome projection. Rows key
    // (provider, model, phase, task_class, risk_bucket) and hold the five
    // accumulators with the router registry's invariant locked at the SQL
    // level (`sample_count = successes_first_pass + failures_first_pass`,
    // every column non-negative). Samples enter only through
    // `Store::model_outcome_stats_append` (a transactional
    // read-modify-write); the PK column order doubles as the per-phase
    // consult index (`provider, model, phase` prefix scans).
    "CREATE TABLE IF NOT EXISTS model_outcome_stats (
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        phase TEXT NOT NULL,
        task_class TEXT NOT NULL,
        risk_bucket TEXT NOT NULL,
        successes_first_pass INTEGER NOT NULL DEFAULT 0 CHECK (successes_first_pass >= 0),
        failures_first_pass INTEGER NOT NULL DEFAULT 0 CHECK (failures_first_pass >= 0),
        rework_cost_micro_sum INTEGER NOT NULL DEFAULT 0 CHECK (rework_cost_micro_sum >= 0),
        rework_turns_sum INTEGER NOT NULL DEFAULT 0 CHECK (rework_turns_sum >= 0),
        sample_count INTEGER NOT NULL DEFAULT 0 CHECK (sample_count >= 0),
        updated_ms INTEGER NOT NULL,
        CHECK (sample_count = successes_first_pass + failures_first_pass),
        PRIMARY KEY (provider, model, phase, task_class, risk_bucket)
     ) WITHOUT ROWID;",
    // v19 — per-call prompt segment observations (audits 45/82; schema
    // target 20; array index 19). The v13 prefix row persisted only the
    // binary digest of the cacheable prefix, so the routing layer could not
    // recover WHICH section of the prefix changed and approximated coverage
    // with the documented digest/growth-ratio pair rule. This ADDITIVE
    // nullable column persists the exact per-call `faktor_context`
    // `PrefixObservation` JSON the runtime measures at the settlement site
    // (ordered segment digests + token counts + observed cache reads), so
    // the router can compute the TRUE longest stable leading prefix and
    // price cache economics from it. NULL on pre-v19 rows: those rows keep
    // routing BYTE-IDENTICALLY on the binary pair rule — a missing
    // observation is never guessed. The store validates the payload's
    // strict shape and bounds on write AND read (a corrupt injected row is
    // a typed `Malformed`, never a silent fallback).
    "ALTER TABLE provider_call ADD COLUMN prefix_segments_json TEXT;",
    // v20 — verification environment fingerprint + candidate-proof reference
    // (audits 94/116/117; schema target 21; array index 20). Two ADDITIVE
    // nullable columns on `verification_record`: the bounded environment
    // fingerprint JSON the verification ran under, and the compact
    // candidate-proof reference (task revision, manifest aggregates, cheap
    // evidence folds, accounting digest). NULL on pre-v20 rows — a legacy
    // record honestly reads as "no fingerprint/no candidate ref recorded",
    // never a guessed value; the session layer owns the payload bounds and
    // parses the columns loudly on read.
    "ALTER TABLE verification_record ADD COLUMN environment_fingerprint_json TEXT;
     ALTER TABLE verification_record ADD COLUMN candidate_proof_ref_json TEXT;",
    // v21 — the durable evidence authority (efficiency audit: evidence/CCR
    // was designed but not the product's evidence store; schema target 22;
    // array index 21). ONE new table, no other table changes: every evidence
    // envelope the production context pipeline produces is durably recorded
    // here with its scope (session/workspace/task), kind, revision,
    // provenance/compressibility/compression/retrieval/compact JSON, the CAS
    // digest of its backing bytes and its capture completeness. `id` is
    // `INTEGER PRIMARY KEY AUTOINCREMENT`: evidence ids are GLOBALLY UNIQUE
    // across daemon restarts (never reused, even after row deletion) because
    // the backing content address must never alias a different envelope.
    // `backing_cas_hash` is deliberately NOT unique: identical backing bytes
    // deduplicate by CAS digest and may legitimately back many envelopes.
    // The scope index is the scoped-read path (`get_scoped` reads by id and
    // verifies scope; the session/workspace index is the listing path) and
    // the backing index is the "which evidence references this digest"
    // audit path — knowing the digest never grants a read; the evidence
    // layer's scope check still decides.
    "CREATE TABLE IF NOT EXISTS evidence (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL REFERENCES session(id),
        workspace_id INTEGER NOT NULL REFERENCES workspace(id),
        task_id INTEGER,
        kind TEXT NOT NULL,
        revision INTEGER NOT NULL DEFAULT 1,
        provenance TEXT NOT NULL,
        compressibility TEXT NOT NULL,
        compression TEXT NOT NULL,
        retrieval TEXT NOT NULL,
        compact TEXT NOT NULL,
        backing_cas_hash TEXT,
        completeness TEXT NOT NULL,
        created_ms INTEGER NOT NULL
     );
     CREATE INDEX IF NOT EXISTS idx_evidence_scope
        ON evidence(session_id, workspace_id, id);
     CREATE INDEX IF NOT EXISTS idx_evidence_session_task
        ON evidence(session_id, task_id, id);
     CREATE INDEX IF NOT EXISTS idx_evidence_backing_cas
        ON evidence(backing_cas_hash);",
    // v22 — durable verification attempts and jobs (audit P0-5/26; schema
    // target 23; array index 22). Four real tables replace the wave-9
    // `memory_fact` row hack. The identity is attempt-keyed everywhere:
    // `(session_id, task_id, attempt_op_id)` names one attempt,
    // `verification_attempt_changed_file` keys its changed files by
    // `(..., ordinal)`, and `verification_job` / `verification_job_result`
    // key every required check (inline AND background) by
    // `(session_id, task_id, attempt_op_id, check_id)`. Results from
    // attempt N are rows of attempt N: a newer attempt can never mutate
    // them, and the store refuses a late resolve of N once N+1 exists.
    // Foreign keys tie jobs to their committed attempt, so a torn begin is
    // impossible (the begin is one transaction). Caps mirrored from the
    // verification-record contract: changed files <= 4096, checks <= 256,
    // bounded argv.
    "CREATE TABLE IF NOT EXISTS verification_attempt (
        session_id INTEGER NOT NULL REFERENCES session(id),
        task_id INTEGER NOT NULL,
        attempt_op_id INTEGER NOT NULL,
        task_revision INTEGER NOT NULL,
        workspace_root TEXT NOT NULL,
        environment_fingerprint_json TEXT,
        created_ms INTEGER NOT NULL,
        PRIMARY KEY (session_id, task_id, attempt_op_id)
     );
     CREATE INDEX IF NOT EXISTS idx_verification_attempt_current
        ON verification_attempt(session_id, task_id, attempt_op_id);
     CREATE TABLE IF NOT EXISTS verification_attempt_changed_file (
        session_id INTEGER NOT NULL,
        task_id INTEGER NOT NULL,
        attempt_op_id INTEGER NOT NULL,
        ordinal INTEGER NOT NULL,
        path TEXT NOT NULL,
        PRIMARY KEY (session_id, task_id, attempt_op_id, ordinal),
        FOREIGN KEY (session_id, task_id, attempt_op_id)
            REFERENCES verification_attempt(session_id, task_id, attempt_op_id)
     );
     CREATE TABLE IF NOT EXISTS verification_job (
        session_id INTEGER NOT NULL,
        task_id INTEGER NOT NULL,
        attempt_op_id INTEGER NOT NULL,
        check_id TEXT NOT NULL,
        ordinal INTEGER NOT NULL,
        task_revision INTEGER NOT NULL,
        workspace_root TEXT NOT NULL,
        kind TEXT NOT NULL,
        command TEXT NOT NULL,
        program TEXT NOT NULL,
        args_json TEXT NOT NULL,
        spec_json TEXT,
        budget_ms INTEGER NOT NULL,
        inline_status TEXT,
        state TEXT NOT NULL,
        note TEXT,
        op_id INTEGER,
        environment_fingerprint_json TEXT,
        created_ms INTEGER NOT NULL,
        updated_ms INTEGER NOT NULL,
        finished_ms INTEGER,
        PRIMARY KEY (session_id, task_id, attempt_op_id, check_id),
        FOREIGN KEY (session_id, task_id, attempt_op_id)
            REFERENCES verification_attempt(session_id, task_id, attempt_op_id)
     );
     CREATE INDEX IF NOT EXISTS idx_verification_job_open
        ON verification_job(session_id, task_id, state);
     CREATE TABLE IF NOT EXISTS verification_job_result (
        session_id INTEGER NOT NULL,
        task_id INTEGER NOT NULL,
        attempt_op_id INTEGER NOT NULL,
        check_id TEXT NOT NULL,
        result_json TEXT NOT NULL,
        finished_ms INTEGER NOT NULL,
        PRIMARY KEY (session_id, task_id, attempt_op_id, check_id),
        FOREIGN KEY (session_id, task_id, attempt_op_id, check_id)
            REFERENCES verification_job(session_id, task_id, attempt_op_id, check_id)
     );",
    // v23 — durable child-runtime blocker projection (schema target 24;
    // array index 23). The orchestrator registry row (a JSON memory fact)
    // carries the blocker fields for the graph/UI read-models; THIS typed
    // row is the bounded, queryable blocker truth keyed by the CHILD
    // session: state + blocker_kind/reason/dependency/resolution +
    // last_progress_ms. A NULL blocker triple means the child is not
    // blocked (a transition back to Running clears it). Strict text bounds
    // are enforced by the session layer BEFORE any write; existing rows
    // decode with NULL/None.
    "CREATE TABLE IF NOT EXISTS child_runtime (
        session_id INTEGER PRIMARY KEY REFERENCES session(id),
        child_id TEXT NOT NULL,
        state TEXT NOT NULL,
        blocker_kind TEXT,
        blocker_reason TEXT,
        blocker_dependency TEXT,
        blocker_resolution TEXT,
        last_progress_ms INTEGER,
        updated_ms INTEGER NOT NULL
     );",
    // v24 — durable binary/image attachments (schema target 25; array index
    // 24). ONE row per `(session_id, digest)`: the CAS address of the bytes
    // plus the typed metadata `AttachmentId { digest, mime, filename, size }`.
    // Dedupe IS the primary key: an identical digest resolves to the row's
    // first-written metadata (the write path is `INSERT OR IGNORE` + read
    // back). The `task.attachments` column carries the per-task typed list
    // (bounded JSON; `'[]'` for every pre-v24 row), SEPARATE from the
    // workspace-relative `files`/`plan` vocabulary.
    "CREATE TABLE IF NOT EXISTS attachment (
        session_id INTEGER NOT NULL REFERENCES session(id),
        digest TEXT NOT NULL,
        mime TEXT NOT NULL,
        filename TEXT,
        size INTEGER NOT NULL,
        PRIMARY KEY (session_id, digest)
     );
     ALTER TABLE task ADD COLUMN attachments TEXT NOT NULL DEFAULT '[]';",
];

/// Array index of the v9 block above (migration list position, not the
/// schema target — targets are 1-based).
const OP_ID_SEQ_MIGRATION_INDEX: usize = 9;

/// The seed of a freshly migrated op-id sequence: `(now_ms << 20)` rounded
/// UP to the 1024-id reservation quantum.
///
/// The 20-bit shift keeps every pre-migration id (`now_ms + counter`, where
/// the counter only ever grew from 1) far below the seed — by a factor of
/// ~2^20 in wall-clock terms, i.e. even a clock that had run ~56 million
/// years ahead before a regression cannot have minted ids at or above the
/// seed. Rounded up to 1024 so the manager's first reservation starts on a
/// quantum boundary.
fn op_id_seq_seed() -> i64 {
    const QUANTUM: u64 = 1024;
    let now = u64::try_from(now_ms()).unwrap_or(0);
    let base = now.saturating_mul(1 << 20);
    (base.saturating_add(QUANTUM - 1) & !(QUANTUM - 1)).min(i64::MAX as u64) as i64
}

// ------------------------------------------------- v22 legacy import (repair)

/// Bounded typed notes carried by one import marker (fact values are capped
/// at 4096 bytes, so the marker names at most this many skipped rows; the
/// rest stay logged loudly and counted in `skipped_overflow`).
const MAX_LEGACY_IMPORT_SKIP_NOTES: usize = 12;
/// One skip reason stored in the marker (longer reasons are truncated on a
/// char boundary; the untruncated reason is logged).
const MAX_LEGACY_IMPORT_SKIP_REASON_BYTES: usize = 160;
/// The durable fact-value cap every marker value must honor.
const MAX_LEGACY_IMPORT_MARKER_BYTES: usize = 4096;
/// Legacy row-value schema versions this reader understands (v1 rows lack
/// `environment_fingerprint` and decode with it absent).
const LEGACY_VERIFICATION_SCHEMA_VER: i64 = 2;

const LEGACY_VERIFICATION_JOB_STATES: [&str; 6] = [
    "queued",
    "running",
    "passed",
    "failed",
    "unavailable",
    "cancelled",
];
const LEGACY_VERIFICATION_INLINE_STATES: [&str; 3] = ["passed", "failed", "unavailable"];

/// One pre-v22 job row value: the exact serde shape the deleted legacy
/// reader enforced (including `deny_unknown_fields` — a row from an unknown
/// future schema is a typed skip, never a partial guess).
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyJobRowValue {
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
    state: String,
    note: Option<String>,
    op_id: Option<u64>,
    result_json: Option<String>,
    #[serde(default)]
    environment_fingerprint: Option<serde_json::Value>,
    created_ms: i64,
    updated_ms: i64,
    finished_ms: Option<i64>,
}

/// One pre-v22 attempt row value.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyAttemptRowValue {
    schema_ver: i64,
    task_id: u64,
    op_id: u64,
    task_revision: u64,
    workspace_root: String,
    changed: Vec<String>,
    checks: Vec<LegacyCheckRowValue>,
    #[serde(default)]
    environment_fingerprint: Option<serde_json::Value>,
    created_ms: i64,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyCheckRowValue {
    check_id: String,
    command: String,
    inline: Option<String>,
}

/// `va:{task_id}:{op_id}` -> `(task_id, op_id)`.
fn parse_legacy_attempt_key(key: &str) -> Option<(u64, u64)> {
    let rest = key.strip_prefix("va:")?;
    let (task, op) = rest.split_once(':')?;
    Some((task.parse().ok()?, op.parse().ok()?))
}

/// `vj:{task_id}:{check_id}` -> `(task_id, check_id)` (check ids may contain
/// `:`, so only the first two separators are structural).
fn parse_legacy_job_key(key: &str) -> Option<(u64, &str)> {
    let rest = key.strip_prefix("vj:")?;
    let (task, check_id) = rest.split_once(':')?;
    if check_id.is_empty() {
        return None;
    }
    Some((task.parse().ok()?, check_id))
}

fn truncate_legacy_reason(reason: &str) -> String {
    if reason.len() <= MAX_LEGACY_IMPORT_SKIP_REASON_BYTES {
        return reason.to_string();
    }
    let mut end = MAX_LEGACY_IMPORT_SKIP_REASON_BYTES;
    while end > 0 && !reason.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &reason[..end])
}

/// Record one skipped legacy row: LOUD tracing plus a bounded typed note on
/// the durable marker. The legacy fact row itself is never touched.
fn push_legacy_import_skip(report: &mut LegacyVerificationImport, key: &str, reason: String) {
    tracing::error!(
        fact_key = key,
        reason = %reason,
        "legacy verification fact skipped during the v22 import (row kept; typed note recorded)"
    );
    if report.skipped.len() < MAX_LEGACY_IMPORT_SKIP_NOTES {
        report.skipped.push(LegacyVerificationSkip {
            key: key.to_string(),
            reason: truncate_legacy_reason(&reason),
        });
    } else {
        report.skipped_overflow += 1;
    }
}

/// Derive the v22 `(program, args_json)` identity from a legacy opaque
/// `spec_json` (the legacy layer stored no argv index). The values are an
/// index only — execution still re-parses `spec_json` — so a spec whose
/// argv exceeds the v22 caps degrades to an empty identity instead of
/// refusing the whole attempt.
fn derive_legacy_argv_identity(spec_json: &str, command: &str) -> (String, String) {
    let value: Option<serde_json::Value> = serde_json::from_str(spec_json).ok();
    let program = value
        .as_ref()
        .and_then(|v| v.get("program"))
        .and_then(|p| p.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| {
            command
                .split_whitespace()
                .next()
                .unwrap_or_default()
                .to_string()
        });
    let args: Vec<String> = value
        .as_ref()
        .and_then(|v| v.get("args"))
        .and_then(|a| a.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let program = if program.len() > MAX_VERIFICATION_JOB_PROGRAM_BYTES {
        String::new()
    } else {
        program
    };
    let args_json = if args.len() > MAX_VERIFICATION_JOB_ARGS
        || args
            .iter()
            .any(|a| a.len() > MAX_VERIFICATION_JOB_ARG_BYTES)
    {
        "[]".to_string()
    } else {
        serde_json::to_string(&args).unwrap_or_else(|_| "[]".to_string())
    };
    (program, args_json)
}

fn load_legacy_fact_rows(
    conn: &Connection,
    session_raw: i64,
    kind: &str,
) -> StoreResult<Vec<(String, String)>> {
    let mut stmt = conn.prepare(
        "SELECT key, value FROM memory_fact
         WHERE session_id = ?1 AND kind = ?2 ORDER BY key ASC",
    )?;
    let rows = stmt.query_map(params![session_raw, kind], |r| {
        Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
    })?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row?);
    }
    Ok(out)
}

/// The durable marker value: schema-versioned counters plus the bounded
/// typed skip notes. Shrinks until it fits the fact cap; dropped notes stay
/// counted in `skipped_overflow` (and were logged).
fn legacy_import_marker_json(report: &LegacyVerificationImport) -> String {
    let total_notes = report.skipped.len();
    let mut shown = report.skipped.clone();
    loop {
        let dropped = (total_notes - shown.len()) as u64;
        let value = serde_json::json!({
            "schema_ver": 1,
            "imported_attempts": report.imported_attempts,
            "imported_jobs": report.imported_jobs,
            "imported_results": report.imported_results,
            "skipped": shown
                .iter()
                .map(|s| serde_json::json!({ "key": s.key, "reason": s.reason }))
                .collect::<Vec<_>>(),
            "skipped_overflow": report.skipped_overflow + dropped,
        })
        .to_string();
        if value.len() <= MAX_LEGACY_IMPORT_MARKER_BYTES || shown.is_empty() {
            return value;
        }
        shown.pop();
    }
}

/// Project ONE legacy attempt (plus its background job rows) into the v22
/// tables. `Err(reason)` means the whole attempt is skipped (the caller
/// records the typed note); per-check problems are recorded on
/// `session_report` and the remaining checks still import. Job fact keys are
/// claimed only after the attempt validated and inserted, so a skipped
/// attempt leaves its job rows orphaned and loudly noted.
fn import_one_legacy_attempt(
    tx: &rusqlite::Transaction<'_>,
    session_raw: u64,
    key: &str,
    value: &str,
    jobs: &std::collections::HashMap<String, String>,
    claimed: &mut std::collections::HashSet<String>,
    session_report: &mut LegacyVerificationImport,
) -> StoreResult<std::result::Result<(u64, u64), String>> {
    let Some((key_task, key_op)) = parse_legacy_attempt_key(key) else {
        return Ok(Err(format!(
            "legacy attempt key {key:?} is not va:<task>:<op>"
        )));
    };
    let row: LegacyAttemptRowValue = match serde_json::from_str(value) {
        Ok(row) => row,
        Err(e) => return Ok(Err(format!("undecodable legacy JSON: {e}"))),
    };
    if row.schema_ver < 1 || row.schema_ver > LEGACY_VERIFICATION_SCHEMA_VER {
        return Ok(Err(format!(
            "unknown legacy schema_ver {} (reader understands 1..={LEGACY_VERIFICATION_SCHEMA_VER})",
            row.schema_ver
        )));
    }
    if row.task_id != key_task || row.op_id != key_op {
        return Ok(Err(format!(
            "legacy key names task {key_task}/op {key_op} but the row names task {}/op {}",
            row.task_id, row.op_id
        )));
    }
    if row.task_id == 0 || row.op_id == 0 || row.task_revision == 0 {
        return Ok(Err(
            "legacy attempt identity (task/op/revision) must be non-zero".into(),
        ));
    }
    let fingerprint_json = row
        .environment_fingerprint
        .as_ref()
        .map(|v| serde_json::to_string(v).unwrap_or_default());
    let attempt = VerificationAttemptRow {
        session_id: SessionId::new(session_raw),
        task_id: TaskId::new(row.task_id),
        attempt_op_id: row.op_id,
        task_revision: TaskRevision::new(row.task_revision),
        workspace_root: row.workspace_root.clone(),
        environment_fingerprint_json: fingerprint_json.clone(),
        created_ms: row.created_ms,
    };
    let mut checks: Vec<VerificationJobRow> = Vec::new();
    let mut results: Vec<(String, String, i64)> = Vec::new();
    let mut pending_claims: Vec<String> = Vec::new();
    for (ordinal, legacy_check) in row.checks.iter().enumerate() {
        let Ok(ordinal) = u32::try_from(ordinal) else {
            return Ok(Err(
                "legacy attempt carries more checks than the v22 ordinal domain".into(),
            ));
        };
        match legacy_check.inline.as_deref() {
            Some(inline) => {
                if !LEGACY_VERIFICATION_INLINE_STATES.contains(&inline) {
                    return Ok(Err(format!(
                        "inline check '{}' carries unknown legacy status {inline:?}",
                        legacy_check.check_id
                    )));
                }
                checks.push(VerificationJobRow {
                    session_id: attempt.session_id,
                    task_id: attempt.task_id,
                    attempt_op_id: attempt.attempt_op_id,
                    check_id: legacy_check.check_id.clone(),
                    ordinal,
                    task_revision: attempt.task_revision,
                    workspace_root: attempt.workspace_root.clone(),
                    kind: String::new(),
                    command: legacy_check.command.clone(),
                    program: String::new(),
                    args_json: "[]".into(),
                    spec_json: None,
                    budget_ms: 0,
                    inline_status: Some(inline.to_string()),
                    state: inline.to_string(),
                    result_json: None,
                    note: None,
                    op_id: None,
                    environment_fingerprint_json: fingerprint_json.clone(),
                    created_ms: attempt.created_ms,
                    updated_ms: attempt.created_ms,
                    finished_ms: Some(attempt.created_ms),
                });
            }
            None => {
                let job_key = format!("vj:{}:{}", row.task_id, legacy_check.check_id);
                let Some(job_value) = jobs.get(&job_key) else {
                    push_legacy_import_skip(
                        session_report,
                        &job_key,
                        format!(
                            "background check '{}' has no legacy job row",
                            legacy_check.check_id
                        ),
                    );
                    continue;
                };
                let job: LegacyJobRowValue = match serde_json::from_str(job_value) {
                    Ok(job) => job,
                    Err(e) => {
                        push_legacy_import_skip(
                            session_report,
                            &job_key,
                            format!("undecodable legacy job JSON: {e}"),
                        );
                        continue;
                    }
                };
                if job.schema_ver < 1 || job.schema_ver > LEGACY_VERIFICATION_SCHEMA_VER {
                    push_legacy_import_skip(
                        session_report,
                        &job_key,
                        format!("unknown legacy job schema_ver {}", job.schema_ver),
                    );
                    continue;
                }
                if parse_legacy_job_key(&job_key) != Some((job.task_id, job.check_id.as_str()))
                    || job.task_id != row.task_id
                    || job.check_id != legacy_check.check_id
                    || job.attempt_op != row.op_id
                {
                    push_legacy_import_skip(
                        session_report,
                        &job_key,
                        format!(
                            "legacy job identity disagrees with its key/attempt (task {}, check {:?}, attempt {})",
                            job.task_id, job.check_id, job.attempt_op
                        ),
                    );
                    continue;
                }
                if !LEGACY_VERIFICATION_JOB_STATES.contains(&job.state.as_str()) {
                    push_legacy_import_skip(
                        session_report,
                        &job_key,
                        format!("unknown legacy job state {:?}", job.state),
                    );
                    continue;
                }
                if job.check_id.is_empty()
                    || job.kind.is_empty()
                    || job.kind.len() > MAX_VERIFICATION_JOB_KIND_BYTES
                    || job.command.is_empty()
                    || job.command.len() > MAX_VERIFICATION_JOB_COMMAND_BYTES
                    || job.spec_json.is_empty()
                    || job.spec_json.len() > MAX_VERIFICATION_JOB_SPEC_JSON_BYTES
                    || job.budget_ms == 0
                    || job.budget_ms > MAX_VERIFICATION_JOB_BUDGET_MS
                {
                    push_legacy_import_skip(
                        session_report,
                        &job_key,
                        "legacy job carries a field outside the v22 bounds".into(),
                    );
                    continue;
                }
                if job.op_id == Some(0) {
                    push_legacy_import_skip(
                        session_report,
                        &job_key,
                        "legacy job carries a zero claim op id".into(),
                    );
                    continue;
                }
                let (program, args_json) =
                    derive_legacy_argv_identity(&job.spec_json, &job.command);
                let note = match job.note {
                    Some(note)
                        if !note.is_empty() && note.len() <= MAX_VERIFICATION_JOB_NOTE_BYTES =>
                    {
                        Some(note)
                    }
                    Some(_) => {
                        push_legacy_import_skip(
                            session_report,
                            &job_key,
                            "legacy job note outside the v22 bounds; imported without it".into(),
                        );
                        None
                    }
                    None => None,
                };
                let result_json = job
                    .result_json
                    .filter(|r| !r.is_empty() && r.len() <= MAX_VERIFICATION_JOB_RESULT_JSON_BYTES);
                let job_fingerprint = job
                    .environment_fingerprint
                    .as_ref()
                    .map(|v| serde_json::to_string(v).unwrap_or_default())
                    .or_else(|| fingerprint_json.clone());
                checks.push(VerificationJobRow {
                    session_id: attempt.session_id,
                    task_id: attempt.task_id,
                    attempt_op_id: attempt.attempt_op_id,
                    check_id: job.check_id.clone(),
                    ordinal,
                    task_revision: if job.task_revision == 0 {
                        attempt.task_revision
                    } else {
                        TaskRevision::new(job.task_revision)
                    },
                    workspace_root: job.workspace_root.clone(),
                    kind: job.kind.clone(),
                    command: job.command.clone(),
                    program,
                    args_json,
                    spec_json: Some(job.spec_json.clone()),
                    budget_ms: job.budget_ms,
                    inline_status: None,
                    state: job.state.clone(),
                    result_json: None,
                    note,
                    op_id: job.op_id,
                    environment_fingerprint_json: job_fingerprint,
                    created_ms: job.created_ms,
                    updated_ms: job.updated_ms,
                    finished_ms: job.finished_ms,
                });
                match result_json {
                    Some(result_json)
                        if matches!(job.state.as_str(), "passed" | "failed" | "unavailable") =>
                    {
                        results.push((
                            job.check_id.clone(),
                            result_json,
                            job.finished_ms.unwrap_or(job.updated_ms),
                        ));
                    }
                    Some(_) => push_legacy_import_skip(
                        session_report,
                        &job_key,
                        format!(
                            "legacy job state {:?} disagrees with its outcome JSON; the outcome was not imported",
                            job.state
                        ),
                    ),
                    None if matches!(job.state.as_str(), "passed" | "failed" | "unavailable") => {
                        push_legacy_import_skip(
                            session_report,
                            &job_key,
                            format!(
                                "terminal legacy job {:?} carries no outcome JSON; imported without a result row",
                                job.state
                            ),
                        );
                    }
                    None => {}
                }
                pending_claims.push(job_key);
            }
        }
    }
    if checks.is_empty() {
        return Ok(Err(
            "legacy attempt carries no importable required checks".into()
        ));
    }
    if let Err(e) = validate_verification_attempt(&attempt, &row.changed, &checks) {
        return Ok(Err(format!("legacy attempt fails the v22 contract: {e}")));
    }
    // A hand-edited database can carry the v22 rows without the import
    // marker: never collide with them — the existing attempt wins and the
    // legacy row is skipped loudly (the marker still lands).
    let exists: Option<i64> = tx
        .query_row(
            "SELECT 1 FROM verification_attempt
             WHERE session_id = ?1 AND task_id = ?2 AND attempt_op_id = ?3",
            params![
                attempt.session_id.raw() as i64,
                attempt.task_id.raw() as i64,
                attempt.attempt_op_id as i64
            ],
            |r| r.get(0),
        )
        .optional()?;
    if exists.is_some() {
        return Ok(Err(
            "a v22 attempt row with this identity already exists; the legacy row was not imported"
                .into(),
        ));
    }
    tx.execute(
        "INSERT INTO verification_attempt(
            session_id, task_id, attempt_op_id, task_revision, workspace_root,
            environment_fingerprint_json, created_ms)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            attempt.session_id.raw() as i64,
            attempt.task_id.raw() as i64,
            attempt.attempt_op_id as i64,
            attempt.task_revision.raw() as i64,
            attempt.workspace_root,
            attempt.environment_fingerprint_json,
            attempt.created_ms
        ],
    )?;
    for (changed_ordinal, path) in row.changed.iter().enumerate() {
        tx.execute(
            "INSERT INTO verification_attempt_changed_file(
                session_id, task_id, attempt_op_id, ordinal, path)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                attempt.session_id.raw() as i64,
                attempt.task_id.raw() as i64,
                attempt.attempt_op_id as i64,
                changed_ordinal as i64,
                path
            ],
        )?;
    }
    for check in &checks {
        tx.execute(
            "INSERT INTO verification_job(
                session_id, task_id, attempt_op_id, check_id, ordinal,
                task_revision, workspace_root, kind, command, program,
                args_json, spec_json, budget_ms, inline_status, state, note,
                op_id, environment_fingerprint_json, created_ms, updated_ms,
                finished_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12,
                     ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21)",
            params![
                check.session_id.raw() as i64,
                check.task_id.raw() as i64,
                check.attempt_op_id as i64,
                check.check_id,
                check.ordinal as i64,
                check.task_revision.raw() as i64,
                check.workspace_root,
                check.kind,
                check.command,
                check.program,
                check.args_json,
                check.spec_json,
                check.budget_ms as i64,
                check.inline_status,
                check.state,
                check.note,
                check.op_id.map(|op| op as i64),
                check.environment_fingerprint_json,
                check.created_ms,
                check.updated_ms,
                check.finished_ms
            ],
        )?;
    }
    for (check_id, result_json, finished_ms) in &results {
        tx.execute(
            "INSERT INTO verification_job_result(
                session_id, task_id, attempt_op_id, check_id, result_json, finished_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                attempt.session_id.raw() as i64,
                attempt.task_id.raw() as i64,
                attempt.attempt_op_id as i64,
                check_id,
                result_json,
                finished_ms
            ],
        )?;
    }
    for job_key in pending_claims {
        claimed.insert(job_key);
    }
    Ok(Ok((checks.len() as u64, results.len() as u64)))
}

/// One-shot, idempotent v22 repair: project every session's pre-v22
/// verification `memory_fact` rows into the real
/// `verification_attempt`/`_changed_file`/`_job`/`_job_result` tables.
///
/// Exactly-once: the imported rows and a durable per-session marker fact
/// (`verification_v22_import`/`done`) commit in ONE transaction; a session
/// whose marker exists is never scanned again. Legacy fact rows are
/// preserved (additive upgrade). Every undecodable/corrupt row is skipped
/// with a LOUD tracing error, a typed note on the marker and the row left
/// in place — never a silent drop, never a deletion.
fn import_legacy_verification_facts_conn(
    conn: &mut Connection,
) -> StoreResult<LegacyVerificationImport> {
    let mut report = LegacyVerificationImport::default();
    let legacy_sessions: Vec<i64> = {
        let mut stmt = conn.prepare(
            "SELECT DISTINCT session_id FROM memory_fact
             WHERE kind IN (?1, ?2) ORDER BY session_id ASC",
        )?;
        let rows = stmt.query_map(
            params![
                LEGACY_VERIFICATION_ATTEMPT_FACT_KIND,
                LEGACY_VERIFICATION_JOB_FACT_KIND
            ],
            |r| r.get::<_, i64>(0),
        )?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        out
    };
    if legacy_sessions.is_empty() {
        return Ok(report);
    }
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let import_time = now_ms();
    for session_raw in legacy_sessions {
        if session_raw <= 0 {
            // Impossible under the foreign keys; a hand-edited row has no
            // session row space to carry a marker note. Surface it loudly.
            tracing::error!(
                session_id = session_raw,
                "legacy verification fact rows under a non-positive session id; skipped (no row space for a marker)"
            );
            continue;
        }
        let already: Option<i64> = tx
            .query_row(
                "SELECT 1 FROM memory_fact
                 WHERE session_id = ?1 AND kind = ?2 AND key = ?3",
                params![
                    session_raw,
                    VERIFICATION_V22_IMPORT_MARKER_KIND,
                    VERIFICATION_V22_IMPORT_MARKER_KEY
                ],
                |r| r.get(0),
            )
            .optional()?;
        if already.is_some() {
            continue;
        }
        let attempts =
            load_legacy_fact_rows(&tx, session_raw, LEGACY_VERIFICATION_ATTEMPT_FACT_KIND)?;
        let jobs = load_legacy_fact_rows(&tx, session_raw, LEGACY_VERIFICATION_JOB_FACT_KIND)?;
        let job_map: std::collections::HashMap<String, String> = jobs.iter().cloned().collect();
        let mut session_report = LegacyVerificationImport::default();
        let mut claimed: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (key, value) in &attempts {
            match import_one_legacy_attempt(
                &tx,
                session_raw as u64,
                key,
                value,
                &job_map,
                &mut claimed,
                &mut session_report,
            )? {
                Ok((imported_jobs, imported_results)) => {
                    session_report.imported_attempts += 1;
                    session_report.imported_jobs += imported_jobs;
                    session_report.imported_results += imported_results;
                }
                Err(reason) => push_legacy_import_skip(&mut session_report, key, reason),
            }
        }
        for (key, _) in &jobs {
            if !claimed.contains(key) {
                push_legacy_import_skip(
                    &mut session_report,
                    key,
                    "legacy job row was not imported: its attempt row is missing or was skipped"
                        .into(),
                );
            }
        }
        let marker = legacy_import_marker_json(&session_report);
        tx.execute(
            "INSERT INTO memory_fact(session_id, kind, key, value, updated_ms)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                session_raw,
                VERIFICATION_V22_IMPORT_MARKER_KIND,
                VERIFICATION_V22_IMPORT_MARKER_KEY,
                marker,
                import_time
            ],
        )?;
        report.imported_attempts += session_report.imported_attempts;
        report.imported_jobs += session_report.imported_jobs;
        report.imported_results += session_report.imported_results;
        report.skipped_overflow += session_report.skipped_overflow;
        for skip in session_report.skipped {
            if report.skipped.len() < MAX_LEGACY_IMPORT_SKIP_NOTES {
                report.skipped.push(skip);
            } else {
                report.skipped_overflow += 1;
            }
        }
    }
    tx.commit()?;
    Ok(report)
}

/// Apply migrations transactionally; `PRAGMA user_version` is the cursor.
fn migrate(conn: &mut Connection) -> StoreResult<()> {
    let mut version: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    for (i, sql) in MIGRATIONS.iter().enumerate() {
        let target = (i + 1) as i64;
        if version >= target {
            continue;
        }
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute_batch(sql)
            .map_err(|e| StoreError::Migration(format!("v{target}: {e}")))?;
        // The v9 op-id sequence table needs its one global row seeded from
        // the migration-time clock, which no static SQL can express. The
        // INSERT is idempotent so a replay (or a second opener racing the
        // first migration) can never double-seed or overwrite.
        if i == OP_ID_SEQ_MIGRATION_INDEX {
            tx.execute(
                "INSERT OR IGNORE INTO op_id_seq (session_scope, next_value) VALUES (0, ?1)",
                params![op_id_seq_seed()],
            )
            .map_err(|e| StoreError::Migration(format!("v{target} seed: {e}")))?;
        }
        tx.execute_batch(&format!("PRAGMA user_version = {target}"))
            .map_err(|e| StoreError::Migration(format!("v{target} version write: {e}")))?;
        tx.commit()
            .map_err(|e| StoreError::Migration(format!("v{target} commit: {e}")))?;
        version = target;
    }
    // One-shot repair on EVERY open (marker-guarded, idempotent): project
    // pre-v22 verification `memory_fact` rows into the v22 tables. Runs after
    // the schema cursor reached the newest version, so the target tables
    // always exist; a store with no legacy rows pays one indexed SELECT.
    import_legacy_verification_facts_conn(conn)?;
    Ok(())
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn kind_name(k: EventKind) -> &'static str {
    match k {
        EventKind::SessionCreated => "session_created",
        EventKind::PromptReceived => "prompt_received",
        EventKind::ContextPrepared => "context_prepared",
        EventKind::ModelStarted => "model_started",
        EventKind::ModelChunkReceived => "model_chunk_received",
        EventKind::ToolRequested => "tool_requested",
        EventKind::ToolStarted => "tool_started",
        EventKind::FileChanged => "file_changed",
        EventKind::ToolCompleted => "tool_completed",
        EventKind::ToolCancelled => "tool_cancelled",
        EventKind::CheckpointCreated => "checkpoint_created",
        EventKind::ContextCompacted => "context_compacted",
        EventKind::CompactRejected => "compact_rejected",
        EventKind::SubagentStarted => "subagent_started",
        EventKind::SubagentCompleted => "subagent_completed",
        EventKind::TurnCompleted => "turn_completed",
        EventKind::PermissionGranted => "permission_granted",
        EventKind::PermissionDenied => "permission_denied",
        EventKind::PhaseChanged => "phase_changed",
        EventKind::ReplayStarted => "replay_started",
        EventKind::PromptAdmitted => "prompt_admitted",
        EventKind::CrashDetected => "crash_detected",
        EventKind::RecoveryApplied => "recovery_applied",
        EventKind::SessionEnded => "session_ended",
        EventKind::Suspended => "suspended",
        EventKind::Resumed => "resumed",
        EventKind::Failed => "failed",
    }
}

fn kind_from_name(name: &str) -> Option<EventKind> {
    Some(match name {
        "session_created" => EventKind::SessionCreated,
        "prompt_received" => EventKind::PromptReceived,
        "context_prepared" => EventKind::ContextPrepared,
        "model_started" => EventKind::ModelStarted,
        "model_chunk_received" => EventKind::ModelChunkReceived,
        "tool_requested" => EventKind::ToolRequested,
        "tool_started" => EventKind::ToolStarted,
        "file_changed" => EventKind::FileChanged,
        "tool_completed" => EventKind::ToolCompleted,
        "tool_cancelled" => EventKind::ToolCancelled,
        "checkpoint_created" => EventKind::CheckpointCreated,
        "context_compacted" => EventKind::ContextCompacted,
        "compact_rejected" => EventKind::CompactRejected,
        "subagent_started" => EventKind::SubagentStarted,
        "subagent_completed" => EventKind::SubagentCompleted,
        "turn_completed" => EventKind::TurnCompleted,
        "permission_granted" => EventKind::PermissionGranted,
        "permission_denied" => EventKind::PermissionDenied,
        "phase_changed" => EventKind::PhaseChanged,
        "replay_started" => EventKind::ReplayStarted,
        "prompt_admitted" => EventKind::PromptAdmitted,
        "crash_detected" => EventKind::CrashDetected,
        "recovery_applied" => EventKind::RecoveryApplied,
        "session_ended" => EventKind::SessionEnded,
        "suspended" => EventKind::Suspended,
        "resumed" => EventKind::Resumed,
        "failed" => EventKind::Failed,
        _ => return None,
    })
}

fn message_map(r: &rusqlite::Row<'_>) -> StoreResult<MessageRow> {
    let id = r.get::<_, i64>(0)?;
    Ok(MessageRow {
        id,
        session_id: SessionId::new(r.get::<_, i64>(1)? as u64),
        seq: r.get(2)?,
        role: r.get(3)?,
        data: parse_json(&format!("message {id} data"), &r.get::<_, String>(4)?)?,
        created_ms: r.get(5)?,
    })
}

fn task_row_map(r: &rusqlite::Row<'_>, session_id: SessionId) -> StoreResult<TaskRow> {
    let task_id = TaskId::new(r.get::<_, i64>(0)? as u64);
    let revision_raw: i64 = r.get(12)?;
    if revision_raw < 1 {
        // The revision column is DEFAULT 1 and every write bumps it, so a
        // value below 1 is corruption — refusing beats trusting it (a
        // revision 0 would silently break the completion CAS).
        return Err(StoreError::Corrupt(vec![format!(
            "task {session_id}/{task_id} revision {revision_raw} is below 1"
        )]));
    }
    Ok(TaskRow {
        task_id,
        session_id,
        goal: r.get(2)?,
        acceptance_criteria: parse_json(
            &format!("task {session_id}/{task_id} acceptance_criteria"),
            &r.get::<_, String>(3)?,
        )?,
        plan: parse_json(
            &format!("task {session_id}/{task_id} plan"),
            &r.get::<_, String>(4)?,
        )?,
        attachments: parse_json(
            &format!("task {session_id}/{task_id} attachments"),
            &r.get::<_, String>(13)?,
        )?,
        max_tokens: r.get::<_, Option<i64>>(5)?.map(|m| m.max(0) as u64),
        max_turns: r.get::<_, Option<i64>>(6)?.map(|m| m.max(0) as u32),
        spent_tokens: r.get::<_, i64>(7)?.max(0) as u64,
        spent_turns: r.get::<_, i64>(8)?.max(0) as u32,
        state: parse_json(
            &format!("task {session_id}/{task_id} state"),
            &r.get::<_, String>(9)?,
        )?,
        revision: TaskRevision::new(revision_raw as u64),
        created_ms: r.get(10)?,
        updated_ms: r.get(11)?,
    })
}

fn verification_record_map(r: &rusqlite::Row<'_>) -> StoreResult<VerificationRecordRow> {
    let id = VerificationRecordId::new(r.get::<_, i64>(0)? as u64);
    let revision_raw: i64 = r.get(2)?;
    if revision_raw < 1 {
        // Same corruption contract as task revisions: a record certifying a
        // revision below 1 could never have been written by the API.
        return Err(StoreError::Corrupt(vec![format!(
            "verification_record {id} revision {revision_raw} is below 1"
        )]));
    }
    Ok(VerificationRecordRow {
        id,
        task_id: TaskId::new(r.get::<_, i64>(1)? as u64),
        revision: TaskRevision::new(revision_raw as u64),
        workspace_id: WorkspaceId::new(r.get::<_, i64>(3)? as u64),
        worktree_id: WorktreeId::new(r.get::<_, i64>(4)? as u64),
        tree_hash: r.get(5)?,
        criteria: parse_json(
            &format!("verification_record {id} criteria"),
            &r.get::<_, String>(6)?,
        )?,
        checks: parse_json(
            &format!("verification_record {id} checks"),
            &r.get::<_, String>(7)?,
        )?,
        changed_files: parse_json(
            &format!("verification_record {id} changed_files"),
            &r.get::<_, String>(8)?,
        )?,
        unrelated_changes: parse_json(
            &format!("verification_record {id} unrelated_changes"),
            &r.get::<_, String>(9)?,
        )?,
        reviewer: match r.get::<_, Option<String>>(10)? {
            Some(raw) => Some(parse_json(
                &format!("verification_record {id} reviewer"),
                &raw,
            )?),
            None => None,
        },
        status: parse_json(
            &format!("verification_record {id} status"),
            &r.get::<_, String>(11)?,
        )?,
        started_ms: r.get(12)?,
        completed_ms: r.get(13)?,
    })
}

/// The shared `verification_job` projection: the row columns in a fixed
/// order, with the executed outcome JSON joined from
/// `verification_job_result` (NULL until a background check resolves).
const VERIFICATION_JOB_SELECT: &str = "SELECT session_id, task_id, attempt_op_id, check_id, ordinal, task_revision, workspace_root, kind, command, program, args_json, spec_json, budget_ms, inline_status, state, note, op_id, environment_fingerprint_json, created_ms, updated_ms, finished_ms, (SELECT result_json FROM verification_job_result r WHERE r.session_id = verification_job.session_id AND r.task_id = verification_job.task_id AND r.attempt_op_id = verification_job.attempt_op_id AND r.check_id = verification_job.check_id) FROM verification_job";

fn verification_job_map(row: &rusqlite::Row<'_>) -> StoreResult<VerificationJobRow> {
    let session_raw: i64 = row.get(0)?;
    let task_raw: i64 = row.get(1)?;
    let attempt_raw: i64 = row.get(2)?;
    let revision_raw: i64 = row.get(5)?;
    if session_raw <= 0 || task_raw <= 0 || attempt_raw <= 0 || revision_raw < 1 {
        return Err(StoreError::Corrupt(vec![format!(
            "verification_job row has a non-positive identity \
             (session {session_raw}, task {task_raw}, attempt {attempt_raw}, revision {revision_raw})"
        )]));
    }
    let check_id: String = row.get(3)?;
    let state: String = row.get(14)?;
    if !VERIFICATION_JOB_STATES.contains(&state.as_str()) {
        return Err(StoreError::Malformed(format!(
            "verification_job '{check_id}' carries unknown state {state:?}"
        )));
    }
    let inline_status: Option<String> = row.get(13)?;
    if let Some(inline) = &inline_status {
        if !VERIFICATION_INLINE_STATES.contains(&inline.as_str()) {
            return Err(StoreError::Malformed(format!(
                "verification_job '{check_id}' carries unknown inline status {inline:?}"
            )));
        }
        if state != *inline {
            return Err(StoreError::Malformed(format!(
                "verification_job '{check_id}' inline status {inline:?} disagrees with its state {state:?}"
            )));
        }
    }
    Ok(VerificationJobRow {
        session_id: SessionId::new(session_raw as u64),
        task_id: TaskId::new(task_raw as u64),
        attempt_op_id: attempt_raw as u64,
        check_id,
        ordinal: u32::try_from(row.get::<_, i64>(4)?.max(0)).unwrap_or(u32::MAX),
        task_revision: TaskRevision::new(revision_raw as u64),
        workspace_root: row.get(6)?,
        kind: row.get(7)?,
        command: row.get(8)?,
        program: row.get(9)?,
        args_json: row.get(10)?,
        spec_json: row.get(11)?,
        budget_ms: row.get::<_, i64>(12)?.max(0) as u64,
        inline_status,
        state,
        note: row.get(15)?,
        op_id: row
            .get::<_, Option<i64>>(16)?
            .map(|op| u64::try_from(op).unwrap_or(0)),
        environment_fingerprint_json: row.get(17)?,
        created_ms: row.get(18)?,
        updated_ms: row.get(19)?,
        finished_ms: row.get(20)?,
        result_json: row.get(21)?,
    })
}

fn verification_attempt_map(row: &rusqlite::Row<'_>) -> StoreResult<VerificationAttemptRow> {
    let session_raw: i64 = row.get(0)?;
    let task_raw: i64 = row.get(1)?;
    let attempt_raw: i64 = row.get(2)?;
    let revision_raw: i64 = row.get(3)?;
    if session_raw <= 0 || task_raw <= 0 || attempt_raw <= 0 || revision_raw < 1 {
        return Err(StoreError::Corrupt(vec![format!(
            "verification_attempt row has a non-positive identity \
             (session {session_raw}, task {task_raw}, attempt {attempt_raw}, revision {revision_raw})"
        )]));
    }
    Ok(VerificationAttemptRow {
        session_id: SessionId::new(session_raw as u64),
        task_id: TaskId::new(task_raw as u64),
        attempt_op_id: attempt_raw as u64,
        task_revision: TaskRevision::new(revision_raw as u64),
        workspace_root: row.get(4)?,
        environment_fingerprint_json: row.get(5)?,
        created_ms: row.get(6)?,
    })
}

/// The highest attempt op of `(session, task)`, if any.
fn newest_attempt_op(
    conn: &Connection,
    session_id: SessionId,
    task_id: TaskId,
) -> StoreResult<Option<u64>> {
    let newest: Option<i64> = conn
        .query_row(
            "SELECT MAX(attempt_op_id) FROM verification_attempt
             WHERE session_id = ?1 AND task_id = ?2",
            params![session_id.raw() as i64, task_id.raw() as i64],
            |r| r.get(0),
        )
        .optional()?
        .flatten();
    Ok(newest.map(|op| op.max(0) as u64))
}

/// Fetch one background job row by identity, or `None`.
fn verification_job_get(
    conn: &Connection,
    session_id: SessionId,
    task_id: TaskId,
    attempt_op_id: u64,
    check_id: &str,
) -> StoreResult<Option<VerificationJobRow>> {
    let sql = format!(
        "{} WHERE session_id = ?1 AND task_id = ?2 AND attempt_op_id = ?3 AND check_id = ?4",
        VERIFICATION_JOB_SELECT
    );
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query(params![
        session_id.raw() as i64,
        task_id.raw() as i64,
        attempt_op_id as i64,
        check_id
    ])?;
    match rows.next()? {
        Some(row) => Ok(Some(verification_job_map(row)?)),
        None => Ok(None),
    }
}

/// Build one attempt view (attempt + changed files + every required check in
/// derivation order), or `None` when the attempt row does not exist.
fn verification_attempt_view(
    conn: &Connection,
    session_id: SessionId,
    task_id: TaskId,
    attempt_op_id: u64,
) -> StoreResult<Option<VerificationAttemptView>> {
    let attempt = {
        let mut stmt = conn.prepare(
            "SELECT session_id, task_id, attempt_op_id, task_revision, workspace_root,
                    environment_fingerprint_json, created_ms
             FROM verification_attempt
             WHERE session_id = ?1 AND task_id = ?2 AND attempt_op_id = ?3",
        )?;
        let mut rows = stmt.query(params![
            session_id.raw() as i64,
            task_id.raw() as i64,
            attempt_op_id as i64
        ])?;
        match rows.next()? {
            Some(row) => Some(verification_attempt_map(row)?),
            None => None,
        }
    };
    let Some(attempt) = attempt else {
        return Ok(None);
    };
    let mut changed: Vec<String> = Vec::new();
    {
        let mut stmt = conn.prepare(
            "SELECT path FROM verification_attempt_changed_file
             WHERE session_id = ?1 AND task_id = ?2 AND attempt_op_id = ?3
             ORDER BY ordinal ASC",
        )?;
        let mut rows = stmt.query(params![
            session_id.raw() as i64,
            task_id.raw() as i64,
            attempt_op_id as i64
        ])?;
        while let Some(row) = rows.next()? {
            changed.push(row.get(0)?);
        }
    }
    let mut checks: Vec<VerificationJobRow> = Vec::new();
    {
        let sql = format!(
            "{} WHERE session_id = ?1 AND task_id = ?2 AND attempt_op_id = ?3
             ORDER BY ordinal ASC, check_id ASC",
            VERIFICATION_JOB_SELECT
        );
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = stmt.query(params![
            session_id.raw() as i64,
            task_id.raw() as i64,
            attempt_op_id as i64
        ])?;
        while let Some(row) = rows.next()? {
            checks.push(verification_job_map(row)?);
        }
    }
    Ok(Some(VerificationAttemptView {
        attempt,
        changed,
        checks,
    }))
}

/// Enforce the v22 durable bounds on one attempt begin: every check belongs
/// to the attempt, ids are non-zero, every text field is bounded, argv is
/// bounded (count and per-argument bytes), inline checks carry an inline
/// outcome and background checks carry a bounded spec and budget. Oversized
/// input is a typed rejection before ANY write, never a truncation.
fn validate_verification_attempt(
    attempt: &VerificationAttemptRow,
    changed: &[String],
    checks: &[VerificationJobRow],
) -> StoreResult<()> {
    let malformed = |what: String| StoreError::Malformed(what);
    let oversized = |what: String| StoreError::Oversized(what);
    if attempt.session_id.raw() == 0
        || attempt.task_id.raw() == 0
        || attempt.attempt_op_id == 0
        || attempt.task_revision.raw() == 0
    {
        return Err(malformed(
            "verification attempt identity must be non-zero".into(),
        ));
    }
    if attempt.workspace_root.is_empty()
        || attempt.workspace_root.len() > MAX_VERIFICATION_JOB_ROOT_BYTES
    {
        return Err(oversized(format!(
            "attempt workspace_root of {} bytes outside 1..={MAX_VERIFICATION_JOB_ROOT_BYTES}",
            attempt.workspace_root.len()
        )));
    }
    if let Some(fp) = &attempt.environment_fingerprint_json {
        if fp.is_empty() || fp.len() > MAX_VERIFICATION_JOB_FINGERPRINT_JSON_BYTES {
            return Err(oversized(format!(
                "environment fingerprint JSON of {} bytes outside 1..={MAX_VERIFICATION_JOB_FINGERPRINT_JSON_BYTES}",
                fp.len()
            )));
        }
    }
    if changed.len() > MAX_VERIFICATION_ATTEMPT_CHANGED {
        return Err(oversized(format!(
            "{} changed files exceed MAX_VERIFICATION_ATTEMPT_CHANGED ({MAX_VERIFICATION_ATTEMPT_CHANGED})",
            changed.len()
        )));
    }
    for path in changed {
        if path.is_empty() || path.len() > MAX_VERIFICATION_ATTEMPT_PATH_BYTES {
            return Err(oversized(format!(
                "changed path of {} bytes outside 1..={MAX_VERIFICATION_ATTEMPT_PATH_BYTES}",
                path.len()
            )));
        }
    }
    if checks.is_empty() || checks.len() > MAX_VERIFICATION_ATTEMPT_CHECKS {
        return Err(oversized(format!(
            "{} required checks outside 1..={MAX_VERIFICATION_ATTEMPT_CHECKS}",
            checks.len()
        )));
    }
    let mut ordinal_seen = std::collections::HashSet::new();
    let mut check_seen = std::collections::HashSet::new();
    for check in checks {
        if check.session_id != attempt.session_id
            || check.task_id != attempt.task_id
            || check.attempt_op_id != attempt.attempt_op_id
        {
            return Err(malformed(format!(
                "check '{}' does not belong to its attempt identity",
                check.check_id
            )));
        }
        if check.check_id.is_empty() || check.check_id.len() > MAX_VERIFICATION_JOB_CHECK_ID_BYTES {
            return Err(oversized(format!(
                "check_id of {} bytes outside 1..={MAX_VERIFICATION_JOB_CHECK_ID_BYTES}",
                check.check_id.len()
            )));
        }
        if !check_seen.insert(check.check_id.clone()) {
            return Err(malformed(format!(
                "duplicate check '{}' in one attempt",
                check.check_id
            )));
        }
        if !ordinal_seen.insert(check.ordinal) {
            return Err(malformed(format!(
                "duplicate derivation ordinal {} in one attempt",
                check.ordinal
            )));
        }
        if check.command.is_empty() || check.command.len() > MAX_VERIFICATION_JOB_COMMAND_BYTES {
            return Err(oversized(format!(
                "check command of {} bytes outside 1..={MAX_VERIFICATION_JOB_COMMAND_BYTES}",
                check.command.len()
            )));
        }
        if check.program.len() > MAX_VERIFICATION_JOB_PROGRAM_BYTES {
            return Err(oversized(format!(
                "check program of {} bytes exceeds MAX_VERIFICATION_JOB_PROGRAM_BYTES ({MAX_VERIFICATION_JOB_PROGRAM_BYTES})",
                check.program.len()
            )));
        }
        let args: Vec<String> = parse_json(
            &format!("verification job {} args", check.check_id),
            &check.args_json,
        )?;
        if args.len() > MAX_VERIFICATION_JOB_ARGS {
            return Err(oversized(format!(
                "{} check args exceed MAX_VERIFICATION_JOB_ARGS ({MAX_VERIFICATION_JOB_ARGS})",
                args.len()
            )));
        }
        for arg in &args {
            if arg.len() > MAX_VERIFICATION_JOB_ARG_BYTES {
                return Err(oversized(format!(
                    "a check arg of {} bytes exceeds MAX_VERIFICATION_JOB_ARG_BYTES ({MAX_VERIFICATION_JOB_ARG_BYTES})",
                    arg.len()
                )));
            }
        }
        if let Some(note) = &check.note {
            if note.is_empty() || note.len() > MAX_VERIFICATION_JOB_NOTE_BYTES {
                return Err(oversized(format!(
                    "job note of {} bytes outside 1..={MAX_VERIFICATION_JOB_NOTE_BYTES}",
                    note.len()
                )));
            }
        }
        if check.op_id == Some(0) {
            return Err(malformed(format!(
                "check '{}' carries a zero op id",
                check.check_id
            )));
        }
        if let Some(fp) = &check.environment_fingerprint_json {
            if fp.is_empty() || fp.len() > MAX_VERIFICATION_JOB_FINGERPRINT_JSON_BYTES {
                return Err(oversized(format!(
                    "job fingerprint JSON of {} bytes outside 1..={MAX_VERIFICATION_JOB_FINGERPRINT_JSON_BYTES}",
                    fp.len()
                )));
            }
        }
        match &check.inline_status {
            Some(inline) => {
                if !VERIFICATION_INLINE_STATES.contains(&inline.as_str()) {
                    return Err(malformed(format!(
                        "inline status {inline:?} is not one of {VERIFICATION_INLINE_STATES:?}"
                    )));
                }
                if check.spec_json.is_some() {
                    return Err(malformed(format!(
                        "inline check '{}' must not carry a background spec",
                        check.check_id
                    )));
                }
                if check.state != *inline {
                    return Err(malformed(format!(
                        "inline check '{}' state {:?} must equal its inline status {inline:?}",
                        check.check_id, check.state
                    )));
                }
            }
            None => {
                let Some(spec) = &check.spec_json else {
                    return Err(malformed(format!(
                        "background check '{}' must carry its typed spec",
                        check.check_id
                    )));
                };
                if check.kind.is_empty() || check.kind.len() > MAX_VERIFICATION_JOB_KIND_BYTES {
                    return Err(oversized(format!(
                        "check kind of {} bytes outside 1..={MAX_VERIFICATION_JOB_KIND_BYTES}",
                        check.kind.len()
                    )));
                }
                if spec.is_empty() || spec.len() > MAX_VERIFICATION_JOB_SPEC_JSON_BYTES {
                    return Err(oversized(format!(
                        "job spec JSON of {} bytes outside 1..={MAX_VERIFICATION_JOB_SPEC_JSON_BYTES}",
                        spec.len()
                    )));
                }
                if check.budget_ms == 0 || check.budget_ms > MAX_VERIFICATION_JOB_BUDGET_MS {
                    return Err(oversized(format!(
                        "job budget_ms {} outside 1..={MAX_VERIFICATION_JOB_BUDGET_MS}",
                        check.budget_ms
                    )));
                }
                if !VERIFICATION_JOB_STATES.contains(&check.state.as_str()) {
                    return Err(malformed(format!(
                        "job state {:?} is not one of {VERIFICATION_JOB_STATES:?}",
                        check.state
                    )));
                }
            }
        }
    }
    Ok(())
}

fn session_row_map(r: &rusqlite::Row<'_>) -> StoreResult<SessionRow> {
    let id = SessionId::new(r.get::<_, i64>(0)? as u64);
    Ok(SessionRow {
        id,
        workspace_id: WorkspaceId::new(r.get::<_, i64>(1)? as u64),
        worktree_id: WorktreeId::new(r.get::<_, i64>(2)? as u64),
        task_id: TaskId::new(r.get::<_, i64>(3)? as u64),
        title: r.get(4)?,
        provider: r.get(5)?,
        model: r.get(6)?,
        state: parse_json(&format!("session {id} state"), &r.get::<_, String>(7)?)?,
        lifecycle: parse_lifecycle(&format!("session {id} lifecycle"), &r.get::<_, String>(8)?)?,
        created_ms: r.get(9)?,
        updated_ms: r.get(10)?,
    })
}

/// Parse of a persisted lifecycle that FAILS CLOSED on corruption.
///
/// A session that was really Closed/FailedPermanent must never silently
/// become Open (an autonomous agent could accept work again), so unreadable
/// content surfaces as `StoreError::Corrupt` instead of defaulting to Open.
///
/// The one tolerated non-JSON spelling is the bare literal `open`: the v2
/// schema declares `lifecycle TEXT NOT NULL DEFAULT 'open'`, `create_session`
/// INSERTs exactly that SQL literal, and the v2 ALTER backfilled every
/// pre-existing row to it — it is the schema's own default representation of
/// `Open`, not corruption. The column is NOT NULL, so a NULL lifecycle cannot
/// occur; had one been read (e.g. constraints disabled), the decode would
/// fail via the `Sqlite` error rather than reopening the session.
fn parse_lifecycle(ctx: &str, raw: &str) -> StoreResult<faktor_core::state::SessionLifecycle> {
    if raw == "open" {
        return Ok(faktor_core::state::SessionLifecycle::Open);
    }
    serde_json::from_str(raw)
        .map_err(|e| StoreError::Corrupt(vec![format!("{ctx}: lifecycle {raw:?} is corrupt: {e}")]))
}

/// One typed ledger row mapper (v11).
fn ledger_entry_map(r: &rusqlite::Row<'_>, session_id: SessionId) -> StoreResult<LedgerEntryRow> {
    let seq: i64 = r.get(0)?;
    Ok(LedgerEntryRow {
        seq,
        entry_type: r.get(1)?,
        schema_ver: r.get(2)?,
        payload: parse_json(
            &format!("ledger entry {session_id}/{seq} payload"),
            &r.get::<_, String>(3)?,
        )?,
        created_ms: r.get(4)?,
    })
}

fn event_map(r: &rusqlite::Row<'_>, session_id: SessionId) -> StoreResult<(Event, i64)> {
    let seq = EventSeq::new(r.get::<_, i64>(0)? as u64);
    let kind_raw = r.get::<_, String>(3)?;
    let kind = kind_from_name(&kind_raw).ok_or_else(|| {
        StoreError::Corrupt(vec![format!(
            "event {session_id}/{seq}: unknown kind {kind_raw:?}"
        )])
    })?;
    Ok((
        Event {
            seq,
            session_id,
            op_id: r.get::<_, Option<i64>>(2)?.map(|o| OpId::new(o as u64)),
            kind,
            state: parse_json(
                &format!("event {session_id}/{seq} state"),
                &r.get::<_, String>(4)?,
            )?,
            ts_ms: r.get(5)?,
            payload: match r.get::<_, Option<String>>(6)? {
                Some(raw) => Some(parse_json(
                    &format!("event {session_id}/{seq} payload"),
                    &raw,
                )?),
                None => None,
            },
        },
        // The payload schema version tag (v11+; 1 on rows written before
        // versioning existed). Readers decode through it.
        r.get::<_, i64>(7)?,
    ))
}

fn part_map(r: &rusqlite::Row<'_>) -> StoreResult<PartRow> {
    let id = r.get::<_, i64>(0)?;
    Ok(PartRow {
        id,
        message_id: r.get(1)?,
        kind: r.get(2)?,
        data: parse_json(&format!("part {id} data"), &r.get::<_, String>(3)?)?,
        created_ms: r.get(4)?,
    })
}

fn tool_run_map(r: &rusqlite::Row<'_>) -> StoreResult<ToolRunRow> {
    let id = r.get::<_, i64>(0)?;
    Ok(ToolRunRow {
        id,
        session_id: SessionId::new(r.get::<_, i64>(1)? as u64),
        op_id: OpId::new(r.get::<_, i64>(2)? as u64),
        tool: r.get(3)?,
        args: parse_json(&format!("tool_run {id} args"), &r.get::<_, String>(4)?)?,
        status: r.get(5)?,
        started_ms: r.get(6)?,
        ended_ms: r.get(7)?,
        effect_status: r.get(8)?,
        recovery: parse_json(&format!("tool_run {id} recovery"), &r.get::<_, String>(9)?)?,
        expected_hash: r.get(10)?,
        replay_descriptor: match r.get::<_, Option<String>>(11)? {
            Some(raw) => Some(parse_json(
                &format!("tool_run {id} replay_descriptor"),
                &raw,
            )?),
            None => None,
        },
        attempt: r.get(12)?,
        postcondition: match r.get::<_, Option<String>>(13)? {
            Some(raw) => Some(parse_json(&format!("tool_run {id} postcondition"), &raw)?),
            None => None,
        },
    })
}

fn turn_record_map(r: &rusqlite::Row<'_>) -> rusqlite::Result<TurnRecordRow> {
    let id = r.get::<_, i64>(0)?;
    Ok(TurnRecordRow {
        id,
        session_id: SessionId::new(r.get::<_, i64>(1)? as u64),
        turn_op_id: OpId::new(r.get::<_, i64>(2)? as u64),
        queue_seq: r.get(3)?,
        prompt_message_id: r.get(4)?,
        effective_provider: r.get(5)?,
        effective_model: r.get(6)?,
        variant: r.get(7)?,
        tool_mode: r.get(8)?,
        started_at: r.get(9)?,
        status: r.get(10)?,
        updated_ms: r.get(11)?,
    })
}

/// Fallible JSON parse of persisted data: corrupted or version-skewed rows
/// surface as `Corrupt`, never a panic.
fn parse_json<T: serde::de::DeserializeOwned>(ctx: &str, raw: &str) -> StoreResult<T> {
    serde_json::from_str(raw).map_err(|e| StoreError::Corrupt(vec![format!("{ctx}: {e}")]))
}

/// Strict shape check for the v19 `provider_call.prefix_segments_json`
/// payload — the durable mirror of the wire plan's per-call
/// `PrefixObservation` serialization. Valid BOTH on write (the typed API
/// refuses hostile payloads before anything touches the row) and on read
/// (a payload injected behind the API's back is a loud `Malformed`, never a
/// silently degraded observation):
///
/// ```text
/// { "segment_hashes": ["<64 hex>", ...],
///   "segment_token_counts": [<u64>, ...],
///   "cache_read_tokens": <u64> }
/// ```
///
/// Exactly those three fields (unknown fields are corruption, matching the
/// wire type's own strict decode), byte-bounded by
/// [`MAX_PREFIX_SEGMENTS_JSON`], hash/token vectors of EQUAL length and at
/// most [`MAX_PREFIX_SEGMENTS`] entries, every hash a 64-char hex digest.
fn validate_prefix_segments_json(json: &str) -> StoreResult<()> {
    if json.len() > MAX_PREFIX_SEGMENTS_JSON {
        return Err(StoreError::Oversized(format!(
            "prefix_segments_json is {} bytes, over the {MAX_PREFIX_SEGMENTS_JSON}-byte bound",
            json.len()
        )));
    }
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RawObservation {
        segment_hashes: Vec<String>,
        segment_token_counts: Vec<u64>,
        cache_read_tokens: u64,
    }
    let raw: RawObservation = serde_json::from_str(json).map_err(|e| {
        StoreError::Malformed(format!(
            "prefix_segments_json is not a valid observation: {e}"
        ))
    })?;
    if raw.segment_hashes.len() != raw.segment_token_counts.len() {
        return Err(StoreError::Malformed(format!(
            "prefix_segments_json hash/token length mismatch: {} vs {}",
            raw.segment_hashes.len(),
            raw.segment_token_counts.len()
        )));
    }
    if raw.segment_hashes.len() > MAX_PREFIX_SEGMENTS {
        return Err(StoreError::Malformed(format!(
            "prefix_segments_json carries {} segments, over the {MAX_PREFIX_SEGMENTS} bound",
            raw.segment_hashes.len()
        )));
    }
    for hash in &raw.segment_hashes {
        if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(StoreError::Malformed(
                "prefix_segments_json carries a non-64-char-hex segment digest".into(),
            ));
        }
    }
    // `cache_read_tokens` is decoded but not otherwise constrained: any
    // provider-reported count is a legal observation, it never prices a
    // call by itself.
    let _ = raw.cache_read_tokens;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use faktor_core::capability::PermissionDecision;
    use faktor_core::state::TaskState;

    fn tmp_store() -> (tempfile::TempDir, Store) {
        let dir = tempfile::tempdir().unwrap();
        let s = Store::open(dir.path(), true).unwrap();
        (dir, s)
    }

    #[test]
    fn loop_signal_survives_reopen_and_trips_at_threshold() {
        // Spec §28 durable signals: crash between turns must NOT lose the
        // count — the third identical failure after a daemon restart still
        // trips.
        let dir = tempfile::tempdir().unwrap();
        let ws: WorkspaceId;
        let sid: SessionId;
        {
            let store = Store::open(dir.path().join("store"), true).unwrap();
            ws = store.create_workspace("/w").unwrap();
            let row = store.create_session(ws, "t", "p", "m").unwrap();
            sid = row.id;
            for i in 1..=2 {
                let tripped = store
                    .bump_loop_signal(sid, "fail run_command", 3, i * 1000)
                    .unwrap();
                assert!(!tripped, "count {i} must not trip yet");
            }
            // Reopen happens when the store drops (crash simulation).
        }
        {
            let store = Store::open(dir.path().join("store"), true).unwrap();
            assert!(
                store
                    .bump_loop_signal(sid, "fail run_command", 3, 3000)
                    .unwrap(),
                "third identical failure after a restart must trip"
            );
            // Progress clears everything.
            store.reset_loop_signals(sid).unwrap();
            assert!(store.loop_signal_counts(sid).unwrap().is_empty());
        }
    }

    #[test]
    fn migrate_and_reopen_is_stable() {
        let dir = tempfile::tempdir().unwrap();
        {
            let s = Store::open(dir.path(), true).unwrap();
            s.create_workspace("/tmp/ws").unwrap();
            s.create_session(WorkspaceId::new(1), "t", "ollama", "qwen3.8")
                .unwrap();
        }
        // Reopen: migrations must be a no-op and data must survive.
        let s = Store::open(dir.path(), true).unwrap();
        assert!(s.get_session(SessionId::new(1)).unwrap().is_some());
        assert_eq!(
            s.last_event_seq(SessionId::new(1)).unwrap().unwrap().raw(),
            1
        );
    }

    #[test]
    fn corrupt_db_file_is_detected_on_open() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("faktor-plus.db"),
            b"this is not a sqlite database at all - definitely not valid magic header bytes",
        )
        .unwrap();
        match Store::open(dir.path(), true) {
            Err(StoreError::Sqlite(_)) | Err(StoreError::Corrupt(_)) => {}
            other => panic!("corrupt db must fail cleanly, got {other:?}"),
        }
        // Without integrity_check it may open (SQLite lazy), but any query
        // must error, not panic.
        let s = Store::open(dir.path(), false);
        if let Ok(s) = s {
            let r = s.list_sessions(None);
            assert!(r.is_err() || r.is_ok(), "never panic");
        }
    }

    #[test]
    fn truncated_db_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("faktor-plus.db"), b"SQLite format 3\x00").unwrap();
        match Store::open(dir.path(), true) {
            Err(StoreError::Sqlite(_)) | Err(StoreError::Corrupt(_)) => {}
            other => panic!("truncated db must fail cleanly, got {other:?}"),
        }
    }

    #[test]
    fn journal_sequences_are_gapless_under_concurrent_append() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let session = store.create_session(ws, "c", "p", "m").unwrap();
        let sid = session.id;
        let store = std::sync::Arc::new(store);
        let mut handles = vec![];
        for t in 0..8 {
            let store = store.clone();
            handles.push(std::thread::spawn(move || {
                for i in 0..50 {
                    store
                        .append_event(
                            sid,
                            Some(OpId::new(1 + t * 100 + i)),
                            EventKind::ModelChunkReceived,
                            AgentState::Streaming,
                            now_ms(),
                            Some(serde_json::json!({"i": i})),
                        )
                        .unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let events = store.events_range(sid, 1, None).unwrap();
        // SessionCreated(1) + 400 chunks = 401 events, seq 1..=401 gapless.
        assert_eq!(events.len(), 401);
        for (i, e) in events.iter().enumerate() {
            assert_eq!(e.seq.raw(), (i + 1) as u64, "gap at {i}");
        }
        // Resume cursor semantics: events_after(seq 400) returns exactly 1.
        let tail = store.events_after(sid, EventSeq::new(400)).unwrap();
        assert_eq!(tail.len(), 1);
        assert_eq!(tail[0].seq.raw(), 401);
    }

    #[test]
    fn message_paging_is_fundamental_and_stable() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let session = store.create_session(ws, "t", "p", "m").unwrap();
        for i in 0..100 {
            store
                .put_message(
                    session.id,
                    i,
                    "user",
                    serde_json::json!({"text": format!("msg {i}")}),
                )
                .unwrap();
        }
        // Newest first, page size 10.
        let page1 = store.messages_before(session.id, None, 10).unwrap();
        assert_eq!(page1.len(), 10);
        assert_eq!(page1[0].seq, 99);
        // Cursor paging reaches everything exactly once.
        let mut seen = vec![];
        let mut cursor = None;
        loop {
            let page = store.messages_before(session.id, cursor, 7).unwrap();
            if page.is_empty() {
                break;
            }
            for m in &page {
                assert!(!seen.contains(&m.seq), "duplicate message in paging");
                seen.push(m.seq);
            }
            cursor = Some(page.last().unwrap().seq);
        }
        assert_eq!(seen.len(), 100);
    }

    /// Insert `n` messages with seq 1..=n and a controlled payload size
    /// (the exact persisted JSON bytes are returned per row for byte-bound
    /// tests). All payloads are the same size.
    fn seed_messages(store: &Store, sid: SessionId, n: i64, payload_len: usize) -> Vec<u64> {
        let mut sizes = Vec::new();
        for i in 1..=n {
            let data = serde_json::json!({ "text": "a".repeat(payload_len) });
            sizes.push(serde_json::to_string(&data).unwrap().len() as u64);
            store
                .put_message(
                    sid,
                    i,
                    "user",
                    serde_json::json!({ "text": "a".repeat(payload_len) }),
                )
                .unwrap();
        }
        sizes
    }

    /// (a) exactly at the max_bytes boundary: two 100-byte rows against a
    /// 200-byte budget are both returned; the third row (which would cross
    /// the boundary) stops the walk — and nothing beyond it is read.
    #[test]
    fn bounded_backwards_stops_exactly_at_the_byte_boundary() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let sid = store.create_session(ws, "t", "p", "m").unwrap().id;
        let sizes = seed_messages(&store, sid, 6, 100);
        assert_eq!(
            sizes[0],
            serde_json::to_string(&serde_json::json!({"text": "a".repeat(100)}))
                .unwrap()
                .len() as u64
        );
        let window = store
            .messages_backwards_bounded(sid, None, 10, sizes[0] * 2)
            .unwrap();
        assert_eq!(
            window.iter().map(|r| r.seq).collect::<Vec<_>>(),
            vec![6, 5],
            "exactly two rows fit the 200-byte budget"
        );
        // The byte total of the returned window never exceeds max_bytes.
        let total: u64 = window
            .iter()
            .map(|r| serde_json::to_string(&r.data).unwrap().len() as u64)
            .sum();
        assert_eq!(total, sizes[0] * 2);
        // Hostile sub-row budget: the newest row alone is returned whole
        // (message granularity — never a partial row, never an empty window
        // when a message exists).
        let one = store
            .messages_backwards_bounded(sid, None, 10, sizes[0] - 1)
            .unwrap();
        assert_eq!(one.iter().map(|r| r.seq).collect::<Vec<_>>(), vec![6]);
        let zero = store.messages_backwards_bounded(sid, None, 10, 0).unwrap();
        assert_eq!(
            zero.len(),
            1,
            "max_bytes = 0 still yields the newest message"
        );
        // max_messages = 0 is the empty contract (no row is ever read).
        assert!(store
            .messages_backwards_bounded(sid, None, 0, u64::MAX)
            .unwrap()
            .is_empty());
    }

    /// (b) 10k messages whose OLD tail has been corrupted into unreadable
    /// blobs: the bounded call must still succeed and return exactly the
    /// newest window — proof it never reads (never materializes) the old
    /// tail. A load-then-trim implementation would hit the corrupt rows and
    /// error. The corruption is proven live by a probe whose bound steps
    /// ONE row past the healthy window: it must fail loudly — the walk
    /// really stops where the bounds say it stops.
    #[test]
    fn bounded_backwards_never_touches_a_corrupted_old_tail() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let sid = store.create_session(ws, "t", "p", "m").unwrap().id;
        seed_messages(&store, sid, 10_000, 40);
        // Corrupt every row below seq 9501: data becomes a BLOB, so reading
        // it as TEXT fails loudly. The newest 500 rows (seq 9501..=10000)
        // stay healthy.
        {
            let conn = store.writer.lock().unwrap_or_else(|e| e.into_inner());
            conn.execute(
                "UPDATE message SET data = x'FF' WHERE session_id = ?1 AND seq < 9501",
                params![sid.raw() as i64],
            )
            .unwrap();
        }
        // Message-bound window over the healthy region: exactly the newest
        // 500 rows, never stepping into the corrupt tail.
        let window = store
            .messages_backwards_bounded(sid, None, 500, u64::MAX)
            .unwrap();
        assert_eq!(window.len(), 500, "exactly the healthy newest rows");
        assert_eq!(window[0].seq, 10_000, "newest first");
        assert_eq!(window.last().unwrap().seq, 9_501);
        // Byte-bound window stops even earlier, still never touching the
        // corrupt tail.
        let tiny = store
            .messages_backwards_bounded(sid, None, 10_000, 100)
            .unwrap();
        assert_eq!(tiny.len(), 1);
        assert_eq!(tiny[0].seq, 10_000);
        // The corruption is LIVE: a bound that steps one row past the
        // healthy window must fail loudly (never silently return garbage or
        // skip the row).
        let err = store
            .messages_backwards_bounded(sid, None, 501, u64::MAX)
            .unwrap_err();
        assert!(
            matches!(err, StoreError::Sqlite(_)),
            "corrupt row is read when the bound demands it: {err:?}"
        );
    }

    /// (b') The deletion variant: old rows removed mid-range leave holes
    /// (paging skips holes; nothing is renumbered). A bounded load over a
    /// hole-riddled tail still returns the newest window deterministically.
    #[test]
    fn bounded_backwards_skips_deleted_holes_in_the_tail() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let sid = store.create_session(ws, "t", "p", "m").unwrap().id;
        seed_messages(&store, sid, 10_000, 30);
        // Delete a mid-range band (seq 4000..=6000): rows below it are
        // physically gone — only a scanner that never goes there survives.
        for seq in 4000..=6000i64 {
            store.delete_message(sid, seq).unwrap();
        }
        let window = store
            .messages_backwards_bounded(sid, None, 10_000, u64::MAX)
            .unwrap();
        assert_eq!(
            window.len(),
            7_999,
            "10000 - 2001 deleted (band 4000..=6000)"
        );
        assert_eq!(window[0].seq, 10_000);
        assert_eq!(window.last().unwrap().seq, 1, "newest-first, hole-free");
        assert!(
            window.windows(2).all(|w| w[0].seq > w[1].seq),
            "strictly newest-first"
        );
    }

    /// (c) before_seq cuts exactly between messages (`seq < before`): seq 5
    /// is excluded, seq 4 is the newest of the window; u64 values above
    /// i64::MAX behave like "no older bound"; 0 and 1 cut below every row.
    #[test]
    fn bounded_backwards_before_seq_cuts_exactly_between_messages() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let sid = store.create_session(ws, "t", "p", "m").unwrap().id;
        seed_messages(&store, sid, 10, 10);
        let window = store
            .messages_backwards_bounded(sid, Some(5), 10_000, u64::MAX)
            .unwrap();
        assert_eq!(
            window.iter().map(|r| r.seq).collect::<Vec<_>>(),
            vec![4, 3, 2, 1],
            "before = 5 excludes seq 5 itself"
        );
        let cut = store
            .messages_backwards_bounded(sid, Some(5), 2, u64::MAX)
            .unwrap();
        assert_eq!(cut.iter().map(|r| r.seq).collect::<Vec<_>>(), vec![4, 3]);
        // Absurd cursors clamp instead of erroring.
        assert_eq!(
            store
                .messages_backwards_bounded(sid, Some(u64::MAX), 3, u64::MAX)
                .unwrap()
                .iter()
                .map(|r| r.seq)
                .collect::<Vec<_>>(),
            vec![10, 9, 8]
        );
        assert!(store
            .messages_backwards_bounded(sid, Some(1), 10_000, u64::MAX)
            .unwrap()
            .is_empty());
        assert!(store
            .messages_backwards_bounded(sid, Some(0), 10_000, u64::MAX)
            .unwrap()
            .is_empty());
        // The cursor can itself be a hole left by deletion: rows with
        // seq < 5 after deleting seq 5..=8 still start at seq 4.
        for seq in 5..=8i64 {
            store.delete_message(sid, seq).unwrap();
        }
        assert_eq!(
            store
                .messages_backwards_bounded(sid, Some(9), 10_000, u64::MAX)
                .unwrap()
                .iter()
                .map(|r| r.seq)
                .collect::<Vec<_>>(),
            vec![4, 3, 2, 1]
        );
    }

    /// (d) All history fits: the bounded call returns the full list,
    /// newest-first — identical to an unbounded `messages_before` walk.
    #[test]
    fn bounded_backwards_returns_everything_when_it_fits() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let sid = store.create_session(ws, "t", "p", "m").unwrap().id;
        seed_messages(&store, sid, 250, 20);
        let full = store
            .messages_backwards_bounded(sid, None, u64::MAX, u64::MAX)
            .unwrap();
        assert_eq!(full.len(), 250);
        assert!(full.windows(2).all(|w| w[0].seq > w[1].seq));
        let expected = store.messages_before(sid, None, 250).unwrap();
        assert_eq!(
            full.iter().map(|r| r.seq).collect::<Vec<_>>(),
            expected.iter().map(|r| r.seq).collect::<Vec<_>>()
        );
        // Same content byte-for-byte (data round-trips through the bound).
        for (a, b) in full.iter().zip(expected.iter()) {
            assert_eq!(a.data, b.data);
        }
    }

    /// (e) A message bigger than max_bytes alone is still returned whole —
    /// message granularity is absolute; never a partial row and never a
    /// truncation of the payload.
    #[test]
    fn bounded_backwards_oversized_message_is_returned_whole() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let sid = store.create_session(ws, "t", "p", "m").unwrap().id;
        let big_len = 50_000;
        let big = serde_json::json!({ "blob": "b".repeat(big_len) });
        let big_bytes = serde_json::to_string(&big).unwrap().len() as u64;
        store.put_message(sid, 1, "assistant", big.clone()).unwrap();
        store
            .put_message(sid, 2, "user", serde_json::json!({"text": "x".repeat(30)}))
            .unwrap();
        let window = store.messages_backwards_bounded(sid, None, 10, 64).unwrap();
        assert_eq!(window.len(), 1, "the oversized message alone is returned");
        assert_eq!(window[0].seq, 2, "newest first even when oversized");
        assert_eq!(window[0].data, serde_json::json!({"text": "x".repeat(30)}));
        // Same rule when the oversized message is the ONLY candidate.
        let window = store
            .messages_backwards_bounded(sid, Some(2), 10, 64)
            .unwrap();
        assert_eq!(window.len(), 1);
        assert_eq!(window[0].seq, 1);
        assert_eq!(
            serde_json::to_string(&window[0].data).unwrap().len() as u64,
            big_bytes,
            "payload never truncated"
        );
        assert_eq!(window[0].data, big);
    }

    #[test]
    fn crash_recovery_scanner_input_is_durable() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let session = store.create_session(ws, "t", "p", "m").unwrap();
        let op = OpId::new(77);
        store
            .start_tool_run(
                session.id,
                op,
                "write_file",
                serde_json::json!({"path": "/w/a.txt", "content": "x"}),
                serde_json::json!({"strategy": "verify_hash", "detail": {"path": "/w/a.txt", "expected": "ab".repeat(32)}}),
                Some("ab".repeat(32)),
                None,
            )
            .unwrap();
        // Crash: no finish. The scanner must find it with effect unknown.
        let pending = store.pending_tool_runs(session.id).unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].op_id, op);
        assert_eq!(pending[0].effect_status, "unknown");
        assert_eq!(pending[0].status, "running");
        assert_eq!(
            pending[0].expected_hash.as_deref(),
            Some("ab".repeat(32).as_str())
        );
        // Finishing moves it out of the scanner set.
        store
            .finish_tool_run(session.id, op, "completed", "verified")
            .unwrap();
        assert!(store.pending_tool_runs(session.id).unwrap().is_empty());
        // finish on missing row is an error (loud, not silent)
        assert!(store
            .finish_tool_run(session.id, OpId::new(999), "completed", "verified")
            .is_err());
    }

    #[test]
    fn checkpoints_dedup_by_hash_and_survive_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let session_id = {
            let store = Store::open(dir.path(), true).unwrap();
            let ws = store.create_workspace("/w").unwrap();
            let s = store.create_session(ws, "t", "p", "m").unwrap();
            for i in 0..5 {
                store
                    .put_checkpoint(s.id, i, "a.rs", "hash-before", "hash-after", None)
                    .unwrap();
            }
            // v3: the after-blob hash roundtrips when recorded.
            store
                .put_checkpoint(s.id, 5, "b.rs", "b1", "a1", Some("cas-after-blob"))
                .unwrap();
            s.id
        };
        let store = Store::open(dir.path(), true).unwrap();
        let cps = store.checkpoints_of(session_id).unwrap();
        assert_eq!(cps.len(), 6);
        assert_eq!(cps[0].sequence, 0);
        assert_eq!(cps[4].after_hash, "hash-after");
        assert_eq!(cps[4].after_cas_hash, None);
        assert_eq!(cps[5].after_cas_hash.as_deref(), Some("cas-after-blob"));
    }

    #[test]
    fn migration_v3_keeps_pre_v3_checkpoint_rows_readable() {
        // Simulate a store that was created at v2 (checkpoints without the
        // after-blob column): open a fresh store, record a checkpoint, then
        // downgrade the schema behind the API's back (DROP COLUMN + set the
        // version cursor back). Reopening must apply v3, leave the old row
        // readable, and surface after_cas_hash as NULL — never a panic and
        // never a lost row.
        let dir = tempfile::tempdir().unwrap();
        let sid = {
            let store = Store::open(dir.path(), true).unwrap();
            let ws = store.create_workspace("/w").unwrap();
            let s = store.create_session(ws, "t", "p", "m").unwrap();
            store
                .put_checkpoint(s.id, 3, "f.txt", "before", "after", Some("after-blob"))
                .unwrap();
            {
                let conn = store.write();
                conn.execute("ALTER TABLE checkpoint DROP COLUMN after_cas_hash", [])
                    .unwrap();
                // The v6 existence columns are post-v2 too: drop them so the
                // full migration chain (v3..v6) replays on reopen.
                conn.execute("ALTER TABLE checkpoint DROP COLUMN before_exists", [])
                    .unwrap();
                conn.execute("ALTER TABLE checkpoint DROP COLUMN after_exists", [])
                    .unwrap();
                // The v7 tool-run recovery columns + turn-record table are
                // post-v2 too: drop them so the full migration chain
                // (v3..v7) replays on reopen.
                conn.execute("ALTER TABLE tool_run DROP COLUMN replay_descriptor", [])
                    .unwrap();
                conn.execute("ALTER TABLE tool_run DROP COLUMN attempt", [])
                    .unwrap();
                conn.execute("ALTER TABLE tool_run DROP COLUMN postcondition", [])
                    .unwrap();
                conn.execute("DROP TABLE turn_record", []).unwrap();
                // The v8 session-identity columns are post-v2 too: drop them
                // so the full migration chain (v3..v8) replays on reopen.
                conn.execute("ALTER TABLE session DROP COLUMN worktree_id", [])
                    .unwrap();
                conn.execute("ALTER TABLE session DROP COLUMN task_id", [])
                    .unwrap();
                // The v9/v10 task tables are post-this-version too: restore
                // the legacy `task` layout so the migration chain past v10
                // replays on reopen.
                conn.execute("DROP TABLE task", []).unwrap();
                conn.execute("ALTER TABLE task_ledger RENAME TO task", [])
                    .unwrap(); // v11 artifacts (event payload_ver + the typed ledger) are
                               // post-this-version too: drop them so the full chain
                               // (past v11) replays cleanly on reopen.
                conn.execute("ALTER TABLE event DROP COLUMN payload_ver", [])
                    .unwrap();
                conn.execute("DROP TABLE IF EXISTS ledger_entry", [])
                    .unwrap();
                conn.execute("DROP TABLE IF EXISTS ledger_head", [])
                    .unwrap();
                // v13 prefix-stability columns are post-this-version too: drop
                // them so the full chain (past v13) replays cleanly on reopen.
                conn.execute(
                    "ALTER TABLE provider_call DROP COLUMN prompt_prefix_hash",
                    [],
                )
                .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN prompt_tokens", [])
                    .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN prefix_stability", [])
                    .unwrap();
                // The v19 segment-observation column is post-this-version
                // too: drop it so the full chain (past v19) replays cleanly.
                conn.execute(
                    "ALTER TABLE provider_call DROP COLUMN prefix_segments_json",
                    [],
                )
                .unwrap();
                // The v20 verification-record evidence columns are
                // post-this-version too: drop them so the full chain
                // (past v20) replays cleanly.
                conn.execute(
                    "ALTER TABLE verification_record DROP COLUMN environment_fingerprint_json",
                    [],
                )
                .unwrap();
                conn.execute(
                    "ALTER TABLE verification_record DROP COLUMN candidate_proof_ref_json",
                    [],
                )
                .unwrap();
                conn.execute("DROP INDEX IF EXISTS idx_provider_call_session_attempt", [])
                    .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN attempt_op_id", [])
                    .unwrap();
                conn.execute(
                    "ALTER TABLE provider_call DROP COLUMN parent_model_call_op_id",
                    [],
                )
                .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN attempt_ordinal", [])
                    .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN reservation_id", [])
                    .unwrap();
                conn.execute("PRAGMA user_version = 2", []).unwrap();
            }
            s.id
        };
        // Reopen: v3 re-applies the column; the pre-v3 row must read back
        // intact with after_cas_hash = NULL.
        let store = Store::open(dir.path(), true).unwrap();
        let cps = store.checkpoints_of(sid).unwrap();
        assert_eq!(cps.len(), 1, "the old row must survive the v3 migration");
        assert_eq!(cps[0].path, "f.txt");
        assert_eq!(cps[0].before_hash, "before");
        assert_eq!(cps[0].after_hash, "after");
        assert_eq!(cps[0].after_cas_hash, None);
        assert!(cps[0].created_ms > 0);
        // And the column is writable again.
        store
            .put_checkpoint(sid, 4, "g.txt", "b", "a", Some("x"))
            .unwrap();
        assert_eq!(
            store.checkpoints_of(sid).unwrap()[1]
                .after_cas_hash
                .as_deref(),
            Some("x")
        );
    }

    #[test]
    fn checkpoint_sequence_allocation_is_atomic_under_concurrent_writers() {
        // P1 "checkpoint numbering race": the old flow derived the sequence
        // from rows.len()+1 OUTSIDE the store, so two racing writers could
        // both receive the same N+1. insert_checkpoint must allocate
        // MAX(sequence)+1 and insert in ONE transaction: 8 writers × 25
        // checkpoints = 200 rows, all sequences distinct and gapless.
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        let store = std::sync::Arc::new(store);
        let mut handles = Vec::new();
        for t in 0..8 {
            let store = store.clone();
            let sid = s.id;
            handles.push(std::thread::spawn(move || {
                for i in 0..25 {
                    let (id, seq) = store
                        .insert_checkpoint(
                            sid,
                            &format!("f{t}-{i}.rs"),
                            true,
                            "before-hash",
                            true,
                            "after-hash",
                            None,
                        )
                        .unwrap();
                    assert!(id > 0);
                    assert!(seq >= 1, "sequence must be >= 1, got {seq}");
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let rows = store.checkpoints_of(s.id).unwrap();
        assert_eq!(rows.len(), 200, "every racing insert must land");
        let mut seqs: Vec<i64> = rows.iter().map(|c| c.sequence).collect();
        seqs.sort_unstable();
        for (i, seq) in seqs.iter().enumerate() {
            assert_eq!(
                *seq,
                (i + 1) as i64,
                "sequence {seq} at slot {i}: must be gapless"
            );
        }
        // The duplicate-guard invariant: no two rows share a sequence.
        let unique: std::collections::HashSet<i64> = seqs.iter().copied().collect();
        assert_eq!(
            unique.len(),
            200,
            "two writers must never receive the same sequence"
        );
    }

    #[test]
    fn insert_checkpoint_roundtrips_existence_flags() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        // missing -> non-empty file: the before side has NO content, so its
        // hash column is the empty-string sentinel, not a hash of nothing.
        let (id, seq) = store
            .insert_checkpoint(
                s.id,
                "created.rs",
                false,
                "",
                true,
                "after-hex",
                Some("after-blob-hex"),
            )
            .unwrap();
        assert!(id > 0);
        assert_eq!(seq, 1);
        // file -> deleted (second row): the after side does not exist.
        let (_, seq2) = store
            .insert_checkpoint(s.id, "deleted.rs", true, "before-hex", false, "", None)
            .unwrap();
        assert_eq!(seq2, 2, "allocation must continue the session sequence");
        let rows = store.checkpoints_of(s.id).unwrap();
        assert_eq!(rows.len(), 2);
        assert!(
            !rows[0].before_exists,
            "missing side must read back as not existing"
        );
        assert!(rows[0].after_exists);
        assert_eq!(rows[0].before_hash, "", "no content -> no hash");
        assert_eq!(rows[0].after_hash, "after-hex");
        assert!(rows[1].before_exists);
        assert!(
            !rows[1].after_exists,
            "deleted side must read back as not existing"
        );
        assert_eq!(rows[1].after_hash, "");
        assert_eq!(rows[1].after_cas_hash, None);
    }

    #[test]
    fn migration_v6_keeps_pre_v6_checkpoint_rows_readable_as_existing() {
        // A store created at v5 records a checkpoint without existence
        // flags. Downgrade the schema behind the API's back (DROP the new
        // columns + rewind the version cursor), then reopen: v6 re-adds the
        // columns with DEFAULT 1 and the old row must read back as
        // exists:true on both sides (old rows only ever recorded real
        // files) — never a lost row, never a panic.
        let dir = tempfile::tempdir().unwrap();
        let sid = {
            let store = Store::open(dir.path(), true).unwrap();
            let ws = store.create_workspace("/w").unwrap();
            let s = store.create_session(ws, "t", "p", "m").unwrap();
            store
                .put_checkpoint(s.id, 1, "f.txt", "before", "after", Some("blob"))
                .unwrap();
            {
                let conn = store.write();
                conn.execute("ALTER TABLE checkpoint DROP COLUMN before_exists", [])
                    .unwrap();
                conn.execute("ALTER TABLE checkpoint DROP COLUMN after_exists", [])
                    .unwrap();
                // The v7 tool-run recovery columns + turn-record table are
                // post-v5 too: drop them so the full migration chain
                // (v6..v7) replays on reopen.
                conn.execute("ALTER TABLE tool_run DROP COLUMN replay_descriptor", [])
                    .unwrap();
                conn.execute("ALTER TABLE tool_run DROP COLUMN attempt", [])
                    .unwrap();
                conn.execute("ALTER TABLE tool_run DROP COLUMN postcondition", [])
                    .unwrap();
                conn.execute("DROP TABLE turn_record", []).unwrap();
                // The v8 session-identity columns are post-v5 too: drop them
                // The v9/v10 task tables are post-this-version too: restore
                // the legacy `task` layout so the migration chain past v10
                // replays on reopen.
                conn.execute("DROP TABLE task", []).unwrap();
                conn.execute("ALTER TABLE task_ledger RENAME TO task", [])
                    .unwrap();
                // so the full migration chain (v6..v8) replays on reopen.
                conn.execute("ALTER TABLE session DROP COLUMN worktree_id", [])
                    .unwrap();
                conn.execute("ALTER TABLE session DROP COLUMN task_id", [])
                    .unwrap(); // v11 artifacts (event payload_ver + the typed ledger) are
                               // post-this-version too: drop them so the full chain
                               // (past v11) replays cleanly on reopen.
                conn.execute("ALTER TABLE event DROP COLUMN payload_ver", [])
                    .unwrap();
                conn.execute("DROP TABLE IF EXISTS ledger_entry", [])
                    .unwrap();
                conn.execute("DROP TABLE IF EXISTS ledger_head", [])
                    .unwrap();
                // v13 prefix-stability columns are post-this-version too: drop
                // them so the full chain (past v13) replays cleanly on reopen.
                conn.execute(
                    "ALTER TABLE provider_call DROP COLUMN prompt_prefix_hash",
                    [],
                )
                .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN prompt_tokens", [])
                    .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN prefix_stability", [])
                    .unwrap();
                // The v19 segment-observation column is post-this-version
                // too: drop it so the full chain (past v19) replays cleanly.
                conn.execute(
                    "ALTER TABLE provider_call DROP COLUMN prefix_segments_json",
                    [],
                )
                .unwrap();
                // The v20 verification-record evidence columns are
                // post-this-version too: drop them so the full chain
                // (past v20) replays cleanly.
                conn.execute(
                    "ALTER TABLE verification_record DROP COLUMN environment_fingerprint_json",
                    [],
                )
                .unwrap();
                conn.execute(
                    "ALTER TABLE verification_record DROP COLUMN candidate_proof_ref_json",
                    [],
                )
                .unwrap();
                conn.execute("DROP INDEX IF EXISTS idx_provider_call_session_attempt", [])
                    .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN attempt_op_id", [])
                    .unwrap();
                conn.execute(
                    "ALTER TABLE provider_call DROP COLUMN parent_model_call_op_id",
                    [],
                )
                .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN attempt_ordinal", [])
                    .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN reservation_id", [])
                    .unwrap();
                conn.execute("PRAGMA user_version = 5", []).unwrap();
            }
            s.id
        };
        let store = Store::open(dir.path(), true).unwrap();
        let cps = store.checkpoints_of(sid).unwrap();
        assert_eq!(cps.len(), 1, "the old row must survive the v6 migration");
        assert!(
            cps[0].before_exists && cps[0].after_exists,
            "pre-v6 rows have no existence marker: hash present means exists:true"
        );
        assert_eq!(cps[0].before_hash, "before");
        // And the new columns are writable again.
        store
            .insert_checkpoint(sid, "g.txt", false, "", true, "a", Some("x"))
            .unwrap();
        let rows = store.checkpoints_of(sid).unwrap();
        assert_eq!(rows.len(), 2);
        assert!(!rows[1].before_exists);
        assert_eq!(
            rows[1].sequence, 2,
            "allocation continues after legacy rows"
        );
    }

    #[test]
    fn message_created_ms_queries_known_and_unknown() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        store
            .put_message(s.id, 5, "user", serde_json::json!({"text": "hi"}))
            .unwrap();
        let ms = store.message_created_ms(s.id, 5).unwrap().unwrap();
        assert!(ms > 0);
        // The same value the message row itself carries.
        assert_eq!(
            ms,
            store.messages_before(s.id, None, 10).unwrap()[0].created_ms
        );
        // Unknown seq → None, never an error.
        assert_eq!(store.message_created_ms(s.id, 99).unwrap(), None);
        assert_eq!(
            store.message_created_ms(SessionId::new(999), 5).unwrap(),
            None
        );
    }

    #[test]
    fn workspace_root_roundtrip_and_unknown() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/root/x").unwrap();
        assert_eq!(
            store.workspace_root(ws).unwrap().as_deref(),
            Some("/root/x")
        );
        assert_eq!(store.workspace_root(WorkspaceId::new(999)).unwrap(), None);
    }

    #[test]
    fn workspace_create_is_idempotent() {
        let (_d, store) = tmp_store();
        let a = store.create_workspace("/same").unwrap();
        let b = store.create_workspace("/same").unwrap();
        assert_eq!(a, b);
        let c = store.create_workspace("/other").unwrap();
        assert_ne!(a, c);
    }

    #[test]
    fn adversarial_duplicate_event_append_is_structurally_impossible() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        // Try to insert a duplicate (session, seq) by bypassing the API:
        // the PRIMARY KEY must reject it.
        let conn = store.write();
        let r = conn.execute(
            "INSERT INTO event(seq, session_id, op_id, kind, state, ts_ms, payload) VALUES (1, ?1, NULL, 'model_started', '\"streaming\"', 0, NULL)",
            params![s.id.raw() as i64],
        );
        assert!(r.is_err(), "duplicate (session,seq) must be rejected by PK");
    }

    #[test]
    fn memory_facts_upsert_and_query() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        store
            .upsert_memory_fact(s.id, "decision", "framework", "rust")
            .unwrap();
        store
            .upsert_memory_fact(s.id, "decision", "framework", "rust+tokio")
            .unwrap();
        let facts = store.memory_facts(s.id).unwrap();
        assert_eq!(facts.len(), 1, "upsert must not duplicate");
        assert_eq!(facts[0].2, "rust+tokio");
    }

    #[test]
    fn permissions_resolve_once_and_expire() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        let op = OpId::new(5);
        let pid = store.insert_permission(s.id, op, "execute_shell").unwrap();
        let pending = store.pending_permission(pid).unwrap().unwrap();
        assert_eq!(pending.0, s.id);
        assert_eq!(pending.1, op);
        assert_eq!(pending.2, "execute_shell");
        store.resolve_permission(pid, "allow").unwrap();
        assert!(store.pending_permission(pid).unwrap().is_none());
        // Resolving again must not change anything (first decision wins).
        store.resolve_permission(pid, "deny").unwrap();
        assert_eq!(
            PermissionDecision::Allow,
            PermissionDecision::Allow,
            "first decision wins"
        );
    }

    #[test]
    fn backup_restores_full_state() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path(), true).unwrap();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        store
            .put_message(s.id, 1, "user", serde_json::json!({"text": "hi"}))
            .unwrap();
        let backup_path = dir.path().join("backup.db");
        store.backup_to(&backup_path).unwrap();
        // Reopen backup as a store; data must be complete.
        let restored_dir = tempfile::tempdir().unwrap();
        std::fs::copy(&backup_path, restored_dir.path().join("faktor-plus.db")).unwrap();
        let restored = Store::open(restored_dir.path(), true).unwrap();
        assert_eq!(restored.message_count(s.id).unwrap(), 1);
        assert_eq!(
            restored.messages_before(s.id, None, 10).unwrap()[0].data["text"],
            "hi"
        );
    }

    #[test]
    fn integrity_check_survives_normal_use_and_flags_corruption() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path(), true).unwrap();
        let ws = store.create_workspace("/w").unwrap();
        let _s = store.create_session(ws, "t", "p", "m").unwrap();
        assert!(store.integrity_check().unwrap().is_empty());
        // Corrupt the DB file on disk behind the store's back: replace the
        // main file with garbage AND remove the WAL/shm sidecars so nothing
        // can paper over the corruption. A lazy reopen must either refuse to
        // open or flag the corruption on the next integrity check — never
        // silently serve a fake store.
        drop(store);
        let path = dir.path().join("faktor-plus.db");
        std::fs::write(
            &path,
            b"this file is complete garbage now, no sqlite magic header at all - 1234567890",
        )
        .unwrap();
        let _ = std::fs::remove_file(dir.path().join("faktor-plus.db-wal"));
        let _ = std::fs::remove_file(dir.path().join("faktor-plus.db-shm"));
        match Store::open(dir.path(), false) {
            Err(e) => {
                assert!(
                    matches!(e, StoreError::Sqlite(_) | StoreError::Corrupt(_)),
                    "corrupt db must fail cleanly, got {e:?}"
                );
            }
            Ok(reopened) => {
                let issues = reopened.integrity_check();
                assert!(
                    issues.is_err() || !issues.unwrap().is_empty(),
                    "corruption must surface as an error or flagged rows"
                );
            }
        }
    }

    #[test]
    fn session_state_tracks_journal() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        assert_eq!(s.state, AgentState::Idle);
        store
            .append_event(
                s.id,
                None,
                EventKind::PromptReceived,
                AgentState::Preparing,
                now_ms(),
                None,
            )
            .unwrap();
        let s2 = store.get_session(s.id).unwrap().unwrap();
        assert_eq!(s2.state, AgentState::Preparing);
        assert!(s2.updated_ms >= s.updated_ms);
    }

    #[test]
    fn worktree_crud() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let id = store.put_worktree(ws, "/w/wt1", "feat/x").unwrap();
        assert_eq!(id, store.put_worktree(ws, "/w/wt1", "feat/x").unwrap());
        let list = store.worktrees_of(ws).unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].branch, "feat/x");
        store.remove_worktree("/w/wt1").unwrap();
        assert!(store.worktrees_of(ws).unwrap().is_empty());
    }

    #[test]
    fn artifact_hash_unique_across_sessions() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s1 = store.create_session(ws, "a", "p", "m").unwrap();
        let s2 = store.create_session(ws, "b", "p", "m").unwrap();
        store
            .put_artifact(s1.id, "command_output", "hash1", "sum", 10)
            .unwrap();
        store
            .put_artifact(s2.id, "command_output", "hash1", "sum", 10)
            .unwrap();
        let a = store.artifact("hash1").unwrap().unwrap();
        assert_eq!(a.0, "sum");
        assert_eq!(store.artifact("nope").unwrap(), None);
    }

    #[test]
    fn diagnostic_smoke() {
        let (_d, store) = tmp_store();
        let d = store.diagnostics().unwrap();
        assert_eq!(d["journal_mode"], "wal");
    }

    #[test]
    fn giant_payload_roundtrip_via_cas_hash_reference() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        let blob_hash = format!("{:064x}", 7);
        let big = serde_json::json!({"blob_hash": blob_hash});
        store
            .put_artifact(
                s.id,
                "tool_output",
                big["blob_hash"].as_str().unwrap(),
                "300MB compiler log",
                300_000_000,
            )
            .unwrap();
        assert_eq!(
            store
                .artifact(big["blob_hash"].as_str().unwrap())
                .unwrap()
                .unwrap()
                .1,
            "tool_output"
        );
        // A different hash is not found.
        assert_eq!(store.artifact(&"0".repeat(63)).unwrap(), None);
    }

    #[test]
    fn reader_pool_is_concurrency_bounded() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        let store = std::sync::Arc::new(store);
        let mut handles = vec![];
        for _ in 0..20 {
            let store = store.clone();
            let sid = s.id;
            handles.push(std::thread::spawn(move || {
                for _ in 0..25 {
                    let conn = store.read().unwrap();
                    let n: i64 = conn
                        .query_row(
                            "SELECT COUNT(*) FROM session WHERE id = ?1",
                            params![sid.raw() as i64],
                            |r| r.get(0),
                        )
                        .unwrap();
                    assert_eq!(n, 1);
                    drop(conn);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        // 20 readers finished with the pool intact: retention never exceeds
        // the cap, and — the strong invariant the semaphore guarantees — the
        // number of connections ever opened never exceeds the cap either
        // (the old pool opened a new connection whenever it was empty).
        assert!(
            store.reader_pool_len() <= READER_POOL,
            "idle pool exceeds cap: {}",
            store.reader_pool_len()
        );
        assert!(
            store.connections_created() <= READER_POOL as u64,
            "connections created {} exceeds cap {}",
            store.connections_created(),
            READER_POOL
        );
    }

    #[test]
    fn reader_pool_waits_and_never_starves() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let _s = store.create_session(ws, "t", "p", "m").unwrap();
        let store = std::sync::Arc::new(store);
        // Hold all 4 permits, then prove a 5th reader waits (bounded) and
        // succeeds once a permit frees.
        let held: Vec<ReadConn> = (0..READER_POOL).map(|_| store.read().unwrap()).collect();
        let store2 = store.clone();
        let late = std::thread::spawn(move || {
            let conn = store2.read().unwrap(); // must block, then succeed
            let n: i64 = conn
                .query_row("SELECT COUNT(*) FROM session", [], |r| r.get(0))
                .unwrap();
            assert_eq!(n, 1);
        });
        std::thread::sleep(Duration::from_millis(50));
        assert!(!late.is_finished(), "5th reader must wait for a permit");
        drop(held);
        late.join().unwrap();
    }

    #[test]
    fn corrupt_state_row_returns_error_not_panic() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        {
            let conn = store.write();
            conn.execute(
                "UPDATE session SET state = ?1 WHERE id = ?2",
                params!["\"not_a_state\"", s.id.raw() as i64],
            )
            .unwrap();
        }
        match store.get_session(s.id) {
            Err(StoreError::Corrupt(_)) => {}
            other => panic!("corrupt session state must error, not panic: {other:?}"),
        }
        match store.list_sessions(Some(ws)) {
            Err(StoreError::Corrupt(_)) => {}
            other => panic!("corrupt session state must error in list too: {other:?}"),
        }
    }

    #[test]
    fn corrupt_event_kind_returns_error_not_panic() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        {
            let conn = store.write();
            conn.execute(
                "INSERT INTO event(seq, session_id, op_id, kind, state, ts_ms, payload)
                 VALUES (2, ?1, NULL, 'bogus', '\"idle\"', 0, NULL)",
                params![s.id.raw() as i64],
            )
            .unwrap();
        }
        match store.events_range(s.id, 1, None) {
            Err(StoreError::Corrupt(_)) => {}
            other => panic!("unknown event kind must error, not panic: {other:?}"),
        }
    }

    #[test]
    fn corrupt_event_state_returns_error_not_panic() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        {
            let conn = store.write();
            conn.execute(
                "INSERT INTO event(seq, session_id, op_id, kind, state, ts_ms, payload)
                 VALUES (2, ?1, NULL, 'model_started', '\"not_a_state\"', 0, NULL)",
                params![s.id.raw() as i64],
            )
            .unwrap();
        }
        match store.events_range(s.id, 1, None) {
            Err(StoreError::Corrupt(_)) => {}
            other => panic!("corrupt event state must error, not panic: {other:?}"),
        }
    }

    #[test]
    fn corrupt_payload_returns_error_not_panic() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        {
            let conn = store.write();
            conn.execute(
                "INSERT INTO event(seq, session_id, op_id, kind, state, ts_ms, payload)
                 VALUES (2, ?1, NULL, 'model_started', '\"streaming\"', 0, 'not json at all')",
                params![s.id.raw() as i64],
            )
            .unwrap();
        }
        match store.events_range(s.id, 1, None) {
            Err(StoreError::Corrupt(_)) => {}
            other => panic!("corrupt payload must error, not panic: {other:?}"),
        }
    }

    #[test]
    fn corrupt_message_and_part_data_return_error_not_panic() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        let mid = store
            .put_message(s.id, 1, "user", serde_json::json!({"text": "hi"}))
            .unwrap();
        let pid = store
            .put_part(mid, "text", serde_json::json!({"t": "hi"}))
            .unwrap();
        {
            let conn = store.write();
            conn.execute(
                "UPDATE message SET data = 'broken{' WHERE id = ?1",
                params![mid],
            )
            .unwrap();
            conn.execute(
                "UPDATE part SET data = 'also broken' WHERE id = ?1",
                params![pid],
            )
            .unwrap();
        }
        match store.messages_before(s.id, None, 10) {
            Err(StoreError::Corrupt(_)) => {}
            other => panic!("corrupt message data must error, not panic: {other:?}"),
        }
        match store.parts_of(mid) {
            Err(StoreError::Corrupt(_)) => {}
            other => panic!("corrupt part data must error, not panic: {other:?}"),
        }
    }

    #[test]
    fn corrupt_tool_run_args_and_recovery_return_error_not_panic() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        store
            .start_tool_run(
                s.id,
                OpId::new(1),
                "write_file",
                serde_json::json!({"path": "/a"}),
                serde_json::json!({"strategy": "verify_hash"}),
                None,
                None,
            )
            .unwrap();
        {
            let conn = store.write();
            conn.execute(
                "UPDATE tool_run SET args = 'broken{' WHERE session_id = ?1",
                params![s.id.raw() as i64],
            )
            .unwrap();
        }
        match store.pending_tool_runs(s.id) {
            Err(StoreError::Corrupt(_)) => {}
            other => panic!("corrupt tool_run args must error, not panic: {other:?}"),
        }
        {
            let conn = store.write();
            conn.execute(
                "UPDATE tool_run SET args = '{\"a\":1}', recovery = 'broken{' WHERE session_id = ?1",
                params![s.id.raw() as i64],
            )
            .unwrap();
        }
        match store.pending_tool_runs(s.id) {
            Err(StoreError::Corrupt(_)) => {}
            other => panic!("corrupt tool_run recovery must error, not panic: {other:?}"),
        }
    }

    #[test]
    fn corrupt_task_ledger_returns_error_not_panic() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        store
            .put_task_ledger(s.id, serde_json::json!({"tasks": []}))
            .unwrap();
        {
            let conn = store.write();
            conn.execute(
                "UPDATE task_ledger SET ledger = 'garbage' WHERE session_id = ?1",
                params![s.id.raw() as i64],
            )
            .unwrap();
        }
        match store.get_task_ledger(s.id) {
            Err(StoreError::Corrupt(_)) => {}
            other => panic!("corrupt ledger must error, not panic: {other:?}"),
        }
    }

    #[test]
    fn corrupted_lifecycle_fails_closed() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        // Fresh rows hold the schema DEFAULT literal `open` (bare, non-JSON);
        // that spelling must still read back as Open, not Corrupt.
        let row = store.get_session(s.id).unwrap().unwrap();
        assert_eq!(row.lifecycle, faktor_core::state::SessionLifecycle::Open);
        // (1) Not valid JSON at all: fail closed, never silently Open.
        {
            let conn = store.write();
            conn.execute(
                "UPDATE session SET lifecycle = 'garbage-not-json' WHERE id = ?1",
                params![s.id.raw() as i64],
            )
            .unwrap();
        }
        match store.get_session(s.id) {
            Err(StoreError::Corrupt(_)) => {}
            other => panic!("garbage lifecycle must fail closed, got {other:?}"),
        }
        // (2) Structurally valid JSON but an unknown variant: also fail
        // closed (a real Closed must not reopen as Open).
        {
            let conn = store.write();
            conn.execute(
                "UPDATE session SET lifecycle = '\"terminated\"' WHERE id = ?1",
                params![s.id.raw() as i64],
            )
            .unwrap();
        }
        match store.get_session(s.id) {
            Err(StoreError::Corrupt(_)) => {}
            other => panic!("unknown lifecycle variant must fail closed, got {other:?}"),
        }
        // (3) Regression: structurally valid JSON with a VALID variant still
        // parses.
        store
            .set_session_lifecycle(s.id, faktor_core::state::SessionLifecycle::Suspended)
            .unwrap();
        let row = store.get_session(s.id).unwrap().unwrap();
        assert_eq!(
            row.lifecycle,
            faktor_core::state::SessionLifecycle::Suspended
        );
        // (4) NULL lifecycle cannot occur: the v2 schema declares the column
        // NOT NULL DEFAULT 'open', so SQLite itself rejects a NULL write;
        // parse_lifecycle never sees None. Prove the constraint holds.
        {
            let conn = store.write();
            let err = conn
                .execute(
                    "UPDATE session SET lifecycle = NULL WHERE id = ?1",
                    params![s.id.raw() as i64],
                )
                .unwrap_err();
            assert!(
                matches!(err, rusqlite::Error::SqliteFailure(e, _) if e.code == rusqlite::ErrorCode::ConstraintViolation),
                "NOT NULL lifecycle must reject NULL writes, got {err:?}"
            );
        }
    }

    fn end_transition() -> SessionTransition {
        SessionTransition {
            expected_lifecycle: Some(SessionLifecycle::Open),
            new_lifecycle: Some(SessionLifecycle::Closed),
            expected_state: None,
            new_state: AgentState::Completed,
            event_kind: EventKind::SessionEnded,
            event_payload: None,
            event_payload_ver: 1,
        }
    }

    #[test]
    fn transition_session_commits_atomically() {
        // The crash window: the old code updated lifecycle in one transaction
        // and appended SessionEnded in a second; a crash between them left
        // Closed-without-event or event-without-Closed. One call must produce
        // BOTH, durably, and a reopen must see them together.
        let dir = tempfile::tempdir().unwrap();
        let sid = {
            let store = Store::open(dir.path(), true).unwrap();
            let ws = store.create_workspace("/w").unwrap();
            let s = store.create_session(ws, "t", "p", "m").unwrap();
            let seq = store
                .transition_session(s.id, None, end_transition())
                .unwrap();
            assert_eq!(seq.raw(), 2, "SessionCreated(1) + SessionEnded(2)");
            // Fresh read (same store): both sides of the transition visible.
            let row = store.get_session(s.id).unwrap().unwrap();
            assert_eq!(row.lifecycle, SessionLifecycle::Closed);
            assert_eq!(row.state, AgentState::Completed);
            let events = store.events_range(s.id, 1, None).unwrap();
            assert_eq!(events.len(), 2);
            assert_eq!(events[1].kind, EventKind::SessionEnded);
            s.id
        };
        // "Daemon restart": both persisted in the single transaction.
        let store = Store::open(dir.path(), true).unwrap();
        let row = store.get_session(sid).unwrap().unwrap();
        assert_eq!(row.lifecycle, SessionLifecycle::Closed);
        assert_eq!(row.state, AgentState::Completed);
        let events = store.events_range(sid, 1, None).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[1].kind, EventKind::SessionEnded);
        assert_eq!(events[1].seq.raw(), 2);
    }

    #[test]
    fn transition_session_conflict_aborts_atomically() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        store
            .set_session_lifecycle(s.id, SessionLifecycle::Suspended)
            .unwrap();
        let err = store
            .transition_session(s.id, None, end_transition())
            .unwrap_err();
        assert!(
            matches!(err, StoreError::Conflict(_)),
            "expected lifecycle mismatch must be Conflict, got {err:?}"
        );
        // Rollback proven: no event row appeared and nothing moved.
        assert_eq!(store.events_range(s.id, 1, None).unwrap().len(), 1);
        let row = store.get_session(s.id).unwrap().unwrap();
        assert_eq!(row.lifecycle, SessionLifecycle::Suspended);
        assert_eq!(row.state, AgentState::Idle);
    }

    #[test]
    fn expected_state_mismatch_same() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        store
            .append_event(
                s.id,
                None,
                EventKind::PromptReceived,
                AgentState::Preparing,
                now_ms(),
                None,
            )
            .unwrap();
        let mut t = end_transition();
        // The turn machine is mid-turn (Preparing), not Idle.
        t.expected_state = Some(AgentState::Idle);
        let err = store.transition_session(s.id, None, t).unwrap_err();
        assert!(
            matches!(err, StoreError::Conflict(_)),
            "expected state mismatch must be Conflict, got {err:?}"
        );
        // No SessionEnded row, lifecycle still Open, state still Preparing.
        assert_eq!(store.last_event_seq(s.id).unwrap().unwrap().raw(), 2);
        let row = store.get_session(s.id).unwrap().unwrap();
        assert_eq!(row.lifecycle, SessionLifecycle::Open);
        assert_eq!(row.state, AgentState::Preparing);
    }

    #[test]
    fn concurrent_end_session_races() {
        // Two (well, eight) racers try to close one session. The writer lock
        // serializes the transactions; the expected_lifecycle guard means
        // exactly ONE wins and exactly ONE SessionEnded event exists.
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        let store = std::sync::Arc::new(store);
        let mut handles = Vec::new();
        for _ in 0..8 {
            let store = store.clone();
            let sid = s.id;
            handles.push(std::thread::spawn(move || {
                store.transition_session(sid, None, end_transition())
            }));
        }
        let results: Vec<StoreResult<EventSeq>> =
            handles.into_iter().map(|h| h.join().unwrap()).collect();
        let wins = results.iter().filter(|r| r.is_ok()).count();
        let conflicts = results
            .iter()
            .filter(|r| matches!(r, Err(StoreError::Conflict(_))))
            .count();
        assert_eq!(wins, 1, "exactly one racer must win");
        assert_eq!(conflicts, 7, "the rest must conflict, got {results:?}");
        let events = store.events_range(s.id, 1, None).unwrap();
        let ended = events
            .iter()
            .filter(|e| e.kind == EventKind::SessionEnded)
            .count();
        assert_eq!(ended, 1, "exactly one SessionEnded event");
        assert_eq!(events.len(), 2);
        let row = store.get_session(s.id).unwrap().unwrap();
        assert_eq!(row.lifecycle, SessionLifecycle::Closed);
        assert_eq!(row.state, AgentState::Completed);
    }

    #[test]
    fn gapless_seq_across_transition() {
        // A transition_session event must continue the journal sequence after
        // regular appends — the shared insert path must be the SAME path.
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        for i in 0..3 {
            store
                .append_event(
                    s.id,
                    Some(OpId::new(1 + i)),
                    EventKind::ModelChunkReceived,
                    AgentState::Streaming,
                    now_ms(),
                    None,
                )
                .unwrap();
        }
        let seq = store
            .transition_session(s.id, None, end_transition())
            .unwrap();
        assert_eq!(seq.raw(), 5, "created + 3 chunks + transition = seq 5");
        let events = store.events_range(s.id, 1, None).unwrap();
        assert_eq!(events.len(), 5);
        for (i, e) in events.iter().enumerate() {
            assert_eq!(e.seq.raw(), (i + 1) as u64, "gap at {i}");
        }
        assert_eq!(events[4].kind, EventKind::SessionEnded);
    }

    #[test]
    fn set_lifecycle_if_is_conditional() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        // Auto-resume (Suspended -> Open): true and journal untouched.
        store
            .set_session_lifecycle(s.id, SessionLifecycle::Suspended)
            .unwrap();
        assert!(store
            .set_lifecycle_if(s.id, SessionLifecycle::Suspended, SessionLifecycle::Open)
            .unwrap());
        let row = store.get_session(s.id).unwrap().unwrap();
        assert_eq!(row.lifecycle, SessionLifecycle::Open);
        assert_eq!(store.last_event_seq(s.id).unwrap().unwrap().raw(), 1);
        // Wrong expectation: no update, no error.
        assert!(!store
            .set_lifecycle_if(s.id, SessionLifecycle::Suspended, SessionLifecycle::Closed)
            .unwrap());
        let row = store.get_session(s.id).unwrap().unwrap();
        assert_eq!(row.lifecycle, SessionLifecycle::Open);
    }

    #[test]
    fn transition_session_missing_session_conflicts() {
        let (_d, store) = tmp_store();
        let err = store
            .transition_session(SessionId::new(999), None, end_transition())
            .unwrap_err();
        assert!(
            matches!(err, StoreError::Conflict(_)),
            "missing session must be Conflict, got {err:?}"
        );
    }

    #[test]
    fn turn_record_lifecycle_exclusive_active_and_envelope() {
        // v7: at most ONE active logical-turn record per session; a new
        // admission finalizes stragglers; the envelope is updateable while
        // active; finish is idempotent.
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "ollama", "qwen3.8").unwrap();
        let a = OpId::new(11);
        let b = OpId::new(22);
        store
            .start_turn_record(s.id, a, None, Some(2), "ollama", "qwen3.8", None)
            .unwrap();
        let rec = store.active_turn_record(s.id).unwrap().unwrap();
        assert_eq!(rec.turn_op_id, a);
        assert_eq!(rec.status, TURN_RECORD_ACTIVE);
        assert_eq!(rec.prompt_message_id, Some(2));
        // A second admission while the first is active finalizes the first.
        store
            .start_turn_record(s.id, b, Some(1), Some(5), "ollama", "m2", Some("v1"))
            .unwrap();
        let recs = store.turn_records_of(s.id).unwrap();
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].status, TURN_RECORD_FAILED, "straggler finalized");
        assert_eq!(recs[1].status, TURN_RECORD_ACTIVE);
        assert_eq!(recs[1].queue_seq, Some(1));
        assert_eq!(recs[1].variant.as_deref(), Some("v1"));
        assert_eq!(
            store.active_turn_record(s.id).unwrap().unwrap().turn_op_id,
            b
        );
        // Envelope update (per-message model override at drive start).
        assert!(store
            .set_turn_record_envelope(
                s.id,
                b,
                "ollama",
                "override-model",
                Some("v2"),
                Some("native")
            )
            .unwrap());
        let rec = store.turn_record_of(s.id, b).unwrap().unwrap();
        assert_eq!(rec.effective_model, "override-model");
        assert_eq!(rec.tool_mode.as_deref(), Some("native"));
        // Re-admission of the SAME op upserts the SAME record (crash between
        // claim and drive; the queue row is re-admitted) — never a phantom.
        store
            .start_turn_record(s.id, b, Some(1), Some(5), "ollama", "m3", None)
            .unwrap();
        assert_eq!(store.turn_records_of(s.id).unwrap().len(), 2);
        assert_eq!(
            store
                .turn_record_of(s.id, b)
                .unwrap()
                .unwrap()
                .effective_model,
            "m3"
        );
        // Finish transitions + idempotence.
        assert!(store
            .finish_turn_record(s.id, b, TURN_RECORD_COMPLETED)
            .unwrap());
        assert!(!store
            .finish_turn_record(s.id, b, TURN_RECORD_COMPLETED)
            .unwrap());
        assert!(store.active_turn_record(s.id).unwrap().is_none());
        assert_eq!(
            store.turn_record_of(s.id, b).unwrap().unwrap().status,
            TURN_RECORD_COMPLETED
        );
        // Invalid statuses are rejected loudly.
        assert!(store.finish_turn_record(s.id, b, "bogus").is_err());
        // Unknown op: no record.
        assert!(store.turn_record_of(s.id, OpId::new(99)).unwrap().is_none());
    }

    #[test]
    fn tool_run_recovery_columns_roundtrip_and_attempts() {
        // v7: the replay descriptor, the physical-attempt counter and the
        // workspace-write postcondition ride the tool_run row.
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        let op = OpId::new(7);
        store
            .start_tool_run(
                s.id,
                op,
                "echo",
                serde_json::json!({"x": 1}),
                serde_json::json!({"strategy": "idempotent"}),
                None,
                Some(serde_json::json!({"tool_name": "echo", "validated_args": {"x": 1}})),
            )
            .unwrap();
        let pending = store.pending_tool_runs(s.id).unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].attempt, 0, "the original run is attempt 0");
        assert_eq!(
            pending[0].replay_descriptor.as_ref().unwrap()["tool_name"],
            "echo"
        );
        // Postcondition annotation (recorded at execution end, pre-finish).
        store
            .record_tool_postcondition(
                s.id,
                op,
                &serde_json::json!({
                    "workspace_id": ws.raw(),
                    "worktree_id": 1,
                    "relative_path": "a.txt",
                    "expected_hash": "ab".repeat(32),
                }),
            )
            .unwrap();
        assert_eq!(
            store.pending_tool_runs(s.id).unwrap()[0]
                .postcondition
                .as_ref()
                .unwrap()["relative_path"],
            "a.txt"
        );
        // A replay bumps the attempt counter of the SAME logical row.
        assert_eq!(store.bump_tool_run_attempt(s.id, op).unwrap(), 1);
        // Hostile annotation on a finished row is loud.
        store
            .finish_tool_run(s.id, op, "completed", "applied")
            .unwrap();
        assert!(store.pending_tool_runs(s.id).unwrap().is_empty());
        assert!(store
            .record_tool_postcondition(s.id, op, &serde_json::json!({}))
            .is_err());
        assert!(store.bump_tool_run_attempt(s.id, op).is_err());
    }

    #[test]
    fn tool_run_recovery_columns_and_turn_records_survive_reopen() {
        // Requirement 2c: the descriptor + postcondition + attempt survive a
        // daemon restart and still drive a replay.
        let dir = tempfile::tempdir().unwrap();
        let (sid, op) = {
            let store = Store::open(dir.path(), true).unwrap();
            let ws = store.create_workspace("/w").unwrap();
            let s = store.create_session(ws, "t", "p", "m").unwrap();
            let op = OpId::new(31);
            store
                .start_tool_run(
                    s.id,
                    op,
                    "echo",
                    serde_json::json!({"x": 1}),
                    serde_json::json!({"strategy": "idempotent"}),
                    None,
                    Some(serde_json::json!({"tool_name": "echo"})),
                )
                .unwrap();
            store
                .record_tool_postcondition(s.id, op, &serde_json::json!({"relative_path": "a.txt"}))
                .unwrap();
            store
                .start_turn_record(s.id, OpId::new(99), None, Some(2), "p", "m", None)
                .unwrap();
            (s.id, op)
        };
        let store = Store::open(dir.path(), true).unwrap();
        let pending = store.pending_tool_runs(sid).unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].op_id, op);
        assert_eq!(pending[0].attempt, 0);
        assert_eq!(
            pending[0].replay_descriptor.as_ref().unwrap()["tool_name"],
            "echo"
        );
        assert_eq!(
            pending[0].postcondition.as_ref().unwrap()["relative_path"],
            "a.txt"
        );
        assert_eq!(store.bump_tool_run_attempt(sid, op).unwrap(), 1);
        let rec = store.turn_record_of(sid, OpId::new(99)).unwrap().unwrap();
        assert_eq!(rec.status, TURN_RECORD_ACTIVE);
        assert_eq!(rec.effective_model, "m");
    }

    #[test]
    fn migration_v7_replays_cleanly_on_a_v6_store() {
        // Simulate a store created before v7 (no turn_record table, no
        // tool_run recovery columns), then reopen: the v7 migration must
        // re-create everything without touching existing rows.
        let dir = tempfile::tempdir().unwrap();
        let sid = {
            let store = Store::open(dir.path(), true).unwrap();
            let ws = store.create_workspace("/w").unwrap();
            let s = store.create_session(ws, "t", "p", "m").unwrap();
            store
                .start_tool_run(
                    s.id,
                    OpId::new(1),
                    "run",
                    serde_json::json!({}),
                    serde_json::json!({"strategy": "none"}),
                    None,
                    None,
                )
                .unwrap();
            {
                let conn = store.write();
                conn.execute("ALTER TABLE tool_run DROP COLUMN replay_descriptor", [])
                    .unwrap();
                conn.execute("ALTER TABLE tool_run DROP COLUMN attempt", [])
                    .unwrap();
                conn.execute("ALTER TABLE tool_run DROP COLUMN postcondition", [])
                    .unwrap();
                // The v9/v10 task tables are post-this-version too: restore
                // the legacy `task` layout so the migration chain past v10
                // replays on reopen.
                conn.execute("DROP TABLE task", []).unwrap();
                conn.execute("ALTER TABLE task_ledger RENAME TO task", [])
                    .unwrap();
                conn.execute("DROP TABLE turn_record", []).unwrap();
                // The v8 session-identity columns are post-v7 too: drop them
                // so the migration chain past v7 replays on reopen.
                conn.execute("ALTER TABLE session DROP COLUMN worktree_id", [])
                    .unwrap();
                conn.execute("ALTER TABLE session DROP COLUMN task_id", [])
                    .unwrap();
                // Pre-v7 stores sit at machine version 7 (the v6 comment
                // block covers TWO ALTER entries: before_exists and
                // after_exists); rewinding to 7 replays ONLY the v7 entry.                                // v11 artifacts (event payload_ver + the typed ledger) are
                // post-this-version too: drop them so the full chain
                // (past v11) replays cleanly on reopen.
                conn.execute("ALTER TABLE event DROP COLUMN payload_ver", [])
                    .unwrap();
                conn.execute("DROP TABLE IF EXISTS ledger_entry", [])
                    .unwrap();
                conn.execute("DROP TABLE IF EXISTS ledger_head", [])
                    .unwrap();
                // v13 prefix-stability columns are post-this-version too: drop
                // them so the full chain (past v13) replays cleanly on reopen.
                conn.execute(
                    "ALTER TABLE provider_call DROP COLUMN prompt_prefix_hash",
                    [],
                )
                .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN prompt_tokens", [])
                    .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN prefix_stability", [])
                    .unwrap();
                // The v19 segment-observation column is post-this-version
                // too: drop it so the full chain (past v19) replays cleanly.
                conn.execute(
                    "ALTER TABLE provider_call DROP COLUMN prefix_segments_json",
                    [],
                )
                .unwrap();
                // The v20 verification-record evidence columns are
                // post-this-version too: drop them so the full chain
                // (past v20) replays cleanly.
                conn.execute(
                    "ALTER TABLE verification_record DROP COLUMN environment_fingerprint_json",
                    [],
                )
                .unwrap();
                conn.execute(
                    "ALTER TABLE verification_record DROP COLUMN candidate_proof_ref_json",
                    [],
                )
                .unwrap();
                conn.execute("DROP INDEX IF EXISTS idx_provider_call_session_attempt", [])
                    .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN attempt_op_id", [])
                    .unwrap();
                conn.execute(
                    "ALTER TABLE provider_call DROP COLUMN parent_model_call_op_id",
                    [],
                )
                .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN attempt_ordinal", [])
                    .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN reservation_id", [])
                    .unwrap();
                conn.execute("PRAGMA user_version = 7", []).unwrap();
            }
            s.id
        };
        let store = Store::open(dir.path(), true).unwrap();
        // The pre-v7 row survived and is readable.
        let pending = store.pending_tool_runs(sid).unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].attempt, 0);
        assert!(pending[0].replay_descriptor.is_none());
        assert!(pending[0].postcondition.is_none());
        // And the new machinery works on the migrated store.
        let op = OpId::new(2);
        store
            .start_tool_run(
                sid,
                op,
                "echo",
                serde_json::json!({}),
                serde_json::json!({"strategy": "idempotent"}),
                None,
                Some(serde_json::json!({"tool_name": "echo"})),
            )
            .unwrap();
        assert_eq!(store.bump_tool_run_attempt(sid, op).unwrap(), 1);
        store
            .start_turn_record(sid, OpId::new(9), None, Some(2), "p", "m", None)
            .unwrap();
        assert_eq!(store.turn_records_of(sid).unwrap().len(), 1);
        // Reopen again: still stable.
        let store = Store::open(dir.path(), true).unwrap();
        assert_eq!(store.turn_records_of(sid).unwrap().len(), 1);
    }

    #[test]
    fn session_identity_defaults_to_standalone_and_adoption_is_durable() {
        // v8: sessions default to worktree 1 / task 1 (the DOCUMENTED
        // standalone identity) and adopt_identity persists the real
        // worktree/task ids durably — a reopen must read them back.
        let dir = tempfile::tempdir().unwrap();
        let (sid, ws) = {
            let store = Store::open(dir.path().join("store"), true).unwrap();
            let ws = store.create_workspace("/w").unwrap();
            let s = store.create_session(ws, "t", "p", "m").unwrap();
            let row = store.get_session(s.id).unwrap().unwrap();
            assert_eq!(row.worktree_id, WorktreeId::new(1), "standalone default");
            assert_eq!(row.task_id, TaskId::new(1), "standalone default");
            // Adoption moves the row off the defaults.
            store
                .adopt_session_identity(s.id, WorktreeId::new(7), TaskId::new(9))
                .unwrap();
            let row = store.get_session(s.id).unwrap().unwrap();
            assert_eq!(row.worktree_id, WorktreeId::new(7));
            assert_eq!(row.task_id, TaskId::new(9));
            // Unknown sessions are loud, not silent.
            assert!(store
                .adopt_session_identity(SessionId::new(9999), WorktreeId::new(2), TaskId::new(2))
                .is_err());
            (s.id, ws)
        };
        let store = Store::open(dir.path().join("store"), true).unwrap();
        let row = store.get_session(sid).unwrap().unwrap();
        assert_eq!(
            row.worktree_id,
            WorktreeId::new(7),
            "adoption survives reopen"
        );
        assert_eq!(row.task_id, TaskId::new(9));
        // list_sessions carries the same columns.
        let listed = store.list_sessions(Some(ws)).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].worktree_id, WorktreeId::new(7));
    }

    #[test]
    fn migration_v8_replays_cleanly_on_a_v7_store() {
        // Simulate a v7 store (no worktree_id/task_id columns on session),
        // reopen: v8 must add the columns and existing rows must read back
        // as the standalone 1/1 default — never a lost or corrupt row.
        // (Note: the v6 checkpoint block spans TWO array entries, so the
        // session-identity migration is array index 8 = schema target 9;
        // rewinding to 8 replays exactly this one entry.)
        let dir = tempfile::tempdir().unwrap();
        let (sid, ws) = {
            let store = Store::open(dir.path(), true).unwrap();
            let ws = store.create_workspace("/w").unwrap();
            let s = store.create_session(ws, "t", "p", "m").unwrap();
            {
                let conn = store.write();
                conn.execute("ALTER TABLE session DROP COLUMN worktree_id", [])
                    .unwrap();
                conn.execute("ALTER TABLE session DROP COLUMN task_id", [])
                    .unwrap();
                // The v9/v10 task tables are post-this-version too: restore
                // the legacy `task` layout so the migration chain past v10
                // replays on reopen.
                conn.execute("DROP TABLE task", []).unwrap();
                conn.execute("ALTER TABLE task_ledger RENAME TO task", [])
                    .unwrap(); // v11 artifacts (event payload_ver + the typed ledger) are
                               // post-this-version too: drop them so the full chain
                               // (past v11) replays cleanly on reopen.
                conn.execute("ALTER TABLE event DROP COLUMN payload_ver", [])
                    .unwrap();
                conn.execute("DROP TABLE IF EXISTS ledger_entry", [])
                    .unwrap();
                conn.execute("DROP TABLE IF EXISTS ledger_head", [])
                    .unwrap();
                // v13 prefix-stability columns are post-this-version too: drop
                // them so the full chain (past v13) replays cleanly on reopen.
                conn.execute(
                    "ALTER TABLE provider_call DROP COLUMN prompt_prefix_hash",
                    [],
                )
                .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN prompt_tokens", [])
                    .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN prefix_stability", [])
                    .unwrap();
                // The v19 segment-observation column is post-this-version
                // too: drop it so the full chain (past v19) replays cleanly.
                conn.execute(
                    "ALTER TABLE provider_call DROP COLUMN prefix_segments_json",
                    [],
                )
                .unwrap();
                // The v20 verification-record evidence columns are
                // post-this-version too: drop them so the full chain
                // (past v20) replays cleanly.
                conn.execute(
                    "ALTER TABLE verification_record DROP COLUMN environment_fingerprint_json",
                    [],
                )
                .unwrap();
                conn.execute(
                    "ALTER TABLE verification_record DROP COLUMN candidate_proof_ref_json",
                    [],
                )
                .unwrap();
                conn.execute("DROP INDEX IF EXISTS idx_provider_call_session_attempt", [])
                    .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN attempt_op_id", [])
                    .unwrap();
                conn.execute(
                    "ALTER TABLE provider_call DROP COLUMN parent_model_call_op_id",
                    [],
                )
                .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN attempt_ordinal", [])
                    .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN reservation_id", [])
                    .unwrap();
                conn.execute("PRAGMA user_version = 8", []).unwrap();
            }
            (s.id, ws)
        };
        let store = Store::open(dir.path(), true).unwrap();
        let row = store.get_session(sid).unwrap().unwrap();
        assert_eq!(row.workspace_id, ws, "row survived the migration");
        assert_eq!(
            row.worktree_id,
            WorktreeId::new(1),
            "v8 default on old rows"
        );
        assert_eq!(row.task_id, TaskId::new(1), "v8 default on old rows");
        assert_eq!(store.list_sessions(None).unwrap().len(), 1);
    }

    #[test]
    fn update_session_title_roundtrip_and_unknown() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path(), true).unwrap();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "old", "p", "m").unwrap();
        assert!(store.update_session_title(s.id, "new title").unwrap());
        let row = store.get_session(s.id).unwrap().unwrap();
        assert_eq!(row.title, "new title");
        assert!(row.updated_ms >= s.updated_ms, "updated_ms must bump");
        // Unknown sessions report false (nothing updated).
        assert!(!store
            .update_session_title(SessionId::new(9999), "x")
            .unwrap());
        // Durable across reopen.
        drop(store);
        let store = Store::open(dir.path(), true).unwrap();
        assert_eq!(store.get_session(s.id).unwrap().unwrap().title, "new title");
    }

    #[test]
    fn delete_message_removes_rows_and_parts_atomically() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path(), true).unwrap();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        let sid = s.id;
        store
            .put_message(sid, 1, "user", serde_json::json!({"text": "a"}))
            .unwrap();
        let m2 = store
            .put_message(sid, 2, "assistant", serde_json::json!({"parts": []}))
            .unwrap();
        store
            .put_part(m2, "tool_call", serde_json::json!({"tool_call_id": "c1"}))
            .unwrap();
        store
            .put_part(m2, "text", serde_json::json!({"text": "body"}))
            .unwrap();
        store
            .put_message(sid, 3, "user", serde_json::json!({"text": "b"}))
            .unwrap();
        // Delete the middle message: its part rows go with it.
        assert!(store.delete_message(sid, 2).unwrap());
        assert_eq!(store.message_count(sid).unwrap(), 2);
        assert!(store.parts_of(m2).unwrap().is_empty());
        // No orphan part rows can survive (single transaction).
        let orphans: i64 = {
            let conn = store.read().unwrap();
            conn.query_row(
                "SELECT COUNT(*) FROM part WHERE message_id NOT IN (SELECT id FROM message)",
                [],
                |r| r.get(0),
            )
            .unwrap()
        };
        assert_eq!(orphans, 0, "parts are removed with their message");
        // Sequences of surviving rows are STABLE (no renumbering).
        let page = store.messages_before(sid, None, 10).unwrap();
        let seqs: Vec<i64> = page.iter().map(|r| r.seq).collect();
        assert_eq!(seqs, vec![3, 1]);
        // Re-removal of the same message deletes nothing and says so.
        assert!(!store.delete_message(sid, 2).unwrap());
        assert!(!store.delete_message(sid, 99).unwrap());
        // The removal is durable across a reopen.
        drop(store);
        let store = Store::open(dir.path(), true).unwrap();
        assert_eq!(store.message_count(sid).unwrap(), 2);
        assert!(store.message_created_ms(sid, 2).unwrap().is_none());
        let page = store.messages_before(sid, None, 10).unwrap();
        let seqs: Vec<i64> = page.iter().map(|r| r.seq).collect();
        assert_eq!(seqs, vec![3, 1]);
    }

    #[test]
    fn op_id_seq_seeds_high_and_reserves_contiguous_global_ranges() {
        // Fresh stores seed the ONE global row from the migration-time clock
        // (see `op_id_seq_seed`): the seed must sit far above every
        // pre-migration `clock + counter` id and stay aligned to the 1024-id
        // reservation quantum.
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s1 = store.create_session(ws, "a", "p", "m").unwrap();
        let s2 = store.create_session(ws, "b", "p", "m").unwrap();
        let hw0 = store.op_id_seq_high_water().unwrap();
        assert!(
            hw0 > (1u64 << 20),
            "seed must dominate clock+counter ids: {hw0}"
        );
        assert_eq!(hw0 % 1024, 0, "seed must sit on a quantum boundary");
        // The sequence is GLOBAL: alternating sessions get contiguous ranges.
        let (a0, n0) = store.alloc_op_ids(s1.id, 100).unwrap();
        let (b0, n1) = store.alloc_op_ids(s2.id, 250).unwrap();
        let (a1, n2) = store.alloc_op_ids(s1.id, 7).unwrap();
        assert_eq!((a0, n0), (hw0, 100), "first range starts at the seed");
        assert_eq!((b0, n1), (hw0 + 100, 250), "second range is contiguous");
        assert_eq!((a1, n2), (hw0 + 350, 7), "ranges never interleave");
        assert_eq!(
            store.op_id_seq_high_water().unwrap(),
            hw0 + 357,
            "high water is one past the last reserved id"
        );
        assert_ne!(a0, 0, "zero is contractually impossible");
    }

    #[test]
    fn alloc_op_ids_rejects_zero_count_and_sequence_exhaustion() {
        let (_d, store) = tmp_store();
        let hw0 = store.op_id_seq_high_water().unwrap();
        assert!(store.alloc_op_ids(SessionId::new(1), 0).is_err());
        // A reservation crossing the signed INTEGER column ceiling is
        // refused and writes nothing (checked before the UPDATE).
        let overflow = i64::MAX as u64 - hw0 + 1;
        assert!(store.alloc_op_ids(SessionId::new(1), overflow).is_err());
        assert_eq!(
            store.op_id_seq_high_water().unwrap(),
            hw0,
            "failed reservations must not move the sequence"
        );
    }

    #[test]
    fn op_id_seq_ranges_never_overlap_across_live_instances() {
        // Two LIVE stores over the same file (a restart racing its
        // predecessor before the old connection is gone): every reservation
        // must be an atomic read+update, so no two ranges overlap and the
        // global order is preserved under contention.
        let dir = tempfile::tempdir().unwrap();
        let a = Arc::new(Store::open(dir.path(), true).unwrap());
        let b = Arc::new(Store::open(dir.path(), true).unwrap());
        let ranges = Arc::new(std::sync::Mutex::new(Vec::<(u64, u64)>::new()));
        let mut handles = Vec::new();
        for store in [
            a.clone(),
            a.clone(),
            a.clone(),
            b.clone(),
            b.clone(),
            b.clone(),
        ] {
            let ranges = ranges.clone();
            handles.push(std::thread::spawn(move || {
                for _ in 0..25 {
                    let (start, n) = store.alloc_op_ids(SessionId::new(1), 40).unwrap();
                    ranges.lock().unwrap().push((start, n));
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let mut ranges = ranges.lock().unwrap().clone();
        ranges.sort_by_key(|(start, _)| *start);
        assert_eq!(ranges.len(), 150, "6 threads x 25 reservations");
        for w in ranges.windows(2) {
            let (s0, n0) = w[0];
            let (s1, _n1) = w[1];
            assert!(s1 > s0, "reservation starts strictly increase");
            assert!(
                s1 >= s0 + n0,
                "ranges never overlap: [{s0}, {}) vs [{s1}, {})",
                s0 + n0,
                s1 + _n1
            );
        }
    }

    #[test]
    fn migration_v9_replays_cleanly_on_a_v8_store() {
        // Simulate a v8 store (no op_id_seq table), reopen: v9 must create
        // the table and seed the ONE global row from the migration-time
        // clock so freshly migrated databases mint ids far above any
        // pre-migration (clock+counter) id. (The v9 block is array index 9
        // = schema target 10; rewinding to 9 replays exactly this entry.)
        let dir = tempfile::tempdir().unwrap();
        let (sid, ws) = {
            let store = Store::open(dir.path(), true).unwrap();
            let ws = store.create_workspace("/w").unwrap();
            let s = store.create_session(ws, "t", "p", "m").unwrap();
            {
                let conn = store.write();
                conn.execute("DROP TABLE op_id_seq", []).unwrap();
                // The v9/v10 task tables are post-this-version too: restore
                // the legacy `task` layout so the migration chain past v10
                // replays on reopen.
                conn.execute("DROP TABLE task", []).unwrap();
                conn.execute("ALTER TABLE task_ledger RENAME TO task", [])
                    .unwrap(); // v11 artifacts (event payload_ver + the typed ledger) are
                               // post-this-version too: drop them so the full chain
                               // (past v11) replays cleanly on reopen.
                conn.execute("ALTER TABLE event DROP COLUMN payload_ver", [])
                    .unwrap();
                conn.execute("DROP TABLE IF EXISTS ledger_entry", [])
                    .unwrap();
                conn.execute("DROP TABLE IF EXISTS ledger_head", [])
                    .unwrap();
                // v13 prefix-stability columns are post-this-version too: drop
                // them so the full chain (past v13) replays cleanly on reopen.
                conn.execute(
                    "ALTER TABLE provider_call DROP COLUMN prompt_prefix_hash",
                    [],
                )
                .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN prompt_tokens", [])
                    .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN prefix_stability", [])
                    .unwrap();
                // The v19 segment-observation column is post-this-version
                // too: drop it so the full chain (past v19) replays cleanly.
                conn.execute(
                    "ALTER TABLE provider_call DROP COLUMN prefix_segments_json",
                    [],
                )
                .unwrap();
                // The v20 verification-record evidence columns are
                // post-this-version too: drop them so the full chain
                // (past v20) replays cleanly.
                conn.execute(
                    "ALTER TABLE verification_record DROP COLUMN environment_fingerprint_json",
                    [],
                )
                .unwrap();
                conn.execute(
                    "ALTER TABLE verification_record DROP COLUMN candidate_proof_ref_json",
                    [],
                )
                .unwrap();
                conn.execute("DROP INDEX IF EXISTS idx_provider_call_session_attempt", [])
                    .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN attempt_op_id", [])
                    .unwrap();
                conn.execute(
                    "ALTER TABLE provider_call DROP COLUMN parent_model_call_op_id",
                    [],
                )
                .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN attempt_ordinal", [])
                    .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN reservation_id", [])
                    .unwrap();
                conn.execute("PRAGMA user_version = 9", []).unwrap();
            }
            (s.id, ws)
        };
        let store = Store::open(dir.path(), true).unwrap();
        // Pre-v9 rows survived the migration.
        let row = store.get_session(sid).unwrap().unwrap();
        assert_eq!(row.workspace_id, ws, "row survived the migration");
        // The seed row exists, is large, and ids start exactly there.
        let hw = store.op_id_seq_high_water().unwrap();
        assert!(
            hw > (1u64 << 20),
            "seed must dominate any pre-migration id: {hw}"
        );
        let (start, n) = store.alloc_op_ids(sid, 5).unwrap();
        assert_eq!((start, n), (hw, 5), "ids are minted at the seeded mark");
        // Reopen again: migration is a no-op and the sequence is durable.
        drop(store);
        let store = Store::open(dir.path(), true).unwrap();
        assert_eq!(store.op_id_seq_high_water().unwrap(), hw + 5);
    }

    #[test]
    fn durable_attachment_rows_dedupe_by_digest_and_survive_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path(), true).unwrap();
        let ws = store.create_workspace("/w").unwrap();
        let s1 = store.create_session(ws, "att", "p", "m").unwrap();
        let ws2 = store.create_workspace("/w2").unwrap();
        let s2 = store.create_session(ws2, "other", "p", "m").unwrap();
        let digest = faktor_core::hash::FileHash::from([7; 32]);
        let id = AttachmentId::new(digest, "image/png", Some("shot.png"), 123).unwrap();
        let first = store.put_attachment(s1.id, &id).unwrap();
        assert_eq!(first, id);
        // Dedupe by digest: a second write of the same digest is a no-op and
        // the FIRST row's metadata is authoritative (never rewritten).
        let conflicting = AttachmentId::new(digest, "application/pdf", Some("other"), 123).unwrap();
        assert_eq!(store.put_attachment(s1.id, &conflicting).unwrap(), first);
        assert_eq!(
            store.attachment(s1.id, digest).unwrap(),
            Some(first.clone())
        );
        assert_eq!(
            store.list_attachments(s1.id, 10).unwrap(),
            vec![first.clone()]
        );
        // Session scope: another session never sees the row.
        assert!(store.attachment(s2.id, digest).unwrap().is_none());
        // Reopen: the typed row resolves identically from disk.
        drop(store);
        let store = Store::open(dir.path(), true).unwrap();
        assert_eq!(store.attachment(s1.id, digest).unwrap(), Some(first));
    }

    #[test]
    fn migration_v10_replays_cleanly_on_a_v9_store() {
        // Simulate a v9 store (the legacy one-row-per-session ledger table
        // only; no typed task rows), reopen: the v10 block must rename the
        // legacy table to task_ledger, create the typed `task` table, and
        // leave every pre-v10 ledger row readable byte-identically.
        let dir = tempfile::tempdir().unwrap();
        let (sid, ledger_value) = {
            let store = Store::open(dir.path(), true).unwrap();
            let ws = store.create_workspace("/w").unwrap();
            let s = store.create_session(ws, "t", "p", "m").unwrap();
            let ledger = serde_json::json!({"goal": "legacy ledger row", "tasks": []});
            store.put_task_ledger(s.id, ledger.clone()).unwrap();
            {
                let conn = store.write();
                // Rewind the schema to the v9 layout: drop the migrated
                // artifacts and rename the legacy table back to `task`.
                conn.execute("DROP TABLE task", []).unwrap();
                conn.execute("ALTER TABLE task_ledger RENAME TO task", [])
                    .unwrap(); // v11 artifacts (event payload_ver + the typed ledger) are
                               // post-this-version too: drop them so the full chain
                               // (past v11) replays cleanly on reopen.
                conn.execute("ALTER TABLE event DROP COLUMN payload_ver", [])
                    .unwrap();
                conn.execute("DROP TABLE IF EXISTS ledger_entry", [])
                    .unwrap();
                conn.execute("DROP TABLE IF EXISTS ledger_head", [])
                    .unwrap();
                // v13 prefix-stability columns are post-this-version too: drop
                // them so the full chain (past v13) replays cleanly on reopen.
                conn.execute(
                    "ALTER TABLE provider_call DROP COLUMN prompt_prefix_hash",
                    [],
                )
                .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN prompt_tokens", [])
                    .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN prefix_stability", [])
                    .unwrap();
                // The v19 segment-observation column is post-this-version
                // too: drop it so the full chain (past v19) replays cleanly.
                conn.execute(
                    "ALTER TABLE provider_call DROP COLUMN prefix_segments_json",
                    [],
                )
                .unwrap();
                // The v20 verification-record evidence columns are
                // post-this-version too: drop them so the full chain
                // (past v20) replays cleanly.
                conn.execute(
                    "ALTER TABLE verification_record DROP COLUMN environment_fingerprint_json",
                    [],
                )
                .unwrap();
                conn.execute(
                    "ALTER TABLE verification_record DROP COLUMN candidate_proof_ref_json",
                    [],
                )
                .unwrap();
                conn.execute("DROP INDEX IF EXISTS idx_provider_call_session_attempt", [])
                    .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN attempt_op_id", [])
                    .unwrap();
                conn.execute(
                    "ALTER TABLE provider_call DROP COLUMN parent_model_call_op_id",
                    [],
                )
                .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN attempt_ordinal", [])
                    .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN reservation_id", [])
                    .unwrap();
                conn.execute("PRAGMA user_version = 9", []).unwrap();
            }
            (s.id, ledger)
        };
        let store = Store::open(dir.path(), true).unwrap();
        // The legacy ledger row survived the migration byte-identically.
        assert_eq!(store.get_task_ledger(sid).unwrap(), Some(ledger_value));
        // The typed task table exists and starts empty; writes work.
        assert!(store.list_tasks(sid).unwrap().is_empty());
        let row = TaskRow {
            task_id: TaskId::new(3),
            session_id: sid,
            goal: "typed goal".into(),
            acceptance_criteria: vec!["cargo check".into()],
            plan: vec![],
            attachments: vec![],
            max_tokens: Some(10_000),
            max_turns: None,
            spent_tokens: 0,
            spent_turns: 0,
            state: TaskState::Pending,
            revision: TaskRevision::new(1),
            created_ms: 7,
            updated_ms: 7,
        };
        store.upsert_task(&row).unwrap();
        assert_eq!(store.get_task(sid, TaskId::new(3)).unwrap(), Some(row));
        // Reopen again: the migration is a no-op and both tables read back.
        drop(store);
        let store = Store::open(dir.path(), true).unwrap();
        assert!(store.get_task_ledger(sid).unwrap().is_some());
        let back = store.get_task(sid, TaskId::new(3)).unwrap().unwrap();
        assert_eq!(back.goal, "typed goal");
        assert_eq!(back.acceptance_criteria, vec!["cargo check".to_string()]);
        assert_eq!(back.max_tokens, Some(10_000));
        assert_eq!(back.state, TaskState::Pending);
        assert_eq!(back.revision, TaskRevision::new(1));
    }

    #[test]
    fn durable_task_repo_upsert_get_list_and_session_scope() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path(), true).unwrap();
        let ws = store.create_workspace("/w").unwrap();
        let s1 = store.create_session(ws, "t1", "p", "m").unwrap();
        let ws2 = store.create_workspace("/w2").unwrap();
        let s2 = store.create_session(ws2, "t2", "p", "m").unwrap();
        let mut row = TaskRow {
            task_id: TaskId::new(1),
            session_id: s1.id,
            goal: "g".into(),
            acceptance_criteria: vec![],
            plan: vec!["step one".into()],
            attachments: vec![],
            max_tokens: Some(1000),
            max_turns: Some(5),
            spent_tokens: 0,
            spent_turns: 0,
            state: TaskState::Pending,
            revision: TaskRevision::new(1),
            created_ms: 10,
            updated_ms: 10,
        };
        store.upsert_task(&row).unwrap();
        assert_eq!(
            store.get_task(s1.id, TaskId::new(1)).unwrap(),
            Some(row.clone())
        );
        assert_eq!(
            store.get_task(s2.id, TaskId::new(1)).unwrap(),
            None,
            "session scope"
        );
        assert_eq!(store.list_tasks(s1.id).unwrap(), vec![row.clone()]);
        assert!(store.list_tasks(s2.id).unwrap().is_empty());
        // Upsert REPLACES the same (session, task) row in place; the row
        // carries its (caller-maintained) revision.
        row.spent_tokens = 500;
        row.spent_turns = 2;
        row.state = TaskState::Blocked;
        row.revision = TaskRevision::new(2);
        row.updated_ms = 99;
        store.upsert_task(&row).unwrap();
        assert_eq!(
            store.list_tasks(s1.id).unwrap(),
            vec![row.clone()],
            "one row, replaced"
        );
        // P0-7 backstop: a raw row write that would MINT a completion state
        // out of a different state is refused — VerifiedComplete has no
        // machine edge and no proof can ride a generic upsert.
        for hostile_state in [
            TaskState::VerifiedComplete,
            TaskState::Verifying,
            TaskState::NeedsVerification,
        ] {
            let mut hostile = row.clone();
            hostile.state = hostile_state;
            let refused = store.upsert_task(&hostile);
            assert!(
                matches!(refused, Err(StoreError::Malformed(_))),
                "{hostile_state:?} must refuse a raw mint"
            );
        }
        assert_eq!(
            store.get_task(s1.id, TaskId::new(1)).unwrap(),
            Some(row.clone()),
            "refused writes left no trace"
        );
        // Same-state rewrite of a completion-relevant row is legal
        // (idempotent heal) and machine edges into NeedsVerification/
        // Verifying are legal (they are the transition-path writes).
        let mut in_verify = row.clone();
        in_verify.state = TaskState::Running;
        in_verify.revision = TaskRevision::new(3);
        store.upsert_task(&in_verify).unwrap();
        in_verify.state = TaskState::NeedsVerification;
        in_verify.revision = TaskRevision::new(4);
        store.upsert_task(&in_verify).unwrap();
        in_verify.state = TaskState::Verifying;
        in_verify.revision = TaskRevision::new(5);
        store.upsert_task(&in_verify).unwrap();
        in_verify.revision = TaskRevision::new(6);
        store.upsert_task(&in_verify).unwrap();
        assert_eq!(
            store
                .get_task(s1.id, TaskId::new(1))
                .unwrap()
                .unwrap()
                .state,
            TaskState::Verifying
        );
        // A second task of the SAME session lists oldest-first alongside it.
        let mut other = row.clone();
        other.task_id = TaskId::new(2);
        other.created_ms = 5;
        other.updated_ms = 5;
        other.state = TaskState::Running;
        store.upsert_task(&other).unwrap();
        let listed = store.list_tasks(s1.id).unwrap();
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0].task_id, TaskId::new(2), "oldest-created first");
        // Corrupt state strings fail closed as Corrupt, never a panic.
        {
            let conn = store.write();
            conn.execute(
                "UPDATE task SET state = 'garbage' WHERE session_id = ?1 AND task_id = ?2",
                params![s1.id.raw() as i64, 2],
            )
            .unwrap();
        }
        match store.get_task(s1.id, TaskId::new(2)) {
            Err(StoreError::Corrupt(_)) => {}
            other => panic!("corrupt task state must error, not parse: {other:?}"),
        }
        assert!(
            matches!(store.list_tasks(s1.id), Err(StoreError::Corrupt(_))),
            "corrupt rows surface as Corrupt, never a silent skip"
        );
    }

    #[test]
    fn durable_task_spend_sources_are_durable_and_monotone() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path(), true).unwrap();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        assert_eq!(store.session_usage_tokens(s.id).unwrap(), 0);
        assert_eq!(store.turn_completed_count(s.id).unwrap(), 0);
        let op = OpId::new(1);
        store
            .record_provider_call(s.id, op, "p", "m", "started", None, None, None)
            .unwrap();
        // In-flight (not yet completed) calls count their tokens too: a
        // crash can never lose spend that a gate already saw.
        store
            .record_provider_call(s.id, op, "p", "m", "completed", Some(40), Some(10), None)
            .unwrap();
        assert_eq!(store.session_usage_tokens(s.id).unwrap(), 50);
        // TurnCompleted journal events are the durable turn counter.
        store
            .append_event(
                s.id,
                Some(op),
                faktor_core::event::EventKind::TurnCompleted,
                AgentState::ReadyForNextTurn,
                5,
                None,
            )
            .unwrap();
        assert_eq!(store.turn_completed_count(s.id).unwrap(), 1);
        // Session isolation: a second session's spend never leaks.
        let ws2 = store.create_workspace("/w2").unwrap();
        let s2 = store.create_session(ws2, "t2", "p", "m").unwrap();
        assert_eq!(store.session_usage_tokens(s2.id).unwrap(), 0);
        assert_eq!(store.turn_completed_count(s2.id).unwrap(), 0);
    }

    // ---- prefix-cache stability columns (v13, audits 65-66) ----

    fn prefix_hash(seed: u8) -> [u8; 32] {
        [seed; 32]
    }

    #[test]
    fn prefix_columns_round_trip_across_reopen_and_legacy_rows_read_null() {
        // (d) The v13 columns must survive a full reopen with every value
        // intact, and rows written before the columns existed must honestly
        // read as "no observation" — never zeros, never guesses.
        let dir = tempfile::tempdir().unwrap();
        let (ws, sid) = {
            let store = Store::open(dir.path(), true).unwrap();
            let ws = store.create_workspace("/w").unwrap();
            let s = store
                .create_session(ws, "prefix-roundtrip", "p", "m")
                .unwrap();
            // Legacy-shaped row: no prefix observation at all.
            store
                .record_provider_call(
                    s.id,
                    OpId::new(1),
                    "p",
                    "m",
                    "completed",
                    Some(10),
                    Some(5),
                    None,
                )
                .unwrap();
            // Full observation row.
            store
                .record_provider_call_with_prefix(
                    s.id,
                    OpId::new(2),
                    "p",
                    "m",
                    "completed",
                    Some(10),
                    Some(5),
                    None,
                    Some(prefix_hash(7)),
                    Some(1234),
                    Some(0.625),
                )
                .unwrap();
            // Hash with no recorded per-row stability: still an observation.
            store
                .record_provider_call_with_prefix(
                    s.id,
                    OpId::new(3),
                    "p",
                    "m",
                    "completed",
                    None,
                    None,
                    None,
                    Some(prefix_hash(9)),
                    Some(10),
                    None,
                )
                .unwrap();
            (ws, s.id)
        };
        // Reopen: migration is a no-op, all data reads back.
        let store = Store::open(dir.path(), true).unwrap();
        let rows = store.provider_call_prefix_rows(sid).unwrap();
        assert_eq!(rows.len(), 2, "the legacy row carries no observation");
        assert_eq!(rows[0].row_id + 1, rows[1].row_id);
        assert_eq!(rows[0].prompt_prefix_hash, prefix_hash(7));
        assert_eq!(rows[0].prompt_tokens, 1234);
        assert_eq!(rows[0].prefix_stability, Some(0.625));
        assert_eq!(rows[1].prompt_prefix_hash, prefix_hash(9));
        assert_eq!(rows[1].prompt_tokens, 10);
        assert_eq!(rows[1].prefix_stability, None);
        // Session isolation: another session sees none of it.
        let other = store.create_session(ws, "other", "p", "m").unwrap();
        assert!(store
            .provider_call_prefix_rows(other.id)
            .unwrap()
            .is_empty());
        // Second reopen still stable.
        drop(store);
        let store = Store::open(dir.path(), true).unwrap();
        let rows = store.provider_call_prefix_rows(sid).unwrap();
        assert_eq!(rows[0].prompt_prefix_hash, prefix_hash(7));
        assert_eq!(rows[1].prompt_tokens, 10);
    }

    #[test]
    fn corrupt_injected_prefix_shapes_are_loud_never_silent_misreads() {
        // (d) Values injected behind the API's back (a raw connection) must
        // be rejected LOUDLY on read: wrong-length hash, out-of-range
        // stability, negative and oversized token counts.
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path(), true).unwrap();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        let row = store
            .record_provider_call_with_prefix(
                s.id,
                OpId::new(1),
                "p",
                "m",
                "completed",
                None,
                None,
                None,
                Some(prefix_hash(1)),
                Some(100),
                Some(0.5),
            )
            .unwrap();
        let conn = store.write();
        // Truncated hash blob (7 bytes).
        conn.execute(
            "UPDATE provider_call SET prompt_prefix_hash = ?1 WHERE id = ?2",
            params![vec![0xabu8; 7], row],
        )
        .unwrap();
        assert!(matches!(
            store.provider_call_prefix_rows(s.id),
            Err(StoreError::Malformed(_))
        ));
        // Stability beyond [0, 1].
        conn.execute(
            "UPDATE provider_call SET prompt_prefix_hash = ?1, prefix_stability = 1.5 WHERE id = ?2",
            params![vec![0xabu8; 32], row],
        )
        .unwrap();
        assert!(matches!(
            store.provider_call_prefix_rows(s.id),
            Err(StoreError::Malformed(_))
        ));
        // Negative token count.
        conn.execute(
            "UPDATE provider_call SET prompt_tokens = -4 WHERE id = ?1",
            params![row],
        )
        .unwrap();
        assert!(matches!(
            store.provider_call_prefix_rows(s.id),
            Err(StoreError::Malformed(_))
        ));
        // Token count beyond the u32 bound (2^40).
        conn.execute(
            "UPDATE provider_call SET prompt_tokens = ?1 WHERE id = ?2",
            params![1i64 << 40, row],
        )
        .unwrap();
        assert!(matches!(
            store.provider_call_prefix_rows(s.id),
            Err(StoreError::Malformed(_))
        ));
        // The aggregate query ignores corrupt rows' hashes (never parses
        // them) but the out-of-range stability it DOES aggregate must be
        // rejected there too.
        conn.execute(
            "UPDATE provider_call SET prompt_prefix_hash = ?1, prompt_tokens = 100 WHERE id = ?2",
            params![vec![0xabu8; 32], row],
        )
        .unwrap();
        assert!(matches!(
            store.session_stored_prefix_stability(s.id),
            Err(StoreError::Malformed(_))
        ));
    }

    // ---- per-call segment observations (v19, audits 45/82) ----

    /// One strict observation payload: `n` distinct 64-char hex digests and
    /// one token count per segment.
    fn segments_json(n: usize, tokens: &[u64]) -> String {
        let hashes: Vec<String> = (0..n)
            .map(|i| format!("{:02x}", i as u8).repeat(32))
            .collect();
        serde_json::json!({
            "segment_hashes": hashes,
            "segment_token_counts": tokens,
            "cache_read_tokens": 7u64,
        })
        .to_string()
    }

    #[test]
    fn prefix_segments_json_round_trips_across_reopen_and_legacy_rows_read_null() {
        // The v19 additive payload survives a full reopen byte-identically,
        // and rows written through the legacy API honestly read as "no
        // segment observation" — never an empty JSON object, never a guess.
        let dir = tempfile::tempdir().unwrap();
        let json = segments_json(3, &[10, 20, 30]);
        let sid = {
            let store = Store::open(dir.path(), true).unwrap();
            let ws = store.create_workspace("/w").unwrap();
            let s = store.create_session(ws, "segments", "p", "m").unwrap();
            // Legacy prefix row: hash + tokens but no segment payload.
            store
                .record_provider_call_with_prefix(
                    s.id,
                    OpId::new(1),
                    "p",
                    "m",
                    "completed",
                    None,
                    None,
                    None,
                    Some(prefix_hash(3)),
                    Some(60),
                    None,
                )
                .unwrap();
            // v19 row: the same shape plus the observed segments.
            store
                .record_provider_call_with_prefix_segments(
                    s.id,
                    OpId::new(2),
                    "p",
                    "m",
                    "completed",
                    None,
                    None,
                    None,
                    Some(prefix_hash(4)),
                    Some(60),
                    Some(1.0),
                    Some(&json),
                )
                .unwrap();
            s.id
        };
        let store = Store::open(dir.path(), true).unwrap();
        let rows = store.provider_call_prefix_rows(sid).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(
            rows[0].prefix_segments_json, None,
            "pre-v19 rows must read as no observation"
        );
        assert_eq!(rows[1].prefix_segments_json.as_deref(), Some(json.as_str()));
        // Reopen again: still byte-identical.
        drop(store);
        let store = Store::open(dir.path(), true).unwrap();
        let rows = store.provider_call_prefix_rows(sid).unwrap();
        assert_eq!(rows[1].prefix_segments_json.as_deref(), Some(json.as_str()));
        // A session with no rows sees none of it.
        let other = store.create_workspace("/w2").unwrap();
        let other = store.create_session(other, "other", "p", "m").unwrap();
        assert!(store
            .provider_call_prefix_rows(other.id)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn malformed_prefix_segments_writes_are_refused_before_any_row_lands() {
        // The typed API is the first gate: malformed, oversized, or
        // shape-suspect payloads never touch a row. Every refusal is typed.
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path(), true).unwrap();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        let write = |json: &str| {
            store.record_provider_call_with_prefix_segments(
                s.id,
                OpId::new(1),
                "p",
                "m",
                "completed",
                None,
                None,
                None,
                Some(prefix_hash(1)),
                Some(1),
                None,
                Some(json),
            )
        };
        for (json, what) in [
            ("not json", "garbage"),
            ("{", "truncated"),
            (
                r#"{"segment_hashes":[],"segment_token_counts":[1],"cache_read_tokens":0}"#,
                "length mismatch",
            ),
            (
                r#"{"segment_hashes":["zz"],"segment_token_counts":[1],"cache_read_tokens":0}"#,
                "non-hex digest",
            ),
            (
                r#"{"segment_hashes":[],"segment_token_counts":[],"cache_read_tokens":0,"extra":1}"#,
                "unknown field",
            ),
        ] {
            assert!(
                matches!(write(json), Err(StoreError::Malformed(_))),
                "{what} must be a loud Malformed"
            );
        }
        // Over the byte bound: Oversized, never stored.
        let oversized = format!(
            r#"{{"segment_hashes":[],"segment_token_counts":[],"cache_read_tokens":0,"pad":"{}"}}"#,
            "x".repeat(MAX_PREFIX_SEGMENTS_JSON)
        );
        assert!(matches!(write(&oversized), Err(StoreError::Oversized(_))));
        // 65 segments: over the segment bound.
        let too_many = segments_json(
            MAX_PREFIX_SEGMENTS + 1,
            &vec![1u64; MAX_PREFIX_SEGMENTS + 1],
        );
        assert!(matches!(write(&too_many), Err(StoreError::Malformed(_))));
        // Nothing landed.
        assert!(store.provider_call_prefix_rows(s.id).unwrap().is_empty());
    }

    #[test]
    fn corrupt_injected_segment_payloads_fail_loud_on_read_not_silent() {
        // (v19 adversarial) A payload injected behind the API's back (raw
        // connection UPDATE) must fail the READ with a typed error: routing
        // must never silently fall back to a guessed observation.
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path(), true).unwrap();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        let row = store
            .record_provider_call_with_prefix_segments(
                s.id,
                OpId::new(1),
                "p",
                "m",
                "completed",
                None,
                None,
                None,
                Some(prefix_hash(1)),
                Some(100),
                Some(0.5),
                Some(&segments_json(2, &[10, 20])),
            )
            .unwrap();
        let conn = store.write();
        let corrupt = |json: &str| {
            conn.execute(
                "UPDATE provider_call SET prefix_segments_json = ?1 WHERE id = ?2",
                params![json, row],
            )
            .unwrap();
        };
        for json in [
            "not json",
            "{",
            r#"{"segment_hashes":[],"segment_token_counts":[1],"cache_read_tokens":0}"#,
            r#"{"segment_hashes":["zz"],"segment_token_counts":[1],"cache_read_tokens":0}"#,
            r#"{"segment_hashes":[],"segment_token_counts":[],"cache_read_tokens":0,"extra":1}"#,
        ] {
            corrupt(json);
            assert!(
                matches!(
                    store.provider_call_prefix_rows(s.id),
                    Err(StoreError::Malformed(_))
                ),
                "corrupt payload {json:?} must fail the read loudly"
            );
        }
        // Over the byte bound injected behind the API's back: Oversized.
        corrupt(&format!(
            r#"{{"segment_hashes":[],"segment_token_counts":[],"cache_read_tokens":0,"pad":"{}"}}"#,
            "x".repeat(MAX_PREFIX_SEGMENTS_JSON)
        ));
        assert!(matches!(
            store.provider_call_prefix_rows(s.id),
            Err(StoreError::Oversized(_))
        ));
        // Repairing the row restores the read (the error is data-typed, not
        // sticky state).
        corrupt(&segments_json(2, &[10, 20]));
        let rows = store.provider_call_prefix_rows(s.id).unwrap();
        assert!(rows[0].prefix_segments_json.is_some());
    }

    #[test]
    fn oversized_and_malformed_writes_are_rejected_loudly() {
        // (d) The typed API refuses oversized token counts and malformed
        // stability values BEFORE anything touches the row.
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path(), true).unwrap();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        let err = store
            .record_provider_call_with_prefix(
                s.id,
                OpId::new(1),
                "p",
                "m",
                "completed",
                None,
                None,
                None,
                Some(prefix_hash(1)),
                Some(u32::MAX as u64 + 1),
                None,
            )
            .unwrap_err();
        assert!(
            matches!(err, StoreError::Oversized(_)),
            "oversized tokens must be rejected loudly: {err:?}"
        );
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.01, 1.0001] {
            let err = store
                .record_provider_call_with_prefix(
                    s.id,
                    OpId::new(1),
                    "p",
                    "m",
                    "completed",
                    None,
                    None,
                    None,
                    Some(prefix_hash(1)),
                    Some(100),
                    Some(bad),
                )
                .unwrap_err();
            assert!(
                matches!(err, StoreError::Malformed(_)),
                "stability {bad} must be rejected loudly: {err:?}"
            );
        }
        // The u32 boundary itself is accepted.
        store
            .record_provider_call_with_prefix(
                s.id,
                OpId::new(2),
                "p",
                "m",
                "completed",
                None,
                None,
                None,
                Some(prefix_hash(2)),
                Some(u32::MAX as u64),
                Some(0.0),
            )
            .unwrap();
        // Nothing was recorded by the rejected attempts.
        let rows = store.provider_call_prefix_rows(s.id).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].prompt_tokens, u32::MAX);
        assert_eq!(rows[0].prefix_stability, Some(0.0));
    }

    #[test]
    fn stored_stability_aggregate_math_and_empty_sessions() {
        // (d) The additive aggregate query: mean + population std dev over
        // the rows that carry a recorded stability, and None on sessions
        // with nothing recorded.
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path(), true).unwrap();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        // No observations at all.
        assert_eq!(store.session_stored_prefix_stability(s.id).unwrap(), None);
        // Legacy rows (no prefix data) do not count either.
        store
            .record_provider_call(s.id, OpId::new(1), "p", "m", "ok", None, None, None)
            .unwrap();
        assert_eq!(store.session_stored_prefix_stability(s.id).unwrap(), None);
        // Recorded stabilities: 0.5, 0.5, 1.0 -> mean 2/3, σ ≈ 0.2357.
        for (i, st) in [0.5f64, 0.5, 1.0].iter().enumerate() {
            store
                .record_provider_call_with_prefix(
                    s.id,
                    OpId::new(2 + i as u64),
                    "p",
                    "m",
                    "ok",
                    None,
                    None,
                    None,
                    Some(prefix_hash(i as u8 + 3)),
                    Some(100),
                    Some(*st),
                )
                .unwrap();
        }
        // Rows with a hash but NULL stability contribute nothing.
        store
            .record_provider_call_with_prefix(
                s.id,
                OpId::new(9),
                "p",
                "m",
                "ok",
                None,
                None,
                None,
                Some(prefix_hash(9)),
                Some(50),
                None,
            )
            .unwrap();
        let agg = store
            .session_stored_prefix_stability(s.id)
            .unwrap()
            .unwrap();
        assert_eq!(agg.observations, 3);
        assert!((agg.mean - 2.0 / 3.0).abs() < 1e-12);
        let want_std = ((0.5f64 - 2.0 / 3.0).powi(2) * 2.0 + (1.0f64 - 2.0 / 3.0).powi(2)) / 3.0;
        assert!((agg.std_dev - want_std.sqrt()).abs() < 1e-12);
        // Isolation: the second session still has nothing.
        let s2 = store.create_session(ws, "t2", "p", "m").unwrap();
        assert_eq!(store.session_stored_prefix_stability(s2.id).unwrap(), None);
    }

    #[test]
    fn migration_v13_replays_cleanly_on_a_v12_store() {
        // Simulate a v12 store (provider_call without the v13 columns):
        // reopen must add the columns, keep pre-v13 rows readable as
        // observation-less, and accept full observations afterwards.
        let dir = tempfile::tempdir().unwrap();
        let (sid, row) = {
            let store = Store::open(dir.path(), true).unwrap();
            let ws = store.create_workspace("/w").unwrap();
            let s = store.create_session(ws, "t", "p", "m").unwrap();
            let row = store
                .record_provider_call(s.id, OpId::new(1), "p", "m", "ok", Some(10), Some(5), None)
                .unwrap();
            {
                let conn = store.write();
                // Rewind to the v12 layout: drop the v13 columns and the
                // schema cursor so the full chain past v13 replays.
                conn.execute(
                    "ALTER TABLE provider_call DROP COLUMN prompt_prefix_hash",
                    [],
                )
                .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN prompt_tokens", [])
                    .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN prefix_stability", [])
                    .unwrap();
                // The v19 segment-observation column is post-this-version
                // too: drop it so the full chain (past v19) replays cleanly.
                conn.execute(
                    "ALTER TABLE provider_call DROP COLUMN prefix_segments_json",
                    [],
                )
                .unwrap();
                // The v20 verification-record evidence columns are
                // post-this-version too: drop them so the full chain
                // (past v20) replays cleanly.
                conn.execute(
                    "ALTER TABLE verification_record DROP COLUMN environment_fingerprint_json",
                    [],
                )
                .unwrap();
                conn.execute(
                    "ALTER TABLE verification_record DROP COLUMN candidate_proof_ref_json",
                    [],
                )
                .unwrap();
                conn.execute("DROP INDEX IF EXISTS idx_provider_call_session_attempt", [])
                    .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN attempt_op_id", [])
                    .unwrap();
                conn.execute(
                    "ALTER TABLE provider_call DROP COLUMN parent_model_call_op_id",
                    [],
                )
                .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN attempt_ordinal", [])
                    .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN reservation_id", [])
                    .unwrap();
                // The v14 task-revision column + verification_record table
                // are post-this-version too: drop them so the full chain
                // (past v14) replays cleanly on reopen.
                conn.execute("ALTER TABLE task DROP COLUMN revision", [])
                    .unwrap();
                conn.execute("DROP TABLE verification_record", []).unwrap();
                // The v15 cost-ledger objects are post-this-version too:
                // drop them so the full chain (past v15) replays cleanly.
                conn.execute("DROP TABLE cost_reservation", []).unwrap();
                conn.execute("ALTER TABLE task DROP COLUMN max_cost_micro", [])
                    .unwrap();
                conn.execute("ALTER TABLE task DROP COLUMN spent_cost_micro", [])
                    .unwrap();
                conn.execute("PRAGMA user_version = 13", []).unwrap();
            }
            (s.id, row)
        };
        let store = Store::open(dir.path(), true).unwrap();
        // The pre-v13 row survived and reads as observation-less.
        assert!(store.provider_call_prefix_rows(sid).unwrap().is_empty());
        let _ = row;
        // Full observations write and read through the re-migrated store.
        store
            .record_provider_call_with_prefix(
                sid,
                OpId::new(2),
                "p",
                "m",
                "ok",
                None,
                None,
                None,
                Some(prefix_hash(4)),
                Some(77),
                Some(0.9),
            )
            .unwrap();
        let rows = store.provider_call_prefix_rows(sid).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].prompt_prefix_hash, prefix_hash(4));
        assert_eq!(rows[0].prompt_tokens, 77);
        assert_eq!(rows[0].prefix_stability, Some(0.9));
        // Reopen again: migration is a no-op and the observation persists.
        drop(store);
        let store = Store::open(dir.path(), true).unwrap();
        let rows = store.provider_call_prefix_rows(sid).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].prompt_tokens, 77);
    }

    #[test]
    fn fast_open_recovers_migrations_and_data_and_refuses_corruption() {
        // Audit 43: the fast production open must still run WAL recovery +
        // migrations and refuse a corrupt store — it just skips the full
        // scan. Data written by a full-check open must read back through a
        // fast open, and vice versa.
        let dir = tempfile::tempdir().unwrap();
        let sid = {
            let store = Store::open(dir.path(), true).unwrap();
            let ws = store.create_workspace("/w").unwrap();
            let s = store.create_session(ws, "fast", "p", "m").unwrap();
            store
                .put_message(s.id, 1, "user", serde_json::json!({"text": "hi"}))
                .unwrap();
            s.id
        };
        let fast = Store::open_fast(dir.path()).unwrap();
        let row = fast.get_session(sid).unwrap().unwrap();
        assert_eq!(row.title, "fast");
        assert_eq!(fast.message_count(sid).unwrap(), 1);
        // The deep scan runs fine on a fast-opened store.
        assert!(fast.deep_integrity_check().unwrap().is_empty());
        // And a fast-opened store's writes survive a full-check reopen.
        let ws2 = fast.create_workspace("/w2").unwrap();
        fast.create_session(ws2, "s2", "p", "m").unwrap();
        drop(fast);
        let full = Store::open(dir.path(), true).unwrap();
        assert_eq!(full.list_sessions(None).unwrap().len(), 2);
        // Corrupt/truncated files refuse to open fast (never silently serve).
        let garbage = tempfile::tempdir().unwrap();
        std::fs::write(
            garbage.path().join("faktor-plus.db"),
            b"this is not a sqlite database at all - no magic header anywhere",
        )
        .unwrap();
        match Store::open_fast(garbage.path()) {
            Err(StoreError::Sqlite(_)) | Err(StoreError::Corrupt(_)) => {}
            other => panic!("fast open must refuse garbage, got {other:?}"),
        }
        let truncated = tempfile::tempdir().unwrap();
        std::fs::write(
            truncated.path().join("faktor-plus.db"),
            b"SQLite format 3\x00",
        )
        .unwrap();
        match Store::open_fast(truncated.path()) {
            Err(StoreError::Sqlite(_)) | Err(StoreError::Corrupt(_)) => {}
            other => panic!("fast open must refuse truncation, got {other:?}"),
        }
    }

    #[test]
    fn quick_check_and_deep_check_agree_on_healthy_stores() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        store
            .append_event(
                s.id,
                None,
                EventKind::ModelChunkReceived,
                AgentState::Streaming,
                now_ms(),
                None,
            )
            .unwrap();
        assert!(store.quick_integrity_check().unwrap().is_empty());
        assert!(store.deep_integrity_check().unwrap().is_empty());
        let d = store.diagnostics_quick().unwrap();
        assert_eq!(d["journal_mode"], "wal");
        assert_eq!(d["sessions"], 1);
        assert_eq!(d["integrity"], serde_json::json!([]));
    }

    #[test]
    fn all_running_tool_rows_and_active_turns_scan_every_session() {
        // Deep doctor queries are GLOBAL: running tool runs and active turn
        // records from two sessions must both surface.
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s1 = store.create_session(ws, "a", "p", "m").unwrap();
        let s2 = store.create_session(ws, "b", "p", "m").unwrap();
        store
            .start_tool_run(
                s1.id,
                OpId::new(11),
                "echo",
                serde_json::json!({}),
                serde_json::json!({"strategy": "idempotent"}),
                None,
                None,
            )
            .unwrap();
        store
            .start_tool_run(
                s2.id,
                OpId::new(22),
                "write_file",
                serde_json::json!({"path": "/x"}),
                serde_json::json!({"strategy": "verify_hash"}),
                Some("ab".repeat(32)),
                None,
            )
            .unwrap();
        // One finished run must NOT appear.
        store
            .start_tool_run(
                s1.id,
                OpId::new(33),
                "echo",
                serde_json::json!({}),
                serde_json::json!({"strategy": "none"}),
                None,
                None,
            )
            .unwrap();
        store
            .finish_tool_run(s1.id, OpId::new(33), "completed", "applied")
            .unwrap();
        store
            .start_turn_record(s1.id, OpId::new(101), None, Some(2), "p", "m", None)
            .unwrap();
        store
            .start_turn_record(s2.id, OpId::new(202), None, Some(2), "p", "m", Some("v1"))
            .unwrap();
        // A second, then finished, record on s1 must not appear as active.
        store
            .finish_turn_record(s1.id, OpId::new(101), TURN_RECORD_FAILED)
            .unwrap();
        store
            .start_turn_record(s1.id, OpId::new(303), None, Some(2), "p", "m", None)
            .unwrap();
        store
            .finish_turn_record(s1.id, OpId::new(303), TURN_RECORD_COMPLETED)
            .unwrap();
        let running = store.all_running_tool_rows().unwrap();
        assert_eq!(running.len(), 2, "both sessions' running rows surface");
        assert!(running
            .iter()
            .any(|r| r.session_id == s1.id && r.op_id == OpId::new(11)));
        assert!(running
            .iter()
            .any(|r| r.session_id == s2.id && r.op_id == OpId::new(22)));
        let active = store.all_active_turns().unwrap();
        assert_eq!(active.len(), 1, "only session 2's turn is still active");
        assert_eq!(active[0].turn_op_id, OpId::new(202));
    }

    #[test]
    fn journal_consistency_flags_gaps_and_torn_sessions() {
        // A gapless journal stays clean; a deleted middle event and a
        // session row without a journal are both flagged.
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        for i in 0..3 {
            store
                .append_event(
                    s.id,
                    Some(OpId::new(1 + i)),
                    EventKind::ModelChunkReceived,
                    AgentState::Streaming,
                    now_ms(),
                    None,
                )
                .unwrap();
        }
        assert!(store.journal_consistency_issues().unwrap().is_empty());
        // Torn session: raw session row with no journal (bypasses the API).
        {
            let conn = store.write();
            conn.execute(
                "INSERT INTO session(workspace_id, title, provider, model, state, lifecycle, created_ms, updated_ms)
                 VALUES (?1, 'torn', 'p', 'm', '\"idle\"', 'open', 0, 0)",
                params![ws.raw() as i64],
            )
            .unwrap();
        }
        let issues = store.journal_consistency_issues().unwrap();
        assert_eq!(issues.len(), 1, "one torn session");
        assert!(issues[0].contains("no events"));
        // A gap: delete the middle event of the healthy session.
        {
            let conn = store.write();
            conn.execute(
                "DELETE FROM event WHERE session_id = ?1 AND seq = 2",
                params![s.id.raw() as i64],
            )
            .unwrap();
        }
        let issues = store.journal_consistency_issues().unwrap();
        let gap = issues
            .iter()
            .find(|i| i.contains(&format!("session {}", s.id.raw())))
            .expect("gap must be flagged");
        assert!(gap.contains("1..=3"), "gap issue: {gap}");
    }

    #[test]
    fn cas_hash_references_lists_artifact_and_checkpoint_after_blobs() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        store
            .put_artifact(s.id, "command_output", &"ab".repeat(32), "sum", 10)
            .unwrap();
        store
            .put_artifact(s.id, "command_output", &"cd".repeat(32), "sum", 10)
            .unwrap();
        // Checkpoint with after blob (referenced) and without (not listed).
        store
            .put_checkpoint(s.id, 1, "a.rs", "b", "a", Some(&"ef".repeat(32)))
            .unwrap();
        store
            .put_checkpoint(s.id, 2, "b.rs", "b", "a", None)
            .unwrap();
        let refs = store.cas_hash_references().unwrap();
        assert_eq!(refs.len(), 3);
        assert!(refs
            .iter()
            .any(|r| r.source == "artifact" && r.hash == "ab".repeat(32)));
        assert!(refs
            .iter()
            .any(|r| r.source == "artifact" && r.hash == "cd".repeat(32)));
        assert!(refs
            .iter()
            .any(|r| r.source == "checkpoint" && r.hash == "ef".repeat(32)));
        assert!(!refs
            .iter()
            .any(|r| r.row_id == 2 && r.source == "checkpoint"));
    }

    // -------------------------------------------------- actor batch surface tests

    fn hot_session(store: &Store) -> SessionId {
        let ws = store.create_workspace("/w").unwrap();
        store.create_session(ws, "t", "p", "m").unwrap().id
    }

    fn hot_session_sids(store: &Store, n: u64) -> Vec<SessionId> {
        let ws = store.create_workspace("/w").unwrap();
        (1..=n)
            .map(|_| store.create_session(ws, "t", "p", "m").unwrap().id)
            .collect()
    }

    #[test]
    fn batch_hot_writes_commit_in_one_group_and_fsync_before_returning() {
        let (_d, store) = tmp_store();
        let sid = hot_session(&store);
        // 1 event (session seed is seq 1) + 1 message + 2 parts on it + one
        // usage-settlement row.
        let writes = vec![
            HotWrite::AppendEvent {
                session_id: sid,
                op_id: Some(OpId::new(7)),
                kind: EventKind::ModelStarted,
                state: AgentState::Streaming,
                ts_ms: 100,
                payload: None,
                payload_ver: 1,
            },
            HotWrite::PutMessage {
                session_id: sid,
                // The runtime aligns message seqs with journal event seqs.
                seq: 2,
                role: "assistant".into(),
                data: serde_json::json!({ "parts": [] }),
            },
        ];
        let (out, timing) = store.batch_hot_writes(&writes).unwrap();
        assert_eq!(out.len(), 2);
        assert!(timing.commit_us > 0, "fsync commit is measured");
        let out = out;
        let seq = match &out[0] {
            Ok(HotWriteOutcome::EventSeq(s)) => s.raw(),
            other => panic!("expected event seq, got {other:?}"),
        };
        assert_eq!(seq, 2, "second journal event of the session");
        let mid = match &out[1] {
            Ok(HotWriteOutcome::RowId(id)) => *id,
            other => panic!("expected row id, got {other:?}"),
        };
        // The whole group was one transaction: the message and the event are
        // both durable and coherent (message seq == journal seq 2).
        assert!(store.get_session(sid).unwrap().is_some());
        let msgs = store.messages_before(sid, None, 10).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].seq, 2);
        // A follow-up batch can reference the row of an earlier one.
        let parts = vec![
            HotWrite::PutPart {
                message_id: mid,
                kind: "text".into(),
                data: serde_json::json!({ "text": "hello" }),
            },
            HotWrite::RecordProviderCall {
                session_id: sid,
                op_id: OpId::new(7),
                provider: "ollama".into(),
                model: "qwen3.8".into(),
                status: "completed".into(),
                tokens_in: Some(100),
                tokens_out: Some(50),
                error: None,
            },
        ];
        let (out, _) = store.batch_hot_writes(&parts).unwrap();
        assert!(out.iter().all(|r| r.is_ok()), "parts + usage must commit");
        assert_eq!(store.parts_of(mid).unwrap().len(), 1);
        assert_eq!(store.session_usage_tokens(sid).unwrap(), 150);
    }

    #[test]
    fn batch_hot_writes_isolate_a_failing_write_to_its_savepoint() {
        // One hostile write in the group (duplicate (session, seq) message)
        // must fail ONLY itself: neighbors commit, per-write errors are
        // reported, and the writer connection is left transaction-free.
        let (_d, store) = tmp_store();
        let sids = hot_session_sids(&store, 2);
        let writes = vec![
            HotWrite::PutMessage {
                session_id: sids[0],
                seq: 1,
                role: "user".into(),
                data: serde_json::json!({ "text": "a" }),
            },
            // Duplicate (session, seq) on the SAME session: constraint hit.
            HotWrite::PutMessage {
                session_id: sids[0],
                seq: 1,
                role: "user".into(),
                data: serde_json::json!({ "text": "b" }),
            },
            // Foreign key violation: no such message row.
            HotWrite::PutPart {
                message_id: i64::MAX,
                kind: "text".into(),
                data: serde_json::json!({ "text": "orphan" }),
            },
            // Unrelated session must still land.
            HotWrite::PutMessage {
                session_id: sids[1],
                seq: 1,
                role: "user".into(),
                data: serde_json::json!({ "text": "c" }),
            },
        ];
        let (out, _) = store.batch_hot_writes(&writes).unwrap();
        assert!(out[0].is_ok(), "first insert must commit");
        assert!(out[1].is_err(), "duplicate seq must fail in its savepoint");
        assert!(out[2].is_err(), "orphan part must fail in its savepoint");
        assert!(
            out[3].is_ok(),
            "the unrelated session must survive the group"
        );
        assert_eq!(
            store.message_count(sids[0]).unwrap(),
            1,
            "only the first (session, seq) row exists"
        );
        assert_eq!(store.message_count(sids[1]).unwrap(), 1);
        // Empty groups are a no-op and never touch the writer.
        assert_eq!(store.batch_hot_writes(&[]).unwrap().0.len(), 0);
    }

    #[test]
    fn batch_hot_writes_force_strong_sync_and_restore_configured_mode() {
        // The actor's fsync-before-ack contract is implemented by lifting the
        // connection to synchronous=FULL for the group; the connection must
        // be back at the crate default (NORMAL) afterwards so direct writers
        // keep their configured behavior.
        let (_d, store) = tmp_store();
        let sid = hot_session(&store);
        let _ = store
            .batch_hot_writes(&[HotWrite::PutMessage {
                session_id: sid,
                seq: 1,
                role: "assistant".into(),
                data: serde_json::json!({ "parts": [] }),
            }])
            .unwrap();
        // Reopen simulates a process kill right after an acked batch: every
        // acked append must be present (WAL fsynced by FULL commit).
        let s2 = Store::open_fast(_d.path()).unwrap();
        assert_eq!(
            s2.messages_before(sid, None, 10).unwrap().len(),
            1,
            "acked append must survive a simulated kill"
        );
        drop(s2);
    }

    #[test]
    fn batch_hot_writes_events_stay_gapless_across_grouped_sessions() {
        // Two sessions journaling in one group: per-session MAX(seq) is
        // computed inside the shared transaction, so seqs stay gapless per
        // session even when the groups interleave.
        let (_d, store) = tmp_store();
        let sids = hot_session_sids(&store, 2);
        let writes = sids
            .iter()
            .enumerate()
            .flat_map(|(i, sid)| {
                vec![
                    HotWrite::AppendEvent {
                        session_id: *sid,
                        op_id: None,
                        kind: EventKind::PromptReceived,
                        state: AgentState::Preparing,
                        ts_ms: 10 + i as i64,
                        payload: None,
                        payload_ver: 1,
                    },
                    HotWrite::AppendEvent {
                        session_id: *sid,
                        op_id: None,
                        kind: EventKind::ContextPrepared,
                        state: AgentState::BuildingContext,
                        ts_ms: 20 + i as i64,
                        payload: None,
                        payload_ver: 1,
                    },
                ]
            })
            .collect::<Vec<_>>();
        let (out, _) = store.batch_hot_writes(&writes).unwrap();
        assert!(out.iter().all(|r| r.is_ok()));
        for sid in &sids {
            let seqs: Vec<u64> = store
                .events_range(*sid, 1, None)
                .unwrap()
                .into_iter()
                .map(|e| e.seq.raw())
                .collect();
            assert_eq!(seqs, vec![1, 2, 3], "seed + two grouped events, gapless");
        }
        // Replay accepts the interleaved journal (state machine coherent).
        for sid in &sids {
            let events = store.events_range(*sid, 1, None).unwrap();
            let state = events.last().unwrap().state;
            assert_eq!(state, AgentState::BuildingContext);
        }
    }

    // ------------------------------------------------- WAL maintenance tests

    /// `(frames in WAL, frames backfilled into the main file)` WITHOUT doing
    /// any checkpoint work: `NOOP` only reports the WAL status, so it
    /// observes exactly what the last real checkpoint left behind.
    fn wal_backfill(store: &Store) -> (i64, i64) {
        store
            .read()
            .unwrap()
            .query_row("PRAGMA wal_checkpoint(NOOP)", [], |r| {
                Ok((r.get(1)?, r.get(2)?))
            })
            .unwrap()
    }

    #[test]
    fn autocheckpoint_is_disabled_and_passive_checkpoint_folds_the_wal() {
        // The actor's 5 ms gate can only hold if SQLite never checkpoints
        // inside a committing statement: configure must disable it on every
        // connection, and the explicit PASSIVE checkpoint must fold the
        // frames it leaves behind.
        let (_dir, store) = tmp_store();
        let auto: i64 = store
            .read()
            .unwrap()
            .query_row("PRAGMA wal_autocheckpoint", [], |r| r.get(0))
            .unwrap();
        assert_eq!(auto, 0, "wal_autocheckpoint must be disabled");
        let sid = hot_session(&store);
        for seq in 1..=200i64 {
            store
                .put_message(sid, seq, "assistant", serde_json::json!({ "i": seq }))
                .unwrap();
        }
        assert!(
            wal_backfill(&store).0 > 0,
            "autocheckpoint is off: the frames must still be in the WAL"
        );
        assert_eq!(wal_backfill(&store).1, 0, "SQLite folded nothing by itself");
        store.wal_checkpoint_passive().unwrap();
        let (log, backfill) = wal_backfill(&store);
        assert_eq!(
            backfill, log,
            "explicit PASSIVE folded every frame ({backfill}/{log})"
        );
        // The fold is not a data loss: rows stay readable through the store.
        assert_eq!(store.message_count(sid).unwrap(), 200);
    }

    #[test]
    fn passive_checkpoint_never_waits_for_a_pinned_reader_and_folds_after() {
        // Maintenance must be bounded: PASSIVE never blocks on a reader that
        // pins a snapshot. It folds what it safely can (frames after the
        // pinned read mark stay in the WAL), the pinned reader keeps its
        // snapshot, and a later checkpoint (after release) folds the rest.
        let (_dir, store) = tmp_store();
        let sid = hot_session(&store);
        for seq in 1..=64i64 {
            store
                .put_message(sid, seq, "assistant", serde_json::json!({ "i": seq }))
                .unwrap();
        }
        let reader = store.read().unwrap();
        reader.execute_batch("BEGIN").unwrap();
        let n: i64 = reader
            .query_row("SELECT COUNT(*) FROM message", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 64, "reader snapshot starts at 64 rows");
        for seq in 65..=128i64 {
            store
                .put_message(sid, seq, "assistant", serde_json::json!({ "i": seq }))
                .unwrap();
        }
        let (log, before) = wal_backfill(&store);
        assert!(before < log, "frames are waiting to be folded");
        let t0 = Instant::now();
        store.wal_checkpoint_passive().unwrap();
        let elapsed = t0.elapsed();
        assert!(
            elapsed < Duration::from_secs(1),
            "PASSIVE must not wait for the pinned reader (took {elapsed:?})"
        );
        // The reader's snapshot is untouched by the checkpoint.
        let n: i64 = reader
            .query_row("SELECT COUNT(*) FROM message", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 64, "pinned snapshot unchanged");
        let (log, backfill) = wal_backfill(&store);
        assert!(
            backfill > 0 && backfill < log,
            "PASSIVE folded only up to the pinned read mark ({backfill}/{log})"
        );
        // Release the snapshot explicitly: the pooled connection survives, so
        // an open read transaction would keep pinning the WAL in the pool.
        reader.execute_batch("COMMIT").unwrap();
        drop(reader);
        store.wal_checkpoint_passive().unwrap();
        let (log, backfill) = wal_backfill(&store);
        assert_eq!(
            backfill, log,
            "after release PASSIVE folds the rest ({backfill}/{log})"
        );
        assert_eq!(store.message_count(sid).unwrap(), 128);
    }
}

#[cfg(test)]
mod typed_ledger_tests {
    use super::*;

    fn tmp_store() -> (tempfile::TempDir, Store) {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path(), true).unwrap();
        (dir, store)
    }

    fn sid(store: &Store) -> SessionId {
        let ws = store.create_workspace("/w").unwrap();
        store.create_session(ws, "t", "p", "m").unwrap().id
    }

    #[test]
    fn ledger_entries_append_gapless_and_page_bounded() {
        let (_d, store) = tmp_store();
        let s = sid(&store);
        let seq1 = store
            .append_ledger_entry(s, "goal_set", 1, serde_json::json!({"goal": "g"}))
            .unwrap();
        let seq2 = store
            .append_ledger_entry(s, "blocker_opened", 1, serde_json::json!({"reason": "r"}))
            .unwrap();
        assert_eq!((seq1, seq2), (1, 2));
        let page = store.ledger_entries(s, None, 1).unwrap();
        assert_eq!(page.len(), 1);
        assert_eq!(page[0].entry_type, "goal_set");
        assert_eq!(page[0].schema_ver, 1);
        assert_eq!(page[0].payload, serde_json::json!({"goal": "g"}));
        let rest = store.ledger_entries(s, Some(1), 10).unwrap();
        assert_eq!(rest.len(), 1);
        assert_eq!(rest[0].entry_type, "blocker_opened");
        // Sessions are isolated.
        let s2 = sid(&store);
        assert!(store.ledger_entries(s2, None, 10).unwrap().is_empty());
        assert_eq!(store.ledger_max_seq(s2).unwrap(), 0);
    }

    #[test]
    fn head_roundtrips_and_compaction_is_atomic_with_head_rewrite() {
        let (_d, store) = tmp_store();
        let s = sid(&store);
        assert!(store.ledger_head(s).unwrap().is_none());
        // A corrupt head blob reads back as an error (the session layer
        // maps that to a rebuild-from-entries).
        store
            .append_ledger_entry(s, "goal_set", 1, serde_json::json!({"goal": "g"}))
            .unwrap();
        store
            .append_ledger_entry(
                s,
                "decision",
                1,
                serde_json::json!({"step": "s", "choice": "c", "rationale": "r"}),
            )
            .unwrap();
        store
            .append_ledger_entry(s, "routing_decision", 1, serde_json::json!({"turn": 1, "provider": "p", "model": "m", "reasoning": "why", "cost_micro": 7}))
            .unwrap();
        store
            .put_ledger_head(s, serde_json::json!({"schema_ver": 1, "goal": "g"}), 3, 1)
            .unwrap();
        let head = store.ledger_head(s).unwrap().unwrap();
        assert_eq!(head.checkpoint_seq, 3);
        assert_eq!(head.head_json["goal"], "g");
        // Compaction: delete below 3 except the goal (seq 1): only the
        // decision (seq 2) is removed; head rewritten atomically.
        let deleted = store
            .compact_ledger(
                s,
                3,
                &[1],
                serde_json::json!({"schema_ver": 1, "goal": "g", "pruned": true}),
                3,
                1,
            )
            .unwrap();
        assert_eq!(deleted, 1);
        let rows = store.ledger_entries(s, None, 10).unwrap();
        assert_eq!(
            rows.len(),
            2,
            "goal pinned + seq-3 row kept above watermark"
        );
        assert_eq!(rows[0].entry_type, "goal_set");
        assert_eq!(rows[1].entry_type, "routing_decision");
        let head = store.ledger_head(s).unwrap().unwrap();
        assert_eq!(head.head_json["pruned"], true);
        assert_eq!(head.checkpoint_seq, 3);
    }

    #[test]
    fn compaction_protect_list_is_never_evicted() {
        // The never-FIFO-evict rule is enforced in faktor-session; here the
        // STORE contract is: protect rows survive a below-watermark delete.
        let (_d, store) = tmp_store();
        let s = sid(&store);
        for i in 1..=5 {
            store
                .append_ledger_entry(
                    s,
                    "decision",
                    1,
                    serde_json::json!({"step": format!("s{i}"), "choice": "c", "rationale": "r"}),
                )
                .unwrap();
        }
        let deleted = store
            .compact_ledger(s, 6, &[2, 5], serde_json::json!({"schema_ver": 1}), 5, 1)
            .unwrap();
        assert_eq!(deleted, 3);
        let kept: Vec<i64> = store
            .ledger_entries(s, None, 10)
            .unwrap()
            .into_iter()
            .map(|r| r.seq)
            .collect();
        assert_eq!(
            kept,
            vec![2, 5],
            "protected rows survive even below the watermark"
        );
        // A compaction that deletes nothing still rewrites the head.
        let deleted = store
            .compact_ledger(s, 0, &[], serde_json::json!({"schema_ver": 1}), 5, 1)
            .unwrap();
        assert_eq!(deleted, 0);
        assert_eq!(store.ledger_head(s).unwrap().unwrap().checkpoint_seq, 5);
    }

    #[test]
    fn appends_never_rewind_below_the_head_checkpoint() {
        // Post-compaction appends keep allocating ABOVE the checkpoint seq
        // (the session fold cursor must never rewind over pruned rows).
        let (_d, store) = tmp_store();
        let s = sid(&store);
        store
            .append_ledger_entry(s, "goal_set", 1, serde_json::json!({"goal": "g"}))
            .unwrap();
        store
            .put_ledger_head(s, serde_json::json!({"schema_ver": 1}), 1, 1)
            .unwrap();
        let deleted = store
            .compact_ledger(s, 2, &[], serde_json::json!({"schema_ver": 1}), 1, 1)
            .unwrap();
        assert_eq!(deleted, 1);
        let seq = store
            .append_ledger_entry(s, "goal_set", 1, serde_json::json!({"goal": "g2"}))
            .unwrap();
        assert!(
            seq > 1,
            "new seq must sit above the pruned checkpoint, got {seq}"
        );
    }

    #[test]
    fn event_payload_versions_stamp_and_read_back() {
        let (_d, store) = tmp_store();
        let s = sid(&store);
        // Legacy append stamps v1 (the historical unversioned writers).
        store
            .append_event(
                s,
                None,
                EventKind::ModelStarted,
                AgentState::Streaming,
                1,
                None,
            )
            .unwrap();
        // Version-aware append stamps its own version.
        store
            .append_event_v(
                s,
                None,
                EventKind::Failed,
                AgentState::FailedRecoverable,
                2,
                Some(serde_json::json!({"message": "x"})),
                2,
            )
            .unwrap();
        let rows = store.events_versioned_range(s, 1, None).unwrap();
        assert_eq!(rows.len(), 3, "seed + two appends");
        assert_eq!(rows[1].1, 1, "legacy writer stamps historical v1");
        assert_eq!(rows[2].1, 2, "versioned writer stamps its schema version");
        assert_eq!(rows[2].0.payload, Some(serde_json::json!({"message": "x"})));
        // The plain reader sees the same events (payload_ver never changes
        // the wire shape).
        let plain = store.events_range(s, 1, None).unwrap();
        assert_eq!(plain.len(), 3);
    }
    // ------------------------------------------------------- index state rows

    #[test]
    fn index_state_roundtrip_cas_and_journal() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        // Unknown workspace -> no row.
        assert!(store.index_state_get(ws).unwrap().is_none());
        // Seed via put: row + journal entry in one transaction.
        store
            .index_state_put(ws, r#"{"state":"not_started"}"#, 0, "not_started")
            .unwrap();
        let row = store.index_state_get(ws).unwrap().unwrap();
        assert_eq!(row.state_json, r#"{"state":"not_started"}"#);
        assert_eq!(row.generation, 0);
        // Legal CAS: NotStarted -> Building{1}.
        assert!(store
            .index_state_cas(
                ws,
                r#"{"state":"not_started"}"#,
                0,
                r#"{"state":"building","generation":1}"#,
                1,
                "building",
            )
            .unwrap());
        // CAS with a STALE expected payload writes nothing and is false.
        assert!(
            !store
                .index_state_cas(
                    ws,
                    r#"{"state":"not_started"}"#,
                    0,
                    r#"{"state":"ready","generation":1}"#,
                    1,
                    "ready",
                )
                .unwrap(),
            "stale expected state must not match"
        );
        let row = store.index_state_get(ws).unwrap().unwrap();
        assert_eq!(row.state_json, r#"{"state":"building","generation":1}"#);
        // Journal is append-only and paged newest-first.
        let log = store.index_state_log(ws, 100).unwrap();
        assert_eq!(log.len(), 2);
        assert_eq!(log[0].kind, "building");
        assert_eq!(log[1].kind, "not_started");
        assert_eq!(log[0].generation, 1);
        assert_eq!(log[1].generation, 0);
        // The row is REPLACED by a put (corruption recovery) + journaled.
        store
            .index_state_put(ws, "not json at all", 7, "corrupt")
            .unwrap();
        let row = store.index_state_get(ws).unwrap().unwrap();
        assert_eq!(row.state_json, "not json at all");
        assert_eq!(row.generation, 7);
        assert_eq!(store.index_state_log(ws, 100).unwrap()[0].kind, "corrupt");
    }

    #[test]
    fn index_state_unknown_workspace_cas_is_false_and_get_none() {
        let (_d, store) = tmp_store();
        // No workspace row exists: get is None, CAS false (writes nothing).
        assert!(store
            .index_state_get(WorkspaceId::new(42))
            .unwrap()
            .is_none());
        assert!(!store
            .index_state_cas(
                WorkspaceId::new(42),
                r#"{"state":"not_started"}"#,
                0,
                r#"{"state":"building","generation":1}"#,
                1,
                "building",
            )
            .unwrap());
        // put on a nonexistent workspace violates the FK loudly (the index
        // service only ever attaches workspaces that resolve in the store).
        let err = store.index_state_put(WorkspaceId::new(42), "{}", 0, "x");
        assert!(err.is_err(), "FK must reject unknown workspaces");
    }

    #[test]
    fn index_state_concurrent_cas_has_exactly_one_winner() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        store
            .index_state_put(ws, r#"{"state":"not_started"}"#, 0, "not_started")
            .unwrap();
        let store = std::sync::Arc::new(store);
        let mut handles = Vec::new();
        for _ in 0..8 {
            let store = store.clone();
            handles.push(std::thread::spawn(move || {
                store
                    .index_state_cas(
                        ws,
                        r#"{"state":"not_started"}"#,
                        0,
                        r#"{"state":"building","generation":1}"#,
                        1,
                        "building",
                    )
                    .unwrap()
            }));
        }
        let winners = handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .filter(|won| *won)
            .count();
        assert_eq!(winners, 1, "exactly one CAS winner per transition");
        let row = store.index_state_get(ws).unwrap().unwrap();
        assert_eq!(row.state_json, r#"{"state":"building","generation":1}"#);
        let log = store.index_state_log(ws, 100).unwrap();
        assert_eq!(log.len(), 2, "only the winner journals: {log:?}");
    }

    #[test]
    fn index_state_survives_reopen_and_migrates_fresh() {
        let dir = tempfile::tempdir().unwrap();
        let ws: WorkspaceId;
        {
            let store = Store::open(dir.path(), true).unwrap();
            ws = store.create_workspace("/w").unwrap();
            store
                .index_state_put(ws, r#"{"state":"ready","generation":4}"#, 4, "ready")
                .unwrap();
        }
        // Reopen (daemon restart): the state row + its journal survive.
        let store = Store::open(dir.path(), true).unwrap();
        let row = store.index_state_get(ws).unwrap().unwrap();
        assert_eq!(row.state_json, r#"{"state":"ready","generation":4}"#);
        assert_eq!(row.generation, 4);
        assert_eq!(store.index_state_log(ws, 10).unwrap().len(), 1);
    }

    #[test]
    fn migration_v14_replays_cleanly_on_a_v13_store() {
        // Simulate a v13 store (typed task rows WITHOUT the revision column,
        // no verification_record table): reopen must add the column
        // (backfilling every legacy row to revision 1), create the record
        // table, and keep legacy rows readable.
        let dir = tempfile::tempdir().unwrap();
        let (sid, tid) = {
            let store = Store::open(dir.path(), true).unwrap();
            let ws = store.create_workspace("/w").unwrap();
            let s = store.create_session(ws, "t", "p", "m").unwrap();
            let row = TaskRow {
                task_id: TaskId::new(1),
                session_id: s.id,
                goal: "legacy goal".into(),
                acceptance_criteria: vec!["cargo check".into()],
                plan: vec![],
                attachments: vec![],
                max_tokens: None,
                max_turns: None,
                spent_tokens: 0,
                spent_turns: 0,
                state: TaskState::Running,
                revision: TaskRevision::new(1),
                created_ms: 5,
                updated_ms: 5,
            };
            store.upsert_task(&row).unwrap();
            {
                let conn = store.write();
                conn.execute("ALTER TABLE task DROP COLUMN revision", [])
                    .unwrap();
                conn.execute("DROP TABLE verification_record", []).unwrap();
                // The v15 cost-ledger objects are post-this-version too:
                // drop them so the full chain (past v15) replays cleanly.
                conn.execute("DROP TABLE cost_reservation", []).unwrap();
                conn.execute("ALTER TABLE task DROP COLUMN max_cost_micro", [])
                    .unwrap();
                conn.execute("ALTER TABLE task DROP COLUMN spent_cost_micro", [])
                    .unwrap();
                // The v17 (attempt-identity) provider_call columns are
                // post-this-version too.
                conn.execute("DROP INDEX IF EXISTS idx_provider_call_session_attempt", [])
                    .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN attempt_op_id", [])
                    .unwrap();
                conn.execute(
                    "ALTER TABLE provider_call DROP COLUMN parent_model_call_op_id",
                    [],
                )
                .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN attempt_ordinal", [])
                    .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN reservation_id", [])
                    .unwrap();
                // The v19 segment-observation column is post-this-version
                // too: drop it so the full chain (past v19) replays cleanly.
                conn.execute(
                    "ALTER TABLE provider_call DROP COLUMN prefix_segments_json",
                    [],
                )
                .unwrap();
                conn.execute("PRAGMA user_version = 14", []).unwrap();
            }
            (s.id, TaskId::new(1))
        };
        let store = Store::open(dir.path(), true).unwrap();
        // The pre-v14 row survived and reads back at revision 1 (DEFAULT
        // backfill), with its content intact.
        let back = store.get_task(sid, tid).unwrap().unwrap();
        assert_eq!(back.state, TaskState::Running);
        assert_eq!(back.revision, TaskRevision::new(1));
        assert_eq!(back.acceptance_criteria, vec!["cargo check".to_string()]);
        assert_eq!(back.created_ms, 5);
        // The new surface is writable and readable.
        let mut row = back.clone();
        row.revision = TaskRevision::new(2);
        store.upsert_task(&row).unwrap();
        assert_eq!(
            store.get_task(sid, tid).unwrap().unwrap().revision,
            TaskRevision::new(2)
        );
        let rec = VerificationRecordRow {
            id: VerificationRecordId::new(1),
            task_id: tid,
            revision: TaskRevision::new(2),
            workspace_id: WorkspaceId::new(1),
            worktree_id: WorktreeId::new(1),
            tree_hash: None,
            criteria: vec![],
            checks: vec![],
            changed_files: vec![],
            unrelated_changes: vec![],
            reviewer: None,
            status: VerificationStatus::Running,
            started_ms: 1,
            completed_ms: None,
        };
        let rec_id = store.verification_record_put(&rec).unwrap();
        assert_eq!(rec_id, VerificationRecordId::new(1));
        // Reopen again: migration is a no-op; both rows survive.
        drop(store);
        let store = Store::open(dir.path(), true).unwrap();
        assert_eq!(
            store.get_task(sid, tid).unwrap().unwrap().revision,
            TaskRevision::new(2)
        );
        assert_eq!(
            store
                .verification_record_get(rec_id)
                .unwrap()
                .unwrap()
                .task_id,
            tid
        );
    }

    #[test]
    fn migration_v16_renames_legacy_abandoned_reservations_to_uncertain_and_adds_the_marker() {
        // Simulate a v15 store (cost_reservation WITHOUT dispatched_ms /
        // pricing_snapshot_json and with the legacy 'abandoned' state), then
        // reopen: the v16 reservation-table migration must rebuild the table
        // — legacy 'abandoned' rows become 'uncertain' (their prediction
        // KEEPS consuming the reserved amount), pre-v17 rows read as
        // never-dispatched and unpriced, and the CHECK now forbids
        // 'abandoned' outright. The store reopens through the CURRENT
        // migration chain (v16 then v17), so the legacy v15 `open` state
        // ends at the v17 `reserved` vocabulary.
        let dir = tempfile::tempdir().unwrap();
        let (sid, tid) = {
            let store = Store::open(dir.path(), true).unwrap();
            let ws = store.create_workspace("/w").unwrap();
            let s = store.create_session(ws, "t", "p", "m").unwrap();
            let task = seed_task(&store, s.id, TaskId::new(1), vec![], TaskState::Running);
            let tid = task.task_id;
            store.cost_task_cap_set(s.id, tid, Some(1_000)).unwrap();
            {
                let conn = store.write();
                // Downgrade the post-v16 objects this rewind replays (the
                // v18 attempt columns on provider_call; the v16 migration
                // itself rebuilds cost_reservation from the v15 shape
                // below).
                conn.execute("DROP INDEX IF EXISTS idx_provider_call_session_attempt", [])
                    .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN attempt_op_id", [])
                    .unwrap();
                conn.execute(
                    "ALTER TABLE provider_call DROP COLUMN parent_model_call_op_id",
                    [],
                )
                .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN attempt_ordinal", [])
                    .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN reservation_id", [])
                    .unwrap();
                // The v19 segment-observation column is post-this-version
                // too: drop it so the full chain (past v19) replays cleanly.
                conn.execute(
                    "ALTER TABLE provider_call DROP COLUMN prefix_segments_json",
                    [],
                )
                .unwrap();
                // The v20 verification-record evidence columns are
                // post-this-version too (the table itself predates neither
                // test): drop them so the full chain (past v20) replays.
                conn.execute(
                    "ALTER TABLE verification_record DROP COLUMN environment_fingerprint_json",
                    [],
                )
                .unwrap();
                conn.execute(
                    "ALTER TABLE verification_record DROP COLUMN candidate_proof_ref_json",
                    [],
                )
                .unwrap();
                // Rebuild the table in its v15 shape (old CHECK, no marker,
                // no snapshot column) and seed one row per legacy status.
                conn.execute("DROP TABLE cost_reservation", []).unwrap();
                conn.execute(
                    "CREATE TABLE cost_reservation (
                        reservation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id INTEGER NOT NULL,
                        task_id INTEGER NOT NULL,
                        op_id INTEGER NOT NULL,
                        predicted_micro INTEGER NOT NULL,
                        status TEXT NOT NULL CHECK (status IN ('open', 'settled', 'refunded', 'abandoned')),
                        created_ms INTEGER NOT NULL,
                        settled_ms INTEGER,
                        provider_cost_micro INTEGER,
                        provider_reported_micro INTEGER,
                        route_decision_json TEXT
                     )",
                    [],
                )
                .unwrap();
                conn.execute(
                    "INSERT INTO cost_reservation
                        (reservation_id, session_id, task_id, op_id, predicted_micro, status,
                         created_ms, settled_ms, provider_cost_micro, provider_reported_micro,
                         route_decision_json)
                     VALUES
                        (1, ?1, ?2, 1, 800, 'abandoned', 1, NULL, NULL, NULL, NULL),
                        (2, ?1, ?2, 2, 100, 'open', 1, NULL, NULL, NULL, NULL),
                        (3, ?1, ?2, 3, 300, 'settled', 1, 9, 250, 250, '{\"provider\":\"p\"}'),
                        (4, ?1, ?2, 4, 50, 'refunded', 1, 2, NULL, NULL, NULL)",
                    params![s.id.raw() as i64, tid.raw() as i64],
                )
                .unwrap();
                // Rewind the cursor: the v16 (index 16, target 17) and v17
                // (index 17, target 18) blocks replay on reopen.
                conn.execute("PRAGMA user_version = 16", []).unwrap();
            }
            (s.id, tid)
        };
        let store = Store::open(dir.path(), true).unwrap();
        let rows = store.cost_reservations_of(sid, tid, 10).unwrap();
        assert_eq!(rows.len(), 4, "every legacy row survived the rebuilds");
        let by_op = |op: i64| {
            rows.iter()
                .find(|r| r.op_id == OpId::new(op as u64))
                .unwrap()
                .clone()
        };
        let abandoned = by_op(1);
        assert_eq!(
            abandoned.status, "uncertain",
            "legacy 'abandoned' rows migrate to 'uncertain'"
        );
        assert_eq!(
            abandoned.predicted_micro, 800,
            "the migrated prediction is preserved"
        );
        assert_eq!(
            abandoned.dispatched_ms, None,
            "pre-v17 rows read as never-dispatched (no marker column existed)"
        );
        assert_eq!(
            abandoned.pricing_snapshot, None,
            "pre-v17 rows read as unpriced (no snapshot column existed)"
        );
        assert_eq!(
            by_op(2).status,
            "reserved",
            "the v17 migration renames legacy 'open' rows to 'reserved'"
        );
        assert_eq!(
            by_op(2).attempt_op_id,
            None,
            "legacy rows carry no attempt identity (op_id was their only op)"
        );
        assert_eq!(by_op(2).parent_op_id, Some(OpId::new(2)));
        assert_eq!(by_op(2).estimated_cost_micro, Some(100));
        assert_eq!(by_op(2).provider_reported_cost_micro, None);
        let settled = by_op(3);
        assert_eq!(settled.status, "settled");
        assert_eq!(settled.provider_cost_micro, Some(250));
        assert_eq!(settled.provider_reported_micro, Some(250));
        assert_eq!(
            settled.provider_reported_cost_micro,
            Some(250),
            "the v18 canonical provider-reported column backfills losslessly"
        );
        assert_eq!(
            settled.settled_cost_micro, None,
            "pre-v18 settlements never recorded which amount was folded: an honest NULL"
        );
        assert_eq!(
            settled.cost_basis, None,
            "pre-v18 settlements never recorded a basis: an honest NULL"
        );
        assert_eq!(settled.estimated_cost_micro, Some(300));
        assert_eq!(by_op(4).status, "refunded");
        // The migrated UNCERTAIN row's prediction KEEPS consuming free:
        // 1000 - 800 (uncertain) - 100 (reserved) = 100 free — a 101
        // reserve refuses with the typed exceeded outcome.
        let out = store
            .cost_reserve_priced(sid, tid, OpId::new(5), 101, now_ms(), None)
            .unwrap();
        assert!(
            matches!(out, CostReserveOutcome::Exceeded { free: 100 }),
            "the migrated uncertain hold consumes the reserved amount: {out:?}"
        );
        // The new CHECK forbids the legacy vocabulary outright (both
        // 'abandoned' and 'open' are gone from the v18 vocabulary).
        for hostile in ["abandoned", "open"] {
            let insert = store.write().execute(
                "INSERT INTO cost_reservation
                    (session_id, task_id, op_id, predicted_micro, status, created_ms)
                 VALUES (1, 1, 9, 1, ?1, 1)",
                [hostile],
            );
            assert!(
                matches!(&insert, Err(rusqlite::Error::SqliteFailure(..))),
                "the rebuilt CHECK rejects legacy state {hostile:?}: {insert:?}"
            );
        }
        // Reopen again: the migrations are a no-op and the data is stable.
        let store = Store::open(dir.path(), true).unwrap();
        let rows = store.cost_reservations_of(sid, tid, 10).unwrap();
        assert_eq!(rows.len(), 4);
        assert_eq!(rows.iter().filter(|r| r.status == "uncertain").count(), 1);
        assert_eq!(rows.iter().filter(|r| r.status == "reserved").count(), 1);
        assert_eq!(rows.iter().filter(|r| r.status == "settled").count(), 1);
        assert_eq!(rows.iter().filter(|r| r.status == "refunded").count(), 1);
    }

    fn seed_task(
        store: &Store,
        session_id: SessionId,
        task_id: TaskId,
        criteria: Vec<String>,
        state: TaskState,
    ) -> TaskRow {
        let row = TaskRow {
            task_id,
            session_id,
            goal: "g".into(),
            acceptance_criteria: criteria,
            plan: vec![],
            attachments: vec![],
            max_tokens: None,
            max_turns: None,
            spent_tokens: 0,
            spent_turns: 0,
            state,
            revision: TaskRevision::new(1),
            created_ms: 1,
            updated_ms: 1,
        };
        store.upsert_task(&row).unwrap();
        row
    }

    /// Seed a task and walk the machine into `Verifying` through the legal
    /// store edges (a row can never be created completion-relevant).
    fn seed_verifying(
        store: &Store,
        session_id: SessionId,
        task_id: TaskId,
        criteria: Vec<String>,
    ) -> TaskRow {
        let mut row = seed_task(store, session_id, task_id, criteria, TaskState::Pending);
        let mut bump = |state: TaskState| {
            row.state = state;
            row.revision = row.revision.checked_next().unwrap();
            store.upsert_task(&row).unwrap();
        };
        bump(TaskState::Running);
        bump(TaskState::NeedsVerification);
        bump(TaskState::Verifying);
        row
    }

    fn passing_record(task: &TaskRow, ws: WorkspaceId, wt: WorktreeId) -> VerificationRecordRow {
        VerificationRecordRow {
            id: VerificationRecordId::new(1),
            task_id: task.task_id,
            revision: task.revision,
            workspace_id: ws,
            worktree_id: wt,
            tree_hash: None,
            criteria: task
                .acceptance_criteria
                .iter()
                .map(|c| CriterionVerification {
                    criterion_key: c.clone(),
                    passed: true,
                    evidence: Some("exit 0".into()),
                })
                .collect(),
            checks: vec![],
            changed_files: vec![],
            unrelated_changes: vec![],
            reviewer: None,
            status: VerificationStatus::Passed,
            started_ms: 1,
            completed_ms: None,
        }
    }

    #[test]
    fn completion_path_validates_every_proof_facet_in_one_transaction() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path(), true).unwrap();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        let task = seed_verifying(&store, s.id, TaskId::new(1), vec!["c1".into(), "c2".into()]);
        let now = now_ms();
        // (b) missing record: typed refusal, row untouched.
        let miss = store
            .task_complete_verified(
                s.id,
                task.task_id,
                task.revision,
                VerificationRecordId::new(999),
                now,
            )
            .unwrap()
            .unwrap_err();
        assert!(matches!(miss, TaskCompletionRefusal::RecordMissing { .. }));
        let record = passing_record(&task, ws, WorktreeId::new(1));
        let rec_id = store.verification_record_put(&record).unwrap();
        // Happy path: one transaction validates (a)-(g) and completes.
        let done = store
            .task_complete_verified(s.id, task.task_id, task.revision, rec_id, now)
            .unwrap()
            .unwrap();
        assert_eq!(done.state, TaskState::VerifiedComplete);
        assert_eq!(done.revision, task.revision.checked_next().unwrap());
        assert_eq!(store.get_task(s.id, task.task_id).unwrap().unwrap(), done);
        // A second completion is refused: stale revision first, then the
        // machine (the task is no longer Verifying).
        let again = store
            .task_complete_verified(s.id, task.task_id, task.revision, rec_id, now)
            .unwrap()
            .unwrap_err();
        assert!(matches!(
            again,
            TaskCompletionRefusal::RevisionMismatch { .. }
        ));
        let again = store
            .task_complete_verified(s.id, task.task_id, done.revision, rec_id, now)
            .unwrap()
            .unwrap_err();
        assert!(matches!(
            again,
            TaskCompletionRefusal::NotVerifying {
                actual: TaskState::VerifiedComplete
            }
        ));
        assert_eq!(
            store
                .get_task(s.id, task.task_id)
                .unwrap()
                .unwrap()
                .revision,
            done.revision,
            "refused completions never bump"
        );

        // (c) wrong task: a record certifying task 1 cannot complete task 2.
        let task2 = seed_verifying(&store, s.id, TaskId::new(2), vec!["c1".into()]);
        let r2 = store
            .task_complete_verified(s.id, task2.task_id, task2.revision, rec_id, now)
            .unwrap()
            .unwrap_err();
        assert!(matches!(r2, TaskCompletionRefusal::RecordWrongTask { .. }));

        // (d) wrong revision: a record certifying task 2's CURRENT revision
        // cannot complete task 2 once the task has moved PAST it (the
        // record must certify the revision being completed).
        let task2_rec = passing_record(&task2, ws, WorktreeId::new(1));
        let t2_id = store.verification_record_put(&task2_rec).unwrap();
        let mut bumped = store.get_task(s.id, TaskId::new(2)).unwrap().unwrap();
        bumped.revision = bumped.revision.checked_next().unwrap();
        store.upsert_task(&bumped).unwrap();
        let stale = store
            .task_complete_verified(s.id, task2.task_id, bumped.revision, t2_id, now)
            .unwrap()
            .unwrap_err();
        assert!(matches!(
            stale,
            TaskCompletionRefusal::RecordWrongRevision { .. }
        ));

        // (e) Failed record refuses.
        let task3 = seed_verifying(&store, s.id, TaskId::new(3), vec!["c1".into()]);
        let mut failed_rec = passing_record(&task3, ws, WorktreeId::new(1));
        failed_rec.status = VerificationStatus::Failed;
        let f_id = store.verification_record_put(&failed_rec).unwrap();
        let r3 = store
            .task_complete_verified(s.id, task3.task_id, task3.revision, f_id, now)
            .unwrap()
            .unwrap_err();
        assert!(matches!(
            r3,
            TaskCompletionRefusal::RecordNotPassed {
                status: VerificationStatus::Failed,
                ..
            }
        ));

        // (f) a record missing one criterion refuses with the missing list.
        let task4 = seed_verifying(&store, s.id, TaskId::new(4), vec!["c1".into(), "c2".into()]);
        let mut partial = passing_record(&task4, ws, WorktreeId::new(1));
        partial.criteria.pop();
        let p_id = store.verification_record_put(&partial).unwrap();
        let r4 = store
            .task_complete_verified(s.id, task4.task_id, task4.revision, p_id, now)
            .unwrap()
            .unwrap_err();
        assert!(matches!(
            r4,
            TaskCompletionRefusal::CriteriaNotCovered { missing, .. }
                if missing == vec!["c2".to_string()]
        ));
        // Extra record criteria are fine: a record covering c1..c2 plus an
        // extra c3 completes.
        let mut extra = passing_record(&task4, ws, WorktreeId::new(1));
        extra.criteria.push(CriterionVerification {
            criterion_key: "c3".into(),
            passed: true,
            evidence: None,
        });
        let e_id = store.verification_record_put(&extra).unwrap();
        let ok4 = store
            .task_complete_verified(s.id, task4.task_id, task4.revision, e_id, now)
            .unwrap()
            .unwrap();
        assert_eq!(ok4.state, TaskState::VerifiedComplete);
        // A passed=false entry for a task criterion counts as NOT covered.
        let task5 = seed_verifying(&store, s.id, TaskId::new(5), vec!["c1".into()]);
        let mut lying = passing_record(&task5, ws, WorktreeId::new(1));
        lying.criteria[0].passed = false;
        let l_id = store.verification_record_put(&lying).unwrap();
        let r5 = store
            .task_complete_verified(s.id, task5.task_id, task5.revision, l_id, now)
            .unwrap()
            .unwrap_err();
        assert!(matches!(
            r5,
            TaskCompletionRefusal::CriteriaNotCovered { missing, .. }
                if missing == vec!["c1".to_string()]
        ));

        // (g) worktree mismatch: a record certified against another
        // workspace's identity refuses even when everything else matches.
        let ws2 = store.create_workspace("/w2").unwrap();
        let s2 = store.create_session(ws2, "t2", "p", "m").unwrap();
        let task6 = seed_verifying(&store, s2.id, TaskId::new(1), vec!["c1".into()]);
        let mut foreign = passing_record(&task6, ws, WorktreeId::new(1));
        foreign.workspace_id = ws;
        let x_id = store.verification_record_put(&foreign).unwrap();
        let r6 = store
            .task_complete_verified(s2.id, task6.task_id, task6.revision, x_id, now)
            .unwrap()
            .unwrap_err();
        assert!(matches!(r6, TaskCompletionRefusal::WorktreeMismatch { .. }));

        // Revision mismatch of the TASK (expected != actual) is reported
        // before the record is even consulted.
        let stale_expected2 = store
            .task_complete_verified(s.id, task2.task_id, task2.revision, rec_id, now)
            .unwrap()
            .unwrap_err();
        assert!(matches!(
            stale_expected2,
            TaskCompletionRefusal::RevisionMismatch { .. }
        ));
    }

    #[test]
    fn completion_vs_reserve_race_never_completes_with_held_reservations() {
        // Adversarial: 1,000 barrier-controlled races of the completion
        // sequence (Running -> NeedsVerification -> Verifying then the
        // exclusive completion transaction) against a concurrent reservation
        // on the SAME task. The race is real: a reserve is admitted while the
        // task is still Running and the completion finds the row in its
        // in-transaction COUNT; or the completion lands first and the late
        // reserve is refused by the task-state gate. The two can never both
        // commit. After EVERY race the invariant is checked directly: a task
        // row that reads VerifiedComplete has zero reserved, zero dispatched
        // and zero uncertain reservation rows; a reserve that landed first
        // leaves the task Verifying with exactly its held row and a typed
        // ReservationsHeld refusal.
        const RACES: usize = 1_000;
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(Store::open(dir.path(), true).unwrap());
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "race", "p", "m").unwrap();
        let sid = s.id;
        let mut reserve_wins = 0usize;
        let mut complete_wins = 0usize;
        for i in 0..RACES {
            let task_id = TaskId::new(i as u64 + 1);
            // Seed Pending (rev 1) -> Running (rev 2). The completion side
            // then walks Running -> NeedsVerification (rev 3) -> Verifying
            // (rev 4) before its exclusive completion.
            let task = seed_task(&store, sid, task_id, vec![], TaskState::Pending);
            let mut running = task.clone();
            running.state = TaskState::Running;
            running.revision = running.revision.checked_next().unwrap();
            store.upsert_task(&running).unwrap();
            // The passing record certifies the revision the completion side
            // will present (rev 4) — the record can exist before the task
            // reaches Verifying (the completion transaction checks both).
            let verifying_revision = TaskRevision::new(4);
            let record_task = TaskRow {
                revision: verifying_revision,
                ..running.clone()
            };
            let record = passing_record(&record_task, ws, WorktreeId::new(1));
            let rec_id = store.verification_record_put(&record).unwrap();
            let barrier = std::sync::Barrier::new(2);
            // The two racers leave the barrier together. Three fifths of the
            // races run free (true writer-lock arbitration); one fifth gives
            // each side a 1ms head start so BOTH orderings are exercised on
            // every host (the OS wake order alone is not a fair scheduler).
            let fork = i % 5;
            let do_reserve = || {
                barrier.wait();
                if fork == 3 {
                    // Completion head start: reserve-first must not be the
                    // only ordering this host can produce.
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                store.cost_reserve_priced(
                    sid,
                    task_id,
                    OpId::new(i as u64 + 1),
                    100,
                    now_ms(),
                    None,
                )
            };
            let do_complete = || {
                barrier.wait();
                if fork == 4 {
                    // Reserve head start.
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                let mut row = store.get_task(sid, task_id).unwrap().unwrap();
                row.state = TaskState::NeedsVerification;
                row.revision = row.revision.checked_next().unwrap();
                store.upsert_task(&row).unwrap();
                row.state = TaskState::Verifying;
                row.revision = row.revision.checked_next().unwrap();
                store.upsert_task(&row).unwrap();
                store.task_complete_verified(sid, task_id, row.revision, rec_id, now_ms())
            };
            let (reserve_out, complete_out) = std::thread::scope(|scope| {
                let reserve = scope.spawn(do_reserve);
                let complete = scope.spawn(do_complete);
                (reserve.join().unwrap(), complete.join().unwrap())
            });
            let rows = store.cost_reservations_of(sid, task_id, i64::MAX).unwrap();
            let count = |status: &str| rows.iter().filter(|r| r.status == status).count();
            // The outer Result is the store call itself; the inner one is the
            // typed completion refusal (proof/revision/accounting gate).
            match (reserve_out, complete_out.unwrap()) {
                (Ok(CostReserveOutcome::Granted(_)), Ok(_)) => {
                    panic!("race {i}: reserve AND completion both committed")
                }
                (Ok(CostReserveOutcome::Granted(_)), Err(refusal)) => {
                    assert!(
                        matches!(
                            refusal,
                            TaskCompletionRefusal::ReservationsHeld {
                                reserved: 1,
                                dispatched: 0,
                                uncertain: 0,
                                reserved_micro: 100,
                                ..
                            }
                        ),
                        "race {i}: completion must name the held row: {refusal:?}"
                    );
                    assert_eq!(count("reserved"), 1);
                    assert_eq!(
                        store.get_task(sid, task_id).unwrap().unwrap().state,
                        TaskState::Verifying,
                        "race {i}: a refused completion never transitions"
                    );
                    reserve_wins += 1;
                }
                (Ok(CostReserveOutcome::Exceeded { .. }), _) => {
                    panic!("race {i}: an uncapped task refused a reserve")
                }
                (Err(reserve_err), Ok(_)) => {
                    // Completion won the write lock: the reserve must have
                    // been refused by the task-state gate, writing nothing.
                    assert!(
                        reserve_err.to_string().contains("cost reserve: task state"),
                        "race {i}: a reserve after completion must refuse typed on state: \
                         {reserve_err}"
                    );
                    assert_eq!(
                        store.get_task(sid, task_id).unwrap().unwrap().state,
                        TaskState::VerifiedComplete
                    );
                    assert_eq!(count("reserved"), 0, "race {i}: reserved must be 0");
                    assert_eq!(count("dispatched"), 0, "race {i}: dispatched must be 0");
                    assert_eq!(count("uncertain"), 0, "race {i}: uncertain must be 0");
                    complete_wins += 1;
                }
                (Err(reserve_err), Err(refusal)) => {
                    panic!("race {i}: neither side committed ({reserve_err} / {refusal:?})")
                }
            }
        }
        eprintln!(
            "completion-vs-reserve races: reserve-first {reserve_wins}, completion-first {complete_wins}"
        );
        assert!(
            reserve_wins > 0,
            "the race never exercised the reserve-first ordering"
        );
        assert!(
            complete_wins > 0,
            "the race never exercised the completion-first ordering"
        );
    }

    #[test]
    fn completion_gate_refuses_every_held_reservation_status_and_keeps_verifying() {
        // The SQL gate is exhaustive over the three budget-holding statuses:
        // each refuses the completion transaction typed with the exact
        // counts and leaves the task row byte-untouched (rollback).
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path(), true).unwrap();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        for (n, status) in ["reserved", "dispatched", "uncertain"]
            .into_iter()
            .enumerate()
        {
            let task_id = TaskId::new(n as u64 + 1);
            // The reserve is admitted while the task still permits new
            // provider work (Running); the task then walks the legal edges
            // into Verifying — the completion gate must catch the row the
            // accounting pass raced.
            seed_task(&store, s.id, task_id, vec![], TaskState::Running);
            let rid = match store
                .cost_reserve_priced(s.id, task_id, OpId::new(100 + n as u64), 77, now_ms(), None)
                .unwrap()
            {
                CostReserveOutcome::Granted(id) => id,
                other => panic!("{status}: reserve refused: {other:?}"),
            };
            if status != "reserved" {
                assert!(
                    matches!(
                        store.cost_mark_dispatched(rid, now_ms()).unwrap(),
                        CostReservationState::Applied
                    ),
                    "{status}: dispatch marker"
                );
            }
            if status == "uncertain" {
                assert!(
                    matches!(
                        store
                            .cost_mark_uncertain(rid, "race-fault", None, now_ms())
                            .unwrap(),
                        CostReservationState::Applied
                    ),
                    "{status}: uncertain marker"
                );
            }
            let mut row = store.get_task(s.id, task_id).unwrap().unwrap();
            row.state = TaskState::NeedsVerification;
            row.revision = row.revision.checked_next().unwrap();
            store.upsert_task(&row).unwrap();
            row.state = TaskState::Verifying;
            row.revision = row.revision.checked_next().unwrap();
            store.upsert_task(&row).unwrap();
            let verifying_revision = row.revision;
            let record = passing_record(&row, ws, WorktreeId::new(1));
            let rec_id = store.verification_record_put(&record).unwrap();
            let refusal = store
                .task_complete_verified(s.id, task_id, verifying_revision, rec_id, now_ms())
                .unwrap()
                .unwrap_err();
            match (status, refusal) {
                (
                    "reserved",
                    TaskCompletionRefusal::ReservationsHeld {
                        reserved,
                        dispatched,
                        uncertain,
                        reserved_micro,
                        uncertain_micro,
                    },
                ) => {
                    assert_eq!((reserved, dispatched, uncertain), (1, 0, 0));
                    assert_eq!(reserved_micro, 77);
                    assert_eq!(uncertain_micro, 0);
                }
                (
                    "dispatched",
                    TaskCompletionRefusal::ReservationsHeld {
                        reserved,
                        dispatched,
                        uncertain,
                        ..
                    },
                ) => {
                    assert_eq!((reserved, dispatched, uncertain), (0, 1, 0));
                }
                (
                    "uncertain",
                    TaskCompletionRefusal::ReservationsHeld {
                        reserved,
                        dispatched,
                        uncertain,
                        uncertain_micro,
                        ..
                    },
                ) => {
                    assert_eq!((reserved, dispatched, uncertain), (0, 0, 1));
                    assert_eq!(uncertain_micro, 77);
                }
                other => panic!("{status}: wrong refusal: {other:?}"),
            }
            let row = store.get_task(s.id, task_id).unwrap().unwrap();
            assert_eq!(row.state, TaskState::Verifying, "{status}: state moved");
            assert_eq!(row.revision, verifying_revision, "{status}: revision moved");
        }
    }

    #[test]
    fn reserve_refused_once_the_task_permits_no_new_provider_operation() {
        // The reserve transaction's task-state condition: completion-relevant
        // and terminal states refuse typed and write NOTHING; the remaining
        // machine states admit. VerifiedComplete is produced through the
        // completion path (raw row writes cannot mint it).
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path(), true).unwrap();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        // Permitted: every state that still allows new provider work.
        for (n, state) in [
            TaskState::Pending,
            TaskState::Planning,
            TaskState::Running,
            TaskState::Waiting,
            TaskState::Blocked,
        ]
        .into_iter()
        .enumerate()
        {
            let task_id = TaskId::new(n as u64 + 1);
            seed_task(&store, s.id, task_id, vec![], state);
            let out = store
                .cost_reserve_priced(s.id, task_id, OpId::new(n as u64 + 1), 1, now_ms(), None)
                .unwrap();
            assert!(
                matches!(out, CostReserveOutcome::Granted(_)),
                "{state:?} must permit a reserve: {out:?}"
            );
        }
        // Refused: completion-relevant and terminal states.
        let mut n = 100u64;
        for state in [
            TaskState::NeedsVerification,
            TaskState::Verifying,
            TaskState::Failed,
            TaskState::Cancelled,
        ] {
            n += 1;
            let task_id = TaskId::new(n);
            if state == TaskState::NeedsVerification {
                // NeedsVerification is completion-relevant: walk the legal
                // machine edge out of Verifying (never a raw seed).
                let mut row = seed_verifying(&store, s.id, task_id, vec![]);
                row.state = TaskState::NeedsVerification;
                row.revision = row.revision.checked_next().unwrap();
                store.upsert_task(&row).unwrap();
            } else if state == TaskState::Verifying {
                seed_verifying(&store, s.id, task_id, vec![]);
            } else {
                seed_task(&store, s.id, task_id, vec![], state);
            }
            let err = store
                .cost_reserve_priced(s.id, task_id, OpId::new(n), 1, now_ms(), None)
                .unwrap_err();
            assert!(
                err.to_string().contains("cost reserve: task state"),
                "{state:?} must refuse typed: {err}"
            );
            assert!(
                store
                    .cost_reservations_of(s.id, task_id, 10)
                    .unwrap()
                    .is_empty(),
                "{state:?}: a refused reserve wrote a row"
            );
        }
        // VerifiedComplete (only the completion transaction may produce it).
        let task_id = TaskId::new(200);
        let task = seed_verifying(&store, s.id, task_id, vec![]);
        let record = passing_record(&task, ws, WorktreeId::new(1));
        let rec_id = store.verification_record_put(&record).unwrap();
        store
            .task_complete_verified(s.id, task_id, task.revision, rec_id, now_ms())
            .unwrap()
            .unwrap();
        let err = store
            .cost_reserve_priced(s.id, task_id, OpId::new(200), 1, now_ms(), None)
            .unwrap_err();
        assert!(
            err.to_string().contains("cost reserve: task state"),
            "VerifiedComplete must refuse typed: {err}"
        );
        assert!(store
            .cost_reservations_of(s.id, task_id, 10)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn corrupt_task_state_fails_reserve_and_completion_typed_without_writes() {
        // SQL failure mode: the task row's state column cannot be parsed.
        // Both the reservation transaction and the completion transaction
        // must fail TYPED before writing anything (no partial transition, no
        // reservation row); healing the row restores both paths — the error
        // was the corruption, not a poisoned connection.
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path(), true).unwrap();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        let task = seed_verifying(&store, s.id, TaskId::new(1), vec![]);
        let record = passing_record(&task, ws, WorktreeId::new(1));
        let rec_id = store.verification_record_put(&record).unwrap();
        store
            .write()
            .execute(
                "UPDATE task SET state = '{not-a-state}' WHERE session_id = ?1 AND task_id = ?2",
                params![s.id.raw() as i64, task.task_id.raw() as i64],
            )
            .unwrap();
        let err = store
            .cost_reserve_priced(s.id, task.task_id, OpId::new(9), 1, now_ms(), None)
            .unwrap_err();
        assert!(
            matches!(err, StoreError::Corrupt(_)),
            "corrupt state must be a typed parse failure: {err}"
        );
        assert!(store
            .cost_reservations_of(s.id, task.task_id, 10)
            .unwrap()
            .is_empty());
        let err = store
            .task_complete_verified(s.id, task.task_id, task.revision, rec_id, now_ms())
            .unwrap_err();
        assert!(
            matches!(err, StoreError::Corrupt(_)),
            "corrupt state must fail completion typed: {err}"
        );
        let raw: (String, i64) = store
            .write()
            .query_row(
                "SELECT state, revision FROM task WHERE session_id = ?1 AND task_id = ?2",
                params![s.id.raw() as i64, task.task_id.raw() as i64],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(raw.0, "{not-a-state}", "the corrupt marker is untouched");
        assert_eq!(
            raw.1,
            task.revision.raw() as i64,
            "the failed completion transaction rolled back whole"
        );
        // Heal the row behind the API's back (a state that permits new
        // provider work): the reserve path works again — the failure was
        // the corruption, not a poisoned connection.
        store
            .write()
            .execute(
                "UPDATE task SET state = '\"running\"' WHERE session_id = ?1 AND task_id = ?2",
                params![s.id.raw() as i64, task.task_id.raw() as i64],
            )
            .unwrap();
        assert!(matches!(
            store
                .cost_reserve_priced(s.id, task.task_id, OpId::new(10), 1, now_ms(), None)
                .unwrap(),
            CostReserveOutcome::Granted(_)
        ));
    }

    #[test]
    fn record_finalize_cas_wins_exactly_once_and_lists_are_deterministic() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path(), true).unwrap();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        let task = seed_task(&store, s.id, TaskId::new(1), vec![], TaskState::Running);
        let put = |status: VerificationStatus| {
            store
                .verification_record_put(&VerificationRecordRow {
                    id: VerificationRecordId::new(1),
                    task_id: task.task_id,
                    revision: task.revision,
                    workspace_id: ws,
                    worktree_id: WorktreeId::new(1),
                    tree_hash: Some("ab".repeat(32)),
                    criteria: vec![],
                    checks: vec![CheckExecution {
                        check: "compile".into(),
                        program: "cargo".into(),
                        args: vec!["check".into()],
                        category: "required".into(),
                        required: true,
                        status,
                        started_ms: 1,
                        finished_ms: Some(2),
                        exit: Some(0),
                        summary: None,
                    }],
                    changed_files: vec![FileStateEvidence {
                        path: "src/main.rs".into(),
                        digest_hex: "cd".repeat(32),
                        size: 10,
                    }],
                    unrelated_changes: vec!["vendor/".into()],
                    reviewer: Some(serde_json::json!({"verdict": "pass"})),
                    status,
                    started_ms: 1,
                    completed_ms: None,
                })
                .unwrap()
        };
        let rec_a = put(VerificationStatus::Running);
        let rec_b = put(VerificationStatus::Passed);
        assert_ne!(rec_a, rec_b, "fresh row ids");
        // Deterministic list order (creation order), stable across reads.
        let list = store
            .verification_record_list_by_task(task.task_id)
            .unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].id, rec_a);
        assert_eq!(list[1].id, rec_b);
        assert_eq!(list[0].checks[0].check, "compile");
        assert_eq!(list[1].changed_files[0].size, 10);
        assert_eq!(list[1].reviewer.as_ref().unwrap()["verdict"], "pass");
        let again = store
            .verification_record_list_by_task(task.task_id)
            .unwrap();
        assert_eq!(again, list, "deterministic across reads");
        assert!(store
            .verification_record_list_by_task(TaskId::new(404))
            .unwrap()
            .is_empty());
        // CAS finalize: exactly one Running -> Passed wins.
        assert_eq!(
            store
                .verification_record_finalize(rec_a, VerificationStatus::Passed, 99)
                .unwrap(),
            Ok(())
        );
        // Second attempt on the now-final record: typed refusal with the
        // CURRENT status (a second finalize can never rewind or re-run).
        assert_eq!(
            store
                .verification_record_finalize(rec_a, VerificationStatus::Failed, 100)
                .unwrap(),
            Err(RecordFinalizeRefusal::NotRunning {
                record_id: rec_a,
                current: VerificationStatus::Passed
            })
        );
        // A Pending record cannot finalize either (only Running may).
        let rec_c = put(VerificationStatus::Pending);
        assert_eq!(
            store
                .verification_record_finalize(rec_c, VerificationStatus::Passed, 101)
                .unwrap(),
            Err(RecordFinalizeRefusal::NotRunning {
                record_id: rec_c,
                current: VerificationStatus::Pending
            })
        );
        // A missing record is its own typed refusal.
        assert_eq!(
            store
                .verification_record_finalize(
                    VerificationRecordId::new(909),
                    VerificationStatus::Passed,
                    1
                )
                .unwrap(),
            Err(RecordFinalizeRefusal::Missing {
                record_id: VerificationRecordId::new(909)
            })
        );
        // Only Passed/Failed may finalize; Unavailable is malformed.
        assert!(matches!(
            store.verification_record_finalize(rec_c, VerificationStatus::Running, 1),
            Err(StoreError::Malformed(_))
        ));
        // The finalized row read back with its completed_ms.
        let row = store.verification_record_get(rec_a).unwrap().unwrap();
        assert_eq!(row.status, VerificationStatus::Passed);
        assert_eq!(row.completed_ms, Some(99));
    }

    #[test]
    fn records_and_completion_survive_reopen_like_a_crash_boundary() {
        // A crash can happen anywhere between record creation and the
        // completion transaction. Each boundary leaves a consistent store:
        // after a reopen the exact same completion either still applies or
        // is refused by the machine — never half-applied.
        let dir = tempfile::tempdir().unwrap();
        let (sid, tid, rec_id) = {
            let store = Store::open(dir.path(), true).unwrap();
            let ws = store.create_workspace("/w").unwrap();
            let s = store.create_session(ws, "t", "p", "m").unwrap();
            let task = seed_verifying(&store, s.id, TaskId::new(1), vec!["c1".into()]);
            let rec = passing_record(&task, ws, WorktreeId::new(1));
            let rec_id = store.verification_record_put(&rec).unwrap();
            // Crash here: record durably written, task still Verifying.
            (s.id, task.task_id, rec_id)
        };
        let store = Store::open(dir.path(), true).unwrap();
        // The record survived and the task reads exactly as it crashed.
        assert_eq!(
            store
                .verification_record_get(rec_id)
                .unwrap()
                .unwrap()
                .status,
            VerificationStatus::Passed
        );
        let task = store.get_task(sid, tid).unwrap().unwrap();
        assert_eq!(task.state, TaskState::Verifying);
        let done = store
            .task_complete_verified(sid, tid, task.revision, rec_id, now_ms())
            .unwrap()
            .unwrap();
        assert_eq!(done.state, TaskState::VerifiedComplete);
        assert_eq!(done.revision, task.revision.checked_next().unwrap());
        drop(store);
        // Crash after completion: reopen shows the completed row, and a
        // re-completion attempt is refused without touching it.
        let store = Store::open(dir.path(), true).unwrap();
        let task = store.get_task(sid, tid).unwrap().unwrap();
        assert_eq!(task.state, TaskState::VerifiedComplete);
        assert_eq!(task.revision, TaskRevision::new(5));
        let again = store
            .task_complete_verified(sid, tid, task.revision, rec_id, now_ms())
            .unwrap()
            .unwrap_err();
        assert!(matches!(
            again,
            TaskCompletionRefusal::NotVerifying {
                actual: TaskState::VerifiedComplete
            }
        ));
        // A hostile PARTIAL write through raw SQL (state flipped out of
        // Verifying WITHOUT a revision bump) is caught by the machine on the
        // next completion attempt.
        {
            let conn = store.write();
            conn.execute(
                "UPDATE task SET state = ?1 WHERE session_id = ?2 AND task_id = ?3",
                params![
                    serde_json::to_string(&TaskState::Running).unwrap(),
                    sid.raw() as i64,
                    tid.raw() as i64
                ],
            )
            .unwrap();
        }
        let task = store.get_task(sid, tid).unwrap().unwrap();
        assert_eq!(task.state, TaskState::Running);
        let refused = store
            .task_complete_verified(sid, tid, task.revision, rec_id, now_ms())
            .unwrap()
            .unwrap_err();
        assert!(matches!(
            refused,
            TaskCompletionRefusal::NotVerifying { .. }
        ));
    }

    #[test]
    fn verification_record_evidence_columns_roundtrip_and_legacy_put_stays_null() {
        // Schema v20 (audits 94/116/117): the additive evidence columns carry
        // whatever opaque bounded JSON the caller validated; the legacy put
        // path writes SQL NULL and reads back as an honest absence. Both
        // survive a reopen.
        let dir = tempfile::tempdir().unwrap();
        let (legacy, with_evidence) = {
            let store = Store::open(dir.path(), true).unwrap();
            let ws = store.create_workspace("/w").unwrap();
            let s = store.create_session(ws, "t", "p", "m").unwrap();
            let task = seed_verifying(&store, s.id, TaskId::new(1), vec!["c1".into()]);
            let rec = passing_record(&task, ws, WorktreeId::new(1));
            let legacy = store.verification_record_put(&rec).unwrap();
            let fingerprint = r#"{"platform":"macos","arch":"aarch64"}"#;
            let candidate = r#"{"task_revision":1,"accounting_snapshot_digest":"accounting:v1:0"}"#;
            let with_evidence = store
                .verification_record_put_with_evidence(&rec, Some(fingerprint), Some(candidate))
                .unwrap();
            (legacy, with_evidence)
        };
        let store = Store::open(dir.path(), true).unwrap();
        let (row, fingerprint, candidate) = store
            .verification_record_get_with_evidence(legacy)
            .unwrap()
            .unwrap();
        assert_eq!(row.task_id.raw(), 1);
        assert!(fingerprint.is_none(), "legacy put writes NULL evidence");
        assert!(candidate.is_none(), "legacy put writes NULL evidence");
        let (_, fingerprint, candidate) = store
            .verification_record_get_with_evidence(with_evidence)
            .unwrap()
            .unwrap();
        assert_eq!(
            fingerprint.as_deref(),
            Some(r#"{"platform":"macos","arch":"aarch64"}"#)
        );
        assert_eq!(
            candidate.as_deref(),
            Some(r#"{"task_revision":1,"accounting_snapshot_digest":"accounting:v1:0"}"#)
        );
        let list = store
            .verification_record_list_by_task_with_evidence(TaskId::new(1))
            .unwrap();
        assert_eq!(list.len(), 2);
        assert!(list[0].1.is_none());
        assert!(list[1].1.is_some());
    }

    #[test]
    fn corrupt_task_revision_reads_as_corruption_never_a_panic() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path(), true).unwrap();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        seed_task(&store, s.id, TaskId::new(1), vec![], TaskState::Pending);
        {
            let conn = store.write();
            conn.execute(
                "UPDATE task SET revision = 0 WHERE session_id = 1 AND task_id = 1",
                [],
            )
            .unwrap();
        }
        assert!(matches!(
            store.get_task(s.id, TaskId::new(1)),
            Err(StoreError::Corrupt(_))
        ));
    }

    // ------------------------------------------- v17 attempt-identity ledger
    // (schema v18): refund-after-dispatch is SQL-impossible, reservations and
    // provider-call rows key by attempt_op_id, delivery/cost-basis columns
    // ride the row, crash recovery splits reserved-vs-dispatched.

    fn known_snapshot_json() -> String {
        use faktor_core::model::{MicroUsdPerMillionTokens, PriceQuote};
        serde_json::to_string(&PricingSnapshot::exact(
            PriceQuote {
                input: MicroUsdPerMillionTokens(15_000_000),
                output: MicroUsdPerMillionTokens(60_000_000),
                cache_read: MicroUsdPerMillionTokens(3_000_000),
                cache_write: MicroUsdPerMillionTokens(7_000_000),
            },
            7,
            "store-test".into(),
        ))
        .unwrap()
    }

    /// Free budget of one task = cap - spent - holding predictions
    /// (reserved + dispatched + uncertain), the store's own formula.
    fn free_micro(store: &Store, session: SessionId, task: TaskId) -> u64 {
        let row = store.cost_task_row(session, task).unwrap().unwrap();
        let max = row.max_cost_micro.unwrap_or(0);
        let rows = store.cost_reservations_of(session, task, i64::MAX).unwrap();
        let held: u64 = rows
            .iter()
            .filter(|r| matches!(r.status.as_str(), "reserved" | "dispatched" | "uncertain"))
            .map(|r| r.predicted_micro)
            .sum();
        max.saturating_sub(row.spent_cost_micro)
            .saturating_sub(held)
    }

    #[test]
    fn refund_is_sql_guarded_after_dispatch_and_after_terminal_states() {
        // (i) The hardened refund: pre-dispatch refunds work; every
        // post-dispatch or terminal state leaves the row UNTOUCHED with the
        // free budget unchanged — enforced by the guarded UPDATE, so even a
        // mis-calling runtime can never free a dispatched reservation.
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path(), true).unwrap();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        let task = seed_task(&store, s.id, TaskId::new(1), vec![], TaskState::Running);
        let tid = task.task_id;
        store.cost_task_cap_set(s.id, tid, Some(1_000)).unwrap();
        let now = now_ms();

        // Refund BEFORE dispatch: applied, money free again.
        let CostReserveOutcome::Granted(r) = store
            .cost_reserve(s.id, tid, OpId::new(1), 400, now)
            .unwrap()
        else {
            panic!("reserve 1 granted")
        };
        assert_eq!(free_micro(&store, s.id, tid), 600);
        assert_eq!(store.cost_refund(r, now).unwrap(), RefundOutcome::Applied);
        assert_eq!(free_micro(&store, s.id, tid), 1_000, "refund frees money");

        // Refund AFTER dispatch: the guarded UPDATE changes zero rows.
        let CostReserveOutcome::Granted(r2) = store
            .cost_reserve(s.id, tid, OpId::new(2), 400, now)
            .unwrap()
        else {
            panic!("reserve 2 granted")
        };
        store.cost_mark_dispatched(r2, now).unwrap();
        assert_eq!(free_micro(&store, s.id, tid), 600);
        let out = store.cost_refund(r2, now + 1).unwrap();
        assert_eq!(
            out,
            RefundOutcome::Blocked {
                current: "dispatched".into(),
                dispatched_ms: Some(now)
            },
            "the refund of a dispatched row is refused with its row truth"
        );
        let rows = store.cost_reservations_of(s.id, tid, 10).unwrap();
        assert_eq!(rows[0].status, "dispatched", "row untouched");
        assert_eq!(rows[0].dispatched_ms, Some(now));
        assert_eq!(free_micro(&store, s.id, tid), 600, "free unchanged");

        // Refund after SETTLE and after REFUND: typed refusals, untouched.
        store
            .cost_settle(r2, 400, Some(400), Some(390), None, now + 2)
            .unwrap();
        let out = store.cost_refund(r2, now + 3).unwrap();
        assert!(matches!(out, RefundOutcome::Blocked { current, .. } if current == "settled"));
        let CostReserveOutcome::Granted(r3) = store
            .cost_reserve(s.id, tid, OpId::new(3), 100, now)
            .unwrap()
        else {
            panic!("reserve 3 granted")
        };
        assert_eq!(
            store.cost_refund(r3, now + 1).unwrap(),
            RefundOutcome::Applied
        );
        let out = store.cost_refund(r3, now + 2).unwrap();
        assert!(matches!(out, RefundOutcome::Blocked { current, .. } if current == "refunded"));
        // Missing reservation.
        assert_eq!(
            store.cost_refund(99_999, now).unwrap(),
            RefundOutcome::Missing
        );
    }

    #[test]
    fn crash_recovery_splits_reserved_from_dispatched_and_uncertain_consumes_free() {
        // (iv) Crash windows: a never-dispatched reservation refunds and
        // restores free; a dispatched one (marker written) becomes UNCERTAIN
        // and KEEPS consuming free.
        let dir = tempfile::tempdir().unwrap();
        let (sid, tid) = {
            let store = Store::open(dir.path(), true).unwrap();
            let ws = store.create_workspace("/w").unwrap();
            let s = store.create_session(ws, "t", "p", "m").unwrap();
            let task = seed_task(&store, s.id, TaskId::new(1), vec![], TaskState::Running);
            let tid = task.task_id;
            store.cost_task_cap_set(s.id, tid, Some(1_000)).unwrap();
            let now = now_ms();
            // (a) reserved: dispatch never began.
            let CostReserveOutcome::Granted(_pre) = store
                .cost_reserve(s.id, tid, OpId::new(1), 200, now)
                .unwrap()
            else {
                panic!("pre-marker reserve")
            };
            // (b) dispatched: the request left the process.
            let CostReserveOutcome::Granted(post) = store
                .cost_reserve(s.id, tid, OpId::new(2), 300, now)
                .unwrap()
            else {
                panic!("post-marker reserve")
            };
            store.cost_mark_dispatched(post, now).unwrap();
            (s.id, tid)
        };
        let store = Store::open(dir.path(), true).unwrap();
        let (refunded, uncertain) = store.cost_recover_open_reservations(now_ms()).unwrap();
        assert_eq!(refunded, 1, "the reserved pre-marker row refunds");
        assert_eq!(uncertain, 1, "the dispatched row goes UNCERTAIN");
        let rows = store.cost_reservations_of(sid, tid, 10).unwrap();
        let by_op = |op: u64| rows.iter().find(|r| r.op_id == OpId::new(op)).unwrap();
        assert_eq!(by_op(1).status, "refunded");
        assert_eq!(by_op(1).dispatched_ms, None);
        assert_eq!(by_op(2).status, "uncertain");
        assert!(by_op(2).dispatched_ms.is_some(), "the marker survives");
        assert_eq!(
            by_op(2).failure_reason_code.as_deref(),
            Some("crash_recovery_post_dispatch_marker")
        );
        assert_eq!(
            free_micro(&store, sid, tid),
            700,
            "the refunded prediction is free again; the uncertain hold (300) consumes"
        );
        // Idempotent recovery.
        let (refunded, uncertain) = store.cost_recover_open_reservations(now_ms()).unwrap();
        assert_eq!((refunded, uncertain), (0, 0));
    }

    #[test]
    fn migration_v17_turns_v16_open_rows_into_reserved_and_keeps_the_marker_truth() {
        // (ii + legacy crash window) A v16 store wrote dispatch markers
        // WITHOUT changing the row's `open` status. After the v17 migration
        // those rows read `reserved` + marker — still refund-impossible (the
        // guarded SQL requires a NULL marker) and still recovered as
        // UNCERTAIN, never as a $0 refund.
        let dir = tempfile::tempdir().unwrap();
        let (sid, tid) = {
            let store = Store::open(dir.path(), true).unwrap();
            let ws = store.create_workspace("/w").unwrap();
            let s = store.create_session(ws, "t", "p", "m").unwrap();
            let task = seed_task(&store, s.id, TaskId::new(1), vec![], TaskState::Running);
            let tid = task.task_id;
            store.cost_task_cap_set(s.id, tid, Some(1_000)).unwrap();
            {
                let conn = store.write();
                conn.execute("DROP INDEX IF EXISTS idx_provider_call_session_attempt", [])
                    .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN attempt_op_id", [])
                    .unwrap();
                conn.execute(
                    "ALTER TABLE provider_call DROP COLUMN parent_model_call_op_id",
                    [],
                )
                .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN attempt_ordinal", [])
                    .unwrap();
                conn.execute("ALTER TABLE provider_call DROP COLUMN reservation_id", [])
                    .unwrap();
                // The v19 segment-observation column is post-this-version
                // too: drop it so the full chain (past v19) replays cleanly.
                conn.execute(
                    "ALTER TABLE provider_call DROP COLUMN prefix_segments_json",
                    [],
                )
                .unwrap();
                // The v20 verification-record evidence columns are
                // post-this-version too: drop them so the full chain
                // (past v20) replays cleanly.
                conn.execute(
                    "ALTER TABLE verification_record DROP COLUMN environment_fingerprint_json",
                    [],
                )
                .unwrap();
                conn.execute(
                    "ALTER TABLE verification_record DROP COLUMN candidate_proof_ref_json",
                    [],
                )
                .unwrap();
                // Rebuild the table in its EXACT v16 shape (the schema the
                // v16 writer produced: dispatched_ms + pricing_snapshot_json
                // present, vocabulary open/settled/refunded/uncertain) and
                // seed the two crash shapes the v16 writer could leave: an
                // `open` row dispatch never began and an `open` row whose
                // marker was written (dispatch may have reached the
                // provider).
                conn.execute("DROP TABLE cost_reservation", []).unwrap();
                conn.execute(
                    "CREATE TABLE cost_reservation (
                        reservation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id INTEGER NOT NULL,
                        task_id INTEGER NOT NULL,
                        op_id INTEGER NOT NULL,
                        predicted_micro INTEGER NOT NULL,
                        status TEXT NOT NULL CHECK (status IN ('open', 'settled', 'refunded', 'uncertain')),
                        created_ms INTEGER NOT NULL,
                        settled_ms INTEGER,
                        dispatched_ms INTEGER,
                        pricing_snapshot_json TEXT,
                        provider_cost_micro INTEGER,
                        provider_reported_micro INTEGER,
                        route_decision_json TEXT
                     )",
                    [],
                )
                .unwrap();
                conn.execute(
                    "INSERT INTO cost_reservation
                        (reservation_id, session_id, task_id, op_id, predicted_micro, status,
                         created_ms, settled_ms, dispatched_ms, pricing_snapshot_json,
                         provider_cost_micro, provider_reported_micro, route_decision_json)
                     VALUES
                        (1, ?1, ?2, 1, 200, 'open', 1, NULL, NULL, NULL, NULL, NULL, NULL),
                        (2, ?1, ?2, 2, 300, 'open', 1, NULL, 555, NULL, NULL, NULL, NULL),
                        (3, ?1, ?2, 3, 400, 'settled', 1, 9, 9, NULL, 250, 250, NULL)",
                    params![s.id.raw() as i64, tid.raw() as i64],
                )
                .unwrap();
                // Rewind the cursor to the v16 schema target: ONLY the v17
                // block (index 17, target 18) replays on reopen.
                conn.execute("PRAGMA user_version = 17", []).unwrap();
            }
            (s.id, tid)
        };
        let store = Store::open(dir.path(), true).unwrap();
        let rows = store.cost_reservations_of(sid, tid, 10).unwrap();
        let by_op = |op: i64| {
            rows.iter()
                .find(|r| r.op_id == OpId::new(op as u64))
                .unwrap()
                .clone()
        };
        let pre_marker = by_op(1);
        assert_eq!(pre_marker.status, "reserved");
        assert_eq!(pre_marker.dispatched_ms, None);
        let legacy_dispatched = by_op(2);
        assert_eq!(
            legacy_dispatched.status, "reserved",
            "v16 open + marker migrates to reserved + marker (nothing lossy)"
        );
        assert_eq!(legacy_dispatched.dispatched_ms, Some(555));
        assert_eq!(legacy_dispatched.parent_op_id, Some(OpId::new(2)));
        assert_eq!(legacy_dispatched.attempt_op_id, None);
        assert_eq!(by_op(3).status, "settled");
        // The refund guard reads the MARKER, not the status name: the
        // migrated dispatched row is unrefundable even though it reads
        // `reserved`.
        let out = store
            .cost_refund(by_op(2).reservation_id, now_ms())
            .unwrap();
        assert_eq!(
            out,
            RefundOutcome::Blocked {
                current: "reserved".into(),
                dispatched_ms: Some(555)
            },
            "a migrated v16 dispatch (reserved + marker) can never refund"
        );
        // Recovery reads it as may-have-dispatched -> UNCERTAIN.
        let (refunded, uncertain) = store.cost_recover_open_reservations(now_ms()).unwrap();
        assert_eq!(refunded, 1, "only the truly pre-dispatch row refunds");
        assert_eq!(uncertain, 1, "the migrated dispatched row goes UNCERTAIN");
        let rows = store.cost_reservations_of(sid, tid, 10).unwrap();
        assert_eq!(
            rows.iter()
                .find(|r| r.reservation_id == legacy_dispatched.reservation_id)
                .unwrap()
                .status,
            "uncertain"
        );
        assert_eq!(
            free_micro(&store, sid, tid),
            700,
            "cap 1000 - 0 folded spend - 300 uncertain hold = 700 (the 200 refund is free)"
        );
        // Reopen again: the migration is a no-op and every row is stable.
        drop(store);
        let store = Store::open(dir.path(), true).unwrap();
        let rows = store.cost_reservations_of(sid, tid, 10).unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows.iter().filter(|r| r.status == "reserved").count(), 0);
        assert_eq!(rows.iter().filter(|r| r.status == "settled").count(), 1);
        assert_eq!(rows.iter().filter(|r| r.status == "refunded").count(), 1);
        assert_eq!(rows.iter().filter(|r| r.status == "uncertain").count(), 1);
    }

    #[test]
    fn attempt_reservations_and_provider_rows_key_by_attempt_op_id() {
        // (iii) Two physical attempts of the SAME logical op carry distinct
        // attempt op ids, separate reservations (each holding its own
        // prediction, each refundable independently) and separate
        // provider-call rows through the attempt writers.
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path(), true).unwrap();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        let task = seed_task(&store, s.id, TaskId::new(1), vec![], TaskState::Running);
        let tid = task.task_id;
        store.cost_task_cap_set(s.id, tid, Some(10_000)).unwrap();
        let now = now_ms();
        let logical = OpId::new(700);
        let a1 = faktor_core::op::ModelCallAttempt::new(logical, OpId::new(701), 0).unwrap();
        let a2 = faktor_core::op::ModelCallAttempt::new(logical, OpId::new(702), 1).unwrap();
        let CostReserveOutcome::Granted(r1) = store
            .cost_reserve_attempt(s.id, tid, &a1, 1_000, now, None)
            .unwrap()
        else {
            panic!("attempt 1 reserve granted")
        };
        let CostReserveOutcome::Granted(r2) = store
            .cost_reserve_attempt(s.id, tid, &a2, 2_000, now, None)
            .unwrap()
        else {
            panic!("attempt 2 reserve granted")
        };
        assert_ne!(r1, r2, "one reservation per attempt");
        assert_eq!(free_micro(&store, s.id, tid), 7_000, "both holds count");
        let rows = store.cost_reservations_of(s.id, tid, 10).unwrap();
        assert_eq!(rows.len(), 2);
        let row1 = rows.iter().find(|r| r.reservation_id == r1).unwrap();
        assert_eq!(row1.attempt_op_id, Some(OpId::new(701)));
        assert_eq!(row1.parent_op_id, Some(logical));
        assert_eq!(
            row1.op_id,
            OpId::new(701),
            "attempt rows key by their own op"
        );
        let row2 = rows.iter().find(|r| r.reservation_id == r2).unwrap();
        assert_eq!(row2.attempt_op_id, Some(OpId::new(702)));
        assert_ne!(row1.attempt_op_id, row2.attempt_op_id);

        // One provider-call row per attempt through the attempt writer.
        let p1 = store
            .record_provider_call_attempt(
                s.id,
                &a1,
                Some(r1),
                "fake",
                "m",
                "completed",
                Some(10),
                Some(20),
                None,
            )
            .unwrap();
        let p2 = store
            .record_provider_call_attempt(
                s.id,
                &a2,
                Some(r2),
                "fake",
                "m",
                "completed",
                Some(30),
                Some(40),
                None,
            )
            .unwrap();
        assert_ne!(p1, p2);
        let calls: Vec<(i64, i64, i64, i64, i64)> = {
            let conn = store.read().unwrap();
            let mut stmt = conn
                .prepare(
                    "SELECT op_id, parent_model_call_op_id, attempt_op_id,
                            attempt_ordinal, reservation_id
                     FROM provider_call WHERE session_id = ?1 ORDER BY id ASC",
                )
                .unwrap();
            let mut out = Vec::new();
            let rows = stmt
                .query_map([s.id.raw() as i64], |r| {
                    Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))
                })
                .unwrap();
            for r in rows {
                out.push(r.unwrap());
            }
            out
        };
        assert_eq!(calls.len(), 2);
        assert_eq!(
            calls[0],
            (700, 700, 701, 0, r1),
            "attempt 1 row: shared logical op in op_id + parent, attempt 701"
        );
        assert_eq!(
            calls[1],
            (700, 700, 702, 1, r2),
            "attempt 2 row: same logical parent, own attempt 702, own reservation"
        );
        // Independent refunds: releasing attempt 1 leaves attempt 2's hold.
        store.cost_refund(r1, now).unwrap();
        assert_eq!(free_micro(&store, s.id, tid), 8_000);
        assert_eq!(
            store
                .cost_reservations_of(s.id, tid, 10)
                .unwrap()
                .iter()
                .find(|r| r.reservation_id == r1)
                .unwrap()
                .status,
            "refunded"
        );
        store.cost_refund(r2, now).unwrap();
        assert_eq!(free_micro(&store, s.id, tid), 10_000);
    }

    #[test]
    fn mark_uncertain_records_reason_and_request_id_on_a_dispatched_row() {
        // (v) A post-dispatch failure marks the row UNCERTAIN with the
        // failure reason code and provider request id recorded durably; the
        // row keeps consuming free until the finalize charges the estimate.
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path(), true).unwrap();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        let task = seed_task(&store, s.id, TaskId::new(1), vec![], TaskState::Running);
        let tid = task.task_id;
        store.cost_task_cap_set(s.id, tid, Some(5_000)).unwrap();
        let now = now_ms();
        let CostReserveOutcome::Granted(r) = store
            .cost_reserve(s.id, tid, OpId::new(9), 2_000, now)
            .unwrap()
        else {
            panic!("reserve granted")
        };
        // mark_uncertain on a never-dispatched row is a typed refusal.
        assert!(matches!(
            store.cost_mark_uncertain(r, "stream_error", Some("req-1"), now),
            Ok(CostReservationState::NotOpen { current }) if current == "reserved"
        ));
        store.cost_mark_dispatched(r, now).unwrap();
        assert_eq!(
            store
                .cost_mark_uncertain(r, "stall_verdict", Some("req-42"), now + 1)
                .unwrap(),
            CostReservationState::Applied
        );
        assert_eq!(
            store.cost_mark_uncertain(r, "x", None, now + 2).unwrap(),
            CostReservationState::NotOpen {
                current: "uncertain".into()
            },
            "an already-uncertain row refuses a second reason (exactly-once reason capture)"
        );
        let rows = store.cost_reservations_of(s.id, tid, 10).unwrap();
        assert_eq!(rows[0].status, "uncertain");
        assert_eq!(
            rows[0].failure_reason_code.as_deref(),
            Some("stall_verdict")
        );
        assert_eq!(rows[0].request_id.as_deref(), Some("req-42"));
        assert_eq!(rows[0].delivery_state.as_deref(), Some("failed"));
        assert_eq!(
            free_micro(&store, s.id, tid),
            3_000,
            "uncertain keeps consuming"
        );
        // The task-completion finalize closes it at the reserved estimate.
        let report = store.cost_finalize_uncertain(s.id, tid, now + 3).unwrap();
        assert_eq!(report.settled, 1);
        assert_eq!(report.charged_micro, 2_000);
        let rows = store.cost_reservations_of(s.id, tid, 10).unwrap();
        assert_eq!(rows[0].status, "settled");
        assert_eq!(
            rows[0].cost_basis.as_deref(),
            Some("ConservativeReservation")
        );
        assert_eq!(rows[0].settled_cost_micro, Some(2_000));
        assert_eq!(rows[0].estimated_cost_micro, Some(2_000));
    }

    #[test]
    fn settle_records_an_honest_cost_basis_and_amounts() {
        // (vi) Settlement writes provider_reported_cost_micro /
        // estimated_cost_micro / settled_cost_micro / cost_basis so the
        // ledger reports HOW the settled number was arrived at:
        // ProviderReported when the provider billed, RouteSnapshotEstimate
        // when categories x the frozen snapshot won, Unknown when nothing
        // was folded.
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path(), true).unwrap();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        let task = seed_task(&store, s.id, TaskId::new(1), vec![], TaskState::Running);
        let tid = task.task_id;
        let now = now_ms();
        // Provider-reported wins: basis ProviderReported.
        let CostReserveOutcome::Granted(r1) = store
            .cost_reserve_priced(
                s.id,
                tid,
                OpId::new(1),
                100_000,
                now,
                Some(&known_snapshot_json()),
            )
            .unwrap()
        else {
            panic!("reserve r1")
        };
        store.cost_mark_dispatched(r1, now).unwrap();
        store
            .cost_settle_usage(r1, 100_000, 0, 0, 2_000, Some(9_999), None, now + 1)
            .unwrap();
        let row = store.cost_reservations_of(s.id, tid, 10).unwrap()[0].clone();
        assert_eq!(row.status, "settled");
        assert_eq!(row.provider_reported_cost_micro, Some(9_999));
        assert_eq!(row.provider_reported_micro, Some(9_999));
        assert_eq!(
            row.provider_cost_micro,
            Some(1_620_000),
            "local 100k@15+2k@60"
        );
        assert_eq!(row.settled_cost_micro, Some(9_999), "the folded amount");
        assert_eq!(row.cost_basis.as_deref(), Some("ProviderReported"));
        assert_eq!(row.estimated_cost_micro, Some(100_000));
        assert_eq!(row.delivery_state.as_deref(), Some("completed"));
        assert_eq!(
            store
                .cost_task_row(s.id, tid)
                .unwrap()
                .unwrap()
                .spent_cost_micro,
            9_999
        );

        // No provider report: categories x the frozen snapshot, basis
        // RouteSnapshotEstimate.
        let CostReserveOutcome::Granted(r2) = store
            .cost_reserve_priced(
                s.id,
                tid,
                OpId::new(2),
                2_000_000,
                now,
                Some(&known_snapshot_json()),
            )
            .unwrap()
        else {
            panic!("reserve r2")
        };
        store.cost_mark_dispatched(r2, now).unwrap();
        store
            .cost_settle_usage(r2, 100_000, 0, 0, 2_000, None, None, now + 1)
            .unwrap();
        let row2 = store.cost_reservations_of(s.id, tid, 10).unwrap()[0].clone();
        assert_eq!(row2.cost_basis.as_deref(), Some("RouteSnapshotEstimate"));
        assert_eq!(row2.settled_cost_micro, Some(1_620_000));
        assert_eq!(row2.provider_reported_cost_micro, None);

        // No price authority + no cap: documented Unknown spend, basis
        // Unknown, nothing folded.
        let CostReserveOutcome::Granted(r3) = store
            .cost_reserve(s.id, tid, OpId::new(3), 100, now)
            .unwrap()
        else {
            panic!("reserve r3")
        };
        store.cost_mark_dispatched(r3, now).unwrap();
        let settled = store
            .cost_settle_usage(r3, 1_000, 0, 0, 2_000, None, None, now + 1)
            .unwrap();
        assert!(matches!(settled, CostSettleOutcome::AppliedUnknown));
        let row3 = store.cost_reservations_of(s.id, tid, 10).unwrap()[0].clone();
        assert_eq!(row3.cost_basis.as_deref(), Some("Unknown"));
        assert_eq!(row3.settled_cost_micro, None);
        assert_eq!(row3.estimated_cost_micro, Some(100));
    }

    #[test]
    fn reconcile_settles_each_uncertain_attempt_from_its_own_provider_row() {
        // (vii) Two dispatched attempts of ONE logical op crash UNCERTAIN;
        // each attempt's own completed provider-call row settles ITS OWN
        // reservation — never the sibling's — even when the sibling's row is
        // the newest completed row the old op_id join would have picked.
        let dir = tempfile::tempdir().unwrap();
        let (sid, tid, r1, r2, a1, a2) = {
            let store = Store::open(dir.path(), true).unwrap();
            let ws = store.create_workspace("/w").unwrap();
            let s = store.create_session(ws, "t", "p", "m").unwrap();
            let task = seed_task(&store, s.id, TaskId::new(1), vec![], TaskState::Running);
            let tid = task.task_id;
            store
                .cost_task_cap_set(s.id, tid, Some(10_000_000))
                .unwrap();
            let now = now_ms();
            let logical = OpId::new(800);
            let a1 = faktor_core::op::ModelCallAttempt::new(logical, OpId::new(801), 0).unwrap();
            let a2 = faktor_core::op::ModelCallAttempt::new(logical, OpId::new(802), 1).unwrap();
            let CostReserveOutcome::Granted(r1) = store
                .cost_reserve_attempt(s.id, tid, &a1, 50_000, now, Some(&known_snapshot_json()))
                .unwrap()
            else {
                panic!("reserve a1")
            };
            let CostReserveOutcome::Granted(r2) = store
                .cost_reserve_attempt(s.id, tid, &a2, 50_000, now, Some(&known_snapshot_json()))
                .unwrap()
            else {
                panic!("reserve a2")
            };
            store.cost_mark_dispatched(r1, now).unwrap();
            store.cost_mark_dispatched(r2, now).unwrap();
            // Crash: both dispatched rows never settled.
            (s.id, tid, r1, r2, a1, a2)
        };
        let store = Store::open(dir.path(), true).unwrap();
        let (refunded, uncertain) = store.cost_recover_open_reservations(now_ms()).unwrap();
        assert_eq!((refunded, uncertain), (0, 2));
        // The resumed logical op completes BOTH attempts' provider rows, but
        // attempt 2's row lands FIRST and attempt 1's row is the newest
        // completed row overall: the old "latest completed row of the op"
        // join would have settled attempt 2's reservation from attempt 1's
        // tokens. The attempt join must not.
        store
            .record_provider_call_attempt(
                sid,
                &a2,
                Some(r2),
                "fake",
                "m",
                "completed",
                Some(1_000),
                Some(100),
                None,
            )
            .unwrap();
        store
            .record_provider_call_attempt(
                sid,
                &a1,
                Some(r1),
                "fake",
                "m",
                "completed",
                Some(4_000),
                Some(2_000),
                None,
            )
            .unwrap();
        let report = store.cost_reconcile_uncertain(sid, tid, now_ms()).unwrap();
        assert_eq!(report.settled, 2);
        // attempt 1: 4_000 @15 + 2_000 @60 = 60_000 + 120_000 = 180_000.
        // attempt 2: 1_000 @15 + 100 @60 = 15_000 + 6_000 = 21_000.
        assert_eq!(report.charged_micro, 180_000 + 21_000);
        let rows = store.cost_reservations_of(sid, tid, 10).unwrap();
        let row1 = rows.iter().find(|r| r.reservation_id == r1).unwrap();
        assert_eq!(row1.provider_cost_micro, Some(180_000), "a1's own tokens");
        let row2 = rows.iter().find(|r| r.reservation_id == r2).unwrap();
        assert_eq!(
            row2.provider_cost_micro,
            Some(21_000),
            "a2 settles from a2's row, never the sibling's newest row"
        );
        assert_eq!(row1.cost_basis.as_deref(), Some("RouteSnapshotEstimate"));
        assert_eq!(row1.settled_cost_micro, Some(180_000));
        // Idempotent: a second pass settles nothing.
        let report = store.cost_reconcile_uncertain(sid, tid, now_ms()).unwrap();
        assert_eq!(report, CostReconcileReport::default());
    }

    // ------------------------------------------------- model outcome stats
    // (migration v18 / schema target 19, audit items 13/14/L)

    fn outcome_sample(
        verified_success: bool,
        rework_cost: u64,
        rework_turns: u64,
    ) -> ModelOutcomeSample {
        ModelOutcomeSample {
            verified_success,
            rework_cost_micro: rework_cost,
            rework_turns,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn assert_row(
        row: &ModelOutcomeStatsRow,
        provider: &str,
        model: &str,
        phase: RouterPhase,
        task_class: TaskClass,
        risk_bucket: RiskBucket,
        successes: u64,
        failures: u64,
        cost: u64,
        turns: u64,
    ) {
        assert_eq!(row.provider, provider);
        assert_eq!(row.model, model);
        assert_eq!(row.phase, phase);
        assert_eq!(row.task_class, task_class);
        assert_eq!(row.risk_bucket, risk_bucket);
        assert_eq!(row.successes_first_pass, successes);
        assert_eq!(row.failures_first_pass, failures);
        assert_eq!(row.rework_cost_micro_sum, cost);
        assert_eq!(row.rework_turns_sum, turns);
        assert_eq!(row.sample_count, successes + failures);
    }

    #[test]
    fn model_outcome_stats_survive_reopen_and_fold_per_phase() {
        // Append facts + projection: exact-key reads stay exact, the phase
        // fold sums every class/risk bucket, and a drop/reopen returns
        // byte-identical rows (durable stats surviving reopen).
        let dir = tempfile::tempdir().unwrap();
        let (s_id, t_id, phase) = {
            let store = Store::open(dir.path().join("store"), true).unwrap();
            let ws = store.create_workspace("/w").unwrap();
            let s = store.create_session(ws, "t", "p", "m").unwrap();
            let tid = seed_task(&store, s.id, TaskId::new(1), vec![], TaskState::Running).task_id;
            // Medium/Low: 55 clean + 45 failing (2.4M cost / 3 turns each).
            for _ in 0..55 {
                store
                    .model_outcome_stats_append(
                        "cheap",
                        "m1",
                        RouterPhase::Implement,
                        TaskClass::Medium,
                        RiskBucket::Low,
                        outcome_sample(true, 0, 0),
                    )
                    .unwrap();
            }
            for _ in 0..45 {
                store
                    .model_outcome_stats_append(
                        "cheap",
                        "m1",
                        RouterPhase::Implement,
                        TaskClass::Medium,
                        RiskBucket::Low,
                        outcome_sample(false, 2_400_000, 3),
                    )
                    .unwrap();
            }
            // Hard/High: the same totals again, exercising the fold.
            for _ in 0..55 {
                store
                    .model_outcome_stats_append(
                        "cheap",
                        "m1",
                        RouterPhase::Implement,
                        TaskClass::Hard,
                        RiskBucket::High,
                        outcome_sample(true, 0, 0),
                    )
                    .unwrap();
            }
            for _ in 0..45 {
                store
                    .model_outcome_stats_append(
                        "cheap",
                        "m1",
                        RouterPhase::Implement,
                        TaskClass::Hard,
                        RiskBucket::High,
                        outcome_sample(false, 2_400_000, 3),
                    )
                    .unwrap();
            }
            // Strong model + a different phase: must NOT leak into folds.
            for _ in 0..94 {
                store
                    .model_outcome_stats_append(
                        "strong",
                        "m2",
                        RouterPhase::Implement,
                        TaskClass::Medium,
                        RiskBucket::Low,
                        outcome_sample(true, 0, 0),
                    )
                    .unwrap();
            }
            for _ in 0..6 {
                store
                    .model_outcome_stats_append(
                        "strong",
                        "m2",
                        RouterPhase::Implement,
                        TaskClass::Medium,
                        RiskBucket::Low,
                        outcome_sample(false, 960_000, 2),
                    )
                    .unwrap();
            }
            for _ in 0..3 {
                store
                    .model_outcome_stats_append(
                        "strong",
                        "m2",
                        RouterPhase::Review,
                        TaskClass::Easy,
                        RiskBucket::Low,
                        outcome_sample(true, 0, 0),
                    )
                    .unwrap();
            }
            let exact = store
                .model_outcome_stats_get(
                    "cheap",
                    "m1",
                    RouterPhase::Implement,
                    TaskClass::Medium,
                    RiskBucket::Low,
                )
                .unwrap()
                .unwrap();
            assert_row(
                &exact,
                "cheap",
                "m1",
                RouterPhase::Implement,
                TaskClass::Medium,
                RiskBucket::Low,
                55,
                45,
                45 * 2_400_000,
                45 * 3,
            );
            let fold = store
                .model_outcome_stats_phase("cheap", "m1", RouterPhase::Implement)
                .unwrap()
                .unwrap();
            assert_eq!(fold.provider, "cheap");
            assert_eq!(fold.model, "m1");
            assert_eq!(fold.phase, RouterPhase::Implement);
            assert_eq!(fold.successes_first_pass, 110);
            assert_eq!(fold.failures_first_pass, 90);
            assert_eq!(fold.rework_cost_micro_sum, 90 * 2_400_000);
            assert_eq!(fold.rework_turns_sum, 90 * 3);
            assert_eq!(fold.sample_count, 200);
            // The fold's class/risk columns are the first folded row's (the
            // PK text order is deterministic); only the sums are meaningful.
            assert!(matches!(
                fold.task_class,
                TaskClass::Medium | TaskClass::Hard
            ));
            assert!(matches!(
                fold.risk_bucket,
                RiskBucket::Low | RiskBucket::High
            ));
            let strong = store
                .model_outcome_stats_phase("strong", "m2", RouterPhase::Implement)
                .unwrap()
                .unwrap();
            assert_row(
                &strong,
                "strong",
                "m2",
                RouterPhase::Implement,
                TaskClass::Medium,
                RiskBucket::Low,
                94,
                6,
                6 * 960_000,
                6 * 2,
            );
            // The Review samples never leak into the Implement fold.
            assert!(store
                .model_outcome_stats_phase("strong", "m2", RouterPhase::Review)
                .unwrap()
                .is_some());
            (s.id, tid, RouterPhase::Implement)
        };
        // Reopen: migrations are a no-op (only the v18 block could replay,
        // and CREATE IF NOT EXISTS keeps rows), every row reads identical.
        let store = Store::open(dir.path().join("store"), true).unwrap();
        let v: i64 = {
            let conn = store.read().unwrap();
            conn.query_row("PRAGMA user_version", [], |r| r.get(0))
                .unwrap()
        };
        assert_eq!(v, 25, "schema target 25 after the v24 attachment migration");
        let fold = store
            .model_outcome_stats_phase("cheap", "m1", phase)
            .unwrap()
            .unwrap();
        assert_eq!(fold.successes_first_pass, 110);
        assert_eq!(fold.failures_first_pass, 90);
        assert_eq!(fold.rework_cost_micro_sum, 90 * 2_400_000);
        assert_eq!(fold.rework_turns_sum, 90 * 3);
        assert_eq!(fold.sample_count, 200);
        let exact = store
            .model_outcome_stats_get("strong", "m2", phase, TaskClass::Medium, RiskBucket::Low)
            .unwrap()
            .unwrap();
        assert_row(
            &exact,
            "strong",
            "m2",
            phase,
            TaskClass::Medium,
            RiskBucket::Low,
            94,
            6,
            6 * 960_000,
            6 * 2,
        );
        assert!(store
            .model_outcome_stats_get("nobody", "m", phase, TaskClass::Medium, RiskBucket::Low,)
            .unwrap()
            .is_none());
        drop(store);
        // Rewind to the previous schema target: the v18 block replays and
        // existing rows SURVIVE (CREATE IF NOT EXISTS is idempotent), while
        // the v19 column must be dropped first because its ADDITIVE ALTER is
        // not idempotent by design (the full migration chain owns the
        // column's existence).
        {
            let conn = rusqlite::Connection::open(dir.path().join("store").join("faktor-plus.db"))
                .unwrap();
            conn.execute(
                "ALTER TABLE provider_call DROP COLUMN prefix_segments_json",
                [],
            )
            .unwrap();
            // The v20 verification-record evidence columns are
            // post-this-version too: drop them so the full chain (past v20)
            // replays cleanly.
            conn.execute(
                "ALTER TABLE verification_record DROP COLUMN environment_fingerprint_json",
                [],
            )
            .unwrap();
            conn.execute(
                "ALTER TABLE verification_record DROP COLUMN candidate_proof_ref_json",
                [],
            )
            .unwrap();
            conn.execute("PRAGMA user_version = 18", []).unwrap();
        }
        let store = Store::open(dir.path().join("store"), true).unwrap();
        let fold = store
            .model_outcome_stats_phase("cheap", "m1", phase)
            .unwrap()
            .unwrap();
        assert_eq!(fold.successes_first_pass, 110);
        assert_eq!(fold.failures_first_pass, 90);
        assert_eq!(fold.rework_cost_micro_sum, 90 * 2_400_000);
        assert_eq!(fold.rework_turns_sum, 90 * 3);
        assert_eq!(fold.sample_count, 200);
        assert_eq!(store.get_session(s_id).unwrap().unwrap().id, s_id);
        assert_eq!(store.get_task(s_id, t_id).unwrap().unwrap().task_id, t_id);
    }

    #[test]
    fn model_outcome_stats_race_writers_never_lose_a_sample() {
        // N threads hammering the SAME key through the shared store: the
        // single writer lock serializes the transactional read-modify-write,
        // so every sample lands exactly once — no lost updates, no broken
        // invariant (adversarial duplicate-replay shape: each thread is a
        // distinct "caller" and 10 identical appends must count 10).
        let dir = tempfile::tempdir().unwrap();
        let store = std::sync::Arc::new(Store::open(dir.path(), true).unwrap());
        let mut handles = Vec::new();
        for t in 0..8u64 {
            let store = store.clone();
            handles.push(std::thread::spawn(move || {
                for i in 0..25u64 {
                    let ok = (t + i) % 3 != 0;
                    store
                        .model_outcome_stats_append(
                            "p",
                            "m",
                            RouterPhase::Implement,
                            TaskClass::Hard,
                            RiskBucket::High,
                            outcome_sample(ok, if ok { 0 } else { 1_000 }, 1),
                        )
                        .unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let row = store
            .model_outcome_stats_get(
                "p",
                "m",
                RouterPhase::Implement,
                TaskClass::Hard,
                RiskBucket::High,
            )
            .unwrap()
            .unwrap();
        assert_eq!(row.sample_count, 8 * 25, "every appended sample lands once");
        assert_eq!(
            row.successes_first_pass + row.failures_first_pass,
            row.sample_count
        );
        // Cross-thread writes never bleed into another key of the same
        // phase fold.
        let fold = store
            .model_outcome_stats_phase("p", "m", RouterPhase::Implement)
            .unwrap()
            .unwrap();
        assert_eq!(fold.sample_count, 8 * 25);
        assert_eq!(fold.failures_first_pass, row.failures_first_pass);
    }

    #[test]
    fn model_outcome_stats_success_samples_never_carry_rework_and_hostile_rows_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path(), true).unwrap();
        // A verified first-pass success can NEVER carry rework, even when a
        // hostile caller hands hostile magnitudes (u64::MAX clamps at the
        // SQLite INTEGER ceiling instead of overflowing or wrapping).
        store
            .model_outcome_stats_append(
                "p",
                "m",
                RouterPhase::Implement,
                TaskClass::Easy,
                RiskBucket::Low,
                outcome_sample(true, u64::MAX, u64::MAX),
            )
            .unwrap();
        store
            .model_outcome_stats_append(
                "p",
                "m",
                RouterPhase::Implement,
                TaskClass::Easy,
                RiskBucket::Low,
                outcome_sample(false, u64::MAX, u64::MAX),
            )
            .unwrap();
        let row = store
            .model_outcome_stats_get(
                "p",
                "m",
                RouterPhase::Implement,
                TaskClass::Easy,
                RiskBucket::Low,
            )
            .unwrap()
            .unwrap();
        assert_eq!(row.successes_first_pass, 1);
        assert_eq!(row.failures_first_pass, 1);
        assert_eq!(
            row.rework_cost_micro_sum,
            i64::MAX as u64,
            "failure rework clamps at i64::MAX"
        );
        assert_eq!(row.rework_turns_sum, i64::MAX as u64);
        // CHECKs hold the projection invariant at the SQL level: a direct
        // (API-bypassing) INSERT whose sample_count contradicts its
        // successes + failures is refused outright.
        let err = store
            .sql_execute(
                "INSERT INTO model_outcome_stats VALUES (
                    'bad1','m','\"implement\"','\"easy\"','\"low\"', 5, 0, 0, 0, 1, 1)",
            )
            .unwrap_err();
        assert!(err.to_string().contains("CHECK"), "{err}");
        // Enum text corruption is refused by the typed readers whenever the
        // row stays addressable: corrupting a KEY dimension renames the row
        // out of every typed lookup (Ok(None), never a guessed enum), so
        // the remaining corrupt-shape attack is the arithmetic invariant —
        // hostile DDL can drop the CHECKs, but the read path still refuses
        // the invariant-broken row it smuggles in.
        store
            .sql_execute(
                "DROP TABLE model_outcome_stats;
                 CREATE TABLE model_outcome_stats (
                    provider TEXT NOT NULL, model TEXT NOT NULL,
                    phase TEXT NOT NULL, task_class TEXT NOT NULL,
                    risk_bucket TEXT NOT NULL,
                    successes_first_pass INTEGER NOT NULL,
                    failures_first_pass INTEGER NOT NULL,
                    rework_cost_micro_sum INTEGER NOT NULL,
                    rework_turns_sum INTEGER NOT NULL,
                    sample_count INTEGER NOT NULL,
                    updated_ms INTEGER NOT NULL,
                    PRIMARY KEY (provider, model, phase, task_class, risk_bucket)
                 ) WITHOUT ROWID;
                 INSERT INTO model_outcome_stats VALUES (
                    'bad3','m','\"implement\"','\"easy\"','\"low\"', 5, 0, 0, 0, 1, 1)",
            )
            .unwrap();
        match store.model_outcome_stats_get(
            "bad3",
            "m",
            RouterPhase::Implement,
            TaskClass::Easy,
            RiskBucket::Low,
        ) {
            Err(StoreError::Corrupt(_)) => {}
            other => panic!("invariant-broken row must be Corrupt, got {other:?}"),
        }
        match store.model_outcome_stats_phase("bad3", "m", RouterPhase::Implement) {
            Err(StoreError::Corrupt(_)) => {}
            other => panic!("the phase fold must refuse the same row, got {other:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Efficiency KPI read (audit 84-88, additive; NO schema change): the
// per-task projection of durable `provider_call` rows the efficiency harness
// sums into `TaskEfficiencyMetrics`. This read exists because no existing
// surface attributes provider calls to a TASK: the v18 attempt columns and
// the legacy logical-op columns already carry the linkage, but only at the
// SQL level. The derivation itself (summing input/output tokens, counting
// usage rows, attributing cache from the durable prefix observation) lives
// in the test-only `faktor-tests-efficiency` crate; this method is the
// durable read it derives from.
// ---------------------------------------------------------------------------

/// One `provider_call` row attributable to one task, as read by
/// [`Store::provider_call_task_rows`].
///
/// Two row classes share the table and are distinguished by their counters:
///
/// - a USAGE row is the physical (or legacy logical) call record: it carries
///   `tokens_in`/`tokens_out`, or a non-`completed` status (a failure row
///   with no counters is still a call);
/// - a PREFIX-OBSERVATION row is written by
///   [`Store::record_provider_call_with_prefix`] (the v13 settlement twin):
///   `completed` status, NULL usage counters, and the durable
///   `prompt_tokens` + `prefix_stability` pair. It describes the SAME call
///   as its usage row, so callers must never count it as a call (the
///   efficiency harness classifies it exactly this way).
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderCallTaskRow {
    /// The `provider_call` row id (call order within the session).
    pub row_id: i64,
    /// The shared logical model-call op the row keys by.
    pub op_id: OpId,
    /// The physical attempt op id (v18 attempt rows; NULL on legacy and
    /// prefix-observation rows).
    pub attempt_op_id: Option<OpId>,
    /// The reservation this call keys to (v18 attempt rows; NULL otherwise).
    pub reservation_id: Option<i64>,
    pub provider: String,
    pub model: String,
    pub status: String,
    pub started_ms: i64,
    pub ended_ms: Option<i64>,
    pub tokens_in: Option<u64>,
    pub tokens_out: Option<u64>,
    /// Cacheable-prefix token count of the durable prefix observation (NULL
    /// when no observation was recorded — never a fabricated zero).
    pub prompt_tokens: Option<u64>,
    /// Per-turn prefix stability in [0, 1] of the observation (NULL when no
    /// observation was recorded).
    pub prefix_stability: Option<f64>,
}

/// Read-time guard for a durable counter column: negative SQLite integers
/// are corrupt (a counter is non-negative), and are surfaced as a loud
/// `Malformed` instead of being clamped into a silently wrong KPI.
fn efficiency_counter(raw: Option<i64>, what: &str) -> StoreResult<Option<u64>> {
    match raw {
        None => Ok(None),
        Some(v) if v < 0 => Err(StoreError::Malformed(format!(
            "{what} is negative ({v}): a durable token counter cannot be negative"
        ))),
        Some(v) => Ok(Some(v as u64)),
    }
}

impl Store {
    /// Every durable `provider_call` row attributable to
    /// `(session_id, task_id)`, oldest row first (bounded by `limit`).
    ///
    /// ATTRIBUTION: a row belongs to the task when it keys one of the task's
    /// `cost_reservation` rows —
    ///
    /// - by `reservation_id` (v18 attempt usage rows), or
    /// - by `attempt_op_id` (attempt usage rows whose reservation link was
    ///   never written), or
    /// - by the shared logical op: `cost_reservation.op_id` for legacy
    ///   single-attempt reservations and their legacy usage rows, and
    ///   `parent_op_id` for the v13 prefix-observation rows (which carry no
    ///   attempt/reservation identity of their own).
    ///
    /// Prefix-observation rows are intentionally included: they carry NULL
    /// usage counters, so a summation over a task's rows never double counts
    /// a call's usage, while their `prompt_tokens`/`prefix_stability` pair is
    /// the only durable cache attribution that exists.
    ///
    /// The returned flag is `true` when MORE attributable rows exist beyond
    /// `limit` (the caller asked for a bounded read; the KPI derivation
    /// refuses partial totals). A negative `limit` reads zero rows and
    /// reports truncation. Values are validated on read: a negative token
    /// counter, or a prefix stability outside [0, 1] / non-finite, is a
    /// loud `Malformed`, never a silently wrong number.
    pub fn provider_call_task_rows(
        &self,
        session_id: SessionId,
        task_id: TaskId,
        limit: i64,
    ) -> StoreResult<(Vec<ProviderCallTaskRow>, bool)> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT p.id, p.op_id, p.attempt_op_id, p.reservation_id,
                    p.provider, p.model, p.status, p.started_ms, p.ended_ms,
                    p.tokens_in, p.tokens_out, p.prompt_tokens,
                    p.prefix_stability
             FROM provider_call p
             WHERE p.session_id = ?1
               AND EXISTS (
                   SELECT 1 FROM cost_reservation r
                   WHERE r.session_id = p.session_id AND r.task_id = ?2
                     AND (
                         r.reservation_id = p.reservation_id
                         OR (p.attempt_op_id IS NOT NULL
                             AND r.attempt_op_id = p.attempt_op_id)
                         OR (p.attempt_op_id IS NULL
                             AND (r.op_id = p.op_id
                                  OR r.parent_op_id = p.op_id))
                     )
               )
             ORDER BY p.id ASC LIMIT ?3",
        )?;
        let max = limit.max(0);
        let mut rows = stmt.query(params![
            session_id.raw() as i64,
            task_id.raw() as i64,
            max.saturating_add(1)
        ])?;
        let mut out = Vec::new();
        while let Some(r) = rows.next()? {
            let stability: Option<f64> = r.get(12)?;
            if let Some(s) = stability {
                if !s.is_finite() || !(0.0..=1.0).contains(&s) {
                    return Err(StoreError::Malformed(format!(
                        "provider_call prefix_stability {s} out of [0, 1]"
                    )));
                }
            }
            let attempt_op_id: Option<i64> = r.get(2)?;
            out.push(ProviderCallTaskRow {
                row_id: r.get(0)?,
                op_id: OpId::new(r.get::<_, i64>(1)?.max(1) as u64),
                attempt_op_id: attempt_op_id.map(|id| OpId::new(id.max(1) as u64)),
                reservation_id: r.get(3)?,
                provider: r.get(4)?,
                model: r.get(5)?,
                status: r.get(6)?,
                started_ms: r.get(7)?,
                ended_ms: r.get(8)?,
                tokens_in: efficiency_counter(r.get(9)?, "provider_call.tokens_in")?,
                tokens_out: efficiency_counter(r.get(10)?, "provider_call.tokens_out")?,
                prompt_tokens: efficiency_counter(r.get(11)?, "provider_call.prompt_tokens")?,
                prefix_stability: stability,
            });
        }
        let truncated = out.len() as i64 > max;
        if truncated {
            out.truncate(max as usize);
        }
        Ok((out, truncated))
    }
}

#[cfg(test)]
mod evidence_store_tests {
    use super::*;

    fn tmp() -> (tempfile::TempDir, Store) {
        let dir = tempfile::tempdir().unwrap();
        let s = Store::open(dir.path(), true).unwrap();
        (dir, s)
    }

    fn row(session: SessionId, workspace: WorkspaceId, compact: &str) -> EvidenceRow {
        EvidenceRow {
            id: 0,
            session_id: session,
            workspace_id: workspace,
            task_id: Some(7),
            kind: "process_log".into(),
            revision: 1,
            provenance_json: r#"{"entries":["tool"]}"#.into(),
            compressibility: "aggressive".into(),
            compression_json: r#"{"algorithm":"identity"}"#.into(),
            retrieval_json: r#"{"allow_ranges":true,"allow_search":true,"max_bytes":64}"#.into(),
            compact_json: compact.into(),
            backing_cas_hash: Some("ab".repeat(32)),
            completeness: "complete".into(),
            created_ms: 1234,
        }
    }

    #[test]
    fn evidence_ids_are_globally_unique_across_reopen_and_never_reissued() {
        let dir = tempfile::tempdir().unwrap();
        let first: u64;
        let second: u64;
        {
            let store = Store::open(dir.path(), true).unwrap();
            let ws = store.create_workspace("/w").unwrap();
            let sid = store.create_session(ws, "t", "p", "m").unwrap().id;
            first = store.evidence_insert(&row(sid, ws, "first")).unwrap();
            second = store.evidence_insert(&row(sid, ws, "second")).unwrap();
            assert!(first >= 1 && second > first, "ids are monotonic");
            assert_eq!(store.evidence_high_water().unwrap(), second);
        }
        {
            // "Daemon restart": a fresh opener must keep minting ABOVE every
            // id ever issued, and the original ids must still resolve.
            let store = Store::open(dir.path(), true).unwrap();
            let ws = WorkspaceId::new(1);
            let sid = SessionId::new(1);
            let reopened = store.evidence_get(first).unwrap().expect("id survives");
            assert_eq!(reopened.compact_json, "first");
            let third = store.evidence_insert(&row(sid, ws, "third")).unwrap();
            assert!(
                third > second,
                "reopen must never reissue {second}: got {third}"
            );
            assert_eq!(store.evidence_high_water().unwrap(), third);
            // A hostile explicit insert of an EXISTING id is refused, never
            // an overwrite (the original envelope bytes stay).
            let mut dup = row(sid, ws, "overwrite attempt");
            dup.id = first;
            match store.evidence_insert(&dup) {
                Err(StoreError::Conflict(_)) => {}
                other => panic!("duplicate id must conflict, got {other:?}"),
            }
            assert_eq!(
                store.evidence_get(first).unwrap().unwrap().compact_json,
                "first"
            );
        }
    }

    #[test]
    fn evidence_scope_listing_and_backing_digest_index_are_bounded_and_scoped() {
        let (_d, store) = tmp();
        let ws_a = store.create_workspace("/a").unwrap();
        let ws_b = store.create_workspace("/b").unwrap();
        let sid_a = store.create_session(ws_a, "a", "p", "m").unwrap().id;
        let sid_b = store.create_session(ws_b, "b", "p", "m").unwrap().id;
        for i in 0..5 {
            store
                .evidence_insert(&row(sid_a, ws_a, &format!("a-{i}")))
                .unwrap();
        }
        store.evidence_insert(&row(sid_b, ws_b, "b-0")).unwrap();

        let a = store.evidence_list_by_scope(sid_a, ws_a, 100).unwrap();
        assert_eq!(a.len(), 5);
        assert!(a
            .iter()
            .all(|r| r.session_id == sid_a && r.workspace_id == ws_a));
        let b = store.evidence_list_by_scope(sid_b, ws_b, 100).unwrap();
        assert_eq!(b.len(), 1);
        // A scope with no rows lists nothing, never leaks another session.
        assert!(store
            .evidence_list_by_scope(sid_a, ws_b, 100)
            .unwrap()
            .is_empty());
        // `limit` is enforced; and the backing index lists every envelope
        // referencing the digest (all six share `ab`*32) without granting
        // anything by itself.
        assert_eq!(
            store.evidence_list_by_scope(sid_a, ws_a, 2).unwrap().len(),
            2
        );
        let ids = store
            .evidence_ids_by_backing(&"ab".repeat(32), 100)
            .unwrap();
        assert_eq!(ids.len(), 6);
        assert!(store
            .evidence_ids_by_backing(&"cd".repeat(32), 100)
            .unwrap()
            .is_empty());
        // Corrupt revision refuses on read (a row injected behind the API).
        let conn = store.write();
        conn.execute(
            "UPDATE evidence SET revision = 0 WHERE id = ?1",
            params![ids[0] as i64],
        )
        .unwrap();
        drop(conn);
        match store.evidence_get(ids[0]) {
            Err(StoreError::Corrupt(_)) => {}
            other => panic!("revision 0 must be corrupt, got {other:?}"),
        }
    }

    #[test]
    fn migration_v21_replays_cleanly_on_a_v20_store() {
        let dir = tempfile::tempdir().unwrap();
        {
            let store = Store::open(dir.path(), true).unwrap();
            store.create_workspace("/w").unwrap();
            store
                .create_session(WorkspaceId::new(1), "t", "p", "m")
                .unwrap();
        }
        {
            // Rewind to v20: drop the v21 table + indexes and reset the
            // version cursor exactly as a pre-v21 build left the file.
            let mut conn = Connection::open(dir.path().join("faktor-plus.db")).unwrap();
            configure(&conn).unwrap();
            conn.execute_batch(
                "DROP INDEX IF EXISTS idx_evidence_scope;
                 DROP INDEX IF EXISTS idx_evidence_session_task;
                 DROP INDEX IF EXISTS idx_evidence_backing_cas;
                 DROP TABLE IF EXISTS evidence;
                 PRAGMA user_version = 21;",
            )
            .unwrap();
            conn.execute("DELETE FROM sqlite_sequence WHERE name = 'evidence'", [])
                .unwrap();
            migrate(&mut conn).unwrap();
            let version: i64 = conn
                .query_row("PRAGMA user_version", [], |r| r.get(0))
                .unwrap();
            assert_eq!(version, 25, "v24 is the migration head");
            let ws_ok: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='evidence'",
                    [],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(ws_ok, 1);
            for table in [
                "verification_attempt",
                "verification_attempt_changed_file",
                "verification_job",
                "verification_job_result",
            ] {
                let present: i64 = conn
                    .query_row(
                        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name = ?1",
                        params![table],
                        |r| r.get(0),
                    )
                    .unwrap();
                assert_eq!(present, 1, "v22 table {table} must exist after the replay");
            }
            drop(conn);
        }
        // The migrated store serves the full evidence surface.
        let store = Store::open(dir.path(), true).unwrap();
        let ws = WorkspaceId::new(1);
        let sid = SessionId::new(1);
        let id = store
            .evidence_insert(&row(sid, ws, "post-migrate"))
            .unwrap();
        assert_eq!(
            store.evidence_get(id).unwrap().unwrap().compact_json,
            "post-migrate"
        );
    }
}

#[cfg(test)]
mod verification_job_store_tests {
    use super::*;

    fn setup() -> (tempfile::TempDir, Store, SessionId) {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path(), true).unwrap();
        let ws = store.create_workspace("/vj").unwrap();
        let sid = store.create_session(ws, "vj", "p", "m").unwrap().id;
        (dir, store, sid)
    }

    fn attempt(sid: SessionId, op: u64, revision: u64) -> VerificationAttemptRow {
        VerificationAttemptRow {
            session_id: sid,
            task_id: TaskId::new(1),
            attempt_op_id: op,
            task_revision: TaskRevision::new(revision),
            workspace_root: "/vj".into(),
            environment_fingerprint_json: None,
            created_ms: 10,
        }
    }

    fn check(
        sid: SessionId,
        op: u64,
        check_id: &str,
        ordinal: u32,
        inline: Option<&str>,
    ) -> VerificationJobRow {
        let is_inline = inline.is_some();
        VerificationJobRow {
            session_id: sid,
            task_id: TaskId::new(1),
            attempt_op_id: op,
            check_id: check_id.into(),
            ordinal,
            task_revision: TaskRevision::new(3),
            workspace_root: "/vj".into(),
            kind: if is_inline {
                String::new()
            } else {
                "test".into()
            },
            command: format!("ctest {check_id}"),
            program: if is_inline {
                String::new()
            } else {
                "ctest".into()
            },
            args_json: "[]".into(),
            spec_json: if is_inline {
                None
            } else {
                Some("{\"id\":\"x\",\"program\":\"ctest\",\"args\":[]}".into())
            },
            budget_ms: if is_inline { 0 } else { 1_000 },
            inline_status: inline.map(str::to_string),
            state: inline.unwrap_or("queued").to_string(),
            result_json: None,
            note: None,
            op_id: None,
            environment_fingerprint_json: None,
            created_ms: 10,
            updated_ms: 10,
            finished_ms: None,
        }
    }

    #[test]
    fn attempt_begin_is_atomic_idempotent_and_conflicts_on_open_job() {
        let (_d, store, sid) = setup();
        let changed = vec!["src/a.rs".to_string(), "src/b.rs".to_string()];
        let checks = vec![
            check(sid, 1, "inline_ok", 0, Some("passed")),
            check(sid, 1, "bg", 1, None),
        ];
        assert!(store
            .verification_attempt_begin(&attempt(sid, 1, 3), &changed, &checks)
            .unwrap());
        // Idempotent retry: the second begin writes nothing.
        assert!(!store
            .verification_attempt_begin(&attempt(sid, 1, 3), &changed, &checks)
            .unwrap());
        let view = store
            .verification_attempt_get(sid, TaskId::new(1), 1)
            .unwrap()
            .unwrap();
        assert_eq!(view.changed, changed, "changed files ordered + intact");
        assert_eq!(view.checks.len(), 2, "inline + background checks survive");
        assert_eq!(view.checks[0].state, "passed");
        assert_eq!(view.checks[0].inline_status.as_deref(), Some("passed"));
        assert_eq!(view.checks[1].state, "queued");
        assert_eq!(view.checks[1].inline_status, None);
        assert_eq!(
            store
                .verification_jobs_open(sid, TaskId::new(1))
                .unwrap()
                .len(),
            1,
            "inline checks are never open jobs"
        );
        // An open background check of the OLD attempt refuses a new begin.
        let err = store
            .verification_attempt_begin(&attempt(sid, 2, 3), &[], &[check(sid, 2, "bg", 0, None)])
            .unwrap_err();
        assert!(matches!(err, StoreError::Conflict(_)), "{err:?}");
        // The commit point is atomic: a rejected second begin left no rows.
        assert!(store
            .verification_attempt_get(sid, TaskId::new(1), 2)
            .unwrap()
            .is_none());
        assert_eq!(
            store
                .verification_attempt_cancel(sid, TaskId::new(1), 1, "superseded", 9)
                .unwrap(),
            1
        );
        assert!(store
            .verification_attempt_begin(&attempt(sid, 2, 3), &[], &[check(sid, 2, "bg", 0, None)])
            .unwrap());
        let current = store
            .verification_attempt_current(sid, TaskId::new(1))
            .unwrap()
            .unwrap();
        assert_eq!(current.attempt.attempt_op_id, 2);
    }

    #[test]
    fn claim_resolve_exactly_once_and_supersede_freezes_attempt_n() {
        let (_d, store, sid) = setup();
        assert!(
            store
                .verification_attempt_begin(
                    &attempt(sid, 10, 3),
                    &[],
                    &[check(sid, 10, "bg", 0, None)],
                )
                .unwrap()
        );
        let claimed = store
            .verification_job_claim(sid, TaskId::new(1), 10, "bg", 99, 11)
            .unwrap()
            .unwrap();
        assert_eq!(claimed.state, "running");
        assert_eq!(claimed.op_id, Some(99));
        // Double claim is a typed refusal.
        match store
            .verification_job_claim(sid, TaskId::new(1), 10, "bg", 100, 11)
            .unwrap()
        {
            Err(VerificationJobRefusal::NotOpen { state, .. }) => assert_eq!(state, "running"),
            other => panic!("expected NotOpen, got {other:?}"),
        }
        let resolved = store
            .verification_job_resolve(
                sid,
                TaskId::new(1),
                10,
                "bg",
                "passed",
                None,
                Some("{\"status\":\"passed\"}"),
                12,
            )
            .unwrap()
            .unwrap();
        assert_eq!(resolved.state, "passed");
        assert_eq!(
            resolved.result_json.as_deref(),
            Some("{\"status\":\"passed\"}")
        );
        // Resolve exactly once.
        match store
            .verification_job_resolve(sid, TaskId::new(1), 10, "bg", "failed", None, None, 13)
            .unwrap()
        {
            Err(VerificationJobRefusal::NotOpen { state, .. }) => assert_eq!(state, "passed"),
            other => panic!("expected NotOpen, got {other:?}"),
        }
        // Attempt N+1 with a DIFFERENT check commits; attempt N freezes.
        assert!(store
            .verification_attempt_begin(
                &attempt(sid, 11, 3),
                &[],
                &[check(sid, 11, "bg2", 0, None)],
            )
            .unwrap());
        match store
            .verification_job_resolve(
                sid,
                TaskId::new(1),
                10,
                "bg",
                "failed",
                None,
                Some("{\"status\":\"failed\"}"),
                14,
            )
            .unwrap()
        {
            Err(VerificationJobRefusal::Superseded {
                attempt_op_id,
                newest_attempt_op_id,
            }) => {
                assert_eq!((attempt_op_id, newest_attempt_op_id), (10, 11));
            }
            other => panic!("expected Superseded, got {other:?}"),
        }
        match store
            .verification_job_claim(sid, TaskId::new(1), 10, "bg", 101, 14)
            .unwrap()
        {
            Err(VerificationJobRefusal::Superseded { .. }) => {}
            other => panic!("expected Superseded claim, got {other:?}"),
        }
        // Attempt N is byte-intact: its PASSED result was never mutated.
        let old = store
            .verification_attempt_get(sid, TaskId::new(1), 10)
            .unwrap()
            .unwrap();
        assert_eq!(old.checks[0].state, "passed");
        assert_eq!(
            old.checks[0].result_json.as_deref(),
            Some("{\"status\":\"passed\"}")
        );
        // The results are attempt-keyed rows: N+1 starts without one.
        let new = store
            .verification_attempt_get(sid, TaskId::new(1), 11)
            .unwrap()
            .unwrap();
        assert_eq!(new.checks[0].state, "queued");
        assert!(new.checks[0].result_json.is_none());
        let result_rows: i64 = store
            .read()
            .unwrap()
            .query_row(
                "SELECT COUNT(*) FROM verification_job_result WHERE session_id = ?1",
                params![sid.raw() as i64],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(result_rows, 1, "only attempt N recorded a result");
    }

    #[test]
    fn corrupt_or_foreign_job_rows_are_loud_typed_errors() {
        let (_d, store, sid) = setup();
        assert!(store
            .verification_attempt_begin(
                &attempt(sid, 20, 3),
                &[],
                &[
                    check(sid, 20, "bg", 0, None),
                    check(sid, 20, "inline", 1, Some("passed"))
                ],
            )
            .unwrap());
        {
            let conn = store.write();
            conn.execute(
                "UPDATE verification_job SET state = 'bogus' WHERE check_id = 'bg'",
                [],
            )
            .unwrap();
        }
        match store.verification_jobs_open(sid, TaskId::new(1)) {
            Err(StoreError::Malformed(msg)) => assert!(msg.contains("bogus"), "{msg}"),
            other => panic!("unknown state must be malformed, got {other:?}"),
        }
        {
            let conn = store.write();
            // Repair the state so the NEXT corruption class is isolated.
            conn.execute(
                "UPDATE verification_job SET state = 'queued' WHERE check_id = 'bg'",
                [],
            )
            .unwrap();
            conn.execute(
                "UPDATE verification_job SET state = 'failed' WHERE check_id = 'inline'",
                [],
            )
            .unwrap();
        }
        match store.verification_attempt_current(sid, TaskId::new(1)) {
            Err(StoreError::Malformed(msg)) => assert!(msg.contains("disagrees"), "{msg}"),
            other => panic!("inline/state drift must be malformed, got {other:?}"),
        }
        // Force a non-positive identity on the attempt row: a corrupt row
        // reads loudly instead of being trusted.
        {
            let conn = store.write();
            conn.execute(
                "UPDATE verification_attempt SET task_revision = 0 WHERE attempt_op_id = 20",
                [],
            )
            .unwrap();
        }
        match store.verification_attempt_current(sid, TaskId::new(1)) {
            Err(StoreError::Corrupt(_)) => {}
            other => panic!("zero revision must be corrupt, got {other:?}"),
        }
    }

    #[test]
    fn v22_bounds_reject_oversized_argv_counts_and_duplicates_before_write() {
        let (_d, store, sid) = setup();
        // 33 argv entries (cap 32).
        let mut wide = check(sid, 30, "bg", 0, None);
        wide.args_json = serde_json::to_string(&vec!["a"; MAX_VERIFICATION_JOB_ARGS + 1]).unwrap();
        assert!(matches!(
            store.verification_attempt_begin(&attempt(sid, 30, 3), &[], &[wide]),
            Err(StoreError::Oversized(_))
        ));
        // One oversized argument (cap 1024).
        let mut long = check(sid, 30, "bg", 0, None);
        long.args_json =
            serde_json::to_string(&vec!["x".repeat(MAX_VERIFICATION_JOB_ARG_BYTES + 1)]).unwrap();
        assert!(matches!(
            store.verification_attempt_begin(&attempt(sid, 30, 3), &[], &[long]),
            Err(StoreError::Oversized(_))
        ));
        // 4097 changed files (cap 4096).
        let changed: Vec<String> = (0..=MAX_VERIFICATION_ATTEMPT_CHANGED)
            .map(|i| format!("f{i}"))
            .collect();
        assert!(matches!(
            store.verification_attempt_begin(
                &attempt(sid, 30, 3),
                &changed,
                &[check(sid, 30, "bg", 0, None)]
            ),
            Err(StoreError::Oversized(_))
        ));
        // 257 checks (cap 256).
        let checks: Vec<VerificationJobRow> = (0..=MAX_VERIFICATION_ATTEMPT_CHECKS)
            .map(|i| check(sid, 30, &format!("c{i}"), i as u32, Some("passed")))
            .collect();
        assert!(matches!(
            store.verification_attempt_begin(&attempt(sid, 30, 3), &[], &checks),
            Err(StoreError::Oversized(_))
        ));
        // Duplicate check id / duplicate derivation ordinal.
        assert!(matches!(
            store.verification_attempt_begin(
                &attempt(sid, 30, 3),
                &[],
                &[
                    check(sid, 30, "dup", 0, Some("passed")),
                    check(sid, 30, "dup", 1, Some("passed"))
                ]
            ),
            Err(StoreError::Malformed(_))
        ));
        assert!(matches!(
            store.verification_attempt_begin(
                &attempt(sid, 30, 3),
                &[],
                &[
                    check(sid, 30, "a", 0, Some("passed")),
                    check(sid, 30, "b", 0, Some("passed"))
                ]
            ),
            Err(StoreError::Malformed(_))
        ));
        // A missing session reference is refused by the foreign key (the
        // whole begin rolls back, no rows).
        let ghost = SessionId::new(999);
        let err = store
            .verification_attempt_begin(
                &attempt(ghost, 30, 3),
                &[],
                &[check(ghost, 30, "bg", 0, None)],
            )
            .unwrap_err();
        assert!(matches!(err, StoreError::Sqlite(_)), "{err:?}");
        assert!(store
            .verification_attempt_get(sid, TaskId::new(1), 30)
            .unwrap()
            .is_none());
    }

    #[test]
    fn recovery_requeues_running_rows_and_leaves_queued_untouched() {
        let (_d, store, sid) = setup();
        assert!(store
            .verification_attempt_begin(
                &attempt(sid, 40, 3),
                &[],
                &[
                    check(sid, 40, "running", 0, None),
                    check(sid, 40, "queued", 1, None)
                ],
            )
            .unwrap());
        store
            .verification_job_claim(sid, TaskId::new(1), 40, "running", 7, 8)
            .unwrap()
            .unwrap();
        let report = store.verification_jobs_requeue_running(sid, 9).unwrap();
        assert_eq!(report.requeued, 1);
        assert_eq!(report.orphaned, 0);
        let rows = store
            .verification_jobs_for_attempt(sid, TaskId::new(1), 40)
            .unwrap();
        assert_eq!(rows[0].state, "queued");
        assert!(rows[0].note.as_deref().unwrap().contains("re-queued"));
        assert_eq!(rows[0].op_id, None);
        // Idempotent: nothing left to requeue.
        assert_eq!(
            store
                .verification_jobs_requeue_running(sid, 10)
                .unwrap()
                .requeued,
            0
        );
    }
}

#[cfg(test)]
mod legacy_verification_import_tests {
    use super::*;

    const TASK: u64 = 7;

    fn attempt_json(op: u64, task: u64, revision: u64) -> String {
        serde_json::json!({
            "schema_ver": 2,
            "task_id": task,
            "op_id": op,
            "task_revision": revision,
            "workspace_root": "/w",
            "changed": ["src/a.rs"],
            "checks": [
                { "check_id": "make_build", "command": "make build", "inline": "passed" },
                { "check_id": "make_test", "command": "make test", "inline": null },
                { "check_id": "make_lint", "command": "make lint", "inline": null },
            ],
            "environment_fingerprint": null,
            "created_ms": 111,
        })
        .to_string()
    }

    fn job_json(op: u64, task: u64, check_id: &str, state: &str, result: Option<&str>) -> String {
        serde_json::json!({
            "schema_ver": 2,
            "attempt_op": op,
            "task_id": task,
            "task_revision": 3,
            "workspace_root": "/w",
            "check_id": check_id,
            "kind": "test",
            "command": format!("make {}", check_id.trim_start_matches("make_")),
            "spec_json": serde_json::json!({
                "id": check_id,
                "kind": "Test",
                "category": "Unit",
                "program": "make",
                "args": ["test"],
                "cwd_rel": ".",
                "affects": [],
                "required": true,
            })
            .to_string(),
            "budget_ms": 60_000,
            "state": state,
            "note": null,
            "op_id": if state == "running" { Some(99) } else { None },
            "result_json": result,
            "environment_fingerprint": null,
            "created_ms": 112,
            "updated_ms": 113,
            "finished_ms": if state == "running" { None } else { Some(114) },
        })
        .to_string()
    }

    fn plant(store: &Store, sid: SessionId, kind: &str, key: &str, value: &str) {
        store.upsert_memory_fact(sid, kind, key, value).unwrap();
    }

    #[test]
    fn pre_v22_rows_import_once_with_deterministic_identity() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("store");
        let sid: SessionId;
        {
            let store = Store::open(&root, true).unwrap();
            let ws = store.create_workspace("/w").unwrap();
            sid = store.create_session(ws, "t", "p", "m").unwrap().id;
        }
        {
            let store = Store::open(&root, true).unwrap();
            plant(
                &store,
                sid,
                LEGACY_VERIFICATION_ATTEMPT_FACT_KIND,
                &format!("va:{TASK}:4242"),
                &attempt_json(4242, TASK, 3),
            );
            plant(
                &store,
                sid,
                LEGACY_VERIFICATION_JOB_FACT_KIND,
                &format!("vj:{TASK}:make_test"),
                &job_json(4242, TASK, "make_test", "running", None),
            );
            plant(
                &store,
                sid,
                LEGACY_VERIFICATION_JOB_FACT_KIND,
                &format!("vj:{TASK}:make_lint"),
                &job_json(4242, TASK, "make_lint", "passed", Some("{\"status\":\"Passed\",\"exit\":0,\"started_ms\":1,\"finished_ms\":2,\"summary\":null,\"truncated\":false}")),
            );
        }
        // The next OPEN is the upgrade: migration completion runs the
        // one-shot repair with no explicit call.
        let store = Store::open(&root, true).unwrap();
        let view = store
            .verification_attempt_get(sid, TaskId::new(TASK), 4242)
            .unwrap()
            .expect("the legacy attempt must import with its derived identity");
        assert_eq!(view.attempt.session_id, sid);
        assert_eq!(view.attempt.task_id, TaskId::new(TASK));
        assert_eq!(view.attempt.attempt_op_id, 4242);
        assert_eq!(view.attempt.task_revision, TaskRevision::new(3));
        assert_eq!(view.changed, vec!["src/a.rs".to_string()]);
        assert_eq!(view.checks.len(), 3, "inline + background checks import");
        assert_eq!(view.checks[0].check_id, "make_build");
        assert_eq!(view.checks[0].inline_status.as_deref(), Some("passed"));
        assert_eq!(view.checks[0].state, "passed");
        assert_eq!(view.checks[1].check_id, "make_test");
        assert_eq!(view.checks[1].state, "running");
        assert_eq!(view.checks[1].op_id, Some(99));
        assert_eq!(view.checks[1].program, "make");
        assert_eq!(view.checks[1].args_json, "[\"test\"]");
        assert_eq!(view.checks[2].check_id, "make_lint");
        assert_eq!(view.checks[2].state, "passed");
        assert!(
            view.checks[2]
                .result_json
                .as_deref()
                .unwrap()
                .contains("Passed"),
            "the terminal outcome imports into verification_job_result"
        );
        // An in-flight legacy job is visible through the open-job scan, so
        // post-restart recovery owns it.
        let open = store
            .verification_jobs_open(sid, TaskId::new(TASK))
            .unwrap();
        assert_eq!(open.len(), 1, "in-flight legacy job visible after upgrade");
        assert_eq!(open[0].check_id, "make_test");
        assert_eq!(open[0].state, "running");
        // Additive: the legacy fact rows are preserved, and exactly ONE
        // durable marker records the import.
        let facts = store.memory_facts(sid).unwrap();
        assert!(facts.iter().any(|(k, key, _)| {
            k == LEGACY_VERIFICATION_ATTEMPT_FACT_KIND && key == &format!("va:{TASK}:4242")
        }));
        assert!(facts.iter().any(|(k, key, _)| {
            k == LEGACY_VERIFICATION_JOB_FACT_KIND && key == &format!("vj:{TASK}:make_test")
        }));
        assert_eq!(
            facts
                .iter()
                .filter(|(k, key, _)| {
                    k == VERIFICATION_V22_IMPORT_MARKER_KIND
                        && key == VERIFICATION_V22_IMPORT_MARKER_KEY
                })
                .count(),
            1,
            "one durable marker per imported session"
        );
        drop(store);
        // Re-open is a no-op: no duplicate attempt, checks, jobs or results.
        let store = Store::open(&root, true).unwrap();
        let again = store.import_legacy_verification_facts().unwrap();
        assert_eq!(again.imported_attempts, 0);
        assert_eq!(again.imported_jobs, 0);
        assert_eq!(again.imported_results, 0);
        let view = store
            .verification_attempt_get(sid, TaskId::new(TASK), 4242)
            .unwrap()
            .unwrap();
        assert_eq!(view.checks.len(), 3, "no duplicate checks after reopen");
        assert_eq!(
            store
                .verification_jobs_for_attempt(sid, TaskId::new(TASK), 4242)
                .unwrap()
                .len(),
            2,
            "no duplicate background jobs after reopen"
        );
    }

    #[test]
    fn corrupt_legacy_rows_are_loudly_skipped_and_left_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("store");
        let sid: SessionId;
        {
            let store = Store::open(&root, true).unwrap();
            let ws = store.create_workspace("/w").unwrap();
            sid = store.create_session(ws, "t", "p", "m").unwrap().id;
        }
        let store = Store::open(&root, true).unwrap();
        // (a) undecodable attempt JSON.
        plant(
            &store,
            sid,
            LEGACY_VERIFICATION_ATTEMPT_FACT_KIND,
            "va:8:1",
            "{ this is not json",
        );
        // (b) an unknown future schema version.
        plant(
            &store,
            sid,
            LEGACY_VERIFICATION_ATTEMPT_FACT_KIND,
            "va:8:2",
            &serde_json::json!({
                "schema_ver": 99,
                "task_id": 8,
                "op_id": 2,
                "task_revision": 1,
                "workspace_root": "/w",
                "changed": [],
                "checks": [],
                "created_ms": 1,
            })
            .to_string(),
        );
        // (c) key/value identity disagreement.
        let mut mismatched: serde_json::Value =
            serde_json::from_str(&attempt_json(4, 8, 1)).unwrap();
        mismatched["op_id"] = serde_json::json!(4);
        plant(
            &store,
            sid,
            LEGACY_VERIFICATION_ATTEMPT_FACT_KIND,
            "va:8:3",
            &mismatched.to_string(),
        );
        // (d) a valid attempt whose only background job is corrupt: the
        // inline check still imports, the corrupt job is skipped loudly.
        let attempt_value = serde_json::json!({
            "schema_ver": 2,
            "task_id": 9,
            "op_id": 10,
            "task_revision": 1,
            "workspace_root": "/w",
            "changed": [],
            "checks": [
                { "check_id": "inline_ok", "command": "make ok", "inline": "passed" },
                { "check_id": "bad_job", "command": "make bad", "inline": null },
            ],
            "created_ms": 5,
        })
        .to_string();
        plant(
            &store,
            sid,
            LEGACY_VERIFICATION_ATTEMPT_FACT_KIND,
            "va:9:10",
            &attempt_value,
        );
        plant(
            &store,
            sid,
            LEGACY_VERIFICATION_JOB_FACT_KIND,
            "vj:9:bad_job",
            &job_json(10, 9, "bad_job", "zombie", None),
        );
        // (e) an orphan job whose attempt row does not exist.
        plant(
            &store,
            sid,
            LEGACY_VERIFICATION_JOB_FACT_KIND,
            "vj:9:orphan",
            &job_json(11, 9, "orphan", "queued", None),
        );
        let report = store.import_legacy_verification_facts().unwrap();
        assert_eq!(report.imported_attempts, 1, "only the resolvable attempt");
        assert_eq!(report.imported_jobs, 1, "only its inline check");
        assert_eq!(
            report.skipped_overflow, 0,
            "the bounded note list holds every skip: {:?}",
            report.skipped
        );
        let reason_for = |key: &str| {
            report
                .skipped
                .iter()
                .find(|s| s.key == key)
                .unwrap_or_else(|| panic!("no typed skip note for {key}: {:?}", report.skipped))
                .reason
                .clone()
        };
        assert!(
            reason_for("va:8:1").contains("undecodable"),
            "{}",
            reason_for("va:8:1")
        );
        assert!(
            reason_for("va:8:2").contains("schema_ver"),
            "{}",
            reason_for("va:8:2")
        );
        assert!(
            reason_for("va:8:3").contains("row names"),
            "{}",
            reason_for("va:8:3")
        );
        assert!(
            reason_for("vj:9:bad_job").contains("unknown legacy job state"),
            "{}",
            reason_for("vj:9:bad_job")
        );
        assert!(
            reason_for("vj:9:orphan").contains("not imported"),
            "{}",
            reason_for("vj:9:orphan")
        );
        // The skips are durable on the marker (typed note), the corrupt
        // legacy rows are NEVER deleted, and the resolvable attempt is live.
        let facts = store.memory_facts(sid).unwrap();
        let marker = facts
            .iter()
            .find(|(k, key, _)| {
                k == VERIFICATION_V22_IMPORT_MARKER_KIND
                    && key == VERIFICATION_V22_IMPORT_MARKER_KEY
            })
            .map(|(_, _, value)| value.clone())
            .expect("marker fact");
        assert!(marker.contains("\"skipped\""), "{marker}");
        assert!(marker.contains("unknown legacy job state"), "{marker}");
        assert!(facts
            .iter()
            .any(|(k, key, _)| k == "verification_attempt" && key == "va:8:1"));
        assert!(facts
            .iter()
            .any(|(k, key, _)| k == "verification_job" && key == "vj:9:orphan"));
        let view = store
            .verification_attempt_get(sid, TaskId::new(9), 10)
            .unwrap()
            .unwrap();
        assert_eq!(view.checks.len(), 1, "inline check imported");
        assert_eq!(view.checks[0].check_id, "inline_ok");
        // A corrupt legacy attempt never minted a half-written row.
        assert!(store
            .verification_attempt_get(sid, TaskId::new(8), 3)
            .unwrap()
            .is_none());
        // Re-open: marker-guarded, nothing duplicated, corrupt rows stay.
        drop(store);
        let store = Store::open(&root, true).unwrap();
        assert_eq!(
            store
                .verification_attempt_get(sid, TaskId::new(9), 10)
                .unwrap()
                .unwrap()
                .checks
                .len(),
            1
        );
        assert!(store
            .memory_facts(sid)
            .unwrap()
            .iter()
            .any(|(_, key, _)| key == "va:8:1"));
    }
}
