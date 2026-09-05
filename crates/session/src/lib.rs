//! faktor-session — the durable session runtime.
//!
//! This crate owns the *durable* half of a Faktor session: the journaled
//! state machine, the conversation view, tool-run ledger, permissions,
//! checkpoints, memory facts, compaction records and crash recovery. It sits
//! directly on `faktor-store` (SQLite) and `faktor-cas` (content-addressed
//! blobs) and speaks the frozen `faktor-protocol::v756` shapes. The agent
//! reasoning loop (`faktor-agent`) drives this crate with commands; tools and
//! providers never touch session persistence directly (Commandment 1).
//!
//! Invariants this crate enforces:
//!
//! - **Every command appends journal events** via `Store::append_event` with a
//!   `StateMachine`-validated transition. An illegal transition returns
//!   `ErrorKind::InvalidState` and leaves **no trace** in the journal or the
//!   session row — validation always precedes any write.
//! - **Recovery is reconstructible and idempotent.** `SessionHandle::recover_all`
//!   scans durable `tool_run` rows that are still `running`, applies the four
//!   actionable recovery strategies (`VerifyHash`, `MarkUnknown`, `Idempotent`,
//!   `Manual`; `None` records no action), finishes them, and journals
//!   `CrashDetected` + one `RecoveryApplied` per op. A second run finds nothing
//!   pending and appends nothing.
//! - **Paging is fundamental.** `messages_page` never loads more than one page
//!   of messages (plus one row to learn `has_more`), and parts are fetched per
//!   message in the page only.
//! - **Bounded everything.** Prompts, message payloads, parts, artifacts,
//!   ledger, recovery file reads and pages all carry hard byte/count limits;
//!   oversized input is rejected *before* any write.
//! - **Zero orphans.** Child processes registered on a session block
//!   `end_session` until released or deliberately transferred; after a crash
//!   the registry is reported and cleared (parent-death handling is the OS's
//!   job; the runtime's is to notice and record).
//! - **Compaction hard invariant.** A compaction is accepted only if the
//!   result is at or below the target *and* the reduction is at least
//!   `CompactionPolicy::min_reduction_ratio` (default 25%). A "summary" that
//!   reduces context by ~1% is rejected with a `CompactRejected` event.
//!
//! The API is deliberately **synchronous** and `Send + Sync`: there is no bare
//! `tokio::spawn` defining application state here. Async callers (the server)
//! wrap heavy calls in `tokio::task::spawn_blocking`; short calls may run
//! inline. All public methods return `faktor_core::Result<T>` so the server can
//! map failures with `faktor_protocol::error::from_core` without a second error
//! surface.

pub mod actor;
pub mod artifacts;
pub mod checkpoints;
pub mod compaction;
pub mod handle;
pub mod journal;
pub mod ledger;
pub mod manager;
pub mod memory;
pub mod messages;
pub mod ops;
pub mod payload;
pub mod process;
pub mod recovery;
pub mod sse;
pub mod task;

pub use actor::{DbActor, DbActorConfig, DbActorStats, StoreHandle};
pub use handle::{AbortReceipt, PromptReceipt, SessionHandle};
pub use ledger::{
    blocker_is_open, LedgerCheckRun, LedgerChild, LedgerCompactReport, LedgerDecision,
    LedgerEntryPage, LedgerHead, LedgerPayload, LedgerPlanStep, LedgerRouting, LedgerVerifySummary,
    LedgerView, TypedLedgerEntry, MAX_LEDGER_PAGE,
};
pub use manager::SessionManager;
pub use ops::{OpKind, PermissionRequest, ToolRunHandle};
pub use payload::{decode_payload, PAYLOAD_SCHEMA_V};
pub use process::OwnedProcess;
pub use recovery::{FileHasher, RecoveredOp, RecoveryAction, RecoveryReport, SystemFileHasher};
pub use sse::JournalFrame;
pub use task::{
    Task, TaskBudget, TaskPatch, MAX_TASK_CRITERIA, MAX_TASK_CRITERION_BYTES, MAX_TASK_GOAL_BYTES,
    MAX_TASK_PLAN_STEPS, MAX_TASK_STEP_BYTES,
};

/// Errors of the session runtime. The public API surface returns
/// [`faktor_core::Error`] (via `From`), so protocol/HTTP mapping stays single-sourced
/// in `faktor-protocol::error::from_core`.
#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("session resource not found: {0}")]
    NotFound(String),
    #[error("illegal state transition {from:?} -> {to:?}: {message}")]
    IllegalTransition {
        from: faktor_core::state::AgentState,
        to: faktor_core::state::AgentState,
        message: String,
    },
    #[error("store failure: {0}")]
    Store(#[from] faktor_store::StoreError),
    #[error("cas failure: {0}")]
    Cas(faktor_cas::CasError),
    #[error("malformed input: {0}")]
    Malformed(String),
    #[error("input exceeds bound: {0}")]
    Oversized(String),
    #[error("conflict: {0}")]
    Conflict(String),
    #[error("operation cancelled")]
    Cancelled,
    #[error("deadline exceeded: {0}")]
    Timeout(String),
    #[error("permission denied: {0}")]
    Permission(String),
    #[error("internal invariant violation: {0}")]
    Internal(String),
}

impl SessionError {
    pub fn illegal(
        from: faktor_core::state::AgentState,
        to: faktor_core::state::AgentState,
    ) -> Self {
        Self::IllegalTransition {
            from,
            to,
            message: format!("{} -> {}", from.label(), to.label()),
        }
    }
}

impl From<faktor_core::Error> for SessionError {
    fn from(e: faktor_core::Error) -> Self {
        match e.kind {
            faktor_core::ErrorKind::NotFound => SessionError::NotFound(e.message),
            faktor_core::ErrorKind::InvalidState { from, to } => SessionError::illegal(from, to),
            faktor_core::ErrorKind::Permission => SessionError::Permission(e.message),
            faktor_core::ErrorKind::Timeout => SessionError::Timeout(e.message),
            faktor_core::ErrorKind::Cancelled => SessionError::Cancelled,
            faktor_core::ErrorKind::Conflict => SessionError::Conflict(e.message),
            faktor_core::ErrorKind::Malformed => SessionError::Malformed(e.message),
            faktor_core::ErrorKind::Oversized => SessionError::Oversized(e.message),
            kind => SessionError::Internal(format!("{kind:?}: {}", e.message)),
        }
    }
}

impl From<faktor_cas::CasError> for SessionError {
    fn from(e: faktor_cas::CasError) -> Self {
        match e {
            // Missing blobs are missing resources, not storage failures.
            faktor_cas::CasError::NotFound(hash) => SessionError::NotFound(hash.to_string()),
            other => SessionError::Cas(other),
        }
    }
}

impl From<SessionError> for faktor_core::Error {
    fn from(e: SessionError) -> Self {
        match e {
            SessionError::NotFound(m) => faktor_core::Error::not_found(m),
            SessionError::IllegalTransition { from, to, message } => {
                faktor_core::Error::new(faktor_core::ErrorKind::InvalidState { from, to }, message)
            }
            SessionError::Store(m) => faktor_core::Error::new(
                faktor_core::ErrorKind::Store,
                format!("session store: {m}"),
            ),
            SessionError::Cas(m) => {
                faktor_core::Error::new(faktor_core::ErrorKind::Store, format!("session cas: {m}"))
            }
            SessionError::Malformed(m) => faktor_core::Error::malformed(m),
            SessionError::Oversized(m) => faktor_core::Error::oversized(m),
            SessionError::Conflict(m) => faktor_core::Error::conflict(m),
            SessionError::Cancelled => faktor_core::Error::cancelled(),
            SessionError::Timeout(m) => faktor_core::Error::timeout(m),
            SessionError::Permission(m) => faktor_core::Error::permission(m),
            SessionError::Internal(m) => faktor_core::Error::internal(m),
        }
    }
}

/// The `EffectStatus` string contract used in `tool_run` rows.
pub(crate) fn effect_str(e: faktor_core::op::EffectStatus) -> &'static str {
    match e {
        faktor_core::op::EffectStatus::Unknown => "unknown",
        faktor_core::op::EffectStatus::Verified => "verified",
        faktor_core::op::EffectStatus::Applied => "applied",
        faktor_core::op::EffectStatus::Failed => "failed",
    }
}

/// Snake_case wire tag for an agent state (mirrors core's serde rename).
pub(crate) fn state_tag(s: faktor_core::state::AgentState) -> String {
    serde_json::to_string(&s)
        .unwrap_or_default()
        .trim_matches('"')
        .to_string()
}

// ---------------------------------------------------------------- bounds

/// Hard limit on a single user prompt (UTF-8 bytes).
pub const MAX_PROMPT_BYTES: usize = 512 * 1024;
/// Hard limit on files attached to one prompt.
pub const MAX_FILES_PER_PROMPT: usize = 64;
/// Hard limit on one attached file path.
pub const MAX_FILE_PATH_BYTES: usize = 4096;
/// Hard limit on one message payload.
pub const MAX_MESSAGE_BYTES: usize = 1 << 20;
/// Hard limit on one message part payload.
pub const MAX_PART_BYTES: usize = 4 << 20;
/// Hard limit on one tool-run argument payload.
pub const MAX_TOOL_ARGS_BYTES: usize = 4 << 20;
/// Hard limit on one artifact blob (decompressed).
pub const MAX_ARTIFACT_BYTES: usize = 64 << 20;
/// Hard limit on an artifact summary string.
pub const MAX_ARTIFACT_SUMMARY_BYTES: usize = 4096;
/// Maximum messages a single page may return.
pub const MAX_PAGE_SIZE: i64 = 200;
/// Hard limit on the task ledger JSON.
pub const MAX_LEDGER_BYTES: usize = 1 << 20;
/// Hard limit on a recovery file read for hash verification.
pub const MAX_VERIFY_BYTES: usize = 64 << 20;

// ---------------------------------------------------------------- layered budgets
// (Audit 26) Lifetimes are LAYERED and bounded; there is deliberately NO
// 24h constant anywhere in this crate:
// - `op_budget_ms` bounds ONE tool/operation (small; owned by the agent
//   runtime's `tool_deadline_ms`, this crate never sees it);
// - `turn_budget_ms` bounds ONE logical turn (default 30 min — the prompt
//   operation's deadline and the runtime's per-turn slice ceiling);
// - a TASK's lifetime is bounded by its durable budget (max_tokens /
//   max_turns) and NEVER by a single future: no runtime future is
//   scheduled for more than one `turn_budget_ms` slice, progress is
//   persisted via the ledger between slices, and the turn loop re-enters.
/// Default wall-clock budget of ONE logical turn (not 24h): a turn's
/// operation deadline and the runtime's per-turn slice ceiling. The task
/// itself spans many turns across restarts, never one future.
pub const DEFAULT_TURN_BUDGET_MS: u64 = 30 * 60 * 1000;

// ---------------------------------------------------------------- helpers

/// Map a store error, translating SQLite constraint violations (duplicate
/// message seq, unknown workspace FK, missing message on part insert, ...)
/// and store-level conflicts (atomic transition expectation mismatches) into
/// a `Conflict` instead of a raw store error. rusqlite renders these as
/// `"UNIQUE constraint failed: ..."`, `"FOREIGN KEY constraint failed: ..."`
/// and `"NOT NULL constraint failed: ..."` — all contain `constraint failed`.
/// A queued prompt admitted atomically as the active turn.
#[derive(Debug, Clone)]
pub struct AdmittedQueuedPrompt {
    pub queue_seq: i64,
    pub op_id: faktor_core::id::OpId,
    pub prompt: String,
    pub files: Vec<String>,
    pub model: Option<String>,
    pub variant: Option<String>,
    pub agent: Option<String>,
    pub message_seq: i64,
}

pub(crate) fn map_store_err(e: faktor_store::StoreError) -> SessionError {
    match e {
        faktor_store::StoreError::Conflict(message) => SessionError::Conflict(message),
        faktor_store::StoreError::Sqlite(sqlite)
            if sqlite.to_string().contains("constraint failed") =>
        {
            SessionError::Conflict(format!("store constraint violation: {sqlite}"))
        }
        other => SessionError::Store(other),
    }
}

/// Size of a JSON value in bytes (for bounds checks before writes).
pub(crate) fn json_bytes(v: &serde_json::Value) -> usize {
    serde_json::to_vec(v).map(|b| b.len()).unwrap_or(usize::MAX)
}
