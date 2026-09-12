//! Real child execution for the orchestrator (audits 20-24).
//!
//! [`OrchestratorRuntime::execute_task`] turns a validated [`TaskPlan`] into
//! REAL runtime work:
//!
//! ```text
//! TaskPlan -> ExecState (durable plan row) -> Scheduler (per-execution
//!   DAG with ceiling limits) -> Child Session + Worktree (SessionManager
//!   rows + real directory) -> AgentRuntime drive (the daemon's own drive
//!   entry; the executor awaits the child's op-record end — never polls)
//! ```
//!
//! Everything a child needs to survive a crash is durable:
//!
//! - **Child session rows** (real `session` rows) carry the adopted
//!   worktree identity; the child's own row space records its
//!   [`faktor_session::ChildIdentity`] (parent, worktree, ownership, item)
//!   and its drive state (Waiting phase, current steering note/model).
//! - **Registry rows** under the parent session (`orchestrator_registry` /
//!   `<run_id>/<child_id>`) are the durable [`ChildRuntime`] records:
//!   ownership mode, worktree, budget, effective capability set, model
//!   policy and state.
//! - **Control rows** under each child (`orchestrator_ctl`) are the durable
//!   steering queue; the AGENT drive applies them at its safe reasoning
//!   boundary and acks each exactly once.
//!
//! Ceilings (audit 24): [`ceilings::MAX_LIVE_CHILDREN`] (32) live children
//! hard-reject with the typed [`ExecError::CeilingExceeded`];
//! `max_reasoning_active` (4) and `max_mutating_active` (2) defer excess
//! ready work to the next wave. The per-execution [`Scheduler`] carries the
//! same class limits as backstop and refuses conflicting registrations with
//! `Conflict` before anything runs.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use faktor_agent::{AgentRuntime, TurnOutcome};
use faktor_core::cancellation::CancellationToken;
use faktor_core::id::{OpId, SessionId, TaskId, WorktreeId};
use faktor_core::op::{OpMeta, RecoveryStrategy};
use faktor_core::retry::RetryPolicy;
use faktor_core::state::AgentState;
use faktor_core::time::{Deadline, SystemClock};
use faktor_scheduler::{OwnershipSet as SchOwnershipSet, ResourceRequest, ScheduledOp, Scheduler};
use faktor_semantic::RiskLevel;
use faktor_session::child::{ChildControl, ChildIdentity, ChildOwnership, ChildPhase};
use faktor_session::{SessionManager, TaskBudget};

use crate::caps::{effective, CapabilitySet};
use crate::{ChildState, OwnershipSpec, WorkItem, WorkKind, WorkState};

// The child blocker vocabulary is the ONE shared core definition: the
// orchestrator's historic `runtime::ChildBlocker` path re-exports it so
// downstream callers keep resolving, while the kind is now a closed enum.
pub use faktor_core::blocker::{BlockerKind, ChildBlocker, ExecutionPhase};

pub mod ceilings {
    //! Audit 24 ceilings. The old `MAX_CHILDREN = 1000` literal is gone:
    //! 1000 was never a useful bound because children now hold REAL
    //! sessions, worktrees, drives and control queues.
    /// Hard ceiling on LIVE (non-terminal) children per execution.
    pub const MAX_LIVE_CHILDREN: usize = 32;
    /// Default ceiling on concurrently reasoning (read-only) children.
    pub const DEFAULT_MAX_REASONING_ACTIVE: usize = 4;
    /// Default ceiling on concurrently active mutating children (2-4).
    pub const DEFAULT_MAX_MUTATING_ACTIVE: usize = 2;
}

/// Registry/plan row kinds in the parent session's durable fact space.
pub const REGISTRY_ROW_KIND: &str = "orchestrator_registry";
pub const PLAN_ROW_KIND: &str = "orchestrator_plan";
/// Durable work-item → child binding rows (wave A3 identity contract):
/// kind `orchestrator_assign`, key `<run_id>/<item_id>` under the parent
/// session. Minted for every SPAWN work item in deterministic plan order
/// at plan compile and committed ATOMICALLY (one store transaction)
/// BEFORE any child spawn; immutable afterwards. Child identity (the
/// `child-N` ids), spawn, re-attach and the operation graph all read these
/// rows — never spawn order, iteration order or completion order.
pub const ASSIGNMENT_ROW_KIND: &str = "orchestrator_assign";
pub const MAX_RUN_ID_CHARS: usize = 64;
/// One child drive may hold the turn at most this long at the op level
/// (the agent's own per-turn slice budget is the tighter bound).
pub const CHILD_OP_DEADLINE_MS: i64 = 2 * 60 * 60 * 1000;

/// Configurable concurrency ceilings of one execution (audit 24).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Ceilings {
    /// Live (non-terminal) children: exceeding this is a hard typed reject.
    pub max_live: usize,
    pub max_reasoning_active: usize,
    pub max_mutating_active: usize,
}

impl Default for Ceilings {
    fn default() -> Self {
        Self {
            max_live: ceilings::MAX_LIVE_CHILDREN,
            max_reasoning_active: ceilings::DEFAULT_MAX_REASONING_ACTIVE,
            max_mutating_active: ceilings::DEFAULT_MAX_MUTATING_ACTIVE,
        }
    }
}

impl Ceilings {
    pub fn validate(&self) -> Result<(), String> {
        if self.max_live == 0 {
            return Err("ceilings.max_live must be >= 1".into());
        }
        if self.max_reasoning_active == 0 {
            return Err("ceilings.max_reasoning_active must be >= 1".into());
        }
        if self.max_mutating_active == 0 {
            return Err("ceilings.max_mutating_active must be >= 1".into());
        }
        Ok(())
    }
}

/// Child-parallelism reduction under semantic risk (audit 54; audit 119 for
/// the risk-after-a-child delta): once a settled child drive reported
/// `High`/`Unknown` semantic risk, the run admits fewer concurrent children
/// (`max_mutating_active`/`max_reasoning_active` drop to 1) so a risky
/// change is never fanned out; `Medium` halves the wave. `None` (no
/// provider consulted), `Safe` and `Low` keep the configured ceilings
/// byte-identically; `max_live` is never raised or lowered (it is the hard
/// registration bound, not a wave knob).
pub fn risk_adjusted_ceilings(ceilings: &Ceilings, risk: Option<RiskLevel>) -> Ceilings {
    let mut adjusted = ceilings.clone();
    match risk {
        Some(RiskLevel::High) | Some(RiskLevel::Unknown) => {
            adjusted.max_mutating_active = 1;
            adjusted.max_reasoning_active = 1;
        }
        Some(RiskLevel::Medium) => {
            adjusted.max_mutating_active = adjusted.max_mutating_active.min(2);
            adjusted.max_reasoning_active = adjusted.max_reasoning_active.min(2);
        }
        Some(RiskLevel::Safe) | Some(RiskLevel::Low) | None => {}
    }
    adjusted
}

/// Severity ordering used to fold settled-child risks (`RiskLevel`'s own
/// `Ord` is the axis order, where Unknown sorts BELOW Safe — not the
/// escalation order). Unknown outranks High: not knowing escalates hardest.
fn risk_severity(level: RiskLevel) -> u8 {
    match level {
        RiskLevel::Unknown => 4,
        RiskLevel::High => 3,
        RiskLevel::Medium => 2,
        RiskLevel::Low => 1,
        RiskLevel::Safe => 0,
    }
}

/// Typed execution error of the orchestrator runtime.
#[derive(Debug, thiserror::Error)]
pub enum ExecError {
    #[error("ceiling exceeded: {class} children (limit {limit}, in use {used}); nothing was registered for this child")]
    CeilingExceeded {
        class: &'static str,
        limit: usize,
        used: usize,
    },
    #[error("ownership overlap: {0} (normalized path sets of concurrent mutating children must be disjoint)")]
    OverlappingExclusiveOwnership(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("conflict: {0}")]
    Conflict(String),
    #[error("invalid child state: {0}")]
    InvalidState(String),
    #[error("malformed input: {0}")]
    Malformed(String),
    #[error("oversized: {0}")]
    Oversized(String),
    #[error("plan validation failed: {0}")]
    InvalidPlan(String),
    #[error("invalid merge approval: {0}")]
    InvalidApproval(String),
    /// The OPTIONAL semantic merge preflight (audit 79) refused a merge: the
    /// provider's base→candidate delta contradicts the staged digests. It is
    /// an ADDITIONAL typed refusal only — a real CAS/file conflict is never
    /// overridden by it (and never resolved by provider data).
    #[error("semantic merge conflict: {0}")]
    SemanticConflict(String),
    #[error("merge decision incomplete: {0}")]
    UndecidedPaths(String),
    /// An AUTHORITATIVE durable write or control-queue ack failed. The
    /// transition was NOT claimed applied, the effect is idempotent, and the
    /// underlying failure classifies as retryable — the SAME call may be
    /// safely retried (which is why `let _ =` on these paths is forbidden).
    #[error("retriable persistence failure during {operation}: {message}")]
    RetriablePersistence { operation: String, message: String },
    #[error("injected crash seam {0} (test seam; durable state left as-is for re-attach)")]
    InjectedCrashSeam(String),
    #[error("internal: {0}")]
    Internal(String),
}

impl ExecError {
    /// Wrap a failed authoritative durable write/ack as a typed retriable
    /// error: the caller may retry the whole (idempotent) control call.
    pub(crate) fn retriable(operation: &str, e: impl std::fmt::Display) -> Self {
        ExecError::RetriablePersistence {
            operation: operation.to_string(),
            message: e.to_string(),
        }
    }

    /// Map a faktor-fs layer error of a merge/snapshot/copy operation onto
    /// the typed orchestrator error space (typed mapping — the merge never
    /// swallows an fs failure).
    pub(crate) fn from_fs(what: &str, root: &std::path::Path, e: faktor_core::Error) -> Self {
        match e.kind {
            faktor_core::ErrorKind::Oversized => {
                ExecError::Oversized(format!("{what} (root {:?}): {}", root, e.message))
            }
            faktor_core::ErrorKind::Conflict => {
                ExecError::Conflict(format!("{what}: {}", e.message))
            }
            faktor_core::ErrorKind::NotFound => {
                ExecError::NotFound(format!("{what}: {}", e.message))
            }
            other => ExecError::Internal(format!("{what}: {:?}: {}", other, e.message)),
        }
    }

    /// Prefix a shadow-service error with its context, keeping the typed
    /// variant (P0-48 wiring: a refused shadow begin stays typed so callers
    /// distinguish Oversized copies, live-shadow conflicts, and vanished
    /// checkouts).
    pub(crate) fn from_shadow(what: impl std::fmt::Display, e: Self) -> Self {
        match e {
            ExecError::Conflict(m) => ExecError::Conflict(format!("{what}: {m}")),
            ExecError::NotFound(m) => ExecError::NotFound(format!("{what}: {m}")),
            ExecError::Oversized(m) => ExecError::Oversized(format!("{what}: {m}")),
            ExecError::InvalidState(m) => ExecError::InvalidState(format!("{what}: {m}")),
            other => other,
        }
    }
}

impl From<faktor_core::Error> for ExecError {
    fn from(e: faktor_core::Error) -> Self {
        match e.kind {
            faktor_core::ErrorKind::NotFound => ExecError::NotFound(e.message),
            faktor_core::ErrorKind::Conflict => ExecError::Conflict(e.message),
            _ => ExecError::Internal(format!(
                "{kind:?}: {message}",
                kind = e.kind,
                message = e.message
            )),
        }
    }
}

/// Deterministic crash seams (adversarial tests only): execution returns
/// [`ExecError::InjectedCrashSeam`] at the FIRST matching point, leaving
/// every durable row exactly as a real crash would.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrashSeam {
    /// After the child session + registry rows exist, before its first
    /// drive is submitted.
    BeforeDrive,
    /// After a child reached its terminal state and its durable rows were
    /// settled, before the parent continuation admits further work.
    AfterChildTerminal,
    /// Controlled merge: fires after the durable in-flight merge record was
    /// written, before any file apply (approve_and_merge crash window).
    AfterMergeRecord,
    /// Controlled merge: fires once `after` files of the apply loop were
    /// processed (any outcome), leaving a partially applied merge that a
    /// replay must reconcile through the CAS.
    MergeApply { after: usize },
    /// Wave A3: fires right after the run's work-item → child assignment
    /// rows committed, BEFORE any child spawn. Re-open must resume with the
    /// SAME child ids (the assignment rows are durable; nothing was minted
    /// at spawn).
    AfterAssignmentsPersisted,
}

/// The child's model policy (typed, bounded, durable).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(default)]
pub struct ModelPolicy {
    /// The model selector this child drives with (None = daemon default).
    pub model: Option<String>,
}

/// Per-child policy for one work item. Capabilities are typed
/// ([`CapabilitySet`]) — no free-form string ever occupies a permission
/// position. Ownership is NEVER carried here: the ITEM's own
/// [`crate::WorkItem::ownership`] is the only write authority (work-entry
/// unification), compiled before any durable row and read back from the
/// work-item → child assignment row at spawn.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ChildSpec {
    pub item_id: String,
    /// `false` executes the item without a real child session.
    pub spawn: bool,
    pub model: Option<String>,
    /// Durable token budget cap (the wave-9 Task budget fields).
    pub max_tokens: Option<u64>,
    /// Attachment file paths of the run (the SDK `PromptRequest.files`
    /// vocabulary). They are part of the DURABLE spec (the plan row carries
    /// the whole spec list), so every child — fresh spawn AND re-attach —
    /// submits the byte-identical set; a durable tampered list is a typed
    /// refusal before any spawn ([`validate_attachment_files`]). Old rows
    /// decode with an empty list (field-level serde default): the
    /// attachment-free run stays byte-identical.
    pub files: Vec<String>,
    /// Task-level typed policy.
    pub task_caps: CapabilitySet,
    /// Child-level typed policy.
    pub child_caps: CapabilitySet,
}

impl Default for ChildSpec {
    fn default() -> Self {
        Self {
            item_id: String::new(),
            spawn: true,
            model: None,
            max_tokens: None,
            files: Vec::new(),
            task_caps: CapabilitySet::new(),
            child_caps: CapabilitySet::new(),
        }
    }
}

impl ChildSpec {
    pub fn new(item_id: impl Into<String>) -> Self {
        Self {
            item_id: item_id.into(),
            ..Default::default()
        }
    }
}

/// Max concurrently OPEN blockers a session may hold (mirrors the session
/// ledger bound; a child blocker reason must stay resolvable there).
pub const MAX_CHILD_BLOCKER_KIND_CHARS: usize = 64;
pub const MAX_CHILD_BLOCKER_REASON_CHARS: usize = 512;
pub const MAX_CHILD_BLOCKER_DEPENDENCY_CHARS: usize = 64;
pub const MAX_CHILD_BLOCKER_RESOLUTION_CHARS: usize = 512;

/// The durable runtime object of ONE child (audit: every field durable).
/// `state` mirrors the child's [`AgentState`] through [`ChildState`].
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ChildRuntime {
    pub child_id: String,
    pub parent_session_id: u64,
    pub run_id: String,
    pub item_id: String,
    pub kind: WorkKind,
    /// The REAL child session row id.
    pub session_id: u64,
    /// The child's durable turn operation id (the child session's op id),
    /// 0 until its first submit.
    pub operation_id: u64,
    pub workspace_id: u64,
    pub worktree_id: u64,
    pub ownership: ChildOwnership,
    /// Normalized exclusive write paths (empty unless `ExclusivePaths`).
    pub ownership_paths: Vec<String>,
    pub state: ChildState,
    /// Durable token budget cap (max_tokens); None = unlimited.
    pub budget_max_tokens: Option<u64>,
    /// effective(child) = parent ∩ task_policy ∩ child_policy.
    pub permissions: CapabilitySet,
    pub model_policy: ModelPolicy,
    /// Durable blocker truth: set when `state == Blocked`, cleared by any
    /// transition back to `Running`. Old rows decode with `None` fields
    /// (field-level serde defaults; v23 adds the typed store projection).
    #[serde(default)]
    pub blocker_kind: Option<String>,
    #[serde(default)]
    pub blocker_reason: Option<String>,
    #[serde(default)]
    pub blocker_dependency: Option<String>,
    #[serde(default)]
    pub blocker_resolution: Option<String>,
    #[serde(default)]
    pub last_progress_ms: Option<i64>,
    pub created_ms: i64,
    pub updated_ms: i64,
    /// The durable base-snapshot id the child started from (audits 70/98):
    /// recorded at spawn for isolated worktree children; `None` for
    /// children without an own worktree. Old durable rows decode with
    /// `None` (field-level serde default).
    #[serde(default)]
    pub base_snapshot_id: Option<String>,
    /// The durable env-snapshot id (audit 97): the child is bound to the
    /// IMMUTABLE instruction-epoch snapshot taken from its environment at
    /// spawn. Context building for this child reads ONLY that snapshot —
    /// later changes to the parent's rules can never bleed in. Every spawn
    /// path binds one; old durable rows decode with `None`, which the
    /// pinned-context reader treats as an incomplete spawn and refuses
    /// loudly (no silent fallback to the live environment).
    #[serde(default)]
    pub env_snapshot_id: Option<String>,
    /// The child's coarse execution phase (additive): persisted at safe
    /// drive boundaries into the child's own durable drive-state row; every
    /// registry read derives the freshest value. Old registry rows decode as
    /// [`ExecutionPhase::Planning`]. Purely a projection — it never changes
    /// lifecycle, scheduling or budget semantics.
    #[serde(default)]
    pub execution_phase: ExecutionPhase,
}

impl ChildRuntime {
    pub fn is_terminal(&self) -> bool {
        self.state.is_terminal()
    }

    pub fn is_mutating(&self) -> bool {
        self.kind.is_mutating()
    }

    /// The typed blocker this row carries (None when not blocked, or when a
    /// hostile kind made it undecodable — durable decode validates first via
    /// [`ChildRuntime::validate_durable`], so a `None` here is either a
    /// genuine non-block or an in-memory construction).
    pub fn blocker(&self) -> Option<ChildBlocker> {
        let kind = BlockerKind::parse(self.blocker_kind.as_deref()?)?;
        Some(ChildBlocker {
            kind,
            reason: self.blocker_reason.clone().unwrap_or_default(),
            dependency: self.blocker_dependency.clone(),
            resolution: self.blocker_resolution.clone(),
            last_progress_ms: self.last_progress_ms,
        })
    }

    /// Strict durable-decode validation of one registry row: a known blocker
    /// kind and the invariant `state == Blocked <=> a populated blocker`.
    /// Corrupt/hostile rows are a loud typed error — never a fabricated
    /// state.
    pub fn validate_durable(&self) -> Result<(), ExecError> {
        match self.blocker_kind.as_deref() {
            None => {}
            Some(raw) if BlockerKind::parse(raw).is_none() => {
                return Err(ExecError::Malformed(format!(
                    "registry row of child {} carries blocker kind {raw:?} outside the closed vocabulary",
                    self.child_id
                )));
            }
            Some(_) => {}
        }
        let blocked = self.state == ChildState::Blocked;
        if blocked {
            let blocker = self.blocker().ok_or_else(|| {
                ExecError::Malformed(format!(
                    "registry row of child {} is Blocked without a blocker",
                    self.child_id
                ))
            })?;
            blocker.validate()?;
        } else if self.blocker_kind.is_some() {
            return Err(ExecError::Malformed(format!(
                "registry row of child {} is {:?} but carries a blocker",
                self.child_id, self.state
            )));
        }
        Ok(())
    }

    /// Mark this row Blocked with the given (validated) blocker. The state
    /// and the blocker fields move together — a Blocked row without a
    /// populated blocker is unrepresentable through this path.
    pub fn set_blocker(&mut self, blocker: &ChildBlocker) -> Result<(), ExecError> {
        blocker.validate()?;
        self.state = ChildState::Blocked;
        self.blocker_kind = Some(blocker.kind.as_str().to_string());
        self.blocker_reason = Some(blocker.reason.clone());
        self.blocker_dependency = blocker.dependency.clone();
        self.blocker_resolution = blocker.resolution.clone();
        self.last_progress_ms = blocker.last_progress_ms;
        Ok(())
    }

    /// Clear every blocker field (a transition back to Running). Returns
    /// the previously recorded blocker, when any, for the audit ledger.
    pub fn clear_blocker(&mut self) -> Option<ChildBlocker> {
        let previous = self.blocker();
        self.blocker_kind = None;
        self.blocker_reason = None;
        self.blocker_dependency = None;
        self.blocker_resolution = None;
        self.last_progress_ms = None;
        previous
    }
}

/// The ONE canonical child-state projection: every surface that turns a
/// durable [`ChildRuntime`] into a [`WorkState`] calls THIS function.
///
/// A local "everything non-terminal is Running" conversion is a truth
/// defect: it erases Paused, Waiting and Blocked. The mapping is total —
/// every [`ChildState`] has exactly one [`WorkState`].
pub fn project_child_state(child: &ChildRuntime) -> WorkState {
    match child.state {
        ChildState::Running => WorkState::Running,
        ChildState::Paused => WorkState::Paused,
        ChildState::Waiting => WorkState::Waiting,
        ChildState::Blocked => WorkState::Blocked,
        ChildState::Done => WorkState::Done,
        ChildState::Failed => WorkState::Failed,
        ChildState::Cancelled => WorkState::Cancelled,
    }
}

/// The stable lowercase machine tag of one child state (the typed store
/// projection's `state` column).
pub fn child_state_tag(state: ChildState) -> &'static str {
    match state {
        ChildState::Running => "running",
        ChildState::Paused => "paused",
        ChildState::Waiting => "waiting",
        ChildState::Blocked => "blocked",
        ChildState::Cancelled => "cancelled",
        ChildState::Done => "done",
        ChildState::Failed => "failed",
    }
}

/// One durable work-item → child binding of a run (wave A3): minted for
/// EVERY spawn work item in deterministic PLAN order at plan compile,
/// persisted atomically BEFORE any child spawn, and immutable afterwards.
/// A child's identity therefore never depends on spawn order, iteration
/// order or completion order — re-attach and the operation graph both ask
/// "what child does the durable row name for this plan item?".
///
/// The row ALSO records the item's OWN ownership (audits 7/8/21/22,
/// work-entry unification): the actual per-work-item write authority, read
/// directly from [`crate::WorkItem::ownership`] at compile — there is no
/// request-level or plan-level fallback. Spawn reads the ownership FROM
/// THIS ROW and never re-derives it from the plan — a crashed executor
/// re-attaches to exactly the ownership it would have spawned with.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WorkItemAssignment {
    pub run_id: String,
    pub item_id: String,
    /// Position of `item_id` in the durable plan's `work_items` (0-based
    /// plan order). Plan children order deterministically by this index.
    pub plan_step_index: usize,
    /// The plan child's durable id (`child-N`, reserved from the shared
    /// child sequence at compile — reviewers mint `review-N` ids strictly
    /// after all plan ids).
    pub child_id: String,
    /// The effective per-item ownership this child was assigned. Rows
    /// without it are hostile/stale (never spawned under a re-derived
    /// ownership).
    pub ownership: OwnershipSpec,
}

/// The read-only global orphan-child scan (`doctor --deep`, P0-97): every
/// durable child identity row (kind `orchestrator`, key `identity`, in the
/// CHILD session's row space) and every executor registry row (kind
/// `orchestrator_registry`, in the PARENT session's row space) is checked
/// against the session table and the filesystem. Nothing is written.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OrphanChildScan {
    /// Child identity rows scanned.
    pub identity_rows: usize,
    /// Executor registry rows scanned.
    pub registry_rows: usize,
    /// Human-readable violations (empty = no orphans).
    pub issues: Vec<String>,
}

/// A durable summary of a finished (or deferred) run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanOutcome {
    /// item id -> WorkState, plan order.
    pub item_states: Vec<(String, WorkState)>,
    pub complete: bool,
    pub failed: Vec<String>,
    pub cancelled: Vec<String>,
    /// Items whose children still wait for a Resume row (re-attach of an
    /// executor that crashed while a child was paused).
    pub waiting: Vec<String>,
    /// Every durable child row of the run.
    pub children: Vec<ChildRuntime>,
}

/// The wire ack of one enqueued child control (audit 23): the durable
/// control-row `queued_seq` plus the exactly-once applied state — `true`
/// when the effect was applied synchronously at enqueue (Cancel, budget
/// patch), `None` while the row waits for the child's next safe boundary
/// (the child's own drive acks it exactly once when it applies).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ControlAck {
    pub queued_seq: u64,
    pub applied: Option<bool>,
}

/// What the durable plan row decodes into on re-attach.
struct DurablePlanRow {
    plan: crate::TaskPlan,
    owner: OwnerContext,
    specs: HashMap<String, ChildSpec>,
    provider: String,
    default_model: String,
    isolated_root: PathBuf,
}

/// The orchestrator session + the worktree the plan works on.
#[derive(Debug, Clone)]
pub struct OwnerContext {
    pub parent_session: SessionId,
    pub workspace_id: u64,
    pub worktree_id: u64,
    /// Real filesystem root of the owner worktree.
    pub root: PathBuf,
}

/// Everything one execution needs that is not part of the plan.
#[derive(Debug, Clone)]
pub struct ExecConfig {
    pub run_id: String,
    pub ceilings: Ceilings,
    /// The parent's own effective capability set.
    pub parent_caps: CapabilitySet,
    pub provider: String,
    /// Model used for children whose spec carries none.
    pub default_model: String,
    /// Root under which isolated child workspaces are created.
    pub isolated_root: PathBuf,
    pub crash_seam: Option<CrashSeam>,
}

/// One finished drive: the child session's real turn op id + the outcome.
#[derive(Debug, Clone)]
struct DriveResult {
    turn_op_id: Option<OpId>,
    result: Result<TurnOutcome, String>,
}

/// In-memory state of one execution. Steering decisions always validate
/// against the DURABLE rows; this is the executor's mirror.
struct ExecState {
    parent_session: SessionId,
    run_id: String,
    plan: crate::TaskPlan,
    owner: OwnerContext,
    config: ExecConfig,
    specs: HashMap<String, ChildSpec>,
    item_states: HashMap<String, WorkState>,
    /// Durable child rows by child id (mirror).
    children: HashMap<String, ChildRuntime>,
    /// Durable work-item → child bindings by item id (wave A3). Seeded
    /// from the compile-time mint; re-attach REPLACES it from the durable
    /// assignment rows (never re-minted in memory).
    assignments: HashMap<String, WorkItemAssignment>,
    /// Scheduler op id of the in-flight drive per child id.
    drive_ops: HashMap<String, OpId>,
    /// Outcomes written by the drive closures, keyed by scheduler op id.
    outcomes: Arc<Mutex<HashMap<OpId, DriveResult>>>,
    next_child_seq: u64,
    crash_fired: bool,
    /// Highest-severity semantic risk any SETTLED child drive reported for
    /// this run (audits 54/119): drives the child-parallelism reduction in
    /// [`Self::admit_ready`]. `None` = no provider was consulted (parity).
    semantic_risk: Option<RiskLevel>,
}

/// The orchestration runtime: the manager + agent it drives children with,
/// and the durable control surface.
///
/// Executions are keyed by RUN ID (audits 7/8/21/22): the runtime mirrors
/// every concurrent execution — one per parent session by the executor's
/// parent-session index — and per-run methods always resolve THEIR OWN
/// mirror entry. Global serialization is gone: concurrency limits live in
/// the per-run scheduler ceilings, the provider limits and the hard live
/// child ceiling, never in a single global execution slot.
pub struct OrchestratorRuntime {
    manager: Arc<SessionManager>,
    agent: Arc<AgentRuntime>,
    /// Live execution mirrors by run id (one entry per installed run; a
    /// terminal run's mirror stays readable until its run is replaced).
    exec: Mutex<HashMap<String, ExecState>>,
}

impl std::fmt::Debug for OrchestratorRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrchestratorRuntime")
            .finish_non_exhaustive()
    }
}

impl OrchestratorRuntime {
    pub fn new(manager: Arc<SessionManager>, agent: Arc<AgentRuntime>) -> Arc<Self> {
        Arc::new(Self {
            manager,
            agent,
            exec: Mutex::new(HashMap::new()),
        })
    }

    pub fn manager(&self) -> Arc<SessionManager> {
        self.manager.clone()
    }

    pub fn agent(&self) -> Arc<AgentRuntime> {
        self.agent.clone()
    }

    /// The durable child rows of one run, oldest child first. Survives any
    /// number of executor crashes and manager reopens.
    pub fn registry_rows(
        manager: Arc<SessionManager>,
        parent: SessionId,
        run_id: &str,
    ) -> faktor_core::Result<Vec<ChildRuntime>> {
        let handle = manager
            .get_session(parent)?
            .ok_or_else(|| faktor_core::Error::not_found(format!("parent session {parent}")))?;
        let mut rows = Vec::new();
        for (kind, key, value) in parent_facts(&handle)? {
            if kind == REGISTRY_ROW_KIND
                && key
                    .strip_prefix(run_id)
                    .is_some_and(|rest| rest.starts_with('/'))
            {
                let mut row: ChildRuntime = serde_json::from_str(&value).map_err(|e| {
                    faktor_core::Error::internal(format!("registry row decode: {e}"))
                })?;
                // Strict durable decode: a corrupt/hostile row (unknown
                // blocker kind, Blocked without a blocker, a non-Blocked row
                // carrying one) fails loudly — never a fabricated state.
                row.validate_durable()
                    .map_err(|e| faktor_core::Error::malformed(e.to_string()))?;
                Self::derive_child_projection(&manager, &mut row);
                rows.push(row);
            }
        }
        // Deterministic order: created_ms first, child id as tie-break (two
        // children spawned in the same millisecond must never order by scan
        // luck).
        rows.sort_by(|a, b| {
            a.created_ms
                .cmp(&b.created_ms)
                .then_with(|| a.child_id.cmp(&b.child_id))
        });
        Ok(rows)
    }

    /// Derive the child projection's mutable truth from its OWN durable
    /// rows: the task row is the ONE budget authority (`budget_max_tokens`
    /// is never trusted from the registry JSON once a task row exists) and
    /// the child's drive-state row is the ONE execution-phase authority.
    /// A missing session/task row leaves the stored value untouched.
    ///
    /// Public because EVERY registry-row reader (the runtime's own
    /// `registry_rows` and the server's native projections) must apply it —
    /// the registry JSON is deliberately not rewritten on control changes.
    pub fn derive_child_projection(manager: &Arc<SessionManager>, row: &mut ChildRuntime) {
        let Ok(Some(handle)) = manager.get_session(SessionId::new(row.session_id)) else {
            return;
        };
        if let Ok(ds) = handle.orchestrator_drive_state_get() {
            row.execution_phase = ds.execution_phase;
        }
        if let Ok(task_id) = handle.task_id() {
            if let Ok(Some(task)) = handle.get_task(task_id) {
                row.budget_max_tokens = task.budget.max_tokens;
            }
        }
    }

    /// The durable presentation projection of one child: the child
    /// session's ledger fold (latest `ChildPresentationChanged` entry wins;
    /// `Foreground` when the child never transitioned). Additive
    /// presentational read — it is NEVER used by scheduling, budgets or
    /// lineage decisions; it only tells UIs whether the child is in the
    /// foreground.
    pub fn child_presentation(
        manager: Arc<SessionManager>,
        row: &ChildRuntime,
    ) -> faktor_core::Result<faktor_session::child::PresentationState> {
        let handle = manager
            .get_session(SessionId::new(row.session_id))?
            .ok_or_else(|| {
                faktor_core::Error::not_found(format!("child session {}", row.session_id))
            })?;
        handle.child_presentation(&row.child_id)
    }

    /// Every durable work-item → child assignment row of one run (kind
    /// [`ASSIGNMENT_ROW_KIND`], key `<run_id>/<item_id>`) under the parent
    /// session. Unparseable values are corruption and decode loudly — never
    /// a silent skip.
    pub fn assignment_rows(
        manager: Arc<SessionManager>,
        parent: SessionId,
        run_id: &str,
    ) -> faktor_core::Result<Vec<WorkItemAssignment>> {
        let handle = manager
            .get_session(parent)?
            .ok_or_else(|| faktor_core::Error::not_found(format!("parent session {parent}")))?;
        let mut rows = Vec::new();
        for (kind, key, value) in parent_facts(&handle)? {
            if kind == ASSIGNMENT_ROW_KIND
                && key
                    .strip_prefix(run_id)
                    .is_some_and(|rest| rest.starts_with('/'))
            {
                let row: WorkItemAssignment = serde_json::from_str(&value).map_err(|e| {
                    faktor_core::Error::internal(format!("assignment row decode: {e}"))
                })?;
                rows.push(row);
            }
        }
        Ok(rows)
    }

    /// Mint the run's item → child bindings in DETERMINISTIC plan order:
    /// every SPAWN work item gets its child id from the shared child
    /// sequence (`child-0`, `child-1`, ...). Auto items (spawn == false)
    /// never spawn a child and consume no id. Callers guarantee a fresh run
    /// (no durable children yet), so the sequence starts at zero; the rows
    /// are persisted atomically by [`Self::put_assignments`] before any
    /// spawn and are the ONLY identity source afterwards.
    ///
    /// Each row records the item's own EXPLICIT ownership, read directly
    /// from [`crate::WorkItem::ownership`] — the only authority there is
    /// (work-entry unification; never a request/plan-level fallback).
    pub(crate) fn compile_assignments(
        run_id: &str,
        plan: &crate::TaskPlan,
        specs: &HashMap<String, ChildSpec>,
    ) -> Vec<WorkItemAssignment> {
        let mut seq: u64 = 0;
        let mut rows = Vec::new();
        for (index, item) in plan.work_items.iter().enumerate() {
            let spawn = specs.get(&item.id).map(|s| s.spawn).unwrap_or(true);
            if !spawn {
                continue;
            }
            rows.push(WorkItemAssignment {
                run_id: run_id.to_string(),
                item_id: item.id.clone(),
                plan_step_index: index,
                child_id: format!("child-{seq}"),
                ownership: item.ownership.clone(),
            });
            seq += 1;
        }
        rows
    }

    /// Shape-check durable assignment rows against the durable plan
    /// (compile-time validation of durable data; shared by re-attach and
    /// the operation graph). Returns every violation (empty = sound):
    ///
    /// - rows must name this run and a KNOWN plan item (an unknown item is
    ///   a tampered/foreign row);
    /// - an assignment for an AUTO item (spawn == false) is stale;
    /// - one item may hold at most ONE assignment row (duplicate
    ///   assignments for one item → typed Conflict);
    /// - two assignments may never bind the same child id;
    /// - `plan_step_index` must equal the item's position in the plan;
    /// - plan child ids must be `child-N` (`review-N` belongs to
    ///   reviewers);
    /// - a durable child row of a spawn item must name EXACTLY the
    ///   assigned child id (no dangling/duplicated plan children).
    ///
    /// MISSING rows for spawn items are deliberately NOT listed here:
    /// callers raise their own typed error (re-attach refuses loudly; the
    /// graph raises [`crate::runtime::graph::GraphError::MissingAssignment`]).
    pub(crate) fn assignment_shape_violations(
        run_id: &str,
        plan: &crate::TaskPlan,
        specs: &HashMap<String, ChildSpec>,
        rows: &[WorkItemAssignment],
        child_rows: &[ChildRuntime],
    ) -> Vec<String> {
        let mut violations = Vec::new();
        // The ownership every row must carry: re-compiled from the durable
        // plan's ITEMS (audits 7/8/21/22). A durable plan that no longer
        // compiles is itself a violation (loud, never silently skipped).
        let effective_by_item = match plan.compile_ownerships() {
            Ok(eff) => eff,
            Err(compile_errs) => {
                violations.push(format!(
                    "durable plan of run {run_id} fails its ownership compile: {}",
                    compile_errs.join("; ")
                ));
                HashMap::new()
            }
        };
        let spawn_items: Vec<&str> = plan
            .work_items
            .iter()
            .filter(|w| specs.get(&w.id).map(|s| s.spawn).unwrap_or(true))
            .map(|w| w.id.as_str())
            .collect();
        let mut by_item: HashMap<&str, usize> = HashMap::new();
        let mut by_child: HashMap<&str, &str> = HashMap::new();
        for row in rows {
            if row.run_id != run_id {
                violations.push(format!(
                    "assignment of item {} names run {} instead of {run_id}",
                    row.item_id, row.run_id
                ));
                continue;
            }
            if !row.child_id.starts_with("child-") {
                violations.push(format!(
                    "assignment of item {} names {child_id}, which is not a plan child id (the review-N ids belong to reviewers)",
                    row.item_id,
                    child_id = row.child_id
                ));
                continue;
            }
            let Some(item) = plan.work_items.iter().find(|w| w.id == row.item_id) else {
                violations.push(format!(
                    "assignment names unknown work item {:?}",
                    row.item_id
                ));
                continue;
            };
            if !spawn_items.contains(&item.id.as_str()) {
                violations.push(format!(
                    "assignment for auto item {:?}, which never spawns a child",
                    item.id
                ));
                continue;
            }
            // (audits 7/8/21/22) The row's recorded ownership must equal the
            // item's EFFECTIVE ownership re-compiled from the durable plan +
            // specs: a row whose ownership disagrees (or a durable plan that
            // no longer compiles) is tampered/stale — spawn never re-derives
            // ownership from memory.
            if let Some(expected) = effective_by_item.get(&item.id) {
                if &row.ownership != expected {
                    violations.push(format!(
                    "assignment of item {} records ownership {own:?} but the durable plan compiles it as {exp:?}",
                    item.id,
                    own = row.ownership,
                    exp = expected
                ));
                }
            } else {
                violations.push(format!(
                    "assignment of item {} records an ownership the durable plan does not compile",
                    item.id
                ));
            }
            let index = plan
                .work_items
                .iter()
                .position(|w| w.id == item.id)
                .expect("item came from the plan");
            if row.plan_step_index != index {
                violations.push(format!(
                    "assignment of item {} records plan step {} but the item sits at step {index}",
                    item.id, row.plan_step_index
                ));
            }
            *by_item.entry(row.item_id.as_str()).or_insert(0) += 1;
            if let Some(other) = by_child.insert(row.child_id.as_str(), &row.item_id) {
                violations.push(format!(
                    "duplicate assignment of child {} to items {} and {}",
                    row.child_id, other, row.item_id
                ));
            }
        }
        for (item, n) in by_item {
            if n > 1 {
                violations.push(format!(
                    "duplicate assignment rows for work item {item:?} ({n} rows)"
                ));
            }
        }
        for row in child_rows {
            if !spawn_items.contains(&row.item_id.as_str()) {
                continue;
            }
            match rows.iter().find(|a| a.item_id == row.item_id) {
                Some(assignment) if assignment.child_id == row.child_id => {}
                Some(assignment) => violations.push(format!(
                    "durable child {} of item {} disagrees with its assignment (assigned {})",
                    row.child_id, row.item_id, assignment.child_id
                )),
                None => violations.push(format!(
                    "durable child {} of item {} has no assignment row",
                    row.child_id, row.item_id
                )),
            }
        }
        violations
    }

    /// Validate the durable assignment rows and return the item → binding
    /// map used by re-attach and every spawn decision. Refuses loudly
    /// (typed Conflict) whenever the durable rows cannot prove a spawn
    /// item's child identity — identity is NEVER minted from memory after
    /// a crash.
    pub(crate) fn checked_assignment_map(
        run_id: &str,
        plan: &crate::TaskPlan,
        specs: &HashMap<String, ChildSpec>,
        rows: &[WorkItemAssignment],
        child_rows: &[ChildRuntime],
    ) -> Result<HashMap<String, WorkItemAssignment>, ExecError> {
        let mut violations =
            Self::assignment_shape_violations(run_id, plan, specs, rows, child_rows);
        let mut map = HashMap::new();
        for (index, item) in plan.work_items.iter().enumerate() {
            if !specs.get(&item.id).map(|s| s.spawn).unwrap_or(true) {
                continue;
            }
            let Some(row) = rows.iter().find(|a| a.item_id == item.id) else {
                violations.push(format!(
                    "work item {:?} (plan step {index}) has no durable assignment row; refusing to fabricate a child id after a crash",
                    item.id
                ));
                continue;
            };
            map.insert(item.id.clone(), row.clone());
        }
        if violations.is_empty() {
            Ok(map)
        } else {
            Err(ExecError::Conflict(format!(
                "assignment rows of run '{run_id}' violate the identity contract: {}",
                violations.join("; ")
            )))
        }
    }

    /// Zero-orphan invariant of the orchestrator registry. Checks:
    /// 1. every registry row's child session row exists and carries the
    ///    same workspace/worktree ids;
    /// 2. every child session's durable identity row names this parent and
    ///    the same worktree + ownership;
    /// 3. no duplicate registry rows;
    /// 4. the reverse direction: no session whose identity names this
    ///    parent exists without an owning registry row of this run.
    ///
    /// Returns every violation (empty = consistent).
    pub fn registry_violations(
        manager: Arc<SessionManager>,
        parent: SessionId,
        run_id: &str,
    ) -> Vec<String> {
        let mut violations = Vec::new();
        let rows = match Self::registry_rows(manager.clone(), parent, run_id) {
            Ok(r) => r,
            Err(e) => {
                violations.push(format!("registry unreadable: {e}"));
                return violations;
            }
        };
        let mut seen = HashSet::new();
        for row in &rows {
            if !seen.insert(row.session_id) {
                violations.push(format!(
                    "duplicate registry row: session {} registered twice",
                    row.session_id
                ));
                continue;
            }
            if row.parent_session_id != parent.raw() || row.run_id != run_id {
                violations.push(format!(
                    "{}: registry row names {}/{} instead of {parent}/{run_id}",
                    row.child_id, row.parent_session_id, row.run_id
                ));
            }
            if row.worktree_id == 0 || row.session_id == 0 {
                violations.push(format!(
                    "{}: registry row without session/worktree id",
                    row.child_id
                ));
                continue;
            }
            let Some(session) = manager
                .get_session(SessionId::new(row.session_id))
                .ok()
                .flatten()
            else {
                violations.push(format!(
                    "{}: child session {} missing",
                    row.child_id, row.session_id
                ));
                continue;
            };
            let Ok(srow) = session.row() else {
                violations.push(format!("{}: session row unreadable", row.child_id));
                continue;
            };
            if srow.workspace_id.raw() != row.workspace_id
                || srow.worktree_id.raw() != row.worktree_id
            {
                violations.push(format!(
                    "{}: session row identity (ws {}, wt {}) disagrees with registry row (ws {}, wt {})",
                    row.child_id,
                    srow.workspace_id,
                    srow.worktree_id,
                    row.workspace_id,
                    row.worktree_id
                ));
            }
            let Ok(Some(identity)) = session.orchestrator_child_identity_get() else {
                violations.push(format!("{}: child identity row missing", row.child_id));
                continue;
            };
            if identity.parent_session_id != parent
                || identity.worktree_id != row.worktree_id
                || identity.ownership != row.ownership
            {
                violations.push(format!(
                    "{}: identity row disagrees with registry row",
                    row.child_id
                ));
            }
        }
        if let Ok(handles) = manager.list_sessions(None) {
            for h in handles {
                let Ok(Some(identity)) = h.orchestrator_child_identity_get() else {
                    continue;
                };
                if identity.parent_session_id != parent {
                    continue;
                }
                if !rows
                    .iter()
                    .any(|r| r.session_id == h.id().raw() && r.run_id == run_id)
                {
                    violations.push(format!(
                        "orphan child session {} (worktree {}) without an owning registry row",
                        h.id(),
                        identity.worktree_id
                    ));
                }
            }
        }
        violations
    }

    /// The GLOBAL orphan-child scan (`doctor --deep`, read-only, P0-97):
    /// unlike [`OrchestratorRuntime::registry_violations`] — which checks one
    /// (parent, run) pair through live handles — this scan reads the raw
    /// durable rows across EVERY session and checks:
    ///
    /// 1. every child identity row names a parent session that still exists;
    /// 2. every registry row names a child session that still exists and is
    ///    keyed by its own `child_id`;
    /// 3. a NON-TERMINAL registry row's worktree row still exists and its
    ///    directory is present on disk (a missing worktree while the child
    ///    is not terminal is a crash-orphaned child).
    ///
    /// Unparseable rows are corruption and are reported, never skipped.
    pub fn orphan_children_scan(manager: Arc<SessionManager>) -> OrphanChildScan {
        let mut scan = OrphanChildScan::default();
        let store = manager.store();
        let session_ids: std::collections::HashSet<u64> = match store.session_ids() {
            Ok(ids) => ids.iter().map(|id| id.raw()).collect(),
            Err(e) => {
                scan.issues.push(format!("session-table scan failed: {e}"));
                return scan;
            }
        };
        let rows = match store.memory_fact_rows_of_kinds(&["orchestrator", REGISTRY_ROW_KIND]) {
            Ok(rows) => rows,
            Err(e) => {
                scan.issues
                    .push(format!("orchestrator fact scan failed: {e}"));
                return scan;
            }
        };
        // worktree_id -> path cache per workspace (lazy: only non-terminal
        // registry rows with a live child need the filesystem).
        let mut wt_paths: std::collections::HashMap<u64, std::collections::HashMap<u64, String>> =
            std::collections::HashMap::new();
        let mut worktree_dir_of = |ws: u64, wt: u64| -> Option<String> {
            if let Some(by_id) = wt_paths.get(&ws) {
                return by_id.get(&wt).cloned();
            }
            let rows = match store.worktrees_of(faktor_core::id::WorkspaceId::new(ws)) {
                Ok(rows) => rows,
                Err(_) => return None,
            };
            let by_id: std::collections::HashMap<u64, String> = rows
                .iter()
                .map(|r| (r.id.max(0) as u64, r.path.clone()))
                .collect();
            let hit = by_id.get(&wt).cloned();
            wt_paths.insert(ws, by_id);
            hit
        };
        // (1) Child identity rows live in the CHILD session's row space and
        // name their parent.
        for fact in rows
            .iter()
            .filter(|f| f.kind == "orchestrator" && f.key == "identity")
        {
            scan.identity_rows += 1;
            let identity: ChildIdentity = match serde_json::from_str(&fact.value) {
                Ok(id) => id,
                Err(e) => {
                    scan.issues.push(format!(
                        "unparseable child identity row under session {} (key {:?}): {e}",
                        fact.session_id, fact.key
                    ));
                    continue;
                }
            };
            if !session_ids.contains(&identity.parent_session_id.raw()) {
                scan.issues.push(format!(
                    "orphan child: child session {} carries an identity row naming parent session {} which has no session row",
                    fact.session_id, identity.parent_session_id
                ));
            }
        }
        // (2)+(3) Registry rows live in the PARENT session's row space.
        for fact in rows.iter().filter(|f| f.kind == REGISTRY_ROW_KIND) {
            scan.registry_rows += 1;
            let key_child = fact
                .key
                .rsplit_once('/')
                .map(|(_, child)| child.to_string())
                .unwrap_or_default();
            let row: ChildRuntime = match serde_json::from_str(&fact.value) {
                Ok(r) => r,
                Err(e) => {
                    scan.issues.push(format!(
                        "unparseable orchestrator registry row under session {} (key {:?}): {e}",
                        fact.session_id, fact.key
                    ));
                    continue;
                }
            };
            if key_child != row.child_id {
                scan.issues.push(format!(
                    "orphan child: registry row under session {} is keyed by {key_child:?} but names child_id {:?}",
                    fact.session_id, row.child_id
                ));
            }
            if !session_ids.contains(&row.session_id) {
                scan.issues.push(format!(
                    "orphan child: registry row {} of session {} references child session {} which has no session row",
                    row.child_id, fact.session_id, row.session_id
                ));
                continue;
            }
            if row.is_terminal() {
                continue;
            }
            match worktree_dir_of(row.workspace_id, row.worktree_id) {
                None => scan.issues.push(format!(
                    "orphan child: non-terminal child {} (session {}) references worktree {}/{} which has no worktree row",
                    row.child_id, row.session_id, row.workspace_id, row.worktree_id
                )),
                Some(path) => {
                    let dir = std::path::Path::new(&path);
                    if !dir.is_dir() {
                        scan.issues.push(format!(
                            "orphan child: non-terminal child {} (session {}) has no worktree directory at {path:?}",
                            row.child_id, row.session_id
                        ));
                    }
                }
            }
        }
        scan
    }

    /// Execute the plan with REAL children (audit 20). A run id that
    /// already has durable registry rows is a Conflict — call
    /// [`OrchestratorRuntime::reattach`] to resume a crashed executor.
    ///
    /// Ownership authority (audits 7/8/21/22, work-entry unification):
    /// before ANY durable row is written the plan is compiled PER ITEM —
    /// the ownership of every item is read from
    /// [`crate::WorkItem::ownership`] (the only authority; legacy
    /// plan-global values convert exactly once at the DTO/durability
    /// boundary, never here), checked against the item's kind, disjointness
    /// is enforced across ALL mutating items (lexically and canonicalized
    /// against the owner root), and read-only items are refused any write
    /// capability — including through their typed policies. Only then are
    /// the plan row and the wave-A3 item→child rows (which carry the
    /// item's ownership) persisted.
    pub async fn execute_task(
        self: &Arc<Self>,
        plan: crate::TaskPlan,
        owner: OwnerContext,
        config: ExecConfig,
        specs: &[ChildSpec],
    ) -> Result<PlanOutcome, ExecError> {
        validate_config(&config)?;
        if self
            .manager
            .get_session(owner.parent_session)?
            .ok_or_else(|| ExecError::NotFound(format!("owner session {}", owner.parent_session)))?
            .orchestrator_child_identity_get()?
            .is_some()
        {
            return Err(ExecError::InvalidState(
                "the owner session is itself an orchestrated child".into(),
            ));
        }
        if !Self::registry_rows(self.manager.clone(), owner.parent_session, &config.run_id)?
            .is_empty()
        {
            return Err(ExecError::Conflict(format!(
                "run '{}' already has durable child rows; call reattach() to resume",
                config.run_id
            )));
        }
        let spec_map = validate_specs(&plan, specs)?;
        // (audits 7/8/21/22) The FULL per-item ownership compile runs BEFORE
        // any durable row: a structurally invalid plan (mixed kinds whose
        // items carry no ownership, overlapping mutating write sets, a
        // read-only item holding write capability) leaves NOTHING behind.
        let effective = plan
            .compile_ownerships()
            .map_err(|errs| ExecError::InvalidPlan(errs.join("; ")))?;
        check_item_policies(&spec_map, &effective)?;
        check_plan_disjointness_canonical(&plan, &effective, &owner)?;
        self.put_plan_row(&plan, &owner, &config, specs)?;
        // (wave A3) The item → child bindings of the WHOLE plan are minted
        // here — before anything spawns — in deterministic plan order and
        // committed in ONE store transaction. Every row carries the item's
        // OWN explicit ownership; spawn (and re-attach) look the ids + the
        // ownership up from these durable rows; nobody re-derives either
        // after a crash.
        let assignments = Self::compile_assignments(&config.run_id, &plan, &spec_map);
        self.put_assignments(owner.parent_session, &assignments)?;
        let run_id = config.run_id.clone();
        let state = self.build_exec_state(plan, owner, config, spec_map);
        self.install_run(state)?;
        // Crash seam: this exact window — assignments durable, no child
        // spawned yet — must re-open with the SAME child ids.
        {
            let mut guard = self.exec.lock().expect("exec lock");
            let exec = guard.get_mut(&run_id).expect("execution installed above");
            self.check_crash(exec, CrashSeam::AfterAssignmentsPersisted)?;
        }
        self.drive_to_outcome(&run_id).await
    }

    /// Re-attach to a crashed execution: children are recovered from the
    /// DURABLE registry rows (never from memory), every non-terminal child
    /// is reconciled against its session row and re-driven from its durable
    /// op record, and pending control rows resume. Idempotent.
    #[allow(clippy::too_many_arguments)]
    pub async fn reattach(
        self: &Arc<Self>,
        parent: SessionId,
        run_id: &str,
        ceilings: Ceilings,
        parent_caps: CapabilitySet,
        default_model: String,
        isolated_root: PathBuf,
        crash_seam: Option<CrashSeam>,
    ) -> Result<PlanOutcome, ExecError> {
        ceilings.validate().map_err(ExecError::InvalidPlan)?;
        let DurablePlanRow {
            plan,
            owner,
            specs,
            provider,
            default_model: durable_model,
            isolated_root: durable_root,
        } = self.plan_row(parent, run_id)?;
        // The durable attachment set is re-validated before any re-drive: a
        // tampered/oversized/hostile stored list refuses the re-attach
        // loudly instead of being submitted.
        for spec in specs.values() {
            validate_attachment_files(&spec.files).map_err(|e| {
                ExecError::InvalidPlan(format!(
                    "durable attached files of work item {}: {e}",
                    spec.item_id
                ))
            })?;
        }
        let _ = isolated_root;
        let config = ExecConfig {
            run_id: run_id.to_string(),
            ceilings,
            parent_caps,
            provider,
            default_model: if default_model.is_empty() {
                durable_model
            } else {
                default_model
            },
            isolated_root: durable_root,
            crash_seam,
        };
        let mut state = self.build_exec_state(plan, owner, config, specs);
        self.reconcile_from_registry(&mut state)?;
        self.install_run(state)?;
        self.drive_to_outcome(run_id).await
    }

    /// Install one execution's mirror under its run id. A re-attach
    /// REPLACES the previous mirror of the same run; concurrent runs of
    /// different sessions each own their entry.
    fn install_run(&self, state: ExecState) -> Result<(), ExecError> {
        let run_id = state.run_id.clone();
        self.exec
            .lock()
            .expect("exec lock")
            .insert(run_id.clone(), state);
        Ok(())
    }

    /// The mirror of the ONE run that owns `child_id` (mirrors are run
    /// scoped; a child belongs to exactly one installed run).
    fn child_mirror(&self, child_id: &str) -> Option<(String, ChildRuntime)> {
        let guard = self.exec.lock().expect("exec lock");
        for (run_id, exec) in guard.iter() {
            if let Some(row) = exec.children.get(child_id) {
                return Some((run_id.clone(), row.clone()));
            }
        }
        None
    }

    // ------------------------------------------------------------ steering

    /// Pause a child: enqueues the durable Pause control; the child's own
    /// drive applies it at its next safe reasoning boundary (never
    /// mid-operation). Non-terminal children only.
    pub fn pause_child(&self, child_id: &str) -> Result<(), ExecError> {
        self.control_child(child_id, ChildControl::Pause)
            .map(|_| ())
    }

    /// Resume a paused/waiting child (durable Resume row).
    pub fn resume_child(&self, child_id: &str) -> Result<(), ExecError> {
        self.control_child(child_id, ChildControl::Resume)
            .map(|_| ())
    }

    /// Cancel a non-terminal child: durable Cancel row + the bounded abort
    /// path on the child's session (the turn ends Cancelled; the session
    /// stays promptable — never a dead session).
    pub fn cancel_child(&self, child_id: &str) -> Result<(), ExecError> {
        self.control_child(child_id, ChildControl::Cancel)
            .map(|_| ())
    }

    /// Steer a child with a guidance note (bounded; durable control row;
    /// applied at the child's next safe reasoning boundary).
    pub fn steer_child(&self, child_id: &str, note: &str) -> Result<(), ExecError> {
        self.control_child(
            child_id,
            ChildControl::Steer {
                note: note.to_string(),
            },
        )
        .map(|_| ())
    }

    /// Change the child's model: takes effect at the child's next provider
    /// selection (durable ChangeModel row applied at the next boundary).
    pub fn change_child_model(&self, child_id: &str, model: &str) -> Result<(), ExecError> {
        self.control_child(
            child_id,
            ChildControl::ChangeModel {
                model: model.to_string(),
            },
        )
        .map(|_| ())
    }

    /// Change the child's durable token budget cap: the Task row is patched
    /// immediately (the wave-9 budget fields gate every genuine turn end)
    /// and the ChangeBudget row is acked — the effect is durable and
    /// idempotent.
    pub fn change_child_budget(&self, child_id: &str, max_tokens: u64) -> Result<(), ExecError> {
        self.control_child(child_id, ChildControl::ChangeBudget { max_tokens })
            .map(|_| ())
    }

    /// Drive one retry of a Failed child (durable Retry row required).
    /// Only Failed children retry; the row is acked when the re-drive is
    /// admitted (never blindly re-run: the agent's submit path runs
    /// session recovery first, and a mid-drive crash resumes the SAME
    /// recorded turn).
    pub fn retry_child(&self, child_id: &str) -> Result<(), ExecError> {
        self.control_child(child_id, ChildControl::Retry)
            .map(|_| ())
    }

    /// Enqueue one control on a child of the ACTIVE execution and report
    /// its exactly-once ack state (audit 23 wire shape). `queued_seq` is
    /// the durable control-row sequence; `applied` is `true` when the
    /// effect was applied synchronously at enqueue (Cancel: bounded abort
    /// fired; ChangeBudget: the wave-9 Task cap patched), `None` while the
    /// control waits for the child's next safe reasoning boundary
    /// (Pause/Resume/Steer/ChangeModel/Retry — the child's own drive acks
    /// the row exactly once when it applies; Retry is acked when the next
    /// re-attach admits the re-drive). Every guard is validated against the
    /// DURABLE child row before anything is written. Children outside the
    /// active execution are `NotFound` — a crashed run must be re-attached
    /// (executor `resume_run`) before its children accept controls.
    pub fn control_child(
        &self,
        child_id: &str,
        control: ChildControl,
    ) -> Result<ControlAck, ExecError> {
        let row = self.durable_child(child_id)?;
        let session = self
            .manager
            .get_session(SessionId::new(row.session_id))?
            .ok_or_else(|| ExecError::NotFound(format!("child session {}", row.session_id)))?;
        let session_terminal = session.state()?.is_terminal();
        match &control {
            ChildControl::Pause | ChildControl::ChangeModel { .. }
                if session_terminal || row.state.is_terminal() =>
            {
                let verb = if matches!(control, ChildControl::Pause) {
                    "pause"
                } else {
                    "change the model of"
                };
                return Err(ExecError::InvalidState(format!(
                    "cannot {verb} child {child_id}: terminal"
                )));
            }
            ChildControl::Cancel if row.state.is_terminal() => {
                return Err(ExecError::InvalidState(format!(
                    "cannot cancel child {child_id}: state {:?} is terminal",
                    row.state
                )));
            }
            ChildControl::Resume
                if !matches!(
                    row.state,
                    ChildState::Paused
                        | ChildState::Waiting
                        | ChildState::Blocked
                        | ChildState::Running
                ) =>
            {
                return Err(ExecError::InvalidState(format!(
                    "cannot resume child {child_id}: state {:?}",
                    row.state
                )));
            }
            ChildControl::Retry if row.state != ChildState::Failed => {
                return Err(ExecError::InvalidState(format!(
                    "cannot retry child {child_id}: only Failed children retry (state {:?})",
                    row.state
                )));
            }
            ChildControl::ChangeModel { model }
                if model.is_empty() || model.chars().count() > 128 =>
            {
                return Err(ExecError::Oversized(
                    "model selector must be 1..=128 characters".into(),
                ));
            }
            ChildControl::Steer { note } if note.chars().count() > 500 => {
                return Err(ExecError::Oversized(
                    "steering note must be 1..=500 characters".into(),
                ));
            }
            // Terminal children refuse steering BEFORE anything is written
            // (no durable queue row for a child that can never apply it).
            ChildControl::Steer { .. } if row.state.is_terminal() || session_terminal => {
                return Err(ExecError::InvalidState(format!(
                    "cannot steer child {child_id}: state {:?} is terminal",
                    row.state
                )));
            }
            // Whitespace-only guidance is malformed at the RUNTIME boundary
            // (the HTTP layer is never the only guard).
            ChildControl::Steer { note } if note.trim().is_empty() => {
                return Err(ExecError::Malformed(
                    "steering note must not be empty or whitespace-only".into(),
                ));
            }
            ChildControl::ChangeBudget { .. } if row.state.is_terminal() => {
                return Err(ExecError::InvalidState(format!(
                    "cannot change the budget of {child_id}: state {:?} is terminal",
                    row.state
                )));
            }
            _ => {}
        }
        // Synchronous effects first: Cancel and ChangeBudget apply at
        // enqueue time and ack their row immediately (a crash between the
        // effect and the ack re-applies the idempotent effect on re-attach;
        // a crash after the ack is harmless — the durable effect rows are
        // the truth the re-attached drive re-reads).
        match control {
            ChildControl::Cancel => {
                // Every authoritative step propagates: an enqueue/ack failure
                // means the control was NOT claimed applied, and the call is
                // safe to retry (the ack is idempotent).
                let msg = session
                    .orchestrator_ctl_enqueue(ChildControl::Cancel)
                    .map_err(|e| ExecError::retriable("cancel control enqueue", e))?;
                session
                    .orchestrator_ctl_ack(msg.seq)
                    .map_err(|e| ExecError::retriable("cancel control ack", e))?;
                // The bounded abort path (existing semantics): fires the
                // turn cancellation token; an abort on a session with no
                // registered op is a no-op that leaves the session
                // promptable. A failed abort is loud (the control row is
                // already acked, so the caller retries the whole call).
                if let Ok(Some(record)) = session.active_turn_record() {
                    session
                        .abort(Some(record.turn_op_id))
                        .map_err(|e| ExecError::retriable("cancel abort", e))?;
                }
                Ok(ControlAck {
                    queued_seq: msg.seq,
                    applied: Some(true),
                })
            }
            ChildControl::ChangeBudget { max_tokens } => {
                // The CHILD TASK ROW is the single budget authority: patch it
                // durably first, then deliver the control row. The child
                // projection is DERIVED from the task row (never written
                // independently), so there is no second budget truth to
                // diverge across a crash/reopen.
                self.agent
                    .seed_task_budget(
                        SessionId::new(row.session_id),
                        &TaskBudget {
                            max_tokens: Some(max_tokens),
                            max_turns: None,
                            spent_tokens: 0,
                            spent_turns: 0,
                        },
                    )
                    .map_err(|e| ExecError::retriable("task budget patch", e))?;
                let msg = session
                    .orchestrator_ctl_enqueue(ChildControl::ChangeBudget { max_tokens })
                    .map_err(|e| ExecError::retriable("budget control enqueue", e))?;
                session
                    .orchestrator_ctl_ack(msg.seq)
                    .map_err(|e| ExecError::retriable("budget control ack", e))?;
                // Project the task row's cap into the live mirror; the
                // durable registry JSON is deliberately NOT rewritten (the
                // task row is the truth and every registry read derives it).
                let derived = child_task_budget_cap(&self.manager, row.session_id);
                let owner_run = {
                    let guard = self.exec.lock().expect("exec lock");
                    guard
                        .iter()
                        .find_map(|(r, e)| e.children.contains_key(child_id).then(|| r.clone()))
                };
                if let Some(run_id) = owner_run {
                    let mut guard = self.exec.lock().expect("exec lock");
                    if let Some(exec) = guard.get_mut(&run_id) {
                        if let Some(c) = exec.children.get_mut(child_id) {
                            c.budget_max_tokens = derived;
                            // The phase projection also follows the child's
                            // own durable rows.
                            c.execution_phase =
                                child_execution_phase(&self.manager, row.session_id);
                        }
                    }
                }
                Ok(ControlAck {
                    queued_seq: msg.seq,
                    applied: Some(true),
                })
            }
            other => {
                let msg = session
                    .orchestrator_ctl_enqueue(other)
                    .map_err(|e| ExecError::retriable("control enqueue", e))?;
                Ok(ControlAck {
                    queued_seq: msg.seq,
                    applied: None,
                })
            }
        }
    }

    /// The live mirror of one child (durable rows are the source of truth;
    /// this refreshes the mirror from the registry).
    pub fn child(&self, child_id: &str) -> Result<Option<ChildRuntime>, ExecError> {
        Ok(self.child_mirror(child_id).map(|(_, row)| row))
    }

    fn durable_child(&self, child_id: &str) -> Result<ChildRuntime, ExecError> {
        self.child_mirror(child_id)
            .map(|(_, row)| row)
            .ok_or_else(|| ExecError::NotFound(format!("unknown child {child_id}")))
    }

    // ------------------------------------------------------------ internals

    fn build_exec_state(
        &self,
        plan: crate::TaskPlan,
        owner: OwnerContext,
        config: ExecConfig,
        specs: HashMap<String, ChildSpec>,
    ) -> ExecState {
        let item_states = plan
            .work_items
            .iter()
            .map(|w| (w.id.clone(), w.completion))
            .collect();
        // (wave A3) The mirror's binding set: compile-minted for fresh runs
        // (identical plan order + explicit item ownership = identical rows),
        // replaced by the DURABLE rows in reconcile_from_registry after a
        // crash. A durable plan that no longer compiles seeds nothing — the
        // durable assignment rows (and their shape checks) then refuse the
        // run loudly at re-attach.
        let assignments: HashMap<String, WorkItemAssignment> =
            Self::compile_assignments(&config.run_id, &plan, &specs)
                .into_iter()
                .map(|a| (a.item_id.clone(), a))
                .collect();
        let next_child_seq = assignments.len() as u64;
        ExecState {
            parent_session: owner.parent_session,
            run_id: config.run_id.clone(),
            plan,
            owner,
            config,
            specs,
            item_states,
            children: HashMap::new(),
            assignments,
            drive_ops: HashMap::new(),
            outcomes: Arc::new(Mutex::new(HashMap::new())),
            next_child_seq,
            crash_fired: false,
            semantic_risk: None,
        }
    }

    fn put_plan_row(
        &self,
        plan: &crate::TaskPlan,
        owner: &OwnerContext,
        config: &ExecConfig,
        specs: &[ChildSpec],
    ) -> Result<(), ExecError> {
        let handle = self
            .manager
            .get_session(owner.parent_session)?
            .ok_or_else(|| {
                ExecError::NotFound(format!("owner session {}", owner.parent_session))
            })?;
        let value = serde_json::json!({
            "plan": plan,
            "owner_ws": owner.workspace_id,
            "owner_wt": owner.worktree_id,
            "owner_root": owner.root.to_string_lossy(),
            "specs": specs,
            "provider": config.provider,
            "default_model": config.default_model,
            "isolated_root": config.isolated_root.to_string_lossy(),
            "created_ms": handle.now_ms(),
        });
        let text = serde_json::to_string(&value)
            .map_err(|e| ExecError::Internal(format!("plan row serialization: {e}")))?;
        handle
            .upsert_memory_fact(PLAN_ROW_KIND, &config.run_id, &text)
            .map_err(|e| ExecError::Internal(format!("plan row write: {e}")))?;
        Ok(())
    }

    /// Persist the run's WHOLE assignment row set in ONE store transaction
    /// (atomic: a crash mid-write can never leave a partial identity set
    /// behind a run that then spawns children). Must be called before any
    /// child spawn of the run.
    fn put_assignments(
        &self,
        parent: SessionId,
        rows: &[WorkItemAssignment],
    ) -> Result<(), ExecError> {
        if rows.is_empty() {
            return Ok(());
        }
        let handle = self
            .manager
            .get_session(parent)?
            .ok_or_else(|| ExecError::NotFound(format!("owner session {parent}")))?;
        let mut keys = Vec::with_capacity(rows.len());
        let mut values = Vec::with_capacity(rows.len());
        for row in rows {
            let key = format!("{}/{}", row.run_id, row.item_id);
            let value = serde_json::to_string(row)
                .map_err(|e| ExecError::Internal(format!("assignment row serialization: {e}")))?;
            if value.len() > 4096 {
                return Err(ExecError::Oversized(format!(
                    "assignment row of item {} exceeds the 4096-byte memory-fact bound",
                    row.item_id
                )));
            }
            keys.push(key);
            values.push(value);
        }
        let facts: Vec<(&str, &str, &str)> = keys
            .iter()
            .zip(&values)
            .map(|(k, v)| (ASSIGNMENT_ROW_KIND, k.as_str(), v.as_str()))
            .collect();
        self.manager
            .store()
            .upsert_memory_facts(handle.id(), &facts)
            .map_err(|e| ExecError::Internal(format!("assignment rows write: {e}")))?;
        Ok(())
    }

    fn plan_row(&self, parent: SessionId, run_id: &str) -> Result<DurablePlanRow, ExecError> {
        let handle = self
            .manager
            .get_session(parent)?
            .ok_or_else(|| ExecError::NotFound(format!("owner session {parent}")))?;
        for (kind, key, value) in
            parent_facts(&handle).map_err(|e| ExecError::Internal(format!("plan row read: {e}")))?
        {
            if kind == PLAN_ROW_KIND && key == run_id {
                let v: serde_json::Value = serde_json::from_str(&value)
                    .map_err(|e| ExecError::Internal(format!("plan row decode: {e}")))?;
                // ONE-TIME compatibility conversion at the durability
                // boundary: old plan rows still carry the plan-global
                // `ownership` field; `from_legacy_value` adopts it onto
                // mutating items exactly once (new rows pass through).
                let plan: crate::TaskPlan =
                    crate::TaskPlan::from_legacy_value(v.get("plan").cloned().unwrap_or_default())
                        .map_err(|e| ExecError::Internal(format!("plan row plan decode: {e}")))?;
                let specs: Vec<ChildSpec> =
                    serde_json::from_value(v.get("specs").cloned().unwrap_or_default())
                        .map_err(|e| ExecError::Internal(format!("plan row specs decode: {e}")))?;
                let spec_map: HashMap<String, ChildSpec> =
                    specs.into_iter().map(|s| (s.item_id.clone(), s)).collect();
                let owner = OwnerContext {
                    parent_session: parent,
                    workspace_id: v.get("owner_ws").and_then(|x| x.as_u64()).unwrap_or(0),
                    worktree_id: v.get("owner_wt").and_then(|x| x.as_u64()).unwrap_or(0),
                    root: v
                        .get("owner_root")
                        .and_then(|x| x.as_str())
                        .map(PathBuf::from)
                        .unwrap_or_default(),
                };
                let provider = v
                    .get("provider")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string();
                let default_model = v
                    .get("default_model")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string();
                let isolated_root = v
                    .get("isolated_root")
                    .and_then(|x| x.as_str())
                    .map(PathBuf::from)
                    .unwrap_or_default();
                return Ok(DurablePlanRow {
                    plan,
                    owner,
                    specs: spec_map,
                    provider,
                    default_model,
                    isolated_root,
                });
            }
        }
        Err(ExecError::NotFound(format!(
            "run '{run_id}' has no durable plan row under session {parent}"
        )))
    }

    fn persist_row(&self, exec: &ExecState, row: &ChildRuntime) -> Result<(), ExecError> {
        let handle = self
            .manager
            .get_session(exec.parent_session)?
            .ok_or_else(|| {
                ExecError::NotFound(format!("parent session {}", exec.parent_session))
            })?;
        let value = serde_json::to_string(row)
            .map_err(|e| ExecError::Internal(format!("registry row serialization: {e}")))?;
        handle
            .upsert_memory_fact(
                REGISTRY_ROW_KIND,
                &format!("{}/{}", exec.run_id, row.child_id),
                &value,
            )
            .map_err(|e| ExecError::Internal(format!("registry row write: {e}")))?;
        Ok(())
    }

    fn check_crash(&self, exec: &mut ExecState, seam: CrashSeam) -> Result<(), ExecError> {
        if exec.crash_fired || exec.config.crash_seam != Some(seam) {
            return Ok(());
        }
        exec.crash_fired = true;
        Err(ExecError::InjectedCrashSeam(format!("{seam:?}")))
    }

    /// Reconcile mirrors + durable rows from the registry and session rows
    /// after a crash (re-attach).
    fn reconcile_from_registry(&self, state: &mut ExecState) -> Result<(), ExecError> {
        let violations =
            Self::registry_violations(self.manager.clone(), state.parent_session, &state.run_id);
        if !violations.is_empty() {
            return Err(ExecError::Internal(format!(
                "zero-orphan registry violation at re-attach: {}",
                violations.join("; ")
            )));
        }
        let rows = Self::registry_rows(self.manager.clone(), state.parent_session, &state.run_id)?;
        // (wave A3) The DURABLE assignment rows are the identity contract
        // of a re-attach: the mirror must agree with them before any child
        // row is trusted or any item is re-spawned. Spawn NEVER re-mints
        // after a crash — every item spawns under the id its durable
        // assignment names (which may be a run that crashed between the
        // assignment transaction and its first spawn: zero child rows is
        // legal here).
        let durable =
            Self::assignment_rows(self.manager.clone(), state.parent_session, &state.run_id)
                .map_err(|e| ExecError::Internal(format!("assignment rows read: {}", e.message)))?;
        state.assignments = Self::checked_assignment_map(
            &state.run_id,
            &state.plan,
            &state.specs,
            &durable,
            &rows,
        )?;
        let mut children = HashMap::new();
        let mut item_states: HashMap<String, WorkState> = state
            .plan
            .work_items
            .iter()
            .map(|w| (w.id.clone(), WorkState::Pending))
            .collect();
        let rows_snapshot = rows.clone();
        for mut row in rows {
            let (reconciled, blocker) = self.reconcile_child_state(&row)?;
            row.state = reconciled;
            // Dependency truth: a non-terminal child whose item still has
            // unfinished dependencies is BLOCKED naming the dependency —
            // never silently re-driven as if it were runnable.
            let blocker = if reconciled.is_terminal() {
                None
            } else if let Some(dep_block) = unmet_dependency(&state.plan, &rows_snapshot, &row) {
                if reconciled == ChildState::Running || blocker.is_none() {
                    row.state = ChildState::Blocked;
                    Some(dep_block)
                } else {
                    blocker
                }
            } else {
                blocker
            };
            match &blocker {
                Some(b) => {
                    row.set_blocker(b)?;
                    self.record_blocker_audit(&row, b);
                }
                None if row.state != ChildState::Blocked => {
                    if let Some(previous) = row.clear_blocker() {
                        self.record_blocker_resolved_audit(&row, &previous);
                    }
                }
                None => {}
            }
            self.persist_row(state, &row)
                .map_err(|e| ExecError::retriable("child registry persist at re-attach", e))?;
            self.persist_child_runtime_row(&row)?;
            let item = row.item_id.clone();
            if !item_states.contains_key(&item) {
                // Reviewer rows (plan-less children) own no plan-item
                // state, but stay in the mirror so their drives resume.
                children.insert(row.child_id.clone(), row);
                continue;
            }
            // Pending -> Running first (a terminal child only got there
            // through a Running item; re-attach must never re-spawn an item
            // that already has a durable child).
            if item_states[&item] == WorkState::Pending
                && can_advance(WorkState::Pending, WorkState::Running)
            {
                item_states.insert(item.clone(), WorkState::Running);
            }
            let target = project_child_state(&row);
            if can_advance(item_states[&item], target) {
                item_states.insert(item, target);
            }
            children.insert(row.child_id.clone(), row);
        }
        // Pending dependents of failed/cancelled items are blocked.
        for w in &state.plan.work_items {
            if item_states[&w.id] == WorkState::Pending
                && w.depends_on.iter().any(|d| {
                    matches!(
                        item_states.get(d),
                        Some(WorkState::Failed | WorkState::Cancelled)
                    )
                })
            {
                item_states.insert(w.id.clone(), WorkState::Blocked);
            }
        }
        // The shared child sequence continues above every durable id —
        // plan child ids (registry + reserved-but-unspawned assignments)
        // AND reviewer ids — so reviewers can never be minted below a
        // reserved plan id.
        let max_seq = children
            .keys()
            .map(String::as_str)
            .chain(state.assignments.values().map(|a| a.child_id.as_str()))
            .filter_map(child_seq_of)
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        state.children = children;
        state.item_states = item_states;
        state.next_child_seq = max_seq;
        state.drive_ops = HashMap::new();
        state.crash_fired = false;
        Ok(())
    }

    /// Classify one child row against its session row + drive state
    /// (re-attach, crash windows included). Returns the child state plus
    /// the typed blocker when the row is (or becomes) Blocked.
    fn reconcile_child_state(
        &self,
        row: &ChildRuntime,
    ) -> Result<(ChildState, Option<ChildBlocker>), ExecError> {
        if row.state.is_terminal() {
            return Ok((row.state, None));
        }
        let session = self
            .manager
            .get_session(SessionId::new(row.session_id))?
            .ok_or_else(|| ExecError::NotFound(format!("child session {}", row.session_id)))?;
        let ds = session.orchestrator_drive_state_get()?;
        if ds.phase == ChildPhase::Waiting {
            return Ok((ChildState::Waiting, None));
        }
        match session.state()? {
            AgentState::Completed => Ok((ChildState::Done, None)),
            AgentState::ReadyForNextTurn => {
                // A genuine end happened; close a turn record left active by
                // a crash between TurnCompleted and finish_turn_record.
                if let Ok(Some(record)) = session.active_turn_record() {
                    session
                        .finish_turn_record(record.turn_op_id, "completed")
                        .map_err(|e| ExecError::retriable("turn record close", e))?;
                }
                Ok((ChildState::Done, None))
            }
            AgentState::Cancelled => Ok((ChildState::Cancelled, None)),
            AgentState::FailedRecoverable | AgentState::FailedPermanent => {
                if let Ok(Some(record)) = session.active_turn_record() {
                    session
                        .finish_turn_record(record.turn_op_id, "failed")
                        .map_err(|e| ExecError::retriable("turn record close", e))?;
                }
                Ok((ChildState::Failed, None))
            }
            // A permission decision is pending: BLOCKED, never Failed and
            // never silently Running.
            AgentState::NeedsUserInput => Ok((
                ChildState::Blocked,
                Some(ChildBlocker::new(
                    BlockerKind::Permission,
                    "waiting for a pending permission decision",
                    "resolve the pending permission request, then resume the child",
                )),
            )),
            // Mid-turn: driveable from the durable op record.
            _ => Ok((ChildState::Running, None)),
        }
    }

    /// The supervision loop of ONE run: settle finished drives, admit new
    /// waves under the ceilings, drive each wave through the scheduler
    /// (paused children park inside their drive and hold the wave until
    /// resumed). Every mirror access is scoped to `run_id` — concurrent
    /// executions of other sessions never share a mirror entry.
    async fn drive_to_outcome(&self, run_id: &str) -> Result<PlanOutcome, ExecError> {
        let mut limits = faktor_core::resource::ResourceLimits::default();
        let scheduler = {
            let guard = self.exec.lock().expect("exec lock");
            let exec = guard
                .get(run_id)
                .ok_or_else(|| ExecError::NotFound(format!("run {run_id} is not installed")))?;
            limits.limits.insert(
                faktor_core::resource::ResourceClass::Cpu,
                exec.config.ceilings.max_reasoning_active,
            );
            limits.limits.insert(
                faktor_core::resource::ResourceClass::DiskWrite,
                exec.config.ceilings.max_mutating_active,
            );
            Scheduler::new(exec.parent_session, Arc::new(SystemClock)).with_limits(limits)
        };
        loop {
            self.settle_finished_drives(run_id)?;
            let admitted = self.admit_ready(run_id, &scheduler)?;
            if admitted == 0 {
                return self.final_outcome(run_id);
            }
            scheduler
                .run_to_completion()
                .await
                .map_err(|e| ExecError::Internal(format!("scheduler wave failed: {e}")))?;
        }
    }

    /// Classify finished drives of ONE run, write durable child rows,
    /// advance items.
    fn settle_finished_drives(&self, run_id: &str) -> Result<(), ExecError> {
        let mut guard = self.exec.lock().expect("exec lock");
        let exec = guard
            .get_mut(run_id)
            .ok_or_else(|| ExecError::NotFound(format!("run {run_id} is not installed")))?;
        let mut outcomes = exec.outcomes.lock().expect("outcome lock");
        let mut done: Vec<(String, ChildRuntime)> = Vec::new();
        for (child_id, op_id) in exec.drive_ops.clone() {
            if let Some(drive) = outcomes.remove(&op_id) {
                let mut row = exec.children.get(&child_id).cloned().ok_or_else(|| {
                    ExecError::NotFound(format!("child {child_id} missing from mirror"))
                })?;
                // The finished drive is authoritative for this child: a
                // retried Failed child that now ends Done/Cancelled flips
                // its durable state; a terminal child is never re-driven
                // otherwise (steering gates enforce that).
                // Semantic risk after the child (audits 54/119): fold the
                // drive's conservative risk into the run so the NEXT
                // admission wave runs with reduced child parallelism.
                if let Some(level) = drive.result.as_ref().ok().and_then(|o| o.semantic_risk) {
                    let escalates = exec
                        .semantic_risk
                        .map(|current| risk_severity(level) > risk_severity(current))
                        .unwrap_or(true);
                    if escalates {
                        exec.semantic_risk = Some(level);
                    }
                    tracing::debug!(
                        child = %child_id,
                        risk = ?level,
                        "settled child reported semantic risk; subsequent child admissions are reduced"
                    );
                }
                let (outcome_state, blocker) = classify_outcome(drive.result);
                if let Some(b) = &blocker {
                    // A blocked drive is NOT a failure and NOT done: the
                    // durable blocker truth (kind + reason + resolution)
                    // rides the row and the typed store projection.
                    row.set_blocker(b)?;
                    row.updated_ms = self.manager.now_ms();
                    self.record_blocker_audit(&row, b);
                } else {
                    if row.state != outcome_state {
                        row.state = outcome_state;
                        row.updated_ms = self.manager.now_ms();
                    }
                    // Any non-Blocked transition clears the durable blocker
                    // truth (Running resumes, terminal ends).
                    if row.state != ChildState::Blocked {
                        if let Some(previous) = row.clear_blocker() {
                            self.record_blocker_resolved_audit(&row, &previous);
                        }
                    }
                }
                if let Some(turn_op) = drive.turn_op_id {
                    row.operation_id = turn_op.raw();
                }
                // Settlement is the integration boundary of a finished drive:
                // persist the phase into the child's own durable row (best
                // effort — a phase write never gates the settlement) and fold
                // it into the projection the run returns.
                self.note_child_phase(&row, ExecutionPhase::Integrating);
                row.execution_phase = ExecutionPhase::Integrating;
                self.persist_row(exec, &row)
                    .map_err(|e| ExecError::retriable("child registry persist at settlement", e))?;
                self.persist_child_runtime_row(&row)?;
                exec.children.insert(child_id.clone(), row.clone());
                exec.drive_ops.remove(&child_id);
                done.push((child_id, row));
            }
        }
        drop(outcomes);
        drop(guard);
        for (child_id, row) in done {
            self.advance_item(run_id, &child_id, &row)?;
        }
        // Crash seam: a child just reached terminal state.
        {
            let terminal_child = {
                let guard = self.exec.lock().expect("exec lock");
                let exec = guard
                    .get(run_id)
                    .ok_or_else(|| ExecError::NotFound(format!("run {run_id} is not installed")))?;
                if !exec.crash_fired
                    && exec.config.crash_seam == Some(CrashSeam::AfterChildTerminal)
                {
                    exec.children
                        .values()
                        .find(|c| c.state.is_terminal())
                        .map(|c| c.child_id.clone())
                } else {
                    None
                }
            };
            if let Some(child_id) = terminal_child {
                let mut guard = self.exec.lock().expect("exec lock");
                let exec = guard
                    .get_mut(run_id)
                    .ok_or_else(|| ExecError::NotFound(format!("run {run_id} is not installed")))?;
                exec.crash_fired = true;
                return Err(ExecError::InjectedCrashSeam(format!(
                    "AfterChildTerminal (child {child_id})"
                )));
            }
        }
        Ok(())
    }

    fn advance_item(
        &self,
        run_id: &str,
        _child_id: &str,
        row: &ChildRuntime,
    ) -> Result<(), ExecError> {
        let mut guard = self.exec.lock().expect("exec lock");
        let exec = guard
            .get_mut(run_id)
            .ok_or_else(|| ExecError::NotFound(format!("run {run_id} is not installed")))?;
        // The ONE canonical projection decides the item state — a Blocked
        // child advances its item to Blocked (never silently Running).
        let target = project_child_state(row);
        if target == WorkState::Running {
            return Ok(());
        }
        let item = row.item_id.clone();
        let cur = *exec.item_states.get(&item).unwrap_or(&WorkState::Pending);
        if cur == target {
            return Ok(());
        }
        if can_advance(cur, target) {
            exec.item_states.insert(item.clone(), target);
        } else if cur == WorkState::Failed
            && matches!(
                target,
                WorkState::Done | WorkState::Cancelled | WorkState::Blocked
            )
        {
            // A retried child completed (or was blocked by a fresh
            // condition): legal chain Failed -> Pending -> Running -> target.
            exec.item_states.insert(item.clone(), WorkState::Pending);
            exec.item_states.insert(item.clone(), WorkState::Running);
            exec.item_states.insert(item.clone(), target);
        } else {
            return Ok(());
        }
        match target {
            WorkState::Failed | WorkState::Cancelled => {
                let mut dependency_blocks: Vec<(String, ChildBlocker)> = Vec::new();
                for w in &exec.plan.work_items {
                    if w.depends_on.contains(&item)
                        && exec.item_states.get(&w.id) == Some(&WorkState::Pending)
                    {
                        exec.item_states.insert(w.id.clone(), WorkState::Blocked);
                        // Any LIVE child of the dependent item is blocked
                        // too, naming the failed dependency.
                        for c in exec.children.values() {
                            if c.item_id == w.id && !c.is_terminal() {
                                dependency_blocks.push((
                                    c.child_id.clone(),
                                    ChildBlocker::dependency(
                                        &item,
                                        format!("waiting on work item {item:?}"),
                                        "wait for the dependency to complete, then resume the child",
                                    ),
                                ));
                            }
                        }
                    }
                }
                for (child_id, blocker) in dependency_blocks {
                    if let Some(c) = exec.children.get_mut(&child_id) {
                        c.set_blocker(&blocker)?;
                        c.updated_ms = self.manager.now_ms();
                        let blocked = c.clone();
                        self.persist_row(exec, &blocked).map_err(|e| {
                            ExecError::retriable("dependency-block registry persist", e)
                        })?;
                        self.persist_child_runtime_row(&blocked)?;
                        self.record_blocker_audit(&blocked, &blocker);
                    }
                }
            }
            WorkState::Done => {
                // A retried item unblocks its dependents when every
                // dependency is Done again.
                for w in &exec.plan.work_items {
                    if w.depends_on.contains(&item)
                        && exec.item_states.get(&w.id) == Some(&WorkState::Blocked)
                        && w.depends_on
                            .iter()
                            .all(|d| exec.item_states.get(d) == Some(&WorkState::Done))
                    {
                        exec.item_states.insert(w.id.clone(), WorkState::Pending);
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Admit ready work of ONE run under the ceilings. Returns the number
    /// of drives registered with the scheduler.
    fn admit_ready(&self, run_id: &str, scheduler: &Scheduler) -> Result<usize, ExecError> {
        let mut admitted = 0usize;
        let mut newly_spawned: Vec<ChildRuntime> = Vec::new();
        {
            let mut guard = self.exec.lock().expect("exec lock");
            let exec = guard
                .get_mut(run_id)
                .ok_or_else(|| ExecError::NotFound(format!("run {run_id} is not installed")))?;
            // (a) Auto items (spawn == false) complete without a child.
            for item in ready_items(&exec.plan, &exec.item_states) {
                let spawn = exec.specs.get(&item).map(|s| s.spawn).unwrap_or(true);
                if !spawn {
                    if can_advance(exec.item_states[&item], WorkState::Running) {
                        exec.item_states.insert(item.clone(), WorkState::Running);
                    }
                    if can_advance(exec.item_states[&item], WorkState::Done) {
                        exec.item_states.insert(item.clone(), WorkState::Done);
                    }
                }
            }
            let ready: Vec<WorkItem> = ready_items(&exec.plan, &exec.item_states)
                .into_iter()
                .filter(|id| exec.specs.get(id).map(|s| s.spawn).unwrap_or(true))
                .map(|id| {
                    exec.plan
                        .work_items
                        .iter()
                        .find(|w| w.id == id)
                        .cloned()
                        .unwrap()
                })
                .collect();
            // Semantic risk reduction (audit 54): a run whose settled
            // children reported High/Unknown risk admits fewer concurrent
            // children from here on. None = configured ceilings unchanged.
            let ceilings = risk_adjusted_ceilings(&exec.config.ceilings, exec.semantic_risk);
            for item in ready {
                // Live ceiling: a hard, typed reject BEFORE each spawn (a
                // child already registered in THIS pass counts).
                let live = exec.children.values().filter(|c| !c.is_terminal()).count();
                if live >= ceilings.max_live {
                    return Err(ExecError::CeilingExceeded {
                        class: "live",
                        limit: ceilings.max_live,
                        used: live,
                    });
                }
                let class = if item.kind.is_mutating() {
                    "mutating"
                } else {
                    "reasoning"
                };
                let ceiling = if class == "mutating" {
                    ceilings.max_mutating_active
                } else {
                    ceilings.max_reasoning_active
                };
                let used = exec
                    .children
                    .values()
                    .filter(|c| !c.is_terminal() && c.is_mutating() == item.kind.is_mutating())
                    .count();
                if used >= ceiling {
                    continue; // defer this class to the next wave
                }
                let child = self.spawn_child(exec, &item)?;
                exec.children.insert(child.child_id.clone(), child.clone());
                exec.item_states.insert(item.id.clone(), WorkState::Running);
                newly_spawned.push(child);
                admitted += 1;
            }
        }
        // Crash seam BeforeDrive fires per spawned child BEFORE its drive is
        // submitted (the child session + registry rows already exist).
        for child in &newly_spawned {
            self.check_crash_seam_before_drive(run_id)?;
            self.submit_drive_op(run_id, scheduler, child)?;
        }
        // (b) Waiting children with a pending Resume row and Failed children
        // with a pending Retry row are re-driven (the retry row is acked at
        // admission: the decision is durable before the drive starts).
        let mut redrives = Vec::new();
        {
            let mut guard = self.exec.lock().expect("exec lock");
            let exec = guard
                .get_mut(run_id)
                .ok_or_else(|| ExecError::NotFound(format!("run {run_id} is not installed")))?;
            for child in exec.children.values() {
                if exec.drive_ops.contains_key(&child.child_id) {
                    continue;
                }
                if child.is_terminal() && child.state != ChildState::Failed {
                    continue;
                }
                let Ok(Some(session)) = self.manager.get_session(SessionId::new(child.session_id))
                else {
                    continue;
                };
                let Ok(pending) = session.orchestrator_ctl_pending() else {
                    continue;
                };
                let mut should_drive = false;
                match child.state {
                    ChildState::Waiting => {
                        should_drive = pending
                            .iter()
                            .any(|r| matches!(r.control, ChildControl::Resume));
                    }
                    ChildState::Failed => {
                        for r in &pending {
                            if matches!(r.control, ChildControl::Retry) {
                                // The retry decision is durable BEFORE the
                                // drive starts: an ack failure refuses the
                                // admission (safe to retry the same call).
                                session
                                    .orchestrator_ctl_ack(r.seq)
                                    .map_err(|e| ExecError::retriable("retry control ack", e))?;
                                should_drive = true;
                                break;
                            }
                        }
                    }
                    // Blocked children only re-drive when their block has
                    // cleared: a dependency blocker whose dependencies are
                    // all Done again, or an explicit durable Resume. A
                    // budget/permission block waits for the operator — an
                    // automatic re-drive would spin on the same refusal.
                    ChildState::Blocked => {
                        let dependency_recovered = child
                            .blocker()
                            .is_some_and(|b| b.kind == BlockerKind::Dependency)
                            && dependencies_done(&exec.plan, &exec.item_states, &child.item_id);
                        let resume_requested = pending
                            .iter()
                            .any(|r| matches!(r.control, ChildControl::Resume));
                        should_drive = dependency_recovered || resume_requested;
                    }
                    // Running without an in-flight drive: an executor
                    // crashed after the child was created (or re-attach
                    // after a mid-drive kill). Re-drive it — the drive
                    // entry continues the SAME recorded turn when one is
                    // active and submits fresh otherwise.
                    ChildState::Running => should_drive = true,
                    _ => {}
                }
                if should_drive {
                    let mut row = child.clone();
                    if row.state != ChildState::Running && !row.is_terminal() {
                        row.state = ChildState::Running;
                        // The transition back to Running CLEARS the durable
                        // blocker truth (audited in the typed ledger).
                        if let Some(previous) = row.clear_blocker() {
                            self.record_blocker_resolved_audit(&row, &previous);
                        }
                        if exec.item_states.get(&row.item_id) == Some(&WorkState::Blocked) {
                            exec.item_states
                                .insert(row.item_id.clone(), WorkState::Running);
                        }
                        self.persist_row(exec, &row)
                            .map_err(|e| ExecError::retriable("re-drive registry persist", e))?;
                        self.persist_child_runtime_row(&row)?;
                    }
                    redrives.push(row);
                }
            }
        }
        for row in redrives {
            self.submit_drive_op(run_id, scheduler, &row)?;
            admitted += 1;
        }
        Ok(admitted)
    }

    fn check_crash_seam_before_drive(&self, run_id: &str) -> Result<(), ExecError> {
        let mut guard = self.exec.lock().expect("exec lock");
        let exec = guard
            .get_mut(run_id)
            .ok_or_else(|| ExecError::NotFound(format!("run {run_id} is not installed")))?;
        self.check_crash(exec, CrashSeam::BeforeDrive)
    }

    /// The REAL child creation: isolated directory + workspace/worktree
    /// rows (SessionManager), the real child session with adopted worktree
    /// identity, and the durable registry row.
    ///
    /// Ownership authority (audits 7/8/21/22): the child's mode + write
    /// paths come EXCLUSIVELY from its durable wave-A3 assignment row
    /// (which the compile recorded BEFORE any spawn) — never from the plan
    /// default, never from a spawn-time re-derivation.
    fn spawn_child(
        &self,
        exec: &mut ExecState,
        item: &WorkItem,
    ) -> Result<ChildRuntime, ExecError> {
        let spec = exec
            .specs
            .get(&item.id)
            .cloned()
            .unwrap_or_else(|| ChildSpec::new(item.id.clone()));
        // (wave A3) A child's identity is its DURABLE assignment, minted at
        // plan compile in plan order — NEVER a spawn-time counter. A spawn
        // without an assignment is a broken mirror/durable state and fails
        // loudly instead of fabricating an id.
        let assignment = exec
            .assignments
            .get(&item.id)
            .ok_or_else(|| {
                ExecError::Conflict(format!(
                    "work item {} has no durable child assignment in run {}; refusing to fabricate a child id at spawn",
                    item.id, exec.run_id
                ))
            })?;
        let child_id = assignment.child_id.clone();
        let seq = child_seq_of(&child_id).ok_or_else(|| {
            ExecError::Conflict(format!(
                "assignment of work item {} names the non-plan child id {child_id}",
                item.id
            ))
        })?;
        // A mutating kind can never spawn under a NoWrites assignment, and a
        // write-capable assignment never lands on a read-only kind: both
        // were compile-rejected; a row that says otherwise is a tampered
        // mirror/durable state and fails loudly (read-only items can never
        // receive write capability).
        if item.kind.is_mutating() && !assignment.ownership.allows_writes() {
            return Err(ExecError::InvalidState(format!(
                "mutating item {} holds the NoWrites assignment {:?}; ownership is compile-immutable",
                item.id, assignment.ownership
            )));
        }
        if !item.kind.is_mutating() && assignment.ownership.allows_writes() {
            return Err(ExecError::InvalidState(format!(
                "read-only item {} holds the write-capable assignment {:?}; write capability is never assigned to a read-only item",
                item.id, assignment.ownership
            )));
        }
        // Semantic-entity ownership (audits 7/8/21/22): the child's writes
        // are provider-scoped entities inside a snapshot — it gets no
        // file-level ownership of the shared worktree (ReadOnlyShared mode;
        // its policy-level write capability is gated below by its item
        // ownership).
        let (mode, ownership_paths) = match &assignment.ownership {
            OwnershipSpec::NoWrites | OwnershipSpec::SemanticEntities { .. } => {
                (ChildOwnership::ReadOnlyShared, Vec::new())
            }
            OwnershipSpec::IsolatedWorktree => (ChildOwnership::IsolatedWorktree, Vec::new()),
            OwnershipSpec::Paths { paths } => (ChildOwnership::ExclusivePaths, paths.clone()),
        };
        let now = self.manager.now_ms();
        let (workspace_id, worktree_id, ownership_paths) = match mode {
            ChildOwnership::ReadOnlyShared => (
                exec.owner.workspace_id,
                exec.owner.worktree_id,
                ownership_paths,
            ),
            ChildOwnership::IsolatedWorktree => {
                let dir = exec
                    .config
                    .isolated_root
                    .join(sanitize_run_id(&exec.run_id))
                    .join(&child_id);
                std::fs::create_dir_all(&dir)
                    .map_err(|e| ExecError::Internal(format!("isolated child dir {dir:?}: {e}")))?;
                let dir_str = dir.to_string_lossy().into_owned();
                let ws = self
                    .manager
                    .create_workspace(&dir_str)
                    .map_err(|e| ExecError::Internal(format!("child workspace row: {e}")))?;
                let wt_raw = self
                    .manager
                    .put_worktree(ws, &dir_str, &format!("orch-{seq}"))
                    .map_err(|e| ExecError::Internal(format!("child worktree row: {e}")))?;
                (ws.raw(), wt_raw as u64, ownership_paths)
            }
            ChildOwnership::ExclusivePaths => {
                if ownership_paths.is_empty() {
                    return Err(ExecError::InvalidState(format!(
                        "exclusive child for item {} declares no ownership paths",
                        item.id
                    )));
                }
                // Audit 21: a mutating child sharing the parent worktree is
                // only acceptable with a PROVABLY DISJOINT normalized
                // ownership set versus every other live mutating child.
                let mine =
                    SchOwnershipSet::new(ownership_paths.clone()).canonicalized(&exec.owner.root);
                for other in exec.children.values() {
                    if other.ownership != ChildOwnership::ExclusivePaths || other.is_terminal() {
                        continue;
                    }
                    let theirs = SchOwnershipSet::new(other.ownership_paths.clone())
                        .canonicalized(&exec.owner.root);
                    if mine.overlaps(&theirs) {
                        return Err(ExecError::OverlappingExclusiveOwnership(format!(
                            "child {child_id} (item {}) writes overlap live child {} (item {})",
                            item.id, other.child_id, other.item_id
                        )));
                    }
                }
                (
                    exec.owner.workspace_id,
                    exec.owner.worktree_id,
                    ownership_paths,
                )
            }
        };
        // Effective capability set at spawn: parent ∩ task ∩ child. A child
        // can never exceed its parent, even when its policy claims more.
        let permissions = effective(&exec.config.parent_caps, &spec.task_caps, &spec.child_caps);
        if !permissions.covered_by(&exec.config.parent_caps) {
            return Err(ExecError::InvalidState(format!(
                "child for item {} would exceed the parent's capability set",
                item.id
            )));
        }
        // (audits 7/8/21/22) A read-only-ownership child never holds write
        // capability: even a policy that claims WriteWorkspace (and a parent
        // that could grant it) must never leak write capability onto a
        // NoWrites item. Loud refusal — never a silent strip.
        if !assignment.ownership.allows_writes()
            && permissions
                .iter()
                .any(|g| g.cap == crate::caps::LatticeCap::WriteWorkspace)
        {
            return Err(ExecError::InvalidState(format!(
                "child for read-only item {} would receive WriteWorkspace capability; \
                 read-only items can never receive write capability",
                item.id
            )));
        }
        let mut row = ChildRuntime {
            child_id,
            parent_session_id: exec.parent_session.raw(),
            run_id: exec.run_id.clone(),
            item_id: item.id.clone(),
            kind: item.kind,
            session_id: 0,
            operation_id: 0,
            workspace_id,
            worktree_id,
            ownership: mode,
            ownership_paths,
            state: ChildState::Running,
            budget_max_tokens: spec.max_tokens,
            permissions,
            model_policy: crate::runtime::ModelPolicy {
                model: spec.model.clone(),
            },
            blocker_kind: None,
            blocker_reason: None,
            blocker_dependency: None,
            blocker_resolution: None,
            last_progress_ms: None,
            created_ms: now,
            updated_ms: now,
            base_snapshot_id: None,
            env_snapshot_id: None,
            execution_phase: ExecutionPhase::Planning,
        };
        let model = row
            .model_policy
            .model
            .clone()
            .unwrap_or_else(|| exec.config.default_model.clone());
        let title = truncate(
            &format!("{} — {}", truncate(&exec.plan.goal, 400), item.summary),
            2000,
        );
        // Audit 98: an isolated child's base snapshot is recorded DURABLY
        // at spawn (parent worktree map = the merge CAS anchors; the
        // child's own start map when it already holds content). A huge
        // parent tree fails the spawn loudly (typed Oversized) — the base
        // snapshot never silently truncates.
        if row.ownership == ChildOwnership::IsolatedWorktree {
            let base_id =
                self.record_spawn_base(exec.parent_session, &exec.run_id, &exec.owner.root, &row)?;
            row.base_snapshot_id = Some(base_id);
        }
        // Audit 97: the child binds its IMMUTABLE env snapshot at spawn —
        // the rule environment of the root its session reads instructions
        // from (the parent worktree for shared/exclusive children, its own
        // worktree for isolated ones), captured NOW and never re-read from
        // the live filesystem afterwards. The durable binding is written
        // BEFORE the child session exists, so a hostile environment beyond
        // the snapshot caps (typed Oversized) fails the spawn without
        // leaving an orphan session — the binding is never silently
        // truncated or skipped.
        let env_root = match row.ownership {
            ChildOwnership::IsolatedWorktree => self.child_worktree_dir(&row)?,
            _ => exec.owner.root.clone(),
        };
        let env_id =
            self.bind_child_env(exec.parent_session, &exec.run_id, &row.child_id, &env_root)?;
        row.env_snapshot_id = Some(env_id);
        let session = self
            .manager
            .create_child_session(
                exec.parent_session,
                faktor_core::id::WorkspaceId::new(row.workspace_id),
                WorktreeId::new(row.worktree_id),
                TaskId::new(1),
                &exec.config.provider,
                &model,
                &title,
                mode,
            )
            .map_err(|e| ExecError::Internal(format!("create_child_session: {e}")))?;
        row.session_id = session.id().raw();
        // The registry row is the durable anchor of the child session: a
        // failed write is loud (never a silently unregistered child).
        self.persist_row(exec, &row)
            .map_err(|e| ExecError::retriable("spawn registry persist", e))?;
        Ok(row)
    }
    /// Register one child drive with the scheduler. The scheduler op is the
    /// executor's handle on the drive; the drive itself runs the REAL
    /// AgentRuntime entry and records the child session's op id durably.
    fn submit_drive_op(
        &self,
        run_id: &str,
        scheduler: &Scheduler,
        child: &ChildRuntime,
    ) -> Result<(), ExecError> {
        let op_id = self.manager.next_op_id();
        let meta = OpMeta::new(
            op_id,
            SessionId::new(child.session_id),
            Deadline::at(self.manager.now_ms().saturating_add(CHILD_OP_DEADLINE_MS)),
            RetryPolicy::default(),
            CancellationToken::new(),
            RecoveryStrategy::None,
            self.manager.now_ms(),
        );
        let class = if child.is_mutating() {
            faktor_core::resource::ResourceClass::DiskWrite
        } else {
            faktor_core::resource::ResourceClass::Cpu
        };
        let writes = SchOwnershipSet::new(child.ownership_paths.clone())
            .canonicalized(&self.exec_root(run_id));
        let agent = self.agent.clone();
        let manager = self.manager.clone();
        let session_id = SessionId::new(child.session_id);
        let prompt = self.child_prompt(run_id, child)?;
        let model_override = child.model_policy.model.clone();
        let max_tokens = child.budget_max_tokens;
        // The run's durable attachment set: read from THIS child's durable
        // spec (persisted with the plan row BEFORE any spawn and decoded
        // identically on re-attach). An absent spec is an empty set — the
        // attachment-free behavior of every previous wave.
        let files = {
            let guard = self.exec.lock().expect("exec lock");
            guard
                .get(run_id)
                .and_then(|exec| exec.specs.get(&child.item_id))
                .map(|spec| spec.files.clone())
                .unwrap_or_default()
        };
        let outcomes = {
            let guard = self.exec.lock().expect("exec lock");
            guard
                .get(run_id)
                .ok_or_else(|| ExecError::NotFound(format!("run {run_id} is not installed")))?
                .outcomes
                .clone()
        };
        let parent_session = {
            let guard = self.exec.lock().expect("exec lock");
            guard
                .get(run_id)
                .ok_or_else(|| ExecError::NotFound(format!("run {run_id} is not installed")))?
                .parent_session
        };
        let run_id_owned = {
            let guard = self.exec.lock().expect("exec lock");
            guard
                .get(run_id)
                .ok_or_else(|| ExecError::NotFound(format!("run {run_id} is not installed")))?
                .run_id
                .clone()
        };
        let child_id = child.child_id.clone();
        let run = Arc::new(move || {
            let manager = manager.clone();
            let agent = agent.clone();
            let prompt = prompt.clone();
            let files = files.clone();
            let model_override = model_override.clone();
            let run_id = run_id_owned.clone();
            let child_id = child_id.clone();
            let outcomes = outcomes.clone();
            drive_op_entry(
                manager,
                agent,
                session_id,
                prompt,
                files,
                model_override,
                max_tokens,
                parent_session,
                run_id,
                child_id,
                op_id,
                outcomes,
            )
        });
        let op = ScheduledOp {
            meta,
            resources: ResourceRequest { class },
            reads: SchOwnershipSet::new(Vec::<String>::new()),
            writes,
            dependencies: Vec::new(),
            run,
        };
        scheduler
            .try_submit(op)
            .map_err(|e| ExecError::Conflict(format!("scheduler refused child op: {e}")))?;
        let mut guard = self.exec.lock().expect("exec lock");
        let exec = guard
            .get_mut(run_id)
            .ok_or_else(|| ExecError::NotFound(format!("run {run_id} is not installed")))?;
        exec.drive_ops.insert(child.child_id.clone(), op_id);
        Ok(())
    }

    fn child_prompt(&self, run_id: &str, child: &ChildRuntime) -> Result<String, ExecError> {
        let guard = self.exec.lock().expect("exec lock");
        let exec = guard
            .get(run_id)
            .ok_or_else(|| ExecError::NotFound(format!("run {run_id} is not installed")))?;
        let summary = exec
            .plan
            .work_items
            .iter()
            .find(|w| w.id == child.item_id)
            .map(|w| w.summary.clone())
            .unwrap_or_default();
        let goal = truncate(&exec.plan.goal, 1000);
        Ok(if summary.is_empty() {
            goal
        } else {
            format!("{goal}\n\nWork item: {summary}")
        })
    }

    fn exec_root(&self, run_id: &str) -> PathBuf {
        let guard = self.exec.lock().expect("exec lock");
        guard
            .get(run_id)
            .map(|e| e.owner.root.clone())
            .unwrap_or_default()
    }

    fn final_outcome(&self, run_id: &str) -> Result<PlanOutcome, ExecError> {
        let guard = self.exec.lock().expect("exec lock");
        let exec = guard
            .get(run_id)
            .ok_or_else(|| ExecError::NotFound(format!("run {run_id} is not installed")))?;
        let mut children: Vec<ChildRuntime> = exec.children.values().cloned().collect();
        children.sort_by(|a, b| {
            a.created_ms
                .cmp(&b.created_ms)
                .then_with(|| a.child_id.cmp(&b.child_id))
        });
        let mut item_states = Vec::new();
        let mut failed = Vec::new();
        let mut cancelled = Vec::new();
        let mut waiting = Vec::new();
        for w in &exec.plan.work_items {
            let st = *exec.item_states.get(&w.id).unwrap_or(&WorkState::Pending);
            item_states.push((w.id.clone(), st));
            match st {
                WorkState::Failed => failed.push(w.id.clone()),
                WorkState::Cancelled => cancelled.push(w.id.clone()),
                WorkState::Running => waiting.push(w.id.clone()),
                _ => {}
            }
        }
        // Non-terminal mirror children whose drives ended (parked waiting)
        // surface as waiting.
        for c in &children {
            if c.state == ChildState::Waiting && !waiting.contains(&c.item_id) {
                waiting.push(c.item_id.clone());
            }
        }
        let complete = item_states
            .iter()
            .all(|(_, s)| matches!(s, WorkState::Done));
        drop(guard);
        Ok(PlanOutcome {
            item_states,
            complete,
            failed,
            cancelled,
            waiting,
            children,
        })
    }
}

// ------------------------------------------------------------------ helpers

use std::collections::HashSet;

/// All durable facts of one session (bounded page scan).
fn parent_facts(
    handle: &faktor_session::SessionHandle,
) -> faktor_core::Result<Vec<(String, String, String)>> {
    let mut out = Vec::new();
    let mut after: Option<(i64, String, String)> = None;
    for _ in 0..64 {
        let page = handle.memory_facts_page(after.as_ref(), 200)?;
        out.extend(page.facts);
        match page.cursor {
            Some(c) => after = Some(c),
            None => break,
        }
    }
    Ok(out)
}

/// The child session's durable Task-row token cap — the ONE budget
/// authority the registry projection derives from (None when no task row or
/// an unlimited cap).
fn child_task_budget_cap(manager: &Arc<SessionManager>, session_id: u64) -> Option<u64> {
    let handle = manager
        .get_session(SessionId::new(session_id))
        .ok()
        .flatten()?;
    let task_id = handle.task_id().ok()?;
    handle
        .get_task(task_id)
        .ok()
        .flatten()
        .and_then(|t| t.budget.max_tokens)
}

/// The child session's durable execution phase (Planning when no drive-state
/// row exists).
fn child_execution_phase(manager: &Arc<SessionManager>, session_id: u64) -> ExecutionPhase {
    manager
        .get_session(SessionId::new(session_id))
        .ok()
        .flatten()
        .and_then(|h| h.orchestrator_drive_state_get().ok())
        .map(|ds| ds.execution_phase)
        .unwrap_or_default()
}
fn validate_config(config: &ExecConfig) -> Result<(), ExecError> {
    config.ceilings.validate().map_err(ExecError::InvalidPlan)?;
    if config.run_id.is_empty()
        || config.run_id.len() > MAX_RUN_ID_CHARS
        || !config.run_id.is_ascii()
        || config.run_id.contains('/')
    {
        return Err(ExecError::Oversized(format!(
            "run id must be 1..={MAX_RUN_ID_CHARS} ASCII characters without '/'"
        )));
    }
    if config.default_model.is_empty() {
        return Err(ExecError::InvalidPlan("default_model is empty".into()));
    }
    if config.provider.is_empty() {
        return Err(ExecError::InvalidPlan("provider is empty".into()));
    }
    Ok(())
}

fn validate_specs(
    plan: &crate::TaskPlan,
    specs: &[ChildSpec],
) -> Result<HashMap<String, ChildSpec>, ExecError> {
    let mut map = HashMap::new();
    for s in specs {
        if !plan.work_items.iter().any(|w| w.id == s.item_id) {
            return Err(ExecError::NotFound(format!(
                "spec names unknown work item {:?}",
                s.item_id
            )));
        }
        if map.insert(s.item_id.clone(), s.clone()).is_some() {
            return Err(ExecError::Conflict(format!(
                "duplicate spec for work item {}",
                s.item_id
            )));
        }
        if s.model
            .as_ref()
            .is_some_and(|m| m.is_empty() || m.len() > 128)
        {
            return Err(ExecError::InvalidPlan(
                "child model must be 1..=128 characters".into(),
            ));
        }
        validate_attachment_files(&s.files).map_err(|e| {
            ExecError::InvalidPlan(format!("attached files of work item {}: {e}", s.item_id))
        })?;
    }
    Ok(map)
}

/// The ONE attachment-file validation rule of a run (shared by the request
/// boundary, the durable spec decode and the server DTO): bounded by the
/// SAME constants the single-session prompt submission enforces
/// ([`faktor_session::MAX_FILES_PER_PROMPT`],
/// [`faktor_session::MAX_FILE_PATH_BYTES`]) plus the typed hostile-path
/// refusal (empty/control-character paths and absolute or `..` traversal
/// components). Returns a typed [`ExecError::Oversized`]/[`ExecError::Malformed`]
/// — never a truncation or a silent drop.
pub fn validate_attachment_files(files: &[String]) -> Result<(), ExecError> {
    if files.len() > faktor_session::MAX_FILES_PER_PROMPT {
        return Err(ExecError::Oversized(format!(
            "{} files exceed MAX_FILES_PER_PROMPT ({})",
            files.len(),
            faktor_session::MAX_FILES_PER_PROMPT
        )));
    }
    for f in files {
        if f.len() > faktor_session::MAX_FILE_PATH_BYTES {
            return Err(ExecError::Oversized(format!(
                "file path of {} bytes exceeds MAX_FILE_PATH_BYTES ({})",
                f.len(),
                faktor_session::MAX_FILE_PATH_BYTES
            )));
        }
        if f.trim().is_empty() {
            return Err(ExecError::Malformed(
                "an attached file path is empty or whitespace-only".into(),
            ));
        }
        if f.chars().any(|c| c.is_control()) {
            return Err(ExecError::Malformed(format!(
                "attached file path {f:?} carries control characters"
            )));
        }
        let path = std::path::Path::new(f);
        let drive_prefixed = f.len() >= 2
            && f.as_bytes()[0].is_ascii_alphabetic()
            && f.as_bytes()[1] == b':'
            && f.as_bytes()
                .get(2)
                .is_some_and(|b| *b == b'/' || *b == b'\\');
        if path.is_absolute() || drive_prefixed {
            return Err(ExecError::Malformed(format!(
                "attached file path {f:?} is absolute; attached files are workspace-relative"
            )));
        }
        if f.split(['/', '\\']).any(|segment| segment == "..") {
            return Err(ExecError::Malformed(format!(
                "attached file path {f:?} traverses outside the workspace ('..')"
            )));
        }
    }
    Ok(())
}

/// (audits 7/8/21/22) The typed-policy half of the per-item ownership
/// compile: a child whose effective ownership grants NO write authority
/// (every read-only item) must never carry WriteWorkspace in its policy —
/// even when the policy claims it and the parent could grant it. A
/// semantic-entity item's writes are provider-scoped entities: its policy
/// may not claim FILE write capability either (that would grant shared-
/// worktree file writes its ownership never authorized).
fn check_item_policies(
    specs: &HashMap<String, ChildSpec>,
    effective: &HashMap<String, OwnershipSpec>,
) -> Result<(), ExecError> {
    for spec in specs.values() {
        let Some(eff) = effective.get(&spec.item_id) else {
            continue;
        };
        let write_claimed = spec
            .task_caps
            .iter()
            .chain(spec.child_caps.iter())
            .any(|g| g.cap == crate::caps::LatticeCap::WriteWorkspace);
        if !write_claimed {
            continue;
        }
        if !eff.allows_writes() {
            return Err(ExecError::InvalidPlan(format!(
                "read-only work item {:?} requests WriteWorkspace in its policy; read-only \
                 items can never receive write capability",
                spec.item_id
            )));
        }
        if matches!(eff, OwnershipSpec::SemanticEntities { .. }) {
            return Err(ExecError::InvalidPlan(format!(
                "semantic-entity work item {:?} requests file-level WriteWorkspace in its \
                 policy; semantic-entity writes are provider-scoped and never touch the \
                 shared worktree",
                spec.item_id
            )));
        }
    }
    Ok(())
}

/// (audits 7/8/21/22) The canonicalized half of the item disjointness
/// check: the compile already rejected lexically overlapping path sets;
/// here every mutating item's path set is resolved against the REAL owner
/// root so fs-equivalent spellings (`src` vs `src/../src`) collide before
/// any spawn — not at a later live-overlap refusal.
pub(crate) fn check_plan_disjointness_canonical(
    plan: &crate::TaskPlan,
    effective: &HashMap<String, OwnershipSpec>,
    owner: &OwnerContext,
) -> Result<(), ExecError> {
    let mut path_items: Vec<(String, &Vec<String>)> = plan
        .work_items
        .iter()
        .filter(|w| w.kind.is_mutating())
        .filter_map(|w| match effective.get(&w.id) {
            Some(OwnershipSpec::Paths { paths }) => Some((w.id.clone(), paths)),
            _ => None,
        })
        .collect();
    path_items.sort_by(|a, b| a.0.cmp(&b.0));
    for i in 0..path_items.len() {
        for j in (i + 1)..path_items.len() {
            let (a_id, a_paths) = &path_items[i];
            let (b_id, b_paths) = &path_items[j];
            let mine = SchOwnershipSet::new((*a_paths).clone()).canonicalized(&owner.root);
            let theirs = SchOwnershipSet::new((*b_paths).clone()).canonicalized(&owner.root);
            if mine.overlaps(&theirs) {
                return Err(ExecError::OverlappingExclusiveOwnership(format!(
                    "work items {a_id:?} and {b_id:?} have overlapping normalized write \
                     ownership; mutating items of one plan must be disjoint before spawn"
                )));
            }
        }
    }
    Ok(())
}

fn ready_items(plan: &crate::TaskPlan, states: &HashMap<String, WorkState>) -> Vec<String> {
    plan.work_items
        .iter()
        .filter(|w| {
            states.get(&w.id) == Some(&WorkState::Pending)
                && w.depends_on
                    .iter()
                    .all(|d| states.get(d) == Some(&WorkState::Done))
        })
        .map(|w| w.id.clone())
        .collect()
}

fn can_advance(from: WorkState, to: WorkState) -> bool {
    if from == to {
        return false;
    }
    match from {
        WorkState::Pending => matches!(to, WorkState::Running | WorkState::Cancelled),
        WorkState::Running => matches!(
            to,
            WorkState::Paused
                | WorkState::Waiting
                | WorkState::Blocked
                | WorkState::Done
                | WorkState::Failed
                | WorkState::Cancelled
        ),
        WorkState::Waiting => matches!(
            to,
            WorkState::Running
                | WorkState::Blocked
                | WorkState::Done
                | WorkState::Failed
                | WorkState::Cancelled
        ),
        WorkState::Paused => {
            matches!(
                to,
                WorkState::Running | WorkState::Cancelled | WorkState::Failed
            )
        }
        WorkState::Blocked => {
            matches!(
                to,
                WorkState::Pending | WorkState::Running | WorkState::Failed
            )
        }
        WorkState::Failed => to == WorkState::Pending,
        WorkState::Done | WorkState::Cancelled => false,
    }
}

fn sanitize_run_id(run_id: &str) -> String {
    run_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// The numeric sequence of a durable child id (`child-N` plan children and
/// `review-N` reviewers share ONE counter; ALL plan ids are minted at plan
/// compile, so a reviewer can never sit below a reserved plan id).
fn child_seq_of(child_id: &str) -> Option<u64> {
    for prefix in ["child-", "review-"] {
        if let Some(n) = child_id.strip_prefix(prefix) {
            if let Ok(n) = n.parse::<u64>() {
                return Some(n);
            }
        }
    }
    None
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_string()
}

/// The REAL child drive: the same entries the daemon uses. An interrupted
/// drive resumes the SAME recorded turn (continue_turn — never a
/// synthesized operation); otherwise submit → drive_receipt. The drive ends
/// when the child's durable op record ends; the executor never polls.
///
/// `files` is the run's immutable attachment set (decoded from the durable
/// child spec): a fresh submit carries EXACTLY it, and a crash-resumed turn
/// re-continues the SAME recorded prompt (whose durable message already
/// carries the same set), so attachments survive acceptance and re-attach.
async fn drive_child_turn(
    manager: Arc<SessionManager>,
    agent: Arc<AgentRuntime>,
    session: SessionId,
    prompt: &str,
    files: &[String],
    model_override: Option<String>,
    max_tokens: Option<u64>,
) -> (Result<TurnOutcome, String>, Option<OpId>) {
    let handle = match manager.get_session(session) {
        Ok(Some(h)) => h,
        Ok(None) => return (Err(format!("child session {session} missing")), None),
        Err(e) => return (Err(e.message), None),
    };
    if let Ok(Some(record)) = handle.active_turn_record() {
        // Crash resume / re-attach: drive the SAME logical turn (recorded
        // op id + envelope). The agent's recovery paths resolve unfinished
        // tool runs first (never blindly re-run).
        let op = record.turn_op_id;
        let res = agent.continue_turn(session).await.map_err(|e| e.message);
        return (res, Some(op));
    }
    if let Some(mt) = max_tokens {
        if let Err(e) = agent.seed_task_budget(
            session,
            &TaskBudget {
                max_tokens: Some(mt),
                max_turns: None,
                spent_tokens: 0,
                spent_turns: 0,
            },
        ) {
            return (Err(e.message), None);
        }
    }
    let receipt = match agent.submit(session, prompt, files) {
        Ok(r) => r,
        Err(e) => return (Err(e.message), None),
    };
    // The child session's REAL op id, mapped durably (audit 20: operation
    // id == the child session's op id).
    let turn_op_id = Some(receipt.op_id);
    if receipt.queued {
        // The per-session queue runner delivers queued prompts; drive the
        // queue to its end (the runner is the daemon's own entry).
        agent.run_session_queue(session).await;
        return (
            Err("child drive queued then drained; classify from session state".to_string()),
            turn_op_id,
        );
    }
    let res = agent
        .drive_receipt(&handle, receipt, model_override)
        .await
        .map_err(|e| e.message);
    (res, turn_op_id)
}

/// The scheduler op body of one child drive: runs the REAL agent drive and
/// records the child's durable op id + outcome (keyed by scheduler op id).
/// Returns the boxed future directly (the scheduler's `OpFn` alias needs
/// `Pin<Box<dyn Future + Send>>`; coercing at this return position keeps the
/// Send obligation inside this function).
#[allow(clippy::too_many_arguments)]
fn drive_op_entry(
    manager: Arc<SessionManager>,
    agent: Arc<AgentRuntime>,
    session_id: SessionId,
    prompt: String,
    files: Vec<String>,
    model_override: Option<String>,
    max_tokens: Option<u64>,
    parent_session: SessionId,
    run_id: String,
    child_id: String,
    op_id: OpId,
    outcomes: Arc<Mutex<HashMap<OpId, DriveResult>>>,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), faktor_core::Error>> + Send>> {
    Box::pin(async move {
        let (res, turn_op_id) = drive_child_turn(
            manager.clone(),
            agent,
            session_id,
            &prompt,
            &files,
            model_override,
            max_tokens,
        )
        .await;
        // Record the child session's REAL op id durably (registry row +
        // identity row) so re-attach maps the child to its op. AUTHORITATIVE:
        // a failed persist is collected and returned AFTER the outcome is
        // recorded, so the scheduler always sees the drive result and the
        // failure is loud (typed Store, retryable) instead of a silent
        // mapping loss.
        let mut op_id_error: Option<String> = None;
        if let Some(turn_op) = turn_op_id {
            match manager.get_session(parent_session) {
                Ok(Some(parent)) => {
                    let key = format!("{run_id}/{child_id}");
                    match parent_facts(&parent) {
                        Ok(facts) => {
                            for (kind, k, value) in facts {
                                if kind == REGISTRY_ROW_KIND && k == key {
                                    match serde_json::from_str::<ChildRuntime>(&value) {
                                        Ok(mut row) => {
                                            row.operation_id = turn_op.raw();
                                            match serde_json::to_string(&row) {
                                                Ok(value) => {
                                                    if let Err(e) = parent.upsert_memory_fact(
                                                        REGISTRY_ROW_KIND,
                                                        &key,
                                                        &value,
                                                    ) {
                                                        op_id_error = Some(format!(
                                                            "child op-id registry persist: {e}"
                                                        ));
                                                    }
                                                }
                                                Err(e) => {
                                                    op_id_error = Some(format!(
                                                        "child op-id registry serialize: {e}"
                                                    ))
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            op_id_error =
                                                Some(format!("child op-id registry decode: {e}"))
                                        }
                                    }
                                    break;
                                }
                            }
                        }
                        Err(e) => op_id_error = Some(format!("child op-id registry scan: {e}")),
                    }
                }
                Ok(None) => {}
                Err(e) => op_id_error = Some(format!("child op-id parent session: {e}")),
            }
            if op_id_error.is_none() {
                match manager.get_session(session_id) {
                    Ok(Some(child_handle)) => {
                        match child_handle.orchestrator_child_identity_get() {
                            Ok(Some(mut identity)) => {
                                identity.operation_id = turn_op.raw();
                                if let Err(e) =
                                    child_handle.orchestrator_child_identity_put(&identity)
                                {
                                    op_id_error =
                                        Some(format!("child op-id identity persist: {e}"));
                                }
                            }
                            Ok(None) => {}
                            Err(e) => op_id_error = Some(format!("child op-id identity read: {e}")),
                        }
                    }
                    Ok(None) => {}
                    Err(e) => op_id_error = Some(format!("child op-id session: {e}")),
                }
            }
        }
        let mut map = outcomes.lock().unwrap();
        map.insert(
            op_id,
            DriveResult {
                turn_op_id,
                result: res,
            },
        );
        drop(map);
        if let Some(message) = op_id_error {
            return Err(faktor_core::Error::new(
                faktor_core::ErrorKind::Store,
                message,
            ));
        }
        Ok(())
    })
}

#[cfg(test)]
#[path = "runtime_tests.rs"]
mod runtime_tests;

/// Controlled child-worktree merging + reviewer worktree semantics
/// (audits 69/70/98/99): ChangeSet staging, approve_and_merge, durable
/// merge records, spawn_reviewer.
#[path = "merge.rs"]
pub mod merge;

/// Durable env-snapshot rows + epoch-pinned context reads (audit 97).
#[path = "env.rs"]
pub(crate) mod env;

/// The single durable operation graph read-model (audit 93).
#[path = "graph.rs"]
pub(crate) mod graph;

/// The TaskExecutor (audits P0-20/21/23/61/90/91): ONE authoritative entry
/// for native task starts. A single-work-item task runs in the existing
/// session with the daemon's own drive path (byte-compatible receipts and
/// events, wrapped with a durable task-linkage row); a multi-item task
/// spawns real child sessions through [`OrchestratorRuntime::execute_task`]
/// (wave 12). There is no second execution architecture.
#[path = "task_executor.rs"]
pub mod task_executor;

/// Shadow mutation roots (P0-48): config-gated shadow copies of the user
/// checkout that mutating single-agent tasks work against; integration
/// back into the user checkout is a conflict-aware CAS commit.
#[path = "shadow.rs"]
pub mod shadow;

/// PR/CI-fix completion step EXECUTION (P2 follow-up): the ordered
/// commit/push/pr runner invoked additively after a run's deterministic
/// verification and before the durable completion gate certifies, recording
/// every outcome through the existing `set_completion_step_status` seam.
#[path = "completion_steps.rs"]
pub mod completion_steps;

/// Map a finished drive to the child's terminal or blocked state plus the
/// durable blocker when one applies. A genuine end whose OWN verification
/// failed is a FAILED child — never a claimed complete. A turn that ended
/// waiting on a permission decision is BLOCKED, and a genuine end whose
/// completion gate was BLOCKED by a budget refusal is BLOCKED with the
/// typed budget blocker — never silently Done.
fn classify_outcome(res: Result<TurnOutcome, String>) -> (ChildState, Option<ChildBlocker>) {
    let outcome = match res {
        Ok(o) => o,
        Err(_) => return (ChildState::Failed, None),
    };
    // A typed budget stop is a durable BLOCK regardless of the turn's
    // recoverable failure state: the child is not failed, it is waiting on
    // capacity.
    if let Some(reason) = &outcome.stop_reason {
        if matches!(
            reason.code,
            faktor_core::state::ReasonCode::BudgetExceeded
                | faktor_core::state::ReasonCode::SpendOverBudget
        ) {
            return (
                ChildState::Blocked,
                Some(ChildBlocker::new(
                    BlockerKind::Budget,
                    reason.detail.clone(),
                    "increase the child's token/cost budget or wait for the run budget, then resume",
                )),
            );
        }
    }
    match outcome.final_state {
        AgentState::Cancelled => (ChildState::Cancelled, None),
        AgentState::NeedsUserInput => (
            ChildState::Blocked,
            Some(ChildBlocker::new(
                BlockerKind::Permission,
                "waiting for a pending permission decision",
                "resolve the pending permission request, then resume the child",
            )),
        ),
        AgentState::ReadyForNextTurn | AgentState::Completed => {
            if outcome.acceptance == Some(faktor_agent::Acceptance::Fail)
                || matches!(
                    outcome.completion,
                    Some(faktor_agent::CompletionGate::FailedVerification { .. })
                )
            {
                return (ChildState::Failed, None);
            }
            // Blocked completion gates are NOT complete: a budget refusal
            // (hard denial / spend over cap) is a durable BLOCKED child.
            if let Some(faktor_agent::CompletionGate::BlockedVerification { reasons }) =
                &outcome.completion
            {
                let budget_reason = reasons.iter().find(|r| {
                    matches!(
                        r.code,
                        faktor_core::state::ReasonCode::BudgetExceeded
                            | faktor_core::state::ReasonCode::SpendOverBudget
                    )
                });
                if let Some(reason) = budget_reason {
                    return (
                        ChildState::Blocked,
                        Some(ChildBlocker::new(
                            BlockerKind::Budget,
                            reason.detail.clone(),
                            "increase the child's token/cost budget or wait for the run budget, then resume",
                        )),
                    );
                }
                return (
                    ChildState::Blocked,
                    Some(ChildBlocker::new(
                        BlockerKind::Verification,
                        reasons
                            .first()
                            .map(|r| r.detail.clone())
                            .unwrap_or_else(|| "verification is blocked".into()),
                        "resolve the blocking verification condition, then resume",
                    )),
                );
            }
            (ChildState::Done, None)
        }
        _ => (ChildState::Failed, None),
    }
}

/// The work item's dependency that is not Done yet (or `None` when every
/// dependency is satisfied). A dependency with no child row counts as its
/// plan `completion` (a pre-satisfied item never spawns).
fn unmet_dependency(
    plan: &crate::TaskPlan,
    rows: &[ChildRuntime],
    row: &ChildRuntime,
) -> Option<ChildBlocker> {
    let item = plan.work_items.iter().find(|w| w.id == row.item_id)?;
    for dep in &item.depends_on {
        let state = rows
            .iter()
            .find(|r| r.item_id == *dep)
            .map(project_child_state)
            .or_else(|| {
                plan.work_items
                    .iter()
                    .find(|w| w.id == *dep)
                    .map(|w| w.completion)
            })
            .unwrap_or(WorkState::Pending);
        if state != WorkState::Done {
            return Some(ChildBlocker::dependency(
                dep,
                format!("waiting on work item {dep:?}"),
                "wait for the dependency to complete, then resume the child",
            ));
        }
    }
    None
}

/// Whether every dependency of `item_id` is Done in the live mirror.
fn dependencies_done(
    plan: &crate::TaskPlan,
    states: &HashMap<String, WorkState>,
    item_id: &str,
) -> bool {
    plan.work_items
        .iter()
        .find(|w| w.id == item_id)
        .is_some_and(|w| {
            w.depends_on
                .iter()
                .all(|d| states.get(d) == Some(&WorkState::Done))
        })
}

impl OrchestratorRuntime {
    /// Persist the typed, bounded child-runtime projection (v23 table) on
    /// the CHILD session: the queryable blocker truth beside the registry
    /// JSON row. AUTHORITATIVE: a failed write is a typed retriable error —
    /// never a silent skip (hostile text can never get here; every producer
    /// validates first). A missing child session is a no-op (re-attach
    /// validates sessions separately and loudly).
    fn persist_child_runtime_row(&self, row: &ChildRuntime) -> Result<(), ExecError> {
        let Some(session) = self
            .manager
            .get_session(SessionId::new(row.session_id))
            .map_err(|e| ExecError::retriable("child session read", e))?
        else {
            return Ok(());
        };
        // Re-validate at the projection boundary: a Blocked row always rides
        // a populated blocker (the shared decode invariant).
        row.validate_durable()?;
        let typed = faktor_session::child::ChildRuntimeBlockerRow {
            child_id: row.child_id.clone(),
            state: child_state_tag(row.state).to_string(),
            blocker: row.blocker(),
            updated_ms: row.updated_ms,
        };
        session
            .orchestrator_child_runtime_put(&typed)
            .map_err(|e| ExecError::retriable("child runtime projection", e))?;
        Ok(())
    }

    /// Best-effort persist of a child's execution phase at a safe boundary
    /// (settlement/re-attach). The phase is a projection, never an
    /// authoritative lifecycle transition, so a failure is logged and the
    /// in-memory projection still carries the intended phase.
    fn note_child_phase(&self, row: &ChildRuntime, phase: ExecutionPhase) {
        let Ok(Some(session)) = self.manager.get_session(SessionId::new(row.session_id)) else {
            return;
        };
        if let Err(e) = session.set_execution_phase(phase) {
            tracing::debug!(
                child = %row.child_id,
                phase = phase.as_str(),
                error = %e.message,
                "child execution phase persist failed"
            );
        }
    }

    /// Audit one blocker open in the child session's typed ledger. Opening
    /// a reason already open is an idempotent no-op; failures are logged,
    /// never fatal (the durable row already carries the truth).
    fn record_blocker_audit(&self, row: &ChildRuntime, blocker: &ChildBlocker) {
        let Ok(Some(session)) = self.manager.get_session(SessionId::new(row.session_id)) else {
            return;
        };
        if let Err(e) = session.ledger_child_blocker_opened(blocker) {
            tracing::debug!(
                child = %row.child_id,
                reason = %blocker.reason,
                error = %e.message,
                "child blocker ledger open failed"
            );
        }
    }

    /// Audit one blocker resolution in the child session's typed ledger.
    fn record_blocker_resolved_audit(&self, row: &ChildRuntime, blocker: &ChildBlocker) {
        let Ok(Some(session)) = self.manager.get_session(SessionId::new(row.session_id)) else {
            return;
        };
        if let Err(e) = session.ledger_child_blocker_resolved(blocker) {
            tracing::debug!(
                child = %row.child_id,
                reason = %blocker.reason,
                error = %e.message,
                "child blocker ledger resolve failed"
            );
        }
    }
}
