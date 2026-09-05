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
use faktor_session::child::{ChildControl, ChildOwnership, ChildPhase};
use faktor_session::{SessionManager, TaskBudget};

use crate::caps::{effective, CapabilitySet};
use crate::{ChildState, WorkItem, WorkKind, WorkState};

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
    #[error("oversized: {0}")]
    Oversized(String),
    #[error("plan validation failed: {0}")]
    InvalidPlan(String),
    #[error("injected crash seam {0} (test seam; durable state left as-is for re-attach)")]
    InjectedCrashSeam(String),
    #[error("internal: {0}")]
    Internal(String),
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
/// position.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ChildSpec {
    pub item_id: String,
    /// `false` executes the item without a real child session.
    pub spawn: bool,
    /// Ownership override; `None` derives from kind + plan ownership.
    pub ownership: Option<ChildOwnership>,
    /// Exclusive write paths (normalized against the owner root) used when
    /// ownership is `ExclusivePaths`.
    pub ownership_paths: Vec<String>,
    pub model: Option<String>,
    /// Durable token budget cap (the wave-9 Task budget fields).
    pub max_tokens: Option<u64>,
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
            ownership: None,
            ownership_paths: Vec::new(),
            model: None,
            max_tokens: None,
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
    pub created_ms: i64,
    pub updated_ms: i64,
}

impl ChildRuntime {
    pub fn is_terminal(&self) -> bool {
        self.state.is_terminal()
    }

    pub fn is_mutating(&self) -> bool {
        self.kind.is_mutating()
    }
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
    /// Scheduler op id of the in-flight drive per child id.
    drive_ops: HashMap<String, OpId>,
    /// Outcomes written by the drive closures, keyed by scheduler op id.
    outcomes: Arc<Mutex<HashMap<OpId, DriveResult>>>,
    next_child_seq: u64,
    crash_fired: bool,
}

/// The orchestration runtime: the manager + agent it drives children with,
/// and the durable control surface. One execution at a time.
pub struct OrchestratorRuntime {
    manager: Arc<SessionManager>,
    agent: Arc<AgentRuntime>,
    exec: Mutex<Option<ExecState>>,
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
            exec: Mutex::new(None),
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
                let row: ChildRuntime = serde_json::from_str(&value).map_err(|e| {
                    faktor_core::Error::internal(format!("registry row decode: {e}"))
                })?;
                rows.push(row);
            }
        }
        rows.sort_by_key(|r| r.created_ms);
        Ok(rows)
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

    /// Execute the plan with REAL children (audit 20). A run id that
    /// already has durable registry rows is a Conflict — call
    /// [`OrchestratorRuntime::reattach`] to resume a crashed executor.
    pub async fn execute_task(
        self: &Arc<Self>,
        plan: crate::TaskPlan,
        owner: OwnerContext,
        config: ExecConfig,
        specs: &[ChildSpec],
    ) -> Result<PlanOutcome, ExecError> {
        validate_config(&config)?;
        plan.validate()
            .map_err(|errs| ExecError::InvalidPlan(errs.join("; ")))?;
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
        self.put_plan_row(&plan, &owner, &config, specs)?;
        let state = self.build_exec_state(plan, owner, config, spec_map);
        *self.exec.lock().expect("exec lock") = Some(state);
        self.drive_to_outcome().await
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
        *self.exec.lock().expect("exec lock") = Some(state);
        self.drive_to_outcome().await
    }

    // ------------------------------------------------------------ steering

    /// Pause a child: enqueues the durable Pause control; the child's own
    /// drive applies it at its next safe reasoning boundary (never
    /// mid-operation). Non-terminal children only.
    pub fn pause_child(&self, child_id: &str) -> Result<(), ExecError> {
        self.enqueue_child_control(child_id, ChildControl::Pause)
    }

    /// Resume a paused/waiting child (durable Resume row).
    pub fn resume_child(&self, child_id: &str) -> Result<(), ExecError> {
        self.enqueue_child_control(child_id, ChildControl::Resume)
    }

    /// Cancel a non-terminal child: durable Cancel row + the bounded abort
    /// path on the child's session (the turn ends Cancelled; the session
    /// stays promptable — never a dead session).
    pub fn cancel_child(&self, child_id: &str) -> Result<(), ExecError> {
        let row = self.durable_child(child_id)?;
        if row.state.is_terminal() {
            return Err(ExecError::InvalidState(format!(
                "cannot cancel child {child_id}: state {:?} is terminal",
                row.state
            )));
        }
        let session = self
            .manager
            .get_session(SessionId::new(row.session_id))?
            .ok_or_else(|| ExecError::NotFound(format!("child session {}", row.session_id)))?;
        let msg = session.orchestrator_ctl_enqueue(ChildControl::Cancel)?;
        let _ = session.orchestrator_ctl_ack(msg.seq);
        // The bounded abort path (existing semantics): fires the turn
        // cancellation token; an abort on a session with no registered op
        // is a no-op that leaves the session promptable.
        if let Ok(Some(record)) = session.active_turn_record() {
            let _ = session.abort(Some(record.turn_op_id));
        }
        Ok(())
    }

    /// Steer a child with a guidance note (bounded; durable control row;
    /// applied at the child's next safe reasoning boundary).
    pub fn steer_child(&self, child_id: &str, note: &str) -> Result<(), ExecError> {
        self.enqueue_child_control(
            child_id,
            ChildControl::Steer {
                note: note.to_string(),
            },
        )
    }

    /// Change the child's model: takes effect at the child's next provider
    /// selection (durable ChangeModel row applied at the next boundary).
    pub fn change_child_model(&self, child_id: &str, model: &str) -> Result<(), ExecError> {
        if model.is_empty() || model.chars().count() > 128 {
            return Err(ExecError::Oversized(
                "model selector must be 1..=128 characters".into(),
            ));
        }
        self.enqueue_child_control(
            child_id,
            ChildControl::ChangeModel {
                model: model.to_string(),
            },
        )
    }

    /// Change the child's durable token budget cap: the Task row is patched
    /// immediately (the wave-9 budget fields gate every genuine turn end)
    /// and the ChangeBudget row is acked — the effect is durable and
    /// idempotent.
    pub fn change_child_budget(&self, child_id: &str, max_tokens: u64) -> Result<(), ExecError> {
        let row = self.durable_child(child_id)?;
        if row.state.is_terminal() {
            return Err(ExecError::InvalidState(format!(
                "cannot change the budget of {child_id}: state {:?} is terminal",
                row.state
            )));
        }
        let session = self
            .manager
            .get_session(SessionId::new(row.session_id))?
            .ok_or_else(|| ExecError::NotFound(format!("child session {}", row.session_id)))?;
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
            .map_err(|e| ExecError::Internal(format!("budget patch: {}", e.message)))?;
        let msg = session.orchestrator_ctl_enqueue(ChildControl::ChangeBudget { max_tokens })?;
        let _ = session.orchestrator_ctl_ack(msg.seq);
        // Reflect the durable cap on the registry row.
        if let Some(exec) = self.exec.lock().expect("exec lock").as_mut() {
            if let Some(c) = exec.children.get_mut(child_id) {
                c.budget_max_tokens = Some(max_tokens);
            }
            if let Some(c) = exec.children.get(child_id) {
                let _ = self.persist_row(exec, c);
            }
        }
        Ok(())
    }

    /// Drive one retry of a Failed child (durable Retry row required).
    /// Only Failed children retry; the row is acked when the re-drive is
    /// admitted (never blindly re-run: the agent's submit path runs
    /// session recovery first, and a mid-drive crash resumes the SAME
    /// recorded turn).
    pub fn retry_child(&self, child_id: &str) -> Result<(), ExecError> {
        let row = self.durable_child(child_id)?;
        if row.state != ChildState::Failed {
            return Err(ExecError::InvalidState(format!(
                "cannot retry child {child_id}: only Failed children retry (state {:?})",
                row.state
            )));
        }
        let session = self
            .manager
            .get_session(SessionId::new(row.session_id))?
            .ok_or_else(|| ExecError::NotFound(format!("child session {}", row.session_id)))?;
        session.orchestrator_ctl_enqueue(ChildControl::Retry)?;
        Ok(())
    }

    /// The live mirror of one child (durable rows are the source of truth;
    /// this refreshes the mirror from the registry).
    pub fn child(&self, child_id: &str) -> Result<Option<ChildRuntime>, ExecError> {
        let guard = self.exec.lock().expect("exec lock");
        Ok(guard
            .as_ref()
            .and_then(|e| e.children.get(child_id).cloned()))
    }

    fn enqueue_child_control(
        &self,
        child_id: &str,
        control: ChildControl,
    ) -> Result<(), ExecError> {
        let row = self.durable_child(child_id)?;
        let session = self
            .manager
            .get_session(SessionId::new(row.session_id))?
            .ok_or_else(|| ExecError::NotFound(format!("child session {}", row.session_id)))?;
        let terminal = session.state()?.is_terminal()
            || matches!(
                row.state,
                ChildState::Done | ChildState::Cancelled | ChildState::Failed
            );
        match &control {
            ChildControl::Pause if terminal => {
                return Err(ExecError::InvalidState(format!(
                    "cannot pause child {child_id}: terminal"
                )));
            }
            ChildControl::Resume => {
                // Resume is only meaningful for paused/waiting children.
                if !matches!(
                    row.state,
                    ChildState::Paused | ChildState::Waiting | ChildState::Running
                ) {
                    return Err(ExecError::InvalidState(format!(
                        "cannot resume child {child_id}: state {:?}",
                        row.state
                    )));
                }
            }
            ChildControl::Steer { .. } => {}
            ChildControl::ChangeModel { .. } if terminal => {
                return Err(ExecError::InvalidState(format!(
                    "cannot change the model of {child_id}: terminal"
                )));
            }
            _ => {}
        }
        session.orchestrator_ctl_enqueue(control)?;
        Ok(())
    }

    fn durable_child(&self, child_id: &str) -> Result<ChildRuntime, ExecError> {
        let guard = self.exec.lock().expect("exec lock");
        let exec = guard
            .as_ref()
            .ok_or_else(|| ExecError::NotFound("no active execution".into()))?;
        exec.children
            .get(child_id)
            .cloned()
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
        ExecState {
            parent_session: owner.parent_session,
            run_id: config.run_id.clone(),
            plan,
            owner,
            config,
            specs,
            item_states,
            children: HashMap::new(),
            drive_ops: HashMap::new(),
            outcomes: Arc::new(Mutex::new(HashMap::new())),
            next_child_seq: 0,
            crash_fired: false,
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
                let plan: crate::TaskPlan =
                    serde_json::from_value(v.get("plan").cloned().unwrap_or_default())
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
        let mut children = HashMap::new();
        let mut item_states: HashMap<String, WorkState> = state
            .plan
            .work_items
            .iter()
            .map(|w| (w.id.clone(), WorkState::Pending))
            .collect();
        for mut row in rows {
            let reconciled = self.reconcile_child_state(&row)?;
            row.state = reconciled;
            let _ = self.persist_row(state, &row);
            let item = row.item_id.clone();
            // Pending -> Running first (a terminal child only got there
            // through a Running item; re-attach must never re-spawn an item
            // that already has a durable child).
            if item_states[&item] == WorkState::Pending
                && can_advance(WorkState::Pending, WorkState::Running)
            {
                item_states.insert(item.clone(), WorkState::Running);
            }
            let target = match reconciled {
                ChildState::Done => Some(WorkState::Done),
                ChildState::Cancelled => Some(WorkState::Cancelled),
                ChildState::Failed => Some(WorkState::Failed),
                _ => None,
            };
            if let Some(t) = target {
                if can_advance(item_states[&item], t) {
                    item_states.insert(item, t);
                }
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
        let max_seq = children
            .keys()
            .filter_map(|c| c.strip_prefix("child-"))
            .filter_map(|n| n.parse::<u64>().ok())
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
    /// (re-attach, crash windows included).
    fn reconcile_child_state(&self, row: &ChildRuntime) -> Result<ChildState, ExecError> {
        if row.state.is_terminal() {
            return Ok(row.state);
        }
        let session = self
            .manager
            .get_session(SessionId::new(row.session_id))?
            .ok_or_else(|| ExecError::NotFound(format!("child session {}", row.session_id)))?;
        let ds = session.orchestrator_drive_state_get()?;
        if ds.phase == ChildPhase::Waiting {
            return Ok(ChildState::Waiting);
        }
        match session.state()? {
            AgentState::Completed => Ok(ChildState::Done),
            AgentState::ReadyForNextTurn => {
                // A genuine end happened; close a turn record left active by
                // a crash between TurnCompleted and finish_turn_record.
                if let Ok(Some(record)) = session.active_turn_record() {
                    let _ = session.finish_turn_record(record.turn_op_id, "completed");
                }
                Ok(ChildState::Done)
            }
            AgentState::Cancelled => Ok(ChildState::Cancelled),
            AgentState::FailedRecoverable | AgentState::FailedPermanent => {
                if let Ok(Some(record)) = session.active_turn_record() {
                    let _ = session.finish_turn_record(record.turn_op_id, "failed");
                }
                Ok(ChildState::Failed)
            }
            AgentState::NeedsUserInput => Ok(ChildState::Failed),
            // Mid-turn: driveable from the durable op record.
            _ => Ok(ChildState::Running),
        }
    }

    /// The supervision loop: settle finished drives, admit new waves under
    /// the ceilings, drive each wave through the scheduler (paused children
    /// park inside their drive and hold the wave until resumed).
    async fn drive_to_outcome(&self) -> Result<PlanOutcome, ExecError> {
        let mut limits = faktor_core::resource::ResourceLimits::default();
        let scheduler = {
            let guard = self.exec.lock().expect("exec lock");
            let exec = guard.as_ref().expect("execution installed");
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
            self.settle_finished_drives()?;
            let admitted = self.admit_ready(&scheduler)?;
            if admitted == 0 {
                return self.final_outcome();
            }
            scheduler
                .run_to_completion()
                .await
                .map_err(|e| ExecError::Internal(format!("scheduler wave failed: {e}")))?;
        }
    }

    /// Classify finished drives, write durable child rows, advance items.
    fn settle_finished_drives(&self) -> Result<(), ExecError> {
        let mut guard = self.exec.lock().expect("exec lock");
        let Some(exec) = guard.as_mut() else {
            return Ok(());
        };
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
                let outcome_state = classify_outcome(drive.result);
                if row.state != outcome_state {
                    row.state = outcome_state;
                    row.updated_ms = self.manager.now_ms();
                }
                if let Some(turn_op) = drive.turn_op_id {
                    row.operation_id = turn_op.raw();
                }
                let _ = self.persist_row(exec, &row);
                exec.children.insert(child_id.clone(), row.clone());
                exec.drive_ops.remove(&child_id);
                done.push((child_id, row));
            }
        }
        drop(outcomes);
        drop(guard);
        for (child_id, row) in done {
            self.advance_item(&child_id, &row)?;
        }
        // Crash seam: a child just reached terminal state.
        {
            let terminal_child = {
                let guard = self.exec.lock().expect("exec lock");
                let exec = guard.as_ref().expect("execution installed");
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
                let exec = guard.as_mut().expect("execution installed");
                exec.crash_fired = true;
                return Err(ExecError::InjectedCrashSeam(format!(
                    "AfterChildTerminal (child {child_id})"
                )));
            }
        }
        Ok(())
    }

    fn advance_item(&self, child_id: &str, row: &ChildRuntime) -> Result<(), ExecError> {
        let mut guard = self.exec.lock().expect("exec lock");
        let exec = guard.as_mut().expect("execution installed");
        let target = match row.state {
            ChildState::Done => WorkState::Done,
            ChildState::Cancelled => WorkState::Cancelled,
            ChildState::Failed => WorkState::Failed,
            _ => return Ok(()),
        };
        let item = row.item_id.clone();
        let cur = *exec.item_states.get(&item).unwrap_or(&WorkState::Pending);
        if cur == target {
            return Ok(());
        }
        if can_advance(cur, target) {
            exec.item_states.insert(item.clone(), target);
        } else if cur == WorkState::Failed
            && matches!(target, WorkState::Done | WorkState::Cancelled)
        {
            // A retried child completed: legal chain Failed -> Pending ->
            // Running -> terminal.
            exec.item_states.insert(item.clone(), WorkState::Pending);
            exec.item_states.insert(item.clone(), WorkState::Running);
            exec.item_states.insert(item.clone(), target);
        } else {
            return Ok(());
        }
        match target {
            WorkState::Failed | WorkState::Cancelled => {
                for w in &exec.plan.work_items {
                    if w.depends_on.contains(&item)
                        && exec.item_states.get(&w.id) == Some(&WorkState::Pending)
                    {
                        exec.item_states.insert(w.id.clone(), WorkState::Blocked);
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
        let _ = child_id;
        Ok(())
    }

    /// Admit ready work under the ceilings. Returns the number of drives
    /// registered with the scheduler.
    fn admit_ready(&self, scheduler: &Scheduler) -> Result<usize, ExecError> {
        let mut admitted = 0usize;
        let mut newly_spawned: Vec<ChildRuntime> = Vec::new();
        {
            let mut guard = self.exec.lock().expect("exec lock");
            let exec = guard.as_mut().expect("execution installed");
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
            for item in ready {
                // Live ceiling: a hard, typed reject BEFORE each spawn (a
                // child already registered in THIS pass counts).
                let live = exec.children.values().filter(|c| !c.is_terminal()).count();
                if live >= exec.config.ceilings.max_live {
                    return Err(ExecError::CeilingExceeded {
                        class: "live",
                        limit: exec.config.ceilings.max_live,
                        used: live,
                    });
                }
                let class = if item.kind.is_mutating() {
                    "mutating"
                } else {
                    "reasoning"
                };
                let ceiling = if class == "mutating" {
                    exec.config.ceilings.max_mutating_active
                } else {
                    exec.config.ceilings.max_reasoning_active
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
            self.check_crash_seam_before_drive(child.child_id.as_str())?;
            self.submit_drive_op(scheduler, child)?;
        }
        // (b) Waiting children with a pending Resume row and Failed children
        // with a pending Retry row are re-driven (the retry row is acked at
        // admission: the decision is durable before the drive starts).
        let mut redrives = Vec::new();
        {
            let mut guard = self.exec.lock().expect("exec lock");
            let exec = guard.as_mut().expect("execution installed");
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
                                let _ = session.orchestrator_ctl_ack(r.seq);
                                should_drive = true;
                                break;
                            }
                        }
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
                        let _ = self.persist_row(exec, &row);
                    }
                    redrives.push(row);
                }
            }
        }
        for row in redrives {
            self.submit_drive_op(scheduler, &row)?;
            admitted += 1;
        }
        Ok(admitted)
    }

    fn check_crash_seam_before_drive(&self, _child_id: &str) -> Result<(), ExecError> {
        let mut guard = self.exec.lock().expect("exec lock");
        let exec = guard.as_mut().expect("execution installed");
        self.check_crash(exec, CrashSeam::BeforeDrive)
    }

    /// The REAL child creation: isolated directory + workspace/worktree
    /// rows (SessionManager), the real child session with adopted worktree
    /// identity, and the durable registry row.
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
        let mode = match spec.ownership {
            Some(m) => m,
            None => derive_ownership(item, &exec.plan)?,
        };
        let seq = exec.next_child_seq;
        exec.next_child_seq += 1;
        let child_id = format!("child-{seq}");
        let now = self.manager.now_ms();
        let (workspace_id, worktree_id, ownership_paths) = match mode {
            ChildOwnership::ReadOnlyShared => {
                (exec.owner.workspace_id, exec.owner.worktree_id, Vec::new())
            }
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
                (ws.raw(), wt_raw as u64, Vec::new())
            }
            ChildOwnership::ExclusivePaths => {
                let paths = if spec.ownership_paths.is_empty() {
                    match &exec.plan.ownership {
                        crate::OwnershipModel::DisjointPaths { paths } => paths.clone(),
                        _ => Vec::new(),
                    }
                } else {
                    spec.ownership_paths.clone()
                };
                if paths.is_empty() {
                    return Err(ExecError::InvalidState(format!(
                        "exclusive child for item {} declares no ownership paths",
                        item.id
                    )));
                }
                // Audit 21: a mutating child sharing the parent worktree is
                // only acceptable with a PROVABLY DISJOINT normalized
                // ownership set versus every other live mutating child.
                let mine = SchOwnershipSet::new(paths.clone()).canonicalized(&exec.owner.root);
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
                (exec.owner.workspace_id, exec.owner.worktree_id, paths)
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
        let row = ChildRuntime {
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
            created_ms: now,
            updated_ms: now,
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
        let mut row = row;
        row.session_id = session.id().raw();
        let _ = self.persist_row(exec, &row);
        Ok(row)
    }

    /// Register one child drive with the scheduler. The scheduler op is the
    /// executor's handle on the drive; the drive itself runs the REAL
    /// AgentRuntime entry and records the child session's op id durably.
    fn submit_drive_op(
        &self,
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
        let writes =
            SchOwnershipSet::new(child.ownership_paths.clone()).canonicalized(&self.exec_root());
        let agent = self.agent.clone();
        let manager = self.manager.clone();
        let session_id = SessionId::new(child.session_id);
        let prompt = self.child_prompt(child)?;
        let model_override = child.model_policy.model.clone();
        let max_tokens = child.budget_max_tokens;
        let outcomes = {
            let guard = self.exec.lock().expect("exec lock");
            guard
                .as_ref()
                .expect("execution installed")
                .outcomes
                .clone()
        };
        let parent_session = {
            let guard = self.exec.lock().expect("exec lock");
            guard.as_ref().expect("execution installed").parent_session
        };
        let run_id = {
            let guard = self.exec.lock().expect("exec lock");
            guard.as_ref().expect("execution installed").run_id.clone()
        };
        let child_id = child.child_id.clone();
        let run = Arc::new(move || {
            let manager = manager.clone();
            let agent = agent.clone();
            let prompt = prompt.clone();
            let model_override = model_override.clone();
            let run_id = run_id.clone();
            let child_id = child_id.clone();
            let outcomes = outcomes.clone();
            drive_op_entry(
                manager,
                agent,
                session_id,
                prompt,
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
        let exec = guard.as_mut().expect("execution installed");
        exec.drive_ops.insert(child.child_id.clone(), op_id);
        Ok(())
    }

    fn child_prompt(&self, child: &ChildRuntime) -> Result<String, ExecError> {
        let guard = self.exec.lock().expect("exec lock");
        let exec = guard.as_ref().expect("execution installed");
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

    fn exec_root(&self) -> PathBuf {
        let guard = self.exec.lock().expect("exec lock");
        guard
            .as_ref()
            .map(|e| e.owner.root.clone())
            .unwrap_or_default()
    }

    fn final_outcome(&self) -> Result<PlanOutcome, ExecError> {
        let guard = self.exec.lock().expect("exec lock");
        let exec = guard.as_ref().expect("execution installed");
        let mut children: Vec<ChildRuntime> = exec.children.values().cloned().collect();
        children.sort_by_key(|c| c.created_ms);
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
        if s.ownership_paths.len() > 64 {
            return Err(ExecError::Oversized(
                "a child may declare at most 64 exclusive paths".into(),
            ));
        }
        for p in &s.ownership_paths {
            if p.is_empty() || p.chars().any(|c| c.is_control()) {
                return Err(ExecError::InvalidPlan(format!(
                    "exclusive ownership path {p:?} is not sane"
                )));
            }
        }
    }
    Ok(map)
}

fn derive_ownership(item: &WorkItem, plan: &crate::TaskPlan) -> Result<ChildOwnership, ExecError> {
    if !item.kind.is_mutating() {
        return Ok(ChildOwnership::ReadOnlyShared);
    }
    match &plan.ownership {
        crate::OwnershipModel::IsolatedWorktree => Ok(ChildOwnership::IsolatedWorktree),
        crate::OwnershipModel::DisjointPaths { paths } if !paths.is_empty() => {
            Ok(ChildOwnership::ExclusivePaths)
        }
        _ => Err(ExecError::InvalidPlan(format!(
            "mutating item {} requires IsolatedWorktree or DisjointPaths ownership",
            item.id
        ))),
    }
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
async fn drive_child_turn(
    manager: Arc<SessionManager>,
    agent: Arc<AgentRuntime>,
    session: SessionId,
    prompt: &str,
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
    let receipt = match agent.submit(session, prompt, &[]) {
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
            model_override,
            max_tokens,
        )
        .await;
        // Record the child session's REAL op id durably (registry row +
        // identity row) so re-attach maps the child to its op.
        if let Some(turn_op) = turn_op_id {
            if let Ok(Some(parent)) = manager.get_session(parent_session) {
                let key = format!("{run_id}/{child_id}");
                if let Ok(facts) = parent_facts(&parent) {
                    for (kind, k, value) in facts {
                        if kind == REGISTRY_ROW_KIND && k == key {
                            if let Ok(mut row) = serde_json::from_str::<ChildRuntime>(&value) {
                                row.operation_id = turn_op.raw();
                                if let Ok(value) = serde_json::to_string(&row) {
                                    let _ =
                                        parent.upsert_memory_fact(REGISTRY_ROW_KIND, &key, &value);
                                }
                            }
                            break;
                        }
                    }
                }
            }
            if let Ok(Some(child_handle)) = manager.get_session(session_id) {
                if let Ok(Some(mut identity)) = child_handle.orchestrator_child_identity_get() {
                    identity.operation_id = turn_op.raw();
                    let _ = child_handle.orchestrator_child_identity_put(&identity);
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
        Ok(())
    })
}

#[cfg(test)]
#[path = "runtime_tests.rs"]
mod runtime_tests;

/// Map a finished drive to the child's terminal state. A genuine end whose
/// OWN verification failed is a FAILED child — never a claimed complete.
fn classify_outcome(res: Result<TurnOutcome, String>) -> ChildState {
    let outcome = match res {
        Ok(o) => o,
        Err(_) => return ChildState::Failed,
    };
    match outcome.final_state {
        AgentState::Cancelled => ChildState::Cancelled,
        AgentState::ReadyForNextTurn | AgentState::Completed => {
            if outcome.acceptance == Some(faktor_agent::Acceptance::Fail)
                || matches!(
                    outcome.completion,
                    Some(faktor_agent::CompletionGate::FailedVerification { .. })
                )
            {
                ChildState::Failed
            } else {
                ChildState::Done
            }
        }
        _ => ChildState::Failed,
    }
}
