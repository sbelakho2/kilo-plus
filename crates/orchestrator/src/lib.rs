//! faktor-orchestrator — durable plan model and child-agent control plane for
//! multi-agent orchestration.
//!
//! This crate is a PURE planning/control model plus policy: it never spawns
//! processes, never touches sessions or files, and holds no I/O. The daemon
//! drives the real runtime from these structures later; everything enters
//! this crate as a method call on [`Orchestrator`] and every call leaves a
//! durable, timestamped event on the control log.
//!
//! Audit contract encoded here:
//! - Read-only children are the DEFAULT. Write access is an explicit,
//!   ownership-checked opt-in (`spawn_child(..., read_only: Some(false))`).
//! - A mutating child must work on a task whose kind writes (Implementation,
//!   Verification) under a plan that owns a disjoint write set or an isolated
//!   worktree. `NoWrites` plans reject mutating children.
//! - Per-child control is a state machine: pause/resume/cancel/retry/model
//!   changes are validated against the child's current state and every
//!   successful operation is recorded in the control log.
//! - Parallelism must be economic. [`parallelism_is_economic`] is a pure
//!   function implementing the documented threshold rule below.
//! - Bounded everything: ids are length- and charset-bounded, children are
//!   capped, steering notes and goals have hard limits.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

pub mod caps;
pub mod runtime;

/// Maximum length of a steering note, in characters.
pub const MAX_STEERING_NOTE_CHARS: usize = 500;
/// Maximum length of a plan goal, in characters.
pub const MAX_GOAL_CHARS: usize = 2000;
/// Maximum length of a work item id, in characters.
pub const MAX_ITEM_ID_CHARS: usize = 64;
/// Maximum length of a child model selector, in characters.
pub const MAX_MODEL_CHARS: usize = 128;
/// Economic-threshold constant for [`parallelism_is_economic`]: parallelizing
/// is only acceptable while `extra_cost_ratio <= PARALLELISM_ECONOMIC_FACTOR *
/// latency_savings_ratio`, i.e. the extra cost you pay must never exceed the
/// latency you buy back (factor 1.0 means cost and savings are compared 1:1).
pub const PARALLELISM_ECONOMIC_FACTOR: f64 = 1.0;
/// A parallel split must win at least this much on latency to be worth its
/// coordination overhead: `latency_savings_ratio >= MIN_LATENCY_WIN_FACTOR`.
pub const MIN_LATENCY_WIN_FACTOR: f64 = 1.5;
/// Error returned when a mutating child is requested for a task/plan that
/// does not own a write set.
pub const ERR_MUTATING_CHILD_NO_WRITES: &str =
    "mutating child requires disjoint write set or isolated worktree";
/// Error returned when changing the model of a Running child.
pub const ERR_RUNNING_MODEL_CHANGE: &str = "running child cannot change model — pause first";

/// What kind of work a [`WorkItem`] represents. Read-only kinds are
/// `Analysis`, `Exploration`, `Review`; mutating kinds are `Implementation`,
/// `Verification`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkKind {
    Analysis,
    Implementation,
    Verification,
    Exploration,
    Review,
}

impl WorkKind {
    /// Whether this kind of work writes to the workspace.
    pub fn is_mutating(self) -> bool {
        matches!(self, Self::Implementation | Self::Verification)
    }
}

/// Execution state of a [`WorkItem`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkState {
    Pending,
    Running,
    Paused,
    Blocked,
    Done,
    Failed,
    Cancelled,
}

impl WorkState {
    /// Terminal states cannot be left (except `Failed`, which may retry).
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Done | Self::Cancelled)
    }
}

/// Which write capability the plan owns. `NoWrites` is the read-only default;
/// anything else is an explicit grant of write capability to mutating items.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OwnershipModel {
    NoWrites,
    DisjointPaths { paths: Vec<String> },
    IsolatedWorktree,
}

impl OwnershipModel {
    /// Whether this model grants any write capability at all.
    pub fn allows_writes(&self) -> bool {
        !matches!(self, Self::NoWrites)
    }
}

/// One unit of planned work in a [`TaskPlan`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkItem {
    pub id: String,
    pub summary: String,
    pub depends_on: Vec<String>,
    pub kind: WorkKind,
    pub acceptance_checks: Vec<String>,
    pub completion: WorkState,
}

impl WorkItem {
    /// Builds a Pending item with no dependencies and no acceptance checks.
    pub fn new(id: impl Into<String>, summary: impl Into<String>, kind: WorkKind) -> Self {
        Self {
            id: id.into(),
            summary: summary.into(),
            depends_on: Vec::new(),
            kind,
            acceptance_checks: Vec::new(),
            completion: WorkState::Pending,
        }
    }
}

/// Durable multi-agent plan. Validate with [`TaskPlan::validate`] before
/// construction; an unvalidated plan is not guaranteed to be executable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskPlan {
    pub goal: String,
    pub non_goals: Vec<String>,
    pub constraints: Vec<String>,
    pub work_items: Vec<WorkItem>,
    pub ownership: OwnershipModel,
}

impl TaskPlan {
    /// Structural + ownership validation. Returns every violation found.
    ///
    /// Rules:
    /// 1. `goal` is bounded to [`MAX_GOAL_CHARS`] characters.
    /// 2. Work item ids are unique, 1..=[`MAX_ITEM_ID_CHARS`] characters,
    ///    ASCII printable (0x21..=0x7E), and contain no `/` or `\`.
    /// 3. Every `depends_on` references an existing work item id (no
    ///    dangling edges) and the dependency graph is acyclic.
    /// 4. A mutating item (Implementation/Verification) requires write
    ///    ownership (DisjointPaths with at least one path, or
    ///    IsolatedWorktree); with `NoWrites` it is rejected.
    /// 5. A read-only item (Analysis/Exploration/Review) MUST have `NoWrites`
    ///    ownership; a write-capable plan containing one is rejected.
    /// 6. Disjoint write paths must not overlap each other pairwise
    ///    (`"a/b"` and `"a"` overlap; so do duplicates).
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errs: Vec<String> = Vec::new();

        if self.goal.chars().count() > MAX_GOAL_CHARS {
            errs.push(format!(
                "goal exceeds {MAX_GOAL_CHARS} characters ({} given)",
                self.goal.chars().count()
            ));
        }

        let mut seen: HashSet<&str> = HashSet::new();
        let has_mutating_item;
        {
            let mut any_mutating = false;
            for item in &self.work_items {
                let id = item.id.as_str();
                if !is_valid_item_id(id) {
                    errs.push(format!(
                        "work item id {id:?} is invalid: expected 1..={MAX_ITEM_ID_CHARS} \
                         ASCII printable characters (0x21..=0x7E) without '/', '\\\\', \
                         whitespace, control, or non-ASCII bytes"
                    ));
                }
                if !seen.insert(id) {
                    errs.push(format!("duplicate work item id {id:?}"));
                }
                match (&self.ownership, item.kind.is_mutating()) {
                    (OwnershipModel::NoWrites, true) => {
                        errs.push(format!(
                            "mutating work item {id:?} ({:?}) requires write ownership \
                             (DisjointPaths or IsolatedWorktree)",
                            item.kind
                        ));
                    }
                    (own, false) if own.allows_writes() => {
                        errs.push(format!(
                            "read-only work item {id:?} ({:?}) requires NoWrites ownership",
                            item.kind
                        ));
                    }
                    _ => {}
                }
                any_mutating |= item.kind.is_mutating();
            }
            has_mutating_item = any_mutating;
        }

        for item in &self.work_items {
            for dep in &item.depends_on {
                if !seen.contains(dep.as_str()) {
                    errs.push(format!(
                        "work item {:?} depends on unknown work item {dep:?}",
                        item.id
                    ));
                }
            }
        }

        if let Some(list) = cycle_items(&self.work_items) {
            errs.push(format!(
                "dependency cycle detected among work items: {}",
                list.iter()
                    .map(|s| format!("{s:?}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        if let OwnershipModel::DisjointPaths { paths } = &self.ownership {
            if paths.is_empty() && has_mutating_item {
                errs.push(
                    "plan declares DisjointPaths ownership with no paths but contains \
                     mutating work items"
                        .to_string(),
                );
            }
            let norm: Vec<String> = paths
                .iter()
                .map(|p| p.trim_end_matches('/').to_string())
                .collect();
            for (i, p) in paths.iter().enumerate() {
                if norm[i].is_empty() {
                    errs.push(format!("empty disjoint ownership path {p:?}"));
                } else if !is_sane_path(&norm[i]) {
                    errs.push(format!(
                        "disjoint ownership path {p:?} contains whitespace, control, \
                         or non-ASCII characters"
                    ));
                }
            }
            for i in 0..norm.len() {
                for j in (i + 1)..norm.len() {
                    if norm[i].is_empty() || norm[j].is_empty() {
                        continue;
                    }
                    if paths_overlap(&norm[i], &norm[j]) {
                        errs.push(format!(
                            "disjoint ownership paths {p:?} and {q:?} overlap",
                            p = paths[i],
                            q = paths[j]
                        ));
                    }
                }
            }
        }

        if errs.is_empty() {
            Ok(())
        } else {
            Err(errs)
        }
    }
}

/// Runtime state of one child agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChildState {
    Running,
    Paused,
    /// The drive yielded at a safe boundary and waits for Resume (the
    /// runtime child mirror of the durable `Waiting` phase).
    Waiting,
    Cancelled,
    Done,
    Failed,
}

impl ChildState {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Cancelled | Self::Done | Self::Failed)
    }
}

/// One entry on a child's control log (durable, timestamped).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChildEvent {
    pub name: String,
    pub at_ms: u64,
}

/// A child agent tracked by the orchestrator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChildAgent {
    pub id: String,
    pub task_id: String,
    pub budget_micro: Option<u64>,
    pub permission_scope: String,
    pub state: ChildState,
    pub model: Option<String>,
    pub steering_note: Option<String>,
    pub control_log: Vec<ChildEvent>,
}

/// One entry on the orchestrator-wide log.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrchestratorEvent {
    pub child_id: String,
    pub name: String,
    pub at_ms: u64,
}

/// Control model over one validated [`TaskPlan`]: spawns children per work
/// item, drives per-child control state machines, records every operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Orchestrator {
    plan: TaskPlan,
    children: HashMap<String, ChildAgent>,
    log: Vec<OrchestratorEvent>,
    #[serde(default)]
    next_child_seq: u64,
}

impl Orchestrator {
    /// Validates `plan` and constructs an orchestrator over it.
    pub fn try_new(plan: TaskPlan) -> Result<Self, Vec<String>> {
        plan.validate()?;
        Ok(Self {
            plan,
            children: HashMap::new(),
            log: Vec::new(),
            next_child_seq: 0,
        })
    }

    /// The validated plan this orchestrator controls.
    pub fn plan(&self) -> &TaskPlan {
        &self.plan
    }

    /// Look up a work item by id.
    pub fn item(&self, id: &str) -> Option<&WorkItem> {
        self.plan.work_items.iter().find(|w| w.id == id)
    }

    /// Look up a child by id.
    pub fn inspect_child(&self, id: &str) -> Option<&ChildAgent> {
        self.children.get(id)
    }

    /// Orchestrator-wide event log, oldest first.
    pub fn events(&self) -> &[OrchestratorEvent] {
        &self.log
    }

    /// (child id, state, budget) for every child, sorted by id.
    pub fn children_overview(&self) -> Vec<(String, ChildState, Option<u64>)> {
        let mut out: Vec<(String, ChildState, Option<u64>)> = self
            .children
            .values()
            .map(|c| (c.id.clone(), c.state, c.budget_micro))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Spawns a child for work item `task_id`.
    ///
    /// `read_only` defaults to TRUE when `None` is passed: children are
    /// read-only unless the caller explicitly opts into writes. A mutating
    /// child (`read_only == Some(false)`) is only accepted when the task
    /// item is a mutating kind under write ownership; otherwise this fails
    /// with [`ERR_MUTATING_CHILD_NO_WRITES`].
    pub fn spawn_child(
        &mut self,
        task_id: &str,
        budget_micro: Option<u64>,
        permission_scope: String,
        model: Option<String>,
        read_only: Option<bool>,
    ) -> Result<ChildAgent, String> {
        if self.children.len() >= crate::runtime::ceilings::MAX_LIVE_CHILDREN {
            return Err(format!(
                "live child ceiling reached: cannot spawn more than {} live children (audit 24)",
                crate::runtime::ceilings::MAX_LIVE_CHILDREN
            ));
        }
        let item = self
            .plan
            .work_items
            .iter()
            .find(|w| w.id == task_id)
            .ok_or_else(|| format!("unknown work item '{task_id}'"))?;
        if let Some(m) = &model {
            if m.is_empty() {
                return Err("child model cannot be empty".to_string());
            }
            if m.chars().count() > MAX_MODEL_CHARS {
                return Err(format!("child model exceeds {MAX_MODEL_CHARS} characters"));
            }
        }
        let wants_writes = !read_only.unwrap_or(true);
        let allows_writes = item.kind.is_mutating() && self.plan.ownership.allows_writes();
        if wants_writes && !allows_writes {
            return Err(ERR_MUTATING_CHILD_NO_WRITES.to_string());
        }
        let at_ms = now_ms();
        let id = format!("child-{}", self.next_child_seq);
        self.next_child_seq += 1;
        let agent = ChildAgent {
            id: id.clone(),
            task_id: task_id.to_string(),
            budget_micro,
            permission_scope,
            state: ChildState::Running,
            model,
            steering_note: None,
            control_log: vec![ChildEvent {
                name: "spawned".to_string(),
                at_ms,
            }],
        };
        self.children.insert(id.clone(), agent.clone());
        self.log.push(OrchestratorEvent {
            child_id: id,
            name: "spawned".to_string(),
            at_ms,
        });
        Ok(agent)
    }

    /// Pauses a Running child. Only Running children can be paused.
    pub fn pause_child(&mut self, id: &str) -> Result<(), String> {
        let at_ms = now_ms();
        let child = self.child_mut(id)?;
        if child.state != ChildState::Running {
            return Err(format!(
                "cannot pause child '{id}': only Running children can be paused (state is {:?})",
                child.state
            ));
        }
        child.state = ChildState::Paused;
        push_child_event(child, "paused", at_ms);
        self.log.push(OrchestratorEvent {
            child_id: id.to_string(),
            name: "paused".to_string(),
            at_ms,
        });
        Ok(())
    }

    /// Resumes a Paused child. Only Paused children can be resumed.
    pub fn resume_child(&mut self, id: &str) -> Result<(), String> {
        let at_ms = now_ms();
        let child = self.child_mut(id)?;
        if child.state != ChildState::Paused {
            return Err(format!(
                "cannot resume child '{id}': only Paused children can be resumed (state is {:?})",
                child.state
            ));
        }
        child.state = ChildState::Running;
        push_child_event(child, "resumed", at_ms);
        self.log.push(OrchestratorEvent {
            child_id: id.to_string(),
            name: "resumed".to_string(),
            at_ms,
        });
        Ok(())
    }

    /// Cancels any non-terminal child (Running or Paused). Terminal children
    /// are immutable.
    pub fn cancel_child(&mut self, id: &str) -> Result<(), String> {
        let at_ms = now_ms();
        let child = self.child_mut(id)?;
        if child.state.is_terminal() {
            return Err(format!(
                "cannot cancel child '{id}': state {:?} is terminal",
                child.state
            ));
        }
        child.state = ChildState::Cancelled;
        push_child_event(child, "cancelled", at_ms);
        self.log.push(OrchestratorEvent {
            child_id: id.to_string(),
            name: "cancelled".to_string(),
            at_ms,
        });
        Ok(())
    }

    /// Steers a child with a guidance note. The note is bounded to
    /// [`MAX_STEERING_NOTE_CHARS`] characters; longer notes are rejected.
    /// Steering does not change state and is allowed in any state.
    pub fn steer_child(&mut self, id: &str, note: &str) -> Result<(), String> {
        if note.chars().count() > MAX_STEERING_NOTE_CHARS {
            return Err(format!(
                "steering note exceeds {MAX_STEERING_NOTE_CHARS} characters"
            ));
        }
        let at_ms = now_ms();
        let child = self.child_mut(id)?;
        child.steering_note = Some(note.to_string());
        push_child_event(child, "steered", at_ms);
        self.log.push(OrchestratorEvent {
            child_id: id.to_string(),
            name: "steered".to_string(),
            at_ms,
        });
        Ok(())
    }

    /// Retries a Failed child, returning it to Running. Only Failed children
    /// can be retried.
    pub fn retry_child(&mut self, id: &str) -> Result<(), String> {
        let at_ms = now_ms();
        let child = self.child_mut(id)?;
        if child.state != ChildState::Failed {
            return Err(format!(
                "cannot retry child '{id}': only Failed children can be retried (state is {:?})",
                child.state
            ));
        }
        child.state = ChildState::Running;
        push_child_event(child, "retried", at_ms);
        self.log.push(OrchestratorEvent {
            child_id: id.to_string(),
            name: "retried".to_string(),
            at_ms,
        });
        Ok(())
    }

    /// Changes the model of a child. A Running child cannot change model:
    /// pause first ([`ERR_RUNNING_MODEL_CHANGE`]).
    pub fn set_child_model(&mut self, id: &str, model: String) -> Result<(), String> {
        if model.is_empty() {
            return Err("child model cannot be empty".to_string());
        }
        if model.chars().count() > MAX_MODEL_CHARS {
            return Err(format!("child model exceeds {MAX_MODEL_CHARS} characters"));
        }
        let at_ms = now_ms();
        let child = self.child_mut(id)?;
        if child.state == ChildState::Running {
            return Err(ERR_RUNNING_MODEL_CHANGE.to_string());
        }
        child.model = Some(model);
        push_child_event(child, "model_changed", at_ms);
        self.log.push(OrchestratorEvent {
            child_id: id.to_string(),
            name: "model_changed".to_string(),
            at_ms,
        });
        Ok(())
    }

    /// Marks a Running or Paused child as Done (executor-reported).
    pub fn mark_child_done(&mut self, id: &str) -> Result<(), String> {
        self.mark_child(id, ChildState::Done, "marked_done", "done")
    }

    /// Marks a Running or Paused child as Failed (executor-reported).
    pub fn mark_child_failed(&mut self, id: &str) -> Result<(), String> {
        self.mark_child(id, ChildState::Failed, "marked_failed", "failed")
    }

    fn mark_child(
        &mut self,
        id: &str,
        target: ChildState,
        event_name: &str,
        verb: &str,
    ) -> Result<(), String> {
        let at_ms = now_ms();
        let child = self.child_mut(id)?;
        if !matches!(child.state, ChildState::Running | ChildState::Paused) {
            return Err(format!(
                "cannot mark child '{id}' {verb}: only Running or Paused children can be \
                 marked (state is {:?})",
                child.state
            ));
        }
        child.state = target;
        push_child_event(child, event_name, at_ms);
        self.log.push(OrchestratorEvent {
            child_id: id.to_string(),
            name: event_name.to_string(),
            at_ms,
        });
        Ok(())
    }

    /// Sets the completion state of a work item, validating the transition.
    ///
    /// Allowed transitions: `Pending -> {Running, Cancelled}`,
    /// `Running -> {Paused, Blocked, Done, Failed, Cancelled}`,
    /// `Paused -> {Running, Cancelled, Failed}`,
    /// `Blocked -> {Pending, Running, Failed}`, `Failed -> Pending`
    /// (retry). `Done` and `Cancelled` are terminal.
    pub fn set_item_state(&mut self, item_id: &str, state: WorkState) -> Result<(), String> {
        let item = self
            .plan
            .work_items
            .iter_mut()
            .find(|w| w.id == item_id)
            .ok_or_else(|| format!("unknown work item '{item_id}'"))?;
        if !can_transition(item.completion, state) {
            return Err(format!(
                "illegal work item state transition {:?} -> {state:?} for item '{item_id}'",
                item.completion
            ));
        }
        item.completion = state;
        Ok(())
    }

    /// Ids of ready items in plan insertion order: items whose state is
    /// Pending and whose every dependency is Done. Deterministic by
    /// construction order of the plan's `work_items`.
    pub fn ready_items(&self) -> Vec<String> {
        let state_of = |id: &str| {
            self.plan
                .work_items
                .iter()
                .find(|w| w.id == id)
                .map(|w| w.completion)
        };
        self.plan
            .work_items
            .iter()
            .filter(|w| {
                w.completion == WorkState::Pending
                    && w.depends_on
                        .iter()
                        .all(|d| state_of(d) == Some(WorkState::Done))
            })
            .map(|w| w.id.clone())
            .collect()
    }

    fn child_mut(&mut self, id: &str) -> Result<&mut ChildAgent, String> {
        self.children
            .get_mut(id)
            .ok_or_else(|| format!("unknown child '{id}'"))
    }
}

/// Pure economic check: does splitting `parallel_count` ways pay?
///
/// `latency_savings_ratio` is the latency win measured as a multiplier of
/// serial latency over parallel latency (`2.0` = twice as fast, i.e. half the
/// wall time). `extra_cost_ratio` is the multiplier of total compute cost
/// paid for parallelism (`1.2` = 20% extra work/coordination overhead; may be
/// `< 1.0` when serial execution would redundantly recompute).
///
/// Parallelizing is economic iff:
/// 1. `parallel_count >= 2` (one worker is never a parallelization), and
/// 2. the inputs are finite, and
/// 3. `extra_cost_ratio <= PARALLELISM_ECONOMIC_FACTOR * latency_savings_ratio`
///    — the extra cost paid must not exceed the latency bought back
///    (threshold constant [`PARALLELISM_ECONOMIC_FACTOR`] = 1.0), and
/// 4. `latency_savings_ratio >= MIN_LATENCY_WIN_FACTOR` — the latency win
///    must be at least 1.5x, otherwise coordination overhead is never worth
///    it.
pub fn parallelism_is_economic(
    parallel_count: usize,
    latency_savings_ratio: f64,
    extra_cost_ratio: f64,
) -> bool {
    if parallel_count < 2 {
        return false;
    }
    if !latency_savings_ratio.is_finite() || !extra_cost_ratio.is_finite() {
        return false;
    }
    extra_cost_ratio <= PARALLELISM_ECONOMIC_FACTOR * latency_savings_ratio
        && latency_savings_ratio >= MIN_LATENCY_WIN_FACTOR
}

fn push_child_event(child: &mut ChildAgent, name: &str, at_ms: u64) {
    child.control_log.push(ChildEvent {
        name: name.to_string(),
        at_ms,
    });
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn can_transition(from: WorkState, to: WorkState) -> bool {
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
        WorkState::Paused => matches!(
            to,
            WorkState::Running | WorkState::Cancelled | WorkState::Failed
        ),
        WorkState::Blocked => matches!(
            to,
            WorkState::Pending | WorkState::Running | WorkState::Failed
        ),
        WorkState::Failed => to == WorkState::Pending,
        WorkState::Done | WorkState::Cancelled => false,
    }
}

fn is_valid_item_id(id: &str) -> bool {
    let bytes = id.as_bytes();
    !bytes.is_empty()
        && bytes.len() <= MAX_ITEM_ID_CHARS
        && bytes
            .iter()
            .all(|&c| (0x21..=0x7E).contains(&c) && c != b'/' && c != b'\\')
}

fn is_sane_path(p: &str) -> bool {
    !p.is_empty() && p.is_ascii() && p.chars().all(|c| (c as u32) >= 0x21)
}

/// `"a/b"` and `"a"` overlap; `"a/b"` and `"ab"` do not.
fn paths_overlap(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    long.starts_with(short) && long.as_bytes().get(short.len()) == Some(&b'/')
}

/// Ids of items that are part of a cycle or unreachable behind one, in plan
/// order; `None` when the graph is acyclic.
fn cycle_items(items: &[WorkItem]) -> Option<Vec<&str>> {
    let n = items.len();
    let pos: HashMap<&str, usize> = items
        .iter()
        .enumerate()
        .map(|(i, w)| (w.id.as_str(), i))
        .collect();
    let mut indegree = vec![0usize; n];
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (from, w) in items.iter().enumerate() {
        for dep in &w.depends_on {
            if let Some(&to) = pos.get(dep.as_str()) {
                dependents[to].push(from);
                indegree[from] += 1;
            }
        }
    }
    let mut queue: VecDeque<usize> = (0..n).filter(|&i| indegree[i] == 0).collect();
    let mut done = vec![false; n];
    while let Some(i) = queue.pop_front() {
        if done[i] {
            continue;
        }
        done[i] = true;
        for &j in &dependents[i] {
            indegree[j] -= 1;
            if indegree[j] == 0 {
                queue.push_back(j);
            }
        }
    }
    let leftover: Vec<&str> = (0..n)
        .filter(|&i| !done[i])
        .map(|i| items[i].id.as_str())
        .collect();
    if leftover.is_empty() {
        None
    } else {
        Some(leftover)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wi(id: &str, kind: WorkKind, deps: &[&str]) -> WorkItem {
        WorkItem {
            id: id.to_string(),
            summary: format!("summary of {id}"),
            depends_on: deps.iter().map(|d| d.to_string()).collect(),
            kind,
            acceptance_checks: vec![],
            completion: WorkState::Pending,
        }
    }

    fn plan(ownership: OwnershipModel, items: Vec<WorkItem>) -> TaskPlan {
        TaskPlan {
            goal: "Ship the feature".to_string(),
            non_goals: vec![],
            constraints: vec![],
            work_items: items,
            ownership,
        }
    }

    fn errs_contain(errs: &[String], needle: &str) -> bool {
        errs.iter().any(|e| e.contains(needle))
    }

    fn validate_errs(p: &TaskPlan) -> Vec<String> {
        p.validate().expect_err("plan should fail validation")
    }

    #[test]
    fn dangling_dep_rejected() {
        let p = plan(
            OwnershipModel::NoWrites,
            vec![wi("a", WorkKind::Analysis, &["ghost"])],
        );
        let errs = validate_errs(&p);
        assert!(errs_contain(
            &errs,
            r#"work item "a" depends on unknown work item "ghost""#
        ));
    }

    #[test]
    fn cycle_rejected() {
        let p = plan(
            OwnershipModel::DisjointPaths {
                paths: vec!["src".to_string()],
            },
            vec![
                wi("a", WorkKind::Implementation, &["b"]),
                wi("b", WorkKind::Implementation, &["a"]),
            ],
        );
        let errs = validate_errs(&p);
        assert!(errs_contain(&errs, "dependency cycle"));
        assert!(errs_contain(&errs, "\"a\""));
        assert!(errs_contain(&errs, "\"b\""));

        let self_cycle = plan(
            OwnershipModel::DisjointPaths {
                paths: vec!["src".to_string()],
            },
            vec![wi("a", WorkKind::Implementation, &["a"])],
        );
        assert!(errs_contain(
            &validate_errs(&self_cycle),
            "dependency cycle"
        ));
    }

    #[test]
    fn mutating_item_without_ownership_rejected() {
        let p = plan(
            OwnershipModel::NoWrites,
            vec![wi("a", WorkKind::Implementation, &[])],
        );
        let errs = validate_errs(&p);
        assert!(errs_contain(&errs, "mutating work item"));
        assert!(errs_contain(&errs, "requires write ownership"));

        let v = plan(
            OwnershipModel::NoWrites,
            vec![wi("a", WorkKind::Verification, &[])],
        );
        assert!(errs_contain(&validate_errs(&v), "mutating work item"));
    }

    #[test]
    fn read_only_item_with_write_ownership_rejected() {
        for ownership in [
            OwnershipModel::DisjointPaths {
                paths: vec!["src".to_string()],
            },
            OwnershipModel::IsolatedWorktree,
        ] {
            for kind in [WorkKind::Analysis, WorkKind::Exploration, WorkKind::Review] {
                let p = plan(ownership.clone(), vec![wi("r", kind, &[])]);
                let errs = validate_errs(&p);
                assert!(errs_contain(&errs, "read-only work item"));
                assert!(errs_contain(&errs, "requires NoWrites ownership"));
            }
        }
    }

    #[test]
    fn valid_plans_accepted() {
        let read_only = plan(
            OwnershipModel::NoWrites,
            vec![
                wi("analysis", WorkKind::Analysis, &[]),
                wi("explore", WorkKind::Exploration, &["analysis"]),
                wi("review", WorkKind::Review, &["explore"]),
            ],
        );
        read_only
            .validate()
            .expect("read-only plan should validate");

        let writing = plan(
            OwnershipModel::DisjointPaths {
                paths: vec!["src/a".to_string(), "src/b".to_string()],
            },
            vec![
                wi("impl-a", WorkKind::Implementation, &[]),
                wi("verif-b", WorkKind::Verification, &["impl-a"]),
            ],
        );
        writing.validate().expect("writing plan should validate");

        let isolated = plan(
            OwnershipModel::IsolatedWorktree,
            vec![wi("impl", WorkKind::Implementation, &[])],
        );
        isolated.validate().expect("isolated plan should validate");

        let empty = TaskPlan {
            goal: "".to_string(),
            non_goals: vec![],
            constraints: vec![],
            work_items: vec![],
            ownership: OwnershipModel::NoWrites,
        };
        empty.validate().expect("empty plan should validate");
    }

    #[test]
    fn overlapping_disjoint_paths_rejected() {
        let cases: [Vec<String>; 5] = [
            vec!["src/a".to_string(), "src".to_string()],
            vec!["src".to_string(), "src/a".to_string()],
            vec!["src".to_string(), "src".to_string()],
            vec!["src/".to_string(), "src".to_string()],
            vec!["a/b/c".to_string(), "a/b".to_string()],
        ];
        for paths in cases {
            let p = plan(
                OwnershipModel::DisjointPaths {
                    paths: paths.clone(),
                },
                vec![wi("a", WorkKind::Implementation, &[])],
            );
            assert!(
                errs_contain(&validate_errs(&p), "overlap"),
                "expected overlap error for {paths:?}"
            );
        }
        let non_overlap = plan(
            OwnershipModel::DisjointPaths {
                paths: vec![
                    "src/foo".to_string(),
                    "src/bar".to_string(),
                    "src2".to_string(),
                    "ab".to_string(),
                ],
            },
            vec![wi("a", WorkKind::Implementation, &[])],
        );
        non_overlap
            .validate()
            .expect("non-overlapping paths should validate");
    }

    #[test]
    fn empty_disjoint_paths_with_mutating_items_rejected() {
        let p = plan(
            OwnershipModel::DisjointPaths { paths: vec![] },
            vec![wi("a", WorkKind::Implementation, &[])],
        );
        assert!(errs_contain(&validate_errs(&p), "no paths"));
    }

    #[test]
    fn hostile_ids_rejected_by_id_bound() {
        let long: String = "a".repeat(MAX_ITEM_ID_CHARS + 1);
        let p = plan(
            OwnershipModel::NoWrites,
            vec![
                wi(&long, WorkKind::Analysis, &[]),
                wi("a b", WorkKind::Analysis, &[]),
                wi("a/b", WorkKind::Analysis, &[]),
                wi("a\\b", WorkKind::Analysis, &[]),
                wi("caf\u{e9}", WorkKind::Analysis, &[]),
                wi("", WorkKind::Analysis, &[]),
            ],
        );
        let errs = validate_errs(&p);
        assert_eq!(errs.len(), 6);
        assert!(errs_contain(&errs, "is invalid"));

        let boundary = plan(
            OwnershipModel::NoWrites,
            vec![wi(&"b".repeat(MAX_ITEM_ID_CHARS), WorkKind::Analysis, &[])],
        );
        boundary.validate().expect("64-char id should validate");
    }

    #[test]
    fn duplicate_ids_rejected() {
        let p = plan(
            OwnershipModel::NoWrites,
            vec![
                wi("a", WorkKind::Analysis, &[]),
                wi("a", WorkKind::Analysis, &[]),
            ],
        );
        assert!(errs_contain(&validate_errs(&p), "duplicate work item id"));
    }

    #[test]
    fn goal_bounded() {
        let long_goal = "g".repeat(MAX_GOAL_CHARS + 1);
        let p = TaskPlan {
            goal: long_goal,
            non_goals: vec![],
            constraints: vec![],
            work_items: vec![],
            ownership: OwnershipModel::NoWrites,
        };
        assert!(errs_contain(
            &validate_errs(&p),
            "goal exceeds 2000 characters"
        ));

        let at_bound = TaskPlan {
            goal: "g".repeat(MAX_GOAL_CHARS),
            non_goals: vec![],
            constraints: vec![],
            work_items: vec![],
            ownership: OwnershipModel::NoWrites,
        };
        at_bound.validate().expect("2000-char goal should validate");
    }

    fn read_orch() -> Orchestrator {
        Orchestrator::try_new(plan(
            OwnershipModel::NoWrites,
            vec![wi("analysis", WorkKind::Analysis, &[])],
        ))
        .expect("plan should validate")
    }

    fn write_orch() -> Orchestrator {
        Orchestrator::try_new(plan(
            OwnershipModel::DisjointPaths {
                paths: vec!["src".to_string()],
            },
            vec![wi("impl", WorkKind::Implementation, &[])],
        ))
        .expect("plan should validate")
    }

    #[test]
    fn spawn_default_read_only_true() {
        let mut o = read_orch();
        let child = o
            .spawn_child("analysis", None, "read".to_string(), None, None)
            .expect("default (None) read_only=true must spawn on NoWrites item");
        assert_eq!(child.state, ChildState::Running);
        assert_eq!(child.task_id, "analysis");
        assert_eq!(o.inspect_child("child-0").unwrap().control_log.len(), 1);

        let explicit = o
            .spawn_child(
                "analysis",
                Some(5_000),
                "read".to_string(),
                None,
                Some(true),
            )
            .expect("Some(true) must spawn on NoWrites item");
        assert_eq!(explicit.budget_micro, Some(5_000));
        assert!(o
            .spawn_child("analysis", None, "read".to_string(), None, Some(true))
            .is_ok());
    }

    #[test]
    fn mutating_child_on_no_writes_item_rejected_exact_message() {
        let mut o = read_orch();
        let err = o
            .spawn_child("analysis", None, "write".to_string(), None, Some(false))
            .expect_err("mutating child on NoWrites item must fail");
        assert_eq!(err, ERR_MUTATING_CHILD_NO_WRITES);
        assert!(o.children_overview().is_empty());

        let mut w = write_orch();
        w.spawn_child("impl", None, "write".to_string(), None, Some(false))
            .expect("mutating child on write-owned item must spawn");
        let ro = w
            .spawn_child("impl", None, "write".to_string(), None, Some(true))
            .expect("read-only child on mutating item must spawn");
        assert_eq!(ro.state, ChildState::Running);
    }

    #[test]
    fn spawn_unknown_task_and_bad_model_rejected() {
        let mut o = read_orch();
        let err = o
            .spawn_child("ghost", None, "x".to_string(), None, Some(true))
            .expect_err("unknown task must fail");
        assert!(err.contains("unknown work item 'ghost'"));

        let err2 = o
            .spawn_child(
                "analysis",
                None,
                "x".to_string(),
                Some(String::new()),
                Some(true),
            )
            .expect_err("empty model must fail");
        assert!(err2.contains("cannot be empty"));

        let long_model = "m".repeat(MAX_MODEL_CHARS + 1);
        let err3 = o
            .spawn_child(
                "analysis",
                None,
                "x".to_string(),
                Some(long_model),
                Some(true),
            )
            .expect_err("overlong model must fail");
        assert!(err3.contains("exceeds"));
    }

    #[test]
    fn spawn_limited_to_live_child_ceiling() {
        let max = crate::runtime::ceilings::MAX_LIVE_CHILDREN;
        let mut w = write_orch();
        for _ in 0..max {
            w.spawn_child("impl", None, "write".to_string(), None, Some(false))
                .expect("spawn under limit");
        }
        let err = w
            .spawn_child("impl", None, "write".to_string(), None, Some(false))
            .expect_err("spawn over the live ceiling must fail");
        assert!(err.contains("live child ceiling"), "{err}");
        assert!(
            !err.contains("1000"),
            "the dead MAX_CHILDREN literal is gone: {err}"
        );
        assert_eq!(w.children_overview().len(), max);
        let ids: std::collections::HashSet<String> = w
            .children_overview()
            .iter()
            .map(|(id, _, _)| id.clone())
            .collect();
        for n in 0..max {
            assert!(
                ids.contains(&format!("child-{n}")),
                "child-{n} missing from the overview"
            );
        }
    }

    #[test]
    fn set_model_on_running_child_rejected() {
        let mut o = read_orch();
        o.spawn_child("analysis", None, "read".to_string(), None, None)
            .unwrap();
        let err = o
            .set_child_model("child-0", "gpt-5".to_string())
            .expect_err("running child must refuse model change");
        assert_eq!(err, ERR_RUNNING_MODEL_CHANGE);

        o.pause_child("child-0").unwrap();
        o.set_child_model("child-0", "gpt-5".to_string())
            .expect("paused child may change model");
        assert_eq!(
            o.inspect_child("child-0").unwrap().model.as_deref(),
            Some("gpt-5")
        );
        let events = &o.inspect_child("child-0").unwrap().control_log;
        assert_eq!(events.last().unwrap().name, "model_changed");
    }

    #[test]
    fn pause_resume_cancel_state_machine_honored() {
        let mut o = read_orch();
        o.spawn_child("analysis", None, "read".to_string(), None, None)
            .unwrap();
        let id = "child-0";

        o.pause_child(id).expect("Running -> Paused");
        assert_eq!(o.inspect_child(id).unwrap().state, ChildState::Paused);
        let err = o
            .pause_child(id)
            .expect_err("pause a Paused child must fail");
        assert!(err.contains("only Running children can be paused"));

        o.resume_child(id).expect("Paused -> Running");
        let err = o
            .resume_child(id)
            .expect_err("resume a Running child must fail");
        assert!(err.contains("only Paused children can be resumed"));

        o.cancel_child(id).expect("Running -> Cancelled");
        assert_eq!(o.inspect_child(id).unwrap().state, ChildState::Cancelled);
        let err = o
            .cancel_child(id)
            .expect_err("cancel a Cancelled child must fail");
        assert!(err.contains("terminal"));
        let err = o
            .pause_child(id)
            .expect_err("pause a Cancelled child must fail");
        assert!(err.contains("only Running children"));

        let id2 = "child-1";
        o.pause_child(id2).expect_err("unknown child");
        assert!(o
            .spawn_child("analysis", None, "read".to_string(), None, None)
            .is_ok());
        o.pause_child(id2).unwrap();
        o.mark_child_done(id2).expect("Paused -> Done");
        assert_eq!(o.inspect_child(id2).unwrap().state, ChildState::Done);
        let err = o
            .cancel_child(id2)
            .expect_err("cancel a Done child must fail");
        assert!(err.contains("terminal"));
        let err = o
            .pause_child(id2)
            .expect_err("pause a Done child must fail");
        assert!(err.contains("only Running children"));
        let err = o
            .resume_child(id2)
            .expect_err("resume a Done child must fail");
        assert!(err.contains("only Paused children"));

        let err = o.cancel_child("nope").expect_err("unknown child id");
        assert!(err.contains("unknown child 'nope'"));
    }

    #[test]
    fn steer_bounded_and_stored() {
        let mut o = read_orch();
        o.spawn_child("analysis", None, "read".to_string(), None, None)
            .unwrap();
        let id = "child-0";
        let max_note = "n".repeat(MAX_STEERING_NOTE_CHARS);
        o.steer_child(id, &max_note)
            .expect("500-char note must be stored");
        assert_eq!(
            o.inspect_child(id).unwrap().steering_note.as_deref(),
            Some(max_note.as_str())
        );

        let too_long = "n".repeat(MAX_STEERING_NOTE_CHARS + 1);
        let err = o
            .steer_child(id, &too_long)
            .expect_err("501-char note must be rejected");
        assert_eq!(
            err,
            format!("steering note exceeds {MAX_STEERING_NOTE_CHARS} characters")
        );
        assert_eq!(o.inspect_child(id).unwrap().control_log.len(), 2);
        assert_eq!(o.inspect_child(id).unwrap().control_log[1].name, "steered");

        o.cancel_child(id).unwrap();
        o.steer_child(id, "late steer note")
            .expect("steer allowed on terminal child");
        assert_eq!(
            o.inspect_child(id).unwrap().steering_note.as_deref(),
            Some("late steer note")
        );
    }

    #[test]
    fn retry_only_from_failed() {
        let mut o = read_orch();
        o.spawn_child("analysis", None, "read".to_string(), None, None)
            .unwrap();
        let id = "child-0";
        let err = o
            .retry_child(id)
            .expect_err("retry a Running child must fail");
        assert!(err.contains("only Failed children"));

        o.mark_child_failed(id).expect("Running -> Failed");
        assert_eq!(o.inspect_child(id).unwrap().state, ChildState::Failed);
        o.retry_child(id).expect("Failed -> Running");
        assert_eq!(o.inspect_child(id).unwrap().state, ChildState::Running);

        o.mark_child_done(id).unwrap();
        let err = o.retry_child(id).expect_err("retry a Done child must fail");
        assert!(err.contains("only Failed children"));
        let err = o
            .mark_child_failed(id)
            .expect_err("mark a Done child failed must fail");
        assert!(err.contains("only Running or Paused"));

        let log_names: Vec<&str> = o
            .inspect_child(id)
            .unwrap()
            .control_log
            .iter()
            .map(|e| e.name.as_str())
            .collect();
        assert_eq!(
            log_names,
            vec!["spawned", "marked_failed", "retried", "marked_done"]
        );
        let at_ms: Vec<u64> = o
            .inspect_child(id)
            .unwrap()
            .control_log
            .iter()
            .map(|e| e.at_ms)
            .collect();
        assert!(
            at_ms.windows(2).all(|w| w[0] <= w[1]),
            "log timestamps must be non-decreasing"
        );
    }

    #[test]
    fn item_state_machine_and_ready_items_ordering() {
        let mut o = Orchestrator::try_new(plan(
            OwnershipModel::DisjointPaths {
                paths: vec!["src".to_string()],
            },
            vec![
                wi("a", WorkKind::Implementation, &[]),
                wi("b", WorkKind::Implementation, &["a"]),
                wi("x", WorkKind::Implementation, &[]),
                wi("c", WorkKind::Implementation, &["a"]),
                wi("d", WorkKind::Verification, &["b"]),
            ],
        ))
        .expect("plan should validate");

        assert_eq!(o.ready_items(), vec!["a", "x"]);
        let err = o
            .set_item_state("d", WorkState::Done)
            .expect_err("Pending -> Done must be illegal");
        assert!(err.contains("illegal work item state transition"));
        let err = o
            .set_item_state("ghost", WorkState::Done)
            .expect_err("unknown item");
        assert!(err.contains("unknown work item 'ghost'"));

        o.set_item_state("a", WorkState::Running).unwrap();
        o.set_item_state("a", WorkState::Done).unwrap();
        assert_eq!(o.ready_items(), vec!["b", "x", "c"]);

        o.set_item_state("b", WorkState::Running).unwrap();
        o.set_item_state("b", WorkState::Done).unwrap();
        assert_eq!(o.ready_items(), vec!["x", "c", "d"]);

        o.set_item_state("x", WorkState::Running).unwrap();
        o.set_item_state("x", WorkState::Paused).unwrap();
        o.set_item_state("x", WorkState::Running).unwrap();
        o.set_item_state("x", WorkState::Blocked).unwrap();
        o.set_item_state("x", WorkState::Pending).unwrap();
        o.set_item_state("x", WorkState::Cancelled).unwrap();
        assert_eq!(o.ready_items(), vec!["c", "d"]);

        let err = o
            .set_item_state("x", WorkState::Pending)
            .expect_err("Cancelled is terminal");
        assert!(err.contains("illegal"));
        o.set_item_state("c", WorkState::Running).unwrap();
        o.set_item_state("c", WorkState::Failed).unwrap();
        o.set_item_state("c", WorkState::Pending).unwrap();
        o.set_item_state("c", WorkState::Running).unwrap();
        o.set_item_state("c", WorkState::Done).unwrap();
        assert_eq!(o.ready_items(), vec!["d"]);
    }

    #[test]
    fn economic_check_both_outcomes() {
        assert!(parallelism_is_economic(3, 2.0, 1.2));
        assert!(parallelism_is_economic(2, MIN_LATENCY_WIN_FACTOR, 0.0));
        assert!(parallelism_is_economic(
            4,
            2.0,
            PARALLELISM_ECONOMIC_FACTOR * 2.0
        ));

        assert!(!parallelism_is_economic(3, 2.0, 2.0 + 0.1));
        assert!(!parallelism_is_economic(3, 1.4, 0.1));
        assert!(!parallelism_is_economic(1, 100.0, 0.0));
        assert!(!parallelism_is_economic(0, 100.0, 0.0));
        assert!(!parallelism_is_economic(2, f64::NAN, 0.0));
        assert!(!parallelism_is_economic(2, 2.0, f64::NAN));
        assert!(!parallelism_is_economic(2, f64::INFINITY, 0.0));
        assert!(!parallelism_is_economic(2, 2.0, f64::NEG_INFINITY));
    }

    #[test]
    fn overview_and_inspect_reflect_control() {
        let mut w = write_orch();
        w.spawn_child("impl", Some(1_000), "src".to_string(), None, Some(false))
            .unwrap();
        w.spawn_child("impl", None, "read".to_string(), None, None)
            .unwrap();
        w.pause_child("child-1").unwrap();
        let overview = w.children_overview();
        assert_eq!(
            overview,
            vec![
                ("child-0".to_string(), ChildState::Running, Some(1_000)),
                ("child-1".to_string(), ChildState::Paused, None),
            ]
        );
        let events = w.events();
        let names: Vec<&str> = events.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, vec!["spawned", "spawned", "paused"]);
        assert_eq!(events[2].child_id, "child-1");
    }

    #[test]
    fn durable_serde_round_trip() {
        let mut w = write_orch();
        w.spawn_child(
            "impl",
            Some(42),
            "src".to_string(),
            Some("m1".to_string()),
            Some(false),
        )
        .unwrap();
        w.pause_child("child-0").unwrap();
        w.steer_child("child-0", "go slower").unwrap();
        w.set_item_state("impl", WorkState::Running).unwrap();

        let json = serde_json::to_string(&w).expect("serialize");
        let mut restored: Orchestrator = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.plan(), w.plan());
        assert_eq!(restored.children_overview(), w.children_overview());
        assert_eq!(restored.events().len(), 3);
        let child = restored.inspect_child("child-0").unwrap();
        assert_eq!(child.state, ChildState::Paused);
        assert_eq!(child.steering_note.as_deref(), Some("go slower"));
        assert_eq!(child.control_log.len(), 3);

        let next = restored
            .spawn_child("impl", None, "read".to_string(), None, None)
            .expect("counter survives round trip");
        assert_eq!(next.id, "child-1");
        assert_eq!(restored.ready_items(), Vec::<String>::new());
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepOutcome {
    Passed,
    Failed { reason: String },
    VerificationFailed { details: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepResult {
    pub outcome: StepOutcome,
    pub cost_micro: u64,
}

pub trait StepExecutor {
    fn run_step(&mut self, item: &WorkItem) -> StepResult;
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BudgetLedger {
    pub reserved_micro: u64,
    pub spent_micro: u64,
    pub spent_by: Vec<(String, u64)>,
    pub overspent: Vec<(String, u64)>,
}

impl BudgetLedger {
    pub fn reserve(&mut self, amount: u64, total_budget: Option<u64>) -> Result<(), String> {
        if let Some(total) = total_budget {
            let next = self
                .reserved_micro
                .checked_add(amount)
                .ok_or_else(|| format!("reserving {amount} micro overflows the ledger"))?;
            if next > total {
                return Err(format!(
                    "reserving {amount} micro exceeds total budget {total} micro; already \
                     reserved {}",
                    self.reserved_micro
                ));
            }
            self.reserved_micro = next;
        } else {
            self.reserved_micro = self.reserved_micro.saturating_add(amount);
        }
        Ok(())
    }

    pub fn settle(&mut self, item: &str, reserved: u64, actual: u64) {
        self.spent_micro = self.spent_micro.saturating_add(actual);
        self.spent_by.push((item.to_string(), actual));
        if actual > reserved {
            self.overspent.push((item.to_string(), actual));
        }
    }

    pub fn remaining(&self, total: Option<u64>) -> Option<u64> {
        total.map(|t| t.saturating_sub(self.reserved_micro))
    }
}

pub fn estimate_cost(item: &WorkItem) -> u64 {
    if item.kind.is_mutating() {
        1_000
    } else {
        100
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Completion {
    Complete,
    Blocked(Vec<String>),
    Pending,
}

#[derive(Debug)]
pub struct OrchestratorExecutor {
    plan: TaskPlan,
    budget_micro: Option<u64>,
    ledger: BudgetLedger,
    state: HashMap<String, WorkState>,
    results: HashMap<String, StepResult>,
}

impl OrchestratorExecutor {
    pub fn try_new(plan: TaskPlan, budget_micro: Option<u64>) -> Result<Self, Vec<String>> {
        plan.validate()?;
        let state = plan
            .work_items
            .iter()
            .map(|w| (w.id.clone(), w.completion))
            .collect();
        Ok(Self {
            plan,
            budget_micro,
            ledger: BudgetLedger::default(),
            state,
            results: HashMap::new(),
        })
    }

    pub fn next_ready(&self) -> Vec<String> {
        self.plan
            .work_items
            .iter()
            .filter(|w| {
                self.state.get(&w.id) == Some(&WorkState::Pending)
                    && w.depends_on
                        .iter()
                        .all(|d| self.state.get(d) == Some(&WorkState::Done))
            })
            .map(|w| w.id.clone())
            .collect()
    }

    pub fn execute_next(&mut self, ex: &mut dyn StepExecutor) -> Result<Vec<String>, String> {
        let ready = self.next_ready();
        let mut executed: Vec<String> = Vec::new();
        for id in ready {
            let Some(item) = self.plan.work_items.iter().find(|w| w.id == id) else {
                continue;
            };
            let est = estimate_cost(item);
            self.ledger
                .reserve(est, self.budget_micro)
                .map_err(|_| format!("budget: items already executed {}", executed.len()))?;
            let result = ex.run_step(item);
            self.ledger.settle(&id, est, result.cost_micro);
            match &result.outcome {
                StepOutcome::Passed => {
                    self.state.insert(id.clone(), WorkState::Done);
                }
                StepOutcome::Failed { .. } | StepOutcome::VerificationFailed { .. } => {
                    self.state.insert(id.clone(), WorkState::Failed);
                    for dependent in &self.plan.work_items {
                        if dependent.depends_on.contains(&id)
                            && self.state.get(&dependent.id) == Some(&WorkState::Pending)
                        {
                            self.state.insert(dependent.id.clone(), WorkState::Blocked);
                        }
                    }
                }
            }
            self.results.insert(id.clone(), result);
            executed.push(id);
        }
        Ok(executed)
    }

    pub fn completion(&self) -> Completion {
        let all_done = self
            .plan
            .work_items
            .iter()
            .all(|w| self.state.get(&w.id) == Some(&WorkState::Done));
        if all_done {
            return Completion::Complete;
        }
        let failed: Vec<String> = self
            .plan
            .work_items
            .iter()
            .filter(|w| {
                self.results.get(&w.id).is_some_and(|r| {
                    matches!(
                        &r.outcome,
                        StepOutcome::Failed { .. } | StepOutcome::VerificationFailed { .. }
                    )
                })
            })
            .map(|w| w.id.clone())
            .collect();
        if failed.is_empty() {
            Completion::Pending
        } else {
            Completion::Blocked(failed)
        }
    }

    pub fn ledger(&self) -> &BudgetLedger {
        &self.ledger
    }

    pub fn results(&self) -> &HashMap<String, StepResult> {
        &self.results
    }
}

#[cfg(test)]
mod executor_tests {
    use super::*;
    use std::collections::HashMap;

    struct Scripted {
        outcomes: HashMap<String, StepOutcome>,
        cost_micro: u64,
        runs: usize,
    }

    impl Scripted {
        fn passing(cost_micro: u64) -> Self {
            Self {
                outcomes: HashMap::new(),
                cost_micro,
                runs: 0,
            }
        }

        fn failing(cost_micro: u64, id: &str, outcome: StepOutcome) -> Self {
            let mut outcomes = HashMap::new();
            outcomes.insert(id.to_string(), outcome);
            Self {
                outcomes,
                cost_micro,
                runs: 0,
            }
        }
    }

    impl StepExecutor for Scripted {
        fn run_step(&mut self, item: &WorkItem) -> StepResult {
            self.runs += 1;
            StepResult {
                outcome: self
                    .outcomes
                    .get(&item.id)
                    .cloned()
                    .unwrap_or(StepOutcome::Passed),
                cost_micro: self.cost_micro,
            }
        }
    }

    fn wi(id: &str, kind: WorkKind, deps: &[&str]) -> WorkItem {
        WorkItem {
            id: id.to_string(),
            summary: format!("summary of {id}"),
            depends_on: deps.iter().map(|d| d.to_string()).collect(),
            kind,
            acceptance_checks: Vec::new(),
            completion: WorkState::Pending,
        }
    }

    fn write_plan(items: Vec<WorkItem>) -> TaskPlan {
        TaskPlan {
            goal: "Ship the feature".to_string(),
            non_goals: Vec::new(),
            constraints: Vec::new(),
            work_items: items,
            ownership: OwnershipModel::DisjointPaths {
                paths: vec!["src".to_string()],
            },
        }
    }

    fn trace(o: &mut OrchestratorExecutor, s: &mut Scripted) -> Vec<Vec<String>> {
        let mut batches: Vec<Vec<String>> = Vec::new();
        loop {
            if o.completion() == Completion::Complete {
                break;
            }
            let batch = o.execute_next(s).expect("trace runs without a budget");
            if batch.is_empty() {
                break;
            }
            batches.push(batch);
        }
        batches
    }

    #[test]
    fn try_new_rejects_invalid_plan() {
        let p = write_plan(vec![wi("a", WorkKind::Implementation, &["ghost"])]);
        let errs = OrchestratorExecutor::try_new(p, None)
            .expect_err("plan with dangling dependency must be rejected");
        assert!(!errs.is_empty());
        assert!(errs.iter().any(|e| e.contains("depends on unknown")));
    }

    #[test]
    fn ready_order_is_plan_order_after_dependency_gate() {
        let p = write_plan(vec![
            wi("a", WorkKind::Implementation, &[]),
            wi("b", WorkKind::Implementation, &["a"]),
            wi("c", WorkKind::Implementation, &["a"]),
            wi("d", WorkKind::Verification, &["b", "c"]),
        ]);
        let mut o = OrchestratorExecutor::try_new(p, None).unwrap();
        let mut s = Scripted::passing(1_000);
        assert_eq!(o.next_ready(), vec!["a"]);
        assert_eq!(o.execute_next(&mut s).unwrap(), vec!["a"]);
        assert_eq!(o.next_ready(), vec!["b", "c"]);
        assert_eq!(o.execute_next(&mut s).unwrap(), vec!["b", "c"]);
        assert_eq!(o.next_ready(), vec!["d"]);
        assert_eq!(o.execute_next(&mut s).unwrap(), vec!["d"]);
        assert_eq!(o.completion(), Completion::Complete);
        assert_eq!(s.runs, 4);
    }

    #[test]
    fn failed_item_blocks_pending_dependents_and_lists_in_completion() {
        let p = write_plan(vec![
            wi("a", WorkKind::Implementation, &[]),
            wi("b", WorkKind::Implementation, &["a"]),
        ]);
        let mut o = OrchestratorExecutor::try_new(p, None).unwrap();
        let mut s = Scripted::failing(
            1_000,
            "a",
            StepOutcome::Failed {
                reason: "boom".to_string(),
            },
        );
        assert_eq!(o.execute_next(&mut s).unwrap(), vec!["a"]);
        assert_eq!(o.next_ready(), Vec::<String>::new());
        assert_eq!(o.execute_next(&mut s).unwrap(), Vec::<String>::new());
        assert_eq!(s.runs, 1);
        assert_eq!(o.completion(), Completion::Blocked(vec!["a".to_string()]));
        let r = o.results().get("a").expect("a has a result");
        assert!(matches!(&r.outcome, StepOutcome::Failed { .. }));
        assert!(!o.results().contains_key("b"));
    }

    #[test]
    fn verification_failure_blocks_and_never_completes() {
        let p = write_plan(vec![
            wi("impl", WorkKind::Implementation, &[]),
            wi("verify", WorkKind::Verification, &["impl"]),
        ]);
        let mut o = OrchestratorExecutor::try_new(p, None).unwrap();
        let mut s = Scripted::failing(
            1_000,
            "verify",
            StepOutcome::VerificationFailed {
                details: "acceptance checks failed".to_string(),
            },
        );
        o.execute_next(&mut s).expect("impl must run");
        assert_eq!(o.next_ready(), vec!["verify"]);
        o.execute_next(&mut s).expect("verify must run");
        assert_eq!(
            o.completion(),
            Completion::Blocked(vec!["verify".to_string()])
        );
        for _ in 0..3 {
            assert!(o.execute_next(&mut s).unwrap().is_empty());
            assert_ne!(o.completion(), Completion::Complete);
        }
        assert_eq!(
            o.completion(),
            Completion::Blocked(vec!["verify".to_string()])
        );
    }

    #[test]
    fn complete_only_when_every_item_passed() {
        let p = write_plan(vec![
            wi("a", WorkKind::Implementation, &[]),
            wi("b", WorkKind::Implementation, &[]),
            wi("c", WorkKind::Implementation, &[]),
        ]);
        let mut o = OrchestratorExecutor::try_new(p.clone(), None).unwrap();
        let mut s = Scripted::passing(1_000);
        o.execute_next(&mut s).unwrap();
        assert_eq!(o.completion(), Completion::Complete);
        assert_eq!(o.results().len(), 3);

        let mut o = OrchestratorExecutor::try_new(p, None).unwrap();
        let mut s = Scripted::failing(
            1_000,
            "b",
            StepOutcome::Failed {
                reason: "nope".to_string(),
            },
        );
        assert_eq!(o.execute_next(&mut s).unwrap(), vec!["a", "b", "c"]);
        assert_ne!(o.completion(), Completion::Complete);
        assert_eq!(o.completion(), Completion::Blocked(vec!["b".to_string()]));
    }

    #[test]
    fn hard_budget_fails_before_running_but_unbounded_succeeds() {
        let p = write_plan(vec![wi("a", WorkKind::Implementation, &[])]);
        let mut o = OrchestratorExecutor::try_new(p.clone(), Some(150)).unwrap();
        let mut s = Scripted::passing(1_000);
        let err = o
            .execute_next(&mut s)
            .expect_err("1000 estimate must exceed 150 budget");
        assert!(err.contains("budget: items already executed 0"), "{err}");
        assert_eq!(s.runs, 0);
        assert!(o.results().is_empty());
        assert_eq!(o.next_ready(), vec!["a"]);
        assert_eq!(o.ledger().reserved_micro, 0);
        assert_eq!(o.ledger().spent_micro, 0);
        assert_eq!(o.ledger().remaining(Some(150)), Some(150));

        let mut unbounded = OrchestratorExecutor::try_new(p.clone(), None).unwrap();
        let mut s = Scripted::passing(1_000);
        assert_eq!(unbounded.execute_next(&mut s).unwrap(), vec!["a"]);
        assert_eq!(unbounded.completion(), Completion::Complete);

        let mut exact = OrchestratorExecutor::try_new(p, Some(1_000)).unwrap();
        let mut s = Scripted::passing(1_000);
        assert_eq!(exact.execute_next(&mut s).unwrap(), vec!["a"]);
        assert_eq!(exact.completion(), Completion::Complete);
        assert_eq!(exact.ledger().remaining(Some(1_000)), Some(0));
    }

    #[test]
    fn actual_cost_over_estimate_is_recorded_as_overspend() {
        let p = write_plan(vec![wi("a", WorkKind::Implementation, &[])]);
        let mut o = OrchestratorExecutor::try_new(p, None).unwrap();
        let mut s = Scripted::passing(5_000);
        o.execute_next(&mut s).unwrap();
        let ledger = o.ledger();
        assert_eq!(ledger.reserved_micro, 1_000);
        assert_eq!(ledger.spent_micro, 5_000);
        assert_eq!(ledger.spent_by, vec![("a".to_string(), 5_000)]);
        assert_eq!(ledger.overspent, vec![("a".to_string(), 5_000)]);
        assert_eq!(o.completion(), Completion::Complete);
    }

    #[test]
    fn empty_plan_is_immediately_complete() {
        let p = TaskPlan {
            goal: String::new(),
            non_goals: Vec::new(),
            constraints: Vec::new(),
            work_items: Vec::new(),
            ownership: OwnershipModel::NoWrites,
        };
        let mut o = OrchestratorExecutor::try_new(p, Some(1)).unwrap();
        assert_eq!(o.completion(), Completion::Complete);
        assert_eq!(o.next_ready(), Vec::<String>::new());
        let mut s = Scripted::passing(100);
        assert_eq!(o.execute_next(&mut s).unwrap(), Vec::<String>::new());
        assert_eq!(s.runs, 0);
    }

    #[test]
    fn two_identical_runs_are_identical() {
        let p = write_plan(vec![
            wi("a", WorkKind::Implementation, &[]),
            wi("b", WorkKind::Implementation, &["a"]),
            wi("x", WorkKind::Implementation, &[]),
            wi("y", WorkKind::Implementation, &["x"]),
            wi("z", WorkKind::Implementation, &["y", "b"]),
        ]);
        let failure = StepOutcome::Failed {
            reason: "deterministic".to_string(),
        };
        let mut first = OrchestratorExecutor::try_new(p.clone(), None).unwrap();
        let mut second = OrchestratorExecutor::try_new(p, None).unwrap();
        let mut s1 = Scripted::failing(1_000, "x", failure.clone());
        let mut s2 = Scripted::failing(1_000, "x", failure);
        let t1 = trace(&mut first, &mut s1);
        let t2 = trace(&mut second, &mut s2);
        assert_eq!(t1, t2);
        assert_eq!(s1.runs, s2.runs);
        assert_eq!(first.completion(), second.completion());
        assert_eq!(first.ledger(), second.ledger());
        assert_eq!(first.results(), second.results());
    }

    #[test]
    fn chain_of_200_items_runs_each_step_at_most_once() {
        let mut items = Vec::new();
        for i in 0..200usize {
            let deps = if i == 0 {
                Vec::new()
            } else {
                vec![format!("item-{}", i - 1)]
            };
            items.push(WorkItem {
                id: format!("item-{i}"),
                summary: format!("step {i}"),
                depends_on: deps,
                kind: WorkKind::Implementation,
                acceptance_checks: Vec::new(),
                completion: WorkState::Pending,
            });
        }
        let p = write_plan(items);
        let mut o = OrchestratorExecutor::try_new(p, None).unwrap();
        let mut s = Scripted::passing(1_000);
        let mut total = 0usize;
        loop {
            if o.completion() == Completion::Complete {
                break;
            }
            let batch = o.execute_next(&mut s).expect("runs without a budget");
            assert!(!batch.is_empty(), "pending plan must make progress");
            assert!(batch.len() <= 1);
            total += batch.len();
            assert!(total <= 200, "no item may run twice");
        }
        assert_eq!(s.runs, 200);
        assert_eq!(o.results().len(), 200);
        assert_eq!(total, 200);
        assert_eq!(o.completion(), Completion::Complete);
        assert_eq!(o.ledger().reserved_micro, 200_000);
    }
}
