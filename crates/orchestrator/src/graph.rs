//! The single durable operation graph read-model (audit 93).
//!
//! A parent task plan plus EVERY child operation of a run — session/op
//! ids, worktree, ownership, state, budget, effective capabilities, plan
//! step linkage, steering history and merge outcomes — assembled as ONE
//! queryable graph. The graph is a pure READ over the durable rows the
//! wave-12/13 runtime already writes:
//!
//! - the plan row (kind `orchestrator_plan`, key = run id) under the
//!   parent session supplies the root node (plan id, goal, steps);
//! - the registry rows (kind `orchestrator_registry`, key
//!   `<run>/<child_id>`) supply every child node; parent linkage needs no
//!   extra row because the plan row names the work items and each registry
//!   row names its item id + parent session + run (plan_step_index is the
//!   item's deterministic position in the plan's `work_items`);
//! - the child session's control rows (kind `orchestrator_ctl`, key
//!   `seq-*`) supply the steering history with exactly-once applied
//!   timestamps;
//! - the merge envelopes + decision/outcome parts (kinds
//!   `orchestrator_merge` / `orchestrator_merge_part`) supply the merge
//!   summary of the child's latest merge.
//!
//! Ordering is deterministic: children sort by plan step order first, then
//! spawn order (durable `created_ms`, then child id). The root state and
//! per-step states are DERIVED from the durable child rows with exactly
//! the semantics of executor re-attach ([`crate::runtime`]'s
//! `reconcile_from_registry`): the graph of a crashed run equals the graph
//! the re-attached executor sees, and both survive any number of manager
//! reopens. Where the parent session holds several runs the graph is
//! ambiguous and refuses loudly (a Conflict naming the runs) — a single
//! graph needs a single plan.

use std::collections::BTreeSet;
use std::path::PathBuf;

use faktor_core::id::SessionId;
use faktor_session::child::{ChildControl, ChildOwnership, PresentationState};

use super::merge::{
    merge_envelopes, parent_handle, read_part_conflicts, read_part_paths, scan_facts,
};
use super::*;

/// Hard cap on one graph: beyond this many durable child rows the
/// assembly refuses loudly (bounded everything) instead of returning an
/// unbounded projection.
pub const MAX_GRAPH_CHILDREN: usize = 256;

/// One plan step of the root node (plan order).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GraphWorkItem {
    pub item_id: String,
    pub kind: WorkKind,
    /// Derived from the durable child rows (re-attach semantics).
    pub state: WorkState,
}

/// Root node of the graph: the durable plan.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GraphRootNode {
    /// The durable run id (the plan row key).
    pub plan_id: String,
    pub goal: String,
    /// Derived plan state, documented in [`derived_root_state`].
    pub state: WorkState,
    /// Every plan step with its derived state, in plan order.
    pub work_items: Vec<GraphWorkItem>,
}

/// One steering/control message applied (or pending) on a child.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SteerEvent {
    pub kind: ChildControl,
    pub seq: u64,
    /// `None` = enqueued, not yet applied (exactly-once ack semantics).
    pub applied_ms: Option<i64>,
}

/// The latest durable merge of one child.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GraphMerge {
    pub change_set_id: String,
    pub merged: Vec<PathBuf>,
    pub rejected: Vec<PathBuf>,
    pub conflicts: Vec<(PathBuf, String)>,
}

/// One child node of the graph: the durable runtime row + steering +
/// merge outcome.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GraphChildNode {
    pub child_id: String,
    /// The REAL child session row id.
    pub session_id: u64,
    /// The child's durable turn operation id (0 before its first submit).
    pub operation_id: u64,
    pub worktree_id: u64,
    pub ownership: ChildOwnership,
    pub state: ChildState,
    /// Durable presentation/attention state of the child, folded from the
    /// child session's typed ledger (latest `ChildPresentationChanged`
    /// entry wins; `Foreground` when the child never transitioned). Purely
    /// presentational: it never affects scheduling ownership, budgets or
    /// lineage. Additive with a serde default so graphs persisted by older
    /// readers decode unchanged.
    #[serde(default)]
    pub presentation: PresentationState,
    /// Durable token budget cap (None = unlimited).
    pub budget: Option<u64>,
    /// Effective capability set (parent ∩ task ∩ child).
    pub capabilities: CapabilitySet,
    /// The position of this child's work item in the plan's `work_items`
    /// (None = a child without a plan item, e.g. a reviewer).
    pub plan_step_index: Option<usize>,
    /// Durable steering history, oldest first (seq order).
    pub steer_events: Vec<SteerEvent>,
    /// The latest durable merge record of the child, when one exists.
    pub merge: Option<GraphMerge>,
}

/// ONE durable operation graph: root plan + children, deterministic order.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OpGraph {
    pub root: GraphRootNode,
    pub children: Vec<GraphChildNode>,
}

/// Typed graph-assembly error (wave A3). Identity failures are DISTINCT
/// variants so callers can match them precisely: a missing assignment, a
/// deleted child row, tampered/duplicate assignment rows — none of these
/// is ever a silent skip or a fabricated id.
#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("run {run_id:?}: plan item {item_id:?} has no durable work-item assignment (identity is never fabricated from memory)")]
    MissingAssignment { run_id: String, item_id: String },
    #[error("run {run_id:?}: the assignment names child {child_id:?}, whose spawn is durable (its env binding exists) but whose child row is gone")]
    MissingChild { run_id: String, child_id: String },
    #[error("graph conflict: {0}")]
    Conflict(String),
    #[error("graph not found: {0}")]
    NotFound(String),
    #[error("graph oversized: {0}")]
    Oversized(String),
    #[error("graph internal: {0}")]
    Internal(String),
}

impl From<faktor_core::Error> for GraphError {
    fn from(e: faktor_core::Error) -> Self {
        match e.kind {
            faktor_core::ErrorKind::NotFound => GraphError::NotFound(e.message),
            faktor_core::ErrorKind::Conflict => GraphError::Conflict(e.message),
            faktor_core::ErrorKind::Oversized => GraphError::Oversized(e.message),
            _ => GraphError::Internal(format!("{:?}: {}", e.kind, e.message)),
        }
    }
}

impl From<ExecError> for GraphError {
    fn from(e: ExecError) -> Self {
        match e {
            ExecError::Conflict(m) => GraphError::Conflict(m),
            ExecError::NotFound(m) => GraphError::NotFound(m),
            ExecError::Oversized(m) => GraphError::Oversized(m),
            other => GraphError::Internal(other.to_string()),
        }
    }
}

impl From<GraphError> for ExecError {
    fn from(e: GraphError) -> Self {
        match e {
            GraphError::MissingAssignment { run_id, item_id } => ExecError::NotFound(format!(
                "run {run_id}: plan item {item_id} has no durable work-item assignment"
            )),
            GraphError::MissingChild { run_id, child_id } => ExecError::NotFound(format!(
                "run {run_id}: assigned child {child_id} has no durable child row"
            )),
            GraphError::Conflict(m) => ExecError::Conflict(m),
            GraphError::NotFound(m) => ExecError::NotFound(m),
            GraphError::Oversized(m) => ExecError::Oversized(m),
            GraphError::Internal(m) => ExecError::Internal(m),
        }
    }
}

impl OrchestratorRuntime {
    /// The operation graph of the parent session's run. Durable rows only:
    /// identical before and after a manager reopen (and identical to the
    /// view a re-attached executor reconstructs). A parent with several
    /// runs is ambiguous and refuses loudly; use [`Self::operation_graph_run`]
    /// to name one.
    pub fn operation_graph(&self, parent_session: SessionId) -> Result<OpGraph, GraphError> {
        let runs = durable_runs(self.manager.clone(), parent_session)?;
        match runs.len() {
            0 => Err(GraphError::NotFound(format!(
                "session {parent_session} has no durable orchestration run"
            ))),
            1 => {
                let run = runs.into_iter().next().expect("len checked");
                self.operation_graph_run(parent_session, &run)
            }
            _ => Err(GraphError::Conflict(format!(
                "session {parent_session} holds {} orchestration runs ({}); the graph of one run needs one plan — name the run",
                runs.len(),
                runs.iter().cloned().collect::<Vec<_>>().join(", ")
            ))),
        }
    }

    /// The operation graph of ONE named run under the parent session.
    ///
    /// Wave A3 identity contract: child nodes are assembled in PLAN order
    /// through the run's DURABLE work-item → child assignment rows — never
    /// `rows.iter().enumerate()`, never `zip(plan.items, rows)`, never
    /// HashMap iteration, never spawn/completion order. For every spawn
    /// work item of the durable plan the assembly fetches its assignment
    /// ([`GraphError::MissingAssignment`] when absent), then the child row
    /// the assignment names ([`GraphError::MissingChild`] when the child's
    /// spawn is durable but its row is gone). Assignment rows that cannot
    /// be sound against the plan (unknown items, duplicates, foreign child
    /// rows) are a typed [`GraphError::Conflict`]. Rows without a plan-item
    /// assignment (reviewers) keep their plan-less node with
    /// `plan_step_index: None`.
    pub fn operation_graph_run(
        &self,
        parent_session: SessionId,
        run_id: &str,
    ) -> Result<OpGraph, GraphError> {
        let plan_row = self.plan_row(parent_session, run_id)?;
        let plan = &plan_row.plan;
        let specs = &plan_row.specs;
        let mut rows = Self::registry_rows(self.manager.clone(), parent_session, run_id)?;
        if rows.len() > MAX_GRAPH_CHILDREN {
            return Err(GraphError::Oversized(format!(
                "run {run_id} holds {} durable child rows (cap {MAX_GRAPH_CHILDREN}); refusing an unbounded graph",
                rows.len()
            )));
        }
        rows.sort_by(|a, b| {
            a.created_ms
                .cmp(&b.created_ms)
                .then_with(|| a.child_id.cmp(&b.child_id))
        });
        let assignments = Self::assignment_rows(self.manager.clone(), parent_session, run_id)?;
        let violations = OrchestratorRuntime::assignment_shape_violations(
            run_id,
            plan,
            specs,
            &assignments,
            &rows,
        );
        if !violations.is_empty() {
            return Err(GraphError::Conflict(format!(
                "run {run_id}: assignment rows violate the identity contract: {}",
                violations.join("; ")
            )));
        }
        let derived = derived_item_states(plan, &rows);
        let root = GraphRootNode {
            plan_id: run_id.to_string(),
            goal: plan.goal.clone(),
            state: derived_root_state(&derived),
            work_items: plan
                .work_items
                .iter()
                .zip(&derived)
                .map(|(w, state)| GraphWorkItem {
                    item_id: w.id.clone(),
                    kind: w.kind,
                    state: *state,
                })
                .collect(),
        };
        let rows_by_child: HashMap<&str, &ChildRuntime> =
            rows.iter().map(|r| (r.child_id.as_str(), r)).collect();
        // Spawn evidence of the run: the env snapshot every spawn binds is
        // the FIRST durable write of a spawn (key `<run>/env-<child>`).
        // The graph uses it to tell "assigned but never admitted" (no
        // node) apart from "child row deleted after a real spawn"
        // (MissingChild).
        let spawn_evidence = env_snapshot_keys(self.manager.clone(), parent_session, run_id)?;
        let mut children = Vec::with_capacity(rows.len());
        let mut claimed: HashSet<&str> = HashSet::new();
        for item in &plan.work_items {
            if !specs.get(&item.id).map(|s| s.spawn).unwrap_or(true) {
                continue; // auto item: never spawns, owns no child node
            }
            let Some(assignment) = assignments.iter().find(|a| a.item_id == item.id) else {
                return Err(GraphError::MissingAssignment {
                    run_id: run_id.to_string(),
                    item_id: item.id.clone(),
                });
            };
            let Some(row) = rows_by_child.get(assignment.child_id.as_str()) else {
                if spawn_evidence.contains(&format!("{run_id}/env-{}", assignment.child_id)) {
                    return Err(GraphError::MissingChild {
                        run_id: run_id.to_string(),
                        child_id: assignment.child_id.clone(),
                    });
                }
                // Assigned, not yet admitted when the durable view was
                // taken: the item simply has no child node yet.
                continue;
            };
            claimed.insert(row.child_id.as_str());
            children.push(self.graph_child_node(
                parent_session,
                run_id,
                row,
                Some(assignment.plan_step_index),
            )?);
        }
        for row in &rows {
            if !claimed.contains(row.child_id.as_str()) {
                children.push(self.graph_child_node(parent_session, run_id, row, None)?);
            }
        }
        children.sort_by(|a, b| {
            (a.plan_step_index.unwrap_or(usize::MAX), a.child_id.clone())
                .cmp(&(b.plan_step_index.unwrap_or(usize::MAX), b.child_id.clone()))
        });
        Ok(OpGraph { root, children })
    }

    /// One child node from a durable row: session steering + latest merge
    /// (pure reads over the durable rows).
    fn graph_child_node(
        &self,
        parent_session: SessionId,
        run_id: &str,
        row: &ChildRuntime,
        plan_step_index: Option<usize>,
    ) -> Result<GraphChildNode, GraphError> {
        let session = self
            .manager
            .get_session(SessionId::new(row.session_id))?
            .ok_or_else(|| {
                GraphError::NotFound(format!(
                    "graph assembly: child {} names missing session {}",
                    row.child_id, row.session_id
                ))
            })?;
        let steer_events = session
            .orchestrator_ctl_all()
            .map_err(|e| {
                GraphError::Internal(format!("steering rows of {}: {}", row.child_id, e.message))
            })?
            .into_iter()
            .map(|ctl| SteerEvent {
                kind: ctl.control,
                seq: ctl.seq,
                applied_ms: ctl.applied_ms,
            })
            .collect();
        let merge =
            latest_child_merge(self.manager.clone(), parent_session, run_id, &row.child_id)?;
        let presentation = session.child_presentation(&row.child_id).map_err(|e| {
            GraphError::Internal(format!(
                "presentation of child {}: {}",
                row.child_id, e.message
            ))
        })?;
        Ok(GraphChildNode {
            child_id: row.child_id.clone(),
            session_id: row.session_id,
            operation_id: row.operation_id,
            worktree_id: row.worktree_id,
            ownership: row.ownership,
            state: row.state,
            presentation,
            budget: row.budget_max_tokens,
            capabilities: row.permissions.clone(),
            plan_step_index,
            steer_events,
            merge,
        })
    }
}

/// `<run>/env-<child>` header keys of a run: the env snapshot a spawn
/// binds is the first durable write of a spawn (its key lives in the
/// parent session's row space, run-scoped, so a hostile child-row deletion
/// is distinguishable from a child that never spawned).
fn env_snapshot_keys(
    manager: Arc<faktor_session::SessionManager>,
    parent: SessionId,
    run_id: &str,
) -> Result<HashSet<String>, GraphError> {
    let handle = parent_handle(&manager, parent)?;
    let mut keys = HashSet::new();
    for (kind, key, _value) in scan_facts(&handle)? {
        if kind == crate::runtime::env::KIND_ENV_SNAPSHOT
            && key
                .strip_prefix(run_id)
                .is_some_and(|rest| rest.starts_with("/env-"))
        {
            keys.insert(key);
        }
    }
    Ok(keys)
}

/// Every run id with durable plan or registry rows under one parent
/// (sorted, deduplicated).
fn durable_runs(
    manager: Arc<faktor_session::SessionManager>,
    parent: SessionId,
) -> Result<BTreeSet<String>, ExecError> {
    let handle = parent_handle(&manager, parent)?;
    let mut runs = BTreeSet::new();
    for (kind, key, _value) in scan_facts(&handle)? {
        match kind.as_str() {
            crate::runtime::PLAN_ROW_KIND => {
                runs.insert(key);
            }
            crate::runtime::REGISTRY_ROW_KIND => {
                if let Some(run) = key.rsplit_once('/').map(|(r, _c)| r) {
                    if !run.is_empty() {
                        runs.insert(run.to_string());
                    }
                }
            }
            _ => {}
        }
    }
    Ok(runs)
}

/// Derive the per-plan-step states from the durable child rows with the
/// exact semantics of executor re-attach (`reconcile_from_registry`):
/// every item starts Pending; a durable child moves its item through
/// Running and then applies the ONE canonical projection
/// ([`crate::project_child_state`]) — Paused, Waiting, Blocked, Done,
/// Failed and Cancelled all survive; a "non-terminal means Running"
/// conversion is a truth defect and is forbidden here. Items whose
/// dependency failed/cancelled are Blocked. Items without a child stay
/// Pending (they were not admitted when the durable view was taken).
///
/// This function is SHARED: the typed operation graph and the native JSON
/// graph both call it, so the two surfaces are byte-identical for the same
/// registry rows.
pub fn derived_item_states(plan: &crate::TaskPlan, rows: &[ChildRuntime]) -> Vec<WorkState> {
    let mut states: HashMap<String, WorkState> = plan
        .work_items
        .iter()
        .map(|w| (w.id.clone(), WorkState::Pending))
        .collect();
    for row in rows {
        let item = row.item_id.clone();
        if !states.contains_key(&item) {
            continue; // reviewer rows (plan-less children) own no plan state
        }
        // A Pending item with a durable child moves through Running first
        // (a terminal child only got there through a Running item; the
        // re-attach must never re-spawn an item that already has a child).
        if states[&item] == WorkState::Pending
            && can_advance(WorkState::Pending, WorkState::Running)
        {
            states.insert(item.clone(), WorkState::Running);
        }
        let target = crate::project_child_state(row);
        if can_advance(states[&item], target) {
            states.insert(item, target);
        }
    }
    for w in &plan.work_items {
        if states[&w.id] == WorkState::Pending
            && w.depends_on.iter().any(|d| {
                matches!(
                    states.get(d),
                    Some(WorkState::Failed | WorkState::Cancelled)
                )
            })
        {
            states.insert(w.id.clone(), WorkState::Blocked);
        }
    }
    plan.work_items.iter().map(|w| states[&w.id]).collect()
}

/// The derived plan state: `Done` when every step is Done; otherwise the
/// first of Failed / Cancelled / Running / Blocked / Paused / Waiting found
/// in plan order; `Pending` when no step moved. Documented precedence
/// (deterministic).
pub fn derived_root_state(states: &[WorkState]) -> WorkState {
    if states.iter().all(|s| *s == WorkState::Done) {
        return WorkState::Done;
    }
    for wanted in [
        WorkState::Failed,
        WorkState::Cancelled,
        WorkState::Running,
        WorkState::Blocked,
        WorkState::Paused,
        WorkState::Waiting,
    ] {
        if states.contains(&wanted) {
            return wanted;
        }
    }
    WorkState::Pending
}

/// The latest durable merge of one child (max envelope seq): its change
/// set id plus the durable decision/outcome parts. A crash-safe in-flight
/// record surfaces with its change set id and empty lists — the durable
/// decision rows decide what was merged, rejected and conflicted.
pub(crate) fn latest_child_merge(
    manager: Arc<faktor_session::SessionManager>,
    parent: SessionId,
    run: &str,
    child_id: &str,
) -> Result<Option<GraphMerge>, ExecError> {
    let envs = merge_envelopes(&manager, parent, run, child_id)?;
    let Some(latest) = envs.into_iter().max_by_key(|e| e.seq) else {
        return Ok(None);
    };
    let mut merged = read_part_paths(
        &manager,
        parent,
        run,
        child_id,
        &latest.cs_id,
        latest.seq,
        "merged",
    )?
    .unwrap_or_default();
    let mut rejected = read_part_paths(
        &manager,
        parent,
        run,
        child_id,
        &latest.cs_id,
        latest.seq,
        "rejected",
    )?
    .unwrap_or_default();
    let mut conflicts =
        read_part_conflicts(&manager, parent, run, child_id, &latest.cs_id, latest.seq)?
            .unwrap_or_default();
    merged.sort();
    rejected.sort();
    conflicts.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    Ok(Some(GraphMerge {
        change_set_id: latest.cs_id,
        merged,
        rejected,
        conflicts,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::ModelPolicy;

    fn child(item: &str, state: ChildState) -> ChildRuntime {
        ChildRuntime {
            child_id: format!("child-{item}"),
            parent_session_id: 1,
            run_id: "run".into(),
            item_id: item.into(),
            kind: WorkKind::Analysis,
            session_id: 2,
            operation_id: 0,
            workspace_id: 1,
            worktree_id: 1,
            ownership: ChildOwnership::ReadOnlyShared,
            ownership_paths: vec![],
            state,
            budget_max_tokens: None,
            permissions: CapabilitySet::new(),
            model_policy: ModelPolicy::default(),
            blocker_kind: None,
            blocker_reason: None,
            blocker_dependency: None,
            blocker_resolution: None,
            last_progress_ms: None,
            created_ms: 1,
            updated_ms: 1,
            base_snapshot_id: None,
            env_snapshot_id: None,
        }
    }

    fn plan_with(items: &[(&str, &[&str])]) -> crate::TaskPlan {
        crate::TaskPlan {
            goal: "g".into(),
            non_goals: vec![],
            constraints: vec![],
            work_items: items
                .iter()
                .map(|(id, deps)| {
                    let mut w = crate::WorkItem::new(*id, format!("work {id}"), WorkKind::Analysis);
                    w.depends_on = deps.iter().map(|d| d.to_string()).collect();
                    w
                })
                .collect(),
        }
    }

    #[test]
    fn project_child_state_is_total_and_exact_per_mapping() {
        for (child_state, want) in [
            (ChildState::Running, WorkState::Running),
            (ChildState::Paused, WorkState::Paused),
            (ChildState::Waiting, WorkState::Waiting),
            (ChildState::Blocked, WorkState::Blocked),
            (ChildState::Done, WorkState::Done),
            (ChildState::Failed, WorkState::Failed),
            (ChildState::Cancelled, WorkState::Cancelled),
        ] {
            assert_eq!(
                crate::project_child_state(&child("x", child_state)),
                want,
                "projection of {child_state:?}"
            );
        }
    }

    #[test]
    fn derived_item_states_carries_every_projected_state_not_just_running() {
        // Adversarial: a non-terminal child that is NOT Running must never
        // be flattened to Running (the historic defect).
        let plan = plan_with(&[
            ("a", &[]),
            ("b", &[]),
            ("c", &[]),
            ("d", &[]),
            ("e", &[]),
            ("f", &[]),
            ("g", &[]),
        ]);
        let rows = vec![
            child("a", ChildState::Running),
            child("b", ChildState::Paused),
            child("c", ChildState::Waiting),
            child("d", ChildState::Blocked),
            child("e", ChildState::Done),
            child("f", ChildState::Failed),
            child("g", ChildState::Cancelled),
        ];
        assert_eq!(
            derived_item_states(&plan, &rows),
            vec![
                WorkState::Running,
                WorkState::Paused,
                WorkState::Waiting,
                WorkState::Blocked,
                WorkState::Done,
                WorkState::Failed,
                WorkState::Cancelled,
            ]
        );
    }

    #[test]
    fn derived_root_state_reports_paused_and_waiting_children() {
        // Direct root tests: a paused item is Paused (previously
        // unreachable), a waiting item is Waiting, failure paths unchanged.
        assert_eq!(derived_root_state(&[WorkState::Paused]), WorkState::Paused);
        assert_eq!(
            derived_root_state(&[WorkState::Waiting]),
            WorkState::Waiting
        );
        assert_eq!(
            derived_root_state(&[WorkState::Blocked]),
            WorkState::Blocked
        );
        assert_eq!(derived_root_state(&[WorkState::Failed]), WorkState::Failed);
        assert_eq!(
            derived_root_state(&[WorkState::Cancelled]),
            WorkState::Cancelled
        );
        assert_eq!(derived_root_state(&[WorkState::Done]), WorkState::Done);
        assert_eq!(
            derived_root_state(&[WorkState::Pending]),
            WorkState::Pending
        );
        // Deterministic precedence: failure beats every non-Done state.
        assert_eq!(
            derived_root_state(&[WorkState::Paused, WorkState::Failed]),
            WorkState::Failed
        );
    }

    #[test]
    fn derived_item_states_blocks_dependents_of_failed_or_cancelled_rows() {
        let plan = plan_with(&[("a", &[]), ("b", &["a"]), ("c", &["a"])]);
        assert_eq!(
            derived_item_states(&plan, &[child("a", ChildState::Failed)]),
            vec![WorkState::Failed, WorkState::Blocked, WorkState::Blocked]
        );
        assert_eq!(
            derived_item_states(&plan, &[child("a", ChildState::Cancelled)]),
            vec![WorkState::Cancelled, WorkState::Blocked, WorkState::Blocked]
        );
        // A reviewer row (plan-less child) never owns a plan step.
        let mut reviewer = child("review", ChildState::Failed);
        reviewer.item_id = "review-of-a".into();
        assert_eq!(
            derived_item_states(&plan, &[reviewer]),
            vec![WorkState::Pending, WorkState::Pending, WorkState::Pending]
        );
    }
}

#[cfg(test)]
mod blocker_tests {
    use crate::runtime::{ChildBlocker, ExecError, MAX_CHILD_BLOCKER_REASON_CHARS};

    #[test]
    fn child_blocker_validate_rejects_hostile_text_with_typed_errors() {
        let huge = ChildBlocker {
            kind: "budget".into(),
            reason: "x".repeat(MAX_CHILD_BLOCKER_REASON_CHARS + 1),
            dependency: None,
            resolution: None,
            last_progress_ms: None,
        };
        assert!(matches!(huge.validate(), Err(ExecError::Oversized(_))));
        let blank = ChildBlocker {
            kind: "  ".into(),
            reason: "r".into(),
            dependency: None,
            resolution: None,
            last_progress_ms: None,
        };
        assert!(matches!(blank.validate(), Err(ExecError::Malformed(_))));
        let control = ChildBlocker {
            kind: "permission".into(),
            reason: "bad\u{0}text".into(),
            dependency: None,
            resolution: None,
            last_progress_ms: None,
        };
        assert!(matches!(control.validate(), Err(ExecError::Malformed(_))));
        let negative = ChildBlocker {
            kind: "dependency".into(),
            reason: "waiting".into(),
            dependency: Some("a".into()),
            resolution: Some("wait".into()),
            last_progress_ms: Some(-1),
        };
        assert!(matches!(negative.validate(), Err(ExecError::Malformed(_))));
        let ok = ChildBlocker::dependency("a", "waiting on work item \"a\"", "wait");
        assert!(ok.validate().is_ok());
    }
}
