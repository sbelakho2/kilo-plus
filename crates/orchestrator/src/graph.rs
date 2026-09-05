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
use faktor_session::child::{ChildControl, ChildOwnership};

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

impl OrchestratorRuntime {
    /// The operation graph of the parent session's run. Durable rows only:
    /// identical before and after a manager reopen (and identical to the
    /// view a re-attached executor reconstructs). A parent with several
    /// runs is ambiguous and refuses loudly; use [`Self::operation_graph_run`]
    /// to name one.
    pub fn operation_graph(&self, parent_session: SessionId) -> Result<OpGraph, ExecError> {
        let runs = durable_runs(self.manager.clone(), parent_session)?;
        match runs.len() {
            0 => Err(ExecError::NotFound(format!(
                "session {parent_session} has no durable orchestration run"
            ))),
            1 => {
                let run = runs.into_iter().next().expect("len checked");
                self.operation_graph_run(parent_session, &run)
            }
            _ => Err(ExecError::Conflict(format!(
                "session {parent_session} holds {} orchestration runs ({}); the graph of one run needs one plan — name the run",
                runs.len(),
                runs.iter().cloned().collect::<Vec<_>>().join(", ")
            ))),
        }
    }

    /// The operation graph of ONE named run under the parent session.
    pub fn operation_graph_run(
        &self,
        parent_session: SessionId,
        run_id: &str,
    ) -> Result<OpGraph, ExecError> {
        let plan_row = self.plan_row(parent_session, run_id)?;
        let mut rows = Self::registry_rows(self.manager.clone(), parent_session, run_id)?;
        if rows.len() > MAX_GRAPH_CHILDREN {
            return Err(ExecError::Oversized(format!(
                "run {run_id} holds {} durable child rows (cap {MAX_GRAPH_CHILDREN}); refusing an unbounded graph",
                rows.len()
            )));
        }
        rows.sort_by(|a, b| {
            a.created_ms
                .cmp(&b.created_ms)
                .then_with(|| a.child_id.cmp(&b.child_id))
        });
        let plan = &plan_row.plan;
        let step_index: std::collections::HashMap<&str, usize> = plan
            .work_items
            .iter()
            .enumerate()
            .map(|(i, w)| (w.id.as_str(), i))
            .collect();
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
        let mut children = Vec::with_capacity(rows.len());
        for row in rows {
            let session = self
                .manager
                .get_session(SessionId::new(row.session_id))?
                .ok_or_else(|| {
                    ExecError::NotFound(format!(
                        "graph assembly: child {} names missing session {}",
                        row.child_id, row.session_id
                    ))
                })?;
            let steer_events = session
                .orchestrator_ctl_all()
                .map_err(|e| {
                    ExecError::Internal(format!("steering rows of {}: {}", row.child_id, e.message))
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
            children.push(GraphChildNode {
                child_id: row.child_id.clone(),
                session_id: row.session_id,
                operation_id: row.operation_id,
                worktree_id: row.worktree_id,
                ownership: row.ownership,
                state: row.state,
                budget: row.budget_max_tokens,
                capabilities: row.permissions,
                plan_step_index: step_index.get(row.item_id.as_str()).copied(),
                steer_events,
                merge,
            });
        }
        children.sort_by(|a, b| {
            (a.plan_step_index.unwrap_or(usize::MAX), a.child_id.clone())
                .cmp(&(b.plan_step_index.unwrap_or(usize::MAX), b.child_id.clone()))
        });
        Ok(OpGraph { root, children })
    }
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
/// every item starts Pending; a durable child moves its item Pending →
/// Running first (a terminal child only got there through a Running item)
/// and then maps the child's terminal state onto the item; items whose
/// dependency failed/cancelled are Blocked. Items without a child stay
/// Pending (they were not admitted when the durable view was taken).
pub(crate) fn derived_item_states(plan: &crate::TaskPlan, rows: &[ChildRuntime]) -> Vec<WorkState> {
    let mut states: HashMap<String, WorkState> = plan
        .work_items
        .iter()
        .map(|w| (w.id.clone(), WorkState::Pending))
        .collect();
    for row in rows {
        let item = row.item_id.clone();
        if states.get(&item) == Some(&WorkState::Pending)
            && can_advance(WorkState::Pending, WorkState::Running)
        {
            states.insert(item.clone(), WorkState::Running);
        }
        let target = match row.state {
            ChildState::Done => Some(WorkState::Done),
            ChildState::Cancelled => Some(WorkState::Cancelled),
            ChildState::Failed => Some(WorkState::Failed),
            _ => None,
        };
        if let Some(t) = target {
            if can_advance(states[&item], t) {
                states.insert(item, t);
            }
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
/// first of Failed / Cancelled / Running / Blocked found in plan order;
/// `Pending` when no step moved. Documented precedence (deterministic).
pub(crate) fn derived_root_state(states: &[WorkState]) -> WorkState {
    if states.iter().all(|s| *s == WorkState::Done) {
        return WorkState::Done;
    }
    for wanted in [
        WorkState::Failed,
        WorkState::Cancelled,
        WorkState::Running,
        WorkState::Blocked,
        WorkState::Paused,
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
