//! The first-class durable Task object (audit 25).
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
//! Bounds are enforced HERE, before any write, and oversized input is
//! REJECTED with an error — never silently truncated: goal <=
//! [`MAX_TASK_GOAL_BYTES`] bytes, criteria <= [`MAX_TASK_CRITERIA`] entries
//! (each <= [`MAX_TASK_CRITERION_BYTES`]), plan <= [`MAX_TASK_PLAN_STEPS`]
//! steps (each <= [`MAX_TASK_STEP_BYTES`]). A patch with a `None` field
//! keeps the row's current value, so updates are read-modify-write safe
//! under the store's single writer.

use faktor_core::id::TaskId;
use faktor_core::state::TaskState;

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
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Task {
    pub task_id: TaskId,
    pub session_id: faktor_core::id::SessionId,
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
            task_id: TaskId::new(0),
            session_id: faktor_core::id::SessionId::new(0),
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

impl From<Task> for faktor_store::TaskRow {
    fn from(t: Task) -> Self {
        Self {
            task_id: t.task_id,
            session_id: t.session_id,
            goal: t.goal,
            acceptance_criteria: t.acceptance_criteria,
            plan: t.plan,
            max_tokens: t.budget.max_tokens,
            max_turns: t.budget.max_turns,
            spent_tokens: t.budget.spent_tokens,
            spent_turns: t.budget.spent_turns,
            state: t.state,
            created_ms: t.created_ms,
            updated_ms: t.updated_ms,
        }
    }
}

/// A bounded update over an existing durable Task. Every field is optional:
/// `None` keeps the row's current value, so `update_task` never clobbers a
/// field its caller did not intend to change (e.g. the runtime preserves a
/// caller-set budget when it patches the gate state).
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

    /// Create (or fully replace) ONE durable task row. Oversized fields are
    /// rejected with an error BEFORE any write — never truncated silently.
    pub fn create_task(&self, task: Task) -> faktor_core::Result<Task> {
        if task.task_id.raw() == 0 || task.session_id.raw() == 0 {
            return Err(
                SessionError::Malformed("task_id and session_id must be non-zero".into()).into(),
            );
        }
        validate_task_fields(&task)?;
        if task.created_ms == 0 {
            return Err(SessionError::Malformed("created_ms must be set".into()).into());
        }
        let row = faktor_store::TaskRow::from(task.clone());
        self.manager
            .store()
            .upsert_task(&row)
            .map_err(crate::map_store_err)?;
        Ok(task)
    }

    /// Patch ONE durable task row. Fields that validate and are present are
    /// applied; the other fields keep their current values. `created_ms` is
    /// preserved by construction (the patch is applied over the stored row).
    pub fn update_task(&self, task_id: TaskId, patch: TaskPatch) -> faktor_core::Result<Task> {
        if task_id.raw() == 0 {
            return Err(SessionError::Malformed("task_id must be non-zero".into()).into());
        }
        let store = self.manager.store();
        let stored = store
            .get_task(self.id, task_id)
            .map_err(crate::map_store_err)?
            .ok_or_else(|| SessionError::NotFound(format!("task {task_id}")))?;
        let mut next = Task::from(stored);
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
            next.state = state;
        }
        validate_task_fields(&next)?;
        next.updated_ms = self.manager.now_ms();
        store
            .upsert_task(&faktor_store::TaskRow::from(next.clone()))
            .map_err(crate::map_store_err)?;
        Ok(next)
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
}

fn validate_task_fields(t: &Task) -> faktor_core::Result<()> {
    if t.goal.len() > MAX_TASK_GOAL_BYTES {
        return Err(SessionError::Oversized(format!(
            "task goal of {} bytes exceeds MAX_TASK_GOAL_BYTES ({MAX_TASK_GOAL_BYTES})",
            t.goal.len()
        ))
        .into());
    }
    if t.acceptance_criteria.len() > MAX_TASK_CRITERIA {
        return Err(SessionError::Oversized(format!(
            "{} acceptance criteria exceed MAX_TASK_CRITERIA ({MAX_TASK_CRITERIA})",
            t.acceptance_criteria.len()
        ))
        .into());
    }
    for c in &t.acceptance_criteria {
        if c.len() > MAX_TASK_CRITERION_BYTES {
            return Err(SessionError::Oversized(format!(
                "a criterion of {} bytes exceeds MAX_TASK_CRITERION_BYTES ({MAX_TASK_CRITERION_BYTES})",
                c.len()
            ))
            .into());
        }
    }
    if t.plan.len() > MAX_TASK_PLAN_STEPS {
        return Err(SessionError::Oversized(format!(
            "{} plan steps exceed MAX_TASK_PLAN_STEPS ({MAX_TASK_PLAN_STEPS})",
            t.plan.len()
        ))
        .into());
    }
    for s in &t.plan {
        if s.len() > MAX_TASK_STEP_BYTES {
            return Err(SessionError::Oversized(format!(
                "a plan step of {} bytes exceeds MAX_TASK_STEP_BYTES ({MAX_TASK_STEP_BYTES})",
                s.len()
            ))
            .into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::tests::{session, test_manager};
    use faktor_core::state::TaskState;

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

    #[test]
    fn task_create_get_update_list_roundtrip() {
        let (_d, m) = test_manager();
        let s = session(&m);
        assert!(s.list_tasks().unwrap().is_empty());
        let t = task(&s);
        let created = s.create_task(t.clone()).unwrap();
        assert_eq!(created.state, TaskState::Pending);
        assert_eq!(s.get_task(created.task_id).unwrap(), Some(created.clone()));
        // Patch: state + spend move forward; untouched fields survive.
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
        // The stored row equals the returned patch (updated_ms bumped).
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
        assert_eq!(err.kind, faktor_core::ErrorKind::NotFound);
    }

    #[test]
    fn oversized_goal_criteria_and_plan_are_rejected_never_truncated() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let mut t = task(&s);
        // goal > 16 KiB
        t.goal = "g".repeat(MAX_TASK_GOAL_BYTES + 1);
        assert!(matches!(
            s.create_task(t.clone()).unwrap_err().kind,
            faktor_core::ErrorKind::Oversized
        ));
        // criteria > 32 entries
        t.goal = "ok".into();
        t.acceptance_criteria = (0..=MAX_TASK_CRITERIA)
            .map(|i| format!("criterion {i}"))
            .collect();
        assert!(matches!(
            s.create_task(t.clone()).unwrap_err().kind,
            faktor_core::ErrorKind::Oversized
        ));
        // one criterion over its entry bound
        t.acceptance_criteria = vec!["c".repeat(MAX_TASK_CRITERION_BYTES + 1)];
        assert!(matches!(
            s.create_task(t.clone()).unwrap_err().kind,
            faktor_core::ErrorKind::Oversized
        ));
        // plan > 256 steps
        t.acceptance_criteria = vec!["c".into()];
        t.plan = (0..=MAX_TASK_PLAN_STEPS)
            .map(|i| format!("step {i}"))
            .collect();
        assert!(matches!(
            s.create_task(t.clone()).unwrap_err().kind,
            faktor_core::ErrorKind::Oversized
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
        // Oversized patch goal: rejected, stored goal untouched.
        let err = s
            .update_task(
                created.task_id,
                TaskPatch {
                    goal: Some("x".repeat(MAX_TASK_GOAL_BYTES + 1)),
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(matches!(err.kind, faktor_core::ErrorKind::Oversized));
        let row = s.get_task(created.task_id).unwrap().unwrap();
        assert_eq!(row.goal, created.goal, "rejected patch left no trace");
        // Oversized patch plan: rejected.
        let err = s
            .update_task(
                created.task_id,
                TaskPatch {
                    plan: Some((0..=MAX_TASK_PLAN_STEPS).map(|i| format!("s{i}")).collect()),
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(matches!(err.kind, faktor_core::ErrorKind::Oversized));
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
        // A rewind attempt is refused by construction: spend never falls.
        let p2 = s.update_task(created.task_id, bump(10)).unwrap();
        assert_eq!(p2.budget.spent_tokens, 50, "spend is monotone");
        // max fields DO update (they are not counters).
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
        assert_eq!(p3.budget.max_turns, Some(1));
        assert_eq!(p3.budget.spent_tokens, 50);
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
}
