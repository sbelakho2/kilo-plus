//! The durable task ledger (spec §8, "Durable task state").
//!
//! Stays compact by construction: every completed turn folds its material
//! into structured fields, so emergency compaction never has to summarize
//! the entire history from scratch.

use faktor_core::error::{Error, ErrorKind};

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct TaskLedger {
    pub goal: String,
    pub constraints: Vec<String>,
    pub completed_steps: Vec<String>,
    pub open_steps: Vec<String>,
    pub decisions: Vec<String>,
    pub known_failures: Vec<String>,
    pub changed_files: Vec<String>,
    pub tests_run: Vec<String>,
    pub tests_failed: Vec<String>,
    pub user_preferences: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct TurnSummary {
    pub steps_completed: Vec<String>,
    pub steps_opened: Vec<String>,
    pub decisions: Vec<String>,
    pub failures: Vec<String>,
    pub files_changed: Vec<String>,
    pub tests_run: Vec<String>,
    pub tests_failed: Vec<String>,
}

const MAX_OPEN_STEPS: usize = 64;
const MAX_DECISIONS: usize = 128;
const MAX_FAILURES: usize = 128;
const MAX_FILES: usize = 256;

impl TaskLedger {
    /// Fold one finished turn into the ledger. Bounded: lists never grow
    /// without limit (oldest entries evicted first).
    pub fn record_turn(&mut self, turn: &TurnSummary) {
        push_bounded(&mut self.completed_steps, &turn.steps_completed, MAX_FILES);
        push_bounded(&mut self.open_steps, &turn.steps_opened, MAX_OPEN_STEPS);
        push_bounded(&mut self.decisions, &turn.decisions, MAX_DECISIONS);
        push_bounded(&mut self.known_failures, &turn.failures, MAX_FAILURES);
        push_bounded(&mut self.changed_files, &turn.files_changed, MAX_FILES);
        push_bounded(&mut self.tests_run, &turn.tests_run, MAX_FILES);
        push_bounded(&mut self.tests_failed, &turn.tests_failed, MAX_FAILURES);
    }

    /// Compact structured render, ~400 tokens max by construction.
    pub fn compact_render(&self) -> String {
        let mut out = String::new();
        if !self.goal.is_empty() {
            out.push_str(&format!("GOAL: {}\n", truncate(&self.goal, 200)));
        }
        if !self.constraints.is_empty() {
            out.push_str("CONSTRAINTS:\n");
            for c in self.constraints.iter().take(8) {
                out.push_str(&format!("- {}\n", truncate(c, 120)));
            }
        }
        if !self.open_steps.is_empty() {
            out.push_str("OPEN STEPS:\n");
            for s in self.open_steps.iter().take(12) {
                out.push_str(&format!("- {}\n", truncate(s, 120)));
            }
        }
        if !self.decisions.is_empty() {
            out.push_str("DECISIONS:\n");
            for d in self.decisions.iter().rev().take(8) {
                out.push_str(&format!("- {}\n", truncate(d, 120)));
            }
        }
        if !self.known_failures.is_empty() {
            out.push_str("KNOWN FAILURES:\n");
            for f in self.known_failures.iter().rev().take(8) {
                out.push_str(&format!("- {}\n", truncate(f, 120)));
            }
        }
        if !self.changed_files.is_empty() {
            out.push_str("CHANGED FILES: ");
            let joined = self
                .changed_files
                .iter()
                .map(|s| truncate(s, 80))
                .collect::<Vec<_>>()
                .join(", ");
            out.push_str(&truncate(&joined, 300));
            out.push('\n');
        }
        out
    }

    pub fn token_estimate(&self) -> usize {
        let e = super::estimator::Estimator;
        e.estimate_tokens(&self.compact_render())
    }

    /// Validate that hostile inputs (absurd lengths) cannot blow the render.
    pub fn validate_sane(&self) -> Result<(), Error> {
        for (label, items) in [
            ("constraints", &self.constraints),
            ("completed_steps", &self.completed_steps),
            ("open_steps", &self.open_steps),
            ("decisions", &self.decisions),
            ("known_failures", &self.known_failures),
            ("changed_files", &self.changed_files),
            ("tests_run", &self.tests_run),
            ("tests_failed", &self.tests_failed),
        ] {
            for item in items {
                if item.len() > 4096 {
                    return Err(Error::new(
                        ErrorKind::Oversized,
                        format!("ledger {label} entry exceeds 4096 chars"),
                    ));
                }
            }
        }
        Ok(())
    }
}

fn push_bounded(dst: &mut Vec<String>, src: &[String], max: usize) {
    for item in src {
        if item.is_empty() {
            continue;
        }
        if !dst.contains(item) {
            dst.push(item.clone());
        }
    }
    if dst.len() > max {
        dst.drain(..dst.len() - max);
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

// ===========================================================================
// Durable task context projection (audit 71)
//
// `TaskLedger` above is the LEGACY working fold the agent turn loop still
// mutates (`record_turn` + `put_task_ledger`); it is kept as the
// compatibility surface for callers this wave may not touch. Compaction
// consumes [`TaskContextProjection`] instead: a READ-ONLY structured view
// assembled from the durable rows that actually own each fact —
//
// - goal / acceptance criteria / task state / plan: the typed `Task` row;
// - decisions / plan / child progress: the typed session ledger rows;
// - checks: the durable `VerificationRecord` rows.
//
// The projection has NO public mutators (locked by a source-scan test):
// fields are private, every method borrows `&self`, and the only
// constructing surface is `from_durable_rows` (the durable-row
// constructor) plus `from_task_ledger` (the compatibility bridge for
// existing tests/early sessions that only hold the legacy fold). Nothing
// born from a transcript can rewrite it; a lying transcript cannot change
// what the projection reports.
// ===========================================================================

/// One durable decision row (mirror of the session-side typed ledger
/// `Decision` payload; the pure context crate must not depend on session).
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ProjectedDecision {
    pub step: String,
    pub choice: String,
    pub rationale: String,
}

/// One durable verification check row (mirror of the session-side
/// `CheckExecution`; checks live on `VerificationRecord` rows).
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ProjectedCheck {
    pub name: String,
    pub status: String,
    pub summary: String,
}

/// Durable child-agent progress (`ChildAgentStarted`/`Finished` rows).
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ProjectedChild {
    pub purpose: String,
    pub outcome: Option<String>,
}

/// The durable rows a [`TaskContextProjection`] is built from. Field set
/// mirrors what the runtime can read without the context crate learning
/// session types.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DurableTaskRows {
    pub task_state: String,
    pub goal: String,
    pub criteria: Vec<String>,
    pub plan_steps: Vec<String>,
    pub decisions: Vec<ProjectedDecision>,
    pub checks: Vec<ProjectedCheck>,
    pub children: Vec<ProjectedChild>,
    pub known_failures: Vec<String>,
    pub changed_files: Vec<String>,
}

/// Render bounds of the summarizable (non-identity) projection.
const PROJECTION_MAX_OPEN_STEPS: usize = 12;
const PROJECTION_MAX_DECISIONS: usize = 8;
const PROJECTION_MAX_CHECKS: usize = 8;
const PROJECTION_MAX_CHILDREN: usize = 8;
const PROJECTION_MAX_FAILURES: usize = 8;
const PROJECTION_FIELD_CHARS: usize = 120;

/// READ-ONLY projection of the durable task state. See the module section
/// docs above. Deliberately NOT `Serialize`/`Deserialize` and with private
/// fields: no caller can construct-or-mutate it around the durable rows.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TaskContextProjection {
    goal: String,
    criteria: Vec<String>,
    task_state: String,
    plan_steps: Vec<String>,
    decisions: Vec<ProjectedDecision>,
    checks: Vec<ProjectedCheck>,
    children: Vec<ProjectedChild>,
    known_failures: Vec<String>,
    changed_files: Vec<String>,
}

impl TaskContextProjection {
    /// The durable-row constructor: the authoritative path. Consumes the
    /// rows by value, so no caller retains a write handle through it.
    pub fn from_durable_rows(rows: DurableTaskRows) -> Self {
        Self {
            goal: rows.goal,
            criteria: rows.criteria,
            task_state: rows.task_state,
            plan_steps: rows.plan_steps,
            decisions: rows.decisions,
            checks: rows.checks,
            children: rows.children,
            known_failures: rows.known_failures,
            changed_files: rows.changed_files,
        }
    }

    /// Compatibility bridge for existing callers/early sessions that only
    /// hold the legacy durable `TaskLedger` fold. The mapping is explicit;
    /// nothing here grants the projection a write path back.
    pub fn from_task_ledger(ledger: &TaskLedger) -> Self {
        let decisions = ledger
            .decisions
            .iter()
            .map(|d| ProjectedDecision {
                step: String::new(),
                choice: d.clone(),
                rationale: String::new(),
            })
            .collect();
        let mut checks: Vec<ProjectedCheck> = ledger
            .tests_run
            .iter()
            .map(|name| ProjectedCheck {
                name: name.clone(),
                status: if ledger.tests_failed.contains(name) {
                    "failed".to_string()
                } else {
                    "passed".to_string()
                },
                summary: String::new(),
            })
            .collect();
        for name in &ledger.tests_failed {
            if !checks.iter().any(|c| &c.name == name) {
                checks.push(ProjectedCheck {
                    name: name.clone(),
                    status: "failed".to_string(),
                    summary: String::new(),
                });
            }
        }
        Self {
            goal: ledger.goal.clone(),
            criteria: ledger.constraints.clone(),
            task_state: String::new(),
            plan_steps: ledger.open_steps.clone(),
            decisions,
            checks,
            children: Vec::new(),
            known_failures: ledger.known_failures.clone(),
            changed_files: ledger.changed_files.clone(),
        }
    }

    // ---------------------------------------------------------- read-only API

    pub fn goal(&self) -> &str {
        &self.goal
    }

    pub fn criteria(&self) -> &[String] {
        &self.criteria
    }

    pub fn task_state(&self) -> &str {
        &self.task_state
    }

    pub fn plan_steps(&self) -> &[String] {
        &self.plan_steps
    }

    pub fn decisions(&self) -> &[ProjectedDecision] {
        &self.decisions
    }

    pub fn checks(&self) -> &[ProjectedCheck] {
        &self.checks
    }

    pub fn children(&self) -> &[ProjectedChild] {
        &self.children
    }

    pub fn known_failures(&self) -> &[String] {
        &self.known_failures
    }

    pub fn changed_files(&self) -> &[String] {
        &self.changed_files
    }

    /// The VERBATIM goal + acceptance criteria block. Never truncated,
    /// never summarized: these bytes carry user-policy instruction
    /// authority and must survive compaction byte-for-byte.
    pub fn identity_render(&self) -> String {
        let mut out = String::new();
        if !self.goal.is_empty() {
            out.push_str("GOAL (verbatim):\n");
            out.push_str(&self.goal);
            out.push('\n');
        }
        if !self.criteria.is_empty() {
            out.push_str("ACCEPTANCE CRITERIA (verbatim):\n");
            for c in &self.criteria {
                out.push_str("- ");
                out.push_str(c);
                out.push('\n');
            }
        }
        out
    }

    /// The durable non-identity facts (state, plan, decisions, checks,
    /// child progress, failures, changed files), bounded. The goal and
    /// criteria NEVER appear here: this is the only shape handed to a
    /// summarizer, so a lossy model can never rewrite the user's ask.
    pub fn summarizable_render(&self) -> String {
        let mut out = String::new();
        if !self.task_state.is_empty() {
            out.push_str(&format!(
                "STATE: {}\n",
                truncate(&self.task_state, PROJECTION_FIELD_CHARS)
            ));
        }
        if !self.plan_steps.is_empty() {
            out.push_str("OPEN STEPS:\n");
            for s in self.plan_steps.iter().take(PROJECTION_MAX_OPEN_STEPS) {
                out.push_str(&format!("- {}\n", truncate(s, PROJECTION_FIELD_CHARS)));
            }
        }
        if !self.decisions.is_empty() {
            out.push_str("DECISIONS (durable):\n");
            for d in self.decisions.iter().take(PROJECTION_MAX_DECISIONS) {
                out.push_str("- ");
                if !d.step.is_empty() {
                    out.push_str(&truncate(&d.step, 60));
                    out.push_str(": ");
                }
                out.push_str(&truncate(&d.choice, PROJECTION_FIELD_CHARS));
                if !d.rationale.is_empty() {
                    out.push_str(" — ");
                    out.push_str(&truncate(&d.rationale, PROJECTION_FIELD_CHARS));
                }
                out.push('\n');
            }
        }
        if !self.checks.is_empty() {
            out.push_str("CHECKS (durable):\n");
            for c in self.checks.iter().take(PROJECTION_MAX_CHECKS) {
                out.push_str(&format!(
                    "- {}: {} {}\n",
                    truncate(&c.name, 80),
                    truncate(&c.status, 32),
                    truncate(&c.summary, PROJECTION_FIELD_CHARS),
                ));
            }
        }
        if !self.children.is_empty() {
            out.push_str("CHILD PROGRESS:\n");
            for child in self.children.iter().take(PROJECTION_MAX_CHILDREN) {
                out.push_str(&format!(
                    "- {}: {}\n",
                    truncate(&child.purpose, PROJECTION_FIELD_CHARS),
                    truncate(
                        child.outcome.as_deref().unwrap_or("running"),
                        PROJECTION_FIELD_CHARS
                    ),
                ));
            }
        }
        if !self.known_failures.is_empty() {
            out.push_str("KNOWN FAILURES:\n");
            for f in self
                .known_failures
                .iter()
                .rev()
                .take(PROJECTION_MAX_FAILURES)
            {
                out.push_str(&format!("- {}\n", truncate(f, PROJECTION_FIELD_CHARS)));
            }
        }
        if !self.changed_files.is_empty() {
            out.push_str("CHANGED FILES: ");
            let joined = self
                .changed_files
                .iter()
                .map(|f| truncate(f, 80))
                .collect::<Vec<_>>()
                .join(", ");
            out.push_str(&truncate(&joined, 300));
            out.push('\n');
        }
        out
    }

    /// Identity + summarizable state (display/telemetry only).
    pub fn compact_render(&self) -> String {
        let mut out = self.identity_render();
        out.push_str(&self.summarizable_render());
        out
    }

    pub fn token_estimate(&self) -> usize {
        let e = super::estimator::Estimator;
        e.estimate_tokens(&self.compact_render())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(n: usize) -> TurnSummary {
        TurnSummary {
            steps_completed: vec![format!("did {n}")],
            steps_opened: vec![format!("next {n}")],
            decisions: vec![format!("decided {n}")],
            failures: vec![format!("failed {n}")],
            files_changed: vec![format!("src/f{n}.rs")],
            tests_run: vec![format!("test_{n}")],
            tests_failed: vec![],
        }
    }

    #[test]
    fn record_turn_folds_all_fields() {
        let mut l = TaskLedger {
            goal: "fix parser".into(),
            ..Default::default()
        };
        l.record_turn(&turn(1));
        assert_eq!(l.completed_steps, vec!["did 1"]);
        assert_eq!(l.decisions, vec!["decided 1"]);
        assert_eq!(l.changed_files, vec!["src/f1.rs"]);
    }

    #[test]
    fn ledger_stays_bounded_after_10k_turns() {
        let mut l = TaskLedger::default();
        for i in 0..10_000 {
            l.record_turn(&turn(i));
        }
        assert!(l.open_steps.len() <= MAX_OPEN_STEPS);
        assert!(l.decisions.len() <= MAX_DECISIONS);
        assert!(l.known_failures.len() <= MAX_FAILURES);
        assert!(l.changed_files.len() <= MAX_FILES);
        // compact_render stays under ~400 tokens.
        let render = l.compact_render();
        assert!(
            render.len() < 400 * 4,
            "render too big: {} chars",
            render.len()
        );
        assert!(
            l.token_estimate() <= 400,
            "token estimate {}",
            l.token_estimate()
        );
    }

    #[test]
    fn duplicates_are_evicted_not_accumulated() {
        let mut l = TaskLedger::default();
        for _ in 0..100 {
            l.record_turn(&turn(7)); // same content every time
        }
        assert!(l.completed_steps.iter().filter(|s| *s == "did 7").count() <= 1);
    }

    #[test]
    fn hostile_entries_rejected_by_validate() {
        let mut l = TaskLedger::default();
        l.constraints.push("x".repeat(5000));
        let err = l.validate_sane().unwrap_err();
        assert!(err.kind == ErrorKind::Oversized);
    }

    #[test]
    fn json_roundtrip_with_missing_fields() {
        // A hostile/missing field set must parse with defaults.
        let v = serde_json::json!({"goal": "g", "decisions": ["a"]});
        let l: TaskLedger = serde_json::from_value(v).unwrap();
        assert_eq!(l.goal, "g");
        assert!(l.constraints.is_empty());
        assert!(l.changed_files.is_empty());
        let back = serde_json::to_value(&l).unwrap();
        assert_eq!(back["goal"], "g");
        // Unknown fields rejected? No: serde(default) tolerates them; the
        // wire contract for the ledger is internal, so tolerate.
        let v = serde_json::json!({"goal": "g", "hack": true});
        assert!(serde_json::from_value::<TaskLedger>(v).is_ok());
    }

    #[test]
    fn render_never_panics_on_unicode_boundaries() {
        let mut l = TaskLedger::default();
        l.decisions.push("é😀".repeat(2000));
        let render = l.compact_render();
        assert!(render.is_char_boundary(render.len()));
    }

    #[test]
    fn goal_and_preferences_render() {
        let l = TaskLedger {
            goal: "ship".into(),
            user_preferences: vec!["rust first".into()],
            changed_files: vec!["src/main.rs".into()],
            ..Default::default()
        };
        let r = l.compact_render();
        assert!(r.contains("GOAL: ship"));
        assert!(r.contains("src/main.rs"));
        assert!(r.contains("CHANGED FILES"));
    }

    #[test]
    fn empty_ledger_renders_bounded() {
        let l = TaskLedger::default();
        let r = l.compact_render();
        assert!(r.is_empty());
        assert_eq!(l.token_estimate(), 0);
    }
}

#[cfg(test)]
mod projection_tests {
    use super::*;

    fn durable_rows() -> DurableTaskRows {
        DurableTaskRows {
            task_state: "Running".into(),
            goal: "GOAL-Ω-verbatim".into(),
            criteria: vec!["CRIT-Ω-one".into(), "CRIT-two".into()],
            plan_steps: vec!["step one".into()],
            decisions: vec![ProjectedDecision {
                step: "2".into(),
                choice: "DURABLE-CHOICE".into(),
                rationale: "because durable".into(),
            }],
            checks: vec![ProjectedCheck {
                name: "cargo test".into(),
                status: "passed".into(),
                summary: "all green".into(),
            }],
            children: vec![ProjectedChild {
                purpose: "child A".into(),
                outcome: Some("done".into()),
            }],
            known_failures: vec!["flaky suite".into()],
            changed_files: vec!["src/lib.rs".into()],
        }
    }

    #[test]
    fn projection_reports_durable_rows_and_identity_is_verbatim() {
        let p = TaskContextProjection::from_durable_rows(durable_rows());
        assert_eq!(p.goal(), "GOAL-Ω-verbatim");
        assert_eq!(p.criteria().len(), 2);
        assert_eq!(p.decisions()[0].choice, "DURABLE-CHOICE");
        assert_eq!(p.checks()[0].status, "passed");
        assert_eq!(p.children()[0].outcome.as_deref(), Some("done"));
        let identity = p.identity_render();
        assert!(identity.contains("GOAL-Ω-verbatim"));
        assert!(identity.contains("CRIT-Ω-one"));
        // The summarizable shape never carries the user's ask.
        let summarizable = p.summarizable_render();
        assert!(summarizable.contains("DURABLE-CHOICE"));
        assert!(summarizable.contains("CHECKS (durable):"));
        assert!(!summarizable.contains("GOAL-Ω-verbatim"));
        assert!(!summarizable.contains("CRIT-Ω-one"));
    }

    #[test]
    fn projection_identity_never_truncates_hostile_lengths() {
        // A hostile-but-legal task row: the FULL goal and criteria bytes
        // must survive identity rendering byte-for-byte (no 200-char cap
        // like the legacy ledger render).
        let goal = format!("G{}G", "😀".repeat(5_000));
        let criterion = format!("C{}C", "é".repeat(4_000));
        let p = TaskContextProjection::from_durable_rows(DurableTaskRows {
            goal: goal.clone(),
            criteria: vec![criterion.clone()],
            ..Default::default()
        });
        let identity = p.identity_render();
        assert!(identity.contains(&goal));
        assert!(identity.contains(&criterion));
        assert_eq!(
            identity.matches(&goal).count(),
            1,
            "identity must carry the goal once, byte-exact"
        );
        assert!(p.token_estimate() >= crate::estimator::Estimator.estimate_tokens(&goal));
    }

    #[test]
    fn compat_constructor_maps_legacy_ledger_rows() {
        let l = TaskLedger {
            goal: "legacy goal".into(),
            constraints: vec!["must not break".into()],
            open_steps: vec!["next".into()],
            decisions: vec!["decided A".into()],
            known_failures: vec!["broke B".into()],
            changed_files: vec!["a.rs".into()],
            tests_run: vec!["t1".into(), "t2".into()],
            tests_failed: vec!["t2".into()],
            ..Default::default()
        };
        let p = TaskContextProjection::from_task_ledger(&l);
        assert_eq!(p.goal(), "legacy goal");
        assert_eq!(p.criteria(), ["must not break"]);
        assert_eq!(p.plan_steps(), ["next"]);
        assert_eq!(p.decisions()[0].choice, "decided A");
        assert_eq!(p.checks().len(), 2);
        assert_eq!(p.checks()[1].status, "failed");
        assert!(p.identity_render().contains("legacy goal"));
    }

    #[test]
    fn task_context_projection_source_has_no_public_mutators() {
        // Audit 71: the projection must have NO write authority. This is a
        // structural source scan (the projection API, not the legacy
        // TaskLedger above): fields private, no `&mut`/`mut self`, and any
        // method that touches `self` borrows it immutably.
        let full = include_str!("ledger.rs");
        let api_source = full
            .split("#[cfg(test)]")
            .next()
            .expect("API section exists");
        let block_after = |marker: &str| -> &str {
            let start = api_source
                .find(marker)
                .unwrap_or_else(|| panic!("{marker} not found"))
                + marker.len();
            let rest = &api_source[start..];
            let end = rest.find("\n}").map(|i| i + 2).unwrap_or(rest.len());
            &rest[..end]
        };
        let struct_block = block_after("pub struct TaskContextProjection {");
        for (i, line) in struct_block.lines().enumerate() {
            assert!(
                !line.trim_start().starts_with("pub "),
                "projection field line {i} is public: {line:?}"
            );
        }
        let impl_block = block_after("impl TaskContextProjection {");
        assert!(
            !impl_block.contains("mut self"),
            "projection impl contains a mutating receiver:\n{impl_block}"
        );
        assert!(
            !impl_block.contains("&mut"),
            "projection impl mentions &mut:\n{impl_block}"
        );
        for line in impl_block.lines().filter(|l| l.contains("pub fn")) {
            if line.contains("self") {
                assert!(
                    line.contains("&self"),
                    "projection method with self receiver must borrow immutably: {line:?}"
                );
            }
        }
        assert!(
            impl_block.contains("pub fn from_durable_rows("),
            "the durable-row constructor must exist"
        );
    }
}
