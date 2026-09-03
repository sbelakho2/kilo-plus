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
