//! Report types + deterministic aggregation.
//!
//! Per task the benchmark reports: verified (the repository's own
//! `verify.sh` exited 0 in the workspace copy within the wall/byte
//! budgets), criteria met, cost (durable spent-cost micro-US dollars),
//! tokens, wall time and attempts (turn count). The aggregate
//! cost-to-verified-success is total spend over verified tasks.

use crate::corpus::Lang;
use crate::score::CriterionResult;
use crate::verify::VerifyOutcome;

/// One executed (or skipped) task of the corpus.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TaskResult {
    pub task_id: String,
    pub lang: String,
    /// Documented skip reason (missing toolchain, missing provider env,
    /// missing daemon binary, …). `None` = the task ran.
    pub skipped: Option<String>,
    /// Verified success: the task's `verify.sh` exited 0 in the workspace
    /// copy (deterministic, repository-native). Never true for skipped
    /// tasks.
    pub verified: bool,
    /// Wall time the task consumed (daemon run + verification), ms.
    pub wall_ms: u64,
    /// True when the daemon run hit the per-task wall budget.
    pub timed_out: bool,
    /// Admitted logical turns of the session (daemon turn records).
    pub attempts: u64,
    /// Durable spent cost of the run in micro-US dollars (provider
    /// reported, settled by the daemon's cost ledger). Honest zero when
    /// the runtime recorded nothing.
    pub spend_micro: u64,
    /// Durable token total of the run.
    pub tokens_total: u64,
    pub criteria: Vec<CriterionResult>,
    /// Number of criteria met (deterministic summary-key scoring).
    pub criteria_met: usize,
    pub criteria_total: usize,
    /// Status of the newest durable verification record of the task, when
    /// one exists (wire text, e.g. `Passed`).
    pub record_status: Option<String>,
    /// The verification outcome of the post-run `verify.sh`, when the
    /// task ran far enough to verify.
    pub verify: Option<VerifyOutcome>,
    /// Optional free-form diagnostic note (never used for scoring).
    pub note: Option<String>,
}

impl TaskResult {
    /// A deterministic skip row.
    pub fn skipped(task_id: &str, lang: Lang, reason: String) -> Self {
        TaskResult {
            task_id: task_id.to_string(),
            lang: lang.to_string(),
            skipped: Some(reason),
            verified: false,
            wall_ms: 0,
            timed_out: false,
            attempts: 0,
            spend_micro: 0,
            tokens_total: 0,
            criteria: Vec::new(),
            criteria_met: 0,
            criteria_total: 0,
            record_status: None,
            verify: None,
            note: None,
        }
    }

    pub fn is_skipped(&self) -> bool {
        self.skipped.is_some()
    }
}

/// Deterministic aggregation over the per-task rows.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Aggregate {
    pub tasks_total: usize,
    pub tasks_ran: usize,
    pub tasks_skipped: usize,
    pub verified: usize,
    pub criteria_met: usize,
    pub criteria_total: usize,
    /// Total durable spend of every RUN task, micro-US dollars.
    pub spend_micro: u64,
    pub tokens_total: u64,
    /// Total wall time of every run task, ms.
    pub wall_ms: u64,
    pub attempts: u64,
    pub timed_out_runs: usize,
    /// Cost per VERIFIED task: total spend / verified count. `None` when
    /// nothing verified (the metric is undefined, never silently 0).
    pub cost_to_verified_micro: Option<u64>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CorpusReport {
    pub results: Vec<TaskResult>,
    pub aggregate: Aggregate,
    pub notes: Vec<String>,
}

impl CorpusReport {
    pub fn new(results: Vec<TaskResult>) -> Self {
        let ran: Vec<&TaskResult> = results.iter().filter(|r| !r.is_skipped()).collect();
        let criteria_total: usize = results.iter().map(|r| r.criteria_total).sum();
        let verified = ran.iter().filter(|r| r.verified).count();
        let spend_micro: u64 = ran.iter().map(|r| r.spend_micro).sum();
        let aggregate = Aggregate {
            tasks_total: results.len(),
            tasks_ran: ran.len(),
            tasks_skipped: results.len() - ran.len(),
            verified,
            criteria_met: results.iter().map(|r| r.criteria_met).sum(),
            criteria_total,
            spend_micro,
            tokens_total: ran.iter().map(|r| r.tokens_total).sum(),
            wall_ms: ran.iter().map(|r| r.wall_ms).sum(),
            attempts: ran.iter().map(|r| r.attempts).sum(),
            timed_out_runs: ran.iter().filter(|r| r.timed_out).count(),
            cost_to_verified_micro: if verified > 0 {
                Some(spend_micro / verified as u64)
            } else {
                None
            },
        };
        CorpusReport {
            results,
            aggregate,
            notes: Vec::new(),
        }
    }

    /// JSON lines: one compact row per task (stable field order).
    pub fn to_json_lines(&self) -> String {
        self.results
            .iter()
            .map(|r| serde_json::to_string(r).unwrap_or_else(|_| "{}".into()))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ran(task: &str, verified: bool, spend_micro: u64) -> TaskResult {
        TaskResult {
            task_id: task.into(),
            lang: "rust".into(),
            skipped: None,
            verified,
            wall_ms: 1000,
            timed_out: false,
            attempts: 3,
            spend_micro,
            tokens_total: 100,
            criteria: Vec::new(),
            criteria_met: 2,
            criteria_total: 4,
            record_status: None,
            verify: None,
            note: None,
        }
    }

    #[test]
    fn aggregate_splits_skips_and_computes_cost_to_verified() {
        let report = CorpusReport::new(vec![
            ran("a", true, 150),
            ran("b", true, 50),
            ran("c", false, 300),
            TaskResult::skipped("d", Lang::Python, "toolchain missing: python3".into()),
        ]);
        let a = &report.aggregate;
        assert_eq!(a.tasks_total, 4);
        assert_eq!(a.tasks_ran, 3);
        assert_eq!(a.tasks_skipped, 1);
        assert_eq!(a.verified, 2);
        assert_eq!(a.spend_micro, 500);
        assert_eq!(a.cost_to_verified_micro, Some(250));
        // Skipped rows never inflate the spend side.
        assert_eq!(report.results[3].spend_micro, 0);
    }

    #[test]
    fn cost_to_verified_is_undefined_not_zero_when_nothing_verified() {
        let report = CorpusReport::new(vec![ran("a", false, 300)]);
        assert_eq!(report.aggregate.cost_to_verified_micro, None);
    }

    #[test]
    fn json_lines_are_stable_and_complete() {
        let report = CorpusReport::new(vec![
            ran("a", true, 7),
            TaskResult::skipped("b", Lang::Java, "no daemon binary".into()),
        ]);
        let lines = report.to_json_lines();
        let mut it = lines.lines();
        let first: serde_json::Value = serde_json::from_str(it.next().unwrap()).unwrap();
        assert_eq!(first["task_id"], "a");
        assert_eq!(first["verified"], true);
        assert_eq!(first["spend_micro"], 7);
        let second: serde_json::Value = serde_json::from_str(it.next().unwrap()).unwrap();
        assert_eq!(second["skipped"], "no daemon binary");
        assert!(it.next().is_none());
    }
}
