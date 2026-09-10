//! Paired A/B execution + reporting.
//!
//! The contract, in order:
//! 1. both arms run the SAME corpus with the SAME provider seed and record
//!    the SAME starting revisions (`assemble` refuses a broken pair);
//! 2. verified-success parity is checked FIRST — if any task that verified
//!    in the baseline fails to verify in the candidate, the report carries
//!    the regression and claims NO KPI deltas at all;
//! 3. only then are per-KPI deltas computed, and only over the PAIRED
//!    VERIFIED set (tasks verified in both arms), so unverified work cannot
//!    flatter either side.

use std::sync::Arc;

use faktor_core::id::{SessionId, TaskId};
use faktor_store::Store;

use crate::harness::{run_scripted_task, seed_router_history, ScriptedTask, TaskPlan};
use crate::metrics::{EfficiencyError, TaskEfficiencyMetrics};
use crate::variant::{wiring_honesty, EfficiencyVariant, VariantHonesty};
use crate::work::{verified_work_score, TaskWorkRow, VerifiedWorkScore};

/// One task's result in one arm: identities, benchmark weight/criteria, the
/// derived durable metrics, and the executed plan.
#[derive(Debug, Clone)]
pub struct TaskArmResult {
    pub task_id: String,
    pub base_revision: String,
    pub session: SessionId,
    pub task: TaskId,
    pub weight: u64,
    pub criteria_total: u32,
    pub criteria_passed: u32,
    pub metrics: TaskEfficiencyMetrics,
    pub plan: TaskPlan,
}

/// One executed arm.
#[derive(Debug, Clone)]
pub struct ArmOutcome {
    pub variant: EfficiencyVariant,
    pub seed: u64,
    pub tasks: Vec<TaskArmResult>,
}

/// One KPI's paired delta over the paired verified set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KpiDelta {
    pub kpi: &'static str,
    pub baseline: u64,
    pub candidate: u64,
    /// `candidate - baseline`, signed: a negative delta is the gain.
    pub delta: i128,
}

/// The complete paired report. `wiring` is the honest per-flag wiring ledger
/// (see [`crate::variant::wiring_honesty`]).
#[derive(Debug, Clone)]
pub struct PairedReport {
    pub baseline: ArmOutcome,
    pub candidate: ArmOutcome,
    /// True when no baseline-verified task lost verification.
    pub verified_parity: bool,
    /// Task ids that regressed (empty when parity holds).
    pub quality_regressions: Vec<String>,
    /// Per-KPI deltas over `deltas_scope`; EMPTY when parity failed (no gain
    /// may be claimed before quality parity).
    pub deltas: Vec<KpiDelta>,
    pub deltas_scope: &'static str,
    pub baseline_work: VerifiedWorkScore,
    pub candidate_work: VerifiedWorkScore,
    pub wiring: [VariantHonesty; 5],
}

type KpiSelector = fn(&TaskEfficiencyMetrics) -> u64;

const KPI_SELECTORS: [(&str, KpiSelector); 14] = [
    ("input_tokens_total", |m| m.input_tokens_total),
    ("input_tokens_cached", |m| m.input_tokens_cached),
    ("input_tokens_noncached", |m| m.input_tokens_noncached),
    ("output_tokens", |m| m.output_tokens),
    ("model_cost_micro", |m| m.model_cost_micro),
    ("evidence_original_bytes", |m| m.evidence_original_bytes),
    ("evidence_sent_tokens", |m| m.evidence_sent_tokens),
    ("evidence_retrieval_count", |m| m.evidence_retrieval_count),
    ("model_calls", |m| m.model_calls),
    ("repair_turns", |m| m.repair_turns),
    ("failed_edits", |m| m.failed_edits),
    ("reverted_edits", |m| m.reverted_edits),
    ("human_interventions", |m| m.human_interventions),
    ("wall_ms", |m| m.wall_ms),
];

fn work_rows(arm: &ArmOutcome) -> Vec<TaskWorkRow> {
    arm.tasks
        .iter()
        .map(|task| TaskWorkRow {
            task_id: task.task_id.clone(),
            weight: task.weight,
            criteria_total: task.criteria_total,
            criteria_passed: task.criteria_passed,
            verified: task.metrics.verified,
            tokens: task
                .metrics
                .input_tokens_total
                .saturating_add(task.metrics.output_tokens),
        })
        .collect()
}

/// Execute one corpus arm through the REAL store schema. The tempdir holds
/// only scripted reads/writes; it is deleted when the arm returns (all KPIs
/// are derived inside this call).
pub fn run_arm(
    corpus: &[ScriptedTask],
    variant: EfficiencyVariant,
    seed: u64,
) -> Result<ArmOutcome, EfficiencyError> {
    let dir =
        tempfile::tempdir().map_err(|e| EfficiencyError::Malformed(format!("tempdir: {e}")))?;
    let store = Arc::new(Store::open_fast(dir.path())?);
    let workspace =
        store.create_workspace(dir.path().join("workspace").to_string_lossy().as_ref())?;
    seed_router_history(&store)?;

    let mut tasks = Vec::with_capacity(corpus.len());
    for (index, task) in corpus.iter().enumerate() {
        let run = run_scripted_task(&store, workspace, index, task, variant, seed)?;
        let metrics = TaskEfficiencyMetrics::derive(&store, run.session, run.task)?;
        let criteria_total = task.criteria.len() as u32;
        tasks.push(TaskArmResult {
            task_id: task.id.to_string(),
            base_revision: run.base_revision,
            session: run.session,
            task: run.task,
            weight: task.weight,
            criteria_total,
            criteria_passed: if metrics.verified { criteria_total } else { 0 },
            metrics,
            plan: run.plan,
        });
    }
    Ok(ArmOutcome {
        variant,
        seed,
        tasks,
    })
}

impl PairedReport {
    /// Assemble (and verify) a report from two arms. Quality parity is
    /// enforced before any delta is computed.
    pub fn assemble(baseline: ArmOutcome, candidate: ArmOutcome) -> Result<Self, EfficiencyError> {
        if baseline.seed != candidate.seed {
            return Err(EfficiencyError::Malformed(format!(
                "paired arms must share the provider seed (baseline {}, candidate {})",
                baseline.seed, candidate.seed
            )));
        }
        if baseline.tasks.len() != candidate.tasks.len() {
            return Err(EfficiencyError::Malformed(format!(
                "paired arms must run the same corpus (baseline {} tasks, candidate {})",
                baseline.tasks.len(),
                candidate.tasks.len()
            )));
        }
        let mut quality_regressions = Vec::new();
        for (b, c) in baseline.tasks.iter().zip(&candidate.tasks) {
            if b.task_id != c.task_id {
                return Err(EfficiencyError::Malformed(format!(
                    "pairing broken: baseline task {} vs candidate task {}",
                    b.task_id, c.task_id
                )));
            }
            if b.base_revision != c.base_revision {
                return Err(EfficiencyError::Malformed(format!(
                    "pairing broken: task {} starts at {} in the baseline and {} in the candidate",
                    b.task_id, b.base_revision, c.base_revision
                )));
            }
            if b.metrics.verified && !c.metrics.verified {
                quality_regressions.push(b.task_id.clone());
            }
        }
        let verified_parity = quality_regressions.is_empty();
        let deltas = if verified_parity {
            KPI_SELECTORS
                .iter()
                .map(|(kpi, selector)| {
                    let baseline_sum: u64 = baseline
                        .tasks
                        .iter()
                        .filter(|t| t.metrics.verified)
                        .map(|t| selector(&t.metrics))
                        .fold(0u64, u64::saturating_add);
                    let candidate_sum: u64 = candidate
                        .tasks
                        .iter()
                        .filter(|t| t.metrics.verified)
                        .map(|t| selector(&t.metrics))
                        .fold(0u64, u64::saturating_add);
                    KpiDelta {
                        kpi,
                        baseline: baseline_sum,
                        candidate: candidate_sum,
                        delta: i128::from(candidate_sum) - i128::from(baseline_sum),
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        let baseline_work = verified_work_score(&work_rows(&baseline));
        let candidate_work = verified_work_score(&work_rows(&candidate));
        Ok(PairedReport {
            baseline,
            candidate,
            verified_parity,
            quality_regressions,
            deltas,
            deltas_scope: "tasks verified in BOTH arms",
            baseline_work,
            candidate_work,
            wiring: wiring_honesty(),
        })
    }

    pub fn kpi(&self, kpi: &str) -> Option<&KpiDelta> {
        self.deltas.iter().find(|d| d.kpi == kpi)
    }

    /// Human-readable summary lines (stable order).
    pub fn summary_lines(&self) -> Vec<String> {
        let mut lines = vec![
            format!(
                "variant={} seed={} verified_parity={} scope={}",
                self.candidate.variant.name(),
                self.candidate.seed,
                self.verified_parity,
                self.deltas_scope
            ),
            format!(
                "verified baseline={}/{} candidate={}/{}",
                self.baseline
                    .tasks
                    .iter()
                    .filter(|t| t.metrics.verified)
                    .count(),
                self.baseline.tasks.len(),
                self.candidate
                    .tasks
                    .iter()
                    .filter(|t| t.metrics.verified)
                    .count(),
                self.candidate.tasks.len()
            ),
            format!(
                "verified_work/token baseline={:?} candidate={:?}",
                self.baseline_work.work_per_token, self.candidate_work.work_per_token
            ),
        ];
        for delta in &self.deltas {
            lines.push(format!(
                "kpi {}: {} -> {} (delta {})",
                delta.kpi, delta.baseline, delta.candidate, delta.delta
            ));
        }
        for regression in &self.quality_regressions {
            lines.push(format!("QUALITY REGRESSION: {regression}"));
        }
        lines
    }
}

/// Run the baseline arm and the candidate arm over the same corpus/seed and
/// assemble the parity-checked report.
pub fn run_paired(
    corpus: &[ScriptedTask],
    variant: EfficiencyVariant,
    seed: u64,
) -> Result<PairedReport, EfficiencyError> {
    let baseline = run_arm(corpus, EfficiencyVariant::BASELINE, seed)?;
    let candidate = run_arm(corpus, variant, seed)?;
    PairedReport::assemble(baseline, candidate)
}
