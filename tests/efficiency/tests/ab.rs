//! Paired A/B runner tests: quality parity first, per-KPI deltas second,
//! pairing/determinism enforced, and the wiring ledger explicit.

use faktor_tests_efficiency::*;

#[test]
fn all_on_pair_keeps_quality_and_reports_gains() {
    let report = run_paired(scripted_corpus(), EfficiencyVariant::all_on(), PAIR_SEED).unwrap();
    assert!(
        report.verified_parity,
        "quality must never regress: {:?}",
        report.quality_regressions
    );
    let baseline_verified = report
        .baseline
        .tasks
        .iter()
        .filter(|t| t.metrics.verified)
        .count();
    let candidate_verified = report
        .candidate
        .tasks
        .iter()
        .filter(|t| t.metrics.verified)
        .count();
    assert_eq!(
        baseline_verified, 4,
        "the corpus has one unverified control"
    );
    assert_eq!(candidate_verified, baseline_verified);

    // The evidence pipeline really shrank (CCR + semantic selection), and
    // the semantic selection retrieved fewer items.
    let sent = report.kpi("evidence_sent_tokens").expect("delta present");
    assert!(sent.delta < 0, "evidence sent must shrink: {sent:?}");
    let retrievals = report
        .kpi("evidence_retrieval_count")
        .expect("delta present");
    assert!(
        retrievals.delta < 0,
        "retrievals must shrink: {retrievals:?}"
    );

    // Failure learning removed a repeated repair; rework-aware routing
    // removed a conflict.
    let repairs = report.kpi("repair_turns").expect("delta present");
    assert!(repairs.delta < 0, "repair turns must shrink: {repairs:?}");
    let failed = report.kpi("failed_edits").expect("delta present");
    assert!(failed.delta < 0, "failed edits must shrink: {failed:?}");

    // Flags never game human steering, and the scripted wall time can only
    // shrink (a removed repair call shortens the scripted span) — it must
    // never grow.
    let interventions = report.kpi("human_interventions").expect("delta present");
    assert_eq!(interventions.delta, 0);
    let wall = report.kpi("wall_ms").expect("delta present");
    assert!(wall.delta <= 0, "wall must not grow: {wall:?}");

    // Same verified work on fewer tokens => a better weighted KPI.
    let baseline = report.baseline_work.work_per_token.expect("tokens > 0");
    let candidate = report.candidate_work.work_per_token.expect("tokens > 0");
    assert!(
        candidate > baseline,
        "verified work per token must improve: {baseline} -> {candidate}"
    );

    for line in report.summary_lines() {
        eprintln!("{line}");
    }
}

#[test]
fn every_flag_is_paired_and_moves_its_own_kpi() {
    let expected: [(EfficiencyFlag, &str); 5] = [
        (EfficiencyFlag::Ccr, "evidence_sent_tokens"),
        (EfficiencyFlag::SemanticContext, "evidence_retrieval_count"),
        (EfficiencyFlag::FailureLearning, "repair_turns"),
        (EfficiencyFlag::TypedHandoff, "input_tokens_total"),
        (EfficiencyFlag::ReworkAwareRouting, "failed_edits"),
    ];
    for (flag, kpi) in expected {
        let variant = flag.with(EfficiencyVariant::BASELINE, true);
        assert!(!variant.is_baseline());
        assert_eq!(variant.name(), flag.label());
        let report = run_paired(scripted_corpus(), variant, PAIR_SEED).unwrap();
        assert!(
            report.verified_parity,
            "{} regressed quality: {:?}",
            flag.label(),
            report.quality_regressions
        );
        let delta = report
            .kpi(kpi)
            .unwrap_or_else(|| panic!("{} has no {kpi} delta", flag.label()));
        assert!(
            delta.delta < 0,
            "{} must shrink {kpi}, got {delta:?}",
            flag.label()
        );
    }
}

#[test]
fn pairing_refuses_a_changed_start_revision() {
    let baseline = run_arm(scripted_corpus(), EfficiencyVariant::BASELINE, PAIR_SEED).unwrap();
    let mut candidate = run_arm(scripted_corpus(), EfficiencyVariant::all_on(), PAIR_SEED).unwrap();
    candidate.tasks[0].base_revision = "rev-mutated-not-paired".to_string();
    match PairedReport::assemble(baseline, candidate) {
        Err(EfficiencyError::Malformed(message)) => {
            assert!(message.contains("pairing broken"), "{message}")
        }
        other => panic!("expected a pairing refusal, got {other:?}"),
    }
}

#[test]
fn quality_regression_refuses_all_gain_claims() {
    let baseline = run_arm(scripted_corpus(), EfficiencyVariant::BASELINE, PAIR_SEED).unwrap();
    let mut candidate = run_arm(scripted_corpus(), EfficiencyVariant::all_on(), PAIR_SEED).unwrap();
    // An adversary hands the candidate duplicate metrics from an unverified
    // task for a baseline-verified one: the report must refuse the gains.
    let unverified = candidate
        .tasks
        .iter()
        .find(|t| !t.metrics.verified)
        .unwrap()
        .metrics;
    candidate.tasks[0].metrics = unverified;
    let report = PairedReport::assemble(baseline, candidate).unwrap();
    assert!(!report.verified_parity);
    assert!(!report.quality_regressions.is_empty());
    assert!(
        report.deltas.is_empty(),
        "no KPI delta may be claimed after a quality regression"
    );
}

#[test]
fn paired_runs_are_deterministic() {
    let corpus = scripted_corpus();
    let a = run_paired(corpus, EfficiencyVariant::all_on(), PAIR_SEED).unwrap();
    let b = run_paired(corpus, EfficiencyVariant::all_on(), PAIR_SEED).unwrap();
    let metrics_a: Vec<_> = a.candidate.tasks.iter().map(|t| t.metrics).collect();
    let metrics_b: Vec<_> = b.candidate.tasks.iter().map(|t| t.metrics).collect();
    assert_eq!(
        metrics_a, metrics_b,
        "durable KPI derivation is deterministic"
    );
    assert_eq!(a.summary_lines(), b.summary_lines());
}

#[test]
fn wiring_ledger_is_explicit_and_complete() {
    let honesty = wiring_honesty();
    assert_eq!(honesty.len(), FLAGS.len());
    for (row, flag) in honesty.iter().zip(FLAGS) {
        assert_eq!(row.flag, flag);
        assert_eq!(row.status.label(), "wired-into-harness");
        assert!(!row.component.is_empty());
    }
    let report = run_paired(scripted_corpus(), EfficiencyVariant::all_on(), PAIR_SEED).unwrap();
    assert_eq!(report.wiring, honesty);
}
