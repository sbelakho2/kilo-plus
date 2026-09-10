//! Verified Work per Token: weighted math + criterion-splitting resistance.

use faktor_tests_efficiency::{verified_work_score, TaskWorkRow};

fn row(
    task_id: &str,
    weight: u64,
    criteria_total: u32,
    criteria_passed: u32,
    verified: bool,
    tokens: u64,
) -> TaskWorkRow {
    TaskWorkRow {
        task_id: task_id.to_string(),
        weight,
        criteria_total,
        criteria_passed,
        verified,
        tokens,
    }
}

#[test]
fn weighted_math_is_exact() {
    let score = verified_work_score(&[
        row("a", 3, 2, 2, true, 100),
        row("b", 1, 3, 0, false, 100),
        row("c", 2, 2, 2, true, 300),
    ]);
    assert_eq!(score.verified_work, 5.0);
    assert_eq!(score.tokens, 500);
    assert_eq!(score.work_per_token, Some(5.0 / 500.0));
}

#[test]
fn no_tokens_yields_undefined_not_zero() {
    let score = verified_work_score(&[row("a", 3, 2, 2, true, 0)]);
    assert_eq!(score.verified_work, 3.0);
    assert_eq!(score.work_per_token, None, "0/0 is undefined, never zero");
}

#[test]
fn criterion_splitting_cannot_inflate_the_score() {
    let original = [row("a", 5, 2, 2, true, 400), row("b", 2, 1, 1, true, 100)];
    let split = [
        // Same task, one criterion split into five extra passing criteria.
        row("a", 5, 7, 7, true, 400),
        // Same task with extra UNMET criteria, too.
        row("b", 2, 6, 1, true, 100),
    ];
    let before = verified_work_score(&original);
    let after = verified_work_score(&split);
    assert_eq!(before, after, "criteria are coverage evidence, not weight");

    // Show why the resistance is meaningful: a naive per-criterion scorer
    // would inflate the split rows. This is the scorer this KPI must NOT be.
    fn naive_per_criterion(rows: &[TaskWorkRow]) -> f64 {
        rows.iter()
            .filter(|r| r.verified)
            .map(|r| r.weight as f64 * f64::from(r.criteria_passed))
            .sum()
    }
    assert!(
        naive_per_criterion(&split) > naive_per_criterion(&original),
        "the naive scorer inflates under splitting; the weighted KPI does not"
    );
}

#[test]
fn unverified_weight_never_enters_the_numerator() {
    let score = verified_work_score(&[
        row("done", 4, 3, 3, true, 200),
        row("failed", 100, 3, 3, false, 10_000),
    ]);
    assert_eq!(score.verified_work, 4.0);
    assert_eq!(score.tokens, 10_200);
    assert_eq!(score.work_per_token, Some(4.0 / 10_200.0));
}
