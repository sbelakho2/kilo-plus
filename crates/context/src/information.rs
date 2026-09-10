//! Information-gain context selection (audit 41/42): instead of ranking
//! candidates by raw utility alone, the planner scores what a candidate
//! still ADDS to the needs the turn must satisfy.
//!
//! # Marginal gain (documented contract)
//!
//! ```text
//! marginal_gain(c, S) = Σ_n  w_n
//!                            · remaining_coverage(n, S)
//!                            · candidate_coverage(c, n)
//!                            · confidence(c)
//!                            · freshness(c)
//! ```
//!
//! where:
//!
//! - `w_n` is [`Need::weight`] — a NaN, infinite or negative weight
//!   contributes 0 (fail-safe);
//! - `remaining_coverage(n, S)` is the uncovered fraction of need `n`
//!   after the already selected set `S`, clamped to `[0, 1]`;
//! - `candidate_coverage(c, n)` is the candidate's declared
//!   [`NeedCoverage::coverage_ppm`] for the need (duplicates summed,
//!   saturating at full coverage), clamped to `[0, 1]`;
//! - `confidence(c)`/`freshness(c)` are the candidate's ppm fields clamped
//!   to `[0, 1]` — a zero or garbage ppm never inflates a score.
//!
//! Every ppm input is divided by `1_000_000` and clamped before any
//! multiplication; a non-finite or negative result is clamped to `0.0`.
//!
//! # Selection
//!
//! [`select_by_information`] ranks eligible candidates by
//! `marginal_gain / estimate_tokens` descending (gains recomputed after
//! every pick, so redundant evidence has diminishing returns), with
//! [`ContextCandidate::id`] ascending as the deterministic tie-break. The
//! phases are:
//!
//! 1. **Required coverage.** Every [`Need::required`] need is covered to
//!    full declared coverage first: [`required_candidates`] force-includes
//!    the minimal deterministic set of candidates that gets there
//!    (Required-marked candidates preferred, then Preferred, then
//!    Optional; declared coverage descending; gain per token descending;
//!    id ascending). A REQUIRED set that alone exceeds the token budget
//!    is a typed [`InformationError::Oversized`] — it is never silently
//!    dropped or truncated.
//! 2. **Information gain.** Remaining candidates compete greedily by gain
//!    per token while they fit the budget.
//! 3. **Conversation.** Message candidates absorb whatever budget the
//!    information phases left, newest-first as the caller ordered them,
//!    stopping at the first message that no longer fits (a contiguous
//!    newest prefix).
//!
//! One S/M/L level per evidence group: candidates sharing
//! `evidence = Some(group)` are variants of one evidence item, and the
//! first one selected suppresses every other variant of that group.

use crate::selection::{CandidateKind, CandidateRequirement, ContextCandidate};

/// ppm denominator: `1_000_000` ppm = 1.0.
const PPM: f64 = 1_000_000.0;
/// Fully covered need, in ppm.
const FULL_COVERAGE_PPM: u64 = 1_000_000;

/// One information need the turn must satisfy.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Need {
    /// Stable need identity; [`NeedCoverage::need_id`] matches it.
    pub id: String,
    /// Relative importance of the need. Non-finite/negative weights are
    /// treated as `0` by every gain computation.
    pub weight: f64,
    /// A required need is covered before any gain-ranked selection and can
    /// never be dropped: if covering it does not fit the budget, selection
    /// fails with [`InformationError::Oversized`].
    pub required: bool,
}

/// The budget one information selection runs under: a token budget plus
/// the needs the selected content must satisfy.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct InformationBudget {
    /// Tokens available to the selection (never exceeded unless selection
    /// fails with [`InformationError::Oversized`]).
    pub token_budget: u32,
    /// Needs to satisfy. Required needs are protected; all needs drive the
    /// marginal-gain ranking.
    pub needs: Vec<Need>,
}

/// The additive failure-aware omission prior seam (audit 68; the learning
/// crate's [`faktor_learning::adjusted_gain`]/`context_prior` contract).
///
/// A `FailurePrior` is any handle that can state, per candidate, the
/// OMISSION RISK of leaving that candidate out of the window. The consumer —
/// never the prior — owns the arithmetic: every risk is passed through
/// [`prior_adjusted_gain`], i.e. `base * clamp(risk, 1.0, 2.0)` via the
/// learning crate's canonical formula, with non-finite risk (NaN, `inf`,
/// `-inf`) treated as the unknown-risk maximum `2.0` and the product
/// sanitized to a finite, non-negative value. The clamp means an omission
/// risk can only ever PROTECT a candidate (raise its gain up to 2x); it can
/// never demote one below its base gain.
///
/// Required criteria are untouchable: [`select_by_information_with_prior`]
/// and the planner never consult the prior for a
/// [`CandidateRequirement::Required`] candidate, and the required-coverage
/// phase is not gain-ranked at all. A hostile prior therefore cannot boost,
/// demote, or even observe required content.
pub trait FailurePrior {
    /// Omission risk of excluding `candidate`. Values below `1.0`/above
    /// `2.0` are clamped and non-finite values are the unknown-risk maximum
    /// `2.0`, exactly per [`prior_adjusted_gain`]; `1.0` is neutral.
    fn omission_risk(&self, candidate: &ContextCandidate) -> f64;
}

/// Apply the audit-68 formula to one base gain: `base * clamp(risk, 1, 2)`
/// through [`faktor_learning::adjusted_gain`], then sanitized to a finite,
/// non-negative result (a hostile base or risk can never produce
/// NaN/inf/negative). Callers MUST NOT invoke this for
/// [`CandidateRequirement::Required`] candidates — required criteria are
/// untouchable.
pub fn prior_adjusted_gain(base: f64, omission_risk: f64) -> f64 {
    let adjusted = faktor_learning::adjusted_gain(base, omission_risk);
    if adjusted.is_finite() && adjusted > 0.0 {
        adjusted
    } else {
        0.0
    }
}

/// Typed failure of [`select_by_information`] / the information-aware
/// planner: the content REQUIRED to cover required needs does not fit the
/// budget. The caller must shrink the conversation/evidence or raise the
/// budget; nothing is silently dropped and no partial required set is
/// returned.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum InformationError {
    /// `required_tokens` tokens of required content against a
    /// `token_budget`-token budget.
    #[error(
        "required evidence alone exceeds the token budget ({required_tokens} > {token_budget})"
    )]
    Oversized {
        /// Total tokens of the minimal set covering all required needs.
        required_tokens: u64,
        /// The budget the selection ran under.
        token_budget: u32,
    },
}

/// A successful information selection: the chosen candidates in plan
/// order, their token total, and the tokens the required phase consumed.
#[derive(Debug, Clone, PartialEq)]
pub struct InformationSelection {
    /// Selected candidates: message candidates first (newest-first input
    /// order), then required/gain-ranked evidence in selection order.
    pub selected: Vec<ContextCandidate>,
    /// `Σ estimate_tokens` of [`InformationSelection::selected`].
    pub selected_tokens: u32,
    /// Tokens of the force-included required set (a subset of
    /// `selected_tokens`).
    pub required_tokens: u32,
}

/// Clamp a ppm quantity into the unit interval: `ppm / 1e6` capped at 1.0.
fn ppm_unit(ppm: u32) -> f64 {
    if ppm >= FULL_COVERAGE_PPM as u32 {
        1.0
    } else {
        f64::from(ppm) / PPM
    }
}

/// Neutralize a need weight: NaN, infinite or non-positive weights are `0`.
fn sanitize_weight(weight: f64) -> f64 {
    if weight.is_finite() && weight > 0.0 {
        weight
    } else {
        0.0
    }
}

/// Raw declared coverage of `need_id`, in ppm, duplicated entries summed
/// and saturating at full coverage.
fn coverage_ppm_for(candidate: &ContextCandidate, need_id: &str) -> u64 {
    let mut ppm: u64 = 0;
    for entry in &candidate.need_coverage {
        if entry.need_id == need_id {
            ppm = ppm.saturating_add(u64::from(entry.coverage_ppm));
            if ppm >= FULL_COVERAGE_PPM {
                return FULL_COVERAGE_PPM;
            }
        }
    }
    ppm.min(FULL_COVERAGE_PPM)
}

/// The candidate's declared coverage of `need_id` in `[0, 1]`, ppm
/// clamped and duplicates summed.
pub fn candidate_coverage(candidate: &ContextCandidate, need_id: &str) -> f64 {
    ppm_unit(u32::try_from(coverage_ppm_for(candidate, need_id)).unwrap_or(u32::MAX))
}

/// The remaining (uncovered) fraction of `need` after `selected`, in
/// `[0, 1]`. Coverage is the ppm-saturating sum of every selected
/// candidate's declaration for the need id.
pub fn remaining_coverage(need: &Need, selected: &[ContextCandidate]) -> f64 {
    let mut covered: u64 = 0;
    for candidate in selected {
        covered = covered.saturating_add(coverage_ppm_for(candidate, &need.id));
        if covered >= FULL_COVERAGE_PPM {
            return 0.0;
        }
    }
    let remaining = 1.0 - (covered as f64) / PPM;
    if remaining.is_finite() {
        remaining.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// The documented marginal gain of adding `candidate` to `selected`
/// against `needs` (see the module docs for the formula and its fail-safe
/// clamps). Never NaN, never negative, never infinite: non-finite or
/// negative results are clamped to `0.0`.
pub fn marginal_gain(
    candidate: &ContextCandidate,
    selected: &[ContextCandidate],
    needs: &[Need],
) -> f64 {
    let confidence = ppm_unit(candidate.confidence_ppm);
    let freshness = ppm_unit(candidate.freshness_ppm);
    if confidence <= 0.0 || freshness <= 0.0 {
        return 0.0;
    }
    let mut gain = 0.0f64;
    for need in needs {
        let weight = sanitize_weight(need.weight);
        if weight <= 0.0 {
            continue;
        }
        let remaining = remaining_coverage(need, selected);
        if remaining <= 0.0 {
            continue;
        }
        let coverage = candidate_coverage(candidate, &need.id);
        if coverage <= 0.0 {
            continue;
        }
        gain += weight * remaining * coverage * confidence * freshness;
    }
    if gain.is_finite() && gain > 0.0 {
        gain
    } else {
        0.0
    }
}

/// Gain per token used to rank candidates. A zero-token candidate has no
/// price and scores `0.0` (and is never selected).
fn score(candidate: &ContextCandidate, selected: &[ContextCandidate], needs: &[Need]) -> f64 {
    if candidate.estimate_tokens == 0 {
        return 0.0;
    }
    marginal_gain(candidate, selected, needs) / f64::from(candidate.estimate_tokens)
}

/// Requirement preference inside the required phase: Required first, then
/// Preferred, then Optional.
fn requirement_rank(requirement: CandidateRequirement) -> u8 {
    match requirement {
        CandidateRequirement::Required => 0,
        CandidateRequirement::Preferred => 1,
        CandidateRequirement::Optional => 2,
    }
}

/// A candidate from the same evidence group was already selected: the
/// S/M/L exclusivity rule. Ungrouped (`evidence = None`) candidates never
/// conflict.
fn group_conflict(selected: &[ContextCandidate], candidate: &ContextCandidate) -> bool {
    match candidate.evidence {
        Some(group) => selected.iter().any(|chosen| chosen.evidence == Some(group)),
        None => false,
    }
}

/// A candidate with this id is already in the selection: never duplicate.
fn already_selected(selected: &[ContextCandidate], candidate: &ContextCandidate) -> bool {
    selected.iter().any(|chosen| chosen.id == candidate.id)
}

/// Deterministic ranking inside [`required_candidates`]: requirement class,
/// then declared coverage of the need (descending), then gain per token
/// (descending), then id ascending.
fn compare_for_need(
    a: &ContextCandidate,
    b: &ContextCandidate,
    need_id: &str,
    selected: &[ContextCandidate],
    needs: &[Need],
) -> std::cmp::Ordering {
    requirement_rank(a.requirement)
        .cmp(&requirement_rank(b.requirement))
        .then_with(|| coverage_ppm_for(b, need_id).cmp(&coverage_ppm_for(a, need_id)))
        .then_with(|| score(b, selected, needs).total_cmp(&score(a, selected, needs)))
        .then_with(|| a.id.cmp(&b.id))
}

/// The minimal deterministic set of candidates that must be included so
/// every required [`Need`] reaches full declared coverage. Required needs
/// are resolved in input order; previously chosen candidates contribute
/// any coverage they declare for later needs. A required need no candidate
/// can cover is left uncovered (there is nothing to include), never
/// fabricated. The result may exceed a token budget — callers detect that
/// via [`InformationError::Oversized`] and MUST NOT drop it silently.
pub fn required_candidates(
    candidates: &[ContextCandidate],
    needs: &[Need],
) -> Vec<ContextCandidate> {
    let mut selected: Vec<ContextCandidate> = Vec::new();
    for need in needs.iter().filter(|need| need.required) {
        let mut covered: u64 = selected
            .iter()
            .map(|candidate| coverage_ppm_for(candidate, &need.id))
            .fold(0u64, u64::saturating_add)
            .min(FULL_COVERAGE_PPM);
        while covered < FULL_COVERAGE_PPM {
            let mut pool: Vec<&ContextCandidate> = candidates
                .iter()
                .filter(|candidate| {
                    candidate.estimate_tokens > 0
                        && !already_selected(&selected, candidate)
                        && !group_conflict(&selected, candidate)
                        && coverage_ppm_for(candidate, &need.id) > 0
                })
                .collect();
            pool.sort_by(|a, b| compare_for_need(a, b, &need.id, &selected, needs));
            let Some(next) = pool.into_iter().next() else {
                break; // nothing left that can cover this need
            };
            covered = covered
                .saturating_add(coverage_ppm_for(next, &need.id))
                .min(FULL_COVERAGE_PPM);
            selected.push(next.clone());
        }
    }
    selected
}

/// Select the context window under `budget` by information gain (audit
/// 41/42). Pure and deterministic: identical input yields an identical
/// selection, and score ties always break on id.
///
/// Required needs are covered first by [`required_candidates`]; when that
/// set alone exceeds `budget.token_budget` the selection fails with
/// [`InformationError::Oversized`] instead of dropping required content.
/// The remaining budget is then filled greedily by
/// [`marginal_gain`]/token (recomputed after every pick), then by Message
/// candidates as a contiguous newest-first prefix. Zero-token candidates
/// are never selected; a candidate id is never selected twice; at most one
/// level per evidence group is selected.
pub fn select_by_information(
    candidates: &[ContextCandidate],
    budget: &InformationBudget,
) -> Result<InformationSelection, InformationError> {
    select_by_information_with(candidates, budget, &|_candidate, base| base)
}

/// [`select_by_information`] with the failure-aware omission prior (audit
/// 68): every non-Required candidate's marginal gain is passed through
/// [`prior_adjusted_gain`] (`base * clamp(risk, 1, 2)`), so omission risk can
/// only protect a candidate up to 2x — it can never demote one. Required
/// criteria are untouchable: the required-coverage phase is byte-identical
/// to [`select_by_information`] and the prior is never even consulted for a
/// [`CandidateRequirement::Required`] candidate.
pub fn select_by_information_with_prior(
    candidates: &[ContextCandidate],
    budget: &InformationBudget,
    prior: &(dyn FailurePrior + Send + Sync),
) -> Result<InformationSelection, InformationError> {
    select_by_information_with(candidates, budget, &|candidate, base| {
        if candidate.requirement == CandidateRequirement::Required {
            base
        } else {
            prior_adjusted_gain(base, prior.omission_risk(candidate))
        }
    })
}

fn select_by_information_with(
    candidates: &[ContextCandidate],
    budget: &InformationBudget,
    gain_of: &impl Fn(&ContextCandidate, f64) -> f64,
) -> Result<InformationSelection, InformationError> {
    let token_budget = u64::from(budget.token_budget);
    let required = required_candidates(candidates, &budget.needs);
    let required_tokens: u64 = required
        .iter()
        .map(|candidate| u64::from(candidate.estimate_tokens))
        .sum();
    if required_tokens > token_budget {
        return Err(InformationError::Oversized {
            required_tokens,
            token_budget: budget.token_budget,
        });
    }

    let mut evidence = required;
    let mut used = required_tokens;

    // Phase 2: greedy information gain. Candidates that do not fit now can
    // never fit later (the budget only shrinks), so they are skipped for
    // the rest of this selection; gains are recomputed every round, which
    // is what makes redundant evidence worth less and less.
    let mut skipped: Vec<String> = Vec::new();
    loop {
        let mut best: Option<(&ContextCandidate, f64)> = None;
        for candidate in candidates {
            if candidate.estimate_tokens == 0
                || already_selected(&evidence, candidate)
                || skipped.iter().any(|id| id == &candidate.id)
                || group_conflict(&evidence, candidate)
            {
                continue;
            }
            let gain = gain_of(
                candidate,
                marginal_gain(candidate, &evidence, &budget.needs),
            );
            if gain <= 0.0 {
                continue;
            }
            let candidate_score = gain / f64::from(candidate.estimate_tokens);
            let better = match best {
                None => true,
                Some((chosen, best_score)) => match candidate_score.total_cmp(&best_score) {
                    std::cmp::Ordering::Greater => true,
                    std::cmp::Ordering::Equal => candidate.id < chosen.id,
                    std::cmp::Ordering::Less => false,
                },
            };
            if better {
                best = Some((candidate, candidate_score));
            }
        }
        let Some((candidate, _)) = best else {
            break;
        };
        let tokens = u64::from(candidate.estimate_tokens);
        if used.saturating_add(tokens) > token_budget {
            // Highest-score candidate does not fit: remember it and try
            // the next-ranked one; a later (smaller) candidate may fit.
            skipped.push(candidate.id.clone());
            continue;
        }
        used = used.saturating_add(tokens);
        evidence.push(candidate.clone());
    }

    // Phase 3: the conversation absorbs the residue as a contiguous
    // newest-first prefix (input order is the loader's newest-first
    // contract). A message that does not fit stops the prefix.
    let mut messages: Vec<ContextCandidate> = Vec::new();
    for candidate in candidates
        .iter()
        .filter(|candidate| candidate.kind == CandidateKind::Message)
    {
        let tokens = u64::from(candidate.estimate_tokens);
        if tokens == 0 {
            continue;
        }
        if used.saturating_add(tokens) > token_budget {
            break;
        }
        used = used.saturating_add(tokens);
        messages.push(candidate.clone());
    }

    let mut selected = messages;
    selected.extend(evidence);
    Ok(InformationSelection {
        selected,
        selected_tokens: u32::try_from(used).unwrap_or(u32::MAX),
        required_tokens: u32::try_from(required_tokens).unwrap_or(u32::MAX),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selection::{EvidenceLevel, NeedCoverage};

    fn need(id: &str, weight: f64, required: bool) -> Need {
        Need {
            id: id.into(),
            weight,
            required,
        }
    }

    fn cand(id: &str, tokens: u32, coverage: &[(&str, u32)]) -> ContextCandidate {
        ContextCandidate {
            id: id.into(),
            kind: CandidateKind::FileNote,
            bytes: (tokens as usize).saturating_mul(3),
            estimate_tokens: tokens,
            utility: 0.0,
            confidence_ppm: 1_000_000,
            freshness_ppm: 1_000_000,
            need_coverage: coverage
                .iter()
                .map(|(need_id, coverage_ppm)| NeedCoverage {
                    need_id: (*need_id).into(),
                    coverage_ppm: *coverage_ppm,
                })
                .collect(),
            ..ContextCandidate::default()
        }
    }

    fn msg(id: &str, tokens: u32) -> ContextCandidate {
        ContextCandidate {
            id: id.into(),
            kind: CandidateKind::Message,
            bytes: (tokens as usize).saturating_mul(3),
            estimate_tokens: tokens,
            utility: 1.0,
            ..ContextCandidate::default()
        }
    }

    fn budget(token_budget: u32, needs: Vec<Need>) -> InformationBudget {
        InformationBudget {
            token_budget,
            needs,
        }
    }

    fn ids(selection: &InformationSelection) -> Vec<String> {
        selection
            .selected
            .iter()
            .map(|candidate| candidate.id.clone())
            .collect()
    }

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "{a} != {b}");
    }

    /// (a) A second copy of the same evidence adds strictly less than the
    /// first, and a fully covered need makes duplicates worth exactly 0:
    /// gain is marginal, not raw.
    #[test]
    fn redundant_evidence_has_diminishing_gain() {
        let needs = vec![need("n1", 1.0, false)];
        let half = cand("half", 10, &[("n1", 500_000)]);
        let first = marginal_gain(&half, &[], &needs);
        approx(first, 0.5);
        let second = marginal_gain(&half, std::slice::from_ref(&half), &needs);
        approx(second, 0.25);
        assert!(second < first, "redundant gain must diminish");
        let full = cand("full", 10, &[("n1", 1_000_000)]);
        assert_eq!(
            marginal_gain(&full, std::slice::from_ref(&full), &needs),
            0.0,
            "fully covered need makes a duplicate worthless"
        );
        // Selection-level: the duplicate never buys a slot once its need is
        // saturated. Both fit the budget, only one is worth including (the
        // score tie breaks on id, deterministically).
        let full_candidate = cand("a-full", 10, &[("n1", 1_000_000)]);
        let duplicate = cand("b-dup", 10, &[("n1", 1_000_000)]);
        let selection =
            select_by_information(&[full_candidate, duplicate], &budget(100, needs)).unwrap();
        assert_eq!(ids(&selection), vec!["a-full"]);
    }

    /// (b) An orthogonal candidate with tiny raw utility beats duplicates
    /// carrying high raw utility: the information formula ranks what a
    /// candidate still covers, not its standalone score.
    #[test]
    fn orthogonal_evidence_beats_duplicate_high_raw_utility() {
        let needs = vec![need("n1", 1.0, false), need("n2", 1.0, false)];
        let mut dup_a = cand("dup-a", 8, &[("n1", 400_000)]);
        dup_a.utility = 1.0;
        let mut dup_b = cand("dup-b", 8, &[("n1", 400_000)]);
        dup_b.utility = 1.0;
        let mut orthogonal = cand("orth", 10, &[("n2", 1_000_000)]);
        orthogonal.utility = 0.01;
        let selection =
            select_by_information(&[dup_a, dup_b, orthogonal], &budget(10, needs)).unwrap();
        assert_eq!(ids(&selection), vec!["orth"]);
        // The raw utility of the duplicates never resurrects them: with
        // their shared need covered by a companion, they are gain-free.
        let companion = cand("dup-a", 4, &[("n1", 400_000)]);
        let dup_b = cand("dup-b", 4, &[("n1", 400_000)]);
        let selection = select_by_information(
            &[companion, dup_b],
            &budget(6, vec![need("n1", 1.0, false), need("n2", 1.0, false)]),
        )
        .unwrap();
        assert!(ids(&selection).contains(&"dup-a".to_string()));
        assert!(
            !ids(&selection).contains(&"dup-b".to_string()),
            "once n1 is saturated the duplicate has gain 0"
        );
    }

    /// (c) A required need is covered even when its candidate ranks far
    /// below an optional high-gain one and would lose every greedy round.
    #[test]
    fn required_need_is_never_dropped() {
        let needs = vec![need("must", 0.01, true), need("nice", 100.0, false)];
        let mut required = cand("required", 50, &[("must", 1_000_000)]);
        required.requirement = CandidateRequirement::Required;
        required.confidence_ppm = 0; // required protection is not gain-based
        let mut optional = cand("optional", 50, &[("nice", 1_000_000)]);
        optional.requirement = CandidateRequirement::Optional;
        let selection =
            select_by_information(&[optional, required.clone()], &budget(60, needs.clone()))
                .unwrap();
        assert_eq!(ids(&selection), vec!["required"]);
        assert_eq!(selection.required_tokens, 50);
        assert_eq!(remaining_coverage(&needs[0], &selection.selected), 0.0);
        // With room for both, the optional filler rides along after the
        // required content.
        let selection =
            select_by_information(&[required], &budget(60, vec![need("must", 0.01, true)]))
                .unwrap();
        assert_eq!(selection.selected_tokens, 50);
    }

    /// (d) Required content alone over budget is a typed Oversized error:
    /// never a silent truncation, never a partial required set.
    #[test]
    fn required_alone_overflow_is_typed_oversized() {
        let needs = vec![need("must", 1.0, true)];
        let mut required = cand("req", 100, &[("must", 1_000_000)]);
        required.requirement = CandidateRequirement::Required;
        let err = select_by_information(&[required.clone()], &budget(10, needs.clone()))
            .expect_err("required content must not be silently dropped");
        assert_eq!(
            err,
            InformationError::Oversized {
                required_tokens: 100,
                token_budget: 10,
            }
        );
        // Two partial required candidates are both mandatory (together
        // they cover the need fully) and still overflow.
        let mut a = cand("a", 60, &[("must", 500_000)]);
        a.requirement = CandidateRequirement::Required;
        let mut b = cand("b", 60, &[("must", 500_000)]);
        b.requirement = CandidateRequirement::Required;
        let err = select_by_information(&[a, b], &budget(100, needs)).expect_err("120 > 100");
        match err {
            InformationError::Oversized {
                required_tokens,
                token_budget,
            } => {
                assert_eq!((required_tokens, token_budget), (120, 100));
            }
        }
    }

    /// (e) A tiny candidate with high information beats a large low-gain
    /// log even when the log was listed first.
    #[test]
    fn small_high_information_evidence_beats_large_log() {
        let needs = vec![need("n1", 1.0, false)];
        let large_log = cand("log", 500, &[("n1", 1_000_000)]);
        let small = cand("tiny", 5, &[("n1", 1_000_000)]);
        let selection = select_by_information(&[large_log, small], &budget(10, needs)).unwrap();
        assert_eq!(ids(&selection), vec!["tiny"]);
        // The large log is not selected even though it would cover the
        // same need: 25x the tokens for the same information.
        assert_eq!(selection.selected_tokens, 5);
    }

    /// (f) Determinism: 100 identical runs are bit-identical, and an
    /// input-order permutation of the same candidate set selects the same
    /// candidates in the same order (ties break on id, never on position).
    #[test]
    fn selection_is_deterministic_100_runs() {
        let needs = vec![
            need("n1", 2.0, false),
            need("n2", 1.0, false),
            need("n3", 0.5, true),
        ];
        let mut candidates = Vec::new();
        for i in 0..12u32 {
            let mut c = cand(
                &format!("c-{i:02}"),
                (i % 4) + 1,
                &[
                    ("n1", ((i * 37) % 1_000_000).max(1)),
                    ("n2", ((i * 91) % 1_000_000).max(1)),
                    ("n3", if i == 7 { 1_000_000 } else { 0 }),
                ],
            );
            c.confidence_ppm = 500_000 + (i * 11_111).min(500_000);
            c.freshness_ppm = 1_000_000 - (i * 7_777).min(1_000_000);
            if i == 7 {
                c.requirement = CandidateRequirement::Required;
            }
            candidates.push(c);
        }
        let request = budget(30, needs);
        let first = select_by_information(&candidates, &request).unwrap();
        for _ in 1..100 {
            assert_eq!(first, select_by_information(&candidates, &request).unwrap());
        }
        let mut reversed = candidates.clone();
        reversed.reverse();
        let permuted = select_by_information(&reversed, &request).unwrap();
        assert_eq!(ids(&first), ids(&permuted));
        assert_eq!(first.selected_tokens, permuted.selected_tokens);
    }

    /// (g) Hostile weights and ppm values never inflate or poison a
    /// score: NaN/infinite/negative weights are 0, huge ppm saturates at
    /// 1.0, and an overflowing sum clamps to 0.
    #[test]
    fn nan_negative_and_huge_ppm_inputs_clamp() {
        let c = {
            let mut c = cand("c", 10, &[("n", u32::MAX)]);
            c.confidence_ppm = u32::MAX;
            c.freshness_ppm = u32::MAX;
            c
        };
        let mut nan_weight = need("n", 1.0, false);
        nan_weight.weight = f64::NAN;
        assert_eq!(marginal_gain(&c, &[], &[nan_weight]), 0.0);
        let mut negative_weight = need("n", 1.0, false);
        negative_weight.weight = -3.0;
        assert_eq!(marginal_gain(&c, &[], &[negative_weight]), 0.0);
        let mut infinite_weight = need("n", 1.0, false);
        infinite_weight.weight = f64::INFINITY;
        assert_eq!(marginal_gain(&c, &[], &[infinite_weight]), 0.0);
        // Huge ppm + weight 1.0 = exactly full coverage.
        approx(marginal_gain(&c, &[], &[need("n", 1.0, false)]), 1.0);
        // Overflowing the f64 sum is clamped to 0 (fail-safe, not inf).
        let huge_a = need("a", f64::MAX, false);
        let huge_b = need("b", f64::MAX, false);
        let both = cand("both", 10, &[("a", 1_000_000), ("b", 1_000_000)]);
        assert_eq!(marginal_gain(&both, &[], &[huge_a, huge_b]), 0.0);
        // Zero-token candidates are never priced and never selected.
        let zero = cand("zero", 0, &[("n", 1_000_000)]);
        let selection =
            select_by_information(&[zero], &budget(100, vec![need("n", 1.0, false)])).unwrap();
        assert!(selection.selected.is_empty());
        // NaN utility on a candidate is irrelevant to the information
        // formula and cannot smuggle it in.
        let mut nan_utility = cand("nan", 5, &[("n", 1_000_000)]);
        nan_utility.utility = f64::NAN;
        let selection = select_by_information(
            &[nan_utility.clone()],
            &budget(100, vec![need("n", 1.0, false)]),
        )
        .unwrap();
        assert_eq!(ids(&selection), vec!["nan"]);
    }

    /// (h) One S/M/L level per evidence group: once a variant of a group
    /// is selected, every sibling variant is suppressed — even when its
    /// marginal gain would still be positive for another need.
    #[test]
    fn sml_group_exclusivity() {
        let needs = vec![
            need("n1", 1.0, false),
            need("n2", 1.0, false),
            need("n3", 1.0, false),
        ];
        // Group 1: the Exact variant ranks higher per token (1/5 vs
        // 1.5/100) and wins; the Summary sibling still has positive gain
        // for n2 but is suppressed by the group rule.
        let mut g1_summary = cand("g1-summary", 100, &[("n1", 1_000_000), ("n2", 500_000)]);
        g1_summary.evidence = Some(1);
        g1_summary.level = EvidenceLevel::Summary;
        let mut g1_exact = cand("g1-exact", 5, &[("n1", 1_000_000)]);
        g1_exact.evidence = Some(1);
        g1_exact.level = EvidenceLevel::Exact;
        // Group 2: the Summary variant ranks higher per token and wins;
        // the Exact sibling is suppressed.
        let mut g2_summary = cand("g2-summary", 5, &[("n3", 1_000_000)]);
        g2_summary.evidence = Some(2);
        g2_summary.level = EvidenceLevel::Summary;
        let mut g2_exact = cand("g2-exact", 100, &[("n3", 1_000_000)]);
        g2_exact.evidence = Some(2);
        g2_exact.level = EvidenceLevel::Exact;
        let selection = select_by_information(
            &[g1_summary, g1_exact, g2_summary, g2_exact],
            &budget(1_000, needs),
        )
        .unwrap();
        let selected = ids(&selection);
        assert!(selected.contains(&"g1-exact".to_string()));
        assert!(
            !selected.contains(&"g1-summary".to_string()),
            "group 1 is already represented by its Exact variant: {selected:?}"
        );
        assert!(selected.contains(&"g2-summary".to_string()));
        assert!(
            !selected.contains(&"g2-exact".to_string()),
            "group 2 is already represented by its Summary variant: {selected:?}"
        );
        assert_eq!(selected.len(), 2, "exactly one level per group");
    }

    /// Ungrouped candidates never suppress each other, and required
    /// coverage saturates so selecting an equivalent required candidate
    /// stops the mandatory phase from over-collecting.
    #[test]
    fn ungrouped_candidates_and_required_saturation() {
        let a = cand("a", 5, &[("n", 1_000_000)]);
        let b = cand("b", 5, &[("n", 1_000_000)]);
        let selection =
            select_by_information(&[a, b], &budget(100, vec![need("n", 1.0, false)])).unwrap();
        assert_eq!(ids(&selection), vec!["a"]);
        let mut required = cand("req", 5, &[("n", 1_000_000)]);
        required.requirement = CandidateRequirement::Required;
        let duplicate = cand("req2", 5, &[("n", 1_000_000)]);
        let selection = select_by_information(
            &[required, duplicate],
            &budget(100, vec![need("n", 1.0, true)]),
        )
        .unwrap();
        assert_eq!(ids(&selection), vec!["req"]);
        assert_eq!(selection.required_tokens, 5);
    }

    /// The conversation absorbs only the residue: required + gain-ranked
    /// evidence take their slots first, messages keep a contiguous
    /// newest-first prefix of what is left.
    #[test]
    fn messages_absorb_the_residue_as_a_contiguous_prefix() {
        let mut required = cand("req", 40, &[("n", 1_000_000)]);
        required.requirement = CandidateRequirement::Required;
        let mut tiny = cand("tiny", 10, &[("n", 1_000_000)]);
        tiny.need_coverage.clear(); // no gain: only the required candidate covers n
        let messages = vec![msg("m3", 20), msg("m2", 20), msg("m1", 20)];
        let mut candidates = messages;
        candidates.push(required);
        candidates.push(tiny);
        let selection =
            select_by_information(&candidates, &budget(90, vec![need("n", 1.0, true)])).unwrap();
        // Required (40) leaves 50: the first two messages fit, the third
        // does not — a contiguous prefix, never a hole.
        assert_eq!(ids(&selection), vec!["m3", "m2", "req"]);
        assert_eq!(selection.selected_tokens, 80);
    }

    /// A closure-backed prior used by the adversarial tests below.
    struct RiskPrior<F: Fn(&ContextCandidate) -> f64>(F);

    impl<F: Fn(&ContextCandidate) -> f64> FailurePrior for RiskPrior<F> {
        fn omission_risk(&self, candidate: &ContextCandidate) -> f64 {
            (self.0)(candidate)
        }
    }

    /// A hostile prior that PANICS the moment it is consulted for a Required
    /// candidate: the consumer must never call it for one.
    struct PanicOnRequired;

    impl FailurePrior for PanicOnRequired {
        fn omission_risk(&self, candidate: &ContextCandidate) -> f64 {
            assert_ne!(
                candidate.requirement,
                CandidateRequirement::Required,
                "the prior must never be consulted for Required candidates"
            );
            2.0
        }
    }

    /// The audit-68 clamp table: `base * clamp(risk, 1, 2)` through the
    /// learning crate, non-finite risk = maximum 2x, hostile risk never
    /// demotes below the base, hostile base never yields NaN/inf/negative.
    #[test]
    fn prior_adjusted_gain_clamps_hostile_risks() {
        approx(prior_adjusted_gain(0.5, 1.0), 0.5);
        approx(prior_adjusted_gain(0.5, 1.5), 0.75);
        approx(prior_adjusted_gain(0.5, 2.0), 1.0);
        approx(prior_adjusted_gain(0.5, 3.0), 1.0);
        approx(prior_adjusted_gain(0.5, f64::MAX), 1.0);
        approx(prior_adjusted_gain(0.5, -100.0), 0.5);
        approx(prior_adjusted_gain(0.5, 0.0), 0.5);
        approx(prior_adjusted_gain(0.5, f64::NAN), 1.0);
        approx(prior_adjusted_gain(0.5, f64::INFINITY), 1.0);
        // Non-finite risk — negative infinity included — is the learning
        // crate's "unknown risk" maximum, never a demotion.
        approx(prior_adjusted_gain(0.5, f64::NEG_INFINITY), 1.0);
        assert_eq!(prior_adjusted_gain(0.0, 2.0), 0.0);
        assert_eq!(prior_adjusted_gain(f64::NAN, 1.0), 0.0);
        assert_eq!(prior_adjusted_gain(-3.0, 2.0), 0.0);
        assert!(
            prior_adjusted_gain(f64::MAX, 2.0).is_finite(),
            "the product is sanitized to a finite value"
        );
    }

    /// Omission risk protects non-Required candidates (up to 2x) and is
    /// NEVER consulted for Required ones; required coverage stays intact.
    #[test]
    fn failure_prior_only_protects_non_required_candidates() {
        let needs = vec![need("must", 0.1, true), need("nice", 1.0, false)];
        let mut required = cand("required", 10, &[("must", 1_000_000)]);
        required.requirement = CandidateRequirement::Required;
        let cheaper = cand("a-cheap", 10, &[("nice", 1_000_000)]);
        let risked = cand("b-risked", 10, &[("nice", 1_000_000)]);
        let pool = [required.clone(), cheaper.clone(), risked.clone()];

        // Baseline: the tie breaks on id ascending, a-cheap wins.
        let base = select_by_information(&pool, &budget(20, needs.clone())).unwrap();
        assert!(ids(&base).contains(&"a-cheap".to_string()));
        assert!(!ids(&base).contains(&"b-risked".to_string()));

        // With a 2x prior on b-risked, the protected candidate wins the
        // residue; the required set is byte-identical.
        let prior = RiskPrior(
            |c: &ContextCandidate| {
                if c.id == "b-risked" {
                    2.0
                } else {
                    1.0
                }
            },
        );
        let boosted =
            select_by_information_with_prior(&pool, &budget(20, needs.clone()), &prior).unwrap();
        assert!(ids(&boosted).contains(&"b-risked".to_string()));
        assert!(!ids(&boosted).contains(&"a-cheap".to_string()));
        assert_eq!(boosted.required_tokens, base.required_tokens);
        assert_eq!(
            remaining_coverage(&needs[0], &boosted.selected),
            0.0,
            "required coverage is untouched"
        );
        // Determinism under the prior: 100 identical runs.
        for _ in 0..100 {
            assert_eq!(
                select_by_information_with_prior(&pool, &budget(20, needs.clone()), &prior)
                    .unwrap(),
                boosted
            );
        }

        // A neutral prior is byte-identical to the baseline selector.
        let neutral = RiskPrior(|_: &ContextCandidate| 1.0);
        assert_eq!(
            select_by_information_with_prior(&pool, &budget(20, needs.clone()), &neutral).unwrap(),
            base,
            "risk 1.0 is the identity: parity when the prior is off"
        );

        // Required candidates are untouchable: a prior that panics if asked
        // about one completes normally and cannot change the required set.
        let guarded =
            select_by_information_with_prior(&pool, &budget(20, needs.clone()), &PanicOnRequired)
                .unwrap();
        assert_eq!(guarded.required_tokens, base.required_tokens);
        assert!(ids(&guarded).contains(&"required".to_string()));
    }

    /// Hostile risks (NaN, negatives, huge, infinite) on the greedy phase
    /// clamp to [1, 2]: selection stays deterministic, budget-respecting,
    /// and never poisons a gain with NaN/inf.
    #[test]
    fn hostile_failure_prior_risks_are_clamped_in_selection() {
        let needs = vec![need("n", 1.0, false)];
        let candidates = [
            cand("a", 10, &[("n", 1_000_000)]),
            cand("b", 10, &[("n", 1_000_000)]),
        ];
        let budget = budget(10, needs);
        for risk in [
            f64::NAN,
            f64::NEG_INFINITY,
            -1e300,
            0.0,
            f64::INFINITY,
            f64::MAX,
        ] {
            let prior = RiskPrior(move |_: &ContextCandidate| risk);
            let selection = select_by_information_with_prior(&candidates, &budget, &prior).unwrap();
            let again = select_by_information_with_prior(&candidates, &budget, &prior).unwrap();
            assert_eq!(selection, again, "deterministic under risk {risk}");
            assert_eq!(selection.selected.len(), 1, "one 10-token candidate fits");
            assert!(selection.selected_tokens <= 10);
            assert!(
                selection.selected.iter().all(|c| c.estimate_tokens == 10),
                "no fabricated candidate under risk {risk}"
            );
        }
        // A negative risk never demotes: b still loses the tie to a.
        let demoting = RiskPrior(|c: &ContextCandidate| if c.id == "b" { -5.0 } else { 1.0 });
        let selection = select_by_information_with_prior(&candidates, &budget, &demoting).unwrap();
        assert_eq!(ids(&selection), vec!["a"]);
        // A huge risk only ever doubles: b's 2x still LOSES no tie-break
        // against a's 2x (both equal, id ascending wins).
        let boosting = RiskPrior(|_: &ContextCandidate| f64::MAX);
        let selection = select_by_information_with_prior(&candidates, &budget, &boosting).unwrap();
        assert_eq!(ids(&selection), vec!["a"]);
    }
}
