//! Utility-per-token context planning (P0-27): the REAL turn path chooses
//! its context window here — by [`crate::selection::select_by_utility`]
//! over the volatile classes — instead of any competing inline selector at
//! the wire/render layer. `wire_plan` only RENDERS what this planner
//! selected; there is exactly one selector on the runtime path.
//!
//! Memory classes (spec §8.4, wave-14):
//!
//! ```text
//! Static     immutable instructions (never competes)
//! SemiStable durable task state + repository knowledge + steering (never
//!            competes; the caller reserved their tokens and rendered them
//!            into the byte-stable cacheable head)
//! Volatile   recent conversation (Message) + retrieved evidence
//!            (Symbol/FileNote/ToolNote/SubagentSummary) — the ONLY classes
//!            that compete, through [`select_by_utility`], with
//!            `budget = token_budget − static_tokens − semi_stable_tokens`
//! ```
//!
//! Conversation Messages are never dropped as a whole until budget
//! exhaustion (the selector's phase-1 cap + refill, plus its last-resort
//! guarantee when a single oversized message would displace the whole
//! conversation). Rules (RepoRule, semi-stable repository knowledge) ride
//! the cacheable head; the planner keeps them verbatim inside the reserved
//! semi-stable budget and trims only the TAIL of the rules list when the
//! caller's reserve cannot hold them (deterministic, reported).
//!
//! Determinism: identical input yields a bit-identical plan. Hostile
//! utilities (NaN/inf) are never evidence of value: they are rejected to 0
//! and can never be selected.

use crate::information::{
    prior_adjusted_gain, required_candidates, select_by_information,
    select_by_information_with_prior, FailurePrior, InformationBudget, InformationError,
};
use crate::selection::{select_by_utility, CandidateKind, CandidateRequirement, ContextCandidate};

/// The §8.4 memory class of one planned candidate, in render order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryOrdering {
    /// Immutable instructions. Never a candidate here (the caller renders
    /// them from configuration, not from selection).
    Static,
    /// Durable task state, repository knowledge and the steering note:
    /// the byte-stable cacheable head. Rule candidates map here.
    SemiStable,
    /// Recent conversation and retrieved evidence: the volatile tail.
    Volatile,
}

impl MemoryOrdering {
    /// The class of a candidate kind under §8.4. Message and every evidence
    /// kind the selector competes are volatile; RepoRule is semi-stable
    /// repository knowledge (the caller reserved its tokens).
    pub fn of(kind: CandidateKind) -> MemoryOrdering {
        match kind {
            CandidateKind::RepoRule => MemoryOrdering::SemiStable,
            CandidateKind::Message
            | CandidateKind::Symbol
            | CandidateKind::FileNote
            | CandidateKind::ToolNote
            | CandidateKind::SubagentSummary => MemoryOrdering::Volatile,
        }
    }

    /// True for candidates that COMPETE in the volatile budget. Rules and
    /// messages excluded: rules are pre-reserved semi-stable content;
    /// messages compete only through their own never-droppable phases.
    pub fn competes(kind: CandidateKind) -> bool {
        matches!(
            kind,
            CandidateKind::Symbol
                | CandidateKind::FileNote
                | CandidateKind::ToolNote
                | CandidateKind::SubagentSummary
        )
    }
}

/// How the caller classifies the head it renders around the selection
/// (informational for the §8.4 ordering; the derived volatile budget is
/// `token_budget − static_tokens − semi_stable_tokens`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PlannerMode {
    /// Token estimate of the StaticPrefix the caller renders first
    /// (instructions; immutable per session).
    pub static_tokens: u32,
    /// Token estimate of the SemiStable head the caller renders after it
    /// (rules + task ledger + repo map + steering note) — the reserve the
    /// volatile selection may never consume.
    pub semi_stable_tokens: u32,
}

/// One planner invocation. `messages` arrive newest-first (the durable
/// bounded-loader contract) and `token_budget` is the TOTAL turn budget:
/// the planner derives the volatile budget by reserving the caller's
/// static + semi-stable estimates, so the conversation can never consume
/// the byte-stable head.
#[derive(Debug, Clone)]
pub struct ContextPlanRequest {
    /// Durable conversation rows as candidates (utility 1.0, recency order
    /// decided by the selector's message phases, never by utility).
    pub messages: Vec<ContextCandidate>,
    /// Repository rules (RepoRule kind, semi-stable). Never competes:
    /// selected in input order inside the semi-stable reserve.
    pub rules: Vec<ContextCandidate>,
    /// Retrieved repository evidence (Symbol/FileNote kinds).
    pub index_evidence: Vec<ContextCandidate>,
    /// Current tool notes (volatile evidence).
    pub tool_notes: Vec<ContextCandidate>,
    /// Compacted-turn summaries of child sessions (volatile evidence).
    pub subagent_summaries: Vec<ContextCandidate>,
    /// Total tokens available to the turn (context_max of the budget).
    pub token_budget: u32,
    /// Evidence below this utility never competes (selector's `min_utility`;
    /// 0.0 admits every positive-utility candidate).
    pub min_utility: f64,
    /// Static/semi-stable token classification of the caller's head.
    pub mode: PlannerMode,
}

impl Default for ContextPlanRequest {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            rules: Vec::new(),
            index_evidence: Vec::new(),
            tool_notes: Vec::new(),
            subagent_summaries: Vec::new(),
            token_budget: 0,
            min_utility: 0.0,
            mode: PlannerMode::default(),
        }
    }
}

/// The planner's decision: exactly what the wire layer may render.
#[derive(Debug, Clone, PartialEq)]
pub struct ContextPlan {
    /// Selected candidates. Ordering within the volatile classes is the
    /// selector's (message phase + utility-per-token phase + refill);
    /// the renderer maps classes back to their sections and never
    /// re-selects.
    pub selected: Vec<ContextCandidate>,
    /// The §8.4 memory class of every entry of [`ContextPlan::selected`]
    /// (parallel, so the renderer can split static/semi-stable/volatile
    /// without ever guessing from a kind name).
    pub ordering: Vec<MemoryOrdering>,
    /// The volatile budget the selection competed for
    /// (`token_budget − static − semi-stable`, saturated at 0).
    pub volatile_budget: u32,
    /// Tokens the selection consumed (`Σ estimate_tokens` of selected).
    pub selected_tokens: u32,
    /// Rules the caller's semi-stable reserve could not hold; only the
    /// TAIL of the rules list is ever dropped (deterministic, never a
    /// hole). The caller renders the head from `selected` only.
    pub rules_dropped: usize,
}

/// Build the turn's context plan (P0-27). Pure and deterministic: the same
/// request always yields the same plan (identical selected ids in
/// identical order, bit-identical counts). Runtime is bounded — a hostile
/// 20k-candidate request plans in far under 50 ms.
///
/// Non-finite candidate utilities (NaN/inf) are rejected to 0 BEFORE
/// selection and can never be selected; huge finite utilities stay finite
/// math (selection never divides by zero: a zero-token candidate has no
/// price and is never selected). A zero volatile budget yields the empty
/// volatile window: with no tokens even the conversation is excluded.
pub fn plan_context(request: ContextPlanRequest) -> ContextPlan {
    plan_context_with_prior(request, None)
}

/// [`plan_context`] with an optional failure-aware omission prior (audit 68):
/// for every non-Required volatile candidate the prior's omission risk is
/// folded into the value the selector ranks by — `base * clamp(risk, 1, 2)`
/// through [`prior_adjusted_gain`], so a prior can only ever protect a
/// candidate up to 2x. Required candidates are never consulted (untouchable)
/// and the rules head is never touched. `None` is bit-identical to
/// [`plan_context`].
pub fn plan_context_with_prior(
    request: ContextPlanRequest,
    prior: Option<&dyn FailurePrior>,
) -> ContextPlan {
    plan_context_inner(request, None, prior).0
}

/// Information-budget-aware planning (audit 41/42): when the budget
/// declares needs, the volatile pool (messages + evidence) is selected by
/// [`select_by_information`] — required coverage first, then marginal
/// information gain per token, then the conversation as the residue. The
/// effective volatile budget is the smaller of the request-derived
/// volatile budget (`token_budget − static − semi-stable`) and
/// [`InformationBudget::token_budget`]: an information budget can tighten
/// the plan, never silently widen the caller's total turn budget.
///
/// With [`InformationBudget::needs`] empty the call is the baseline
/// [`plan_context`] path (utility per token) and cannot fail. When
/// REQUIRED content alone exceeds the effective budget the call fails with
/// the typed [`InformationError::Oversized`] — the plan is never returned
/// with required content silently dropped or truncated.
pub fn plan_context_with_information(
    request: ContextPlanRequest,
    information: InformationBudget,
) -> Result<ContextPlan, InformationError> {
    plan_context_with_information_and_prior(request, information, None)
}

/// [`plan_context_with_information`] with an optional failure-aware omission
/// prior (audit 68). On the information path each non-Required candidate's
/// marginal gain is adjusted by `base * clamp(risk, 1, 2)` before ranking
/// ([`select_by_information_with_prior`]); required needs are covered by
/// [`required_candidates`] exactly as without a prior, and the prior is never
/// consulted for a [`CandidateRequirement::Required`] candidate. When the
/// information needs are empty this degrades to the utility path of
/// [`plan_context_with_prior`]. `None` is bit-identical to
/// [`plan_context_with_information`].
pub fn plan_context_with_information_and_prior(
    request: ContextPlanRequest,
    information: InformationBudget,
    prior: Option<&dyn FailurePrior>,
) -> Result<ContextPlan, InformationError> {
    let (plan, error) = plan_context_inner(request, Some(information), prior);
    match error {
        Some(error) => Err(error),
        None => Ok(plan),
    }
}

/// Fold the omission prior into every non-Required candidate's utility
/// (`base * clamp(risk, 1, 2)`), leaving Required candidates untouched and
/// unconsulted. A hostile risk or base is sanitized by
/// [`prior_adjusted_gain`] to a finite, non-negative value, so NaN/inf can
/// never enter the selector's ordering.
fn apply_prior(pool: &mut [ContextCandidate], prior: &dyn FailurePrior) {
    for candidate in pool.iter_mut() {
        if candidate.requirement == CandidateRequirement::Required {
            continue;
        }
        candidate.utility = prior_adjusted_gain(candidate.utility, prior.omission_risk(candidate));
    }
}

fn plan_context_inner(
    request: ContextPlanRequest,
    information: Option<InformationBudget>,
    prior: Option<&dyn FailurePrior>,
) -> (ContextPlan, Option<InformationError>) {
    let volatile_budget = request
        .token_budget
        .saturating_sub(request.mode.static_tokens)
        .saturating_sub(request.mode.semi_stable_tokens);

    // Hostile utilities are not evidence: NaN/inf never select.
    let sanitize = |c: &mut ContextCandidate| {
        if !c.utility.is_finite() {
            c.utility = 0.0;
        }
    };
    let messages: Vec<ContextCandidate> = request
        .messages
        .iter()
        .cloned()
        .map(|mut c| {
            sanitize(&mut c);
            c
        })
        .collect();
    let mut rules = request.rules.clone();
    for c in &mut rules {
        sanitize(c);
    }
    let mut evidence: Vec<ContextCandidate> = request
        .index_evidence
        .iter()
        .chain(request.tool_notes.iter())
        .chain(request.subagent_summaries.iter())
        .cloned()
        .map(|mut c| {
            sanitize(&mut c);
            c
        })
        .collect();

    // Rules ride the semi-stable reserve, never the volatile budget. When
    // the caller's reserve cannot hold them, drop from the TAIL of the
    // rules list (deterministic; the first rules are the ones the caller
    // ordered first).
    let reserve = u64::from(request.mode.semi_stable_tokens);
    let mut rules_kept = rules.len();
    let mut rules_tokens: u64 = rules.iter().map(|r| u64::from(r.estimate_tokens)).sum();
    while rules_tokens > reserve && rules_kept > 0 {
        rules_kept -= 1;
        rules_tokens = rules_tokens.saturating_sub(u64::from(rules[rules_kept].estimate_tokens));
    }
    rules.truncate(rules_kept);
    let rules_dropped = request.rules.len() - rules_kept;

    // One pool, one selector: volatile classes compete together. Rules are
    // pre-reserved semi-stable head content — they never compete for the
    // volatile budget, they ride the plan ahead of the volatile window.
    // With needs, the pool competes by information gain instead of raw
    // utility (audit 41/42); without needs the baseline selector is kept
    // bit-identical.
    let mut pool = messages;
    pool.append(&mut evidence);
    let (volatile_selected, information_error) = match information {
        Some(info) if !info.needs.is_empty() => {
            let effective = InformationBudget {
                token_budget: volatile_budget.min(info.token_budget),
                needs: info.needs,
            };
            let selected = match prior {
                Some(prior) => select_by_information_with_prior(&pool, &effective, prior),
                None => select_by_information(&pool, &effective),
            };
            match selected {
                Ok(selection) => (selection.selected, None),
                Err(error) => {
                    // Never silently drop required content: surface the
                    // mandatory set in the internal plan and let the
                    // checked planner API return the typed error.
                    (required_candidates(&pool, &effective.needs), Some(error))
                }
            }
        }
        _ => {
            if let Some(prior) = prior {
                apply_prior(&mut pool, prior);
            }
            (
                select_by_utility(&pool, volatile_budget, request.min_utility),
                None,
            )
        }
    };
    let mut selected = rules;
    let mut ordering: Vec<MemoryOrdering> = vec![MemoryOrdering::SemiStable; selected.len()];
    selected.extend(volatile_selected);
    for c in &selected[rules_kept..] {
        ordering.push(MemoryOrdering::of(c.kind));
    }
    let selected_tokens: u32 = selected
        .iter()
        .map(|c| c.estimate_tokens)
        .fold(0u32, u32::saturating_add);
    (
        ContextPlan {
            selected,
            ordering,
            volatile_budget,
            selected_tokens,
            rules_dropped,
        },
        information_error,
    )
}

/// Convenience for callers that keep candidate content elsewhere: the ids
/// the plan selected (in plan order) for `kind`.
pub fn selected_ids(plan: &ContextPlan) -> Vec<String> {
    plan.selected.iter().map(|c| c.id.clone()).collect()
}

#[cfg(test)]
mod tests {
    // Test fixtures build requests field-by-field from Default for
    // readability; the planner itself has no such style.
    #![allow(clippy::field_reassign_with_default)]
    use super::*;

    fn msg(id: u32, tokens: u32) -> ContextCandidate {
        ContextCandidate {
            id: format!("m{id:05}"),
            kind: CandidateKind::Message,
            bytes: (tokens as usize).saturating_mul(3),
            estimate_tokens: tokens,
            utility: 1.0,
            ..ContextCandidate::default()
        }
    }

    fn evidence(id: &str, kind: CandidateKind, tokens: u32, utility: f64) -> ContextCandidate {
        ContextCandidate {
            id: id.into(),
            kind,
            bytes: (tokens as usize).saturating_mul(3),
            estimate_tokens: tokens,
            utility,
            ..ContextCandidate::default()
        }
    }

    fn msgs_newest_first(n: u32, tokens: u32) -> Vec<ContextCandidate> {
        (1..=n).rev().map(|i| msg(i, tokens)).collect()
    }

    fn run(req: ContextPlanRequest) -> ContextPlan {
        plan_context(req)
    }

    fn msg_count(plan: &ContextPlan) -> usize {
        plan.selected
            .iter()
            .filter(|c| c.kind == CandidateKind::Message)
            .count()
    }

    fn msg_ids(plan: &ContextPlan) -> Vec<String> {
        plan.selected
            .iter()
            .filter(|c| c.kind == CandidateKind::Message)
            .map(|c| c.id.clone())
            .collect()
    }

    /// The planner-only adversarial (a): 50 evidence items of 10 tokens
    /// each plus a high-utility symbol against a 100-token volatile budget
    /// with 50 ten-token messages. The OLD behavior (the wire layer's
    /// wholesale `include_evidence = false` fallback when the budget ran
    /// tight) dropped the ENTIRE evidence section and kept the newest
    /// messages. The planner instead trades MESSAGE SLOTS for evidence: the
    /// symbol is selected and the message window SHRANK below the
    /// no-evidence counterfactual — evidence survives by utility per token.
    #[test]
    fn symbol_survives_and_message_slots_shrink_instead_of_dropping_all_evidence() {
        let mut req = ContextPlanRequest::default();
        req.messages = msgs_newest_first(50, 10);
        req.mode.static_tokens = 0;
        req.mode.semi_stable_tokens = 0;
        req.token_budget = 100;
        // 50 dull evidence items (10 tokens each, low utility) + one symbol.
        for i in 0..50u32 {
            req.index_evidence.push(evidence(
                &format!("note-{i:03}"),
                CandidateKind::FileNote,
                10,
                0.05,
            ));
        }
        req.index_evidence
            .push(evidence("sym::parser", CandidateKind::Symbol, 5, 0.95));
        let plan = run(req.clone());
        // Evidence is NOT dropped wholesale: the symbol survives.
        assert!(
            plan.selected.iter().any(|c| c.id == "sym::parser"),
            "the symbol must be selected (no wholesale evidence drop)"
        );
        let evidence_selected: Vec<&ContextCandidate> = plan
            .selected
            .iter()
            .filter(|c| c.kind != CandidateKind::Message)
            .collect();
        assert!(
            !evidence_selected.is_empty(),
            "selected evidence must be non-empty while the message window shrank"
        );
        // The message window shrank: fewer messages than the 10 the same
        // budget holds without evidence (the selector's reserve + the
        // symbol's 5 tokens cost message slots).
        let without_evidence = {
            let mut no_ev = req.clone();
            no_ev.index_evidence.clear();
            run(no_ev)
        };
        assert_eq!(msg_count(&without_evidence), 10, "messages alone fill 100");
        let window = msg_count(&plan);
        assert!(window < 10, "message window must shrink: {window} kept");
        // Kept messages are the NEWEST window (contiguous, no holes) and
        // the budget is respected.
        let ids = msg_ids(&plan);
        assert_eq!(ids.len(), window);
        assert!(ids.windows(2).all(|w| w[0] > w[1]), "newest first: {ids:?}");
        let total: u64 = plan
            .selected
            .iter()
            .map(|c| u64::from(c.estimate_tokens))
            .sum();
        assert!(total <= 100, "budget respected: {total}");
        assert_eq!(u64::from(plan.selected_tokens), total);
    }

    /// Planner determinism (b): 100 runs of an identical request produce
    /// bit-identical plans (ids, order, counts).
    #[test]
    fn identical_inputs_produce_identical_plans_100_runs() {
        let mut req = ContextPlanRequest::default();
        req.messages = msgs_newest_first(30, 10);
        req.token_budget = 150;
        req.mode.static_tokens = 20;
        req.mode.semi_stable_tokens = 10;
        req.rules
            .push(evidence("rules::a", CandidateKind::RepoRule, 6, 1.0));
        for i in 0..12u32 {
            req.index_evidence.push(evidence(
                &format!("sym::{i}"),
                CandidateKind::Symbol,
                (i % 5) + 2,
                0.3 + (i as f64 / 40.0),
            ));
        }
        let first = plan_context(req.clone());
        for _ in 1..100 {
            let again = plan_context(req.clone());
            assert_eq!(first, again);
        }
        // Rules are semi-stable: present, ordered first-class, never
        // competing for the volatile budget.
        assert!(first
            .selected
            .iter()
            .any(|c| c.kind == CandidateKind::RepoRule && c.id == "rules::a"));
    }

    /// Reorder-flip evidence (c): the planner's evidence selection is a
    /// function of the candidate SET, not its input order — a turn-to-turn
    /// score/order flip of the same files yields the same selected ids in
    /// the same class order, so the volatile tail the wire layer renders
    /// after the cacheable boundary never moves it (the boundary itself is
    /// locked by the wire_plan byte-stability tests).
    #[test]
    fn evidence_reorder_flip_leaves_the_selection_and_boundary_intact() {
        let base = |order: [(f64, u32); 2]| {
            let mut req = ContextPlanRequest::default();
            req.messages = msgs_newest_first(4, 10);
            req.token_budget = 200;
            for (u, t) in order {
                let which = if u > 0.5 { "src/b.rs" } else { "src/a.rs" };
                req.index_evidence
                    .push(evidence(which, CandidateKind::Symbol, t, u));
            }
            plan_context(req)
        };
        // Turn 1 ranks b above a; turn 2 flips the scores AND the input
        // order. The same two files were always going to fit — what must
        // not happen is the selection changing shape class-wise.
        let p1 = base([(1.0, 10), (0.9, 10)]);
        let p2 = base([(0.9, 10), (1.0, 10)]);
        let mut ids1: Vec<&str> = p1
            .selected
            .iter()
            .filter(|c| c.kind == CandidateKind::Symbol)
            .map(|c| c.id.as_str())
            .collect();
        ids1.sort_unstable();
        let mut ids2: Vec<&str> = p2
            .selected
            .iter()
            .filter(|c| c.kind == CandidateKind::Symbol)
            .map(|c| c.id.as_str())
            .collect();
        ids2.sort_unstable();
        assert_eq!(
            ids1, ids2,
            "score flips must not change which evidence fits"
        );
        // Class order is stable across the flip: messages first (newest
        // prefix), every volatile class after — identical ordering vectors.
        assert_eq!(p1.ordering, p2.ordering);
        // And a same-set input reorder with EQUAL utilities yields an
        // IDENTICAL plan (ties break on id, never on input position).
        let same = |order: [&str; 2]| {
            let mut req = ContextPlanRequest::default();
            req.messages = msgs_newest_first(4, 10);
            req.token_budget = 200;
            for p in order {
                req.index_evidence
                    .push(evidence(p, CandidateKind::Symbol, 10, 1.0));
            }
            plan_context(req)
        };
        assert_eq!(
            same(["src/a.rs", "src/b.rs"]),
            same(["src/b.rs", "src/a.rs"])
        );
    }

    /// Hostile inputs (d): 10k candidates with huge and non-finite
    /// utilities plan in bounded time (<50 ms), never select NaN/inf
    /// utilities, respect the budget, and reproduce exactly.
    #[test]
    fn hostile_candidates_plan_in_bounded_time_and_nan_never_selects() {
        let mut req = ContextPlanRequest::default();
        req.messages = msgs_newest_first(2_000, 10);
        req.token_budget = 500;
        req.mode.static_tokens = 100;
        req.mode.semi_stable_tokens = 50;
        for i in 0..8_000u32 {
            let u = match i % 5 {
                0 => f64::NAN,
                1 => f64::INFINITY,
                2 => f64::NEG_INFINITY,
                3 => 1e300 * (i as f64),
                _ => 0.5 + (i % 100) as f64 / 200.0,
            };
            req.index_evidence.push(evidence(
                &format!("h-{i:05}"),
                CandidateKind::FileNote,
                (i % 7) + 1,
                u,
            ));
        }
        assert_eq!(req.index_evidence.len(), 8_000);
        let started = std::time::Instant::now();
        let plan = plan_context(req.clone());
        let elapsed = started.elapsed();
        assert!(
            elapsed.as_millis() < 50,
            "planning 10k candidates took {elapsed:?}"
        );
        // NaN/inf utilities were rejected to 0 and can never be selected.
        for c in &plan.selected {
            if c.kind != CandidateKind::Message {
                assert!(c.utility.is_finite() && c.utility > 0.0, "{c:?}");
            }
        }
        let total: u64 = plan
            .selected
            .iter()
            .map(|c| u64::from(c.estimate_tokens))
            .sum();
        assert!(total <= 500, "total budget respected: {total}");
        let run2 = plan_context(req);
        assert_eq!(plan, run2, "deterministic under hostile utilities");
    }

    /// 20k-message session (e): the provider window is the PLANNER'S
    /// bounded window — evidence still reserves its slice, the newest
    /// messages survive as a contiguous prefix, and planning stays <50 ms.
    #[test]
    fn twenty_thousand_message_session_gets_a_bounded_planner_window() {
        let mut req = ContextPlanRequest::default();
        req.messages = msgs_newest_first(20_000, 10);
        req.token_budget = 1_000;
        req.mode.static_tokens = 300;
        req.mode.semi_stable_tokens = 200;
        req.index_evidence
            .push(evidence("sym::hot", CandidateKind::Symbol, 5, 1.0));
        let started = std::time::Instant::now();
        let plan = plan_context(req.clone());
        let elapsed = started.elapsed();
        assert!(elapsed.as_millis() < 50, "20k messages took {elapsed:?}");
        // The window is bounded by the volatile budget (1000 - 500 = 500
        // tokens → 49 ten-token messages + the 5-token symbol).
        let window = msg_count(&plan);
        let total: u64 = plan
            .selected
            .iter()
            .map(|c| u64::from(c.estimate_tokens))
            .sum();
        assert!(total <= 1_000);
        assert!(window < 50, "window must be bounded: {window}");
        assert!(plan.selected.iter().any(|c| c.id == "sym::hot"));
        // The kept messages are the newest contiguous prefix (the provider
        // sees exactly the planner's window — nothing older sneaks in).
        let ids = msg_ids(&plan);
        assert_eq!(ids.len(), window);
        assert_eq!(ids[0], "m20000", "newest message first");
        assert!(ids.windows(2).all(|w| w[0] > w[1]));
        let last_seq: u32 = ids
            .last()
            .unwrap()
            .strip_prefix('m')
            .unwrap()
            .parse()
            .unwrap();
        assert!(last_seq >= 20_000 - 60, "newest window ends near the tail"); // Deterministic across runs.
        assert_eq!(plan, plan_context(req));
    }

    /// Rules ride the semi-stable reserve; when the reserve cannot hold
    /// them only the TAIL of the rules list drops — never a hole, never a
    /// competing message or evidence token.
    #[test]
    fn rules_never_compete_and_oversized_reserves_trim_from_the_tail() {
        let mut req = ContextPlanRequest::default();
        req.messages = msgs_newest_first(5, 10);
        req.token_budget = 100;
        req.mode.semi_stable_tokens = 40;
        for i in 0..6u32 {
            req.rules.push(evidence(
                &format!("rule-{i}"),
                CandidateKind::RepoRule,
                10,
                1.0,
            ));
        }
        let plan = plan_context(req.clone());
        // Reserve 40 holds rule-0..rule-3 (40 tokens); the TAIL drops.
        let kept: Vec<&ContextCandidate> = plan
            .selected
            .iter()
            .filter(|c| c.kind == CandidateKind::RepoRule)
            .collect();
        assert_eq!(kept.len(), 4);
        for (i, c) in kept.iter().enumerate() {
            assert_eq!(c.id, format!("rule-{i}"), "never a hole in the rules");
        }
        assert_eq!(plan.rules_dropped, 2);
        // Rules never consumed volatile budget: with no evidence the whole
        // remaining 60 tokens are the messages' (all 5 messages of 10
        // tokens fit the 60-token volatile budget).
        assert_eq!(msg_count(&plan), 5);
        assert_eq!(plan.volatile_budget, 60);
    }

    /// Zero budget: the empty volatile window is legal; the rules (head
    /// content) do not resurrect anything.
    #[test]
    fn zero_volatile_budget_yields_an_empty_window() {
        let mut req = ContextPlanRequest::default();
        req.messages = msgs_newest_first(10, 10);
        req.token_budget = 100;
        req.mode.static_tokens = 100;
        let plan = plan_context(req);
        assert!(plan.selected.is_empty());
        assert_eq!(plan.volatile_budget, 0);
    }

    /// A one-budget-accounting check: every selected candidate is a member
    /// of the input pool exactly once (no fabrication, no duplication).
    #[test]
    fn selection_is_subset_and_duplicate_free() {
        let mut req = ContextPlanRequest::default();
        req.messages = msgs_newest_first(8, 10);
        req.token_budget = 100;
        req.tool_notes
            .push(evidence("tool::note1", CandidateKind::ToolNote, 5, 0.9));
        req.subagent_summaries.push(evidence(
            "sub::summary",
            CandidateKind::SubagentSummary,
            20,
            0.8,
        ));
        req.index_evidence
            .push(evidence("sym::x", CandidateKind::Symbol, 3, 0.99));
        let plan = plan_context(req.clone());
        let ids: Vec<String> = plan.selected.iter().map(|c| c.id.clone()).collect();
        let set: std::collections::HashSet<String> = ids.into_iter().collect();
        assert_eq!(set.len(), plan.selected.len(), "no duplicates");
        let pool: std::collections::HashSet<String> = req
            .messages
            .iter()
            .chain(req.index_evidence.iter())
            .chain(req.tool_notes.iter())
            .chain(req.subagent_summaries.iter())
            .map(|c| c.id.clone())
            .collect();
        for c in &plan.selected {
            assert!(pool.contains(&c.id), "fabricated candidate {}", c.id);
        }
        // Ordering vector is parallel and class-correct.
        for (c, class) in plan.selected.iter().zip(&plan.ordering) {
            assert_eq!(*class, MemoryOrdering::of(c.kind));
            assert_ne!(*class, MemoryOrdering::Static);
        }
    }

    use crate::information::{InformationBudget, InformationError, Need};
    use crate::selection::{CandidateRequirement, NeedCoverage};

    fn covered(id: &str, tokens: u32, coverage_ppm: u32, utility: f64) -> ContextCandidate {
        let mut c = evidence(id, CandidateKind::FileNote, tokens, utility);
        c.confidence_ppm = 1_000_000;
        c.freshness_ppm = 1_000_000;
        c.need_coverage = vec![NeedCoverage {
            need_id: "n1".into(),
            coverage_ppm,
        }];
        c
    }

    fn info(token_budget: u32, needs: Vec<Need>) -> InformationBudget {
        InformationBudget {
            token_budget,
            needs,
        }
    }

    /// Needs route the volatile pool through information selection: the
    /// small high-information candidate wins where the baseline selector
    /// keeps the large raw-utility one; empty needs delegate to the
    /// baseline bit-identically.
    #[test]
    fn needs_switch_to_information_selection_and_empty_needs_keep_the_baseline() {
        let mut req = ContextPlanRequest::default();
        req.token_budget = 50;
        req.index_evidence.push(covered("log", 50, 1_000_000, 1.0));
        req.index_evidence.push(covered("tiny", 5, 1_000_000, 0.05));
        let baseline = plan_context(req.clone());
        assert!(baseline.selected.iter().any(|c| c.id == "log"));
        assert!(!baseline.selected.iter().any(|c| c.id == "tiny"));

        let needs = vec![Need {
            id: "n1".into(),
            weight: 1.0,
            required: false,
        }];
        let plan = plan_context_with_information(req.clone(), info(50, needs)).unwrap();
        assert!(plan.selected.iter().any(|c| c.id == "tiny"));
        assert!(!plan.selected.iter().any(|c| c.id == "log"));
        assert_eq!(plan.volatile_budget, 50);

        let same = plan_context_with_information(req, info(50, Vec::new())).unwrap();
        assert_eq!(same, baseline, "no needs: baseline selector preserved");
    }

    /// Required content that cannot fit the effective budget is the typed
    /// Oversized error from the checked planner surface, never a plan with
    /// the required candidate silently dropped.
    #[test]
    fn required_overflow_surfaces_typed_oversized_from_the_planner() {
        let mut req = ContextPlanRequest::default();
        req.token_budget = 10;
        let mut required = covered("must", 100, 1_000_000, 0.0);
        required.requirement = CandidateRequirement::Required;
        req.index_evidence.push(required);
        let err = plan_context_with_information(
            req,
            info(
                10,
                vec![Need {
                    id: "n1".into(),
                    weight: 1.0,
                    required: true,
                }],
            ),
        )
        .expect_err("required content must not be silently dropped");
        assert_eq!(
            err,
            InformationError::Oversized {
                required_tokens: 100,
                token_budget: 10,
            }
        );
    }

    /// The information budget can only tighten the effective volatile
    /// budget; the plan still reports the request-derived volatile budget
    /// and never exceeds either bound.
    #[test]
    fn information_budget_tightens_but_never_widens_the_plan() {
        let mut req = ContextPlanRequest::default();
        req.token_budget = 100;
        req.messages = msgs_newest_first(3, 10);
        req.index_evidence.push(covered("tiny", 5, 1_000_000, 0.0));
        let needs = vec![Need {
            id: "n1".into(),
            weight: 1.0,
            required: false,
        }];
        let plan = plan_context_with_information(req, info(10, needs)).unwrap();
        assert_eq!(plan.volatile_budget, 100);
        assert_eq!(plan.selected_tokens, 5, "effective budget was 10 tokens");
        assert!(plan.selected.iter().any(|c| c.id == "tiny"));
        assert!(
            plan.selected
                .iter()
                .all(|c| c.kind != CandidateKind::Message),
            "no message fits the tightened budget after the evidence"
        );
    }

    /// A closure-backed [`FailurePrior`] for the adversarial planner tests.
    struct RiskPrior<F: Fn(&ContextCandidate) -> f64>(F);

    impl<F: Fn(&ContextCandidate) -> f64> FailurePrior for RiskPrior<F> {
        fn omission_risk(&self, candidate: &ContextCandidate) -> f64 {
            (self.0)(candidate)
        }
    }

    /// A hostile prior that PANICS the moment it is consulted for a Required
    /// candidate: the planner must never call it for one.
    struct PanicOnRequiredPrior;

    impl FailurePrior for PanicOnRequiredPrior {
        fn omission_risk(&self, candidate: &ContextCandidate) -> f64 {
            assert_ne!(
                candidate.requirement,
                CandidateRequirement::Required,
                "the prior must never be consulted for Required candidates"
            );
            2.0
        }
    }

    fn need_covered(id: &str, need_id: &str, tokens: u32, coverage_ppm: u32) -> ContextCandidate {
        let mut c = evidence(id, CandidateKind::FileNote, tokens, 0.0);
        c.confidence_ppm = 1_000_000;
        c.freshness_ppm = 1_000_000;
        c.need_coverage = vec![NeedCoverage {
            need_id: need_id.into(),
            coverage_ppm,
        }];
        c
    }

    /// Parity when the prior is off: `None` is bit-identical to the existing
    /// planner surface on both the utility and the information path.
    #[test]
    fn prior_off_is_bit_identical_to_the_existing_planner_surface() {
        let mut req = ContextPlanRequest::default();
        req.messages = msgs_newest_first(12, 10);
        req.token_budget = 120;
        req.mode.static_tokens = 10;
        req.rules
            .push(evidence("rules::a", CandidateKind::RepoRule, 5, 1.0));
        req.index_evidence
            .push(evidence("sym::a", CandidateKind::Symbol, 5, 0.9));
        assert_eq!(
            plan_context(req.clone()),
            plan_context_with_prior(req.clone(), None),
            "prior None must be byte-identical"
        );
        let needs = vec![Need {
            id: "n1".into(),
            weight: 1.0,
            required: false,
        }];
        assert_eq!(
            plan_context_with_information(req.clone(), info(60, needs.clone())).unwrap(),
            plan_context_with_information_and_prior(req, info(60, needs), None).unwrap(),
            "prior None must be byte-identical on the information path"
        );
    }

    /// The prior doubles protected non-Required candidates at most; Required
    /// candidates keep their exact utility and are never consulted — a prior
    /// that would boost a Required candidate is ignored.
    #[test]
    fn prior_boosts_optional_and_required_is_untouchable() {
        let mut req = ContextPlanRequest::default();
        req.token_budget = 100;
        let mut required = evidence("required-note", CandidateKind::FileNote, 10, 0.5);
        required.requirement = CandidateRequirement::Required;
        req.index_evidence.push(required);
        req.index_evidence
            .push(evidence("optional-note", CandidateKind::FileNote, 10, 0.3));
        let prior = RiskPrior(|_: &ContextCandidate| 2.0);
        let baseline = plan_context(req.clone());
        let adjusted = plan_context_with_prior(req.clone(), Some(&prior));
        let utility = |plan: &ContextPlan, id: &str| {
            plan.selected.iter().find(|c| c.id == id).map(|c| c.utility)
        };
        assert_eq!(utility(&baseline, "required-note"), Some(0.5));
        assert_eq!(utility(&baseline, "optional-note"), Some(0.3));
        assert_eq!(
            utility(&adjusted, "required-note"),
            Some(0.5),
            "Required utilities are untouchable"
        );
        assert_eq!(
            utility(&adjusted, "optional-note"),
            Some(0.6),
            "non-Required utilities get exactly the clamped 2x"
        );

        // Only one 10-token slot: the prior panics if it is ever asked about
        // the Required note (so this completing proves it was never
        // consulted), while the optional note's 2x boost still loses to the
        // Required note's untouched 0.9.
        let mut req = ContextPlanRequest::default();
        req.token_budget = 10;
        let mut required = evidence("required-note", CandidateKind::FileNote, 10, 0.9);
        required.requirement = CandidateRequirement::Required;
        req.index_evidence.push(required);
        req.index_evidence
            .push(evidence("optional-note", CandidateKind::FileNote, 10, 0.05));
        let guarded = plan_context_with_prior(req, Some(&PanicOnRequiredPrior));
        assert_eq!(guarded.selected.len(), 1);
        assert_eq!(guarded.selected[0].id, "required-note");
        assert_eq!(
            guarded.selected[0].utility, 0.9,
            "the crafted Required boost is ignored"
        );
    }

    /// The information path plumbs the same prior into gain ranking: a 2x
    /// protected optional wins the residue while the required coverage set
    /// and required tokens stay identical.
    #[test]
    fn information_prior_boosts_gain_and_keeps_required_coverage_intact() {
        let mut req = ContextPlanRequest::default();
        req.token_budget = 20;
        let mut required = need_covered("required", "must", 10, 1_000_000);
        required.requirement = CandidateRequirement::Required;
        req.index_evidence.push(required);
        req.index_evidence
            .push(need_covered("a-cheap", "nice", 10, 500_000));
        req.index_evidence
            .push(need_covered("b-risked", "nice", 10, 500_000));
        let needs = vec![
            Need {
                id: "must".into(),
                weight: 0.1,
                required: true,
            },
            Need {
                id: "nice".into(),
                weight: 1.0,
                required: false,
            },
        ];
        let base = plan_context_with_information(req.clone(), info(20, needs.clone())).unwrap();
        assert!(base.selected.iter().any(|c| c.id == "a-cheap"));
        assert!(!base.selected.iter().any(|c| c.id == "b-risked"));

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
            plan_context_with_information_and_prior(req, info(20, needs), Some(&prior)).unwrap();
        assert!(boosted.selected.iter().any(|c| c.id == "b-risked"));
        assert!(!boosted.selected.iter().any(|c| c.id == "a-cheap"));
        assert!(
            boosted.selected.iter().any(|c| c.id == "required"),
            "required coverage is never displaced"
        );
        assert_eq!(
            boosted.selected_tokens, base.selected_tokens,
            "the required overhead is unchanged"
        );
    }

    /// Hostile prior risks (NaN/inf/negative/huge) clamp to [1, 2]: no
    /// selected utility exceeds 2x its base, nothing non-finite enters the
    /// plan, the budget holds, and 25 replans are bit-identical.
    #[test]
    fn hostile_prior_risks_never_exceed_double_and_stay_deterministic() {
        let mut req = ContextPlanRequest::default();
        req.token_budget = 50;
        for i in 0..64u32 {
            req.index_evidence.push(evidence(
                &format!("e-{i:03}"),
                CandidateKind::FileNote,
                10,
                0.25,
            ));
        }
        for risk in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1e300, f64::MAX] {
            let prior = RiskPrior(move |_: &ContextCandidate| risk);
            let first = plan_context_with_prior(req.clone(), Some(&prior));
            for _ in 0..25 {
                assert_eq!(first, plan_context_with_prior(req.clone(), Some(&prior)));
            }
            let total: u64 = first
                .selected
                .iter()
                .map(|c| u64::from(c.estimate_tokens))
                .sum();
            assert!(total <= 50, "budget respected under risk {risk}");
            assert!(!first.selected.is_empty());
            for c in &first.selected {
                assert!(
                    c.utility.is_finite() && c.utility > 0.0,
                    "non-finite/zero utility leaked under risk {risk}: {c:?}"
                );
                assert!(c.utility <= 0.5, "risk {risk} exceeded the 2x clamp: {c:?}");
            }
        }
    }
}
