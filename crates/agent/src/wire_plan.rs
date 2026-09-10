//! The runtime's wire-plan entry (P0-27 + audit 33): the plan the provider
//! receives is built from the [`faktor_context::planner`]'s
//! `ContextCandidate` selection — there is exactly ONE selector on this
//! path. The renderer (`faktor_context::plan_wire_request`) is PURE: it
//! never trims. When the render does not fit, the renderer returns the typed
//! `WirePlanError::Oversized { actual_tokens, section_costs }` and THIS
//! entry deterministically replans a smaller volatile window (bounded loop)
//! instead of deleting sections. Required content that alone exceeds the
//! budget is a terminal typed Oversized.
//!
//! Budgeting: the static head and the tool schemas are measured EXACTLY
//! through the renderer itself (a probe with an empty volatile tail — the
//! render never trims head or tools, so the probe is exact); the volatile
//! budget left over competes in the planner. A small deterministic reserve
//! covers the volatile-tail render envelope (section headers + per-item
//! separators) so the final render fits without extra replans; when a
//! formula ever drifts, the bounded replan loop shrinks the volatile budget
//! by the reported overage and re-renders — it NEVER deletes a section.

use faktor_context::information::{FailurePrior, InformationBudget};
use faktor_context::planner::{
    plan_context, plan_context_with_information_and_prior, ContextPlanRequest, PlannerMode,
};
use faktor_context::selection::{CandidateKind, ContextCandidate};
use faktor_context::wire_plan::{plan_wire_request, SectionCosts, WirePlan, WirePlanError};
use faktor_context::{
    estimate_for_model, ContextBudget, Estimator, Evidence, TaskLedger, TokenCache,
};
use faktor_provider::{ContentKind, RequestMessage, Role, ToolSpec};

/// The rendered volatile-tail section header (mirror of the renderer's
/// literal). Reserved ahead of the volatile budget so the rendered system
/// (head + header + blocks) never exceeds the estimate the planner priced.
const EVIDENCE_HEADER: &str = "\n## Retrieved evidence\n";

/// Per-evidence-block render overhead above the block's own text estimate
/// (`\n### ` + path + `\n` + snippet + `\n`, then the concatenation safety
/// margin). Each block's candidate carries `est(block) + 2`; the header is
/// reserved separately with +3 slack (see the module docs for the
/// inequality).
const BLOCK_OVERHEAD_TOKENS: u32 = 2;

/// Bound of the deterministic replan loop (audit 33): each iteration shrinks
/// the volatile budget by at least the reported overage, so the loop
/// converges; the bound only guards against a hostile arithmetic drift —
/// exhausting it returns the LAST typed Oversized error, never a trimmed
/// plan.
pub const MAX_REPLANS: u32 = 8;

/// Count one TEXT run through the model-targeted token cache (P0-81):
/// `estimate_for_model` routes the run under the tokenizer the plan's
/// model maps to and falls back to the conservative generic estimator —
/// the SAME value the renderer's internal accounting produces today — so
/// the budget lockstep with `faktor_context::wire_plan` is unchanged while
/// repeat (model, content-hash) pairs stop re-estimating.
fn text_tokens(model: &str, cache: &TokenCache, text: &str) -> usize {
    // Budget with the SAME conservative estimator the renderer charges
    // (`faktor_context::wire_plan` measures every section through
    // `Estimator`). The model-targeted count still routes through the
    // cache (P0-81), but a real BPE backend counts repetitive text cheaper
    // than the generic floor — pricing a candidate BELOW what the renderer
    // charges let the bounded replan loop overshoot the budget and silently
    // converge to an EMPTY conversation window (`plan.messages.is_empty()`)
    // while the budget still had room. Taking the max keeps the documented
    // lockstep: planner price >= renderer price, so a selected window
    // always renders inside the budget.
    let counted = usize::try_from(estimate_for_model(model, text, cache)).unwrap_or(usize::MAX);
    counted.max(Estimator.estimate_tokens(text))
}

/// The renderer's per-message envelope constants, mirrored exactly:
/// `2` for role + message envelope and `1` per content part (kept in lock
/// with `faktor_context::wire_plan::estimate_message`; a drift here only
/// shifts the deterministic reserve, never the budget). Text runs go
/// through the cache ([`text_tokens`]); structured JSON inputs keep the
/// estimator's direct formula (nothing to cache — the value is already
/// materialized and the renderer charges it verbatim).
fn estimate_message(model: &str, cache: &TokenCache, est: &Estimator, m: &RequestMessage) -> usize {
    let mut t = 2usize;
    for p in &m.content {
        t = t.saturating_add(match &p.kind {
            ContentKind::Text { text } => text_tokens(model, cache, text),
            ContentKind::Reasoning { text } => text_tokens(model, cache, text),
            ContentKind::Image { url } => text_tokens(model, cache, url).max(1),
            ContentKind::ToolCall { id, name, input } => text_tokens(model, cache, id)
                .saturating_add(text_tokens(model, cache, name))
                .saturating_add(est.estimate_json(input))
                .saturating_add(2),
            ContentKind::ToolResult { content, is_error } => {
                text_tokens(model, cache, content).saturating_add(usize::from(*is_error))
            }
        });
        t = t.saturating_add(1);
    }
    t
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

/// The single runtime wire-plan entry: plan from candidates, then render.
///
/// `history` is the chronological (oldest-first) bounded conversation the
/// loader produced; `evidence` every retrieved-evidence hit of the turn.
/// The planner picks the window; the returned [`WirePlan`] carries exactly
/// that window. The renderer is pure, so an overflow is handled here by a
/// BOUNDED deterministic replan loop over the volatile window:
///
/// ```text
/// render -> Err(Oversized { actual_tokens, section_costs })
///        -> required alone too large? terminal typed Oversized
///        -> else volatile_budget -= (actual_tokens - context_max) + 1
///        -> replan (select window + pairing-aware trim) -> render ...
/// ```
///
/// No iteration of the loop can delete a conceptual section: evidence and
/// history only SHRINK through the one selector; static/rules/contract/
/// tools/map/progress are never touched.
///
/// P0-81: `model` is the model the plan targets (the runtime's routed
/// model) and `cache` the runtime's [`TokenCache`]; every text run the
/// planner prices routes through `estimate_for_model` under that model's
/// tokenizer. Today that falls back to the conservative generic estimator
/// — byte-for-byte the renderer's values, so the budget lockstep is
/// unchanged — while repeat (tokenizer, content-hash) pairs inside the
/// plan loop hit the cache instead of re-estimating.
#[allow(clippy::too_many_arguments)]
pub fn plan_wire_turn(
    instructions: &str,
    system_extra: &str,
    tool_schemas: &[ToolSpec],
    project_rules: &str,
    ledger: &TaskLedger,
    repo_map: &str,
    history: &[RequestMessage],
    evidence: &[Evidence],
    budget: &ContextBudget,
    model: &str,
    cache: &TokenCache,
) -> Result<WirePlan, WirePlanError> {
    plan_wire_turn_with_prior(
        instructions,
        system_extra,
        tool_schemas,
        project_rules,
        ledger,
        repo_map,
        history,
        evidence,
        budget,
        model,
        cache,
        None,
    )
}

/// [`plan_wire_turn`] with the optional failure-aware omission prior (audit
/// 68) threaded into the ONE planner call: when `prior` is `Some`, the
/// selector runs through
/// [`plan_context_with_information_and_prior`] (empty information needs —
/// the production path carries no needs today, so this is exactly the
/// utility path plus the prior's `base * clamp(risk, 1, 2)` adjustment of
/// every non-Required candidate); when `None`, the EXACT
/// [`plan_context`] baseline runs. A hostile prior (NaN/inf/negative/huge
/// risks) can only ever protect a candidate up to 2x and never panics —
/// sanitization belongs to `faktor_context`'s `prior_adjusted_gain`, and
/// Required candidates are never consulted. Head bytes, tool schemas and
/// the cacheable boundary are prior-independent: the prior may only change
/// WHICH volatile messages/evidence the planner selects.
///
/// The production caller ([`crate::runtime`]) passes the prior only when
/// `AgentDeps.efficiency.failure_learning` is on AND
/// `AgentDeps.context_prior` is installed; with either off this is
/// byte-identical to [`plan_wire_turn`].
#[allow(clippy::too_many_arguments)]
pub fn plan_wire_turn_with_prior(
    instructions: &str,
    system_extra: &str,
    tool_schemas: &[ToolSpec],
    project_rules: &str,
    ledger: &TaskLedger,
    repo_map: &str,
    history: &[RequestMessage],
    evidence: &[Evidence],
    budget: &ContextBudget,
    model: &str,
    cache: &TokenCache,
    prior: Option<&(dyn FailurePrior + Send + Sync)>,
) -> Result<WirePlan, WirePlanError> {
    let context_max = budget.context_max();
    if context_max == 0 {
        return Err(WirePlanError::Oversized {
            actual_tokens: 0,
            section_costs: SectionCosts::default(),
        });
    }
    let est = Estimator;

    // Exact fixed costs through the renderer: one probe with tools, one
    // without — the system head is identical in both, so the difference is
    // exactly the schema estimate and the tool-less total is the head. A
    // probe failure means the REQUIRED content alone is too large: terminal
    // typed Oversized, unchanged.
    let with_tools = plan_wire_request(
        instructions,
        system_extra,
        tool_schemas,
        project_rules,
        ledger,
        repo_map,
        &[],
        &[],
        "",
        budget,
    )?;
    let head_only = plan_wire_request(
        instructions,
        system_extra,
        &[],
        project_rules,
        ledger,
        repo_map,
        &[],
        &[],
        "",
        budget,
    )?;
    let head_tokens = head_only.total_tokens;
    let tools_tokens = with_tools.total_tokens.saturating_sub(head_tokens);

    // Volatile budget: what the planner competes for. The header reserve
    // (+3 slack) guarantees the rendered system — head + header + selected
    // blocks — never exceeds the plan the renderer will produce; the
    // bounded replan loop below is the safety net if a formula drifts.
    let header_reserve = text_tokens(model, cache, EVIDENCE_HEADER).saturating_add(3);
    let mut volatile_budget = u32::try_from(
        context_max
            .saturating_sub(head_tokens)
            .saturating_sub(tools_tokens)
            .saturating_sub(header_reserve),
    )
    .unwrap_or(u32::MAX);

    // The selector runs over the whole turn content: 20k-message sessions
    // still get a bounded planner window here, never a loader-size window.
    // Candidates are priced ONCE per plan call (each message/block text is
    // hashed and cache-looked-up a single time; the shrink passes below
    // only re-run the pure planner over the same priced candidates).
    let (message_candidates, evidence_candidates, ev_by_id) =
        price_candidates(history, evidence, model, cache, &est);
    let mut last_error: Option<WirePlanError> = None;
    for _ in 0..MAX_REPLANS {
        let (messages_kept, evidence_kept) = select_window(
            &message_candidates,
            &evidence_candidates,
            &ev_by_id,
            evidence,
            volatile_budget,
            prior,
        );
        // The planner window is a contiguous newest suffix; a suffix that
        // starts on a tool result whose call was cut off would dangle, so
        // the window drops that leading result (a result never dangles
        // without its call) — the renderer never rewrites the history.
        let messages_kept = pairing_aware_window(history, messages_kept);
        let plan_messages = &history[history.len() - messages_kept..];
        match plan_wire_request(
            instructions,
            system_extra,
            tool_schemas,
            project_rules,
            ledger,
            repo_map,
            plan_messages,
            &evidence_kept,
            "",
            budget,
        ) {
            Ok(rendered) => return Ok(rendered),
            Err(err) => {
                // Required content alone cannot be replanned away: terminal.
                if err.section_costs().required_tokens() > context_max {
                    return Err(err);
                }
                // Deterministic shrink: at least the reported overage + 1.
                let over = err.actual_tokens().saturating_sub(context_max).max(1);
                let next = volatile_budget.saturating_sub(u32::try_from(over).unwrap_or(u32::MAX));
                last_error = Some(err);
                if next == volatile_budget {
                    // The volatile window is already empty and still does
                    // not fit: bounded end state, never a second selector.
                    break;
                }
                volatile_budget = next;
            }
        }
    }
    Err(last_error.unwrap_or(WirePlanError::Oversized {
        actual_tokens: context_max.saturating_add(1),
        section_costs: SectionCosts::default(),
    }))
}

/// Keep a planner message window pairing-aware (the renderer never rewrites
/// history): while the OLDEST kept message is a user message carrying a
/// tool result whose call is not inside the window, drop that oldest
/// message. Calls themselves are never dropped: a call's result, when it
/// exists, always sits newer than the call and therefore inside a suffix
/// window that contains the call.
fn pairing_aware_window(history: &[RequestMessage], mut kept: usize) -> usize {
    kept = kept.min(history.len());
    while kept > 0 {
        let start = history.len() - kept;
        let head = &history[start];
        if head.role != Role::User || !has_tool_result(head) {
            break;
        }
        let calls: std::collections::HashSet<&str> = history[start..]
            .iter()
            .flat_map(|m| m.content.iter())
            .filter_map(|p| match &p.kind {
                ContentKind::ToolCall { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        let dangling = head.content.iter().any(|p| match &p.kind {
            ContentKind::ToolResult { .. } => p
                .tool_call_id
                .as_deref()
                .is_none_or(|id| !calls.contains(id)),
            _ => false,
        });
        if dangling {
            kept -= 1;
        } else {
            break;
        }
    }
    kept
}

fn has_tool_result(m: &RequestMessage) -> bool {
    m.content
        .iter()
        .any(|p| matches!(p.kind, ContentKind::ToolResult { .. }))
}

/// Price every candidate ONCE per plan call (P0-81): message and evidence
/// block texts go through the model-targeted cache. Returns the message
/// candidates (newest-first), the evidence candidates and the evidence
/// index map, all reused by every replan pass of the caller.
#[allow(clippy::type_complexity)]
fn price_candidates(
    history: &[RequestMessage],
    evidence: &[Evidence],
    model: &str,
    cache: &TokenCache,
    est: &Estimator,
) -> (
    Vec<ContextCandidate>,
    Vec<ContextCandidate>,
    std::collections::HashMap<String, usize>,
) {
    // Message candidates newest-first (the durable loader contract), sized
    // by the renderer's exact per-message accounting (text runs through the
    // cache — repeat plans of identical content hit instead of estimating).
    let mut messages: Vec<ContextCandidate> = Vec::with_capacity(history.len());
    for (i, m) in history.iter().rev().enumerate() {
        let tokens = estimate_message(model, cache, est, m);
        messages.push(ContextCandidate {
            id: format!("msg:{i}"),
            kind: CandidateKind::Message,
            bytes: 0,
            estimate_tokens: u32::try_from(tokens).unwrap_or(u32::MAX),
            utility: 1.0,
            ..ContextCandidate::default()
        });
    }
    // Evidence candidates: the exact rendered block text (the renderer's
    // `truncate(snippet, 1500)` shape) plus the per-block envelope, so the
    // render of a selected block can never cost more than its candidate
    // priced. Utilities sanitized: non-finite → 0 (never selected), huge →
    // clamped to [0, 1].
    let mut seen = std::collections::HashSet::new();
    let mut ev_candidates: Vec<ContextCandidate> = Vec::with_capacity(evidence.len());
    let mut ev_by_id: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (idx, ev) in evidence.iter().enumerate() {
        if !seen.insert(ev.path.clone()) {
            continue; // duplicate paths render once (first occurrence wins)
        }
        let block = format!("\n### {}\n{}\n", ev.path, truncate(&ev.snippet, 1500));
        let tokens =
            text_tokens(model, cache, &block).saturating_add(BLOCK_OVERHEAD_TOKENS as usize);
        let utility = if ev.score.is_finite() {
            ev.score.clamp(0.0, 1.0)
        } else {
            0.0
        };
        ev_by_id.insert(ev.path.clone(), idx);
        ev_candidates.push(ContextCandidate {
            id: ev.path.clone(),
            kind: CandidateKind::Symbol,
            bytes: block.len(),
            estimate_tokens: u32::try_from(tokens).unwrap_or(u32::MAX),
            utility,
            ..ContextCandidate::default()
        });
    }
    (messages, ev_candidates, ev_by_id)
}

/// Run the pure planner over the already-priced candidates and map its
/// selection back to concrete slices: `messages_kept` (the newest
/// contiguous window of the oldest-first `history`) and the kept evidence
/// entries (renderer order).
///
/// With `prior` present the call goes through
/// [`plan_context_with_information_and_prior`] with EMPTY information needs
/// (the production path declares none today): that is the utility path with
/// the prior folded into non-Required utilities, and it cannot fail — the
/// typed `InformationError::Oversized` is only produced by a non-empty
/// required-needs selection. With `prior` absent the original
/// [`plan_context`] call runs verbatim (parity).
fn select_window(
    message_candidates: &[ContextCandidate],
    evidence_candidates: &[ContextCandidate],
    ev_by_id: &std::collections::HashMap<String, usize>,
    evidence: &[Evidence],
    volatile_budget: u32,
    prior: Option<&(dyn FailurePrior + Send + Sync)>,
) -> (usize, Vec<Evidence>) {
    let request = ContextPlanRequest {
        messages: message_candidates.to_vec(),
        rules: Vec::new(),
        index_evidence: evidence_candidates.to_vec(),
        tool_notes: Vec::new(),
        subagent_summaries: Vec::new(),
        token_budget: volatile_budget,
        min_utility: 0.0,
        mode: PlannerMode {
            static_tokens: 0,
            semi_stable_tokens: 0,
        },
    };
    let plan = match prior {
        Some(prior) => plan_context_with_information_and_prior(
            request,
            InformationBudget::default(),
            Some(prior),
        )
        .expect("empty information needs cannot produce a typed information error"),
        None => plan_context(request),
    };
    let messages_kept = plan
        .selected
        .iter()
        .filter(|c| c.kind == CandidateKind::Message)
        .count();
    let mut evidence_kept: Vec<Evidence> = Vec::new();
    for c in &plan.selected {
        if let Some(&idx) = ev_by_id.get(&c.id) {
            evidence_kept.push(evidence[idx].clone());
        }
    }
    (messages_kept, evidence_kept)
}

/// The REAL dimensions of one planned wire request, surfaced for the
/// routing consult (attempt-accounting audit D): the input estimate is the
/// plan's OWN total (the exact bytes the wire request will carry — the
/// planner's render never exceeds it) and the output cap is what the caller
/// will accept from the model. The routed decision prices and qualifies
/// against these — never against a hard-coded 16384/2048 guess made before
/// the plan existed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlannedRequestDimensions {
    /// The plan's own input estimate in tokens (`WirePlan::total_tokens`
    /// after the planner's final render).
    pub input_estimate_tokens: u64,
    /// The execution output cap in tokens (the caller's planned bound;
    /// typically the planning model's max output).
    pub output_cap_tokens: u64,
}

/// Surface the planned request's real token counts for routing. The plan is
/// the FINAL wire plan of the iteration (after compaction replanning): its
/// `total_tokens` IS the input estimate the provider will be asked to read,
/// and `output_cap_tokens` bounds the output the route must price.
pub fn planned_request_dimensions(
    plan: &WirePlan,
    output_cap_tokens: usize,
) -> PlannedRequestDimensions {
    PlannedRequestDimensions {
        input_estimate_tokens: u64::try_from(plan.total_tokens).unwrap_or(u64::MAX),
        output_cap_tokens: u64::try_from(output_cap_tokens).unwrap_or(u64::MAX),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faktor_context::wire_plan::WirePlan;
    use faktor_context::{PromptSegments, WirePlanError};

    #[test]
    fn planned_dimensions_surface_the_plans_real_totals() {
        let plan = WirePlan {
            system: String::new(),
            messages: vec![],
            tools: vec![],
            total_tokens: 12_345,
            cacheable_prefix_len: 0,
            prompt_segments: PromptSegments::default(),
        };
        let dims = planned_request_dimensions(&plan, 4096);
        assert_eq!(
            dims.input_estimate_tokens, 12_345,
            "the route input estimate IS the plan's own total — the old 16384 guess is gone"
        );
        assert_eq!(dims.output_cap_tokens, 4096);
    }

    use faktor_context::ledger::TaskLedger;
    use faktor_provider::{ContentPart, Role};

    fn ledger() -> TaskLedger {
        TaskLedger {
            goal: "fix the parser".into(),
            open_steps: vec!["reproduce crash".into()],
            ..Default::default()
        }
    }

    fn tool(name: &str) -> ToolSpec {
        ToolSpec {
            name: name.into(),
            description: format!("{name} description"),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
            }),
        }
    }

    fn text_history(n: usize) -> Vec<RequestMessage> {
        (0..n)
            .map(|i| RequestMessage {
                role: if i % 2 == 0 {
                    Role::User
                } else {
                    Role::Assistant
                },
                content: vec![ContentPart::text(format!("turn {i}: {}", "x".repeat(780)))],
            })
            .collect()
    }

    fn small_history(n: usize) -> Vec<RequestMessage> {
        (0..n)
            .map(|i| RequestMessage {
                role: Role::User,
                content: vec![ContentPart::text(format!("small turn {i}"))],
            })
            .collect()
    }

    fn evidence(n: usize) -> Vec<Evidence> {
        (0..n)
            .map(|i| Evidence {
                path: format!("src/f{i}.rs"),
                snippet: format!("fn f{i}() {{}} // {}", "y".repeat(200)),
                score: (n - i) as f64,
            })
            .collect()
    }

    /// The model every plan-wiring test targets (o200k_base family).
    const TEST_MODEL: &str = "gpt-5";

    fn cache() -> TokenCache {
        TokenCache::new()
    }

    /// The plan the provider receives must be byte-identical in the head
    /// and exact in the window: history slices handed to the renderer are
    /// the planner's (contiguous newest window), the volatile tail follows
    /// the cacheable boundary, and the renderer never trims.
    #[test]
    fn planner_window_renders_without_renderer_trimming() {
        let b = ContextBudget::default();
        let cache = cache();
        let history = text_history(400);
        let ev = evidence(6);
        let plan = plan_wire_turn(
            "You are Faktor.\n",
            "",
            &[tool("echo")],
            "",
            &ledger(),
            "",
            &history,
            &ev,
            &b,
            TEST_MODEL,
            &cache,
        )
        .unwrap();
        assert!(plan.total_tokens <= b.context_max());
        // The planner window is a suffix of history (newest kept), and the
        // renderer did NOT trim further: messages equal the planned slice.
        assert!(plan.messages.len() < 400 && !plan.messages.is_empty());
        assert_eq!(
            plan.messages,
            history[history.len() - plan.messages.len()..],
            "the render must send exactly the planner's window"
        );
        // Evidence survives budget pressure through utility, never as a
        // wholesale-dropped section.
        assert!(plan.system.contains("## Retrieved evidence"));
        // Byte-stable head: the boundary sits before the volatile tail and
        // equals the head a tool-less static render produces.
        let probe = plan_wire_request(
            "You are Faktor.\n",
            "",
            &[],
            "",
            &ledger(),
            "",
            &[],
            &[],
            "",
            &b,
        )
        .unwrap();
        assert_eq!(
            probe.cacheable_prefix().unwrap(),
            plan.cacheable_prefix().unwrap()
        );
    }

    /// Tight-budget adversarial: 50 evidence items of 10 tokens + a
    /// high-utility symbol vs a small volatile budget. The OLD renderer
    /// dropped the whole evidence section when the budget ran tight; the
    /// planner trades message slots instead, and the typed Oversized replan
    /// loop never deletes a section.
    #[test]
    fn tight_budget_keeps_evidence_and_shrinks_the_message_window() {
        // A tight budget: context_max = 200 tokens total.
        let b = ContextBudget {
            system: 200,
            tools: 0,
            working: 0,
            retrieved: 0,
            recent: 0,
            output_reserve: 0,
            safety: 0,
        };
        let msgs = small_history(50);
        let mut ev: Vec<Evidence> = (0..50u32)
            .map(|i| Evidence {
                path: format!("dull/{i}.rs"),
                snippet: "d".repeat(80),
                score: 0.01,
            })
            .collect();
        ev.push(Evidence {
            path: "src/parser.rs".into(),
            snippet: "pub fn parse() {}".into(),
            score: 1.0,
        });
        let cache = cache();
        let plan = plan_wire_turn(
            "s",
            "",
            &[],
            "",
            &TaskLedger::default(),
            "",
            &msgs,
            &ev,
            &b,
            TEST_MODEL,
            &cache,
        )
        .unwrap();
        assert!(
            plan.system.contains("src/parser.rs"),
            "high-utility evidence must survive a tight budget"
        );
        assert!(plan.total_tokens <= b.context_max());
        // What the old code did (drop ALL evidence, keep max messages):
        // the planner-only counterfactual with no evidence at all.
        let counterfactual = plan_wire_turn(
            "s",
            "",
            &[],
            "",
            &TaskLedger::default(),
            "",
            &msgs,
            &[],
            &b,
            TEST_MODEL,
            &cache,
        )
        .unwrap();
        assert!(
            plan.messages.len() < counterfactual.messages.len(),
            "message window must shrink instead of dropping all evidence: {} vs {}",
            plan.messages.len(),
            counterfactual.messages.len()
        );
        assert!(plan.messages.len() < msgs.len(), "window is bounded");
        // Contiguity: still a newest suffix.
        assert_eq!(
            plan.messages,
            msgs[msgs.len() - plan.messages.len()..],
            "never a hole in the conversation window"
        );
    }

    /// Required content alone over budget is a TERMINAL typed Oversized at
    /// this entry too (no replan can remove instructions/rules/contract).
    #[test]
    fn required_content_alone_too_large_is_a_terminal_typed_oversized() {
        let b = ContextBudget {
            system: 100,
            tools: 0,
            working: 0,
            retrieved: 0,
            recent: 0,
            output_reserve: 0,
            safety: 0,
        };
        let err = plan_wire_turn(
            &"i".repeat(4000),
            "steer",
            &[tool("echo")],
            &"r".repeat(4000),
            &ledger(),
            &"m".repeat(2000),
            &small_history(10),
            &evidence(2),
            &b,
            TEST_MODEL,
            &cache(),
        )
        .unwrap_err();
        match err {
            WirePlanError::Oversized {
                actual_tokens,
                section_costs,
            } => {
                assert!(actual_tokens > b.context_max());
                assert!(section_costs.required_tokens() > b.context_max());
            }
        }
    }

    /// Determinism: identical inputs → identical plan 50 runs (whole wire
    /// plan, byte-for-byte).
    #[test]
    fn plan_wire_turn_is_deterministic() {
        let b = ContextBudget::default();
        let cache = cache();
        let history = text_history(120);
        let ev = evidence(8);
        let first = plan_wire_turn(
            "You are Faktor.\n",
            "steer",
            &[tool("echo")],
            "rules",
            &ledger(),
            "map",
            &history,
            &ev,
            &b,
            TEST_MODEL,
            &cache,
        )
        .unwrap();
        for _ in 1..50 {
            let again = plan_wire_turn(
                "You are Faktor.\n",
                "steer",
                &[tool("echo")],
                "rules",
                &ledger(),
                "map",
                &history,
                &ev,
                &b,
                TEST_MODEL,
                &cache,
            )
            .unwrap();
            assert_eq!(first.system, again.system);
            assert_eq!(first.messages, again.messages);
            assert_eq!(first.cacheable_prefix_len, again.cacheable_prefix_len);
            assert_eq!(first.prompt_segments, again.prompt_segments);
        }
    }

    /// P0-81 wiring: the planner's text runs route through the cache under
    /// the model the plan targets. Identical re-plans hit (same content
    /// hash + same tokenizer identity); the plan math is unchanged because
    /// the fallback is the estimator's own values.
    #[test]
    fn wire_plan_routes_text_counts_through_the_model_cache() {
        let b = ContextBudget::default();
        let cache = cache();
        let history = text_history(120);
        let ev = evidence(8);
        let plan = |model: &str, cache: &TokenCache| {
            plan_wire_turn(
                "You are Faktor.\n",
                "steer",
                &[tool("echo")],
                "rules",
                &ledger(),
                "map",
                &history,
                &ev,
                &b,
                model,
                cache,
            )
            .unwrap()
        };
        let first = plan(TEST_MODEL, &cache);
        assert!(cache.misses() > 0, "the first plan must populate the cache");
        let misses_after_first = cache.misses();
        // Identical content under the same model/tokenizer: the re-plan's
        // text runs (system header, messages, evidence blocks) hit instead
        // of re-estimating.
        let again = plan(TEST_MODEL, &cache);
        assert!(
            cache.misses() == misses_after_first && cache.hits() > 0,
            "a byte-identical re-plan must be all hits"
        );
        assert_eq!(first.total_tokens, again.total_tokens);
        assert_eq!(first.messages, again.messages);
        // A different model family (cl100k) re-keys the SAME content as new
        // entries but still yields the identical conservative plan.
        let before = cache.misses();
        let cl100k_plan = plan("gpt-4", &cache);
        assert!(
            cache.misses() > before,
            "a different tokenizer family must miss on identical content"
        );
        assert_eq!(cl100k_plan.total_tokens, first.total_tokens);
    }

    /// Cacheable-boundary regression: reorder-flip evidence (score + input
    /// order) never moves the boundary and never changes the hashed head.
    #[test]
    fn evidence_flip_never_moves_the_cacheable_boundary() {
        let b = ContextBudget::default();
        let cache = cache();
        let common = ("You are Faktor.\n", "rules", TaskLedger::default(), "map");
        let run = |ev: &[Evidence]| {
            plan_wire_turn(
                common.0,
                "",
                &[tool("echo")],
                common.1,
                &common.2,
                common.3,
                &[],
                ev,
                &b,
                TEST_MODEL,
                &cache,
            )
            .unwrap()
        };
        let p1 = run(&[
            Evidence {
                path: "src/a.rs".into(),
                snippet: "a".into(),
                score: 1.0,
            },
            Evidence {
                path: "src/b.rs".into(),
                snippet: "b".into(),
                score: 9.0,
            },
        ]);
        let p2 = run(&[
            Evidence {
                path: "src/b.rs".into(),
                snippet: "b".into(),
                score: 1.0,
            },
            Evidence {
                path: "src/a.rs".into(),
                snippet: "a".into(),
                score: 9.0,
            },
        ]);
        assert_eq!(p1.cacheable_prefix_len, p2.cacheable_prefix_len);
        assert_eq!(
            p1.cacheable_prefix().unwrap(),
            p2.cacheable_prefix().unwrap()
        );
        let prefix = p1.cacheable_prefix().unwrap();
        assert!(!prefix.contains("src/a.rs") && !prefix.contains("## Retrieved evidence"));
    }

    /// A planner window that would start in the middle of a tool exchange
    /// must drop the dangling leading RESULT (the renderer no longer sweeps
    /// history): every surviving result is answered by a call inside the
    /// window, and calls are never dropped.
    #[test]
    fn pairing_aware_window_never_dangles_a_tool_result() {
        let mut history = Vec::new();
        for i in 0..40 {
            let id = format!("call_{i}");
            history.push(RequestMessage {
                role: Role::User,
                content: vec![ContentPart::text(format!("prompt {i}"))],
            });
            history.push(RequestMessage {
                role: Role::Assistant,
                content: vec![ContentPart::tool_call(
                    id.clone(),
                    "echo",
                    serde_json::json!({"x": i}),
                )],
            });
            history.push(RequestMessage {
                role: Role::User,
                content: vec![ContentPart::tool_result(
                    format!("result {i} {}", "y".repeat(600)),
                    false,
                    id,
                )],
            });
        }
        // Force the window to start EXACTLY on a tool-result message: keep
        // the newest 4 messages so history[len-kept] is a tool result whose
        // call was cut off (history[116] is the result of call_38).
        let kept = 4;
        let start = history.len() - kept;
        assert!(
            history[start]
                .content
                .iter()
                .any(|p| matches!(&p.kind, ContentKind::ToolResult { .. })),
            "fixture must start on a tool result"
        );
        assert_eq!(pairing_aware_window(&history, kept), kept - 1);
        assert!(!has_tool_result(&history[history.len() - (kept - 1)]));
        // A well-paired window is untouched.
        assert_eq!(pairing_aware_window(&history, 3), 3);
        assert_eq!(pairing_aware_window(&history, 2), 2);
        assert_eq!(pairing_aware_window(&history, 0), 0);
        // And an end-to-end tight plan never sends a dangling result.
        let b = ContextBudget {
            system: 900,
            tools: 0,
            working: 0,
            retrieved: 0,
            recent: 0,
            output_reserve: 0,
            safety: 0,
        };
        let plan = plan_wire_turn(
            "s",
            "",
            &[tool("echo")],
            "",
            &TaskLedger::default(),
            "",
            &history,
            &[],
            &b,
            TEST_MODEL,
            &cache(),
        )
        .unwrap();
        let mut seen_calls: std::collections::HashSet<String> = std::collections::HashSet::new();
        for m in &plan.messages {
            if m.role == Role::User {
                for p in &m.content {
                    if let ContentKind::ToolResult { .. } = &p.kind {
                        let id = p.tool_call_id.as_deref().unwrap();
                        assert!(seen_calls.contains(id), "result {id} dangles");
                    }
                }
            }
            for p in &m.content {
                if let ContentKind::ToolCall { id, .. } = &p.kind {
                    seen_calls.insert(id.clone());
                }
            }
        }
    }

    /// 20k-message session: the provider receives a bounded window and the
    /// window is exactly the planner's (not the loader's, not a greedy
    /// fill): the newest messages ride inside the volatile budget while
    /// evidence keeps its reserved slice.
    #[test]
    fn twenty_thousand_message_session_gets_the_planner_window() {
        let b = ContextBudget::default();
        let history = text_history(20_000);
        let mut ev = evidence(3);
        ev[0].score = 1e9; // hostile huge score: clamped, still selected
        ev.push(Evidence {
            path: "sym.rs".into(),
            snippet: "fn hot() {}".into(),
            score: f64::NAN, // hostile NaN: rejected, never selected
        });
        let cache = cache();
        let plan = plan_wire_turn(
            "You are Faktor.\n",
            "",
            &[tool("echo")],
            "rules",
            &ledger(),
            "map",
            &history,
            &ev,
            &b,
            TEST_MODEL,
            &cache,
        )
        .unwrap();
        // (Performance is a release-mode distribution gate in
        // tests/performance — see perf_context_plan_20k_message_window.)
        assert!(plan.messages.len() < 20_000, "window must be bounded");
        assert!(!plan.messages.is_empty());
        assert_eq!(
            plan.messages,
            history[history.len() - plan.messages.len()..],
            "the planner picked the newest window"
        );
        assert!(
            !plan.system.contains("sym.rs"),
            "NaN-score evidence must never be selected"
        );
        assert!(plan.system.contains("src/f2.rs"));
        assert!(plan.total_tokens <= b.context_max());
    }

    /// Hostile content never panics and never exceeds the budget: unicode,
    /// huge snippets, duplicate paths, zero/negative scores.
    #[test]
    fn hostile_evidence_and_history_never_panic_and_stay_bounded() {
        let b = ContextBudget::default();
        let mut hostile = Vec::new();
        for i in 0..50 {
            hostile.push(RequestMessage {
                role: Role::User,
                content: vec![ContentPart::text(format!("😀{i} {}", "汉".repeat(3000)))],
            });
        }
        let mut ev = evidence(6);
        for e in &mut ev {
            e.snippet = "é".repeat(5000);
            e.score = -1.0; // never selected
        }
        ev.push(Evidence {
            path: "dup.rs".into(),
            snippet: "first".into(),
            score: 0.9,
        });
        ev.push(Evidence {
            path: "dup.rs".into(),
            snippet: "second".into(),
            score: 0.99,
        });
        ev.push(Evidence {
            path: "zero.rs".into(),
            snippet: "zero".into(),
            score: 0.0,
        });
        let cache = cache();
        let plan = plan_wire_turn(
            &"s".repeat(500),
            "",
            &[tool("echo")],
            "r",
            &ledger(),
            "m",
            &hostile,
            &ev,
            &b,
            TEST_MODEL,
            &cache,
        )
        .unwrap();
        assert!(plan.total_tokens <= b.context_max());
        assert!(!plan.messages.is_empty());
        // Duplicate path rendered once.
        assert_eq!(plan.system.matches("### dup.rs").count(), 1);
        assert!(plan.system.is_char_boundary(plan.cacheable_prefix_len));
    }

    /// A closure-backed [`FailurePrior`] for the adversarial wiring tests.
    struct RiskPrior<F: Fn(&ContextCandidate) -> f64>(F);

    impl<F: Fn(&ContextCandidate) -> f64> FailurePrior for RiskPrior<F> {
        fn omission_risk(&self, candidate: &ContextCandidate) -> f64 {
            (self.0)(candidate)
        }
    }

    fn plan_with_prior(
        history: &[RequestMessage],
        ev: &[Evidence],
        budget: &ContextBudget,
        cache: &TokenCache,
        prior: Option<&(dyn FailurePrior + Send + Sync)>,
    ) -> WirePlan {
        plan_wire_turn_with_prior(
            "You are Faktor.\n",
            "steer",
            &[tool("echo")],
            "rules",
            &ledger(),
            "map",
            history,
            ev,
            budget,
            TEST_MODEL,
            cache,
            prior,
        )
        .unwrap()
    }

    /// Parity when the prior is off (or absent): the new prior-aware entry
    /// with `None` — and with a neutrally-scored (risk 1.0) handle — is
    /// byte-identical to the legacy [`plan_wire_turn`] baseline on the whole
    /// rendered plan (system, messages, tools, canonical counts, boundary).
    #[test]
    fn prior_off_is_byte_identical_to_the_baseline_plan() {
        let b = ContextBudget::default();
        let cache = cache();
        let history = text_history(120);
        let ev = evidence(8);
        let baseline = plan_wire_turn(
            "You are Faktor.\n",
            "steer",
            &[tool("echo")],
            "rules",
            &ledger(),
            "map",
            &history,
            &ev,
            &b,
            TEST_MODEL,
            &cache,
        )
        .unwrap();
        let via_api = plan_with_prior(&history, &ev, &b, &cache, None);
        assert_eq!(baseline, via_api, "None must be the legacy plan verbatim");
        let neutral = RiskPrior(|_: &ContextCandidate| 1.0);
        let neutral_plan = plan_with_prior(&history, &ev, &b, &cache, Some(&neutral));
        assert_eq!(
            baseline, neutral_plan,
            "risk 1.0 is neutral: the plan must stay byte-identical"
        );
    }

    /// A prior may change WHICH volatile evidence the planner selects — but
    /// only that: the cacheable head (instructions/rules/ledger/map/steering
    /// and the tool schemas), the messages window contract and the total
    /// budget are prior-independent. Crafted: two equal-priced evidence
    /// blocks compete for room for exactly one; the prior doubles `a`'s
    /// omission risk (0.5 -> 1.0) so it wins over `b` (0.6).
    #[test]
    fn prior_on_changes_only_the_non_required_volatile_selection() {
        let b = ContextBudget {
            system: 260,
            tools: 0,
            working: 0,
            retrieved: 0,
            recent: 0,
            output_reserve: 0,
            safety: 0,
        };
        let cache = cache();
        // No conversation: the two evidence blocks are the only volatile
        // competitors, and both are Optional (never consulted for Required).
        let history: Vec<RequestMessage> = Vec::new();
        let ev = vec![
            Evidence {
                path: "src/a.rs".into(),
                snippet: "a".repeat(400),
                score: 0.5,
            },
            Evidence {
                path: "src/b.rs".into(),
                snippet: "b".repeat(400),
                score: 0.6,
            },
        ];
        let off = plan_with_prior(&history, &ev, &b, &cache, None);
        assert!(off.total_tokens <= b.context_max());
        assert!(
            off.system.contains("### src/b.rs"),
            "baseline: the 0.6 block wins the single slot; system={}",
            off.system
        );
        assert!(
            !off.system.contains("### src/a.rs"),
            "baseline: only one block fits; system={}",
            off.system
        );
        assert!(!off.system.contains("### src/a.rs"));
        assert!(off.messages.is_empty());

        let boost_a = RiskPrior(
            |c: &ContextCandidate| {
                if c.id == "src/a.rs" {
                    2.0
                } else {
                    1.0
                }
            },
        );
        let on = plan_with_prior(&history, &ev, &b, &cache, Some(&boost_a));
        assert!(
            on.system.contains("### src/a.rs"),
            "prior on: the protected 0.5 block wins the slot"
        );
        assert!(!on.system.contains("### src/b.rs"));
        assert_ne!(off.system, on.system, "the prior must change selection");
        assert_eq!(
            off.cacheable_prefix().unwrap(),
            on.cacheable_prefix().unwrap(),
            "head bytes are prior-independent"
        );
        assert_eq!(off.cacheable_prefix_len, on.cacheable_prefix_len);
        assert_eq!(off.messages, on.messages, "message window unchanged");
        assert_eq!(off.tools, on.tools);
        assert!(on.total_tokens <= b.context_max());
    }

    /// Hostile prior risks (NaN, +-inf, huge negative, huge positive, zero)
    /// never panic, never exceed the budget, stay deterministic, and the
    /// prior can never force a non-finite or over-2x utility into a plan:
    /// `prior_adjusted_gain` owns the clamp.
    #[test]
    fn hostile_prior_risks_never_panic_and_stay_bounded_and_deterministic() {
        let b = ContextBudget::default();
        let cache = cache();
        let history = text_history(150);
        let ev = evidence(10);
        for risk in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -1e300,
            0.0,
            f64::MAX,
        ] {
            let prior = RiskPrior(move |_: &ContextCandidate| risk);
            let first = plan_with_prior(&history, &ev, &b, &cache, Some(&prior));
            assert!(
                first.total_tokens <= b.context_max(),
                "risk {risk} overran the budget"
            );
            let again = plan_with_prior(&history, &ev, &b, &cache, Some(&prior));
            assert_eq!(first, again, "risk {risk} must stay deterministic");
        }
    }

    /// A prior that panics on a Required candidate can never be triggered
    /// from this path: every candidate the wire entry prices is
    /// non-Required (Optional), so a guarded hostile prior completes.
    #[test]
    fn prior_is_never_consulted_for_required_candidates_on_the_wire_path() {
        struct PanicOnRequired;
        impl FailurePrior for PanicOnRequired {
            fn omission_risk(&self, candidate: &ContextCandidate) -> f64 {
                assert_ne!(
                    candidate.requirement,
                    faktor_context::CandidateRequirement::Required,
                    "the wire path must never hand the prior a Required candidate"
                );
                1.0
            }
        }
        let b = ContextBudget::default();
        let cache = cache();
        let history = text_history(60);
        let ev = evidence(4);
        let plan = plan_with_prior(&history, &ev, &b, &cache, Some(&PanicOnRequired));
        assert!(plan.total_tokens <= b.context_max());
    }
}
