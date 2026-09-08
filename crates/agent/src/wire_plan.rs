//! The runtime's wire-plan entry (P0-27): the plan the provider receives is
//! built from the [`faktor_context::planner`]'s `ContextCandidate`
//! selection — there is exactly ONE selector on this path. The renderer
//! (`faktor_context::plan_wire_request`) only renders: every history slice
//! and evidence list handed to it is the planner's window, so the
//! renderer's own trim loop is provably inert here (guarded below, never
//! relied on). Byte-stable `cacheable_prefix` semantics are untouched: the
//! static + semi-stable head is rendered by the same renderer as wave-14,
//! so the hashed head stays byte-identical under volatile churn.
//!
//! Budgeting: the static head and the tool schemas are measured EXACTLY
//! through the renderer itself (a probe with an empty volatile tail — the
//! render never trims head or tools, so the probe is exact); the volatile
//! budget left over competes in the planner. A small deterministic reserve
//! covers the volatile-tail render envelope (section headers + per-item
//! separators) so the final render can never exceed `context_max` and the
//! renderer's trim loop can never fire on the runtime path.

use faktor_context::planner::{plan_context, ContextPlanRequest, PlannerMode};
use faktor_context::selection::{CandidateKind, ContextCandidate};
use faktor_context::wire_plan::{plan_wire_request, WirePlan};
use faktor_context::{
    estimate_for_model, ContextBudget, Estimator, Evidence, TaskLedger, TokenCache,
};
use faktor_provider::{ContentKind, RequestMessage, ToolSpec};

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

/// Count one TEXT run through the model-targeted token cache (P0-81):
/// `estimate_for_model` routes the run under the tokenizer the plan's
/// model maps to and falls back to the conservative generic estimator —
/// the SAME value the renderer's internal accounting produces today — so
/// the budget lockstep with `faktor_context::wire_plan` is unchanged while
/// repeat (model, content-hash) pairs stop re-estimating.
fn text_tokens(model: &str, cache: &TokenCache, text: &str) -> usize {
    usize::try_from(estimate_for_model(model, text, cache)).unwrap_or(usize::MAX)
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
/// that window. Deterministic; an oversized static head or schema set is
/// `Err(Oversized)` exactly as the renderer alone would have reported.
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
) -> faktor_core::Result<WirePlan> {
    let context_max = budget.context_max();
    if context_max == 0 {
        return Err(faktor_core::error::Error::new(
            faktor_core::error::ErrorKind::Oversized,
            "context budget leaves no room for content",
        ));
    }
    let est = Estimator;

    // Exact fixed costs through the renderer: one probe with tools, one
    // without — the system head is identical in both, so the difference is
    // exactly the schema estimate and the tool-less total is the head.
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
    // blocks — never exceeds the plan the renderer will produce, so the
    // renderer's own trim loop stays inert (see module docs). The header is
    // static text: it is a cache hit on every plan after the first.
    let header_reserve = text_tokens(model, cache, EVIDENCE_HEADER).saturating_add(3);
    let volatile_budget = u32::try_from(
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
    let mut budget_used = volatile_budget;
    for _ in 0..8u32 {
        let (messages_kept, evidence_kept) = select_window(
            &message_candidates,
            &evidence_candidates,
            &ev_by_id,
            evidence,
            budget_used,
        );
        let plan_messages = &history[history.len() - messages_kept..];
        let rendered = plan_wire_request(
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
        )?;
        if rendered.messages.len() == messages_kept && rendered.total_tokens <= context_max {
            // The renderer kept exactly the planner's window: its inline
            // trim never fired (one selector). Deterministic end state.
            return Ok(rendered);
        }
        // Unreachable drift guard (see the module docs' inequality: per-item
        // envelopes + the header reserve dominate the rendered tail, so the
        // render can never exceed the plan). If a formula ever drifts, shrink
        // the volatile budget deterministically and re-plan; the empty
        // window always fits (the probes rendered above), so this converges.
        let over = rendered.total_tokens.saturating_sub(context_max);
        budget_used = budget_used.saturating_sub(over as u32).saturating_sub(1);
    }
    Err(faktor_core::error::Error::new(
        faktor_core::error::ErrorKind::Oversized,
        "planner envelope drift: render exceeded the planned budget",
    ))
}

/// Price every candidate ONCE per plan call (P0-81): message and evidence
/// block texts go through the model-targeted cache. Returns the message
/// candidates (newest-first), the evidence candidates and the evidence
/// index map, all reused by every shrink pass of the caller.
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
        });
    }
    (messages, ev_candidates, ev_by_id)
}

/// Run the pure planner over the already-priced candidates and map its
/// selection back to concrete slices: `messages_kept` (the newest
/// contiguous window of the oldest-first `history`) and the kept evidence
/// entries (renderer order).
fn select_window(
    message_candidates: &[ContextCandidate],
    evidence_candidates: &[ContextCandidate],
    ev_by_id: &std::collections::HashMap<String, usize>,
    evidence: &[Evidence],
    volatile_budget: u32,
) -> (usize, Vec<Evidence>) {
    let plan = plan_context(ContextPlanRequest {
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
    });
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

    #[test]
    fn planned_dimensions_surface_the_plans_real_totals() {
        let plan = WirePlan {
            system: String::new(),
            messages: vec![],
            tools: vec![],
            total_tokens: 12_345,
            cacheable_prefix_len: 0,
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
    /// the cacheable boundary, and the renderer's trim loop never fired.
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
        assert_eq!(probe.system, plan.cacheable_prefix().unwrap());
    }

    /// Tight-budget adversarial: 50 evidence items of 10 tokens + a
    /// high-utility symbol vs a 100-token volatile budget. The OLD behavior
    /// dropped the whole evidence section (`include_evidence = false`)
    /// when the budget ran tight; the planner trades message slots instead.
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
        let old_way = plan_wire_request(
            "s",
            "",
            &[],
            "",
            &TaskLedger::default(),
            "",
            &msgs,
            &[],
            "",
            &b,
        )
        .unwrap();
        assert!(
            plan.messages.len() < old_way.messages.len(),
            "message window must shrink instead of dropping all evidence: {} vs {}",
            plan.messages.len(),
            old_way.messages.len()
        );
        assert!(plan.messages.len() < msgs.len(), "window is bounded");
        // Contiguity: still a newest suffix.
        assert_eq!(
            plan.messages,
            msgs[msgs.len() - plan.messages.len()..],
            "never a hole in the conversation window"
        );
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
}
