//! Wire request planning (audit round 5, P0): the context budget must bound
//! the provider request (conservative normalized estimate — not a
//! provider-tokenizer-exact count), and every conceptual element must appear
//! exactly once:
//!
//! ```text
//! system    = STATIC PREFIX + SEMI-STABLE rendered as ONE byte-stable head
//!             (architecture §8.4: instructions, project rules; task ledger,
//!             repo map; steering note last inside the head) + a volatile
//!             tail (retrieved evidence, current errors) appended AFTER the
//!             head — volatile content is appended last so provider prompt
//!             caching is not invalidated by every new turn. The exact head
//!             byte range is exposed as `WirePlan::cacheable_prefix_len` /
//!             `WirePlan::cacheable_prefix()`: the byte-truth cacheable
//!             prefix of the sent request (class-ordered per §8.4:
//!             StaticPrefix before SemiStable; evidence/recent turns/errors
//!             never precede the boundary).
//! messages  = the structured history, each message exactly once, with tool
//!             calls/results as structured parts.
//! tools     = the schemas, once — never embedded in `system`.
//! ```
//!
//! `total_tokens = estimate(system) + Σ estimate(message) + estimate(tools)`
//! is enforced BEFORE anything reaches the wire. Deterministic trimming when
//! over: oldest history messages first, PAIRING-AWARE (dropping an assistant
//! tool-call message also drops the following user message carrying its tool
//! result — a result never dangles without its call), then evidence, then
//! errors. Still over with empty history → `Err(Oversized)`. The planner
//! never returns an unbudgeted plan.

use std::collections::HashSet;

use faktor_core::error::{Error, ErrorKind};
use faktor_provider::{ContentKind, RequestMessage, Role, ToolSpec};

use crate::assembler::Evidence;
use crate::budget::ContextBudget;
use crate::estimator::Estimator;
use crate::ledger::TaskLedger;

/// A conservative normalized-request estimate (`estimator.rs`: never below
/// the chars/3.4 floor, plus hand-written envelope estimates);
/// provider-specific tokenizers are future work.
#[derive(Debug, Clone, PartialEq)]
pub struct WirePlan {
    pub system: String,
    pub messages: Vec<RequestMessage>,
    pub tools: Vec<ToolSpec>,
    pub total_tokens: usize,
    /// Byte length of the cacheable-prefix head of [`WirePlan::system`]
    /// (architecture §8.4, audits 65-66): the StaticPrefix + SemiStable
    /// render (instructions, project rules, task ledger, repo map, steering
    /// note). Volatile content (retrieved evidence, current errors) is
    /// appended after this boundary and can never precede it. The planner
    /// copies the head verbatim into `system`, so the boundary is a char
    /// boundary of the render by construction; callers that record prefix
    /// observations (the usage-settlement fill site) hash
    /// `&system[..cacheable_prefix_len]` as the exact bytes the wire
    /// request sent.
    pub cacheable_prefix_len: usize,
}

impl WirePlan {
    /// The byte-exact cacheable-prefix head of [`WirePlan::system`]
    /// (StaticPrefix + SemiStable, architecture §8.4). `None` only when the
    /// boundary is not on a char boundary of the render — impossible by
    /// construction (the head is copied verbatim); callers must hash
    /// nothing on `None` rather than guess.
    pub fn cacheable_prefix(&self) -> Option<&str> {
        self.system.get(..self.cacheable_prefix_len)
    }
}

/// Plan the wire request under the budget. `total_tokens` never exceeds
/// `budget.context_max()`; an untrimmable plan is `Err(Oversized)`, never an
/// unbudgeted success.
#[allow(clippy::too_many_arguments)]
pub fn plan_wire_request(
    instructions: &str,
    system_extra: &str,
    tool_schemas: &[ToolSpec],
    project_rules: &str,
    ledger: &TaskLedger,
    repo_map: &str,
    history: &[RequestMessage],
    evidence: &[Evidence],
    errors: &str,
    budget: &ContextBudget,
) -> faktor_core::Result<WirePlan> {
    let est = Estimator;
    let context_max = budget.context_max();
    if context_max == 0 {
        return Err(Error::new(
            ErrorKind::Oversized,
            "context budget leaves no room for content",
        ));
    }

    // The cacheable prefix: static + semi-stable only. Evidence/errors are a
    // volatile tail appended later (documented; the prefix stays byte-stable).
    // Class order inside the head (§8.4): instructions + project rules
    // (StaticPrefix) first, then task ledger + repo map + steering note
    // (SemiStable). Volatile content never precedes `cacheable_prefix_len`.
    let static_part =
        build_static_part(instructions, system_extra, project_rules, ledger, repo_map);
    let tools_tokens = estimate_tools(&est, tool_schemas);
    let mut messages = history.to_vec();
    let mut messages_tokens = estimate_messages(&est, &messages);
    let mut include_evidence = !evidence.is_empty();
    let mut include_errors = !errors.is_empty();

    loop {
        let system = render_system(
            &static_part,
            evidence,
            include_evidence,
            errors,
            include_errors,
        );
        let system_tokens = est.estimate_tokens(&system);
        let total = system_tokens
            .saturating_add(messages_tokens)
            .saturating_add(tools_tokens);
        if total <= context_max {
            return Ok(WirePlan {
                // The head is copied verbatim (push_str of the same String),
                // so `static_part.len()` is the byte-truth boundary inside
                // `system`: everything from there on is volatile tail.
                cacheable_prefix_len: static_part.len(),
                system,
                messages,
                tools: tool_schemas.to_vec(),
                total_tokens: total,
            });
        }
        // Trim order (deterministic): oldest history messages first, pairing
        // aware; then the evidence tail; then the errors tail.
        if !messages.is_empty() {
            drop_oldest_pairing_aware(&est, &mut messages, &mut messages_tokens);
            continue;
        }
        if include_evidence {
            include_evidence = false;
            continue;
        }
        if include_errors {
            include_errors = false;
            continue;
        }
        return Err(Error::new(
            ErrorKind::Oversized,
            format!(
                "wire request estimated at {total} tokens exceeds budget context_max {} even with empty history",
                budget.context_max()
            ),
        ));
    }
}

/// The byte-stable cacheable head (architecture §8.4), ordered by class so
/// the most stable content leads and rewrite-prone content ends at the
/// volatile boundary:
///
/// ```text
/// StaticPrefix: instructions, then project rules
/// SemiStable:   task ledger, then repository map
/// SemiStable:   steering note LAST (durable control state — a Steer
///               REPLACES the note, so a rewrite can only invalidate the
///               volatile tail, never the static prefix or the ledger)
/// ```
///
/// Never contains tool schemas or conversation. The volatile tail
/// (evidence, errors) is appended after this head by `render_system` and
/// can never move into it.
fn build_static_part(
    instructions: &str,
    system_extra: &str,
    project_rules: &str,
    ledger: &TaskLedger,
    repo_map: &str,
) -> String {
    let mut out = String::new();
    out.push_str(instructions);
    if !project_rules.is_empty() {
        out.push_str("\n## Project rules\n");
        out.push_str(project_rules);
    }
    out.push_str("\n## Task state\n");
    out.push_str(&ledger.compact_render());
    if !repo_map.is_empty() {
        out.push_str("\n## Repository map\n");
        out.push_str(&truncate(repo_map, 2000));
    }
    if !system_extra.is_empty() {
        out.push_str("\n## Steering\n");
        out.push_str(system_extra);
    }
    out
}

/// The volatile tail appended AFTER the cacheable head: retrieved evidence
/// (highest score first, bounded snippets), then current errors. Nothing in
/// this function can move bytes before `static_part` — the boundary is
/// exactly `static_part.len()` into the assembled system.
fn render_system(
    static_part: &str,
    evidence: &[Evidence],
    include_evidence: bool,
    errors: &str,
    include_errors: bool,
) -> String {
    let mut out = String::new();
    out.push_str(static_part);
    if include_evidence {
        let mut scored: Vec<&Evidence> = evidence.iter().collect();
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.path.cmp(&b.path))
        });
        out.push_str("\n## Retrieved evidence\n");
        for ev in scored {
            out.push_str("\n### ");
            out.push_str(&ev.path);
            out.push('\n');
            out.push_str(&truncate(&ev.snippet, 1500));
            out.push('\n');
        }
    }
    if include_errors {
        out.push_str("\n## Current errors\n");
        out.push_str(errors);
    }
    out
}

/// Drop the oldest message. When it is an assistant message carrying a tool
/// call, the immediately following user message carrying its tool result is
/// dropped with it (the result never dangles without its call). A final
/// sweep removes any still-dangling tool results.
fn drop_oldest_pairing_aware(
    est: &Estimator,
    messages: &mut Vec<RequestMessage>,
    messages_tokens: &mut usize,
) {
    if messages.is_empty() {
        return;
    }
    let dropped = messages.remove(0);
    *messages_tokens = messages_tokens.saturating_sub(estimate_message(est, &dropped));
    if dropped.role == Role::Assistant && has_tool_call(&dropped) {
        if let Some(next) = messages.first() {
            if next.role == Role::User && has_tool_result(next) {
                let t = estimate_message(est, next);
                messages.remove(0);
                *messages_tokens = messages_tokens.saturating_sub(t);
            }
        }
    }
    let removed = sweep_dangling_tool_results(messages);
    for m in &removed {
        *messages_tokens = messages_tokens.saturating_sub(estimate_message(est, m));
    }
}

/// Adversarial sweep: any user message whose tool result names a call that is
/// not present earlier in the remaining history is dropped whole — a result
/// never dangles without its call, even when the input history is hostile.
fn sweep_dangling_tool_results(messages: &mut Vec<RequestMessage>) -> Vec<RequestMessage> {
    let mut seen_calls: HashSet<String> = HashSet::new();
    let mut removed = Vec::new();
    let mut i = 0usize;
    while i < messages.len() {
        let m = &messages[i];
        if m.role == Role::User && has_tool_result(m) {
            let dangling = m.content.iter().any(|p| match &p.kind {
                ContentKind::ToolResult { .. } => p
                    .tool_call_id
                    .as_deref()
                    .is_none_or(|id| !seen_calls.contains(id)),
                _ => false,
            });
            if dangling {
                removed.push(messages.remove(i));
                continue;
            }
        }
        for p in &m.content {
            if let ContentKind::ToolCall { id, .. } = &p.kind {
                seen_calls.insert(id.clone());
            }
        }
        i += 1;
    }
    removed
}

fn has_tool_call(m: &RequestMessage) -> bool {
    m.content
        .iter()
        .any(|p| matches!(p.kind, ContentKind::ToolCall { .. }))
}

fn has_tool_result(m: &RequestMessage) -> bool {
    m.content
        .iter()
        .any(|p| matches!(p.kind, ContentKind::ToolResult { .. }))
}

fn estimate_messages(est: &Estimator, messages: &[RequestMessage]) -> usize {
    messages.iter().map(|m| estimate_message(est, m)).sum()
}

fn estimate_message(est: &Estimator, m: &RequestMessage) -> usize {
    let mut t = 2usize; // role + message envelope
    for p in &m.content {
        t = t.saturating_add(match &p.kind {
            ContentKind::Text { text } => est.estimate_tokens(text),
            ContentKind::Reasoning { text } => est.estimate_tokens(text),
            ContentKind::Image { url } => est.estimate_tokens(url).max(1),
            ContentKind::ToolCall { id, name, input } => est
                .estimate_tokens(id)
                .saturating_add(est.estimate_tokens(name))
                .saturating_add(est.estimate_json(input))
                .saturating_add(2),
            ContentKind::ToolResult { content, is_error } => est
                .estimate_tokens(content)
                .saturating_add(usize::from(*is_error)),
        });
        t = t.saturating_add(1); // part envelope
    }
    t
}

fn estimate_tools(est: &Estimator, specs: &[ToolSpec]) -> usize {
    specs
        .iter()
        .map(|s| {
            est.estimate_tokens(&s.name)
                .saturating_add(est.estimate_tokens(&s.description))
                .saturating_add(est.estimate_json(&s.input_schema))
                .saturating_add(2)
        })
        .sum()
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
    use faktor_provider::ContentPart;

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

    fn evidence(n: usize) -> Vec<Evidence> {
        (0..n)
            .map(|i| Evidence {
                path: format!("src/f{i}.rs"),
                snippet: format!("fn f{i}() {{}} // {}", "y".repeat(200)),
                score: (n - i) as f64,
            })
            .collect()
    }

    #[test]
    fn wire_plan_counts_everything_once() {
        // The wire plan must contain every conceptual element exactly once:
        // instructions + ledger in system, history once in messages, schemas
        // once in tools — and the system render must NOT duplicate the tool
        // schema JSON or the conversation text.
        let b = ContextBudget::default();
        let history = vec![
            RequestMessage {
                role: Role::User,
                content: vec![ContentPart::text("first user turn")],
            },
            RequestMessage {
                role: Role::Assistant,
                content: vec![ContentPart::tool_call(
                    "c1",
                    "read_file",
                    serde_json::json!({"path": "x"}),
                )],
            },
            RequestMessage {
                role: Role::User,
                content: vec![ContentPart::tool_result("ok", false, "c1")],
            },
        ];
        let tools = vec![tool("read_file")];
        let plan = plan_wire_request(
            "You are Faktor.\n",
            "extra",
            &tools,
            "no global state",
            &ledger(),
            "src/",
            &history,
            &evidence(1),
            "boom",
            &b,
        )
        .unwrap();
        // System: instructions + ledger + rules, never the conversation.
        assert!(plan.system.starts_with("You are Faktor.\n"));
        assert!(plan.system.contains("GOAL: fix the parser"));
        assert!(plan.system.contains("no global state"));
        assert!(!plan.system.contains("first user turn"));
        assert!(!plan.system.contains("read_file"));
        // Messages: exactly the history, once, in order.
        assert_eq!(plan.messages, history);
        // Tools: exactly the schemas, once.
        assert_eq!(plan.tools, tools);
        let tool_json = serde_json::to_string(&tools[0]).unwrap();
        assert!(
            !plan.system.contains(&tool_json),
            "tool schema JSON must not leak into system"
        );
    }

    #[test]
    fn wire_plan_budget_enforced_on_total() {
        // 400 messages x ~260 tokens each against the 32K profile: the plan
        // must come in under context_max (25_000) with the OLDEST messages
        // dropped first, and the accounting must stay exact after trimming.
        let b = ContextBudget::default();
        let history = text_history(400);
        let plan = plan_wire_request(
            "You are Faktor.\n",
            "",
            &[],
            "",
            &TaskLedger::default(),
            "",
            &history,
            &[],
            "",
            &b,
        )
        .unwrap();
        assert!(
            plan.total_tokens <= b.context_max(),
            "{} > {}",
            plan.total_tokens,
            b.context_max()
        );
        assert!(plan.total_tokens > 0);
        assert!(plan.messages.len() < 400, "oldest messages must be dropped");
        assert!(!plan.messages.is_empty());
        // Oldest dropped, newest retained: the plan is a suffix of history.
        let kept = plan.messages.len();
        assert_eq!(plan.messages, history[400 - kept..]);
        // Exact math: total == estimate(system) + Σ estimate(message) + estimate(tools).
        let est = Estimator;
        let expected = est.estimate_tokens(&plan.system)
            + estimate_messages(&est, &plan.messages)
            + estimate_tools(&est, &plan.tools);
        assert_eq!(plan.total_tokens, expected, "accounting drifted after trim");
    }

    #[test]
    fn wire_plan_pairing_aware_trim() {
        // 50 pairs of (assistant tool-call message, following user
        // tool-result message) with a budget that cannot hold them all:
        // dropping a call must take its result with it — no result may dangle
        // without its call, and every surviving call keeps its result.
        let b = ContextBudget::default();
        let mut history = Vec::new();
        for i in 0..50 {
            let id = format!("call_{i}");
            history.push(RequestMessage {
                role: Role::Assistant,
                content: vec![
                    ContentPart::text(format!("calling tool for {i}")),
                    ContentPart::tool_call(id.clone(), "echo", serde_json::json!({"x": i})),
                ],
            });
            history.push(RequestMessage {
                role: Role::User,
                content: vec![ContentPart::tool_result(
                    format!("result {i} {}", "y".repeat(3000)),
                    i % 3 == 0,
                    id,
                )],
            });
        }
        let plan = plan_wire_request(
            "You are Faktor.\n",
            "",
            &[tool("echo")],
            "",
            &TaskLedger::default(),
            "",
            &history,
            &[],
            "",
            &b,
        )
        .unwrap();
        assert!(plan.total_tokens <= b.context_max());
        assert!(plan.messages.len() < history.len(), "trimming must happen");
        assert!(
            plan.messages.len().is_multiple_of(2),
            "calls and results must be dropped as pairs"
        );
        let calls = plan
            .messages
            .iter()
            .flat_map(|m| m.content.iter())
            .filter(|p| matches!(p.kind, ContentKind::ToolCall { .. }))
            .count();
        let results = plan
            .messages
            .iter()
            .flat_map(|m| m.content.iter())
            .filter(|p| matches!(p.kind, ContentKind::ToolResult { .. }))
            .count();
        assert_eq!(calls, results, "every surviving call keeps its result");
        // No result dangles: each ToolResult is answered by a ToolCall earlier
        // in the remaining history.
        let mut seen: HashSet<String> = HashSet::new();
        for m in &plan.messages {
            if m.role == Role::User {
                for p in &m.content {
                    if let ContentKind::ToolResult { .. } = &p.kind {
                        let id = p.tool_call_id.as_deref().unwrap();
                        assert!(seen.contains(id), "result {id} dangles without its call");
                    }
                }
            }
            for p in &m.content {
                if let ContentKind::ToolCall { id, .. } = &p.kind {
                    seen.insert(id.clone());
                }
            }
        }
    }

    #[test]
    fn wire_plan_oversized_even_empty_history_is_err() {
        // Tiny budget + a static prefix that alone exceeds it: the plan must
        // be Err(Oversized) — never an unbudgeted Ok, even with empty history.
        let b = ContextBudget {
            system: 200,
            tools: 0,
            working: 0,
            retrieved: 0,
            recent: 0,
            output_reserve: 0,
            safety: 0,
        };
        let err = plan_wire_request(
            &"i".repeat(5000),
            "",
            &[],
            "",
            &TaskLedger::default(),
            "",
            &[],
            &[],
            "",
            &b,
        )
        .unwrap_err();
        assert!(err.kind == ErrorKind::Oversized);
        // Project rules are not truncated: hostile rules alone must error.
        let err = plan_wire_request(
            "",
            "",
            &[],
            &"r".repeat(10_000),
            &TaskLedger::default(),
            "",
            &[],
            &[],
            "",
            &b,
        )
        .unwrap_err();
        assert!(err.kind == ErrorKind::Oversized);
        // Tool schemas are counted in the budget: a hostile schema alone must
        // error even though it never enters `system`.
        let err = plan_wire_request(
            "",
            "",
            &[ToolSpec {
                name: "t".into(),
                description: "d".repeat(10_000),
                input_schema: serde_json::json!({}),
            }],
            "",
            &TaskLedger::default(),
            "",
            &[],
            &[],
            "",
            &b,
        )
        .unwrap_err();
        assert!(err.kind == ErrorKind::Oversized);
        // Zero context_max is never a plan either.
        let zero = ContextBudget {
            system: 0,
            tools: 0,
            working: 0,
            retrieved: 0,
            recent: 0,
            output_reserve: 0,
            safety: 0,
        };
        assert!(
            plan_wire_request(
                "",
                "",
                &[],
                "",
                &TaskLedger::default(),
                "",
                &[],
                &[],
                "",
                &zero
            )
            .unwrap_err()
            .kind
                == ErrorKind::Oversized
        );
    }

    #[test]
    fn wire_plan_evidence_errors_after_static_prefix() {
        // System must START with the static prefix; evidence and errors come
        // after it (volatile tail) — and the static part stays byte-identical
        // across turns so provider prefix caching keeps working.
        let b = ContextBudget::default();
        let prefix = "You are Faktor.\nStay cacheable.";
        let plan = plan_wire_request(
            prefix,
            "",
            &[tool("echo")],
            "rules",
            &ledger(),
            "repo map",
            &[],
            &[Evidence {
                path: "src/a.rs".into(),
                snippet: "fn main()".into(),
                score: 1.0,
            }],
            "current errors here",
            &b,
        )
        .unwrap();
        assert!(plan.system.starts_with(prefix));
        let ev_pos = plan.system.find("src/a.rs").unwrap();
        let err_pos = plan.system.find("current errors here").unwrap();
        assert!(ev_pos > prefix.len(), "evidence must follow the prefix");
        assert!(err_pos > ev_pos, "errors must follow evidence");
        // Turn 2 with different volatile content: everything before the
        // evidence header must be byte-identical.
        let plan2 = plan_wire_request(
            prefix,
            "",
            &[tool("echo")],
            "rules",
            &ledger(),
            "repo map",
            &[],
            &[Evidence {
                path: "src/b.rs".into(),
                snippet: "fn other()".into(),
                score: 9.0,
            }],
            "totally different errors",
            &b,
        )
        .unwrap();
        fn head(s: &str) -> &str {
            s.split("## Retrieved evidence").next().unwrap()
        }
        assert_eq!(head(&plan.system), head(&plan2.system), "prefix drifted");
    }

    #[test]
    fn wire_plan_32k_local_exact_math() {
        // The 32K local profile: total() = 32_000, context_max() = 25_000,
        // and an in-budget plan reports total_tokens as the exact sum of the
        // three components — nothing more, nothing less.
        let b = ContextBudget::default();
        assert_eq!(b.total(), 32_000);
        assert_eq!(b.context_max(), 25_000);
        let history = vec![
            RequestMessage {
                role: Role::User,
                content: vec![
                    ContentPart::text("build the parser"),
                    ContentPart::reasoning("let me think"),
                ],
            },
            RequestMessage {
                role: Role::Assistant,
                content: vec![ContentPart::tool_call(
                    "c1",
                    "echo",
                    serde_json::json!({"msg": "hi"}),
                )],
            },
            RequestMessage {
                role: Role::User,
                content: vec![ContentPart::tool_result("echo: hi", false, "c1")],
            },
        ];
        let tools = vec![tool("echo"), tool("read_file")];
        let plan = plan_wire_request(
            "sys",
            "extra",
            &tools,
            "rules",
            &ledger(),
            "map",
            &history,
            &evidence(1),
            "err",
            &b,
        )
        .unwrap();
        // Everything fits: nothing trimmed, messages/tools intact.
        assert_eq!(plan.messages, history);
        assert_eq!(plan.tools, tools);
        let est = Estimator;
        let expected = est.estimate_tokens(&plan.system)
            + estimate_messages(&est, &plan.messages)
            + estimate_tools(&est, &plan.tools);
        assert_eq!(plan.total_tokens, expected, "exact math broken");
        assert!(plan.total_tokens <= b.context_max());
        assert_eq!(
            b.effective_usage(plan.total_tokens),
            plan.total_tokens as f64 / 25_000.0
        );
    }

    #[test]
    fn wire_plan_unicode_and_hostile_inputs_never_panic() {
        // Oversized unicode, absurd evidence, absurd errors, deep JSON tool
        // schemas: the planner must trim deterministically and never panic.
        let b = ContextBudget::default();
        let mut hostile = Vec::new();
        for i in 0..100 {
            hostile.push(RequestMessage {
                role: Role::User,
                content: vec![ContentPart::text(format!("😀{i} {}", "汉".repeat(3000)))],
            });
        }
        let tools = vec![ToolSpec {
            name: "t".into(),
            description: "d".into(),
            input_schema: {
                let mut v = serde_json::Value::Null;
                for _ in 0..50 {
                    v = serde_json::json!([v]);
                }
                v
            },
        }];
        let plan = plan_wire_request(
            &"s".repeat(5_000),
            "",
            &tools,
            &"r".repeat(2_000),
            &TaskLedger {
                goal: "g".repeat(5_000),
                ..Default::default()
            },
            &"m".repeat(10_000),
            &hostile,
            &[Evidence {
                path: "p".into(),
                snippet: "é".repeat(10_000),
                score: 1.0,
            }],
            &"e".repeat(10_000),
            &b,
        )
        .unwrap();
        assert!(plan.total_tokens <= b.context_max());
        assert!(
            !plan.messages.is_empty(),
            "hostile input must still be trimmed, not lost"
        );
        assert!(plan.system.is_char_boundary(plan.system.len()));
    }

    // ---------------------------------------------------------------- prefix
    // Byte-truth prefix-cache ordering (architecture §8.4, audits 65-66):
    // the plan exposes the exact cacheable-prefix head
    // (`system[..cacheable_prefix_len]`) and volatile content can never
    // move into it.

    fn ledger_with(mut l: TaskLedger, goal: &str) -> TaskLedger {
        l.goal = goal.into();
        l
    }

    fn evidence_set(paths: &[(&str, f64)]) -> Vec<Evidence> {
        paths
            .iter()
            .map(|(path, score)| Evidence {
                path: (*path).into(),
                snippet: format!("snippet of {path}"),
                score: *score,
            })
            .collect()
    }

    #[test]
    fn cacheable_prefix_boundary_never_precedes_static_or_semi_stable_content() {
        // Regression (a): two consecutive turns with the SAME
        // system+instructions+ledger but DIFFERENT volatile tails must
        // produce byte-identical cacheable prefixes (identical digest of
        // the exact head segment), and the volatile content must sit
        // strictly AFTER the boundary — never before it.
        let b = ContextBudget::default();
        let static_input = (
            "You are Faktor.\nStay cacheable.",
            "rules",
            &ledger_with(TaskLedger::default(), "fix the parser"),
            "src/\nlib/",
        );
        let plan_a = plan_wire_request(
            static_input.0,
            "",
            &[tool("echo")],
            static_input.1,
            static_input.2,
            static_input.3,
            &[],
            &evidence_set(&[("src/a.rs", 1.0), ("src/b.rs", 9.0)]),
            "error one",
            &b,
        )
        .unwrap();
        let plan_b = plan_wire_request(
            static_input.0,
            "",
            &[tool("echo")],
            static_input.1,
            static_input.2,
            static_input.3,
            &[],
            &evidence_set(&[("src/z.rs", 0.1)]),
            "a completely different current error",
            &b,
        )
        .unwrap();
        // The boundary is a char boundary and the head is byte-truthful.
        let pa = plan_a.cacheable_prefix().unwrap();
        let pb = plan_b.cacheable_prefix().unwrap();
        assert_eq!(pa, pb, "volatile tail changes must not move the head");
        assert_eq!(
            plan_a.cacheable_prefix_len, plan_b.cacheable_prefix_len,
            "boundary must not move"
        );
        assert!(pa.len() <= plan_a.system.len() && pa.len() <= plan_b.system.len());
        // Volatile content never precedes the boundary in EITHER plan.
        for plan in [&plan_a, &plan_b] {
            let prefix = plan.cacheable_prefix().unwrap();
            for forbidden in [
                "## Retrieved evidence",
                "## Current errors",
                "src/a.rs",
                "src/b.rs",
                "src/z.rs",
                "error one",
            ] {
                assert!(
                    !prefix.contains(forbidden),
                    "volatile content {forbidden:?} leaked into the cacheable head: {prefix:?}"
                );
            }
            let tail = &plan.system[plan.cacheable_prefix_len..];
            assert!(
                tail.contains("## Retrieved evidence") || tail.contains("## Current errors"),
                "the volatile tail must follow the boundary"
            );
        }
        // Digest equality (what the fill site records) follows byte truth.
        assert_eq!(blake3::hash(pa.as_bytes()), blake3::hash(pb.as_bytes()));
    }

    #[test]
    fn ledger_change_moves_boundary_but_keeps_static_part_stable() {
        // Regression (b): a turn whose durable ledger state changed moves
        // the prefix boundary (the head grows) while the static part —
        // everything before the semi-stable task-state section — stays
        // byte-identical.
        let b = ContextBudget::default();
        // The append-consistency sub-case needs the task-state section at
        // the END of the head (no repo map after it), so an end-of-render
        // ledger append is a pure byte-prefix growth.
        let base = (
            "You are Faktor.\n",
            "rules",
            &ledger_with(TaskLedger::default(), "fix the parser"),
            "",
        );
        let p1 = plan_wire_request(
            base.0,
            "",
            &[tool("echo")],
            base.1,
            base.2,
            base.3,
            &[],
            &[],
            "",
            &b,
        )
        .unwrap();
        // The next turn's ledger gained durable state (a touched file — the
        // LAST rendered section, so an end-of-render append).
        let mut l2 = base.2.clone();
        l2.changed_files.push("src/ledger.rs".into());
        let p2 = plan_wire_request(
            base.0,
            "",
            &[tool("echo")],
            base.1,
            &l2,
            base.3,
            &[],
            &[],
            "",
            &b,
        )
        .unwrap();
        let h1 = p1.cacheable_prefix().unwrap();
        let h2 = p2.cacheable_prefix().unwrap();
        assert!(
            p2.cacheable_prefix_len > p1.cacheable_prefix_len,
            "the boundary must move right when the ledger grows"
        );
        assert!(
            h2.starts_with(h1),
            "an end-of-render ledger append must be append-consistent with the previous head"
        );
        // Growth semantics: the old head is the byte-prefix of the new one,
        // so the digest pair is append-consistent (coverage = old/new < 1).
        let ratio = h1.len() as f64 / h2.len() as f64;
        assert!(ratio > 0.0 && ratio < 1.0, "append-consistent growth");
        // A MID-RENDER ledger change (goal extended + criteria appended)
        // still moves the boundary right and keeps the static part stable —
        // only the semi-stable task-state region rewrites.
        let mut l3 = base.2.clone();
        l3.goal.push_str(" (extended criteria)");
        l3.constraints.push("no unsafe".into());
        let p3 = plan_wire_request(
            base.0,
            "",
            &[tool("echo")],
            base.1,
            &l3,
            base.3,
            &[],
            &[],
            "",
            &b,
        )
        .unwrap();
        let h3 = p3.cacheable_prefix().unwrap();
        assert!(p3.cacheable_prefix_len > p1.cacheable_prefix_len);
        assert_ne!(h3, h1, "the ledger rewrite changes the head digest");
        // The STATIC part (before the semi-stable task-state marker) is
        // byte-identical across EVERY boundary move.
        let marker = "## Task state";
        fn static_part<'a>(plan: &'a WirePlan, marker: &str) -> &'a str {
            let m = plan.system.find(marker).unwrap();
            &plan.system[..m]
        }
        for plan in [&p1, &p2, &p3] {
            assert!(static_part(plan, marker).starts_with(base.0));
        }
        assert_eq!(static_part(&p1, marker), static_part(&p2, marker));
        assert_eq!(static_part(&p1, marker), static_part(&p3, marker));
    }

    #[test]
    fn evidence_reorder_never_moves_the_prefix_boundary() {
        // Regression (c): evidence/symbol facts whose score order flips
        // turn-to-turn are VOLATILE — the reorder must not move the prefix
        // boundary and the hashed head must stay byte-identical. Only
        // classes 1-2 (StaticPrefix + SemiStable) precede the boundary.
        let b = ContextBudget::default();
        let common = (
            "You are Faktor.\n",
            "",
            &ledger_with(TaskLedger::default(), "same goal"),
            "",
        );
        // Turn 1 ranks b.rs above a.rs; turn 2 flips the order.
        let p1 = plan_wire_request(
            common.0,
            "",
            &[tool("echo")],
            common.1,
            common.2,
            common.3,
            &[],
            &evidence_set(&[("src/a.rs", 1.0), ("src/b.rs", 9.0)]),
            "",
            &b,
        )
        .unwrap();
        let p2 = plan_wire_request(
            common.0,
            "",
            &[tool("echo")],
            common.1,
            common.2,
            common.3,
            &[],
            &evidence_set(&[("src/a.rs", 9.0), ("src/b.rs", 1.0)]),
            "",
            &b,
        )
        .unwrap();
        assert_eq!(
            p1.cacheable_prefix_len, p2.cacheable_prefix_len,
            "evidence reorder moved the boundary"
        );
        assert_eq!(
            p1.cacheable_prefix().unwrap(),
            p2.cacheable_prefix().unwrap(),
            "evidence reorder changed the hashed head"
        );
        // Equal-score ties are broken by path: any evidence permutation
        // still renders strictly after the boundary.
        let mut paths: Vec<Evidence> = vec![
            Evidence {
                path: "z.rs".into(),
                snippet: "z".into(),
                score: 5.0,
            },
            Evidence {
                path: "a.rs".into(),
                snippet: "a".into(),
                score: 5.0,
            },
            Evidence {
                path: "m.rs".into(),
                snippet: "m".into(),
                score: 5.0,
            },
        ];
        let base = (
            "You are Faktor.\n",
            "",
            &ledger_with(TaskLedger::default(), "tie goal"),
            "",
        );
        for _ in 0..6 {
            let plan = plan_wire_request(
                base.0,
                "",
                &[tool("echo")],
                base.1,
                base.2,
                base.3,
                &[],
                &paths,
                "",
                &b,
            )
            .unwrap();
            let prefix = plan.cacheable_prefix().unwrap();
            let rendered_tail = &plan.system[plan.cacheable_prefix_len..];
            for ev in &paths {
                assert!(
                    !prefix.contains(ev.path.as_str()),
                    "evidence path {} leaked before the boundary",
                    ev.path
                );
                assert!(
                    rendered_tail.contains(ev.path.as_str()),
                    "evidence path {} missing from the volatile tail",
                    ev.path
                );
            }
            paths.rotate_left(1);
        }
    }

    #[test]
    fn steering_note_is_semi_stable_rendered_last_inside_the_head() {
        // The steering note is durable control state (SemiStable), NOT
        // StaticPrefix: it renders AFTER the ledger and repo map so a steer
        // rewrite can only invalidate content after it — the static prefix
        // (instructions + rules) and the task state never sit downstream of
        // a rewrite-prone block. Same-char-count steer rewrites must keep
        // the static prefix byte-identical.
        let b = ContextBudget::default();
        let plan = |note: &str| {
            plan_wire_request(
                "You are Faktor.\n",
                note,
                &[tool("echo")],
                "rules",
                &ledger_with(TaskLedger::default(), "goal"),
                "repo map",
                &[],
                &evidence_set(&[("src/a.rs", 1.0)]),
                "boom",
                &b,
            )
            .unwrap()
        };
        let p1 = plan("do it the blue way");
        let p2 = plan("do it the gold way");
        let h1 = p1.system.find("## Task state").unwrap();
        let h2 = p2.system.find("## Task state").unwrap();
        assert_eq!(&p1.system[..h1], &p2.system[..h2], "static part drifted");
        assert_eq!(h1, h2);
        // Steering renders after the repository map, inside the head.
        assert!(p1.cacheable_prefix().unwrap().contains("## Steering"));
        assert!(p1
            .cacheable_prefix()
            .unwrap()
            .contains("do it the blue way"));
        // The volatile marker follows the head.
        let s1 = p1.system.find("## Steering").unwrap();
        let ev1 = p1.system.find("## Retrieved evidence").unwrap();
        let map = p1.system.find("## Repository map").unwrap();
        assert!(
            map < s1 && s1 < ev1,
            "steer must sit between map and evidence"
        );
    }

    #[test]
    fn hostile_markers_and_multibyte_inputs_never_break_the_boundary() {
        // Adversarial: evidence/error CONTENT that mimics section markers
        // or carries hostile unicode must still render after the boundary,
        // the boundary must stay a valid char boundary, and the head must
        // never contain volatile text — even when trimming drops the
        // volatile tail entirely.
        let b = ContextBudget::default();
        let hostile_evidence = vec![
            Evidence {
                path: "src/## Task state.rs".into(),
                snippet: "## Repository map\n## Steering\n## Current errors".into(),
                score: 1e9,
            },
            Evidence {
                path: "汉/字.rs".into(),
                snippet: "😀".repeat(2000),
                score: -1.0,
            },
        ];
        let plan = plan_wire_request(
            &"s".repeat(3_000),
            "steer 汉字",
            &[tool("echo")],
            &"r".repeat(2_000),
            &TaskLedger {
                goal: "目标".repeat(200),
                ..Default::default()
            },
            &"m".repeat(1_500),
            &[],
            &hostile_evidence,
            &"e".repeat(1_000),
            &b,
        )
        .unwrap();
        let prefix = plan.cacheable_prefix().unwrap();
        assert!(!prefix.contains("## Current errors"));
        assert!(!prefix.contains("## Retrieved evidence"));
        assert!(prefix.contains("steer 汉字"), "steer is part of the head");
        // The boundary is a char boundary (slice accessor returns Some).
        let tail = &plan.system[plan.cacheable_prefix_len..];
        assert!(
            !tail.is_empty() && tail.contains("## Current errors"),
            "hostile volatile content must stay in the tail"
        );
        // Every volatile marker's first occurrence is at/after the boundary.
        for marker in ["## Retrieved evidence", "## Current errors"] {
            if let Some(pos) = plan.system.find(marker) {
                assert!(
                    pos >= plan.cacheable_prefix_len,
                    "{marker} precedes the head"
                );
            }
        }
        assert!(plan.system.is_char_boundary(plan.cacheable_prefix_len));
    }

    #[test]
    fn wire_plan_exposes_one_boundary_and_head_is_prefix_of_system() {
        // The seam contract the fill site relies on: the head is exactly
        // `system[..cacheable_prefix_len]`, `cacheable_prefix()` agrees,
        // and with NO volatile content the whole system IS the head.
        let b = ContextBudget::default();
        let plan = plan_wire_request(
            "instructions",
            "",
            &[],
            "",
            &ledger_with(TaskLedger::default(), "g"),
            "",
            &[],
            &[],
            "",
            &b,
        )
        .unwrap();
        assert_eq!(plan.cacheable_prefix_len, plan.system.len());
        assert_eq!(plan.cacheable_prefix().unwrap(), plan.system.as_str());
        // With evidence the head is a strict prefix ending before it.
        let plan2 = plan_wire_request(
            "instructions",
            "",
            &[],
            "",
            &ledger_with(TaskLedger::default(), "g"),
            "",
            &[],
            &evidence_set(&[("src/a.rs", 1.0)]),
            "err",
            &b,
        )
        .unwrap();
        assert!(plan2.cacheable_prefix_len < plan2.system.len());
        assert_eq!(
            &plan2.system[..plan2.cacheable_prefix_len],
            plan2.cacheable_prefix().unwrap()
        );
        assert!(
            plan2.system[plan2.cacheable_prefix_len..].starts_with("\n## Retrieved evidence"),
            "the volatile tail must begin exactly at the boundary"
        );
    }
}
