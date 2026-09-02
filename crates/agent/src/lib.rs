//! kilop-agent — the durable agent reasoning loop.
//!
//! Drives `kilop-session` with commands, consumes `kilop-provider` streams,
//! schedules tools through `kilop-scheduler`, and keeps context bounded via
//! `kilop-context`. Rules (from the architecture spec):
//!
//! - **No provider-name conditionals.** Behavior comes from
//!   `ModelCapabilities`; provider quirks stay inside adapters.
//! - **State-aware continuation.** A provider stream that dies mid-tool never
//!   replays the turn; the journal determines the continuation point.
//! - **Repair once, never five times.** Malformed tool JSON gets one
//!   deterministic repair pass; repeated identical failures trip the loop
//!   detector and stop the turn.
//! - **Bounded context before sending.** Budget enforced by the assembler;
//!   compaction triggers proactively at the configured usage fraction.

pub mod loop_detect;
pub mod runtime;
pub mod tool;
pub mod tool_json;

pub use loop_detect::LoopDetector;
pub use runtime::{
    AgentCard, AgentDeps, AgentRuntime, EvidenceProvider, EvidenceQuery, NoEvidence,
    PermissionRequester, ToolArtifactSink, TurnOutcome,
};
pub use tool::{RecoveryHint, Tool, ToolOutcome, ToolRegistry, ToolRunCtx};
pub use tool_json::{parse_tool_calls, repair_json, ToolCallMode};

/// The agent may never match on provider names (Commandment 4). This test
/// locks that invariant structurally across the whole crate.
#[cfg(test)]
mod no_provider_switching {
    #[test]
    fn agent_source_has_no_provider_name_conditionals() {
        // Scan production sources only (skip test modules, whose own
        // assertions necessarily mention the forbidden literals).
        let mut sources = String::new();
        for file in [
            "lib.rs",
            "runtime.rs",
            "tool.rs",
            "tool_json.rs",
            "loop_detect.rs",
        ] {
            let path = format!("{}/src/{file}", env!("CARGO_MANIFEST_DIR"));
            let text = std::fs::read_to_string(&path).unwrap_or_default();
            let text = strip_test_modules(&text);
            sources.push_str(&text);
        }
        for needle in [
            "if provider ==",
            "match provider",
            "provider == \"deepseek\"",
            "provider == \"ollama\"",
            "provider == \"openai\"",
        ] {
            assert!(
                !sources.contains(needle),
                "agent source must not contain {needle:?}"
            );
        }
    }

    fn strip_test_modules(src: &str) -> String {
        // Remove #[cfg(test)] blocks so the invariant test cannot see its
        // own literals.
        let mut out = String::new();
        let mut rest = src;
        while let Some(idx) = rest.find("#[cfg(test)]") {
            out.push_str(&rest[..idx]);
            rest = &rest[idx + "#[cfg(test)]".len()..];
            // Skip to the closing brace of the mod at depth 0.
            let mut depth = 0i32;
            let mut consumed = 0usize;
            let mut found = false;
            for (i, c) in rest.char_indices() {
                match c {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            consumed = i + 1;
                            found = true;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            if found {
                rest = &rest[consumed..];
            }
        }
        out.push_str(rest);
        out
    }
}
