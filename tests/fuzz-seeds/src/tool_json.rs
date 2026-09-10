//! Harness: tool-call JSON repair + parse (`faktor_agent::tool_json`).
//!
//! The model emits tool calls as text that is *almost* JSON; the runtime
//! allows ONE deterministic repair pass. Adversarial properties:
//!
//! - `repair_json` never panics, is deterministic (same input → same output)
//!   and any repaired value must reparse from its own serialization to the
//!   identical value;
//! - `parse_tool_calls` in every [`ToolCallMode`] never panics, is bounded
//!   (≤64 calls per the module contract) and deterministic;
//! - `extract_from_text` obeys the same bound and determinism;
//! - `validate_native_call` turns every parsed call into a typed accept or a
//!   typed rejection, never a panic (hostile names/sizes included).

use faktor_agent::tool_json::{
    extract_from_text, parse_tool_calls, repair_json, validate_native_call, ToolCallMode,
};

use super::Outcome;

/// Fuzz JSON repair + tool-call parsing.
pub fn harness_tool_json_repair_parse(bytes: &[u8]) -> Outcome {
    let text = String::from_utf8_lossy(bytes);

    let repaired = repair_json(&text);
    if repaired != repair_json(&text) {
        return Outcome::Violation("repair_json is not deterministic".into());
    }
    if let Some(value) = repaired {
        match serde_json::to_string(&value) {
            Ok(reencoded) => match serde_json::from_str::<serde_json::Value>(&reencoded) {
                Ok(again) if again == value => {}
                other => {
                    return Outcome::Violation(format!(
                        "a repaired JSON value does not reparse to itself: {other:?}"
                    ))
                }
            },
            Err(e) => {
                return Outcome::Violation(format!(
                    "a repaired JSON value failed to serialize: {e}"
                ))
            }
        }
    }

    for mode in [
        ToolCallMode::Native,
        ToolCallMode::NativeWithRepair,
        ToolCallMode::StructuredFallback,
    ] {
        let first = parse_tool_calls(&text, mode);
        if first != parse_tool_calls(&text, mode) {
            return Outcome::Violation(format!("{mode:?} tool parsing is not deterministic"));
        }
        if first.len() > 64 {
            return Outcome::Violation(format!(
                "{mode:?} returned {} calls over the 64-call bound",
                first.len()
            ));
        }
        for call in &first {
            if call.name.is_empty() {
                return Outcome::Violation(format!("{mode:?} returned an empty tool name"));
            }
            // Typed validation only: a rejection is fine, a panic is not.
            let _ = validate_native_call(&call.name, &call.input);
        }
    }

    let extracted = extract_from_text(&text);
    if extracted.len() > 64 {
        return Outcome::Violation(format!(
            "extract_from_text returned {} calls over the 64-call bound",
            extracted.len()
        ));
    }
    if extracted != extract_from_text(&text) {
        return Outcome::Violation("extract_from_text is not deterministic".into());
    }
    Outcome::Clean
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Lcg;

    /// Deterministic corpus: well-formed calls, fenced/single-quoted/
    /// trailing-comma near-misses, wrappers, oversized hostile blobs and
    /// raw bytes — 2000 iterations.
    #[test]
    fn seeded_pseudo_fuzz_tool_json_repair_parse_2000() {
        let mut lcg = Lcg::new(0x5EED_0000_0000_0008);
        let mut clean = 0;
        for i in 0..2000u64 {
            let text = match i % 7 {
                0 => format!(
                    "{{\"name\":\"read_file\",\"input\":{{\"path\":\"f{}.rs\"}}}}",
                    lcg.below(100)
                ),
                1 => format!(
                    "```json\n{{\"name\":\"write_file\",\"input\":{{\"path\":\"a.rs\",\"content\":\"{}\"}},}}\n```",
                    "λ".repeat(1 + lcg.below(5) as usize)
                ),
                2 => format!(
                    "<tool_call>{{'name': 'exec', 'arguments': {{'cmd': 'echo {}'}}}}</tool_call>",
                    lcg.next_u64() % 1000
                ),
                3 => format!(
                    "prose before {{name: read_file, input: {{path: {:?}}}}} prose after",
                    format!("/tmp/x{}", lcg.next_u64() % 100)
                ),
                4 => {
                    // Oversized: over the 64 KiB/128 KiB parser bounds.
                    "x".repeat(70_000 + lcg.below(70_000) as usize)
                }
                5 => {
                    // Raw bytes (usually invalid UTF-8).
                    let n = 1 + lcg.below(600) as usize;
                    String::from_utf8_lossy(
                        &(0..n).map(|_| lcg.next_u64() as u8).collect::<Vec<u8>>(),
                    )
                    .into_owned()
                }
                _ => format!(
                    "[{{\"id\":\"call_{}\",\"name\":\"tool_{}\",\"input\":[{},{}]}}]",
                    lcg.next_u64() % 100,
                    lcg.below(40),
                    lcg.next_u64(),
                    lcg.next_u64()
                ),
            };
            match harness_tool_json_repair_parse(text.as_bytes()) {
                Outcome::Clean => clean += 1,
                v => panic!("iteration {i}: {v}"),
            }
        }
        assert!(clean > 0);
    }

    #[test]
    fn tool_json_harness_is_deterministic() {
        let mut lcg = Lcg::new(0x7017);
        for _ in 0..50 {
            let n = 1 + lcg.below(500) as usize;
            let bytes: Vec<u8> = (0..n).map(|_| lcg.next_u64() as u8).collect();
            assert_eq!(
                harness_tool_json_repair_parse(&bytes),
                harness_tool_json_repair_parse(&bytes)
            );
        }
    }

    /// A repaired trailing-comma payload is still a valid call (sanity).
    #[test]
    fn repair_pipeline_finds_expected_call() {
        let text = r#"{"name":"read_file","input":{"path":"a.rs",}}"#;
        let calls = parse_tool_calls(text, ToolCallMode::NativeWithRepair);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].input["path"], "a.rs");
    }
}
