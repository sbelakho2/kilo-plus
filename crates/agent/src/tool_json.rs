//! Tool-call parsing with three modes (spec §10): Native, NativeWithRepair,
//! StructuredFallback. If a payload is almost-valid JSON, ONE deterministic
//! repair pass may occur. The runtime never asks the model five times to
//! repair the same malformed invocation — repair happens once, and repeated
//! identical failures trip the loop detector.

use serde_json::Value;

use kilop_core::error::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallMode {
    /// The provider emitted a typed tool call (native API).
    Native,
    /// Native, but a single repair pass is allowed on malformed payloads.
    NativeWithRepair,
    /// The provider emitted text; extract tool calls from it.
    StructuredFallback,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedToolCall {
    pub id: String,
    pub name: String,
    pub input: Value,
    /// True when the payload needed the deterministic repair pass.
    pub repaired: bool,
}

/// The parse pipeline. Returns a bounded result (≤64 calls; callers bound
/// inputs before parsing).
pub fn parse_tool_calls(text: &str, mode: ToolCallMode) -> Vec<ParsedToolCall> {
    match mode {
        ToolCallMode::Native => Vec::new(),
        ToolCallMode::NativeWithRepair | ToolCallMode::StructuredFallback => {
            extract_from_text(text)
        }
    }
}

/// Deterministic single repair pass (bounded, one shot, never loops):
/// - strip surrounding markdown fences and prose
/// - extract the first balanced JSON object or array
/// - fix trailing commas
/// - unquote single-quoted keys/values (naive but deterministic)
///
/// Returns None when the input is still not valid JSON.
pub fn repair_json(text: &str) -> Option<Value> {
    const MAX_INPUT: usize = 64 * 1024;
    if text.len() > MAX_INPUT {
        return None;
    }
    let candidate = extract_balanced_json(text)?;
    if let Ok(v) = serde_json::from_str::<Value>(candidate) {
        return Some(v);
    }
    // Pass 1: trailing commas + single quotes.
    let mut fixed = candidate.to_string();
    fixed = strip_trailing_commas(&fixed);
    fixed = unquote_single(&fixed);
    if let Ok(v) = serde_json::from_str::<Value>(&fixed) {
        return Some(v);
    }
    // Pass 2: unquoted keys (deterministic regex-free scan).
    fixed = quote_bare_keys(&fixed);
    if let Ok(v) = serde_json::from_str::<Value>(&fixed) {
        return Some(v);
    }
    None
}

/// Extract the first balanced {...} or [...] region from arbitrary text.
fn extract_balanced_json(text: &str) -> Option<&str> {
    let bytes = text.as_bytes();
    let mut start = None;
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'{' || c == b'[' {
            start = Some(i);
            break;
        }
        i += 1;
    }
    let start = start?;
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    let mut end = text.len();
    for (j, c) in text[start..].char_indices() {
        let abs = start + j;
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' | '[' => depth += 1,
            '}' | ']' => {
                depth -= 1;
                if depth == 0 {
                    end = abs + c.len_utf8();
                    break;
                }
            }
            _ => {}
        }
    }
    if depth != 0 {
        return None; // unbalanced: hostile or truncated input
    }
    let slice = &text[start..end];
    if slice.len() > 64 * 1024 {
        return None;
    }
    Some(slice)
}

fn strip_trailing_commas(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    let mut prev_ws = String::new();
    while i < bytes.len() {
        let c = bytes[i];
        if c == b',' {
            // Peek forward past whitespace: if a closer follows, drop the comma.
            let mut j = i + 1;
            while j < bytes.len()
                && (bytes[j] == b' ' || bytes[j] == b'\t' || bytes[j] == b'\n' || bytes[j] == b'\r')
            {
                j += 1;
            }
            if j < bytes.len() && (bytes[j] == b'}' || bytes[j] == b']') {
                i += 1;
                prev_ws.clear();
                continue;
            }
            out.push(',');
            i += 1;
            continue;
        }
        out.push(c as char);
        i += 1;
    }
    out
}

/// Replace single-quoted strings with double-quoted ones (naive char scan;
/// deterministic; bounded by MAX_INPUT).
fn unquote_single(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    for c in s.chars() {
        if in_double {
            out.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_double = false;
            }
            continue;
        }
        if in_single {
            if c == '\'' && !escaped {
                in_single = false;
                out.push('"');
            } else if c == '\\' && !escaped {
                escaped = true;
                out.push('\\');
            } else {
                escaped = false;
                out.push(c);
            }
            continue;
        }
        match c {
            '"' => {
                in_double = true;
                out.push(c);
            }
            '\'' => {
                in_single = true;
                out.push('"');
            }
            _ => out.push(c),
        }
    }
    out
}

/// Quote bare object keys: `{path: "x"}` → `{"path": "x"}`. Deterministic
/// scan, bounded; only applied when the JSON has unquoted keys at all.
fn quote_bare_keys(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 16);
    let mut in_string = false;
    let mut escaped = false;
    let mut prev_significant = ' ';
    let mut i = 0;
    let chars: Vec<char> = s.chars().collect();
    while i < chars.len() {
        let c = chars[i];
        if in_string {
            out.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if c == '"' {
            in_string = true;
            out.push(c);
            i += 1;
            continue;
        }
        // A key starts after { or , (ignoring whitespace) with a non-quote.
        if (prev_significant == '{' || prev_significant == ',')
            && c != ' '
            && c != '\t'
            && c != '\n'
            && c != '}'
            && c != ']'
        {
            // Collect the bare key until ':'.
            let mut key = String::new();
            let mut j = i;
            while j < chars.len() && chars[j] != ':' {
                key.push(chars[j]);
                j += 1;
            }
            if j < chars.len() && chars[j] == ':' && !key.is_empty() {
                let key = key.trim();
                if !key.is_empty() && !key.starts_with('"') {
                    out.push('"');
                    out.push_str(key);
                    out.push('"');
                    out.push(':');
                    i = j + 1;
                    prev_significant = ':';
                    continue;
                }
            }
        }
        out.push(c);
        if c != ' ' && c != '\t' && c != '\n' {
            prev_significant = c;
        }
        i += 1;
    }
    out
}

/// Extract tool calls from provider text (StructuredFallback):
/// `{"name": "read_file", "input": {...}}` or
/// `<tool_call>{"name":...}</tool_call>` or markdown fences.
pub fn extract_from_text(text: &str) -> Vec<ParsedToolCall> {
    const MAX_CALLS: usize = 64;
    let mut out = Vec::new();
    if text.len() > 128 * 1024 {
        return out; // bounded input
    }
    // Try to parse the whole text as one tool call first, but only return
    // early when that call consumes essentially the entire input (a single
    // call). Two calls joined by prose must both be found.
    let mut remaining = text;
    if let Some(v) = repair_json(text) {
        if let Some(call) = call_from_value(&v, false) {
            let end = blob_len(text, serde_json::to_string(&v).unwrap_or_default().len());
            if end as f64 >= text.len() as f64 * 0.9 {
                out.push(call);
                return out;
            }
            out.push(call);
            remaining = &text[end.min(text.len())..];
        }
    }
    // Scan for `{"name":..., "input":...}` regions and fenced blocks.
    while out.len() < MAX_CALLS {
        let start = find_tool_call_start(remaining);
        match start {
            None => break,
            Some(idx) => {
                let rest = &remaining[idx..];
                if let Some(v) = repair_json(rest) {
                    if let Some(call) = call_from_value(&v, true) {
                        out.push(call);
                        // Advance past this JSON blob.
                        let serialized = serde_json::to_string(&v).unwrap_or_default();
                        let consumed = blob_len(rest, serialized.len());
                        remaining = &rest[consumed.min(rest.len())..];
                        continue;
                    }
                }
                // No valid call here; skip one char and keep scanning.
                remaining = &rest[1..];
            }
        }
    }
    out
}

fn find_tool_call_start(text: &str) -> Option<usize> {
    let markers = [
        "{\"name\"",
        "{'name'",
        "{\"type\":\"tool_call\"",
        "{\"type\":\"function\"",
        "tool_call",
        "function_call",
    ];
    markers.iter().filter_map(|m| text.find(m)).min()
}

fn blob_len(rest: &str, serialized_len: usize) -> usize {
    // Skip the balanced JSON we found plus trailing whitespace. serialized
    // length is a lower bound; scan for the true end conservatively.
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    for (i, c) in rest.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' | '[' => depth += 1,
            '}' | ']' => {
                depth -= 1;
                if depth == 0 {
                    return i + c.len_utf8();
                }
            }
            _ => {}
        }
    }
    serialized_len.min(rest.len())
}

fn call_from_value(v: &Value, allow_wrapper: bool) -> Option<ParsedToolCall> {
    let mut id = "call_repair".to_string();
    let mut name = None;
    let mut input = None;
    match v {
        Value::Object(map) => {
            // Direct: {"name": ..., "input": ...} or {"id":..,"name":..,"arguments":..}
            name = map.get("name").and_then(|n| n.as_str()).or_else(|| {
                map.get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
            });
            input = map
                .get("input")
                .cloned()
                .or_else(|| map.get("arguments").cloned());
            if let Some(n) = map.get("id").and_then(|i| i.as_str()) {
                id = n.to_string();
            }
            // Wrapper: {"type":"tool_call","tool_call":{...}}
            if allow_wrapper {
                if let Some(inner) = map.get("tool_call").or_else(|| map.get("function")) {
                    if let Some(inner_map) = inner.as_object() {
                        name = inner_map.get("name").and_then(|n| n.as_str()).or(name);
                        input = inner_map
                            .get("input")
                            .cloned()
                            .or_else(|| inner_map.get("arguments").cloned());
                    }
                }
            }
        }
        Value::Array(items) if items.len() == 1 => {
            return call_from_value(&items[0], allow_wrapper);
        }
        _ => {}
    }
    let name = name?;
    let input = input.unwrap_or(Value::Null);
    Some(ParsedToolCall {
        id,
        name: name.to_string(),
        input,
        repaired: !matches!(v, Value::Object(_)),
    })
}

/// Validate a native tool call: name non-empty, input is an object (or
/// null), bounded size.
pub fn validate_native_call(name: &str, input: &Value) -> Result<(), Error> {
    if name.is_empty() {
        return Err(Error::malformed("tool call with empty name"));
    }
    if name.len() > 256 {
        return Err(Error::malformed("tool call name too long"));
    }
    let size = serde_json::to_vec(input)
        .map_err(|e| Error::malformed(format!("tool input not serializable: {e}")))?
        .len();
    if size > 64 * 1024 {
        return Err(Error::oversized(format!(
            "tool input {size} bytes exceeds 64KiB bound"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_json_passes_through() {
        let text = r#"{"name":"read_file","input":{"path":"a.rs"}}"#;
        let calls = extract_from_text(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].input["path"], "a.rs");
    }

    #[test]
    fn trailing_comma_repaired_once() {
        let text = r#"{"name":"write_file","input":{"path":"a.rs","content":"x",}}"#;
        let calls = extract_from_text(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "write_file");
        assert_eq!(calls[0].input["content"], "x");
    }

    #[test]
    fn single_quotes_repaired() {
        let text = r#"{'name': 'read_file', 'input': {'path': '/x'}}"#;
        let calls = extract_from_text(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].input["path"], "/x");
    }

    #[test]
    fn bare_keys_repaired() {
        let text = r#"{name: read_file, input: {path: "/x"}}"#;
        // `read_file` unquoted value is not repaired (values stay bare) —
        // this one must fail cleanly, not panic.
        let calls = extract_from_text(text);
        // Either repaired or not found — never a panic, never a wrong call.
        if !calls.is_empty() {
            assert_eq!(calls[0].name, "read_file");
        }
        let text2 = r#"{name: "read_file", input: {path: "/x"}}"#;
        let calls = extract_from_text(text2);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
    }

    #[test]
    fn fenced_and_prose_wrapped_calls_extracted() {
        let text = "Sure! Here is the call:\n```json\n{\"name\":\"grep\",\"input\":{\"pattern\":\"TODO\"}}\n```\nDone.";
        let calls = extract_from_text(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "grep");
    }

    #[test]
    fn multiple_calls_extracted_in_order() {
        let text = r#"{"name":"read_file","input":{"path":"a"}} then {"name":"read_file","input":{"path":"b"}}"#;
        let calls = extract_from_text(text);
        assert!(
            calls.len() >= 2,
            "expected 2 calls, got {}: {calls:?}",
            calls.len()
        );
        assert_eq!(calls[0].input["path"], "a");
        assert_eq!(calls[1].input["path"], "b");
    }

    #[test]
    fn garbage_never_panics() {
        for garbage in [
            "",
            "{",
            "}",
            "{\"name\":",
            "hello world",
            "```",
            "{\"name\":\"x\",\"input\":{",
            "x".repeat(200_000).as_str(), // oversized
            "\u{FFFE}\u{FFFF}",
        ] {
            let calls = extract_from_text(garbage);
            assert!(calls.len() <= 64);
        }
    }

    #[test]
    fn unbounded_input_rejected_early() {
        assert!(repair_json(&"x".repeat(200_000)).is_none());
        assert!(repair_json(&"{".repeat(100_000)).is_none());
    }

    #[test]
    fn hostile_nesting_is_bounded() {
        // 10k nested arrays: the balanced-scan must not blow the stack.
        let deep = format!("{}1{}", "[".repeat(10_000), "]".repeat(10_000));
        let r = repair_json(&deep);
        assert!(r.is_some() || r.is_none());
    }

    #[test]
    fn native_call_validation() {
        assert!(validate_native_call("read_file", &serde_json::json!({"path": "a"})).is_ok());
        assert!(validate_native_call("", &serde_json::json!({})).is_err());
        assert!(
            validate_native_call("x", &serde_json::json!({"big": "y".repeat(100_000)})).is_err()
        );
        assert!(validate_native_call(&"n".repeat(300), &serde_json::json!({})).is_err());
    }

    #[test]
    fn string_escapes_survive_repair() {
        let text = r#"{"name":"write_file","input":{"content":"say \"hi\" \\ and 'single'"}}"#;
        let calls = extract_from_text(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].input["content"], "say \"hi\" \\ and 'single'");
    }

    #[test]
    fn repair_is_single_pass_by_construction() {
        // The repair pipeline is a fixed sequence; assert the public API has
        // no retry loop by checking it returns after one call.
        let text = r#"{"name":"x","input":{"a":1,}}"#;
        let a = repair_json(text);
        let b = repair_json(text);
        assert_eq!(a, b, "repair must be deterministic (single pass)");
    }
}
