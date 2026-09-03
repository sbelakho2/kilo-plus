//! The strict translation layer between the v7.5.6 wire compatibility
//! surface (subset) (`super::wire`) and the internal domain model
//! (`faktor-core` errors, `super::{Message, Part}` which the session runtime
//! persists, and session rows).
//!
//! Directional policy (documented, never panics):
//!
//! - `wire → internal` (incoming): every `WirePart` variant maps to a
//!   `Part`. Text/Reasoning map 1:1. `Tool` maps to a `ToolCall` unless its
//!   state is `completed`/`error`, in which case it maps to a `ToolResult`
//!   built from `output` (the wire keeps the pair-by-`callID` join: the
//!   tool's `name`/`input` live on the original call part). `Subtask`,
//!   `Retry` and `Compaction` map to a `Summary`. `File`, `StepStart`,
//!   `StepFinish`, `Snapshot`, `Patch` and `Agent` have no internal
//!   equivalent; they map to a `Text` part carrying a `[marker: ...]` prefix
//!   so the content survives verbatim. Malformed (oversized) input is a
//!   loud `ErrorKind::Oversized`, never a panic.
//! - `internal → wire` (outgoing): `Text`/`Reasoning` map 1:1, `ToolCall`
//!   maps to `Tool` (state carried, `output` null), `ToolResult` maps to
//!   `Tool` with `state: "completed"` and the body as `output` (the call's
//!   `name`/`input` are not recoverable from a result — they degrade to
//!   `"unknown"`/`null`; clients join by `callID`), and `Summary` maps to
//!   `Subtask`. This direction is infallible: internal parts are already
//!   bounded by the runtime.
//!
//! Ids: internal `u64` ids ↔ wire strings via `id.to_string()` /
//! `wire_id_to_u64`, which accepts only non-empty all-digit non-zero values
//! and rejects everything else (negative, `+`, whitespace, overflow, `0`).
//!
//! Create/prompt args: `SessionCreateRequest` collapses to
//! `(provider, model, workspace, title)` and `MessageSendRequest` collapses
//! to `(prompt, files)` with hard bounds (prompt text is the concatenation
//! of `Text`/`File` parts truncated to the first 512KiB; at most 100 file
//! paths, each at most 4096 bytes).

use faktor_core::error::{Error, ErrorKind};

use super::wire::*;
use super::{Message, Part, ToolResultBody};

/// A text part is bounded to this many bytes in either direction.
pub const MAX_MAPPER_PART_BYTES: usize = 4 << 20;
/// The prompt text is the concatenation of text-bearing parts, bounded.
pub const MAX_MAPPER_PROMPT_BYTES: usize = 512 * 1024;
/// At most this many file paths may ride one message send.
pub const MAX_MAPPER_FILES: usize = 100;
/// One attached file path bound.
pub const MAX_MAPPER_FILE_PATH_BYTES: usize = 4096;
/// Provider/model ids and the workspace root bound.
pub const MAX_MAPPER_ID_BYTES: usize = 256;
/// Session title bound.
pub const MAX_MAPPER_TITLE_BYTES: usize = 4096;

fn malformed(msg: impl Into<String>) -> Error {
    Error::new(ErrorKind::Malformed, msg)
}

fn oversized(msg: impl Into<String>) -> Error {
    Error::new(ErrorKind::Oversized, msg)
}

fn bounded(s: &str, max: usize) -> Result<(), Error> {
    if s.len() > max {
        return Err(oversized(format!(
            "value of {} bytes exceeds bound {max}",
            s.len()
        )));
    }
    Ok(())
}

/// Parse a wire id (string) into the internal `u64`. Only non-empty,
/// all-digit, non-zero values are accepted; anything else (empty, `0`,
/// negative, `+1`, `1.5`, whitespace, overflow) is `Malformed`.
pub fn wire_id_to_u64(id: &str) -> Result<u64, Error> {
    if id.is_empty() || !id.bytes().all(|b| b.is_ascii_digit()) {
        return Err(malformed(format!("malformed id {id:?}")));
    }
    let raw: u64 = id
        .parse()
        .map_err(|_| malformed(format!("malformed id {id:?}")))?;
    if raw == 0 {
        return Err(malformed("id cannot be 0"));
    }
    Ok(raw)
}

/// The wire string for an internal `u64` id.
pub fn u64_to_wire_id(id: u64) -> String {
    id.to_string()
}

/// `WirePart → internal Part`. Oversized payloads are a loud error; nothing
/// panics.
pub fn wire_part_to_internal(w: &WirePart) -> Result<Part, Error> {
    Ok(match w {
        WirePart::Text { text } => {
            bounded(text, MAX_MAPPER_PART_BYTES)?;
            Part::Text { text: text.clone() }
        }
        WirePart::Reasoning { text } => {
            bounded(text, MAX_MAPPER_PART_BYTES)?;
            Part::Reasoning { text: text.clone() }
        }
        WirePart::Subtask { label, note } => {
            let text = match (label, note) {
                (Some(l), Some(n)) => format!("{l}: {n}"),
                (Some(l), None) => l.clone(),
                (None, Some(n)) => n.clone(),
                (None, None) => "subtask".to_string(),
            };
            bounded(&text, MAX_MAPPER_PART_BYTES)?;
            Part::Summary { text }
        }
        WirePart::File {
            path,
            content,
            mode: _,
        } => {
            let mut text = format!("[file: {path}]");
            if let Some(c) = content {
                text.push('\n');
                text.push_str(c);
            }
            bounded(&text, MAX_MAPPER_PART_BYTES)?;
            Part::Text { text }
        }
        WirePart::Tool {
            call_id,
            name,
            input,
            state,
            output,
        } => match state.as_deref() {
            Some("completed") | Some("error") => {
                let out = output.as_ref().unwrap_or(&serde_json::Value::Null);
                let excerpt = out
                    .get("excerpt")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| {
                        out.as_str()
                            .map(String::from)
                            .unwrap_or_else(|| "tool completed".to_string())
                    });
                bounded(&excerpt, MAX_MAPPER_PART_BYTES)?;
                Part::ToolResult {
                    tool_call_id: call_id.clone(),
                    result: ToolResultBody {
                        excerpt,
                        exit_code: out
                            .get("exit_code")
                            .and_then(|v| if v.is_null() { None } else { v.as_i64() })
                            .map(|v| v as i32),
                        artifact: out
                            .get("artifact")
                            .and_then(|v| v.as_str())
                            .map(String::from),
                        slice_hint: out
                            .get("slice_hint")
                            .and_then(|v| v.as_str())
                            .map(String::from),
                    },
                }
            }
            _ => {
                let state = state.clone().unwrap_or_else(|| "running".to_string());
                bounded(&state, MAX_MAPPER_PART_BYTES)?;
                if serde_json::to_vec(input)
                    .map(|b| b.len())
                    .unwrap_or(usize::MAX)
                    > MAX_MAPPER_PART_BYTES
                {
                    return Err(oversized(format!(
                        "tool input exceeds bound {MAX_MAPPER_PART_BYTES}"
                    )));
                }
                Part::ToolCall {
                    tool_call_id: call_id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                    state,
                }
            }
        },
        WirePart::StepStart { id, title } => {
            let text = format!(
                "[step start: {}]",
                title.as_deref().or(id.as_deref()).unwrap_or("unknown")
            );
            bounded(&text, MAX_MAPPER_PART_BYTES)?;
            Part::Text { text }
        }
        WirePart::StepFinish { id, title, outcome } => {
            let text = format!(
                "[step finish: {}: {}]",
                title.as_deref().or(id.as_deref()).unwrap_or("unknown"),
                outcome.as_deref().unwrap_or("done")
            );
            bounded(&text, MAX_MAPPER_PART_BYTES)?;
            Part::Text { text }
        }
        WirePart::Snapshot {
            session_id,
            message_id,
        } => {
            let text = format!(
                "[snapshot: session {} message {}]",
                session_id.as_deref().unwrap_or("?"),
                message_id.as_deref().unwrap_or("?")
            );
            bounded(&text, MAX_MAPPER_PART_BYTES)?;
            Part::Text { text }
        }
        WirePart::Patch {
            session_id,
            message_id,
            path,
            diff,
        } => {
            let mut text = format!(
                "[patch: session {} message {} path {}]",
                session_id.as_deref().unwrap_or("?"),
                message_id.as_deref().unwrap_or("?"),
                path.as_deref().unwrap_or("?")
            );
            if let Some(d) = diff {
                text.push('\n');
                text.push_str(d);
            }
            bounded(&text, MAX_MAPPER_PART_BYTES)?;
            Part::Text { text }
        }
        WirePart::Agent { id, name, state } => {
            let text = format!(
                "[agent: {} state {}]",
                name.as_deref().or(id.as_deref()).unwrap_or("?"),
                state.as_deref().unwrap_or("?")
            );
            bounded(&text, MAX_MAPPER_PART_BYTES)?;
            Part::Text { text }
        }
        WirePart::Retry { reason } => {
            let text = reason
                .clone()
                .unwrap_or_else(|| "retry requested".to_string());
            bounded(&text, MAX_MAPPER_PART_BYTES)?;
            Part::Summary { text }
        }
        WirePart::Compaction {
            before_tokens,
            after_tokens,
        } => {
            let text = format!(
                "compaction: before {} after {}",
                before_tokens.unwrap_or(0),
                after_tokens.unwrap_or(0)
            );
            Part::Summary { text }
        }
    })
}

/// `internal Part → WirePart`. Infallible: internal parts are bounded by the
/// runtime, so no size check is needed. `ToolResult` cannot recover the
/// original call's `name`/`input` (documented degradation).
pub fn internal_part_to_wire(p: &Part) -> WirePart {
    match p {
        Part::Text { text } => WirePart::Text { text: text.clone() },
        Part::Reasoning { text } => WirePart::Reasoning { text: text.clone() },
        Part::ToolCall {
            tool_call_id,
            name,
            input,
            state,
        } => WirePart::Tool {
            call_id: tool_call_id.clone(),
            name: name.clone(),
            input: input.clone(),
            state: Some(state.clone()),
            output: None,
        },
        Part::ToolResult {
            tool_call_id,
            result,
        } => WirePart::Tool {
            call_id: tool_call_id.clone(),
            name: "unknown".to_string(),
            input: serde_json::Value::Null,
            state: Some("completed".to_string()),
            output: Some(serde_json::json!({
                "excerpt": result.excerpt,
                "exit_code": result.exit_code,
                "artifact": result.artifact,
                "slice_hint": result.slice_hint,
            })),
        },
        Part::Summary { text } => WirePart::Subtask {
            label: Some(text.clone()),
            note: None,
        },
    }
}

/// `WireMessage → internal Message`. Wire ids must parse as strict internal
/// ids; the role must be one of the frozen set. The internal `seq` is not
/// carried on the wire (documented: the wire message omits `seq`; paging
/// cursors are sequences on the server side), so it starts at 0 and the
/// caller assigns the durable sequence when persisting.
pub fn wire_message_to_internal(w: &WireMessage) -> Result<Message, Error> {
    let session_id = wire_id_to_u64(&w.session_id)?.to_string();
    let id = wire_id_to_u64(&w.message_id)?.to_string();
    if !matches!(w.role.as_str(), "user" | "assistant" | "system") {
        return Err(malformed(format!("invalid role {:?}", w.role)));
    }
    let mut parts = Vec::with_capacity(w.parts.len());
    for p in &w.parts {
        parts.push(wire_part_to_internal(p)?);
    }
    Ok(Message {
        id,
        role: w.role.clone(),
        session_id,
        seq: 0,
        created_ms: w.created_ms,
        parts,
    })
}

/// `internal Message → WireMessage`. The internal seq does not exist on the
/// wire (documented); `provider_id`/`model_id` are filled by the caller from
/// the session row when known.
pub fn internal_message_to_wire(m: &Message) -> Result<WireMessage, Error> {
    // Defense in depth: internal ids are numeric by construction.
    let _ = wire_id_to_u64(&m.id)?;
    let _ = wire_id_to_u64(&m.session_id)?;
    let mut parts = Vec::with_capacity(m.parts.len());
    for p in &m.parts {
        parts.push(internal_part_to_wire(p));
    }
    Ok(WireMessage {
        session_id: m.session_id.clone(),
        message_id: m.id.clone(),
        role: m.role.clone(),
        parts,
        created_ms: m.created_ms,
        provider_id: None,
        model_id: None,
    })
}

/// `internal Message → WireMessageEntry`, the frozen page/send shape
/// `{info: Message, parts: Part[]}`.
///
/// Wire identity rule (documented): `info.messageID` is the message's
/// durable SEQUENCE — the same identity the revert/unrevert/diff surfaces
/// consume (`message_created_ms(session, seq)`). On a single-session store
/// the sequence equals the row id, so the two conventions coincide there.
/// `provider_id`/`model_id` are filled by the caller from the session row.
pub fn internal_message_to_wire_entry(m: &Message) -> Result<WireMessageEntry, Error> {
    // Defense in depth: internal ids are numeric by construction.
    let _ = wire_id_to_u64(&m.id)?;
    let _ = wire_id_to_u64(&m.session_id)?;
    let mut parts = Vec::with_capacity(m.parts.len());
    for p in &m.parts {
        parts.push(internal_part_to_wire(p));
    }
    Ok(WireMessageEntry {
        info: WireMessageInfo {
            session_id: m.session_id.clone(),
            message_id: m.seq.to_string(),
            role: m.role.clone(),
            created_ms: m.created_ms,
            provider_id: None,
            model_id: None,
        },
        parts,
    })
}

/// The internal create-session arguments derived from a wire request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateArgs {
    pub provider: String,
    pub model: String,
    pub workspace: String,
    pub title: String,
}

/// `SessionCreateRequest → (provider, model, workspace, title)`.
/// provider = `model.providerID` (or `"default"`), model = `model.id`,
/// workspace = the `x-faktor-directory` header, else `workspaceID`, else `"."`,
/// title = `title` or `"New session"`.
pub fn create_args(
    req: &SessionCreateRequest,
    directory_header: Option<&str>,
) -> Result<CreateArgs, Error> {
    let provider = req
        .model
        .as_ref()
        .map(|m| m.provider_id.clone())
        .filter(|p| !p.is_empty())
        .unwrap_or_else(|| "default".to_string());
    bounded(&provider, MAX_MAPPER_ID_BYTES)?;
    // The real v7.5.6 contract: `model` is OPTIONAL — the daemon picks its
    // configured default when absent (mirrors `provider` defaulting above).
    let model = req
        .model
        .as_ref()
        .map(|m| m.id.clone())
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| "default".to_string());
    bounded(&model, MAX_MAPPER_ID_BYTES)?;
    let workspace = directory_header
        .map(str::trim)
        .filter(|w| !w.is_empty())
        .map(String::from)
        .or_else(|| {
            req.workspace_id
                .as_ref()
                .map(|w| w.trim().to_string())
                .filter(|w| !w.is_empty())
        })
        .unwrap_or_else(|| ".".to_string());
    bounded(&workspace, MAX_MAPPER_ID_BYTES)?;
    let title = req
        .title
        .as_ref()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .unwrap_or_else(|| "New session".to_string());
    bounded(&title, MAX_MAPPER_TITLE_BYTES)?;
    Ok(CreateArgs {
        provider,
        model,
        workspace,
        title,
    })
}

/// The internal prompt arguments derived from a wire message send.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptArgs {
    pub prompt: String,
    pub files: Vec<String>,
}

/// `MessageSendRequest → (prompt, files)`. The prompt is the concatenation
/// of `Text` parts and `File` part contents, truncated to the first
/// `MAX_MAPPER_PROMPT_BYTES`; `files` is the `File` part paths, bounded to
/// `MAX_MAPPER_FILES` entries of at most `MAX_MAPPER_FILE_PATH_BYTES` each.
pub fn prompt_args(req: &MessageSendRequest) -> Result<PromptArgs, Error> {
    let mut prompt = String::new();
    let mut files = Vec::new();
    for part in &req.parts {
        match part {
            WirePart::Text { text } => push_bounded(&mut prompt, text, MAX_MAPPER_PROMPT_BYTES),
            WirePart::File { path, content, .. } => {
                if files.len() >= MAX_MAPPER_FILES {
                    return Err(oversized(format!(
                        "more than {MAX_MAPPER_FILES} file parts"
                    )));
                }
                bounded(path, MAX_MAPPER_FILE_PATH_BYTES)?;
                files.push(path.clone());
                if let Some(c) = content {
                    push_bounded(&mut prompt, c, MAX_MAPPER_PROMPT_BYTES);
                }
            }
            // Control-plane parts (subtask, tool, step, snapshot, patch,
            // agent, retry, compaction) carry no prompt text; the turn is
            // driven by the text/file parts only.
            _ => {}
        }
    }
    Ok(PromptArgs { prompt, files })
}

/// Append `s`, never exceeding `max` bytes (the first `max` bytes win).
fn push_bounded(out: &mut String, s: &str, max: usize) {
    if out.len() >= max {
        return;
    }
    let room = max - out.len();
    let take = s.len().min(room);
    let mut end = take;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    out.push_str(&s[..end]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn part(name: &str) -> WirePart {
        match name {
            "text" => WirePart::Text { text: "hi".into() },
            "subtask" => WirePart::Subtask {
                label: Some("l".into()),
                note: Some("n".into()),
            },
            "reasoning" => WirePart::Reasoning { text: "r".into() },
            "file" => WirePart::File {
                path: "a.rs".into(),
                content: Some("fn".into()),
                mode: Some("edit".into()),
            },
            "tool-running" => WirePart::Tool {
                call_id: "c1".into(),
                name: "read_file".into(),
                input: serde_json::json!({"p": 1}),
                state: Some("running".into()),
                output: None,
            },
            "tool-completed" => WirePart::Tool {
                call_id: "c1".into(),
                name: "read_file".into(),
                input: serde_json::json!({"p": 1}),
                state: Some("completed".into()),
                output: Some(serde_json::json!({
                    "excerpt": "1 | fn main",
                    "exit_code": 0,
                    "artifact": "artifact://abc",
                    "slice_hint": "artifact://abc?slice=0"
                })),
            },
            "tool-error" => WirePart::Tool {
                call_id: "c1".into(),
                name: "run".into(),
                input: serde_json::json!({}),
                state: Some("error".into()),
                output: Some(serde_json::json!({"excerpt": "exit 1", "exit_code": 1})),
            },
            "stepStart" => WirePart::StepStart {
                id: Some("s1".into()),
                title: Some("t".into()),
            },
            "stepFinish" => WirePart::StepFinish {
                id: Some("s1".into()),
                title: Some("t".into()),
                outcome: Some("ok".into()),
            },
            "snapshot" => WirePart::Snapshot {
                session_id: Some("1".into()),
                message_id: Some("2".into()),
            },
            "patch" => WirePart::Patch {
                session_id: None,
                message_id: None,
                path: Some("a.rs".into()),
                diff: Some("@@".into()),
            },
            "agent" => WirePart::Agent {
                id: Some("a1".into()),
                name: Some("n".into()),
                state: Some("idle".into()),
            },
            "retry" => WirePart::Retry {
                reason: Some("r".into()),
            },
            "compaction" => WirePart::Compaction {
                before_tokens: Some(100),
                after_tokens: Some(50),
            },
            other => panic!("unknown fixture part {other}"),
        }
    }

    #[test]
    fn every_wire_part_variant_roundtrips_exactly_or_documented() {
        // Exact roundtrips: wire → internal → wire reproduces the part.
        for name in ["text", "reasoning", "tool-running"] {
            let w = part(name);
            let internal = wire_part_to_internal(&w).unwrap();
            let back = internal_part_to_wire(&internal);
            assert_eq!(w, back, "roundtrip drift for {name}");
        }
        // A subtask without a note is exact; with a note the label/note pair
        // collapses into the summary text (documented).
        let w = WirePart::Subtask {
            label: Some("l".into()),
            note: None,
        };
        assert_eq!(
            internal_part_to_wire(&wire_part_to_internal(&w).unwrap()),
            w
        );
        let w = part("subtask");
        assert_eq!(
            internal_part_to_wire(&wire_part_to_internal(&w).unwrap()),
            WirePart::Subtask {
                label: Some("l: n".into()),
                note: None
            },
            "note collapses into the label on the way back (documented)"
        );
        // Control-plane parts map to marker text; the content survives in
        // the marker (documented; the internal domain has no such parts).
        let cases: Vec<(&str, WirePart)> = vec![
            (
                "file",
                WirePart::Text {
                    text: "[file: a.rs]\nfn".into(),
                },
            ),
            (
                "stepStart",
                WirePart::Text {
                    text: "[step start: t]".into(),
                },
            ),
            (
                "stepFinish",
                WirePart::Text {
                    text: "[step finish: t: ok]".into(),
                },
            ),
            (
                "snapshot",
                WirePart::Text {
                    text: "[snapshot: session 1 message 2]".into(),
                },
            ),
            (
                "patch",
                WirePart::Text {
                    text: "[patch: session ? message ? path a.rs]\n@@".into(),
                },
            ),
            (
                "agent",
                WirePart::Text {
                    text: "[agent: n state idle]".into(),
                },
            ),
        ];
        for (name, expected) in cases {
            let internal = wire_part_to_internal(&part(name)).unwrap();
            assert_eq!(
                internal_part_to_wire(&internal),
                expected,
                "documented marker mapping drift for {name}"
            );
        }
        // Retry/compaction land as summaries, which surface as subtasks
        // (documented).
        assert_eq!(
            internal_part_to_wire(&wire_part_to_internal(&part("retry")).unwrap()),
            WirePart::Subtask {
                label: Some("r".into()),
                note: None
            }
        );
        assert_eq!(
            internal_part_to_wire(&wire_part_to_internal(&part("compaction")).unwrap()),
            WirePart::Subtask {
                label: Some("compaction: before 100 after 50".into()),
                note: None
            }
        );
        // Documented lossy: a completed tool loses name/input on the way back
        // (the result part only carries callID + output; clients join by
        // callID).
        let w = part("tool-completed");
        let internal = wire_part_to_internal(&w).unwrap();
        match internal {
            Part::ToolResult { .. } => {}
            other => panic!("completed tool must become ToolResult, got {other:?}"),
        }
        let back = internal_part_to_wire(&internal);
        match back {
            WirePart::Tool {
                call_id,
                name,
                input,
                state,
                output,
            } => {
                assert_eq!(call_id, "c1");
                assert_eq!(name, "unknown");
                assert!(input.is_null());
                assert_eq!(state.as_deref(), Some("completed"));
                let out = output.unwrap();
                assert_eq!(out["excerpt"], "1 | fn main");
                assert_eq!(out["exit_code"], 0);
                assert_eq!(out["artifact"], "artifact://abc");
                assert_eq!(out["slice_hint"], "artifact://abc?slice=0");
            }
            other => panic!("expected Tool, got {other:?}"),
        }
        // The error state maps to a ToolResult whose exit_code survives.
        let internal = wire_part_to_internal(&part("tool-error")).unwrap();
        match internal {
            Part::ToolResult { result, .. } => {
                assert_eq!(result.excerpt, "exit 1");
                assert_eq!(result.exit_code, Some(1));
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
        // Internal Summary → wire Subtask (documented direction).
        let w = internal_part_to_wire(&Part::Summary { text: "sum".into() });
        assert_eq!(
            w,
            WirePart::Subtask {
                label: Some("sum".into()),
                note: None
            }
        );
    }

    #[test]
    fn oversized_wire_parts_are_rejected_not_truncated() {
        let big = "x".repeat(MAX_MAPPER_PART_BYTES + 1);
        for w in [
            WirePart::Text { text: big.clone() },
            WirePart::Reasoning { text: big.clone() },
            WirePart::File {
                path: "a".into(),
                content: Some(big.clone()),
                mode: None,
            },
            WirePart::Tool {
                call_id: "c".into(),
                name: "n".into(),
                input: serde_json::Value::Null,
                state: Some("completed".into()),
                output: Some(serde_json::json!({"excerpt": big})),
            },
        ] {
            let err = wire_part_to_internal(&w).unwrap_err();
            assert_eq!(err.kind, ErrorKind::Oversized, "{w:?}");
        }
        // A completed tool with an oversized excerpt via a scalar output.
        let err = wire_part_to_internal(&WirePart::Tool {
            call_id: "c".into(),
            name: "n".into(),
            input: serde_json::Value::Null,
            state: Some("error".into()),
            output: Some(serde_json::Value::String(big)),
        })
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Oversized);
    }

    #[test]
    fn id_roundtrip_and_malicious_ids_rejected() {
        for raw in [1u64, 7, 1001, u64::MAX] {
            let s = u64_to_wire_id(raw);
            assert_eq!(wire_id_to_u64(&s).unwrap(), raw);
        }
        for evil in [
            "",
            "0",
            "-1",
            "+1",
            "1.5",
            "abc",
            "1 2",
            " 1",
            "1 ",
            "1_0",
            "1e3",
            "18446744073709551616", // u64::MAX + 1
            "\u{200b}1",            // zero-width space
        ] {
            let err = wire_id_to_u64(evil).unwrap_err();
            assert_eq!(
                err.kind,
                ErrorKind::Malformed,
                "id {evil:?} must be rejected"
            );
        }
        // All-digit but non-canonical (leading zero) is accepted: it is still
        // a well-formed numeric id.
        assert_eq!(wire_id_to_u64("007").unwrap(), 7);
    }

    fn wire_message(session: &str, id: &str, role: &str) -> WireMessage {
        WireMessage {
            session_id: session.into(),
            message_id: id.into(),
            role: role.into(),
            parts: vec![WirePart::Text { text: "hi".into() }],
            created_ms: 5,
            provider_id: None,
            model_id: None,
        }
    }

    #[test]
    fn wire_message_maps_to_internal_and_back() {
        let w = wire_message("42", "7", "user");
        let m = wire_message_to_internal(&w).unwrap();
        assert_eq!(m.session_id, "42");
        assert_eq!(m.id, "7");
        assert_eq!(m.role, "user");
        assert_eq!(m.parts.len(), 1);
        let back = internal_message_to_wire(&m).unwrap();
        assert_eq!(back, w, "message roundtrip must be exact");
        // The internal seq does not exist on the wire; provider/model are
        // filled by the caller from the session row.
        assert_eq!(m.seq, 0);
    }

    #[test]
    fn internal_message_maps_to_wire_entry_with_seq_identity() {
        let m = Message {
            id: "99".into(),
            role: "assistant".into(),
            session_id: "42".into(),
            seq: 5,
            created_ms: 7,
            parts: vec![
                Part::Text { text: "hi".into() },
                Part::ToolResult {
                    tool_call_id: "c1".into(),
                    result: ToolResultBody {
                        excerpt: "1 | fn main".into(),
                        exit_code: Some(0),
                        artifact: None,
                        slice_hint: None,
                    },
                },
            ],
        };
        let e = internal_message_to_wire_entry(&m).unwrap();
        // Top level is exactly {info, parts}: parts live OUTSIDE info.
        assert_eq!(
            serde_json::to_value(&e).unwrap()["parts"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(e.info.session_id, "42");
        // The wire messageID is the durable SEQUENCE (documented identity).
        assert_eq!(e.info.message_id, "5");
        assert_eq!(e.info.role, "assistant");
        assert_eq!(e.info.created_ms, 7);
        assert_eq!(e.info.provider_id, None);
        assert_eq!(e.parts.len(), 2);
        // Malicious internal ids are rejected loudly.
        let evil = Message {
            id: "abc".into(),
            ..m.clone()
        };
        assert!(internal_message_to_wire_entry(&evil).is_err());
        let evil = Message {
            session_id: "0".into(),
            ..m
        };
        assert!(internal_message_to_wire_entry(&evil).is_err());
    }

    #[test]
    fn malicious_message_ids_and_roles_rejected() {
        for evil in ["0", "-1", "abc", ""] {
            assert!(
                wire_message_to_internal(&wire_message(evil, "1", "user")).is_err(),
                "session id {evil:?} must be rejected"
            );
            assert!(
                wire_message_to_internal(&wire_message("1", evil, "user")).is_err(),
                "message id {evil:?} must be rejected"
            );
        }
        for role in ["root", "", "USER"] {
            assert!(
                wire_message_to_internal(&wire_message("1", "1", role)).is_err(),
                "role {role:?} must be rejected"
            );
        }
        // Unknown session ids on the internal side are rejected too
        // (defense in depth; internal messages are numeric by construction).
        assert!(internal_message_to_wire(&Message {
            id: "abc".into(),
            role: "user".into(),
            session_id: "1".into(),
            seq: 1,
            created_ms: 0,
            parts: vec![],
        })
        .is_err());
    }

    #[test]
    fn create_args_derive_provider_model_workspace_title() {
        // Full request: everything explicit.
        let req = SessionCreateRequest {
            parent_id: None,
            title: Some("Fix parser".into()),
            agent: None,
            model: Some(SessionModel {
                id: "qwen3.8".into(),
                provider_id: "ollama".into(),
                variant: None,
            }),
            metadata: None,
            permission: None,
            platform: None,
            workspace_id: Some("/home/u/proj".into()),
            sandbox_inheritance_token: None,
        };
        let a = create_args(&req, Some("/hdr/dir")).unwrap();
        assert_eq!(a.provider, "ollama");
        assert_eq!(a.model, "qwen3.8");
        assert_eq!(
            a.workspace, "/hdr/dir",
            "x-faktor-directory wins over workspaceID"
        );
        assert_eq!(a.title, "Fix parser");
        // workspaceID falls back when the header is absent/empty.
        let a = create_args(&req, None).unwrap();
        assert_eq!(a.workspace, "/home/u/proj");
        let a = create_args(&req, Some("   ")).unwrap();
        assert_eq!(a.workspace, "/home/u/proj", "blank header is ignored");
        // No model → both provider and model default (real v7.5.6 contract:
        // model is optional; the daemon picks its configured default).
        let bare = SessionCreateRequest {
            model: None,
            ..req.clone()
        };
        let a = create_args(&bare, None).unwrap();
        assert_eq!(a.provider, "default");
        assert_eq!(a.model, "default");
        let no_model_id = SessionCreateRequest {
            model: Some(SessionModel {
                id: "".into(),
                provider_id: "p".into(),
                variant: None,
            }),
            ..req.clone()
        };
        let a = create_args(&no_model_id, None).unwrap();
        assert_eq!(a.model, "default", "empty model id defaults too");
        // Defaults: title and workspace.
        let minimal = SessionCreateRequest {
            model: Some(SessionModel {
                id: "m".into(),
                provider_id: "".into(),
                variant: None,
            }),
            ..SessionCreateRequest {
                parent_id: None,
                title: None,
                agent: None,
                model: None,
                metadata: None,
                permission: None,
                platform: None,
                workspace_id: None,
                sandbox_inheritance_token: None,
            }
        };
        let a = create_args(&minimal, None).unwrap();
        assert_eq!(a.provider, "default");
        assert_eq!(a.workspace, ".");
        assert_eq!(a.title, "New session");
        // Bounds: oversized metadata is a loud 413-class error.
        let big = "x".repeat(MAX_MAPPER_ID_BYTES + 1);
        let big = SessionCreateRequest {
            workspace_id: Some(big),
            ..minimal.clone()
        };
        assert_eq!(
            create_args(&big, None).unwrap_err().kind,
            ErrorKind::Oversized
        );
        let big = "x".repeat(MAX_MAPPER_TITLE_BYTES + 1);
        let big = SessionCreateRequest {
            title: Some(big),
            ..minimal
        };
        assert_eq!(
            create_args(&big, None).unwrap_err().kind,
            ErrorKind::Oversized
        );
    }

    #[test]
    fn prompt_args_concatenate_text_and_file_parts_with_bounds() {
        let base = MessageSendRequest {
            message_id: None,
            model: MessageModel {
                provider_id: "p".into(),
                model_id: "m".into(),
            },
            agent: None,
            no_reply: None,
            tools: None,
            format: None,
            system: None,
            variant: None,
            snapshot_initialization: None,
            editor_context: None,
            parts: Vec::new(),
        };
        let req = MessageSendRequest {
            parts: vec![
                WirePart::Text {
                    text: "fix it".into(),
                },
                WirePart::File {
                    path: "a.rs".into(),
                    content: Some("fn main".into()),
                    mode: None,
                },
                WirePart::Reasoning {
                    text: "ignored".into(),
                },
                WirePart::Tool {
                    call_id: "c".into(),
                    name: "n".into(),
                    input: serde_json::Value::Null,
                    state: None,
                    output: None,
                },
            ],
            ..base.clone()
        };
        let a = prompt_args(&req).unwrap();
        assert_eq!(a.prompt, "fix itfn main", "text+file contents concatenate");
        assert_eq!(a.files, vec!["a.rs".to_string()]);
        // Empty parts → empty prompt (the endpoint rejects it with 400).
        assert_eq!(prompt_args(&base).unwrap().prompt, "");
        // More than 100 files → Oversized.
        let many: Vec<WirePart> = (0..=MAX_MAPPER_FILES)
            .map(|i| WirePart::File {
                path: format!("f{i}"),
                content: None,
                mode: None,
            })
            .collect();
        let req = MessageSendRequest {
            parts: many,
            ..base.clone()
        };
        let err = prompt_args(&req).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Oversized);
        // Oversized file path → Oversized.
        let req = MessageSendRequest {
            parts: vec![WirePart::File {
                path: "x".repeat(MAX_MAPPER_FILE_PATH_BYTES + 1),
                content: None,
                mode: None,
            }],
            ..base.clone()
        };
        assert_eq!(prompt_args(&req).unwrap_err().kind, ErrorKind::Oversized);
        // The prompt is truncated to the first 512KiB, never grown past it.
        let big = "x".repeat(MAX_MAPPER_PROMPT_BYTES + 1000);
        let req = MessageSendRequest {
            parts: vec![
                WirePart::Text { text: big.clone() },
                WirePart::Text {
                    text: "tail".into(),
                },
            ],
            ..base.clone()
        };
        let a = prompt_args(&req).unwrap();
        assert_eq!(a.prompt.len(), MAX_MAPPER_PROMPT_BYTES);
        assert!(!a.prompt.ends_with("tail"), "beyond the bound is dropped");
        // The bound never splits a UTF-8 character.
        let req = MessageSendRequest {
            parts: vec![WirePart::Text {
                text: "x".repeat(MAX_MAPPER_PROMPT_BYTES - 1) + "é",
            }],
            ..base
        };
        let a = prompt_args(&req).unwrap();
        assert!(a.prompt.is_char_boundary(a.prompt.len()));
    }

    #[test]
    fn non_text_parts_map_to_markers_without_panicking() {
        // Every control-plane variant maps deterministically, never panics,
        // and the marker text survives the return trip verbatim.
        let cases: Vec<(&str, &str)> = vec![
            ("file", "[file: a.rs]"),
            ("stepStart", "[step start:"),
            ("stepFinish", "[step finish:"),
            ("snapshot", "[snapshot:"),
            ("patch", "[patch:"),
            ("agent", "[agent:"),
        ];
        for (name, prefix) in cases {
            let internal = wire_part_to_internal(&part(name)).unwrap();
            match &internal {
                Part::Text { text } => {
                    assert!(text.starts_with(prefix), "{name}: {text}");
                    let back = internal_part_to_wire(&internal);
                    match back {
                        WirePart::Text { text } => assert_eq!(text, internal_text(&internal)),
                        other => panic!("{name} must return as text, got {other:?}"),
                    }
                }
                other => panic!("{name} must map to a marker text part, got {other:?}"),
            }
        }
        // The markers carry the content verbatim.
        let internal = wire_part_to_internal(&part("file")).unwrap();
        match internal {
            Part::Text { text } => {
                assert!(text.starts_with("[file: a.rs]"), "{text}");
                assert!(text.contains("fn"));
            }
            other => panic!("file must map to a marker text part, got {other:?}"),
        }
        let internal = wire_part_to_internal(&part("patch")).unwrap();
        match internal {
            Part::Text { text } => {
                assert!(text.starts_with("[patch:"), "{text}");
                assert!(text.ends_with("@@"));
            }
            other => panic!("patch must map to a marker text part, got {other:?}"),
        }
    }

    fn internal_text(p: &Part) -> String {
        match p {
            Part::Text { text } | Part::Reasoning { text } | Part::Summary { text } => text.clone(),
            other => panic!("expected text-bearing part, got {other:?}"),
        }
    }
}
