//! The v7.5.6 wire types. Field presence, naming, and nullability are frozen.
//! Request types use `deny_unknown_fields`: an unknown field is a protocol
//! drift signal and must fail loudly, not be ignored.

use kilop_core::event::{Event, EventKind};
use kilop_core::model::ModelCapabilities;
use serde::{Deserialize, Serialize};

/// The exact stdout line the daemon prints after binding (frozen client
/// contract; server-utils.ts in Kilo parses this, not a JSON handshake).
pub const STARTUP_LINE_TEMPLATE: &str = "kilo server listening on http://127.0.0.1:{port}";

/// The startup line for a bound port. Nothing else may be printed on stdout.
pub fn startup_line(port: u16) -> String {
    format!("kilo server listening on http://127.0.0.1:{port}")
}

/// Internal legacy handshake detail: the frontend no longer reads a JSON
/// handshake (the startup line above is the frozen contract), but the type
/// is kept for the old server tests and the compatibility token plumbing.
/// The daemon never prints this line on stdout.
pub const HANDSHAKE_PREFIX: &str = "KILO_PLUS_HANDSHAKE";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Handshake {
    pub version: String,
    pub protocol: String,
    pub pid: u64,
    pub auth_token: String,
    pub port: u16,
}

impl Handshake {
    pub fn to_line(&self) -> String {
        format!(
            "{HANDSHAKE_PREFIX} {}",
            serde_json::to_string(self).expect("handshake serializes")
        )
    }

    pub fn from_line(line: &str) -> Option<Self> {
        let line = line.trim();
        let json = line.strip_prefix(HANDSHAKE_PREFIX)?;
        serde_json::from_str(json.trim()).ok()
    }
}

// ------------------------------------------------------------------ REST API

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HelloRequest {
    pub client: String,
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HelloResponse {
    pub ok: bool,
    pub version: String,
    pub protocol: String,
    /// True when the auth header is required for other endpoints.
    pub auth_required: bool,
    /// Named providers the daemon knows about (never secrets).
    pub providers: Vec<String>,
}

// ------------------------------------------------------ SDK-shaped REST surface
// The generated v2 SDK surface of real v7.5.6. The old `/api/...` aliases stay
// wired, but these are the primary contract.

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HealthResponse {
    pub ok: bool,
    pub version: String,
    pub protocol: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SdkPromptRequest {
    pub session_id: String,
    pub prompt: String,
    /// File mentions the user attached.
    #[serde(default)]
    pub files: Vec<String>,
    /// Per-request model override hints; opaque to the daemon.
    pub models: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SdkAbortRequest {
    pub session_id: String,
    pub op_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SdkMessagesQuery {
    pub session_id: String,
    /// Message cursor; page returns messages with seq < before (newest first).
    pub before: Option<i64>,
    #[serde(default = "default_limit")]
    pub limit: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SdkStateQuery {
    pub session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SdkSessionQuery {
    pub session_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionListEntry {
    pub id: String,
    pub title: String,
    pub provider: String,
    pub model: String,
    pub state: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionListResponse {
    pub sessions: Vec<SessionListEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PermissionListEntry {
    pub id: String,
    pub session_id: String,
    pub capability: String,
    pub detail: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PermissionListResponse {
    pub permissions: Vec<PermissionListEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QuestionListResponse {
    pub questions: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct QuestionReplyRequest {
    pub question_id: String,
    pub decision: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NetworkListResponse {
    pub networks: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NetworkReplyRequest {
    pub network_id: String,
    pub decision: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConfigGetResponse {
    pub config: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ConfigSetRequest {
    pub config: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConfigSetResponse {
    pub ok: bool,
}

/// The `/global/event` resume cursor query: `after=<n>` replays events with
/// id > n. Oversized values are clamped by the server to what the ring can
/// serve (never an error).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GlobalEventsQuery {
    pub after: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CreateSessionRequest {
    pub provider: String,
    pub model: String,
    pub workspace: Option<String>,
    pub title: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CreateSessionResponse {
    pub id: String,
    pub title: String,
    pub created_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PromptRequest {
    pub prompt: String,
    /// File mentions the user attached.
    #[serde(default)]
    pub files: Vec<String>,
    /// If set, resumes this operation instead of starting a new turn.
    pub op_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PromptResponse {
    pub op_id: String,
    pub accepted: bool,
    pub queued: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MessagesQuery {
    /// Message cursor; page returns messages with seq < before (newest first).
    pub before: Option<i64>,
    #[serde(default = "default_limit")]
    pub limit: i64,
    /// SSE resume cursor for the event stream.
    pub events_after: Option<i64>,
}

fn default_limit() -> i64 {
    100
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessagesPage {
    pub session_id: String,
    pub messages: Vec<Message>,
    /// True when older messages exist (there is another page).
    pub has_more: bool,
    pub next_before: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionState {
    pub session_id: String,
    pub state: String,
    pub title: String,
    pub last_event_seq: i64,
    pub agent_state: AgentStateView,
    pub task_ledger: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentStateView {
    pub state: String,
    pub label: String,
    pub active: bool,
    pub terminal: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PermissionDecisionRequest {
    pub permission_id: String,
    pub decision: String, // "allow" | "deny"
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PermissionDecisionResponse {
    pub ok: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AbortRequest {
    pub op_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AbortResponse {
    pub aborted: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProviderList {
    pub providers: Vec<ProviderInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProviderInfo {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub models: Vec<ModelInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub capabilities: ModelCapabilities,
}

// ------------------------------------------------------------------ messages

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub id: String,
    pub role: String, // "user" | "assistant" | "system"
    pub session_id: String,
    pub seq: i64,
    pub created_ms: i64,
    pub parts: Vec<Part>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Part {
    Text {
        text: String,
    },
    Reasoning {
        text: String,
    },
    ToolCall {
        tool_call_id: String,
        name: String,
        input: serde_json::Value,
        state: String,
    },
    ToolResult {
        tool_call_id: String,
        result: ToolResultBody,
    },
    Summary {
        text: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ToolResultBody {
    /// Last N lines, error lines, exit code, important matches — never the
    /// full unbounded output.
    pub excerpt: String,
    pub exit_code: Option<i32>,
    pub artifact: Option<String>,
    /// Pointer into the durable artifact for slice reads.
    pub slice_hint: Option<String>,
}

// ------------------------------------------------------------------ provider config

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ProviderConfig {
    pub id: String,
    pub kind: String,
    pub base_url: String,
    /// Environment variable name holding the API key (never the key itself).
    pub api_key_env: Option<String>,
    pub models: Vec<ModelConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub id: String,
    pub name: String,
    pub capabilities: ModelCapabilities,
}

// ------------------------------------------------------------ global event bus
// The real v7.5.6 global event stream wraps every payload in an envelope
// carrying the workspace identity; the payload's `type` is the discriminator
// (the SSE `event:` field is optional). Frozen field presence.

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GlobalEvent {
    /// Workspace root the event belongs to (null for daemon-global events).
    pub directory: Option<String>,
    pub project: Option<serde_json::Value>,
    pub workspace: Option<serde_json::Value>,
    pub payload: GlobalEventPayload,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GlobalEventPayload {
    SessionCreated {
        session_id: String,
    },
    SessionTurnOpen {
        session_id: String,
        turn_id: String,
    },
    SessionTurnClose {
        session_id: String,
        turn_id: String,
    },
    SessionQueueChanged {
        session_id: String,
        queue_len: u64,
    },
    BackgroundProcessUpdated {
        process_id: String,
        status: String,
    },
    InteractiveTerminalData {
        session_id: String,
        data: String,
    },
    SandboxStatusChanged {
        session_id: String,
        status: String,
    },
    IndexingStatus {
        workspace: String,
        status: String,
    },
    MessagePartUpdated {
        session_id: String,
        message_id: String,
        part: Part,
    },
    SessionNextTextDelta {
        session_id: String,
        delta: String,
    },
    SessionNextReasoningDelta {
        session_id: String,
        delta: String,
    },
    SessionNextToolCalled {
        session_id: String,
        tool: String,
    },
    SessionStateChanged {
        session_id: String,
        state: String,
    },
    Error {
        session_id: Option<String>,
        code: String,
        message: String,
    },
}

impl GlobalEventPayload {
    /// The frozen SSE `type` discriminator for this payload.
    pub fn type_name(&self) -> &'static str {
        match self {
            GlobalEventPayload::SessionCreated { .. } => "session_created",
            GlobalEventPayload::SessionTurnOpen { .. } => "session_turn_open",
            GlobalEventPayload::SessionTurnClose { .. } => "session_turn_close",
            GlobalEventPayload::SessionQueueChanged { .. } => "session_queue_changed",
            GlobalEventPayload::BackgroundProcessUpdated { .. } => "background_process_updated",
            GlobalEventPayload::InteractiveTerminalData { .. } => "interactive_terminal_data",
            GlobalEventPayload::SandboxStatusChanged { .. } => "sandbox_status_changed",
            GlobalEventPayload::IndexingStatus { .. } => "indexing_status",
            GlobalEventPayload::MessagePartUpdated { .. } => "message_part_updated",
            GlobalEventPayload::SessionNextTextDelta { .. } => "session_next_text_delta",
            GlobalEventPayload::SessionNextReasoningDelta { .. } => "session_next_reasoning_delta",
            GlobalEventPayload::SessionNextToolCalled { .. } => "session_next_tool_called",
            GlobalEventPayload::SessionStateChanged { .. } => "session_state_changed",
            GlobalEventPayload::Error { .. } => "error",
        }
    }
}

impl GlobalEvent {
    /// Project one journal event onto the global event envelope. `directory`
    /// is the workspace root of the connection (null when unknown).
    ///
    /// Text-bearing chunk events (payload carries `text`/`delta`/`reasoning`)
    /// become delta payloads; the server coalesces consecutive text deltas
    /// before framing. Chunk events whose payload only carries `text_len`
    /// (the current journal) have no delta text here and return `None` — the
    /// server recovers the text by diffing the stored message parts instead.
    pub fn from_journal_event(e: &Event, directory: Option<String>) -> Option<GlobalEvent> {
        let sid = e.session_id.to_string();
        let wrap = |payload: GlobalEventPayload| GlobalEvent {
            directory: directory.clone(),
            project: None,
            workspace: None,
            payload,
        };
        match e.kind {
            EventKind::SessionCreated => {
                Some(wrap(GlobalEventPayload::SessionCreated { session_id: sid }))
            }
            EventKind::PromptReceived => Some(wrap(GlobalEventPayload::SessionTurnOpen {
                session_id: sid,
                turn_id: turn_id(e),
            })),
            EventKind::TurnCompleted => Some(wrap(GlobalEventPayload::SessionTurnClose {
                session_id: sid,
                turn_id: turn_id(e),
            })),
            EventKind::ModelChunkReceived => e.payload.as_ref().and_then(|p| {
                if let Some(t) = p
                    .get("text")
                    .or_else(|| p.get("delta"))
                    .and_then(|v| v.as_str())
                {
                    return Some(wrap(GlobalEventPayload::SessionNextTextDelta {
                        session_id: sid,
                        delta: t.to_string(),
                    }));
                }
                if let Some(t) = p.get("reasoning").and_then(|v| v.as_str()) {
                    return Some(wrap(GlobalEventPayload::SessionNextReasoningDelta {
                        session_id: sid,
                        delta: t.to_string(),
                    }));
                }
                let message_id = p.get("message_id").and_then(|v| {
                    v.as_str()
                        .map(String::from)
                        .or_else(|| v.as_i64().map(|i| i.to_string()))
                });
                let part = p
                    .get("part")
                    .and_then(|v| serde_json::from_value(v.clone()).ok());
                match (message_id, part) {
                    (Some(message_id), Some(part)) => {
                        Some(wrap(GlobalEventPayload::MessagePartUpdated {
                            session_id: sid,
                            message_id,
                            part,
                        }))
                    }
                    _ => None,
                }
            }),
            EventKind::ToolRequested => e.payload.as_ref().map(|p| {
                let tool = p
                    .get("tool")
                    .and_then(|v| v.as_str())
                    .or_else(|| p.get("capability").and_then(|v| v.as_str()))
                    .unwrap_or("unknown")
                    .to_string();
                wrap(GlobalEventPayload::SessionNextToolCalled {
                    session_id: sid,
                    tool,
                })
            }),
            EventKind::Failed => Some(wrap(GlobalEventPayload::Error {
                session_id: Some(sid),
                code: "agent_failed".into(),
                message: e
                    .payload
                    .as_ref()
                    .and_then(|p| p.get("message"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("agent failed")
                    .to_string(),
            })),
            // Every other journal kind is projected as a state change; the
            // label is the state the journal recorded after the event.
            _ => Some(wrap(GlobalEventPayload::SessionStateChanged {
                session_id: sid,
                state: e.state.label().to_string(),
            })),
        }
    }
}

/// The turn identity for turn open/close pairing: the durable op id when the
/// journal carried one, else a stable per-sequence fallback.
fn turn_id(e: &Event) -> String {
    e.op_id
        .map(|o| o.to_string())
        .unwrap_or_else(|| format!("turn-{}", e.seq.raw()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kilop_core::id::{EventSeq, OpId, SessionId};
    use kilop_core::state::AgentState;

    #[test]
    fn startup_line_is_the_frozen_stdout_contract() {
        assert_eq!(
            startup_line(45678),
            "kilo server listening on http://127.0.0.1:45678"
        );
        // Port 0 never appears on the wire; the bound port is substituted.
        let line = startup_line(0);
        assert!(line.ends_with(":0"));
        // The template documents the shape.
        assert_eq!(
            STARTUP_LINE_TEMPLATE.replace("{port}", "45678"),
            startup_line(45678)
        );
    }

    #[test]
    fn handshake_line_roundtrip_and_garbage() {
        let h = Handshake {
            version: "0.1.0".into(),
            protocol: "v756".into(),
            pid: 4242,
            auth_token: "tok-123".into(),
            port: 45678,
        };
        let line = h.to_line();
        assert!(line.starts_with("KILO_PLUS_HANDSHAKE "));
        assert_eq!(Handshake::from_line(&line), Some(h));
        assert_eq!(Handshake::from_line("junk"), None);
        assert_eq!(Handshake::from_line("KILO_PLUS_HANDSHAKE {"), None);
        assert_eq!(Handshake::from_line(""), None);
        // Trailing garbage after a valid line is rejected (prefix stripped,
        // then full JSON parse).
        assert_eq!(Handshake::from_line(&format!("{line} extra")), None);
        // The startup line is NOT a JSON handshake: from_line rejects it.
        assert_eq!(Handshake::from_line(&startup_line(45678)), None);
    }

    #[test]
    fn deny_unknown_fields_on_requests() {
        let ok = CreateSessionRequest {
            provider: "ollama".into(),
            model: "qwen3.8".into(),
            workspace: None,
            title: None,
        };
        let json = serde_json::to_string(&ok).unwrap();
        assert!(serde_json::from_str::<CreateSessionRequest>(&json).is_ok());
        // Drift: extra field must fail loudly.
        let evil = r#"{"provider":"ollama","model":"qwen3.8","secret_config":"hax"}"#;
        assert!(serde_json::from_str::<CreateSessionRequest>(evil).is_err());
        // Missing required field fails too.
        let missing = r#"{"provider":"ollama"}"#;
        assert!(serde_json::from_str::<CreateSessionRequest>(missing).is_err());
    }

    #[test]
    fn null_behavior_is_explicit() {
        // workspace/title may be null or absent — both are None.
        let a: CreateSessionRequest =
            serde_json::from_str(r#"{"provider":"p","model":"m","workspace":null,"title":null}"#)
                .unwrap();
        let b: CreateSessionRequest =
            serde_json::from_str(r#"{"provider":"p","model":"m"}"#).unwrap();
        assert_eq!(a.workspace, None);
        assert_eq!(a.title, None);
        assert_eq!(b.workspace, None);
        assert_eq!(a, b);
        // But role may not be null in a message.
        let msg = r#"{"id":"1","role":null,"session_id":"2","seq":0,"created_ms":0,"parts":[]}"#;
        assert!(serde_json::from_str::<Message>(msg).is_err());
    }

    #[test]
    fn message_part_shape_is_frozen() {
        let m = Message {
            id: "msg-1".into(),
            role: "assistant".into(),
            session_id: "sess-1".into(),
            seq: 3,
            created_ms: 1700000000000,
            parts: vec![
                Part::Text {
                    text: "hello".into(),
                },
                Part::ToolCall {
                    tool_call_id: "call_1".into(),
                    name: "read_file".into(),
                    input: serde_json::json!({"path": "a.rs"}),
                    state: "pending".into(),
                },
                Part::ToolResult {
                    tool_call_id: "call_1".into(),
                    result: ToolResultBody {
                        excerpt: "1 | fn main".into(),
                        exit_code: Some(0),
                        artifact: Some("artifact://abc".into()),
                        slice_hint: Some("artifact://abc?slice=200&len=100".into()),
                    },
                },
            ],
        };
        let v = serde_json::to_value(&m).unwrap();
        assert_eq!(v["parts"][1]["type"], "tool_call");
        assert_eq!(v["parts"][2]["type"], "tool_result");
        // Unknown part type rejected.
        let evil = serde_json::json!({"type": "escape_hatch", "text": "x"});
        assert!(serde_json::from_value::<Part>(evil).is_err());
        // Missing `type` rejected.
        assert!(serde_json::from_value::<Part>(serde_json::json!({"text": "x"})).is_err());
        // Tool result without artifact is legal (artifact optional).
        let r = ToolResultBody {
            excerpt: "x".into(),
            exit_code: None,
            artifact: None,
            slice_hint: None,
        };
        assert!(
            serde_json::from_value::<ToolResultBody>(serde_json::to_value(&r).unwrap()).is_ok()
        );
    }

    #[test]
    fn messages_query_defaults_lock_paging() {
        let q: MessagesQuery = serde_json::from_str("{}").unwrap();
        assert_eq!(q.limit, 100);
        assert_eq!(q.before, None);
        assert_eq!(q.events_after, None);
        let q: MessagesQuery = serde_json::from_str(r#"{"limit":1}"#).unwrap();
        assert_eq!(q.limit, 1);
        let q: MessagesQuery = serde_json::from_str(r#"{"limit":-5}"#).unwrap();
        assert_eq!(q.limit, -5, "negative limit is the server's job to clamp");
    }

    #[test]
    fn session_state_view_reflects_machine() {
        let v = SessionState {
            session_id: "s1".into(),
            state: "streaming".into(),
            title: "t".into(),
            last_event_seq: 12,
            agent_state: AgentStateView {
                state: "streaming".into(),
                label: "streaming".into(),
                active: true,
                terminal: false,
            },
            task_ledger: None,
        };
        let json = serde_json::to_string(&v).unwrap();
        let back: SessionState = serde_json::from_str(&json).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn global_event_envelope_is_frozen_and_strict() {
        let ge = GlobalEvent {
            directory: Some("/home/u/proj".into()),
            project: None,
            workspace: None,
            payload: GlobalEventPayload::SessionCreated {
                session_id: "s1".into(),
            },
        };
        let json = serde_json::to_value(&ge).unwrap();
        assert_eq!(json["payload"]["type"], "session_created");
        let back: GlobalEvent = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(back, ge);
        // The wire bytes preserve declaration order: directory, project,
        // workspace, payload (frozen field presence).
        let bytes = serde_json::to_string(&ge).unwrap();
        assert!(
            bytes.starts_with(
                "{\"directory\":\"/home/u/proj\",\"project\":null,\"workspace\":null,\"payload\":"
            ),
            "field order drift: {bytes}"
        );
        // The envelope has exactly these four fields.
        let mut keys: Vec<&str> = json
            .as_object()
            .unwrap()
            .keys()
            .map(|k| k.as_str())
            .collect();
        keys.sort_unstable();
        assert_eq!(keys, vec!["directory", "payload", "project", "workspace"]);
        // Unknown envelope fields are drift and must fail loudly.
        let evil = serde_json::json!({
            "directory": null, "project": null, "workspace": null,
            "payload": {"type": "session_created", "session_id": "s1"},
            "smuggled": 1,
        });
        assert!(serde_json::from_value::<GlobalEvent>(evil).is_err());
        // Missing payload fails too.
        let missing = serde_json::json!({"directory": null, "project": null, "workspace": null});
        assert!(serde_json::from_value::<GlobalEvent>(missing).is_err());
    }

    #[test]
    fn global_event_payload_tags_are_snake_case_and_frozen() {
        let cases: Vec<(&str, GlobalEventPayload)> = vec![
            (
                "session_turn_open",
                GlobalEventPayload::SessionTurnOpen {
                    session_id: "s".into(),
                    turn_id: "t1".into(),
                },
            ),
            (
                "session_next_text_delta",
                GlobalEventPayload::SessionNextTextDelta {
                    session_id: "s".into(),
                    delta: "hi".into(),
                },
            ),
            (
                "session_next_reasoning_delta",
                GlobalEventPayload::SessionNextReasoningDelta {
                    session_id: "s".into(),
                    delta: "think".into(),
                },
            ),
            (
                "session_next_tool_called",
                GlobalEventPayload::SessionNextToolCalled {
                    session_id: "s".into(),
                    tool: "read_file".into(),
                },
            ),
            (
                "session_state_changed",
                GlobalEventPayload::SessionStateChanged {
                    session_id: "s".into(),
                    state: "streaming".into(),
                },
            ),
            (
                "error",
                GlobalEventPayload::Error {
                    session_id: Some("s".into()),
                    code: "boom".into(),
                    message: "x".into(),
                },
            ),
        ];
        for (tag, payload) in cases {
            let json = serde_json::to_value(&payload).unwrap();
            assert_eq!(json["type"], tag, "tag drift for {tag}");
            let back: GlobalEventPayload = serde_json::from_value(json).unwrap();
            assert_eq!(back, payload);
        }
        // Unknown payload types are rejected (frozen discriminator set).
        let evil = serde_json::json!({"type": "escape_hatch", "session_id": "s"});
        assert!(serde_json::from_value::<GlobalEventPayload>(evil).is_err());
        // Missing `type` rejected.
        assert!(
            serde_json::from_value::<GlobalEventPayload>(serde_json::json!({"delta": "x"}))
                .is_err()
        );
        // Error payload's session_id may be null or absent.
        let a =
            serde_json::json!({"type": "error", "session_id": null, "code": "c", "message": "m"});
        let b = serde_json::json!({"type": "error", "code": "c", "message": "m"});
        let pa: GlobalEventPayload = serde_json::from_value(a).unwrap();
        let pb: GlobalEventPayload = serde_json::from_value(b).unwrap();
        assert_eq!(pa, pb);
    }

    fn journal_event(
        kind: EventKind,
        op_id: Option<OpId>,
        payload: Option<serde_json::Value>,
    ) -> Event {
        Event::new(
            EventSeq::new(1),
            SessionId::new(1),
            op_id,
            kind,
            AgentState::Idle,
            0,
            payload,
        )
    }

    #[test]
    fn from_journal_event_maps_every_kind_without_panicking() {
        // Exhaustive: every kind produces Some (or a documented None for
        // text_len-only chunks), never a panic.
        let kinds = [
            EventKind::SessionCreated,
            EventKind::PromptReceived,
            EventKind::ContextPrepared,
            EventKind::ModelStarted,
            EventKind::ModelChunkReceived,
            EventKind::ToolRequested,
            EventKind::ToolStarted,
            EventKind::FileChanged,
            EventKind::ToolCompleted,
            EventKind::ToolCancelled,
            EventKind::CheckpointCreated,
            EventKind::ContextCompacted,
            EventKind::CompactRejected,
            EventKind::SubagentStarted,
            EventKind::SubagentCompleted,
            EventKind::TurnCompleted,
            EventKind::PermissionGranted,
            EventKind::PermissionDenied,
            EventKind::CrashDetected,
            EventKind::RecoveryApplied,
            EventKind::SessionEnded,
            EventKind::Suspended,
            EventKind::Resumed,
            EventKind::Failed,
        ];
        for kind in kinds {
            let e = journal_event(kind, None, Some(serde_json::json!({"message_id": 1})));
            let _ = GlobalEvent::from_journal_event(&e, Some("/w".into()));
        }
        // The current journal's chunk payload (text_len only) is None here:
        // the server recovers the text from message parts instead.
        let chunk = journal_event(
            EventKind::ModelChunkReceived,
            None,
            Some(serde_json::json!({"message_id": 5, "text_len": 3})),
        );
        assert!(GlobalEvent::from_journal_event(&chunk, None).is_none());
    }

    #[test]
    fn from_journal_event_projects_lifecycle_and_deltas() {
        let dir = Some("/home/u/proj".to_string());
        let ev = |kind, op, payload| journal_event(kind, op, payload);

        let ge = GlobalEvent::from_journal_event(
            &ev(EventKind::SessionCreated, None, None),
            dir.clone(),
        )
        .unwrap();
        assert_eq!(ge.directory, dir);
        assert_eq!(ge.project, None);
        assert_eq!(ge.workspace, None);
        assert_eq!(
            ge.payload,
            GlobalEventPayload::SessionCreated {
                session_id: "1".into()
            }
        );

        // Turn open/close pair on the same op id.
        let open = GlobalEvent::from_journal_event(
            &ev(
                EventKind::PromptReceived,
                Some(OpId::new(7)),
                Some(serde_json::json!({"queued": false})),
            ),
            dir.clone(),
        )
        .unwrap();
        let close = GlobalEvent::from_journal_event(
            &ev(EventKind::TurnCompleted, Some(OpId::new(7)), None),
            dir.clone(),
        )
        .unwrap();
        assert_eq!(
            open.payload,
            GlobalEventPayload::SessionTurnOpen {
                session_id: "1".into(),
                turn_id: "7".into()
            }
        );
        assert_eq!(
            close.payload,
            GlobalEventPayload::SessionTurnClose {
                session_id: "1".into(),
                turn_id: "7".into()
            }
        );
        // No op id: stable per-sequence fallback, still paired by construction.
        let open = GlobalEvent::from_journal_event(
            &ev(EventKind::PromptReceived, None, None),
            dir.clone(),
        )
        .unwrap();
        match open.payload {
            GlobalEventPayload::SessionTurnOpen { turn_id, .. } => assert_eq!(turn_id, "turn-1"),
            other => panic!("expected turn open, got {other:?}"),
        }

        // Text deltas ride the envelope; the type field is the discriminator.
        let chunk = ev(
            EventKind::ModelChunkReceived,
            None,
            Some(serde_json::json!({"message_id": 3, "text": "hel", "text_len": 3})),
        );
        let ge = GlobalEvent::from_journal_event(&chunk, dir.clone()).unwrap();
        assert_eq!(
            ge.payload,
            GlobalEventPayload::SessionNextTextDelta {
                session_id: "1".into(),
                delta: "hel".into()
            }
        );
        // `delta` is accepted as an alias for the same field.
        let chunk = ev(
            EventKind::ModelChunkReceived,
            None,
            Some(serde_json::json!({"message_id": 3, "delta": "lo"})),
        );
        let ge = GlobalEvent::from_journal_event(&chunk, dir.clone()).unwrap();
        match ge.payload {
            GlobalEventPayload::SessionNextTextDelta { delta, .. } => assert_eq!(delta, "lo"),
            other => panic!("expected text delta, got {other:?}"),
        }
        // Reasoning deltas have their own payload type.
        let chunk = ev(
            EventKind::ModelChunkReceived,
            None,
            Some(serde_json::json!({"message_id": 3, "reasoning": "hmm"})),
        );
        let ge = GlobalEvent::from_journal_event(&chunk, dir.clone()).unwrap();
        assert_eq!(
            ge.payload,
            GlobalEventPayload::SessionNextReasoningDelta {
                session_id: "1".into(),
                delta: "hmm".into()
            }
        );
        // Part-bearing chunks become message_part_updated (message_id may be
        // a string or a number).
        let chunk = ev(
            EventKind::ModelChunkReceived,
            None,
            Some(serde_json::json!({
                "message_id": 9,
                "part": {"type": "text", "text": "x"}
            })),
        );
        let ge = GlobalEvent::from_journal_event(&chunk, dir.clone()).unwrap();
        match ge.payload {
            GlobalEventPayload::MessagePartUpdated {
                message_id, part, ..
            } => {
                assert_eq!(message_id, "9");
                assert_eq!(part, Part::Text { text: "x".into() });
            }
            other => panic!("expected message_part_updated, got {other:?}"),
        }

        // Tool requests become session_next_tool_called with the capability
        // tag as the tool name.
        let tr = ev(
            EventKind::ToolRequested,
            None,
            Some(serde_json::json!({"permission_id": 1, "capability": "execute_shell"})),
        );
        let ge = GlobalEvent::from_journal_event(&tr, dir.clone()).unwrap();
        assert_eq!(
            ge.payload,
            GlobalEventPayload::SessionNextToolCalled {
                session_id: "1".into(),
                tool: "execute_shell".into()
            }
        );

        // Failures become error payloads.
        let failed = ev(
            EventKind::Failed,
            None,
            Some(serde_json::json!({"message": "provider stream died"})),
        );
        let ge = GlobalEvent::from_journal_event(&failed, dir).unwrap();
        assert_eq!(
            ge.payload,
            GlobalEventPayload::Error {
                session_id: Some("1".into()),
                code: "agent_failed".into(),
                message: "provider stream died".into(),
            }
        );

        // Interior kinds are state changes carrying the journaled label.
        let ge = GlobalEvent::from_journal_event(&ev(EventKind::ContextPrepared, None, None), None)
            .unwrap();
        assert_eq!(
            ge.payload,
            GlobalEventPayload::SessionStateChanged {
                session_id: "1".into(),
                state: "idle".into()
            }
        );
    }

    #[test]
    fn sdk_request_bodies_deny_unknown_fields() {
        let ok = SdkPromptRequest {
            session_id: "1".into(),
            prompt: "hi".into(),
            files: vec![],
            models: None,
        };
        let json = serde_json::to_string(&ok).unwrap();
        assert!(serde_json::from_str::<SdkPromptRequest>(&json).is_ok());
        // Drift: unknown body field fails loudly.
        let evil = r#"{"session_id":"1","prompt":"hi","secret_config":"hax"}"#;
        assert!(serde_json::from_str::<SdkPromptRequest>(evil).is_err());
        // files/models are optional.
        let minimal = r#"{"session_id":"1","prompt":"hi"}"#;
        let parsed: SdkPromptRequest = serde_json::from_str(minimal).unwrap();
        assert_eq!(parsed.files, Vec::<String>::new());
        assert_eq!(parsed.models, None);
        // Missing required fields fail.
        assert!(serde_json::from_str::<SdkPromptRequest>(r#"{"prompt":"hi"}"#).is_err());
        assert!(serde_json::from_str::<SdkPromptRequest>(r#"{"session_id":"1"}"#).is_err());
        // abort: op_id optional.
        let a: SdkAbortRequest = serde_json::from_str(r#"{"session_id":"1"}"#).unwrap();
        assert_eq!(a.op_id, None);
        assert!(serde_json::from_str::<SdkAbortRequest>(r#"{"op_id":"x"}"#).is_err());
        // config/set with an oversized nesting value is a Value: parse ok,
        // size limits are the server's job.
        let c: ConfigSetRequest = serde_json::from_str(r#"{"config":{"a":1}}"#).unwrap();
        assert_eq!(c.config, serde_json::json!({"a": 1}));
    }
}
