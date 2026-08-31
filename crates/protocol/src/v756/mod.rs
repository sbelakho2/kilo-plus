//! The v7.5.6 wire types. Field presence, naming, and nullability are frozen.
//! Request types use `deny_unknown_fields`: an unknown field is a protocol
//! drift signal and must fail loudly, not be ignored.

use kilop_core::model::ModelCapabilities;
use serde::{Deserialize, Serialize};

/// The handshake line the daemon prints on stdout after binding:
/// `KILO_PLUS_HANDSHAKE <json>`.
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
    Text { text: String },
    Reasoning { text: String },
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
    Summary { text: String },
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

#[cfg(test)]
mod tests {
    use super::*;

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
                Part::Text { text: "hello".into() },
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
        assert!(serde_json::from_value::<ToolResultBody>(serde_json::to_value(&r).unwrap()).is_ok());
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
}
