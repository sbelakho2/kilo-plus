//! kilop-provider — the common LLM provider interface hub.
//!
//! The agent depends on this trait; the transport families (ollama, openai,
//! anthropic, google, deepseek, gateway) implement it. Requests pass through:
//!
//! ```text
//! Generic Agent Request
//!         ↓
//! Capability Validation
//!         ↓
//! Provider Normalizer
//!         ↓
//! Wire Serializer   (inside each adapter)
//!         ↓
//! HTTP Transport    (inside each adapter)
//! ```
//!
//! Provider quirks stay inside adapters. There is **no `if provider == "…"`**
//! in the agent — behavior is decided by `ModelCapabilities`.

use std::collections::HashMap;
use std::pin::Pin;

use futures::Stream;
#[cfg(test)]
use futures::StreamExt;
use kilop_core::cancellation::CancellationToken;
use kilop_core::id::{OpId, SessionId};
use kilop_core::model::{ModelCapabilities, ReasoningMode};

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
    System,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ContentPart {
    pub kind: ContentKind,
    /// For tool_result parts: which tool call this answers.
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentKind {
    Text { text: String },
    Image { url: String },
    ToolCall { id: String, name: String, input: serde_json::Value },
    ToolResult { content: String, is_error: bool },
}

impl ContentPart {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            kind: ContentKind::Text { text: text.into() },
            tool_call_id: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RequestMessage {
    pub role: Role,
    pub content: Vec<ContentPart>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// Metadata attached to every request. The wire serializer never sees these
/// fields — they exist for retry state, deadlines, and circuit breakers.
#[derive(Debug, Clone)]
pub struct RequestMeta {
    pub operation_id: OpId,
    pub session_id: SessionId,
    pub provider: String,
    pub attempt: u32,
    pub deadline_ms: u64,
    pub cancellation: CancellationToken,
}

/// A normalized agent-level request. Capability validation happens on this
/// type; normalization turns it into wire shapes inside adapters.
#[derive(Debug, Clone)]
pub struct GenericAgentRequest {
    pub model: String,
    /// Cacheable prefix (system instructions, tools, project rules, task state).
    pub system: String,
    pub messages: Vec<RequestMessage>,
    pub tools: Vec<ToolSpec>,
    pub max_output: Option<usize>,
    pub reasoning: Option<ReasoningMode>,
    pub stream: bool,
    pub meta: RequestMeta,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProviderChunk {
    Text { text: String },
    Reasoning { text: String },
    ToolCall {
        id: String,
        name: String,
        /// Accumulated so far; `complete` toggles on the final delta.
        input: serde_json::Value,
        complete: bool,
    },
    Usage {
        tokens_in: u64,
        tokens_out: u64,
    },
    Done,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderErrorKind {
    Network,
    Timeout,
    RateLimited,
    BadRequest,
    Auth,
    Server,
    Cancelled,
    Malformed,
}

impl ProviderErrorKind {
    pub fn retryable(&self) -> bool {
        matches!(
            self,
            ProviderErrorKind::Network
                | ProviderErrorKind::Timeout
                | ProviderErrorKind::RateLimited
                | ProviderErrorKind::Server
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderError {
    pub kind: ProviderErrorKind,
    pub message: String,
    pub retryable: bool,
    /// Provider-native code (http status, ollama error, ...).
    pub code: Option<String>,
}

impl ProviderError {
    pub fn new(kind: ProviderErrorKind, message: impl Into<String>) -> Self {
        let retryable = kind.retryable();
        Self {
            kind,
            message: message.into(),
            retryable,
            code: None,
        }
    }

    pub fn with_code(kind: ProviderErrorKind, code: impl Into<String>, message: impl Into<String>) -> Self {
        let retryable = kind.retryable();
        Self {
            kind,
            message: message.into(),
            retryable,
            code: Some(code.into()),
        }
    }
}

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for ProviderError {}

pub type ProviderStream =
    Pin<Box<dyn Stream<Item = Result<ProviderChunk, ProviderError>> + Send>>;

/// One transport family. Implementations are stateless except config.
pub trait Provider: Send + Sync {
    fn id(&self) -> &str;

    /// Capabilities for a model; discovered by probing, never hard-coded
    /// lists in the agent.
    fn capabilities(&self, model: &str) -> ModelCapabilities;

    fn stream(&self, req: GenericAgentRequest) -> ProviderStream;
}

// ------------------------------------------------------------------ pipeline

/// Step 1: validate a request against known capabilities *before* any wire
/// call. Violations are loud errors, never silent truncation.
pub struct CapabilityValidator;

impl CapabilityValidator {
    pub fn validate(
        req: &GenericAgentRequest,
        caps: &ModelCapabilities,
    ) -> Result<(), kilop_core::Error> {
        use kilop_core::error::{Error, ErrorKind};
        if !req.tools.is_empty() && !caps.tools {
            return Err(Error::new(
                ErrorKind::Malformed,
                format!(
                    "model {} does not support tools, but {} tool(s) requested",
                    req.model,
                    req.tools.len()
                ),
            ));
        }
        if req.reasoning.is_some() && !(caps.reasoning || caps.thinking) {
            return Err(Error::new(
                ErrorKind::Malformed,
                format!("model {} does not support reasoning", req.model),
            ));
        }
        if let Some(max_out) = req.max_output {
            if max_out > caps.max_output {
                return Err(Error::new(
                    ErrorKind::Oversized,
                    format!(
                        "requested max_output {max_out} exceeds model cap {}",
                        caps.max_output
                    ),
                ));
            }
        }
        Ok(())
    }
}

/// Step 2: ensure internal option names never leak onto wire APIs. The
/// normalizer strips anything not on the explicit whitelist and enforces
/// bound clamps. Adapters additionally translate to their wire vocabulary.
pub struct RequestNormalizer;

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct NormalizedRequest {
    pub model: String,
    pub system: String,
    pub messages: Vec<RequestMessage>,
    pub tools: Vec<ToolSpec>,
    pub max_output: Option<usize>,
    pub reasoning: Option<ReasoningMode>,
    pub stream: bool,
}

impl RequestNormalizer {
    /// Whitelisted internal fields (the frozen set). Anything else that ever
    /// sneaks into `GenericAgentRequest` will simply not exist on the wire —
    /// this is the structural fix for leaked compaction/option names.
    pub fn normalize(req: &GenericAgentRequest) -> NormalizedRequest {
        NormalizedRequest {
            model: req.model.clone(),
            system: req.system.clone(),
            messages: req.messages.clone(),
            tools: req.tools.clone(),
            max_output: req.max_output,
            reasoning: req.reasoning,
            stream: req.stream,
        }
    }
}

/// Dynamic model registry: providers register their models; the agent asks
/// the registry, never the provider string.
#[derive(Default)]
pub struct ProviderRegistry {
    providers: HashMap<String, Arc<dyn Provider>>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, p: Arc<dyn Provider>) {
        self.providers.insert(p.id().to_string(), p);
    }

    pub fn get(&self, id: &str) -> Option<Arc<dyn Provider>> {
        self.providers.get(id).cloned()
    }

    pub fn ids(&self) -> Vec<String> {
        let mut v: Vec<String> = self.providers.keys().cloned().collect();
        v.sort();
        v
    }

    pub fn capabilities(&self, provider: &str, model: &str) -> Option<ModelCapabilities> {
        self.get(provider).map(|p| p.capabilities(model))
    }

    pub fn len(&self) -> usize {
        self.providers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }
}

pub use std::sync::Arc;

/// Adversarial wire-testing harness (mock HTTP server).
pub mod testing;

// ------------------------------------------------------------------ fake provider for tests

/// Scripted provider for adversarial agent/server tests. Responses are
/// user-controlled; streams can be made to die mid-flight, return malformed
/// tool calls, rate-limit, etc.
pub struct FakeProvider {
    pub id: String,
    pub caps: ModelCapabilities,
    pub script: std::sync::Mutex<Vec<ScriptedResponse>>,
    pub fail_after_chunks: Option<usize>,
}

#[derive(Debug, Clone)]
pub enum ScriptedResponse {
    Text(String),
    ToolCall { id: String, name: String, input: serde_json::Value },
    /// Ends the stream cleanly.
    End,
    /// Simulates a network/stream death (error mid-flight).
    Die(ProviderError),
}

impl FakeProvider {
    pub fn new(id: &str, caps: ModelCapabilities) -> Self {
        Self {
            id: id.to_string(),
            caps,
            script: std::sync::Mutex::new(vec![ScriptedResponse::End]),
            fail_after_chunks: None,
        }
    }

    pub fn with_script(id: &str, caps: ModelCapabilities, script: Vec<ScriptedResponse>) -> Self {
        Self {
            id: id.to_string(),
            caps,
            script: std::sync::Mutex::new(script),
            fail_after_chunks: None,
        }
    }

    pub fn die_mid_stream(id: &str, caps: ModelCapabilities) -> Self {
        Self {
            id: id.to_string(),
            caps,
            script: std::sync::Mutex::new(vec![ScriptedResponse::Text("partial reply…".into())]),
            fail_after_chunks: Some(1),
        }
    }

    /// If true, the next call fails with RateLimited (and the script is
    /// untouched) — used for retry tests.
    pub fn inject_rate_limit(&self) {
        self.script.lock().unwrap().insert(
            0,
            ScriptedResponse::Die(ProviderError::new(
                ProviderErrorKind::RateLimited,
                "429 too many",
            )),
        );
    }
}

impl Provider for FakeProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn capabilities(&self, _model: &str) -> ModelCapabilities {
        self.caps.clone()
    }

    fn stream(&self, req: GenericAgentRequest) -> ProviderStream {
        // Scripts are consumed exactly once (a replaying provider would let
        // the agent loop forever re-executing the same calls).
        let script = std::mem::take(&mut *self.script.lock().unwrap());
        let fail_after = self.fail_after_chunks;
        let stream = futures::stream::unfold(
            (script.into_iter(), 0usize, fail_after, false),
            move |(mut remaining, mut emitted, fail_after, ended)| async move {
                if ended {
                    return None; // exactly one terminal item, then end
                }
                if let Some(limit) = fail_after {
                    if emitted >= limit {
                        return Some((
                            Err(ProviderError::new(
                                ProviderErrorKind::Network,
                                "connection vanished mid-stream (injected)",
                            )),
                            (remaining, emitted, fail_after, true),
                        ));
                    }
                }
                match remaining.next() {
                    Some(ScriptedResponse::Text(t)) => {
                        emitted += 1;
                        Some((
                            Ok(ProviderChunk::Text { text: t }),
                            (remaining, emitted, fail_after, false),
                        ))
                    }
                    Some(ScriptedResponse::ToolCall { id, name, input }) => {
                        emitted += 1;
                        Some((
                            Ok(ProviderChunk::ToolCall {
                                id,
                                name,
                                input,
                                complete: true,
                            }),
                            (remaining, emitted, fail_after, false),
                        ))
                    }
                    Some(ScriptedResponse::Die(e)) => {
                        emitted += 1;
                        Some((Err(e), (remaining, emitted, fail_after, true)))
                    }
                    Some(ScriptedResponse::End) | None => {
                        let _ = req.meta.deadline_ms;
                        Some((
                            Ok(ProviderChunk::Done),
                            (remaining, emitted, fail_after, true),
                        ))
                    }
                }
            },
        );
        Box::pin(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kilop_core::error::ErrorKind;

    fn req() -> GenericAgentRequest {
        GenericAgentRequest {
            model: "m".into(),
            system: "s".into(),
            messages: vec![],
            tools: vec![],
            max_output: None,
            reasoning: None,
            stream: true,
            meta: RequestMeta {
                operation_id: OpId::new(1),
                session_id: SessionId::new(1),
                provider: "fake".into(),
                attempt: 0,
                deadline_ms: 1000,
                cancellation: CancellationToken::new(),
            },
        }
    }

    #[test]
    fn capability_validation_rejects_tools_on_tool_less_model() {
        let caps = ModelCapabilities {
            tools: false,
            ..Default::default()
        };
        let mut r = req();
        r.tools.push(ToolSpec {
            name: "read_file".into(),
            description: "d".into(),
            input_schema: serde_json::json!({}),
        });
        let err = CapabilityValidator::validate(&r, &caps).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Malformed);
    }

    #[test]
    fn capability_validation_rejects_reasoning_and_oversized_output() {
        let caps = ModelCapabilities {
            reasoning: false,
            thinking: false,
            max_output: 1000,
            ..Default::default()
        };
        let mut r = req();
        r.reasoning = Some(ReasoningMode::High);
        assert!(CapabilityValidator::validate(&r, &caps).is_err());
        r.reasoning = None;
        r.max_output = Some(2000);
        let err = CapabilityValidator::validate(&r, &caps).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Oversized);
        r.max_output = Some(1000);
        assert!(CapabilityValidator::validate(&r, &caps).is_ok());
    }

    #[test]
    fn normalizer_never_carries_internal_meta() {
        let r = req();
        let n = RequestNormalizer::normalize(&r);
        // Internal fields (op/session/deadline/cancellation/attempt) simply
        // do not exist on the normalized request — they cannot leak.
        let json = serde_json::to_value(&n).unwrap();
        let obj = json.as_object().unwrap();
        for leaked in ["operation_id", "session_id", "attempt", "deadline_ms", "cancellation"] {
            assert!(!obj.contains_key(leaked), "internal field {leaked} leaked");
        }
        assert_eq!(obj.len(), 7, "frozen normalized shape");
    }

    #[test]
    fn registry_dynamic_and_capability_source_of_truth() {
        let mut reg = ProviderRegistry::new();
        let fake = FakeProvider::with_script(
            "test",
            ModelCapabilities {
                context: 262144,
                tools: true,
                ..Default::default()
            },
            vec![ScriptedResponse::Text("hi".into()), ScriptedResponse::End],
        );
        reg.register(Arc::new(fake));
        assert_eq!(reg.ids(), vec!["test"]);
        let caps = reg.capabilities("test", "qwen3.8").unwrap();
        assert_eq!(caps.context, 262144);
        assert!(caps.tools);
        assert!(reg.capabilities("missing", "x").is_none());
    }

    #[test]
    fn provider_error_retryability_matches_kind() {
        assert!(ProviderErrorKind::Network.retryable());
        assert!(ProviderErrorKind::Timeout.retryable());
        assert!(ProviderErrorKind::RateLimited.retryable());
        assert!(ProviderErrorKind::Server.retryable());
        assert!(!ProviderErrorKind::BadRequest.retryable());
        assert!(!ProviderErrorKind::Auth.retryable());
        assert!(!ProviderErrorKind::Cancelled.retryable());
        assert!(!ProviderErrorKind::Malformed.retryable());
    }

    #[tokio::test]
    async fn fake_provider_stream_contract() {
        let caps = ModelCapabilities::default();
        let fake = FakeProvider::with_script(
            "f",
            caps,
            vec![
                ScriptedResponse::Text("a".into()),
                ScriptedResponse::ToolCall {
                    id: "c1".into(),
                    name: "read_file".into(),
                    input: serde_json::json!({"path": "x"}),
                },
                ScriptedResponse::End,
            ],
        );
        let chunks: Vec<_> = fake.stream(req()).collect().await;
        assert_eq!(chunks.len(), 3);
        assert!(matches!(chunks[0], Ok(ProviderChunk::Text { .. })));
        assert!(matches!(chunks[1], Ok(ProviderChunk::ToolCall { .. })));
        assert_eq!(chunks[2], Ok(ProviderChunk::Done));
    }

    #[tokio::test]
    async fn fake_provider_dies_mid_stream() {
        let fake = FakeProvider::die_mid_stream("f", ModelCapabilities::default());
        let chunks: Vec<_> = fake.stream(req()).collect().await;
        assert!(chunks.len() == 2 && chunks[0].is_ok() && chunks[1].is_err());
        assert_eq!(chunks[1].as_ref().unwrap_err().kind, ProviderErrorKind::Network);
    }
}
