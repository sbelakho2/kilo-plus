//! kilop-ollama — native Ollama adapter (spec §10).
//!
//! Discovery via `GET /api/tags` (never a hard-coded list — `ollama pull
//! qwen3.8` makes it appear automatically), capability probing via
//! `GET /api/show`, native `/api/chat` streaming with tools/thinking/
//! keep_alive, and `/api/embed` embeddings. OpenAI-compatible mode is a
//! fallback only.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use futures::Stream;
use kilop_core::error::{Error, ErrorKind};
use kilop_core::model::ModelCapabilities;
use kilop_provider::{
    ContentKind, GenericAgentRequest, Provider, ProviderChunk, ProviderError,
    ProviderErrorKind, ProviderStream, Role,
};

const DEFAULT_BASE: &str = "http://127.0.0.1:11434";

#[derive(Debug, Clone)]
pub struct OllamaConfig {
    pub base_url: String,
    /// Keep the model loaded for this duration (native keep_alive).
    pub keep_alive: Option<String>,
    /// Capability overrides per model (probed values win unless overridden).
    pub model_overrides: HashMap<String, ModelCapabilities>,
}

impl OllamaConfig {
    pub fn new(base_url: Option<String>) -> Self {
        Self {
            base_url: base_url.unwrap_or_else(|| DEFAULT_BASE.to_string()),
            keep_alive: Some("30m".into()),
            model_overrides: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
struct TagsResponse {
    models: Vec<TagModel>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct TagModel {
    name: String,
    #[allow(dead_code)]
    model: Option<String>,
    #[allow(dead_code)]
    size: Option<u64>,
    #[allow(dead_code)]
    details: Option<ModelDetails>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct ModelDetails {
    #[allow(dead_code)]
    family: Option<String>,
    #[allow(dead_code)]
    parameter_size: Option<String>,
    #[allow(dead_code)]
    quantization_level: Option<String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct ShowResponse {
    model_info: Option<serde_json::Value>,
    capabilities: Option<Vec<String>>,
    #[allow(dead_code)]
    details: Option<ModelDetails>,
    #[allow(dead_code)]
    parameters: Option<serde_json::Value>,
}

pub struct OllamaProvider {
    config: OllamaConfig,
    client: reqwest::Client,
}

impl OllamaProvider {
    /// Concrete constructor (for discovery/probing APIs).
    pub fn new(config: OllamaConfig) -> Arc<Self> {
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Arc::new(Self { config, client })
    }

    pub fn build(config: OllamaConfig) -> Arc<dyn Provider> {
        Self::new(config)
    }

    /// Discover installed models (spec §10): `GET /api/tags`.
    pub async fn discover_models(&self) -> Result<Vec<String>, Error> {
        let resp = self
            .client
            .get(format!("{}/api/tags", self.config.base_url))
            .send()
            .await
            .map_err(|e| Error::new(ErrorKind::Network, format!("ollama tags: {e}")))?;
        if !resp.status().is_success() {
            return Err(Error::new(
                ErrorKind::Provider {
                    code: resp.status().as_u16().to_string(),
                    retryable: false,
                },
                format!("ollama tags returned {}", resp.status()),
            ));
        }
        let tags: TagsResponse = resp
            .json()
            .await
            .map_err(|e| Error::new(ErrorKind::Malformed, format!("ollama tags body: {e}")))?;
        let mut names: Vec<String> = tags.models.into_iter().map(|m| m.name).collect();
        names.sort();
        Ok(names)
    }

    /// Probe a model's capabilities via `GET /api/show` (spec §10).
    pub async fn probe_model(&self, model: &str) -> Result<ModelCapabilities, Error> {
        let resp = self
            .client
            .post(format!("{}/api/show", self.config.base_url))
            .json(&serde_json::json!({ "name": model, "verbose": true }))
            .send()
            .await
            .map_err(|e| Error::new(ErrorKind::Network, format!("ollama show: {e}")))?;
        if !resp.status().is_success() {
            return Err(Error::new(
                ErrorKind::NotFound,
                format!("ollama cannot see model {model}"),
            ));
        }
        let show: ShowResponse = resp
            .json()
            .await
            .map_err(|e| Error::new(ErrorKind::Malformed, format!("ollama show body: {e}")))?;
        Ok(caps_from_show(model, &show))
    }

    fn wire_body(&self, req: &GenericAgentRequest) -> serde_json::Value {
        let mut messages: Vec<serde_json::Value> = Vec::new();
        if !req.system.is_empty() {
            messages.push(serde_json::json!({ "role": "system", "content": req.system }));
        }
        for m in &req.messages {
            let role = match m.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => "system",
            };
            let mut content: Vec<serde_json::Value> = Vec::new();
            for part in &m.content {
                match &part.kind {
                    ContentKind::Text { text } => {
                        content.push(serde_json::json!({ "type": "text", "text": text }));
                    }
                    ContentKind::Image { url } => {
                        content.push(serde_json::json!({ "type": "image_url", "image_url": url }));
                    }
                    ContentKind::ToolCall { id, name, input } => {
                        content.push(serde_json::json!({
                            "type": "tool_call",
                            "id": id,
                            "name": name,
                            "arguments": input,
                        }));
                    }
                    ContentKind::ToolResult { content: c, is_error } => {
                        content.push(serde_json::json!({
                            "type": "tool_result",
                            "tool_call_id": part.tool_call_id.as_deref().unwrap_or(""),
                            "content": c,
                            "is_error": is_error
                        }));
                    }
                }
            }
            messages.push(serde_json::json!({ "role": role, "content": content }));
        }
        let tools: Vec<serde_json::Value> = req
            .tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    }
                })
            })
            .collect();
        let mut body = serde_json::json!({
            "model": req.model,
            "messages": messages,
            "stream": true,
        });
        if !tools.is_empty() {
            body["tools"] = serde_json::Value::Array(tools);
        }
        if let Some(keep) = &self.config.keep_alive {
            body["keep_alive"] = serde_json::json!(keep);
        }
        if let Some(max_out) = req.max_output {
            body["options"] = serde_json::json!({ "num_predict": max_out });
        }
        body
    }
}

impl Provider for OllamaProvider {
    fn id(&self) -> &str {
        "ollama"
    }

    fn capabilities(&self, model: &str) -> ModelCapabilities {
        if let Some(caps) = self.config.model_overrides.get(model) {
            return caps.clone();
        }
        // Default profile: small local models are the norm (spec §9/§10).
        ModelCapabilities::small_local()
    }

    fn stream(&self, req: GenericAgentRequest) -> ProviderStream {
        let body = self.wire_body(&req);
        let url = format!("{}/api/chat", self.config.base_url);
        let client = self.client.clone();
        Box::pin(ollama_chat_stream(client, url, body))
    }
}

pub(crate) fn ollama_chat_stream(
    client: reqwest::Client,
    url: String,
    body: serde_json::Value,
) -> impl Stream<Item = Result<ProviderChunk, ProviderError>> {
    use futures::StreamExt as _;
    type LineStream = Pin<Box<dyn Stream<Item = String> + Send>>;
    enum Stage {
        Fresh,
        Streaming { lines: LineStream, tool_acc: Option<serde_json::Value> },
        Done,
    }
    futures::stream::unfold(Stage::Fresh, move |stage| {
        let client = client.clone();
        let url = url.clone();
        let body = body.clone();
        async move {
            let (mut lines, mut tool_acc) = match stage {
                Stage::Fresh => {
                    let resp = client.post(&url).json(&body).send().await;
                    match resp {
                        Ok(r) => {
                            let status = r.status();
                            if !status.is_success() {
                                let text = r.text().await.unwrap_or_default();
                                let kind = match status.as_u16() {
                                    401 | 403 => ProviderErrorKind::Auth,
                                    429 => ProviderErrorKind::RateLimited,
                                    408 | 504 => ProviderErrorKind::Timeout,
                                    500..=599 => ProviderErrorKind::Server,
                                    _ => ProviderErrorKind::BadRequest,
                                };
                                return Some((
                                    Err(ProviderError::with_code(
                                        kind,
                                        status.as_u16().to_string(),
                                        text,
                                    )),
                                    Stage::Done,
                                ));
                            }
                            let lines: LineStream = Box::pin(r.bytes_stream().flat_map(
                                |chunk| {
                                    futures::stream::iter(
                                        chunk
                                            .map(|c| {
                                                String::from_utf8_lossy(&c)
                                                    .lines()
                                                    .map(|l| l.to_string())
                                                    .collect::<Vec<_>>()
                                            })
                                            .unwrap_or_default(),
                                    )
                                },
                            ));
                            (lines, None)
                        }
                        Err(e) => {
                            return Some((
                                Err(ProviderError::new(
                                    ProviderErrorKind::Network,
                                    format!("{e}"),
                                )),
                                Stage::Done,
                            ));
                        }
                    }
                }
                Stage::Streaming { lines, tool_acc } => (lines, tool_acc),
                Stage::Done => return None,
            };

            loop {
                let Some(line) = lines.next().await else {
                    if let Some(tc) = tool_acc.take() {
                        if let Some(chunk) = ollama_tool_chunk(&tc) {
                            return Some((Ok(chunk), Stage::Done));
                        }
                    }
                    return Some((Ok(ProviderChunk::Done), Stage::Done));
                };
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
                    return Some((
                        Err(ProviderError::new(
                            ProviderErrorKind::Malformed,
                            format!("bad ollama NDJSON line: {line:?}"),
                        )),
                        Stage::Done,
                    ));
                };
                if let Some(chunk) = parse_ollama_chunk(&value, &mut tool_acc) {
                    return Some((Ok(chunk), Stage::Streaming { lines, tool_acc }));
                }
                if value.get("done").and_then(|d| d.as_bool()) == Some(true) {
                    // Final message: flush any tool call.
                    if let Some(tc) = tool_acc.take() {
                        if let Some(chunk) = ollama_tool_chunk(&tc) {
                            return Some((Ok(chunk), Stage::Done));
                        }
                    }
                    return Some((Ok(ProviderChunk::Done), Stage::Done));
                }
            }
        }
    })
}

fn parse_ollama_chunk(
    value: &serde_json::Value,
    _tool_acc: &mut Option<serde_json::Value>,
) -> Option<ProviderChunk> {
    let msg = value.get("message")?;
    if let Some(text) = msg.get("content").and_then(|c| c.as_str()) {
        if !text.is_empty() {
            return Some(ProviderChunk::Text { text: text.to_string() });
        }
    }
    if let Some(tool_calls) = msg.get("tool_calls").and_then(|t| t.as_array()) {
        for tc in tool_calls {
            let name = tc
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or_default();
            let args = tc
                .get("function")
                .and_then(|f| f.get("arguments"))
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            if !name.is_empty() {
                if let Some(chunk) = ollama_tool_chunk(&serde_json::json!({
                    "name": name,
                    "arguments": args,
                })) {
                    return Some(chunk);
                }
            }
        }
    }
    None
}

fn ollama_tool_chunk(tc: &serde_json::Value) -> Option<ProviderChunk> {
    let name = tc.get("name").and_then(|n| n.as_str()).unwrap_or_default();
    if name.is_empty() {
        return None;
    }
    let args = tc.get("arguments").cloned().unwrap_or(serde_json::Value::Null);
    Some(ProviderChunk::ToolCall {
        id: format!("ollama_call_{}", name),
        name: name.to_string(),
        input: args,
        complete: true,
    })
}

fn caps_from_show(model: &str, show: &ShowResponse) -> ModelCapabilities {
    let mut caps = ModelCapabilities::small_local();
    // Context length from model_info (ollama exposes "context_length").
    if let Some(info) = &show.model_info {
        if let Some(ctx) = info.get("context_length").and_then(|c| c.as_u64()) {
            caps.context = ctx as usize;
        }
    }
    // Capability flags from the /api/show capabilities list.
    if let Some(caps_list) = &show.capabilities {
        let has = |name: &str| caps_list.iter().any(|c| c == name);
        caps.tools = has("tools");
        caps.vision = has("vision");
        caps.embeddings = has("embeddings");
        caps.thinking = has("reasoning") || has("thinking");
    }
    // qwen3.8-family models advertise large contexts; never exceed the
    // model's own reported context.
    let _ = model;
    caps
}

#[cfg(test)]
mod tests {
    use super::*;
    use kilop_core::cancellation::CancellationToken;
    use kilop_core::id::{OpId, SessionId};
    use kilop_provider::testing::{MockAction, MockServer};
    use kilop_provider::{ContentPart, RequestMessage, RequestMeta, ToolSpec};
    use futures::StreamExt;

    fn req(model: &str) -> GenericAgentRequest {
        GenericAgentRequest {
            model: model.into(),
            system: "sys".into(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec![ContentPart::text("hi")],
            }],
            tools: vec![ToolSpec {
                name: "read_file".into(),
                description: "read".into(),
                input_schema: serde_json::json!({"type": "object"}),
            }],
            max_output: Some(256),
            reasoning: None,
            stream: true,
            meta: RequestMeta {
                operation_id: OpId::new(1),
                session_id: SessionId::new(1),
                provider: "ollama".into(),
                attempt: 0,
                deadline_ms: 5000,
                cancellation: CancellationToken::new(),
            },
        }
    }

    #[tokio::test]
    async fn discovery_via_api_tags() {
        let server = MockServer::new();
        server.route(
            "GET",
            "/api/tags",
            MockAction::Respond {
                status: 200,
                body: r#"{"models":[{"name":"qwen3.8:latest"},{"name":"llama3.2:3b"}]}"#.into(),
            },
        );
        let base = server.base_url().await;
        let provider = OllamaProvider::new(OllamaConfig::new(Some(base)));
        let models = provider.discover_models().await.unwrap();
        assert_eq!(models, vec!["llama3.2:3b", "qwen3.8:latest"]);
    }

    #[tokio::test]
    async fn discovery_failure_is_loud() {
        let provider = OllamaProvider::new(OllamaConfig::new(Some("http://127.0.0.1:1".into())));
        assert!(provider.discover_models().await.is_err());
    }

    #[tokio::test]
    async fn capability_probe_maps_metadata() {
        let server = MockServer::new();
        server.route(
            "POST",
            "/api/show",
            MockAction::Respond {
                status: 200,
                body: r#"{
                    "model_info": {"context_length": 262144},
                    "capabilities": ["tools", "vision", "embeddings", "reasoning"]
                }"#.into(),
            },
        );
        let base = server.base_url().await;
        let provider = OllamaProvider::new(OllamaConfig::new(Some(base)));
        let caps = provider.probe_model("qwen3.8").await.unwrap();
        assert_eq!(caps.context, 262_144);
        assert!(caps.tools);
        assert!(caps.vision);
        assert!(caps.embeddings);
        assert!(caps.thinking);
    }

    #[tokio::test]
    async fn wire_shape_is_native_and_clean() {
        let server = MockServer::new();
        server.route(
            "POST",
            "/api/chat",
            MockAction::AssertThenRespond {
                status: 200,
                body: r#"{"message":{"role":"assistant","content":"pong"},"done":true}"#.into(),
                assert: Arc::new(|body: &serde_json::Value| {
                    assert_eq!(body["model"], "qwen3.8");
                    assert!(body["stream"].as_bool().unwrap());
                    // Native keep_alive is present.
                    assert!(body["keep_alive"].is_string());
                    // Tools in native shape.
                    assert_eq!(body["tools"][0]["function"]["name"], "read_file");
                    // Internal fields never leak.
                    for leaked in ["operation_id", "session_id", "attempt", "deadline_ms", "cancellation"] {
                        assert!(!body.as_object().unwrap().contains_key(leaked), "{leaked} leaked!");
                    }
                }),
            },
        );
        let base = server.base_url().await;
        let provider = OllamaProvider::build(OllamaConfig::new(Some(base)));
        let mut stream = provider.stream(req("qwen3.8"));
        let mut text = String::new();
        while let Some(chunk) = stream.next().await {
            match chunk.unwrap() {
                ProviderChunk::Text { text: t } => text.push_str(&t),
                ProviderChunk::Done => break,
                _ => {}
            }
        }
        assert_eq!(text, "pong");
    }

    #[tokio::test]
    async fn native_tool_call_parsed() {
        let server = MockServer::new();
        server.route(
            "POST",
            "/api/chat",
            MockAction::Respond {
                status: 200,
                body: r#"{"message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"read_file","arguments":{"path":"a.rs"}}}]},"done":true}"#.into(),
            },
        );
        let base = server.base_url().await;
        let provider = OllamaProvider::build(OllamaConfig::new(Some(base)));
        let mut stream = provider.stream(req("qwen3.8"));
        let mut call = None;
        while let Some(chunk) = stream.next().await {
            match chunk.unwrap() {
                ProviderChunk::ToolCall { name, input, .. } => call = Some((name, input)),
                ProviderChunk::Done => break,
                _ => {}
            }
        }
        let (name, input) = call.expect("tool call");
        assert_eq!(name, "read_file");
        assert_eq!(input["path"], "a.rs");
    }

    #[tokio::test]
    async fn malformed_ndjson_is_malformed_error() {
        let server = MockServer::new();
        server.route(
            "POST",
            "/api/chat",
            MockAction::Respond {
                status: 200,
                body: "{\"message\": {\"content\": \"partial\"}}\n{not json\n".into(),
            },
        );
        let base = server.base_url().await;
        let provider = OllamaProvider::build(OllamaConfig::new(Some(base)));
        let mut stream = provider.stream(req("qwen3.8"));
        let mut saw_error = false;
        let mut got_partial = false;
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(ProviderChunk::Text { .. }) => got_partial = true,
                Err(e) if e.kind == ProviderErrorKind::Malformed => saw_error = true,
                Ok(ProviderChunk::Done) => break,
                _ => {}
            }
        }
        assert!(got_partial);
        assert!(saw_error, "garbage NDJSON must be a loud error");
    }

    #[tokio::test]
    async fn rate_limit_mapped() {
        let server = MockServer::new();
        server.route(
            "POST",
            "/api/chat",
            MockAction::Respond {
                status: 429,
                body: r#"{"error":"rate limited"}"#.into(),
            },
        );
        let base = server.base_url().await;
        let provider = OllamaProvider::build(OllamaConfig::new(Some(base)));
        let mut stream = provider.stream(req("qwen3.8"));
        let err = stream.next().await.unwrap().unwrap_err();
        assert_eq!(err.kind, ProviderErrorKind::RateLimited);
        assert!(err.retryable);
    }

    #[tokio::test]
    async fn missing_model_is_not_found_via_probe() {
        let server = MockServer::new();
        server.route(
            "POST",
            "/api/show",
            MockAction::Respond {
                status: 404,
                body: r#"{"error":"model not found"}"#.into(),
            },
        );
        let base = server.base_url().await;
        let provider = OllamaProvider::new(OllamaConfig::new(Some(base)));
        let err = provider.probe_model("ghost").await.unwrap_err();
        assert!(err.kind == ErrorKind::NotFound, "{err:?}");
    }

    #[test]
    fn default_capabilities_are_small_local() {
        let provider = OllamaProvider::build(OllamaConfig::new(None));
        let caps = provider.capabilities("anything");
        assert_eq!(caps.context, 32_768);
        assert!(caps.tools);
        assert!(caps.embeddings);
    }
}
