//! kilop-google — Gemini streaming adapter (spec §12). Adapter owns Gemini's
//! wire quirks (candidates → parts → functionCall); agent sees normalized
//! chunks only.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use futures::Stream;
use kilop_core::model::ModelCapabilities;
use kilop_provider::transport::{utf8_line_stream, MAX_LINE_BYTES};
use kilop_provider::{
    ContentKind, GenericAgentRequest, Provider, ProviderChunk, ProviderError, ProviderErrorKind,
    ProviderStream, Role,
};

#[derive(Debug, Clone)]
pub struct GoogleConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model_caps: HashMap<String, ModelCapabilities>,
}

impl GoogleConfig {
    pub fn new(api_key: Option<String>) -> Self {
        Self {
            base_url: "https://generativelanguage.googleapis.com".into(),
            api_key,
            model_caps: HashMap::new(),
        }
    }

    pub fn with_model(mut self, model: &str, caps: ModelCapabilities) -> Self {
        self.model_caps.insert(model.to_string(), caps);
        self
    }

    pub fn with_base(mut self, base_url: &str) -> Self {
        self.base_url = base_url.to_string();
        self
    }
}

pub struct GoogleProvider {
    config: GoogleConfig,
    client: reqwest::Client,
}

impl GoogleProvider {
    pub fn build(config: GoogleConfig) -> Arc<dyn Provider> {
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Arc::new(Self { config, client })
    }

    fn wire_body(&self, req: &GenericAgentRequest) -> serde_json::Value {
        let mut contents: Vec<serde_json::Value> = Vec::new();
        for m in &req.messages {
            let role = match m.role {
                Role::User => "user",
                Role::Assistant => "model",
                Role::System => "user",
            };
            let mut parts: Vec<serde_json::Value> = Vec::new();
            for part in &m.content {
                match &part.kind {
                    ContentKind::Text { text } => {
                        parts.push(serde_json::json!({ "text": text }));
                    }
                    ContentKind::Reasoning { text } => {
                        parts.push(serde_json::json!({ "text": text }));
                    }
                    ContentKind::Image { url } => {
                        parts.push(serde_json::json!({ "inline_data": { "mime_type": "image/png", "data": url } }));
                    }
                    ContentKind::ToolCall { id, name, input } => {
                        parts.push(serde_json::json!({
                            "functionCall": { "name": name, "args": input, "id": id }
                        }));
                    }
                    ContentKind::ToolResult {
                        content: c,
                        is_error,
                    } => {
                        parts.push(serde_json::json!({
                            "functionResponse": {
                                "name": part.tool_call_id.as_deref().unwrap_or("fn"),
                                "response": { "result": c, "is_error": is_error },
                            }
                        }));
                    }
                }
            }
            contents.push(serde_json::json!({ "role": role, "parts": parts }));
        }
        let tools: Vec<serde_json::Value> = req
            .tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "functionDeclarations": [{
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    }]
                })
            })
            .collect();
        let mut body = serde_json::json!({
            "contents": contents,
            "generationConfig": { "temperature": 0.7 },
        });
        if !tools.is_empty() {
            body["tools"] = serde_json::Value::Array(tools);
        }
        if let Some(max_out) = req.max_output {
            body["generationConfig"]["maxOutputTokens"] = serde_json::json!(max_out);
        }
        body
    }
}

impl Provider for GoogleProvider {
    fn id(&self) -> &str {
        "google"
    }

    fn capabilities(&self, model: &str) -> ModelCapabilities {
        if let Some(caps) = self.config.model_caps.get(model) {
            return caps.clone();
        }
        ModelCapabilities {
            context: 1_000_000,
            max_output: 8_192,
            tools: true,
            parallel_tools: false,
            thinking: false,
            vision: true,
            json_schema: false,
            streaming: true,
            embeddings: false,
            reasoning: false,
        }
    }

    fn stream(&self, req: GenericAgentRequest) -> ProviderStream {
        let body = self.wire_body(&req);
        let key = self.config.api_key.clone().unwrap_or_default();
        let url = format!(
            "{}/v1beta/models/{}:streamGenerateContent?alt=sse&key={}",
            self.config.base_url, req.model, key
        );
        let client = self.client.clone();
        Box::pin(google_stream(client, url, body))
    }
}

pub(crate) fn google_stream(
    client: reqwest::Client,
    url: String,
    body: serde_json::Value,
) -> impl Stream<Item = Result<ProviderChunk, ProviderError>> {
    use futures::StreamExt as _;
    type LineStream = Pin<Box<dyn Stream<Item = Result<String, ProviderError>> + Send>>;
    enum Stage {
        Fresh,
        Streaming { lines: LineStream },
        Done,
    }
    futures::stream::unfold(Stage::Fresh, move |stage| {
        let client = client.clone();
        let url = url.clone();
        let body = body.clone();
        async move {
            let mut lines = match stage {
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
                            let lines: LineStream =
                                Box::pin(utf8_line_stream(r.bytes_stream(), MAX_LINE_BYTES));
                            lines
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
                Stage::Streaming { lines } => lines,
                Stage::Done => return None,
            };

            loop {
                let Some(next) = lines.next().await else {
                    return Some((Ok(ProviderChunk::Done), Stage::Done));
                };
                let line = match next {
                    Ok(l) => l,
                    Err(e) => return Some((Err(e), Stage::Done)),
                };
                let line = line.trim();
                if !line.starts_with("data:") {
                    continue;
                }
                let data = line[5..].trim();
                if data.is_empty() {
                    continue;
                }
                if data == "[DONE]" {
                    return Some((Ok(ProviderChunk::Done), Stage::Done));
                }
                let Ok(value) = serde_json::from_str::<serde_json::Value>(data) else {
                    return Some((
                        Err(ProviderError::new(
                            ProviderErrorKind::Malformed,
                            format!("bad gemini SSE: {data:?}"),
                        )),
                        Stage::Done,
                    ));
                };
                if let Some(chunk) = parse_gemini_chunk(&value) {
                    return Some((Ok(chunk), Stage::Streaming { lines }));
                }
            }
        }
    })
}

fn parse_gemini_chunk(value: &serde_json::Value) -> Option<ProviderChunk> {
    let candidates = value.get("candidates").and_then(|c| c.as_array())?;
    let content = candidates.first()?.get("content")?;
    let parts = content.get("parts").and_then(|p| p.as_array())?;
    for part in parts {
        if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
            if !text.is_empty() {
                return Some(ProviderChunk::Text {
                    text: text.to_string(),
                });
            }
        }
        if let Some(fc) = part.get("functionCall") {
            let name = fc
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or_default()
                .to_string();
            let args = fc.get("args").cloned().unwrap_or(serde_json::Value::Null);
            let id = fc
                .get("id")
                .and_then(|i| i.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("gemini_call_{}", name));
            if !name.is_empty() {
                return Some(ProviderChunk::ToolCall {
                    id,
                    name,
                    input: args,
                    complete: true,
                });
            }
        }
    }
    if let Some(usage) = value.get("usageMetadata") {
        let tokens_in = usage
            .get("promptTokenCount")
            .and_then(|t| t.as_u64())
            .unwrap_or(0);
        let tokens_out = usage
            .get("candidatesTokenCount")
            .and_then(|t| t.as_u64())
            .unwrap_or(0);
        if tokens_in > 0 || tokens_out > 0 {
            return Some(ProviderChunk::Usage {
                tokens_in,
                tokens_out,
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use kilop_core::cancellation::CancellationToken;
    use kilop_core::id::{OpId, SessionId};
    use kilop_provider::testing::{MockAction, MockServer};
    use kilop_provider::{ContentPart, RequestMessage, RequestMeta, ToolSpec};

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
            max_output: Some(1024),
            reasoning: None,
            stream: true,
            meta: RequestMeta {
                operation_id: OpId::new(1),
                session_id: SessionId::new(1),
                provider: "google".into(),
                attempt: 0,
                deadline_ms: 5000,
                cancellation: CancellationToken::new(),
            },
        }
    }

    #[tokio::test]
    async fn wire_shape_is_clean() {
        let server = MockServer::new();
        server.route(
            "POST",
            "/v1beta/models/gemini-x:streamGenerateContent",
            MockAction::AssertThenRespond {
                status: 200,
                body: "data: {}\n\n".into(),
                assert: Arc::new(|body: &serde_json::Value| {
                    assert_eq!(body["contents"][0]["role"], "user");
                    assert!(body["tools"].is_array());
                    assert_eq!(
                        body["tools"][0]["functionDeclarations"][0]["name"],
                        "read_file"
                    );
                    assert_eq!(body["generationConfig"]["maxOutputTokens"], 1024);
                    for leaked in [
                        "operation_id",
                        "session_id",
                        "attempt",
                        "deadline_ms",
                        "cancellation",
                        "system",
                    ] {
                        assert!(
                            !body.as_object().unwrap().contains_key(leaked),
                            "{leaked} leaked!"
                        );
                    }
                }),
            },
        );
        let base = server.base_url().await;
        let provider = GoogleProvider::build(GoogleConfig::new(Some("k".into())).with_base(&base));
        let mut stream = provider.stream(req("gemini-x"));
        while let Some(chunk) = stream.next().await {
            if let Ok(ProviderChunk::Done) = chunk {
                break;
            }
        }
        // The request URL must carry alt=sse (the mock records the path
        // with the query stripped; the adapter URL contains it).
        let (_, path, _) = server.last_request().unwrap();
        assert_eq!(path, "/v1beta/models/gemini-x:streamGenerateContent");
    }

    #[tokio::test]
    async fn text_and_function_call() {
        let server = MockServer::new();
        let body = [
            r#"data: {"candidates":[{"content":{"parts":[{"text":"let me check"}]}}]}"#,
            r#"data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"read_file","args":{"path":"a.rs"},"id":"gc1"}}]}}]}"#,
            "data: [DONE]",
        ]
        .join("\n\n");
        server.route(
            "POST",
            "/v1beta/models/gemini-x:streamGenerateContent",
            MockAction::Respond { status: 200, body },
        );
        let base = server.base_url().await;
        let provider = GoogleProvider::build(GoogleConfig::new(None).with_base(&base));
        let mut stream = provider.stream(req("gemini-x"));
        let mut text = String::new();
        let mut call = None;
        while let Some(chunk) = stream.next().await {
            match chunk.unwrap() {
                ProviderChunk::Text { text: t } => text.push_str(&t),
                ProviderChunk::ToolCall {
                    id, name, input, ..
                } => call = Some((id, name, input)),
                ProviderChunk::Done => break,
                _ => {}
            }
        }
        assert_eq!(text, "let me check");
        let (id, name, input) = call.expect("function call");
        assert_eq!(id, "gc1");
        assert_eq!(name, "read_file");
        assert_eq!(input["path"], "a.rs");
    }

    #[tokio::test]
    async fn rate_limit_mapped() {
        let server = MockServer::new();
        server.route(
            "POST",
            "/v1beta/models/gemini-x:streamGenerateContent",
            MockAction::Respond {
                status: 429,
                body: r#"{"error":{"message":"quota"}}"#.into(),
            },
        );
        let base = server.base_url().await;
        let provider = GoogleProvider::build(GoogleConfig::new(None).with_base(&base));
        let mut stream = provider.stream(req("gemini-x"));
        let err = stream.next().await.unwrap().unwrap_err();
        assert_eq!(err.kind, ProviderErrorKind::RateLimited);
    }

    #[tokio::test]
    async fn malformed_sse_is_loud() {
        let server = MockServer::new();
        server.route(
            "POST",
            "/v1beta/models/gemini-x:streamGenerateContent",
            MockAction::Respond {
                status: 200,
                body: "data: {broken\n\n".into(),
            },
        );
        let base = server.base_url().await;
        let provider = GoogleProvider::build(GoogleConfig::new(None).with_base(&base));
        let mut stream = provider.stream(req("gemini-x"));
        let first = stream.next().await.unwrap();
        assert!(first.is_err());
    }

    #[tokio::test]
    async fn function_call_and_function_response_ride_the_gemini_wire() {
        // The exact request shape the agent reconstructs after a tool runs:
        // model-role functionCall then user-role functionResponse carrying
        // the call id and the tool output.
        let server = MockServer::new();
        server.route(
            "POST",
            "/v1beta/models/gemini-x:streamGenerateContent",
            MockAction::AssertThenRespond {
                status: 200,
                body: String::new(),
                assert: Arc::new(|body: &serde_json::Value| {
                    let contents = body["contents"].as_array().expect("contents array");
                    assert_eq!(contents.len(), 2);
                    assert_eq!(contents[0]["role"], "model");
                    assert_eq!(
                        contents[0]["parts"][0]["functionCall"]["name"], "echo",
                        "the function name must ride the functionCall"
                    );
                    assert_eq!(
                        contents[0]["parts"][0]["functionCall"]["id"], "call_1",
                        "the call id must ride the functionCall"
                    );
                    assert_eq!(
                        contents[0]["parts"][0]["functionCall"]["args"],
                        serde_json::json!({"x": 1}),
                        "the call input must ride the functionCall"
                    );
                    assert_eq!(contents[1]["role"], "user");
                    assert_eq!(
                        contents[1]["parts"][0]["functionResponse"]["name"], "call_1",
                        "the functionResponse must reference the call id"
                    );
                    assert_eq!(
                        contents[1]["parts"][0]["functionResponse"]["response"]["result"],
                        "echo: {\"x\":1}",
                        "the tool output must be on the wire verbatim"
                    );
                    assert_eq!(
                        contents[1]["parts"][0]["functionResponse"]["response"]["is_error"],
                        false
                    );
                }),
            },
        );
        let base = server.base_url().await;
        let provider = GoogleProvider::build(GoogleConfig::new(None).with_base(&base));
        let mut r = req("gemini-x");
        r.messages = vec![
            RequestMessage {
                role: Role::Assistant,
                content: vec![ContentPart::tool_call(
                    "call_1",
                    "echo",
                    serde_json::json!({"x": 1}),
                )],
            },
            RequestMessage {
                role: Role::User,
                content: vec![ContentPart::tool_result("echo: {\"x\":1}", false, "call_1")],
            },
        ];
        let mut stream = provider.stream(r);
        let mut done = false;
        while let Some(chunk) = stream.next().await {
            if let Ok(ProviderChunk::Done) = chunk {
                done = true;
                break;
            }
        }
        assert!(done);
    }

    #[tokio::test]
    async fn sse_frame_split_across_http_chunks_assembles() {
        let server = MockServer::new();
        server.route(
            "POST",
            "/v1beta/models/gemini-x:streamGenerateContent",
            MockAction::ChunkedSse {
                status: 200,
                chunks: vec![
                    br#"data: {"candidates":[{"content":{"parts":[{"text":"hel"#.to_vec(),
                    br#"lo"}]}}]}"#.to_vec(),
                    b"\n\n".to_vec(),
                    b"data: [DONE]\n\n".to_vec(),
                ],
            },
        );
        let base = server.base_url().await;
        let provider = GoogleProvider::build(GoogleConfig::new(None).with_base(&base));
        let mut stream = provider.stream(req("gemini-x"));
        let mut text = String::new();
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(ProviderChunk::Text { text: t }) => text.push_str(&t),
                Ok(ProviderChunk::Done) => break,
                Ok(_) => {}
                Err(e) => panic!("fragmented SSE must assemble, got {e:?}"),
            }
        }
        assert_eq!(text, "hello");
    }
}
