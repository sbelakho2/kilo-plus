//! kilop-openai — OpenAI Chat Completions and OpenAI-compatible endpoints
//! (spec §12). The adapter owns provider quirks; the agent never sees them.
//! The wire serializer produces exactly the frozen OpenAI shapes — internal
//! option names can never leak onto the wire (locked by tests).
//!
//! The Responses API is NOT implemented: selecting `OpenAiFamily::Responses`
//! fails honestly with an explicit error (use the Chat family).

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use futures::Stream;
use kilop_core::model::ModelCapabilities;
use kilop_provider::transport::{guarded_lines, utf8_line_stream, StreamDeadlines, MAX_LINE_BYTES};

/// Stream hang controls (audit round 9): first-byte / idle bounds from
/// the transport defaults. The overall bound stays 0 (disabled) — the
/// operation's own lifetime governs long generations; only deliberate
/// callers (tests, bounded proxies) set it.
fn stream_deadlines(_request: &GenericAgentRequest) -> StreamDeadlines {
    StreamDeadlines::default()
}
use kilop_provider::{
    ContentKind, ContentPart, GenericAgentRequest, Provider, ProviderChunk, ProviderError,
    ProviderErrorKind, ProviderStream, RequestMessage, Role,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAiFamily {
    /// POST /chat/completions (OpenAI, DeepSeek, most compatible servers).
    Chat,
    /// Unavailable: the Responses codec is not implemented. Selecting it
    /// yields an explicit error — nothing sends to POST /responses.
    Responses,
}

/// Adapter-level quirks that change wire lowering. DeepSeek-style endpoints
/// require the assistant's prior `reasoning_content` to be replayed on
/// subsequent tool iterations and a non-null `content` next to `tool_calls`;
/// plain OpenAI endpoints must never see those extensions. Defaults are the
/// plain OpenAI behavior (all flags off).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OpenAiQuirks {
    /// Replay assistant reasoning as a message-level `reasoning_content`
    /// field (Chat Completions style) instead of content blocks.
    pub requires_reasoning_replay_with_tools: bool,
    /// Always emit a `content` field (empty string when there is no text)
    /// on assistant messages that carry `tool_calls`.
    pub requires_assistant_content_with_tool_calls: bool,
}

#[derive(Debug, Clone)]
pub struct OpenAiConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    pub family: OpenAiFamily,
    /// Explicit capability overrides per model; defaults are conservative.
    pub models: HashMap<String, ModelCapabilities>,
}

impl OpenAiConfig {
    pub fn chat(base_url: impl Into<String>, api_key: Option<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key,
            family: OpenAiFamily::Chat,
            models: HashMap::new(),
        }
    }

    /// Selects the Responses family, which is NOT implemented: every stream
    /// fails with an explicit error ("use the Chat family").
    pub fn responses(base_url: impl Into<String>, api_key: Option<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key,
            family: OpenAiFamily::Responses,
            models: HashMap::new(),
        }
    }

    pub fn with_model(mut self, model: &str, caps: ModelCapabilities) -> Self {
        self.models.insert(model.to_string(), caps);
        self
    }

    pub fn with_default_caps(mut self, caps: ModelCapabilities) -> Self {
        self.models.insert("*".to_string(), caps);
        self
    }
}

/// A reqwest client with the adapter's standard connect timeout.
pub fn default_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
}

/// Authorization headers for a bearer API key (empty map when keyless).
pub fn authorization_headers(api_key: Option<&str>) -> reqwest::header::HeaderMap {
    let mut h = reqwest::header::HeaderMap::new();
    if let Some(key) = api_key {
        if let Ok(v) = format!("Bearer {key}").parse() {
            h.insert("authorization", v);
        }
    }
    h
}

pub struct OpenAiProvider {
    config: OpenAiConfig,
    client: reqwest::Client,
    quirks: OpenAiQuirks,
}

impl OpenAiProvider {
    pub fn build(config: OpenAiConfig) -> Arc<dyn Provider> {
        Self::build_with_quirks(config, OpenAiQuirks::default())
    }

    /// Build with adapter-level quirks (DeepSeek profiles set these; plain
    /// OpenAI endpoints keep the defaults).
    pub fn build_with_quirks(config: OpenAiConfig, quirks: OpenAiQuirks) -> Arc<dyn Provider> {
        Arc::new(Self {
            config,
            client: default_client(),
            quirks,
        })
    }

    fn wire_body(&self, req: &GenericAgentRequest) -> serde_json::Value {
        chat_completions_body(req, &self.quirks)
    }
}

// ------------------------------------------------------------ chat lowering

/// Emit accumulated tool-result messages. Consecutive tool results bound to
/// the SAME call id merge into one `role: "tool"` message (crash-retry
/// duplicates stay valid for the API); distinct call ids are separate
/// messages, in order.
fn flush_tool_results(out: &mut Vec<serde_json::Value>, chain: &mut Vec<(String, String)>) {
    for (call_id, content) in chain.drain(..) {
        out.push(serde_json::json!({
            "role": "tool",
            "tool_call_id": call_id,
            "content": content,
        }));
    }
}

/// Lower one generic message into a wire message (never into tool blocks:
/// Chat Completions assistant messages carry `tool_calls`, tool results are
/// `role: "tool"` messages, reasoning rides `reasoning_content` when the
/// endpoint requires replay).
fn lower_role_message(
    m: &RequestMessage,
    parts: &[&ContentPart],
    quirks: &OpenAiQuirks,
) -> serde_json::Value {
    match m.role {
        Role::System => {
            let mut content: Vec<serde_json::Value> = Vec::new();
            for p in parts.iter().copied() {
                if let ContentKind::Text { text } = &p.kind {
                    content.push(serde_json::json!({ "type": "text", "text": text }));
                }
            }
            serde_json::json!({ "role": "system", "content": content })
        }
        Role::User => {
            let mut content: Vec<serde_json::Value> = Vec::new();
            for p in parts.iter().copied() {
                match &p.kind {
                    ContentKind::Text { text } => {
                        content.push(serde_json::json!({ "type": "text", "text": text }));
                    }
                    ContentKind::Image { url } => {
                        content.push(serde_json::json!({
                            "type": "image_url",
                            "image_url": { "url": url }
                        }));
                    }
                    _ => {} // reasoning/tool parts are not user wire content
                }
            }
            serde_json::json!({ "role": "user", "content": content })
        }
        Role::Assistant => {
            let mut text: Vec<serde_json::Value> = Vec::new();
            let mut reasoning: Vec<String> = Vec::new();
            let mut calls: Vec<serde_json::Value> = Vec::new();
            for p in parts.iter().copied() {
                match &p.kind {
                    ContentKind::Text { text: t } => {
                        text.push(serde_json::json!({ "type": "text", "text": t }));
                    }
                    ContentKind::Reasoning { text: t } => reasoning.push(t.clone()),
                    ContentKind::ToolCall { id, name, input } => {
                        // arguments MUST be a JSON string of the object.
                        let arguments =
                            serde_json::to_string(&input).unwrap_or_else(|_| "{}".into());
                        calls.push(serde_json::json!({
                            "id": id,
                            "type": "function",
                            "function": { "name": name, "arguments": arguments }
                        }));
                    }
                    _ => {}
                }
            }
            let mut msg = serde_json::json!({ "role": "assistant" });
            if quirks.requires_reasoning_replay_with_tools {
                // DeepSeek-style: reasoning is replayed at message level and
                // never appears as a content block.
                if !reasoning.is_empty() {
                    msg["reasoning_content"] = serde_json::json!(reasoning.join("\n"));
                }
                let content = if text.is_empty() {
                    serde_json::Value::String(String::new())
                } else {
                    serde_json::Value::Array(text)
                };
                msg["content"] = content;
            } else if !calls.is_empty() {
                // Chat Completions: an assistant tool-calling message carries
                // TEXT-ONLY content blocks (plus tool_calls below) — reasoning
                // is skipped for families that do not replay it.
                msg["content"] = serde_json::Value::Array(text);
            } else {
                // Keep prior reasoning mapped as content blocks per the API
                // when the family supports them (no tool calls involved).
                for t in reasoning {
                    text.push(serde_json::json!({ "type": "reasoning", "text": t }));
                }
                msg["content"] = serde_json::Value::Array(text);
            }
            if !calls.is_empty() {
                msg["tool_calls"] = serde_json::Value::Array(calls);
            }
            msg
        }
    }
}

/// Lower the generic history into Chat Completions wire messages. Tool
/// results never become `{type: "tool_result"}` content blocks inside user
/// messages; they are separate `role: "tool"` messages.
fn lower_chat_messages(
    messages: &[RequestMessage],
    quirks: &OpenAiQuirks,
) -> Vec<serde_json::Value> {
    let mut out: Vec<serde_json::Value> = Vec::new();
    let mut tool_chain: Vec<(String, String)> = Vec::new();
    for m in messages {
        let mut rest: Vec<&ContentPart> = Vec::new();
        let mut results: Vec<&ContentPart> = Vec::new();
        for part in &m.content {
            if matches!(part.kind, ContentKind::ToolResult { .. }) {
                results.push(part);
            } else {
                rest.push(part);
            }
        }
        if !rest.is_empty() {
            flush_tool_results(&mut out, &mut tool_chain);
            out.push(lower_role_message(m, &rest, quirks));
            flush_tool_results(&mut out, &mut tool_chain);
        }
        for part in results {
            let content = match &part.kind {
                ContentKind::ToolResult { content, .. } => content.clone(),
                _ => continue,
            };
            let call_id = part.tool_call_id.clone().unwrap_or_default();
            if let Some((last_id, last_content)) = tool_chain.last_mut() {
                if *last_id == call_id {
                    last_content.push('\n');
                    last_content.push_str(&content);
                    continue;
                }
            }
            tool_chain.push((call_id, content));
        }
    }
    flush_tool_results(&mut out, &mut tool_chain);
    out
}

/// The POST /chat/completions body for a normalized request (public so the
/// gateway path builds the identical chat shape). Internal names can never
/// leak: the body is assembled field-by-field.
pub fn chat_completions_body(
    req: &GenericAgentRequest,
    quirks: &OpenAiQuirks,
) -> serde_json::Value {
    let messages = lower_chat_messages(&req.messages, quirks);
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
        "stream": req.stream,
    });
    if !tools.is_empty() {
        body["tools"] = serde_json::Value::Array(tools);
        body["tool_choice"] = serde_json::json!("auto");
    }
    if let Some(max_out) = req.max_output {
        body["max_tokens"] = serde_json::json!(max_out);
    }
    if let Some(reasoning) = req.reasoning {
        match reasoning {
            kilop_core::model::ReasoningMode::Off => {}
            kilop_core::model::ReasoningMode::Low => {
                body["reasoning_effort"] = serde_json::json!("low");
            }
            kilop_core::model::ReasoningMode::Medium => {
                body["reasoning_effort"] = serde_json::json!("medium");
            }
            kilop_core::model::ReasoningMode::High => {
                body["reasoning_effort"] = serde_json::json!("high");
            }
        }
    }
    body
}

impl Provider for OpenAiProvider {
    fn id(&self) -> &str {
        "openai"
    }

    fn known_models(&self) -> Vec<String> {
        let mut out: Vec<String> = self.config.models.keys().cloned().collect();
        if !out.contains(&"default".to_string()) {
            out.push("default".into());
        }
        out.sort();
        out
    }

    fn capabilities(&self, model: &str) -> ModelCapabilities {
        if let Some(caps) = self.config.models.get(model) {
            return caps.clone();
        }
        if let Some(caps) = self.config.models.get("*") {
            return caps.clone();
        }
        ModelCapabilities {
            context: 128_000,
            max_output: 16_384,
            tools: true,
            parallel_tools: true,
            thinking: false,
            vision: true,
            json_schema: true,
            streaming: true,
            embeddings: false,
            reasoning: false,
        }
    }

    fn stream(&self, req: GenericAgentRequest) -> ProviderStream {
        if self.config.family == OpenAiFamily::Responses {
            // The Responses codec is not implemented. Failing loudly beats
            // sending a chat-shaped body to /responses (the old behavior).
            return Box::pin(futures::stream::iter(vec![Err(ProviderError::new(
                ProviderErrorKind::Malformed,
                "OpenAI Responses API codec is not implemented; use the Chat family",
            ))]));
        }
        let body = self.wire_body(&req);
        let url = format!("{}/chat/completions", self.config.base_url);
        let client = self.client.clone();
        let headers = authorization_headers(self.config.api_key.as_deref());
        let deadlines = stream_deadlines(&req);
        let cancel = req.meta.cancellation.clone();
        Box::pin(openai_stream(
            client,
            url,
            headers,
            Vec::new(),
            body,
            deadlines,
            Some(cancel),
        ))
    }
}

/// Flush all accumulated tool-call fragments into the pending queue (index
/// order) and pop the next complete call, if any.
fn flush_and_pop(
    accs: &mut Vec<serde_json::Value>,
    pending: &mut std::collections::VecDeque<serde_json::Value>,
) -> Option<ProviderChunk> {
    if !accs.is_empty() {
        accs.sort_by_key(|a| a.get("index").and_then(|i| i.as_u64()).unwrap_or(0));
        pending.extend(accs.drain(..));
    }
    while let Some(tc) = pending.pop_front() {
        if let Some(chunk) = tool_chunk(&tc) {
            return Some(chunk);
        }
    }
    None
}

/// OpenAI SSE transport. `extra_headers` (name/value) are applied to the
/// request before send — used by the gateway path, empty elsewhere.
pub fn openai_stream(
    client: reqwest::Client,
    url: String,
    headers: reqwest::header::HeaderMap,
    extra_headers: Vec<(String, String)>,
    body: serde_json::Value,
    deadlines: StreamDeadlines,
    cancel: Option<kilop_core::cancellation::CancellationToken>,
) -> impl Stream<Item = Result<ProviderChunk, ProviderError>> {
    use futures::StreamExt as _;
    type LineStream = Pin<Box<dyn Stream<Item = Result<String, ProviderError>> + Send>>;

    // None = request not sent yet; Some = streaming lines. Tool-call
    // fragments accumulate PER INDEX (parallel calls never collide); the
    // pending queue drains complete calls in index order once a finishing
    // marker (finish_reason, [DONE], or stream end) appears.
    enum Stage {
        Fresh,
        Streaming {
            lines: LineStream,
            accs: Vec<serde_json::Value>,
            pending: std::collections::VecDeque<serde_json::Value>,
        },
        Done,
    }

    futures::stream::unfold(Stage::Fresh, move |stage| {
        let client = client.clone();
        let url = url.clone();
        let headers = headers.clone();
        let extra_headers = extra_headers.clone();
        let body = body.clone();
        let deadlines = deadlines;
        let cancel = cancel.clone();
        async move {
            // Lazily send the request on the first poll.
            let (mut lines, mut accs, mut pending) = match stage {
                Stage::Fresh => {
                    let mut extra = reqwest::header::HeaderMap::new();
                    for (name, value) in &extra_headers {
                        if let (Ok(k), Ok(v)) = (
                            reqwest::header::HeaderName::from_bytes(name.as_bytes()),
                            reqwest::header::HeaderValue::from_str(value),
                        ) {
                            extra.insert(k, v);
                        }
                    }
                    let resp = client
                        .post(&url)
                        .headers(headers)
                        .headers(extra)
                        .json(&body)
                        .send()
                        .await;
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
                            let lines: LineStream = Box::pin(guarded_lines(
                                utf8_line_stream(r.bytes_stream(), MAX_LINE_BYTES),
                                deadlines,
                                cancel.clone(),
                            ));
                            (lines, Vec::new(), std::collections::VecDeque::new())
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
                Stage::Streaming {
                    lines,
                    accs,
                    pending,
                } => (lines, accs, pending),
                Stage::Done => return None,
            };

            // Consume lines until a chunk is produced (or the stream ends).
            loop {
                if let Some(tc) = pending.pop_front() {
                    if let Some(chunk) = tool_chunk(&tc) {
                        return Some((
                            Ok(chunk),
                            Stage::Streaming {
                                lines,
                                accs,
                                pending,
                            },
                        ));
                    }
                    continue;
                }
                let Some(next) = lines.next().await else {
                    // Stream end: a server that never sent finish_reason must
                    // still complete its tool calls here.
                    if let Some(chunk) = flush_and_pop(&mut accs, &mut pending) {
                        return Some((
                            Ok(chunk),
                            Stage::Streaming {
                                lines,
                                accs,
                                pending,
                            },
                        ));
                    }
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
                if data == "[DONE]" {
                    if let Some(chunk) = flush_and_pop(&mut accs, &mut pending) {
                        return Some((
                            Ok(chunk),
                            Stage::Streaming {
                                lines,
                                accs,
                                pending,
                            },
                        ));
                    }
                    return Some((Ok(ProviderChunk::Done), Stage::Done));
                }
                let Ok(value) = serde_json::from_str::<serde_json::Value>(data) else {
                    return Some((
                        Err(ProviderError::new(
                            ProviderErrorKind::Malformed,
                            format!("bad SSE line: {data:?}"),
                        )),
                        Stage::Done,
                    ));
                };
                if let Some(chunk) = parse_chat_chunk(&value, &mut accs, &mut pending) {
                    let stage = match chunk {
                        ProviderChunk::Done => Stage::Done,
                        _ => Stage::Streaming {
                            lines,
                            accs,
                            pending,
                        },
                    };
                    return Some((Ok(chunk), stage));
                }
            }
        }
    })
}

/// Parse one SSE frame. Tool-call deltas accumulate PER `index` into `accs`
/// (parallel calls never clobber each other); a finishing marker
/// (`finish_reason: tool_calls|stop`) flushes complete calls into `pending`
/// and returns the first one. Frames without a chunk yield `None`.
fn parse_chat_chunk(
    value: &serde_json::Value,
    accs: &mut Vec<serde_json::Value>,
    pending: &mut std::collections::VecDeque<serde_json::Value>,
) -> Option<ProviderChunk> {
    if let Some(choices) = value.get("choices").and_then(|c| c.as_array()) {
        let choice = choices.first()?;
        let delta = choice.get("delta")?;
        if let Some(text) = delta.get("content").and_then(|c| c.as_str()) {
            if !text.is_empty() {
                return Some(ProviderChunk::Text {
                    text: text.to_string(),
                });
            }
        }
        if let Some(reasoning) = delta.get("reasoning_content").and_then(|c| c.as_str()) {
            if !reasoning.is_empty() {
                return Some(ProviderChunk::Reasoning {
                    text: reasoning.to_string(),
                });
            }
        }
        if let Some(tool_calls) = delta.get("tool_calls").and_then(|t| t.as_array()) {
            for tc in tool_calls {
                let index = tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0);
                let id = tc.get("id").and_then(|i| i.as_str()).unwrap_or_default();
                let name = tc
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
                    .unwrap_or_default();
                let arguments = tc
                    .get("function")
                    .and_then(|f| f.get("arguments"))
                    .and_then(|a| a.as_str())
                    .unwrap_or_default();
                // Index-keyed slot: fragments of index N never mix with
                // fragments of a simultaneous call at index M.
                let slot = accs
                    .iter()
                    .position(|a| a.get("index").and_then(|i| i.as_u64()) == Some(index));
                let slot = match slot {
                    Some(s) => s,
                    None => {
                        accs.push(serde_json::json!({
                            "index": index,
                            "id": if id.is_empty() {
                                format!("call_{index}")
                            } else {
                                id.to_string()
                            },
                            "name": String::new(),
                            "arguments": String::new(),
                        }));
                        accs.len() - 1
                    }
                };
                if !id.is_empty() {
                    accs[slot]["id"] = serde_json::json!(id);
                }
                if !name.is_empty() {
                    accs[slot]["name"] = serde_json::json!(name);
                }
                if !arguments.is_empty() {
                    let cur = accs[slot]["arguments"].as_str().unwrap_or("").to_string();
                    accs[slot]["arguments"] = serde_json::json!(format!("{cur}{arguments}"));
                }
            }
        }
        // A call completes ONLY at a finishing marker — fragments keep
        // accumulating until then (finish_reason may ride a frame that
        // carries no tool_calls at all).
        if let Some(reason) = choice.get("finish_reason").and_then(|r| r.as_str()) {
            if reason == "tool_calls" || reason == "stop" {
                return flush_and_pop(accs, pending);
            }
        }
    }
    if let Some(usage) = value.get("usage") {
        let tokens_in = usage
            .get("prompt_tokens")
            .and_then(|t| t.as_u64())
            .unwrap_or(0);
        let tokens_out = usage
            .get("completion_tokens")
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

fn tool_chunk(tc: &serde_json::Value) -> Option<ProviderChunk> {
    let id = tc
        .get("id")
        .and_then(|i| i.as_str())
        .unwrap_or("call_0")
        .to_string();
    let name = tc
        .get("name")
        .and_then(|n| n.as_str())
        .unwrap_or_default()
        .to_string();
    if name.is_empty() {
        return None;
    }
    let arguments = tc.get("arguments").and_then(|a| a.as_str()).unwrap_or("");
    let input = if arguments.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::from_str(arguments).unwrap_or(serde_json::Value::Null)
    };
    Some(ProviderChunk::ToolCall {
        id,
        name,
        input,
        complete: true,
    })
}

/// Build messages from a generic request (shared by compatible adapters).
pub fn messages_from(req: &GenericAgentRequest) -> Vec<RequestMessage> {
    req.messages.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use kilop_core::cancellation::CancellationToken;
    use kilop_core::id::{OpId, SessionId};
    use kilop_provider::testing::{sse_body, MockAction, MockServer};
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
            max_output: Some(1000),
            reasoning: None,
            stream: true,
            meta: RequestMeta {
                operation_id: OpId::new(1),
                session_id: SessionId::new(1),
                provider: "openai".into(),
                attempt: 0,
                deadline_ms: 5000,
                cancellation: CancellationToken::new(),
            },
        }
    }

    #[tokio::test]
    async fn wire_body_has_no_internal_leakage() {
        let server = MockServer::new();
        server.route(
            "POST",
            "/chat/completions",
            MockAction::AssertThenRespond {
                status: 200,
                body: sse_body(&[
                    serde_json::json!({"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}),
                ]),
                assert: Arc::new(|body: &serde_json::Value| {
                    // Frozen wire shape: exactly the OpenAI fields.
                    assert_eq!(body["model"], "m1");
                    assert!(body["messages"].is_array());
                    assert_eq!(body["messages"][0]["role"], "user");
                    assert!(body["stream"].as_bool().unwrap());
                    assert_eq!(body["max_tokens"], 1000);
                    assert_eq!(body["tools"][0]["type"], "function");
                    assert_eq!(body["tool_choice"], "auto");
                    // Internal fields must NEVER appear on the wire.
                    for leaked in ["operation_id", "session_id", "attempt", "deadline_ms", "cancellation", "system", "op_id"] {
                        assert!(!body.as_object().unwrap().contains_key(leaked), "{leaked} leaked!");
                    }
                    // The `system` prompt must not leak as a top-level field.
                    assert!(!body.as_object().unwrap().contains_key("system"));
                }),
            },
        );
        let base = server.base_url().await;
        let provider = OpenAiProvider::build(OpenAiConfig::chat(base, Some("sk-test".into())));
        let mut stream = provider.stream(req("m1"));
        let mut texts = String::new();
        while let Some(chunk) = stream.next().await {
            match chunk.unwrap() {
                ProviderChunk::Text { text } => texts.push_str(&text),
                ProviderChunk::Done => break,
                _ => {}
            }
        }
        assert_eq!(texts, "ok");
    }

    #[tokio::test]
    async fn tool_call_accumulates_and_completes() {
        let server = MockServer::new();
        let body = sse_body(&[
            serde_json::json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"read_file","arguments":"{\"path\":"}}]}}]}),
            serde_json::json!({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"a.rs\"}"}}]}}]}),
            serde_json::json!({"choices":[{"delta":{},"finish_reason":"tool_calls"}]}),
        ]);
        server.route(
            "POST",
            "/chat/completions",
            MockAction::Respond { status: 200, body },
        );
        let base = server.base_url().await;
        let provider = OpenAiProvider::build(OpenAiConfig::chat(base, None));
        let mut stream = provider.stream(req("m"));
        let mut call = None;
        while let Some(chunk) = stream.next().await {
            match chunk.unwrap() {
                ProviderChunk::ToolCall {
                    name,
                    input,
                    complete,
                    ..
                } => {
                    assert!(complete);
                    call = Some((name, input));
                }
                ProviderChunk::Done => break,
                _ => {}
            }
        }
        let (name, input) = call.expect("tool call emitted");
        assert_eq!(name, "read_file");
        assert_eq!(input["path"], "a.rs");
    }

    #[tokio::test]
    async fn rate_limit_and_auth_mapped() {
        let server = MockServer::new();
        server.route(
            "POST",
            "/chat/completions",
            MockAction::Respond {
                status: 429,
                body: r#"{"error":{"message":"rate limited"}}"#.into(),
            },
        );
        let base = server.base_url().await;
        let provider = OpenAiProvider::build(OpenAiConfig::chat(base, Some("k".into())));
        let mut stream = provider.stream(req("m"));
        let err = stream.next().await.unwrap().unwrap_err();
        assert_eq!(err.kind, ProviderErrorKind::RateLimited);
        assert!(err.retryable);
    }

    #[tokio::test]
    async fn malformed_sse_line_is_malformed_error() {
        let server = MockServer::new();
        server.route(
            "POST",
            "/chat/completions",
            MockAction::Respond {
                status: 200,
                body: "data: {not json}\n\ndata: [DONE]\n\n".into(),
            },
        );
        let base = server.base_url().await;
        let provider = OpenAiProvider::build(OpenAiConfig::chat(base, None));
        let mut stream = provider.stream(req("m"));
        let first = stream.next().await.unwrap();
        assert!(first.is_err(), "malformed SSE must be an error");
    }

    #[tokio::test]
    async fn stream_ends_without_done_still_terminates() {
        let server = MockServer::new();
        server.route(
            "POST",
            "/chat/completions",
            MockAction::Respond {
                status: 200,
                body: "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n".into(),
            },
        );
        let base = server.base_url().await;
        let provider = OpenAiProvider::build(OpenAiConfig::chat(base, None));
        let mut stream = provider.stream(req("m"));
        let mut done = false;
        let mut got_text = false;
        while let Some(chunk) = stream.next().await {
            match chunk.unwrap() {
                ProviderChunk::Text { .. } => got_text = true,
                ProviderChunk::Done => {
                    done = true;
                    break;
                }
                _ => {}
            }
        }
        assert!(done && got_text, "stream must terminate with Done");
    }

    #[tokio::test]
    async fn network_death_maps_to_network_error() {
        // No server listening on this port: connect error.
        let provider = OpenAiProvider::build(OpenAiConfig::chat("http://127.0.0.1:1", None));
        let mut stream = provider.stream(req("m"));
        let first = stream.next().await.unwrap();
        assert!(first.is_err());
        assert_eq!(first.unwrap_err().kind, ProviderErrorKind::Network);
    }

    #[test]
    fn capabilities_default_and_override() {
        let p = OpenAiProvider::build(OpenAiConfig::chat("http://x", None));
        let caps = p.capabilities("unknown-model");
        assert!(caps.tools);
        assert_eq!(caps.context, 128_000);
        let p = OpenAiProvider::build(OpenAiConfig::chat("http://x", None).with_model(
            "small",
            ModelCapabilities {
                context: 8192,
                tools: false,
                ..Default::default()
            },
        ));
        assert!(!p.capabilities("small").tools);
        assert_eq!(p.capabilities("small").context, 8192);
    }

    #[tokio::test]
    async fn responses_family_errors_honestly() {
        // The Responses codec is not implemented; the old adapter sent a
        // chat-shaped body to /responses. Selection must now fail loudly
        // and nothing may ever hit /responses.
        let server = MockServer::new();
        server.route(
            "POST",
            "/responses",
            MockAction::Respond {
                status: 200,
                body: "{}".into(),
            },
        );
        let base = server.base_url().await;
        let provider = OpenAiProvider::build(OpenAiConfig::responses(base, None));
        let mut stream = provider.stream(req("m"));
        let err = stream.next().await.unwrap().unwrap_err();
        assert_eq!(err.kind, ProviderErrorKind::Malformed);
        assert!(
            err.message
                .contains("OpenAI Responses API codec is not implemented")
                && err.message.contains("use the Chat family"),
            "error must name the honest remedy: {}",
            err.message
        );
        assert_eq!(server.request_count(), 0, "nothing may reach /responses");
    }

    #[tokio::test]
    async fn assistant_tool_call_and_tool_result_lower_to_wire_messages() {
        // (a) The exact request shape after a tool runs: the assistant turn
        // carries `tool_calls` (NOT {type:"tool_call"} content blocks) and
        // the result is a separate role:"tool" message (NOT a
        // {type:"tool_result"} block inside a user message).
        let server = MockServer::new();
        server.route(
            "POST",
            "/chat/completions",
            MockAction::AssertThenRespond {
                status: 200,
                body: String::new(),
                assert: Arc::new(|body: &serde_json::Value| {
                    let raw = serde_json::to_string(body).unwrap();
                    for banned in ["\"type\":\"tool_call\"", "\"type\":\"tool_result\""] {
                        assert!(
                            !raw.contains(banned),
                            "lowered body must not contain {banned}: {raw}"
                        );
                    }
                    let msgs = body["messages"].as_array().expect("messages array");
                    assert_eq!(msgs.len(), 2, "assistant + tool message");
                    assert_eq!(msgs[0]["role"], "assistant");
                    // content is text-only; the call rides tool_calls.
                    assert_eq!(msgs[0]["content"].as_array().unwrap().len(), 1);
                    assert_eq!(msgs[0]["content"][0]["type"], "text");
                    let tc = &msgs[0]["tool_calls"][0];
                    assert_eq!(tc["id"], "call_1");
                    assert_eq!(tc["type"], "function");
                    assert_eq!(tc["function"]["name"], "echo");
                    assert_eq!(
                        tc["function"]["arguments"], r#"{"x":1}"#,
                        "arguments must be the JSON STRING of the object"
                    );
                    assert_eq!(msgs[1]["role"], "tool");
                    assert_eq!(msgs[1]["tool_call_id"], "call_1");
                    assert_eq!(msgs[1]["content"], "echo: {\"x\":1}");
                    assert!(
                        msgs[1].get("tool_calls").is_none(),
                        "tool messages never carry tool_calls"
                    );
                }),
            },
        );
        let base = server.base_url().await;
        let provider = OpenAiProvider::build(OpenAiConfig::chat(base, None));
        let mut r = req("m");
        r.messages = vec![
            RequestMessage {
                role: Role::Assistant,
                content: vec![
                    ContentPart::text("calling now"),
                    ContentPart::tool_call("call_1", "echo", serde_json::json!({"x": 1})),
                ],
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
    async fn parallel_tool_calls_lower_to_two_tool_calls_entries() {
        // (b) One assistant turn with two parallel calls → two tool_calls
        // entries with distinct ids and stringified JSON arguments; content
        // stays text-only.
        let server = MockServer::new();
        server.route(
            "POST",
            "/chat/completions",
            MockAction::AssertThenRespond {
                status: 200,
                body: String::new(),
                assert: Arc::new(|body: &serde_json::Value| {
                    let raw = serde_json::to_string(body).unwrap();
                    for banned in ["\"type\":\"tool_call\"", "\"type\":\"tool_result\""] {
                        assert!(!raw.contains(banned), "no tool blocks allowed: {raw}");
                    }
                    let msgs = body["messages"].as_array().unwrap();
                    assert_eq!(msgs.len(), 1);
                    let calls = msgs[0]["tool_calls"].as_array().unwrap();
                    assert_eq!(calls.len(), 2);
                    let ids: Vec<&str> = calls.iter().map(|c| c["id"].as_str().unwrap()).collect();
                    assert_eq!(ids, vec!["call_a", "call_b"], "ids must stay distinct");
                    assert_eq!(calls[0]["function"]["name"], "read_file");
                    assert_eq!(calls[0]["function"]["arguments"], r#"{"path":"a.rs"}"#);
                    assert_eq!(calls[1]["function"]["name"], "list_dir");
                    assert_eq!(
                        calls[1]["function"]["arguments"],
                        r#"{"path":"src","depth":2}"#
                    );
                    let content = msgs[0]["content"].as_array().unwrap();
                    assert_eq!(content.len(), 1, "text-only content");
                    assert_eq!(content[0]["type"], "text");
                }),
            },
        );
        let base = server.base_url().await;
        let provider = OpenAiProvider::build(OpenAiConfig::chat(base, None));
        let mut r = req("m");
        r.messages = vec![RequestMessage {
            role: Role::Assistant,
            content: vec![
                ContentPart::text("two calls"),
                ContentPart::tool_call("call_a", "read_file", serde_json::json!({"path": "a.rs"})),
                ContentPart::tool_call(
                    "call_b",
                    "list_dir",
                    serde_json::json!({"path": "src", "depth": 2}),
                ),
            ],
        }];
        let mut stream = provider.stream(r);
        while let Some(chunk) = stream.next().await {
            if let Ok(ProviderChunk::Done) = chunk {
                break;
            }
        }
    }

    #[tokio::test]
    async fn two_index_keyed_tool_calls_accumulate_independently() {
        // (c) Two SIMULTANEOUS tool calls whose fragments interleave across
        // frames: each index accumulates its own arguments and both complete
        // at the finishing marker, in index order.
        let server = MockServer::new();
        let body = sse_body(&[
            serde_json::json!({"choices":[{"delta":{"tool_calls":[
                {"index":0,"id":"c1","function":{"name":"read_file","arguments":"{\"path\":"}},
                {"index":1,"id":"c2","function":{"name":"sum","arguments":"{\"nums\":"}}
            ]}}]}),
            serde_json::json!({"choices":[{"delta":{"tool_calls":[
                {"index":0,"function":{"arguments":"\"a.rs\""}},
                {"index":1,"function":{"arguments":"[1,2]"}}
            ]}}]}),
            serde_json::json!({"choices":[{"delta":{"tool_calls":[
                {"index":0,"function":{"arguments":"}"}},
                {"index":1,"function":{"arguments":"}"}}
            ]}}]}),
            serde_json::json!({"choices":[{"delta":{},"finish_reason":"tool_calls"}]}),
        ]);
        server.route(
            "POST",
            "/chat/completions",
            MockAction::Respond { status: 200, body },
        );
        let base = server.base_url().await;
        let provider = OpenAiProvider::build(OpenAiConfig::chat(base, None));
        let mut stream = provider.stream(req("m"));
        let mut calls: Vec<(String, serde_json::Value, bool)> = Vec::new();
        while let Some(chunk) = stream.next().await {
            match chunk.unwrap() {
                ProviderChunk::ToolCall {
                    id,
                    name: _,
                    input,
                    complete,
                } => calls.push((id, input, complete)),
                ProviderChunk::Done => break,
                _ => {}
            }
        }
        assert_eq!(calls.len(), 2, "both calls must complete: {calls:?}");
        assert!(
            calls.iter().all(|(_, _, complete)| *complete),
            "chunks appear only at the finishing marker, marked complete"
        );
        assert_eq!(calls[0].0, "c1", "index 0 flushes first");
        assert_eq!(calls[0].1["path"], "a.rs");
        assert_eq!(calls[1].0, "c2");
        assert_eq!(calls[1].1["nums"], serde_json::json!([1, 2]));
    }

    #[tokio::test]
    async fn sse_frame_split_across_http_chunks_assembles() {
        // Adversarial transport: a well-behaved server whose frame is
        // fragmented MID-LINE by HTTP chunking. The old per-chunk
        // `.lines()` code corrupted this into garbage lines.
        let server = MockServer::new();
        server.route(
            "POST",
            "/chat/completions",
            MockAction::ChunkedSse {
                status: 200,
                chunks: vec![
                    b"data: {\"choices\":[{\"delta\":{\"content\":\"par".to_vec(),
                    b"tial reply\"}}]}".to_vec(),
                    b"\n\n".to_vec(),
                    b"data: [DONE]\n\n".to_vec(),
                ],
            },
        );
        let base = server.base_url().await;
        let provider = OpenAiProvider::build(OpenAiConfig::chat(base, None));
        let mut stream = provider.stream(req("gpt-x"));
        let mut text = String::new();
        let mut done = false;
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(ProviderChunk::Text { text: t }) => text.push_str(&t),
                Ok(ProviderChunk::Done) => {
                    done = true;
                    break;
                }
                Err(e) => panic!("fragmented SSE must assemble, got {e:?}"),
                _ => {}
            }
        }
        assert!(done);
        assert_eq!(text, "partial reply");
    }

    #[tokio::test]
    async fn multibyte_rune_split_across_http_chunks_assembles() {
        // Split "héllo" between the two bytes of é across HTTP chunks.
        let server = MockServer::new();
        let e = "é".as_bytes();
        let mut c1 = b"data: {\"choices\":[{\"delta\":{\"content\":\"h".to_vec();
        c1.push(e[0]);
        let mut c2 = vec![e[1]];
        c2.extend_from_slice(b"llo\"}}]}");
        server.route(
            "POST",
            "/chat/completions",
            MockAction::ChunkedSse {
                status: 200,
                chunks: vec![c1, c2, b"\n\n".to_vec(), b"data: [DONE]\n\n".to_vec()],
            },
        );
        let base = server.base_url().await;
        let provider = OpenAiProvider::build(OpenAiConfig::chat(base, None));
        let mut stream = provider.stream(req("gpt-x"));
        let mut text = String::new();
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(ProviderChunk::Text { text: t }) => text.push_str(&t),
                Ok(ProviderChunk::Done) => break,
                Err(e) => panic!("split rune must assemble, got {e:?}"),
                _ => {}
            }
        }
        assert_eq!(text, "héllo");
    }

    #[tokio::test]
    async fn oversized_sse_line_is_loud_error_and_stream_ends() {
        // Hostile: one giant unbroken line. Bounded memory + loud error.
        let mut big = "data: ".to_string();
        big.extend(std::iter::repeat_n('x', 2 * 1024 * 1024));
        let server = MockServer::new();
        server.route(
            "POST",
            "/chat/completions",
            MockAction::ChunkedSse {
                status: 200,
                chunks: vec![big.into_bytes(), b"\n\ndata: [DONE]\n\n".to_vec()],
            },
        );
        let base = server.base_url().await;
        let provider = OpenAiProvider::build(OpenAiConfig::chat(base, None));
        let mut stream = provider.stream(req("gpt-x"));
        let mut saw_err = false;
        while let Some(chunk) = stream.next().await {
            match chunk {
                Err(e) if e.kind == ProviderErrorKind::Malformed => {
                    saw_err = true;
                    break;
                }
                Ok(_) => {}
                Err(e) => panic!("expected Malformed, got {e:?}"),
            }
        }
        assert!(saw_err, "oversized SSE line must be a loud Malformed error");
    }

    #[tokio::test]
    async fn silent_server_never_hangs_the_stream() {
        // Audit round 9 (P1): an already-connected provider that sends
        // nothing must surface a retryable Timeout — the transport guard's
        // first-byte deadline — instead of hanging the turn forever.
        let server = MockServer::new();
        server.route(
            "POST",
            "/chat/completions",
            MockAction::Silent { status: 200 },
        );
        let base = server.base_url().await;
        let client = reqwest::Client::new();
        let headers = authorization_headers(None);
        let deadlines = kilop_provider::transport::StreamDeadlines {
            first_byte_ms: 300,
            idle_ms: 300,
            overall_ms: 3000,
        };
        let body =
            serde_json::json!({"model": "gpt-x", "messages": [{"role": "user", "content": "hi"}]});
        let mut stream = Box::pin(openai_stream(
            client,
            format!("{base}/chat/completions"),
            headers,
            vec![],
            body,
            deadlines,
            None,
        ));
        let item = tokio::time::timeout(std::time::Duration::from_secs(10), stream.next())
            .await
            .expect("silent server must terminate via the transport guard")
            .expect("an error item");
        let err = item.expect_err("must be an error");
        assert!(
            matches!(err.kind, ProviderErrorKind::Timeout),
            "expected retryable timeout: {err:?}"
        );
    }
}
