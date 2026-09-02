//! kilop-ollama — native Ollama adapter (spec §10).
//!
//! Discovery via `GET /api/tags` (never a hard-coded list — `ollama pull
//! qwen3.8` makes it appear automatically), capability probing via
//! `GET /api/show`, native `/api/chat` streaming with tools/thinking/
//! keep_alive, and `/api/embed` embeddings. OpenAI-compatible mode is a
//! fallback only.
//!
//! Wire shapes are Ollama-native (`docs/api.md`), never OpenAI-style
//! content arrays: `message` objects carry `role`, a plain-string
//! `content`, optional `thinking`/`images`/`tool_calls`; a tool round trip
//! is an assistant message with `tool_calls` followed by role-`tool`
//! messages; the thinking knob is the top-level `/api/chat` `think`
//! parameter (boolean or `"low"`/`"medium"`/`"high"` level).

use std::collections::{HashMap, VecDeque};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use futures::Stream;
use kilop_core::error::{Error, ErrorKind};
use kilop_core::model::{ModelCapabilities, ReasoningMode};
use kilop_provider::transport::{utf8_line_stream, MAX_LINE_BYTES};
use kilop_provider::{
    ContentKind, GenericAgentRequest, Provider, ProviderChunk, ProviderError, ProviderErrorKind,
    ProviderStream, RequestMessage, Role,
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

/// Context-window facts for one model: the model's maximum (from
/// `/api/show` — the same number `probe_model` reports as
/// `ModelCapabilities::context`) and the runtime-effective window after the
/// `/api/ps` allocation is applied: `min(model_max, allocated)`, `None`
/// while the model is not loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OllamaContext {
    pub model_max: usize,
    pub runtime_effective: Option<usize>,
}

pub struct OllamaProvider {
    config: OllamaConfig,
    client: reqwest::Client,
    /// Live-probed capabilities (spec §10: discovery/probing drive behavior
    /// — never a hard-coded list). Written by [`refresh_from_live`] and read
    /// by `capabilities()`; empty until the daemon warms the provider.
    probed: std::sync::RwLock<HashMap<String, ModelCapabilities>>,
    /// Per-provider-response sequence: tool ids are
    /// `ollama:<response_seq>:<tool_index>` so ids stay unique across every
    /// response a session streams (see [`OllamaProvider::stream`]).
    response_seq: AtomicU64,
}

impl OllamaProvider {
    /// Concrete constructor (for discovery/probing APIs).
    pub fn new(config: OllamaConfig) -> Arc<Self> {
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Arc::new(Self {
            config,
            client,
            probed: std::sync::RwLock::new(HashMap::new()),
            response_seq: AtomicU64::new(0),
        })
    }

    /// Live capability warm-up (spec §10): `GET /api/tags` discovers the
    /// installed models, each is probed via `GET /api/show`, and the results
    /// drive `capabilities()` from then on. Bounded: at most
    /// `MAX_PROBED_MODELS` models, 10s per probe. A failed probe leaves the
    /// model on its default capabilities; discovery failure surfaces.
    pub async fn refresh_from_live(&self) -> Result<usize, Error> {
        const MAX_PROBED_MODELS: usize = 64;
        let models = self.discover_models().await?;
        let mut map: HashMap<String, ModelCapabilities> = HashMap::new();
        for model in models.iter().take(MAX_PROBED_MODELS) {
            match tokio::time::timeout(std::time::Duration::from_secs(10), self.probe_model(model))
                .await
            {
                Ok(Ok(caps)) => {
                    map.insert(model.clone(), caps);
                }
                Ok(Err(e)) => {
                    tracing::warn!("ollama probe {model} failed: {e}");
                }
                Err(_) => {
                    tracing::warn!("ollama probe {model} timed out");
                }
            }
        }
        let n = map.len();
        *self.probed.write().unwrap() = map;
        Ok(n)
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

    /// `GET /api/ps`: the context window currently ALLOCATED for `model` in
    /// the loaded process. `Ok(None)` when the model is not loaded (or the
    /// daemon does not report an allocation); hostile bodies are loud
    /// errors, never a panic.
    pub async fn ps_allocated(&self, model: &str) -> Result<Option<usize>, Error> {
        let resp = self
            .client
            .get(format!("{}/api/ps", self.config.base_url))
            .send()
            .await
            .map_err(|e| Error::new(ErrorKind::Network, format!("ollama ps: {e}")))?;
        if !resp.status().is_success() {
            return Err(Error::new(
                ErrorKind::Provider {
                    code: resp.status().as_u16().to_string(),
                    retryable: false,
                },
                format!("ollama /api/ps returned {}", resp.status()),
            ));
        }
        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| Error::new(ErrorKind::Malformed, format!("ollama ps body: {e}")))?;
        Ok(ps_allocated_context(&body, model)?.map(|c| c as usize))
    }

    /// Probe both context numbers for `model`: the model maximum
    /// (`/api/show`, exactly what `probe_model` reports as
    /// `ModelCapabilities::context`) and the runtime-effective window
    /// (`/api/ps`, `min(model_max, allocated)`; `None` while unloaded).
    pub async fn runtime_context(&self, model: &str) -> Result<OllamaContext, Error> {
        let caps = self.probe_model(model).await?;
        let allocated = self.ps_allocated(model).await?;
        let runtime_effective = allocated.map(|allocated| caps.context.min(allocated));
        Ok(OllamaContext {
            model_max: caps.context,
            runtime_effective,
        })
    }

    /// Effective context = `min(model max from /api/show, allocated from
    /// /api/ps)`; `Ok(None)` when the model is not currently loaded.
    pub async fn effective_context(&self, model: &str) -> Result<Option<usize>, Error> {
        Ok(self.runtime_context(model).await?.runtime_effective)
    }

    /// Native wire serializer (P0 "Qwen3.8 not first-class"): the generic
    /// content-block model is lowered into real Ollama message objects,
    /// never OpenAI-style `{"type": ...}` content arrays. Shapes follow
    /// Ollama's `/api/chat` contract (`docs/api.md`): each message carries
    /// `role` + plain-string `content`, optional `thinking` (assistant
    /// reasoning), optional base64 `images`, optional `tool_calls`
    /// (`{"function": {"name", "arguments"}}`); tool results are role
    /// `tool` messages (with the documented `tool_name` when the answered
    /// call is visible in this request). `GenericAgentRequest::system`
    /// becomes the first `role: system` message; images are omitted when
    /// the request carries none.
    fn wire_body(&self, req: &GenericAgentRequest) -> serde_json::Value {
        let mut messages: Vec<serde_json::Value> = Vec::new();
        if !req.system.is_empty() {
            messages.push(serde_json::json!({ "role": "system", "content": req.system }));
        }
        // tool_call id -> tool name seen so far in this request: role-"tool"
        // messages can carry the native tool_name only when the call it
        // answers was part of this request.
        let mut call_names: HashMap<String, String> = HashMap::new();
        for m in &req.messages {
            messages.extend(lower_native_message(m, &mut call_names));
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
        // Native thinking knob: driven by the request's stored
        // ReasoningMode (never guessed from text). Off -> think:false;
        // Low/Medium/High -> the documented effort level when the model
        // profile indicates effort-level reasoning (caps.reasoning), else
        // the boolean knob (caps.thinking); omitted entirely when the
        // profile supports neither.
        if let Some(mode) = req.reasoning {
            if let Some(knob) = thinking_knob(mode, &self.capabilities(&req.model)) {
                body["think"] = knob;
            }
        }
        body
    }
}

impl Provider for OllamaProvider {
    fn id(&self) -> &str {
        "ollama"
    }

    fn known_models(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for k in self.config.model_overrides.keys() {
            out.push(k.clone());
        }
        for k in self.probed.read().unwrap().keys() {
            if !out.contains(k) {
                out.push(k.clone());
            }
        }
        if !out.contains(&"default".to_string()) {
            out.push("default".into());
        }
        out.sort();
        out
    }

    fn capabilities(&self, model: &str) -> ModelCapabilities {
        if let Some(caps) = self.config.model_overrides.get(model) {
            return caps.clone();
        }
        if let Some(caps) = self.probed.read().unwrap().get(model) {
            return caps.clone();
        }
        // Default profile: small local models are the norm (spec §9/§10);
        // a live probe replaces this once the daemon warms the provider.
        ModelCapabilities::small_local()
    }

    fn stream(&self, req: GenericAgentRequest) -> ProviderStream {
        let body = self.wire_body(&req);
        let url = format!("{}/api/chat", self.config.base_url);
        let client = self.client.clone();
        // One provider response per stream call: the response sequence
        // increments monotonically so tool ids (ollama:<seq>:<idx>) never
        // collide across responses of the same provider instance.
        let response_seq = self.response_seq.fetch_add(1, Ordering::Relaxed);
        Box::pin(ollama_chat_stream(client, url, body, response_seq))
    }
}

pub(crate) fn ollama_chat_stream(
    client: reqwest::Client,
    url: String,
    body: serde_json::Value,
    response_seq: u64,
) -> impl Stream<Item = Result<ProviderChunk, ProviderError>> {
    use futures::StreamExt as _;
    type LineStream = Pin<Box<dyn Stream<Item = Result<String, ProviderError>> + Send>>;
    enum Stage {
        Fresh,
        Streaming {
            lines: LineStream,
            /// Chunks produced by ONE parsed frame, drained one per poll so
            /// a frame carrying N tool calls yields N ToolCall chunks in
            /// order (the old loop dropped every call after the first).
            pending: VecDeque<ProviderChunk>,
            /// Frame with `done: true` seen; emit Done once pending drains.
            finished: bool,
        },
        Done,
    }
    futures::stream::unfold(Stage::Fresh, move |stage| {
        let client = client.clone();
        let url = url.clone();
        let body = body.clone();
        async move {
            let (mut lines, mut pending, mut finished) = match stage {
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
                            (lines, VecDeque::new(), false)
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
                    pending,
                    finished,
                } => (lines, pending, finished),
                Stage::Done => return None,
            };

            loop {
                // Drain one frame's chunks before reading the next line.
                if let Some(chunk) = pending.pop_front() {
                    return Some((
                        Ok(chunk),
                        Stage::Streaming {
                            lines,
                            pending,
                            finished,
                        },
                    ));
                }
                if finished {
                    return Some((Ok(ProviderChunk::Done), Stage::Done));
                }
                let Some(next) = lines.next().await else {
                    return Some((Ok(ProviderChunk::Done), Stage::Done));
                };
                let line = match next {
                    Ok(l) => l,
                    Err(e) => return Some((Err(e), Stage::Done)),
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
                let chunks = parse_ollama_frame(&value, response_seq);
                pending = VecDeque::from(chunks);
                // A `done: true` frame only terminates the stream when it
                // carried no chunk: a final frame can still hold tool_calls
                // (or content), and a hostile server may keep sending.
                if pending.is_empty() && value.get("done").and_then(|d| d.as_bool()) == Some(true) {
                    finished = true;
                }
            }
        }
    })
}

/// One native NDJSON frame -> 0..n chunks, in the order they belong to the
/// conversation: `message.thinking` (a Reasoning chunk) before `content`
/// (a Text chunk), then one ToolCall chunk per `tool_calls` entry in array
/// order. Tool ids: a provider-supplied `id` wins; otherwise
/// `ollama:<response_seq>:<tool_index>` — never a synthesized
/// `ollama_call_<name>` (the old ids collided across calls and responses).
fn parse_ollama_frame(value: &serde_json::Value, response_seq: u64) -> Vec<ProviderChunk> {
    let mut chunks: Vec<ProviderChunk> = Vec::new();
    let Some(msg) = value.get("message") else {
        return chunks;
    };
    if let Some(thinking) = msg.get("thinking").and_then(|t| t.as_str()) {
        if !thinking.is_empty() {
            chunks.push(ProviderChunk::Reasoning {
                text: thinking.to_string(),
            });
        }
    }
    if let Some(text) = msg.get("content").and_then(|c| c.as_str()) {
        if !text.is_empty() {
            chunks.push(ProviderChunk::Text {
                text: text.to_string(),
            });
        }
    }
    if let Some(tool_calls) = msg.get("tool_calls").and_then(|t| t.as_array()) {
        for (idx, tc) in tool_calls.iter().enumerate() {
            let function = tc.get("function");
            let name = function
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or_default();
            if name.is_empty() {
                continue;
            }
            let args = function
                .and_then(|f| f.get("arguments"))
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            let id = tc
                .get("id")
                .and_then(|i| i.as_str())
                .filter(|i| !i.is_empty())
                .map(str::to_string)
                .unwrap_or_else(|| format!("ollama:{response_seq}:{idx}"));
            chunks.push(ProviderChunk::ToolCall {
                id,
                name: name.to_string(),
                input: args,
                complete: true,
            });
        }
    }
    chunks
}

fn caps_from_show(model: &str, show: &ShowResponse) -> ModelCapabilities {
    let mut caps = ModelCapabilities::small_local();
    // Context length from model_info: real /api/show responses carry
    // architecture-prefixed keys ("qwen3.context_length", never a bare
    // "context_length") — see context_length_from_model_info.
    if let Some(info) = &show.model_info {
        if let Some(ctx) = context_length_from_model_info(info) {
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
        // A probed "reasoning" capability means the model accepts effort
        // LEVELS for the native think knob ("low"/"medium"/"high");
        // boolean-only thinking models never advertise it.
        caps.reasoning = has("reasoning");
    }
    // qwen3.8-family models advertise large contexts; never exceed the
    // model's own reported context.
    let _ = model;
    caps
}

/// Context length from a parsed `/api/show` `model_info` value.
///
/// Real responses carry architecture-prefixed GGUF metadata:
/// `{"general.architecture": "qwen3", "qwen3.context_length": 262144}`
/// (newer daemons also nest: `{"general": {"architecture": "qwen3"}}`).
/// Lookup order: read the architecture from the `general` section (nested
/// or dotted), then `<arch>.context_length` (dotted or nested); fall back
/// to the unique key ending in `.context_length` (or the bare
/// `context_length` as a last resort). Ambiguity — several context keys
/// with no declared architecture — is NOT guessed.
fn context_length_from_model_info(info: &serde_json::Value) -> Option<u64> {
    let arch = info
        .get("general")
        .and_then(|g| g.get("architecture"))
        .and_then(|a| a.as_str())
        .or_else(|| info.get("general.architecture").and_then(|a| a.as_str()));
    if let Some(arch) = arch {
        if let Some(ctx) = info
            .get(format!("{arch}.context_length"))
            .and_then(|c| c.as_u64())
        {
            return Some(ctx);
        }
        if let Some(ctx) = info
            .get(arch)
            .and_then(|section| section.get("context_length"))
            .and_then(|c| c.as_u64())
        {
            return Some(ctx);
        }
    }
    let mut candidates: Vec<u64> = Vec::new();
    if let serde_json::Value::Object(map) = info {
        for (k, v) in map {
            if k == "context_length" || k.ends_with(".context_length") {
                if let Some(ctx) = v.as_u64() {
                    candidates.push(ctx);
                }
            }
        }
    }
    if candidates.len() == 1 {
        Some(candidates[0])
    } else {
        None
    }
}

/// Allocated context for `model` from a parsed `GET /api/ps` body. Real
/// loaded-model entries report the allocated window under
/// `details.context_length` (never `size`/`size_vram` — those are bytes,
/// not tokens). Matching tolerates a missing tag: entry name equals the
/// model name or extends it as `name:<tag>`. Hostile bodies are loud
/// errors; an unloaded model is `Ok(None)`; a model entry without a
/// reported allocation is `Ok(None)`. Never panics.
fn ps_allocated_context(body: &serde_json::Value, model: &str) -> Result<Option<u64>, Error> {
    let Some(models) = body.get("models") else {
        return Err(Error::new(
            ErrorKind::Malformed,
            "ollama /api/ps body lacks a models array",
        ));
    };
    let Some(models) = models.as_array() else {
        return Err(Error::new(
            ErrorKind::Malformed,
            "ollama /api/ps models is not an array",
        ));
    };
    for entry in models {
        let Some(entry) = entry.as_object() else {
            return Err(Error::new(
                ErrorKind::Malformed,
                "ollama /api/ps entry is not an object",
            ));
        };
        let Some(name) = entry.get("name").and_then(|n| n.as_str()) else {
            return Err(Error::new(
                ErrorKind::Malformed,
                "ollama /api/ps entry lacks a string name",
            ));
        };
        let matches = name == model
            || name
                .strip_prefix(model)
                .is_some_and(|rest| rest.starts_with(':'));
        if !matches {
            continue;
        }
        let allocated = entry
            .get("details")
            .and_then(|d| d.get("context_length"))
            .and_then(|c| c.as_u64())
            .or_else(|| entry.get("context_length").and_then(|c| c.as_u64()));
        return Ok(allocated);
    }
    Ok(None)
}

/// One in-flight native Ollama message object being assembled. Content
/// parts of the same target role coalesce (assistant messages may carry
/// content + thinking + tool_calls at once); a part that targets a
/// different role (or a tool result) flushes it first, so the emitted
/// message order always mirrors the generic part order.
#[derive(Default)]
struct NativeMessage {
    role: &'static str,
    content: Vec<String>,
    thinking: Vec<String>,
    images: Vec<String>,
    tool_calls: Vec<serde_json::Value>,
}

impl NativeMessage {
    fn new(role: &'static str) -> Self {
        Self {
            role,
            ..Self::default()
        }
    }
}

fn role_name(role: &Role) -> &'static str {
    match role {
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::System => "system",
    }
}

fn flush_native(acc: &mut Option<NativeMessage>, out: &mut Vec<serde_json::Value>) {
    let Some(a) = acc.take() else { return };
    let mut obj = serde_json::Map::new();
    obj.insert("role".into(), serde_json::json!(a.role));
    obj.insert("content".into(), serde_json::json!(a.content.join("\n")));
    if !a.thinking.is_empty() {
        obj.insert("thinking".into(), serde_json::json!(a.thinking.join("\n")));
    }
    if !a.images.is_empty() {
        obj.insert("images".into(), serde_json::json!(a.images));
    }
    if !a.tool_calls.is_empty() {
        obj.insert("tool_calls".into(), serde_json::Value::Array(a.tool_calls));
    }
    out.push(serde_json::Value::Object(obj));
}

fn switch_role(
    acc: &mut Option<NativeMessage>,
    out: &mut Vec<serde_json::Value>,
    role: &'static str,
) {
    if acc.as_ref().is_some_and(|a| a.role != role) {
        flush_native(acc, out);
    }
    if acc.is_none() {
        *acc = Some(NativeMessage::new(role));
    }
}

/// Ollama's native /api/chat expects raw base64 in `images`. Internal
/// image parts may carry a `data:<mime>;base64,` URI — strip the prefix;
/// anything else passes through untouched.
fn native_image_payload(url: &str) -> String {
    let Some(rest) = url.strip_prefix("data:") else {
        return url.to_string();
    };
    match rest.split_once(',') {
        Some((_, payload)) => payload.to_string(),
        None => url.to_string(),
    }
}

/// Lower one generic request message into 0..n native Ollama message
/// objects, preserving part order:
///
/// - text parts become plain `content` on the declared role (user text ->
///   user content; assistant text -> assistant content);
/// - assistant reasoning parts become the message's native `thinking`
///   field; reasoning recorded under any other role is lifted onto a
///   role-assistant message (only assistant messages may carry thinking);
/// - `ToolCall` parts become native `{"function": {"name", "arguments"}}`
///   entries on an assistant message (`tool_calls` exist nowhere else);
///   their ids are remembered so results can name the call they answer;
/// - `ToolResult` parts become role-`tool` messages with the raw content,
///   plus the documented `tool_name` when the answered call was seen
///   earlier in this request. Ollama's native tool message has no
///   `is_error`/`tool_call_id` fields; content passes through verbatim.
fn lower_native_message(
    m: &RequestMessage,
    call_names: &mut HashMap<String, String>,
) -> Vec<serde_json::Value> {
    let mut out: Vec<serde_json::Value> = Vec::new();
    let mut acc: Option<NativeMessage> = None;
    for part in &m.content {
        match &part.kind {
            ContentKind::ToolResult { content, .. } => {
                flush_native(&mut acc, &mut out);
                let mut tm = serde_json::json!({ "role": "tool", "content": content });
                if let Some(cid) = part.tool_call_id.as_deref() {
                    if let Some(name) = call_names.get(cid) {
                        tm["tool_name"] = serde_json::json!(name);
                    }
                }
                out.push(tm);
            }
            ContentKind::ToolCall { id, name, input } => {
                switch_role(&mut acc, &mut out, "assistant");
                let a = acc.as_mut().expect("switch_role guarantees an accumulator");
                a.tool_calls.push(serde_json::json!({
                    "function": { "name": name, "arguments": input }
                }));
                if !id.is_empty() {
                    call_names.insert(id.clone(), name.clone());
                }
            }
            ContentKind::Reasoning { text } => {
                switch_role(&mut acc, &mut out, "assistant");
                let a = acc.as_mut().expect("switch_role guarantees an accumulator");
                a.thinking.push(text.clone());
            }
            ContentKind::Text { text } => {
                switch_role(&mut acc, &mut out, role_name(&m.role));
                let a = acc.as_mut().expect("switch_role guarantees an accumulator");
                a.content.push(text.clone());
            }
            ContentKind::Image { url } => {
                switch_role(&mut acc, &mut out, role_name(&m.role));
                let a = acc.as_mut().expect("switch_role guarantees an accumulator");
                a.images.push(native_image_payload(url));
            }
        }
    }
    flush_native(&mut acc, &mut out);
    out
}

/// Native /api/chat thinking knob driven by the request's stored
/// ReasoningMode: `Off` -> `think: false`; `Low`/`Medium`/`High` -> the
/// documented effort level string when the model profile indicates
/// effort-level reasoning (`ModelCapabilities::reasoning`), else the
/// boolean knob for boolean-thinking models (`ModelCapabilities::thinking`).
/// A profile supporting neither leaves the knob absent (server default).
fn thinking_knob(mode: ReasoningMode, caps: &ModelCapabilities) -> Option<serde_json::Value> {
    match mode {
        ReasoningMode::Off => Some(serde_json::json!(false)),
        ReasoningMode::Low => effort_or_boolean_knob("low", caps),
        ReasoningMode::Medium => effort_or_boolean_knob("medium", caps),
        ReasoningMode::High => effort_or_boolean_knob("high", caps),
    }
}

fn effort_or_boolean_knob(effort: &str, caps: &ModelCapabilities) -> Option<serde_json::Value> {
    if caps.reasoning {
        Some(serde_json::json!(effort))
    } else if caps.thinking {
        Some(serde_json::json!(true))
    } else {
        None
    }
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

    /// Drain a provider stream, panicking on any error (tests that assert
    /// error paths collect the chunks inline instead).
    async fn stream_chunks(
        provider: &dyn Provider,
        request: GenericAgentRequest,
    ) -> Vec<ProviderChunk> {
        let mut stream = provider.stream(request);
        let mut out = Vec::new();
        while let Some(chunk) = stream.next().await {
            out.push(chunk.unwrap_or_else(|e| panic!("unexpected provider error: {e:?}")));
        }
        out
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
                // Real /api/show metadata: architecture-prefixed keys, not a
                // bare "context_length".
                body: r#"{
                    "model_info": {
                        "general.architecture": "qwen3",
                        "qwen3.context_length": 262144
                    },
                    "capabilities": ["tools", "vision", "embeddings", "reasoning"]
                }"#
                .into(),
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
        assert!(caps.reasoning, "a reasoning capability means effort levels");
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
                    // Messages are native objects: role + plain-string
                    // content — never OpenAI-style typed content arrays.
                    assert_eq!(
                        body["messages"],
                        serde_json::json!([
                            { "role": "system", "content": "sys" },
                            { "role": "user", "content": "hi" },
                        ])
                    );
                    // Internal fields never leak.
                    for leaked in [
                        "operation_id",
                        "session_id",
                        "attempt",
                        "deadline_ms",
                        "cancellation",
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

    #[tokio::test]
    async fn ndjson_frame_split_across_http_chunks_assembles() {
        // Ollama streams NDJSON; the old per-chunk .lines() corrupts a
        // frame split by HTTP chunking. Must reassemble.
        let server = MockServer::new();
        server.route(
            "POST",
            "/api/chat",
            MockAction::ChunkedSse {
                status: 200,
                chunks: vec![
                    b"{\"message\":{\"role\":\"assistant\",\"content\":\"par".to_vec(),
                    b"tial\"},\"done\":true}\n".to_vec(),
                    b"{\"message\":{\"role\":\"assistant\",\"content\":\" tail\"},\"done\":false}\n".to_vec(),
                    b"{\"done\":true}\n".to_vec(),
                ],
            },
        );
        let base = server.base_url().await;
        let provider = OllamaProvider::build(OllamaConfig::new(Some(base)));
        let mut stream = provider.stream(req("qwen3.8"));
        let mut text = String::new();
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(ProviderChunk::Text { text: t }) => text.push_str(&t),
                Ok(ProviderChunk::Done) => break,
                Ok(_) => {}
                Err(e) => panic!("fragmented NDJSON must assemble, got {e:?}"),
            }
        }
        assert_eq!(text, "partial tail");
    }

    #[tokio::test]
    async fn refresh_from_live_drives_capabilities() {
        // Spec §10: discovery/probing drive behavior — after warm-up,
        // capabilities() reflects the LIVE probed model, not the default
        // small-local constant. (The audit: probe APIs existed but nothing
        // ever called them.)
        let server = MockServer::new();
        server.route(
            "GET",
            "/api/tags",
            MockAction::Respond {
                status: 200,
                body: r#"{"models":[{"name":"qwen3.8:latest"},{"name":"dead-server-model"}]}"#
                    .into(),
            },
        );
        server.route(
            "POST",
            "/api/show",
            MockAction::Respond {
                status: 200,
                // Real /api/show metadata (architecture-prefixed keys).
                body: r#"{
                    "model_info": {
                        "general.architecture": "qwen3",
                        "qwen3.context_length": 262144
                    },
                    "capabilities": ["tools", "vision", "reasoning"]
                }"#
                .into(),
            },
        );
        let base = server.base_url().await;
        let provider = OllamaProvider::new(OllamaConfig::new(Some(base)));
        // Before warm-up: the conservative default.
        assert_eq!(
            provider.capabilities("qwen3.8:latest").context,
            ModelCapabilities::small_local().context,
            "pre-warm default is the conservative profile"
        );
        let n = provider.refresh_from_live().await.unwrap();
        assert_eq!(n, 2, "both discovered models probed");
        let caps = provider.capabilities("qwen3.8:latest");
        assert_eq!(caps.context, 262_144);
        assert!(caps.tools);
        assert!(caps.vision);
        assert!(caps.thinking);
        // Unknown models stay on the default.
        assert_eq!(
            provider.capabilities("not-installed").context,
            ModelCapabilities::small_local().context
        );
    }

    #[tokio::test]
    async fn refresh_survives_hostile_tags_body() {
        // A garbage /api/tags response must surface as an error, and a
        // dead probe target must not poison the cache.
        let server = MockServer::new();
        server.route(
            "GET",
            "/api/tags",
            MockAction::Respond {
                status: 200,
                body: "{not json".into(),
            },
        );
        let base = server.base_url().await;
        let provider = OllamaProvider::new(OllamaConfig::new(Some(base)));
        assert!(provider.refresh_from_live().await.is_err());
        assert_eq!(
            provider.capabilities("qwen3.8:latest").context,
            ModelCapabilities::small_local().context
        );
    }

    #[tokio::test]
    async fn native_wire_frame_shape() {
        // P0: the generic request must lower to REAL Ollama message objects
        // (role + plain-string content) — never OpenAI-style content arrays.
        // A request with system + user text and NO images must not carry an
        // images field at all.
        let server = MockServer::new();
        server.route(
            "POST",
            "/api/chat",
            MockAction::AssertThenRespond {
                status: 200,
                body: r#"{"message":{"role":"assistant","content":"ok"},"done":true}"#.into(),
                assert: Arc::new(|body: &serde_json::Value| {
                    assert_eq!(
                        body["messages"],
                        serde_json::json!([
                            { "role": "system", "content": "sys" },
                            { "role": "user", "content": "hi" },
                        ]),
                        "system is mapped to the first message; user text is a plain content string"
                    );
                    for msg in body["messages"].as_array().unwrap() {
                        for key in msg.as_object().unwrap().keys() {
                            assert!(
                                ["role", "content", "thinking", "images", "tool_calls"]
                                    .contains(&key.as_str()),
                                "message carries a non-native field: {key}"
                            );
                        }
                    }
                    let user_msg = &body["messages"][1];
                    assert!(
                        !user_msg.as_object().unwrap().contains_key("images"),
                        "no images in the request -> the images field is omitted"
                    );
                    assert!(
                        !serde_json::to_string(&body["messages"])
                            .unwrap()
                            .contains("\"type\""),
                        "typed content blocks must never reach the Ollama wire"
                    );
                }),
            },
        );
        let base = server.base_url().await;
        let provider = OllamaProvider::build(OllamaConfig::new(Some(base)));
        let chunks = stream_chunks(&*provider, req("qwen3.8")).await;
        assert!(matches!(chunks.last(), Some(ProviderChunk::Done)));
    }

    #[tokio::test]
    async fn images_lower_to_native_base64_array() {
        // Content parts of kind image become the native `images` array
        // (raw base64 — a data: URI prefix is stripped), never an OpenAI
        // image_url block.
        let server = MockServer::new();
        server.route(
            "POST",
            "/api/chat",
            MockAction::Respond {
                status: 200,
                body: r#"{"done":true}"#.into(),
            },
        );
        let base = server.base_url().await;
        let provider = OllamaProvider::build(OllamaConfig::new(Some(base)));
        let mut r = req("qwen3.8");
        r.messages.push(RequestMessage {
            role: Role::User,
            content: vec![
                ContentPart::text("what is this?"),
                ContentPart {
                    kind: ContentKind::Image {
                        url: "data:image/png;base64,QUJD".into(),
                    },
                    tool_call_id: None,
                },
            ],
        });
        let chunks = stream_chunks(&*provider, r).await;
        assert!(matches!(chunks.last(), Some(ProviderChunk::Done)));
        let (_, _, raw) = server.last_request().unwrap();
        let body: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let msg = &body["messages"][2];
        assert_eq!(
            msg["content"], "what is this?",
            "text survives alongside images"
        );
        assert_eq!(
            msg["images"],
            serde_json::json!(["QUJD"]),
            "data-URI prefix stripped to base64"
        );
        assert_eq!(
            body["messages"][1]["images"],
            serde_json::Value::Null,
            "image-less messages omit images"
        );
    }

    /// Stream one /api/chat request against a scratch mock server and
    /// return the recorded request body (asserting the stream completed).
    async fn chat_request_body(
        reasoning: Option<ReasoningMode>,
        effort_capable: bool,
    ) -> serde_json::Value {
        let server = MockServer::new();
        server.route(
            "POST",
            "/api/chat",
            MockAction::Respond {
                status: 200,
                body: r#"{"done":true}"#.into(),
            },
        );
        let base = server.base_url().await;
        let mut cfg = OllamaConfig::new(Some(base));
        if effort_capable {
            cfg.model_overrides.insert(
                "qwen3.8".into(),
                ModelCapabilities {
                    reasoning: true,
                    thinking: true,
                    ..ModelCapabilities::default()
                },
            );
        }
        let provider = OllamaProvider::new(cfg);
        let mut r = req("qwen3.8");
        r.reasoning = reasoning;
        let chunks = stream_chunks(&*provider, r).await;
        assert!(matches!(chunks.last(), Some(ProviderChunk::Done)));
        let (_, _, raw) = server.last_request().unwrap();
        serde_json::from_str(&raw).unwrap()
    }

    #[tokio::test]
    async fn thinking_request_body() {
        // The request's stored ReasoningMode drives the native /api/chat
        // `think` knob — never guessed from content text.
        let off = chat_request_body(Some(ReasoningMode::Off), false).await;
        assert_eq!(off["think"], serde_json::json!(false), "Off -> think:false");
        let bool_knob = chat_request_body(Some(ReasoningMode::Medium), false).await;
        assert_eq!(
            bool_knob["think"],
            serde_json::json!(true),
            "boolean-thinking profile -> think:true for any non-Off level"
        );
        let effort = chat_request_body(Some(ReasoningMode::High), true).await;
        assert_eq!(
            effort["think"],
            serde_json::json!("high"),
            "effort-capable profile -> documented level string"
        );
        let low = chat_request_body(Some(ReasoningMode::Low), true).await;
        assert_eq!(low["think"], serde_json::json!("low"));
        let none = chat_request_body(None, false).await;
        assert!(
            !none.as_object().unwrap().contains_key("think"),
            "no reasoning mode -> knob omitted (server default)"
        );
    }

    #[tokio::test]
    async fn thinking_response_parsed() {
        // Native /api/chat surfaces thinking as message.thinking. A frame
        // carrying thinking AND content must yield the Reasoning chunk
        // before the Text chunk.
        let server = MockServer::new();
        server.route(
            "POST",
            "/api/chat",
            MockAction::Respond {
                status: 200,
                body: r#"{"message":{"role":"assistant","thinking":"let me reason","content":"the answer"},"done":true}"#.into(),
            },
        );
        let base = server.base_url().await;
        let provider = OllamaProvider::build(OllamaConfig::new(Some(base)));
        let chunks = stream_chunks(&*provider, req("qwen3.8")).await;
        let mut kinds = Vec::new();
        for chunk in &chunks {
            match chunk {
                ProviderChunk::Reasoning { text } => {
                    assert_eq!(text, "let me reason");
                    kinds.push("reasoning");
                }
                ProviderChunk::Text { text } => {
                    assert_eq!(text, "the answer");
                    kinds.push("text");
                }
                ProviderChunk::Done => kinds.push("done"),
                other => panic!("unexpected chunk {other:?}"),
            }
        }
        assert_eq!(
            kinds,
            ["reasoning", "text", "done"],
            "thinking precedes text"
        );
    }

    #[tokio::test]
    async fn multiple_tool_calls_one_frame() {
        // ONE native frame with N tool_calls must yield N ToolCall chunks
        // in array order — the old parser returned after the first call
        // and silently dropped the rest.
        let server = MockServer::new();
        server.route(
            "POST",
            "/api/chat",
            MockAction::Respond {
                status: 200,
                body: r#"{"message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"read_file","arguments":{"path":"a.rs"}}},{"function":{"name":"write_file","arguments":{"path":"b.txt","content":"hi"}}},{"function":{"name":"read_file","arguments":{"path":"c.rs"}}}]},"done":true}"#.into(),
            },
        );
        let base = server.base_url().await;
        let provider = OllamaProvider::build(OllamaConfig::new(Some(base)));
        let chunks = stream_chunks(&*provider, req("qwen3.8")).await;
        let calls: Vec<(String, String, serde_json::Value, bool)> = chunks
            .iter()
            .filter_map(|c| match c {
                ProviderChunk::ToolCall {
                    id,
                    name,
                    input,
                    complete,
                } => Some((id.clone(), name.clone(), input.clone(), *complete)),
                _ => None,
            })
            .collect();
        assert_eq!(calls.len(), 3, "all three calls must surface as chunks");
        assert_eq!(calls[0].0, "ollama:0:0");
        assert_eq!(calls[0].1, "read_file");
        assert_eq!(calls[0].2["path"], "a.rs");
        assert_eq!(calls[1].0, "ollama:0:1");
        assert_eq!(calls[1].1, "write_file");
        assert_eq!(calls[1].2["content"], "hi");
        assert_eq!(calls[2].0, "ollama:0:2");
        assert_eq!(calls[2].1, "read_file");
        assert_eq!(calls[2].2["path"], "c.rs");
        assert!(
            calls.iter().all(|(_, _, _, complete)| *complete),
            "a native tool call arrives complete"
        );
        assert!(matches!(chunks.last(), Some(ProviderChunk::Done)));
    }

    #[tokio::test]
    async fn tool_id_uniqueness() {
        // Ids are ollama:<response-seq>:<tool-index>: two calls of the same
        // tool inside ONE response get distinct ids, and ids from a second
        // response never collide with the first. Never ollama_call_<name>.
        let server = MockServer::new();
        server.route(
            "POST",
            "/api/chat",
            MockAction::Sequence {
                actions: vec![
                    MockAction::Respond {
                        status: 200,
                        body: r#"{"message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"read_file","arguments":{"path":"a.rs"}}},{"function":{"name":"read_file","arguments":{"path":"b.rs"}}}]},"done":true}"#.into(),
                    },
                    MockAction::Respond {
                        status: 200,
                        body: r#"{"message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"read_file","arguments":{"path":"c.rs"}}}]},"done":true}"#.into(),
                    },
                ],
            },
        );
        let base = server.base_url().await;
        let provider = OllamaProvider::new(OllamaConfig::new(Some(base)));
        let ids: Vec<String> = stream_chunks(&*provider, req("qwen3.8"))
            .await
            .into_iter()
            .filter_map(|c| match c {
                ProviderChunk::ToolCall { id, .. } => Some(id),
                _ => None,
            })
            .collect();
        let ids2: Vec<String> = stream_chunks(&*provider, req("qwen3.8"))
            .await
            .into_iter()
            .filter_map(|c| match c {
                ProviderChunk::ToolCall { id, .. } => Some(id),
                _ => None,
            })
            .collect();
        assert_eq!(
            ids,
            vec!["ollama:0:0".to_string(), "ollama:0:1".to_string()]
        );
        assert_eq!(ids2, vec!["ollama:1:0".to_string()]);
        let mut all = ids.clone();
        all.extend(ids2.iter().cloned());
        let mut sorted = all.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            all.len(),
            "ids must be unique across responses"
        );
        assert!(
            ids.iter()
                .chain(ids2.iter())
                .all(|id| id.starts_with("ollama:") && !id.contains("ollama_call")),
            "ids are ollama:<seq>:<idx>, never synthesized name ids"
        );
    }

    #[tokio::test]
    async fn tool_result_lowering() {
        // A request replaying a finished tool round trip serializes as the
        // native form: assistant tool_calls, then role-"tool" messages with
        // the raw content and the documented tool_name when the answered
        // call is visible in this request (an orphan result is sent without
        // a tool_name, never fabricated).
        let server = MockServer::new();
        server.route(
            "POST",
            "/api/chat",
            MockAction::Respond {
                status: 200,
                body: r#"{"done":true}"#.into(),
            },
        );
        let base = server.base_url().await;
        let provider = OllamaProvider::build(OllamaConfig::new(Some(base)));
        let mut r = req("qwen3.8");
        r.messages.push(RequestMessage {
            role: Role::Assistant,
            content: vec![ContentPart::tool_call(
                "call_1",
                "read_file",
                serde_json::json!({"path": "a.rs"}),
            )],
        });
        r.messages.push(RequestMessage {
            role: Role::User,
            content: vec![
                ContentPart::tool_result("fn main() {}", false, "call_1"),
                ContentPart::tool_result("boom", true, "call_missing"),
            ],
        });
        let chunks = stream_chunks(&*provider, r).await;
        assert!(matches!(chunks.last(), Some(ProviderChunk::Done)));
        let (_, _, raw) = server.last_request().unwrap();
        let body: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(
            body["messages"],
            serde_json::json!([
                { "role": "system", "content": "sys" },
                { "role": "user", "content": "hi" },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "function": { "name": "read_file", "arguments": { "path": "a.rs" } }
                    }]
                },
                { "role": "tool", "content": "fn main() {}", "tool_name": "read_file" },
                { "role": "tool", "content": "boom" }
            ])
        );
    }

    #[tokio::test]
    async fn architecture_prefixed_context() {
        // Real /api/show reports "<arch>.context_length" (never a bare
        // "context_length"); the architecture comes from the general
        // section, nested or dotted.
        async fn probe(body: &str) -> ModelCapabilities {
            let server = MockServer::new();
            server.route(
                "POST",
                "/api/show",
                MockAction::Respond {
                    status: 200,
                    body: body.into(),
                },
            );
            let base = server.base_url().await;
            let provider = OllamaProvider::new(OllamaConfig::new(Some(base)));
            provider.probe_model("qwen3.8").await.unwrap()
        }
        // Nested general section + dotted arch key (field-report fixture).
        let nested = probe(
            r#"{"model_info":{"general":{"architecture":"qwen3"},"qwen3.context_length":262144}}"#,
        )
        .await;
        assert_eq!(nested.context, 262_144);
        // Flat real metadata.
        let flat = probe(
            r#"{"model_info":{"general.architecture":"gemma3","gemma3.context_length":131072}}"#,
        )
        .await;
        assert_eq!(flat.context, 131_072);
        // Fallback: no architecture declared, exactly one ".context_length".
        let fallback = probe(r#"{"model_info":{"llama.context_length":131072}}"#).await;
        assert_eq!(fallback.context, 131_072);
        // Hostile: several context keys with no declared architecture are
        // NOT guessed (a wrong guess silently truncates conversations).
        let ambiguous = probe(
            r#"{"model_info":{"llama.context_length":131072,"qwen3.context_length":262144}}"#,
        )
        .await;
        assert_eq!(ambiguous.context, ModelCapabilities::small_local().context);
        // Hostile: architecture declared but no matching context key.
        let missing = probe(r#"{"model_info":{"general.architecture":"qwen3"}}"#).await;
        assert_eq!(missing.context, ModelCapabilities::small_local().context);
    }

    #[tokio::test]
    async fn runtime_context_min() {
        // /api/ps reports the window ALLOCATED in the loaded process; the
        // effective context is min(model max from /api/show, allocation).
        let server = MockServer::new();
        server.route(
            "POST",
            "/api/show",
            MockAction::Respond {
                status: 200,
                body: r#"{"model_info":{"general.architecture":"qwen3","qwen3.context_length":262144}}"#.into(),
            },
        );
        server.route(
            "GET",
            "/api/ps",
            MockAction::Respond {
                status: 200,
                body: r#"{"models":[{"name":"qwen3.8:latest","size_vram":42,"details":{"context_length":8192}}]}"#.into(),
            },
        );
        let base = server.base_url().await;
        let provider = OllamaProvider::new(OllamaConfig::new(Some(base)));
        let ctx = provider.runtime_context("qwen3.8").await.unwrap();
        assert_eq!(ctx.model_max, 262_144, "model max comes from /api/show");
        assert_eq!(
            ctx.runtime_effective,
            Some(8192),
            "runtime effective is the smaller ps allocation"
        );
        assert_eq!(
            provider.effective_context("qwen3.8").await.unwrap(),
            Some(8192)
        );
    }

    #[tokio::test]
    async fn runtime_context_clamps_allocation_above_model_max() {
        // A ps allocation above the model's own maximum is clamped: the
        // model can never serve more than its declared context.
        let server = MockServer::new();
        server.route(
            "POST",
            "/api/show",
            MockAction::Respond {
                status: 200,
                body: r#"{"model_info":{"general.architecture":"qwen3","qwen3.context_length":262144}}"#.into(),
            },
        );
        server.route(
            "GET",
            "/api/ps",
            MockAction::Respond {
                status: 200,
                body:
                    r#"{"models":[{"name":"qwen3.8:latest","details":{"context_length":524288}}]}"#
                        .into(),
            },
        );
        let base = server.base_url().await;
        let provider = OllamaProvider::new(OllamaConfig::new(Some(base)));
        assert_eq!(
            provider.effective_context("qwen3.8").await.unwrap(),
            Some(262_144),
            "effective context never exceeds the model maximum"
        );
    }

    #[tokio::test]
    async fn effective_context_survives_hostile_ps() {
        async fn probe_with_ps(ps_status: u16, ps_body: &str) -> kilop_core::Result<Option<usize>> {
            let server = MockServer::new();
            server.route(
                "POST",
                "/api/show",
                MockAction::Respond {
                    status: 200,
                    body: r#"{"model_info":{"general.architecture":"qwen3","qwen3.context_length":262144}}"#.into(),
                },
            );
            server.route(
                "GET",
                "/api/ps",
                MockAction::Respond {
                    status: ps_status,
                    body: ps_body.into(),
                },
            );
            let base = server.base_url().await;
            let provider = OllamaProvider::new(OllamaConfig::new(Some(base)));
            provider.effective_context("qwen3.8").await
        }
        // Garbage body: loud error, never a panic.
        let err = probe_with_ps(200, "{not json").await.unwrap_err();
        assert_eq!(err.kind, ErrorKind::Malformed, "{err}");
        // Non-2xx: loud error.
        let err = probe_with_ps(500, "{}").await.unwrap_err();
        assert!(
            !err.retryable,
            "a dead /api/ps must not be blindly retried: {err}"
        );
        assert!(
            matches!(
                &err.kind,
                ErrorKind::Provider { code, retryable: false } if code == "500"
            ),
            "{err:?}"
        );
        // Model not loaded: Ok(None), the caller falls back to model max.
        let none = probe_with_ps(
            200,
            r#"{"models":[{"name":"llama3.2:latest","details":{"context_length":8192}}]}"#,
        )
        .await
        .unwrap();
        assert_eq!(none, None);
        // Hostile field types are treated as absent, not panics.
        let none = probe_with_ps(
            200,
            r#"{"models":[{"name":"qwen3.8:latest","details":{"context_length":"8192"}}]}"#,
        )
        .await
        .unwrap();
        assert_eq!(none, None);
    }

    #[test]
    fn ps_allocated_context_never_panics_on_hostile_bodies() {
        assert_eq!(
            ps_allocated_context(&serde_json::json!({"nope": 1}), "m")
                .unwrap_err()
                .kind,
            ErrorKind::Malformed
        );
        assert!(ps_allocated_context(&serde_json::json!({"models": {}}), "m").is_err());
        assert!(ps_allocated_context(&serde_json::json!({"models": [42]}), "m").is_err());
        assert!(
            ps_allocated_context(
                &serde_json::json!({"models": [{"details": {"context_length": 1}}]}),
                "m"
            )
            .is_err(),
            "an entry without a name is corrupt"
        );
        assert_eq!(
            ps_allocated_context(&serde_json::json!({"models": []}), "m").unwrap(),
            None
        );
        // Tag-less matching: entry "qwen3.8:latest" answers model "qwen3.8".
        let body = serde_json::json!({"models": [{"name": "qwen3.8:latest", "details": {"context_length": 8192}}]});
        assert_eq!(ps_allocated_context(&body, "qwen3.8").unwrap(), Some(8192));
        assert_eq!(
            ps_allocated_context(&body, "qwen3.8:latest").unwrap(),
            Some(8192)
        );
        assert_eq!(ps_allocated_context(&body, "qwen3.8-abl").unwrap(), None);
        // Non-numeric allocations are absent, never a panic.
        let str_ctx = serde_json::json!({"models": [{"name": "qwen3.8:latest", "details": {"context_length": "8192"}}]});
        assert_eq!(ps_allocated_context(&str_ctx, "qwen3.8").unwrap(), None);
        // Entry-level context_length also counts.
        let top =
            serde_json::json!({"models": [{"name": "qwen3.8:latest", "context_length": 4096}]});
        assert_eq!(ps_allocated_context(&top, "qwen3.8").unwrap(), Some(4096));
        // size_vram is BYTES, never mistaken for a token budget.
        let vram = serde_json::json!({"models": [{"name": "qwen3.8:latest", "size_vram": 999999}]});
        assert_eq!(ps_allocated_context(&vram, "qwen3.8").unwrap(), None);
    }

    #[test]
    fn context_lookup_handles_nested_and_dotted_arch_shapes() {
        let nested_section = serde_json::json!({
            "general": {"architecture": "qwen3"},
            "qwen3": {"context_length": 262144},
        });
        assert_eq!(
            context_length_from_model_info(&nested_section),
            Some(262_144)
        );
        let bare = serde_json::json!({"context_length": 32768});
        assert_eq!(context_length_from_model_info(&bare), Some(32_768));
        let hostile_negative = serde_json::json!({"llama.context_length": -1});
        assert_eq!(context_length_from_model_info(&hostile_negative), None);
        let non_numeric = serde_json::json!({"llama.context_length": "131072"});
        assert_eq!(context_length_from_model_info(&non_numeric), None);
        let no_info = serde_json::json!({});
        assert_eq!(context_length_from_model_info(&no_info), None);
    }

    #[test]
    fn native_image_payload_strips_data_uri_prefix() {
        assert_eq!(native_image_payload("data:image/png;base64,QUJD"), "QUJD");
        assert_eq!(native_image_payload("data:,YWJj"), "YWJj");
        assert_eq!(native_image_payload("QUJD"), "QUJD");
        assert_eq!(
            native_image_payload("data:no-comma"),
            "data:no-comma",
            "malformed data URIs pass through untouched"
        );
    }

    #[test]
    fn lowering_splits_mixed_part_roles_preserving_order() {
        // One generic message with parts of every kind: text stays on the
        // declared role, tool results become role-"tool" messages at their
        // position, tool calls ride an assistant message, and nothing is
        // reordered.
        let m = RequestMessage {
            role: Role::User,
            content: vec![
                ContentPart::text("a"),
                ContentPart::tool_result("out", false, "c1"),
                ContentPart::tool_call("c1", "read_file", serde_json::json!({"path": "x"})),
                ContentPart::reasoning("think"),
                ContentPart::text("b"),
            ],
        };
        let mut names = HashMap::new();
        let native = lower_native_message(&m, &mut names);
        assert_eq!(
            native,
            vec![
                serde_json::json!({"role": "user", "content": "a"}),
                // The result precedes its call in the parts: no tool_name.
                serde_json::json!({"role": "tool", "content": "out"}),
                // Tool call + reasoning coalesce onto ONE assistant message
                // (assistant messages may carry tool_calls and thinking
                // together — the native fields stay orthogonal).
                serde_json::json!({
                    "role": "assistant",
                    "content": "",
                    "thinking": "think",
                    "tool_calls": [{
                        "function": { "name": "read_file", "arguments": { "path": "x" } }
                    }]
                }),
                serde_json::json!({"role": "user", "content": "b"}),
            ]
        );
    }

    #[test]
    fn tool_result_names_the_call_it_answers_when_visible() {
        // The native form pairs the result with its call only when the
        // call appeared earlier in this request.
        let call_msg = RequestMessage {
            role: Role::Assistant,
            content: vec![ContentPart::tool_call(
                "call_1",
                "read_file",
                serde_json::json!({"path": "a.rs"}),
            )],
        };
        let result_msg = RequestMessage {
            role: Role::User,
            content: vec![ContentPart::tool_result("contents", false, "call_1")],
        };
        let mut names = HashMap::new();
        let mut native = Vec::new();
        native.extend(lower_native_message(&call_msg, &mut names));
        native.extend(lower_native_message(&result_msg, &mut names));
        assert_eq!(
            native,
            vec![
                serde_json::json!({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "function": { "name": "read_file", "arguments": { "path": "a.rs" } }
                    }]
                }),
                serde_json::json!({
                    "role": "tool",
                    "content": "contents",
                    "tool_name": "read_file"
                }),
            ]
        );
    }
}
