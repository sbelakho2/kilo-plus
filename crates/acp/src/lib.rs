//! faktor-acp — ACP v1 (Agent Client Protocol, protocol version 1) agent
//! server.
//!
//! Purpose (audit note): expose an ACP agent server so ACP-capable editors
//! (Zed and others) can drive the Faktor core alongside the native API and
//! the legacy compat surface. This crate implements the WIRE + lifecycle
//! only; agent behavior is delegated to the injected backend seam, so the
//! daemon can attach the real runtime without this crate knowing it.
//!
//! # Wire surface (official Agent Client Protocol v1 subset)
//!
//! JSON-RPC 2.0 over Content-Length framed streams (see [`protocol`]).
//! Field names follow the official ACP v1 schema; the crate's pre-1.0
//! deviations (`protocolVersion: "0.1.0"`, `sessionID`, `text`,
//! `"agent"`, `session/abort`, result-passthrough prompt responses) are
//! renamed — deprecated wire aliases are still *accepted* on parse but
//! never produced.
//!
//! | method           | request params                             | response                              |
//! |------------------|--------------------------------------------|---------------------------------------|
//! | `initialize`     | `{protocolVersion: 1, ...}`                | `{protocolVersion: 1, agentCapabilities, authMethods}` |
//! | `session/new`    | opaque backend params                      | `{sessionId}`                         |
//! | `session/prompt` | `{sessionId, prompt:[{type:"text",text}]}` | `{stopReason}` (see below)            |
//! | `session/cancel` | `{sessionId}` (notification or request)    | none (notification) / `{}` (request)  |
//! | `session/update` | agent→client notification `{sessionId, update}` | —                               |
//!
//! * Version handshake: `initialize` accepts `protocolVersion: 1` (the
//!   official ACP v1 major version) and rejects anything else loudly with
//!   a typed error — never a silent fallback.
//! * Prompt outcome: a turn ends with the official `stopReason` response
//!   (`"end_turn"` on success, `"cancelled"` when cancelled). The sync
//!   backend's opaque result JSON — which official ACP v1 has no slot for —
//!   rides the official `_meta` extension member of the result and is
//!   omitted when null. Turn failures answer the prompt with the official
//!   error format (`-32603` internal error, backend message in `data`).
//! * A cancelled turn MUST answer the original `session/prompt` with
//!   `{stopReason: "cancelled"}` and MUST NOT emit further content frames
//!   after cancellation is observed.
//! * Error format (official): `{code, message}` with `data` omitted when
//!   absent. Codes: `-32700` parse, `-32600` invalid request, `-32601`
//!   method not found, `-32602` invalid params, `-32603` internal error,
//!   ACP-reserved-range `-32001` (session busy). `-32000` is not used
//!   (officially "Authentication required").
//! * Streaming: the streaming seam emits `session/update` notifications
//!   for the official kinds this surface can populate (`agent_message_chunk`
//!   text content) plus the crate's documented `agentStateChanged` status
//!   frames (`{kind: "agentStateChanged", agentState: {status:
//!   "idle"|"busy"|"error", message?}}`, see [`agent_state_changed_update`]).
//! * Extensions kept from the pre-conformance surface: `agent_info`
//!   (agent metadata), `session/list` (`{sessions: [{sessionId}, ..]}`),
//!   `shutdown` (request answered `{"ok": true}`, then the loop ends), and
//!   the deprecated `session/abort` alias of `session/cancel`.
//!
//! # Runtime architecture (async dispatch + cancellation)
//!
//! One connection is served by four cooperating roles:
//!
//! 1. **Reader task** — pulls frames from the transport, decodes
//!    JSON-RPC, and routes. `session/cancel` (and the deprecated
//!    `session/abort`) short-circuit *here*: the session's cancellation
//!    token fires synchronously and, for sync backends, the legacy
//!    `abort` hook runs off-thread. A cancel never waits behind a full
//!    writer queue before it lands.
//! 2. **Dispatcher task** — owns the per-session state machine and routes
//!    `session/prompt` to per-session operation tasks. At most one running
//!    turn plus one queued prompt per session (FIFO); deeper concurrency
//!    is refused with the typed busy error `-32001`.
//! 3. **Per-session operation tasks** — run one prompt turn against the
//!    backend, stream `session/update` frames, observe the cancellation
//!    token, and answer the original prompt id with the terminal
//!    `stopReason` response *before* promoting a queued prompt, so
//!    per-session wire order holds by construction.
//! 4. **Writer task** — owns the transport output. Every frame travels a
//!    bounded queue: the **main queue** (capacity
//!    [`AcpConfig::writer_queue_capacity`], default 64) carries responses
//!    and `session/update` notifications, and the **cancel lane**
//!    (fixed capacity [`CANCEL_LANE_CAPACITY`] = 16) carries cancel
//!    acknowledgements. The writer drains the lane strictly before the
//!    main queue, so a cancel answer never waits behind a full queue of
//!    prompt frames. Full queues backpressure senders — nothing buffers
//!    unboundedly.
//!
//! # Bounded everything
//!
//! - Declared frames are capped at [`protocol::MAX_FRAME_BYTES`] (16 MiB)
//!   by the parser; request params at [`MAX_PARAMS_BYTES`] (1 MiB);
//!   responses and notifications at [`MAX_RESPONSE_BYTES`] (8 MiB) — an
//!   oversize item is refused with `-32603`, never truncated or buffered.
//! - Session bookkeeping is capped ([`AcpConfig::max_sessions`]); idle
//!   entries are evicted first.
//! - Sync backend prompts run on their operation task (same contract as
//!   the pre-conformance single-task loop): the daemon must attach a
//!   backend that answers promptly or is internally time-boxed. Mid-run
//!   cancellation of a *sync* prompt is delivered through the legacy
//!   `abort` hook (the daemon maps it onto its own cancellation); the
//!   *streaming* seam observes the crate's cancellation token directly.
//!   Wire outcomes are identical: exactly one terminal response per turn,
//!   `"cancelled"` iff the cancel reached the turn before its terminal
//!   decision point.
//!
//! # Out of official ACP v1 scope (honestly absent, never faked)
//!
//! Tool calls/plans/permission requests, terminals, the client file
//! system methods, MCP server connections, `session/load`, and
//! `authenticate` are not part of this crate's backend seam and are not
//! advertised (`agentCapabilities.loadSession: false`; no
//! `mcpCapabilities`). Prompt content blocks other than plain text are
//! refused with `-32602`.

use futures::future::BoxFuture;
use serde_json::{json, Map, Value};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::sync::{mpsc, Notify};
use tokio::time::Duration;

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::protocol::AcpMethod;

pub mod protocol;

/// Official ACP v1 protocol version accepted by the `initialize` handshake.
pub const PROTOCOL_VERSION: u64 = 1;

/// Fixed capacity of the high-priority cancel lane (see module docs).
pub const CANCEL_LANE_CAPACITY: usize = 16;

/// Cap on the serialized `params` of one incoming request (1 MiB).
pub const MAX_PARAMS_BYTES: usize = 1024 * 1024;

/// Cap on one serialized frame written to the wire (8 MiB).
pub const MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024;

/// JSON-RPC well-known error codes (official ACP v1 messages).
pub const PARSE_ERROR: i64 = -32700;
pub const INVALID_REQUEST: i64 = -32600;
pub const METHOD_NOT_FOUND: i64 = -32601;
pub const INVALID_PARAMS: i64 = -32602;
pub const INTERNAL_ERROR: i64 = -32603;

/// ACP-reserved-range error (official range -32000..=-32099): a prompt
/// turn is already running (or queued) for the session.
pub const SESSION_BUSY: i64 = -32001;

/// ACP-reserved-range error: the server's per-session capacity is
/// exhausted (a bounded-resource refusal, never unbounded growth).
pub const SESSION_LIMIT: i64 = -32003;

const MAX_METHOD_LEN: usize = 128;
const READ_CHUNK: usize = 64 * 1024;
const SHUTDOWN_METHOD: &str = "shutdown";

/// Official canonical error messages.
const MSG_PARSE_ERROR: &str = "Parse error";
const MSG_METHOD_NOT_FOUND: &str = "Method not found";
const MSG_INTERNAL_ERROR: &str = "Internal error";
const MSG_SESSION_BUSY: &str = "A prompt turn is already in progress for this session";
const MSG_SESSION_LIMIT: &str = "session capacity exhausted";

/// Server-side sizing/behavior knobs. All queues are bounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcpConfig {
    /// Capacity of the main outgoing frame queue (responses + updates).
    pub writer_queue_capacity: usize,
    /// Capacity of the reader→dispatcher request queue.
    pub request_queue_capacity: usize,
    /// Upper bound on tracked sessions (idle entries are evicted first).
    pub max_sessions: usize,
    /// Bounded wait granted to a streaming backend after cancellation
    /// before the operation task answers `stopReason: cancelled` anyway.
    pub cancel_grace: Duration,
    /// Bounded wait for the dispatcher/writer tasks to wind down at EOF.
    pub shutdown_timeout: Duration,
}

impl Default for AcpConfig {
    fn default() -> Self {
        Self {
            writer_queue_capacity: 64,
            request_queue_capacity: 64,
            max_sessions: 1024,
            cancel_grace: Duration::from_secs(2),
            shutdown_timeout: Duration::from_secs(5),
        }
    }
}

/// Status values of the crate's `agentStateChanged` status frames
/// (official ACP v1 enum: `idle`, `busy`, `error`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentStateStatus {
    Busy,
    Idle,
    Error,
}

impl AgentStateStatus {
    /// Exact on-the-wire status string.
    pub fn as_wire_str(self) -> &'static str {
        match self {
            AgentStateStatus::Busy => "busy",
            AgentStateStatus::Idle => "idle",
            AgentStateStatus::Error => "error",
        }
    }
}

/// `session/update` frame body: `agentStateChanged` status frame
/// (`{kind, agentState: {status, message?}}`). This is this crate's
/// documented status channel: the official ACP v1 schema has no state
/// frame kind, so status frames are emitted only by backends that opt in
/// (see module docs on out-of-scope surface).
pub fn agent_state_changed_update(status: AgentStateStatus, message: Option<&str>) -> Value {
    let mut state = Map::new();
    state.insert("status".into(), json!(status.as_wire_str()));
    if let Some(message) = message {
        state.insert("message".into(), json!(message));
    }
    let mut update = Map::new();
    update.insert("kind".into(), json!("agentStateChanged"));
    update.insert("agentState".into(), Value::Object(state));
    Value::Object(update)
}

/// `session/update` frame body: official `agent_message_chunk` content
/// frame (`{sessionUpdate, content: {type: "text", text}}`).
pub fn text_chunk_update(text: &str) -> Value {
    json!({
        "sessionUpdate": "agent_message_chunk",
        "content": { "type": "text", "text": text },
    })
}

/// Official `session/update` notification params: `{sessionId, update}`.
pub fn session_update_params(session_id: &str, update: Value) -> Value {
    json!({ "sessionId": session_id, "update": update })
}

/// Cooperative cancellation token shared by a running turn, its streaming
/// backend, and the cancel path of the reader task.
#[derive(Clone, Debug, Default)]
pub struct CancelToken {
    inner: Arc<CancelInner>,
}

#[derive(Debug, Default)]
struct CancelInner {
    fired: AtomicBool,
    notify: Notify,
}

impl CancelToken {
    fn new() -> Self {
        Self::default()
    }

    /// Fire the token. Idempotent: a fired token never unfires.
    pub fn cancel(&self) {
        if !self.inner.fired.swap(true, Ordering::AcqRel) {
            self.inner.notify.notify_waiters();
        }
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.inner.fired.load(Ordering::Acquire)
    }

    /// Resolves once cancellation has been requested.
    pub async fn cancelled(&self) {
        loop {
            if self.is_cancelled() {
                return;
            }
            let notified = self.inner.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.is_cancelled() {
                return;
            }
            notified.await;
        }
    }
}

impl PartialEq for CancelToken {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for CancelToken {}

/// Why an emission failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmitError {
    /// The turn was cancelled; the emitter must stop producing frames.
    Cancelled,
    /// The outgoing connection closed (writer task ended).
    Closed,
    /// The frame exceeds [`MAX_RESPONSE_BYTES`].
    TooLarge,
}

impl std::fmt::Display for EmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmitError::Cancelled => write!(f, "turn cancelled"),
            EmitError::Closed => write!(f, "connection closed"),
            EmitError::TooLarge => write!(f, "frame exceeds the response bound"),
        }
    }
}

/// Streaming context handed to [`AcpStreamBackend::prompt`]: the only way
/// a running turn puts frames on the wire, plus cancellation observation.
#[derive(Clone)]
pub struct PromptCtx {
    main_tx: mpsc::Sender<Vec<u8>>,
    session_id: Arc<str>,
    token: CancelToken,
}

impl PromptCtx {
    /// Emit one `session/update` notification for the current session.
    /// Resolves once the frame is queued on the bounded main queue;
    /// resolves with `Cancelled` as soon as the turn's token fires — a
    /// mid-frame cancel stops further frames without waiting for space.
    pub async fn emit(&self, update: Value) -> Result<(), EmitError> {
        if self.token.is_cancelled() {
            return Err(EmitError::Cancelled);
        }
        let params = session_update_params(&self.session_id, update);
        let body_len = serde_json::to_vec(&params)
            .map(|b| b.len())
            .unwrap_or(usize::MAX);
        if body_len > MAX_RESPONSE_BYTES {
            return Err(EmitError::TooLarge);
        }
        let frame = notification_frame_bytes("session/update", &params);
        tokio::select! {
            biased;
            _ = self.token.cancelled() => Err(EmitError::Cancelled),
            sent = self.main_tx.send(frame) => {
                sent.map_err(|_| EmitError::Closed)
            }
        }
    }

    /// Convenience: emit a [`text_chunk_update`] frame.
    pub async fn emit_text(&self, text: &str) -> Result<(), EmitError> {
        self.emit(text_chunk_update(text)).await
    }

    /// Convenience: emit an [`agent_state_changed_update`] status frame.
    pub async fn emit_agent_state(
        &self,
        status: AgentStateStatus,
        message: Option<&str>,
    ) -> Result<(), EmitError> {
        self.emit(agent_state_changed_update(status, message)).await
    }

    /// Resolves when the turn is cancelled (same as the token).
    pub async fn cancelled(&self) {
        self.token.cancelled().await
    }

    /// Whether the turn is cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.token.is_cancelled()
    }
}

/// The seam: deterministic, injectable agent behavior. The daemon attaches
/// the real Faktor runtime here; this crate only wires JSON-RPC to it.
///
/// The synchronous surface is the pre-conformance contract: `prompt` runs
/// as one blocking turn on the session's operation task (implementations
/// must not block indefinitely and should be internally cancellable via
/// [`AcpBackend::abort`]). Prefer the additive async surface
/// [`AcpStreamBackend`] for cancellable, streaming turns.
pub trait AcpBackend: Send + Sync {
    /// Agent metadata surfaced by the `agent_info` extension
    /// (e.g. `{"name": .., "version": ..}`).
    fn agent_info(&self) -> Value;
    /// Create a session; `Err` surfaces as `-32603` (internal error).
    fn create_session(&self, params: &Value) -> Result<String, String>;
    /// Run one prompt turn against `session_id`; `Err` surfaces as
    /// `-32603`.
    fn prompt(&self, session_id: &str, text: &str) -> Result<Value, String>;
    /// Best-effort abort of the active turn of `session_id`; invoked from
    /// the cancel path when a cancel lands on a running sync turn.
    fn abort(&self, session_id: &str) -> Result<(), String>;
    /// Current session ids (e.g. for the `session/list` extension).
    fn list_sessions(&self) -> Vec<String>;
}

impl<T: AcpBackend + ?Sized> AcpBackend for Arc<T> {
    fn agent_info(&self) -> Value {
        (**self).agent_info()
    }
    fn create_session(&self, params: &Value) -> Result<String, String> {
        (**self).create_session(params)
    }
    fn prompt(&self, session_id: &str, text: &str) -> Result<Value, String> {
        (**self).prompt(session_id, text)
    }
    fn abort(&self, session_id: &str) -> Result<(), String> {
        (**self).abort(session_id)
    }
    fn list_sessions(&self) -> Vec<String> {
        (**self).list_sessions()
    }
}

/// Additive async seam: a backend whose prompt runs are streams of frames
/// that observe the crate's cancellation token. This is the surface on
/// which mid-frame cancellation is guaranteed: once the token fires,
/// [`PromptCtx::emit`] stops resolving successfully, the backend returns,
/// and the operation task answers the original prompt id with
/// `{stopReason: "cancelled"}` after at most [`AcpConfig::cancel_grace`].
pub trait AcpStreamBackend: Send + Sync {
    /// Agent metadata (see [`AcpBackend::agent_info`]).
    fn agent_info(&self) -> Value;
    /// Create a session (see [`AcpBackend::create_session`]).
    fn create_session(&self, params: &Value) -> Result<String, String>;
    /// Current session ids.
    fn list_sessions(&self) -> Vec<String>;
    /// Run one prompt turn. `ctx` is the only way to stream frames and the
    /// only cancellation observation point. Returning before cancellation
    /// yields the `end_turn`/error outcome; returning after cancellation
    /// still yields `stopReason: cancelled` (cancellation always wins the
    /// race at the turn's terminal decision point).
    fn prompt<'a>(
        &'a self,
        session_id: &'a str,
        ctx: &'a PromptCtx,
        text: &'a str,
    ) -> BoxFuture<'a, Result<Value, String>>;
}

impl<T: AcpStreamBackend + ?Sized> AcpStreamBackend for Arc<T> {
    fn agent_info(&self) -> Value {
        (**self).agent_info()
    }
    fn create_session(&self, params: &Value) -> Result<String, String> {
        (**self).create_session(params)
    }
    fn list_sessions(&self) -> Vec<String> {
        (**self).list_sessions()
    }
    fn prompt<'a>(
        &'a self,
        session_id: &'a str,
        ctx: &'a PromptCtx,
        text: &'a str,
    ) -> BoxFuture<'a, Result<Value, String>> {
        (**self).prompt(session_id, ctx, text)
    }
}

/// One session's prompt state: at most one running turn plus one queued
/// prompt (FIFO); everything else is refused with the typed busy error.
#[derive(Debug, Default)]
struct SessionState {
    active: Option<ActiveTurn>,
}

#[derive(Debug)]
struct ActiveTurn {
    token: CancelToken,
    queued: Option<PromptJob>,
}

#[derive(Debug)]
struct PromptJob {
    id: u64,
    text: String,
}

/// Outcome of one prompt turn at its terminal decision point.
#[derive(Debug)]
enum TurnOutcome {
    Completed(Value),
    Failed(String),
    Cancelled,
}

#[derive(Clone)]
enum Engine {
    Sync(Arc<dyn AcpBackend>),
    Stream(Arc<dyn AcpStreamBackend>),
}

impl Engine {
    fn agent_info(&self) -> Value {
        match self {
            Engine::Sync(b) => b.agent_info(),
            Engine::Stream(b) => b.agent_info(),
        }
    }

    fn create_session(&self, params: &Value) -> Result<String, String> {
        match self {
            Engine::Sync(b) => b.create_session(params),
            Engine::Stream(b) => b.create_session(params),
        }
    }

    fn list_sessions(&self) -> Vec<String> {
        match self {
            Engine::Sync(b) => b.list_sessions(),
            Engine::Stream(b) => b.list_sessions(),
        }
    }

    fn is_sync(&self) -> bool {
        matches!(self, Engine::Sync(_))
    }

    /// Run one full turn. Stream backends run as a cancellable future;
    /// sync backends run their blocking call on this task exactly like the
    /// pre-conformance loop (their `abort` hook carries cancellation).
    async fn run_turn(
        &self,
        session_id: &str,
        job: &PromptJob,
        main_tx: mpsc::Sender<Vec<u8>>,
        token: CancelToken,
        config: AcpConfig,
    ) -> TurnOutcome {
        match self {
            Engine::Sync(backend) => {
                // Blocking call, same thread-context contract as before.
                let result = backend.prompt(session_id, &job.text);
                if token.is_cancelled() {
                    TurnOutcome::Cancelled
                } else {
                    match result {
                        Ok(value) => TurnOutcome::Completed(value),
                        Err(message) => TurnOutcome::Failed(message),
                    }
                }
            }
            Engine::Stream(backend) => {
                let ctx = PromptCtx {
                    main_tx,
                    session_id: Arc::from(session_id),
                    token: token.clone(),
                };
                let future = backend.prompt(session_id, &ctx, &job.text);
                let mut future = std::pin::pin!(future);
                tokio::select! {
                    biased;
                    _ = token.cancelled() => {
                        // Bounded wind-down: grant the backend a grace
                        // period to observe cancellation and return, then
                        // answer cancelled regardless.
                        match tokio::time::timeout(config.cancel_grace, &mut future).await {
                            Ok(_) => TurnOutcome::Cancelled,
                            Err(_elapsed) => {
                                tracing::debug!(
                                    session_id,
                                    grace_ms = config.cancel_grace.as_millis(),
                                    "acp: streaming backend did not stop within cancel grace"
                                );
                                TurnOutcome::Cancelled
                            }
                        }
                    }
                    result = &mut future => {
                        if token.is_cancelled() {
                            return TurnOutcome::Cancelled;
                        }
                        match result {
                            Ok(value) => TurnOutcome::Completed(value),
                            Err(message) => TurnOutcome::Failed(message),
                        }
                    }
                }
            }
        }
    }
}

/// One JSON-RPC message decoded from the transport.
#[derive(Debug)]
enum Incoming {
    Request {
        id: u64,
        method: String,
        params: Value,
    },
    Notification {
        method: String,
        params: Value,
    },
    Invalid {
        id: Value,
        code: i64,
        message: String,
    },
}

/// Decode one parsed message into a request, a notification, or a
/// well-formed error response to send back (JSON-RPC 2.0 §5.1 semantics).
fn classify(value: Value) -> Incoming {
    fn invalid(id: Value, code: i64, message: impl Into<String>) -> Incoming {
        Incoming::Invalid {
            id,
            code,
            message: message.into(),
        }
    }

    let Some(obj) = value.as_object() else {
        return invalid(Value::Null, INVALID_REQUEST, "message is not a JSON object");
    };
    match obj.get("jsonrpc") {
        Some(Value::String(s)) if s != "2.0" => {
            return invalid(
                Value::Null,
                INVALID_REQUEST,
                "jsonrpc version must be \"2.0\"",
            )
        }
        _ => {}
    }
    let Some(method) = obj.get("method").and_then(Value::as_str) else {
        return invalid(
            Value::Null,
            INVALID_REQUEST,
            "missing string field \"method\"",
        );
    };
    if method.len() > MAX_METHOD_LEN {
        return invalid(
            Value::Null,
            INVALID_REQUEST,
            format!("method exceeds {MAX_METHOD_LEN}-byte bound"),
        );
    }

    let id_field = obj.get("id");
    if id_field.is_none() || id_field.is_some_and(Value::is_null) {
        let params = obj.get("params").cloned().unwrap_or(Value::Null);
        return Incoming::Notification {
            method: method.to_string(),
            params,
        };
    }

    // A request: id must be a non-negative integer. Echo the original id
    // when it was numeric but not representable; null otherwise.
    let id_echo = match id_field {
        Some(v @ Value::Number(_)) => v.clone(),
        _ => Value::Null,
    };
    let Some(id) = id_field.and_then(Value::as_u64) else {
        return invalid(
            id_echo,
            INVALID_REQUEST,
            "request id must be a non-negative integer",
        );
    };

    let params = match obj.get("params") {
        None | Some(Value::Null) => json!({}),
        Some(p) if p.is_object() => p.clone(),
        Some(_) => {
            return invalid(
                Value::Number(id.into()),
                INVALID_PARAMS,
                "params must be a JSON object",
            )
        }
    };
    if serde_json::to_vec(&params)
        .map(|bytes| bytes.len())
        .unwrap_or(usize::MAX)
        > MAX_PARAMS_BYTES
    {
        return invalid(
            Value::Number(id.into()),
            INVALID_REQUEST,
            "request params exceed the 1 MiB params bound",
        );
    }
    Incoming::Request {
        id,
        method: method.to_string(),
        params,
    }
}

/// ACP agent server. Cheap to build; `serve_connection`/`run_stdio` own
/// the connection until EOF or `shutdown`.
#[derive(Clone)]
pub struct AcpServer {
    engine: Engine,
    config: AcpConfig,
}

impl AcpServer {
    /// Serve with the synchronous backend seam (pre-conformance contract,
    /// kept source-compatible).
    pub fn new<B: AcpBackend + 'static>(backend: B) -> Self {
        Self {
            engine: Engine::Sync(Arc::new(backend)),
            config: AcpConfig::default(),
        }
    }

    /// Serve with the additive streaming backend seam.
    pub fn new_streaming<B: AcpStreamBackend + 'static>(backend: B) -> Self {
        Self {
            engine: Engine::Stream(Arc::new(backend)),
            config: AcpConfig::default(),
        }
    }

    /// Override the sizing/behavior knobs.
    pub fn with_config(mut self, config: AcpConfig) -> Self {
        self.config = config;
        self
    }

    /// Serve ACP over an arbitrary async reader/writer pair (in-memory
    /// pipes for tests, stdio for production). Reader, dispatcher,
    /// per-session operation tasks and the writer task run concurrently
    /// (module docs). Returns `Ok(())` on EOF or `shutdown`.
    pub async fn serve_connection<R, W>(&self, mut reader: R, writer: W) -> Result<(), String>
    where
        R: AsyncRead + Unpin + Send + 'static,
        W: AsyncWrite + Unpin + Send + 'static,
    {
        let config = self.config;
        let (main_tx, main_rx) = mpsc::channel(config.writer_queue_capacity);
        let (lane_tx, lane_rx) = mpsc::channel(CANCEL_LANE_CAPACITY);
        let (request_tx, request_rx) = mpsc::channel(config.request_queue_capacity);
        let registry = Registry::new(config.max_sessions, config.cancel_grace);

        let writer_handle = tokio::spawn(writer_task(writer, main_rx, lane_rx));
        let dispatcher_handle = tokio::spawn(dispatcher_task(
            self.engine.clone(),
            registry.clone(),
            main_tx.clone(),
            request_rx,
        ));

        // Reader loop (this task): parse, short-circuit cancels, forward
        // everything else to the dispatcher.
        let mut buf: Vec<u8> = Vec::new();
        let mut chunk = vec![0u8; READ_CHUNK];
        let mut reader_error: Option<String> = None;
        'read: loop {
            let n = match reader.read(&mut chunk).await {
                Ok(0) => {
                    tracing::debug!("acp: input closed; ending serve loop");
                    break;
                }
                Ok(n) => n,
                Err(e) => {
                    reader_error = Some(format!("read error: {e}"));
                    break;
                }
            };
            buf.extend_from_slice(&chunk[..n]);
            loop {
                match protocol::parse_frame_detailed(&buf) {
                    Ok(Some((consumed, value))) => {
                        buf.drain(..consumed);
                        match classify(value) {
                            Incoming::Invalid { id, code, message } => {
                                let frame = error_frame_value(&id, code, &message, None);
                                if send_checked(&main_tx, frame).await.is_err() {
                                    break 'read;
                                }
                            }
                            Incoming::Notification { method, params } => {
                                if method == SHUTDOWN_METHOD {
                                    break 'read;
                                }
                                if is_cancel_method(&method) {
                                    // Notification form: nothing to answer.
                                    self.handle_cancel(&params, None, &registry, &lane_tx).await;
                                } else {
                                    tracing::info!(
                                        method = %method,
                                        params = %params,
                                        "acp: ignoring notification"
                                    );
                                }
                            }
                            Incoming::Request { id, method, params } => {
                                if method == SHUTDOWN_METHOD {
                                    let frame = result_frame(id, &json!({ "ok": true }));
                                    let _ = send_checked(&main_tx, frame).await;
                                    break 'read;
                                }
                                if is_cancel_method(&method) {
                                    // Short-circuit: cancel never waits
                                    // behind the dispatcher or the main
                                    // writer queue; acknowledgements use
                                    // the high-priority cancel lane.
                                    self.handle_cancel(&params, Some(id), &registry, &lane_tx)
                                        .await;
                                } else if request_tx
                                    .send(Incoming::Request { id, method, params })
                                    .await
                                    .is_err()
                                {
                                    // Dispatcher gone (writer failed or
                                    // connection ending).
                                    break 'read;
                                }
                            }
                        }
                    }
                    Ok(None) => break,
                    Err(err) => {
                        tracing::warn!(
                            error = %err.message,
                            fatal = err.fatal,
                            "acp: frame-level parse error"
                        );
                        // Official parse-error frame (null id), then either
                        // recover (byte boundary known) or end the stream.
                        let frame =
                            error_frame_value(&Value::Null, PARSE_ERROR, MSG_PARSE_ERROR, None);
                        if send_checked(&main_tx, frame).await.is_err() {
                            break 'read;
                        }
                        if err.fatal {
                            break 'read;
                        }
                        buf.drain(..err.consumed.min(buf.len()));
                    }
                }
            }
        }

        // Wind-down: cancel every running turn, drop our queue handles,
        // then join dispatcher and writer (bounded by shutdown_timeout).
        registry.cancel_all();
        drop(request_tx);
        drop(main_tx);
        drop(lane_tx);

        let mut result = match reader_error {
            Some(e) => Err(e),
            None => Ok(()),
        };
        let timeout = config.shutdown_timeout;
        match tokio::time::timeout(timeout, dispatcher_handle).await {
            Ok(Ok(Ok(()))) => {}
            Ok(Ok(Err(e))) => {
                if result.is_ok() {
                    result = Err(e);
                }
            }
            Ok(Err(join)) => {
                if result.is_ok() {
                    result = Err(format!("dispatcher task failed: {join}"));
                }
            }
            Err(_elapsed) => {
                tracing::warn!("acp: dispatcher task did not stop within shutdown timeout");
            }
        }
        match tokio::time::timeout(timeout, writer_handle).await {
            Ok(Ok(Ok(()))) => {}
            Ok(Ok(Err(e))) => {
                // A writer IO failure is the connection's real error.
                result = Err(e);
            }
            Ok(Err(join)) => {
                if result.is_ok() {
                    result = Err(format!("writer task failed: {join}"));
                }
            }
            Err(_elapsed) => {
                tracing::warn!("acp: writer task did not stop within shutdown timeout");
            }
        }
        result
    }

    /// Serve ACP over stdin/stdout. Callers must route their logging to
    /// stderr or elsewhere: stdout carries ONLY framed protocol bytes.
    pub async fn run_stdio(self) -> Result<(), String> {
        self.serve_connection(tokio::io::stdin(), tokio::io::stdout())
            .await
    }

    /// Cancel path (runs on the reader task): fire the session's token
    /// immediately, poke sync backends through their legacy abort hook on
    /// another thread, and acknowledge id-bearing cancels through the
    /// high-priority cancel lane.
    async fn handle_cancel(
        &self,
        params: &Value,
        id: Option<u64>,
        registry: &Registry,
        lane_tx: &mpsc::Sender<Vec<u8>>,
    ) {
        match require_session_id(params) {
            Ok(session_id) => {
                let fired = registry.cancel(&session_id);
                tracing::debug!(session_id = %session_id, active = fired, "acp: session cancel");
                if fired && self.engine.is_sync() {
                    self.fire_sync_abort(session_id);
                }
            }
            Err(e) => {
                // Malformed cancel: notifications are dropped (nothing can
                // answer them), requests get the typed error via the lane.
                if let Some(id) = id {
                    let frame = error_frame(id, e.code, &e.message, e.data);
                    let _ = lane_tx.send(frame).await;
                } else {
                    tracing::warn!(
                        params = %params,
                        "acp: dropping malformed cancel notification"
                    );
                }
                return;
            }
        }
        // Uniform acknowledgement: session/cancel is a notification in
        // official ACP v1, so the ack carries no outcome; the turn's
        // terminal `stopReason` response is the outcome signal.
        if let Some(id) = id {
            let frame = result_frame(id, &json!({}));
            let _ = lane_tx.send(frame).await;
        }
    }

    /// Best-effort legacy abort for sync backends. Runs off-thread when the
    /// runtime supports it; never blocks the reader on a hostile abort.
    fn fire_sync_abort(&self, session_id: String) {
        let Engine::Sync(backend) = &self.engine else {
            return;
        };
        let backend = backend.clone();
        match tokio::runtime::Handle::current().runtime_flavor() {
            tokio::runtime::RuntimeFlavor::MultiThread => {
                tokio::task::spawn_blocking(move || {
                    if let Err(e) = backend.abort(&session_id) {
                        tracing::debug!(session_id = %session_id, error = %e, "acp: abort hook");
                    }
                });
            }
            _ => {
                // Single-thread runtime (tests): run inline; the abort
                // contract says implementations return quickly.
                if let Err(e) = backend.abort(&session_id) {
                    tracing::debug!(session_id = %session_id, error = %e, "acp: abort hook");
                }
            }
        }
    }
}

fn is_cancel_method(method: &str) -> bool {
    method == AcpMethod::SessionCancel.as_str() || method == AcpMethod::SessionAbort.as_str()
}

/// Writer task: drains the cancel lane strictly before the main queue, so
/// cancel acknowledgements never wait behind a full queue of prompt
/// frames. Ends when both channels are closed and drained.
async fn writer_task<W: AsyncWrite + Unpin>(
    mut writer: W,
    mut main_rx: mpsc::Receiver<Vec<u8>>,
    mut lane_rx: mpsc::Receiver<Vec<u8>>,
) -> Result<(), String> {
    let mut main_open = true;
    let mut lane_open = true;
    while main_open || lane_open {
        // Strict lane priority: drain every queued cancel acknowledgement
        // before touching the main queue.
        if lane_open {
            loop {
                match lane_rx.try_recv() {
                    Ok(frame) => write_frame(&mut writer, &frame).await?,
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => {
                        lane_open = false;
                        break;
                    }
                }
            }
        }
        if main_open {
            match main_rx.try_recv() {
                Ok(frame) => {
                    write_frame(&mut writer, &frame).await?;
                    continue;
                }
                Err(mpsc::error::TryRecvError::Empty) => {}
                Err(mpsc::error::TryRecvError::Disconnected) => main_open = false,
            }
        }
        if !main_open && !lane_open {
            break;
        }
        // Nothing immediately available: block; lane stays preferred.
        tokio::select! {
            biased;
            lane = lane_rx.recv(), if lane_open => {
                match lane {
                    Some(frame) => write_frame(&mut writer, &frame).await?,
                    None => lane_open = false,
                }
            }
            main = main_rx.recv(), if main_open => {
                match main {
                    Some(frame) => write_frame(&mut writer, &frame).await?,
                    None => main_open = false,
                }
            }
        }
    }
    writer
        .flush()
        .await
        .map_err(|e| format!("flush error: {e}"))
}

async fn write_frame<W: AsyncWrite + Unpin>(writer: &mut W, body: &[u8]) -> Result<(), String> {
    writer
        .write_all(body)
        .await
        .map_err(|e| format!("write error: {e}"))?;
    writer
        .flush()
        .await
        .map_err(|e| format!("flush error: {e}"))
}

/// Dispatcher task: owns the per-session state machine; routes prompts to
/// per-session operation tasks; answers everything else directly.
async fn dispatcher_task(
    engine: Engine,
    registry: Registry,
    main_tx: mpsc::Sender<Vec<u8>>,
    mut request_rx: mpsc::Receiver<Incoming>,
) -> Result<(), String> {
    while let Some(Incoming::Request { id, method, params }) = request_rx.recv().await {
        dispatch_request(&engine, &registry, &main_tx, id, &method, &params).await?;
    }
    Ok(())
}

/// Dispatch one request on the dispatcher task.
async fn dispatch_request(
    engine: &Engine,
    registry: &Registry,
    main_tx: &mpsc::Sender<Vec<u8>>,
    id: u64,
    method: &str,
    params: &Value,
) -> Result<(), String> {
    match method {
        "initialize" => {
            let frame = initialize_response(id, params);
            send_checked(main_tx, frame).await
        }
        "agent_info" => {
            let frame = result_frame(id, &engine.agent_info());
            send_checked(main_tx, frame).await
        }
        "session/new" => match engine.create_session(params) {
            Ok(session_id) => {
                let frame = result_frame(id, &json!({ "sessionId": session_id }));
                send_checked(main_tx, frame).await
            }
            Err(message) => {
                let frame = internal_error_frame(id, message);
                send_checked(main_tx, frame).await
            }
        },
        "session/prompt" => dispatch_prompt(engine, registry, main_tx, id, params).await,
        "session/list" => {
            let sessions = engine
                .list_sessions()
                .into_iter()
                .map(|session_id| json!({ "sessionId": session_id }))
                .collect::<Vec<_>>();
            let frame = result_frame(id, &json!({ "sessions": sessions }));
            send_checked(main_tx, frame).await
        }
        other => {
            let frame = error_frame(id, METHOD_NOT_FOUND, MSG_METHOD_NOT_FOUND, None);
            tracing::debug!(method = %other, "acp: unknown method");
            send_checked(main_tx, frame).await
        }
    }
}

/// Per-session bookkeeping: one active turn (plus one queued prompt) per
/// session; idle entries are evicted first once `max_sessions` is hit.
#[derive(Clone)]
struct Registry {
    inner: Arc<Mutex<RegistryInner>>,
}

struct RegistryInner {
    sessions: HashMap<String, SessionState>,
    order: VecDeque<String>,
    max_sessions: usize,
    active_turns: usize,
    cancel_grace: Duration,
}

impl Registry {
    fn new(max_sessions: usize, cancel_grace: Duration) -> Self {
        Self {
            inner: Arc::new(Mutex::new(RegistryInner {
                sessions: HashMap::new(),
                order: VecDeque::new(),
                max_sessions,
                active_turns: 0,
                cancel_grace,
            })),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, RegistryInner> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn cancel_grace(&self) -> Duration {
        self.lock().cancel_grace
    }

    fn cancel(&self, session_id: &str) -> bool {
        let mut inner = self.lock();
        match inner
            .sessions
            .get_mut(session_id)
            .and_then(|s| s.active.as_mut())
        {
            Some(active) => {
                active.token.cancel();
                true
            }
            None => false,
        }
    }

    fn cancel_all(&self) {
        let inner = self.lock();
        for state in inner.sessions.values() {
            if let Some(active) = &state.active {
                active.token.cancel();
            }
        }
    }

    /// Admit a prompt turn for a session.
    /// `Start(job, token)`: the caller must run the turn on a fresh
    /// operation task; `Queued`: runs after the active turn's terminal
    /// response; `Busy`: queue depth already full; `Full`: the server's
    /// per-session capacity is exhausted (no idle entries left to evict).
    fn admit(&self, session_id: &str, job: PromptJob) -> Admit {
        let mut inner = self.lock();
        // Make room for a new entry first (never evicts active turns).
        if !inner.sessions.contains_key(session_id) {
            inner.order.push_back(session_id.to_string());
            let target = inner.max_sessions.saturating_sub(1);
            self.evict_idle_to(&mut inner, target);
        }
        let max_sessions = inner.max_sessions;
        let active_turns = inner.active_turns;
        let state = inner.sessions.entry(session_id.to_string()).or_default();
        if let Some(active) = &mut state.active {
            if active.queued.is_none() {
                active.queued = Some(job);
                return Admit::Queued;
            }
            return Admit::Busy;
        }
        if active_turns >= max_sessions && max_sessions > 0 {
            return Admit::Full;
        }
        let token = CancelToken::new();
        state.active = Some(ActiveTurn {
            token: token.clone(),
            queued: None,
        });
        inner.active_turns += 1;
        Admit::Start(job, token)
    }

    /// Evict idle entries until at most `target` entries remain. Active
    /// entries are skipped (bounded scan); nothing is ever evicted while
    /// its turn is running.
    fn evict_idle_to(&self, inner: &mut RegistryInner, target: usize) {
        let mut scanned = 0usize;
        while inner.sessions.len() > target && scanned < inner.sessions.len() {
            scanned += 1;
            let Some(oldest) = inner.order.pop_front() else {
                break;
            };
            let Some(state) = inner.sessions.get(&oldest) else {
                continue;
            };
            if state.active.is_none() {
                inner.sessions.remove(&oldest);
            } else {
                // Active entries are not evictable; revisit later.
                inner.order.push_back(oldest);
            }
        }
    }

    /// Terminal handoff after the operation task enqueued the turn's
    /// terminal response: promote the queued prompt (fresh token) or park
    /// the session as idle. Returns the job to run next, if any.
    fn finish(&self, session_id: &str, token: &CancelToken) -> Option<(PromptJob, CancelToken)> {
        let mut inner = self.lock();
        let state = inner.sessions.get_mut(session_id)?;
        let active = state.active.as_mut()?;
        if active.token != *token {
            // A newer turn already owns the session (cannot normally
            // happen: promotions happen under the same lock).
            return None;
        }
        match active.queued.take() {
            Some(job) => {
                let next_token = CancelToken::new();
                active.token = next_token.clone();
                Some((job, next_token))
            }
            None => {
                state.active = None;
                inner.active_turns = inner.active_turns.saturating_sub(1);
                None
            }
        }
    }
}

enum Admit {
    Start(PromptJob, CancelToken),
    Queued,
    Busy,
    /// Per-session capacity exhausted (bounded resource refusal).
    Full,
}

fn require_session_id(params: &Value) -> Result<String, ServerError> {
    let id = params
        .get("sessionId")
        .and_then(Value::as_str)
        .or_else(|| params.get("sessionID").and_then(Value::as_str)); // deprecated alias
    match id {
        Some(id) if !id.is_empty() => Ok(id.to_string()),
        _ => Err(ServerError::invalid_params(
            "missing string field \"sessionId\"",
        )),
    }
}

/// Extract the plain-text content of an official prompt message (array of
/// `{"type":"text","text":...}` content blocks). The deprecated `text`
/// string alias is accepted. Anything the seam cannot honestly represent
/// is refused with `-32602`.
fn require_prompt_text(params: &Value) -> Result<String, ServerError> {
    if let Some(prompt) = params.get("prompt") {
        let blocks = prompt.as_array().ok_or_else(|| {
            ServerError::invalid_params("\"prompt\" must be an array of content blocks")
        })?;
        if blocks.is_empty() {
            return Err(ServerError::invalid_params(
                "\"prompt\" must contain at least one content block",
            ));
        }
        let mut parts = Vec::new();
        for block in blocks {
            let kind = block.get("type").and_then(Value::as_str);
            match (kind, block.get("text").and_then(Value::as_str)) {
                (Some("text"), Some(text)) => parts.push(text.to_string()),
                (Some(other), _) => {
                    return Err(ServerError::invalid_params(format!(
                        "unsupported content block type \"{other}\": this agent accepts text blocks only"
                    )))
                }
                _ => {
                    return Err(ServerError::invalid_params(
                        "content block must be {\"type\":\"text\",\"text\":\"...\"}",
                    ))
                }
            }
        }
        Ok(parts.join("\n"))
    } else if let Some(text) = params.get("text").and_then(Value::as_str) {
        // Deprecated pre-conformance alias.
        Ok(text.to_string())
    } else {
        Err(ServerError::invalid_params(
            "missing \"prompt\" (array of text content blocks)",
        ))
    }
}

/// Admit a `session/prompt` into the per-session state machine. Immediate
/// parameter errors answer right away; a Start/Queued admission answers
/// asynchronously with the turn's terminal `stopReason` response.
async fn dispatch_prompt(
    engine: &Engine,
    registry: &Registry,
    main_tx: &mpsc::Sender<Vec<u8>>,
    id: u64,
    params: &Value,
) -> Result<(), String> {
    let session_id = match require_session_id(params) {
        Ok(s) => s,
        Err(e) => return respond_error(main_tx, id, e).await,
    };
    let text = match require_prompt_text(params) {
        Ok(t) => t,
        Err(e) => return respond_error(main_tx, id, e).await,
    };
    let job = PromptJob { id, text };
    match registry.admit(&session_id, job) {
        Admit::Start(job, token) => {
            spawn_turn(
                engine.clone(),
                registry.clone(),
                main_tx.clone(),
                session_id,
                job,
                token,
            );
            Ok(())
        }
        Admit::Queued => Ok(()),
        Admit::Busy => {
            let frame = error_frame(id, SESSION_BUSY, MSG_SESSION_BUSY, None);
            send_checked(main_tx, frame).await
        }
        Admit::Full => {
            let frame = error_frame(id, SESSION_LIMIT, MSG_SESSION_LIMIT, None);
            send_checked(main_tx, frame).await
        }
    }
}

fn spawn_turn(
    engine: Engine,
    registry: Registry,
    main_tx: mpsc::Sender<Vec<u8>>,
    session_id: String,
    job: PromptJob,
    token: CancelToken,
) {
    let main_tx_2 = main_tx.clone();
    let cancel_grace = registry.cancel_grace();
    std::mem::drop(tokio::spawn(async move {
        // The only per-session work in flight: run the turn, enqueue its
        // terminal response, then promote the queued prompt (if any).
        // A panicking backend must not leave the prompt unanswered.
        let config = AcpConfig {
            cancel_grace,
            ..AcpConfig::default()
        };
        let run = engine.run_turn(&session_id, &job, main_tx, token.clone(), config);
        let outcome = futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(run))
            .await
            .unwrap_or_else(|_panic| {
                tracing::error!(session_id = %session_id, "acp: backend panicked during prompt");
                TurnOutcome::Failed("backend panicked during prompt".to_string())
            });
        let frame = terminal_frame(job.id, outcome);
        if send_checked(&main_tx_2, frame).await.is_err() {
            // Writer is gone; the connection is ending.
            return;
        }
        if let Some((next_job, next_token)) = registry.finish(&session_id, &token) {
            spawn_turn(
                engine, registry, main_tx_2, session_id, next_job, next_token,
            );
        }
    }));
}

/// Terminal frame for one prompt turn: the official `stopReason` result,
/// or the official internal-error frame on backend failure.
fn terminal_frame(id: u64, outcome: TurnOutcome) -> Vec<u8> {
    match outcome {
        TurnOutcome::Completed(value) => {
            let mut result = Map::new();
            result.insert("stopReason".into(), json!("end_turn"));
            if !value.is_null() {
                // Official `_meta` extension member: the backend seam's
                // opaque report has no other official slot.
                result.insert("_meta".into(), value);
            }
            result_frame(id, &Value::Object(result))
        }
        TurnOutcome::Cancelled => {
            let mut result = Map::new();
            result.insert("stopReason".into(), json!("cancelled"));
            result_frame(id, &Value::Object(result))
        }
        TurnOutcome::Failed(message) => internal_error_frame(id, message),
    }
}

/// The official `initialize` response, or the typed version error for
/// anything that is not protocol version 1 (no silent fallback).
fn initialize_response(id: u64, params: &Value) -> Vec<u8> {
    let version = params.get("protocolVersion");
    let version_ok = match version {
        Some(Value::Number(n)) => n.as_u64() == Some(PROTOCOL_VERSION),
        Some(Value::String(s)) => s.as_str() == PROTOCOL_VERSION.to_string(),
        _ => false,
    };
    if !version_ok {
        let raw = version.cloned().unwrap_or(Value::Null);
        let mut data = Map::new();
        data.insert("protocolVersion".into(), raw);
        data.insert("supportedProtocolVersion".into(), json!(PROTOCOL_VERSION));
        let message = format!(
            "unsupported protocol version; this agent supports protocol version {PROTOCOL_VERSION}"
        );
        return error_frame(id, INVALID_PARAMS, &message, Some(Value::Object(data)));
    }
    let result = json!({
        "protocolVersion": PROTOCOL_VERSION,
        "agentCapabilities": {
            "loadSession": false,
            "promptCapabilities": {
                "audio": false,
                "embeddedContext": false,
                "image": false,
            },
        },
        "authMethods": [],
    });
    result_frame(id, &result)
}

#[derive(Debug)]
struct ServerError {
    code: i64,
    message: String,
    data: Option<Value>,
}

impl ServerError {
    fn invalid_params(message: impl Into<String>) -> Self {
        Self {
            code: INVALID_PARAMS,
            message: message.into(),
            data: None,
        }
    }
}

async fn respond_error(
    main_tx: &mpsc::Sender<Vec<u8>>,
    id: u64,
    error: ServerError,
) -> Result<(), String> {
    let frame = error_frame(id, error.code, &error.message, error.data);
    send_checked(main_tx, frame).await
}

/// Official error object: `{code, message}` with `data` omitted when absent.
fn error_object(code: i64, message: &str, data: Option<Value>) -> Value {
    let mut object = Map::new();
    object.insert("code".into(), json!(code));
    object.insert("message".into(), json!(message));
    if let Some(data) = data {
        object.insert("data".into(), data);
    }
    Value::Object(object)
}

fn error_frame_value(id: &Value, code: i64, message: &str, data: Option<Value>) -> Vec<u8> {
    let body = json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": error_object(code, message, data),
    });
    encode_or_internal(&body)
}

fn error_frame(id: u64, code: i64, message: &str, data: Option<Value>) -> Vec<u8> {
    error_frame_value(&Value::Number(id.into()), code, message, data)
}

/// `-32603` internal error with the backend message in `data` (the
/// official `into_internal_error` convention).
fn internal_error_frame(id: u64, message: String) -> Vec<u8> {
    let frame = error_frame(id, INTERNAL_ERROR, MSG_INTERNAL_ERROR, Some(json!(message)));
    if frame.len() > MAX_RESPONSE_BYTES {
        return error_frame(id, INTERNAL_ERROR, MSG_INTERNAL_ERROR, None);
    }
    frame
}

fn result_frame(id: u64, result: &Value) -> Vec<u8> {
    let body = json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    });
    let frame = encode_or_internal(&body);
    if frame.len() > MAX_RESPONSE_BYTES {
        // Refuse, never truncate: an oversized backend result must not be
        // silently cut.
        return internal_error_frame(id, "backend result exceeds the 8 MiB response bound".into());
    }
    frame
}

fn notification_frame_bytes(method: &str, params: &Value) -> Vec<u8> {
    let body = json!({
        "jsonrpc": "2.0",
        "id": null,
        "method": method,
        "params": params,
    });
    encode_or_internal(&body)
}

fn encode_or_internal(body: &Value) -> Vec<u8> {
    protocol::encode(body).expect("protocol frame encodes")
}

async fn send_checked(main_tx: &mpsc::Sender<Vec<u8>>, frame: Vec<u8>) -> Result<(), String> {
    main_tx
        .send(frame)
        .await
        .map_err(|_| "writer queue closed".to_string())
}

#[cfg(test)]
mod golden;

#[cfg(test)]
mod tests;
