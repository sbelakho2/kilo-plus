//! Official-client harness: the OFFICIAL `agent-client-protocol` crate drives
//! a real `faktor-acp::AcpServer` over byte streams.
//!
//! # Conformance
//!
//! There are no transport shims: `faktor-acp` speaks the official
//! newline-delimited JSON (NDJSON) transport directly, echoes the official
//! SDK's string/UUID request ids verbatim, omits `id` on notifications, and
//! wires `$/cancel_request` to cancellation. The official client's
//! `ByteStreams` transport is used UNMODIFIED; the only thing this harness
//! adds is a byte recorder (`WireTrace`) that observes what each side wrote
//! so tests can assert raw wire facts (no `id` member on notifications, the
//! server's frames are NDJSON lines, response ids match the UUID request
//! ids). Recording never rewrites a byte in either direction.
//!
//! Everything else is the official client end to end: the official
//! connection, JSON-RPC layer, request/notification dispatch, cancellation
//! notification, and typed ACP schema.

#![allow(dead_code)]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_client_protocol::schema::v1::{
    CancelNotification, ClientCapabilities, ContentBlock, InitializeRequest, LoadSessionRequest,
    NewSessionRequest, PromptRequest, RequestPermissionOutcome, RequestPermissionRequest,
    RequestPermissionResponse, SelectedPermissionOutcome, SessionId, SessionNotification,
    TextContent,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::{
    Agent, ByteStreams, Client, ConnectionTo, Error, ErrorCode, UntypedMessage,
};
use futures::future::BoxFuture;
use serde_json::{json, Value};
use tokio::io::{AsyncRead, AsyncWrite, DuplexStream, ReadBuf};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

use faktor_acp::{
    AcpServer, AcpStreamBackend, BackendCapabilities, EmitError, LoadSessionError, PromptCtx,
};
use faktor_protocol::v756::{Message as NativeMessage, MessagesPage, PageMeta, Part as NativePart};

/// Bound used by every test wait; a hang is a failure, never a skip.
pub const WAIT: Duration = Duration::from_secs(10);

/// In-process duplex capacity (must exceed one max-size frame for the tests).
pub const PIPE: usize = 4 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Fake backend
// ---------------------------------------------------------------------------

/// Deterministic streaming backend. Every mode is cancellable through the
/// crate's own `PromptCtx::cancelled` seam.
#[derive(Clone)]
pub struct FakeBackend {
    inner: Arc<FakeInner>,
}

struct FakeInner {
    chunks: Vec<String>,
    /// Emit a `faktor.agentStateChanged` extension frame around the chunks.
    state_frames: bool,
    /// After the chunks: `None` completes; `Some(ms)` waits up to `ms` for
    /// cancellation and completes if none arrives.
    hold_ms: Option<u64>,
    load_page: Option<MessagesPage>,
    sessions: Mutex<Vec<String>>,
    emitted: Mutex<Vec<String>>,
    post_cancel_emit_attempts: AtomicUsize,
    started: AtomicBool,
    completed: AtomicBool,
    first_chunk: tokio::sync::Notify,
}

impl FakeBackend {
    pub fn new(chunks: &[&str]) -> Self {
        Self {
            inner: Arc::new(FakeInner {
                chunks: chunks.iter().map(|c| (*c).to_string()).collect(),
                state_frames: false,
                hold_ms: None,
                load_page: None,
                sessions: Mutex::new(Vec::new()),
                emitted: Mutex::new(Vec::new()),
                post_cancel_emit_attempts: AtomicUsize::new(0),
                started: AtomicBool::new(false),
                completed: AtomicBool::new(false),
                first_chunk: tokio::sync::Notify::new(),
            }),
        }
    }

    pub fn with_state_frames(mut self, on: bool) -> Self {
        Arc::get_mut(&mut self.inner).expect("unique").state_frames = on;
        self
    }

    pub fn with_hold_ms(mut self, ms: u64) -> Self {
        Arc::get_mut(&mut self.inner).expect("unique").hold_ms = Some(ms);
        self
    }

    pub fn with_load_page(mut self, page: MessagesPage) -> Self {
        Arc::get_mut(&mut self.inner).expect("unique").load_page = Some(page);
        self
    }

    pub fn emitted(&self) -> Vec<String> {
        self.inner.emitted.lock().unwrap().clone()
    }

    pub fn started(&self) -> bool {
        self.inner.started.load(Ordering::SeqCst)
    }

    pub fn completed(&self) -> bool {
        self.inner.completed.load(Ordering::SeqCst)
    }

    pub fn post_cancel_emit_attempts(&self) -> usize {
        self.inner.post_cancel_emit_attempts.load(Ordering::SeqCst)
    }

    pub async fn await_first_chunk(&self) {
        self.inner.first_chunk.notified().await;
    }
}

impl AcpStreamBackend for FakeBackend {
    fn agent_info(&self) -> Value {
        json!({ "name": "faktor-official-test-agent", "version": "0.0.0" })
    }

    fn create_session(&self, _params: &Value) -> Result<String, String> {
        let id = "sess-1".to_string();
        self.inner.sessions.lock().unwrap().push(id.clone());
        Ok(id)
    }

    fn list_sessions(&self) -> Vec<String> {
        self.inner.sessions.lock().unwrap().clone()
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            load_session: self.inner.load_page.is_some(),
            mcp_http: false,
            mcp_sse: false,
        }
    }

    fn load_session(&self, session_id: &str) -> Result<MessagesPage, LoadSessionError> {
        match &self.inner.load_page {
            Some(page) if page.session_id == session_id => Ok(page.clone()),
            Some(_) => Err(LoadSessionError::NotFound),
            None => Err(LoadSessionError::Unsupported),
        }
    }

    fn prompt<'a>(
        &'a self,
        _session_id: &'a str,
        ctx: &'a PromptCtx,
        _text: &'a str,
    ) -> BoxFuture<'a, Result<Value, String>> {
        let inner = Arc::clone(&self.inner);
        Box::pin(async move {
            inner.started.store(true, Ordering::SeqCst);
            if inner.state_frames {
                let _ = ctx
                    .emit_agent_state(faktor_acp::AgentStateStatus::Busy, None)
                    .await;
            }
            for chunk in &inner.chunks {
                match ctx.emit_text(chunk).await {
                    Ok(()) => {
                        inner.emitted.lock().unwrap().push(chunk.clone());
                        inner.first_chunk.notify_one();
                    }
                    Err(EmitError::Cancelled) => break,
                    Err(other) => return Err(format!("emit failed: {other}")),
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            if let Some(ms) = inner.hold_ms {
                let _ = tokio::time::timeout(Duration::from_millis(ms), ctx.cancelled()).await;
            }
            if ctx.is_cancelled() {
                // Wire invariant: after cancellation no further frame may be
                // emitted; the attempt is recorded as evidence.
                if matches!(
                    ctx.emit_text("post-cancel").await,
                    Err(EmitError::Cancelled)
                ) {
                    inner
                        .post_cancel_emit_attempts
                        .fetch_add(1, Ordering::SeqCst);
                }
                return Ok(json!({ "cancelled": true }));
            }
            if inner.state_frames {
                let _ = ctx
                    .emit_agent_state(faktor_acp::AgentStateStatus::Idle, None)
                    .await;
            }
            inner.completed.store(true, Ordering::SeqCst);
            Ok(json!({ "turns": 1 }))
        })
    }
}

/// A bounded native v756 history page (newest-first), as the daemon's
/// message service produces it. Replayed chronologically by `session/load`:
/// user text, then (assistant) thought, tool call, answer text.
pub fn history_page(session: &str) -> MessagesPage {
    MessagesPage {
        session_id: session.to_string(),
        messages: vec![
            NativeMessage {
                id: "m2".into(),
                role: "assistant".into(),
                session_id: session.to_string(),
                seq: 2,
                created_ms: 0,
                parts: vec![
                    NativePart::Reasoning {
                        text: "thought".into(),
                    },
                    NativePart::ToolCall {
                        tool_call_id: "call-1".into(),
                        name: "echo".into(),
                        input: json!({ "x": 1 }),
                        state: "running".into(),
                    },
                    NativePart::Text {
                        text: "answer".into(),
                    },
                ],
            },
            NativeMessage {
                id: "m1".into(),
                role: "user".into(),
                session_id: session.to_string(),
                seq: 1,
                created_ms: 0,
                parts: vec![NativePart::Text {
                    text: "question".into(),
                }],
            },
        ],
        has_more: false,
        next_before: None,
        page: PageMeta::default(),
    }
}

// ---------------------------------------------------------------------------
// Wire trace: a byte recorder, not a shim
// ---------------------------------------------------------------------------

/// Raw bytes each side wrote, in order. The recorder wraps the transport
/// streams and never mutates a byte; `server_frames`/`client_frames` parse
/// the recorded NDJSON lines for assertions only.
#[derive(Default)]
pub struct WireTrace {
    server_bytes: Mutex<Vec<u8>>,
    client_bytes: Mutex<Vec<u8>>,
}

impl WireTrace {
    fn record_server(&self, bytes: &[u8]) {
        self.server_bytes.lock().unwrap().extend_from_slice(bytes);
    }

    fn record_client(&self, bytes: &[u8]) {
        self.client_bytes.lock().unwrap().extend_from_slice(bytes);
    }

    /// Every complete frame the server emitted, parsed in order.
    pub fn server_frames(&self) -> Vec<Value> {
        parse_lines(&self.server_bytes.lock().unwrap())
    }

    /// Every complete frame the client emitted, parsed in order.
    pub fn client_frames(&self) -> Vec<Value> {
        parse_lines(&self.client_bytes.lock().unwrap())
    }

    /// The exact bytes the server wrote (NDJSON lines).
    pub fn server_raw(&self) -> Vec<u8> {
        self.server_bytes.lock().unwrap().clone()
    }

    pub fn client_methods(&self) -> Vec<String> {
        self.client_frames()
            .iter()
            .filter_map(|frame| frame.get("method").and_then(Value::as_str))
            .map(str::to_string)
            .collect()
    }

    pub fn server_updates(&self) -> Vec<Value> {
        self.server_frames()
            .into_iter()
            .filter(|frame| frame.get("method").and_then(Value::as_str) == Some("session/update"))
            .collect()
    }
}

fn parse_lines(bytes: &[u8]) -> Vec<Value> {
    bytes
        .split(|byte| *byte == b'\n')
        .filter_map(|line| {
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            if line.iter().all(u8::is_ascii_whitespace) {
                return None;
            }
            serde_json::from_slice(line).ok()
        })
        .collect()
}

#[derive(Clone, Copy)]
enum Direction {
    Client,
    Server,
}

/// Transparent byte recorder around a transport half. `poll_write` records
/// bytes written by that side (the official client or the server); reads are
/// untouched passthrough.
struct Recorder<T> {
    inner: T,
    trace: Arc<WireTrace>,
    direction: Direction,
}

impl<T> Recorder<T> {
    fn client(inner: T, trace: Arc<WireTrace>) -> Self {
        Self {
            inner,
            trace,
            direction: Direction::Client,
        }
    }

    fn server(inner: T, trace: Arc<WireTrace>) -> Self {
        Self {
            inner,
            trace,
            direction: Direction::Server,
        }
    }

    fn record(&self, bytes: &[u8]) {
        match self.direction {
            Direction::Client => self.trace.record_client(bytes),
            Direction::Server => self.trace.record_server(bytes),
        }
    }
}

/// Reads are pure passthrough: every byte read on one side was written by
/// the other side and is already recorded at its write origin, so recording
/// reads here would mix the two directions.
impl<T: AsyncRead + Unpin> AsyncRead for Recorder<T> {
    fn poll_read(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::pin::Pin::new(&mut self.get_mut().inner).poll_read(cx, buf)
    }
}

impl<T: AsyncWrite + Unpin> AsyncWrite for Recorder<T> {
    fn poll_write(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        let this = self.get_mut();
        let poll = std::pin::Pin::new(&mut this.inner).poll_write(cx, buf);
        if let std::task::Poll::Ready(Ok(written)) = &poll {
            if *written > 0 {
                this.record(&buf[..*written]);
            }
        }
        poll
    }

    fn poll_flush(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::pin::Pin::new(&mut self.get_mut().inner).poll_flush(cx)
    }

    fn poll_shutdown(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::pin::Pin::new(&mut self.get_mut().inner).poll_shutdown(cx)
    }
}

// ---------------------------------------------------------------------------
// Harness: server + official client transport
// ---------------------------------------------------------------------------

pub struct ServerHarness {
    client_io: Option<DuplexStream>,
    pub server_task: tokio::task::JoinHandle<Result<(), String>>,
    pub trace: Arc<WireTrace>,
    pub backend: FakeBackend,
}

impl ServerHarness {
    pub fn start(backend: FakeBackend) -> Self {
        let (server_side, client_io) = tokio::io::duplex(PIPE);
        let trace = Arc::new(WireTrace::default());

        let server = AcpServer::new_streaming(backend.clone());
        let server_task = tokio::spawn({
            let trace = Arc::clone(&trace);
            async move {
                let (reader, writer) = tokio::io::split(server_side);
                server
                    .serve_connection(reader, Recorder::server(writer, trace))
                    .await
            }
        });

        Self {
            client_io: Some(client_io),
            server_task,
            trace,
            backend,
        }
    }

    fn take_client_io(&mut self) -> DuplexStream {
        self.client_io.take().expect("client transport taken once")
    }
}

// ---------------------------------------------------------------------------
// Official client
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct NotificationLog {
    entries: Mutex<Vec<(String, Value)>>,
    first: tokio::sync::Notify,
}

impl NotificationLog {
    pub fn push(&self, method: &str, params: Value) {
        self.entries
            .lock()
            .unwrap()
            .push((method.to_string(), params));
        self.first.notify_one();
    }

    pub fn len(&self) -> usize {
        self.entries.lock().unwrap().len()
    }

    pub fn entries(&self) -> Vec<(String, Value)> {
        self.entries.lock().unwrap().clone()
    }

    /// All `session/update` update bodies in arrival order.
    pub fn updates(&self) -> Vec<Value> {
        self.update_params()
            .into_iter()
            .map(|params| params["update"].clone())
            .collect()
    }

    /// All `session/update` params in arrival order.
    pub fn update_params(&self) -> Vec<Value> {
        self.entries
            .lock()
            .unwrap()
            .iter()
            .filter(|(method, _)| method == "session/update")
            .map(|(_, params)| params.clone())
            .collect()
    }

    pub async fn await_first_update(&self) {
        loop {
            if !self.updates().is_empty() {
                return;
            }
            let notified = self.first.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if !self.updates().is_empty() {
                return;
            }
            notified.await;
        }
    }
}

/// Connect the OFFICIAL client (with a raw notification collector and a
/// YOLO permission responder) to the harness server, run `main`, and return
/// its result. The official SDK owns the JSON-RPC layer end to end and its
/// `ByteStreams` transport is used unmodified.
pub async fn connect_official<R>(
    harness: &mut ServerHarness,
    log: Arc<NotificationLog>,
    main: impl AsyncFnOnce(ConnectionTo<Agent>) -> Result<R, Error>,
) -> Result<R, Error> {
    let io = harness.take_client_io();
    let trace = Arc::clone(&harness.trace);
    let (read, write) = tokio::io::split(io);
    let transport = ByteStreams::new(
        Recorder::client(write, Arc::clone(&trace)).compat_write(),
        Recorder::client(read, trace).compat(),
    );

    let notif_log = Arc::clone(&log);
    let permission_log = Arc::clone(&log);
    Client
        .builder()
        .name("faktor-official-test")
        .on_receive_notification(
            async move |notification: UntypedMessage, _cx| {
                notif_log.push(notification.method(), notification.params().clone());
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .on_receive_request(
            async move |request: RequestPermissionRequest, responder, _cx| {
                permission_log.push("client:request_permission", json!({ "request": request }));
                let option = request.options.first().map(|o| o.option_id.clone());
                match option {
                    Some(id) => responder.respond(RequestPermissionResponse::new(
                        RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(id)),
                    )),
                    None => responder.respond(RequestPermissionResponse::new(
                        RequestPermissionOutcome::Cancelled,
                    )),
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .connect_with(transport, main)
        .await
}

// ---------------------------------------------------------------------------
// Small typed helpers
// ---------------------------------------------------------------------------

pub async fn initialize(
    cx: &ConnectionTo<Agent>,
    client_capabilities: ClientCapabilities,
) -> Result<agent_client_protocol::schema::v1::InitializeResponse, Error> {
    cx.send_request(
        InitializeRequest::new(ProtocolVersion::V1).client_capabilities(client_capabilities),
    )
    .block_task()
    .await
}

pub async fn new_session(cx: &ConnectionTo<Agent>) -> Result<SessionId, Error> {
    let response = cx
        .send_request(NewSessionRequest::new("/work"))
        .block_task()
        .await?;
    Ok(response.session_id)
}

pub async fn prompt(
    cx: &ConnectionTo<Agent>,
    session_id: SessionId,
    text: &str,
) -> Result<agent_client_protocol::schema::v1::PromptResponse, Error> {
    cx.send_request(PromptRequest::new(
        session_id,
        vec![ContentBlock::Text(TextContent::new(text))],
    ))
    .block_task()
    .await
}

pub async fn cancel(cx: &ConnectionTo<Agent>, session_id: SessionId) -> Result<(), Error> {
    cx.send_notification(CancelNotification::new(session_id))
}

pub async fn load_session(
    cx: &ConnectionTo<Agent>,
    session_id: SessionId,
) -> Result<agent_client_protocol::schema::v1::LoadSessionResponse, Error> {
    cx.send_request(LoadSessionRequest::new(session_id, "/work"))
        .block_task()
        .await
}

/// Parse one recorded update through the OFFICIAL typed schema (proves the
/// server's frame is consumable by the official client's model, not just by
/// the raw collector).
pub fn typed_update(params: &Value) -> SessionNotification {
    serde_json::from_value(params.clone()).expect("official SessionNotification parse")
}

pub fn text_of(update: &Value) -> Option<String> {
    update
        .get("content")
        .and_then(|content| content.get("text"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

/// Conformance check: every notification the server emitted omits the `id`
/// member entirely (the official SDK classifies an id-bearing frame as a
/// request and would drop every update).
pub fn assert_notifications_omit_id(harness: &ServerHarness) {
    for frame in harness.trace.server_frames() {
        if frame.get("method").is_some() {
            assert!(
                frame.get("id").is_none(),
                "server notification carries an id member: {frame}"
            );
        }
    }
}

/// Conformance check: the server wrote NDJSON lines, never Content-Length
/// headers (the official `ByteStreams` client cannot parse the latter).
pub fn assert_ndjson_wire(harness: &ServerHarness) {
    let raw = harness.trace.server_raw();
    let text = String::from_utf8_lossy(&raw);
    assert!(
        !text.contains("Content-Length:"),
        "server emitted legacy Content-Length framing to an NDJSON peer: {text}"
    );
    assert!(
        raw.is_empty() || raw.ends_with(b"\n"),
        "server NDJSON output must be newline-terminated"
    );
}

pub fn assert_error_code(error: &Error, code: ErrorCode) {
    assert_eq!(error.code, code, "unexpected official error: {error:?}");
}
