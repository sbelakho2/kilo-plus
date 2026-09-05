//! Adversarial tests for the ACP v1 wire + async dispatch architecture.
//!
//! Coverage map (audits 57/58 conformance work):
//! (a) golden-shape assertions for every implemented request/response
//! (b) protocolVersion: 1 accepted, anything else rejected loudly
//! (c) cancel storm: 100 cancels -> exactly one terminal cancelled frame
//! (d) writer queue full: bounded, cancel still lands via the cancel lane
//! (e) cancel racing prompt completion: both orders deterministic
//! (f) malformed JSON -> official error frame, server keeps serving
//! (g) unknown method -> official error frame
//! (h) oversized frame (config bound) -> typed error frame, connection ends
//! (i) two concurrent sessions never interleave frames per session

use crate::golden as g;
use crate::protocol;
use crate::*;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{duplex, AsyncReadExt, AsyncWriteExt, DuplexStream};

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------

/// Test client. Two independent duplexes keep the directions isolated so a
/// stalled server->client direction never blocks client->server traffic
/// (used by the writer-full test).
struct Peer {
    to_server: DuplexStream,
    from_server: DuplexStream,
    buf: Vec<u8>,
}

impl Peer {
    async fn send_value(&mut self, value: &Value) {
        let bytes = protocol::encode(value).expect("test frame encodes");
        self.to_server.write_all(&bytes).await.expect("test write");
    }

    async fn send_raw(&mut self, bytes: &[u8]) {
        self.to_server.write_all(bytes).await.expect("test write");
    }

    /// Read one complete frame with a hard timeout. `None` on EOF.
    async fn recv_frame(&mut self) -> Option<(Vec<u8>, Value)> {
        let mut chunk = [0u8; 4096];
        loop {
            if let Ok(Some((consumed, value))) = protocol::parse_frame(&self.buf) {
                let raw = self.buf[..consumed].to_vec();
                self.buf.drain(..consumed);
                return Some((raw, value));
            }
            let n =
                tokio::time::timeout(Duration::from_secs(15), self.from_server.read(&mut chunk))
                    .await
                    .expect("test read timeout")
                    .expect("test read");
            if n == 0 {
                return None;
            }
            self.buf.extend_from_slice(&chunk[..n]);
        }
    }

    /// Read frames until `predicate` matches; panics past the deadline.
    async fn recv_until(&mut self, what: &str, predicate: impl Fn(&Value) -> bool) -> Value {
        for _ in 0..15000 {
            if let Some((_raw, frame)) = self.recv_frame().await {
                if predicate(&frame) {
                    return frame;
                }
            } else {
                panic!("connection ended while waiting for {what}");
            }
        }
        panic!("timed out waiting for {what}");
    }

    async fn recv_error(&mut self) -> Value {
        self.recv_until("error frame", |f| f.get("error").is_some())
            .await
    }
}

/// Assert the received raw frame is byte-for-byte the canonical
/// serialization of the fixture (serde_json map keys serialize sorted, so
/// re-encoding the fixture reproduces the produced bytes exactly).
fn assert_canonical(raw: &[u8], fixture: &str) {
    let expected = protocol::encode(&serde_json::from_str::<Value>(fixture).unwrap()).unwrap();
    assert_eq!(
        String::from_utf8_lossy(raw),
        String::from_utf8_lossy(&expected),
        "frame bytes differ from the canonical fixture serialization"
    );
}

fn assert_semantic(frame: &Value, fixture: &str) {
    let expected: Value = serde_json::from_str(fixture).unwrap();
    assert_eq!(frame, &expected, "frame does not match fixture shape");
}

fn spawn_server(
    server: AcpServer,
    cap_server_to_client: usize,
) -> (tokio::task::JoinHandle<Result<(), String>>, Peer) {
    let (c2s_read, c2s_write) = duplex(1024 * 1024);
    let (s2c_read, s2c_write) = duplex(cap_server_to_client);
    let peer = Peer {
        to_server: c2s_write,
        from_server: s2c_read,
        buf: Vec::new(),
    };
    let handle = tokio::spawn(async move { server.serve_connection(c2s_read, s2c_write).await });
    (handle, peer)
}

fn response_result(frame: &Value) -> &Value {
    frame.get("result").expect("result frame")
}

fn terminal_of(frame: &Value) -> Option<&str> {
    frame
        .get("result")
        .and_then(|r| r.get("stopReason"))
        .and_then(Value::as_str)
}

fn is_update(frame: &Value) -> Option<String> {
    if frame.get("method").and_then(Value::as_str) == Some("session/update") {
        frame
            .get("params")
            .and_then(|p| p.get("sessionId"))
            .and_then(Value::as_str)
            .map(str::to_string)
    } else {
        None
    }
}

fn update_text(frame: &Value) -> String {
    frame
        .get("params")
        .and_then(|p| p.get("update"))
        .and_then(|u| u.get("content"))
        .and_then(|c| c.get("text"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

// ---------------------------------------------------------------------------
// Backends
// ---------------------------------------------------------------------------

/// Synchronous echo backend with scripted failures (text "explode" fails).
#[derive(Clone)]
struct EchoBackend {
    sessions: Arc<Mutex<Vec<String>>>,
    abort_calls: Arc<AtomicUsize>,
}

impl EchoBackend {
    fn new() -> Self {
        Self {
            sessions: Arc::new(Mutex::new(Vec::new())),
            abort_calls: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl AcpBackend for EchoBackend {
    fn agent_info(&self) -> Value {
        json!({ "name": "test-agent", "version": "0.0.0" })
    }
    fn create_session(&self, params: &Value) -> Result<String, String> {
        match params.get("mode").and_then(Value::as_str) {
            Some("fail") => Err("create refused".into()),
            _ => {
                let mut sessions = self.sessions.lock().unwrap();
                let id = format!("sess-{}", sessions.len() + 1);
                sessions.push(id.clone());
                Ok(id)
            }
        }
    }
    fn prompt(&self, _session_id: &str, text: &str) -> Result<Value, String> {
        if text == "explode" {
            return Err("backend exploded".into());
        }
        Ok(json!({ "echo": text }))
    }
    fn abort(&self, _session_id: &str) -> Result<(), String> {
        self.abort_calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn list_sessions(&self) -> Vec<String> {
        self.sessions.lock().unwrap().clone()
    }
}

/// Sync backend whose prompt can be parked on a gate so the test controls
/// exactly when a running turn completes (racing-cancel tests).
#[derive(Clone)]
struct GateBackend {
    started: Arc<AtomicBool>,
    abort_calls: Arc<AtomicUsize>,
    gate_used: Arc<AtomicBool>,
    gate_rx: Arc<Mutex<Option<std::sync::mpsc::Receiver<()>>>>,
    gate_tx: std::sync::mpsc::Sender<()>,
}

impl GateBackend {
    fn new() -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        Self {
            started: Arc::new(AtomicBool::new(false)),
            abort_calls: Arc::new(AtomicUsize::new(0)),
            gate_used: Arc::new(AtomicBool::new(false)),
            gate_rx: Arc::new(Mutex::new(Some(rx))),
            gate_tx: tx,
        }
    }
}

impl AcpBackend for GateBackend {
    fn agent_info(&self) -> Value {
        json!({ "name": "gate-agent", "version": "0.0.0" })
    }
    fn create_session(&self, _params: &Value) -> Result<String, String> {
        Ok("sess-1".to_string())
    }
    fn list_sessions(&self) -> Vec<String> {
        vec!["sess-1".to_string()]
    }
    fn prompt(&self, _session_id: &str, text: &str) -> Result<Value, String> {
        self.started.store(true, Ordering::SeqCst);
        if text == "gate" && !self.gate_used.swap(true, Ordering::SeqCst) {
            let rx = self.gate_rx.lock().unwrap().take().expect("gate receiver");
            let _ = rx.recv(); // parks the running turn until the test says so
        }
        Ok(json!({ "echo": text }))
    }
    fn abort(&self, _session_id: &str) -> Result<(), String> {
        self.abort_calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

/// Streaming backend: cancellable frame streams. Behavior chosen by text:
/// - "flood": 500 text chunks (cancel-safe), counts successful emissions
/// - "states": agentStateChanged busy/idle + one text chunk
/// - "alt-a"/"alt-b": lockstep emissions across two sessions
/// - "slow": waits for a release signal, then returns
/// - anything else: one chunk + Ok(echo)
#[derive(Clone)]
struct StreamBackend {
    sessions: Arc<Mutex<Vec<String>>>,
    emissions: Arc<AtomicUsize>,
    slow_rx: Arc<Mutex<Option<mpsc::Receiver<()>>>>,
    slow_tx: mpsc::Sender<()>,
    alt: Arc<AltCoord>,
}

#[derive(Clone)]
struct AltCoord {
    /// A -> B turn permit sender; B consumes it before emitting b_i.
    ab_tx: mpsc::Sender<()>,
    ab_rx: Arc<Mutex<Option<mpsc::Receiver<()>>>>,
    /// B -> A turn permit.
    ba_tx: mpsc::Sender<()>,
    ba_rx: Arc<Mutex<Option<mpsc::Receiver<()>>>>,
}

impl StreamBackend {
    fn new() -> Self {
        let (ab_tx, ab_rx) = mpsc::channel(1);
        let (ba_tx, ba_rx) = mpsc::channel(1);
        let (slow_tx, slow_rx) = mpsc::channel(1);
        Self {
            sessions: Arc::new(Mutex::new(Vec::new())),
            emissions: Arc::new(AtomicUsize::new(0)),
            slow_rx: Arc::new(Mutex::new(Some(slow_rx))),
            slow_tx,
            alt: Arc::new(AltCoord {
                ab_tx,
                ab_rx: Arc::new(Mutex::new(Some(ab_rx))),
                ba_tx,
                ba_rx: Arc::new(Mutex::new(Some(ba_rx))),
            }),
        }
    }

    fn slow_release(&self) -> mpsc::Sender<()> {
        self.slow_tx.clone()
    }

    async fn flood_until_cancelled(ctx: &PromptCtx, emissions: &AtomicUsize) -> Result<(), String> {
        for i in 0..500 {
            if ctx.emit_text(&format!("chunk-{i}")).await.is_err() {
                return Err("cancelled".into());
            }
            emissions.fetch_add(1, Ordering::SeqCst);
            // Yield periodically so the reader task can process cancels
            // even while the writer queue never fills (fairness).
            if i % 8 == 0 {
                tokio::task::yield_now().await;
                if ctx.is_cancelled() {
                    return Err("cancelled".into());
                }
            }
        }
        Ok(())
    }

    async fn alt_a<'a>(&'a self, ctx: &'a PromptCtx) -> Result<Value, String> {
        let mut rx = self.alt.ba_rx.lock().unwrap().take().expect("ba receiver");
        for i in 1..=3 {
            ctx.emit_text(&format!("a{i}"))
                .await
                .map_err(|e| e.to_string())?;
            self.alt.ab_tx.send(()).await.map_err(|_| "peer gone")?;
            rx.recv().await.expect("B never handed the turn back");
        }
        Ok(json!({ "echo": "alt-a" }))
    }

    async fn alt_b<'a>(&'a self, ctx: &'a PromptCtx) -> Result<Value, String> {
        let mut rx = self.alt.ab_rx.lock().unwrap().take().expect("ab receiver");
        for i in 1..=3 {
            rx.recv().await.expect("A never handed the turn");
            ctx.emit_text(&format!("b{i}"))
                .await
                .map_err(|e| e.to_string())?;
            self.alt.ba_tx.send(()).await.map_err(|_| "peer gone")?;
        }
        Ok(json!({ "echo": "alt-b" }))
    }
}

impl AcpStreamBackend for StreamBackend {
    fn agent_info(&self) -> Value {
        json!({ "name": "stream-agent", "version": "0.0.0" })
    }
    fn create_session(&self, _params: &Value) -> Result<String, String> {
        let mut sessions = self.sessions.lock().unwrap();
        let id = format!("sess-{}", sessions.len() + 1);
        sessions.push(id.clone());
        Ok(id)
    }
    fn list_sessions(&self) -> Vec<String> {
        self.sessions.lock().unwrap().clone()
    }
    fn prompt<'a>(
        &'a self,
        _session_id: &'a str,
        ctx: &'a PromptCtx,
        text: &'a str,
    ) -> BoxFuture<'a, Result<Value, String>> {
        Box::pin(async move {
            match text {
                "flood" => {
                    ctx.emit_agent_state(AgentStateStatus::Busy, None)
                        .await
                        .map_err(|e| e.to_string())?;
                    Self::flood_until_cancelled(ctx, &self.emissions).await?;
                    ctx.emit_agent_state(AgentStateStatus::Idle, None)
                        .await
                        .map_err(|e| e.to_string())?;
                    Ok(json!({ "echo": "flood" }))
                }
                "states" => {
                    ctx.emit_agent_state(AgentStateStatus::Busy, Some("thinking hard"))
                        .await
                        .map_err(|e| e.to_string())?;
                    ctx.emit_text("partial").await.map_err(|e| e.to_string())?;
                    ctx.emit_agent_state(AgentStateStatus::Idle, None)
                        .await
                        .map_err(|e| e.to_string())?;
                    Ok(json!({ "echo": "states" }))
                }
                "alt-a" => self.alt_a(ctx).await,
                "alt-b" => self.alt_b(ctx).await,
                "slow" => {
                    let mut rx = self.slow_rx.lock().unwrap().take().expect("slow receiver");
                    rx.recv().await.expect("release");
                    Ok(json!({ "echo": "slow" }))
                }
                _ => {
                    ctx.emit_text(text).await.map_err(|e| e.to_string())?;
                    Ok(json!({ "echo": text }))
                }
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Golden-shape tests
// ---------------------------------------------------------------------------

#[test]
fn update_frame_builders_serialize_exactly() {
    // Byte-exact canonical serialization of the public frame builders.
    assert_eq!(
        agent_state_changed_update(AgentStateStatus::Busy, None).to_string(),
        r#"{"agentState":{"status":"busy"},"kind":"agentStateChanged"}"#
    );
    assert_eq!(
        agent_state_changed_update(AgentStateStatus::Busy, Some("thinking hard")).to_string(),
        r#"{"agentState":{"message":"thinking hard","status":"busy"},"kind":"agentStateChanged"}"#
    );
    assert_eq!(
        agent_state_changed_update(AgentStateStatus::Error, Some("boom")).to_string(),
        r#"{"agentState":{"message":"boom","status":"error"},"kind":"agentStateChanged"}"#
    );
    assert_eq!(
        agent_state_changed_update(AgentStateStatus::Idle, None).to_string(),
        r#"{"agentState":{"status":"idle"},"kind":"agentStateChanged"}"#
    );
    assert_eq!(
        text_chunk_update("partial").to_string(),
        r#"{"content":{"text":"partial","type":"text"},"sessionUpdate":"agent_message_chunk"}"#
    );
    let params = session_update_params("sess-1", text_chunk_update("partial"));
    assert_eq!(
        params.to_string(),
        r#"{"sessionId":"sess-1","update":{"content":{"text":"partial","type":"text"},"sessionUpdate":"agent_message_chunk"}}"#
    );
}

#[tokio::test]
async fn a_golden_wire_shapes_end_to_end() {
    let (_handle, mut peer) = spawn_server(AcpServer::new(EchoBackend::new()), 1024 * 1024);

    // initialize: official request shape accepted, official response shape.
    let request: Value = serde_json::from_str(g::INITIALIZE_REQUEST_V1).unwrap();
    peer.send_value(&request).await;
    let (raw, frame) = peer.recv_frame().await.expect("initialize response");
    assert_semantic(&frame, g::INITIALIZE_RESPONSE);
    assert_canonical(&raw, g::INITIALIZE_RESPONSE);

    // session/new -> {sessionId}.
    let request: Value = serde_json::from_str(g::SESSION_NEW_REQUEST).unwrap();
    peer.send_value(&request).await;
    let (raw, frame) = peer.recv_frame().await.expect("session/new response");
    assert_semantic(&frame, g::SESSION_NEW_RESPONSE);
    assert_canonical(&raw, g::SESSION_NEW_RESPONSE);

    // session/prompt (official content-block message) -> stopReason + _meta.
    let request: Value = serde_json::from_str(g::PROMPT_REQUEST).unwrap();
    peer.send_value(&request).await;
    let (raw, frame) = peer.recv_frame().await.expect("prompt response");
    assert_semantic(&frame, g::PROMPT_RESPONSE_END_TURN);
    assert_canonical(&raw, g::PROMPT_RESPONSE_END_TURN);

    // Backend failure -> official internal error frame with data.
    let boom = json!({ "jsonrpc": "2.0", "id": 3, "method": "session/prompt",
                       "params": { "sessionId": "sess-1", "prompt": [{ "type": "text", "text": "explode" }] } });
    peer.send_value(&boom).await;
    let (raw, frame) = peer.recv_frame().await.expect("internal error");
    assert_semantic(&frame, g::ERROR_INTERNAL_BACKEND);
    assert_canonical(&raw, g::ERROR_INTERNAL_BACKEND);

    // session/cancel on an idle session -> empty-result ack (notification
    // semantics; no state corruption).
    let cancel: Value = serde_json::from_str(g::CANCEL_REQUEST).unwrap();
    peer.send_value(&cancel).await;
    let (raw, frame) = peer.recv_frame().await.expect("cancel ack");
    assert_semantic(&frame, g::CANCEL_ACK_RESPONSE);
    assert_canonical(&raw, g::CANCEL_ACK_RESPONSE);

    // The sync path emits NO session/update frames; everything above was a
    // direct response. The connection still serves afterwards.
    let again = json!({ "jsonrpc": "2.0", "id": 7, "method": "session/new", "params": {} });
    peer.send_value(&again).await;
    let frame = peer.recv_frame().await.expect("second session").1;
    assert_eq!(response_result(&frame)["sessionId"], "sess-2");

    // Invalid params error golden.
    let no_sid = json!({ "jsonrpc": "2.0", "id": 3, "method": "session/prompt", "params": {} });
    peer.send_value(&no_sid).await;
    let frame = peer.recv_frame().await.expect("invalid params").1;
    assert_semantic(&frame, g::ERROR_INVALID_PARAMS_SESSION);
}

#[tokio::test]
async fn a_update_frames_match_official_and_state_shapes() {
    let backend = StreamBackend::new();
    let (_handle, mut peer) = spawn_server(AcpServer::new_streaming(backend), 1024 * 1024);

    // Prompt "states" on session sess-1 (created via session/new first so
    // the fixture sessionId matches).
    let new = json!({ "jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {} });
    peer.send_value(&new).await;
    peer.recv_frame().await.expect("session/new");

    let prompt = json!({ "jsonrpc": "2.0", "id": 3, "method": "session/prompt",
                         "params": { "sessionId": "sess-1", "prompt": [{ "type": "text", "text": "states" }] } });
    peer.send_value(&prompt).await;

    let busy = peer
        .recv_until("busy frame", |f| {
            is_update(f).as_deref() == Some("sess-1")
                && f["params"]["update"]["kind"] == "agentStateChanged"
                && f["params"]["update"]["agentState"]["status"] == "busy"
        })
        .await;
    assert_semantic(&busy, g::UPDATE_FRAME_STATE_BUSY_MESSAGE);

    let chunk = peer
        .recv_until("text chunk", |f| {
            is_update(f).as_deref() == Some("sess-1")
                && f["params"]["update"]["sessionUpdate"] == "agent_message_chunk"
        })
        .await;
    assert_semantic(&chunk, g::UPDATE_FRAME_TEXT_CHUNK);

    let idle = peer
        .recv_until("idle frame", |f| {
            is_update(f).as_deref() == Some("sess-1")
                && f["params"]["update"]["kind"] == "agentStateChanged"
                && f["params"]["update"]["agentState"]["status"] == "idle"
        })
        .await;
    assert_semantic(&idle, g::UPDATE_FRAME_STATE_IDLE);

    let terminal = peer
        .recv_until("terminal", |f| f.get("id") == Some(&json!(3)))
        .await;
    assert_eq!(terminal_of(&terminal), Some("end_turn"));
}

#[test]
fn a_update_frame_goldens_reference_exact_shapes() {
    // Pin the golden constants themselves: every asserted fixture is
    // structurally valid JSON with the exact fields of the shape they
    // document.
    for fixture in [
        g::INITIALIZE_REQUEST_V1,
        g::INITIALIZE_RESPONSE,
        g::SESSION_NEW_REQUEST,
        g::SESSION_NEW_RESPONSE,
        g::PROMPT_REQUEST,
        g::PROMPT_RESPONSE_END_TURN,
        g::PROMPT_RESPONSE_CANCELLED,
        g::CANCEL_NOTIFICATION,
        g::CANCEL_REQUEST,
        g::CANCEL_ACK_RESPONSE,
        g::UPDATE_FRAME_STATE_BUSY,
        g::UPDATE_FRAME_STATE_BUSY_MESSAGE,
        g::UPDATE_FRAME_STATE_IDLE,
        g::UPDATE_FRAME_STATE_ERROR,
        g::UPDATE_FRAME_TEXT_CHUNK,
        g::ERROR_METHOD_NOT_FOUND,
        g::ERROR_PARSE,
        g::ERROR_INTERNAL_BACKEND,
        g::ERROR_INVALID_PARAMS_SESSION,
        g::ERROR_SESSION_BUSY,
    ] {
        let v: Value = serde_json::from_str(fixture).expect("golden fixture is valid JSON");
        assert!(v.is_object());
    }
}

// ---------------------------------------------------------------------------
// (b) Version handshake
// ---------------------------------------------------------------------------

#[tokio::test]
async fn b_protocol_version_1_accepted_others_rejected_loudly() {
    let (_handle, mut peer) = spawn_server(AcpServer::new(EchoBackend::new()), 1024 * 1024);

    // protocolVersion 1 (official) accepted.
    let v1: Value = serde_json::from_str(g::INITIALIZE_REQUEST_V1).unwrap();
    peer.send_value(&v1).await;
    let (raw, frame) = peer.recv_frame().await.expect("v1 accepted");
    assert_semantic(&frame, g::INITIALIZE_RESPONSE);
    assert_canonical(&raw, g::INITIALIZE_RESPONSE);
    assert_eq!(response_result(&frame)["protocolVersion"], 1);

    // protocolVersion 2 -> typed error, no silent fallback.
    let v2: Value = serde_json::from_str(g::INITIALIZE_REQUEST_V2).unwrap();
    peer.send_value(&v2).await;
    let (raw, frame) = peer.recv_frame().await.expect("v2 rejected");
    assert_semantic(&frame, g::INITIALIZE_ERROR_V2);
    assert_canonical(&raw, g::INITIALIZE_ERROR_V2);
    assert_eq!(frame["error"]["code"], -32602);
    assert_eq!(frame["error"]["data"]["supportedProtocolVersion"], 1);
    assert_eq!(frame["error"]["data"]["protocolVersion"], 2);

    // Legacy string versions (the crate's own old wire) -> loud rejection.
    let legacy: Value = serde_json::from_str(g::INITIALIZE_REQUEST_LEGACY_STRING).unwrap();
    peer.send_value(&legacy).await;
    let (raw, frame) = peer.recv_frame().await.expect("legacy rejected");
    assert_semantic(&frame, g::INITIALIZE_ERROR_LEGACY_STRING);
    assert_canonical(&raw, g::INITIALIZE_ERROR_LEGACY_STRING);

    // Missing protocolVersion -> typed error too.
    let missing = json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {} });
    peer.send_value(&missing).await;
    let frame = peer.recv_frame().await.expect("missing version rejected").1;
    assert_eq!(frame["error"]["code"], -32602);

    // The string "1" is accepted (it is protocol version 1).
    let one_str = json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize",
                          "params": { "protocolVersion": "1" } });
    peer.send_value(&one_str).await;
    let frame = peer.recv_frame().await.expect("string 1 accepted").1;
    assert_eq!(response_result(&frame)["protocolVersion"], 1);

    // The server keeps serving after every rejection (no state corruption).
    let new = json!({ "jsonrpc": "2.0", "id": 2, "method": "session/new", "params": {} });
    peer.send_value(&new).await;
    let frame = peer
        .recv_frame()
        .await
        .expect("session/new after rejections")
        .1;
    assert_eq!(response_result(&frame)["sessionId"], "sess-1");
}

// ---------------------------------------------------------------------------
// (c) Cancel storm
// ---------------------------------------------------------------------------

#[tokio::test]
async fn c_cancel_storm_single_terminal_no_dead_session() {
    let backend = StreamBackend::new();
    let (_handle, mut peer) = spawn_server(AcpServer::new_streaming(backend), 1024 * 1024);

    let new = json!({ "jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {} });
    peer.send_value(&new).await;
    peer.recv_frame().await.expect("session/new");

    // Start a long flood.
    let prompt = json!({ "jsonrpc": "2.0", "id": 3, "method": "session/prompt",
                         "params": { "sessionId": "sess-1", "prompt": [{ "type": "text", "text": "flood" }] } });
    peer.send_value(&prompt).await;
    peer.recv_until("first update", |f| is_update(f).is_some())
        .await;

    // 100 cancels (official notification form) against the running prompt.
    let cancel: Value = serde_json::from_str(g::CANCEL_NOTIFICATION).unwrap();
    for _ in 0..100 {
        peer.send_value(&cancel).await;
    }

    // Drain: exactly one terminal cancel frame, no duplicate cancels.
    // Cancel notifications are answerless; no frame may carry a second
    // terminal or a cancel response.
    let mut terminals = Vec::new();
    let mut chunks = 0usize;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        match tokio::time::timeout(remaining, peer.recv_frame()).await {
            Ok(Some((_raw, frame))) => {
                if let Some(reason) = terminal_of(&frame) {
                    terminals.push((frame["id"].clone(), reason.to_string()));
                    break;
                }
                if is_update(&frame).is_some()
                    && frame["params"]["update"]["sessionUpdate"] == "agent_message_chunk"
                {
                    chunks += 1;
                }
                // Notifications never produce a matching response frame.
                assert!(
                    frame.get("method") != Some(&json!("session/cancel")),
                    "a cancel notification must not be answered"
                );
            }
            Ok(None) => break,
            Err(_) => break,
        }
    }
    assert_eq!(
        terminals.len(),
        1,
        "exactly one terminal frame expected, got {terminals:?}"
    );
    assert_eq!(terminals[0].1, "cancelled");
    assert_eq!(terminals[0].0, json!(3));
    // The model stream stopped early: nowhere near the full 500 chunks.
    assert!(
        chunks < 500,
        "chunks after cancel must stop early ({chunks} emitted)"
    );

    // No dead session: the next prompt runs to completion.
    let again = json!({ "jsonrpc": "2.0", "id": 6, "method": "session/prompt",
                        "params": { "sessionId": "sess-1", "prompt": [{ "type": "text", "text": "again" }] } });
    peer.send_value(&again).await;
    let terminal = peer
        .recv_until("second prompt terminal", |f| f.get("id") == Some(&json!(6)))
        .await;
    assert_eq!(terminal_of(&terminal), Some("end_turn"));
    assert_eq!(response_result(&terminal)["_meta"]["echo"], "again");
}

// ---------------------------------------------------------------------------
// (d) Writer queue full: bounded, cancel lane still lands
// ---------------------------------------------------------------------------

#[tokio::test]
async fn d_writer_queue_full_is_bounded_and_cancel_lane_lands() {
    let config = AcpConfig {
        writer_queue_capacity: 2,
        ..AcpConfig::default()
    };
    let backend = StreamBackend::new();
    let emissions = backend.emissions.clone();
    let (_handle, mut peer) = spawn_server(
        AcpServer::new_streaming(backend).with_config(config),
        512, // tiny server->client buffer: the writer stalls quickly
    );

    let new = json!({ "jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {} });
    peer.send_value(&new).await;
    peer.recv_frame().await.expect("session/new");

    // Flood while the client does not read: the writer stalls on the tiny
    // transport buffer and the main queue (capacity 2) fills.
    let prompt = json!({ "jsonrpc": "2.0", "id": 3, "method": "session/prompt",
                         "params": { "sessionId": "sess-1", "prompt": [{ "type": "text", "text": "flood" }] } });
    peer.send_value(&prompt).await;

    // Wait for the backend to stall: emissions stop growing while the
    // client reads nothing => queue depth is bounded by the capacity.
    tokio::time::sleep(Duration::from_millis(200)).await;
    let stalled = emissions.load(Ordering::SeqCst);
    tokio::time::sleep(Duration::from_millis(200)).await;
    let still = emissions.load(Ordering::SeqCst);
    assert_eq!(
        stalled, still,
        "emissions must stall, not buffer unboundedly"
    );
    assert!(
        stalled <= 4,
        "bounded main queue (capacity 2) must cap in-flight frames, got {stalled}"
    );
    assert!(stalled >= 1, "flood must have started");

    // Cancel lands while the main queue is full: id-bearing request; the
    // ack must flow through the high-priority cancel lane.
    let cancel = json!({ "jsonrpc": "2.0", "id": 9, "method": "session/cancel",
                         "params": { "sessionId": "sess-1" } });
    peer.send_value(&cancel).await;
    tokio::time::sleep(Duration::from_millis(200)).await;
    // The backend must have stopped emitting (token fired mid-run).
    let post_cancel = emissions.load(Ordering::SeqCst);
    assert_eq!(post_cancel, stalled, "cancel must stop further emissions");

    // Resume reading: everything drains; the cancel ack is not stuck
    // behind the full queue of prompt frames.
    let mut saw_ack = false;
    let mut terminal: Option<Value> = None;
    let mut chunks_after_ack = 0usize;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    while tokio::time::Instant::now() < deadline && terminal.is_none() {
        match tokio::time::timeout(Duration::from_secs(2), peer.recv_frame()).await {
            Ok(Some((_raw, frame))) => {
                if frame.get("id") == Some(&json!(9)) && frame.get("result").is_some() {
                    assert_eq!(frame["result"], json!({}), "cancel ack shape");
                    saw_ack = true;
                    continue;
                }
                if is_update(&frame).is_some()
                    && frame["params"]["update"]["sessionUpdate"] == "agent_message_chunk"
                {
                    if saw_ack {
                        chunks_after_ack += 1;
                    }
                    continue;
                }
                if let Some(reason) = terminal_of(&frame) {
                    terminal = Some(frame.clone());
                    assert_eq!(reason, "cancelled");
                }
            }
            Ok(None) => break,
            Err(_) => break,
        }
    }
    let terminal = terminal.expect("cancelled terminal must arrive after resuming reads");
    assert!(saw_ack, "cancel ack must be emitted");
    assert_eq!(terminal["id"], json!(3));
    assert_eq!(terminal_of(&terminal), Some("cancelled"));
    assert!(
        chunks_after_ack <= 2,
        "at most the bounded queue (capacity 2) may drain after the cancel ack, got {chunks_after_ack}"
    );

    // No dead session and no reader deadlock: a new prompt completes.
    let again = json!({ "jsonrpc": "2.0", "id": 6, "method": "session/prompt",
                        "params": { "sessionId": "sess-1", "prompt": [{ "type": "text", "text": "again" }] } });
    peer.send_value(&again).await;
    let terminal = peer
        .recv_until("post-stall prompt", |f| f.get("id") == Some(&json!(6)))
        .await;
    assert_eq!(terminal_of(&terminal), Some("end_turn"));
}

// ---------------------------------------------------------------------------
// (e) Cancel racing prompt completion, both orders
// ---------------------------------------------------------------------------

async fn e_gate_started(backend: &GateBackend) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while !backend.started.load(Ordering::SeqCst) {
        assert!(
            tokio::time::Instant::now() < deadline,
            "gate prompt never started"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn e_cancel_before_terminal_yields_cancelled_state() {
    let backend = GateBackend::new();
    let (_handle, mut peer) = spawn_server(AcpServer::new(backend.clone()), 1024 * 1024);

    let new = json!({ "jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {} });
    peer.send_value(&new).await;
    peer.recv_frame().await.expect("session/new");

    // The turn parks inside the backend (a RUNNING prompt).
    let prompt = json!({ "jsonrpc": "2.0", "id": 3, "method": "session/prompt",
                         "params": { "sessionId": "sess-1", "prompt": [{ "type": "text", "text": "gate" }] } });
    peer.send_value(&prompt).await;
    e_gate_started(&backend).await;

    // Cancel while the turn is running: the cancel reaches the running
    // prompt (legacy abort hook for the sync seam).
    let cancel = json!({ "jsonrpc": "2.0", "id": 5, "method": "session/cancel",
                         "params": { "sessionId": "sess-1" } });
    peer.send_value(&cancel).await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(
        backend.abort_calls.load(Ordering::SeqCst),
        1,
        "cancel must reach the running prompt through the abort hook"
    );

    // Let the parked backend return; the terminal decision point must pick
    // cancelled.
    backend.gate_tx.send(()).unwrap();

    let mut ack = false;
    let mut terminal: Option<Value> = None;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    while tokio::time::Instant::now() < deadline && terminal.is_none() {
        match tokio::time::timeout(Duration::from_secs(2), peer.recv_frame()).await {
            Ok(Some((_raw, frame))) => {
                if frame.get("id") == Some(&json!(5)) && frame.get("result").is_some() {
                    assert_semantic(&frame, g::CANCEL_ACK_RESPONSE);
                    ack = true;
                } else if let Some(reason) = terminal_of(&frame) {
                    assert_eq!(frame["id"], json!(3));
                    assert_eq!(reason, "cancelled");
                    assert_semantic(&frame, g::PROMPT_RESPONSE_CANCELLED);
                    terminal = Some(frame);
                }
            }
            Ok(None) => break,
            Err(_) => break,
        }
    }
    assert!(ack, "cancel ack missing");
    assert!(terminal.is_some(), "cancelled terminal missing");

    // No dead session: the next prompt completes normally.
    let again = json!({ "jsonrpc": "2.0", "id": 6, "method": "session/prompt",
                        "params": { "sessionId": "sess-1", "prompt": [{ "type": "text", "text": "again" }] } });
    peer.send_value(&again).await;
    let terminal = peer
        .recv_until("post-cancel prompt", |f| f.get("id") == Some(&json!(6)))
        .await;
    assert_eq!(terminal_of(&terminal), Some("end_turn"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn e_terminal_before_cancel_is_a_noop_ack() {
    let backend = GateBackend::new();
    let (_handle, mut peer) = spawn_server(AcpServer::new(backend.clone()), 1024 * 1024);

    let new = json!({ "jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {} });
    peer.send_value(&new).await;
    peer.recv_frame().await.expect("session/new");

    let prompt = json!({ "jsonrpc": "2.0", "id": 3, "method": "session/prompt",
                         "params": { "sessionId": "sess-1", "prompt": [{ "type": "text", "text": "gate" }] } });
    peer.send_value(&prompt).await;
    e_gate_started(&backend).await;

    // Terminal first: release the turn BEFORE any cancel arrives.
    backend.gate_tx.send(()).unwrap();
    let terminal = peer
        .recv_until("end_turn terminal", |f| f.get("id") == Some(&json!(3)))
        .await;
    assert_eq!(terminal_of(&terminal), Some("end_turn"));

    // The cancel now races a finished turn: acknowledged as a no-op (idle
    // semantics per official notification behavior) — no second terminal,
    // no state corruption.
    let cancel = json!({ "jsonrpc": "2.0", "id": 5, "method": "session/cancel",
                         "params": { "sessionId": "sess-1" } });
    peer.send_value(&cancel).await;
    let ack = peer
        .recv_until("cancel ack", |f| f.get("id") == Some(&json!(5)))
        .await;
    assert_semantic(&ack, g::CANCEL_ACK_RESPONSE);
    assert_eq!(
        backend.abort_calls.load(Ordering::SeqCst),
        0,
        "nothing to abort"
    );

    // Exactly one terminal frame was produced for the whole turn.
    let again = json!({ "jsonrpc": "2.0", "id": 6, "method": "session/prompt",
                        "params": { "sessionId": "sess-1", "prompt": [{ "type": "text", "text": "again" }] } });
    peer.send_value(&again).await;
    let terminal2 = peer
        .recv_until("post terminal prompt", |f| f.get("id") == Some(&json!(6)))
        .await;
    assert_eq!(terminal_of(&terminal2), Some("end_turn"));
}

// ---------------------------------------------------------------------------
// (f)/(g)/(h) Malformed JSON, unknown method, oversized frame
// ---------------------------------------------------------------------------

#[tokio::test]
async fn f_malformed_json_yields_official_error_and_serving_continues() {
    let (_handle, mut peer) = spawn_server(AcpServer::new(EchoBackend::new()), 1024 * 1024);

    // Unparseable body inside a well-framed message.
    peer.send_raw(b"Content-Length: 8\r\n\r\n{\"broken").await;
    let (raw, frame) = peer.recv_frame().await.expect("parse error frame");
    assert_semantic(&frame, g::ERROR_PARSE);
    assert_canonical(&raw, g::ERROR_PARSE);
    assert!(frame["id"].is_null());

    // Garbage that is not even JSON-RPC-shaped (valid JSON, wrong shape).
    let body = b"\"hello\"";
    let framed = format!(
        "Content-Length: {}\r\n\r\n{}",
        body.len(),
        String::from_utf8_lossy(body)
    );
    peer.send_raw(framed.as_bytes()).await;
    let frame = peer.recv_error().await;
    assert_eq!(frame["error"]["code"], -32600);

    // The server keeps serving without panicking or corrupting sessions.
    let new = json!({ "jsonrpc": "2.0", "id": 2, "method": "session/new", "params": {} });
    peer.send_value(&new).await;
    let frame = peer.recv_frame().await.expect("session/new").1;
    assert_eq!(response_result(&frame)["sessionId"], "sess-1");
}

#[tokio::test]
async fn g_unknown_method_yields_official_error_and_serving_continues() {
    let (_handle, mut peer) = spawn_server(AcpServer::new(EchoBackend::new()), 1024 * 1024);

    let unknown = json!({ "jsonrpc": "2.0", "id": 9, "method": "bogus/method", "params": {} });
    peer.send_value(&unknown).await;
    let (raw, frame) = peer.recv_frame().await.expect("method-not-found frame");
    assert_semantic(&frame, g::ERROR_METHOD_NOT_FOUND);
    assert_canonical(&raw, g::ERROR_METHOD_NOT_FOUND);
    assert_eq!(frame["error"]["code"], -32601);
    assert!(
        frame["error"].get("data").is_none(),
        "no data member when absent"
    );

    let new = json!({ "jsonrpc": "2.0", "id": 2, "method": "session/new", "params": {} });
    peer.send_value(&new).await;
    let frame = peer.recv_frame().await.expect("session/new").1;
    assert_eq!(response_result(&frame)["sessionId"], "sess-1");
}

#[tokio::test]
async fn h_oversized_frame_yields_typed_error_and_connection_ends() {
    let (_handle, mut peer) = spawn_server(AcpServer::new(EchoBackend::new()), 1024 * 1024);

    // Declared Content-Length beyond the 16 MiB frame bound: hostile
    // header, refused before any body is buffered.
    let hostile = format!("Content-Length: {}\r\n\r\n", 20 * 1024 * 1024);
    peer.send_raw(hostile.as_bytes()).await;

    let (raw, frame) = peer.recv_frame().await.expect("typed error frame");
    assert_semantic(&frame, g::ERROR_PARSE);
    assert_canonical(&raw, g::ERROR_PARSE);
    assert!(frame["id"].is_null());

    // The connection ends after the typed error (fatal framing violation).
    assert!(
        peer.recv_frame().await.is_none(),
        "connection must close after a fatal framing error"
    );
}

// ---------------------------------------------------------------------------
// (i) Two concurrent sessions: per-session order preserved
// ---------------------------------------------------------------------------

#[tokio::test]
async fn i_concurrent_prompts_keep_per_session_frame_order() {
    let backend = StreamBackend::new();
    let (_handle, mut peer) = spawn_server(AcpServer::new_streaming(backend), 1024 * 1024);

    for id in [1u64, 2] {
        let new = json!({ "jsonrpc": "2.0", "id": id, "method": "session/new", "params": {} });
        peer.send_value(&new).await;
        peer.recv_frame().await.expect("session/new");
    }

    // Launch both turns back to back: A and B are different sessions, so
    // their operation tasks run concurrently and lockstep through one
    // shared writer (deterministic alternation via turn permits).
    let prompt_a = json!({ "jsonrpc": "2.0", "id": 10, "method": "session/prompt",
                           "params": { "sessionId": "sess-1", "prompt": [{ "type": "text", "text": "alt-a" }] } });
    let prompt_b = json!({ "jsonrpc": "2.0", "id": 11, "method": "session/prompt",
                           "params": { "sessionId": "sess-2", "prompt": [{ "type": "text", "text": "alt-b" }] } });
    peer.send_value(&prompt_a).await;
    peer.send_value(&prompt_b).await;

    // The shared writer must emit the strict alternation a1 b1 a2 b2 a3 b3:
    // no session's frames are reordered or duplicated on the wire.
    let expected = ["a1", "b1", "a2", "b2", "a3", "b3"];
    for (i, expected_text) in expected.iter().enumerate() {
        let frame = peer
            .recv_until(&format!("update {expected_text}"), |f| {
                is_update(f).is_some() && update_text(f) == *expected_text
            })
            .await;
        let session = is_update(&frame).unwrap();
        let parity = i % 2 == 0;
        assert_eq!(
            session == "sess-1",
            parity,
            "update {expected_text} arrived from the wrong session"
        );
    }

    // Both turns then complete with their own terminals (either order).
    let mut saw_a = false;
    let mut saw_b = false;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    while (!saw_a || !saw_b) && tokio::time::Instant::now() < deadline {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        match tokio::time::timeout(remaining, peer.recv_frame()).await {
            Ok(Some((_raw, frame))) => match frame.get("id").and_then(Value::as_u64) {
                Some(10) => {
                    assert_eq!(terminal_of(&frame), Some("end_turn"));
                    saw_a = true;
                }
                Some(11) => {
                    assert_eq!(terminal_of(&frame), Some("end_turn"));
                    saw_b = true;
                }
                _ => panic!("unexpected frame: {frame}"),
            },
            Ok(None) => panic!("connection ended before both terminals"),
            Err(_) => break,
        }
    }
    assert!(
        saw_a && saw_b,
        "both sessions must terminate (a={saw_a} b={saw_b})"
    );
}

// ---------------------------------------------------------------------------
// Session state machine: same-session queueing, busy refusal, order
// ---------------------------------------------------------------------------

#[tokio::test]
async fn same_session_prompts_serialize_and_third_is_busy() {
    let backend = StreamBackend::new();
    let release = backend.slow_release();
    let (_handle, mut peer) = spawn_server(AcpServer::new_streaming(backend), 1024 * 1024);

    let new = json!({ "jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {} });
    peer.send_value(&new).await;
    peer.recv_frame().await.expect("session/new");

    let slow = json!({ "jsonrpc": "2.0", "id": 10, "method": "session/prompt",
                       "params": { "sessionId": "sess-1", "prompt": [{ "type": "text", "text": "slow" }] } });
    let fast1 = json!({ "jsonrpc": "2.0", "id": 11, "method": "session/prompt",
                        "params": { "sessionId": "sess-1", "prompt": [{ "type": "text", "text": "one" }] } });
    let fast2 = json!({ "jsonrpc": "2.0", "id": 12, "method": "session/prompt",
                        "params": { "sessionId": "sess-1", "prompt": [{ "type": "text", "text": "two" }] } });
    peer.send_value(&slow).await;
    peer.send_value(&fast1).await;
    peer.send_value(&fast2).await;

    // Turn 10 runs, 11 queues behind it, 12 is refused with the typed busy
    // error while the session is occupied.
    let busy = peer
        .recv_until("busy error", |f| f.get("id") == Some(&json!(12)))
        .await;
    assert_semantic(&busy, g::ERROR_SESSION_BUSY);

    // Release the slow turn; terminals arrive in FIFO order 10 then 11 and
    // a same-session prompt after that runs immediately.
    release.send(()).await.unwrap();
    let t10 = peer
        .recv_until("terminal 10", |f| f.get("id") == Some(&json!(10)))
        .await;
    assert_eq!(terminal_of(&t10), Some("end_turn"));
    let t11 = peer
        .recv_until("terminal 11", |f| f.get("id") == Some(&json!(11)))
        .await;
    assert_eq!(terminal_of(&t11), Some("end_turn"));
    assert_eq!(response_result(&t11)["_meta"]["echo"], "one");

    let fast3 = json!({ "jsonrpc": "2.0", "id": 13, "method": "session/prompt",
                        "params": { "sessionId": "sess-1", "prompt": [{ "type": "text", "text": "three" }] } });
    peer.send_value(&fast3).await;
    let t13 = peer
        .recv_until("terminal 13", |f| f.get("id") == Some(&json!(13)))
        .await;
    assert_eq!(terminal_of(&t13), Some("end_turn"));
}

// ---------------------------------------------------------------------------
// Deprecated aliases and cancel on unknown sessions
// ---------------------------------------------------------------------------

#[tokio::test]
async fn deprecated_aliases_still_work_and_cancel_never_poisons() {
    let (_handle, mut peer) = spawn_server(AcpServer::new(EchoBackend::new()), 1024 * 1024);

    // Legacy prompt params (sessionID + text) accepted.
    let legacy_prompt = json!({ "jsonrpc": "2.0", "id": 3, "method": "session/prompt",
                                "params": { "sessionID": "sess-unknown", "text": "legacy" } });
    peer.send_value(&legacy_prompt).await;
    let terminal = peer
        .recv_until("legacy prompt terminal", |f| f.get("id") == Some(&json!(3)))
        .await;
    assert_eq!(terminal_of(&terminal), Some("end_turn"));

    // Legacy session/abort alias: identical semantics to session/cancel.
    let abort = json!({ "jsonrpc": "2.0", "id": 5, "method": "session/abort",
                        "params": { "sessionId": "sess-unknown" } });
    peer.send_value(&abort).await;
    let ack = peer
        .recv_until("abort alias ack", |f| f.get("id") == Some(&json!(5)))
        .await;
    assert_eq!(ack["result"], json!({}));

    // Cancel for a session with no running turn and an unknown session id
    // is still a clean ack — no error, no state corruption.
    let cancel_unknown = json!({ "jsonrpc": "2.0", "id": 6, "method": "session/cancel",
                                 "params": { "sessionId": "does-not-exist" } });
    peer.send_value(&cancel_unknown).await;
    let ack = peer
        .recv_until("unknown-session cancel ack", |f| {
            f.get("id") == Some(&json!(6))
        })
        .await;
    assert_eq!(ack["result"], json!({}));

    let new = json!({ "jsonrpc": "2.0", "id": 2, "method": "session/new", "params": {} });
    peer.send_value(&new).await;
    let frame = peer.recv_frame().await.expect("session/new").1;
    assert_eq!(response_result(&frame)["sessionId"], "sess-1");
}

// ---------------------------------------------------------------------------
// Shutdown lifecycle and EOF
// ---------------------------------------------------------------------------

#[tokio::test]
async fn shutdown_request_is_answered_then_connection_ends() {
    let (_handle, mut peer) = spawn_server(AcpServer::new(EchoBackend::new()), 1024 * 1024);

    let shutdown = json!({ "jsonrpc": "2.0", "id": 1, "method": "shutdown", "params": {} });
    peer.send_value(&shutdown).await;
    let frame = peer.recv_frame().await.expect("shutdown response").1;
    assert_eq!(response_result(&frame)["ok"], true);
    assert!(
        peer.recv_frame().await.is_none(),
        "connection ends after shutdown"
    );
}

// ---------------------------------------------------------------------------
// Oversized backend results are refused, never truncated
// ---------------------------------------------------------------------------

#[tokio::test]
async fn oversized_backend_result_is_refused_not_truncated() {
    #[derive(Clone)]
    struct HugeBackend;
    impl AcpBackend for HugeBackend {
        fn agent_info(&self) -> Value {
            json!({})
        }
        fn create_session(&self, _p: &Value) -> Result<String, String> {
            Ok("sess-1".into())
        }
        fn list_sessions(&self) -> Vec<String> {
            vec![]
        }
        fn prompt(&self, _s: &str, _t: &str) -> Result<Value, String> {
            Ok(json!({ "huge": "x".repeat(MAX_RESPONSE_BYTES) }))
        }
        fn abort(&self, _s: &str) -> Result<(), String> {
            Ok(())
        }
    }
    let (_handle, mut peer) = spawn_server(AcpServer::new(HugeBackend), 1024 * 1024);

    let prompt = json!({ "jsonrpc": "2.0", "id": 3, "method": "session/prompt",
                         "params": { "sessionId": "sess-1", "prompt": [{ "type": "text", "text": "x" }] } });
    peer.send_value(&prompt).await;
    let frame = peer.recv_error().await;
    assert_eq!(frame["error"]["code"], -32603);
    let message = frame["error"]["data"].as_str().unwrap_or_default();
    assert!(
        message.contains("8 MiB"),
        "refusal must name the bound: {message}"
    );
}

// ---------------------------------------------------------------------------
// EmitTooLarge surfaces to the streaming backend as a typed error
// ---------------------------------------------------------------------------

#[tokio::test]
async fn oversized_emit_is_rejected_without_buffering() {
    #[derive(Clone)]
    struct HugeStream;
    impl AcpStreamBackend for HugeStream {
        fn agent_info(&self) -> Value {
            json!({})
        }
        fn create_session(&self, _p: &Value) -> Result<String, String> {
            Ok("sess-1".into())
        }
        fn list_sessions(&self) -> Vec<String> {
            vec![]
        }
        fn prompt<'a>(
            &'a self,
            _sid: &'a str,
            ctx: &'a PromptCtx,
            _text: &'a str,
        ) -> BoxFuture<'a, Result<Value, String>> {
            Box::pin(async move {
                let huge = text_chunk_update(&"x".repeat(MAX_RESPONSE_BYTES + 1));
                match ctx.emit(huge).await {
                    Err(EmitError::TooLarge) => Err("refused oversized emit".into()),
                    other => Err(format!("expected TooLarge, got {other:?}")),
                }
            })
        }
    }
    let (_handle, mut peer) = spawn_server(AcpServer::new_streaming(HugeStream), 1024 * 1024);

    let prompt = json!({ "jsonrpc": "2.0", "id": 3, "method": "session/prompt",
                         "params": { "sessionId": "sess-1", "prompt": [{ "type": "text", "text": "x" }] } });
    peer.send_value(&prompt).await;
    let frame = peer.recv_error().await;
    assert_eq!(frame["error"]["code"], -32603);
}
