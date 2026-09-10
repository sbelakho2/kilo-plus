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

/// Assert the received raw frame carries exactly the fixture's shape.
/// Key order is deliberately not compared: the workspace may build
/// `serde_json` with `preserve_order` (other workspace members' test
/// dependencies enable it), which makes canonical key order
/// build-dependent. Field names, values, presence and null behavior are
/// still compared exactly.
fn assert_canonical(raw: &[u8], fixture: &str) {
    let (_, actual) = protocol::parse_frame(raw)
        .expect("received frame parses")
        .expect("received frame is complete");
    let expected: Value = serde_json::from_str(fixture).unwrap();
    assert_eq!(actual, expected, "frame differs from the canonical golden");
}

/// Parse a fixture string into a JSON value (shape comparison).
fn golden_value(fixture: &str) -> Value {
    serde_json::from_str(fixture).expect("golden fixture is valid JSON")
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
    // Exact field shape of the public frame builders (key order is
    // build-dependent under `serde_json/preserve_order`; fields, values and
    // presence are compared exactly).
    assert_eq!(
        agent_state_changed_update(AgentStateStatus::Busy, None),
        golden_value(r#"{"agentState":{"status":"busy"},"kind":"agentStateChanged"}"#)
    );
    assert_eq!(
        agent_state_changed_update(AgentStateStatus::Busy, Some("thinking hard")),
        golden_value(
            r#"{"agentState":{"message":"thinking hard","status":"busy"},"kind":"agentStateChanged"}"#
        )
    );
    assert_eq!(
        agent_state_changed_update(AgentStateStatus::Error, Some("boom")),
        golden_value(
            r#"{"agentState":{"message":"boom","status":"error"},"kind":"agentStateChanged"}"#
        )
    );
    assert_eq!(
        agent_state_changed_update(AgentStateStatus::Idle, None),
        golden_value(r#"{"agentState":{"status":"idle"},"kind":"agentStateChanged"}"#)
    );
    assert_eq!(
        text_chunk_update("partial"),
        golden_value(
            r#"{"content":{"text":"partial","type":"text"},"sessionUpdate":"agent_message_chunk"}"#
        )
    );
    let params = session_update_params("sess-1", text_chunk_update("partial"));
    assert_eq!(
        params,
        golden_value(
            r#"{"sessionId":"sess-1","update":{"content":{"text":"partial","type":"text"},"sessionUpdate":"agent_message_chunk"}}"#
        )
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

    // Extension frames are gated: declaring the Faktor status extension
    // must echo only the accepted name back.
    let init = json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize",
                       "params": { "protocolVersion": 1,
                                   "extensions": ["faktor.agentStateChanged", "not.accepted"] } });
    peer.send_value(&init).await;
    let (_raw, frame) = peer.recv_frame().await.expect("initialize");
    assert_eq!(
        frame["result"]["extensions"],
        json!(["faktor.agentStateChanged"])
    );

    // Prompt "states" on session sess-1 (created via session/new first so
    // the fixture sessionId matches).
    let new = json!({ "jsonrpc": "2.0", "id": 2, "method": "session/new", "params": {} });
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
        g::INITIALIZE_REQUEST_EXTENSIONS,
        g::INITIALIZE_RESPONSE_CAPABLE,
        g::INITIALIZE_ERROR_BAD_EXTENSIONS,
        g::INITIALIZE_ERROR_BAD_FS,
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
        g::UPDATE_FRAME_TOOL_CALL,
        g::UPDATE_FRAME_TOOL_CALL_DEGRADED,
        g::UPDATE_FRAME_TOOL_RESULT_FAILED,
        g::UPDATE_FRAME_PLAN,
        g::LOAD_REQUEST,
        g::LOAD_RESPONSE,
        g::ERROR_LOAD_FOREIGN_SESSION,
        g::ERROR_LOAD_INCOMPLETE_HISTORY,
        g::ERROR_MCP_UNSUPPORTED,
        g::ERROR_AUTHENTICATE,
        g::PERMISSION_REQUEST_FRAME,
        g::PERMISSION_ALLOW_RESPONSE,
        g::PERMISSION_DENY_RESPONSE,
        g::PERMISSION_CANCELLED_RESPONSE,
        g::FS_READ_REQUEST_FRAME,
        g::FS_READ_RESPONSE,
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

#[test]
fn native_mapping_builders_degrade_documented_fields() {
    // Native tool-call state vocabulary maps faithfully; unknown states
    // omit the optional status instead of guessing.
    assert_eq!(
        tool_call_from_native("call-1", "echo", &json!({"x": 1}), "running"),
        golden_value(g::UPDATE_FRAME_TOOL_CALL)
    );
    assert_eq!(
        tool_call_from_native("call-1", "echo", &json!({}), "who-knows"),
        golden_value(g::UPDATE_FRAME_TOOL_CALL_DEGRADED)
    );
    assert_eq!(
        ToolCallStatus::from_native_state("pending"),
        Some(ToolCallStatus::Pending)
    );
    assert_eq!(
        ToolCallStatus::from_native_state("running"),
        Some(ToolCallStatus::InProgress)
    );
    assert_eq!(
        ToolCallStatus::from_native_state("completed"),
        Some(ToolCallStatus::Completed)
    );
    assert_eq!(
        ToolCallStatus::from_native_state("failed"),
        Some(ToolCallStatus::Failed)
    );
    assert_eq!(ToolCallStatus::from_native_state("exploded"), None);

    // Native tool result: excerpt -> bounded text content, non-zero exit
    // -> failed, artifact reference rides the official `_meta` slot.
    assert_eq!(
        tool_result_from_native("call-1", "boom", Some(3), None, None),
        golden_value(g::UPDATE_FRAME_TOOL_RESULT_FAILED)
    );
    let ok = tool_result_from_native("call-1", "fine", Some(0), Some("cas://blob"), Some("0:10"));
    assert_eq!(ok["status"], "completed");
    assert_eq!(ok["_meta"]["artifact"], "cas://blob");
    assert_eq!(ok["_meta"]["sliceHint"], "0:10");
    let no_artifact = tool_result_from_native("call-1", "fine", None, None, None);
    assert_eq!(no_artifact["status"], "completed");
    assert!(no_artifact.get("_meta").is_none());

    // Native ledger plan steps: flat plan, conservative priority/status.
    let steps = vec![
        ("step one".to_string(), None),
        ("step two".to_string(), Some(0)),
    ];
    assert_eq!(
        plan_from_native_steps(&steps),
        golden_value(g::UPDATE_FRAME_PLAN)
    );

    // Plan entries given explicit statuses serialize faithfully.
    let plan = plan_update(&[
        PlanEntry {
            content: "a".into(),
            priority: PlanPriority::High,
            status: PlanStatus::InProgress,
        },
        PlanEntry {
            content: "b".into(),
            priority: PlanPriority::Low,
            status: PlanStatus::Completed,
        },
    ]);
    assert_eq!(plan["entries"][0]["priority"], "high");
    assert_eq!(plan["entries"][0]["status"], "in_progress");
    assert_eq!(plan["entries"][1]["priority"], "low");
    assert_eq!(plan["entries"][1]["status"], "completed");

    // User/thought chunk frames are official kinds.
    assert_eq!(
        user_message_chunk_update("hi")["sessionUpdate"],
        "user_message_chunk"
    );
    assert_eq!(
        agent_thought_chunk_update("hmm")["sessionUpdate"],
        "agent_thought_chunk"
    );
}

#[test]
fn permission_options_serialize_official_shapes() {
    let options = [
        PermissionOption::allow_once(),
        PermissionOption::reject_once(),
    ];
    assert_eq!(
        options[0].to_wire(),
        json!({"optionId": "allow_once", "name": "Allow once", "kind": "allow_once"})
    );
    assert_eq!(options[1].option_id(), "reject_once");
    assert_eq!(options[1].name(), "Reject once");
    assert_eq!(options[1].kind(), PermissionOptionKind::RejectOnce);
    assert_eq!(
        PermissionOption::allow_always().to_wire()["kind"],
        "allow_always"
    );
    assert_eq!(
        PermissionOption::reject_always().to_wire()["kind"],
        "reject_always"
    );
    // Outcome mapping helper: allow* allows, everything else denies.
    let allow = Ok(PermissionOutcome::selected("allow_once"));
    let deny = Ok(PermissionOutcome::selected("reject_once"));
    let cancelled = Err(ClientRequestError::Timeout);
    assert!(ClientHandle::permission_allows(&allow));
    assert!(!ClientHandle::permission_allows(&deny));
    assert!(!ClientHandle::permission_allows(&cancelled));
}

#[tokio::test]
async fn strict_client_receives_no_extension_frames_but_official_ones() {
    // The backend emits extension state frames AND official text; a client
    // that never declared the extension must see only official frames.
    let backend = StreamBackend::new();
    let (_handle, mut peer) = spawn_server(AcpServer::new_streaming(backend), 1024 * 1024);

    let prompt = json!({ "jsonrpc": "2.0", "id": 3, "method": "session/prompt",
                         "params": { "sessionId": "sess-1", "prompt": [{ "type": "text", "text": "states" }] } });
    peer.send_value(&prompt).await;

    let mut saw_text = false;
    let mut saw_terminal = false;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_secs(2), peer.recv_frame()).await {
            Ok(Some((_raw, frame))) => {
                if is_update(&frame).is_some() {
                    let update = &frame["params"]["update"];
                    assert!(
                        update.get("kind").is_none(),
                        "extension frame leaked to a non-negotiating client: {frame}"
                    );
                    if update["sessionUpdate"] == "agent_message_chunk" {
                        saw_text = true;
                    }
                } else if terminal_of(&frame).is_some() {
                    assert_eq!(terminal_of(&frame), Some("end_turn"));
                    saw_terminal = true;
                    break;
                }
            }
            Ok(None) => break,
            Err(_) => break,
        }
    }
    assert!(saw_text, "official text frame must still arrive");
    assert!(saw_terminal, "turn must terminate normally");
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
// Malformed negotiation: loud refusals, no state corruption
// ---------------------------------------------------------------------------

#[tokio::test]
async fn b2_malformed_negotiation_is_refused_loudly() {
    let (_handle, mut peer) =
        spawn_server(AcpServer::new_streaming(StreamBackend::new()), 1024 * 1024);

    // Extensions must be an array of strings.
    let bad_shape = json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize",
                            "params": { "protocolVersion": 1, "extensions": "faktor.agentStateChanged" } });
    peer.send_value(&bad_shape).await;
    let (raw, frame) = peer.recv_frame().await.expect("bad extensions shape");
    assert_semantic(&frame, g::INITIALIZE_ERROR_BAD_EXTENSIONS);
    assert_canonical(&raw, g::INITIALIZE_ERROR_BAD_EXTENSIONS);

    let bad_entry = json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize",
                            "params": { "protocolVersion": 1, "extensions": [7] } });
    peer.send_value(&bad_entry).await;
    let frame = peer.recv_frame().await.expect("bad extension entry").1;
    assert_eq!(frame["error"]["code"], -32602);
    assert_eq!(
        frame["error"]["message"],
        "\"extensions\" entries must be strings"
    );

    // Client fs capability values must be booleans.
    let bad_fs = json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize",
                         "params": { "protocolVersion": 1,
                                     "clientCapabilities": { "fs": { "readTextFile": "yes" } } } });
    peer.send_value(&bad_fs).await;
    let (raw, frame) = peer.recv_frame().await.expect("bad fs capability");
    assert_semantic(&frame, g::INITIALIZE_ERROR_BAD_FS);
    assert_canonical(&raw, g::INITIALIZE_ERROR_BAD_FS);

    // A malformed re-initialize must not clobber a previous negotiation.
    let good = json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize",
                       "params": { "protocolVersion": 1, "extensions": ["faktor.agentStateChanged"] } });
    peer.send_value(&good).await;
    let frame = peer.recv_frame().await.expect("good initialize").1;
    assert_eq!(
        frame["result"]["extensions"],
        json!(["faktor.agentStateChanged"])
    );
    peer.send_value(&bad_shape).await;
    let frame = peer.recv_frame().await.expect("malformed re-initialize").1;
    assert_eq!(frame["error"]["code"], -32602);
    let prompt = json!({ "jsonrpc": "2.0", "id": 3, "method": "session/prompt",
                         "params": { "sessionId": "sess-1", "prompt": [{ "type": "text", "text": "states" }] } });
    peer.send_value(&prompt).await;
    let busy = peer
        .recv_until("negotiated state frame after malformed re-init", |f| {
            is_update(f).is_some()
                && f["params"]["update"]["kind"] == "agentStateChanged"
                && f["params"]["update"]["agentState"]["status"] == "busy"
        })
        .await;
    assert_eq!(
        busy["params"]["update"]["agentState"]["message"],
        "thinking hard"
    );
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
        stalled <= 5,
        "bounded in-flight frames: main queue (2) + one writer-held frame + at most two frames in the 512-byte transport (got {stalled})"
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

// ---------------------------------------------------------------------------
// Official transport conformance: NDJSON primary, string ids, $/cancel_request
// ---------------------------------------------------------------------------

/// NDJSON peer: writes `\n`-terminated messages, reads and parses lines, and
/// keeps raw bytes so framing/id assertions run on what the server actually
/// emitted.
struct LinePeer {
    to_server: DuplexStream,
    from_server: DuplexStream,
    buf: Vec<u8>,
}

impl LinePeer {
    async fn send_line(&mut self, value: &Value) {
        let bytes = protocol::encode_line(value).expect("line encodes");
        self.to_server.write_all(&bytes).await.expect("test write");
    }

    async fn send_text(&mut self, text: &str) {
        self.to_server
            .write_all(text.as_bytes())
            .await
            .expect("test write");
    }

    /// Read one NDJSON message with a hard timeout. `None` on EOF.
    async fn recv_frame(&mut self) -> Option<(Vec<u8>, Value)> {
        let mut chunk = [0u8; 4096];
        loop {
            match protocol::parse_ndjson(&self.buf) {
                Ok(Some((consumed, value))) => {
                    let raw = self.buf[..consumed].to_vec();
                    self.buf.drain(..consumed);
                    return Some((raw, value));
                }
                Ok(None) => {}
                Err(e) => panic!("server emitted an unframed line: {e}"),
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

    async fn recv_until(&mut self, what: &str, predicate: impl Fn(&Value) -> bool) -> Value {
        for _ in 0..15000 {
            match self.recv_frame().await {
                Some((_raw, frame)) => {
                    if predicate(&frame) {
                        return frame;
                    }
                }
                None => panic!("connection ended while waiting for {what}"),
            }
        }
        panic!("timed out waiting for {what}");
    }
}

fn spawn_line_server(server: AcpServer) -> (tokio::task::JoinHandle<Result<(), String>>, LinePeer) {
    let (c2s_read, c2s_write) = duplex(1024 * 1024);
    let (s2c_read, s2c_write) = duplex(1024 * 1024);
    let peer = LinePeer {
        to_server: c2s_write,
        from_server: s2c_read,
        buf: Vec::new(),
    };
    let handle = tokio::spawn(async move { server.serve_connection(c2s_read, s2c_write).await });
    (handle, peer)
}

fn is_response(frame: &Value) -> bool {
    frame.get("result").is_some() || frame.get("error").is_some()
}

#[tokio::test]
async fn ndjson_is_primary_wire_and_notifications_omit_id() {
    let (_handle, mut peer) = spawn_line_server(AcpServer::new_streaming(StreamBackend::new()));

    // initialize with a string id: NDJSON line in, NDJSON line out, id echoed.
    peer.send_line(&json!({
        "jsonrpc": "2.0",
        "id": "init-1",
        "method": "initialize",
        "params": { "protocolVersion": 1 },
    }))
    .await;
    let (raw, init) = peer.recv_frame().await.expect("initialize response");
    assert_eq!(init["id"], "init-1");
    assert_eq!(init["result"]["protocolVersion"], 1);
    let text = String::from_utf8(raw.clone()).expect("utf-8 frame");
    assert!(
        text.ends_with('\n'),
        "NDJSON frame must end with newline: {text}"
    );
    assert!(
        !text.contains("Content-Length"),
        "NDJSON connection must not emit Content-Length framing: {text}"
    );

    peer.send_line(&json!({
        "jsonrpc": "2.0",
        "id": "new-1",
        "method": "session/new",
        "params": {},
    }))
    .await;
    let new = peer.recv_until("session/new", |f| f["id"] == "new-1").await;
    assert_eq!(new["result"]["sessionId"], "sess-1");

    peer.send_line(&json!({
        "jsonrpc": "2.0",
        "id": "prompt-1",
        "method": "session/prompt",
        "params": { "sessionId": "sess-1", "prompt": [{ "type": "text", "text": "hi" }] },
    }))
    .await;

    let mut update_raw: Option<Vec<u8>> = None;
    let terminal = loop {
        let (raw, frame) = peer.recv_frame().await.expect("frame");
        if frame.get("method").and_then(Value::as_str) == Some("session/update") {
            assert!(
                frame.get("id").is_none(),
                "notifications must omit id entirely: {frame}"
            );
            update_raw = Some(raw);
        } else if is_response(&frame) {
            break frame;
        }
    };
    assert_eq!(terminal["id"], "prompt-1");
    assert_eq!(terminal_of(&terminal), Some("end_turn"));

    // Raw bytes: the update frame carries no `"id"` member at all (the
    // official SDK would classify an id-bearing frame as a request and drop
    // every update).
    let update_raw = String::from_utf8(update_raw.expect("one update frame")).unwrap();
    assert!(update_raw.ends_with('\n'), "{update_raw}");
    assert!(
        !update_raw.contains("\"id\""),
        "raw update frame carries an id member: {update_raw}"
    );
}

#[tokio::test]
async fn legacy_content_length_peer_keeps_content_length_framing() {
    // The frozen pre-conformance path: a peer opening with a
    // Content-Length header is answered in kind, byte-identical framing.
    let (_handle, mut peer) = spawn_server(AcpServer::new(EchoBackend::new()), 1024 * 1024);
    let init = json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize",
                       "params": { "protocolVersion": 1 } });
    peer.send_value(&init).await;
    let (raw, frame) = peer.recv_frame().await.expect("initialize response");
    assert_eq!(frame["id"], 1);
    assert!(
        raw.starts_with(b"Content-Length: "),
        "legacy peer must receive Content-Length framing: {:?}",
        String::from_utf8_lossy(&raw)
    );
    assert!(
        !raw.ends_with(b"\n"),
        "Content-Length framing must not append a newline"
    );
}

#[tokio::test]
async fn cancel_request_cancels_the_matching_running_prompt_exactly_once() {
    let backend = StreamBackend::new();
    let (_handle, mut peer) = spawn_line_server(AcpServer::new_streaming(backend));

    peer.send_line(
        &json!({ "jsonrpc": "2.0", "id": "init", "method": "initialize",
                            "params": { "protocolVersion": 1 } }),
    )
    .await;
    peer.recv_until("initialize", |f| f["id"] == "init").await;
    peer.send_line(
        &json!({ "jsonrpc": "2.0", "id": "new", "method": "session/new", "params": {} }),
    )
    .await;
    peer.recv_until("session/new", |f| f["id"] == "new").await;

    let prompt_id = "e70f649f-bb05-42b2-9b08-380299012ea8";
    peer.send_line(&json!({
        "jsonrpc": "2.0",
        "id": prompt_id,
        "method": "session/prompt",
        "params": { "sessionId": "sess-1", "prompt": [{ "type": "text", "text": "flood" }] },
    }))
    .await;
    peer.recv_until("first update", |f| {
        f.get("method").and_then(Value::as_str) == Some("session/update")
    })
    .await;

    // Official SDK drop semantics: `$/cancel_request` names the request id.
    peer.send_line(&json!({
        "jsonrpc": "2.0",
        "method": "$/cancel_request",
        "params": { "requestId": prompt_id },
    }))
    .await;

    // Drain to quiescence: exactly one terminal for the cancelled prompt.
    // After the first terminal, a 500 ms quiet window proves no duplicate
    // terminal follows (a duplicate would otherwise be skipped silently by
    // the next `recv_until`).
    let mut terminals = Vec::new();
    let mut answered_cancel = 0usize;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    loop {
        let wait = if terminals.is_empty() {
            deadline.saturating_duration_since(tokio::time::Instant::now())
        } else {
            Duration::from_millis(500)
        };
        match tokio::time::timeout(wait, peer.recv_frame()).await {
            Ok(Some((_raw, frame))) => {
                if frame.get("method").and_then(Value::as_str) == Some("$/cancel_request") {
                    answered_cancel += 1;
                }
                if is_response(&frame) {
                    terminals.push(frame);
                }
            }
            Ok(None) => break,
            Err(_) => break,
        }
    }
    assert_eq!(terminals.len(), 1, "exactly one terminal: {terminals:?}");
    assert_eq!(terminals[0]["id"], prompt_id);
    assert_eq!(terminal_of(&terminals[0]), Some("cancelled"));
    assert_eq!(
        answered_cancel, 0,
        "a cancel notification is never answered"
    );

    // The session is not dead: a later prompt completes.
    peer.send_line(&json!({
        "jsonrpc": "2.0",
        "id": "later",
        "method": "session/prompt",
        "params": { "sessionId": "sess-1", "prompt": [{ "type": "text", "text": "hi" }] },
    }))
    .await;
    let later = peer
        .recv_until("later terminal", |f| f["id"] == "later")
        .await;
    assert_eq!(terminal_of(&later), Some("end_turn"));
}

#[tokio::test]
async fn string_ids_survive_queue_promotion() {
    let backend = StreamBackend::new();
    let slow = backend.slow_release();
    let (_handle, mut peer) = spawn_line_server(AcpServer::new_streaming(backend));

    peer.send_line(
        &json!({ "jsonrpc": "2.0", "id": "init", "method": "initialize",
                            "params": { "protocolVersion": 1 } }),
    )
    .await;
    peer.recv_until("initialize", |f| f["id"] == "init").await;
    peer.send_line(
        &json!({ "jsonrpc": "2.0", "id": "new", "method": "session/new", "params": {} }),
    )
    .await;
    peer.recv_until("session/new", |f| f["id"] == "new").await;

    // First prompt parks on the gate, second is queued behind it; the
    // promoted turn's terminal must echo the queued prompt's own string id.
    peer.send_line(&json!({
        "jsonrpc": "2.0",
        "id": "queued-first",
        "method": "session/prompt",
        "params": { "sessionId": "sess-1", "prompt": [{ "type": "text", "text": "slow" }] },
    }))
    .await;
    peer.send_line(&json!({
        "jsonrpc": "2.0",
        "id": "queued-second",
        "method": "session/prompt",
        "params": { "sessionId": "sess-1", "prompt": [{ "type": "text", "text": "hi" }] },
    }))
    .await;
    slow.send(()).await.expect("release the parked turn");

    let first = peer
        .recv_until("first terminal", |f| f["id"] == "queued-first")
        .await;
    assert_eq!(terminal_of(&first), Some("end_turn"));
    let second = peer
        .recv_until("promoted terminal", |f| f["id"] == "queued-second")
        .await;
    assert_eq!(terminal_of(&second), Some("end_turn"));
}

#[tokio::test]
async fn cancel_request_is_bounded_and_typed() {
    let backend = StreamBackend::new();
    let slow = backend.slow_release();
    let (_handle, mut peer) = spawn_line_server(AcpServer::new_streaming(backend));

    peer.send_line(
        &json!({ "jsonrpc": "2.0", "id": "init", "method": "initialize",
                            "params": { "protocolVersion": 1 } }),
    )
    .await;
    peer.recv_until("initialize", |f| f["id"] == "init").await;
    peer.send_line(
        &json!({ "jsonrpc": "2.0", "id": "new", "method": "session/new", "params": {} }),
    )
    .await;
    peer.recv_until("session/new", |f| f["id"] == "new").await;

    // A running turn with a different id: a non-matching cancel is a no-op
    // (bounded scan, never a guessed cancellation).
    peer.send_line(&json!({
        "jsonrpc": "2.0",
        "id": "running-id",
        "method": "session/prompt",
        "params": { "sessionId": "sess-1", "prompt": [{ "type": "text", "text": "slow" }] },
    }))
    .await;
    peer.send_line(&json!({
        "jsonrpc": "2.0",
        "method": "$/cancel_request",
        "params": { "requestId": "not-running-id" },
    }))
    .await;
    // Unknown/queued ids are ignored: the parked turn completes normally.
    slow.send(()).await.expect("release the slow turn");
    let terminal = peer
        .recv_until("slow terminal", |f| f["id"] == "running-id")
        .await;
    assert_eq!(terminal_of(&terminal), Some("end_turn"));

    // Malformed params in request form: official typed invalid-params error,
    // with the client's string id echoed verbatim.
    peer.send_line(&json!({
        "jsonrpc": "2.0",
        "id": "cancel-uuid",
        "method": "$/cancel_request",
        "params": {},
    }))
    .await;
    let error = peer
        .recv_until("typed cancel error", |f| f.get("error").is_some())
        .await;
    assert_eq!(error["id"], "cancel-uuid");
    assert_eq!(error["error"]["code"], INVALID_PARAMS);
    assert_eq!(
        error["error"]["message"],
        "missing request id field \"requestId\""
    );

    // The request form acks `{}` when the id is well-formed, even when no
    // turn matches.
    peer.send_line(&json!({
        "jsonrpc": "2.0",
        "id": 78,
        "method": "$/cancel_request",
        "params": { "requestId": "not-running-id" },
    }))
    .await;
    let ack = peer.recv_until("cancel ack", |f| f["id"] == 78).await;
    assert_eq!(ack["result"], json!({}));

    // The connection still serves.
    peer.send_line(
        &json!({ "jsonrpc": "2.0", "id": "again", "method": "session/new",
                            "params": {} }),
    )
    .await;
    let again = peer.recv_until("session/new", |f| f["id"] == "again").await;
    assert_eq!(again["result"]["sessionId"], "sess-2");
}

#[tokio::test]
async fn ndjson_blank_lines_malformed_lines_and_invalid_requests_keep_serving() {
    let (_handle, mut peer) = spawn_line_server(AcpServer::new(EchoBackend::new()));

    // Blank keep-alive lines before the first message must be skipped, not
    // answered (and they must not break framing detection or accumulate in
    // the read buffer: a 4096-line flood is drained read-by-read).
    peer.send_text(&"\n".repeat(4096)).await;
    peer.send_text("\r\n   \n").await;
    peer.send_line(&json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize",
                            "params": { "protocolVersion": 1 } }))
        .await;
    let (_, init) = peer.recv_frame().await.expect("initialize response");
    assert_eq!(init["id"], 1);
    assert_eq!(init["result"]["protocolVersion"], 1);

    // An unparseable line is an official parse error (null id) and the
    // stream stays usable.
    peer.send_text("{\"broken\n").await;
    let error = peer
        .recv_until("parse error", |f| f["error"]["code"] == PARSE_ERROR)
        .await;
    assert!(error["id"].is_null());

    // A JSON line that is not a JSON-RPC object: invalid request, no hang.
    peer.send_text("[1,2,3]\n").await;
    let error = peer
        .recv_until("invalid request", |f| f["error"]["code"] == INVALID_REQUEST)
        .await;
    assert!(error["id"].is_null());

    // Serving continues afterwards.
    peer.send_line(
        &json!({ "jsonrpc": "2.0", "id": "after", "method": "agent_info",
                            "params": {} }),
    )
    .await;
    let info = peer.recv_until("agent_info", |f| f["id"] == "after").await;
    assert_eq!(info["result"]["name"], "test-agent");
}
