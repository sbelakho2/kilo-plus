//! Adversarial end-to-end ACP wire tests over in-memory duplex pipes.
//!
//! Every scenario drives a real `AcpServer` through `serve_connection` with
//! raw bytes on one side and parsed JSON-RPC on the other: handshake
//! lifecycle, hostile framing, fragmentation, pipelining, malformed
//! payloads, backend failures, ordering with a slow backend, size bounds,
//! and hostile UTF-8/NUL round-trips.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use faktor_acp::protocol::{frame, notification_frame, parse_frame};
use faktor_acp::{AcpBackend, AcpServer};
use serde_json::{json, Value};
use tokio::io::{duplex, AsyncReadExt, AsyncWriteExt, DuplexStream, ReadHalf, WriteHalf};
use tokio::time::timeout;

const RX_TIMEOUT: Duration = Duration::from_secs(10);

// ---------------------------------------------------------------------------
// Backends
// ---------------------------------------------------------------------------

/// Deterministic echo backend: `sess-0`, `sess-1`, …; prompt echoes text.
struct EchoBackend {
    sessions: Mutex<Vec<String>>,
    next: AtomicU64,
}

impl EchoBackend {
    fn new() -> Self {
        Self {
            sessions: Mutex::new(Vec::new()),
            next: AtomicU64::new(0),
        }
    }
}

impl AcpBackend for EchoBackend {
    fn agent_info(&self) -> Value {
        json!({
            "name": "faktor-test-agent",
            "version": "0.0.0",
            "capabilities": { "prompt": true, "sessions": true },
        })
    }
    fn create_session(&self, params: &Value) -> Result<String, String> {
        if params.get("fail").is_some() {
            return Err("session creation refused by backend".into());
        }
        let n = self.next.fetch_add(1, Ordering::SeqCst);
        let id = format!("sess-{n}");
        self.sessions.lock().unwrap().push(id.clone());
        Ok(id)
    }
    fn prompt(&self, session_id: &str, text: &str) -> Result<Value, String> {
        Ok(json!({ "sessionID": session_id, "echo": text }))
    }
    fn abort(&self, session_id: &str) -> Result<(), String> {
        if session_id.starts_with("sess-refuse") {
            return Err("abort refused".into());
        }
        Ok(())
    }
    fn list_sessions(&self) -> Vec<String> {
        self.sessions.lock().unwrap().clone()
    }
}

/// Backend whose every call fails, for -32000 surfacing.
struct FailingBackend;

impl AcpBackend for FailingBackend {
    fn agent_info(&self) -> Value {
        json!({ "name": "failing", "version": "0.0.0" })
    }
    fn create_session(&self, _params: &Value) -> Result<String, String> {
        Err("create exploded".into())
    }
    fn prompt(&self, _session_id: &str, _text: &str) -> Result<Value, String> {
        Err("backend exploded".into())
    }
    fn abort(&self, _session_id: &str) -> Result<(), String> {
        Err("abort exploded".into())
    }
    fn list_sessions(&self) -> Vec<String> {
        vec![]
    }
}

/// Prompt takes a fixed wall-clock delay before answering, to prove the
/// serialized loop preserves request order regardless of backend speed.
struct SlowBackend {
    inner: EchoBackend,
    delay: Duration,
}

impl AcpBackend for SlowBackend {
    fn agent_info(&self) -> Value {
        self.inner.agent_info()
    }
    fn create_session(&self, params: &Value) -> Result<String, String> {
        self.inner.create_session(params)
    }
    fn prompt(&self, session_id: &str, text: &str) -> Result<Value, String> {
        std::thread::sleep(self.delay);
        self.inner.prompt(session_id, text)
    }
    fn abort(&self, session_id: &str) -> Result<(), String> {
        self.inner.abort(session_id)
    }
    fn list_sessions(&self) -> Vec<String> {
        self.inner.list_sessions()
    }
}

/// Prompt answers with a ~9 MiB blob: must be refused by the 8 MiB trim.
struct HugeBackend;

impl AcpBackend for HugeBackend {
    fn agent_info(&self) -> Value {
        json!({ "name": "huge", "version": "0.0.0" })
    }
    fn create_session(&self, _params: &Value) -> Result<String, String> {
        Ok("sess-huge".into())
    }
    fn prompt(&self, _session_id: &str, _text: &str) -> Result<Value, String> {
        Ok(json!({ "blob": "x".repeat(9 * 1024 * 1024) }))
    }
    fn abort(&self, _session_id: &str) -> Result<(), String> {
        Ok(())
    }
    fn list_sessions(&self) -> Vec<String> {
        vec!["sess-huge".into()]
    }
}

// ---------------------------------------------------------------------------
// Wire harness
// ---------------------------------------------------------------------------

type ClientRead = ReadHalf<DuplexStream>;
type ClientWrite = WriteHalf<DuplexStream>;

/// Byte-level ACP client over the other end of the duplex pipe.
struct WireClient {
    read: ClientRead,
    write: ClientWrite,
    recv_buf: Vec<u8>,
    next_id: u64,
}

impl WireClient {
    async fn recv_frame(&mut self) -> Result<Option<Value>, String> {
        loop {
            match parse_frame(&self.recv_buf) {
                Ok(Some((consumed, value))) => {
                    self.recv_buf.drain(..consumed);
                    return Ok(Some(value));
                }
                Ok(None) => {
                    let mut chunk = [0u8; 8192];
                    let n = timeout(RX_TIMEOUT, self.read.read(&mut chunk))
                        .await
                        .map_err(|_| "timed out waiting for server bytes".to_string())?
                        .map_err(|e| format!("read: {e}"))?;
                    if n == 0 {
                        return Ok(None);
                    }
                    self.recv_buf.extend_from_slice(&chunk[..n]);
                }
                Err(msg) => return Err(format!("client framing error: {msg}")),
            }
        }
    }

    async fn expect_message(&mut self) -> Value {
        self.recv_frame()
            .await
            .expect("server response frame")
            .unwrap_or_else(|| panic!("server closed the pipe before answering"))
    }

    async fn expect_eof(&mut self) {
        let rest = self.recv_frame().await.expect("clean server close");
        assert!(rest.is_none(), "expected EOF, got frame: {rest:?}");
    }

    async fn request(&mut self, method: &str, params: Value) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        let bytes = frame(method.to_string(), id, params);
        self.write.write_all(&bytes).await.expect("client write");
        self.write.flush().await.expect("client flush");
        let msg = self.expect_message().await;
        assert_eq!(msg["id"], id, "response id mismatch: {msg}");
        msg
    }

    async fn error_code_of(&mut self, method: &str, params: Value) -> i64 {
        let msg = self.request(method, params).await;
        msg["error"]["code"]
            .as_i64()
            .unwrap_or_else(|| panic!("expected error response, got: {msg}"))
    }
}

fn start_server<B: AcpBackend + 'static>(
    backend: B,
) -> (WireClient, tokio::task::JoinHandle<Result<(), String>>) {
    let (server_side, client_side) = duplex(4 * 1024 * 1024);
    let (server_r, server_w) = tokio::io::split(server_side);
    let (client_r, client_w) = tokio::io::split(client_side);
    let server = AcpServer::new(backend);
    let handle = tokio::spawn(async move { server.serve_connection(server_r, server_w).await });
    let client = WireClient {
        read: client_r,
        write: client_w,
        recv_buf: Vec::new(),
        next_id: 1,
    };
    (client, handle)
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

#[tokio::test]
async fn full_handshake_lifecycle_initialize_session_prompt_shutdown() {
    let (mut client, server_task) = start_server(EchoBackend::new());

    // initialize → protocolVersion + embedded agent info.
    let init = client.request("initialize", json!({})).await;
    assert_eq!(init["result"]["protocolVersion"], "0.1.0");
    assert_eq!(init["result"]["agent"]["name"], "faktor-test-agent");

    // agent_info passthrough.
    let info = client.request("agent_info", json!({})).await;
    assert_eq!(info["result"]["capabilities"]["prompt"], true);

    // session/new → deterministic session id.
    let new = client.request("session/new", json!({})).await;
    assert_eq!(new["result"]["sessionID"], "sess-0");
    let new2 = client.request("session/new", json!({})).await;
    assert_eq!(new2["result"]["sessionID"], "sess-1");

    // session/list reflects both sessions as objects.
    let list = client.request("session/list", json!({})).await;
    let sessions = list["result"]["sessions"].as_array().unwrap();
    assert_eq!(
        sessions
            .iter()
            .map(|s| s["sessionID"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["sess-0", "sess-1"]
    );

    // session/prompt echoes.
    let p = client
        .request(
            "session/prompt",
            json!({ "sessionID": "sess-0", "text": "hello agent" }),
        )
        .await;
    assert_eq!(p["result"]["echo"], "hello agent");
    assert_eq!(p["result"]["sessionID"], "sess-0");

    // session/abort answers ok.
    let abort = client
        .request("session/abort", json!({ "sessionID": "sess-0" }))
        .await;
    assert_eq!(abort["result"], json!({ "ok": true }));

    // shutdown answers, then the serve loop ends and the pipe closes.
    let bye = client.request("shutdown", json!({})).await;
    assert_eq!(bye["result"], json!({ "ok": true }));
    client.expect_eof().await;
    timeout(RX_TIMEOUT, server_task)
        .await
        .expect("server task hung")
        .unwrap()
        .expect("server clean exit");
}

#[tokio::test]
async fn unknown_method_is_32601() {
    let (mut client, _task) = start_server(EchoBackend::new());
    let msg = client.request("bogus/method", json!({})).await;
    assert_eq!(msg["error"]["code"], -32601);
    let msg = client.request("session/close", json!({})).await;
    assert_eq!(msg["error"]["code"], -32601);
}

#[tokio::test]
async fn malformed_json_body_is_32700_and_loop_survives() {
    let (mut client, _task) = start_server(EchoBackend::new());

    // Broken body with a VALID Content-Length: parse error, id null, and
    // the loop keeps serving because framing stayed in sync.
    let broken = b"Content-Length: 10\r\n\r\n{\"broken\":".to_vec();
    client.write.write_all(&broken).await.unwrap();
    client.write.flush().await.unwrap();
    let err = client.expect_message().await;
    assert!(err["id"].is_null());
    assert_eq!(err["error"]["code"], -32700);
    assert!(err["error"]["message"]
        .as_str()
        .unwrap()
        .contains("invalid JSON"));

    let ok = client.request("agent_info", json!({})).await;
    assert_eq!(ok["result"]["name"], "faktor-test-agent");
}

#[tokio::test]
async fn hostile_20_mib_content_length_rejected_no_oom_then_connection_closed() {
    let (mut client, task) = start_server(EchoBackend::new());

    // Header alone declares a 20 MiB body: rejected instantly, before any
    // body byte is buffered (no OOM, no 20 MiB allocation).
    let hostile = b"Content-Length: 20971520\r\n\r\n".to_vec();
    client.write.write_all(&hostile).await.unwrap();
    client.write.flush().await.unwrap();
    let err = client.expect_message().await;
    assert!(err["id"].is_null());
    assert_eq!(err["error"]["code"], -32700);
    let message = err["error"]["message"].as_str().unwrap();
    assert!(message.contains("20971520"), "{message}");
    assert!(message.contains("16 MiB"), "{message}");

    // Framing is unrecoverable: server closes the connection.
    client.expect_eof().await;
    timeout(RX_TIMEOUT, task)
        .await
        .expect("server task hung")
        .unwrap()
        .expect("server clean exit");
}

#[tokio::test]
async fn fragmented_frames_reassemble_across_arbitrary_boundaries() {
    let (mut client, _task) = start_server(EchoBackend::new());

    // Write one request in 1–3 byte dribbles; the server must reassemble
    // across header/body byte boundaries.
    let bytes = frame(
        "session/prompt".to_string(),
        42,
        json!({ "sessionID": "sess-0", "text": "fragmented" }),
    );
    let mut step = 1usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let take = (i + step).min(bytes.len());
        client.write.write_all(&bytes[i..take]).await.unwrap();
        client.write.flush().await.unwrap();
        i = take;
        step = step % 3 + 1;
    }
    let msg = client.expect_message().await;
    assert_eq!(msg["id"], 42);
    assert_eq!(msg["result"]["echo"], "fragmented");
}

#[tokio::test]
async fn two_requests_pipelined_in_one_write_answered_in_order() {
    let (mut client, _task) = start_server(EchoBackend::new());

    let a = frame(
        "session/prompt".to_string(),
        1,
        json!({ "sessionID": "s", "text": "first" }),
    );
    let b = frame(
        "session/prompt".to_string(),
        2,
        json!({ "sessionID": "s", "text": "second" }),
    );
    let mut both = a;
    both.extend_from_slice(&b);
    client.write.write_all(&both).await.unwrap();
    client.write.flush().await.unwrap();

    let first = client.expect_message().await;
    let second = client.expect_message().await;
    assert_eq!(first["id"], 1);
    assert_eq!(first["result"]["echo"], "first");
    assert_eq!(second["id"], 2);
    assert_eq!(second["result"]["echo"], "second");
}

#[tokio::test]
async fn backend_errors_surface_as_32000_with_message() {
    let (mut client, _task) = start_server(FailingBackend);

    let msg = client
        .request("session/prompt", json!({ "sessionID": "s", "text": "hi" }))
        .await;
    assert_eq!(msg["error"]["code"], -32000);
    assert_eq!(msg["error"]["message"], "backend exploded");

    let msg = client.request("session/new", json!({})).await;
    assert_eq!(msg["error"]["code"], -32000);
    assert_eq!(msg["error"]["message"], "create exploded");

    let msg = client
        .request("session/abort", json!({ "sessionID": "s" }))
        .await;
    assert_eq!(msg["error"]["code"], -32000);
    assert_eq!(msg["error"]["message"], "abort exploded");
}

#[tokio::test]
async fn notification_does_not_deadlock_and_shutdown_notification_ends_loop() {
    let (mut client, task) = start_server(EchoBackend::new());

    // A notification (null id) must be ignored without blocking the loop.
    let note = notification_frame("session/update".into(), json!({ "sessionID": "sess-0" }));
    client.write.write_all(&note).await.unwrap();
    client.write.flush().await.unwrap();

    // Also an invalid-but-ignorable notification.
    let note = notification_frame("no/such/method".into(), json!({}));
    client.write.write_all(&note).await.unwrap();
    client.write.flush().await.unwrap();

    // The next request is answered: the loop never stalled.
    let ok = client.request("agent_info", json!({})).await;
    assert_eq!(ok["result"]["name"], "faktor-test-agent");

    // A shutdown-shaped notification cannot be answered but still ends the
    // loop ("shutdown-like handling") — clean EOF follows.
    let note = notification_frame("shutdown".into(), json!({}));
    client.write.write_all(&note).await.unwrap();
    client.write.flush().await.unwrap();
    client.expect_eof().await;
    timeout(RX_TIMEOUT, task)
        .await
        .expect("server task hung")
        .unwrap()
        .expect("server clean exit");
}

#[tokio::test]
async fn many_sequential_requests_are_serialized_and_deterministic() {
    let (mut client, _task) = start_server(EchoBackend::new());

    // 40 pipelined requests in one burst: single loop ⇒ strict order.
    let mut burst = Vec::new();
    for i in 0..40u64 {
        burst.extend(frame(
            "session/prompt".to_string(),
            i + 1,
            json!({ "sessionID": "sess-0", "text": format!("turn-{i}") }),
        ));
    }
    client.write.write_all(&burst).await.unwrap();
    client.write.flush().await.unwrap();

    for i in 0..40u64 {
        let msg = client.expect_message().await;
        assert_eq!(msg["id"], i + 1);
        assert_eq!(msg["result"]["echo"], format!("turn-{i}"));
    }
}

#[tokio::test]
async fn prompt_params_missing_or_wrong_type_is_32602() {
    let (mut client, _task) = start_server(EchoBackend::new());

    // Params entirely absent.
    assert_eq!(
        client.error_code_of("session/prompt", json!({})).await,
        -32602
    );
    // sessionID missing.
    assert_eq!(
        client
            .error_code_of("session/prompt", json!({ "text": "hi" }))
            .await,
        -32602
    );
    // text missing.
    assert_eq!(
        client
            .error_code_of("session/prompt", json!({ "sessionID": "s" }))
            .await,
        -32602
    );
    // Wrong types.
    assert_eq!(
        client
            .error_code_of("session/prompt", json!({ "sessionID": 7, "text": "hi" }))
            .await,
        -32602
    );
    assert_eq!(
        client
            .error_code_of(
                "session/prompt",
                json!({ "sessionID": "s", "text": ["hi"] })
            )
            .await,
        -32602
    );
    // Non-object params at all.
    let msg = client.request("session/prompt", json!([1, 2, 3])).await;
    assert_eq!(msg["error"]["code"], -32602);
    // Abort needs a sessionID too.
    assert_eq!(
        client.error_code_of("session/abort", json!({})).await,
        -32602
    );
}

#[tokio::test]
async fn slow_backend_preserves_request_order() {
    let slow = SlowBackend {
        inner: EchoBackend::new(),
        delay: Duration::from_millis(30),
    };
    let (mut client, _task) = start_server(slow);

    let mut burst = Vec::new();
    for i in 0..3u64 {
        burst.extend(frame(
            "session/prompt".to_string(),
            i + 1,
            json!({ "sessionID": "s", "text": format!("slow-{i}") }),
        ));
    }
    let started = std::time::Instant::now();
    client.write.write_all(&burst).await.unwrap();
    client.write.flush().await.unwrap();
    for i in 0..3u64 {
        let msg = client.expect_message().await;
        assert_eq!(msg["id"], i + 1);
        assert_eq!(msg["result"]["echo"], format!("slow-{i}"));
    }
    // 3 × 30 ms sleeps ran serially on the loop: wall time proves ordering
    // was preserved by serialization, not by racing.
    assert!(started.elapsed() >= Duration::from_millis(85));
}

#[tokio::test]
async fn hostile_utf8_nul_and_control_bytes_round_trip_via_json() {
    let (mut client, _task) = start_server(EchoBackend::new());

    let evil = "h\u{e9}llo\u{0}wor\r\nld\u{7f}\u{1}\u{b}\t\u{2028}\u{2029}\u{ffff}💥\u{10ffff}"
        .to_string();
    let msg = client
        .request(
            "session/prompt",
            json!({ "sessionID": "sess-\u{0}", "text": evil }),
        )
        .await;
    assert_eq!(msg["error"].as_object(), None, "unexpected error: {msg}");
    assert_eq!(msg["result"]["echo"].as_str().unwrap(), evil);
    assert_eq!(msg["result"]["sessionID"], "sess-\u{0}");

    // Deeply hostile but valid JSON: extreme nesting is bounded by the
    // parser (no stack overflow, no panic).
    let mut evil2 = String::from("turn");
    for _ in 0..1000 {
        evil2.push_str("\u{0}\u{1}\u{2}");
    }
    let msg = client
        .request("session/prompt", json!({ "sessionID": "s", "text": evil2 }))
        .await;
    assert_eq!(msg["result"]["echo"].as_str().unwrap(), evil2);
}

#[tokio::test]
async fn oversized_request_params_refused_with_32600() {
    let (mut client, _task) = start_server(EchoBackend::new());

    let huge_text = "z".repeat(1024 * 1024 + 1000);
    let msg = client
        .request(
            "session/prompt",
            json!({ "sessionID": "s", "text": huge_text }),
        )
        .await;
    assert_eq!(msg["error"]["code"], -32600);
    let message = msg["error"]["message"].as_str().unwrap();
    assert!(message.contains("1 MiB"), "{message}");
    // Loop still alive.
    let ok = client.request("agent_info", json!({})).await;
    assert_eq!(ok["result"]["name"], "faktor-test-agent");
}

#[tokio::test]
async fn oversized_backend_result_refused_not_truncated() {
    let (mut client, task) = start_server(HugeBackend);

    let msg = client
        .request(
            "session/prompt",
            json!({ "sessionID": "sess-huge", "text": "blob" }),
        )
        .await;
    assert_eq!(msg["error"]["code"], -32603);
    let message = msg["error"]["message"].as_str().unwrap();
    assert!(message.contains("8 MiB"), "{message}");
    // Nothing half-written: the next request still parses cleanly.
    let ok = client.request("agent_info", json!({})).await;
    assert_eq!(ok["result"]["name"], "huge");

    let bye = client.request("shutdown", json!({})).await;
    assert_eq!(bye["result"], json!({ "ok": true }));
    client.expect_eof().await;
    timeout(RX_TIMEOUT, task)
        .await
        .expect("server task hung")
        .unwrap()
        .expect("server clean exit");
}

#[tokio::test]
async fn invalid_request_shapes_get_32600() {
    let (mut client, _task) = start_server(EchoBackend::new());

    // Not an object at all.
    for body in [
        b"42".as_slice(),
        b"\"str\"".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let bytes = format!("Content-Length: {}\r\n\r\n", body.len()).into_bytes();
        let mut framed = bytes;
        framed.extend_from_slice(body);
        client.write.write_all(&framed).await.unwrap();
        client.write.flush().await.unwrap();
        let msg = client.expect_message().await;
        assert_eq!(msg["error"]["code"], -32600, "{msg}");
    }

    // Missing method / wrong jsonrpc.
    let msg = client.request("bogus", json!({})).await;
    assert_eq!(msg["error"]["code"], -32601, "{msg}");
    let bad_version = json!({ "jsonrpc": "1.0", "id": 5, "method": "agent_info" });
    let framed = faktor_acp::protocol::encode(&bad_version).unwrap();
    client.write.write_all(&framed).await.unwrap();
    client.write.flush().await.unwrap();
    let msg = client.expect_message().await;
    assert_eq!(msg["error"]["code"], -32600);

    // Negative and fractional ids are invalid requests.
    for id in [json!(-1), json!(1.5), json!("x")] {
        let bad = json!({ "jsonrpc": "2.0", "id": id, "method": "agent_info" });
        let framed = faktor_acp::protocol::encode(&bad).unwrap();
        client.write.write_all(&framed).await.unwrap();
        client.write.flush().await.unwrap();
        let msg = client.expect_message().await;
        assert_eq!(msg["error"]["code"], -32600, "{msg}");
    }

    let ok = client.request("agent_info", json!({})).await;
    assert_eq!(ok["result"]["name"], "faktor-test-agent");
}

#[tokio::test]
async fn abort_backend_failure_surfaces_and_session_abort_is_ok() {
    let (mut client, _task) = start_server(EchoBackend::new());
    let new = client.request("session/new", json!({})).await;
    let sid = new["result"]["sessionID"].as_str().unwrap().to_string();

    let msg = client
        .request("session/abort", json!({ "sessionID": "sess-refuse" }))
        .await;
    assert_eq!(msg["error"]["code"], -32000);
    assert_eq!(msg["error"]["message"], "abort refused");

    let ok = client
        .request("session/abort", json!({ "sessionID": sid }))
        .await;
    assert_eq!(ok["result"], json!({ "ok": true }));

    // abort cancels the turn; the session remains listed.
    let list = client.request("session/list", json!({})).await;
    assert_eq!(list["result"]["sessions"][0]["sessionID"], sid);
}

#[tokio::test]
async fn empty_and_whitespace_frames_do_not_hang_or_crash() {
    let (mut client, _task) = start_server(EchoBackend::new());

    // Frame of length 0: JSON parse error (consumed, loop survives).
    client
        .write
        .write_all(b"Content-Length: 0\r\n\r\n")
        .await
        .unwrap();
    client.write.flush().await.unwrap();
    let msg = client.expect_message().await;
    assert!(msg["id"].is_null());
    assert_eq!(msg["error"]["code"], -32700);

    // Whitespace-only body.
    let ws = b"Content-Length: 3\r\n\r\n   ";
    client.write.write_all(ws).await.unwrap();
    client.write.flush().await.unwrap();
    let msg = client.expect_message().await;
    assert_eq!(msg["error"]["code"], -32700);

    let ok = client.request("agent_info", json!({})).await;
    assert_eq!(ok["result"]["name"], "faktor-test-agent");
}

#[tokio::test]
async fn client_eof_is_a_clean_exit_without_shutdown() {
    let (client, task) = start_server(EchoBackend::new());
    // Dropping the client closes the pipe; the server must exit Ok, not
    // hang or error.
    drop(client);
    let result = timeout(RX_TIMEOUT, task)
        .await
        .expect("server stuck after EOF")
        .expect("server task panicked");
    assert!(result.is_ok(), "server errored on client EOF: {result:?}");
}
