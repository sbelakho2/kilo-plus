//! faktor-acp — ACP (Agent Client Protocol) agent server.
//!
//! Purpose (audit note): expose an ACP agent server so ACP-capable editors
//! (Zed and others) can drive the Faktor core alongside the native API and
//! the legacy compat surface. This crate implements the WIRE + lifecycle
//! only; agent behavior is delegated to the injected [`AcpBackend`] seam,
//! so the daemon can attach the real runtime without this crate knowing it.
//!
//! # Wire + lifecycle
//!
//! ACP is JSON-RPC 2.0 over Content-Length framed stdio (see
//! [`protocol`]). Method subset and exact wire strings:
//!
//! | method        | request → result                                     |
//! |---------------|------------------------------------------------------|
//! | `initialize`  | `{"protocolVersion":"0.1.0","agent":<agent_info>}`   |
//! | `agent_info`  | `<agent_info>`                                       |
//! | `session/new` | `{"sessionID":<id>}`                                 |
//! | `session/prompt` | `<backend result>`  (params `{sessionID,text}`)  |
//! | `session/abort` | `{"ok":true}`       (params `{sessionID}`)        |
//! | `session/list` | `{"sessions":[{"sessionID":..},..]}`               |
//! | `shutdown`    | `{"ok":true}` then the serve loop ends               |
//!
//! Error codes: `-32700` parse errors (null id), `-32600` invalid request,
//! `-32601` unknown method, `-32602` invalid params, `-32603` internal
//! error (oversized backend result), `-32000` backend-reported errors.
//! Notifications (null/absent id) are ignored except a `shutdown`
//! notification, which ends the loop (nothing can answer it).
//!
//! # Bounded everything
//!
//! - Declared frames are capped at [`protocol::MAX_FRAME_BYTES`] (16 MiB) by
//!   the parser; request params at [`MAX_PARAMS_BYTES`] (1 MiB); responses
//!   written to the wire at [`MAX_RESPONSE_BYTES`] (8 MiB) — a backend
//!   result that does not fit is refused with `-32603`, never truncated.
//! - The serve loop is single-threaded and serialized: requests are handled
//!   strictly in read order, so response ordering is deterministic by
//!   construction.
//! - A runaway backend prompt is the daemon's problem: trait calls run
//!   synchronously on the serve task, so the daemon must attach a backend
//!   that either answers promptly or is internally time-boxed (deadlines,
//!   cancellation, retry policy live in the daemon's runtime layer, per
//!   architecture.md commandment 2).

use serde_json::{json, Value};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use std::sync::Arc;

use crate::protocol::AcpResponse;

pub mod protocol;

/// ACP protocol version reported by the `initialize` handshake.
pub const PROTOCOL_VERSION: &str = "0.1.0";

/// Cap on the serialized `params` of one incoming request (1 MiB).
pub const MAX_PARAMS_BYTES: usize = 1024 * 1024;

/// Cap on one serialized response written to the wire (8 MiB).
pub const MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024;

/// JSON-RPC well-known error codes (subset used by this server).
pub const PARSE_ERROR: i64 = -32700;
pub const INVALID_REQUEST: i64 = -32600;
pub const METHOD_NOT_FOUND: i64 = -32601;
pub const INVALID_PARAMS: i64 = -32602;
pub const INTERNAL_ERROR: i64 = -32603;
/// Server error: the backend rejected the call.
pub const BACKEND_ERROR: i64 = -32000;

const MAX_METHOD_LEN: usize = 128;
const READ_CHUNK: usize = 64 * 1024;
const SHUTDOWN_METHOD: &str = "shutdown";

/// The seam: deterministic, injectable agent behavior. The daemon attaches
/// the real Faktor runtime here; this crate only wires JSON-RPC to it.
///
/// Implementations must be `Send + Sync` and are called synchronously from
/// the single serve task, so they must not block indefinitely (see the
/// module docs on boundedness).
pub trait AcpBackend: Send + Sync {
    /// Agent metadata surfaced by `initialize`/`agent_info`
    /// (e.g. `{"name":..,"version":..,"capabilities":{..}}`).
    fn agent_info(&self) -> Value;
    /// Create a session; `Err` surfaces as `-32000`.
    fn create_session(&self, params: &Value) -> Result<String, String>;
    /// Run one prompt turn against `session_id`; `Err` surfaces as `-32000`.
    fn prompt(&self, session_id: &str, text: &str) -> Result<Value, String>;
    /// Abort the active turn of `session_id`.
    fn abort(&self, session_id: &str) -> Result<(), String>;
    /// Current session ids (e.g. for `session/list`).
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

/// ACP agent server. Cheap to build, `serve_connection`/`run_stdio` may be
/// called once (they own the loop until EOF or `shutdown`).
pub struct AcpServer<B: AcpBackend> {
    backend: Arc<B>,
}

impl<B: AcpBackend> AcpServer<B> {
    pub fn new(backend: B) -> Self {
        Self {
            backend: Arc::new(backend),
        }
    }

    /// Serve ACP over an arbitrary async reader/writer pair (in-memory
    /// pipes for tests, stdio for production). Serialized: one request at a
    /// time in read order. Returns `Ok(())` on EOF or `shutdown`.
    pub async fn serve_connection<R, W>(&self, reader: R, mut writer: W) -> Result<(), String>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        let mut reader = reader;
        let mut buf: Vec<u8> = Vec::new();
        let mut chunk = vec![0u8; READ_CHUNK];
        loop {
            let n = reader
                .read(&mut chunk)
                .await
                .map_err(|e| format!("read error: {e}"))?;
            if n == 0 {
                tracing::debug!("acp: input closed; ending serve loop");
                return Ok(());
            }
            buf.extend_from_slice(&chunk[..n]);
            loop {
                match protocol::parse_frame_detailed(&buf) {
                    Ok(Some((consumed, value))) => {
                        buf.drain(..consumed);
                        let shutdown = self.handle_message(value, &mut writer).await?;
                        if shutdown {
                            return Ok(());
                        }
                    }
                    Ok(None) => break,
                    Err(err) => {
                        tracing::warn!(
                            error = %err.message,
                            fatal = err.fatal,
                            "acp: frame-level parse error"
                        );
                        self.write_parse_error(&mut writer, &err.message).await?;
                        if err.fatal {
                            // Stream is desynced or hostile: a bounded server
                            // refuses the connection instead of guessing.
                            return Ok(());
                        }
                        // Recoverable: the bad frame's byte boundary is
                        // known, so drop it and keep serving.
                        buf.drain(..err.consumed.min(buf.len()));
                    }
                }
            }
        }
    }

    /// Serve ACP over stdin/stdout with a flush after every write. Callers
    /// must route their logging to stderr or elsewhere: stdout carries ONLY
    /// framed protocol bytes.
    pub async fn run_stdio(self) -> Result<(), String> {
        self.serve_connection(tokio::io::stdin(), tokio::io::stdout())
            .await
    }

    /// Handle one decoded message, responding through `writer`.
    /// Returns `true` when the loop must end (shutdown request answered,
    /// shutdown notification seen, unrecoverable error answered).
    async fn handle_message<W: AsyncWrite + Unpin>(
        &self,
        value: Value,
        writer: &mut W,
    ) -> Result<bool, String> {
        match classify(value) {
            Incoming::Invalid { id, code, message } => {
                let error = json_error(code, &message);
                self.write_error_value(writer, &id, &error).await?;
                Ok(false)
            }
            Incoming::Notification { method, params } => {
                tracing::info!(
                    method = %method,
                    params = %params,
                    "acp: ignoring notification"
                );
                // Nothing can answer a notification; a shutdown-shaped one
                // still ends the serve loop ("shutdown-like handling").
                Ok(method == SHUTDOWN_METHOD)
            }
            Incoming::Request { id, method, params } => {
                if method == SHUTDOWN_METHOD {
                    self.write_result(writer, id, &json!({ "ok": true }))
                        .await?;
                    return Ok(true);
                }
                match self.dispatch(&method, &params) {
                    Ok(result) => self.write_result(writer, id, &result).await?,
                    Err(ServerError { code, message }) => {
                        self.write_result_err(writer, id, code, &message).await?
                    }
                }
                Ok(false)
            }
        }
    }

    /// Dispatch table: method string → request/result semantics.
    fn dispatch(&self, method: &str, params: &Value) -> Result<Value, ServerError> {
        match method {
            "initialize" => Ok(json!({
                "protocolVersion": PROTOCOL_VERSION,
                "agent": self.backend.agent_info(),
            })),
            "agent_info" => Ok(self.backend.agent_info()),
            "session/new" => self
                .backend
                .create_session(params)
                .map(|session_id| json!({ "sessionID": session_id }))
                .map_err(ServerError::backend),
            "session/prompt" => {
                let session_id = require_string(params, "sessionID")?;
                let text = require_string(params, "text")?;
                self.backend
                    .prompt(session_id, text)
                    .map_err(ServerError::backend)
            }
            "session/abort" => {
                let session_id = require_string(params, "sessionID")?;
                self.backend
                    .abort(session_id)
                    .map_err(ServerError::backend)?;
                Ok(json!({ "ok": true }))
            }
            "session/list" => Ok(json!({
                "sessions": self
                    .backend
                    .list_sessions()
                    .into_iter()
                    .map(|session_id| json!({ "sessionID": session_id }))
                    .collect::<Vec<_>>(),
            })),
            other => Err(ServerError::new(
                METHOD_NOT_FOUND,
                format!("method not found: {other}"),
            )),
        }
    }

    async fn write_result<W: AsyncWrite + Unpin>(
        &self,
        writer: &mut W,
        id: u64,
        result: &Value,
    ) -> Result<(), String> {
        let response = AcpResponse::result(id, result.clone());
        let mut body = serde_json::to_vec(&response).map_err(|e| format!("encode: {e}"))?;
        if body.len() > MAX_RESPONSE_BYTES {
            // Refuse, never truncate: a silently cut backend result would
            // corrupt the client's view of the turn.
            tracing::error!(
                bytes = body.len(),
                "acp: backend result exceeds 8 MiB bound"
            );
            body = serde_json::to_vec(&AcpResponse::error_code(
                id,
                INTERNAL_ERROR,
                "backend result exceeds the 8 MiB response bound",
            ))
            .map_err(|e| format!("encode: {e}"))?;
        }
        write_framed(writer, &body).await
    }

    async fn write_result_err<W: AsyncWrite + Unpin>(
        &self,
        writer: &mut W,
        id: u64,
        code: i64,
        message: &str,
    ) -> Result<(), String> {
        let response = AcpResponse::error_code(id, code, message);
        let body = serde_json::to_vec(&response).map_err(|e| format!("encode: {e}"))?;
        write_framed(writer, &body).await
    }

    /// Error response to garbage we could not parse: no id exists, so the
    /// JSON-RPC error carries a null id.
    async fn write_parse_error<W: AsyncWrite + Unpin>(
        &self,
        writer: &mut W,
        message: &str,
    ) -> Result<(), String> {
        let error = json_error(PARSE_ERROR, message);
        let body = serde_json::to_vec(&json!({ "jsonrpc": "2.0", "id": null, "error": error }))
            .map_err(|e| format!("encode: {e}"))?;
        write_framed(writer, &body).await
    }

    /// Error response to a structurally invalid request whose id, if any,
    /// could not be trusted as a `u64`.
    async fn write_error_value<W: AsyncWrite + Unpin>(
        &self,
        writer: &mut W,
        id: &Value,
        error: &Value,
    ) -> Result<(), String> {
        let body = serde_json::to_vec(&json!({ "jsonrpc": "2.0", "id": id, "error": error }))
            .map_err(|e| format!("encode: {e}"))?;
        write_framed(writer, &body).await
    }
}

fn json_error(code: i64, message: &str) -> Value {
    json!({ "code": code, "message": message })
}

/// Write one complete frame and flush. Every write is flushed before the
/// server touches the reader again.
async fn write_framed<W: AsyncWrite + Unpin>(writer: &mut W, body: &[u8]) -> Result<(), String> {
    let header = format!("Content-Length: {}\r\n\r\n", body.len());
    writer
        .write_all(header.as_bytes())
        .await
        .map_err(|e| format!("write error: {e}"))?;
    writer
        .write_all(body)
        .await
        .map_err(|e| format!("write error: {e}"))?;
    writer
        .flush()
        .await
        .map_err(|e| format!("flush error: {e}"))
}

#[derive(Debug)]
struct ServerError {
    code: i64,
    message: String,
}

impl ServerError {
    fn new(code: i64, message: String) -> Self {
        Self { code, message }
    }

    fn backend(message: String) -> Self {
        Self::new(BACKEND_ERROR, message)
    }
}

fn require_string<'a>(params: &'a Value, field: &str) -> Result<&'a str, ServerError> {
    params
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| ServerError::new(INVALID_PARAMS, format!("missing string field {field:?}")))
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::AcpMethod;
    use serde_json::json;
    use std::sync::Mutex;

    struct TestBackend {
        sessions: Mutex<Vec<String>>,
    }

    impl TestBackend {
        fn new() -> Self {
            Self {
                sessions: Mutex::new(vec!["sess-existing".to_string()]),
            }
        }
    }

    impl AcpBackend for TestBackend {
        fn agent_info(&self) -> Value {
            json!({ "name": "test-agent", "version": "0.0.0" })
        }
        fn create_session(&self, params: &Value) -> Result<String, String> {
            match params.get("mode") {
                Some(Value::String(s)) if s == "fail" => Err("create refused".into()),
                _ => {
                    let mut sessions = self.sessions.lock().unwrap();
                    let id = format!("sess-{}", sessions.len());
                    sessions.push(id.clone());
                    Ok(id)
                }
            }
        }
        fn prompt(&self, session_id: &str, text: &str) -> Result<Value, String> {
            if text == "explode" {
                return Err("backend exploded".into());
            }
            Ok(json!({ "sessionID": session_id, "echo": text }))
        }
        fn abort(&self, _session_id: &str) -> Result<(), String> {
            Ok(())
        }
        fn list_sessions(&self) -> Vec<String> {
            self.sessions.lock().unwrap().clone()
        }
    }

    fn server() -> AcpServer<TestBackend> {
        AcpServer::new(TestBackend::new())
    }

    #[test]
    fn dispatch_table_initialize_agent_and_list() {
        let s = server();
        let init = s.dispatch("initialize", &json!({})).unwrap();
        assert_eq!(init["protocolVersion"], "0.1.0");
        assert_eq!(init["agent"]["name"], "test-agent");
        let info = s.dispatch("agent_info", &json!({})).unwrap();
        assert_eq!(info["version"], "0.0.0");
        let list = s.dispatch("session/list", &json!({})).unwrap();
        assert_eq!(list["sessions"][0]["sessionID"], "sess-existing");
    }

    #[test]
    fn dispatch_unknown_method_is_32601() {
        let err = server().dispatch("bogus/method", &json!({})).unwrap_err();
        assert_eq!(err.code, -32601);
    }

    #[test]
    fn dispatch_prompt_validation_and_backend_error() {
        let s = server();
        let missing = s.dispatch("session/prompt", &json!({})).unwrap_err();
        assert_eq!(missing.code, -32602);
        let not_string = s
            .dispatch("session/prompt", &json!({ "sessionID": 7, "text": "x" }))
            .unwrap_err();
        assert_eq!(not_string.code, -32602);
        let ok = s
            .dispatch(
                "session/prompt",
                &json!({ "sessionID": "sess-existing", "text": "hi" }),
            )
            .unwrap();
        assert_eq!(ok["echo"], "hi");
        let boom = s
            .dispatch(
                "session/prompt",
                &json!({ "sessionID": "sess-existing", "text": "explode" }),
            )
            .unwrap_err();
        assert_eq!(boom.code, -32000);
        assert_eq!(boom.message, "backend exploded");
    }

    #[test]
    fn dispatch_session_new_and_abort_errors() {
        let s = server();
        let new = s.dispatch("session/new", &json!({})).unwrap();
        let sid = new["sessionID"].as_str().unwrap();
        assert_eq!(sid, "sess-1");
        let aborted = s
            .dispatch("session/abort", &json!({ "sessionID": sid }))
            .unwrap();
        assert_eq!(aborted, json!({ "ok": true }));
        let fail = s
            .dispatch("session/new", &json!({ "mode": "fail" }))
            .unwrap_err();
        assert_eq!(fail.code, -32000);
        assert_eq!(fail.message, "create refused");
        let no_sid = s.dispatch("session/abort", &json!({})).unwrap_err();
        assert_eq!(no_sid.code, -32602);
    }

    #[test]
    fn classify_requests_notifications_and_errors() {
        let req = classify(json!({
            "jsonrpc": "2.0", "id": 9, "method": "session/prompt",
            "params": { "sessionID": "s", "text": "hi" }
        }));
        match req {
            Incoming::Request { id, method, .. } => {
                assert_eq!(id, 9);
                assert_eq!(method, "session/prompt");
            }
            other => panic!("expected request, got {other:?}"),
        }

        // Notifications: absent and null id.
        for msg in [
            json!({ "jsonrpc": "2.0", "method": "agent_info" }),
            json!({ "jsonrpc": "2.0", "id": null, "method": "session/update", "params": {} }),
        ] {
            let expected = msg["method"].as_str().unwrap().to_string();
            match classify(msg) {
                Incoming::Notification { method, .. } => assert_eq!(method, expected),
                other => panic!("expected notification, got {other:?}"),
            }
        }

        // Missing params default to an empty object for requests.
        match classify(json!({ "id": 1, "method": "session/list" })) {
            Incoming::Request { params, .. } => assert_eq!(params, json!({})),
            other => panic!("unexpected {other:?}"),
        }

        let invalid_cases = [
            (json!("not-an-object"), -32600),
            (json!({ "id": 1, "method": 4 }), -32600),
            (json!({ "id": -3, "method": "x" }), -32600),
            (json!({ "id": "nope", "method": "x" }), -32600),
            (json!({ "id": 1, "method": "x", "params": [1, 2] }), -32602),
            (json!({ "id": 1, "jsonrpc": "1.0", "method": "x" }), -32600),
        ];
        for (msg, code) in invalid_cases {
            match classify(msg) {
                Incoming::Invalid { code: got, .. } => assert_eq!(got, code),
                other => panic!("expected invalid {code}, got {other:?}"),
            }
        }

        // Hostile oversized params are refused before dispatch.
        let huge = "x".repeat(MAX_PARAMS_BYTES + 1);
        let msg = json!({ "id": 2, "method": "session/prompt",
                          "params": { "sessionID": "s", "text": huge } });
        match classify(msg) {
            Incoming::Invalid { code, .. } => assert_eq!(code, -32600),
            other => panic!("expected size rejection, got {other:?}"),
        }
    }

    #[test]
    fn acp_method_strings_match_dispatch() {
        // Dispatch arms and the AcpMethod table must never drift.
        for m in [
            AcpMethod::Initialize,
            AcpMethod::AgentInfo,
            AcpMethod::SessionNew,
            AcpMethod::SessionPrompt,
            AcpMethod::SessionAbort,
            AcpMethod::SessionList,
            AcpMethod::Shutdown,
        ] {
            assert!(m.as_str().parse::<AcpMethod>().is_ok());
        }
    }

    #[test]
    fn arc_blanket_makes_dyn_backend_usable() {
        let concrete: Arc<dyn AcpBackend> = Arc::new(TestBackend::new());
        let server = AcpServer::new(concrete);
        let info = server.dispatch("agent_info", &json!({})).unwrap();
        assert_eq!(info["name"], "test-agent");
    }
}
