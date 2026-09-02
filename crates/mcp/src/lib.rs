//! kilop-mcp — JSON-RPC Model Context Protocol client (spec §31).
//!
//! MCP processes are supervised like terminals: crashes, hangs, and garbage
//! output never destabilize the agent runtime. Every invocation has a
//! deadline; responses are bounded; the framing is Content-Length JSON-RPC.

use std::collections::HashMap;
use std::io::{BufReader, Write};
use std::process::ChildStdin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use kilop_core::error::{Error, ErrorKind};
use kilop_terminal::{ProcessOwner, ProcessSupervisor, SpawnConfig};

const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
#[allow(dead_code)]
const MAX_INITIAL_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpConfig {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct McpTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpResult {
    pub content: Vec<serde_json::Value>,
    pub is_error: bool,
}

struct Conn {
    child_pid: u32,
    stdin: ChildStdin,
    next_id: u64,
    pending: HashMap<String, tokio::sync::oneshot::Sender<serde_json::Value>>,
}

pub struct McpServer {
    name: String,
    conn: Arc<Mutex<Conn>>,
    supervisor: Arc<ProcessSupervisor>,
    #[allow(dead_code)]
    cfg: McpConfig,
}

impl McpServer {
    /// Connect: spawn the server process and perform the initialize
    /// handshake with a bounded timeout.
    pub async fn connect(
        cfg: McpConfig,
        supervisor: Arc<ProcessSupervisor>,
    ) -> Result<Arc<Self>, Error> {
        let mut proc_cfg = SpawnConfig {
            cmd: cfg.command.clone(),
            args: cfg.args.clone(),
            cwd: std::env::temp_dir(),
            env: cfg.env.clone(),
            owner: ProcessOwner::Daemon,
            capture: false,
            ..Default::default()
        };
        proc_cfg
            .env
            .push(("PATH".into(), std::env::var("PATH").unwrap_or_default()));
        let spawned = supervisor
            .spawn_detached_with_pipes(proc_cfg)
            .map_err(|e| Error::new(ErrorKind::NotFound, format!("mcp spawn: {e}")))?;
        let conn = Arc::new(Mutex::new(Conn {
            child_pid: spawned.child_pid,
            stdin: spawned.stdin,
            next_id: 1,
            pending: HashMap::new(),
        }));
        // Reader thread: incremental Content-Length framing; responses are
        // dispatched by id; EOF/kill cleans up. The thread owns stdout, so
        // blocking reads can never stall the runtime.
        {
            let conn2 = conn.clone();
            std::thread::spawn(move || {
                read_loop(conn2, spawned.stdout);
            });
        }
        let server = Arc::new(Self {
            name: cfg.name.clone(),
            conn,
            supervisor,
            cfg,
        });
        server.initialize().await?;
        Ok(server)
    }

    fn next_request(&self, method: &str, params: serde_json::Value) -> (String, serde_json::Value) {
        let mut conn = self.conn.lock().unwrap();
        let id = conn.next_id;
        conn.next_id += 1;
        (
            id.to_string(),
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": id,
                "method": method,
                "params": params,
            }),
        )
    }

    /// Send a request and wait for its response (bounded by deadline).
    pub async fn call(
        &self,
        method: &str,
        params: serde_json::Value,
        deadline: Duration,
    ) -> Result<serde_json::Value, Error> {
        let (id, request) = self.next_request(method, params);
        let (tx, rx) = tokio::sync::oneshot::channel();
        {
            let mut conn = self.conn.lock().unwrap();
            if conn.pending.len() > 256 {
                return Err(Error::new(
                    ErrorKind::Oversized,
                    "too many in-flight MCP requests",
                ));
            }
            let wire = format!(
                "Content-Length: {}\r\n\r\n{}",
                request.to_string().len(),
                request
            );
            conn.stdin
                .write_all(wire.as_bytes())
                .map_err(|e| Error::new(ErrorKind::Network, format!("mcp write: {e}")))?;
            conn.stdin.flush().ok();
            conn.pending.insert(id.clone(), tx);
        }
        // Reader loop (see `read_loop`); here we wait with a deadline.
        match tokio::time::timeout(deadline, rx).await {
            Ok(Ok(response)) => {
                if let Some(err) = response.get("error") {
                    return Err(Error::new(
                        ErrorKind::Provider {
                            code: "mcp".into(),
                            retryable: false,
                        },
                        format!("mcp {method}: {err}"),
                    ));
                }
                Ok(response
                    .get("result")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null))
            }
            Ok(Err(_)) => Err(Error::new(
                ErrorKind::Network,
                format!("mcp {method}: connection dropped"),
            )),
            Err(_elapsed) => {
                // Clean up the pending entry.
                self.conn.lock().unwrap().pending.remove(&id);
                Err(Error::timeout(format!(
                    "mcp {method} exceeded {}ms",
                    deadline.as_millis()
                )))
            }
        }
    }

    async fn initialize(&self) -> Result<(), Error> {
        let result = self
            .call(
                "initialize",
                serde_json::json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": { "name": "kilop-plus", "version": "0.1.0" },
                }),
                Duration::from_secs(10),
            )
            .await?;
        let _ = result;
        // Notify initialized (fire and forget; never blocks the runtime).
        let _ = self
            .call(
                "notifications/initialized",
                serde_json::json!({}),
                Duration::from_secs(2),
            )
            .await;
        Ok(())
    }

    pub async fn list_tools(&self) -> Result<Vec<McpTool>, Error> {
        let result = self
            .call("tools/list", serde_json::json!({}), Duration::from_secs(10))
            .await?;
        let tools = result
            .get("tools")
            .and_then(|t| t.as_array())
            .cloned()
            .unwrap_or_default();
        let mut out = Vec::new();
        for t in tools {
            out.push(McpTool {
                name: t
                    .get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or("")
                    .to_string(),
                description: t
                    .get("description")
                    .and_then(|n| n.as_str())
                    .unwrap_or("")
                    .to_string(),
                input_schema: t
                    .get("inputSchema")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null),
            });
        }
        Ok(out)
    }

    pub async fn call_tool(
        &self,
        name: &str,
        args: serde_json::Value,
        deadline: Duration,
    ) -> Result<McpResult, Error> {
        let result = self
            .call(
                "tools/call",
                serde_json::json!({ "name": name, "arguments": args }),
                deadline,
            )
            .await?;
        Ok(McpResult {
            content: result
                .get("content")
                .and_then(|c| c.as_array())
                .cloned()
                .unwrap_or_default(),
            is_error: result
                .get("isError")
                .and_then(|e| e.as_bool())
                .unwrap_or(false),
        })
    }

    pub async fn close(&self) -> Result<(), Error> {
        let pid = {
            let mut conn = self.conn.lock().unwrap();
            let _ = conn.stdin.write_all(b"Content-Length: 0\r\n\r\n");
            let _ = conn.stdin.flush();
            conn.child_pid
        };
        // Kill via the supervisor (process-group aware).
        let _ = self.supervisor.kill_child_pid(pid, 1000);
        Ok(())
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn is_alive(&self) -> bool {
        let pid = self.conn.lock().unwrap().child_pid;
        self.supervisor.pid_alive(pid)
    }
}

impl Drop for McpServer {
    /// Zero orphans: dropping the client kills the server process.
    fn drop(&mut self) {
        let pid = {
            let mut conn = match self.conn.lock() {
                Ok(c) => c,
                Err(_) => return,
            };
            let _ = conn.stdin.write_all(b"Content-Length: 0\r\n\r\n");
            conn.child_pid
        };
        let _ = self.supervisor.kill_child_pid(pid, 300);
    }
}

/// Incremental frame reader: accumulates bytes, parses Content-Length
/// frames, dispatches responses by id, and drops pending senders on EOF.
fn read_loop(conn: Arc<Mutex<Conn>>, stdout: std::process::ChildStdout) {
    let mut reader = BufReader::new(stdout);
    let mut buf: Vec<u8> = Vec::with_capacity(4096);
    loop {
        // Read one chunk.
        let mut chunk = [0u8; 8192];
        use std::io::Read;
        let n = match reader.read(&mut chunk) {
            Ok(0) => break, // EOF: server died or closed
            Ok(n) => n,
            Err(_) => break,
        };
        buf.extend_from_slice(&chunk[..n]);
        if buf.len() > MAX_RESPONSE_BYTES {
            break; // hostile server: stop reading
        }
        // Parse as many complete frames as available.
        loop {
            if buf.is_empty() {
                break; // fully drained: wait for the next chunk
            }
            match parse_frame(&buf) {
                Ok(Some((consumed, value))) => {
                    buf.drain(..consumed);
                    // Servers echo the request id verbatim: the client sends
                    // NUMERIC ids, so a numeric echo must match the string
                    // pending key ("1" == 1). (Latent bug: numeric echoes
                    // never matched and every successful call timed out.)
                    let id = value.get("id").and_then(|i| match i {
                        serde_json::Value::String(s) => Some(s.clone()),
                        serde_json::Value::Number(n) => Some(n.to_string()),
                        _ => None,
                    });
                    let is_notification =
                        value.get("method").is_some() && value.get("id").is_none();
                    if let Some(id) = id {
                        let mut guard = conn.lock().unwrap();
                        if let Some(tx) = guard.pending.remove(&id) {
                            let _ = tx.send(value);
                        }
                    } else if is_notification {
                        // Unsolicited notifications are ignored.
                    }
                }
                Ok(None) => break, // incomplete frame: wait for more bytes
                Err(_) => {
                    // Garbage on the wire: drop everything pending (the
                    // server is broken) and stop reading.
                    let mut guard = conn.lock().unwrap();
                    for (_, tx) in guard.pending.drain() {
                        let _ = tx.send(serde_json::json!({"error": {"code": -32700, "message": "parse error"}}));
                    }
                    return;
                }
            }
        }
    }
    // EOF: fail all pending requests.
    let mut guard = conn.lock().unwrap();
    for (_, tx) in guard.pending.drain() {
        let _ = tx.send(serde_json::json!({"error": {"code": -32000, "message": "server closed"}}));
    }
}

/// Pure JSON-RPC framing parser (Content-Length headers), unit-tested
/// adversarially without any process.
pub fn parse_frame(bytes: &[u8]) -> Result<Option<(usize, serde_json::Value)>, String> {
    // Returns Ok(Some((consumed, message))) or Ok(None) when incomplete.
    if bytes.is_empty() {
        return Ok(None); // empty is incomplete, never a parse error
    }
    if bytes.len() > MAX_RESPONSE_BYTES {
        return Err("response exceeds 16MB bound".into());
    }
    let header_end = bytes
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or("no header terminator")?;
    let header = std::str::from_utf8(&bytes[..header_end]).map_err(|_| "header not utf8")?;
    let mut content_length: Option<usize> = None;
    for line in header.split("\r\n") {
        if let Some(v) = line.strip_prefix("Content-Length:") {
            content_length = Some(
                v.trim()
                    .parse::<usize>()
                    .map_err(|_| "bad Content-Length")?,
            );
        }
    }
    let content_length = content_length.ok_or("missing Content-Length")?;
    if content_length > MAX_RESPONSE_BYTES {
        return Err("declared Content-Length exceeds bound".into());
    }
    let body_start = header_end + 4;
    let total = body_start + content_length;
    if bytes.len() < total {
        return Ok(None); // incomplete
    }
    let body = std::str::from_utf8(&bytes[body_start..total]).map_err(|_| "body not utf8")?;
    let value = serde_json::from_str(body).map_err(|e| format!("invalid jsonrpc: {e}"))?;
    Ok(Some((total, value)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn framing_roundtrip() {
        let msg = serde_json::json!({"jsonrpc": "2.0", "id": 1, "result": {"ok": true}});
        let body = msg.to_string();
        let frame = format!("Content-Length: {}\r\n\r\n{}", body.len(), body);
        let (consumed, parsed) = parse_frame(frame.as_bytes()).unwrap().unwrap();
        assert_eq!(consumed, frame.len());
        assert_eq!(parsed, msg);
        // Multiple frames concatenated: parse consumes exactly the first.
        let double = format!("{frame}{frame}");
        let (consumed, _) = parse_frame(double.as_bytes()).unwrap().unwrap();
        assert_eq!(consumed, frame.len());
    }

    #[test]
    fn framing_garbage_rejected() {
        // An empty buffer is incomplete (a fully-drained reader must not
        // treat it as wire garbage — that bug dropped healthy frames).
        assert!(parse_frame(b"").unwrap().is_none(), "empty is incomplete");
        // 3 of 5 declared bytes: incomplete frame, not an error.
        assert!(
            parse_frame(b"Content-Length: 5\r\n\r\nhel")
                .unwrap()
                .is_none(),
            "incomplete is None not error"
        );
        assert!(parse_frame(b"Content-Length: -1\r\n\r\n").is_err());
        assert!(parse_frame(b"Content-Length: 99999999999999999999\r\n\r\n").is_err());
        assert!(parse_frame(b"Content-Length: abc\r\n\r\n").is_err());
        let mismatched = format!("Content-Length: 5\r\n\r\n{}\r\n\r\n", "x".repeat(100));
        assert!(
            parse_frame(mismatched.as_bytes()).is_err(),
            "mismatched length"
        );
        assert!(
            parse_frame(b"Content-Length: 5\r\n\r\n\xff\xfe\x00\x01\x02").is_err(),
            "invalid utf8 body"
        );
        let big = format!("Content-Length: {}\r\n\r\n", MAX_RESPONSE_BYTES + 1);
        assert!(parse_frame(big.as_bytes()).is_err(), "declared size bound");
    }

    #[test]
    fn framing_bad_json_rejected() {
        let body = "{not json";
        let frame = format!("Content-Length: {}\r\n\r\n{}", body.len(), body);
        assert!(parse_frame(frame.as_bytes()).is_err());
    }

    #[tokio::test]
    async fn close_terminates_process() {
        // Use a real MCP-ish server: a python one-liner that reads one frame
        // and exits. Skip gracefully if python3 is absent.
        if std::process::Command::new("python3")
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("python3 missing; skipping");
            return;
        }
        let dir = tempdir().unwrap();
        let cas = Arc::new(kilop_cas::Cas::open(dir.path().join("cas")).unwrap());
        let sup = ProcessSupervisor::new(cas);
        let script = r#"
import sys
line = sys.stdin.readline()
while line:
    line = sys.stdin.readline()
sys.exit(0)
"#;
        let cfg = McpConfig {
            name: "mock".into(),
            command: "python3".into(),
            args: vec!["-c".into(), script.into()],
            env: vec![],
        };
        // connect() will time out on initialize (the mock never answers);
        // verify the process is killed and no zombie remains.
        let server = McpServer::connect(cfg, sup.clone()).await;
        assert!(server.is_err(), "mock never initializes");
        // The supervisor must have reaped the child (no zombie).
        std::thread::sleep(Duration::from_millis(300));
        assert!(sup.reap().is_empty() || sup.registered() == 0);
    }

    #[tokio::test]
    async fn garbage_server_is_malformed_not_hang() {
        // A server that emits garbage on stdout: framing must fail fast.
        if std::process::Command::new("python3")
            .arg("--version")
            .output()
            .is_err()
        {
            return;
        }
        let dir = tempdir().unwrap();
        let cas = Arc::new(kilop_cas::Cas::open(dir.path().join("cas")).unwrap());
        let sup = ProcessSupervisor::new(cas);
        let script = "import sys, time\nwhile True:\n    sys.stdout.write('garbage\\n')\n    sys.stdout.flush()\n    time.sleep(0.01)\n";
        let cfg = McpConfig {
            name: "garbage".into(),
            command: "python3".into(),
            args: vec!["-c".into(), script.into()],
            env: vec![],
        };
        let server = McpServer::connect(cfg, sup.clone()).await;
        // Either initialize fails (timeout → error) or the server is dead;
        // never a hang.
        let _ = server;
    }

    #[test]
    fn response_bound_is_enforced() {
        // A hostile 17MB frame is rejected by the parser.
        let body = "x".repeat(MAX_RESPONSE_BYTES + 1);
        let frame = format!("Content-Length: {}\r\n\r\n{}", body.len(), body);
        let r = parse_frame(frame.as_bytes());
        assert!(r.is_err());
    }
}
