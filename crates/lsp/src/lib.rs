//! faktor-lsp — workspace-scoped language server integration (spec §32).
//!
//! Language servers are workspace resources, not session resources: one
//! daemon shares `rust-analyzer`/`typescript-language-server`/`pyright`
//! across sessions on the same workspace. Requests are multiplexed with
//! per-request ids; heavy servers are unloaded after workspace inactivity.
//!
//! # LSP protocol correctness
//!
//! - The server process is owned by the REAL `WorkspaceId` of the workspace
//!   it serves — never a placeholder id — and runs with its working
//!   directory at the workspace root, which is also what `initialize`
//!   reports as `rootUri`.
//! - JSON-RPC requests and notifications are distinct paths: a request
//!   carries an id and waits for its response; a notification is
//!   fire-and-forget with no id and no pending entry, so a server that
//!   (correctly) never answers a notification cannot stall the client.
//! - The `initialized` notification is sent right after the `initialize`
//!   RESULT and before any `didOpen`. Lifecycle shutdown is a request with
//!   an awaited response, followed by the `exit` notification, followed by a
//!   bounded grace wait for the process; the kill is the fallback, never
//!   the first move.
//! - Stderr is drained continuously into a bounded byte ring (recent tail
//!   kept for diagnostics; total bytes counted unboundedly) so a verbose
//!   server can never block itself on a full stderr pipe.

use std::collections::{HashMap, VecDeque};
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::ChildStdin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use faktor_core::error::{Error, ErrorKind};
use faktor_core::id::WorkspaceId;
use faktor_terminal::{ProcessOwner, ProcessSupervisor, SpawnConfig};

const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
/// Bounded everything: at most this many requests may be awaiting responses.
const MAX_INFLIGHT_REQUESTS: usize = 1024;
/// Stderr diagnostics ring: bytes beyond the cap are dropped from the head.
const STDERR_RING_CAP: usize = 64 * 1024;
/// Deadlines of the graceful lifecycle (request → exit notification → wait).
const SHUTDOWN_REQUEST_MS: u64 = 5_000;
const EXIT_GRACE_MS: u64 = 2_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LspConfig {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    /// The REAL workspace root: the server's working directory AND the
    /// `rootUri` reported in `initialize` (file:// URI of this path).
    pub root: PathBuf,
}

/// Shared connection state; guarded so the reader thread, drain threads and
/// async request/notify paths never interleave a write.
struct LspConn {
    child_pid: u32,
    stdin: ChildStdin,
    next_id: u64,
    pending: HashMap<String, tokio::sync::oneshot::Sender<serde_json::Value>>,
}

/// Bounded stderr tail in BYTES (recent tail kept for diagnostics). Total
/// bytes drained are counted separately — flooding is observable even
/// though the retained tail is capped.
#[derive(Debug, Default)]
struct StderrRing {
    bytes: VecDeque<u8>,
    total: u64,
    cap: usize,
}

impl StderrRing {
    fn new(cap: usize) -> Self {
        Self {
            bytes: VecDeque::with_capacity(cap.min(4096)),
            total: 0,
            cap,
        }
    }

    fn push(&mut self, chunk: &[u8]) {
        self.total = self.total.saturating_add(chunk.len() as u64);
        if chunk.len() >= self.cap {
            // A single chunk larger than the cap: the ring IS this chunk's
            // tail (older bytes are gone — the cap is a hard bound).
            let keep = &chunk[chunk.len() - self.cap..];
            self.bytes.clear();
            self.bytes.extend(keep);
            return;
        }
        let over = self
            .bytes
            .len()
            .saturating_add(chunk.len())
            .saturating_sub(self.cap);
        if over > 0 {
            // `over < self.bytes.len()`: chunk.len() < cap by the branch above.
            self.bytes.drain(..over);
        }
        self.bytes.extend(chunk);
    }

    fn tail_lossy(&self) -> String {
        String::from_utf8_lossy(&self.bytes.iter().copied().collect::<Vec<u8>>()).into_owned()
    }
}

/// Minimal file:// URI from an absolute path: unreserved RFC 3986
/// characters plus `/` and `:` pass through; everything else (spaces,
/// non-ASCII, ...) is percent-encoded per byte.
fn file_uri(path: &Path) -> String {
    let raw = path.to_string_lossy();
    let mut out = String::with_capacity(raw.len() + 8);
    out.push_str("file://");
    for b in raw.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' | b'/' | b':' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Idle bookkeeping hook: the manager installs it so every request AND
/// notification touches the workspace's last-used stamp.
type Activity = Arc<dyn Fn() + Send + Sync>;

pub struct LspClient {
    conn: Arc<Mutex<LspConn>>,
    supervisor: Arc<ProcessSupervisor>,
    /// Set when the server's stdout reached EOF (the server exited); new
    /// requests then fail fast instead of hanging until a deadline.
    exited: Arc<AtomicBool>,
    stderr: Arc<Mutex<StderrRing>>,
    activity: Option<Activity>,
}

impl LspClient {
    async fn connect(
        cfg: &LspConfig,
        workspace: WorkspaceId,
        supervisor: Arc<ProcessSupervisor>,
        activity: Option<Activity>,
    ) -> Result<Arc<Self>, Error> {
        let proc_cfg = SpawnConfig {
            cmd: cfg.command.clone(),
            args: cfg.args.clone(),
            cwd: cfg.root.clone(),
            env: vec![("PATH".into(), std::env::var("PATH").unwrap_or_default())],
            // The REAL workspace owns the daemon: never a placeholder id.
            owner: ProcessOwner::Workspace(workspace),
            capture: false,
            ..Default::default()
        };
        let spawned = supervisor
            .spawn_detached_with_pipes(proc_cfg)
            .map_err(|e| Error::new(ErrorKind::NotFound, format!("lsp spawn: {e}")))?;
        let conn = Arc::new(Mutex::new(LspConn {
            child_pid: spawned.child_pid,
            stdin: spawned.stdin,
            next_id: 1,
            pending: HashMap::new(),
        }));
        let exited = Arc::new(AtomicBool::new(false));
        // Reader thread (stdout): incremental Content-Length framing;
        // responses are dispatched by id. EOF marks the server exited and
        // fails every pending request.
        {
            let conn2 = conn.clone();
            let exited2 = exited.clone();
            std::thread::spawn(move || read_loop(conn2, exited2, spawned.stdout));
        }
        // Stderr drain thread: a verbose server must never block itself on a
        // full stderr pipe; the bounded ring keeps the recent tail.
        let stderr = Arc::new(Mutex::new(StderrRing::new(STDERR_RING_CAP)));
        {
            let ring = stderr.clone();
            std::thread::spawn(move || stderr_drain(spawned.stderr, ring));
        }
        Ok(Arc::new(Self {
            conn,
            supervisor,
            exited,
            stderr,
            activity,
        }))
    }

    /// REQUEST path: carries an id, registers a pending entry, awaits the
    /// response (bounded by `deadline`). A server that never answers is a
    /// timeout — never a hang.
    async fn request(
        &self,
        method: &str,
        params: serde_json::Value,
        deadline: Duration,
    ) -> Result<serde_json::Value, Error> {
        if self.exited.load(Ordering::SeqCst) {
            return Err(Error::new(
                ErrorKind::Network,
                format!("lsp server exited before {method}"),
            ));
        }
        let (id, request) = {
            let mut conn = self.conn.lock().unwrap();
            if conn.pending.len() >= MAX_INFLIGHT_REQUESTS {
                return Err(Error::new(
                    ErrorKind::Oversized,
                    "too many in-flight lsp requests",
                ));
            }
            let id = conn.next_id;
            conn.next_id += 1;
            let request = serde_json::json!({
                "jsonrpc": "2.0",
                "id": id,
                "method": method,
                "params": params,
            });
            (id.to_string(), request)
        };
        let (tx, rx) = tokio::sync::oneshot::channel();
        {
            let mut conn = self.conn.lock().unwrap();
            let wire = format!(
                "Content-Length: {}\r\n\r\n{}",
                request.to_string().len(),
                request
            );
            conn.stdin
                .write_all(wire.as_bytes())
                .map_err(|e| Error::new(ErrorKind::Network, format!("lsp write: {e}")))?;
            conn.stdin.flush().ok();
            conn.pending.insert(id.clone(), tx);
        }
        if let Some(a) = &self.activity {
            a();
        }
        match tokio::time::timeout(deadline, rx).await {
            Ok(Ok(v)) => {
                if let Some(err) = v.get("error") {
                    return Err(Error::new(
                        ErrorKind::Provider {
                            code: "lsp".into(),
                            retryable: false,
                        },
                        format!("lsp {method}: {err}"),
                    ));
                }
                Ok(v.get("result").cloned().unwrap_or(serde_json::Value::Null))
            }
            Ok(Err(_)) => Err(Error::new(ErrorKind::Network, "lsp connection dropped")),
            Err(_) => {
                self.conn.lock().unwrap().pending.remove(&id);
                Err(Error::timeout(format!(
                    "lsp {method} exceeded {}ms",
                    deadline.as_millis()
                )))
            }
        }
    }

    /// NOTIFICATION path: fire-and-forget. No id, no pending entry, no
    /// response wait — the LSP server is NOT supposed to answer a
    /// notification, and the client must not act as if one were coming.
    /// Notifications to an already-exited server are dropped silently
    /// (their delivery is best-effort by design); a live-server write
    /// failure surfaces loudly.
    fn notify(&self, method: &str, params: serde_json::Value) -> Result<(), Error> {
        if self.exited.load(Ordering::SeqCst) {
            return Ok(());
        }
        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        {
            let mut conn = self.conn.lock().unwrap();
            let wire = format!(
                "Content-Length: {}\r\n\r\n{}",
                notification.to_string().len(),
                notification
            );
            if let Err(e) = conn.stdin.write_all(wire.as_bytes()) {
                if self.exited.load(Ordering::SeqCst) {
                    return Ok(()); // the server died mid-write; nothing to notify
                }
                return Err(Error::new(
                    ErrorKind::Network,
                    format!("lsp notify {method}: {e}"),
                ));
            }
            conn.stdin.flush().ok();
        }
        if let Some(a) = &self.activity {
            a();
        }
        Ok(())
    }

    /// `initialize` REQUEST with the REAL workspace root as `rootUri`
    /// (spec §3.15: servers resolve project roots from this URI).
    pub async fn initialize(&self, root: &Path) -> Result<serde_json::Value, Error> {
        self.request(
            "initialize",
            serde_json::json!({
                "processId": null,
                "rootUri": file_uri(root),
                "capabilities": {},
            }),
            Duration::from_secs(10),
        )
        .await
    }

    /// `initialized` NOTIFICATION — must arrive right after the initialize
    /// RESULT and before any `didOpen` (spec §3.15).
    pub fn notify_initialized(&self) -> Result<(), Error> {
        self.notify("initialized", serde_json::json!({}))
    }

    /// `textDocument/didOpen` NOTIFICATION: the server must never answer
    /// it, so this returns as soon as the frame is written — no response
    /// wait, no timeout.
    pub fn did_open(&self, uri: &str, text: &str) -> Result<(), Error> {
        self.notify(
            "textDocument/didOpen",
            serde_json::json!({
                "textDocument": {
                    "uri": uri,
                    "languageId": language_of(uri),
                    "version": 1,
                    "text": text,
                }
            }),
        )
    }

    pub async fn document_symbols(&self, uri: &str) -> Result<Vec<serde_json::Value>, Error> {
        let result = self
            .request(
                "textDocument/documentSymbol",
                serde_json::json!({ "textDocument": { "uri": uri } }),
                Duration::from_secs(10),
            )
            .await?;
        Ok(result.as_array().cloned().unwrap_or_default())
    }

    /// Graceful LSP lifecycle: `shutdown` REQUEST (response awaited),
    /// `exit` NOTIFICATION, then a bounded grace wait for the process; the
    /// group kill is the fallback if the server does not exit on its own.
    pub async fn shutdown(&self) -> Result<(), Error> {
        // (1) shutdown REQUEST: await the server's response (bounded). A
        // failing/absent response is tolerated: exit+kill still proceed.
        let _ = self
            .request(
                "shutdown",
                serde_json::Value::Null,
                Duration::from_millis(SHUTDOWN_REQUEST_MS),
            )
            .await;
        // (2) exit NOTIFICATION: fire-and-forget.
        let _ = self.notify("exit", serde_json::Value::Null);
        // (3) bounded grace: poll for the process to exit; only then kill.
        let pid = self.conn.lock().map(|c| c.child_pid).unwrap_or(0);
        if pid != 0 && self.supervisor.pid_alive(pid) {
            let deadline = tokio::time::Instant::now() + Duration::from_millis(EXIT_GRACE_MS);
            while tokio::time::Instant::now() < deadline && self.supervisor.pid_alive(pid) {
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            if self.supervisor.pid_alive(pid) {
                tracing::warn!(
                    "lsp server {pid} did not exit within {}ms of exit; killing",
                    EXIT_GRACE_MS
                );
                let _ = self.supervisor.kill_child_pid(pid, 500);
            }
        }
        self.exited.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// Recent stderr tail (bounded; lossy UTF-8) for diagnostics.
    pub fn stderr_tail(&self) -> String {
        self.stderr.lock().unwrap().tail_lossy()
    }

    /// Total stderr bytes drained (never bounded) — proves a flood was
    /// actually drained and never blocked the server.
    pub fn stderr_total_bytes(&self) -> u64 {
        self.stderr.lock().unwrap().total
    }
}

impl Drop for LspClient {
    fn drop(&mut self) {
        self.exited.store(true, Ordering::SeqCst);
        let pid = self.conn.lock().map(|c| c.child_pid).unwrap_or(0);
        if pid != 0 {
            let _ = self.supervisor.kill_child_pid(pid, 200);
        }
    }
}

fn language_of(uri: &str) -> String {
    let ext = uri.rsplit('.').next().unwrap_or("");
    match ext {
        "rs" => "rust".into(),
        "py" => "python".into(),
        "ts" | "tsx" => "typescript".into(),
        "js" | "jsx" => "javascript".into(),
        "go" => "go".into(),
        _ => "plaintext".into(),
    }
}

/// Workspace-scoped registry: servers are shared per workspace and unloaded
/// on idle. `last_used` is the single idle clock: it is touched when a
/// server is STARTED and on EVERY request/notification after that (the
/// manager installs the touch hook on each client), so an actively-used old
/// server is never unloaded while a merely-created one is.
pub struct LspManager {
    supervisor: Arc<ProcessSupervisor>,
    clients: Mutex<HashMap<WorkspaceId, Arc<LspClient>>>,
    last_used: Arc<Mutex<HashMap<WorkspaceId, i64>>>,
}

impl LspManager {
    pub fn new(supervisor: Arc<ProcessSupervisor>) -> Self {
        Self {
            supervisor,
            clients: Mutex::new(HashMap::new()),
            last_used: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// The touch hook installed on every client: each request/notification
    /// refreshes this workspace's idle stamp through the manager's own
    /// bookkeeping map.
    fn activity_hook(&self, workspace: WorkspaceId) -> Activity {
        let last_used = self.last_used.clone();
        Arc::new(move || {
            if let Ok(mut m) = last_used.lock() {
                m.insert(workspace, now_ms());
            }
        })
    }

    pub async fn start(
        &self,
        workspace: WorkspaceId,
        cfg: LspConfig,
    ) -> Result<Arc<LspClient>, Error> {
        {
            let clients = self.clients.lock().unwrap();
            if let Some(c) = clients.get(&workspace) {
                // Reuse IS use: refresh the idle stamp.
                self.touch(workspace);
                return Ok(c.clone());
            }
        }
        let client = LspClient::connect(
            &cfg,
            workspace,
            self.supervisor.clone(),
            Some(self.activity_hook(workspace)),
        )
        .await?;
        // Handshake: initialize REQUEST (awaited) → initialized NOTIFICATION
        // (fire-and-forget, before any didOpen).
        client.initialize(&cfg.root).await?;
        client.notify_initialized()?;
        self.clients
            .lock()
            .unwrap()
            .insert(workspace, client.clone());
        self.touch(workspace);
        Ok(client)
    }

    pub async fn client(&self, workspace: WorkspaceId) -> Result<Arc<LspClient>, Error> {
        let clients = self.clients.lock().unwrap();
        clients
            .get(&workspace)
            .cloned()
            .ok_or_else(|| Error::not_found(format!("no LSP for workspace {workspace}")))
    }

    /// Graceful teardown: shutdown request → exit notification → bounded
    /// exit wait (kill only as the fallback).
    pub async fn shutdown(&self, workspace: WorkspaceId) -> Result<(), Error> {
        let client = self.clients.lock().unwrap().remove(&workspace);
        if let Some(c) = client {
            c.shutdown().await?;
        }
        self.last_used.lock().unwrap().remove(&workspace);
        Ok(())
    }

    pub fn active(&self) -> Vec<WorkspaceId> {
        let mut v: Vec<WorkspaceId> = self.clients.lock().unwrap().keys().copied().collect();
        v.sort_by_key(|w| w.raw());
        v
    }

    fn touch(&self, workspace: WorkspaceId) {
        if let Ok(mut m) = self.last_used.lock() {
            m.insert(workspace, now_ms());
        }
    }

    /// Unload servers idle for longer than `idle_ms` (spec §21/§32).
    /// Synchronous by contract (daemon sweep): idle servers are killed
    /// directly — the graceful shutdown request/exit dance is the
    /// interactive teardown path, not the idle sweep's.
    pub fn unload_idle(&self, idle_ms: i64) -> Vec<WorkspaceId> {
        let now = now_ms();
        let stale: Vec<WorkspaceId> = self
            .last_used
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, t)| now - **t > idle_ms)
            .map(|(w, _)| *w)
            .collect();
        for w in &stale {
            if let Some(c) = self.clients.lock().unwrap().remove(w) {
                let pid = c.conn.lock().map(|c| c.child_pid).unwrap_or(0);
                if pid != 0 {
                    let _ = self.supervisor.kill_child_pid(pid, 200);
                }
            }
            self.last_used.lock().unwrap().remove(w);
        }
        stale
    }
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Stdout reader (own thread): incremental Content-Length framing; responses
/// are dispatched by id; EOF fails every pending request and marks the
/// server exited. The thread owns stdout, so blocking reads can never stall
/// the async runtime.
fn read_loop(
    conn: Arc<Mutex<LspConn>>,
    exited: Arc<AtomicBool>,
    stdout: std::process::ChildStdout,
) {
    let mut reader = BufReader::new(stdout);
    let mut buf: Vec<u8> = Vec::with_capacity(4096);
    use std::io::Read;
    loop {
        let mut chunk = [0u8; 8192];
        let n = match reader.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => n,
            Err(_) => break,
        };
        buf.extend_from_slice(&chunk[..n]);
        if buf.len() > MAX_RESPONSE_BYTES {
            break;
        }
        loop {
            match faktor_mcp::parse_frame(&buf) {
                Ok(Some((consumed, value))) => {
                    buf.drain(..consumed);
                    let id = value
                        .get("id")
                        .and_then(|i| i.as_u64())
                        .map(|i| i.to_string());
                    if let Some(id) = id {
                        let mut guard = conn.lock().unwrap();
                        if let Some(tx) = guard.pending.remove(&id) {
                            let _ = tx.send(value);
                        }
                    }
                }
                Ok(None) => break,
                Err(_) => {
                    let mut guard = conn.lock().unwrap();
                    for (_, tx) in guard.pending.drain() {
                        let _ = tx.send(serde_json::json!({"error": {"code": -32700, "message": "parse error"}}));
                    }
                    exited.store(true, Ordering::SeqCst);
                    return;
                }
            }
        }
    }
    exited.store(true, Ordering::SeqCst);
    let mut guard = conn.lock().unwrap();
    for (_, tx) in guard.pending.drain() {
        let _ = tx.send(serde_json::json!({"error": {"code": -32000, "message": "server closed"}}));
    }
}

/// Stderr drain (own thread): read everything into the bounded ring — an
/// unread stderr pipe would let a verbose server block itself on a full
/// pipe and stall every request.
fn stderr_drain(mut stderr: std::process::ChildStderr, ring: Arc<Mutex<StderrRing>>) {
    use std::io::Read;
    let mut buf = [0u8; 8192];
    loop {
        match stderr.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => ring.lock().unwrap().push(&buf[..n]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    const MOCK: &str = r#"
import json, os, sys, threading

def send(obj):
    body = json.dumps(obj).encode("utf-8")
    sys.stdout.buffer.write(b"Content-Length: %d\r\n\r\n" % len(body))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()

def read_msg():
    cl = 0
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        if line.lower().startswith(b"content-length:"):
            cl = int(line.split(b":")[1])
    if cl == 0:
        return {}
    return json.loads(sys.stdin.buffer.read(cl))

def log(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

mode = sys.argv[1]
expected_root = sys.argv[2]
expected_uri = sys.argv[3]

if mode == "flood":
    def flood_loop():
        chunk = b"z" * 8192
        while True:
            sys.stderr.buffer.write(chunk)
            sys.stderr.buffer.flush()
    threading.Thread(target=flood_loop, daemon=True).start()

initialized_seen = False
didopen_seen = False

def check(cond, what):
    log(("OK " if cond else "FAIL ") + what)

while True:
    msg = read_msg()
    if msg is None:
        break
    if "method" not in msg:
        continue
    m = msg["method"]
    if m == "initialize":
        p = msg.get("params", {})
        check(p.get("rootUri") == expected_uri, "initialize rootUri")
        check(os.path.realpath(os.getcwd()) == os.path.realpath(expected_root), "initialize cwd")
        send({"jsonrpc": "2.0", "id": msg["id"], "result": {
            "capabilities": {},
            "serverInfo": {"name": "mock-lsp", "version": "1"},
        }})
        log("SENT initialize result")
    elif m == "initialized":
        initialized_seen = True
        log("GOT initialized")
    elif m == "textDocument/didOpen":
        didopen_seen = True
        check(initialized_seen, "didOpen after initialized")
        log("GOT didOpen")
    elif m == "textDocument/documentSymbol":
        send({"jsonrpc": "2.0", "id": msg["id"], "result": []})
    elif m == "shutdown":
        send({"jsonrpc": "2.0", "id": msg["id"], "result": None})
        log("SENT shutdown result")
    elif m == "exit":
        log("GOT exit")
        break
log("mock exiting")
"#;

    fn python_available() -> bool {
        std::process::Command::new("python3")
            .arg("--version")
            .output()
            .is_ok()
    }

    /// A real language-server stand-in (written to a temp file at test
    /// time): speaks Content-Length JSON-RPC on stdio, checks the handshake
    /// contract on behalf of the "server side" and reports via stderr
    /// markers that the client drains into its bounded ring.
    fn mock_server_script() -> std::path::PathBuf {
        let dir = tempdir().unwrap();
        let path = dir.path().join("lsp_mock.py");
        std::fs::write(&path, MOCK).unwrap();
        std::mem::forget(dir); // the temp file lives for the test process
        path
    }

    fn supervisor() -> (tempfile::TempDir, Arc<ProcessSupervisor>) {
        let dir = tempdir().unwrap();
        let cas = Arc::new(faktor_cas::Cas::open(dir.path().join("cas")).unwrap());
        (dir, ProcessSupervisor::new(cas))
    }

    fn cfg_with(root: PathBuf, mode: &str, expected_uri: &str) -> (LspConfig, String) {
        let script = mock_server_script();
        (
            LspConfig {
                name: "mock".into(),
                command: "python3".into(),
                args: vec![
                    script.to_str().unwrap().into(),
                    mode.into(),
                    root.to_str().unwrap().into(),
                    expected_uri.into(),
                ],
                root,
            },
            script.to_str().unwrap().to_string(),
        )
    }

    /// Poll the client's stderr ring until `needle` appears (markers are
    /// written by the mock and drained asynchronously).
    async fn wait_for_stderr(client: &LspClient, needle: &str) -> String {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            let tail = client.stderr_tail();
            if tail.contains(needle) || std::time::Instant::now() > deadline {
                return tail;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    #[test]
    fn language_of_mapping() {
        assert_eq!(language_of("file:///x/main.rs"), "rust");
        assert_eq!(language_of("file:///x/a.py"), "python");
        assert_eq!(language_of("file:///x/a.ts"), "typescript");
        assert_eq!(language_of("file:///x/a.go"), "go");
        assert_eq!(language_of("file:///x/a.xyz"), "plaintext");
    }

    #[test]
    fn file_uri_encodes_and_roundtrips() {
        assert_eq!(file_uri(Path::new("/a/b/c.rs")), "file:///a/b/c.rs");
        assert_eq!(file_uri(Path::new("/a b/ü.rs")), "file:///a%20b/%C3%BC.rs");
    }

    #[test]
    fn stderr_ring_caps_bytes_and_keeps_the_tail() {
        let mut ring = StderrRing::new(16);
        ring.push(b"abcdefghijklmnop"); // exactly the cap
        assert_eq!(ring.bytes.len(), 16);
        ring.push(b"q"); // one byte over
        assert_eq!(ring.bytes.len(), 16, "hard byte cap");
        assert_eq!(
            ring.tail_lossy(),
            "bcdefghijklmnopq",
            "head dropped, tail kept"
        );
        assert_eq!(ring.total, 17, "total is never bounded");
        // A chunk bigger than the whole ring replaces it by its own tail.
        ring.push(&b"x".repeat(64));
        assert_eq!(ring.bytes.len(), 16);
        assert_eq!(ring.total, 81);
        assert_eq!(ring.tail_lossy(), "x".repeat(16));
    }

    #[tokio::test]
    async fn missing_server_is_not_found_and_manager_survives() {
        let dir = tempdir().unwrap();
        let cas = Arc::new(faktor_cas::Cas::open(dir.path().join("cas")).unwrap());
        let sup = ProcessSupervisor::new(cas);
        let mgr = LspManager::new(sup);
        let result = mgr
            .start(
                WorkspaceId::new(1),
                LspConfig {
                    name: "ghost".into(),
                    command: "/nonexistent-lsp".into(),
                    args: vec![],
                    root: dir.path().to_path_buf(),
                },
            )
            .await;
        // connect() returns Err(NotFound) OR times out (either is a clean
        // failure — never a panic, never a zombie).
        if let Err(e) = result {
            assert!(e.kind == ErrorKind::NotFound, "{e:?}");
        }
        // The manager remains usable.
        assert!(mgr.active().is_empty());
        assert!(mgr.client(WorkspaceId::new(1)).await.is_err());
    }

    #[tokio::test]
    async fn idle_unload_removes_clients() {
        let dir = tempdir().unwrap();
        let cas = Arc::new(faktor_cas::Cas::open(dir.path().join("cas")).unwrap());
        let sup = ProcessSupervisor::new(cas);
        let mgr = LspManager::new(sup);
        // No server started → idle unload is a no-op.
        assert!(mgr.unload_idle(1).is_empty());
        assert!(mgr.active().is_empty());
    }

    #[test]
    fn framing_reuse() {
        // The LSP crate reuses faktor-mcp's framing; verify a valid frame.
        let body = serde_json::json!({"jsonrpc": "2.0", "id": 1, "result": null}).to_string();
        let frame = format!("Content-Length: {}\r\n\r\n{}", body.len(), body);
        let (consumed, v) = faktor_mcp::parse_frame(frame.as_bytes()).unwrap().unwrap();
        assert_eq!(consumed, frame.len());
        assert_eq!(v["id"], 1);
    }

    #[test]
    fn hostile_frames_never_panic() {
        for garbage in [
            b"".as_slice(),
            b"\x00\x01".as_slice(),
            b"Content-Length: x\r\n\r\n".as_slice(),
            b"Content-Length: 5\r\n\r\n".as_slice(),
        ] {
            let _ = faktor_mcp::parse_frame(garbage);
        }
    }

    /// (i) connect sends initialize with the REAL rootUri and cwd, receives
    /// the result, sends the initialized NOTIFICATION (server asserts it
    /// receives initialized before any didOpen); the process is owned by the
    /// REAL workspace id.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn initialize_carries_real_root_and_owner() {
        if !python_available() {
            eprintln!("python3 missing; skipping");
            return;
        }
        let root_dir = tempdir().unwrap();
        let root = root_dir.path().to_path_buf();
        let expected_uri = file_uri(&root);
        let (_d, sup) = supervisor();
        let mgr = LspManager::new(sup.clone());
        let real_ws = WorkspaceId::new(42); // never 1: the owner must be real
        let (cfg, _script) = cfg_with(root.clone(), "plain", &expected_uri);
        let client = tokio::time::timeout(Duration::from_secs(15), mgr.start(real_ws, cfg))
            .await
            .expect("start timeout")
            .expect("start failed");

        // The spawned daemon is owned by the REAL workspace id.
        let alive = sup.alive();
        assert!(
            alive
                .iter()
                .any(|h| h.owner == ProcessOwner::Workspace(real_ws)),
            "server must be owned by the real workspace: {alive:?}"
        );
        assert!(
            !alive
                .iter()
                .any(|h| h.owner == ProcessOwner::Workspace(WorkspaceId::new(1))),
            "never a placeholder workspace-1 owner"
        );

        // didOpen is a notification; the server-side assertions (rootUri,
        // cwd, initialized-before-didOpen) surface as stderr markers.
        client
            .did_open(&format!("{expected_uri}/main.rs"), "fn main() {}")
            .unwrap();
        let symbols = tokio::time::timeout(
            Duration::from_secs(10),
            client.document_symbols(&format!("{expected_uri}/main.rs")),
        )
        .await
        .expect("documentSymbol timeout")
        .expect("documentSymbol failed");
        assert!(symbols.is_empty());

        let tail = wait_for_stderr(&client, "mock exiting").await;
        assert!(
            tail.contains("OK initialize rootUri"),
            "rootUri must be the REAL workspace root: {tail}"
        );
        assert!(
            tail.contains("OK initialize cwd"),
            "cwd must be the REAL workspace root: {tail}"
        );
        assert!(
            tail.contains("OK didOpen after initialized"),
            "initialized must precede didOpen: {tail}"
        );
        mgr.shutdown(real_ws).await.unwrap();
    }

    /// (ii) didOpen is a NOTIFICATION: the stand-in sends NO response and
    /// the client must NOT time out or error (the old request path waited
    /// for a response that is not coming by design).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn did_open_notification_never_waits_for_a_response() {
        if !python_available() {
            eprintln!("python3 missing; skipping");
            return;
        }
        let root_dir = tempdir().unwrap();
        let root = root_dir.path().to_path_buf();
        let expected_uri = file_uri(&root);
        let (_d, sup) = supervisor();
        let mgr = LspManager::new(sup);
        let (cfg, _script) = cfg_with(root.clone(), "plain", &expected_uri);
        let client =
            tokio::time::timeout(Duration::from_secs(15), mgr.start(WorkspaceId::new(9), cfg))
                .await
                .expect("start timeout")
                .expect("start failed");
        let t0 = std::time::Instant::now();
        client
            .did_open(&format!("{expected_uri}/a.rs"), "fn a() {}")
            .unwrap();
        let elapsed = t0.elapsed();
        assert!(
            elapsed < Duration::from_secs(2),
            "a notification must return immediately, took {elapsed:?}"
        );
        // The server processed it (no response was ever expected).
        let tail = wait_for_stderr(&client, "GOT didOpen").await;
        assert!(!tail.contains("FAIL didOpen after initialized"), "{tail}");
        mgr.shutdown(WorkspaceId::new(9)).await.unwrap();
    }

    /// (iii) shutdown REQUEST receives a response, then the exit
    /// notification arrives, then the process exits within the grace —
    /// kill is never the first move.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shutdown_request_then_exit_notification_within_grace() {
        if !python_available() {
            eprintln!("python3 missing; skipping");
            return;
        }
        let root_dir = tempdir().unwrap();
        let root = root_dir.path().to_path_buf();
        let expected_uri = file_uri(&root);
        let (_d, sup) = supervisor();
        let mgr = LspManager::new(sup.clone());
        let ws = WorkspaceId::new(13);
        let (cfg, _script) = cfg_with(root.clone(), "plain", &expected_uri);
        let client = tokio::time::timeout(Duration::from_secs(15), mgr.start(ws, cfg))
            .await
            .expect("start timeout")
            .expect("start failed");
        let pid = client.conn.lock().unwrap().child_pid;
        assert!(sup.pid_alive(pid), "server must be alive before shutdown");
        let t0 = std::time::Instant::now();
        mgr.shutdown(ws).await.unwrap();
        let elapsed = t0.elapsed();
        assert!(
            elapsed < Duration::from_millis(EXIT_GRACE_MS + 2_000),
            "process must exit within the bounded grace: {elapsed:?}"
        );
        assert!(!sup.pid_alive(pid), "server must have exited on its own");
        // The mock answered shutdown and only then received exit — proving
        // the request/response/notification ordering, not a blind kill.
        let tail = wait_for_stderr(&client, "GOT exit").await;
        assert!(
            tail.contains("SENT shutdown result"),
            "shutdown response must be awaited: {tail}"
        );
        assert!(
            tail.contains("GOT exit"),
            "exit notification must arrive: {tail}"
        );
        assert!(
            tail.contains("mock exiting"),
            "process exits after exit: {tail}"
        );
        assert!(mgr.active().is_empty());
    }

    /// (iv) a stand-in flooding stderr with megabytes keeps running (the
    /// bounded ring drains the pipe — no deadlock) and the client still
    /// answers requests.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stderr_flood_is_drained_bounded_and_never_blocks() {
        if !python_available() {
            eprintln!("python3 missing; skipping");
            return;
        }
        let root_dir = tempdir().unwrap();
        let root = root_dir.path().to_path_buf();
        let expected_uri = file_uri(&root);
        let (_d, sup) = supervisor();
        let mgr = LspManager::new(sup);
        let ws = WorkspaceId::new(21);
        let (cfg, _script) = cfg_with(root.clone(), "flood", &expected_uri);
        let client = tokio::time::timeout(Duration::from_secs(15), mgr.start(ws, cfg))
            .await
            .expect("start timeout")
            .expect("start failed");
        // Let the flood run; the client must keep answering requests.
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while client.stderr_total_bytes() < 2 * 1024 * 1024 && std::time::Instant::now() < deadline
        {
            let symbols = tokio::time::timeout(
                Duration::from_secs(5),
                client.document_symbols("file:///x.rs"),
            )
            .await
            .expect("request timeout while flooding")
            .expect("request failed while flooding");
            assert!(symbols.is_empty());
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let total = client.stderr_total_bytes();
        assert!(
            total >= 2 * 1024 * 1024,
            "stderr must actually be drained, only {total} bytes"
        );
        let tail = client.stderr_tail();
        assert!(
            tail.len() <= STDERR_RING_CAP,
            "retained tail must be bounded: {}",
            tail.len()
        );
        mgr.shutdown(ws).await.unwrap();
    }

    /// (v) idle: a server created and left idle is unloaded after the
    /// manager's idle period; an OLD server that was actively used right
    /// before the sweep is NOT unloaded (touch-on-use proven).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn idle_unload_ignores_recently_used_old_servers() {
        if !python_available() {
            eprintln!("python3 missing; skipping");
            return;
        }
        let idle_ms = 250i64;
        let root_dir = tempdir().unwrap();
        let root = root_dir.path().to_path_buf();
        let expected_uri = file_uri(&root);
        let (_d, sup) = supervisor();
        let mgr = LspManager::new(sup.clone());
        let idle_ws = WorkspaceId::new(30);
        let (cfg_idle, _s) = cfg_with(root.clone(), "plain", &expected_uri);
        let _idle_client =
            tokio::time::timeout(Duration::from_secs(15), mgr.start(idle_ws, cfg_idle))
                .await
                .expect("start timeout")
                .expect("start failed");
        // Let the idle server age past the idle period...
        tokio::time::sleep(Duration::from_millis((idle_ms + 150) as u64)).await;
        // ...then start a second server and let IT age too.
        let used_ws = WorkspaceId::new(31);
        let (cfg_used, _s2) = cfg_with(root.clone(), "plain", &expected_uri);
        let used_client =
            tokio::time::timeout(Duration::from_secs(15), mgr.start(used_ws, cfg_used))
                .await
                .expect("start timeout")
                .expect("start failed");
        tokio::time::sleep(Duration::from_millis((idle_ms + 150) as u64)).await;
        // BOTH creation stamps are now stale. Actively use the old server:
        // the request must refresh its idle stamp.
        used_client
            .document_symbols("file:///x.rs")
            .await
            .expect("request on the old-but-used server failed");
        let unloaded = mgr.unload_idle(idle_ms);
        assert!(
            unloaded.contains(&idle_ws),
            "created-and-idle server must be unloaded: {unloaded:?}"
        );
        assert!(
            !unloaded.contains(&used_ws),
            "an old server used right before the sweep must survive: {unloaded:?}"
        );
        assert_eq!(mgr.active(), vec![used_ws]);
        // The idle server's process is gone; the used one's is alive.
        let alive_pids: Vec<u32> = sup.alive().iter().map(|h| h.pid).collect();
        assert_eq!(
            alive_pids.len(),
            1,
            "only the used server survives: {alive_pids:?}"
        );
        mgr.shutdown(used_ws).await.unwrap();
        assert!(mgr.active().is_empty());
    }
}
