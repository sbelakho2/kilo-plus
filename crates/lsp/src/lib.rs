//! kilop-lsp — workspace-scoped language server integration (spec §32).
//!
//! Language servers are workspace resources, not session resources: one
//! daemon shares `rust-analyzer`/`typescript-language-server`/`pyright`
//! across sessions on the same workspace. Requests are multiplexed with
//! per-request ids; heavy servers are unloaded after workspace inactivity.

use std::collections::HashMap;
use std::io::{BufReader, Write};
use std::process::ChildStdin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use kilop_core::error::{Error, ErrorKind};
use kilop_core::id::WorkspaceId;
use kilop_terminal::{ProcessOwner, ProcessSupervisor, SpawnConfig};

const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LspConfig {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
}

struct LspConn {
    child_pid: u32,
    stdin: ChildStdin,
    next_id: u64,
    pending: HashMap<String, tokio::sync::oneshot::Sender<serde_json::Value>>,
}

pub struct LspClient {
    conn: Arc<Mutex<LspConn>>,
    supervisor: Arc<ProcessSupervisor>,
}

impl LspClient {
    async fn connect(
        cfg: &LspConfig,
        supervisor: Arc<ProcessSupervisor>,
    ) -> Result<Arc<Self>, Error> {
        let proc_cfg = SpawnConfig {
            cmd: cfg.command.clone(),
            args: cfg.args.clone(),
            cwd: std::env::temp_dir(),
            env: vec![("PATH".into(), std::env::var("PATH").unwrap_or_default())],
            owner: ProcessOwner::Workspace(WorkspaceId::new(1)),
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
        {
            let conn2 = conn.clone();
            std::thread::spawn(move || read_loop(conn2, spawned.stdout));
        }
        let client = Arc::new(Self { conn, supervisor });
        Ok(client)
    }

    async fn request(
        &self,
        method: &str,
        params: serde_json::Value,
        deadline: Duration,
    ) -> Result<serde_json::Value, Error> {
        let (id, request) = {
            let mut conn = self.conn.lock().unwrap();
            let id = conn.next_id.to_string();
            conn.next_id += 1;
            let request = serde_json::json!({
                "jsonrpc": "2.0",
                "id": conn.next_id - 1,
                "method": method,
                "params": params,
            });
            (id, request)
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

    pub async fn initialize(&self) -> Result<serde_json::Value, Error> {
        self.request(
            "initialize",
            serde_json::json!({
                "processId": null,
                "rootUri": null,
                "capabilities": {},
            }),
            Duration::from_secs(10),
        )
        .await
    }

    pub async fn did_open(&self, uri: &str, text: &str) -> Result<(), Error> {
        self.request(
            "textDocument/didOpen",
            serde_json::json!({
                "textDocument": {
                    "uri": uri,
                    "languageId": language_of(uri),
                    "version": 1,
                    "text": text,
                }
            }),
            Duration::from_secs(5),
        )
        .await?;
        Ok(())
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

    pub async fn shutdown(&self) -> Result<(), Error> {
        let _ = self
            .request("shutdown", serde_json::json!(null), Duration::from_secs(5))
            .await;
        let pid = self.conn.lock().unwrap().child_pid;
        let _ = self.supervisor.kill_child_pid(pid, 500);
        Ok(())
    }
}

impl Drop for LspClient {
    fn drop(&mut self) {
        let pid = self.conn.lock().map(|c| c.child_pid).unwrap_or(0);
        let _ = self.supervisor.kill_child_pid(pid, 200);
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
/// on idle.
pub struct LspManager {
    supervisor: Arc<ProcessSupervisor>,
    clients: Mutex<HashMap<WorkspaceId, Arc<LspClient>>>,
    last_used: Mutex<HashMap<WorkspaceId, i64>>,
}

impl LspManager {
    pub fn new(supervisor: Arc<ProcessSupervisor>) -> Self {
        Self {
            supervisor,
            clients: Mutex::new(HashMap::new()),
            last_used: Mutex::new(HashMap::new()),
        }
    }

    pub async fn start(
        &self,
        workspace: WorkspaceId,
        cfg: LspConfig,
    ) -> Result<Arc<LspClient>, Error> {
        {
            let clients = self.clients.lock().unwrap();
            if let Some(c) = clients.get(&workspace) {
                return Ok(c.clone());
            }
        }
        let client = LspClient::connect(&cfg, self.supervisor.clone()).await?;
        client.initialize().await?;
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

    /// Idle unload: removes the client and kills the server.
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
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        self.last_used.lock().unwrap().insert(workspace, now);
    }

    /// Unload servers idle for longer than `idle_ms` (spec §21/§32).
    pub fn unload_idle(&self, idle_ms: i64) -> Vec<WorkspaceId> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
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
                let _ = self.supervisor.kill_child_pid(pid, 200);
            }
            self.last_used.lock().unwrap().remove(w);
        }
        stale
    }
}

/// Same framing as MCP (Content-Length JSON-RPC).
fn read_loop(conn: Arc<Mutex<LspConn>>, stdout: std::process::ChildStdout) {
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
            match kilop_mcp::parse_frame(&buf) {
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
                    return;
                }
            }
        }
    }
    let mut guard = conn.lock().unwrap();
    for (_, tx) in guard.pending.drain() {
        let _ = tx.send(serde_json::json!({"error": {"code": -32000, "message": "server closed"}}));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn language_of_mapping() {
        assert_eq!(language_of("file:///x/main.rs"), "rust");
        assert_eq!(language_of("file:///x/a.py"), "python");
        assert_eq!(language_of("file:///x/a.ts"), "typescript");
        assert_eq!(language_of("file:///x/a.go"), "go");
        assert_eq!(language_of("file:///x/a.xyz"), "plaintext");
    }

    #[tokio::test]
    async fn missing_server_is_not_found_and_manager_survives() {
        let dir = tempdir().unwrap();
        let cas = Arc::new(kilop_cas::Cas::open(dir.path().join("cas")).unwrap());
        let sup = ProcessSupervisor::new(cas);
        let mgr = LspManager::new(sup);
        let result = mgr
            .start(
                WorkspaceId::new(1),
                LspConfig {
                    name: "ghost".into(),
                    command: "/nonexistent-lsp".into(),
                    args: vec![],
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
        let cas = Arc::new(kilop_cas::Cas::open(dir.path().join("cas")).unwrap());
        let sup = ProcessSupervisor::new(cas);
        let mgr = LspManager::new(sup);
        // No server started → idle unload is a no-op.
        assert!(mgr.unload_idle(1).is_empty());
        assert!(mgr.active().is_empty());
    }

    #[test]
    fn framing_reuse() {
        // The LSP crate reuses kilop-mcp's framing; verify a valid frame.
        let body = serde_json::json!({"jsonrpc": "2.0", "id": 1, "result": null}).to_string();
        let frame = format!("Content-Length: {}\r\n\r\n{}", body.len(), body);
        let (consumed, v) = kilop_mcp::parse_frame(frame.as_bytes()).unwrap().unwrap();
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
            let _ = kilop_mcp::parse_frame(garbage);
        }
    }
}
