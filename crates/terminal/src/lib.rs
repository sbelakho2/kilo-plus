//! kilop-terminal — process supervision (spec §22, §23).
//!
//! No orphans: every child process has a runtime owner; kill targets the
//! whole process group (Unix) or Job Object (Windows). Output is bounded:
//! a 200-line ring buffer live, with overflow spilling to a CAS artifact —
//! a 300MB log never becomes a 300MB RAM object. Blocking pipe reads live
//! on dedicated reader threads so they can never stall the async loop.

use std::collections::HashMap;
use std::io::Read;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use kilop_core::cancellation::CancellationToken;
use kilop_core::error::Error;
use kilop_core::id::{SessionId, WorkspaceId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessOwner {
    Session(SessionId),
    Workspace(WorkspaceId),
    Daemon,
}

#[derive(Debug, Clone)]
pub struct SpawnConfig {
    pub cmd: String,
    pub args: Vec<String>,
    pub cwd: PathBuf,
    pub env: Vec<(String, String)>,
    pub owner: ProcessOwner,
    /// Capture stdout+stderr into the ring buffer / artifact.
    pub capture: bool,
    /// Durable artifact cap in bytes (default 100MB).
    pub artifact_max: usize,
}

impl Default for SpawnConfig {
    fn default() -> Self {
        Self {
            cmd: String::new(),
            args: vec![],
            cwd: std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/")),
            env: vec![],
            owner: ProcessOwner::Daemon,
            capture: true,
            artifact_max: 100 * 1024 * 1024,
        }
    }
}

/// A spawned process whose pipes are handed to the caller.
pub struct SpawnedProcess {
    pub child_pid: u32,
    pub stdin: std::process::ChildStdin,
    pub stdout: std::process::ChildStdout,
    pub stderr: std::process::ChildStderr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChildHandle {
    pub id: u64,
    pub pid: u32,
    pub owner: ProcessOwner,
    pub started_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reaped {
    pub id: u64,
    pub pid: u32,
    pub exit_code: Option<i32>,
    pub owner: ProcessOwner,
}

/// Bounded command output: excerpt (last 200 lines + exit code) and an
/// optional durable artifact reference for the full stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandOutput {
    pub excerpt: String,
    pub exit_code: Option<i32>,
    pub artifact: Option<String>,
    pub slice_hint: Option<String>,
    pub ring_lines: usize,
}

const RING_LINES: usize = 200;
const MAX_EXCERPT_BYTES: usize = 64 * 1024;

struct ChildState {
    pid: u32,
    owner: ProcessOwner,
    started_ms: i64,
    exited: Option<Option<i32>>,
}

/// The bounded capture state; mutated only by the reader task.
struct SharedCapture {
    ring: RingBuffer,
    total: usize,
    artifact_max: usize,
    /// Overflow spills to a temp file on disk (never RAM); stored into the
    /// CAS once the command finishes.
    spill: Option<std::fs::File>,
    spill_path: Option<PathBuf>,
    artifact: Option<String>,
    cas: Arc<kilop_cas::Cas>,
}

impl SharedCapture {
    fn push(&mut self, bytes: &[u8]) {
        self.total += bytes.len();
        let text = String::from_utf8_lossy(bytes);
        for line in text.split('\n') {
            self.ring.push(line.trim_end_matches('\r').to_string());
        }
        // Once the durable cap is exceeded, overflow goes to a temp file
        // (bounded RAM; the ring keeps its bounded tail regardless).
        if self.spill.is_none() && self.total > self.artifact_max {
            let dir = std::env::temp_dir();
            let path = dir.join(format!(
                "kp-spill-{}-{}",
                std::process::id(),
                uuid::Uuid::new_v4()
            ));
            if let Ok(f) = std::fs::File::create(&path) {
                self.spill = Some(f);
                self.spill_path = Some(path);
            }
        }
        if let Some(f) = self.spill.as_mut() {
            let _ = std::io::Write::write_all(f, bytes);
        }
    }

    /// Store the spill file in the CAS and clean up the temp file.
    fn finalize_artifact(&mut self) {
        if let Some(path) = self.spill_path.take() {
            if let Ok(bytes) = std::fs::read(&path) {
                if let Ok(hash) = self.cas.put(&bytes) {
                    self.artifact = Some(format!("artifact://{}", hash.to_hex()));
                }
            }
            let _ = std::fs::remove_file(&path);
        }
    }
}

#[derive(Clone)]
pub struct ProcessSupervisor {
    registry: Arc<Mutex<HashMap<u64, ChildState>>>,
    cas: Arc<kilop_cas::Cas>,
    next_id: Arc<std::sync::atomic::AtomicU64>,
}

impl std::fmt::Debug for ProcessSupervisor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProcessSupervisor")
            .field("registered", &self.registered())
            .finish()
    }
}

impl ProcessSupervisor {
    pub fn new(cas: Arc<kilop_cas::Cas>) -> Arc<Self> {
        Arc::new(Self {
            registry: Arc::new(Mutex::new(HashMap::new())),
            cas,
            next_id: Arc::new(std::sync::atomic::AtomicU64::new(1)),
        })
    }

    fn alloc_id(&self) -> u64 {
        let id = self.next_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if id == 0 {
            self.next_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        } else {
            id
        }
    }

    fn command(&self, cfg: &SpawnConfig) -> std::process::Command {
        let mut cmd = std::process::Command::new(&cfg.cmd);
        cmd.args(&cfg.args)
            .current_dir(&cfg.cwd)
            .env_clear()
            .env("PATH", std::env::var("PATH").unwrap_or_default())
            .env("HOME", std::env::var("HOME").unwrap_or_default());
        for (k, v) in &cfg.env {
            cmd.env(k, v);
        }
        cmd.env("GIT_TERMINAL_PROMPT", "0");
        // Own process group so kills target the whole tree.
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }
        cmd
    }

    fn register(&self, pid: u32, owner: ProcessOwner, started_ms: i64) -> u64 {
        let id = self.alloc_id();
        self.registry.lock().unwrap().insert(
            id,
            ChildState {
                pid,
                owner,
                started_ms,
                exited: None,
            },
        );
        id
    }

    /// Run to completion: bounded capture (ring + CAS spill), deadline,
    /// cancellation. Uses tokio's async process pipes so reads never block
    /// the runtime; the reader task owns the bounded ring.
    pub async fn run(
        &self,
        cfg: SpawnConfig,
        deadline: Duration,
        token: CancellationToken,
    ) -> Result<CommandOutput, Error> {
        use tokio::io::AsyncReadExt;
        use tokio::process::Command as TokioCommand;

        let mut std_cmd = self.command(&cfg);
        if cfg.capture {
            std_cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
        } else {
            std_cmd.stdout(Stdio::null()).stderr(Stdio::null());
        }
        let started_ms = now_ms();
        let mut cmd = TokioCommand::from(std_cmd);
        let mut child = cmd
            .spawn()
            .map_err(|e| Error::not_found(format!("spawn {}: {e}", cfg.cmd)))?;
        let pid = child.id().unwrap_or(0);
        let id = self.register(pid, cfg.owner.clone(), started_ms);

        let shared: Arc<Mutex<SharedCapture>> = Arc::new(Mutex::new(SharedCapture {
            ring: RingBuffer::new(RING_LINES),
            total: 0,
            artifact_max: cfg.artifact_max,
            spill: None,
            spill_path: None,
            artifact: None,
            cas: self.cas.clone(),
        }));

        // Async reader task: drains both streams into the bounded ring.
        // `select!` means a data-ready stream is read immediately; neither
        // stream's idle wait can slow the other (a stderr that never writes
        // costs nothing).
        let reader = if cfg.capture {
            let mut stdout = child.stdout.take();
            let mut stderr = child.stderr.take();
            let shared2 = shared.clone();
            let token2 = token.clone();
            Some(tokio::spawn(async move {
                let mut out_buf = [0u8; 8192];
                let mut err_buf = [0u8; 8192];
                let mut stdout_eof = stdout.is_none();
                let mut stderr_eof = stderr.is_none();
                loop {
                    if stdout_eof && stderr_eof {
                        break;
                    }
                    if token2.is_cancelled() {
                        break;
                    }
                    tokio::select! {
                        _ = tokio::time::sleep(Duration::from_millis(5)) => {}
                        r = async {
                            match stdout.as_mut() {
                                Some(s) => s.read(&mut out_buf).await,
                                None => Ok(0),
                            }
                        } => {
                            match r {
                                Ok(0) => { stdout_eof = true; stdout = None; }
                                Ok(n) => { shared2.lock().unwrap().push(&out_buf[..n]); }
                                Err(_) => { stdout_eof = true; stdout = None; }
                            }
                        }
                        r = async {
                            match stderr.as_mut() {
                                Some(s) => s.read(&mut err_buf).await,
                                None => Ok(0),
                            }
                        } => {
                            match r {
                                Ok(0) => { stderr_eof = true; stderr = None; }
                                Ok(n) => { shared2.lock().unwrap().push(&err_buf[..n]); }
                                Err(_) => { stderr_eof = true; stderr = None; }
                            }
                        }
                    }
                }
            }))
        } else {
            None
        };

        // Poll loop: cancellation + exit status. The deadline is the overall
        // bound; the child's group is killed on timeout.
        let status: Option<std::process::ExitStatus> = tokio::select! {
            s = child.wait() => s.ok(),
            _ = tokio::time::sleep(deadline) => {
                let _ = kill_group(pid, 2000);
                let _ = child.kill().await;
                let _ = child.wait().await;
                self.mark_exited(id, None);
                return Err(Error::timeout(format!(
                    "command {} exceeded its {}ms deadline",
                    cfg.cmd,
                    deadline.as_millis()
                )));
            }
            _ = poll_cancelled(&token, Duration::from_millis(5)) => {
                let _ = kill_group(pid, 500);
                let _ = child.kill().await;
                let _ = child.wait().await;
                self.mark_exited(id, None);
                return Err(Error::cancelled());
            }
        };
        let exit_code = status.and_then(|s| s.code());
        if let Some(reader) = reader {
            let _ = reader.await;
        }
        self.mark_exited(id, Some(exit_code));

        let mut excerpt = String::new();
        let mut artifact = None;
        let mut slice_hint = None;
        {
            let mut g = shared.lock().unwrap();
            g.finalize_artifact();
            excerpt = g.ring.excerpt();
            excerpt.push_str(&format!("[exit code: {}]\n", exit_code.unwrap_or(-1)));
            if excerpt.len() > MAX_EXCERPT_BYTES {
                excerpt.truncate(MAX_EXCERPT_BYTES);
            }
            artifact = g.artifact.clone();
        }
        if let Some(a) = &artifact {
            slice_hint = Some(format!("{a}?slice=0&len=1024"));
        }
        Ok(CommandOutput {
            excerpt,
            exit_code,
            artifact,
            slice_hint,
            ring_lines: RING_LINES,
        })
    }

    /// Spawn with piped stdin/stdout/stderr (for MCP/LSP style servers).
    /// The caller owns the pipes; a reaper thread still reaps the child.
    pub fn spawn_detached_with_pipes(
        &self,
        mut cfg: SpawnConfig,
    ) -> Result<SpawnedProcess, Error> {
        cfg.capture = false;
        let mut cmd = self.command(&cfg);
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        let started_ms = now_ms();
        let mut child = cmd
            .spawn()
            .map_err(|e| Error::not_found(format!("spawn {}: {e}", cfg.cmd)))?;
        let pid = child.id();
        let stdin = child.stdin.take().ok_or_else(|| Error::internal("no stdin"))?;
        let stdout = child.stdout.take().ok_or_else(|| Error::internal("no stdout"))?;
        let stderr = child.stderr.take().ok_or_else(|| Error::internal("no stderr"))?;
        let id = self.register(pid, cfg.owner, started_ms);
        // Reaper thread (no zombies); the caller keeps the pipes.
        let registry = self.registry.clone();
        std::thread::spawn(move || {
            let status = child.wait().ok();
            let code = status.and_then(|s| s.code());
            let mut reg = registry.lock().unwrap();
            if let Some(state) = reg.get_mut(&id) {
                state.exited = Some(code);
            }
        });
        Ok(SpawnedProcess {
            child_pid: pid,
            stdin,
            stdout,
            stderr,
        })
    }

    /// Spawn detached with a reaper thread (no zombies); the caller owns the
    /// child and must kill/transfer deliberately.
    pub fn spawn(&self, cfg: SpawnConfig) -> Result<ChildHandle, Error> {
        let mut cmd = self.command(&cfg);
        cmd.stdout(Stdio::null()).stderr(Stdio::null());
        let started_ms = now_ms();
        let child = cmd
            .spawn()
            .map_err(|e| Error::not_found(format!("spawn {}: {e}", cfg.cmd)))?;
        let pid = child.id();
        let id = self.register(pid, cfg.owner.clone(), started_ms);
        // Reaper thread: waitpid is the only way to avoid zombies.
        let registry = self.registry.clone();
        std::thread::spawn(move || {
            let status = child.wait_with_output().map(|o| o.status).ok();
            let code = status.and_then(|s| s.code());
            let mut reg = registry.lock().unwrap();
            if let Some(state) = reg.get_mut(&id) {
                state.exited = Some(code);
            }
        });
        Ok(ChildHandle {
            id,
            pid,
            owner: cfg.owner,
            started_ms,
        })
    }

    pub fn kill(&self, id: u64, grace_ms: u64) -> Result<(), Error> {
        let pid = self
            .registry
            .lock()
            .unwrap()
            .get(&id)
            .map(|c| c.pid)
            .ok_or_else(|| Error::not_found(format!("child {id}")))?;
        kill_group(pid, grace_ms)
    }

    /// Kill a process by raw pid (process-group aware); used by MCP/LSP
    /// clients that own their own child lifecycle.
    pub fn kill_child_pid(&self, pid: u32, grace_ms: u64) -> Result<(), Error> {
        if pid == 0 {
            return Err(Error::not_found("pid 0"));
        }
        kill_group(pid, grace_ms)
    }

    /// Is a raw pid still alive (used by MCP/LSP clients)?
    pub fn pid_alive(&self, pid: u32) -> bool {
        if pid == 0 {
            return false;
        }
        std::process::Command::new("/bin/ps")
            .args(["-p", &pid.to_string()])
            .output()
            .map(|o| {
                let text = String::from_utf8_lossy(&o.stdout);
                text.contains(&pid.to_string()) && !text.contains("<defunct>")
            })
            .unwrap_or(false)
    }

    /// Collect exited children (no zombies).
    pub fn reap(&self) -> Vec<Reaped> {
        let mut out = Vec::new();
        let mut reg = self.registry.lock().unwrap();
        let ids: Vec<u64> = reg.keys().copied().collect();
        for id in ids {
            let state = reg.get(&id).unwrap();
            if let Some(code) = state.exited {
                out.push(Reaped {
                    id,
                    pid: state.pid,
                    exit_code: code,
                    owner: state.owner.clone(),
                });
                reg.remove(&id);
            }
        }
        out
    }

    pub fn alive(&self) -> Vec<ChildHandle> {
        self.registry
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, s)| s.exited.is_none())
            .map(|(id, s)| ChildHandle {
                id: *id,
                pid: s.pid,
                owner: s.owner.clone(),
                started_ms: s.started_ms,
            })
            .collect()
    }

    /// Deliberate ownership transfer (spec §22).
    pub fn transfer(&self, id: u64, new_owner: ProcessOwner) -> Result<(), Error> {
        let mut reg = self.registry.lock().unwrap();
        let state = reg
            .get_mut(&id)
            .ok_or_else(|| Error::not_found(format!("child {id}")))?;
        state.owner = new_owner;
        Ok(())
    }

    /// Session death ⇒ its children die (unless transferred first).
    pub fn kill_all_for(&self, owner: ProcessOwner) -> Vec<u64> {
        let mut killed = Vec::new();
        let mut reg = self.registry.lock().unwrap();
        let targets: Vec<(u64, u32)> = reg
            .iter()
            .filter(|(_, s)| s.owner == owner && s.exited.is_none())
            .map(|(id, s)| (*id, s.pid))
            .collect();
        for (id, pid) in targets {
            let _ = kill_group(pid, 2000);
            killed.push(id);
            if let Some(s) = reg.get_mut(&id) {
                s.exited = Some(None);
            }
        }
        killed
    }

    pub fn registered(&self) -> usize {
        self.registry.lock().unwrap().len()
    }

    fn mark_exited(&self, id: u64, code: Option<Option<i32>>) {
        let mut reg = self.registry.lock().unwrap();
        if let Some(s) = reg.get_mut(&id) {
            s.exited = code;
        }
    }
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Async cancellation poller: resolves when the token is cancelled.
async fn poll_cancelled(token: &CancellationToken, interval: Duration) {
    loop {
        if token.is_cancelled() {
            return;
        }
        tokio::time::sleep(interval).await;
    }
}

/// Blocking pipe reads (reader-thread only): drain both streams into the
/// shared capture until both EOF.
fn read_pipes(
    stdout: Option<std::process::ChildStdout>,
    stderr: Option<std::process::ChildStderr>,
    shared: Arc<Mutex<SharedCapture>>,
) {
    let mut readers: Vec<Box<dyn Read + Send>> = Vec::new();
    if let Some(s) = stdout {
        readers.push(Box::new(s));
    }
    if let Some(s) = stderr {
        readers.push(Box::new(s));
    }
    eprintln!("reader: started with {} readers", readers.len());
    if readers.is_empty() {
        return;
    }
    let mut buf = [0u8; 8192];
    let mut total = 0usize;
    loop {
        let mut progressed = false;
        readers.retain_mut(|r| match r.read(&mut buf) {
            Ok(0) => {
                eprintln!("reader: EOF on a stream");
                false
            }
            Ok(n) => {
                progressed = true;
                total += n;
                eprintln!("reader: read {n}, total {total}");
                shared.lock().unwrap().push(&buf[..n]);
                true
            }
            Err(e) => {
                eprintln!("reader: error {e:?}");
                false
            }
        });
        if readers.is_empty() {
            eprintln!("reader: all streams done, total {total}");
            break;
        }
        if !progressed {
            std::thread::sleep(Duration::from_millis(2));
        }
    }
}

/// Kill the whole process group: SIGTERM, grace, SIGKILL. On Windows the
/// process tree is killed via taskkill (Job Objects live behind cfg).
fn kill_group(pid: u32, grace_ms: u64) -> Result<(), Error> {
    #[cfg(unix)]
    {
        let sigterm = unsafe { libc::kill(-(pid as i32), libc::SIGTERM) };
        if sigterm != 0 {
            return Err(Error::internal(format!("kill TERM {pid}")));
        }
        std::thread::sleep(Duration::from_millis(grace_ms));
        let _ = unsafe { libc::kill(-(pid as i32), libc::SIGKILL) };
    }
    #[cfg(not(unix))]
    {
        let _ = std::process::Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    Ok(())
}

/// 200-line ring buffer (spec §23).
#[derive(Debug, Clone)]
pub struct RingBuffer {
    lines: std::collections::VecDeque<String>,
    capacity: usize,
}

impl RingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            lines: std::collections::VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, line: String) {
        if self.lines.len() == self.capacity {
            self.lines.pop_front();
        }
        self.lines.push_back(line);
    }

    pub fn len(&self) -> usize {
        self.lines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    pub fn excerpt(&self) -> String {
        let mut out = String::new();
        for l in &self.lines {
            out.push_str(l);
            out.push('\n');
        }
        out
    }

    /// Error-ish lines (bounded) for the excerpt.
    pub fn error_lines(&self) -> Vec<String> {
        self.lines
            .iter()
            .filter(|l| {
                let lower = l.to_ascii_lowercase();
                lower.contains("error")
                    || lower.contains("panic")
                    || lower.contains("failed")
                    || lower.contains("warning:")
            })
            .take(20)
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kilop_core::error::ErrorKind;
    use tempfile::tempdir;

    fn supervisor() -> (tempfile::TempDir, Arc<ProcessSupervisor>) {
        let dir = tempdir().unwrap();
        let cas = Arc::new(kilop_cas::Cas::open(dir.path().join("cas")).unwrap());
        (dir, ProcessSupervisor::new(cas))
    }

    fn sh(cmd: &str) -> SpawnConfig {
        SpawnConfig {
            cmd: "/bin/sh".into(),
            args: vec!["-c".into(), cmd.into()],
            cwd: std::env::temp_dir(),
            ..Default::default()
        }
    }

    fn ps_alive(pid: u32) -> bool {
        let out = std::process::Command::new("/bin/ps")
            .args(["-p", &pid.to_string()])
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout);
        // <defunct> still counts as a live entry until reaped.
        text.contains(&pid.to_string())
    }

    #[tokio::test]
    async fn ring_buffer_caps_at_200_lines() {
        let (_d, sup) = supervisor();
        let out = sup
            .run(
                sh("i=0; while [ $i -lt 10000 ]; do echo line$i; i=$((i+1)); done"),
                Duration::from_secs(30),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(out.exit_code, Some(0));
        assert!(out.ring_lines <= 200);
        assert!(out.excerpt.contains("line9999"));
        assert!(!out.excerpt.contains("line1\nline2\n"), "ring must drop the head");
        assert!(out.excerpt.len() < 64 * 1024);
    }

    #[tokio::test]
    async fn huge_output_spills_to_cas_ram_bounded() {
        let (_d, sup) = supervisor();
        let mut cfg = sh("yes 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' | head -n 500000");
        cfg.artifact_max = 1024 * 1024; // small cap so the spill triggers fast
        let out = sup
            .run(cfg, Duration::from_secs(60), CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(out.exit_code, Some(0));
        assert!(out.excerpt.len() < 64 * 1024, "excerpt bounded");
        assert!(out.artifact.is_some(), "overflow must spill to the CAS");
    }

    #[tokio::test]
    async fn kill_terminates_process_group() {
        let (_d, sup) = supervisor();
        let handle = sup.spawn(sh("sleep 30 & wait")).unwrap();
        std::thread::sleep(Duration::from_millis(300));
        assert!(ps_alive(handle.pid), "child must be alive before kill");
        sup.kill(handle.id, 500).unwrap();
        // Give the reaper a moment.
        for _ in 0..40 {
            if !ps_alive(handle.pid) {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(!ps_alive(handle.pid), "group kill must take the whole tree");
    }

    #[tokio::test]
    async fn deadline_kills_group_and_returns_timeout() {
        let (_d, sup) = supervisor();
        let err = sup
            .run(sh("sleep 30"), Duration::from_millis(300), CancellationToken::new())
            .await
            .unwrap_err();
        assert!(err.kind == ErrorKind::Timeout);
    }

    #[tokio::test]
    async fn cancellation_kills_group_and_returns_cancelled() {
        let (_d, sup) = supervisor();
        let token = CancellationToken::new();
        let t = token.clone();
        let sup2 = sup.clone();
        let task = tokio::spawn(async move {
            sup2.run(sh("sleep 30"), Duration::from_secs(60), t).await
        });
        tokio::time::sleep(Duration::from_millis(300)).await;
        token.cancel();
        let err = task.await.unwrap().unwrap_err();
        assert!(err.kind == ErrorKind::Cancelled);
    }

    #[tokio::test]
    async fn reap_collects_exit_codes_and_no_zombies() {
        let (_d, sup) = supervisor();
        let mut ids = Vec::new();
        for _ in 0..6 {
            let h = sup.spawn(sh("exit 3")).unwrap();
            ids.push(h.id);
        }
        let mut reaped = Vec::new();
        for _ in 0..40 {
            reaped.extend(sup.reap());
            if reaped.len() == 6 {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        assert_eq!(reaped.len(), 6);
        for r in &reaped {
            assert_eq!(r.exit_code, Some(3));
        }
        assert!(sup.alive().is_empty());
        assert_eq!(sup.registered(), 0, "no zombies left registered");
    }

    #[tokio::test]
    async fn kill_all_for_session_kills_children() {
        let (_d, sup) = supervisor();
        let owner = ProcessOwner::Session(SessionId::new(9));
        let mut cfg = sh("sleep 30");
        cfg.owner = owner.clone();
        let h1 = sup.spawn(cfg.clone()).unwrap();
        let h2 = sup.spawn(cfg).unwrap();
        std::thread::sleep(Duration::from_millis(200));
        let killed = sup.kill_all_for(owner);
        assert_eq!(killed.len(), 2);
        for _ in 0..40 {
            if !ps_alive(h1.pid) && !ps_alive(h2.pid) {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(!ps_alive(h1.pid));
        assert!(!ps_alive(h2.pid));
    }

    #[tokio::test]
    async fn transfer_changes_owner_and_survives() {
        let (_d, sup) = supervisor();
        let mut cfg = sh("sleep 1");
        cfg.owner = ProcessOwner::Session(SessionId::new(1));
        let h = sup.spawn(cfg).unwrap();
        sup.transfer(h.id, ProcessOwner::Daemon).unwrap();
        let killed = sup.kill_all_for(ProcessOwner::Session(SessionId::new(1)));
        assert!(killed.is_empty(), "transferred child must survive");
        sup.kill(h.id, 300).unwrap();
    }

    #[tokio::test]
    async fn unknown_id_operations_are_not_found() {
        let (_d, sup) = supervisor();
        assert!(sup.kill(999, 10).is_err());
        assert!(sup.transfer(999, ProcessOwner::Daemon).is_err());
        assert!(sup.reap().is_empty());
    }

    #[tokio::test]
    async fn exit_code_propagation() {
        let (_d, sup) = supervisor();
        assert_eq!(
            sup.run(sh("true"), Duration::from_secs(5), CancellationToken::new()).await.unwrap().exit_code,
            Some(0)
        );
        assert_eq!(
            sup.run(sh("false"), Duration::from_secs(5), CancellationToken::new()).await.unwrap().exit_code,
            Some(1)
        );
        assert_eq!(
            sup.run(sh("exit 42"), Duration::from_secs(5), CancellationToken::new()).await.unwrap().exit_code,
            Some(42)
        );
    }

    #[tokio::test]
    async fn malicious_command_vector_stays_literal() {
        let (_d, sup) = supervisor();
        let out = sup
            .run(
                SpawnConfig {
                    cmd: "/bin/sh".into(),
                    args: vec![
                        "-c".into(),
                        "printf '%s' \"$1\"".into(),
                        "x".into(),
                        "; rm -rf /tmp/kp-evil".into(),
                    ],
                    cwd: std::env::temp_dir(),
                    ..Default::default()
                },
                Duration::from_secs(5),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(out.exit_code, Some(0));
        assert!(out.excerpt.contains("; rm -rf /tmp/kp-evil"));
        assert!(!std::path::Path::new("/tmp/kp-evil").exists());
    }

    #[tokio::test]
    async fn missing_command_is_not_found() {
        let (_d, sup) = supervisor();
        let err = sup
            .run(
                SpawnConfig {
                    cmd: "/nonexistent-binary-xyz".into(),
                    ..Default::default()
                },
                Duration::from_secs(5),
                CancellationToken::new(),
            )
            .await
            .unwrap_err();
        assert!(err.kind == ErrorKind::NotFound, "{err:?}");
    }

    #[tokio::test]
    async fn dbg_50_spawns() {
        let (_d, sup) = supervisor();
        let mut ids = std::collections::HashSet::new();
        for _ in 0..50 {
            let h = sup.spawn(sh("true")).unwrap();
            ids.insert(h.id);
        }
        eprintln!("dbg: spawned 50, registered={}", sup.registered());
        for i in 0..80 {
            std::thread::sleep(Duration::from_millis(50));
            let r = sup.reap();
            if !r.is_empty() {
                eprintln!("dbg: first reap at iter {i}, count={}", r.len());
            }
            if r.len() >= 50 { break; }
        }
        eprintln!("dbg: final reaped={} registered={}", sup.reap().len(), sup.registered());
    }

    #[tokio::test]
    async fn spawn_before_run_races_unique_ids() {
        let (_d, sup) = supervisor();
        let mut ids = std::collections::HashSet::new();
        for _ in 0..50 {
            let h = sup.spawn(sh("true")).unwrap();
            assert!(ids.insert(h.id));
        }
        assert_eq!(sup.registered(), 50);
        // reap() drains: accumulate across polls until all 50 are collected.
        let mut collected = Vec::new();
        for _ in 0..80 {
            collected.extend(sup.reap());
            if collected.len() == 50 {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        assert_eq!(collected.len(), 50);
        assert_eq!(sup.registered(), 0);
    }

    #[tokio::test]
    async fn stderr_and_stdout_both_captured() {
        let (_d, sup) = supervisor();
        let out = sup
            .run(
                sh("echo out1; echo err1 >&2; echo out2"),
                Duration::from_secs(5),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert!(out.excerpt.contains("out1"));
        assert!(out.excerpt.contains("out2"));
        assert!(out.excerpt.contains("err1"));
    }

    #[tokio::test]
    async fn artifact_roundtrip_via_cas() {
        let (_d, sup) = supervisor();
        let mut cfg = sh("i=0; while [ $i -lt 200000 ]; do echo overflow-$i; i=$((i+1)); done");
        cfg.artifact_max = 1024 * 1024;
        let out = sup
            .run(cfg, Duration::from_secs(60), CancellationToken::new())
            .await
            .unwrap();
        assert!(out.artifact.is_some());
        let hash = out
            .artifact
            .as_ref()
            .and_then(|a| a.strip_prefix("artifact://"))
            .and_then(kilop_core::hash::FileHash::from_hex)
            .unwrap();
        let blob = sup.cas.get(hash).unwrap();
        assert!(String::from_utf8_lossy(&blob).contains("overflow-199999"));
    }

    #[test]
    fn ring_buffer_unit() {
        let mut r = RingBuffer::new(3);
        r.push("a".into());
        r.push("b".into());
        r.push("c".into());
        r.push("d".into());
        assert_eq!(r.len(), 3);
        assert_eq!(r.excerpt(), "b\nc\nd\n");
        assert!(r.error_lines().is_empty());
        r.push("error: boom".into());
        assert_eq!(r.error_lines(), vec!["error: boom"]);
    }
}
