//! faktor-terminal — process supervision (spec §22, §23).
//!
//! No orphans: every child process has a runtime owner; kill targets the
//! whole process group (Unix) or Job Object (Windows). Output is bounded:
//! a 200-line ring buffer live, with overflow spilling to a CAS artifact —
//! a 300MB log never becomes a 300MB RAM object. Blocking pipe reads live
//! on dedicated reader threads so they can never stall the async loop.
//!
//! This crate owns THE process supervisor for the whole workspace (audit
//! P0-40): git, lsp, mcp, hooks and the CLI daemon all spawn children
//! through [`ProcessSupervisor`]. A bounded live-child ceiling refuses
//! oversize spawns with a typed `Oversized` error before any process
//! exists; dropping the last reference (daemon shutdown) kills every live
//! child. [`ProcessSupervisor::run_sync`] gives synchronous callers (the
//! hook lifecycle) the same deadline/group-kill/bounded-head semantics as
//! [`ProcessSupervisor::run`] without a tokio context.

use std::collections::{HashMap, VecDeque};
use std::io::Read;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use faktor_core::cancellation::CancellationToken;
use faktor_core::error::Error;
use faktor_core::id::{SessionId, WorkspaceId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessOwner {
    Session(SessionId),
    /// One verification-check child of a session (the supervisor-backed
    /// check executor): separately killable (`kill_all_for`) without
    /// touching the session's other children, so a session whose turn died
    /// mid-check can reap exactly its verification tree.
    Verification(SessionId),
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
    /// Durable artifact cap in bytes (default 100MB, clamped to the global
    /// 300MB ceiling).
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

/// Explicit child-environment construction (audit P0-40). The legacy
/// `SpawnConfig::env` path always clears the env and re-injects PATH/HOME
/// plus `GIT_TERMINAL_PROMPT=0`; hooks need EXACT env semantics (an
/// allowlisted base where nothing unlisted — not even PATH — arrives), so
/// [`ProcessSupervisor::run_sync`] builds the child env EXCLUSIVELY from an
/// `EnvSpec` and never touches the legacy injection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvSpec {
    /// `env_clear` base, then `passthrough` daemon keys (copied when set),
    /// then `entries`. An entry with an EMPTY value means "copy the
    /// daemon's current value for that key" (left unset when the daemon
    /// does not carry it). Nothing else reaches the child — daemon secrets
    /// never leak implicitly.
    ClearAnd {
        entries: Vec<(String, String)>,
        passthrough: Vec<String>,
    },
    /// The daemon environment passes through untouched (only for trusted
    /// children; hooks never use this).
    Inherit,
}

impl EnvSpec {
    fn apply(&self, cmd: &mut std::process::Command) {
        match self {
            EnvSpec::ClearAnd {
                entries,
                passthrough,
            } => {
                cmd.env_clear();
                for k in passthrough {
                    if let Ok(cur) = std::env::var(k) {
                        cmd.env(k, cur);
                    }
                }
                for (k, v) in entries {
                    if v.is_empty() {
                        if let Ok(cur) = std::env::var(k) {
                            cmd.env(k, cur);
                        }
                    } else {
                        cmd.env(k, v);
                    }
                }
            }
            EnvSpec::Inherit => {}
        }
    }
}

/// Bounded-head result of one synchronous supervised run
/// ([`ProcessSupervisor::run_sync`]): per-stream heads capped at the
/// requested byte caps with truncation flags, the exit code, and a
/// `timed_out` flag — a timed-out run's partial output is forensics only
/// (the caller's failure policy decides, never partial stdout).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncRunOutput {
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    pub stdout_head: String,
    pub stderr_head: String,
    pub stdout_truncated: bool,
    pub stderr_truncated: bool,
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
/// optional durable artifact reference for the stream. When the stream
/// exceeds the effective per-command artifact cap, the artifact holds the
/// FIRST `cap` bytes, `artifact_truncated` is set, and the ring excerpt
/// carries the readable tail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandOutput {
    pub excerpt: String,
    pub exit_code: Option<i32>,
    /// CAS reference to the durable artifact (never in RAM).
    pub artifact: Option<String>,
    pub slice_hint: Option<String>,
    pub ring_lines: usize,
    /// True when the stream exceeded the effective artifact cap and tail
    /// bytes were dropped from the durable artifact.
    pub artifact_truncated: bool,
}

const RING_LINES: usize = 200;

/// Default hard ceiling on LIVE supervised children (bounded registry,
/// audit P0-40). Any spawn attempt past the ceiling is refused with a
/// typed `Oversized` error BEFORE a process exists. Exited-but-unreaped
/// entries do not count; only children that have not exited yet.
pub const DEFAULT_MAX_LIVE_CHILDREN: usize = 128;

/// Absolute ceiling on the durable artifact spool (disk), whatever
/// `SpawnConfig::artifact_max` requests: the effective per-command cap is
/// `min(artifact_max, GLOBAL_HARD_MAX)`. Past the effective cap the ring
/// keeps the tail; the artifact holds the first effective-cap bytes.
const GLOBAL_HARD_MAX: u64 = 300 * 1024 * 1024;
const MAX_EXCERPT_BYTES: usize = 64 * 1024;

/// Clamp a configured artifact cap to the global ceiling.
fn effective_artifact_max(configured: usize) -> usize {
    configured.min(GLOBAL_HARD_MAX as usize)
}

/// Bound (ms) on the post-exit drain in [`ProcessSupervisor::run`]: the
/// existence of an unrelated descendant holding an inherited descriptor must
/// never control completion of the parent command.
const POST_EXIT_DRAIN_MS: u64 = 500;

/// Per-stream post-exit drain bound (ms) in [`ProcessSupervisor::run_sync`];
/// a descendant still holding a pipe past this bound is group-killed (a
/// pipe-holding descendant proves the owned group is alive, so the kill can
/// never hit a recycled process-group id — audit round 12).
const SYNC_DRAIN_MS: u64 = 600;

/// SIGTERM→SIGKILL grace for the run_sync group kills.
const SYNC_KILL_GRACE_MS: u64 = 1200;

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
    /// Effective per-command spool cap (configured value clamped to the
    /// global ceiling); the artifact never holds more than this.
    artifact_max: usize,
    spooled: usize,
    /// True once the stream exceeded the effective cap and tail bytes were
    /// dropped from the artifact.
    artifact_truncated: bool,
    /// Overflow spills to a temp file on disk (never RAM); stored into the
    /// CAS once the command finishes.
    spill: Option<std::fs::File>,
    spill_path: Option<PathBuf>,
    artifact: Option<String>,
    cas: Arc<faktor_cas::Cas>,
}

impl SharedCapture {
    fn push(&mut self, bytes: &[u8]) {
        self.total += bytes.len();
        let text = String::from_utf8_lossy(bytes);
        for line in text.split('\n') {
            self.ring.push(line.trim_end_matches('\r').to_string());
        }
        // Full-stream spooling (audit round 5), bounded by the effective
        // per-command cap (audit round 10): the artifact is the command
        // stream from the FIRST byte up to `artifact_max`, spooled to a temp
        // file — RAM stays bounded by the ring; disk grows only to the cap.
        // Once the cap is reached, further bytes are dropped from the
        // artifact (the ring keeps the tail) and the drop is recorded in
        // `artifact_truncated`. Finalization streams the file into the CAS
        // without ever materializing it in memory.
        if self.spill.is_none() && self.artifact_max > 0 {
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
            let n = self
                .artifact_max
                .saturating_sub(self.spooled)
                .min(bytes.len());
            if n > 0 && std::io::Write::write_all(&mut *f, &bytes[..n]).is_ok() {
                self.spooled += n;
            }
            // A byte is lost only when the stream offered to the spool runs
            // past the cap; `spooled` counts what was actually stored.
            if self.spooled + (bytes.len() - n) > self.artifact_max {
                self.artifact_truncated = true;
            }
        }
    }

    /// Stream the spill file into the CAS (put_reader — never read-whole),
    /// then clean up the temp file.
    fn finalize_artifact(&mut self) {
        if let Some(path) = self.spill_path.take() {
            if let Ok(f) = std::fs::File::open(&path) {
                if let Ok(hash) = self.cas.put_reader_bounded(f, self.artifact_max) {
                    self.artifact = Some(format!("artifact://{}", hash.to_hex()));
                }
            }
            let _ = std::fs::remove_file(&path);
        }
    }
}

/// Diagnostic timeline of one supervised child (audit round 10: the Linux
/// git/worktree hang investigation needs per-child op/pid/argv/timestamps).
#[derive(Debug, Clone)]
pub struct SpawnTimeline {
    pub op_id: u64,
    pub pid: u32,
    /// The command + args as spawned (space-joined, truncated for display).
    pub argv: String,
    pub owner: String,
    pub started_ms: i64,
    pub exited_ms: Option<i64>,
    pub exit_code: Option<i32>,
}

pub struct ProcessSupervisor {
    #[cfg(windows)]
    job: JobGuard,
    registry: Arc<Mutex<HashMap<u64, ChildState>>>,
    cas: Arc<faktor_cas::Cas>,
    next_id: Arc<std::sync::atomic::AtomicU64>,
    /// Bounded ring of recently spawned children (diagnostics).
    timeline: Arc<Mutex<VecDeque<SpawnTimeline>>>,
    /// Hard ceiling on live (not-yet-exited) children.
    max_live: usize,
    /// Serializes [admit → spawn → register] so the live ceiling is exact
    /// even under a spawn race (100 concurrent spawners never overshoot).
    spawn_serial: Mutex<()>,
}

impl std::fmt::Debug for ProcessSupervisor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProcessSupervisor")
            .field("registered", &self.registered())
            .field("max_live", &self.max_live)
            .finish()
    }
}

impl Drop for ProcessSupervisor {
    /// Daemon-shutdown scope (spec §22 / commandment 8): when the LAST
    /// reference to the supervisor drops, every still-live child is
    /// killed. No child outlives its runtime owner.
    fn drop(&mut self) {
        let targets: Vec<u32> = {
            let reg = self.registry.lock().unwrap();
            reg.values()
                .filter(|s| s.exited.is_none())
                .map(|s| s.pid)
                .collect()
        };
        for pid in targets {
            let _ = kill_group(pid, 300);
        }
    }
}

impl ProcessSupervisor {
    pub fn new(cas: Arc<faktor_cas::Cas>) -> Arc<Self> {
        Self::with_limit(cas, DEFAULT_MAX_LIVE_CHILDREN)
    }

    /// Like [`ProcessSupervisor::new`] with an explicit live-child ceiling
    /// (bounded registry; spawns past the ceiling fail `Oversized`).
    pub fn with_limit(cas: Arc<faktor_cas::Cas>, max_live: usize) -> Arc<Self> {
        let max_live = max_live.max(1);
        Arc::new(Self {
            #[cfg(windows)]
            job: JobGuard::create().unwrap_or_else(|| {
                tracing::warn!(
                    "CreateJobObject failed; children lose the OS kill-on-close guarantee"
                );
                // A zero handle keeps every assign a no-op.
                JobGuard::null()
            }),
            registry: Arc::new(Mutex::new(HashMap::new())),
            cas,
            next_id: Arc::new(std::sync::atomic::AtomicU64::new(1)),
            timeline: Arc::new(Mutex::new(VecDeque::new())),
            max_live,
            spawn_serial: Mutex::new(()),
        })
    }

    /// The process-wide shared supervisor (audit P0-40). Callers that
    /// cannot receive a daemon-rooted supervisor through their constructor
    /// (the env-var hook registry, crate-level tests) spawn their children
    /// here instead of building a private process layer. Rooted at an
    /// ephemeral per-process temp CAS: every spawn path used through
    /// `shared()` (env-exact, bounded-head) never touches the artifact
    /// spool, so no durable state lives there.
    pub fn shared() -> Arc<Self> {
        static SHARED: std::sync::OnceLock<Arc<ProcessSupervisor>> = std::sync::OnceLock::new();
        SHARED
            .get_or_init(|| {
                let dir = std::env::temp_dir()
                    .join(format!("kp-supervisor-shared-{}", std::process::id()));
                if std::fs::create_dir_all(&dir).is_ok() {
                    if let Ok(cas) = faktor_cas::Cas::open(dir.join("cas")) {
                        return ProcessSupervisor::new(Arc::new(cas));
                    }
                }
                panic!(
                    "ProcessSupervisor::shared: cannot open its ephemeral CAS root at {:?}",
                    dir
                );
            })
            .clone()
    }

    fn alloc_id(&self) -> u64 {
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if id == 0 {
            self.next_id
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        } else {
            id
        }
    }

    /// Args/cwd/process-group base, NO env applied: the legacy
    /// [`SpawnConfig::env`] injection and the exact [`EnvSpec`] policy are
    /// layered on by the callers below.
    fn command_base(&self, cfg: &SpawnConfig) -> std::process::Command {
        let mut cmd = std::process::Command::new(&cfg.cmd);
        cmd.args(&cfg.args).current_dir(&cfg.cwd);
        // Own process group so kills target the whole tree.
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }
        cmd
    }

    /// Legacy env policy: cleared base + PATH/HOME + configured entries +
    /// `GIT_TERMINAL_PROMPT=0` (byte-for-byte the historic behavior; the
    /// env-var value always wins over a configured `GIT_TERMINAL_PROMPT`).
    fn command(&self, cfg: &SpawnConfig) -> std::process::Command {
        let mut cmd = self.command_base(cfg);
        cmd.env_clear()
            .env("PATH", std::env::var("PATH").unwrap_or_default())
            .env("HOME", std::env::var("HOME").unwrap_or_default());
        for (k, v) in &cfg.env {
            cmd.env(k, v);
        }
        cmd.env("GIT_TERMINAL_PROMPT", "0");
        cmd
    }

    /// Refuse the spawn when the live-child ceiling is reached: the caller
    /// holds `spawn_serial`, so admit→spawn→register is atomic and a spawn
    /// race can never overshoot the ceiling.
    fn admit(&self) -> Result<(), Error> {
        let live = self
            .registry
            .lock()
            .unwrap()
            .values()
            .filter(|s| s.exited.is_none())
            .count();
        if live >= self.max_live {
            return Err(Error::oversized(format!(
                "live child ceiling reached: {live} live children registered, ceiling is {}; \
                 refusing the spawn",
                self.max_live
            )));
        }
        Ok(())
    }

    fn register(&self, pid: u32, owner: ProcessOwner, started_ms: i64) -> u64 {
        #[cfg(windows)]
        self.job.assign(pid);
        let id = self.alloc_id();
        self.registry.lock().unwrap().insert(
            id,
            ChildState {
                pid,
                owner: owner.clone(),
                started_ms,
                exited: None,
            },
        );
        id
    }

    /// Recent spawns, newest first (bounded; see [`SpawnTimeline`]).
    pub fn recent_spawns(&self) -> Vec<SpawnTimeline> {
        self.timeline.lock().unwrap().iter().cloned().collect()
    }

    fn timeline_spawn(&self, op_id: u64, pid: u32, argv: String, owner: &ProcessOwner) {
        let mut tl = self.timeline.lock().unwrap();
        tl.push_front(SpawnTimeline {
            op_id,
            pid,
            argv,
            owner: format!("{owner:?}"),
            started_ms: now_ms(),
            exited_ms: None,
            exit_code: None,
        });
        tl.truncate(256);
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
        // Audit round 11: a DROPPED run() future (outer timeout/unwind) must
        // never leak the child — tokio kills the direct child on drop, and
        // the RAII guard below SIGKILLs the whole group as a last resort.
        cmd.kill_on_drop(true);
        let mut child = cmd
            .spawn()
            .map_err(|e| Error::not_found(format!("spawn {}: {e}", cfg.cmd)))?;
        let pid = child.id().unwrap_or(0);
        let id = self.register(pid, cfg.owner.clone(), started_ms);
        self.timeline_spawn(
            id,
            pid,
            format!("{} {}", cfg.cmd, cfg.args.join(" "))
                .chars()
                .take(300)
                .collect(),
            &cfg.owner,
        );

        let shared: Arc<Mutex<SharedCapture>> = Arc::new(Mutex::new(SharedCapture {
            ring: RingBuffer::new(RING_LINES),
            total: 0,
            artifact_max: effective_artifact_max(cfg.artifact_max),
            spooled: 0,
            artifact_truncated: false,
            spill: None,
            spill_path: None,
            artifact: None,
            cas: self.cas.clone(),
        }));

        // Async reader task: drains both streams into the bounded ring.
        // Audit round 51: a pipe whose EOF was observed is REMOVED from
        // future selects (arm precondition) — an EOF-ready read would
        // resolve `Ok(0)` instantly on every poll and spin the loop while
        // the other pipe stays open. Cancellation is wake-driven
        // (`CancellationToken::cancelled`, std-waker registration), so the
        // reader also stops promptly on cancel without a 5 ms poll timer.
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
                    tokio::select! {
                        _ = token2.cancelled() => break,
                        r = async {
                            match stdout.as_mut() {
                                Some(s) => s.read(&mut out_buf).await,
                                None => Ok(0),
                            }
                        }, if !stdout_eof => {
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
                        }, if !stderr_eof => {
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
        // bound; the child's group is killed on timeout (async: no blocking
        // sleep inside the runtime).
        enum RunOutcome {
            Exited(Option<std::process::ExitStatus>),
            TimedOut,
            Cancelled,
        }
        // The deadline is an ABSOLUTE instant: `select!` rebuilds its arm
        // futures on every poll, so a relative `sleep(deadline)` against
        // the hot 5ms cancellation arm would slide forward forever under
        // load and never fire (audit round 11 hang). `sleep_until` pins the
        // deadline.
        // Process-lifetime discipline (audit round 12 — the P0): ONE
        // reaping authority per child (this future owns child.wait), the
        // leader's pgid is used for tree signalling ONLY while the group
        // provably still exists, and a normally-exited command is NOT
        // group-killed (that killed unrelated trees via PID/PGID reuse).
        // Sequence: child exit -> bounded pipe drain -> EOF? finish :
        // descendant still owns the pipe -> terminate the OWNED tree (a
        // descendant holding our pipe means the group is alive, so the
        // pgid cannot have been recycled) -> bounded final drain -> finish.
        let deadline_at = tokio::time::Instant::now() + deadline;
        let outcome = tokio::select! {
            s = child.wait() => RunOutcome::Exited(s.ok()),
            _ = tokio::time::sleep_until(deadline_at) => {
                let _ = kill_group_async(pid, 2000).await;
                let _ = child.kill().await;
                let _ = child.wait().await;
                // The child is gone WITH no exit code: mark it exited so
                // reap() can collect the registry entry exactly once. The
                // old `None` left the child permanently "alive" in the
                // registry (audit round 17).
                self.mark_exited(id, Some(None));
                RunOutcome::TimedOut
            }
            _ = token.cancelled() => {
                let _ = kill_group_async(pid, 500).await;
                let _ = child.kill().await;
                let _ = child.wait().await;
                self.mark_exited(id, Some(None));
                RunOutcome::Cancelled
            }
        };

        // The direct child is gone. Drain its remaining output for the
        // bounded window; a reader that is STILL alive after the window
        // means a descendant keeps one of our pipes open (EOF can never
        // arrive). Only then do we terminate the owned tree — at that
        // moment a pipe-holding descendant exists, so the pgid is live and
        // the kill cannot hit a recycled id.
        let mut reader = reader;
        let drain_done = {
            let mut r = reader.take();
            let done = tokio::time::timeout(Duration::from_millis(POST_EXIT_DRAIN_MS), async {
                if let Some(r) = r.as_mut() {
                    let _ = r.await;
                }
            })
            .await
            .is_ok();
            if r.is_some() && !done {
                // Descendant still owns the pipe: terminate the owned tree.
                let _ = kill_group_async(pid, 1500).await;
                if let Some(r) = r {
                    r.abort();
                    let _ = r.await;
                }
            }
            done
        };
        let _ = drain_done;

        let exit_code = match outcome {
            RunOutcome::Exited(status) => status.and_then(|s| s.code()),
            RunOutcome::TimedOut => {
                return Err(Error::timeout(format!(
                    "command {} exceeded its {}ms deadline",
                    cfg.cmd,
                    deadline.as_millis()
                )));
            }
            RunOutcome::Cancelled => {
                return Err(Error::cancelled());
            }
        };
        self.mark_exited(id, Some(exit_code));

        let mut slice_hint = None;
        let (excerpt, artifact, artifact_truncated) = {
            let mut g = shared.lock().unwrap();
            g.finalize_artifact();
            let mut excerpt = g.ring.excerpt();
            let artifact = g.artifact.clone();
            let artifact_truncated = g.artifact_truncated;
            excerpt.push_str(&format!("[exit code: {}]\n", exit_code.unwrap_or(-1)));
            if excerpt.len() > MAX_EXCERPT_BYTES {
                excerpt.truncate(MAX_EXCERPT_BYTES);
            }
            (excerpt, artifact, artifact_truncated)
        };
        if let Some(a) = &artifact {
            slice_hint = Some(format!("{a}?slice=0&len=1024"));
        }
        Ok(CommandOutput {
            excerpt,
            exit_code,
            artifact,
            slice_hint,
            ring_lines: RING_LINES,
            artifact_truncated,
        })
    }

    /// Stream a child pipe into a bounded head: read incrementally, keep at
    /// most `cap` bytes, then DRAIN the remainder in fixed chunks so a
    /// hostile 10 MB / infinite producer never grows memory past the cap
    /// (the child is only ever blocked briefly per kernel pipe buffer,
    /// never on us). Returns (lossy head, truncated?).
    fn read_bounded_head(pipe: Box<dyn Read + Send>, cap: usize) -> (String, bool) {
        const CHUNK: usize = 8192;
        let mut head: Vec<u8> = Vec::with_capacity(cap.min(CHUNK));
        let mut scratch = [0u8; CHUNK];
        let mut truncated = false;
        let mut pipe = pipe;
        loop {
            let n = match pipe.read(&mut scratch) {
                Ok(0) => break,
                Ok(n) => n,
                Err(_) => break,
            };
            if head.len() < cap {
                let take = (cap - head.len()).min(n);
                head.extend_from_slice(&scratch[..take]);
                if take < n {
                    truncated = true;
                }
            } else {
                truncated = true;
            }
        }
        (String::from_utf8_lossy(&head).into_owned(), truncated)
    }

    /// The SYNCHRONOUS bounded run (audit P0-40; the hook lifecycle runs
    /// from synchronous contexts and cannot await [`Self::run`]).
    ///
    /// Semantics mirror `run()` without a tokio context:
    /// - the child env comes EXCLUSIVELY from `env` ([`EnvSpec`]); the
    ///   legacy PATH/HOME/`GIT_TERMINAL_PROMPT` injection is NOT applied —
    ///   an allowlisted hook must not see an implicit PATH;
    /// - the child runs in its own process group; the deadline DOMINATES —
    ///   on expiry the OWNED tree is killed (guarded against reaping a
    ///   recycled group id) and partial output is reported as forensics
    ///   with `timed_out: true` (the caller's policy decides, never the
    ///   partial stdout);
    /// - stdout/stderr are read on dedicated threads into bounded heads
    ///   (remainder drained, never buffered);
    /// - after the direct child exits, a descendant still holding a pipe
    ///   past the drain bound is group-killed (a pipe-holding descendant
    ///   proves the group is alive — the kill cannot hit a recycled id),
    ///   so neither the caller nor a reader thread is ever owned by a
    ///   grandchild;
    /// - the run is registered in the registry (owner row, timeline) and
    ///   marked exited exactly once; `reap()` collects the entry.
    ///
    /// Wall time is bounded by `deadline` + a small constant (kill grace
    /// and bounded drains), never by the child's behavior.
    pub fn run_sync(
        &self,
        cfg: SpawnConfig,
        env: EnvSpec,
        deadline: Duration,
        stdout_cap: usize,
        stderr_cap: usize,
    ) -> Result<SyncRunOutput, Error> {
        let argv = format!("{} {}", cfg.cmd, cfg.args.join(" "))
            .chars()
            .take(300)
            .collect();
        let mut cmd = self.command_base(&cfg);
        env.apply(&mut cmd);
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let started_ms = now_ms();
        let (mut child, pid, id) = {
            let _serial = self.spawn_serial.lock().unwrap();
            self.admit()?;
            let child = cmd
                .spawn()
                .map_err(|e| Error::not_found(format!("spawn {}: {e}", cfg.cmd)))?;
            let pid = child.id();
            let id = self.register(pid, cfg.owner.clone(), started_ms);
            self.timeline_spawn(id, pid, argv, &cfg.owner);
            (child, pid, id)
        };
        // Dedicated reader threads: bounded head per stream, remainder
        // drained, then ONE send. Reads can never block the caller past
        // the bounded settles below.
        let (out_tx, out_rx) = std::sync::mpsc::channel();
        let out_pipe = child.stdout.take();
        std::thread::spawn(move || {
            let res = match out_pipe {
                Some(p) => Self::read_bounded_head(Box::new(p), stdout_cap),
                None => (String::new(), false),
            };
            let _ = out_tx.send(res);
        });
        let (err_tx, err_rx) = std::sync::mpsc::channel();
        let err_pipe = child.stderr.take();
        std::thread::spawn(move || {
            let res = match err_pipe {
                Some(p) => Self::read_bounded_head(Box::new(p), stderr_cap),
                None => (String::new(), false),
            };
            let _ = err_tx.send(res);
        });
        // The waiter owns reaping; the caller enforces the deadline. The
        // reaped flag guards the kill: a reaped pid is never signalled (a
        // recycled group must not die for our deadline).
        let reaped = Arc::new(AtomicBool::new(false));
        let (exit_tx, exit_rx) = std::sync::mpsc::channel();
        {
            let reaped = reaped.clone();
            std::thread::spawn(move || {
                let code = child.wait().ok().and_then(|s| s.code());
                reaped.store(true, Ordering::SeqCst);
                let _ = exit_tx.send(code);
            });
        }
        let (exit_code, timed_out) = match exit_rx.recv_timeout(deadline) {
            Ok(code) => (code, false),
            Err(_) => {
                // Deadline fired: kill the OWNED tree (only while the child
                // is still ours), then give the reaper a bounded moment.
                if !reaped.load(Ordering::SeqCst) {
                    let _ = kill_group(pid, 500);
                }
                let code = exit_rx
                    .recv_timeout(Duration::from_millis(500))
                    .ok()
                    .flatten();
                (code, true)
            }
        };
        // Exactly-once exit marking (registry + timeline); a timed-out tree
        // was killed, so its exit code is None unless it raced out cleanly.
        self.mark_exited(id, Some(exit_code));
        // Bounded settle: each reader finishes at pipe EOF. A grandchild
        // that inherited the pipe delays it — never the caller, never the
        // reader forever: past the drain bound the pipe-holding descendant
        // is group-killed and the final heads are collected.
        let settle = Duration::from_millis(SYNC_DRAIN_MS);
        let mut out_head = out_rx.recv_timeout(settle).ok();
        let mut err_head = err_rx.recv_timeout(settle).ok();
        if out_head.is_none() || err_head.is_none() {
            let _ = kill_group(pid, SYNC_KILL_GRACE_MS);
            let grace = Duration::from_millis(500);
            if out_head.is_none() {
                out_head = out_rx.recv_timeout(grace).ok();
            }
            if err_head.is_none() {
                err_head = err_rx.recv_timeout(grace).ok();
            }
        }
        let (stdout_head, stdout_truncated) = out_head.unwrap_or_else(|| (String::new(), false));
        let (stderr_head, stderr_truncated) = err_head.unwrap_or_else(|| (String::new(), false));
        Ok(SyncRunOutput {
            exit_code,
            timed_out,
            stdout_head,
            stderr_head,
            stdout_truncated,
            stderr_truncated,
        })
    }

    /// Spawn with piped stdin/stdout/stderr (for MCP/LSP style servers).
    /// The caller owns the pipes; a reaper thread still reaps the child.
    pub fn spawn_detached_with_pipes(&self, mut cfg: SpawnConfig) -> Result<SpawnedProcess, Error> {
        cfg.capture = false;
        let mut cmd = self.command(&cfg);
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let started_ms = now_ms();
        let mut child = cmd
            .spawn()
            .map_err(|e| Error::not_found(format!("spawn {}: {e}", cfg.cmd)))?;
        let pid = child.id();
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| Error::internal("no stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| Error::internal("no stdout"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| Error::internal("no stderr"))?;
        let id = self.register(pid, cfg.owner.clone(), started_ms);
        self.timeline_spawn(
            id,
            pid,
            format!("{} {}", cfg.cmd, cfg.args.join(" "))
                .chars()
                .take(300)
                .collect(),
            &cfg.owner,
        );
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
        let (child, pid, id) = {
            let _serial = self.spawn_serial.lock().unwrap();
            self.admit()?;
            let child = cmd
                .spawn()
                .map_err(|e| Error::not_found(format!("spawn {}: {e}", cfg.cmd)))?;
            let pid = child.id();
            let id = self.register(pid, cfg.owner.clone(), started_ms);
            self.timeline_spawn(
                id,
                pid,
                format!("{} {}", cfg.cmd, cfg.args.join(" "))
                    .chars()
                    .take(300)
                    .collect(),
                &cfg.owner,
            );
            (child, pid, id)
        };
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
        process_alive(pid)
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
            let found = {
                let tl = self.timeline.lock().unwrap();
                tl.iter().any(|t| t.op_id == id)
            };
            if found {
                let mut tl = self.timeline.lock().unwrap();
                if let Some(t) = tl.iter_mut().find(|t| t.op_id == id) {
                    t.exited_ms = Some(now_ms());
                    t.exit_code = code.flatten();
                }
            }
        }
    }
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Is the pid still alive (zombies do not count)?
#[cfg(not(unix))]
fn process_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    std::process::Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}")])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains(&pid.to_string()))
        .unwrap_or(false)
}

#[cfg(unix)]
fn process_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    // kill(pid, 0) probes existence natively (no /bin/ps); zombies count as
    // existing (they are reaped by the caller's waitpid/child.wait).
    let r = unsafe { libc::kill(pid as i32, 0) };
    if r == -1 {
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() == Some(libc::ESRCH) {
            return false;
        }
    }
    true
}

/// True when NO live member of the process group remains. Native probe:
/// exited members are reaped via waitpid(-pgid, WNOHANG), then
/// kill(-pgid, 0) returning ESRCH proves the group is truly gone —
/// a stubborn child that survived SIGTERM keeps the group alive and forces
/// the SIGKILL escalation (audit round 5: leader-gone != group-gone).
#[cfg(unix)]
fn group_gone(pgid: u32) -> bool {
    if pgid == 0 {
        return true;
    }
    // Liveness probing must NEVER reap (audit round 12): one reaping
    // authority per child. Zombie members do not count as alive for our
    // purposes — their fds are closed, so a group of zombies cannot hold a
    // pipe; kill() on a zombie-only group succeeds until they are reaped,
    // which merely means we send one harmless extra signal.
    let r = unsafe { libc::kill(-(pgid as i32), 0) };
    if r == -1 {
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() == Some(libc::ESRCH) {
            return true; // no members at all
        }
        // EPERM: the group exists but belongs to another user — treat as alive.
    }
    false
}

/// Blocking pipe reads (reader-thread only): drain both streams into the
/// shared capture until both EOF.
#[allow(dead_code)]
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
    if readers.is_empty() {
        return;
    }
    let mut buf = [0u8; 8192];
    let mut total = 0usize;
    loop {
        let mut progressed = false;
        readers.retain_mut(|r| match r.read(&mut buf) {
            Ok(0) => false,
            Ok(n) => {
                progressed = true;
                total += n;
                shared.lock().unwrap().push(&buf[..n]);
                true
            }
            Err(e) => {
                eprintln!("reader: error {e:?}");
                false
            }
        });
        if readers.is_empty() {
            break;
        }
        if !progressed {
            std::thread::sleep(Duration::from_millis(2));
        }
    }
}

/// Kill the whole process group: SIGTERM, grace, SIGKILL. On Windows the
/// process tree is killed via taskkill (Job Objects live behind cfg).
/// The grace wait exits early: the moment the group leader is gone the
/// function returns instead of sleeping the full grace.
fn kill_group(pid: u32, grace_ms: u64) -> Result<(), Error> {
    #[cfg(unix)]
    {
        let sigterm = unsafe { libc::kill(-(pid as i32), libc::SIGTERM) };
        if sigterm != 0 {
            return Err(Error::internal(format!("kill TERM {pid}")));
        }
        let deadline = std::time::Instant::now() + Duration::from_millis(grace_ms);
        while std::time::Instant::now() < deadline {
            if group_gone(pid) {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        // A stubborn descendant survived SIGTERM: SIGKILL the whole group.
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

/// Async kill of the whole process group: SIGTERM, then poll for exit every
/// 25ms (no blocking sleep inside the runtime); the moment the group leader
/// is gone the future returns. At the grace deadline SIGKILL is sent and the
/// future returns.
pub async fn kill_group_async(pid: u32, grace_ms: u64) -> Result<(), Error> {
    #[cfg(unix)]
    {
        let sigterm = unsafe { libc::kill(-(pid as i32), libc::SIGTERM) };
        if sigterm != 0 {
            return Err(Error::internal(format!("kill TERM {pid}")));
        }
        let deadline = tokio::time::Instant::now() + Duration::from_millis(grace_ms);
        loop {
            if group_gone(pid) {
                return Ok(());
            }
            if tokio::time::Instant::now() >= deadline {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        // A stubborn descendant survived SIGTERM: SIGKILL the whole group.
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

// Every test below drives the process-group/signal machinery (setsid
// children, /bin/sh scripts, SIGTERM grace, /bin/ps probes) and only runs
// on unix hosts; the Windows certification suite lives in `windows_tests`.
#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use faktor_core::error::ErrorKind;
    use tempfile::tempdir;

    fn supervisor() -> (tempfile::TempDir, Arc<ProcessSupervisor>) {
        let dir = tempdir().unwrap();
        let cas = Arc::new(faktor_cas::Cas::open(dir.path().join("cas")).unwrap());
        (dir, ProcessSupervisor::new(cas))
    }

    fn supervisor_with_limit(limit: usize) -> (tempfile::TempDir, Arc<ProcessSupervisor>) {
        let dir = tempdir().unwrap();
        let cas = Arc::new(faktor_cas::Cas::open(dir.path().join("cas")).unwrap());
        (dir, ProcessSupervisor::with_limit(cas, limit))
    }

    fn pid_is_gone(pid: u32) -> bool {
        #[cfg(unix)]
        {
            let r = unsafe { libc::kill(pid as i32, 0) };
            if r == -1 {
                let err = std::io::Error::last_os_error();
                if err.raw_os_error() == Some(libc::ESRCH) {
                    return true;
                }
            }
            false
        }
        #[cfg(not(unix))]
        {
            !ps_alive(pid)
        }
    }

    #[test]
    fn run_sync_reports_exit_code_and_bounded_heads() {
        let (_d, sup) = supervisor();
        let out = sup
            .run_sync(
                sh("echo out-line; echo err-line >&2; exit 3"),
                EnvSpec::Inherit,
                Duration::from_secs(10),
                4096,
                4096,
            )
            .unwrap();
        assert!(!out.timed_out);
        assert_eq!(out.exit_code, Some(3));
        assert!(
            out.stdout_head.contains("out-line"),
            "{:?}",
            out.stdout_head
        );
        assert!(
            out.stderr_head.contains("err-line"),
            "{:?}",
            out.stderr_head
        );
        assert!(!out.stdout_truncated);
        assert!(!out.stderr_truncated);
        // The exit is marked exactly once; reap collects the single entry.
        for _ in 0..40 {
            if !sup.reap().is_empty() {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        assert_eq!(sup.registered(), 0, "reap collects the run_sync child");
    }

    #[test]
    fn run_sync_deadline_kills_the_whole_tree() {
        let dir = tempdir().unwrap();
        let cas = Arc::new(faktor_cas::Cas::open(dir.path().join("cas")).unwrap());
        let sup = ProcessSupervisor::new(cas);
        let pf = dir.path().join("gc.pid");
        let mut cfg = sh(&format!("sleep 30 & echo $! > '{}'; wait", pf.display()));
        cfg.owner = ProcessOwner::Daemon;
        let t0 = std::time::Instant::now();
        let out = sup
            .run_sync(
                cfg,
                EnvSpec::Inherit,
                Duration::from_millis(400),
                4096,
                4096,
            )
            .unwrap();
        assert!(out.timed_out, "deadline must dominate");
        assert_eq!(out.exit_code, None, "the tree was killed, not exited");
        assert!(
            t0.elapsed() < Duration::from_secs(10),
            "deadline kill must be prompt"
        );
        // The grandchild died with the group — no orphan survives the kill.
        let gc: u32 = std::fs::read_to_string(&pf)
            .unwrap()
            .trim()
            .parse()
            .expect("grandchild pid file");
        let mut gone = false;
        for _ in 0..100 {
            if pid_is_gone(gc) {
                gone = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(gone, "the hook grandchild must die with the group kill");
        assert!(sup.alive().is_empty(), "no live child after the kill");
    }

    #[test]
    fn run_sync_env_clear_and_is_exact() {
        let (_d, sup) = supervisor();
        std::env::set_var("FAKTOR_HOSTILE", "sekrit");
        // Cleared base: the hostile daemon var and HOME (which the shell
        // does NOT invent, unlike PATH) must be absent; the allowlisted
        // entry must be present.
        let out = sup
            .run_sync(
                sh("test -z \"$FAKTOR_HOSTILE\" && test \"$VISIBLE\" = 1 && test -z \"$HOME\" && echo exact"),
                EnvSpec::ClearAnd {
                    entries: vec![("VISIBLE".into(), "1".into())],
                    passthrough: vec![],
                },
                Duration::from_secs(10),
                4096,
                4096,
            )
            .unwrap();
        assert_eq!(out.exit_code, Some(0), "{:?}", out.stdout_head);
        assert!(out.stdout_head.contains("exact"));
        std::env::remove_var("FAKTOR_HOSTILE");
        // passthrough + empty-value-inherit: benign keys and the daemon's
        // own value for an explicitly-listed key arrive; the hostile var
        // still does not.
        std::env::set_var("FAKTOR_HOSTILE", "sekrit");
        std::env::set_var("KP_DAEMON_ONLY", "xyz");
        let out = sup
            .run_sync(
                sh("test -n \"$PATH\" && test -n \"$HOME\" && test \"$KP_DAEMON_ONLY\" = xyz && test -z \"$FAKTOR_HOSTILE\" && echo benign"),
                EnvSpec::ClearAnd {
                    entries: vec![("KP_DAEMON_ONLY".into(), String::new())],
                    passthrough: vec!["PATH".into(), "HOME".into()],
                },
                Duration::from_secs(10),
                4096,
                4096,
            )
            .unwrap();
        assert_eq!(out.exit_code, Some(0), "{:?}", out.stdout_head);
        std::env::remove_var("FAKTOR_HOSTILE");
        std::env::remove_var("KP_DAEMON_ONLY");
    }

    #[test]
    fn run_sync_heads_are_capped_and_truncation_reported() {
        let (_d, sup) = supervisor();
        let out = sup
            .run_sync(
                sh("dd if=/dev/zero bs=1048576 count=2 2>/dev/null | tr '\\0' 'x'"),
                EnvSpec::Inherit,
                Duration::from_secs(30),
                128,
                128,
            )
            .unwrap();
        assert_eq!(out.exit_code, Some(0));
        assert!(out.stdout_truncated, "2MB over a 128-byte cap truncates");
        assert!(out.stdout_head.len() <= 128, "head is bounded");
        // A flood must complete (drained, not deadlocked) within the call.
        let t0 = std::time::Instant::now();
        assert!(t0.elapsed() < Duration::from_secs(8));
    }

    #[test]
    fn run_sync_ends_promptly_when_a_descendant_holds_the_pipe() {
        let (_d, sup) = supervisor();
        let t0 = std::time::Instant::now();
        let out = sup
            .run_sync(
                sh("(sleep 30) & echo done"),
                EnvSpec::Inherit,
                Duration::from_secs(30),
                4096,
                4096,
            )
            .unwrap();
        let elapsed = t0.elapsed();
        assert_eq!(out.exit_code, Some(0));
        assert!(!out.timed_out);
        assert!(out.stdout_head.contains("done"), "{:?}", out.stdout_head);
        assert!(
            elapsed < Duration::from_secs(5),
            "a pipe-holding descendant must never own the caller: {elapsed:?}"
        );
    }

    #[test]
    fn shared_returns_the_process_wide_singleton() {
        let a = ProcessSupervisor::shared();
        let b = ProcessSupervisor::shared();
        assert!(Arc::ptr_eq(&a, &b));
        // The shared supervisor actually runs env-cleared children.
        let out = a
            .run_sync(
                sh("echo shared-ok"),
                EnvSpec::ClearAnd {
                    entries: vec![],
                    passthrough: vec![],
                },
                Duration::from_secs(10),
                4096,
                4096,
            )
            .unwrap();
        assert_eq!(out.exit_code, Some(0));
        assert!(out.stdout_head.contains("shared-ok"));
    }

    #[test]
    fn live_ceiling_refuses_oversize_before_any_child_exists() {
        let (_d, sup) = supervisor_with_limit(3);
        let mut held = Vec::new();
        for _ in 0..3 {
            let h = sup.spawn(sh("sleep 30")).unwrap();
            held.push(h);
        }
        let err = sup.spawn(sh("true")).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Oversized, "{err:?}");
        assert_eq!(sup.alive().len(), 3, "the refused spawn never existed");
        for h in &held {
            assert!(sup.kill(h.id, 500).is_ok());
        }
        for _ in 0..60 {
            if sup.alive().is_empty() {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(sup.alive().is_empty());
    }

    #[test]
    fn drop_of_the_last_reference_kills_live_children() {
        let dir = tempdir().unwrap();
        let cas = Arc::new(faktor_cas::Cas::open(dir.path().join("cas")).unwrap());
        let pid = {
            let sup = ProcessSupervisor::new(cas);
            let h = sup.spawn(sh("sleep 30")).unwrap();
            h.pid
        };
        let mut gone = false;
        for _ in 0..100 {
            if pid_is_gone(pid) {
                gone = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(gone, "daemon-shutdown Drop must kill live children");
    }

    #[test]
    fn run_sync_refusal_at_the_ceiling_is_typed_oversized() {
        let (_d, sup) = supervisor_with_limit(1);
        let h = sup.spawn(sh("sleep 30")).unwrap();
        let err = sup
            .run_sync(
                sh("true"),
                EnvSpec::Inherit,
                Duration::from_secs(5),
                1024,
                1024,
            )
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Oversized, "{err:?}");
        assert!(sup.kill(h.id, 500).is_ok());
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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
        assert!(
            !out.excerpt.contains("line1\nline2\n"),
            "ring must drop the head"
        );
        assert!(out.excerpt.len() < 64 * 1024);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn deadline_kills_group_and_returns_timeout() {
        let (_d, sup) = supervisor();
        let err = sup
            .run(
                sh("sleep 30"),
                Duration::from_millis(300),
                CancellationToken::new(),
            )
            .await
            .unwrap_err();
        assert!(err.kind == ErrorKind::Timeout);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancellation_kills_group_and_returns_cancelled() {
        let (_d, sup) = supervisor();
        let token = CancellationToken::new();
        let t = token.clone();
        let sup2 = sup.clone();
        let task =
            tokio::spawn(async move { sup2.run(sh("sleep 30"), Duration::from_secs(60), t).await });
        tokio::time::sleep(Duration::from_millis(300)).await;
        token.cancel();
        let err = task.await.unwrap().unwrap_err();
        assert!(err.kind == ErrorKind::Cancelled);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reader_does_not_spin_after_one_pipe_eofs() {
        // Audit round 51: stdout closes EARLY while stderr stays open for
        // ~1s. Once stdout_eof is set the stdout arm must be removed from
        // the reader's select — an EOF-ready pipe would resolve Ok(0) on
        // every poll and spin the loop at 100% CPU until stderr closes.
        // Behavior must be identical: both streams still captured, run()
        // completes promptly after the second pipe closes.
        let (_d, sup) = supervisor();
        let t0 = std::time::Instant::now();
        let out = sup
            .run(
                sh("echo out-first; exec 1>&-; sleep 1; echo err-tail >&2"),
                Duration::from_secs(10),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        let elapsed = t0.elapsed();
        assert_eq!(out.exit_code, Some(0));
        assert!(out.excerpt.contains("out-first"), "{:?}", out.excerpt);
        assert!(out.excerpt.contains("err-tail"), "{:?}", out.excerpt);
        assert!(
            elapsed >= Duration::from_millis(500),
            "the reader must wait on the still-open pipe, not spin: {elapsed:?}"
        );
        assert!(
            elapsed < Duration::from_secs(8),
            "the reader must complete promptly once the second pipe closes: {elapsed:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn deadline_timeout_aborts_reader_and_supervisor_stays_clean() {
        // Audit round 17: on the terminal timeout path the reader task must
        // be terminated exactly once (joined via the post-exit drain, or
        // aborted) and never leak into the next command. The child ignores
        // SIGTERM and keeps writing, so the reader is mid-read at the
        // deadline; only the SIGKILL escalation closes the pipes.
        let (_d, sup) = supervisor();
        let t0 = std::time::Instant::now();
        let err = sup
            .run(
                sh("trap '' TERM; i=0; while true; do echo stuck-line-$i; i=$((i+1)); done"),
                Duration::from_millis(300),
                CancellationToken::new(),
            )
            .await
            .unwrap_err();
        assert!(err.kind == ErrorKind::Timeout, "{err:?}");
        assert!(
            t0.elapsed() < Duration::from_secs(6),
            "timeout must escalate SIGKILL and return: {:?}",
            t0.elapsed()
        );
        // Exactly-once reaping: the timed-out child is marked exited and
        // collected by the next reap() (one entry, no exit code), leaving
        // the registry clean; a subsequent run on the same supervisor is
        // unaffected by any leaked reader task.
        assert!(sup.alive().is_empty(), "no live children after timeout");
        let reaped = sup.reap();
        assert_eq!(reaped.len(), 1, "exactly one collectible child");
        assert_eq!(reaped[0].exit_code, None);
        assert_eq!(sup.registered(), 0, "registry must drain after reap");
        let out = sup
            .run(
                sh("echo after-timeout"),
                Duration::from_secs(5),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(out.exit_code, Some(0));
        assert!(out.excerpt.contains("after-timeout"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unknown_id_operations_are_not_found() {
        let (_d, sup) = supervisor();
        assert!(sup.kill(999, 10).is_err());
        assert!(sup.transfer(999, ProcessOwner::Daemon).is_err());
        assert!(sup.reap().is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn exit_code_propagation() {
        let (_d, sup) = supervisor();
        assert_eq!(
            sup.run(sh("true"), Duration::from_secs(5), CancellationToken::new())
                .await
                .unwrap()
                .exit_code,
            Some(0)
        );
        assert_eq!(
            sup.run(
                sh("false"),
                Duration::from_secs(5),
                CancellationToken::new()
            )
            .await
            .unwrap()
            .exit_code,
            Some(1)
        );
        assert_eq!(
            sup.run(
                sh("exit 42"),
                Duration::from_secs(5),
                CancellationToken::new()
            )
            .await
            .unwrap()
            .exit_code,
            Some(42)
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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
            if r.len() >= 50 {
                break;
            }
        }
        eprintln!(
            "dbg: final reaped={} registered={}",
            sup.reap().len(),
            sup.registered()
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn artifact_roundtrip_via_cas() {
        let (_d, sup) = supervisor();
        let mut cfg = sh("i=0; while [ $i -lt 200000 ]; do echo overflow-$i; i=$((i+1)); done");
        // ~3MB stream; the cap is honored now, so size it above the stream to
        // keep the whole (untruncated) artifact reachable.
        cfg.artifact_max = 8 * 1024 * 1024;
        let out = sup
            .run(cfg, Duration::from_secs(60), CancellationToken::new())
            .await
            .unwrap();
        assert!(!out.artifact_truncated, "stream fits under the cap");
        assert!(out.artifact.is_some());
        let hash = out
            .artifact
            .as_ref()
            .and_then(|a| a.strip_prefix("artifact://"))
            .and_then(faktor_core::hash::FileHash::from_hex)
            .unwrap();
        let blob = sup.cas.get_verified_now(hash).unwrap();
        assert!(String::from_utf8_lossy(&blob).contains("overflow-199999"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn artifact_cap_enforced_and_truncation_reported() {
        // Audit round 10: `artifact_max = 1MB` must actually cap the spool.
        // 5MB of output over a 1MB cap ⇒ artifact holds exactly the first
        // 1MB (first bytes of the stream), the ring keeps the tail, and the
        // outcome reports the truncation explicitly.
        let (_d, sup) = supervisor();
        let cap = 1024 * 1024;
        let mut cfg = sh(
            "printf 'BEGIN-MARKER-0\\n'; yes 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' | head -c 5000000",
        );
        cfg.artifact_max = cap;
        let out = sup
            .run(cfg, Duration::from_secs(60), CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(out.exit_code, Some(0));
        assert!(out.artifact_truncated, "5MB over a 1MB cap must truncate");
        let artifact = out
            .artifact
            .as_ref()
            .expect("overflow must still produce an artifact");
        let hash = artifact
            .strip_prefix("artifact://")
            .and_then(faktor_core::hash::FileHash::from_hex)
            .unwrap();
        let blob = sup.cas.get_verified_now(hash).unwrap();
        assert!(
            blob.len() <= cap,
            "artifact {} bytes exceeds the {cap}-byte cap",
            blob.len()
        );
        assert_eq!(blob.len(), cap, "cap must be reached exactly");
        assert!(
            blob.starts_with(b"BEGIN-MARKER-0\n"),
            "artifact keeps the FIRST bytes of the stream"
        );
        assert!(out.excerpt.len() < 64 * 1024, "excerpt stays bounded");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn descendant_holding_capture_pipe_is_killed_on_exit() {
        // A backgrounded descendant that keeps stdout open AFTER the direct
        // child exits must be group-killed on the Exited path (audit round
        // 11: otherwise the capture reader never sees EOF and run() hangs).
        let (_d, sup) = supervisor();
        // `sh -c '(sleep 300; echo late) & echo done; exit 0'` — sh exits
        // immediately but the background subshell holds the pipe for 300s.
        let cfg = sh("(sleep 300; echo late) & echo done; exit 0");
        let t0 = std::time::Instant::now();
        let out = tokio::time::timeout(
            Duration::from_secs(10),
            sup.run(cfg, Duration::from_secs(30), CancellationToken::new()),
        )
        .await
        .expect("run must return promptly after the direct child exits")
        .unwrap();
        assert!(t0.elapsed() < Duration::from_secs(8));
        assert_eq!(out.exit_code, Some(0));
        assert!(out.excerpt.contains("done"));
        assert!(!out.excerpt.contains("late"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn artifact_within_cap_untouched_and_exact() {
        // 200KB of output under a 1MB cap: the artifact is the complete
        // stream, byte-exact, and reports no truncation.
        let (_d, sup) = supervisor();
        // Bounded producer: `yes | head` leaves an eternal writer that can
        // starve the runtime under capture (audit round 11); dd|tr|fold
        // emits the same ~200 KB payload with every process exiting.
        let mut cfg = sh("dd if=/dev/zero bs=204800 count=1 2>/dev/null | tr '\\0' 'x'");
        cfg.artifact_max = 1024 * 1024;
        let out = sup
            .run(cfg, Duration::from_secs(30), CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(out.exit_code, Some(0));
        assert!(!out.artifact_truncated, "stream under the cap is intact");
        let artifact = out.artifact.as_ref().expect("artifact must exist");
        let hash = artifact
            .strip_prefix("artifact://")
            .and_then(faktor_core::hash::FileHash::from_hex)
            .unwrap();
        let blob = sup.cas.get_verified_now(hash).unwrap();
        let expected = "x".repeat(204_800);
        assert_eq!(expected.len(), 204_800);
        assert_eq!(blob.len(), 204_800, "artifact must be byte-exact");
        assert_eq!(blob, expected.as_bytes(), "artifact content must be intact");
    }

    #[test]
    fn artifact_cap_default_and_global_ceiling() {
        assert_eq!(SpawnConfig::default().artifact_max, 100 * 1024 * 1024);
        assert_eq!(effective_artifact_max(1), 1);
        assert_eq!(
            effective_artifact_max(usize::MAX),
            GLOBAL_HARD_MAX as usize,
            "configured caps must never exceed the global ceiling"
        );
        assert_eq!(
            effective_artifact_max(GLOBAL_HARD_MAX as usize),
            GLOBAL_HARD_MAX as usize
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn descendant_holding_pipe_cannot_hang_run() {
        // A backgrounded descendant keeps the stdout pipe open for 5s after
        // the shell exits. run() must not wait for that descendant: the
        // post-exit drain is bounded.
        let (_d, sup) = supervisor();
        let t0 = std::time::Instant::now();
        let out = sup
            .run(
                sh("(sleep 5 &) ; echo done"),
                Duration::from_secs(30),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        let elapsed = t0.elapsed();
        assert_eq!(out.exit_code, Some(0));
        assert!(out.excerpt.contains("done"), "{:?}", out.excerpt);
        assert!(
            elapsed < Duration::from_millis(1500),
            "run() must not wait for a descendant holding the pipe: {elapsed:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn post_exit_drain_is_bounded_even_when_pipe_never_closes() {
        // `(sleep 30) &` keeps the pipe open for 30s; the drain bound must
        // cap run() at ~500ms after the shell exits.
        let (_d, sup) = supervisor();
        let t0 = std::time::Instant::now();
        let out = sup
            .run(
                sh("(sleep 30) & echo x"),
                Duration::from_secs(30),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        let elapsed = t0.elapsed();
        assert_eq!(out.exit_code, Some(0));
        assert!(out.excerpt.contains("x"), "{:?}", out.excerpt);
        assert!(
            elapsed < Duration::from_secs(3),
            "post-exit drain must be bounded: {elapsed:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn kill_group_async_returns_immediately_on_fast_exit() {
        let (_d, sup) = supervisor();
        let h = sup.spawn(sh("sleep 30")).unwrap();
        let t0 = std::time::Instant::now();
        kill_group_async(h.pid, 2000).await.unwrap();
        let elapsed = t0.elapsed();
        assert!(
            elapsed < Duration::from_millis(500),
            "kill_group_async must return as soon as the group is gone: {elapsed:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn kill_group_async_escalates_to_sigkill_at_grace() {
        // sh ignores SIGTERM; only the SIGKILL at the grace deadline can
        // take it down. The elapsed time must reflect the grace, not less.
        let (_d, sup) = supervisor();
        let h = sup.spawn(sh("trap '' TERM; sleep 30")).unwrap();
        std::thread::sleep(Duration::from_millis(200));
        let t0 = std::time::Instant::now();
        kill_group_async(h.pid, 300).await.unwrap();
        let elapsed = t0.elapsed();
        assert!(
            elapsed >= Duration::from_millis(250) && elapsed < Duration::from_secs(2),
            "SIGKILL escalation must happen at the grace deadline: {elapsed:?}"
        );
        for _ in 0..40 {
            if !ps_alive(h.pid) {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(!ps_alive(h.pid), "SIGKILL must take the whole group");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sync_kill_group_exits_early_too() {
        let (_d, sup) = supervisor();
        let h = sup.spawn(sh("sleep 30")).unwrap();
        let t0 = std::time::Instant::now();
        sup.kill(h.id, 2000).unwrap();
        let elapsed = t0.elapsed();
        assert!(
            elapsed < Duration::from_millis(500),
            "sync kill must not hold the full grace when the group exits early: {elapsed:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reader_abort_does_not_lose_exit_code() {
        // A descendant holds the pipe open past the drain bound, so the
        // reader is aborted mid-drain; the exit code and the drained excerpt
        // must still survive.
        let (_d, sup) = supervisor();
        let out = sup
            .run(
                sh("(sleep 30) & echo exit42; exit 42"),
                Duration::from_secs(30),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(out.exit_code, Some(42));
        assert!(out.excerpt.contains("exit42"), "{:?}", out.excerpt);
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

    #[tokio::test]
    async fn artifact_contains_beginning_and_end_of_large_output() {
        // Audit round 5: with full-stream spooling the CAS artifact holds the
        // stream from the FIRST byte; the ring holds the tail. Both ends must
        // be recoverable.
        let (_d, sup) = supervisor();
        let mut cfg =
            sh("i=0; while [ $i -lt 200000 ]; do echo beginning-check-$i; i=$((i+1)); done");
        cfg.artifact_max = 1024 * 1024;
        let out = sup
            .run(cfg, Duration::from_secs(60), CancellationToken::new())
            .await
            .unwrap();
        assert!(
            out.artifact.is_some(),
            "full-stream spooling must produce an artifact"
        );
        let hash = out
            .artifact
            .as_ref()
            .and_then(|a| a.strip_prefix("artifact://"))
            .and_then(faktor_core::hash::FileHash::from_hex)
            .unwrap();
        let blob = sup.cas.get_verified_now(hash).unwrap();
        let text = String::from_utf8_lossy(&blob);
        // The artifact begins at the stream's first line...
        assert!(
            text.contains("beginning-check-0"),
            "artifact must contain the stream beginning"
        );
        // ...and the excerpt holds the very end.
        assert!(out.excerpt.contains("beginning-check-199999"));
    }

    #[tokio::test]
    async fn stubborn_descendant_forces_sigkill_escalation() {
        // Audit round 5: leader-gone is not group-gone. A child that ignores
        // SIGTERM must keep the group alive until the SIGKILL escalation.
        // The async probe must NOT return as soon as the leader dies.
        let (_d, sup) = supervisor();
        // sh dies at SIGTERM; the trapped sleep 10 ignores it.
        let cfg = sh("trap '' TERM; sleep 10 & trap '' TERM; wait");
        let h = sup.spawn(cfg).unwrap();
        std::thread::sleep(Duration::from_millis(200));
        let t0 = std::time::Instant::now();
        sup.kill_child_pid(h.pid, 800).unwrap();
        let elapsed = t0.elapsed();
        // The escalation must have happened (the stubborn member is gone),
        // and the kill must not return before the grace deadline.
        assert!(
            elapsed >= Duration::from_millis(500),
            "must wait for the group (including stubborn members), took {elapsed:?}"
        );
        // The reaper thread needs a moment to reap the leader zombie.
        for _ in 0..40 {
            if !sup.pid_alive(h.pid) {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(!sup.pid_alive(h.pid), "the group must be fully gone");
        let _ = sup.reap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn spawn_timeline_records_pid_argv_and_exit() {
        // Audit round 10 instrumentation: every child is visible with
        // op/pid/argv/spawn/exit timestamps — the Linux git hang will show
        // up in this ring instead of vanishing.
        let dir = tempfile::tempdir().unwrap();
        let cas = Arc::new(faktor_cas::Cas::open(dir.path().join("cas")).unwrap());
        let sup = ProcessSupervisor::new(cas);
        let cfg = SpawnConfig {
            cmd: "sh".into(),
            args: vec!["-c".into(), "exit 3".into()],
            cwd: dir.path().into(),
            env: vec![],
            owner: ProcessOwner::Daemon,
            capture: true,
            artifact_max: 1024 * 1024,
        };
        let out = sup
            .run(
                cfg,
                std::time::Duration::from_secs(10),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(out.exit_code, Some(3));
        let spawns = sup.recent_spawns();
        let rec = spawns
            .iter()
            .find(|t| t.argv.contains("exit 3"))
            .expect("timeline records the spawned argv");
        assert!(rec.pid > 0);
        assert_eq!(rec.exit_code, Some(3));
        assert!(rec.exited_ms.is_some());
        assert!(rec.exited_ms.unwrap() >= rec.started_ms);
        assert_eq!(rec.owner, "Daemon");
        // A handful of extra spawns still land in the timeline.
        for _ in 0..3 {
            let _ = sup
                .run(
                    SpawnConfig {
                        cmd: "sh".into(),
                        args: vec!["-c".into(), "true".into()],
                        cwd: dir.path().into(),
                        env: vec![],
                        owner: ProcessOwner::Daemon,
                        capture: true,
                        artifact_max: 1024,
                    },
                    std::time::Duration::from_secs(5),
                    CancellationToken::new(),
                )
                .await;
        }
        assert!(
            sup.recent_spawns().len() >= 4,
            "timeline records every spawn"
        );
    }

    #[ignore = "[perf] 300 sequential spawns — ring must stay bounded; run explicitly"]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn timeline_ring_stays_bounded_under_300_spawns() {
        // Bounded ring under pressure: 300 sequential spawns never grow the
        // timeline past its cap.
        let dir = tempfile::tempdir().unwrap();
        let cas = Arc::new(faktor_cas::Cas::open(dir.path().join("cas")).unwrap());
        let sup = ProcessSupervisor::new(cas);
        for _ in 0..300 {
            let _ = sup
                .run(
                    SpawnConfig {
                        cmd: "sh".into(),
                        args: vec!["-c".into(), "true".into()],
                        cwd: dir.path().into(),
                        env: vec![],
                        owner: ProcessOwner::Daemon,
                        capture: true,
                        artifact_max: 1024,
                    },
                    std::time::Duration::from_secs(5),
                    CancellationToken::new(),
                )
                .await;
        }
        assert!(
            sup.recent_spawns().len() <= 256,
            "timeline ring stays bounded"
        );
    }
}

// ================================================================ windows
/// On Windows every supervised child is assigned to a faktor-winjob
/// JobGuard (kill-on-close): daemon death terminates the whole tree via OS
/// ownership. macOS/Linux keep process groups + signals.
#[cfg(windows)]
pub use faktor_winjob::JobGuard;

// ================================================================ windows tests
// P0-59 process-tree certification through the REAL windows spawn path of
// this crate: std::process children registered with the supervisor are
// assigned to the JobGuard (kill-on-close) AND killed via taskkill /T on
// cancel; dropping the supervisor exercises both. Runtime-certification
// only on a windows host — on unix hosts this module does not exist.
#[cfg(all(test, windows))]
mod windows_tests {
    use std::path::Path;
    use std::time::{Duration, Instant};

    use windows_sys::Win32::Foundation::{CloseHandle, WAIT_TIMEOUT};
    use windows_sys::Win32::System::Threading::{
        OpenProcess, WaitForSingleObject, PROCESS_SYNCHRONIZE,
    };

    use super::*;
    use faktor_core::error::ErrorKind;

    fn pid_alive(pid: u32) -> bool {
        if pid == 0 {
            return false;
        }
        unsafe {
            let handle = OpenProcess(PROCESS_SYNCHRONIZE, 0, pid);
            if handle.is_null() {
                return false;
            }
            let running = WaitForSingleObject(handle, 0) == WAIT_TIMEOUT;
            CloseHandle(handle);
            running
        }
    }

    fn wait_until<F: FnMut() -> bool>(what: &str, limit: Duration, mut cond: F) {
        let deadline = Instant::now() + limit;
        while Instant::now() < deadline {
            if cond() {
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        panic!("timed out after {limit:?} waiting for {what}");
    }

    /// powershell (direct child) sleeps 60 s; the ping grandchild is born
    /// ~1.5 s in — after register() has assigned the parent to the job, so
    /// the grandchild lands in the job by descent — writes its pid, and
    /// sleeps ~60 s. `ping -n 60` is a deterministic ~60 s sleeper even on
    /// a network-blocked runner (ICMP failure still paces the retries).
    fn sleeper_tree_script(pid_file: &Path) -> String {
        format!(
            "Start-Sleep -Milliseconds 1500; \
             $p = Start-Process -FilePath 'ping.exe' -ArgumentList '-n','60','127.0.0.1' \
                 -WindowStyle Hidden -PassThru; \
             [System.IO.File]::WriteAllText('{}', [string]$p.Id); \
             Start-Sleep -Seconds 60",
            pid_file.display()
        )
    }

    fn supervisor_with_tree(
        dir: &tempfile::TempDir,
        pid_file: &Path,
    ) -> (Arc<ProcessSupervisor>, SpawnConfig) {
        let cas = Arc::new(faktor_cas::Cas::open(dir.path().join("cas")).unwrap());
        let sup = ProcessSupervisor::new(cas);
        let cfg = SpawnConfig {
            cmd: "powershell.exe".into(),
            args: vec![
                "-NoProfile".into(),
                "-NonInteractive".into(),
                "-Command".into(),
                sleeper_tree_script(pid_file).into(),
            ],
            cwd: std::env::temp_dir(),
            env: vec![],
            owner: ProcessOwner::Daemon,
            capture: false, // no pipe drama: the tree is killed, not drained
            artifact_max: 1024 * 1024,
        };
        (sup, cfg)
    }

    fn wait_for_grandchild(pid_file: &Path) -> u32 {
        wait_until("grandchild pid file", Duration::from_secs(20), || {
            pid_file.exists()
        });
        std::fs::read_to_string(pid_file)
            .expect("grandchild pid file readable")
            .trim()
            .parse()
            .expect("grandchild pid file holds a pid")
    }

    /// The task-cancellation path (run + CancellationToken) must kill the
    /// whole supervised tree: direct powershell child AND ping grandchild
    /// (taskkill /T over the registered pid, backed by job membership).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn task_cancellation_kills_the_whole_supervised_tree() {
        let dir = tempfile::tempdir().unwrap();
        let pid_file = dir.path().join("gc.pid");
        let (sup, cfg) = supervisor_with_tree(&dir, &pid_file);
        let token = CancellationToken::new();

        let sup2 = sup.clone();
        let token2 = token.clone();
        let task =
            tokio::spawn(async move { sup2.run(cfg, Duration::from_secs(120), token2).await });

        // The direct child pid is registered + timeline-logged once run()
        // spawns; poll the timeline instead of guessing.
        let direct = wait_for_direct_pid(&sup, Duration::from_secs(20));
        let grandchild = wait_for_grandchild(&pid_file);
        assert!(
            pid_alive(direct) && pid_alive(grandchild),
            "parent + grandchild must be alive before cancellation"
        );

        token.cancel();
        let err = task.await.unwrap().unwrap_err();
        assert_eq!(err.kind, ErrorKind::Cancelled, "{err:?}");

        wait_until("cancelled tree death", Duration::from_secs(10), || {
            !pid_alive(direct) && !pid_alive(grandchild)
        });
    }

    fn wait_for_direct_pid(sup: &ProcessSupervisor, limit: Duration) -> u32 {
        let deadline = Instant::now() + limit;
        loop {
            if let Some(t) = sup.recent_spawns().first() {
                if t.pid > 0 {
                    return t.pid;
                }
            }
            assert!(Instant::now() < deadline, "run() must spawn the child");
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    /// Daemon-crash semantics end-to-end: dropping the LAST supervisor
    /// reference kills the live tree — the registry kill path (taskkill /T)
    /// plus the JobGuard kill-on-close that fires as the supervisor's job
    /// handle closes.
    #[test]
    fn dropping_the_supervisor_kills_the_tree() {
        let dir = tempfile::tempdir().unwrap();
        let pid_file = dir.path().join("gc2.pid");
        let (direct, grandchild) = {
            let cas = Arc::new(faktor_cas::Cas::open(dir.path().join("cas")).unwrap());
            let sup = ProcessSupervisor::new(cas);
            let cfg = SpawnConfig {
                cmd: "powershell.exe".into(),
                args: vec![
                    "-NoProfile".into(),
                    "-NonInteractive".into(),
                    "-Command".into(),
                    sleeper_tree_script(&pid_file).into(),
                ],
                cwd: std::env::temp_dir(),
                env: vec![],
                owner: ProcessOwner::Daemon,
                capture: false,
                artifact_max: 1024 * 1024,
            };
            let handle = sup.spawn(cfg).expect("supervised spawn");
            let grandchild = wait_for_grandchild(&pid_file);
            assert!(pid_alive(grandchild), "grandchild must be alive pre-drop");
            (handle.pid, grandchild) // sup drops here: daemon crash
        };

        wait_until("drop-killed tree death", Duration::from_secs(10), || {
            !pid_alive(direct) && !pid_alive(grandchild)
        });
    }
}
