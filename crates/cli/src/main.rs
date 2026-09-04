//! faktor-cli — `serve`, `run`, `doctor`, `sessions`, `acp` (spec §34, §42,
//! §43, and the ACP agent server over the daemon).
//!
//! `serve --port 0` prints the exact frozen startup line
//! `faktor server listening on http://127.0.0.1:<port>` so the frozen v7.5.6
//! extension connects exactly as it did to the old CLI. Nothing else goes to
//! stdout. Auth comes from the frontend-generated `FAKTOR_SERVER_PASSWORD`
//! environment variable; the daemon never prints it.

use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use evidence::RepoEvidence;
use faktor_acp::{AcpBackend, AcpServer};
use faktor_agent::{AgentDeps, AgentRuntime, ToolCallMode, ToolRegistry};
use faktor_core::id::SessionId;
use faktor_core::time::SystemClock;
use faktor_provider::{Provider, ProviderRegistry};
use faktor_server::permission::ChannelPermissionRequester;
use faktor_server::{ServerDeps, ServerPassword};
use faktor_session::SessionManager;
use serde_json::{json, Value};

mod config;
mod evidence;
mod mcp_bridge;
mod tools;

#[derive(Parser)]
#[command(
    name = "faktor-plus",
    version,
    about = "Faktor — same Kilo UX, native Rust engine"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Start the daemon (frozen `kilo serve --port 0` compatible).
    Serve {
        #[arg(long, default_value_t = 0)]
        port: u16,
        #[arg(long, default_value = "~/.faktor")]
        data_dir: String,
        #[arg(long)]
        config: Option<String>,
    },
    /// Headless: create a session and run one prompt.
    Run {
        prompt: String,
        #[arg(long, default_value = "fake")]
        provider: String,
        #[arg(long, default_value = "default")]
        model: String,
        #[arg(long, default_value = ".")]
        workspace: String,
        #[arg(long, default_value = "~/.faktor")]
        data_dir: String,
    },
    /// Self-check: storage, CAS, permissions, providers.
    Doctor {
        #[arg(long, default_value = "~/.faktor")]
        data_dir: String,
        /// Run the full deep scan: complete store integrity check, CAS blob
        /// verification, global recovery-row scan, dangling CAS references
        /// and journal projection consistency. Plain mode keeps the bounded
        /// quick checks.
        #[arg(long)]
        deep: bool,
    },
    /// ACP (Agent Client Protocol) stdio agent server over the real daemon
    /// graph. Framed JSON-RPC on stdout ONLY; logs stay on stderr.
    Acp {
        #[arg(long, default_value = "~/.faktor")]
        data_dir: String,
    },
    /// List sessions.
    Sessions {
        #[arg(long, default_value = "~/.faktor")]
        data_dir: String,
    },
}

fn expand(p: &str) -> PathBuf {
    if p == "~" {
        return std::env::home_dir().unwrap_or_else(|| PathBuf::from("."));
    }
    if let Some(rest) = p.strip_prefix("~/") {
        return std::env::home_dir()
            .map(|h| h.join(rest))
            .unwrap_or_else(|| PathBuf::from(p));
    }
    PathBuf::from(p)
}

#[tokio::main]
async fn main() {
    // Logging goes to stderr: stdout is the frozen startup-line contract.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let cli = Cli::parse();
    match cli.command {
        Command::Serve {
            port,
            data_dir,
            config,
        } => {
            serve(port, expand(&data_dir), config.map(|c| expand(&c))).await;
        }
        Command::Run {
            prompt,
            provider,
            model,
            workspace,
            data_dir,
        } => {
            run(
                prompt,
                &provider,
                &model,
                expand(&workspace),
                expand(&data_dir),
            )
            .await;
        }
        Command::Doctor { data_dir, deep } => {
            doctor(expand(&data_dir), deep).await;
        }
        Command::Acp { data_dir } => {
            acp(expand(&data_dir)).await;
        }
        Command::Sessions { data_dir } => {
            sessions(expand(&data_dir)).await;
        }
    }
}

/// The daemon dependency graph (session, agent, permissions) plus the
/// supervised MCP servers whose tools ride the registry.
pub type DaemonGraph = (
    Arc<SessionManager>,
    Arc<AgentRuntime>,
    Arc<ChannelPermissionRequester>,
    Vec<Arc<faktor_mcp::McpServer>>,
);

/// Build the full daemon dependency graph (providers, tools, session,
/// agent, permissions). The real filesystem stack is wired here: workspace
/// service, transactional edit engine, CAS-backed checkpoints, sandbox
/// policy engine, and the process supervisor.
pub fn build_daemon(
    data_dir: &std::path::Path,
    config: Option<config::Config>,
) -> Result<DaemonGraph, String> {
    let config = config.unwrap_or_default();
    std::fs::create_dir_all(data_dir).map_err(|e| e.to_string())?;
    let session = SessionManager::open(data_dir.join("store"), data_dir.join("cas"), true)
        .map_err(|e| e.to_string())?;
    build_daemon_on(session, config, vec![])
}

/// Async daemon build with the MCP layer (spec §31): configured servers are
/// spawned supervised BEFORE the agent is constructed so their dynamic
/// tools land in the registry next to the builtins (name collisions never
/// overwrite a builtin). A server that fails to connect is a loud warning,
/// not a daemon failure — the rest of the daemon still serves.
pub async fn build_daemon_with_mcp(
    data_dir: &std::path::Path,
    config: Option<config::Config>,
) -> Result<DaemonGraph, String> {
    build_daemon_with_mcp_and_chunks(data_dir, config, None).await
}

pub async fn build_daemon_with_mcp_and_chunks(
    data_dir: &std::path::Path,
    config: Option<config::Config>,
    chunk_tx: Option<std::sync::Arc<faktor_agent::ChunkSink>>,
) -> Result<DaemonGraph, String> {
    build_daemon_with_mcp_inner(data_dir, config, chunk_tx, false).await
}

/// Fast-start variant of [`build_daemon_with_mcp_and_chunks`]: the store is
/// opened with the bounded quick check (`SessionManager::open_quick`) instead
/// of the full integrity scan. `serve` — the production normal start — uses
/// this (audit 43); the deep scan lives under `doctor --deep` and crash
/// forensics. WAL recovery and migrations are NEVER skipped by the fast
/// path.
async fn build_daemon_with_mcp_and_chunks_fast(
    data_dir: &std::path::Path,
    config: Option<config::Config>,
    chunk_tx: Option<std::sync::Arc<faktor_agent::ChunkSink>>,
) -> Result<DaemonGraph, String> {
    build_daemon_with_mcp_inner(data_dir, config, chunk_tx, true).await
}

async fn build_daemon_with_mcp_inner(
    data_dir: &std::path::Path,
    config: Option<config::Config>,
    chunk_tx: Option<std::sync::Arc<faktor_agent::ChunkSink>>,
    fast_open: bool,
) -> Result<DaemonGraph, String> {
    let config = config.unwrap_or_default();
    let entries = config.mcp_servers()?;
    std::fs::create_dir_all(data_dir).map_err(|e| e.to_string())?;
    let session = if fast_open {
        SessionManager::open_quick(data_dir.join("store"), data_dir.join("cas"))
    } else {
        SessionManager::open(data_dir.join("store"), data_dir.join("cas"), true)
    }
    .map_err(|e| e.to_string())?;
    // Spawn the servers first so the agent registry can see their tools.
    let mut servers: Vec<Arc<faktor_mcp::McpServer>> = Vec::new();
    let mut mcp_tools: Vec<faktor_agent::Tool> = Vec::new();
    for entry in entries {
        let cfg = faktor_mcp::McpConfig {
            name: entry.name.clone(),
            command: entry.command,
            args: entry.args,
            env: vec![],
        };
        // Each server gets its own supervisor rooted at the same CAS; the
        // McpServer holds the Arc so children live for the daemon lifetime.
        let supervisor = faktor_terminal::ProcessSupervisor::new(session.cas());
        match tokio::time::timeout(
            std::time::Duration::from_secs(10),
            faktor_mcp::McpServer::connect(cfg, supervisor),
        )
        .await
        {
            Ok(Ok(server)) => {
                let name = server.name().to_string();
                match server.list_tools().await {
                    Ok(tools) => {
                        let n = tools.len();
                        for t in tools {
                            let tool = mcp_bridge::mcp_tool(server.clone(), &t);
                            mcp_tools.push(tool);
                        }
                        tracing::info!("mcp server {name}: {n} tool(s) wired");
                    }
                    Err(e) => {
                        tracing::warn!("mcp server {name}: tool listing failed: {e}");
                    }
                }
                servers.push(server);
            }
            Ok(Err(e)) => {
                tracing::warn!("mcp server {} failed to connect: {e}", entry.name);
            }
            Err(_) => {
                tracing::warn!("mcp server {} connect timed out after 10s", entry.name);
            }
        }
    }
    // Now build the core graph on the SAME store with the MCP tools.
    let config = config::Config {
        providers: config.providers,
        model: config.model,
        compaction_model: config.compaction_model,
        compact_at_usage: config.compact_at_usage,
        instructions: config.instructions,
        mcp: vec![],
    };
    let graph = build_daemon_on_with_sink(session, config, mcp_tools, chunk_tx)?;
    let (session, agent, permissions, _) = graph;
    Ok((session, agent, permissions, servers))
}

/// Bounded wall cap for ONE verification check through the supervisor
/// (the agent hook additionally bounds the whole batch).
const VERIFY_CMD_SECS: u64 = 30;

/// Maximum FAKTOR_HOOKS entries honored (bounding the env surface).
const MAX_ENV_HOOKS: usize = 8;

/// Parse the optional `FAKTOR_HOOKS` env into hook specs (pure fn, unit
/// tested). Format: semicolon-separated entries `event:command [args...]`;
/// the event is the snake_case `faktor_hooks::HookEvent` name (`pre_tool`,
/// `post_tool`, `task_complete`, …). Bounds: at most [`MAX_ENV_HOOKS`]
/// entries, ids are `env-N`. Every parsed spec runs with an env allowlist
/// (only `FAKTOR_HOOK_INPUT` passes through — use absolute command paths)
/// and the default FailClosed failure policy. Malformed entries (no colon,
/// unknown event, empty command) are warned about and skipped; a hostile
/// env can never panic or unboundedly grow the registry.
pub fn parse_hooks_env(raw: &str) -> Vec<faktor_hooks::HookSpec> {
    let mut out: Vec<faktor_hooks::HookSpec> = Vec::new();
    for entry in raw.split(';') {
        if out.len() >= MAX_ENV_HOOKS {
            tracing::warn!(
                "FAKTOR_HOOKS: at most {MAX_ENV_HOOKS} hooks are honored; ignoring the rest"
            );
            break;
        }
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let (event_name, rest) = match entry.split_once(':') {
            Some(v) => v,
            None => {
                tracing::warn!(
                    "FAKTOR_HOOKS: skipping malformed entry {entry:?} (expected event:command)"
                );
                continue;
            }
        };
        let event: faktor_hooks::HookEvent = match serde_json::from_value(
            serde_json::Value::String(event_name.trim().to_string()),
        ) {
            Ok(e) => e,
            Err(_) => {
                tracing::warn!(
                    "FAKTOR_HOOKS: skipping entry {entry:?}: unknown event {event_name:?}"
                );
                continue;
            }
        };
        let mut parts = rest.split_whitespace();
        let command = match parts.next() {
            Some(c) if !c.is_empty() => c.to_string(),
            _ => {
                tracing::warn!("FAKTOR_HOOKS: skipping entry {entry:?}: empty command");
                continue;
            }
        };
        let args: Vec<String> = parts.map(str::to_string).collect();
        out.push(faktor_hooks::HookSpec {
            id: format!("env-{}", out.len()),
            events: vec![event],
            command,
            args,
            env_allowlist: true,
            ..Default::default()
        });
    }
    out
}

/// Build the optional lifecycle-hook registry from `FAKTOR_HOOKS` (daemon
/// build time). Each parsed spec is logged; a spec the registry rejects is
/// a loud warning, never a daemon failure.
fn env_hook_registry() -> Option<Arc<faktor_hooks::HookRegistry>> {
    let specs = parse_hooks_env(&std::env::var("FAKTOR_HOOKS").unwrap_or_default());
    if specs.is_empty() {
        return None;
    }
    let registry = Arc::new(faktor_hooks::HookRegistry::new());
    for spec in specs {
        tracing::info!(
            "hook {}: {:?} -> {} {}",
            spec.id,
            spec.events,
            spec.command,
            spec.args.join(" ")
        );
        if let Err(e) = registry.register(spec) {
            tracing::warn!("FAKTOR_HOOKS: hook rejected: {e}");
        }
    }
    Some(registry)
}

/// Default workspace root determinable from the daemon config for the lazy
/// instructions loader (audit 31). Session workspace roots are per-session
/// and unknown at daemon build; the only single default root would come
/// from a `workspace`/`root` key in the config FILE shape — the file shape
/// today carries none, so no root is determinable and this is None. When
/// the config gains such a field, return it here and the loader below wires
/// `Instructions::load` at it.
fn config_default_root(_config: &config::Config) -> Option<PathBuf> {
    None
}

/// The lazy repository-instructions loader for the daemon graph: `Some`
/// only when a single default root is determinable from the config (see
/// [`config_default_root`]). Otherwise `None`, plus a one-line note —
/// loading repository rules at the wrong root would silently misapply them
/// to every session, and per-session root wiring belongs to the runtime.
fn daemon_instructions_loader(
    config: &config::Config,
) -> Option<Arc<faktor_instructions::Instructions>> {
    match config_default_root(config) {
        Some(root) => {
            tracing::info!("repository instructions loaded from {}", root.display());
            Some(Arc::new(faktor_instructions::Instructions::load(&root)))
        }
        None => {
            tracing::info!(
                "config has no workspace root: repository instructions loader not wired at daemon build (roots are per-session)"
            );
            None
        }
    }
}

/// Run one derived verification command through the process supervisor via
/// `sh -c` (the check commands are single shell strings). Ok only on exit
/// code 0; spawn errors, timeouts and non-zero exits are Err. The async
/// supervisor runs on its own thread+runtime because the verifier closure
/// is synchronous.
fn supervised_verify(
    supervisor: Arc<faktor_terminal::ProcessSupervisor>,
    cwd: std::path::PathBuf,
    command: &str,
) -> Result<(), String> {
    let command = command.to_string();
    let worker = std::thread::Builder::new()
        .name("faktor-verify".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| format!("verification runtime: {e}"))?;
            rt.block_on(async move {
                let cfg = faktor_terminal::SpawnConfig {
                    cmd: "sh".into(),
                    args: vec!["-c".into(), command.clone()],
                    cwd,
                    owner: faktor_terminal::ProcessOwner::Daemon,
                    capture: false,
                    artifact_max: 1024 * 1024,
                    ..Default::default()
                };
                let deadline = std::time::Duration::from_secs(VERIFY_CMD_SECS);
                let token = faktor_core::cancellation::CancellationToken::new();
                match supervisor.run(cfg, deadline, token).await {
                    Ok(out) if out.exit_code == Some(0) => Ok(()),
                    Ok(out) => Err(format!(
                        "verification command exited {:?}: {command}",
                        out.exit_code
                    )),
                    Err(e) => Err(format!("verification command failed to run: {e}")),
                }
            })
        })
        .map_err(|e| format!("verification worker spawn: {e}"))?;
    worker
        .join()
        .map_err(|_| "verification worker panicked".to_string())?
}

/// Shared core: identical to [`build_daemon`] but registers `extra_tools`
/// (MCP tools) after the builtins on the GIVEN already-open store — a
/// collision never replaces a builtin.
fn build_daemon_on(
    session: Arc<SessionManager>,
    config: config::Config,
    extra_tools: Vec<faktor_agent::Tool>,
) -> Result<DaemonGraph, String> {
    build_daemon_on_with_sink(session, config, extra_tools, None)
}

fn build_daemon_on_with_sink(
    session: Arc<SessionManager>,
    config: config::Config,
    extra_tools: Vec<faktor_agent::Tool>,
    chunk_tx: Option<std::sync::Arc<faktor_agent::ChunkSink>>,
) -> Result<DaemonGraph, String> {
    let mut tools = ToolRegistry::new();
    tools.register(tools::read_file_tool());
    tools.register(tools::write_file_tool());
    tools.register(tools::edit_file_tool());
    tools.register(tools::search_tool());
    tools.register(tools::run_command_tool());
    for t in extra_tools {
        if tools.names().contains(&t.name) {
            tracing::warn!(
                "mcp tool {} collides with a builtin; the builtin wins",
                t.name
            );
            continue;
        }
        tools.register(t);
    }
    // Lazy repository instructions (audit 31): wired only when the config
    // determines a single default workspace root; otherwise None (per-session
    // roots are runtime wiring, outside this build seam). Computed before
    // `config.providers` is consumed below so the whole config is borrowable.
    let instructions_loader = daemon_instructions_loader(&config);
    let mut providers = ProviderRegistry::new();
    let mut ollama_warmers: Vec<Arc<faktor_ollama::OllamaProvider>> = Vec::new();
    for p in config.providers {
        // Ollama providers are kept CONCRETE for live probing (spec §10):
        // warm-up must reach the instance the registry serves.
        if let Some(ollama) = p.build_ollama() {
            let dyn_arc: Arc<dyn Provider> = ollama.clone();
            providers.register(dyn_arc);
            ollama_warmers.push(ollama);
            continue;
        }
        match p.build() {
            Ok(provider) => providers.register(provider),
            Err(e) => tracing::warn!("provider {} failed to build: {e}", p.id()),
        }
    }
    let cas = session.cas();
    let store = session.store();
    let workspaces = faktor_fs::WorkspaceFileService::new();
    let edit = Arc::new(faktor_edit::EditEngine::new(workspaces.clone()));
    let snapshots = Arc::new(faktor_snapshot::CheckpointStore::new(cas.clone(), store));
    let sandbox = Arc::new(faktor_sandbox::PermissionEngine::new(
        faktor_sandbox::SandboxPolicy::default(),
        None,
    ));
    let supervisor = faktor_terminal::ProcessSupervisor::new(cas.clone());
    let permissions = ChannelPermissionRequester::new(std::time::Duration::from_secs(300));
    // The verification engine runs REQUIRED checks the agent derived from
    // its own file changes through this supervisor-backed closure. Checks
    // run as `sh -c` in the daemon's working directory (the CLI convention:
    // the daemon cwd is the served workspace root).
    let verifier = {
        let supervisor = supervisor.clone();
        let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("/"));
        Arc::new(faktor_verify::Verifier::new(Arc::new(move |cmd: &str| {
            supervised_verify(supervisor.clone(), cwd.clone(), cmd)
        })))
    };
    // Lifecycle hooks (audit): optional FAKTOR_HOOKS env, parsed by the
    // bounded pure `parse_hooks_env` at daemon build time.
    let hooks = env_hook_registry();
    let agent = AgentRuntime::new(AgentDeps {
        session: session.clone(),
        providers: Arc::new(providers),
        chunk_sink: chunk_tx,
        permission_requester: permissions.clone(),
        evidence: Arc::new(RepoEvidence::new(session.clone())),
        tools: Arc::new(tools),
        cas: Some(cas),
        workspaces,
        edit: Some(edit),
        snapshots: Some(snapshots),
        sandbox: Some(sandbox),
        supervisor: Some(supervisor),
        verifier: Some(verifier),
        hooks,
        instructions_loader,
        router: None,
        budget_micro: None,
        model: config.model.clone(),
        compaction_model: config.compaction_model,
        compact_at_usage: config.compact_at_usage,
        instructions: config.instructions,
        clock: Arc::new(SystemClock),
        tool_call_mode: ToolCallMode::NativeWithRepair,
        tool_deadline_ms: 30_000,
        retry_policy: faktor_core::retry::RetryPolicy::default(),
    })
    .map_err(|e| e.to_string())?;
    for ollama in ollama_warmers {
        warm_ollama(ollama);
    }
    Ok((session, agent, permissions, vec![]))
}

/// Automatic-backup interval (audit 44): at most one snapshot per
/// `BACKUP_MIN_INTERVAL_SECS` of wall time — unless the newest backup no
/// longer matches the store's size, which means the store changed since the
/// snapshot was taken (a crash-recovery run counts).
const BACKUP_MIN_INTERVAL_SECS: u64 = 3600;
/// Retention quota (spec §24): keep at most this many complete backups…
const BACKUP_MAX_FILES: usize = 8;
/// …and at most this many bytes across the whole backups directory
/// (drop the oldest while either bound is exceeded).
const BACKUP_MAX_TOTAL_BYTES: u64 = 4 * 1024 * 1024 * 1024;
/// Post-readiness delay before the startup backup task acts: the daemon is
/// announced and accepting connections well before any snapshot work starts.
const BACKUP_START_DELAY: std::time::Duration = std::time::Duration::from_millis(300);

/// Every COMPLETE backup under `<data_dir>/backups` (`faktor-plus-*.db`),
/// newest by mtime first. In-progress snapshots write under a `.db.tmp-*`
/// name and are renamed into place only when complete, so they are invisible
/// here by construction.
fn list_backups(data_dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let backups = data_dir.join("backups");
    let Ok(files) = std::fs::read_dir(&backups) else {
        return Vec::new();
    };
    let mut out: Vec<std::path::PathBuf> = files
        .flatten()
        .map(|f| f.path())
        .filter(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            name.starts_with("faktor-plus-") && name.ends_with(".db")
        })
        .collect();
    out.sort_by_key(|p| {
        std::fs::metadata(p)
            .and_then(|m| m.modified())
            .unwrap_or(std::time::UNIX_EPOCH)
    });
    out.reverse();
    out
}

/// Interval + staleness gate: the startup backup is due when no backup
/// exists, when the newest is older than [`BACKUP_MIN_INTERVAL_SECS`], or
/// when the newest no longer matches the store file's size (the daemon
/// wrote since it was taken).
fn backup_due(data_dir: &std::path::Path) -> bool {
    let db_path = data_dir.join("store").join("faktor-plus.db");
    let Ok(db_meta) = std::fs::metadata(&db_path) else {
        return false;
    };
    let Some(newest) = list_backups(data_dir).into_iter().next() else {
        return true;
    };
    let meta = std::fs::metadata(&newest).ok();
    let stale = meta
        .as_ref()
        .and_then(|m| m.modified().ok())
        .map(|m| {
            m.elapsed()
                .map(|e| e >= std::time::Duration::from_secs(BACKUP_MIN_INTERVAL_SECS))
                .unwrap_or(true)
        })
        .unwrap_or(true);
    let resized = meta.map(|m| m.len()).unwrap_or(0) != db_meta.len();
    stale || resized
}

/// Remove interrupted-backup temp files older than an hour (a crashed writer
/// can leave them behind; live writers are always younger). Best effort.
fn sweep_stale_backup_tmp(backups: &std::path::Path) {
    let Ok(files) = std::fs::read_dir(backups) else {
        return;
    };
    for f in files.flatten() {
        let p = f.path();
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if !name.contains(".db.tmp-") || !older_than(&p, std::time::Duration::from_secs(3600)) {
            continue;
        }
        let _ = std::fs::remove_file(&p);
    }
}

/// True when the file's mtime is at least `age` in the past (missing or
/// unreadable files are never "stale": fail closed).
fn older_than(p: &std::path::Path, age: std::time::Duration) -> bool {
    std::fs::metadata(p)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|m| m.elapsed().ok())
        .map(|e| e > age)
        .unwrap_or(false)
}

/// Online backup with rotation (spec §24): one crash-safe snapshot per
/// daemon start the interval gate admits; retention keeps the newest
/// [`BACKUP_MAX_FILES`] and never more than [`BACKUP_MAX_TOTAL_BYTES`] total.
/// The snapshot is written to a `.db.tmp-*` name and renamed into place, so
/// a crash mid-backup can never leave a partial file that reads as a
/// complete backup (and the gate/retention scans never see one). Best
/// effort — a backup failure never stops the daemon.
fn rotate_backup(store: &faktor_store::Store, data_dir: &std::path::Path) {
    let backups = data_dir.join("backups");
    if std::fs::create_dir_all(&backups).is_err() {
        return;
    }
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let dest = backups.join(format!("faktor-plus-{ts}.db"));
    let tmp = backups.join(format!("faktor-plus-{ts}.db.tmp-{}", std::process::id()));
    if let Err(e) = store.backup_to(&tmp) {
        let _ = std::fs::remove_file(&tmp);
        tracing::warn!("automatic backup failed: {e}");
        return;
    }
    if let Err(e) = std::fs::rename(&tmp, &dest) {
        let _ = std::fs::remove_file(&tmp);
        tracing::warn!("automatic backup finalize failed: {e}");
        return;
    }
    tracing::info!("automatic backup written to {}", dest.display());
    // Retention quota: drop the OLDEST files while the count exceeds
    // BACKUP_MAX_FILES or the total bytes exceed BACKUP_MAX_TOTAL_BYTES.
    // The just-written snapshot is newest and never a candidate.
    let files = list_backups(data_dir);
    let mut total: u64 = files
        .iter()
        .filter_map(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .sum();
    let mut kept = files.len();
    for victim in files.iter().rev() {
        let over_count = kept > BACKUP_MAX_FILES;
        let over_bytes = total > BACKUP_MAX_TOTAL_BYTES && kept > 1;
        if !over_count && !over_bytes {
            break;
        }
        if let Ok(m) = std::fs::metadata(victim) {
            total = total.saturating_sub(m.len());
        }
        if std::fs::remove_file(victim).is_err() {
            break;
        }
        kept -= 1;
    }
    // Opportunistic sweep of interrupted-writer debris from crashed runs.
    sweep_stale_backup_tmp(&backups);
}

/// Live capability warm-up for one Ollama provider (spec §10): the
/// concrete Arc is owned by the spawned thread, so probing reaches the
/// SAME instance the registry serves. Best-effort, never blocks.
fn warm_ollama(ollama: Arc<faktor_ollama::OllamaProvider>) {
    std::thread::spawn(move || {
        let rt = match tokio::runtime::Runtime::new() {
            Ok(rt) => rt,
            Err(_) => return,
        };
        match rt.block_on(ollama.refresh_from_live()) {
            Ok(n) => {
                tracing::info!("ollama: probed {n} model(s) from live discovery");
            }
            Err(e) => {
                tracing::warn!("ollama warm-up failed (defaults stay): {e}");
            }
        }
    });
}

/// Config resolution for `serve` (audit 31): an EXPLICIT --config path must
/// load STRICTLY (parse + semantic validation) — any failure is an Err the
/// caller turns into a startup error (exit 1); the daemon never boots on a
/// config it cannot fully honor. Without --config, defaults + best-effort
/// discovery stay lenient and nothing here can fail startup.
fn serve_config(config_path: Option<PathBuf>) -> Result<config::Config, String> {
    match config_path {
        Some(path) => config::Config::load_strict(&path)
            .map_err(|e| format!("config {}: {e}", path.display())),
        None => Ok(config::Config::default()),
    }
}

async fn serve(port: u16, data_dir: PathBuf, config_path: Option<PathBuf>) {
    if let Err(e) = serve_impl(port, data_dir, config_path, None, None).await {
        tracing::error!("{e}");
        std::process::exit(1);
    }
}

/// Shared daemon serve core (audit 44 ordering): `agent.recover()` -> bind
/// -> print the frozen startup line -> spawn the gated backup task. A backup
/// can NEVER delay readiness: the task is spawned only after the startup
/// line, waits [`BACKUP_START_DELAY`], and is gated by policy.
///
/// `ready_tx` fires right after the startup line is printed (test probe for
/// the "startup line before any backup file exists" ordering guarantee);
/// `shutdown_rx`, when present, ends the daemon and ABORTS the backup task
/// first. Production passes neither and then runs until killed, exactly like
/// the historic `std::future::pending()` tail.
#[allow(clippy::too_many_arguments)]
async fn serve_impl(
    port: u16,
    data_dir: PathBuf,
    config_path: Option<PathBuf>,
    ready_tx: Option<tokio::sync::oneshot::Sender<()>>,
    shutdown_rx: Option<tokio::sync::oneshot::Receiver<()>>,
) -> Result<(), String> {
    let config = match serve_config(config_path) {
        Ok(cfg) => cfg,
        Err(e) => return Err(format!("config error: {e}")),
    };
    // Live chunk path (audit 41): BOUNDED channel (1024 events) + sink-side
    // coalescing under backpressure — a slow SSE consumer can never grow
    // the agent's memory. The drainer spawn lives in serve().
    let (chunk_sink, chunk_rx) = faktor_agent::ChunkSink::channel();
    let (session, agent, permissions, _mcp_servers) =
        build_daemon_with_mcp_and_chunks_fast(&data_dir, Some(config), Some(chunk_sink))
            .await
            .map_err(|e| format!("daemon build failed: {e}"))?;
    let store = session.store();
    // Crash recovery runs before the first request (spec §7) — and before
    // bind, so all recovery work is done before readiness is announced.
    if let Err(e) = agent.recover() {
        tracing::error!("recovery failed: {e}");
    }
    let mut deps = ServerDeps::new(session, agent, permissions);
    deps.chunk_rx = Some(chunk_rx);
    // The frontend generates the secret and passes it via env; the
    // daemon reads it here and never prints it.
    deps.server_password = ServerPassword::from_env();
    // The workspace root rides the global event envelope.
    deps.directory = std::env::current_dir()
        .ok()
        .map(|d| d.display().to_string());
    // Wire the native snapshot store so the wire revert/unrevert/diff
    // endpoints restore real files: the checkpoint store shares the
    // daemon's store + CAS (same rows, same blobs).
    let fs = faktor_fs::WorkspaceFileService::new();
    let snapshots = Arc::new(faktor_snapshot::CheckpointStore::new(
        deps.session.cas(),
        deps.session.store(),
    ));
    deps = deps.with_snapshots(fs, snapshots);
    // Bind BEFORE readiness and BEFORE any backup work (audit 44): the
    // historic code ran rotate_backup synchronously between recover() and
    // bind, so a slow or cold backup delayed first-request readiness.
    let handle = faktor_server::serve(deps, port)
        .await
        .map_err(|e| format!("failed to bind: {e}"))?;
    // The frozen stdout line; nothing else may be printed. Readiness is now
    // announced — no backup has run yet and, by construction, cannot have.
    println!("{}", handle.startup_line);
    tracing::info!("faktor-plus serving on {}", handle.addr);
    if let Some(tx) = ready_tx {
        let _ = tx.send(());
    }
    // Automatic online backup (spec §24), post-ready and low priority: a
    // delayed spawned task, gated by policy (audit 44: interval +
    // staleness), so it never delays startup and never piles hourly
    // snapshots onto rapid restarts. Best effort, runs exactly once.
    let backup_data_dir = data_dir.clone();
    let backup_task = tokio::task::spawn(async move {
        tokio::time::sleep(BACKUP_START_DELAY).await;
        if backup_due(&backup_data_dir) {
            rotate_backup(&store, &backup_data_dir);
        } else {
            tracing::info!(
                "startup backup skipped: a backup newer than {BACKUP_MIN_INTERVAL_SECS}s exists"
            );
        }
    });
    // Keep the daemon alive; when a shutdown is signaled, abort the backup
    // task first so a snapshot can never outlive its owning runtime.
    match shutdown_rx {
        Some(rx) => {
            let _ = rx.await;
            backup_task.abort();
        }
        None => std::future::pending::<()>().await,
    }
    Ok(())
}

async fn run(prompt: String, provider: &str, model: &str, workspace: PathBuf, data_dir: PathBuf) {
    match build_daemon(&data_dir, None) {
        Ok((session, agent, _permissions, _mcp)) => {
            let ws = match session.create_workspace(workspace.to_str().unwrap_or(".")) {
                Ok(ws) => ws,
                Err(e) => {
                    eprintln!("workspace error: {e}");
                    std::process::exit(1);
                }
            };
            let row = match session.create_session(ws, "cli run", provider, model) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("session error: {e}");
                    std::process::exit(1);
                }
            };
            match agent.run_turn(row.id(), &prompt, &[]).await {
                Ok(outcome) => {
                    println!("final state: {}", outcome.final_state.label());
                }
                Err(e) => {
                    eprintln!("turn error: {e}");
                    std::process::exit(1);
                }
            }
        }
        Err(e) => {
            eprintln!("daemon build failed: {e}");
            std::process::exit(1);
        }
    }
}

/// The ACP agent server (`faktor-cli acp`): build the REAL daemon graph over
/// the data dir (config from `faktor-plus.json` in the data dir when present,
/// else defaults; NO MCP layer — the acp surface needs the same providers,
/// tools, session store and agent the native daemon serves) and serve the
/// ACP wire protocol on stdin/stdout until EOF or `shutdown`.
async fn acp(data_dir: PathBuf) {
    let config = load_acp_config(&data_dir);
    if let Err(e) = std::fs::create_dir_all(&data_dir) {
        eprintln!("data dir error: {e}");
        std::process::exit(1);
    }
    let session = match SessionManager::open(data_dir.join("store"), data_dir.join("cas"), true) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("store error: {e}");
            std::process::exit(1);
        }
    };
    // The ACP surface is stdio-only: no SSE subscribers exist, so there is
    // no chunk sink (None = the runtime skips live-chunk overhead entirely).
    let (session, agent, _permissions, _mcp_servers) =
        match build_daemon_on_with_sink(session, config, vec![], None) {
            Ok(graph) => graph,
            Err(e) => {
                eprintln!("daemon build failed: {e}");
                std::process::exit(1);
            }
        };
    // Crash recovery runs before the first request (spec §7), like serve.
    if let Err(e) = agent.recover() {
        tracing::error!("recovery failed: {e}");
    }
    let backend = DaemonAcpBackend::new(session, agent);
    match AcpServer::new(backend).run_stdio().await {
        Ok(()) => {}
        Err(e) => {
            eprintln!("acp server error: {e}");
            std::process::exit(1);
        }
    }
}

/// Daemon config for `acp`: `faktor-plus.json` next to the data dir when it
/// exists; a broken file is a loud warning that falls back to defaults (the
/// daemon still serves — same policy as `serve`).
fn load_acp_config(data_dir: &std::path::Path) -> config::Config {
    let path = data_dir.join("faktor-plus.json");
    if !path.exists() {
        return config::Config::default();
    }
    match config::Config::load(&path) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("config error: {e}; using defaults");
            config::Config::default()
        }
    }
}

/// The ACP backend over the REAL daemon: sessions, turns, and abort go
/// through the durable `SessionManager` and `AgentRuntime` (crash recovery,
/// journal, cancellation, and providers included). ACP wire/lifecycle
/// handling lives in `faktor-acp`; this seam only maps requests onto the
/// runtime, mirroring how the native server endpoints drive the daemon.
struct DaemonAcpBackend {
    session: Arc<SessionManager>,
    agent: Arc<AgentRuntime>,
}

impl DaemonAcpBackend {
    fn new(session: Arc<SessionManager>, agent: Arc<AgentRuntime>) -> Self {
        Self { session, agent }
    }

    /// The session's provider: `params.provider` when given, else the ONLY
    /// registered provider instance (an ACP client without a provider
    /// preference binds the daemon's single provider deterministically).
    /// Zero or several providers refuse loudly instead of guessing.
    fn resolve_provider(&self, params: &Value) -> Result<String, String> {
        if let Some(p) = params.get("provider").and_then(Value::as_str) {
            return Ok(p.to_string());
        }
        let ids = self.agent.deps().providers.ids();
        match ids.len() {
            1 => Ok(ids.into_iter().next().expect("len 1")),
            0 => Err(
                "session/new: no providers are registered; configure one or pass params.provider"
                    .into(),
            ),
            _ => Err(format!(
                "session/new: multiple providers registered ({ids:?}); pass params.provider"
            )),
        }
    }
}

impl AcpBackend for DaemonAcpBackend {
    fn agent_info(&self) -> Value {
        let mut families: Vec<String> = self
            .agent
            .deps()
            .providers
            .all()
            .iter()
            .map(|p| p.id().to_string())
            .collect();
        families.sort();
        families.dedup();
        json!({
            "name": "Faktor",
            "version": faktor_core::VERSION,
            "providerFamilies": families,
        })
    }

    fn create_session(&self, params: &Value) -> Result<String, String> {
        let workspace = params
            .get("workspace")
            .and_then(Value::as_str)
            .unwrap_or("/");
        let title = params.get("title").and_then(Value::as_str).unwrap_or("acp");
        let model = params
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| self.agent.deps().model.clone());
        let provider = self.resolve_provider(params)?;
        let ws = self
            .session
            .create_workspace(workspace)
            .map_err(|e| e.message)?;
        let row = self
            .session
            .create_session(ws, title, &provider, &model)
            .map_err(|e| e.message)?;
        // Session lifecycle hook (audit): sessions are created here (the
        // session manager), not in the runtime — the daemon entry fires
        // SessionStart best-effort right after creation. Native server-side
        // session creation lives in crates/server, outside the CLI.
        self.agent
            .run_lifecycle_hook(faktor_hooks::HookEvent::SessionStart, row.id());
        Ok(row.id().to_string())
    }

    fn prompt(&self, session_id: &str, text: &str) -> Result<Value, String> {
        let sid = parse_session_id(session_id)?;
        if text.trim().is_empty() {
            return Err("prompt must not be empty".into());
        }
        if text.len() > faktor_session::MAX_PROMPT_BYTES {
            return Err(format!(
                "prompt of {} bytes exceeds the {} byte bound",
                text.len(),
                faktor_session::MAX_PROMPT_BYTES
            ));
        }
        // The AcpBackend seam is synchronous (one serialized ACP request at
        // a time); the real turn is async, so bridge sync → async on the
        // serve task via block_in_place (multi-threaded daemon runtime).
        let agent = self.agent.clone();
        let outcome = tokio::task::block_in_place(move || {
            tokio::runtime::Handle::current().block_on(agent.run_turn(sid, text, &[]))
        })
        .map_err(|e| e.message)?;
        if outcome.queued {
            // The prompt durably queued behind another actor's active turn
            // (a second ACP connection or the native API): hand the durable
            // queue to the runtime's per-session runner, like the server.
            let agent = self.agent.clone();
            tokio::task::spawn(async move { agent.run_session_queue(sid).await });
        }
        let status = if outcome.queued {
            "queued"
        } else {
            "completed"
        };
        Ok(json!({
            "status": status,
            "finalState": outcome.final_state,
        }))
    }

    fn abort(&self, session_id: &str) -> Result<(), String> {
        let sid = parse_session_id(session_id)?;
        self.agent.abort(sid).map(|_| ()).map_err(|e| e.message)
    }

    fn list_sessions(&self) -> Vec<String> {
        self.session
            .list_sessions(None)
            .map(|handles| handles.iter().map(|h| h.id().to_string()).collect())
            .unwrap_or_default()
    }
}

/// Session ids ride the ACP wire as plain decimal strings. Hostile ids
/// (non-numeric, zero, overflowing u64) are loud errors — never a panic.
fn parse_session_id(s: &str) -> Result<SessionId, String> {
    let raw: u64 = s.parse().map_err(|_| format!("invalid session id {s:?}"))?;
    if raw == 0 {
        return Err("invalid session id \"0\"".into());
    }
    Ok(SessionId::new(raw))
}

/// Human-readable outcome of one doctor run. The shell wrapper prints the
/// lines and exits non-zero when `issues > 0`; tests call [`doctor_run`]
/// directly so a failing run never exits the test process.
struct DoctorReport {
    lines: Vec<String>,
    issues: usize,
}

/// `faktor-plus doctor [--deep]`: plain mode opens with the bounded quick
/// path and reports quick checks; `--deep` additionally runs the full store
/// scan, the CAS blob verification, the global recovery-row scan, the
/// dangling-CAS-reference check (artifact rows + checkpoint after-blobs) and
/// the journal projection consistency checks. Issues are always SURFACED,
/// never repaired: the only automatic repair is stale-temp-file removal,
/// documented in [`remove_stale_temp_files`].
async fn doctor(data_dir: PathBuf, deep: bool) {
    let report = doctor_run(&data_dir, deep);
    for line in &report.lines {
        println!("{line}");
    }
    if report.issues == 0 {
        println!("doctor: all checks passed");
    } else {
        println!("doctor: {} issue(s)", report.issues);
        std::process::exit(1);
    }
}

fn doctor_run(data_dir: &std::path::Path, deep: bool) -> DoctorReport {
    let mut lines: Vec<String> = Vec::new();
    let mut issues = 0usize;
    match SessionManager::open_quick(data_dir.join("store"), data_dir.join("cas")) {
        Ok(session) => {
            lines.push("store: ok".into());
            let store = session.store();
            // Plain doctor uses the SAME bounded check the fast open runs
            // (audit 73): the full scan is a --deep concern.
            match store.diagnostics_quick() {
                Ok(d) => {
                    lines.push(serde_json::to_string_pretty(&d).unwrap());
                }
                Err(e) => {
                    lines.push(format!("store diagnostics failed: {e}"));
                    issues += 1;
                }
            }
            // Global unfinished-run count (plain doctor): the old report
            // only scanned session 1; every session counts now.
            match store.all_running_tool_rows() {
                Ok(runs) => {
                    lines.push(format!(
                        "unfinished tool runs across sessions: {}",
                        runs.len()
                    ));
                }
                Err(e) => {
                    lines.push(format!("running tool-run scan failed: {e}"));
                    issues += 1;
                }
            }
            // The ONE automatic repair doctor performs: stale temp/debris
            // removal only (documented). Everything deeper is surfaced, never
            // healed.
            let mut removed = Vec::new();
            remove_stale_temp_files(data_dir, &session, &mut removed);
            for path in removed {
                lines.push(format!("removed stale temp file: {path}"));
            }
            if deep {
                deep_doctor(&session, &mut lines, &mut issues);
            }
        }
        Err(e) => {
            lines.push(format!("store: FAILED ({e})"));
            issues += 1;
        }
    }
    DoctorReport { lines, issues }
}

/// `doctor --deep`: the full integrity scan, the CAS verification, the
/// cross-session recovery-row scan, the dangling-CAS-reference scan (which
/// also covers checkpoint after-blob refs) and journal projection checks.
/// None of these checks write to the store or the CAS: corruption is listed
/// and left alone (a second run must find the same issues).
fn deep_doctor(session: &Arc<SessionManager>, lines: &mut Vec<String>, issues: &mut usize) {
    let store = session.store();
    let add_issue = |line: String, lines: &mut Vec<String>, issues: &mut usize| {
        lines.push(line);
        *issues += 1;
    };
    // 1. Full store scan (the bounded quick check is NOT enough here).
    match store.deep_integrity_check() {
        Ok(found) if found.is_empty() => lines.push("deep store integrity scan: ok".into()),
        Ok(found) => {
            for issue in &found {
                lines.push(format!("deep store integrity scan: {issue}"));
            }
            *issues += found.len();
        }
        Err(e) => add_issue(
            format!("deep store integrity scan failed: {e}"),
            lines,
            issues,
        ),
    }
    // 2. CAS blob verification: every blob is decompressed and re-hashed.
    let cas = session.cas();
    let corrupted = cas.verify_integrity();
    if corrupted.is_empty() {
        lines.push("cas blob verification: ok".into());
    } else {
        let n = corrupted.len();
        for h in corrupted {
            lines.push(format!("cas blob corrupt: {}", h.to_hex()));
        }
        *issues += n;
    }
    // 3. Dangling CAS references (artifact rows + checkpoint after-blobs).
    match store.cas_hash_references() {
        Ok(refs) => {
            let mut dangling = Vec::new();
            for r in &refs {
                match faktor_core::hash::FileHash::from_hex(&r.hash) {
                    None => dangling.push(format!(
                        "{} row {} holds a malformed CAS hash {}",
                        r.source, r.row_id, r.hash
                    )),
                    Some(h) if !cas.has(h) => dangling.push(format!(
                        "{} row {} references missing CAS blob {}",
                        r.source, r.row_id, r.hash
                    )),
                    Some(_) => {}
                }
            }
            if dangling.is_empty() {
                lines.push(format!(
                    "cas references: {} hash(es) all present",
                    refs.len()
                ));
            } else {
                let n = dangling.len();
                for d in dangling {
                    lines.push(format!("dangling cas reference: {d}"));
                }
                *issues += n;
            }
        }
        Err(e) => add_issue(format!("cas reference scan failed: {e}"), lines, issues),
    }
    // 4. Global recovery-row scan (INFORMATIONAL: a live daemon legitimately
    // has running rows; the report makes a crashed daemon's backlog visible).
    match store.all_running_tool_rows() {
        Ok(runs) => {
            lines.push(format!(
                "running tool runs across all sessions: {}",
                runs.len()
            ));
            for r in runs.iter().take(10) {
                lines.push(format!(
                    "  session {} op {} tool {} status {} effect {} started_ms {}",
                    r.session_id, r.op_id, r.tool, r.status, r.effect_status, r.started_ms
                ));
            }
        }
        Err(e) => add_issue(format!("running tool-run scan failed: {e}"), lines, issues),
    }
    match store.all_active_turns() {
        Ok(turns) => lines.push(format!(
            "active logical turns across all sessions: {}",
            turns.len()
        )),
        Err(e) => add_issue(format!("active turn scan failed: {e}"), lines, issues),
    }
    // 5. Journal projection consistency (gapless 1..=N per session).
    match store.journal_consistency_issues() {
        Ok(problems) if problems.is_empty() => lines.push("journal consistency: ok".into()),
        Ok(problems) => {
            for p in &problems {
                lines.push(format!("journal inconsistency: {p}"));
            }
            *issues += problems.len();
        }
        Err(e) => add_issue(
            format!("journal consistency scan failed: {e}"),
            lines,
            issues,
        ),
    }
}

/// Doctor's ONLY automatic repair (documented; audit 73/74): remove STALE
/// TEMP/debris files — a leftover rollback-journal sidecar next to a WAL
/// store, interrupted-backup temp files, and crashed CAS writer temps.
/// Age-guarded so a live writer's files are never touched. Everything else
/// (hash mismatches, journal inconsistencies, dangling references) is
/// surfaced as an issue and NEVER auto-repaired.
fn remove_stale_temp_files(
    data_dir: &std::path::Path,
    session: &Arc<SessionManager>,
    removed: &mut Vec<String>,
) {
    // (a) Rollback-journal debris: WAL-mode stores only create a -journal
    // sidecar during recovery; our open already replayed any real one, so a
    // -journal surviving next to a live -wal is stale debris. Without the
    // -wal marker the journal mode is unknowable — never delete.
    let store_dir = data_dir.join("store");
    let db_stem = store_dir.join("faktor-plus.db");
    let wal_live = db_stem.with_extension("db-wal").exists();
    let journal = db_stem.with_extension("db-journal");
    let hour = std::time::Duration::from_secs(3600);
    if wal_live && older_than(&journal, hour) && std::fs::remove_file(&journal).is_ok() {
        removed.push(journal.display().to_string());
    }
    // (b) Interrupted-backup temp files (crashed writers; see rotate_backup).
    let backups = data_dir.join("backups");
    if let Ok(files) = std::fs::read_dir(&backups) {
        for f in files.flatten() {
            let p = f.path();
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.contains(".db.tmp-") && older_than(&p, hour) && std::fs::remove_file(&p).is_ok()
            {
                removed.push(p.display().to_string());
            }
        }
    }
    // (c) Crashed CAS writer temps (Cas tmp files are pid-uuid-tagged).
    let day = std::time::Duration::from_secs(24 * 3600);
    let cas_tmp = session.cas().root().join("tmp");
    if let Ok(files) = std::fs::read_dir(&cas_tmp) {
        for f in files.flatten() {
            let p = f.path();
            if older_than(&p, day) && std::fs::remove_file(&p).is_ok() {
                removed.push(p.display().to_string());
            }
        }
    }
}

async fn sessions(data_dir: PathBuf) {
    match SessionManager::open(data_dir.join("store"), data_dir.join("cas"), false) {
        Ok(session) => match session.list_sessions(None) {
            Ok(rows) => {
                for r in rows {
                    let state = r.state().map(|s| s.label()).unwrap_or("unknown");
                    let title = r.title().unwrap_or_default();
                    let provider = r.provider().unwrap_or_default();
                    let model = r.model().unwrap_or_default();
                    println!("{}  {title}  {provider}  {model}  [{state}]", r.id());
                }
            }
            Err(e) => eprintln!("error: {e}"),
        },
        Err(e) => eprintln!("error: {e}"),
    }
}

// Keep SessionId referenced for future commands.
#[allow(dead_code)]
fn _sid(_: SessionId) {}

#[cfg(test)]
mod tests {
    use super::*;
    use faktor_core::model::ModelCapabilities;
    use faktor_core::state::AgentState;
    use faktor_provider::{FakeProvider, ScriptedResponse};
    use std::pin::Pin;

    /// Permission requester that never blocks on a UI (text-only turns never
    /// ask, but AgentDeps requires one deterministically).
    struct AlwaysAllow;
    impl faktor_agent::PermissionRequester for AlwaysAllow {
        fn request(
            &self,
            _session: SessionId,
            _permission: &faktor_session::PermissionRequest,
        ) -> Pin<
            Box<
                dyn std::future::Future<
                        Output = faktor_core::Result<faktor_core::capability::PermissionDecision>,
                    > + Send,
            >,
        > {
            Box::pin(async { Ok(faktor_core::capability::PermissionDecision::Allow) })
        }
    }

    /// Minimal REAL daemon AgentDeps over an open session manager: text-only
    /// turns, no MCP/verifier/supervisor (nothing here ever runs a process).
    fn test_agent(session: Arc<SessionManager>, registry: ProviderRegistry) -> Arc<AgentRuntime> {
        let cas = session.cas();
        let deps = AgentDeps {
            session: session.clone(),
            providers: Arc::new(registry),
            chunk_sink: None,
            permission_requester: Arc::new(AlwaysAllow),
            evidence: Arc::new(faktor_agent::NoEvidence),
            tools: Arc::new(ToolRegistry::new()),
            cas: Some(cas),
            workspaces: faktor_fs::WorkspaceFileService::new(),
            edit: None,
            snapshots: None,
            sandbox: None,
            supervisor: None,
            verifier: None,
            hooks: None,
            instructions_loader: None,
            router: None,
            budget_micro: None,
            model: "default".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "You are Faktor.".into(),
            clock: Arc::new(SystemClock),
            tool_call_mode: ToolCallMode::Native,
            tool_deadline_ms: 2000,
            retry_policy: faktor_core::retry::RetryPolicy::default(),
        };
        AgentRuntime::new(deps).unwrap()
    }

    /// A minimal REAL daemon over a temp data dir: one scripted provider
    /// registered under the instance id "fake" (the single registered
    /// instance, so the ACP session defaults resolve to it deterministically).
    fn acp_test_daemon(
        script: Vec<ScriptedResponse>,
    ) -> (tempfile::TempDir, Arc<SessionManager>, Arc<AgentRuntime>) {
        let dir = tempfile::tempdir().unwrap();
        let session =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let mut registry = ProviderRegistry::new();
        registry.register(Arc::new(FakeProvider::with_script(
            "fake",
            ModelCapabilities {
                tools: true,
                ..Default::default()
            },
            script,
        )));
        let agent = test_agent(session.clone(), registry);
        (dir, session, agent)
    }

    fn handle(session: &Arc<SessionManager>, sid: &str) -> faktor_session::SessionHandle {
        session
            .get_session(SessionId::new(sid.parse().unwrap()))
            .unwrap()
            .unwrap()
    }

    #[test]
    fn expand_home_and_relative() {
        assert_eq!(expand("."), PathBuf::from("."));
        let home = expand("~");
        assert_eq!(expand("~/x"), home.join("x"));
    }

    #[test]
    fn serve_config_is_strict_only_for_an_explicit_path() {
        // Audit 31: without --config, defaults (nothing can fail startup);
        // with an explicit --config, parse+validation failures are startup
        // errors — never a silent fallback to defaults.
        assert_eq!(
            serve_config(None).unwrap().model,
            config::Config::default().model,
            "no --config stays lenient"
        );
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("serve.json");
        std::fs::write(&path, "{not json").unwrap();
        let e = serve_config(Some(path.clone()))
            .expect_err("an explicit broken config must fail startup");
        assert!(e.contains("serve.json"), "{e}");
        // Unknown fields and duplicate provider ids fail strict too.
        std::fs::write(&path, r#"{"model": "m", "surprise": 1}"#).unwrap();
        assert!(serve_config(Some(path.clone())).is_err());
        std::fs::write(
            &path,
            r#"{"providers": [
                {"kind": "ollama", "id": "twice", "base_url": null},
                {"kind": "open_ai", "id": "twice", "base_url": "http://x"}
            ]}"#,
        )
        .unwrap();
        let e = serve_config(Some(path.clone())).expect_err("duplicate ids fail strict");
        assert!(e.contains("twice"), "{e}");
        // A healthy explicit config still loads.
        std::fs::write(
            &path,
            r#"{"config_version": 1, "model": "m", "providers": [
                {"kind": "ollama", "id": "o", "base_url": null}
            ]}"#,
        )
        .unwrap();
        assert_eq!(serve_config(Some(path)).unwrap().model, "m");
    }

    #[test]
    fn instructions_loader_wires_only_a_config_default_root() {
        // Audit 31: the daemon cannot invent a workspace root — loading
        // repository rules at the wrong root would silently misapply them.
        // The config file shape carries no workspace/root field today, so
        // no single default root is determinable at build and the loader is
        // None by construction (per-session roots are runtime wiring).
        let cfg = config::Config::default();
        assert!(config_default_root(&cfg).is_none());
        assert!(daemon_instructions_loader(&cfg).is_none());
    }

    #[test]
    fn parse_hooks_env_skips_malformed_entries_and_bounds_the_count() {
        // Hostile/malformed env: garbage entries are skipped, never a panic
        // and never a registered hook. Bounded to MAX_ENV_HOOKS entries.
        let raw = "  ; no_colon_here ; pre_tool: ; nope:true; bogus_event:/bin/true";
        let specs = parse_hooks_env(raw);
        assert!(specs.is_empty(), "every entry is malformed: {specs:?}");

        let ok = "task_complete:/bin/true";
        let specs = parse_hooks_env(ok);
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].id, "env-0");
        assert_eq!(specs[0].events, vec![faktor_hooks::HookEvent::TaskComplete]);
        assert_eq!(specs[0].command, "/bin/true");
        assert!(specs[0].args.is_empty());
        assert!(specs[0].env_allowlist, "env allowlist is mandatory");
        assert_eq!(
            specs[0].failure_policy,
            faktor_hooks::FailurePolicy::FailClosed,
            "FailClosed is the default failure policy"
        );

        // Malformed entries between valid ones are skipped; ids stay
        // contiguous over the parsed (not the raw) positions.
        let mixed = "pre_tool:/bin/a arg1;garbage;post_tool:/bin/b";
        let specs = parse_hooks_env(mixed);
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].id, "env-0");
        assert_eq!(specs[0].events, vec![faktor_hooks::HookEvent::PreTool]);
        assert_eq!(specs[0].args, vec!["arg1".to_string()]);
        assert_eq!(specs[1].id, "env-1");
        assert_eq!(specs[1].events, vec![faktor_hooks::HookEvent::PostTool]);

        // A hostile env with more than MAX_ENV_HOOKS entries is capped.
        let many = (0..100)
            .map(|i| format!("pre_tool:/bin/true {i}"))
            .collect::<Vec<_>>()
            .join(";");
        let specs = parse_hooks_env(&many);
        assert_eq!(specs.len(), MAX_ENV_HOOKS);
    }

    #[test]
    fn parse_hooks_env_recognizes_every_hook_event_name() {
        // Every snake_case event name the runtime can fire must parse, so an
        // env entry never silently drops a supported event.
        let names = [
            ("session_start", faktor_hooks::HookEvent::SessionStart),
            ("session_resume", faktor_hooks::HookEvent::SessionResume),
            ("task_start", faktor_hooks::HookEvent::TaskStart),
            ("pre_model", faktor_hooks::HookEvent::PreModel),
            ("post_model", faktor_hooks::HookEvent::PostModel),
            ("pre_tool", faktor_hooks::HookEvent::PreTool),
            ("post_tool", faktor_hooks::HookEvent::PostTool),
            ("tool_error", faktor_hooks::HookEvent::ToolError),
            ("pre_edit", faktor_hooks::HookEvent::PreEdit),
            ("post_edit", faktor_hooks::HookEvent::PostEdit),
            ("pre_commit", faktor_hooks::HookEvent::PreCommit),
            ("subagent_start", faktor_hooks::HookEvent::SubagentStart),
            ("subagent_stop", faktor_hooks::HookEvent::SubagentStop),
            ("agent_error", faktor_hooks::HookEvent::AgentError),
            ("agent_stop", faktor_hooks::HookEvent::AgentStop),
            ("task_complete", faktor_hooks::HookEvent::TaskComplete),
            ("session_end", faktor_hooks::HookEvent::SessionEnd),
        ];
        for (name, event) in names {
            let specs = parse_hooks_env(&format!("{name}:/bin/true"));
            assert_eq!(specs.len(), 1, "event {name} must parse");
            assert_eq!(specs[0].events, vec![event], "event {name}");
        }
    }

    #[test]
    fn parsed_env_hook_registers_and_fires_on_a_real_registry() {
        // End-to-end shape of the daemon wiring: a parsed spec registers on
        // a real HookRegistry and the hook FIRES for its event (the audit
        // log gains the record). /bin/echo exists on the CI platforms.
        let specs = parse_hooks_env("post_tool:/bin/echo hook-fired");
        assert_eq!(specs.len(), 1);
        let registry = Arc::new(faktor_hooks::HookRegistry::new());
        for spec in specs {
            registry.register(spec).unwrap();
        }
        let verdict = registry.run(
            faktor_hooks::HookEvent::PostTool,
            &faktor_hooks::HookInput {
                event: faktor_hooks::HookEvent::PostTool,
                ..Default::default()
            },
        );
        assert_eq!(verdict, faktor_hooks::HookVerdict::Allow);
        let audit = registry.audit();
        assert_eq!(audit.len(), 1);
        assert_eq!(audit[0].hook_id, "env-0");
        assert_eq!(audit[0].event, faktor_hooks::HookEvent::PostTool);
    }

    #[test]
    fn agent_info_names_faktor_and_lists_registered_provider_families() {
        let (_dir, session, agent) = acp_test_daemon(vec![]);
        let backend = DaemonAcpBackend::new(session, agent);
        let info = backend.agent_info();
        assert_eq!(info["name"], "Faktor");
        assert_eq!(info["version"], faktor_core::VERSION);
        assert_eq!(
            info["providerFamilies"],
            json!(["fake"]),
            "the scripted provider's family id must surface"
        );
    }

    #[test]
    fn create_session_applies_defaults_and_list_sessions_sees_it() {
        let (_dir, session, agent) = acp_test_daemon(vec![]);
        let backend = DaemonAcpBackend::new(session.clone(), agent);

        // Defaults: workspace "/", title "acp", daemon model, single
        // registered provider.
        let sid = backend.create_session(&json!({})).unwrap();
        assert!(!sid.is_empty());
        let row = handle(&session, &sid).row().unwrap();
        assert_eq!(row.title, "acp");
        assert_eq!(row.provider, "fake");
        assert_eq!(row.model, "default");
        assert!(backend.list_sessions().contains(&sid));

        // Explicit params override every default and stay listable.
        let sid2 = backend
            .create_session(&json!({
                "workspace": "/elsewhere",
                "title": "zed-import",
                "provider": "fake",
                "model": "m",
            }))
            .unwrap();
        let row2 = handle(&session, &sid2).row().unwrap();
        assert_eq!(row2.title, "zed-import");
        assert_eq!(row2.model, "m");
        assert_eq!(
            session
                .store()
                .workspace_root(row2.workspace_id)
                .unwrap()
                .as_deref(),
            Some("/elsewhere")
        );
        assert!(backend.list_sessions().contains(&sid2));
        assert_eq!(backend.list_sessions().len(), 2);

        // No registered provider: refusing loudly beats a phantom session.
        let (_d2, session3, agent3) = {
            let dir2 = tempfile::tempdir().unwrap();
            let s = SessionManager::open(dir2.path().join("store"), dir2.path().join("cas"), true)
                .unwrap();
            let a = test_agent(s.clone(), ProviderRegistry::new());
            (dir2, s, a)
        };
        let backend3 = DaemonAcpBackend::new(session3, agent3);
        let err = backend3.create_session(&json!({})).unwrap_err();
        assert!(err.contains("no providers"), "{err}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn prompt_runs_a_real_turn_and_reports_the_final_state() {
        let (_dir, session, agent) = acp_test_daemon(vec![
            ScriptedResponse::Text("pong".into()),
            ScriptedResponse::End,
        ]);
        let backend = DaemonAcpBackend::new(session, agent);
        let sid = backend.create_session(&json!({})).unwrap();

        let result = backend.prompt(&sid, "ping").unwrap();
        assert_eq!(result["status"], "completed");
        assert_eq!(result["finalState"], "ready_for_next_turn");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn prompt_on_unknown_session_errors() {
        let (_dir, session, agent) = acp_test_daemon(vec![
            ScriptedResponse::Text("pong".into()),
            ScriptedResponse::End,
        ]);
        let backend = DaemonAcpBackend::new(session, agent);
        let unknown = format!("{}", u64::MAX - 1);
        let err = backend.prompt(&unknown, "hi").unwrap_err();
        assert!(err.contains(&unknown), "{err}");
        assert!(backend.abort(&unknown).is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn abort_cancels_a_live_turn_and_keeps_the_session_usable() {
        let (_dir, session, agent) = acp_test_daemon(vec![
            ScriptedResponse::Text("pong".into()),
            ScriptedResponse::End,
        ]);
        let backend = DaemonAcpBackend::new(session.clone(), agent.clone());

        // A: the turn is durably ACTIVE (Preparing, live op registered, never
        // driven) — abort must land the machine ReadyForNextTurn.
        let sid_a = backend
            .create_session(&json!({ "title": "mid-flight" }))
            .unwrap();
        agent
            .submit(SessionId::new(sid_a.parse().unwrap()), "stop me", &[])
            .unwrap();
        assert!(handle(&session, &sid_a).state().unwrap().is_active());
        assert!(backend.abort(&sid_a).is_ok());
        assert_eq!(
            handle(&session, &sid_a).state().unwrap(),
            AgentState::ReadyForNextTurn
        );

        // B: an idle abort (Stop cancels the turn, never the session) keeps
        // the session promptable — a real turn still completes afterwards.
        let sid_b = backend.create_session(&json!({})).unwrap();
        assert!(backend.abort(&sid_b).is_ok());
        let result = backend.prompt(&sid_b, "ping").unwrap();
        assert_eq!(result["status"], "completed");
        assert_eq!(result["finalState"], "ready_for_next_turn");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn hostile_inputs_are_errs_never_panics() {
        let (_dir, session, agent) = acp_test_daemon(vec![
            ScriptedResponse::Text("pong".into()),
            ScriptedResponse::End,
        ]);
        let backend = DaemonAcpBackend::new(session.clone(), agent);
        let sid = backend.create_session(&json!({})).unwrap();

        // Empty and whitespace prompts are refused before the runtime.
        assert!(backend.prompt(&sid, "").is_err());
        assert!(backend.prompt(&sid, "   \n\t ").is_err());

        // Oversized prompts are refused by the daemon bound (never a panic,
        // never an unbounded journal write).
        let huge = "x".repeat(5 * 1024 * 1024);
        let err = backend.prompt(&sid, &huge).unwrap_err();
        assert!(
            err.contains("exceeds") && err.contains("bound"),
            "oversized prompt must be refused, got: {err}"
        );

        // Hostile session ids: non-numeric, zero, overflowing u64.
        for bad in ["abc", "0", "18446744073709551616", "-1", ""] {
            assert!(backend.prompt(bad, "hi").is_err(), "prompt {bad:?} refused");
            assert!(backend.abort(bad).is_err(), "abort {bad:?} refused");
        }

        // A deleted (Closed) session refuses prompts and aborts loudly.
        let dead = backend.create_session(&json!({})).unwrap();
        session
            .delete_session(SessionId::new(dead.parse().unwrap()))
            .unwrap();
        assert!(backend.prompt(&dead, "hi").is_err());
        assert!(backend.abort(&dead).is_err());

        // None of the hostile inputs touched the live session or crashed the
        // daemon: a real turn still completes.
        let result = backend.prompt(&sid, "still alive").unwrap();
        assert_eq!(result["finalState"], "ready_for_next_turn");
    }

    /// Write one fake complete backup `faktor-plus-{ts_ms}.db` of `size`
    /// bytes with an explicit mtime (deterministic ordering for gate and
    /// retention tests).
    fn write_backup(
        dir: &std::path::Path,
        ts_ms: u64,
        mtime: std::time::SystemTime,
        size: u64,
    ) -> std::path::PathBuf {
        let p = dir.join("backups").join(format!("faktor-plus-{ts_ms}.db"));
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, vec![0u8; size as usize]).unwrap();
        let f = std::fs::OpenOptions::new().write(true).open(&p).unwrap();
        f.set_modified(mtime).unwrap();
        p
    }

    #[test]
    fn backup_gate_skips_a_fresh_matching_snapshot_and_honors_staleness_and_size() {
        let hour = std::time::Duration::from_secs(3600);
        // Fresh store for each scenario so mtime ordering stays unambiguous.
        // (a) No backups at all: due.
        {
            let dir = tempfile::tempdir().unwrap();
            let session =
                SessionManager::open_quick(dir.path().join("store"), dir.path().join("cas"))
                    .unwrap();
            drop(session);
            assert!(backup_due(dir.path()), "no backups: startup backup is due");
        }
        // (b) A fresh backup whose size matches the store: NOT due (audit 44
        // interval gate — rapid restarts must not pile hourly snapshots).
        {
            let dir = tempfile::tempdir().unwrap();
            let session =
                SessionManager::open_quick(dir.path().join("store"), dir.path().join("cas"))
                    .unwrap();
            let db_len = std::fs::metadata(dir.path().join("store").join("faktor-plus.db"))
                .unwrap()
                .len();
            drop(session);
            write_backup(dir.path(), 1, std::time::SystemTime::now(), db_len);
            assert!(
                !backup_due(dir.path()),
                "fresh same-size backup must gate the snapshot"
            );
        }
        // (c) Same size but old: due.
        {
            let dir = tempfile::tempdir().unwrap();
            let session =
                SessionManager::open_quick(dir.path().join("store"), dir.path().join("cas"))
                    .unwrap();
            let db_len = std::fs::metadata(dir.path().join("store").join("faktor-plus.db"))
                .unwrap()
                .len();
            drop(session);
            write_backup(
                dir.path(),
                1,
                std::time::SystemTime::now() - 2 * hour,
                db_len,
            );
            assert!(backup_due(dir.path()), "old snapshot is stale: due");
        }
        // (d) Fresh but the store RESIZED since the snapshot (crash-recovery
        // wrote): due even inside the interval.
        {
            let dir = tempfile::tempdir().unwrap();
            let session =
                SessionManager::open_quick(dir.path().join("store"), dir.path().join("cas"))
                    .unwrap();
            let db_len = std::fs::metadata(dir.path().join("store").join("faktor-plus.db"))
                .unwrap()
                .len();
            drop(session);
            write_backup(
                dir.path(),
                1,
                std::time::SystemTime::now() - std::time::Duration::from_secs(600),
                db_len - 1,
            );
            assert!(
                backup_due(dir.path()),
                "a resized store makes a fresh snapshot stale: due"
            );
        }
    }

    #[test]
    fn rotate_backup_enforces_the_count_quota_and_keeps_the_newest() {
        // 10 old backups + the new snapshot = 11 candidates; retention must
        // drop the OLDEST until BACKUP_MAX_FILES (8) remain — never the
        // snapshot just written.
        let dir = tempfile::tempdir().unwrap();
        let session =
            SessionManager::open_quick(dir.path().join("store"), dir.path().join("cas")).unwrap();
        let store = session.store();
        let db_len = std::fs::metadata(dir.path().join("store").join("faktor-plus.db"))
            .unwrap()
            .len();
        let now = std::time::SystemTime::now();
        for i in 0..10u64 {
            write_backup(
                dir.path(),
                1000 + i,
                now - std::time::Duration::from_secs((i + 1) * 3600),
                100,
            );
        }
        rotate_backup(&store, dir.path());
        let files = list_backups(dir.path());
        assert_eq!(
            files.len(),
            BACKUP_MAX_FILES,
            "11 candidates must be rotated down to {BACKUP_MAX_FILES}"
        );
        // The newest survivor is the snapshot just written (matches the
        // store size; the seeded fakes are 100 bytes).
        let newest_len = std::fs::metadata(&files[0]).unwrap().len();
        assert_eq!(newest_len, db_len, "the fresh snapshot must survive");
        // No interrupted-writer temp files are ever listed or kept.
        assert!(files.iter().all(|p| !p.to_string_lossy().contains(".tmp-")));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn startup_line_precedes_backup_and_the_interval_gate_skips_a_fresh_snapshot() {
        // Audit 44: serve must print the startup line BEFORE any backup file
        // exists, and when the interval gate says skip, no backup may appear
        // even after the delayed backup task would have fired.
        let dir = tempfile::tempdir().unwrap();
        let session =
            SessionManager::open_quick(dir.path().join("store"), dir.path().join("cas")).unwrap();
        let db_len = std::fs::metadata(dir.path().join("store").join("faktor-plus.db"))
            .unwrap()
            .len();
        drop(session);
        // Seed a fresh backup matching the store: the gate says skip.
        write_backup(dir.path(), 1, std::time::SystemTime::now(), db_len);
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let dir2 = dir.path().to_path_buf();
        let daemon = tokio::task::spawn(async move {
            serve_impl(0, dir2, None, Some(ready_tx), Some(shutdown_rx)).await
        });
        // The startup line is printed (readiness): no backup has run yet.
        tokio::time::timeout(std::time::Duration::from_secs(30), ready_rx)
            .await
            .expect("serve must reach the startup line")
            .expect("ready signal");
        // Wait past the post-ready delay + margin: the gate must still skip.
        tokio::time::sleep(BACKUP_START_DELAY + std::time::Duration::from_millis(500)).await;
        assert_eq!(
            list_backups(dir.path()).len(),
            1,
            "the interval gate must skip the backup on a fresh snapshot"
        );
        let _ = shutdown_tx.send(());
        tokio::time::timeout(std::time::Duration::from_secs(10), daemon)
            .await
            .expect("daemon must stop on shutdown")
            .expect("serve_impl returns Ok")
            .unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn startup_backup_runs_only_after_readiness_when_due() {
        // With no existing backup the startup backup IS due, but it must run
        // strictly AFTER the startup line: at readiness no backup file
        // exists yet; it appears only after the post-ready delay.
        let dir = tempfile::tempdir().unwrap();
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let dir2 = dir.path().to_path_buf();
        let daemon = tokio::task::spawn(async move {
            serve_impl(0, dir2, None, Some(ready_tx), Some(shutdown_rx)).await
        });
        tokio::time::timeout(std::time::Duration::from_secs(30), ready_rx)
            .await
            .expect("serve must reach the startup line")
            .expect("ready signal");
        assert!(
            list_backups(dir.path()).is_empty(),
            "startup line must precede any backup file"
        );
        // The delayed task then writes exactly one snapshot.
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            if list_backups(dir.path()).len() == 1 {
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "the gated backup task must run after readiness"
            );
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        let _ = shutdown_tx.send(());
        tokio::time::timeout(std::time::Duration::from_secs(10), daemon)
            .await
            .expect("daemon must stop on shutdown")
            .expect("serve_impl returns Ok")
            .unwrap();
    }

    #[test]
    fn doctor_plain_passes_on_a_healthy_store_and_fails_on_corruption() {
        let dir = tempfile::tempdir().unwrap();
        {
            let session =
                SessionManager::open_quick(dir.path().join("store"), dir.path().join("cas"))
                    .unwrap();
            session
                .create_session(session.create_workspace("/w").unwrap(), "t", "p", "m")
                .unwrap();
        }
        let report = doctor_run(dir.path(), false);
        assert_eq!(report.issues, 0, "healthy store: {:?}", report.lines);
        assert!(report.lines.iter().any(|l| l == "store: ok"));
        assert!(report
            .lines
            .iter()
            .any(|l| l.contains("\"journal_mode\": \"wal\"")));
        // Corrupt store: plain doctor fails loudly (never exits the process
        // from doctor_run; the wrapper owns the exit code).
        let garbage = tempfile::tempdir().unwrap();
        std::fs::write(
            garbage.path().join("store"),
            b"not a directory; the open must fail cleanly",
        )
        .unwrap();
        let report = doctor_run(garbage.path(), false);
        assert!(report.issues >= 1, "{:?}", report.lines);
        assert!(report.lines.iter().any(|l| l.starts_with("store: FAILED")));
    }

    #[test]
    fn doctor_deep_flags_a_corrupt_cas_blob_and_never_silently_heals() {
        // Audit 73/74: doctor --deep lists a corrupted CAS blob as an issue
        // and a SECOND run finds the SAME issue — corruption is surfaced,
        // never repaired.
        let dir = tempfile::tempdir().unwrap();
        let hash_hex = {
            let session =
                SessionManager::open_quick(dir.path().join("store"), dir.path().join("cas"))
                    .unwrap();
            let ws = session.create_workspace("/w").unwrap();
            let sid = session.create_session(ws, "t", "p", "m").unwrap().id();
            let cas = session.cas();
            let hash = cas.put(b"doctor blob").unwrap();
            session
                .store()
                .put_artifact(sid, "command_output", &hash.to_hex(), "sum", 11)
                .unwrap();
            // Corrupt the blob behind the CAS's back: present file, wrong
            // content (not zstd, so verify_integrity flags it).
            let blob = cas.root().join(hash.cas_path());
            std::fs::write(&blob, b"this is not zstd-compressed content").unwrap();
            hash.to_hex()
        };
        let first = doctor_run(dir.path(), true);
        assert!(first.issues > 0, "{:?}", first.lines);
        assert!(
            first
                .lines
                .iter()
                .any(|l| l.contains("cas blob corrupt") && l.contains(&hash_hex)),
            "{:?}",
            first.lines
        );
        // Second run: still failing — no silent healing.
        let second = doctor_run(dir.path(), true);
        assert!(second.issues > 0, "{:?}", second.lines);
        assert!(
            second.lines.iter().any(|l| l.contains(&hash_hex)),
            "{:?}",
            second.lines
        );
    }

    #[test]
    fn doctor_deep_flags_dangling_and_malformed_cas_references() {
        let dir = tempfile::tempdir().unwrap();
        {
            let session =
                SessionManager::open_quick(dir.path().join("store"), dir.path().join("cas"))
                    .unwrap();
            let ws = session.create_workspace("/w").unwrap();
            let sid = session.create_session(ws, "t", "p", "m").unwrap().id();
            let missing = "ab".repeat(32);
            session
                .store()
                .put_artifact(sid, "command_output", &missing, "sum", 10)
                .unwrap();
            session
                .store()
                .put_artifact(sid, "command_output", "not-a-hex-hash", "sum", 10)
                .unwrap();
            let real = session.cas().put(b"present").unwrap();
            session
                .store()
                .put_artifact(sid, "command_output", &real.to_hex(), "sum", 7)
                .unwrap();
        }
        let report = doctor_run(dir.path(), true);
        assert!(report.issues >= 2, "{:?}", report.lines);
        assert!(
            report.lines.iter().any(|l| {
                l.contains("dangling cas reference")
                    && l.contains(&"ab".repeat(32))
                    && l.contains("missing CAS blob")
            }),
            "{:?}",
            report.lines
        );
        assert!(
            report
                .lines
                .iter()
                .any(|l| l.contains("malformed CAS hash")),
            "{:?}",
            report.lines
        );
        // Rerun: the same refs are still dangling (no repair happened).
        let again = doctor_run(dir.path(), true);
        assert!(again.issues >= 2, "{:?}", again.lines);
    }

    #[test]
    fn doctor_deep_reports_global_running_rows_without_failing() {
        // Deep doctor surfaces cross-session recovery rows as INFORMATION
        // (a live daemon legitimately has running rows) — zero issues on an
        // otherwise healthy store.
        let dir = tempfile::tempdir().unwrap();
        {
            let session =
                SessionManager::open_quick(dir.path().join("store"), dir.path().join("cas"))
                    .unwrap();
            let ws = session.create_workspace("/w").unwrap();
            let sid = session.create_session(ws, "t", "p", "m").unwrap().id();
            session
                .store()
                .start_tool_run(
                    sid,
                    faktor_core::id::OpId::new(7),
                    "echo",
                    serde_json::json!({}),
                    serde_json::json!({"strategy": "none"}),
                    None,
                    None,
                )
                .unwrap();
            session
                .store()
                .start_turn_record(
                    sid,
                    faktor_core::id::OpId::new(9),
                    None,
                    Some(2),
                    "p",
                    "m",
                    None,
                )
                .unwrap();
        }
        let report = doctor_run(dir.path(), true);
        assert_eq!(report.issues, 0, "{:?}", report.lines);
        assert!(
            report
                .lines
                .iter()
                .any(|l| l.contains("running tool runs across all sessions: 1")),
            "{:?}",
            report.lines
        );
        assert!(
            report
                .lines
                .iter()
                .any(|l| l.contains("active logical turns across all sessions: 1")),
            "{:?}",
            report.lines
        );
    }
}
