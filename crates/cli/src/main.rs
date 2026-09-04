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
        Command::Doctor { data_dir } => {
            doctor(expand(&data_dir)).await;
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
    chunk_tx: Option<tokio::sync::mpsc::UnboundedSender<faktor_agent::ChunkEvent>>,
) -> Result<DaemonGraph, String> {
    let config = config.unwrap_or_default();
    let entries = config.mcp_servers()?;
    std::fs::create_dir_all(data_dir).map_err(|e| e.to_string())?;
    let session = SessionManager::open(data_dir.join("store"), data_dir.join("cas"), true)
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
    chunk_tx: Option<tokio::sync::mpsc::UnboundedSender<faktor_agent::ChunkEvent>>,
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
        instructions_loader: None,
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

/// Online backup with rotation (spec §24 "automatic backups"): one
/// crash-safe snapshot per daemon start, newest MAX_BACKUPS kept. Best
/// effort — a backup failure never stops the daemon.
fn rotate_backup(store: &faktor_store::Store, data_dir: &std::path::Path) {
    const MAX_BACKUPS: usize = 8;
    let backups = data_dir.join("backups");
    if std::fs::create_dir_all(&backups).is_err() {
        return;
    }
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let dest = backups.join(format!("faktor-plus-{ts}.db"));
    if let Err(e) = store.backup_to(&dest) {
        tracing::warn!("automatic backup failed: {e}");
        return;
    }
    tracing::info!("automatic backup written to {}", dest.display());
    // Rotation: keep the newest MAX_BACKUPS files.
    let Ok(files) = std::fs::read_dir(&backups) else {
        return;
    };
    let mut names: Vec<String> = files
        .flatten()
        .filter_map(|f| {
            let n = f.file_name().to_string_lossy().into_owned();
            (n.starts_with("faktor-plus-") && n.ends_with(".db")).then_some(n)
        })
        .collect();
    names.sort();
    names.reverse();
    for old in names.iter().skip(MAX_BACKUPS) {
        let _ = std::fs::remove_file(backups.join(old));
    }
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

async fn serve(port: u16, data_dir: PathBuf, config_path: Option<PathBuf>) {
    let config = config_path
        .map(|p| {
            config::Config::load(&p).unwrap_or_else(|e| {
                tracing::error!("config error: {e}; using defaults");
                config::Config::default()
            })
        })
        .unwrap_or_default();
    let (chunk_tx, chunk_rx) = tokio::sync::mpsc::unbounded_channel();
    match build_daemon_with_mcp_and_chunks(&data_dir, Some(config), Some(chunk_tx)).await {
        Ok((session, agent, permissions, _mcp_servers)) => {
            // Crash recovery runs before the first request (spec §7).
            if let Err(e) = agent.recover() {
                tracing::error!("recovery failed: {e}");
            }
            // Automatic online backup at daemon start (spec §24): the SQLite
            // backup API is crash-safe while the daemon runs; the newest
            // MAX_BACKUPS rotate, oldest deleted.
            rotate_backup(&session.store(), &data_dir);
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
            match faktor_server::serve(deps, port).await {
                Ok(handle) => {
                    // The frozen stdout line; nothing else may be printed.
                    println!("{}", handle.startup_line);
                    tracing::info!("faktor-plus serving on {}", handle.addr);
                    std::future::pending::<()>().await;
                }
                Err(e) => {
                    tracing::error!("failed to bind: {e}");
                    std::process::exit(1);
                }
            }
        }
        Err(e) => {
            tracing::error!("daemon build failed: {e}");
            std::process::exit(1);
        }
    }
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
    let (chunk_tx, _chunk_rx) = tokio::sync::mpsc::unbounded_channel();
    let (session, agent, _permissions, _mcp_servers) =
        match build_daemon_on_with_sink(session, config, vec![], Some(chunk_tx)) {
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

async fn doctor(data_dir: PathBuf) {
    let mut issues = 0usize;
    match SessionManager::open(data_dir.join("store"), data_dir.join("cas"), true) {
        Ok(session) => {
            println!("store: ok");
            match session.integrity_report() {
                Ok(d) => println!("{}", serde_json::to_string_pretty(&d).unwrap()),
                Err(e) => {
                    println!("store diagnostics failed: {e}");
                    issues += 1;
                }
            }
        }
        Err(e) => {
            println!("store: FAILED ({e})");
            issues += 1;
        }
    }
    if issues == 0 {
        println!("doctor: all checks passed");
    } else {
        println!("doctor: {issues} issue(s)");
        std::process::exit(1);
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
}
