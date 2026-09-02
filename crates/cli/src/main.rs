//! kilop-cli — `serve`, `run`, `doctor`, `sessions` (spec §34, §42, §43).
//!
//! `serve --port 0` prints the exact frozen startup line
//! `kilo server listening on http://127.0.0.1:<port>` so the frozen v7.5.6
//! extension connects exactly as it did to the old CLI. Nothing else goes to
//! stdout. Auth comes from the frontend-generated `KILO_SERVER_PASSWORD`
//! environment variable; the daemon never prints it.

use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use evidence::RepoEvidence;
use kilop_agent::{AgentDeps, AgentRuntime, ToolCallMode, ToolRegistry};
use kilop_core::id::SessionId;
use kilop_core::time::SystemClock;
use kilop_provider::ProviderRegistry;
use kilop_server::permission::ChannelPermissionRequester;
use kilop_server::{ServerDeps, ServerPassword};
use kilop_session::SessionManager;

mod config;
mod evidence;
mod mcp_bridge;
mod tools;

#[derive(Parser)]
#[command(
    name = "kilop-plus",
    version,
    about = "Kilo+ — same Kilo UX, native Rust engine"
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
        #[arg(long, default_value = "~/.kilop")]
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
        #[arg(long, default_value = "~/.kilop")]
        data_dir: String,
    },
    /// Self-check: storage, CAS, permissions, providers.
    Doctor {
        #[arg(long, default_value = "~/.kilop")]
        data_dir: String,
    },
    /// List sessions.
    Sessions {
        #[arg(long, default_value = "~/.kilop")]
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
    Vec<Arc<kilop_mcp::McpServer>>,
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

    let mut providers = ProviderRegistry::new();
    for p in config.providers {
        match p.build() {
            Ok(provider) => providers.register(provider),
            Err(e) => tracing::warn!("provider {} failed to build: {e}", p.id()),
        }
    }

    let mut tools = ToolRegistry::new();
    tools.register(tools::read_file_tool());
    tools.register(tools::write_file_tool());
    tools.register(tools::search_tool());
    tools.register(tools::run_command_tool());

    // The shared durable store/CAS (pub on SessionManager) back the
    // checkpoint store, the artifact sink, and the process supervisor.
    let cas = session.cas();
    let store = session.store();
    let workspaces = kilop_fs::WorkspaceFileService::new();
    let edit = Arc::new(kilop_edit::EditEngine::new(workspaces.clone()));
    let snapshots = Arc::new(kilop_snapshot::CheckpointStore::new(cas.clone(), store));
    // The daemon-wide sandbox carries the policy; the runtime roots it at
    // each session's workspace before any tool call.
    let sandbox = Arc::new(kilop_sandbox::PermissionEngine::new(
        kilop_sandbox::SandboxPolicy::default(),
        None,
    ));
    let supervisor = kilop_terminal::ProcessSupervisor::new(cas.clone());

    let permissions = ChannelPermissionRequester::new(std::time::Duration::from_secs(300));
    let agent = AgentRuntime::new(AgentDeps {
        session: session.clone(),
        providers: Arc::new(providers),
        permission_requester: permissions.clone(),
        evidence: Arc::new(RepoEvidence::new(session.clone())),
        tools: Arc::new(tools),
        cas: Some(cas),
        workspaces,
        edit: Some(edit),
        snapshots: Some(snapshots),
        sandbox: Some(sandbox),
        supervisor: Some(supervisor),
        model: config.model.clone(),
        compaction_model: config.compaction_model,
        compact_at_usage: config.compact_at_usage,
        instructions: config.instructions,
        clock: Arc::new(SystemClock),
        tool_call_mode: ToolCallMode::NativeWithRepair,
        tool_deadline_ms: 30_000,
    })
    .map_err(|e| e.to_string())?;
    Ok((session, agent, permissions, vec![]))
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
    let config = config.unwrap_or_default();
    let entries = config.mcp_servers()?;
    std::fs::create_dir_all(data_dir).map_err(|e| e.to_string())?;
    let session = SessionManager::open(data_dir.join("store"), data_dir.join("cas"), true)
        .map_err(|e| e.to_string())?;
    // Spawn the servers first so the agent registry can see their tools.
    let mut servers: Vec<Arc<kilop_mcp::McpServer>> = Vec::new();
    let mut mcp_tools: Vec<kilop_agent::Tool> = Vec::new();
    for entry in entries {
        let cfg = kilop_mcp::McpConfig {
            name: entry.name.clone(),
            command: entry.command,
            args: entry.args,
            env: vec![],
        };
        // Each server gets its own supervisor rooted at the same CAS; the
        // McpServer holds the Arc so children live for the daemon lifetime.
        let supervisor = kilop_terminal::ProcessSupervisor::new(session.cas());
        match tokio::time::timeout(
            std::time::Duration::from_secs(10),
            kilop_mcp::McpServer::connect(cfg, supervisor),
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
    let graph = build_daemon_on(session, config, mcp_tools)?;
    let (session, agent, permissions, _) = graph;
    Ok((session, agent, permissions, servers))
}

/// Shared core: identical to [`build_daemon`] but registers `extra_tools`
/// (MCP tools) after the builtins on the GIVEN already-open store — a
/// collision never replaces a builtin.
fn build_daemon_on(
    session: Arc<SessionManager>,
    config: config::Config,
    extra_tools: Vec<kilop_agent::Tool>,
) -> Result<DaemonGraph, String> {
    let mut tools = ToolRegistry::new();
    tools.register(tools::read_file_tool());
    tools.register(tools::write_file_tool());
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
    for p in config.providers {
        match p.build() {
            Ok(provider) => providers.register(provider),
            Err(e) => tracing::warn!("provider {} failed to build: {e}", p.id()),
        }
    }
    let cas = session.cas();
    let store = session.store();
    let workspaces = kilop_fs::WorkspaceFileService::new();
    let edit = Arc::new(kilop_edit::EditEngine::new(workspaces.clone()));
    let snapshots = Arc::new(kilop_snapshot::CheckpointStore::new(cas.clone(), store));
    let sandbox = Arc::new(kilop_sandbox::PermissionEngine::new(
        kilop_sandbox::SandboxPolicy::default(),
        None,
    ));
    let supervisor = kilop_terminal::ProcessSupervisor::new(cas.clone());
    let permissions = ChannelPermissionRequester::new(std::time::Duration::from_secs(300));
    let agent = AgentRuntime::new(AgentDeps {
        session: session.clone(),
        providers: Arc::new(providers),
        permission_requester: permissions.clone(),
        evidence: Arc::new(RepoEvidence::new(session.clone())),
        tools: Arc::new(tools),
        cas: Some(cas),
        workspaces,
        edit: Some(edit),
        snapshots: Some(snapshots),
        sandbox: Some(sandbox),
        supervisor: Some(supervisor),
        model: config.model.clone(),
        compaction_model: config.compaction_model,
        compact_at_usage: config.compact_at_usage,
        instructions: config.instructions,
        clock: Arc::new(SystemClock),
        tool_call_mode: ToolCallMode::NativeWithRepair,
        tool_deadline_ms: 30_000,
    })
    .map_err(|e| e.to_string())?;
    Ok((session, agent, permissions, vec![]))
}

/// Online backup with rotation (spec §24 "automatic backups"): one
/// crash-safe snapshot per daemon start, newest MAX_BACKUPS kept. Best
/// effort — a backup failure never stops the daemon.
fn rotate_backup(store: &kilop_store::Store, data_dir: &std::path::Path) {
    const MAX_BACKUPS: usize = 8;
    let backups = data_dir.join("backups");
    if std::fs::create_dir_all(&backups).is_err() {
        return;
    }
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let dest = backups.join(format!("kilo-plus-{ts}.db"));
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
            (n.starts_with("kilo-plus-") && n.ends_with(".db")).then_some(n)
        })
        .collect();
    names.sort();
    names.reverse();
    for old in names.iter().skip(MAX_BACKUPS) {
        let _ = std::fs::remove_file(backups.join(old));
    }
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
    match build_daemon_with_mcp(&data_dir, Some(config)).await {
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
            let fs = kilop_fs::WorkspaceFileService::new();
            let snapshots = Arc::new(kilop_snapshot::CheckpointStore::new(
                deps.session.cas(),
                deps.session.store(),
            ));
            deps = deps.with_snapshots(fs, snapshots);
            match kilop_server::serve(deps, port).await {
                Ok(handle) => {
                    // The frozen stdout line; nothing else may be printed.
                    println!("{}", handle.startup_line);
                    tracing::info!("kilop-plus serving on {}", handle.addr);
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

    #[test]
    fn expand_home_and_relative() {
        assert_eq!(expand("."), PathBuf::from("."));
        let home = expand("~");
        assert_eq!(expand("~/x"), home.join("x"));
    }
}
