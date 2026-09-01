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
use kilop_agent::{AgentDeps, AgentRuntime, NoEvidence, ToolCallMode, ToolRegistry};
use kilop_core::id::SessionId;
use kilop_core::time::SystemClock;
use kilop_provider::ProviderRegistry;
use kilop_server::permission::ChannelPermissionRequester;
use kilop_server::{ServerDeps, ServerPassword};
use kilop_session::SessionManager;

mod config;
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

/// The daemon dependency graph (session, agent, permissions).
pub type DaemonGraph = (
    Arc<SessionManager>,
    Arc<AgentRuntime>,
    Arc<ChannelPermissionRequester>,
);

/// Build the full daemon dependency graph (providers, tools, session,
/// agent, permissions).
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

    let permissions = ChannelPermissionRequester::new(std::time::Duration::from_secs(300));
    let agent = AgentRuntime::new(AgentDeps {
        session: session.clone(),
        providers: Arc::new(providers),
        permission_requester: permissions.clone(),
        evidence: Arc::new(NoEvidence),
        tools: Arc::new(tools),
        cas: None,
        model: config.model.clone(),
        compaction_model: config.compaction_model,
        compact_at_usage: config.compact_at_usage,
        instructions: config.instructions,
        clock: Arc::new(SystemClock),
        tool_call_mode: ToolCallMode::NativeWithRepair,
        tool_deadline_ms: 30_000,
    })
    .map_err(|e| e.to_string())?;
    Ok((session, agent, permissions))
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
    match build_daemon(&data_dir, Some(config)) {
        Ok((session, agent, permissions)) => {
            // Crash recovery runs before the first request (spec §7).
            if let Err(e) = agent.recover() {
                tracing::error!("recovery failed: {e}");
            }
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
        Ok((session, agent, _permissions)) => {
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
