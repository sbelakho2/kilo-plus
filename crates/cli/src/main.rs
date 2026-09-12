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
use faktor_core::CapabilitySet;
use faktor_provider::egress::{HttpTransport, OutboundScanConfig, PolicyCheckedHttpTransport};
use faktor_provider::{Provider, ProviderRegistry};
use faktor_security::registry::SecretRegistry;
use faktor_server::permission::ChannelPermissionRequester;
use faktor_server::{ServerDeps, ServerPassword};
use faktor_session::SessionManager;
use faktor_terminal::{ProcessOwner, ProcessSupervisor};
use serde_json::{json, Value};

mod config;
mod evidence;
mod graph;
mod mcp_bridge;
mod tools;

use graph::DaemonGraph;

#[derive(Parser)]
#[command(
    name = "faktor",
    version,
    about = "Faktor — native Rust agent engine, daemon and CLI"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Start the daemon; prints the frozen v7.5.6 startup line on stdout.
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
        /// verification, global recovery-row scan, dangling CAS references,
        /// journal projection consistency and the audit invariants (dangling
        /// cost reservations, verification-record/task consistency, active-
        /// turn recoverable owners, orphan children, process ownership).
        /// Plain mode keeps the bounded quick checks.
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

/// Build the full daemon dependency graph (audit 12/17): every lifetime
/// authority is built EXACTLY ONCE in the order of
/// [`graph::DAEMON_CONSTRUCTION_ORDER`] — steps 1-2 (store/session + CAS)
/// run here, step 3 (the ONE supervisor) right after, and steps 4-16 are
/// inline in [`build_daemon_core`]. Serve/ACP/commands never construct a
/// supervisor, ledger, index or executor of their own — they take
/// references from the returned graph. The real filesystem stack is wired
/// in the core: workspace service, transactional edit engine, CAS-backed
/// checkpoints, sandbox policy engine, and the process supervisor.
pub fn build_daemon(
    data_dir: &std::path::Path,
    config: Option<config::Config>,
) -> Result<DaemonGraph, String> {
    let config = config.unwrap_or_default();
    std::fs::create_dir_all(data_dir).map_err(|e| e.to_string())?;
    // 1-2: the durable store + CAS (full integrity scan on this entry).
    let session = SessionManager::open(data_dir.join("store"), data_dir.join("cas"), true)
        .map_err(|e| e.to_string())?;
    // 3: ONE daemon supervisor (audit P0-40): the SAME Arc supervises every
    // MCP server child, every hook child, every tool/terminal child.
    let supervisor = ProcessSupervisor::new(session.cas());
    build_daemon_core(
        data_dir,
        session,
        supervisor,
        config,
        vec![],
        None,
        graph::SemanticCfg::default(),
    )
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
    build_daemon_with_mcp_inner(
        data_dir,
        config,
        chunk_tx,
        false,
        graph::SemanticCfg::default(),
    )
    .await
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
    semantic: graph::SemanticCfg,
) -> Result<DaemonGraph, String> {
    build_daemon_with_mcp_inner(data_dir, config, chunk_tx, true, semantic).await
}

async fn build_daemon_with_mcp_inner(
    data_dir: &std::path::Path,
    config: Option<config::Config>,
    chunk_tx: Option<std::sync::Arc<faktor_agent::ChunkSink>>,
    fast_open: bool,
    semantic: graph::SemanticCfg,
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
    // ONE daemon supervisor (audit P0-40): the SAME Arc supervises every
    // MCP server child, every hook child, every tool/terminal child. The
    // servers are spawned first so the agent registry can see their tools.
    let supervisor = ProcessSupervisor::new(session.cas());
    let mut servers: Vec<Arc<faktor_mcp::McpServer>> = Vec::new();
    let mut mcp_tools: Vec<faktor_agent::Tool> = Vec::new();
    for entry in entries {
        let cfg = faktor_mcp::McpConfig {
            name: entry.name.clone(),
            command: entry.command,
            args: entry.args,
            env: vec![],
        };
        // Shared daemon supervisor: the McpServer holds the Arc so the
        // child lives for the daemon lifetime, in the SAME bounded
        // registry as hooks and terminals.
        match tokio::time::timeout(
            std::time::Duration::from_secs(10),
            faktor_mcp::McpServer::connect(cfg, supervisor.clone()),
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
    // Now build the core graph on the SAME store with the MCP tools (steps
    // 4-16 of the construction order; the servers already ride the ONE
    // supervisor above).
    let mut graph = build_daemon_core(
        data_dir, session, supervisor, config, mcp_tools, chunk_tx, semantic,
    )?;
    graph.mcp_servers = servers;
    Ok(graph)
}

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

/// The capability envelope the daemon grants its hook registry: the full
/// lattice (the operator-configured FAKTOR_HOOKS commands are as trusted
/// as the daemon env that named them — the envelope check refuses only
/// scopes a future config surface tries to grant beyond this).
const DAEMON_HOOK_ENVELOPE: CapabilitySet = CapabilitySet::ALL;

/// Build an optional lifecycle-hook registry over the DAEMON supervisor
/// (audit P0-40). Each parsed spec is logged; a spec the registry rejects
/// is a loud warning, never a daemon failure.
fn env_hook_registry(
    supervisor: &Arc<ProcessSupervisor>,
) -> Option<Arc<faktor_hooks::HookRegistry>> {
    let specs = parse_hooks_env(&std::env::var("FAKTOR_HOOKS").unwrap_or_default());
    hook_registry(supervisor, specs)
}

/// Shared hook-registry construction (test seam + env path): the registry
/// is rooted at the GIVEN supervisor and granted the daemon envelope, so
/// every hook child lands in the daemon's single bounded registry.
fn hook_registry(
    supervisor: &Arc<ProcessSupervisor>,
    specs: Vec<faktor_hooks::HookSpec>,
) -> Option<Arc<faktor_hooks::HookRegistry>> {
    if specs.is_empty() {
        return None;
    }
    let registry = Arc::new(faktor_hooks::HookRegistry::with_supervisor(
        supervisor.clone(),
        DAEMON_HOOK_ENVELOPE,
    ));
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

/// Durable workspace roots for the instruction resolver (P0-32 + P0-48):
/// every resolution goes through the daemon's SessionManager workspace
/// table — the process CWD and any static config default root are NEVER
/// consulted. A workspace row without a root (or an unknown workspace id)
/// resolves to `None`, which the resolver turns into the documented Empty
/// instruction set. While EXACTLY ONE session of the workspace carries a
/// live shadow row (a shadowed single-agent drive), the workspace's rules
/// resolve from that shadow root ([tasks] shadow_mutation): the drive reads
/// the instruction environment of the world it mutates. Ambiguity (more
/// than one live shadow — hostile residue the executor discipline never
/// produces) degrades loudly to the stored root, never a guess.
struct SessionWorkspaceRoots(Arc<SessionManager>);

impl SessionWorkspaceRoots {
    fn resolve(&self, workspace_id: u64) -> Option<PathBuf> {
        if workspace_id == 0 {
            return None;
        }
        let ws = faktor_core::id::WorkspaceId::new(workspace_id);
        match self.0.live_workspace_shadow_root(ws) {
            Ok(Some(shadow)) => {
                tracing::debug!(
                    workspace = %workspace_id, shadow = %shadow.display(),
                    "workspace instructions resolve from the live shadow root (shadow mutation drive)"
                );
                Some(shadow)
            }
            Ok(None) => match self.0.workspace_root(ws) {
                Ok(Some(root)) => Some(root),
                Ok(None) => None,
                Err(e) => {
                    tracing::warn!(error = %e, workspace = %workspace_id,
                        "durable workspace-root lookup failed; resolving no instructions for this session");
                    None
                }
            },
            Err(e) => {
                tracing::warn!(error = %e, workspace = %workspace_id,
                    "ambiguous live shadows on this workspace; resolving instructions from the stored workspace root");
                match self.0.workspace_root(ws) {
                    Ok(Some(root)) => Some(root),
                    Ok(None) => None,
                    Err(e) => {
                        tracing::warn!(error = %e, workspace = %workspace_id,
                            "durable workspace-root lookup failed; resolving no instructions for this session");
                        None
                    }
                }
            }
        }
    }
}

impl faktor_instructions::WorkspaceRootProvider for SessionWorkspaceRoots {
    fn workspace_root(&self, workspace_id: u64) -> Option<PathBuf> {
        self.resolve(workspace_id)
    }
}

/// The daemon's per-workspace instruction resolver (P0-32): ONE resolver
/// over the real SessionManager, shared by every AgentDeps. Roots are
/// resolved per session from the durable workspace table at resolve time —
/// there is no single default root at daemon build (audit 31's per-session
/// wiring, now implemented instead of a None loader).
fn daemon_instructions_resolver(
    session: &Arc<SessionManager>,
) -> Arc<faktor_instructions::InstructionResolver> {
    Arc::new(faktor_instructions::InstructionResolver::new(
        Arc::new(SessionWorkspaceRoots(session.clone())),
        faktor_instructions::DEFAULT_RESOLVER_CACHE_ENTRIES,
    ))
}

/// The outbound secret-scan config of the daemon's provider transports:
/// every request body is whole-payload scanned with a [`SecretRegistry`]
/// fed from the CONFIGURED provider keys (the same env values the adapter
/// constructions read). Keys are registered without any logging — the
/// registry's Debug stays redacted (counts only).
fn daemon_outbound_scan(config: &config::Config) -> OutboundScanConfig {
    let mut registry = SecretRegistry::new();
    for p in &config.providers {
        if let Some(key) = p.key() {
            registry.register(key.as_bytes());
        }
    }
    OutboundScanConfig {
        registry: Some(Arc::new(registry)),
        ..Default::default()
    }
}

/// The ONE egress transport every configured adapter executes through:
/// policy-checked with the daemon's SandboxPolicy network gate installed
/// (default-deny on any destination the allowlist does not match, BEFORE a
/// connect) and the outbound whole-payload secret scan attached.
fn daemon_egress_transport(
    policy: &faktor_sandbox::SandboxPolicy,
    scan: OutboundScanConfig,
) -> Arc<dyn HttpTransport> {
    Arc::new(PolicyCheckedHttpTransport::with_policy_and_scan(
        policy.network.installed().cloned(),
        Some(scan),
    ))
}

/// The daemon verification service from the configured `[verification]`
/// section: sane values build the typed service under the section's
/// policy; `quick_max_s: 0` yields the DISABLED service (fail closed —
/// mutating turns classify Unverified, never silently complete).
///
/// The executor rides THE daemon supervisor (audit P0-5/P0-6 process
/// consolidation): verification shares the single process runtime — its
/// live-child ceiling, its capture ring and its whole-tree kill paths —
/// instead of spawning a second process layer. The graph core calls this
/// at step 12 of the construction order with a FIELD borrow of the config
/// (the provider loop has consumed the rest of the config by then).
fn daemon_verification(
    section: &config::VerificationCfg,
    supervisor: &Arc<ProcessSupervisor>,
) -> Arc<faktor_agent::VerificationService> {
    match section.policy() {
        Some(policy) => faktor_agent::VerificationService::new(
            Arc::new(faktor_verify::exec::AsyncCheckExecutor::from_supervisor(
                supervisor.clone(),
            )),
            policy,
        ),
        None => faktor_agent::VerificationService::disabled(),
    }
}

/// Map the parsed `[efficiency]` section onto the agent's flag type
/// (additive; every flag defaults `false`). The runtime applies
/// `failure_learning` to the context prior; the remaining flags are carried
/// by `AgentDeps` for their efficiency components.
fn efficiency_flags(cfg: &config::EfficiencyCfg) -> faktor_agent::EfficiencyFlags {
    faktor_agent::EfficiencyFlags {
        failure_learning: cfg.failure_learning,
        ccr: cfg.ccr,
        typed_handoff: cfg.typed_handoff,
        semantic_context: cfg.semantic_context,
        rework_routing: cfg.rework_routing,
    }
}

use faktor_learning::LearningStore as _;

/// The production failure-learning prior adapter (audit 68, closing audits
/// 65-69/82): the planner's
/// [`faktor_context::information::FailurePrior`] over the DURABLE learning
/// corpora of the daemon's session manager, snapshotted as a key -> risk
/// index so a per-candidate lookup is one map probe with no allocation.
///
/// The index is built through
/// [`faktor_learning::SessionLearningStore`] over the sessions' typed
/// `learning_record` ledger rows: no learning state lives only in memory,
/// and a daemon reopen re-reads the same rows (reopen-safe). The cached
/// index is refreshed when the per-session ledger stamp advances, so a
/// learning mined DURING this daemon's lifetime protects its matching
/// evidence candidate on the next plan — the production loop closes without
/// a restart.
///
/// Lookup is keyed by [`faktor_context::ContextCandidate::omission_keys`]
/// — the learning identities the wire planner surfaces from
/// `learning:<digest>` evidence paths — never by the candidate's render id.
///
/// Hostile-value contract: every risk is clamped to `[1, 2]` by
/// `omission_risk_of` and is always finite, so this adapter can only ever
/// PROTECT a candidate up to 2x. Panics are NOT caught by the runtime (the
/// planner consults this daemon-global handle in-process), so this adapter
/// is total: a poisoned mutex is recovered, hostile keys are ignored, and a
/// corrupt ledger row is logged loudly while that session's corpus stays
/// neutral — never a failed turn.
struct LearningRiskPrior {
    session: Arc<SessionManager>,
    cache: std::sync::Mutex<RiskCache>,
}

/// The cached merged omission-risk index plus the durable stamp it was built
/// from: `(session id, newest ledger seq)` sorted by session id. The stamp
/// changes exactly when a session's typed ledger advances — the only way a
/// learning corpus can grow.
#[derive(Debug, Default)]
struct RiskCache {
    stamp: Vec<(u64, i64)>,
    risks: std::collections::HashMap<faktor_core::FileHash, f64>,
}

impl LearningRiskPrior {
    /// Build the adapter over the daemon's session manager, snapshotting
    /// every durable corpus ONCE (reopen-safe). An empty corpus yields an
    /// empty index, so every lookup is neutral (`1.0`) and context selection
    /// stays byte-identical to the flag-off path.
    fn from_session(session: Arc<SessionManager>) -> Self {
        let stamp = Self::stamp(&session);
        let risks = Self::merged_risks(&session);
        tracing::info!(
            learnings = risks.len(),
            "failure_learning enabled: durable learning corpora read from the session manager (empty corpus => neutral risk 1.0)"
        );
        Self {
            session,
            cache: std::sync::Mutex::new(RiskCache { stamp, risks }),
        }
    }

    /// Cheap durable stamp of every session's typed ledger (one session-list
    /// query plus one indexed MAX per session). A failed read contributes a
    /// `-1` sentinel so the next successful read still differs and forces a
    /// rebuild instead of silently trusting a stale corpus.
    fn stamp(session: &Arc<SessionManager>) -> Vec<(u64, i64)> {
        let store = session.store();
        let rows = match store.list_sessions(None) {
            Ok(rows) => rows,
            Err(error) => {
                tracing::error!(%error, "learning prior: session list read failed; corpus stamp is unknown");
                return Vec::new();
            }
        };
        let mut stamp: Vec<(u64, i64)> = rows
            .iter()
            .map(|row| (row.id.raw(), store.ledger_max_seq(row.id).unwrap_or(-1)))
            .collect();
        stamp.sort_unstable();
        stamp
    }

    /// Merge every session's durable corpus index (keyed by pattern AND
    /// failure digest, max risk per key). A corrupt/unreadable corpus is
    /// LOUD but non-fatal: that session contributes nothing (neutral), the
    /// rest of the daemon keeps serving, and the next stamp check retries.
    fn merged_risks(
        session: &Arc<SessionManager>,
    ) -> std::collections::HashMap<faktor_core::FileHash, f64> {
        let mut risks = std::collections::HashMap::new();
        let rows = match session.list_sessions(None) {
            Ok(rows) => rows,
            Err(error) => {
                tracing::error!(%error, "learning prior: session list read failed; every corpus stays neutral");
                return risks;
            }
        };
        for handle in rows {
            let corpus = match faktor_learning::SessionLearningStore::open(
                handle.clone(),
                faktor_learning::DEFAULT_MEMORY_CAPACITY,
            ) {
                Ok(corpus) => corpus,
                Err(error) => {
                    tracing::error!(
                        session = %handle.id(),
                        %error,
                        "learning prior: corrupt/unreadable learning corpus; this session stays neutral"
                    );
                    continue;
                }
            };
            for learning in corpus.all() {
                let risk = faktor_learning::omission_risk_of(learning.confidence_ppm);
                for key in [learning.pattern_digest(), learning.pattern.failure.digest()] {
                    risks
                        .entry(key)
                        .and_modify(|existing: &mut f64| *existing = existing.max(risk))
                        .or_insert(risk);
                }
            }
        }
        risks
    }

    /// The current cache, rebuilt from the durable ledger rows when the
    /// stamp advanced.
    fn cache(&self) -> std::sync::MutexGuard<'_, RiskCache> {
        let mut cache = match self.cache.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let stamp = Self::stamp(&self.session);
        if stamp != cache.stamp {
            let risks = Self::merged_risks(&self.session);
            tracing::info!(
                learnings = risks.len(),
                "learning prior: corpus stamp advanced; durable learning index re-read"
            );
            cache.stamp = stamp;
            cache.risks = risks;
        }
        cache
    }
}

impl faktor_context::information::FailurePrior for LearningRiskPrior {
    fn omission_risk(&self, candidate: &faktor_context::ContextCandidate) -> f64 {
        // Non-learning candidates expose no keys: neutral without touching
        // the durable stamp (the overwhelmingly common case).
        if candidate.omission_keys.is_empty() {
            return faktor_learning::OMISSION_RISK_NEUTRAL;
        }
        let cache = self.cache();
        faktor_learning::omission_risk_for_keys(&cache.risks, &candidate.omission_keys)
    }
}

/// Build the daemon's failure-learning prior handle (audit 68): `Some` ONLY
/// when `[efficiency] failure_learning` is on; `None` otherwise (the
/// runtime then takes the byte-identical baseline planner path). `Some`
/// always carries the durable corpus over the daemon's session manager.
fn daemon_context_prior(
    enabled: bool,
    session: &Arc<SessionManager>,
) -> Option<Arc<dyn faktor_context::information::FailurePrior + Send + Sync>> {
    if !enabled {
        return None;
    }
    Some(Arc::new(LearningRiskPrior::from_session(session.clone())))
}

/// The graph construction core (audit 12/17): steps 4-16 of
/// [`graph::DAEMON_CONSTRUCTION_ORDER`] are built HERE, inline and in the
/// documented order (the ordering test scans this function's body).
/// Callers open the store and create the ONE supervisor (steps 1-3) — the
/// async MCP connect must ride that same supervisor — and everything else
/// of the daemon lifetime is this function's construction.
fn build_daemon_core(
    data_dir: &std::path::Path,
    session: Arc<SessionManager>,
    supervisor: Arc<ProcessSupervisor>,
    config: config::Config,
    extra_tools: Vec<faktor_agent::Tool>,
    chunk_tx: Option<std::sync::Arc<faktor_agent::ChunkSink>>,
    semantic: graph::SemanticCfg,
) -> Result<DaemonGraph, String> {
    // Step 4 — checked transport/security: the daemon's ONE sandbox policy
    // from the `[sandbox]` section (destination gate + OS-level
    // network-isolation guarantee), the outbound whole-payload secret scan
    // (P0-36/37/38; keys registered without logging — Debug stays redacted),
    // and the ONE policy-checked + secret-scanned egress transport every
    // configured adapter executes through. No adapter is ever constructed
    // with a permissive default transport.
    let sandbox_policy = config
        .sandbox_policy()
        .map_err(|e| format!("sandbox config: {e}"))?;
    let egress = daemon_outbound_scan(&config);
    let transport = daemon_egress_transport(&sandbox_policy, egress);
    // Steps 5-6 — provider registry + catalog/pricing: every configured
    // adapter is built through the checked transport; Ollama providers are
    // kept CONCRETE for live probing (spec §10: warm-up must reach the
    // instance the registry serves). Catalog/pricing rows (built-in
    // tables + configured overrides/ceilings) ride the adapter
    // constructions and the registry's catalog rows.
    let mut providers = ProviderRegistry::new();
    let mut ollama_warmers: Vec<Arc<faktor_ollama::OllamaProvider>> = Vec::new();
    for p in config.providers {
        if let Some(ollama) = p.build_ollama(transport.clone()) {
            let dyn_arc: Arc<dyn Provider> = ollama.clone();
            providers
                .try_register(dyn_arc)
                .map_err(|e| format!("provider {} failed to register: {e}", p.id()))?;
            ollama_warmers.push(ollama);
            continue;
        }
        match p.build(transport.clone()) {
            Ok(provider) => providers
                .try_register(provider)
                .map_err(|e| format!("provider {} failed to register: {e}", p.id()))?,
            Err(e) => tracing::warn!("provider {} failed to build: {e}", p.id()),
        }
    }
    let providers = Arc::new(providers);
    let store = session.store();
    let cas = session.cas();
    // Step 7 — economic routing + the durable verified-outcome registry
    // (P0-2/6/12): the routing policy is built from the REGISTERED
    // providers + the config's mode (Economy default; a Pinned mode that
    // names an unregistered provider/model refuses the daemon at boot —
    // never a silent Economy). The outcome registry rides this store
    // (audit items 13/14/L): verified samples the runtime records at the
    // deterministic gate sites land here and every later route consult
    // reads them back — routing and the recorded outcome history share
    // one store-backed registry, exactly like the pricing authority.
    let routing_mode = config
        .routing_mode
        .clone()
        .unwrap_or(faktor_core::model::RoutingMode::Economy);
    let routing = graph::economic_routing_policy_with_outcomes(
        &providers,
        routing_mode,
        Arc::new(faktor_agent::StoreOutcomeStore::new(store.clone())),
    )
    .map_err(|e| format!("routing config error: {e}"))?;
    // Step 8 — the durable cost ledger over THIS daemon's store (P0-6/12):
    // one reservation per paid model call, settled exactly once. Crash
    // recovery abandons every OPEN reservation of a previous process BEFORE
    // the first turn (never counted as spent).
    let budgets = faktor_session::DurableBudgetLedger::new(session.clone());
    budgets.recover_after_restart();
    // Step 9 — the repository IndexService (audits 30/64) over the SAME
    // store + workspace service + data root the runtime's evidence ladder
    // hosts; the graph pre-hosts the durable index authority at boot. A
    // hostile/unwritable data root degrades exactly like the runtime's own
    // lazy host: warn + None — the daemon keeps serving on the bounded
    // evidence scan, never a broken first prompt.
    let workspaces = faktor_fs::WorkspaceFileService::new();
    let index_data_root = store
        .path()
        .parent()
        .map(|p| p.join("index_data"))
        .unwrap_or_else(|| std::path::PathBuf::from("index_data"));
    let index = match faktor_index::IndexService::open(
        store.clone(),
        index_data_root,
        workspaces.clone(),
    ) {
        Ok(svc) => {
            tracing::info!("repository IndexService hosted");
            Some(svc)
        }
        Err(e) => {
            tracing::warn!("repository IndexService unavailable: {e}");
            None
        }
    };
    // Step 10 — evidence/cold: the daemon's evidence provider (spec §20):
    // the bounded per-workspace scan + search every session's context
    // engine consults while the index has no Ready generation.
    let repo_evidence = Arc::new(RepoEvidence::new(session.clone()));
    // Step 11 — per-workspace repository instructions (P0-32): the
    // resolver is built over the daemon's SessionManager workspace table
    // ONCE — every later resolution reads a session's DURABLE workspace
    // root, never a process CWD and never a static config default root.
    // Sessions whose workspace carries no root resolve to an Empty set.
    let instructions_resolver = daemon_instructions_resolver(&session);
    // Step 12 — the typed verification engine (P0-9/10 migration): REQUIRED
    // checks the agent derives from its OWN file changes execute as
    // (program, argv) specs through the async executor ON THE DAEMON
    // SUPERVISOR (audit P0-5/P0-6) — never `sh -c`, never a second process
    // runtime. Budgets come from the configured [verification] section
    // (defaults: quick <= 60 s, unit <= 600 s inline, full = durable
    // background verification jobs; quick_max_s 0 = disabled + fail closed).
    let verification = daemon_verification(&config.verification, &supervisor);
    // Steps 13-16 — semantic + learning + memory + tokenizers: the ONE
    // semantic-provider registry built from the strict `[semantic]` section
    // over THIS daemon's supervisor + checked transport (the SAME Arc flows
    // to the agent and the server introspection surface); the durable
    // failure-learning prior handle built once over this daemon's session
    // store; the project-memory authority over the SAME store; and the
    // tokenizer registry with the real local backends (unregistered
    // identities keep their conservative UpperBound label).
    let semantic = graph::semantic_registry(&semantic, &supervisor, &transport)?;
    let learning = daemon_context_prior(config.efficiency.failure_learning, &session);
    let memory = graph::DaemonMemory::new(store.clone());
    let tokenizers = Arc::new(faktor_context::TokenizerRegistry::with_builtin_backends());
    // The builtin tool registry + the MCP tools (a collision never replaces
    // a builtin) and the engine layer the runtime hands its tools: edit
    // engine, CAS-backed checkpoints, the permission engine over the
    // daemon's sandbox policy, the permission channel, and the lifecycle
    // hooks rooted at the daemon's ONE supervisor.
    let mut tools = ToolRegistry::new();
    tools.register(tools::read_file_tool());
    tools.register(tools::write_file_tool());
    tools.register(tools::edit_file_tool());
    tools.register(tools::search_tool());
    tools.register(tools::run_command_tool());
    // Coordination board: the tools are agent-crate definitions, the board
    // authority is THIS daemon's session manager (run-family scoped).
    let board_gateway = Arc::new(tools::SessionBoardGateway::new(session.clone()));
    tools.register(faktor_agent::board_post_tool(board_gateway.clone()));
    tools.register(faktor_agent::board_read_tool(board_gateway));
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
    let edit = Arc::new(faktor_edit::EditEngine::new(workspaces.clone()));
    let snapshots = Arc::new(faktor_snapshot::CheckpointStore::new(cas.clone(), store));
    let sandbox = Arc::new(faktor_sandbox::PermissionEngine::new(sandbox_policy, None));
    let permissions = ChannelPermissionRequester::new(std::time::Duration::from_secs(300));
    let hooks = env_hook_registry(&supervisor);
    // Step 13 — the AgentRuntime over every authority above: the routing
    // policy, the budget ledger, the evidence provider, the instruction
    // resolver, the verification service and the ONE supervisor all enter
    // the runtime through this single deps literal.
    let agent = AgentRuntime::new(AgentDeps {
        session: session.clone(),
        providers: providers.clone(),
        chunk_sink: chunk_tx,
        permission_requester: permissions.clone(),
        evidence: repo_evidence.clone(),
        tools: Arc::new(tools),
        cas: Some(cas),
        workspaces,
        edit: Some(edit),
        snapshots: Some(snapshots),
        sandbox: Some(sandbox),
        supervisor: Some(supervisor.clone()),
        verification: verification.clone(),
        hooks,
        instructions_resolver: instructions_resolver.clone(),
        routing: routing.clone(),
        budgets: budgets.clone(),
        model: config.model.clone(),
        compaction_model: config.compaction_model,
        compact_at_usage: config.compact_at_usage,
        instructions: config.instructions,
        clock: Arc::new(SystemClock),
        tool_call_mode: ToolCallMode::NativeWithRepair,
        tool_deadline_ms: 30_000,
        retry_policy: faktor_core::retry::RetryPolicy::default(),
        semantic: semantic.clone(),
        // Audit 68: the parsed `failure_learning` flag decides whether a
        // prior handle is installed (and the runtime then applies it);
        // `false` installs `None` and the whole `[efficiency]` section
        // rides the additive default. The handle was built ONCE above and
        // the graph holds the SAME Arc.
        context_prior: learning.clone(),
        efficiency: efficiency_flags(&config.efficiency),
    })
    .map_err(|e| e.to_string())?;
    for ollama in ollama_warmers {
        warm_ollama(ollama);
    }
    // Steps 14-16 — the orchestration authorities (audits P0-20/21/23/61,
    // P0-48 + wave-24): OrchestratorRuntime + ShadowRoots + the ONE
    // TaskExecutor over the SAME orchestrator. The executor ALWAYS carries
    // the shadow service; the configured MutationMode decides usage only —
    // DirectCompat keeps every run's direct behavior byte-identical.
    let orchestrator =
        faktor_orchestrator::runtime::OrchestratorRuntime::new(session.clone(), agent.clone());
    // Shadow mutation roots: rooted at `<data dir>/shadows`. reconcile()
    // at boot handles crash residue; the service's Drop removes every
    // shadow on graceful daemon shutdown.
    let shadows_root = data_dir.join(faktor_orchestrator::runtime::shadow::SHADOWS_DIR_NAME);
    let shadows =
        faktor_orchestrator::runtime::shadow::ShadowRoots::new(session.clone(), shadows_root);
    if let Err(e) = shadows.reconcile() {
        tracing::warn!(error = %e, "shadow reconcile after daemon start");
    }
    let tasks = faktor_orchestrator::runtime::task_executor::TaskExecutor::new_with_mode(
        &orchestrator,
        session.clone(),
        agent.clone(),
        Some(shadows.clone()),
        config.tasks.mutation_mode,
    );
    // P2 completion-step execution: the strict `[completion]` section feeds
    // the executor's commit/push/PR policy. An absent section keeps the
    // inert defaults (push origin, base main, unconfigured PR => Skipped);
    // an invalid section fails daemon startup instead of half-running.
    tasks
        .configure_completion_steps(
            config
                .completion
                .steps_config()
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
    // THE durable evidence authority is the runtime's own allocation: the
    // graph stores the same `Arc` the runtime's compiler/archiver use, and
    // serve hands the same `Arc` to the native server. No second authority
    // is constructed, so ids/scope/backing can never disagree between the
    // runtime and the server.
    let evidence = agent.evidence_authority().clone();
    Ok(DaemonGraph {
        session,
        supervisor,
        transport,
        providers,
        permissions,
        mcp_servers: vec![],
        routing,
        budgets,
        index,
        repo_evidence,
        instructions: instructions_resolver,
        verification,
        semantic,
        learning,
        memory,
        tokenizers,
        agent,
        evidence,
        orchestrator,
        shadows,
        tasks,
    })
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
/// name and are published into place only when complete (one atomic
/// `faktor_fs::atomic::atomic_adopt`), so they are invisible here by
/// construction.
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
/// The snapshot is written to a `.db.tmp-*` name and published as ONE
/// atomic step through `faktor_fs::atomic::atomic_adopt` (fsync the temp,
/// rename into place, fsync the directory), so a crash mid-backup can never
/// leave a partial file that reads as a complete backup (and the
/// gate/retention scans never see one). Best effort — a backup failure
/// never stops the daemon.
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
    if let Err(e) = faktor_fs::atomic::atomic_adopt(&tmp, &dest) {
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

/// Startup-backup task seam (P0-46): the async wrapper sleeps the
/// post-readiness delay, applies the interval/staleness gate, and runs the
/// SYNC snapshot+rotation on the blocking pool — never on a Tokio worker.
/// Returns the JoinHandle the shutdown path drains.
fn spawn_startup_backup(
    store: Arc<faktor_store::Store>,
    data_dir: std::path::PathBuf,
) -> tokio::task::JoinHandle<()> {
    tokio::task::spawn(async move {
        tokio::time::sleep(BACKUP_START_DELAY).await;
        if backup_due(&data_dir) {
            let store = store.clone();
            let dir = data_dir.clone();
            // spawn_blocking: the SQLite backup API is synchronous and can
            // hold the caller's thread for the whole snapshot; a Tokio
            // worker must never sit in it (P0-46 worker starvation).
            if let Err(e) = tokio::task::spawn_blocking(move || rotate_backup(&store, &dir)).await {
                tracing::warn!("startup backup worker failed: {e}");
            }
        } else {
            tracing::info!(
                "startup backup skipped: a backup newer than {BACKUP_MIN_INTERVAL_SECS}s exists"
            );
        }
    })
}

/// Shutdown drain for the startup-backup task: waits a bounded window for
/// the backup to finish (the daemon announced readiness long ago, so the
/// snapshot is normally long done), then aborts the async wrapper. The
/// abort cuts the sleep/gate, NOT the blocking snapshot (spawn_blocking
/// closures cannot be force-killed; the snapshot is bounded and finishes at
/// its own pace) — the daemon never waits unboundedly on shutdown.
async fn drain_startup_backup(mut backup_task: tokio::task::JoinHandle<()>) {
    const BACKUP_DRAIN_GRACE: std::time::Duration = std::time::Duration::from_secs(30);
    // `&mut JoinHandle` is a Future: the handle stays borrowable so the
    // timeout path can still abort the async wrapper.
    match tokio::time::timeout(BACKUP_DRAIN_GRACE, &mut backup_task).await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => tracing::warn!("startup backup task panicked: {e}"),
        Err(_) => {
            tracing::warn!("startup backup drain timed out; aborting the async wrapper");
            backup_task.abort();
        }
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

/// Config resolution for `serve` (audit 31): an EXPLICIT --config path must
/// load STRICTLY (parse + semantic validation) — any failure is an Err the
/// caller turns into a startup error (exit 1); the daemon never boots on a
/// config it cannot fully honor. Without --config, defaults + best-effort
/// discovery stay lenient and nothing here can fail startup.
/// Test-facing convenience over [`serve_config_and_semantic`].
#[cfg(test)]
fn serve_config(config_path: Option<PathBuf>) -> Result<config::Config, String> {
    Ok(serve_config_and_semantic(config_path)?.0)
}

/// The strict daemon config PLUS the additive `[semantic]` section (audits
/// 48-54/58/79). The section is extracted from the raw document BEFORE the
/// frozen `Config` shape parses the rest, so it is strictly additive with an
/// empty default: absent/null keeps [`graph::SemanticCfg::default`] and a
/// present section is parsed with `deny_unknown_fields` (a typo'd key fails
/// startup, never silently changes behavior). Every other key keeps exactly
/// the strict load semantics (`Config` parse + `validate`).
fn serve_config_and_semantic(
    config_path: Option<PathBuf>,
) -> Result<(config::Config, graph::SemanticCfg), String> {
    let Some(path) = config_path else {
        return Ok((config::Config::default(), graph::SemanticCfg::default()));
    };
    let text =
        std::fs::read_to_string(&path).map_err(|e| format!("config {}: {e}", path.display()))?;
    let mut value: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("config {}: {e}", path.display()))?;
    let semantic = match value
        .as_object_mut()
        .and_then(|object| object.remove("semantic"))
    {
        None | Some(serde_json::Value::Null) => graph::SemanticCfg::default(),
        Some(section) => serde_json::from_value(section)
            .map_err(|e| format!("config {} [semantic]: {e}", path.display()))?,
    };
    let config: config::Config =
        serde_json::from_value(value).map_err(|e| format!("config {}: {e}", path.display()))?;
    config
        .validate()
        .map_err(|e| format!("config {}: {e}", path.display()))?;
    // Strict provider-section validation (bounded caps, unique ids, valid
    // command/args/endpoint/timeout/auth-env for every external provider):
    // a hostile section refuses startup here, before any graph authority or
    // child process exists.
    semantic
        .validate()
        .map_err(|e| format!("config {} [semantic]: {e}", path.display()))?;
    Ok((config, semantic))
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
    let (config, semantic) = match serve_config_and_semantic(config_path) {
        Ok(loaded) => loaded,
        Err(e) => return Err(format!("config error: {e}")),
    };
    // Live chunk path (audit 41): BOUNDED channel (1024 events) + sink-side
    // coalescing under backpressure — a slow SSE consumer can never grow
    // the agent's memory. The drainer spawn lives in serve().
    let (chunk_sink, chunk_rx) = faktor_agent::ChunkSink::channel();
    // The whole graph is built HERE in the construction region (steps
    // 1-16 of graph::DAEMON_CONSTRUCTION_ORDER): store/session + CAS →
    // supervisor → checked transport → providers → router → budgets →
    // index → evidence → instructions → verification → agent →
    // orchestrator → shadows → tasks. Serve constructs NOTHING of its own.
    let graph =
        build_daemon_with_mcp_and_chunks_fast(&data_dir, Some(config), Some(chunk_sink), semantic)
            .await
            .map_err(|e| format!("daemon build failed: {e}"))?;
    let session = graph.session.clone();
    let agent = graph.agent.clone();
    let store = session.store();
    // Crash recovery runs before the first request (spec §7) — and before
    // bind, so all recovery work is done before readiness is announced.
    if let Err(e) = agent.recover() {
        tracing::error!("recovery failed: {e}");
    }
    // Step 17 — the server surface consumes the graph's authorities: the
    // orchestrator and the TaskExecutor of ServerDeps are the SAME
    // instances the graph built (no second execution authority is ever
    // constructed in a daemon lifetime). The graph stays alive for the
    // whole serve below, so every authority — supervisor, shadow service,
    // budget ledger, index — lives exactly as long as the daemon.
    let mut deps = ServerDeps::new_with(
        session,
        agent,
        graph.permissions.clone(),
        graph.orchestrator.clone(),
        graph.tasks.clone(),
        graph.budgets.clone(),
    );
    deps.chunk_rx = Some(chunk_rx);
    // The ONE semantic-provider registry: the SAME Arc the graph built and
    // the agent holds — the native introspection endpoints inspect only
    // `deps.semantic` (no parallel registry exists anywhere).
    deps = deps.with_semantic_registry(graph.semantic.clone());
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
    // Wire the daemon's DURABLE evidence store of record (audit 82/CCR):
    // the native server receives the graph's ONE authority (the SAME `Arc`
    // the runtime's ContextCompiler selects from and the runtime's archiver
    // inserts into). No parallel authority is constructed here, so ids are
    // globally unique across restart, scope checks are enforced by the one
    // authority, and a foreign session can never read even knowing a
    // backing digest.
    deps = deps.with_evidence_store(Arc::new(std::sync::RwLock::new(Box::new(
        graph.evidence.clone(),
    ))));
    // Bind BEFORE readiness and BEFORE any backup work (audit 44): the
    // historic code ran rotate_backup synchronously between recover() and
    // bind, so a slow or cold backup delayed first-request readiness.
    let handle = faktor_server::serve(deps, port)
        .await
        .map_err(|e| format!("failed to bind: {e}"))?;
    // The frozen stdout line; nothing else may be printed. Readiness is now
    // announced — no backup has run yet and, by construction, cannot have.
    println!("{}", handle.startup_line);
    tracing::info!("faktor serving on {}", handle.addr);
    if let Some(tx) = ready_tx {
        let _ = tx.send(());
    }
    // Automatic online backup (spec §24), post-ready and low priority: a
    // delayed spawned task, gated by policy (audit 44: interval +
    // staleness), so it never delays startup and never piles hourly
    // snapshots onto rapid restarts. Best effort, runs exactly once. The
    // SYNC backup/rotation work runs on the blocking pool
    // (`spawn_blocking` — P0-46: the sync SQLite snapshot used to run
    // inline on a Tokio worker, stalling every other task on that worker
    // for the whole snapshot); the async wrapper only sleeps, gates, and
    // joins.
    let backup_task = spawn_startup_backup(store, data_dir.clone());
    // Keep the daemon alive; when a shutdown is signaled, DRAIN the backup
    // task (bounded) before the daemon returns, aborting only the async
    // wrapper on timeout — a snapshot can never outlive its owning runtime.
    match shutdown_rx {
        Some(rx) => {
            let _ = rx.await;
            drain_startup_backup(backup_task).await;
        }
        None => std::future::pending::<()>().await,
    }
    Ok(())
}

async fn run(prompt: String, provider: &str, model: &str, workspace: PathBuf, data_dir: PathBuf) {
    match build_daemon(&data_dir, None) {
        Ok(graph) => {
            let (session, agent) = (graph.session, graph.agent);
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
    // The ACP surface is stdio-only: no SSE subscribers exist, so there is
    // no chunk sink (None = the runtime skips live-chunk overhead entirely).
    // The daemon graph is built whole by [`build_daemon`] (the construction
    // region; steps 1-16) — ACP constructs no supervisor or executor of its
    // own and takes its session/agent references from the graph.
    let graph = match build_daemon(&data_dir, Some(config)) {
        Ok(graph) => graph,
        Err(e) => {
            eprintln!("daemon build failed: {e}");
            std::process::exit(1);
        }
    };
    let (session, agent) = (graph.session.clone(), graph.agent.clone());
    // Crash recovery runs before the first request (spec §7), like serve.
    if let Err(e) = agent.recover() {
        tracing::error!("recovery failed: {e}");
    }
    // The ACP prompt surface enters the SAME product execution authority the
    // daemon server uses (the graph's ONE TaskExecutor over the ONE session
    // store) — ACP translates its wire prompts, it never drives the agent.
    let prompts =
        faktor_server::native::PromptExecutionService::new(graph.tasks.clone(), session.clone());
    let backend = DaemonAcpBackend::new(session, agent, prompts);
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
    /// The ONE product execution entry every ACP `session/prompt` goes
    /// through: an ordinary prompt becomes an in-session run through the
    /// daemon's TaskExecutor (default shadow mutation), identical to the
    /// Native and SDK prompt surfaces.
    prompts: Arc<faktor_server::native::PromptExecutionService>,
}

impl DaemonAcpBackend {
    fn new(
        session: Arc<SessionManager>,
        agent: Arc<AgentRuntime>,
        prompts: Arc<faktor_server::native::PromptExecutionService>,
    ) -> Self {
        Self {
            session,
            agent,
            prompts,
        }
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
        // The ONE execution entry: the prompt becomes an in-session run
        // through the daemon's TaskExecutor (default shadow mutation). The
        // AcpBackend seam is synchronous (one serialized ACP request at a
        // time); the service call is async, so bridge sync → async on the
        // serve task via block_in_place (multi-threaded daemon runtime).
        let service = self.prompts.clone();
        let request = faktor_server::native::PromptRequest {
            prompt: text.to_string(),
            ..Default::default()
        };
        let receipt = tokio::task::block_in_place(move || {
            tokio::runtime::Handle::current().block_on(service.prompt(sid, request))
        })
        .map_err(|e| e.to_string())?;
        let session = self.session.clone();
        if receipt.queued {
            // The prompt durably queued behind another actor's active turn;
            // the executor's own runner delivers it. Report the queued
            // acceptance, mirroring the previous backend behavior.
            let state = session
                .get_session(sid)
                .map_err(|e| e.message)?
                .ok_or_else(|| format!("session {sid}"))?
                .state()
                .map_err(|e| e.message)?;
            return Ok(json!({
                "status": "queued",
                "finalState": state,
            }));
        }
        // Accepted: wait for the turn machine to leave the mid-turn states
        // (the same durable wait the wire prompt path performs), then report
        // the machine's final state.
        let state = tokio::task::block_in_place(move || {
            tokio::runtime::Handle::current().block_on(async move {
                loop {
                    let handle = match session.get_session(sid) {
                        Ok(Some(h)) => h,
                        Ok(None) => return Err(format!("session {sid}")),
                        Err(e) => return Err(e.message),
                    };
                    match handle.state() {
                        Ok(s) if !faktor_server::native::turn_machine_busy(s) => return Ok(s),
                        Ok(_) => {}
                        Err(e) => return Err(e.message),
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(25)).await;
                }
            })
        })?;
        Ok(json!({
            "status": "completed",
            "finalState": state,
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

/// `faktor-cli doctor [--deep]`: plain mode opens with the bounded quick
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
/// also covers checkpoint after-blob refs), the journal projection checks and
/// the P0-97 audit invariants — dangling cost reservations, verification-
/// record/task consistency, active-turn recoverable owners, orphan children
/// (durable orchestrator rows) and the daemon-level process-ownership
/// report. None of these checks write to the store or the CAS: corruption is
/// listed and left alone (a second run must find the same issues).
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
    //    Each referenced blob is STRICT-verified (decode + re-hash, P0-52):
    //    a corrupt-but-present blob is reported as corrupt, only a true
    //    absence is "missing". Path existence alone is never validity.
    match store.cas_hash_references() {
        Ok(refs) => {
            let mut dangling = Vec::new();
            let mut present = 0usize;
            for r in &refs {
                match faktor_core::hash::FileHash::from_hex(&r.hash) {
                    None => dangling.push(format!(
                        "{} row {} holds a malformed CAS hash {}",
                        r.source, r.row_id, r.hash
                    )),
                    Some(h) => match cas.verify_now(&h.to_hex()) {
                        Ok(_) => present += 1,
                        Err(faktor_cas::CasError::NotFound(_)) => dangling.push(format!(
                            "{} row {} references missing CAS blob {}",
                            r.source, r.row_id, r.hash
                        )),
                        Err(e) => dangling.push(format!(
                            "{} row {} references CORRUPT CAS blob {} ({e})",
                            r.source, r.row_id, r.hash
                        )),
                    },
                }
            }
            if dangling.is_empty() {
                lines.push(format!("cas references: {} hash(es) all verified", present));
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
    // 6. Durable cost-ledger invariant (P0-97): a reservation row is the
    //    ledger's handle onto its task envelope, so a row whose task row is
    //    gone can never settle or refund and its prediction silently leaves
    //    the cap math. Per-status counts are reported either way.
    match store.cost_reservation_invariants() {
        Ok(s) => {
            lines.push(format!(
                "cost reservations: {} (open {}, settled {}, refunded {}, uncertain {})",
                s.total, s.open, s.settled, s.refunded, s.uncertain
            ));
            if s.dangling.is_empty() {
                lines.push("dangling cost reservations: none".into());
            } else {
                for d in &s.dangling {
                    lines.push(format!(
                        "dangling cost reservation: reservation {} (session {} task {} op {}, status {}) references a task row that no longer exists (predicted {} micro)",
                        d.reservation_id, d.session_id, d.task_id, d.op_id, d.status, d.predicted_micro
                    ));
                }
                *issues += s.dangling.len();
            }
        }
        Err(e) => add_issue(format!("cost reservation scan failed: {e}"), lines, issues),
    }
    // 7. Verification-record consistency (P0-97, wave-16 invariant): records
    //    must reference existing task rows; a Passed record may certify the
    //    current revision only of a VerifiedComplete task; and a
    //    VerifiedComplete task must carry the Passed record its completion
    //    consumed. MUST fail loudly — completion proof is the row's only
    //    justification.
    match store.verification_record_invariants() {
        Ok(s) => {
            lines.push(format!(
                "verification records: {} record(s), {} completion-relevant task(s), {} VerifiedComplete task(s)",
                s.total_records, s.relevant_tasks, s.completed_tasks
            ));
            if s.issues.is_empty() {
                lines.push("verification consistency: ok".into());
            } else {
                for i in &s.issues {
                    lines.push(format!(
                        "verification inconsistency [{}]: {}",
                        i.kind, i.detail
                    ));
                }
                *issues += s.issues.len();
            }
        }
        Err(e) => add_issue(
            format!("verification consistency scan failed: {e}"),
            lines,
            issues,
        ),
    }
    // 8. Active-turn recoverable owners (P0-97, read-only semantics): a LIVE
    //    daemon legitimately owns active rows in memory, so the only durable
    //    question is whether a crashed daemon's recovery could own them — a
    //    prompt message row, a prompt-queue row, a journal event naming the
    //    turn op or a tool-run row must exist.
    match store.active_turn_ownership_invariants() {
        Ok(s) => {
            lines.push(format!(
                "active turns with recoverable owners: {} of {} (a live daemon owns the rest in memory)",
                s.recoverable, s.active_turns
            ));
            if !s.unrecoverable.is_empty() {
                for u in &s.unrecoverable {
                    lines.push(format!(
                        "active turn without recoverable owner: {}",
                        u.detail
                    ));
                }
                *issues += s.unrecoverable.len();
            }
        }
        Err(e) => add_issue(
            format!("active-turn ownership scan failed: {e}"),
            lines,
            issues,
        ),
    }
    // 9. Orphan children (P0-97/100, read-only): every durable child
    //    identity row must name an existing parent session, every executor
    //    registry row must name an existing child session under its own
    //    child_id, and a NON-TERMINAL child's worktree row + directory must
    //    still exist.
    let orphans =
        faktor_orchestrator::runtime::OrchestratorRuntime::orphan_children_scan(session.clone());
    lines.push(format!(
        "orphan children: {} child identity row(s), {} registry row(s) scanned",
        orphans.identity_rows, orphans.registry_rows
    ));
    if orphans.issues.is_empty() {
        lines.push("orphan children: none".into());
    } else {
        for i in &orphans.issues {
            lines.push(format!("orphan child: {i}"));
        }
        *issues += orphans.issues.len();
    }
    // 10. Orphan processes (P0-97): the session layer keeps NO durable
    //     process-ownership rows — ownership is the in-memory per-session
    //     registry, which dies with its owner session and is reported and
    //     cleared by crash recovery, and the daemon-level supervisor, whose
    //     Drop kills every still-live child when its last reference goes.
    //     Doctor is a separate process, so it reports the daemon-level live
    //     map of THIS process (informational: the zero-orphan guarantee is
    //     an in-process lifetime contract, not durable rows another process
    //     could audit).
    {
        let shared = ProcessSupervisor::shared();
        let alive = shared.alive();
        let session_owned = alive
            .iter()
            .filter(|c| matches!(c.owner, ProcessOwner::Session(_)))
            .count();
        lines.push(format!(
            "process ownership: 0 durable session-owned process row(s) (ownership is in-memory only); daemon-level live children in this process: {} alive ({} registered, {} session-owned)",
            alive.len(),
            shared.registered(),
            session_owned
        ));
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
    use faktor_core::CancellationToken;
    use faktor_core::{Capability, CapabilitySet, OpId};
    use faktor_provider::testing::{sse_body, MockAction, MockServer};
    use faktor_provider::{
        ContentPart, FakeProvider, GenericAgentRequest, ProviderChunk, ProviderError,
        RequestMessage, RequestMeta, Role, ScriptedResponse, ToolSpec,
    };
    use faktor_terminal::{EnvSpec, ProcessOwner, SpawnConfig};
    use futures::StreamExt;
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
            verification: faktor_agent::VerificationService::disabled(),
            hooks: None,
            instructions_resolver: daemon_instructions_resolver(&session),
            // Test graph: the passthrough pin (session-configured
            // provider/model win) + the REAL durable ledger over this
            // session manager (reservations ride the tempdir store).
            routing: faktor_agent::FixedRoutingPolicy::passthrough(),
            budgets: faktor_session::DurableBudgetLedger::new(session.clone()),
            model: "default".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "You are Faktor.".into(),
            clock: Arc::new(SystemClock),
            tool_call_mode: ToolCallMode::Native,
            tool_deadline_ms: 2000,
            retry_policy: faktor_core::retry::RetryPolicy::default(),
            semantic: faktor_agent::fallback_semantic_registry(),
            context_prior: None,
            efficiency: Default::default(),
        };
        AgentRuntime::new(deps).unwrap()
    }

    /// The daemon's ONE execution facade over a test graph: the same
    /// construction the production entries use (one OrchestratorRuntime +
    /// TaskExecutor over the same session/agent, wrapped once in the
    /// PromptExecutionService every adapter calls).
    fn test_prompts(
        session: &Arc<SessionManager>,
        agent: &Arc<AgentRuntime>,
    ) -> Arc<faktor_server::native::PromptExecutionService> {
        let orchestrator =
            faktor_orchestrator::runtime::OrchestratorRuntime::new(session.clone(), agent.clone());
        let tasks = faktor_orchestrator::runtime::task_executor::TaskExecutor::new(
            &orchestrator,
            session.clone(),
            agent.clone(),
            None,
        );
        faktor_server::native::PromptExecutionService::new(tasks, session.clone())
    }

    /// The daemon's real write_file shape for ACP shadow tests: writes
    /// through the session's resolved workspace (the live shadow while a
    /// shadowed drive is running).
    fn acp_write_tool() -> faktor_agent::Tool {
        use faktor_agent::tool::RecoveryHint;
        use faktor_agent::{ToolOutcome, ToolRunCtx};
        use faktor_core::resource::ResourceClass;
        faktor_agent::Tool {
            name: "write_file".into(),
            description: "writes a real file".into(),
            input_schema: serde_json::json!({"type": "object"}),
            resource_class: ResourceClass::DiskWrite,
            capability: None,
            recovery_hint: RecoveryHint::WorkspaceWrite,
            path_args: vec!["path".into()],
            execute: Arc::new(move |ctx: ToolRunCtx, args| {
                Box::pin(async move {
                    let ws = ctx
                        .workspace
                        .ok_or_else(|| faktor_core::error::Error::internal("no workspace wired"))?;
                    let path = args.get("path").and_then(|p| p.as_str()).unwrap_or("");
                    let content = args
                        .get("content")
                        .and_then(|c| c.as_str())
                        .unwrap_or_default();
                    ws.write_atomic(std::path::Path::new(path), content.as_bytes())
                        .map_err(|e| {
                            faktor_core::error::Error::internal(format!("write {path}: {e}"))
                        })?;
                    Ok(ToolOutcome {
                        text: format!("wrote {path}"),
                        exit_code: Some(0),
                        ..Default::default()
                    })
                })
            }),
        }
    }

    /// An ACP host over the FULL execution wiring: one session/agent, one
    /// shadow-carrying TaskExecutor, one PromptExecutionService the backend
    /// was constructed with. `owner` is the real checkout the ACP session
    /// points at.
    struct AcpShadowRig {
        dir: tempfile::TempDir,
        session: Arc<SessionManager>,
        #[allow(dead_code)]
        agent: Arc<AgentRuntime>,
        service: Arc<faktor_server::native::PromptExecutionService>,
        backend: DaemonAcpBackend,
    }

    fn acp_shadow_rig(scripts: Vec<ScriptedResponse>) -> AcpShadowRig {
        let dir = tempfile::tempdir().unwrap();
        let session =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let mut registry = ProviderRegistry::new();
        registry
            .try_register(Arc::new(FakeProvider::with_script(
                "fake",
                ModelCapabilities {
                    tools: true,
                    ..Default::default()
                },
                scripts,
            )))
            .unwrap();
        let mut tools = ToolRegistry::new();
        tools.register(acp_write_tool());
        let deps = AgentDeps {
            session: session.clone(),
            providers: Arc::new(registry),
            chunk_sink: None,
            permission_requester: Arc::new(AlwaysAllow),
            evidence: Arc::new(faktor_agent::NoEvidence),
            tools: Arc::new(tools),
            cas: Some(session.cas()),
            workspaces: faktor_fs::WorkspaceFileService::new(),
            edit: None,
            snapshots: None,
            sandbox: None,
            supervisor: None,
            verification: faktor_agent::VerificationService::disabled(),
            hooks: None,
            instructions_resolver: daemon_instructions_resolver(&session),
            routing: faktor_agent::FixedRoutingPolicy::passthrough(),
            budgets: faktor_session::DurableBudgetLedger::new(session.clone()),
            model: "default".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "You are Faktor.".into(),
            clock: Arc::new(SystemClock),
            tool_call_mode: ToolCallMode::Native,
            tool_deadline_ms: 60_000,
            retry_policy: faktor_core::retry::RetryPolicy::default(),
            semantic: faktor_agent::fallback_semantic_registry(),
            context_prior: None,
            efficiency: Default::default(),
        };
        let agent = AgentRuntime::new(deps).unwrap();
        let orchestrator =
            faktor_orchestrator::runtime::OrchestratorRuntime::new(session.clone(), agent.clone());
        let shadows = faktor_orchestrator::runtime::shadow::ShadowRoots::new(
            session.clone(),
            dir.path().join("shadows"),
        );
        let tasks = faktor_orchestrator::runtime::task_executor::TaskExecutor::new_with_mode(
            &orchestrator,
            session.clone(),
            agent.clone(),
            Some(shadows),
            faktor_orchestrator::runtime::task_executor::MutationMode::Shadow,
        );
        let service = faktor_server::native::PromptExecutionService::new(tasks, session.clone());
        let backend = DaemonAcpBackend::new(session.clone(), agent.clone(), service.clone());
        AcpShadowRig {
            dir,
            session,
            agent,
            service,
            backend,
        }
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
        registry
            .try_register(Arc::new(FakeProvider::with_script(
                "fake",
                ModelCapabilities {
                    tools: true,
                    ..Default::default()
                },
                script,
            )))
            .unwrap();
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

    // ---- durable learning prior (audits 65-69/82) ----

    /// A REAL session manager with one workspace/session but an EMPTY
    /// durable learning corpus (no learning rows yet).
    fn empty_learning_session() -> (
        tempfile::TempDir,
        Arc<SessionManager>,
        faktor_core::id::SessionId,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let manager =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let ws = manager.create_workspace("/w").unwrap();
        let session = manager
            .create_session(ws, "learning", "fake", "m")
            .unwrap()
            .id();
        (dir, manager, session)
    }

    /// Mine ONE verified recovery into the session's durable ledger corpus
    /// through the REAL `SessionLearningStore` adapter — the same durable
    /// rows the runtime's mining hook appends. Returns the stored learning.
    fn mine_durable_corpus(
        manager: &Arc<SessionManager>,
        session: faktor_core::id::SessionId,
    ) -> faktor_learning::ProjectLearning {
        use faktor_core::id::VerificationRecordId;
        use faktor_learning::{
            ActionDescriptor, ActionFingerprint, EnvironmentFingerprint, EpisodeId,
            FailureDescriptor, FailureEpisode, FailureFingerprint, LearningService,
            LearningStore as _, ProjectScope, SessionLearningStore, TaskClass,
            DEFAULT_MEMORY_CAPACITY,
        };

        let handle = manager.get_session(session).unwrap().unwrap();
        let scope = ProjectScope::new(handle.row().unwrap().workspace_id, "w").unwrap();
        let episode = FailureEpisode::new(
            EpisodeId::new(1),
            TaskClass::new("bugfix").unwrap(),
            EnvironmentFingerprint::new(scope, "linux", "rustc", None).unwrap(),
            ActionFingerprint::of(
                &ActionDescriptor::new("edit", "src/lib.rs", Some("parse"), "attempt").unwrap(),
            ),
            FailureFingerprint::of(
                &FailureDescriptor::new("test_failure", None, "assertion failed").unwrap(),
            ),
        )
        .with_recovery_actions(vec![ActionFingerprint::of(
            &ActionDescriptor::new("edit", "src/lib.rs", Some("parse"), "guard").unwrap(),
        )])
        .unwrap()
        .verified(VerificationRecordId::new(11));
        let store = SessionLearningStore::open(handle, DEFAULT_MEMORY_CAPACITY).unwrap();
        let mut service = LearningService::new(store);
        service.mine_and_store(&[episode]).unwrap();
        service.store().all()[0].clone()
    }

    fn keyed_candidate(id: &str, keys: Vec<String>) -> faktor_context::ContextCandidate {
        faktor_context::ContextCandidate {
            id: id.into(),
            omission_keys: keys,
            ..Default::default()
        }
    }

    /// Audit 68 production wiring: the flag alone decides whether the
    /// durable session-manager-backed adapter is installed; with no learning
    /// rows the corpus index is empty and every candidate is neutral (byte
    /// parity with the flag-off path), keyed candidates included.
    #[test]
    fn failure_learning_prior_installs_only_on_flag_and_is_neutral_when_empty() {
        let (_dir, manager, _session) = empty_learning_session();
        assert!(
            daemon_context_prior(false, &manager).is_none(),
            "flag off => no prior handle"
        );
        let prior = daemon_context_prior(true, &manager).expect("flag on => learning adapter");
        for id in ["msg:0", "src/lib.rs", ""] {
            assert_eq!(
                prior.omission_risk(&keyed_candidate(id, Vec::new())),
                1.0,
                "empty corpus => neutral risk for {id:?}"
            );
        }
        assert_eq!(
            prior.omission_risk(&keyed_candidate("msg:0", vec!["0".repeat(64)])),
            1.0,
            "empty corpus => a digest-shaped omission key is neutral too"
        );
    }

    /// The durable adapter over a REAL mined corpus: the candidate's
    /// `omission_keys` (never its render id) resolve to the learning's
    /// confidence-scaled risk (`1 + ppm/1e6`); empty/hostile/oversized keys
    /// can neither panic nor leave `[1, 2]`; and a fresh manager over the
    /// same data dir rebuilds the identical durable index (reopen-safe).
    #[test]
    fn durable_prior_resolves_omission_keys_ignores_render_ids_and_survives_reopen() {
        let (dir, manager, session) = empty_learning_session();
        let stored = mine_durable_corpus(&manager, session);
        assert_eq!(stored.confidence_ppm, 400_000, "one verified sample");
        let pattern = stored.pattern_digest().to_hex();
        let failure = stored.pattern.failure.digest().to_hex();

        let prior = daemon_context_prior(true, &manager).expect("flag on => durable prior");
        assert_eq!(
            prior.omission_risk(&keyed_candidate("msg:0", vec![failure.clone()])),
            1.4
        );
        assert_eq!(
            prior.omission_risk(&keyed_candidate("msg:0", vec![pattern.clone()])),
            1.4
        );
        assert_eq!(
            prior.omission_risk(&keyed_candidate("msg:0", vec![failure.to_uppercase()],)),
            1.4,
            "hex parsing is case-insensitive"
        );
        assert_eq!(
            prior.omission_risk(&keyed_candidate("msg:0", Vec::new())),
            1.0
        );
        // A render id that HAPPENS to be the digest is ignored: only
        // omission_keys are lookup keys.
        let id_only = faktor_context::ContextCandidate {
            id: failure.clone(),
            ..Default::default()
        };
        assert_eq!(
            prior.omission_risk(&id_only),
            1.0,
            "lookup must key omission_keys, never candidate.id"
        );
        // The max over several keys wins; hostile keys stay neutral.
        assert_eq!(
            prior.omission_risk(&keyed_candidate(
                "msg:0",
                vec!["0".repeat(64), failure.clone(), "src/lib.rs".into()],
            )),
            1.4
        );
        for key in [
            String::new(),
            "src/lib.rs".to_string(),
            "\0\u{1f600}".to_string(),
            "0".repeat(4096),
        ] {
            let risk = prior.omission_risk(&keyed_candidate("msg:0", vec![key.clone()]));
            assert!(
                risk.is_finite() && (1.0..=2.0).contains(&risk),
                "hostile key {key:?} produced {risk}"
            );
            assert_eq!(risk, 1.0, "{key:?}");
        }

        // Reopen-safety: a fresh manager over the same data dir rebuilds the
        // same durable index from the same ledger rows.
        drop(prior);
        drop(manager);
        let reopened =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let prior = daemon_context_prior(true, &reopened).expect("flag on");
        assert_eq!(
            prior.omission_risk(&keyed_candidate("msg:0", vec![failure])),
            1.4
        );
    }

    /// The loop closes in-process (no restart): a prior built while the
    /// corpus was empty re-reads the durable ledger when the stamp advances
    /// and protects the freshly mined learning on the next lookup.
    #[test]
    fn durable_prior_refreshes_without_restart_when_the_corpus_is_mined() {
        let (_dir, manager, session) = empty_learning_session();
        let prior = daemon_context_prior(true, &manager).expect("flag on");
        assert_eq!(
            prior.omission_risk(&keyed_candidate("msg:0", vec!["0".repeat(64)])),
            1.0
        );
        let stored = mine_durable_corpus(&manager, session);
        let failure = stored.pattern.failure.digest().to_hex();
        assert_eq!(
            prior.omission_risk(&keyed_candidate("msg:0", vec![failure])),
            1.4,
            "the ledger stamp advanced; the index must re-read without a restart"
        );
    }

    /// Hostile/corrupt ledger rows are LOUD but non-fatal: the prior stays
    /// installed, keeps serving, and leaves the affected corpus neutral —
    /// including on a stamp-advancing refresh.
    #[test]
    fn corrupt_learning_row_is_loud_but_leaves_the_prior_neutral_and_total() {
        let (_dir, manager, session) = empty_learning_session();
        let handle = manager.get_session(session).unwrap().unwrap();
        handle
            .ledger_learning_record(faktor_session::LEARNING_RECORD_LEARNING, "not json")
            .unwrap();
        let prior =
            daemon_context_prior(true, &manager).expect("flag on => prior despite corrupt row");
        assert_eq!(
            prior.omission_risk(&keyed_candidate("msg:0", vec!["a".repeat(64)])),
            1.0
        );
        // Another corrupt row advances the stamp: the refresh is still
        // non-fatal and still neutral.
        handle
            .ledger_learning_record(faktor_session::LEARNING_RECORD_LEARNING, "still not json")
            .unwrap();
        assert_eq!(
            prior.omission_risk(&keyed_candidate("msg:0", vec!["a".repeat(64)])),
            1.0
        );
    }

    /// The full production selection loop (audits 65-69/82): a durable mined
    /// corpus -> the wire planner's `learning:<digest>` evidence exposes the
    /// digest through `omission_keys` -> the CLI's durable prior protects it
    /// enough to flip the single evidence slot, and the exact same corpus
    /// survives a manager reopen.
    #[test]
    fn durable_prior_flips_planner_selection_and_survives_reopen() {
        use faktor_context::assembler::Evidence;
        use faktor_context::budget::ContextBudget;
        use faktor_context::information::FailurePrior;
        use faktor_context::ledger::TaskLedger;
        use faktor_context::wire_plan::WirePlan;
        use faktor_context::TokenCache;

        fn plan(
            evidence: &[Evidence],
            budget: &ContextBudget,
            ledger: &TaskLedger,
            cache: &TokenCache,
            prior: Option<&(dyn FailurePrior + Send + Sync)>,
        ) -> WirePlan {
            faktor_agent::wire_plan::plan_wire_turn_with_prior(
                "You are a test agent.\n",
                "",
                &[],
                "",
                ledger,
                "",
                &[],
                evidence,
                budget,
                "gpt-5",
                cache,
                prior,
            )
            .unwrap()
        }

        let (dir, manager, session) = empty_learning_session();
        let stored = mine_durable_corpus(&manager, session);
        let failure = stored.pattern.failure.digest().to_hex();

        // Two equal-priced evidence blocks compete for exactly one slot;
        // only the learning-sourced one carries an omission key.
        let evidence = vec![
            Evidence {
                path: format!("learning:{failure}"),
                snippet: "x".repeat(96),
                score: 0.5,
            },
            Evidence {
                path: "src/b.rs".into(),
                snippet: "x".repeat(400),
                score: 0.6,
            },
        ];
        let budget = ContextBudget {
            system: 130,
            tools: 0,
            working: 0,
            retrieved: 0,
            recent: 0,
            output_reserve: 0,
            safety: 0,
        };
        let ledger = TaskLedger::default();
        let cache = TokenCache::new();

        let off = plan(&evidence, &budget, &ledger, &cache, None);
        assert!(
            off.system.contains("### src/b.rs") && !off.system.contains("### learning:"),
            "baseline: the 0.6 block wins the single slot; system={}",
            off.system
        );

        let prior = daemon_context_prior(true, &manager).expect("flag on");
        let on = plan(&evidence, &budget, &ledger, &cache, Some(prior.as_ref()));
        assert!(
            on.system.contains(&format!("### learning:{failure}")),
            "the mined learning's omission risk must protect its candidate"
        );
        assert!(!on.system.contains("### src/b.rs"));
        assert_eq!(
            off.cacheable_prefix().unwrap(),
            on.cacheable_prefix().unwrap(),
            "the prior may only change volatile selection"
        );

        // Durable across reopen: the same corpus rebuilds the same prior.
        drop(prior);
        drop(manager);
        let reopened =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let prior = daemon_context_prior(true, &reopened).expect("flag on");
        let again = plan(&evidence, &budget, &ledger, &cache, Some(prior.as_ref()));
        assert!(again.system.contains(&format!("### learning:{failure}")));
    }

    /// The parsed `[efficiency]` switches thread 1:1 onto the agent-side
    /// flags (`failure_learning` included); the additive default is all-off.
    #[test]
    fn efficiency_flags_mirror_every_parsed_switch() {
        let all = config::EfficiencyCfg {
            failure_learning: true,
            ccr: true,
            typed_handoff: true,
            semantic_context: true,
            rework_routing: true,
        };
        assert_eq!(
            efficiency_flags(&all),
            faktor_agent::EfficiencyFlags {
                failure_learning: true,
                ccr: true,
                typed_handoff: true,
                semantic_context: true,
                rework_routing: true,
            }
        );
        assert!(efficiency_flags(&all).failure_learning);
        assert_eq!(
            efficiency_flags(&config::EfficiencyCfg::default()),
            faktor_agent::EfficiencyFlags::default(),
            "additive default: every flag off"
        );
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
    fn semantic_section_is_additive_strict_and_empty_by_default() {
        // The `[semantic]` section (audits 48-54/58/79) is additive: absent
        // keeps the fallback-only registry; a present section parses under
        // the SAME strict document rules as every other config key.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("serve.json");
        let (_cfg, semantic) = serve_config_and_semantic(None).unwrap();
        assert_eq!(semantic, graph::SemanticCfg::default());
        std::fs::write(&path, r#"{"config_version": 1, "model": "m"}"#).unwrap();
        let (cfg, semantic) = serve_config_and_semantic(Some(path.clone())).unwrap();
        assert_eq!(cfg.model, "m");
        assert_eq!(semantic, graph::SemanticCfg::default(), "empty default");
        std::fs::write(
            &path,
            r#"{"config_version": 1, "model": "m", "semantic": {
                "max_payload_bytes": 2048, "max_entity_refs": 64
            }}"#,
        )
        .unwrap();
        let (cfg, semantic) = serve_config_and_semantic(Some(path.clone())).unwrap();
        assert_eq!(cfg.model, "m");
        assert_eq!(semantic.max_payload_bytes, Some(2048));
        assert_eq!(semantic.max_entity_refs, Some(64));
        let supervisor =
            ProcessSupervisor::new(Arc::new(faktor_cas::Cas::new(dir.path().join("cas"))));
        let transport: Arc<dyn HttpTransport> = Arc::new(PolicyCheckedHttpTransport::permissive());
        let registry = graph::semantic_registry(&semantic, &supervisor, &transport).unwrap();
        assert!(
            registry.providers().is_empty(),
            "caps-only section registers no provider"
        );
        // A configured external provider builds through the daemon's own
        // authorities (registration itself spawns nothing).
        std::fs::write(
            &path,
            r#"{"config_version": 1, "model": "m", "semantic": {"providers": [
                {"kind": "process", "id": "local-proc", "command": "/bin/true", "timeout_ms": 1000},
                {"kind": "http", "id": "remote", "endpoint": "http://provider.example/semantic", "timeout_ms": 1000}
            ]}}"#,
        )
        .unwrap();
        let (_cfg, semantic) = serve_config_and_semantic(Some(path.clone())).unwrap();
        let registry = graph::semantic_registry(&semantic, &supervisor, &transport).unwrap();
        let ids: Vec<String> = registry
            .providers()
            .iter()
            .map(|p| p.id().as_str().to_string())
            .collect();
        assert_eq!(ids, vec!["local-proc".to_string(), "remote".to_string()]);
        // Duplicate provider ids are refused at strict load.
        std::fs::write(
            &path,
            r#"{"model": "m", "semantic": {"providers": [
                {"kind": "process", "id": "dup", "command": "/bin/true", "timeout_ms": 1000},
                {"kind": "http", "id": "dup", "endpoint": "http://x", "timeout_ms": 1000}
            ]}}"#,
        )
        .unwrap();
        let e = serve_config_and_semantic(Some(path.clone())).expect_err("duplicate ids");
        assert!(e.contains("[semantic]") && e.contains("dup"), "{e}");
        // Hostile provider values (zero timeout, non-http endpoint) refuse
        // startup here, before any graph authority exists.
        for bad in [
            r#"{"model": "m", "semantic": {"providers": [
                {"kind": "process", "id": "p", "command": "/bin/true", "timeout_ms": 0}
            ]}}"#,
            r#"{"model": "m", "semantic": {"providers": [
                {"kind": "http", "id": "h", "endpoint": "ftp://x", "timeout_ms": 1000}
            ]}}"#,
        ] {
            std::fs::write(&path, bad).unwrap();
            let e = serve_config_and_semantic(Some(path.clone())).expect_err("hostile provider");
            assert!(e.contains("[semantic]"), "{e}");
        }
        // Strict inside the section: a typo'd key fails startup.
        std::fs::write(&path, r#"{"model": "m", "semantic": {"surprise": true}}"#).unwrap();
        let e = serve_config_and_semantic(Some(path.clone())).expect_err("strict section");
        assert!(e.contains("[semantic]"), "{e}");
        // Unknown fields OUTSIDE the section still fail exactly as before.
        std::fs::write(&path, r#"{"model": "m", "surprise": 1}"#).unwrap();
        assert!(serve_config_and_semantic(Some(path)).is_err());
    }

    #[test]
    fn semantic_registry_arc_is_the_one_authority_for_agent_and_server() {
        // Audit 83 lock: the graph builds the semantic registry EXACTLY
        // once; the SAME Arc flows to AgentDeps and to ServerDeps (the
        // exact serve_impl assembly), so the native introspection surface
        // can never report a parallel registry.
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let session = SessionManager::open(data.join("store"), data.join("cas"), true).unwrap();
        let supervisor = ProcessSupervisor::new(session.cas());
        let semantic = graph::SemanticCfg {
            providers: vec![faktor_semantic::SemanticProviderConfig::Process {
                id: faktor_semantic::SemanticProviderId::parse("graph-proc").unwrap(),
                command: "/bin/true".to_string(),
                args: vec![],
                timeout_ms: 1_000,
            }],
            ..Default::default()
        };
        let graph = build_daemon_core(
            &data,
            session,
            supervisor,
            config::Config::default(),
            vec![],
            None,
            semantic,
        )
        .expect("daemon core builds with a configured semantic provider");
        assert_eq!(graph.semantic.providers().len(), 1);
        assert!(
            Arc::ptr_eq(graph.agent.semantic_registry(), &graph.semantic),
            "the agent must hold the graph's semantic Arc"
        );
        // The ServerDeps assembly mirrors serve_impl exactly.
        let mut deps = ServerDeps::new_with(
            graph.session.clone(),
            graph.agent.clone(),
            graph.permissions.clone(),
            graph.orchestrator.clone(),
            graph.tasks.clone(),
            graph.budgets.clone(),
        );
        deps = deps.with_semantic_registry(graph.semantic.clone());
        let served = deps.semantic.as_ref().expect("semantic wired");
        assert!(
            Arc::ptr_eq(served, graph.agent.semantic_registry()),
            "the native surface (deps.semantic) and the agent must share ONE Arc"
        );
        assert_eq!(served.providers()[0].id().as_str(), "graph-proc");
    }

    #[test]
    fn daemon_instructions_resolver_uses_durable_session_roots_only() {
        // P0-32: the daemon resolver must never invent a root — loading
        // repository rules at the wrong root would silently misapply them.
        // Resolution goes through the SessionManager workspace table only:
        // an unknown/rootless workspace is an Empty set (documented), and a
        // durable workspace root serves its AGENTS.md with live-epoch
        // semantics (rewrite -> new epoch/content; pinned old epoch serves
        // the cached old tree).
        let dir = tempfile::tempdir().unwrap();
        let session =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let resolver = daemon_instructions_resolver(&session);
        // Unknown workspace id: Empty, never an error, never the CWD.
        assert!(resolver.resolve(999, None).unwrap().is_empty());
        // A durable workspace root serves the tree at that root.
        let repo = dir.path().join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        std::fs::write(repo.join("AGENTS.md"), "always: durable daemon rules\n").unwrap();
        let ws = session.create_workspace(repo.to_str().unwrap()).unwrap();
        let loaded = resolver.resolve(ws.raw(), None).unwrap();
        assert!(loaded
            .active_for("anything", &[])
            .iter()
            .any(|i| i.content.contains("durable daemon rules")));
        let e1 = loaded.epoch().unwrap();
        // A rewrite moves the epoch and the served content; the pinned old
        // epoch still sees the old content through the cache.
        std::fs::write(repo.join("AGENTS.md"), "always: rewritten daemon rules\n").unwrap();
        let v2 = resolver.resolve(ws.raw(), None).unwrap();
        assert_ne!(v2.epoch().unwrap(), e1);
        assert!(v2
            .active_for("x", &[])
            .iter()
            .any(|i| i.content.contains("rewritten daemon rules")));
        let pinned = resolver.resolve(ws.raw(), Some(e1)).unwrap();
        assert!(
            pinned
                .active_for("x", &[])
                .iter()
                .any(|i| i.content.contains("durable daemon rules")),
            "a pinned old epoch must still serve the old tree"
        );
    }
    #[test]
    fn daemon_instructions_resolver_reads_the_live_shadow_of_a_shadowed_workspace() {
        // P0-48: while exactly ONE session of the workspace carries a live
        // shadow row, the resolver's rules come from the SHADOW root (the
        // drive reads the instruction environment of the world it mutates);
        // with no live shadow (or after retirement) the stored workspace
        // root serves byte-identically; ambiguity is a loud degrade.
        let dir = tempfile::tempdir().unwrap();
        let session =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let resolver = daemon_instructions_resolver(&session);
        let repo = dir.path().join("repo");
        let shadow = dir.path().join("shadows").join("1").join("sh-x");
        std::fs::create_dir_all(&repo).unwrap();
        std::fs::create_dir_all(&shadow).unwrap();
        std::fs::write(repo.join("AGENTS.md"), "always: user-checkout rules\n").unwrap();
        std::fs::write(shadow.join("AGENTS.md"), "always: shadow-world rules\n").unwrap();
        let ws = session.create_workspace(repo.to_str().unwrap()).unwrap();
        let handle = session.create_session(ws, "t", "fake", "m").unwrap();
        let loaded = resolver.resolve(ws.raw(), None).unwrap();
        assert!(loaded
            .active_for("anything", &[])
            .iter()
            .any(|i| i.content.contains("user-checkout rules")));
        // Begin a durable shadow (the exact row the ShadowRoots service
        // writes): the workspace's rules now resolve from the shadow.
        let row = faktor_session::ShadowRow {
            session_id: handle.id().raw(),
            shadow_id: "sh-x".into(),
            base_root: repo.to_str().unwrap().into(),
            root: shadow.to_str().unwrap().into(),
            state: faktor_session::ShadowRowState::Active,
            base_entries: 1,
            base_bytes: 1,
            created_ms: session.now_ms(),
        };
        session.put_shadow_row(handle.id(), &row).unwrap();
        let shadowed = resolver.resolve(ws.raw(), None).unwrap();
        assert!(
            shadowed
                .active_for("anything", &[])
                .iter()
                .any(|i| i.content.contains("shadow-world rules")),
            "the live shadow re-points instruction loading"
        );
        // Retire the shadow: the stored root is authoritative again.
        let mut retired = row;
        retired.state = faktor_session::ShadowRowState::Integrated;
        session.put_shadow_row(handle.id(), &retired).unwrap();
        let back = resolver.resolve(ws.raw(), None).unwrap();
        assert!(back
            .active_for("anything", &[])
            .iter()
            .any(|i| i.content.contains("user-checkout rules")));
        // Two live shadows on one workspace: ambiguous — loud degrade to the
        // stored root (never a guessed root).
        let other = session.create_session(ws, "other", "fake", "m").unwrap();
        let mut other_row = faktor_session::ShadowRow {
            session_id: other.id().raw(),
            shadow_id: "sh-y".into(),
            base_root: repo.to_str().unwrap().into(),
            root: shadow.to_str().unwrap().into(),
            state: faktor_session::ShadowRowState::Active,
            base_entries: 1,
            base_bytes: 1,
            created_ms: session.now_ms(),
        };
        session.put_shadow_row(other.id(), &other_row).unwrap();
        let mut revived = retired;
        revived.state = faktor_session::ShadowRowState::IntegrationBlocked;
        session.put_shadow_row(handle.id(), &revived).unwrap();
        other_row.state = faktor_session::ShadowRowState::Active;
        let loaded = resolver.resolve(ws.raw(), None).unwrap();
        assert!(
            loaded
                .active_for("anything", &[])
                .iter()
                .any(|i| i.content.contains("user-checkout rules")),
            "ambiguous shadows never hijack the stored root"
        );
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
        let prompts = test_prompts(&session, &agent);
        let backend = DaemonAcpBackend::new(session, agent, prompts);
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
        let prompts = test_prompts(&session, &agent);
        let backend = DaemonAcpBackend::new(session.clone(), agent, prompts);

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
        let backend3 = DaemonAcpBackend::new(
            session3.clone(),
            agent3.clone(),
            test_prompts(&session3, &agent3),
        );
        let err = backend3.create_session(&json!({})).unwrap_err();
        assert!(err.contains("no providers"), "{err}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn prompt_runs_a_real_turn_and_reports_the_final_state() {
        let (_dir, session, agent) = acp_test_daemon(vec![
            ScriptedResponse::Text("pong".into()),
            ScriptedResponse::End,
        ]);
        let prompts = test_prompts(&session, &agent);
        let backend = DaemonAcpBackend::new(session, agent, prompts);
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
        let prompts = test_prompts(&session, &agent);
        let backend = DaemonAcpBackend::new(session, agent, prompts);
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
        let prompts = test_prompts(&session, &agent);
        let backend = DaemonAcpBackend::new(session.clone(), agent.clone(), prompts);

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
        let prompts = test_prompts(&session, &agent);
        let backend = DaemonAcpBackend::new(session.clone(), agent, prompts);
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn acp_session_prompt_uses_the_shadow_and_keeps_the_owner_untouched() {
        // Ordinary chat through ACP `session/prompt` goes through the SAME
        // PromptExecutionService as Native and SDK compat: its write lands
        // in the daemon-owned shadow of the session workspace; the owner
        // checkout stays byte-untouched until a verified integration.
        const OWNER: &str = "pub fn value() -> u64 {\n    let base_amount: u64 = 40;\n    let increment: u64 = 1;\n    base_amount.saturating_add(increment)\n}\n";
        const IMPL: &str = "pub fn value() -> u64 {\n    let base_amount: u64 = 10;\n    let increment: u64 = 32;\n    base_amount.saturating_add(increment)\n}\n";
        let rig = acp_shadow_rig(vec![
            ScriptedResponse::ToolCall {
                id: "c1".into(),
                name: "write_file".into(),
                input: json!({"path": "src/lib.rs", "content": IMPL}),
            },
            ScriptedResponse::Text("done".into()),
            ScriptedResponse::End,
        ]);
        let owner = rig.dir.path().join("owner");
        std::fs::create_dir_all(owner.join("src")).unwrap();
        std::fs::write(owner.join("src/lib.rs"), OWNER).unwrap();
        let sid = rig
            .backend
            .create_session(&json!({"workspace": owner.to_str().unwrap()}))
            .unwrap();

        let result = rig.backend.prompt(&sid, "implement the change").unwrap();
        assert_eq!(result["status"], "completed", "{result}");
        let sid = SessionId::new(sid.parse().unwrap());
        let shadow = rig
            .session
            .shadow_row(sid)
            .unwrap()
            .expect("an ordinary mutating ACP prompt must begin a shadow");
        assert_eq!(shadow.state, faktor_session::ShadowRowState::Active);
        assert_eq!(
            std::fs::read(std::path::Path::new(&shadow.root).join("src/lib.rs")).unwrap(),
            IMPL.as_bytes(),
            "the edit landed in the shadow"
        );
        assert_eq!(
            std::fs::read(owner.join("src/lib.rs")).unwrap(),
            OWNER.as_bytes(),
            "the owner checkout is byte-untouched"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sdk_and_acp_prompts_hit_the_same_execution_service() {
        // Identity spy: ACP drives the exact PromptExecutionService instance
        // it was constructed with, and the service method the SDK compat
        // surface calls executes on the SAME task/session authorities.
        use faktor_server::native::{set_prompt_observer, PromptCallKind};
        use std::sync::Mutex;
        let rig = acp_shadow_rig(vec![
            ScriptedResponse::Text("pong".into()),
            ScriptedResponse::End,
        ]);
        let tasks_ptr = Arc::as_ptr(rig.service.tasks()) as usize;
        let sessions_ptr = Arc::as_ptr(rig.service.sessions()) as usize;
        let seen: Arc<Mutex<Vec<(PromptCallKind, usize, usize)>>> = Arc::new(Mutex::new(vec![]));
        let sink = seen.clone();
        set_prompt_observer(Some(Arc::new(move |call| {
            if call.tasks_ptr == tasks_ptr {
                sink.lock()
                    .unwrap()
                    .push((call.kind, call.tasks_ptr, call.sessions_ptr));
            }
        })));
        let owner = rig.dir.path().join("owner");
        std::fs::create_dir_all(&owner).unwrap();
        let sid = rig
            .backend
            .create_session(&json!({"workspace": owner.to_str().unwrap()}))
            .unwrap();
        // ACP session/prompt.
        rig.backend.prompt(&sid, "ping").unwrap();
        // The SDK compat surface's exact service call (the server builds
        // its facade over the same deps and calls `prompt`).
        let request = faktor_server::native::PromptRequest {
            prompt: "sdk ping".into(),
            ..Default::default()
        };
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(
                rig.service
                    .prompt(SessionId::new(sid.parse().unwrap()), request),
            )
        })
        .unwrap();
        set_prompt_observer(None);
        let calls = seen.lock().unwrap().clone();
        assert!(
            calls.len() >= 2,
            "ACP and the SDK service call must both be observed: {calls:?}"
        );
        for (kind, tasks, sessions) in &calls {
            assert_eq!(*tasks, tasks_ptr, "call {kind:?} ran on another executor");
            assert_eq!(
                *sessions, sessions_ptr,
                "call {kind:?} ran on another store"
            );
        }
        // The backend's field is the same Arc the caller constructed.
        assert!(Arc::ptr_eq(&rig.backend.prompts, &rig.service));
    }

    #[test]
    fn acp_production_never_drives_the_agent_directly() {
        // Static scan (work-entry unification): the ACP host's backend may
        // only translate wire prompts into PromptExecutionService calls —
        // no direct AgentRuntime drive entry may remain in its body.
        let src = include_str!("main.rs");
        let lines: Vec<&str> = src.lines().collect();
        let start = lines
            .iter()
            .position(|l| l.contains("impl AcpBackend for DaemonAcpBackend {"))
            .expect("the ACP backend impl exists");
        // The trait impl runs until the first column-0 closing brace after
        // its header (it contains no nested column-0 items).
        let end = lines
            .iter()
            .enumerate()
            .skip(start + 1)
            .find(|(_, l)| l.starts_with('}'))
            .map(|(j, _)| j)
            .expect("the ACP backend impl closes");
        for (i, l) in lines[start..end].iter().enumerate() {
            for token in [
                ".run_session_queue(",
                ".drive_receipt(",
                "agent.submit(",
                ".run_turn(",
            ] {
                assert!(
                    !l.contains(token),
                    "DaemonAcpBackend:{}: direct agent drive {token:?}: {l}",
                    start + i + 1
                );
            }
        }
        assert!(
            lines[start..end]
                .iter()
                .any(|l| l.contains("service.prompt(")),
            "The ACP prompt must be delegated to PromptExecutionService::prompt"
        );
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

    #[test]
    fn backup_finalize_is_one_atomic_adoption_and_never_lists_partial_temp() {
        // A crashed writer's `.db.tmp-*` is invisible to the gate/retention
        // scans; the finalize step publishes it whole (fsync + rename +
        // directory fsync through the shared authority) and consumes it.
        let dir = tempfile::tempdir().unwrap();
        let session =
            SessionManager::open_quick(dir.path().join("store"), dir.path().join("cas")).unwrap();
        let store = session.store();
        // Seed a fresh same-size fake so the gate would otherwise skip.
        let db_len = std::fs::metadata(dir.path().join("store").join("faktor-plus.db"))
            .unwrap()
            .len();
        write_backup(dir.path(), 1, std::time::SystemTime::now(), db_len);
        let backups = dir.path().join("backups");
        // Crash residue: a partially written snapshot under the temp name.
        let tmp = backups.join(format!("faktor-plus-999.db.tmp-{}", std::process::id()));
        std::fs::write(&tmp, b"partial sqlite pages").unwrap();
        assert!(
            !list_backups(dir.path()).iter().any(|p| p == &tmp),
            "an in-progress temp is never a complete backup"
        );
        // The atomic adoption publishes it and consumes the temp.
        let dest = backups.join("faktor-plus-999.db");
        faktor_fs::atomic::atomic_adopt(&tmp, &dest).unwrap();
        assert!(!tmp.exists(), "the adopted temp is renamed, not copied");
        assert_eq!(std::fs::read(&dest).unwrap(), b"partial sqlite pages");
        assert!(list_backups(dir.path()).iter().any(|p| p == &dest));
        // A missing temp fails loudly and never mints a destination.
        let ghost = backups.join("faktor-plus-1000.db.tmp-ghost");
        assert!(
            faktor_fs::atomic::atomic_adopt(&ghost, &backups.join("faktor-plus-1000.db")).is_err()
        );
        assert!(!backups.join("faktor-plus-1000.db").exists());
        drop(store);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn backup_blocking_work_never_occupies_the_single_tokio_worker() {
        // P0-46: the sync snapshot+rotation is executed via spawn_blocking.
        // With ONE Tokio worker, a probe task spawned WHILE the snapshot is
        // in flight must complete promptly — if the SQLite backup ran
        // inline on the worker, nothing else could run until the whole
        // snapshot finished (worker starvation).
        let dir = tempfile::tempdir().unwrap();
        let session =
            SessionManager::open_quick(dir.path().join("store"), dir.path().join("cas")).unwrap();
        // Seed a session + message so a recursive insert can build a
        // multi-hundred-thousand-row part table (a snapshot that takes real
        // time; the probe needs an observable overlap window).
        let ws = session.create_workspace("/w").unwrap();
        let sid = session.create_session(ws, "seed", "p", "m").unwrap().id();
        let mid = session
            .store()
            .put_message(sid, 1, "user", serde_json::json!({"text": "x"}))
            .unwrap();
        let store = session.store();
        for _ in 0..2 {
            store
                .sql_execute(&format!(
                    "WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM cnt WHERE x < 300000)
                     INSERT INTO part(message_id, kind, data, created_ms)
                     SELECT {mid}, 'text', '{{\"text\":\"padding\"}}', 1 FROM cnt;"
                ))
                .unwrap();
        }
        // An already-due gate (no backups exist yet): the task sleeps the
        // post-ready delay, then snapshots on the blocking pool.
        let backup = spawn_startup_backup(store, dir.path().to_path_buf());
        // Wait (bounded) until the blocking snapshot observably started: the
        // in-progress `.db.tmp-*` file exists only while backup_to runs.
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            let started = std::fs::read_dir(dir.path().join("backups"))
                .map(|rd| {
                    rd.flatten()
                        .any(|f| f.file_name().to_string_lossy().contains(".db.tmp-"))
                })
                .unwrap_or(false);
            if started {
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "the blocking snapshot never started"
            );
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
        // The snapshot is mid-flight on the blocking pool: a probe on the
        // SINGLE worker must run in well under the snapshot's duration.
        let probe = tokio::time::timeout(
            std::time::Duration::from_millis(200),
            tokio::task::spawn(async { 42 }),
        )
        .await
        .expect("the worker must stay responsive while the backup blocks")
        .expect("probe task panicked");
        assert_eq!(probe, 42);
        // And the backup finishes with exactly one complete snapshot.
        tokio::time::timeout(std::time::Duration::from_secs(60), backup)
            .await
            .expect("the backup task must finish")
            .expect("backup task panicked");
        assert_eq!(list_backups(dir.path()).len(), 1, "one complete snapshot");
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
            // The active turn's durable anchor: admission materializes the
            // prompt message at the PromptReceived journal seq, and the turn
            // record names that seq. Without the anchor the turn would be an
            // unrecoverable active turn (nothing could own it after a crash).
            session
                .store()
                .put_message(sid, 2, "user", serde_json::json!({"text": "x"}))
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
        assert!(
            report
                .lines
                .iter()
                .any(|l| l.contains("active turns with recoverable owners: 1 of 1")),
            "{:?}",
            report.lines
        );
    }

    // ---------------------------------- doctor deep invariants (P0-97/74/100)

    /// Drive one REAL task to `VerifiedComplete` through the session APIs —
    /// the ONLY legal path — with a live OPEN reservation on its row. Used
    /// by every corruption test as the healthy baseline.
    fn seed_verified_complete(
        m: &Arc<SessionManager>,
    ) -> (faktor_core::id::SessionId, faktor_core::id::TaskId, i64) {
        use faktor_core::id::{SessionId, TaskId};
        use faktor_core::state::{
            CriterionVerification, TaskState, TaskTransition, VerificationStatus,
        };
        let ws = m.create_workspace("/w").unwrap();
        let s = m.create_session(ws, "t", "p", "m").unwrap();
        let sid: SessionId = s.id();
        let task_id = TaskId::new(42);
        let now = m.now_ms();
        s.create_task(faktor_session::Task {
            task_id,
            session_id: sid,
            goal: "make it so".into(),
            acceptance_criteria: vec!["c1".into()],
            plan: vec![],
            budget: faktor_session::TaskBudget::default(),
            state: TaskState::Running,
            created_ms: now,
            updated_ms: now,
        })
        .unwrap();
        let r1 = s.task_revision(task_id).unwrap();
        s.transition_task(task_id, r1, TaskTransition::RequestVerification, None)
            .unwrap();
        let r2 = s.task_revision(task_id).unwrap();
        s.transition_task(task_id, r2, TaskTransition::StartVerification, None)
            .unwrap();
        let r3 = s.task_revision(task_id).unwrap();
        let record = s
            .create_verification_record(
                task_id,
                None,
                vec![CriterionVerification {
                    criterion_key: "c1".into(),
                    passed: true,
                    evidence: None,
                }],
                vec![],
                vec![],
                vec![],
                None,
                VerificationStatus::Passed,
                now,
            )
            .unwrap();
        s.complete_verified_task(task_id, r3, record).unwrap();
        // A live open reservation against a REAL task row in a
        // provider-permitting state: a VerifiedComplete task forbids new
        // provider operations (and a completion cannot carry an open
        // reservation), so the healthy baseline parks the reservation on a
        // second Running task of the same session.
        let reservation_task = TaskId::new(43);
        s.create_task(faktor_session::Task {
            task_id: reservation_task,
            session_id: sid,
            goal: "hold a live reservation".into(),
            acceptance_criteria: vec![],
            plan: vec![],
            budget: faktor_session::TaskBudget::default(),
            state: TaskState::Running,
            created_ms: m.now_ms(),
            updated_ms: m.now_ms(),
        })
        .unwrap();
        m.store()
            .cost_task_cap_set(sid, reservation_task, Some(1_000_000))
            .unwrap();
        let op = m.next_op_id();
        let granted = m
            .store()
            .cost_reserve(sid, reservation_task, op, 1000, now)
            .unwrap();
        let reservation_id = match granted {
            faktor_store::CostReserveOutcome::Granted(id) => id,
            _ => panic!("reservation must be granted"),
        };
        (sid, task_id, reservation_id)
    }

    /// Raw sqlite handle for crafting corruption AFTER the manager closed
    /// (doctor reopens the same file afterwards). The db lives at
    /// `store/faktor-plus.db` under the doctor data dir.
    fn raw_corruption_conn(data_dir: &std::path::Path) -> rusqlite::Connection {
        rusqlite::Connection::open(data_dir.join("store").join("faktor-plus.db")).unwrap()
    }

    #[test]
    fn doctor_deep_passes_every_p097_section_on_a_healthy_store() {
        // The full audit surface passes on data produced ONLY through the
        // real APIs: a VerifiedComplete task + its Passed record, a live
        // reservation on the real task row, an orchestrated child identity
        // row + one well-formed non-terminal registry row whose worktree
        // directory exists.
        let dir = tempfile::tempdir().unwrap();
        {
            let m = SessionManager::open_quick(dir.path().join("store"), dir.path().join("cas"))
                .unwrap();
            let ws = m.create_workspace("/w").unwrap();
            let parent = m.create_session(ws, "parent", "p", "m").unwrap();
            let (sid, task_id, _reservation) = seed_verified_complete(&m);
            let wt_path = dir.path().join("wt");
            std::fs::create_dir_all(&wt_path).unwrap();
            let wt_id = m
                .put_worktree(ws, wt_path.to_str().unwrap(), "feat/x")
                .unwrap();
            let child = m
                .create_child_session(
                    parent.id(),
                    ws,
                    faktor_core::WorktreeId::new(wt_id as u64),
                    faktor_core::TaskId::new(42),
                    "p",
                    "m",
                    "child",
                    faktor_session::ChildOwnership::ReadOnlyShared,
                )
                .unwrap();
            assert_eq!(parent.id().raw(), 1, "parent session id");
            assert_eq!(sid.raw(), 2, "verified task's session id");
            assert_eq!(child.id().raw(), 3, "child session id");
            // A well-formed NON-terminal registry row over the child (the
            // executor's durable shape, parent row space), whose worktree
            // dir exists.
            let runtime = faktor_orchestrator::runtime::ChildRuntime {
                child_id: "child-3".into(),
                parent_session_id: parent.id().raw(),
                run_id: "run-1".into(),
                item_id: "w1".into(),
                kind: faktor_orchestrator::WorkKind::Exploration,
                session_id: child.id().raw(),
                operation_id: 0,
                workspace_id: ws.raw(),
                worktree_id: wt_id as u64,
                ownership: faktor_session::ChildOwnership::ReadOnlyShared,
                ownership_paths: vec![],
                state: faktor_orchestrator::ChildState::Running,
                budget_max_tokens: None,
                permissions: faktor_orchestrator::caps::CapabilitySet::default(),
                model_policy: faktor_orchestrator::runtime::ModelPolicy::default(),
                blocker_kind: None,
                blocker_reason: None,
                blocker_dependency: None,
                blocker_resolution: None,
                last_progress_ms: None,
                created_ms: 1,
                updated_ms: 1,
                base_snapshot_id: None,
                env_snapshot_id: None,
            };
            let value = serde_json::to_string(&runtime).unwrap();
            parent
                .upsert_memory_fact(
                    "orchestrator_registry",
                    &format!("run-1/{}", runtime.child_id),
                    &value,
                )
                .unwrap();
            // Keep the (unused) id referenced so the data stays typed.
            let _ = task_id;
        }
        let report = doctor_run(dir.path(), true);
        assert_eq!(report.issues, 0, "{:?}", report.lines);
        let text = report.lines.join("\n");
        assert!(
            report
                .lines
                .iter()
                .any(|l| l == "cost reservations: 1 (open 1, settled 0, refunded 0, uncertain 0)"),
            "{text}"
        );
        assert!(text.contains("dangling cost reservations: none"), "{text}");
        assert!(
            text.contains("verification records: 1 record(s), 1 completion-relevant task(s), 1 VerifiedComplete task(s)"),
            "{text}"
        );
        assert!(text.contains("verification consistency: ok"), "{text}");
        assert!(text.contains("journal consistency: ok"), "{text}");
        assert!(
            text.contains("orphan children: 1 child identity row(s), 1 registry row(s) scanned"),
            "{text}"
        );
        assert!(text.contains("orphan children: none"), "{text}");
        assert!(
            text.contains("active turns with recoverable owners: 0 of 0"),
            "{text}"
        );
        assert!(
            text.contains("process ownership: 0 durable session-owned process row(s)"),
            "{text}"
        );
        // None of the failing prefixes may appear.
        for bad in [
            "dangling cost reservation:",
            "verification inconsistency",
            "orphan child:",
            "active turn without recoverable owner",
        ] {
            assert!(!text.contains(bad), "{bad} present in: {text}");
        }
    }

    #[test]
    fn doctor_deep_flags_dangling_open_and_settled_cost_reservations() {
        // Raw insert: RESERVED + SETTLED + UNCERTAIN reservation rows whose
        // task row does not exist (no store API can produce them —
        // cost_reserve refuses a missing task). Deep doctor must report the
        // typed section with the per-status count (reserved folds into the
        // in-flight "open" bucket) and one failing line per dangling row.
        // The legacy 'open'/'abandoned' vocabulary is dead: the v17 schema
        // CHECK rejects them at insert.
        let dir = tempfile::tempdir().unwrap();
        {
            let m = SessionManager::open_quick(dir.path().join("store"), dir.path().join("cas"))
                .unwrap();
            m.create_session(m.create_workspace("/w").unwrap(), "t", "p", "m")
                .unwrap();
        }
        {
            let conn = raw_corruption_conn(dir.path());
            conn.execute(
                "INSERT INTO cost_reservation(session_id, task_id, op_id, predicted_micro, status, created_ms)
                 VALUES (1, 424242, 5, 1234, 'reserved', 1),
                        (1, 424243, 6, 999, 'settled', 1),
                        (1, 424244, 7, 100, 'uncertain', 1)",
                [],
            )
            .unwrap();
            for dead in ["open", "abandoned"] {
                let legacy = conn.execute(
                    "INSERT INTO cost_reservation(session_id, task_id, op_id, predicted_micro, status, created_ms)
                     VALUES (1, 424245, 8, 50, ?, 1)",
                    [dead],
                );
                assert!(
                    legacy.is_err(),
                    "the v17 CHECK forbids the legacy {dead:?} vocabulary"
                );
            }
        }
        let report = doctor_run(dir.path(), true);
        assert!(report.issues >= 3, "{:?}", report.lines);
        let text = report.lines.join("\n");
        assert!(
            text.contains("cost reservations: 3 (open 1, settled 1, refunded 0, uncertain 1)"),
            "{text}"
        );
        assert!(
            report.lines.iter().any(|l| {
                l.contains("dangling cost reservation: reservation 1")
                    && l.contains("task 424242")
                    && l.contains("status reserved")
            }),
            "{text}"
        );
        assert!(
            report
                .lines
                .iter()
                .any(|l| l.contains("task 424243") && l.contains("status settled")),
            "{text}"
        );
        assert!(
            report
                .lines
                .iter()
                .any(|l| l.contains("task 424244") && l.contains("status uncertain")),
            "{text}"
        );
    }

    #[test]
    fn doctor_deep_flags_a_passed_record_for_a_missing_task() {
        // The store's raw record insert is deliberately unvalidated (the
        // session layer is the guard) — so a Passed record can reference a
        // task that never existed. Deep doctor must FAIL the wave-16
        // section with the exact kind and counts.
        let dir = tempfile::tempdir().unwrap();
        {
            let m = SessionManager::open_quick(dir.path().join("store"), dir.path().join("cas"))
                .unwrap();
            let _s = m
                .create_session(m.create_workspace("/w").unwrap(), "t", "p", "m")
                .unwrap();
            let row = faktor_store::VerificationRecordRow {
                id: faktor_core::id::VerificationRecordId::new(1), // ignored
                task_id: faktor_core::id::TaskId::new(777),
                revision: faktor_core::id::TaskRevision::new(1),
                workspace_id: faktor_core::id::WorkspaceId::new(1),
                worktree_id: faktor_core::id::WorktreeId::new(1),
                tree_hash: None,
                criteria: vec![],
                checks: vec![],
                changed_files: vec![],
                unrelated_changes: vec![],
                reviewer: None,
                status: faktor_core::state::VerificationStatus::Passed,
                started_ms: 1,
                completed_ms: None,
            };
            m.store().verification_record_put(&row).unwrap();
        }
        let report = doctor_run(dir.path(), true);
        assert!(report.issues >= 1, "{:?}", report.lines);
        let text = report.lines.join("\n");
        assert!(
            report.lines.iter().any(|l| {
                l.contains("verification inconsistency [record_without_task]")
                    && l.contains("task 777")
            }),
            "{text}"
        );
        assert!(
            text.contains("verification records: 1 record(s), 0 completion-relevant task(s), 0 VerifiedComplete task(s)"),
            "{text}"
        );
    }

    #[test]
    fn doctor_deep_flags_passed_record_certifying_an_uncompleted_task() {
        // A Passed record may only certify the CURRENT revision of a
        // VerifiedComplete task. Raw-putting one against a Running task at
        // its revision is the exact wave-16 bypass deep doctor must catch.
        let dir = tempfile::tempdir().unwrap();
        {
            let m = SessionManager::open_quick(dir.path().join("store"), dir.path().join("cas"))
                .unwrap();
            let ws = m.create_workspace("/w").unwrap();
            let s = m.create_session(ws, "t", "p", "m").unwrap();
            let task_id = faktor_core::id::TaskId::new(9);
            let now = m.now_ms();
            s.create_task(faktor_session::Task {
                task_id,
                session_id: s.id(),
                goal: "g".into(),
                acceptance_criteria: vec![],
                plan: vec![],
                budget: faktor_session::TaskBudget::default(),
                state: faktor_core::state::TaskState::Running,
                created_ms: now,
                updated_ms: now,
            })
            .unwrap();
            let row = faktor_store::VerificationRecordRow {
                id: faktor_core::id::VerificationRecordId::new(1), // ignored
                task_id,
                revision: faktor_core::id::TaskRevision::new(1),
                workspace_id: ws,
                worktree_id: faktor_core::id::WorktreeId::new(1),
                tree_hash: None,
                criteria: vec![],
                checks: vec![],
                changed_files: vec![],
                unrelated_changes: vec![],
                reviewer: None,
                status: faktor_core::state::VerificationStatus::Passed,
                started_ms: 1,
                completed_ms: None,
            };
            m.store().verification_record_put(&row).unwrap();
        }
        let report = doctor_run(dir.path(), true);
        assert!(report.issues >= 1, "{:?}", report.lines);
        let text = report.lines.join("\n");
        assert!(
            report.lines.iter().any(|l| {
                l.contains("verification inconsistency [passed_on_uncompleted]")
                    && l.contains("task 1/9")
                    && l.contains("revision 1")
            }),
            "{text}"
        );
    }

    #[test]
    fn doctor_deep_flags_verified_complete_task_whose_record_was_deleted() {
        // A legitimately completed task first passes every section; deleting
        // its consumed Passed record (raw SQL) must make the same dir FAIL
        // the wave-16 section — VerifiedComplete without its completion
        // proof.
        let dir = tempfile::tempdir().unwrap();
        {
            let m = SessionManager::open_quick(dir.path().join("store"), dir.path().join("cas"))
                .unwrap();
            seed_verified_complete(&m);
        }
        let clean = doctor_run(dir.path(), true);
        assert_eq!(clean.issues, 0, "{:?}", clean.lines);
        {
            let conn = raw_corruption_conn(dir.path());
            conn.execute("DELETE FROM verification_record", []).unwrap();
        }
        let report = doctor_run(dir.path(), true);
        assert!(report.issues >= 1, "{:?}", report.lines);
        let text = report.lines.join("\n");
        assert!(
            report.lines.iter().any(|l| {
                l.contains("verification inconsistency [verified_without_record]")
                    && l.contains("task 1/42")
                    && l.contains("at revision 4")
                    && l.contains("completion revision 3")
            }),
            "{text}"
        );
        assert!(
            text.contains("verification records: 0 record(s), 1 completion-relevant task(s), 1 VerifiedComplete task(s)"),
            "{text}"
        );
        assert!(
            text.contains("cost reservations: 1 (open 1, settled 0, refunded 0, uncertain 0)"),
            "{text}"
        );
    }

    #[test]
    fn doctor_deep_flags_active_turn_without_recoverable_owner() {
        // Raw insert of an active turn_record with NO durable anchor (no
        // prompt message, no queue row, no journal event, no tool-run row
        // names its op): after a crash nothing could own this turn.
        let dir = tempfile::tempdir().unwrap();
        {
            let m = SessionManager::open_quick(dir.path().join("store"), dir.path().join("cas"))
                .unwrap();
            m.create_session(m.create_workspace("/w").unwrap(), "t", "p", "m")
                .unwrap();
        }
        {
            let conn = raw_corruption_conn(dir.path());
            conn.execute(
                "INSERT INTO turn_record(session_id, turn_op_id, started_at, status, updated_ms)
                 VALUES (1, 999999, 1, 'active', 1)",
                [],
            )
            .unwrap();
        }
        let report = doctor_run(dir.path(), true);
        assert!(report.issues >= 1, "{:?}", report.lines);
        let text = report.lines.join("\n");
        assert!(
            text.contains("active turns with recoverable owners: 0 of 1"),
            "{text}"
        );
        assert!(
            report.lines.iter().any(|l| {
                l.contains("active turn without recoverable owner") && l.contains("op 999999")
            }),
            "{text}"
        );
    }

    #[test]
    fn doctor_deep_flags_orphan_child_rows_and_a_missing_worktree_dir() {
        // Three orphan-child corruptions: an unparseable registry row, a
        // child identity row naming a parent session that does not exist,
        // and a well-formed NON-terminal registry row whose worktree
        // directory vanished from disk.
        let dir = tempfile::tempdir().unwrap();
        let wt_row_id = {
            let m = SessionManager::open_quick(dir.path().join("store"), dir.path().join("cas"))
                .unwrap();
            let ws = m.create_workspace("/w").unwrap();
            let parent = m.create_session(ws, "parent", "p", "m").unwrap();
            let _child = m.create_session(ws, "child", "p", "m").unwrap();
            let wt_path = dir.path().join("vanished-wt");
            std::fs::create_dir_all(&wt_path).unwrap();
            let wt_id = m
                .put_worktree(ws, wt_path.to_str().unwrap(), "feat/x")
                .unwrap();
            // A well-formed NON-terminal registry row (child session 2 is
            // live; the worktree DIRECTORY is removed below).
            let runtime = faktor_orchestrator::runtime::ChildRuntime {
                child_id: "child-2".into(),
                parent_session_id: parent.id().raw(),
                run_id: "run-1".into(),
                item_id: "w1".into(),
                kind: faktor_orchestrator::WorkKind::Exploration,
                session_id: 2,
                operation_id: 0,
                workspace_id: ws.raw(),
                worktree_id: wt_id as u64,
                ownership: faktor_session::ChildOwnership::ReadOnlyShared,
                ownership_paths: vec![],
                state: faktor_orchestrator::ChildState::Running,
                budget_max_tokens: None,
                permissions: faktor_orchestrator::caps::CapabilitySet::default(),
                model_policy: faktor_orchestrator::runtime::ModelPolicy::default(),
                blocker_kind: None,
                blocker_reason: None,
                blocker_dependency: None,
                blocker_resolution: None,
                last_progress_ms: None,
                created_ms: 1,
                updated_ms: 1,
                base_snapshot_id: None,
                env_snapshot_id: None,
            };
            let value = serde_json::to_string(&runtime).unwrap();
            parent
                .upsert_memory_fact(
                    "orchestrator_registry",
                    &format!("run-1/{}", runtime.child_id),
                    &value,
                )
                .unwrap();
            std::fs::remove_dir_all(&wt_path).unwrap();
            wt_id
        };
        {
            let conn = raw_corruption_conn(dir.path());
            // Unparseable registry row under session 1 (corruption, never a
            // silent skip).
            conn.execute(
                "INSERT INTO memory_fact(session_id, kind, key, value, updated_ms)
                 VALUES (1, 'orchestrator_registry', 'run-1/child-x', '{not-json', 1)",
                [],
            )
            .unwrap();
            // Child identity naming a parent session that has no row.
            conn.execute(
                "INSERT INTO memory_fact(session_id, kind, key, value, updated_ms)
                 VALUES (2, 'orchestrator', 'identity',
                         '{\"parent_session_id\":777,\"workspace_id\":1,\"worktree_id\":1,\"item_id\":\"i\",\"task_goal\":\"\",\"operation_id\":0,\"ownership\":\"read_only_shared\",\"model\":\"\",\"created_ms\":1}',
                         1)",
                [],
            )
            .unwrap();
        }
        let report = doctor_run(dir.path(), true);
        assert!(report.issues >= 3, "{:?}", report.lines);
        let text = report.lines.join("\n");
        assert!(
            report
                .lines
                .iter()
                .any(|l| l.contains("orphan child: unparseable orchestrator registry row")),
            "{text}"
        );
        assert!(
            report.lines.iter().any(|l| {
                l.contains("orphan child: child session 2 carries an identity row naming parent session 777")
            }),
            "{text}"
        );
        assert!(
            report.lines.iter().any(|l| {
                l.contains("orphan child: non-terminal child child-2")
                    && l.contains("no worktree directory")
            }),
            "{text}"
        );
        assert!(
            text.contains("orphan children: 1 child identity row(s), 2 registry row(s) scanned"),
            "{text}"
        );
        let _ = wt_row_id;
    }

    // ------------------------------------------------------------- wiring

    /// One chat request whose user text can echo configured secrets.
    fn chat_req(text: &str) -> GenericAgentRequest {
        GenericAgentRequest {
            model: "m".into(),
            system: "sys".into(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec![ContentPart::text(text)],
            }],
            tools: vec![ToolSpec {
                name: "read_file".into(),
                description: "read".into(),
                input_schema: serde_json::json!({"type": "object"}),
            }],
            max_output: Some(64),
            reasoning: None,
            stream: true,
            meta: RequestMeta {
                operation_id: OpId::new(1),
                session_id: SessionId::new(1),
                provider: "cli-wiring-test".into(),
                attempt: 0,
                deadline_ms: 10_000,
                cancellation: CancellationToken::new(),
            },
        }
    }

    /// Drive one chat call to completion; `Err` carries the provider error
    /// (a policy/secret refusal arrives before any server contact).
    async fn chat_text(provider: Arc<dyn Provider>, text: &str) -> Result<String, ProviderError> {
        let mut stream = provider.stream(chat_req(text));
        let mut out = String::new();
        while let Some(item) = stream.next().await {
            match item {
                Ok(ProviderChunk::Text { text: t }) => out.push_str(&t),
                Ok(ProviderChunk::Done) => break,
                Ok(_) => {}
                Err(e) => return Err(e),
            }
        }
        Ok(out)
    }

    /// A config with ONE OpenAI-compatible provider at `base` and the given
    /// sandbox network rows (`None` = no sandbox section = crate defaults).
    fn egress_cfg(
        dir: &std::path::Path,
        file: &str,
        base: &str,
        key_env: Option<&str>,
        rows: Option<&[String]>,
    ) -> config::Config {
        let mut body = serde_json::json!({
            "model": "m",
            "providers": [{
                "kind": "open_ai",
                "id": "mocked",
                "base_url": base,
                "api_key_env": key_env,
            }],
        });
        if let Some(rows) = rows {
            body["sandbox"] = serde_json::json!({ "network": rows });
        }
        let path = dir.join(file);
        std::fs::write(&path, serde_json::to_string(&body).unwrap()).unwrap();
        config::Config::load(&path).unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn daemon_provider_transports_carry_the_destination_policy() {
        let server = MockServer::new();
        server.route(
            "POST",
            "/chat/completions",
            MockAction::Respond {
                status: 200,
                body: sse_body(&[serde_json::json!({
                    "choices": [{"delta": {"content": "allowed"}, "finish_reason": "stop"}]
                })]),
            },
        );
        let base = server.base_url().await;
        let port: u16 = base.rsplit(':').next().unwrap().parse().unwrap();
        let allow_row = format!("http://127.0.0.1:{port}");

        // (a) The allow row rides into the CONSTRUCTED transports: the chat
        // request reaches the mock and streams.
        let dir = tempfile::tempdir().unwrap();
        let cfg = egress_cfg(
            dir.path(),
            "allow.json",
            &base,
            None,
            Some(std::slice::from_ref(&allow_row)),
        );
        let graph = build_daemon(dir.path(), Some(cfg)).unwrap();
        let provider = graph.providers.get("mocked").expect("provider registered");
        let text = chat_text(provider, "hello").await.expect("allowed chat");
        assert_eq!(text, "allowed");
        assert_eq!(server.request_count(), 1, "the allowed request arrived");
        drop(graph);

        // (b) Rows that deny the actual host (only a DIFFERENT port is
        // allowlisted): the SAME call fails pre-connect with the typed
        // denial and is never retried — the server sees nothing new.
        let dir2 = tempfile::tempdir().unwrap();
        let wrong_row = format!("http://127.0.0.1:{}", port.wrapping_add(1));
        let cfg = egress_cfg(dir2.path(), "deny.json", &base, None, Some(&[wrong_row]));
        let graph = build_daemon(dir2.path(), Some(cfg)).unwrap();
        let err = chat_text(graph.providers.get("mocked").unwrap(), "hello")
            .await
            .expect_err("a denied destination must fail the chat");
        assert!(err.message.contains("denied"), "{}", err.message);
        assert!(!err.retryable, "policy denials are never retried");
        assert_eq!(server.request_count(), 1, "deny happened before connect");
        drop(graph);

        // (c) An empty row list denies EVERY destination before connect.
        let dir3 = tempfile::tempdir().unwrap();
        let cfg = egress_cfg(dir3.path(), "denyall.json", &base, None, Some(&[]));
        let graph = build_daemon(dir3.path(), Some(cfg)).unwrap();
        let err = chat_text(graph.providers.get("mocked").unwrap(), "hello")
            .await
            .expect_err("an empty allowlist denies everything");
        assert!(err.message.contains("denied"), "{}", err.message);
        assert_eq!(server.request_count(), 1);
        drop(graph);

        // (d) The DEFAULT daemon (no sandbox section) enforces the sandbox
        // crate's frozen provider-endpoint allowlist: the localhost mock is
        // NOT on it, so egress is denied pre-connect — adapters are never
        // permissively default-transported.
        let dir4 = tempfile::tempdir().unwrap();
        let cfg = egress_cfg(dir4.path(), "default.json", &base, None, None);
        let graph = build_daemon(dir4.path(), Some(cfg)).unwrap();
        let err = chat_text(graph.providers.get("mocked").unwrap(), "hello")
            .await
            .expect_err("the frozen default allowlist denies the mock host");
        assert!(err.message.contains("denied"), "{}", err.message);
        assert_eq!(server.request_count(), 1, "default policy: no connect");
        drop(graph);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn daemon_secret_registry_blocks_a_configured_key_echo_before_connect() {
        const KEY_ENV: &str = "KP_CLI_WIRING_FAKE_KEY";
        const SECRET: &str = "kp-cli-secret-token-91f7c2e8d4";
        // The value must not trip the frozen GENERIC scan patterns (sk-*,
        // ghp_, AKIA, ...): only the configured-secret registry can catch
        // it, so a block proves the registry was populated from the key.
        std::env::set_var(KEY_ENV, SECRET);
        let server = MockServer::new();
        server.route(
            "POST",
            "/chat/completions",
            MockAction::Respond {
                status: 200,
                body: sse_body(&[serde_json::json!({
                    "choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]
                })]),
            },
        );
        let base = server.base_url().await;
        let port: u16 = base.rsplit(':').next().unwrap().parse().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let cfg = egress_cfg(
            dir.path(),
            "secret.json",
            &base,
            Some(KEY_ENV),
            Some(&[format!("http://127.0.0.1:{port}")]),
        );
        let graph = build_daemon(dir.path(), Some(cfg)).unwrap();
        let provider = graph.providers.get("mocked").unwrap();
        // Control: a clean body reaches the mock (the scan does not
        // false-positive on ordinary chat text).
        let text = chat_text(provider.clone(), "hello").await.unwrap();
        assert_eq!(text, "ok");
        assert_eq!(server.request_count(), 1);
        // The key echo: the whole payload is scanned and the request is
        // denied BEFORE any connect — the mock never sees it.
        let err = chat_text(provider, SECRET)
            .await
            .expect_err("a configured key echoed in the body must be blocked");
        assert!(err.message.contains("secret"), "{}", err.message);
        assert!(!err.retryable);
        assert_eq!(
            server.request_count(),
            1,
            "the key-echo request never arrived at the server"
        );
        drop(graph);
        std::env::remove_var(KEY_ENV);
    }

    #[test]
    fn verification_config_unknown_fields_fail_and_zero_disables_the_service() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.json");
        // Sane values parse (strictly) and drive the daemon service: the
        // derived per-check budget is the configured cap.
        std::fs::write(
            &path,
            r#"{"verification": {"quick_max_s": 30, "unit_max_s": 120, "full_as_background": false}}"#,
        )
        .unwrap();
        let cfg = serve_config(Some(path.clone())).expect("explicit sane config");
        let supervisor = ProcessSupervisor::shared();
        let service = daemon_verification(&cfg.verification, &supervisor);
        assert!(!service.is_disabled());
        let policy = service.policy();
        assert_eq!(policy.quick_max, std::time::Duration::from_secs(30));
        assert_eq!(policy.unit_max, std::time::Duration::from_secs(120));
        assert!(!policy.full_as_background);
        // Budget probe through the service (same decision the genuine-end
        // site applies per check category).
        let quick = faktor_verify::exec::CheckSpec::new(
            "q",
            faktor_verify::exec::CheckKind::Compile,
            faktor_verify::exec::CheckCategory::Quick,
            "cargo",
            ["check"],
            true,
        );
        assert_eq!(
            service.budget_for(&quick),
            faktor_verify::exec::BudgetDecision::RunInline(std::time::Duration::from_secs(30))
        );
        // An explicit unknown field inside [verification] fails startup.
        std::fs::write(&path, r#"{"verification": {"bogus": 1}}"#).unwrap();
        let e = serve_config(Some(path.clone())).expect_err("unknown field fails startup");
        assert!(e.contains("unknown field"), "{e}");
        // quick_max_s = 0 yields the DISABLED service: fail closed.
        std::fs::write(
            &path,
            r#"{"verification": {"quick_max_s": 0, "unit_max_s": 0}}"#,
        )
        .unwrap();
        let cfg = serve_config(Some(path)).expect("zero quick budget is a valid config");
        let supervisor = ProcessSupervisor::shared();
        let service = daemon_verification(&cfg.verification, &supervisor);
        assert!(
            service.is_disabled(),
            "quick_max_s = 0 must disable verification (fail closed)"
        );
    }

    #[test]
    fn network_guarantee_required_flows_into_the_daemon_sandbox_gate() {
        // Authority direction (audit P0-39): the configured guarantee
        // reaches the daemon's PermissionEngine, which DECIDES the spawn
        // requirement (Required => DenyAll) and performs NO platform
        // pre-judgement. The capability verdict stays the plain rule (Ask
        // by default); enforcement honesty belongs to the spawn layer,
        // which refuses typed when it cannot isolate.
        let cfg = config::Config {
            sandbox: config::SandboxCfg {
                network_guarantee: faktor_sandbox::SandboxGuarantee::Required,
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(
            cfg.sandbox_policy().unwrap().network_guarantee,
            faktor_sandbox::SandboxGuarantee::Required
        );
        let dir = tempfile::tempdir().unwrap();
        let graph = build_daemon(dir.path(), Some(cfg)).unwrap();
        let sandbox = graph
            .agent
            .deps()
            .sandbox
            .clone()
            .expect("daemon sandbox wired");
        assert_eq!(
            sandbox.policy().network_guarantee,
            faktor_sandbox::SandboxGuarantee::Required,
            "the guarantee surfaces into the daemon SandboxPolicy"
        );
        assert_eq!(
            sandbox.spawn_network_requirement(),
            faktor_terminal::NetworkIsolationRequirement::DenyAll,
            "Required must reach the spawn seam as DenyAll"
        );
        let decision = sandbox.evaluate(&Capability::ExecuteShell {
            command: "echo hi".into(),
        });
        assert_eq!(
            decision,
            faktor_core::PermissionDecision::Ask,
            "no preflight platform guessing: the rule decides (Ask default)"
        );
        drop(graph);
    }

    #[test]
    fn daemon_hooks_run_env_clear_exact_through_the_daemon_supervisor() {
        // (b) cli-level hook harness: the daemon hook registry constructor
        // (supervisor-rooted, daemon envelope) runs a hook whose env is
        // cleared EXACTLY (only explicit entries + FAKTOR_HOOK_INPUT), and
        // the audit row lands with the bounded stdout.
        std::env::set_var("KP_DAEMON_ONLY_SECRET", "must-not-leak-to-hooks");
        let dir = tempfile::tempdir().unwrap();
        let session =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let supervisor = ProcessSupervisor::new(session.cas());
        let spec = faktor_hooks::HookSpec {
            id: "env-0".into(),
            events: vec![faktor_hooks::HookEvent::PreTool],
            command: "/usr/bin/env".into(),
            args: vec![],
            env: vec![("KP_HOOK_VISIBLE".into(), "visible".into())],
            env_allowlist: true,
            deadline_ms: 5000,
            failure_policy: faktor_hooks::FailurePolicy::FailClosed,
            ..Default::default()
        };
        let registry = hook_registry(&supervisor, vec![spec]).expect("registry built");
        let verdict = registry.run(
            faktor_hooks::HookEvent::PreTool,
            &faktor_hooks::HookInput::default(),
        );
        assert_eq!(verdict, faktor_hooks::HookVerdict::Allow);
        let audit = registry.audit();
        assert_eq!(audit.len(), 1);
        assert_eq!(audit[0].hook_id, "env-0");
        let stdout = &audit[0].stdout_head;
        assert!(
            stdout.contains("KP_HOOK_VISIBLE=visible"),
            "explicit entries pass: {stdout}"
        );
        assert!(
            stdout.contains("FAKTOR_HOOK_INPUT="),
            "the input JSON rides the env: {stdout}"
        );
        assert!(
            !stdout.contains("KP_DAEMON_ONLY_SECRET"),
            "env-clear exact: the daemon env must never reach the hook: {stdout}"
        );
        // The run went through the supervisor registry and left nothing
        // alive (bounded child, reaped).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !supervisor.alive().is_empty() {
            assert!(std::time::Instant::now() < deadline, "child leaked");
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        std::env::remove_var("KP_DAEMON_ONLY_SECRET");
    }

    #[test]
    fn daemon_hook_deadline_kills_the_group_and_audits_the_refusal() {
        // (b) deadline group-kill through the daemon supervisor: an
        // over-deadline hook is killed process-group-wide, its audit row
        // records the refusal (never the partial output), and no child is
        // left behind.
        let dir = tempfile::tempdir().unwrap();
        let session =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let supervisor = ProcessSupervisor::new(session.cas());
        let spec = faktor_hooks::HookSpec {
            id: "slow".into(),
            events: vec![faktor_hooks::HookEvent::PreTool],
            command: "/bin/sh".into(),
            args: vec!["-c".into(), "sleep 30".into()],
            env_allowlist: true,
            deadline_ms: 300,
            failure_policy: faktor_hooks::FailurePolicy::FailClosed,
            ..Default::default()
        };
        let registry = hook_registry(&supervisor, vec![spec]).expect("registry built");
        let verdict = registry.run(
            faktor_hooks::HookEvent::PreTool,
            &faktor_hooks::HookInput::default(),
        );
        assert!(
            matches!(verdict, faktor_hooks::HookVerdict::Deny { .. }),
            "fail-closed deadline refusal: {verdict:?}"
        );
        let audit = registry.audit();
        assert_eq!(audit.len(), 1);
        assert_eq!(audit[0].hook_id, "slow");
        assert_eq!(audit[0].verdict, "deny");
        assert!(
            audit[0].duration_ms >= 200,
            "the deadline dominated: {} ms",
            audit[0].duration_ms
        );
        assert!(
            audit[0].exit_code.is_none() && audit[0].stdout_head.is_empty(),
            "killed before output; partial output never decides: {:?}",
            audit[0]
        );
        // Group-kill: no live child survives the deadline.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !supervisor.alive().is_empty() {
            assert!(std::time::Instant::now() < deadline, "killed child leaked");
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn mcp_hook_and_terminal_children_share_one_daemon_supervisor() {
        // (a) The daemon owns EXACTLY ONE supervisor. Two MCP servers, a
        // long hook and a long terminal run CONCURRENTLY all admit into the
        // SAME bounded registry: while the hook and the terminal are live,
        // the single supervisor accounts 4 live children (2 mcp + hook +
        // terminal) — the old per-server supervisors would have split them
        // across registries.
        if std::process::Command::new("python3")
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("python3 missing; skipping");
            return;
        }
        let fixture = format!("{}/tests/fixtures/mcp_mock.py", env!("CARGO_MANIFEST_DIR"));
        assert!(
            std::path::Path::new(&fixture).exists(),
            "mcp fixture missing at {fixture}"
        );
        let mcp_entries = vec![
            config::McpEntry {
                name: "mock-a".into(),
                command: "python3".into(),
                args: vec![fixture.clone()],
            },
            config::McpEntry {
                name: "mock-b".into(),
                command: "python3".into(),
                args: vec![fixture.clone()],
            },
        ];
        let cfg = config::Config {
            mcp: mcp_entries,
            ..Default::default()
        };
        // The env hook path registers NO long hook here (the env format
        // cannot carry an unquoted `sleep 3` argument vector), so the long
        // hook is registered through the SAME cli helper `serve` uses
        // (env_hook_registry delegates here) onto the daemon's supervisor.
        let dir = tempfile::tempdir().unwrap();
        let graph = build_daemon_with_mcp(dir.path(), Some(cfg))
            .await
            .expect("daemon with two mcp servers builds");
        let supervisor = graph
            .agent
            .deps()
            .supervisor
            .clone()
            .expect("daemon supervisor wired");
        let registry = hook_registry(
            &supervisor,
            vec![faktor_hooks::HookSpec {
                id: "env-0".into(),
                events: vec![faktor_hooks::HookEvent::PreTool],
                command: "/bin/sh".into(),
                args: vec!["-c".into(), "sleep 3".into()],
                env_allowlist: true,
                deadline_ms: 20_000,
                failure_policy: faktor_hooks::FailurePolicy::FailClosed,
                ..Default::default()
            }],
        )
        .expect("long hook registered");
        let hooks = registry;
        // Both MCP children live in the daemon supervisor's registry.
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            if supervisor.alive().len() == 2 {
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "two mcp children never appeared: {:?}",
                supervisor.alive()
            );
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert!(graph.mcp_servers.iter().all(|s| s.is_alive()));
        // A long hook AND a long terminal child concurrently with the MCP
        // servers: one registry counts all four.
        let registry = hooks.clone();
        let hook_thread = std::thread::spawn(move || {
            registry.run(
                faktor_hooks::HookEvent::PreTool,
                &faktor_hooks::HookInput::default(),
            )
        });
        let sup = supervisor.clone();
        let terminal_thread = std::thread::spawn(move || {
            sup.run_sync(
                SpawnConfig {
                    cmd: "/bin/sh".into(),
                    args: vec!["-c".into(), "sleep 3".into()],
                    cwd: std::env::temp_dir(),
                    env: EnvSpec::Minimal,
                    owner: ProcessOwner::Daemon,
                    ..Default::default()
                },
                std::time::Duration::from_secs(30),
                64 * 1024,
                64 * 1024,
            )
        });
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            if supervisor.alive().len() == 4 {
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "expected 4 live children in the ONE registry, saw {:?}",
                supervisor.alive()
            );
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        let hook_verdict = hook_thread.join().expect("hook thread panicked");
        assert_eq!(
            hook_verdict,
            faktor_hooks::HookVerdict::Allow,
            "the long hook exits cleanly within its deadline"
        );
        terminal_thread
            .join()
            .expect("terminal thread panicked")
            .expect("the terminal child must complete");
        // Both finished: the MCP children remain, counted by the SAME
        // supervisor the hooks and terminal ran through.
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            if supervisor.alive().len() == 2 {
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "children leaked after the runs: {:?}",
                supervisor.alive()
            );
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        drop(graph);
    }

    #[test]
    fn full_scope_env_hooks_run_under_the_daemon_envelope() {
        // The daemon envelope grants the full capability lattice: an env
        // hook whose typed scope is the TOP element still registers and
        // runs (construction proof of the with_supervisor envelope seam).
        let dir = tempfile::tempdir().unwrap();
        let session =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let supervisor = ProcessSupervisor::new(session.cas());
        let spec = faktor_hooks::HookSpec {
            id: "full".into(),
            events: vec![faktor_hooks::HookEvent::TaskComplete],
            command: "/bin/echo".into(),
            args: vec!["done".into()],
            permission_scope: CapabilitySet::ALL,
            ..Default::default()
        };
        let registry = hook_registry(&supervisor, vec![spec]).expect("registry built");
        let verdict = registry.run(
            faktor_hooks::HookEvent::TaskComplete,
            &faktor_hooks::HookInput::default(),
        );
        assert_eq!(verdict, faktor_hooks::HookVerdict::Allow);
        assert_eq!(registry.audit().len(), 1);
    }
}
