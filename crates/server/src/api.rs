//! The daemon's HTTP/SSE surface.
//!
//! Two surfaces coexist (architecture §16): the **Faktor Native Protocol
//! v1** endpoints (`/session/{id}/projection`, `/models`,
//! `/capabilities`, plus the audit 55-56 `/native/...` mounts:
//! liveness/readiness, durable session listings and the usage aggregate;
//! documented in `docs/native-protocol.md`) — the
//! daemon's own contract, UI compatibility being the target — and the
//! **v7.5.6 wire compatibility surface (subset)** retained as
//! migration/test glue against the old UI:
//! the SDK-shaped REST surface (`/session/...`, `/permission/...`,
//! `/provider/list`, `/global/health`, `/global/event`,
//! `/question/...`, `/network/...`, `/config/...`) and the wire surface
//! the frozen v7.5.6 extension actually calls (`/session`,
//! `/session/{sessionID}`, `/session/{sessionID}/message`,
//! `/session/{sessionID}/abort`, `/session/{sessionID}/diff`,
//! `/session/{sessionID}/revert`, `/session/{sessionID}/unrevert`), all
//! behind password auth (`FAKTOR_SERVER_PASSWORD` via
//! `Authorization: Basic base64("kilo:"+password)`, with the Bearer and
//! `x-faktor-server-password` forms retained). The old `/api/...` routes stay
//! wired as aliases; their tests must keep passing.

//! Layout after the audit 81-83/94 split: handler bodies live in
//! [`crate::native`] (Faktor Native Protocol v1) and [`crate::compat`]
//! (frozen v7.5.6/SDK glue). This module keeps router assembly plus the
//! re-exports the tests and the daemon entry points consume; the native
//! layer never imports the compatibility layer.
//!
//! Two surfaces coexist (architecture §16): the **Faktor Native Protocol
//! v1** endpoints and the **v7.5.6 wire compatibility surface (subset)**.
//! Both stay behind password auth.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::routing::{delete, get, post};
use axum::Router;
use tokio::sync::oneshot;
use tower_http::limit::RequestBodyLimitLayer;

use faktor_agent::AgentRuntime;
use faktor_protocol::v756::{startup_line, Handshake};
use faktor_session::SessionManager;

use crate::auth::{AuthToken, ServerPassword};
use crate::global::GlobalEventBus;
use crate::permission::ChannelPermissionRequester;

pub(crate) use crate::compat::*;
pub(crate) use crate::native::*;

/// Handle to the daemon's evidence store: an evidence store behind a
/// process-wide lock, so the server can hold it while producers (future
/// audit) insert. Constructed through [`empty_evidence_store`] in hosts
/// that do not yet run a capture path.
pub type EvidenceStoreHandle =
    Arc<std::sync::RwLock<Box<dyn faktor_evidence::store::EvidenceStore + Send + Sync>>>;

/// The default empty evidence store: an in-memory store that retains
/// bounded backing bytes. `/native/evidence/{id}` answers an honest 404
/// until a producer inserts, never a fabricated envelope.
pub fn empty_evidence_store() -> EvidenceStoreHandle {
    Arc::new(std::sync::RwLock::new(Box::new(
        faktor_evidence::store::MemoryEvidenceStore::new(4 * 1024 * 1024),
    )))
}

const MAX_BODY_BYTES: usize = 10 * 1024 * 1024;

pub struct ServerDeps {
    pub session: Arc<SessionManager>,
    pub agent: Arc<AgentRuntime>,
    pub permissions: Arc<ChannelPermissionRequester>,
    /// The orchestration runtime (audits P0-20/21/23/61): the AUTHORITATIVE
    /// executor of multi-agent tasks and the durable control surface the
    /// `/native/agents/{child}/...` endpoints drive. Non-optional in
    /// production: the CLI wires the real runtime in the daemon graph
    /// region; `ServerDeps::new` (tests) builds one over the same
    /// session+agent.
    pub orchestrator: Arc<faktor_orchestrator::runtime::OrchestratorRuntime>,
    /// The TaskExecutor (audits P0-20/21/23/61/90/91): ONE entry for native
    /// task starts — single-item tasks drive the existing session with the
    /// daemon's own prompt path, multi-item tasks spawn real children
    /// through [`ServerDeps::orchestrator`].
    pub tasks: Arc<faktor_orchestrator::runtime::task_executor::TaskExecutor>,
    /// The daemon's durable cost ledger (audit 12/17): the SAME ledger the
    /// graph built. Handlers that touch per-task/per-child caps use this
    /// authority instead of constructing a second ledger per request.
    pub budgets: Arc<faktor_session::DurableBudgetLedger>,
    /// Legacy per-start token (old tests); the frontend uses the password.
    pub auth_token: AuthToken,
    /// The password the frontend generated and passed via `FAKTOR_SERVER_PASSWORD`.
    pub server_password: ServerPassword,
    /// Workspace root carried on global event envelopes.
    pub directory: Option<String>,
    pub version: String,
    /// Real workspace file service for revert/unrevert/diff (None = the wire
    /// surface refuses with an honest 409).
    pub fs: Option<Arc<faktor_fs::WorkspaceFileService>>,
    /// Real checkpoint store for revert/unrevert/diff (None = honest 409).
    pub snapshots: Option<Arc<faktor_snapshot::CheckpointStore>>,
    /// The daemon's evidence store (audit 82): scope-checked reads of
    /// captured evidence envelopes for the native evidence endpoints.
    /// `None` = no store wired in this host (the endpoints answer 503).
    pub evidence: Option<EvidenceStoreHandle>,
    /// The semantic provider registry (audit 83) surfaced by
    /// `/native/semantic/{status,capabilities}`. `None` = no provider is
    /// configured; the endpoints report the fallback registry shape.
    pub semantic: Option<Arc<faktor_semantic::registry::SemanticProviderRegistry>>,
    /// Live chunk stream from the agent (audit round 11): when present,
    /// serve() drains it into low-latency session.next.*.delta frames.
    /// Bounded (audit 41): the agent's [`faktor_agent::ChunkSink`] sender
    /// half coalesces ephemeral deltas under backpressure instead of
    /// growing memory; this receiver half stays drained eagerly into the
    /// bounded global ring.
    pub chunk_rx: Option<tokio::sync::mpsc::Receiver<faktor_agent::ChunkEvent>>,
    /// Deterministic readiness knob (audit 55; mirrors the suggested
    /// `FAKTOR_SIMULATE_NOT_READY=1` gate as a field — an env gate would
    /// race parallel tests in one process). When true, the ready flag stays
    /// false after serve() setup, so `GET /native/ready` keeps answering
    /// 503 `{"ready":false}`. Production callers never set it.
    pub simulate_not_ready: bool,
}

impl ServerDeps {
    /// Default construction (embedded hosts + tests): builds a runtime over
    /// the same session+agent. The daemon NEVER uses this path for its
    /// production surface — `serve` assembles `ServerDeps` from the named
    /// daemon graph's instances through [`ServerDeps::new_with`], so no
    /// second orchestrator/executor is ever constructed in a daemon
    /// lifetime (audit 12/17).
    pub fn new(
        session: Arc<SessionManager>,
        agent: Arc<AgentRuntime>,
        permissions: Arc<ChannelPermissionRequester>,
    ) -> Self {
        let orchestrator =
            faktor_orchestrator::runtime::OrchestratorRuntime::new(session.clone(), agent.clone());
        // No shadow service in this embedded/test shape (test harnesses);
        // the daemon's ONE TaskExecutor construction path is the CLI graph.
        let tasks = faktor_orchestrator::runtime::task_executor::TaskExecutor::new(
            &orchestrator,
            session.clone(),
            agent.clone(),
            None,
        );
        let budgets = faktor_session::DurableBudgetLedger::new(session.clone());
        Self::new_with(session, agent, permissions, orchestrator, tasks, budgets)
    }

    /// Assemble the server surface over GRAPH-PROVIDED runtime authorities
    /// (audit 12/17): production passes the daemon graph's orchestrator,
    /// TaskExecutor and budget ledger — the SAME instances the rest of the
    /// daemon uses — so the server never constructs a second execution or
    /// money authority.
    pub fn new_with(
        session: Arc<SessionManager>,
        agent: Arc<AgentRuntime>,
        permissions: Arc<ChannelPermissionRequester>,
        orchestrator: Arc<faktor_orchestrator::runtime::OrchestratorRuntime>,
        tasks: Arc<faktor_orchestrator::runtime::task_executor::TaskExecutor>,
        budgets: Arc<faktor_session::DurableBudgetLedger>,
    ) -> Self {
        Self {
            session,
            agent,
            permissions,
            orchestrator,
            tasks,
            budgets,
            auth_token: AuthToken::generate(),
            server_password: ServerPassword::from_env(),
            directory: None,
            version: faktor_core::VERSION.to_string(),
            fs: None,
            snapshots: None,
            evidence: None,
            semantic: None,
            chunk_rx: None,
            simulate_not_ready: false,
        }
    }

    /// Wire the real native snapshot store so `/session/{id}/revert`,
    /// `/unrevert` and `/diff` actually restore files. Both must be provided
    /// together; with `None` the endpoints keep their honest 409.
    pub fn with_snapshots(
        mut self,
        fs: Arc<faktor_fs::WorkspaceFileService>,
        snapshots: Arc<faktor_snapshot::CheckpointStore>,
    ) -> Self {
        self.fs = Some(fs);
        self.snapshots = Some(snapshots);
        self
    }

    /// Wire the daemon's evidence store so the native evidence endpoints can
    /// serve scope-checked reads. Production wires the graph's store here;
    /// embedded hosts leave it `None` and answer an honest 503.
    pub fn with_evidence_store(mut self, store: EvidenceStoreHandle) -> Self {
        self.evidence = Some(store);
        self
    }

    /// Wire the semantic provider registry surfaced by the native semantic
    /// introspection endpoints. `None` reports the fallback-only shape.
    pub fn with_semantic_registry(
        mut self,
        registry: Arc<faktor_semantic::registry::SemanticProviderRegistry>,
    ) -> Self {
        self.semantic = Some(registry);
        self
    }

    /// Legacy JSON handshake line (test-only detail; never printed by the
    /// CLI — the frontend parses the startup line instead).
    pub fn handshake_line(&self, addr: SocketAddr) -> String {
        Handshake {
            version: self.version.clone(),
            protocol: faktor_core::PROTOCOL_V756.to_string(),
            pid: std::process::id() as u64,
            auth_token: self.auth_token.as_str().to_string(),
            port: addr.port(),
        }
        .to_line()
    }

    /// The frozen stdout line: `faktor server listening on http://127.0.0.1:<port>`.
    pub fn startup_line(&self, addr: SocketAddr) -> String {
        startup_line(addr.port())
    }
}

pub struct ServerHandle {
    pub addr: SocketAddr,
    pub shutdown: oneshot::Sender<()>,
    /// Legacy JSON handshake line (kept for old tests; not printed).
    pub handshake: String,
    /// The frozen startup line the CLI prints on stdout after binding.
    pub startup_line: String,
}

/// Bind (port 0 = ephemeral) and serve. Returns once listening.
pub async fn serve(mut deps: ServerDeps, port: u16) -> std::io::Result<ServerHandle> {
    // Bind first, then compute the lines (needs the bound address) and
    // finally move the deps into the router.
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    let addr = listener.local_addr()?;
    let handshake = deps.handshake_line(addr);
    let startup_line = deps.startup_line(addr);
    // Readiness (audit 55): the flag starts false and flips true ONLY when
    // setup completes. Recovery runs before serve in the caller (the CLI
    // opens the store and runs `agent.recover()` first), migrations applied
    // at store open are implicit, and the required runtime components are
    // non-optional `Arc`s in `ServerDeps` — so the flip at the end of setup
    // is exactly the ready moment. `simulate_not_ready` (tests) keeps it
    // false forever: /native/ready answers 503 {ready:false}.
    let ready = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let set_ready = !deps.simulate_not_ready;
    let bus = Arc::new(GlobalEventBus::new(
        deps.session.clone(),
        deps.directory.clone(),
    ));
    // Live chunk fan-out (audit round 11): low-latency session.next.*.delta
    // frames from the agent's bounded, coalescing stream (audit 41),
    // independent of the journal re-diff window. The task ends when the
    // sender half is dropped; push_chunk only appends to the bounded global
    // ring, so a slow SSE subscriber can never back up this drainer.
    if let Some(mut rx) = deps.chunk_rx.take() {
        let bus2 = bus.clone();
        tokio::spawn(async move {
            while let Some(chunk) = rx.recv().await {
                bus2.push_chunk(chunk);
            }
        });
    }
    let app = Router::new()
        // Legacy aliases (frozen for old tests).
        .route("/api/hello", get(hello))
        .route("/api/session", post(create_session))
        .route("/api/sessions", get(list_sessions))
        .route("/api/session/{id}", get(session_state))
        .route("/api/session/{id}/state", get(session_state))
        .route("/api/session/{id}/messages", get(messages))
        .route("/api/session/{id}/events", get(events))
        .route("/api/session/{id}/prompt", post(prompt))
        .route("/api/session/{id}/abort", post(abort))
        .route("/api/perm/{id}/resolve", post(resolve_permission))
        .route("/api/provider", get(provider_list))
        // SDK-shaped primary surface.
        .route("/session/create", post(create_session))
        .route("/session/prompt", post(sdk_prompt))
        .route("/session/abort", post(sdk_abort))
        .route("/session/messages", get(sdk_messages))
        .route("/session/state", get(sdk_session_state))
        .route("/session/list", get(list_sessions))
        .route("/permission/reply", post(permission_reply))
        .route("/permission/list", get(permission_list))
        .route("/provider/list", get(provider_list))
        .route("/global/health", get(health))
        .route("/global/event", get(global_events))
        .route("/question/reply", post(question_reply))
        .route("/question/list", get(question_list))
        .route("/network/reply", post(network_reply))
        .route("/network/list", get(network_list))
        .route("/config/get", get(config_get))
        .route("/config/set", post(config_set))
        // v7.5.6 wire compatibility surface (subset): the routes the frozen
        // extension actually calls.
        .route(
            "/session",
            post(wire_create_session).get(wire_list_sessions),
        )
        .route(
            "/session/{sessionID}",
            get(wire_session_summary)
                .post(wire_session_update)
                .delete(wire_session_delete),
        )
        .route("/session/{sessionID}/fork", post(wire_session_fork))
        .route(
            "/session/{sessionID}/summarize",
            post(wire_session_summarize),
        )
        .route(
            "/session/{sessionID}/message",
            post(wire_message_send).get(wire_messages_page),
        )
        .route(
            "/session/{sessionID}/message/{messageID}",
            delete(wire_message_delete),
        )
        .route("/session/{sessionID}/abort", post(wire_abort))
        .route("/session/{sessionID}/diff", get(wire_diff))
        .route("/session/{sessionID}/revert", post(wire_revert))
        .route("/session/{sessionID}/unrevert", post(wire_unrevert))
        .route("/session/{sessionID}/state", get(wire_session_state))
        .route("/session/{sessionID}/status", get(wire_session_state))
        .route("/session/status", get(wire_session_status_query))
        .route("/question/reject", post(question_reject))
        .route("/network/reject", post(network_reject))
        .route("/config/update", post(config_update))
        .route("/config/warnings", get(config_warnings))
        .route("/config/overlay", post(config_overlay))
        .route("/config/overlayUpdate", post(config_overlay_update))
        .route("/pty/create", post(pty_create))
        .route("/pty/update", post(pty_update))
        .route("/pty/remove", post(pty_remove))
        .route("/pty/{pty_id}/output", get(pty_output))
        .route("/global/dispose", post(dispose_all_sessions))
        .route("/instance/dispose", post(dispose_all_sessions))
        .route("/instance/reload", post(instance_reload))
        .route("/auth/set", post(auth_set))
        .route("/auth/remove", post(auth_remove))
        // Faktor Native Protocol v1 (docs/native-protocol.md): the daemon's
        // OWN surface, optimized around this runtime. UI compatibility is
        // the target — these handlers speak native JSON, never the v7.5.6
        // wire DTOs.
        .route("/session/{id}/projection", get(native_session_projection))
        .route("/models", get(native_models))
        .route("/capabilities", get(native_capabilities))
        // Native Protocol v1, audit 55-56 wiring: liveness/readiness plus
        // the durable session listings under an explicit /native prefix.
        // Every handler is auth-gated; request bodies parse with the strict
        // native DTOs (deny_unknown_fields — a typo is a 400).
        .route("/native/health", get(native_health))
        .route("/native/ready", get(native_ready))
        .route("/native/usage", get(native_usage))
        .route("/native/session/{id}/turns", get(native_session_turns))
        .route("/native/session/{id}/tasks", get(native_session_tasks))
        .route(
            "/native/session/{id}/checkpoints",
            get(native_session_checkpoints),
        )
        .route(
            "/native/session/{id}/verification",
            get(native_session_verification),
        )
        .route("/native/session/{id}/agents", get(native_session_agents))
        // Presentation/attention continuity (additive): record one durable
        // foreground/background transition of a child of this session. It
        // never changes scheduling ownership, budgets or lineage.
        .route(
            "/native/session/{id}/agents/{child}/presentation",
            post(native_agent_presentation),
        )
        .route(
            "/native/session/{id}/terminal",
            get(native_session_terminal),
        )
        .route("/native/session/{id}/abort", post(native_session_abort))
        .route("/native/orchestrator/graph", get(native_orchestrator_graph))
        // Native agent state + control (audits P0-20/21/23/61): the real
        // child agents of the session's task runs (GET), and first-class
        // pause/resume/cancel/retry/steer/model/budget over the runtime's
        // durable child_commands queue (exactly-once applied semantics).
        .route("/native/agents", get(native_agents))
        .route("/native/agents/{child_id}/pause", post(native_agent_pause))
        .route(
            "/native/agents/{child_id}/resume",
            post(native_agent_resume),
        )
        .route(
            "/native/agents/{child_id}/cancel",
            post(native_agent_cancel),
        )
        .route("/native/agents/{child_id}/retry", post(native_agent_retry))
        .route("/native/agents/{child_id}/steer", post(native_agent_steer))
        .route("/native/agents/{child_id}/model", post(native_agent_model))
        .route(
            "/native/agents/{child_id}/budget",
            post(native_agent_budget),
        )
        // Audit P0-62/63/64 native surface (additive; strict DTOs): the
        // session-owned terminal projection (scoped listing + owned spawn +
        // bounded lifetime-event log), the durable cursor surfaces
        // (messages / journal events), the provider registry view, the
        // per-session authoritative usage read and the durable
        // verification-evidence read of one task.
        .route("/native/terminals", get(native_terminals))
        .route(
            "/native/session/{id}/terminal/events",
            get(native_terminal_events),
        )
        .route("/native/session/{id}/terminal", post(native_terminal_spawn))
        .route("/native/messages", get(native_messages))
        .route("/native/events", get(native_events))
        .route("/native/providers", get(native_providers))
        .route("/native/session/{id}/usage", get(native_session_usage))
        .route(
            "/native/session/{id}/tasks/{task_id}/verification",
            get(native_task_verification),
        )
        // Native task runs (wave-24): the ONE HTTP surface that starts a
        // task through the daemon's TaskExecutor (POST), lists the
        // session's durable task runs with per-run state (GET), reads one
        // run's state, and cancels one run at the TASK level. Strict DTOs;
        // hostile bodies are 400s; no second start architecture exists.
        .route(
            "/native/session/{id}/task-runs",
            get(native_task_runs).post(native_task_run_start),
        )
        .route(
            "/native/session/{id}/task-runs/{run_id}",
            get(native_task_run_state),
        )
        .route(
            "/native/session/{id}/task-runs/{run_id}/cancel",
            post(native_task_run_cancel),
        )
        // Multi-candidate implementation tournaments (additive): start an
        // N = 2..=4 candidate tournament through the executor's ONE entry
        // (identical goal+criteria, isolated candidate worktrees) and read
        // its durable state. Strict DTOs; hostile bodies are 400s.
        .route(
            "/native/session/{id}/tournament",
            post(native_tournament_start),
        )
        .route(
            "/native/session/{id}/tournament/{tournament_id}",
            get(native_tournament_state),
        )
        // Additive durable listing of the session's tournaments (the
        // summary projection of the pinned ledger lifecycle rows).
        .route(
            "/native/session/{id}/tournaments",
            get(native_tournaments_list),
        )
        // Additive tournament control (strict DTOs): decide runs the
        // deterministic comparison (typed 404 unknown / 409 non-open or no
        // eligible winner) and abort discards every candidate with the
        // reason on the terminal durable row (typed 404/409).
        .route(
            "/native/session/{id}/tournaments/{tournament_id}/decide",
            post(native_tournament_decide),
        )
        .route(
            "/native/session/{id}/tournaments/{tournament_id}/abort",
            post(native_tournament_abort),
        )
        .route("/native/evidence/{id}", get(native_evidence_get))
        .route(
            "/native/evidence/{id}/retrieve",
            post(native_evidence_retrieve),
        )
        .route("/native/semantic/status", get(native_semantic_status))
        .route(
            "/native/semantic/capabilities",
            get(native_semantic_capabilities),
        )
        .layer(RequestBodyLimitLayer::new(MAX_BODY_BYTES))
        .with_state(AppState {
            deps: Arc::new(deps),
            bus,
            config: Arc::new(std::sync::RwLock::new(serde_json::Value::Object(
                Default::default(),
            ))),
            auth: Arc::new(std::sync::RwLock::new(None)),
            ptys: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            next_pty_id: Arc::new(std::sync::atomic::AtomicU64::new(1)),
            terminal_owners: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            terminal_events: Arc::new(std::sync::Mutex::new(std::collections::VecDeque::new())),
            next_terminal_event_id: Arc::new(std::sync::atomic::AtomicU64::new(1)),
            ready: ready.clone(),
        });
    if set_ready {
        ready.store(true, std::sync::atomic::Ordering::SeqCst);
    }
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = shutdown_rx.await;
            })
            .await
            .ok();
    });
    Ok(ServerHandle {
        addr,
        shutdown: shutdown_tx,
        handshake,
        startup_line,
    })
}

// ------------------------------------------------------------------ state

#[derive(Clone)]
pub(crate) struct AppState {
    pub(crate) deps: Arc<ServerDeps>,
    pub(crate) bus: Arc<GlobalEventBus>,
    pub(crate) config: Arc<std::sync::RwLock<serde_json::Value>>,
    /// Runtime server-password override (`auth.set`); `None` = the startup
    /// env password (`ServerDeps.server_password`) applies (`auth.remove`).
    pub(crate) auth: Arc<std::sync::RwLock<Option<ServerPassword>>>,
    /// Live PTYs (audit round 11): session-owned interactive terminals,
    /// Unix real implementation; other platforms refuse at creation.
    pub(crate) ptys: Arc<std::sync::Mutex<std::collections::HashMap<u64, faktor_pty::Pty>>>,
    pub(crate) next_pty_id: Arc<std::sync::atomic::AtomicU64>,
    /// Native ownership of registered PTYs (audit P0-62): one entry per
    /// `ptys` key that a SESSION-owned spawn registered. Entries absent from
    /// this map are unowned legacy rows (the daemon-level `/pty/create`
    /// surface predates ownership); a session-scoped native view never
    /// projects them. Additive: nothing here changes the daemon-wide maps.
    pub(crate) terminal_owners: Arc<
        std::sync::Mutex<
            std::collections::HashMap<u64, crate::native::terminal::NativeTerminalOwnership>,
        >,
    >,
    /// Bounded per-daemon log of session-owned terminal lifetime events
    /// (audit P0-62): `created` at spawn, `exited` when the swept process
    /// dies. Ring-bounded; ids ascend from 1.
    pub(crate) terminal_events:
        Arc<std::sync::Mutex<std::collections::VecDeque<(u64, serde_json::Value)>>>,
    pub(crate) next_terminal_event_id: Arc<std::sync::atomic::AtomicU64>,
    /// Readiness flag (audit 55): set true at the END of serve() setup, after
    /// the store was opened/migrated/recovered by the caller (cli runs
    /// `agent.recover()` before serve) and every required runtime component
    /// is in place (`SessionManager` is a non-optional `Arc` in
    /// `ServerDeps`, so its presence is structural). `GET /native/ready`
    /// answers 200 `{ready:true}` once it is set; 503 `{ready:false}`
    /// before/without it (`ServerDeps.simulate_not_ready`).
    pub(crate) ready: Arc<std::sync::atomic::AtomicBool>,
}

// ------------------------------------------------------------------ handlers

#[cfg(test)]
mod tests {
    use super::*;
    use faktor_core::capability::PermissionDecision;
    use faktor_core::id::{SessionId, WorkspaceId};
    use faktor_core::model::ModelCapabilities;
    use faktor_evidence::store::EvidenceStore as _;
    use faktor_protocol::v756::*;
    use faktor_provider::FakeProvider;
    use faktor_session::BudgetAuthority;
    use std::time::Duration;

    /// The runtime + executor pair every test `ServerDeps` carries (the
    /// real orchestrator over the test store — the endpoints under test
    /// drive exactly what production drives).
    fn orch_pair(
        session: Arc<SessionManager>,
        agent: Arc<AgentRuntime>,
    ) -> (
        Arc<faktor_orchestrator::runtime::OrchestratorRuntime>,
        Arc<faktor_orchestrator::runtime::task_executor::TaskExecutor>,
    ) {
        let orchestrator =
            faktor_orchestrator::runtime::OrchestratorRuntime::new(session.clone(), agent.clone());
        let tasks = faktor_orchestrator::runtime::task_executor::TaskExecutor::new(
            &orchestrator,
            session,
            agent,
            None,
        );
        (orchestrator, tasks)
    }

    #[test]
    fn handshake_line_is_frozen_shape() {
        let session = SessionManager::open(
            std::env::temp_dir().join("kp-hs-store"),
            std::env::temp_dir().join("kp-hs-cas"),
            false,
        )
        .unwrap();
        let agent = AgentRuntime::new(faktor_agent::AgentDeps {
            session: SessionManager::open(
                std::env::temp_dir().join("kp-hs-store2"),
                std::env::temp_dir().join("kp-hs-cas2"),
                false,
            )
            .unwrap(),
            providers: Arc::new(faktor_provider::ProviderRegistry::new()),
            chunk_sink: None,
            permission_requester: ChannelPermissionRequester::new(Duration::from_secs(1)),
            evidence: Arc::new(faktor_agent::NoEvidence),
            tools: Arc::new(faktor_agent::ToolRegistry::new()),
            cas: None,
            workspaces: faktor_fs::WorkspaceFileService::new(),
            edit: None,
            snapshots: None,
            sandbox: None,
            supervisor: None,
            verification: faktor_agent::VerificationService::disabled(),
            model: "m".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "i".into(),
            hooks: None,
            instructions_resolver: faktor_instructions::no_roots_resolver(),
            routing: faktor_agent::FixedRoutingPolicy::passthrough(),
            budgets: Arc::new(faktor_session::NoopBudget),
            clock: Arc::new(faktor_core::time::SystemClock),
            tool_call_mode: faktor_agent::ToolCallMode::Native,
            tool_deadline_ms: 1000,
            retry_policy: faktor_core::retry::RetryPolicy::default(),
            semantic: faktor_agent::fallback_semantic_registry(),
            context_prior: None,
            efficiency: Default::default(),
        })
        .unwrap();
        let permissions = ChannelPermissionRequester::new(Duration::from_secs(1));
        let (orchestrator, tasks) = orch_pair(session.clone(), agent.clone());
        let deps = ServerDeps {
            budgets: faktor_session::DurableBudgetLedger::new(session.clone()),
            session,
            agent,
            permissions,
            orchestrator,
            tasks,
            auth_token: AuthToken::generate(),
            server_password: ServerPassword::generate(),
            directory: None,
            version: "0.1.0".into(),
            fs: None,
            snapshots: None,
            chunk_rx: None,
            simulate_not_ready: false,
            evidence: None,
            semantic: None,
        };
        let addr: SocketAddr = "127.0.0.1:45678".parse().unwrap();
        let line = deps.handshake_line(addr);
        assert!(line.starts_with("FAKTOR_PLUS_HANDSHAKE "));
        let hs = Handshake::from_line(&line).unwrap();
        assert_eq!(hs.protocol, "v756");
        assert_eq!(hs.port, 45678);
        assert_eq!(hs.auth_token, deps.auth_token.as_str());
        // The frozen stdout contract is the startup line, and the password
        // never appears in it (no token on stdout).
        let startup = deps.startup_line(addr);
        assert_eq!(startup, "faktor server listening on http://127.0.0.1:45678");
        assert!(!startup.contains(&deps.server_password.as_str()[..8]));
    }

    #[tokio::test]
    async fn unauthorized_requests_rejected_before_handlers() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        // No token.
        let resp = client
            .post(format!("http://{}/api/session", handle.addr))
            .json(&serde_json::json!({"provider": "fake", "model": "m"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);
        // Wrong token.
        let resp = client
            .post(format!("http://{}/api/session", handle.addr))
            .bearer_auth("wrong")
            .json(&serde_json::json!({"provider": "fake", "model": "m"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn hello_is_public_and_correct() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let resp = client
            .get(format!("http://{}/api/hello", handle.addr))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["protocol"], "v756");
        assert_eq!(body["auth_required"], true);
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn full_flow_create_prompt_messages_state() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let token = deps.auth_token.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        // Create session.
        let resp = client
            .post(format!("{base}/api/session"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({
                "provider": "fake",
                "model": "m",
                "workspace": "/tmp",
                "title": "t1",
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid = created["id"].as_str().unwrap().to_string();

        // Prompt.
        let resp = client
            .post(format!("{base}/api/session/{sid}/prompt"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"prompt": "hi", "files": []}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let pr: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(pr["accepted"], true);

        // State reflects the turn (may still be running — poll until ready).
        // Deadline-based, host-speed independent: 100 fixed 20 ms polls was
        // a wall-clock assumption the slower Windows runner exhausted while
        // the drive was still `preparing`. The terminal set and the
        // assertion are unchanged, so a genuinely stuck turn still fails
        // here (with the observed state in the panic).
        let deadline = tokio::time::Instant::now() + Duration::from_secs(90);
        let state = loop {
            let resp = client
                .get(format!("{base}/api/session/{sid}/state"))
                .bearer_auth(token.as_str())
                .send()
                .await
                .unwrap();
            let body: serde_json::Value = resp.json().await.unwrap();
            let state = body["agent_state"]["state"]
                .as_str()
                .unwrap_or("")
                .to_string();
            if matches!(
                state.as_str(),
                "ready_for_next_turn" | "completed" | "cancelled"
            ) {
                break state;
            }
            if tokio::time::Instant::now() >= deadline {
                break state;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        };
        assert_eq!(state, "ready_for_next_turn", "turn must complete");

        // Messages contain the exchange.
        let resp = client
            .get(format!("{base}/api/session/{sid}/messages?limit=10"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        let page: serde_json::Value = resp.json().await.unwrap();
        assert!(page["messages"].as_array().unwrap().len() >= 2, "{page}");

        // Malformed body → 400; unknown route → 404; unknown session → 404.
        let resp = client
            .post(format!("{base}/api/session/{sid}/prompt"))
            .bearer_auth(token.as_str())
            .body("{not json")
            .header("content-type", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);
        let resp = client
            .get(format!("{base}/api/nope"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);
        let resp = client
            .get(format!("{base}/api/session/999999/state"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);

        let _ = handle.shutdown.send(());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sse_streams_and_resumes_from_cursor() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let token = deps.auth_token.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        let resp = client
            .post(format!("{base}/api/session"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"provider": "fake", "model": "m"}))
            .send()
            .await
            .unwrap();
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid = created["id"].as_str().unwrap().to_string();

        // Subscribe before the prompt so we see the whole sequence.
        let mut sse = client
            .get(format!("{base}/api/session/{sid}/events?events_after=0"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap()
            .bytes_stream();

        client
            .post(format!("{base}/api/session/{sid}/prompt"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"prompt": "hi"}))
            .send()
            .await
            .unwrap();

        use futures_util::StreamExt;
        let mut saw_state = false;
        let mut text = String::new();
        for _ in 0..200 {
            match tokio::time::timeout(Duration::from_millis(200), sse.next()).await {
                Ok(Some(Ok(chunk))) => {
                    text.push_str(&String::from_utf8_lossy(&chunk));
                    if text.contains("agent_state_changed") {
                        saw_state = true;
                    }
                    if saw_state {
                        break;
                    }
                }
                Ok(Some(Err(_))) => break,
                Ok(None) | Err(_) => break,
            }
        }
        assert!(saw_state, "SSE must deliver state events; got: {text}");

        // Resume from a cursor: events_after=1 skips the SessionCreated frame.
        let resp = client
            .get(format!("{base}/api/session/{sid}/events?events_after=1"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn permission_flow_blocks_until_resolved() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let token = deps.auth_token.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        let resp = client
            .post(format!("{base}/api/session"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"provider": "fake", "model": "m"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

        // The fake provider script makes a tool call, so the turn blocks on
        // permission. Resolve it through the frozen API.
        let mut registry = faktor_provider::ProviderRegistry::new();
        registry
            .try_register(Arc::new(FakeProvider::with_script(
                "fake",
                ModelCapabilities {
                    tools: true,
                    ..Default::default()
                },
                vec![
                    faktor_provider::ScriptedResponse::ToolCall {
                        id: "c1".into(),
                        name: "echo".into(),
                        input: serde_json::json!({"x": 1}),
                    },
                    faktor_provider::ScriptedResponse::End,
                ],
            )))
            .unwrap();
        let session =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let permissions = ChannelPermissionRequester::new(Duration::from_secs(5));
        let mut tools = faktor_agent::ToolRegistry::new();
        tools.register(faktor_agent::Tool {
            name: "echo".into(),
            description: "d".into(),
            input_schema: serde_json::json!({}),
            resource_class: faktor_core::resource::ResourceClass::Cpu,
            capability: None,
            recovery_hint: faktor_agent::RecoveryHint::Idempotent,
            path_args: vec![],
            execute: Arc::new(|_ctx, _args| {
                Box::pin(async move { Ok(faktor_agent::ToolOutcome::default()) })
            }),
        });
        let agent = AgentRuntime::new(faktor_agent::AgentDeps {
            session: session.clone(),
            providers: Arc::new(registry),
            chunk_sink: None,
            permission_requester: permissions.clone(),
            evidence: Arc::new(faktor_agent::NoEvidence),
            tools: Arc::new(tools),
            cas: None,
            workspaces: faktor_fs::WorkspaceFileService::new(),
            edit: None,
            snapshots: None,
            sandbox: None,
            supervisor: None,
            verification: faktor_agent::VerificationService::disabled(),
            model: "m".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "You are a test server agent.".into(),
            hooks: None,
            instructions_resolver: faktor_instructions::no_roots_resolver(),
            routing: faktor_agent::FixedRoutingPolicy::passthrough(),
            budgets: Arc::new(faktor_session::NoopBudget),
            clock: Arc::new(faktor_core::time::SystemClock),
            tool_call_mode: faktor_agent::ToolCallMode::Native,
            tool_deadline_ms: 2000,
            retry_policy: faktor_core::retry::RetryPolicy::default(),
            semantic: faktor_agent::fallback_semantic_registry(),
            context_prior: None,
            efficiency: Default::default(),
        })
        .unwrap();
        // Replace the running server's deps by serving a second one on the
        // same store (the first server's fake provider has no tool call, so
        // the permission test needs its own instance).
        let (orchestrator, tasks) = orch_pair(session.clone(), agent.clone());
        let deps2 = ServerDeps {
            budgets: faktor_session::DurableBudgetLedger::new(session.clone()),
            session: session.clone(),
            agent,
            permissions: permissions.clone(),
            orchestrator,
            tasks,
            auth_token: token.clone(),
            server_password: ServerPassword::generate(),
            directory: None,
            version: "0.1.0".into(),
            fs: None,
            snapshots: None,
            chunk_rx: None,
            simulate_not_ready: false,
            evidence: None,
            semantic: None,
        };
        let handle2 = serve(deps2, 0).await.unwrap();
        let base2 = format!("http://{}", handle2.addr);
        let resp = client
            .post(format!("{base2}/api/session"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"provider": "fake", "model": "m"}))
            .send()
            .await
            .unwrap();
        let created2: serde_json::Value = resp.json().await.unwrap();
        let sid2 = created2["id"].as_str().unwrap().to_string();
        client
            .post(format!("{base2}/api/session/{sid2}/prompt"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"prompt": "use tools"}))
            .send()
            .await
            .unwrap();

        // The turn blocks on permission; resolve through the API.
        let mut resolved = false;
        for _ in 0..100 {
            if let Some(pid) = permissions.pending_ids().first().copied() {
                let resp = client
                    .post(format!("{base2}/api/perm/{pid}/resolve"))
                    .bearer_auth(token.as_str())
                    .json(&serde_json::json!({
                        "permission_id": pid.to_string(),
                        "decision": "allow",
                    }))
                    .send()
                    .await
                    .unwrap();
                assert_eq!(resp.status(), 200);
                resolved = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(resolved, "permission must surface and resolve");

        // The turn must now complete.
        let mut done = false;
        for _ in 0..100 {
            let id = parse_session_id(&sid2).unwrap();
            let state = session.get_session(id).unwrap().unwrap().state().unwrap();
            if matches!(state, faktor_core::state::AgentState::ReadyForNextTurn) {
                done = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(done, "turn must finish after permission grant");

        // Double resolve → conflict.
        let resp = client
            .post(format!("{base2}/api/perm/1/resolve"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"permission_id": "1", "decision": "allow"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 409);
        let _ = handle.shutdown.send(());
        let _ = handle2.shutdown.send(());
    }

    #[tokio::test]
    async fn sdk_routes_require_password_and_health_requires_basic() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let pw = deps.server_password.clone();
        let token = deps.auth_token.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        // /global/health requires auth now (the frozen client authenticates
        // every request, this one included). Basic is accepted.
        let resp = client
            .get(format!("{base}/global/health"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);
        let resp = client
            .get(format!("{base}/global/health"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["ok"], true);
        assert_eq!(body["protocol"], "v756");
        assert!(body["version"].is_string());
        // Wrong Basic credentials are rejected.
        let resp = client
            .get(format!("{base}/global/health"))
            .basic_auth("kilo", Some("wrong"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);

        // Every other endpoint requires the password. Bodies are valid (the
        // auth gate runs inside the handler, after extraction) so the 401 is
        // the auth gate, not a parse error.
        let cases: &[(&str, &str, serde_json::Value)] = &[
            (
                "post",
                "/session/create",
                serde_json::json!({"provider": "fake", "model": "m"}),
            ),
            (
                "post",
                "/session/prompt",
                serde_json::json!({"session_id": "1", "prompt": "x"}),
            ),
            (
                "post",
                "/session/abort",
                serde_json::json!({"session_id": "1"}),
            ),
            (
                "get",
                "/session/messages?session_id=1",
                serde_json::json!({}),
            ),
            ("get", "/session/state?session_id=1", serde_json::json!({})),
            ("get", "/session/list", serde_json::json!({})),
            ("get", "/global/health", serde_json::json!({})),
            ("get", "/session", serde_json::json!({})),
            (
                "post",
                "/session",
                serde_json::json!({"model": {"id": "m", "providerID": "fake"}}),
            ),
            ("get", "/session/1", serde_json::json!({})),
            (
                "post",
                "/session/1/message",
                serde_json::json!({"model": {"providerID": "fake", "modelID": "m"},
                    "parts": [{"type": "text", "text": "hi"}]}),
            ),
            ("get", "/session/1/message?limit=1", serde_json::json!({})),
            ("post", "/session/1/abort", serde_json::json!({})),
            ("get", "/session/1/diff", serde_json::json!({})),
            (
                "post",
                "/session/1/revert",
                serde_json::json!({"messageID": "1"}),
            ),
            (
                "post",
                "/session/1/unrevert",
                serde_json::json!({"messageID": "1"}),
            ),
            (
                "post",
                "/permission/reply",
                serde_json::json!({"permission_id": "1", "decision": "allow"}),
            ),
            ("get", "/permission/list", serde_json::json!({})),
            ("get", "/provider/list", serde_json::json!({})),
            ("get", "/global/event?after=0", serde_json::json!({})),
            (
                "post",
                "/question/reply",
                serde_json::json!({"question_id": "q", "decision": "d"}),
            ),
            ("get", "/question/list", serde_json::json!({})),
            (
                "post",
                "/network/reply",
                serde_json::json!({"network_id": "n", "decision": "d"}),
            ),
            ("get", "/network/list", serde_json::json!({})),
            ("get", "/config/get", serde_json::json!({})),
            ("post", "/config/set", serde_json::json!({"config": {}})),
            ("get", "/session/status?session_id=1", serde_json::json!({})),
            ("get", "/session/1/status", serde_json::json!({})),
            ("post", "/session/1/fork", serde_json::json!({})),
            ("post", "/session/1/summarize", serde_json::json!({})),
            ("delete", "/session/1", serde_json::json!({})),
            ("delete", "/session/1/message/1", serde_json::json!({})),
            (
                "post",
                "/question/reject",
                serde_json::json!({"question_id": "1"}),
            ),
            (
                "post",
                "/network/reject",
                serde_json::json!({"network_id": "1"}),
            ),
            (
                "post",
                "/config/update",
                serde_json::json!({"config": {"model": "m"}}),
            ),
            ("get", "/config/warnings", serde_json::json!({})),
            ("post", "/config/overlay", serde_json::json!({"config": {}})),
            (
                "post",
                "/config/overlayUpdate",
                serde_json::json!({"config": {}}),
            ),
            ("post", "/pty/create", serde_json::json!({})),
            ("post", "/pty/update", serde_json::json!({})),
            ("post", "/pty/remove", serde_json::json!({})),
            ("post", "/global/dispose", serde_json::json!({})),
            ("post", "/instance/dispose", serde_json::json!({})),
            ("post", "/instance/reload", serde_json::json!({})),
            ("post", "/auth/set", serde_json::json!({"password": null})),
            ("post", "/auth/remove", serde_json::json!({})),
        ];
        for (method, path, body) in cases {
            let resp = if *method == "get" {
                client.get(format!("{base}{path}")).send().await.unwrap()
            } else if *method == "delete" {
                client.delete(format!("{base}{path}")).send().await.unwrap()
            } else {
                client
                    .post(format!("{base}{path}"))
                    .json(body)
                    .send()
                    .await
                    .unwrap()
            };
            assert_eq!(
                resp.status(),
                401,
                "{method} {path} without password must be 401"
            );
            let body: serde_json::Value = resp.json().await.unwrap();
            assert_eq!(body["error"]["code"], "unauthorized");
        }

        // Wrong password is rejected in both header forms.
        let resp = client
            .post(format!("{base}/session/create"))
            .bearer_auth("wrong-password")
            .json(&serde_json::json!({"provider": "fake", "model": "m"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);
        let resp = client
            .post(format!("{base}/session/create"))
            .header("x-faktor-server-password", "wrong-password")
            .json(&serde_json::json!({"provider": "fake", "model": "m"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);

        // The password works in all three header forms.
        let resp = client
            .post(format!("{base}/session/create"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"provider": "fake", "model": "m"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let resp = client
            .post(format!("{base}/session/create"))
            .bearer_auth(pw.as_str())
            .json(&serde_json::json!({"provider": "fake", "model": "m"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let resp = client
            .post(format!("{base}/session/create"))
            .basic_auth("kilo", Some(pw.as_str()))
            .json(&serde_json::json!({"provider": "fake", "model": "m"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        // A wrong Basic username is rejected.
        let resp = client
            .post(format!("{base}/session/create"))
            .basic_auth("admin", Some(pw.as_str()))
            .json(&serde_json::json!({"provider": "fake", "model": "m"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);
        // The legacy per-start bearer token still works.
        let resp = client
            .post(format!("{base}/session/create"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"provider": "fake", "model": "m"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn wire_surface_full_flow_with_basic_auth() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let pw = deps.server_password.clone();
        let session = deps.session.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let basic = |r: reqwest::RequestBuilder| r.basic_auth("kilo", Some(pw.as_str()));

        // POST /session: the x-faktor-directory header wins over workspaceID,
        // and the model.providerID drives the provider.
        let resp = basic(
            client
                .post(format!("{base}/session"))
                .header("x-faktor-directory", "/tmp")
                .json(&serde_json::json!({
                    "parentID": null,
                    "title": "wire t1",
                    "agent": "default",
                    "model": {"id": "m", "providerID": "fake", "variant": null},
                    "metadata": {"origin": "audit-round-2"},
                    "permission": null,
                    "platform": "darwin",
                    "workspaceID": "/ignored",
                    "sandboxInheritanceToken": null,
                })),
        )
        .send()
        .await
        .unwrap();
        assert_eq!(resp.status(), 200, "create must succeed");
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid = created["sessionID"].as_str().unwrap().to_string();
        assert_eq!(created["title"], "wire t1");
        assert!(created["createdMs"].as_i64().unwrap() > 0);
        // The created session row carries the header workspace, not
        // workspaceID (the header wins by contract).
        let sid_parsed = parse_session_id(&sid).unwrap();
        let row = session
            .get_session(sid_parsed)
            .unwrap()
            .unwrap()
            .row()
            .unwrap();
        let ws = session.create_workspace("/tmp").unwrap();
        assert_eq!(row.workspace_id, ws, "header workspace must win");
        assert_eq!(row.provider, "fake");
        assert_eq!(row.model, "m");

        // GET /session lists it.
        let resp = basic(client.get(format!("{base}/session")))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let list: serde_json::Value = resp.json().await.unwrap();
        let ids: Vec<&str> = list["sessions"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|s| s["sessionID"].as_str())
            .collect();
        assert!(ids.contains(&sid.as_str()));
        let summary = &list["sessions"]
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["sessionID"].as_str() == Some(sid.as_str()))
            .unwrap();
        assert!(summary["createdMs"].as_i64().unwrap() > 0);
        assert!(summary["updatedMs"].as_i64().unwrap() > 0);
        assert!(summary["state"].is_string());

        // GET /session/{sessionID} summary.
        let resp = basic(client.get(format!("{base}/session/{sid}")))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["sessionID"], sid);
        assert_eq!(body["title"], "wire t1");

        // POST /session/{sessionID}/message with a full parts[] payload.
        let resp = basic(client.post(format!("{base}/session/{sid}/message")).json(
            &serde_json::json!({
                "messageID": null,
                "model": {"providerID": "fake", "modelID": "m"},
                "agent": null,
                "noReply": false,
                "tools": ["read_file"],
                "format": null,
                "system": null,
                "variant": null,
                "snapshotInitialization": false,
                "editorContext": {"file": "a.rs"},
                "parts": [
                    {"type": "text", "text": "fix it"},
                    {"type": "file", "path": "b.rs", "content": "fn b() {}", "mode": "edit"},
                    {"type": "tool", "callID": "c1", "name": "read_file",
                     "input": {"path": "a.rs"}, "state": "running", "output": null}
                ]
            }),
        ))
        .send()
        .await
        .unwrap();
        assert_eq!(resp.status(), 200);
        // Frozen send shape: {info: AssistantMessage, parts: Part[]} — the
        // info is the durable assistant message of the accepted turn, the
        // parts its wire parts (top level has exactly info+parts; the old
        // {messageID, accepted, queued} envelope is gone).
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(
            body.as_object().unwrap().keys().collect::<Vec<_>>(),
            vec!["info", "parts"]
        );
        assert_eq!(body["info"]["sessionID"], sid);
        assert_eq!(body["info"]["role"], "assistant");
        let assistant_seq: i64 = body["info"]["messageID"].as_str().unwrap().parse().unwrap();
        assert!(assistant_seq > 1, "assistant lands after the user prompt");
        assert!(body["info"]["createdMs"].as_i64().unwrap() > 0);
        assert_eq!(body["info"]["providerID"], "fake");
        assert_eq!(body["info"]["modelID"], "m");
        let send_parts = body["parts"].as_array().unwrap();
        assert!(!send_parts.is_empty(), "{body}");
        assert!(
            send_parts
                .iter()
                .any(|p| p["type"] == "text" && p["text"] == "pong"),
            "the fake provider's reply rides the parts: {body}"
        );

        // The wire messages page is the frozen array of {info, parts}.
        let resp = basic(client.get(format!("{base}/session/{sid}/message?limit=10")))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(resp.headers().get("x-has-more").unwrap(), "false");
        let page: serde_json::Value = resp.json().await.unwrap();
        let messages = page.as_array().unwrap();
        assert!(messages.len() >= 2, "{page}");
        // Newest first; entries are {info, parts} with wire field names.
        let first = &messages[0];
        assert_eq!(
            first.as_object().unwrap().keys().collect::<Vec<_>>(),
            vec!["info", "parts"]
        );
        assert!(first["info"]["messageID"]
            .as_str()
            .unwrap()
            .parse::<u64>()
            .is_ok());
        assert!(first["info"]["createdMs"].as_i64().unwrap() > 0);
        assert_eq!(first["info"]["providerID"], "fake");
        assert_eq!(first["info"]["modelID"], "m");
        // The assistant reply text survives as a wire text part.
        let text = first["parts"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["type"] == "text")
            .map(|p| p["text"].as_str().unwrap_or(""))
            .unwrap_or("");
        assert_eq!(text, "pong");
        // The PROMPT message itself appears with its text part (user rows
        // are projected from their stored text).
        let prompt = messages
            .iter()
            .find(|m| m["info"]["role"] == "user")
            .expect("the user prompt message must be on the page");
        assert_eq!(prompt["info"]["messageID"], "2", "first user seq is 2");
        let prompt_text = prompt["parts"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["type"] == "text")
            .map(|p| p["text"].as_str().unwrap_or(""))
            .unwrap_or("");
        // The stored prompt is the mapper's text+file concatenation.
        assert!(
            prompt_text.contains("fix it"),
            "prompt text must appear: {prompt_text:?}"
        );
        assert!(
            prompt_text.contains("fn b() {}"),
            "file content rides the prompt: {prompt_text:?}"
        );
        // Paging: before=1 (nothing older than seq 1) is an empty page with
        // x-has-more false; unknown cursors are the server's clamp.
        let resp = basic(client.get(format!("{base}/session/{sid}/message?before=1&limit=1")))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(resp.headers().get("x-has-more").unwrap(), "false");
        assert_eq!(
            resp.json::<serde_json::Value>().await.unwrap(),
            serde_json::json!([])
        );

        // POST abort with the frozen body shape.
        let resp = basic(
            client
                .post(format!("{base}/session/{sid}/abort"))
                .json(&serde_json::json!({"messageID": null})),
        )
        .send()
        .await
        .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(body["aborted"].is_array());

        // Adversarial: empty parts → 400; unknown body fields → 422; empty
        // body message → 400; unknown session → 404; non-numeric id → 400.
        let resp = basic(client.post(format!("{base}/session/{sid}/message")).json(
            &serde_json::json!({
                "model": {"providerID": "fake", "modelID": "m"},
                "parts": []
            }),
        ))
        .send()
        .await
        .unwrap();
        assert_eq!(resp.status(), 400);
        let resp = basic(client.post(format!("{base}/session/{sid}/message")).json(
            &serde_json::json!({
                "model": {"providerID": "fake", "modelID": "m"},
                "parts": [{"type": "text", "text": "x"}],
                "smuggled": true
            }),
        ))
        .send()
        .await
        .unwrap();
        assert_eq!(resp.status(), 422);
        // Only control-plane parts → the mapped prompt is empty → 400.
        let resp = basic(client.post(format!("{base}/session/{sid}/message")).json(
            &serde_json::json!({
                "model": {"providerID": "fake", "modelID": "m"},
                "parts": [{"type": "reasoning", "text": "think"}]
            }),
        ))
        .send()
        .await
        .unwrap();
        assert_eq!(resp.status(), 400);
        // Unknown wire part kind → 422 (deny_unknown_fields on the union).
        let resp = basic(client.post(format!("{base}/session/{sid}/message")).json(
            &serde_json::json!({
                "model": {"providerID": "fake", "modelID": "m"},
                "parts": [{"type": "escape_hatch", "text": "x"}]
            }),
        ))
        .send()
        .await
        .unwrap();
        assert_eq!(resp.status(), 422);
        for path in [
            "/session/999999",
            "/session/999999/message",
            "/session/999999/abort",
            "/session/999999/diff",
            "/session/999999/revert",
            "/session/999999/unrevert",
        ] {
            let resp = if path.ends_with("/revert") {
                basic(
                    client
                        .post(format!("{base}{path}"))
                        .json(&serde_json::json!({"messageID": "1"})),
                )
                .send()
                .await
                .unwrap()
            } else if path.ends_with("/message") {
                basic(
                    client
                        .post(format!("{base}{path}"))
                        .json(&serde_json::json!({
                            "model": {"providerID": "fake", "modelID": "m"},
                            "parts": [{"type": "text", "text": "x"}]
                        })),
                )
                .send()
                .await
                .unwrap()
            } else if path.ends_with("/abort") {
                basic(
                    client
                        .post(format!("{base}{path}"))
                        .json(&serde_json::json!({})),
                )
                .send()
                .await
                .unwrap()
            } else if path.ends_with("/unrevert") {
                // unrevert shares revert's strict body contract.
                basic(
                    client
                        .post(format!("{base}{path}"))
                        .json(&serde_json::json!({"messageID": "1"})),
                )
                .send()
                .await
                .unwrap()
            } else {
                basic(client.get(format!("{base}{path}")))
                    .send()
                    .await
                    .unwrap()
            };
            assert_eq!(resp.status(), 404, "{path} must 404");
        }
        let resp = basic(client.get(format!("{base}/session/not-a-number")))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);
        let resp = basic(client.get(format!("{base}/session/0")))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);
        // /session/{sessionID} GET with an unknown session → 404; with the
        // known one it already worked above.
        let resp = basic(client.get(format!("{base}/session/999999")))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);

        let _ = handle.shutdown.send(());
    }

    /// A wire-testing daemon whose provider records the model of every
    /// request streamed through it (asserts the per-message override
    /// actually reaches the agent).
    fn recording_wire_deps(root: &std::path::Path, provider: Arc<FakeProvider>) -> ServerDeps {
        let mut registry = faktor_provider::ProviderRegistry::new();
        registry.try_register(provider).unwrap();
        let session = SessionManager::open(root.join("store"), root.join("cas"), true).unwrap();
        let permissions = ChannelPermissionRequester::new(Duration::from_secs(5));
        let agent = AgentRuntime::new(faktor_agent::AgentDeps {
            session: session.clone(),
            providers: Arc::new(registry),
            chunk_sink: None,
            permission_requester: permissions.clone(),
            evidence: Arc::new(faktor_agent::NoEvidence),
            tools: Arc::new(faktor_agent::ToolRegistry::new()),
            cas: None,
            workspaces: faktor_fs::WorkspaceFileService::new(),
            edit: None,
            snapshots: None,
            sandbox: None,
            supervisor: None,
            verification: faktor_agent::VerificationService::disabled(),
            model: "m".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "You are a test server agent.".into(),
            hooks: None,
            instructions_resolver: faktor_instructions::no_roots_resolver(),
            routing: faktor_agent::FixedRoutingPolicy::passthrough(),
            budgets: Arc::new(faktor_session::NoopBudget),
            clock: Arc::new(faktor_core::time::SystemClock),
            tool_call_mode: faktor_agent::ToolCallMode::Native,
            tool_deadline_ms: 2000,
            retry_policy: faktor_core::retry::RetryPolicy::default(),
            semantic: faktor_agent::fallback_semantic_registry(),
            context_prior: None,
            efficiency: Default::default(),
        })
        .unwrap();
        let (orchestrator, tasks) = orch_pair(session.clone(), agent.clone());
        ServerDeps {
            budgets: faktor_session::DurableBudgetLedger::new(session.clone()),
            session,
            agent,
            permissions,
            orchestrator,
            tasks,
            auth_token: AuthToken::generate(),
            server_password: ServerPassword::generate(),
            directory: None,
            version: "0.1.0".into(),
            fs: None,
            snapshots: None,
            chunk_rx: None,
            simulate_not_ready: false,
            evidence: None,
            semantic: None,
        }
    }

    #[tokio::test]
    async fn message_model_override_applied_via_wire() {
        let dir = tempfile::tempdir().unwrap();
        let provider = Arc::new(FakeProvider::with_script(
            "fake",
            ModelCapabilities {
                tools: true,
                ..Default::default()
            },
            vec![
                faktor_provider::ScriptedResponse::Text("pong".into()),
                faktor_provider::ScriptedResponse::End,
            ],
        ));
        let deps = recording_wire_deps(dir.path(), provider.clone());
        let pw = deps.server_password.clone();
        let session = deps.session.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let basic = |r: reqwest::RequestBuilder| r.basic_auth("kilo", Some(pw.as_str()));

        // Session configured with model m1.
        let resp = basic(
            client
                .post(format!("{base}/session"))
                .json(&serde_json::json!({"model": {"id": "m1", "providerID": "fake"}})),
        )
        .send()
        .await
        .unwrap();
        assert_eq!(resp.status(), 200);
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid = created["sessionID"].as_str().unwrap().to_string();
        let sid_parsed = parse_session_id(&sid).unwrap();
        let row = session
            .get_session(sid_parsed)
            .unwrap()
            .unwrap()
            .row()
            .unwrap();
        assert_eq!(row.model, "m1");

        // Message overriding to m2 within the same provider.
        let resp = basic(client.post(format!("{base}/session/{sid}/message")).json(
            &serde_json::json!({
                "model": {"providerID": "fake", "modelID": "m2"},
                "parts": [{"type": "text", "text": "use m2"}],
            }),
        ))
        .send()
        .await
        .unwrap();
        assert_eq!(resp.status(), 200);
        // Frozen send shape: the durable assistant message of the accepted
        // turn (the response arrived AFTER the turn completed) with its
        // parts; info carries the model that was actually used.
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["info"]["role"], "assistant");
        assert_eq!(body["info"]["sessionID"], sid);
        assert!(body["info"]["messageID"]
            .as_str()
            .unwrap()
            .parse::<u64>()
            .is_ok());
        assert_eq!(body["info"]["modelID"], "m2");
        assert!(!body["parts"].as_array().unwrap().is_empty());
        assert!(
            body.as_object().unwrap().get("accepted").is_none(),
            "the old envelope is gone: {body}"
        );

        // The agent's wire request carried m2 — the override applies.
        let mut recorded = None;
        for _ in 0..100 {
            if let Some(m) = provider.last_request_model() {
                recorded = Some(m);
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(
            recorded.as_deref(),
            Some("m2"),
            "the override must reach the agent's wire request"
        );
        // The journaled session row keeps its configured model.
        let row = session
            .get_session(sid_parsed)
            .unwrap()
            .unwrap()
            .row()
            .unwrap();
        assert_eq!(row.model, "m1", "override must not mutate the session row");
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn message_model_provider_mismatch_409() {
        let dir = tempfile::tempdir().unwrap();
        let provider = Arc::new(FakeProvider::with_script(
            "fake",
            ModelCapabilities {
                tools: true,
                ..Default::default()
            },
            vec![
                faktor_provider::ScriptedResponse::Text("pong".into()),
                faktor_provider::ScriptedResponse::End,
            ],
        ));
        let deps = recording_wire_deps(dir.path(), provider.clone());
        let pw = deps.server_password.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let basic = |r: reqwest::RequestBuilder| r.basic_auth("kilo", Some(pw.as_str()));

        let resp = basic(
            client
                .post(format!("{base}/session"))
                .json(&serde_json::json!({"model": {"id": "m1", "providerID": "fake"}})),
        )
        .send()
        .await
        .unwrap();
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid = created["sessionID"].as_str().unwrap().to_string();

        // A provider that is not the session's provider: honest 409, and
        // nothing is spawned (no request can reach the provider).
        let resp = basic(client.post(format!("{base}/session/{sid}/message")).json(
            &serde_json::json!({
                "model": {"providerID": "other", "modelID": "m2"},
                "parts": [{"type": "text", "text": "hi"}],
            }),
        ))
        .send()
        .await
        .unwrap();
        assert_eq!(resp.status(), 409);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["ok"], false);
        assert_eq!(body["message"], "provider mismatch");
        assert!(
            provider.last_request_model().is_none(),
            "a mismatched message must never reach the provider"
        );

        // The session still accepts a matching message afterwards.
        let resp = basic(client.post(format!("{base}/session/{sid}/message")).json(
            &serde_json::json!({
                "model": {"providerID": "fake", "modelID": "m1"},
                "parts": [{"type": "text", "text": "hi"}],
            }),
        ))
        .send()
        .await
        .unwrap();
        assert_eq!(resp.status(), 200);
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn message_without_model_uses_session_model() {
        let dir = tempfile::tempdir().unwrap();
        let provider = Arc::new(FakeProvider::with_script(
            "fake",
            ModelCapabilities {
                tools: true,
                ..Default::default()
            },
            vec![
                faktor_provider::ScriptedResponse::Text("pong".into()),
                faktor_provider::ScriptedResponse::End,
            ],
        ));
        let deps = recording_wire_deps(dir.path(), provider.clone());
        let pw = deps.server_password.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let basic = |r: reqwest::RequestBuilder| r.basic_auth("kilo", Some(pw.as_str()));

        // Session configured with model m1; the message carries the
        // session's own model (no effective override).
        let resp = basic(
            client
                .post(format!("{base}/session"))
                .json(&serde_json::json!({"model": {"id": "m1", "providerID": "fake"}})),
        )
        .send()
        .await
        .unwrap();
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid = created["sessionID"].as_str().unwrap().to_string();

        let resp = basic(client.post(format!("{base}/session/{sid}/message")).json(
            &serde_json::json!({
                "model": {"providerID": "fake", "modelID": "m1"},
                "parts": [{"type": "text", "text": "plain"}],
            }),
        ))
        .send()
        .await
        .unwrap();
        assert_eq!(resp.status(), 200);

        let mut recorded = None;
        for _ in 0..100 {
            if let Some(m) = provider.last_request_model() {
                recorded = Some(m);
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(
            recorded.as_deref(),
            Some("m1"),
            "the session model must be used when nothing overrides it"
        );
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn wire_diff_revert_unrevert_are_honest_stubs() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let pw = deps.server_password.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        let resp = client
            .post(format!("{base}/session"))
            .basic_auth("kilo", Some(pw.as_str()))
            .json(&serde_json::json!({"model": {"id": "m", "providerID": "fake"}}))
            .send()
            .await
            .unwrap();
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid = created["sessionID"].as_str().unwrap().to_string();

        // diff: frozen SnapshotFileDiff[] shape — an honest empty array
        // when the session has no checkpoint rows (nothing to diff).
        let resp = client
            .get(format!("{base}/session/{sid}/diff"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(
            body,
            serde_json::json!([]),
            "no checkpoints → the frozen array projection is empty"
        );
        // Same for the filter forms: unknown message → honest 409.
        let resp = client
            .get(format!("{base}/session/{sid}/diff?message=99"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 409);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["ok"], false);
        assert!(body["message"]
            .as_str()
            .unwrap()
            .contains("unknown message id"));
        // A non-session diff path is a loud 404 like every other wire route.
        let resp = client
            .get(format!("{base}/session/999999/diff"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);

        // revert/unrevert: honest {ok:false} + message with 409, never a
        // silent success.
        for path in ["revert", "unrevert"] {
            let resp = client
                .post(format!("{base}/session/{sid}/{path}"))
                .basic_auth("kilo", Some(pw.as_str()))
                .json(&serde_json::json!({"messageID": "1"}))
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 409, "{path} must be refused honestly");
            let body: serde_json::Value = resp.json().await.unwrap();
            assert_eq!(body["ok"], false);
            assert!(body["message"].as_str().unwrap().contains("unavailable"));
        }
        // Malformed revert body / message id → 400/422; missing body → 422.
        let resp = client
            .post(format!("{base}/session/{sid}/revert"))
            .basic_auth("kilo", Some(pw.as_str()))
            .json(&serde_json::json!({"messageID": "not-a-number"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);
        let resp = client
            .post(format!("{base}/session/{sid}/revert"))
            .basic_auth("kilo", Some(pw.as_str()))
            .json(&serde_json::json!({}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 422, "missing messageID is a strict-body 422");
        let resp = client
            .post(format!("{base}/session/{sid}/revert"))
            .basic_auth("kilo", Some(pw.as_str()))
            .json(&serde_json::json!({"messageID": "1", "extra": 1}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 422);
        let _ = handle.shutdown.send(());
    }

    /// A daemon whose wire snapshot surface is wired to the real native
    /// store: same store + CAS the session manager opened, plus a file
    /// service. Returns the deps, the checkpoint store used to record edits,
    /// and the file service.
    fn wire_snapshot_deps(
        root: &std::path::Path,
    ) -> (
        ServerDeps,
        Arc<faktor_snapshot::CheckpointStore>,
        Arc<faktor_fs::WorkspaceFileService>,
    ) {
        let deps = test_deps(root);
        let fs = faktor_fs::WorkspaceFileService::new();
        let snapshots = Arc::new(faktor_snapshot::CheckpointStore::new(
            deps.session.cas(),
            deps.session.store(),
        ));
        let deps = deps.with_snapshots(fs.clone(), snapshots.clone());
        (deps, snapshots, fs)
    }

    #[tokio::test]
    async fn revert_restores_file_via_wire() {
        let dir = tempfile::tempdir().unwrap();
        let ws_root = dir.path().join("ws");
        std::fs::create_dir_all(&ws_root).unwrap();
        let (deps, snapshots, fs) = wire_snapshot_deps(dir.path());
        let session_mgr = deps.session.clone();
        let pw = deps.server_password.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        // Create a session rooted at the real workspace dir.
        let resp = client
            .post(format!("{base}/session"))
            .basic_auth("kilo", Some(pw.as_str()))
            .header("x-faktor-directory", ws_root.to_str().unwrap())
            .json(&serde_json::json!({"model": {"id": "m", "providerID": "fake"}}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid: u64 = created["sessionID"].as_str().unwrap().parse().unwrap();
        let session = faktor_core::id::SessionId::new(sid);

        // Record a checkpoint exactly like the edit engine would: original
        // content captured, file edited, after-content stored in the CAS.
        let file = ws_root.join("notes.txt");
        std::fs::write(&file, b"original\n").unwrap();
        let before = snapshots
            .before_write(session, "notes.txt", b"original\n")
            .unwrap();
        let ws_handle = fs
            .open(faktor_core::WorkspaceId::new(sid), ws_root.clone())
            .unwrap();
        let after = ws_handle
            .write_atomic(std::path::Path::new("notes.txt"), b"edited by agent\n")
            .unwrap();
        snapshots
            .after_write(session, "notes.txt", before, after, 0, b"edited by agent\n")
            .unwrap();
        // The message the user asks to revert to arrives AFTER the edit was
        // checkpointed (revert-to-message = undo everything since it).
        let store = session_mgr.store();
        store
            .put_message(session, 1, "user", serde_json::json!({"text": "fix it"}))
            .unwrap();
        assert_eq!(std::fs::read(&file).unwrap(), b"edited by agent\n");

        // POST revert: the file must be restored to the pre-edit state.
        let resp = client
            .post(format!("{base}/session/{sid}/revert"))
            .basic_auth("kilo", Some(pw.as_str()))
            .json(&serde_json::json!({"messageID": "1"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "{:?}", resp.text().await);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["ok"], true);
        let restored = body["restored"][0].clone();
        assert_eq!(restored["path"], "notes.txt");
        assert_eq!(restored["hash"], before.to_hex());
        assert_eq!(std::fs::read(&file).unwrap(), b"original\n");
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn revert_conflict_409_via_wire() {
        let dir = tempfile::tempdir().unwrap();
        let ws_root = dir.path().join("ws");
        std::fs::create_dir_all(&ws_root).unwrap();
        let (deps, snapshots, fs) = wire_snapshot_deps(dir.path());
        let session_mgr = deps.session.clone();
        let pw = deps.server_password.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        let resp = client
            .post(format!("{base}/session"))
            .basic_auth("kilo", Some(pw.as_str()))
            .header("x-faktor-directory", ws_root.to_str().unwrap())
            .json(&serde_json::json!({"model": {"id": "m", "providerID": "fake"}}))
            .send()
            .await
            .unwrap();
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid: u64 = created["sessionID"].as_str().unwrap().parse().unwrap();
        let session = faktor_core::id::SessionId::new(sid);

        let file = ws_root.join("notes.txt");
        std::fs::write(&file, b"original\n").unwrap();
        let before = snapshots
            .before_write(session, "notes.txt", b"original\n")
            .unwrap();
        let ws_handle = fs
            .open(faktor_core::WorkspaceId::new(sid), ws_root.clone())
            .unwrap();
        let after = ws_handle
            .write_atomic(std::path::Path::new("notes.txt"), b"edited by agent\n")
            .unwrap();
        snapshots
            .after_write(session, "notes.txt", before, after, 0, b"edited by agent\n")
            .unwrap();
        session_mgr
            .store()
            .put_message(session, 1, "user", serde_json::json!({"text": "fix it"}))
            .unwrap();
        // The user edits the file independently after the agent's edit:
        // revert must conflict and never clobber.
        std::fs::write(&file, b"user owns this now\n").unwrap();

        let resp = client
            .post(format!("{base}/session/{sid}/revert"))
            .basic_auth("kilo", Some(pw.as_str()))
            .json(&serde_json::json!({"messageID": "1"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 409);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["ok"], false);
        assert_eq!(body["conflict"]["path"], "notes.txt");
        assert_eq!(
            std::fs::read(&file).unwrap(),
            b"user owns this now\n",
            "a conflict must never overwrite the user's content"
        );
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn unrevert_restores_after_state_via_wire() {
        let dir = tempfile::tempdir().unwrap();
        let ws_root = dir.path().join("ws");
        std::fs::create_dir_all(&ws_root).unwrap();
        let (deps, snapshots, fs) = wire_snapshot_deps(dir.path());
        let session_mgr = deps.session.clone();
        let pw = deps.server_password.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        let resp = client
            .post(format!("{base}/session"))
            .basic_auth("kilo", Some(pw.as_str()))
            .header("x-faktor-directory", ws_root.to_str().unwrap())
            .json(&serde_json::json!({"model": {"id": "m", "providerID": "fake"}}))
            .send()
            .await
            .unwrap();
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid: u64 = created["sessionID"].as_str().unwrap().parse().unwrap();
        let session = faktor_core::id::SessionId::new(sid);

        let file = ws_root.join("notes.txt");
        std::fs::write(&file, b"original\n").unwrap();
        let before = snapshots
            .before_write(session, "notes.txt", b"original\n")
            .unwrap();
        let ws_handle = fs
            .open(faktor_core::WorkspaceId::new(sid), ws_root.clone())
            .unwrap();
        let after = ws_handle
            .write_atomic(std::path::Path::new("notes.txt"), b"edited by agent\n")
            .unwrap();
        snapshots
            .after_write(session, "notes.txt", before, after, 0, b"edited by agent\n")
            .unwrap();
        session_mgr
            .store()
            .put_message(session, 1, "user", serde_json::json!({"text": "fix it"}))
            .unwrap();

        // revert → pre-edit state; unrevert → the after state comes back.
        let resp = client
            .post(format!("{base}/session/{sid}/revert"))
            .basic_auth("kilo", Some(pw.as_str()))
            .json(&serde_json::json!({"messageID": "1"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(std::fs::read(&file).unwrap(), b"original\n");
        let resp = client
            .post(format!("{base}/session/{sid}/unrevert"))
            .basic_auth("kilo", Some(pw.as_str()))
            .json(&serde_json::json!({"messageID": "1"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "{:?}", resp.text().await);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["ok"], true);
        assert_eq!(body["restored"][0]["hash"], after.to_hex());
        assert_eq!(std::fs::read(&file).unwrap(), b"edited by agent\n");
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn diff_returns_projected_array_with_filters_via_wire() {
        // Frozen shape: SnapshotFileDiff[] — one entry per recorded
        // file-change checkpoint row, newest first, with added|deleted|
        // modified status and (only with ?full=1) the unified diff content.
        // Filters: ?message=<seq> limits to ONE checkpoint (the newest one
        // recorded at-or-before that message), ?file=<rel> filters paths.
        let dir = tempfile::tempdir().unwrap();
        let ws_root = dir.path().join("ws");
        std::fs::create_dir_all(&ws_root).unwrap();
        let (deps, snapshots, fs) = wire_snapshot_deps(dir.path());
        let session_mgr = deps.session.clone();
        let pw = deps.server_password.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        let resp = client
            .post(format!("{base}/session"))
            .basic_auth("kilo", Some(pw.as_str()))
            .header("x-faktor-directory", ws_root.to_str().unwrap())
            .json(&serde_json::json!({"model": {"id": "m", "providerID": "fake"}}))
            .send()
            .await
            .unwrap();
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid: u64 = created["sessionID"].as_str().unwrap().parse().unwrap();
        let session = faktor_core::id::SessionId::new(sid);

        // No checkpoints yet: the frozen array projection is empty.
        let resp = client
            .get(format!("{base}/session/{sid}/diff"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(
            resp.json::<serde_json::Value>().await.unwrap(),
            serde_json::json!([])
        );

        // Timeline: message1, then edit A (f.txt modified), then message2,
        // then creation B (created-empty.txt), then deletion C (f.txt).
        let store = session_mgr.store();
        store
            .put_message(session, 1, "user", serde_json::json!({"text": "one"}))
            .unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;

        let before_text = "line1\nline2\nline3\nline4\nold\nline6\nline7\n";
        let after_text = "line1\nline2\nline3\nline4\nnew\nline6\nline7\n";
        let file = ws_root.join("f.txt");
        std::fs::write(&file, before_text).unwrap();
        let before = snapshots
            .before_write(session, "f.txt", before_text.as_bytes())
            .unwrap();
        let ws_handle = fs
            .open(faktor_core::WorkspaceId::new(sid), ws_root.clone())
            .unwrap();
        let after = ws_handle
            .write_atomic(std::path::Path::new("f.txt"), after_text.as_bytes())
            .unwrap();
        snapshots
            .after_write(session, "f.txt", before, after, 0, after_text.as_bytes())
            .unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;

        store
            .put_message(session, 2, "user", serde_json::json!({"text": "two"}))
            .unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Creation: an empty-file row must project status "added", never a
        // no-op (hash("")==hash("")).
        let empty_hash = snapshots
            .before_write(session, "created-empty.txt", b"")
            .unwrap();
        let file2 = ws_root.join("created-empty.txt");
        snapshots
            .record_change(
                session,
                "created-empty.txt",
                faktor_snapshot::FileState::missing(),
                None,
                faktor_snapshot::FileState::existing(empty_hash),
                Some(b""),
            )
            .unwrap();
        std::fs::write(&file2, b"").unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Deletion C: a pure-removal row.
        snapshots
            .record_change(
                session,
                "f.txt",
                faktor_snapshot::FileState::existing(after),
                None,
                faktor_snapshot::FileState::missing(),
                None,
            )
            .unwrap();

        // Default projection: ALL rows, newest first, statuses only.
        let resp = client
            .get(format!("{base}/session/{sid}/diff"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        let arr = body.as_array().unwrap();
        assert_eq!(arr.len(), 3, "{body}");
        let statuses: Vec<(&str, &str)> = arr
            .iter()
            .map(|e| (e["path"].as_str().unwrap(), e["status"].as_str().unwrap()))
            .collect();
        assert_eq!(
            statuses,
            vec![
                ("f.txt", "deleted"),
                ("created-empty.txt", "added"),
                ("f.txt", "modified")
            ],
            "newest checkpoint first with exact status tags"
        );
        // Without ?full=1 entries carry path+status only (no diff).
        for e in arr {
            assert!(!e.as_object().unwrap().contains_key("diff"), "{e}");
        }

        // ?file=<rel> filters the projection to that path.
        let resp = client
            .get(format!("{base}/session/{sid}/diff?file=f.txt"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        let body: serde_json::Value = resp.json().await.unwrap();
        let arr = body.as_array().unwrap();
        assert_eq!(arr.len(), 2, "{body}");
        assert!(
            arr.iter().all(|e| e["path"] == "f.txt"),
            "file filter must apply: {body}"
        );
        // Unknown file → empty array (200, never an error).
        let resp = client
            .get(format!("{base}/session/{sid}/diff?file=nope.rs"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(
            resp.json::<serde_json::Value>().await.unwrap(),
            serde_json::json!([])
        );

        // ?message=<seq> limits to ONE checkpoint: message 1 predates every
        // checkpoint → empty; message 2 (recorded after edit A, before B/C)
        // → exactly edit A's row.
        let resp = client
            .get(format!("{base}/session/{sid}/diff?message=1"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.json::<serde_json::Value>().await.unwrap(),
            serde_json::json!([]),
            "no checkpoint existed at message 1"
        );
        let resp = client
            .get(format!("{base}/session/{sid}/diff?message=2"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        let body: serde_json::Value = resp.json().await.unwrap();
        let arr = body.as_array().unwrap();
        assert_eq!(arr.len(), 1, "{body}");
        assert_eq!(arr[0]["path"], "f.txt");
        assert_eq!(arr[0]["status"], "modified");
        // An unknown message is an honest 409, never an empty success.
        let resp = client
            .get(format!("{base}/session/{sid}/diff?message=99"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 409);
        assert_eq!(resp.json::<serde_json::Value>().await.unwrap()["ok"], false);

        // ?full=1 adds the unified content to every entry (resolution via
        // the CAS), newest first.
        let resp = client
            .get(format!("{base}/session/{sid}/diff?full=1"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        let arr = body.as_array().unwrap();
        assert_eq!(arr.len(), 3);
        // Newest entry: the deletion of f.txt diff must be pure removals.
        let del = &arr[0];
        assert_eq!(del["status"], "deleted");
        let diff = del["diff"].as_str().unwrap();
        assert!(
            diff.lines().any(|l| l == "-new"),
            "deletion must diff as removals: {diff}"
        );
        // Oldest entry: the modification of f.txt with full context.
        let modified_entry = &arr[2];
        assert_eq!(modified_entry["status"], "modified");
        let diff = modified_entry["diff"].as_str().unwrap();
        assert!(diff.lines().any(|l| l == "-old"), "removal missing: {diff}");
        assert!(
            diff.lines().any(|l| l == "+new"),
            "addition missing: {diff}"
        );
        assert!(
            diff.lines().any(|l| l == " line2"),
            "context missing: {diff}"
        );
        // The creation entry carries the added status with full content too.
        assert_eq!(arr[1]["status"], "added");
        assert!(arr[1]["diff"].is_string());
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn revert_unknown_message_id_409_via_wire() {
        let dir = tempfile::tempdir().unwrap();
        let ws_root = dir.path().join("ws");
        std::fs::create_dir_all(&ws_root).unwrap();
        let (deps, _snapshots, _fs) = wire_snapshot_deps(dir.path());
        let pw = deps.server_password.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        let resp = client
            .post(format!("{base}/session"))
            .basic_auth("kilo", Some(pw.as_str()))
            .header("x-faktor-directory", ws_root.to_str().unwrap())
            .json(&serde_json::json!({"model": {"id": "m", "providerID": "fake"}}))
            .send()
            .await
            .unwrap();
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid = created["sessionID"].as_str().unwrap().to_string();
        // No message with seq 42 exists: honest 409, never a silent no-op.
        let resp = client
            .post(format!("{base}/session/{sid}/revert"))
            .basic_auth("kilo", Some(pw.as_str()))
            .json(&serde_json::json!({"messageID": "42"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 409);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["ok"], false);
        assert!(body["message"]
            .as_str()
            .unwrap()
            .contains("unknown message id"));
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn sdk_full_flow_create_prompt_state_messages_abort_list() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let pw = deps.server_password.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        // Create.
        let resp = client
            .post(format!("{base}/session/create"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({
                "provider": "fake",
                "model": "m",
                "workspace": "/tmp",
                "title": "sdk t1",
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid = created["id"].as_str().unwrap().to_string();
        assert_eq!(created["title"], "sdk t1");
        assert!(created["created_ms"].as_i64().unwrap() > 0);

        // Prompt with files + models (models is opaque, must be accepted).
        let resp = client
            .post(format!("{base}/session/prompt"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({
                "session_id": sid,
                "prompt": "hi",
                "files": ["a.rs"],
                "models": {"main": {"provider": "fake", "model": "m"}},
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let pr: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(pr["accepted"], true);
        assert_eq!(pr["queued"], false);
        assert!(pr["op_id"].is_string());

        // State converges.
        let mut state = String::new();
        for _ in 0..100 {
            let resp = client
                .get(format!("{base}/session/state?session_id={sid}"))
                .header("x-faktor-server-password", pw.as_str())
                .send()
                .await
                .unwrap();
            let body: serde_json::Value = resp.json().await.unwrap();
            state = body["agent_state"]["state"]
                .as_str()
                .unwrap_or("")
                .to_string();
            if matches!(
                state.as_str(),
                "ready_for_next_turn" | "completed" | "cancelled"
            ) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(state, "ready_for_next_turn", "turn must complete");

        // Messages page.
        let resp = client
            .get(format!("{base}/session/messages?session_id={sid}&limit=10"))
            .header("x-faktor-server-password", pw.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let page: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(page["session_id"], sid);
        assert!(page["messages"].as_array().unwrap().len() >= 2, "{page}");

        // Abort (nothing running now): frozen shape, no error.
        let resp = client
            .post(format!("{base}/session/abort"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"session_id": sid}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let ab: serde_json::Value = resp.json().await.unwrap();
        assert!(ab["aborted"].is_array());

        // List contains the session.
        let resp = client
            .get(format!("{base}/session/list"))
            .header("x-faktor-server-password", pw.as_str())
            .send()
            .await
            .unwrap();
        let list: serde_json::Value = resp.json().await.unwrap();
        let ids: Vec<&str> = list["sessions"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|s| s["id"].as_str())
            .collect();
        assert!(ids.contains(&sid.as_str()));

        // Unknown sessions are loud 404s on every SDK route.
        for (method, path) in [
            ("get", "/session/state?session_id=999999"),
            ("get", "/session/messages?session_id=999999"),
        ] {
            let resp = if method == "get" {
                client
                    .get(format!("{base}{path}"))
                    .header("x-faktor-server-password", pw.as_str())
                    .send()
                    .await
                    .unwrap()
            } else {
                unreachable!()
            };
            assert_eq!(resp.status(), 404, "{path}");
        }
        let resp = client
            .post(format!("{base}/session/prompt"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"session_id": "999999", "prompt": "x"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);
        let resp = client
            .post(format!("{base}/session/abort"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"session_id": "999999"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);

        // Malformed ids and empty prompts are 400s.
        let resp = client
            .post(format!("{base}/session/prompt"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"session_id": "0", "prompt": "x"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);
        let resp = client
            .post(format!("{base}/session/prompt"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"session_id": sid, "prompt": "   "}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);
        // Unknown fields in the SDK body are protocol drift (422 from the
        // deny_unknown_fields extraction gate).
        let resp = client
            .post(format!("{base}/session/prompt"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"session_id": sid, "prompt": "x", "evil": true}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 422);

        let _ = handle.shutdown.send(());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn global_event_stream_delivers_envelopes_and_resumes() {
        use futures_util::StreamExt;
        let dir = tempfile::tempdir().unwrap();
        let mut deps = test_deps(dir.path());
        deps.directory = Some("/w".into());
        let pw = deps.server_password.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        let mut sse = client
            .get(format!("{base}/global/event?after=0"))
            .header("x-faktor-server-password", pw.as_str())
            .send()
            .await
            .unwrap()
            .bytes_stream();

        let resp = client
            .post(format!("{base}/session/create"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"provider": "fake", "model": "m"}))
            .send()
            .await
            .unwrap();
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid = created["id"].as_str().unwrap().to_string();

        // Read frames until session_created arrives; record its SSE id.
        let mut buf = String::new();
        let mut created_id = None;
        for _ in 0..300 {
            match tokio::time::timeout(Duration::from_millis(200), sse.next()).await {
                Ok(Some(Ok(chunk))) => {
                    buf.push_str(&String::from_utf8_lossy(&chunk));
                    if let Some(id) = frame_id_containing(&buf, "session_created") {
                        created_id = Some(id);
                        break;
                    }
                }
                Ok(Some(Err(_))) | Ok(None) | Err(_) => break,
            }
        }
        let created_id = created_id.expect("session_created frame must arrive");
        // The envelope carries the directory on every frame.
        assert!(
            buf.contains("\"directory\":\"/w\""),
            "envelope directory missing"
        );

        // Prompt and read the turn_open frame.
        client
            .post(format!("{base}/session/prompt"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"session_id": sid, "prompt": "hi"}))
            .send()
            .await
            .unwrap();
        let mut saw_turn_open = false;
        for _ in 0..300 {
            match tokio::time::timeout(Duration::from_millis(200), sse.next()).await {
                Ok(Some(Ok(chunk))) => {
                    buf.push_str(&String::from_utf8_lossy(&chunk));
                    if buf.contains("session_turn_open") {
                        saw_turn_open = true;
                        break;
                    }
                }
                Ok(Some(Err(_))) | Ok(None) | Err(_) => break,
            }
        }
        assert!(saw_turn_open, "stream must deliver turn_open; got: {buf}");
        drop(sse);

        // Resume after the created frame: no replay of session_created, but
        // the subsequent events are delivered with strictly larger ids.
        let mut sse2 = client
            .get(format!("{base}/global/event?after={created_id}"))
            .header("x-faktor-server-password", pw.as_str())
            .send()
            .await
            .unwrap()
            .bytes_stream();
        client
            .post(format!("{base}/session/prompt"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"session_id": sid, "prompt": "again"}))
            .send()
            .await
            .unwrap();
        let mut buf2 = String::new();
        let mut resumed = false;
        for _ in 0..300 {
            match tokio::time::timeout(Duration::from_millis(200), sse2.next()).await {
                Ok(Some(Ok(chunk))) => {
                    buf2.push_str(&String::from_utf8_lossy(&chunk));
                    if buf2.contains("session_turn_open") {
                        resumed = true;
                        break;
                    }
                }
                Ok(Some(Err(_))) | Ok(None) | Err(_) => break,
            }
        }
        assert!(
            resumed,
            "resumed stream must deliver new frames; got: {buf2}"
        );
        assert!(
            !buf2.contains("session_created"),
            "resume after {created_id} must not replay session_created"
        );
        // Every resumed frame's id is strictly greater than the cursor.
        for (id, ge) in parse_global_frames(&buf2) {
            assert!(
                id > created_id,
                "resume cursor violated: {id} <= {created_id}"
            );
            assert!(ge.payload.type_name() != "session_created");
        }
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn global_event_oversized_after_is_clamped_and_negative_rejected() {
        use futures_util::StreamExt;
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let pw = deps.server_password.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        // u64::MAX is clamped (stream stays open, never an error).
        let resp = client
            .get(format!("{base}/global/event?after={}", u64::MAX))
            .header("x-faktor-server-password", pw.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let mut sse = resp.bytes_stream();
        let mut saw_data = false;
        for _ in 0..50 {
            match tokio::time::timeout(Duration::from_millis(200), sse.next()).await {
                Ok(Some(Ok(chunk))) => {
                    let text = String::from_utf8_lossy(&chunk);
                    if text.contains("data:") {
                        saw_data = true;
                        break;
                    }
                }
                _ => break,
            }
        }
        assert!(
            saw_data,
            "clamped stream must stay alive (heartbeat/frames)"
        );
        drop(sse);

        // Negative after is malformed: 400.
        let resp = client
            .get(format!("{base}/global/event?after=-1"))
            .header("x-faktor-server-password", pw.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);

        // No password on the event stream: 401.
        let resp = client
            .get(format!("{base}/global/event?after=0"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn permission_reply_and_list_via_sdk() {
        let dir = tempfile::tempdir().unwrap();
        let mut registry = faktor_provider::ProviderRegistry::new();
        registry
            .try_register(Arc::new(FakeProvider::with_script(
                "fake",
                ModelCapabilities {
                    tools: true,
                    ..Default::default()
                },
                vec![
                    faktor_provider::ScriptedResponse::ToolCall {
                        id: "c1".into(),
                        name: "echo".into(),
                        input: serde_json::json!({"x": 1}),
                    },
                    faktor_provider::ScriptedResponse::End,
                ],
            )))
            .unwrap();
        let session =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let permissions = ChannelPermissionRequester::new(Duration::from_secs(5));
        let mut tools = faktor_agent::ToolRegistry::new();
        tools.register(faktor_agent::Tool {
            name: "echo".into(),
            description: "d".into(),
            input_schema: serde_json::json!({}),
            resource_class: faktor_core::resource::ResourceClass::Cpu,
            capability: None,
            recovery_hint: faktor_agent::RecoveryHint::Idempotent,
            path_args: vec![],
            execute: Arc::new(|_ctx, _args| {
                Box::pin(async move { Ok(faktor_agent::ToolOutcome::default()) })
            }),
        });
        let agent = AgentRuntime::new(faktor_agent::AgentDeps {
            session: session.clone(),
            providers: Arc::new(registry),
            chunk_sink: None,
            permission_requester: permissions.clone(),
            evidence: Arc::new(faktor_agent::NoEvidence),
            tools: Arc::new(tools),
            cas: None,
            workspaces: faktor_fs::WorkspaceFileService::new(),
            edit: None,
            snapshots: None,
            sandbox: None,
            supervisor: None,
            verification: faktor_agent::VerificationService::disabled(),
            model: "m".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "You are a test server agent.".into(),
            hooks: None,
            instructions_resolver: faktor_instructions::no_roots_resolver(),
            routing: faktor_agent::FixedRoutingPolicy::passthrough(),
            budgets: Arc::new(faktor_session::NoopBudget),
            clock: Arc::new(faktor_core::time::SystemClock),
            tool_call_mode: faktor_agent::ToolCallMode::Native,
            tool_deadline_ms: 2000,
            retry_policy: faktor_core::retry::RetryPolicy::default(),
            semantic: faktor_agent::fallback_semantic_registry(),
            context_prior: None,
            efficiency: Default::default(),
        })
        .unwrap();
        let (orchestrator, tasks) = orch_pair(session.clone(), agent.clone());
        let deps = ServerDeps {
            budgets: faktor_session::DurableBudgetLedger::new(session.clone()),
            session: session.clone(),
            agent,
            permissions: permissions.clone(),
            orchestrator,
            tasks,
            auth_token: AuthToken::generate(),
            server_password: ServerPassword::generate(),
            directory: None,
            version: "0.1.0".into(),
            fs: None,
            snapshots: None,
            chunk_rx: None,
            simulate_not_ready: false,
            evidence: None,
            semantic: None,
        };
        let pw = deps.server_password.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        let resp = client
            .post(format!("{base}/session/create"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"provider": "fake", "model": "m"}))
            .send()
            .await
            .unwrap();
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid = created["id"].as_str().unwrap().to_string();
        client
            .post(format!("{base}/session/prompt"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"session_id": sid, "prompt": "use tools"}))
            .send()
            .await
            .unwrap();

        // The permission surfaces in /permission/list with its session.
        let mut pid = None;
        let perm_deadline = std::time::Instant::now() + Duration::from_secs(90);
        loop {
            if std::time::Instant::now() >= perm_deadline {
                break;
            }
            let resp = client
                .get(format!("{base}/permission/list?session_id={sid}"))
                .header("x-faktor-server-password", pw.as_str())
                .send()
                .await
                .unwrap();
            let list: serde_json::Value = resp.json().await.unwrap();
            let perms = list["permissions"].as_array().unwrap();
            assert!(
                perms
                    .iter()
                    .all(|p| p["session_id"].as_str() == Some(sid.as_str())),
                "session filter must apply"
            );
            if let Some(first) = perms.first() {
                assert_eq!(first["capability"], "execute_shell");
                assert!(first["detail"].is_object());
                pid = first["id"].as_str().map(String::from);
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let pid = pid.expect("permission must surface in /permission/list");

        // Resolve through /permission/reply.
        let resp = client
            .post(format!("{base}/permission/reply"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"permission_id": pid, "decision": "allow"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["ok"], true);

        // The turn completes.
        let mut done = false;
        for _ in 0..100 {
            let id = parse_session_id(&sid).unwrap();
            let state = session.get_session(id).unwrap().unwrap().state().unwrap();
            if matches!(state, faktor_core::state::AgentState::ReadyForNextTurn) {
                done = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(done, "turn must finish after permission grant");

        // The resolved permission is gone from the list.
        let resp = client
            .get(format!("{base}/permission/list"))
            .header("x-faktor-server-password", pw.as_str())
            .send()
            .await
            .unwrap();
        let list: serde_json::Value = resp.json().await.unwrap();
        assert!(list["permissions"].as_array().unwrap().is_empty());

        // Double reply → 409; malformed ids/decisions → 400.
        let resp = client
            .post(format!("{base}/permission/reply"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"permission_id": pid, "decision": "allow"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 409);
        let resp = client
            .post(format!("{base}/permission/reply"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"permission_id": "bogus", "decision": "allow"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);
        let resp = client
            .post(format!("{base}/permission/reply"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"permission_id": "1", "decision": "maybe"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn question_network_and_config_endpoints() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let pw = deps.server_password.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        // Questions: empty list; unknown replies are loud 404s.
        let resp = client
            .get(format!("{base}/question/list"))
            .header("x-faktor-server-password", pw.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["questions"], serde_json::json!([]));
        let resp = client
            .post(format!("{base}/question/reply"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"question_id": "q1", "decision": "allow"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);
        let resp = client
            .post(format!("{base}/question/reply"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"question_id": "", "decision": "allow"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);

        // Networks: same shapes.
        let resp = client
            .get(format!("{base}/network/list"))
            .header("x-faktor-server-password", pw.as_str())
            .send()
            .await
            .unwrap();
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["networks"], serde_json::json!([]));
        let resp = client
            .post(format!("{base}/network/reply"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"network_id": "n1", "decision": "deny"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);

        // Config: set → get roundtrip.
        let resp = client
            .post(format!("{base}/config/set"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"config": {"model": "qwen3.8", "nested": {"a": [1, 2]}}}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let resp = client
            .get(format!("{base}/config/get"))
            .header("x-faktor-server-password", pw.as_str())
            .send()
            .await
            .unwrap();
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["config"]["model"], "qwen3.8");
        assert_eq!(body["config"]["nested"]["a"], serde_json::json!([1, 2]));

        // Oversized config is rejected (bounded everything).
        let big = "x".repeat(1024 * 1024 + 1);
        let resp = client
            .post(format!("{base}/config/set"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"config": {"blob": big}}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 413);

        // Config is still the previous value after the rejection.
        let resp = client
            .get(format!("{base}/config/get"))
            .header("x-faktor-server-password", pw.as_str())
            .send()
            .await
            .unwrap();
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["config"]["model"], "qwen3.8");

        // All three areas require auth.
        for (method, path) in [
            ("get", "/config/get"),
            ("get", "/question/list"),
            ("get", "/network/list"),
        ] {
            let resp = if method == "get" {
                client.get(format!("{base}{path}")).send().await.unwrap()
            } else {
                unreachable!()
            };
            assert_eq!(resp.status(), 401, "{path}");
        }
        let _ = handle.shutdown.send(());
    }

    fn frame_id_containing(buf: &str, needle: &str) -> Option<u64> {
        for frame in buf.split("\n\n") {
            if !frame.contains(needle) {
                continue;
            }
            for line in frame.lines() {
                if let Some(id) = line.strip_prefix("id: ") {
                    return id.trim().parse().ok();
                }
            }
        }
        None
    }

    fn parse_global_frames(buf: &str) -> Vec<(u64, GlobalEvent)> {
        buf.split("\n\n")
            .filter_map(GlobalEvent::from_frame)
            .collect()
    }

    fn test_deps(root: &std::path::Path) -> ServerDeps {
        test_deps_with(root, vec![])
    }

    fn test_deps_with(
        root: &std::path::Path,
        extra_providers: Vec<Arc<dyn faktor_provider::Provider>>,
    ) -> ServerDeps {
        let mut registry = faktor_provider::ProviderRegistry::new();
        registry
            .try_register(Arc::new(FakeProvider::with_script(
                "fake",
                ModelCapabilities {
                    tools: true,
                    ..Default::default()
                },
                vec![
                    faktor_provider::ScriptedResponse::Text("pong".into()),
                    faktor_provider::ScriptedResponse::End,
                ],
            )))
            .unwrap();
        for p in extra_providers {
            registry.try_register(p).unwrap();
        }
        let session = SessionManager::open(root.join("store"), root.join("cas"), true).unwrap();
        let permissions = ChannelPermissionRequester::new(Duration::from_secs(5));
        let agent = AgentRuntime::new(faktor_agent::AgentDeps {
            session: session.clone(),
            providers: Arc::new(registry),
            chunk_sink: None,
            permission_requester: permissions.clone(),
            evidence: Arc::new(faktor_agent::NoEvidence),
            tools: Arc::new(faktor_agent::ToolRegistry::new()),
            cas: None,
            workspaces: faktor_fs::WorkspaceFileService::new(),
            edit: None,
            snapshots: None,
            sandbox: None,
            supervisor: None,
            verification: faktor_agent::VerificationService::disabled(),
            model: "m".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "You are a test server agent.".into(),
            hooks: None,
            instructions_resolver: faktor_instructions::no_roots_resolver(),
            routing: faktor_agent::FixedRoutingPolicy::passthrough(),
            budgets: Arc::new(faktor_session::NoopBudget),
            clock: Arc::new(faktor_core::time::SystemClock),
            tool_call_mode: faktor_agent::ToolCallMode::Native,
            tool_deadline_ms: 2000,
            retry_policy: faktor_core::retry::RetryPolicy::default(),
            semantic: faktor_agent::fallback_semantic_registry(),
            context_prior: None,
            efficiency: Default::default(),
        })
        .unwrap();
        let (orchestrator, tasks) = orch_pair(session.clone(), agent.clone());
        ServerDeps {
            budgets: faktor_session::DurableBudgetLedger::new(session.clone()),
            session,
            agent,
            permissions,
            orchestrator,
            tasks,
            auth_token: AuthToken::generate(),
            server_password: ServerPassword::generate(),
            directory: None,
            version: "0.1.0".into(),
            fs: None,
            snapshots: None,
            chunk_rx: None,
            simulate_not_ready: false,
            evidence: None,
            semantic: None,
        }
    }

    #[tokio::test]
    async fn legacy_prompt_on_unknown_session_is_404_with_real_op_id() {
        // Audit round 8: the legacy prompt answered 200 accepted:true for
        // sessions that do not exist, and the op_id was hardcoded "turn".
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let token = deps.auth_token.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        // Unknown session id.
        let resp = client
            .post(format!("{base}/api/session/999999/prompt"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"prompt": "hi", "files": []}))
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            404,
            "unknown session must 404, never a phantom 200"
        );
        // A real session returns a REAL operation id (never the literal
        // "turn") — abort correlation depends on it.
        let resp = client
            .post(format!("{base}/api/session"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({
                "provider": "fake",
                "model": "m",
                "workspace": "/tmp",
                "title": "t-opid",
            }))
            .send()
            .await
            .unwrap();
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid = created["id"].as_str().unwrap().to_string();
        let resp = client
            .post(format!("{base}/api/session/{sid}/prompt"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"prompt": "second prompt", "files": []}))
            .send()
            .await
            .unwrap();
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["accepted"], true);
        let op = body["op_id"].as_str().unwrap_or("");
        assert!(
            !op.is_empty() && op != "turn",
            "op_id must be real, got {op:?}"
        );
        // The op_id parses as a u64 operation id.
        assert!(op.parse::<u64>().is_ok());
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn sdk_abort_honors_targeted_op_id() {
        // The SDK abort body carries an op_id; aborting one queued prompt
        // must cancel exactly that row and leave the session machine
        // untouched (audit round 8: the field was ignored and abort was
        // always all-ops).
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let token = deps.auth_token.clone();
        let manager = deps.session.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        // Deterministic busy state, handle-side: prompt A lands the machine
        // in Preparing (not PROMPTABLE); prompt B durably queues.
        let ws = manager.create_workspace("/tmp").unwrap();
        let session = manager.create_session(ws, "t-abort", "fake", "m").unwrap();
        let session_id = session.id().to_string();
        let _ = session.submit_prompt("first", &[]).unwrap();
        let second = session.submit_prompt("second", &[]).unwrap();
        assert!(second.queued, "second prompt must queue behind Preparing");
        let op_id = second.op_id.to_string();
        // Targeted abort of the QUEUED prompt via the SDK surface.
        let resp = client
            .post(format!("{base}/session/abort"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"session_id": session_id, "op_id": op_id}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let aborted: serde_json::Value = resp.json().await.unwrap();
        let list = aborted["aborted"].as_array().unwrap();
        assert!(
            list.iter().any(|o| o.as_str() == Some(op_id.as_str())),
            "targeted abort must report the cancelled op: {list:?}"
        );
        // The queued row is durably cancelled; the machine never moved.
        assert_eq!(
            session.state().unwrap(),
            faktor_core::state::AgentState::Preparing,
            "a queued-prompt kill must not touch the state machine"
        );
        assert_eq!(session.queued_prompt_count().unwrap(), 0);
        let _ = handle.shutdown.send(());
    }

    // ------------------------------------------------------------------
    // P0 wire-compat round: the added operations (status aliases, fork,
    // summarize, delete, deleteMessage, question/network over the permission
    // machinery, config update/warnings/overlay, pty rejection, dispose,
    // auth rotation) each do real work and refuse loudly where the runtime
    // cannot honor them.

    #[tokio::test]
    async fn session_get_and_status_aliases_serve_the_state_projection() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let pw = deps.server_password.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let basic = |r: reqwest::RequestBuilder| r.basic_auth("kilo", Some(pw.as_str()));

        let resp = basic(
            client
                .post(format!("{base}/session"))
                .json(&serde_json::json!({"model": {"id": "m", "providerID": "fake"}})),
        )
        .send()
        .await
        .unwrap();
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid = created["sessionID"].as_str().unwrap().to_string();

        // session.get == the summary handler (GET /session/{sessionID}).
        let resp = basic(client.get(format!("{base}/session/{sid}")))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let summary: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(summary["sessionID"], sid);
        assert!(summary["title"].is_string());
        assert!(summary["state"].is_string());

        // /session/{sessionID}/status == the state projection.
        let resp = basic(client.get(format!("{base}/session/{sid}/status")))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let view: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(view["session_id"], sid);
        assert_eq!(view["agent_state"]["state"], "idle");

        // /session/status?session_id= == the same view.
        let resp = basic(client.get(format!("{base}/session/status?session_id={sid}")))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let view2: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(view, view2);

        // Both aliases are loud 404s for unknown sessions.
        for path in [
            "/session/999999/status",
            "/session/status?session_id=999999",
        ] {
            let resp = basic(client.get(format!("{base}{path}")))
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 404, "{path}");
        }
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn fork_copies_history_and_stays_independent() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let pw = deps.server_password.clone();
        let session_mgr = deps.session.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let basic = |r: reqwest::RequestBuilder| r.basic_auth("kilo", Some(pw.as_str()));

        // Source session with one completed exchange.
        let resp = basic(
            client
                .post(format!("{base}/session"))
                .json(&serde_json::json!({
                    "title": "orig",
                    "model": {"id": "m", "providerID": "fake"}
                })),
        )
        .send()
        .await
        .unwrap();
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid = created["sessionID"].as_str().unwrap().to_string();
        let resp = basic(client.post(format!("{base}/session/{sid}/message")).json(
            &serde_json::json!({
                "model": {"providerID": "fake", "modelID": "m"},
                "parts": [{"type": "text", "text": "hello fork"}],
            }),
        ))
        .send()
        .await
        .unwrap();
        assert_eq!(resp.status(), 200);

        let source_page = || async {
            let resp = basic(client.get(format!("{base}/session/{sid}/message?limit=100")))
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 200);
            resp.json::<serde_json::Value>().await.unwrap()
        };
        let before = source_page().await;
        assert!(before.as_array().unwrap().len() >= 2, "{before}");

        // Fork: a NEW session titled "<orig> (fork)".
        let resp = basic(client.post(format!("{base}/session/{sid}/fork")))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "{:?}", resp.text().await);
        let forked: serde_json::Value = resp.json().await.unwrap();
        let fork_sid = forked["sessionID"].as_str().unwrap().to_string();
        assert_ne!(fork_sid, sid);
        assert_eq!(forked["title"], "orig (fork)");
        assert!(forked["createdMs"].as_i64().unwrap() > 0);

        // The fork's message array equals the source's once the ids that
        // differ BY CONSTRUCTION are normalized (the fork is a new session
        // with its own sessionID and its own row createdMs): same messages,
        // same order, same parts.
        let resp = basic(client.get(format!("{base}/session/{fork_sid}/message?limit=100")))
            .send()
            .await
            .unwrap();
        let after = resp.json::<serde_json::Value>().await.unwrap();
        assert_eq!(
            normalize_page(&after),
            normalize_page(&before),
            "fork history must equal the source's"
        );

        // Independence: new messages on the ORIGINAL never appear on the
        // fork (the fake provider's script is one-shot, so the new message
        // is appended durably handle-side, exactly like a turn would).
        let original = session_mgr
            .get_session(parse_session_id(&sid).unwrap())
            .unwrap()
            .unwrap();
        let mid = original
            .put_message(
                original.proposed_message_seq().unwrap(),
                "user",
                serde_json::json!({"text": "third turn"}),
            )
            .unwrap();
        original.put_text_part(mid, "direct text").unwrap();
        let grown = source_page().await;
        assert!(grown.as_array().unwrap().len() > before.as_array().unwrap().len());
        let resp = basic(client.get(format!("{base}/session/{fork_sid}/message?limit=100")))
            .send()
            .await
            .unwrap();
        assert_eq!(
            normalize_page(&resp.json::<serde_json::Value>().await.unwrap()),
            normalize_page(&after),
            "the fork must not see the original's new messages"
        );
        let _ = handle.shutdown.send(());
    }

    /// Drop the ids that differ by construction between a session and its
    /// fork (info.sessionID and the row createdMs) for equality checks.
    fn normalize_page(page: &serde_json::Value) -> serde_json::Value {
        let mut out = page.clone();
        if let Some(arr) = out.as_array_mut() {
            for entry in arr {
                if let Some(info) = entry["info"].as_object_mut() {
                    info.remove("sessionID");
                    info.remove("createdMs");
                }
            }
        }
        out
    }

    #[tokio::test]
    async fn fork_unknown_session_is_404() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let pw = deps.server_password.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let resp = client
            .post(format!("{base}/session/999999/fork"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn summarize_returns_a_bounded_digest() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let pw = deps.server_password.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        let resp = client
            .post(format!("{base}/session"))
            .basic_auth("kilo", Some(pw.as_str()))
            .json(&serde_json::json!({
                "title": "digest me",
                "model": {"id": "m", "providerID": "fake"}
            }))
            .send()
            .await
            .unwrap();
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid = created["sessionID"].as_str().unwrap().to_string();

        // Empty session: bounded digest still answers.
        let resp = client
            .post(format!("{base}/session/{sid}/summarize"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["sessionID"], sid);
        assert_eq!(body["title"], "digest me");
        assert!(body["summary"].is_string());

        // After a turn the summary digests the newest messages' text.
        let resp = client
            .post(format!("{base}/session/{sid}/message"))
            .basic_auth("kilo", Some(pw.as_str()))
            .json(&serde_json::json!({
                "model": {"providerID": "fake", "modelID": "m"},
                "parts": [{"type": "text", "text": "summarize this"}],
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let resp = client
            .post(format!("{base}/session/{sid}/summarize"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        let body: serde_json::Value = resp.json().await.unwrap();
        let summary = body["summary"].as_str().unwrap();
        assert!(summary.contains("summarize this"), "{summary}");
        assert!(summary.contains("pong"), "{summary}");
        // Bounded: never a huge blob.
        assert!(summary.len() < 16 * 1024);
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn session_delete_refuses_mid_turn_and_ends_durably() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let pw = deps.server_password.clone();
        let manager = deps.session.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        // A busy session (handle-side submit → Preparing, no driver): DELETE
        // must refuse with an explicit 409, never silently "succeed".
        let ws = manager.create_workspace("/tmp").unwrap();
        let busy = manager.create_session(ws, "busy", "fake", "m").unwrap();
        let busy_id = busy.id().to_string();
        busy.submit_prompt("first", &[]).unwrap();
        assert!(busy.state().unwrap().is_active());
        let resp = client
            .delete(format!("{base}/session/{busy_id}"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 409, "mid-turn delete must be refused");
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["ok"], false);
        assert!(body["message"].as_str().unwrap().contains("mid-turn"));

        // An idle session deletes: durable end (lifecycle Closed, state
        // Completed), prompts refused afterwards.
        let resp = client
            .post(format!("{base}/session"))
            .basic_auth("kilo", Some(pw.as_str()))
            .json(&serde_json::json!({"model": {"id": "m", "providerID": "fake"}}))
            .send()
            .await
            .unwrap();
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid: u64 = created["sessionID"].as_str().unwrap().parse().unwrap();
        let resp = client
            .delete(format!("{base}/session/{sid}"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["ok"], true);
        let row = manager
            .get_session(faktor_core::id::SessionId::new(sid))
            .unwrap()
            .unwrap()
            .row()
            .unwrap();
        assert!(row.lifecycle.is_terminal(), "durable Closed tombstone");
        assert_eq!(row.state, faktor_core::state::AgentState::Completed);
        // Prompts on the deleted session are refused (never a phantom run).
        let resp = client
            .post(format!("{base}/session/{sid}/message"))
            .basic_auth("kilo", Some(pw.as_str()))
            .json(&serde_json::json!({
                "model": {"providerID": "fake", "modelID": "m"},
                "parts": [{"type": "text", "text": "nope"}],
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 409, "deleted sessions refuse prompts");
        // Double delete is a loud conflict, and the tombstone is durable
        // across a manager reopen.
        let resp = client
            .delete(format!("{base}/session/{sid}"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 409);
        drop(manager);
        let reopened =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let row = reopened
            .get_session(faktor_core::id::SessionId::new(sid))
            .unwrap()
            .unwrap()
            .row()
            .unwrap();
        assert!(row.lifecycle.is_terminal(), "Closed survives reopen");
        // Unknown session delete → 404.
        let resp = client
            .delete(format!("{base}/session/999999"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn delete_message_refuses_dependencies_and_removes_durably() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let pw = deps.server_password.clone();
        let manager = deps.session.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        // Seed rows directly: seq1 user, seq2 assistant with a tool_call,
        // seq3 assistant with the tool_result referencing call c1, seq4
        // plain text.
        let ws = manager.create_workspace("/tmp").unwrap();
        let s = manager.create_session(ws, "t-del", "fake", "m").unwrap();
        let sid = s.id().to_string();
        let store = manager.store();
        store
            .put_message(s.id(), 1, "user", serde_json::json!({"text": "run tools"}))
            .unwrap();
        let m2 = store
            .put_message(s.id(), 2, "assistant", serde_json::json!({"parts": []}))
            .unwrap();
        store
            .put_part(
                m2,
                "tool_call",
                serde_json::json!({
                    "tool_call_id": "c1",
                    "name": "echo",
                    "input": {"x": 1},
                    "state": "completed"
                }),
            )
            .unwrap();
        let m3 = store
            .put_message(s.id(), 3, "assistant", serde_json::json!({"parts": []}))
            .unwrap();
        store
            .put_part(
                m3,
                "tool_result",
                serde_json::json!({"tool_call_id": "c1", "excerpt": "out"}),
            )
            .unwrap();
        store
            .put_message(s.id(), 4, "user", serde_json::json!({"text": "plain"}))
            .unwrap();

        // Unknown message → 404; malformed id → explicit refusal.
        let resp = client
            .delete(format!("{base}/session/{sid}/message/99"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);
        let resp = client
            .delete(format!("{base}/session/{sid}/message/abc"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 409);

        // The tool-result message has a dependency → refused with the clear
        // dependency error.
        let resp = client
            .delete(format!("{base}/session/{sid}/message/3"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 409);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["ok"], false);
        assert!(
            body["message"]
                .as_str()
                .unwrap()
                .contains("tool-result dependencies"),
            "{body}"
        );
        // The tool-call message is referenced by that result → same refusal.
        let resp = client
            .delete(format!("{base}/session/{sid}/message/2"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 409);
        assert_eq!(store.message_count(s.id()).unwrap(), 4);

        // A dependency-free message is removed DURABLY: {ok:true}, the row
        // and its parts are gone, and the surviving sequences are stable.
        let resp = client
            .delete(format!("{base}/session/{sid}/message/1"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "{:?}", resp.text().await);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["ok"], true);
        assert_eq!(store.message_count(s.id()).unwrap(), 3);
        assert_eq!(store.message_created_ms(s.id(), 1).unwrap(), None);
        // Surviving rows keep their sequences (2, 3, 4); a second delete of
        // the same message is an honest 404.
        let resp = client
            .delete(format!("{base}/session/{sid}/message/1"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);
        let resp = client
            .delete(format!("{base}/session/{sid}/message/4"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        // The deleted message no longer appears in the wire page.
        let resp = client
            .get(format!("{base}/session/{sid}/message"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let page: serde_json::Value = resp.json().await.unwrap();
        let seqs: Vec<&str> = page
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|e| e["info"]["messageID"].as_str())
            .collect();
        assert_eq!(seqs, vec!["3", "2"], "rows removed, seqs stable: {seqs:?}");
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn delete_message_refuses_in_flight_newest_message() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let pw = deps.server_password.clone();
        let manager = deps.session.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let ws = manager.create_workspace("/tmp").unwrap();
        let s = manager
            .create_session(ws, "t-inflight", "fake", "m")
            .unwrap();
        let sid = s.id().to_string();
        // An active turn whose assistant reply (the newest message) is
        // mid-stream.
        s.submit_prompt("stream me", &[]).unwrap();
        // The prompt materializes at seq 2; the streaming assistant reply
        // (the newest message, identity = durable seq) is seq 3.
        let mid = s
            .put_message(3, "assistant", serde_json::json!({"parts": []}))
            .unwrap();
        s.put_text_part(mid, "partial").unwrap();
        s.append_event(
            faktor_core::event::EventKind::ContextPrepared,
            faktor_core::state::AgentState::BuildingContext,
            None,
            None,
        )
        .unwrap();
        s.append_event(
            faktor_core::event::EventKind::ModelStarted,
            faktor_core::state::AgentState::WaitingForModel,
            None,
            None,
        )
        .unwrap();
        s.append_event(
            faktor_core::event::EventKind::ModelChunkReceived,
            faktor_core::state::AgentState::Streaming,
            None,
            None,
        )
        .unwrap();
        assert!(s.state().unwrap().is_active());
        let resp = client
            .delete(format!("{base}/session/{sid}/message/3"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 409);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(
            body["message"].as_str().unwrap().contains("in flight"),
            "{body}"
        );
        assert_eq!(s.message_count().unwrap(), 2, "nothing was removed");
        // The just-streamed message is gone from the wire page only AFTER
        // the turn is over; while active it stays.
        assert!(s.state().unwrap().is_active());
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn session_update_persists_title_durably() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let pw = deps.server_password.clone();
        let manager = deps.session.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let ws = manager.create_workspace("/tmp").unwrap();
        let s = manager
            .create_session(ws, "orig title", "fake", "m")
            .unwrap();
        let sid = s.id().to_string();

        // Rename via the wire surface.
        let resp = client
            .post(format!("{base}/session/{sid}"))
            .basic_auth("kilo", Some(pw.as_str()))
            .json(&serde_json::json!({"title": "renamed by wire"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "{:?}", resp.text().await);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["sessionID"], sid);
        assert_eq!(body["title"], "renamed by wire");
        assert!(body["updatedMs"].as_i64().unwrap() > 0);
        // The GET summary reads the durable row.
        let resp = client
            .get(format!("{base}/session/{sid}"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        let summary: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(summary["title"], "renamed by wire");
        assert_eq!(s.title().unwrap(), "renamed by wire");
        // Control characters are stripped by the session layer.
        let resp = client
            .post(format!("{base}/session/{sid}"))
            .basic_auth("kilo", Some(pw.as_str()))
            .json(&serde_json::json!({"title": "clean\n\tname\u{7f}done"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["title"], "cleannamedone");
        // Hostile titles refuse.
        let resp = client
            .post(format!("{base}/session/{sid}"))
            .basic_auth("kilo", Some(pw.as_str()))
            .json(&serde_json::json!({"title": "\n\r\u{0}"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400, "control-only title refuses");
        let long = "x".repeat(300);
        let resp = client
            .post(format!("{base}/session/{sid}"))
            .basic_auth("kilo", Some(pw.as_str()))
            .json(&serde_json::json!({"title": long}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 413, "oversized title refuses");
        // Unknown fields (the per-turn envelope) are protocol drift.
        // The per-turn envelope fields (model/provider) are protocol drift:
        // the strict DTO rejects them (the wire client never sends them).
        let resp = client
            .post(format!("{base}/session/{sid}"))
            .basic_auth("kilo", Some(pw.as_str()))
            .json(&serde_json::json!({"title": "x", "model": {"id": "m"}}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 422, "{:?}", resp.text().await);
        assert_eq!(
            s.title().unwrap(),
            "cleannamedone",
            "nothing hostile landed"
        );
        // Unknown session → 404.
        let resp = client
            .post(format!("{base}/session/9999"))
            .basic_auth("kilo", Some(pw.as_str()))
            .json(&serde_json::json!({"title": "x"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);
        // The durable row keeps the last good title after a full reopen of
        // the manager on the SAME data dir.
        drop(handle);
        let m2 =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let row = m2.get_session(s.id()).unwrap().unwrap().row().unwrap();
        assert_eq!(row.title, "cleannamedone", "title persists across reopen");
    }

    #[tokio::test]
    async fn question_and_network_ops_resolve_pending_permissions() {
        let dir = tempfile::tempdir().unwrap();
        let mut registry = faktor_provider::ProviderRegistry::new();
        registry
            .try_register(Arc::new(FakeProvider::with_script(
                "fake",
                ModelCapabilities {
                    tools: true,
                    ..Default::default()
                },
                vec![
                    faktor_provider::ScriptedResponse::ToolCall {
                        id: "c1".into(),
                        name: "echo".into(),
                        input: serde_json::json!({"x": 1}),
                    },
                    faktor_provider::ScriptedResponse::ToolCall {
                        id: "c2".into(),
                        name: "curl".into(),
                        input: serde_json::json!({"url": "https://example.com"}),
                    },
                    faktor_provider::ScriptedResponse::End,
                ],
            )))
            .unwrap();
        let session =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let permissions = ChannelPermissionRequester::new(Duration::from_secs(5));
        let mut tools = faktor_agent::ToolRegistry::new();
        tools.register(faktor_agent::Tool {
            name: "echo".into(),
            description: "d".into(),
            input_schema: serde_json::json!({}),
            resource_class: faktor_core::resource::ResourceClass::Cpu,
            capability: None,
            recovery_hint: faktor_agent::RecoveryHint::Idempotent,
            path_args: vec![],
            execute: Arc::new(|_ctx, _args| {
                Box::pin(async move { Ok(faktor_agent::ToolOutcome::default()) })
            }),
        });
        // A REAL network capability request (Capability::Network) — the
        // frozen network surface maps to these.
        tools.register(faktor_agent::Tool {
            name: "curl".into(),
            description: "d".into(),
            input_schema: serde_json::json!({}),
            resource_class: faktor_core::resource::ResourceClass::Cpu,
            capability: Some(faktor_core::capability::Capability::Network {
                destination: "https://example.com".into(),
            }),
            recovery_hint: faktor_agent::RecoveryHint::UnknownEffect,
            path_args: vec![],
            execute: Arc::new(|_ctx, _args| {
                Box::pin(async move { Ok(faktor_agent::ToolOutcome::default()) })
            }),
        });
        let agent = AgentRuntime::new(faktor_agent::AgentDeps {
            session: session.clone(),
            providers: Arc::new(registry),
            chunk_sink: None,
            permission_requester: permissions.clone(),
            evidence: Arc::new(faktor_agent::NoEvidence),
            tools: Arc::new(tools),
            cas: None,
            workspaces: faktor_fs::WorkspaceFileService::new(),
            edit: None,
            snapshots: None,
            sandbox: None,
            supervisor: None,
            verification: faktor_agent::VerificationService::disabled(),
            model: "m".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "You are a test server agent.".into(),
            hooks: None,
            instructions_resolver: faktor_instructions::no_roots_resolver(),
            routing: faktor_agent::FixedRoutingPolicy::passthrough(),
            budgets: Arc::new(faktor_session::NoopBudget),
            clock: Arc::new(faktor_core::time::SystemClock),
            tool_call_mode: faktor_agent::ToolCallMode::Native,
            tool_deadline_ms: 2000,
            retry_policy: faktor_core::retry::RetryPolicy::default(),
            semantic: faktor_agent::fallback_semantic_registry(),
            context_prior: None,
            efficiency: Default::default(),
        })
        .unwrap();
        let (orchestrator, tasks) = orch_pair(session.clone(), agent.clone());
        let deps = ServerDeps {
            budgets: faktor_session::DurableBudgetLedger::new(session.clone()),
            session: session.clone(),
            agent,
            permissions: permissions.clone(),
            orchestrator,
            tasks,
            auth_token: AuthToken::generate(),
            server_password: ServerPassword::generate(),
            directory: None,
            version: "0.1.0".into(),
            fs: None,
            snapshots: None,
            chunk_rx: None,
            simulate_not_ready: false,
            evidence: None,
            semantic: None,
        };
        let pw = deps.server_password.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        let resp = client
            .post(format!("{base}/session/create"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"provider": "fake", "model": "m"}))
            .send()
            .await
            .unwrap();
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid = created["id"].as_str().unwrap().to_string();
        // Non-blocking prompt: the turn parks on the two permission hops.
        client
            .post(format!("{base}/session/prompt"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"session_id": sid, "prompt": "network please"}))
            .send()
            .await
            .unwrap();

        // The shell-class request surfaces under /question/list; the
        // network-class one under /network/list — never mixed. The tool
        // batch requests permissions SEQUENTIALLY, so the shell question
        // parks first and the network request only parks after it resolves.
        let mut question_id = None;
        for _ in 0..100 {
            let resp = client
                .get(format!("{base}/question/list?session_id={sid}"))
                .header("x-faktor-server-password", pw.as_str())
                .send()
                .await
                .unwrap();
            let list: serde_json::Value = resp.json().await.unwrap();
            for q in list["questions"].as_array().unwrap() {
                assert_ne!(q["capability"], "network", "shell class only: {q}");
                assert_eq!(q["session_id"], sid);
                if q["capability"] == "execute_shell" {
                    question_id = q["id"].as_str().map(String::from);
                }
            }
            if question_id.is_some() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let question_id = question_id.expect("shell permission must surface as a question");
        // Nothing is pending on the network surface yet (the shell request
        // parks BEFORE the batch reaches the network call).
        let resp = client
            .get(format!("{base}/network/list"))
            .header("x-faktor-server-password", pw.as_str())
            .send()
            .await
            .unwrap();
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["networks"], serde_json::json!([]));

        // Cross-class attempts are unknown on the other surface (404), and
        // unknown ids stay 404.
        let resp = client
            .post(format!("{base}/question/reply"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"question_id": "q1", "decision": "allow"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);

        // question.reply (allow) resolves the shell hop for real.
        let resp = client
            .post(format!("{base}/question/reply"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"question_id": question_id, "decision": "allow"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["ok"], true);

        // With the shell hop granted, the batch reaches the network call:
        // it parks on the network surface with the exact capability tag.
        let mut network_id = None;
        for _ in 0..100 {
            let resp = client
                .get(format!("{base}/network/list?session_id={sid}"))
                .header("x-faktor-server-password", pw.as_str())
                .send()
                .await
                .unwrap();
            let list: serde_json::Value = resp.json().await.unwrap();
            for n in list["networks"].as_array().unwrap() {
                assert_eq!(n["capability"], "network", "network class only: {n}");
                assert_eq!(n["session_id"], sid);
                network_id = n["id"].as_str().map(String::from);
            }
            if network_id.is_some() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let network_id = network_id.expect("network permission must surface as a network");
        // The shell permission is NOT a network: cross-class 404.
        let resp = client
            .post(format!("{base}/network/reply"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"network_id": question_id, "decision": "deny"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404, "shell ids are not network requests");

        // network.reject is deny, and the network hop is resolved.
        let resp = client
            .post(format!("{base}/network/reject"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"network_id": network_id}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        // A resolved id is no longer pending: a second attempt is a loud
        // 404 (the id is unknown to the open-request set — same semantics
        // as the reply surface), never a silent double-deny.
        let resp = client
            .post(format!("{base}/network/reject"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"network_id": network_id}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404, "resolved ids leave the pending set");

        // Both lists drain as the hops resolve.
        let mut drained = false;
        for _ in 0..100 {
            let resp = client
                .get(format!("{base}/question/list"))
                .header("x-faktor-server-password", pw.as_str())
                .send()
                .await
                .unwrap();
            let q: serde_json::Value = resp.json().await.unwrap();
            let resp = client
                .get(format!("{base}/network/list"))
                .header("x-faktor-server-password", pw.as_str())
                .send()
                .await
                .unwrap();
            let n: serde_json::Value = resp.json().await.unwrap();
            if q["questions"].as_array().unwrap().is_empty()
                && n["networks"].as_array().unwrap().is_empty()
            {
                drained = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(drained, "resolved permissions must leave both lists");
        // The denied network tool made the turn end (deny returns the
        // machine to a non-busy landing state); nothing is left pending.
        let mut done = false;
        for _ in 0..100 {
            let st = session
                .get_session(faktor_core::id::SessionId::new(sid.parse().unwrap()))
                .unwrap()
                .unwrap()
                .state()
                .unwrap();
            if !turn_machine_busy(st) {
                done = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(done, "turn must finish after both hops resolved");
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn config_update_warnings_overlay_and_overlay_update() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let pw = deps.server_password.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        // update applies ONLY the daemon-editable keys onto the store.
        let resp = client
            .post(format!("{base}/config/update"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"config": {
                "model": "qwen3.8",
                "compact_at_usage": 0.8,
                "instructions": "be brief"
            }}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        // A second update merges, preserving earlier keys.
        let resp = client
            .post(format!("{base}/config/update"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"config": {"model": "gpt-x"}}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let resp = client
            .get(format!("{base}/config/get"))
            .header("x-faktor-server-password", pw.as_str())
            .send()
            .await
            .unwrap();
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["config"]["model"], "gpt-x");
        assert_eq!(body["config"]["compact_at_usage"], 0.8);
        assert_eq!(body["config"]["instructions"], "be brief");

        // Provider keys are NOT daemon-editable: clear 400, nothing applied.
        let resp = client
            .post(format!("{base}/config/update"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"config": {"providers": {"ollama": {}}}}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("not daemon-editable"),
            "{body}"
        );
        // Warnings: a full-replace config/set can smuggle anything in; the
        // warning surface reports it instead of silently accepting.
        let resp = client
            .post(format!("{base}/config/set"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"config": {
                "compact_at_usage": 7,
                "model": 5,
                "smuggled_key": true
            }}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let resp = client
            .get(format!("{base}/config/warnings"))
            .header("x-faktor-server-password", pw.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        let warnings = body["warnings"].as_array().unwrap();
        assert!(
            warnings
                .iter()
                .any(|w| w.as_str().unwrap().contains("compact_at_usage")),
            "{warnings:?}"
        );
        assert!(
            warnings
                .iter()
                .any(|w| w.as_str().unwrap().contains("smuggled_key")),
            "{warnings:?}"
        );
        // A valid config warns about nothing (overlay = full replace, so no
        // smuggled key survives from the previous config/set).
        let resp = client
            .post(format!("{base}/config/overlay"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"config": {
                "model": "m", "compact_at_usage": 0.5, "instructions": "i"
            }}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let resp = client
            .get(format!("{base}/config/warnings"))
            .header("x-faktor-server-password", pw.as_str())
            .send()
            .await
            .unwrap();
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["warnings"], serde_json::json!([]));

        // overlay replaces the whole view; overlayUpdate merges into it.
        let resp = client
            .post(format!("{base}/config/overlay"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"config": {"a": 1}}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let resp = client
            .post(format!("{base}/config/overlayUpdate"))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"config": {"b": 2}}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let resp = client
            .get(format!("{base}/config/get"))
            .header("x-faktor-server-password", pw.as_str())
            .send()
            .await
            .unwrap();
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["config"], serde_json::json!({"a": 1, "b": 2}));
        // Non-object configs are malformed on every apply surface.
        for path in ["/config/update", "/config/overlay", "/config/overlayUpdate"] {
            let resp = client
                .post(format!("{base}{path}"))
                .header("x-faktor-server-password", pw.as_str())
                .json(&serde_json::json!({"config": [1, 2]}))
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 400, "{path}");
        }
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn pty_lifecycle_through_the_wire() {
        // Audit round 11: PTYs are real on Unix — create/update(write+
        // resize)/output/remove round-trip through the HTTP surface. The
        // old explicit-409 test is replaced by this one; non-Unix keeps
        // the honest refusal path in the handler itself.
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let pw = deps.server_password.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let auth = |r: reqwest::RequestBuilder| r.header("x-faktor-server-password", pw.as_str());
        // Create a shell that echoes a typed line back.
        let resp = auth(
            client
                .post(format!("{base}/pty/create"))
                .json(&serde_json::json!({
                    "command": "sh",
                    "args": ["-c", "stty -echo; read x; echo out:$x; sleep 2"],
                    "cols": 80,
                    "rows": 24,
                })),
        )
        .send()
        .await
        .unwrap();
        #[cfg(unix)]
        {
            assert_eq!(resp.status(), 200, "pty/create must succeed on unix");
            let created: serde_json::Value = resp.json().await.unwrap();
            let pty_id = created["pty_id"].as_str().unwrap().to_string();
            assert!(created["pid"].as_u64().unwrap() > 0);
            // Write input + resize in one update.
            let resp = auth(
                client
                    .post(format!("{base}/pty/update"))
                    .json(&serde_json::json!({
                        "pty_id": pty_id,
                        "data": "hello wire\n",
                        "rows": 33,
                        "cols": 121,
                    })),
            )
            .send()
            .await
            .unwrap();
            assert_eq!(resp.status(), 200);
            // Poll the output snapshot until the echo arrives.
            let mut saw = false;
            for _ in 0..100 {
                let resp = auth(client.get(format!("{base}/pty/{pty_id}/output")))
                    .send()
                    .await
                    .unwrap();
                let body: serde_json::Value = resp.json().await.unwrap();
                let out = body["output"].as_str().unwrap_or("");
                if out.contains("out:hello wire") {
                    saw = true;
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
            assert!(saw, "the pty must echo the wire input back");
            // Remove kills and cleans up; second remove stays idempotent.
            let resp = auth(
                client
                    .post(format!("{base}/pty/remove"))
                    .json(&serde_json::json!({"pty_id": pty_id})),
            )
            .send()
            .await
            .unwrap();
            assert_eq!(resp.status(), 200);
            let resp = auth(
                client
                    .post(format!("{base}/pty/remove"))
                    .json(&serde_json::json!({"pty_id": pty_id})),
            )
            .send()
            .await
            .unwrap();
            assert_eq!(resp.status(), 200);
            // Unknown pty output is a loud 404.
            let resp = auth(client.get(format!("{base}/pty/999999/output")))
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 404);
        }
        #[cfg(not(unix))]
        {
            // Windows now has a REAL ConPTY backend (journaled, session-
            // owned, lifecycle-tested in faktor-pty): creation succeeds
            // exactly like Unix, and the remove path above proves the
            // session-owned teardown. The old 409 expectation encoded the
            // pre-ConPTY era and is deliberately gone.
            assert_eq!(
                resp.status(),
                200,
                "ConPTY is a real implementation: creation must succeed"
            );
        }
        let _ = handle.shutdown.send(());
    }
    #[tokio::test]
    async fn dispose_ends_all_sessions_and_reload_acknowledges() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let pw = deps.server_password.clone();
        let session = deps.session.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        let resp = client
            .post(format!("{base}/session"))
            .basic_auth("kilo", Some(pw.as_str()))
            .json(&serde_json::json!({"model": {"id": "m", "providerID": "fake"}}))
            .send()
            .await
            .unwrap();
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid: u64 = created["sessionID"].as_str().unwrap().parse().unwrap();
        // One completed exchange so dispose has real sessions to end.
        let resp = client
            .post(format!("{base}/session/{sid}/message"))
            .basic_auth("kilo", Some(pw.as_str()))
            .json(&serde_json::json!({
                "model": {"providerID": "fake", "modelID": "m"},
                "parts": [{"type": "text", "text": "hi"}],
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

        // instance.reload re-runs daemon recovery (idempotent) → ok.
        let resp = client
            .post(format!("{base}/instance/reload"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["ok"], true);

        // global.dispose ends every session durably.
        let resp = client
            .post(format!("{base}/global/dispose"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["ok"], true);
        let row = session
            .get_session(faktor_core::id::SessionId::new(sid))
            .unwrap()
            .unwrap()
            .row()
            .unwrap();
        assert!(row.lifecycle.is_terminal(), "dispose ends sessions durably");
        assert_eq!(row.state, faktor_core::state::AgentState::Completed);
        // A second dispose over zero live sessions still answers ok.
        let resp = client
            .post(format!("{base}/instance/dispose"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn auth_set_rotates_the_password_and_remove_restores_env_password() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let startup_pw = deps.server_password.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        let new_pw = "x".repeat(64);
        // Rotate to an explicit secret.
        let resp = client
            .post(format!("{base}/auth/set"))
            .basic_auth("kilo", Some(startup_pw.as_str()))
            .json(&serde_json::json!({"password": new_pw}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["ok"], true);
        assert_eq!(body["password"], new_pw);

        // The OLD password is rejected everywhere; the new one works.
        let resp = client
            .get(format!("{base}/global/health"))
            .basic_auth("kilo", Some(startup_pw.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            401,
            "old password must be rejected after set"
        );
        let resp = client
            .get(format!("{base}/global/health"))
            .basic_auth("kilo", Some(new_pw.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        // Rotating without a password generates a fresh secret (returned).
        let resp = client
            .post(format!("{base}/auth/set"))
            .basic_auth("kilo", Some(new_pw.as_str()))
            .json(&serde_json::json!({"password": null}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        let rotated = body["password"].as_str().unwrap().to_string();
        assert_ne!(rotated, new_pw);
        assert_eq!(rotated.len(), 64);
        // Malformed passwords (empty / oversized) are 400s.
        let resp = client
            .post(format!("{base}/auth/set"))
            .basic_auth("kilo", Some(rotated.as_str()))
            .json(&serde_json::json!({"password": ""}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);
        // auth.remove returns to the STARTUP env password.
        let resp = client
            .post(format!("{base}/auth/remove"))
            .basic_auth("kilo", Some(rotated.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let resp = client
            .get(format!("{base}/global/health"))
            .basic_auth("kilo", Some(rotated.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401, "rotated password dies with remove");
        let resp = client
            .get(format!("{base}/global/health"))
            .basic_auth("kilo", Some(startup_pw.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "env password semantics restored");
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn queued_message_send_is_202_with_an_empty_assistant_placeholder() {
        // A message accepted behind an active logical turn queues durably:
        // the response is HTTP 202 + the standard {info, parts} shape with
        // empty parts and an empty messageID (nothing is materialized yet —
        // documented choice; the frozen DTO rejects an extra queued flag).
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let pw = deps.server_password.clone();
        let manager = deps.session.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        // Busy session: prompt A lands the machine in Preparing (no driver).
        let ws = manager.create_workspace("/tmp").unwrap();
        let busy = manager.create_session(ws, "t-queue", "fake", "m").unwrap();
        let sid = busy.id().to_string();
        busy.submit_prompt("first", &[]).unwrap();
        assert!(busy.state().unwrap().is_active());

        let resp = client
            .post(format!("{base}/session/{sid}/message"))
            .basic_auth("kilo", Some(pw.as_str()))
            .json(&serde_json::json!({
                "model": {"providerID": "fake", "modelID": "m"},
                "parts": [{"type": "text", "text": "queued prompt"}],
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 202, "queueing is marked by HTTP 202");
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["info"]["role"], "assistant");
        assert_eq!(body["info"]["sessionID"], sid);
        assert_eq!(body["info"]["messageID"], "", "nothing materialized yet");
        assert_eq!(body["parts"], serde_json::json!([]));
        assert!(
            body.as_object().unwrap().get("queued").is_none(),
            "the frozen DTO carries no queued field: {body}"
        );
        // The prompt really queued durably.
        assert_eq!(busy.queued_prompt_count().unwrap(), 1);
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn message_send_with_no_assistant_content_is_an_honest_502() {
        // A provider that ends cleanly WITHOUT any content produces no
        // durable assistant row: the frozen send shape cannot be built, so
        // the endpoint fails loudly instead of fabricating a message.
        let dir = tempfile::tempdir().unwrap();
        let provider = Arc::new(FakeProvider::with_script(
            "fake",
            ModelCapabilities::default(),
            vec![faktor_provider::ScriptedResponse::End],
        ));
        let deps = recording_wire_deps(dir.path(), provider);
        let pw = deps.server_password.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        let resp = client
            .post(format!("{base}/session"))
            .basic_auth("kilo", Some(pw.as_str()))
            .json(&serde_json::json!({"model": {"id": "m", "providerID": "fake"}}))
            .send()
            .await
            .unwrap();
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid = created["sessionID"].as_str().unwrap().to_string();

        let resp = client
            .post(format!("{base}/session/{sid}/message"))
            .basic_auth("kilo", Some(pw.as_str()))
            .json(&serde_json::json!({
                "model": {"providerID": "fake", "modelID": "m"},
                "parts": [{"type": "text", "text": "say nothing"}],
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 502);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["ok"], false);
        assert!(body["message"]
            .as_str()
            .unwrap()
            .contains("without an assistant reply"));
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn provider_list_serves_real_models_and_capabilities() {
        // The model selector must enumerate what the daemon can ACTUALLY
        // serve: an adapter with configured models lists them with their
        // real capabilities (audit: one fabricated 'default' per provider).
        let dir = tempfile::tempdir().unwrap();
        // Register an OpenAI adapter with two known models on top of the
        // fake test provider.
        let mut caps = std::collections::HashMap::new();
        caps.insert(
            "gpt-x".to_string(),
            faktor_core::model::ModelCapabilities {
                context: 128_000,
                max_output: 16_384,
                tools: true,
                ..Default::default()
            },
        );
        let openai = faktor_openai::OpenAiProvider::build(faktor_openai::OpenAiConfig {
            base_url: "http://127.0.0.1:1/v1".into(),
            api_key: None,
            family: faktor_openai::OpenAiFamily::Chat,
            models: caps,
        });
        let deps = test_deps_with(dir.path(), vec![openai]);
        let token = deps.auth_token.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let resp = client
            .get(format!("{base}/provider/list"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        let providers = body["providers"].as_array().unwrap();
        // The openai adapter entry lists gpt-x (its real model) with the
        // configured context.
        let openai_entry = providers
            .iter()
            .find(|p| p["kind"] == "openai")
            .expect("openai adapter listed");
        let models = openai_entry["models"].as_array().unwrap();
        assert!(
            models
                .iter()
                .any(|m| m["id"] == "gpt-x" && m["capabilities"]["context"] == 128_000),
            "real model with real capabilities: {models:?}"
        );
        let _ = handle.shutdown.send(());
    }

    // ---------------------------------------------------------------- native v1

    #[tokio::test]
    async fn native_projection_idle_session_shape_auth_and_errors() {
        // GET /session/{id}/projection on a session that never ran a turn:
        // the row-backed projection is honest (idle, no task data), every
        // native endpoint demands auth, unknown sessions 404 and malformed
        // ids 400.
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let token = deps.auth_token.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        let resp = client
            .post(format!("{base}/api/session"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({
                "provider": "fake",
                "model": "m",
                "workspace": "/tmp",
                "title": "t-proj",
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid = created["id"].as_str().unwrap().to_string();

        // Native endpoints are auth-required like every daemon route.
        for path in [
            format!("/session/{sid}/projection"),
            "/models".to_string(),
            "/capabilities".to_string(),
        ] {
            let resp = client.get(format!("{base}{path}")).send().await.unwrap();
            assert_eq!(resp.status(), 401, "{path}");
        }

        // Idle projection: row state, no task data yet.
        let resp = client
            .get(format!("{base}/session/{sid}/projection"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["session"]["id"], sid);
        assert_eq!(body["session"]["provider"], "fake");
        assert_eq!(body["session"]["model"], "m");
        assert_eq!(body["session"]["lifecycle"], "open");
        assert_eq!(body["state"]["machine"], "idle");
        assert_eq!(body["state"]["label"], "idle");
        assert_eq!(body["state"]["active"], false);
        assert_eq!(body["state"]["terminal"], false);
        assert!(
            body["activeModel"].is_null(),
            "no turn record before the first turn: {body}"
        );
        assert!(body["activeTool"].is_null(), "nothing running: {body}");
        assert!(body["progress"].is_null());
        assert_eq!(body["filesChanged"], serde_json::json!([]));
        assert_eq!(body["verification"], serde_json::json!([]));
        assert!(
            body["lastCheckpoint"].is_null(),
            "no checkpoint service wired in tests"
        );
        assert!(body["contextUsage"].is_null());
        assert_eq!(body["queued"], 0);
        assert!(
            body["prefixStability"].is_null(),
            "no prefix observation before any driven turn: {body}"
        );

        // Unknown session → 404; non-numeric id → 400.
        let resp = client
            .get(format!("{base}/session/999999/projection"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);
        let resp = client
            .get(format!("{base}/session/abc/projection"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn native_projection_after_driven_turn_reports_ledger_files() {
        // Drive a real turn whose tool call changes a file (write_file →
        // durable ledger changed_files), then assert the projection maps
        // the durable state: ledger files, turn-record model envelope and
        // terminal machine state.
        let dir = tempfile::tempdir().unwrap();
        let mut registry = faktor_provider::ProviderRegistry::new();
        registry
            .try_register(Arc::new(FakeProvider::with_script(
                "fake",
                ModelCapabilities {
                    tools: true,
                    ..Default::default()
                },
                vec![
                    faktor_provider::ScriptedResponse::ToolCall {
                        id: "c1".into(),
                        name: "write_file".into(),
                        input: serde_json::json!({"path": "src/a.txt"}),
                    },
                    faktor_provider::ScriptedResponse::End,
                ],
            )))
            .unwrap();
        let mut tools = faktor_agent::ToolRegistry::new();
        tools.register(faktor_agent::Tool {
            name: "write_file".into(),
            description: "write a file".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"],
            }),
            resource_class: faktor_core::resource::ResourceClass::DiskWrite,
            capability: None,
            recovery_hint: faktor_agent::RecoveryHint::WorkspaceWrite,
            path_args: vec!["path".into()],
            execute: Arc::new(|_ctx, _args| {
                Box::pin(async move {
                    Ok(faktor_agent::ToolOutcome {
                        text: "wrote src/a.txt".into(),
                        exit_code: Some(0),
                        effect_status: faktor_core::op::EffectStatus::Applied,
                        ..Default::default()
                    })
                })
            }),
        });
        let session =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let permissions = ChannelPermissionRequester::new(Duration::from_secs(5));
        let agent = AgentRuntime::new(faktor_agent::AgentDeps {
            session: session.clone(),
            providers: Arc::new(registry),
            chunk_sink: None,
            permission_requester: permissions.clone(),
            evidence: Arc::new(faktor_agent::NoEvidence),
            tools: Arc::new(tools),
            cas: None,
            workspaces: faktor_fs::WorkspaceFileService::new(),
            edit: None,
            snapshots: None,
            sandbox: None,
            supervisor: None,
            verification: faktor_agent::VerificationService::disabled(),
            model: "m".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "You are a test server agent.".into(),
            hooks: None,
            instructions_resolver: faktor_instructions::no_roots_resolver(),
            routing: faktor_agent::FixedRoutingPolicy::passthrough(),
            budgets: Arc::new(faktor_session::NoopBudget),
            clock: Arc::new(faktor_core::time::SystemClock),
            tool_call_mode: faktor_agent::ToolCallMode::Native,
            tool_deadline_ms: 2000,
            retry_policy: faktor_core::retry::RetryPolicy::default(),
            semantic: faktor_agent::fallback_semantic_registry(),
            context_prior: None,
            efficiency: Default::default(),
        })
        .unwrap();
        let deps = ServerDeps::new(session, agent, permissions.clone());
        let token = deps.auth_token.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        // Session via the wire surface (its message endpoint is the test
        // pattern for driving a full turn synchronously).
        let resp = client
            .post(format!("{base}/session"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({
                "title": "t-drive",
                "model": {"id": "m", "providerID": "fake"},
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid = created["sessionID"].as_str().unwrap().to_string();

        // The fake provider makes one write_file tool call; the turn
        // blocks on the permission hop until the daemon resolves it.
        let drive = async {
            let resp = client
                .post(format!("{base}/session/{sid}/message"))
                .bearer_auth(token.as_str())
                .json(&serde_json::json!({
                    "model": {"providerID": "fake", "modelID": "m"},
                    "parts": [{"type": "text", "text": "change src/a.txt"}],
                }))
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 200, "{:?}", resp.text().await);
        };
        let resolve = async {
            for _ in 0..100 {
                if let Some(pid) = permissions.pending_ids().first().copied() {
                    let resp = client
                        .post(format!("{base}/api/perm/{pid}/resolve"))
                        .bearer_auth(token.as_str())
                        .json(&serde_json::json!({
                            "permission_id": pid.to_string(),
                            "decision": "allow",
                        }))
                        .send()
                        .await
                        .unwrap();
                    assert_eq!(resp.status(), 200);
                    return;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            panic!("tool permission never surfaced");
        };
        tokio::join!(drive, resolve);

        // Wait for the machine to land on its terminal turn state.
        let mut body = serde_json::Value::Null;
        for _ in 0..100 {
            let resp = client
                .get(format!("{base}/session/{sid}/projection"))
                .bearer_auth(token.as_str())
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 200);
            body = resp.json().await.unwrap();
            if body["state"]["machine"] == "ready_for_next_turn" {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(
            body["state"]["machine"], "ready_for_next_turn",
            "turn must complete: {body}"
        );
        // The durable ledger's changed file appears in the projection...
        let files = body["filesChanged"].as_array().unwrap();
        assert!(
            files.iter().any(|f| f == "src/a.txt"),
            "ledger changed files must surface: {files:?}"
        );
        // ...the turn record's effective envelope is the activeModel...
        assert_eq!(body["activeModel"]["provider"], "fake");
        assert_eq!(body["activeModel"]["model"], "m");
        // ...and nothing is left running or queued.
        assert!(body["activeTool"].is_null(), "{body}");
        assert_eq!(body["verification"], serde_json::json!([]));
        assert_eq!(body["queued"], 0);
        // The driven turn settled provider calls, so the additive prefix
        // stability aggregate is present (its exact value is asserted in
        // the dedicated text-only test below — a tool turn rewrites the
        // head between its two calls, so only presence is pinned here).
        assert!(
            body["prefixStability"].is_object(),
            "prefix stability must surface after a driven turn: {body}"
        );
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn native_projection_prefix_stability_reflects_recorded_observations() {
        // v13 fill-site projection: null before any provider call settled
        // (asserted in the idle-shape test), then the durable aggregate of
        // the recorded per-call prefix observations — a single text-only
        // turn records exactly one observation with per-row stability 1.0
        // (nothing preceded it), so the projected aggregate must reflect
        // that recorded value: observations 1, mean 1.0, stdDev 0.0.
        let dir = tempfile::tempdir().unwrap();
        let mut registry = faktor_provider::ProviderRegistry::new();
        registry
            .try_register(Arc::new(FakeProvider::with_script(
                "fake",
                ModelCapabilities {
                    tools: true,
                    ..Default::default()
                },
                vec![
                    faktor_provider::ScriptedResponse::Text("pong".into()),
                    faktor_provider::ScriptedResponse::End,
                ],
            )))
            .unwrap();
        let session =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let permissions = ChannelPermissionRequester::new(Duration::from_secs(5));
        let agent = AgentRuntime::new(faktor_agent::AgentDeps {
            session: session.clone(),
            providers: Arc::new(registry),
            chunk_sink: None,
            permission_requester: permissions.clone(),
            evidence: Arc::new(faktor_agent::NoEvidence),
            tools: Arc::new(faktor_agent::ToolRegistry::new()),
            cas: None,
            workspaces: faktor_fs::WorkspaceFileService::new(),
            edit: None,
            snapshots: None,
            sandbox: None,
            supervisor: None,
            verification: faktor_agent::VerificationService::disabled(),
            model: "m".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "You are a test server agent.".into(),
            hooks: None,
            instructions_resolver: faktor_instructions::no_roots_resolver(),
            routing: faktor_agent::FixedRoutingPolicy::passthrough(),
            budgets: Arc::new(faktor_session::NoopBudget),
            clock: Arc::new(faktor_core::time::SystemClock),
            tool_call_mode: faktor_agent::ToolCallMode::Native,
            tool_deadline_ms: 2000,
            retry_policy: faktor_core::retry::RetryPolicy::default(),
            semantic: faktor_agent::fallback_semantic_registry(),
            context_prior: None,
            efficiency: Default::default(),
        })
        .unwrap();
        let deps = ServerDeps::new(session.clone(), agent, permissions.clone());
        let token = deps.auth_token.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        let resp = client
            .post(format!("{base}/session"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({
                "title": "t-prefix",
                "model": {"id": "m", "providerID": "fake"},
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid = created["sessionID"].as_str().unwrap().to_string();

        let projection = || async {
            client
                .get(format!("{base}/session/{sid}/projection"))
                .bearer_auth(token.as_str())
                .send()
                .await
                .unwrap()
                .json::<serde_json::Value>()
                .await
                .unwrap()
        };

        let resp = client
            .post(format!("{base}/session/{sid}/message"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({
                "model": {"providerID": "fake", "modelID": "m"},
                "parts": [{"type": "text", "text": "hi"}],
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "{:?}", resp.text().await);

        // Wait for the terminal machine state, then read the projection.
        let mut body = serde_json::Value::Null;
        for _ in 0..200 {
            body = projection().await;
            if body["state"]["machine"] == "ready_for_next_turn" {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(body["state"]["machine"], "ready_for_next_turn", "{body}");
        let ps = &body["prefixStability"];
        assert_eq!(ps["observations"], 1, "one settled provider call: {body}");
        assert_eq!(ps["mean"], 1.0, "first observation is stable by definition");
        assert_eq!(ps["stdDev"], 0.0);
        // The projection reflects the DURABLE recorded value: the store's
        // single observation row carries stability 1.0 (the aggregate mean
        // is computed over exactly that row).
        let rows = session
            .store()
            .provider_call_prefix_rows(SessionId::new(sid.as_str().parse::<u64>().unwrap()))
            .unwrap();
        assert_eq!(rows.len(), 1, "exactly one prefix observation row");
        assert!(rows[0].prompt_tokens > 0, "tokens recorded: {rows:?}");
        assert_ne!(rows[0].prompt_prefix_hash, [0u8; 32], "hash recorded");
        assert_eq!(rows[0].prefix_stability, Some(1.0));
        let agg = session
            .store()
            .session_stored_prefix_stability(SessionId::new(sid.as_str().parse::<u64>().unwrap()));
        assert_eq!(
            agg.unwrap().unwrap().mean,
            ps["mean"].as_f64().unwrap(),
            "projection must reflect the store aggregate"
        );
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn native_models_and_capabilities_serve_registered_models() {
        // GET /models and GET /capabilities must enumerate what the daemon
        // can ACTUALLY serve: an adapter registered with two configured
        // models appears in both surfaces with their real capabilities
        // (mirroring the provider/list introspection).
        let dir = tempfile::tempdir().unwrap();
        let mut caps = std::collections::HashMap::new();
        caps.insert(
            "gpt-x".to_string(),
            ModelCapabilities {
                context: 128_000,
                max_output: 16_384,
                tools: true,
                ..Default::default()
            },
        );
        caps.insert(
            "gpt-y".to_string(),
            ModelCapabilities {
                context: 64_000,
                max_output: 8_192,
                reasoning: true,
                ..Default::default()
            },
        );
        let openai = faktor_openai::OpenAiProvider::build(faktor_openai::OpenAiConfig {
            base_url: "http://127.0.0.1:1/v1".into(),
            api_key: None,
            family: faktor_openai::OpenAiFamily::Chat,
            models: caps,
        });
        let deps = test_deps_with(dir.path(), vec![openai]);
        let token = deps.auth_token.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        // /models: flat, deterministic, one entry per provider x model.
        let resp = client
            .get(format!("{base}/models"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let list: Vec<serde_json::Value> = resp.json().await.unwrap();
        assert!(
            list.iter().any(|m| m["provider"] == "openai"
                && m["model"] == "gpt-x"
                && m["context"] == 128_000
                && m["maxOutput"] == 16_384
                && m["tools"] == true
                && m["source"] == "providerCatalog"),
            "gpt-x with real capabilities: {list:?}"
        );
        assert!(
            list.iter().any(|m| m["provider"] == "openai"
                && m["model"] == "gpt-y"
                && m["reasoning"] == true),
            "gpt-y reasoning flag: {list:?}"
        );
        assert!(
            list.iter()
                .any(|m| m["provider"] == "fake" && m["model"] == "default"),
            "registered fake provider still catalogued: {list:?}"
        );
        // Deterministic ordering: sorted by provider then model.
        let keys: Vec<(&str, &str)> = list
            .iter()
            .map(|m| {
                (
                    m["provider"].as_str().unwrap_or(""),
                    m["model"].as_str().unwrap_or(""),
                )
            })
            .collect();
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        assert_eq!(keys, sorted, "catalog must be deterministically ordered");

        // /capabilities: map provider -> {models, runtimeContextLimitSupported}.
        let resp = client
            .get(format!("{base}/capabilities"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        let openai_entry = body.get("openai").expect("openai provider key present");
        let models = openai_entry["models"].as_array().unwrap();
        assert!(
            models
                .iter()
                .any(|m| m["id"] == "gpt-x" && m["capabilities"]["context"] == 128_000),
            "gpt-x capabilities: {models:?}"
        );
        assert!(
            models
                .iter()
                .any(|m| m["id"] == "gpt-y" && m["capabilities"]["max_output"] == 8_192),
            "gpt-y capabilities: {models:?}"
        );
        assert_eq!(openai_entry["runtimeContextLimitSupported"], false);
        let fake_entry = body.get("fake").expect("fake provider key present");
        assert_eq!(fake_entry["runtimeContextLimitSupported"], false);
        let _ = handle.shutdown.send(());
    }

    // --------------------------------------------------- native v1: audits 55-56
    // /native/health + /native/ready semantics, the durable session
    // listings, the strict abort DTO and the cross-session usage aggregate.

    #[tokio::test]
    async fn native_health_and_ready_semantics() {
        // health answers 200 whenever the process responds; ready answers
        // 200 ONLY after serve() setup completed (recovery ran, migrations
        // applied, components in place) — and 503 before that moment, which
        // the simulate_not_ready knob keeps observable in tests.
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let token = deps.auth_token.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        // Both are auth-gated like every daemon route.
        let resp = client
            .get(format!("{base}/native/health"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);
        let resp = client
            .get(format!("{base}/native/ready"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);

        // Post-serve: health = liveness {ok, version}, ready = 200
        // {ready:true} (the flag flips at the very end of serve() setup, so
        // a test can only observe true after serve returns).
        let resp = client
            .get(format!("{base}/native/health"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["ok"], true);
        assert!(body["version"].is_string());
        let resp = client
            .get(format!("{base}/native/ready"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["ready"], true);
        let _ = handle.shutdown.send(());

        // The not-ready window (deterministic test knob): with
        // simulate_not_ready the flag never flips, so ready is 503
        // {ready:false} even after serve returned — health stays 200.
        let deps = {
            let mut d = test_deps(dir.path());
            d.simulate_not_ready = true;
            d
        };
        let token = deps.auth_token.clone();
        let handle = serve(deps, 0).await.unwrap();
        let base2 = format!("http://{}", handle.addr);
        let resp = client
            .get(format!("{base2}/native/ready"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 503);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["ready"], false);
        let resp = client
            .get(format!("{base2}/native/health"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn native_turns_lists_a_driven_turn_and_hostile_ids_are_loud() {
        // Drive a REAL turn through the HTTP surface (FakeProvider pong),
        // then read it back from /native/session/{id}/turns as a completed
        // durable turn record with its envelope.
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let token = deps.auth_token.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        let resp = client
            .post(format!("{base}/api/session"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"provider": "fake", "model": "m"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid = created["id"].as_str().unwrap().to_string();
        let resp = client
            .post(format!("{base}/api/session/{sid}/prompt"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"prompt": "hi", "files": []}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

        // Poll the native turns listing until the durable record lands.
        let mut body = serde_json::Value::Null;
        for _ in 0..200 {
            let resp = client
                .get(format!("{base}/native/session/{sid}/turns"))
                .bearer_auth(token.as_str())
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 200);
            body = resp.json().await.unwrap();
            let done = body
                .as_array()
                .unwrap()
                .iter()
                .any(|t| t["status"] == "completed");
            if done {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let turns = body.as_array().unwrap();
        let last = turns.first().expect("at least one completed turn");
        assert_eq!(last["status"], "completed");
        assert_eq!(last["provider"], "fake");
        assert_eq!(last["model"], "m");
        assert!(last["opId"].as_str().unwrap().parse::<u64>().is_ok());
        assert!(last["startedAt"].as_i64().unwrap_or(0) > 0);

        // Unauth 401; hostile ids: 0 and non-numeric → 400, unknown → 404.
        let resp = client
            .get(format!("{base}/native/session/{sid}/turns"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);
        for hostile in ["0", "abc", "184467440737095516150"] {
            let resp = client
                .get(format!("{base}/native/session/{hostile}/turns"))
                .bearer_auth(token.as_str())
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 400, "hostile id {hostile}");
        }
        let resp = client
            .get(format!("{base}/native/session/999999/turns"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn native_tasks_verification_agents_and_terminal_reflect_durable_rows() {
        // The listings are row-backed: an injected durable ledger, an
        // injected verification fact, no turn records, no PTYs. Reading
        // them back must be exact; hostile ids are loud.
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let token = deps.auth_token.clone();
        let manager = deps.session.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let ws = manager.create_workspace("/tmp").unwrap();
        let session = manager.create_session(ws, "t-rows", "fake", "m").unwrap();
        let sid = session.id().to_string();
        let h = manager.get_session(session.id()).unwrap().unwrap();
        h.put_task_ledger(serde_json::json!({
            "goal": "implement the native surface",
            "constraints": ["rust"],
            "completed_steps": ["mount routes"],
            "open_steps": ["wire abort", "aggregate usage"],
            "decisions": ["strict DTOs"],
            "known_failures": [],
            "changed_files": ["crates/server/src/api.rs"],
            "tests_run": ["cargo check"],
            "tests_failed": [],
            "user_preferences": [],
        }))
        .unwrap();
        h.upsert_memory_fact("verification", "fmt", "failed:make fmt")
            .unwrap();

        // tasks: exactly one entry, typed from the ledger + the fact.
        let resp = client
            .get(format!("{base}/native/session/{sid}/tasks"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        let tasks = body.as_array().unwrap();
        assert_eq!(tasks.len(), 1, "{body}");
        assert_eq!(tasks[0]["goal"], "implement the native surface");
        assert_eq!(tasks[0]["state"], "in_progress");
        assert_eq!(
            tasks[0]["milestones"]["open"],
            serde_json::json!(["wire abort", "aggregate usage"])
        );
        assert_eq!(
            tasks[0]["milestones"]["completed"],
            serde_json::json!(["mount routes"])
        );
        assert_eq!(
            tasks[0]["changedFiles"],
            serde_json::json!(["crates/server/src/api.rs"])
        );
        assert_eq!(
            tasks[0]["verification"],
            serde_json::json!([{"id": "fmt", "detail": "failed:make fmt", "status": "failed"}])
        );

        // verification: the durable fact is owed-failed; no pending runs.
        let resp = client
            .get(format!("{base}/native/session/{sid}/verification"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["owed"], serde_json::json!([]));
        assert_eq!(body["failedChecks"][0]["id"], "fmt");
        assert_eq!(body["failedChecks"][0]["detail"], "failed:make fmt");

        // agents: no background agents yet (orchestration not landed).
        let resp = client
            .get(format!("{base}/native/session/{sid}/agents"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(
            resp.json::<serde_json::Value>().await.unwrap(),
            serde_json::json!([])
        );

        // terminal: no PTYs exist on this daemon → the empty view.
        let resp = client
            .get(format!("{base}/native/session/{sid}/terminal"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(
            resp.json::<serde_json::Value>().await.unwrap(),
            serde_json::json!([])
        );

        // turns: no turn ever ran → empty. checkpoints: no service wired
        // in test_deps → empty.
        let resp = client
            .get(format!("{base}/native/session/{sid}/turns"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.json::<serde_json::Value>().await.unwrap(),
            serde_json::json!([])
        );
        let resp = client
            .get(format!("{base}/native/session/{sid}/checkpoints"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.json::<serde_json::Value>().await.unwrap(),
            serde_json::json!([])
        );

        // The same hostile/unknown treatment applies to every listing.
        for path in [
            "/native/session/0/tasks".to_string(),
            "/native/session/abc/verification".to_string(),
            "/native/session/0/agents".to_string(),
            "/native/session/abc/terminal".to_string(),
        ] {
            let resp = client
                .get(format!("{base}{path}"))
                .bearer_auth(token.as_str())
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 400, "{path}");
        }
        for path in [
            "/native/session/999999/tasks".to_string(),
            "/native/session/999999/checkpoints".to_string(),
            "/native/session/999999/verification".to_string(),
            "/native/session/999999/agents".to_string(),
            "/native/session/999999/terminal".to_string(),
        ] {
            let resp = client
                .get(format!("{base}{path}"))
                .bearer_auth(token.as_str())
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 404, "{path}");
        }
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn native_checkpoints_reflect_written_rows_when_service_wired() {
        // With the real checkpoint service wired, recorded file changes
        // surface as checkpoint rows (newest first); without rows the
        // listing is empty but live.
        let dir = tempfile::tempdir().unwrap();
        let (deps, snapshots, _fs) = wire_snapshot_deps(dir.path());
        let token = deps.auth_token.clone();
        let manager = deps.session.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let ws = manager.create_workspace("/tmp").unwrap();
        let session = manager.create_session(ws, "t-cp", "fake", "m").unwrap();
        let sid = session.id().to_string();

        // Empty before any write.
        let resp = client
            .get(format!("{base}/native/session/{sid}/checkpoints"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(
            resp.json::<serde_json::Value>().await.unwrap(),
            serde_json::json!([])
        );

        // Two real checkpoint rows, exactly like the edit engine records.
        let before = snapshots
            .before_write(session.id(), "notes.txt", b"original\n")
            .unwrap();
        let after = snapshots
            .before_write(session.id(), "notes.txt", b"edited\n")
            .unwrap();
        snapshots
            .after_write(session.id(), "notes.txt", before, after, 0, b"edited\n")
            .unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
        let before2 = snapshots
            .before_write(session.id(), "a.rs", b"one")
            .unwrap();
        let after2 = snapshots
            .before_write(session.id(), "a.rs", b"two")
            .unwrap();
        snapshots
            .after_write(session.id(), "a.rs", before2, after2, 0, b"two")
            .unwrap();

        let resp = client
            .get(format!("{base}/native/session/{sid}/checkpoints"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let rows: serde_json::Value = resp.json().await.unwrap();
        let rows = rows.as_array().unwrap();
        assert_eq!(rows.len(), 2);
        // Newest first (higher sequence first).
        assert_eq!(rows[0]["path"], "a.rs");
        assert_eq!(rows[1]["path"], "notes.txt");
        assert_eq!(rows[1]["beforeHash"], before.to_hex());
        assert_eq!(rows[1]["afterHash"], after.to_hex());
        assert!(rows[0]["createdMs"].as_i64().unwrap_or(0) > 0);
        assert!(rows[0]["restoredMs"].is_null());
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn native_abort_strict_dto_and_targeted_kill() {
        // The native abort is sdk_abort semantics behind the STRICT native
        // DTO (audit 56): any unknown body field — a typo included — is a
        // 400 before anything runs; a valid body kills exactly the targeted
        // queued op and leaves the machine untouched.
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let token = deps.auth_token.clone();
        let manager = deps.session.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let ws = manager.create_workspace("/tmp").unwrap();
        let session = manager
            .create_session(ws, "t-abort-native", "fake", "m")
            .unwrap();
        let session_id = session.id().to_string();
        let _ = session.submit_prompt("first", &[]).unwrap();
        let second = session.submit_prompt("second", &[]).unwrap();
        assert!(second.queued, "second prompt must queue behind Preparing");
        let op_id = second.op_id.to_string();

        // Strict DTO rejections: unknown field, realistic typo, missing
        // session_id, unparseable op_id, path/body mismatch, hostile path.
        for evil in [
            format!(r#"{{"session_id":"{session_id}","bogus":1}}"#),
            format!(r#"{{"session_id":"{session_id}","hardBudegt":true}}"#),
        ] {
            let resp = client
                .post(format!("{base}/native/session/{session_id}/abort"))
                .bearer_auth(token.as_str())
                .header("content-type", "application/json")
                .body(evil.clone())
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 400, "{evil}");
        }
        let resp = client
            .post(format!("{base}/native/session/{session_id}/abort"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"op_id": "1"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400, "missing session_id");
        let resp = client
            .post(format!("{base}/native/session/{session_id}/abort"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"session_id": session_id, "op_id": "not-a-number"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400, "unparseable op_id");
        let resp = client
            .post(format!("{base}/native/session/999999/abort"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"session_id": session_id}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400, "path/body session mismatch");
        let resp = client
            .post(format!("{base}/native/session/abc/abort"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"session_id": session_id}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400, "hostile path id");

        // Unauth with a VALID body → 401 (auth gate runs in the handler).
        let resp = client
            .post(format!("{base}/native/session/{session_id}/abort"))
            .json(&serde_json::json!({"session_id": session_id, "op_id": op_id}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);

        // Valid targeted abort of the queued prompt.
        let resp = client
            .post(format!("{base}/native/session/{session_id}/abort"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"session_id": session_id, "op_id": op_id}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let aborted: serde_json::Value = resp.json().await.unwrap();
        assert!(
            aborted["aborted"]
                .as_array()
                .unwrap()
                .iter()
                .any(|o| o.as_str() == Some(op_id.as_str())),
            "{aborted}"
        );
        assert_eq!(
            session.state().unwrap(),
            faktor_core::state::AgentState::Preparing,
            "a queued-prompt kill must not touch the state machine"
        );
        assert_eq!(session.queued_prompt_count().unwrap(), 0);

        // Unknown session → 404 for a valid body.
        let resp = client
            .post(format!("{base}/native/session/999999/abort"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"session_id": "999999"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn native_terminal_lists_live_ptys_with_id_pid_alive() {
        // A live PTY on the daemon appears in /native/session/{id}/terminal
        // (the session id is validated, but PTYs have no session binding
        // yet — all daemon PTYs are listed). Platforms that refuse PTY
        // spawns must still serve the empty listing honestly.
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let token = deps.auth_token.clone();
        let manager = deps.session.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let ws = manager.create_workspace("/tmp").unwrap();
        let session = manager.create_session(ws, "t-pty", "fake", "m").unwrap();
        let sid = session.id().to_string();

        let resp = client
            .post(format!("{base}/pty/create"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"command": "/bin/sleep", "args": ["30"]}))
            .send()
            .await
            .unwrap();
        if resp.status() != 200 {
            // Platform refusal (documented): the terminal view stays empty.
            let resp = client
                .get(format!("{base}/native/session/{sid}/terminal"))
                .bearer_auth(token.as_str())
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 200);
            assert_eq!(
                resp.json::<serde_json::Value>().await.unwrap(),
                serde_json::json!([])
            );
            let _ = handle.shutdown.send(());
            return;
        }
        let created: serde_json::Value = resp.json().await.unwrap();
        let pty_id = created["pty_id"].as_str().unwrap().to_string();
        let pid = created["pid"].as_u64().unwrap_or(0);
        assert!(pid > 0);

        // The live pty lists with its id, pid and aliveness.
        let resp = client
            .get(format!("{base}/native/session/{sid}/terminal"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        let entries = body.as_array().unwrap();
        let mine = entries
            .iter()
            .find(|e| e["id"] == pty_id)
            .expect("the live pty must be listed");
        assert_eq!(mine["pid"], pid);
        assert_eq!(mine["alive"], true);

        // Removing it clears the listing (and remove is idempotent).
        let resp = client
            .post(format!("{base}/pty/remove"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"pty_id": pty_id}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let resp = client
            .get(format!("{base}/native/session/{sid}/terminal"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.json::<serde_json::Value>().await.unwrap(),
            serde_json::json!([])
        );
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn native_usage_aggregates_budget_and_spent_across_sessions() {
        // /native/usage sums the durable usage facts (kind "usage", keys
        // budget/spent) across sessions; hostile non-numeric values are
        // skipped, sessions without usage facts are counted but not listed.
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let token = deps.auth_token.clone();
        let manager = deps.session.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let ws = manager.create_workspace("/tmp").unwrap();
        let s1 = manager.create_session(ws, "t-u1", "fake", "m").unwrap();
        let ws2 = manager.create_workspace("/tmp2").unwrap();
        let s2 = manager.create_session(ws2, "t-u2", "fake", "m").unwrap();
        let ws3 = manager.create_workspace("/tmp3").unwrap();
        let _s3 = manager.create_session(ws3, "t-u3", "fake", "m").unwrap();
        // Real usage facts...
        s1.upsert_memory_fact("usage", "budget", "90000").unwrap();
        s1.upsert_memory_fact("usage", "spent", "1234").unwrap();
        s2.upsert_memory_fact("usage", "budget", "10000").unwrap();
        s2.upsert_memory_fact("usage", "spent", "42").unwrap();
        // ...hostile rows (non-numeric / other kinds / other keys) never
        // break the aggregate.
        s2.upsert_memory_fact("usage", "budget", "not-a-number")
            .unwrap();
        s2.upsert_memory_fact("usage", "rogue", "900000").unwrap();
        s2.upsert_memory_fact("preference", "budget", "700000")
            .unwrap();

        let resp = client
            .get(format!("{base}/native/usage"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);
        let resp = client
            .get(format!("{base}/native/usage"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["sessions"], 3);
        // The later hostile budget upsert REPLACED s2's numeric budget
        // (upsert semantics) with a non-numeric value, which is skipped:
        // totals carry only the numeric facts.
        assert_eq!(body["totals"]["budget"], 90000);
        assert_eq!(body["totals"]["spent"], 1276);
        let per = body["perSession"].as_array().unwrap();
        assert_eq!(per.len(), 2, "fact-less sessions are counted, not listed");
        assert!(per.iter().any(|e| {
            e["sessionId"] == s1.id().to_string() && e["budget"] == 90000 && e["spent"] == 1234
        }));
        assert!(per.iter().any(|e| {
            e["sessionId"] == s2.id().to_string() && e["budget"].is_null() && e["spent"] == 42
        }));
        let _ = handle.shutdown.send(());
    }

    // ------------------------------------ orchestration graph (audit 93)

    /// Seed one parent session with a wave-12/13-shaped durable run: a
    /// plan row (items b then a), two registry children (b is Done with a
    /// staged merge; a is Running), one child with steering rows. Returns
    /// the parent session id and the child session ids.
    fn seed_orchestration_graph(
        manager: &Arc<SessionManager>,
    ) -> (SessionId, SessionId, SessionId) {
        let ws = manager.create_workspace("/orch-root").unwrap();
        let parent = manager.create_session(ws, "orch", "fake", "m").unwrap();
        let child_a = manager.create_session(ws, "child-a", "fake", "m").unwrap();
        let child_b = manager.create_session(ws, "child-b", "fake", "m").unwrap();
        let now = 1_700_000_000_000i64;
        let plan_row = serde_json::json!({
            "plan": {
                "goal": "Ship the graph",
                "non_goals": [],
                "constraints": [],
                "work_items": [
                    {"id": "b", "summary": "work b", "depends_on": [], "kind": "Analysis",
                     "acceptance_checks": [], "completion": "Pending"},
                    {"id": "a", "summary": "work a", "depends_on": ["b"], "kind": "Analysis",
                     "acceptance_checks": [], "completion": "Pending"},
                ]
            },
            "owner_ws": ws.raw(),
            "owner_wt": 1,
            "owner_root": "/orch-root",
            "specs": [],
            "provider": "fake",
            "default_model": "m",
            "isolated_root": "/iso",
            "created_ms": now,
        });
        parent
            .upsert_memory_fact(ORCH_PLAN_KIND, "run-1", &plan_row.to_string())
            .unwrap();
        let registry = |child_id: &str, item: &str, sid: u64, state: &str, ms: i64| {
            serde_json::json!({
                "child_id": child_id,
                "parent_session_id": parent.id().raw(),
                "run_id": "run-1",
                "item_id": item,
                "kind": "Analysis",
                "session_id": sid,
                "operation_id": 0,
                "workspace_id": ws.raw(),
                "worktree_id": sid,
                "ownership": "read_only_shared",
                "ownership_paths": [],
                "state": state,
                "budget_max_tokens": 1000,
                "permissions": [],
                "model_policy": {"model": null},
                "created_ms": ms,
                "updated_ms": ms,
            })
        };
        parent
            .upsert_memory_fact(
                ORCH_REGISTRY_KIND,
                "run-1/child-0",
                &registry("child-0", "b", child_b.id().raw(), "Done", now).to_string(),
            )
            .unwrap();
        parent
            .upsert_memory_fact(
                ORCH_REGISTRY_KIND,
                "run-1/child-1",
                &registry("child-1", "a", child_a.id().raw(), "Running", now + 10).to_string(),
            )
            .unwrap();
        // Steering history on the a-child: one applied Pause, one pending
        // Steer — seq order matters.
        let pause = child_a
            .orchestrator_ctl_enqueue(faktor_session::child::ChildControl::Pause)
            .unwrap();
        child_a
            .orchestrator_ctl_enqueue(faktor_session::child::ChildControl::Steer {
                note: "focus the api".into(),
            })
            .unwrap();
        child_a.orchestrator_ctl_ack(pause.seq).unwrap();
        // A durable merge record + parts for the Done b-child.
        let cs = "base-child-0-cs";
        let envelope = serde_json::json!({
            "seq": 1, "child_id": "child-0", "cs_id": cs,
            "status": "applied", "approved_count": 2, "rejected_count": 0,
            "merged_count": 1, "conflict_count": 1,
            "created_ms": now, "finished_ms": now + 5, "details": "x",
        });
        parent
            .upsert_memory_fact(
                ORCH_MERGE_KIND,
                &format!("run-1/child-0/merge/{cs}/1"),
                &envelope.to_string(),
            )
            .unwrap();
        for (part, value) in [
            ("merged", serde_json::json!(["keep.rs"])),
            ("rejected", serde_json::json!([])),
            (
                "conflicts",
                serde_json::json!([["src/a.rs", "parent moved past the base snapshot"]]),
            ),
        ] {
            let key = format!("run-1/child-0/merge/{cs}/1/part/{part}");
            parent
                .upsert_memory_fact(ORCH_MERGE_PART_KIND, &key, "{\"chunks\":1}")
                .unwrap();
            parent
                .upsert_memory_fact(
                    ORCH_MERGE_PART_KIND,
                    &format!("{key}/c001"),
                    &value.to_string(),
                )
                .unwrap();
        }
        (parent.id(), child_a.id(), child_b.id())
    }

    #[tokio::test]
    async fn native_presentation_continuity_is_durable_scoped_and_typed() {
        // Foreground -> background -> foreground continuity: the durable
        // presentation fold of the child session, surfaced in the agent
        // listing + typed graph, never a scheduling or lineage change.
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let manager = deps.session.clone();
        let token = deps.auth_token.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        // child-0 = Done row, child-1 = Running row.
        let (parent, child_a, child_b) = seed_orchestration_graph(&manager);
        let presentation_path = |session: &str, child: &str| {
            format!("/native/session/{session}/agents/{child}/presentation")
        };

        // Unauthenticated is 401 before anything else.
        let resp = client
            .post(format!(
                "{base}{}",
                presentation_path(&parent.to_string(), "child-1")
            ))
            .json(&serde_json::json!({"state": "background"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);

        // Strict DTO: every hostile body is a plain 400, never a 422 and
        // never a silent default.
        for body in [
            serde_json::json!({}),
            serde_json::json!({"state": "paused"}),
            serde_json::json!({"state": "BACKGROUND"}),
            serde_json::json!({"state": 1}),
            serde_json::json!({"state": null}),
            serde_json::json!({"state": "background", "extra": true}),
            serde_json::json!("background"),
        ] {
            let resp = client
                .post(format!(
                    "{base}{}",
                    presentation_path(&parent.to_string(), "child-1")
                ))
                .bearer_auth(token.as_str())
                .json(&body)
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 400, "hostile body must 400: {body}");
        }
        let resp = client
            .post(format!(
                "{base}{}",
                presentation_path(&parent.to_string(), "child-1")
            ))
            .bearer_auth(token.as_str())
            .header("content-type", "application/json")
            .body("{not json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);

        // Hostile/unknown/foreign ids are typed 404s scoped to the path
        // session (a child id of another session is never resolved).
        let other = manager
            .create_session(
                manager.create_workspace("/other").unwrap(),
                "other",
                "fake",
                "m",
            )
            .unwrap()
            .id();
        for (session, child) in [
            (parent.to_string(), "child-9".to_string()),
            (parent.to_string(), "..".to_string()),
            ("999999".to_string(), "child-1".to_string()),
            (other.to_string(), "child-1".to_string()),
        ] {
            let resp = client
                .post(format!("{base}{}", presentation_path(&session, &child)))
                .bearer_auth(token.as_str())
                .json(&serde_json::json!({"state": "background"}))
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 404, "session {session} child {child}");
        }
        let resp = client
            .post(format!(
                "{base}{}",
                presentation_path("not-a-number", "child-1")
            ))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"state": "background"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);

        // A running child flips to background durably; the same-state set is
        // an idempotent no-op that writes nothing.
        let resp = client
            .post(format!(
                "{base}{}",
                presentation_path(&parent.to_string(), "child-1")
            ))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"state": "background"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let ack: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(ack["child_id"], "child-1");
        assert_eq!(ack["presentation"], "background");
        assert_eq!(ack["changed"], true);
        let resp = client
            .post(format!(
                "{base}{}",
                presentation_path(&parent.to_string(), "child-1")
            ))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"state": "background"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let ack: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(ack["changed"], false);
        // Durable truth: the child session's ledger fold holds Background.
        let ca = manager.get_session(child_a).unwrap().unwrap();
        assert_eq!(
            ca.child_presentation("child-1").unwrap(),
            faktor_session::child::PresentationState::Background
        );

        // The native agent listing and the typed graph both carry the field;
        // an untouched child stays foreground.
        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/agents?session={parent}"),
        )
        .await;
        assert_eq!(resp.status(), 200);
        let listing: serde_json::Value = resp.json().await.unwrap();
        let c1 = listing
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["agent_id"] == "child-1")
            .expect("child-1 listed");
        assert_eq!(c1["presentation"], "background");
        let c0 = listing
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["agent_id"] == "child-0")
            .expect("child-0 listed");
        assert_eq!(c0["presentation"], "foreground");
        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/orchestrator/graph?session={parent}"),
        )
        .await;
        assert_eq!(resp.status(), 200);
        let graph: serde_json::Value = resp.json().await.unwrap();
        let g1 = graph["children"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["child_id"] == "child-1")
            .expect("child-1 graphed");
        assert_eq!(g1["presentation"], "background");

        // Background -> foreground: the SAME ChildId/session continues and
        // the state folds back.
        let resp = client
            .post(format!(
                "{base}{}",
                presentation_path(&parent.to_string(), "child-1")
            ))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"state": "foreground"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let ack: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(ack["presentation"], "foreground");
        assert_eq!(ack["changed"], true);
        assert_eq!(c1["session_id"], g1["session_id"]);

        // Terminal children are Background-only: child-0's durable row is
        // Done. The currently-foreground no-op stays legal, the background
        // flip is accepted, and the foreground revival is a typed 409 that
        // writes nothing.
        let resp = client
            .post(format!(
                "{base}{}",
                presentation_path(&parent.to_string(), "child-0")
            ))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"state": "foreground"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(
            resp.json::<serde_json::Value>().await.unwrap()["changed"],
            false
        );
        let resp = client
            .post(format!(
                "{base}{}",
                presentation_path(&parent.to_string(), "child-0")
            ))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"state": "background"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let resp = client
            .post(format!(
                "{base}{}",
                presentation_path(&parent.to_string(), "child-0")
            ))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"state": "foreground"}))
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            409,
            "terminal child refuses foreground revival"
        );
        let cb = manager.get_session(child_b).unwrap().unwrap();
        assert_eq!(
            cb.child_presentation("child-0").unwrap(),
            faktor_session::child::PresentationState::Background,
            "the refused revival wrote nothing"
        );
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn native_orchestrator_graph_projects_the_durable_run() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let manager = deps.session.clone();
        let token = deps.auth_token.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let (parent, _ca, cb) = seed_orchestration_graph(&manager);
        // Unauthenticated is 401 like every /native handler.
        let resp = client
            .get(format!(
                "{base}/native/orchestrator/graph?session={}",
                parent
            ))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);
        // Authenticated: the full graph JSON.
        let resp = client
            .get(format!(
                "{base}/native/orchestrator/graph?session={}",
                parent
            ))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let g: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(g["plan_id"], "run-1");
        assert_eq!(g["goal"], "Ship the graph");
        // b is Done (durable child row), a is still Pending behind... no:
        // a has a durable Running child, so the step is Running; root is
        // Running (not all Done).
        let steps: Vec<&str> = g["work_items"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| w["item_id"].as_str().unwrap())
            .collect();
        assert_eq!(steps, vec!["b", "a"]);
        assert_eq!(g["work_items"][0]["state"], "Done");
        assert_eq!(g["work_items"][1]["state"], "Running");
        assert_eq!(g["state"], "Running");
        // Children ordered by plan step (b child first even though its
        // created_ms is smaller anyway; verify the linkage values).
        let children = g["children"].as_array().unwrap();
        assert_eq!(children.len(), 2);
        assert_eq!(children[0]["child_id"], "child-0");
        assert_eq!(children[0]["plan_step_index"], 0);
        assert_eq!(children[0]["session_id"], cb.raw());
        assert_eq!(children[0]["worktree_id"], cb.raw());
        assert_eq!(children[0]["ownership"], "read_only_shared");
        assert_eq!(children[0]["state"], "Done");
        assert_eq!(children[0]["budget"], 1000);
        assert!(children[0]["capabilities"].is_array());
        // The merge record of the Done child.
        let m = &children[0]["merge"];
        assert_eq!(m["change_set_id"], "base-child-0-cs");
        assert_eq!(m["merged"], serde_json::json!(["keep.rs"]));
        assert_eq!(m["rejected"], serde_json::json!([]));
        assert_eq!(m["conflicts"][0][0], "src/a.rs");
        assert_eq!(children[1]["child_id"], "child-1");
        assert_eq!(children[1]["plan_step_index"], 1);
        assert_eq!(children[1]["state"], "Running");
        assert!(children[1]["merge"].is_null());
        // Steering history of the a-child: applied pause first, pending
        // steer second, seq order preserved.
        let events = children[1]["steer_events"].as_array().unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0]["kind"]["kind"], "pause");
        assert!(!events[0]["applied_ms"].is_null());
        assert_eq!(events[1]["kind"]["kind"], "steer");
        assert_eq!(events[1]["kind"]["note"], "focus the api");
        assert!(events[1]["applied_ms"].is_null());
        assert!(events[0]["seq"].as_u64().unwrap() < events[1]["seq"].as_u64().unwrap());
        let _ = handle.shutdown.send(());
    }

    /// The canonical projection is ONE function: the JSON graph surface and
    /// the agent/task-run aggregation derive byte-identical states for the
    /// same registry rows, and a Blocked child's durable blocker truth rides
    /// both surfaces (the old "non-terminal means Running" conversion is
    /// gone).
    #[tokio::test]
    async fn native_graph_and_agents_share_the_canonical_child_projection() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let manager = deps.session.clone();
        let token = deps.auth_token.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let (parent, _ca, _cb) = seed_orchestration_graph(&manager);
        // Rewrite child-1 (item a) into a Blocked child carrying durable
        // blocker truth.
        let parent_handle = manager.get_session(parent).unwrap().unwrap();
        let mut raw = None;
        let mut after = None;
        loop {
            let page = parent_handle
                .memory_facts_page(after.as_ref(), 200)
                .unwrap();
            for (kind, key, value) in &page.facts {
                if kind == ORCH_REGISTRY_KIND && key == "run-1/child-1" {
                    raw = Some(value.clone());
                }
            }
            match page.cursor {
                Some(c) => after = Some(c),
                None => break,
            }
        }
        let mut row: faktor_orchestrator::runtime::ChildRuntime =
            serde_json::from_str(&raw.expect("child-1 registry row")).unwrap();
        row.set_blocker(&faktor_orchestrator::runtime::ChildBlocker {
            kind: "permission".into(),
            reason: "waiting for a pending permission decision".into(),
            dependency: None,
            resolution: Some("resolve the pending permission request".into()),
            last_progress_ms: Some(42),
        })
        .unwrap();
        parent_handle
            .upsert_memory_fact(
                ORCH_REGISTRY_KIND,
                "run-1/child-1",
                &serde_json::to_string(&row).unwrap(),
            )
            .unwrap();
        // Surface 1: the JSON operation graph.
        let g: serde_json::Value = client
            .get(format!("{base}/native/orchestrator/graph?session={parent}"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(g["work_items"][1]["state"], "Blocked");
        assert_eq!(g["state"], "Blocked");
        let graph_child = g["children"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["child_id"] == "child-1")
            .expect("child-1 graph node");
        assert_eq!(graph_child["state"], "Blocked");
        assert_eq!(graph_child["blocker"]["kind"], "permission");
        assert_eq!(graph_child["blocker"]["last_progress_ms"], 42);
        // Surface 2: the agent listing (the task-run projection delegates
        // to the SAME body).
        let entries: serde_json::Value = client
            .get(format!("{base}/native/agents?session={parent}"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let self_entry = entries
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["kind"] == "self")
            .expect("self entry");
        let child = entries
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["agent_id"] == "child-1")
            .expect("child entry");
        assert_eq!(
            self_entry["state"], g["state"],
            "root state must be byte-identical across surfaces"
        );
        assert_eq!(
            child["state"], graph_child["state"],
            "child state must be byte-identical across surfaces"
        );
        assert_eq!(child["blocker"]["kind"], "permission");
        assert_eq!(child["blocker"]["last_progress_ms"], 42);
        let runs: serde_json::Value = client
            .get(format!("{base}/native/session/{parent}/task-runs"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let run = runs
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["run_id"] == "run-1")
            .expect("task run entry");
        assert_eq!(
            run["state"], g["state"],
            "task-run state must use the same projection"
        );
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn native_orchestrator_graph_hostile_ids_404_and_corrupt_rows_are_loud() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let manager = deps.session.clone();
        let token = deps.auth_token.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let (parent, _ca, _cb) = seed_orchestration_graph(&manager);
        // Hostile session ids: non-numeric, zero, negative, absurd, and a
        // session that exists but holds no orchestration run — all 404
        // (never a phantom graph, never a 200).
        for hostile in [
            "abc".to_string(),
            "0".to_string(),
            "-1".to_string(),
            "999999999".to_string(),
            "1;drop".to_string(),
            "%2e%2e".to_string(),
        ] {
            let resp = client
                .get(format!(
                    "{base}/native/orchestrator/graph?session={hostile}"
                ))
                .bearer_auth(token.as_str())
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 404, "hostile session {hostile:?} must 404");
        }
        // A real session with no orchestration rows: 404.
        let ws = manager.create_workspace("/plain").unwrap();
        let plain = manager.create_session(ws, "plain", "fake", "m").unwrap();
        let resp = client
            .get(format!(
                "{base}/native/orchestrator/graph?session={}",
                plain.id()
            ))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);
        // A second run under the PARENT: the graph is ambiguous — loud 409
        // naming both runs.
        let plan_row = serde_json::json!({
            "plan": {"goal": "second", "non_goals": [], "constraints": [],
                     "work_items": []},
            "created_ms": 1,
        });
        manager
            .get_session(parent)
            .unwrap()
            .unwrap()
            .upsert_memory_fact(ORCH_PLAN_KIND, "run-2", &plan_row.to_string())
            .unwrap();
        let resp = client
            .get(format!(
                "{base}/native/orchestrator/graph?session={}",
                parent
            ))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 409);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(body["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("run-1"));
        assert!(body["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("run-2"));
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn native_orchestrator_graph_tampered_registry_row_is_a_loud_500() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let manager = deps.session.clone();
        let token = deps.auth_token.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let (parent, _ca, _cb) = seed_orchestration_graph(&manager);
        // Corrupt one registry row: the projection refuses loudly (500)
        // instead of serving a silently partial graph.
        let parent_handle = manager.get_session(parent).unwrap().unwrap();
        parent_handle
            .upsert_memory_fact(ORCH_REGISTRY_KIND, "run-1/child-1", "{corrupt")
            .unwrap();
        let resp = client
            .get(format!(
                "{base}/native/orchestrator/graph?session={}",
                parent
            ))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 500);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap_or("")
                .contains("run-1/child-1"),
            "{body}"
        );
        let _ = handle.shutdown.send(());
    }

    // ------------------------------------------------- native agents + control

    /// Chunk-paced per-call provider (real-drive control windows).
    struct PacedScriptedProvider {
        inner: FakeProvider,
        calls: std::sync::Mutex<Vec<Vec<faktor_provider::ScriptedResponse>>>,
        script_index: std::sync::atomic::AtomicUsize,
        request_count: std::sync::atomic::AtomicUsize,
        chunk_delay_ms: u64,
    }

    impl PacedScriptedProvider {
        fn new(
            caps: ModelCapabilities,
            per_call_scripts: Vec<Vec<faktor_provider::ScriptedResponse>>,
            chunk_delay_ms: u64,
        ) -> Arc<Self> {
            Arc::new(Self {
                inner: FakeProvider::new("fake", caps),
                calls: std::sync::Mutex::new(per_call_scripts),
                script_index: std::sync::atomic::AtomicUsize::new(0),
                request_count: std::sync::atomic::AtomicUsize::new(0),
                chunk_delay_ms,
            })
        }
    }

    impl faktor_provider::Provider for PacedScriptedProvider {
        fn id(&self) -> &str {
            "fake"
        }
        fn capabilities(&self, model: &str) -> ModelCapabilities {
            self.inner.capabilities(model)
        }
        fn stream(
            &self,
            _req: faktor_provider::GenericAgentRequest,
        ) -> faktor_provider::ProviderStream {
            use futures_util::StreamExt;
            self.request_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let i = self
                .script_index
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let script: Vec<faktor_provider::ScriptedResponse> = self
                .calls
                .lock()
                .unwrap()
                .get(i)
                .cloned()
                .unwrap_or_else(|| vec![faktor_provider::ScriptedResponse::End]);
            let delay = self.chunk_delay_ms;
            let stream = futures_util::stream::iter(script).then(move |s| async move {
                if delay > 0 {
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                }
                match s {
                    faktor_provider::ScriptedResponse::Text(t) => {
                        Ok(faktor_provider::ProviderChunk::Text { text: t })
                    }
                    faktor_provider::ScriptedResponse::ToolCall { id, name, input } => {
                        Ok(faktor_provider::ProviderChunk::ToolCall {
                            id,
                            name,
                            input,
                            complete: true,
                        })
                    }
                    faktor_provider::ScriptedResponse::Die(e) => Err(e),
                    faktor_provider::ScriptedResponse::End => {
                        Ok(faktor_provider::ProviderChunk::Done)
                    }
                    faktor_provider::ScriptedResponse::Reasoning(_) => unreachable!(),
                }
            });
            Box::pin(stream)
        }
    }

    /// Explicit deterministic failure injection for the frozen-wire
    /// regression test: EVERY stream call fails with a typed provider error
    /// before any chunk is produced. The failure is platform-independent —
    /// it depends on no workspace path, environment or host timing.
    struct AlwaysFailsProvider;

    impl faktor_provider::Provider for AlwaysFailsProvider {
        fn id(&self) -> &str {
            "fake"
        }
        fn capabilities(&self, _model: &str) -> ModelCapabilities {
            ModelCapabilities {
                tools: true,
                ..Default::default()
            }
        }
        fn stream(
            &self,
            _req: faktor_provider::GenericAgentRequest,
        ) -> faktor_provider::ProviderStream {
            let failed: Result<faktor_provider::ProviderChunk, faktor_provider::ProviderError> =
                Err(faktor_provider::ProviderError::new(
                    faktor_provider::ProviderErrorKind::Malformed,
                    "frozen wire: injected provider failure (deterministic)",
                ));
            Box::pin(futures_util::stream::iter(vec![failed]))
        }
    }

    struct AllowAll;
    impl faktor_agent::PermissionRequester for AllowAll {
        fn request(
            &self,
            _session: SessionId,
            _permission: &faktor_session::ops::PermissionRequest,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = faktor_core::Result<PermissionDecision>> + Send>,
        > {
            Box::pin(async { Ok(PermissionDecision::Allow) })
        }
    }

    /// The echo tool the paced roundtrips call (a second reasoning
    /// iteration gives the control boundary a real window).
    fn echo_tool() -> faktor_agent::Tool {
        faktor_agent::Tool {
            name: "echo".into(),
            description: "echo its input".into(),
            input_schema: serde_json::json!({"type": "object"}),
            resource_class: faktor_core::resource::ResourceClass::Cpu,
            capability: None,
            recovery_hint: faktor_agent::RecoveryHint::Idempotent,
            path_args: vec![],
            execute: Arc::new(
                |_ctx: faktor_agent::ToolRunCtx, _input: serde_json::Value| {
                    Box::pin(async move { Ok(faktor_agent::ToolOutcome::default()) })
                },
            ),
        }
    }

    /// Server deps whose ONLY provider is the paced scripted one (a
    /// deterministic control window for real orchestrated drives).
    fn paced_test_deps(root: &std::path::Path, paced: Arc<PacedScriptedProvider>) -> ServerDeps {
        let mut registry = faktor_provider::ProviderRegistry::new();
        registry.try_register(paced).unwrap();
        let session = SessionManager::open(root.join("store"), root.join("cas"), true).unwrap();
        let permissions = ChannelPermissionRequester::new(Duration::from_secs(5));
        let mut tools = faktor_agent::ToolRegistry::new();
        tools.register(echo_tool());
        let agent = AgentRuntime::new(faktor_agent::AgentDeps {
            session: session.clone(),
            providers: Arc::new(registry),
            chunk_sink: None,
            permission_requester: Arc::new(AllowAll),
            evidence: Arc::new(faktor_agent::NoEvidence),
            tools: Arc::new(tools),
            cas: None,
            workspaces: faktor_fs::WorkspaceFileService::new(),
            edit: None,
            snapshots: None,
            sandbox: None,
            supervisor: None,
            verification: faktor_agent::VerificationService::disabled(),
            model: "m".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "You are a test server agent.".into(),
            hooks: None,
            instructions_resolver: faktor_instructions::no_roots_resolver(),
            routing: faktor_agent::FixedRoutingPolicy::passthrough(),
            budgets: Arc::new(faktor_session::NoopBudget),
            clock: Arc::new(faktor_core::time::SystemClock),
            tool_call_mode: faktor_agent::ToolCallMode::Native,
            tool_deadline_ms: 2000,
            retry_policy: faktor_core::retry::RetryPolicy::default(),
            semantic: faktor_agent::fallback_semantic_registry(),
            context_prior: None,
            efficiency: Default::default(),
        })
        .unwrap();
        let (orchestrator, tasks) = orch_pair(session.clone(), agent.clone());
        ServerDeps {
            budgets: faktor_session::DurableBudgetLedger::new(session.clone()),
            session,
            agent,
            permissions,
            orchestrator,
            tasks,
            auth_token: AuthToken::generate(),
            server_password: ServerPassword::generate(),
            directory: None,
            version: "0.1.0".into(),
            fs: None,
            snapshots: None,
            chunk_rx: None,
            simulate_not_ready: false,
            evidence: None,
            semantic: None,
        }
    }

    fn read_workspace_caps() -> faktor_orchestrator::caps::CapabilitySet {
        use faktor_orchestrator::caps::{CapabilityGrant, LatticeCap, ScopePattern};
        faktor_orchestrator::caps::CapabilitySet::from_grants(vec![CapabilityGrant::new(
            LatticeCap::ReadWorkspace,
            ScopePattern::new("*").unwrap(),
        )])
        .unwrap()
    }

    fn read_child_spec(item: &str) -> faktor_orchestrator::runtime::ChildSpec {
        let mut s = faktor_orchestrator::runtime::ChildSpec::new(item);
        s.child_caps = read_workspace_caps();
        s.task_caps = read_workspace_caps();
        s
    }

    /// A real owner session for an orchestrated run (registered worktree on
    /// a real directory).
    fn orch_owner_env(
        manager: &Arc<SessionManager>,
        root: &std::path::Path,
    ) -> (
        SessionId,
        faktor_orchestrator::runtime::OwnerContext,
        std::path::PathBuf,
    ) {
        use faktor_core::id::{TaskId, WorktreeId};
        let owner_dir = root.join("owner");
        std::fs::create_dir_all(&owner_dir).unwrap();
        let ws = manager
            .create_workspace(owner_dir.to_str().unwrap())
            .unwrap();
        let wt = WorktreeId::new(
            manager
                .put_worktree(ws, owner_dir.to_str().unwrap(), "main")
                .unwrap() as u64,
        );
        let parent = manager
            .create_session(ws, "orch-owner", "fake", "m")
            .unwrap()
            .id();
        manager.adopt_identity(parent, wt, TaskId::new(1)).unwrap();
        let isolated = root.join("isolated");
        std::fs::create_dir_all(&isolated).unwrap();
        (
            parent,
            faktor_orchestrator::runtime::OwnerContext {
                parent_session: parent,
                workspace_id: ws.raw(),
                worktree_id: wt.raw(),
                root: owner_dir,
            },
            isolated,
        )
    }

    fn analysis_plan(items: &[&str]) -> faktor_orchestrator::TaskPlan {
        use faktor_orchestrator::{OwnershipSpec, WorkItem, WorkKind};
        faktor_orchestrator::TaskPlan {
            goal: "Ship the analysis".into(),
            non_goals: vec![],
            constraints: vec![],
            work_items: items
                .iter()
                .map(|id| WorkItem {
                    id: id.to_string(),
                    summary: format!("work {id}"),
                    depends_on: vec![],
                    kind: WorkKind::Analysis,
                    ownership: OwnershipSpec::NoWrites,
                    required_capabilities: faktor_orchestrator::caps::CapabilitySet::new(),
                    acceptance_checks: vec![],
                    completion: faktor_orchestrator::WorkState::Pending,
                })
                .collect(),
        }
    }

    async fn get_agents(base: &str, token: &str, path: &str) -> serde_json::Value {
        let client = reqwest::Client::new();
        let resp = client
            .get(format!("{base}{path}"))
            .bearer_auth(token)
            .send()
            .await
            .unwrap();
        if resp.status() != 200 {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            panic!("{path} -> {status} body: {body}");
        }
        resp.json().await.unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn native_agents_lists_and_controls_real_children_mid_flight() {
        let dir = tempfile::tempdir().unwrap();
        // Real children over the server's runtime: paced two-iteration
        // roundtrips keep the drives mid-flight long enough for HTTP
        // controls to land deterministically.
        let paced = PacedScriptedProvider::new(
            ModelCapabilities {
                tools: true,
                ..Default::default()
            },
            vec![
                vec![
                    faktor_provider::ScriptedResponse::Text("analyzing".into()),
                    faktor_provider::ScriptedResponse::ToolCall {
                        id: "c1".into(),
                        name: "echo".into(),
                        input: serde_json::json!({"text": "hello"}),
                    },
                    faktor_provider::ScriptedResponse::End,
                ],
                vec![
                    faktor_provider::ScriptedResponse::Text("done".into()),
                    faktor_provider::ScriptedResponse::End,
                ],
                vec![faktor_provider::ScriptedResponse::End],
            ],
            4000,
        );
        let deps = paced_test_deps(dir.path(), paced);
        let orch = deps.orchestrator.clone();
        let manager = deps.session.clone();
        let token = deps.auth_token.clone();
        let handle = serve(deps, 0).await.unwrap();
        let base = format!("http://{}", handle.addr);
        let (parent, owner, isolated) = orch_owner_env(&manager, dir.path());

        let config = faktor_orchestrator::runtime::ExecConfig {
            run_id: "run-http".into(),
            ceilings: faktor_orchestrator::runtime::Ceilings::default(),
            parent_caps: read_workspace_caps(),
            provider: "fake".into(),
            default_model: "m".into(),
            isolated_root: isolated.clone(),
            crash_seam: None,
        };
        let plan = analysis_plan(&["a", "b"]);
        let specs = vec![read_child_spec("a"), read_child_spec("b")];
        let run = tokio::spawn(async move {
            orch.execute_task(plan, owner, config, &specs)
                .await
                .unwrap()
        });

        // The agent listing shows the parent's own run + both children.
        let client = reqwest::Client::new();
        let mut entries = Vec::new();
        for _ in 0..200 {
            let resp = client
                .get(format!("{base}/native/agents?session={parent}"))
                .bearer_auth(token.as_str())
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 200);
            let v: serde_json::Value = resp.json().await.unwrap();
            let kids: Vec<&serde_json::Value> = v
                .as_array()
                .unwrap()
                .iter()
                .filter(|e| e["kind"] == "child")
                .collect();
            if kids.len() >= 2 {
                entries = v.as_array().unwrap().clone();
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        assert_eq!(entries.len(), 3, "self + two children: {entries:?}");
        assert_eq!(entries[0]["kind"], "self");
        assert_eq!(entries[0]["run_id"], "run-http");
        assert_eq!(entries[0]["ownership"], "self");
        assert_eq!(entries[0]["state"], "Running");
        assert_eq!(entries[0]["goal"], "Ship the analysis");
        assert!(entries[1]["item_id"] == "a" || entries[2]["item_id"] == "a");
        // Children carry real session ids, worktree identity, live model
        // and progress while their drives are in flight. The provider is the
        // child session's OWN durable row (the catalog join key), so it must
        // match the run's provider even when another provider serves the same
        // model id.
        for e in entries.iter().filter(|e| e["kind"] == "child") {
            assert_eq!(e["run_id"], "run-http");
            assert_ne!(e["session_id"].as_u64().unwrap_or(0), 0);
            assert_eq!(e["ownership"], "read_only_shared");
            assert_eq!(e["state"], "Running");
            assert_eq!(e["model"], "m");
            assert_eq!(e["provider"], "fake");
        }

        // Mid-flight budget change: applied synchronously and durably
        // visible on the child row through the listing.
        let resp = client
            .post(format!("{base}/native/agents/child-0/budget"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"max_tokens": 4321}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let ack: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(ack["queuedSeq"], 1);
        assert_eq!(ack["applied"], true);
        // Steer + model enqueue durably with the exact wire shape and are
        // applied at the drive's next safe reasoning boundary (the pause
        // boundary machine itself is the wave-12 harness's contract; this
        // endpoint test freezes the queue + ack wire and the durable
        // application visible on the child's drive-state row).
        for (path, seq, body) in [
            (
                "steer",
                2,
                Some(serde_json::json!({"text": "focus the api surface"})),
            ),
            ("model", 3, Some(serde_json::json!({"model": "default"}))),
        ] {
            let mut rb = client
                .post(format!("{base}/native/agents/child-0/{path}"))
                .bearer_auth(token.as_str());
            if let Some(b) = body {
                rb = rb.json(&b);
            }
            let resp = rb.send().await.unwrap();
            assert_eq!(resp.status(), 200, "{path}");
            let ack: serde_json::Value = resp.json().await.unwrap();
            assert_eq!(ack["queuedSeq"], seq, "{path}");
            assert!(ack["applied"].is_null(), "{path}");
        }

        // The run completes naturally; both children are Done. The queue
        // wire for steer/model is frozen above; their exactly-once
        // application at a reasoning boundary is the wave-12 runtime
        // harness's contract (pause/steer/cancel/budget drives), exercised
        // in this crate's own suite.
        let outcome = tokio::time::timeout(Duration::from_secs(180), run)
            .await
            .expect("run must settle")
            .expect("executor drive panicked");
        assert!(outcome.complete, "{outcome:?}");

        // The terminal listing reflects the durable rows: the budget patch
        // sits on the child row and both children are Done.
        let resp = client
            .get(format!("{base}/native/agents?session={parent}"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        let v: serde_json::Value = resp.json().await.unwrap();
        let entries = v.as_array().unwrap();
        assert_eq!(entries.len(), 3);
        let c0 = entries
            .iter()
            .find(|e| e["agent_id"] == "child-0")
            .expect("done child listed");
        assert_eq!(c0["state"], "Done");
        assert_eq!(c0["budget"], 4321, "budget change visible on the child row");
        let c1 = entries
            .iter()
            .find(|e| e["agent_id"] == "child-1")
            .expect("done child listed");
        assert_eq!(c1["state"], "Done");
        let root = entries
            .iter()
            .find(|e| e["kind"] == "self")
            .expect("self entry");
        assert_eq!(root["state"], "Done");
        // Pause after the run is a typed terminal refusal once the mirror
        // settled (409, never a silent no-op).
        let resp = client
            .post(format!("{base}/native/agents/child-0/pause"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 409, "terminal children refuse pause");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn native_agents_lists_insession_task_runs_and_empty_only_without_runs() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let token = deps.auth_token.clone();
        let tasks = deps.tasks.clone();
        let manager = deps.session.clone();
        let handle = serve(deps, 0).await.unwrap();
        let base = format!("http://{}", handle.addr);
        let ws = manager.create_workspace("/plain").unwrap();
        let sid = manager
            .create_session(ws, "plain", "fake", "m")
            .unwrap()
            .id();

        // Genuinely no task run -> the empty array (never a phantom).
        let v = get_agents(
            &base,
            token.as_str(),
            &format!("/native/session/{sid}/agents"),
        )
        .await;
        assert_eq!(v, serde_json::json!([]));

        // A TaskExecutor single-item task (the one-work-item case of the
        // SAME executor that spawns orchestrated children) drives this
        // session through the daemon's own prompt path and shows up as the
        // parent's own task run.
        let req = faktor_orchestrator::runtime::task_executor::TaskRunRequest {
            goal: "analyze the module boundaries".into(),
            work_items: vec![faktor_orchestrator::WorkItem::new(
                "a1",
                "analyze the module boundaries",
                faktor_orchestrator::WorkKind::Analysis,
            )],
            ..Default::default()
        };
        let receipt = tasks.start_task(sid, req).expect("single-item start");
        assert_eq!(
            receipt.mode,
            faktor_orchestrator::runtime::task_executor::TaskRunMode::InSession
        );
        assert!(receipt.run_id.starts_with("tx-"));

        // The listing polls to the run's terminal state and shows the
        // session's own run with the session's worktree identity.
        let mut seen = None;
        for _ in 0..200 {
            let v = get_agents(
                &base,
                token.as_str(),
                &format!("/native/agents?session={sid}"),
            )
            .await;
            let entries = v.as_array().unwrap();
            if entries.is_empty() {
                tokio::time::sleep(Duration::from_millis(50)).await;
                continue;
            }
            seen = Some(entries.clone());
            if entries[0]["state"] == "Done" {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let entries = seen.expect("the in-session run must appear");
        assert_eq!(entries.len(), 1);
        let e = &entries[0];
        assert_eq!(e["kind"], "self");
        assert_eq!(e["run_id"], receipt.run_id);
        assert_eq!(e["session_id"], sid.raw());
        assert_eq!(e["state"], "Done");
        assert_eq!(e["goal"], "analyze the module boundaries");
        assert_eq!(e["item_ids"], serde_json::json!(["a1"]));
        assert_eq!(e["ownership"], "self");
        assert!(e["budget"].is_null());
        let session_row = manager.get_session(sid).unwrap().unwrap().row().unwrap();
        assert_eq!(e["worktree_id"], session_row.worktree_id.raw());
        // The path-id form lists the same truth (progress ticks between
        // polls, so the live fields are compared individually).
        let v2 = get_agents(
            &base,
            token.as_str(),
            &format!("/native/session/{sid}/agents"),
        )
        .await;
        let e2 = &v2.as_array().unwrap()[0];
        for key in [
            "agent_id",
            "kind",
            "run_id",
            "session_id",
            "worktree_id",
            "goal",
            "state",
            "model",
            "budget",
            "ownership",
        ] {
            assert_eq!(e2.get(key), e.get(key), "{key}");
        }
        assert_eq!(e2["item_ids"], e["item_ids"]);
        // Hostile session ids are typed 404 on the query endpoint.
        for hostile in ["abc", "0", "-1", "999999999", "1;drop"] {
            let client = reqwest::Client::new();
            let resp = client
                .get(format!("{base}/native/agents?session={hostile}"))
                .bearer_auth(token.as_str())
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 404, "hostile {hostile:?}");
        }
        let _ = handle.shutdown.send(());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn native_agent_control_guards_and_hostile_inputs_are_typed() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let orch = deps.orchestrator.clone();
        let token = deps.auth_token.clone();
        let manager = deps.session.clone();
        let handle = serve(deps, 0).await.unwrap();
        let base = format!("http://{}", handle.addr);
        let (parent, owner, isolated) = orch_owner_env(&manager, dir.path());

        // A run whose executor crashed right after the child was created
        // (BeforeDrive seam): the durable child row is live (Running) and
        // the mirror is parked — a deterministic control window without a
        // racing drive.
        let config = faktor_orchestrator::runtime::ExecConfig {
            run_id: "run-seam".into(),
            ceilings: faktor_orchestrator::runtime::Ceilings::default(),
            parent_caps: read_workspace_caps(),
            provider: "fake".into(),
            default_model: "m".into(),
            isolated_root: isolated.clone(),
            crash_seam: Some(faktor_orchestrator::runtime::CrashSeam::BeforeDrive),
        };
        let plan = analysis_plan(&["a"]);
        let specs = vec![read_child_spec("a")];
        let res = orch
            .execute_task(plan, owner, config, &specs)
            .await
            .expect_err("the seam must fire");
        assert!(
            matches!(
                res,
                faktor_orchestrator::runtime::ExecError::InjectedCrashSeam(_)
            ),
            "{res:?}"
        );

        // Hostile child ids and bodies: typed 404/400, never a panic.
        let client = reqwest::Client::new();
        let post = |path: &str, body: Option<serde_json::Value>| {
            let mut rb = client
                .post(format!("{base}{path}"))
                .bearer_auth(token.as_str());
            if let Some(b) = body {
                rb = rb.json(&b);
            }
            rb.send()
        };
        for path in [
            "/native/agents/child-9/pause",
            "/native/agents/nope/cancel",
            "/native/agents/child-0/../../pause",
        ] {
            let resp = post(path, None).await.unwrap();
            assert_eq!(resp.status(), 404, "{path}");
        }
        let resp = post("/native/agents/child-0/budget", Some(serde_json::json!({})))
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);
        // A cost budget change is a synchronous durable effect (audit 9/H):
        // the child's task-row `max_cost_micro` is patched and its budget
        // scope is enrolled under the run root. No queue row exists for it
        // (queuedSeq null) and the token axis is untouched.
        let resp = post(
            "/native/agents/child-0/budget",
            Some(serde_json::json!({"max_cost_micro": 500})),
        )
        .await
        .unwrap();
        assert_eq!(resp.status(), 200);
        let ack: serde_json::Value = resp.json().await.unwrap();
        assert!(ack["queuedSeq"].is_null(), "{ack}");
        assert_eq!(ack["applied"], true);
        let v = get_agents(
            &base,
            token.as_str(),
            &format!("/native/agents?session={parent}"),
        )
        .await;
        let child_session = v
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["agent_id"] == "child-0")
            .and_then(|e| e["session_id"].as_u64())
            .expect("child-0 listed with its session");
        let ledger = faktor_session::DurableBudgetLedger::new(manager.clone());
        let cost_view = ledger
            .session_budget_view(
                SessionId::new(child_session),
                faktor_core::id::TaskId::new(1),
            )
            .expect("durable budget view");
        assert_eq!(
            cost_view.max_cost_micro,
            Some(500),
            "the cost change landed on the child's durable task-row cap"
        );
        assert_eq!(
            ledger
                .scope_of(SessionId::new(child_session))
                .unwrap()
                .map(|s| s.child_id),
            Some("child-0".to_string()),
            "the child is enrolled under its run root"
        );
        // Zero on either axis is ambiguous (the store reads 0 as unlimited)
        // and refuses typed on both axes.
        let resp = post(
            "/native/agents/child-0/budget",
            Some(serde_json::json!({"max_cost_micro": 0})),
        )
        .await
        .unwrap();
        assert_eq!(resp.status(), 400);
        let resp = post(
            "/native/agents/child-0/budget",
            Some(serde_json::json!({"max_tokens": 0})),
        )
        .await
        .unwrap();
        assert_eq!(resp.status(), 400);
        let resp = post(
            "/native/agents/child-0/budget",
            Some(serde_json::json!({"max_tokens": 5, "max_cost_micro": 5})),
        )
        .await
        .unwrap();
        assert_eq!(resp.status(), 400);
        let resp = post(
            "/native/agents/child-0/steer",
            Some(serde_json::json!({"text": "x".repeat(501)})),
        )
        .await
        .unwrap();
        assert_eq!(resp.status(), 400, "steering notes are bounded");
        let resp = post(
            "/native/agents/child-0/model",
            Some(serde_json::json!({"model": "gpt-99"})),
        )
        .await
        .unwrap();
        assert_eq!(resp.status(), 404, "model not in the provider registry");
        let resp = post("/native/agents/child-0/retry", None).await.unwrap();
        assert_eq!(resp.status(), 409, "only Failed children retry");
        let resp = post("/native/agents/child-0/resume", None).await.unwrap();
        assert_eq!(resp.status(), 200);

        // Valid controls enqueue durably with the exactly-once ack shape.
        let resp = post("/native/agents/child-0/pause", None).await.unwrap();
        assert_eq!(resp.status(), 200);
        let ack: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(ack["queuedSeq"], 2, "the earlier resume took seq 1");
        assert!(ack["applied"].is_null());
        let resp = post(
            "/native/agents/child-0/steer",
            Some(serde_json::json!({"text": "look at the seam"})),
        )
        .await
        .unwrap();
        let ack: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(ack["queuedSeq"], 3);
        assert!(ack["applied"].is_null());
        let resp = post(
            "/native/agents/child-0/model",
            Some(serde_json::json!({"model": "default"})),
        )
        .await
        .unwrap();
        let ack: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(ack["queuedSeq"], 4);
        assert!(ack["applied"].is_null());
        let resp = post(
            "/native/agents/child-0/budget",
            Some(serde_json::json!({"max_tokens": 99})),
        )
        .await
        .unwrap();
        let ack: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(ack["queuedSeq"], 5);
        assert_eq!(ack["applied"], true);
        let resp = post("/native/agents/child-0/cancel", None).await.unwrap();
        let ack: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(ack["queuedSeq"], 6);
        assert_eq!(ack["applied"], true);

        // The listing reflects the durable rows (budget patch on the child
        // row; the run's own entry derives from the children).
        let v = get_agents(
            &base,
            token.as_str(),
            &format!("/native/agents?session={}", parent),
        )
        .await;
        let entries = v.as_array().unwrap();
        assert_eq!(entries.len(), 2);
        let child = entries
            .iter()
            .find(|e| e["agent_id"] == "child-0")
            .expect("child listed");
        assert_eq!(
            child["budget"], 99,
            "budget change visible on the child row"
        );
        assert_eq!(child["run_id"], "run-seam");
        let root = entries
            .iter()
            .find(|e| e["kind"] == "self")
            .expect("self entry");
        assert_eq!(root["run_id"], "run-seam");
        assert_eq!(root["state"], "Running");
        let _ = handle.shutdown.send(());
    }

    // ------------------------------------------------- audits P0-62/63/64

    async fn native_get(
        client: &reqwest::Client,
        base: &str,
        token: &AuthToken,
        path: &str,
    ) -> reqwest::Response {
        client
            .get(format!("{base}{path}"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap()
    }

    /// Spawn a session-owned terminal through the native endpoint. Returns
    /// `None` when the platform refuses PTY spawns (documented skip).
    async fn native_spawn_terminal(
        client: &reqwest::Client,
        base: &str,
        token: &AuthToken,
        sid: &str,
        command: &str,
        args: &[&str],
    ) -> Option<serde_json::Value> {
        let resp = client
            .post(format!("{base}/native/session/{sid}/terminal"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({ "command": command, "args": args }))
            .send()
            .await
            .unwrap();
        if resp.status() != 200 {
            assert_eq!(resp.status(), 400, "platform refusal is a 400");
            return None;
        }
        Some(resp.json().await.unwrap())
    }

    async fn native_remove_terminal(
        client: &reqwest::Client,
        base: &str,
        token: &AuthToken,
        pty_id: &str,
    ) {
        let resp = client
            .post(format!("{base}/pty/remove"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({ "pty_id": pty_id }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }

    /// Create a typed task row of a session (task machine semantics: only
    /// creation; revision starts at 1).
    fn seed_typed_task(
        handle: &faktor_session::SessionHandle,
        task_id: u64,
        max_tokens: Option<u64>,
        max_turns: Option<u32>,
        goal: &str,
    ) {
        let now = handle.now_ms();
        handle
            .create_task(faktor_session::Task {
                task_id: faktor_core::id::TaskId::new(task_id),
                session_id: handle.id(),
                goal: goal.into(),
                acceptance_criteria: vec![],
                plan: vec![],
                budget: faktor_session::TaskBudget {
                    max_tokens,
                    max_turns,
                    spent_tokens: 0,
                    spent_turns: 0,
                },
                state: faktor_core::state::TaskState::Running,
                created_ms: now,
                updated_ms: now,
            })
            .unwrap();
    }

    #[tokio::test]
    async fn native_terminal_ownership_scope_isolation_and_lifetime_events() {
        // P0-62: terminals spawned under session A carry {session_id,
        // task_id, agent_id, operation_id} ownership; the session-scoped
        // view of B never shows A's terminal even when the daemon owns both,
        // and the bounded lifetime log is session-scoped too. Hostile ids,
        // bounds and strict DTO rejections behave like every native route.
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let token = deps.auth_token.clone();
        let manager = deps.session.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let ws = manager.create_workspace("/nt-root").unwrap();
        let a = manager.create_session(ws, "t-pty-a", "fake", "m").unwrap();
        let b = manager.create_session(ws, "t-pty-b", "fake", "m").unwrap();
        let a_sid = a.id().to_string();
        let b_sid = b.id().to_string();

        // Hostile/unknown sessions and strict DTO checks first.
        for path in [
            "/native/terminals?session=0",
            "/native/terminals?session=abc",
        ] {
            let resp = native_get(&client, &base, &token, path).await;
            assert_eq!(resp.status(), 400, "{path}");
        }
        let resp = native_get(&client, &base, &token, "/native/terminals?session=999999").await;
        assert_eq!(resp.status(), 404);
        let resp = native_get(&client, &base, &token, "/native/terminals?session=1&limt=2").await;
        assert_eq!(resp.status(), 400, "unknown query field is a 400");

        // Spawn owned terminals under A and B.
        let Some(pty_a) =
            native_spawn_terminal(&client, &base, &token, &a_sid, "/bin/sleep", &["60"]).await
        else {
            // Platform refusal: the session-scoped view stays empty + the
            // documented unowned/note shape still serves.
            let resp = native_get(
                &client,
                &base,
                &token,
                &format!("/native/terminals?session={a_sid}"),
            )
            .await;
            assert_eq!(resp.status(), 200);
            let body: serde_json::Value = resp.json().await.unwrap();
            assert_eq!(body["terminals"], serde_json::json!([]));
            assert_eq!(body["unowned"], 0);
            let _ = handle.shutdown.send(());
            return;
        };
        let pty_a_id = pty_a["ptyId"].as_str().unwrap().to_string();
        let pid_a = pty_a["pid"].as_u64().unwrap();
        assert!(pid_a > 0);
        assert_eq!(pty_a["sessionId"], a_sid);
        assert_eq!(pty_a["taskId"], "1", "standalone session task identity");
        assert!(pty_a["agentId"].is_null());
        let op_a = pty_a["operationId"].as_str().unwrap().to_string();
        assert!(!op_a.is_empty());

        let Some(pty_b) =
            native_spawn_terminal(&client, &base, &token, &b_sid, "/bin/sleep", &["60"]).await
        else {
            let _ = native_remove_terminal(&client, &base, &token, &pty_a_id).await;
            let _ = handle.shutdown.send(());
            return;
        };
        let pty_b_id = pty_b["ptyId"].as_str().unwrap().to_string();

        // A's scoped view: ONLY A's terminal with full ownership.
        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/terminals?session={a_sid}"),
        )
        .await;
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["sessionId"], a_sid);
        assert_eq!(body["unowned"], 0);
        assert_eq!(body["note"], "");
        let rows = body["terminals"].as_array().unwrap();
        assert_eq!(rows.len(), 1, "{body}");
        assert_eq!(rows[0]["id"], pty_a_id);
        assert_eq!(rows[0]["pid"], pid_a);
        assert_eq!(rows[0]["alive"], true);
        assert_eq!(rows[0]["sessionId"], a_sid);
        assert_eq!(rows[0]["taskId"], "1");
        assert_eq!(rows[0]["operationId"], op_a);
        assert!(rows[0]["spawnedMs"].as_i64().unwrap_or(0) > 0);
        assert!(rows[0]["agentId"].is_null());

        // B's scoped view NEVER contains A's terminal although the daemon
        // owns both: B sees exactly its own row.
        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/terminals?session={b_sid}"),
        )
        .await;
        let body: serde_json::Value = resp.json().await.unwrap();
        let rows = body["terminals"].as_array().unwrap();
        assert_eq!(rows.len(), 1, "B sees only its own terminal: {body}");
        assert_eq!(rows[0]["id"], pty_b_id);
        assert!(
            rows.iter().all(|r| r["id"] != pty_a_id),
            "A's terminal must never surface in B's view: {body}"
        );

        // A's view still holds only A's terminal after B spawned one.
        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/terminals?session={a_sid}"),
        )
        .await;
        let body: serde_json::Value = resp.json().await.unwrap();
        let rows = body["terminals"].as_array().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["id"], pty_a_id);

        // The legacy daemon-level view lists both (frozen pre-P0-62 shape)
        // and annotates ownership additively when known.
        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/session/{a_sid}/terminal"),
        )
        .await;
        assert_eq!(resp.status(), 200);
        let rows: serde_json::Value = resp.json().await.unwrap();
        let rows = rows.as_array().unwrap();
        assert_eq!(rows.len(), 2, "daemon view lists both PTYs: {rows:?}");
        let mine = rows
            .iter()
            .find(|r| r["id"] == pty_a_id)
            .expect("A's terminal in the daemon view");
        assert_eq!(mine["sessionId"], a_sid, "ownership annotated: {mine}");
        let other = rows
            .iter()
            .find(|r| r["id"] == pty_b_id)
            .expect("B's terminal in the daemon view");
        assert_eq!(other["sessionId"], b_sid);

        // Session-scoped lifetime events: created frames with the terminal
        // ownership; B's log never contains A's frames.
        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/session/{a_sid}/terminal/events"),
        )
        .await;
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["sessionId"], a_sid);
        let events = body["events"].as_array().unwrap();
        assert_eq!(events.len(), 1, "{body}");
        assert_eq!(events[0]["type"], "created");
        assert_eq!(events[0]["ptyId"], pty_a_id);
        assert_eq!(events[0]["pid"], pid_a);
        assert_eq!(body["hasMore"], false);
        assert!(body["nextCursor"].is_null());
        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/session/{b_sid}/terminal/events"),
        )
        .await;
        let body: serde_json::Value = resp.json().await.unwrap();
        let events = body["events"].as_array().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0]["ptyId"], pty_b_id,
            "no cross-session frames: {body}"
        );

        // Exited events: kill A's terminal (compat control keeps working on
        // native-owned rows), then the next native read sweeps and logs it.
        native_remove_terminal(&client, &base, &token, &pty_a_id).await;
        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/session/{a_sid}/terminal/events"),
        )
        .await;
        let body: serde_json::Value = resp.json().await.unwrap();
        let events = body["events"].as_array().unwrap();
        assert_eq!(events.len(), 2, "created + exited: {body}");
        assert_eq!(events[1]["type"], "exited");
        assert_eq!(events[1]["ptyId"], pty_a_id);
        assert_eq!(events[1]["pid"], pid_a, "exit event keeps the spawn pid");
        assert_eq!(
            body["hasMore"], false,
            "the whole log fits one page: {body}"
        );
        assert!(body["nextCursor"].is_null());
        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/terminals?session={a_sid}"),
        )
        .await;
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["terminals"], serde_json::json!([]), "{body}");

        // Bounds + hostile ids + strict DTO on the event log and spawn.
        for path in [
            format!("/native/session/{a_sid}/terminal/events?limit=0"),
            format!("/native/session/{a_sid}/terminal/events?limit=201"),
            format!("/native/session/{a_sid}/terminal/events?limt=5"),
            "/native/session/abc/terminal/events".to_string(),
            "/native/session/0/terminal/events".to_string(),
        ] {
            let resp = native_get(&client, &base, &token, &path).await;
            assert_eq!(resp.status(), 400, "{path}");
        }
        let resp = native_get(
            &client,
            &base,
            &token,
            "/native/session/999999/terminal/events",
        )
        .await;
        assert_eq!(resp.status(), 404);
        // Unknown spawn session / hostile body / unknown body field.
        let resp = client
            .post(format!("{base}/native/session/999999/terminal"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"command": "/bin/sleep", "args": ["1"]}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);
        let resp = client
            .post(format!("{base}/native/session/{a_sid}/terminal"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"command": "/bin/sleep", "bogus": true}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400, "unknown body field is a 400");
        let resp = client
            .post(format!("{base}/native/session/{a_sid}/terminal"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"args": []}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400, "missing command");
        let resp = client
            .post(format!("{base}/native/session/{a_sid}/terminal"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"command": "x".repeat(5000)}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400, "oversized command rejected");
        let resp = client
            .post(format!("{base}/native/session/{a_sid}/terminal"))
            .json(&serde_json::json!({"command": "/bin/sleep", "args": ["1"]}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);

        // Cleanup: kill B's terminal so no test child outlives the test.
        native_remove_terminal(&client, &base, &token, &pty_b_id).await;
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn native_messages_cursor_paging_no_dup_gap_and_isolation() {
        // P0-64a: cursor pages over the durable message rows. A 100-row
        // fixture pages with hasMore/nextBefore semantics — every row
        // appears exactly once (no duplicate, no gap); hostile session ids,
        // oversized limits and unknown query fields are rejected; session B
        // never sees A's rows.
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let token = deps.auth_token.clone();
        let manager = deps.session.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let ws = manager.create_workspace("/msg-root").unwrap();
        let a = manager.create_session(ws, "t-msg-a", "fake", "m").unwrap();
        let b = manager.create_session(ws, "t-msg-b", "fake", "m").unwrap();
        let a_sid = a.id().to_string();
        let b_sid = b.id().to_string();
        let ha = manager.get_session(a.id()).unwrap().unwrap();
        let hb = manager.get_session(b.id()).unwrap().unwrap();
        // 100 durable rows under A (seq 1..=100), one with a text part.
        for seq in 1..=100i64 {
            let mid = ha
                .put_message(
                    seq,
                    if seq % 2 == 0 { "assistant" } else { "user" },
                    serde_json::json!({"text": format!("m{seq}")}),
                )
                .unwrap();
            if seq == 50 {
                ha.put_text_part(mid, "part-of-50").unwrap();
            }
        }
        for seq in 1..=3i64 {
            hb.put_message(seq, "user", serde_json::json!({"text": format!("b{seq}")}))
                .unwrap();
        }

        // Page across the whole 100-row fixture.
        let mut seen: Vec<i64> = Vec::new();
        let mut before: Option<i64> = None;
        let mut pages = 0;
        loop {
            let cursor = before.map(|b| format!("&before={b}")).unwrap_or_default();
            let resp = native_get(
                &client,
                &base,
                &token,
                &format!("/native/messages?session={a_sid}&limit=30{cursor}"),
            )
            .await;
            assert_eq!(resp.status(), 200);
            let body: serde_json::Value = resp.json().await.unwrap();
            assert_eq!(body["sessionId"], a_sid);
            let msgs = body["messages"].as_array().unwrap();
            pages += 1;
            assert!(!msgs.is_empty());
            assert!(msgs.len() <= 30);
            for m in msgs {
                let seq = m["seq"].as_i64().unwrap();
                assert!(seen.last().map(|s| seq < *s).unwrap_or(true), "descending");
                assert!(seen.iter().all(|s| *s != seq), "no duplicate {seq}");
                seen.push(seq);
                assert!(m["role"].is_string());
                assert!(m["createdMs"].as_i64().unwrap_or(0) > 0);
                assert_eq!(m["data"]["text"], format!("m{seq}"));
            }
            let has_more = body["hasMore"].as_bool().unwrap();
            before = body["nextBefore"].as_i64();
            if !has_more {
                assert!(before.is_none());
                break;
            }
            assert!(before.is_some(), "next page cursor present");
            assert!(pages < 10, "paging must terminate");
        }
        assert_eq!(seen.len(), 100, "every row exactly once");
        assert_eq!(*seen.first().unwrap(), 100);
        assert_eq!(*seen.last().unwrap(), 1);

        // The part row came through with its part.
        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/messages?session={a_sid}&limit=200&before=51"),
        )
        .await;
        let body: serde_json::Value = resp.json().await.unwrap();
        let msgs = body["messages"].as_array().unwrap();
        let m50 = msgs.iter().find(|m| m["seq"] == 50).unwrap();
        assert_eq!(m50["parts"][0]["kind"], "text");
        assert_eq!(m50["parts"][0]["data"]["text"], "part-of-50");

        // Isolation: B's pages contain only B's rows.
        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/messages?session={b_sid}&limit=10"),
        )
        .await;
        let body: serde_json::Value = resp.json().await.unwrap();
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        assert!(msgs
            .iter()
            .all(|m| m["data"]["text"].as_str().unwrap().starts_with('b')));

        // Hostile ids, oversized limits, strict DTO, auth.
        for path in [
            "/native/messages?session=0".to_string(),
            "/native/messages?session=abc".to_string(),
            "/native/messages?session=1&limit=0".to_string(),
            "/native/messages?session=1&limit=201".to_string(),
            "/native/messages?session=1&before=0".to_string(),
            "/native/messages?session=1&before=-3".to_string(),
            "/native/messages?session=1&limt=5".to_string(),
        ] {
            let resp = native_get(&client, &base, &token, &path).await;
            assert_eq!(resp.status(), 400, "{path}");
        }
        let resp = native_get(&client, &base, &token, "/native/messages?session=999999").await;
        assert_eq!(resp.status(), 404);
        let resp = client
            .get(format!("{base}/native/messages?session={a_sid}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn native_events_journal_page_no_dup_gap_and_bounds() {
        // P0-64b: the native twin of the journal stream pages the durable
        // event rows with seq > after ascending; a 300-event fixture pages
        // without a duplicate or a gap.
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let token = deps.auth_token.clone();
        let manager = deps.session.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let ws = manager.create_workspace("/ev-root").unwrap();
        let a = manager.create_session(ws, "t-ev-a", "fake", "m").unwrap();
        let b = manager.create_session(ws, "t-ev-b", "fake", "m").unwrap();
        let a_sid = a.id().to_string();
        let b_sid = b.id().to_string();
        for _ in 0..300 {
            a.force_append_event(
                faktor_core::event::EventKind::PhaseChanged,
                faktor_core::state::AgentState::WaitingForModel,
                None,
                None,
            )
            .unwrap();
        }
        // Session B gets a small independent journal.
        b.force_append_event(
            faktor_core::event::EventKind::PhaseChanged,
            faktor_core::state::AgentState::WaitingForModel,
            None,
            None,
        )
        .unwrap();

        let mut all: Vec<u64> = Vec::new();
        let mut after: u64 = 0;
        let mut pages = 0;
        loop {
            let resp = native_get(
                &client,
                &base,
                &token,
                &format!("/native/events?session={a_sid}&after={after}&limit=100"),
            )
            .await;
            assert_eq!(resp.status(), 200);
            let body: serde_json::Value = resp.json().await.unwrap();
            assert_eq!(body["sessionId"], a_sid);
            let events = body["events"].as_array().unwrap();
            pages += 1;
            assert!(!events.is_empty());
            assert!(events.len() <= 100);
            for e in events {
                let seq = e["seq"].as_u64().unwrap();
                assert!(all.last().map(|s| seq > *s).unwrap_or(true), "ascending");
                assert!(all.iter().all(|s| *s != seq), "no duplicate {seq}");
                all.push(seq);
                if seq == 1 {
                    assert_eq!(e["kind"], "session_created");
                    assert_eq!(e["state"], "idle");
                } else {
                    assert_eq!(e["kind"], "phase_changed");
                    assert_eq!(e["state"], "waiting_for_model");
                }
                assert!(e["opId"].is_null());
                assert!(e["tsMs"].as_i64().unwrap_or(0) > 0);
            }
            let has_more = body["hasMore"].as_bool().unwrap();
            if has_more {
                after = body["nextCursor"].as_u64().unwrap();
            } else {
                assert!(body["nextCursor"].is_null());
                break;
            }
            assert!(pages < 10, "paging must terminate");
        }
        assert_eq!(all.len(), 301, "session_created + 300 forced events");
        assert_eq!(all[0], 1);
        assert_eq!(*all.last().unwrap(), 301);
        assert!(
            all.windows(2).all(|w| w[1] == w[0] + 1),
            "gapless journal paging"
        );

        // Isolation: B's journal pages only its own events.
        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/events?session={b_sid}&limit=10"),
        )
        .await;
        let body: serde_json::Value = resp.json().await.unwrap();
        let events = body["events"].as_array().unwrap();
        assert_eq!(events.len(), 2, "{body}");

        // Bounds, hostile ids, unknown query fields, auth.
        for path in [
            "/native/events?session=0".to_string(),
            "/native/events?session=abc".to_string(),
            "/native/events?session=1&limit=0".to_string(),
            "/native/events?session=1&limit=257".to_string(),
            "/native/events?session=1&aftr=3".to_string(),
        ] {
            let resp = native_get(&client, &base, &token, &path).await;
            assert_eq!(resp.status(), 400, "{path}");
        }
        let resp = native_get(&client, &base, &token, "/native/events?session=999999").await;
        assert_eq!(resp.status(), 404);
        let resp = client
            .get(format!("{base}/native/events?session={a_sid}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn native_providers_registry_and_health_snapshot() {
        // P0-64c: the registry view carries provider identity, models with
        // real capabilities, source provenance and the honest health
        // snapshot; secrets never leak.
        let dir = tempfile::tempdir().unwrap();
        let mut caps = std::collections::HashMap::new();
        caps.insert(
            "gpt-x".to_string(),
            ModelCapabilities {
                context: 128_000,
                max_output: 16_384,
                tools: true,
                ..Default::default()
            },
        );
        let openai = faktor_openai::OpenAiProvider::build(faktor_openai::OpenAiConfig {
            base_url: "http://127.0.0.1:1/v1".into(),
            api_key: Some("sk-super-secret".into()),
            family: faktor_openai::OpenAiFamily::Chat,
            models: caps,
        });
        let deps = test_deps_with(dir.path(), vec![openai]);
        let token = deps.auth_token.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        let resp = native_get(&client, &base, &token, "/native/providers").await;
        assert_eq!(resp.status(), 200);
        let list: serde_json::Value = resp.json().await.unwrap();
        let entries = list.as_array().unwrap();
        assert!(entries.len() >= 2, "fake + openai registered: {list}");
        // Deterministic order by instance id.
        let ids: Vec<&str> = entries
            .iter()
            .map(|e| e["instanceId"].as_str().unwrap())
            .collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(ids, sorted);
        let fake = entries.iter().find(|e| e["instanceId"] == "fake").unwrap();
        assert_eq!(fake["family"], "fake");
        assert_eq!(fake["health"]["status"], "registered");
        assert!(fake["health"]["note"]
            .as_str()
            .unwrap()
            .contains("adapter-private"));
        assert!(!fake["models"].as_array().unwrap().is_empty());
        let oai = entries
            .iter()
            .find(|e| e["instanceId"] == "openai")
            .unwrap();
        let models = oai["models"].as_array().unwrap();
        let gpt_x = models.iter().find(|m| m["model"] == "gpt-x").unwrap();
        assert_eq!(gpt_x["context"], 128_000);
        assert_eq!(gpt_x["source"], "providerCatalog");
        assert_eq!(oai["runtimeContextLimitSupported"], false);
        // Secrets never reach this surface.
        let raw = list.to_string().to_lowercase();
        assert!(!raw.contains("sk-super-secret"), "api keys never leak");
        assert!(!raw.contains("api_key"));

        let resp = client
            .get(format!("{base}/native/providers"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn native_usage_reads_durable_rows_exactly_and_survives_reopen() {
        // P0-63: the usage endpoints read the DURABLE rows — provider_call
        // token columns (incl. prefix observations) and the per-task
        // cost_reservation rows with their route decisions. Numbers match
        // the stored rows exactly, sessions stay isolated, and a reopened
        // store (brand-new manager + server over the same data root) serves
        // the identical JSON.
        let dir = tempfile::tempdir().unwrap();
        let (expected_a, expected_global, sid_a) = {
            let deps = test_deps(dir.path());
            let token = deps.auth_token.clone();
            let manager = deps.session.clone();
            let handle = serve(deps, 0).await.unwrap();
            let client = reqwest::Client::new();
            let base = format!("http://{}", handle.addr);
            let ws_a = manager.create_workspace("/usage-a").unwrap();
            let ws_b = manager.create_workspace("/usage-b").unwrap();
            let a = manager
                .create_session(ws_a, "t-usage-a", "fake", "m")
                .unwrap();
            let b = manager
                .create_session(ws_b, "t-usage-b", "fake", "m")
                .unwrap();
            let ha = manager.get_session(a.id()).unwrap().unwrap();
            let hb = manager.get_session(b.id()).unwrap().unwrap();
            // Typed task rows: A owns task 1 (capped) and an untouched task
            // 2 (null-safe pre-first-reservation view); B owns task 1.
            seed_typed_task(&ha, 1, Some(5000), Some(3), "usage-a");
            seed_typed_task(&ha, 2, Some(1000), None, "usage-a-extra");
            seed_typed_task(&hb, 1, Some(100), None, "usage-b");
            let store = manager.store();
            let ta1 = faktor_core::id::TaskId::new(1);
            let tb1 = faktor_core::id::TaskId::new(1);
            store
                .cost_task_cap_set(a.id(), ta1, Some(1_000_000))
                .unwrap();
            store.cost_task_cap_set(b.id(), tb1, Some(50_000)).unwrap();
            let now = manager.now_ms();
            // A's provider calls: two completed with prefix observations
            // (cacheable-prefix token columns) + one failed row (NULL
            // counters never count).
            let op1 = manager.next_op_id();
            let op2 = manager.next_op_id();
            let op3 = manager.next_op_id();
            ha.settle_usage_with_prefix(
                op1,
                "fake",
                "m",
                "completed",
                Some(900),
                Some(100),
                None,
                Some([7u8; 32]),
                Some(400),
            )
            .unwrap();
            ha.settle_usage_with_prefix(
                op2,
                "fake",
                "m",
                "completed",
                Some(500),
                Some(50),
                None,
                Some([9u8; 32]),
                Some(460),
            )
            .unwrap();
            ha.record_provider_call(op3, "fake", "m", "failed", None, None, Some("boom"))
                .unwrap();
            // B's call: completed WITHOUT a prefix observation.
            let opb = manager.next_op_id();
            hb.settle_usage_with_prefix(
                opb,
                "fake",
                "m",
                "completed",
                Some(7),
                Some(3),
                None,
                None,
                None,
            )
            .unwrap();
            // A's reservations: one settled (with route JSON), one refunded,
            // one left open.
            let faktor_store::CostReserveOutcome::Granted(r1) =
                store.cost_reserve(a.id(), ta1, op1, 5000, now).unwrap()
            else {
                panic!("reserve r1 must be granted");
            };
            store
                .cost_settle(
                    r1,
                    1000,
                    Some(1000),
                    Some(990),
                    Some(
                        r#"{"provider":"fake","model":"m","estimated_cost_micro":90,"estimated_latency_ms":1,"reasoning":"passthrough","considered":1,"source":"configured"}"#,
                    ),
                    now + 1,
                )
                .unwrap();
            let faktor_store::CostReserveOutcome::Granted(r2) =
                store.cost_reserve(a.id(), ta1, op2, 3000, now).unwrap()
            else {
                panic!("reserve r2 must be granted");
            };
            store.cost_refund(r2, now + 2).unwrap();
            let faktor_store::CostReserveOutcome::Granted(_r3) = store
                .cost_reserve(a.id(), ta1, manager.next_op_id(), 200, now)
                .unwrap()
            else {
                panic!("reserve r3 must be granted");
            };
            // B's reservation: one settled with NO provider report.
            let faktor_store::CostReserveOutcome::Granted(rb) =
                store.cost_reserve(b.id(), tb1, opb, 4000, now).unwrap()
            else {
                panic!("reserve rb must be granted");
            };
            store
                .cost_settle(rb, 40, Some(40), None, None, now + 1)
                .unwrap();

            // ---- per-session authoritative usage of A
            let resp = native_get(
                &client,
                &base,
                &token,
                &format!("/native/session/{}/usage", a.id()),
            )
            .await;
            assert_eq!(resp.status(), 200);
            let ua: serde_json::Value = resp.json().await.unwrap();
            assert_eq!(ua["sessionId"], a.id().to_string());
            // provider-call tokens exactly the stored input+output columns.
            assert_eq!(ua["providerCalls"]["tokens"], 1550);
            // prefix observations mirror the store rows.
            let prefix = store.provider_call_prefix_rows(a.id()).unwrap();
            let observed = ua["providerCalls"]["prefixObservations"]
                .as_array()
                .unwrap();
            assert_eq!(observed.len(), prefix.len());
            for (row, o) in prefix.iter().zip(observed) {
                assert_eq!(o["rowId"], row.row_id);
                assert_eq!(o["promptTokens"], row.prompt_tokens as u64);
                assert_eq!(
                    o["stability"],
                    row.prefix_stability
                        .map(|v| serde_json::json!(v))
                        .unwrap_or(serde_json::Value::Null)
                );
            }
            // The stability aggregate equals the store's aggregate (f64
            // equality through the JSON wire is checked within one ulp: the
            // client-side serde_json parser is not the round-trip-exact
            // parser, so a long decimal can land one ulp off the stored
            // double; the endpoint itself serves the store's exact value).
            let agg = store
                .session_stored_prefix_stability(a.id())
                .unwrap()
                .unwrap();
            let ps = &ua["prefixStability"];
            assert_eq!(ps["observations"], agg.observations);
            let near = |x: f64, y: f64| {
                (x - y).abs() <= 2.0 * f64::EPSILON * x.abs().max(y.abs()).max(1.0)
            };
            assert!(
                near(ps["mean"].as_f64().unwrap(), agg.mean),
                "mean differs by more than 1 ulp: {:?} vs {:?}",
                ps["mean"],
                agg.mean
            );
            assert!(
                near(ps["stdDev"].as_f64().unwrap(), agg.std_dev),
                "stdDev differs: {:?} vs {:?}",
                ps["stdDev"],
                agg.std_dev
            );
            // Task entries: durable budget envelope + reservation rows.
            let tasks = ua["tasks"].as_array().unwrap();
            assert_eq!(tasks.len(), 2, "{ua}");
            let t1 = &tasks[0];
            assert_eq!(t1["taskId"], "1");
            assert_eq!(t1["budget"]["maxTokens"], 5000);
            assert_eq!(t1["budget"]["maxTurns"], 3);
            assert_eq!(t1["budget"]["spentTokens"], 0);
            assert_eq!(t1["budget"]["spentCostMicro"], 1000);
            assert_eq!(t1["budget"]["maxCostMicro"], 1_000_000);
            assert_eq!(t1["budget"]["openReservedMicro"], 200);
            let res = &t1["reservations"];
            assert_eq!(res["open"]["count"], 1);
            assert_eq!(res["open"]["predictedMicro"], 200);
            assert_eq!(res["settled"]["count"], 1);
            assert_eq!(res["settled"]["predictedMicro"], 5000);
            assert_eq!(res["settled"]["spentMicro"], 1000);
            assert_eq!(res["settled"]["providerReportedMicro"], 990);
            assert_eq!(res["refunded"]["count"], 1);
            assert_eq!(res["refunded"]["predictedMicro"], 3000);
            assert_eq!(res["uncertain"]["count"], 0);
            let routes = res["routeDecisions"].as_array().unwrap();
            assert_eq!(routes.len(), 1);
            assert_eq!(routes[0]["reservationId"], r1);
            assert_eq!(routes[0]["spentMicro"], 1000);
            assert_eq!(routes[0]["decision"]["provider"], "fake");
            assert_eq!(routes[0]["decision"]["estimated_cost_micro"], 90);
            // Untouched task 2: null-safe pre-first-reservation budget.
            let t2 = &tasks[1];
            assert_eq!(t2["taskId"], "2");
            assert_eq!(t2["budget"]["maxTokens"], 1000);
            assert_eq!(t2["budget"]["maxCostMicro"], serde_json::Value::Null);
            assert_eq!(t2["budget"]["spentCostMicro"], 0);
            assert_eq!(t2["budget"]["openReservedMicro"], 0);
            assert_eq!(t2["reservations"]["settled"]["count"], 0);

            // ---- isolation: B's usage never carries A's rows.
            let resp = native_get(
                &client,
                &base,
                &token,
                &format!("/native/session/{}/usage", b.id()),
            )
            .await;
            let ub: serde_json::Value = resp.json().await.unwrap();
            assert_eq!(ub["providerCalls"]["tokens"], 10, "B only sees its call");
            assert_eq!(
                ub["providerCalls"]["prefixObservations"],
                serde_json::json!([]),
                "B recorded no prefix"
            );
            assert_eq!(ub["tasks"][0]["taskId"], "1");
            assert_eq!(ub["tasks"][0]["budget"]["spentCostMicro"], 40);
            assert_eq!(ub["tasks"][0]["reservations"]["settled"]["count"], 1);
            assert_eq!(ub["tasks"][0]["reservations"]["settled"]["spentMicro"], 40);
            assert_eq!(ub["tasks"].as_array().unwrap().len(), 1);

            // ---- global aggregate: durable numbers over every session.
            let resp = native_get(&client, &base, &token, "/native/usage").await;
            assert_eq!(resp.status(), 200);
            let gu: serde_json::Value = resp.json().await.unwrap();
            assert_eq!(gu["sessions"], 2);
            assert_eq!(gu["durable"]["providerCalls"]["tokens"], 1560);
            assert_eq!(gu["durable"]["providerCalls"]["prefixObservations"], 2);
            assert_eq!(gu["durable"]["providerCalls"]["prefixTokens"], 860);
            assert_eq!(
                gu["durable"]["providerCalls"]["prefixStabilityObservations"],
                2
            );
            assert_eq!(gu["durable"]["taskSpend"]["settledCostMicro"], 1040);
            let res = &gu["durable"]["reservations"];
            assert_eq!(res["settled"]["count"], 2);
            assert_eq!(res["settled"]["spentMicro"], 1040);
            assert_eq!(res["settled"]["providerReportedMicro"], 990);
            assert_eq!(res["refunded"]["count"], 1);
            assert_eq!(res["refunded"]["predictedMicro"], 3000);
            assert_eq!(res["open"]["count"], 1);
            assert_eq!(res["open"]["predictedMicro"], 200);
            assert_eq!(res["uncertain"]["count"], 0);

            // ---- crash-recovery semantics (schema v17+): a crash closes
            // every surviving in-flight reservation split on the durable
            // dispatch marker — this one never left the process
            // (`reserved`, marker NULL), so recovery REFUNDS it (never
            // spent, its prediction released); only dispatched-marker rows
            // go UNCERTAIN. The aggregate follows.
            store.cost_abandon_open_reservations(now + 5).unwrap();
            let resp = native_get(
                &client,
                &base,
                &token,
                &format!("/native/session/{}/usage", a.id()),
            )
            .await;
            let ua_after: serde_json::Value = resp.json().await.unwrap();
            assert_eq!(ua_after["tasks"][0]["budget"]["openReservedMicro"], 0);
            assert_eq!(ua_after["tasks"][0]["budget"]["spentCostMicro"], 1000);
            assert_eq!(ua_after["tasks"][0]["reservations"]["open"]["count"], 0);
            assert_eq!(
                ua_after["tasks"][0]["reservations"]["uncertain"]["count"],
                0
            );
            assert_eq!(ua_after["tasks"][0]["reservations"]["refunded"]["count"], 2);
            assert_eq!(
                ua_after["tasks"][0]["reservations"]["refunded"]["predictedMicro"],
                3200
            );
            let resp = native_get(&client, &base, &token, "/native/usage").await;
            let gu_after: serde_json::Value = resp.json().await.unwrap();
            assert_eq!(gu_after["durable"]["reservations"]["refunded"]["count"], 2);
            assert_eq!(gu_after["durable"]["reservations"]["uncertain"]["count"], 0);

            // Capture the authoritative snapshots for the reopen check.
            let expected_a = ua_after;
            let expected_global = gu_after;
            let _ = handle.shutdown.send(());
            (expected_a, expected_global, a.id())
        };
        // ---- reopen durability: a brand-new manager (and server) over the
        // same data root serves the IDENTICAL usage JSON.
        let deps2 = test_deps(dir.path());
        let token2 = deps2.auth_token.clone();
        let handle2 = serve(deps2, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base2 = format!("http://{}", handle2.addr);
        let resp = native_get(
            &client,
            &base2,
            &token2,
            &format!("/native/session/{sid_a}/usage"),
        )
        .await;
        assert_eq!(resp.status(), 200);
        let reopened: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(reopened, expected_a, "usage survives a reopen exactly");
        let resp = native_get(&client, &base2, &token2, "/native/usage").await;
        let reopened_global: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(reopened_global, expected_global);
        let _ = handle2.shutdown.send(());
    }

    /// Provider for the real-turn usage test (P0-63): a canonical usage
    /// frame with zero uncached input, only cache reads/writes + output
    /// (anthropic-style split semantics) plus a provider-reported USD cost.
    /// The runtime settlement prices each line and folds the categories
    /// into the recorded input/output totals.
    #[derive(Clone)]
    struct CacheUsageProvider;

    impl faktor_provider::Provider for CacheUsageProvider {
        fn id(&self) -> &str {
            "fake"
        }

        fn capabilities(&self, _model: &str) -> ModelCapabilities {
            ModelCapabilities {
                tools: true,
                ..Default::default()
            }
        }

        fn stream(
            &self,
            _req: faktor_provider::GenericAgentRequest,
        ) -> faktor_provider::ProviderStream {
            let items: Vec<Result<faktor_provider::ProviderChunk, faktor_provider::ProviderError>> = vec![
                Ok(faktor_provider::ProviderChunk::Text {
                    text: "pong".into(),
                }),
                Ok(faktor_provider::ProviderChunk::Usage(
                    faktor_provider::CanonicalUsage {
                        uncached_input_tokens: 0,
                        cache_read_tokens: 7,
                        cache_write_tokens: 2,
                        output_tokens: 3,
                        reasoning_tokens: 0,
                        reported_cost: Some(faktor_provider::ReportedCost {
                            micro_usd: 123,
                            currency: faktor_provider::ReportedCurrency::Usd,
                            source: faktor_provider::ReportedCostSource::ProviderUsage,
                            request_id: Some("req-cache-1".into()),
                        }),
                        request_id: Some("req-cache-1".into()),
                    },
                )),
                Ok(faktor_provider::ProviderChunk::Done),
            ];
            Box::pin(futures_util::stream::iter(items))
        }
    }

    /// The full test deps builder with an explicit provider registry and a
    /// REAL durable cost ledger wired over the same session manager (the
    /// usage E2E test drives reservations through the actual runtime).
    fn test_deps_full(
        root: &std::path::Path,
        providers: Vec<Arc<dyn faktor_provider::Provider>>,
    ) -> ServerDeps {
        let mut registry = faktor_provider::ProviderRegistry::new();
        for p in providers {
            registry.try_register(p).unwrap();
        }
        let session = SessionManager::open(root.join("store"), root.join("cas"), true).unwrap();
        let ledger = faktor_session::DurableBudgetLedger::new(session.clone());
        let budgets: Arc<dyn faktor_session::BudgetAuthority> = ledger.clone();
        let permissions = ChannelPermissionRequester::new(Duration::from_secs(5));
        let agent = AgentRuntime::new(faktor_agent::AgentDeps {
            session: session.clone(),
            providers: Arc::new(registry),
            chunk_sink: None,
            permission_requester: permissions.clone(),
            evidence: Arc::new(faktor_agent::NoEvidence),
            tools: Arc::new(faktor_agent::ToolRegistry::new()),
            cas: None,
            workspaces: faktor_fs::WorkspaceFileService::new(),
            edit: None,
            snapshots: None,
            sandbox: None,
            supervisor: None,
            verification: faktor_agent::VerificationService::disabled(),
            model: "m".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "You are a test server agent.".into(),
            hooks: None,
            instructions_resolver: faktor_instructions::no_roots_resolver(),
            routing: faktor_agent::FixedRoutingPolicy::passthrough(),
            budgets,
            clock: Arc::new(faktor_core::time::SystemClock),
            tool_call_mode: faktor_agent::ToolCallMode::Native,
            tool_deadline_ms: 2000,
            retry_policy: faktor_core::retry::RetryPolicy::default(),
            semantic: faktor_agent::fallback_semantic_registry(),
            context_prior: None,
            efficiency: Default::default(),
        })
        .unwrap();
        let (orchestrator, tasks) = orch_pair(session.clone(), agent.clone());
        ServerDeps {
            budgets: faktor_session::DurableBudgetLedger::new(session.clone()),
            session,
            agent,
            permissions,
            orchestrator,
            tasks,
            auth_token: AuthToken::generate(),
            server_password: ServerPassword::generate(),
            directory: None,
            version: "0.1.0".into(),
            fs: None,
            snapshots: None,
            chunk_rx: None,
            simulate_not_ready: false,
            evidence: None,
            semantic: None,
        }
    }

    #[tokio::test]
    async fn native_usage_real_turn_with_cache_and_reasoning_tokens() {
        // P0-63 end-to-end: a REAL turn through the wire surface against a
        // provider whose usage frame carries only cache reads/writes +
        // reasoning tokens and a provider-reported cost. The runtime
        // settlement persists the folded totals and the reservation rows;
        // /native/session/{id}/usage then reports EXACTLY the stored rows.
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps_full(dir.path(), vec![Arc::new(CacheUsageProvider)]);
        let token = deps.auth_token.clone();
        let manager = deps.session.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let ws = manager.create_workspace("/usage-e2e").unwrap();
        let s = manager
            .create_session(ws, "t-usage-e2e", "fake", "m")
            .unwrap();
        let sid = s.id().to_string();
        seed_typed_task(&s, 1, None, None, "usage-e2e");
        // The session row task identity drives the reserve (task 1 row).
        assert_eq!(s.task_id().unwrap().raw(), 1);

        // Drive the turn through the wire surface exactly like the UI.
        let resp = client
            .post(format!("{base}/session/{sid}/message"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({
                "model": {"providerID": "fake", "modelID": "m"},
                "parts": [{"type": "text", "text": "hi"}],
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "{:?}", resp.text().await);
        let mut body = serde_json::Value::Null;
        for _ in 0..300 {
            let resp = native_get(
                &client,
                &base,
                &token,
                &format!("/session/{sid}/projection"),
            )
            .await;
            assert_eq!(resp.status(), 200);
            body = resp.json().await.unwrap();
            if body["state"]["machine"] == "ready_for_next_turn" {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(body["state"]["machine"], "ready_for_next_turn", "{body}");

        // The stored rows: folded cache/reasoning totals, prefix observation,
        // one settled reservation with the reported cost.
        let store = manager.store();
        let tokens = store.session_usage_tokens(s.id()).unwrap();
        assert_eq!(
            tokens, 12,
            "cache reads 7 + writes 2 + reasoning 3 (in=9,out=3)"
        );
        let prefix = store.provider_call_prefix_rows(s.id()).unwrap();
        assert_eq!(
            prefix.len(),
            1,
            "the completed call recorded its prefix: {prefix:?}"
        );
        let cost = store
            .cost_task_row(s.id(), faktor_core::id::TaskId::new(1))
            .unwrap()
            .unwrap();
        assert_eq!(cost.spent_cost_micro, 123, "provider-reported cost wins");
        let reservations = store
            .cost_reservations_of(s.id(), faktor_core::id::TaskId::new(1), 10)
            .unwrap();
        assert_eq!(reservations.len(), 1);
        assert_eq!(reservations[0].status, "settled");
        // The passthrough test policy consulted NO pricing authority (its
        // decision snapshot is None), so there is no honest locally
        // calculated amount: the provider-reported cost is the ONLY amount
        // and wins both the v18 canonical columns (`settled_cost_micro`,
        // `provider_reported_cost_micro`) and the folded task spend. The
        // pre-B2 "tokens x 1 microUSD local estimate" was abolished — a
        // fabricated number never lands next to a real report.
        assert_eq!(reservations[0].provider_reported_cost_micro, Some(123));
        assert_eq!(reservations[0].provider_reported_micro, Some(123));
        assert_eq!(reservations[0].settled_cost_micro, Some(123));
        assert_eq!(reservations[0].provider_cost_micro, None);
        assert_eq!(
            reservations[0].cost_basis.as_deref(),
            Some(faktor_store::COST_BASIS_PROVIDER_REPORTED)
        );

        // The endpoint reports exactly the stored rows.
        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/session/{sid}/usage"),
        )
        .await;
        assert_eq!(resp.status(), 200);
        let u: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(u["providerCalls"]["tokens"], tokens);
        let obs = u["providerCalls"]["prefixObservations"].as_array().unwrap();
        assert_eq!(obs.len(), prefix.len());
        assert_eq!(obs[0]["promptTokens"], prefix[0].prompt_tokens as u64);
        assert_eq!(
            obs[0]["stability"],
            prefix[0]
                .prefix_stability
                .map(|v| serde_json::json!(v))
                .unwrap_or(serde_json::Value::Null)
        );
        let tasks = u["tasks"].as_array().unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0]["budget"]["spentCostMicro"], cost.spent_cost_micro);
        assert_eq!(tasks[0]["budget"]["maxCostMicro"], serde_json::Value::Null);
        assert_eq!(tasks[0]["budget"]["openReservedMicro"], 0);
        let res = &tasks[0]["reservations"];
        assert_eq!(res["settled"]["count"], 1);
        assert_eq!(
            res["settled"]["spentMicro"], 123,
            "the folded actual (provider-reported) is what was spent"
        );
        assert_eq!(res["settled"]["providerReportedMicro"], 123);
        let routes = res["routeDecisions"].as_array().unwrap();
        assert!(!routes.is_empty(), "the routed call records its decision");
        assert_eq!(routes[0]["providerReportedMicro"], 123);
        assert_eq!(routes[0]["spentMicro"], 123);
        assert!(
            routes[0]["decision"]["provider"] == "fake" || routes[0]["decision"].is_object(),
            "{routes:?}"
        );

        // Usage of a never-used sibling session is empty, never A's rows.
        let ws2 = manager.create_workspace("/usage-e2e-b").unwrap();
        let b = manager
            .create_session(ws2, "t-usage-e2e-b", "fake", "m")
            .unwrap();
        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/session/{}/usage", b.id()),
        )
        .await;
        let ub: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(ub["providerCalls"]["tokens"], 0);
        assert_eq!(ub["tasks"], serde_json::json!([]));
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn native_verification_evidence_scoped_to_session_and_task() {
        // P0-64e: /native/session/{id}/tasks/{task_id}/verification returns
        // the durable VerificationRecord rows (checks/criteria/changed
        // files) of the session's OWN task only. Another session querying
        // the same numeric task id gets a typed 404 or an empty list when a
        // different workspace holds records under that id — evidence never
        // crosses sessions.
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let token = deps.auth_token.clone();
        let manager = deps.session.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let ws_a = manager.create_workspace("/ver-a").unwrap();
        let ws_b = manager.create_workspace("/ver-b").unwrap();
        let a = manager
            .create_session(ws_a, "t-ver-a", "fake", "m")
            .unwrap();
        let b = manager
            .create_session(ws_b, "t-ver-b", "fake", "m")
            .unwrap();
        // C shares B's workspace and (for the cross-workspace guard) adopts
        // the SAME numeric task id A owns — yet C must never see A's
        // evidence rows, which certify A's workspace.
        let c = manager
            .create_session(ws_b, "t-ver-c", "fake", "m")
            .unwrap();
        manager
            .adopt_identity(
                a.id(),
                faktor_core::WorktreeId::new(3),
                faktor_core::id::TaskId::new(7),
            )
            .unwrap();
        manager
            .adopt_identity(
                b.id(),
                faktor_core::WorktreeId::new(4),
                faktor_core::id::TaskId::new(9),
            )
            .unwrap();
        manager
            .adopt_identity(
                c.id(),
                faktor_core::WorktreeId::new(5),
                faktor_core::id::TaskId::new(7),
            )
            .unwrap();
        let store = manager.store();
        let row_a = a.row().unwrap();
        let row_b = b.row().unwrap();
        let rev = faktor_core::id::TaskRevision::new(1);
        let put =
            |row: &faktor_store::VerificationRecordRow| store.verification_record_put(row).unwrap();
        let rec_for = |session_row: &faktor_store::SessionRow, task_id: u64, check: &str| {
            faktor_store::VerificationRecordRow {
                id: faktor_core::id::VerificationRecordId::new(1),
                task_id: faktor_core::id::TaskId::new(task_id),
                revision: rev,
                workspace_id: session_row.workspace_id,
                worktree_id: session_row.worktree_id,
                tree_hash: Some("ab".repeat(32)),
                criteria: vec![faktor_core::state::CriterionVerification {
                    criterion_key: "tests pass".into(),
                    passed: true,
                    evidence: Some("ran".into()),
                }],
                checks: vec![faktor_core::state::CheckExecution {
                    check: check.into(),
                    program: "cargo".into(),
                    args: vec!["test".into()],
                    category: "required".into(),
                    required: true,
                    status: faktor_core::state::VerificationStatus::Passed,
                    started_ms: 1,
                    finished_ms: Some(2),
                    exit: Some(0),
                    summary: Some("ok".into()),
                }],
                changed_files: vec![faktor_core::state::FileStateEvidence {
                    path: "crates/server/src/api.rs".into(),
                    digest_hex: "cd".repeat(32),
                    size: 42,
                }],
                unrelated_changes: vec!["README.md".into()],
                reviewer: None,
                status: faktor_core::state::VerificationStatus::Passed,
                started_ms: 1,
                completed_ms: Some(2),
            }
        };
        let ra = put(&rec_for(&row_a, 7, "cargo test -p faktor-session"));
        let rb = put(&rec_for(&row_b, 9, "cargo test -p faktor-server"));

        // A's task-7 evidence: checks/criteria/changed files all present.
        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/session/{}/tasks/7/verification", a.id()),
        )
        .await;
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["sessionId"], a.id().to_string());
        assert_eq!(body["taskId"], "7");
        let records = body["records"].as_array().unwrap();
        assert_eq!(records.len(), 1, "{body}");
        let rec = &records[0];
        assert_eq!(rec["recordId"], ra.to_string());
        assert_eq!(rec["revision"], "1");
        assert_eq!(rec["status"], "passed");
        assert_eq!(rec["criteria"][0]["criterionKey"], "tests pass");
        assert_eq!(rec["criteria"][0]["passed"], true);
        assert_eq!(rec["checks"][0]["check"], "cargo test -p faktor-session");
        assert_eq!(rec["checks"][0]["program"], "cargo");
        assert_eq!(rec["checks"][0]["args"], serde_json::json!(["test"]));
        assert_eq!(rec["checks"][0]["status"], "passed");
        assert_eq!(rec["checks"][0]["exit"], 0);
        assert_eq!(rec["changedFiles"][0]["path"], "crates/server/src/api.rs");
        assert_eq!(rec["changedFiles"][0]["size"], 42);
        assert_eq!(rec["unrelatedChanges"], serde_json::json!(["README.md"]));
        assert_eq!(rec["completedMs"], 2);

        // B never sees A's task-7 evidence: 7 is not B's task → typed 404.
        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/session/{}/tasks/7/verification", b.id()),
        )
        .await;
        assert_eq!(resp.status(), 404);
        let err: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(err["error"]["code"], "not_found");
        // A cannot read B's task-9 records either.
        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/session/{}/tasks/9/verification", a.id()),
        )
        .await;
        assert_eq!(resp.status(), 404);
        // B's OWN task 9 serves its record.
        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/session/{}/tasks/9/verification", b.id()),
        )
        .await;
        let body: serde_json::Value = resp.json().await.unwrap();
        let records = body["records"].as_array().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0]["recordId"], rb.to_string());
        assert_eq!(
            records[0]["checks"][0]["check"],
            "cargo test -p faktor-server"
        );
        // C (another workspace, SAME numeric task id 7) cannot reach A's
        // record: records certify A's workspace, so C's view is the honest
        // empty list — evidence never crosses sessions or workspaces.
        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/session/{}/tasks/7/verification", c.id()),
        )
        .await;
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(
            body["records"],
            serde_json::json!([]),
            "records of another workspace never surface: {body}"
        );

        // Hostile ids, unknown sessions, auth.
        for path in [
            format!("/native/session/{}/tasks/0/verification", a.id()),
            format!("/native/session/{}/tasks/abc/verification", a.id()),
            "/native/session/abc/tasks/7/verification".to_string(),
            "/native/session/0/tasks/7/verification".to_string(),
        ] {
            let resp = native_get(&client, &base, &token, &path).await;
            assert_eq!(resp.status(), 400, "{path}");
        }
        let resp = native_get(
            &client,
            &base,
            &token,
            "/native/session/999999/tasks/7/verification",
        )
        .await;
        assert_eq!(resp.status(), 404);
        let resp = client
            .get(format!(
                "{base}/native/session/{}/tasks/7/verification",
                a.id()
            ))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn native_tasks_carry_progress_and_durable_budget() {
        // P0-64d: the /native/session/{id}/tasks entry additively carries
        // `progress` (the live bounded progress record, null when nothing
        // ran) and `budget` — the DURABLE budget envelope of the typed task
        // row (token + cost-ledger columns and the open reservation sum).
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let token = deps.auth_token.clone();
        let manager = deps.session.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let ws = manager.create_workspace("/task-budget").unwrap();
        let s = manager
            .create_session(ws, "t-task-budget", "fake", "m")
            .unwrap();
        let sid = s.id().to_string();
        let h = manager.get_session(s.id()).unwrap().unwrap();
        // Durable task ledger (the tasks endpoint's base) + a typed row with
        // a budget + one open and one settled reservation.
        h.put_task_ledger(serde_json::json!({
            "goal": "wire durable budgets",
            "completed_steps": ["mount endpoints"],
            "open_steps": ["ship"],
            "changed_files": ["crates/server/src/api.rs"],
        }))
        .unwrap();
        seed_typed_task(&h, 1, Some(8000), Some(4), "wire durable budgets");
        let store = manager.store();
        store
            .cost_task_cap_set(s.id(), faktor_core::id::TaskId::new(1), Some(250_000))
            .unwrap();
        let now = manager.now_ms();
        store
            .cost_reserve(
                s.id(),
                faktor_core::id::TaskId::new(1),
                manager.next_op_id(),
                60,
                now,
            )
            .unwrap();
        let faktor_store::CostReserveOutcome::Granted(settled_id) = store
            .cost_reserve(
                s.id(),
                faktor_core::id::TaskId::new(1),
                manager.next_op_id(),
                500,
                now,
            )
            .unwrap()
        else {
            panic!("settled reserve granted");
        };
        store
            .cost_settle(settled_id, 100, Some(100), None, None, now + 1)
            .unwrap();

        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/session/{sid}/tasks"),
        )
        .await;
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        let tasks = body.as_array().unwrap();
        assert_eq!(tasks.len(), 1);
        let entry = &tasks[0];
        assert!(
            entry.get("progress").is_some(),
            "progress key present: {entry}"
        );
        assert_eq!(entry["budget"]["maxTokens"], 8000);
        assert_eq!(entry["budget"]["maxTurns"], 4);
        assert_eq!(entry["budget"]["spentTokens"], 0);
        assert_eq!(entry["budget"]["spentTurns"], 0);
        assert_eq!(entry["budget"]["maxCostMicro"], 250_000);
        assert_eq!(entry["budget"]["spentCostMicro"], 100);
        assert_eq!(entry["budget"]["openReservedMicro"], 60);
        let _ = handle.shutdown.send(());
    }

    // ---------------------------------------- max_cost_micro task control E2E
    // (audit 9/H: TaskRunRequest.max_cost_micro flows to the task row cap and a
    // REAL drive whose first model-call reserve exceeds the cap fails with the
    // typed budget refusal — nothing is reserved, nothing is spent.)

    #[tokio::test]
    async fn native_single_item_task_max_cost_micro_caps_the_real_drive() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps_full(dir.path(), vec![Arc::new(CacheUsageProvider)]);
        let token = deps.auth_token.clone();
        let manager = deps.session.clone();
        let tasks = deps.tasks.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let ws = manager.create_workspace("/cap-e2e").unwrap();
        let s = manager
            .create_session(ws, "t-cap-e2e", "fake", "m")
            .unwrap();
        let sid = s.id();
        // A 1-micro cap: every real reserve of the drive (the route estimate is
        // far larger) is refused at admission — the refusal writes NOTHING.
        let req = faktor_orchestrator::runtime::task_executor::TaskRunRequest {
            goal: "spend against the cost cap".into(),
            work_items: vec![faktor_orchestrator::WorkItem::new(
                "a1",
                "spend against the cost cap",
                faktor_orchestrator::WorkKind::Analysis,
            )],
            max_cost_micro: Some(1),
            ..Default::default()
        };
        let receipt = tasks.start_task(sid, req).expect("single-item start");
        assert_eq!(
            receipt.mode,
            faktor_orchestrator::runtime::task_executor::TaskRunMode::InSession
        );
        // The cap was durable before the detached drive ran its first call...
        let h = manager.get_session(sid).unwrap().unwrap();
        let task_id = h.task_id().unwrap();
        let ledger = faktor_session::DurableBudgetLedger::new(manager.clone());
        assert_eq!(
            ledger
                .session_budget_view(sid, task_id)
                .expect("durable budget view")
                .max_cost_micro,
            Some(1),
            "the request cap landed on the task row"
        );
        // ...and the drive ends FailedRecoverable (budget exceeded): the spy
        // provider was never billed — no reservation row, zero spent.
        let mut seen = None;
        for _ in 0..240 {
            let state = h.state().unwrap();
            if state == faktor_core::state::AgentState::FailedRecoverable {
                seen = Some(state);
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        assert_eq!(
            seen,
            Some(faktor_core::state::AgentState::FailedRecoverable),
            "the capped drive must fail recoverable, never silently spend"
        );
        let view = ledger
            .session_budget_view(sid, task_id)
            .expect("durable budget view");
        assert_eq!(view.max_cost_micro, Some(1));
        assert_eq!(view.spent_cost_micro, 0, "a refused reserve spends nothing");
        assert_eq!(view.open_reservations, 0, "a refused reserve writes no row");
        assert_eq!(view.uncertain_reservations, 0);
        // The native agent listing reflects the failed run.
        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/agents?session={sid}"),
        )
        .await;
        assert_eq!(resp.status(), 200);
        let entries: serde_json::Value = resp.json().await.unwrap();
        let e = &entries.as_array().unwrap()[0];
        assert_eq!(e["kind"], "self");
        assert_eq!(e["state"], "Failed");
        let _ = handle.shutdown.send(());
    }

    // =========================================== native task runs (wave-24)
    // The native task-start surface: POST /native/session/{id}/task-runs is
    // the ONE HTTP edge into TaskExecutor::start_task (shadow mutation is
    // the production default; DirectCompat keeps the byte-identical direct
    // behavior), GET list/state read the durable runs, and the task-level
    // cancel is the executor's single cancel authority. Tests below drive
    // REAL shadowed worktrees through the HTTP layer, attack the strict
    // DTO, freeze the parity of DirectCompat, and source-scan the crate for
    // any second task-start edge.

    /// Session-scoped workspace root provider mirroring the daemon graph:
    /// the live shadow of a shadowed workspace re-points instruction
    /// loading at the shadow root.
    struct NativeRealRoots(Arc<SessionManager>);
    impl faktor_instructions::WorkspaceRootProvider for NativeRealRoots {
        fn workspace_root(&self, workspace_id: u64) -> Option<std::path::PathBuf> {
            use faktor_core::id::WorkspaceId;
            if workspace_id == 0 {
                return None;
            }
            let ws = WorkspaceId::new(workspace_id);
            match self.0.live_workspace_shadow_root(ws) {
                Ok(Some(root)) => Some(root),
                Ok(None) | Err(_) => self.0.workspace_root(ws).ok().flatten(),
            }
        }
    }

    /// A REAL write_file: writes through the session's resolved workspace
    /// root (the shadow root while a shadowed drive is live). `park` parks
    /// the FIRST invocation after the write landed (the deterministic
    /// mid-drive window of the shadowed HTTP tests).
    fn native_real_write_tool(
        park: Option<(
            Arc<tokio::sync::Notify>,
            Arc<std::sync::atomic::AtomicUsize>,
        )>,
    ) -> faktor_agent::Tool {
        use faktor_agent::tool::RecoveryHint;
        use faktor_agent::{ToolOutcome, ToolRunCtx};
        use faktor_core::resource::ResourceClass;
        let gate = park.as_ref().map(|(g, _)| g.clone());
        let fired = park.as_ref().map(|(_, f)| f.clone());
        faktor_agent::Tool {
            name: "write_file".into(),
            description: "writes a real file".into(),
            input_schema: serde_json::json!({"type": "object"}),
            resource_class: ResourceClass::DiskWrite,
            capability: None,
            recovery_hint: RecoveryHint::WorkspaceWrite,
            path_args: vec!["path".into()],
            execute: Arc::new(move |ctx: ToolRunCtx, args| {
                let gate = gate.clone();
                let fired = fired.clone();
                Box::pin(async move {
                    let Some(ws) = &ctx.workspace else {
                        return Err(faktor_core::error::Error::internal("no workspace wired"));
                    };
                    let path = args.get("path").and_then(|p| p.as_str()).unwrap_or("");
                    let content = args
                        .get("content")
                        .and_then(|c| c.as_str())
                        .unwrap_or_default();
                    ws.write_atomic(std::path::Path::new(path), content.as_bytes())
                        .map_err(|e| {
                            faktor_core::error::Error::internal(format!("write {path}: {e}"))
                        })?;
                    if let (Some(g), Some(f)) = (gate, fired) {
                        if f.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                            g.notified().await;
                        }
                    }
                    Ok(ToolOutcome {
                        text: format!("wrote {path}"),
                        exit_code: Some(0),
                        ..Default::default()
                    })
                })
            }),
        }
    }

    /// A CPU tool whose FIRST invocation parks the drive mid-flight (the
    /// deterministic cancel window).
    fn native_parking_tool(
        gate: Arc<tokio::sync::Notify>,
        fired: Arc<std::sync::atomic::AtomicUsize>,
    ) -> faktor_agent::Tool {
        use faktor_agent::tool::RecoveryHint;
        use faktor_agent::ToolOutcome;
        use faktor_core::resource::ResourceClass;
        faktor_agent::Tool {
            name: "pause".into(),
            description: "parks once".into(),
            input_schema: serde_json::json!({"type": "object"}),
            resource_class: ResourceClass::Cpu,
            capability: None,
            recovery_hint: RecoveryHint::Idempotent,
            path_args: vec![],
            execute: Arc::new(move |_ctx, _args| {
                let gate = gate.clone();
                let fired = fired.clone();
                Box::pin(async move {
                    if fired.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                        gate.notified().await;
                    }
                    Ok(ToolOutcome {
                        text: "parked".into(),
                        exit_code: Some(0),
                        ..Default::default()
                    })
                })
            }),
        }
    }

    /// A shadowed (or direct) native rig: the real agent with a REAL write
    /// tool + a real workspace, the executor carrying the ShadowRoots
    /// service per `service`, and the executor's default mutation mode per
    /// `mode`. `scripts` serve the drive's model calls.
    struct NativeTaskRig {
        deps: ServerDeps,
        manager: Arc<SessionManager>,
        parent: SessionId,
        owner_root: std::path::PathBuf,
        gate: Arc<tokio::sync::Notify>,
        fired: Arc<std::sync::atomic::AtomicUsize>,
    }

    fn native_task_rig(
        root: &std::path::Path,
        scripts: Vec<Vec<faktor_provider::ScriptedResponse>>,
        parked_write: bool,
        service: bool,
        mode: faktor_orchestrator::runtime::task_executor::MutationMode,
    ) -> NativeTaskRig {
        use faktor_core::model::ModelCapabilities;
        let paced = PacedScriptedProvider::new(
            ModelCapabilities {
                tools: true,
                ..Default::default()
            },
            scripts,
            5,
        );
        native_task_rig_with_provider(root, paced, parked_write, service, mode)
    }

    /// The same rig with an EXPLICIT provider: the deterministic
    /// failure-injection seam (a stub whose stream returns a typed error on
    /// every platform, with no workspace/path/host-speed dependence).
    fn native_task_rig_with_provider(
        root: &std::path::Path,
        provider: Arc<dyn faktor_provider::Provider>,
        parked_write: bool,
        service: bool,
        mode: faktor_orchestrator::runtime::task_executor::MutationMode,
    ) -> NativeTaskRig {
        use faktor_core::id::{TaskId, WorktreeId};
        let manager = SessionManager::open(root.join("store"), root.join("cas"), true).unwrap();
        let mut registry = faktor_provider::ProviderRegistry::new();
        registry.try_register(provider).unwrap();
        let permissions = ChannelPermissionRequester::new(Duration::from_secs(5));
        let gate = Arc::new(tokio::sync::Notify::new());
        let fired = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut tools = faktor_agent::ToolRegistry::new();
        if parked_write {
            tools.register(native_real_write_tool(Some((gate.clone(), fired.clone()))));
        } else {
            tools.register(native_real_write_tool(None));
        }
        tools.register(native_parking_tool(gate.clone(), fired.clone()));
        let resolver = Arc::new(faktor_instructions::InstructionResolver::new(
            Arc::new(NativeRealRoots(manager.clone())),
            faktor_instructions::DEFAULT_RESOLVER_CACHE_ENTRIES,
        ));
        let agent = AgentRuntime::new(faktor_agent::AgentDeps {
            session: manager.clone(),
            providers: Arc::new(registry),
            chunk_sink: None,
            permission_requester: Arc::new(AllowAll),
            evidence: Arc::new(faktor_agent::NoEvidence),
            tools: Arc::new(tools),
            cas: None,
            workspaces: faktor_fs::WorkspaceFileService::new(),
            edit: None,
            snapshots: None,
            sandbox: None,
            supervisor: None,
            verification: faktor_agent::VerificationService::disabled(),
            hooks: None,
            instructions_resolver: resolver,
            routing: faktor_agent::FixedRoutingPolicy::passthrough(),
            budgets: Arc::new(faktor_session::NoopBudget),
            model: "m".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "You are a test server agent.".into(),
            clock: Arc::new(faktor_core::time::SystemClock),
            tool_call_mode: faktor_agent::ToolCallMode::Native,
            tool_deadline_ms: 120_000,
            retry_policy: faktor_core::retry::RetryPolicy::default(),
            semantic: faktor_agent::fallback_semantic_registry(),
            context_prior: None,
            efficiency: Default::default(),
        })
        .unwrap();
        let owner_root = root.join("owner");
        std::fs::create_dir_all(&owner_root).unwrap();
        let ws = manager
            .create_workspace(owner_root.to_str().unwrap())
            .unwrap();
        let wt = WorktreeId::new(
            manager
                .put_worktree(ws, owner_root.to_str().unwrap(), "main")
                .unwrap() as u64,
        );
        let parent = manager
            .create_session(ws, "native-task-rig", "fake", "m")
            .unwrap()
            .id();
        manager.adopt_identity(parent, wt, TaskId::new(1)).unwrap();
        let orchestrator =
            faktor_orchestrator::runtime::OrchestratorRuntime::new(manager.clone(), agent.clone());
        let tasks = if service {
            let shadows = faktor_orchestrator::runtime::shadow::ShadowRoots::new(
                manager.clone(),
                root.join("shadows"),
            );
            faktor_orchestrator::runtime::task_executor::TaskExecutor::new_with_mode(
                &orchestrator,
                manager.clone(),
                agent.clone(),
                Some(shadows),
                mode,
            )
        } else {
            faktor_orchestrator::runtime::task_executor::TaskExecutor::new(
                &orchestrator,
                manager.clone(),
                agent.clone(),
                None,
            )
        };
        let deps = ServerDeps {
            budgets: faktor_session::DurableBudgetLedger::new(manager.clone()),
            session: manager.clone(),
            agent,
            permissions,
            orchestrator,
            tasks,
            auth_token: AuthToken::generate(),
            server_password: ServerPassword::generate(),
            directory: None,
            version: "0.1.0".into(),
            fs: None,
            snapshots: None,
            chunk_rx: None,
            simulate_not_ready: false,
            evidence: None,
            semantic: None,
        };
        NativeTaskRig {
            deps,
            manager,
            parent,
            owner_root,
            gate,
            fired,
        }
    }

    fn seed_native_owner(root: &std::path::Path) {
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"x\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .unwrap();
        std::fs::write(root.join("src/lib.rs"), NATIVE_OWNER_LIB_RS).unwrap();
    }

    /// The owner checkout's ORIGINAL content (distinct from the drive's
    /// write so integration is byte-observable).
    const NATIVE_OWNER_LIB_RS: &str = "pub fn value() -> u64 {\n    let base_amount: u64 = 40;\n    let increment: u64 = 1;\n    base_amount.saturating_add(increment)\n}\n";
    const NATIVE_IMPL_LIB_RS: &str = "pub fn value() -> u64 {\n    let base_amount: u64 = 10;\n    let increment: u64 = 32;\n    base_amount.saturating_add(increment)\n}\n";

    async fn native_wait_session_state(
        manager: &Arc<SessionManager>,
        sid: SessionId,
        want: faktor_core::state::AgentState,
    ) {
        for _ in 0..1500 {
            if manager.get_session(sid).unwrap().unwrap().state().unwrap() == want {
                return;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        panic!("session {sid} never reached {want:?}");
    }

    /// The human-verifier seam over a manager (the ONLY producer of
    /// VerifiedComplete): drive the durable task row to Verifying, land a
    /// passing record (covering every acceptance criterion of the row) and
    /// complete.
    fn certify_native_task(manager: &Arc<SessionManager>, sid: SessionId) {
        use faktor_core::state::{TaskState, TaskTransition, VerificationStatus};
        let h = manager.get_session(sid).unwrap().unwrap();
        let task_id = h.task_id().unwrap();
        let criteria = h
            .get_task(task_id)
            .unwrap()
            .unwrap()
            .acceptance_criteria
            .into_iter()
            .map(|criterion_key| faktor_core::state::CriterionVerification {
                criterion_key,
                passed: true,
                evidence: None,
            })
            .collect::<Vec<_>>();
        for _ in 0..8 {
            let task = h.get_task(task_id).unwrap().unwrap();
            let target = match task.state {
                TaskState::Pending => TaskTransition::StartRunning,
                TaskState::Planning => TaskTransition::PlanComplete,
                TaskState::Running => TaskTransition::RequestVerification,
                TaskState::Waiting => TaskTransition::ResumeFromWaiting,
                TaskState::Blocked => TaskTransition::Unblock,
                TaskState::NeedsVerification => TaskTransition::StartVerification,
                TaskState::Verifying => break,
                s => panic!("cannot certify a task at {s:?}"),
            };
            let rev = h.task_revision(task_id).unwrap();
            h.transition_task(task_id, rev, target, None).unwrap();
        }
        let task = h.get_task(task_id).unwrap().unwrap();
        assert_eq!(task.state, TaskState::Verifying);
        let record = h
            .create_verification_record(
                task_id,
                None,
                criteria,
                vec![],
                vec![],
                vec![],
                None,
                VerificationStatus::Passed,
                h.now_ms(),
            )
            .unwrap();
        let rev = h.task_revision(task_id).unwrap();
        let task = h.get_task(task_id).unwrap().unwrap();
        h.complete_verified_task(task_id, rev, record)
            .unwrap_or_else(|e| {
                panic!(
                    "complete_verified_task at {rev:?} state {:?}: {e}",
                    task.state
                )
            });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn native_task_run_start_shadowed_drive_keeps_checkout_then_integrates() {
        // The wave-24 E2E through the HTTP layer: POST starts a mutating
        // single-item task (production default: shadowed) that really
        // drives; the run's write lands in the SHADOW while the owner
        // checkout stays byte-untouched MID-drive; the run listing/state
        // endpoints reflect it; a verified completion integrates the owner
        // checkout through the executor's own settle paths (never a manual
        // finalize call from the test).
        let dir = tempfile::tempdir().unwrap();
        let rig = native_task_rig(
            dir.path(),
            vec![
                vec![
                    faktor_provider::ScriptedResponse::ToolCall {
                        id: "c1".into(),
                        name: "write_file".into(),
                        input: serde_json::json!({
                            "path": "src/lib.rs",
                            "content": NATIVE_IMPL_LIB_RS,
                        }),
                    },
                    faktor_provider::ScriptedResponse::Text("done".into()),
                    faktor_provider::ScriptedResponse::End,
                ],
                vec![faktor_provider::ScriptedResponse::End],
            ],
            true,
            true,
            faktor_orchestrator::runtime::task_executor::MutationMode::Shadow,
        );
        seed_native_owner(&rig.owner_root);
        let NativeTaskRig {
            deps,
            manager,
            parent: sid,
            owner_root,
            gate,
            fired,
        } = rig;
        let token = deps.auth_token.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        // POST the strict native start (goal only: absent work_items = one
        // MUTATING main item; absent mutation_mode = the daemon default
        // Shadow).
        let resp = client
            .post(format!("{base}/native/session/{sid}/task-runs"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"goal": "implement the change"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let start: serde_json::Value = resp.json().await.unwrap();
        let run_id = start["run_id"].as_str().unwrap().to_string();
        assert!(run_id.starts_with("tx-"), "{start}");
        assert_eq!(start["task_id"], 1);
        let row = manager
            .shadow_row(sid)
            .unwrap()
            .expect("shadow row at begin");
        let shadow_dir = std::path::PathBuf::from(&row.root);
        assert_eq!(row.state, faktor_session::ShadowRowState::Active);

        // Mid-drive: the first write landed inside the SHADOW and parked
        // the drive; the user checkout is byte-untouched.
        for _ in 0..3000 {
            if fired.load(std::sync::atomic::Ordering::SeqCst) >= 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert_eq!(fired.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(
            std::fs::read(shadow_dir.join("src/lib.rs")).unwrap(),
            NATIVE_IMPL_LIB_RS.as_bytes(),
            "the write landed in the SHADOW"
        );
        assert_eq!(
            std::fs::read(owner_root.join("src/lib.rs")).unwrap(),
            NATIVE_OWNER_LIB_RS.as_bytes(),
            "the owner checkout is byte-untouched MID-drive"
        );
        assert_eq!(
            manager.active_root(sid).unwrap(),
            Some(shadow_dir.clone()),
            "the live shadow re-points the session"
        );
        // The task-runs list reflects the live run with its state.
        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/session/{sid}/task-runs"),
        )
        .await;
        assert_eq!(resp.status(), 200);
        let list: serde_json::Value = resp.json().await.unwrap();
        let entry = list
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["run_id"] == run_id)
            .expect("the live run is listed");
        assert_eq!(entry["task_id"], 1);
        assert_eq!(entry["mode"], "in_session");
        assert_eq!(entry["goal"], "implement the change");
        assert_eq!(entry["item_ids"], serde_json::json!(["main"]));
        // The per-run state read matches the list entry.
        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/session/{sid}/task-runs/{run_id}"),
        )
        .await;
        assert_eq!(resp.status(), 200);
        let one: serde_json::Value = resp.json().await.unwrap();
        for key in ["task_id", "run_id", "mode", "goal", "item_ids"] {
            assert_eq!(one.get(key), entry.get(key), "{key}");
        }

        // Release the drive; the verified completion (certified through the
        // durable machine) integrates through the executor's own settle
        // paths — no finalize endpoint, no manual call.
        gate.notify_waiters();
        native_wait_session_state(
            &manager,
            sid,
            faktor_core::state::AgentState::ReadyForNextTurn,
        )
        .await;
        tokio::time::sleep(Duration::from_millis(400)).await;
        certify_native_task(&manager, sid);
        for _ in 0..600 {
            let ok = std::fs::read(owner_root.join("src/lib.rs"))
                .map(|b| b == NATIVE_IMPL_LIB_RS.as_bytes())
                .unwrap_or(false)
                && !shadow_dir.exists();
            if ok {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        assert_eq!(
            std::fs::read(owner_root.join("src/lib.rs")).unwrap(),
            NATIVE_IMPL_LIB_RS.as_bytes(),
            "VerifiedComplete integration lands the changed file"
        );
        assert!(!shadow_dir.exists(), "clean integration removes the shadow");
        assert_eq!(
            manager.shadow_row(sid).unwrap().unwrap().state,
            faktor_session::ShadowRowState::Integrated
        );
        // The run's terminal state reads Done on the task-runs surface.
        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/session/{sid}/task-runs/{run_id}"),
        )
        .await;
        let done: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(done["state"], "Done", "{done}");
        let _ = handle.shutdown.send(());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn native_shadow_run_via_extension_client_session_survives_worktree_id_shift() {
        // Shadow 409 root cause, end to end through the EXACT extension
        // client path: `POST /session/create` (workspace root) followed by
        // `POST /native/session/{id}/task-runs` with the shadow default.
        // The regression: a session created over HTTP carries the
        // standalone default worktree 1. When its workspace ALREADY holds
        // an owner worktree row with another id (any worktree row from an
        // earlier project on the same daemon), the executor's adoption
        // early-returned on "workspace has worktrees" and the shadowed run
        // refused with a typed 409 "no registered worktree row". The daemon
        // must register the session's own workspace/worktree at creation
        // and self-heal older sessions at task-run start.
        let dir = tempfile::tempdir().unwrap();
        let rig = native_task_rig(
            dir.path(),
            vec![
                vec![
                    faktor_provider::ScriptedResponse::ToolCall {
                        id: "c1".into(),
                        name: "write_file".into(),
                        input: serde_json::json!({
                            "path": "src/lib.rs",
                            "content": NATIVE_IMPL_LIB_RS,
                        }),
                    },
                    faktor_provider::ScriptedResponse::Text("done".into()),
                    faktor_provider::ScriptedResponse::End,
                ],
                vec![faktor_provider::ScriptedResponse::End],
            ],
            false,
            true,
            faktor_orchestrator::runtime::task_executor::MutationMode::Shadow,
        );
        seed_native_owner(&rig.owner_root);
        let NativeTaskRig {
            deps,
            manager,
            owner_root,
            ..
        } = rig;
        let token = deps.auth_token.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        // Force the regression's precondition: the owner workspace's
        // worktree row is NOT id 1 (a decoy workspace registered first,
        // then the owner row recreated), so the standalone default 1 names
        // no row of this workspace. The rig's own session is irrelevant;
        // the extension creates a FRESH session via the wire.
        let ws = manager
            .create_workspace(owner_root.to_str().unwrap())
            .unwrap();
        manager
            .remove_worktree(owner_root.to_str().unwrap())
            .unwrap();
        let decoy = manager
            .create_workspace(dir.path().join("decoy").to_str().unwrap())
            .unwrap();
        let decoy_wt = manager
            .put_worktree(decoy, dir.path().join("decoy").to_str().unwrap(), "main")
            .unwrap();
        assert_eq!(
            decoy_wt, 1,
            "the standalone default id is taken by the decoy"
        );
        let owner_wt = manager
            .put_worktree(ws, owner_root.to_str().unwrap(), "main")
            .unwrap();
        assert_ne!(
            owner_wt, 1,
            "the owner row id shifted away from the default"
        );

        // The extension's session creation: POST /session/create with the
        // window's workspace root. Creation-time registration must adopt
        // the session onto the workspace's real owner worktree.
        let resp = client
            .post(format!("{base}/session/create"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({
                "provider": "fake",
                "model": "m",
                "workspace": owner_root.to_str().unwrap(),
                "title": "extension session",
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid =
            SessionId::try_from(created["id"].as_str().unwrap().parse::<u64>().unwrap()).unwrap();
        assert_eq!(
            manager
                .get_session(sid)
                .unwrap()
                .unwrap()
                .row()
                .unwrap()
                .worktree_id,
            faktor_core::id::WorktreeId::new(owner_wt as u64),
            "the created session must be registered on its workspace owner row"
        );

        // Adversarial self-heal probe: an OLDER session (or one created
        // before creation-time registration existed) still holds the
        // standalone default. The task-run start must re-register it
        // instead of refusing the shadowed run with a 409.
        manager
            .adopt_identity(
                sid,
                faktor_core::id::WorktreeId::new(1),
                faktor_core::id::TaskId::new(1),
            )
            .unwrap();

        // The shadowed mutating run via the extension client path: goal
        // only, mutation_mode omitted = daemon default (Shadow).
        let resp = client
            .post(format!("{base}/native/session/{sid}/task-runs"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"goal": "implement the change"}))
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            200,
            "an unregistered session must be adopted at task-run start, never 409ed: {:?}",
            resp.text().await
        );
        native_wait_session_state(
            &manager,
            sid,
            faktor_core::state::AgentState::ReadyForNextTurn,
        )
        .await;
        // The run really worked in a daemon-owned shadow; the owner
        // checkout stayed byte-untouched.
        let shadow = manager
            .shadow_row(sid)
            .unwrap()
            .expect("an ordinary native mutating prompt must begin a shadow");
        assert_eq!(shadow.state, faktor_session::ShadowRowState::Active);
        assert_eq!(
            std::fs::read(std::path::Path::new(&shadow.root).join("src/lib.rs")).unwrap(),
            NATIVE_IMPL_LIB_RS.as_bytes(),
            "the edit landed in the shadow"
        );
        assert_eq!(
            std::fs::read(owner_root.join("src/lib.rs")).unwrap(),
            NATIVE_OWNER_LIB_RS.as_bytes(),
            "the owner checkout is byte-untouched"
        );
        assert_eq!(
            manager
                .get_session(sid)
                .unwrap()
                .unwrap()
                .row()
                .unwrap()
                .worktree_id,
            faktor_core::id::WorktreeId::new(owner_wt as u64),
            "the self-heal re-adopted the workspace owner row"
        );
        let _ = handle.shutdown.send(());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn native_mutating_multi_agent_task_isolated_then_explicit_integration_and_restart() {
        // The audit E2E: POST one 3-stage analysis -> implementation ->
        // review plan. Ownership is EXPLICIT per item (read-only items hold
        // NoWrites, the mutating item owns an IsolatedWorktree); the DAEMON
        // allocates the candidate root itself (the DTO carries no path).
        // Asserts: accepted, durable assignments, real child sessions,
        // implementation inside the daemon-owned candidate root, the owner
        // checkout untouched, an EXPLICIT integration that makes the
        // candidate visible to review, and a restart that preserves child
        // ids + run.
        use faktor_orchestrator::runtime::OrchestratorRuntime;

        let dir = tempfile::tempdir().unwrap();
        let rig = native_task_rig(
            dir.path(),
            vec![
                // child-0 analysis (read-only).
                vec![
                    faktor_provider::ScriptedResponse::Text("analysis done".into()),
                    faktor_provider::ScriptedResponse::End,
                ],
                // child-1 implementation: a REAL write inside its isolated
                // worktree. (The candidate starts empty; the write creates
                // `candidate.txt` at its root.)
                vec![
                    faktor_provider::ScriptedResponse::ToolCall {
                        id: "w1".into(),
                        name: "write_file".into(),
                        input: serde_json::json!({
                            "path": "candidate.txt",
                            "content": NATIVE_IMPL_LIB_RS,
                        }),
                    },
                    faktor_provider::ScriptedResponse::Text("implemented".into()),
                    faktor_provider::ScriptedResponse::End,
                ],
                // child-2 review (read-only).
                vec![
                    faktor_provider::ScriptedResponse::Text("review ok".into()),
                    faktor_provider::ScriptedResponse::End,
                ],
            ],
            false,
            false,
            faktor_orchestrator::runtime::task_executor::MutationMode::Shadow,
        );
        seed_native_owner(&rig.owner_root);
        let NativeTaskRig {
            deps,
            manager,
            parent: sid,
            owner_root,
            ..
        } = rig;
        let orchestrator = deps.orchestrator.clone();
        let tasks = deps.tasks.clone();
        let token = deps.auth_token.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        let resp = client
            .post(format!("{base}/native/session/{sid}/task-runs"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({
                "goal": "3-stage change",
                "work_items": [
                    {"id": "analyze", "kind": "Analysis", "ownership": "no_writes"},
                    {
                        "id": "implement",
                        "kind": "Implementation",
                        "depends_on": ["analyze"],
                        "ownership": "isolated_worktree",
                    },
                    {
                        "id": "review",
                        "kind": "Review",
                        "depends_on": ["implement"],
                        "ownership": "no_writes",
                    },
                ],
            }))
            .send()
            .await
            .unwrap();
        let status = resp.status();
        let start: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(status, 200, "{start}");
        assert_eq!(start["task_id"], 1);
        let run_id = start["run_id"].as_str().unwrap().to_string();
        // The list surface reports the orchestrated mode (the POST receipt
        // predates the detached plan row).
        let mut modes = Vec::new();
        for _ in 0..200 {
            let resp = native_get(
                &client,
                &base,
                &token,
                &format!("/native/session/{sid}/task-runs"),
            )
            .await;
            assert_eq!(resp.status(), 200);
            let list: serde_json::Value = resp.json().await.unwrap();
            modes = list
                .as_array()
                .unwrap()
                .iter()
                .filter(|e| e["run_id"] == run_id.as_str())
                .map(|e| e["mode"].clone())
                .collect();
            if !modes.is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert_eq!(modes, vec![serde_json::json!("orchestrated")], "{start}");

        // Durable assignments exist BEFORE/while the children drive.
        let assignments =
            OrchestratorRuntime::assignment_rows(manager.clone(), sid, &run_id).unwrap();
        assert_eq!(assignments.len(), 3, "one durable assignment per item");
        let a_of = |id: &str| assignments.iter().find(|a| a.item_id == id).unwrap();
        assert_eq!(
            a_of("analyze").ownership,
            faktor_core::state::OwnershipSpec::NoWrites
        );
        assert_eq!(
            a_of("implement").ownership,
            faktor_core::state::OwnershipSpec::IsolatedWorktree
        );
        assert_eq!(
            a_of("review").ownership,
            faktor_core::state::OwnershipSpec::NoWrites
        );

        // Wait for the whole run to reach its terminal item states.
        for _ in 0..600 {
            let rows = OrchestratorRuntime::registry_rows(manager.clone(), sid, &run_id).unwrap();
            if rows.len() == 3 && rows.iter().all(|c| c.state.is_terminal()) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        let rows = OrchestratorRuntime::registry_rows(manager.clone(), sid, &run_id).unwrap();
        assert_eq!(rows.len(), 3, "three real child rows");
        let row_of = |id: &str| rows.iter().find(|r| r.item_id == id).unwrap();
        for id in ["analyze", "implement", "review"] {
            assert_ne!(row_of(id).session_id, 0, "child {id} has a real session");
            assert_ne!(row_of(id).operation_id, 0, "child {id} was really driven");
            assert_eq!(row_of(id).state, faktor_orchestrator::ChildState::Done);
        }
        // Child sessions are visible on the native agents surface.
        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/agents?session={sid}"),
        )
        .await;
        assert_eq!(resp.status(), 200);
        let entries: serde_json::Value = resp.json().await.unwrap();
        let children: Vec<&serde_json::Value> = entries
            .as_array()
            .unwrap()
            .iter()
            .filter(|e| e["kind"] == "child")
            .collect();
        assert_eq!(children.len(), 3, "children visible: {entries}");

        // The implementation ran in the DAEMON-allocated candidate root
        // (never a client path): its workspace lives under
        // `<run_roots.root()>/s<sid>/<run_id>`.
        let impl_row = row_of("implement");
        assert_eq!(
            impl_row.ownership,
            faktor_session::child::ChildOwnership::IsolatedWorktree
        );
        let impl_root = manager
            .workspace_root(WorkspaceId::new(impl_row.workspace_id))
            .unwrap()
            .expect("implementation workspace root");
        let expected = tasks
            .run_roots()
            .root()
            .join(format!("s{}", sid.raw()))
            .join(&run_id);
        assert!(
            std::path::Path::new(&impl_root).starts_with(&expected),
            "the implementation lives under the daemon-allocated candidate root \
             ({impl_root:?} vs {expected:?})"
        );
        assert!(
            std::path::Path::new(&impl_root).ends_with("child-1"),
            "the implementation child owns its own isolated child root"
        );
        assert_eq!(
            std::fs::read(std::path::Path::new(&impl_root).join("candidate.txt")).unwrap(),
            NATIVE_IMPL_LIB_RS.as_bytes(),
            "the implementation wrote inside its candidate"
        );
        // Owner checkout unchanged until an EXPLICIT integration.
        assert_eq!(
            std::fs::read(owner_root.join("src/lib.rs")).unwrap(),
            NATIVE_OWNER_LIB_RS.as_bytes(),
            "the owner checkout is byte-untouched before integration"
        );
        assert!(
            !owner_root.join("candidate.txt").exists(),
            "nothing of the candidate leaked into the owner checkout"
        );

        // Explicit integration of the candidate into the owner checkout.
        let cs = orchestrator
            .stage_child_changes(&impl_row.child_id)
            .unwrap();
        assert!(
            cs.files.iter().any(|f| f.path.ends_with("candidate.txt")),
            "the staged change set holds the candidate write: {:?}",
            cs.files
        );
        let approved: Vec<std::path::PathBuf> = cs
            .files
            .iter()
            .filter(|f| f.child_hash.is_some())
            .map(|f| f.path.clone())
            .collect();
        let outcome = orchestrator
            .approve_and_merge(&impl_row.child_id, &cs.id(), &approved, &[])
            .unwrap();
        assert!(
            outcome.merged.iter().any(|p| p.ends_with("candidate.txt")),
            "the candidate merged explicitly: {outcome:?}"
        );
        assert_eq!(
            std::fs::read(owner_root.join("candidate.txt")).unwrap(),
            NATIVE_IMPL_LIB_RS.as_bytes(),
            "integration lands the candidate in the owner checkout"
        );
        // Review now SEES the candidate: a reviewer tree copied from the
        // current parent state contains the integrated bytes.
        let reviewer = orchestrator.spawn_reviewer(&impl_row.child_id).unwrap();
        let reviewer_root = manager
            .workspace_root(WorkspaceId::new(reviewer.workspace_id))
            .unwrap()
            .expect("reviewer workspace root");
        assert_eq!(
            std::fs::read(std::path::Path::new(&reviewer_root).join("candidate.txt")).unwrap(),
            NATIVE_IMPL_LIB_RS.as_bytes(),
            "review sees the integrated candidate"
        );
        let _ = handle.shutdown.send(());
        drop(client);
        drop(orchestrator);
        drop(tasks);

        // Restart on the same data dir: the run + child ids survive.
        drop(manager);
        let reopened =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let assignments2 =
            OrchestratorRuntime::assignment_rows(reopened.clone(), sid, &run_id).unwrap();
        assert_eq!(assignments2, assignments, "assignments survive a restart");
        let rows2 = OrchestratorRuntime::registry_rows(reopened, sid, &run_id).unwrap();
        assert_eq!(
            rows2.len(),
            4,
            "the three plan children + the reviewer survive"
        );
        let ids: Vec<&str> = rows2.iter().map(|r| r.child_id.as_str()).collect();
        for id in ["child-0", "child-1", "child-2"] {
            assert!(ids.contains(&id), "child id {id} survives: {ids:?}");
        }
        for r in &rows2 {
            assert_ne!(r.session_id, 0, "child session ids survive a restart");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sdk_compat_prompt_translates_to_the_one_executor_with_direct_mutation() {
        // Ordinary chat through the SDK compatibility surface
        // (`POST /session/{id}/prompt`) goes through
        // PromptExecutionService -> TaskExecutor (the durable in-session run
        // appears in the native task-run listing) and the write lands in the
        // OWNER checkout: compatibility surfaces translate the mutation
        // policy as direct (COMPAT_MUTATION_MODE) because they must never
        // wait on the executor's synchronous O(workspace) shadow begin.
        let dir = tempfile::tempdir().unwrap();
        let rig = native_task_rig(
            dir.path(),
            vec![vec![
                faktor_provider::ScriptedResponse::ToolCall {
                    id: "c1".into(),
                    name: "write_file".into(),
                    input: serde_json::json!({
                        "path": "src/lib.rs",
                        "content": NATIVE_IMPL_LIB_RS,
                    }),
                },
                faktor_provider::ScriptedResponse::Text("done".into()),
                faktor_provider::ScriptedResponse::End,
            ]],
            false,
            true,
            faktor_orchestrator::runtime::task_executor::MutationMode::Shadow,
        );
        seed_native_owner(&rig.owner_root);
        let NativeTaskRig {
            deps,
            manager,
            parent: sid,
            owner_root,
            ..
        } = rig;
        let token = deps.auth_token.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        let resp = client
            .post(format!("{base}/session/prompt"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({
                "session_id": sid.to_string(),
                "prompt": "implement the change",
                "files": []
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["accepted"], true, "{body}");
        assert_ne!(body["op_id"], "turn", "the real session op id rides");
        native_wait_session_state(
            &manager,
            sid,
            faktor_core::state::AgentState::ReadyForNextTurn,
        )
        .await;
        // Direct mutation: the owner checkout holds the edit, no shadow was
        // ever begun (the synchronous O(workspace) copy never rides the
        // legacy request).
        assert!(
            manager.shadow_row(sid).unwrap().is_none(),
            "compat prompts select the direct mutation policy"
        );
        assert_eq!(
            std::fs::read(owner_root.join("src/lib.rs")).unwrap(),
            NATIVE_IMPL_LIB_RS.as_bytes(),
            "the compat drive wrote the owner checkout directly"
        );
        // ONE execution path: the durable in-session run linkage row exists
        // and is listed by the native surface.
        let resp = client
            .get(format!("{base}/native/session/{sid}/task-runs"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let runs: serde_json::Value = resp.json().await.unwrap();
        let run = runs
            .as_array()
            .and_then(|runs| {
                runs.iter().find(|r| {
                    r["mode"] == "in_session"
                        && r["run_id"].as_str().is_some_and(|id| id.starts_with("tx-"))
                })
            })
            .unwrap_or_else(|| {
                panic!("the compat prompt must leave a durable executor run: {runs}")
            });
        assert_eq!(run["goal"], "implement the change", "{run}");
        let _ = handle.shutdown.send(());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn native_ordinary_prompt_uses_the_shadow_executor_and_keeps_the_owner_untouched() {
        // The NATIVE ordinary prompt (no explicit work items) keeps the
        // daemon default shadow mutation: its write lands in the daemon
        // shadow and the owner checkout stays byte-untouched until a
        // verified integration. (The moved coverage of the pre-regression
        // `sdk_compat_...` test: the native surface is where shadowing is
        // the promise; compatibility surfaces are direct.)
        let dir = tempfile::tempdir().unwrap();
        let rig = native_task_rig(
            dir.path(),
            vec![vec![
                faktor_provider::ScriptedResponse::ToolCall {
                    id: "c1".into(),
                    name: "write_file".into(),
                    input: serde_json::json!({
                        "path": "src/lib.rs",
                        "content": NATIVE_IMPL_LIB_RS,
                    }),
                },
                faktor_provider::ScriptedResponse::Text("done".into()),
                faktor_provider::ScriptedResponse::End,
            ]],
            false,
            true,
            faktor_orchestrator::runtime::task_executor::MutationMode::Shadow,
        );
        seed_native_owner(&rig.owner_root);
        let NativeTaskRig {
            deps,
            manager,
            parent: sid,
            owner_root,
            ..
        } = rig;
        let token = deps.auth_token.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        let resp = client
            .post(format!("{base}/native/session/{sid}/task-runs"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"goal": "implement the change"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        native_wait_session_state(
            &manager,
            sid,
            faktor_core::state::AgentState::ReadyForNextTurn,
        )
        .await;
        // The write stayed in the shadow; the owner checkout is untouched
        // until a verified integration.
        let shadow = manager
            .shadow_row(sid)
            .unwrap()
            .expect("an ordinary native mutating prompt must begin a shadow");
        assert_eq!(shadow.state, faktor_session::ShadowRowState::Active);
        assert_eq!(
            std::fs::read(std::path::Path::new(&shadow.root).join("src/lib.rs")).unwrap(),
            NATIVE_IMPL_LIB_RS.as_bytes(),
            "the edit landed in the shadow"
        );
        assert_eq!(
            std::fs::read(owner_root.join("src/lib.rs")).unwrap(),
            NATIVE_OWNER_LIB_RS.as_bytes(),
            "the owner checkout is byte-untouched"
        );
        let _ = handle.shutdown.send(());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn frozen_wire_message_returns_promptly_without_shadow_copy_and_keeps_shape() {
        // Regression (JetBrains split-mode smoke): the frozen v7.5.6
        // `POST /session/{id}/message` was translated into the executor's
        // production default (Shadow), so `TaskExecutor::start_in_session`
        // ran a SYNCHRONOUS `ShadowRoots::begin_shadow` copy of the session
        // workspace inline in the request. With the smoke's workspace (the
        // daemon CWD: a huge checkout) the POST blew the 5 s wire timeout,
        // and the synchronous copy wedged every other route. The compat
        // translation must select DirectCompat for the frozen wire: the
        // prompt still travels the ONE execution path (durable run row,
        // detached recoverable drive), but the request never waits on the
        // unrelated shadow-begin background work.
        let dir = tempfile::tempdir().unwrap();
        // Deterministic failure injection: an explicit stub whose stream
        // returns a TYPED provider error before any chunk, on every
        // platform. The drive therefore fails identically everywhere (no
        // workspace-path or host-speed dependence); it still travels the
        // ONE executor path (durable run row, detached recoverable drive)
        // and must surface the honest frozen-wire 502.
        let rig = native_task_rig_with_provider(
            dir.path(),
            Arc::new(AlwaysFailsProvider),
            false,
            true,
            faktor_orchestrator::runtime::task_executor::MutationMode::Shadow,
        );
        seed_native_owner(&rig.owner_root);
        let NativeTaskRig {
            deps,
            manager,
            parent: sid,
            ..
        } = rig;
        let pw = deps.server_password.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let basic = |r: reqwest::RequestBuilder| r.basic_auth("kilo", Some(pw.as_str()));

        // Bounded, not host-speed-bound: a handler that wedges forever
        // (e.g. on a synchronous shadow copy) still fails this test; the
        // deterministic typed failure above lands the machine terminal on
        // any host, and the no-shadow assertion below proves the actual
        // regression contract.
        let resp =
            tokio::time::timeout(
                Duration::from_secs(90),
                basic(client.post(format!("{base}/session/{sid}/message")).json(
                    &serde_json::json!({
                        "messageID": null,
                        "model": {"providerID": "fake", "modelID": "m"},
                        "parts": [{"type": "text", "text": "ping from the frozen wire"}],
                    }),
                ))
                .send(),
            )
            .await
            .expect("the frozen message handler must answer within the wire timeout")
            .unwrap();
        assert_eq!(
            resp.status(),
            502,
            "a turn without an assistant reply is an honest frozen-wire 502"
        );
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["ok"], false, "frozen failure shape: {body}");
        assert!(body["message"].is_string(), "{body}");
        // The wire flow selected the direct path: no shadow was ever begun
        // (the synchronous O(workspace) copy is exactly the regression).
        assert!(
            manager.shadow_row(sid).unwrap().is_none(),
            "the frozen wire must not begin a shadow worktree copy"
        );
        // The smoke's settle predicate: the turn lands terminal, never stuck
        // mid-machine. Deadline-based like the POST bound above — a slow
        // host may still be finishing the drive here.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(90);
        let mut settled = manager.get_session(sid).unwrap().unwrap().state().unwrap();
        loop {
            if matches!(
                settled,
                faktor_core::state::AgentState::ReadyForNextTurn
                    | faktor_core::state::AgentState::FailedRecoverable
            ) {
                break;
            }
            if tokio::time::Instant::now() >= deadline {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
            settled = manager.get_session(sid).unwrap().unwrap().state().unwrap();
        }
        assert!(
            matches!(
                settled,
                faktor_core::state::AgentState::ReadyForNextTurn
                    | faktor_core::state::AgentState::FailedRecoverable
            ),
            "unexpected settled state {settled:?}"
        );
        // The smoke's GET /session/{id}/message?limit=5: the frozen bare
        // array of {info, parts}; the user prompt row is durable. Bounded
        // for a slow host, never a strict wall-clock assumption.
        let resp = tokio::time::timeout(
            Duration::from_secs(30),
            basic(client.get(format!("{base}/session/{sid}/message?limit=5"))).send(),
        )
        .await
        .expect("the frozen message page must answer within the wire timeout")
        .unwrap();
        assert_eq!(resp.status(), 200);
        let page: serde_json::Value = resp.json().await.unwrap();
        let messages = page.as_array().expect("frozen page is a bare array");
        assert!(!messages.is_empty(), "expected >= 1 message, got {page}");
        let first = &messages[0];
        assert_eq!(first["info"]["role"], "user");
        assert!(
            first["info"]["messageID"]
                .as_str()
                .is_some_and(|m| !m.is_empty()),
            "the user row carries its durable id: {first}"
        );
        assert_eq!(first["parts"][0]["type"], "text");
        assert_eq!(first["parts"][0]["text"], "ping from the frozen wire");
        let _ = handle.shutdown.send(());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn legacy_plan_global_ownership_converts_once_at_the_native_dto_boundary() {
        // A legacy client posts ONE plan-global ownership with two mutating
        // items that carry none of their own: the DTO boundary converts it
        // ONCE onto the items (the runtime never sees the plan-global
        // value), and the run is accepted with explicit per-item ownership
        // on the durable assignment rows.
        use faktor_orchestrator::runtime::OrchestratorRuntime;
        let dir = tempfile::tempdir().unwrap();
        let rig = native_task_rig(
            dir.path(),
            vec![],
            false,
            false,
            faktor_orchestrator::runtime::task_executor::MutationMode::Shadow,
        );
        seed_native_owner(&rig.owner_root);
        let NativeTaskRig {
            deps,
            manager,
            parent: sid,
            ..
        } = rig;
        let token = deps.auth_token.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        let resp = client
            .post(format!("{base}/native/session/{sid}/task-runs"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({
                "goal": "legacy ownership",
                "ownership": "IsolatedWorktree",
                "work_items": [
                    {"id": "a", "kind": "Implementation"},
                    {"id": "b", "kind": "Implementation", "depends_on": ["a"]},
                ],
            }))
            .send()
            .await
            .unwrap();
        let status = resp.status();
        let start: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(status, 200, "{start}");
        let run_id = start["run_id"].as_str().unwrap().to_string();
        let mut assignments =
            OrchestratorRuntime::assignment_rows(manager.clone(), sid, &run_id).unwrap();
        for _ in 0..400 {
            if !assignments.is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
            assignments =
                OrchestratorRuntime::assignment_rows(manager.clone(), sid, &run_id).unwrap();
        }
        assert_eq!(assignments.len(), 2);
        for a in &assignments {
            assert_eq!(
                a.ownership,
                faktor_core::state::OwnershipSpec::IsolatedWorktree,
                "the legacy plan-global value converted onto item {}",
                a.item_id
            );
        }
        for _ in 0..600 {
            let rows = OrchestratorRuntime::registry_rows(manager.clone(), sid, &run_id).unwrap();
            if rows.len() == 2 && rows.iter().all(|c| c.state.is_terminal()) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let _ = handle.shutdown.send(());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sdk_and_native_prompts_hit_the_same_execution_service() {
        // The identity spy: one observer records every PromptExecutionService
        // call with the `Arc` pointers of the underlying TaskExecutor +
        // SessionManager. The SDK compat prompt and the native task start
        // must report the SAME pointers (one execution authority, never a
        // per-adapter runtime).
        use crate::native::{set_prompt_observer, PromptCallKind};
        use std::sync::Mutex;
        let dir = tempfile::tempdir().unwrap();
        let rig = native_task_rig(
            dir.path(),
            vec![],
            false,
            false,
            faktor_orchestrator::runtime::task_executor::MutationMode::Shadow,
        );
        seed_native_owner(&rig.owner_root);
        let NativeTaskRig {
            deps,
            manager,
            parent: sid,
            ..
        } = rig;
        let tasks_ptr = std::sync::Arc::as_ptr(&deps.tasks) as usize;
        let sessions_ptr = std::sync::Arc::as_ptr(&deps.session) as usize;
        let seen: Arc<Mutex<Vec<(PromptCallKind, usize, usize)>>> = Arc::new(Mutex::new(vec![]));
        let sink = seen.clone();
        set_prompt_observer(Some(Arc::new(move |call| {
            // Other tests run in parallel in this binary; only calls on THIS
            // test's executor are the tripwire.
            if call.tasks_ptr == tasks_ptr {
                sink.lock()
                    .unwrap()
                    .push((call.kind, call.tasks_ptr, call.sessions_ptr));
            }
        })));
        let token = deps.auth_token.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        // SDK compat prompt.
        let resp = client
            .post(format!("{base}/session/prompt"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({
                "session_id": sid.to_string(),
                "prompt": "hello",
                "files": []
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        native_wait_session_state(
            &manager,
            sid,
            faktor_core::state::AgentState::ReadyForNextTurn,
        )
        .await;
        // Native ordinary prompt (goal only).
        let resp = client
            .post(format!("{base}/native/session/{sid}/task-runs"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"goal": "hello native"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        set_prompt_observer(None);

        let calls = seen.lock().unwrap().clone();
        assert!(
            calls.len() >= 2,
            "both adapter prompts must hit the one service: {calls:?}"
        );
        assert!(
            calls.iter().any(|(k, _, _)| *k == PromptCallKind::Prompt),
            "the SDK/native prompt calls were observed: {calls:?}"
        );
        for (kind, tasks, sessions) in &calls {
            assert_eq!(
                *tasks, tasks_ptr,
                "call {kind:?} must execute on the ONE TaskExecutor"
            );
            assert_eq!(
                *sessions, sessions_ptr,
                "call {kind:?} must execute on the ONE SessionManager"
            );
        }
        let _ = handle.shutdown.send(());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn native_task_run_direct_compat_mode_is_byte_identical_to_no_service() {
        // The HTTP parity test: the same goal/scripts on (a) a daemon
        // carrying the shadow service in DirectCompat mode (the configured
        // production escape hatch) and (b) a daemon without any shadow
        // service (the historical wiring) must produce byte-identical
        // outcomes — DirectCompat selects the direct workspace and nothing
        // else.
        async fn drive_one(
            root: &std::path::Path,
            service: bool,
        ) -> (Vec<u8>, i64, serde_json::Value) {
            let rig = native_task_rig(
                root,
                vec![
                    vec![
                        faktor_provider::ScriptedResponse::ToolCall {
                            id: "c1".into(),
                            name: "write_file".into(),
                            input: serde_json::json!({
                                "path": "src/lib.rs",
                                "content": "pub fn value() -> u64 {\n    let base_amount: u64 = 11;\n    let increment: u64 = 31;\n    base_amount.saturating_add(increment)\n}\n",
                            }),
                        },
                        faktor_provider::ScriptedResponse::Text("done".into()),
                        faktor_provider::ScriptedResponse::End,
                    ],
                    vec![faktor_provider::ScriptedResponse::End],
                ],
                false,
                service,
                faktor_orchestrator::runtime::task_executor::MutationMode::DirectCompat,
            );
            seed_native_owner(&rig.owner_root);
            let NativeTaskRig {
                deps,
                manager,
                parent: sid,
                owner_root,
                ..
            } = rig;
            let token = deps.auth_token.clone();
            let handle = serve(deps, 0).await.unwrap();
            let base = format!("http://{}", handle.addr);
            let client = reqwest::Client::new();
            // mutation_mode omitted: the daemon default decides
            // (DirectCompat on the service daemon; no service on the
            // historical one).
            let resp = client
                .post(format!("{base}/native/session/{sid}/task-runs"))
                .bearer_auth(token.as_str())
                .json(&serde_json::json!({"goal": "implement the change"}))
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 200);
            let start: serde_json::Value = resp.json().await.unwrap();
            assert!(
                start["run_id"].as_str().unwrap().starts_with("tx-"),
                "{start}"
            );
            assert_eq!(start["task_id"], 1);
            native_wait_session_state(
                &manager,
                sid,
                faktor_core::state::AgentState::ReadyForNextTurn,
            )
            .await;
            // DirectCompat never shadows — even when the service exists.
            assert!(
                manager.shadow_row(sid).unwrap().is_none(),
                "no shadow row may exist"
            );
            assert_eq!(manager.active_root(sid).unwrap(), None, "no re-pointing");
            let final_bytes = std::fs::read(owner_root.join("src/lib.rs")).unwrap();
            let h = manager.get_session(sid).unwrap().unwrap();
            let messages = h.message_count().unwrap();
            // A completed run reads Done on the per-run surface.
            let mut done = None;
            for _ in 0..1200 {
                let resp = client
                    .get(format!(
                        "{base}/native/session/{sid}/task-runs/{}",
                        start["run_id"].as_str().unwrap()
                    ))
                    .bearer_auth(token.as_str())
                    .send()
                    .await
                    .unwrap();
                let v: serde_json::Value = resp.json().await.unwrap();
                done = Some(v.clone());
                if v["state"] == "Done" {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            let _ = handle.shutdown.send(());
            (final_bytes, messages, done.unwrap())
        }
        let dir = tempfile::tempdir().unwrap();
        let (bytes_a, msgs_a, run_a) = drive_one(&dir.path().join("direct"), true).await;
        let (bytes_b, msgs_b, run_b) = drive_one(&dir.path().join("plain"), false).await;
        assert_eq!(
            bytes_a, bytes_b,
            "byte-identical owner content on both sides"
        );
        assert_eq!(
            String::from_utf8_lossy(&bytes_a),
            "pub fn value() -> u64 {\n    let base_amount: u64 = 11;\n    let increment: u64 = 31;\n    base_amount.saturating_add(increment)\n}\n"
        );
        // Durable parity: same message streams, same run projections.
        assert_eq!(msgs_a, msgs_b, "byte-identical message streams");
        assert_eq!(run_a["state"], run_b["state"]);
        assert_eq!(run_a["goal"], run_b["goal"]);
        assert_eq!(run_a["item_ids"], run_b["item_ids"]);
        assert_eq!(run_a["task_id"], run_b["task_id"]);
        assert_eq!(run_a["mode"], "in_session");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn native_task_run_start_hostile_dtos_are_typed_400s() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let token = deps.auth_token.clone();
        let manager = deps.session.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let ws = manager.create_workspace("/plain").unwrap();
        let sid = manager
            .create_session(ws, "hostile", "fake", "m")
            .unwrap()
            .id();

        // Unauthenticated is 401 before anything else.
        let resp = client
            .post(format!("{base}/native/session/{sid}/task-runs"))
            .json(&serde_json::json!({"goal": "x"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);

        let oversized_goal = "x".repeat(2001);
        let hostile_bodies: Vec<serde_json::Value> = vec![
            serde_json::json!({}),
            serde_json::json!({"goal": ""}),
            serde_json::json!({"goal": "x", "bogus": 1}),
            serde_json::json!({"goal": "x", "mutation_mode": "nonsense"}),
            serde_json::json!({"goal": "x", "mutation_mode": "Shadow"}),
            serde_json::json!({"goal": "x", "routing_mode": "economy"}),
            serde_json::json!({"goal": oversized_goal}),
            serde_json::json!({"goal": "x", "max_tokens": "many"}),
            serde_json::json!({"goal": "x", "criteria": (0..=faktor_session::MAX_TASK_CRITERIA).map(|i| format!("criterion {i}")).collect::<Vec<_>>()}),
            serde_json::json!({"goal": "x", "criteria": vec!["c".repeat(faktor_session::MAX_TASK_CRITERION_BYTES + 1)]}),
            serde_json::json!({"goal": "x", "work_items": [{"id": "a", "kind": "Implementation"}, {"id": "b", "kind": "Implementation"}]}),
            serde_json::json!({"goal": "x", "work_items": [{"id": "a a/..", "kind": "Analysis"}]}),
            serde_json::json!({"goal": "x", "work_items": [{"id": "a", "kind": "Analysis"}, {"id": "a", "kind": "Analysis"}]}),
            serde_json::json!({"goal": "x", "work_items": [{"id": "a", "kind": "NoSuchKind"}]}),
            serde_json::json!({"goal": "x", "work_items": [{"id": "a", "kind": "Analysis", "extra": 1}]}),
            serde_json::json!({"goal": "x", "work_items": "not-an-array"}),
        ];
        for body in hostile_bodies {
            let resp = client
                .post(format!("{base}/native/session/{sid}/task-runs"))
                .bearer_auth(token.as_str())
                .json(&body)
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 400, "hostile body must 400: {body}");
        }
        // Non-JSON bodies are plain 400s; unknown sessions are 404s.
        let resp = client
            .post(format!("{base}/native/session/{sid}/task-runs"))
            .bearer_auth(token.as_str())
            .header("content-type", "application/json")
            .body("{not json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);
        let resp = client
            .post(format!("{base}/native/session/999999/task-runs"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"goal": "x"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);
        // No provider call ever happened on the hostile attempts.
        let h = manager.get_session(sid).unwrap().unwrap();
        assert_eq!(h.message_count().unwrap(), 0, "hostile starts never drive");
        let _ = handle.shutdown.send(());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn native_tournament_start_hostile_dtos_are_typed_400s() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let token = deps.auth_token.clone();
        let manager = deps.session.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let ws = manager.create_workspace("/plain").unwrap();
        let sid = manager
            .create_session(ws, "hostile-tournament", "fake", "m")
            .unwrap()
            .id();

        // Unauthenticated is 401 before anything else.
        let resp = client
            .post(format!("{base}/native/session/{sid}/tournament"))
            .json(&serde_json::json!({"goal": "x", "criteria": ["c"], "n": 2}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);

        let oversized_goal = "x".repeat(513);
        let hostile_bodies: Vec<serde_json::Value> = vec![
            serde_json::json!({}),
            serde_json::json!({"goal": "x", "criteria": ["c"]}),
            serde_json::json!({"goal": "", "criteria": ["c"], "n": 2}),
            serde_json::json!({"goal": "x", "criteria": [], "n": 2}),
            serde_json::json!({"goal": "x", "criteria": ["c"], "n": 0}),
            serde_json::json!({"goal": "x", "criteria": ["c"], "n": 1}),
            serde_json::json!({"goal": "x", "criteria": ["c"], "n": 5}),
            serde_json::json!({"goal": "x", "criteria": ["c"], "n": "two"}),
            serde_json::json!({"goal": "x", "criteria": ["c"], "n": 2, "bogus": 1}),
            serde_json::json!({"goal": "x", "criteria": ["c"], "n": 2, "mutation_mode": "nonsense"}),
            serde_json::json!({"goal": oversized_goal, "criteria": ["c"], "n": 2}),
            serde_json::json!({"goal": "x", "criteria": ["c".repeat(600)], "n": 2}),
        ];
        for body in hostile_bodies {
            let resp = client
                .post(format!("{base}/native/session/{sid}/tournament"))
                .bearer_auth(token.as_str())
                .json(&body)
                .send()
                .await
                .unwrap();
            assert_eq!(
                resp.status(),
                400,
                "hostile tournament body must 400: {body}"
            );
        }
        // Non-JSON bodies are plain 400s; unknown sessions are 404s; an
        // unknown tournament id is a 404 (never a phantom tournament).
        let resp = client
            .post(format!("{base}/native/session/{sid}/tournament"))
            .bearer_auth(token.as_str())
            .header("content-type", "application/json")
            .body("{not json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);
        let resp = client
            .post(format!("{base}/native/session/999999/tournament"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"goal": "x", "criteria": ["c"], "n": 2}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);
        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/session/{sid}/tournament/does-not-exist"),
        )
        .await;
        assert_eq!(resp.status(), 404);
        // No provider call ever happened on the hostile attempts.
        let h = manager.get_session(sid).unwrap().unwrap();
        assert_eq!(h.message_count().unwrap(), 0, "hostile starts never drive");
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn native_tournaments_list_summarizes_the_durable_fold() {
        // The additive listing folds the pinned tournament lifecycle rows:
        // id, state, candidate count, winner and the decision timestamp.
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let manager = deps.session.clone();
        let token = deps.auth_token.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let ws = manager.create_workspace("/plain").unwrap();
        let s = manager
            .create_session(ws, "tour-list", "fake", "m")
            .unwrap();
        let sid = s.id();

        // Unauthenticated is 401.
        let resp = client
            .get(format!("{base}/native/session/{sid}/tournaments"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);
        // A session without tournaments is an empty list, never a phantom.
        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/session/{sid}/tournaments"),
        )
        .await;
        assert_eq!(resp.status(), 200);
        assert_eq!(
            resp.json::<serde_json::Value>().await.unwrap(),
            serde_json::json!([])
        );
        // Hostile session ids are typed (malformed 400 / unknown 404).
        let resp = native_get(&client, &base, &token, "/native/session/nope/tournaments").await;
        assert_eq!(resp.status(), 400);
        let resp = native_get(&client, &base, &token, "/native/session/999999/tournaments").await;
        assert_eq!(resp.status(), 404);

        // Seed one OPEN and one DECIDED tournament through the typed ledger.
        let criteria = vec![faktor_session::ledger::TournamentCriterionRow {
            id: "c1".into(),
            spec: "cargo test".into(),
        }];
        let candidates: Vec<faktor_session::ledger::TournamentCandidateRow> = (0..2)
            .map(|i| faktor_session::ledger::TournamentCandidateRow {
                child_id: format!("child-{i}"),
                worktree: String::new(),
                base_revision: String::new(),
            })
            .collect();
        s.ledger_tournament_started("tour-open", "run-open", "open goal", &criteria, &candidates)
            .unwrap();
        s.ledger_tournament_started("tour-done", "run-done", "done goal", &criteria, &candidates)
            .unwrap();
        s.ledger_tournament_decided(
            "tour-done",
            Some("child-1"),
            faktor_session::TOURNAMENT_OUTCOME_DECIDED,
            "winner child-1",
        )
        .unwrap();

        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/session/{sid}/tournaments"),
        )
        .await;
        assert_eq!(resp.status(), 200);
        let list: serde_json::Value = resp.json().await.unwrap();
        let list = list.as_array().unwrap();
        assert_eq!(list.len(), 2);
        let open = list.iter().find(|e| e["id"] == "tour-open").unwrap();
        assert_eq!(open["state"], "open");
        assert_eq!(open["candidate_count"], 2);
        assert!(open["winner"].is_null());
        assert!(open["decided_ms"].is_null());
        let done = list.iter().find(|e| e["id"] == "tour-done").unwrap();
        assert_eq!(done["state"], "decided");
        assert_eq!(done["candidate_count"], 2);
        assert_eq!(done["winner"], "child-1");
        assert!(done["decided_ms"].as_i64().unwrap_or(0) > 0);
        // The listing is a durable fold: it survives a ledger compaction
        // (tournament rows are pinned).
        s.compact_typed_ledger().unwrap();
        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/session/{sid}/tournaments"),
        )
        .await;
        let list: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(list.as_array().unwrap().len(), 2);
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn native_tournament_decide_and_abort_are_strict_and_engine_gated() {
        // The additive decide/abort routes over the typed ledger: happy
        // decide (deterministic winner + losers discarded), happy abort
        // (terminal row with the reason), and every refusal boundary —
        // unknown ids 404, non-open/no-eligible-winner 409, hostile bodies
        // and oversized reasons 400, missing auth 401. The engine stays the
        // ONE authority: the routes never mutate the fold directly.
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let manager = deps.session.clone();
        let token = deps.auth_token.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let ws = manager.create_workspace("/plain").unwrap();
        let s = manager
            .create_session(ws, "tour-control", "fake", "m")
            .unwrap();
        let sid = s.id();

        let criteria = vec![faktor_session::ledger::TournamentCriterionRow {
            id: "c1".into(),
            spec: "cargo test".into(),
        }];
        let candidates: Vec<faktor_session::ledger::TournamentCandidateRow> = (0..2)
            .map(|i| faktor_session::ledger::TournamentCandidateRow {
                child_id: format!("child-{i}"),
                worktree: String::new(),
                base_revision: String::new(),
            })
            .collect();
        let derived = faktor_orchestrator::tournament::derive_check_specs(&[
            faktor_orchestrator::tournament::Criterion {
                id: "c1".into(),
                spec: "cargo test".into(),
            },
        ]);
        let settlement = |child_id: &str, rank: &str, cost: u64| {
            faktor_session::ledger::TournamentSettlementRow {
                child_id: child_id.into(),
                worktree: String::new(),
                base_revision: String::new(),
                state: "done".into(),
                verification: Some(7),
                verification_pass: Some(true),
                checks: derived.clone(),
                review: Some(rank.into()),
                reviewer: Some("review-0".into()),
                cost_micro: cost,
                wall_ms: 100,
                reason: "settled".into(),
            }
        };

        // Unauthenticated is 401 before anything else.
        let resp = client
            .post(format!(
                "{base}/native/session/{sid}/tournaments/tour-x/decide"
            ))
            .json(&serde_json::json!({}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);

        // A tournament with no eligible candidate refuses decide as a typed
        // 409 (nothing was persisted; a second decide reads the same state).
        s.ledger_tournament_started("tour-bare", "run-bare", "bare goal", &criteria, &candidates)
            .unwrap();
        let resp = client
            .post(format!(
                "{base}/native/session/{sid}/tournaments/tour-bare/decide"
            ))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 409);
        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/session/{sid}/tournament/tour-bare"),
        )
        .await;
        assert_eq!(resp.status(), 200);
        assert_eq!(
            resp.json::<serde_json::Value>().await.unwrap()["state"],
            "open"
        );

        // Happy decide: clean review outranks the cheaper concern reviewer.
        s.ledger_tournament_started(
            "tour-happy",
            "run-happy",
            "happy goal",
            &criteria,
            &candidates,
        )
        .unwrap();
        s.ledger_candidate_settled("tour-happy", &settlement("child-0", "clean", 500))
            .unwrap();
        s.ledger_candidate_settled("tour-happy", &settlement("child-1", "concern", 1))
            .unwrap();
        let resp = client
            .post(format!(
                "{base}/native/session/{sid}/tournaments/tour-happy/decide"
            ))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let decided: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(decided["tournament_id"], "tour-happy");
        assert_eq!(decided["winner"], "child-0");
        assert!(decided["rationale"]
            .as_str()
            .unwrap_or("")
            .contains("child-0"));
        let discarded = decided["discarded"].as_array().unwrap();
        assert_eq!(discarded.len(), 1);
        assert_eq!(discarded[0]["child_id"], "child-1");
        // Wrong state: deciding or aborting the decided tournament is 409.
        let resp = client
            .post(format!(
                "{base}/native/session/{sid}/tournaments/tour-happy/decide"
            ))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 409);
        let resp = client
            .post(format!(
                "{base}/native/session/{sid}/tournaments/tour-happy/abort"
            ))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"reason": "too late"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 409);
        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/session/{sid}/tournament/tour-happy"),
        )
        .await;
        let state: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(state["state"], "decided");
        assert_eq!(state["winner"], "child-0");
        assert_eq!(state["candidates"][1]["state"], "discarded");

        // Happy abort: the terminal row carries the reason and every
        // candidate is discarded; a second abort is a typed 409.
        s.ledger_tournament_started(
            "tour-abort",
            "run-abort",
            "abort goal",
            &criteria,
            &candidates,
        )
        .unwrap();
        s.ledger_candidate_settled("tour-abort", &settlement("child-0", "clean", 10))
            .unwrap();
        let resp = client
            .post(format!(
                "{base}/native/session/{sid}/tournaments/tour-abort/abort"
            ))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"reason": "operator stopped it"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let aborted: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(aborted["id"], "tour-abort");
        assert_eq!(aborted["state"], "aborted");
        assert!(aborted["winner"].is_null());
        for candidate in aborted["candidates"].as_array().unwrap() {
            assert_eq!(candidate["state"], "discarded");
        }
        let resp = client
            .post(format!(
                "{base}/native/session/{sid}/tournaments/tour-abort/abort"
            ))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 409);

        // Hostile boundaries: unknown tournament 404, unknown session 404,
        // strict bodies 400 (unknown member / non-JSON / missing body /
        // oversized reason).
        let resp = client
            .post(format!(
                "{base}/native/session/{sid}/tournaments/nope/decide"
            ))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);
        let resp = client
            .post(format!(
                "{base}/native/session/{sid}/tournaments/nope/abort"
            ))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);
        let resp = client
            .post(format!(
                "{base}/native/session/999999/tournaments/nope/decide"
            ))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);
        for body in [
            serde_json::json!({"bogus": 1}),
            serde_json::json!({"reason": "decide takes no reason"}),
        ] {
            let resp = client
                .post(format!(
                    "{base}/native/session/{sid}/tournaments/tour-bare/decide"
                ))
                .bearer_auth(token.as_str())
                .json(&body)
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 400, "hostile decide body: {body}");
        }
        let resp = client
            .post(format!(
                "{base}/native/session/{sid}/tournaments/tour-bare/decide"
            ))
            .bearer_auth(token.as_str())
            .header("content-type", "application/json")
            .body("{not json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);
        let resp = client
            .post(format!(
                "{base}/native/session/{sid}/tournaments/tour-bare/decide"
            ))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);
        let resp = client
            .post(format!(
                "{base}/native/session/{sid}/tournaments/tour-bare/abort"
            ))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"reason": "x".repeat(600)}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);
        let _ = handle.shutdown.send(());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn native_task_run_list_state_and_cancel_reflect_the_durable_run() {
        // List/state/cancel over HTTP on a real session: a completed
        // read-only run reads Done on both surfaces and refuses cancel; a
        // mid-flight run is cancelled at the task level (durable row
        // Cancelled, drive aborted) and stays Cancelled; hostile ids stay
        // typed.
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let token = deps.auth_token.clone();
        let manager = deps.session.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let ws = manager.create_workspace("/plain").unwrap();
        let sid = manager
            .create_session(ws, "list-cancel", "fake", "m")
            .unwrap()
            .id();

        // Fresh session: an empty task-run list and typed 404 per-run reads.
        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/session/{sid}/task-runs"),
        )
        .await;
        assert_eq!(resp.status(), 200);
        assert_eq!(
            resp.json::<serde_json::Value>().await.unwrap(),
            serde_json::json!([])
        );
        for hostile in ["tx-1", "run-x", "..", "a/b"] {
            let resp = native_get(
                &client,
                &base,
                &token,
                &format!("/native/session/{sid}/task-runs/{hostile}"),
            )
            .await;
            assert_eq!(resp.status(), 404, "hostile run id {hostile:?}");
        }

        // A completed read-only run (explicit Analysis item + criteria):
        // the list/state reflect the terminal run; cancelling it is a 409.
        let resp = client
            .post(format!("{base}/native/session/{sid}/task-runs"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({
                "goal": "analyze the module boundaries",
                "criteria": ["the analysis names the seams"],
                "work_items": [{"id": "a1", "kind": "Analysis"}],
                "max_tokens": 100_000,
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let start: serde_json::Value = resp.json().await.unwrap();
        let run_id = start["run_id"].as_str().unwrap().to_string();
        assert_eq!(start["task_id"], 1);
        // Criteria rode the durable task row.
        let h = manager.get_session(sid).unwrap().unwrap();
        assert_eq!(
            h.get_task(faktor_core::id::TaskId::new(1))
                .unwrap()
                .unwrap()
                .acceptance_criteria,
            vec!["the analysis names the seams".to_string()]
        );
        let mut state = None;
        for _ in 0..300 {
            let resp = native_get(
                &client,
                &base,
                &token,
                &format!("/native/session/{sid}/task-runs/{run_id}"),
            )
            .await;
            let v: serde_json::Value = resp.json().await.unwrap();
            state = Some(v.clone());
            if v["state"] == "Done" {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let entry = state.expect("the run must settle");
        assert_eq!(entry["state"], "Done");
        assert_eq!(entry["run_id"], run_id);
        assert_eq!(entry["mode"], "in_session");
        assert_eq!(entry["goal"], "analyze the module boundaries");
        assert_eq!(entry["item_ids"], serde_json::json!(["a1"]));
        // The list carries the same projection.
        let resp = native_get(
            &client,
            &base,
            &token,
            &format!("/native/session/{sid}/task-runs"),
        )
        .await;
        let list: serde_json::Value = resp.json().await.unwrap();
        let list_entry = list
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["run_id"] == run_id)
            .expect("settled run listed");
        assert_eq!(list_entry["state"], "Done");
        // A run whose task row is durably TERMINAL (verified complete)
        // refuses cancel — a typed 409, never a silent no-op.
        certify_native_task(&manager, sid);
        let resp = client
            .post(format!(
                "{base}/native/session/{sid}/task-runs/{run_id}/cancel"
            ))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 409, "terminal runs refuse cancel");
        let resp = client
            .post(format!("{base}/native/session/{sid}/task-runs/nope/cancel"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404, "unknown runs are typed 404s");
        let _ = handle.shutdown.send(());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn native_task_run_cancel_aborts_a_mid_flight_in_session_run() {
        // A mid-flight in-session drive (paced text-only provider, no
        // tools) is cancelled at the TASK level: the drive is aborted
        // durably, the task row turns Cancelled, and the task-runs surface
        // reads Cancelled.
        let dir = tempfile::tempdir().unwrap();
        let paced = PacedScriptedProvider::new(
            faktor_core::model::ModelCapabilities {
                tools: true,
                ..Default::default()
            },
            vec![
                (0..400)
                    .map(|i| faktor_provider::ScriptedResponse::Text(format!("tick {i}")))
                    .chain(std::iter::once(faktor_provider::ScriptedResponse::End))
                    .collect(),
                vec![faktor_provider::ScriptedResponse::End],
            ],
            10,
        );
        let deps = paced_test_deps(dir.path(), paced);
        let token = deps.auth_token.clone();
        let manager = deps.session.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let ws = manager.create_workspace("/plain").unwrap();
        let sid = manager
            .create_session(ws, "cancel-mid", "fake", "m")
            .unwrap()
            .id();
        let resp = client
            .post(format!("{base}/native/session/{sid}/task-runs"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({
                "goal": "long-running analysis",
                "work_items": [{"id": "a1", "kind": "Analysis"}],
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let start: serde_json::Value = resp.json().await.unwrap();
        let run_id = start["run_id"].as_str().unwrap().to_string();
        // Wait for the drive to be mid-flight (the session is actively
        // working — anything but parked/terminal), then cancel at the task
        // level.
        for _ in 0..600 {
            let st = manager.get_session(sid).unwrap().unwrap().state().unwrap();
            if !st.is_terminal()
                && !matches!(
                    st,
                    faktor_core::state::AgentState::ReadyForNextTurn
                        | faktor_core::state::AgentState::Idle
                        | faktor_core::state::AgentState::Suspended
                )
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        let resp = client
            .post(format!(
                "{base}/native/session/{sid}/task-runs/{run_id}/cancel"
            ))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let ack: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(ack["run_id"], run_id);
        assert_eq!(ack["cancelled"], true);
        // The durable outcome: task row Cancelled, session parked, task-runs
        // state Cancelled; a second cancel is a typed 409.
        for _ in 0..300 {
            let resp = native_get(
                &client,
                &base,
                &token,
                &format!("/native/session/{sid}/task-runs/{run_id}"),
            )
            .await;
            let v: serde_json::Value = resp.json().await.unwrap();
            if v["state"] == "Cancelled" {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let h = manager.get_session(sid).unwrap().unwrap();
        let task = h
            .get_task(faktor_core::id::TaskId::new(1))
            .unwrap()
            .unwrap();
        assert_eq!(
            task.state,
            faktor_core::state::TaskState::Cancelled,
            "the task row is durably Cancelled"
        );
        let resp = client
            .post(format!(
                "{base}/native/session/{sid}/task-runs/{run_id}/cancel"
            ))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            409,
            "a cancelled run is never cancelled twice"
        );
        let _ = handle.shutdown.send(());
    }

    #[test]
    fn server_reaches_task_start_only_through_the_executor() {
        // The single-authority source scan: in the NON-TEST server code the
        // ONLY TaskExecutor start edge is the native start handler, the
        // ONLY TaskRunRequest construction lives in that same handler, and
        // the legacy prompt-drive helper (submit_and_run) never constructs a
        // task run — the prompt surface stays a prompt surface. The scan
        // spans every server source file of the audit 81-83/94 split
        // (api router assembly + native/* + compat/*).
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let api_src = std::fs::read_to_string(root.join("api.rs")).expect("api.rs source");
        let mut src = api_src
            .split_once("mod tests {")
            .expect("the tests module marker exists")
            .0
            .to_string();
        for module in [
            "native/mod.rs",
            "native/session.rs",
            "native/task.rs",
            "native/agents.rs",
            "native/evidence.rs",
            "native/verification.rs",
            "native/terminal.rs",
            "native/usage.rs",
            "native/semantic.rs",
            "native/models.rs",
            "compat/mod.rs",
            "compat/sdk.rs",
            "compat/v756.rs",
        ] {
            src.push_str(
                &std::fs::read_to_string(root.join(module)).expect("server module source"),
            );
            src.push('\n');
        }
        let lines: Vec<&str> = src.lines().collect();
        let is_decl_start = |l: &str| {
            l.starts_with("async fn ")
                || l.starts_with("fn ")
                || l.starts_with("pub(crate) async fn ")
                || l.starts_with("pub(crate) fn ")
        };
        let in_handler = |i: usize| {
            let Some(start) = lines
                .iter()
                .position(|l| l.contains("async fn native_task_run_start("))
            else {
                return false;
            };
            let end = lines
                .iter()
                .enumerate()
                .skip(start + 1)
                .find(|(_, l)| is_decl_start(l))
                .map(|(j, _)| j)
                .expect("a declaration follows the start handler");
            i > start && i < end
        };
        // Exactly one `.start_task(` call, inside the native start handler.
        let starts: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter(|(_, l)| l.contains(".start_task("))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(starts.len(), 1, "one task-start call site: {starts:?}");
        assert!(
            in_handler(starts[0]),
            "the start call must live in native_task_run_start, at line {}",
            starts[0]
        );
        assert!(
            lines[starts[0]].contains("prompts.start_task"),
            "the native handler reaches the executor ONLY through the \
             PromptExecutionService: {}",
            lines[starts[0]]
        );
        // Exactly one TaskRunRequest construction, in the same handler.
        let requests: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter(|(_, l)| l.contains("task_executor::TaskRunRequest {"))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(requests.len(), 1, "one TaskRunRequest site: {requests:?}");
        assert!(in_handler(requests[0]), "line {}", requests[0]);
        // The legacy prompt helper never touches run rows or the executor.
        let helper = lines
            .iter()
            .position(|l| l.contains("fn submit_and_run("))
            .expect("submit_and_run exists");
        let mut depth = 0usize;
        let mut helper_end = lines.len();
        let mut opened = false;
        for (j, l) in lines.iter().enumerate().skip(helper) {
            depth = depth
                .saturating_add(l.chars().filter(|&c| c == '{').count())
                .saturating_sub(l.chars().filter(|&c| c == '}').count());
            opened |= depth > 0;
            if opened && j > helper && depth == 0 {
                helper_end = j + 1;
                break;
            }
        }
        for l in &lines[helper..helper_end] {
            assert!(
                !l.contains(".start_task(") && !l.contains("task_executor::TaskRunRequest"),
                "prompt helper must never start a task run: {l}"
            );
        }
        // (work-entry unification) NO non-test server code drives the agent
        // directly: every ordinary prompt and every explicit task start goes
        // through the PromptExecutionService (compat translates DTOs only).
        let drives: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter(|(_, l)| {
                l.contains(".run_session_queue(")
                    || l.contains(".drive_receipt(")
                    || l.contains("agent.submit(")
            })
            .map(|(i, _)| i)
            .collect();
        assert!(
            drives.is_empty(),
            "no direct AgentRuntime drive may remain in server production code: {drives:?}"
        );
        // ... and the ONE start edge is the prompt service, which itself
        // wraps the executor (never the agent).
        let service_src = std::fs::read_to_string(root.join("native/prompt.rs"))
            .expect("native/prompt.rs source");
        assert!(
            service_src.contains("self.tasks.start_task("),
            "PromptExecutionService must start runs through the TaskExecutor"
        );
        for (i, l) in service_src.lines().enumerate() {
            assert!(
                !l.contains(".run_session_queue(")
                    && !l.contains(".drive_receipt(")
                    && !l.contains("agent.submit("),
                "prompt service line {} drives the agent directly: {l}",
                i + 1
            );
        }
    }

    // ------------------------------------------------ native evidence (audit 82)

    /// One complete, range-readable evidence envelope with retained backing
    /// (`b"hello"`), owned by `session`/`workspace`.
    fn evidence_envelope(
        id: u64,
        session: u64,
        workspace: u64,
        allow_ranges: bool,
    ) -> faktor_evidence::types::EvidenceEnvelope {
        faktor_evidence::types::EvidenceEnvelope::new(
            faktor_evidence::types::EvidenceId(id),
            faktor_evidence::types::EvidenceKind::ProcessLog,
            SessionId::new(session),
            WorkspaceId::new(workspace),
            None,
            None,
            faktor_evidence::types::ProvenanceSet::new([
                faktor_evidence::types::ProvenanceSource::Tool,
            ]),
            faktor_evidence::types::Compressibility::Reversible,
            faktor_evidence::types::CompactRepresentation {
                grammar: "log-v1".into(),
                body: "hello".into(),
            },
            Some([3u8; 32]),
            faktor_evidence::types::BackingCompleteness::Complete,
            faktor_evidence::types::CompressionRecord::identity(5),
            faktor_evidence::types::RetrievalPolicy::new(allow_ranges, true, 1024),
        )
        .unwrap()
    }

    fn evidence_handle(
        envelopes: Vec<faktor_evidence::types::EvidenceEnvelope>,
    ) -> EvidenceStoreHandle {
        let mut store = faktor_evidence::store::MemoryEvidenceStore::new(4096);
        for env in envelopes {
            store.insert(env, Some(b"hello".to_vec())).unwrap();
        }
        Arc::new(std::sync::RwLock::new(
            Box::new(store) as Box<dyn faktor_evidence::store::EvidenceStore + Send + Sync>
        ))
    }

    #[tokio::test]
    async fn native_evidence_foreign_scope_is_typed_denial() {
        // Knowing a valid evidence id is not authorization (audit 82): the
        // same id read under a foreign session's scope is a typed 403.
        let dir = tempfile::tempdir().unwrap();
        let mut deps = test_deps(dir.path());
        let ws = deps.session.create_workspace("/tmp").unwrap();
        let owner = deps
            .session
            .create_session(ws, "owner", "fake", "m")
            .unwrap();
        let foreign = deps
            .session
            .create_session(ws, "foreign", "fake", "m")
            .unwrap();
        let owner_row = owner.row().unwrap();
        let foreign_row = foreign.row().unwrap();
        deps.evidence = Some(evidence_handle(vec![evidence_envelope(
            7,
            owner_row.id.raw(),
            owner_row.workspace_id.raw(),
            true,
        )]));
        let pw = deps.server_password.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        // The owning session reads its own evidence.
        let resp = client
            .get(format!(
                "{base}/native/evidence/7?session={}",
                owner_row.id.raw()
            ))
            .header("x-faktor-server-password", pw.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["id"], 7);
        assert_eq!(body["sessionId"], owner_row.id.raw());
        assert_eq!(body["backingRetained"], true);
        // A foreign session that knows the id gets no bytes and no oracle:
        // 403 with the typed denial code.
        let resp = client
            .get(format!(
                "{base}/native/evidence/7?session={}",
                foreign_row.id.raw()
            ))
            .header("x-faktor-server-password", pw.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 403, "foreign scope must be denied");
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["error"]["code"], "evidence_access_denied");
        // The retrieve path enforces the same scope.
        let resp = client
            .post(format!(
                "{base}/native/evidence/7/retrieve?session={}",
                foreign_row.id.raw()
            ))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"selector": "all"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 403);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["error"]["code"], "evidence_access_denied");
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn native_evidence_unknown_and_oversized_are_honest_errors() {
        let dir = tempfile::tempdir().unwrap();
        let mut deps = test_deps(dir.path());
        let ws = deps.session.create_workspace("/tmp").unwrap();
        let session = deps.session.create_session(ws, "t", "fake", "m").unwrap();
        let row = session.row().unwrap();
        deps.evidence = Some(evidence_handle(vec![evidence_envelope(
            7,
            row.id.raw(),
            row.workspace_id.raw(),
            true,
        )]));
        let pw = deps.server_password.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        // Unknown id: 404, never a scope oracle.
        let resp = client
            .get(format!(
                "{base}/native/evidence/999?session={}",
                row.id.raw()
            ))
            .header("x-faktor-server-password", pw.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);
        // Oversized search selector: the bound is enforced before the store.
        let oversized = "x".repeat(5000);
        let resp = client
            .post(format!(
                "{base}/native/evidence/7/retrieve?session={}",
                row.id.raw()
            ))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({
                "selector": "search",
                "query": oversized,
                "max_hits": 1,
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400, "oversized selector is a loud 400");
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["error"]["code"], "evidence_oversized");
        // Oversized item selector: same bound.
        let ids: Vec<u64> = (1..=300).collect();
        let resp = client
            .post(format!(
                "{base}/native/evidence/7/retrieve?session={}",
                row.id.raw()
            ))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"selector": "items", "ids": ids}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);
        // A bounded selector on owned evidence actually retrieves.
        let resp = client
            .post(format!(
                "{base}/native/evidence/7/retrieve?session={}",
                row.id.raw()
            ))
            .header("x-faktor-server-password", pw.as_str())
            .json(&serde_json::json!({"selector": "all"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["bytesBase64"], "aGVsbG8=");
        assert_eq!(body["byteLen"], 5);
        let _ = handle.shutdown.send(());
    }

    // ---------------------------------------------- native semantic (audit 83)

    #[tokio::test]
    async fn native_semantic_status_falls_back_without_provider() {
        // No registry configured is a 200 fallback shape, never a 500.
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let pw = deps.server_password.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);
        let resp = client
            .get(format!("{base}/native/semantic/status"))
            .header("x-faktor-server-password", pw.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["configured"], false);
        assert!(body["providers"].as_array().unwrap().is_empty());
        assert!(!body["fallback"]["id"].as_str().unwrap().is_empty());
        assert!(body["fallback"]["version"].is_u64());
        assert!(body["snapshotState"]["fallback"].is_boolean());
        // Capabilities mirror the same fallback-only registry.
        let resp = client
            .get(format!("{base}/native/semantic/capabilities"))
            .header("x-faktor-server-password", pw.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(body["providers"].as_array().unwrap().is_empty());
        assert!(body["union"]["operations"]["snapshot"].is_boolean());
        assert!(!body["fallback"]["capabilities"]
            .as_object()
            .unwrap()
            .is_empty());
        let _ = handle.shutdown.send(());
    }

    // ------------------------------------------------ split invariants

    #[test]
    fn native_modules_never_depend_on_the_compat_surface() {
        // Dependency direction (audits 81-83): the native layer never
        // imports the v7.5.6 compatibility DTOs or the compat module; the
        // compat layer may import native's shared glue.
        let sources: [(&str, &str); 11] = [
            ("native/mod.rs", include_str!("native/mod.rs")),
            ("native/prompt.rs", include_str!("native/prompt.rs")),
            ("native/session.rs", include_str!("native/session.rs")),
            ("native/task.rs", include_str!("native/task.rs")),
            ("native/agents.rs", include_str!("native/agents.rs")),
            ("native/evidence.rs", include_str!("native/evidence.rs")),
            (
                "native/verification.rs",
                include_str!("native/verification.rs"),
            ),
            ("native/terminal.rs", include_str!("native/terminal.rs")),
            ("native/usage.rs", include_str!("native/usage.rs")),
            ("native/semantic.rs", include_str!("native/semantic.rs")),
            ("native/models.rs", include_str!("native/models.rs")),
        ];
        for (name, src) in sources {
            for (idx, line) in src.lines().enumerate() {
                let code = line.trim_start();
                if code.starts_with("//") {
                    continue;
                }
                assert!(
                    !line.contains("crate::compat"),
                    "{name}:{} references the compat module: {line}",
                    idx + 1
                );
                assert!(
                    !line.contains("faktor_protocol::v756"),
                    "{name}:{} references v7.5.6 DTOs: {line}",
                    idx + 1
                );
            }
        }
        let compat = concat!(
            include_str!("compat/mod.rs"),
            include_str!("compat/sdk.rs"),
            include_str!("compat/v756.rs")
        );
        assert!(
            compat.contains("crate::native::"),
            "compat depends on native's shared glue"
        );
    }
}
