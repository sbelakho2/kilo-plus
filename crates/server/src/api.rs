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

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{sse::Event, IntoResponse, Response, Sse};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use futures_util::stream::Stream;
use tokio::sync::oneshot;
use tower_http::limit::RequestBodyLimitLayer;

use faktor_agent::AgentRuntime;
use faktor_core::capability::PermissionDecision;
use faktor_core::error::Error;
use faktor_core::id::SessionId;
use faktor_core::model::ModelCapabilities;
use faktor_core::state::AgentState;
use faktor_core::state::SessionLifecycle;
use faktor_protocol::error::ApiError;
use faktor_protocol::v756::*;
use faktor_protocol::v756::{
    mapper as wire_mapper, wire::AbortBody, wire::DiffStatus, wire::MessageSendRequest,
    wire::MessageSendResponse, wire::RevertBody, wire::RevertResponse, wire::SessionCreateRequest,
    wire::SessionCreateResponse, wire::SessionListResponse, wire::SessionSummarizeResponse,
    wire::SessionSummary, wire::SessionUpdateRequest, wire::SessionUpdateResponse,
    wire::SnapshotFileDiff, wire::WireMessageEntry, wire::WireMessageInfo, wire::WirePart,
};
use faktor_session::SessionManager;

use crate::auth::{check_bearer, check_password, AuthToken, ServerPassword};
use crate::global::GlobalEventBus;
use crate::permission::{ChannelPermissionRequester, PendingPermission};

const MAX_BODY_BYTES: usize = 10 * 1024 * 1024;
const MAX_CONFIG_BYTES: usize = 1024 * 1024;
const HEARTBEAT_SECS: u64 = 15;
const POLL_INTERVAL_MS: u64 = 100;

pub struct ServerDeps {
    pub session: Arc<SessionManager>,
    pub agent: Arc<AgentRuntime>,
    pub permissions: Arc<ChannelPermissionRequester>,
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
    pub fn new(
        session: Arc<SessionManager>,
        agent: Arc<AgentRuntime>,
        permissions: Arc<ChannelPermissionRequester>,
    ) -> Self {
        Self {
            session,
            agent,
            permissions,
            auth_token: AuthToken::generate(),
            server_password: ServerPassword::from_env(),
            directory: None,
            version: faktor_core::VERSION.to_string(),
            fs: None,
            snapshots: None,
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
        .route(
            "/native/session/{id}/terminal",
            get(native_session_terminal),
        )
        .route("/native/session/{id}/abort", post(native_session_abort))
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
struct AppState {
    deps: Arc<ServerDeps>,
    bus: Arc<GlobalEventBus>,
    config: Arc<std::sync::RwLock<serde_json::Value>>,
    /// Runtime server-password override (`auth.set`); `None` = the startup
    /// env password (`ServerDeps.server_password`) applies (`auth.remove`).
    auth: Arc<std::sync::RwLock<Option<ServerPassword>>>,
    /// Live PTYs (audit round 11): session-owned interactive terminals,
    /// Unix real implementation; other platforms refuse at creation.
    ptys: Arc<std::sync::Mutex<std::collections::HashMap<u64, faktor_pty::Pty>>>,
    next_pty_id: Arc<std::sync::atomic::AtomicU64>,
    /// Readiness flag (audit 55): set true at the END of serve() setup, after
    /// the store was opened/migrated/recovered by the caller (cli runs
    /// `agent.recover()` before serve) and every required runtime component
    /// is in place (`SessionManager` is a non-optional `Arc` in
    /// `ServerDeps`, so its presence is structural). `GET /native/ready`
    /// answers 200 `{ready:true}` once it is set; 503 `{ready:false}`
    /// before/without it (`ServerDeps.simulate_not_ready`).
    ready: Arc<std::sync::atomic::AtomicBool>,
}

// ------------------------------------------------------------------ handlers

async fn hello(State(state): State<AppState>) -> Response {
    Json(HelloResponse {
        ok: true,
        version: state.deps.version.clone(),
        protocol: faktor_core::PROTOCOL_V756.to_string(),
        auth_required: true,
        providers: state.deps.agent.deps().providers.ids(),
    })
    .into_response()
}

fn authed(headers: &HeaderMap, state: &AppState) -> Result<(), ApiError> {
    let authorization = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());
    let x_kilo = headers
        .get("x-faktor-server-password")
        .and_then(|v| v.to_str().ok());
    // The frozen v7.5.6 extension sends `Basic base64("kilo:"+password)` for
    // every request; the Faktor-native `x-faktor-server-password` header and the
    // legacy per-start token keep the old clients and tests working. The
    // effective password is the `auth.set` override when one is active,
    // else the startup env password (`auth.remove` returns to it).
    let auth = state.auth.read().expect("auth override poisoned");
    let password = auth.as_ref().unwrap_or(&state.deps.server_password);
    if password.check_authorization(authorization)
        || check_password(password, None, x_kilo)
        || check_bearer(&state.deps.auth_token, authorization)
    {
        Ok(())
    } else {
        Err(ApiError {
            code: "unauthorized",
            message: "missing or invalid server password".into(),
            http_status: 401,
            retryable: false,
        })
    }
}

/// `GET /global/health` — auth-required (the frozen v7.5.6 client
/// authenticates every request, this one included).
async fn health(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    Json(HealthResponse {
        ok: true,
        version: state.deps.version.clone(),
        protocol: faktor_core::PROTOCOL_V756.to_string(),
    })
    .into_response()
}

async fn create_session(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<CreateSessionRequest>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let workspace = req.workspace.clone().unwrap_or_else(|| ".".into());
    let ws = match state.deps.session.create_workspace(&workspace) {
        Ok(ws) => ws,
        Err(e) => return api_err(&e),
    };
    let title = req.title.clone().unwrap_or_else(|| "New session".into());
    match state
        .deps
        .session
        .create_session(ws, &title, &req.provider, &req.model)
    {
        Ok(row) => match row.row() {
            Ok(row_data) => Json(CreateSessionResponse {
                id: row_data.id.to_string(),
                title,
                created_ms: row_data.created_ms,
            })
            .into_response(),
            Err(e) => api_err(&e),
        },
        Err(e) => api_err(&e),
    }
}

async fn list_sessions(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    match state.deps.session.list_sessions(None) {
        Ok(rows) => Json(serde_json::json!({
            "sessions": rows.iter().map(|r| {
                let title = r.title().unwrap_or_default();
                let provider = r.provider().unwrap_or_default();
                let model = r.model().unwrap_or_default();
                let state = r.state().map(|s| s.label()).unwrap_or("unknown");
                serde_json::json!({
                    "id": r.id().to_string(),
                    "title": title,
                    "provider": provider,
                    "model": model,
                    "state": state,
                })
            }).collect::<Vec<_>>(),
        }))
        .into_response(),
        Err(e) => api_err(&e),
    }
}

fn parse_session_id(id: &str) -> Result<SessionId, ApiError> {
    let raw: u64 = id.parse().map_err(|_| ApiError {
        code: "malformed",
        message: format!("invalid session id {id:?}"),
        http_status: 400,
        retryable: false,
    })?;
    if raw == 0 {
        return Err(ApiError {
            code: "malformed",
            message: "session id cannot be 0".into(),
            http_status: 400,
            retryable: false,
        });
    }
    Ok(SessionId::new(raw))
}

async fn session_state(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let sid = match parse_session_id(&id) {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::from_u16(e.http_status).unwrap(),
                Json(e.to_json()),
            )
                .into_response()
        }
    };
    let handle = match state.deps.session.get_session(sid) {
        Ok(Some(h)) => h,
        Ok(None) => {
            let e = ApiError {
                code: "not_found",
                message: format!("session {sid}"),
                http_status: 404,
                retryable: false,
            };
            return (StatusCode::NOT_FOUND, Json(e.to_json())).into_response();
        }
        Err(e) => return api_err(&e),
    };
    match handle.session_state_view() {
        Ok(view) => Json(view).into_response(),
        Err(e) => api_err(&e),
    }
}

async fn messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Query(q): Query<MessagesQuery>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let sid = match parse_session_id(&id) {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::from_u16(e.http_status).unwrap(),
                Json(e.to_json()),
            )
                .into_response()
        }
    };
    let handle = match state.deps.session.get_session(sid) {
        Ok(Some(h)) => h,
        Ok(None) => {
            let e = ApiError {
                code: "not_found",
                message: format!("session {sid}"),
                http_status: 404,
                retryable: false,
            };
            return (StatusCode::NOT_FOUND, Json(e.to_json())).into_response();
        }
        Err(e) => return api_err(&e),
    };
    match handle.messages_page(q.before, q.limit) {
        Ok(page) => Json(page).into_response(),
        Err(e) => api_err(&e),
    }
}

/// Submit a prompt synchronously so the HTTP response carries the TRUE
/// queued state, then spawn the turn (or the queue runner) detached
/// (audit round 6). Returns the receipt's queued flag.
fn submit_and_run(
    agent: &std::sync::Arc<faktor_agent::AgentRuntime>,
    session: faktor_core::id::SessionId,
    prompt: &str,
    files: &[String],
    model: Option<String>,
) -> faktor_core::Result<faktor_session::PromptReceipt> {
    let receipt = agent.submit(session, prompt, files)?;
    let queued = receipt.queued;
    let agent2 = agent.clone();
    if queued {
        // The prompt durably queued behind the active logical turn; the
        // per-session runner delivers it after that turn completes.
        tokio::spawn(async move {
            agent2.run_session_queue(session).await;
        });
    } else {
        let handle = match agent2.deps().session.get_session(session) {
            Ok(Some(h)) => h,
            _ => return Ok(receipt),
        };
        let receipt2 = receipt.clone();
        tokio::spawn(async move {
            let _ = agent2.drive_receipt(&handle, receipt2, model).await;
        });
    }
    Ok(receipt)
}

async fn prompt(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(req): Json<PromptRequest>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    if req.prompt.trim().is_empty() {
        let e = ApiError {
            code: "malformed",
            message: "prompt must not be empty".into(),
            http_status: 400,
            retryable: false,
        };
        return (StatusCode::BAD_REQUEST, Json(e.to_json())).into_response();
    }
    let sid = match parse_session_id(&id) {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::from_u16(e.http_status).unwrap(),
                Json(e.to_json()),
            )
                .into_response()
        }
    };
    // Unknown sessions are 404, never a phantom 200 (audit round 8).
    match state.deps.session.get_session(sid) {
        Ok(Some(_)) => {}
        Ok(None) => {
            return wire_status(not_found(&format!("session {sid}")));
        }
        Err(e) => return api_err(&e),
    }
    let prompt_text = req.prompt.clone();
    let files = req.files.clone();
    // Synchronous submission so the response carries the TRUE queued state
    // and the REAL operation id (audit: op_id was hardcoded "turn").
    let receipt = match submit_and_run(&state.deps.agent, sid, &prompt_text, &files, None) {
        Ok(r) => r,
        Err(e) => return api_err(&e),
    };
    Json(PromptResponse {
        op_id: receipt.op_id.to_string(),
        accepted: true,
        queued: receipt.queued,
    })
    .into_response()
}

async fn abort(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let sid = match parse_session_id(&id) {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::from_u16(e.http_status).unwrap(),
                Json(e.to_json()),
            )
                .into_response()
        }
    };
    match state.deps.agent.abort(sid) {
        Ok(ops) => Json(AbortResponse {
            aborted: ops.iter().map(|o| o.to_string()).collect(),
        })
        .into_response(),
        Err(e) => api_err(&e),
    }
}

async fn resolve_permission(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(req): Json<PermissionDecisionRequest>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    match resolve_permission_body(&state.deps, &id, &req.decision) {
        Ok(()) => Json(PermissionDecisionResponse { ok: true }).into_response(),
        Err(e) => (
            StatusCode::from_u16(e.http_status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            Json(e.to_json()),
        )
            .into_response(),
    }
}

/// `POST /permission/reply` — SDK form of the same resolution.
async fn permission_reply(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<PermissionDecisionRequest>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    match resolve_permission_body(&state.deps, &req.permission_id, &req.decision) {
        Ok(()) => Json(PermissionDecisionResponse { ok: true }).into_response(),
        Err(e) => (
            StatusCode::from_u16(e.http_status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            Json(e.to_json()),
        )
            .into_response(),
    }
}

fn resolve_permission_body(
    deps: &ServerDeps,
    permission_id: &str,
    decision: &str,
) -> Result<(), ApiError> {
    let pid: i64 = match permission_id.parse() {
        Ok(p) if p > 0 => p,
        _ => {
            return Err(ApiError {
                code: "malformed",
                message: format!("invalid permission id {permission_id:?}"),
                http_status: 400,
                retryable: false,
            });
        }
    };
    let decision = match decision {
        "allow" => PermissionDecision::Allow,
        "deny" => PermissionDecision::Deny,
        other => {
            return Err(ApiError {
                code: "malformed",
                message: format!("invalid decision {other:?}"),
                http_status: 400,
                retryable: false,
            });
        }
    };
    if !deps.permissions.resolve(pid, decision) {
        return Err(ApiError {
            code: "conflict",
            message: format!("permission {pid} unknown or already resolved"),
            http_status: 409,
            retryable: false,
        });
    }
    Ok(())
}

/// `GET /permission/list?session_id=` — pending permission requests.
async fn permission_list(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<SdkSessionQuery>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let views = state.deps.permissions.pending_views();
    let permissions: Vec<PermissionListEntry> = views
        .iter()
        .filter(|v| {
            q.session_id
                .as_ref()
                .is_none_or(|sid| v.session_id.to_string() == *sid)
        })
        .map(|v| PermissionListEntry {
            id: v.id.to_string(),
            session_id: v.session_id.to_string(),
            capability: v.capability.clone(),
            detail: v.detail.clone(),
        })
        .collect();
    Json(PermissionListResponse { permissions }).into_response()
}

// ------------------------------------------------------------------ SDK handlers

/// `POST /session/prompt` — session_id in the body.
async fn sdk_prompt(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<SdkPromptRequest>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    if req.prompt.trim().is_empty() {
        let e = ApiError {
            code: "malformed",
            message: "prompt must not be empty".into(),
            http_status: 400,
            retryable: false,
        };
        return (StatusCode::BAD_REQUEST, Json(e.to_json())).into_response();
    }
    let sid = match parse_session_id(&req.session_id) {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::from_u16(e.http_status).unwrap(),
                Json(e.to_json()),
            )
                .into_response();
        }
    };
    match state.deps.session.get_session(sid) {
        Ok(Some(_)) => {}
        Ok(None) => {
            let e = ApiError {
                code: "not_found",
                message: format!("session {sid}"),
                http_status: 404,
                retryable: false,
            };
            return (StatusCode::NOT_FOUND, Json(e.to_json())).into_response();
        }
        Err(e) => return api_err(&e),
    }
    // Unknown sessions are 404, never a phantom 200 (audit round 8).
    match state.deps.session.get_session(sid) {
        Ok(Some(_)) => {}
        Ok(None) => {
            return wire_status(not_found(&format!("session {sid}")));
        }
        Err(e) => return api_err(&e),
    }
    let prompt_text = req.prompt.clone();
    let files = req.files.clone();
    // Synchronous submission so the response carries the TRUE queued state
    // and the REAL operation id (audit: op_id was hardcoded "turn").
    let receipt = match submit_and_run(&state.deps.agent, sid, &prompt_text, &files, None) {
        Ok(r) => r,
        Err(e) => return api_err(&e),
    };
    Json(PromptResponse {
        op_id: receipt.op_id.to_string(),
        accepted: true,
        queued: receipt.queued,
    })
    .into_response()
}

/// `POST /session/abort` — session_id in the body.
async fn sdk_abort(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<SdkAbortRequest>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let sid = match parse_session_id(&req.session_id) {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::from_u16(e.http_status).unwrap(),
                Json(e.to_json()),
            )
                .into_response();
        }
    };
    match state.deps.session.get_session(sid) {
        Ok(Some(_)) => {}
        Ok(None) => {
            let e = ApiError {
                code: "not_found",
                message: format!("session {sid}"),
                http_status: 404,
                retryable: false,
            };
            return (StatusCode::NOT_FOUND, Json(e.to_json())).into_response();
        }
        Err(e) => return api_err(&e),
    }
    // Targeted abort (audit round 8): the request op_id is honored — one
    // queued prompt can be killed without touching the active turn.
    let target = match &req.op_id {
        Some(raw) => match raw.parse::<u64>() {
            Ok(v) => Some(faktor_core::id::OpId::new(v)),
            Err(_) => {
                return wire_refused(&format!("invalid op_id {raw:?}"));
            }
        },
        None => None,
    };
    match state.deps.agent.abort_op(sid, target) {
        Ok(ops) => Json(AbortResponse {
            aborted: ops.iter().map(|o| o.to_string()).collect(),
        })
        .into_response(),
        Err(e) => api_err(&e),
    }
}

/// `GET /session/messages?session_id=&before=&limit=`
async fn sdk_messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<SdkMessagesQuery>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let sid = match parse_session_id(&q.session_id) {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::from_u16(e.http_status).unwrap(),
                Json(e.to_json()),
            )
                .into_response();
        }
    };
    let handle = match state.deps.session.get_session(sid) {
        Ok(Some(h)) => h,
        Ok(None) => {
            let e = ApiError {
                code: "not_found",
                message: format!("session {sid}"),
                http_status: 404,
                retryable: false,
            };
            return (StatusCode::NOT_FOUND, Json(e.to_json())).into_response();
        }
        Err(e) => return api_err(&e),
    };
    match handle.messages_page(q.before, q.limit) {
        Ok(page) => Json(page).into_response(),
        Err(e) => api_err(&e),
    }
}

/// `GET /session/state?session_id=`
async fn sdk_session_state(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<SdkStateQuery>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let sid = match parse_session_id(&q.session_id) {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::from_u16(e.http_status).unwrap(),
                Json(e.to_json()),
            )
                .into_response();
        }
    };
    let handle = match state.deps.session.get_session(sid) {
        Ok(Some(h)) => h,
        Ok(None) => {
            let e = ApiError {
                code: "not_found",
                message: format!("session {sid}"),
                http_status: 404,
                retryable: false,
            };
            return (StatusCode::NOT_FOUND, Json(e.to_json())).into_response();
        }
        Err(e) => return api_err(&e),
    };
    match handle.session_state_view() {
        Ok(view) => Json(view).into_response(),
        Err(e) => api_err(&e),
    }
}

// ------------------------------------------------- v7.5.6 wire surface (subset)
// The routes the frozen v7.5.6 extension actually calls. Path params are
// wire session ids (numeric strings): non-numeric → 400, unknown → 404.

/// The `x-faktor-directory` header value (the workspace root the extension
/// operates on). Bounded by the mapper.
fn directory_header(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("x-faktor-directory")
        .and_then(|v| v.to_str().ok())
}

/// The snake_case agent-state tag for machine-readable summaries.
fn agent_state_tag(s: AgentState) -> String {
    serde_json::to_string(&s)
        .unwrap_or_else(|_| "unknown".into())
        .trim_matches('"')
        .to_string()
}

/// `POST /session` — create a session from the wire request. The workspace
/// comes from the `x-faktor-directory` header, else `workspaceID`.
async fn wire_create_session(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<SessionCreateRequest>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let args = match wire_mapper::create_args(&req, directory_header(&headers)) {
        Ok(a) => a,
        Err(e) => return api_err(&e),
    };
    let ws = match state.deps.session.create_workspace(&args.workspace) {
        Ok(ws) => ws,
        Err(e) => return api_err(&e),
    };
    match state
        .deps
        .session
        .create_session(ws, &args.title, &args.provider, &args.model)
    {
        Ok(handle) => match handle.row() {
            Ok(row) => Json(SessionCreateResponse {
                session_id: row.id.to_string(),
                title: row.title,
                created_ms: row.created_ms,
            })
            .into_response(),
            Err(e) => api_err(&e),
        },
        Err(e) => api_err(&e),
    }
}

/// `GET /session` — the session list.
async fn wire_list_sessions(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let mut sessions = Vec::new();
    match state.deps.session.list_sessions(None) {
        Ok(handles) => {
            for h in handles {
                // A row that vanished mid-list is skipped, never fatal.
                if let Ok(row) = h.row() {
                    sessions.push(SessionSummary {
                        session_id: row.id.to_string(),
                        title: row.title,
                        state: agent_state_tag(row.state),
                        created_ms: row.created_ms,
                        updated_ms: row.updated_ms,
                    });
                }
            }
        }
        Err(e) => return api_err(&e),
    }
    Json(SessionListResponse { sessions }).into_response()
}

/// `GET /session/{sessionID}` — one session summary (404 when unknown).
async fn wire_session_summary(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(session_id): Path<String>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let sid = match parse_session_id(&session_id) {
        Ok(s) => s,
        Err(e) => return wire_status(e),
    };
    let handle = match state.deps.session.get_session(sid) {
        Ok(Some(h)) => h,
        Ok(None) => return wire_status(not_found(&format!("session {sid}"))),
        Err(e) => return api_err(&e),
    };
    match handle.row() {
        Ok(row) => Json(SessionSummary {
            session_id: row.id.to_string(),
            title: row.title,
            state: agent_state_tag(row.state),
            created_ms: row.created_ms,
            updated_ms: row.updated_ms,
        })
        .into_response(),
        Err(e) => api_err(&e),
    }
}

/// `GET /session/{sessionID}/state` — the wire-style state projection (UI
/// reconnects with GET /session/{id}/state and SSE resumes from the journal
/// sequence — spec §7). Same view as the legacy endpoint, wire error codes.
async fn wire_session_state(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(session_id): Path<String>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let sid = match parse_session_id(&session_id) {
        Ok(s) => s,
        Err(e) => return wire_status(e),
    };
    let handle = match state.deps.session.get_session(sid) {
        Ok(Some(h)) => h,
        Ok(None) => return wire_status(not_found(&format!("session {sid}"))),
        Err(e) => return api_err(&e),
    };
    match handle.session_state_view() {
        Ok(view) => Json(view).into_response(),
        Err(e) => api_err(&e),
    }
}

/// `POST /session/{sessionID}/message` — send one message and return the
/// frozen `{info: AssistantMessage, parts: Part[]}` shape for the durable
/// assistant message the accepted turn produced.
///
/// Turn semantics (documented, audit P0): the request runs the full logical
/// turn to its terminal machine state BEFORE responding — the turn (or the
/// queue runner for queued prompts) is spawned detached exactly like the
/// legacy prompt handler, and this handler waits on the durable state
/// machine (progress is visible over SSE meanwhile). The response `info` is
/// therefore built from the REAL durable assistant message row, and `parts`
/// from its parts — never a synthetic receipt. A prompt that durably QUEUED
/// behind an active logical turn has no assistant message yet: the handler
/// answers `202 Accepted` with the standard shape and empty `parts`
/// (`info.messageID` is empty — nothing is materialized until the queued
/// turn starts; the client polls the page / SSE). Queueing is signaled by
/// the HTTP status, never by a DTO field: the frozen `{info, parts}` type
/// rejects unknown fields (deny_unknown_fields), so a `queued` flag inside
/// the DTO would be protocol drift.
///
/// The per-message `model` override APPLIES: the provider must equal the
/// session's provider (else an honest 409), and the model id is used for
/// this turn only — the journaled session row keeps its configured model.
async fn wire_message_send(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(session_id): Path<String>,
    Json(req): Json<MessageSendRequest>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    if req.parts.is_empty() {
        let e = ApiError {
            code: "malformed",
            message: "message parts must not be empty".into(),
            http_status: 400,
            retryable: false,
        };
        return (StatusCode::BAD_REQUEST, Json(e.to_json())).into_response();
    }
    let args = match wire_mapper::prompt_args(&req) {
        Ok(a) => a,
        Err(e) => return api_err(&e),
    };
    if args.prompt.trim().is_empty() {
        let e = ApiError {
            code: "malformed",
            message: "message must carry a text or file part".into(),
            http_status: 400,
            retryable: false,
        };
        return (StatusCode::BAD_REQUEST, Json(e.to_json())).into_response();
    }
    let sid = match parse_session_id(&session_id) {
        Ok(s) => s,
        Err(e) => return wire_status(e),
    };
    let handle = match state.deps.session.get_session(sid) {
        Ok(Some(h)) => h,
        Ok(None) => return wire_status(not_found(&format!("session {sid}"))),
        Err(e) => return api_err(&e),
    };
    let row = match handle.row() {
        Ok(r) => r,
        Err(e) => return api_err(&e),
    };
    // The per-message override is a model id WITHIN the session's provider;
    // a provider mismatch is protocol drift (the frozen client never sends
    // one) and is refused honestly — nothing is spawned, nothing mutates.
    if req.model.provider_id != row.provider {
        return wire_refused("provider mismatch");
    }
    let provider_id = Some(row.provider.clone());
    let model_id = Some(req.model.model_id.clone());
    // The sequence the user message will occupy (its row is created inside
    // submit); every durable row after it belongs to this turn.
    let user_seq = match handle.proposed_message_seq() {
        Ok(seq) => seq,
        Err(e) => return api_err(&e),
    };
    // Synchronous submission so the response carries the TRUE queued state;
    // the turn (or queue runner) is spawned detached (spec §7 + audit r6).
    // The model override is per-message: the session row is untouched.
    let receipt = match submit_and_run(
        &state.deps.agent,
        sid,
        &args.prompt,
        &args.files,
        model_id.clone(),
    ) {
        Ok(r) => r,
        Err(e) => return api_err(&e),
    };
    if receipt.queued {
        // Queued: no durable assistant message exists yet (the queued
        // prompt's user message materializes only at admission). 202 marks
        // acceptance; the empty messageID documents "nothing yet".
        let resp = MessageSendResponse {
            info: WireMessageInfo {
                session_id: sid.to_string(),
                message_id: String::new(),
                role: "assistant".into(),
                created_ms: state.deps.session.now_ms(),
                provider_id,
                model_id,
            },
            parts: Vec::new(),
        };
        return (StatusCode::ACCEPTED, Json(resp)).into_response();
    }
    // Accepted: wait for the turn machine to leave the mid-turn states
    // (Preparing/…/UpdatingMemory). The runtime always lands the machine in
    // ReadyForNextTurn (or Completed/Cancelled/Failed*/NeedsUserInput — an
    // error journals FailedRecoverable, never a stuck machine), then this
    // handler projects the NEWEST durable assistant row of this turn.
    loop {
        match handle.state() {
            Ok(s) if !turn_machine_busy(s) => break,
            Ok(_) => {}
            Err(e) => return api_err(&e),
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    // Walk the newest-first pages until an assistant row newer than the
    // user prompt is found (a turn that never produced content yields none).
    let mut cursor: Option<i64> = None;
    let mut found: Option<WireMessageEntry> = None;
    'pages: loop {
        let page = match handle.messages_page(cursor, 100) {
            Ok(p) => p,
            Err(e) => return api_err(&e),
        };
        for m in &page.messages {
            if m.seq > user_seq && m.role == "assistant" {
                match wire_mapper::internal_message_to_wire_entry(m) {
                    Ok(mut e) => {
                        e.info.provider_id = provider_id.clone();
                        e.info.model_id = model_id.clone();
                        found = Some(e);
                    }
                    Err(e) => return api_err(&e),
                }
                break 'pages;
            }
        }
        match page.next_before {
            Some(b) => cursor = Some(b),
            None => break,
        }
    }
    match found {
        Some(entry) => Json(MessageSendResponse {
            info: entry.info,
            parts: entry.parts,
        })
        .into_response(),
        // The turn ended without any assistant content (e.g. the provider
        // failed before the first chunk): honest failure, never a fake
        // assistant message.
        None => (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({
                "ok": false,
                "message": "turn ended without an assistant reply",
            })),
        )
            .into_response(),
    }
}

/// `GET /session/{sessionID}/message?before=&limit=` — newest-first paging
/// over the frozen page shape: a bare ARRAY of `{info: Message, parts:
/// Part[]}` (the old envelope is gone). `before` is the internal message
/// sequence cursor (the wire omits `seq`; documented); when an older page
/// exists the response carries `x-has-more: true` (the DTO has no room for
/// paging fields — the frozen entry type rejects unknown fields).
#[derive(serde::Deserialize)]
struct WireMessagesQuery {
    before: Option<i64>,
    #[serde(default = "wire_default_limit")]
    limit: i64,
}

fn wire_default_limit() -> i64 {
    100
}

/// One durable part row → the wire part union. Mirrors the session layer's
/// own projection; unknown/corrupt kinds fail the page loudly.
fn wire_part_from_row(kind: &str, data: &serde_json::Value) -> Result<WirePart, Error> {
    let s = |key: &str| -> Result<String, Error> {
        data.get(key)
            .and_then(|v| v.as_str())
            .map(String::from)
            .ok_or_else(|| Error::malformed(format!("part row kind {kind:?} is missing `{key}`")))
    };
    Ok(match kind {
        "text" => WirePart::Text { text: s("text")? },
        "reasoning" => WirePart::Reasoning { text: s("text")? },
        "summary" => WirePart::Subtask {
            label: Some(s("text")?),
            note: None,
        },
        "tool_call" => WirePart::Tool {
            call_id: s("tool_call_id")?,
            name: s("name")?,
            input: data
                .get("input")
                .cloned()
                .unwrap_or(serde_json::Value::Null),
            state: Some(s("state")?),
            output: None,
        },
        "tool_result" => WirePart::Tool {
            call_id: s("tool_call_id")?,
            name: "unknown".into(),
            input: serde_json::Value::Null,
            state: Some("completed".into()),
            output: Some(serde_json::json!({
                "excerpt": s("excerpt")?,
                "exit_code": data.get("exit_code").and_then(|v| if v.is_null() { None } else { v.as_i64() }),
                "artifact": data.get("artifact").and_then(|v| v.as_str()),
                "slice_hint": data.get("slice_hint").and_then(|v| v.as_str()),
            })),
        },
        other => {
            return Err(Error::malformed(format!(
                "unknown part kind {other:?} in message row"
            )))
        }
    })
}

async fn wire_messages_page(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(session_id): Path<String>,
    Query(q): Query<WireMessagesQuery>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let sid = match parse_session_id(&session_id) {
        Ok(s) => s,
        Err(e) => return wire_status(e),
    };
    let handle = match state.deps.session.get_session(sid) {
        Ok(Some(h)) => h,
        Ok(None) => return wire_status(not_found(&format!("session {sid}"))),
        Err(e) => return api_err(&e),
    };
    let (provider_id, model_id) = match handle.row() {
        Ok(row) => (Some(row.provider), Some(row.model)),
        Err(_) => (None, None),
    };
    let store = state.deps.session.store();
    // One page + 1 probe row: paging never loads more than one page.
    let limit = q.limit.clamp(1, 100);
    let mut rows = match store.messages_before(sid, q.before, limit as u64 + 1) {
        Ok(rows) => rows,
        Err(e) => return store_err(&e),
    };
    let has_more = rows.len() as i64 > limit;
    if has_more {
        rows.truncate(limit as usize);
    }
    let mut entries = Vec::with_capacity(rows.len());
    for row in rows {
        let mut parts = Vec::new();
        match store.parts_of(row.id) {
            Ok(part_rows) => {
                for p in &part_rows {
                    match wire_part_from_row(&p.kind, &p.data) {
                        Ok(w) => parts.push(w),
                        // A corrupt part row fails the page loudly (the
                        // legacy route has the same rule): never silently
                        // drop content.
                        Err(e) => return api_err(&e),
                    }
                }
            }
            Err(e) => return store_err(&e),
        }
        // Prompt messages themselves appear WITH their parts: user rows are
        // stored as {text, files} message data with no part rows, so the
        // text is projected as the wire text part here.
        if parts.is_empty() && row.role == "user" {
            if let Some(text) = row.data.get("text").and_then(|v| v.as_str()) {
                if !text.is_empty() {
                    parts.push(WirePart::Text {
                        text: text.to_string(),
                    });
                }
            }
        }
        entries.push(WireMessageEntry {
            info: WireMessageInfo {
                session_id: row.session_id.to_string(),
                message_id: row.seq.to_string(),
                role: row.role.clone(),
                created_ms: row.created_ms,
                provider_id: provider_id.clone(),
                model_id: model_id.clone(),
            },
            parts,
        });
    }
    let mut resp = Json(entries).into_response();
    // Paging signal lives in a header (the frozen entry DTO is strict).
    resp.headers_mut().insert(
        "x-has-more",
        HeaderValue::from_static(if has_more { "true" } else { "false" }),
    );
    resp
}

/// `POST /session/{sessionID}/abort` — body `{ messageID? }`.
async fn wire_abort(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(session_id): Path<String>,
    body: Option<Json<AbortBody>>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    // The body is optional; the optional message_id is not resolvable to an
    // operation in this runtime, so the abort targets the whole session (the
    // legacy handler has the same semantics).
    let _ = body;
    let sid = match parse_session_id(&session_id) {
        Ok(s) => s,
        Err(e) => return wire_status(e),
    };
    match state.deps.session.get_session(sid) {
        Ok(None) => return wire_status(not_found(&format!("session {sid}"))),
        Err(e) => return api_err(&e),
        Ok(Some(_)) => {}
    }
    match state.deps.agent.abort(sid) {
        Ok(ops) => Json(AbortResponse {
            aborted: ops.iter().map(|o| o.to_string()).collect(),
        })
        .into_response(),
        Err(e) => api_err(&e),
    }
}

/// The session row for a wire session id; `Ok(None)` when unknown.
fn wire_session_row(
    state: &AppState,
    sid: SessionId,
) -> Result<Option<faktor_store::SessionRow>, Box<Response>> {
    match state.deps.session.get_session(sid) {
        Ok(Some(handle)) => handle.row().map(Some).map_err(|e| Box::new(api_err(&e))),
        Ok(None) => Ok(None),
        Err(e) => Err(Box::new(api_err(&e))),
    }
}

fn store_err(e: &faktor_store::StoreError) -> Response {
    api_err(&Error::new(
        faktor_core::error::ErrorKind::Store,
        format!("store: {e}"),
    ))
}

/// `GET /session/{sessionID}/diff?message=&file=&full=1` — the frozen
/// `SnapshotFileDiff[]` projection (a bare array, newest checkpoint first).
/// Each recorded file-change row becomes one entry with the status derived
/// from its recorded before→after existence/content transition.
///
/// Filters (documented, audit P0):
/// - `?message=<seq>` limits the projection to ONE checkpoint: the newest
///   checkpoint recorded at-or-before that message's `created_ms` (the same
///   selection revert uses). Unknown message → honest 409.
/// - `?file=<rel path>` keeps only the entries whose recorded path equals
///   the given relative path (exact match; no filesystem access happens).
/// - `?full=1` adds the full unified diff text (before/after content
///   resolved through the CAS) to each entry. Without it entries carry
///   path+status only.
async fn wire_diff(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(session_id): Path<String>,
    Query(q): Query<WireDiffQuery>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let sid = match parse_session_id(&session_id) {
        Ok(s) => s,
        Err(e) => return wire_status(e),
    };
    let handle = match state.deps.session.get_session(sid) {
        Ok(Some(h)) => h,
        Ok(None) => return wire_status(not_found(&format!("session {sid}"))),
        Err(e) => return api_err(&e),
    };
    let _ = match handle.row() {
        Ok(r) => r,
        Err(e) => return api_err(&e),
    };
    let store = state.deps.session.store();
    let mut rows = match store.checkpoints_of(sid) {
        Ok(rows) => rows, // ascending by sequence
        Err(e) => return store_err(&e),
    };
    if let Some(raw) = q.message.as_deref().filter(|m| !m.is_empty()) {
        // The message identity is the durable sequence (same surface as
        // revert). `message_created_ms` is the authoritative lookup.
        let seq: i64 = match raw.parse() {
            Ok(s) if s > 0 => s,
            _ => {
                return wire_refused(&format!("diff: malformed message {raw:?}"));
            }
        };
        let message_ms = match store.message_created_ms(sid, seq) {
            Ok(ms) => ms,
            Err(e) => return store_err(&e),
        };
        let Some(message_ms) = message_ms else {
            return wire_refused(&format!("diff: unknown message id {seq}"));
        };
        // Newest checkpoint recorded at-or-before the message: one
        // checkpoint = the rows of that checkpoint sequence.
        let latest_sequence = match rows
            .iter()
            .filter(|c| c.created_ms <= message_ms)
            .max_by_key(|c| c.sequence)
        {
            Some(c) => c.sequence,
            None => {
                rows.clear();
                return Json(Vec::<SnapshotFileDiff>::new()).into_response();
            }
        };
        rows.retain(|c| c.sequence == latest_sequence);
    }
    if let Some(file) = q.file.as_deref().filter(|f| !f.is_empty()) {
        rows.retain(|c| c.path == file);
    }
    let cas = state.deps.session.cas();
    let mut entries = Vec::with_capacity(rows.len());
    // Full content needs the CAS blobs of both sides.
    for row in rows.into_iter().rev() {
        let status = checkpoint_diff_status(&row);
        let diff = if q.full.as_deref() == Some("1") {
            let before_bytes = if row.before_exists {
                match diff_cas_bytes(&cas, &row.before_hash) {
                    Ok(b) => b,
                    Err(resp) => return resp,
                }
            } else {
                Vec::new()
            };
            let after_bytes = if row.after_exists {
                // Pre-after-blob rows (hashes only) cannot produce content:
                // refused honestly, exactly like the snapshot diff_latest.
                let Some(after_cas_raw) = row.after_cas_hash.as_deref() else {
                    return wire_refused(&format!(
                        "diff unavailable: after-content missing for checkpoint {} (recorded before after-blob storage)",
                        row.id
                    ));
                };
                match diff_cas_bytes(&cas, after_cas_raw) {
                    Ok(b) => b,
                    Err(resp) => return resp,
                }
            } else {
                Vec::new()
            };
            Some(
                faktor_snapshot::diff_lines(&before_bytes, &after_bytes)
                    .iter()
                    .map(faktor_snapshot::DiffLine::render)
                    .collect::<Vec<_>>()
                    .join("\n"),
            )
        } else {
            None
        };
        entries.push(SnapshotFileDiff {
            path: row.path,
            status,
            diff,
        });
    }
    Json(entries).into_response()
}

/// The query parameters of `GET /session/{sessionID}/diff`.
#[derive(serde::Deserialize)]
struct WireDiffQuery {
    /// Message sequence (or id — identical on single-session stores)
    /// limiting the projection to one checkpoint.
    message: Option<String>,
    /// Exact relative path filter.
    file: Option<String>,
    /// `1` = include the full unified diff content per entry.
    full: Option<String>,
}

/// The frozen diff status of one checkpoint row, derived from the recorded
/// before→after transition (exactly like `ChangeStatus::from_transition`;
/// degenerate equal-state rows project as modified).
fn checkpoint_diff_status(row: &faktor_store::CheckpointRow) -> DiffStatus {
    match (row.before_exists, row.after_exists) {
        (false, true) => DiffStatus::Added,
        (true, false) => DiffStatus::Deleted,
        (true, true) | (false, false) => DiffStatus::Modified,
    }
}

/// Resolve a stored hex FileHash to its CAS bytes (the diff full-content
/// path). Missing/corrupt blobs are an honest refusal, never a fake diff.
#[allow(clippy::result_large_err)]
fn diff_cas_bytes(cas: &Arc<faktor_cas::Cas>, hex: &str) -> Result<Vec<u8>, Response> {
    let hash = match faktor_core::hash::FileHash::from_hex(hex) {
        Some(h) => h,
        None => {
            return Err(wire_refused(&format!(
                "diff unavailable: corrupt stored hash {hex:?}"
            )))
        }
    };
    match cas.get(hash) {
        Ok(bytes) => Ok(bytes),
        Err(e) => Err(wire_refused(&format!(
            "diff unavailable: content missing from the CAS ({e})"
        ))),
    }
}

/// The newest checkpoint row of `session` recorded at or before `message_ms`
/// (the revert/unrevert target). `None` when nothing qualifies.
fn checkpoint_before(
    store: &faktor_store::Store,
    session: SessionId,
    message_ms: i64,
) -> Result<Option<faktor_store::CheckpointRow>, Box<Response>> {
    let rows = match store.checkpoints_of(session) {
        Ok(rows) => rows,
        Err(e) => return Err(Box::new(store_err(&e))),
    };
    Ok(rows
        .into_iter()
        .filter(|c| c.created_ms <= message_ms)
        .max_by_key(|c| c.sequence))
}

/// The workspace handle + snapshot identity the wire snapshot ops run on.
fn open_snapshot_target(
    state: &AppState,
    workspace_id: faktor_core::WorkspaceId,
) -> Result<(faktor_fs::WorkspaceHandle, faktor_core::WorkspaceIdentity), Box<Response>> {
    let (Some(fs), Some(_)) = (&state.deps.fs, &state.deps.snapshots) else {
        return Err(Box::new(wire_refused("snapshots unavailable")));
    };
    let store = state.deps.session.store();
    let Some(root) = (match store.workspace_root(workspace_id) {
        Ok(r) => r,
        Err(e) => return Err(Box::new(store_err(&e))),
    }) else {
        return Err(Box::new(wire_refused(
            "snapshots unavailable: workspace root unknown",
        )));
    };
    let handle = match fs.open(workspace_id, std::path::PathBuf::from(&root)) {
        Ok(h) => h,
        Err(_) => {
            return Err(Box::new(wire_refused(
                "snapshots unavailable: workspace not openable",
            )))
        }
    };
    let identity = faktor_core::WorkspaceIdentity::new(
        workspace_id,
        faktor_core::WorktreeId::new(1),
        faktor_core::TaskId::new(1),
    );
    Ok((handle, identity))
}

/// `POST /session/{sessionID}/revert` — roll the session back to the latest
/// checkpoint recorded at or before the message id: the pre-edit content is
/// written back atomically, verified against the recorded hash. Independent
/// user edits are never clobbered (409 conflict).
async fn wire_revert(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(session_id): Path<String>,
    Json(req): Json<RevertBody>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let message_seq = match wire_mapper::wire_id_to_u64(&req.message_id) {
        Ok(s) => s as i64,
        Err(e) => return api_err(&e),
    };
    let sid = match parse_session_id(&session_id) {
        Ok(s) => s,
        Err(e) => return wire_status(e),
    };
    let Some(row) = (match wire_session_row(&state, sid) {
        Ok(r) => r,
        Err(resp) => return *resp,
    }) else {
        return wire_status(not_found(&format!("session {sid}")));
    };
    if state.deps.snapshots.is_none() {
        // Not wired: the honest stub behavior, never a silent success.
        return wire_refused("revert unavailable: snapshots unavailable");
    }
    let store = state.deps.session.store();
    let Some(message_ms) = (match store.message_created_ms(sid, message_seq) {
        Ok(ms) => ms,
        Err(e) => return store_err(&e),
    }) else {
        return wire_refused(&format!(
            "revert unavailable: unknown message id {message_seq}"
        ));
    };
    let Some(latest) = (match checkpoint_before(&store, sid, message_ms) {
        Ok(c) => c,
        Err(resp) => return *resp,
    }) else {
        return wire_refused(&format!(
            "revert unavailable: no checkpoint before message {message_seq}"
        ));
    };
    let (handle, identity) = match open_snapshot_target(&state, row.workspace_id) {
        Ok(pair) => pair,
        Err(resp) => return *resp,
    };
    let snapshots = state.deps.snapshots.as_ref().unwrap();
    match snapshots.rollback(&handle, &identity, sid, latest.id) {
        Ok(faktor_snapshot::RollbackOutcome::Restored { path, hash }) => Json(serde_json::json!({
            "ok": true,
            // hash is null when the rollback DELETED the file (the before
            // state was missing).
            "restored": [{"path": path, "hash": hash.map(|h| h.to_hex())}],
        }))
        .into_response(),
        Ok(faktor_snapshot::RollbackOutcome::Conflict { path, .. }) => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "ok": false,
                "message": "conflict: file changed independently",
                "conflict": {"path": path},
            })),
        )
            .into_response(),
        Err(e) => wire_refused(&format!("revert unavailable: {e}")),
    }
}

/// `POST /session/{sessionID}/unrevert` — redo: restore the checkpoint's
/// AFTER state (the mirror of revert). Same conflict rules: only rewrites
/// when the current content still matches the state revert left behind.
async fn wire_unrevert(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(session_id): Path<String>,
    Json(req): Json<RevertBody>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let message_seq = match wire_mapper::wire_id_to_u64(&req.message_id) {
        Ok(s) => s as i64,
        Err(e) => return api_err(&e),
    };
    let sid = match parse_session_id(&session_id) {
        Ok(s) => s,
        Err(e) => return wire_status(e),
    };
    let Some(row) = (match wire_session_row(&state, sid) {
        Ok(r) => r,
        Err(resp) => return *resp,
    }) else {
        return wire_status(not_found(&format!("session {sid}")));
    };
    if state.deps.snapshots.is_none() {
        return wire_refused("unrevert unavailable: snapshots unavailable");
    }
    let store = state.deps.session.store();
    let Some(message_ms) = (match store.message_created_ms(sid, message_seq) {
        Ok(ms) => ms,
        Err(e) => return store_err(&e),
    }) else {
        return wire_refused(&format!(
            "unrevert unavailable: unknown message id {message_seq}"
        ));
    };
    let Some(latest) = (match checkpoint_before(&store, sid, message_ms) {
        Ok(c) => c,
        Err(resp) => return *resp,
    }) else {
        return wire_refused(&format!(
            "unrevert unavailable: no checkpoint before message {message_seq}"
        ));
    };
    let (handle, identity) = match open_snapshot_target(&state, row.workspace_id) {
        Ok(pair) => pair,
        Err(resp) => return *resp,
    };
    let snapshots = state.deps.snapshots.as_ref().unwrap();
    match snapshots.redo(&handle, &identity, sid, latest.id) {
        Ok(faktor_snapshot::RollbackOutcome::Restored { path, hash }) => Json(serde_json::json!({
            "ok": true,
            // hash is null when the unrevert DELETED the file (the after
            // state was missing).
            "restored": [{"path": path, "hash": hash.map(|h| h.to_hex())}],
        }))
        .into_response(),
        Ok(faktor_snapshot::RollbackOutcome::Conflict { path, .. }) => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "ok": false,
                "message": "conflict: file changed independently",
                "conflict": {"path": path},
            })),
        )
            .into_response(),
        Err(e) => wire_refused(&format!("unrevert unavailable: {e}")),
    }
}

// ------------------------------------------------ wire session lifecycle ops
// The remaining frozen session operations. Fork/summarize/delete/deleteMessage
// all do REAL work through the daemon; delete and deleteMessage refuse loudly
// when the runtime cannot honor them (mid-turn, tool-result dependencies, or
// durable-row removal the store does not expose in this workspace slice).

/// `GET /session/status?session_id=` — the SDK-style state projection (the
/// alias of `GET /session/state?session_id=` the frozen client also calls).
async fn wire_session_status_query(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<SdkStateQuery>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let sid = match parse_session_id(&q.session_id) {
        Ok(s) => s,
        Err(e) => return wire_status(e),
    };
    let handle = match state.deps.session.get_session(sid) {
        Ok(Some(h)) => h,
        Ok(None) => return wire_status(not_found(&format!("session {sid}"))),
        Err(e) => return api_err(&e),
    };
    match handle.session_state_view() {
        Ok(view) => Json(view).into_response(),
        Err(e) => api_err(&e),
    }
}

/// `POST /session/{sessionID}/fork` — create a NEW session that durably
/// copies the source's message history (rows + parts, in order), with the
/// title `<orig> (fork)` and the same workspace/provider/model.
async fn wire_session_fork(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(session_id): Path<String>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let sid = match parse_session_id(&session_id) {
        Ok(s) => s,
        Err(e) => return wire_status(e),
    };
    let source = match state.deps.session.get_session(sid) {
        Ok(Some(h)) => h,
        Ok(None) => return wire_status(not_found(&format!("session {sid}"))),
        Err(e) => return api_err(&e),
    };
    let row = match source.row() {
        Ok(r) => r,
        Err(e) => return api_err(&e),
    };
    let title = format!("{} (fork)", row.title);
    match state.deps.session.fork_session(sid, &title) {
        Ok(fork) => match fork.row() {
            Ok(fork_row) => Json(SessionCreateResponse {
                session_id: fork_row.id.to_string(),
                title: fork_row.title,
                created_ms: fork_row.created_ms,
            })
            .into_response(),
            Err(e) => api_err(&e),
        },
        Err(e) => api_err(&e),
    }
}

/// Bounded digest: at most this many message texts feed a summarize digest.
const SUMMARIZE_LAST_MESSAGES: usize = 3;
/// Hard bound on the returned summary text.
const SUMMARIZE_MAX_BYTES: usize = 4096;

/// `POST /session/{sessionID}/summarize` — a bounded summary from the
/// session title + the newest messages' text (nothing heavy is stored or
/// journaled). Text is truncated to [`SUMMARIZE_MAX_BYTES`].
async fn wire_session_summarize(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(session_id): Path<String>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let sid = match parse_session_id(&session_id) {
        Ok(s) => s,
        Err(e) => return wire_status(e),
    };
    let handle = match state.deps.session.get_session(sid) {
        Ok(Some(h)) => h,
        Ok(None) => return wire_status(not_found(&format!("session {sid}"))),
        Err(e) => return api_err(&e),
    };
    let row = match handle.row() {
        Ok(r) => r,
        Err(e) => return api_err(&e),
    };
    let store = state.deps.session.store();
    let rows = match store.messages_before(sid, None, SUMMARIZE_LAST_MESSAGES as u64) {
        Ok(rows) => rows,
        Err(e) => return store_err(&e),
    };
    // Newest first: role + text digest. User text lives in the message data;
    // assistant text in the text-part rows.
    let mut digest = String::new();
    for m in rows {
        let mut text = m
            .data
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if let Ok(parts) = store.parts_of(m.id) {
            let mut part_texts = Vec::new();
            for p in parts {
                if p.kind == "text" {
                    if let Some(t) = p.data.get("text").and_then(|v| v.as_str()) {
                        part_texts.push(t.to_string());
                    }
                }
            }
            if !part_texts.is_empty() {
                text = part_texts.join(" ");
            }
        }
        if text.trim().is_empty() {
            continue;
        }
        let line = format!("{}: {}\n", m.role, text.trim());
        push_bounded(&mut digest, &line, SUMMARIZE_MAX_BYTES);
    }
    if digest.is_empty() {
        digest = "No messages yet.".into();
    }
    Json(SessionSummarizeResponse {
        session_id: sid.to_string(),
        title: row.title,
        summary: digest,
    })
    .into_response()
}

/// Append `s`, never exceeding `max` bytes without splitting a char.
fn push_bounded(out: &mut String, s: &str, max: usize) {
    if out.len() >= max {
        return;
    }
    let room = max - out.len();
    let take = s.len().min(room);
    let mut end = take;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    out.push_str(&s[..end]);
}

/// `POST /session/{sessionID}` — the frozen `session.update` operation
/// (title/model/provider update). Title is the one durable session-row
/// field the daemon owns: the update strips control characters, bounds the
/// result to 1..=200 chars, and persists through the session layer
/// (store row + bumped `updated_ms`). Unknown sessions are honest 404s;
/// hostile titles (empty after stripping, oversized, protocol drift fields)
/// refuse before any write.
async fn wire_session_update(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(session_id): Path<String>,
    Json(req): Json<SessionUpdateRequest>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let sid = match parse_session_id(&session_id) {
        Ok(s) => s,
        Err(e) => return wire_status(e),
    };
    let handle = match state.deps.session.get_session(sid) {
        Ok(Some(h)) => h,
        Ok(None) => return wire_status(not_found(&format!("session {sid}"))),
        Err(e) => return api_err(&e),
    };
    let Some(title) = req.title else {
        return wire_status(ApiError {
            code: "malformed",
            message: "session.update requires a title".into(),
            http_status: 400,
            retryable: false,
        });
    };
    match handle.update_session_title(&title) {
        Ok(()) => match handle.row() {
            Ok(row) => Json(SessionUpdateResponse {
                session_id: sid.to_string(),
                title: row.title,
                updated_ms: row.updated_ms,
            })
            .into_response(),
            Err(e) => api_err(&e),
        },
        Err(e) => api_err(&e),
    }
}

/// `DELETE /session/{sessionID}` — delete a session: refused while the
/// session is mid-turn (active turn record or active machine state);
/// otherwise the session is durably ended (`SessionEnded` journal event +
/// lifecycle Closed), lingering queued prompts are cancelled and the
/// per-session in-process registries are closed. The durable row is kept
/// (the Closed tombstone reads as `completed`; a store-level row-drop API
/// does not exist in this slice of the workspace).
async fn wire_session_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(session_id): Path<String>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let sid = match parse_session_id(&session_id) {
        Ok(s) => s,
        Err(e) => return wire_status(e),
    };
    match state.deps.session.delete_session(sid) {
        Ok(()) => Json(OkResponse { ok: true }).into_response(),
        Err(e) => {
            if matches!(
                e.kind,
                faktor_core::error::ErrorKind::Conflict
                    | faktor_core::error::ErrorKind::InvalidState { .. }
            ) {
                wire_refused(&e.message)
            } else {
                api_err(&e)
            }
        }
    }
}

/// `DELETE /session/{sessionID}/message/{messageID}` — delete ONE message
/// row and its parts durably (P1 "deleteMessage gaps"). Removal is refused
/// when the message has tool-result dependencies (a tool_result part on the
/// message, or a tool_call part a tool_result elsewhere references), refused
/// while it is the active turn's in-flight newest message, and unknown
/// messages are honest 404s. Otherwise the session layer removes the rows
/// in ONE store transaction; message sequences stay stable (paging skips
/// the hole, never renumbers). The journal is intentionally untouched — it
/// is the durable log of what happened.
async fn wire_message_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((session_id, message_id)): Path<(String, String)>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let sid = match parse_session_id(&session_id) {
        Ok(s) => s,
        Err(e) => return wire_status(e),
    };
    let seq: i64 = match message_id.parse() {
        Ok(s) if s > 0 => s,
        _ => {
            return wire_refused(&format!(
                "deleteMessage: malformed message id {message_id:?}"
            ));
        }
    };
    // The session must exist (the store is reached below); the message
    // identity is the durable sequence (same surface as revert/diff).
    let handle = match state.deps.session.get_session(sid) {
        Ok(Some(h)) => h,
        Ok(None) => return wire_status(not_found(&format!("session {sid}"))),
        Err(e) => return api_err(&e),
    };
    // The session layer owns the checks (existence, in-flight turn,
    // tool-result dependencies) and the durable one-transaction removal.
    match handle.delete_message(seq) {
        Ok(()) => Json(OkResponse { ok: true }).into_response(),
        Err(e) => match e.kind {
            faktor_core::error::ErrorKind::NotFound => {
                wire_status(not_found(&format!("message {seq} of session {sid}")))
            }
            faktor_core::error::ErrorKind::Conflict => wire_refused(&e.message),
            _ => api_err(&e),
        },
    }
}

// ---------------------------------------------------------- pty (unsupported)
// The daemon supervises non-interactive child processes (ProcessSupervisor)
// but exposes NO PTY abstraction: no pty handle can be created or read
// incrementally through it, so create/update/remove are REJECTED with a
// documented code — never a fake success and never a hang.

/// `POST /pty/create` — spawn a session-scoped interactive terminal.
/// Body: {command, args?, cwd?, rows?, cols?}. Returns {pty_id, pid}.
/// Non-Unix platforms refuse honestly (ConPTY/Job Objects are the declared
/// platform blocker).
async fn pty_create(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Option<Json<serde_json::Value>>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let Some(Json(body)) = body else {
        return wire_refused("pty/create requires a body");
    };
    let command = match body.get("command").and_then(|c| c.as_str()) {
        Some(c) if !c.is_empty() && c.len() <= 4096 => c.to_string(),
        _ => return wire_refused("pty/create requires a non-empty command"),
    };
    let args: Vec<String> = body
        .get("args")
        .and_then(|a| a.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect()
        })
        .unwrap_or_default();
    if args.iter().any(|a| a.len() > 4096) {
        return wire_refused("pty/create args are oversized");
    }
    let rows = body.get("rows").and_then(|r| r.as_u64()).unwrap_or(24);
    let cols = body.get("cols").and_then(|c| c.as_u64()).unwrap_or(80);
    let rows = u16::try_from(rows).unwrap_or(24).max(1);
    let cols = u16::try_from(cols).unwrap_or(80).max(1);
    let cfg = faktor_pty::PtyConfig {
        command,
        args,
        cwd: body
            .get("cwd")
            .and_then(|c| c.as_str())
            .map(|s| s.to_string()),
        env: vec![],
        rows,
        cols,
    };
    // Spawning is quick (non-blocking master) but do it off the async
    // thread to be safe with process setup.
    let pty = match tokio::task::spawn_blocking(move || faktor_pty::Pty::spawn(&cfg)).await {
        Ok(Ok(p)) => p,
        Ok(Err(e)) => {
            return (StatusCode::BAD_REQUEST, Json(api_error_json(&e))).into_response();
        }
        Err(_) => return wire_refused("pty spawn task failed"),
    };
    let id = state
        .next_pty_id
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let pid = pty.pid();
    state.ptys.lock().unwrap().insert(id, pty);
    Json(serde_json::json!({ "ok": true, "pty_id": id.to_string(), "pid": pid })).into_response()
}

/// `POST /pty/update` — write input and/or resize. Body: {pty_id,
/// data?, rows?, cols?}. A pty that no longer exists is a loud 404.
async fn pty_update(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Option<Json<serde_json::Value>>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let Some(Json(body)) = body else {
        return wire_refused("pty/update requires a body");
    };
    let id = match body
        .get("pty_id")
        .and_then(|i| i.as_str())
        .and_then(|s| s.parse::<u64>().ok())
    {
        Some(id) => id,
        None => return wire_refused("pty/update requires pty_id"),
    };
    let ptys = state.ptys.lock().unwrap();
    let pty = match ptys.get(&id) {
        Some(p) => p,
        None => return wire_status(not_found(&format!("pty {id}"))),
    };
    if let Some(data) = body.get("data").and_then(|d| d.as_str()) {
        if let Err(e) = pty.write_all(data.as_bytes()) {
            return api_err(&e);
        }
    }
    if let (Some(r), Some(c)) = (
        body.get("rows").and_then(|v| v.as_u64()),
        body.get("cols").and_then(|v| v.as_u64()),
    ) {
        let r = u16::try_from(r).unwrap_or(24).max(1);
        let c = u16::try_from(c).unwrap_or(80).max(1);
        if let Err(e) = pty.resize(r, c) {
            return api_err(&e);
        }
    }
    Json(serde_json::json!({ "ok": true })).into_response()
}

/// `POST /pty/remove` — terminate and close. Body: {pty_id}. Idempotent
/// for unknown ids (the terminal is already gone).
async fn pty_remove(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Option<Json<serde_json::Value>>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let Some(Json(body)) = body else {
        return wire_refused("pty/remove requires a body");
    };
    let id = match body
        .get("pty_id")
        .and_then(|i| i.as_str())
        .and_then(|s| s.parse::<u64>().ok())
    {
        Some(id) => id,
        None => return wire_refused("pty/remove requires pty_id"),
    };
    if let Some(mut pty) = state.ptys.lock().unwrap().remove(&id) {
        pty.kill();
    }
    Json(serde_json::json!({ "ok": true })).into_response()
}

/// `GET /pty/{id}/output` — snapshot available output (does NOT drain).
async fn pty_output(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let id = match id.parse::<u64>() {
        Ok(id) => id,
        Err(_) => return wire_refused("invalid pty id"),
    };
    let ptys = state.ptys.lock().unwrap();
    let pty = match ptys.get(&id) {
        Some(p) => p,
        None => return wire_status(not_found(&format!("pty {id}"))),
    };
    let out = pty.snapshot();
    let text = String::from_utf8_lossy(&out).into_owned();
    Json(serde_json::json!({ "ok": true, "output": text, "alive": pty.is_alive() })).into_response()
}

fn api_error_json(e: &faktor_core::error::Error) -> serde_json::Value {
    serde_json::json!({ "ok": false, "code": format!("{:?}", e.kind).to_lowercase(), "message": e.message })
}

// ------------------------------------------------------------ disposal & auth

/// `POST /global/dispose` and `POST /instance/dispose` — stop everything:
/// every supervised process owned by a session is killed via the agent
/// (which owns the supervisor), then each session is durably ended
/// (SessionEnded journal event + lifecycle Closed). Honest refusal with the
/// first failing session when any cannot end.
async fn dispose_all_sessions(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let handles = match state.deps.session.list_sessions(None) {
        Ok(h) => h,
        Err(e) => return api_err(&e),
    };
    for handle in handles {
        let id = handle.id();
        // Idempotent: sessions that are already durably ended are skipped
        // (a second dispose must still answer ok:true).
        let row = match handle.row() {
            Ok(r) => r,
            Err(e) => return api_err(&e),
        };
        if row.lifecycle.is_terminal() {
            continue;
        }
        // Cancel any live turn first so the durable end transition is legal
        // from the landing state.
        let _ = state.deps.agent.abort(id);
        if let Err(e) = state.deps.agent.end_session(id) {
            if e.kind == faktor_core::error::ErrorKind::NotFound {
                continue; // vanished mid-dispose
            }
            return wire_refused(&format!("dispose incomplete: session {id}: {}", e.message));
        }
    }
    Json(OkResponse { ok: true }).into_response()
}

/// `POST /instance/reload` — re-run the daemon's crash recovery sweep over
/// every session (idempotent) and acknowledge.
async fn instance_reload(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    match state.deps.agent.recover() {
        Ok(_reports) => Json(OkResponse { ok: true }).into_response(),
        Err(e) => api_err(&e),
    }
}

/// Upper bound on an `auth.set` password (bounded everything).
const MAX_AUTH_PASSWORD_BYTES: usize = 1024;

/// `POST /auth/set` — rotate the server password. `password` absent rotates
/// to a fresh random secret; either way the response carries the new
/// effective secret so the client can keep authenticating. Every other
/// endpoint immediately checks the new secret (old credentials → 401).
async fn auth_set(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<AuthSetRequest>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let password = match req.password {
        Some(p) if p.is_empty() || p.len() > MAX_AUTH_PASSWORD_BYTES => {
            let e = ApiError {
                code: "malformed",
                message: format!(
                    "password must be non-empty and at most {MAX_AUTH_PASSWORD_BYTES} bytes"
                ),
                http_status: 400,
                retryable: false,
            };
            return (StatusCode::BAD_REQUEST, Json(e.to_json())).into_response();
        }
        Some(p) => p,
        None => ServerPassword::generate().as_str().to_string(),
    };
    *state.auth.write().unwrap() = Some(ServerPassword::new(password.clone()));
    Json(AuthSetResponse { ok: true, password }).into_response()
}

/// `POST /auth/remove` — drop the runtime override: authentication returns
/// to the startup env password (`FAKTOR_SERVER_PASSWORD` at daemon start).
async fn auth_remove(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    *state.auth.write().unwrap() = None;
    Json(OkResponse { ok: true }).into_response()
}

/// The states that mean "a logical turn is occupying the session machine"
/// (the wait condition of `POST /session/{id}/message`). Everything else —
/// Idle, ReadyForNextTurn, Completed, Cancelled, FailedRecoverable/
/// FailedPermanent, NeedsUserInput, Suspended — means the accepted turn has
/// finished (or never started).
fn turn_machine_busy(s: AgentState) -> bool {
    matches!(
        s,
        AgentState::Preparing
            | AgentState::BuildingContext
            | AgentState::WaitingForModel
            | AgentState::Streaming
            | AgentState::ToolRequested
            | AgentState::WaitingForPermission
            | AgentState::ExecutingTool
            | AgentState::Validating
            | AgentState::UpdatingMemory
    )
}

fn wire_refused(message: &str) -> Response {
    (
        StatusCode::CONFLICT,
        Json(RevertResponse {
            ok: false,
            message: Some(message.to_string()),
        }),
    )
        .into_response()
}

fn not_found(message: &str) -> ApiError {
    ApiError {
        code: "not_found",
        message: message.to_string(),
        http_status: 404,
        retryable: false,
    }
}

fn wire_status(e: ApiError) -> Response {
    (
        StatusCode::from_u16(e.http_status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        Json(e.to_json()),
    )
        .into_response()
}

// ------------------------------------------------- question/network/config
// This runtime has no separate question/network subsystems: questions and
// network requests ARE pending permission requests (the daemon's one
// interactive-hop machinery). The frozen surfaces are served over the real
// pending-permission state, split by capability class:
// - question.* → pending permissions whose capability is NOT network;
// - network.* → pending permissions whose capability IS `network`;
// - reply = resolve with the body's decision; reject = resolve Deny.
// Unknown ids stay loud 404s; double resolution is a loud 409 (never
// silent success for something that does not exist).

/// The frozen capability tag a network permission carries.
const NETWORK_CAPABILITY: &str = "network";

fn pending_permission_json(p: &PendingPermission) -> serde_json::Value {
    serde_json::json!({
        "id": p.id.to_string(),
        "session_id": p.session_id.to_string(),
        "capability": p.capability,
        "detail": p.detail,
    })
}

/// Resolve one pending permission by wire id through the permission
/// requester. `class`: `Some("network")` restricts to network requests,
/// `Some("question")` to everything else.
fn resolve_pending_permission(
    deps: &ServerDeps,
    raw_id: &str,
    decision: &str,
    class: &str,
) -> Result<(), ApiError> {
    let id: i64 = match raw_id.parse() {
        Ok(p) if p > 0 => p,
        _ => {
            return Err(ApiError {
                code: "not_found",
                message: format!("{} {raw_id} unknown", class_kind(class)),
                http_status: 404,
                retryable: false,
            })
        }
    };
    let decision = match decision {
        "allow" => PermissionDecision::Allow,
        "deny" => PermissionDecision::Deny,
        other => {
            return Err(ApiError {
                code: "malformed",
                message: format!("invalid decision {other:?}"),
                http_status: 400,
                retryable: false,
            })
        }
    };
    // The permission must exist AND belong to the requested class: an id
    // from the other class is unknown HERE (it stays resolvable through its
    // own surface).
    let pending = deps
        .permissions
        .pending_views()
        .into_iter()
        .find(|v| v.id == id);
    let Some(view) = pending else {
        return Err(ApiError {
            code: "not_found",
            message: format!("{} {raw_id} unknown", class_kind(class)),
            http_status: 404,
            retryable: false,
        });
    };
    let is_network = view.capability == NETWORK_CAPABILITY;
    match class {
        "network" if !is_network => {
            return Err(ApiError {
                code: "not_found",
                message: format!("network {raw_id} unknown"),
                http_status: 404,
                retryable: false,
            })
        }
        "question" if is_network => {
            return Err(ApiError {
                code: "not_found",
                message: format!("question {raw_id} unknown"),
                http_status: 404,
                retryable: false,
            })
        }
        _ => {}
    }
    if !deps.permissions.resolve(id, decision) {
        return Err(ApiError {
            code: "conflict",
            message: format!("{} {raw_id} unknown or already resolved", class_kind(class)),
            http_status: 409,
            retryable: false,
        });
    }
    Ok(())
}

fn class_kind(class: &str) -> &'static str {
    if class == "network" {
        "network"
    } else {
        "question"
    }
}

async fn question_reply(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<QuestionReplyRequest>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    if req.question_id.trim().is_empty() || req.decision.trim().is_empty() {
        let e = ApiError {
            code: "malformed",
            message: "question_id and decision are required".into(),
            http_status: 400,
            retryable: false,
        };
        return (StatusCode::BAD_REQUEST, Json(e.to_json())).into_response();
    }
    match resolve_pending_permission(&state.deps, &req.question_id, &req.decision, "question") {
        Ok(()) => Json(PermissionDecisionResponse { ok: true }).into_response(),
        Err(e) => (
            StatusCode::from_u16(e.http_status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            Json(e.to_json()),
        )
            .into_response(),
    }
}

/// `POST /question/reject` — deny is the whole semantics (never allow).
async fn question_reject(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<QuestionRejectRequest>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    if req.question_id.trim().is_empty() {
        let e = ApiError {
            code: "malformed",
            message: "question_id is required".into(),
            http_status: 400,
            retryable: false,
        };
        return (StatusCode::BAD_REQUEST, Json(e.to_json())).into_response();
    }
    match resolve_pending_permission(&state.deps, &req.question_id, "deny", "question") {
        Ok(()) => Json(PermissionDecisionResponse { ok: true }).into_response(),
        Err(e) => (
            StatusCode::from_u16(e.http_status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            Json(e.to_json()),
        )
            .into_response(),
    }
}

async fn question_list(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<SdkSessionQuery>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let questions: Vec<serde_json::Value> = state
        .deps
        .permissions
        .pending_views()
        .into_iter()
        .filter(|v| v.capability != NETWORK_CAPABILITY)
        .filter(|v| {
            q.session_id
                .as_ref()
                .is_none_or(|sid| v.session_id.to_string() == *sid)
        })
        .map(|v| pending_permission_json(&v))
        .collect();
    Json(QuestionListResponse { questions }).into_response()
}

async fn network_reply(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<NetworkReplyRequest>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    if req.network_id.trim().is_empty() || req.decision.trim().is_empty() {
        let e = ApiError {
            code: "malformed",
            message: "network_id and decision are required".into(),
            http_status: 400,
            retryable: false,
        };
        return (StatusCode::BAD_REQUEST, Json(e.to_json())).into_response();
    }
    match resolve_pending_permission(&state.deps, &req.network_id, &req.decision, "network") {
        Ok(()) => Json(PermissionDecisionResponse { ok: true }).into_response(),
        Err(e) => (
            StatusCode::from_u16(e.http_status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            Json(e.to_json()),
        )
            .into_response(),
    }
}

/// `POST /network/reject` — deny is the whole semantics.
async fn network_reject(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<NetworkRejectRequest>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    if req.network_id.trim().is_empty() {
        let e = ApiError {
            code: "malformed",
            message: "network_id is required".into(),
            http_status: 400,
            retryable: false,
        };
        return (StatusCode::BAD_REQUEST, Json(e.to_json())).into_response();
    }
    match resolve_pending_permission(&state.deps, &req.network_id, "deny", "network") {
        Ok(()) => Json(PermissionDecisionResponse { ok: true }).into_response(),
        Err(e) => (
            StatusCode::from_u16(e.http_status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            Json(e.to_json()),
        )
            .into_response(),
    }
}

async fn network_list(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<SdkSessionQuery>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let networks: Vec<serde_json::Value> = state
        .deps
        .permissions
        .pending_views()
        .into_iter()
        .filter(|v| v.capability == NETWORK_CAPABILITY)
        .filter(|v| {
            q.session_id
                .as_ref()
                .is_none_or(|sid| v.session_id.to_string() == *sid)
        })
        .map(|v| pending_permission_json(&v))
        .collect();
    Json(NetworkListResponse { networks }).into_response()
}

async fn config_get(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let config = state.config.read().unwrap().clone();
    Json(ConfigGetResponse { config }).into_response()
}

/// The daemon-editable top-level config keys (`config.update` allowlist):
/// provider configuration is out of scope by design — only the model,
/// the compaction threshold and the system instructions may be applied.
const CONFIG_EDITABLE_KEYS: [&str; 3] = ["model", "compact_at_usage", "instructions"];

/// `POST /config/set` — the SDK full-replacement form (kept for the old
/// tests/clients): the whole config view is replaced, bounded.
async fn config_set(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ConfigSetRequest>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let bytes = serde_json::to_vec(&req.config).unwrap_or_default();
    if bytes.len() > MAX_CONFIG_BYTES {
        return config_oversized(bytes.len());
    }
    *state.config.write().unwrap() = req.config;
    Json(ConfigSetResponse { ok: true }).into_response()
}

fn config_oversized(bytes: usize) -> Response {
    let e = ApiError {
        code: "oversized",
        message: format!("config of {bytes} bytes exceeds {MAX_CONFIG_BYTES}"),
        http_status: 413,
        retryable: false,
    };
    (StatusCode::PAYLOAD_TOO_LARGE, Json(e.to_json())).into_response()
}

fn config_must_be_object(config: &serde_json::Value) -> Option<Response> {
    if !config.is_object() {
        let e = ApiError {
            code: "malformed",
            message: "config must be a JSON object".into(),
            http_status: 400,
            retryable: false,
        };
        return Some((StatusCode::BAD_REQUEST, Json(e.to_json())).into_response());
    }
    None
}

/// `POST /config/update` — apply ONLY the daemon-editable keys
/// (model/compact_at_usage/instructions) onto the stored config; any other
/// top-level key is rejected with a clear error, never silently dropped.
async fn config_update(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ConfigSetRequest>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let bytes = serde_json::to_vec(&req.config).unwrap_or_default();
    if bytes.len() > MAX_CONFIG_BYTES {
        return config_oversized(bytes.len());
    }
    if let Some(resp) = config_must_be_object(&req.config) {
        return resp;
    }
    let incoming = req.config.as_object().unwrap();
    for key in incoming.keys() {
        if !CONFIG_EDITABLE_KEYS.contains(&key.as_str()) {
            let e = ApiError {
                code: "malformed",
                message: format!(
                    "config key {key:?} is not daemon-editable; allowed keys: {}",
                    CONFIG_EDITABLE_KEYS.join(", ")
                ),
                http_status: 400,
                retryable: false,
            };
            return (StatusCode::BAD_REQUEST, Json(e.to_json())).into_response();
        }
    }
    {
        let mut config = state.config.write().unwrap();
        let target = config
            .as_object_mut()
            .expect("daemon config is always an object");
        for (key, value) in incoming {
            target.insert(key.clone(), value.clone());
        }
    }
    Json(ConfigSetResponse { ok: true }).into_response()
}

/// `GET /config/warnings` — real validation warnings over the stored config
/// (empty when everything validates).
async fn config_warnings(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let config = state.config.read().unwrap().clone();
    let mut warnings = Vec::new();
    if let Some(obj) = config.as_object() {
        for (key, value) in obj {
            match key.as_str() {
                "model" => {
                    if !value.is_string() {
                        warnings.push("config \"model\" must be a string".into());
                    }
                }
                "compact_at_usage" => {
                    let ok = value.as_f64().is_some_and(|v| (0.0..=1.0).contains(&v));
                    if !ok {
                        warnings
                            .push("config \"compact_at_usage\" must be a number in [0, 1]".into());
                    }
                }
                "instructions" => {
                    if !value.is_string() {
                        warnings.push("config \"instructions\" must be a string".into());
                    }
                }
                other => warnings.push(format!(
                    "unknown config key {other:?} (daemon-editable keys: {})",
                    CONFIG_EDITABLE_KEYS.join(", ")
                )),
            }
        }
    }
    Json(ConfigWarningsResponse { warnings }).into_response()
}

/// `POST /config/overlay` — store a bounded overlay, replacing the whole
/// daemon config view. `POST /config/overlayUpdate` shallow-merges instead.
async fn config_overlay(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ConfigSetRequest>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let bytes = serde_json::to_vec(&req.config).unwrap_or_default();
    if bytes.len() > MAX_CONFIG_BYTES {
        return config_oversized(bytes.len());
    }
    if let Some(resp) = config_must_be_object(&req.config) {
        return resp;
    }
    *state.config.write().unwrap() = req.config;
    Json(ConfigSetResponse { ok: true }).into_response()
}

/// `POST /config/overlayUpdate` — bounded shallow merge of the overlay keys
/// into the current config view.
async fn config_overlay_update(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ConfigSetRequest>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let bytes = serde_json::to_vec(&req.config).unwrap_or_default();
    if bytes.len() > MAX_CONFIG_BYTES {
        return config_oversized(bytes.len());
    }
    if let Some(resp) = config_must_be_object(&req.config) {
        return resp;
    }
    let incoming = req.config.as_object().unwrap();
    {
        let mut config = state.config.write().unwrap();
        let target = config
            .as_object_mut()
            .expect("daemon config is always an object");
        for (key, value) in incoming {
            target.insert(key.clone(), value.clone());
        }
    }
    Json(ConfigSetResponse { ok: true }).into_response()
}

async fn provider_list(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let ids = state.deps.agent.deps().providers.ids();
    let mut providers = Vec::new();
    for id in ids {
        let models = if let Some(p) = state.deps.agent.deps().providers.get(&id) {
            // Dynamic registry (audit round 8): the adapter's REAL model
            // list with REAL capabilities — the model selector can now
            // enumerate what the daemon can actually serve.
            p.known_models()
                .into_iter()
                .map(|m| ModelInfo {
                    id: m.clone(),
                    name: m.clone(),
                    capabilities: p.capabilities(&m),
                })
                .collect()
        } else {
            vec![]
        };
        providers.push(ProviderInfo {
            id: id.clone(),
            name: id.clone(),
            kind: id.clone(),
            models,
        });
    }
    Json(ProviderList { providers }).into_response()
}

// ------------------------------------------------------------------ native v1
// Faktor Native Protocol v1 (docs/native-protocol.md): the daemon's own
// HTTP surface. UI compatibility is the target; these handlers map to
// durable runtime state only (row, journal, ledger, turn records,
// tool-run rows) and never fabricate v7.5.6 frames.

/// The snake_case lifecycle tag for native projections.
fn lifecycle_tag(l: SessionLifecycle) -> String {
    serde_json::to_string(&l)
        .unwrap_or_else(|_| "unknown".into())
        .trim_matches('"')
        .to_string()
}

/// Bound on the verification list of one projection (hostile rows capped).
const MAX_NATIVE_VERIFICATION: usize = 32;

/// One native projection snapshot (docs/native-protocol.md, GET
/// /session/{id}/projection). Every field maps to what the server can
/// read durably: state/session come from the session row (same source as
/// the v7.5.6 state handler); activeModel is the effective provider/model
/// envelope of the current or most recent logical turn (durable turn
/// records); activeTool is the newest still-running durable tool-run row;
/// filesChanged comes from the durable task ledger (`changed_files`);
/// lastCheckpoint is the newest checkpoint row, present only when a
/// checkpoint service is wired (`ServerDeps.snapshots`); verification
/// lists still-open tool runs whose durable recovery strategy is
/// MarkUnknown (unknown external effects are forced to verification —
/// spec §7), bounded; queued is the durable queued-prompt count.
/// `progress` and `contextUsage` are always null in this revision: the
/// runtime has no numeric progress channel, and provider-call usage rows
/// have no durable read API yet — the machine state, activeTool and the
/// journal carry the phase information.
fn build_native_projection(
    deps: &ServerDeps,
    handle: &faktor_session::SessionHandle,
) -> faktor_core::Result<serde_json::Value> {
    let row = handle.row()?;
    let state = row.state;
    // Durable task ledger: changed files (bounded by construction in the
    // ledger; hostile rows are read defensively — strings only, capped).
    let mut files_changed: Vec<String> = Vec::new();
    if let Some(ledger) = handle.get_task_ledger()? {
        if let Some(files) = ledger.get("changed_files").and_then(|f| f.as_array()) {
            for f in files.iter().take(256) {
                if let Some(p) = f.as_str() {
                    files_changed.push(p.to_string());
                }
            }
        }
    }
    // Effective model envelope: newest durable turn record (oldest first
    // in the store), null before the first turn.
    let active_model = handle.turn_records()?.last().map(|t| {
        serde_json::json!({
            "provider": t.effective_provider,
            "model": t.effective_model,
            "variant": t.variant,
        })
    });
    // Active tool: the newest durable tool-run row that is still running
    // (an interrupted row is reconstructed by crash recovery before the
    // next turn; none pending = no active tool).
    let pending = handle.pending_tool_runs()?;
    let active_tool = pending
        .iter()
        .max_by_key(|r| (r.started_ms, r.id))
        .map(|r| {
            serde_json::json!({
                "tool": r.tool,
                "opId": r.op_id.to_string(),
                "startedMs": r.started_ms,
                "status": r.status,
            })
        });
    // Verification: still-open runs carrying unknown external effects
    // (recovery strategy mark_unknown) are owed verification (§7).
    let verification: Vec<serde_json::Value> = pending
        .iter()
        .filter(|r| r.recovery.get("strategy").and_then(|s| s.as_str()) == Some("mark_unknown"))
        .take(MAX_NATIVE_VERIFICATION)
        .map(|r| {
            serde_json::json!({
                "opId": r.op_id.to_string(),
                "tool": r.tool,
                "startedMs": r.started_ms,
                "effectStatus": r.effect_status,
            })
        })
        .collect();
    // Checkpoint presence requires the real snapshot service; the newest
    // durable checkpoint row is projected when one exists.
    let last_checkpoint = if deps.snapshots.is_some() {
        handle
            .checkpoints_of()?
            .into_iter()
            .max_by_key(|c| (c.sequence, c.id))
            .map(|c| {
                serde_json::json!({
                    "sequence": c.sequence,
                    "path": c.path,
                    "createdMs": c.created_ms,
                    "restoredMs": c.restored_ms,
                })
            })
    } else {
        None
    };
    Ok(serde_json::json!({
        "session": {
            "id": row.id.to_string(),
            "title": row.title,
            "provider": row.provider,
            "model": row.model,
            "lifecycle": lifecycle_tag(row.lifecycle),
        },
        "state": {
            "machine": agent_state_tag(state),
            "label": state.label(),
            "active": state.is_active(),
            "terminal": state.is_terminal(),
        },
        "activeModel": active_model,
        "activeTool": active_tool,
        "progress": serde_json::Value::Null,
        "filesChanged": files_changed,
        "lastCheckpoint": last_checkpoint,
        "verification": verification,
        "contextUsage": serde_json::Value::Null,
        "queued": handle.queued_prompt_count()?.max(0),
    }))
}

/// `GET /session/{id}/projection` — native v1 session projection
/// (auth-required like every native endpoint).
async fn native_session_projection(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let sid = match parse_session_id(&id) {
        Ok(s) => s,
        Err(e) => return wire_status(e),
    };
    let handle = match state.deps.session.get_session(sid) {
        Ok(Some(h)) => h,
        Ok(None) => return wire_status(not_found(&format!("session {sid}"))),
        Err(e) => return api_err(&e),
    };
    match build_native_projection(&state.deps, &handle) {
        Ok(v) => Json(v).into_response(),
        Err(e) => api_err(&e),
    }
}

/// Provenance of one native catalog entry (docs/native-protocol.md):
/// `liveProbe` when the provider reports a LIVE runtime context limit
/// for the model (e.g. an Ollama `/api/ps` allocation); otherwise
/// `providerCatalog` for entries carrying a non-default capability
/// profile (configured or probed), and `conservativeDefault` for entries
/// still at the fail-safe default profile (unprobed).
fn catalog_source(
    p: &dyn faktor_provider::Provider,
    model: &str,
    caps: &ModelCapabilities,
) -> &'static str {
    if p.runtime_context_limit(model).is_some() {
        "liveProbe"
    } else if *caps == ModelCapabilities::default() {
        "conservativeDefault"
    } else {
        "providerCatalog"
    }
}

/// `GET /models` — the flat native model catalog: every registered
/// provider instance × its known models × capabilities
/// (docs/native-protocol.md).
async fn native_models(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let mut out = Vec::new();
    for p in state.deps.agent.deps().providers.all() {
        // Registry key (instance id) — the same id session rows store.
        let instance = p.identity().instance_id.clone();
        for model in p.known_models() {
            let caps = p.capabilities(&model);
            let source = catalog_source(p.as_ref(), &model, &caps);
            out.push(serde_json::json!({
                "provider": instance,
                "model": model,
                "context": caps.context,
                "maxOutput": caps.max_output,
                "tools": caps.tools,
                "parallelTools": caps.parallel_tools,
                "reasoning": caps.reasoning,
                "thinking": caps.thinking,
                "vision": caps.vision,
                "structuredOutput": caps.json_schema,
                "embeddings": caps.embeddings,
                "streaming": caps.streaming,
                "source": source,
            }));
        }
    }
    out.sort_by(|a, b| {
        (a["provider"].as_str(), a["model"].as_str())
            .cmp(&(b["provider"].as_str(), b["model"].as_str()))
    });
    Json(out).into_response()
}

/// `GET /capabilities` — native introspection map:
/// `{ "<provider>": { models: [{id, capabilities}],
/// runtimeContextLimitSupported } }` (docs/native-protocol.md).
async fn native_capabilities(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let mut sorted = Vec::new();
    for p in state.deps.agent.deps().providers.all() {
        let instance = p.identity().instance_id.clone();
        let mut models = Vec::new();
        let mut live = false;
        for m in p.known_models() {
            if p.runtime_context_limit(&m).is_some() {
                live = true;
            }
            models.push(serde_json::json!({ "id": m, "capabilities": p.capabilities(&m) }));
        }
        models.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
        sorted.push((
            instance,
            serde_json::json!({
                "models": models,
                "runtimeContextLimitSupported": live,
            }),
        ));
    }
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    let mut map = serde_json::Map::new();
    for (instance, entry) in sorted {
        map.insert(instance, entry);
    }
    Json(serde_json::Value::Object(map)).into_response()
}

// ------------------------------------------------- native v1: audits 55-56
// Liveness/readiness and the durable session listings (all auth-gated like
// every daemon route). Response bodies are native JSON over durable state:
// session rows, turn records, the task ledger, checkpoint rows, memory
// facts and live PTYs — never v7.5.6 wire shapes. Hostile ids are 400
// (unparseable/0) or 404 (unknown); handlers never panic on them.

/// Bound of one native newest-first listing (bounded everything).
const MAX_NATIVE_LIST: usize = 500;

/// `GET /native/health` — liveness: 200 `{ok:true, version}` whenever the
/// process responds (auth-gated like `/global/health`). Conceptually the
/// same "process is up" answer as health; readiness is `/native/ready`.
async fn native_health(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    Json(serde_json::json!({
        "ok": true,
        "version": state.deps.version.clone(),
    }))
    .into_response()
}

/// `GET /native/ready` — readiness: 200 `{ready:true}` only when the
/// session store has recovered (the flag flips at the end of serve()
/// setup, after the caller opened/recovered the store and wired every
/// runtime component — see `serve()`), else 503 `{ready:false}`.
async fn native_ready(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    if state.ready.load(std::sync::atomic::Ordering::SeqCst) {
        Json(serde_json::json!({ "ready": true })).into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "ready": false })),
        )
            .into_response()
    }
}

/// Resolve the native path session id: malformed (non-numeric/0) → 400,
/// unknown → 404, store errors → 500. Shared by every `/native/session`
/// handler.
fn native_resolve_session(
    state: &AppState,
    id: &str,
) -> Result<faktor_session::SessionHandle, Box<Response>> {
    let sid = match parse_session_id(id) {
        Ok(s) => s,
        Err(e) => return Err(Box::new(wire_status(e))),
    };
    match state.deps.session.get_session(sid) {
        Ok(Some(h)) => Ok(h),
        Ok(None) => Err(Box::new(wire_status(not_found(&format!("session {sid}"))))),
        Err(e) => Err(Box::new(api_err(&e))),
    }
}

/// The durable verification facts of one session: memory facts of kind
/// `verification` (the runtime records one per failed REQUIRED check with
/// key = check id, value = "failed:<command>"). Bounded.
fn native_verification_facts(handle: &faktor_session::SessionHandle) -> Vec<serde_json::Value> {
    let facts = handle.memory_facts().unwrap_or_default();
    facts
        .iter()
        .filter(|(kind, _, _)| kind == "verification")
        .take(MAX_NATIVE_LIST)
        .map(|(_, key, value)| {
            serde_json::json!({
                "id": key,
                "detail": value,
                "status": "failed",
            })
        })
        .collect()
}

/// `GET /native/session/{id}/turns` — the durable turn records of the
/// session (one per admitted logical turn): envelope, status, timestamps.
/// Newest first, bounded.
async fn native_session_turns(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let handle = match native_resolve_session(&state, &id) {
        Ok(h) => h,
        Err(r) => return *r,
    };
    match handle.turn_records() {
        Ok(rows) => {
            let out: Vec<serde_json::Value> = rows
                .iter()
                .rev()
                .take(MAX_NATIVE_LIST)
                .map(|t| {
                    serde_json::json!({
                        "opId": t.turn_op_id.to_string(),
                        "status": t.status,
                        "provider": t.effective_provider,
                        "model": t.effective_model,
                        "variant": t.variant,
                        "toolMode": t.tool_mode,
                        "startedAt": t.started_at,
                        "updatedMs": t.updated_ms,
                        "queueSeq": t.queue_seq,
                        "promptMessageId": t.prompt_message_id,
                    })
                })
                .collect();
            Json(out).into_response()
        }
        Err(e) => api_err(&e),
    }
}

/// Defensive string-array reader over the durable ledger JSON (hostile
/// values are skipped, capped at MAX_NATIVE_LIST).
fn ledger_strings(ledger: &serde_json::Value, key: &str) -> Vec<String> {
    ledger
        .get(key)
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|s| s.as_str())
                .take(MAX_NATIVE_LIST)
                .map(|s| s.to_string())
                .collect()
        })
        .unwrap_or_default()
}

/// `GET /native/session/{id}/tasks` — the durable task ledger as typed
/// JSON (goal, milestones, decisions, failures, changed files) plus the
/// session's durable verification facts. One entry per tracked task; today
/// the session ledger is single-task, so the array is either `[]` (no
/// task data yet) or one entry. The ledger row is read defensively: only
/// strings are copied, arrays are capped.
async fn native_session_tasks(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let handle = match native_resolve_session(&state, &id) {
        Ok(h) => h,
        Err(r) => return *r,
    };
    let ledger = match handle.get_task_ledger() {
        Ok(l) => l,
        Err(e) => return api_err(&e),
    };
    let Some(ledger) = ledger else {
        // No ledger row yet: no tracked task.
        return Json(serde_json::json!([])).into_response();
    };
    let goal = ledger
        .get("goal")
        .and_then(|g| g.as_str())
        .unwrap_or("")
        .to_string();
    let completed = ledger_strings(&ledger, "completed_steps");
    let open = ledger_strings(&ledger, "open_steps");
    if goal.is_empty()
        && completed.is_empty()
        && open.is_empty()
        && ledger_strings(&ledger, "decisions").is_empty()
        && ledger_strings(&ledger, "changed_files").is_empty()
    {
        // A stored-but-empty ledger (a turn ran without task data) is not a
        // task; list nothing rather than a phantom entry.
        return Json(serde_json::json!([])).into_response();
    }
    // State derivation (documented): the live machine wins ("running");
    // open milestones or a fresh goal with no completed work yet are
    // "in_progress"; completed work with nothing left open is "done".
    let row = match handle.row() {
        Ok(r) => r,
        Err(e) => return api_err(&e),
    };
    let state_tag = if row.state.is_active() {
        "running"
    } else if !open.is_empty() || (completed.is_empty() && !goal.is_empty()) {
        "in_progress"
    } else if !completed.is_empty() {
        "done"
    } else {
        "idle"
    };
    Json(serde_json::json!([{
        "goal": goal,
        "constraints": ledger_strings(&ledger, "constraints"),
        "state": state_tag,
        "milestones": { "completed": completed, "open": open },
        "decisions": ledger_strings(&ledger, "decisions"),
        "failures": ledger_strings(&ledger, "known_failures"),
        "changedFiles": ledger_strings(&ledger, "changed_files"),
        "tests": {
            "run": ledger_strings(&ledger, "tests_run"),
            "failed": ledger_strings(&ledger, "tests_failed"),
        },
        "preferences": ledger_strings(&ledger, "user_preferences"),
        "verification": native_verification_facts(&handle),
    }]))
    .into_response()
}

/// `GET /native/session/{id}/checkpoints` — the durable checkpoint rows of
/// the session (newest first), each with sequence/path/before-after hashes
/// and the restore audit. Empty array when the daemon runs without a
/// checkpoint service wired (`ServerDeps.snapshots` is `None`) or no
/// checkpoint was recorded yet.
async fn native_session_checkpoints(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let handle = match native_resolve_session(&state, &id) {
        Ok(h) => h,
        Err(r) => return *r,
    };
    if state.deps.snapshots.is_none() {
        return Json(serde_json::json!([])).into_response();
    }
    match handle.checkpoints_of() {
        Ok(rows) => {
            let mut rows = rows;
            rows.sort_by_key(|c| std::cmp::Reverse((c.sequence, c.id)));
            let out: Vec<serde_json::Value> = rows
                .iter()
                .take(MAX_NATIVE_LIST)
                .map(|c| {
                    serde_json::json!({
                        "sequence": c.sequence,
                        "path": c.path,
                        "beforeHash": c.before_hash,
                        "afterHash": c.after_hash,
                        "beforeExists": c.before_exists,
                        "afterExists": c.after_exists,
                        "createdMs": c.created_ms,
                        "restoredMs": c.restored_ms,
                    })
                })
                .collect();
            Json(out).into_response()
        }
        Err(e) => api_err(&e),
    }
}

/// `GET /native/session/{id}/verification` — everything the session owes
/// verification (audit 55 wiring): `owed` = still-open durable tool runs
/// whose recovery strategy is mark_unknown (unknown external effects are
/// forced to verification — spec §7), `failedChecks` = the durable
/// verification facts (one per failed REQUIRED check, recorded at genuine
/// turn ends). Both bounded; empty arrays when nothing is owed.
async fn native_session_verification(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let handle = match native_resolve_session(&state, &id) {
        Ok(h) => h,
        Err(r) => return *r,
    };
    let pending = match handle.pending_tool_runs() {
        Ok(p) => p,
        Err(e) => return api_err(&e),
    };
    let owed: Vec<serde_json::Value> = pending
        .iter()
        .filter(|r| r.recovery.get("strategy").and_then(|s| s.as_str()) == Some("mark_unknown"))
        .take(MAX_NATIVE_LIST)
        .map(|r| {
            serde_json::json!({
                "opId": r.op_id.to_string(),
                "tool": r.tool,
                "startedMs": r.started_ms,
                "status": r.status,
                "effectStatus": r.effect_status,
            })
        })
        .collect();
    Json(serde_json::json!({
        "owed": owed,
        "failedChecks": native_verification_facts(&handle),
    }))
    .into_response()
}

/// `GET /native/session/{id}/agents` — background agents owned by the
/// session. Always an empty array in this revision: child sessions appear
/// when orchestration (Agent Manager subagent sessions) lands in the
/// runtime. The endpoint exists so the UI can poll the shape now.
async fn native_session_agents(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    if let Err(r) = native_resolve_session(&state, &id) {
        return *r;
    }
    Json(serde_json::json!([])).into_response()
}

/// `GET /native/session/{id}/terminal` — the session's terminal view.
/// Live PTYs have no durable session binding yet, so this lists every live
/// PTY of the daemon (id + pid + alive) — session-scoped ownership is
/// documented as the next wiring step. Session id is still validated
/// (unknown → 404); hostile ids 400.
async fn native_session_terminal(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    if let Err(r) = native_resolve_session(&state, &id) {
        return *r;
    }
    let ptys = state.ptys.lock().expect("ptys poisoned");
    let mut entries: Vec<(u64, u32, bool)> = ptys
        .iter()
        .take(MAX_NATIVE_LIST)
        .map(|(pty_id, p)| (*pty_id, p.pid(), p.is_alive()))
        .collect();
    entries.sort_by_key(|(id, _, _)| *id);
    Json(serde_json::json!(entries
        .iter()
        .map(|(id, pid, alive)| serde_json::json!({ "id": id.to_string(), "pid": pid, "alive": alive }))
        .collect::<Vec<_>>()))
    .into_response()
}

/// `POST /native/session/{id}/abort` — the native abort (audit 56): the
/// strict `NativeAbortRequest` body (`deny_unknown_fields` — an unknown
/// field or typo is a 400) carries the session id, which must match the
/// path id. `op_id` targets one queued prompt or the active turn; absent =
/// abort everything. Unknown sessions are 404; the semantics are
/// `sdk_abort`'s (queued-prompt kills never touch the machine).
async fn native_session_abort(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    body: Result<Json<NativeAbortRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    // Strict native DTO: every body rejection (syntax AND data errors —
    // unknown fields, typos, missing fields) is a plain 400, never a 422.
    let Json(req) = match body {
        Ok(b) => b,
        Err(_) => {
            let e = ApiError {
                code: "malformed",
                message: "invalid native abort request body".into(),
                http_status: 400,
                retryable: false,
            };
            return wire_status(e);
        }
    };
    let sid = match parse_session_id(&id) {
        Ok(s) => s,
        Err(e) => return wire_status(e),
    };
    let body_sid = match parse_session_id(&req.session_id) {
        Ok(s) => s,
        Err(e) => return wire_status(e),
    };
    if sid != body_sid {
        let e = ApiError {
            code: "malformed",
            message: format!("path session id {sid} does not match body session id {body_sid}"),
            http_status: 400,
            retryable: false,
        };
        return wire_status(e);
    }
    match state.deps.session.get_session(sid) {
        Ok(Some(_)) => {}
        Ok(None) => return wire_status(not_found(&format!("session {sid}"))),
        Err(e) => return api_err(&e),
    }
    let target = match &req.op_id {
        Some(raw) => match raw.parse::<u64>() {
            Ok(v) => Some(faktor_core::id::OpId::new(v)),
            Err(_) => {
                let e = ApiError {
                    code: "malformed",
                    message: format!("invalid op_id {raw:?}"),
                    http_status: 400,
                    retryable: false,
                };
                return wire_status(e);
            }
        },
        None => None,
    };
    match state.deps.agent.abort_op(sid, target) {
        Ok(ops) => Json(serde_json::json!({
            "aborted": ops.iter().map(|o| o.to_string()).collect::<Vec<_>>(),
        }))
        .into_response(),
        Err(e) => api_err(&e),
    }
}

/// `GET /native/usage` — aggregate of the durable context-usage facts the
/// runtime records across sessions (memory facts of kind `usage`, keys
/// `budget`/`spent`, numeric string values). No runtime path writes those
/// facts yet, so today the totals are honest zeros and `perSession` is
/// empty unless a future audit wires the writer — the aggregate shape is
/// frozen here so clients can rely on it. Values that are not plain
/// integers are skipped (hostile rows can never break the aggregate).
async fn native_usage(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    const MAX_SESSIONS: usize = 10_000;
    let mut budget_total: i64 = 0;
    let mut spent_total: i64 = 0;
    let mut per_session: Vec<serde_json::Value> = Vec::new();
    // A session that vanishes mid-list is skipped, never fatal (the wire
    // list has the same convention).
    let mut sessions = match state.deps.session.list_sessions(None) {
        Ok(s) => s,
        Err(e) => return api_err(&e),
    };
    sessions.truncate(MAX_SESSIONS);
    for handle in &sessions {
        let facts = match handle.memory_facts() {
            Ok(f) => f,
            Err(_) => continue,
        };
        let mut budget: Option<i64> = None;
        let mut spent: Option<i64> = None;
        for (kind, key, value) in facts {
            if kind != "usage" {
                continue;
            }
            let parsed = value.parse::<i64>().ok();
            match key.as_str() {
                "budget" => budget = parsed,
                "spent" => spent = parsed,
                _ => {}
            }
        }
        if budget.is_none() && spent.is_none() {
            continue;
        }
        budget_total = budget_total.saturating_add(budget.unwrap_or(0));
        spent_total = spent_total.saturating_add(spent.unwrap_or(0));
        per_session.push(serde_json::json!({
            "sessionId": handle.id().to_string(),
            "budget": budget,
            "spent": spent,
        }));
    }
    Json(serde_json::json!({
        "sessions": sessions.len(),
        "totals": { "budget": budget_total, "spent": spent_total },
        "perSession": per_session,
    }))
    .into_response()
}

// ------------------------------------------------------------------ SSE

async fn events(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Query(q): Query<MessagesQuery>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let sid = match parse_session_id(&id) {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::from_u16(e.http_status).unwrap(),
                Json(e.to_json()),
            )
                .into_response()
        }
    };
    // The SSE cursor is the raw sequence (0 = from the beginning); the
    // journal is queried as `seq > cursor` via events_range.
    let cursor: i64 = q.events_after.unwrap_or(0).max(0);
    let handle = match state.deps.session.get_session(sid) {
        Ok(Some(h)) => h,
        Ok(None) => {
            let e = ApiError {
                code: "not_found",
                message: format!("session {sid}"),
                http_status: 404,
                retryable: false,
            };
            return (StatusCode::NOT_FOUND, Json(e.to_json())).into_response();
        }
        Err(e) => return api_err(&e),
    };

    // Catch-up frames from the journal, then poll for new ones. The journal
    // is the source of truth; a reconnect resumes exactly from the cursor.
    let stream = journal_stream(handle, cursor);
    Sse::new(stream)
        .keep_alive(
            axum::response::sse::KeepAlive::new()
                .interval(Duration::from_secs(HEARTBEAT_SECS))
                .text("keep-alive"),
        )
        .into_response()
}

fn journal_stream(
    handle: faktor_session::SessionHandle,
    cursor: i64,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> + Send + 'static {
    // State: (handle, next cursor, queue of ready frames).
    // Poll the journal; the journal is the source of truth and the cursor is
    // the SSE resume point. Heartbeats keep proxies alive.
    futures_util::stream::unfold(
        (handle, cursor, VecDeque::<Event>::new()),
        move |(handle, mut cursor, mut queue)| async move {
            if let Some(ev) = queue.pop_front() {
                return Some((
                    Ok::<Event, std::convert::Infallible>(ev),
                    (handle, cursor, queue),
                ));
            }
            // seq > cursor (cursor 0 = everything from seq 1).
            let events = handle
                .events_range(cursor.saturating_add(1) as u64, None)
                .unwrap_or_default();
            let mut batch = VecDeque::new();
            let mut advanced = false;
            for e in events {
                if let Some((event, _)) = faktor_protocol::sse::project_event(&e) {
                    batch.push_back(sse_event(faktor_session::JournalFrame {
                        seq: e.seq,
                        event,
                    }));
                }
                cursor = e.seq.raw() as i64;
                advanced = true;
            }
            if advanced {
                if let Some(ev) = batch.pop_front() {
                    return Some((
                        Ok::<Event, std::convert::Infallible>(ev),
                        (handle, cursor, batch),
                    ));
                }
            }
            tokio::time::sleep(Duration::from_millis(POLL_INTERVAL_MS)).await;
            Some((
                Ok::<Event, std::convert::Infallible>(sse_event_heartbeat()),
                (handle, cursor, queue),
            ))
        },
    )
}

fn sse_event(frame: faktor_session::JournalFrame) -> Event {
    let seq = frame.seq.raw();
    let json = serde_json::to_string(&frame.event).unwrap_or_else(|_| "{}".into());
    Event::default()
        .event(frame.event.event_type())
        .id(seq.to_string())
        .data(json)
}

fn sse_event_heartbeat() -> Event {
    Event::default().event("heartbeat").data("{}")
}

// ------------------------------------------------------------------ global SSE
// `GET /global/event?after=<n>` streams GlobalEvent envelopes as
// `id: <n>\ndata: <json>\n\n` frames (no `event:` field — the payload's
// `type` carries the discriminator). `after` is the resume cursor; oversized
// values are clamped to what the bounded ring can serve.

async fn global_events(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<GlobalEventsQuery>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let after = q.after.unwrap_or(0);
    let stream = global_stream(state.bus.clone(), after);
    Sse::new(stream)
        .keep_alive(
            axum::response::sse::KeepAlive::new()
                .interval(Duration::from_secs(HEARTBEAT_SECS))
                .text("keep-alive"),
        )
        .into_response()
}

fn global_stream(
    bus: Arc<GlobalEventBus>,
    after: u64,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> + Send + 'static {
    // Poll the bus for new frames (id > cursor), emit, then wait before the
    // next poll; heartbeats keep proxies alive. The bus cursors make the
    // poll idempotent across concurrent connections.
    futures_util::stream::unfold(
        (bus, after, VecDeque::<(u64, GlobalEvent)>::new()),
        |(bus, mut cursor, mut queue)| async move {
            if let Some((id, ge)) = queue.pop_front() {
                return Some((
                    Ok::<Event, std::convert::Infallible>(global_frame(id, ge)),
                    (bus, cursor, queue),
                ));
            }
            bus.poll_once();
            let frames = bus.frames_after(cursor);
            let mut batch = VecDeque::new();
            let mut advanced = false;
            for (id, ge) in frames {
                batch.push_back((id, ge));
                cursor = id;
                advanced = true;
            }
            if advanced {
                if let Some((id, ge)) = batch.pop_front() {
                    return Some((
                        Ok::<Event, std::convert::Infallible>(global_frame(id, ge)),
                        (bus, cursor, batch),
                    ));
                }
            }
            tokio::time::sleep(Duration::from_millis(POLL_INTERVAL_MS)).await;
            Some((
                Ok::<Event, std::convert::Infallible>(sse_event_heartbeat()),
                (bus, cursor, queue),
            ))
        },
    )
}

fn global_frame(id: u64, ge: GlobalEvent) -> Event {
    let json = serde_json::to_string(&ge).unwrap_or_else(|_| "{}".into());
    Event::default().id(id.to_string()).data(json)
}

fn api_err(e: &Error) -> Response {
    let api = faktor_protocol::error::from_core(e);
    (
        StatusCode::from_u16(api.http_status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        Json(api.to_json()),
    )
        .into_response()
}

fn _unused(_: Body) {}

#[cfg(test)]
mod tests {
    use super::*;
    use faktor_core::model::ModelCapabilities;
    use faktor_provider::FakeProvider;

    #[test]
    fn handshake_line_is_frozen_shape() {
        let deps = ServerDeps {
            session: SessionManager::open(
                std::env::temp_dir().join("kp-hs-store"),
                std::env::temp_dir().join("kp-hs-cas"),
                false,
            )
            .unwrap(),
            agent: AgentRuntime::new(faktor_agent::AgentDeps {
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
                verifier: None,
                model: "m".into(),
                compaction_model: None,
                compact_at_usage: 0.65,
                instructions: "i".into(),
                hooks: None,
                instructions_loader: None,
                router: None,
                budget_micro: None,
                clock: Arc::new(faktor_core::time::SystemClock),
                tool_call_mode: faktor_agent::ToolCallMode::Native,
                tool_deadline_ms: 1000,
                retry_policy: faktor_core::retry::RetryPolicy::default(),
            })
            .unwrap(),
            permissions: ChannelPermissionRequester::new(Duration::from_secs(1)),
            auth_token: AuthToken::generate(),
            server_password: ServerPassword::generate(),
            directory: None,
            version: "0.1.0".into(),
            fs: None,
            snapshots: None,
            chunk_rx: None,
            simulate_not_ready: false,
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
        let mut state = String::new();
        for _ in 0..100 {
            let resp = client
                .get(format!("{base}/api/session/{sid}/state"))
                .bearer_auth(token.as_str())
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

    #[tokio::test]
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
        registry.register(Arc::new(FakeProvider::with_script(
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
        )));
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
            verifier: None,
            model: "m".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "You are a test server agent.".into(),
            hooks: None,
            instructions_loader: None,
            router: None,
            budget_micro: None,
            clock: Arc::new(faktor_core::time::SystemClock),
            tool_call_mode: faktor_agent::ToolCallMode::Native,
            tool_deadline_ms: 2000,
            retry_policy: faktor_core::retry::RetryPolicy::default(),
        })
        .unwrap();
        // Replace the running server's deps by serving a second one on the
        // same store (the first server's fake provider has no tool call, so
        // the permission test needs its own instance).
        let deps2 = ServerDeps {
            session: session.clone(),
            agent,
            permissions: permissions.clone(),
            auth_token: token.clone(),
            server_password: ServerPassword::generate(),
            directory: None,
            version: "0.1.0".into(),
            fs: None,
            snapshots: None,
            chunk_rx: None,
            simulate_not_ready: false,
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
        registry.register(provider);
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
            verifier: None,
            model: "m".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "You are a test server agent.".into(),
            hooks: None,
            instructions_loader: None,
            router: None,
            budget_micro: None,
            clock: Arc::new(faktor_core::time::SystemClock),
            tool_call_mode: faktor_agent::ToolCallMode::Native,
            tool_deadline_ms: 2000,
            retry_policy: faktor_core::retry::RetryPolicy::default(),
        })
        .unwrap();
        ServerDeps {
            session,
            agent,
            permissions,
            auth_token: AuthToken::generate(),
            server_password: ServerPassword::generate(),
            directory: None,
            version: "0.1.0".into(),
            fs: None,
            snapshots: None,
            chunk_rx: None,
            simulate_not_ready: false,
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

    #[tokio::test]
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
        registry.register(Arc::new(FakeProvider::with_script(
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
        )));
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
            verifier: None,
            model: "m".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "You are a test server agent.".into(),
            hooks: None,
            instructions_loader: None,
            router: None,
            budget_micro: None,
            clock: Arc::new(faktor_core::time::SystemClock),
            tool_call_mode: faktor_agent::ToolCallMode::Native,
            tool_deadline_ms: 2000,
            retry_policy: faktor_core::retry::RetryPolicy::default(),
        })
        .unwrap();
        let deps = ServerDeps {
            session: session.clone(),
            agent,
            permissions: permissions.clone(),
            auth_token: AuthToken::generate(),
            server_password: ServerPassword::generate(),
            directory: None,
            version: "0.1.0".into(),
            fs: None,
            snapshots: None,
            chunk_rx: None,
            simulate_not_ready: false,
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
        for _ in 0..100 {
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
        registry.register(Arc::new(FakeProvider::with_script(
            "fake",
            ModelCapabilities {
                tools: true,
                ..Default::default()
            },
            vec![
                faktor_provider::ScriptedResponse::Text("pong".into()),
                faktor_provider::ScriptedResponse::End,
            ],
        )));
        for p in extra_providers {
            registry.register(p);
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
            verifier: None,
            model: "m".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "You are a test server agent.".into(),
            hooks: None,
            instructions_loader: None,
            router: None,
            budget_micro: None,
            clock: Arc::new(faktor_core::time::SystemClock),
            tool_call_mode: faktor_agent::ToolCallMode::Native,
            tool_deadline_ms: 2000,
            retry_policy: faktor_core::retry::RetryPolicy::default(),
        })
        .unwrap();
        ServerDeps {
            session,
            agent,
            permissions,
            auth_token: AuthToken::generate(),
            server_password: ServerPassword::generate(),
            directory: None,
            version: "0.1.0".into(),
            fs: None,
            snapshots: None,
            chunk_rx: None,
            simulate_not_ready: false,
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
        registry.register(Arc::new(FakeProvider::with_script(
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
        )));
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
            verifier: None,
            model: "m".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "You are a test server agent.".into(),
            hooks: None,
            instructions_loader: None,
            router: None,
            budget_micro: None,
            clock: Arc::new(faktor_core::time::SystemClock),
            tool_call_mode: faktor_agent::ToolCallMode::Native,
            tool_deadline_ms: 2000,
            retry_policy: faktor_core::retry::RetryPolicy::default(),
        })
        .unwrap();
        let deps = ServerDeps {
            session: session.clone(),
            agent,
            permissions: permissions.clone(),
            auth_token: AuthToken::generate(),
            server_password: ServerPassword::generate(),
            directory: None,
            version: "0.1.0".into(),
            fs: None,
            snapshots: None,
            chunk_rx: None,
            simulate_not_ready: false,
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
            assert_eq!(
                resp.status(),
                409,
                "pty creation refuses loudly without a pty implementation"
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
        registry.register(Arc::new(FakeProvider::with_script(
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
        )));
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
            verifier: None,
            model: "m".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "You are a test server agent.".into(),
            hooks: None,
            instructions_loader: None,
            router: None,
            budget_micro: None,
            clock: Arc::new(faktor_core::time::SystemClock),
            tool_call_mode: faktor_agent::ToolCallMode::Native,
            tool_deadline_ms: 2000,
            retry_policy: faktor_core::retry::RetryPolicy::default(),
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
}
