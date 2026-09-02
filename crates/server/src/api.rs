//! The v7.5.6 wire compatibility surface (subset) over HTTP/SSE.
//!
//! Primary contract: the SDK-shaped REST surface (`/session/...`,
//! `/permission/...`, `/provider/list`, `/global/health`, `/global/event`,
//! `/question/...`, `/network/...`, `/config/...`) and the wire surface the
//! frozen v7.5.6 extension actually calls (`/session`,
//! `/session/{sessionID}`, `/session/{sessionID}/message`,
//! `/session/{sessionID}/abort`, `/session/{sessionID}/diff`,
//! `/session/{sessionID}/revert`, `/session/{sessionID}/unrevert`), all
//! behind password auth (`KILO_SERVER_PASSWORD` via
//! `Authorization: Basic base64("kilo:"+password)`, with the Bearer and
//! `x-kilo-server-password` forms retained). The old `/api/...` routes stay
//! wired as aliases; their tests must keep passing.

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{sse::Event, IntoResponse, Response, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream::Stream;
use tokio::sync::oneshot;
use tower_http::limit::RequestBodyLimitLayer;

use kilop_agent::AgentRuntime;
use kilop_core::capability::PermissionDecision;
use kilop_core::error::Error;
use kilop_core::id::SessionId;
use kilop_core::state::AgentState;
use kilop_protocol::error::ApiError;
use kilop_protocol::v756::*;
use kilop_protocol::v756::{
    mapper as wire_mapper, wire::AbortBody, wire::MessageSendRequest, wire::RevertBody,
    wire::RevertResponse, wire::SessionCreateRequest, wire::SessionCreateResponse,
    wire::SessionListResponse, wire::SessionSummary, wire::WireMessage, wire::WireMessagesPage,
};
use kilop_session::SessionManager;

use crate::auth::{check_bearer, check_password, AuthToken, ServerPassword};
use crate::global::GlobalEventBus;
use crate::permission::ChannelPermissionRequester;

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
    /// The password the frontend generated and passed via `KILO_SERVER_PASSWORD`.
    pub server_password: ServerPassword,
    /// Workspace root carried on global event envelopes.
    pub directory: Option<String>,
    pub version: String,
    /// Real workspace file service for revert/unrevert/diff (None = the wire
    /// surface refuses with an honest 409).
    pub fs: Option<Arc<kilop_fs::WorkspaceFileService>>,
    /// Real checkpoint store for revert/unrevert/diff (None = honest 409).
    pub snapshots: Option<Arc<kilop_snapshot::CheckpointStore>>,
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
            version: kilop_core::VERSION.to_string(),
            fs: None,
            snapshots: None,
        }
    }

    /// Wire the real native snapshot store so `/session/{id}/revert`,
    /// `/unrevert` and `/diff` actually restore files. Both must be provided
    /// together; with `None` the endpoints keep their honest 409.
    pub fn with_snapshots(
        mut self,
        fs: Arc<kilop_fs::WorkspaceFileService>,
        snapshots: Arc<kilop_snapshot::CheckpointStore>,
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
            protocol: kilop_core::PROTOCOL_V756.to_string(),
            pid: std::process::id() as u64,
            auth_token: self.auth_token.as_str().to_string(),
            port: addr.port(),
        }
        .to_line()
    }

    /// The frozen stdout line: `kilo server listening on http://127.0.0.1:<port>`.
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
pub async fn serve(deps: ServerDeps, port: u16) -> std::io::Result<ServerHandle> {
    // Bind first, then compute the lines (needs the bound address) and
    // finally move the deps into the router.
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    let addr = listener.local_addr()?;
    let handshake = deps.handshake_line(addr);
    let startup_line = deps.startup_line(addr);
    let bus = Arc::new(GlobalEventBus::new(
        deps.session.clone(),
        deps.directory.clone(),
    ));
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
        .route("/session/{sessionID}", get(wire_session_summary))
        .route(
            "/session/{sessionID}/message",
            post(wire_message_send).get(wire_messages_page),
        )
        .route("/session/{sessionID}/abort", post(wire_abort))
        .route("/session/{sessionID}/diff", get(wire_diff))
        .route("/session/{sessionID}/revert", post(wire_revert))
        .route("/session/{sessionID}/unrevert", post(wire_unrevert))
        .route("/session/{sessionID}/state", get(wire_session_state))
        .layer(RequestBodyLimitLayer::new(MAX_BODY_BYTES))
        .with_state(AppState {
            deps: Arc::new(deps),
            bus,
            config: Arc::new(std::sync::RwLock::new(serde_json::Value::Object(
                Default::default(),
            ))),
        });
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
}

// ------------------------------------------------------------------ handlers

async fn hello(State(state): State<AppState>) -> Response {
    Json(HelloResponse {
        ok: true,
        version: state.deps.version.clone(),
        protocol: kilop_core::PROTOCOL_V756.to_string(),
        auth_required: true,
        providers: state.deps.agent.deps().providers.ids(),
    })
    .into_response()
}

fn authed(headers: &HeaderMap, deps: &ServerDeps) -> Result<(), ApiError> {
    let authorization = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());
    let x_kilo = headers
        .get("x-kilo-server-password")
        .and_then(|v| v.to_str().ok());
    // The frozen v7.5.6 extension sends `Basic base64("kilo:"+password)` for
    // every request; the Kilo+-native `x-kilo-server-password` header and the
    // legacy per-start token keep the old clients and tests working.
    if deps.server_password.check_authorization(authorization)
        || check_password(&deps.server_password, None, x_kilo)
        || check_bearer(&deps.auth_token, authorization)
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
    if let Err(e) = authed(&headers, &state.deps) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    Json(HealthResponse {
        ok: true,
        version: state.deps.version.clone(),
        protocol: kilop_core::PROTOCOL_V756.to_string(),
    })
    .into_response()
}

async fn create_session(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<CreateSessionRequest>,
) -> Response {
    if let Err(e) = authed(&headers, &state.deps) {
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
    if let Err(e) = authed(&headers, &state.deps) {
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
    if let Err(e) = authed(&headers, &state.deps) {
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
    if let Err(e) = authed(&headers, &state.deps) {
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
    agent: &std::sync::Arc<kilop_agent::AgentRuntime>,
    session: kilop_core::id::SessionId,
    prompt: &str,
    files: &[String],
    model: Option<String>,
) -> kilop_core::Result<kilop_session::PromptReceipt> {
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
    if let Err(e) = authed(&headers, &state.deps) {
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
    if let Err(e) = authed(&headers, &state.deps) {
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
    if let Err(e) = authed(&headers, &state.deps) {
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
    if let Err(e) = authed(&headers, &state.deps) {
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
    if let Err(e) = authed(&headers, &state.deps) {
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
    if let Err(e) = authed(&headers, &state.deps) {
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
    if let Err(e) = authed(&headers, &state.deps) {
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
            Ok(v) => Some(kilop_core::id::OpId::new(v)),
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
    if let Err(e) = authed(&headers, &state.deps) {
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
    if let Err(e) = authed(&headers, &state.deps) {
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

/// The `x-kilo-directory` header value (the workspace root the extension
/// operates on). Bounded by the mapper.
fn directory_header(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("x-kilo-directory")
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
/// comes from the `x-kilo-directory` header, else `workspaceID`.
async fn wire_create_session(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<SessionCreateRequest>,
) -> Response {
    if let Err(e) = authed(&headers, &state.deps) {
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
    if let Err(e) = authed(&headers, &state.deps) {
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
    if let Err(e) = authed(&headers, &state.deps) {
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
    if let Err(e) = authed(&headers, &state.deps) {
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

/// `POST /session/{sessionID}/message` — run one turn detached, exactly like
/// the legacy prompt handler. Empty `parts` → 400. The response's
/// `message_id` is the durable message *sequence* the user prompt will
/// occupy (the row id is assigned when the turn lands; the message page and
/// the SSE stream carry the real ids). The per-message `model` override
/// APPLIES: the provider must equal the session's provider (else an honest
/// 409), and the model id is used for this turn only — the journaled
/// session row keeps its configured model.
async fn wire_message_send(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(session_id): Path<String>,
    Json(req): Json<MessageSendRequest>,
) -> Response {
    if let Err(e) = authed(&headers, &state.deps) {
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
    let message_id = match handle.proposed_message_seq() {
        Ok(seq) => seq.to_string(),
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
        Some(req.model.model_id.clone()),
    ) {
        Ok(r) => r,
        Err(e) => return api_err(&e),
    };
    Json(kilop_protocol::v756::wire::MessageSendResponse {
        message_id,
        accepted: true,
        queued: receipt.queued,
    })
    .into_response()
}

/// `GET /session/{sessionID}/message?before=&limit=` — newest-first paging,
/// mapped to the wire envelope. `before` is the internal message sequence
/// cursor (the wire message omits `seq`; documented).
#[derive(serde::Deserialize)]
struct WireMessagesQuery {
    before: Option<i64>,
    #[serde(default = "wire_default_limit")]
    limit: i64,
}

fn wire_default_limit() -> i64 {
    100
}

async fn wire_messages_page(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(session_id): Path<String>,
    Query(q): Query<WireMessagesQuery>,
) -> Response {
    if let Err(e) = authed(&headers, &state.deps) {
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
    match handle.messages_page(q.before, q.limit) {
        Ok(page) => {
            let mut messages = Vec::with_capacity(page.messages.len());
            for m in &page.messages {
                match wire_mapper::internal_message_to_wire(m) {
                    Ok(w) => messages.push(WireMessage {
                        provider_id: provider_id.clone(),
                        model_id: model_id.clone(),
                        ..w
                    }),
                    // A corrupt part row fails the page loudly (the legacy
                    // route has the same rule): never silently drop content.
                    Err(e) => return api_err(&e),
                }
            }
            Json(WireMessagesPage {
                session_id: page.session_id,
                messages,
                has_more: page.has_more,
            })
            .into_response()
        }
        Err(e) => api_err(&e),
    }
}

/// `POST /session/{sessionID}/abort` — body `{ messageID? }`.
async fn wire_abort(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(session_id): Path<String>,
    body: Option<Json<AbortBody>>,
) -> Response {
    if let Err(e) = authed(&headers, &state.deps) {
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
) -> Result<Option<kilop_store::SessionRow>, Box<Response>> {
    match state.deps.session.get_session(sid) {
        Ok(Some(handle)) => handle.row().map(Some).map_err(|e| Box::new(api_err(&e))),
        Ok(None) => Ok(None),
        Err(e) => Err(Box::new(api_err(&e))),
    }
}

fn store_err(e: &kilop_store::StoreError) -> Response {
    api_err(&Error::new(
        kilop_core::error::ErrorKind::Store,
        format!("store: {e}"),
    ))
}

/// `GET /session/{sessionID}/diff` — real unified diff of the latest
/// checkpoint's before/after contents plus the status derived from the
/// recorded existence transition (added/deleted/modified). Frozen shape:
/// 200 with `diff`/`path`/`status` all null when there is nothing to diff.
async fn wire_diff(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(session_id): Path<String>,
) -> Response {
    if let Err(e) = authed(&headers, &state.deps) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
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
    let (Some(fs), Some(snapshots)) = (&state.deps.fs, &state.deps.snapshots) else {
        // Nothing wired: the frozen diff shape with honest nulls (the
        // extension must never see a fake diff).
        return Json(serde_json::json!({ "diff": null, "path": null, "status": null }))
            .into_response();
    };
    let store = state.deps.session.store();
    let Some(root) = (match store.workspace_root(row.workspace_id) {
        Ok(r) => r,
        Err(e) => return store_err(&e),
    }) else {
        return wire_refused("diff unavailable: workspace root unknown");
    };
    let handle = match fs.open(row.workspace_id, std::path::PathBuf::from(&root)) {
        Ok(h) => h,
        Err(_) => return wire_refused("diff unavailable: workspace not openable"),
    };
    let identity = kilop_core::WorkspaceIdentity::new(
        row.workspace_id,
        kilop_core::WorktreeId::new(1),
        kilop_core::TaskId::new(1),
    );
    match snapshots.diff_latest(&handle, &identity, sid) {
        Ok(Some(result)) => Json(serde_json::json!({
            "diff": result.diff,
            "path": result.path,
            "status": result.status.as_str(),
        }))
        .into_response(),
        Ok(None) => {
            Json(serde_json::json!({ "diff": null, "path": null, "status": null })).into_response()
        }
        Err(e) => {
            if e.kind == kilop_core::error::ErrorKind::NotFound {
                return Json(serde_json::json!({ "diff": null, "path": null, "status": null }))
                    .into_response();
            }
            wire_refused(&format!("diff unavailable: {e}"))
        }
    }
}

/// The newest checkpoint row of `session` recorded at or before `message_ms`
/// (the revert/unrevert target). `None` when nothing qualifies.
fn checkpoint_before(
    store: &kilop_store::Store,
    session: SessionId,
    message_ms: i64,
) -> Result<Option<kilop_store::CheckpointRow>, Box<Response>> {
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
    workspace_id: kilop_core::WorkspaceId,
) -> Result<(kilop_fs::WorkspaceHandle, kilop_core::WorkspaceIdentity), Box<Response>> {
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
    let identity = kilop_core::WorkspaceIdentity::new(
        workspace_id,
        kilop_core::WorktreeId::new(1),
        kilop_core::TaskId::new(1),
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
    if let Err(e) = authed(&headers, &state.deps) {
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
        Ok(kilop_snapshot::RollbackOutcome::Restored { path, hash }) => Json(serde_json::json!({
            "ok": true,
            // hash is null when the rollback DELETED the file (the before
            // state was missing).
            "restored": [{"path": path, "hash": hash.map(|h| h.to_hex())}],
        }))
        .into_response(),
        Ok(kilop_snapshot::RollbackOutcome::Conflict { path, .. }) => (
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
    if let Err(e) = authed(&headers, &state.deps) {
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
        Ok(kilop_snapshot::RollbackOutcome::Restored { path, hash }) => Json(serde_json::json!({
            "ok": true,
            // hash is null when the unrevert DELETED the file (the after
            // state was missing).
            "restored": [{"path": path, "hash": hash.map(|h| h.to_hex())}],
        }))
        .into_response(),
        Ok(kilop_snapshot::RollbackOutcome::Conflict { path, .. }) => (
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

// ------------------------------------------------- question/network/config stubs
// This runtime has no question, network, or persistent config subsystems yet;
// the endpoints exist with the frozen shapes, and unknown ids are loud 404s
// (never silent success for something that does not exist).

async fn question_reply(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<QuestionReplyRequest>,
) -> Response {
    if let Err(e) = authed(&headers, &state.deps) {
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
    let e = ApiError {
        code: "not_found",
        message: format!("question {} unknown", req.question_id),
        http_status: 404,
        retryable: false,
    };
    (StatusCode::NOT_FOUND, Json(e.to_json())).into_response()
}

async fn question_list(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(e) = authed(&headers, &state.deps) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    Json(QuestionListResponse {
        questions: Vec::new(),
    })
    .into_response()
}

async fn network_reply(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<NetworkReplyRequest>,
) -> Response {
    if let Err(e) = authed(&headers, &state.deps) {
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
    let e = ApiError {
        code: "not_found",
        message: format!("network {} unknown", req.network_id),
        http_status: 404,
        retryable: false,
    };
    (StatusCode::NOT_FOUND, Json(e.to_json())).into_response()
}

async fn network_list(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(e) = authed(&headers, &state.deps) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    Json(NetworkListResponse {
        networks: Vec::new(),
    })
    .into_response()
}

async fn config_get(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(e) = authed(&headers, &state.deps) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let config = state.config.read().unwrap().clone();
    Json(ConfigGetResponse { config }).into_response()
}

async fn config_set(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ConfigSetRequest>,
) -> Response {
    if let Err(e) = authed(&headers, &state.deps) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let bytes = serde_json::to_vec(&req.config).unwrap_or_default();
    if bytes.len() > MAX_CONFIG_BYTES {
        let e = ApiError {
            code: "oversized",
            message: format!("config of {} bytes exceeds {MAX_CONFIG_BYTES}", bytes.len()),
            http_status: 413,
            retryable: false,
        };
        return (StatusCode::PAYLOAD_TOO_LARGE, Json(e.to_json())).into_response();
    }
    *state.config.write().unwrap() = req.config;
    Json(ConfigSetResponse { ok: true }).into_response()
}

async fn provider_list(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(e) = authed(&headers, &state.deps) {
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

// ------------------------------------------------------------------ SSE

async fn events(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Query(q): Query<MessagesQuery>,
) -> Response {
    if let Err(e) = authed(&headers, &state.deps) {
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
    handle: kilop_session::SessionHandle,
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
                if let Some((event, _)) = kilop_protocol::sse::project_event(&e) {
                    batch.push_back(sse_event(kilop_session::JournalFrame { seq: e.seq, event }));
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

fn sse_event(frame: kilop_session::JournalFrame) -> Event {
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
    if let Err(e) = authed(&headers, &state.deps) {
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
    let api = kilop_protocol::error::from_core(e);
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
    use kilop_core::model::ModelCapabilities;
    use kilop_provider::FakeProvider;

    #[test]
    fn handshake_line_is_frozen_shape() {
        let deps = ServerDeps {
            session: SessionManager::open(
                std::env::temp_dir().join("kp-hs-store"),
                std::env::temp_dir().join("kp-hs-cas"),
                false,
            )
            .unwrap(),
            agent: AgentRuntime::new(kilop_agent::AgentDeps {
                session: SessionManager::open(
                    std::env::temp_dir().join("kp-hs-store2"),
                    std::env::temp_dir().join("kp-hs-cas2"),
                    false,
                )
                .unwrap(),
                providers: Arc::new(kilop_provider::ProviderRegistry::new()),
                permission_requester: ChannelPermissionRequester::new(Duration::from_secs(1)),
                evidence: Arc::new(kilop_agent::NoEvidence),
                tools: Arc::new(kilop_agent::ToolRegistry::new()),
                cas: None,
                workspaces: kilop_fs::WorkspaceFileService::new(),
                edit: None,
                snapshots: None,
                sandbox: None,
                supervisor: None,
                model: "m".into(),
                compaction_model: None,
                compact_at_usage: 0.65,
                instructions: "i".into(),
                clock: Arc::new(kilop_core::time::SystemClock),
                tool_call_mode: kilop_agent::ToolCallMode::Native,
                tool_deadline_ms: 1000,
                retry_policy: kilop_core::retry::RetryPolicy::default(),
            })
            .unwrap(),
            permissions: ChannelPermissionRequester::new(Duration::from_secs(1)),
            auth_token: AuthToken::generate(),
            server_password: ServerPassword::generate(),
            directory: None,
            version: "0.1.0".into(),
            fs: None,
            snapshots: None,
        };
        let addr: SocketAddr = "127.0.0.1:45678".parse().unwrap();
        let line = deps.handshake_line(addr);
        assert!(line.starts_with("KILO_PLUS_HANDSHAKE "));
        let hs = Handshake::from_line(&line).unwrap();
        assert_eq!(hs.protocol, "v756");
        assert_eq!(hs.port, 45678);
        assert_eq!(hs.auth_token, deps.auth_token.as_str());
        // The frozen stdout contract is the startup line, and the password
        // never appears in it (no token on stdout).
        let startup = deps.startup_line(addr);
        assert_eq!(startup, "kilo server listening on http://127.0.0.1:45678");
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
        let mut registry = kilop_provider::ProviderRegistry::new();
        registry.register(Arc::new(FakeProvider::with_script(
            "fake",
            ModelCapabilities {
                tools: true,
                ..Default::default()
            },
            vec![
                kilop_provider::ScriptedResponse::ToolCall {
                    id: "c1".into(),
                    name: "echo".into(),
                    input: serde_json::json!({"x": 1}),
                },
                kilop_provider::ScriptedResponse::End,
            ],
        )));
        let session =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let permissions = ChannelPermissionRequester::new(Duration::from_secs(5));
        let mut tools = kilop_agent::ToolRegistry::new();
        tools.register(kilop_agent::Tool {
            name: "echo".into(),
            description: "d".into(),
            input_schema: serde_json::json!({}),
            resource_class: kilop_core::resource::ResourceClass::Cpu,
            capability: None,
            recovery_hint: kilop_agent::RecoveryHint::Idempotent,
            path_args: vec![],
            execute: Arc::new(|_ctx, _args| {
                Box::pin(async move { Ok(kilop_agent::ToolOutcome::default()) })
            }),
        });
        let agent = AgentRuntime::new(kilop_agent::AgentDeps {
            session: session.clone(),
            providers: Arc::new(registry),
            permission_requester: permissions.clone(),
            evidence: Arc::new(kilop_agent::NoEvidence),
            tools: Arc::new(tools),
            cas: None,
            workspaces: kilop_fs::WorkspaceFileService::new(),
            edit: None,
            snapshots: None,
            sandbox: None,
            supervisor: None,
            model: "m".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "You are a test server agent.".into(),
            clock: Arc::new(kilop_core::time::SystemClock),
            tool_call_mode: kilop_agent::ToolCallMode::Native,
            tool_deadline_ms: 2000,
            retry_policy: kilop_core::retry::RetryPolicy::default(),
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
            if matches!(state, kilop_core::state::AgentState::ReadyForNextTurn) {
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
        ];
        for (method, path, body) in cases {
            let resp = if *method == "get" {
                client.get(format!("{base}{path}")).send().await.unwrap()
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
            .header("x-kilo-server-password", "wrong-password")
            .json(&serde_json::json!({"provider": "fake", "model": "m"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);

        // The password works in all three header forms.
        let resp = client
            .post(format!("{base}/session/create"))
            .header("x-kilo-server-password", pw.as_str())
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

        // POST /session: the x-kilo-directory header wins over workspaceID,
        // and the model.providerID drives the provider.
        let resp = basic(
            client
                .post(format!("{base}/session"))
                .header("x-kilo-directory", "/tmp")
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
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["accepted"], true);
        assert_eq!(body["queued"], false);
        assert!(body["messageID"].as_str().unwrap().parse::<u64>().is_ok());

        // The turn converges, then the wire messages page reflects it.
        let mut done = false;
        for _ in 0..100 {
            if let Ok(Some(h)) = session.get_session(sid_parsed) {
                if matches!(
                    h.state(),
                    Ok(kilop_core::state::AgentState::ReadyForNextTurn)
                ) {
                    done = true;
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(done, "turn must complete");
        let resp = basic(client.get(format!("{base}/session/{sid}/message?limit=10")))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let page: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(page["sessionID"], sid);
        assert!(page["hasMore"].is_boolean());
        let messages = page["messages"].as_array().unwrap();
        assert!(messages.len() >= 2, "{page}");
        // Wire field names: messageID, createdMs, providerID/modelID filled.
        let first = &messages[0];
        assert!(first["messageID"].as_str().is_some());
        assert!(first["createdMs"].as_i64().unwrap() > 0);
        assert_eq!(first["providerID"], "fake");
        assert_eq!(first["modelID"], "m");
        // The prompt text part survived the roundtrip as a text part.
        let text = first["parts"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["type"] == "text")
            .map(|p| p["text"].as_str().unwrap_or(""))
            .unwrap_or("");
        assert!(!text.is_empty());
        // Paging: before=1 (seq 1 exists) must not error; unknown cursors
        // are the server's clamp, never an error.
        let resp = basic(client.get(format!("{base}/session/{sid}/message?before=1&limit=1")))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

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
        let mut registry = kilop_provider::ProviderRegistry::new();
        registry.register(provider);
        let session = SessionManager::open(root.join("store"), root.join("cas"), true).unwrap();
        let permissions = ChannelPermissionRequester::new(Duration::from_secs(5));
        let agent = AgentRuntime::new(kilop_agent::AgentDeps {
            session: session.clone(),
            providers: Arc::new(registry),
            permission_requester: permissions.clone(),
            evidence: Arc::new(kilop_agent::NoEvidence),
            tools: Arc::new(kilop_agent::ToolRegistry::new()),
            cas: None,
            workspaces: kilop_fs::WorkspaceFileService::new(),
            edit: None,
            snapshots: None,
            sandbox: None,
            supervisor: None,
            model: "m".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "You are a test server agent.".into(),
            clock: Arc::new(kilop_core::time::SystemClock),
            tool_call_mode: kilop_agent::ToolCallMode::Native,
            tool_deadline_ms: 2000,
            retry_policy: kilop_core::retry::RetryPolicy::default(),
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
                kilop_provider::ScriptedResponse::Text("pong".into()),
                kilop_provider::ScriptedResponse::End,
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
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["accepted"], true);
        assert_eq!(body["queued"], false);
        assert!(body["messageID"].as_str().unwrap().parse::<u64>().is_ok());

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
                kilop_provider::ScriptedResponse::Text("pong".into()),
                kilop_provider::ScriptedResponse::End,
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
                kilop_provider::ScriptedResponse::Text("pong".into()),
                kilop_provider::ScriptedResponse::End,
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

        // diff: frozen shape, honest null (200).
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
            serde_json::json!({"diff": null, "path": null, "status": null}),
            "frozen shape with nulls when nothing is wired/diffable"
        );

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
        Arc<kilop_snapshot::CheckpointStore>,
        Arc<kilop_fs::WorkspaceFileService>,
    ) {
        let deps = test_deps(root);
        let fs = kilop_fs::WorkspaceFileService::new();
        let snapshots = Arc::new(kilop_snapshot::CheckpointStore::new(
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
            .header("x-kilo-directory", ws_root.to_str().unwrap())
            .json(&serde_json::json!({"model": {"id": "m", "providerID": "fake"}}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid: u64 = created["sessionID"].as_str().unwrap().parse().unwrap();
        let session = kilop_core::id::SessionId::new(sid);

        // Record a checkpoint exactly like the edit engine would: original
        // content captured, file edited, after-content stored in the CAS.
        let file = ws_root.join("notes.txt");
        std::fs::write(&file, b"original\n").unwrap();
        let before = snapshots
            .before_write(session, "notes.txt", b"original\n")
            .unwrap();
        let ws_handle = fs
            .open(kilop_core::WorkspaceId::new(sid), ws_root.clone())
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
            .header("x-kilo-directory", ws_root.to_str().unwrap())
            .json(&serde_json::json!({"model": {"id": "m", "providerID": "fake"}}))
            .send()
            .await
            .unwrap();
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid: u64 = created["sessionID"].as_str().unwrap().parse().unwrap();
        let session = kilop_core::id::SessionId::new(sid);

        let file = ws_root.join("notes.txt");
        std::fs::write(&file, b"original\n").unwrap();
        let before = snapshots
            .before_write(session, "notes.txt", b"original\n")
            .unwrap();
        let ws_handle = fs
            .open(kilop_core::WorkspaceId::new(sid), ws_root.clone())
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
            .header("x-kilo-directory", ws_root.to_str().unwrap())
            .json(&serde_json::json!({"model": {"id": "m", "providerID": "fake"}}))
            .send()
            .await
            .unwrap();
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid: u64 = created["sessionID"].as_str().unwrap().parse().unwrap();
        let session = kilop_core::id::SessionId::new(sid);

        let file = ws_root.join("notes.txt");
        std::fs::write(&file, b"original\n").unwrap();
        let before = snapshots
            .before_write(session, "notes.txt", b"original\n")
            .unwrap();
        let ws_handle = fs
            .open(kilop_core::WorkspaceId::new(sid), ws_root.clone())
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
    async fn diff_returns_unified_text_via_wire() {
        let dir = tempfile::tempdir().unwrap();
        let ws_root = dir.path().join("ws");
        std::fs::create_dir_all(&ws_root).unwrap();
        let (deps, snapshots, fs) = wire_snapshot_deps(dir.path());
        let pw = deps.server_password.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        let resp = client
            .post(format!("{base}/session"))
            .basic_auth("kilo", Some(pw.as_str()))
            .header("x-kilo-directory", ws_root.to_str().unwrap())
            .json(&serde_json::json!({"model": {"id": "m", "providerID": "fake"}}))
            .send()
            .await
            .unwrap();
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid: u64 = created["sessionID"].as_str().unwrap().parse().unwrap();
        let session = kilop_core::id::SessionId::new(sid);

        // No checkpoints yet: the frozen null shape.
        let resp = client
            .get(format!("{base}/session/{sid}/diff"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(
            resp.json::<serde_json::Value>().await.unwrap(),
            serde_json::json!({"diff": null, "path": null, "status": null})
        );

        let before_text = "line1\nline2\nline3\nline4\nold\nline6\nline7\n";
        let after_text = "line1\nline2\nline3\nline4\nnew\nline6\nline7\n";
        let file = ws_root.join("f.txt");
        std::fs::write(&file, before_text).unwrap();
        let before = snapshots
            .before_write(session, "f.txt", before_text.as_bytes())
            .unwrap();
        let ws_handle = fs
            .open(kilop_core::WorkspaceId::new(sid), ws_root.clone())
            .unwrap();
        let after = ws_handle
            .write_atomic(std::path::Path::new("f.txt"), after_text.as_bytes())
            .unwrap();
        snapshots
            .after_write(session, "f.txt", before, after, 0, after_text.as_bytes())
            .unwrap();

        let resp = client
            .get(format!("{base}/session/{sid}/diff"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["path"], "f.txt");
        assert_eq!(
            body["status"], "modified",
            "the wire diff must project the FileState transition status"
        );
        let diff = body["diff"].as_str().unwrap();
        assert!(diff.lines().any(|l| l == "-old"), "removal missing: {diff}");
        assert!(
            diff.lines().any(|l| l == "+new"),
            "addition missing: {diff}"
        );
        assert!(
            diff.lines().any(|l| l == " line2"),
            "context missing: {diff}"
        );

        // An empty-file CREATION row projects status "added": the wire diff
        // must never mistake it for a no-op (hash("")==hash("")).
        let empty_hash = snapshots
            .before_write(session, "created-empty.txt", b"")
            .unwrap();
        let file2 = ws_root.join("created-empty.txt");
        let created_id = snapshots
            .record_change(
                session,
                "created-empty.txt",
                kilop_snapshot::FileState::missing(),
                None,
                kilop_snapshot::FileState::existing(empty_hash),
                Some(b""),
            )
            .unwrap();
        let _ = created_id;
        std::fs::write(&file2, b"").unwrap();
        let resp = client
            .get(format!("{base}/session/{sid}/diff"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["path"], "created-empty.txt");
        assert_eq!(body["status"], "added");

        // A deletion row projects status "deleted" with a pure-removal diff.
        let deleted_id = snapshots
            .record_change(
                session,
                "f.txt",
                kilop_snapshot::FileState::existing(after),
                None,
                kilop_snapshot::FileState::missing(),
                None,
            )
            .unwrap();
        let _ = deleted_id;
        let resp = client
            .get(format!("{base}/session/{sid}/diff"))
            .basic_auth("kilo", Some(pw.as_str()))
            .send()
            .await
            .unwrap();
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["path"], "f.txt");
        assert_eq!(body["status"], "deleted");
        let diff = body["diff"].as_str().unwrap();
        assert!(
            diff.lines().any(|l| l == "-new"),
            "deletion must diff as removals: {diff}"
        );
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
            .header("x-kilo-directory", ws_root.to_str().unwrap())
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
            .header("x-kilo-server-password", pw.as_str())
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
            .header("x-kilo-server-password", pw.as_str())
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
                .header("x-kilo-server-password", pw.as_str())
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
            .header("x-kilo-server-password", pw.as_str())
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
            .header("x-kilo-server-password", pw.as_str())
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
            .header("x-kilo-server-password", pw.as_str())
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
                    .header("x-kilo-server-password", pw.as_str())
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
            .header("x-kilo-server-password", pw.as_str())
            .json(&serde_json::json!({"session_id": "999999", "prompt": "x"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);
        let resp = client
            .post(format!("{base}/session/abort"))
            .header("x-kilo-server-password", pw.as_str())
            .json(&serde_json::json!({"session_id": "999999"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);

        // Malformed ids and empty prompts are 400s.
        let resp = client
            .post(format!("{base}/session/prompt"))
            .header("x-kilo-server-password", pw.as_str())
            .json(&serde_json::json!({"session_id": "0", "prompt": "x"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);
        let resp = client
            .post(format!("{base}/session/prompt"))
            .header("x-kilo-server-password", pw.as_str())
            .json(&serde_json::json!({"session_id": sid, "prompt": "   "}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);
        // Unknown fields in the SDK body are protocol drift (422 from the
        // deny_unknown_fields extraction gate).
        let resp = client
            .post(format!("{base}/session/prompt"))
            .header("x-kilo-server-password", pw.as_str())
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
            .header("x-kilo-server-password", pw.as_str())
            .send()
            .await
            .unwrap()
            .bytes_stream();

        let resp = client
            .post(format!("{base}/session/create"))
            .header("x-kilo-server-password", pw.as_str())
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
            .header("x-kilo-server-password", pw.as_str())
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
            .header("x-kilo-server-password", pw.as_str())
            .send()
            .await
            .unwrap()
            .bytes_stream();
        client
            .post(format!("{base}/session/prompt"))
            .header("x-kilo-server-password", pw.as_str())
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
            .header("x-kilo-server-password", pw.as_str())
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
            .header("x-kilo-server-password", pw.as_str())
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
        let mut registry = kilop_provider::ProviderRegistry::new();
        registry.register(Arc::new(FakeProvider::with_script(
            "fake",
            ModelCapabilities {
                tools: true,
                ..Default::default()
            },
            vec![
                kilop_provider::ScriptedResponse::ToolCall {
                    id: "c1".into(),
                    name: "echo".into(),
                    input: serde_json::json!({"x": 1}),
                },
                kilop_provider::ScriptedResponse::End,
            ],
        )));
        let session =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let permissions = ChannelPermissionRequester::new(Duration::from_secs(5));
        let mut tools = kilop_agent::ToolRegistry::new();
        tools.register(kilop_agent::Tool {
            name: "echo".into(),
            description: "d".into(),
            input_schema: serde_json::json!({}),
            resource_class: kilop_core::resource::ResourceClass::Cpu,
            capability: None,
            recovery_hint: kilop_agent::RecoveryHint::Idempotent,
            path_args: vec![],
            execute: Arc::new(|_ctx, _args| {
                Box::pin(async move { Ok(kilop_agent::ToolOutcome::default()) })
            }),
        });
        let agent = AgentRuntime::new(kilop_agent::AgentDeps {
            session: session.clone(),
            providers: Arc::new(registry),
            permission_requester: permissions.clone(),
            evidence: Arc::new(kilop_agent::NoEvidence),
            tools: Arc::new(tools),
            cas: None,
            workspaces: kilop_fs::WorkspaceFileService::new(),
            edit: None,
            snapshots: None,
            sandbox: None,
            supervisor: None,
            model: "m".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "You are a test server agent.".into(),
            clock: Arc::new(kilop_core::time::SystemClock),
            tool_call_mode: kilop_agent::ToolCallMode::Native,
            tool_deadline_ms: 2000,
            retry_policy: kilop_core::retry::RetryPolicy::default(),
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
        };
        let pw = deps.server_password.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        let resp = client
            .post(format!("{base}/session/create"))
            .header("x-kilo-server-password", pw.as_str())
            .json(&serde_json::json!({"provider": "fake", "model": "m"}))
            .send()
            .await
            .unwrap();
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid = created["id"].as_str().unwrap().to_string();
        client
            .post(format!("{base}/session/prompt"))
            .header("x-kilo-server-password", pw.as_str())
            .json(&serde_json::json!({"session_id": sid, "prompt": "use tools"}))
            .send()
            .await
            .unwrap();

        // The permission surfaces in /permission/list with its session.
        let mut pid = None;
        for _ in 0..100 {
            let resp = client
                .get(format!("{base}/permission/list?session_id={sid}"))
                .header("x-kilo-server-password", pw.as_str())
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
            .header("x-kilo-server-password", pw.as_str())
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
            if matches!(state, kilop_core::state::AgentState::ReadyForNextTurn) {
                done = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(done, "turn must finish after permission grant");

        // The resolved permission is gone from the list.
        let resp = client
            .get(format!("{base}/permission/list"))
            .header("x-kilo-server-password", pw.as_str())
            .send()
            .await
            .unwrap();
        let list: serde_json::Value = resp.json().await.unwrap();
        assert!(list["permissions"].as_array().unwrap().is_empty());

        // Double reply → 409; malformed ids/decisions → 400.
        let resp = client
            .post(format!("{base}/permission/reply"))
            .header("x-kilo-server-password", pw.as_str())
            .json(&serde_json::json!({"permission_id": pid, "decision": "allow"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 409);
        let resp = client
            .post(format!("{base}/permission/reply"))
            .header("x-kilo-server-password", pw.as_str())
            .json(&serde_json::json!({"permission_id": "bogus", "decision": "allow"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);
        let resp = client
            .post(format!("{base}/permission/reply"))
            .header("x-kilo-server-password", pw.as_str())
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
            .header("x-kilo-server-password", pw.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["questions"], serde_json::json!([]));
        let resp = client
            .post(format!("{base}/question/reply"))
            .header("x-kilo-server-password", pw.as_str())
            .json(&serde_json::json!({"question_id": "q1", "decision": "allow"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);
        let resp = client
            .post(format!("{base}/question/reply"))
            .header("x-kilo-server-password", pw.as_str())
            .json(&serde_json::json!({"question_id": "", "decision": "allow"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);

        // Networks: same shapes.
        let resp = client
            .get(format!("{base}/network/list"))
            .header("x-kilo-server-password", pw.as_str())
            .send()
            .await
            .unwrap();
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["networks"], serde_json::json!([]));
        let resp = client
            .post(format!("{base}/network/reply"))
            .header("x-kilo-server-password", pw.as_str())
            .json(&serde_json::json!({"network_id": "n1", "decision": "deny"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);

        // Config: set → get roundtrip.
        let resp = client
            .post(format!("{base}/config/set"))
            .header("x-kilo-server-password", pw.as_str())
            .json(&serde_json::json!({"config": {"model": "qwen3.8", "nested": {"a": [1, 2]}}}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let resp = client
            .get(format!("{base}/config/get"))
            .header("x-kilo-server-password", pw.as_str())
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
            .header("x-kilo-server-password", pw.as_str())
            .json(&serde_json::json!({"config": {"blob": big}}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 413);

        // Config is still the previous value after the rejection.
        let resp = client
            .get(format!("{base}/config/get"))
            .header("x-kilo-server-password", pw.as_str())
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
        extra_providers: Vec<Arc<dyn kilop_provider::Provider>>,
    ) -> ServerDeps {
        let mut registry = kilop_provider::ProviderRegistry::new();
        registry.register(Arc::new(FakeProvider::with_script(
            "fake",
            ModelCapabilities {
                tools: true,
                ..Default::default()
            },
            vec![
                kilop_provider::ScriptedResponse::Text("pong".into()),
                kilop_provider::ScriptedResponse::End,
            ],
        )));
        for p in extra_providers {
            registry.register(p);
        }
        let session = SessionManager::open(root.join("store"), root.join("cas"), true).unwrap();
        let permissions = ChannelPermissionRequester::new(Duration::from_secs(5));
        let agent = AgentRuntime::new(kilop_agent::AgentDeps {
            session: session.clone(),
            providers: Arc::new(registry),
            permission_requester: permissions.clone(),
            evidence: Arc::new(kilop_agent::NoEvidence),
            tools: Arc::new(kilop_agent::ToolRegistry::new()),
            cas: None,
            workspaces: kilop_fs::WorkspaceFileService::new(),
            edit: None,
            snapshots: None,
            sandbox: None,
            supervisor: None,
            model: "m".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "You are a test server agent.".into(),
            clock: Arc::new(kilop_core::time::SystemClock),
            tool_call_mode: kilop_agent::ToolCallMode::Native,
            tool_deadline_ms: 2000,
            retry_policy: kilop_core::retry::RetryPolicy::default(),
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
            kilop_core::state::AgentState::Preparing,
            "a queued-prompt kill must not touch the state machine"
        );
        assert_eq!(session.queued_prompt_count().unwrap(), 0);
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
            kilop_core::model::ModelCapabilities {
                context: 128_000,
                max_output: 16_384,
                tools: true,
                ..Default::default()
            },
        );
        let openai = kilop_openai::OpenAiProvider::build(kilop_openai::OpenAiConfig {
            base_url: "http://127.0.0.1:1/v1".into(),
            api_key: None,
            family: kilop_openai::OpenAiFamily::Chat,
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
}
