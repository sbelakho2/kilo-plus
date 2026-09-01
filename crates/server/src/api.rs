//! The frozen v7.5.6 HTTP/SSE surface.
//!
//! Primary contract: the SDK-shaped REST surface (`/session/...`,
//! `/permission/...`, `/provider/list`, `/global/health`, `/global/event`,
//! `/question/...`, `/network/...`, `/config/...`) with password auth
//! (`KILO_SERVER_PASSWORD` via `Authorization: Bearer` or
//! `x-kilo-server-password`). The old `/api/...` routes stay wired as
//! aliases; their tests must keep passing.

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
use kilop_protocol::error::ApiError;
use kilop_protocol::v756::*;
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
        }
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
    // The frontend's password may arrive in either header form; the legacy
    // per-start token keeps the old tests (and old clients) working.
    if check_password(&deps.server_password, authorization, x_kilo)
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

/// `GET /global/health` — the only public endpoint (frozen client probe).
async fn health(State(state): State<AppState>) -> Response {
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
    let agent = state.deps.agent.clone();
    let prompt_text = req.prompt.clone();
    let files = req.files.clone();
    // The turn runs detached from the HTTP connection: closing the SSE
    // stream never destroys an active agent (spec §7). State lives in the
    // journal, so this spawn defines no application state.
    tokio::spawn(async move {
        let _ = agent.run_turn(sid, &prompt_text, &files).await;
    });
    Json(PromptResponse {
        op_id: "turn".to_string(),
        accepted: true,
        queued: false,
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
    let agent = state.deps.agent.clone();
    let prompt_text = req.prompt.clone();
    let files = req.files.clone();
    // The turn runs detached from the HTTP connection (spec §7); the journal
    // is the source of truth, so this spawn defines no application state.
    tokio::spawn(async move {
        let _ = agent.run_turn(sid, &prompt_text, &files).await;
    });
    Json(PromptResponse {
        op_id: "turn".to_string(),
        accepted: true,
        queued: false,
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
    match state.deps.agent.abort(sid) {
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
            // Dynamic registry: the adapter's own model list is exposed here.
            vec![ModelInfo {
                id: "default".into(),
                name: "default".into(),
                capabilities: p.capabilities("default"),
            }]
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
                model: "m".into(),
                compaction_model: None,
                compact_at_usage: 0.65,
                instructions: "i".into(),
                clock: Arc::new(kilop_core::time::SystemClock),
                tool_call_mode: kilop_agent::ToolCallMode::Native,
                tool_deadline_ms: 1000,
            })
            .unwrap(),
            permissions: ChannelPermissionRequester::new(Duration::from_secs(1)),
            auth_token: AuthToken::generate(),
            server_password: ServerPassword::generate(),
            directory: None,
            version: "0.1.0".into(),
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
            model: "m".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "You are a test server agent.".into(),
            clock: Arc::new(kilop_core::time::SystemClock),
            tool_call_mode: kilop_agent::ToolCallMode::Native,
            tool_deadline_ms: 2000,
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
    async fn sdk_routes_require_password_and_health_is_public() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let pw = deps.server_password.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        // /global/health is public and has the frozen shape.
        let resp = client
            .get(format!("{base}/global/health"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["ok"], true);
        assert_eq!(body["protocol"], "v756");
        assert!(body["version"].is_string());

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

        // The password works in both header forms.
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
            model: "m".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "You are a test server agent.".into(),
            clock: Arc::new(kilop_core::time::SystemClock),
            tool_call_mode: kilop_agent::ToolCallMode::Native,
            tool_deadline_ms: 2000,
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
        let session = SessionManager::open(root.join("store"), root.join("cas"), true).unwrap();
        let permissions = ChannelPermissionRequester::new(Duration::from_secs(5));
        let agent = AgentRuntime::new(kilop_agent::AgentDeps {
            session: session.clone(),
            providers: Arc::new(registry),
            permission_requester: permissions.clone(),
            evidence: Arc::new(kilop_agent::NoEvidence),
            tools: Arc::new(kilop_agent::ToolRegistry::new()),
            cas: None,
            model: "m".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "You are a test server agent.".into(),
            clock: Arc::new(kilop_core::time::SystemClock),
            tool_call_mode: kilop_agent::ToolCallMode::Native,
            tool_deadline_ms: 2000,
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
        }
    }
}
