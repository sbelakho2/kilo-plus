//! SDK-shaped legacy REST surface (`/session/...`, `/permission/...`,
//! `/global/...`, `/question/...`, `/network/...`, `/config/...`, PTY and auth).

use crate::auth::ServerPassword;
use crate::global::GlobalEventBus;
use crate::permission::PendingPermission;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{sse::Event, IntoResponse, Response, Sse};
use axum::Json;
use faktor_core::capability::PermissionDecision;
use faktor_protocol::error::ApiError;
use futures_util::stream::Stream;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use super::submit_and_run;
use crate::api::{AppState, ServerDeps};
use crate::native::{
    api_err, api_error_json, authed, exec_error_response, not_found, parse_session_id,
    wire_refused, wire_status,
};
use faktor_protocol::v756::*;

pub(crate) const MAX_CONFIG_BYTES: usize = 1024 * 1024;

pub(crate) const HEARTBEAT_SECS: u64 = 15;

pub(crate) const POLL_INTERVAL_MS: u64 = 100;

/// Journal catch-up page: one bounded page of SSE frames per poll, so a
/// reconnect against a huge journal never loads it all into memory at once.
pub(crate) const EVENT_CATCHUP_PAGE: u64 = 256;

pub(crate) async fn hello(State(state): State<AppState>) -> Response {
    Json(HelloResponse {
        ok: true,
        version: state.deps.version.clone(),
        protocol: faktor_core::PROTOCOL_V756.to_string(),
        auth_required: true,
        providers: state.deps.agent.deps().providers.ids(),
    })
    .into_response()
}

/// `GET /global/health` — auth-required (the frozen v7.5.6 client
/// authenticates every request, this one included).
pub(crate) async fn health(State(state): State<AppState>, headers: HeaderMap) -> Response {
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

pub(crate) async fn create_session(
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

pub(crate) async fn list_sessions(State(state): State<AppState>, headers: HeaderMap) -> Response {
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

pub(crate) async fn session_state(
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

pub(crate) async fn messages(
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

pub(crate) async fn prompt(
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
    let receipt = match submit_and_run(&state, sid, &prompt_text, &files, None).await {
        Ok(r) => r,
        Err(e) => return exec_error_response(&e),
    };
    Json(PromptResponse {
        op_id: receipt.op_id.to_string(),
        accepted: true,
        queued: receipt.queued,
    })
    .into_response()
}

pub(crate) async fn abort(
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

pub(crate) async fn resolve_permission(
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
pub(crate) async fn permission_reply(
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

pub(crate) fn resolve_permission_body(
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
pub(crate) async fn permission_list(
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
pub(crate) async fn sdk_prompt(
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
    let receipt = match submit_and_run(&state, sid, &prompt_text, &files, None).await {
        Ok(r) => r,
        Err(e) => return exec_error_response(&e),
    };
    Json(PromptResponse {
        op_id: receipt.op_id.to_string(),
        accepted: true,
        queued: receipt.queued,
    })
    .into_response()
}

/// `POST /session/abort` — session_id in the body.
pub(crate) async fn sdk_abort(
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
pub(crate) async fn sdk_messages(
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
pub(crate) async fn sdk_session_state(
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
pub(crate) fn directory_header(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("x-faktor-directory")
        .and_then(|v| v.to_str().ok())
}

/// `POST /pty/create` — spawn a session-scoped interactive terminal.
/// Body: {command, args?, cwd?, rows?, cols?}. Returns {pty_id, pid}.
/// Non-Unix platforms refuse honestly (ConPTY/Job Objects are the declared
/// platform blocker).
pub(crate) async fn pty_create(
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
        env: faktor_pty::EnvSpec::default_baseline(),
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
pub(crate) async fn pty_update(
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
pub(crate) async fn pty_remove(
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
pub(crate) async fn pty_output(
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

/// `POST /global/dispose` and `POST /instance/dispose` — stop everything:
/// every supervised process owned by a session is killed via the agent
/// (which owns the supervisor), then each session is durably ended
/// (SessionEnded journal event + lifecycle Closed). Honest refusal with the
/// first failing session when any cannot end.
pub(crate) async fn dispose_all_sessions(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Response {
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
pub(crate) async fn instance_reload(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    match state.deps.agent.recover() {
        Ok(_reports) => Json(OkResponse { ok: true }).into_response(),
        Err(e) => api_err(&e),
    }
}

/// Upper bound on an `auth.set` password (bounded everything).
pub(crate) const MAX_AUTH_PASSWORD_BYTES: usize = 1024;

/// `POST /auth/set` — rotate the server password. `password` absent rotates
/// to a fresh random secret; either way the response carries the new
/// effective secret so the client can keep authenticating. Every other
/// endpoint immediately checks the new secret (old credentials → 401).
pub(crate) async fn auth_set(
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
pub(crate) async fn auth_remove(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    *state.auth.write().unwrap() = None;
    Json(OkResponse { ok: true }).into_response()
}

/// The frozen capability tag a network permission carries.
pub(crate) const NETWORK_CAPABILITY: &str = "network";

pub(crate) fn pending_permission_json(p: &PendingPermission) -> serde_json::Value {
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
pub(crate) fn resolve_pending_permission(
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

pub(crate) fn class_kind(class: &str) -> &'static str {
    if class == "network" {
        "network"
    } else {
        "question"
    }
}

pub(crate) async fn question_reply(
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
pub(crate) async fn question_reject(
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

pub(crate) async fn question_list(
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

pub(crate) async fn network_reply(
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
pub(crate) async fn network_reject(
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

pub(crate) async fn network_list(
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

pub(crate) async fn config_get(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let config = state.config.read().unwrap().clone();
    Json(ConfigGetResponse { config }).into_response()
}

/// The daemon-editable top-level config keys (`config.update` allowlist):
/// provider configuration is out of scope by design — only the model,
/// the compaction threshold and the system instructions may be applied.
pub(crate) const CONFIG_EDITABLE_KEYS: [&str; 3] = ["model", "compact_at_usage", "instructions"];

/// `POST /config/set` — the SDK full-replacement form (kept for the old
/// tests/clients): the whole config view is replaced, bounded.
pub(crate) async fn config_set(
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

pub(crate) fn config_oversized(bytes: usize) -> Response {
    let e = ApiError {
        code: "oversized",
        message: format!("config of {bytes} bytes exceeds {MAX_CONFIG_BYTES}"),
        http_status: 413,
        retryable: false,
    };
    (StatusCode::PAYLOAD_TOO_LARGE, Json(e.to_json())).into_response()
}

pub(crate) fn config_must_be_object(config: &serde_json::Value) -> Option<Response> {
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
pub(crate) async fn config_update(
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
pub(crate) async fn config_warnings(State(state): State<AppState>, headers: HeaderMap) -> Response {
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
pub(crate) async fn config_overlay(
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
pub(crate) async fn config_overlay_update(
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

pub(crate) async fn provider_list(State(state): State<AppState>, headers: HeaderMap) -> Response {
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

pub(crate) async fn events(
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

pub(crate) fn journal_stream(
    handle: faktor_session::SessionHandle,
    cursor: i64,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> + Send + 'static {
    // Journal catch-up is PAGED (bounded everything): each poll loads at
    // most EVENT_CATCHUP_PAGE frames into the in-memory queue, so a
    // reconnect against a huge journal can never balloon RAM. Every frame
    // carries its `id:` sequence — the resume cursor — and the stream
    // simply continues across pages, so "more pages" is implicit and
    // deterministic: no duplicate and no gap on replay from any cursor.
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
            // seq > cursor (cursor 0 = everything from seq 1). One bounded
            // page per poll; the next poll continues exactly after it.
            let events = handle
                .events_range(cursor.saturating_add(1) as u64, Some(EVENT_CATCHUP_PAGE))
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

pub(crate) fn sse_event(frame: faktor_session::JournalFrame) -> Event {
    let seq = frame.seq.raw();
    let json = serde_json::to_string(&frame.event).unwrap_or_else(|_| "{}".into());
    Event::default()
        .event(frame.event.event_type())
        .id(seq.to_string())
        .data(json)
}

pub(crate) fn sse_event_heartbeat() -> Event {
    Event::default().event("heartbeat").data("{}")
}

// ------------------------------------------------------------------ global SSE
// `GET /global/event?after=<n>` streams GlobalEvent envelopes as
// `id: <n>\ndata: <json>\n\n` frames (no `event:` field — the payload's
// `type` carries the discriminator). `after` is the resume cursor; oversized
// values are clamped to what the bounded ring can serve.

pub(crate) async fn global_events(
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

pub(crate) fn global_stream(
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

pub(crate) fn global_frame(id: u64, ge: GlobalEvent) -> Event {
    let json = serde_json::to_string(&ge).unwrap_or_else(|_| "{}".into());
    Event::default().id(id.to_string()).data(json)
}
