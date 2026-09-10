//! Faktor Native Protocol v1 handlers (docs/native-protocol.md).
//!
//! Split out of the `api` monolith (audits 81-83/94). Native modules depend
//! only on core/runtime crates and on this module's shared glue; they never
//! import the v7.5.6 compatibility surface (`crate::compat`).

use crate::auth::{check_bearer, check_password};
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use faktor_core::error::Error;
use faktor_core::id::SessionId;
use faktor_core::state::AgentState;
use faktor_protocol::error::ApiError;

use crate::api::{AppState, ServerDeps};

pub(crate) mod agents;
pub(crate) mod evidence;
pub(crate) mod models;
pub(crate) mod semantic;
pub(crate) mod session;
pub(crate) mod task;
pub(crate) mod terminal;
pub(crate) mod usage;
pub(crate) mod verification;

pub(crate) use agents::*;
pub(crate) use evidence::*;
pub(crate) use models::*;
pub(crate) use semantic::*;
pub(crate) use session::*;
pub(crate) use task::*;
pub(crate) use terminal::*;
pub(crate) use usage::*;
pub(crate) use verification::*;

pub(crate) fn authed(headers: &HeaderMap, state: &AppState) -> Result<(), ApiError> {
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

pub(crate) fn parse_session_id(id: &str) -> Result<SessionId, ApiError> {
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

/// The snake_case agent-state tag for machine-readable summaries.
pub(crate) fn agent_state_tag(s: AgentState) -> String {
    serde_json::to_string(&s)
        .unwrap_or_else(|_| "unknown".into())
        .trim_matches('"')
        .to_string()
}

pub(crate) fn api_error_json(e: &faktor_core::error::Error) -> serde_json::Value {
    serde_json::json!({ "ok": false, "code": format!("{:?}", e.kind).to_lowercase(), "message": e.message })
}

// ------------------------------------------------------------ disposal & auth

pub(crate) fn wire_refused(message: &str) -> Response {
    // Same frozen body as the wire `RevertResponse { ok: false, message }`
    // (the field always serializes here), without importing the v7.5.6 DTO
    // into the native layer.
    (
        StatusCode::CONFLICT,
        Json(serde_json::json!({ "ok": false, "message": message })),
    )
        .into_response()
}

pub(crate) fn not_found(message: &str) -> ApiError {
    ApiError {
        code: "not_found",
        message: message.to_string(),
        http_status: 404,
        retryable: false,
    }
}

pub(crate) fn wire_status(e: ApiError) -> Response {
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

/// Bound of one native newest-first listing (bounded everything).
pub(crate) const MAX_NATIVE_LIST: usize = 500;

/// `GET /native/health` — liveness: 200 `{ok:true, version}` whenever the
/// process responds (auth-gated like `/global/health`). Conceptually the
/// same "process is up" answer as health; readiness is `/native/ready`.
pub(crate) async fn native_health(State(state): State<AppState>, headers: HeaderMap) -> Response {
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
pub(crate) async fn native_ready(State(state): State<AppState>, headers: HeaderMap) -> Response {
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
pub(crate) fn native_resolve_session(
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

/// Hard page cap of one native cursor page; a `limit` above it is a 400
/// (oversized limits are rejected, never silently clamped).
pub(crate) const MAX_NATIVE_CURSOR_PAGE: i64 = 200;

/// Journal event page cap of the native events twin (same catch-up page
/// bound as the SSE journal stream).
pub(crate) const MAX_NATIVE_EVENT_PAGE: i64 = 256;

/// Map a store error into the server's core error surface (the session
/// crate owns the store-error taxonomy; this crate only translates).
pub(crate) fn store_err_to_core(e: faktor_store::StoreError) -> faktor_core::Error {
    faktor_core::Error::from(faktor_session::SessionError::from(e))
}

pub(crate) fn internal_graph_err(message: String) -> ApiError {
    ApiError {
        code: "internal",
        message,
        http_status: 500,
        retryable: false,
    }
}

// ---------------------------------------------------- native agent state + control
// The REAL agent state of one session (audits P0-20/21/23/61/90/91): every
// orchestrated run's children (registry rows + the child sessions' durable
// identity/drive-state rows + live progress) and the parent's own task
// runs (TaskExecutor in-session runs via their linkage rows, orchestrated
// runs via their plan rows). The listing is a pure durable-row read — an
// empty array ONLY when the session genuinely has no task run. Controls
// enqueue durable child_commands rows with the wave-12 exactly-once
// semantics and report {queuedSeq, applied}.

/// Bound on one agent listing (bounded everything).
pub(crate) const MAX_AGENT_ENTRIES: usize = 500;

pub(crate) fn malformed_body(message: &str) -> ApiError {
    ApiError {
        code: "malformed",
        message: message.to_string(),
        http_status: 400,
        retryable: false,
    }
}

// ------------------------------------------------------------------ SSE

pub(crate) fn api_err(e: &Error) -> Response {
    let api = faktor_protocol::error::from_core(e);
    (
        StatusCode::from_u16(api.http_status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        Json(api.to_json()),
    )
        .into_response()
}
