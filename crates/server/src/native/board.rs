//! Native coordination-board surface (additive).
//!
//! `GET  /native/session/{id}/board?since=<rev>&limit=` reads one bounded
//! newest-first page of the session's RUN-FAMILY board; `POST` appends one
//! bounded post AS the path session. The durable authority, family scoping
//! (a board id from another run family is a typed permission refusal even
//! when the caller knows every raw id), terminal-child write refusal, reset
//! CAS and every text bound live in `faktor_session::board` — this module is
//! only the strict wire shell:
//!
//! - query/body DTOs use `deny_unknown_fields`, so a typo or an extra member
//!   is a 400 (never silently ignored);
//! - `since` must be a positive revision cursor, `limit` must be
//!   `1..=MAX_BOARD_PAGE`; anything else is a typed 400;
//! - unknown/malformed session ids are typed 404/400; board refusals map to
//!   their typed HTTP classes (403 permission, 413 oversized, 409 conflict).

use axum::extract::rejection::QueryRejection;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::api::AppState;
use crate::native::{api_err, authed, malformed_body, native_resolve_session, wire_status};

/// The page bound of one native board read (mirrors the session board page).
const NATIVE_BOARD_MAX_LIMIT: usize = faktor_session::board::MAX_BOARD_PAGE;

/// Strict query DTO: `?since=<rev>&limit=<n>` only (`deny_unknown_fields` —
/// a typo is a plain 400, never a silently ignored cursor).
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeBoardQuery {
    since: Option<u64>,
    limit: Option<usize>,
}

/// Strict post body: bounded subject/body plus optional refs (the session
/// layer re-validates every bound BEFORE any write; `deny_unknown_fields`
/// makes a hostile extra member a 400).
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeBoardPostBody {
    subject: String,
    body: String,
    #[serde(default)]
    refs: Vec<String>,
}

fn bad_query(message: &str) -> Response {
    wire_status(malformed_body(message))
}

/// `GET /native/session/{id}/board` — one bounded newest-first page of the
/// path session's own run-family board. Reads by a tracked CHILD session are
/// recorded durably by the session layer (its own read markers); the root
/// agent reads untracked.
pub(crate) async fn native_session_board(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    query: Result<Query<NativeBoardQuery>, QueryRejection>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let Query(query) = match query {
        Ok(q) => q,
        Err(_) => return bad_query("invalid native board query"),
    };
    // A revision cursor is a BoardPostId: 0 is never a valid post (the
    // session layer would treat it as "before the first post"). Refuse it
    // loudly instead of answering an empty phantom page.
    if query.since == Some(0) {
        return bad_query("since must be a positive board revision");
    }
    let limit = query.limit.unwrap_or(NATIVE_BOARD_MAX_LIMIT);
    if limit == 0 || limit > NATIVE_BOARD_MAX_LIMIT {
        return bad_query(
            "limit must be 1..=MAX_BOARD_PAGE (oversized limits are rejected, never clamped)",
        );
    }
    let handle = match native_resolve_session(&state, &id) {
        Ok(h) => h,
        Err(r) => return *r,
    };
    match handle.board_read_posts(None, query.since, limit, false) {
        Ok(page) => Json(serde_json::json!({
            "board_id": page.board_id.root().raw(),
            "revision": page.revision,
            "posts": page.posts,
            "next_before_revision": page.next_before_revision,
            "has_more": page.has_more,
        }))
        .into_response(),
        Err(e) => api_err(&e),
    }
}

/// `POST /native/session/{id}/board` — append one bounded post as the path
/// session (the durable authority decides the family board and refuses
/// terminal children before any byte is written). `201` with the created
/// post; hostile/oversized text is a typed 400/413.
pub(crate) async fn native_session_board_post(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    body: Result<Json<NativeBoardPostBody>, axum::extract::rejection::JsonRejection>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let Json(body) = match body {
        Ok(b) => b,
        Err(_) => return wire_status(malformed_body("invalid native board post body")),
    };
    let handle = match native_resolve_session(&state, &id) {
        Ok(h) => h,
        Err(r) => return *r,
    };
    match handle.board_post(&body.subject, &body.body, &body.refs) {
        Ok(post) => (StatusCode::CREATED, Json(post)).into_response(),
        Err(e) => api_err(&e),
    }
}
