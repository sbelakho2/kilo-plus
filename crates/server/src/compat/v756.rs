//! Frozen v7.5.6 wire compatibility surface (subset).

use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use faktor_core::error::Error;
use faktor_core::id::SessionId;
use faktor_protocol::error::ApiError;
use faktor_protocol::v756::*;
use faktor_protocol::v756::{
    mapper as wire_mapper, wire::AbortBody, wire::DiffStatus, wire::MessageSendRequest,
    wire::MessageSendResponse, wire::RevertBody, wire::SessionCreateRequest,
    wire::SessionCreateResponse, wire::SessionListResponse, wire::SessionSummarizeResponse,
    wire::SessionSummary, wire::SessionUpdateRequest, wire::SessionUpdateResponse,
    wire::SnapshotFileDiff, wire::WireMessageEntry, wire::WireMessageInfo, wire::WirePart,
};
use std::sync::Arc;
use std::time::Duration;

use super::{sdk::directory_header, submit_and_run, turn_machine_busy};
use crate::api::AppState;
use crate::native::{
    agent_state_tag, api_err, authed, exec_error_response, not_found, parse_session_id,
    wire_refused, wire_status,
};

/// `POST /session` — create a session from the wire request. The workspace
/// comes from the `x-faktor-directory` header, else `workspaceID`.
pub(crate) async fn wire_create_session(
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
pub(crate) async fn wire_list_sessions(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Response {
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
pub(crate) async fn wire_session_summary(
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
pub(crate) async fn wire_session_state(
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
pub(crate) async fn wire_message_send(
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
    let receipt =
        match submit_and_run(&state, sid, &args.prompt, &args.files, model_id.clone()).await {
            Ok(r) => r,
            Err(e) => return exec_error_response(&e),
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
pub(crate) struct WireMessagesQuery {
    before: Option<i64>,
    #[serde(default = "wire_default_limit")]
    limit: i64,
}

pub(crate) fn wire_default_limit() -> i64 {
    100
}

/// One durable part row → the wire part union. Mirrors the session layer's
/// own projection; unknown/corrupt kinds fail the page loudly.
pub(crate) fn wire_part_from_row(kind: &str, data: &serde_json::Value) -> Result<WirePart, Error> {
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

pub(crate) async fn wire_messages_page(
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
pub(crate) async fn wire_abort(
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
pub(crate) fn wire_session_row(
    state: &AppState,
    sid: SessionId,
) -> Result<Option<faktor_store::SessionRow>, Box<Response>> {
    match state.deps.session.get_session(sid) {
        Ok(Some(handle)) => handle.row().map(Some).map_err(|e| Box::new(api_err(&e))),
        Ok(None) => Ok(None),
        Err(e) => Err(Box::new(api_err(&e))),
    }
}

pub(crate) fn store_err(e: &faktor_store::StoreError) -> Response {
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
pub(crate) async fn wire_diff(
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
pub(crate) struct WireDiffQuery {
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
pub(crate) fn checkpoint_diff_status(row: &faktor_store::CheckpointRow) -> DiffStatus {
    match (row.before_exists, row.after_exists) {
        (false, true) => DiffStatus::Added,
        (true, false) => DiffStatus::Deleted,
        (true, true) | (false, false) => DiffStatus::Modified,
    }
}

/// Resolve a stored hex FileHash to its CAS bytes (the diff full-content
/// path). Missing/corrupt blobs are an honest refusal, never a fake diff.
#[allow(clippy::result_large_err)]
pub(crate) fn diff_cas_bytes(cas: &Arc<faktor_cas::Cas>, hex: &str) -> Result<Vec<u8>, Response> {
    let hash = match faktor_core::hash::FileHash::from_hex(hex) {
        Some(h) => h,
        None => {
            return Err(wire_refused(&format!(
                "diff unavailable: corrupt stored hash {hex:?}"
            )))
        }
    };
    match cas.get_verified_now(hash) {
        Ok(bytes) => Ok(bytes),
        Err(e) => Err(wire_refused(&format!(
            "diff unavailable: content missing from the CAS ({e})"
        ))),
    }
}

/// The newest checkpoint row of `session` recorded at or before `message_ms`
/// (the revert/unrevert target). `None` when nothing qualifies.
pub(crate) fn checkpoint_before(
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
/// P0-48 root re-pointing: the workspace is resolved through the SESSION's
/// effective root (a live shadow re-points revert/unrevert at the shadow
/// world the drive mutates; un-shadowed sessions keep the stored workspace
/// root byte-identically).
pub(crate) fn open_snapshot_target(
    state: &AppState,
    session_id: SessionId,
) -> Result<(faktor_fs::WorkspaceHandle, faktor_core::WorkspaceIdentity), Box<Response>> {
    let (Some(fs), Some(_)) = (&state.deps.fs, &state.deps.snapshots) else {
        return Err(Box::new(wire_refused("snapshots unavailable")));
    };
    let row = match state.deps.session.store().get_session(session_id) {
        Ok(Some(r)) => r,
        Ok(None) => {
            return Err(Box::new(wire_refused(
                "snapshots unavailable: session unknown",
            )))
        }
        Err(e) => return Err(Box::new(store_err(&e))),
    };
    let Some(root) = (match state.deps.session.resolve_workspace_root(session_id) {
        Ok(r) => r,
        Err(e) => {
            return Err(Box::new(wire_refused(&format!(
                "snapshots unavailable: workspace root resolution failed ({e})"
            ))))
        }
    }) else {
        return Err(Box::new(wire_refused(
            "snapshots unavailable: workspace root unknown",
        )));
    };
    let workspace_id = row.workspace_id;
    let handle = match fs.open(workspace_id, root) {
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
pub(crate) async fn wire_revert(
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
    let Some(_row) = (match wire_session_row(&state, sid) {
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
    let (handle, identity) = match open_snapshot_target(&state, sid) {
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
pub(crate) async fn wire_unrevert(
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
    let Some(_row) = (match wire_session_row(&state, sid) {
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
    let (handle, identity) = match open_snapshot_target(&state, sid) {
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
pub(crate) async fn wire_session_status_query(
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
pub(crate) async fn wire_session_fork(
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
pub(crate) const SUMMARIZE_LAST_MESSAGES: usize = 3;

/// Hard bound on the returned summary text.
pub(crate) const SUMMARIZE_MAX_BYTES: usize = 4096;

/// `POST /session/{sessionID}/summarize` — a bounded summary from the
/// session title + the newest messages' text (nothing heavy is stored or
/// journaled). Text is truncated to [`SUMMARIZE_MAX_BYTES`].
pub(crate) async fn wire_session_summarize(
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
pub(crate) fn push_bounded(out: &mut String, s: &str, max: usize) {
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
pub(crate) async fn wire_session_update(
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
pub(crate) async fn wire_session_delete(
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
pub(crate) async fn wire_message_delete(
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
