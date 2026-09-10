//! Durable session projections and cursor pages of the native protocol.

use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use faktor_core::state::SessionLifecycle;
use faktor_protocol::error::ApiError;

use super::verification::native_verification_facts;
use super::*;
use crate::api::AppState;

/// The snake_case lifecycle tag for native projections.
pub(crate) fn lifecycle_tag(l: SessionLifecycle) -> String {
    serde_json::to_string(&l)
        .unwrap_or_else(|_| "unknown".into())
        .trim_matches('"')
        .to_string()
}

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
/// carry no context-usage aggregate yet — the machine state, activeTool and
/// the journal carry the phase information. `prefixStability` (v13) IS
/// populated from the durable per-call prefix observations once a turn has
/// settled one.
pub(crate) fn build_native_projection(
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
        "progress": deps
            .agent
            .progress_view(row.id)
            .unwrap_or(serde_json::Value::Null),
        "filesChanged": files_changed,
        "lastCheckpoint": last_checkpoint,
        "verification": verification,
        "contextUsage": serde_json::Value::Null,
        "queued": handle.queued_prompt_count()?.max(0),
        // Additive prefix-cache stability (v13, audits 65-66): the session's
        // stored aggregate over the per-call prefix observations recorded by
        // the usage-settlement fill site; null before any completed provider
        // call recorded one (fresh session or pre-v13 rows).
        "prefixStability": handle
            .stored_prefix_stability()?
            .map(|a| {
                serde_json::json!({
                    "observations": a.observations,
                    "mean": a.mean,
                    "stdDev": a.std_dev,
                })
            }),
    }))
}

/// `GET /session/{id}/projection` — native v1 session projection
/// (auth-required like every native endpoint).
pub(crate) async fn native_session_projection(
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

/// `GET /native/session/{id}/turns` — the durable turn records of the
/// session (one per admitted logical turn): envelope, status, timestamps.
/// Newest first, bounded.
pub(crate) async fn native_session_turns(
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
pub(crate) fn ledger_strings(ledger: &serde_json::Value, key: &str) -> Vec<String> {
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
pub(crate) async fn native_session_tasks(
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
    // Additive progress + budget (audit P0-64): `progress` is the session's
    // live bounded progress record (null before the runtime tracked one);
    // `budget` is the DURABLE budget envelope of the session's typed task
    // row — token and monetary caps/spend from the task + cost-ledger
    // columns and the open (in-flight) reservation micro sum — null when no
    // typed task row exists yet (null-safe pre-first-reservation: a typed
    // row without any reservation reads openReservedMicro 0).
    let mut budget: Option<serde_json::Value> = None;
    if let Ok(Some(task)) = handle.get_task(row.task_id) {
        let cost = state
            .deps
            .session
            .store()
            .cost_task_row(handle.id(), row.task_id)
            .ok()
            .flatten();
        // In-flight reservations (schema v17 vocabulary: `reserved` —
        // dispatch never provably began — and `dispatched`, the request left
        // the process and may have billed) both hold their prediction; the
        // v15-era `open` state was renamed away at schema v17 and never
        // occurs in a migrated store.
        let open_micro = state
            .deps
            .session
            .store()
            .cost_reservations_of(handle.id(), row.task_id, MAX_NATIVE_RESERVATIONS_SCAN)
            .map(|rs| {
                rs.iter()
                    .filter(|r| r.status == "reserved" || r.status == "dispatched")
                    .fold(0u64, |acc, r| acc.saturating_add(r.predicted_micro))
            })
            .unwrap_or(0);
        budget = Some(serde_json::json!({
            "maxTokens": task.budget.max_tokens,
            "maxTurns": task.budget.max_turns,
            "spentTokens": task.budget.spent_tokens,
            "spentTurns": task.budget.spent_turns,
            "maxCostMicro": cost.as_ref().and_then(|c| c.max_cost_micro),
            "spentCostMicro": cost.as_ref().map(|c| c.spent_cost_micro).unwrap_or(0),
            "openReservedMicro": open_micro,
        }));
    }
    let progress = state
        .deps
        .agent
        .progress_view(handle.id())
        .unwrap_or(serde_json::Value::Null);
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
        "progress": progress,
        "budget": budget,
    }]))
    .into_response()
}

/// `GET /native/session/{id}/checkpoints` — the durable checkpoint rows of
/// the session (newest first), each with sequence/path/before-after hashes
/// and the restore audit. Empty array when the daemon runs without a
/// checkpoint service wired (`ServerDeps.snapshots` is `None`) or no
/// checkpoint was recorded yet.
pub(crate) async fn native_session_checkpoints(
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

/// `GET /native/session/{id}/agents` — the real agent listing of one
/// session (path-id form of `/native/agents?session=`): every orchestrated
/// run's children plus the parent's own task runs, projected from the
/// durable rows ([`native_agents_body`]). Empty ONLY when the session
/// genuinely has no task run.
pub(crate) async fn native_session_agents(
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
    match native_agents_body(&state, &handle) {
        Ok(entries) => Json(entries).into_response(),
        Err(e) => wire_status(e),
    }
}

/// Native-local twin of the frozen wire `NativeAbortRequest` (same fields,
/// same `deny_unknown_fields` strictness, so the wire shape is identical).
/// The native layer never imports v7.5.6 DTOs; follow-up: promote this DTO
/// to a `faktor-protocol` native module once the compat surface is retired.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeAbortRequest {
    pub session_id: String,
    pub op_id: Option<String>,
}

/// `POST /native/session/{id}/abort` — the native abort (audit 56): the
/// strict `NativeAbortRequest` body (`deny_unknown_fields` — an unknown
/// field or typo is a 400) carries the session id, which must match the
/// path id. `op_id` targets one queued prompt or the active turn; absent =
/// abort everything. Unknown sessions are 404; the semantics are
/// `sdk_abort`'s (queued-prompt kills never touch the machine).
pub(crate) async fn native_session_abort(
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

/// Strict native cursor page: `session` required; `before`/`limit`
/// optional. An unknown query field is a 400.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeMessagesQuery {
    session: String,
    #[serde(default)]
    before: Option<i64>,
    #[serde(default)]
    limit: Option<u64>,
}

/// One native message page bound (bounded everything): oversized `limit`
/// values are rejected with a 400, never silently clamped.
pub(crate) fn page_limit(limit: Option<u64>, max: i64) -> Result<i64, ApiError> {
    match limit {
        None => Ok(max),
        Some(0) => Err(malformed_body("limit must be >= 1")),
        Some(l) if l as i64 > max => Err(malformed_body(&format!(
            "limit {l} exceeds the native page bound {max}"
        ))),
        Some(l) => Ok(l as i64),
    }
}

/// `GET /native/messages?session=<id>&before=<seq>&limit=<n>` — cursor
/// paging over the durable message rows of one session (audit P0-64):
/// newest first, `before` cuts strictly (`seq < before`; absent = the
/// newest page), the page cap is 200. `hasMore`/`nextBefore` give the next
/// older page; rows are gapless per session, so paging across a fixture
/// never duplicates and never gaps. Parts are loaded per message in the
/// page only. Hostile ids 400, unknown sessions 404.
pub(crate) async fn native_messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<NativeMessagesQuery>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    if let Some(before) = q.before {
        if before < 1 {
            return wire_status(malformed_body("before must be >= 1"));
        }
    }
    let limit = match page_limit(q.limit, MAX_NATIVE_CURSOR_PAGE) {
        Ok(l) => l,
        Err(e) => return wire_status(e),
    };
    let handle = match native_resolve_session(&state, &q.session) {
        Ok(h) => h,
        Err(r) => return *r,
    };
    let store = state.deps.session.store();
    let rows = match store.messages_before(handle.id(), q.before, limit as u64 + 1) {
        Ok(r) => r,
        Err(e) => return api_err(&store_err_to_core(e)),
    };
    let has_more = rows.len() as i64 > limit;
    let mut page_rows = rows;
    page_rows.truncate(limit as usize);
    let next_before = if has_more {
        page_rows.last().map(|r| r.seq)
    } else {
        None
    };
    let mut messages: Vec<serde_json::Value> = Vec::new();
    for row in page_rows {
        let parts: Vec<serde_json::Value> = match store.parts_of(row.id) {
            Ok(ps) => ps
                .iter()
                .map(|p| {
                    serde_json::json!({
                        "kind": p.kind,
                        "createdMs": p.created_ms,
                        "data": p.data,
                    })
                })
                .collect(),
            Err(_) => Vec::new(),
        };
        messages.push(serde_json::json!({
            "seq": row.seq,
            "id": row.id,
            "role": row.role,
            "createdMs": row.created_ms,
            "data": row.data,
            "parts": parts,
        }));
    }
    Json(serde_json::json!({
        "sessionId": handle.id().to_string(),
        "messages": messages,
        "hasMore": has_more,
        "nextBefore": next_before,
    }))
    .into_response()
}

/// Strict native journal-page query of `/native/events`.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeEventsQuery {
    session: String,
    #[serde(default)]
    after: Option<u64>,
    #[serde(default)]
    limit: Option<u64>,
}

/// `GET /native/events?session=<id>&after=<seq>&limit=<n>` — the native
/// twin of the `/api/session/{id}/events` journal stream (audit P0-64):
/// durable journal events with `seq > after` ascending, one strict-DTO page
/// at a time (`hasMore`/`nextCursor`), same bounded catch-up paging as the
/// SSE journal poll (page cap 256). `after` is the raw per-session journal
/// sequence (0 = from the beginning). Unknown sessions 404; hostile ids
/// 400.
pub(crate) async fn native_events(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<NativeEventsQuery>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let limit = match page_limit(q.limit, MAX_NATIVE_EVENT_PAGE) {
        Ok(l) => l,
        Err(e) => return wire_status(e),
    };
    let handle = match native_resolve_session(&state, &q.session) {
        Ok(h) => h,
        Err(r) => return *r,
    };
    let after = q.after.unwrap_or(0);
    let events = match handle.events_range(after.saturating_add(1), Some(limit as u64 + 1)) {
        Ok(e) => e,
        Err(e) => return api_err(&e),
    };
    let has_more = events.len() as i64 > limit;
    let mut page_events = events;
    page_events.truncate(limit as usize);
    let next_cursor = if has_more {
        page_events.last().map(|e| e.seq.raw())
    } else {
        None
    };
    let rows: Vec<serde_json::Value> = page_events
        .iter()
        .map(|e| {
            serde_json::json!({
                "seq": e.seq.raw(),
                "kind": serde_json::to_string(&e.kind)
                    .unwrap_or_default()
                    .trim_matches('"'),
                "state": agent_state_tag(e.state),
                "opId": e.op_id.map(|o| o.to_string()),
                "tsMs": e.ts_ms,
                "payload": e.payload,
            })
        })
        .collect();
    Json(serde_json::json!({
        "sessionId": handle.id().to_string(),
        "events": rows,
        "hasMore": has_more,
        "nextCursor": next_cursor.map(|v| serde_json::json!(v)).unwrap_or(serde_json::Value::Null),
    }))
    .into_response()
}
