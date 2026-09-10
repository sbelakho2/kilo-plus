//! Session-owned terminal projection, ownership and bounded lifetime events.

use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use faktor_core::id::SessionId;

use super::*;
use crate::api::AppState;

/// `GET /native/session/{id}/terminal` — the legacy DAEMON-LEVEL terminal
/// view (frozen by the pre-P0-62 compat tests): every registered PTY of the
/// daemon (`{id, pid, alive}`), session id validated for route symmetry.
///
/// Since audit P0-62 the SESSION-SCOPED projection is
/// `GET /native/terminals?session=<id>`: a session view never contains
/// another session's terminals, and rows that carry ownership additionally
/// project it here additively (`sessionId`/`taskId`/`agentId`/
/// `operationId`/`spawnedMs`); unowned legacy rows keep the plain shape.
pub(crate) async fn native_session_terminal(
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
    native_terminal_sweep(&state);
    // Lock order is owners → ptys (sweep included), never the reverse.
    let owners = state
        .terminal_owners
        .lock()
        .expect("terminal owners poisoned");
    let ptys = state.ptys.lock().expect("ptys poisoned");
    let mut ids: Vec<u64> = ptys.keys().copied().collect();
    ids.sort_unstable();
    let rows: Vec<serde_json::Value> = ids
        .iter()
        .take(MAX_NATIVE_LIST)
        .map(|pty_id| {
            let p = ptys.get(pty_id).expect("id from ptys keys");
            let mut row = serde_json::json!({
                "id": pty_id.to_string(),
                "pid": p.pid(),
                "alive": p.is_alive(),
            });
            if let Some(owner) = owners.get(pty_id) {
                row["sessionId"] = serde_json::json!(owner.session_id.to_string());
                row["taskId"] = serde_json::json!(owner.task_id.to_string());
                row["agentId"] = serde_json::json!(owner.agent_id);
                row["operationId"] = serde_json::json!(owner.operation_id.to_string());
                row["spawnedMs"] = serde_json::json!(owner.spawned_ms);
            }
            row
        })
        .collect();
    Json(serde_json::json!(rows)).into_response()
}

/// Bound of one session-scoped terminal listing.
pub(crate) const MAX_NATIVE_TERMINALS: usize = 256;

/// Bounded daemon-wide terminal lifetime-event log depth.
pub(crate) const TERMINAL_EVENT_RING: usize = 512;

/// Cap on one terminal spawn (mirrors `/pty/create`).
pub(crate) const MAX_NATIVE_TERMINAL_FIELD_BYTES: usize = 4096;

/// Cap on one terminal spawn arg list.
pub(crate) const MAX_NATIVE_TERMINAL_ARGS: usize = 256;

/// The ownership of ONE session-owned terminal (P0-62). Additive rows over
/// the daemon PTY registry: `{session_id, task_id, agent_id,
/// operation_id}`. Unowned legacy rows (daemon-level `/pty/create` spawns
/// predating ownership) carry no entry and are NEVER projected into a
/// session-scoped view — the daemon owning both sessions' terminals never
/// leaks one session's terminal into another's listing.
#[derive(Debug, Clone)]
pub(crate) struct NativeTerminalOwnership {
    session_id: SessionId,
    task_id: faktor_core::id::TaskId,
    /// The orchestrator child agent owning the terminal, when one spawned
    /// it; a direct session-owned terminal has `None`.
    agent_id: Option<String>,
    operation_id: faktor_core::id::OpId,
    /// The child pid at spawn (kept on the ownership row so an `exited`
    /// event still names the pid when the daemon row was already removed).
    pid: u32,
    spawned_ms: i64,
}

/// Append one session-owned terminal lifetime event to the bounded ring
/// (ids ascend from 1; at most [`TERMINAL_EVENT_RING`] entries survive).
pub(crate) fn native_terminal_event_push(state: &AppState, event: serde_json::Value) -> u64 {
    let id = state
        .next_terminal_event_id
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let mut ring = state
        .terminal_events
        .lock()
        .expect("terminal events poisoned");
    ring.push_back((id, event));
    while ring.len() > TERMINAL_EVENT_RING {
        ring.pop_front();
    }
    id
}

/// Lazy lifetime sweep of session-owned terminals (P0-62): a terminal whose
/// process exited since the last native read is removed from the daemon
/// registry and one `exited` event lands in the bounded event log. Runs
/// inside the native reads (listing + event log) — deliberately no
/// background task and no unbounded lifetime (the bus philosophy: polling
/// is lazy). Unowned legacy rows are never swept; their lifecycle stays
/// daemon-level.
pub(crate) fn native_terminal_sweep(state: &AppState) {
    let mut owners = state
        .terminal_owners
        .lock()
        .expect("terminal owners poisoned");
    let mut dead: Vec<(u64, NativeTerminalOwnership)> = Vec::new();
    {
        let mut ptys = state.ptys.lock().expect("ptys poisoned");
        for (id, owner) in owners.iter() {
            match ptys.get(id) {
                Some(p) if p.is_alive() => {}
                Some(_) | None => {
                    ptys.remove(id);
                    dead.push((*id, owner.clone()));
                }
            }
        }
    }
    for (id, owner) in dead {
        owners.remove(&id);
        native_terminal_event_push(
            state,
            serde_json::json!({
                "type": "exited",
                "ptyId": id.to_string(),
                "pid": owner.pid,
                "tsMs": state.deps.session.now_ms(),
                "sessionId": owner.session_id.to_string(),
            }),
        );
    }
}

/// One session-owned terminal row of the native view (P0-62): the daemon
/// PTY facts plus its durable ownership. `agentId` is null for direct
/// session-owned spawns.
pub(crate) fn native_terminal_row(
    ptys: &std::collections::HashMap<u64, faktor_pty::Pty>,
    id: u64,
    owner: &NativeTerminalOwnership,
) -> serde_json::Value {
    let pty = ptys.get(&id);
    serde_json::json!({
        "id": id.to_string(),
        "pid": pty.map(|p| p.pid()).unwrap_or(0),
        "alive": pty.map(|p| p.is_alive()).unwrap_or(false),
        "sessionId": owner.session_id.to_string(),
        "taskId": owner.task_id.to_string(),
        "agentId": owner.agent_id,
        "operationId": owner.operation_id.to_string(),
        "spawnedMs": owner.spawned_ms,
    })
}

/// Strict query DTO of the session-scoped terminal listing
/// (`?session=<id>` only).
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeTerminalsQuery {
    session: String,
}

/// `GET /native/terminals?session=<id>` — the session-owned terminal
/// projection (audit P0-62). Returns ONLY the terminals whose durable
/// ownership names `session`: `{sessionId, terminals: [{id, pid, alive,
/// sessionId, taskId, agentId, operationId, spawnedMs}], unowned, note}`.
/// Unowned legacy rows (the daemon-level `/pty/create` surface predates
/// ownership) are NEVER projected into a session view — they are counted in
/// `unowned` and named in `note`, so a caller filtering session A can never
/// see session B's terminals, and a caller of a session with no owned
/// terminal gets a documented empty list plus the legacy-row reason, not a
/// silent leak. Hostile session ids are 400; unknown sessions 404.
pub(crate) async fn native_terminals(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<NativeTerminalsQuery>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let handle = match native_resolve_session(&state, &q.session) {
        Ok(h) => h,
        Err(r) => return *r,
    };
    let sid = handle.id();
    native_terminal_sweep(&state);
    // Lock order is owners → ptys everywhere (sweep included), so two
    // concurrent native reads can never deadlock on the maps.
    let owners = state
        .terminal_owners
        .lock()
        .expect("terminal owners poisoned");
    let ptys = state.ptys.lock().expect("ptys poisoned");
    let mut rows: Vec<serde_json::Value> = Vec::new();
    let mut unowned: u64 = 0;
    let mut ids: Vec<u64> = ptys.keys().copied().collect();
    ids.sort_unstable();
    for id in ids {
        match owners.get(&id) {
            Some(owner) if owner.session_id == sid => {
                rows.push(native_terminal_row(&ptys, id, owner));
            }
            Some(_) => {} // another session's terminal: invisible here.
            None => unowned += 1,
        }
        if rows.len() >= MAX_NATIVE_TERMINALS {
            break;
        }
    }
    let note = if unowned > 0 {
        format!(
            "{unowned} daemon-level PTY row(s) carry no session ownership (spawned through the legacy /pty/create surface before P0-62) and are excluded from every session-scoped view"
        )
    } else {
        String::new()
    };
    Json(serde_json::json!({
        "sessionId": sid.to_string(),
        "terminals": rows,
        "unowned": unowned,
        "note": note,
    }))
    .into_response()
}

/// Strict query DTO of the terminal event log page.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeTerminalEventsQuery {
    #[serde(default)]
    after: Option<u64>,
    #[serde(default)]
    limit: Option<u64>,
}

/// `GET /native/session/{id}/terminal/events?after=<n>&limit=<n>` — the
/// session-owned terminal LIFETIME event log (audit P0-62): bounded
/// `created`/`exited` frames of the session's terminals, `id` ascending
/// strictly above `after`. Page semantics match the native cursor pages
/// (`hasMore` + `nextCursor`); the log is a bounded daemon ring, so frames
/// older than the ring window are gone (an oversized `after` simply returns
/// nothing — never an error). Exit events are appended lazily by the native
/// reads when a swept terminal's process died. Unknown sessions 404,
/// hostile ids 400.
pub(crate) async fn native_terminal_events(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Query(q): Query<NativeTerminalEventsQuery>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let handle = match native_resolve_session(&state, &id) {
        Ok(h) => h,
        Err(r) => return *r,
    };
    let limit = match page_limit(q.limit, MAX_NATIVE_CURSOR_PAGE) {
        Ok(l) => l,
        Err(e) => return wire_status(e),
    };
    native_terminal_sweep(&state);
    let after = q.after.unwrap_or(0);
    let ring = state
        .terminal_events
        .lock()
        .expect("terminal events poisoned");
    let sid_str = handle.id().to_string();
    let mut page: Vec<serde_json::Value> = Vec::new();
    let mut more = false;
    let mut last: Option<u64> = None;
    for (eid, event) in ring.iter() {
        if *eid <= after {
            continue;
        }
        if event.get("sessionId").and_then(|s| s.as_str()) != Some(sid_str.as_str()) {
            continue;
        }
        if page.len() as i64 >= limit {
            more = true;
            break;
        }
        let mut row = event.clone();
        row["id"] = serde_json::json!(eid);
        page.push(row);
        last = Some(*eid);
    }
    Json(serde_json::json!({
        "sessionId": sid_str,
        "events": page,
        "hasMore": more,
        "nextCursor": if more { last.map(|v| serde_json::json!(v)) } else { Some(serde_json::Value::Null) },
    }))
    .into_response()
}

/// Strict native body of a session-owned terminal spawn
/// (`deny_unknown_fields` — a typo is a 400).
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeTerminalSpawnBody {
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    cwd: Option<String>,
    #[serde(default)]
    rows: Option<u32>,
    #[serde(default)]
    cols: Option<u32>,
}

/// `POST /native/session/{id}/terminal` — spawn a SESSION-OWNED terminal
/// (audit P0-62): the row carries `{session_id, task_id, agent_id,
/// operation_id}` ownership (task id = the session's durable task identity;
/// the operation id is minted from the durable op sequence — an ownership
/// label, never a journaled operation; agent id is null for direct spawns).
/// The strict body mirrors `/pty/create` (`{command, args?, cwd?, rows?,
/// cols?}`) and registers the PTY in the SAME daemon registry, so the
/// legacy daemon-level controls (update/output/remove) keep working on it —
/// the ownership is purely additive. Unknown sessions 404; hostile ids and
/// malformed/oversized bodies 400.
pub(crate) async fn native_terminal_spawn(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    body: Result<Json<NativeTerminalSpawnBody>, axum::extract::rejection::JsonRejection>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    // Strict native DTO: every body rejection is a plain 400 (unknown
    // fields, typos, missing fields).
    let Json(body) = match body {
        Ok(b) => b,
        Err(_) => return wire_status(malformed_body("invalid native terminal spawn body")),
    };
    let handle = match native_resolve_session(&state, &id) {
        Ok(h) => h,
        Err(r) => return *r,
    };
    let sid = handle.id();
    let row = match handle.row() {
        Ok(r) => r,
        Err(e) => return api_err(&e),
    };
    if body.command.is_empty() || body.command.len() > MAX_NATIVE_TERMINAL_FIELD_BYTES {
        return wire_status(malformed_body(
            "terminal command must be non-empty and at most 4096 bytes",
        ));
    }
    if body.args.len() > MAX_NATIVE_TERMINAL_ARGS
        || body
            .args
            .iter()
            .any(|a| a.len() > MAX_NATIVE_TERMINAL_FIELD_BYTES)
    {
        return wire_status(malformed_body("terminal args are oversized"));
    }
    if let Some(cwd) = &body.cwd {
        if cwd.len() > MAX_NATIVE_TERMINAL_FIELD_BYTES {
            return wire_status(malformed_body("terminal cwd is oversized"));
        }
    }
    let rows = u16::try_from(body.rows.unwrap_or(24))
        .unwrap_or(u16::MAX)
        .max(1);
    let cols = u16::try_from(body.cols.unwrap_or(80))
        .unwrap_or(u16::MAX)
        .max(1);
    let cfg = faktor_pty::PtyConfig {
        command: body.command.clone(),
        args: body.args.clone(),
        cwd: body.cwd.clone(),
        env: vec![],
        rows,
        cols,
    };
    let pty = match tokio::task::spawn_blocking(move || faktor_pty::Pty::spawn(&cfg)).await {
        Ok(Ok(p)) => p,
        Ok(Err(e)) => {
            return (StatusCode::BAD_REQUEST, Json(api_error_json(&e))).into_response();
        }
        Err(_) => return wire_refused("pty spawn task failed"),
    };
    let pty_id = state
        .next_pty_id
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let pid = pty.pid();
    let operation_id = state.deps.session.next_op_id();
    let owner = NativeTerminalOwnership {
        session_id: sid,
        task_id: row.task_id,
        agent_id: None,
        operation_id,
        pid,
        spawned_ms: state.deps.session.now_ms(),
    };
    // Lock order is owners → ptys (sweep included), never the reverse.
    state
        .terminal_owners
        .lock()
        .expect("terminal owners poisoned")
        .insert(pty_id, owner);
    state
        .ptys
        .lock()
        .expect("ptys poisoned")
        .insert(pty_id, pty);
    native_terminal_event_push(
        &state,
        serde_json::json!({
            "type": "created",
            "ptyId": pty_id.to_string(),
            "pid": pid,
            "tsMs": state.deps.session.now_ms(),
            "sessionId": sid.to_string(),
        }),
    );
    Json(serde_json::json!({
        "ok": true,
        "ptyId": pty_id.to_string(),
        "pid": pid,
        "sessionId": sid.to_string(),
        "taskId": row.task_id.to_string(),
        "agentId": serde_json::Value::Null,
        "operationId": operation_id.to_string(),
    }))
    .into_response()
}
