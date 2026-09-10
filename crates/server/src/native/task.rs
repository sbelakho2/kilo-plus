//! Native task runs, driven by the daemon's one TaskExecutor.

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use faktor_core::state::TaskState;
use faktor_protocol::error::ApiError;

use super::*;
use crate::api::AppState;

/// Linkage-row kind of TaskExecutor in-session runs (orchestrator crate).
pub(crate) const TASK_RUN_ROW_KIND: &str =
    faktor_orchestrator::runtime::task_executor::TASK_RUN_ROW_KIND;

/// The state tag of ONE in-session task run: the session's durable typed
/// task row wins when it is TERMINAL (VerifiedComplete -> "Done", Failed,
/// Cancelled) — a cancelled or verified run must not read as merely
/// "parked"; otherwise the session's live AgentState tag applies.
pub(crate) fn in_session_run_state_tag(
    handle: &faktor_session::SessionHandle,
    session_row: &faktor_store::SessionRow,
) -> &'static str {
    match handle
        .get_task(session_row.task_id)
        .ok()
        .flatten()
        .map(|t| t.state)
    {
        Some(TaskState::VerifiedComplete) => "Done",
        Some(TaskState::Failed) => "Failed",
        Some(TaskState::Cancelled) => "Cancelled",
        _ => session_run_state_tag(session_row.state),
    }
}

/// The task-run projection of a session (wave-24): every TaskExecutor run
/// — in-session (linkage rows) and orchestrated (plan rows + children) —
/// with its durable run identity, per-run state and goal. Built from
/// [`native_agents_body`]'s parent `self` entries (the SAME durable-row
/// derivation the agent listing serves; no second projection can drift),
/// with the run `mode` re-derived from the durable row kinds that name the
/// run. Empty ONLY when the session genuinely has no task run.
pub(crate) fn native_task_run_entries(
    state: &AppState,
    handle: &faktor_session::SessionHandle,
) -> Result<Vec<serde_json::Value>, ApiError> {
    let facts = orchestrator_graph_facts(handle).map_err(internal_graph_err)?;
    let session_row = handle
        .row()
        .map_err(|e| faktor_protocol::error::from_core(&e))?;
    let task_id = session_row.task_id.raw();
    let mut out = Vec::new();
    for e in native_agents_body(state, handle)? {
        if e.get("kind").and_then(|k| k.as_str()) != Some("self") {
            continue;
        }
        let Some(run_id) = e.get("run_id").and_then(|r| r.as_str()) else {
            continue;
        };
        let mode = if facts
            .iter()
            .any(|(kind, key, _)| kind == TASK_RUN_ROW_KIND && key == run_id)
        {
            "in_session"
        } else {
            "orchestrated"
        };
        out.push(serde_json::json!({
            "task_id": task_id,
            "run_id": run_id,
            "mode": mode,
            "state": e.get("state").cloned().unwrap_or(serde_json::Value::String("Unknown".into())),
            "goal": e.get("goal").cloned().unwrap_or(serde_json::Value::Null),
            "item_ids": e.get("item_ids").cloned().unwrap_or(serde_json::json!([])),
            "model": e.get("model").cloned().unwrap_or(serde_json::Value::Null),
        }));
    }
    Ok(out)
}

/// The one-run projection ([`native_task_run_entries`] filtered); an
/// unknown run id is a typed NotFound — a per-run read never answers with a
/// phantom.
pub(crate) fn native_task_run_entry(
    state: &AppState,
    handle: &faktor_session::SessionHandle,
    run_id: &str,
) -> Result<serde_json::Value, ApiError> {
    for e in native_task_run_entries(state, handle)? {
        if e.get("run_id").and_then(|r| r.as_str()) == Some(run_id) {
            return Ok(e);
        }
    }
    Err(not_found(&format!(
        "task run {run_id:?} under session {}",
        handle.id()
    )))
}

/// `GET /native/session/{id}/task-runs` — the session's durable task runs
/// with per-run state (see [`native_task_run_entries`]).
pub(crate) async fn native_task_runs(
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
    match native_task_run_entries(&state, &handle) {
        Ok(entries) => Json(entries).into_response(),
        Err(e) => wire_status(e),
    }
}

/// `GET /native/session/{id}/task-runs/{run_id}` — the durable state of ONE
/// task run. Unknown runs are typed 404s.
pub(crate) async fn native_task_run_state(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((id, run_id)): Path<(String, String)>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let handle = match native_resolve_session(&state, &id) {
        Ok(h) => h,
        Err(r) => return *r,
    };
    match native_task_run_entry(&state, &handle, &run_id) {
        Ok(entry) => Json(entry).into_response(),
        Err(e) => wire_status(e),
    }
}

/// The strict request DTO of ONE native task start
/// (`POST /native/session/{id}/task-runs`). `work_items` is optional: an
/// absent list starts the goal as a single MUTATING work item (`main`) —
/// under the production default (`mutation_mode: shadow` omitted) that run
/// works in a daemon-owned shadow of the checkout and integrates on a
/// verified completion. `criteria` ride the durable task row's acceptance
/// criteria. `mutation_mode` overrides the daemon default for this run.
/// `routing_mode` is parsed strictly but refused until the daemon routing
/// policy (the single routing authority) can honor a per-run override —
/// never silently ignored.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StartTaskRunRequest {
    goal: String,
    criteria: Option<Vec<String>>,
    work_items: Option<Vec<NativeTaskRunWorkItem>>,
    model: Option<String>,
    max_tokens: Option<u64>,
    max_cost_micro: Option<u64>,
    mutation_mode: Option<faktor_orchestrator::runtime::task_executor::MutationMode>,
    routing_mode: Option<faktor_core::model::RoutingMode>,
}

/// One wire work item of [`StartTaskRunRequest`]. `kind` speaks the
/// orchestrator's own JSON vocabulary (the same `"Analysis"` /
/// `"Implementation"` / ... strings the durable plan rows carry).
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeTaskRunWorkItem {
    id: String,
    kind: faktor_orchestrator::WorkKind,
    summary: Option<String>,
    depends_on: Option<Vec<String>>,
    acceptance_checks: Option<Vec<String>>,
}

/// The capability ceiling of a native task run's parent (ReadWorkspace on
/// the whole workspace — the ceiling the orchestrator's own test surface
/// grants read-only runs). Every actual tool call of a drive still passes
/// the permission requester; this is the run's typed policy record, never
/// a permission grant.
pub(crate) fn native_run_parent_caps() -> faktor_orchestrator::caps::CapabilitySet {
    use faktor_orchestrator::caps::{CapabilityGrant, LatticeCap, ScopePattern};
    faktor_orchestrator::caps::CapabilitySet::from_grants(vec![CapabilityGrant::new(
        LatticeCap::ReadWorkspace,
        ScopePattern::new("*").expect("wildcard pattern"),
    )])
    .expect("wildcard grant is sane")
}

/// `POST /native/session/{id}/task-runs` — start ONE task through the
/// daemon's TaskExecutor ([`ServerDeps::tasks`]; the ONLY task-start
/// authority the server reaches). The strict DTO is validated by the
/// executor itself (goal/work-item/criteria bounds and plan rules are typed
/// 400s, never silent truncation). The response carries the run's durable
/// identity + task id + its current per-run state.
///
/// Multi-item MUTATING plans are refused by the executor's own validation
/// (they need an isolated child root this surface does not carry); a
/// single-item mutating task is the intended shape and is shadowed by
/// default.
pub(crate) async fn native_task_run_start(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    body: Result<Json<StartTaskRunRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    // Strict native DTO: every body rejection (syntax AND data errors —
    // unknown fields, typos, bad enum values, missing fields) is a plain
    // 400, never a 422.
    let Json(req) = match body {
        Ok(b) => b,
        Err(_) => {
            return wire_status(ApiError {
                code: "malformed",
                message: "invalid native task-run start body".into(),
                http_status: 400,
                retryable: false,
            })
        }
    };
    let handle = match native_resolve_session(&state, &id) {
        Ok(h) => h,
        Err(r) => return *r,
    };
    let sid = handle.id();
    // Per-run routing overrides are refused until the daemon's routing
    // policy — the single routing authority — can honor them: a parsed-but-
    // ignored policy field would silently claim a mode the run never used.
    if req.routing_mode.is_some() {
        let e = ApiError {
            code: "unsupported",
            message:
                "routing_mode on a task start is not supported yet; the daemon routing policy is the single authority"
                    .into(),
            http_status: 400,
            retryable: false,
        };
        return wire_status(e);
    }
    let work_items: Vec<faktor_orchestrator::WorkItem> = match req.work_items {
        Some(items) => items
            .into_iter()
            .map(|w| faktor_orchestrator::WorkItem {
                id: w.id,
                summary: w.summary.unwrap_or_default(),
                depends_on: w.depends_on.unwrap_or_default(),
                kind: w.kind,
                acceptance_checks: w.acceptance_checks.unwrap_or_default(),
                completion: faktor_orchestrator::WorkState::Pending,
            })
            .collect(),
        None => vec![faktor_orchestrator::WorkItem::new(
            "main",
            req.goal.clone(),
            faktor_orchestrator::WorkKind::Implementation,
        )],
    };
    let request = faktor_orchestrator::runtime::task_executor::TaskRunRequest {
        goal: req.goal,
        work_items,
        model: req.model,
        max_tokens: req.max_tokens,
        max_cost_micro: req.max_cost_micro,
        criteria: req.criteria.unwrap_or_default(),
        mutation_mode: req.mutation_mode,
        parent_caps: native_run_parent_caps(),
        ..Default::default()
    };
    let receipt = match state.deps.tasks.start_task(sid, request) {
        Ok(r) => r,
        Err(e) => return exec_error_response(&e),
    };
    // The response carries the durable run identity + the run's own state
    // projection (the same derivation the list/state endpoints serve).
    let entry = match native_task_run_entry(&state, &handle, &receipt.run_id) {
        Ok(e) => e,
        Err(e) => return wire_status(e),
    };
    Json(serde_json::json!({
        "task_id": entry.get("task_id").cloned().unwrap_or(serde_json::Value::Null),
        "run_id": entry.get("run_id").cloned().unwrap_or(serde_json::Value::Null),
        "state": entry.get("state").cloned().unwrap_or(serde_json::Value::Null),
    }))
    .into_response()
}

/// `POST /native/session/{id}/task-runs/{run_id}/cancel` — task-level
/// cancel of ONE run through the executor's single cancel authority
/// ([`TaskExecutor::cancel_run`]): an in-session run aborts its drive op,
/// cancels the session's task row and discards the run's shadow; an
/// orchestrated run fans a durable Cancel control to every non-terminal
/// child. Unknown runs are typed 404s; already-terminal runs are typed
/// 409s. Response state converges through the durable rows (the per-run
/// GET/list endpoints reflect it once applied).
pub(crate) async fn native_task_run_cancel(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((id, run_id)): Path<(String, String)>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let handle = match native_resolve_session(&state, &id) {
        Ok(h) => h,
        Err(r) => return *r,
    };
    match state.deps.tasks.cancel_run(handle.id(), &run_id) {
        Ok(()) => Json(serde_json::json!({
            "run_id": run_id,
            "cancelled": true,
        }))
        .into_response(),
        Err(e) => exec_error_response(&e),
    }
}
