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
/// `ownership` is the LEGACY plan-global value of old clients: it is
/// converted ONCE onto mutating items that carry no explicit ownership of
/// their own (per-item `work_items[].ownership` always wins); the converted
/// plan reaches the executor with explicit per-item ownership only.
/// `routing_mode` is parsed strictly but refused until the daemon routing
/// policy (the single routing authority) can honor a per-run override —
/// never silently ignored.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StartTaskRunRequest {
    goal: String,
    criteria: Option<Vec<String>>,
    work_items: Option<Vec<NativeTaskRunWorkItem>>,
    /// Legacy plan-global ownership (old clients only): converted at THIS
    /// boundary, never consulted by the runtime.
    ownership: Option<faktor_orchestrator::OwnershipModel>,
    model: Option<String>,
    max_tokens: Option<u64>,
    max_cost_micro: Option<u64>,
    mutation_mode: Option<faktor_orchestrator::runtime::task_executor::MutationMode>,
    routing_mode: Option<faktor_core::model::RoutingMode>,
}

/// One wire work item of [`StartTaskRunRequest`]. `kind` speaks the
/// orchestrator's own JSON vocabulary (the same `"Analysis"` /
/// `"Implementation"` / ... strings the durable plan rows carry).
/// `ownership` is the item's OWN [`faktor_orchestrator::OwnershipSpec`]; a
/// mutating item without one (and without a legacy plan-global conversion)
/// is an InvalidPlan, never defaulted.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeTaskRunWorkItem {
    id: String,
    kind: faktor_orchestrator::WorkKind,
    summary: Option<String>,
    depends_on: Option<Vec<String>>,
    acceptance_checks: Option<Vec<String>>,
    #[serde(default)]
    ownership: Option<faktor_orchestrator::OwnershipSpec>,
    #[serde(default)]
    required_capabilities: Option<faktor_orchestrator::caps::CapabilitySet>,
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
/// Multi-item MUTATING plans are accepted: `TaskExecutor::start_task`
/// allocates the run's daemon-owned isolated root itself (through the
/// executor's single `CandidateWorkspaceService` authority), so this
/// surface never carries a filesystem path; a single-item mutating task is
/// shadowed by default.
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
    // The DTO boundary: per-item ownership is explicit; the legacy
    // plan-global `ownership` value is converted exactly ONCE onto mutating
    // items that carry none of their own (never consulted by the runtime
    // afterwards). A mutating item that still holds NoWrites fails the
    // executor's plan validation loudly.
    //
    // Shadow 409 root cause: a session created by any surface before (or
    // without) creation-time registration carries the standalone default
    // worktree 1; the executor's owner adoption only runs when the
    // workspace holds NO worktree row at all, so a workspace with rows
    // left the session unbound and shadowed runs refused with a 409. The
    // registration is idempotent and self-healing for older sessions.
    if let Err(e) = state.deps.session.ensure_owner_worktree(sid) {
        return api_err(&e);
    }
    let prompts = PromptExecutionService::from_state(&state);
    let receipt = match req.work_items {
        Some(items) => {
            let mut work_items: Vec<faktor_orchestrator::WorkItem> = items
                .into_iter()
                .map(|w| {
                    let mut item = faktor_orchestrator::WorkItem::with_ownership(
                        w.id,
                        w.summary.unwrap_or_default(),
                        w.kind,
                        w.ownership
                            .unwrap_or(faktor_orchestrator::OwnershipSpec::NoWrites),
                    );
                    item.depends_on = w.depends_on.unwrap_or_default();
                    item.acceptance_checks = w.acceptance_checks.unwrap_or_default();
                    if let Some(caps) = w.required_capabilities {
                        item.required_capabilities = caps;
                    }
                    item
                })
                .collect();
            if let Some(legacy) = &req.ownership {
                faktor_orchestrator::adopt_legacy_ownership(&mut work_items, legacy);
            }
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
            match prompts.start_task(sid, request) {
                Ok(r) => r,
                Err(e) => return exec_error_response(&e),
            }
        }
        None => {
            // The ordinary native prompt: one in-session mutating run
            // through the SAME PromptExecutionService (shadow by default).
            let request = PromptRequest {
                prompt: req.goal,
                model: req.model,
                criteria: req.criteria.unwrap_or_default(),
                mutation_mode: req.mutation_mode,
                ..Default::default()
            };
            let prompt_receipt = match prompts.prompt(sid, request).await {
                Ok(r) => r,
                Err(e) => return exec_error_response(&e),
            };
            faktor_orchestrator::runtime::task_executor::TaskRunReceipt {
                run_id: prompt_receipt.run_id,
                mode: faktor_orchestrator::runtime::task_executor::TaskRunMode::InSession,
                op_id: Some(prompt_receipt.op_id),
                queued: prompt_receipt.queued,
            }
        }
    };
    // The response carries the durable run identity + the run's own state
    // projection (the same derivation the list/state endpoints serve). An
    // orchestrated run's plan row is committed by the detached drive AFTER
    // this response is minted, so its receipt is answered directly (the
    // run converges through the list/state endpoints).
    let entry = match receipt.mode {
        faktor_orchestrator::runtime::task_executor::TaskRunMode::InSession => {
            match native_task_run_entry(&state, &handle, &receipt.run_id) {
                Ok(e) => e,
                Err(e) => return wire_status(e),
            }
        }
        faktor_orchestrator::runtime::task_executor::TaskRunMode::Orchestrated => {
            serde_json::json!({
                "task_id": handle
                    .row()
                    .map(|r| r.task_id.raw())
                    .unwrap_or(0),
                "run_id": receipt.run_id,
                "state": "Pending",
            })
        }
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

/// The strict request DTO of ONE native tournament start
/// (`POST /native/session/{id}/tournament`). `n` is the candidate count
/// (the executor enforces the typed 2..=4 band), `criteria` the acceptance
/// criteria fanned out byte-identically to every candidate, and
/// `mutation_mode` overrides the daemon default for the candidate drives.
/// Unknown fields, typos, a missing body field or an out-of-band `n` are
/// plain 400s.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StartTournamentRequest {
    goal: String,
    criteria: Vec<String>,
    n: usize,
    model: Option<String>,
    max_tokens: Option<u64>,
    max_cost_micro: Option<u64>,
    mutation_mode: Option<faktor_orchestrator::runtime::task_executor::MutationMode>,
    files: Option<Vec<String>>,
}

/// The capability ceiling of a tournament's parent: read + write on the
/// whole workspace — candidate writes land ONLY in daemon-owned isolated
/// worktrees, and integration stays the explicit approved-merge path. The
/// per-tool permission requester still gates every actual call.
pub(crate) fn native_tournament_parent_caps() -> faktor_orchestrator::caps::CapabilitySet {
    faktor_orchestrator::runtime::task_executor::child_caps(
        faktor_orchestrator::WorkKind::Implementation,
    )
}

/// `POST /native/session/{id}/tournament` — start a multi-candidate
/// implementation tournament (N = 2..=4) through the executor's ONE
/// tournament entry: N isolated candidate worktrees + children under the
/// existing executor/assignments machinery, all receiving the identical
/// goal+criteria, plus the durable `TournamentStarted` ledger anchor. The
/// response is the durable tournament state (unknown/hostile bodies are
/// 400s; no automatic integration happens at any point).
pub(crate) async fn native_tournament_start(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    body: Result<Json<StartTournamentRequest>, axum::extract::rejection::JsonRejection>,
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
                message: "invalid native tournament start body".into(),
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
    // Same self-healing owner registration as the task-run start (older
    // sessions may carry the standalone default worktree id); tournament
    // candidate children need a real owner worktree.
    if let Err(e) = state.deps.session.ensure_owner_worktree(sid) {
        return api_err(&e);
    }
    let request = faktor_orchestrator::runtime::task_executor::TournamentStartRequest {
        goal: req.goal,
        criteria: req.criteria,
        n: req.n,
        mutation_mode: req.mutation_mode,
        model: req.model,
        max_tokens: req.max_tokens,
        max_cost_micro: req.max_cost_micro,
        files: req.files.unwrap_or_default(),
        parent_caps: native_tournament_parent_caps(),
        ..Default::default()
    };
    let receipt = match state.deps.tasks.start_tournament_with(sid, request) {
        Ok(r) => r,
        Err(e) => return exec_error_response(&e),
    };
    // Converge through the durable rows: the just-persisted anchor is read
    // back so the response is the real tournament state, never a phantom.
    match state
        .deps
        .tasks
        .tournament_state(sid, &receipt.tournament_id)
    {
        Ok(tournament) => Json(serde_json::json!({
            "tournament_id": receipt.tournament_id,
            "run_id": receipt.run_id,
            "candidates": receipt.candidates,
            "state": tournament.state,
            "winner": tournament.winner,
        }))
        .into_response(),
        Err(e) => exec_error_response(&e),
    }
}

/// `GET /native/session/{id}/tournament/{tournament_id}` — the durable
/// state of ONE tournament reconstructed from its ledger rows
/// (`TournamentStarted` + settlements + decision), with running candidates
/// refreshed from the run's registry rows. Unknown ids are typed 404s.
pub(crate) async fn native_tournament_state(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((id, tournament_id)): Path<(String, String)>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let handle = match native_resolve_session(&state, &id) {
        Ok(h) => h,
        Err(r) => return *r,
    };
    match state
        .deps
        .tasks
        .tournament_state(handle.id(), &tournament_id)
    {
        Ok(tournament) => Json(tournament).into_response(),
        Err(e) => exec_error_response(&e),
    }
}
