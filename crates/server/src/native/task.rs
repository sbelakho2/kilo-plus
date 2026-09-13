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
    /// Attached file paths (the same ordinary-prompt vocabulary the SDK
    /// `PromptRequest.files` carries): additive so IDE clients can attach
    /// files to an explicit task start. Absent/empty leaves the run
    /// attachment-free; the daemon never reads the filesystem at THIS
    /// boundary (the drive's own permission requester gates every tool).
    files: Option<Vec<String>>,
    /// Durable typed binary/image attachments (`AttachmentId` rows) uploaded
    /// through `POST /native/session/{id}/attachments` BEFORE this start.
    /// SEPARATE from `files`: CAS bytes addressed by digest, never workspace
    /// paths. Strict (`deny_unknown_fields` on the id itself); every digest
    /// must resolve to a byte-identical durable row of THIS session and an
    /// image id is refused loudly (code `unsupported`) — provider
    /// media/content parts are not wired, so the server never pretends an
    /// attachment reached a model. Absent/empty = the attachment-free path.
    #[serde(default)]
    attachments: Option<Vec<faktor_core::attachment::AttachmentId>>,
    /// The PR/CI-fix completion contract of this run (P2): when present with
    /// at least one requested step, the executor records it durably BEFORE
    /// the first model call and `VerifiedComplete` requires a durable
    /// `Succeeded` step-status row for every requested step. Parsed
    /// STRICTLY (`deny_unknown_fields`; missing or non-boolean members are
    /// a plain 400 — never a silent default). Absent = today's behavior.
    completion_contract: Option<faktor_core::completion::CompletionContract>,
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

/// True when a task-start body must be refused with a typed 400 because a
/// non-default completion contract arrived on the plain-prompt path (no
/// explicit `work_items`): that path carries no contract seam, so dropping
/// the contract silently would claim steps the run never recorded.
fn completion_contract_needs_work_items(
    has_work_items: bool,
    completion_contract: Option<faktor_core::completion::CompletionContract>,
) -> bool {
    !has_work_items && completion_contract.is_some_and(|c| !c.is_default())
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
    let Json(mut req) = match body {
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
    let files = req.files.take().unwrap_or_default();
    // Additive attachment validation at the DTO boundary: the SAME one rule
    // the executor enforces (MAX_FILES_PER_PROMPT / MAX_FILE_PATH_BYTES +
    // the typed hostile-path refusal) is a plain 400 BEFORE any dispatch —
    // on both the explicit-work-items and the plain-prompt path.
    if let Err(e) = faktor_orchestrator::runtime::validate_attachment_files(&files) {
        return exec_error_response(&e);
    }
    // Binary attachments: every digest must resolve to a byte-identical
    // durable row of THIS session (uploaded first) and images are refused
    // loudly — BEFORE any shadow/task/run row, so a refused start leaves no
    // partial durable admission.
    let attachments = req.attachments.take().unwrap_or_default();
    if let Err(e) = validate_wire_attachments(&handle, &attachments) {
        return wire_status(e);
    }
    // P2: the plain-prompt path (`work_items` absent) carries no completion
    // contract seam; a non-default contract is refused loudly here, never
    // silently dropped. The default all-false contract is accepted and
    // changes nothing.
    if completion_contract_needs_work_items(req.work_items.is_none(), req.completion_contract) {
        return wire_status(ApiError {
            code: "unsupported",
            message: "completion_contract requires explicit work_items; the plain-prompt path carries no contract seam"
                .into(),
            http_status: 400,
            retryable: false,
        });
    }
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
                files,
                attachments: attachments.clone(),
                parent_caps: native_run_parent_caps(),
                completion_contract: req.completion_contract,
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
                files,
                attachments,
                model: req.model,
                criteria: req.criteria.unwrap_or_default(),
                mutation_mode: req.mutation_mode,
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
    // Additive attachment validation (the ONE shared rule): hostile or
    // oversized candidate file lists are plain 400s before any durable row.
    if let Err(e) =
        faktor_orchestrator::runtime::validate_attachment_files(req.files.as_deref().unwrap_or(&[]))
    {
        return exec_error_response(&e);
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

/// The native wire mapping of one tournament error (the executor's private
/// mapping mirrored for the additive listing read: unknown ids are 404s,
/// corrupt/ledger failures are loud 500s, typed shape refusals are 400s).
fn tournament_error_response(e: &faktor_orchestrator::tournament::TournamentError) -> Response {
    use faktor_orchestrator::tournament::TournamentError as T;
    let (code, status) = match e {
        T::NotFound(_) => ("not_found", 404),
        T::Oversized(_) => ("oversized", 400),
        T::NotOpen(_) | T::DuplicateSettlement(_) | T::NoEligibleWinner(_) => ("conflict", 409),
        T::Corrupt { .. } | T::Ledger(_) | T::Cleanup(_) => ("internal", 500),
        T::InvalidCandidateCount { .. }
        | T::InvalidCriteriaCount { .. }
        | T::InvalidCriterion(_)
        | T::DuplicateCriterion(_)
        | T::InvalidId(_)
        | T::UnknownCandidate(_)
        | T::IllegalSettlementState(_)
        | T::VerificationSpecDrift
        | T::ReviewNotIndependent(_) => ("malformed", 400),
    };
    wire_status(ApiError {
        code,
        message: e.to_string(),
        http_status: status,
        retryable: false,
    })
}

/// Max bytes of a native abort reason (bounded everything; the rationale
/// persisted on the durable decision row is never operator-unbounded).
pub(crate) const MAX_ABORT_REASON_BYTES: usize = 512;

/// Strict request DTO of the additive tournament decide route
/// (`POST /native/session/{id}/tournaments/{tournament_id}/decide`): the
/// engine's comparison takes NO operator input, so the body is exactly `{}`
/// (a typo, a hostile member or a missing body is a plain 400).
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeTournamentDecideBody {}

/// Strict request DTO of the additive tournament abort route
/// (`POST /native/session/{id}/tournaments/{tournament_id}/abort`): an
/// optional bounded `reason` recorded verbatim on the durable decision row.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeTournamentAbortBody {
    #[serde(default)]
    reason: Option<String>,
}

/// `POST /native/session/{id}/tournaments/{tournament_id}/decide` — run the
/// deterministic comparison of ONE durable tournament through the executor's
/// ONE decide authority ([`faktor_orchestrator::runtime::task_executor::TaskExecutor::decide_tournament`]):
/// the `TournamentDecided` audit row is persisted, every loser is discarded
/// and the PROPOSED winner is returned (integration stays the explicit
/// approved-merge path). Engine refusals are typed: unknown ids 404, a
/// non-open tournament or no eligible candidate 409 (the candidate band must
/// have settled with a passing verification and an independent review).
/// Strict body (`{}` only); hostile bodies are 400s.
pub(crate) async fn native_tournament_decide(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((id, tournament_id)): Path<(String, String)>,
    body: Result<Json<NativeTournamentDecideBody>, axum::extract::rejection::JsonRejection>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let Json(_) = match body {
        Ok(b) => b,
        Err(_) => return wire_status(malformed_body("invalid native tournament decide body")),
    };
    let handle = match native_resolve_session(&state, &id) {
        Ok(h) => h,
        Err(r) => return *r,
    };
    match state
        .deps
        .tasks
        .decide_tournament(handle.id(), &tournament_id)
    {
        Ok(decision) => Json(serde_json::json!({
            "tournament_id": decision.tournament_id,
            "winner": decision.winner.child_id,
            "rationale": decision.rationale,
            "discarded": decision
                .discarded
                .iter()
                .map(|(child_id, reason)| serde_json::json!({
                    "child_id": child_id,
                    "reason": reason,
                }))
                .collect::<Vec<_>>(),
        }))
        .into_response(),
        Err(e) => exec_error_response(&e),
    }
}

/// `POST /native/session/{id}/tournaments/{tournament_id}/abort` — discard
/// ONE tournament through the executor's ONE abort authority
/// ([`faktor_orchestrator::runtime::task_executor::TaskExecutor::abort_tournament`]):
/// every candidate is settled terminal + its worktree removed, the terminal
/// `TournamentDecided { outcome: aborted }` row records the reason and no
/// winner is proposed. The response is the reconstructed durable tournament.
/// Unknown ids are 404s; a terminal tournament is a typed 409. Strict body
/// (`{"reason"?: "..."}`, bounded); hostile bodies are 400s.
pub(crate) async fn native_tournament_abort(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((id, tournament_id)): Path<(String, String)>,
    body: Result<Json<NativeTournamentAbortBody>, axum::extract::rejection::JsonRejection>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let Json(body) = match body {
        Ok(b) => b,
        Err(_) => return wire_status(malformed_body("invalid native tournament abort body")),
    };
    let reason = body.reason.unwrap_or_default();
    if reason.len() > MAX_ABORT_REASON_BYTES {
        return wire_status(malformed_body(&format!(
            "abort reason must be <= {MAX_ABORT_REASON_BYTES} bytes"
        )));
    }
    let handle = match native_resolve_session(&state, &id) {
        Ok(h) => h,
        Err(r) => return *r,
    };
    match state
        .deps
        .tasks
        .abort_tournament(handle.id(), &tournament_id, &reason)
    {
        Ok(tournament) => Json(tournament).into_response(),
        Err(e) => exec_error_response(&e),
    }
}

/// `GET /native/session/{id}/tournaments` — the durable listing of the
/// session's tournaments (id, state, candidate count, winner, decided_ms),
/// folded from the typed ledger rows (`TournamentStarted` + settlements +
/// `TournamentDecided`). Open tournaments carry `winner: null` and
/// `decided_ms: null`; corrupt rows are loud 500s, never a silently partial
/// list. Hostile sessions are typed (malformed 400 / unknown 404).
pub(crate) async fn native_tournaments_list(
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
    match faktor_orchestrator::tournament::Tournament::summaries(&handle) {
        Ok(list) => Json(list).into_response(),
        Err(e) => tournament_error_response(&e),
    }
}

#[cfg(test)]
mod attachment_dto_tests {
    //! Additive cover of the task-start attachment validation: the SAME one
    //! rule the executor enforces is applied at the DTO boundary and maps to
    //! a plain 400 (via [`super::exec_error_response`]) before any dispatch —
    //! hostile and oversized lists never reach a spawn.
    use super::exec_error_response;
    use faktor_orchestrator::runtime::validate_attachment_files;

    fn status_of(files: &[String]) -> u16 {
        let err = validate_attachment_files(files).expect_err("hostile/oversized list must refuse");
        exec_error_response(&err).status().as_u16()
    }

    #[test]
    fn task_attachment_lists_are_validated_with_the_shared_rule() {
        // Workspace-relative paths (including nested and dot segments) pass.
        assert!(validate_attachment_files(&[
            "src/a.rs".to_string(),
            "docs/sub/b.md".to_string(),
            "./c.txt".to_string(),
        ])
        .is_ok());
        // Oversized COUNT is a typed 400.
        let many: Vec<String> = (0..faktor_session::MAX_FILES_PER_PROMPT + 1)
            .map(|i| format!("f{i}.rs"))
            .collect();
        assert_eq!(status_of(&many), 400, "count over MAX_FILES_PER_PROMPT");
        // Oversized PATH is a typed 400.
        let huge = vec!["x".repeat(faktor_session::MAX_FILE_PATH_BYTES + 1)];
        assert_eq!(status_of(&huge), 400, "path over MAX_FILE_PATH_BYTES");
        // Hostile paths are typed 400s: absolute, traversal, empty,
        // control-character, and Windows drive-prefixed.
        for hostile in [
            "/etc/passwd",
            "../secrets",
            "src/../../secrets",
            "",
            "   ",
            "bad\0path",
            "C:\\Windows",
        ] {
            assert_eq!(status_of(&[hostile.to_string()]), 400, "{hostile:?}");
        }
    }

    /// The additive binary `attachments` DTO member is STRICT and SEPARATE
    /// from `files`: a valid typed id parses, unknown/missing/hostile
    /// members are 400s, and a typo of the field name never silently drops
    /// the attachments.
    #[test]
    fn task_binary_attachments_are_strict_and_separate_from_files() {
        use super::StartTaskRunRequest;
        let digest = "a".repeat(64);
        let valid = serde_json::json!({
            "goal": "g",
            "files": ["src/a.rs"],
            "attachments": [
                {"digest": digest, "mime": "application/pdf", "filename": "spec.pdf", "size": 7},
            ],
        });
        let req: StartTaskRunRequest = serde_json::from_value(valid).unwrap();
        let attachments = req.attachments.expect("attachments parsed");
        assert_eq!(attachments.len(), 1);
        assert_eq!(attachments[0].mime, "application/pdf");
        assert_eq!(req.files.as_deref(), Some(&["src/a.rs".to_string()][..]));
        // A missing REQUIRED id member (`size`) is a serde 400 (never a
        // silently partial id); `filename` is legitimately optional.
        let missing = serde_json::json!({
            "goal": "g",
            "attachments": [{"digest": digest, "mime": "application/pdf"}],
        });
        assert!(serde_json::from_value::<StartTaskRunRequest>(missing).is_err());
        // An unknown id member is a 400 (AttachmentId is deny_unknown_fields).
        let unknown_member = serde_json::json!({
            "goal": "g",
            "attachments": [
                {"digest": digest, "mime": "application/pdf", "filename": null, "size": 7, "url": "data:x"},
            ],
        });
        assert!(serde_json::from_value::<StartTaskRunRequest>(unknown_member).is_err());
        // A typo of the field name is a 400: the DTO never silently drops
        // attachments into a run that never declared them.
        let typo = serde_json::json!({
            "goal": "g",
            "attachment": [{"digest": digest, "mime": "application/pdf", "filename": null, "size": 7}],
        });
        assert!(serde_json::from_value::<StartTaskRunRequest>(typo).is_err());
        // A malformed digest (not 64 hex) is a 400.
        let bad_digest = serde_json::json!({
            "goal": "g",
            "attachments": [{"digest": "zz", "mime": "application/pdf", "filename": null, "size": 7}],
        });
        assert!(serde_json::from_value::<StartTaskRunRequest>(bad_digest).is_err());
    }
}

#[cfg(test)]
mod completion_contract_dto_tests {
    //! Adversarial strict-DTO covers for the additive `completion_contract`
    //! field. Every `Err` below is mapped to a plain 400 by
    //! [`native_task_run_start`]'s uniform body-rejection handling (never a
    //! 422, never a silent default).
    use super::{completion_contract_needs_work_items, StartTaskRunRequest};
    use faktor_core::completion::CompletionContract;

    fn parse(value: serde_json::Value) -> Result<StartTaskRunRequest, String> {
        serde_json::from_value(value).map_err(|e| e.to_string())
    }

    /// The rejection message of a hostile body (the DTO itself is not
    /// `Debug`; only the error shape matters here).
    fn expect_err(value: serde_json::Value) -> String {
        match parse(value.clone()) {
            Ok(_) => panic!("hostile DTO must be rejected: {value}"),
            Err(e) => e,
        }
    }

    /// The plain-prompt path (no `work_items`) refuses a non-default
    /// contract with a typed 400 instead of silently dropping it; the
    /// default contract and the explicit-work-items path are accepted.
    #[test]
    fn non_default_contract_on_the_plain_prompt_path_is_a_typed_400() {
        let push = CompletionContract {
            include_commit: false,
            include_push: true,
            include_pr: false,
        };
        assert!(completion_contract_needs_work_items(false, Some(push)));
        assert!(!completion_contract_needs_work_items(true, Some(push)));
        assert!(!completion_contract_needs_work_items(
            false,
            Some(CompletionContract::default())
        ));
        assert!(!completion_contract_needs_work_items(false, None));
    }

    #[test]
    fn completion_contract_is_strict_typed_and_never_defaults_silently() {
        // Absent = today's behavior (no contract, no durable row, no gate).
        let req = parse(serde_json::json!({"goal": "g"})).unwrap();
        assert!(req.completion_contract.is_none());
        // An explicit all-false contract parses and IS the default behavior.
        let req = parse(serde_json::json!({
            "goal": "g",
            "completion_contract": {
                "include_commit": false,
                "include_push": false,
                "include_pr": false,
            },
        }))
        .unwrap();
        assert_eq!(req.completion_contract, Some(CompletionContract::default()));
        // A full non-default contract parses.
        let req = parse(serde_json::json!({
            "goal": "g",
            "completion_contract": {
                "include_commit": true,
                "include_push": true,
                "include_pr": true,
            },
        }))
        .unwrap();
        assert_eq!(
            req.completion_contract,
            Some(CompletionContract {
                include_commit: true,
                include_push: true,
                include_pr: true,
            })
        );

        // A missing member is a 400, never a silent false.
        let err = expect_err(serde_json::json!({
            "goal": "g",
            "completion_contract": {"include_commit": true},
        }));
        assert!(err.contains("missing field"), "{err}");
        // A non-boolean member is a 400.
        let err = expect_err(serde_json::json!({
            "goal": "g",
            "completion_contract": {
                "include_commit": "yes",
                "include_push": false,
                "include_pr": false,
            },
        }));
        assert!(err.contains("invalid type"), "{err}");
        // An unknown member inside the contract is a 400.
        let err = expect_err(serde_json::json!({
            "goal": "g",
            "completion_contract": {
                "include_commit": true,
                "include_push": false,
                "include_pr": false,
                "include_release": true,
            },
        }));
        assert!(err.contains("unknown field"), "{err}");
        // A non-object contract (string / bool / array) is a 400.
        for hostile in [
            serde_json::json!("include_push"),
            serde_json::json!(true),
            serde_json::json!([true, true, true]),
        ] {
            let err = expect_err(serde_json::json!({
                "goal": "g",
                "completion_contract": hostile,
            }));
            assert!(err.contains("invalid type"), "{err}");
        }
        // An unknown TOP-LEVEL field is still a 400 (the DTO strictness is
        // unchanged by the additive field).
        let err = expect_err(serde_json::json!({
            "goal": "g",
            "completion_contracts": {"include_commit": true},
        }));
        assert!(err.contains("unknown field"), "{err}");
    }
}
