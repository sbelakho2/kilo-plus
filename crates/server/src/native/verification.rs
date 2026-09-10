//! Durable verification-evidence reads of the native protocol.

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use faktor_protocol::error::ApiError;

use super::*;
use crate::api::AppState;

/// Bound on the verification list of one projection (hostile rows capped).
pub(crate) const MAX_NATIVE_VERIFICATION: usize = 32;

/// The durable verification facts of one session: memory facts of kind
/// `verification` (the runtime records one per failed REQUIRED check with
/// key = check id, value = "failed:<command>"). Bounded.
pub(crate) fn native_verification_facts(
    handle: &faktor_session::SessionHandle,
) -> Vec<serde_json::Value> {
    let facts = handle.memory_facts().unwrap_or_default();
    facts
        .iter()
        .filter(|(kind, _, _)| kind == "verification")
        .take(MAX_NATIVE_LIST)
        .map(|(_, key, value)| {
            serde_json::json!({
                "id": key,
                "detail": value,
                "status": "failed",
            })
        })
        .collect()
}

/// `GET /native/session/{id}/verification` — everything the session owes
/// verification (audit 55 wiring): `owed` = still-open durable tool runs
/// whose recovery strategy is mark_unknown (unknown external effects are
/// forced to verification — spec §7), `failedChecks` = the durable
/// verification facts (one per failed REQUIRED check, recorded at genuine
/// turn ends). Both bounded; empty arrays when nothing is owed.
pub(crate) async fn native_session_verification(
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
    let pending = match handle.pending_tool_runs() {
        Ok(p) => p,
        Err(e) => return api_err(&e),
    };
    let owed: Vec<serde_json::Value> = pending
        .iter()
        .filter(|r| r.recovery.get("strategy").and_then(|s| s.as_str()) == Some("mark_unknown"))
        .take(MAX_NATIVE_LIST)
        .map(|r| {
            serde_json::json!({
                "opId": r.op_id.to_string(),
                "tool": r.tool,
                "startedMs": r.started_ms,
                "status": r.status,
                "effectStatus": r.effect_status,
            })
        })
        .collect();
    Json(serde_json::json!({
        "owed": owed,
        "failedChecks": native_verification_facts(&handle),
    }))
    .into_response()
}

/// The typed evidence projection of ONE durable verification record
/// (wave-16 table, audit P0-64): checks/criteria/changed-files with the
/// record's certification envelope. Field names are camelCase; content is
/// the record's stored, bounded data.
pub(crate) fn native_verification_record_row(
    r: &faktor_store::VerificationRecordRow,
) -> serde_json::Value {
    let criteria: Vec<serde_json::Value> = r
        .criteria
        .iter()
        .map(|c| {
            serde_json::json!({
                "criterionKey": c.criterion_key,
                "passed": c.passed,
                "evidence": c.evidence,
            })
        })
        .collect();
    let checks: Vec<serde_json::Value> = r
        .checks
        .iter()
        .map(|c| {
            serde_json::json!({
                "check": c.check,
                "program": c.program,
                "args": c.args,
                "category": c.category,
                "required": c.required,
                "status": serde_json::to_string(&c.status)
                    .unwrap_or_default()
                    .trim_matches('"'),
                "startedMs": c.started_ms,
                "finishedMs": c.finished_ms,
                "exit": c.exit,
                "summary": c.summary,
            })
        })
        .collect();
    let files: Vec<serde_json::Value> = r
        .changed_files
        .iter()
        .map(|f| {
            serde_json::json!({
                "path": f.path,
                "digestHex": f.digest_hex,
                "size": f.size,
            })
        })
        .collect();
    serde_json::json!({
        "recordId": r.id.to_string(),
        "revision": r.revision.to_string(),
        "workspaceId": r.workspace_id.to_string(),
        "worktreeId": r.worktree_id.to_string(),
        "treeHash": r.tree_hash,
        "criteria": criteria,
        "checks": checks,
        "changedFiles": files,
        "unrelatedChanges": r.unrelated_changes,
        "reviewer": r.reviewer,
        "status": serde_json::to_string(&r.status).unwrap_or_default().trim_matches('"'),
        "startedMs": r.started_ms,
        "completedMs": r.completed_ms,
    })
}

/// `GET /native/session/{id}/tasks/{task_id}/verification` — the durable
/// verification-record rows (wave-16 `verification_record` table) of ONE
/// task of ONE session (audit P0-64), newest first, with checks/criteria/
/// changed files. The task id is resolved through the SESSION's own typed
/// task rows (or its durable task identity), so one session's records can
/// never surface through another session's path. Hostile session/task ids
/// are 400; unknown session or task are typed 404s; a task without records
/// is an empty list.
pub(crate) async fn native_task_verification(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((id, task_id)): Path<(String, String)>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let handle = match native_resolve_session(&state, &id) {
        Ok(h) => h,
        Err(r) => return *r,
    };
    let raw: u64 = match task_id.parse() {
        Ok(v) if v > 0 => v,
        _ => {
            return wire_status(malformed_body(&format!(
                "invalid task id {task_id:?} for session {}",
                handle.id()
            )))
        }
    };
    let requested = faktor_core::id::TaskId::new(raw);
    let row = match handle.row() {
        Ok(r) => r,
        Err(e) => return api_err(&e),
    };
    // Session-scoped resolution: the requested task must be the session's
    // durable task identity OR one of its typed task rows. Anything else is
    // a typed 404 — never a peek into another session's task space.
    let owned = match handle.list_tasks() {
        Ok(t) => t,
        Err(e) => return api_err(&e),
    };
    let known = requested == row.task_id || owned.iter().any(|t| t.task_id == requested);
    if !known {
        let e = ApiError {
            code: "not_found",
            message: format!(
                "task {requested} of session {} has no verification records (unknown task)",
                handle.id()
            ),
            http_status: 404,
            retryable: false,
        };
        return wire_status(e);
    }
    let store = state.deps.session.store();
    let records = match store.verification_record_list_by_task(requested) {
        Ok(r) => r,
        Err(e) => return api_err(&store_err_to_core(e)),
    };
    // Records certify a task revision inside the session's workspace; the
    // workspace guard keeps same-numeric-id tasks of OTHER workspaces out of
    // this session's view.
    let records: Vec<&faktor_store::VerificationRecordRow> = records
        .iter()
        .filter(|r| r.workspace_id == row.workspace_id)
        .collect();
    let out: Vec<serde_json::Value> = records
        .iter()
        .rev()
        .take(MAX_NATIVE_LIST)
        .map(|r| native_verification_record_row(r))
        .collect();
    Json(serde_json::json!({
        "sessionId": handle.id().to_string(),
        "taskId": requested.to_string(),
        "records": out,
    }))
    .into_response()
}

// ------------------------------------------------------- orchestration graph
