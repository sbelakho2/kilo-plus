//! Native agent control surface: durable child agents, orchestrator graph
//! projection and exactly-once child commands.

use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use faktor_core::id::SessionId;
use faktor_protocol::error::ApiError;

use super::*;
use crate::api::AppState;

/// Wave-12/13 durable row kinds of the orchestration runtime (owned by
/// `faktor-orchestrator`; the projection here reads them generically):
/// plan rows (key = run id), registry rows (key `<run>/<child_id>`),
/// merge envelopes (key `<run>/<child_id>/merge/<cs_id>/<seq>`) and merge
/// parts (key `<run>/<child_id>/merge/<cs_id>/<seq>/part/<name>`).
pub(crate) const ORCH_PLAN_KIND: &str = "orchestrator_plan";

pub(crate) const ORCH_REGISTRY_KIND: &str = "orchestrator_registry";

pub(crate) const ORCH_MERGE_KIND: &str = "orchestrator_merge";

pub(crate) const ORCH_MERGE_PART_KIND: &str = "orchestrator_merge_part";

/// Cap on graph child nodes (mirrors the orchestrator's own cap).
pub(crate) const MAX_GRAPH_CHILDREN: usize = 256;

/// Strict query DTO: `?session=<id>` only; an unknown field is a 400.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeOrchestratorGraphQuery {
    session: String,
}

/// Bounded page walk of one session's durable rows: 200 facts per page,
/// at most 256 pages — beyond that the projection REFUSES loudly (a
/// hostile row space is never silently truncated into a partial graph).
pub(crate) fn orchestrator_graph_facts(
    handle: &faktor_session::SessionHandle,
) -> Result<Vec<(String, String, String)>, String> {
    let mut out = Vec::new();
    let mut after: Option<(i64, String, String)> = None;
    for _ in 0..256 {
        let page = handle
            .memory_facts_page(after.as_ref(), 200)
            .map_err(|e| format!("fact page scan: {}", e.message))?;
        out.extend(page.facts);
        match page.cursor {
            Some(c) => after = Some(c),
            None => return Ok(out),
        }
    }
    Err("memory-fact scan exceeded 256 pages; refusing a partial graph".to_string())
}

/// Rejoin one chunked row set (header + `key/cNNN` chunks) into its chunk
/// strings. Header rows carry `{"chunks": n}`; a count mismatch is loud.
pub(crate) fn orchestrator_chunks_of(
    facts: &[(String, String, String)],
    kind: &str,
    key: &str,
) -> Result<Option<Vec<String>>, String> {
    let mut header: Option<serde_json::Value> = None;
    let mut chunks: Vec<(usize, String)> = Vec::new();
    for (k, kk, v) in facts {
        if k != kind {
            continue;
        }
        if kk == key {
            header = Some(
                serde_json::from_str(v)
                    .map_err(|e| format!("chunked row header decode {kind}/{key}: {e}"))?,
            );
        } else if let Some(rest) = kk.strip_prefix(key).and_then(|r| r.strip_prefix("/c")) {
            let idx: usize = rest
                .parse()
                .map_err(|_| format!("hostile chunk key {kind}/{kk}"))?;
            chunks.push((idx, v.clone()));
        }
    }
    let Some(header) = header else {
        return Ok(None);
    };
    chunks.sort_by_key(|(i, _)| *i);
    let n = header.get("chunks").and_then(|c| c.as_u64()).unwrap_or(0) as usize;
    if chunks.len() != n {
        return Err(format!(
            "chunked row {kind}/{key} is incomplete ({} of {n} chunks)",
            chunks.len()
        ));
    }
    Ok(Some(chunks.into_iter().map(|(_, v)| v).collect()))
}

pub(crate) fn orchestrator_graph_row_error(
    facts: &[(String, String, String)],
    kind: &str,
    key: &str,
) -> String {
    let raw = facts
        .iter()
        .find(|(k, kk, _)| k == kind && kk == key)
        .map(|(_, _, v)| v.clone())
        .unwrap_or_default();
    format!("stored row {kind}/{key} of {:?} is not valid JSON", raw)
}

/// `GET /native/orchestrator/graph?session=<id>` (audit 93) — the single
/// durable operation graph of one orchestration run, assembled as a READ
/// over the wave-12/13 rows of the parent session: the plan row (root),
/// the registry rows (children), each child session's control rows
/// (steering history with exactly-once applied timestamps) and the merge
/// envelopes/parts (merge outcome). Auth-gated like every `/native`
/// handler.
///
/// Wire contract (documented — mirrors the typed `OpGraph` of
/// `faktor-orchestrator`):
///
/// ```json
/// { "plan_id": "<run id>",
///   "goal": "...",
///   "state": "Running|Done|Failed|Cancelled|Blocked|Pending",
///   "work_items": [ { "item_id": "...", "kind": "...", "state": "..." } ],
///   "children": [ { "child_id": "...", "session_id": 2,
///       "operation_id": 0, "worktree_id": 1, "ownership": "...",
///       "state": "...", "budget": null, "capabilities": [],
///       "plan_step_index": 0,
///       "steer_events": [ { "kind": { "kind": "pause" }, "seq": 1,
///                            "applied_ms": 123 } ],
///       "merge": null | { "change_set_id": "...", "merged": ["a.rs"],
///                         "rejected": [], "conflicts": [] } } ] }
/// ```
///
/// Ordering is deterministic (plan step order, then spawn order); states
/// are derived from the durable child rows with the executor re-attach
/// semantics. Errors: hostile/missing session ids (non-numeric, 0,
/// unknown, or a session without any orchestration run) are 404; a parent
/// session holding several runs is a 409 naming the runs; corrupted rows
/// are 500 — never a silently partial graph.
pub(crate) async fn native_orchestrator_graph(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<NativeOrchestratorGraphQuery>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let graph_404 = |m: String| {
        let e = ApiError {
            code: "not_found",
            message: m,
            http_status: 404,
            retryable: false,
        };
        (StatusCode::NOT_FOUND, Json(e.to_json())).into_response()
    };
    // Hostile ids are 404 for this endpoint (a graph is identified by a
    // session; anything that cannot be one has no graph).
    let Ok(raw) = q.session.parse::<u64>() else {
        return graph_404(format!("invalid session id {:?}", q.session));
    };
    if raw == 0 {
        return graph_404("session id cannot be 0".to_string());
    }
    let sid = SessionId::new(raw);
    let handle = match state.deps.session.get_session(sid) {
        Ok(Some(h)) => h,
        Ok(None) => return graph_404(format!("session {sid} has no orchestration graph")),
        Err(e) => return api_err(&e),
    };
    let facts = match orchestrator_graph_facts(&handle) {
        Ok(f) => f,
        Err(m) => {
            let e = ApiError {
                code: "internal",
                message: m,
                http_status: 500,
                retryable: false,
            };
            return wire_status(e);
        }
    };
    // Every run with durable plan or registry rows, sorted.
    let mut runs: Vec<String> = Vec::new();
    let push_run = |runs: &mut Vec<String>, run: &str| {
        if !runs.iter().any(|r| r == run) {
            runs.push(run.to_string());
        }
    };
    for (kind, key, _) in &facts {
        match kind.as_str() {
            ORCH_PLAN_KIND => push_run(&mut runs, key),
            ORCH_REGISTRY_KIND => {
                if let Some(run) = key.rsplit_once('/').map(|(r, _)| r) {
                    push_run(&mut runs, run);
                }
            }
            _ => {}
        }
    }
    runs.sort();
    if runs.is_empty() {
        return graph_404(format!("session {sid} has no orchestration graph"));
    }
    if runs.len() > 1 {
        let e = ApiError {
            code: "conflict",
            message: format!(
                "session {sid} holds {} orchestration runs ({}); the graph of one run needs one plan",
                runs.len(),
                runs.join(", ")
            ),
            http_status: 409,
            retryable: false,
        };
        return wire_status(e);
    }
    let run = runs.into_iter().next().expect("len checked");
    // ---- plan row: root node.
    let plan_value: serde_json::Value = {
        let chunks = match orchestrator_chunks_of(&facts, ORCH_PLAN_KIND, &run) {
            Ok(c) => c,
            Err(m) => return wire_status(internal_graph_err(m)),
        };
        let raw = chunks.map(|c| c.join("")).unwrap_or_default();
        let raw = if raw.is_empty() {
            facts
                .iter()
                .find(|(k, kk, _)| k == ORCH_PLAN_KIND && kk == &run)
                .map(|(_, _, v)| v.clone())
                .unwrap_or_default()
        } else {
            raw
        };
        match serde_json::from_str::<serde_json::Value>(&raw) {
            Ok(v) => v,
            Err(_) => {
                return wire_status(internal_graph_err(orchestrator_graph_row_error(
                    &facts,
                    ORCH_PLAN_KIND,
                    &run,
                )))
            }
        }
    };
    let plan = plan_value.get("plan").cloned().unwrap_or_default();
    let goal = plan
        .get("goal")
        .and_then(|g| g.as_str())
        .unwrap_or("")
        .to_string();
    let work_items = plan
        .get("work_items")
        .and_then(|w| w.as_array())
        .cloned()
        .unwrap_or_default();
    // ---- registry rows: children (parse errors are loud).
    let prefix = format!("{run}/");
    let mut child_rows: Vec<(String, serde_json::Value)> = Vec::new();
    for (kind, key, value) in &facts {
        if kind != ORCH_REGISTRY_KIND {
            continue;
        }
        let Some(rest) = key.strip_prefix(&prefix) else {
            continue;
        };
        if rest.is_empty() || rest.contains('/') {
            return wire_status(internal_graph_err(format!(
                "hostile registry row key {key:?} under run {run}"
            )));
        }
        match serde_json::from_str::<serde_json::Value>(value) {
            Ok(v) => child_rows.push((rest.to_string(), v)),
            Err(_) => {
                return wire_status(internal_graph_err(orchestrator_graph_row_error(
                    &facts,
                    ORCH_REGISTRY_KIND,
                    key,
                )))
            }
        }
    }
    if child_rows.len() > MAX_GRAPH_CHILDREN {
        return wire_status(internal_graph_err(format!(
            "run {run} holds {} durable child rows (cap {MAX_GRAPH_CHILDREN}); refusing an unbounded graph",
            child_rows.len()
        )));
    }
    // plan_step_index: position of the child's item id in the plan.
    let step_index = |item: &str| {
        work_items
            .iter()
            .position(|w| w.get("id").and_then(|i| i.as_str()) == Some(item))
    };
    // Per-child durable state strings (the stored ChildRuntime JSON shape:
    // PascalCase state tags).
    let child_state_of = |row: &serde_json::Value| -> &'static str {
        match row.get("state").and_then(|s| s.as_str()).unwrap_or("") {
            "Done" => "Done",
            "Cancelled" => "Cancelled",
            "Failed" => "Failed",
            _ => "Running",
        }
    };
    // Derive per-step states with the orchestrator's re-attach semantics:
    // items start Pending; a durable child moves its item to Running first
    // and its terminal state maps Done/Failed/Cancelled; Pending items
    // behind failed/cancelled steps are Blocked.
    let mut step_states: Vec<&'static str> = work_items
        .iter()
        .map(|w| {
            let mut st = "Pending";
            for (_, child) in &child_rows {
                if child.get("item_id").and_then(|i| i.as_str())
                    != w.get("id").and_then(|i| i.as_str())
                {
                    continue;
                }
                if st == "Pending" {
                    st = "Running";
                }
                let t = child_state_of(child);
                if (st == "Running" || st == "Pending") && t != "Running" {
                    st = t;
                }
            }
            st
        })
        .collect();
    for (i, w) in work_items.iter().enumerate() {
        if step_states[i] != "Pending" {
            continue;
        }
        let deps: Vec<String> = w
            .get("depends_on")
            .and_then(|d| d.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        if deps.iter().any(|d| {
            work_items
                .iter()
                .position(|x| x.get("id").and_then(|i| i.as_str()) == Some(d.as_str()))
                .map(|j| matches!(step_states[j], "Failed" | "Cancelled"))
                .unwrap_or(false)
        }) {
            step_states[i] = "Blocked";
        }
    }
    let root_state = if step_states.iter().all(|s| *s == "Done") {
        "Done"
    } else {
        [
            "Failed",
            "Cancelled",
            "Running",
            "Blocked",
            "Paused",
            "Pending",
        ]
        .iter()
        .find(|wanted| step_states.contains(wanted))
        .copied()
        .unwrap_or("Pending")
    };
    // ---- children projection.
    let mut children: Vec<(Option<usize>, i64, String, serde_json::Value)> = Vec::new();
    for (child_id, row) in child_rows {
        let idx = row
            .get("item_id")
            .and_then(|i| i.as_str())
            .and_then(step_index);
        let created_ms = row.get("created_ms").and_then(|c| c.as_i64()).unwrap_or(0);
        children.push((idx, created_ms, child_id, row));
    }
    children.sort_by(|a, b| {
        (a.0.unwrap_or(usize::MAX), a.2.clone()).cmp(&(b.0.unwrap_or(usize::MAX), b.2.clone()))
    });
    let mut out_children = Vec::with_capacity(children.len());
    for (_idx, _created, child_id, row) in children {
        // Steering history from the child session's control rows.
        let sid_raw = row.get("session_id").and_then(|s| s.as_u64()).unwrap_or(0);
        let steer_events: Vec<serde_json::Value> = if sid_raw == 0 {
            Vec::new()
        } else {
            match state.deps.session.get_session(SessionId::new(sid_raw)) {
                Ok(Some(ch)) => match ch.orchestrator_ctl_all() {
                    Ok(ctl) => ctl
                        .iter()
                        .map(|c| {
                            serde_json::json!({
                                "kind": c.control,
                                "seq": c.seq,
                                "applied_ms": c.applied_ms,
                            })
                        })
                        .collect(),
                    Err(e) => {
                        return wire_status(internal_graph_err(format!(
                            "steering rows of child {child_id}: {}",
                            e.message
                        )))
                    }
                },
                Ok(None) => {
                    return wire_status(internal_graph_err(format!(
                        "graph assembly: child {child_id} names missing session {sid_raw}"
                    )))
                }
                Err(e) => return api_err(&e),
            }
        };
        // Latest durable merge envelope + its parts.
        let merge_prefix = format!("{run}/{child_id}/merge/");
        let mut envelopes: Vec<(u64, serde_json::Value)> = Vec::new();
        for (kind, key, value) in &facts {
            if kind != ORCH_MERGE_KIND {
                continue;
            }
            let Some(rest) = key.strip_prefix(&merge_prefix) else {
                continue;
            };
            let Some((_cs, seq_s)) = rest.rsplit_once('/') else {
                return wire_status(internal_graph_err(format!("hostile merge row key {key:?}")));
            };
            let Ok(seq) = seq_s.parse::<u64>() else {
                return wire_status(internal_graph_err(format!("hostile merge row key {key:?}")));
            };
            let Ok(v) = serde_json::from_str::<serde_json::Value>(value) else {
                return wire_status(internal_graph_err(format!(
                    "merge envelope {key:?} is not valid JSON"
                )));
            };
            envelopes.push((seq, v));
        }
        envelopes.sort_by_key(|(seq, _)| *seq);
        let merge: Option<serde_json::Value> = envelopes.last().map(|(seq, env)| {
            let cs_id = env
                .get("cs_id")
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string();
            let part_prefix = format!("{run}/{child_id}/merge/{cs_id}/{seq}/part/");
            let mut merged: Vec<serde_json::Value> = Vec::new();
            let mut rejected: Vec<serde_json::Value> = Vec::new();
            let mut conflicts: Vec<serde_json::Value> = Vec::new();
            for part in ["merged", "rejected", "conflicts"] {
                let key = format!("{part_prefix}{part}");
                let Ok(Some(chunks)) = orchestrator_chunks_of(&facts, ORCH_MERGE_PART_KIND, &key)
                else {
                    continue;
                };
                for c in &chunks {
                    let Ok(items) = serde_json::from_str::<serde_json::Value>(c) else {
                        continue;
                    };
                    let Some(items) = items.as_array() else {
                        continue;
                    };
                    for item in items {
                        match part {
                            "merged" | "rejected" => {
                                if let Some(p) = item.as_str() {
                                    (if part == "merged" {
                                        &mut merged
                                    } else {
                                        &mut rejected
                                    })
                                    .push(serde_json::json!(p));
                                }
                            }
                            _ => {
                                if let Some(p) = item.as_array() {
                                    if p.len() == 2 {
                                        conflicts.push(serde_json::json!([
                                            p[0].as_str().unwrap_or(""),
                                            p[1].as_str().unwrap_or("")
                                        ]));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            merged.sort_by(|a, b| a.as_str().unwrap_or("").cmp(b.as_str().unwrap_or("")));
            rejected.sort_by(|a, b| a.as_str().unwrap_or("").cmp(b.as_str().unwrap_or("")));
            conflicts.sort_by(|a, b| {
                let ka = a[0].as_str().unwrap_or("");
                let kb = b[0].as_str().unwrap_or("");
                ka.cmp(kb)
                    .then_with(|| a[1].as_str().unwrap_or("").cmp(b[1].as_str().unwrap_or("")))
            });
            serde_json::json!({
                "change_set_id": cs_id,
                "merged": merged,
                "rejected": rejected,
                "conflicts": conflicts,
            })
        });
        out_children.push(serde_json::json!({
            "child_id": child_id,
            "session_id": row.get("session_id").cloned().unwrap_or(serde_json::Value::Null),
            "operation_id": row.get("operation_id").cloned().unwrap_or(serde_json::Value::Null),
            "worktree_id": row.get("worktree_id").cloned().unwrap_or(serde_json::Value::Null),
            "ownership": row.get("ownership").cloned().unwrap_or(serde_json::Value::Null),
            "state": row.get("state").cloned().unwrap_or(serde_json::Value::Null),
            "budget": row.get("budget_max_tokens").cloned().unwrap_or(serde_json::Value::Null),
            "capabilities": row.get("permissions").cloned().unwrap_or_else(|| serde_json::json!([])),
            "plan_step_index": row.get("item_id").and_then(|i| i.as_str()).and_then(step_index),
            "steer_events": steer_events,
            "merge": merge,
        }));
    }
    Json(serde_json::json!({
        "plan_id": run,
        "goal": goal,
        "state": root_state,
        "work_items": work_items
            .iter()
            .zip(&step_states)
            .map(|(w, st)| serde_json::json!({
                "item_id": w.get("id").cloned().unwrap_or(serde_json::Value::Null),
                "kind": w.get("kind").cloned().unwrap_or(serde_json::Value::Null),
                "state": st,
            }))
            .collect::<Vec<_>>(),
        "children": out_children,
    }))
    .into_response()
}

/// Strict native request bodies (deny_unknown_fields — a typo is a 400).
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeSteerBody {
    text: String,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeModelBody {
    model: String,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeBudgetBody {
    max_tokens: Option<u64>,
    max_cost_micro: Option<u64>,
}

/// Strict query DTO: `?session=<id>` only (mirrors the graph endpoint).
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeAgentsQuery {
    session: String,
}

/// Chunk-aware value reader for one durable row: the PLAIN value when the
/// row is a single JSON document, the header + `key/cNNN` chunks joined
/// when the row was written chunked. Missing rows are `None`.
pub(crate) fn orchestrator_row_value(
    facts: &[(String, String, String)],
    kind: &str,
    key: &str,
) -> Result<Option<String>, String> {
    let header: Option<&(String, String, String)> =
        facts.iter().find(|(k, kk, _)| k == kind && kk == key);
    let is_header = header.is_some_and(|(_, _, v)| {
        serde_json::from_str::<serde_json::Value>(v)
            .ok()
            .and_then(|j| j.get("chunks").and_then(|c| c.as_u64()))
            .unwrap_or(0)
            > 0
    });
    if let Some(row) = header {
        if !is_header {
            return Ok(Some(row.2.clone()));
        }
        if let Some(chunks) = orchestrator_chunks_of(facts, kind, key)? {
            return Ok(Some(chunks.concat()));
        }
    }
    Ok(None)
}

/// The live state of one child: its durable ChildState, overlaid with the
/// child session's durable drive phase — a drive parked at a pause boundary
/// reads Waiting even when the registry row has not flipped yet.
pub(crate) fn child_live_state(
    row: &faktor_orchestrator::runtime::ChildRuntime,
    drive: &faktor_session::child::DriveState,
) -> &'static str {
    if !row.state.is_terminal() && drive.phase == faktor_session::child::ChildPhase::Waiting {
        return "Waiting";
    }
    match row.state {
        faktor_orchestrator::ChildState::Running => "Running",
        faktor_orchestrator::ChildState::Paused => "Paused",
        faktor_orchestrator::ChildState::Waiting => "Waiting",
        faktor_orchestrator::ChildState::Cancelled => "Cancelled",
        faktor_orchestrator::ChildState::Done => "Done",
        faktor_orchestrator::ChildState::Failed => "Failed",
    }
}

/// The effective model of one child: the drive state's applied model wins,
/// then the durable identity row, then the registry row's policy, then the
/// run's default model.
pub(crate) fn child_model(
    drive: &faktor_session::child::DriveState,
    identity: Option<&faktor_session::child::ChildIdentity>,
    row: &faktor_orchestrator::runtime::ChildRuntime,
    default_model: Option<&str>,
) -> Option<String> {
    if !drive.current_model.is_empty() {
        return Some(drive.current_model.clone());
    }
    if let Some(id) = identity {
        if !id.model.is_empty() {
            return Some(id.model.clone());
        }
    }
    row.model_policy
        .model
        .clone()
        .or_else(|| default_model.map(str::to_string))
}

/// The durable "latest result" summary of one child (wave-13/14 records):
/// its task goal plus the latest durable merge envelope's counts.
pub(crate) fn child_result_entry(
    state: &AppState,
    facts: &[(String, String, String)],
    run: &str,
    row: &faktor_orchestrator::runtime::ChildRuntime,
) -> Result<serde_json::Value, ApiError> {
    let mut summary = String::new();
    if let Ok(Some(h)) = state
        .deps
        .session
        .get_session(SessionId::new(row.session_id))
    {
        summary = h
            .orchestrator_child_identity_get()
            .ok()
            .flatten()
            .map(|i| i.task_goal)
            .unwrap_or_default();
    }
    let mut merge: Option<serde_json::Value> = None;
    let prefix = format!("{run}/{}/merge/", row.child_id);
    let mut best: Option<(u64, &str)> = None;
    for (kind, key, value) in facts {
        if kind != ORCH_MERGE_KIND {
            continue;
        }
        if let Some(rest) = key.strip_prefix(&prefix) {
            if let Some((_cs, seq_s)) = rest.rsplit_once('/') {
                if let Ok(seq) = seq_s.parse::<u64>() {
                    if best.map(|(b, _)| seq > b).unwrap_or(true) {
                        best = Some((seq, value));
                    }
                }
            }
        }
    }
    if let Some((_seq, value)) = best {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(value) {
            merge = Some(serde_json::json!({
                "change_set_id": v.get("cs_id").cloned().unwrap_or(serde_json::Value::Null),
                "merged": v.get("merged_count").cloned().unwrap_or(serde_json::Value::Null),
                "rejected": v.get("rejected_count").cloned().unwrap_or(serde_json::Value::Null),
                "conflicts": v.get("conflict_count").cloned().unwrap_or(serde_json::Value::Null),
            }));
        }
    }
    Ok(serde_json::json!({
        "summary": summary,
        "merge": merge,
    }))
}

/// The live session-state tag of one TaskExecutor in-session run (single
/// task per session; the session's live row state is the run's state).
pub(crate) fn session_run_state_tag(agent_state: faktor_core::state::AgentState) -> &'static str {
    use faktor_core::state::AgentState::*;
    match agent_state {
        Completed | ReadyForNextTurn => "Done",
        Cancelled => "Cancelled",
        FailedRecoverable | FailedPermanent => "Failed",
        NeedsUserInput => "Blocked",
        Idle => "Pending",
        _ => "Running",
    }
}

/// One child entry of the listing.
#[allow(clippy::too_many_arguments)]
pub(crate) fn native_child_entry(
    state: &AppState,
    facts: &[(String, String, String)],
    run: &str,
    row: faktor_orchestrator::runtime::ChildRuntime,
    default_model: Option<&str>,
) -> Result<serde_json::Value, ApiError> {
    let session = match state
        .deps
        .session
        .get_session(SessionId::new(row.session_id))
    {
        Ok(Some(h)) => h,
        Ok(None) => {
            return Err(internal_graph_err(format!(
                "agents listing: child {} names missing session {}",
                row.child_id, row.session_id
            )))
        }
        Err(e) => return Err(faktor_protocol::error::from_core(&e)),
    };
    let drive = session.orchestrator_drive_state_get().map_err(|e| {
        internal_graph_err(format!(
            "drive state of child {}: {}",
            row.child_id, e.message
        ))
    })?;
    let identity = session.orchestrator_child_identity_get().map_err(|e| {
        internal_graph_err(format!(
            "identity row of child {}: {}",
            row.child_id, e.message
        ))
    })?;
    let goal = identity
        .as_ref()
        .map(|i| i.task_goal.clone())
        .unwrap_or_default();
    let progress = state
        .deps
        .agent
        .progress_view(SessionId::new(row.session_id));
    let result = child_result_entry(state, facts, run, &row)?;
    Ok(serde_json::json!({
        "agent_id": row.child_id,
        "kind": "child",
        "run_id": run,
        "session_id": row.session_id,
        "worktree_id": row.worktree_id,
        "item_id": row.item_id,
        "item_kind": row.kind,
        "state": child_live_state(&row, &drive),
        "model": child_model(&drive, identity.as_ref(), &row, default_model),
        "budget": row.budget_max_tokens,
        "ownership": row.ownership,
        "capabilities": row.permissions,
        "progress": progress,
        "result": result,
        "goal": goal,
    }))
}

/// `GET /native/agents?session=<id>` (and `/native/session/{id}/agents`):
/// the real agent listing of one session — every orchestrated run's
/// children plus the parent's own task runs. Empty ONLY when the session
/// genuinely has no task run. Durable rows only; tampered rows are loud
/// 500s, never a silently partial listing.
pub(crate) fn native_agents_body(
    state: &AppState,
    handle: &faktor_session::SessionHandle,
) -> Result<Vec<serde_json::Value>, ApiError> {
    let parent = handle.id();
    let facts = orchestrator_graph_facts(handle).map_err(internal_graph_err)?;

    let push_run = |runs: &mut Vec<String>, run: &str| {
        if !runs.iter().any(|r| r == run) {
            runs.push(run.to_string());
        }
    };
    let mut in_session: Vec<String> = Vec::new();
    let mut orchestrated: Vec<String> = Vec::new();
    for (kind, key, _) in &facts {
        match kind.as_str() {
            TASK_RUN_ROW_KIND => push_run(&mut in_session, key),
            ORCH_PLAN_KIND => push_run(&mut orchestrated, key),
            ORCH_REGISTRY_KIND => {
                if let Some(run) = key.rsplit_once('/').map(|(r, _c)| r) {
                    if !run.is_empty() {
                        push_run(&mut orchestrated, run);
                    }
                }
            }
            _ => {}
        }
    }
    in_session.sort();
    orchestrated.sort();

    let mut entries: Vec<serde_json::Value> = Vec::new();
    let mut push = |e: serde_json::Value| {
        if entries.len() < MAX_AGENT_ENTRIES {
            entries.push(e);
        }
    };

    // (a) In-session task runs: the parent's own single-item runs.
    for key in &in_session {
        let value = match orchestrator_row_value(&facts, TASK_RUN_ROW_KIND, key) {
            Ok(Some(v)) => v,
            Ok(None) => continue,
            Err(m) => return Err(internal_graph_err(m)),
        };
        let row = match faktor_orchestrator::runtime::task_executor::TaskRunRow::decode(&value) {
            Ok(r) => r,
            Err(m) => {
                return Err(internal_graph_err(format!(
                    "stored taskexec run row {TASK_RUN_ROW_KIND}/{key}: {m}"
                )))
            }
        };
        let session_row = handle
            .row()
            .map_err(|e| faktor_protocol::error::from_core(&e))?;
        let budget = handle
            .get_task(session_row.task_id)
            .map_err(|e| faktor_protocol::error::from_core(&e))?
            .and_then(|t| t.budget.max_tokens);
        let progress = state.deps.agent.progress_view(parent);
        // State: the durable typed task row wins when terminal (wave-24) —
        // a cancelled/verified/failed run reads as such even though the
        // session itself is parked.
        let run_state = in_session_run_state_tag(handle, &session_row);
        push(serde_json::json!({
            "agent_id": key,
            "kind": "self",
            "run_id": key,
            "session_id": parent.raw(),
            "worktree_id": session_row.worktree_id.raw(),
            "goal": row.goal,
            "item_ids": row.item_ids,
            "state": run_state,
            "model": session_row.model,
            "budget": budget,
            "ownership": "self",
            "capabilities": [],
            "progress": progress,
            "result": serde_json::Value::Null,
        }));
    }

    // (b) Orchestrated runs: the parent run + its real children.
    for run in &orchestrated {
        // The durable plan row (goal, steps, default model, owner).
        let plan_json: Option<serde_json::Value> =
            match orchestrator_row_value(&facts, ORCH_PLAN_KIND, run) {
                Ok(Some(v)) => match serde_json::from_str::<serde_json::Value>(&v) {
                    Ok(j) => Some(j),
                    Err(_) => {
                        return Err(internal_graph_err(format!(
                        "stored row {ORCH_PLAN_KIND}/{run} of session {parent} is not valid JSON"
                    )))
                    }
                },
                Ok(None) => None,
                Err(m) => return Err(internal_graph_err(m)),
            };
        let default_model = plan_json
            .as_ref()
            .and_then(|v| v.get("default_model"))
            .and_then(|m| m.as_str())
            .map(str::to_string);
        let plan_goal = plan_json
            .as_ref()
            .and_then(|v| v.get("plan"))
            .and_then(|p| p.get("goal"))
            .and_then(|g| g.as_str())
            .unwrap_or("")
            .to_string();
        // Plan steps in plan order: (id, kind, depends_on).
        let work_items: Vec<(String, String, Vec<String>)> = plan_json
            .as_ref()
            .and_then(|v| v.get("plan"))
            .and_then(|p| p.get("work_items"))
            .and_then(|a| a.as_array())
            .cloned()
            .unwrap_or_default()
            .iter()
            .filter_map(|w| {
                let id = w.get("id").and_then(|i| i.as_str())?.to_string();
                let kind = w
                    .get("kind")
                    .and_then(|k| k.as_str())
                    .unwrap_or("")
                    .to_string();
                let deps = w
                    .get("depends_on")
                    .and_then(|d| d.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_str().map(str::to_string))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                Some((id, kind, deps))
            })
            .collect();

        // Every durable child row of this run (typed; tampered rows loud).
        let prefix = format!("{run}/");
        let mut children: Vec<(i64, String, faktor_orchestrator::runtime::ChildRuntime)> =
            Vec::new();
        for (kind, key, _) in &facts {
            if kind != ORCH_REGISTRY_KIND {
                continue;
            }
            let Some(rest) = key.strip_prefix(&prefix) else {
                continue;
            };
            if rest.is_empty() || rest.contains('/') {
                return Err(internal_graph_err(format!(
                    "hostile registry row key {key:?} under run {run}"
                )));
            }
            let value = match orchestrator_row_value(&facts, ORCH_REGISTRY_KIND, key) {
                Ok(Some(v)) => v,
                Ok(None) => continue,
                Err(m) => return Err(internal_graph_err(m)),
            };
            let row: faktor_orchestrator::runtime::ChildRuntime =
                serde_json::from_str(&value).map_err(|_| {
                    internal_graph_err(format!(
                        "stored row {ORCH_REGISTRY_KIND}/{key} of session {parent} is not valid JSON"
                    ))
                })?;
            children.push((row.created_ms, rest.to_string(), row));
        }
        children.sort_by_key(|a| (a.0, a.1.clone()));

        // The parent's own task run: state derived with the orchestrator's
        // re-attach semantics — a durable child moves its step to Running
        // first, its terminal state maps Done/Failed/Cancelled, and Pending
        // steps behind failed/cancelled dependencies are Blocked.
        let mut step_states: Vec<&'static str> = work_items
            .iter()
            .map(|(id, _, _)| {
                let mut st = "Pending";
                for (_, _, c) in &children {
                    if &c.item_id != id {
                        continue;
                    }
                    if st == "Pending" {
                        st = "Running";
                    }
                    let t = match c.state {
                        faktor_orchestrator::ChildState::Done => "Done",
                        faktor_orchestrator::ChildState::Cancelled => "Cancelled",
                        faktor_orchestrator::ChildState::Failed => "Failed",
                        _ => "Running",
                    };
                    if (st == "Running" || st == "Pending") && t != "Running" {
                        st = t;
                    }
                }
                st
            })
            .collect();
        for (i, (_id, _kind, deps)) in work_items.iter().enumerate() {
            if step_states[i] != "Pending" {
                continue;
            }
            let step_of = |dep: &str| work_items.iter().position(|(d, _, _)| d == dep);
            if deps
                .iter()
                .filter_map(|d| step_of(d))
                .any(|j| matches!(step_states[j], "Failed" | "Cancelled"))
            {
                step_states[i] = "Blocked";
            }
        }
        let root_state = if step_states.iter().all(|s| *s == "Done") {
            "Done"
        } else {
            [
                "Failed",
                "Cancelled",
                "Running",
                "Blocked",
                "Paused",
                "Pending",
            ]
            .iter()
            .find(|wanted| step_states.contains(wanted))
            .copied()
            .unwrap_or("Pending")
        };
        let owner_wt = handle
            .row()
            .map_err(|e| faktor_protocol::error::from_core(&e))?
            .worktree_id
            .raw();
        let progress = state.deps.agent.progress_view(parent);
        push(serde_json::json!({
            "agent_id": run,
            "kind": "self",
            "run_id": run,
            "session_id": parent.raw(),
            "worktree_id": owner_wt,
            "goal": plan_goal,
            "item_ids": work_items.iter().map(|(id, _, _)| id.clone()).collect::<Vec<_>>(),
            "state": root_state,
            "model": serde_json::Value::Null,
            "budget": serde_json::Value::Null,
            "ownership": "self",
            "capabilities": [],
            "progress": progress,
            "result": serde_json::Value::Null,
        }));
        for (_created, _child_id, row) in children {
            push(native_child_entry(
                state,
                &facts,
                run,
                row,
                default_model.as_deref(),
            )?);
        }
    }
    Ok(entries)
}

/// Shared error mapping of one orchestrator control into the wire.
pub(crate) fn exec_error_response(e: &faktor_orchestrator::runtime::ExecError) -> Response {
    let (code, status) = match e {
        faktor_orchestrator::runtime::ExecError::NotFound(_) => ("not_found", 404),
        faktor_orchestrator::runtime::ExecError::Conflict(_) => ("conflict", 409),
        faktor_orchestrator::runtime::ExecError::InvalidState(_) => ("conflict", 409),
        faktor_orchestrator::runtime::ExecError::CeilingExceeded { .. } => {
            ("ceiling_exceeded", 429)
        }
        faktor_orchestrator::runtime::ExecError::Oversized(_)
        | faktor_orchestrator::runtime::ExecError::InvalidPlan(_)
        | faktor_orchestrator::runtime::ExecError::InvalidApproval(_) => ("malformed", 400),
        _ => ("internal", 500),
    };
    let e = ApiError {
        code,
        message: e.to_string(),
        http_status: status,
        retryable: false,
    };
    wire_status(e)
}

/// One control enqueue on a child of the active execution. Hostile/unknown
/// child ids are typed 404; terminal-state refusals are 409; the durable
/// row's exactly-once state answers as `{queuedSeq, applied}`.
pub(crate) fn agent_control(
    state: &AppState,
    child_id: &str,
    control: faktor_session::child::ChildControl,
) -> Response {
    match state.deps.orchestrator.control_child(child_id, control) {
        Ok(ack) => Json(serde_json::json!({
            "queuedSeq": ack.queued_seq,
            "applied": ack.applied,
        }))
        .into_response(),
        Err(e) => exec_error_response(&e),
    }
}

/// Whether the model selector is served by a registered provider (the
/// catalog the `/models` endpoint lists).
pub(crate) fn model_known(state: &AppState, model: &str) -> bool {
    state
        .deps
        .agent
        .deps()
        .providers
        .all()
        .iter()
        .any(|p| p.known_models().iter().any(|m| m == model))
}

/// `GET /native/agents?session=<id>` — see [`native_agents_body`].
/// Hostile/missing session ids are typed 404 (a listing without a session
/// is never an empty phantom).
pub(crate) async fn native_agents(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<NativeAgentsQuery>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let Ok(raw) = q.session.parse::<u64>() else {
        return wire_status(not_found(&format!("invalid session id {:?}", q.session)));
    };
    if raw == 0 {
        return wire_status(not_found("session id cannot be 0"));
    }
    let sid = SessionId::new(raw);
    let handle = match state.deps.session.get_session(sid) {
        Ok(Some(h)) => h,
        Ok(None) => return wire_status(not_found(&format!("session {sid}"))),
        Err(e) => return api_err(&e),
    };
    match native_agents_body(&state, &handle) {
        Ok(entries) => Json(entries).into_response(),
        Err(e) => wire_status(e),
    }
}

// --------------------------------------------------- native task runs (wave-24)
// The ONE task-start surface of the daemon's HTTP layer: POST starts a task
// through `state.deps.tasks.start_task` (the TaskExecutor — the same single
// authority the daemon graph wires; there is no second start architecture
// reachable from the server), GET lists the session's durable task runs
// with per-run state, and the per-run GET/cancel read and drive one run.
// Request bodies parse with the strict native DTOs (deny_unknown_fields —
// a typo or hostile value is a 400). Unknown sessions and unknown runs are
// typed 404s; run-level conflicts (terminal runs, live crash residue) are
// typed 409s.

/// `GET /native/agents/{child_id}/pause` — durable Pause control; applied
/// at the child's next safe reasoning boundary (`applied: null` until the
/// drive acks it exactly once).
pub(crate) async fn native_agent_pause(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(child_id): Path<String>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    agent_control(
        &state,
        &child_id,
        faktor_session::child::ChildControl::Pause,
    )
}

pub(crate) async fn native_agent_resume(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(child_id): Path<String>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    agent_control(
        &state,
        &child_id,
        faktor_session::child::ChildControl::Resume,
    )
}

pub(crate) async fn native_agent_cancel(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(child_id): Path<String>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    agent_control(
        &state,
        &child_id,
        faktor_session::child::ChildControl::Cancel,
    )
}

pub(crate) async fn native_agent_retry(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(child_id): Path<String>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    agent_control(
        &state,
        &child_id,
        faktor_session::child::ChildControl::Retry,
    )
}

/// `POST /native/agents/{child_id}/steer` — body `{"text": ...}` (bounded
/// note, durable row, applied at the next boundary).
pub(crate) async fn native_agent_steer(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(child_id): Path<String>,
    body: Result<Json<NativeSteerBody>, axum::extract::rejection::JsonRejection>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let Json(body) = match body {
        Ok(b) => b,
        Err(_) => return wire_status(malformed_body("invalid native steer body")),
    };
    agent_control(
        &state,
        &child_id,
        faktor_session::child::ChildControl::Steer { note: body.text },
    )
}

/// `POST /native/agents/{child_id}/model` — body `{"model": "..."}`. The
/// selector must be served by a registered provider (catalog check), else
/// a typed 404. Takes effect at the next provider selection.
pub(crate) async fn native_agent_model(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(child_id): Path<String>,
    body: Result<Json<NativeModelBody>, axum::extract::rejection::JsonRejection>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let Json(body) = match body {
        Ok(b) => b,
        Err(_) => return wire_status(malformed_body("invalid native model body")),
    };
    if !model_known(&state, &body.model) {
        let e = ApiError {
            code: "not_found",
            message: format!(
                "model {:?} is not served by any registered provider",
                body.model
            ),
            http_status: 404,
            retryable: false,
        };
        return wire_status(e);
    }
    agent_control(
        &state,
        &child_id,
        faktor_session::child::ChildControl::ChangeModel { model: body.model },
    )
}

/// `POST /native/agents/{child_id}/budget` — body is EXACTLY ONE of
/// `{"max_tokens": N}` or `{"max_cost_micro": N}` (deny_unknown_fields;
/// hostile bodies are typed 400s, 0 on either axis is refused as ambiguous —
/// the store reads a NULL/0 cap as "unlimited", so an explicit zero cap can
/// never mean "remove the cap"). The body maps onto the typed budget-change
/// vocabulary ([`faktor_session::child::ChildBudgetChange`]); each axis is a
/// synchronous durable effect:
/// - `ChangeTokenBudget` → the historic token path: the wave-9 task-row
///   `max_tokens` patch, delivered exactly once through the `ChangeBudget`
///   control queue (acked at enqueue; `applied: true`);
/// - `ChangeCostBudget` → the child cost-cap effect (audit 9/H): the child's
///   durable task-row `max_cost_micro` plus its budget-scope enrollment under
///   the run's root row (root + child subscope admission). A direct durable
///   effect — no drive-boundary consumer exists for it, so no queue row is
///   written (`queuedSeq: null`).
pub(crate) async fn native_agent_budget(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(child_id): Path<String>,
    body: Result<Json<NativeBudgetBody>, axum::extract::rejection::JsonRejection>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let Json(body) = match body {
        Ok(b) => b,
        Err(_) => return wire_status(malformed_body("invalid native budget body")),
    };
    let change = match (body.max_tokens, body.max_cost_micro) {
        (Some(max_tokens), None) => {
            faktor_session::child::ChildBudgetChange::ChangeTokenBudget { max_tokens }
        }
        (None, Some(max_cost_micro)) => {
            faktor_session::child::ChildBudgetChange::ChangeCostBudget { max_cost_micro }
        }
        _ => {
            return wire_status(malformed_body(
                "send exactly one of max_tokens or max_cost_micro",
            ))
        }
    };
    if let Err(e) = change.validate() {
        return wire_status(malformed_body(&e.to_string()));
    }
    match change {
        faktor_session::child::ChildBudgetChange::ChangeTokenBudget { max_tokens } => {
            agent_control(
                &state,
                &child_id,
                faktor_session::child::ChildControl::ChangeBudget { max_tokens },
            )
        }
        faktor_session::child::ChildBudgetChange::ChangeCostBudget { max_cost_micro } => {
            child_cost_cap_change(&state, &child_id, max_cost_micro)
        }
    }
}

/// The synchronous durable effect of a ChangeCostBudget on one child (see
/// [`native_agent_budget`]): the child's task-row cost cap is patched and
/// its budget scope is enrolled under the run root through the durable
/// ledger. Terminal children refuse (409, mirroring the ChangeBudget
/// guard); unknown children are typed 404s.
pub(crate) fn child_cost_cap_change(
    state: &AppState,
    child_id: &str,
    max_cost_micro: u64,
) -> Response {
    let row = match state.deps.orchestrator.child(child_id) {
        Ok(Some(r)) => r,
        Ok(None) => {
            let e = ApiError {
                code: "not_found",
                message: format!("unknown child {child_id}"),
                http_status: 404,
                retryable: false,
            };
            return wire_status(e);
        }
        Err(e) => return exec_error_response(&e),
    };
    let session = match state
        .deps
        .session
        .get_session(SessionId::new(row.session_id))
    {
        Ok(Some(h)) => h,
        Ok(None) => {
            return wire_status(ApiError {
                code: "not_found",
                message: format!("child session {}", row.session_id),
                http_status: 404,
                retryable: false,
            })
        }
        Err(e) => return api_err(&e),
    };
    let terminal = match session.state() {
        Ok(s) => s.is_terminal(),
        Err(e) => return api_err(&e),
    };
    if row.state.is_terminal() || terminal {
        let e = ApiError {
            code: "conflict",
            message: format!("cannot change the budget of {child_id}: state is terminal"),
            http_status: 409,
            retryable: false,
        };
        return wire_status(e);
    }
    match state.deps.budgets.change_child_scope_cap(
        SessionId::new(row.session_id),
        child_id,
        Some(max_cost_micro),
    ) {
        Ok(()) => Json(serde_json::json!({
            "queuedSeq": serde_json::Value::Null,
            "applied": true,
        }))
        .into_response(),
        Err(e) => {
            let core: faktor_core::Error = e.into();
            api_err(&core)
        }
    }
}
