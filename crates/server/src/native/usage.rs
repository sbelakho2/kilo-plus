//! Authoritative usage reads: per-session and cross-session aggregates.

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use faktor_core::id::SessionId;

use super::*;
use crate::api::AppState;

/// `GET /native/usage` — cross-session usage aggregate.
///
/// Two documented layers:
///
/// - `totals.budget|spent` + `perSession` keep the FROZEN legacy view over
///   the memory facts of kind `usage` (keys `budget`/`spent`). No runtime
///   path writes those facts — they exist for simulators/tooling only — so
///   the totals are honest zeros when nothing recorded them.
/// - `durable` is the AUTHORITATIVE aggregate over the persisted rows:
///   every `provider_call` row of every session (tokens = the persisted
///   `tokens_in`+`tokens_out` totals — cache reads/writes and reasoning are
///   folded into these two counters by the usage settlement BEFORE
///   persistence, so the rows carry exactly what is summed here), the
///   per-row prefix observations, and every `cost_reservation` row of every
///   typed task (in-flight reserved/dispatched + settled/refunded/uncertain
///   with the folded spends and the provider-reported micro totals). Hostile
///   rows can never break the
///   aggregate: unparseable reservation JSON is skipped per row, sessions
///   are read defensively, and the reservation scan is bounded — when the
///   per-task scan cap is hit the response says `truncated: true` instead of
///   pretending to be exact.
pub(crate) async fn native_usage(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    const MAX_SESSIONS: usize = 10_000;
    let mut budget_total: i64 = 0;
    let mut spent_total: i64 = 0;
    let mut per_session: Vec<serde_json::Value> = Vec::new();
    // A session that vanishes mid-list is skipped, never fatal (the wire
    // list has the same convention).
    let mut sessions = match state.deps.session.list_sessions(None) {
        Ok(s) => s,
        Err(e) => return api_err(&e),
    };
    sessions.truncate(MAX_SESSIONS);
    // ---- durable layer (authoritative; P0-63)
    let mut durable_tokens: u64 = 0;
    let mut sessions_with_calls: u64 = 0;
    let mut prefix_rows: u64 = 0;
    let mut prefix_tokens: u64 = 0;
    let mut prefix_stability_observations: u64 = 0;
    let mut res_open_count: u64 = 0;
    let mut res_open_predicted: u64 = 0;
    let mut res_settled_count: u64 = 0;
    let mut res_settled_predicted: u64 = 0;
    let mut res_settled_spent: u64 = 0;
    let mut res_settled_reported: u64 = 0;
    let mut res_refunded_count: u64 = 0;
    let mut res_refunded_predicted: u64 = 0;
    let mut res_uncertain_count: u64 = 0;
    let mut res_uncertain_predicted: u64 = 0;
    let mut task_budget_spent_micro: u64 = 0;
    let mut scan_truncated = false;
    for handle in &sessions {
        let store = state.deps.session.store();
        let sid = handle.id();
        let session_tokens = match store.session_usage_tokens(sid) {
            Ok(t) => t,
            Err(_) => continue, // vanished/corrupt session row mid-scan
        };
        if session_tokens > 0 {
            sessions_with_calls += 1;
        }
        durable_tokens = durable_tokens.saturating_add(session_tokens);
        match store.provider_call_prefix_rows(sid) {
            Ok(rows) => {
                prefix_rows = prefix_rows.saturating_add(rows.len() as u64);
                for r in rows {
                    prefix_tokens = prefix_tokens.saturating_add(r.prompt_tokens as u64);
                    if r.prefix_stability.is_some() {
                        prefix_stability_observations += 1;
                    }
                }
            }
            Err(_) => continue, // a hostile prefix row fails this session's read loudly
        }
        let tasks = match handle.list_tasks() {
            Ok(t) => t,
            Err(_) => continue,
        };
        for task in tasks.iter().take(MAX_NATIVE_LIST) {
            if let Ok(Some(cost)) = store.cost_task_row(sid, task.task_id) {
                task_budget_spent_micro =
                    task_budget_spent_micro.saturating_add(cost.spent_cost_micro);
            }
            let rows =
                match store.cost_reservations_of(sid, task.task_id, MAX_NATIVE_RESERVATIONS_SCAN) {
                    Ok(r) => r,
                    Err(_) => continue,
                };
            if rows.len() as i64 >= MAX_NATIVE_RESERVATIONS_SCAN {
                scan_truncated = true;
            }
            for row in rows {
                // Group vocabulary is the schema v17+ state set: in-flight
                // `reserved`/`dispatched`, terminal `settled`/`refunded`
                // and crash-recovery `uncertain` — the v15 `open`/
                // `abandoned` names were renamed at v16/v17 and never
                // occur in a migrated store.
                match row.status.as_str() {
                    "reserved" | "dispatched" => {
                        res_open_count += 1;
                        res_open_predicted = res_open_predicted.saturating_add(row.predicted_micro);
                    }
                    "settled" => {
                        res_settled_count += 1;
                        res_settled_predicted =
                            res_settled_predicted.saturating_add(row.predicted_micro);
                        res_settled_spent =
                            res_settled_spent.saturating_add(reservation_settled_spent_micro(&row));
                        res_settled_reported = res_settled_reported
                            .saturating_add(reservation_provider_reported_micro(&row));
                    }
                    "refunded" => {
                        res_refunded_count += 1;
                        res_refunded_predicted =
                            res_refunded_predicted.saturating_add(row.predicted_micro);
                    }
                    "uncertain" => {
                        res_uncertain_count += 1;
                        res_uncertain_predicted =
                            res_uncertain_predicted.saturating_add(row.predicted_micro);
                    }
                    // Unknown status strings (hostile rows) contribute
                    // nothing: the aggregate never guesses.
                    _ => {}
                }
            }
        }
    }
    let mut durable = serde_json::json!({
        "sessionsWithCalls": sessions_with_calls,
        "providerCalls": {
            "tokens": durable_tokens,
            "prefixObservations": prefix_rows,
            "prefixTokens": prefix_tokens,
            "prefixStabilityObservations": prefix_stability_observations,
        },
        "taskSpend": { "settledCostMicro": task_budget_spent_micro },
        "reservations": {
            "open": { "count": res_open_count, "predictedMicro": res_open_predicted },
            "settled": {
                "count": res_settled_count,
                "predictedMicro": res_settled_predicted,
                "spentMicro": res_settled_spent,
                "providerReportedMicro": res_settled_reported,
            },
            "refunded": { "count": res_refunded_count, "predictedMicro": res_refunded_predicted },
            "uncertain": {
                "count": res_uncertain_count,
                "predictedMicro": res_uncertain_predicted,
            },
        },
    });
    if scan_truncated {
        durable["truncated"] = serde_json::json!(true);
    }
    for handle in &sessions {
        let facts = match handle.memory_facts() {
            Ok(f) => f,
            Err(_) => continue,
        };
        let mut budget: Option<i64> = None;
        let mut spent: Option<i64> = None;
        for (kind, key, value) in facts {
            if kind != "usage" {
                continue;
            }
            let parsed = value.parse::<i64>().ok();
            match key.as_str() {
                "budget" => budget = parsed,
                "spent" => spent = parsed,
                _ => {}
            }
        }
        if budget.is_none() && spent.is_none() {
            continue;
        }
        budget_total = budget_total.saturating_add(budget.unwrap_or(0));
        spent_total = spent_total.saturating_add(spent.unwrap_or(0));
        per_session.push(serde_json::json!({
            "sessionId": handle.id().to_string(),
            "budget": budget,
            "spent": spent,
        }));
    }
    Json(serde_json::json!({
        "sessions": sessions.len(),
        "totals": { "budget": budget_total, "spent": spent_total },
        "perSession": per_session,
        "durable": durable,
    }))
    .into_response()
}

// ------------------------------------------------------ native v1: audits 62-64
// P0-62 (session-owned terminal projection), P0-63 (authoritative usage from
// the durable rows) and P0-64 (cursor messages, journal event pages,
// providers, task budget/progress, verification evidence). Every handler is
// auth-gated like every native route; query/body DTOs are strict
// (deny_unknown_fields — a typo is a 400); hostile ids are 400
// (unparseable/0) or typed 404 (unknown).

/// Reservation rows scanned per task for a usage aggregate. Every paid
/// model call settles exactly one reservation row; a hostile row flood past
/// this cap truncates LOUDLY (`truncated: true`), never silently.
pub(crate) const MAX_NATIVE_RESERVATIONS_SCAN: i64 = 10_000;

/// Cap on parsed route-decision summaries of one task.
pub(crate) const MAX_NATIVE_ROUTE_DECISIONS: usize = 200;

/// The amount one SETTLED reservation actually folded into the task's spent
/// total: the v18 canonical `settled_cost_micro`, falling back for pre-v18
/// rows (which never recorded which amount was folded) to the legacy
/// winner — the provider-reported amount when the frame carried one, else
/// the locally calculated amount. Hostile all-NULL rows fold zero.
pub(crate) fn reservation_settled_spent_micro(row: &faktor_store::CostReservationRow) -> u64 {
    row.settled_cost_micro
        .or(row.provider_reported_micro)
        .or(row.provider_cost_micro)
        .unwrap_or(0)
}

/// The provider-reported amount of one settled reservation (the v18
/// canonical `provider_reported_cost_micro`; its pre-v18 twin fallback).
pub(crate) fn reservation_provider_reported_micro(row: &faktor_store::CostReservationRow) -> u64 {
    row.provider_reported_cost_micro
        .or(row.provider_reported_micro)
        .unwrap_or(0)
}

/// The durable reservation aggregate of ONE task: counts and micro sums per
/// status over the task's `cost_reservation` rows plus the parsed
/// route-decision summaries of its settled rows (audit P0-63). The scan is
/// bounded ([`MAX_NATIVE_RESERVATIONS_SCAN`]); a flood past the cap
/// truncates loudly (`truncated: true`).
pub(crate) fn native_task_reservation_view(
    store: &faktor_store::Store,
    session_id: SessionId,
    task_id: faktor_core::id::TaskId,
) -> Result<serde_json::Value, faktor_core::Error> {
    let rows = store
        .cost_reservations_of(session_id, task_id, MAX_NATIVE_RESERVATIONS_SCAN)
        .map_err(store_err_to_core)?;
    let mut open_count: u64 = 0;
    let mut open_predicted: u64 = 0;
    let mut settled_count: u64 = 0;
    let mut settled_predicted: u64 = 0;
    let mut settled_spent: u64 = 0;
    let mut settled_reported: u64 = 0;
    let mut refunded_count: u64 = 0;
    let mut refunded_predicted: u64 = 0;
    let mut uncertain_count: u64 = 0;
    let mut uncertain_predicted: u64 = 0;
    let mut route_decisions: Vec<serde_json::Value> = Vec::new();
    // The store lists newest first, so the summaries naturally keep the
    // newest decisions within the cap. Group vocabulary is the schema v17+
    // state set: in-flight = `reserved`/`dispatched` (the v15 `open` was
    // renamed at v17), crash recovery closes pre-dispatch rows REFUNDED and
    // dispatched-marker rows UNCERTAIN — the v15 `abandoned` state no longer
    // exists and never occurs in a migrated store.
    for row in &rows {
        match row.status.as_str() {
            "reserved" | "dispatched" => {
                open_count += 1;
                open_predicted = open_predicted.saturating_add(row.predicted_micro);
            }
            "settled" => {
                settled_count += 1;
                settled_predicted = settled_predicted.saturating_add(row.predicted_micro);
                settled_spent = settled_spent.saturating_add(reservation_settled_spent_micro(row));
                settled_reported =
                    settled_reported.saturating_add(reservation_provider_reported_micro(row));
                if route_decisions.len() < MAX_NATIVE_ROUTE_DECISIONS {
                    let route = row
                        .route_decision_json
                        .as_deref()
                        .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
                        .unwrap_or_else(|| {
                            serde_json::json!({ "raw": row.route_decision_json.as_deref().unwrap_or("") })
                        });
                    route_decisions.push(serde_json::json!({
                        "reservationId": row.reservation_id,
                        "predictedMicro": row.predicted_micro,
                        "spentMicro": reservation_settled_spent_micro(row),
                        "providerReportedMicro": reservation_provider_reported_micro(row),
                        "decision": route,
                    }));
                }
            }
            "refunded" => {
                refunded_count += 1;
                refunded_predicted = refunded_predicted.saturating_add(row.predicted_micro);
            }
            "uncertain" => {
                uncertain_count += 1;
                uncertain_predicted = uncertain_predicted.saturating_add(row.predicted_micro);
            }
            // Unknown status strings (hostile rows) contribute nothing.
            _ => {}
        }
    }
    let truncated = rows.len() as i64 >= MAX_NATIVE_RESERVATIONS_SCAN;
    let mut v = serde_json::json!({
        "open": { "count": open_count, "predictedMicro": open_predicted },
        "settled": {
            "count": settled_count,
            "predictedMicro": settled_predicted,
            "spentMicro": settled_spent,
            "providerReportedMicro": settled_reported,
        },
        "refunded": { "count": refunded_count, "predictedMicro": refunded_predicted },
        "uncertain": { "count": uncertain_count, "predictedMicro": uncertain_predicted },
        "routeDecisions": route_decisions,
    });
    if truncated {
        v["truncated"] = serde_json::json!(true);
    }
    Ok(v)
}

/// The authoritative durable usage view of ONE session (audit P0-63):
/// provider-call token aggregates over the persisted columns (the
/// settlement folds cache reads/writes and reasoning into the recorded
/// input/output totals BEFORE persistence, so the rows — and therefore this
/// view — carry exactly those totals), the per-row prefix observations,
/// and per typed task: budget envelope + reservation aggregates + route
/// decision summaries. Store errors are loud; hostile rows can only
/// truncate, never fabricate.
pub(crate) fn native_session_usage_view(
    state: &AppState,
    handle: &faktor_session::SessionHandle,
) -> Result<serde_json::Value, faktor_core::Error> {
    let store = state.deps.session.store();
    let sid = handle.id();
    let tokens = store.session_usage_tokens(sid).map_err(store_err_to_core)?;
    let prefix_rows = store
        .provider_call_prefix_rows(sid)
        .map_err(store_err_to_core)?;
    let prefix_view: Vec<serde_json::Value> = prefix_rows
        .iter()
        .take(MAX_NATIVE_LIST)
        .map(|r| {
            serde_json::json!({
                "rowId": r.row_id,
                "promptTokens": r.prompt_tokens,
                "stability": r.prefix_stability,
            })
        })
        .collect();
    let stability = handle.stored_prefix_stability()?.map(|a| {
        serde_json::json!({
            "observations": a.observations,
            "mean": a.mean,
            "stdDev": a.std_dev,
        })
    });
    let mut tasks: Vec<serde_json::Value> = Vec::new();
    let typed = handle.list_tasks()?;
    for task in typed.iter().take(MAX_NATIVE_LIST) {
        let cost = store
            .cost_task_row(sid, task.task_id)
            .map_err(store_err_to_core)?;
        let reservations = native_task_reservation_view(&store, sid, task.task_id)?;
        let open_micro = reservations["open"]["predictedMicro"].as_u64().unwrap_or(0);
        tasks.push(serde_json::json!({
            "taskId": task.task_id.to_string(),
            "budget": {
                "maxTokens": task.budget.max_tokens,
                "maxTurns": task.budget.max_turns,
                "spentTokens": task.budget.spent_tokens,
                "spentTurns": task.budget.spent_turns,
                "maxCostMicro": cost.as_ref().and_then(|c| c.max_cost_micro),
                "spentCostMicro": cost.as_ref().map(|c| c.spent_cost_micro).unwrap_or(0),
                "openReservedMicro": open_micro,
            },
            "reservations": reservations,
        }));
    }
    Ok(serde_json::json!({
        "sessionId": sid.to_string(),
        "providerCalls": {
            "tokens": tokens,
            "prefixObservations": prefix_view,
        },
        "prefixStability": stability,
        "tasks": tasks,
    }))
}

/// `GET /native/session/{id}/usage` — the authoritative usage of ONE
/// session over its durable rows (audit P0-63; see
/// [`native_session_usage_view`] for the exact row sources). Hostile ids
/// 400, unknown sessions 404.
pub(crate) async fn native_session_usage(
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
    match native_session_usage_view(&state, &handle) {
        Ok(v) => Json(v).into_response(),
        Err(e) => api_err(&e),
    }
}
