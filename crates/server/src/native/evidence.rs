//! Native evidence reads (audits 81-83/94): typed, bounded, scope-checked
//! access to the daemon's evidence store.
//!
//! Two endpoints:
//!
//! - `GET /native/evidence/{id}` — envelope metadata only (never the
//!   compact body): what the evidence is, its session/workspace scope, its
//!   retrieval policy and whether backing bytes are retained.
//! - `POST /native/evidence/{id}/retrieve` — one typed
//!   [`RetrievalSelector`] body (strict DTO, deny-unknown-fields); the
//!   selector is bounded before the store is touched.
//!
//! The access context is derived from the AUTHENTICATED session named on
//! the query (`session=<id>`): the session must exist, and its durable
//! workspace id plus session id form the [`EvidenceAccessContext`]. A
//! caller that presents a known evidence id of another session gets a typed
//! 403 (`evidence_access_denied`); an unknown id is 404. Knowing an id is
//! never authorization.

use axum::extract::rejection::JsonRejection;

use faktor_evidence::retrieve::{self, RetrievalSelector, MAX_ITEMS, MAX_QUERY_BYTES};
use faktor_evidence::store::{EvidenceAccessContext, EvidenceStore, StoredEvidence};
use faktor_evidence::types::{EvidenceError, EvidenceId};

use super::*;

/// Query shared by both endpoints: the authenticated session whose scope is
/// presented. Strict native DTO.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeEvidenceQuery {
    session: String,
}

/// Strict native selector body. Mirrors [`RetrievalSelector`] one-for-one;
/// there is no default selector, so a body that does not say what to read is
/// a 400.
#[derive(Debug, serde::Deserialize)]
#[serde(tag = "selector", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum NativeRetrievalSelector {
    All,
    ByteRange { start: u64, end: u64 },
    LineRange { start: u64, end: u64 },
    Search { query: String, max_hits: u32 },
    Items { ids: Vec<u64> },
}

fn evidence_unavailable() -> Response {
    wire_status(ApiError {
        code: "evidence_unavailable",
        message: "no evidence store is wired into this daemon".to_string(),
        http_status: 503,
        retryable: true,
    })
}

/// Map the evidence layer's typed errors: access denial is a 403, an
/// oversized selector a 400, malformed input a 400, a refused invariant a
/// 409 — never a silent downgrade.
fn evidence_error_response(e: EvidenceError) -> Response {
    let (status, code) = match &e {
        EvidenceError::AccessDenied(_) => (StatusCode::FORBIDDEN, "evidence_access_denied"),
        EvidenceError::Oversized { .. } => (StatusCode::BAD_REQUEST, "evidence_oversized"),
        EvidenceError::Malformed(_) => (StatusCode::BAD_REQUEST, "evidence_malformed"),
        EvidenceError::Refused(_) => (StatusCode::CONFLICT, "evidence_refused"),
    };
    wire_status(ApiError {
        code,
        message: e.to_string(),
        http_status: status.as_u16(),
        retryable: false,
    })
}

/// Resolve the request's authenticated session and derive its evidence
/// access context. Unknown/malformed sessions are the shared native 400/404.
fn evidence_context(
    state: &AppState,
    query: &NativeEvidenceQuery,
) -> Result<EvidenceAccessContext, Box<Response>> {
    let handle = native_resolve_session(state, &query.session)?;
    let row = match handle.row() {
        Ok(row) => row,
        Err(e) => return Err(Box::new(api_err(&e))),
    };
    Ok(EvidenceAccessContext::new(
        handle.id().raw(),
        row.workspace_id.raw(),
        None,
    ))
}

/// Run `f` with the store read lock held. The store is optional: hosts that
/// never wired one answer an honest 503, never a fabricated envelope. The
/// lock is process-wide and the closure is synchronous, so no lock guard is
/// ever held across an await.
fn with_evidence<T>(
    state: &AppState,
    f: impl FnOnce(&dyn EvidenceStore) -> Result<T, EvidenceError>,
) -> Result<T, Box<Response>> {
    let Some(store) = state.deps.evidence.as_ref() else {
        return Err(Box::new(evidence_unavailable()));
    };
    let guard = store
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    f(&**guard).map_err(|e| Box::new(evidence_error_response(e)))
}

/// The store never removes rows, so a separate existence probe cannot race
/// a scope-checked read; it only decides 404 vs 403.
fn evidence_exists(state: &AppState, id: EvidenceId) -> Result<bool, Box<Response>> {
    with_evidence(state, |store| Ok(store.get(id).is_some()))
}

/// Scope-checked metadata view: envelope identity/scope/retrieval policy
/// plus backing retention. The compact body itself is never auto-dumped.
fn evidence_view(stored: &StoredEvidence) -> serde_json::Value {
    let env = &stored.envelope;
    serde_json::json!({
        "id": env.id.0,
        "kind": serde_json::to_value(env.kind).unwrap_or(serde_json::Value::Null),
        "sessionId": env.session_id.raw(),
        "workspaceId": env.workspace_id.raw(),
        "taskId": env.task_id,
        "sourceRevision": env.source_revision,
        "compressibility": serde_json::to_value(env.compressibility).unwrap_or(serde_json::Value::Null),
        "backingCompleteness": serde_json::to_value(env.backing_completeness).unwrap_or(serde_json::Value::Null),
        "backingRetained": stored.backing.is_some(),
        "backingLen": stored.backing.as_ref().map(|b| b.len()),
        "allowRanges": env.retrieval.allow_ranges,
        "allowSearch": env.retrieval.allow_search,
        "maxBytes": env.retrieval.max_bytes,
    })
}

/// `GET /native/evidence/{id}?session=<id>` — envelope metadata of one
/// evidence id, scope-checked against the authenticated session.
pub(crate) async fn native_evidence_get(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Query(query): Query<NativeEvidenceQuery>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let evidence_id = match id.parse::<EvidenceId>() {
        Ok(id) => id,
        Err(e) => return evidence_error_response(e),
    };
    let ctx = match evidence_context(&state, &query) {
        Ok(ctx) => ctx,
        Err(r) => return *r,
    };
    match evidence_exists(&state, evidence_id) {
        Ok(false) => return wire_status(not_found(&format!("evidence {evidence_id}"))),
        Ok(true) => {}
        Err(resp) => return *resp,
    }
    match with_evidence(&state, |store| {
        store
            .get_scoped(evidence_id, &ctx)
            .map(|stored| evidence_view(&stored))
    }) {
        Ok(view) => Json(view).into_response(),
        Err(resp) => *resp,
    }
}

/// `POST /native/evidence/{id}/retrieve?session=<id>` — one typed selector
/// over the scope-checked evidence bytes.
pub(crate) async fn native_evidence_retrieve(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Query(query): Query<NativeEvidenceQuery>,
    body: Result<Json<NativeRetrievalSelector>, JsonRejection>,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let Json(selector) = match body {
        Ok(b) => b,
        Err(_) => return wire_status(malformed_body("invalid native retrieval selector")),
    };
    let selector = match native_selector_to_retrieval(selector) {
        Ok(selector) => selector,
        Err(e) => return evidence_error_response(e),
    };
    let evidence_id = match id.parse::<EvidenceId>() {
        Ok(id) => id,
        Err(e) => return evidence_error_response(e),
    };
    let ctx = match evidence_context(&state, &query) {
        Ok(ctx) => ctx,
        Err(r) => return *r,
    };
    match evidence_exists(&state, evidence_id) {
        Ok(false) => return wire_status(not_found(&format!("evidence {evidence_id}"))),
        Ok(true) => {}
        Err(resp) => return *resp,
    }
    match with_evidence(&state, |store| {
        let retrieved = retrieve::retrieve(store, &ctx, evidence_id, selector)?;
        Ok(serde_json::json!({
            "id": retrieved.id.0,
            "selector": retrieved.selector_echo,
            "bytesBase64": base64::Engine::encode(
                &base64::engine::general_purpose::STANDARD,
                &retrieved.bytes,
            ),
            "byteLen": retrieved.bytes.len(),
            "truncatedByPolicy": retrieved.truncated_by_policy,
        }))
    }) {
        Ok(view) => Json(view).into_response(),
        Err(resp) => *resp,
    }
}

/// Bounded conversion from the strict wire DTO to the typed selector. The
/// bounds (query bytes, item count) are enforced BEFORE the store is
/// touched, so an oversized selector is a 400 and never a partial read.
fn native_selector_to_retrieval(
    selector: NativeRetrievalSelector,
) -> Result<RetrievalSelector, EvidenceError> {
    Ok(match selector {
        NativeRetrievalSelector::All => RetrievalSelector::All,
        NativeRetrievalSelector::ByteRange { start, end } => {
            RetrievalSelector::ByteRange { start, end }
        }
        NativeRetrievalSelector::LineRange { start, end } => {
            RetrievalSelector::LineRange { start, end }
        }
        NativeRetrievalSelector::Search { query, max_hits } => {
            if query.len() > MAX_QUERY_BYTES {
                return Err(EvidenceError::Oversized {
                    max: MAX_QUERY_BYTES,
                    actual: query.len(),
                });
            }
            RetrievalSelector::Search { query, max_hits }
        }
        NativeRetrievalSelector::Items { ids } => {
            if ids.len() > MAX_ITEMS {
                return Err(EvidenceError::Oversized {
                    max: MAX_ITEMS,
                    actual: ids.len(),
                });
            }
            RetrievalSelector::Items {
                ids: ids.into_iter().map(EvidenceId).collect(),
            }
        }
    })
}
