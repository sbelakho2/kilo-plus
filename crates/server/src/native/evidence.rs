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
//! workspace id plus session id form the [`EvidenceAccessContext`]. The
//! DEFAULT scope is the session's own task ([`EvidenceAccessScope::Task`],
//! exactly what the runtime/model uses); an authenticated UI that needs
//! cross-task reads asks explicitly for `scope=admin`
//! ([`EvidenceAccessScope::SessionAdmin`]). No authorization decision is
//! derived from a missing/None task id. A caller that presents a known
//! evidence id outside its scope gets a typed 403
//! (`evidence_access_denied`); an unknown id is 404. Knowing an id is never
//! authorization.
//!
//! EVERY store/CAS read (existence probe, envelope decode, backing fetch)
//! runs through the session manager's bounded DB read pool
//! ([`DbReadService::submit_storage`]) — file/SQLite work never runs inline
//! on a Tokio worker and there is no unbounded `spawn_blocking`.

use axum::extract::rejection::JsonRejection;

use faktor_evidence::retrieve::{self, RetrievalSelector, MAX_ITEMS, MAX_QUERY_BYTES};
use faktor_evidence::store::{
    EvidenceAccessContext, EvidenceAccessScope, EvidenceStore, StoredEvidence,
};
use faktor_evidence::types::{EvidenceError, EvidenceId};

use super::*;

/// Query shared by both endpoints: the authenticated session whose scope is
/// presented, plus the optional EXPLICIT scope (`admin` = the authenticated
/// session-administrator reader; absent = the session's own task). Strict
/// native DTO.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeEvidenceQuery {
    session: String,
    #[serde(default)]
    scope: Option<String>,
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

/// Resolve the request's authenticated session and derive its DECLARED
/// evidence access scope. Unknown/malformed sessions are the shared native
/// 400/404; an unknown scope value is a loud 400 (never a silent admin
/// downgrade).
fn evidence_context(
    state: &AppState,
    query: &NativeEvidenceQuery,
) -> Result<EvidenceAccessContext, Box<Response>> {
    let handle = native_resolve_session(state, &query.session)?;
    let row = match handle.row() {
        Ok(row) => row,
        Err(e) => return Err(Box::new(api_err(&e))),
    };
    let scope = match query.scope.as_deref() {
        // Default: exactly the task the session is acting for, the same
        // scope the runtime/model uses.
        None => match handle.task_id() {
            Ok(task_id) => EvidenceAccessScope::Task(task_id),
            Err(e) => return Err(Box::new(api_err(&e))),
        },
        // The authenticated session administrator (cross-task reads).
        Some("admin") => EvidenceAccessScope::SessionAdmin,
        Some(other) => {
            return Err(Box::new(wire_status(malformed_body(&format!(
                "unknown evidence scope {other:?}; expected `admin`"
            )))))
        }
    };
    Ok(EvidenceAccessContext::scoped(
        handle.id().raw(),
        row.workspace_id.raw(),
        scope,
    ))
}

/// Run one evidence read through the session manager's BOUNDED read pool.
/// The store is optional: hosts that never wired one answer an honest 503,
/// never a fabricated envelope. The closure executes on a dedicated read
/// worker (never inline on the Tokio worker, never an unbounded
/// `spawn_blocking`), receives the locked store, and returns `Ok(None)` when
/// the evidence does not exist (404 at the caller).
async fn with_evidence<T: Send + 'static>(
    state: &AppState,
    f: impl FnOnce(&(dyn EvidenceStore + Send + Sync)) -> Result<Option<T>, EvidenceError>
        + Send
        + 'static,
) -> Result<Option<T>, Box<Response>> {
    let Some(handle) = state.deps.evidence.clone() else {
        return Err(Box::new(evidence_unavailable()));
    };
    let outcome = state
        .deps
        .session
        .read_service()
        .submit_storage(move |_store| {
            let guard = handle
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            f(&**guard)
        })
        .await;
    match outcome {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(e)) => Err(Box::new(evidence_error_response(e))),
        Err(e) => Err(Box::new(api_err(&e))),
    }
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

/// `GET /native/evidence/{id}?session=<id>[&scope=admin]` — envelope
/// metadata of one evidence id, scope-checked against the authenticated
/// session.
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
    match with_evidence(&state, move |store| {
        if !store.exists(evidence_id)? {
            return Ok(None);
        }
        store
            .get_scoped(evidence_id, &ctx)
            .map(|stored| Some(evidence_view(&stored)))
    })
    .await
    {
        Ok(Some(view)) => Json(view).into_response(),
        Ok(None) => wire_status(not_found(&format!("evidence {evidence_id}"))),
        Err(resp) => *resp,
    }
}

/// `POST /native/evidence/{id}/retrieve?session=<id>[&scope=admin]` — one
/// typed selector over the scope-checked evidence bytes.
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
    match with_evidence(&state, move |store| {
        if !store.exists(evidence_id)? {
            return Ok(None);
        }
        retrieve::retrieve(store, &ctx, evidence_id, selector).map(Some)
    })
    .await
    {
        Ok(Some(retrieved)) => Json(serde_json::json!({
            "id": retrieved.id.0,
            "selector": retrieved.selector_echo,
            "bytesBase64": base64::Engine::encode(
                &base64::engine::general_purpose::STANDARD,
                &retrieved.bytes,
            ),
            "byteLen": retrieved.bytes.len(),
            "truncatedByPolicy": retrieved.truncated_by_policy,
        }))
        .into_response(),
        Ok(None) => wire_status(not_found(&format!("evidence {evidence_id}"))),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::EvidenceStoreHandle;
    use crate::permission::ChannelPermissionRequester;
    use faktor_core::id::{SessionId, WorkspaceId};
    use faktor_evidence::store::MemoryEvidenceStore;
    use faktor_evidence::types::{
        BackingCompleteness, CompactRepresentation, Compressibility, CompressionRecord,
        EvidenceEnvelope, EvidenceKind, ProvenanceSet, ProvenanceSource, RetrievalPolicy,
    };
    use faktor_session::{DbReadKind, SessionManager};
    use std::time::Duration;

    fn evidence_envelope(
        id: u64,
        session: u64,
        workspace: u64,
        task: Option<u64>,
    ) -> EvidenceEnvelope {
        EvidenceEnvelope::new(
            EvidenceId(id),
            EvidenceKind::ProcessLog,
            SessionId::new(session),
            WorkspaceId::new(workspace),
            task,
            None,
            ProvenanceSet::new([ProvenanceSource::Tool]),
            Compressibility::Reversible,
            CompactRepresentation {
                grammar: "log-v1".into(),
                body: "hello".into(),
            },
            Some([3u8; 32]),
            BackingCompleteness::Complete,
            CompressionRecord::identity(5),
            RetrievalPolicy::new(true, true, 1024),
        )
        .unwrap()
    }

    fn evidence_handle(envelopes: Vec<EvidenceEnvelope>) -> EvidenceStoreHandle {
        let mut store = MemoryEvidenceStore::new(4096);
        for env in envelopes {
            store.insert(env, Some(b"hello".to_vec())).unwrap();
        }
        std::sync::Arc::new(std::sync::RwLock::new(
            Box::new(store) as Box<dyn faktor_evidence::store::EvidenceStore + Send + Sync>
        ))
    }

    fn test_state(root: &std::path::Path) -> AppState {
        let session = SessionManager::open(root.join("store"), root.join("cas"), true).unwrap();
        let permissions = ChannelPermissionRequester::new(Duration::from_secs(5));
        let agent = faktor_agent::AgentRuntime::new(faktor_agent::AgentDeps {
            session: session.clone(),
            providers: std::sync::Arc::new(faktor_provider::ProviderRegistry::new()),
            chunk_sink: None,
            permission_requester: permissions.clone(),
            evidence: std::sync::Arc::new(faktor_agent::NoEvidence),
            tools: std::sync::Arc::new(faktor_agent::ToolRegistry::new()),
            cas: None,
            workspaces: faktor_fs::WorkspaceFileService::new(),
            edit: None,
            snapshots: None,
            sandbox: None,
            supervisor: None,
            verification: faktor_agent::VerificationService::disabled(),
            model: "m".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "i".into(),
            hooks: None,
            instructions_resolver: faktor_instructions::no_roots_resolver(),
            routing: faktor_agent::FixedRoutingPolicy::passthrough(),
            budgets: std::sync::Arc::new(faktor_session::NoopBudget),
            clock: std::sync::Arc::new(faktor_core::time::SystemClock),
            tool_call_mode: faktor_agent::ToolCallMode::Native,
            tool_deadline_ms: 1000,
            retry_policy: faktor_core::retry::RetryPolicy::default(),
            semantic: faktor_agent::fallback_semantic_registry(),
            context_prior: None,
            efficiency: Default::default(),
        })
        .unwrap();
        let deps = ServerDeps::new(session, agent, permissions);
        AppState {
            bus: std::sync::Arc::new(crate::global::GlobalEventBus::new(
                deps.session.clone(),
                None,
            )),
            deps: std::sync::Arc::new(deps),
            config: std::sync::Arc::new(std::sync::RwLock::new(serde_json::Value::Object(
                Default::default(),
            ))),
            auth: std::sync::Arc::new(std::sync::RwLock::new(None)),
            ptys: std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            next_pty_id: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(1)),
            terminal_owners: std::sync::Arc::new(std::sync::Mutex::new(
                std::collections::HashMap::new(),
            )),
            terminal_events: std::sync::Arc::new(std::sync::Mutex::new(
                std::collections::VecDeque::new(),
            )),
            next_terminal_event_id: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(1)),
            ready: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    fn auth_headers(state: &AppState) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-faktor-server-password",
            state
                .deps
                .server_password
                .as_str()
                .parse::<axum::http::HeaderValue>()
                .unwrap(),
        );
        headers
    }

    /// A native retrieve is served by the BOUNDED read pool (storage tag),
    /// never inline on the Tokio worker.
    #[tokio::test]
    async fn native_retrieve_runs_on_the_bounded_read_service_and_is_tagged() {
        let dir = tempfile::tempdir().unwrap();
        let mut state = test_state(dir.path());
        let ws = state.deps.session.create_workspace("/tmp").unwrap();
        let session = state
            .deps
            .session
            .create_session(ws, "t", "fake", "m")
            .unwrap();
        let row = session.row().unwrap();
        let handle = evidence_handle(vec![evidence_envelope(7, row.id.raw(), ws.raw(), None)]);
        std::sync::Arc::get_mut(&mut state.deps).unwrap().evidence = Some(handle);

        let before = state.deps.session.read_service().stats();
        let response = native_evidence_retrieve(
            State(state.clone()),
            auth_headers(&state),
            Path("7".to_string()),
            Query(NativeEvidenceQuery {
                session: row.id.raw().to_string(),
                scope: None,
            }),
            Ok(Json(NativeRetrievalSelector::All)),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["bytesBase64"], "aGVsbG8=");
        assert_eq!(body["byteLen"], 5);

        let after = state.deps.session.read_service().stats();
        assert_eq!(
            after.kind(DbReadKind::Storage),
            before.kind(DbReadKind::Storage) + 1,
            "the native retrieve must be a storage read on the bounded pool"
        );
        assert_eq!(after.enqueued, before.enqueued + 1);
        assert_eq!(after.completed, before.completed + 1);
    }

    /// Task scope and admin scope are distinct: the session's own task sees
    /// its own + task-less evidence; `scope=admin` is the explicit
    /// cross-task reader; a bogus scope is a loud 400.
    #[tokio::test]
    async fn native_task_scope_defaults_and_admin_scope_is_explicit() {
        let dir = tempfile::tempdir().unwrap();
        let mut state = test_state(dir.path());
        let ws = state.deps.session.create_workspace("/tmp").unwrap();
        let session = state
            .deps
            .session
            .create_session(ws, "t", "fake", "m")
            .unwrap();
        let row = session.row().unwrap();
        let own_task = session.task_id().unwrap();
        assert_eq!(own_task.raw(), 1);
        let handle = evidence_handle(vec![
            evidence_envelope(7, row.id.raw(), ws.raw(), None),
            evidence_envelope(8, row.id.raw(), ws.raw(), Some(2)),
        ]);
        std::sync::Arc::get_mut(&mut state.deps).unwrap().evidence = Some(handle);
        let headers = auth_headers(&state);

        // Task-less evidence is visible to the session's own task scope.
        let response = native_evidence_get(
            State(state.clone()),
            headers.clone(),
            Path("7".to_string()),
            Query(NativeEvidenceQuery {
                session: row.id.raw().to_string(),
                scope: None,
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);

        // Task 2's evidence is NOT visible to the default task scope...
        let response = native_evidence_get(
            State(state.clone()),
            headers.clone(),
            Path("8".to_string()),
            Query(NativeEvidenceQuery {
                session: row.id.raw().to_string(),
                scope: None,
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::FORBIDDEN);

        // ...but the explicit admin scope reads it.
        let response = native_evidence_get(
            State(state.clone()),
            headers.clone(),
            Path("8".to_string()),
            Query(NativeEvidenceQuery {
                session: row.id.raw().to_string(),
                scope: Some("admin".to_string()),
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);

        // A fabricated scope never downgrades to admin.
        let response = native_evidence_get(
            State(state.clone()),
            headers,
            Path("8".to_string()),
            Query(NativeEvidenceQuery {
                session: row.id.raw().to_string(),
                scope: Some("superuser".to_string()),
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    /// Unknown ids stay 404 and missing stores stay the honest 503; the
    /// read-pool path never turns either into a store panic.
    #[tokio::test]
    async fn native_unknown_id_and_unwired_store_are_honest() {
        let dir = tempfile::tempdir().unwrap();
        let mut state = test_state(dir.path());
        let ws = state.deps.session.create_workspace("/tmp").unwrap();
        let session = state
            .deps
            .session
            .create_session(ws, "t", "fake", "m")
            .unwrap();
        let row = session.row().unwrap();

        // No store wired: 503, never a fabricated envelope.
        let response = native_evidence_get(
            State(state.clone()),
            auth_headers(&state),
            Path("7".to_string()),
            Query(NativeEvidenceQuery {
                session: row.id.raw().to_string(),
                scope: None,
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

        std::sync::Arc::get_mut(&mut state.deps).unwrap().evidence =
            Some(evidence_handle(vec![evidence_envelope(
                7,
                row.id.raw(),
                ws.raw(),
                None,
            )]));
        let response = native_evidence_get(
            State(state.clone()),
            auth_headers(&state),
            Path("999".to_string()),
            Query(NativeEvidenceQuery {
                session: row.id.raw().to_string(),
                scope: None,
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}
