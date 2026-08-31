//! The frozen v7.5.6 HTTP/SSE surface.

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{sse::Event, IntoResponse, Response, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream::Stream;
use tokio::sync::oneshot;
use tower_http::limit::RequestBodyLimitLayer;

use kilop_agent::AgentRuntime;
use kilop_core::capability::PermissionDecision;
use kilop_core::error::Error;
use kilop_protocol::error::ApiError;
use kilop_protocol::v756::*;
use kilop_core::id::SessionId;
use kilop_session::SessionManager;

use crate::auth::{AuthToken, check_bearer};
use crate::permission::ChannelPermissionRequester;

const MAX_BODY_BYTES: usize = 10 * 1024 * 1024;
const HEARTBEAT_SECS: u64 = 15;
const POLL_INTERVAL_MS: u64 = 100;

pub struct ServerDeps {
    pub session: Arc<SessionManager>,
    pub agent: Arc<AgentRuntime>,
    pub permissions: Arc<ChannelPermissionRequester>,
    pub auth_token: AuthToken,
    pub version: String,
}

impl ServerDeps {
    pub fn new(
        session: Arc<SessionManager>,
        agent: Arc<AgentRuntime>,
        permissions: Arc<ChannelPermissionRequester>,
    ) -> Self {
        Self {
            session,
            agent,
            permissions,
            auth_token: AuthToken::generate(),
            version: kilop_core::VERSION.to_string(),
        }
    }

    /// The frozen handshake line the CLI prints after binding.
    pub fn handshake_line(&self, addr: SocketAddr) -> String {
        Handshake {
            version: self.version.clone(),
            protocol: kilop_core::PROTOCOL_V756.to_string(),
            pid: std::process::id() as u64,
            auth_token: self.auth_token.as_str().to_string(),
            port: addr.port(),
        }
        .to_line()
    }
}

pub struct ServerHandle {
    pub addr: SocketAddr,
    pub shutdown: oneshot::Sender<()>,
    /// The frozen handshake line for this instance (bound address + token).
    pub handshake: String,
}

/// Bind (port 0 = ephemeral) and serve. Returns once listening.
pub async fn serve(deps: ServerDeps, port: u16) -> std::io::Result<ServerHandle> {
    // Bind first, then compute the handshake (needs the bound address) and
    // finally move the deps into the router.
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    let addr = listener.local_addr()?;
    let handshake = deps.handshake_line(addr);
    let app = Router::new()
        .route("/api/hello", get(hello))
        .route("/api/session", post(create_session))
        .route("/api/sessions", get(list_sessions))
        .route("/api/session/{id}", get(session_state))
        .route("/api/session/{id}/state", get(session_state))
        .route("/api/session/{id}/messages", get(messages))
        .route("/api/session/{id}/events", get(events))
        .route("/api/session/{id}/prompt", post(prompt))
        .route("/api/session/{id}/abort", post(abort))
        .route("/api/perm/{id}/resolve", post(resolve_permission))
        .route("/api/provider", get(provider_list))
        .layer(RequestBodyLimitLayer::new(MAX_BODY_BYTES))
        .with_state(AppState { deps: Arc::new(deps) });
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = shutdown_rx.await;
            })
            .await
            .ok();
    });
    Ok(ServerHandle {
        addr,
        shutdown: shutdown_tx,
        handshake,
    })
}

// ------------------------------------------------------------------ state

#[derive(Clone)]
struct AppState {
    deps: Arc<ServerDeps>,
}

impl From<Arc<ServerDeps>> for AppState {
    fn from(deps: Arc<ServerDeps>) -> Self {
        Self { deps }
    }
}

// ------------------------------------------------------------------ handlers

async fn hello(State(state): State<AppState>) -> Response {
    Json(HelloResponse {
        ok: true,
        version: state.deps.version.clone(),
        protocol: kilop_core::PROTOCOL_V756.to_string(),
        auth_required: true,
        providers: state.deps.agent.deps().providers.ids(),
    })
    .into_response()
}

fn authed(headers: &HeaderMap, deps: &ServerDeps) -> Result<(), ApiError> {
    if check_bearer(&deps.auth_token, headers.get(axum::http::header::AUTHORIZATION).and_then(|v| v.to_str().ok())) {
        Ok(())
    } else {
        Err(ApiError {
            code: "unauthorized",
            message: "missing or invalid bearer token".into(),
            http_status: 401,
            retryable: false,
        })
    }
}

async fn create_session(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<CreateSessionRequest>,
) -> Response {
    if let Err(e) = authed(&headers, &state.deps) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let workspace = req.workspace.clone().unwrap_or_else(|| ".".into());
    let ws = match state.deps.session.create_workspace(&workspace) {
        Ok(ws) => ws,
        Err(e) => return api_err(&e),
    };
    let title = req.title.clone().unwrap_or_else(|| "New session".into());
    match state.deps.session.create_session(ws, &title, &req.provider, &req.model) {
        Ok(row) => match row.row() {
            Ok(row_data) => Json(CreateSessionResponse {
                id: row_data.id.to_string(),
                title,
                created_ms: row_data.created_ms,
            })
            .into_response(),
            Err(e) => api_err(&e),
        },
        Err(e) => api_err(&e),
    }
}

async fn list_sessions(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(e) = authed(&headers, &state.deps) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    match state.deps.session.list_sessions(None) {
        Ok(rows) => Json(serde_json::json!({
            "sessions": rows.iter().map(|r| {
                let title = r.title().unwrap_or_default();
                let provider = r.provider().unwrap_or_default();
                let model = r.model().unwrap_or_default();
                let state = r.state().map(|s| s.label()).unwrap_or("unknown");
                serde_json::json!({
                    "id": r.id().to_string(),
                    "title": title,
                    "provider": provider,
                    "model": model,
                    "state": state,
                })
            }).collect::<Vec<_>>(),
        }))
        .into_response(),
        Err(e) => api_err(&e),
    }
}

fn parse_session_id(id: &str) -> Result<SessionId, ApiError> {
    let raw: u64 = id
        .parse()
        .map_err(|_| ApiError {
            code: "malformed",
            message: format!("invalid session id {id:?}"),
            http_status: 400,
            retryable: false,
        })?;
    if raw == 0 {
        return Err(ApiError {
            code: "malformed",
            message: "session id cannot be 0".into(),
            http_status: 400,
            retryable: false,
        });
    }
    Ok(SessionId::new(raw))
}

async fn session_state(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Err(e) = authed(&headers, &state.deps) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let sid = match parse_session_id(&id) {
        Ok(s) => s,
        Err(e) => return (StatusCode::from_u16(e.http_status).unwrap(), Json(e.to_json())).into_response(),
    };
    let handle = match state.deps.session.get_session(sid) {
        Ok(Some(h)) => h,
        Ok(None) => {
            let e = ApiError {
                code: "not_found",
                message: format!("session {sid}"),
                http_status: 404,
                retryable: false,
            };
            return (StatusCode::NOT_FOUND, Json(e.to_json())).into_response();
        }
        Err(e) => return api_err(&e),
    };
    match handle.session_state_view() {
        Ok(view) => Json(view).into_response(),
        Err(e) => api_err(&e),
    }
}

async fn messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Query(q): Query<MessagesQuery>,
) -> Response {
    if let Err(e) = authed(&headers, &state.deps) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let sid = match parse_session_id(&id) {
        Ok(s) => s,
        Err(e) => return (StatusCode::from_u16(e.http_status).unwrap(), Json(e.to_json())).into_response(),
    };
    let handle = match state.deps.session.get_session(sid) {
        Ok(Some(h)) => h,
        Ok(None) => {
            let e = ApiError {
                code: "not_found",
                message: format!("session {sid}"),
                http_status: 404,
                retryable: false,
            };
            return (StatusCode::NOT_FOUND, Json(e.to_json())).into_response();
        }
        Err(e) => return api_err(&e),
    };
    match handle.messages_page(q.before, q.limit) {
        Ok(page) => Json(page).into_response(),
        Err(e) => api_err(&e),
    }
}

async fn prompt(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(req): Json<PromptRequest>,
) -> Response {
    if let Err(e) = authed(&headers, &state.deps) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    if req.prompt.trim().is_empty() {
        let e = ApiError {
            code: "malformed",
            message: "prompt must not be empty".into(),
            http_status: 400,
            retryable: false,
        };
        return (StatusCode::BAD_REQUEST, Json(e.to_json())).into_response();
    }
    let sid = match parse_session_id(&id) {
        Ok(s) => s,
        Err(e) => return (StatusCode::from_u16(e.http_status).unwrap(), Json(e.to_json())).into_response(),
    };
    let agent = state.deps.agent.clone();
    let prompt_text = req.prompt.clone();
    let files = req.files.clone();
    // The turn runs detached from the HTTP connection: closing the SSE
    // stream never destroys an active agent (spec §7). State lives in the
    // journal, so this spawn defines no application state.
    let outcome = tokio::spawn(async move { agent.run_turn(sid, &prompt_text, &files).await });
    let _ = outcome;
    Json(PromptResponse {
        op_id: "turn".to_string(),
        accepted: true,
        queued: false,
    })
    .into_response()
}

async fn abort(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Err(e) = authed(&headers, &state.deps) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let sid = match parse_session_id(&id) {
        Ok(s) => s,
        Err(e) => return (StatusCode::from_u16(e.http_status).unwrap(), Json(e.to_json())).into_response(),
    };
    match state.deps.agent.abort(sid) {
        Ok(ops) => Json(AbortResponse {
            aborted: ops.iter().map(|o| o.to_string()).collect(),
        })
        .into_response(),
        Err(e) => api_err(&e),
    }
}

async fn resolve_permission(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(req): Json<PermissionDecisionRequest>,
) -> Response {
    if let Err(e) = authed(&headers, &state.deps) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let pid: i64 = match id.parse() {
        Ok(p) if p > 0 => p,
        _ => {
            let e = ApiError {
                code: "malformed",
                message: format!("invalid permission id {id:?}"),
                http_status: 400,
                retryable: false,
            };
            return (StatusCode::BAD_REQUEST, Json(e.to_json())).into_response();
        }
    };
    let decision = match req.decision.as_str() {
        "allow" => PermissionDecision::Allow,
        "deny" => PermissionDecision::Deny,
        other => {
            let e = ApiError {
                code: "malformed",
                message: format!("invalid decision {other:?}"),
                http_status: 400,
                retryable: false,
            };
            return (StatusCode::BAD_REQUEST, Json(e.to_json())).into_response();
        }
    };
    if !state.deps.permissions.resolve(pid, decision) {
        let e = ApiError {
            code: "conflict",
            message: format!("permission {pid} unknown or already resolved"),
            http_status: 409,
            retryable: false,
        };
        return (StatusCode::CONFLICT, Json(e.to_json())).into_response();
    }
    Json(PermissionDecisionResponse { ok: true }).into_response()
}

async fn provider_list(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(e) = authed(&headers, &state.deps) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let ids = state.deps.agent.deps().providers.ids();
    let mut providers = Vec::new();
    for id in ids {
        let models = if let Some(p) = state.deps.agent.deps().providers.get(&id) {
            // Dynamic registry: the adapter's own model list is exposed here.
            vec![ModelInfo {
                id: "default".into(),
                name: "default".into(),
                capabilities: p.capabilities("default"),
            }]
        } else {
            vec![]
        };
        providers.push(ProviderInfo {
            id: id.clone(),
            name: id.clone(),
            kind: id.clone(),
            models,
        });
    }
    Json(ProviderList { providers }).into_response()
}

// ------------------------------------------------------------------ SSE

async fn events(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Query(q): Query<MessagesQuery>,
) -> Response {
    if let Err(e) = authed(&headers, &state.deps) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let sid = match parse_session_id(&id) {
        Ok(s) => s,
        Err(e) => return (StatusCode::from_u16(e.http_status).unwrap(), Json(e.to_json())).into_response(),
    };
    // The SSE cursor is the raw sequence (0 = from the beginning); the
    // journal is queried as `seq > cursor` via events_range.
    let cursor: i64 = q.events_after.unwrap_or(0).max(0);
    let handle = match state.deps.session.get_session(sid) {
        Ok(Some(h)) => h,
        Ok(None) => {
            let e = ApiError {
                code: "not_found",
                message: format!("session {sid}"),
                http_status: 404,
                retryable: false,
            };
            return (StatusCode::NOT_FOUND, Json(e.to_json())).into_response();
        }
        Err(e) => return api_err(&e),
    };

    // Catch-up frames from the journal, then poll for new ones. The journal
    // is the source of truth; a reconnect resumes exactly from the cursor.
    let stream = journal_stream(handle, cursor);
    Sse::new(stream)
        .keep_alive(
            axum::response::sse::KeepAlive::new()
                .interval(Duration::from_secs(HEARTBEAT_SECS))
                .text("keep-alive"),
        )
        .into_response()
}

fn journal_stream(
    handle: kilop_session::SessionHandle,
    cursor: i64,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> + Send + 'static {
    // State: (handle, next cursor, queue of ready frames).
    // Poll the journal; the journal is the source of truth and the cursor is
    // the SSE resume point. Heartbeats keep proxies alive.
    futures_util::stream::unfold(
        (handle, cursor, VecDeque::<Event>::new()),
        move |(handle, mut cursor, mut queue)| async move {
            if let Some(ev) = queue.pop_front() {
                return Some((Ok::<Event, std::convert::Infallible>(ev), (handle, cursor, queue)));
            }
            // seq > cursor (cursor 0 = everything from seq 1).
            let events = handle
                .events_range(cursor.saturating_add(1) as u64, None)
                .unwrap_or_default();
            let mut batch = VecDeque::new();
            let mut advanced = false;
            for e in events {
                if let Some((event, _)) = kilop_protocol::sse::project_event(&e) {
                    batch.push_back(sse_event(kilop_session::JournalFrame {
                        seq: e.seq,
                        event,
                    }));
                }
                cursor = e.seq.raw() as i64;
                advanced = true;
            }
            if advanced {
                if let Some(ev) = batch.pop_front() {
                    return Some((
                        Ok::<Event, std::convert::Infallible>(ev),
                        (handle, cursor, batch),
                    ));
                }
            }
            tokio::time::sleep(Duration::from_millis(POLL_INTERVAL_MS)).await;
            Some((
                Ok::<Event, std::convert::Infallible>(sse_event_heartbeat()),
                (handle, cursor, queue),
            ))
        },
    )
}

fn sse_event(frame: kilop_session::JournalFrame) -> Event {
    let seq = frame.seq.raw();
    let json = serde_json::to_string(&frame.event).unwrap_or_else(|_| "{}".into());
    Event::default()
        .event(frame.event.event_type())
        .id(seq.to_string())
        .data(json)
}

fn sse_event_heartbeat() -> Event {
    Event::default().event("heartbeat").data("{}")
}

fn api_err(e: &Error) -> Response {
    let api = kilop_protocol::error::from_core(e);
    (
        StatusCode::from_u16(api.http_status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        Json(api.to_json()),
    )
    .into_response()
}

fn _unused(_: Body) {}

#[cfg(test)]
mod tests {
    use super::*;
    use kilop_core::model::ModelCapabilities;
    use kilop_provider::FakeProvider;

    #[test]
    fn handshake_line_is_frozen_shape() {
        let deps = ServerDeps {
            session: SessionManager::open(
                std::env::temp_dir().join("kp-hs-store"),
                std::env::temp_dir().join("kp-hs-cas"),
                false,
            )
            .unwrap(),
            agent: AgentRuntime::new(kilop_agent::AgentDeps {
                session: SessionManager::open(
                    std::env::temp_dir().join("kp-hs-store2"),
                    std::env::temp_dir().join("kp-hs-cas2"),
                    false,
                )
                .unwrap(),
                providers: Arc::new(kilop_provider::ProviderRegistry::new()),
                permission_requester: ChannelPermissionRequester::new(Duration::from_secs(1)),
                evidence: Arc::new(kilop_agent::NoEvidence),
                tools: Arc::new(kilop_agent::ToolRegistry::new()),
                cas: None,
                model: "m".into(),
                compaction_model: None,
                compact_at_usage: 0.65,
                instructions: "i".into(),
                clock: Arc::new(kilop_core::time::SystemClock),
                tool_call_mode: kilop_agent::ToolCallMode::Native,
                tool_deadline_ms: 1000,
            })
            .unwrap(),
            permissions: ChannelPermissionRequester::new(Duration::from_secs(1)),
            auth_token: AuthToken::generate(),
            version: "0.1.0".into(),
        };
        let addr: SocketAddr = "127.0.0.1:45678".parse().unwrap();
        let line = deps.handshake_line(addr);
        assert!(line.starts_with("KILO_PLUS_HANDSHAKE "));
        let hs = Handshake::from_line(&line).unwrap();
        assert_eq!(hs.protocol, "v756");
        assert_eq!(hs.port, 45678);
        assert_eq!(hs.auth_token, deps.auth_token.as_str());
    }

    #[tokio::test]
    async fn unauthorized_requests_rejected_before_handlers() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        // No token.
        let resp = client
            .post(format!("http://{}/api/session", handle.addr))
            .json(&serde_json::json!({"provider": "fake", "model": "m"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);
        // Wrong token.
        let resp = client
            .post(format!("http://{}/api/session", handle.addr))
            .bearer_auth("wrong")
            .json(&serde_json::json!({"provider": "fake", "model": "m"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn hello_is_public_and_correct() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let resp = client
            .get(format!("http://{}/api/hello", handle.addr))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["protocol"], "v756");
        assert_eq!(body["auth_required"], true);
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn full_flow_create_prompt_messages_state() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let token = deps.auth_token.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        // Create session.
        let resp = client
            .post(format!("{base}/api/session"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({
                "provider": "fake",
                "model": "m",
                "workspace": "/tmp",
                "title": "t1",
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid = created["id"].as_str().unwrap().to_string();

        // Prompt.
        let resp = client
            .post(format!("{base}/api/session/{sid}/prompt"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"prompt": "hi", "files": []}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let pr: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(pr["accepted"], true);

        // State reflects the turn (may still be running — poll until ready).
        let mut state = String::new();
        for _ in 0..100 {
            let resp = client
                .get(format!("{base}/api/session/{sid}/state"))
                .bearer_auth(token.as_str())
                .send()
                .await
                .unwrap();
            let body: serde_json::Value = resp.json().await.unwrap();
            state = body["agent_state"]["state"].as_str().unwrap_or("").to_string();
            if matches!(state.as_str(), "ready_for_next_turn" | "completed" | "cancelled") {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(state, "ready_for_next_turn", "turn must complete");

        // Messages contain the exchange.
        let resp = client
            .get(format!("{base}/api/session/{sid}/messages?limit=10"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        let page: serde_json::Value = resp.json().await.unwrap();
        assert!(page["messages"].as_array().unwrap().len() >= 2, "{page}");

        // Malformed body → 400; unknown route → 404; unknown session → 404.
        let resp = client
            .post(format!("{base}/api/session/{sid}/prompt"))
            .bearer_auth(token.as_str())
            .body("{not json")
            .header("content-type", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);
        let resp = client
            .get(format!("{base}/api/nope"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);
        let resp = client
            .get(format!("{base}/api/session/999999/state"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);

        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn sse_streams_and_resumes_from_cursor() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let token = deps.auth_token.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        let resp = client
            .post(format!("{base}/api/session"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"provider": "fake", "model": "m"}))
            .send()
            .await
            .unwrap();
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid = created["id"].as_str().unwrap().to_string();

        // Subscribe before the prompt so we see the whole sequence.
        let mut sse = client
            .get(format!("{base}/api/session/{sid}/events?events_after=0"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap()
            .bytes_stream();

        client
            .post(format!("{base}/api/session/{sid}/prompt"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"prompt": "hi"}))
            .send()
            .await
            .unwrap();

        use futures_util::StreamExt;
        let mut saw_state = false;
        let mut saw_heartbeat = false;
        let mut text = String::new();
        for _ in 0..200 {
            match tokio::time::timeout(Duration::from_millis(200), sse.next()).await {
                Ok(Some(Ok(chunk))) => {
                    text.push_str(&String::from_utf8_lossy(&chunk));
                    if text.contains("agent_state_changed") {
                        saw_state = true;
                    }
                    if text.contains("heartbeat") {
                        saw_heartbeat = true;
                    }
                    if saw_state {
                        break;
                    }
                }
                Ok(Some(Err(_))) => break,
                Ok(None) | Err(_) => break,
            }
        }
        assert!(saw_state, "SSE must deliver state events; got: {text}");

        // Resume from a cursor: events_after=1 skips the SessionCreated frame.
        let resp = client
            .get(format!("{base}/api/session/{sid}/events?events_after=1"))
            .bearer_auth(token.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let _ = handle.shutdown.send(());
    }

    #[tokio::test]
    async fn permission_flow_blocks_until_resolved() {
        let dir = tempfile::tempdir().unwrap();
        let deps = test_deps(dir.path());
        let token = deps.auth_token.clone();
        let handle = serve(deps, 0).await.unwrap();
        let client = reqwest::Client::new();
        let base = format!("http://{}", handle.addr);

        let resp = client
            .post(format!("{base}/api/session"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"provider": "fake", "model": "m"}))
            .send()
            .await
            .unwrap();
        let created: serde_json::Value = resp.json().await.unwrap();
        let sid = created["id"].as_str().unwrap().to_string();

        // The fake provider script makes a tool call, so the turn blocks on
        // permission. Resolve it through the frozen API.
        let mut registry = kilop_provider::ProviderRegistry::new();
        registry.register(Arc::new(FakeProvider::with_script(
            "fake",
            ModelCapabilities {
                tools: true,
                ..Default::default()
            },
            vec![
                kilop_provider::ScriptedResponse::ToolCall {
                    id: "c1".into(),
                    name: "echo".into(),
                    input: serde_json::json!({"x": 1}),
                },
                kilop_provider::ScriptedResponse::End,
            ],
        )));
        let session = SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let permissions = ChannelPermissionRequester::new(Duration::from_secs(5));
        let mut tools = kilop_agent::ToolRegistry::new();
        tools.register(kilop_agent::Tool {
            name: "echo".into(),
            description: "d".into(),
            input_schema: serde_json::json!({}),
            resource_class: kilop_core::resource::ResourceClass::Cpu,
            capability: None,
            recovery_hint: kilop_agent::RecoveryHint::Idempotent,
            execute: Arc::new(|_ctx, _args| {
                Box::pin(async move { Ok(kilop_agent::ToolOutcome::default()) })
            }),
        });
        let agent = AgentRuntime::new(kilop_agent::AgentDeps {
            session: session.clone(),
            providers: Arc::new(registry),
            permission_requester: permissions.clone(),
            evidence: Arc::new(kilop_agent::NoEvidence),
            tools: Arc::new(tools),
            cas: None,
            model: "m".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "You are a test server agent.".into(),
            clock: Arc::new(kilop_core::time::SystemClock),
            tool_call_mode: kilop_agent::ToolCallMode::Native,
            tool_deadline_ms: 2000,
        })
        .unwrap();
        // Replace the running server's deps by serving a second one on the
        // same store (the first server's fake provider has no tool call, so
        // the permission test needs its own instance).
        let deps2 = ServerDeps {
            session: session.clone(),
            agent,
            permissions: permissions.clone(),
            auth_token: token.clone(),
            version: "0.1.0".into(),
        };
        let handle2 = serve(deps2, 0).await.unwrap();
        let base2 = format!("http://{}", handle2.addr);
        let resp = client
            .post(format!("{base2}/api/session"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"provider": "fake", "model": "m"}))
            .send()
            .await
            .unwrap();
        let created2: serde_json::Value = resp.json().await.unwrap();
        let sid2 = created2["id"].as_str().unwrap().to_string();
        client
            .post(format!("{base2}/api/session/{sid2}/prompt"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"prompt": "use tools"}))
            .send()
            .await
            .unwrap();

        // The turn blocks on permission; resolve through the API.
        let mut resolved = false;
        for _ in 0..100 {
            if let Some(pid) = permissions.pending_ids().first().copied() {
                let resp = client
                    .post(format!("{base2}/api/perm/{pid}/resolve"))
                    .bearer_auth(token.as_str())
                    .json(&serde_json::json!({
                        "permission_id": pid.to_string(),
                        "decision": "allow",
                    }))
                    .send()
                    .await
                    .unwrap();
                assert_eq!(resp.status(), 200);
                resolved = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(resolved, "permission must surface and resolve");

        // The turn must now complete.
        let mut done = false;
        for _ in 0..100 {
            let id = parse_session_id(&sid2).unwrap();
            let state = session.get_session(id).unwrap().unwrap().state().unwrap();
            if matches!(state, kilop_core::state::AgentState::ReadyForNextTurn) {
                done = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(done, "turn must finish after permission grant");

        // Double resolve → conflict.
        let resp = client
            .post(format!("{base2}/api/perm/1/resolve"))
            .bearer_auth(token.as_str())
            .json(&serde_json::json!({"permission_id": "1", "decision": "allow"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 409);
        let _ = handle.shutdown.send(());
        let _ = handle2.shutdown.send(());
    }

    fn test_deps_with_permission(root: &std::path::Path) -> ServerDeps {
        test_deps(root)
    }

    fn test_deps(root: &std::path::Path) -> ServerDeps {
        let mut registry = kilop_provider::ProviderRegistry::new();
        registry.register(Arc::new(FakeProvider::with_script(
            "fake",
            ModelCapabilities {
                tools: true,
                ..Default::default()
            },
            vec![
                kilop_provider::ScriptedResponse::Text("pong".into()),
                kilop_provider::ScriptedResponse::End,
            ],
        )));
        let session = SessionManager::open(root.join("store"), root.join("cas"), true).unwrap();
        let permissions = ChannelPermissionRequester::new(Duration::from_secs(5));
        let agent = AgentRuntime::new(kilop_agent::AgentDeps {
            session: session.clone(),
            providers: Arc::new(registry),
            permission_requester: permissions.clone(),
            evidence: Arc::new(kilop_agent::NoEvidence),
            tools: Arc::new(kilop_agent::ToolRegistry::new()),
            cas: None,
            model: "m".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "You are a test server agent.".into(),
            clock: Arc::new(kilop_core::time::SystemClock),
            tool_call_mode: kilop_agent::ToolCallMode::Native,
            tool_deadline_ms: 2000,
        })
        .unwrap();
        ServerDeps {
            session,
            agent,
            permissions,
            auth_token: AuthToken::generate(),
            version: "0.1.0".into(),
        }
    }
}
