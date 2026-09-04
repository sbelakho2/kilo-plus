//! End-to-end integration tests: the daemon as a whole, adversarially.
//! Server + agent + session + provider + persistence working together:
//! crash-restart recovery, SSE resume, permission flow, compaction,
//! paging, hostile payloads, and the full deterministic-provider
//! conversation flow over the real HTTP + SSE surface.

#![cfg_attr(
    not(test),
    allow(dead_code, unused_imports, unused_variables, unused_mut)
)] // test-harness crate: the lib view exists only for clippy
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use faktor_agent::{
    AgentDeps, AgentRuntime, NoEvidence, PermissionRequester, Tool, ToolOutcome, ToolRegistry,
};
use faktor_core::capability::PermissionDecision;
use faktor_core::id::SessionId;
use faktor_core::model::ModelCapabilities;
use faktor_core::state::AgentState;
use faktor_core::time::SystemClock;
use faktor_protocol::sse::SseEvent;
use faktor_provider::{
    FakeProvider, GenericAgentRequest, Provider, ProviderChunk, ProviderRegistry, ProviderStream,
    ScriptedResponse,
};
use faktor_server::permission::ChannelPermissionRequester;
use faktor_server::{serve, ServerDeps};
use faktor_session::SessionManager;
use tempfile::tempdir;
use tokio::sync::watch;

fn test_agent(
    session: Arc<SessionManager>,
    script: Vec<ScriptedResponse>,
    permissions: Arc<dyn PermissionRequester>,
) -> Arc<AgentRuntime> {
    let mut registry = ProviderRegistry::new();
    registry.register(Arc::new(FakeProvider::with_script(
        "fake",
        ModelCapabilities {
            tools: true,
            ..Default::default()
        },
        script,
    )));
    agent_with_registry(session, registry, permissions)
}

/// An agent over an already-populated provider registry. The registry's
/// instance id decides the session provider name that resolves (the wire
/// tests create sessions with provider "fake").
fn agent_with_registry(
    session: Arc<SessionManager>,
    registry: ProviderRegistry,
    permissions: Arc<dyn PermissionRequester>,
) -> Arc<AgentRuntime> {
    agent_with_registry_and_sink(session, registry, permissions, None).0
}

/// Sink variant: returns (agent, chunk receiver) for live-frame tests.
fn agent_with_registry_and_sink(
    session: Arc<SessionManager>,
    registry: ProviderRegistry,
    permissions: Arc<dyn PermissionRequester>,
    sink: Option<Arc<faktor_agent::ChunkSink>>,
) -> (
    Arc<AgentRuntime>,
    Option<tokio::sync::mpsc::Receiver<faktor_agent::ChunkEvent>>,
) {
    let mut tools = ToolRegistry::new();
    tools.register(Tool {
        name: "echo".into(),
        description: "d".into(),
        input_schema: serde_json::json!({}),
        resource_class: faktor_core::resource::ResourceClass::Cpu,
        capability: None,
        recovery_hint: faktor_agent::RecoveryHint::Idempotent,
        path_args: vec![],
        execute: Arc::new(|_ctx, args| {
            Box::pin(async move {
                Ok(ToolOutcome {
                    text: format!("echo: {args}"),
                    exit_code: Some(0),
                    ..Default::default()
                })
            })
        }),
    });
    let (sink_field, rx) = match sink {
        Some(tx) => (Some(tx), None),
        None => (None, None),
    };
    let agent = AgentRuntime::new(AgentDeps {
        session,
        providers: Arc::new(registry),
        chunk_sink: sink_field,
        permission_requester: permissions,
        evidence: Arc::new(NoEvidence),
        tools: Arc::new(tools),
        cas: None,
        workspaces: faktor_fs::WorkspaceFileService::new(),
        edit: None,
        snapshots: None,
        sandbox: None,
        supervisor: None,
        verifier: None,
        hooks: None,
        instructions_loader: None,
        model: "m".into(),
        compaction_model: None,
        compact_at_usage: 0.65,
        instructions: "You are a test agent.".into(),
        clock: Arc::new(SystemClock),
        tool_call_mode: faktor_agent::ToolCallMode::Native,
        tool_deadline_ms: 2000,
        retry_policy: faktor_core::retry::RetryPolicy::default(),
    })
    .unwrap();
    (agent, rx)
}

/// Daemon restart: reopen the same store, run recovery, verify no state
/// loss and no double tool execution.
#[tokio::test]
async fn daemon_restart_recovers_without_state_loss() {
    let dir = tempdir().unwrap();
    let root = dir.path();
    #[derive(Clone)]
    struct AlwaysAllow;
    impl PermissionRequester for AlwaysAllow {
        fn request(
            &self,
            _s: SessionId,
            _p: &faktor_session::ops::PermissionRequest,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = faktor_core::Result<PermissionDecision>> + Send>,
        > {
            Box::pin(async { Ok(PermissionDecision::Allow) })
        }
    }
    let (session, agent) = {
        let session = SessionManager::open(root.join("store"), root.join("cas"), true).unwrap();
        let agent = test_agent(
            session.clone(),
            vec![
                ScriptedResponse::ToolCall {
                    id: "c1".into(),
                    name: "echo".into(),
                    input: serde_json::json!({"x": 1}),
                },
                ScriptedResponse::Text("done".into()),
                ScriptedResponse::End,
            ],
            Arc::new(AlwaysAllow),
        );
        (session, agent)
    };
    let ws = session.create_workspace("/w").unwrap();
    let row = session
        .create_session(ws, "restart test", "fake", "m")
        .unwrap();
    let outcome = agent.run_turn(row.id(), "use echo", &[]).await.unwrap();
    assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);
    drop(session);
    drop(agent);

    // CRASH: reopen the daemon store.
    let session2 = SessionManager::open(root.join("store"), root.join("cas"), true).unwrap();
    let perm2 = Arc::new(AlwaysAllow);
    let agent2 = test_agent(
        session2.clone(),
        vec![ScriptedResponse::End], // fresh provider: NO tool calls this time
        perm2,
    );
    // Recovery finds no pending runs (the tool finished before the crash)
    // and the session is intact.
    let reports = agent2.recover().unwrap();
    let pending_anywhere: usize = reports.iter().map(|r| r.crashed_ops.len()).sum();
    assert_eq!(
        pending_anywhere, 0,
        "no unfinished ops after a clean finish: {reports:?}"
    );
    let handle = session2.get_session(row.id()).unwrap().unwrap();
    let page = handle.messages_page(None, 100).unwrap();
    let texts: Vec<&String> = page
        .messages
        .iter()
        .flat_map(|m| m.parts.iter())
        .filter_map(|p| match p {
            faktor_protocol::v756::Part::Text { text } => Some(text),
            _ => None,
        })
        .collect();
    assert!(
        texts.iter().any(|t| t.contains("done")),
        "assistant text survived the restart"
    );
    let has_tool_result = page
        .messages
        .iter()
        .flat_map(|m| m.parts.iter())
        .any(|p| matches!(p, faktor_protocol::v756::Part::ToolResult { .. }));
    assert!(has_tool_result, "tool result survived the restart");
}

/// SSE resume: subscribe with a cursor after a full turn; new events only.
#[tokio::test]
async fn sse_resumes_from_cursor_after_reconnect() {
    let dir = tempdir().unwrap();
    let session =
        SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
    let perm = ChannelPermissionRequester::new(Duration::from_secs(5));
    let agent = test_agent(
        session.clone(),
        vec![
            ScriptedResponse::Text("hello".into()),
            ScriptedResponse::End,
        ],
        perm.clone(),
    );
    let deps = ServerDeps::new(session.clone(), agent.clone(), perm);
    let handle = serve(deps, 0).await.unwrap();
    let client = reqwest::Client::new();
    let base = format!("http://{}", handle.addr);

    // Get the real token from the handshake line.
    let token = {
        let hs = faktor_protocol::v756::Handshake::from_line(&handle.handshake).unwrap();
        hs.auth_token
    };

    let resp = client
        .post(format!("{base}/api/session"))
        .bearer_auth(&token)
        .json(&serde_json::json!({"provider": "fake", "model": "m"}))
        .send()
        .await
        .unwrap();
    let created: serde_json::Value = resp.json().await.unwrap();
    let sid = created["id"].as_str().unwrap().to_string();

    client
        .post(format!("{base}/api/session/{sid}/prompt"))
        .bearer_auth(&token)
        .json(&serde_json::json!({"prompt": "hi"}))
        .send()
        .await
        .unwrap();

    // Wait for the turn to finish.
    let mut state = String::new();
    for _ in 0..100 {
        let body: serde_json::Value = client
            .get(format!("{base}/api/session/{sid}/state"))
            .bearer_auth(&token)
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        state = body["agent_state"]["state"]
            .as_str()
            .unwrap_or("")
            .to_string();
        if state == "ready_for_next_turn" {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(state, "ready_for_next_turn");

    // The session state endpoint reports the last event seq; a reconnect
    // with that cursor must deliver only future events (here: none new, so
    // the stream stays open and heartbeats arrive).
    let body: serde_json::Value = client
        .get(format!("{base}/api/session/{sid}/state"))
        .bearer_auth(&token)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let last_seq = body["last_event_seq"].as_i64().unwrap();

    use futures_util::StreamExt;
    let mut sse = client
        .get(format!(
            "{base}/api/session/{sid}/events?events_after={last_seq}"
        ))
        .bearer_auth(&token)
        .send()
        .await
        .unwrap()
        .bytes_stream();
    let mut saw_heartbeat = false;
    let mut text = String::new();
    for _ in 0..150 {
        match tokio::time::timeout(Duration::from_millis(200), sse.next()).await {
            Ok(Some(Ok(chunk))) => {
                text.push_str(&String::from_utf8_lossy(&chunk));
                if text.contains("heartbeat") {
                    saw_heartbeat = true;
                    break;
                }
            }
            _ => break,
        }
    }
    assert!(
        saw_heartbeat,
        "reconnect at the tail must stay open with heartbeats"
    );
    let _ = handle.shutdown.send(());
}

/// Hostile HTTP: oversized body, malformed JSON, bad ids — all clean 4xx.
#[tokio::test]
async fn hostile_http_is_clean_4xx() {
    let dir = tempdir().unwrap();
    let session =
        SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
    let perm = ChannelPermissionRequester::new(Duration::from_secs(5));
    let agent = test_agent(session.clone(), vec![ScriptedResponse::End], perm.clone());
    let deps = ServerDeps::new(session.clone(), agent, perm);
    let handle = serve(deps, 0).await.unwrap();
    let token = faktor_protocol::v756::Handshake::from_line(&handle.handshake)
        .unwrap()
        .auth_token;
    let client = reqwest::Client::new();
    let base = format!("http://{}", handle.addr);

    // Malformed JSON → 400.
    let resp = client
        .post(format!("{base}/api/session"))
        .bearer_auth(&token)
        .body("{broken")
        .header("content-type", "application/json")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);

    // Oversized body (>10MB) → 413 (or an early-close connection error;
    // either way the daemon must survive and keep serving).
    let big = serde_json::json!({"provider": "fake", "model": "m", "title": "x".repeat(11 * 1024 * 1024)});
    match client
        .post(format!("{base}/api/session"))
        .bearer_auth(&token)
        .json(&big)
        .send()
        .await
    {
        Ok(resp) => assert_eq!(resp.status(), 413),
        Err(_e) => { /* early close is acceptable; daemon liveness below */ }
    }
    // The daemon is still alive and serving.
    let resp = client
        .get(format!("{base}/api/hello"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // Bad ids → 400; missing → 404.
    let resp = client
        .get(format!("{base}/api/session/not-a-number/state"))
        .bearer_auth(&token)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let resp = client
        .get(format!("{base}/api/session/0/state"))
        .bearer_auth(&token)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let resp = client
        .get(format!("{base}/api/session/999999/state"))
        .bearer_auth(&token)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);

    // Permission decisions: invalid values → 400; unknown id → 409.
    let resp = client
        .post(format!("{base}/api/perm/1/resolve"))
        .bearer_auth(&token)
        .json(&serde_json::json!({"permission_id": "1", "decision": "maybe"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    // A second resolve of the same id loses the race (409).
    let resp = client
        .post(format!("{base}/api/perm/424242/resolve"))
        .bearer_auth(&token)
        .json(&serde_json::json!({"permission_id": "424242", "decision": "allow"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let resp = client
        .post(format!("{base}/api/perm/424242/resolve"))
        .bearer_auth(&token)
        .json(&serde_json::json!({"permission_id": "424242", "decision": "deny"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 409);
    let _ = handle.shutdown.send(());
}

/// Permission flow end-to-end: the agent blocks on the durable permission,
/// the HTTP API resolves it, the tool runs exactly once.
#[tokio::test]
async fn permission_flow_end_to_end() {
    let dir = tempdir().unwrap();
    let session =
        SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
    let perm = ChannelPermissionRequester::new(Duration::from_secs(10));
    let agent = test_agent(
        session.clone(),
        vec![
            ScriptedResponse::ToolCall {
                id: "c1".into(),
                name: "echo".into(),
                input: serde_json::json!({"x": 1}),
            },
            ScriptedResponse::End,
        ],
        perm.clone(),
    );
    let deps = ServerDeps::new(session.clone(), agent.clone(), perm.clone());
    let handle = serve(deps, 0).await.unwrap();
    let token = faktor_protocol::v756::Handshake::from_line(&handle.handshake)
        .unwrap()
        .auth_token;
    let client = reqwest::Client::new();
    let base = format!("http://{}", handle.addr);

    let resp = client
        .post(format!("{base}/api/session"))
        .bearer_auth(&token)
        .json(&serde_json::json!({"provider": "fake", "model": "m"}))
        .send()
        .await
        .unwrap();
    let sid = resp.json::<serde_json::Value>().await.unwrap()["id"]
        .as_str()
        .unwrap()
        .to_string();

    let turn = tokio::spawn({
        let agent = agent.clone();
        let sid = sid.clone();
        async move {
            let id: u64 = sid.parse().unwrap();
            agent.run_turn(SessionId::new(id), "use echo", &[]).await
        }
    });

    // The turn blocks on permission; resolve it via the API.
    let mut resolved = false;
    for _ in 0..100 {
        if let Some(pid) = perm.pending_ids().first().copied() {
            let resp = client
                .post(format!("{base}/api/perm/{pid}/resolve"))
                .bearer_auth(&token)
                .json(&serde_json::json!({"permission_id": pid.to_string(), "decision": "allow"}))
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 200);
            resolved = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(resolved, "permission must surface");

    let outcome = tokio::time::timeout(Duration::from_secs(10), turn)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);
    // The tool ran exactly once.
    let sh = session
        .get_session(SessionId::new(sid.parse().unwrap()))
        .unwrap()
        .unwrap();
    assert!(sh.pending_tool_runs().unwrap().is_empty());
    let _ = handle.shutdown.send(());
}

/// Compaction under a growing session never exceeds the budget and cannot
/// loop (death-spiral guard at the daemon level).
#[tokio::test]
async fn compaction_under_load_never_spirals() {
    let dir = tempdir().unwrap();
    let session =
        SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
    let perm = ChannelPermissionRequester::new(Duration::from_secs(5));
    let mut deps = test_agent_deps(session.clone(), perm.clone());
    deps.compact_at_usage = 0.01; // aggressive trigger
    deps.instructions = "You are a test agent.".into();
    let agent = AgentRuntime::new(deps).unwrap();
    let ws = session.create_workspace("/w").unwrap();
    let row = session.create_session(ws, "compact", "fake", "m").unwrap();

    // 30 turns with growing responses.
    for i in 0..30 {
        let script = vec![
            ScriptedResponse::Text(format!("turn {i} {}", "x".repeat(2000))),
            ScriptedResponse::End,
        ];
        // Rebuild the provider per turn (FakeProvider scripts are single-use).
        let mut registry = ProviderRegistry::new();
        registry.register(Arc::new(FakeProvider::with_script(
            "fake",
            ModelCapabilities {
                tools: true,
                ..Default::default()
            },
            script,
        )));
        let agent2 = {
            let mut deps = test_agent_deps(session.clone(), perm.clone());
            deps.compact_at_usage = 0.01;
            deps.instructions = "You are a test agent.".into();
            deps.providers = Arc::new(registry);
            AgentRuntime::new(deps).unwrap()
        };
        let outcome = agent2
            .run_turn(row.id(), &format!("turn {i}"), &[])
            .await
            .unwrap();
        assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);
    }
    // The session is still coherent and paged.
    let handle = session.get_session(row.id()).unwrap().unwrap();
    let page = handle.messages_page(None, 5).unwrap();
    assert!(page.messages.len() <= 5);
    assert!(page.has_more);
    let _ = agent;
}

fn test_agent_deps(
    session: Arc<SessionManager>,
    permissions: Arc<ChannelPermissionRequester>,
) -> AgentDeps {
    let mut registry = ProviderRegistry::new();
    registry.register(Arc::new(FakeProvider::with_script(
        "fake",
        ModelCapabilities {
            tools: true,
            ..Default::default()
        },
        vec![ScriptedResponse::End],
    )));
    AgentDeps {
        session,
        providers: Arc::new(registry),
        chunk_sink: None,
        permission_requester: permissions,
        evidence: Arc::new(NoEvidence),
        tools: Arc::new(ToolRegistry::new()),
        cas: None,
        workspaces: faktor_fs::WorkspaceFileService::new(),
        edit: None,
        snapshots: None,
        sandbox: None,
        supervisor: None,
        verifier: None,
        hooks: None,
        instructions_loader: None,
        model: "m".into(),
        compaction_model: None,
        compact_at_usage: 0.65,
        instructions: "i".into(),
        clock: Arc::new(SystemClock),
        tool_call_mode: faktor_agent::ToolCallMode::Native,
        tool_deadline_ms: 2000,
        retry_policy: faktor_core::retry::RetryPolicy::default(),
    }
}

/// 100k-message session loads in constant time (paging is fundamental).
#[tokio::test]
async fn hundred_thousand_message_session_loads_constantly() {
    let dir = tempdir().unwrap();
    let session =
        SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
    let ws = session.create_workspace("/w").unwrap();
    let row = session.create_session(ws, "big", "fake", "m").unwrap();
    let big_handle = session.get_session(row.id()).unwrap().unwrap();
    for i in 1..=100_000 {
        big_handle
            .put_message(i, "user", serde_json::json!({"text": format!("m{i}")}))
            .unwrap();
    }
    let handle = session.get_session(row.id()).unwrap().unwrap();
    let t0 = std::time::Instant::now();
    let page = handle.messages_page(None, 100).unwrap();
    let load_ms = t0.elapsed();
    assert_eq!(page.messages.len(), 100);
    assert!(page.has_more);
    // 100k messages: the initial page must not depend on history size.
    assert!(
        load_ms < Duration::from_millis(500),
        "page load took {load_ms:?}"
    );
    // A second page via the cursor.
    let page2 = handle
        .messages_page(Some(page.next_before.unwrap()), 100)
        .unwrap();
    assert_eq!(page2.messages.len(), 100);
    assert!(page2.messages[0].seq < page.messages[0].seq);
}

/// The synthetic fixture repository indexes correctly: symbols extracted
/// from real files (spec §19/§44 fixtures/repositories).
#[test]
fn fixture_repository_indexes() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/repositories/parser-demo");
    let mut idx = faktor_index::WorkspaceIndex::new();
    let ws = faktor_core::id::WorkspaceId::new(1);
    let mut files = 0usize;
    for entry in std::fs::read_dir(root.join("src")).unwrap().flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            let bytes = std::fs::read(&path).unwrap();
            idx.index_file(ws, &path, &bytes, 1).unwrap();
            files += 1;
        }
    }
    assert!(files >= 3, "fixture repo must have rust sources");
    // Symbols from the fixture: Lexer struct, Parser struct, new, parse,
    // next_token, lexer_advances test, parses_identifiers test.
    let all: Vec<_> = ["lexer.rs", "parser.rs"]
        .iter()
        .flat_map(|f| idx.symbols_in(ws, &root.join("src").join(f)))
        .collect();
    let names: std::collections::HashSet<&str> = all.iter().map(|s| s.name.as_str()).collect();
    for expected in [
        "Lexer",
        "Parser",
        "next_token",
        "parse",
        "lexer_advances",
        "parses_identifiers",
    ] {
        assert!(
            names.contains(expected),
            "missing symbol {expected}: {names:?}"
        );
    }
    // Tests are classified as Test symbols.
    let tests = all
        .iter()
        .filter(|s| s.kind == faktor_index::SymbolKind::Test)
        .count();
    assert_eq!(tests, 2);
    // Lexical search finds a token from the fixture.
    assert!(!idx.files_for_token(ws, "lexer", 10).is_empty());
}

/// The frozen protocol surface survives a full daemon lifecycle: hello,
/// create, prompt, messages, state — byte shapes asserted.
#[tokio::test]
async fn frozen_protocol_surface_lifecycle() {
    let dir = tempdir().unwrap();
    let session =
        SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
    let perm = ChannelPermissionRequester::new(Duration::from_secs(5));
    let agent = test_agent(
        session.clone(),
        vec![ScriptedResponse::Text("pong".into()), ScriptedResponse::End],
        perm.clone(),
    );
    let deps = ServerDeps::new(session.clone(), agent, perm);
    let handle = serve(deps, 0).await.unwrap();
    let client = reqwest::Client::new();
    let base = format!("http://{}", handle.addr);
    let token = faktor_protocol::v756::Handshake::from_line(&handle.handshake)
        .unwrap()
        .auth_token;

    // hello: public, frozen shape.
    let body: serde_json::Value = client
        .get(format!("{base}/api/hello"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(body["protocol"], "v756");
    assert_eq!(body["auth_required"], true);
    assert_eq!(body["ok"], true);
    assert!(body["version"].is_string());
    // Frozen field presence: exactly these keys.
    let hello_keys: Vec<&str> = body
        .as_object()
        .unwrap()
        .keys()
        .map(|k| k.as_str())
        .collect();
    assert_eq!(hello_keys.len(), 5);

    // Session create + prompt lifecycle.
    let resp = client
        .post(format!("{base}/api/session"))
        .bearer_auth(&token)
        .json(&serde_json::json!({
            "provider": "fake",
            "model": "m",
            "workspace": "/tmp",
            "title": "frozen",
        }))
        .send()
        .await
        .unwrap();
    let created: serde_json::Value = resp.json().await.unwrap();
    assert!(created["id"].is_string());
    assert_eq!(created["title"], "frozen");
    let sid = created["id"].as_str().unwrap().to_string();

    let resp = client
        .post(format!("{base}/api/session/{sid}/prompt"))
        .bearer_auth(&token)
        .json(&serde_json::json!({"prompt": "ping", "files": []}))
        .send()
        .await
        .unwrap();
    let pr: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(pr["accepted"], true);
    assert_eq!(pr["queued"], false);

    // Wait for the turn; then messages page has the frozen Message shape.
    for _ in 0..100 {
        let body: serde_json::Value = client
            .get(format!("{base}/api/session/{sid}/state"))
            .bearer_auth(&token)
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        if body["agent_state"]["state"].as_str() == Some("ready_for_next_turn") {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let page: serde_json::Value = client
        .get(format!("{base}/api/session/{sid}/messages?limit=10"))
        .bearer_auth(&token)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let messages = page["messages"].as_array().unwrap();
    assert!(messages.len() >= 2);
    // Frozen Message field presence.
    for m in messages {
        for key in ["id", "role", "session_id", "seq", "created_ms", "parts"] {
            assert!(m.as_object().unwrap().contains_key(key), "missing {key}");
        }
    }
    let _ = handle.shutdown.send(());
}

// ------------------------------------------------------------------ wire flow

/// One queued response sequence of [`QueueProvider`]. `FakeProvider`'s
/// script is consumed whole by the FIRST stream request (a later request
/// would see an empty script and end instantly), so a multi-turn daemon
/// cannot reuse it per turn: this wrapper pops one script per NEW stream
/// request (one request per logical turn with text-only scripts) and serves
/// an immediate terminal stream when the queue is exhausted — a turn can
/// never hang on a scripted provider.
struct QueueProvider {
    caps: ModelCapabilities,
    scripts: Mutex<VecDeque<QueueItem>>,
    /// The abort gate for a `Hold` request: opened by the test once the
    /// abort of the in-flight turn is durable.
    release: watch::Receiver<bool>,
}

enum QueueItem {
    /// Serve exactly these responses on the next stream request, then Done.
    Script(Vec<ScriptedResponse>),
    /// Keep the next stream request open (yielding nothing) until the test
    /// releases the gate, then end the stream cleanly.
    Hold,
}

impl QueueProvider {
    fn new(scripts: Vec<QueueItem>, release: watch::Receiver<bool>) -> Self {
        Self {
            caps: ModelCapabilities {
                tools: true,
                ..Default::default()
            },
            scripts: Mutex::new(scripts.into()),
            release,
        }
    }
}

/// Replay one script exactly like `FakeProvider` (every chunk one stream
/// item; `End`/exhaustion emits exactly one terminal `Done`).
fn play_script(script: Vec<ScriptedResponse>) -> ProviderStream {
    Box::pin(futures_util::stream::unfold(
        (script.into_iter(), false),
        |(mut remaining, ended)| async move {
            if ended {
                return None;
            }
            match remaining.next() {
                Some(ScriptedResponse::Text(t)) => {
                    Some((Ok(ProviderChunk::Text { text: t }), (remaining, false)))
                }
                Some(ScriptedResponse::Reasoning(t)) => {
                    Some((Ok(ProviderChunk::Reasoning { text: t }), (remaining, false)))
                }
                Some(ScriptedResponse::ToolCall { id, name, input }) => Some((
                    Ok(ProviderChunk::ToolCall {
                        id,
                        name,
                        input,
                        complete: true,
                    }),
                    (remaining, false),
                )),
                Some(ScriptedResponse::Die(e)) => Some((Err(e), (remaining, true))),
                Some(ScriptedResponse::End) | None => {
                    Some((Ok(ProviderChunk::Done), (remaining, true)))
                }
            }
        },
    ))
}

/// A stream that yields nothing until `release` turns true, then ends
/// cleanly. The watch (not a oneshot/Notify) makes the release idempotent:
/// opening the gate before the stream polls cannot be lost.
fn hold_stream(release: watch::Receiver<bool>) -> ProviderStream {
    Box::pin(futures_util::stream::unfold(
        (false, release),
        |(ended, mut release)| async move {
            if ended {
                return None;
            }
            loop {
                if *release.borrow() {
                    break;
                }
                // The sender outlives the test; an error only means the
                // gate will never open — end instead of hanging forever.
                if release.changed().await.is_err() {
                    break;
                }
            }
            Some((Ok(ProviderChunk::Done), (true, release)))
        },
    ))
}

impl Provider for QueueProvider {
    fn id(&self) -> &str {
        "fake"
    }

    fn capabilities(&self, _model: &str) -> ModelCapabilities {
        self.caps.clone()
    }

    fn stream(&self, _req: GenericAgentRequest) -> ProviderStream {
        match self.scripts.lock().unwrap().pop_front() {
            Some(QueueItem::Script(script)) => play_script(script),
            Some(QueueItem::Hold) => hold_stream(self.release.clone()),
            // A surplus request (nothing scripted): end instantly so the
            // turn terminates instead of hanging.
            None => play_script(vec![ScriptedResponse::End]),
        }
    }
}

/// Drain every complete SSE frame from the raw byte buffer. Heartbeat and
/// keep-alive frames have no parseable SseEvent shape and are dropped.
fn drain_sse(buffer: &Arc<Mutex<String>>, frames: &mut Vec<(u64, SseEvent)>) {
    let mut buf = buffer.lock().unwrap();
    while let Some(pos) = buf.find("\n\n") {
        let end = pos + 2;
        let frame = buf[..end].to_string();
        buf.drain(..end);
        if let Some(parsed) = SseEvent::from_frame(&frame) {
            frames.push(parsed);
        }
    }
}

/// The `state` label of an `agent_state_changed` frame (the journal's state
/// label: "streaming", "ready", ...).
fn sse_state_label(frame: &(u64, SseEvent)) -> Option<String> {
    match &frame.1 {
        SseEvent::AgentStateChanged { state, .. } => Some(state.clone()),
        _ => None,
    }
}

/// Subscribe to the per-session SSE stream and collect raw bytes into a
/// shared buffer until the task is aborted.
fn spawn_sse_collector(
    client: reqwest::Client,
    url: String,
    token: String,
) -> (tokio::task::JoinHandle<()>, Arc<Mutex<String>>) {
    let buffer: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
    let sink = buffer.clone();
    let task = tokio::spawn(async move {
        use futures_util::StreamExt;
        let Ok(resp) = client.get(&url).bearer_auth(&token).send().await else {
            return;
        };
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let Ok(bytes) = chunk else { break };
            let Ok(text) = String::from_utf8(bytes.to_vec()) else {
                continue;
            };
            sink.lock().unwrap().push_str(&text);
        }
    });
    (task, buffer)
}

/// Poll `ok()` (which drains the SSE buffer into `frames`) until it holds.
async fn wait_for(mut ok: impl FnMut() -> bool, what: &str) {
    for _ in 0..500 {
        if ok() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("timed out waiting for {what}");
}

async fn session_state(
    client: &reqwest::Client,
    base: &str,
    sid: &str,
    token: &str,
) -> serde_json::Value {
    client
        .get(format!("{base}/api/session/{sid}/state"))
        .bearer_auth(token)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap()
}

/// Poll the durable state view until `agent_state.state` equals `want`.
async fn wait_state(
    client: &reqwest::Client,
    base: &str,
    sid: &str,
    token: &str,
    want: &str,
) -> serde_json::Value {
    for _ in 0..500 {
        let body = session_state(client, base, sid, token).await;
        if body["agent_state"]["state"].as_str() == Some(want) {
            return body;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("session state never reached {want}");
}

/// `POST /session/{sessionID}/message` with one text part (the frozen wire
/// turn endpoint); returns (status, body).
async fn wire_send_message(
    client: &reqwest::Client,
    base: &str,
    sid: &str,
    token: &str,
    prompt: &str,
) -> (u16, serde_json::Value) {
    let resp = client
        .post(format!("{base}/session/{sid}/message"))
        .bearer_auth(token)
        .json(&serde_json::json!({
            "model": {"providerID": "fake", "modelID": "m"},
            "parts": [{"type": "text", "text": prompt}],
        }))
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let body = resp.json().await.unwrap();
    (status, body)
}

/// The text of every wire text part of one `{info, parts}` entry.
fn entry_texts(entry: &serde_json::Value) -> Vec<String> {
    entry["parts"]
        .as_array()
        .map(|parts| {
            parts
                .iter()
                .filter(|p| p["type"] == "text")
                .filter_map(|p| p["text"].as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

/// The P0 conversation flow over the REAL HTTP + SSE surface with a
/// deterministic in-process provider: create session, first turn ("ping" →
/// assembled "pong") through `POST /session/{id}/message`, the durable
/// message page, live per-turn SSE state frames, a second successful turn
/// ("pong2") on a fresh script, an SSE reconnect resumed from the session's
/// last event seq (post-cursor events only, no earlier duplicates), and a
/// targeted abort of a third in-flight turn that lands the session
/// ReadyForNextTurn.
///
/// Text-only scripts: the runtime makes one provider stream request per
/// logical turn, and `QueueProvider` pops one script per request, so every
/// NEW turn sees fresh content. The third request holds on a gate the test
/// releases only after the abort is durable.
///
/// Note on SSE text: this runtime slice never journals per-chunk text
/// deltas (parts are flushed durably in bounded segments), so the wire
/// carries no `message_part_updated` frames for a live turn — the asserted
/// live frames are the per-turn `agent_state_changed` stream ("streaming"
/// through "ready") and the text itself is asserted from the durable
/// `{info, parts}` surfaces.
#[tokio::test]
async fn deterministic_provider_full_wire_conversation_flow() {
    let dir = tempdir().unwrap();
    let session =
        SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
    let perm = ChannelPermissionRequester::new(Duration::from_secs(5));
    let (release_tx, release_rx) = watch::channel(false);
    let provider = QueueProvider::new(
        vec![
            // Turn 1: "po" then "ng" — the assembled assistant text is "pong".
            QueueItem::Script(vec![
                ScriptedResponse::Text("po".into()),
                ScriptedResponse::Text("ng".into()),
                ScriptedResponse::End,
            ]),
            // Turn 2: a distinct fresh answer for the second logical turn.
            QueueItem::Script(vec![
                ScriptedResponse::Text("pong2".into()),
                ScriptedResponse::End,
            ]),
            // Turn 3: stays in flight until the gate opens after the abort.
            QueueItem::Hold,
        ],
        release_rx,
    );
    let mut registry = ProviderRegistry::new();
    registry.register(Arc::new(provider));
    // Live chunk stream: the agent forwards streaming text so subscribers
    // see session.next.text.delta frames at low latency.
    let (sink, rx) = faktor_agent::ChunkSink::channel();
    let (agent, _chunk_rx) =
        agent_with_registry_and_sink(session.clone(), registry, perm.clone(), Some(sink));
    let mut deps = ServerDeps::new(session.clone(), agent, perm);
    deps.chunk_rx = Some(rx);
    let handle = serve(deps, 0).await.unwrap();
    let token = faktor_protocol::v756::Handshake::from_line(&handle.handshake)
        .unwrap()
        .auth_token;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .build()
        .unwrap();
    let base = format!("http://{}", handle.addr);

    // 1. Create the session through the wire surface.
    let resp = client
        .post(format!("{base}/session"))
        .bearer_auth(&token)
        .json(&serde_json::json!({"model": {"id": "m", "providerID": "fake"}}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let created: serde_json::Value = resp.json().await.unwrap();
    let sid = created["sessionID"].as_str().unwrap().to_string();
    assert!(created["createdMs"].as_i64().unwrap() > 0);

    // Subscribe to the SSE stream BEFORE the first prompt; the subscription
    // is live once the SessionCreated frame (session_updated) arrives.
    let url = format!("{base}/api/session/{sid}/events?events_after=0");
    let (sse1, buf1) = spawn_sse_collector(client.clone(), url, token.clone());
    // Global stream: the live chunk fan-out lands on the GLOBAL ring (the
    // per-session events endpoint is journal-only by design).
    let (sse_global, buf_global) = spawn_sse_collector(
        client.clone(),
        format!("{base}/global/event?after=0"),
        token.clone(),
    );
    let frames_global: Vec<(u64, SseEvent)> = Vec::new();
    let mut frames1: Vec<(u64, SseEvent)> = Vec::new();
    wait_for(
        || {
            drain_sse(&buf1, &mut frames1);
            frames1
                .iter()
                .any(|(_, e)| matches!(e, SseEvent::SessionUpdated { .. }))
        },
        "initial SSE frame",
    )
    .await;
    let turn1_begin = frames1.len();

    // 2. First prompt: the daemon runs the turn inside the request and the
    // response exposes the assembled durable assistant message as the
    // frozen {info, parts} shape with the "pong" text.
    let (status, body) = wire_send_message(&client, &base, &sid, &token, "ping").await;
    assert_eq!(status, 200, "{body}");
    let keys: Vec<&str> = body
        .as_object()
        .unwrap()
        .keys()
        .map(|k| k.as_str())
        .collect();
    assert_eq!(keys, ["info", "parts"]);
    assert_eq!(body["info"]["sessionID"], sid);
    assert_eq!(body["info"]["role"], "assistant");
    let mid1: u64 = body["info"]["messageID"].as_str().unwrap().parse().unwrap();
    assert!(mid1 > 1, "assistant lands after the user prompt: {body}");
    assert!(body["info"]["createdMs"].as_i64().unwrap() > 0);
    assert_eq!(body["info"]["providerID"], "fake");
    assert_eq!(body["info"]["modelID"], "m");
    assert!(
        body["parts"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["type"] == "text" && p["text"] == "pong"),
        "the assembled 'pong' text rides the response parts: {body}"
    );

    // 4. Live per-turn SSE while the first turn ran: at least the streaming
    // phase (when the text was produced) and the terminal ready frame.
    wait_for(
        || {
            drain_sse(&buf1, &mut frames1);
            let labels: Vec<String> = frames1[turn1_begin..]
                .iter()
                .filter_map(sse_state_label)
                .collect();
            labels.iter().any(|s| s == "streaming") && labels.iter().any(|s| s == "ready")
        },
        "live per-turn SSE frames of turn 1",
    )
    .await;
    assert!(
        frames1[turn1_begin..]
            .iter()
            .all(|(_, e)| matches!(e, SseEvent::AgentStateChanged { .. })),
        "turn 1 emits state frames only: {:?}",
        frames1[turn1_begin..]
            .iter()
            .map(|(_, e)| e.event_type())
            .collect::<Vec<_>>()
    );
    wait_state(&client, &base, &sid, &token, "ready_for_next_turn").await;

    // 3. The durable assistant message exists on the wire messages page:
    // the bare array of {info, parts}, with the pong text part.
    let resp = client
        .get(format!("{base}/session/{sid}/message?limit=10"))
        .bearer_auth(&token)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let page: serde_json::Value = resp.json().await.unwrap();
    let entries = page.as_array().unwrap();
    let assistants: Vec<&serde_json::Value> = entries
        .iter()
        .filter(|e| e["info"]["role"] == "assistant")
        .collect();
    assert_eq!(assistants.len(), 1, "{page}");
    // Live chunk fan-out (audit round 11): the sink pushes text deltas
    // onto the GLOBAL event ring; the global stream delivers GlobalEvent
    // envelopes (a different shape than the per-session SseEvent frames),
    // so assert on the raw SSE payload.
    wait_for(
        || {
            let raw = buf_global.lock().unwrap().clone();
            raw.contains("session_next_text_delta")
        },
        "live session.next.text.delta frame from the chunk sink",
    )
    .await;
    drop((sse_global, frames_global));
    assert_eq!(entry_texts(assistants[0]), ["pong"]);
    assert_eq!(
        assistants[0]["info"]["messageID"],
        body["info"]["messageID"]
    );

    // 5. A SECOND prompt is a second successful turn with its own fresh
    // script ("pong2") and lands durably.
    let (status, body) = wire_send_message(&client, &base, &sid, &token, "ping2").await;
    assert_eq!(status, 200, "{body}");
    let mid2: u64 = body["info"]["messageID"].as_str().unwrap().parse().unwrap();
    assert!(mid2 > mid1, "the second assistant message is newer");
    assert!(
        body["parts"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["type"] == "text" && p["text"] == "pong2"),
        "{body}"
    );
    wait_for(
        || {
            drain_sse(&buf1, &mut frames1);
            let ready: usize = frames1[turn1_begin..]
                .iter()
                .filter_map(sse_state_label)
                .filter(|s| s == "ready")
                .count();
            ready >= 2
        },
        "live per-turn SSE frames of turn 2",
    )
    .await;
    let state = wait_state(&client, &base, &sid, &token, "ready_for_next_turn").await;

    // 6. Disconnect and RECONNECT with the session's last event seq: only
    // events after the cursor may arrive.
    let last_seq: i64 = state["last_event_seq"].as_i64().unwrap();
    sse1.abort();
    let url = format!("{base}/api/session/{sid}/events?events_after={last_seq}");
    let (sse2, buf2) = spawn_sse_collector(client.clone(), url, token.clone());
    let mut frames2: Vec<(u64, SseEvent)> = Vec::new();
    tokio::time::sleep(Duration::from_millis(300)).await;
    drain_sse(&buf2, &mut frames2);
    assert!(
        frames2.iter().all(|(id, _)| *id as i64 > last_seq),
        "resume from the cursor must never replay earlier events: {frames2:?}"
    );

    // 7. Third prompt: the provider request stays in flight (QueueItem::Hold
    // yields nothing until the gate opens). The SDK surface carries the
    // REAL op id, and /session/abort targets exactly that op.
    let resp = client
        .post(format!("{base}/session/prompt"))
        .bearer_auth(&token)
        .json(&serde_json::json!({"session_id": sid, "prompt": "never finishes"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let pr: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(pr["accepted"], true);
    assert_eq!(pr["queued"], false);
    let op_id = pr["op_id"].as_str().unwrap().to_string();
    wait_state(&client, &base, &sid, &token, "streaming").await;
    // The reconnected stream delivers the events that occurred after the
    // cursor while the turn is in flight.
    wait_for(
        || {
            drain_sse(&buf2, &mut frames2);
            frames2
                .iter()
                .filter_map(sse_state_label)
                .any(|s| s == "streaming")
        },
        "post-cursor live frames of the in-flight turn",
    )
    .await;
    // Targeted abort of the running operation.
    let resp = client
        .post(format!("{base}/session/abort"))
        .bearer_auth(&token)
        .json(&serde_json::json!({"session_id": sid, "op_id": op_id}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let ab: serde_json::Value = resp.json().await.unwrap();
    let aborted = ab["aborted"].as_array().unwrap();
    assert_eq!(
        aborted
            .iter()
            .filter_map(|o| o.as_str())
            .collect::<Vec<_>>(),
        [op_id.as_str()],
        "{ab}"
    );
    // The machine lands ReadyForNextTurn (Stop cancels the turn, never the
    // session — review P0-2).
    wait_state(&client, &base, &sid, &token, "ready_for_next_turn").await;
    // Only after the abort is durable does the provider stream end.
    release_tx.send(true).unwrap();
    wait_for(
        || {
            drain_sse(&buf2, &mut frames2);
            frames2
                .iter()
                .filter_map(sse_state_label)
                .any(|s| s == "ready")
        },
        "post-abort state frame",
    )
    .await;
    // No frame older than the resume cursor ever reappeared.
    assert!(
        frames2.iter().all(|(id, _)| *id as i64 > last_seq),
        "resume delivered duplicates or older events: {frames2:?}"
    );
    sse2.abort();

    // The aborted third turn left its user message but NO assistant reply,
    // and the two earlier turns stayed durable (newest first).
    tokio::time::sleep(Duration::from_millis(500)).await;
    let resp = client
        .get(format!("{base}/session/{sid}/message?limit=10"))
        .bearer_auth(&token)
        .send()
        .await
        .unwrap();
    let page: serde_json::Value = resp.json().await.unwrap();
    let entries = page.as_array().unwrap();
    let roles: Vec<&str> = entries
        .iter()
        .filter_map(|e| e["info"]["role"].as_str())
        .collect();
    assert_eq!(roles, ["user", "assistant", "user", "assistant", "user"]);
    let assistant_texts: Vec<Vec<String>> = entries
        .iter()
        .filter(|e| e["info"]["role"] == "assistant")
        .map(entry_texts)
        .collect();
    assert_eq!(
        assistant_texts,
        [vec!["pong2".to_string()], vec!["pong".to_string()]]
    );
    assert!(
        entry_texts(&entries[0])
            .iter()
            .any(|t| t.contains("never finishes")),
        "the aborted turn's user message is durable: {page}"
    );
    assert_eq!(
        session_state(&client, &base, &sid, &token).await["agent_state"]["state"],
        "ready_for_next_turn",
        "the session stays usable after the targeted abort"
    );
    let _ = handle.shutdown.send(());
}

/// A genuinely provider-less daemon keeps the honest 502 semantics: the
/// wire message endpoint runs the turn, the missing model backend fails it
/// durably, no assistant message exists, and the harness must NOT special-
/// case the failure as success.
#[tokio::test]
async fn provider_less_message_send_is_an_honest_502() {
    let dir = tempdir().unwrap();
    let session =
        SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
    let perm = ChannelPermissionRequester::new(Duration::from_secs(5));
    let agent = agent_with_registry(session.clone(), ProviderRegistry::new(), perm.clone());
    let deps = ServerDeps::new(session.clone(), agent, perm);
    let handle = serve(deps, 0).await.unwrap();
    let token = faktor_protocol::v756::Handshake::from_line(&handle.handshake)
        .unwrap()
        .auth_token;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .build()
        .unwrap();
    let base = format!("http://{}", handle.addr);

    let resp = client
        .post(format!("{base}/session"))
        .bearer_auth(&token)
        .json(&serde_json::json!({"model": {"id": "m", "providerID": "fake"}}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let created: serde_json::Value = resp.json().await.unwrap();
    let sid = created["sessionID"].as_str().unwrap().to_string();

    // No provider is registered for the daemon: the turn honestly fails and
    // the endpoint answers 502 — never a synthetic success.
    let (status, body) = wire_send_message(&client, &base, &sid, &token, "ping").await;
    assert_eq!(status, 502, "{body}");
    assert_eq!(body["ok"], false);
    // The failure is durable: the machine lands FailedRecoverable (a
    // promptable state — the daemon is not wedged).
    wait_state(&client, &base, &sid, &token, "failed_recoverable").await;
    // No assistant message was ever materialized; the user prompt row is.
    let resp = client
        .get(format!("{base}/session/{sid}/message?limit=10"))
        .bearer_auth(&token)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let page: serde_json::Value = resp.json().await.unwrap();
    let entries = page.as_array().unwrap();
    assert!(
        entries.iter().all(|e| e["info"]["role"] == "user"),
        "provider-less turn must not fabricate an assistant: {page}"
    );
    // The daemon stays alive and serving.
    let resp = client
        .get(format!("{base}/api/hello"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let _ = handle.shutdown.send(());
}
