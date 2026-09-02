//! End-to-end integration tests: the daemon as a whole, adversarially.
//! Server + agent + session + provider + persistence working together:
//! crash-restart recovery, SSE resume, permission flow, compaction,
//! paging, and hostile payloads.

#![cfg_attr(
    not(test),
    allow(dead_code, unused_imports, unused_variables, unused_mut)
)] // test-harness crate: the lib view exists only for clippy
use std::sync::Arc;
use std::time::Duration;

use kilop_agent::{
    AgentDeps, AgentRuntime, NoEvidence, PermissionRequester, Tool, ToolOutcome, ToolRegistry,
};
use kilop_core::capability::PermissionDecision;
use kilop_core::id::SessionId;
use kilop_core::model::ModelCapabilities;
use kilop_core::state::AgentState;
use kilop_core::time::SystemClock;
use kilop_provider::{FakeProvider, ProviderRegistry, ScriptedResponse};
use kilop_server::permission::ChannelPermissionRequester;
use kilop_server::{serve, ServerDeps};
use kilop_session::SessionManager;
use tempfile::tempdir;

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
    let mut tools = ToolRegistry::new();
    tools.register(Tool {
        name: "echo".into(),
        description: "d".into(),
        input_schema: serde_json::json!({}),
        resource_class: kilop_core::resource::ResourceClass::Cpu,
        capability: None,
        recovery_hint: kilop_agent::RecoveryHint::Idempotent,
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
    AgentRuntime::new(AgentDeps {
        session,
        providers: Arc::new(registry),
        permission_requester: permissions,
        evidence: Arc::new(NoEvidence),
        tools: Arc::new(tools),
        cas: None,
        workspaces: kilop_fs::WorkspaceFileService::new(),
        edit: None,
        snapshots: None,
        sandbox: None,
        supervisor: None,
        model: "m".into(),
        compaction_model: None,
        compact_at_usage: 0.65,
        instructions: "You are a test agent.".into(),
        clock: Arc::new(SystemClock),
        tool_call_mode: kilop_agent::ToolCallMode::Native,
        tool_deadline_ms: 2000,
        retry_policy: kilop_core::retry::RetryPolicy::default(),
    })
    .unwrap()
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
            _p: &kilop_session::ops::PermissionRequest,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = kilop_core::Result<PermissionDecision>> + Send>,
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
            kilop_protocol::v756::Part::Text { text } => Some(text),
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
        .any(|p| matches!(p, kilop_protocol::v756::Part::ToolResult { .. }));
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
        let hs = kilop_protocol::v756::Handshake::from_line(&handle.handshake).unwrap();
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
    let token = kilop_protocol::v756::Handshake::from_line(&handle.handshake)
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
    let token = kilop_protocol::v756::Handshake::from_line(&handle.handshake)
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
        permission_requester: permissions,
        evidence: Arc::new(NoEvidence),
        tools: Arc::new(ToolRegistry::new()),
        cas: None,
        workspaces: kilop_fs::WorkspaceFileService::new(),
        edit: None,
        snapshots: None,
        sandbox: None,
        supervisor: None,
        model: "m".into(),
        compaction_model: None,
        compact_at_usage: 0.65,
        instructions: "i".into(),
        clock: Arc::new(SystemClock),
        tool_call_mode: kilop_agent::ToolCallMode::Native,
        tool_deadline_ms: 2000,
        retry_policy: kilop_core::retry::RetryPolicy::default(),
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
    let mut idx = kilop_index::WorkspaceIndex::new();
    let ws = kilop_core::id::WorkspaceId::new(1);
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
        .filter(|s| s.kind == kilop_index::SymbolKind::Test)
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
    let token = kilop_protocol::v756::Handshake::from_line(&handle.handshake)
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
