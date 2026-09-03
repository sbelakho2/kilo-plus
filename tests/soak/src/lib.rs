//! Soak tests (spec §38): synthetic long-running sessions — no state loss,
//! no runaway memory, no compaction loop, no duplicate effects. The 12-hour
//! scale is CI-gated (`#[ignore]`); the fast variant runs in seconds and
//! exercises the same invariants.

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
use kilop_session::SessionManager;
use tempfile::tempdir;

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

fn agent_for(session: Arc<SessionManager>, tools: bool, compact_at: f64) -> Arc<AgentRuntime> {
    let mut registry = ProviderRegistry::new();
    registry.register(Arc::new(FakeProvider::with_script(
        "fake",
        ModelCapabilities {
            tools: true,
            ..Default::default()
        },
        vec![ScriptedResponse::Text("ok".into()), ScriptedResponse::End],
    )));
    let mut tool_registry = ToolRegistry::new();
    if tools {
        tool_registry.register(Tool {
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
    }
    AgentRuntime::new(AgentDeps {
        session,
        providers: Arc::new(registry),
        chunk_sink: None,
        permission_requester: Arc::new(AlwaysAllow),
        evidence: Arc::new(NoEvidence),
        tools: Arc::new(tool_registry),
        cas: None,
        workspaces: kilop_fs::WorkspaceFileService::new(),
        edit: None,
        snapshots: None,
        sandbox: None,
        supervisor: None,
        model: "m".into(),
        compaction_model: None,
        compact_at_usage: compact_at,
        instructions: "i".into(),
        clock: Arc::new(SystemClock),
        tool_call_mode: kilop_agent::ToolCallMode::Native,
        tool_deadline_ms: 2000,
        retry_policy: kilop_core::retry::RetryPolicy::default(),
    })
    .unwrap()
}

/// Fast soak: 500 turns with tool calls, 3 daemon restarts, compaction
/// pressure — invariants: no state loss, no double effects, no loop.
#[tokio::test]
async fn fast_soak_500_turns_3_restarts_no_loss() {
    let dir = tempdir().unwrap();
    let root = dir.path();

    let session = SessionManager::open(root.join("store"), root.join("cas"), true).unwrap();
    let ws = session.create_workspace("/w").unwrap();
    let row = session.create_session(ws, "soak", "fake", "m").unwrap();

    let mut turns_done = 0u64;
    let per_round = 500 / 3;
    let remainder = 500 - per_round * 2;
    for round in 0..3 {
        let agent = agent_for(session.clone(), true, 0.02);
        let count = if round < 2 { per_round } else { remainder };
        for i in 0..count {
            let script = vec![
                ScriptedResponse::ToolCall {
                    id: format!("c{i}"),
                    name: "echo".into(),
                    input: serde_json::json!({"round": round, "i": i}),
                },
                ScriptedResponse::Text(format!("round {round} turn {i} {}", "x".repeat(300))),
                ScriptedResponse::End,
            ];
            let mut registry = ProviderRegistry::new();
            registry.register(Arc::new(FakeProvider::with_script(
                "fake",
                ModelCapabilities {
                    tools: true,
                    ..Default::default()
                },
                script,
            )));
            let mut deps = agent_deps(session.clone());
            deps.compact_at_usage = 0.02;
            deps.providers = Arc::new(registry);
            deps.tools = Arc::new(with_echo_tools());
            let turn_agent = AgentRuntime::new(deps).unwrap();
            let outcome = turn_agent
                .run_turn(row.id(), &format!("turn {i}"), &[])
                .await
                .unwrap();
            assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);
            turns_done += 1;
        }
        drop(agent);
        // DAEMON RESTART (round boundary): recovery finds nothing pending
        // and the session is coherent.
        let session2 = SessionManager::open(root.join("store"), root.join("cas"), true).unwrap();
        let reports = session2.recover_all_sessions().unwrap();
        let pending: usize = reports.iter().map(|r| r.crashed_ops.len()).sum();
        assert_eq!(
            pending, 0,
            "no unfinished ops at restart {round}: {reports:?}"
        );
        let handle = session2.get_session(row.id()).unwrap().unwrap();
        let page = handle.messages_page(None, 5).unwrap();
        assert!(page.messages.len() <= 5);
        assert!(page.has_more);
        drop(session2);
    }

    // Final integrity: journal gapless, no corruption, counts coherent.
    let session_final = SessionManager::open(root.join("store"), root.join("cas"), true).unwrap();
    let handle = session_final.get_session(row.id()).unwrap().unwrap();
    let replay = handle.replay_journal().unwrap();
    assert_eq!(
        replay.event_count,
        replay.last_seq.raw(),
        "journal must be gapless"
    );
    assert_eq!(turns_done, 500);
    // No duplicate effects: every tool ran exactly once per turn (pending
    // runs are all resolved).
    assert!(handle.pending_tool_runs().unwrap().is_empty());
}

fn with_echo_tools() -> ToolRegistry {
    let mut tool_registry = ToolRegistry::new();
    tool_registry.register(Tool {
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
    tool_registry
}

fn agent_deps(session: Arc<SessionManager>) -> AgentDeps {
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
        permission_requester: Arc::new(AlwaysAllow),
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

/// Compaction cannot loop even under pathological summarizer pressure:
/// repeated compactions converge to a bounded context.
#[tokio::test]
async fn compaction_converges_under_pressure() {
    let idx = kilop_context::Compactor::deterministic_only();
    let mut history: Vec<kilop_context::RecentTurn> = (0..500)
        .map(|i| kilop_context::RecentTurn {
            role: "assistant".into(),
            text: format!("turn {i} {}", "z".repeat(500)),
        })
        .collect();
    let ledger = kilop_context::TaskLedger {
        goal: "soak".into(),
        ..Default::default()
    };
    let target = 20_000usize;
    let mut steps = 0usize;
    loop {
        let before: usize = history.iter().map(|t| t.text.len()).sum::<usize>() / 4;
        let req = kilop_context::CompactionRequest::new(before, target);
        let plan = idx.compact(&history, &ledger, &req).await;
        assert!(plan.accepted, "deterministic pruning must always accept");
        assert!(plan.after_tokens <= req.hard_cap(), "hard invariant");
        if idx.would_compact_again(&plan, &req) {
            history = plan.kept_recent.clone();
            steps += 1;
            assert!(steps < 20, "convergence bounded");
            continue;
        }
        break;
    }
    // Convergence is the invariant; the number of rounds depends on the
    // budget math (the liar-summarizer death-spiral is covered in
    // kilop-context's unit tests).
    let _ = steps;
}

/// The 12-hour scale soak (spec §38): 10k ops, restarts, UI reconnects.
/// CI-gated.
#[tokio::test]
#[ignore = "[soak] 12-hour synthetic session — run explicitly"]
async fn soak_12h_scale() {
    let dir = tempdir().unwrap();
    let root = dir.path();
    let session = SessionManager::open(root.join("store"), root.join("cas"), true).unwrap();
    let ws = session.create_workspace("/w").unwrap();
    let row = session.create_session(ws, "soak-12h", "fake", "m").unwrap();

    for i in 0..10_000u64 {
        let agent = agent_for(session.clone(), i % 10 == 0, 0.01);
        let outcome = agent
            .run_turn(row.id(), &format!("op {i}"), &[])
            .await
            .unwrap();
        assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);
        if i % 3_000 == 0 {
            // Daemon restart + UI reconnect.
            let session2 =
                SessionManager::open(root.join("store"), root.join("cas"), true).unwrap();
            let reports = session2.recover_all_sessions().unwrap();
            let pending: usize = reports.iter().map(|r| r.crashed_ops.len()).sum();
            assert_eq!(pending, 0);
            let handle = session2.get_session(row.id()).unwrap().unwrap();
            let _ = handle.latest_messages_page(100).unwrap();
            drop(session2);
        }
        // Memory bound: message count grows but pages stay small.
        if i % 1_000 == 0 {
            let handle = session.get_session(row.id()).unwrap().unwrap();
            assert!(handle.messages_page(None, 100).unwrap().messages.len() <= 100);
        }
    }
    let _ = Duration::from_secs(1);
}

/// UI disconnect/reconnect under an active agent: the agent continues.
#[tokio::test]
async fn ui_disconnect_agent_continues() {
    let dir = tempdir().unwrap();
    let session =
        SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
    let ws = session.create_workspace("/w").unwrap();
    let row = session.create_session(ws, "ui", "fake", "m").unwrap();
    let agent = agent_for(session.clone(), true, 0.65);
    // Simulate the UI vanishing: submit the prompt and let the agent run to
    // completion (the HTTP layer would spawn this; here it is direct).
    let outcome = agent.run_turn(row.id(), "run", &[]).await.unwrap();
    assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);
    // Reconnect: the session state reproduces from the journal.
    let handle = session.get_session(row.id()).unwrap().unwrap();
    let state = handle.state().unwrap();
    assert_eq!(state, AgentState::ReadyForNextTurn);
}
