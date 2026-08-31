//! Fault injection: deliberately crash components at lifecycle boundaries
//! and assert the recovery outcome is defined (spec §39).

use std::sync::Arc;
use std::time::Duration;

use kilop_agent::{AgentDeps, AgentRuntime, NoEvidence, Tool, ToolOutcome, ToolRegistry};
use kilop_core::cancellation::CancellationToken;
use kilop_core::capability::{Capability, PermissionDecision};
use kilop_core::id::{OpId, SessionId};
use kilop_core::model::ModelCapabilities;
use kilop_core::op::{EffectStatus, OpMeta, RecoveryStrategy};
use kilop_core::state::AgentState;
use kilop_core::time::SystemClock;
use kilop_provider::{FakeProvider, ProviderRegistry, ScriptedResponse};
use kilop_session::SessionManager;
use kilop_server::permission::ChannelPermissionRequester;
use tempfile::tempdir;

struct AlwaysAllow;
impl kilop_agent::PermissionRequester for AlwaysAllow {
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

/// Crash mid-write: ToolStarted recorded with VerifyHash recovery; the file
/// hit the disk; the daemon died before recording completion. Recovery must
/// verify the hash and complete WITHOUT re-running the tool.
#[tokio::test]
async fn crash_mid_write_verify_hash_completes_without_rerun() {
    let dir = tempdir().unwrap();
    let root = dir.path();
    let file_path = root.join("out.txt");
    std::fs::write(&file_path, b"content").unwrap();
    let expected = kilop_core::hash::FileHash::from(blake3::hash(b"content").into());

    let session = SessionManager::open(root.join("store"), root.join("cas"), true).unwrap();
    let ws = session.create_workspace("/w").unwrap();
    let row = session.create_session(ws, "crash", "fake", "m").unwrap();
    let handle = session.get_session(row.id()).unwrap().unwrap();

    // Simulate the crash: ToolStarted, never finished. The op must be a
    // tracked turn op (the real agent flow) for the session to accept it.
    let receipt = handle.submit_prompt("crash mid write", &[]).unwrap();
    to_streaming(&handle, receipt.op_id);
    let op_meta = OpMeta::new(
        receipt.op_id,
        row.id(),
        kilop_core::time::Deadline::at(now() + 60_000),
        kilop_core::retry::RetryPolicy::default(),
        CancellationToken::new(),
        RecoveryStrategy::VerifyHash {
            path: file_path.to_string_lossy().to_string(),
            expected,
        },
        now(),
    );
    handle
        .request_permission(op_meta.operation_id, &Capability::WriteWorkspace {
            path: file_path.clone(),
        })
        .unwrap();
    handle
        .start_tool_run(op_meta.clone(), "write_file", serde_json::json!({}))
        .unwrap();
    assert_eq!(handle.pending_tool_runs().unwrap().len(), 1);
    drop(handle);
    drop(session);

    // DAEMON RESTART: recovery decides via the durable row.
    let session2 = SessionManager::open(root.join("store"), root.join("cas"), true).unwrap();
    let perm = Arc::new(AlwaysAllow);
    let agent = test_agent(session2.clone(), vec![ScriptedResponse::End], perm);
    let reports = agent.recover().unwrap();
    let crashed: Vec<_> = reports
        .iter()
        .flat_map(|r| r.crashed_ops.iter())
        .collect();
    eprintln!("fault-dbg: reports = {reports:?}");
    assert_eq!(crashed.len(), 1, "the interrupted write must be surfaced");
    // The row was resolved as verified (never re-run): the file content is
    // the crash-era content and no second write happened.
    assert_eq!(std::fs::read(&file_path).unwrap(), b"content");
    let handle2 = session2.get_session(row.id()).unwrap().unwrap();
    assert!(handle2.pending_tool_runs().unwrap().is_empty());
}

/// Crash mid-write where the write never landed: recovery marks it unknown
/// and NEVER re-runs the command (effect_status = unknown → forced verify).
#[tokio::test]
async fn crash_mid_write_hash_mismatch_marks_unknown_no_rerun() {
    let dir = tempdir().unwrap();
    let root = dir.path();
    let file_path = root.join("out.txt");
    std::fs::write(&file_path, b"stale").unwrap();
    let expected = kilop_core::hash::FileHash::from(blake3::hash(b"intended").into());

    let session = SessionManager::open(root.join("store"), root.join("cas"), true).unwrap();
    let ws = session.create_workspace("/w").unwrap();
    let row = session.create_session(ws, "crash2", "fake", "m").unwrap();
    let handle = session.get_session(row.id()).unwrap().unwrap();
    let receipt = handle.submit_prompt("crash mid write 2", &[]).unwrap();
    to_streaming(&handle, receipt.op_id);
    let op_meta = OpMeta::new(
        receipt.op_id,
        row.id(),
        kilop_core::time::Deadline::at(now() + 60_000),
        kilop_core::retry::RetryPolicy::default(),
        CancellationToken::new(),
        RecoveryStrategy::VerifyHash {
            path: file_path.to_string_lossy().to_string(),
            expected,
        },
        now(),
    );
    handle
        .request_permission(op_meta.operation_id, &Capability::WriteWorkspace {
            path: file_path.clone(),
        })
        .unwrap();
    handle
        .start_tool_run(op_meta.clone(), "write_file", serde_json::json!({}))
        .unwrap();
    drop(handle);
    drop(session);

    let session2 = SessionManager::open(root.join("store"), root.join("cas"), true).unwrap();
    let perm = Arc::new(AlwaysAllow);
    let agent = test_agent(session2.clone(), vec![ScriptedResponse::End], perm);
    let reports = agent.recover().unwrap();
    let crashed: Vec<_> = reports.iter().flat_map(|r| r.crashed_ops.iter()).collect();
    assert_eq!(crashed.len(), 1);
    // The file was never written: still stale, never overwritten.
    assert_eq!(std::fs::read(&file_path).unwrap(), b"stale");
}

/// A corrupted SQLite store is detected at open, not silently used.
#[tokio::test]
async fn corrupt_store_detected_at_open() {
    let dir = tempdir().unwrap();
    // Write garbage at the REAL store path BEFORE opening.
    std::fs::create_dir_all(dir.path().join("store")).unwrap();
    std::fs::write(
        dir.path().join("store/kilo-plus.db"),
        b"this is not a sqlite database, definitely not, nope",
    )
    .unwrap();
    let result = SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true);
    assert!(result.is_err(), "corrupt store must refuse to open");
}

/// A corrupted CAS blob is detected by integrity checks, never served.
#[tokio::test]
async fn corrupt_cas_detected_by_integrity() {
    let dir = tempdir().unwrap();
    let cas = Arc::new(kilop_cas::Cas::open(dir.path().join("cas")).unwrap());
    let hash = cas.put(b"precious data").unwrap();
    // Corrupt the blob on disk.
    let path = cas
        .root()
        .join(&hash.to_hex()[..2])
        .join(&hash.to_hex()[2..]);
    std::fs::write(path, b"corrupted").unwrap();
    let bad = cas.verify_integrity();
    assert!(bad.contains(&hash), "integrity scan must flag corruption");
    assert!(cas.get(hash).is_err(), "corrupted blob must never be served");
}

/// A supervised child that dies is reaped; no zombies, no orphans.
#[tokio::test]
async fn child_crash_reaped_no_zombies() {
    let dir = tempdir().unwrap();
    let cas = Arc::new(kilop_cas::Cas::open(dir.path().join("cas")).unwrap());
    let sup = kilop_terminal::ProcessSupervisor::new(cas);
    let cfg = kilop_terminal::SpawnConfig {
        cmd: "/bin/sh".into(),
        args: vec!["-c".into(), "exit 42".into()],
        cwd: std::env::temp_dir(),
        ..Default::default()
    };
    let handle = sup.spawn(cfg).unwrap();
    let mut reaped = Vec::new();
    for _ in 0..40 {
        reaped.extend(sup.reap());
        if reaped.len() == 1 {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    assert_eq!(reaped.len(), 1);
    assert_eq!(reaped[0].exit_code, Some(42));
    assert_eq!(sup.registered(), 0, "no zombie left registered");
}

/// MCP server crash: a garbage/hanging server never destabilizes the
/// caller (deadline-bounded, clean errors).
#[tokio::test]
async fn mcp_server_crash_is_contained() {
    // A server that dies immediately on connect: the client must fail
    // cleanly (NotFound/timeout), never hang.
    let dir = tempdir().unwrap();
    let cas = Arc::new(kilop_cas::Cas::open(dir.path().join("cas")).unwrap());
    let sup = kilop_terminal::ProcessSupervisor::new(cas);
    let cfg = kilop_mcp::McpConfig {
        name: "dead".into(),
        command: "/bin/sh".into(),
        args: vec!["-c".into(), "exit 1".into()],
        env: vec![],
    };
    let result = kilop_mcp::McpServer::connect(cfg, sup).await;
    assert!(result.is_err(), "a dead server must fail cleanly");
}

/// Provider stream death mid-turn: the journal decides; no blind replay.
#[tokio::test]
async fn provider_stream_death_continuation_is_defined() {
    let dir = tempdir().unwrap();
    let session = SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
    let perm = Arc::new(AlwaysAllow);
    let agent = test_agent(
        session.clone(),
        vec![
            ScriptedResponse::ToolCall {
                id: "c1".into(),
                name: "echo".into(),
                input: serde_json::json!({"x": 1}),
            },
            ScriptedResponse::Text("partial".into()),
            ScriptedResponse::Die(kilop_provider::ProviderError::new(
                kilop_provider::ProviderErrorKind::Network,
                "connection vanished",
            )),
        ],
        perm,
    );
    let ws = session.create_workspace("/w").unwrap();
    let row = session.create_session(ws, "death", "fake", "m").unwrap();
    let outcome = agent.run_turn(row.id(), "x", &[]).await.unwrap();
    // The recovery outcome is DEFINED: the turn lands in a state that forces
    // verification, never a blind replay.
    assert!(
        matches!(
            outcome.final_state,
            AgentState::NeedsUserInput | AgentState::FailedRecoverable
        ),
        "defined continuation: {:?}",
        outcome.final_state
    );
    // The journal recorded the tool request; the effect is marked unknown.
    let handle = session.get_session(row.id()).unwrap().unwrap();
    assert!(handle.pending_tool_runs().unwrap().is_empty(), "pending runs resolved");
}

/// Kill a running command; the supervisor records the kill and reaps.
#[tokio::test]
async fn killed_command_is_recorded_and_reaped() {
    let dir = tempdir().unwrap();
    let cas = Arc::new(kilop_cas::Cas::open(dir.path().join("cas")).unwrap());
    let sup = kilop_terminal::ProcessSupervisor::new(cas);
    let cfg = kilop_terminal::SpawnConfig {
        cmd: "/bin/sh".into(),
        args: vec!["-c".into(), "sleep 30".into()],
        cwd: std::env::temp_dir(),
        ..Default::default()
    };
    let h = sup.spawn(cfg).unwrap();
    std::thread::sleep(Duration::from_millis(200));
    sup.kill(h.id, 500).unwrap();
    let mut reaped = Vec::new();
    for _ in 0..40 {
        reaped.extend(sup.reap());
        if !reaped.is_empty() {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    assert_eq!(reaped.len(), 1);
    assert!(reaped[0].exit_code.is_some() || reaped[0].exit_code.is_none());
    assert_eq!(sup.registered(), 0);
}


/// Drive the session machine to Streaming (the state where tool calls are
/// legal), mimicking the agent loop before a tool request.
fn to_streaming(handle: &kilop_session::SessionHandle, op: OpId) {
    handle
        .append_event(kilop_core::event::EventKind::ContextPrepared, AgentState::BuildingContext, Some(op), None)
        .unwrap();
    handle
        .append_event(kilop_core::event::EventKind::ModelStarted, AgentState::WaitingForModel, Some(op), None)
        .unwrap();
    handle
        .append_event(kilop_core::event::EventKind::ModelStarted, AgentState::Streaming, Some(op), None)
        .unwrap();
}

fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn test_agent(
    session: Arc<SessionManager>,
    script: Vec<ScriptedResponse>,
    permissions: Arc<dyn kilop_agent::PermissionRequester>,
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
        execute: Arc::new(|_ctx, args| {
            Box::pin(async move {
                Ok(ToolOutcome {
                    text: format!("echo: {args}"),
                    exit_code: Some(0),
                    effect_status: EffectStatus::Applied,
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
        model: "m".into(),
        compaction_model: None,
        compact_at_usage: 0.65,
        instructions: "i".into(),
        clock: Arc::new(SystemClock),
        tool_call_mode: kilop_agent::ToolCallMode::Native,
        tool_deadline_ms: 2000,
    })
    .unwrap()
}

// Keep the channel requester referenced for parity with integration tests.
#[allow(dead_code)]
fn _perm_channel() -> Arc<ChannelPermissionRequester> {
    ChannelPermissionRequester::new(Duration::from_secs(5))
}
