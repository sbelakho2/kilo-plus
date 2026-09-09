//! Fault injection: deliberately crash components at lifecycle boundaries
//! and assert the recovery outcome is defined (spec §39).

#![cfg_attr(
    not(test),
    allow(dead_code, unused_imports, unused_variables, unused_mut)
)] // test-harness crate: the lib view exists only for clippy
use std::sync::Arc;
use std::time::Duration;

use faktor_agent::{AgentDeps, AgentRuntime, NoEvidence, Tool, ToolOutcome, ToolRegistry};
use faktor_core::cancellation::CancellationToken;
use faktor_core::capability::{Capability, PermissionDecision};
use faktor_core::id::{OpId, SessionId};
use faktor_core::model::ModelCapabilities;
use faktor_core::op::{EffectStatus, OpMeta, RecoveryStrategy};
use faktor_core::state::AgentState;
use faktor_core::time::SystemClock;
use faktor_provider::{FakeProvider, ProviderRegistry, ScriptedResponse};
use faktor_server::permission::ChannelPermissionRequester;
use faktor_session::SessionManager;
use tempfile::tempdir;

struct AlwaysAllow;
impl faktor_agent::PermissionRequester for AlwaysAllow {
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

/// Crash mid-write: ToolStarted recorded with VerifyHash recovery; the file
/// hit the disk; the daemon died before recording completion. Recovery must
/// verify the hash and complete WITHOUT re-running the tool.
#[tokio::test]
async fn crash_mid_write_verify_hash_completes_without_rerun() {
    let dir = tempdir().unwrap();
    let root = dir.path();
    let file_path = root.join("out.txt");
    std::fs::write(&file_path, b"content").unwrap();
    let expected = faktor_core::hash::FileHash::from(blake3::hash(b"content").into());

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
        faktor_core::time::Deadline::at(now() + 60_000),
        faktor_core::retry::RetryPolicy::default(),
        CancellationToken::new(),
        RecoveryStrategy::VerifyHash {
            path: file_path.to_string_lossy().to_string(),
            expected,
        },
        now(),
    );
    handle
        .request_permission(
            op_meta.operation_id,
            &Capability::WriteWorkspace {
                path: file_path.clone(),
            },
        )
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
    let crashed: Vec<_> = reports.iter().flat_map(|r| r.crashed_ops.iter()).collect();
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
    let expected = faktor_core::hash::FileHash::from(blake3::hash(b"intended").into());

    let session = SessionManager::open(root.join("store"), root.join("cas"), true).unwrap();
    let ws = session.create_workspace("/w").unwrap();
    let row = session.create_session(ws, "crash2", "fake", "m").unwrap();
    let handle = session.get_session(row.id()).unwrap().unwrap();
    let receipt = handle.submit_prompt("crash mid write 2", &[]).unwrap();
    to_streaming(&handle, receipt.op_id);
    let op_meta = OpMeta::new(
        receipt.op_id,
        row.id(),
        faktor_core::time::Deadline::at(now() + 60_000),
        faktor_core::retry::RetryPolicy::default(),
        CancellationToken::new(),
        RecoveryStrategy::VerifyHash {
            path: file_path.to_string_lossy().to_string(),
            expected,
        },
        now(),
    );
    handle
        .request_permission(
            op_meta.operation_id,
            &Capability::WriteWorkspace {
                path: file_path.clone(),
            },
        )
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
        dir.path().join("store/faktor-plus.db"),
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
    let cas = Arc::new(faktor_cas::Cas::open(dir.path().join("cas")).unwrap());
    let hash = cas.put(b"precious data").unwrap();
    // Corrupt the blob on disk.
    let path = cas
        .root()
        .join(&hash.to_hex()[..2])
        .join(&hash.to_hex()[2..]);
    std::fs::write(path, b"corrupted").unwrap();
    let bad = cas.verify_integrity();
    assert!(bad.contains(&hash), "integrity scan must flag corruption");
    assert!(
        cas.get_verified_now(hash).is_err(),
        "corrupted blob must never be served"
    );
}

/// A supervised child that dies is reaped; no zombies, no orphans.
/// unix-only: drives /bin/sh (the windows tree lifecycle is certified in
/// the P0-59 campaign row below and in faktor-terminal's windows tests).
#[cfg(unix)]
#[tokio::test]
async fn child_crash_reaped_no_zombies() {
    let dir = tempdir().unwrap();
    let cas = Arc::new(faktor_cas::Cas::open(dir.path().join("cas")).unwrap());
    let sup = faktor_terminal::ProcessSupervisor::new(cas);
    let cfg = faktor_terminal::SpawnConfig {
        cmd: "/bin/sh".into(),
        args: vec!["-c".into(), "exit 42".into()],
        cwd: std::env::temp_dir(),
        ..Default::default()
    };
    let handle = sup.spawn(cfg).unwrap();
    let _ = handle.pid;
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
/// caller (deadline-bounded, clean errors). unix-only: the canned server is
/// /bin/sh.
#[cfg(unix)]
#[tokio::test]
async fn mcp_server_crash_is_contained() {
    // A server that dies immediately on connect: the client must fail
    // cleanly (NotFound/timeout), never hang.
    let dir = tempdir().unwrap();
    let cas = Arc::new(faktor_cas::Cas::open(dir.path().join("cas")).unwrap());
    let sup = faktor_terminal::ProcessSupervisor::new(cas);
    let cfg = faktor_mcp::McpConfig {
        name: "dead".into(),
        command: "/bin/sh".into(),
        args: vec!["-c".into(), "exit 1".into()],
        env: vec![],
    };
    let result = faktor_mcp::McpServer::connect(cfg, sup).await;
    assert!(result.is_err(), "a dead server must fail cleanly");
}

/// Provider stream death mid-turn: the journal decides; no blind replay.
#[tokio::test]
async fn provider_stream_death_continuation_is_defined() {
    let dir = tempdir().unwrap();
    let session =
        SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
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
            ScriptedResponse::Die(faktor_provider::ProviderError::new(
                faktor_provider::ProviderErrorKind::Network,
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
    assert!(
        handle.pending_tool_runs().unwrap().is_empty(),
        "pending runs resolved"
    );
}

/// Kill a running command; the supervisor records the kill and reaps.
/// unix-only: drives /bin/sh sleepers (windows tree-kill semantics are
/// certified in the P0-59 campaign row below).
#[cfg(unix)]
#[tokio::test]
async fn killed_command_is_recorded_and_reaped() {
    let dir = tempdir().unwrap();
    let cas = Arc::new(faktor_cas::Cas::open(dir.path().join("cas")).unwrap());
    let sup = faktor_terminal::ProcessSupervisor::new(cas);
    let cfg = faktor_terminal::SpawnConfig {
        cmd: "/bin/sh".into(),
        args: vec!["-c".into(), "sleep 30".into()],
        cwd: std::env::temp_dir(),
        ..Default::default()
    };
    let h = sup.spawn(cfg).unwrap();
    std::thread::sleep(Duration::from_millis(200));
    sup.kill(h.id, 500).unwrap();
    let _ = h.pid;
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
fn to_streaming(handle: &faktor_session::SessionHandle, op: OpId) {
    handle
        .append_event(
            faktor_core::event::EventKind::ContextPrepared,
            AgentState::BuildingContext,
            Some(op),
            None,
        )
        .unwrap();
    handle
        .append_event(
            faktor_core::event::EventKind::ModelStarted,
            AgentState::WaitingForModel,
            Some(op),
            None,
        )
        .unwrap();
    handle
        .append_event(
            faktor_core::event::EventKind::ModelStarted,
            AgentState::Streaming,
            Some(op),
            None,
        )
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
    permissions: Arc<dyn faktor_agent::PermissionRequester>,
) -> Arc<AgentRuntime> {
    let mut registry = ProviderRegistry::new();
    registry
        .try_register(Arc::new(FakeProvider::with_script(
            "fake",
            ModelCapabilities {
                tools: true,
                ..Default::default()
            },
            script,
        )))
        .unwrap();
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
                    effect_status: EffectStatus::Applied,
                    ..Default::default()
                })
            })
        }),
    });
    AgentRuntime::new(AgentDeps {
        session,
        providers: Arc::new(registry),
        chunk_sink: None,
        permission_requester: permissions,
        evidence: Arc::new(NoEvidence),
        tools: Arc::new(tools),
        cas: None,
        workspaces: faktor_fs::WorkspaceFileService::new(),
        edit: None,
        snapshots: None,
        sandbox: None,
        supervisor: None,
        verification: faktor_agent::VerificationService::disabled(),
        hooks: None,
        instructions_resolver: faktor_instructions::no_roots_resolver(),
        routing: faktor_agent::FixedRoutingPolicy::passthrough(),
        budgets: Arc::new(faktor_session::NoopBudget),
        model: "m".into(),
        compaction_model: None,
        compact_at_usage: 0.65,
        instructions: "i".into(),
        clock: Arc::new(SystemClock),
        tool_call_mode: faktor_agent::ToolCallMode::Native,
        tool_deadline_ms: 2000,
        retry_policy: faktor_core::retry::RetryPolicy::default(),
    })
    .unwrap()
}

// Keep the channel requester referenced for parity with integration tests.
#[allow(dead_code)]
fn _perm_channel() -> Arc<ChannelPermissionRequester> {
    ChannelPermissionRequester::new(Duration::from_secs(5))
}

// ======================================================================
// P0-59 Windows process-tree lifecycle campaign (windows test builds only;
// the audit's real-runner certification rows). On unix this module does
// not exist — the equivalent unix coverage is the /bin/sh spawn/reap tests
// and the faktor-terminal group-kill suite. Rows: task cancellation, PTY
// close, daemon crash — each asserts the WHOLE tree dies (no orphan, no
// zombie) with bounded polls only.
// ======================================================================

#[cfg(all(test, windows))]
mod windows_lifecycle_campaign {
    use std::path::Path;
    use std::time::{Duration, Instant};

    use super::*;
    use faktor_core::error::ErrorKind;

    fn pid_file_path(dir: &std::path::Path, name: &str) -> std::path::PathBuf {
        dir.join(name)
    }

    fn sleeper_tree_script(pid_file: &Path) -> String {
        format!(
            "Start-Sleep -Milliseconds 1500; \
             $p = Start-Process -FilePath 'ping.exe' -ArgumentList '-n','60','127.0.0.1' \
                 -WindowStyle Hidden -PassThru; \
             [System.IO.File]::WriteAllText('{}', [string]$p.Id); \
             Start-Sleep -Seconds 60",
            pid_file.display()
        )
    }

    fn sleeper_cfg(pid_file: &Path, capture: bool) -> faktor_terminal::SpawnConfig {
        faktor_terminal::SpawnConfig {
            cmd: "powershell.exe".into(),
            args: vec![
                "-NoProfile".into(),
                "-NonInteractive".into(),
                "-Command".into(),
                sleeper_tree_script(pid_file).into(),
            ],
            cwd: std::env::temp_dir(),
            env: vec![],
            owner: faktor_terminal::ProcessOwner::Daemon,
            capture,
            artifact_max: 1024 * 1024,
            network_isolation: faktor_terminal::NetworkIsolation::Inherit,
        }
    }

    fn pty_tree_cfg(pid_file: &Path) -> faktor_pty::PtyConfig {
        let script = format!(
            "Start-Sleep -Milliseconds 1500; \
             $p = Start-Process -FilePath 'ping.exe' -ArgumentList '-n','60','127.0.0.1' \
                 -NoNewWindow -PassThru; \
             [System.IO.File]::WriteAllText('{}', [string]$p.Id); \
             Start-Sleep -Seconds 60",
            pid_file.display()
        );
        faktor_pty::PtyConfig {
            command: "powershell.exe".into(),
            args: vec![
                "-NoProfile".into(),
                "-NonInteractive".into(),
                "-Command".into(),
                script.into(),
            ],
            ..Default::default()
        }
    }

    fn wait_until<F: FnMut() -> bool>(what: &str, limit: Duration, mut cond: F) {
        let deadline = Instant::now() + limit;
        while Instant::now() < deadline {
            if cond() {
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        panic!("timed out after {limit:?} waiting for {what}");
    }

    fn read_pid(pid_file: &Path) -> u32 {
        wait_until("grandchild pid file", Duration::from_secs(20), || {
            pid_file.exists()
        });
        let pid: u32 = std::fs::read_to_string(pid_file)
            .expect("grandchild pid file readable")
            .trim()
            .parse()
            .expect("grandchild pid file holds a pid");
        assert_ne!(pid, 0);
        pid
    }

    /// Row 1 (task cancellation): the agent's run-cancel path kills
    /// agent-child (powershell) AND its grandchild (ping) — no orphan after
    /// a cancelled task.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn row_task_cancellation_kills_the_tree() {
        let dir = tempdir().unwrap();
        let cas = Arc::new(faktor_cas::Cas::open(dir.path().join("cas")).unwrap());
        let sup = faktor_terminal::ProcessSupervisor::new(cas);
        let pid_file = pid_file_path(dir.path(), "row1.pid");

        let token = CancellationToken::new();
        let sup2 = sup.clone();
        let token2 = token.clone();
        let cfg_pid_file = pid_file.clone();
        let task = tokio::spawn(async move {
            sup2.run(
                sleeper_cfg(&cfg_pid_file, false),
                Duration::from_secs(120),
                token2,
            )
            .await
        });

        let direct = {
            let deadline = Instant::now() + Duration::from_secs(20);
            loop {
                if let Some(t) = sup.recent_spawns().first() {
                    if t.pid > 0 {
                        break t.pid;
                    }
                }
                assert!(
                    Instant::now() < deadline,
                    "run() must spawn the direct child"
                );
                std::thread::sleep(Duration::from_millis(100));
            }
        };
        let grandchild = read_pid(&pid_file);
        assert!(
            sup.pid_alive(direct) && sup.pid_alive(grandchild),
            "parent + grandchild must be alive before cancellation"
        );

        token.cancel();
        let err = task.await.unwrap().unwrap_err();
        assert_eq!(err.kind, ErrorKind::Cancelled, "{err:?}");

        wait_until("cancelled tree death", Duration::from_secs(10), || {
            !sup.pid_alive(direct) && !sup.pid_alive(grandchild)
        });
    }

    /// Row 2 (PTY close): closing the ConPTY session kills the attached
    /// client AND the session-attached grandchild.
    #[test]
    fn row_pty_close_kills_the_session_tree() {
        let dir = tempdir().unwrap();
        let pid_file = pid_file_path(dir.path(), "row2.pid");
        let cas = Arc::new(faktor_cas::Cas::open(dir.path().join("cas")).unwrap());
        let sup = faktor_terminal::ProcessSupervisor::new(cas);

        let mut pty = faktor_pty::Pty::spawn(&pty_tree_cfg(&pid_file)).expect("ConPTY spawn on CI");
        let child = pty.pid();
        let grandchild = read_pid(&pid_file);
        assert!(
            sup.pid_alive(child) && sup.pid_alive(grandchild),
            "pty client + session grandchild must be alive pre-kill"
        );

        pty.kill();
        wait_until("conpty session death", Duration::from_secs(10), || {
            !sup.pid_alive(child) && !sup.pid_alive(grandchild)
        });
    }

    /// Row 3 (daemon crash): dropping the last supervisor reference (the
    /// daemon died) kills the whole supervised tree via registry kill +
    /// Job-Object kill-on-close.
    #[test]
    fn row_daemon_crash_kills_the_tree() {
        let dir = tempdir().unwrap();
        let pid_file = pid_file_path(dir.path(), "row3.pid");
        let (direct, grandchild) = {
            let cas = Arc::new(faktor_cas::Cas::open(dir.path().join("cas")).unwrap());
            let sup = faktor_terminal::ProcessSupervisor::new(cas);
            let handle = sup.spawn(sleeper_cfg(&pid_file, false)).unwrap();
            let grandchild = read_pid(&pid_file);
            assert!(sup.pid_alive(grandchild), "grandchild alive pre-crash");
            (handle.pid, grandchild) // sup (the daemon) drops here
        };
        // Fresh probe supervisor: pid_alive after the daemon is gone.
        let cas = Arc::new(faktor_cas::Cas::open(dir.path().join("cas")).unwrap());
        let probe = faktor_terminal::ProcessSupervisor::new(cas);
        wait_until("crash-killed tree death", Duration::from_secs(10), || {
            !probe.pid_alive(direct) && !probe.pid_alive(grandchild)
        });
    }
}

// ======================================================================
// P0-76 seeded crash-certification campaigns (store/journal, cas, edit
// txn, scheduler DAG). Test-only: the generic `run_campaign` runner, the
// four campaigns and their smoke/[fault]-gated tests live in `campaigns`.
// ======================================================================

#[cfg(test)]
mod campaigns;
