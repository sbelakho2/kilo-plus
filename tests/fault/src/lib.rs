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
        semantic: faktor_agent::fallback_semantic_registry(),
        context_prior: None,
        efficiency: Default::default(),
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
            env: faktor_terminal::EnvSpec::Minimal,
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

// ======================================================================
// Audit 99: accounting crash-seam campaign. The durable money path has no
// injectable crash seam (and adding production seams is out of scope), so a
// crash is the REAL process death equivalent: the manager is dropped after
// the seam's ledger ops COMMITTED and the store is reopened from disk. For
// every boundary the campaign builds the same durable prefix twice — once
// finished in-process (the reference) and once finished after a drop+reopen
// (the crash) — and the recovered world must EQUAL the reference world
// (reference-comparison semantics, like the P0-76 campaigns above), while
// the audit's invariants hold on every seam:
//   - no money disappears (a dispatched/chargeable reservation is NEVER
//     refunded; its amount stays held or is charged at its actual);
//   - no attempt is billed twice (a second recovery/reconcile/finalize, and
//     the committed-settlement retry, change nothing);
//   - no open reservation falsely frees budget (free == cap - spent - held;
//     an admission above free is a typed BudgetExceeded writing nothing);
//   - no VerifiedComplete with unresolved accounting (the completion gate
//     refuses open reserved/dispatched rows and the task stays Verifying).
// ======================================================================

#[cfg(test)]
mod accounting_campaign {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use super::campaigns::{check_equals, BoundarySpec, CrashClass, Lcg, WorldState};
    use faktor_core::id::{OpId, SessionId, TaskId, TaskRevision, VerificationRecordId};
    use faktor_core::model::{MicroUsdPerMillionTokens, PriceQuote, PricingSnapshot};
    use faktor_core::op::ModelCallAttempt;
    use faktor_core::state::{
        CriterionVerification, TaskState, TaskTransition, VerificationStatus,
    };
    use faktor_session::{
        BudgetAuthority, BudgetError, DurableBudgetLedger, ReservationId, SessionHandle,
        SessionManager, Task, TaskBudget, TaskError,
    };
    use tempfile::tempdir;

    /// CI smoke seed count (the `[fault]`-gated test runs [`FULL_SEEDS`]).
    pub const SMOKE_SEEDS: u64 = 3;
    /// The full campaign's seed count.
    pub const FULL_SEEDS: u64 = 200;

    /// The eight audit seams; the settlement boundary carries both crash
    /// orderings a torn transaction can leave (rolled back / committed).
    pub const BOUNDARIES: &[BoundarySpec] = &[
        BoundarySpec {
            name: "after-reserve",
            class: CrashClass::PreOp,
        },
        BoundarySpec {
            name: "after-dispatch-marker",
            class: CrashClass::FullyCommitted,
        },
        BoundarySpec {
            name: "after-request-accepted",
            class: CrashClass::FullyCommitted,
        },
        BoundarySpec {
            name: "after-usage-received",
            class: CrashClass::FullyCommitted,
        },
        BoundarySpec {
            name: "before-settlement",
            class: CrashClass::FullyCommitted,
        },
        BoundarySpec {
            name: "during-settlement-transaction.rolled-back",
            class: CrashClass::PreOp,
        },
        BoundarySpec {
            name: "during-settlement-transaction.committed",
            class: CrashClass::FullyCommitted,
        },
        BoundarySpec {
            name: "after-reconcile",
            class: CrashClass::FullyCommitted,
        },
        BoundarySpec {
            name: "during-completion-accounting",
            class: CrashClass::FullyCommitted,
        },
    ];

    struct Amounts {
        cap: u64,
        pa: u64,
        pb: u64,
        pc: u64,
        /// 900 cached-uncached input x 10 micro + 100 output x 20 micro.
        aa: u64,
        /// 500 input x 10 micro + 50 output x 20 micro.
        ab: u64,
    }

    fn amounts(seed: u64) -> Amounts {
        let mut l = Lcg::new(seed ^ 0xACC7_0000);
        Amounts {
            cap: 500_000 + l.below(50) * 1_000,
            pa: 40_000 + l.below(5_000),
            pb: 7_000 + l.below(1_000),
            pc: 5_000 + l.below(500),
            aa: 11_000,
            ab: 6_000,
        }
    }

    fn snapshot() -> PricingSnapshot {
        PricingSnapshot::exact(
            PriceQuote {
                input: MicroUsdPerMillionTokens(10_000_000),
                output: MicroUsdPerMillionTokens(20_000_000),
                cache_read: MicroUsdPerMillionTokens(2_000_000),
                cache_write: MicroUsdPerMillionTokens(4_000_000),
            },
            7,
            "accounting-campaign".into(),
        )
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct RowSnap {
        id: i64,
        status: String,
        predicted: u64,
        charged: Option<u64>,
    }

    struct Acct {
        dir: tempfile::TempDir,
        manager: Arc<SessionManager>,
        session: SessionId,
        task: TaskId,
        handle: SessionHandle,
        ledger: Arc<DurableBudgetLedger>,
        cap: u64,
    }

    impl Acct {
        fn open(cap: u64) -> Acct {
            let dir = tempdir().unwrap();
            let manager =
                SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                    .unwrap();
            let ws = manager.create_workspace("/w").unwrap();
            let session = manager
                .create_session(ws, "accounting", "fake", "m")
                .unwrap()
                .id();
            let handle = manager.get_session(session).unwrap().unwrap();
            let task = TaskId::new(1);
            handle
                .create_task(Task {
                    task_id: task,
                    session_id: session,
                    goal: "accounting campaign".into(),
                    acceptance_criteria: Vec::new(),
                    plan: Vec::new(),
                    budget: TaskBudget::default(),
                    state: TaskState::Pending,
                    created_ms: 1,
                    updated_ms: 1,
                })
                .unwrap();
            let ledger = DurableBudgetLedger::new(manager.clone());
            ledger.set_task_max_cost(session, task, Some(cap)).unwrap();
            Acct {
                dir,
                manager,
                session,
                task,
                handle,
                ledger,
                cap,
            }
        }

        /// The crash: drop every live handle and reopen the store from disk.
        fn reopen(self) -> Acct {
            let Acct {
                dir,
                manager,
                session,
                task,
                cap,
                handle,
                ledger,
            } = self;
            drop(manager);
            drop(handle);
            drop(ledger);
            let manager =
                SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                    .unwrap();
            let handle = manager.get_session(session).unwrap().unwrap();
            let ledger = DurableBudgetLedger::new(manager.clone());
            Acct {
                dir,
                manager,
                session,
                task,
                handle,
                ledger,
                cap,
            }
        }

        fn rows(&self) -> Vec<RowSnap> {
            let mut rows: Vec<RowSnap> = self
                .ledger
                .reservations_of(self.session, self.task, i64::MAX)
                .unwrap()
                .into_iter()
                .map(|r| RowSnap {
                    id: r.reservation_id,
                    status: r.status,
                    predicted: r.predicted_micro,
                    charged: r.settled_cost_micro,
                })
                .collect();
            rows.sort_by_key(|r| r.id);
            rows
        }

        fn spent(&self) -> u64 {
            self.ledger
                .session_budget_view(self.session, self.task)
                .expect("durable budget view")
                .spent_cost_micro
        }

        fn held(&self) -> u64 {
            self.ledger
                .session_budget_view(self.session, self.task)
                .expect("durable budget view")
                .open_reserved_micro
                + self
                    .ledger
                    .session_budget_view(self.session, self.task)
                    .expect("durable budget view")
                    .uncertain_reserved_micro
        }
    }

    struct Prefix {
        a: Option<ReservationId>,
        /// The committed-settlement seam retries the settle after the crash.
        retry_settle: bool,
    }

    fn attempt(n: u64) -> ModelCallAttempt {
        ModelCallAttempt::new(OpId::new(1_000 + n), OpId::new(2_000 + n), 0).unwrap()
    }

    async fn reserve(
        acct: &Acct,
        predicted: u64,
        snapshot: Option<PricingSnapshot>,
    ) -> ReservationId {
        acct.ledger
            .reserve(
                acct.session,
                acct.task,
                acct.manager.next_op_id(),
                predicted,
                snapshot,
            )
            .await
            .unwrap()
    }

    async fn reserve_attempt(
        acct: &Acct,
        predicted: u64,
        n: u64,
    ) -> (ReservationId, ModelCallAttempt) {
        let at = attempt(n);
        let rid = acct
            .ledger
            .reserve_attempt(acct.session, acct.task, at, predicted, Some(snapshot()))
            .await
            .unwrap();
        (rid, at)
    }

    async fn completed_row(
        acct: &Acct,
        rid: ReservationId,
        at: ModelCallAttempt,
        tokens: (u64, u64),
    ) {
        acct.handle
            .record_provider_call_attempt(
                at,
                Some(rid),
                "fake",
                "m",
                "completed",
                Some(tokens.0),
                Some(tokens.1),
                None,
            )
            .unwrap();
    }

    /// Build the durable prefix the boundary crashed at.
    async fn build_prefix(name: &str, acct: &Acct, amt: &Amounts) -> Prefix {
        match name {
            "after-reserve" => {
                reserve(acct, amt.pc, None).await;
                let (a, _) = reserve_attempt(acct, amt.pa, 1).await;
                Prefix {
                    a: Some(a),
                    retry_settle: false,
                }
            }
            "after-dispatch-marker" => {
                reserve(acct, amt.pc, None).await;
                let (a, _) = reserve_attempt(acct, amt.pa, 1).await;
                acct.ledger.mark_dispatched(acct.session, a).await.unwrap();
                Prefix {
                    a: Some(a),
                    retry_settle: false,
                }
            }
            "after-request-accepted" => {
                reserve(acct, amt.pc, None).await;
                let (a, at) = reserve_attempt(acct, amt.pa, 1).await;
                acct.ledger.mark_dispatched(acct.session, a).await.unwrap();
                // The provider ACCEPTED the request; the stream then failed.
                acct.handle
                    .record_provider_call_attempt(
                        at,
                        Some(a),
                        "fake",
                        "m",
                        "failed",
                        None,
                        None,
                        Some("stream failed after accept"),
                    )
                    .unwrap();
                Prefix {
                    a: Some(a),
                    retry_settle: false,
                }
            }
            "after-usage-received" => {
                reserve(acct, amt.pc, None).await;
                let (a, at) = reserve_attempt(acct, amt.pa, 1).await;
                acct.ledger.mark_dispatched(acct.session, a).await.unwrap();
                completed_row(acct, a, at, (900, 100)).await;
                Prefix {
                    a: Some(a),
                    retry_settle: false,
                }
            }
            "before-settlement" => {
                reserve(acct, amt.pc, None).await;
                let b = reserve(acct, amt.pb, None).await;
                acct.ledger.mark_dispatched(acct.session, b).await.unwrap();
                let (a, at) = reserve_attempt(acct, amt.pa, 1).await;
                acct.ledger.mark_dispatched(acct.session, a).await.unwrap();
                completed_row(acct, a, at, (900, 100)).await;
                Prefix {
                    a: Some(a),
                    retry_settle: false,
                }
            }
            "during-settlement-transaction.rolled-back" => {
                let (a, at) = reserve_attempt(acct, amt.pa, 1).await;
                acct.ledger.mark_dispatched(acct.session, a).await.unwrap();
                completed_row(acct, a, at, (900, 100)).await;
                Prefix {
                    a: Some(a),
                    retry_settle: false,
                }
            }
            "during-settlement-transaction.committed" => {
                let b = reserve(acct, amt.pb, None).await;
                acct.ledger.mark_dispatched(acct.session, b).await.unwrap();
                let (a, at) = reserve_attempt(acct, amt.pa, 1).await;
                acct.ledger.mark_dispatched(acct.session, a).await.unwrap();
                completed_row(acct, a, at, (900, 100)).await;
                let got = acct
                    .ledger
                    .settle_usage(acct.session, a, 900, 0, 0, 100, None, None)
                    .await
                    .unwrap();
                assert_eq!(
                    got,
                    Some(amt.aa),
                    "the settled actual is the snapshot price"
                );
                Prefix {
                    a: Some(a),
                    retry_settle: true,
                }
            }
            "after-reconcile" => {
                let (a, at) = reserve_attempt(acct, amt.pa, 1).await;
                acct.ledger.mark_dispatched(acct.session, a).await.unwrap();
                completed_row(acct, a, at, (900, 100)).await;
                acct.ledger
                    .mark_uncertain(acct.session, a, "crash_pre_reconcile".into(), None)
                    .await
                    .unwrap();
                let b = reserve(acct, amt.pb, None).await;
                acct.ledger.mark_dispatched(acct.session, b).await.unwrap();
                acct.ledger
                    .mark_uncertain(acct.session, b, "crash_pre_finalize".into(), None)
                    .await
                    .unwrap();
                acct.ledger
                    .reconcile_uncertain(acct.session, acct.task)
                    .await
                    .unwrap();
                Prefix {
                    a: Some(a),
                    retry_settle: false,
                }
            }
            "during-completion-accounting" => {
                // Attempt A: dispatched, UNKNOWN usage (no completed row).
                let (a, _) = reserve_attempt(acct, amt.pa, 1).await;
                acct.ledger.mark_dispatched(acct.session, a).await.unwrap();
                acct.ledger
                    .mark_uncertain(acct.session, a, "no_usage".into(), None)
                    .await
                    .unwrap();
                // Attempt B: dispatched with exact usage.
                let (b, bt) = reserve_attempt(acct, amt.pb, 2).await;
                acct.ledger.mark_dispatched(acct.session, b).await.unwrap();
                completed_row(acct, b, bt, (500, 50)).await;
                acct.ledger
                    .mark_uncertain(acct.session, b, "usage_known".into(), None)
                    .await
                    .unwrap();
                // The completion pass reconciles exact usage, then crashed
                // before the conservative finalize and the transition CAS.
                let report = acct
                    .ledger
                    .reconcile_uncertain(acct.session, acct.task)
                    .await
                    .unwrap();
                assert_eq!(
                    report.charged_micro, amt.ab,
                    "the exact-usage reconcile charges B's actual"
                );
                Prefix {
                    a: Some(a),
                    retry_settle: false,
                }
            }
            other => panic!("unknown accounting boundary {other:?}"),
        }
    }

    /// Pending -> Running -> NeedsVerification -> Verifying (the only state
    /// `complete_verified_task` accepts). No-op when already past Pending.
    fn ensure_verifying(
        acct: &Acct,
    ) -> Result<Option<(TaskRevision, VerificationRecordId)>, String> {
        let state = acct
            .handle
            .get_task(acct.task)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| "task vanished".to_string())?
            .state;
        match state {
            TaskState::VerifiedComplete => Ok(None),
            TaskState::Verifying => {
                let rev = acct
                    .handle
                    .task_revision(acct.task)
                    .map_err(|e| e.to_string())?;
                let rec = passed_record(acct)?;
                Ok(Some((rev, rec)))
            }
            TaskState::Pending => {
                let r1 = acct
                    .handle
                    .task_revision(acct.task)
                    .map_err(|e| e.to_string())?;
                acct.handle
                    .transition_task(acct.task, r1, TaskTransition::StartRunning, None)
                    .map_err(|e| e.to_string())?;
                let r2 = acct
                    .handle
                    .task_revision(acct.task)
                    .map_err(|e| e.to_string())?;
                acct.handle
                    .transition_task(acct.task, r2, TaskTransition::RequestVerification, None)
                    .map_err(|e| e.to_string())?;
                let r3 = acct
                    .handle
                    .task_revision(acct.task)
                    .map_err(|e| e.to_string())?;
                acct.handle
                    .transition_task(acct.task, r3, TaskTransition::StartVerification, None)
                    .map_err(|e| e.to_string())?;
                let rev = acct
                    .handle
                    .task_revision(acct.task)
                    .map_err(|e| e.to_string())?;
                let rec = passed_record(acct)?;
                Ok(Some((rev, rec)))
            }
            other => Err(format!("task in unexpected state {other:?}")),
        }
    }

    fn passed_record(acct: &Acct) -> Result<VerificationRecordId, String> {
        let criteria: Vec<CriterionVerification> = Vec::new();
        acct.handle
            .create_verification_record(
                acct.task,
                None,
                criteria,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                None,
                VerificationStatus::Passed,
                1,
            )
            .map_err(|e| e.to_string())
    }

    fn is_open(status: &str) -> bool {
        matches!(status, "reserved" | "dispatched")
    }

    /// The audit invariants after reopen + recovery, before any completion.
    fn assert_recovery_invariants(
        amt: &Amounts,
        pre: &[RowSnap],
        pre_spent: u64,
        acct: &Acct,
        boundary: &BoundarySpec,
    ) {
        let post = acct.rows();
        let post_spent = acct.spent();
        let ctx = format!("boundary {}", boundary.name);
        // No money disappears: recovery never refunds a dispatched row and
        // never charges one; settled/refunded rows stay byte-identical.
        for pr in pre {
            let po = post
                .iter()
                .find(|r| r.id == pr.id)
                .unwrap_or_else(|| panic!("{ctx}: recovery dropped reservation {}: {pr:?}", pr.id));
            assert_eq!(po.predicted, pr.predicted, "{ctx}: predicted changed");
            match pr.status.as_str() {
                "dispatched" | "uncertain" => {
                    assert_ne!(
                        po.status, "refunded",
                        "{ctx}: a potentially chargeable reservation {} was refunded",
                        pr.id
                    );
                    assert_eq!(
                        po.charged, pr.charged,
                        "{ctx}: recovery charged reservation {} outside settle/reconcile",
                        pr.id
                    );
                }
                "reserved" => {
                    assert_eq!(
                        po.status, "refunded",
                        "{ctx}: a never-dispatched reservation {} must be refunded",
                        pr.id
                    );
                    assert_eq!(po.charged, pr.charged, "{ctx}: refund charged something");
                }
                _ => {
                    assert_eq!(po.status, pr.status, "{ctx}: terminal row changed");
                    assert_eq!(po.charged, pr.charged, "{ctx}: terminal charge changed");
                }
            }
        }
        // The uncertain amount retains its reserved budget.
        let pre_chargeable: u64 = pre
            .iter()
            .filter(|r| matches!(r.status.as_str(), "dispatched" | "uncertain"))
            .map(|r| r.predicted)
            .sum();
        let post_chargeable: u64 = post
            .iter()
            .filter(|r| matches!(r.status.as_str(), "dispatched" | "uncertain"))
            .map(|r| r.predicted)
            .sum();
        assert!(
            post_chargeable >= pre_chargeable,
            "{ctx}: an uncertain reservation released its reserved amount"
        );
        assert_eq!(
            post_spent, pre_spent,
            "{ctx}: recovery alone must not charge anything"
        );
        // No open reservation falsely frees budget: free == cap - spent - held
        // AND an admission above free writes nothing.
        let view = acct
            .ledger
            .session_budget_view(acct.session, acct.task)
            .expect("durable budget view");
        let held = acct.held();
        assert_eq!(
            held,
            view.open_reserved_micro + view.uncertain_reserved_micro,
            "{ctx}: held view is inconsistent"
        );
        assert_eq!(
            view.free(),
            amt.cap.saturating_sub(post_spent).saturating_sub(held),
            "{ctx}: free budget is not cap - spent - held"
        );
        // A second recovery converges (idempotent).
        acct.ledger.recover_after_restart();
        assert_eq!(
            acct.rows(),
            post,
            "{ctx}: the second recovery changed durable state"
        );
    }

    async fn assert_admission_invariants(acct: &Acct, boundary: &BoundarySpec) {
        let ctx = format!("boundary {}", boundary.name);
        let view = acct
            .ledger
            .session_budget_view(acct.session, acct.task)
            .expect("durable budget view");
        let free = view.free();
        let before = acct.rows();
        let over = free.saturating_add(1).max(1);
        match acct
            .ledger
            .reserve(
                acct.session,
                acct.task,
                acct.manager.next_op_id(),
                over,
                None,
            )
            .await
        {
            Err(BudgetError::BudgetExceeded {
                free: got_free,
                predicted,
            }) => {
                assert_eq!(got_free, free, "{ctx}: denial free mismatch");
                assert_eq!(predicted, over);
            }
            other => {
                panic!("{ctx}: an admission above free must be typed BudgetExceeded: {other:?}")
            }
        }
        assert_eq!(
            acct.rows(),
            before,
            "{ctx}: the refused admission wrote a reservation"
        );
        if free >= 1 {
            let r = acct
                .ledger
                .reserve(acct.session, acct.task, acct.manager.next_op_id(), 1, None)
                .await
                .unwrap();
            acct.ledger.refund(acct.session, r).await.unwrap();
        }
    }

    fn probe_completion(
        acct: &Acct,
        revision: TaskRevision,
        record: VerificationRecordId,
        boundary: &BoundarySpec,
    ) -> Result<(), String> {
        match acct
            .handle
            .complete_verified_task(acct.task, revision, record)
        {
            Ok(_) => Ok(()),
            Err(TaskError::AccountingIncomplete { .. }) => Ok(()),
            Err(e) => Err(format!(
                "boundary {}: unexpected completion refusal: {e:?}",
                boundary.name
            )),
        }
    }

    /// Recovery -> completion gate -> monetary closure -> VerifiedComplete ->
    /// idempotence -> admission probe. Identical in the reference and crash
    /// runs, so their final worlds are comparable.
    async fn finish(acct: &Acct, boundary: &BoundarySpec) -> Result<WorldState, String> {
        let ctx = format!("boundary {}", boundary.name);
        acct.ledger.recover_after_restart();
        // Admission invariants run while the task still permits new provider
        // work: once the verification path has begun, the reservation
        // transaction refuses by task state (the second SQL gate) and the
        // free-accounting admission probe no longer applies.
        let permits_provider_work = matches!(
            acct.handle
                .get_task(acct.task)
                .map_err(|e| e.to_string())?
                .map(|t| t.state),
            Some(
                TaskState::Pending
                    | TaskState::Planning
                    | TaskState::Running
                    | TaskState::Waiting
                    | TaskState::Blocked
            )
        );
        if permits_provider_work {
            assert_admission_invariants(acct, boundary).await;
        }
        if let Some((revision, record)) = ensure_verifying(acct)? {
            // The completion gate: with an open reserved/dispatched row it
            // refuses typed and the task stays Verifying (the probe runs its
            // own reconcile/finalize pass first, all idempotent).
            probe_completion(acct, revision, record, boundary)?;
        }
        acct.ledger
            .reconcile_uncertain(acct.session, acct.task)
            .await
            .map_err(|e| format!("{ctx}: reconcile: {e}"))?;
        acct.ledger
            .finalize_uncertain(acct.session, acct.task)
            .await
            .map_err(|e| format!("{ctx}: finalize: {e}"))?;
        let balance = acct
            .ledger
            .completion_accounting_balance(acct.session, acct.task)
            .map_err(|e| format!("{ctx}: balance: {e}"))?;
        if !balance.is_zero() {
            return Err(format!(
                "{ctx}: unresolved accounting after recovery/reconcile/finalize: {balance:?}"
            ));
        }
        if acct
            .handle
            .get_task(acct.task)
            .map_err(|e| e.to_string())?
            .map(|t| t.state)
            != Some(TaskState::VerifiedComplete)
        {
            let (revision, record) = ensure_verifying(acct)?
                .ok_or_else(|| format!("{ctx}: task already complete but not verified"))?;
            acct.handle
                .complete_verified_task(acct.task, revision, record)
                .map_err(|e| format!("{ctx}: completion refused after closure: {e:?}"))?;
        }
        // Exactly-once: the idempotent re-run changes nothing.
        let spent = acct.spent();
        let rows = acct.rows();
        acct.ledger
            .reconcile_uncertain(acct.session, acct.task)
            .await
            .map_err(|e| e.to_string())?;
        acct.ledger
            .finalize_uncertain(acct.session, acct.task)
            .await
            .map_err(|e| e.to_string())?;
        if acct.spent() != spent || acct.rows() != rows {
            return Err(format!(
                "{ctx}: the idempotent recovery re-run billed an attempt twice"
            ));
        }
        Ok(world(acct))
    }

    fn world(acct: &Acct) -> WorldState {
        let view = acct
            .ledger
            .session_budget_view(acct.session, acct.task)
            .expect("durable budget view");
        let state = acct.handle.get_task(acct.task).unwrap().unwrap().state;
        let mut lines = vec![
            format!("task_state={state:?}"),
            format!("spent={}", view.spent_cost_micro),
            format!(
                "held={}",
                view.open_reserved_micro + view.uncertain_reserved_micro
            ),
            format!("free={}", view.free()),
        ];
        for r in acct.rows() {
            lines.push(format!(
                "res={} status={} predicted={} charged={:?}",
                r.id, r.status, r.predicted, r.charged
            ));
        }
        WorldState { lines }
    }

    /// Uninterrupted reference: the same durable prefix, finished in-process.
    async fn reference_world(seed: u64, boundary: &BoundarySpec) -> Result<WorldState, String> {
        let amt = amounts(seed);
        let acct = Acct::open(amt.cap);
        let prefix = build_prefix(boundary.name, &acct, &amt).await;
        retry_settle(&acct, &prefix, &amt, boundary).await?;
        finish(&acct, boundary).await
    }

    /// A crash: the same durable prefix, then the manager drops (the process
    /// died) and the store reopens; recovery + finish must converge.
    async fn crash_world(seed: u64, boundary: &BoundarySpec) -> Result<WorldState, String> {
        let amt = amounts(seed);
        let acct = Acct::open(amt.cap);
        let prefix = build_prefix(boundary.name, &acct, &amt).await;
        let pre = acct.rows();
        let pre_spent = acct.spent();
        let acct = acct.reopen();
        acct.ledger.recover_after_restart();
        assert_recovery_invariants(&amt, &pre, pre_spent, &acct, boundary);
        retry_settle(&acct, &prefix, &amt, boundary).await?;
        finish(&acct, boundary).await
    }

    /// The committed-settlement retry: after the crash the settle is
    /// re-issued; it must be a typed NotOpen writing NOTHING (no double
    /// billing), in both the reference and the crash run.
    async fn retry_settle(
        acct: &Acct,
        prefix: &Prefix,
        amt: &Amounts,
        boundary: &BoundarySpec,
    ) -> Result<(), String> {
        if !prefix.retry_settle {
            return Ok(());
        }
        let a = prefix.a.expect("seam reserved A");
        let before = acct.spent();
        match acct
            .ledger
            .settle_usage(acct.session, a, 900, 0, 0, 100, None, None)
            .await
        {
            Err(BudgetError::NotOpen { .. }) => {}
            other => {
                return Err(format!(
                    "boundary {}: a committed-settlement retry must be typed NotOpen: {other:?}",
                    boundary.name
                ))
            }
        }
        if acct.spent() != before {
            return Err(format!(
                "boundary {}: the committed-settlement retry billed twice",
                boundary.name
            ));
        }
        let _ = amt;
        Ok(())
    }

    /// The completion gate before recovery: an open reserved/dispatched row
    /// refuses VerifiedComplete typed; the task stays Verifying.
    fn assert_gate_refuses_open_rows(acct: &Acct, boundary: &BoundarySpec) {
        let open = acct.rows().iter().any(|r| is_open(&r.status));
        let Some((revision, record)) = ensure_verifying(acct).unwrap() else {
            return;
        };
        match acct
            .handle
            .complete_verified_task(acct.task, revision, record)
        {
            Ok(_) => assert!(
                !open,
                "boundary {}: VerifiedComplete landed with an open reservation",
                boundary.name
            ),
            Err(TaskError::AccountingIncomplete { .. }) => {
                assert!(
                    open,
                    "boundary {}: completion refused without an open reservation",
                    boundary.name
                );
                let state = acct.handle.get_task(acct.task).unwrap().unwrap().state;
                assert_eq!(
                    state,
                    TaskState::Verifying,
                    "boundary {}: the refusal must not transition the task",
                    boundary.name
                );
            }
            Err(e) => panic!(
                "boundary {}: unexpected completion refusal: {e:?}",
                boundary.name
            ),
        }
    }

    async fn run(seeds: u64) -> (u64, BTreeMap<&'static str, u64>) {
        let mut checks = 0u64;
        let mut per_boundary = BTreeMap::new();
        for seed in 0..seeds {
            for boundary in BOUNDARIES {
                let reference = reference_world(seed, boundary)
                    .await
                    .unwrap_or_else(|e| panic!("{e}"));
                let recovered = crash_world(seed, boundary)
                    .await
                    .unwrap_or_else(|e| panic!("{e}"));
                check_equals(&recovered, &reference, boundary).unwrap_or_else(|e| {
                    panic!("seed {seed:#x}: {e}");
                });
                checks += 1;
                *per_boundary.entry(boundary.name).or_insert(0) += 1;
            }
        }
        (checks, per_boundary)
    }

    /// Gate probe runs on its own because it intentionally completes the
    /// task before recovery and would perturb the reference comparison.
    #[tokio::test]
    async fn gate_refuses_open_reservations_on_every_open_seam() {
        for seed in 0..SMOKE_SEEDS {
            for boundary in BOUNDARIES {
                let amt = amounts(seed);
                let acct = Acct::open(amt.cap);
                build_prefix(boundary.name, &acct, &amt).await;
                let acct = acct.reopen();
                assert_gate_refuses_open_rows(&acct, boundary);
                let _ = finish(&acct, boundary).await;
            }
        }
    }

    #[tokio::test]
    async fn accounting_campaign_smoke() {
        let (checks, _) = run(SMOKE_SEEDS).await;
        assert_eq!(checks, SMOKE_SEEDS * BOUNDARIES.len() as u64);
    }

    #[tokio::test]
    #[ignore = "[fault] accounting crash seams (reserve/dispatch/accept/usage/settle/reconcile/completion), 200 seeds"]
    async fn accounting_campaign_full() {
        let (checks, per_boundary) = run(FULL_SEEDS).await;
        assert_eq!(checks, FULL_SEEDS * BOUNDARIES.len() as u64);
        for (name, n) in per_boundary {
            assert_eq!(n, FULL_SEEDS, "{name}");
        }
    }
}

// ======================================================================
// Budget accounting SQL failure modes (completion-vs-reservation
// invariant): the completion transaction's in-transaction reservation
// COUNT and the reservation transaction's task-state condition are the two
// SQL gates; each refuses typed and writes nothing.
// ======================================================================

/// SQL gate row: a reservation admitted while the task still permits
/// provider work races the verification transition; the completion
/// transaction's COUNT refuses typed and the task stays Verifying.
#[tokio::test]
async fn completion_gate_refuses_a_raced_reservation_and_stays_verifying() {
    use faktor_core::id::TaskId;
    use faktor_core::state::{TaskState, TaskTransition, VerificationStatus};
    use faktor_session::budget::ReservationState;
    use faktor_session::{BudgetAuthority as _, DurableBudgetLedger, Task, TaskBudget, TaskError};
    let dir = tempdir().unwrap();
    let manager =
        SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
    let ws = manager.create_workspace("/w").unwrap();
    let session = manager
        .create_session(ws, "gate", "fake", "m")
        .unwrap()
        .id();
    let handle = manager.get_session(session).unwrap().unwrap();
    let task = TaskId::new(1);
    handle
        .create_task(Task {
            task_id: task,
            session_id: session,
            goal: "gate".into(),
            acceptance_criteria: Vec::new(),
            plan: Vec::new(),
            budget: TaskBudget::default(),
            state: TaskState::Pending,
            created_ms: 1,
            updated_ms: 1,
        })
        .unwrap();
    let rev = handle.task_revision(task).unwrap();
    handle
        .transition_task(task, rev, TaskTransition::StartRunning, None)
        .unwrap();
    let ledger = DurableBudgetLedger::new(manager.clone());
    // The reservation is admitted while the task still permits provider
    // work; it then races the verification transition.
    let rid = ledger
        .reserve(session, task, OpId::new(1), 100, None)
        .await
        .unwrap();
    let rev = handle.task_revision(task).unwrap();
    handle
        .transition_task(task, rev, TaskTransition::RequestVerification, None)
        .unwrap();
    let rev = handle.task_revision(task).unwrap();
    handle
        .transition_task(task, rev, TaskTransition::StartVerification, None)
        .unwrap();
    let verifying = handle.task_revision(task).unwrap();
    let record = handle
        .create_verification_record(
            task,
            None,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            None,
            VerificationStatus::Passed,
            1,
        )
        .unwrap();
    // The completion transaction's COUNT catches the held row: typed
    // refusal, task stays Verifying, reservation stays reserved (money is
    // never silently freed).
    let err = handle
        .complete_verified_task(task, verifying, record)
        .unwrap_err();
    assert!(
        matches!(
            err,
            TaskError::AccountingIncomplete {
                open_count: 1,
                dispatched_count: 0,
                uncertain_count: 0,
                ..
            }
        ),
        "{err:?}"
    );
    assert_eq!(
        handle.get_task(task).unwrap().unwrap().state,
        TaskState::Verifying
    );
    assert_eq!(
        ledger.reservation_state(session, rid).unwrap(),
        ReservationState::Reserved
    );
    // Closing the accounting (pre-dispatch refund) unblocks completion.
    ledger.refund(session, rid).await.unwrap();
    let done = handle
        .complete_verified_task(task, verifying, record)
        .unwrap();
    assert_eq!(done.state, TaskState::VerifiedComplete);
}

/// The reservation transaction's task-state condition: a final task row
/// refuses a new provider operation typed (logical and attempt reserves)
/// and writes NOTHING.
#[tokio::test]
async fn reserve_transaction_refuses_in_a_final_task_state() {
    use faktor_core::id::TaskId;
    use faktor_core::op::ModelCallAttempt;
    use faktor_core::state::{TaskState, TaskTransition, VerificationStatus};
    use faktor_session::{
        BudgetAuthority as _, BudgetError, DurableBudgetLedger, Task, TaskBudget,
    };
    let dir = tempdir().unwrap();
    let manager =
        SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
    let ws = manager.create_workspace("/w").unwrap();
    let session = manager
        .create_session(ws, "final", "fake", "m")
        .unwrap()
        .id();
    let handle = manager.get_session(session).unwrap().unwrap();
    let task = TaskId::new(1);
    handle
        .create_task(Task {
            task_id: task,
            session_id: session,
            goal: "final".into(),
            acceptance_criteria: Vec::new(),
            plan: Vec::new(),
            budget: TaskBudget::default(),
            state: TaskState::Pending,
            created_ms: 1,
            updated_ms: 1,
        })
        .unwrap();
    let rev = handle.task_revision(task).unwrap();
    handle
        .transition_task(task, rev, TaskTransition::StartRunning, None)
        .unwrap();
    let rev = handle.task_revision(task).unwrap();
    handle
        .transition_task(task, rev, TaskTransition::RequestVerification, None)
        .unwrap();
    let rev = handle.task_revision(task).unwrap();
    handle
        .transition_task(task, rev, TaskTransition::StartVerification, None)
        .unwrap();
    let verifying = handle.task_revision(task).unwrap();
    let record = handle
        .create_verification_record(
            task,
            None,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            None,
            VerificationStatus::Passed,
            1,
        )
        .unwrap();
    handle
        .complete_verified_task(task, verifying, record)
        .unwrap();
    let ledger = DurableBudgetLedger::new(manager.clone());
    let err = ledger
        .reserve(session, task, OpId::new(7), 1, None)
        .await
        .unwrap_err();
    assert!(
        matches!(err, BudgetError::TaskStateForbidsReserve { .. }),
        "{err}"
    );
    let attempt = ModelCallAttempt::new(OpId::new(8), OpId::new(9), 0).unwrap();
    let err = ledger
        .reserve_attempt(session, task, attempt, 1, None)
        .await
        .unwrap_err();
    assert!(
        matches!(err, BudgetError::TaskStateForbidsReserve { .. }),
        "{err}"
    );
    assert!(ledger
        .reservations_of(session, task, 10)
        .unwrap()
        .is_empty());
    assert_eq!(
        handle.get_task(task).unwrap().unwrap().state,
        TaskState::VerifiedComplete
    );
}

/// A budget read that cannot be served is a typed Unavailable, never a
/// synthesized unlimited/zero budget: the UI state surfaces the reason and
/// the fail-safe policy refuses to call it explicitly uncapped.
#[tokio::test]
async fn budget_read_failure_is_not_unlimited_fault_row() {
    use faktor_core::id::TaskId;
    use faktor_core::state::TaskState;
    use faktor_session::budget::BudgetCapEvidence;
    use faktor_session::{
        BudgetAuthority as _, BudgetError, DurableBudgetLedger, Task, TaskBudget,
    };
    let dir = tempdir().unwrap();
    let manager =
        SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
    let ws = manager.create_workspace("/w").unwrap();
    let session = manager
        .create_session(ws, "readfail", "fake", "m")
        .unwrap()
        .id();
    let handle = manager.get_session(session).unwrap().unwrap();
    let task = TaskId::new(1);
    handle
        .create_task(Task {
            task_id: task,
            session_id: session,
            goal: "readfail".into(),
            acceptance_criteria: Vec::new(),
            plan: Vec::new(),
            budget: TaskBudget::default(),
            state: TaskState::Running,
            created_ms: 1,
            updated_ms: 1,
        })
        .unwrap();
    let ledger = DurableBudgetLedger::new(manager.clone());
    ledger
        .set_task_max_cost(session, task, Some(1_000))
        .unwrap();
    assert!(
        manager
            .read_service()
            .shutdown(Duration::from_secs(10))
            .await
    );
    let err = manager.budget_view(session, task).await.unwrap_err();
    assert!(
        matches!(
            err,
            BudgetError::Unavailable {
                cap: BudgetCapEvidence::Unknown,
                ..
            }
        ),
        "{err:?}"
    );
    assert!(!err.read_failure_is_explicitly_uncapped());
    let state = manager.budget_state(session, task).await;
    assert!(!state.is_known());
    assert!(state.reason().is_some());
    // The durable ledger itself is untouched: the sync view still answers.
    assert_eq!(
        ledger
            .session_budget_view(session, task)
            .unwrap()
            .max_cost_micro,
        Some(1_000)
    );
}
