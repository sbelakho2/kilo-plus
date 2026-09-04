//! faktor-tests-quality — the adversarial quality suite (audit item 12).
//!
//! Drives the REAL runtime (SessionManager + AgentRuntime on temp dirs,
//! scripted FakeProviders, real file effects) against the agent-failure
//! taxonomy and the hard invariants: cross-turn durable loop detection,
//! within-turn alternation stops, crash-recovery side-effect discipline,
//! compaction goal survival, completion-review blocking, exactly-one
//! completion record per logical turn, stale-write refusal and the
//! never-re-run rule for unknown effects. Happy paths are never asserted
//! alone: every test tries to break the invariant it locks.
//!
//! Everything lives under `#[cfg(test)]` (this is a test-harness crate; the
//! lib view exists only so the workspace builds it).

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    use faktor_agent::{
        AgentDeps, AgentRuntime, NoEvidence, PermissionRequester, RecoveryHint, Tool, ToolOutcome,
        ToolRegistry,
    };
    use faktor_core::capability::PermissionDecision;
    use faktor_core::error::Error;
    use faktor_core::event::EventKind;
    use faktor_core::id::SessionId;
    use faktor_core::model::ModelCapabilities;
    use faktor_core::op::EffectStatus;
    use faktor_core::resource::ResourceClass;
    use faktor_core::state::AgentState;
    use faktor_core::time::SystemClock;
    use faktor_provider::{FakeProvider, ProviderRegistry, ScriptedResponse};
    use faktor_session::SessionManager;
    use serde_json::json;
    use tempfile::{tempdir, TempDir};

    // ------------------------------------------------------------- harness

    struct AlwaysAllow;
    impl PermissionRequester for AlwaysAllow {
        fn request(
            &self,
            _session: SessionId,
            _permission: &faktor_session::ops::PermissionRequest,
        ) -> Pin<
            Box<dyn std::future::Future<Output = faktor_core::Result<PermissionDecision>> + Send>,
        > {
            Box::pin(async { Ok(PermissionDecision::Allow) })
        }
    }

    fn tools_caps() -> ModelCapabilities {
        ModelCapabilities {
            tools: true,
            ..Default::default()
        }
    }

    fn fake(script: Vec<ScriptedResponse>) -> FakeProvider {
        FakeProvider::with_script("fake", tools_caps(), script)
    }

    fn open(dir: &TempDir) -> Arc<SessionManager> {
        SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap()
    }

    /// A fresh AgentDeps over `manager` (opening its own CAS handle under the
    /// shared test dir). Compact-at-usage and the verifier are injectable:
    /// the failure tests need 0.0 (always compact), the review test needs a
    /// real (fake-run) verifier.
    fn deps(
        manager: Arc<SessionManager>,
        dir: &TempDir,
        provider: FakeProvider,
        tools: Vec<Tool>,
        compact_at_usage: f64,
        verifier: Option<Arc<faktor_verify::Verifier>>,
    ) -> AgentDeps {
        let mut registry = ProviderRegistry::new();
        registry.register(Arc::new(provider));
        let mut tool_registry = ToolRegistry::new();
        for t in tools {
            tool_registry.register(t);
        }
        AgentDeps {
            session: manager,
            providers: Arc::new(registry),
            chunk_sink: None,
            permission_requester: Arc::new(AlwaysAllow),
            evidence: Arc::new(NoEvidence),
            tools: Arc::new(tool_registry),
            cas: Some(Arc::new(
                faktor_cas::Cas::open(dir.path().join("cas")).unwrap(),
            )),
            workspaces: faktor_fs::WorkspaceFileService::new(),
            edit: None,
            snapshots: None,
            sandbox: None,
            supervisor: None,
            verifier,
            hooks: None,
            instructions_loader: None,
            router: None,
            budget_micro: None,
            model: "m".into(),
            compaction_model: None,
            compact_at_usage,
            instructions: "You are a test agent.".into(),
            clock: Arc::new(SystemClock),
            tool_call_mode: faktor_agent::ToolCallMode::Native,
            tool_deadline_ms: 2000,
            retry_policy: faktor_core::retry::RetryPolicy::default(),
        }
    }

    fn runtime(
        manager: Arc<SessionManager>,
        dir: &TempDir,
        provider: FakeProvider,
        tools: Vec<Tool>,
    ) -> Arc<AgentRuntime> {
        AgentRuntime::new(deps(manager, dir, provider, tools, 0.65, None)).unwrap()
    }

    fn handle(manager: &Arc<SessionManager>, sid: SessionId) -> faktor_session::SessionHandle {
        manager.get_session(sid).unwrap().unwrap()
    }

    fn new_session(manager: &Arc<SessionManager>, title: &str) -> SessionId {
        let ws = manager.create_workspace("/w").unwrap();
        manager.create_session(ws, title, "fake", "m").unwrap().id()
    }

    fn new_session_in(manager: &Arc<SessionManager>, root: &Path, title: &str) -> SessionId {
        let ws = manager.create_workspace(root.to_str().unwrap()).unwrap();
        manager.create_session(ws, title, "fake", "m").unwrap().id()
    }

    fn ledger_of(handle: &faktor_session::SessionHandle) -> faktor_context::ledger::TaskLedger {
        serde_json::from_value(handle.get_task_ledger().unwrap().unwrap()).unwrap()
    }

    fn tool_call(id: &str, name: &str, input: serde_json::Value) -> ScriptedResponse {
        ScriptedResponse::ToolCall {
            id: id.into(),
            name: name.into(),
            input,
        }
    }

    /// A tool that always succeeds (exit 0) — genuine progress.
    fn ok_tool(name: &str) -> Tool {
        let name_owned = name.to_string();
        Tool {
            name: name.to_string(),
            description: "succeeds".into(),
            input_schema: json!({"type": "object"}),
            resource_class: ResourceClass::Cpu,
            capability: None,
            recovery_hint: RecoveryHint::Idempotent,
            path_args: vec![],
            execute: Arc::new(move |_ctx, args| {
                let name = name_owned.clone();
                Box::pin(async move {
                    Ok(ToolOutcome {
                        text: format!("{name} ok: {args}"),
                        exit_code: Some(0),
                        ..Default::default()
                    })
                })
            }),
        }
    }

    /// A tool that FAILS with `text` and the given exit code (a failing
    /// command: unknown external effect, never re-run after a crash).
    fn fail_tool(name: &str, text: &str, exit: i32) -> Tool {
        let name_owned = name.to_string();
        let text_owned = text.to_string();
        Tool {
            name: name.to_string(),
            description: "fails".into(),
            input_schema: json!({"type": "object"}),
            resource_class: ResourceClass::Cpu,
            capability: None,
            recovery_hint: RecoveryHint::UnknownEffect,
            path_args: vec![],
            execute: Arc::new(move |_ctx, args| {
                let name = name_owned.clone();
                let text = text_owned.clone();
                Box::pin(async move {
                    Ok(ToolOutcome {
                        text: format!("{name}: {text}: {args}"),
                        exit_code: Some(exit),
                        ..Default::default()
                    })
                })
            }),
        }
    }

    /// A REAL workspace write tool: resolves the relative `path` against the
    /// session workspace root and writes the `content` bytes atomically.
    /// Files are real; the end-of-turn review reads them back from disk.
    fn real_write_tool() -> Tool {
        Tool {
            name: "write_file".into(),
            description: "writes a real file in the workspace".into(),
            input_schema: json!({"type": "object"}),
            resource_class: ResourceClass::DiskWrite,
            capability: None,
            recovery_hint: RecoveryHint::WorkspaceWrite,
            path_args: vec!["path".into()],
            execute: Arc::new(|ctx, args| {
                Box::pin(async move {
                    let Some(ws) = &ctx.workspace else {
                        return Err(Error::internal("no workspace wired"));
                    };
                    let path = args.get("path").and_then(|p| p.as_str()).unwrap_or("");
                    let content = args
                        .get("content")
                        .and_then(|c| c.as_str())
                        .unwrap_or_default();
                    ws.write_atomic(Path::new(path), content.as_bytes())
                        .map_err(|e| Error::internal(format!("write {path}: {e}")))?;
                    Ok(ToolOutcome {
                        text: "wrote".into(),
                        exit_code: Some(0),
                        ..Default::default()
                    })
                })
            }),
        }
    }

    /// Turn-ends events of one session.
    fn events(manager: &Arc<SessionManager>, sid: SessionId) -> Vec<faktor_core::event::Event> {
        handle(manager, sid).events_range(1, None).unwrap()
    }

    fn count_events(manager: &Arc<SessionManager>, sid: SessionId, kind: EventKind) -> usize {
        events(manager, sid)
            .iter()
            .filter(|e| e.kind == kind)
            .count()
    }

    /// Wait until the in-tool gate opens (the tool's execute has run past
    /// its durable ToolStarted) or fail after a bound.
    async fn wait_until_started(rx: &mut tokio::sync::watch::Receiver<bool>) {
        tokio::time::timeout(Duration::from_secs(20), async {
            while !*rx.borrow() {
                if rx.changed().await.is_err() {
                    break;
                }
            }
        })
        .await
        .expect("tool execution must start");
    }

    // ====================================================================
    // 1. AGENT-FAILURE MATRIX
    // ====================================================================

    /// Matrix 1a: three consecutive turns, each ending in the SAME failing
    /// command (tool exit 1). The per-turn LoopDetector cannot see across
    /// turns, so the durable loop signal (runtime.rs `durable_loop_signals`,
    /// keyed `fail <failure text>`, bumped per all-failing turn) must trip on
    /// the THIRD identical all-failing turn — and must NOT trip earlier.
    #[tokio::test]
    async fn repeated_identical_command_replans() {
        let dir = tempdir().unwrap();
        let manager = open(&dir);
        let sid = new_session(&manager, "cross-turn loop");
        let boom = fail_tool("run_command", "boom", 1);
        let mut outcomes = Vec::new();
        for i in 0..3 {
            let provider = fake(vec![
                tool_call(
                    &format!("c{i}"),
                    "run_command",
                    json!({"command": "cargo check -p faktor-core"}),
                ),
                ScriptedResponse::End,
            ]);
            let rt = AgentRuntime::new(deps(
                manager.clone(),
                &dir,
                provider,
                vec![boom.clone()],
                0.65,
                None,
            ))
            .unwrap();
            let outcome = rt
                .run_turn(sid, &format!("attempt {i}"), &[])
                .await
                .unwrap();
            outcomes.push(outcome);
        }
        // Turns 1-2 are NOT stopped: the durable window only trips at 3.
        assert!(!outcomes[0].loop_stopped && !outcomes[1].loop_stopped);
        assert_eq!(outcomes[0].final_state, AgentState::ReadyForNextTurn);
        assert_eq!(outcomes[1].final_state, AgentState::ReadyForNextTurn);
        assert!(
            outcomes[2].loop_stopped,
            "the 3rd identical all-failing turn must trip the durable detector"
        );
        assert_eq!(outcomes[2].final_state, AgentState::FailedRecoverable);
        // Only the two non-tripped turns journaled a completion record.
        assert_eq!(count_events(&manager, sid, EventKind::TurnCompleted), 2);
        // The failure itself is durable task state (ledger), exactly once.
        let ledger = ledger_of(&handle(&manager, sid));
        assert_eq!(
            ledger
                .known_failures
                .iter()
                .filter(|f| f.contains("boom"))
                .count(),
            1,
            "{:?}",
            ledger.known_failures
        );
    }

    /// Matrix 1b: a SINGLE turn whose provider script alternates tool calls
    /// A,B,A,B,A,B. Within the turn the runtime feeds every call of the batch
    /// through one `LoopDetector` (drive_turn -> run_tool_calls); the
    /// A->B->A->B oscillation detector (loop_detect.rs) trips at the FIFTH
    /// step — before the batch can do any more work. outcome.loop_stopped.
    #[tokio::test]
    async fn a_b_alternation_stops() {
        let dir = tempdir().unwrap();
        let manager = open(&dir);
        let sid = new_session(&manager, "alternation");
        let mut script = Vec::new();
        for i in 0..6 {
            let (name, id) = if i % 2 == 0 {
                ("alpha", format!("a{i}"))
            } else {
                ("beta", format!("b{i}"))
            };
            script.push(ScriptedResponse::ToolCall {
                id,
                name: name.into(),
                input: json!({}),
            });
        }
        script.push(ScriptedResponse::End);
        let rt = AgentRuntime::new(deps(
            manager.clone(),
            &dir,
            fake(script),
            vec![ok_tool("alpha"), ok_tool("beta")],
            0.65,
            None,
        ))
        .unwrap();
        let outcome = rt.run_turn(sid, "oscillate", &[]).await.unwrap();
        // Both tools succeed when they run, so the stop is caused by the
        // call SEQUENCE (alternation), never by a failure.
        assert!(
            outcome.loop_stopped,
            "A,B,A,B,(A) alternation within one batch must trip the detector"
        );
        assert_eq!(outcome.final_state, AgentState::FailedRecoverable);
        // The turn ends on a promptable failure state, never wedged.
        let h = handle(&manager, sid);
        assert_eq!(h.state().unwrap(), AgentState::FailedRecoverable);
        let receipt = h.submit_prompt("retry with a different plan", &[]).unwrap();
        assert!(
            receipt.accepted && !receipt.queued,
            "a loop-stopped session must accept the re-plan"
        );
    }

    /// Matrix 1c: crash recovery must not duplicate real side effects. A
    /// REAL turn drives a REAL idempotent fs tool; the runtime is dropped
    /// mid-tool (durable ToolStarted exists, the machine is left at
    /// ExecutingTool), a NEW runtime reopens the SAME store dir, recovers,
    /// and continues the same logical turn. The tool's physical execution
    /// counter (a temp file the tool appends to) must end at exactly 1 and
    /// the data file must exist exactly once — replay is ONE new physical
    /// attempt of the SAME logical run (ReplayStarted, never a second
    /// ToolStarted row).
    #[tokio::test]
    async fn no_duplicate_side_effects_after_recovery() {
        let dir = tempdir().unwrap();
        let ws_root = dir.path().join("ws");
        std::fs::create_dir_all(&ws_root).unwrap();
        let counter_path = ws_root.join("executions.txt");
        let data_path = ws_root.join("data.txt");

        let (tx, mut rx) = tokio::sync::watch::channel(false);
        let park_tool = Tool {
            name: "durable_write".into(),
            description: "idempotent write that never gets to run".into(),
            input_schema: json!({"type": "object"}),
            resource_class: ResourceClass::DiskWrite,
            capability: None,
            recovery_hint: RecoveryHint::Idempotent,
            path_args: vec!["path".into()],
            execute: {
                let tx = tx.clone();
                Arc::new(move |_ctx, _args| {
                    let tx = tx.clone();
                    Box::pin(async move {
                        // Signal AFTER the run row is durable, BEFORE any side
                        // effect; the drive is dropped at this park point.
                        let _ = tx.send(true);
                        let never: std::future::Pending<Result<ToolOutcome, Error>> =
                            std::future::pending();
                        never.await
                    })
                })
            },
        };
        let turn_op: faktor_core::id::OpId;
        let session: SessionId;
        {
            let manager1 = open(&dir);
            session = new_session_in(&manager1, &ws_root, "exactly once");
            let handle1 = handle(&manager1, session);
            let receipt = handle1.submit_prompt("write data.txt", &[]).unwrap();
            assert!(!receipt.queued);
            turn_op = receipt.op_id;
            let provider = fake(vec![
                tool_call(
                    "c1",
                    "durable_write",
                    json!({"path": "data.txt", "content": "hello"}),
                ),
                ScriptedResponse::End,
            ]);
            let rt = AgentRuntime::new(deps(
                manager1.clone(),
                &dir,
                provider,
                vec![park_tool],
                0.65,
                None,
            ))
            .unwrap();
            let drive = tokio::spawn({
                let rt = rt.clone();
                let manager = manager1.clone();
                async move {
                    let h = manager.get_session(session).unwrap().unwrap();
                    rt.drive_receipt(&h, receipt, None).await
                }
            });
            wait_until_started(&mut rx).await;
            drive.abort();
            let _ = drive.await; // JoinError: the crash
                                 // CRASH RESIDUE: durable ToolStarted, unfinished run row.
            let pending = handle1.pending_tool_runs().unwrap();
            assert_eq!(pending.len(), 1, "one interrupted run row");
            assert_eq!(pending[0].tool, "durable_write");
            assert_eq!(handle1.state().unwrap(), AgentState::ExecutingTool);
            assert_eq!(
                count_events(&manager1, session, EventKind::ToolStarted),
                1,
                "the crashed attempt journaled exactly one ToolStarted"
            );
            assert!(
                !data_path.exists() && !counter_path.exists(),
                "the crash landed before any side effect"
            );
            drop(rt);
            drop(manager1);
        }

        // REOPEN the same store dir. The recovery sweep must DEFER the
        // idempotent run (status "running", RerunAllowed) — recover() itself
        // never executes tools.
        let executions = Arc::new(AtomicUsize::new(0));
        let replay_tool = Tool {
            name: "durable_write".into(),
            description: "idempotent write".into(),
            input_schema: json!({"type": "object"}),
            resource_class: ResourceClass::DiskWrite,
            capability: None,
            recovery_hint: RecoveryHint::Idempotent,
            path_args: vec!["path".into()],
            execute: {
                let executions = executions.clone();
                let counter_path = counter_path.clone();
                let data_path = data_path.clone();
                Arc::new(move |_ctx, _args| {
                    let executions = executions.clone();
                    let counter_path = counter_path.clone();
                    let data_path = data_path.clone();
                    Box::pin(async move {
                        executions.fetch_add(1, Ordering::SeqCst);
                        std::fs::write(&counter_path, "x\n").unwrap();
                        std::fs::write(&data_path, b"hello").unwrap();
                        Ok(ToolOutcome {
                            text: "wrote".into(),
                            exit_code: Some(0),
                            ..Default::default()
                        })
                    })
                })
            },
        };
        let manager2 = open(&dir);
        let rt2 = AgentRuntime::new(deps(
            manager2.clone(),
            &dir,
            fake(vec![
                ScriptedResponse::Text("recovered".into()),
                ScriptedResponse::End,
            ]),
            vec![replay_tool],
            0.65,
            None,
        ))
        .unwrap();
        let reports = rt2.recover().unwrap();
        let report = reports.iter().find(|r| r.session_id == session).unwrap();
        assert_eq!(report.crashed_ops.len(), 1);
        assert_eq!(
            report.crashed_ops[0].status, "running",
            "the sweep defers, it never executes"
        );
        assert_eq!(executions.load(Ordering::SeqCst), 0);
        assert!(
            !data_path.exists(),
            "recover() must not re-run the idempotent tool"
        );
        // continue_turn replays the SAME logical run exactly once.
        let outcome = rt2.continue_turn(session).await.unwrap();
        assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);
        assert_eq!(
            outcome.op_id, turn_op,
            "the SAME logical turn completes after recovery"
        );
        assert_eq!(
            executions.load(Ordering::SeqCst),
            1,
            "the tool executed exactly once in total"
        );
        assert_eq!(
            std::fs::read(&data_path).unwrap(),
            b"hello",
            "the data file exists exactly once, with the exact bytes"
        );
        let h2 = handle(&manager2, session);
        assert!(h2.pending_tool_runs().unwrap().is_empty());
        let evs = events(&manager2, session);
        assert_eq!(
            evs.iter()
                .filter(|e| e.kind == EventKind::ToolStarted)
                .count(),
            1,
            "replay never creates a second run row"
        );
        assert_eq!(
            evs.iter()
                .filter(|e| e.kind == EventKind::ReplayStarted)
                .count(),
            1,
            "exactly one replay attempt"
        );
        assert_eq!(
            evs.iter()
                .filter(|e| e.kind == EventKind::TurnCompleted && e.op_id == Some(turn_op))
                .count(),
            1,
            "one completion for the resumed logical turn"
        );
    }

    /// Matrix 1d: compaction must never lose the durable task state. A
    /// session's ledger is seeded with a REAL goal (session title) and a REAL
    /// known failure through the runtime; then many text-heavy turns run with
    /// compact_at_usage = 0.0 so every turn with history compacts (hard
    /// invariant: a successful compaction preserves the ledger whole — the
    /// plan's ledger is the clone of the loaded one, written back durably).
    ///
    /// The SEED turn runs at the normal 0.65 trigger: an automatic compaction
    /// attempt over a ~zero-token transcript would be a pathological trigger
    /// (the session layer hard-rejects any "after > before" record), so the
    /// first 0.0 trigger only fires once real 2KB+ history exists.
    #[tokio::test]
    async fn compaction_never_loses_goal_or_failures() {
        let dir = tempdir().unwrap();
        let manager = open(&dir);
        let title = "fix the payments parser: adversarial goal marker GOLDMARK-42";
        let sid = new_session(&manager, title);
        let seed_script = vec![
            tool_call(
                "c1",
                "run_command",
                json!({"command": "cargo check -p payments"}),
            ),
            tool_call(
                "c2",
                "write_file",
                json!({"path": "src/payments.rs", "content": "pub fn parse() {}"}),
            ),
            // ~2KB of assistant text: the durable history the first
            // aggressive compaction shrinks (never a tiny-transcript trigger).
            ScriptedResponse::Text(format!("seed analysis {}", "a".repeat(2000))),
            ScriptedResponse::End,
        ];
        let seed_tools = vec![
            fail_tool("run_command", "check failed: 3 errors in payments.rs", 1),
            ok_tool("write_file"),
        ];
        let rt0 = AgentRuntime::new(deps(
            manager.clone(),
            &dir,
            fake(seed_script),
            seed_tools,
            0.65,
            None,
        ))
        .unwrap();
        let o0 = rt0.run_turn(sid, "fix payments", &[]).await.unwrap();
        assert_eq!(o0.final_state, AgentState::ReadyForNextTurn);
        assert!(
            !o0.compacted,
            "the seed turn stays under the 0.65 trigger and seeds full history"
        );
        // PRE-CONDITION: the durable ledger really carries goal + failure.
        let h0 = handle(&manager, sid);
        let ledger0 = ledger_of(&h0);
        assert_eq!(ledger0.goal, title, "goal seeded from the session title");
        assert!(
            ledger0
                .known_failures
                .iter()
                .any(|f| f.contains("payments.rs")),
            "known failures must be seeded: {:?}",
            ledger0.known_failures
        );
        // Force ≥5 real compactions through repeated 2KB turns.
        let mut compacted = 0usize;
        for t in 0..14 {
            let provider = fake(vec![
                ScriptedResponse::Text(format!("compaction turn {t} {}", "y".repeat(2000))),
                ScriptedResponse::End,
            ]);
            let rt = AgentRuntime::new(deps(manager.clone(), &dir, provider, vec![], 0.0, None))
                .unwrap();
            let outcome = rt
                .run_turn(sid, &format!("keep going {t} {}", "z".repeat(2000)), &[])
                .await
                .unwrap();
            assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);
            if outcome.compacted {
                compacted += 1;
            }
        }
        assert!(
            compacted >= 5,
            "the wire footprint must force repeated compactions, got {compacted}"
        );
        let ledger = ledger_of(&handle(&manager, sid));
        assert_eq!(
            ledger.goal, title,
            "the goal must survive {compacted} compactions verbatim"
        );
        assert!(
            ledger
                .known_failures
                .iter()
                .any(|f| f.contains("check failed") && f.contains("payments.rs")),
            "known failures must survive {compacted} compactions: {:?}",
            ledger.known_failures
        );
    }

    /// Matrix 1e: a write whose head looks like a test (describe/it markers)
    /// but contains ZERO assertion tokens is a weakened test: the end-of-turn
    /// completion review (same genuine ends, real file heads from disk) must
    /// verdict "block". The review is ADVISORY (runtime docs) — it never
    /// fails the turn: final_state stays ReadyForNextTurn while the verdict
    /// is block.
    #[tokio::test]
    async fn weakened_tests_are_blocked_by_review() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("ws");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::create_dir_all(root.join("tests")).unwrap();
        std::fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"x\"\nversion = \"0.1.0\"\n",
        )
        .unwrap();
        std::fs::write(root.join("src/lib.rs"), "pub fn f() -> u32 { 1 }\n").unwrap();
        let manager = open(&dir);
        let sid = new_session_in(&manager, &root, "weakened tests");
        // Fake run closure: derived checks (none for a .js change in a Rust
        // repo) would pass; the review signal scan is what must block.
        let verifier = Arc::new(faktor_verify::Verifier::new(Arc::new(|_cmd: &str| Ok(()))));
        let hollow = "describe(\"calculator\", () => {\n    it(\"adds\", () => {\n        const got = calc.add(1, 2);\n    });\n});\n";
        let provider = fake(vec![
            tool_call(
                "c1",
                "write_file",
                json!({"path": "tests/calc_spec.js", "content": hollow}),
            ),
            ScriptedResponse::Text("done".into()),
            ScriptedResponse::End,
        ]);
        let rt = AgentRuntime::new(deps(
            manager.clone(),
            &dir,
            provider,
            vec![real_write_tool()],
            0.65,
            Some(verifier),
        ))
        .unwrap();
        let outcome = rt
            .run_turn(sid, "add a test for the calculator", &[])
            .await
            .unwrap();
        assert_eq!(
            outcome.final_state,
            AgentState::ReadyForNextTurn,
            "the review is advisory and never fails the turn (runtime.rs TurnOutcome::review)"
        );
        let review = outcome
            .review
            .expect("a verifier + changed files must produce the review");
        assert_eq!(review["verdict"], "block", "{review}");
        let blocking: Vec<String> = review["blocking"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|s| s.as_str().map(str::to_string))
            .collect();
        assert!(
            blocking
                .iter()
                .any(|b| b.contains("weakened test file without assertions")
                    && b.contains("tests/calc_spec.js")),
            "blocking must name the weakened file: {blocking:?}"
        );
        let files = review["evidence"]["files"].as_array().unwrap();
        assert!(
            files
                .iter()
                .any(|f| f["path"] == "tests/calc_spec.js" && f["weakened_test_suspect"] == true),
            "{review}"
        );
        // The write itself is real and untouched by the review.
        assert_eq!(
            std::fs::read_to_string(root.join("tests/calc_spec.js")).unwrap(),
            hollow
        );
    }

    // ====================================================================
    // 2. HARD INVARIANTS
    // ====================================================================

    /// Invariant 2f: one logical turn — however many tool batches it drives —
    /// journals EXACTLY ONE TurnCompleted, bound to that turn's op, and never
    /// enters ReadyForNextTurn mid-turn. Locked across TWO real turns of the
    /// same session: each op gets exactly one completion record.
    #[tokio::test]
    async fn exactly_one_completion_record_per_turn() {
        let dir = tempdir().unwrap();
        let manager = open(&dir);
        let sid = new_session(&manager, "one completion each");
        let echo = ok_tool("echo");
        let o1 = {
            let provider = fake(vec![
                tool_call("c1", "echo", json!({"x": 1})),
                ScriptedResponse::Text("answer one".into()),
                ScriptedResponse::End,
            ]);
            let rt = AgentRuntime::new(deps(
                manager.clone(),
                &dir,
                provider,
                vec![echo.clone()],
                0.65,
                None,
            ))
            .unwrap();
            rt.run_turn(sid, "turn one", &[]).await.unwrap()
        };
        let o2 = {
            let provider = fake(vec![
                tool_call("c1", "echo", json!({"x": 2})),
                ScriptedResponse::Text("answer two".into()),
                ScriptedResponse::End,
            ]);
            let rt = AgentRuntime::new(deps(
                manager.clone(),
                &dir,
                provider,
                vec![echo],
                0.65,
                None,
            ))
            .unwrap();
            rt.run_turn(sid, "turn two", &[]).await.unwrap()
        };
        assert_eq!(o1.final_state, AgentState::ReadyForNextTurn);
        assert_eq!(o2.final_state, AgentState::ReadyForNextTurn);
        let evs = events(&manager, sid);
        let completed: Vec<_> = evs
            .iter()
            .filter(|e| e.kind == EventKind::TurnCompleted)
            .collect();
        assert_eq!(completed.len(), 2, "{completed:?}");
        assert_eq!(
            completed
                .iter()
                .filter(|e| e.op_id == Some(o1.op_id))
                .count(),
            1,
            "turn one has exactly one completion"
        );
        assert_eq!(
            completed
                .iter()
                .filter(|e| e.op_id == Some(o2.op_id))
                .count(),
            1,
            "turn two has exactly one completion"
        );
        assert_ne!(o1.op_id, o2.op_id);
        // ReadyForNextTurn is journaled exactly twice: only the two genuine
        // ends, never between batches.
        let ready = evs
            .iter()
            .filter(|e| e.state == AgentState::ReadyForNextTurn)
            .count();
        assert_eq!(ready, 2, "ReadyForNextTurn only at the two genuine ends");
    }

    /// Invariant 2g: a write whose expected base hash is STALE (the model's
    /// intent was built against file state that no longer exists — here a
    /// user edit landed between turns) must be REFUSED without touching the
    /// file. Real workspace writes through the runtime, hashes of the actual
    /// bytes on disk; the positive control (fresh hash) must still apply.
    #[tokio::test]
    async fn stale_hash_write_refused() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("ws");
        std::fs::create_dir_all(&root).unwrap();
        let manager = open(&dir);
        let sid = new_session_in(&manager, &root, "stale writes");
        let guarded = Tool {
            name: "guarded_write".into(),
            description: "write that refuses a stale expected base hash".into(),
            input_schema: json!({"type": "object"}),
            resource_class: ResourceClass::DiskWrite,
            capability: None,
            recovery_hint: RecoveryHint::WorkspaceWrite,
            path_args: vec!["path".into()],
            execute: Arc::new(|ctx, args| {
                Box::pin(async move {
                    let Some(ws) = &ctx.workspace else {
                        return Err(Error::internal("no workspace wired"));
                    };
                    let path = args
                        .get("path")
                        .and_then(|p| p.as_str())
                        .unwrap_or("data.txt");
                    let content = args.get("content").and_then(|c| c.as_str()).unwrap_or("");
                    let expected = args
                        .get("expected_hash")
                        .and_then(|e| e.as_str())
                        .filter(|e| !e.is_empty());
                    let current: Option<Vec<u8>> =
                        ws.read(Path::new(path), 1 << 20).ok().map(|d| d.bytes);
                    if let Some(expected) = expected {
                        let Some(cur) = &current else {
                            return Ok(ToolOutcome {
                                text: format!(
                                    "REFUSED missing base for {path:?} (expected {expected}); no write"
                                ),
                                exit_code: Some(1),
                                ..Default::default()
                            });
                        };
                        let actual = blake3::hash(cur).to_hex().to_string();
                        if actual != expected {
                            return Ok(ToolOutcome {
                                text: format!(
                                    "REFUSED stale base for {path:?}: current {actual} != expected {expected}; file untouched"
                                ),
                                exit_code: Some(1),
                                ..Default::default()
                            });
                        }
                    }
                    ws.write_atomic(Path::new(path), content.as_bytes())
                        .map_err(|e| Error::internal(format!("write {path:?}: {e}")))?;
                    Ok(ToolOutcome {
                        text: format!("wrote {path:?}"),
                        exit_code: Some(0),
                        ..Default::default()
                    })
                })
            }),
        };
        let rt = runtime(
            manager.clone(),
            &dir,
            fake(vec![
                tool_call(
                    "c1",
                    "guarded_write",
                    json!({"path": "data.txt", "content": "v1-model"}),
                ),
                ScriptedResponse::End,
            ]),
            vec![guarded.clone()],
        );
        rt.run_turn(sid, "write v1", &[]).await.unwrap();
        assert_eq!(
            std::fs::read_to_string(root.join("data.txt")).unwrap(),
            "v1-model"
        );
        let h1 = blake3::hash(b"v1-model").to_hex().to_string();
        // The USER edits the file between the model's write intent and the
        // model's next attempt (the concurrent-edit window made real).
        std::fs::write(root.join("data.txt"), b"v2-user").unwrap();
        // Turn B: the model still thinks the base is v1 -> refused.
        let rt = runtime(
            manager.clone(),
            &dir,
            fake(vec![
                tool_call(
                    "c1",
                    "guarded_write",
                    json!({"path": "data.txt", "content": "v1-model-REWRITE", "expected_hash": h1}),
                ),
                ScriptedResponse::End,
            ]),
            vec![guarded.clone()],
        );
        rt.run_turn(sid, "overwrite v1 again", &[]).await.unwrap();
        assert_eq!(
            std::fs::read_to_string(root.join("data.txt")).unwrap(),
            "v2-user",
            "the stale write must never land"
        );
        // The refusal is visible in the durable tool result.
        let h = handle(&manager, sid);
        let rows = h.messages_before(None, 20).unwrap();
        let mut refused = false;
        for row in rows {
            for part in h.parts_of(row.id).unwrap() {
                if part.kind == "tool_result"
                    && part
                        .data
                        .get("excerpt")
                        .and_then(|e| e.as_str())
                        .is_some_and(|e| e.contains("REFUSED stale base"))
                {
                    refused = true;
                }
            }
        }
        assert!(refused, "the refusal must be durable as the tool result");
        // Positive control: the SAME write with the CURRENT base applies.
        let h2 = blake3::hash(b"v2-user").to_hex().to_string();
        let rt = runtime(
            manager.clone(),
            &dir,
            fake(vec![
                tool_call(
                    "c1",
                    "guarded_write",
                    json!({"path": "data.txt", "content": "v3-model", "expected_hash": h2}),
                ),
                ScriptedResponse::End,
            ]),
            vec![guarded],
        );
        let o = rt
            .run_turn(sid, "write v3 on the current base", &[])
            .await
            .unwrap();
        assert_eq!(o.final_state, AgentState::ReadyForNextTurn);
        assert_eq!(
            std::fs::read_to_string(root.join("data.txt")).unwrap(),
            "v3-model"
        );
    }

    /// Invariant 2h: a tool with UNKNOWN external effects that crashed
    /// mid-run (durable ToolStarted, runtime dropped while the tool was
    /// executing, side effect already visible on disk) is marked failed with
    /// effect unknown by recovery and is NEVER re-executed: the side-effect
    /// counter stays exactly 1 and no ReplayStarted is ever journaled.
    #[tokio::test]
    async fn zero_duplicated_non_idempotent_effects() {
        let dir = tempdir().unwrap();
        let ws_root = dir.path().join("ws");
        std::fs::create_dir_all(&ws_root).unwrap();
        let counter_path = ws_root.join("side_effects.txt");
        let session: SessionId;
        {
            let manager1 = open(&dir);
            session = new_session_in(&manager1, &ws_root, "unknown effect");
            let (tx, mut rx) = tokio::sync::watch::channel(false);
            let crash_tool = Tool {
                name: "run_cmd".into(),
                description: "command with unknown external effects".into(),
                input_schema: json!({"type": "object"}),
                resource_class: ResourceClass::Cpu,
                capability: None,
                recovery_hint: RecoveryHint::UnknownEffect,
                path_args: vec![],
                execute: {
                    let counter_path = counter_path.clone();
                    let tx = tx.clone();
                    Arc::new(move |_ctx, _args| {
                        let tx = tx.clone();
                        let counter_path = counter_path.clone();
                        Box::pin(async move {
                            // The side effect LANDS, then the runtime is
                            // dropped mid-tool (the durable ToolStarted row
                            // exists; no finish is ever journaled).
                            let mut f = std::fs::OpenOptions::new()
                                .create(true)
                                .append(true)
                                .open(&counter_path)
                                .unwrap();
                            use std::io::Write;
                            writeln!(f, "side effect").unwrap();
                            let _ = tx.send(true);
                            let never: std::future::Pending<Result<ToolOutcome, Error>> =
                                std::future::pending();
                            never.await
                        })
                    })
                },
            };
            let h1 = handle(&manager1, session);
            let receipt = h1
                .submit_prompt("run the side-effectful command", &[])
                .unwrap();
            let provider = fake(vec![
                tool_call("c1", "run_cmd", json!({"command": "rm -rf /tmp/x"})),
                ScriptedResponse::End,
            ]);
            let rt = AgentRuntime::new(deps(
                manager1.clone(),
                &dir,
                provider,
                vec![crash_tool],
                0.65,
                None,
            ))
            .unwrap();
            let drive = tokio::spawn({
                let rt = rt.clone();
                let manager = manager1.clone();
                async move {
                    let h = manager.get_session(session).unwrap().unwrap();
                    rt.drive_receipt(&h, receipt, None).await
                }
            });
            wait_until_started(&mut rx).await;
            drive.abort();
            let _ = drive.await; // JoinError: the crash
            assert_eq!(
                std::fs::read_to_string(&counter_path)
                    .unwrap()
                    .lines()
                    .count(),
                1,
                "the effect ran exactly once before the crash"
            );
            assert_eq!(h1.state().unwrap(), AgentState::ExecutingTool);
            assert_eq!(h1.pending_tool_runs().unwrap().len(), 1);
            drop(rt);
            drop(manager1);
        }
        // REOPEN: recovery must mark the run failed/unknown and NEVER
        // re-execute it. The phase-2 runtime registers a counting tool for
        // the same name: if recovery ever replayed, the counter would move.
        let attempts = Arc::new(AtomicUsize::new(0));
        let counting = Tool {
            name: "run_cmd".into(),
            description: "counts executions".into(),
            input_schema: json!({"type": "object"}),
            resource_class: ResourceClass::Cpu,
            capability: None,
            recovery_hint: RecoveryHint::UnknownEffect,
            path_args: vec![],
            execute: {
                let attempts = attempts.clone();
                Arc::new(move |_ctx, _args| {
                    let attempts = attempts.clone();
                    Box::pin(async move {
                        attempts.fetch_add(1, Ordering::SeqCst);
                        Ok(ToolOutcome {
                            text: "ran".into(),
                            exit_code: Some(0),
                            ..Default::default()
                        })
                    })
                })
            },
        };
        let manager2 = open(&dir);
        let rt2 = AgentRuntime::new(deps(
            manager2.clone(),
            &dir,
            fake(vec![ScriptedResponse::End]),
            vec![counting],
            0.65,
            None,
        ))
        .unwrap();
        let reports = rt2.recover().unwrap();
        let report = reports.iter().find(|r| r.session_id == session).unwrap();
        assert_eq!(report.crashed_ops.len(), 1);
        assert_eq!(
            report.crashed_ops[0].status, "failed",
            "unknown-effect runs are finished failed, never deferred"
        );
        assert_eq!(
            report.crashed_ops[0].effect,
            EffectStatus::Unknown,
            "the effect stays unknown, never claimed applied"
        );
        assert_eq!(
            attempts.load(Ordering::SeqCst),
            0,
            "the tool must never be re-executed after recovery"
        );
        assert_eq!(
            std::fs::read_to_string(&counter_path)
                .unwrap()
                .lines()
                .count(),
            1,
            "the real side effect happened exactly once, total"
        );
        assert_eq!(
            count_events(&manager2, session, EventKind::ReplayStarted),
            0
        );
        let h2 = handle(&manager2, session);
        assert!(h2.pending_tool_runs().unwrap().is_empty());
        assert_eq!(h2.state().unwrap(), AgentState::FailedRecoverable);
    }
}
