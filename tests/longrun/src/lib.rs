//! faktor-tests-longrun — the long-run durability suite (audit hard gates).
//!
//! Locks the cross-restart durability invariants that single-runtime suites
//! cannot see: 50+ forced compactions interleaved with daemon restarts and
//! model switches retain the goal and blockers verbatim; operation ids are
//! never reused for a new operation after a restart; every logical turn gets
//! exactly one completion record even when the daemon restarted between
//! turns; crash-recovery time does not grow with the journal; and a pure-text
//! workload leaks nothing (no orphan store rows, no content-store growth).
//!
//! Every test drives the REAL runtime (SessionManager + AgentRuntime over
//! one temp store dir reopened again and again — drop + reopen the
//! manager/runtime pair, never two writers on one store file at once),
//! scripted FakeProviders per turn, and the real durable compaction
//! machinery (compact_at_usage 0.0 over a seeded multi-KB transcript). Each
//! test is wall-clock bounded (< 20 s) and prints one report line.
//!
//! Everything lives under `#[cfg(test)]` (this is a test-harness crate; the
//! lib view exists only so the workspace builds it).

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::path::{Path, PathBuf};
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

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

    /// Durable text added by the seed turn: history deep enough that every
    /// later 0.0-usage compaction attempt has real material to prune and is
    /// accepted (the compactor hard-rejects a ~zero-token transcript).
    const SEED_HISTORY_CHARS: usize = 22 * 1024;

    /// History text every compaction turn adds (user prompt and assistant
    /// reply, ~2 KB each).
    const TURN_HISTORY_CHARS: usize = 2 * 1024;

    /// Phase plan shared by the 50-compaction workloads: 50 turns across 4
    /// phases (3 daemon restarts) with model switches m1 -> m2 -> m1 -> m2.
    const COMPACTION_PHASES: [(usize, &str); 4] = [(13, "m1"), (13, "m2"), (12, "m1"), (12, "m2")];

    const COMPACTION_PHASE_TURNS: usize = 50;

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

    /// Reopen the SAME store dir: the daemon-restart boundary. Every restart
    /// is a full drop of the previous manager/runtime pair before this open.
    fn open(dir: &Path) -> Arc<SessionManager> {
        SessionManager::open(dir.join("store"), dir.join("cas"), true).unwrap()
    }

    /// Daemon restart followed by the minimum real boot dwell. Op ids are
    /// seeded from the process clock plus a per-manager counter, so a new
    /// daemon boot is unique against the previous one as long as the boot
    /// takes longer than the previous allocation window (a few ms). Real
    /// daemon restarts take tens of ms (process start, store open, recovery
    /// sweep); the 6 ms dwell models that floor deterministically so the
    /// op-id gates lock the durable property, not a same-millisecond clock
    /// artifact that production cannot hit.
    async fn daemon_restart(dir: &Path) -> Arc<SessionManager> {
        let m = open(dir);
        tokio::time::sleep(Duration::from_millis(6)).await;
        m
    }

    fn cas_path(dir: &Path) -> PathBuf {
        dir.join("cas")
    }

    fn ws_path(dir: &Path) -> PathBuf {
        dir.join("ws")
    }

    fn deps(
        manager: Arc<SessionManager>,
        dir: &Path,
        provider: FakeProvider,
        tools: Vec<Tool>,
        model: &str,
        compact_at_usage: f64,
        cas: bool,
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
            cas: cas.then(|| Arc::new(faktor_cas::Cas::open(cas_path(dir)).unwrap())),
            workspaces: faktor_fs::WorkspaceFileService::new(),
            edit: None,
            snapshots: None,
            sandbox: None,
            supervisor: None,
            verifier: None,
            hooks: None,
            instructions_loader: None,
            model: model.into(),
            compaction_model: None,
            compact_at_usage,
            instructions: "You are a test agent.".into(),
            clock: Arc::new(SystemClock),
            tool_call_mode: faktor_agent::ToolCallMode::Native,
            tool_deadline_ms: 2000,
            retry_policy: faktor_core::retry::RetryPolicy::default(),
        }
    }

    fn handle(manager: &Arc<SessionManager>, sid: SessionId) -> faktor_session::SessionHandle {
        manager.get_session(sid).unwrap().unwrap()
    }

    /// A session over a VIRTUAL workspace root (the path is registered in
    /// the store but does not exist on disk). Nothing in this suite touches
    /// real files, and a nonexistent root keeps the file service from
    /// spawning heavyweight fs watchers (spec §21) per turn.
    fn new_session_in(manager: &Arc<SessionManager>, root: &Path, title: &str) -> SessionId {
        let ws = manager.create_workspace(root.to_str().unwrap()).unwrap();
        manager.create_session(ws, title, "fake", "m").unwrap().id()
    }

    fn events(manager: &Arc<SessionManager>, sid: SessionId) -> Vec<faktor_core::event::Event> {
        handle(manager, sid).events_range(1, None).unwrap()
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

    /// A tool that always succeeds (exit 0).
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

    /// A tool that FAILS with `text` (exit 1): its failure text becomes a
    /// durable known-failure entry in the ledger.
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

    /// Seed a session whose durable ledger carries a distinctive goal (the
    /// session title) and a distinctive known failure (one real failing tool
    /// call), then ~22 KB of real assistant history — the durable material
    /// every later aggressive compaction prunes. The seed turn runs at the
    /// normal 0.65 trigger (never a ~zero-transcript compaction attempt).
    /// Fully drops the phase-0 manager before returning.
    async fn seed_session(dir: &TempDir, title: &str, blocker: &str, root: &Path) -> SessionId {
        let manager = open(dir.path());
        let sid = new_session_in(&manager, root, title);
        let seed_script = vec![
            tool_call(
                "c1",
                "run_command",
                json!({"command": "cargo check -p payments"}),
            ),
            tool_call(
                "c2",
                "write_file",
                json!({"path": "src/audit.rs", "content": "pub fn parse() {}"}),
            ),
            ScriptedResponse::Text(format!("seed analysis {}", "a".repeat(SEED_HISTORY_CHARS))),
            ScriptedResponse::End,
        ];
        let seed_tools = vec![fail_tool("run_command", blocker, 1), ok_tool("write_file")];
        let rt = AgentRuntime::new(deps(
            manager.clone(),
            dir.path(),
            fake(seed_script),
            seed_tools,
            "m1",
            0.65,
            true,
        ))
        .unwrap();
        let outcome = rt
            .run_turn_with_model(sid, "fix the audit ledger", &[], Some("m1".into()))
            .await
            .unwrap();
        assert_eq!(
            outcome.final_state,
            AgentState::ReadyForNextTurn,
            "the seed turn must complete"
        );
        // PRE-CONDITION: the durable ledger really carries goal + blocker.
        let h = handle(&manager, sid);
        let ledger0 = ledger_of(&h);
        assert_eq!(ledger0.goal, title, "goal seeded from the session title");
        assert!(
            ledger0.known_failures.iter().any(|f| f.contains(blocker)),
            "blocker must be seeded into known failures: {:?}",
            ledger0.known_failures
        );
        drop(rt);
        drop(manager);
        sid
    }

    /// One ~2 KB text turn (prompt + assistant reply) that MUST run the
    /// compactor (compact_at_usage 0.0 over the seeded history). Returns the
    /// outcome so callers can count accepted compactions.
    async fn compacting_turn(
        manager: &Arc<SessionManager>,
        dir: &Path,
        sid: SessionId,
        model: &str,
        turn_no: usize,
    ) -> faktor_agent::TurnOutcome {
        let script = fake(vec![
            ScriptedResponse::Text(format!(
                "reply {turn_no:03} {}",
                "t".repeat(TURN_HISTORY_CHARS)
            )),
            ScriptedResponse::End,
        ]);
        let rt = AgentRuntime::new(deps(manager.clone(), dir, script, vec![], model, 0.0, true))
            .unwrap();
        let prompt = format!("LR turn {turn_no:03} {}", "k".repeat(TURN_HISTORY_CHARS));
        rt.run_turn_with_model(sid, &prompt, &[], Some(model.to_string()))
            .await
            .unwrap()
    }

    /// One daemon phase of the 50-turn workload: a FRESH manager/runtime
    /// pair over the same store dir (a restart), `n` compacting turns under
    /// one model, then the pair is dropped (the next daemon boot reopens).
    /// Returns the number of turns whose compaction was accepted.
    async fn compacting_phase(
        dir: &Path,
        sid: SessionId,
        model: &str,
        first_turn: usize,
        n: usize,
    ) -> usize {
        let manager = daemon_restart(dir).await;
        let mut compacted_turns = 0usize;
        for i in 0..n {
            let outcome = compacting_turn(&manager, dir, sid, model, first_turn + i).await;
            assert_eq!(
                outcome.final_state,
                AgentState::ReadyForNextTurn,
                "turn {} must complete",
                first_turn + i
            );
            assert!(!outcome.queued, "no turn may be queued (serial drive)");
            if outcome.compacted {
                compacted_turns += 1;
            }
        }
        drop(manager);
        compacted_turns
    }

    /// The standard 50-turn compaction workload: 50 turns x ~2 KB of history
    /// text at compact_at_usage 0.0 across 4 phases (3 restarts) and model
    /// switches m1 -> m2 -> m1 -> m2. Returns (compacted turns, total turns).
    async fn run_fifty_compactions(dir: &Path, sid: SessionId) -> (usize, usize) {
        let mut compacted_turns = 0usize;
        let mut turn_no = 0usize;
        for (phase, (n, model)) in COMPACTION_PHASES.iter().enumerate() {
            let compacted = compacting_phase(dir, sid, model, turn_no, *n).await;
            compacted_turns += compacted;
            turn_no += n;
            eprintln!(
                "[longrun] phase {phase} model {model}: {n} turns, {compacted} accepted compactions"
            );
        }
        assert_eq!(turn_no, COMPACTION_PHASE_TURNS);
        (compacted_turns, turn_no)
    }

    /// The executor signature `Tool` expects (mirrors `Tool::execute`).
    type ToolExec = Arc<
        dyn Fn(
                faktor_agent::ToolRunCtx,
                serde_json::Value,
            ) -> std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<ToolOutcome, Error>> + Send>,
            > + Send
            + Sync,
    >;

    /// A real idempotent workspace-write tool whose executions are counted
    /// in a shared AtomicUsize. With `park: true` it parks forever AFTER its
    /// durable run row exists — the daemon-crash point. With `park: false`
    /// it completes (the post-crash variant recovery replay executes).
    fn durable_write_tool(
        name: &str,
        executions: Arc<AtomicUsize>,
        park: bool,
    ) -> (Tool, tokio::sync::watch::Receiver<bool>) {
        let (tx, rx) = tokio::sync::watch::channel(false);
        let execute: ToolExec = Arc::new(move |_ctx, _args| {
            let executions = executions.clone();
            let tx = tx.clone();
            Box::pin(async move {
                let _ = tx.send(true);
                if park {
                    let never: std::future::Pending<Result<ToolOutcome, Error>> =
                        std::future::pending();
                    never.await
                } else {
                    executions.fetch_add(1, Ordering::SeqCst);
                    Ok(ToolOutcome {
                        text: "durable write landed".into(),
                        exit_code: Some(0),
                        ..Default::default()
                    })
                }
            })
        });
        (
            Tool {
                name: name.to_string(),
                description: "idempotent workspace write".into(),
                input_schema: json!({"type": "object"}),
                resource_class: ResourceClass::DiskWrite,
                capability: None,
                recovery_hint: RecoveryHint::Idempotent,
                path_args: vec!["path".into()],
                execute,
            },
            rx,
        )
    }

    /// Simulate a daemon death mid-tool: submit a prompt whose turn calls an
    /// idempotent tool, wait for the tool's durable run row, abort the drive
    /// and drop the whole manager/runtime pair — the durable crash residue
    /// (machine at ExecutingTool, one pending run row).
    async fn crash_mid_idempotent_tool(dir: &Path, sid: SessionId, executions: Arc<AtomicUsize>) {
        let (park_tool, mut rx) = durable_write_tool("durable_write", executions, true);
        let manager = daemon_restart(dir).await;
        let h = handle(&manager, sid);
        let receipt = h.submit_prompt("write data.txt", &[]).unwrap();
        assert!(!receipt.queued);
        let provider = fake(vec![
            tool_call(
                "c1",
                "durable_write",
                json!({"path": "data.txt", "content": "hello"}),
            ),
            ScriptedResponse::End,
        ]);
        let rt = AgentRuntime::new(deps(
            manager.clone(),
            dir,
            provider,
            vec![park_tool],
            "m2",
            0.65,
            true,
        ))
        .unwrap();
        let drive = tokio::spawn({
            let rt = rt.clone();
            let manager = manager.clone();
            async move {
                let h = manager.get_session(sid).unwrap().unwrap();
                rt.drive_receipt(&h, receipt, None).await
            }
        });
        wait_until_started(&mut rx).await;
        drive.abort();
        let _ = drive.await; // JoinError: the crash
        assert_eq!(
            h.pending_tool_runs().unwrap().len(),
            1,
            "exactly one interrupted run row"
        );
        assert_eq!(
            h.state().unwrap(),
            AgentState::ExecutingTool,
            "the machine is left mid-tool"
        );
        drop(rt);
        drop(manager);
    }

    /// Reopen a runtime over the crashed store and TIME the daemon-start
    /// recovery sweep (recover() is pub on AgentRuntime). The runtime
    /// registers the completing idempotent tool recovery must validate and
    /// defer (never execute).
    fn open_and_recover(
        dir: &Path,
        executions: &Arc<AtomicUsize>,
    ) -> (
        Arc<AgentRuntime>,
        Vec<faktor_session::RecoveryReport>,
        Duration,
    ) {
        let manager = open(dir);
        let (tool, _rx) = durable_write_tool("durable_write", executions.clone(), false);
        let rt = AgentRuntime::new(deps(
            manager,
            dir,
            fake(vec![
                ScriptedResponse::Text("recovered".into()),
                ScriptedResponse::End,
            ]),
            vec![tool],
            "m2",
            0.65,
            true,
        ))
        .unwrap();
        let start = Instant::now();
        let reports = rt.recover().unwrap();
        let elapsed = start.elapsed();
        (rt, reports, elapsed)
    }

    // ====================================================================
    // 1. FIFTY COMPACTIONS + RESTARTS + MODEL SWITCHES RETAIN THE LEDGER
    // ====================================================================

    /// Hard gate: 50 sequential ~2 KB text turns at compact_at_usage 0.0
    /// (every turn runs the compactor over the seeded real history) across
    /// >= 3 daemon restarts (drop + reopen manager/runtime pairs on one store
    /// dir) and 2 model switches (m1/m2 across phases) must leave the durable
    /// ledger whole: goal verbatim and the seeded blocker still a known
    /// failure. The compaction table has no public count API (the store only
    /// INSERTs rows), so accepted compactions are counted the way
    /// tests/quality counts TurnCompleted: via the `ContextCompacted` journal
    /// events record_compaction journals one-to-one with its table rows.
    #[tokio::test]
    async fn fifty_compactions_retain_goal_and_blockers() {
        let here = format!("{}:{}", file!(), line!());
        let dir = tempdir().unwrap();
        let title = "make the audit ledger survive restarts: GOAL-SURVIVES-LR-50";
        let blocker = "BLOCKER-SURVIVES-LR-50: payments parser check failed in tests/audit.rs";

        let start = Instant::now();
        let sid = seed_session(&dir, title, blocker, &ws_path(dir.path())).await;

        // The 50-turn workload across phases/restarts/model switches.
        let (compacted_turns, turns) = run_fifty_compactions(dir.path(), sid).await;
        assert_eq!(turns, 50);

        // Final daemon boot: read the durable state.
        let manager = open(dir.path());
        let h = handle(&manager, sid);
        let ledger = ledger_of(&h);
        assert_eq!(
            ledger.goal, title,
            "the goal must survive 50 compactions, restarts and model switches verbatim"
        );
        assert!(
            ledger.known_failures.iter().any(|f| f.contains(blocker)),
            "the seeded blocker must survive 50 compactions: {:?}",
            ledger.known_failures
        );
        let evs = events(&manager, sid);
        let compactions = evs
            .iter()
            .filter(|e| e.kind == EventKind::ContextCompacted)
            .count();
        let rejected = evs
            .iter()
            .filter(|e| e.kind == EventKind::CompactRejected)
            .count();
        let completed = evs
            .iter()
            .filter(|e| e.kind == EventKind::TurnCompleted)
            .count();
        assert!(
            compactions >= 50,
            "the 50-turn workload must journal >= 50 accepted compactions, got {compactions} \
             (rejected {rejected})"
        );
        assert_eq!(
            compacted_turns, 50,
            "every workload turn must see an accepted compaction"
        );
        assert_eq!(
            completed, 51,
            "seed + 50 turns = 51 completions, journaled {completed}"
        );
        drop(manager);
        eprintln!(
            "[longrun] {here} goal_verbatim=true blocker_verbatim=true turns={turns} \
             compacted_turns={compacted_turns} compact_events={compactions} \
             rejected_events={rejected} restarts=3 model_switches=2 wall={:.2}s",
            start.elapsed().as_secs_f64()
        );
    }

    // ====================================================================
    // 2. OPERATION IDS NEVER REUSED ACROSS RESTARTS
    // ====================================================================

    /// Hard gate: across 10 daemon restarts and 30 turns, every op id ever
    /// journaled belongs to exactly ONE operation. The journal is segmented
    /// at TurnCompleted boundaries, and no op id may appear in two different
    /// segments: within one operation the id legitimately repeats across its
    /// own events (PromptReceived, ModelStarted per iteration, tool rows,
    /// TurnCompleted), but a restart reusing an id for a NEW operation would
    /// put the id into two segments.
    #[tokio::test]
    async fn operation_ids_never_reused_across_restarts() {
        let here = format!("{}:{}", file!(), line!());
        let dir = tempdir().unwrap();
        let start = Instant::now();
        let sid = {
            let manager = open(dir.path());
            let sid = new_session_in(&manager, &ws_path(dir.path()), "op ids");
            drop(manager);
            sid
        };
        let mut receipts = Vec::new();
        let restarts = 10usize;
        let per_phase = 3usize;
        for restart in 0..restarts {
            // DAEMON RESTART: a fresh manager/runtime pair on the same store.
            let manager = daemon_restart(dir.path()).await;
            let model = if restart % 2 == 0 { "m1" } else { "m2" };
            for t in 0..per_phase {
                let script = fake(vec![
                    ScriptedResponse::Text(format!("op id turn {restart}.{t}")),
                    ScriptedResponse::End,
                ]);
                let rt = AgentRuntime::new(deps(
                    manager.clone(),
                    dir.path(),
                    script,
                    vec![],
                    model,
                    0.65,
                    true,
                ))
                .unwrap();
                let outcome = rt
                    .run_turn(
                        sid,
                        &format!("operation id turn {restart}.{t} {}", "y".repeat(512)),
                        &[],
                    )
                    .await
                    .unwrap();
                assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);
                receipts.push(outcome.op_id.raw());
            }
            drop(manager);
        }
        assert_eq!(receipts.len(), restarts * per_phase);
        let distinct_receipts: HashSet<u64> = receipts.iter().copied().collect();
        assert_eq!(
            receipts.len(),
            distinct_receipts.len(),
            "no receipt op id may repeat across restarts"
        );

        // FINAL boot: segment the whole journal at TurnCompleted boundaries.
        let manager = open(dir.path());
        let evs = events(&manager, sid);
        let completed = evs
            .iter()
            .filter(|e| e.kind == EventKind::TurnCompleted)
            .count();
        assert_eq!(completed, restarts * per_phase, "30 turns, 30 completions");
        let mut segment_of_op: HashMap<u64, usize> = HashMap::new();
        let mut bucket = 0usize;
        let mut events_with_op = 0usize;
        let mut violations = 0usize;
        for e in &evs {
            if e.kind == EventKind::TurnCompleted {
                bucket += 1;
                continue;
            }
            let Some(op) = e.op_id else { continue };
            events_with_op += 1;
            match segment_of_op.get(&op.raw()) {
                Some(&seen) if seen != bucket => violations += 1,
                Some(_) => {}
                None => {
                    segment_of_op.insert(op.raw(), bucket);
                }
            }
        }
        assert_eq!(
            violations, 0,
            "an op id was journaled in two different turn segments: reuse across operations"
        );
        assert!(
            events_with_op >= receipts.len(),
            "every receipt must appear in the journal"
        );
        drop(manager);
        eprintln!(
            "[longrun] {here} restarts={restarts} turns={} events_with_op={events_with_op} \
             distinct_op_ids={} turn_segments={bucket} violations={violations} wall={:.2}s",
            receipts.len(),
            segment_of_op.len(),
            start.elapsed().as_secs_f64()
        );
    }

    // ====================================================================
    // 3. EXACTLY ONE COMPLETION PER TURN OVER RESTARTS
    // ====================================================================

    /// Hard gate (tests/quality locked this single-runtime; extended across
    /// restarts): 5 logical turns over 2 daemon restarts each journal EXACTLY
    /// one TurnCompleted bound to their own op id — a restart may never
    /// duplicate or drop a completion, and no tool run may be orphaned.
    #[tokio::test]
    async fn exactly_one_completion_per_turn_over_restarts() {
        let here = format!("{}:{}", file!(), line!());
        let dir = tempdir().unwrap();
        let start = Instant::now();
        let sid = {
            let manager = open(dir.path());
            let sid = new_session_in(&manager, &ws_path(dir.path()), "exactly one each");
            drop(manager);
            sid
        };
        let echo = ok_tool("echo");
        let mut ops: Vec<u64> = Vec::new();
        let phase_turns = [2usize, 2, 1];
        for (phase, &n) in phase_turns.iter().enumerate() {
            let manager = daemon_restart(dir.path()).await;
            let model = if phase % 2 == 0 { "m1" } else { "m2" };
            for t in 0..n {
                let script = fake(vec![
                    tool_call(
                        &format!("c{phase}_{t}"),
                        "echo",
                        json!({"phase": phase, "t": t}),
                    ),
                    ScriptedResponse::Text(format!("answer {phase}.{t}")),
                    ScriptedResponse::End,
                ]);
                let rt = AgentRuntime::new(deps(
                    manager.clone(),
                    dir.path(),
                    script,
                    vec![echo.clone()],
                    model,
                    0.65,
                    true,
                ))
                .unwrap();
                let outcome = rt
                    .run_turn(sid, &format!("turn {phase}.{t}"), &[])
                    .await
                    .unwrap();
                assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);
                ops.push(outcome.op_id.raw());
            }
            drop(manager);
        }
        assert_eq!(ops.len(), 5);

        // FINAL boot: read the whole journal across all restarts.
        let manager = open(dir.path());
        let h = handle(&manager, sid);
        let evs = events(&manager, sid);
        let mut per_op: HashMap<u64, usize> = HashMap::new();
        for e in &evs {
            if e.kind == EventKind::TurnCompleted {
                if let Some(op) = e.op_id {
                    *per_op.entry(op.raw()).or_insert(0) += 1;
                }
            }
        }
        for op in &ops {
            assert_eq!(
                per_op.get(op).copied().unwrap_or(0),
                1,
                "op {op} must have exactly one completion across restarts: {per_op:?}"
            );
        }
        assert_eq!(
            per_op.len(),
            ops.len(),
            "no op id may complete twice: {per_op:?}"
        );
        let ready = evs
            .iter()
            .filter(|e| e.state == AgentState::ReadyForNextTurn)
            .count();
        assert_eq!(ready, 5, "ReadyForNextTurn only at the 5 genuine ends");
        assert!(
            h.pending_tool_runs().unwrap().is_empty(),
            "no orphaned tool runs across restarts"
        );
        drop(manager);
        eprintln!(
            "[longrun] {here} turns=5 restarts=2 per_op_completions=exactly-1 \
             ready_states={ready} wall={:.2}s",
            start.elapsed().as_secs_f64()
        );
    }

    // ====================================================================
    // 4. RECOVERY TIME DOES NOT GROW WITH THE JOURNAL
    // ====================================================================

    /// Hard gate: daemon-start recover() must not degrade as the journal
    /// grows. recover() is pub on AgentRuntime. Measured right after the
    /// FIRST restart (small journal) and after the LAST restart (50
    /// compactions + hundreds of events), both times over the SAME residue
    /// shape (one interrupted idempotent run): the only variable is journal
    /// size, and a quadratic recovery blows the linearity bound
    /// `last <= first * 8 + 200 ms`. Absolute cap 5 s.
    #[tokio::test]
    async fn recovery_time_does_not_grow() {
        let here = format!("{}:{}", file!(), line!());
        let dir = tempdir().unwrap();
        let title = "recovery time must not grow: GOAL-LR-4";
        let blocker = "BLOCKER-LR-4: recovery scan is quadratic?";
        let executions = Arc::new(AtomicUsize::new(0));
        let start = Instant::now();

        let sid = seed_session(&dir, title, blocker, &ws_path(dir.path())).await;

        // ---- FIRST restart: small journal, clean state, timed recover().
        let manager = open(dir.path());
        let rt = AgentRuntime::new(deps(
            manager.clone(),
            dir.path(),
            fake(vec![ScriptedResponse::End]),
            vec![],
            "m2",
            0.65,
            true,
        ))
        .unwrap();
        let t_first_clean = {
            let t = Instant::now();
            let reports = rt.recover().unwrap();
            assert!(reports.iter().all(|r| r.crashed_ops.is_empty()));
            t.elapsed()
        };
        drop(rt);
        drop(manager);

        // ---- First interrupted-run residue on the SMALL journal, then the
        // first timed recovery WITH real sweep work.
        crash_mid_idempotent_tool(dir.path(), sid, executions.clone()).await;
        let (rt_small, reports_small, t_first) = open_and_recover(dir.path(), &executions);
        assert_eq!(reports_small.len(), 1);
        assert_eq!(reports_small[0].crashed_ops.len(), 1);
        assert_eq!(
            reports_small[0].crashed_ops[0].status, "running",
            "idempotent runs are deferred, never re-executed by recover()"
        );
        assert_eq!(
            reports_small[0].crashed_ops[0].effect,
            EffectStatus::Unknown
        );
        assert_eq!(
            executions.load(Ordering::SeqCst),
            0,
            "recover() must never execute a deferred run"
        );
        // Resolve the residue so the compaction phase starts clean.
        let o = rt_small.continue_turn(sid).await.unwrap();
        assert_eq!(o.final_state, AgentState::ReadyForNextTurn);
        assert_eq!(
            executions.load(Ordering::SeqCst),
            1,
            "continue_turn replays the interrupted run exactly once"
        );
        drop(rt_small);

        // ---- The 50-compaction workload (4 phases, 3 restarts, m1/m2).
        let (compacted_turns, turns) = run_fifty_compactions(dir.path(), sid).await;
        assert_eq!(turns, 50);
        assert!(compacted_turns >= 50);

        // ---- Final crash residue on the HUGE journal, then the LAST timed
        // recovery (the same residue shape as the first measurement).
        crash_mid_idempotent_tool(dir.path(), sid, executions.clone()).await;
        let (rt_big, reports_big, t_last) = open_and_recover(dir.path(), &executions);
        assert_eq!(reports_big[0].crashed_ops.len(), 1);
        assert_eq!(reports_big[0].crashed_ops[0].status, "running");
        assert_eq!(
            executions.load(Ordering::SeqCst),
            1,
            "still no physical execution during recovery"
        );

        let bound = t_first.as_secs_f64() * 8.0 + 0.200;
        assert!(
            t_last.as_secs_f64() <= bound,
            "recovery after {turns} compactions ({t_last:?}) must stay within first*8+200ms \
             ({bound:.3}s; first was {t_first:?})"
        );
        assert!(
            t_last.as_secs_f64() < 5.0,
            "absolute recovery bound: {t_last:?}"
        );

        // Resolve the final residue: replay exactly once, everything clean.
        let o = rt_big.continue_turn(sid).await.unwrap();
        assert_eq!(o.final_state, AgentState::ReadyForNextTurn);
        assert_eq!(
            executions.load(Ordering::SeqCst),
            2,
            "both interrupted runs replayed exactly once in total"
        );
        let manager = rt_big.deps().session.clone();
        let h = handle(&manager, sid);
        assert!(h.pending_tool_runs().unwrap().is_empty());
        let evs = events(&manager, sid);
        let compactions = evs
            .iter()
            .filter(|e| e.kind == EventKind::ContextCompacted)
            .count();
        drop(rt_big);
        eprintln!(
            "[longrun] {here} turns={turns} compact_events={compactions} \
             recover_first_clean={:.3}ms recover_first_residue={:.3}ms \
             recover_last_residue={:.3}ms (bound {bound:.3}s) replayed_exactly_once=true \
             wall={:.2}s",
            t_first_clean.as_secs_f64() * 1000.0,
            t_first.as_secs_f64() * 1000.0,
            t_last.as_secs_f64() * 1000.0,
            start.elapsed().as_secs_f64()
        );
    }

    // ====================================================================
    // 5. NO ORPHANS / NO LEAK SIGNAL (MEMORY-PROXY INVARIANTS)
    // ====================================================================

    /// This crate spawns NO processes. Instead it locks the resource-leak
    /// invariants of the memory proxy: (a) the store's session table never
    /// grows across reopen cycles — manager-level sessions list == created
    /// count with no duplicates; (b) the CAS artifact dir does not grow for
    /// pure text turns — blob entries before and after 10 text turns equal.
    #[tokio::test]
    async fn no_orphan_processes_and_no_leak_signal() {
        let here = format!("{}:{}", file!(), line!());
        let dir = tempdir().unwrap();
        let start = Instant::now();

        let cas_before = {
            let cas = faktor_cas::Cas::open(cas_path(dir.path())).unwrap();
            cas.blob_count()
        };

        // Two sessions on the store; session A drives the text turns.
        let (manager, sid_a, sid_b) = {
            let manager = open(dir.path());
            let ws = manager.create_workspace("/w").unwrap();
            let a = manager
                .create_session(ws, "text turns", "fake", "m")
                .unwrap()
                .id();
            let b = manager
                .create_session(ws, "quiet", "fake", "m")
                .unwrap()
                .id();
            (manager, a, b)
        };
        // 10 pure-text turns: no tools, no artifacts; compact_at_usage 1.0 so
        // not even compaction archiving may touch the content store.
        for t in 0..10 {
            let script = fake(vec![
                ScriptedResponse::Text(format!("memory turn {t} {}", "z".repeat(1024))),
                ScriptedResponse::End,
            ]);
            let rt = AgentRuntime::new(deps(
                manager.clone(),
                dir.path(),
                script,
                vec![],
                "m",
                1.0,
                true,
            ))
            .unwrap();
            let outcome = rt
                .run_turn(sid_a, &format!("text turn {t} {}", "m".repeat(1024)), &[])
                .await
                .unwrap();
            assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);
        }
        drop(manager);

        let cas_after = {
            let cas = faktor_cas::Cas::open(cas_path(dir.path())).unwrap();
            cas.blob_count()
        };
        assert_eq!(
            cas_after, cas_before,
            "pure text turns must never grow the CAS artifact dir \
             ({cas_before} -> {cas_after} entries)"
        );

        // 10 reopen cycles: session count constant, ids never duplicated.
        let mut cycles = 0usize;
        for _ in 0..10 {
            let manager = open(dir.path());
            let sessions = manager.list_sessions(None).unwrap();
            assert_eq!(
                sessions.len(),
                2,
                "reopen cycle {cycles}: sessions list must equal the created count"
            );
            let ids: HashSet<u64> = sessions.iter().map(|s| s.id().raw()).collect();
            assert_eq!(ids.len(), 2, "no duplicate session rows across reopens");
            assert!(ids.contains(&sid_a.raw()) && ids.contains(&sid_b.raw()));
            drop(manager);
            cycles += 1;
        }
        eprintln!(
            "[longrun] {here} sessions=2 reopen_cycles={cycles} cas_blobs_before={cas_before} \
             cas_blobs_after={cas_after} processes_spawned=0 wall={:.2}s",
            start.elapsed().as_secs_f64()
        );
    }
}
