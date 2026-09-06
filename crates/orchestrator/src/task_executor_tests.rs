//! Adversarial tests of the TaskExecutor (audits P0-20/21/23/61/90/91).
//!
//! Children are driven by a REAL `AgentRuntime` over a REAL
//! `SessionManager` with scripted providers (no network). The tests break
//! the invariants: byte-parity of the single-item path versus the daemon's
//! own direct prompt drive, dispatch of multi-item runs onto real child
//! sessions, refusal of new runs over live crash residue, refusal of a
//! second concurrent orchestrated run (the runtime executes one run at a
//! time), and `resume_run` re-attaching a failed run through the durable
//! Retry row (applied exactly once). The wave-12 runtime tests already
//! prove the queue crash windows; here the EXECUTOR-level continuation is
//! exercised.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use faktor_agent::{AgentDeps, AgentRuntime, NoEvidence, PermissionRequester, ToolRegistry};
use faktor_core::capability::PermissionDecision;
use faktor_core::id::{SessionId, TaskId, WorktreeId};
use faktor_core::model::ModelCapabilities;
use faktor_core::time::SystemClock;
use faktor_provider::{
    FakeProvider, Provider, ProviderChunk, ProviderError, ProviderRegistry, ProviderStream,
    ScriptedResponse,
};
use faktor_session::SessionManager;

use crate::caps::{CapabilityGrant, CapabilitySet, LatticeCap, ScopePattern};
use crate::runtime::task_executor::{
    TaskExecutor, TaskRunMode, TaskRunRequest, TaskRunRow, TASK_RUN_ROW_KIND,
};
use crate::runtime::{CrashSeam, OrchestratorRuntime};
use crate::{OwnershipModel, TaskPlan, WorkItem, WorkKind};

// ------------------------------------------------------------------ fixture

struct AlwaysAllow;
impl PermissionRequester for AlwaysAllow {
    fn request(
        &self,
        _session: SessionId,
        _permission: &faktor_session::ops::PermissionRequest,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = faktor_core::Result<PermissionDecision>> + Send>,
    > {
        Box::pin(async { Ok(PermissionDecision::Allow) })
    }
}

/// Per-call scripted provider: serves one script per stream call (extra
/// calls end immediately), counts calls. Deterministic failure-then-recovery
/// runs need per-call scripts.
struct PerCallProvider {
    inner: FakeProvider,
    calls: StdMutex<Vec<Vec<ScriptedResponse>>>,
    script_index: AtomicUsize,
    request_count: AtomicUsize,
}

impl PerCallProvider {
    fn new(
        id: &str,
        caps: ModelCapabilities,
        per_call_scripts: Vec<Vec<ScriptedResponse>>,
    ) -> Self {
        Self {
            inner: FakeProvider::new(id, caps),
            calls: StdMutex::new(per_call_scripts),
            script_index: AtomicUsize::new(0),
            request_count: AtomicUsize::new(0),
        }
    }

    fn count(&self) -> usize {
        self.request_count.load(Ordering::SeqCst)
    }
}

impl Provider for PerCallProvider {
    fn id(&self) -> &str {
        "fake"
    }
    fn capabilities(&self, model: &str) -> ModelCapabilities {
        self.inner.capabilities(model)
    }
    fn stream(&self, _req: faktor_provider::GenericAgentRequest) -> ProviderStream {
        use futures::StreamExt;
        self.request_count.fetch_add(1, Ordering::SeqCst);
        let i = self.script_index.fetch_add(1, Ordering::SeqCst);
        let script: Vec<ScriptedResponse> = self
            .calls
            .lock()
            .unwrap()
            .get(i)
            .cloned()
            .unwrap_or_else(|| vec![ScriptedResponse::End]);
        let stream = futures::stream::iter(script).map(|s| match s {
            ScriptedResponse::Text(t) => Ok(ProviderChunk::Text { text: t }),
            ScriptedResponse::ToolCall { id, name, input } => Ok(ProviderChunk::ToolCall {
                id,
                name,
                input,
                complete: true,
            }),
            ScriptedResponse::Die(e) => Err(e),
            ScriptedResponse::End => Ok(ProviderChunk::Done),
            ScriptedResponse::Reasoning(_) => unreachable!("no reasoning scripts"),
        });
        Box::pin(stream)
    }
}

/// Provider whose streams park until the gate opens (mid-flight windows).
struct GatedProvider {
    caps: ModelCapabilities,
    gate: Arc<tokio::sync::Notify>,
    open: Arc<AtomicUsize>,
    request_count: AtomicUsize,
}

impl GatedProvider {
    fn open(&self) {
        self.open.store(1, Ordering::SeqCst);
        self.gate.notify_waiters();
    }
    fn count(&self) -> usize {
        self.request_count.load(Ordering::SeqCst)
    }
}

impl Provider for GatedProvider {
    fn id(&self) -> &str {
        "fake"
    }
    fn capabilities(&self, _model: &str) -> ModelCapabilities {
        self.caps.clone()
    }
    fn stream(&self, _req: faktor_provider::GenericAgentRequest) -> ProviderStream {
        use futures::StreamExt;
        self.request_count.fetch_add(1, Ordering::SeqCst);
        let open = self.open.clone();
        let gate = self.gate.clone();
        let s = futures::stream::once(async move {
            while open.load(Ordering::SeqCst) == 0 {
                gate.notified().await;
            }
            Ok(ProviderChunk::Text {
                text: "gated".into(),
            })
        })
        .chain(futures::stream::once(async { Ok(ProviderChunk::Done) }));
        Box::pin(s)
    }
}

/// One real executor environment (mirrors the runtime test env).
struct Env {
    manager: Arc<SessionManager>,
    agent: Arc<AgentRuntime>,
    provider: Arc<PerCallProvider>,
    orchestrator: Arc<OrchestratorRuntime>,
    executor: Arc<TaskExecutor>,
    parent: SessionId,
    owner_root: std::path::PathBuf,
    isolated_root: std::path::PathBuf,
}

fn read_caps() -> CapabilitySet {
    CapabilitySet::from_grants(vec![CapabilityGrant::new(
        LatticeCap::ReadWorkspace,
        ScopePattern::new("*").unwrap(),
    )])
    .unwrap()
}

fn open_env(root: &std::path::Path, scripts: Vec<Vec<ScriptedResponse>>) -> Arc<Env> {
    let manager = SessionManager::open(root.join("store"), root.join("cas"), true).unwrap();
    let caps = ModelCapabilities {
        tools: true,
        parallel_tools: true,
        ..Default::default()
    };
    let provider = Arc::new(PerCallProvider::new("fake", caps, scripts));
    let mut registry = ProviderRegistry::new();
    registry.register(provider.clone());
    let agent = build_agent(manager.clone(), registry);
    let owner_root = root.join("owner");
    std::fs::create_dir_all(&owner_root).unwrap();
    let ws = manager
        .create_workspace(owner_root.to_str().unwrap())
        .unwrap();
    let wt = WorktreeId::new(
        manager
            .put_worktree(ws, owner_root.to_str().unwrap(), "main")
            .unwrap() as u64,
    );
    let parent = manager
        .create_session(ws, "task-owner", "fake", "m")
        .unwrap()
        .id();
    manager.adopt_identity(parent, wt, TaskId::new(1)).unwrap();
    let isolated_root = root.join("isolated");
    std::fs::create_dir_all(&isolated_root).unwrap();
    let orchestrator = OrchestratorRuntime::new(manager.clone(), agent.clone());
    let executor = TaskExecutor::new(&orchestrator, manager.clone(), agent.clone());
    Arc::new(Env {
        manager,
        agent,
        provider,
        orchestrator,
        executor,
        parent,
        owner_root,
        isolated_root,
    })
}

fn build_agent(manager: Arc<SessionManager>, registry: ProviderRegistry) -> Arc<AgentRuntime> {
    AgentRuntime::new(AgentDeps {
        session: manager.clone(),
        providers: Arc::new(registry),
        chunk_sink: None,
        permission_requester: Arc::new(AlwaysAllow),
        evidence: Arc::new(NoEvidence),
        tools: Arc::new(ToolRegistry::new()),
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
        instructions: "You are a test agent.".into(),
        clock: Arc::new(SystemClock),
        tool_call_mode: faktor_agent::ToolCallMode::Native,
        tool_deadline_ms: 5000,
        retry_policy: faktor_core::retry::RetryPolicy::default(),
    })
    .unwrap()
}

fn wi(id: &str, kind: WorkKind, deps: &[&str]) -> WorkItem {
    WorkItem {
        id: id.to_string(),
        summary: format!("work {id}"),
        depends_on: deps.iter().map(|d| d.to_string()).collect(),
        kind,
        acceptance_checks: vec![],
        completion: crate::WorkState::Pending,
    }
}

fn request(goal: &str, items: Vec<WorkItem>, env: &Env) -> TaskRunRequest {
    TaskRunRequest {
        goal: goal.to_string(),
        work_items: items,
        parent_caps: read_caps(),
        isolated_root: env.isolated_root.clone(),
        ..Default::default()
    }
}

async fn wait_until(mut cond: impl FnMut() -> bool, timeout_secs: u64) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(timeout_secs);
    while !cond() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "wait_until timed out after {timeout_secs}s"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

fn state_of(env: &Env, sid: SessionId) -> faktor_core::state::AgentState {
    env.manager
        .get_session(sid)
        .unwrap()
        .unwrap()
        .state()
        .unwrap()
}

fn done_script() -> Vec<Vec<ScriptedResponse>> {
    vec![
        vec![ScriptedResponse::Text("done".into()), ScriptedResponse::End],
        vec![ScriptedResponse::End],
    ]
}

// ------------------------------------------------------------------- tests

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn single_item_task_matches_the_direct_prompt_path_byte_for_byte() {
    let dir = tempfile::tempdir().unwrap();
    // Executor-driven session A versus the direct daemon drive on session
    // B: same provider scripts, same goal — on SEPARATE stores (two
    // managers must never share one SQLite file). The TaskExecutor must
    // produce the same durable op record + session outcome — a wrapper
    // around the one drive path, never a second architecture.
    let env_a = open_env(&dir.path().join("a"), done_script());
    let env_b = open_env(&dir.path().join("b"), done_script());
    let goal = "analyze the module boundaries";

    // A: through the TaskExecutor (single item = the one-work-item plan).
    let req_a = request(goal, vec![wi("a1", WorkKind::Analysis, &[])], &env_a);
    let receipt_a = env_a
        .executor
        .start_task(env_a.parent, req_a)
        .expect("single-item start");
    assert_eq!(receipt_a.mode, TaskRunMode::InSession);
    assert!(!receipt_a.queued);
    let op_a = receipt_a.op_id.expect("real op id");

    // B: the previous direct path (agent.submit + detached drive).
    let receipt_b = env_b
        .agent
        .submit(env_b.parent, goal, &[])
        .expect("direct submit");
    assert!(!receipt_b.queued);
    let handle_b = env_b.manager.get_session(env_b.parent).unwrap().unwrap();
    let receipt_b2 = receipt_b.clone();
    let agent_b = env_b.agent.clone();
    tokio::spawn(async move {
        let _ = agent_b.drive_receipt(&handle_b, receipt_b2, None).await;
    });

    wait_until(
        || state_of(&env_a, env_a.parent) == faktor_core::state::AgentState::ReadyForNextTurn,
        30,
    )
    .await;
    wait_until(
        || state_of(&env_b, env_b.parent) == faktor_core::state::AgentState::ReadyForNextTurn,
        30,
    )
    .await;

    // The same durable outcome on both sides.
    let ha = env_a.manager.get_session(env_a.parent).unwrap().unwrap();
    let hb = env_b.manager.get_session(env_b.parent).unwrap().unwrap();
    assert_eq!(
        ha.message_count().unwrap(),
        hb.message_count().unwrap(),
        "same message stream as the direct path"
    );
    let rec_a = ha.turn_record(op_a).unwrap().unwrap();
    let rec_b = hb.turn_record(receipt_b.op_id).unwrap().unwrap();
    assert_eq!(rec_a.status, rec_b.status);
    assert_eq!(rec_a.effective_provider, rec_b.effective_provider);
    assert_eq!(rec_a.effective_model, rec_b.effective_model);

    // TaskExecutor extras: the durable task row exists on A and matches
    // the direct path's row on B (the daemon's end-of-turn content sync
    // converges the row's goal to the session ledger — the run's own goal
    // is preserved in the linkage row).
    let task_a = ha.get_task(TaskId::new(1)).unwrap().expect("task row");
    let task_b = hb.get_task(TaskId::new(1)).unwrap().expect("task row");
    assert_eq!(task_a.goal, task_b.goal);
    assert_eq!(task_a.budget, task_b.budget);
    let facts = ha.memory_facts().unwrap();
    let run_row = facts
        .iter()
        .find(|(kind, key, _)| kind == TASK_RUN_ROW_KIND && key == &receipt_a.run_id)
        .expect("durable linkage row");
    let decoded = TaskRunRow::decode(&run_row.2).unwrap();
    assert_eq!(decoded.op_id, Some(op_a.raw()));
    assert_eq!(decoded.mode, TaskRunMode::InSession);
    assert_eq!(
        decoded.goal, goal,
        "the run's own goal lives in the linkage row"
    );
    assert_eq!(decoded.item_ids, vec!["a1".to_string()]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn multi_item_task_spawns_real_children_and_completes() {
    let dir = tempfile::tempdir().unwrap();
    let env = open_env(dir.path(), done_script());
    let req = request(
        "split the analysis",
        vec![
            wi("a", WorkKind::Analysis, &[]),
            wi("b", WorkKind::Analysis, &["a"]),
        ],
        &env,
    );
    let receipt = env
        .executor
        .start_task(env.parent, req)
        .expect("multi-item start");
    assert_eq!(receipt.mode, TaskRunMode::Orchestrated);
    assert_eq!(receipt.op_id, None);
    assert!(receipt.run_id.starts_with("run-"));

    // Real children appear under the run and drive to terminal success.
    wait_until(
        || {
            OrchestratorRuntime::registry_rows(env.manager.clone(), env.parent, &receipt.run_id)
                .map(|rows| rows.len() == 2 && rows.iter().all(|c| c.state.is_terminal()))
                .unwrap_or(false)
        },
        60,
    )
    .await;
    let rows = OrchestratorRuntime::registry_rows(env.manager.clone(), env.parent, &receipt.run_id)
        .unwrap();
    assert_eq!(rows.len(), 2);
    for c in &rows {
        assert_eq!(c.state, crate::ChildState::Done);
        assert_ne!(c.session_id, 0, "real child session");
        assert_ne!(c.operation_id, 0, "child drive recorded its op");
        assert_eq!(
            c.ownership,
            faktor_session::child::ChildOwnership::ReadOnlyShared
        );
    }
    // Read-only children share the OWNER worktree (no isolated worktrees).
    assert_eq!(rows[0].worktree_id, rows[1].worktree_id);
    let owner_wt = env
        .manager
        .get_session(env.parent)
        .unwrap()
        .unwrap()
        .row()
        .unwrap()
        .worktree_id;
    assert_eq!(rows[0].worktree_id, owner_wt.raw());
    // Both children were really driven (provider calls >= 2) and the
    // executor slot freed itself.
    assert!(env.provider.count() >= 2, "driven {}", env.provider.count());
    assert!(env.executor.active_run().is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn start_refuses_when_a_live_run_was_left_by_a_crashed_executor() {
    let dir = tempfile::tempdir().unwrap();
    let env = open_env(dir.path(), done_script());
    let parent_row = env
        .manager
        .get_session(env.parent)
        .unwrap()
        .unwrap()
        .row()
        .unwrap();
    // Crashed-executor simulation: execute_task with the BeforeDrive seam
    // leaves a durable child row in Running state under the plan row.
    let plan = TaskPlan {
        goal: "crashed run".into(),
        non_goals: vec![],
        constraints: vec![],
        work_items: vec![wi("a", WorkKind::Analysis, &[])],
        ownership: OwnershipModel::NoWrites,
    };
    let mut spec = crate::runtime::ChildSpec::new("a");
    spec.child_caps = read_caps();
    spec.task_caps = read_caps();
    let config = crate::runtime::ExecConfig {
        run_id: "run-crash".into(),
        ceilings: crate::runtime::Ceilings::default(),
        parent_caps: read_caps(),
        provider: "fake".into(),
        default_model: "m".into(),
        isolated_root: env.isolated_root.clone(),
        crash_seam: Some(CrashSeam::BeforeDrive),
    };
    let res = tokio::time::timeout(
        Duration::from_secs(30),
        env.orchestrator.execute_task(
            plan,
            crate::runtime::OwnerContext {
                parent_session: env.parent,
                workspace_id: parent_row.workspace_id.raw(),
                worktree_id: parent_row.worktree_id.raw(),
                root: env.owner_root.clone(),
            },
            config,
            &[spec],
        ),
    )
    .await
    .expect("the seam fires fast");
    assert!(
        matches!(res, Err(crate::runtime::ExecError::InjectedCrashSeam(_))),
        "{res:?}"
    );

    // A new task on the same session must REFUSE (typed Conflict naming the
    // live run) instead of clobbering the mirror of the crashed one.
    let req = request(
        "new task over crash residue",
        vec![
            wi("x", WorkKind::Analysis, &[]),
            wi("y", WorkKind::Analysis, &["x"]),
        ],
        &env,
    );
    let err = env
        .executor
        .start_task(env.parent, req.clone())
        .expect_err("live residue blocks a new run");
    assert!(
        matches!(err, crate::runtime::ExecError::Conflict(_)),
        "{err:?}"
    );
    assert!(err.to_string().contains("run-crash"), "{err}");

    // resume_run re-attaches the crashed run and drives it to completion.
    env.executor
        .resume_run(
            env.parent,
            "run-crash",
            crate::runtime::Ceilings::default(),
            read_caps(),
            None,
        )
        .expect("resume accepted");
    wait_until(
        || {
            OrchestratorRuntime::registry_rows(env.manager.clone(), env.parent, "run-crash")
                .map(|rows| {
                    rows.first()
                        .is_some_and(|c| c.state == crate::ChildState::Done)
                })
                .unwrap_or(false)
        },
        60,
    )
    .await;

    // After the resumed run is terminal the session accepts new tasks.
    let receipt = env
        .executor
        .start_task(env.parent, req)
        .expect("new run after resume");
    assert_eq!(receipt.mode, TaskRunMode::Orchestrated);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn second_orchestrated_run_is_refused_while_one_is_active() {
    let dir = tempfile::tempdir().unwrap();
    // A gate provider keeps the first run mid-flight so the single
    // execution slot is observably occupied.
    let manager =
        SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
    let gated = Arc::new(GatedProvider {
        caps: ModelCapabilities {
            tools: true,
            ..Default::default()
        },
        gate: Arc::new(tokio::sync::Notify::new()),
        open: Arc::new(AtomicUsize::new(0)),
        request_count: AtomicUsize::new(0),
    });
    let mut registry = ProviderRegistry::new();
    registry.register(gated.clone());
    let agent = build_agent(manager.clone(), registry);
    let owner_root = dir.path().join("owner");
    std::fs::create_dir_all(&owner_root).unwrap();
    let ws = manager
        .create_workspace(owner_root.to_str().unwrap())
        .unwrap();
    let wt = WorktreeId::new(
        manager
            .put_worktree(ws, owner_root.to_str().unwrap(), "main")
            .unwrap() as u64,
    );
    let parent = manager
        .create_session(ws, "gated", "fake", "m")
        .unwrap()
        .id();
    manager.adopt_identity(parent, wt, TaskId::new(1)).unwrap();
    let isolated = dir.path().join("isolated");
    std::fs::create_dir_all(&isolated).unwrap();
    let orchestrator = OrchestratorRuntime::new(manager.clone(), agent.clone());
    let executor = TaskExecutor::new(&orchestrator, manager.clone(), agent.clone());

    let req = || TaskRunRequest {
        goal: "gated run".into(),
        work_items: vec![
            wi("a", WorkKind::Analysis, &[]),
            wi("b", WorkKind::Analysis, &[]),
        ],
        parent_caps: read_caps(),
        isolated_root: isolated.clone(),
        ..Default::default()
    };
    let first = executor
        .start_task(parent, req())
        .expect("first run starts");
    // Wait until the first child exists and its drive is parked on the gate
    // (mid-flight).
    wait_until(
        || {
            OrchestratorRuntime::registry_rows(manager.clone(), parent, &first.run_id)
                .map(|rows| !rows.is_empty())
                .unwrap_or(false)
        },
        30,
    )
    .await;
    wait_until(|| gated.count() >= 1, 30).await;

    // A second orchestrated start is refused while the first is active
    // (typed Conflict — the runtime executes one run at a time).
    let err = executor
        .start_task(parent, req())
        .expect_err("busy refusal");
    assert!(
        matches!(err, crate::runtime::ExecError::Conflict(_)),
        "{err:?}"
    );
    assert!(err.to_string().contains(&first.run_id), "{err}");
    // resume_run of the ACTIVE run is refused too (no double drive).
    let err2 = executor
        .resume_run(
            parent,
            &first.run_id,
            crate::runtime::Ceilings::default(),
            read_caps(),
            None,
        )
        .expect_err("double drive refused");
    assert!(err2.to_string().contains("already being driven"), "{err2}");

    // Releasing the gate lets the first run finish and free the slot.
    gated.open();
    wait_until(
        || {
            OrchestratorRuntime::registry_rows(manager.clone(), parent, &first.run_id)
                .map(|rows| rows.iter().all(|c| c.state.is_terminal()))
                .unwrap_or(false)
        },
        60,
    )
    .await;
    assert!(executor.active_run().is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resume_run_retries_a_failed_child_from_a_durable_row() {
    let dir = tempfile::tempdir().unwrap();
    // First provider stream dies permanently; the retry's re-drive succeeds.
    let scripts: Vec<Vec<ScriptedResponse>> = vec![
        vec![ScriptedResponse::Die(ProviderError::new(
            faktor_provider::ProviderErrorKind::Malformed,
            "injected permanent failure",
        ))],
        vec![
            ScriptedResponse::Text("recovered".into()),
            ScriptedResponse::End,
        ],
        vec![ScriptedResponse::End],
    ];
    let env = open_env(dir.path(), scripts);
    // Two items keep the dispatch ORCHESTRATED (a real child session) while
    // only the "a" item actually spawns: "auto" completes without a child.
    let mut req = request(
        "failing run",
        vec![
            wi("a", WorkKind::Analysis, &[]),
            wi("auto", WorkKind::Analysis, &[]),
        ],
        &env,
    );
    // "auto" completes without a child; only "a" spawns.
    req.auto_items = vec!["auto".to_string()];
    let receipt = env
        .executor
        .start_task(env.parent, req.clone())
        .expect("start");
    // Wait for the run's first drive to fail (permanent provider error).
    let mut waited = 0;
    loop {
        let rows =
            OrchestratorRuntime::registry_rows(env.manager.clone(), env.parent, &receipt.run_id)
                .unwrap_or_default();
        if rows
            .first()
            .is_some_and(|c| c.state == crate::ChildState::Failed)
        {
            break;
        }
        waited += 1;
        assert!(waited < 240, "run1 never Failed: {rows:?}");
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    // resume_run WITHOUT a pending Retry row does NOT blindly re-run: the
    // child stays Failed (never an automatic infinite retry loop).
    env.executor
        .resume_run(
            env.parent,
            &receipt.run_id,
            crate::runtime::Ceilings::default(),
            read_caps(),
            None,
        )
        .expect("resume accepted");
    tokio::time::sleep(Duration::from_millis(300)).await;
    let rows = OrchestratorRuntime::registry_rows(env.manager.clone(), env.parent, &receipt.run_id)
        .unwrap();
    assert_eq!(rows[0].state, crate::ChildState::Failed);
    // A human retry enqueues the durable Retry row; resume_run admits the
    // re-drive exactly once and completes.
    env.orchestrator
        .retry_child("child-0")
        .expect("retry enqueued");
    env.executor
        .resume_run(
            env.parent,
            &receipt.run_id,
            crate::runtime::Ceilings::default(),
            read_caps(),
            None,
        )
        .expect("retry resume");
    wait_until(
        || {
            OrchestratorRuntime::registry_rows(env.manager.clone(), env.parent, &receipt.run_id)
                .map(|rows| {
                    rows.first()
                        .is_some_and(|c| c.state == crate::ChildState::Done)
                })
                .unwrap_or(false)
        },
        60,
    )
    .await;
    let rows = OrchestratorRuntime::registry_rows(env.manager.clone(), env.parent, &receipt.run_id)
        .unwrap();
    assert_eq!(rows[0].state, crate::ChildState::Done);
    let session = env
        .manager
        .get_session(SessionId::new(rows[0].session_id))
        .unwrap()
        .unwrap();
    let ctl = session.orchestrator_ctl_all().unwrap();
    let retry = ctl
        .iter()
        .find(|r| matches!(r.control, faktor_session::child::ChildControl::Retry))
        .expect("durable retry row");
    assert!(
        retry.applied(),
        "retry decision durable before the re-drive"
    );
    // A second resume of a fully terminal run is refused (nothing to drive).
    let err = env
        .executor
        .resume_run(
            env.parent,
            &receipt.run_id,
            crate::runtime::Ceilings::default(),
            read_caps(),
            None,
        )
        .expect_err("nothing to resume");
    assert!(err.to_string().contains("nothing to resume"), "{err}");
}

#[test]
fn hostile_requests_are_rejected_before_any_write() {
    let dir = tempfile::tempdir().unwrap();
    let manager =
        SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
    let ws = manager.create_workspace("/w").unwrap();
    let parent = manager
        .create_session(ws, "hostile", "fake", "m")
        .unwrap()
        .id();
    let agent = build_agent(manager.clone(), ProviderRegistry::new());
    let orch = OrchestratorRuntime::new(manager.clone(), agent.clone());
    let executor = TaskExecutor::new(&orch, manager.clone(), agent.clone());
    let isolated = dir.path().join("isolated");

    let mut req = TaskRunRequest {
        isolated_root: isolated.clone(),
        work_items: vec![wi("a", WorkKind::Analysis, &[])],
        ..Default::default()
    };

    req.goal = "  ".into();
    let err = req.validate().expect_err("blank goal rejected");
    assert!(err.to_string().contains("goal is empty"));

    req.goal = "x".repeat(crate::MAX_GOAL_CHARS + 1);
    assert!(matches!(
        req.validate().expect_err("overlong goal rejected"),
        crate::runtime::ExecError::Oversized(_)
    ));

    req.goal = "fine".into();
    req.work_items = vec![];
    let err = req.validate().expect_err("no work items rejected");
    assert!(err.to_string().contains("at least one work item"));

    req.work_items = vec![wi("a", WorkKind::Analysis, &[])];
    req.model = Some("m".repeat(129));
    assert!(matches!(
        req.validate().expect_err("overlong model rejected"),
        crate::runtime::ExecError::Oversized(_)
    ));
    req.model = None;

    // A single MUTATING item is legal: it drives the session's own
    // worktree (the current normal path), never a spawn.
    req.work_items = vec![wi("a", WorkKind::Implementation, &[])];
    req.validate()
        .expect("single mutating item = the session's own drive");
    // ... but a multi-item plan needs an isolated_root for mutating work.
    req.work_items = vec![
        wi("a", WorkKind::Implementation, &[]),
        wi("b", WorkKind::Implementation, &["a"]),
    ];
    req.isolated_root = std::path::PathBuf::new();
    let err = req
        .validate()
        .expect_err("mutating multi-item needs isolated root");
    assert!(err.to_string().contains("isolated_root"), "{err}");
    req.isolated_root = isolated.clone();
    // Mixed read-only + mutating items are not a valid plan.
    req.work_items = vec![
        wi("a", WorkKind::Implementation, &[]),
        wi("b", WorkKind::Analysis, &["a"]),
    ];
    let err = req.validate().expect_err("mixed kinds rejected");
    assert!(err.to_string().contains("read-only work item"), "{err}");

    // Unknown session: typed NotFound, nothing written.
    let ok = TaskRunRequest {
        goal: "ok".into(),
        work_items: vec![wi("a", WorkKind::Analysis, &[])],
        isolated_root: isolated,
        ..Default::default()
    };
    let err = executor
        .start_task(faktor_core::id::SessionId::new(9999), ok.clone())
        .unwrap_err();
    assert!(matches!(err, crate::runtime::ExecError::NotFound(_)));

    // An orchestrated child session refuses task starts.
    let child = manager
        .create_child_session(
            parent,
            ws,
            WorktreeId::new(1),
            TaskId::new(1),
            "fake",
            "m",
            "child",
            faktor_session::child::ChildOwnership::ReadOnlyShared,
        )
        .unwrap();
    let err = executor.start_task(child.id(), ok).unwrap_err();
    assert!(
        matches!(err, crate::runtime::ExecError::InvalidState(_)),
        "{err:?}"
    );
}
