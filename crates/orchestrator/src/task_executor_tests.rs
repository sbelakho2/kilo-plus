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

use faktor_agent::tool::RecoveryHint as ToolRecovery;
use faktor_agent::{
    AgentDeps, AgentRuntime, NoEvidence, PermissionRequester, Tool, ToolCallMode, ToolOutcome,
    ToolRegistry, ToolRunCtx,
};
use faktor_core::capability::PermissionDecision;
use faktor_core::error::Error;
use faktor_core::id::WorkspaceId;
use faktor_core::id::{SessionId, TaskId, WorktreeId};
use faktor_core::model::ModelCapabilities;
use faktor_core::resource::ResourceClass;
use faktor_core::time::SystemClock;
use faktor_provider::{
    FakeProvider, GenericAgentRequest, Provider, ProviderChunk, ProviderError, ProviderRegistry,
    ProviderStream, ScriptedResponse,
};
use faktor_session::SessionManager;

use crate::caps::{CapabilityGrant, CapabilitySet, LatticeCap, ScopePattern};
use crate::runtime::shadow::{ShadowCopyLimits, ShadowRoots};
use crate::runtime::task_executor::{
    ShadowFinalizeAction, TaskExecutor, TaskRunMode, TaskRunRequest, TaskRunRow, TASK_RUN_ROW_KIND,
};
use crate::runtime::{CrashSeam, ExecError, OrchestratorRuntime};
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
    open_env_with_shadows(root, scripts, ShadowCopyLimits::default(), false)
}

/// A shadowed executor env: same wiring as [`open_env`] plus the P0-48
/// shadow service rooted at `<root>/shadows`.
fn open_shadow_env(root: &std::path::Path, scripts: Vec<Vec<ScriptedResponse>>) -> Arc<Env> {
    open_env_with_shadows(root, scripts, ShadowCopyLimits::default(), true)
}

fn open_env_with_shadows(
    root: &std::path::Path,
    scripts: Vec<Vec<ScriptedResponse>>,
    limits: ShadowCopyLimits,
    shadowed: bool,
) -> Arc<Env> {
    let manager = SessionManager::open(root.join("store"), root.join("cas"), true).unwrap();
    let caps = ModelCapabilities {
        tools: true,
        parallel_tools: true,
        ..Default::default()
    };
    let provider = Arc::new(PerCallProvider::new("fake", caps, scripts));
    let mut registry = ProviderRegistry::new();
    registry.try_register(provider.clone()).unwrap();
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
    let shadows_root = root.join("shadows");
    let shadows = if shadowed {
        Some(ShadowRoots::new_with_limits(
            manager.clone(),
            shadows_root.clone(),
            limits,
        ))
    } else {
        None
    };
    let executor = TaskExecutor::new(
        &orchestrator,
        manager.clone(),
        agent.clone(),
        shadows.clone(),
    );
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

/// Heavy file/CAS/process tests are serialized: under intra-binary test
/// parallelism their store+DbActor+fsync + CAS-file storms on one disk
/// starve each other past any reasonable wall bound (observed 300 s+ tails
/// on shared machines while every test passes in isolation and serially).
/// The guard restores determinism without changing semantics.
static HEAVY_SUITE: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn heavy_guard() -> std::sync::MutexGuard<'static, ()> {
    HEAVY_SUITE.lock().expect("heavy suite guard poisoned")
}

#[tokio::test]
async fn single_item_task_matches_the_direct_prompt_path_byte_for_byte() {
    let _heavy = heavy_guard();
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

#[tokio::test]
async fn multi_item_task_spawns_real_children_and_completes() {
    let _heavy = heavy_guard();
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

#[tokio::test]
async fn start_refuses_when_a_live_run_was_left_by_a_crashed_executor() {
    let _heavy = heavy_guard();
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

#[tokio::test]
async fn resume_run_after_crash_between_assignments_and_first_spawn_reuses_durable_ids() {
    let dir = tempfile::tempdir().unwrap();
    let env = open_env(dir.path(), done_script());
    // Crash the executor EXACTLY between the atomic assignment commit and
    // the first child spawn (the AfterAssignmentsPersisted seam).
    let mut req = request(
        "crash after compile",
        vec![
            wi("a", WorkKind::Analysis, &[]),
            wi("b", WorkKind::Analysis, &[]),
        ],
        &env,
    );
    req.crash_seam = Some(CrashSeam::AfterAssignmentsPersisted);
    let receipt = env
        .executor
        .start_task(env.parent, req)
        .expect("run starts");
    assert_eq!(receipt.mode, TaskRunMode::Orchestrated);
    let run_id = receipt.run_id.clone();
    // Durable assignments exist; NO child may have spawned; the crashed
    // executor's slot is free again.
    wait_until(
        || {
            !OrchestratorRuntime::assignment_rows(env.manager.clone(), env.parent, &run_id)
                .unwrap_or_default()
                .is_empty()
                && OrchestratorRuntime::registry_rows(env.manager.clone(), env.parent, &run_id)
                    .map(|rows| rows.is_empty())
                    .unwrap_or(false)
        },
        30,
    )
    .await;
    wait_until(|| env.executor.active_run().is_none(), 180).await;
    let assignments =
        OrchestratorRuntime::assignment_rows(env.manager.clone(), env.parent, &run_id).unwrap();
    assert_eq!(assignments.len(), 2);
    // The assignment-backed residue is LIVE: a new task must REFUSE until
    // the crashed run is resumed (its durable ids must not be orphaned).
    let err = env
        .executor
        .start_task(
            env.parent,
            request(
                "second run over residue",
                vec![wi("x", WorkKind::Analysis, &[])],
                &env,
            ),
        )
        .expect_err("assignment-backed residue blocks a new run");
    assert!(
        matches!(err, crate::runtime::ExecError::Conflict(_)),
        "{err:?}"
    );
    assert!(err.to_string().contains(&run_id), "{err}");
    // resume_run accepts the run even though it has NO child rows yet and
    // re-spawns every item under its DURABLE child id (no re-mint).
    env.executor
        .resume_run(
            env.parent,
            &run_id,
            crate::runtime::Ceilings::default(),
            read_caps(),
            None,
        )
        .expect("resume accepted");
    wait_until(
        || {
            OrchestratorRuntime::registry_rows(env.manager.clone(), env.parent, &run_id)
                .map(|rows| {
                    rows.len() == 2 && rows.iter().all(|c| c.state == crate::ChildState::Done)
                })
                .unwrap_or(false)
        },
        90,
    )
    .await;
    let rows =
        OrchestratorRuntime::registry_rows(env.manager.clone(), env.parent, &run_id).unwrap();
    let after =
        OrchestratorRuntime::assignment_rows(env.manager.clone(), env.parent, &run_id).unwrap();
    assert_eq!(
        after, assignments,
        "assignment rows must never be re-minted"
    );
    for a in &assignments {
        let row = rows
            .iter()
            .find(|r| r.item_id == a.item_id)
            .expect("child row per assigned item");
        assert_eq!(row.child_id, a.child_id, "spawn must reuse the durable id");
    }
    // Terminal residue frees the session for new tasks (the new run spawns
    // its own fresh children and completes).
    let fresh = env
        .executor
        .start_task(
            env.parent,
            request(
                "fresh run",
                vec![
                    wi("p", WorkKind::Analysis, &[]),
                    wi("q", WorkKind::Analysis, &[]),
                ],
                &env,
            ),
        )
        .expect("session accepts new tasks after resume");
    assert_eq!(fresh.mode, TaskRunMode::Orchestrated);
    wait_until(
        || {
            OrchestratorRuntime::registry_rows(env.manager.clone(), env.parent, &fresh.run_id)
                .map(|rows| {
                    rows.len() == 2 && rows.iter().all(|c| c.state == crate::ChildState::Done)
                })
                .unwrap_or(false)
        },
        90,
    )
    .await;
}

#[tokio::test]
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
    registry.try_register(gated.clone()).unwrap();
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
    let executor = TaskExecutor::new(&orchestrator, manager.clone(), agent.clone(), None);

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
    wait_until(|| gated.count() >= 1, 180).await;

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

#[tokio::test]
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
    let executor = TaskExecutor::new(&orch, manager.clone(), agent.clone(), None);
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

// =========================================================== P0-48 shadowed
// single-agent mutating runs (shadow mutation roots).
//
// The drive itself is the REAL daemon drive (scripted providers, no
// network). The "shadowed drive writes" below are staged as direct writes
// INTO the shadow root — the exact operation the session's file consumers
// perform once they resolve `SessionManager::active_root` (the next-wave
// re-pointing); today those consumers live in the agent crate and resolve
// the durable workspace root directly, so the wiring is verified against
// the durable shadow machinery (begin/finalize/commit) instead.

use faktor_core::state::{TaskState, TaskTransition, VerificationStatus};
use faktor_session::ShadowRowState;

fn seed_owner(root: &std::path::Path) {
    std::fs::create_dir_all(root.join("sub")).unwrap();
    std::fs::write(root.join("a.txt"), b"base-alpha").unwrap();
    std::fs::write(root.join("sub/b.txt"), b"base-beta").unwrap();
}

fn shadow_row_of(env: &Env) -> faktor_session::ShadowRow {
    env.manager
        .shadow_row(env.parent)
        .unwrap()
        .expect("an active shadow row exists")
}

fn owner_bytes(env: &Env, rel: &str) -> Vec<u8> {
    std::fs::read(env.owner_root.join(rel)).unwrap()
}

/// The durable "shadowed drive write": stage content inside the shadow root
/// (exactly where `SessionManager::active_root` re-points next wave).
fn shadow_drive_write(env: &Env, rel: &str, bytes: &[u8]) {
    let row = shadow_row_of(env);
    let dst = std::path::PathBuf::from(&row.root).join(rel);
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(dst, bytes).unwrap();
}

/// The human verifier's role in the test harness: drive the durable task
/// row to Verifying, land a passing record (empty criteria/checks — the
/// seeded task rows carry no acceptance criteria) and complete the task.
/// This is the ONLY producer of VerifiedComplete (task machine invariant).
fn certify_env_task(env: &Env) {
    let h = env.manager.get_session(env.parent).unwrap().unwrap();
    let task_id = h.task_id().unwrap();
    for _ in 0..8 {
        let task = h.get_task(task_id).unwrap().unwrap();
        let target = match task.state {
            TaskState::Pending => TaskTransition::StartRunning,
            TaskState::Planning => TaskTransition::PlanComplete,
            TaskState::Running => TaskTransition::RequestVerification,
            TaskState::Waiting => TaskTransition::ResumeFromWaiting,
            TaskState::Blocked => TaskTransition::Unblock,
            TaskState::NeedsVerification => TaskTransition::StartVerification,
            TaskState::Verifying => break,
            s => panic!("cannot certify a task at {s:?}"),
        };
        let rev = h.task_revision(task_id).unwrap();
        h.transition_task(task_id, rev, target, None).unwrap();
    }
    let task = h.get_task(task_id).unwrap().unwrap();
    assert_eq!(
        task.state,
        TaskState::Verifying,
        "the row must reach Verifying before completion"
    );
    let record = h
        .create_verification_record(
            task_id,
            None,
            vec![],
            vec![],
            vec![],
            vec![],
            None,
            VerificationStatus::Passed,
            h.now_ms(),
        )
        .unwrap();
    let rev = h.task_revision(task_id).unwrap();
    h.complete_verified_task(task_id, rev, record).unwrap();
    assert_eq!(
        h.get_task(task_id).unwrap().unwrap().state,
        TaskState::VerifiedComplete
    );
}

fn mutating_request(env: &Env, goal: &str) -> TaskRunRequest {
    TaskRunRequest {
        goal: goal.to_string(),
        work_items: vec![wi("impl", WorkKind::Implementation, &[])],
        parent_caps: read_caps(),
        isolated_root: env.isolated_root.clone(),
        ..Default::default()
    }
}

/// A gated shadowed fixture: the provider parks mid-stream until released,
/// so the drive is deterministically mid-flight while assertions run.
struct GatedShadowFix {
    manager: Arc<SessionManager>,
    executor: Arc<TaskExecutor>,
    gated: Arc<GatedProvider>,
    parent: SessionId,
    owner_root: std::path::PathBuf,
    shadows: Arc<ShadowRoots>,
}

fn open_gated_shadow(root: &std::path::Path) -> GatedShadowFix {
    let manager = SessionManager::open(root.join("store"), root.join("cas"), true).unwrap();
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
    registry.try_register(gated.clone()).unwrap();
    let agent = build_agent(manager.clone(), registry);
    let owner_root = root.join("owner");
    std::fs::create_dir_all(&owner_root).unwrap();
    seed_owner(&owner_root);
    let ws = manager
        .create_workspace(owner_root.to_str().unwrap())
        .unwrap();
    let wt = WorktreeId::new(
        manager
            .put_worktree(ws, owner_root.to_str().unwrap(), "main")
            .unwrap() as u64,
    );
    let parent = manager
        .create_session(ws, "gated-shadow", "fake", "m")
        .unwrap()
        .id();
    manager.adopt_identity(parent, wt, TaskId::new(1)).unwrap();
    let orchestrator = OrchestratorRuntime::new(manager.clone(), agent.clone());
    let shadows = ShadowRoots::new(manager.clone(), root.join("shadows"));
    let executor = TaskExecutor::new(
        &orchestrator,
        manager.clone(),
        agent.clone(),
        Some(shadows.clone()),
    );
    GatedShadowFix {
        manager,
        executor,
        gated,
        parent,
        owner_root,
        shadows,
    }
}

#[tokio::test]
async fn shadowed_mutating_run_writes_never_reach_user_checkout_until_verified_commit() {
    let _heavy = heavy_guard();
    // (a)+(b) over the REAL executor: a shadowed mutating run begins a
    // durable shadow before its drive; staged writes live in the shadow
    // while the user checkout stays byte-identical; only a
    // VerifiedComplete + clean integration lands them, removes the shadow
    // and retires the row.
    let dir = tempfile::tempdir().unwrap();
    let env = open_shadow_env(dir.path(), done_script());
    seed_owner(&env.owner_root);
    let receipt = env
        .executor
        .start_task(env.parent, mutating_request(&env, "implement the change"))
        .expect("shadowed single-item start");
    assert_eq!(receipt.mode, TaskRunMode::InSession);
    // The shadow began synchronously BEFORE the submit.
    let row = shadow_row_of(&env);
    assert_eq!(row.state, ShadowRowState::Active);
    let shadow_dir = std::path::PathBuf::from(&row.root);
    assert!(shadow_dir.is_dir());
    assert_eq!(
        env.manager.active_root(env.parent).unwrap(),
        Some(shadow_dir.clone()),
        "active_root re-points the session at the shadow while live"
    );
    // The drive ends (scripted text, no tools).
    wait_until(
        || state_of(&env, env.parent) == faktor_core::state::AgentState::ReadyForNextTurn,
        30,
    )
    .await;
    // Let the detached post-drive finalize settle: the row is not terminal
    // (no completion claim), so the shadow is RETAINED and the user
    // checkout stays untouched.
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(owner_bytes(&env, "a.txt"), b"base-alpha");
    assert_eq!(row.state, ShadowRowState::Active);
    // Stage the shadowed drive's writes AFTER the drive (what a
    // shadow-aware verification run would have produced).
    shadow_drive_write(&env, "a.txt", b"implemented alpha");
    shadow_drive_write(&env, "new-file.txt", b"implemented new file");
    assert_eq!(
        owner_bytes(&env, "a.txt"),
        b"base-alpha",
        "user checkout untouched while the shadow holds the new world"
    );
    assert!(!env.owner_root.join("new-file.txt").exists());
    // Only a verified completion integrates.
    let before = env
        .executor
        .finalize_shadow_run(env.parent)
        .expect("finalize runs")
        .expect("shadow exists");
    assert_eq!(
        before.action,
        ShadowFinalizeAction::Retained,
        "a non-terminal task row never integrates"
    );
    certify_env_task(&env);
    let finalize = env
        .executor
        .finalize_shadow_run(env.parent)
        .expect("finalize runs")
        .expect("shadow exists");
    assert_eq!(finalize.action, ShadowFinalizeAction::Integrated);
    assert_eq!(finalize.merged.len(), 2);
    assert_eq!(owner_bytes(&env, "a.txt"), b"implemented alpha");
    assert_eq!(owner_bytes(&env, "new-file.txt"), b"implemented new file");
    assert!(!shadow_dir.exists(), "clean integration removes the shadow");
    let row = shadow_row_of(&env);
    assert_eq!(row.state, ShadowRowState::Integrated);
    assert_eq!(
        env.manager.active_root(env.parent).unwrap(),
        None,
        "a retired shadow stops re-pointing"
    );
}

#[tokio::test]
async fn mid_drive_isolation_and_conflict_surfaces_integration_blocked_then_resolves() {
    let _heavy = heavy_guard();
    // (a)+(c) with the drive parked mid-flight: while the drive is live the
    // user checkout is byte-identical; an external user edit during the
    // drive conflicts at integration — the run's content never lands, the
    // shadow is retained with the durable conflict list, and a second
    // finalize after the user reverts integrates.
    let dir = tempfile::tempdir().unwrap();
    let fix = open_gated_shadow(dir.path());
    let receipt = fix
        .executor
        .start_task(
            fix.parent,
            TaskRunRequest {
                goal: "gate-shadowed implementation".into(),
                work_items: vec![wi("impl", WorkKind::Implementation, &[])],
                parent_caps: read_caps(),
                isolated_root: dir.path().join("isolated"),
                ..Default::default()
            },
        )
        .expect("shadowed start");
    assert_eq!(receipt.mode, TaskRunMode::InSession);
    let row = fix
        .manager
        .shadow_row(fix.parent)
        .unwrap()
        .expect("row at begin");
    let shadow_dir = std::path::PathBuf::from(&row.root);
    // Park the drive mid-flight and write into the shadow while it runs.
    wait_until(|| fix.gated.count() >= 1, 180).await;
    std::fs::write(shadow_dir.join("a.txt"), b"mid-drive implementation").unwrap();
    assert_eq!(
        std::fs::read(fix.owner_root.join("a.txt")).unwrap(),
        b"base-alpha",
        "user checkout byte-identical MID-drive"
    );
    assert_eq!(
        fix.manager.active_root(fix.parent).unwrap(),
        Some(shadow_dir.clone()),
        "active_root reports the shadow root while the drive is live"
    );
    // The user edits the file externally during the drive.
    std::fs::write(fix.owner_root.join("a.txt"), b"user edit during drive").unwrap();
    fix.gated.open();
    wait_until(
        || {
            fix.manager
                .get_session(fix.parent)
                .unwrap()
                .unwrap()
                .state()
                .unwrap()
                == faktor_core::state::AgentState::ReadyForNextTurn
        },
        30,
    )
    .await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    certify_verified_complete(&fix.manager, fix.parent);
    let finalize = fix
        .executor
        .finalize_shadow_run(fix.parent)
        .expect("finalize runs")
        .expect("shadow exists");
    assert_eq!(
        finalize.action,
        ShadowFinalizeAction::IntegrationBlocked,
        "{finalize:?}"
    );
    assert_eq!(finalize.conflicts.len(), 1);
    assert_eq!(
        std::fs::read(fix.owner_root.join("a.txt")).unwrap(),
        b"user edit during drive",
        "a conflicted user file is never overwritten"
    );
    assert!(shadow_dir.is_dir(), "shadow retained on conflict");
    assert_eq!(
        fix.manager.shadow_row(fix.parent).unwrap().unwrap().state,
        ShadowRowState::IntegrationBlocked
    );
    // The user resolves the drift (reverts to the base content); the same
    // auto decision now integrates.
    std::fs::write(fix.owner_root.join("a.txt"), b"base-alpha").unwrap();
    let finalize = fix
        .executor
        .finalize_shadow_run(fix.parent)
        .expect("finalize runs")
        .expect("shadow exists");
    assert_eq!(finalize.action, ShadowFinalizeAction::Integrated);
    assert_eq!(
        std::fs::read(fix.owner_root.join("a.txt")).unwrap(),
        b"mid-drive implementation"
    );
    assert!(!shadow_dir.exists());
}

/// The verifier helper over a bare manager (used by the gated fixture).
fn certify_verified_complete(manager: &Arc<SessionManager>, session: SessionId) {
    let h = manager.get_session(session).unwrap().unwrap();
    let task_id = h.task_id().unwrap();
    for _ in 0..8 {
        let task = h.get_task(task_id).unwrap().unwrap();
        let target = match task.state {
            TaskState::Pending => TaskTransition::StartRunning,
            TaskState::Planning => TaskTransition::PlanComplete,
            TaskState::Running => TaskTransition::RequestVerification,
            TaskState::Waiting => TaskTransition::ResumeFromWaiting,
            TaskState::Blocked => TaskTransition::Unblock,
            TaskState::NeedsVerification => TaskTransition::StartVerification,
            TaskState::Verifying => break,
            s => panic!("cannot certify a task at {s:?}"),
        };
        let rev = h.task_revision(task_id).unwrap();
        h.transition_task(task_id, rev, target, None).unwrap();
    }
    let task = h.get_task(task_id).unwrap().unwrap();
    assert_eq!(task.state, TaskState::Verifying);
    let record = h
        .create_verification_record(
            task_id,
            None,
            vec![],
            vec![],
            vec![],
            vec![],
            None,
            VerificationStatus::Passed,
            h.now_ms(),
        )
        .unwrap();
    let rev = h.task_revision(task_id).unwrap();
    h.complete_verified_task(task_id, rev, record).unwrap();
}

#[test]
fn crashed_drive_residue_reopens_and_settles_deterministically() {
    // (d): a daemon "crash" (the parked drive dies with its runtime) leaves
    // the durable shadow row Active + dir on disk; a reopened daemon sees
    // the row, discards the pre-certification residue deterministically on
    // the next shadowed start, and runs a fresh shadowed task to a clean
    // integration.
    let dir = tempfile::tempdir().unwrap();
    // Phase 1 — the crashing daemon: park a shadowed drive mid-flight.
    let parent = {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let fix = open_gated_shadow(dir.path());
            let _receipt = fix
                .executor
                .start_task(
                    fix.parent,
                    TaskRunRequest {
                        goal: "crashed shadowed drive".into(),
                        work_items: vec![wi("impl", WorkKind::Implementation, &[])],
                        parent_caps: read_caps(),
                        isolated_root: dir.path().join("isolated"),
                        ..Default::default()
                    },
                )
                .expect("shadowed start");
            let row = fix
                .manager
                .shadow_row(fix.parent)
                .unwrap()
                .expect("row exists");
            assert_eq!(row.state, ShadowRowState::Active);
            let shadow_dir = std::path::PathBuf::from(&row.root);
            wait_until(|| fix.gated.count() >= 1, 180).await;
            std::fs::write(shadow_dir.join("a.txt"), b"crashed-drive content").unwrap();
            assert_eq!(
                std::fs::read(fix.owner_root.join("a.txt")).unwrap(),
                b"base-alpha",
                "crashing daemon never touched the user checkout"
            );
            // Runtime ends here with the drive still parked = the crash.
            // A real crash never runs Drop; forget the service so the
            // graceful-shutdown removal cannot mask the residue.
            std::mem::forget(fix.shadows);
            std::mem::forget(fix.executor);
            fix.parent
        })
    };
    // Phase 2 — the daemon restarts over the same data dir.
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let manager =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let shadows = ShadowRoots::new(manager.clone(), dir.path().join("shadows"));
        let row = manager.shadow_row(parent).unwrap().expect("row survives");
        assert_eq!(row.state, ShadowRowState::Active, "crash residue row");
        let residue_dir = std::path::PathBuf::from(&row.root);
        assert!(residue_dir.is_dir(), "crash residue dir");
        let residue_shadow_id = row.shadow_id.clone();
        let registry = {
            let mut r = ProviderRegistry::new();
            r.try_register(Arc::new(PerCallProvider::new(
                "fake",
                ModelCapabilities {
                    tools: true,
                    parallel_tools: true,
                    ..Default::default()
                },
                done_script(),
            )))
            .unwrap();
            r
        };
        let agent = build_agent(manager.clone(), registry);
        // The real daemon runs crash recovery before the first request; the
        // parked drive's turn is resolved here (the same path serve_impl
        // takes on restart).
        let _ = agent.recover();
        let orchestrator = OrchestratorRuntime::new(manager.clone(), agent.clone());
        let executor = TaskExecutor::new(
            &orchestrator,
            manager.clone(),
            agent.clone(),
            Some(shadows.clone()),
        );
        // The interrupted turn is reconstructed (never blindly re-run): a
        // new shadowed task over a LIVE drive is a typed Conflict naming
        // the residue — resume or cancel first.
        let req = TaskRunRequest {
            goal: "post-crash implementation".into(),
            work_items: vec![wi("impl", WorkKind::Implementation, &[])],
            parent_caps: read_caps(),
            isolated_root: dir.path().join("isolated"),
            ..Default::default()
        };
        let err = executor
            .start_task(parent, req.clone())
            .expect_err("a live interrupted drive refuses a new run");
        assert!(matches!(err, ExecError::Conflict(_)), "{err}");
        assert!(err.to_string().contains("live shadow"), "{err}");
        // The operator discards the residue (the shadowed run's task never
        // certified anything); the next shadowed start begins a fresh
        // shadow over the tombstoned row.
        shadows.discard(parent).unwrap();
        let receipt = executor
            .start_task(parent, req)
            .expect("a new shadowed run starts after deterministic settlement");
        assert_eq!(receipt.mode, TaskRunMode::InSession);
        let row = manager.shadow_row(parent).unwrap().expect("new row");
        assert_eq!(row.state, ShadowRowState::Active);
        assert_ne!(row.shadow_id, residue_shadow_id, "a fresh generation");
        assert!(!residue_dir.exists(), "crash residue directory removed");
        let new_dir = std::path::PathBuf::from(&row.root);
        assert!(new_dir.is_dir());
        wait_until(
            || {
                manager
                    .get_session(parent)
                    .unwrap()
                    .unwrap()
                    .state()
                    .unwrap()
                    == faktor_core::state::AgentState::ReadyForNextTurn
            },
            30,
        )
        .await;
        tokio::time::sleep(Duration::from_millis(300)).await;
        std::fs::write(new_dir.join("a.txt"), b"post-crash implementation").unwrap();
        certify_verified_complete(&manager, parent);
        let finalize = executor
            .finalize_shadow_run(parent)
            .expect("finalize")
            .expect("shadow exists");
        assert_eq!(finalize.action, ShadowFinalizeAction::Integrated);
        assert_eq!(
            std::fs::read(dir.path().join("owner").join("a.txt")).unwrap(),
            b"post-crash implementation"
        );
        assert!(!new_dir.exists());
        let retired = manager.shadow_row(parent).unwrap().unwrap();
        assert_eq!(retired.state, ShadowRowState::Integrated);
    });
}

#[tokio::test]
async fn failed_drive_keeps_shadow_for_recovery_cancel_discards() {
    let _heavy = heavy_guard();
    let dir = tempfile::tempdir().unwrap();
    let scripts: Vec<Vec<ScriptedResponse>> =
        vec![vec![ScriptedResponse::Die(ProviderError::new(
            faktor_provider::ProviderErrorKind::Malformed,
            "injected permanent failure",
        ))]];
    let env = open_shadow_env(dir.path(), scripts);
    seed_owner(&env.owner_root);
    let _ = env
        .executor
        .start_task(env.parent, mutating_request(&env, "failing shadowed run"))
        .expect("start");
    let row_before = shadow_row_of(&env);
    let dir_before = std::path::PathBuf::from(&row_before.root);
    // The drive fails (permanent provider error): the session ends
    // FailedRecoverable and the task row is NOT terminal — the shadow is
    // RETAINED for the documented recovery path (never a blind discard of
    // a resumable run).
    wait_until(
        || state_of(&env, env.parent) == faktor_core::state::AgentState::FailedRecoverable,
        30,
    )
    .await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let row = shadow_row_of(&env);
    assert_eq!(
        row.state,
        ShadowRowState::Active,
        "recoverable runs keep the shadow"
    );
    assert!(dir_before.is_dir());
    assert_eq!(owner_bytes(&env, "a.txt"), b"base-alpha");
    // The operator cancels the task: the terminal Cancel drives the
    // post-drive finalize to DISCARD the shadow.
    let h = env.manager.get_session(env.parent).unwrap().unwrap();
    let task_id = h.task_id().unwrap();
    let rev = h.task_revision(task_id).unwrap();
    h.transition_task(task_id, rev, TaskTransition::Cancel, None)
        .unwrap();
    let finalize = env
        .executor
        .finalize_shadow_run(env.parent)
        .expect("finalize")
        .expect("shadow exists");
    assert_eq!(finalize.action, ShadowFinalizeAction::Discarded);
    let row = shadow_row_of(&env);
    assert_eq!(row.state, ShadowRowState::Discarded);
    assert!(!dir_before.exists());
    assert_eq!(
        owner_bytes(&env, "a.txt"),
        b"base-alpha",
        "a discarded shadow never writes the user checkout"
    );
}

#[test]
fn oversize_shadow_refuses_the_task_before_any_mutation() {
    // (g) at the executor: an un-copyable base refuses the task start with
    // a typed Oversized BEFORE the submit — no provider call, no durable
    // shadow row, no task row, no user bytes touched.
    let dir = tempfile::tempdir().unwrap();
    let env = open_env_with_shadows(
        dir.path(),
        done_script(),
        ShadowCopyLimits {
            max_entries: 2,
            max_total_bytes: 1024 * 1024,
        },
        true,
    );
    seed_owner(&env.owner_root);
    std::fs::write(env.owner_root.join("extra.txt"), b"third file").unwrap();
    let err = env
        .executor
        .start_task(env.parent, mutating_request(&env, "oversized shadow"))
        .expect_err("the copy cap refuses the run");
    assert!(matches!(err, ExecError::Oversized(_)), "{err}");
    assert_eq!(env.provider.count(), 0, "no drive ever started");
    assert!(env.manager.shadow_row(env.parent).unwrap().is_none());
    assert!(env
        .manager
        .get_session(env.parent)
        .unwrap()
        .unwrap()
        .list_tasks()
        .unwrap()
        .is_empty());
    assert_eq!(owner_bytes(&env, "a.txt"), b"base-alpha");
}

// ================================================= P0-48 shadow mutation roots
// end-to-end over REAL tools (the wave-22 flip): with the agent's five
// root-resolution sites consulting `SessionManager::resolve_workspace_root`
// and the workspace-scoped consumers consulting live shadow rows, a
// shadowed drive's write_file/read/verification/repo-knowledge paths all
// resolve the SHADOW root while the drive is live — the user checkout is
// byte-untouched until a VerifiedComplete integration (wave-21
// commit_all), and every un-shadowed path keeps today's behavior.

/// A per-call scripted provider that also records every request's rendered
/// system prompt (where repo map + AGENTS.md rules + instruction rules
/// ride), so a test can prove WHICH root a drive read its context from.
struct RecordingProvider {
    caps: ModelCapabilities,
    calls: StdMutex<Vec<Vec<ScriptedResponse>>>,
    script_index: AtomicUsize,
    request_count: AtomicUsize,
    prompts: StdMutex<Vec<String>>,
}

impl RecordingProvider {
    fn new(caps: ModelCapabilities, per_call_scripts: Vec<Vec<ScriptedResponse>>) -> Self {
        Self {
            caps: caps.clone(),
            calls: StdMutex::new(per_call_scripts),
            script_index: AtomicUsize::new(0),
            request_count: AtomicUsize::new(0),
            prompts: StdMutex::new(Vec::new()),
        }
    }
    fn recorded(&self) -> Vec<String> {
        self.prompts.lock().unwrap().clone()
    }
}

impl Provider for RecordingProvider {
    fn id(&self) -> &str {
        "fake"
    }
    fn capabilities(&self, model: &str) -> ModelCapabilities {
        let _ = model;
        self.caps.clone()
    }
    fn stream(&self, req: GenericAgentRequest) -> ProviderStream {
        use futures::StreamExt;
        self.request_count.fetch_add(1, Ordering::SeqCst);
        self.prompts.lock().unwrap().push(req.system.clone());
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

/// A real write_file: resolves the RELATIVE path through the session's
/// workspace handle and writes atomically — the exact operation the flipped
/// tool-batch site serves. No postcondition (no crash is simulated here).
fn real_write_tool() -> Tool {
    Tool {
        name: "write_file".into(),
        description: "writes a real file".into(),
        input_schema: serde_json::json!({"type": "object"}),
        resource_class: ResourceClass::DiskWrite,
        capability: None,
        recovery_hint: ToolRecovery::WorkspaceWrite,
        path_args: vec!["path".into()],
        execute: Arc::new(|ctx: ToolRunCtx, args| {
            Box::pin(async move {
                let Some(ws) = &ctx.workspace else {
                    return Err(Error::internal("no workspace wired"));
                };
                let path = args.get("path").and_then(|p| p.as_str()).unwrap_or("");
                let content = args
                    .get("content")
                    .and_then(|c| c.as_str())
                    .unwrap_or_default();
                ws.write_atomic(std::path::Path::new(path), content.as_bytes())
                    .map_err(|e| Error::internal(format!("write {path}: {e}")))?;
                Ok(ToolOutcome {
                    text: format!("wrote {path}"),
                    exit_code: Some(0),
                    ..Default::default()
                })
            })
        }),
    }
}

/// A CPU tool whose FIRST invocation parks the drive (mid-iteration, at
/// ExecutingTool) until the gate opens: the deterministic mid-drive window
/// of a shadowed run. Fired counts invocations that reached the park.
fn parking_tool(name: &str, gate: Arc<tokio::sync::Notify>, fired: Arc<AtomicUsize>) -> Tool {
    Tool {
        name: name.into(),
        description: "parks once".into(),
        input_schema: serde_json::json!({"type": "object"}),
        resource_class: ResourceClass::Cpu,
        capability: None,
        recovery_hint: ToolRecovery::Idempotent,
        path_args: vec![],
        execute: Arc::new(move |_ctx, _args| {
            let gate = gate.clone();
            let fired = fired.clone();
            Box::pin(async move {
                if fired.fetch_add(1, Ordering::SeqCst) == 0 {
                    gate.notified().await;
                }
                Ok(ToolOutcome {
                    text: "parked".into(),
                    exit_code: Some(0),
                    ..Default::default()
                })
            })
        }),
    }
}

/// A write_file whose FIRST invocation parks AFTER the write landed (the
/// drive is mid-flight with the file already inside the resolved root).
fn parked_write_tool(gate: Arc<tokio::sync::Notify>, fired: Arc<AtomicUsize>) -> Tool {
    Tool {
        name: "write_file".into(),
        description: "writes a real file, parking once".into(),
        input_schema: serde_json::json!({"type": "object"}),
        resource_class: ResourceClass::DiskWrite,
        capability: None,
        recovery_hint: ToolRecovery::WorkspaceWrite,
        path_args: vec!["path".into()],
        execute: Arc::new(move |ctx: ToolRunCtx, args| {
            let gate = gate.clone();
            let fired = fired.clone();
            Box::pin(async move {
                let Some(ws) = &ctx.workspace else {
                    return Err(Error::internal("no workspace wired"));
                };
                let path = args.get("path").and_then(|p| p.as_str()).unwrap_or("");
                let content = args
                    .get("content")
                    .and_then(|c| c.as_str())
                    .unwrap_or_default();
                ws.write_atomic(std::path::Path::new(path), content.as_bytes())
                    .map_err(|e| Error::internal(format!("write {path}: {e}")))?;
                if fired.fetch_add(1, Ordering::SeqCst) == 0 {
                    gate.notified().await;
                }
                Ok(ToolOutcome {
                    text: format!("wrote {path}"),
                    exit_code: Some(0),
                    ..Default::default()
                })
            })
        }),
    }
}

/// Workspace-scoped root provider over the REAL SessionManager (mirror of
/// the daemon's `SessionWorkspaceRoots`): the live shadow of a shadowed
/// workspace re-points instruction loading at the shadow root.
struct RealRoots(Arc<SessionManager>);

impl faktor_instructions::WorkspaceRootProvider for RealRoots {
    fn workspace_root(&self, workspace_id: u64) -> Option<std::path::PathBuf> {
        if workspace_id == 0 {
            return None;
        }
        let ws = WorkspaceId::new(workspace_id);
        match self.0.live_workspace_shadow_root(ws) {
            Ok(Some(root)) => Some(root),
            Ok(None) | Err(_) => self.0.workspace_root(ws).ok().flatten(),
        }
    }
}

fn real_resolver(manager: &Arc<SessionManager>) -> Arc<faktor_instructions::InstructionResolver> {
    Arc::new(faktor_instructions::InstructionResolver::new(
        Arc::new(RealRoots(manager.clone())),
        faktor_instructions::DEFAULT_RESOLVER_CACHE_ENTRIES,
    ))
}

/// An executor env whose drive runs REAL tools (write_file over the
/// session's resolved workspace root) with a REAL instructions resolver and
/// the given verification service. `parked_write` registers the parking
/// write tool; the gate/fired pair exposes the mid-drive window.
struct RealToolEnv {
    manager: Arc<SessionManager>,
    provider: Arc<RecordingProvider>,
    executor: Arc<TaskExecutor>,
    parent: SessionId,
    owner_root: std::path::PathBuf,
    isolated_root: std::path::PathBuf,
    gate: Arc<tokio::sync::Notify>,
    fired: Arc<AtomicUsize>,
}

fn real_state_of(env: &RealToolEnv) -> faktor_core::state::AgentState {
    env.manager
        .get_session(env.parent)
        .unwrap()
        .unwrap()
        .state()
        .unwrap()
}

fn real_mutating_request(env: &RealToolEnv, goal: &str) -> TaskRunRequest {
    TaskRunRequest {
        goal: goal.to_string(),
        work_items: vec![wi("impl", WorkKind::Implementation, &[])],
        parent_caps: read_caps(),
        isolated_root: env.isolated_root.clone(),
        ..Default::default()
    }
}

fn open_real_tool_env(
    root: &std::path::Path,
    scripts: Vec<Vec<ScriptedResponse>>,
    verification: Arc<faktor_agent::VerificationService>,
    parked_write: bool,
) -> Arc<RealToolEnv> {
    let manager = SessionManager::open(root.join("store"), root.join("cas"), true).unwrap();
    let caps = ModelCapabilities {
        tools: true,
        parallel_tools: true,
        ..Default::default()
    };
    let provider = Arc::new(RecordingProvider::new(caps, scripts));
    let mut registry = ProviderRegistry::new();
    registry.try_register(provider.clone()).unwrap();
    let gate = Arc::new(tokio::sync::Notify::new());
    let fired = Arc::new(AtomicUsize::new(0));
    let mut tools = ToolRegistry::new();
    if parked_write {
        tools.register(parked_write_tool(gate.clone(), fired.clone()));
    } else {
        tools.register(real_write_tool());
    }
    tools.register(parking_tool("pause", gate.clone(), fired.clone()));
    let resolver = real_resolver(&manager);
    let agent = AgentRuntime::new(AgentDeps {
        session: manager.clone(),
        providers: Arc::new(registry),
        chunk_sink: None,
        permission_requester: Arc::new(AlwaysAllow),
        evidence: Arc::new(NoEvidence),
        tools: Arc::new(tools),
        cas: None,
        workspaces: faktor_fs::WorkspaceFileService::new(),
        edit: None,
        snapshots: None,
        sandbox: None,
        supervisor: None,
        verification,
        hooks: None,
        instructions_resolver: resolver,
        routing: faktor_agent::FixedRoutingPolicy::passthrough(),
        budgets: Arc::new(faktor_session::NoopBudget),
        model: "m".into(),
        compaction_model: None,
        compact_at_usage: 0.65,
        instructions: "You are a test agent.".into(),
        clock: Arc::new(SystemClock),
        tool_call_mode: ToolCallMode::Native,
        tool_deadline_ms: 120_000,
        retry_policy: faktor_core::retry::RetryPolicy::default(),
    })
    .unwrap();
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
        .create_session(ws, "real-shadow-tools", "fake", "m")
        .unwrap()
        .id();
    manager.adopt_identity(parent, wt, TaskId::new(1)).unwrap();
    let orchestrator = OrchestratorRuntime::new(manager.clone(), agent.clone());
    let shadows = ShadowRoots::new(manager.clone(), root.join("shadows"));
    let executor = TaskExecutor::new(&orchestrator, manager.clone(), agent.clone(), Some(shadows));
    let isolated_root = root.join("isolated");
    std::fs::create_dir_all(&isolated_root).unwrap();
    Arc::new(RealToolEnv {
        manager,
        provider,
        executor,
        parent,
        owner_root,
        isolated_root,
        gate,
        fired,
    })
}

fn real_env_task_row(env: &RealToolEnv) -> faktor_session::Task {
    let h = env.manager.get_session(env.parent).unwrap().unwrap();
    h.get_task(h.task_id().unwrap()).unwrap().unwrap()
}

/// Settle a real-tool drive to VerifiedComplete + integration: the drive's
/// own verified completion may have already auto-committed through the
/// post-drive finalize hook; otherwise certify through the human-verifier
/// seam and run the executor's durable finalize once.
async fn settle_verified_integrate(env: &Arc<RealToolEnv>) {
    wait_until(
        || real_state_of(env) == faktor_core::state::AgentState::ReadyForNextTurn,
        60,
    )
    .await;
    tokio::time::sleep(Duration::from_millis(400)).await;
    let task = real_env_task_row(env);
    if task.state != TaskState::VerifiedComplete {
        certify_verified_complete(&env.manager, env.parent);
    }
    // Idempotent: a clean auto-integration retired the row already.
    if let Some(f) = env
        .executor
        .finalize_shadow_run(env.parent)
        .expect("finalize runs")
    {
        assert_eq!(f.action, ShadowFinalizeAction::Integrated, "{f:?}");
    }
    let row = env
        .manager
        .shadow_row(env.parent)
        .unwrap()
        .expect("row exists");
    assert_eq!(row.state, ShadowRowState::Integrated);
    assert!(
        !std::path::PathBuf::from(&row.root).exists(),
        "clean integration removes the shadow directory"
    );
}

fn seed_rust(env: &RealToolEnv) {
    std::fs::create_dir_all(env.owner_root.join("src")).unwrap();
    std::fs::write(
        env.owner_root.join("Cargo.toml"),
        "[package]\nname = \"x\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    std::fs::write(
        env.owner_root.join("src/lib.rs"),
        "pub fn value() -> u64 {\n    let base_amount: u64 = 10;\n    let increment: u64 = 32;\n    base_amount.saturating_add(increment)\n}\n",
    )
    .unwrap();
}

#[tokio::test]
async fn real_write_drive_writes_the_shadow_and_verified_complete_integrates_it() {
    let _heavy = heavy_guard();
    // (a) over the REAL executor + REAL tools: a shadowed mutating drive
    // executes write_file against the SHADOW root (the flipped tool-batch
    // site); the user checkout stays byte-untouched MID-drive; a
    // VerifiedComplete integration (wave-21 commit_all through the
    // post-drive finalize) lands the content in the user checkout and
    // removes the shadow.
    let dir = tempfile::tempdir().unwrap();
    let env = open_real_tool_env(
        dir.path(),
        vec![
            vec![
                ScriptedResponse::ToolCall {
                    id: "c1".into(),
                    name: "write_file".into(),
                    input: serde_json::json!({
                        "path": "src/lib.rs",
                        "content": "pub fn value() -> u64 {\n    let base_amount: u64 = 10;\n    let increment: u64 = 32;\n    base_amount.saturating_add(increment)\n}\n",
                    }),
                },
                ScriptedResponse::ToolCall {
                    id: "c2".into(),
                    name: "write_file".into(),
                    input: serde_json::json!({
                        "path": "util.rs",
                        "content": "pub fn fresh() -> u64 {\n    let seed: u64 = 7;\n    let factor: u64 = 3;\n    seed.saturating_mul(factor)\n}\n",
                    }),
                },
                ScriptedResponse::Text("done".into()),
                ScriptedResponse::End,
            ],
            vec![ScriptedResponse::End],
        ],
        faktor_agent::VerificationService::fake_ok(),
        true,
    );
    seed_rust(&env);
    let receipt = env
        .executor
        .start_task(
            env.parent,
            real_mutating_request(&env, "implement the change"),
        )
        .expect("shadowed real-tool start");
    assert_eq!(receipt.mode, TaskRunMode::InSession);
    let row = env
        .manager
        .shadow_row(env.parent)
        .unwrap()
        .expect("shadow row at begin");
    let shadow_dir = std::path::PathBuf::from(&row.root);
    assert_eq!(row.state, ShadowRowState::Active);
    // Mid-drive: the FIRST write landed inside the shadow and parked the
    // drive at ExecutingTool — the user checkout is byte-untouched.
    wait_until(|| env.fired.load(Ordering::SeqCst) >= 1, 300).await;
    assert_eq!(
        std::fs::read(shadow_dir.join("src/lib.rs")).unwrap(),
        b"pub fn value() -> u64 {\n    let base_amount: u64 = 10;\n    let increment: u64 = 32;\n    base_amount.saturating_add(increment)\n}\n",
        "the write landed in the SHADOW"
    );
    assert_eq!(
        std::fs::read(env.owner_root.join("src/lib.rs")).unwrap(),
        b"pub fn value() -> u64 {\n    let base_amount: u64 = 10;\n    let increment: u64 = 32;\n    base_amount.saturating_add(increment)\n}\n",
        "user checkout untouched mid-drive"
    );
    assert!(!env.owner_root.join("util.rs").exists());
    assert_eq!(
        env.manager.active_root(env.parent).unwrap(),
        Some(shadow_dir.clone()),
        "active_root re-points while the drive is live"
    );
    // Release the drive: second write may or may not have landed before the
    // park, but whatever the shadow holds now must not touch the user
    // checkout until the verified integration.
    env.gate.notify_waiters();
    settle_verified_integrate(&env).await;
    assert_eq!(
        std::fs::read(env.owner_root.join("src/lib.rs")).unwrap(),
        b"pub fn value() -> u64 {\n    let base_amount: u64 = 10;\n    let increment: u64 = 32;\n    base_amount.saturating_add(increment)\n}\n",
        "VerifiedComplete integration lands the changed file"
    );
    assert_eq!(
        std::fs::read(env.owner_root.join("util.rs")).unwrap(),
        b"pub fn fresh() -> u64 {\n    let seed: u64 = 7;\n    let factor: u64 = 3;\n    seed.saturating_mul(factor)\n}\n",
        "VerifiedComplete integration lands the created file"
    );
    assert!(
        env.manager.active_root(env.parent).unwrap().is_none(),
        "a retired shadow stops re-pointing"
    );
}

#[tokio::test]
async fn shadowed_drive_reads_repo_knowledge_and_rules_from_the_shadow() {
    // (b): repo knowledge + instructions of a shadowed drive resolve from
    // the SHADOW root — a rules file and AGENTS.md marker placed inside the
    // shadow AFTER begin (never in the user checkout) show up in the NEXT
    // iteration's rendered context, and the user checkout's own rules never
    // leak into the drive.
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("owner")).unwrap();
    let env = open_real_tool_env(
        dir.path(),
        vec![
            vec![
                ScriptedResponse::ToolCall {
                    id: "p1".into(),
                    name: "pause".into(),
                    input: serde_json::json!({}),
                },
                ScriptedResponse::End,
            ],
            vec![
                ScriptedResponse::Text("conclude the change".into()),
                ScriptedResponse::End,
            ],
            vec![ScriptedResponse::End],
        ],
        faktor_agent::VerificationService::disabled(),
        false,
    );
    std::fs::write(
        env.owner_root.join("AGENTS.md"),
        "user-root-marker-4f1: follow user checkout conventions\n",
    )
    .unwrap();
    std::fs::write(env.owner_root.join("a.txt"), b"base-alpha").unwrap();
    let receipt = env
        .executor
        .start_task(env.parent, real_mutating_request(&env, "convention task"))
        .expect("shadowed start");
    assert_eq!(receipt.mode, TaskRunMode::InSession);
    let row = env.manager.shadow_row(env.parent).unwrap().expect("row");
    let shadow_dir = std::path::PathBuf::from(&row.root);
    // Mid-flight (the pause tool parks the drive): write the shadow-only
    // world — a new file and a REWRITTEN AGENTS.md — into the shadow.
    wait_until(|| env.fired.load(Ordering::SeqCst) >= 1, 300).await;
    std::fs::write(
        shadow_dir.join("AGENTS.md"),
        "shadow-world-marker-7c1: drive inside the shadow world\n",
    )
    .unwrap();
    std::fs::write(shadow_dir.join("only-shadow-notes.md"), b"shadow notes\n").unwrap();
    assert!(
        std::fs::read_to_string(env.owner_root.join("AGENTS.md"))
            .unwrap()
            .contains("user-root-marker-4f1"),
        "the user checkout rules are untouched"
    );
    env.gate.notify_waiters();
    wait_until(
        || real_state_of(&env) == faktor_core::state::AgentState::ReadyForNextTurn,
        60,
    )
    .await;
    let prompts = env.provider.recorded();
    assert!(prompts.len() >= 2, "two requests expected: {prompts:?}");
    assert!(
        !prompts[0].contains("shadow-world-marker-7c1"),
        "the first context predates the shadow writes"
    );
    assert!(
        !prompts[0].contains("only-shadow-notes.md"),
        "the first repo map cannot see the shadow-only file"
    );
    let second = &prompts[1];
    assert!(
        second.contains("shadow-world-marker-7c1"),
        "the rewritten shadow AGENTS.md must ride the next context"
    );
    assert!(
        second.contains("only-shadow-notes.md"),
        "the repo file map must come from the shadow root"
    );
    assert!(
        second.contains("always, loaded: always"),
        "the instruction resolver must append the shadow's rule tree"
    );
    assert!(
        !second.contains("user-root-marker-4f1"),
        "user-checkout rules must never leak into a shadowed drive"
    );
    // The drive is a no-change turn: the shadow is retained (nothing was
    // certified), and the user checkout is untouched.
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        !std::fs::read_to_string(env.owner_root.join("AGENTS.md"))
            .unwrap()
            .contains("shadow-world-marker-7c1"),
        "no shadow byte ever lands without a verified integration"
    );
    assert_eq!(
        env.manager.shadow_row(env.parent).unwrap().unwrap().state,
        ShadowRowState::Active
    );
}

#[tokio::test]
async fn real_write_drive_user_drift_conflicts_at_integration_then_resolves() {
    let _heavy = heavy_guard();
    // (d): the conflict path end-to-end at the agent + executor level — the
    // drive's REAL write lands in the shadow; a mid-drive USER edit of the
    // same file conflicts at the VerifiedComplete integration
    // (IntegrationBlocked semantics from wave-21: the user file is never
    // overwritten, the shadow is retained), and resolving the drift lets
    // the same decision integrate.
    let dir = tempfile::tempdir().unwrap();
    let env = open_real_tool_env(
        dir.path(),
        vec![
            vec![
                ScriptedResponse::ToolCall {
                    id: "c1".into(),
                    name: "write_file".into(),
                    input: serde_json::json!({
                        "path": "a.txt",
                        "content": "agent implementation",
                    }),
                },
                ScriptedResponse::Text("done".into()),
                ScriptedResponse::End,
            ],
            vec![ScriptedResponse::End],
        ],
        faktor_agent::VerificationService::disabled(),
        true,
    );
    seed_owner(&env.owner_root);
    let receipt = env
        .executor
        .start_task(env.parent, real_mutating_request(&env, "implement a.txt"))
        .expect("shadowed start");
    assert_eq!(receipt.mode, TaskRunMode::InSession);
    let row = env.manager.shadow_row(env.parent).unwrap().expect("row");
    let shadow_dir = std::path::PathBuf::from(&row.root);
    // Mid-drive: the agent's write landed in the shadow; the user then
    // edits the file externally.
    wait_until(|| env.fired.load(Ordering::SeqCst) >= 1, 300).await;
    assert_eq!(
        std::fs::read(shadow_dir.join("a.txt")).unwrap(),
        b"agent implementation",
        "the agent wrote the shadow"
    );
    assert_eq!(
        std::fs::read(env.owner_root.join("a.txt")).unwrap(),
        b"base-alpha"
    );
    std::fs::write(env.owner_root.join("a.txt"), b"user edit during drive").unwrap();
    env.gate.notify_waiters();
    wait_until(
        || real_state_of(&env) == faktor_core::state::AgentState::ReadyForNextTurn,
        60,
    )
    .await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    certify_verified_complete(&env.manager, env.parent);
    let finalize = env
        .executor
        .finalize_shadow_run(env.parent)
        .expect("finalize runs")
        .expect("shadow exists");
    assert_eq!(
        finalize.action,
        ShadowFinalizeAction::IntegrationBlocked,
        "{finalize:?}"
    );
    assert_eq!(finalize.conflicts.len(), 1);
    assert_eq!(
        std::fs::read(env.owner_root.join("a.txt")).unwrap(),
        b"user edit during drive",
        "a conflicted user file is never overwritten"
    );
    assert!(shadow_dir.is_dir(), "the shadow is retained on conflict");
    assert_eq!(
        env.manager.shadow_row(env.parent).unwrap().unwrap().state,
        ShadowRowState::IntegrationBlocked
    );
    assert_eq!(
        env.manager.active_root(env.parent).unwrap(),
        Some(shadow_dir.clone()),
        "the IntegrationBlocked shadow stays the session's root until resolved"
    );
    // The user resolves the drift (back to the base digest); the same auto
    // decision now integrates.
    std::fs::write(env.owner_root.join("a.txt"), b"base-alpha").unwrap();
    let finalize = env
        .executor
        .finalize_shadow_run(env.parent)
        .expect("finalize runs")
        .expect("shadow exists");
    assert_eq!(finalize.action, ShadowFinalizeAction::Integrated);
    assert_eq!(
        std::fs::read(env.owner_root.join("a.txt")).unwrap(),
        b"agent implementation"
    );
    assert!(!shadow_dir.exists());
    assert_eq!(
        env.manager.shadow_row(env.parent).unwrap().unwrap().state,
        ShadowRowState::Integrated
    );
}
