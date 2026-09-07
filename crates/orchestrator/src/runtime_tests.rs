//! Adversarial tests for the orchestrator runtime wiring (audits 20-24).
//!
//! Children are driven by a REAL `AgentRuntime` over a REAL `SessionManager`
//! with a scripted, chunk-paced provider (no network). Every test attempts
//! to break the invariants: crashes at seams, mid-drive kills, paused
//! drives, duplicate steer application, ceiling pressure, ownership
//! overlap and zero-orphan registry checks after every crash.

use super::graph::{GraphError, OpGraph};
use super::*;
use crate::caps::{CapabilityGrant, LatticeCap, ScopePattern};
use crate::{ChildState, OwnershipModel, TaskPlan, WorkItem, WorkKind, WorkState};
use faktor_agent::tool::RecoveryHint as ToolRecovery;
use faktor_agent::{
    AgentDeps, AgentRuntime, NoEvidence, PermissionRequester, Tool, ToolCallMode, ToolOutcome,
    ToolRegistry, ToolRunCtx,
};
use faktor_core::capability::PermissionDecision;
use faktor_core::id::{SessionId, TaskId, WorktreeId};
use faktor_core::model::ModelCapabilities;
use faktor_core::time::SystemClock;
use faktor_provider::{
    FakeProvider, GenericAgentRequest, Provider, ProviderChunk, ProviderError, ProviderErrorKind,
    ProviderRegistry, ProviderStream, ScriptedResponse,
};
use faktor_session::child::{ChildControl, ChildOwnership};
use faktor_session::SessionManager;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};

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

/// Chunk-paced scripted provider: serves one script per stream call
/// (extra calls end immediately), records every request's model + system
/// text, counts calls, and paces chunks so tests can steer mid-iteration.
struct ScriptedPacedProvider {
    inner: FakeProvider,
    /// One script per stream call, in order (extra calls → End).
    calls: StdMutex<Vec<Vec<ScriptedResponse>>>,
    request_count: AtomicUsize,
    /// model of every request, in order.
    request_models: StdMutex<Vec<String>>,
    /// system text of every request, in order.
    request_systems: StdMutex<Vec<String>>,
    chunk_delay_ms: u64,
    /// Optional per-stream-call pacing override (per-child jitter): call
    /// `i` waits `per_call_delay_ms[i]` ms per chunk when present, else the
    /// shared `chunk_delay_ms`.
    per_call_delay_ms: StdMutex<Vec<u64>>,
    script_index: AtomicUsize,
}

impl ScriptedPacedProvider {
    fn new(
        id: &str,
        caps: ModelCapabilities,
        per_call_scripts: Vec<Vec<ScriptedResponse>>,
        chunk_delay_ms: u64,
    ) -> Arc<Self> {
        Arc::new(Self {
            inner: FakeProvider::new(id, caps),
            calls: StdMutex::new(per_call_scripts),
            request_count: AtomicUsize::new(0),
            request_models: StdMutex::new(Vec::new()),
            request_systems: StdMutex::new(Vec::new()),
            chunk_delay_ms,
            per_call_delay_ms: StdMutex::new(Vec::new()),
            script_index: AtomicUsize::new(0),
        })
    }

    fn count(&self) -> usize {
        self.request_count.load(Ordering::SeqCst)
    }

    fn model_of(&self, idx: usize) -> Option<String> {
        self.request_models.lock().unwrap().get(idx).cloned()
    }

    fn system_of(&self, idx: usize) -> Option<String> {
        self.request_systems.lock().unwrap().get(idx).cloned()
    }

    /// Per-child jitter: override the per-chunk pacing of stream call `i`
    /// (random spawn-completion delays of the randomized reopen test).
    fn set_per_call_delays(&self, delays: Vec<u64>) {
        *self.per_call_delay_ms.lock().unwrap() = delays;
    }
}

impl Provider for ScriptedPacedProvider {
    fn id(&self) -> &str {
        "fake"
    }

    fn capabilities(&self, _model: &str) -> ModelCapabilities {
        self.inner.capabilities(_model)
    }

    fn stream(&self, req: GenericAgentRequest) -> ProviderStream {
        self.request_count.fetch_add(1, Ordering::SeqCst);
        self.request_models.lock().unwrap().push(req.model.clone());
        self.request_systems
            .lock()
            .unwrap()
            .push(req.system.clone());
        let i = self.script_index.fetch_add(1, Ordering::SeqCst);
        let script: Vec<ScriptedResponse> = self
            .calls
            .lock()
            .unwrap()
            .get(i)
            .cloned()
            .unwrap_or_else(|| vec![ScriptedResponse::End]);
        let delay = self
            .per_call_delay_ms
            .lock()
            .unwrap()
            .get(i)
            .copied()
            .unwrap_or(self.chunk_delay_ms);
        Box::pin(futures_stream_paced(script, delay))
    }
}

fn futures_stream_paced(
    script: Vec<ScriptedResponse>,
    delay_ms: u64,
) -> std::pin::Pin<Box<dyn futures::Stream<Item = Result<ProviderChunk, ProviderError>> + Send>> {
    use futures::StreamExt;
    let stream = futures::stream::iter(script).then(move |s| async move {
        if delay_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
        }
        match s {
            ScriptedResponse::Text(t) => Ok(ProviderChunk::Text { text: t }),
            ScriptedResponse::Reasoning(t) => Ok(ProviderChunk::Reasoning { text: t }),
            ScriptedResponse::ToolCall { id, name, input } => Ok(ProviderChunk::ToolCall {
                id,
                name,
                input,
                complete: true,
            }),
            ScriptedResponse::Die(e) => Err(e),
            ScriptedResponse::End => Ok(ProviderChunk::Done),
        }
    });
    Box::pin(stream)
}

/// A single real agent test environment: manager + runtime + paced provider.
/// The store root is owned by the CALLER (a `TempDir` the test keeps alive)
/// so a "daemon restart" test can drop the environment and reopen the SAME
/// directory through a fresh manager — exactly like a process restart.
struct Env {
    manager: Arc<SessionManager>,
    #[allow(dead_code)]
    agent: Arc<AgentRuntime>,
    provider: Arc<ScriptedPacedProvider>,
    orchestrator: Arc<OrchestratorRuntime>,
    /// The orchestrator's own session (parent).
    parent: SessionId,
    owner: OwnerContext,
    isolated_root: std::path::PathBuf,
}

fn echo_tool() -> Tool {
    Tool {
        name: "echo".into(),
        description: "echo its input".into(),
        input_schema: serde_json::json!({"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}),
        resource_class: faktor_core::resource::ResourceClass::Cpu,
        capability: None,
        recovery_hint: ToolRecovery::UnknownEffect,
        path_args: vec![],
        execute: Arc::new(|_ctx: ToolRunCtx, input: serde_json::Value| {
            Box::pin(async move {
                Ok(ToolOutcome {
                    text: input
                        .get("text")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string(),
                    exit_code: Some(0),
                    ..Default::default()
                })
            })
        }),
    }
}

fn caps_tools() -> ModelCapabilities {
    ModelCapabilities {
        tools: true,
        parallel_tools: true,
        ..Default::default()
    }
}

fn open_env(
    root: &std::path::Path,
    per_call_scripts: Vec<Vec<ScriptedResponse>>,
    chunk_delay_ms: u64,
) -> Env {
    let manager = SessionManager::open(root.join("store"), root.join("cas"), true).unwrap();
    let provider =
        ScriptedPacedProvider::new("fake", caps_tools(), per_call_scripts, chunk_delay_ms);
    let mut registry = ProviderRegistry::new();
    registry.try_register(provider.clone()).unwrap();
    let mut tool_registry = ToolRegistry::new();
    tool_registry.register(echo_tool());
    let workspaces = faktor_fs::WorkspaceFileService::new();
    let deps = AgentDeps {
        session: manager.clone(),
        providers: Arc::new(registry),
        chunk_sink: None,
        permission_requester: Arc::new(AlwaysAllow),
        evidence: Arc::new(NoEvidence),
        tools: Arc::new(tool_registry),
        cas: Some(Arc::new(faktor_cas::Cas::open(root.join("cas")).unwrap())),
        workspaces: workspaces.clone(),
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
        tool_call_mode: ToolCallMode::Native,
        tool_deadline_ms: 5000,
        retry_policy: faktor_core::retry::RetryPolicy::default(),
    };
    let agent = AgentRuntime::new(deps).unwrap();
    // The owner session + its workspace (a real directory under the temp
    // dir so tools/evidence can resolve it).
    let owner_dir = root.join("owner");
    std::fs::create_dir_all(&owner_dir).unwrap();
    let owner_ws = manager
        .create_workspace(owner_dir.to_str().unwrap())
        .unwrap();
    let owner_wt = WorktreeId::new(
        manager
            .put_worktree(owner_ws, owner_dir.to_str().unwrap(), "main")
            .unwrap() as u64,
    );
    let parent = manager
        .create_session(owner_ws, "orchestrator", "fake", "m")
        .unwrap()
        .id();
    manager
        .adopt_identity(parent, owner_wt, TaskId::new(1))
        .unwrap();
    let isolated_root = root.join("isolated");
    std::fs::create_dir_all(&isolated_root).unwrap();
    let orch = OrchestratorRuntime::new(manager.clone(), agent.clone());
    Env {
        manager,
        agent,
        provider,
        orchestrator: orch,
        parent,
        owner: OwnerContext {
            parent_session: parent,
            workspace_id: owner_ws.raw(),
            worktree_id: owner_wt.raw(),
            root: owner_dir.clone(),
        },
        isolated_root,
    }
}

fn wi(id: &str, kind: WorkKind, deps: &[&str]) -> WorkItem {
    WorkItem {
        id: id.to_string(),
        summary: format!("work {id}"),
        depends_on: deps.iter().map(|d| d.to_string()).collect(),
        kind,
        acceptance_checks: vec![],
        completion: WorkState::Pending,
    }
}

fn plan(ownership: OwnershipModel, items: Vec<WorkItem>) -> TaskPlan {
    TaskPlan {
        goal: "Ship the feature".to_string(),
        non_goals: vec![],
        constraints: vec![],
        work_items: items,
        ownership,
    }
}

fn base_config(env: &Env, run_id: &str) -> ExecConfig {
    ExecConfig {
        run_id: run_id.to_string(),
        ceilings: Ceilings::default(),
        parent_caps: CapabilitySet::from_grants(vec![CapabilityGrant::new(
            LatticeCap::WriteWorkspace,
            ScopePattern::new("*").unwrap(),
        )])
        .unwrap(),
        provider: "fake".to_string(),
        default_model: "m".to_string(),
        isolated_root: env.isolated_root.clone(),
        crash_seam: None,
    }
}

fn read_caps() -> CapabilitySet {
    CapabilitySet::from_grants(vec![CapabilityGrant::new(
        LatticeCap::ReadWorkspace,
        ScopePattern::new("*").unwrap(),
    )])
    .unwrap()
}

fn spec(item_id: &str) -> ChildSpec {
    let mut s = ChildSpec::new(item_id);
    s.child_caps = read_caps();
    s.task_caps = read_caps();
    s
}

fn assert_registry_consistent(env: &Env, run_id: &str) {
    let violations =
        OrchestratorRuntime::registry_violations(env.manager.clone(), env.parent, run_id);
    assert!(violations.is_empty(), "registry violated: {violations:?}");
}

fn roundtrip_script() -> Vec<Vec<ScriptedResponse>> {
    vec![
        vec![
            ScriptedResponse::Text("analyzing".into()),
            ScriptedResponse::ToolCall {
                id: "c1".into(),
                name: "echo".into(),
                input: serde_json::json!({"text": "hello"}),
            },
            ScriptedResponse::End,
        ],
        vec![ScriptedResponse::Text("done".into()), ScriptedResponse::End],
    ]
}

fn empty_script() -> Vec<Vec<ScriptedResponse>> {
    vec![vec![ScriptedResponse::End]]
}

/// Run execute_task to completion inside a timeout (drive crashes hang
/// otherwise).
async fn run_exec(
    env: &Arc<Env>,
    plan: TaskPlan,
    config: ExecConfig,
    specs: Vec<ChildSpec>,
) -> Result<PlanOutcome, ExecError> {
    tokio::time::timeout(
        std::time::Duration::from_secs(90),
        env.orchestrator
            .execute_task(plan, env.owner.clone(), config, &specs),
    )
    .await
    .expect("execute_task exceeded the 90s hard test bound")
}

// ------------------------------------------------------------------- tests

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn end_to_end_disjoint_mutating_children_run_on_the_owner_worktree() {
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(dir.path(), roundtrip_script(), 2));
    let p = plan(
        OwnershipModel::DisjointPaths {
            paths: vec!["src/a".into(), "src/b".into()],
        },
        vec![
            wi("impl-a", WorkKind::Implementation, &[]),
            wi("impl-b", WorkKind::Implementation, &[]),
        ],
    );
    let mut sa = spec("impl-a");
    sa.ownership = Some(ChildOwnership::ExclusivePaths);
    sa.ownership_paths = vec!["src/a".into()];
    let mut sb = spec("impl-b");
    sb.ownership = Some(ChildOwnership::ExclusivePaths);
    sb.ownership_paths = vec!["src/b".into()];
    let outcome = run_exec(&env, p, base_config(&env, "run-1"), vec![sa, sb])
        .await
        .expect("execution succeeds");
    assert!(outcome.complete, "{outcome:?}");
    assert_eq!(
        outcome.item_states,
        vec![
            ("impl-a".to_string(), WorkState::Done),
            ("impl-b".to_string(), WorkState::Done),
        ]
    );
    assert_eq!(outcome.children.len(), 2);
    for c in &outcome.children {
        assert_eq!(c.state, ChildState::Done);
        assert!(c.session_id != 0, "real session id");
        assert!(c.operation_id != 0, "maps to the child session's op id");
        assert_eq!(c.ownership, ChildOwnership::ExclusivePaths);
        assert_eq!(
            c.worktree_id, env.owner.worktree_id,
            "shares the owner worktree"
        );
    }
    assert_registry_consistent(&env, "run-1");
    // Every child really was driven through the agent (per-child drives,
    // not bookkeeping): at least one provider request per child.
    assert!(
        env.provider.count() >= 2,
        "provider was driven {}",
        env.provider.count()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn isolated_mutating_children_get_real_directories_and_workspaces() {
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(dir.path(), empty_script(), 1));
    let p = plan(
        OwnershipModel::IsolatedWorktree,
        vec![wi("impl", WorkKind::Implementation, &[])],
    );
    let outcome = run_exec(&env, p, base_config(&env, "run-iso"), vec![spec("impl")])
        .await
        .expect("execution succeeds");
    assert!(outcome.complete);
    let c = &outcome.children[0];
    assert_eq!(c.ownership, ChildOwnership::IsolatedWorktree);
    // The isolated directory exists under the data dir and is registered as
    // a real workspace + worktree row pair.
    let dir = env.isolated_root.join("run-iso").join(&c.child_id);
    assert!(dir.is_dir(), "isolated child dir must exist: {dir:?}");
    let srow = env
        .manager
        .get_session(SessionId::new(c.session_id))
        .unwrap()
        .unwrap()
        .row()
        .unwrap();
    assert_eq!(srow.workspace_id.raw(), c.workspace_id);
    assert_eq!(srow.worktree_id.raw(), c.worktree_id);
    let wt_rows = env
        .manager
        .worktrees_of(faktor_core::id::WorkspaceId::new(c.workspace_id))
        .unwrap();
    assert_eq!(wt_rows.len(), 1);
    assert_eq!(wt_rows[0].id as u64, c.worktree_id);
    assert!(wt_rows[0].path.ends_with(&c.child_id));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_only_children_share_the_parent_worktree_concurrently() {
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(dir.path(), empty_script(), 1));
    let p = plan(
        OwnershipModel::NoWrites,
        vec![
            wi("analysis", WorkKind::Analysis, &[]),
            wi("review", WorkKind::Review, &["analysis"]),
        ],
    );
    let outcome = run_exec(
        &env,
        p,
        base_config(&env, "run-ro"),
        vec![spec("analysis"), spec("review")],
    )
    .await
    .expect("execution succeeds");
    assert!(outcome.complete, "{outcome:?}");
    for c in &outcome.children {
        assert_eq!(c.ownership, ChildOwnership::ReadOnlyShared);
        assert_eq!(c.worktree_id, env.owner.worktree_id);
        assert_eq!(c.state, ChildState::Done);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pause_parks_at_a_safe_boundary_never_mid_operation() {
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(dir.path(), roundtrip_script(), 15));
    let p = plan(
        OwnershipModel::NoWrites,
        vec![wi("analysis", WorkKind::Analysis, &[])],
    );
    let run = env.orchestrator.clone();
    let owner = env.owner.clone();
    let config = base_config(&env, "run-pause");
    let handle = tokio::spawn(async move {
        run.execute_task(p, owner, config, &[spec("analysis")])
            .await
            .unwrap()
    });
    // Wait until the first provider request is in flight (mid-iteration),
    // then pause: the durable control must NOT interrupt the stream.
    wait_until(|| env.provider.count() >= 1, 60).await;
    tokio::time::sleep(std::time::Duration::from_millis(40)).await;
    env.orchestrator.pause_child("child-0").expect("pause ok");
    // The pause is applied at the next boundary: requests freeze.
    wait_until(|| paused_waiting(&env, "child-0"), 10).await;
    let frozen = env.provider.count();
    tokio::time::sleep(std::time::Duration::from_millis(400)).await;
    assert_eq!(env.provider.count(), frozen, "paused child must not reason");
    // Resume: the drive continues at the boundary and completes.
    env.orchestrator.resume_child("child-0").expect("resume ok");
    let outcome = tokio::time::timeout(std::time::Duration::from_secs(60), handle)
        .await
        .expect("drive must finish after resume")
        .expect("execute task panicked");
    assert!(outcome.complete, "{outcome:?}");
    assert!(env.provider.count() > frozen, "drive resumed");
    // The pause control row was applied exactly once.
    let session = env
        .manager
        .get_session(SessionId::new(outcome.children[0].session_id))
        .unwrap()
        .unwrap();
    let rows = session.orchestrator_ctl_all().unwrap();
    let pause = rows
        .iter()
        .find(|r| matches!(r.control, ChildControl::Pause))
        .expect("pause row");
    assert!(pause.applied(), "pause row acked");
    let resume = rows
        .iter()
        .find(|r| matches!(r.control, ChildControl::Resume))
        .expect("resume row");
    assert!(resume.applied());
    assert_registry_consistent(&env, "run-pause");
}

fn paused_waiting(env: &Env, child_id: &str) -> bool {
    let Some(child) = env.orchestrator.child(child_id).ok().flatten() else {
        return false;
    };
    child.state == ChildState::Waiting
        || env
            .manager
            .get_session(SessionId::new(child.session_id))
            .ok()
            .flatten()
            .and_then(|h| h.orchestrator_drive_state_get().ok())
            .map(|ds| ds.phase == ChildPhase::Waiting)
            .unwrap_or(false)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancel_reaches_a_running_prompt_within_the_bounded_cancel_path() {
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(dir.path(), roundtrip_script(), 30));
    let p = plan(
        OwnershipModel::NoWrites,
        vec![wi("analysis", WorkKind::Analysis, &[])],
    );
    let run = env.orchestrator.clone();
    let owner = env.owner.clone();
    let config = base_config(&env, "run-cancel");
    let handle = tokio::spawn(async move {
        run.execute_task(p, owner, config, &[spec("analysis")])
            .await
            .unwrap()
    });
    wait_until(|| env.provider.count() >= 1, 60).await;
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    env.orchestrator.cancel_child("child-0").expect("cancel ok");
    let outcome = tokio::time::timeout(std::time::Duration::from_secs(60), handle)
        .await
        .expect("cancel must reach the running prompt in bounded time")
        .expect("execute task panicked");
    assert_eq!(
        outcome.item_states,
        vec![("analysis".to_string(), WorkState::Cancelled)]
    );
    assert!(!outcome.complete);
    assert_eq!(outcome.cancelled, vec!["analysis".to_string()]);
    // No dead session: the child session is still promptable afterwards.
    let sid = SessionId::new(outcome.children[0].session_id);
    let session = env.manager.get_session(sid).unwrap().unwrap();
    assert!(
        session.submit_prompt("ping after cancel", &[]).is_ok(),
        "cancelled child session must stay promptable"
    );
    // The cancel control row is acked.
    let rows = session.orchestrator_ctl_all().unwrap();
    let cancel = rows
        .iter()
        .find(|r| matches!(r.control, ChildControl::Cancel))
        .expect("cancel row");
    assert!(cancel.applied());
    assert_registry_consistent(&env, "run-cancel");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn steer_applies_at_the_next_provider_selection_with_exactly_once_acks() {
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(dir.path(), roundtrip_script(), 15));
    let p = plan(
        OwnershipModel::NoWrites,
        vec![wi("analysis", WorkKind::Analysis, &[])],
    );
    let run = env.orchestrator.clone();
    let owner = env.owner.clone();
    let config = base_config(&env, "run-steer");
    let handle = tokio::spawn(async move {
        run.execute_task(p, owner, config, &[spec("analysis")])
            .await
            .unwrap()
    });
    wait_until(|| env.provider.count() >= 1, 60).await;
    tokio::time::sleep(std::time::Duration::from_millis(40)).await;
    env.orchestrator
        .steer_child("child-0", "focus on the API surface")
        .expect("steer ok");
    let outcome = tokio::time::timeout(std::time::Duration::from_secs(60), handle)
        .await
        .expect("drive must finish")
        .expect("execute task panicked");
    assert!(outcome.complete);
    // The note rode the SECOND provider selection's system text.
    let sys = env.provider.system_of(1).unwrap_or_default();
    assert!(
        sys.contains("focus on the API surface"),
        "note missing from request 2 system: {sys}"
    );
    let sid = SessionId::new(outcome.children[0].session_id);
    let session = env.manager.get_session(sid).unwrap().unwrap();
    let rows = session.orchestrator_ctl_all().unwrap();
    let steer = rows
        .iter()
        .find(|r| matches!(r.control, ChildControl::Steer { .. }))
        .expect("steer row");
    assert!(steer.applied(), "steer acked");
    // The durable current note persists on the drive state row.
    let ds = session.orchestrator_drive_state_get().unwrap();
    assert_eq!(ds.current_note, "focus on the API surface");
    assert_registry_consistent(&env, "run-steer");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn steer_restart_mid_queue_applies_each_durable_message_exactly_once() {
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(dir.path(), roundtrip_script(), 15));
    let p = plan(
        OwnershipModel::NoWrites,
        vec![wi("analysis", WorkKind::Analysis, &[])],
    );
    // Crash the executor mid-drive AFTER enqueueing a steer (row pending,
    // unapplied at the moment the executor dies).
    let run = env.orchestrator.clone();
    let owner = env.owner.clone();
    let config = base_config(&env, "run-steer-crash");
    let handle = tokio::spawn(async move {
        run.execute_task(p, owner, config, &[spec("analysis")])
            .await
            .unwrap()
    });
    wait_until(|| env.provider.count() >= 1, 60).await;
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    env.orchestrator
        .steer_child("child-0", "durable note")
        .expect("steer ok");
    tokio::time::sleep(std::time::Duration::from_millis(15)).await;
    handle.abort(); // mid-drive kill (drop the executor future)
    let _ = handle.await;
    let rows =
        OrchestratorRuntime::registry_rows(env.manager.clone(), env.parent, "run-steer-crash")
            .unwrap();
    assert_eq!(rows.len(), 1);
    let sid = SessionId::new(rows[0].session_id);
    let parent = env.parent;
    let caps = base_config(&env, "x").parent_caps;
    let isolated_root = env.isolated_root.clone();
    assert_registry_consistent(&env, "run-steer-crash");
    drop(env); // daemon restart: the whole manager goes down
               // Re-attach through a FRESH manager over the same store: the
               // interrupted turn resumes from its durable op record and the pending
               // steer is applied exactly once (tool-free continuation, see the
               // mid-drive test notes).
    let env2 = Arc::new(open_env(
        dir.path(),
        vec![
            vec![
                ScriptedResponse::Text("resumed".into()),
                ScriptedResponse::End,
            ],
            vec![ScriptedResponse::End],
        ],
        1,
    ));
    let outcome = tokio::time::timeout(
        std::time::Duration::from_secs(90),
        env2.orchestrator.reattach(
            parent,
            "run-steer-crash",
            Ceilings::default(),
            caps.clone(),
            "m".to_string(),
            isolated_root.clone(),
            None,
        ),
    )
    .await
    .expect("reattach exceeded the hard bound")
    .expect("reattach ok");
    assert!(outcome.complete, "{outcome:?}");
    let session = env2.manager.get_session(sid).unwrap().unwrap();
    let rows = session.orchestrator_ctl_all().unwrap();
    let steer = rows
        .iter()
        .find(|r| matches!(r.control, ChildControl::Steer { .. }))
        .expect("steer row");
    assert!(steer.applied(), "exactly-once ack after restart");
    let ds = session.orchestrator_drive_state_get().unwrap();
    assert_eq!(ds.current_note, "durable note");
    // A second re-attach must not re-apply anything (idempotent).
    let outcome2 = tokio::time::timeout(
        std::time::Duration::from_secs(60),
        env2.orchestrator.reattach(
            parent,
            "run-steer-crash",
            Ceilings::default(),
            caps,
            "m".to_string(),
            isolated_root,
            None,
        ),
    )
    .await
    .expect("second reattach exceeded the hard bound")
    .expect("second reattach ok");
    assert!(outcome2.complete);
    let rows2 = session.orchestrator_ctl_all().unwrap();
    let steer2 = rows2
        .iter()
        .find(|r| matches!(r.control, ChildControl::Steer { .. }))
        .unwrap();
    assert_eq!(steer.applied_ms, steer2.applied_ms, "applied exactly once");
    assert_registry_consistent(&env2, "run-steer-crash");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn model_change_takes_effect_at_the_next_provider_selection() {
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(dir.path(), roundtrip_script(), 15));
    let p = plan(
        OwnershipModel::NoWrites,
        vec![wi("analysis", WorkKind::Analysis, &[])],
    );
    let mut s = spec("analysis");
    s.model = Some("m1".into());
    let run = env.orchestrator.clone();
    let owner = env.owner.clone();
    let config = base_config(&env, "run-model");
    let handle =
        tokio::spawn(async move { run.execute_task(p, owner, config, &[s]).await.unwrap() });
    wait_until(|| env.provider.count() >= 1, 60).await;
    assert_eq!(env.provider.model_of(0).as_deref(), Some("m1"));
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    env.orchestrator
        .change_child_model("child-0", "m2")
        .expect("model change ok");
    let outcome = tokio::time::timeout(std::time::Duration::from_secs(60), handle)
        .await
        .expect("drive must finish")
        .expect("task panicked");
    assert!(outcome.complete);
    assert_eq!(
        env.provider.model_of(1).as_deref(),
        Some("m2"),
        "second provider selection must use the steered model"
    );
    assert_registry_consistent(&env, "run-model");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn budget_change_patches_the_durable_task_row_caps() {
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(dir.path(), roundtrip_script(), 1));
    let p = plan(
        OwnershipModel::NoWrites,
        vec![wi("analysis", WorkKind::Analysis, &[])],
    );
    let run = env.orchestrator.clone();
    let owner = env.owner.clone();
    let config = base_config(&env, "run-budget");
    let handle = tokio::spawn(async move {
        run.execute_task(p, owner, config, &[spec("analysis")])
            .await
            .unwrap()
    });
    wait_until(|| env.provider.count() >= 1, 60).await;
    env.orchestrator
        .change_child_budget("child-0", 123_456)
        .expect("budget change ok");
    let outcome = tokio::time::timeout(std::time::Duration::from_secs(60), handle)
        .await
        .expect("drive must finish")
        .expect("task panicked");
    assert!(outcome.complete);
    let sid = SessionId::new(outcome.children[0].session_id);
    let session = env.manager.get_session(sid).unwrap().unwrap();
    let task = session.get_task(TaskId::new(1)).unwrap().expect("task row");
    assert_eq!(
        task.budget.max_tokens,
        Some(123_456),
        "durable wave-9 cap patched"
    );
    let child_row = outcome
        .children
        .iter()
        .find(|c| c.child_id == "child-0")
        .unwrap();
    assert_eq!(child_row.budget_max_tokens, Some(123_456));
    let rows = session.orchestrator_ctl_all().unwrap();
    let budget = rows
        .iter()
        .find(|r| matches!(r.control, ChildControl::ChangeBudget { .. }))
        .expect("budget row");
    assert!(budget.applied());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn live_ceiling_hard_rejects_with_a_typed_error_before_registration() {
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(dir.path(), roundtrip_script(), 30));
    let p = plan(
        OwnershipModel::NoWrites,
        vec![
            wi("a", WorkKind::Analysis, &[]),
            wi("b", WorkKind::Analysis, &[]),
            wi("c", WorkKind::Analysis, &[]),
        ],
    );
    let mut config = base_config(&env, "run-ceil");
    config.ceilings = Ceilings {
        max_live: 2,
        max_reasoning_active: 8,
        max_mutating_active: 8,
    };
    let err = run_exec(&env, p, config, vec![spec("a"), spec("b"), spec("c")])
        .await
        .expect_err("three live children with max_live=2 must be a typed reject");
    match err {
        ExecError::CeilingExceeded { class, limit, used } => {
            assert_eq!(class, "live");
            assert_eq!(limit, 2);
            assert_eq!(used, 2);
        }
        other => panic!("expected CeilingExceeded, got {other:?}"),
    }
    // Two durable children exist; the third was never registered anywhere.
    let rows =
        OrchestratorRuntime::registry_rows(env.manager.clone(), env.parent, "run-ceil").unwrap();
    assert_eq!(rows.len(), 2);
    assert_registry_consistent(&env, "run-ceil");
    // Re-attach with roomier ceilings completes the plan.
    let outcome = tokio::time::timeout(
        std::time::Duration::from_secs(90),
        env.orchestrator.reattach(
            env.parent,
            "run-ceil",
            Ceilings {
                max_live: 8,
                ..Ceilings::default()
            },
            base_config(&env, "x").parent_caps,
            "m".to_string(),
            env.isolated_root.clone(),
            None,
        ),
    )
    .await
    .expect("reattach exceeded the hard bound")
    .expect("reattach ok");
    assert!(outcome.complete, "{outcome:?}");
    assert_eq!(outcome.children.len(), 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn overlapping_exclusive_ownership_is_refused_before_spawn() {
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(dir.path(), roundtrip_script(), 30));
    // Real dirs so canonicalization collapses spellings onto the same root.
    let _ = std::fs::create_dir_all(env.owner.root.join("src"));
    let p = plan(
        OwnershipModel::DisjointPaths {
            paths: vec!["src".into()],
        },
        vec![
            wi("a", WorkKind::Implementation, &[]),
            wi("b", WorkKind::Implementation, &[]),
        ],
    );
    // Both children are live CONCURRENTLY and claim the SAME normalized
    // write path: the second spawn must be refused before it happens.
    let mut sa = spec("a");
    sa.ownership = Some(ChildOwnership::ExclusivePaths);
    sa.ownership_paths = vec!["src".into()];
    let mut sb = spec("b");
    sb.ownership = Some(ChildOwnership::ExclusivePaths);
    sb.ownership_paths = vec!["src/../src".into()];
    let err = run_exec(&env, p, base_config(&env, "run-overlap"), vec![sa, sb])
        .await
        .expect_err("overlapping normalized write sets must be refused");
    assert!(
        matches!(err, ExecError::OverlappingExclusiveOwnership(_)),
        "{err:?}"
    );
    let rows =
        OrchestratorRuntime::registry_rows(env.manager.clone(), env.parent, "run-overlap").unwrap();
    assert_eq!(rows.len(), 1, "the overlapping child was never spawned");
    assert_registry_consistent(&env, "run-overlap");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn child_capability_policy_is_intersected_and_never_exceeds_the_parent() {
    // A child policy demanding the whole workspace under a parent that only
    // owns src/ is clamped; the durable row carries the effective set.
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(dir.path(), empty_script(), 1));
    let p = plan(
        OwnershipModel::NoWrites,
        vec![wi("analysis", WorkKind::Analysis, &[])],
    );
    let mut config = base_config(&env, "run-caps");
    config.parent_caps = CapabilitySet::from_grants(vec![CapabilityGrant::new(
        LatticeCap::ReadWorkspace,
        ScopePattern::new("src").unwrap(),
    )])
    .unwrap();
    let mut s = spec("analysis");
    s.task_caps = CapabilitySet::from_grants(vec![CapabilityGrant::new(
        LatticeCap::ReadWorkspace,
        ScopePattern::new("*").unwrap(),
    )])
    .unwrap();
    s.child_caps = s.task_caps.clone();
    let outcome = run_exec(&env, p, config, vec![s])
        .await
        .expect("execution succeeds");
    let c = &outcome.children[0];
    assert_eq!(c.permissions.len(), 1);
    let grant = c.permissions.iter().next().unwrap();
    assert_eq!(grant.scope.as_str(), "src", "clamped to the parent's scope");
    assert!(c.permissions.covered_by(&c.permissions));
    // And the durable registry row carries the same effective set.
    let rows =
        OrchestratorRuntime::registry_rows(env.manager.clone(), env.parent, "run-caps").unwrap();
    assert_eq!(rows[0].permissions, c.permissions);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn crash_after_child_created_before_drive_reattaches_and_completes() {
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(dir.path(), roundtrip_script(), 2));
    let p = plan(
        OwnershipModel::NoWrites,
        vec![
            wi("a", WorkKind::Analysis, &[]),
            wi("b", WorkKind::Analysis, &["a"]),
        ],
    );
    let mut config = base_config(&env, "run-crash-bd");
    config.crash_seam = Some(CrashSeam::BeforeDrive);
    let err = run_exec(&env, p.clone(), config, vec![spec("a"), spec("b")])
        .await
        .expect_err("the BeforeDrive seam must fire");
    assert!(matches!(err, ExecError::InjectedCrashSeam(_)), "{err:?}");
    let rows = OrchestratorRuntime::registry_rows(env.manager.clone(), env.parent, "run-crash-bd")
        .unwrap();
    assert_eq!(
        rows.len(),
        1,
        "exactly one child was created before the crash"
    );
    assert_registry_consistent(&env, "run-crash-bd");
    // The child session exists but nothing was driven yet.
    let session = env
        .manager
        .get_session(SessionId::new(rows[0].session_id))
        .unwrap()
        .unwrap();
    assert_eq!(session.state().unwrap(), AgentState::Idle);
    assert!(session.active_turn_record().unwrap().is_none());
    // Re-attach drives it from its durable rows to completion.
    let outcome = tokio::time::timeout(
        std::time::Duration::from_secs(90),
        env.orchestrator.reattach(
            env.parent,
            "run-crash-bd",
            Ceilings::default(),
            base_config(&env, "x").parent_caps,
            "m".to_string(),
            env.isolated_root.clone(),
            None,
        ),
    )
    .await
    .expect("reattach exceeded the hard bound")
    .expect("reattach ok");
    assert!(outcome.complete, "{outcome:?}");
    assert_eq!(outcome.children.len(), 2);
    assert_registry_consistent(&env, "run-crash-bd");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mid_drive_executor_kill_resumes_the_same_recorded_turn() {
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(dir.path(), roundtrip_script(), 60));
    let p = plan(
        OwnershipModel::NoWrites,
        vec![wi("a", WorkKind::Analysis, &[])],
    );
    let run = env.orchestrator.clone();
    let owner = env.owner.clone();
    let config = base_config(&env, "run-kill");
    let handle = tokio::spawn(async move {
        let _ = run.execute_task(p, owner, config, &[spec("a")]).await;
    });
    // Kill the executor while the child drive is genuinely mid-turn.
    wait_until(|| env.provider.count() >= 1, 60).await;
    // Kill while request 1 is genuinely mid-stream (60 ms per chunk).
    tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    handle.abort();
    let _ = handle.await;
    let mid_state = env
        .manager
        .get_session(SessionId::new(2))
        .unwrap()
        .unwrap()
        .state()
        .unwrap();
    assert!(
        mid_state.is_active(),
        "kill must land mid-turn, got {mid_state:?}"
    );
    let rows =
        OrchestratorRuntime::registry_rows(env.manager.clone(), env.parent, "run-kill").unwrap();
    assert_eq!(rows.len(), 1);
    let sid = SessionId::new(rows[0].session_id);
    let parent = env.parent;
    let caps = base_config(&env, "x").parent_caps;
    let isolated_root = env.isolated_root.clone();
    assert_registry_consistent(&env, "run-kill");
    // The interrupted turn record is durable and active.
    let record = env
        .manager
        .get_session(sid)
        .unwrap()
        .unwrap()
        .active_turn_record()
        .unwrap();
    assert!(
        record.is_some(),
        "mid-drive kill leaves an active turn record"
    );
    drop(env); // daemon restart
               // A FRESH manager resumes the SAME recorded turn (continue_turn —
               // never a synthesized operation). The continuation streams a plain
               // text-only final response (no new tool calls: a resumed logical turn
               // is not re-tracked in-process, so it must not propose tools).
    let env2 = Arc::new(open_env(
        dir.path(),
        vec![
            vec![
                ScriptedResponse::Text("continuing after restart".into()),
                ScriptedResponse::End,
            ],
            vec![ScriptedResponse::End],
        ],
        1,
    ));
    let outcome = tokio::time::timeout(
        std::time::Duration::from_secs(90),
        env2.orchestrator.reattach(
            parent,
            "run-kill",
            Ceilings::default(),
            caps,
            "m".to_string(),
            isolated_root,
            None,
        ),
    )
    .await
    .expect("reattach exceeded the hard bound")
    .expect("reattach ok");
    assert!(outcome.complete, "{outcome:?}");
    assert_eq!(outcome.children[0].state, ChildState::Done);
    let session = env2
        .manager
        .get_session(SessionId::new(outcome.children[0].session_id))
        .unwrap()
        .unwrap();
    assert!(session.active_turn_record().unwrap().is_none());
    assert_eq!(session.state().unwrap(), AgentState::ReadyForNextTurn);
    assert_registry_consistent(&env2, "run-kill");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn crash_after_child_terminal_leaves_consistent_state_for_parent_continuation() {
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(dir.path(), roundtrip_script(), 1));
    let p = plan(
        OwnershipModel::NoWrites,
        vec![
            wi("a", WorkKind::Analysis, &[]),
            wi("b", WorkKind::Analysis, &["a"]),
        ],
    );
    let mut config = base_config(&env, "run-crash-at");
    config.crash_seam = Some(CrashSeam::AfterChildTerminal);
    let err = run_exec(&env, p, config, vec![spec("a"), spec("b")])
        .await
        .expect_err("the AfterChildTerminal seam must fire");
    assert!(matches!(err, ExecError::InjectedCrashSeam(_)), "{err:?}");
    let rows = OrchestratorRuntime::registry_rows(env.manager.clone(), env.parent, "run-crash-at")
        .unwrap();
    assert_eq!(rows.len(), 1, "only the first wave's child was created");
    assert_eq!(
        rows[0].state,
        ChildState::Done,
        "child terminal before the crash"
    );
    assert_registry_consistent(&env, "run-crash-at");
    // Re-attach: the Done child is not re-driven; its dependent item b is
    // admitted and completes.
    let outcome = tokio::time::timeout(
        std::time::Duration::from_secs(90),
        env.orchestrator.reattach(
            env.parent,
            "run-crash-at",
            Ceilings::default(),
            base_config(&env, "x").parent_caps,
            "m".to_string(),
            env.isolated_root.clone(),
            None,
        ),
    )
    .await
    .expect("reattach exceeded the hard bound")
    .expect("reattach ok");
    assert!(outcome.complete, "{outcome:?}");
    assert_eq!(outcome.children.len(), 2);
    assert!(outcome.children.iter().all(|c| c.state == ChildState::Done));
    assert_registry_consistent(&env, "run-crash-at");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn failed_child_retries_only_from_failed_with_a_durable_row() {
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(
        dir.path(),
        vec![
            vec![ScriptedResponse::Die(ProviderError::new(
                ProviderErrorKind::Malformed,
                "injected permanent failure",
            ))],
            roundtrip_script()[0].clone(),
            vec![
                ScriptedResponse::Text("recovered".into()),
                ScriptedResponse::End,
            ],
        ],
        1,
    ));
    let p = plan(
        OwnershipModel::NoWrites,
        vec![wi("a", WorkKind::Analysis, &[])],
    );
    let outcome = run_exec(&env, p, base_config(&env, "run-fail"), vec![spec("a")])
        .await
        .expect("execution returns (failed)");
    assert!(!outcome.complete);
    assert_eq!(outcome.failed, vec!["a".to_string()]);
    let rows =
        OrchestratorRuntime::registry_rows(env.manager.clone(), env.parent, "run-fail").unwrap();
    assert_eq!(rows[0].state, ChildState::Failed);
    assert_registry_consistent(&env, "run-fail");
    // Retry only works from Failed and requires a durable Retry row.
    let err = env.orchestrator.resume_child("child-0").unwrap_err();
    assert!(matches!(err, ExecError::InvalidState(_)), "{err:?}");
    // Re-attach drives the retry (the durable plan row says the child
    // failed; the pending Retry row admits exactly one re-drive).
    let retried = tokio::time::timeout(
        std::time::Duration::from_secs(90),
        env.orchestrator.reattach(
            env.parent,
            "run-fail",
            Ceilings::default(),
            base_config(&env, "x").parent_caps,
            "m".to_string(),
            env.isolated_root.clone(),
            None,
        ),
    )
    .await
    .expect("reattach exceeded the hard bound");
    match retried {
        Ok(outcome) if !outcome.complete => {
            // The scripted provider still fails on re-attach (its second
            // stream error): the child stays Failed with no pending retry —
            // never an automatic infinite retry loop.
            let rows =
                OrchestratorRuntime::registry_rows(env.manager.clone(), env.parent, "run-fail")
                    .unwrap();
            assert_eq!(rows[0].state, ChildState::Failed);
            // Now a human retry: enqueue the durable Retry row and drive.
            env.orchestrator
                .retry_child("child-0")
                .expect("retry enqueued");
            let done = tokio::time::timeout(
                std::time::Duration::from_secs(90),
                env.orchestrator.reattach(
                    env.parent,
                    "run-fail",
                    Ceilings::default(),
                    base_config(&env, "x").parent_caps,
                    "m".to_string(),
                    env.isolated_root.clone(),
                    None,
                ),
            )
            .await
            .expect("retry reattach exceeded the hard bound")
            .expect("retry reattach ok");
            assert!(done.complete, "{done:?}");
            assert_eq!(done.children[0].state, ChildState::Done);
            let session = env
                .manager
                .get_session(SessionId::new(done.children[0].session_id))
                .unwrap()
                .unwrap();
            let rows = session.orchestrator_ctl_all().unwrap();
            let retry = rows
                .iter()
                .find(|r| matches!(r.control, ChildControl::Retry))
                .expect("retry row");
            assert!(
                retry.applied(),
                "retry decision is durable before the drive"
            );
            assert_registry_consistent(&env, "run-fail");
        }
        other => panic!("unexpected retry outcome: {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn registry_and_identity_rows_survive_a_full_manager_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let (parent, run_id, child_count) = {
        // Full env on a path we control (need a real dir for the owner).
        let manager =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let provider = Arc::new(FakeProvider::with_script(
            "fake",
            caps_tools(),
            vec![ScriptedResponse::End],
        ));
        let mut registry = ProviderRegistry::new();
        registry.try_register(provider).unwrap();
        let mut tool_registry = ToolRegistry::new();
        tool_registry.register(echo_tool());
        let deps = AgentDeps {
            session: manager.clone(),
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
            tool_call_mode: ToolCallMode::Native,
            tool_deadline_ms: 5000,
            retry_policy: faktor_core::retry::RetryPolicy::default(),
        };
        let agent = AgentRuntime::new(deps).unwrap();
        let owner_dir = dir.path().join("owner");
        std::fs::create_dir_all(&owner_dir).unwrap();
        let owner_ws = manager
            .create_workspace(owner_dir.to_str().unwrap())
            .unwrap();
        let owner_wt = WorktreeId::new(
            manager
                .put_worktree(owner_ws, owner_dir.to_str().unwrap(), "main")
                .unwrap() as u64,
        );
        let parent = manager
            .create_session(owner_ws, "orchestrator", "fake", "m")
            .unwrap()
            .id();
        manager
            .adopt_identity(parent, owner_wt, TaskId::new(1))
            .unwrap();
        let isolated_root = dir.path().join("isolated");
        std::fs::create_dir_all(&isolated_root).unwrap();
        let orch = OrchestratorRuntime::new(manager.clone(), agent.clone());
        let p = plan(
            OwnershipModel::NoWrites,
            vec![wi("a", WorkKind::Analysis, &[])],
        );
        let run_id = "run-reopen";
        let cfg = ExecConfig {
            run_id: run_id.to_string(),
            ceilings: Ceilings::default(),
            parent_caps: read_caps(),
            provider: "fake".to_string(),
            default_model: "m".to_string(),
            isolated_root: isolated_root.clone(),
            crash_seam: None,
        };
        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(90),
            orch.execute_task(
                p,
                OwnerContext {
                    parent_session: parent,
                    workspace_id: owner_ws.raw(),
                    worktree_id: owner_wt.raw(),
                    root: owner_dir.clone(),
                },
                cfg,
                &[spec("a")],
            ),
        )
        .await
        .expect("execution bound")
        .expect("execution ok");
        assert!(outcome.complete);
        let _ = agent;
        drop(orch);
        (parent, run_id.to_string(), outcome.children.len())
    };
    // Manager reopen: registry rows, child identity rows and the plan row
    // all survive; the zero-orphan invariant holds on the reopened store.
    let manager =
        SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
    let violations = OrchestratorRuntime::registry_violations(manager.clone(), parent, &run_id);
    assert!(
        violations.is_empty(),
        "violations after reopen: {violations:?}"
    );
    let rows = OrchestratorRuntime::registry_rows(manager.clone(), parent, &run_id).unwrap();
    assert_eq!(rows.len(), child_count);
    assert_eq!(rows[0].state, ChildState::Done);
    let session = manager
        .get_session(SessionId::new(rows[0].session_id))
        .unwrap()
        .unwrap();
    let identity = session.orchestrator_child_identity_get().unwrap().unwrap();
    assert_eq!(identity.parent_session_id, parent);
    assert_eq!(identity.worktree_id, rows[0].worktree_id);
    assert_eq!(identity.operation_id, rows[0].operation_id);
}

// ------------------------------------------------------------------ helper

async fn wait_until(mut cond: impl FnMut() -> bool, timeout_secs: u64) {
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    while !cond() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "condition not reached within {timeout_secs}s"
        );
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
}

// ============================================================ wave-13 merge
// Audits 98/99 + 70: controlled child-worktree merging over the REAL
// wave-12 runtime. Every test attempts to break the invariants: parent
// edits between base snapshot and approval, hostile approval paths, partial
// decisions, crash windows between the durable merge record and the CAS
// applies, oversized change sets, concurrent parent writers during reviewer
// spawns, and digest preservation after merges.

use crate::runtime::merge::{self, MAX_CHANGES};

fn isolated_plan() -> TaskPlan {
    plan(
        OwnershipModel::IsolatedWorktree,
        vec![wi("impl", WorkKind::Implementation, &[])],
    )
}

/// Run an isolated Implementation plan to terminal success (the child
/// drives with the empty script: no tools, instant Done).
async fn run_isolated_done(env: &Arc<Env>, run_id: &str) -> PlanOutcome {
    let outcome = run_exec(
        env,
        isolated_plan(),
        base_config(env, run_id),
        vec![spec("impl")],
    )
    .await
    .expect("execution succeeds");
    assert!(outcome.complete, "{outcome:?}");
    assert_eq!(outcome.children[0].state, ChildState::Done);
    outcome
}

fn owner_write(env: &Env, rel: &str, content: &[u8]) {
    let p = env.owner.root.join(rel);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(p, content).unwrap();
}

fn owner_read(env: &Env, rel: &str) -> Vec<u8> {
    std::fs::read(env.owner.root.join(rel)).unwrap()
}

fn child_dir(env: &Env, run_id: &str, child_id: &str) -> std::path::PathBuf {
    let dir = env.isolated_root.join(run_id).join(child_id);
    assert!(dir.is_dir(), "child dir must exist: {dir:?}");
    dir
}

fn file_hash(content: &[u8]) -> faktor_core::hash::FileHash {
    faktor_core::hash::FileHash::from(blake3_hash(content))
}

fn blake3_hash(content: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    let h = blake3::hash(content);
    out.copy_from_slice(h.as_bytes());
    out
}

fn write_owner_file(env: &Env, rel: &str, content: &[u8]) {
    owner_write(env, rel, content);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn merge_conflict_reports_parent_change_and_keeps_parent_bytes_intact() {
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(dir.path(), empty_script(), 1));
    // Parent state BEFORE the child spawn (the base snapshot).
    write_owner_file(&env, "src/a.rs", b"v1-parent-original");
    run_isolated_done(&env, "run-conflict").await;
    // The parent moved on AFTER the base snapshot...
    write_owner_file(&env, "src/a.rs", b"v2-parent-edited-after-base");
    // ...and the child's worktree proposes its own version.
    let child = child_dir(&env, "run-conflict", "child-0");
    std::fs::create_dir_all(child.join("src")).unwrap();
    std::fs::write(child.join("src/a.rs"), b"child-version-c").unwrap();
    let cs = env
        .orchestrator
        .stage_child_changes("child-0")
        .expect("staging succeeds");
    assert_eq!(cs.files.len(), 1);
    assert_eq!(cs.files[0].path, std::path::PathBuf::from("src/a.rs"));
    assert_eq!(cs.files[0].child_hash, Some(file_hash(b"child-version-c")));
    assert_eq!(
        cs.files[0].base_hash,
        Some(file_hash(b"v1-parent-original")),
        "the CAS expectation is the base-snapshot hash of the parent path"
    );
    let outcome = env
        .orchestrator
        .approve_and_merge(
            "child-0",
            &cs.id(),
            &[std::path::PathBuf::from("src/a.rs")],
            &[],
        )
        .expect("merge returns (with a conflict)");
    assert!(outcome.merged.is_empty(), "{outcome:?}");
    assert_eq!(outcome.conflicts.len(), 1, "{outcome:?}");
    assert_eq!(outcome.conflicts[0].0, std::path::PathBuf::from("src/a.rs"));
    assert!(
        outcome.conflicts[0].1.contains("base snapshot"),
        "{}",
        outcome.conflicts[0].1
    );
    // Parent bytes intact; the child change was NOT applied.
    assert_eq!(owner_read(&env, "src/a.rs"), b"v2-parent-edited-after-base");
    // The durable record says failed with the conflict surfaced.
    let envs = merge::merge_envelopes(&env.manager, env.parent, "run-conflict", "child-0").unwrap();
    assert_eq!(envs.len(), 1);
    assert_eq!(envs[0].status, merge::MergeStatus::Failed);
    assert_eq!(envs[0].conflict_count, 1);
    assert!(envs[0].finished_ms.is_some());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn merge_approval_rejects_traversal_case_and_absolute_paths_typed() {
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(dir.path(), empty_script(), 1));
    run_isolated_done(&env, "run-traversal").await;
    let child = child_dir(&env, "run-traversal", "child-0");
    std::fs::create_dir_all(child.join("sub")).unwrap();
    std::fs::write(child.join("sub/a.rs"), b"child-a").unwrap();
    std::fs::write(child.join("sub/b.rs"), b"child-b").unwrap();
    let cs = env
        .orchestrator
        .stage_child_changes("child-0")
        .expect("stage");
    let a = std::path::PathBuf::from("sub/a.rs");
    let b = std::path::PathBuf::from("sub/b.rs");
    // Each hostile approval is a typed error and NOTHING is applied.
    for evil in [
        std::path::PathBuf::from("../sub/a.rs"),
        std::path::PathBuf::from("/etc/passwd"),
        std::path::PathBuf::from(".."),
        std::path::PathBuf::from("sub/../sub/a.rs"),
        std::path::PathBuf::from("SUB/A.RS"), // case-variant of a real path
    ] {
        let err = env
            .orchestrator
            .approve_and_merge(
                "child-0",
                &cs.id(),
                &[evil.clone(), b.clone()],
                std::slice::from_ref(&a),
            )
            .unwrap_err();
        assert!(
            matches!(err, ExecError::InvalidApproval(_)),
            "{evil:?} must be a typed invalid approval, got {err:?}"
        );
        // Nothing was applied and no durable merge record exists.
        assert!(!env.owner.root.join("sub/a.rs").exists());
        assert!(!env.owner.root.join("sub/b.rs").exists());
        let envs =
            merge::merge_envelopes(&env.manager, env.parent, "run-traversal", "child-0").unwrap();
        assert!(
            envs.is_empty(),
            "no durable record after rejected approvals"
        );
    }
    // The valid full decision afterwards merges both files.
    let outcome = env
        .orchestrator
        .approve_and_merge("child-0", &cs.id(), &[a.clone(), b.clone()], &[])
        .expect("valid approval merges");
    assert_eq!(outcome.merged, vec![a.clone(), b.clone()]);
    assert_eq!(owner_read(&env, "sub/a.rs"), b"child-a");
    assert_eq!(owner_read(&env, "sub/b.rs"), b"child-b");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn partial_approval_is_atomic_and_rejections_are_durable_forever() {
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(dir.path(), empty_script(), 1));
    run_isolated_done(&env, "run-partial").await;
    let child = child_dir(&env, "run-partial", "child-0");
    std::fs::write(child.join("a.rs"), b"child-a").unwrap();
    std::fs::write(child.join("b.rs"), b"child-b").unwrap();
    let cs = env
        .orchestrator
        .stage_child_changes("child-0")
        .expect("stage");
    let a = std::path::PathBuf::from("a.rs");
    let b = std::path::PathBuf::from("b.rs");
    // Approving only ONE of two files without rejecting the other: an
    // explicit error and NOTHING merges (atomic decision check).
    let err = env
        .orchestrator
        .approve_and_merge("child-0", &cs.id(), std::slice::from_ref(&a), &[])
        .unwrap_err();
    assert!(
        matches!(err, ExecError::UndecidedPaths(_)),
        "undecided paths must error: {err:?}"
    );
    assert!(!env.owner.root.join("a.rs").exists());
    assert!(!env.owner.root.join("b.rs").exists());
    let envs = merge::merge_envelopes(&env.manager, env.parent, "run-partial", "child-0").unwrap();
    assert!(
        envs.is_empty(),
        "no durable record after the atomic decision error"
    );
    // Full decision: approve a, durably reject b.
    let outcome = env
        .orchestrator
        .approve_and_merge(
            "child-0",
            &cs.id(),
            std::slice::from_ref(&a),
            std::slice::from_ref(&b),
        )
        .expect("merge");
    assert_eq!(outcome.merged, vec![a.clone()]);
    assert_eq!(outcome.rejected, vec![b.clone()]);
    assert!(outcome.conflicts.is_empty());
    // The rejection is durable: a LATER decision that revises the durable
    // one (swap what was rejected into the approved set) is refused, and
    // b NEVER lands.
    let err = env
        .orchestrator
        .approve_and_merge(
            "child-0",
            &cs.id(),
            std::slice::from_ref(&b),
            std::slice::from_ref(&a),
        )
        .unwrap_err();
    assert!(
        matches!(err, ExecError::Conflict(_)),
        "revising the durable decision must be refused: {err:?}"
    );
    assert!(!env.owner.root.join("b.rs").exists());
    // Digest preservation: the merged parent file IS the child file.
    assert_eq!(owner_read(&env, "a.rs"), b"child-a");
    let snap = faktor_fs::snapshot_tree(&env.owner.root, 100).unwrap();
    let a_entry = snap
        .iter()
        .find(|e| e.path == std::path::Path::new("a.rs"))
        .expect("merged file present");
    assert_eq!(a_entry.hash, file_hash(b"child-a"));
    assert_eq!(
        cs.files.iter().find(|f| f.path == a).unwrap().child_hash,
        Some(a_entry.hash)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn merge_crash_after_record_before_applies_replays_complete_cas_applies() {
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(dir.path(), empty_script(), 1));
    // The seam is armed for the whole execution but only fires inside
    // approve_and_merge (execute_task never checks it).
    let mut config = base_config(&env, "run-record-crash");
    config.crash_seam = Some(CrashSeam::AfterMergeRecord);
    let outcome = run_exec(&env, isolated_plan(), config, vec![spec("impl")])
        .await
        .expect("execution succeeds (the merge seam does not fire here)");
    assert!(outcome.complete);
    let child = child_dir(&env, "run-record-crash", "child-0");
    std::fs::create_dir_all(child.join("src")).unwrap();
    std::fs::write(child.join("src/a.rs"), b"child-a").unwrap();
    std::fs::write(child.join("src/b.rs"), b"child-b").unwrap();
    let cs = env
        .orchestrator
        .stage_child_changes("child-0")
        .expect("stage");
    let a = std::path::PathBuf::from("src/a.rs");
    let b = std::path::PathBuf::from("src/b.rs");
    // Crash right after the durable merge record, before any apply.
    let err = env
        .orchestrator
        .approve_and_merge("child-0", &cs.id(), &[a.clone(), b.clone()], &[])
        .unwrap_err();
    assert!(matches!(err, ExecError::InjectedCrashSeam(_)), "{err:?}");
    assert!(!env.owner.root.join("src/a.rs").exists());
    assert!(!env.owner.root.join("src/b.rs").exists());
    let envs =
        merge::merge_envelopes(&env.manager, env.parent, "run-record-crash", "child-0").unwrap();
    assert_eq!(envs.len(), 1);
    assert!(
        envs[0].in_flight(),
        "in-flight record is durable before applies"
    );
    // Replay with the SAME decision completes the CAS applies.
    let outcome = env
        .orchestrator
        .approve_and_merge("child-0", &cs.id(), &[a.clone(), b.clone()], &[])
        .expect("replay completes");
    assert_eq!(outcome.merged, vec![a.clone(), b.clone()]);
    assert!(outcome.conflicts.is_empty());
    assert_eq!(owner_read(&env, "src/a.rs"), b"child-a");
    assert_eq!(owner_read(&env, "src/b.rs"), b"child-b");
    let envs =
        merge::merge_envelopes(&env.manager, env.parent, "run-record-crash", "child-0").unwrap();
    assert_eq!(envs[0].status, merge::MergeStatus::Applied);
    assert!(envs[0].finished_ms.is_some());
    // A further replay is idempotent (all AlreadyCurrent).
    let replay = env
        .orchestrator
        .approve_and_merge("child-0", &cs.id(), &[a.clone(), b.clone()], &[])
        .expect("idempotent replay");
    assert_eq!(replay.merged.len(), 2);
    assert_eq!(owner_read(&env, "src/a.rs"), b"child-a");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn merge_crash_mid_applies_replay_skips_already_applied_and_reconciles() {
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(dir.path(), empty_script(), 1));
    let mut config = base_config(&env, "run-mid-crash");
    config.crash_seam = Some(CrashSeam::MergeApply { after: 1 });
    let outcome = run_exec(&env, isolated_plan(), config, vec![spec("impl")])
        .await
        .expect("execution succeeds");
    assert!(outcome.complete);
    let child = child_dir(&env, "run-mid-crash", "child-0");
    std::fs::write(child.join("m0.rs"), b"child-m0").unwrap();
    std::fs::write(child.join("m1.rs"), b"child-m1").unwrap();
    std::fs::write(child.join("m2.rs"), b"child-m2").unwrap();
    let cs = env
        .orchestrator
        .stage_child_changes("child-0")
        .expect("stage");
    let paths: Vec<std::path::PathBuf> = cs.files.iter().map(|f| f.path.clone()).collect();
    assert_eq!(paths.len(), 3);
    let err = env
        .orchestrator
        .approve_and_merge("child-0", &cs.id(), &paths, &[])
        .unwrap_err();
    assert!(matches!(err, ExecError::InjectedCrashSeam(_)), "{err:?}");
    // Deterministic staged order: the FIRST path was applied, the rest not.
    assert_eq!(owner_read(&env, "m0.rs"), b"child-m0");
    assert!(!env.owner.root.join("m1.rs").exists());
    assert!(!env.owner.root.join("m2.rs").exists());
    let envs =
        merge::merge_envelopes(&env.manager, env.parent, "run-mid-crash", "child-0").unwrap();
    assert!(
        envs[0].in_flight(),
        "record stays in-flight after the crash"
    );
    // Replay: m0 already holds the child digest -> AlreadyCurrent, m1/m2
    // apply through the CAS. Conflicting content would surface, never skip.
    let outcome = env
        .orchestrator
        .approve_and_merge("child-0", &cs.id(), &paths, &[])
        .expect("replay reconciles");
    assert_eq!(outcome.merged.len(), 3, "{outcome:?}");
    assert!(outcome.conflicts.is_empty());
    assert_eq!(owner_read(&env, "m0.rs"), b"child-m0");
    assert_eq!(owner_read(&env, "m1.rs"), b"child-m1");
    assert_eq!(owner_read(&env, "m2.rs"), b"child-m2");
    let envs =
        merge::merge_envelopes(&env.manager, env.parent, "run-mid-crash", "child-0").unwrap();
    assert_eq!(envs[0].status, merge::MergeStatus::Applied);
    assert_eq!(envs[0].merged_count, 3);
    // A DIFFERENT decision on replay is refused: decisions are durable.
    let err = env
        .orchestrator
        .approve_and_merge("child-0", &cs.id(), &paths[..1], &paths[1..])
        .unwrap_err();
    assert!(matches!(err, ExecError::Conflict(_)), "{err:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn oversized_change_set_fails_loudly_and_leaves_the_parent_untouched() {
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(dir.path(), empty_script(), 1));
    write_owner_file(&env, "keep.rs", b"parent-keep");
    run_isolated_done(&env, "run-oversize").await;
    let child = child_dir(&env, "run-oversize", "child-0");
    for i in 0..(MAX_CHANGES + 1) {
        std::fs::write(child.join(format!("f{i:05}.rs")), format!("x{i}")).unwrap();
    }
    let err = env.orchestrator.stage_child_changes("child-0").unwrap_err();
    assert!(
        matches!(err, ExecError::Oversized(_)),
        "a change set beyond the cap must fail loudly: {err:?}"
    );
    assert!(err.to_string().contains("2000"), "{err:?}");
    // Nothing staged: the structured result carries no change set, and the
    // parent tree is untouched.
    let result = env
        .orchestrator
        .child_result("child-0")
        .expect("child result readable");
    assert!(
        result.change_set.is_none(),
        "oversized set must not be staged"
    );
    assert_eq!(owner_read(&env, "keep.rs"), b"parent-keep");
    assert!(
        env.orchestrator
            .manager
            .get_session(env.parent)
            .unwrap()
            .unwrap()
            .memory_facts()
            .unwrap()
            .iter()
            .all(|(kind, key, _)| {
                !(kind == crate::runtime::merge::KIND_CS)
                    || !key.starts_with("run-oversize/child-0/")
            }),
        "no change-set rows may exist after the oversized refusal"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reviewer_spawn_copies_whole_files_under_concurrent_cas_writers_and_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(dir.path(), empty_script(), 1));
    let payload_a = format!("AAAAAAAA-{}", "x".repeat(8192));
    let payload_b = format!("BBBBBBBB-{}", "y".repeat(8192));
    write_owner_file(&env, "f1.bin", payload_a.as_bytes());
    write_owner_file(&env, "f2.bin", payload_a.as_bytes());
    run_isolated_done(&env, "run-review").await;
    // Reviewer spawn while a writer keeps CAS-replacing the parent files:
    // the copy must succeed and every copied file must be a WHOLE payload
    // (the writer's completed atomic result or the pre-state — never torn).
    let fs_service = faktor_fs::WorkspaceFileService::new();
    let parent_handle = fs_service
        .open(
            faktor_core::id::WorkspaceId::new(7777),
            env.owner.root.clone(),
        )
        .expect("parent workspace open");
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let writer = {
        let stop = stop.clone();
        let handle = parent_handle.clone();
        let pa = payload_a.clone();
        let pb = payload_b.clone();
        std::thread::spawn(move || {
            let mut turn = 0usize;
            while !stop.load(std::sync::atomic::Ordering::SeqCst) {
                let (first, second) = if turn.is_multiple_of(2) {
                    (pa.as_bytes(), pb.as_bytes())
                } else {
                    (pb.as_bytes(), pa.as_bytes())
                };
                for (name, content) in [("f1.bin", first), ("f2.bin", second)] {
                    if let Ok(d) = handle.read(std::path::Path::new(name), 1_000_000) {
                        let _ = handle.write_atomic_cas(
                            std::path::Path::new(name),
                            d.full_hash().expect("payloads fit the 1 MB read bound"),
                            content,
                        );
                    }
                }
                turn += 1;
            }
        })
    };
    let reviewer = env
        .orchestrator
        .spawn_reviewer("child-0")
        .expect("reviewer spawn must succeed under the concurrent writer");
    stop.store(true, std::sync::atomic::Ordering::SeqCst);
    writer.join().unwrap();
    assert!(reviewer.child_id.starts_with("review-"));
    assert_eq!(reviewer.ownership, ChildOwnership::IsolatedWorktree);
    assert_eq!(reviewer.kind, WorkKind::Review);
    assert_eq!(
        reviewer.base_snapshot_id.as_deref(),
        Some(merge::base_id_of(&reviewer.child_id).as_str())
    );
    let reviewer_root = child_dir(&env, "run-review", &reviewer.child_id);
    let copied = faktor_fs::snapshot_tree(&reviewer_root, 100).unwrap();
    // The racing CAS writer may hold an in-flight `.kp-tmp-*` temp in the
    // OWNER root at directory-list time, so the reviewer copy legitimately
    // contains such an artifact beside the real files. The invariant under
    // test is the REAL entries: each is a WHOLE payload (the writer's
    // completed atomic result or the pre-state — never a torn real file).
    // The kp-tmp artifact is the race itself, not a torn f1/f2.
    let is_tmp = |p: &std::path::Path| {
        p.file_name()
            .map(|n| n.to_string_lossy().contains(".kp-tmp-"))
            .unwrap_or(false)
    };
    let mut real_files: Vec<&faktor_fs::SnapshotEntry> = Vec::new();
    for e in &copied {
        if !is_tmp(&e.path) {
            real_files.push(e);
        }
    }
    assert_eq!(
        real_files.len(),
        2,
        "reviewer tree must hold exactly f1.bin + f2.bin: {:?}",
        copied.iter().map(|e| e.path.clone()).collect::<Vec<_>>()
    );
    let mut names: Vec<String> = real_files
        .iter()
        .map(|e| e.path.to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(names, vec!["f1.bin".to_string(), "f2.bin".to_string()]);
    for e in &real_files {
        let text = String::from_utf8(std::fs::read(reviewer_root.join(&e.path)).unwrap()).unwrap();
        assert!(
            text == payload_a || text == payload_b,
            "torn reviewer copy of {}: {:?}",
            e.path.display(),
            text
        );
    }
    // The reviewer's durable base rows equal the copied content exactly —
    // for the real files; a raced temp artifact may carry its own base row
    // (the base records the copy manifest as-is).
    let base = merge::read_base_map(
        &env.manager,
        env.parent,
        "run-review",
        &reviewer.child_id,
        "parent",
    )
    .unwrap()
    .expect("reviewer base rows recorded");
    let mut by_path: std::collections::HashMap<std::path::PathBuf, _> = base.into_iter().collect();
    for e in &real_files {
        assert_eq!(
            by_path.remove(&e.path),
            Some(e.hash),
            "base rows must equal the copied manifest for {}",
            e.path.display()
        );
    }
    for (path, _) in by_path {
        assert!(is_tmp(&path), "unexpected extra base row {:?}", path);
    }
    assert_registry_consistent(&env, "run-review");
    // Manager reopen: reviewer rows + base rows survive and the zero-orphan
    // invariant holds on the reopened store.
    let parent = env.parent;
    drop(env);
    let manager =
        SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
    let violations =
        OrchestratorRuntime::registry_violations(manager.clone(), parent, "run-review");
    assert!(
        violations.is_empty(),
        "violations after reopen with reviewer rows: {violations:?}"
    );
    let rows = OrchestratorRuntime::registry_rows(manager.clone(), parent, "run-review").unwrap();
    assert!(
        rows.iter().any(|r| r.child_id == reviewer.child_id),
        "reviewer registry row survives reopen"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reviewer_spawn_during_an_inflight_merge_sees_only_whole_states() {
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(dir.path(), empty_script(), 1));
    // Parent has a real base file; the child proposes a new version.
    write_owner_file(&env, "m0.rs", b"base-v0");
    let mut config = base_config(&env, "run-inflight");
    config.crash_seam = Some(CrashSeam::AfterMergeRecord);
    let outcome = run_exec(&env, isolated_plan(), config, vec![spec("impl")])
        .await
        .expect("execution ok");
    assert!(outcome.complete);
    let child = child_dir(&env, "run-inflight", "child-0");
    std::fs::write(child.join("m0.rs"), b"child-version-v1").unwrap();
    let cs = env
        .orchestrator
        .stage_child_changes("child-0")
        .expect("stage");
    let m0 = std::path::PathBuf::from("m0.rs");
    // Crash AFTER the durable merge record, BEFORE any apply: the parent
    // still holds the pre-state and the merge is durably in flight.
    let err = env
        .orchestrator
        .approve_and_merge("child-0", &cs.id(), std::slice::from_ref(&m0), &[])
        .unwrap_err();
    assert!(matches!(err, ExecError::InjectedCrashSeam(_)), "{err:?}");
    let envs = merge::merge_envelopes(&env.manager, env.parent, "run-inflight", "child-0").unwrap();
    assert!(
        envs[0].in_flight(),
        "uncommitted merge is durably in flight"
    );
    assert_eq!(owner_read(&env, "m0.rs"), b"base-v0");
    // A reviewer spawned NOW copies the CURRENT parent state — consistent
    // whole files (the pre-merge base), never a partial merge.
    let reviewer = env
        .orchestrator
        .spawn_reviewer("child-0")
        .expect("reviewer spawns during the in-flight merge");
    let reviewer_root = child_dir(&env, "run-inflight", &reviewer.child_id);
    assert_eq!(
        std::fs::read(reviewer_root.join("m0.rs")).unwrap(),
        b"base-v0"
    );
    // Resume the merge (replay): now the parent holds the child version.
    env.orchestrator
        .approve_and_merge("child-0", &cs.id(), std::slice::from_ref(&m0), &[])
        .expect("replay completes the merge");
    assert_eq!(owner_read(&env, "m0.rs"), b"child-version-v1");
    assert_eq!(
        env.orchestrator
            .child_result("child-0")
            .unwrap()
            .merges
            .last()
            .unwrap()
            .status,
        merge::MergeStatus::Applied
    );
    // A SECOND reviewer spawned after the merge sees the merged state as
    // whole files too.
    let reviewer2 = env
        .orchestrator
        .spawn_reviewer("child-0")
        .expect("second reviewer spawns");
    let reviewer2_root = child_dir(&env, "run-inflight", &reviewer2.child_id);
    assert_eq!(
        std::fs::read(reviewer2_root.join("m0.rs")).unwrap(),
        b"child-version-v1"
    );
    assert_registry_consistent(&env, "run-inflight");
}

// ============================================================ audit 93: the
// single durable operation graph (read-model over plan/registry/control/
// merge rows), and audit 97: immutable env-snapshot binding at spawn with
// NO hidden-state bleed into children.

/// Raw durable rows of the env kinds under a session (bounded page walk;
/// sorted for byte comparison). `None` = all env rows of the session.
fn env_rows_of(env: &Env, prefix: Option<&str>) -> Vec<(String, String, String)> {
    let handle = env.manager.get_session(env.parent).unwrap().unwrap();
    let mut out: Vec<(String, String, String)> = Vec::new();
    let mut after: Option<(i64, String, String)> = None;
    loop {
        let page = handle.memory_facts_page(after.as_ref(), 200).unwrap();
        for (kind, key, value) in page.facts {
            if (kind == super::env::KIND_ENV_SNAPSHOT || kind == super::env::KIND_ENV_CONTENT)
                && prefix
                    .map(|p| key == p || key.starts_with(&format!("{p}/")))
                    .unwrap_or(true)
            {
                out.push((kind, key, value));
            }
        }
        match page.cursor {
            Some(c) => after = Some(c),
            None => break,
        }
    }
    out.sort();
    out
}

/// Snapshot HEADER rows only (hex-ish keys without `/cNNN` chunk rows).
fn env_snapshot_headers(env: &Env, run: &str) -> Vec<String> {
    env_rows_of(env, Some(run))
        .into_iter()
        .filter(|(kind, key, _)| kind == super::env::KIND_ENV_SNAPSHOT && !key.contains("/c"))
        .map(|(_, key, _)| key)
        .collect()
}

/// Content-header rows only (hex keys without chunk rows): the dedupe unit.
fn env_content_headers(env: &Env) -> Vec<String> {
    env_rows_of(env, None)
        .into_iter()
        .filter(|(kind, key, _)| {
            kind == super::env::KIND_ENV_CONTENT
                && key.len() == 16
                && key.chars().all(|c| c.is_ascii_hexdigit())
        })
        .map(|(_, key, _)| key)
        .collect()
}

fn owner_agents(env: &Env, content: &str) {
    write_owner_file(env, "AGENTS.md", content.as_bytes());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn graph_three_child_run_is_identical_across_crash_and_manager_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(dir.path(), empty_script(), 1));
    // Plan steps deliberately OUT of spawn order to prove the step linkage
    // is derived from the durable plan, never from spawn order.
    let p = plan(
        OwnershipModel::NoWrites,
        vec![
            wi("b", WorkKind::Analysis, &[]),
            wi("a", WorkKind::Analysis, &[]),
            wi("c", WorkKind::Analysis, &[]),
        ],
    );
    let mut config = base_config(&env, "run-graph");
    config.crash_seam = Some(CrashSeam::BeforeDrive);
    let err = run_exec(&env, p, config, vec![spec("b"), spec("a"), spec("c")])
        .await
        .expect_err("the BeforeDrive seam fires after every ready child was registered");
    assert!(matches!(err, ExecError::InjectedCrashSeam(_)), "{err:?}");
    let rows =
        OrchestratorRuntime::registry_rows(env.manager.clone(), env.parent, "run-graph").unwrap();
    assert_eq!(rows.len(), 3, "three children registered before the crash");
    // (wave A3) Identity is the DURABLE assignment contract, made BEFORE
    // any spawn: plan step 0 (item b) owns child-0, step 1 (a) child-1,
    // step 2 (c) child-2 — regardless of spawn/completion/scan order.
    let assignments =
        OrchestratorRuntime::assignment_rows(env.manager.clone(), env.parent, "run-graph").unwrap();
    assert_eq!(
        assignments.len(),
        3,
        "one durable assignment per spawn item"
    );
    for (i, (item, expected)) in ["b", "a", "c"]
        .iter()
        .zip(["child-0", "child-1", "child-2"])
        .enumerate()
    {
        let a = assignments
            .iter()
            .find(|a| a.item_id == *item)
            .unwrap_or_else(|| panic!("no assignment for plan item {item}"));
        assert_eq!(a.child_id, expected, "plan step {i} must own {expected}");
        assert_eq!(a.plan_step_index, i, "plan_step_index must be plan order");
        let r = rows
            .iter()
            .find(|r| r.child_id == a.child_id)
            .expect("assignment child must have a durable child row");
        assert_eq!(r.item_id, *item);
    }
    let child_of_item = |item: &str| {
        rows.iter()
            .find(|r| r.item_id == item)
            .expect("row per item")
            .clone()
    };
    // Steering history by ITEM (never by registry position): the b-child
    // gets one APPLIED and one PENDING control; the a-child a pending
    // Retry; the c-child stays clean.
    let b = child_of_item("b");
    let a = child_of_item("a");
    let session_of = |r: &ChildRuntime| {
        env.manager
            .get_session(SessionId::new(r.session_id))
            .unwrap()
            .unwrap()
    };
    let pause = session_of(&b)
        .orchestrator_ctl_enqueue(ChildControl::Pause)
        .unwrap();
    let steer = session_of(&b)
        .orchestrator_ctl_enqueue(ChildControl::Steer {
            note: "focus on the API surface".into(),
        })
        .unwrap();
    session_of(&b).orchestrator_ctl_ack(pause.seq).unwrap();
    let retry = session_of(&a)
        .orchestrator_ctl_enqueue(ChildControl::Retry)
        .unwrap();
    let graph1 = env
        .orchestrator
        .operation_graph(env.parent)
        .expect("graph assembles before the reopen");
    assert_eq!(graph1.root.plan_id, "run-graph");
    assert_eq!(graph1.root.goal, "Ship the feature");
    assert_eq!(graph1.root.state, WorkState::Running);
    // Children are ordered by PLAN STEP, not spawn order: step 0 = item b
    // (spawned first anyway), then a, then c.
    let steps: Vec<Option<usize>> = graph1.children.iter().map(|c| c.plan_step_index).collect();
    assert_eq!(steps, vec![Some(0), Some(1), Some(2)]);
    let by_step = |i: usize| &graph1.children[i];
    let (c0ev, c0applied): (Vec<String>, Vec<bool>) = by_step(0)
        .steer_events
        .iter()
        .map(|e| (format!("{:?}", e.kind), e.applied_ms.is_some()))
        .unzip();
    assert_eq!(
        c0ev,
        vec![
            "Pause".to_string(),
            "Steer { note: \"focus on the API surface\" }".to_string()
        ]
    );
    assert_eq!(c0applied, vec![true, false]);
    assert_eq!(by_step(0).steer_events[0].seq, pause.seq);
    assert_eq!(by_step(0).steer_events[1].seq, steer.seq);
    assert_eq!(by_step(1).steer_events.len(), 1);
    assert_eq!(by_step(1).steer_events[0].seq, retry.seq);
    assert!(by_step(1).steer_events[0].applied_ms.is_none());
    assert!(by_step(2).steer_events.is_empty());
    // Each child node is the ASSIGNED child of its plan step — the node at
    // plan step i is child-{i} and names the real session + worktree +
    // durable state of THAT child (identity never comes from registry
    // position or rows.iter().enumerate()).
    for (i, (item, expected)) in ["b", "a", "c"]
        .iter()
        .zip(["child-0", "child-1", "child-2"])
        .enumerate()
    {
        let node = by_step(i);
        assert_eq!(node.child_id, expected, "step {i} node must be {expected}");
        let r = rows
            .iter()
            .find(|r| r.child_id == expected)
            .expect("child row exists");
        assert_eq!(r.item_id, *item);
        assert_eq!(node.session_id, r.session_id);
        assert_eq!(node.worktree_id, r.worktree_id);
        assert_eq!(node.state, ChildState::Running);
        assert!(
            r.env_snapshot_id.as_deref() == Some(&format!("env-{expected}")),
            "every child is env-bound at spawn"
        );
    }
    // The registry holds exactly the assigned children: no dangling child
    // rows on the assignment side and no assignment without its child row.
    let assigned: std::collections::BTreeSet<&str> =
        assignments.iter().map(|a| a.child_id.as_str()).collect();
    assert_eq!(
        rows.iter()
            .map(|r| r.child_id.as_str())
            .collect::<std::collections::BTreeSet<_>>(),
        assigned,
        "child rows and assignment rows must agree one-to-one"
    );
    // Manager reopen: the graph is assembled from DURABLE rows only and is
    // bit-for-bit identical (same children, same steering, same order).
    let parent = env.parent;
    drop(env);
    let env2 = Arc::new(open_env(dir.path(), empty_script(), 1));
    let graph2 = env2
        .orchestrator
        .operation_graph(parent)
        .expect("graph assembles after the reopen");
    assert_eq!(
        graph2, graph1,
        "graph must survive a manager reopen bit-for-bit"
    );
    assert_registry_consistent(&env2, "run-graph");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn graph_root_state_derivation_matches_reattach_semantics() {
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(dir.path(), empty_script(), 1));
    let p = plan(
        OwnershipModel::NoWrites,
        vec![
            wi("a", WorkKind::Analysis, &[]),
            wi("b", WorkKind::Analysis, &["a"]),
        ],
    );
    let mut config = base_config(&env, "run-states");
    config.crash_seam = Some(CrashSeam::AfterChildTerminal);
    let err = run_exec(&env, p, config, vec![spec("a"), spec("b")])
        .await
        .expect_err("seam fires after child a reached Done");
    assert!(matches!(err, ExecError::InjectedCrashSeam(_)), "{err:?}");
    let graph = env.orchestrator.operation_graph(env.parent).unwrap();
    let states: Vec<String> = graph
        .root
        .work_items
        .iter()
        .map(|w| format!("{:?}", w.state))
        .collect();
    assert_eq!(states, vec!["Done".to_string(), "Pending".to_string()]);
    assert_eq!(graph.children.len(), 1);
    assert_eq!(graph.children[0].state, ChildState::Done);
    // The derived view equals the re-attached executor's view: b gets
    // admitted and the run completes.
    let outcome = tokio::time::timeout(
        std::time::Duration::from_secs(90),
        env.orchestrator.reattach(
            env.parent,
            "run-states",
            Ceilings::default(),
            base_config(&env, "x").parent_caps,
            "m".to_string(),
            env.isolated_root.clone(),
            None,
        ),
    )
    .await
    .expect("bound")
    .expect("reattach completes");
    assert!(outcome.complete);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn graph_merge_field_reflects_durable_merged_rejected_and_conflicts() {
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(dir.path(), empty_script(), 1));
    // One run, one child: base snapshot records a.rs v1 + keep.rs/drop.rs
    // ABSENT at spawn.
    write_owner_file(&env, "src/a.rs", b"parent-v1");
    run_isolated_done(&env, "run-merge-graph").await;
    // The parent moves on AFTER the base snapshot (a.rs conflict source)...
    write_owner_file(&env, "src/a.rs", b"parent-moved-on-v2");
    // ...and the child's worktree proposes its own files.
    let child = child_dir(&env, "run-merge-graph", "child-0");
    std::fs::create_dir_all(child.join("src")).unwrap();
    std::fs::write(child.join("src/a.rs"), b"child-a").unwrap();
    std::fs::write(child.join("keep.rs"), b"child-keep").unwrap();
    std::fs::write(child.join("drop.rs"), b"child-drop").unwrap();
    let cs = env
        .orchestrator
        .stage_child_changes("child-0")
        .expect("stage");
    let a = std::path::PathBuf::from("src/a.rs");
    let keep = std::path::PathBuf::from("keep.rs");
    let drop_path = std::path::PathBuf::from("drop.rs");
    let outcome = env
        .orchestrator
        .approve_and_merge(
            "child-0",
            &cs.id(),
            &[a.clone(), keep.clone()],
            std::slice::from_ref(&drop_path),
        )
        .expect("partial merge applies");
    assert_eq!(outcome.merged, vec![keep.clone()]);
    assert_eq!(outcome.rejected, vec![drop_path.clone()]);
    assert_eq!(outcome.conflicts.len(), 1, "{outcome:?}");
    assert_eq!(outcome.conflicts[0].0, a);
    // The graph's merge field mirrors the durable parts: merged, rejected
    // AND conflicts of the same merge record.
    let graph = env.orchestrator.operation_graph(env.parent).unwrap();
    let m = graph.children[0].merge.as_ref().expect("merge recorded");
    assert_eq!(m.change_set_id, cs.id());
    assert_eq!(m.merged, vec![keep.clone()]);
    assert_eq!(m.rejected, vec![drop_path.clone()]);
    assert_eq!(m.conflicts.len(), 1);
    assert_eq!(m.conflicts[0].0, std::path::PathBuf::from("src/a.rs"));
    assert!(m.conflicts[0].1.contains("base snapshot"), "{m:?}");
    // Durable decisions are immutable: a DIFFERENT decision on the same
    // change set is refused loudly (replay must carry the identical
    // approved/rejected sets).
    let err = env
        .orchestrator
        .approve_and_merge(
            "child-0",
            &cs.id(),
            &[a.clone(), drop_path.clone()],
            std::slice::from_ref(&keep),
        )
        .unwrap_err();
    assert!(matches!(err, ExecError::Conflict(_)), "{err:?}");
    // Manager reopen: the merge field is a pure read of the rows.
    let parent = env.parent;
    drop(env);
    let env2 = Arc::new(open_env(dir.path(), empty_script(), 1));
    let graph2 = env2.orchestrator.operation_graph(parent).unwrap();
    assert_eq!(graph2.children[0].merge, graph.children[0].merge);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn graph_refuses_hostile_sessions_ambiguous_runs_and_tampered_rows() {
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(dir.path(), empty_script(), 1));
    let p = plan(
        OwnershipModel::NoWrites,
        vec![wi("a", WorkKind::Analysis, &[])],
    );
    for run in ["run-hostile-1", "run-hostile-2"] {
        let mut config = base_config(&env, run);
        config.crash_seam = Some(CrashSeam::BeforeDrive);
        let err = run_exec(&env, p.clone(), config, vec![spec("a")])
            .await
            .expect_err("seam fires");
        assert!(matches!(err, ExecError::InjectedCrashSeam(_)), "{err:?}");
    }
    // Unknown parent: typed NotFound (404-equivalent).
    let err = env
        .orchestrator
        .operation_graph(SessionId::new(9999))
        .unwrap_err();
    assert!(matches!(err, GraphError::NotFound(_)), "{err:?}");
    // Ambiguous parent (two runs): loud Conflict naming both runs.
    let err = env.orchestrator.operation_graph(env.parent).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        matches!(err, GraphError::Conflict(_))
            && msg.contains("run-hostile-1")
            && msg.contains("run-hostile-2"),
        "{err:?}"
    );
    // Run-scoped graphs still work, deterministically per run.
    let g1 = env
        .orchestrator
        .operation_graph_run(env.parent, "run-hostile-1")
        .unwrap();
    assert_eq!(g1.root.plan_id, "run-hostile-1");
    // A tampered plan row (garbage JSON) is a LOUD error on the run-scoped
    // read, never a silently partial graph.
    let parent = env.manager.get_session(env.parent).unwrap().unwrap();
    parent
        .upsert_memory_fact(PLAN_ROW_KIND, "run-hostile-1", "{not json")
        .unwrap();
    let err = env
        .orchestrator
        .operation_graph_run(env.parent, "run-hostile-1")
        .unwrap_err();
    assert!(
        matches!(err, GraphError::Internal(_)),
        "tampered plan row must error loudly: {err:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn graph_hostile_assignment_rows_are_typed_errors_never_silent_skips() {
    // Each hostile case gets its OWN environment: a tampered row (or a
    // deleted child row) breaks run-scoped invariants of its run, so the
    // cases must never share a parent session.
    async fn crash_run(env: &Arc<Env>, run: &str, ids: &[&str], deps: &[&[&str]]) {
        let p = plan(
            OwnershipModel::NoWrites,
            ids.iter()
                .zip(deps)
                .map(|(id, d)| wi(id, WorkKind::Analysis, d))
                .collect(),
        );
        let specs: Vec<ChildSpec> = p.work_items.iter().map(|w| spec(&w.id)).collect();
        let mut config = base_config(env, run);
        config.crash_seam = Some(CrashSeam::BeforeDrive);
        let err = run_exec(env, p, config, specs)
            .await
            .expect_err("seam fires");
        assert!(matches!(err, ExecError::InjectedCrashSeam(_)), "{err:?}");
    }
    let delete_fact = |env: &Env, kind: &str, key: &str| {
        env.manager
            .store()
            .sql_execute(&format!(
                "DELETE FROM memory_fact WHERE session_id = {} AND kind = '{kind}' AND key = '{key}'",
                env.parent.raw()
            ))
            .unwrap();
    };
    // (a) An assignment row GONE for a not-yet-spawned item (dependent) →
    // typed MissingAssignment naming the item; never a fabricated id and
    // never a silent skip (its child row never existed, so the shape check
    // cannot mask it — only the per-item assignment fetch decides).
    let dir_a = tempfile::tempdir().unwrap();
    let env_a = Arc::new(open_env(dir_a.path(), empty_script(), 1));
    crash_run(&env_a, "run-host-a", &["a", "b"], &[&[], &["a"]]).await;
    let rows_a =
        OrchestratorRuntime::registry_rows(env_a.manager.clone(), env_a.parent, "run-host-a")
            .unwrap();
    assert_eq!(
        rows_a.len(),
        1,
        "only a (no deps) was admitted before the crash"
    );
    delete_fact(&env_a, ASSIGNMENT_ROW_KIND, "run-host-a/b");
    let err = env_a
        .orchestrator
        .operation_graph_run(env_a.parent, "run-host-a")
        .unwrap_err();
    assert!(
        matches!(err, GraphError::MissingAssignment { ref item_id, .. } if item_id == "b"),
        "deleted assignment must surface as typed MissingAssignment: {err:?}"
    );
    // (b) A child row DELETED after a real spawn (its env binding exists) →
    // typed MissingChild naming the assigned child.
    let dir_b = tempfile::tempdir().unwrap();
    let env_b = Arc::new(open_env(dir_b.path(), empty_script(), 1));
    crash_run(&env_b, "run-host-b", &["a", "b", "c"], &[&[], &[], &[]]).await;
    delete_fact(&env_b, REGISTRY_ROW_KIND, "run-host-b/child-1");
    let err = env_b
        .orchestrator
        .operation_graph_run(env_b.parent, "run-host-b")
        .unwrap_err();
    assert!(
        matches!(err, GraphError::MissingChild { ref child_id, .. } if child_id == "child-1"),
        "deleted child row must surface as typed MissingChild: {err:?}"
    );
    // (c) A second assignment row for an already-assigned item (hostile
    // duplicate) → typed Conflict at compile, on the graph AND on re-attach
    // — never two children for one item.
    let dir_c = tempfile::tempdir().unwrap();
    let env_c = Arc::new(open_env(dir_c.path(), empty_script(), 1));
    crash_run(&env_c, "run-host-c", &["a", "b"], &[&[], &[]]).await;
    let dup = WorkItemAssignment {
        run_id: "run-host-c".into(),
        item_id: "a".into(),
        plan_step_index: 0,
        child_id: "child-9".into(),
    };
    let parent_h = env_c.manager.get_session(env_c.parent).unwrap().unwrap();
    parent_h
        .upsert_memory_fact(
            ASSIGNMENT_ROW_KIND,
            "run-host-c/zzz",
            &serde_json::to_string(&dup).unwrap(),
        )
        .unwrap();
    let err = env_c
        .orchestrator
        .operation_graph_run(env_c.parent, "run-host-c")
        .unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        matches!(err, GraphError::Conflict(_)) && msg.contains("duplicate"),
        "duplicate assignment must be a typed Conflict: {err:?}"
    );
    let err = env_c
        .orchestrator
        .reattach(
            env_c.parent,
            "run-host-c",
            Ceilings::default(),
            base_config(&env_c, "x").parent_caps,
            "m".into(),
            env_c.isolated_root.clone(),
            None,
        )
        .await
        .unwrap_err();
    assert!(
        matches!(err, ExecError::Conflict(_)),
        "re-attach must refuse duplicate assignments loudly: {err:?}"
    );
    // (d) An assignment re-pointed at an UNKNOWN item (tampered row) →
    // typed Conflict naming the unknown item; the original item is never
    // silently skipped.
    let dir_d = tempfile::tempdir().unwrap();
    let env_d = Arc::new(open_env(dir_d.path(), empty_script(), 1));
    crash_run(&env_d, "run-host-d", &["a"], &[&[]]).await;
    let ghost = WorkItemAssignment {
        run_id: "run-host-d".into(),
        item_id: "ghost-item".into(),
        plan_step_index: 0,
        child_id: "child-0".into(),
    };
    env_d
        .manager
        .get_session(env_d.parent)
        .unwrap()
        .unwrap()
        .upsert_memory_fact(
            ASSIGNMENT_ROW_KIND,
            "run-host-d/a",
            &serde_json::to_string(&ghost).unwrap(),
        )
        .unwrap();
    let err = env_d
        .orchestrator
        .operation_graph_run(env_d.parent, "run-host-d")
        .unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        matches!(err, GraphError::Conflict(_)) && msg.contains("ghost-item"),
        "tampered assignment must be a typed Conflict: {err:?}"
    );
}

/// Deterministic LCG for the randomized reopen test (fixed seed: the exact
/// 100-iteration sequence reproduces on every run).
fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 33
}

/// Wave A3 equality helper: two graphs are equal up to TIMESTAMPS ONLY
/// (steering applied_ms). Identity, order, states, budget, capabilities,
/// plan linkage and every non-timestamp field must match exactly.
fn assert_graphs_equal_modulo_timestamps(before: &OpGraph, after: &OpGraph, ctx: &str) {
    let norm = |g: &OpGraph| {
        let mut g = g.clone();
        for c in &mut g.children {
            for e in &mut c.steer_events {
                e.applied_ms = None;
            }
        }
        g
    };
    assert_eq!(norm(before), norm(after), "{ctx}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn graph_identity_stays_plan_order_across_100_random_jittered_crash_reopens() {
    let dir = tempfile::tempdir().unwrap();
    let mut seed: u64 = 0xA3_5EED_0001;
    for iter in 0..100u64 {
        let run = format!("run-rand-{iter:03}");
        let n = 3 + (lcg_next(&mut seed) % 3) as usize; // 3..=5 items
                                                        // Per-child jitter: random per-chunk pacing per provider stream.
                                                        // Plan shapes are randomized too: the item ORDER is a fresh random
                                                        // permutation each iteration, so child ids (plan order) never line
                                                        // up with alphabetical item order — spawn-order-based identity
                                                        // would fail here. The per-run reasoning ceiling is raised above n
                                                        // so EVERY assigned child is admitted in the first wave.
        let mut delays = Vec::with_capacity(n);
        let mut specs = Vec::with_capacity(n);
        let mut items = Vec::with_capacity(n);
        for i in 0..n {
            let id = format!("{run}-w{i}");
            items.push(wi(&id, WorkKind::Analysis, &[]));
            specs.push(spec(&id));
            delays.push(lcg_next(&mut seed) % 5);
        }
        // Random permutation of the plan order (Fisher-Yates over the LCG).
        for i in (1..n).rev() {
            let j = (lcg_next(&mut seed) as usize) % (i + 1);
            items.swap(i, j);
        }
        let p = plan(OwnershipModel::NoWrites, items);
        let env = Arc::new(open_env(dir.path(), empty_script(), 0));
        env.provider.set_per_call_delays(delays);
        // Crash the executor right after every assigned child was spawned
        // and registered (BeforeDrive: durable rows complete, NO drive ever
        // started — the crash residue is deterministic and cheap to reopen,
        // exactly like the fixed graph tests, but over a randomized plan).
        let mut config = base_config(&env, &run);
        config.ceilings.max_reasoning_active = 8;
        config.crash_seam = Some(CrashSeam::BeforeDrive);
        let err = run_exec(&env, p, config, specs)
            .await
            .expect_err("BeforeDrive fires after every ready child was spawned");
        assert!(
            matches!(err, ExecError::InjectedCrashSeam(_)),
            "iteration {iter}: {err:?}"
        );
        // Random steering rows (with random acks) on live children: the
        // durable control history carries real timestamps that must NOT
        // leak into the before/after comparison.
        let rows =
            OrchestratorRuntime::registry_rows(env.manager.clone(), env.parent, &run).unwrap();
        assert_eq!(rows.len(), n, "iteration {iter}: every item spawned");
        for row in rows.iter().filter(|c| !c.state.is_terminal()).take(2) {
            if let Some(session) = env
                .manager
                .get_session(SessionId::new(row.session_id))
                .unwrap()
            {
                if let Ok(ctl) = session.orchestrator_ctl_enqueue(ChildControl::Steer {
                    note: format!("jittered note {iter}"),
                }) {
                    if lcg_next(&mut seed).is_multiple_of(2) {
                        let _ = session.orchestrator_ctl_ack(ctl.seq);
                    }
                }
            }
        }
        let graph_before = env
            .orchestrator
            .operation_graph_run(env.parent, &run)
            .expect("graph before restart assembles");
        // No dangling either side: one child row per assignment and one
        // assignment per child row (plan children only in these runs).
        let assignments =
            OrchestratorRuntime::assignment_rows(env.manager.clone(), env.parent, &run).unwrap();
        assert_eq!(assignments.len(), n, "iteration {iter}");
        let assigned: std::collections::BTreeSet<&str> =
            assignments.iter().map(|a| a.child_id.as_str()).collect();
        assert_eq!(
            rows.iter()
                .map(|r| r.child_id.as_str())
                .collect::<std::collections::BTreeSet<_>>(),
            assigned,
            "iteration {iter}: child rows and assignments must agree one-to-one"
        );
        // Identity in plan order: node i is exactly child-{i}.
        for (i, node) in graph_before.children.iter().enumerate() {
            assert_eq!(node.plan_step_index, Some(i), "iteration {iter}");
            assert_eq!(node.child_id, format!("child-{i}"), "iteration {iter}");
        }
        let parent = env.parent;
        drop(env); // manager reopen (daemon restart)
        let env2 = Arc::new(open_env(dir.path(), empty_script(), 0));
        let graph_after = env2
            .orchestrator
            .operation_graph_run(parent, &run)
            .expect("graph after restart assembles");
        assert_graphs_equal_modulo_timestamps(
            &graph_before,
            &graph_after,
            &format!("iteration {iter}: graph before restart != after restart"),
        );
        // The restarted manager reads the SAME identity rows: node i is
        // still child-{i}, and every assignment still has its child row.
        let rows2 = OrchestratorRuntime::registry_rows(env2.manager.clone(), parent, &run).unwrap();
        assert_eq!(rows2.len(), n);
        let assigns2 =
            OrchestratorRuntime::assignment_rows(env2.manager.clone(), parent, &run).unwrap();
        assert_eq!(
            assigns2, assignments,
            "iteration {iter}: rows were re-minted"
        );
        for (i, node) in graph_after.children.iter().enumerate() {
            assert_eq!(node.child_id, format!("child-{i}"), "iteration {iter}");
            assert!(rows2.iter().any(|r| r.child_id == node.child_id));
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn crash_between_assignment_persist_and_first_spawn_resumes_with_identical_child_ids() {
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(dir.path(), empty_script(), 1));
    let p = plan(
        OwnershipModel::NoWrites,
        vec![
            wi("a", WorkKind::Analysis, &[]),
            wi("b", WorkKind::Analysis, &[]),
            wi("c", WorkKind::Analysis, &[]),
        ],
    );
    // Crash exactly between the atomic assignment commit and the first
    // spawn: the seam fires after the assignment transaction, before any
    // child session/registry row exists.
    let mut config = base_config(&env, "run-assign-crash");
    config.crash_seam = Some(CrashSeam::AfterAssignmentsPersisted);
    let err = run_exec(&env, p, config, vec![spec("a"), spec("b"), spec("c")])
        .await
        .expect_err("the seam fires before the first spawn");
    assert!(matches!(err, ExecError::InjectedCrashSeam(_)), "{err:?}");
    let before =
        OrchestratorRuntime::assignment_rows(env.manager.clone(), env.parent, "run-assign-crash")
            .unwrap();
    assert_eq!(before.len(), 3, "three durable assignments committed");
    assert!(
        OrchestratorRuntime::registry_rows(env.manager.clone(), env.parent, "run-assign-crash")
            .unwrap()
            .is_empty(),
        "NO child may spawn before the assignments committed"
    );
    // Reopen: the resumed run must spawn under the SAME durable ids — the
    // mirror re-attach may never re-mint.
    let parent = env.parent;
    drop(env);
    let env2 = Arc::new(open_env(dir.path(), empty_script(), 1));
    let outcome = tokio::time::timeout(
        std::time::Duration::from_secs(90),
        env2.orchestrator.reattach(
            parent,
            "run-assign-crash",
            Ceilings::default(),
            base_config(&env2, "x").parent_caps,
            "m".into(),
            env2.isolated_root.clone(),
            None,
        ),
    )
    .await
    .expect("bound")
    .expect("re-attach drives the pre-spawn crash residue to completion");
    assert!(outcome.complete, "{outcome:?}");
    assert_eq!(outcome.children.len(), 3);
    let after =
        OrchestratorRuntime::assignment_rows(env2.manager.clone(), parent, "run-assign-crash")
            .unwrap();
    assert_eq!(
        after, before,
        "assignment rows must never be re-minted or rewritten"
    );
    for (i, id) in ["a", "b", "c"].iter().enumerate() {
        let row = outcome
            .children
            .iter()
            .find(|c| c.item_id == *id)
            .unwrap_or_else(|| panic!("no child row for item {id}"));
        assert_eq!(
            row.child_id,
            format!("child-{i}"),
            "item {id} must keep its pre-crash id"
        );
    }
    let graph = env2
        .orchestrator
        .operation_graph_run(parent, "run-assign-crash")
        .expect("graph assembles after resume");
    for (i, node) in graph.children.iter().enumerate() {
        assert_eq!(node.plan_step_index, Some(i));
        assert_eq!(node.child_id, format!("child-{i}"));
        assert_eq!(node.state, ChildState::Done);
    }
    assert_registry_consistent(&env2, "run-assign-crash");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn env_snapshot_pins_spawn_rules_across_parent_change_reopen_and_compaction() {
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(dir.path(), empty_script(), 1));
    owner_agents(&env, "always: spawn-time rule V1\n");
    let p = plan(
        OwnershipModel::NoWrites,
        vec![wi("a", WorkKind::Analysis, &[])],
    );
    let mut config = base_config(&env, "run-env");
    config.crash_seam = Some(CrashSeam::BeforeDrive);
    run_exec(&env, p, config, vec![spec("a")])
        .await
        .expect_err("crash after the spawn bound the env");
    let rows =
        OrchestratorRuntime::registry_rows(env.manager.clone(), env.parent, "run-env").unwrap();
    let binding = rows[0]
        .env_snapshot_id
        .clone()
        .expect("child is env-bound at spawn");
    let before_compaction = env_rows_of(&env, Some("run-env"));
    // (a) The parent's AGENTS.md changes AFTER the child spawns...
    owner_agents(&env, "always: parent changed to rule V2\n");
    // ...yet the child's context rules are the spawn-time ones, provably:
    // the served evidence carries the OLD rule text.
    let pinned = env
        .orchestrator
        .pinned_context_instructions(env.parent, "run-env", "child-0")
        .expect("pinned read works");
    let active = pinned.active_for("anything", &[]);
    assert!(
        active
            .iter()
            .any(|i| i.content.contains("spawn-time rule V1")),
        "turn evidence must include the OLD rule text: {active:?}"
    );
    assert!(
        !active.iter().any(|i| i.content.contains("rule V2")),
        "the parent's later change must never bleed into the child"
    );
    // (c) The durable rows survive a REAL compaction (interior journal
    // event on the parent) and a manager reopen byte-identically.
    let parent = env.manager.get_session(env.parent).unwrap().unwrap();
    parent.submit_prompt("compact me", &[]).unwrap();
    let rec = parent
        .record_compaction_defaults(100_000, 50_000, 80_000, "deterministic_prune")
        .unwrap();
    assert!(rec.accepted, "{rec:?}");
    assert_eq!(env_rows_of(&env, Some("run-env")), before_compaction);
    let parent_raw = env.parent;
    drop(env);
    // AGENTS.md still holds V2 on disk: any live fallback would serve V2.
    let env2 = Arc::new(open_env(dir.path(), empty_script(), 1));
    let rows2 =
        OrchestratorRuntime::registry_rows(env2.manager.clone(), parent_raw, "run-env").unwrap();
    assert_eq!(rows2[0].env_snapshot_id, rows[0].env_snapshot_id);
    assert_eq!(
        env_rows_of_env2(&env2, parent_raw, "run-env"),
        before_compaction
    );
    let pinned_after = env2
        .orchestrator
        .pinned_context_instructions(parent_raw, "run-env", "child-0")
        .expect("pinned read after reopen");
    assert!(
        pinned_after
            .active_for("anything", &[])
            .iter()
            .any(|i| i.content.contains("spawn-time rule V1")),
        "a resumed turn of the child after restart must still see the spawn-time rules"
    );
    assert_eq!(pinned_after.epoch(), pinned.epoch());
    assert_eq!(binding, rows2[0].env_snapshot_id.clone().unwrap());
}

/// env_rows_of against a manager that is NOT the Env's own (reopen).
fn env_rows_of_env2(env: &Env, parent: SessionId, prefix: &str) -> Vec<(String, String, String)> {
    env_rows_of_env2_opt(env, parent, Some(prefix))
}

fn env_rows_of_env2_opt(
    env: &Env,
    parent: SessionId,
    prefix: Option<&str>,
) -> Vec<(String, String, String)> {
    let handle = env.manager.get_session(parent).unwrap().unwrap();
    let mut out: Vec<(String, String, String)> = Vec::new();
    let mut after: Option<(i64, String, String)> = None;
    loop {
        let page = handle.memory_facts_page(after.as_ref(), 200).unwrap();
        for (kind, key, value) in page.facts {
            if (kind == super::env::KIND_ENV_SNAPSHOT || kind == super::env::KIND_ENV_CONTENT)
                && prefix
                    .map(|p| key == p || key.starts_with(&format!("{p}/")))
                    .unwrap_or(true)
            {
                out.push((kind, key, value));
            }
        }
        match page.cursor {
            Some(c) => after = Some(c),
            None => break,
        }
    }
    out.sort();
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_children_at_different_epochs_read_different_rules_each_consistent() {
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(dir.path(), empty_script(), 1));
    owner_agents(&env, "always: rules epoch E1\n");
    let p = plan(
        OwnershipModel::NoWrites,
        vec![wi("a", WorkKind::Analysis, &[])],
    );
    let mut c1 = base_config(&env, "run-e1");
    c1.crash_seam = Some(CrashSeam::BeforeDrive);
    run_exec(&env, p.clone(), c1, vec![spec("a")])
        .await
        .expect_err("crash after run-e1 spawn");
    // The parent environment moves BEFORE the second child spawns: child 2
    // binds the NEW epoch, child 1 keeps the OLD one — both durable.
    owner_agents(&env, "always: rules epoch E2\n");
    let mut c2 = base_config(&env, "run-e2");
    c2.crash_seam = Some(CrashSeam::BeforeDrive);
    run_exec(&env, p, c2, vec![spec("a")])
        .await
        .expect_err("crash after run-e2 spawn");
    // Concurrent reads (both runs durable at the same time): each child
    // sees exactly its own spawn-time rules.
    let s1 = env
        .orchestrator
        .env_snapshot_of(env.parent, "run-e1", "child-0")
        .unwrap()
        .expect("run-e1 binding");
    let s2 = env
        .orchestrator
        .env_snapshot_of(env.parent, "run-e2", "child-0")
        .unwrap()
        .expect("run-e2 binding");
    assert_ne!(s1.instruction_epoch, s2.instruction_epoch);
    let i1 = env
        .orchestrator
        .pinned_context_instructions(env.parent, "run-e1", "child-0")
        .unwrap();
    let i2 = env
        .orchestrator
        .pinned_context_instructions(env.parent, "run-e2", "child-0")
        .unwrap();
    let v1 = i1.active_for("x", &[]);
    let v2 = i2.active_for("x", &[]);
    assert!(v1[0].content.contains("epoch E1"));
    assert!(v2[0].content.contains("epoch E2"));
    // And after a full manager reopen the two stay consistent and distinct.
    let parent_raw = env.parent;
    drop(env);
    let env2 = Arc::new(open_env(dir.path(), empty_script(), 1));
    let i1b = env2
        .orchestrator
        .pinned_context_instructions(parent_raw, "run-e1", "child-0")
        .unwrap();
    let i2b = env2
        .orchestrator
        .pinned_context_instructions(parent_raw, "run-e2", "child-0")
        .unwrap();
    assert_eq!(i1b.active_for("x", &[])[0].content, v1[0].content);
    assert_eq!(i2b.active_for("x", &[])[0].content, v2[0].content);
    assert_ne!(i1b.epoch(), i2b.epoch());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unchanged_env_between_spawns_dedupes_content_rows_by_rules_hash() {
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(dir.path(), empty_script(), 1));
    owner_agents(&env, "always: dedupe me\n");
    let p = plan(
        OwnershipModel::NoWrites,
        vec![wi("a", WorkKind::Analysis, &[])],
    );
    for run in ["run-d1", "run-d2"] {
        let mut c = base_config(&env, run);
        c.crash_seam = Some(CrashSeam::BeforeDrive);
        run_exec(&env, p.clone(), c, vec![spec("a")])
            .await
            .expect_err("crash after spawn");
    }
    assert_eq!(
        env_snapshot_headers(&env, "run-d1").len() + env_snapshot_headers(&env, "run-d2").len(),
        2,
        "one snapshot row per child"
    );
    assert_eq!(
        env_content_headers(&env).len(),
        1,
        "unchanged env across two spawns stores the rule bytes exactly once"
    );
    // The env changes: a second distinct content row appears.
    owner_agents(&env, "always: dedupe me now with v2\n");
    let mut c = base_config(&env, "run-d3");
    c.crash_seam = Some(CrashSeam::BeforeDrive);
    run_exec(&env, p, c, vec![spec("a")])
        .await
        .expect_err("crash after spawn");
    assert_eq!(
        env_content_headers(&env).len(),
        2,
        "changed env adds exactly one new content row (no rewrite of the old)"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn oversized_rule_env_fails_the_spawn_loudly_with_no_rows_or_sessions() {
    // Path cap: more rule files than MAX_SNAPSHOT_PATHS (64).
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(dir.path(), empty_script(), 1));
    for i in 0..65 {
        owner_write(
            &env,
            &format!(".cursor/rules/f{i:04}.mdc"),
            format!("# Scope: k{i}\nrule\n").as_bytes(),
        );
    }
    let p = plan(
        OwnershipModel::NoWrites,
        vec![wi("a", WorkKind::Analysis, &[])],
    );
    let err = run_exec(
        &env,
        p.clone(),
        base_config(&env, "run-cap"),
        vec![spec("a")],
    )
    .await
    .expect_err("oversized env must fail the spawn loudly");
    assert!(
        matches!(err, ExecError::Oversized(_)) && err.to_string().contains("rule files"),
        "{err:?}"
    );
    assert!(
        OrchestratorRuntime::registry_rows(env.manager.clone(), env.parent, "run-cap")
            .unwrap()
            .is_empty(),
        "no registry row: the failed spawn registered nothing"
    );
    assert!(
        env_rows_of(&env, Some("run-cap")).is_empty(),
        "no env rows: the failed capture stored nothing"
    );
    // Total-byte cap on a FRESH data dir (the path-cap run above polluted
    // its owner tree): 10 fat rule files, each read saturating at the
    // 64 KiB per-file bound -> 640 KiB > the 512 KiB total bound.
    let dir2 = tempfile::tempdir().unwrap();
    let env2 = Arc::new(open_env(dir2.path(), empty_script(), 1));
    for i in 0..10 {
        owner_write(
            &env2,
            &format!(".cursor/rules/g{i:02}.mdc"),
            &vec![b'x'; 64 * 1024],
        );
    }
    let err2 = run_exec(&env2, p, base_config(&env2, "run-bytes"), vec![spec("a")])
        .await
        .expect_err("oversized total env bytes must fail loudly");
    assert!(
        matches!(err2, ExecError::Oversized(_)) && err2.to_string().contains("total bytes"),
        "{err2:?}"
    );
    assert!(
        OrchestratorRuntime::registry_rows(env2.manager.clone(), env2.parent, "run-bytes")
            .unwrap()
            .is_empty(),
        "no registry row after the byte-cap failure"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bound_child_refuses_live_fallback_when_binding_or_rows_are_gone_or_tampered() {
    let dir = tempfile::tempdir().unwrap();
    let env = Arc::new(open_env(dir.path(), empty_script(), 1));
    owner_agents(&env, "always: binding rule V1\n");
    let p = plan(
        OwnershipModel::NoWrites,
        vec![wi("a", WorkKind::Analysis, &[])],
    );
    let mut config = base_config(&env, "run-d");
    config.crash_seam = Some(CrashSeam::BeforeDrive);
    run_exec(&env, p, config, vec![spec("a")])
        .await
        .expect_err("crash after spawn");
    let parent = env.manager.get_session(env.parent).unwrap().unwrap();
    // The live env is now V2: any silent live fallback would be detectable.
    owner_agents(&env, "always: binding rule V2\n");
    let registry_json = parent
        .memory_facts()
        .unwrap()
        .into_iter()
        .find(|(k, key, _)| k == REGISTRY_ROW_KIND && key == "run-d/child-0")
        .map(|(_, _, v)| v)
        .expect("registry row");
    let mut row_json: serde_json::Value = serde_json::from_str(&registry_json).unwrap();
    // (d-i) A binding that names a snapshot whose durable rows are gone:
    // loud NotFound, never a live read.
    row_json["env_snapshot_id"] = serde_json::json!("env-ghost");
    parent
        .upsert_memory_fact(REGISTRY_ROW_KIND, "run-d/child-0", &row_json.to_string())
        .unwrap();
    let err = env
        .orchestrator
        .pinned_context_instructions(env.parent, "run-d", "child-0")
        .expect_err("ghost binding refused");
    assert!(
        matches!(err, ExecError::NotFound(_)) && err.to_string().contains("fallback"),
        "{err:?}"
    );
    // (d-ii) A binding REMOVED entirely: incomplete spawn, loud, typed.
    row_json["env_snapshot_id"] = serde_json::Value::Null;
    parent
        .upsert_memory_fact(REGISTRY_ROW_KIND, "run-d/child-0", &row_json.to_string())
        .unwrap();
    let err = env
        .orchestrator
        .pinned_context_instructions(env.parent, "run-d", "child-0")
        .expect_err("missing binding refused");
    assert!(
        matches!(err, ExecError::InvalidState(_)) && err.to_string().contains("fallback"),
        "{err:?}"
    );
    // (d-iii) A TAMPERED content row (bytes that no longer hash to the
    // recorded records): loud InvalidState, and the live V2 file is never
    // served instead.
    let mut row_json: serde_json::Value = serde_json::from_str(&registry_json).unwrap();
    row_json["env_snapshot_id"] = serde_json::json!("env-child-0");
    parent
        .upsert_memory_fact(REGISTRY_ROW_KIND, "run-d/child-0", &row_json.to_string())
        .unwrap();
    let facts = parent.memory_facts().unwrap();
    let tamper_key = facts
        .iter()
        .find(|(k, key, _)| {
            k == super::env::KIND_ENV_CONTENT
                && key.len() == 16
                && key.chars().all(|c| c.is_ascii_hexdigit())
        })
        .map(|(_, key, _)| key.clone())
        .expect("content header row");
    let chunk_key = facts
        .iter()
        .find(|(k, key, _)| {
            k == super::env::KIND_ENV_CONTENT && key.starts_with(&format!("{tamper_key}/c"))
        })
        .map(|(_, key, _)| key.clone())
        .expect("content chunk row");
    let chunk_value = facts
        .iter()
        .find(|(k, key, _)| k == super::env::KIND_ENV_CONTENT && *key == chunk_key)
        .map(|(_, _, v)| v.clone())
        .unwrap();
    let evil = chunk_value.replace("binding rule V1", "binding rule EVIL");
    assert_ne!(evil, chunk_value, "the tamper changes the stored bytes");
    parent
        .upsert_memory_fact(super::env::KIND_ENV_CONTENT, &chunk_key, &evil)
        .unwrap();
    let err = env
        .orchestrator
        .pinned_context_instructions(env.parent, "run-d", "child-0")
        .expect_err("tampered content refused");
    assert!(
        matches!(err, ExecError::InvalidState(_)) && err.to_string().contains("hash"),
        "{err:?}"
    );
    assert!(
        !err.to_string().contains("binding rule"),
        "the live V2 file is never served instead: {err:?}"
    );
}
