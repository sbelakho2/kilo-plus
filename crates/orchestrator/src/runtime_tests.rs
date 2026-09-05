//! Adversarial tests for the orchestrator runtime wiring (audits 20-24).
//!
//! Children are driven by a REAL `AgentRuntime` over a REAL `SessionManager`
//! with a scripted, chunk-paced provider (no network). Every test attempts
//! to break the invariants: crashes at seams, mid-drive kills, paused
//! drives, duplicate steer application, ceiling pressure, ownership
//! overlap and zero-orphan registry checks after every crash.

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
        let delay = self.chunk_delay_ms;
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
    registry.register(provider.clone());
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
        verifier: None,
        hooks: None,
        instructions_loader: None,
        router: None,
        budget_micro: None,
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
        registry.register(provider);
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
            verifier: None,
            hooks: None,
            instructions_loader: None,
            router: None,
            budget_micro: None,
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
                        let _ =
                            handle.write_atomic_cas(std::path::Path::new(name), d.hash, content);
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
    assert_eq!(copied.len(), 2);
    for e in &copied {
        let text = String::from_utf8(std::fs::read(reviewer_root.join(&e.path)).unwrap()).unwrap();
        assert!(
            text == payload_a || text == payload_b,
            "torn reviewer copy of {}: {:?}",
            e.path.display(),
            text
        );
    }
    // The reviewer's durable base rows equal the copied content exactly.
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
    for e in &copied {
        assert_eq!(
            by_path.remove(&e.path),
            Some(e.hash),
            "base rows must equal the copied manifest for {}",
            e.path.display()
        );
    }
    assert!(by_path.is_empty());
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
