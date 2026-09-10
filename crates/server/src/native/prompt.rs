//! `PromptExecutionService` — the ONE product execution entry for ordinary
//! prompts and task starts (audit: "normal chat still bypasses the
//! shadow/TaskExecutor architecture").
//!
//! Every protocol adapter (Native, SDK compatibility, ACP, VS Code,
//! JetBrains) translates its DTOs into calls on this service:
//!
//! - [`PromptExecutionService::prompt`] — an ordinary chat prompt becomes an
//!   IN-SESSION run through the daemon's [`TaskExecutor`]; under the
//!   production default ([`MutationMode::Shadow`]) a mutating prompt works
//!   in the daemon-owned shadow and only the verified integration commit
//!   writes the user checkout. No adapter ever calls `AgentRuntime`'s drive
//!   entries directly again.
//! - [`PromptExecutionService::start_task`] — an explicit (multi-work-item)
//!   task run goes through the SAME executor; the daemon allocates the
//!   isolated candidate root itself (a client never supplies a filesystem
//!   path).
//!
//! The service is a stateless facade over the daemon's ONE `TaskExecutor`
//! and ONE `SessionManager`; constructing it per call cannot create a second
//! execution authority, and both the SDK compat surface and the ACP backend
//! observe the same underlying instances (see the identity spy tests).

use std::sync::{Arc, Mutex};

use faktor_core::id::{OpId, SessionId};
use faktor_orchestrator::runtime::task_executor::{
    MutationMode, TaskExecutor, TaskRunReceipt, TaskRunRequest,
};
use faktor_orchestrator::runtime::ExecError;
use faktor_orchestrator::{WorkItem, WorkKind};
use faktor_session::SessionManager;

use crate::api::AppState;

/// One protocol-neutral ordinary prompt (the translated DTO).
#[derive(Debug, Clone, Default)]
pub struct PromptRequest {
    /// The prompt text (also the run goal).
    pub prompt: String,
    /// Attached file paths (the SDK prompt vocabulary).
    pub files: Vec<String>,
    /// Per-run model override.
    pub model: Option<String>,
    /// Acceptance criteria ridden onto the run's durable task row.
    pub criteria: Vec<String>,
    /// Per-run mutation policy override (`None` = the daemon default —
    /// shadow mutation in production).
    pub mutation_mode: Option<MutationMode>,
}

/// The receipt of one accepted ordinary prompt: the in-session run that now
/// carries it (durable linkage row + task row).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptReceipt {
    pub run_id: String,
    /// The session's durable turn op id (ordinary prompts are single-item
    /// in-session runs, so this is always present).
    pub op_id: OpId,
    pub queued: bool,
    pub accepted: bool,
}

/// What kind of call an observer saw (identity-spy tests).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptCallKind {
    Prompt,
    StartTask,
}

/// One observed call: the underlying execution authorities are the identity
/// that matters (the facade is stateless).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptCall {
    pub kind: PromptCallKind,
    pub session: SessionId,
    /// `Arc::as_ptr` of the `TaskExecutor` the call executed on.
    pub tasks_ptr: usize,
    /// `Arc::as_ptr` of the `SessionManager` the call executed on.
    pub sessions_ptr: usize,
}

type PromptObserver = Arc<dyn Fn(&PromptCall) + Send + Sync>;

static OBSERVER: Mutex<Option<PromptObserver>> = Mutex::new(None);

/// Install (or clear) the process-wide prompt observer. Test instrumentation;
/// production never installs one, and the lock is a single uncontended
/// mutex on the prompt path.
pub fn set_prompt_observer(observer: Option<PromptObserver>) {
    *OBSERVER.lock().unwrap_or_else(|p| p.into_inner()) = observer;
}

fn observe(service: &PromptExecutionService, kind: PromptCallKind, session: SessionId) {
    let observer = OBSERVER.lock().unwrap_or_else(|p| p.into_inner()).clone();
    if let Some(observer) = observer {
        observer(&PromptCall {
            kind,
            session,
            tasks_ptr: Arc::as_ptr(service.tasks()) as usize,
            sessions_ptr: Arc::as_ptr(service.sessions()) as usize,
        });
    }
}

/// The states that mean "a logical turn is occupying the session machine"
/// (the wait condition every prompt adapter uses before projecting the
/// turn's result). Everything else — Idle, ReadyForNextTurn, Completed,
/// Cancelled, FailedRecoverable/FailedPermanent, NeedsUserInput, Suspended —
/// means the accepted turn has finished (or never started).
pub fn turn_machine_busy(s: faktor_core::state::AgentState) -> bool {
    matches!(
        s,
        faktor_core::state::AgentState::Preparing
            | faktor_core::state::AgentState::BuildingContext
            | faktor_core::state::AgentState::WaitingForModel
            | faktor_core::state::AgentState::Streaming
            | faktor_core::state::AgentState::ToolRequested
            | faktor_core::state::AgentState::WaitingForPermission
            | faktor_core::state::AgentState::ExecutingTool
            | faktor_core::state::AgentState::Validating
            | faktor_core::state::AgentState::UpdatingMemory
    )
}

/// The ONE entry authority for ordinary prompts and task starts.
pub struct PromptExecutionService {
    tasks: Arc<TaskExecutor>,
    sessions: Arc<SessionManager>,
}

impl std::fmt::Debug for PromptExecutionService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PromptExecutionService")
            .finish_non_exhaustive()
    }
}

impl PromptExecutionService {
    pub fn new(tasks: Arc<TaskExecutor>, sessions: Arc<SessionManager>) -> Arc<Self> {
        Arc::new(Self { tasks, sessions })
    }

    /// The daemon's ONE executor authority (identity that all adapters
    /// share).
    pub fn tasks(&self) -> &Arc<TaskExecutor> {
        &self.tasks
    }

    /// The daemon's ONE session authority.
    pub fn sessions(&self) -> &Arc<SessionManager> {
        &self.sessions
    }

    /// The stateless facade over the SERVER deps (constructing it never
    /// builds a second authority).
    pub(crate) fn from_state(state: &AppState) -> Arc<Self> {
        Self::new(state.deps.tasks.clone(), state.deps.session.clone())
    }

    // The daemon-owned candidate root allocation is the executor's own
    // single authority: every orchestrated run allocates its isolated root
    // through [`faktor_orchestrator::runtime::task_executor::CandidateWorkspaceService`]
    // (the DTO never carries a filesystem path).

    /// Run ONE ordinary chat prompt as an in-session task run. The prompt
    /// becomes a single MUTATING work item so the daemon's mutation policy
    /// (shadow by default) gates every write the turn performs.
    pub async fn prompt(
        &self,
        session: SessionId,
        request: PromptRequest,
    ) -> Result<PromptReceipt, ExecError> {
        if request.prompt.trim().is_empty() {
            return Err(ExecError::InvalidPlan("prompt is empty".into()));
        }
        let item = WorkItem::new("main", request.prompt.clone(), WorkKind::Implementation);
        let run = TaskRunRequest {
            goal: request.prompt,
            work_items: vec![item],
            model: request.model,
            criteria: request.criteria,
            files: request.files,
            mutation_mode: request.mutation_mode,
            ..Default::default()
        };
        let receipt = self.tasks.start_task(session, run)?;
        observe(self, PromptCallKind::Prompt, session);
        let op_id = receipt.op_id.ok_or_else(|| {
            ExecError::Internal("ordinary prompt did not yield an in-session op id".into())
        })?;
        Ok(PromptReceipt {
            run_id: receipt.run_id,
            op_id,
            queued: receipt.queued,
            accepted: true,
        })
    }

    /// Start ONE explicit task run (the native task-start surface). The
    /// executor allocates the isolated root for multi-item mutating plans
    /// itself.
    pub fn start_task(
        &self,
        session: SessionId,
        request: TaskRunRequest,
    ) -> Result<TaskRunReceipt, ExecError> {
        let receipt = self.tasks.start_task(session, request)?;
        observe(self, PromptCallKind::StartTask, session);
        Ok(receipt)
    }
}
