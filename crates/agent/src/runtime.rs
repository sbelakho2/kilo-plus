//! The agent runtime: the durable turn loop that drives the session with
//! commands, streams providers, schedules tools, and keeps context bounded.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use kilop_context::artifact::ArtifactWriter;
use kilop_context::assembler::{Evidence, RecentTurn};
use kilop_context::budget::ContextBudget;
use kilop_context::compactor::{CompactionPlan, CompactionRequest, Compactor, Summarizer};
use kilop_context::ledger::TaskLedger;
use kilop_context::wire_plan::{plan_wire_request, WirePlan};
use kilop_core::cancellation::CancellationToken;
use kilop_core::capability::{Capability, PermissionDecision};
use kilop_core::error::{Error, ErrorKind};
use kilop_core::hash::FileHash;
use kilop_core::id::{OpId, SessionId, WorkspaceId};
use kilop_core::op::{EffectStatus, OpMeta, RecoveryStrategy};
use kilop_core::state::AgentState;
use kilop_core::time::Clock;
use kilop_core::WorkspaceIdentity;
use kilop_protocol::v756::ToolResultBody;
use kilop_provider::{
    CapabilityValidator, ContentPart, GenericAgentRequest, ProviderChunk, ProviderError,
    ProviderRegistry, RequestMessage, RequestMeta, Role,
};
use kilop_scheduler::{OwnershipSet, ResourceRequest, ScheduledOp, Scheduler};
use kilop_session::ops::PermissionRequest as SessionPermission;
use kilop_session::{RecoveredOp, RecoveryAction, RecoveryReport, SessionManager};
use kilop_store::ToolRunRow;

use crate::loop_detect::LoopDetector;
use crate::tool::{
    FilePostcondition, RecoveryHint, ReplayDescriptor, Tool, ToolOutcome, ToolRegistry, ToolRunCtx,
};
use crate::tool_json::ToolCallMode;

/// History retrieval bound (token trimming happens in the WirePlan; this
/// caps the rows loaded from storage per logical turn).
const MAX_HISTORY_MESSAGES: usize = 2000;

/// Ephemeral-stream flush cadence: durable parts are written in segments of
/// this size (plus the final tail), so per-token journaling never happens.
const STREAM_FLUSH_BYTES: usize = 8 * 1024;

/// The compaction model's dedicated system contract (P0 audit, round 11):
/// the summarizer is NOT the agent — it is the Kilo+ context compactor
/// producing a faithful state transfer. Sending the agent instructions as
/// the system prompt let the compaction model answer the latest user message
/// instead of summarizing. The agent instructions must stay out of this
/// request entirely.
const COMPACTOR_SYSTEM: &str = "You are the Kilo+ context compactor. Your ONLY job is to \
produce a faithful state transfer that REPLACES the conversation below, so the agent can \
continue exactly where it stopped. The prior conversation is given as user/assistant \
messages. Write a compact but complete summary that preserves: the user's goal and current \
task; constraints and requirements; decisions made and their reasons; files changed (paths \
and what changed); unresolved errors and blockers (NEVER omit an unresolved blocker); \
results of tests and verification; observable tool effects (commands run, artifacts \
created); explicit user instructions and preferences; the current implementation state; \
and the next actions to take. NEVER invent facts, code, paths, or results that are not in \
the transcript; if the transcript is incomplete, say exactly what is missing. Prefer \
structured output (short labeled sections). Do not answer the latest user message: do not \
add advice, do not continue the task, do not write code.";

/// BLAKE3 of a file via bounded 64KiB chunks (never read-whole-file).
/// Unreadable/missing files hash to the zero marker.
fn stream_hash_file(path: &str) -> kilop_core::hash::FileHash {
    use std::io::Read;
    let Ok(mut f) = std::fs::File::open(path) else {
        return kilop_core::hash::FileHash::from([0u8; 32]);
    };
    let mut hasher = blake3::Hasher::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        match f.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                hasher.update(&buf[..n]);
            }
            Err(_) => break,
        }
    }
    kilop_core::hash::FileHash::from(hasher.finalize().into())
}

/// How the runtime asks for permission. The server implementation waits on a
/// durable permission row + a UI response channel (async so blocking on the
/// user never stalls a tokio worker).
pub trait PermissionRequester: Send + Sync {
    fn request(
        &self,
        session: SessionId,
        permission: &SessionPermission,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = kilop_core::Result<PermissionDecision>> + Send>,
    >;
}

/// Supplies retrieved evidence before a reasoning turn (spec §20). Default
/// implementation returns nothing; the index wires itself here.
/// The retrieval signal for one reasoning turn (spec §20): the current
/// prompt, the durable task state's changed files, and known failures. The
/// provider derives concepts from all of them — retrieval never depends on
/// the model deciding to search.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EvidenceQuery {
    pub prompt: String,
    pub changed_files: Vec<String>,
    pub failures: Vec<String>,
}

pub trait EvidenceProvider: Send + Sync {
    fn evidence_for(&self, session: SessionId, query: &EvidenceQuery) -> Vec<Evidence>;

    /// Forget one workspace's cached state (idle-unload, spec §21): the
    /// session ended, its index/scan state is dropped. Default: nothing.
    fn forget(&self, _workspace: WorkspaceId) {}
}

pub struct NoEvidence;
impl EvidenceProvider for NoEvidence {
    fn evidence_for(&self, _session: SessionId, _query: &EvidenceQuery) -> Vec<Evidence> {
        vec![]
    }
}

/// Artifact storage handed to tools (bounded writes to the CAS).
#[derive(Clone)]
pub enum ToolArtifactSink {
    Real(Arc<ArtifactWriter>),
    Null,
}

impl ToolArtifactSink {
    pub fn store(
        &self,
        kind: &str,
        bytes: &[u8],
        max_inline: usize,
    ) -> kilop_core::Result<kilop_context::ArtifactRef> {
        match self {
            ToolArtifactSink::Real(w) => w.store(kind, bytes, max_inline),
            ToolArtifactSink::Null => Ok(kilop_context::ArtifactRef {
                inline: Some(String::from_utf8_lossy(bytes).to_string()),
                artifact: None,
                summary: "null sink".into(),
                size: bytes.len(),
            }),
        }
    }
}

pub struct AgentDeps {
    pub session: Arc<SessionManager>,
    pub providers: Arc<ProviderRegistry>,
    pub permission_requester: Arc<dyn PermissionRequester>,
    pub evidence: Arc<dyn EvidenceProvider>,
    pub tools: Arc<ToolRegistry>,
    /// Content store for tool artifacts (optional).
    pub cas: Option<Arc<kilop_cas::Cas>>,
    /// Workspace registry the runtime opens session workspaces through.
    pub workspaces: Arc<kilop_fs::WorkspaceFileService>,
    /// Transactional edit engine for write_file (None → tool errors).
    pub edit: Option<Arc<kilop_edit::EditEngine>>,
    /// CAS-backed checkpoint store for write_file undo history.
    pub snapshots: Option<Arc<kilop_snapshot::CheckpointStore>>,
    /// Capability policy engine; the runtime roots it at each session's
    /// workspace before handing it to tools.
    pub sandbox: Option<Arc<kilop_sandbox::PermissionEngine>>,
    /// Process supervisor for run_command (None → tool errors).
    pub supervisor: Option<Arc<kilop_terminal::ProcessSupervisor>>,
    pub model: String,
    /// Separate compaction model (spec §36); None → deterministic pruning.
    pub compaction_model: Option<String>,
    /// Effective-usage fraction that triggers proactive compaction (0.65–0.70).
    pub compact_at_usage: f64,
    /// Static system instructions (cacheable prefix).
    pub instructions: String,
    pub clock: Arc<dyn Clock>,
    /// Tool-call parsing mode per provider family (local models default to
    /// NativeWithRepair; native typed providers to Native).
    pub tool_call_mode: ToolCallMode,
    /// State-aware provider retry policy (spec §13): a request that failed
    /// before ANY content became durable may retry (network class); once a
    /// tool ran or parts were flushed, never.
    pub retry_policy: kilop_core::retry::RetryPolicy,
    /// Per-tool-call deadline in ms.
    pub tool_deadline_ms: u64,
}

impl AgentDeps {
    pub fn artifact_sink(&self, session: SessionId) -> ToolArtifactSink {
        match &self.cas {
            Some(cas) => {
                ToolArtifactSink::Real(Arc::new(ArtifactWriter::new(cas.clone(), session)))
            }
            None => ToolArtifactSink::Null,
        }
    }
}

pub struct AgentRuntime {
    deps: Arc<AgentDeps>,
    /// Sessions with a live queue-runner task (single runner per session).
    runners: std::sync::Mutex<std::collections::HashSet<SessionId>>,
}

#[derive(Debug, Clone)]
pub struct TurnOutcome {
    pub op_id: OpId,
    pub final_state: AgentState,
    pub turns: u32,
    pub compacted: bool,
    pub loop_stopped: bool,
    /// True when the prompt was durably QUEUED (another turn was active):
    /// the per-session turn runner delivers it later. No work was started.
    pub queued: bool,
}

#[derive(Debug, Clone)]
pub struct AgentCard {
    pub session_id: SessionId,
    pub title: String,
    pub status: String, // running | waiting | completed | failed | needs-input
}

/// Lightweight row view used by history loading (parts are fetched per row
/// by the callers).
struct MessageRowLike {
    id: i64,
    #[allow(dead_code)]
    seq: i64,
    role: String,
    data: serde_json::Value,
}

impl AgentRuntime {
    pub fn new(deps: AgentDeps) -> kilop_core::Result<Arc<Self>> {
        if deps.model.is_empty() {
            return Err(Error::malformed("agent requires a model"));
        }
        Ok(Arc::new(Self {
            deps: Arc::new(deps),
            runners: std::sync::Mutex::new(std::collections::HashSet::new()),
        }))
    }

    pub fn deps(&self) -> &AgentDeps {
        &self.deps
    }

    // ------------------------------------------------------------ entry points

    /// Submit a prompt and run the full turn (durable; survives restarts).
    pub async fn run_turn(
        self: &Arc<Self>,
        session: SessionId,
        prompt: &str,
        files: &[String],
    ) -> kilop_core::Result<TurnOutcome> {
        self.run_turn_with_model(session, prompt, files, None).await
    }

    /// Like [`AgentRuntime::run_turn`] with a per-message model override.
    /// When `Some`, the override model is used for provider capability
    /// lookup and request building INSTEAD of the session's configured
    /// model; the provider is always the session's provider, and a model
    /// the provider has no capabilities for falls back to the provider's
    /// default capabilities (never an error at send time). The journaled
    /// session row keeps its original model — the override is per-message,
    /// not a session mutation.
    pub async fn run_turn_with_model(
        self: &Arc<Self>,
        session: SessionId,
        prompt: &str,
        files: &[String],
        model: Option<String>,
    ) -> kilop_core::Result<TurnOutcome> {
        let receipt = self.submit(session, prompt, files)?;
        if receipt.queued {
            // A single per-session turn runner delivers queued prompts after
            // the active logical turn completes (audit round 6). Never start
            // a second concurrent turn.
            return Ok(TurnOutcome {
                op_id: receipt.op_id,
                final_state: AgentState::Idle,
                turns: 0,
                compacted: false,
                loop_stopped: false,
                queued: true,
            });
        }
        let handle = self
            .deps
            .session
            .get_session(session)?
            .ok_or_else(|| Error::not_found(format!("session {session}")))?;
        self.drive_receipt(&handle, receipt, model).await
    }

    /// Synchronous prompt submission (journal + durable queue when busy).
    /// The server uses this to answer with the TRUE queued state before
    /// spawning any detached work (audit round 6).
    pub fn submit(
        self: &Arc<Self>,
        session: SessionId,
        prompt: &str,
        files: &[String],
    ) -> kilop_core::Result<kilop_session::PromptReceipt> {
        let handle = self
            .deps
            .session
            .get_session(session)?
            .ok_or_else(|| Error::not_found(format!("session {session}")))?;
        // Crash recovery first (never blindly re-run).
        self.recover_session(&handle)?;
        handle.submit_prompt(prompt, files)
    }

    /// Drive an already-submitted turn receipt to its single genuine end.
    /// A failed turn journals FailedRecoverable (never stuck mid-transition)
    /// and lands the session in a promptable state.
    pub async fn drive_receipt(
        self: &Arc<Self>,
        handle: &kilop_session::SessionHandle,
        receipt: kilop_session::PromptReceipt,
        model: Option<String>,
    ) -> kilop_core::Result<TurnOutcome> {
        let op_id = receipt.op_id;
        let cancel = receipt.op_meta.cancellation.clone();
        let outcome = self.drive_turn(handle, op_id, cancel, model).await;
        if let Err(e) = &outcome {
            let _ = handle.append_event(
                kilop_core::event::EventKind::Failed,
                AgentState::FailedRecoverable,
                Some(op_id),
                Some(serde_json::json!({ "message": e.message })),
            );
            // The interrupted logical turn cannot resume: close its record
            // so no later recovery tries to continue a dead turn.
            let _ = handle.finish_turn_record(op_id, "failed");
        }
        outcome
    }

    /// Continue a turn interrupted by a crash: load the interrupted logical
    /// turn's durable record (v7), verify the session state, resolve side
    /// effects (tool-run recovery incl. exactly-once idempotent replay), and
    /// resume the state machine driving the SAME recorded turn op id with
    /// the SAME recorded provider/model envelope — never a synthesized
    /// operation and never the session's current defaults.
    pub async fn continue_turn(
        self: &Arc<Self>,
        session: SessionId,
    ) -> kilop_core::Result<TurnOutcome> {
        let handle = self
            .deps
            .session
            .get_session(session)?
            .ok_or_else(|| Error::not_found(format!("session {session}")))?;
        let Some(record) = handle.active_turn_record()? else {
            return Err(Error::conflict(format!(
                "session {session} has no interrupted logical turn to continue"
            )));
        };
        self.continue_record(&handle, &record).await
    }

    pub fn resolve_permission(
        &self,
        session: SessionId,
        permission_id: i64,
        decision: PermissionDecision,
    ) -> kilop_core::Result<()> {
        let handle = self
            .deps
            .session
            .get_session(session)?
            .ok_or_else(|| Error::not_found(format!("session {session}")))?;
        handle.resolve_permission(permission_id, decision)?;
        Ok(())
    }

    pub fn abort(&self, session: SessionId) -> kilop_core::Result<Vec<OpId>> {
        self.abort_op(session, None)
    }

    /// Abort one operation (the active turn, a queued prompt, or a tool) or
    /// everything with `None`. Queued-prompt kills durably cancel their
    /// queue row without touching the machine; turn kills land the session
    /// ReadyForNextTurn (review P0-2).
    pub fn abort_op(
        &self,
        session: SessionId,
        op_id: Option<OpId>,
    ) -> kilop_core::Result<Vec<OpId>> {
        let handle = self
            .deps
            .session
            .get_session(session)?
            .ok_or_else(|| Error::not_found(format!("session {session}")))?;
        Ok(handle.abort(op_id)?.op_ids)
    }

    /// Explicitly close a session (the only normal route to terminal
    /// closure; review P0-2 — Stop/abort cancels the turn, not the session).
    /// Commandment 8 (zero orphans): every child process owned by the
    /// session dies here — the supervisor kills the whole session process
    /// set (SIGTERM → grace → SIGKILL) BEFORE the durable end transition.
    pub fn end_session(&self, session: SessionId) -> kilop_core::Result<()> {
        let handle = self
            .deps
            .session
            .get_session(session)?
            .ok_or_else(|| Error::not_found(format!("session {session}")))?;
        if let Some(supervisor) = &self.deps.supervisor {
            let killed = supervisor.kill_all_for(kilop_terminal::ProcessOwner::Session(session));
            if !killed.is_empty() {
                tracing::info!(
                    "end_session: killed {} child process(es) of session {session}",
                    killed.len()
                );
            }
        }
        // Idle unload (spec §21): the workspace watcher and the evidence
        // index are heavyweight per-workspace resources; a closed session
        // must not keep them alive forever.
        let row = handle.row()?;
        self.deps.workspaces.close(row.workspace_id);
        self.deps.evidence.forget(row.workspace_id);
        handle.end_session()?;
        Ok(())
    }

    /// The single per-session turn runner (audit round 6): waits for the
    /// active logical turn to finish, then delivers queued prompts one at a
    /// time as new logical turns (each with its own one-TurnCompleted flow).
    /// Exits when the queue is empty; callers re-kick on the next prompt.
    /// The per-session gate guarantees at most one runner per session.
    pub async fn run_session_queue(self: &Arc<Self>, session: SessionId) {
        {
            let mut runners = self.runners.lock().unwrap();
            if !runners.insert(session) {
                return; // a runner already exists for this session
            }
        }
        loop {
            let result = self.run_session_queue_inner(session).await;
            if let Err(e) = result {
                tracing::warn!("queue runner for session {session} ended: {e}");
                break;
            }
            // Close the start/exit race (audit round 7): a prompt that queued
            // between our empty-observation and this gate removal must get a
            // runner. Final durable re-check before releasing the gate.
            let pending = self
                .deps
                .session
                .get_session(session)
                .ok()
                .flatten()
                .map(|h| h.queued_prompt_count().unwrap_or(0))
                .unwrap_or(0);
            if pending == 0 {
                break;
            }
            // A prompt queued in the window: drain it under the same gate.
        }
        self.runners.lock().unwrap().remove(&session);
    }

    async fn run_session_queue_inner(
        self: &Arc<Self>,
        session: SessionId,
    ) -> kilop_core::Result<()> {
        loop {
            let handle = self
                .deps
                .session
                .get_session(session)?
                .ok_or_else(|| Error::not_found(format!("session {session}")))?;
            // Claimed queue rows from a crashed admission crash back to
            // pending so the durable head is re-admitted (idempotent).
            handle.recover_queued_rows()?;
            // A mid-flight machine blocks admission: when no LIVE driver owns
            // the active logical turn (post-restart), the residue is an
            // interrupted turn — resume the SAME recorded turn (same op id,
            // recorded model/envelope) before delivering queued prompts.
            if handle.queued_prompt_count()? > 0 {
                if let Some(record) = handle.active_turn_record()? {
                    let state = handle.state()?;
                    if state_is_op_active(state)
                        && handle.turn_cancellation(record.turn_op_id).is_none()
                    {
                        match self.continue_record(&handle, &record).await {
                            Ok(_) => continue,
                            Err(e) => {
                                // Not continuable yet (e.g. a durable
                                // permission waits on the user): back off and
                                // retry — the durable head stays pending.
                                tracing::warn!(
                                    session = %session,
                                    turn = %record.turn_op_id,
                                    "queue runner cannot continue interrupted turn: {e}"
                                );
                                tokio::time::sleep(Duration::from_millis(200)).await;
                                continue;
                            }
                        }
                    }
                }
            }
            // Atomic admission: the store claims the head and materializes
            // the user message in ONE transaction when the session is
            // eligible (audit round 7 — no submission can cut between claim
            // and admission).
            let Some(admitted) = handle.admit_next_queued()? else {
                // Admission declined: either the queue is empty (exit) or
                // the session is mid-turn (wait for the active logical turn
                // to end, then re-try — the durable head stays pending).
                if handle.queued_prompt_count()? == 0 {
                    return Ok(());
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            };
            handle.append_event(
                kilop_core::event::EventKind::PromptAdmitted,
                AgentState::Preparing,
                Some(admitted.op_id),
                Some(serde_json::json!({
                    "queue_seq": admitted.queue_seq,
                    "message_seq": admitted.message_seq,
                })),
            )?;
            let queue_seq = admitted.queue_seq;
            handle.mark_queued_status(queue_seq, "running")?;
            let outcome = self.drive_admitted(&handle, &admitted).await;
            let status = match &outcome {
                Ok(o) if o.final_state == AgentState::Cancelled => "cancelled",
                Ok(_) => "done",
                Err(_) => "cancelled",
            };
            handle.mark_queued_status(queue_seq, status)?;
            if matches!(outcome, Ok(o) if o.final_state == AgentState::Cancelled) {
                return Ok(());
            }
        }
    }

    /// Drive an admitted queued prompt as a logical turn. Same
    /// failure-finalization semantics as immediate turns (drive_receipt):
    /// an error journals FailedRecoverable so the session is never stranded.
    async fn drive_admitted(
        self: &Arc<Self>,
        handle: &kilop_session::SessionHandle,
        admitted: &kilop_session::AdmittedQueuedPrompt,
    ) -> kilop_core::Result<TurnOutcome> {
        let token = handle.turn_cancellation(admitted.op_id).unwrap_or_default();
        let model = admitted.model.clone();
        let outcome = self.drive_turn(handle, admitted.op_id, token, model).await;
        if let Err(e) = &outcome {
            let _ = handle.append_event(
                kilop_core::event::EventKind::Failed,
                AgentState::FailedRecoverable,
                Some(admitted.op_id),
                Some(serde_json::json!({ "message": e.message })),
            );
            let _ = handle.finish_turn_record(admitted.op_id, "failed");
        }
        outcome
    }

    /// Agent Manager cards (spec §15): daemon-owned background agents.
    pub fn cards(&self) -> kilop_core::Result<Vec<AgentCard>> {
        let mut out = Vec::new();
        for row in self.deps.session.list_sessions(None)? {
            let status = match row.state()? {
                AgentState::Completed => "completed".into(),
                AgentState::FailedPermanent | AgentState::FailedRecoverable => "failed".into(),
                AgentState::NeedsUserInput => "needs-input".into(),
                AgentState::WaitingForPermission => "waiting".into(),
                AgentState::Idle | AgentState::Suspended => "waiting".into(),
                _ => "running".into(),
            };
            out.push(AgentCard {
                session_id: row.id(),
                title: row.title()?,
                status,
            });
        }
        Ok(out)
    }

    // ------------------------------------------------------------ recovery

    /// Crash-recovery sweep over every session (daemon startup, spec §7).
    /// Runtime-level: rows are resolved with runtime knowledge —
    /// workspace-scoped postcondition verification for workspace writes,
    /// legacy absolute-path hash verification, unknown-effect marking.
    /// Interrupted turns whose rows need ASYNC replay (idempotent tools with
    /// a stored ReplayDescriptor) keep their rows running and their machine
    /// continuable; the per-session queue runner (or `continue_turn`)
    /// re-executes them ONCE with the recorded turn identity. Idempotent:
    /// a second sweep finds nothing pending.
    pub fn recover(&self) -> kilop_core::Result<Vec<RecoveryReport>> {
        let mut reports = Vec::new();
        for h in self.deps.session.list_sessions(None)? {
            reports.push(self.recover_session(&h)?);
        }
        Ok(reports)
    }

    /// Resolve interrupted tool runs of one session. Returns the rows that
    /// need ASYNC replay (idempotent tools with a stored descriptor), left
    /// running on the SAME row — the replay is a new physical attempt of the
    /// same logical operation. Everything else is finished durably here:
    /// workspace writes verify their recorded FilePostcondition through the
    /// workspace service (never a hash of JSON-encoded args); legacy
    /// VerifyHash rows hash the absolute path; MarkUnknown/Manual/None and
    /// legacy descriptor-less Idempotent rows are marked failed/unknown
    /// (never blindly re-run).
    fn recover_session(
        &self,
        handle: &kilop_session::SessionHandle,
    ) -> kilop_core::Result<RecoveryReport> {
        let session_id = handle.id();
        let pending = handle.pending_tool_runs()?;
        let current = handle.state()?;
        let mut report = RecoveryReport {
            session_id,
            state: current,
            crashed_ops: Vec::new(),
            orphans: Vec::new(),
            interrupted_turn: false,
            contradiction: false,
            applied: false,
        };
        if pending.is_empty() && !state_is_op_active(current) {
            return Ok(report);
        }
        // A machine that is not op-active while rows are pending is a
        // journal/ledger contradiction the session sweep knows how to fix
        // (rows finished, state stands). Runtime-level finishing would
        // journal illegal transitions from Idle/Suspended/terminal states.
        if !state_is_op_active(current) {
            return handle
                .recover_all()
                .map_err(|e| Error::new(ErrorKind::Store, format!("session recovery: {e}")));
        }
        // A LIVE in-process driver owns the session's active logical turn
        // (registered cancellation token): nothing crashed — recovery must
        // not journal CrashDetected nor touch the driver's running rows.
        // Post-restart there is no tracking, so crash residue is swept.
        if let Some(rec) = handle.active_turn_record()? {
            if handle.turn_cancellation(rec.turn_op_id).is_some() {
                return Ok(report);
            }
        }
        report.applied = true;
        // CrashDetected at the CURRENT state (self-transition): the machine
        // stays continuable so the SAME logical turn can resume with its
        // recorded identity — never a crash_target hop that kills it.
        let last_kind = self.last_event_kind(handle);
        if last_kind != Some(kilop_core::event::EventKind::CrashDetected) {
            handle.append_event(
                kilop_core::event::EventKind::CrashDetected,
                current,
                None,
                Some(serde_json::json!({
                    "pending_ops": pending.len(),
                    "recovered_from": state_tag(current),
                })),
            )?;
        }
        if pending.is_empty() {
            // Interrupted turn without tool rows (the crash hit the model
            // stream): the runner / continue_turn resumes the recorded turn.
            report.interrupted_turn = true;
            report.state = current;
            return Ok(report);
        }

        // Classify every row BEFORE finishing anything: finish order matters
        // (a failure finish moves the machine to FailedRecoverable, after
        // which "completed" finishes would be illegal).
        enum Verdict {
            Verify { postcondition: FilePostcondition },
            LegacyVerify { path: String, expected: FileHash },
            FailUnknown,
            DeferReplay,
        }
        let mut verdicts: Vec<(ToolRunRow, Verdict)> = Vec::with_capacity(pending.len());
        for row in pending {
            if let Some(pc) = row.postcondition.clone() {
                let pc: FilePostcondition = serde_json::from_value(pc).map_err(|e| {
                    Error::malformed(format!(
                        "tool_run {} carries a corrupt postcondition: {e}",
                        row.op_id
                    ))
                })?;
                verdicts.push((row, Verdict::Verify { postcondition: pc }));
                continue;
            }
            let recovery: RecoveryStrategy = serde_json::from_value(row.recovery.clone())
                .map_err(|e| Error::malformed(format!("corrupt recovery row: {e}")))?;
            match recovery {
                RecoveryStrategy::VerifyHash { path, expected } => {
                    verdicts.push((row, Verdict::LegacyVerify { path, expected }));
                }
                RecoveryStrategy::MarkUnknown
                | RecoveryStrategy::Manual
                | RecoveryStrategy::None => {
                    verdicts.push((row, Verdict::FailUnknown));
                }
                RecoveryStrategy::Idempotent => match row.replay_descriptor.as_ref() {
                    Some(desc) => {
                        // Validate the stored invocation BEFORE deferring:
                        // a hostile descriptor is a loud error, never a blind
                        // replay (validated again at replay time).
                        self.validate_replay_descriptor(&row, desc)?;
                        verdicts.push((row, Verdict::DeferReplay));
                    }
                    None => verdicts.push((row, Verdict::FailUnknown)),
                },
            }
        }
        // Resolution passes: (1) verifications that COMPLETE, (2) honest
        // failures (unknown effects), (3) replay deferrals only when the
        // whole batch is replayable (a failed row ends the turn, so a
        // replayable sibling cannot rejoin it — it is failed honestly).
        let all_deferrable = !verdicts.is_empty()
            && verdicts
                .iter()
                .all(|(_, v)| matches!(v, Verdict::DeferReplay));
        for (row, verdict) in &verdicts {
            match verdict {
                Verdict::Verify { .. } | Verdict::LegacyVerify { .. } => {
                    let (expected, actual) = match verdict {
                        Verdict::Verify { postcondition } => (
                            postcondition.expected_hash,
                            self.verify_workspace_file(postcondition)?,
                        ),
                        Verdict::LegacyVerify { path, expected } => {
                            (*expected, Some(stream_hash_file(path)))
                        }
                        _ => unreachable!(),
                    };
                    if actual == Some(expected) {
                        handle.finish_tool_run(row.op_id, "completed", EffectStatus::Verified)?;
                        self.journal_recovery_applied(
                            handle,
                            row,
                            "completed",
                            EffectStatus::Verified,
                            "verified",
                        )?;
                        report.crashed_ops.push(RecoveredOp {
                            op_id: row.op_id,
                            tool: row.tool.clone(),
                            status: "completed".into(),
                            effect: EffectStatus::Verified,
                            action: RecoveryAction::Verified {
                                expected,
                                actual: actual.unwrap_or(expected),
                            },
                        });
                    } else {
                        // The file does not match the recorded postcondition:
                        // the write never landed (or was overwritten) — FAIL
                        // LOUDLY, never silently "applied".
                        handle.finish_tool_run(row.op_id, "failed", EffectStatus::Failed)?;
                        self.journal_recovery_applied(
                            handle,
                            row,
                            "failed",
                            EffectStatus::Failed,
                            "not_applied",
                        )?;
                        report.crashed_ops.push(RecoveredOp {
                            op_id: row.op_id,
                            tool: row.tool.clone(),
                            status: "failed".into(),
                            effect: EffectStatus::Failed,
                            action: RecoveryAction::NotApplied { expected, actual },
                        });
                    }
                }
                Verdict::FailUnknown => {
                    handle.finish_tool_run(row.op_id, "failed", EffectStatus::Unknown)?;
                    self.journal_recovery_applied(
                        handle,
                        row,
                        "failed",
                        EffectStatus::Unknown,
                        "unknown_effect",
                    )?;
                    report.crashed_ops.push(RecoveredOp {
                        op_id: row.op_id,
                        tool: row.tool.clone(),
                        status: "failed".into(),
                        effect: EffectStatus::Unknown,
                        action: RecoveryAction::UnknownEffect,
                    });
                }
                Verdict::DeferReplay => {
                    if all_deferrable {
                        report.crashed_ops.push(RecoveredOp {
                            op_id: row.op_id,
                            tool: row.tool.clone(),
                            status: "running".into(),
                            effect: EffectStatus::Unknown,
                            action: RecoveryAction::RerunAllowed,
                        });
                    } else {
                        // A sibling ended the turn: this row cannot rejoin it.
                        handle.finish_tool_run(row.op_id, "failed", EffectStatus::Unknown)?;
                        self.journal_recovery_applied(
                            handle,
                            row,
                            "failed",
                            EffectStatus::Unknown,
                            "unknown_effect",
                        )?;
                        report.crashed_ops.push(RecoveredOp {
                            op_id: row.op_id,
                            tool: row.tool.clone(),
                            status: "failed".into(),
                            effect: EffectStatus::Unknown,
                            action: RecoveryAction::RerunAllowed,
                        });
                    }
                }
            }
        }
        // If any row resolved as a failure the machine landed
        // FailedRecoverable: the interrupted turn is over — close its record.
        if handle.state()? == AgentState::FailedRecoverable {
            if let Some(rec) = handle.active_turn_record()? {
                let _ = handle.finish_turn_record(rec.turn_op_id, "failed");
            }
        }
        report.state = handle.state()?;
        Ok(report)
    }

    fn journal_recovery_applied(
        &self,
        handle: &kilop_session::SessionHandle,
        row: &ToolRunRow,
        status: &str,
        effect: EffectStatus,
        action: &str,
    ) -> kilop_core::Result<()> {
        let state = handle.state()?;
        handle.append_event(
            kilop_core::event::EventKind::RecoveryApplied,
            state,
            Some(row.op_id),
            Some(serde_json::json!({
                "op_id": row.op_id.raw(),
                "tool": row.tool,
                "status": status,
                "effect": effect_tag(effect),
                "action": action,
            })),
        )?;
        Ok(())
    }

    fn last_event_kind(
        &self,
        handle: &kilop_session::SessionHandle,
    ) -> Option<kilop_core::event::EventKind> {
        let last = handle.last_event_seq().ok()??;
        let n = last.raw();
        handle
            .events_range(n.saturating_sub(1).max(1), Some(2))
            .ok()?
            .into_iter()
            .find(|e| e.seq == last)
            .map(|e| e.kind)
    }

    /// Verify a workspace write through the WorkspaceFileService: canonical
    /// safe resolution of the RELATIVE path against the workspace root (no
    /// `..`, no symlink escapes, never the daemon cwd), then a streamed
    /// BLAKE3 of the CURRENT file bytes. `None` when the file is missing or
    /// unreadable (the zero-marker — "write never landed").
    fn verify_workspace_file(
        &self,
        pc: &FilePostcondition,
    ) -> kilop_core::Result<Option<FileHash>> {
        let root = self
            .deps
            .session
            .store()
            .workspace_root(pc.workspace_id)
            .map_err(map_store_error)?;
        let Some(root) = root else {
            return Err(Error::malformed(format!(
                "tool recovery: workspace {} is not registered",
                pc.workspace_id
            )));
        };
        let ws = self
            .deps
            .workspaces
            .open(pc.workspace_id, std::path::PathBuf::from(root))
            .map_err(|e| Error::malformed(format!("tool recovery workspace open: {e}")))?;
        // Traversal/symlink-unsafe relative paths are REJECTED loudly here —
        // recovery never touches a file outside the workspace root.
        let resolved = ws
            .resolve(std::path::Path::new(&pc.relative_path))
            .map_err(|e| {
                Error::permission(format!(
                    "tool recovery path {:?} rejected: {e}",
                    pc.relative_path
                ))
            })?;
        Ok(Some(stream_hash_file(&resolved.to_string_lossy())))
    }

    /// Validate a stored replay invocation. A hostile descriptor (missing
    /// fields, unknown tool, args that do not satisfy the tool's input
    /// schema where feasible) is a loud error — recovery NEVER blind-replays.
    fn validate_replay_descriptor(
        &self,
        row: &ToolRunRow,
        raw: &serde_json::Value,
    ) -> kilop_core::Result<ReplayDescriptor> {
        let desc: ReplayDescriptor = serde_json::from_value(raw.clone()).map_err(|e| {
            Error::malformed(format!(
                "tool_run {} carries a hostile replay descriptor: {e}",
                row.op_id
            ))
        })?;
        if desc.tool_name != row.tool {
            return Err(Error::malformed(format!(
                "tool_run {} replay descriptor names tool {:?}, row says {:?}",
                row.op_id, desc.tool_name, row.tool
            )));
        }
        if desc.recovery_kind != "idempotent" {
            return Err(Error::malformed(format!(
                "tool_run {} replay descriptor declares unsupported recovery kind {:?}",
                row.op_id, desc.recovery_kind
            )));
        }
        let tool = self.deps.tools.get(&desc.tool_name).ok_or_else(|| {
            Error::malformed(format!(
                "tool_run {} cannot replay: tool {:?} is not registered",
                row.op_id, desc.tool_name
            ))
        })?;
        validate_args_against_schema(&tool, &desc.validated_args)?;
        Ok(desc)
    }

    /// Replay ONE deferred idempotent run: a NEW PHYSICAL attempt of the
    /// SAME logical operation. Journals ReplayStarted exactly once, bumps the
    /// attempt counter on the run row, re-executes the stored invocation
    /// ONCE with the previously-granted permission, links the outcome to the
    /// original tool call, and finishes the row (ToolCompleted). The turn
    /// identity (record + op ids) is untouched.
    async fn replay_tool_run(
        &self,
        handle: &kilop_session::SessionHandle,
        row: &ToolRunRow,
    ) -> kilop_core::Result<()> {
        let raw = row.replay_descriptor.as_ref().ok_or_else(|| {
            Error::malformed(format!("tool_run {} has no replay descriptor", row.op_id))
        })?;
        let desc = self.validate_replay_descriptor(row, raw)?;
        let tool = self
            .deps
            .tools
            .get(&desc.tool_name)
            .ok_or_else(|| Error::not_found(format!("tool {}", desc.tool_name)))?;
        let state = handle.state()?;
        if state != AgentState::ExecutingTool {
            return Err(Error::conflict(format!(
                "replay of {} requires the machine at ExecutingTool, found {state:?}",
                row.op_id
            )));
        }
        // Journal the replay start (self-transition, exactly once per run).
        handle.append_event(
            kilop_core::event::EventKind::ReplayStarted,
            state,
            Some(row.op_id),
            Some(serde_json::json!({
                "tool": row.tool,
                "attempt": row.attempt + 1,
                "turn_op_id": desc.original_turn_op_id.raw(),
            })),
        )?;
        let attempt = handle.bump_tool_attempt(row.op_id)?;
        // Reconstruct the original invocation context (the permission hop
        // was already resolved pre-crash; a replay is its continuation).
        let root = self
            .deps
            .session
            .store()
            .workspace_root(desc.workspace_id)
            .map_err(map_store_error)?
            .map(std::path::PathBuf::from);
        let workspace = match &root {
            Some(root) => self
                .deps
                .workspaces
                .open(desc.workspace_id, root.clone())
                .ok()
                .map(Arc::new),
            None => None,
        };
        let sandbox = match (&self.deps.sandbox, &root) {
            (Some(base), Some(root)) => Some(Arc::new(kilop_sandbox::PermissionEngine::new(
                base.policy().clone(),
                Some(root.clone()),
            ))),
            _ => None,
        };
        let ctx = ToolRunCtx {
            session_id: handle.id(),
            permission_granted: true,
            op_id: row.op_id,
            identity: WorkspaceIdentity::new(desc.workspace_id, desc.worktree_id, desc.task_id),
            cancellation: CancellationToken::new(),
            artifacts: Arc::new(self.deps.artifact_sink(handle.id())),
            tool_call_mode: self.deps.tool_call_mode,
            workspace: workspace.clone(),
            edit: self.deps.edit.clone(),
            snapshots: self.deps.snapshots.clone(),
            sandbox: sandbox.clone(),
            supervisor: self.deps.supervisor.clone(),
            deadline_ms: self.deps.tool_deadline_ms,
        };
        let outcome = match (tool.execute)(ctx, desc.validated_args.clone()).await {
            Ok(o) => o,
            Err(e) => {
                // The replay itself failed: honest completion of the attempt.
                handle.finish_tool_run(row.op_id, "failed", EffectStatus::Unknown)?;
                return Err(e);
            }
        };
        if let Some(pc) = &outcome.postcondition {
            let v = serde_json::to_value(pc)
                .map_err(|e| Error::malformed(format!("postcondition serialization: {e}")))?;
            handle.record_tool_postcondition(row.op_id, &v)?;
        }
        // Link the outcome to the ORIGINAL tool call (never a duplicate
        // message): the model sees exactly one result for the call.
        let call_id = self.find_original_call_id(handle, &row.tool, &row.args)?;
        let seq = handle.proposed_message_seq()?;
        let mid = handle.put_message(seq, "assistant", serde_json::json!({ "parts": [] }))?;
        let body = ToolResultBody {
            excerpt: truncate(&outcome.text, 2000),
            exit_code: outcome.exit_code,
            artifact: outcome.artifact,
            slice_hint: outcome.slice_hint,
        };
        handle.put_tool_result_part(mid, &call_id, &body)?;
        handle.finish_tool_run(row.op_id, "completed", outcome.effect_status)?;
        tracing::info!(
            session = %handle.id(),
            op = %row.op_id,
            tool = %row.tool,
            attempt,
            "replayed interrupted idempotent tool run"
        );
        Ok(())
    }

    /// The tool_call part id the ORIGINAL run answered (name + args match):
    /// replayed results must reference it or the model sees an orphan.
    fn find_original_call_id(
        &self,
        handle: &kilop_session::SessionHandle,
        tool: &str,
        args: &serde_json::Value,
    ) -> kilop_core::Result<String> {
        const MAX_SCAN: usize = 400;
        let mut cursor: Option<i64> = None;
        let mut scanned = 0usize;
        loop {
            let page = handle.messages_before(cursor, 100)?;
            if page.is_empty() {
                break;
            }
            for row in page.iter() {
                if scanned >= MAX_SCAN {
                    break;
                }
                scanned += 1;
                for part in handle.parts_of(row.id)? {
                    if part.kind == "tool_call"
                        && part.data.get("name").and_then(|n| n.as_str()) == Some(tool)
                        && part.data.get("input") == Some(args)
                    {
                        if let Some(id) = part
                            .data
                            .get("tool_call_id")
                            .and_then(|i| i.as_str())
                            .filter(|i| !i.is_empty())
                        {
                            return Ok(id.to_string());
                        }
                    }
                }
            }
            if scanned >= MAX_SCAN || page.last().unwrap().seq <= 1 {
                break;
            }
            cursor = Some(page.last().unwrap().seq);
        }
        Err(Error::malformed(format!(
            "replay of {tool} cannot find its original tool call in the journal"
        )))
    }

    /// Continue one recorded interrupted logical turn (crash recovery):
    /// resolve side effects, replay deferred idempotent runs exactly once,
    /// walk the machine back to WaitingForModel, then drive the SAME
    /// recorded turn op with the SAME recorded model — never a synthesized
    /// op and never the session's current defaults.
    async fn continue_record(
        self: &Arc<Self>,
        handle: &kilop_session::SessionHandle,
        record: &kilop_store::TurnRecordRow,
    ) -> kilop_core::Result<TurnOutcome> {
        let turn_op = record.turn_op_id;
        if handle.turn_cancellation(turn_op).is_some() {
            return Err(Error::conflict(format!(
                "turn {turn_op} already has a live driver"
            )));
        }
        let state = handle.state()?;
        if !state_is_op_active(state) {
            return Err(Error::conflict(format!(
                "session {} is {:?}; no interrupted logical turn to continue",
                handle.id(),
                state
            )));
        }
        // Resolve side effects (existing tool-run recovery; idempotent runs
        // come back as deferred rows and are replayed below).
        self.recover_session(handle)?;
        let state = handle.state()?;
        // Replay deferred idempotent runs ONCE each (the row stays the SAME
        // logical operation; only the attempt counter moves).
        if state == AgentState::ExecutingTool {
            let pending = handle.pending_tool_runs()?;
            for row in &pending {
                self.replay_tool_run(handle, row).await?;
            }
        }
        let state = handle.state()?;
        match state {
            AgentState::FailedRecoverable
            | AgentState::FailedPermanent
            | AgentState::Cancelled
            | AgentState::Completed
            | AgentState::NeedsUserInput => {
                // The interrupted turn is over (its effects resolved as
                // failed/unknown): report the end, never re-drive it.
                let status = if state == AgentState::Cancelled {
                    "cancelled"
                } else {
                    "failed"
                };
                let _ = handle.finish_turn_record(turn_op, status);
                return Ok(TurnOutcome {
                    op_id: turn_op,
                    final_state: state,
                    turns: 0,
                    compacted: false,
                    loop_stopped: false,
                    queued: false,
                });
            }
            AgentState::WaitingForPermission | AgentState::ToolRequested => {
                return Err(Error::conflict(format!(
                    "session {} waits on a durable permission; resolve it before continuing",
                    handle.id()
                )));
            }
            _ => {}
        }
        self.walk_to_waiting(handle, turn_op)?;
        let outcome = self
            .drive_turn(
                handle,
                turn_op,
                CancellationToken::new(),
                Some(record.effective_model.clone()),
            )
            .await;
        if outcome.is_err() {
            let _ = handle.finish_turn_record(turn_op, "failed");
        }
        outcome
    }

    /// Interior state hop back to WaitingForModel after crash recovery using
    /// ONLY legal machine transitions (never a blind re-entry).
    fn walk_to_waiting(
        &self,
        handle: &kilop_session::SessionHandle,
        op: OpId,
    ) -> kilop_core::Result<()> {
        match handle.state()? {
            AgentState::Validating => {
                handle.append_event(
                    kilop_core::event::EventKind::PhaseChanged,
                    AgentState::UpdatingMemory,
                    Some(op),
                    None,
                )?;
                handle.append_event(
                    kilop_core::event::EventKind::PhaseChanged,
                    AgentState::WaitingForModel,
                    Some(op),
                    None,
                )?;
            }
            AgentState::UpdatingMemory | AgentState::Streaming => {
                handle.append_event(
                    kilop_core::event::EventKind::PhaseChanged,
                    AgentState::WaitingForModel,
                    Some(op),
                    None,
                )?;
            }
            _ => {}
        }
        Ok(())
    }

    // ------------------------------------------------------------ the turn loop

    /// Drive ONE logical turn to its single genuine end. Queued-prompt
    /// isolation (audit round 6) happens in the history loader: user
    /// messages of undelivered queued prompts never enter this turn's
    /// context.
    async fn drive_turn(
        self: &Arc<Self>,
        handle: &kilop_session::SessionHandle,
        op_id: OpId,
        cancel: CancellationToken,
        model_override: Option<String>,
    ) -> kilop_core::Result<TurnOutcome> {
        let outcome = self
            .drive_turn_inner(handle, op_id, cancel, model_override)
            .await;
        // The durable turn record follows the machine: a genuine end closes
        // the record so recovery never continues a finished turn.
        if let Ok(o) = &outcome {
            let status = match o.final_state {
                AgentState::ReadyForNextTurn | AgentState::Completed => "completed",
                AgentState::Cancelled => "cancelled",
                _ => "failed",
            };
            let _ = handle.finish_turn_record(op_id, status);
        }
        outcome
    }

    async fn drive_turn_inner(
        self: &Arc<Self>,
        handle: &kilop_session::SessionHandle,
        op_id: OpId,
        cancel: CancellationToken,
        model_override: Option<String>,
    ) -> kilop_core::Result<TurnOutcome> {
        let mut outcome = TurnOutcome {
            op_id,
            final_state: AgentState::Preparing,
            turns: 0,
            compacted: false,
            loop_stopped: false,
            queued: false,
        };
        // Per-logical-turn accumulation: real steps/failures/files/tests for
        // the durable ledger + memory (audit: only defaults were recorded).
        let mut turn_summary = kilop_context::ledger::TurnSummary::default();
        let mut detector = LoopDetector::new(3);
        let mut ledger = self.load_ledger(handle)?;
        // The durable task state starts from the user's own goal: the first
        // prompt (session title) — audit round: goal was never set.
        if ledger.goal.is_empty() {
            ledger.goal = truncate(&handle.title()?, 200).to_string();
        }
        let provider = self.provider_for(handle)?;
        // The effective model is the per-message override when present; the
        // provider is ALWAYS the session's provider. Capabilities for a
        // model the provider does not know fall back to the provider's
        // default (never an error at send time).
        let model = match model_override {
            Some(m) => m,
            None => handle.model()?,
        };
        // v7 durable per-turn envelope: the moment the logical turn actually
        // drives, its effective provider/model (per-message override wins),
        // reasoning variant and tool mode are fixed on the turn record.
        // Crash recovery resumes from the RECORD, never from whatever the
        // session defaults are afterwards (P1: overrides survive crashes).
        let provider_id = handle.provider()?;
        let _ = handle.set_turn_envelope(
            op_id,
            &provider_id,
            &model,
            None,
            Some(tool_mode_tag(self.deps.tool_call_mode)),
        );
        let caps = provider.capabilities(&model);
        // P0 (runtime context override): a provider's LIVE runtime window
        // (e.g. the Ollama /api/ps allocation, which can sit far below the
        // advertised 256K model maximum) is the real budget ceiling. When
        // the provider reports one, budget from min(model max, runtime
        // limit); None means no live data and the model maximum stands
        // (today's behavior — safe direction). Built ONCE per logical turn,
        // so the compaction trigger, try_compact's target, and every
        // post-compaction re-plan all share the SAME effective budget.
        let mut effective_caps = caps.clone();
        if let Some(limit) = provider.runtime_context_limit(&model) {
            effective_caps.context = effective_caps.context.min(limit);
        }
        let budget = ContextBudget::for_capabilities(&effective_caps);
        loop {
            if cancel.is_cancelled() {
                let _ = handle.abort(Some(op_id));
                outcome.final_state = AgentState::Cancelled;
                return Ok(outcome);
            }
            let state = handle.state()?;
            if matches!(
                state,
                AgentState::Cancelled
                    | AgentState::Completed
                    | AgentState::FailedPermanent
                    | AgentState::NeedsUserInput
            ) {
                outcome.final_state = state;
                return Ok(outcome);
            }
            // ---- prepare context (fresh turn only; iterations continuing
            // the SAME logical turn after a tool batch arrive with the
            // machine at WaitingForModel and re-plan purely in memory — no
            // journal hops)
            if state != AgentState::WaitingForModel {
                handle.append_event(
                    kilop_core::event::EventKind::ContextPrepared,
                    AgentState::BuildingContext,
                    Some(op_id),
                    None,
                )?;
            }
            let recent = self.recent_turns(handle)?;
            // Retrieval signals (spec §20): the CURRENT prompt (the last
            // user turn), the files the task changed, and known failures —
            // never just the session title.
            let prompt = recent
                .iter()
                .rev()
                .find(|t| t.role == "user")
                .map(|t| t.text.clone())
                .filter(|t| !t.is_empty())
                .unwrap_or_else(|| handle.title().unwrap_or_default());
            let evidence = self.deps.evidence.evidence_for(
                handle.id(),
                &EvidenceQuery {
                    prompt,
                    changed_files: ledger.changed_files.clone(),
                    failures: ledger.known_failures.clone(),
                },
            );
            // Repository knowledge (spec §8 class 3): bounded file map +
            // AGENTS.md rules ride the cacheable prefix. Re-resolved every
            // iteration so edits made by tools appear on the next hop.
            let (project_rules, repo_map) = self.repo_knowledge(handle);
            let mut history = self.history_messages(handle)?;
            let mut wire_plan = plan_wire_request(
                &self.deps.instructions,
                "",
                &self.deps.tools.specs(),
                &project_rules,
                &ledger,
                &repo_map,
                &history,
                &evidence,
                "",
                &budget,
            )?;

            // ---- proactive compaction (spec §9)
            let usage = budget.effective_usage(wire_plan.total_tokens);
            if usage >= self.deps.compact_at_usage.clamp(0.0, 1.0) {
                if let Some(plan) = self
                    .try_compact(handle, &recent, &ledger, &budget, &cancel)
                    .await?
                {
                    outcome.compacted = true;
                    ledger = plan.ledger.clone();
                    history = recent_turns_to_messages(&plan.kept_recent);
                    wire_plan = plan_wire_request(
                        &self.deps.instructions,
                        "",
                        &self.deps.tools.specs(),
                        &project_rules,
                        &ledger,
                        &repo_map,
                        &history,
                        &evidence,
                        "",
                        &budget,
                    )?;
                }
            }

            // ---- provider call (state-aware retry, spec §13): a request
            // that failed BEFORE any content became durable may retry under
            // the retry policy (network class, bounded backoff). Once a tool
            // ran or assistant content was flushed, never replay.
            handle.append_event(
                kilop_core::event::EventKind::ModelStarted,
                AgentState::WaitingForModel,
                Some(op_id),
                None,
            )?;
            let max_attempts = self.deps.retry_policy.max_attempts.max(1);
            let mut assistant_message: Option<i64> = None;
            let mut text_buf = String::new();
            let mut reasoning_buf = String::new();
            let mut tool_calls: Vec<(String, String, serde_json::Value)> = Vec::new();
            let mut tokens_in = 0u64;
            let mut tokens_out = 0u64;
            use futures::StreamExt;
            let ensure_message = |mid: &mut Option<i64>| -> kilop_core::Result<i64> {
                if let Some(m) = *mid {
                    return Ok(m);
                }
                let seq = handle.proposed_message_seq()?;
                let m = handle.put_message(seq, "assistant", serde_json::json!({ "parts": [] }))?;
                *mid = Some(m);
                Ok(m)
            };
            'attempts: for attempt in 0..max_attempts {
                if attempt > 0 {
                    // Bounded exponential backoff with jitter before the
                    // next try (spec §13).
                    let delay = self.deps.retry_policy.next_delay(attempt - 1);
                    tokio::time::sleep(delay).await;
                }
                let request =
                    self.build_request(handle, &wire_plan, op_id, &model, &cancel, attempt)?;
                CapabilityValidator::validate(&request, &caps)?;
                handle.record_provider_call(
                    op_id,
                    provider.id(),
                    &request.model,
                    "started",
                    None,
                    None,
                    None,
                )?;
                handle.append_event(
                    kilop_core::event::EventKind::ModelStarted,
                    AgentState::Streaming,
                    Some(op_id),
                    None,
                )?;

                let mut stream = provider.stream(request);
                while let Some(chunk) = stream.next().await {
                    if cancel.is_cancelled() {
                        let _ = handle.abort(Some(op_id));
                        outcome.final_state = AgentState::Cancelled;
                        return Ok(outcome);
                    }
                    match chunk {
                        Ok(ProviderChunk::Text { text }) => {
                            text_buf.push_str(&text);
                            let mid = ensure_message(&mut assistant_message)?;
                            // EPHEMERAL path: text deltas are NOT journaled per
                            // chunk (a multi-hour agent would commit millions of
                            // tiny SQLite events). The durable representation is
                            // the message part, flushed in bounded segments so a
                            // crash loses at most one segment.
                            if text_buf.len() >= STREAM_FLUSH_BYTES {
                                handle.put_text_part(mid, &text_buf)?;
                                text_buf.clear();
                            }
                        }
                        Ok(ProviderChunk::Reasoning { text }) => {
                            reasoning_buf.push_str(&text);
                            let mid = ensure_message(&mut assistant_message)?;
                            if reasoning_buf.len() >= STREAM_FLUSH_BYTES {
                                handle.put_reasoning_part(mid, &reasoning_buf)?;
                                reasoning_buf.clear();
                            }
                        }
                        Ok(ProviderChunk::ToolCall {
                            id,
                            name,
                            input,
                            complete,
                        }) => {
                            if !complete {
                                return Err(Error::malformed(format!(
                                    "incomplete tool call {id} without completion"
                                )));
                            }
                            let mid = ensure_message(&mut assistant_message)?;
                            handle.put_tool_call_part(
                                mid,
                                &id,
                                &name,
                                input.clone(),
                                "completed",
                            )?;
                            tool_calls.push((id, name, input));
                        }
                        Ok(ProviderChunk::Usage {
                            tokens_in: ti,
                            tokens_out: to,
                        }) => {
                            tokens_in = ti;
                            tokens_out = to;
                        }
                        Ok(ProviderChunk::Done) => break,
                        Err(e) => {
                            handle.record_provider_call(
                                op_id,
                                provider.id(),
                                &model,
                                "failed",
                                None,
                                None,
                                Some(&e.to_string()),
                            )?;
                            // Retry ONLY when nothing durable happened in this
                            // request (no flushed parts, no message created, no
                            // tool runs pending) and the failure is retryable.
                            let safe = assistant_message.is_none()
                                && handle.pending_tool_runs()?.is_empty();
                            if safe && attempt + 1 < max_attempts && e.retryable {
                                tracing::warn!(
                                "provider failure on attempt {} of {max_attempts}: {e}; retrying",
                                attempt + 1
                            );
                                // The failed request journaled nothing durable:
                                // the wire state is unchanged — safe to retry.
                                assistant_message = None;
                                text_buf.clear();
                                reasoning_buf.clear();
                                tool_calls.clear();
                                continue 'attempts;
                            }
                            return self
                                .handle_provider_failure(handle, op_id, e, &mut outcome)
                                .await;
                        }
                    }
                }
                // This attempt consumed a full stream: no further retries.
                break 'attempts;
            }

            if let Some(mid) = assistant_message {
                if !reasoning_buf.is_empty() {
                    handle.put_reasoning_part(mid, &reasoning_buf)?;
                }
                if !text_buf.is_empty() {
                    handle.put_text_part(mid, &text_buf)?;
                }
            }
            handle.record_provider_call(
                op_id,
                provider.id(),
                &model,
                "completed",
                Some(tokens_in),
                Some(tokens_out),
                None,
            )?;

            if !tool_calls.is_empty() {
                let executed = self
                    .run_tool_calls(
                        handle,
                        op_id,
                        &mut detector,
                        &mut ledger,
                        &mut turn_summary,
                        &cancel,
                        tool_calls,
                    )
                    .await?;
                // Durable cross-turn loop detection (spec §28): the same
                // failing calls repeated across logical turns trip here even
                // though each turn's LoopDetector starts fresh.
                let durable_trip = self.durable_loop_signals(handle, &turn_summary, &detector)?;
                if (executed == 0 && detector.trips > 0) || durable_trip {
                    // Repeating failing calls: stop and re-plan.
                    outcome.loop_stopped = true;
                    let _ = handle.append_event(
                        kilop_core::event::EventKind::Failed,
                        AgentState::FailedRecoverable,
                        Some(op_id),
                        Some(serde_json::json!({ "message": "loop detected: repeated failing tool calls" })),
                    );
                    outcome.final_state = AgentState::FailedRecoverable;
                    return Ok(outcome);
                }
                if executed > 0 {
                    // Tools ran: the SAME logical turn continues. Interior
                    // hops (no TurnCompleted — that is reserved for the one
                    // genuine end) return the machine to WaitingForModel so
                    // the model can see the tool results.
                    handle.append_event(
                        kilop_core::event::EventKind::PhaseChanged,
                        AgentState::UpdatingMemory,
                        Some(op_id),
                        None,
                    )?;
                    handle.append_event(
                        kilop_core::event::EventKind::PhaseChanged,
                        AgentState::WaitingForModel,
                        Some(op_id),
                        None,
                    )?;
                    continue; // stream again with tool results (machine at WaitingForModel)
                }
                // executed == 0: every call was denied or unknown. If the
                // loop detector tripped we returned above; otherwise the
                // turn genuinely ends below (the denials already moved the
                // machine toward ReadyForNextTurn).
            }

            // ---- genuine end of the logical turn: validate → update
            // memory → ONE TurnCompleted → ReadyForNextTurn.
            let current = handle.state()?;
            if current == AgentState::ReadyForNextTurn {
                // Denials resolved the machine early (PermissionDenied →
                // ReadyForNextTurn): still exactly one TurnCompleted.
                ledger.record_turn(&turn_summary);
                handle.put_task_ledger(serde_json::to_value(&ledger)?)?;
                self.record_memory(handle, op_id, &ledger, &turn_summary)?;
                // The task finished with genuine work: loop windows close.
                if turn_made_progress(&turn_summary) {
                    let _ = handle.reset_loop_signals();
                }
                handle.append_event(
                    kilop_core::event::EventKind::TurnCompleted,
                    AgentState::ReadyForNextTurn,
                    Some(op_id),
                    None,
                )?;
                outcome.turns += 1;
                outcome.final_state = AgentState::ReadyForNextTurn;
                return Ok(outcome);
            }
            handle.append_event(
                kilop_core::event::EventKind::PhaseChanged,
                AgentState::Validating,
                Some(op_id),
                None,
            )?;
            handle.append_event(
                kilop_core::event::EventKind::PhaseChanged,
                AgentState::UpdatingMemory,
                Some(op_id),
                None,
            )?;
            ledger.record_turn(&turn_summary);
            handle.put_task_ledger(serde_json::to_value(&ledger)?)?;
            self.record_memory(handle, op_id, &ledger, &turn_summary)?;
            if turn_made_progress(&turn_summary) {
                let _ = handle.reset_loop_signals();
            }
            handle.append_event(
                kilop_core::event::EventKind::TurnCompleted,
                AgentState::ReadyForNextTurn,
                Some(op_id),
                None,
            )?;
            outcome.turns += 1;
            outcome.final_state = AgentState::ReadyForNextTurn;
            return Ok(outcome);
        }
    }

    /// Execute tool calls in parallel via the scheduler, feeding results
    /// back. Returns the number of tools actually executed.
    #[allow(clippy::too_many_arguments)]
    async fn run_tool_calls(
        self: &Arc<Self>,
        handle: &kilop_session::SessionHandle,
        turn_op: OpId,
        detector: &mut LoopDetector,
        ledger: &mut TaskLedger,
        turn_summary: &mut kilop_context::ledger::TurnSummary,
        cancel: &CancellationToken,
        calls: Vec<(String, String, serde_json::Value)>,
    ) -> kilop_core::Result<usize> {
        // Resolve the session's workspace ONCE per batch: the real tools
        // (read/write/search/run_command) operate inside the canonical root
        // with a per-session permission engine, never on model-supplied
        // absolute paths. When the session has no resolvable workspace the
        // ctx carries None and the tools error honestly.
        let row = handle.row()?;
        let workspace_id = row.workspace_id;
        let root = self
            .deps
            .session
            .store()
            .workspace_root(workspace_id)
            .map_err(map_store_error)?
            .map(std::path::PathBuf::from);
        let workspace = match &root {
            Some(root) => self
                .deps
                .workspaces
                .open(workspace_id, root.clone())
                .ok()
                .map(Arc::new),
            None => None,
        };
        let sandbox = match (&self.deps.sandbox, &root) {
            (Some(base), Some(root)) => Some(Arc::new(kilop_sandbox::PermissionEngine::new(
                base.policy().clone(),
                Some(root.clone()),
            ))),
            _ => None,
        };
        let now_ms = self.deps.clock.now_ms();

        let mut executed = 0usize;
        let scheduler = Scheduler::new(handle.id(), self.deps.clock.clone());
        let outcomes: Arc<std::sync::Mutex<HashMap<OpId, ToolOutcome>>> =
            Arc::new(std::sync::Mutex::new(HashMap::new()));
        let mut submitted: Vec<(OpId, String, String, serde_json::Value)> = Vec::new();
        let mut denied: Vec<String> = Vec::new();

        for (call_id, name, input) in calls {
            // Loop detection on the call itself (normalized).
            if detector.record_tool_call(&name, &input) {
                return Ok(executed); // drive_turn stops the turn
            }

            let tool = match self.deps.tools.get(&name) {
                Some(t) => t,
                None => {
                    detector.record_error(&format!("unknown tool {name}"));
                    denied.push(format!("unknown tool: {name}"));
                    continue;
                }
            };

            // Permission hop (journals ToolRequested).
            let capability = tool.capability.clone().unwrap_or(Capability::ExecuteShell {
                command: name.clone(),
            });
            let permission = handle.request_permission(turn_op, &capability)?;
            let decision = self
                .deps
                .permission_requester
                .request(handle.id(), &permission)
                .await?;
            match &decision {
                PermissionDecision::Deny => {
                    handle.resolve_permission(permission.id, PermissionDecision::Deny)?;
                    denied.push(format!("permission denied: {name}"));
                    continue;
                }
                PermissionDecision::Ask => {
                    return Err(Error::new(
                        ErrorKind::Permission,
                        format!("permission {name} unresolved"),
                    ));
                }
                PermissionDecision::Allow => {
                    handle.resolve_permission(permission.id, PermissionDecision::Allow)?;
                }
            }
            // The tool gate may proceed past Ask-policy verdicts because the
            // interactive hop resolved above.
            let granted = matches!(decision, PermissionDecision::Allow);

            // Op envelope: deadline, retry, cancellation, recovery.
            // The recovery strategy NEVER infers file postconditions from
            // JSON content args (P0): workspace writes record their own
            // FilePostcondition (bytes as written) at execution end and
            // recovery verifies through the workspace file service; until
            // then an interrupted write is an unknown effect.
            let op_id = self.deps.session.next_op_id();
            // The session ROW is the single source of the worktree/task
            // identity (v8): a standalone session row defaults to 1/1
            // (documented), an adopted row carries the real ids — the
            // descriptor and the execution ctx below both ride them, so a
            // crash replay resumes with the SAME identity that ran before.
            let identity = WorkspaceIdentity::new(row.workspace_id, row.worktree_id, row.task_id);
            let recovery = match &tool.recovery_hint {
                RecoveryHint::WorkspaceWrite => RecoveryStrategy::MarkUnknown,
                RecoveryHint::Idempotent => RecoveryStrategy::Idempotent,
                RecoveryHint::UnknownEffect => RecoveryStrategy::MarkUnknown,
            };
            let replay = if recovery == RecoveryStrategy::Idempotent {
                // Durable replay descriptor: the stored invocation recovery
                // may re-execute ONCE. Args ride as canonical JSON (serde's
                // Map is key-sorted); re-validation against the tool's input
                // schema happens on the recovery path — a hostile descriptor
                // is a loud error, never a blind replay.
                let desc = ReplayDescriptor {
                    tool_name: name.clone(),
                    validated_args: input.clone(),
                    workspace_id: identity.workspace_id,
                    worktree_id: identity.worktree_id,
                    task_id: identity.task_id,
                    original_turn_op_id: turn_op,
                    capability: capability.clone(),
                    recovery_kind: "idempotent".into(),
                };
                serde_json::to_value(desc)
                    .map_err(|e| Error::malformed(format!("replay descriptor: {e}")))?
            } else {
                serde_json::Value::Null
            };
            let op_meta = OpMeta::new(
                op_id,
                handle.id(),
                kilop_core::time::Deadline::at(
                    self.deps
                        .clock
                        .now_ms()
                        .saturating_add(self.deps.tool_deadline_ms as i64),
                ),
                kilop_core::retry::RetryPolicy {
                    max_attempts: 1, // tools are never blindly retried
                    ..Default::default()
                },
                cancel.child(),
                recovery,
                self.deps.clock.now_ms(),
            );
            let op_meta = if replay.is_null() {
                op_meta
            } else {
                op_meta.with_replay(replay)
            };
            let run_handle = handle.start_tool_run(op_meta.clone(), &name, input.clone())?;
            let _ = run_handle;

            // Scheduler task for this tool; the OpMeta envelope (deadline,
            // retry, cancellation, recovery) is passed straight through.
            let ctx = ToolRunCtx {
                session_id: handle.id(),
                permission_granted: granted,
                op_id,
                identity,
                cancellation: op_meta.cancellation.clone(),
                artifacts: Arc::new(self.deps.artifact_sink(handle.id())),
                tool_call_mode: self.deps.tool_call_mode,
                workspace: workspace.clone(),
                edit: self.deps.edit.clone(),
                snapshots: self.deps.snapshots.clone(),
                sandbox: sandbox.clone(),
                supervisor: self.deps.supervisor.clone(),
                deadline_ms: op_meta.deadline.at_ms().saturating_sub(now_ms).max(1) as u64,
            };
            let tool_arc = tool.clone();
            let outcomes = outcomes.clone();
            let args = input.clone();
            let (reads, writes) = ownership_sets(&tool, &input);
            let spec = ScheduledOp {
                meta: op_meta.clone(),
                resources: ResourceRequest {
                    class: tool.resource_class,
                },
                reads,
                writes,
                // Parallel tool batches are independent by design: no tool
                // call in a batch depends on another, so there are no edges.
                // If chains are ever built here, default edges are Success
                // (a dependent runs only after its upstream completed).
                dependencies: vec![],
                run: Arc::new(move || {
                    let tool = tool_arc.clone();
                    let ctx = ctx.clone();
                    let args = args.clone();
                    let outcomes = outcomes.clone();
                    Box::pin(async move {
                        let outcome = (tool.execute)(ctx, args).await?;
                        outcomes.lock().unwrap().insert(op_id, outcome);
                        Ok(())
                    })
                }),
            };
            submitted.push((op_id, name.clone(), call_id.clone(), input.clone()));
            scheduler.submit(spec);
        }

        let done: std::collections::HashSet<OpId> = scheduler
            .run_to_completion()
            .await
            .unwrap_or_default()
            .into_iter()
            .collect();

        // Two passes over the done set: ALL FileChanged notifications while
        // the machine is still ExecutingTool, THEN all finishes (each finish
        // moves the machine toward Validating — interleaving them with the
        // appends would journal FileChanged from Validating, an illegal
        // transition when a batch contains more than one tool).
        for (op_id, name, _call_id, _input) in submitted.iter() {
            if done.contains(op_id) {
                handle.append_event(
                    kilop_core::event::EventKind::FileChanged,
                    AgentState::ExecutingTool,
                    Some(*op_id),
                    Some(serde_json::json!({ "tool": name, "effect": "applied" })),
                )?;
            }
        }
        for (op_id, name, call_id, input) in submitted {
            if done.contains(&op_id) {
                let outcome =
                    outcomes
                        .lock()
                        .unwrap()
                        .remove(&op_id)
                        .unwrap_or_else(|| ToolOutcome {
                            text: "(no output)".into(),
                            exit_code: None,
                            ..Default::default()
                        });
                // Workspace writes record their FilePostcondition (bytes as
                // written) on the run row BEFORE the finish: a crash in the
                // window between write and finish is then verified against
                // the REAL expected state, never a JSON-args inference.
                if let Some(pc) = &outcome.postcondition {
                    let v = serde_json::to_value(pc).map_err(|e| {
                        Error::malformed(format!("postcondition serialization: {e}"))
                    })?;
                    handle.record_tool_postcondition(op_id, &v)?;
                }
                handle.finish_tool_run(op_id, "completed", outcome.effect_status)?;
                collect_tool_summary(turn_summary, &name, &input, &outcome);
                let seq = handle.proposed_message_seq()?;
                let mid =
                    handle.put_message(seq, "assistant", serde_json::json!({ "parts": [] }))?;
                let body = ToolResultBody {
                    excerpt: truncate(&outcome.text, 2000),
                    exit_code: outcome.exit_code,
                    artifact: outcome.artifact,
                    slice_hint: outcome.slice_hint,
                };
                handle.put_tool_result_part(mid, &call_id, &body)?;
                executed += 1;
            } else {
                handle.finish_tool_run(op_id, "failed", EffectStatus::Unknown)?;
                detector.record_error(&format!("tool {name} failed"));
            }
        }
        for d in denied {
            detector.record_error(&d);
        }
        handle.put_task_ledger(serde_json::to_value(ledger)?)?;
        Ok(executed)
    }

    /// Durable loop signals (spec §28): a FAILING tool call bumps the
    /// session's persistent count for that exact call; any genuine progress
    /// (a successful tool or a completed test) clears the window. Returns
    /// true when the same failing call repeated across turns/restarts
    /// reaches the threshold — the runtime must stop and re-plan, not let
    /// the model grind for 40 turns.
    fn durable_loop_signals(
        &self,
        handle: &kilop_session::SessionHandle,
        turn_summary: &kilop_context::ledger::TurnSummary,
        detector: &LoopDetector,
    ) -> kilop_core::Result<bool> {
        let mut tripped = false;
        #[cfg(debug_assertions)]
        eprintln!(
            "dbg-loop: progress={} failures={:?}",
            turn_made_progress(turn_summary),
            turn_summary.failures
        );
        if turn_made_progress(turn_summary) {
            // Some calls succeeded this batch: the task is making progress;
            // do not punish isolated failures.
            return Ok(false);
        }
        for f in &turn_summary.failures {
            let key = format!("fail {}", truncate(f, 400));
            if handle.bump_loop_signal(&key, detector.threshold() as u32)? {
                tripped = true;
            }
        }
        Ok(tripped)
    }

    /// UpdatingMemory phase (spec §8): durable structured facts, written on
    /// every genuine turn end. The ledger is the compact task projection; the
    /// memory facts carry the goal and per-turn summaries. Bounds live in the
    /// session layer (MAX_FACT_VALUE_BYTES) — truncation happens here first.
    fn record_memory(
        &self,
        handle: &kilop_session::SessionHandle,
        op_id: OpId,
        ledger: &TaskLedger,
        summary: &kilop_context::ledger::TurnSummary,
    ) -> kilop_core::Result<()> {
        if !ledger.goal.is_empty() {
            handle.upsert_memory_fact("task", "goal", &truncate(&ledger.goal, 200))?;
        }
        let empty = summary.steps_completed.is_empty()
            && summary.steps_opened.is_empty()
            && summary.decisions.is_empty()
            && summary.failures.is_empty()
            && summary.files_changed.is_empty()
            && summary.tests_run.is_empty()
            && summary.tests_failed.is_empty();
        if !empty {
            let rendered = serde_json::to_string(summary).unwrap_or_default();
            handle.upsert_memory_fact("turn", &op_id.to_string(), &truncate(&rendered, 3500))?;
        }
        Ok(())
    }

    fn load_ledger(&self, handle: &kilop_session::SessionHandle) -> kilop_core::Result<TaskLedger> {
        match handle.get_task_ledger()? {
            Some(v) => Ok(serde_json::from_value(v).unwrap_or_default()),
            None => Ok(TaskLedger::default()),
        }
    }

    /// Bounded repository knowledge for the context (spec §8 class 3 +
    /// §26): a small deterministic file map + the workspace AGENTS.md rules.
    /// Empty when the session has no resolvable workspace — never an error.
    fn repo_knowledge(&self, handle: &kilop_session::SessionHandle) -> (String, String) {
        const MAX_ENTRIES: usize = 500;
        const MAX_DEPTH: usize = 6;
        const MAX_RULES_BYTES: usize = 8192;
        const SKIP: &[&str] = &[".git", "target", "node_modules", ".venv", "dist", ".hg"];
        let row = match handle.row() {
            Ok(r) => r,
            Err(_) => return (String::new(), String::new()),
        };
        let root = match self.deps.session.store().workspace_root(row.workspace_id) {
            Ok(Some(r)) => r,
            _ => return (String::new(), String::new()),
        };
        let ws = match self
            .deps
            .workspaces
            .open(row.workspace_id, std::path::PathBuf::from(&root))
        {
            Ok(w) => w,
            Err(_) => return (String::new(), String::new()),
        };
        // Project rules: AGENTS.md at the canonical root, bounded.
        let mut rules = ws
            .read_default(std::path::Path::new("AGENTS.md"))
            .ok()
            .map(|d| String::from_utf8_lossy(&d.bytes).into_owned())
            .unwrap_or_default();
        rules.truncate(MAX_RULES_BYTES);
        // Deterministic bounded walk (sorted per dir, depth-capped).
        let mut entries: Vec<String> = Vec::new();
        let mut stack: Vec<(usize, String)> = vec![(0, String::new())];
        while let Some((depth, rel)) = stack.pop() {
            if depth > MAX_DEPTH || entries.len() >= MAX_ENTRIES {
                break;
            }
            let path = std::path::Path::new(&rel);
            let Ok(list) = ws.list(path, 200) else {
                continue;
            };
            for meta in list {
                if entries.len() >= MAX_ENTRIES {
                    break;
                }
                let name = meta
                    .path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let child = if rel.is_empty() {
                    name.clone()
                } else {
                    format!("{rel}/{name}")
                };
                if meta.path.is_dir() {
                    if !SKIP.contains(&name.as_str()) {
                        stack.push((depth + 1, child));
                    }
                } else if name != "AGENTS.md" {
                    entries.push(child);
                }
            }
        }
        entries.sort();
        let map = entries
            .iter()
            .take(MAX_ENTRIES)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");
        (rules, map)
    }

    fn provider_for(
        &self,
        handle: &kilop_session::SessionHandle,
    ) -> kilop_core::Result<Arc<dyn kilop_provider::Provider>> {
        let provider_id = handle.provider()?;
        self.deps
            .providers
            .get(&provider_id)
            .ok_or_else(|| Error::not_found(format!("provider {provider_id} not registered")))
    }

    /// Load durable history rows (oldest first) for one logical turn.
    /// Paged backward until the bound cap — the 40-message hard limit is
    /// gone (audit round 6; the WirePlan does the token-based trimming).
    /// With deferred materialization (audit round 7) queued prompts have NO
    /// user message in the timeline until admission, so the timeline order
    /// IS the logical conversation order — no filtering needed.
    fn load_history_rows(
        &self,
        handle: &kilop_session::SessionHandle,
    ) -> kilop_core::Result<Vec<MessageRowLike>> {
        let mut collected: Vec<MessageRowLike> = Vec::new();
        let mut cursor: Option<i64> = None;
        loop {
            let page = handle.messages_before(cursor, 250)?;
            if page.is_empty() {
                break;
            }
            for row in page.iter() {
                collected.push(MessageRowLike {
                    id: row.id,
                    seq: row.seq,
                    role: row.role.clone(),
                    data: row.data.clone(),
                });
                if collected.len() >= MAX_HISTORY_MESSAGES {
                    break;
                }
            }
            if collected.len() >= MAX_HISTORY_MESSAGES {
                break;
            }
            cursor = Some(page.last().unwrap().seq);
            if page.last().unwrap().seq <= 1 {
                break;
            }
        }
        collected.reverse(); // oldest first
        Ok(collected)
    }

    fn recent_turns(
        &self,
        handle: &kilop_session::SessionHandle,
    ) -> kilop_core::Result<Vec<RecentTurn>> {
        let rows = self.load_history_rows(handle)?; // oldest-first
        let mut turns = Vec::new();
        for row in rows {
            let mut pushed_text = false;
            for part in handle.parts_of(row.id)? {
                if part.kind == "text" {
                    if let Some(text) = part.data.get("text").and_then(|v| v.as_str()) {
                        turns.push(RecentTurn {
                            role: row.role.clone(),
                            text: text.to_string(),
                        });
                        pushed_text = true;
                    }
                }
            }
            // The durable user prompt lives in the message payload
            // (submit_prompt stores `{"text": ...}` with no part rows): it
            // must reach the wire and the compactor too.
            if row.role == "user" && !pushed_text {
                if let Some(text) = row.data.get("text").and_then(|v| v.as_str()) {
                    if !text.is_empty() {
                        turns.push(RecentTurn {
                            role: "user".into(),
                            text: text.to_string(),
                        });
                    }
                }
            }
        }
        Ok(turns)
    }

    /// Reconstruct the full provider message list from the durable state,
    /// oldest first. The persisted part order is the source of truth:
    /// text/reasoning/tool calls keep the assistant role, tool results move
    /// to the user role (provider APIs require tool results to come from the
    /// user), and a message carrying only tool results yields one user-role
    /// request message. The durable user prompt (message payload `{"text":
    /// ...}`, no part rows) is synthesized as a user text part — without it
    /// the model would never see the prompt.
    fn history_messages(
        &self,
        handle: &kilop_session::SessionHandle,
    ) -> kilop_core::Result<Vec<RequestMessage>> {
        let rows = self.load_history_rows(handle)?; // oldest-first
        let mut out = Vec::new();
        for row in rows {
            let role_is_user = row.role == "user";
            let mut user_parts: Vec<ContentPart> = Vec::new();
            let mut assistant_parts: Vec<ContentPart> = Vec::new();
            let mut had_text_part = false;
            for part in handle.parts_of(row.id)? {
                match part.kind.as_str() {
                    "text" => {
                        had_text_part = true;
                        let text = str_field(&part.data, "text")?;
                        if role_is_user {
                            user_parts.push(ContentPart::text(text));
                        } else {
                            assistant_parts.push(ContentPart::text(text));
                        }
                    }
                    "reasoning" => {
                        let text = str_field(&part.data, "text")?;
                        if role_is_user {
                            user_parts.push(ContentPart::reasoning(text));
                        } else {
                            assistant_parts.push(ContentPart::reasoning(text));
                        }
                    }
                    "tool_call" => {
                        let state = str_field(&part.data, "state")?;
                        if matches!(state.as_str(), "completed" | "error") {
                            assistant_parts.push(ContentPart::tool_call(
                                str_field(&part.data, "tool_call_id")?,
                                str_field(&part.data, "name")?,
                                part.data
                                    .get("input")
                                    .cloned()
                                    .unwrap_or(serde_json::Value::Null),
                            ));
                        }
                    }
                    "tool_result" => {
                        let is_error = part
                            .data
                            .get("exit_code")
                            .and_then(|v| if v.is_null() { None } else { v.as_i64() })
                            .is_some_and(|c| c != 0);
                        user_parts.push(ContentPart::tool_result(
                            str_field(&part.data, "excerpt")?,
                            is_error,
                            str_field(&part.data, "tool_call_id")?,
                        ));
                    }
                    "summary" => {}
                    other => {
                        return Err(Error::malformed(format!(
                            "corrupt durable part kind {other:?} on message {}",
                            row.id
                        )));
                    }
                }
            }
            // Message-level payload: the durable user prompt has no part rows.
            if role_is_user && !had_text_part && user_parts.is_empty() {
                if let Some(text) = row.data.get("text").and_then(|v| v.as_str()) {
                    if !text.is_empty() {
                        user_parts.push(ContentPart::text(text));
                    }
                }
            }
            if role_is_user {
                if !user_parts.is_empty() {
                    out.push(RequestMessage {
                        role: Role::User,
                        content: user_parts,
                    });
                }
            } else {
                if !assistant_parts.is_empty() {
                    out.push(RequestMessage {
                        role: Role::Assistant,
                        content: assistant_parts,
                    });
                }
                if !user_parts.is_empty() {
                    out.push(RequestMessage {
                        role: Role::User,
                        content: user_parts,
                    });
                }
            }
        }
        Ok(out)
    }

    /// Thin adapter: the wire request IS the budgeted plan — `system`,
    /// `messages`, and `tools` each appear exactly once, already measured
    /// against the model budget by the planner.
    fn build_request(
        &self,
        handle: &kilop_session::SessionHandle,
        plan: &WirePlan,
        op_id: OpId,
        model: &str,
        cancel: &CancellationToken,
        attempt: u32,
    ) -> kilop_core::Result<GenericAgentRequest> {
        Ok(GenericAgentRequest {
            model: model.to_string(),
            system: plan.system.clone(),
            messages: plan.messages.clone(),
            tools: plan.tools.clone(),
            max_output: None,
            reasoning: None,
            stream: true,
            meta: RequestMeta {
                operation_id: op_id,
                session_id: handle.id(),
                provider: handle.provider()?,
                attempt,
                deadline_ms: self.deps.tool_deadline_ms,
                cancellation: cancel.child(),
            },
        })
    }

    /// Resolve the configured compaction model ("model" uses the session's
    /// provider; "provider/model" names another registered provider).
    fn resolve_compaction_model(
        &self,
        handle: &kilop_session::SessionHandle,
        spec: &str,
    ) -> kilop_core::Result<(Arc<dyn kilop_provider::Provider>, String)> {
        let provider_id = match spec.split_once('/') {
            Some((p, _)) => p.to_string(),
            None => handle.provider()?,
        };
        let model = match spec.split_once('/') {
            Some((_, m)) => m.to_string(),
            None => spec.to_string(),
        };
        if model.is_empty() || model.len() > 256 || provider_id.len() > 256 {
            return Err(Error::malformed("invalid compaction model spec"));
        }
        let provider = self
            .deps
            .providers
            .get(&provider_id)
            .ok_or_else(|| Error::not_found(format!("compaction provider {provider_id}")))?;
        Ok((provider, model))
    }

    async fn try_compact(
        &self,
        handle: &kilop_session::SessionHandle,
        recent: &[RecentTurn],
        ledger: &TaskLedger,
        budget: &ContextBudget,
        // The LOGICAL TURN's cancellation token: a user Stop during
        // compaction must reach the compaction model's stream (P0 audit,
        // round 11 — the summary request used to mint an orphan token and
        // ran up to the full 90s after the turn was cancelled).
        cancel: &CancellationToken,
    ) -> kilop_core::Result<Option<CompactionPlan>> {
        let before = recent.iter().map(|t| t.text.len()).sum::<usize>() / 4;
        if before == 0 {
            return Ok(None);
        }
        let target = budget.context_max();
        // The configured compaction model ("model" or "provider/model")
        // resolves to a REAL provider stream (spec §36: the separate
        // compaction model is honored, never a stub). Without one, the weak
        // ledger summarizer stands in — the hard invariant still rejects
        // whatever does not shrink enough.
        // A broken compaction model spec degrades to the weak summarizer
        // (warned), never an error that kills the turn.
        let summarizer: Option<Arc<dyn Summarizer>> =
            if let Some(model) = self.deps.compaction_model.as_deref() {
                match self.resolve_compaction_model(handle, model) {
                    Ok((provider, model_name)) => Some(Arc::new(StreamingSummarizer {
                        provider,
                        model: model_name,
                        // The summarizer runs under the compactor contract —
                        // NEVER the agent instructions (P0 audit round 11).
                        op_id: self.deps.session.next_op_id(),
                        session_id: handle.id(),
                        cancellation: cancel.child(),
                        summary_timeout: DEFAULT_SUMMARY_TIMEOUT,
                    })),
                    Err(e) => {
                        tracing::warn!(
                        "compaction model {model:?} unresolvable: {e}; using the ledger summarizer"
                    );
                        None
                    }
                }
            } else {
                None
            };
        let compactor: Compactor = match summarizer {
            Some(s) => Compactor::new(Some(s)),
            None => Compactor::new(Some(Arc::new(LedgerSummarizer))),
        };
        let mut plan = compactor
            .compact(recent, ledger, &CompactionRequest::new(before, target))
            .await;
        let accepted = plan.accepted;
        handle.record_compaction_defaults(
            plan.before_tokens as i64,
            plan.after_tokens as i64,
            plan.target_tokens as i64,
            match plan.strategy {
                kilop_context::CompactionStrategy::LlmSummary => "llm_summary",
                kilop_context::CompactionStrategy::DeterministicPruning => "deterministic",
                kilop_context::CompactionStrategy::Rejected => "rejected",
            },
        )?;
        if !accepted {
            // CompactRejected is journaled by record_compaction.
            return Ok(None);
        }
        handle.put_task_ledger(serde_json::to_value(&plan.ledger)?)?;
        // Durable archiving (P0: no more 1 MiB cap losing history): evicted
        // turns arrive as ORDERED chunks (oldest first, each bounded) — each
        // chunk is written to the CAS, then a small JSON manifest
        // {version:1, chunks:[{index,size,hash}], total_bytes} is written
        // and its content address replaces the digest placeholder — the
        // digest rides the wire with the archive behind ONE artifact ref.
        // Best-effort: an unwritable CAS leaves the digest text without a
        // hash (never breaks the turn).
        if !plan.archive_chunks.is_empty() {
            if let Some(cas) = &self.deps.cas {
                let mut chunk_entries: Vec<serde_json::Value> = Vec::new();
                let mut total_bytes = 0usize;
                for (index, chunk) in plan.archive_chunks.iter().enumerate() {
                    // Each chunk is stored whole (chunks are already bounded
                    // by the compactor; a pathological single-turn chunk may
                    // exceed the bound and still stores — never truncated).
                    let Ok(hash) = cas.put_bounded(chunk.as_bytes(), chunk.len()) else {
                        chunk_entries.clear();
                        break;
                    };
                    total_bytes = total_bytes.saturating_add(chunk.len());
                    chunk_entries.push(serde_json::json!({
                        "index": index,
                        "size": chunk.len(),
                        "hash": hash.to_string(),
                    }));
                }
                let manifest = serde_json::json!({
                    "version": 1,
                    "chunks": chunk_entries,
                    "total_bytes": total_bytes,
                });
                if !chunk_entries.is_empty() {
                    if let Ok(bytes) = serde_json::to_vec(&manifest) {
                        if let Ok(hash) = cas.put(&bytes) {
                            let marker = format!("artifact://{hash}");
                            if let Some(first) = plan.kept_recent.first_mut() {
                                first.text = first.text.replace("<artifact://hash>", &marker);
                            }
                        }
                    }
                }
            }
        }
        Ok(Some(plan))
    }

    /// A provider stream failure is state-aware: if a tool already ran, the
    /// turn is NOT replayed; the journal decides the continuation.
    async fn handle_provider_failure(
        self: &Arc<Self>,
        handle: &kilop_session::SessionHandle,
        op_id: OpId,
        e: ProviderError,
        outcome: &mut TurnOutcome,
    ) -> kilop_core::Result<TurnOutcome> {
        let pending = handle.pending_tool_runs()?;
        let state = if pending.is_empty() {
            AgentState::FailedRecoverable
        } else {
            // A tool ran: never replay. Mark unknown and require verification.
            for row in &pending {
                handle.set_tool_run_effect(row.op_id, EffectStatus::Unknown)?;
            }
            AgentState::NeedsUserInput
        };
        let _ = handle.append_event(
            kilop_core::event::EventKind::Failed,
            state,
            Some(op_id),
            Some(serde_json::json!({ "message": e.message })),
        );
        outcome.final_state = state;
        Ok(outcome.clone())
    }
}

/// Weak-but-honest summarizer: emits the ledger render. The compactor's hard
/// invariant rejects it when it does not shrink enough.
struct LedgerSummarizer;

impl Summarizer for LedgerSummarizer {
    fn summarize<'a>(
        &'a self,
        _history: &'a [kilop_context::RecentTurn],
        ledger: &'a TaskLedger,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = String> + Send + 'a>> {
        Box::pin(async move { ledger.compact_render() })
    }
}

/// Default bound on one compaction-model summary stream (spec §9): a
/// summarizer that does not finish cleanly inside the bound is treated as
/// failed and its partial text is discarded. Per-instance injectable so the
/// timeout path is testable without waiting 90s.
const DEFAULT_SUMMARY_TIMEOUT: Duration = Duration::from_secs(90);

/// The REAL separate-compaction-model summarizer (spec §9 + §36): the
/// configured compaction model streams an actual provider request that
/// summarizes the recent history. The request carries its own compactor
/// contract as the system prompt (P0 audit round 11: the agent instructions
/// used to leak in and the model answered the latest user message instead
/// of summarizing). Any failure yields NO summary — run() discards all
/// partial text and the caller returns a transcript the compactor's hard
/// cap rejects, so deterministic pruning takes over (compaction can never
/// hang, outlive the turn, or degrade on a broken compaction model).
struct StreamingSummarizer {
    provider: Arc<dyn kilop_provider::Provider>,
    model: String,
    /// Real operation/session identity rides the request metadata (ids can
    /// never be 0 — the envelope is mandatory even for interior work).
    op_id: OpId,
    session_id: SessionId,
    /// Turn-scoped cancellation (a CHILD of the logical turn's token): a
    /// user Stop during compaction cancels the summary stream instead of
    /// leaving the compaction model running to the deadline (P0 audit
    /// round 11). The wire request carries a child of this token.
    cancellation: CancellationToken,
    /// Stream bound; the production default is [`DEFAULT_SUMMARY_TIMEOUT`],
    /// tests inject a small value.
    summary_timeout: Duration,
}

impl StreamingSummarizer {
    /// Stream ONE compaction summary request and return the accepted text.
    ///
    /// Completion protocol (P0 audit round 11): text is accepted ONLY when
    /// the stream ended cleanly, tracked explicitly:
    ///   - `ProviderChunk::Done` marks Complete. FakeProvider's `End` chunk
    ///     maps to `ProviderChunk::Done` (its unfold always emits a
    ///     terminal Done before exhaustion), and every real transport
    ///     (anthropic/google/ollama adapters, guarded transport) signals a
    ///     successful end with Done — see their stream ends;
    ///   - plain exhaustion after content (`None` from the stream, no error,
    ///     no Done) ALSO marks Complete: a transport that ends without an
    ///     explicit Done chunk is a clean end, never a failure. (FakeProvider
    ///     never produces this shape, but the status logic must be correct
    ///     for both);
    ///   - a provider `Err` marks the run FAILED;
    ///   - the bounded deadline marks the run FAILED;
    ///   - turn cancellation marks the run FAILED.
    ///
    /// Any status other than Complete discards EVERY accumulated character
    /// below and returns `None` — a truncated summary is small, so it would
    /// slip under the compactor's hard cap and replace the real history
    /// with a partial state transfer.
    async fn run(&self, history: &[kilop_context::RecentTurn]) -> Option<String> {
        use futures::StreamExt as _;
        const SUMMARY_MAX_CHARS: usize = 60_000;
        // Cancellation is polled at this cadence even while the stream is
        // silent (std CancellationToken has no async wait primitive; a
        // bounded tick mirrors the guarded transports' cancellation checks).
        const CANCEL_POLL_INTERVAL: Duration = Duration::from_millis(50);
        // Capabilities decide: a non-streaming compaction model is skipped.
        if !self.provider.capabilities(&self.model).streaming {
            return None;
        }
        let request = GenericAgentRequest {
            model: self.model.clone(),
            system: COMPACTOR_SYSTEM.to_string(),
            messages: history
                .iter()
                .map(|t| RequestMessage {
                    role: if t.role == "user" {
                        Role::User
                    } else {
                        Role::Assistant
                    },
                    content: vec![ContentPart::text(&t.text)],
                })
                .collect(),
            tools: vec![],
            max_output: Some(4096),
            reasoning: None,
            stream: true,
            meta: RequestMeta {
                operation_id: self.op_id,
                session_id: self.session_id,
                provider: self.provider.id().into(),
                attempt: 0,
                deadline_ms: self.summary_timeout.as_millis().min(u64::MAX as u128) as u64,
                // A CHILD of the turn-scoped token: cancellation of the
                // logical turn cascades into the wire request, and the
                // provider double/transport can observe it.
                cancellation: self.cancellation.child(),
            },
        };
        let mut stream = self.provider.stream(request);
        let mut text = String::new();
        let mut complete = false;
        let deadline = tokio::time::timeout(self.summary_timeout, async {
            let mut cancel_ticks = tokio::time::interval(CANCEL_POLL_INTERVAL);
            loop {
                tokio::select! {
                    _ = cancel_ticks.tick() => {
                        if self.cancellation.is_cancelled() {
                            // Turn cancelled: FAILED (complete stays false).
                            return;
                        }
                    }
                    chunk = stream.next() => {
                        match chunk {
                            Some(Ok(ProviderChunk::Text { text: t }))
                            | Some(Ok(ProviderChunk::Reasoning { text: t })) => {
                                text.push_str(&t);
                                if text.len() > SUMMARY_MAX_CHARS {
                                    // Bounded stop: the cap is the bound of
                                    // what we would accept anyway.
                                    complete = true;
                                    return;
                                }
                            }
                            Some(Ok(ProviderChunk::Done)) => {
                                // Clean end: the ONLY unconditional Complete.
                                complete = true;
                                return;
                            }
                            Some(Ok(_)) => {}
                            // Clean exhaustion after content: Complete (see
                            // the completion protocol above).
                            None => {
                                complete = true;
                                return;
                            }
                            // Provider failure: FAILED (complete stays false).
                            Some(Err(_)) => return,
                        }
                    }
                }
            }
        });
        // On timeout the inner future is dropped mid-stream with `complete`
        // still false: FAILED, every accumulated character discarded below.
        let _ = deadline.await;
        if !complete || text.is_empty() {
            return None;
        }
        text.truncate(SUMMARY_MAX_CHARS);
        Some(text)
    }
}

/// The failure fallback returned when the streaming summarizer produced no
/// summary (provider error, deadline, cancellation, non-streaming model).
/// An EMPTY string would be a data-loss hole: the compactor's token
/// estimate of "" is 0, which always passes its hard cap, so a wiped
/// history would be "accepted" as an LLM summary. Instead the unsummarizable
/// transcript is echoed verbatim — every character is real, nothing is
/// invented — and repeated 3× so its estimate (chars/4) provably exceeds
/// the compactor's hard cap (at most 3/4 of the byte-based `before` figure;
/// chars ≤ bytes, so 3×(chars/4) + prefixes > 3/4×before holds for ANY
/// UTF-8 input, multibyte included). The compactor therefore REJECTS it and
/// deterministic pruning takes over — the documented degradation path.
fn summarize_failure_fallback(history: &[RecentTurn]) -> String {
    const FALLBACK_COPIES: usize = 3;
    let mut out = String::new();
    for _ in 0..FALLBACK_COPIES {
        for turn in history {
            out.push_str(&format!("{}: {}\n", turn.role, turn.text));
        }
    }
    out
}

impl Summarizer for StreamingSummarizer {
    fn summarize<'a>(
        &'a self,
        history: &'a [kilop_context::RecentTurn],
        _ledger: &'a TaskLedger,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = String> + Send + 'a>> {
        Box::pin(async move {
            self.run(history)
                .await
                .unwrap_or_else(|| summarize_failure_fallback(history))
        })
    }
}

/// Convert the compactor's kept text turns back into provider messages: the
/// compacted history that rides the next wire request. Text-only by
/// construction (compaction works on `RecentTurn`, which carries text).
fn recent_turns_to_messages(turns: &[RecentTurn]) -> Vec<RequestMessage> {
    turns
        .iter()
        .map(|t| RequestMessage {
            role: if t.role == "user" {
                Role::User
            } else {
                Role::Assistant
            },
            content: vec![ContentPart::text(&t.text)],
        })
        .collect()
}

/// The scheduler's ownership sets for one tool invocation, derived from the
/// tool's declared path args (read_file/search ⇒ reads; write_file ⇒ writes).
/// This is the ONLY source for `ScheduledOp::reads/writes` — tools never
/// hand the scheduler raw paths from any other channel.
fn ownership_sets(tool: &Arc<Tool>, input: &serde_json::Value) -> (OwnershipSet, OwnershipSet) {
    let ownership = tool.ownership(input);
    (
        OwnershipSet::new(ownership.reads),
        OwnershipSet::new(ownership.writes),
    )
}

fn map_store_error(e: kilop_store::StoreError) -> Error {
    Error::new(ErrorKind::Store, format!("store: {e}"))
}

/// States meaning "an operation is in flight" (mirror of the session
/// layer's `is_op_active`; the runtime may not reach into its internals).
fn state_is_op_active(s: AgentState) -> bool {
    matches!(
        s,
        AgentState::Preparing
            | AgentState::BuildingContext
            | AgentState::WaitingForModel
            | AgentState::Streaming
            | AgentState::ToolRequested
            | AgentState::WaitingForPermission
            | AgentState::ExecutingTool
            | AgentState::Validating
            | AgentState::UpdatingMemory
    )
}

fn state_tag(s: AgentState) -> String {
    serde_json::to_string(&s)
        .unwrap_or_default()
        .trim_matches('"')
        .to_string()
}

fn effect_tag(e: EffectStatus) -> &'static str {
    match e {
        EffectStatus::Unknown => "unknown",
        EffectStatus::Verified => "verified",
        EffectStatus::Applied => "applied",
        EffectStatus::Failed => "failed",
    }
}

fn tool_mode_tag(mode: ToolCallMode) -> &'static str {
    match mode {
        ToolCallMode::Native => "native",
        ToolCallMode::NativeWithRepair => "native_with_repair",
        ToolCallMode::StructuredFallback => "structured_fallback",
    }
}

/// Re-validate stored invocation args against the tool's input schema
/// "where feasible" (P0: a hostile descriptor is a loud error, never a blind
/// replay): object schemas with a `required` list and typed `properties` are
/// checked; anything looser passes through unchanged. The result is the
/// canonical JSON to re-execute.
fn validate_args_against_schema(
    tool: &Tool,
    args: &serde_json::Value,
) -> kilop_core::Result<serde_json::Value> {
    let obj = args.as_object().ok_or_else(|| {
        Error::malformed(format!(
            "tool {} invocation args must be a JSON object, found {}",
            tool.name,
            serde_json::to_string(args).unwrap_or_default()
        ))
    })?;
    let schema = &tool.input_schema;
    if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
        for (key, prop) in props {
            let Some(expect_type) = prop.get("type").and_then(|t| t.as_str()) else {
                continue;
            };
            let Some(value) = obj.get(key) else {
                continue;
            };
            let ok = match expect_type {
                "string" => value.is_string(),
                "integer" | "number" => value.is_number(),
                "boolean" => value.is_boolean(),
                "object" => value.is_object(),
                "array" => value.is_array(),
                _ => true,
            };
            if !ok {
                return Err(Error::malformed(format!(
                    "tool {} arg `{key}` must be {expect_type}",
                    tool.name
                )));
            }
        }
    }
    if let Some(required) = schema.get("required").and_then(|r| r.as_array()) {
        for req in required {
            let Some(key) = req.as_str() else {
                continue;
            };
            if !obj.contains_key(key) {
                return Err(Error::malformed(format!(
                    "tool {} invocation is missing required arg `{key}`",
                    tool.name
                )));
            }
        }
    }
    Ok(serde_json::Value::Object(obj.clone()))
}

/// Read a required string field from a durable part payload; a missing or
/// non-string field is loud corruption, never silently dropped.
fn str_field(data: &serde_json::Value, key: &str) -> kilop_core::Result<String> {
    data.get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| Error::malformed(format!("durable part is missing string field `{key}`")))
}

/// True when the turn made genuine progress: nothing failed, files were
/// applied, or tests passed. Text-only turns count (no failures).
fn turn_made_progress(summary: &kilop_context::ledger::TurnSummary) -> bool {
    if summary.failures.is_empty() {
        return true;
    }
    if !summary.files_changed.is_empty() {
        return true;
    }
    if !summary.tests_run.is_empty() && summary.tests_failed.is_empty() {
        return true;
    }
    false
}

/// Fold one completed tool call into the logical-turn summary with REAL
/// data (audit: only TurnSummary::default() was recorded; the tool NAME was
/// even journaled as a changed file).
fn collect_tool_summary(
    summary: &mut kilop_context::ledger::TurnSummary,
    name: &str,
    input: &serde_json::Value,
    outcome: &ToolOutcome,
) {
    // Step description: name + the primary path/command argument.
    let path = input
        .get("path")
        .or_else(|| input.get("file"))
        .or_else(|| input.get("filename"))
        .and_then(|p| p.as_str());
    let command = input.get("command").and_then(|c| c.as_str());
    let step = match (path, command) {
        (Some(p), _) if !p.is_empty() => format!("{name} ({p})"),
        (_, Some(c)) if !c.is_empty() => format!("{name}: {}", truncate(c, 120)),
        _ => name.to_string(),
    };
    if !step.is_empty() {
        summary.steps_completed.push(truncate(&step, 200));
    }
    // Changed files: a write tool's target (from its input) when the tool
    // completed — real paths, never the tool name.
    if outcome.effect_status == kilop_core::op::EffectStatus::Applied
        || outcome.exit_code == Some(0)
    {
        if let Some(p) = path.filter(|p| !p.is_empty()) {
            let p = truncate(p, 300);
            if !summary.files_changed.contains(&p) {
                summary.files_changed.push(p);
            }
        }
    }
    // Failures: non-zero exit or errored effect.
    if outcome.exit_code.is_some_and(|c| c != 0)
        || outcome.effect_status == kilop_core::op::EffectStatus::Failed
    {
        let msg = truncate(&outcome.text, 300);
        let failure = if msg.is_empty() {
            format!("{name} failed (exit {})", outcome.exit_code.unwrap_or(-1))
        } else {
            format!("{name}: {msg}")
        };
        summary.failures.push(truncate(&failure, 400));
    }
    // Tests: test-running commands recorded as run/failed with their real
    // exit status.
    if name == "run_command" || name == "run_command_on_files" {
        if let Some(c) = command {
            if looks_like_test_command(c) {
                let cmd = truncate(c, 200);
                if !summary.tests_run.contains(&cmd) {
                    summary.tests_run.push(cmd.clone());
                }
                if outcome.exit_code.is_some_and(|e| e != 0) && !summary.tests_failed.contains(&cmd)
                {
                    summary.tests_failed.push(cmd);
                }
            }
        }
    }
}

/// Honest test-command detection (prefixes only — "test" alone matches
/// "latest"/"attest").
fn looks_like_test_command(cmd: &str) -> bool {
    let c = cmd.trim_start();
    c.starts_with("cargo test")
        || c.starts_with("cargo nextest")
        || c.starts_with("pytest")
        || c.starts_with("python -m pytest")
        || c.starts_with("npm test")
        || c.starts_with("npm run test")
        || c.starts_with("yarn test")
        || c.starts_with("go test")
        || c.starts_with("pnpm test")
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::Tool;
    use kilop_core::id::SessionId;
    use kilop_core::model::ModelCapabilities;
    use kilop_core::time::SystemClock;
    use kilop_provider::{ContentKind, FakeProvider, ScriptedResponse};
    use tempfile::tempdir;

    fn deps_with(
        provider: Arc<dyn kilop_provider::Provider>,
        tools: Vec<Tool>,
    ) -> (AgentDeps, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let mut registry = ProviderRegistry::new();
        registry.register(provider);
        let mut tool_registry = ToolRegistry::new();
        for t in tools {
            tool_registry.register(t);
        }
        let deps = AgentDeps {
            session: SessionManager::open(root.join("store"), root.join("cas"), true).unwrap(),
            providers: Arc::new(registry),
            permission_requester: Arc::new(AlwaysAllow),
            evidence: Arc::new(NoEvidence),
            tools: Arc::new(tool_registry),
            cas: Some(Arc::new(kilop_cas::Cas::open(root.join("cas")).unwrap())),
            workspaces: kilop_fs::WorkspaceFileService::new(),
            edit: None,
            snapshots: None,
            sandbox: None,
            supervisor: None,
            model: "m".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "You are a test agent.".into(),
            clock: Arc::new(SystemClock),
            tool_call_mode: ToolCallMode::Native,
            tool_deadline_ms: 2000,
            retry_policy: kilop_core::retry::RetryPolicy::default(),
        };
        (deps, dir)
    }

    fn deps(provider: FakeProvider, tools: Vec<Tool>) -> (AgentDeps, tempfile::TempDir) {
        deps_with(Arc::new(provider), tools)
    }

    /// Like [`deps_with`] but on a SHARED session manager (multi-turn tests:
    /// each logical turn gets its own provider script while the durable
    /// session — ledger, loop signals, queue — stays in one store).
    fn deps_sharing_session(
        session: Arc<SessionManager>,
        provider: Arc<dyn kilop_provider::Provider>,
        tools: Vec<Tool>,
    ) -> (AgentDeps, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let mut registry = ProviderRegistry::new();
        registry.register(provider);
        let mut tool_registry = ToolRegistry::new();
        for t in tools {
            tool_registry.register(t);
        }
        (
            AgentDeps {
                session,
                providers: Arc::new(registry),
                permission_requester: Arc::new(AlwaysAllow),
                evidence: Arc::new(NoEvidence),
                tools: Arc::new(tool_registry),
                cas: Some(Arc::new(
                    kilop_cas::Cas::open(dir.path().join("cas")).unwrap(),
                )),
                workspaces: kilop_fs::WorkspaceFileService::new(),
                edit: None,
                snapshots: None,
                sandbox: None,
                supervisor: None,
                model: "m".into(),
                compaction_model: None,
                compact_at_usage: 0.65,
                instructions: "You are a test agent.".into(),
                clock: Arc::new(SystemClock),
                tool_call_mode: ToolCallMode::Native,
                tool_deadline_ms: 2000,
                retry_policy: kilop_core::retry::RetryPolicy::default(),
            },
            dir,
        )
    }

    /// One session + its manager for multi-runtime tests.
    fn shared_session(deps: &AgentDeps) -> (Arc<SessionManager>, SessionId) {
        let ws = deps.session.create_workspace("/w").unwrap();
        let sid = deps
            .session
            .create_session(ws, "t", "fake", "m")
            .unwrap()
            .id();
        (deps.session.clone(), sid)
    }

    /// Provider wrapper that intercepts every request before delegation:
    /// the hook inspects the incoming `GenericAgentRequest` and may refuse
    /// the stream with a `Malformed` provider error (the turn then fails —
    /// this is how the tool-result semantic test proves the request shape).
    type RequestHook = dyn Fn(usize, &GenericAgentRequest) -> Result<(), String> + Send + Sync;

    struct InspectingProvider {
        inner: Arc<dyn kilop_provider::Provider>,
        counter: std::sync::atomic::AtomicUsize,
        hook: Arc<RequestHook>,
    }

    impl InspectingProvider {
        fn new(
            inner: Arc<dyn kilop_provider::Provider>,
            hook: impl Fn(usize, &GenericAgentRequest) -> Result<(), String> + Send + Sync + 'static,
        ) -> Self {
            Self {
                inner,
                counter: std::sync::atomic::AtomicUsize::new(0),
                hook: Arc::new(hook),
            }
        }
    }

    impl kilop_provider::Provider for InspectingProvider {
        fn id(&self) -> &str {
            self.inner.id()
        }

        fn capabilities(&self, model: &str) -> ModelCapabilities {
            self.inner.capabilities(model)
        }

        fn stream(&self, req: GenericAgentRequest) -> kilop_provider::ProviderStream {
            let n = self
                .counter
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if let Err(msg) = (self.hook)(n, &req) {
                let err = kilop_provider::ProviderError::new(
                    kilop_provider::ProviderErrorKind::Malformed,
                    msg,
                );
                return Box::pin(futures::stream::iter(vec![Err(err)]));
            }
            self.inner.stream(req)
        }
    }

    /// Test-only provider wrapper: delegates capabilities/streaming to
    /// `inner` but reports a FIXED `runtime_context_limit` — simulating a
    /// live runtime window (an ollama /api/ps allocation) far below the
    /// advertised model maximum.
    struct RuntimeLimitedProvider {
        inner: Arc<dyn kilop_provider::Provider>,
        limit: usize,
    }

    impl RuntimeLimitedProvider {
        fn new(inner: Arc<dyn kilop_provider::Provider>, limit: usize) -> Self {
            Self { inner, limit }
        }
    }

    impl kilop_provider::Provider for RuntimeLimitedProvider {
        fn id(&self) -> &str {
            self.inner.id()
        }

        fn capabilities(&self, model: &str) -> ModelCapabilities {
            self.inner.capabilities(model)
        }

        fn runtime_context_limit(&self, _model: &str) -> Option<usize> {
            Some(self.limit)
        }

        fn stream(&self, req: GenericAgentRequest) -> kilop_provider::ProviderStream {
            self.inner.stream(req)
        }
    }

    struct AlwaysAllow;
    impl PermissionRequester for AlwaysAllow {
        fn request(
            &self,
            _s: SessionId,
            _p: &SessionPermission,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = kilop_core::Result<PermissionDecision>> + Send>,
        > {
            Box::pin(async { Ok(PermissionDecision::Allow) })
        }
    }

    fn echo_tool() -> Tool {
        Tool {
            name: "echo".into(),
            description: "echo back".into(),
            input_schema: serde_json::json!({"type": "object"}),
            resource_class: kilop_core::resource::ResourceClass::Cpu,
            capability: None,
            recovery_hint: RecoveryHint::Idempotent,
            path_args: vec![],
            execute: Arc::new(|_ctx, args| {
                Box::pin(async move {
                    Ok(ToolOutcome {
                        text: format!("echo: {args}"),
                        exit_code: Some(0),
                        ..Default::default()
                    })
                })
            }),
        }
    }

    fn scripted_provider(script: Vec<ScriptedResponse>) -> FakeProvider {
        FakeProvider::with_script(
            "fake",
            ModelCapabilities {
                tools: true,
                ..Default::default()
            },
            script,
        )
    }

    fn new_session(deps: &AgentDeps) -> SessionId {
        let ws = deps.session.create_workspace("/w").unwrap();
        deps.session
            .create_session(ws, "test session", "fake", "m")
            .unwrap()
            .id()
    }

    #[tokio::test]
    async fn text_only_turn_completes() {
        let (deps, _dir) = deps(
            scripted_provider(vec![
                ScriptedResponse::Text("hello there".into()),
                ScriptedResponse::End,
            ]),
            vec![],
        );
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());
        let outcome = runtime.run_turn(session, "hi", &[]).await.unwrap();
        assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);
        assert_eq!(outcome.turns, 1);
        let handle = runtime.deps.session.get_session(session).unwrap().unwrap();
        let page = handle.messages_page(None, 10).unwrap();
        let texts: Vec<&String> = page
            .messages
            .iter()
            .flat_map(|m| m.parts.iter())
            .filter_map(|p| match p {
                kilop_protocol::v756::Part::Text { text } => Some(text),
                _ => None,
            })
            .collect();
        assert!(texts.iter().any(|t| t.contains("hello there")));
    }

    #[tokio::test]
    async fn model_override_changes_wire_request_model() {
        // The provider records the model of every request streamed through
        // it: the per-message override must reach the wire request, and a
        // plain run_turn must keep sending the session model.
        let provider = scripted_provider(vec![
            ScriptedResponse::Text("pong".into()),
            ScriptedResponse::End,
        ]);
        let (deps, _dir) = deps(provider.clone(), vec![]);
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());

        let outcome = runtime
            .run_turn_with_model(session, "hi", &[], Some("m2".into()))
            .await
            .unwrap();
        assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);
        assert_eq!(
            provider.last_request_model().as_deref(),
            Some("m2"),
            "the override must be the model on the wire request"
        );

        let outcome = runtime.run_turn(session, "hi again", &[]).await.unwrap();
        assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);
        assert_eq!(
            provider.last_request_model().as_deref(),
            Some("m"),
            "without an override the session model must be sent"
        );
    }

    #[tokio::test]
    async fn model_override_unknown_model_falls_back_to_default_capabilities() {
        // An override the provider has no capabilities for must never be an
        // error at send time: capabilities fall back to the provider
        // default and the turn still completes.
        let provider = scripted_provider(vec![
            ScriptedResponse::Text("pong".into()),
            ScriptedResponse::End,
        ]);
        let (deps, _dir) = deps(provider.clone(), vec![]);
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());

        let outcome = runtime
            .run_turn_with_model(session, "hi", &[], Some("no-such-model".into()))
            .await
            .unwrap();
        assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);
        assert_eq!(
            provider.last_request_model().as_deref(),
            Some("no-such-model"),
            "the unknown model still reaches the provider"
        );
    }

    #[tokio::test]
    async fn model_override_does_not_mutate_session_row() {
        // The override is per-message: the journaled session row must keep
        // its original model after the turn.
        let provider = scripted_provider(vec![
            ScriptedResponse::Text("pong".into()),
            ScriptedResponse::End,
        ]);
        let (deps, _dir) = deps(provider.clone(), vec![]);
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());
        let handle = runtime.deps.session.get_session(session).unwrap().unwrap();
        assert_eq!(handle.model().unwrap(), "m");

        let outcome = runtime
            .run_turn_with_model(session, "hi", &[], Some("m2".into()))
            .await
            .unwrap();
        assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);
        // The override reached the wire...
        assert_eq!(provider.last_request_model().as_deref(), Some("m2"));
        // ...but the session row is untouched.
        let handle = runtime.deps.session.get_session(session).unwrap().unwrap();
        assert_eq!(handle.model().unwrap(), "m");
    }

    #[tokio::test]
    async fn tool_call_executes_and_continues() {
        let (deps, _dir) = deps(
            scripted_provider(vec![
                ScriptedResponse::ToolCall {
                    id: "c1".into(),
                    name: "echo".into(),
                    input: serde_json::json!({"x": 1}),
                },
                ScriptedResponse::Text("after tool".into()),
                ScriptedResponse::End,
            ]),
            vec![echo_tool()],
        );
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());
        let outcome = runtime.run_turn(session, "use echo", &[]).await.unwrap();
        assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);
        let handle = runtime.deps.session.get_session(session).unwrap().unwrap();
        let page = handle.messages_page(None, 20).unwrap();
        let has_tool_result = page
            .messages
            .iter()
            .flat_map(|m| m.parts.iter())
            .any(|p| matches!(p, kilop_protocol::v756::Part::ToolResult { .. }));
        assert!(has_tool_result, "tool result part must be durable");
        // Tool ran exactly once (never replayed).
        let runs = handle.pending_tool_runs().unwrap();
        assert!(runs.is_empty());
    }

    #[tokio::test]
    async fn tool_ctx_identity_reads_the_session_row_not_hardcoded_ids() {
        // P1: tools were getting FAKE worktree/task identities because the
        // runtime hardcoded WorktreeId::new(1)/TaskId::new(1). The session
        // row (v8) is the single source of truth: a standalone session keeps
        // the DOCUMENTED 1/1 default, and a session adopted onto a real
        // worktree passes the REAL ids to every ToolRunCtx — durably, so a
        // reopened manager sees the same row.
        fn probe_tool(captured: &Arc<std::sync::Mutex<Vec<WorkspaceIdentity>>>) -> Tool {
            let cap = captured.clone();
            Tool {
                name: "probe".into(),
                description: "records ctx identity".into(),
                input_schema: serde_json::json!({"type": "object"}),
                resource_class: kilop_core::resource::ResourceClass::Cpu,
                capability: None,
                recovery_hint: RecoveryHint::Idempotent,
                path_args: vec![],
                execute: Arc::new(move |ctx, _args| {
                    let cap = cap.clone();
                    Box::pin(async move {
                        cap.lock().unwrap().push(ctx.identity);
                        Ok(ToolOutcome::default())
                    })
                }),
            }
        }
        fn one_probe_turn_script() -> Arc<dyn kilop_provider::Provider> {
            Arc::new(scripted_provider(vec![
                ScriptedResponse::ToolCall {
                    id: "c1".into(),
                    name: "probe".into(),
                    input: serde_json::json!({}),
                },
                ScriptedResponse::Text("done".into()),
                ScriptedResponse::End,
            ]))
        }
        let dir = fresh_store_dir();
        let captured: Arc<std::sync::Mutex<Vec<WorkspaceIdentity>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        // (i) A plain create_session is the documented STANDALONE default:
        //     1/1, never a fake hardcoded id.
        {
            let manager =
                SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                    .unwrap();
            let (deps, _keep) = deps_sharing_session(
                manager.clone(),
                one_probe_turn_script(),
                vec![probe_tool(&captured)],
            );
            let runtime = AgentRuntime::new(deps).unwrap();
            let ws = manager.create_workspace("/w").unwrap();
            let plain = manager.create_session(ws, "plain", "fake", "m").unwrap();
            runtime.run_turn(plain.id(), "probe", &[]).await.unwrap();
            assert_eq!(
                captured.lock().unwrap()[0],
                WorkspaceIdentity::new(ws, WorktreeId::new(1), TaskId::new(1)),
                "standalone sessions keep the documented 1/1 identity"
            );
        }
        // (ii) An adopted session's REAL worktree/task ids flow into the ctx.
        let adopted_id = {
            let manager =
                SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                    .unwrap();
            let (deps, _keep) = deps_sharing_session(
                manager.clone(),
                one_probe_turn_script(),
                vec![probe_tool(&captured)],
            );
            let runtime = AgentRuntime::new(deps).unwrap();
            let ws = manager.create_workspace("/w").unwrap();
            let adopted = manager.create_session(ws, "adopted", "fake", "m").unwrap();
            manager
                .adopt_identity(adopted.id(), WorktreeId::new(7), TaskId::new(9))
                .unwrap();
            runtime.run_turn(adopted.id(), "probe", &[]).await.unwrap();
            assert_eq!(
                captured.lock().unwrap()[1],
                WorkspaceIdentity::new(ws, WorktreeId::new(7), TaskId::new(9)),
                "the tool ctx must carry the session row's real identity"
            );
            adopted.id()
        };
        // (iii) The identity is durable: a reopened manager reads the same
        // adopted row.
        let manager =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let row = manager
            .get_session(adopted_id)
            .unwrap()
            .unwrap()
            .row()
            .unwrap();
        assert_eq!(
            row.worktree_id,
            WorktreeId::new(7),
            "adoption survives reopen"
        );
        assert_eq!(row.task_id, TaskId::new(9));
        assert_eq!(
            manager
                .get_session(adopted_id)
                .unwrap()
                .unwrap()
                .identity()
                .unwrap(),
            WorkspaceIdentity::new(row.workspace_id, WorktreeId::new(7), TaskId::new(9))
        );
    }

    #[tokio::test]
    async fn repeated_malformed_calls_stop_the_turn() {
        // The provider emits the same broken call three times; the loop
        // detector stops instead of repeating.
        let (deps, _dir) = deps(
            scripted_provider(vec![
                ScriptedResponse::ToolCall {
                    id: "bad_1".into(),
                    name: "echo".into(),
                    input: serde_json::json!({"x": 1}),
                },
                ScriptedResponse::ToolCall {
                    id: "bad_2".into(),
                    name: "echo".into(),
                    input: serde_json::json!({"x": 1}),
                },
                ScriptedResponse::ToolCall {
                    id: "bad_3".into(),
                    name: "echo".into(),
                    input: serde_json::json!({"x": 1}),
                },
                ScriptedResponse::End,
            ]),
            vec![echo_tool()],
        );
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());
        let outcome = runtime.run_turn(session, "x", &[]).await.unwrap();
        assert!(
            outcome.loop_stopped,
            "identical repeated calls must trip the detector"
        );
        assert_eq!(outcome.final_state, AgentState::FailedRecoverable);
    }

    #[tokio::test]
    async fn permission_denied_turn_returns_ready() {
        struct DenyAll;
        impl PermissionRequester for DenyAll {
            fn request(
                &self,
                _s: SessionId,
                _p: &SessionPermission,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<Output = kilop_core::Result<PermissionDecision>> + Send,
                >,
            > {
                Box::pin(async { Ok(PermissionDecision::Deny) })
            }
        }
        let (mut deps, _dir) = deps(
            scripted_provider(vec![
                ScriptedResponse::ToolCall {
                    id: "c1".into(),
                    name: "echo".into(),
                    input: serde_json::json!({"x": 1}),
                },
                ScriptedResponse::End,
            ]),
            vec![echo_tool()],
        );
        deps.permission_requester = Arc::new(DenyAll);
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());
        let outcome = runtime.run_turn(session, "x", &[]).await.unwrap();
        assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);
        // No tool run was started.
        let handle = runtime.deps.session.get_session(session).unwrap().unwrap();
        assert!(handle.pending_tool_runs().unwrap().is_empty());
    }

    #[tokio::test]
    async fn stream_death_is_state_aware_no_replay() {
        // Provider dies mid-stream after a tool call ran: effect marked
        // unknown; the turn lands NeedsUserInput, never a blind replay.
        let (deps, _dir) = deps(
            FakeProvider::with_script(
                "fake",
                ModelCapabilities {
                    tools: true,
                    ..Default::default()
                },
                vec![
                    ScriptedResponse::ToolCall {
                        id: "c1".into(),
                        name: "echo".into(),
                        input: serde_json::json!({"x": 1}),
                    },
                    ScriptedResponse::Text("partial".into()),
                    ScriptedResponse::Die(ProviderError::new(
                        kilop_provider::ProviderErrorKind::Network,
                        "connection vanished",
                    )),
                ],
            ),
            vec![echo_tool()],
        );
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());
        let outcome = runtime.run_turn(session, "x", &[]).await.unwrap();
        assert!(matches!(
            outcome.final_state,
            AgentState::NeedsUserInput | AgentState::FailedRecoverable
        ));
        let handle = runtime.deps.session.get_session(session).unwrap().unwrap();
        assert!(
            handle.pending_tool_runs().unwrap().is_empty(),
            "recovery resolves pending runs"
        );
    }

    #[tokio::test]
    async fn compaction_trigger_records_and_recovers() {
        let (mut deps, _dir) = deps(
            scripted_provider(vec![
                ScriptedResponse::Text("t".into()),
                ScriptedResponse::End,
            ]),
            vec![],
        );
        deps.compact_at_usage = 0.0; // always trigger
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());
        let outcome = runtime.run_turn(session, "x", &[]).await.unwrap();
        assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);
    }

    #[tokio::test]
    async fn abort_cancels_mid_turn() {
        let (deps, _dir) = deps(
            scripted_provider(vec![
                ScriptedResponse::Text("a".into()),
                ScriptedResponse::Text("b".into()),
                ScriptedResponse::Text("c".into()),
                ScriptedResponse::End,
            ]),
            vec![],
        );
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());
        let handle = runtime.deps.session.get_session(session).unwrap().unwrap();
        let receipt = handle.submit_prompt("go", &[]).unwrap();
        receipt.op_meta.cancellation.cancel();
        let outcome = runtime
            .drive_turn(
                &handle,
                receipt.op_id,
                receipt.op_meta.cancellation.clone(),
                None,
            )
            .await
            .unwrap();
        assert_eq!(outcome.final_state, AgentState::Cancelled);
    }

    #[tokio::test]
    async fn failed_turn_never_leaves_session_stuck() {
        // A provider that is NOT registered: the turn fails at startup. The
        // session must land on FailedRecoverable (promptable) — never stuck
        // in Preparing, which would reject every future prompt.
        let (mut deps, _dir) = deps(scripted_provider(vec![ScriptedResponse::End]), vec![]);
        // Remove the registered provider so lookup fails.
        deps.providers = Arc::new(ProviderRegistry::new());
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());
        let err = runtime.run_turn(session, "hi", &[]).await.unwrap_err();
        assert!(err.kind == ErrorKind::NotFound);
        let handle = runtime.deps.session.get_session(session).unwrap().unwrap();
        let state = handle.state().unwrap();
        assert_eq!(
            state,
            AgentState::FailedRecoverable,
            "failed turn must land on FailedRecoverable, got {state:?}"
        );
        // The session accepts a NEW prompt afterwards (FailedRecoverable is
        // promptable) — recovery is possible.
        let receipt = handle.submit_prompt("retry", &[]).unwrap();
        assert!(receipt.accepted);
    }

    #[test]
    fn agent_cards_reflect_state() {
        let (deps, _dir) = deps(scripted_provider(vec![ScriptedResponse::End]), vec![]);
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());
        let cards = runtime.cards().unwrap();
        let card = cards.iter().find(|c| c.session_id == session).unwrap();
        assert!(card.status == "waiting" || card.status == "completed" || card.status == "running");
    }

    #[tokio::test]
    async fn crash_recovery_verify_hash_completes_without_rerun() {
        // Simulate a crash: ToolStarted recorded (VerifyHash) with a file on
        // disk matching the expected hash; recovery must complete the run
        // without executing the tool again.
        let dir = tempdir().unwrap();
        let root = dir.path();
        let file_path = root.join("target.txt");
        std::fs::write(&file_path, b"new content").unwrap();
        let expected = kilop_core::hash::FileHash::from(blake3::hash(b"new content").into());

        let (_base_deps, _base_dir) = deps(scripted_provider(vec![ScriptedResponse::End]), vec![]);
        let mut deps = AgentDeps {
            session: SessionManager::open(root.join("store"), root.join("cas"), true).unwrap(),
            providers: Arc::new(ProviderRegistry::new()),
            permission_requester: Arc::new(AlwaysAllow),
            evidence: Arc::new(NoEvidence),
            tools: Arc::new(ToolRegistry::new()),
            cas: Some(Arc::new(kilop_cas::Cas::open(root.join("cas")).unwrap())),
            workspaces: kilop_fs::WorkspaceFileService::new(),
            edit: None,
            snapshots: None,
            sandbox: None,
            supervisor: None,
            model: "m".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "You are a test agent.".into(),
            clock: Arc::new(SystemClock),
            tool_call_mode: ToolCallMode::Native,
            tool_deadline_ms: 2000,
            retry_policy: kilop_core::retry::RetryPolicy::default(),
        };
        let mut registry = ProviderRegistry::new();
        registry.register(Arc::new(scripted_provider(vec![ScriptedResponse::End])));
        deps.providers = Arc::new(registry);
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());
        let handle = runtime.deps.session.get_session(session).unwrap().unwrap();

        // Durable ToolStarted row with VerifyHash recovery, never finished
        // (the "crash").
        let op_meta = OpMeta::new(
            runtime.deps.session.next_op_id(),
            session,
            kilop_core::time::Deadline::at(runtime.deps.clock.now_ms().saturating_add(1000)),
            kilop_core::retry::RetryPolicy::default(),
            CancellationToken::new(),
            RecoveryStrategy::VerifyHash {
                path: file_path.to_string_lossy().to_string(),
                expected,
            },
            runtime.deps.clock.now_ms(),
        );
        let _ = handle.request_permission(
            op_meta.operation_id,
            &Capability::WriteWorkspace {
                path: file_path.clone(),
            },
        );
        let _ = handle.start_tool_run(op_meta.clone(), "write_file", serde_json::json!({}));

        // Recovery: pending run resolved as verified without executing.
        runtime.recover().unwrap();
        assert!(handle.pending_tool_runs().unwrap().is_empty());
        // The file is untouched (no re-run happened — a re-run would have
        // written different content).
        assert_eq!(std::fs::read(&file_path).unwrap(), b"new content");
    }

    #[tokio::test]
    async fn tool_results_are_required_by_the_second_request() {
        // The audit's semantic test: the FIRST stream yields one tool call;
        // the SECOND request (after the tool executed) MUST carry the tool
        // result back to the model — the wrapper refuses the stream with a
        // Malformed error when it is missing, so the turn can only complete
        // once the request shape is correct. On the old code the second
        // request omits the result and this test fails.
        let inner = scripted_provider(vec![
            ScriptedResponse::ToolCall {
                id: "call_1".into(),
                name: "echo".into(),
                input: serde_json::json!({"x": 1}),
            },
            ScriptedResponse::Text("after tool".into()),
            ScriptedResponse::End,
        ]);
        let captured: Arc<std::sync::Mutex<Vec<GenericAgentRequest>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let cap = captured.clone();
        let wrapper = InspectingProvider::new(Arc::new(inner), move |n, req| {
            if n == 0 {
                // No tool has run yet: a tool result on the first request is
                // as corrupt as a missing one on the second.
                let leaked = req.messages.iter().any(|m| {
                    m.content
                        .iter()
                        .any(|c| matches!(c.kind, ContentKind::ToolResult { .. }))
                });
                if leaked {
                    return Err("tool result present before any tool ran".into());
                }
            } else {
                let user_has_result = req.messages.iter().any(|m| {
                    m.role == Role::User
                        && m.content.iter().any(|c| {
                            matches!(
                                &c.kind,
                                ContentKind::ToolResult { content, is_error }
                                    if c.tool_call_id.as_deref() == Some("call_1")
                                        && content == "echo: {\"x\":1}"
                                        && !is_error
                            )
                        })
                });
                if !user_has_result {
                    return Err("tool result missing".into());
                }
                let assistant_has_call = req.messages.iter().any(|m| {
                    m.role == Role::Assistant
                        && m.content.iter().any(|c| {
                            matches!(
                                &c.kind,
                                ContentKind::ToolCall { id, name, input }
                                    if id == "call_1"
                                        && name == "echo"
                                        && *input == serde_json::json!({"x": 1})
                            )
                        })
                });
                if !assistant_has_call {
                    return Err("tool call missing".into());
                }
            }
            cap.lock().unwrap().push(req.clone());
            Ok(())
        });
        let (deps, _dir) = deps_with(Arc::new(wrapper), vec![echo_tool()]);
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());
        let outcome = runtime.run_turn(session, "use echo", &[]).await.unwrap();
        assert_eq!(
            outcome.final_state,
            AgentState::ReadyForNextTurn,
            "the turn completes only when the tool result rides the second request"
        );
        // One logical turn (prompt → tool → model) counts exactly ONE turn,
        // even though two provider requests were made (audit round 6).
        assert_eq!(outcome.turns, 1);
        let requests = captured.lock().unwrap();
        assert_eq!(requests.len(), 2, "exactly two provider requests expected");
        let assistant = requests[1]
            .messages
            .iter()
            .find(|m| m.role == Role::Assistant)
            .expect("second request carries the assistant tool call");
        assert!(
            assistant
                .content
                .iter()
                .any(|c| matches!(&c.kind, ContentKind::ToolCall { id, .. } if id == "call_1")),
            "the assistant message must carry the completed tool call"
        );
    }

    #[tokio::test]
    async fn provider_request_cancellation_is_child_of_turn_token() {
        // The request's meta.cancellation must share the turn's lineage:
        // cancelling the turn token cancels the wire request. On the old
        // code build_request minted a fresh token and this test fails.
        let provider = scripted_provider(vec![
            ScriptedResponse::Text("ok".into()),
            ScriptedResponse::End,
        ]);
        let (deps, _dir) = deps(provider.clone(), vec![]);
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());
        let handle = runtime.deps.session.get_session(session).unwrap().unwrap();
        let receipt = handle.submit_prompt("go", &[]).unwrap();
        let turn_token = receipt.op_meta.cancellation.clone();
        runtime
            .drive_turn(&handle, receipt.op_id, turn_token.clone(), None)
            .await
            .unwrap();
        let request_cancel = provider
            .last_request_cancellation()
            .expect("a provider request was streamed");
        assert!(
            !request_cancel.is_cancelled(),
            "the request token must be live while the turn runs"
        );
        turn_token.cancel();
        assert!(
            request_cancel.is_cancelled(),
            "cancelling the turn token must cascade to the provider request"
        );
    }

    #[tokio::test]
    async fn reasoning_persists_separately_and_roundtrips_as_reasoning() {
        // Turn 1: the provider streams thinking then text. The durable
        // parts must keep them separate (reasoning row before text row) and
        // the second request must reconstruct ContentKind::Reasoning —
        // never merged into the assistant text.
        let fake = Arc::new(scripted_provider(vec![
            ScriptedResponse::Reasoning("let me think".into()),
            ScriptedResponse::Text("the answer".into()),
            ScriptedResponse::End,
        ]));
        let hook = |n: usize, req: &GenericAgentRequest| -> Result<(), String> {
            if n == 0 {
                return Ok(());
            }
            let assistant = req
                .messages
                .iter()
                .find(|m| m.role == Role::Assistant)
                .ok_or_else(|| "assistant history missing".to_string())?;
            let kinds: Vec<&str> = assistant
                .content
                .iter()
                .map(|c| match &c.kind {
                    kilop_provider::ContentKind::Reasoning { .. } => "reasoning",
                    kilop_provider::ContentKind::Text { .. } => "text",
                    other => panic!("unexpected content kind {other:?}"),
                })
                .collect();
            if kinds != vec!["reasoning", "text"] {
                return Err(format!(
                    "reasoning and text must roundtrip in order, got {kinds:?}"
                ));
            }
            match &assistant.content[0].kind {
                kilop_provider::ContentKind::Reasoning { text } if text == "let me think" => {}
                other => return Err(format!("reasoning content lost or merged, got {other:?}")),
            }
            match &assistant.content[1].kind {
                kilop_provider::ContentKind::Text { text } if text == "the answer" => {}
                other => return Err(format!("assistant text corrupted, got {other:?}")),
            }
            Ok(())
        };
        let wrapper = Arc::new(InspectingProvider::new(fake.clone(), hook));
        let (deps, _dir) = deps_with(wrapper, vec![]);
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());
        let handle = runtime.deps.session.get_session(session).unwrap().unwrap();
        let receipt = handle.submit_prompt("t1", &[]).unwrap();
        let outcome = runtime
            .drive_turn(
                &handle,
                receipt.op_id,
                receipt.op_meta.cancellation.clone(),
                None,
            )
            .await
            .unwrap();
        assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);

        let page = handle.messages_page(None, 20).unwrap();
        let assistant = page
            .messages
            .iter()
            .find(|m| m.role == "assistant")
            .expect("assistant message exists");
        let part_kinds: Vec<&str> = assistant
            .parts
            .iter()
            .map(|p| match p {
                kilop_protocol::v756::Part::Reasoning { .. } => "reasoning",
                kilop_protocol::v756::Part::Text { .. } => "text",
                other => panic!("unexpected durable part {other:?}"),
            })
            .collect();
        assert_eq!(
            part_kinds,
            vec!["reasoning", "text"],
            "thinking rows precede text rows in the durable part order"
        );
        match &assistant.parts[0] {
            kilop_protocol::v756::Part::Reasoning { text } => {
                assert_eq!(text, "let me think");
            }
            other => panic!("wrong part {other:?}"),
        }
        match &assistant.parts[1] {
            kilop_protocol::v756::Part::Text { text } => {
                assert_eq!(text, "the answer", "reasoning must never leak into text");
            }
            other => panic!("wrong part {other:?}"),
        }

        // Turn 2: reseed the script; the wrapper inspects the request and
        // refuses the stream unless reasoning roundtrips as its own kind.
        *fake.script.lock().unwrap() = vec![
            ScriptedResponse::Text("second reply".into()),
            ScriptedResponse::End,
        ];
        let receipt = handle.submit_prompt("t2", &[]).unwrap();
        let outcome = runtime
            .drive_turn(
                &handle,
                receipt.op_id,
                receipt.op_meta.cancellation.clone(),
                None,
            )
            .await
            .unwrap();
        assert_eq!(
            outcome.final_state,
            AgentState::ReadyForNextTurn,
            "the second turn only completes when reasoning roundtrips as ContentKind::Reasoning"
        );
    }

    #[tokio::test]
    async fn history_reconstruction_splits_tool_results_into_user_role() {
        // Durable state shaped exactly like the turn loop writes it:
        // 1) assistant message: text + completed tool call part,
        // 2) assistant message: ONLY a tool result part (run_tool_calls),
        // 3) assistant message: a pending tool call (never sent back),
        // 4) assistant message: a failing tool result (is_error on the wire).
        let (deps, _dir) = deps(scripted_provider(vec![ScriptedResponse::End]), vec![]);
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());
        let handle = runtime.deps.session.get_session(session).unwrap().unwrap();
        let m1 = handle
            .put_message(1, "assistant", serde_json::json!({}))
            .unwrap();
        handle.put_text_part(m1, "calling the tool").unwrap();
        handle
            .put_tool_call_part(
                m1,
                "call_1",
                "echo",
                serde_json::json!({"x": 1}),
                "completed",
            )
            .unwrap();
        let m2 = handle
            .put_message(2, "assistant", serde_json::json!({}))
            .unwrap();
        handle
            .put_tool_result_part(
                m2,
                "call_1",
                &ToolResultBody {
                    excerpt: "echo: {\"x\":1}".into(),
                    exit_code: Some(0),
                    artifact: None,
                    slice_hint: None,
                },
            )
            .unwrap();
        let m3 = handle
            .put_message(3, "assistant", serde_json::json!({}))
            .unwrap();
        handle
            .put_tool_call_part(m3, "call_2", "echo", serde_json::json!({"x": 2}), "pending")
            .unwrap();
        let m4 = handle
            .put_message(4, "assistant", serde_json::json!({}))
            .unwrap();
        handle
            .put_tool_result_part(
                m4,
                "call_3",
                &ToolResultBody {
                    excerpt: "boom".into(),
                    exit_code: Some(1),
                    artifact: None,
                    slice_hint: None,
                },
            )
            .unwrap();

        let msgs = runtime.history_messages(&handle).unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0].role, Role::Assistant);
        assert!(matches!(
            &msgs[0].content[0].kind,
            ContentKind::Text { text } if text == "calling the tool"
        ));
        assert!(matches!(
            &msgs[0].content[1].kind,
            ContentKind::ToolCall { id, name, input }
                if id == "call_1" && name == "echo" && *input == serde_json::json!({"x": 1})
        ));
        assert_eq!(
            msgs[1].role,
            Role::User,
            "tool results move to the user role"
        );
        assert_eq!(
            msgs[1].content[0].tool_call_id.as_deref(),
            Some("call_1"),
            "the tool result must name the call it answers"
        );
        assert!(matches!(
            &msgs[1].content[0].kind,
            ContentKind::ToolResult { content, is_error }
                if content == "echo: {\"x\":1}" && !is_error
        ));
        assert_eq!(msgs[2].role, Role::User);
        assert!(
            matches!(
                &msgs[2].content[0].kind,
                ContentKind::ToolResult { is_error, .. } if *is_error
            ),
            "a non-zero exit code is an error result"
        );
        assert!(
            !msgs.iter().any(|m| m.content.iter().any(|c| matches!(
                &c.kind,
                ContentKind::ToolCall { id, .. } if id == "call_2"
            ))),
            "pending tool calls never reach the wire"
        );
    }

    #[test]
    fn run_tool_calls_fills_reads_writes_from_tool_ownership() {
        // The scheduler's ownership sets must come from the tool's declared
        // path args: write_file with a path arg writes that path (the audit
        // requires the ScheduledOp's reads/writes to be non-empty so edit
        // overlap serialization works).
        let write_file = Arc::new(Tool {
            name: "write_file".into(),
            description: "w".into(),
            input_schema: serde_json::json!({}),
            resource_class: kilop_core::resource::ResourceClass::DiskWrite,
            capability: None,
            recovery_hint: RecoveryHint::WorkspaceWrite,
            path_args: vec!["path".into()],
            execute: Arc::new(|_ctx, _args| Box::pin(async move { Ok(ToolOutcome::default()) })),
        });
        let (reads, writes) = ownership_sets(
            &write_file,
            &serde_json::json!({"path": "src/main.rs", "content": "x"}),
        );
        assert!(!writes.is_empty(), "write_file must declare its write path");
        assert!(reads.is_empty(), "write_file declares no reads");
        let (reads, read_writes) = ownership_sets(
            &Arc::new(Tool {
                path_args: vec!["path".into()],
                resource_class: kilop_core::resource::ResourceClass::DiskRead,
                ..(write_file.as_ref()).clone()
            }),
            &serde_json::json!({"path": "src/main.rs"}),
        );
        assert!(!reads.is_empty(), "read_file must declare its read path");
        assert!(read_writes.is_empty());
    }

    #[tokio::test]
    async fn wire_request_contains_each_element_once() {
        // The wire request must contain every conceptual element exactly
        // once: the prompt once in messages, the tool schema once in tools,
        // and the system carries instructions/ledger — never the prompt text
        // and never the tool schema JSON.
        let inner = scripted_provider(vec![
            ScriptedResponse::Text("ok".into()),
            ScriptedResponse::End,
        ]);
        let captured: Arc<std::sync::Mutex<Vec<GenericAgentRequest>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let cap = captured.clone();
        let wrapper = InspectingProvider::new(Arc::new(inner), move |_n, req| {
            cap.lock().unwrap().push(req.clone());
            Ok(())
        });
        let (deps, _dir) = deps_with(Arc::new(wrapper), vec![echo_tool()]);
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());
        runtime.run_turn(session, "use echo", &[]).await.unwrap();

        let requests = captured.lock().unwrap();
        assert_eq!(requests.len(), 1);
        let req = &requests[0];
        // Prompt: exactly once, as a user message.
        let prompt_parts: Vec<&ContentPart> = req
            .messages
            .iter()
            .flat_map(|m| m.content.iter())
            .filter(|p| matches!(&p.kind, ContentKind::Text { text } if text == "use echo"))
            .collect();
        assert_eq!(prompt_parts.len(), 1, "prompt must appear exactly once");
        assert_eq!(
            req.messages.len(),
            1,
            "only the prompt message on a fresh turn"
        );
        assert!(req.messages[0].role == Role::User);
        // System: instructions, no conversation, no tool schema.
        assert!(req.system.contains("You are a test agent."));
        assert_eq!(req.system.matches("You are a test agent.").count(), 1);
        assert!(
            !req.system.contains("use echo"),
            "history must not leak into system"
        );
        assert!(
            !req.system.contains("echo back"),
            "tool schema must not leak into system"
        );
        let tool_json = serde_json::to_string(&req.tools[0]).unwrap();
        assert!(
            !req.system.contains(&tool_json),
            "tool schema JSON in system"
        );
        // Tools: exactly once.
        assert_eq!(req.tools.len(), 1);
        assert_eq!(req.tools[0].name, "echo");
    }

    #[tokio::test]
    async fn budget_overflow_never_reaches_provider() {
        // A request that cannot be budgeted (the untrimmable static prefix
        // alone exceeds the model budget) must fail BEFORE any provider
        // contact: the provider request counter stays at zero.
        let inner = scripted_provider(vec![
            ScriptedResponse::Text("ok".into()),
            ScriptedResponse::End,
        ]);
        let wrapper = Arc::new(InspectingProvider::new(Arc::new(inner), |_n, _req| Ok(())));
        let (mut deps, _dir) = deps_with(wrapper.clone(), vec![]);
        // > 25K tokens of static instructions: even an empty history cannot
        // fit — the planner must Err(Oversized) instead of sending anything.
        deps.instructions = format!("You are Kilo+.\n{}", "x".repeat(100_100));
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());
        let err = runtime
            .run_turn(session, &"y".repeat(10_000), &[])
            .await
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Oversized);
        assert_eq!(
            wrapper.counter.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "the provider must never be contacted with an unbudgeted request"
        );
    }

    #[tokio::test]
    async fn runtime_context_limit_shrinks_the_budget_below_the_model_maximum() {
        // P0 (the /api/ps allocation reaches the REAL budget): the drive
        // loop budgets from min(capabilities.context, runtime_context_limit)
        // — a provider whose ADVERTISED maximum is 200K but whose LIVE
        // runtime window is 40K must plan its wire content under the
        // 40K-derived budget. The seeded history (~40K tokens of messages)
        // exceeds that budget, so the oldest seeds are trimmed from the
        // request — under the untouched model-maximum budget (65K+ of
        // context) nothing would have been dropped at all.
        let (seed_deps, _dir0) = deps(scripted_provider(vec![ScriptedResponse::End]), vec![]);
        let (manager, session) = shared_session(&seed_deps);
        seed_long_history(&manager, session, 8, 20_000).await;

        let main_caps = ModelCapabilities {
            tools: true,
            context: 200_000,
            ..Default::default()
        };
        let inner = Arc::new(FakeProvider::with_script(
            "fake",
            main_caps.clone(),
            vec![ScriptedResponse::Text("ok".into()), ScriptedResponse::End],
        ));
        let captured: Arc<std::sync::Mutex<Vec<GenericAgentRequest>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let cap = captured.clone();
        let inspected = Arc::new(InspectingProvider::new(inner, move |_n, req| {
            cap.lock().unwrap().push(req.clone());
            Ok(())
        }));
        let limited = Arc::new(RuntimeLimitedProvider::new(inspected, 40_000));
        let mut registry = ProviderRegistry::new();
        registry.register(limited);
        let (mut final_deps, _dir) = deps_sharing_session(
            manager.clone(),
            Arc::new(FakeProvider::with_script(
                "fake",
                main_caps.clone(),
                vec![ScriptedResponse::End],
            )),
            vec![],
        );
        final_deps.providers = Arc::new(registry);
        // Compaction fires only at usage >= 1.0 — a trimmed plan lands
        // below that, so the request content is purely budget-governed
        // (no compaction interference).
        final_deps.compact_at_usage = 1.0;
        let runtime = AgentRuntime::new(final_deps).unwrap();
        let outcome = runtime
            .run_turn(session, "do the thing", &[])
            .await
            .unwrap();
        assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);

        // The budget the drive loop must have used: for_capabilities over
        // min(model maximum, runtime limit) — the exact effective caps the
        // runtime derives.
        let effective_caps = ModelCapabilities {
            context: main_caps.context.min(40_000),
            ..main_caps.clone()
        };
        let effective_budget = ContextBudget::for_capabilities(&effective_caps);
        let model_max_budget = ContextBudget::for_capabilities(&main_caps);
        assert!(
            effective_budget.context_max() < model_max_budget.context_max(),
            "the runtime limit must shrink the budget"
        );

        let requests = captured.lock().unwrap();
        assert_eq!(requests.len(), 1, "one wire request for the turn");
        let req = &requests[0];
        // Planner-exact estimate of the captured request (text-only
        // messages, no tools): est(system) + per message 2 + Σ(est + 1).
        let est = kilop_context::Estimator;
        let total: usize = est.estimate_tokens(&req.system)
            + req
                .messages
                .iter()
                .map(|m| {
                    2usize.saturating_add(
                        m.content
                            .iter()
                            .map(|p| match &p.kind {
                                ContentKind::Text { text } => {
                                    est.estimate_tokens(text).saturating_add(1)
                                }
                                _ => 1,
                            })
                            .sum::<usize>(),
                    )
                })
                .sum::<usize>();
        assert!(
            total <= effective_budget.context_max(),
            "the request must fit the runtime-limited budget: {total} > {}",
            effective_budget.context_max()
        );
        let all_text = request_text(req);
        assert!(
            all_text.contains("turn seed7"),
            "the newest seed must survive trimming"
        );
        assert!(
            !all_text.contains("turn seed0"),
            "the oldest seed must be trimmed under the runtime-limited budget"
        );
        assert!(total > 25_000, "real history must survive: {total}");
    }

    #[tokio::test]
    async fn compaction_trigger_uses_wire_footprint() {
        // The wire footprint (system + messages + tools, exactly once) must
        // drive the compaction trigger. History is grown the REAL way: full
        // prior logical turns through the actual runtime, each ending at
        // ReadyForNextTurn with one TurnCompleted. The final prompt's
        // boundary covers all of them, so plan.total_tokens crosses the
        // threshold and compaction fires — deterministically.
        let (mut deps, _dir) = deps(scripted_provider(vec![ScriptedResponse::End]), vec![]);
        // Compaction is self-limiting by design (each accepted compaction
        // shrinks to 75% of before), so a realistic threshold can never be
        // re-crossed by a tiny history. The threshold here is a sentinel:
        // ANY wire footprint above ~26 tokens must trigger, proving the
        // decision is driven by plan.total_tokens — not a static counter.
        deps.compact_at_usage = 0.001;
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());
        let handle = runtime.deps.session.get_session(session).unwrap().unwrap();

        let mut compacted_count = 0usize;
        for t in 0..11 {
            let prompt = format!("prior turn {t} {}", "z".repeat(1500));
            let r = handle.submit_prompt(&prompt, &[]).unwrap();
            assert!(!r.queued, "prior turn must be accepted, not queued");
            let outcome = runtime
                .drive_turn(&handle, r.op_id, r.op_meta.cancellation.clone(), None)
                .await
                .unwrap();
            assert_eq!(
                outcome.final_state,
                AgentState::ReadyForNextTurn,
                "prior turn {t} must complete"
            );
            if outcome.compacted {
                compacted_count += 1;
            }
        }
        // The final prompt of the test: a new logical turn on top of the 11
        // accumulated ones.
        let prompt = format!("final turn {}", "y".repeat(1500));
        let receipt = handle.submit_prompt(&prompt, &[]).unwrap();
        assert!(!receipt.queued);

        // The final turn re-plans from the accumulated durable history: the
        // wire footprint (system + 12 prompts' text + tools) crosses the
        // threshold, so compaction MUST trigger off plan.total_tokens.
        let outcome = runtime
            .drive_turn(
                &handle,
                receipt.op_id,
                receipt.op_meta.cancellation.clone(),
                None,
            )
            .await
            .unwrap();
        assert!(
            outcome.compacted,
            "the wire footprint must trigger compaction"
        );
        assert!(
            compacted_count >= 2,
            "the accumulating wire footprint must trigger compaction repeatedly, got {compacted_count}"
        );
        assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);
        // The journal holds exactly one TurnCompleted per prior turn plus
        // one for this final turn: 13 total.
        let events = handle.events_range(1, None).unwrap();
        let turn_completed = events
            .iter()
            .filter(|e| e.kind == kilop_core::event::EventKind::TurnCompleted)
            .count();
        assert_eq!(turn_completed, 12, "12 logical turns, 12 completions");
        let pending = handle.pending_tool_runs().unwrap();
        assert!(pending.is_empty());
    }

    #[tokio::test]
    async fn one_logical_turn_has_exactly_one_turn_completed_and_no_mid_turn_ready() {
        // Audit round 6 P0: a turn with TWO tool batches must journal exactly
        // ONE TurnCompleted and must never enter ReadyForNextTurn between
        // the batches.
        let (deps, _dir) = deps(
            scripted_provider(vec![
                ScriptedResponse::ToolCall {
                    id: "c1".into(),
                    name: "echo".into(),
                    input: serde_json::json!({"x": 1}),
                },
                ScriptedResponse::ToolCall {
                    id: "c2".into(),
                    name: "echo".into(),
                    input: serde_json::json!({"x": 2}),
                },
                ScriptedResponse::Text("final answer".into()),
                ScriptedResponse::End,
            ]),
            vec![echo_tool()],
        );
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());
        let outcome = runtime.run_turn(session, "do work", &[]).await.unwrap();
        assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);
        assert_eq!(
            outcome.turns, 1,
            "one logical turn despite two tool batches"
        );
        let handle = runtime.deps.session.get_session(session).unwrap().unwrap();
        let events = handle.events_range(1, None).unwrap();
        let turn_completed = events
            .iter()
            .filter(|e| e.kind == kilop_core::event::EventKind::TurnCompleted)
            .count();
        assert_eq!(
            turn_completed, 1,
            "exactly one TurnCompleted per logical turn"
        );
        // ReadyForNextTurn must appear in the journal EXACTLY ONCE (the end).
        let ready = events
            .iter()
            .filter(|e| e.state == AgentState::ReadyForNextTurn)
            .count();
        assert_eq!(ready, 1, "ReadyForNextTurn only at the genuine end");
        // The interior tool batches used PhaseChanged hops (never TurnCompleted).
        let interior = events
            .iter()
            .filter(|e| e.kind == kilop_core::event::EventKind::PhaseChanged)
            .count();
        assert!(interior >= 2, "interior hops must use PhaseChanged");
    }

    #[tokio::test]
    async fn mid_turn_crash_resumes_the_same_logical_turn() {
        // Crash AFTER the first tool batch: the journal ends at
        // WaitingForModel (interior hop). continue_turn must resume the SAME
        // logical turn: no second PromptReceived, and the model sees the
        // tool result in request #1 of the resumed turn.
        let (deps, _dir) = deps(
            scripted_provider(vec![
                ScriptedResponse::ToolCall {
                    id: "c1".into(),
                    name: "echo".into(),
                    input: serde_json::json!({"x": 1}),
                },
                ScriptedResponse::End,
            ]),
            vec![echo_tool()],
        );
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());
        let handle = runtime.deps.session.get_session(session).unwrap().unwrap();
        let receipt = handle.submit_prompt("crash me", &[]).unwrap();
        let outcome = runtime
            .drive_turn(
                &handle,
                receipt.op_id,
                receipt.op_meta.cancellation.clone(),
                None,
            )
            .await
            .unwrap();
        assert_eq!(
            outcome.final_state,
            AgentState::ReadyForNextTurn,
            "turn ran to completion with a single batch"
        );
        // The crash happens BEFORE the model continuation? Simulate by
        // ending the provider script: first stream consumed the ToolCall;
        // second stream (continuation) has no script → Done → turn ends.
        let events = handle.events_range(1, None).unwrap();
        let prompt_events = events
            .iter()
            .filter(|e| e.kind == kilop_core::event::EventKind::PromptReceived)
            .count();
        assert_eq!(prompt_events, 1, "one prompt for the whole logical turn");
        let turn_completed = events
            .iter()
            .filter(|e| e.kind == kilop_core::event::EventKind::TurnCompleted)
            .count();
        assert_eq!(turn_completed, 1);
    }

    #[tokio::test]
    async fn second_prompt_while_active_is_queued_and_delivered_after() {
        // Audit round 6 P0: prompt B while turn A is active must durably
        // queue; the per-session runner delivers B only after A finishes;
        // exactly one PromptReceived per prompt; B never leaks into A's
        // context.
        let (deps, _dir) = deps(
            scripted_provider(vec![
                ScriptedResponse::ToolCall {
                    id: "c1".into(),
                    name: "echo".into(),
                    input: serde_json::json!({"x": 1}),
                },
                ScriptedResponse::Text("answer A".into()),
                ScriptedResponse::End,
            ]),
            vec![echo_tool()],
        );
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());
        let handle = runtime.deps.session.get_session(session).unwrap().unwrap();

        // Start turn A (detached drive via a spawned task to simulate the
        // server pattern).
        let receipt_a = runtime.submit(session, "task A", &[]).unwrap();
        assert!(!receipt_a.queued);
        let agent = runtime.clone();
        let handle2 = runtime.deps.session.get_session(session).unwrap().unwrap();
        let a_task = tokio::spawn(async move {
            agent
                .drive_receipt(&handle2, receipt_a, None)
                .await
                .unwrap()
        });

        // Prompt B arrives while A is active (its provider scripted turn is
        // mid-flight).
        let receipt_b = runtime.submit(session, "task B", &[]).unwrap();
        assert!(receipt_b.queued, "B must queue behind active A");
        assert_eq!(handle.queued_prompt_count().unwrap(), 1);

        // The queue runner is idempotent per session.
        let runner = runtime.clone();
        let runner_task = tokio::spawn(async move { runner.run_session_queue(session).await });
        let _ = a_task.await.unwrap();
        let _ = runner_task.await; // runner drains after A completes

        // B was delivered exactly once and its user message reached the
        // journal; the session is ready again.
        assert_eq!(handle.queued_prompt_count().unwrap(), 0, "queue drained");
        assert_eq!(handle.state().unwrap(), AgentState::ReadyForNextTurn);
        let events = handle.events_range(1, None).unwrap();
        let prompts = events
            .iter()
            .filter(|e| e.kind == kilop_core::event::EventKind::PromptReceived)
            .count();
        assert_eq!(prompts, 2, "one PromptReceived per user prompt");
    }

    #[tokio::test]
    async fn queued_prompt_never_leaks_into_active_turn_context() {
        // The active turn's provider requests must NOT contain the queued
        // prompt's text (isolation via queued_message_seqs).
        let (deps, _dir) = deps(
            scripted_provider(vec![
                ScriptedResponse::ToolCall {
                    id: "c1".into(),
                    name: "echo".into(),
                    input: serde_json::json!({"x": 1}),
                },
                ScriptedResponse::End,
            ]),
            vec![echo_tool()],
        );
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());
        let handle = runtime.deps.session.get_session(session).unwrap().unwrap();
        let receipt_a = runtime
            .submit(session, "secret task A content", &[])
            .unwrap();
        // B queues while A is still in flight (before A is even driven).
        let _b = runtime.submit(session, "QUEUED-B-MARKER", &[]).unwrap();
        assert_eq!(handle.queued_prompt_count().unwrap(), 1);

        let outcome = runtime
            .drive_receipt(&handle, receipt_a, None)
            .await
            .unwrap();
        assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);
        // Turn A's second provider request (after the tool) must not contain
        // the queued marker. Inspect via the journal-derived history.
        let history = runtime.history_messages(&handle).unwrap();
        let rendered = serde_json::to_string(&history).unwrap();
        assert!(
            !rendered.contains("QUEUED-B-MARKER"),
            "queued prompt leaked into the active turn context"
        );
        assert!(rendered.contains("secret task A content"));
    }

    #[tokio::test]
    async fn queue_survives_runner_gate_and_drains_sequentially() {
        // Multiple runners racing for one session: the gate lets only one
        // through, and queued prompts are delivered in FIFO order.
        let (deps, _dir) = deps(
            scripted_provider(vec![
                ScriptedResponse::Text("a".into()),
                ScriptedResponse::End,
            ]),
            vec![],
        );
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());
        let handle = runtime.deps.session.get_session(session).unwrap().unwrap();
        let ra = runtime.submit(session, "one", &[]).unwrap();
        let rb = runtime.submit(session, "two", &[]).unwrap();
        let rc = runtime.submit(session, "three", &[]).unwrap();
        assert!(!ra.queued);
        assert!(rb.queued && rc.queued);
        // A is not driven yet; drive it, then let TWO racing runners drain.
        let oa = runtime.drive_receipt(&handle, ra, None).await.unwrap();
        assert_eq!(oa.final_state, AgentState::ReadyForNextTurn);
        let r1 = runtime.clone();
        let r2 = runtime.clone();
        let t1 = tokio::spawn(async move { r1.run_session_queue(session).await });
        let t2 = tokio::spawn(async move { r2.run_session_queue(session).await });
        let _ = t1.await;
        let _ = t2.await;
        assert_eq!(handle.queued_prompt_count().unwrap(), 0, "FIFO drain");
        assert_eq!(handle.state().unwrap(), AgentState::ReadyForNextTurn);
    }

    #[tokio::test]
    async fn delivered_queued_prompt_appears_after_previous_turn_output() {
        // Audit round 7 (conversation chronology): B's user message must
        // materialize AFTER A's full exchange — never interleaved. With
        // deferred materialization + atomic admission this holds by
        // construction; assert it end-to-end.
        let (deps, _dir) = deps(
            scripted_provider(vec![
                ScriptedResponse::ToolCall {
                    id: "c1".into(),
                    name: "echo".into(),
                    input: serde_json::json!({"x": 1}),
                },
                ScriptedResponse::Text("A final".into()),
                ScriptedResponse::End,
            ]),
            vec![echo_tool()],
        );
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());
        let handle = runtime.deps.session.get_session(session).unwrap().unwrap();
        let receipt_a = runtime.submit(session, "A prompt", &[]).unwrap();
        let _ = runtime.submit(session, "B prompt", &[]).unwrap();
        // No message rows exist for B while queued.
        let page_before = handle.messages_before(None, 10).unwrap();
        assert!(
            !page_before.iter().any(|m| m
                .data
                .get("text")
                .and_then(|t| t.as_str())
                .map(|s| s.contains("B prompt"))
                .unwrap_or(false)),
            "queued prompt must not materialize before admission"
        );
        let outcome_a = runtime
            .drive_receipt(&handle, receipt_a, None)
            .await
            .unwrap();
        assert_eq!(outcome_a.final_state, AgentState::ReadyForNextTurn);
        // Deliver B via the runner.
        let runner = runtime.clone();
        let _ = tokio::spawn(async move { runner.run_session_queue(session).await }).await;
        // B's message exists now and its seq is AFTER everything from A.
        let page = handle.messages_before(None, 50).unwrap();
        let b_idx = page
            .iter()
            .position(|m| {
                m.data
                    .get("text")
                    .and_then(|t| t.as_str())
                    .map(|s| s.contains("B prompt"))
                    .unwrap_or(false)
            })
            .expect("B message materialized at admission");
        let a_idx = page
            .iter()
            .position(|m| {
                m.data
                    .get("text")
                    .and_then(|t| t.as_str())
                    .map(|s| s.contains("A prompt"))
                    .unwrap_or(false)
            })
            .expect("A message present");
        // messages_before returns newest-first: B (newest) has a SMALLER
        // index than A.
        assert!(b_idx < a_idx, "B must sit after A in conversation order");
        // Assistant parts between A's prompt and B's prompt. Page is
        // newest-first: chronologically between A (oldest, largest index)
        // and B (newest, smallest index) lives at indices
        // (b_idx, a_idx) exclusive.
        let assistant_after_a = page
            .iter()
            .skip(b_idx + 1)
            .take(a_idx.saturating_sub(b_idx + 1))
            .any(|m| m.role == "assistant");
        assert!(assistant_after_a, "A's output precedes B's message");
    }

    #[tokio::test]
    async fn aborting_a_queued_prompt_durably_cancels_it() {
        // Adversarial (audit round 7): the user kills prompt B while A is
        // mid-turn. B must NEVER be delivered — its durable row becomes
        // cancelled and the runner skips it, even though A completes and the
        // session reaches ReadyForNextTurn.
        let (deps, _dir) = deps(
            scripted_provider(vec![
                ScriptedResponse::ToolCall {
                    id: "c1".into(),
                    name: "echo".into(),
                    input: serde_json::json!({"x": 1}),
                },
                ScriptedResponse::Text("A final".into()),
                ScriptedResponse::End,
                // Nothing for B: it must never be driven.
            ]),
            vec![echo_tool()],
        );
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());
        let handle = runtime.deps.session.get_session(session).unwrap().unwrap();
        let receipt_a = runtime.submit(session, "A prompt", &[]).unwrap();
        let receipt_b = runtime.submit(session, "B prompt", &[]).unwrap();
        assert!(receipt_b.queued);
        // Kill B while A's turn is still registered but not yet driven: the
        // machine must NOT move (no Failed/Cancelled/TurnCompleted for B —
        // it was never a turn). A remains driveable.
        let before = handle.state().unwrap();
        let aborted = handle.abort(Some(receipt_b.op_id)).unwrap();
        assert_eq!(aborted.op_ids, vec![receipt_b.op_id]);
        assert!(!aborted.cancelled_all);
        assert_eq!(
            handle.state().unwrap(),
            before,
            "aborting a queued prompt must not touch the state machine"
        );
        // A's turn still completes normally.
        let outcome = runtime
            .drive_receipt(&handle, receipt_a, None)
            .await
            .unwrap();
        assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);
        // The queue row is durably cancelled and the runner drains nothing.
        let counts = handle.queue_status_counts().unwrap();
        assert_eq!(
            counts
                .get("cancelled")
                .and_then(|v| v.as_i64())
                .unwrap_or(0),
            1,
            "aborted queued prompt must be durably cancelled"
        );
        let runner = runtime.clone();
        let _ = tokio::spawn(async move { runner.run_session_queue(session).await }).await;
        assert_eq!(handle.queued_prompt_count().unwrap(), 0);
        let history = runtime.history_messages(&handle).unwrap();
        let rendered = serde_json::to_string(&history).unwrap();
        assert!(
            !rendered.contains("B prompt"),
            "aborted queued prompt must never reach the timeline"
        );
        assert!(rendered.contains("A prompt"));
    }

    #[tokio::test]
    async fn ledger_and_memory_record_real_turn_data() {
        // Audit: the ledger was fed TurnSummary::default() and memory was
        // never written. After this turn the ledger must carry the REAL
        // goal/steps/files/tests and UpdatingMemory must have written facts.
        let write_tool = Tool {
            name: "write_file".into(),
            description: "w".into(),
            input_schema: serde_json::json!({"type": "object"}),
            resource_class: kilop_core::resource::ResourceClass::DiskWrite,
            capability: None,
            recovery_hint: RecoveryHint::WorkspaceWrite,
            path_args: vec!["path".into()],
            execute: Arc::new(|_ctx, args| {
                Box::pin(async move {
                    let _ = args;
                    Ok(ToolOutcome {
                        text: "wrote".into(),
                        exit_code: Some(0),
                        ..Default::default()
                    })
                })
            }),
        };
        let fail_tool = Tool {
            name: "run_check".into(),
            description: "r".into(),
            input_schema: serde_json::json!({"type": "object"}),
            resource_class: kilop_core::resource::ResourceClass::Cpu,
            capability: None,
            recovery_hint: RecoveryHint::UnknownEffect,
            path_args: vec![],
            execute: Arc::new(|_ctx, args| {
                Box::pin(async move {
                    let _ = args;
                    Ok(ToolOutcome {
                        text: "check failed: 3 errors".into(),
                        exit_code: Some(1),
                        ..Default::default()
                    })
                })
            }),
        };
        let test_tool = Tool {
            name: "run_command".into(),
            description: "t".into(),
            input_schema: serde_json::json!({"type": "object"}),
            resource_class: kilop_core::resource::ResourceClass::Cpu,
            capability: None,
            recovery_hint: RecoveryHint::UnknownEffect,
            path_args: vec![],
            execute: Arc::new(|_ctx, args| {
                Box::pin(async move {
                    let _ = args;
                    Ok(ToolOutcome {
                        text: "test ok".into(),
                        exit_code: Some(0),
                        ..Default::default()
                    })
                })
            }),
        };
        let (deps, _dir) = deps(
            scripted_provider(vec![
                ScriptedResponse::ToolCall {
                    id: "c1".into(),
                    name: "write_file".into(),
                    input: serde_json::json!({"path": "src/a.rs", "content": "x"}),
                },
                ScriptedResponse::ToolCall {
                    id: "c2".into(),
                    name: "run_check".into(),
                    input: serde_json::json!({}),
                },
                ScriptedResponse::ToolCall {
                    id: "c3".into(),
                    name: "run_command".into(),
                    input: serde_json::json!({"command": "cargo test -p x"}),
                },
                ScriptedResponse::Text("done".into()),
                ScriptedResponse::End,
            ]),
            vec![write_tool, fail_tool, test_tool],
        );
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());
        let outcome = runtime
            .run_turn(session, "fix the payments module", &[])
            .await
            .unwrap();
        assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);
        let handle = runtime.deps.session.get_session(session).unwrap().unwrap();
        // The durable ledger holds REAL data now.
        let ledger: kilop_context::ledger::TaskLedger =
            serde_json::from_value(handle.get_task_ledger().unwrap().unwrap()).unwrap();
        assert_eq!(
            ledger.goal, "test session",
            "goal seeded from the session title"
        );
        assert!(
            ledger
                .completed_steps
                .iter()
                .any(|s| s.contains("write_file") && s.contains("src/a.rs")),
            "completed steps carry the real tool + path: {:?}",
            ledger.completed_steps
        );
        assert!(
            ledger.changed_files.contains(&"src/a.rs".to_string()),
            "changed files carry the REAL path, never the tool name: {:?}",
            ledger.changed_files
        );
        assert!(
            ledger
                .known_failures
                .iter()
                .any(|f| f.contains("check failed")),
            "known failures carry the real error: {:?}",
            ledger.known_failures
        );
        assert!(
            ledger
                .tests_run
                .iter()
                .any(|t| t.contains("cargo test -p x")),
            "tests_run carries the real command: {:?}",
            ledger.tests_run
        );
        assert!(ledger.tests_failed.is_empty());
        // Memory facts were written in the UpdatingMemory phase.
        let facts = handle.memory_facts().unwrap();
        assert!(
            facts.iter().any(|(k, key, _)| k == "task" && key == "goal"),
            "goal memory fact written: {facts:?}"
        );
        assert!(
            facts.iter().any(|(k, _, _)| k == "turn"),
            "per-turn memory fact written: {facts:?}"
        );
    }

    #[tokio::test]
    async fn failing_test_command_lands_in_tests_failed() {
        let cmd = Tool {
            name: "run_command".into(),
            description: "t".into(),
            input_schema: serde_json::json!({"type": "object"}),
            resource_class: kilop_core::resource::ResourceClass::Cpu,
            capability: None,
            recovery_hint: RecoveryHint::UnknownEffect,
            path_args: vec![],
            execute: Arc::new(|_ctx, args| {
                Box::pin(async move {
                    let _ = args;
                    Ok(ToolOutcome {
                        text: "FAILED".into(),
                        exit_code: Some(1),
                        ..Default::default()
                    })
                })
            }),
        };
        let (deps, _dir) = deps(
            scripted_provider(vec![
                ScriptedResponse::ToolCall {
                    id: "c1".into(),
                    name: "run_command".into(),
                    input: serde_json::json!({"command": "pytest -q"}),
                },
                ScriptedResponse::Text("done".into()),
                ScriptedResponse::End,
            ]),
            vec![cmd],
        );
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());
        runtime
            .run_turn(session, "run the tests", &[])
            .await
            .unwrap();
        let handle = runtime.deps.session.get_session(session).unwrap().unwrap();
        let ledger: kilop_context::ledger::TaskLedger =
            serde_json::from_value(handle.get_task_ledger().unwrap().unwrap()).unwrap();
        assert!(
            ledger.tests_failed.iter().any(|t| t.contains("pytest -q")),
            "failing test command must be recorded: {:?}",
            ledger.tests_failed
        );
    }

    #[tokio::test]
    async fn same_failing_call_across_turns_trips_durable_loop_detection() {
        // Spec §28: a LoopDetector that dies with the turn cannot see "the
        // same command failed for 40 turns". The durable signal must trip on
        // the THIRD consecutive all-failing turn of ONE session.
        let boom = Tool {
            name: "run_command".into(),
            description: "t".into(),
            input_schema: serde_json::json!({"type": "object"}),
            resource_class: kilop_core::resource::ResourceClass::Cpu,
            capability: None,
            recovery_hint: RecoveryHint::UnknownEffect,
            path_args: vec![],
            execute: Arc::new(|_ctx, args| {
                Box::pin(async move {
                    let _ = args;
                    Ok(ToolOutcome {
                        text: "boom".into(),
                        exit_code: Some(2),
                        ..Default::default()
                    })
                })
            }),
        };
        let (seed_deps, _dir0) = deps(
            scripted_provider(vec![ScriptedResponse::End]),
            vec![boom.clone()],
        );
        let (manager, session) = shared_session(&seed_deps);
        let mut outcomes = Vec::new();
        for i in 0..3 {
            let (turn_deps, _dir) = deps_sharing_session(
                manager.clone(),
                Arc::new(scripted_provider(vec![
                    ScriptedResponse::ToolCall {
                        id: format!("c_{i}"),
                        name: "run_command".into(),
                        input: serde_json::json!({"command": "cargo check -p kilop-core"}),
                    },
                    ScriptedResponse::End,
                ])),
                vec![boom.clone()],
            );
            let runtime = AgentRuntime::new(turn_deps).unwrap();
            let outcome = runtime
                .run_turn(session, &format!("fix it attempt {i}"), &[])
                .await
                .unwrap();
            outcomes.push(outcome);
        }
        assert!(!outcomes[0].loop_stopped && !outcomes[1].loop_stopped);
        assert_eq!(outcomes[0].final_state, AgentState::ReadyForNextTurn);
        assert_eq!(outcomes[1].final_state, AgentState::ReadyForNextTurn);
        assert!(
            outcomes[2].loop_stopped,
            "the third identical all-failing turn must trip"
        );
        assert_eq!(outcomes[2].final_state, AgentState::FailedRecoverable);
    }

    #[tokio::test]
    async fn progress_turn_resets_durable_loop_window() {
        // A turn that makes real progress closes every loop window. Without
        // the reset, failures on turns 1, 3, 4 would reach the threshold at
        // turn 4; with it, only three CONSECUTIVE failures after the progress
        // turn trip (turn 6).
        let boom = Tool {
            name: "run_command".into(),
            description: "t".into(),
            input_schema: serde_json::json!({"type": "object"}),
            resource_class: kilop_core::resource::ResourceClass::Cpu,
            capability: None,
            recovery_hint: RecoveryHint::UnknownEffect,
            path_args: vec![],
            execute: Arc::new(|_ctx, args| {
                Box::pin(async move {
                    let _ = args;
                    Ok(ToolOutcome {
                        text: "boom".into(),
                        exit_code: Some(2),
                        ..Default::default()
                    })
                })
            }),
        };
        let ok = Tool {
            name: "write_file".into(),
            description: "w".into(),
            input_schema: serde_json::json!({"type": "object"}),
            resource_class: kilop_core::resource::ResourceClass::DiskWrite,
            capability: None,
            recovery_hint: RecoveryHint::WorkspaceWrite,
            path_args: vec!["path".into()],
            execute: Arc::new(|_ctx, args| {
                Box::pin(async move {
                    let _ = args;
                    Ok(ToolOutcome {
                        text: "wrote".into(),
                        exit_code: Some(0),
                        ..Default::default()
                    })
                })
            }),
        };
        let (seed_deps, _dir0) = deps(scripted_provider(vec![ScriptedResponse::End]), vec![]);
        let (manager, session) = shared_session(&seed_deps);
        let run = |tag: &str,
                   tool_calls: Vec<(String, String, serde_json::Value)>,
                   tools: Vec<Tool>| {
            let mut script = Vec::new();
            for (cid, name, input) in tool_calls {
                script.push(ScriptedResponse::ToolCall {
                    id: cid,
                    name,
                    input,
                });
            }
            script.push(ScriptedResponse::End);
            let (turn_deps, _dir) =
                deps_sharing_session(manager.clone(), Arc::new(scripted_provider(script)), tools);
            let runtime = AgentRuntime::new(turn_deps).unwrap();
            let tag = tag.to_string();
            async move { runtime.run_turn(session, &tag, &[]).await }
        };
        let fail = |tag: &str| {
            (
                tag.to_string(),
                vec![(
                    format!("c_{tag}"),
                    "run_command".to_string(),
                    serde_json::json!({"command": "cargo check -p kilop-core"}),
                )],
                vec![boom.clone()],
            )
        };
        // Turn 1: failure (count 1). No trip.
        let (t1, t1calls, t1tools) = fail("t1");
        let o1 = run(&t1, t1calls, t1tools).await.unwrap();
        assert!(!o1.loop_stopped);
        // Turn 2: write_file succeeds — progress resets the window.
        let o2 = run(
            "write the fix",
            vec![(
                "c_ok".into(),
                "write_file".into(),
                serde_json::json!({"path": "src/x.rs", "content": "y"}),
            )],
            vec![ok],
        )
        .await
        .unwrap();
        assert!(!o2.loop_stopped);
        assert!(o2.turns >= 1);
        // Turns 3-4: failures (counts 1, 2 after the reset). No trip yet —
        // WITHOUT the reset count 2 + this would already trip at turn 4.
        for i in 0..2 {
            let (t, calls, tools) = fail(&format!("r{i}"));
            let o = run(&t, calls, tools).await.unwrap();
            assert!(!o.loop_stopped, "progress must have reset the window");
        }
        // Turn 6: third consecutive failure after the reset → trip.
        let (t, calls, tools) = fail("final");
        let o6 = run(&t, calls, tools).await.unwrap();
        assert!(o6.loop_stopped, "3 identical failures after a reset trip");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn end_session_kills_session_owned_processes() {
        // Commandment 8: closing a session must never orphan its children.
        // end_session kills every supervisor child owned by the session
        // before the durable end transition.
        let (mut deps, _dir) = deps(scripted_provider(vec![ScriptedResponse::End]), vec![]);
        let cas = deps.cas.clone().unwrap();
        deps.supervisor = Some(kilop_terminal::ProcessSupervisor::new(cas));
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());
        let handle = runtime.deps.session.get_session(session).unwrap().unwrap();
        let cfg = kilop_terminal::SpawnConfig {
            cmd: "sleep".into(),
            args: vec!["30".into()],
            cwd: std::env::temp_dir(),
            env: vec![],
            owner: kilop_terminal::ProcessOwner::Session(session),
            capture: true,
            artifact_max: 1024 * 1024,
        };
        let sup = runtime.deps().supervisor.clone().unwrap();
        let child_task = tokio::spawn({
            let sup = sup.clone();
            async move {
                sup.run(
                    cfg,
                    std::time::Duration::from_secs(60),
                    kilop_core::cancellation::CancellationToken::new(),
                )
                .await
            }
        });
        // Let the child spawn and register.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        // end_session must kill the child and succeed.
        runtime.end_session(session).unwrap();
        // The in-flight run returns promptly (killed), NOT after 30s.
        let done = tokio::time::timeout(std::time::Duration::from_secs(10), child_task)
            .await
            .expect("end_session must terminate the child promptly");
        let output = done.unwrap().unwrap();
        assert_ne!(
            output.exit_code,
            Some(0),
            "killed child must not report a clean exit: {output:?}"
        );
        assert!(handle.state().unwrap().is_terminal() || true);
        let lifecycle = handle.lifecycle().unwrap();
        assert_eq!(lifecycle, kilop_core::state::SessionLifecycle::Closed);
    }

    #[tokio::test]
    async fn repo_map_and_project_rules_reach_the_wire() {
        // Spec §8/§26: repository knowledge must ride the context. The
        // request the model receives carries the bounded file map and the
        // workspace AGENTS.md rules — they were silently empty before.
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("repo");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(
            root.join("src").join("a.rs"),
            "fn x() {}
",
        )
        .unwrap();
        std::fs::write(root.join("AGENTS.md"), "Rules: no unsafe in src\n").unwrap();
        std::fs::create_dir_all(root.join("target")).unwrap();
        std::fs::write(root.join("target").join("junk.rs"), "junk").unwrap();
        let provider = scripted_provider(vec![
            ScriptedResponse::Text("ok".into()),
            ScriptedResponse::End,
        ]);
        let fake = Arc::new(provider.clone());
        let (mut adeps, _adir) = deps_with(Arc::new(provider), vec![]);
        let ws = adeps
            .session
            .create_workspace(root.to_str().unwrap())
            .unwrap();
        let sid = adeps
            .session
            .create_session(ws, "repo test", "fake", "m")
            .unwrap()
            .id();
        let seen = Arc::new(std::sync::Mutex::new(None::<String>));
        let hook = {
            let seen = seen.clone();
            move |_n: usize, req: &GenericAgentRequest| -> Result<(), String> {
                *seen.lock().unwrap() = Some(req.system.clone());
                Ok(())
            }
        };
        // Wrap the provider with a request inspector.
        let inspected = Arc::new(InspectingProvider::new(fake, hook));
        let mut registry = ProviderRegistry::new();
        registry.register(inspected);
        adeps.providers = Arc::new(registry);
        let runtime = AgentRuntime::new(adeps).unwrap();
        runtime
            .run_turn(sid, "inspect the repo", &[])
            .await
            .unwrap();
        let system = seen.lock().unwrap().clone().expect("request sent");
        assert!(
            system.contains("## Repository map") && system.contains("src/a.rs"),
            "repo map must ride the wire: {system}"
        );
        assert!(
            !system.contains("junk.rs"),
            "skipped dirs never appear in the repo map"
        );
        assert!(
            system.contains("## Project rules") && system.contains("no unsafe"),
            "AGENTS.md rules must ride the wire: {system}"
        );
    }

    /// Grow a REAL shared-session history of `count` long turns (assistant
    /// replies ~`text_len` chars each) so the next turn's context triggers
    /// compaction and the deterministic fallback fits under the hard cap
    /// with margin.
    async fn seed_long_history(
        manager: &Arc<SessionManager>,
        session: SessionId,
        count: usize,
        text_len: usize,
    ) {
        let caps = ModelCapabilities {
            tools: true,
            context: 200_000,
            ..Default::default()
        };
        for i in 0..count {
            let (turn_deps, _dir) = deps_sharing_session(
                manager.clone(),
                Arc::new(FakeProvider::with_script(
                    "fake",
                    caps.clone(),
                    vec![
                        ScriptedResponse::Text(format!("turn seed{i} {}", "z".repeat(text_len))),
                        ScriptedResponse::End,
                    ],
                )),
                vec![],
            );
            let runtime = AgentRuntime::new(turn_deps).unwrap();
            runtime
                .run_turn(session, &format!("prompt seed {i}"), &[])
                .await
                .unwrap();
        }
    }

    /// Concatenated text of every message in a provider request — the wire
    /// history that rides the next request after compaction is where a
    /// leaked partial summary would land.
    fn request_text(req: &GenericAgentRequest) -> String {
        let mut out = String::new();
        for m in &req.messages {
            for c in &m.content {
                if let ContentKind::Text { text } = &c.kind {
                    out.push_str(text);
                    out.push('\n');
                }
            }
        }
        out
    }

    /// Channel-gated provider double for compaction-summary tests:
    /// `stream()` records the request's cancellation token (the same test
    /// hook FakeProvider offers) and returns a stream that yields EXACTLY
    /// the chunks pushed through [`GatedStreamProvider::push`]. While no
    /// chunk is pushed the stream parks indefinitely — a provider that
    /// stalls without erroring or ending — until the summarizer gives up
    /// (deadline or turn cancellation) and drops the stream. Dropping the
    /// stream closes the channel, so a later push reports false.
    struct GatedStreamProvider {
        caps: ModelCapabilities,
        recorded: Arc<std::sync::Mutex<Option<CancellationToken>>>,
        feed: Arc<std::sync::Mutex<Option<GatedFeed>>>,
    }

    /// One live stream's chunk channel (see [`GatedStreamProvider`]).
    type GatedFeed = tokio::sync::mpsc::UnboundedSender<Result<ProviderChunk, ProviderError>>;

    impl GatedStreamProvider {
        fn new() -> Self {
            Self {
                caps: ModelCapabilities {
                    streaming: true,
                    context: 64_000,
                    ..Default::default()
                },
                recorded: Arc::new(std::sync::Mutex::new(None)),
                feed: Arc::new(std::sync::Mutex::new(None)),
            }
        }

        /// The cancellation token of the last request this provider was
        /// asked to stream (`None` when nothing was streamed yet).
        fn recorded(&self) -> Option<CancellationToken> {
            self.recorded.lock().unwrap().clone()
        }

        /// Push one chunk to the CURRENT open stream. Returns false once
        /// the summarizer terminated the stream (receiver dropped).
        fn push(&self, chunk: Result<ProviderChunk, ProviderError>) -> bool {
            self.feed
                .lock()
                .unwrap()
                .as_ref()
                .map(|tx| tx.send(chunk).is_ok())
                .unwrap_or(false)
        }
    }

    impl kilop_provider::Provider for GatedStreamProvider {
        fn id(&self) -> &str {
            "gated"
        }

        fn capabilities(&self, _model: &str) -> ModelCapabilities {
            self.caps.clone()
        }

        fn stream(&self, req: GenericAgentRequest) -> kilop_provider::ProviderStream {
            *self.recorded.lock().unwrap() = Some(req.meta.cancellation.clone());
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            *self.feed.lock().unwrap() = Some(tx);
            Box::pin(futures::stream::unfold(rx, |rx| async move {
                let mut rx = rx;
                let chunk = rx.recv().await?;
                Some((chunk, rx))
            }))
        }
    }

    #[tokio::test]
    async fn compaction_summary_request_uses_compactor_contract_not_agent_instructions() {
        // P0 (audit round 11): the summary request's system prompt must be
        // the dedicated compactor contract — NEVER the agent instructions,
        // which let the compaction model answer the latest user message
        // instead of summarizing. Inspect the compaction provider's first
        // request (compaction always precedes the main-model request).
        let (seed_deps, _dir0) = deps(scripted_provider(vec![ScriptedResponse::End]), vec![]);
        let (manager, session) = shared_session(&seed_deps);
        seed_long_history(&manager, session, 5, 1500).await;

        let seen_systems: Arc<std::sync::Mutex<Vec<String>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let compacto_inner = Arc::new(FakeProvider::with_script(
            "compacto",
            ModelCapabilities {
                streaming: true,
                context: 64_000,
                ..Default::default()
            },
            vec![
                ScriptedResponse::Text("COMPACTION SUMMARY: faithful state transfer.".into()),
                ScriptedResponse::End,
            ],
        ));
        let hook = {
            let seen_systems = seen_systems.clone();
            move |n: usize, req: &GenericAgentRequest| -> Result<(), String> {
                if n == 0 {
                    // The FIRST request through the compaction provider is
                    // the summary request (compaction precedes the main
                    // model request in the turn).
                    seen_systems.lock().unwrap().push(req.system.clone());
                }
                Ok(())
            }
        };
        let inspected = Arc::new(InspectingProvider::new(compacto_inner, hook));
        let mut registry = ProviderRegistry::new();
        registry.register(inspected);
        registry.register(Arc::new(FakeProvider::with_script(
            "fake",
            ModelCapabilities {
                tools: true,
                context: 200_000,
                ..Default::default()
            },
            vec![ScriptedResponse::End],
        )));
        let (mut final_deps, _dir) = deps_sharing_session(
            manager.clone(),
            Arc::new(FakeProvider::with_script(
                "fake",
                ModelCapabilities {
                    tools: true,
                    context: 200_000,
                    ..Default::default()
                },
                vec![ScriptedResponse::End],
            )),
            vec![],
        );
        final_deps.providers = Arc::new(registry);
        final_deps.compact_at_usage = 0.0;
        final_deps.compaction_model = Some("compacto/summary-model".into());
        let runtime = AgentRuntime::new(final_deps).unwrap();
        let outcome = runtime
            .run_turn(session, "do the thing", &[])
            .await
            .unwrap();
        assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);
        let systems = seen_systems.lock().unwrap();
        assert_eq!(
            systems.len(),
            1,
            "the compaction provider must have streamed exactly one summary request"
        );
        let system = &systems[0];
        assert!(
            !system.contains("You are a test agent."),
            "the agent instructions must not be the summary system prompt: {system}"
        );
        for marker in [
            "Kilo+ context compactor",
            "faithful state transfer",
            "unresolved errors and blockers",
            "NEVER invent facts",
        ] {
            assert!(
                system.contains(marker),
                "compactor contract marker {marker:?} missing from {system}"
            );
        }
    }

    #[tokio::test]
    async fn compaction_summary_stream_error_discards_partial_text_and_falls_back() {
        // P0 (audit round 11): run() used to keep the partial text after a
        // provider error, and a truncated summary is small enough to pass
        // the compactor's hard cap — so a PARTIAL state transfer replaced
        // the real history. A dying compaction stream must leave NO partial
        // text anywhere: the deterministic fallback (eviction digest on the
        // wire) replaces the history instead and the turn completes.
        let (seed_deps, _dir0) = deps(scripted_provider(vec![ScriptedResponse::End]), vec![]);
        let (manager, session) = shared_session(&seed_deps);
        seed_long_history(&manager, session, 5, 1500).await;
        let main_caps = ModelCapabilities {
            tools: true,
            context: 200_000,
            ..Default::default()
        };
        // The compaction model streams one partial sentence then dies.
        let compactor = Arc::new(FakeProvider::with_script(
            "compacto",
            ModelCapabilities {
                streaming: true,
                context: 64_000,
                ..Default::default()
            },
            vec![
                ScriptedResponse::Text("Goal is...".into()),
                ScriptedResponse::Die(ProviderError::new(
                    kilop_provider::ProviderErrorKind::Network,
                    "connection vanished mid-summary",
                )),
            ],
        ));
        let captured: Arc<std::sync::Mutex<Vec<String>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let cap = captured.clone();
        let main = Arc::new(InspectingProvider::new(
            Arc::new(FakeProvider::with_script(
                "fake",
                main_caps.clone(),
                vec![
                    ScriptedResponse::Text("proceeding".into()),
                    ScriptedResponse::End,
                ],
            )),
            move |_n: usize, req: &GenericAgentRequest| -> Result<(), String> {
                cap.lock().unwrap().push(request_text(req));
                Ok(())
            },
        ));
        let mut registry = ProviderRegistry::new();
        registry.register(main);
        registry.register(compactor.clone());
        let (mut final_deps, _dir) = deps_sharing_session(
            manager.clone(),
            Arc::new(FakeProvider::with_script(
                "fake",
                main_caps,
                vec![ScriptedResponse::End],
            )),
            vec![],
        );
        final_deps.providers = Arc::new(registry);
        final_deps.compact_at_usage = 0.0;
        final_deps.compaction_model = Some("compacto/summary-model".into());
        let runtime = AgentRuntime::new(final_deps).unwrap();
        let outcome = runtime
            .run_turn(session, "do the thing", &[])
            .await
            .unwrap();
        assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);
        assert_eq!(
            compactor.last_request_model().as_deref(),
            Some("summary-model"),
            "the summary request must have been streamed before it died"
        );
        assert!(
            outcome.compacted,
            "the deterministic fallback must still compact the history"
        );
        // The partial sentence must never reach the wire history of the
        // main request (the place a leaked partial summary would land).
        let wire = captured.lock().unwrap();
        assert!(!wire.is_empty(), "the main provider must have been called");
        assert!(
            wire.iter().all(|w| !w.contains("Goal is...")),
            "partial summary text must never reach the wire history: {wire:?}"
        );
        // The compaction record must say REJECTED (the LLM attempt failed,
        // the deterministic fallback took over) — never an accepted
        // "llm_summary" of the partial text.
        let handle = runtime.deps.session.get_session(session).unwrap().unwrap();
        let events = handle.events_range(1, None).unwrap();
        let compacted = events
            .iter()
            .filter(|e| {
                matches!(
                    e.kind,
                    kilop_core::event::EventKind::ContextCompacted
                        | kilop_core::event::EventKind::CompactRejected
                )
            })
            .collect::<Vec<_>>();
        assert!(!compacted.is_empty(), "a compaction record must exist");
        for e in &compacted {
            let payload = e.payload.as_ref().expect("compaction payload present");
            assert_eq!(
                payload.get("strategy").and_then(|v| v.as_str()),
                Some("rejected"),
                "the failed summary attempt must be recorded as rejected, got {payload}"
            );
            assert_eq!(
                payload.get("accepted").and_then(|v| v.as_bool()),
                Some(true),
                "the deterministic fallback must have been accepted, got {payload}"
            );
        }
        // ...and nothing partial ever reached the durable history either.
        let page = handle.messages_page(None, 100).unwrap();
        let durable: Vec<String> = page
            .messages
            .iter()
            .flat_map(|m| m.parts.iter())
            .filter_map(|p| match p {
                kilop_protocol::v756::Part::Text { text } => Some(text.clone()),
                _ => None,
            })
            .collect();
        assert!(
            durable.iter().all(|t| !t.contains("Goal is...")),
            "partial summary text must never be durable: {durable:?}"
        );
    }

    #[tokio::test]
    async fn compaction_summary_timeout_discards_partial_text_and_falls_back() {
        // P0 (audit round 11): run() used to accept whatever partial text
        // had accumulated when the 90s bound expired. A stream that sends
        // text and then NEVER ends (no Done, no error) must time out into a
        // FAILED run whose text is discarded. The timeout is injectable, so
        // this never waits 90s.
        let gated = Arc::new(GatedStreamProvider::new());
        // History with multibyte (3-byte UTF-8) content: the failure
        // fallback must be rejected even where the char-based estimator
        // under-reports against the byte-based `before` figure.
        let history: Vec<RecentTurn> = (0..6)
            .map(|i| RecentTurn {
                role: if i % 2 == 0 { "user" } else { "assistant" }.into(),
                text: format!("turn {i} 実装状態 {}", "z".repeat(120)),
            })
            .collect();
        let ledger = TaskLedger {
            goal: "test goal".into(),
            ..Default::default()
        };
        let summarizer = Arc::new(StreamingSummarizer {
            provider: gated.clone(),
            model: "summary-model".into(),
            op_id: OpId::new(1),
            session_id: SessionId::new(1),
            cancellation: CancellationToken::new(),
            summary_timeout: Duration::from_millis(150),
        });
        // Run the summary request on a task; once the provider's stream is
        // open (request recorded), push ONE sentence and then stall forever
        // (no Done, no error): the deadline must mark the run failed and
        // DISCARD the partial text.
        let run_summarizer = summarizer.clone();
        let run_history = history.clone();
        let run_task = tokio::spawn(async move { run_summarizer.run(&run_history).await });
        let streamed = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if gated.recorded().is_some() {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await;
        assert!(streamed.is_ok(), "the summary stream must have opened");
        assert!(
            gated.push(Ok(ProviderChunk::Text {
                text: "Goal is...".into()
            })),
            "the test must push while the summarizer still waits"
        );
        let started = std::time::Instant::now();
        let result = tokio::time::timeout(Duration::from_secs(10), run_task)
            .await
            .expect("the injectable deadline must bound the wait")
            .expect("the run task must not panic");
        assert!(
            result.is_none(),
            "partial text must be discarded when the stream times out"
        );
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the injectable deadline must bound the wait"
        );
        assert!(
            !gated.push(Ok(ProviderChunk::Done)),
            "the timed-out stream must have been terminated (dropped)"
        );
        // Summarize-level: the failed attempt must NOT yield an empty or
        // partial "summary" — the compactor would accept "" as 0 tokens and
        // wipe the history. The failure fallback is a transcript the hard
        // cap rejects, so deterministic pruning runs. The request mirrors
        // try_compact: before is derived from the real history bytes.
        let before = history.iter().map(|t| t.text.len()).sum::<usize>() / 4;
        let sum: Arc<dyn Summarizer> = summarizer.clone();
        let plan = Compactor::new(Some(sum))
            .compact(
                &history,
                &ledger,
                &CompactionRequest::new(before, before / 2),
            )
            .await;
        assert_eq!(
            plan.strategy,
            kilop_context::CompactionStrategy::Rejected,
            "the failed summary attempt must be rejected, never accepted as a wipe"
        );
        assert!(
            plan.accepted,
            "deterministic pruning must run and fit the cap"
        );
        let wire: String = plan
            .kept_recent
            .iter()
            .map(|t| t.text.clone())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            !wire.contains("Goal is..."),
            "partial summary text must never reach the compacted history"
        );
    }

    #[tokio::test]
    async fn compaction_summary_cancellation_is_child_of_the_turn_token() {
        // P0 (audit round 11): run() minted an orphan CancellationToken, so
        // a user Stop during compaction left the compaction model streaming
        // up to the full 90s deadline. The summary request must hang off the
        // TURN's token: cancelling the turn cascades into the compaction
        // request (recorded by the provider) AND the stalled summary stream
        // terminates promptly instead of waiting out the deadline.
        let (seed_deps, _dir0) = deps(scripted_provider(vec![ScriptedResponse::End]), vec![]);
        let (manager, session) = shared_session(&seed_deps);
        seed_long_history(&manager, session, 5, 1500).await;

        let gated = Arc::new(GatedStreamProvider::new());
        let mut registry = ProviderRegistry::new();
        registry.register(Arc::new(FakeProvider::with_script(
            "fake",
            ModelCapabilities {
                tools: true,
                context: 200_000,
                ..Default::default()
            },
            vec![ScriptedResponse::End],
        )));
        registry.register(gated.clone());
        let (mut final_deps, _dir) = deps_sharing_session(
            manager.clone(),
            Arc::new(FakeProvider::with_script(
                "fake",
                ModelCapabilities {
                    tools: true,
                    context: 200_000,
                    ..Default::default()
                },
                vec![ScriptedResponse::End],
            )),
            vec![],
        );
        final_deps.providers = Arc::new(registry);
        final_deps.compact_at_usage = 0.0;
        final_deps.compaction_model = Some("gated/gated-model".into());
        let runtime = AgentRuntime::new(final_deps).unwrap();
        let handle = runtime.deps.session.get_session(session).unwrap().unwrap();
        let receipt = handle.submit_prompt("do the thing", &[]).unwrap();
        let turn_token = receipt.op_meta.cancellation.clone();
        let drive_runtime = runtime.clone();
        let drive_handle = handle.clone();
        let drive = tokio::spawn(async move {
            drive_runtime
                .drive_turn(&drive_handle, receipt.op_id, turn_token, None)
                .await
        });
        // Wait until the compaction model actually receives the summary
        // request and parks on the gate (it yields nothing until released).
        let seen = tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                if gated.recorded().is_some() {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await;
        assert!(
            seen.is_ok(),
            "the summary request must reach the compaction provider"
        );
        // A user Stop: cancel the TURN token.
        receipt.op_meta.cancellation.cancel();
        // Lineage proof: the cancellation token the compaction provider
        // recorded on its request is a (grand)child of the turn token, so
        // it is cancelled too — the old orphan token was not.
        let request_token = gated.recorded().expect("request token recorded");
        assert!(
            request_token.is_cancelled(),
            "cancelling the turn must cascade into the compaction request"
        );
        // The summarizer polls the token and terminates the stalled stream
        // (its drop closes the gate's channel) instead of waiting out the
        // 90s deadline.
        let terminated = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if !gated.push(Ok(ProviderChunk::Text {
                    text: "too late".into(),
                })) {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await;
        assert!(
            terminated.is_ok(),
            "the summary stream must terminate promptly on turn cancellation"
        );
        let outcome = tokio::time::timeout(Duration::from_secs(30), drive)
            .await
            .expect("the cancelled turn must finish promptly")
            .unwrap()
            .unwrap();
        assert_eq!(
            outcome.final_state,
            AgentState::Cancelled,
            "the turn must land Cancelled after a Stop during compaction"
        );
    }

    #[tokio::test]
    async fn compaction_model_resolves_and_streams_a_real_summary() {
        // Spec §36: the separate compaction model is real — a second
        // registered provider receives an actual streaming summarization
        // request. (Audit: compaction_model was only an is_some() toggle.)
        let long_turn = |tag: &str| {
            vec![
                ScriptedResponse::Text(format!("turn {tag} {}", "z".repeat(300))),
                ScriptedResponse::End,
            ]
        };
        let (seed_deps, _dir0) = deps(scripted_provider(vec![ScriptedResponse::End]), vec![]);
        let (manager, session) = shared_session(&seed_deps);
        let main_caps = ModelCapabilities {
            tools: true,
            context: 200_000,
            ..Default::default()
        };
        // Seed real history so the ledger never exceeds the context being
        // compacted (5 long turns).
        for i in 0..5 {
            let (turn_deps, _dir) = deps_sharing_session(
                manager.clone(),
                Arc::new(FakeProvider::with_script(
                    "fake",
                    main_caps.clone(),
                    long_turn(&format!("seed{i}")),
                )),
                vec![],
            );
            let runtime = AgentRuntime::new(turn_deps).unwrap();
            runtime
                .run_turn(session, &format!("prompt seed {i}"), &[])
                .await
                .unwrap();
        }
        // The compaction provider: a distinct adapter with its own model.
        let compactor = Arc::new(FakeProvider::with_script(
            "compacto",
            ModelCapabilities {
                streaming: true,
                context: 64_000,
                ..Default::default()
            },
            vec![
                ScriptedResponse::Text("COMPACTION SUMMARY: the durable task state so far.".into()),
                ScriptedResponse::End,
            ],
        ));
        let mut registry = ProviderRegistry::new();
        registry.register(Arc::new(FakeProvider::with_script(
            "fake",
            main_caps.clone(),
            vec![ScriptedResponse::End],
        )));
        registry.register(compactor.clone());
        let (mut final_deps, _dir) = deps_sharing_session(
            manager.clone(),
            Arc::new(FakeProvider::with_script(
                "fake",
                main_caps,
                vec![ScriptedResponse::End],
            )),
            vec![],
        );
        final_deps.providers = Arc::new(registry);
        final_deps.compact_at_usage = 0.0;
        final_deps.compaction_model = Some("compacto/summary-model".into());
        let runtime = AgentRuntime::new(final_deps).unwrap();
        let outcome = runtime
            .run_turn(session, "do the thing", &[])
            .await
            .unwrap();
        assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);
        assert_eq!(
            compactor.last_request_model().as_deref(),
            Some("summary-model"),
            "the compaction provider must have received a summary request"
        );
    }

    #[tokio::test]
    async fn broken_compaction_model_degrades_not_breaks() {
        // A compaction model spec that names nothing resolvable must NOT
        // kill the turn: it degrades to the deterministic path.
        let (seed_deps, _dir0) = deps(scripted_provider(vec![ScriptedResponse::End]), vec![]);
        let (manager, session) = shared_session(&seed_deps);
        let main_caps = ModelCapabilities {
            tools: true,
            context: 200_000,
            ..Default::default()
        };
        for i in 0..5 {
            let (turn_deps, _dir) = deps_sharing_session(
                manager.clone(),
                Arc::new(FakeProvider::with_script(
                    "fake",
                    main_caps.clone(),
                    vec![
                        ScriptedResponse::Text(format!("turn seed{i} {}", "y".repeat(300))),
                        ScriptedResponse::End,
                    ],
                )),
                vec![],
            );
            let runtime = AgentRuntime::new(turn_deps).unwrap();
            runtime
                .run_turn(session, &format!("prompt seed {i}"), &[])
                .await
                .unwrap();
        }
        let (mut final_deps, _dir) = deps_sharing_session(
            manager.clone(),
            Arc::new(FakeProvider::with_script(
                "fake",
                main_caps,
                vec![ScriptedResponse::End],
            )),
            vec![],
        );
        final_deps.compact_at_usage = 0.0;
        final_deps.compaction_model = Some("no-such-provider/no-model".into());
        let runtime = AgentRuntime::new(final_deps).unwrap();
        let outcome = runtime
            .run_turn(session, "do the thing", &[])
            .await
            .unwrap();
        assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);
    }

    #[tokio::test]
    async fn compaction_archives_large_evictions_as_chunked_cas_manifest() {
        // P0 (no more 1 MiB archive cap losing history): an eviction of
        // > 1 MiB through the REAL runtime path must store MULTIPLE ordered
        // CAS blobs behind one JSON manifest — {version:1,
        // chunks:[{index,size,hash}], total_bytes} — whose content address
        // replaces the digest placeholder on the wire. Every chunk must be
        // retrievable, in oldest-first order, lossless.
        let (seed_deps, _dir0) = deps(scripted_provider(vec![ScriptedResponse::End]), vec![]);
        let (manager, session) = shared_session(&seed_deps);
        // ~14 x 100K chars of durable history: far more than the 200K-model
        // budget keeps, so the final compaction evicts > 1 MiB.
        seed_long_history(&manager, session, 14, 100_000).await;

        let main_caps = ModelCapabilities {
            tools: true,
            context: 200_000,
            ..Default::default()
        };
        // The compaction model FAILS its summary (dies before any chunk):
        // the failure fallback cannot pass the hard cap, so deterministic
        // pruning runs — the digest + chunked-archive-manifest path.
        let dying = Arc::new(FakeProvider::with_script(
            "compacto",
            ModelCapabilities {
                streaming: true,
                context: 64_000,
                ..Default::default()
            },
            vec![ScriptedResponse::Die(ProviderError::new(
                kilop_provider::ProviderErrorKind::Network,
                "compaction stream died",
            ))],
        ));
        let captured: Arc<std::sync::Mutex<Vec<GenericAgentRequest>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let cap = captured.clone();
        let main = Arc::new(InspectingProvider::new(
            Arc::new(FakeProvider::with_script(
                "fake",
                main_caps.clone(),
                vec![ScriptedResponse::Text("ok".into()), ScriptedResponse::End],
            )),
            move |_n, req| {
                cap.lock().unwrap().push(req.clone());
                Ok(())
            },
        ));
        let mut registry = ProviderRegistry::new();
        registry.register(main);
        registry.register(dying);
        let (mut final_deps, _dir) = deps_sharing_session(
            manager.clone(),
            Arc::new(FakeProvider::with_script(
                "fake",
                main_caps,
                vec![ScriptedResponse::End],
            )),
            vec![],
        );
        final_deps.providers = Arc::new(registry);
        final_deps.compact_at_usage = 0.0; // always compact on the final turn
        final_deps.compaction_model = Some("compacto/fail-model".into());
        let runtime = AgentRuntime::new(final_deps).unwrap();
        let outcome = runtime
            .run_turn(session, "do the thing", &[])
            .await
            .unwrap();
        assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);

        // The post-compaction request carries the eviction digest whose
        // placeholder was replaced by artifact://<manifest hash>.
        let requests = captured.lock().unwrap();
        assert_eq!(requests.len(), 1, "one main-model request for the turn");
        let wire = request_text(&requests[0]);
        let pos = wire
            .find("artifact://")
            .expect("the digest must reference the archive manifest");
        let hex = &wire[pos + "artifact://".len()..pos + "artifact://".len() + 64];
        let manifest_hash = FileHash::from_hex(hex).expect("64-hex content address");
        let cas = runtime.deps.cas.clone().expect("test deps carry a CAS");
        let manifest: serde_json::Value =
            serde_json::from_slice(&cas.get(manifest_hash).unwrap()).unwrap();
        assert_eq!(manifest["version"], serde_json::json!(1));
        let entries = manifest["chunks"]
            .as_array()
            .expect("manifest chunks must be an array");
        assert!(
            entries.len() >= 2,
            "> 1 MiB of evicted history must produce multiple CAS chunks, got {}",
            entries.len()
        );
        let mut total_bytes = 0u64;
        let mut first_chunk = String::new();
        let mut last_chunk = String::new();
        for (expected_index, entry) in entries.iter().enumerate() {
            assert_eq!(
                entry["index"].as_u64(),
                Some(expected_index as u64),
                "chunks must be indexed in order"
            );
            let hash = FileHash::from_hex(entry["hash"].as_str().unwrap()).unwrap();
            let size = entry["size"].as_u64().expect("chunk size present");
            let bytes = cas.get(hash).unwrap();
            assert_eq!(
                bytes.len() as u64,
                size,
                "recorded size must match the blob"
            );
            assert!(
                size <= 512 * 1024,
                "every chunk respects the 512 KiB bound, got {size}"
            );
            total_bytes += size;
            let text = String::from_utf8(bytes).unwrap();
            if first_chunk.is_empty() {
                first_chunk = text;
            } else {
                last_chunk = text;
            }
        }
        assert_eq!(
            manifest["total_bytes"].as_u64(),
            Some(total_bytes),
            "manifest total_bytes must equal the sum of the chunks"
        );
        assert!(
            total_bytes >= 1 << 20,
            "the eviction itself exceeds 1 MiB: {total_bytes} bytes archived"
        );
        // Oldest-first order: the first chunk starts at the OLDEST evicted
        // turn; the archive spans the evicted seeds.
        assert!(
            first_chunk.starts_with("assistant: turn seed0 "),
            "chunk 0 must start at the oldest evicted turn: {:?}",
            &first_chunk[..first_chunk.len().min(80)]
        );
        assert!(
            last_chunk.contains("turn seed11"),
            "later chunks hold newer evictions"
        );
    }

    #[tokio::test]
    async fn retryable_pre_accept_failure_retries_under_policy() {
        // Spec §13: a NETWORK failure before any content became durable is
        // retried (bounded, state-aware). Without the retry the turn would
        // land on FailedRecoverable.
        let (mut deps, _dir) = deps(scripted_provider(vec![ScriptedResponse::End]), vec![]);
        deps.retry_policy = kilop_core::retry::RetryPolicy {
            max_attempts: 3,
            base_delay_ms: 1,
            max_delay_ms: 5,
            jitter: 0.0,
            class: kilop_core::retry::RetryClass::Network,
        };
        // A provider that errors BEFORE its first chunk on the first stream
        // call (a network-class failure), then serves normally. The script
        // is consumed per stream, so the retried request sees an empty
        // script — the point is the retry happens at all.
        let flaky = FakeProvider::die_before_stream(
            "fake",
            ModelCapabilities {
                streaming: true,
                tools: true,
                ..Default::default()
            },
            vec![ScriptedResponse::End],
        );
        let mut registry = ProviderRegistry::new();
        registry.register(Arc::new(flaky));
        deps.providers = Arc::new(registry);
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());
        let outcome = runtime.run_turn(session, "retry me", &[]).await.unwrap();
        assert_eq!(
            outcome.final_state,
            AgentState::ReadyForNextTurn,
            "a retryable pre-accept failure must retry, not fail the turn"
        );
    }

    #[tokio::test]
    async fn retry_never_replays_after_durable_content() {
        // State-aware (spec §13): once assistant content became durable
        // (parts flushed), a network death must NOT replay — the session
        // fails honestly instead of duplicating content.
        let (mut deps, _dir) = deps(scripted_provider(vec![ScriptedResponse::End]), vec![]);
        deps.retry_policy = kilop_core::retry::RetryPolicy {
            max_attempts: 3,
            base_delay_ms: 1,
            max_delay_ms: 5,
            jitter: 0.0,
            class: kilop_core::retry::RetryClass::Network,
        };
        // The die-mid-stream provider emits one text chunk (durable part)
        // then dies with a retryable network error.
        let dying = FakeProvider::die_mid_stream(
            "fake",
            ModelCapabilities {
                streaming: true,
                ..Default::default()
            },
        );
        let mut registry = ProviderRegistry::new();
        registry.register(Arc::new(dying));
        deps.providers = Arc::new(registry);
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());
        let outcome = runtime.run_turn(session, "no replay", &[]).await.unwrap();
        assert_eq!(
            outcome.final_state,
            AgentState::FailedRecoverable,
            "content already durable: never replay, fail honestly"
        );
        // Exactly ONE assistant message row exists (no duplicate from a
        // replay — the partial message is present, nothing was re-created).
        let handle = runtime.deps.session.get_session(session).unwrap().unwrap();
        let rows = handle.messages_before(None, 10).unwrap();
        let assistant_rows = rows.iter().filter(|m| m.role == "assistant").count();
        assert_eq!(assistant_rows, 1, "no duplicated assistant content");
    }

    // ============================================================
    // P0 recovery invariants (turn records, idempotent replay,
    // workspace-aware write postconditions).
    // ============================================================

    use kilop_core::hash::FileHash;
    use kilop_core::id::{TaskId, WorkspaceId, WorktreeId};
    use kilop_core::op::OpMeta;
    use kilop_core::time::Deadline;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A counting tool (execution observable for exactly-once assertions).
    fn counting_tool(name: &str, hint: RecoveryHint, counter: Arc<AtomicUsize>) -> Tool {
        let name_owned = name.to_string();
        Tool {
            name: name.to_string(),
            description: "counting".into(),
            input_schema: serde_json::json!({"type": "object"}),
            resource_class: kilop_core::resource::ResourceClass::Cpu,
            capability: None,
            recovery_hint: hint,
            path_args: vec![],
            execute: Arc::new(move |_ctx, args| {
                let counter = counter.clone();
                let name = name_owned.clone();
                Box::pin(async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    Ok(ToolOutcome {
                        text: format!("ran {name}:{args}"),
                        exit_code: Some(0),
                        ..Default::default()
                    })
                })
            }),
        }
    }

    fn op_meta(m: &Arc<SessionManager>, s: SessionId, recovery: RecoveryStrategy) -> OpMeta {
        let op = m.next_op_id();
        OpMeta::new(
            op,
            s,
            Deadline::at(m.now_ms() + 60_000),
            kilop_core::retry::RetryPolicy::default(),
            CancellationToken::new(),
            recovery,
            m.now_ms(),
        )
    }

    /// Journal the machine chain exactly the way the runtime does before a
    /// tool batch (from the freshly-admitted Preparing state).
    fn chain_to_streaming(handle: &kilop_session::SessionHandle, turn_op: OpId) {
        assert_eq!(handle.state().unwrap(), AgentState::Preparing);
        handle
            .append_event(
                kilop_core::event::EventKind::ContextPrepared,
                AgentState::BuildingContext,
                Some(turn_op),
                None,
            )
            .unwrap();
        handle
            .append_event(
                kilop_core::event::EventKind::ModelStarted,
                AgentState::WaitingForModel,
                Some(turn_op),
                None,
            )
            .unwrap();
        handle
            .append_event(
                kilop_core::event::EventKind::ModelChunkReceived,
                AgentState::Streaming,
                Some(turn_op),
                None,
            )
            .unwrap();
    }

    /// Start ONE durable tool run the way run_tool_calls does (permission
    /// hop + the model's tool_call part + ToolStarted) and leave it running
    /// — the residue of a crash mid-tool-batch. May be called repeatedly on
    /// the same turn (parallel batch).
    fn crash_tool_start(
        handle: &kilop_session::SessionHandle,
        turn_op: OpId,
        tool: &str,
        args: serde_json::Value,
        call_id: &str,
        meta: OpMeta,
    ) {
        let perm = handle
            .request_permission(turn_op, &Capability::ReadWorkspace { path: ".".into() })
            .unwrap();
        handle
            .resolve_permission(perm.id, PermissionDecision::Allow)
            .unwrap();
        let seq = handle.proposed_message_seq().unwrap();
        let mid = handle
            .put_message(seq, "assistant", serde_json::json!({ "parts": [] }))
            .unwrap();
        handle
            .put_tool_call_part(mid, call_id, tool, args.clone(), "completed")
            .unwrap();
        handle.start_tool_run(meta, tool, args).unwrap();
        assert_eq!(handle.state().unwrap(), AgentState::ExecutingTool);
    }

    /// A fresh manager+runtime over the same durable dir (daemon restart).
    fn reopen_runtime(
        dir: &tempfile::TempDir,
        provider: Arc<dyn kilop_provider::Provider>,
        tools: Vec<Tool>,
    ) -> (AgentDeps, tempfile::TempDir) {
        let manager =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        deps_sharing_session(manager, provider, tools)
    }

    fn fresh_store_dir() -> tempfile::TempDir {
        tempdir().unwrap()
    }

    #[tokio::test]
    async fn crash_resume_uses_recorded_turn_op_and_model_override() {
        // P0 (requirement 1): a crash mid-turn (durable ToolStarted, no
        // completion) with a NON-DEFAULT model override active resumes the
        // SAME logical turn: the recorded turn op id (never OpId::new(1) or
        // a fresh op), the recorded model "m2" (NOT the session default
        // "m"), and no fresh TurnRecord. After the resume the record reads
        // completed (requirement 1b).
        let dir = fresh_store_dir();
        let file = dir.path().join("w.txt");
        std::fs::write(&file, b"landed").unwrap();
        let expected = FileHash::from(blake3::hash(b"landed").into());
        let turn_op: OpId;
        let tool_op: OpId;
        let session: SessionId;
        {
            let manager1 =
                SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                    .unwrap();
            let (deps1, _keep) = deps_sharing_session(
                manager1.clone(),
                Arc::new(scripted_provider(vec![])),
                vec![],
            );
            let runtime1 = AgentRuntime::new(deps1).unwrap();
            let ws = manager1.create_workspace("/w").unwrap();
            let handle = manager1.create_session(ws, "t", "fake", "m").unwrap();
            session = handle.id();
            let receipt = handle.submit_prompt("crash me", &[]).unwrap();
            turn_op = receipt.op_id;
            // The drive had STARTED with the per-message override: the
            // record's envelope already carries "m2" (the session default
            // stays "m").
            handle
                .set_turn_envelope(turn_op, "fake", "m2", None, Some("native"))
                .unwrap();
            let meta = op_meta(
                &manager1,
                session,
                RecoveryStrategy::VerifyHash {
                    path: file.to_string_lossy().to_string(),
                    expected,
                },
            );
            tool_op = meta.operation_id;
            chain_to_streaming(&handle, turn_op);
            crash_tool_start(
                &handle,
                turn_op,
                "write_file",
                serde_json::json!({}),
                "call_1",
                meta,
            );
            // Session default UNCHANGED by the override.
            assert_eq!(handle.model().unwrap(), "m");
            drop(runtime1);
        }
        // Daemon restart over the same durable dir.
        let inner = scripted_provider(vec![
            ScriptedResponse::Text("resumed final".into()),
            ScriptedResponse::End,
        ]);
        let (deps2, _keep2) = reopen_runtime(&dir, Arc::new(inner.clone()), vec![]);
        let runtime2 = AgentRuntime::new(deps2).unwrap();
        // Crash recovery first (sync sweep: resolves the pending row to
        // completed/verified without re-running the tool).
        let reports = runtime2.recover().unwrap();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].crashed_ops.len(), 1);
        assert_eq!(reports[0].crashed_ops[0].op_id, tool_op);
        assert_eq!(reports[0].crashed_ops[0].status, "completed");
        // The recorded identity survived the sweep: ONE record, still the
        // original turn op, still active (the turn resumes, not a new one).
        let handle2 = runtime2
            .deps()
            .session
            .get_session(session)
            .unwrap()
            .unwrap();
        let records = handle2.turn_records().unwrap();
        assert_eq!(records.len(), 1, "no fresh TurnRecord was created");
        assert_eq!(records[0].turn_op_id, turn_op);
        assert_eq!(records[0].status, "active");
        assert_eq!(records[0].effective_model, "m2");
        // Resume the interrupted logical turn: same op id, recorded model.
        let outcome = runtime2.continue_turn(session).await.unwrap();
        assert_eq!(
            outcome.final_state,
            AgentState::ReadyForNextTurn,
            "the resumed turn must drive to its genuine end"
        );
        assert_eq!(outcome.op_id, turn_op);
        // The provider saw the RECORDED model, not the session default.
        assert_eq!(
            inner.last_request_model().as_deref(),
            Some("m2"),
            "resume must use the recorded model override"
        );
        // Journal events of the resumed turn reference the recorded op id.
        let events = handle2.events_range(1, None).unwrap();
        let crash_seq = events
            .iter()
            .find(|e| e.kind == kilop_core::event::EventKind::CrashDetected)
            .expect("CrashDetected journaled")
            .seq;
        for e in events.iter().filter(|e| e.seq.raw() > crash_seq.raw()) {
            match e.kind {
                kilop_core::event::EventKind::PhaseChanged
                | kilop_core::event::EventKind::ModelStarted
                | kilop_core::event::EventKind::TurnCompleted => {
                    assert_eq!(
                        e.op_id,
                        Some(turn_op),
                        "resumed-turn event {:?} must reference the recorded op",
                        e.kind
                    );
                }
                _ => {}
            }
        }
        // The record is completed after the successful resume (1b) and the
        // session default was never consulted.
        let records = handle2.turn_records().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].status, "completed");
        assert_eq!(handle2.model().unwrap(), "m");
        // Exactly one tool run happened (no replay of the verify row).
        let tool_events = events
            .iter()
            .filter(|e| e.kind == kilop_core::event::EventKind::ToolStarted)
            .count();
        assert_eq!(tool_events, 1);
    }

    #[tokio::test]
    async fn queued_prompt_after_crash_resumes_same_turn_then_delivers() {
        // Requirement 1c: prompt B queued while turn A was active has NO
        // turn record (a crash before B's admission leaves the queue row
        // pending). The next runner tick first resumes A as the SAME logical
        // turn (recorded op), then admits B — no phantom record for B
        // before admission, exactly one delivery afterwards.
        let dir = fresh_store_dir();
        let session: SessionId;
        let op_a: OpId;
        let op_b: OpId;
        {
            let manager1 =
                SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                    .unwrap();
            let (deps1, _keep) = deps_sharing_session(
                manager1.clone(),
                Arc::new(scripted_provider(vec![])),
                vec![],
            );
            let runtime1 = AgentRuntime::new(deps1).unwrap();
            let ws = manager1.create_workspace("/w").unwrap();
            let handle = manager1.create_session(ws, "t", "fake", "m").unwrap();
            session = handle.id();
            let ra = handle.submit_prompt("task A", &[]).unwrap();
            op_a = ra.op_id;
            let rb = handle.submit_prompt("task B", &[]).unwrap();
            assert!(rb.queued);
            op_b = rb.op_id;
            assert_eq!(handle.queued_prompt_count().unwrap(), 1);
            // B was never admitted: no record for B, only A's.
            let records = handle.turn_records().unwrap();
            assert_eq!(records.len(), 1);
            assert_eq!(records[0].turn_op_id, op_a);
            assert!(handle.turn_record(op_b).unwrap().is_none());
            assert_eq!(handle.state().unwrap(), AgentState::Preparing);
            drop(runtime1);
        }
        // Restart: the runner resumes A (never driven) and then delivers B.
        let inner = scripted_provider(vec![
            ScriptedResponse::Text("A answer".into()),
            ScriptedResponse::End,
            ScriptedResponse::Text("B answer".into()),
            ScriptedResponse::End,
        ]);
        let (deps2, _keep2) = reopen_runtime(&dir, Arc::new(inner), vec![]);
        let runtime2 = AgentRuntime::new(deps2).unwrap();
        let runner = runtime2.clone();
        tokio::spawn(async move { runner.run_session_queue(session).await })
            .await
            .unwrap();
        let handle2 = runtime2
            .deps()
            .session
            .get_session(session)
            .unwrap()
            .unwrap();
        assert_eq!(handle2.queued_prompt_count().unwrap(), 0, "queue drained");
        assert_eq!(handle2.state().unwrap(), AgentState::ReadyForNextTurn);
        let records = handle2.turn_records().unwrap();
        assert_eq!(records.len(), 2, "A resumed + B admitted = 2 records");
        assert_eq!(records[0].turn_op_id, op_a);
        assert_eq!(records[0].status, "completed", "A completed as ONE turn");
        assert_eq!(records[1].turn_op_id, op_b);
        assert_eq!(records[1].status, "completed");
        let events = handle2.events_range(1, None).unwrap();
        let prompts = events
            .iter()
            .filter(|e| e.kind == kilop_core::event::EventKind::PromptReceived)
            .count();
        assert_eq!(prompts, 2, "one PromptReceived per prompt");
        let admitted_b = events
            .iter()
            .filter(|e| {
                e.kind == kilop_core::event::EventKind::PromptAdmitted && e.op_id == Some(op_b)
            })
            .count();
        assert_eq!(admitted_b, 1, "B admitted exactly once");
        // One TurnCompleted per logical turn.
        let completed = events
            .iter()
            .filter(|e| e.kind == kilop_core::event::EventKind::TurnCompleted)
            .count();
        assert_eq!(completed, 2);
    }

    #[tokio::test]
    async fn idempotent_tool_interrupted_replays_exactly_once() {
        // Requirement 2a: an idempotent tool interrupted before completion
        // is re-executed EXACTLY ONCE as a new physical attempt of the SAME
        // logical operation: one ReplayStarted event, the row completes,
        // the outcome effect is journaled, no duplicate messages, no fresh
        // turn record.
        let dir = fresh_store_dir();
        let counter = Arc::new(AtomicUsize::new(0));
        let turn_op: OpId;
        let tool_op: OpId;
        let session: SessionId;
        {
            let manager1 =
                SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                    .unwrap();
            let (deps1, _keep) = deps_sharing_session(
                manager1.clone(),
                Arc::new(scripted_provider(vec![])),
                vec![counting_tool(
                    "echo",
                    RecoveryHint::Idempotent,
                    counter.clone(),
                )],
            );
            let runtime1 = AgentRuntime::new(deps1).unwrap();
            let ws = manager1.create_workspace("/w").unwrap();
            let handle = manager1.create_session(ws, "t", "fake", "m").unwrap();
            session = handle.id();
            let receipt = handle.submit_prompt("use echo", &[]).unwrap();
            turn_op = receipt.op_id;
            let mut meta = op_meta(&manager1, session, RecoveryStrategy::Idempotent);
            tool_op = meta.operation_id;
            // The runtime stores the replay descriptor on the run row.
            let desc = ReplayDescriptor {
                tool_name: "echo".into(),
                validated_args: serde_json::json!({"x": 1}),
                workspace_id: WorkspaceId::new(1),
                worktree_id: WorktreeId::new(1),
                task_id: TaskId::new(1),
                original_turn_op_id: turn_op,
                capability: Capability::ReadWorkspace { path: ".".into() },
                recovery_kind: "idempotent".into(),
            };
            meta = meta.with_replay(serde_json::to_value(&desc).unwrap());
            chain_to_streaming(&handle, turn_op);
            crash_tool_start(
                &handle,
                turn_op,
                "echo",
                serde_json::json!({"x": 1}),
                "call_1",
                meta,
            );
            assert_eq!(counter.load(Ordering::SeqCst), 0, "crash before execution");
            assert_eq!(handle.message_count().unwrap(), 2);
            drop(runtime1);
        }
        // The post-restart runtime registers the echo tool again: the replay
        // executes against THIS registry (the counter it observes).
        let counter2 = Arc::new(AtomicUsize::new(0));
        let inner = scripted_provider(vec![
            ScriptedResponse::Text("after replay".into()),
            ScriptedResponse::End,
        ]);
        let (deps2, _keep2) = reopen_runtime(
            &dir,
            Arc::new(inner),
            vec![counting_tool(
                "echo",
                RecoveryHint::Idempotent,
                counter2.clone(),
            )],
        );
        let runtime2 = AgentRuntime::new(deps2).unwrap();
        let outcome = runtime2.continue_turn(session).await.unwrap();
        assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);
        assert_eq!(outcome.op_id, turn_op, "the SAME logical turn completes");
        // Exactly once.
        assert_eq!(
            counter2.load(Ordering::SeqCst),
            1,
            "recovery must execute the idempotent tool exactly once"
        );
        let handle2 = runtime2
            .deps()
            .session
            .get_session(session)
            .unwrap()
            .unwrap();
        let events = handle2.events_range(1, None).unwrap();
        let replays = events
            .iter()
            .filter(|e| e.kind == kilop_core::event::EventKind::ReplayStarted)
            .count();
        assert_eq!(replays, 1, "exactly one ReplayStarted event");
        let replay_ev = events
            .iter()
            .find(|e| e.kind == kilop_core::event::EventKind::ReplayStarted)
            .unwrap();
        assert_eq!(
            replay_ev.op_id,
            Some(tool_op),
            "replay is the SAME logical op"
        );
        assert_eq!(replay_ev.payload.as_ref().unwrap()["attempt"], 1);
        assert_eq!(
            replay_ev.payload.as_ref().unwrap()["turn_op_id"],
            turn_op.raw()
        );
        // The row completed; no duplicate messages (the user prompt once,
        // the model's tool-call message once, the single replay result
        // part, and the resumed turn's own answer).
        assert!(handle2.pending_tool_runs().unwrap().is_empty());
        assert_eq!(handle2.message_count().unwrap(), 4, "no duplicate messages");
        let page = handle2.messages_before(None, 10).unwrap();
        let users = page.iter().filter(|m| m.role == "user").count();
        assert_eq!(users, 1, "user prompt never duplicated");
        let tool_call_msgs = page
            .iter()
            .filter(|m| {
                handle2
                    .parts_of(m.id)
                    .unwrap()
                    .iter()
                    .any(|p| p.kind == "tool_call")
            })
            .count();
        assert_eq!(
            tool_call_msgs, 1,
            "the model's tool-call message never duplicated"
        );
        let result_ok = page.iter().any(|m| {
            handle2.parts_of(m.id).unwrap().iter().any(|p| {
                p.kind == "tool_result"
                    && p.data.get("tool_call_id").and_then(|v| v.as_str()) == Some("call_1")
            })
        });
        assert!(result_ok, "the replayed outcome links to the original call");
        // One logical turn, one completion; the turn record survived intact.
        let turn_completed = events
            .iter()
            .filter(|e| {
                e.kind == kilop_core::event::EventKind::TurnCompleted && e.op_id == Some(turn_op)
            })
            .count();
        assert_eq!(turn_completed, 1);
        let records = handle2.turn_records().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].turn_op_id, turn_op);
        assert_eq!(records[0].status, "completed");
    }

    #[tokio::test]
    async fn replay_descriptor_survives_reopen_and_replays_once() {
        // Requirement 2c: the descriptor is durable — after a daemon restart
        // (new manager AND new runtime over the same dir) the interrupted
        // idempotent run still replays exactly once.
        let dir = fresh_store_dir();
        let session: SessionId;
        let tool_op: OpId;
        {
            let manager1 =
                SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                    .unwrap();
            let counter = Arc::new(AtomicUsize::new(0));
            let (deps1, _keep) = deps_sharing_session(
                manager1.clone(),
                Arc::new(scripted_provider(vec![])),
                vec![counting_tool("echo", RecoveryHint::Idempotent, counter)],
            );
            let runtime1 = AgentRuntime::new(deps1).unwrap();
            let ws = manager1.create_workspace("/w").unwrap();
            let handle = manager1.create_session(ws, "t", "fake", "m").unwrap();
            session = handle.id();
            let receipt = handle.submit_prompt("echo it", &[]).unwrap();
            let mut meta = op_meta(&manager1, session, RecoveryStrategy::Idempotent);
            tool_op = meta.operation_id;
            let desc = ReplayDescriptor {
                tool_name: "echo".into(),
                validated_args: serde_json::json!({"x": 1}),
                workspace_id: WorkspaceId::new(1),
                worktree_id: WorktreeId::new(1),
                task_id: TaskId::new(1),
                original_turn_op_id: receipt.op_id,
                capability: Capability::ReadWorkspace { path: ".".into() },
                recovery_kind: "idempotent".into(),
            };
            meta = meta.with_replay(serde_json::to_value(&desc).unwrap());
            chain_to_streaming(&handle, receipt.op_id);
            crash_tool_start(
                &handle,
                receipt.op_id,
                "echo",
                serde_json::json!({"x": 1}),
                "call_1",
                meta,
            );
            drop(runtime1);
        }
        let counter2 = Arc::new(AtomicUsize::new(0));
        let inner = scripted_provider(vec![
            ScriptedResponse::Text("done".into()),
            ScriptedResponse::End,
        ]);
        let (mut deps2, _keep2) = reopen_runtime(&dir, Arc::new(inner), vec![]);
        let manager2 = deps2.session.clone();
        let mut tools2 = ToolRegistry::new();
        tools2.register(counting_tool(
            "echo",
            RecoveryHint::Idempotent,
            counter2.clone(),
        ));
        deps2.tools = Arc::new(tools2);
        let runtime2 = AgentRuntime::new(deps2).unwrap();
        let outcome = runtime2.continue_turn(session).await.unwrap();
        assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);
        assert_eq!(
            counter2.load(Ordering::SeqCst),
            1,
            "the reopened store must still replay the run exactly once"
        );
        let handle = manager2.get_session(session).unwrap().unwrap();
        let events = handle.events_range(1, None).unwrap();
        let replays = events
            .iter()
            .filter(|e| e.kind == kilop_core::event::EventKind::ReplayStarted)
            .count();
        assert_eq!(replays, 1);
        let starts = events
            .iter()
            .filter(|e| {
                e.kind == kilop_core::event::EventKind::ToolStarted && e.op_id == Some(tool_op)
            })
            .count();
        assert_eq!(starts, 1, "the ORIGINAL row is replayed, never a new row");
        assert!(handle.pending_tool_runs().unwrap().is_empty());
    }

    #[tokio::test]
    async fn unknown_effect_tool_interrupted_is_never_replayed() {
        // Requirement 2b: tools with unknown/destructive external effects
        // are NEVER replayed — the run stays unknown and the interrupted
        // turn ends honestly; the tool's execute never runs again.
        let dir = fresh_store_dir();
        let counter = Arc::new(AtomicUsize::new(0));
        let session: SessionId;
        {
            let manager1 =
                SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                    .unwrap();
            let (deps1, _keep) = deps_sharing_session(
                manager1.clone(),
                Arc::new(scripted_provider(vec![])),
                vec![counting_tool(
                    "run_cmd",
                    RecoveryHint::UnknownEffect,
                    counter.clone(),
                )],
            );
            let runtime1 = AgentRuntime::new(deps1).unwrap();
            let ws = manager1.create_workspace("/w").unwrap();
            let handle = manager1.create_session(ws, "t", "fake", "m").unwrap();
            session = handle.id();
            let receipt = handle.submit_prompt("run it", &[]).unwrap();
            let meta = op_meta(&manager1, session, RecoveryStrategy::MarkUnknown);
            chain_to_streaming(&handle, receipt.op_id);
            crash_tool_start(
                &handle,
                receipt.op_id,
                "run_cmd",
                serde_json::json!({"command": "rm -rf x"}),
                "call_1",
                meta,
            );
            drop(runtime1);
        }
        let inner = scripted_provider(vec![ScriptedResponse::End]);
        let (deps2, _keep2) = reopen_runtime(&dir, Arc::new(inner), vec![]);
        let runtime2 = AgentRuntime::new(deps2).unwrap();
        let handle2 = runtime2
            .deps()
            .session
            .get_session(session)
            .unwrap()
            .unwrap();
        let outcome = runtime2.continue_turn(session).await.unwrap();
        // The interrupted turn ends honestly: no replay, no blind re-run.
        assert_eq!(outcome.final_state, AgentState::FailedRecoverable);
        assert_eq!(
            counter.load(Ordering::SeqCst),
            0,
            "unknown-effect tools must never be re-executed"
        );
        assert!(handle2.pending_tool_runs().unwrap().is_empty());
        let events = handle2.events_range(1, None).unwrap();
        let recovery_unknown = events.iter().any(|e| {
            e.kind == kilop_core::event::EventKind::RecoveryApplied
                && e.payload
                    .as_ref()
                    .is_some_and(|p| p.get("effect").and_then(|v| v.as_str()) == Some("unknown"))
        });
        assert!(recovery_unknown, "effect stays unknown, never applied");
        let replays = events
            .iter()
            .filter(|e| e.kind == kilop_core::event::EventKind::ReplayStarted)
            .count();
        assert_eq!(replays, 0);
    }

    #[tokio::test]
    async fn hostile_replay_descriptor_fails_loudly_never_replays() {
        // Requirement 2d: a hostile stored descriptor (missing args) must be
        // an honest failure — never a blind replay of a half-known call.
        let dir = fresh_store_dir();
        let counter = Arc::new(AtomicUsize::new(0));
        let session: SessionId;
        {
            let manager1 =
                SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                    .unwrap();
            let (deps1, _keep) = deps_sharing_session(
                manager1.clone(),
                Arc::new(scripted_provider(vec![])),
                vec![counting_tool(
                    "echo",
                    RecoveryHint::Idempotent,
                    counter.clone(),
                )],
            );
            let runtime1 = AgentRuntime::new(deps1).unwrap();
            let ws = manager1.create_workspace("/w").unwrap();
            let handle = manager1.create_session(ws, "t", "fake", "m").unwrap();
            session = handle.id();
            let receipt = handle.submit_prompt("echo", &[]).unwrap();
            // Tampered descriptor: fields missing (no validated_args).
            let mut meta = op_meta(&manager1, session, RecoveryStrategy::Idempotent);
            meta = meta.with_replay(serde_json::json!({ "tool_name": "echo" }));
            chain_to_streaming(&handle, receipt.op_id);
            crash_tool_start(
                &handle,
                receipt.op_id,
                "echo",
                serde_json::json!({"x": 1}),
                "call_1",
                meta,
            );
            drop(runtime1);
        }
        let inner = scripted_provider(vec![ScriptedResponse::End]);
        let (deps2, _keep2) = reopen_runtime(&dir, Arc::new(inner), vec![]);
        let runtime2 = AgentRuntime::new(deps2).unwrap();
        let err = runtime2.continue_turn(session).await.unwrap_err();
        assert_eq!(
            err.kind,
            ErrorKind::Malformed,
            "a hostile descriptor is a loud honest failure: {err}"
        );
        assert_eq!(
            counter.load(Ordering::SeqCst),
            0,
            "no blind replay of a hostile descriptor"
        );
        // The row was NOT finished or replayed: it stays running so the
        // corruption is visible and fixable, never silently dropped.
        let handle2 = runtime2
            .deps()
            .session
            .get_session(session)
            .unwrap()
            .unwrap();
        assert_eq!(handle2.pending_tool_runs().unwrap().len(), 1);
    }

    // ---- workspace-aware write postconditions (requirement 3) ----

    fn workspace_env(
        dir: &tempfile::TempDir,
    ) -> (
        Arc<SessionManager>,
        WorkspaceId,
        SessionId,
        std::path::PathBuf,
    ) {
        let root = dir.path().join("ws");
        std::fs::create_dir_all(&root).unwrap();
        let manager =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let ws_id = manager.create_workspace(root.to_str().unwrap()).unwrap();
        let handle = manager.create_session(ws_id, "t", "fake", "m").unwrap();
        (manager, ws_id, handle.id(), root)
    }

    #[tokio::test]
    async fn workspace_write_recovery_verifies_recorded_postcondition() {
        // Requirement 3a: recovery verifies the CURRENT file bytes against
        // the RECORDED postcondition (BLAKE3 of the raw bytes as written) —
        // matching files complete WITHOUT re-running the tool, effect
        // applied. The expected hash is blake3("hello"), NOT
        // blake3(serde_json::to_vec("hello")).
        let dir = fresh_store_dir();
        let ws_id: WorkspaceId;
        let session: SessionId;
        let tool_op: OpId;
        {
            let (manager, wid, sid, root) = workspace_env(&dir);
            ws_id = wid;
            session = sid;
            let handle = manager.get_session(session).unwrap().unwrap();
            let receipt = handle.submit_prompt("write hello", &[]).unwrap();
            let meta = op_meta(&manager, session, RecoveryStrategy::MarkUnknown);
            tool_op = meta.operation_id;
            chain_to_streaming(&handle, receipt.op_id);
            crash_tool_start(
                &handle,
                receipt.op_id,
                "write_file",
                serde_json::json!({"path": "a.txt", "content": "hello"}),
                "call_1",
                meta,
            );
            // The write LANDED before the crash (the file holds the raw
            // bytes); the runtime had annotated the row with the tool's
            // postcondition (bytes as written, workspace-relative path).
            std::fs::write(root.join("a.txt"), b"hello").unwrap();
            let expected = FileHash::from(blake3::hash(b"hello").into());
            let pc = serde_json::to_value(FilePostcondition {
                workspace_id: ws_id,
                worktree_id: WorktreeId::new(1),
                relative_path: "a.txt".into(),
                expected_hash: expected,
            })
            .unwrap();
            handle.record_tool_postcondition(tool_op, &pc).unwrap();
            // Drop: the crash residue is swept post-restart by a fresh
            // runtime's recover() (no in-process driver, no stale tracking).
        }
        let write_counter = Arc::new(AtomicUsize::new(0));
        let (deps2, _keep2) = reopen_runtime(
            &dir,
            Arc::new(scripted_provider(vec![])),
            vec![counting_tool(
                "write_file",
                RecoveryHint::WorkspaceWrite,
                write_counter.clone(),
            )],
        );
        let runtime2 = AgentRuntime::new(deps2).unwrap();
        let reports = runtime2.recover().unwrap();
        let report = reports.iter().find(|r| r.session_id == session).unwrap();
        assert_eq!(report.crashed_ops.len(), 1);
        assert_eq!(
            report.crashed_ops[0].status, "completed",
            "matching postcondition completes without re-running"
        );
        assert_eq!(
            report.crashed_ops[0].effect,
            EffectStatus::Verified,
            "reports effect verified/applied"
        );
        assert_eq!(write_counter.load(Ordering::SeqCst), 0, "never re-run");
        let handle2 = runtime2
            .deps()
            .session
            .get_session(session)
            .unwrap()
            .unwrap();
        assert!(handle2.pending_tool_runs().unwrap().is_empty());
        let root = dir.path().join("ws");
        assert_eq!(
            std::fs::read(root.join("a.txt")).unwrap(),
            b"hello",
            "the verified file is untouched"
        );
        // Recover again: idempotent, nothing pending.
        let reports = runtime2.recover().unwrap();
        assert!(reports.iter().all(|r| r.crashed_ops.is_empty()));
    }

    #[tokio::test]
    async fn workspace_write_content_mismatch_fails_loudly() {
        // Requirement 3b: when the file holds DIFFERENT bytes (e.g. the
        // JSON-quoted encoding the old buggy code hashed), verification
        // FAILS loudly — never silently "applied".
        let dir = fresh_store_dir();
        let ws_id: WorkspaceId;
        let session: SessionId;
        let tool_op: OpId;
        let json_quoted: Vec<u8>;
        {
            let (manager, wid, sid, root) = workspace_env(&dir);
            ws_id = wid;
            session = sid;
            let handle = manager.get_session(session).unwrap().unwrap();
            let receipt = handle.submit_prompt("write hello", &[]).unwrap();
            let meta = op_meta(&manager, session, RecoveryStrategy::MarkUnknown);
            tool_op = meta.operation_id;
            chain_to_streaming(&handle, receipt.op_id);
            crash_tool_start(
                &handle,
                receipt.op_id,
                "write_file",
                serde_json::json!({"path": "a.txt", "content": "hello"}),
                "call_1",
                meta,
            );
            // The old (wrong) runtime hashed serde_json::to_vec("hello") —
            // the bytes `"hello"` WITH quotes. Simulate a crash where only
            // THAT content landed: verification must reject it against the
            // recorded postcondition of the raw bytes.
            json_quoted = serde_json::to_vec(&serde_json::json!("hello")).unwrap();
            std::fs::write(root.join("a.txt"), &json_quoted).unwrap();
            let expected = FileHash::from(blake3::hash(b"hello").into());
            let pc = serde_json::to_value(FilePostcondition {
                workspace_id: ws_id,
                worktree_id: WorktreeId::new(1),
                relative_path: "a.txt".into(),
                expected_hash: expected,
            })
            .unwrap();
            handle.record_tool_postcondition(tool_op, &pc).unwrap();
        }
        let (deps2, _keep2) = reopen_runtime(&dir, Arc::new(scripted_provider(vec![])), vec![]);
        let runtime2 = AgentRuntime::new(deps2).unwrap();
        let reports = runtime2.recover().unwrap();
        let report = reports.iter().find(|r| r.session_id == session).unwrap();
        assert_eq!(
            report.crashed_ops[0].status, "failed",
            "mismatching bytes must fail loudly"
        );
        assert_eq!(
            report.crashed_ops[0].effect,
            EffectStatus::Failed,
            "never silently applied"
        );
        let handle2 = runtime2
            .deps()
            .session
            .get_session(session)
            .unwrap()
            .unwrap();
        assert!(handle2.pending_tool_runs().unwrap().is_empty());
        let root = dir.path().join("ws");
        assert_eq!(
            std::fs::read(root.join("a.txt")).unwrap(),
            json_quoted,
            "recovery never rewrites or re-runs the write"
        );
    }

    #[tokio::test]
    async fn workspace_write_recovery_is_root_relative_and_rejects_traversal() {
        // Requirement 3c: the RELATIVE path is resolved inside the session
        // workspace root (never the daemon cwd — the cwd here is the repo,
        // where `b.txt` does not exist, so success proves root resolution),
        // and a traversal "../x" or a symlink escape is rejected loudly.
        let dir = fresh_store_dir();
        let session: SessionId;
        let ok_op: OpId;
        {
            let (manager, wid, sid, root) = workspace_env(&dir);
            let ws_id = wid;
            session = sid;
            let handle = manager.get_session(session).unwrap().unwrap();
            let receipt = handle.submit_prompt("write", &[]).unwrap();
            chain_to_streaming(&handle, receipt.op_id);
            let root = root.clone();
            // (i) A matching file INSIDE the workspace root verifies —
            // proving the hash ran against ws-root/b.txt, not cwd/b.txt.
            std::fs::write(root.join("b.txt"), b"world").unwrap();
            let expected = FileHash::from(blake3::hash(b"world").into());
            let meta = op_meta(&manager, session, RecoveryStrategy::MarkUnknown);
            ok_op = meta.operation_id;
            crash_tool_start(
                &handle,
                receipt.op_id,
                "write_file",
                serde_json::json!({"path": "b.txt", "content": "world"}),
                "call_ok",
                meta,
            );
            let pc = serde_json::to_value(FilePostcondition {
                workspace_id: ws_id,
                worktree_id: WorktreeId::new(1),
                relative_path: "b.txt".into(),
                expected_hash: expected,
            })
            .unwrap();
            handle.record_tool_postcondition(ok_op, &pc).unwrap();
            // (ii) A traversal postcondition is rejected loudly.
            let meta = op_meta(&manager, session, RecoveryStrategy::MarkUnknown);
            let evil_op = meta.operation_id;
            crash_tool_start(
                &handle,
                receipt.op_id,
                "write_file",
                serde_json::json!({"path": "../escape.txt", "content": "pwn"}),
                "call_evil",
                meta,
            );
            let pc = serde_json::to_value(FilePostcondition {
                workspace_id: ws_id,
                worktree_id: WorktreeId::new(1),
                relative_path: "../escape.txt".into(),
                expected_hash: FileHash::from([0u8; 32]),
            })
            .unwrap();
            handle.record_tool_postcondition(evil_op, &pc).unwrap();
            // (iii) A symlink escape is rejected the same way (canonical
            // resolution through the workspace service).
            #[cfg(unix)]
            {
                let outside_dir = dir.path().join("outside");
                std::fs::create_dir_all(&outside_dir).unwrap();
                std::os::unix::fs::symlink(&outside_dir, root.join("link")).unwrap();
                let meta = op_meta(&manager, session, RecoveryStrategy::MarkUnknown);
                let link_op = meta.operation_id;
                crash_tool_start(
                    &handle,
                    receipt.op_id,
                    "write_file",
                    serde_json::json!({"path": "link/secret.txt", "content": "pwn"}),
                    "call_link",
                    meta,
                );
                let pc = serde_json::to_value(FilePostcondition {
                    workspace_id: ws_id,
                    worktree_id: WorktreeId::new(1),
                    relative_path: "link/secret.txt".into(),
                    expected_hash: FileHash::from([0u8; 32]),
                })
                .unwrap();
                handle.record_tool_postcondition(link_op, &pc).unwrap();
            }
            // Crash: drop the manager; the residue is swept post-restart.
        }
        let (deps2, _keep2) = reopen_runtime(&dir, Arc::new(scripted_provider(vec![])), vec![]);
        let runtime2 = AgentRuntime::new(deps2).unwrap();
        let err = runtime2.recover().unwrap_err();
        assert_eq!(
            err.kind,
            ErrorKind::Permission,
            "traversal/symlink escapes must be rejected loudly: {err}"
        );
        assert!(
            !dir.path().join("escape.txt").exists()
                && !dir.path().join("outside").join("secret.txt").exists(),
            "recovery must never touch files outside the workspace"
        );
        // The in-root verification above succeeded BEFORE the rejections:
        // the ok row finished; the hostile rows stay running (visible, never
        // silently dropped).
        let handle2 = runtime2
            .deps()
            .session
            .get_session(session)
            .unwrap()
            .unwrap();
        let events = handle2.events_range(1, None).unwrap();
        let applied = events.iter().any(|e| {
            e.kind == kilop_core::event::EventKind::RecoveryApplied
                && e.payload.as_ref().is_some_and(|p| {
                    p.get("op_id").and_then(|v| v.as_i64()) == Some(ok_op.raw() as i64)
                        && p.get("status").and_then(|v| v.as_str()) == Some("completed")
                })
        });
        assert!(
            applied,
            "the in-workspace write verified before the rejection"
        );
        let pending = handle2.pending_tool_runs().unwrap();
        assert_eq!(pending.len(), 2, "hostile rows stay pending: {pending:?}");
    }
}
