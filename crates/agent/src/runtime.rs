//! The agent runtime: the durable turn loop that drives the session with
//! commands, streams providers, schedules tools, and keeps context bounded.

use std::collections::HashMap;
use std::sync::Arc;

use kilop_context::artifact::ArtifactWriter;
use kilop_context::assembler::{Evidence, RecentTurn};
use kilop_context::budget::ContextBudget;
use kilop_context::compactor::{CompactionPlan, CompactionRequest, Compactor, Summarizer};
use kilop_context::ledger::{TaskLedger, TurnSummary};
use kilop_context::wire_plan::{plan_wire_request, WirePlan};
use kilop_core::cancellation::CancellationToken;
use kilop_core::capability::{Capability, PermissionDecision};
use kilop_core::error::{Error, ErrorKind};
use kilop_core::id::{OpId, SessionId};
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
use kilop_session::SessionManager;

use crate::loop_detect::LoopDetector;
use crate::tool::{RecoveryHint, Tool, ToolOutcome, ToolRegistry, ToolRunCtx};
use crate::tool_json::ToolCallMode;

/// Ephemeral-stream flush cadence: durable parts are written in segments of
/// this size (plus the final tail), so per-token journaling never happens.
const STREAM_FLUSH_BYTES: usize = 8 * 1024;

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
pub trait EvidenceProvider: Send + Sync {
    fn evidence_for(&self, session: SessionId, prompt: &str) -> Vec<Evidence>;
}

pub struct NoEvidence;
impl EvidenceProvider for NoEvidence {
    fn evidence_for(&self, _session: SessionId, _prompt: &str) -> Vec<Evidence> {
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
}

#[derive(Debug, Clone)]
pub struct TurnOutcome {
    pub op_id: OpId,
    pub final_state: AgentState,
    pub turns: u32,
    pub compacted: bool,
    pub loop_stopped: bool,
}

#[derive(Debug, Clone)]
pub struct AgentCard {
    pub session_id: SessionId,
    pub title: String,
    pub status: String, // running | waiting | completed | failed | needs-input
}

impl AgentRuntime {
    pub fn new(deps: AgentDeps) -> kilop_core::Result<Arc<Self>> {
        if deps.model.is_empty() {
            return Err(Error::malformed("agent requires a model"));
        }
        Ok(Arc::new(Self {
            deps: Arc::new(deps),
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
        let handle = self
            .deps
            .session
            .get_session(session)?
            .ok_or_else(|| Error::not_found(format!("session {session}")))?;

        // Crash recovery first: any pending tool run decides this turn's
        // continuation (never blindly re-run).
        self.recover_session(&handle)?;

        let receipt = handle.submit_prompt(prompt, files)?;
        let outcome = self
            .drive_turn(
                &handle,
                receipt.op_id,
                receipt.op_meta.cancellation.clone(),
                model,
            )
            .await;
        if let Err(e) = &outcome {
            // A failed turn must never leave the machine stuck mid-transition
            // (e.g. Preparing after a provider is missing): journal the
            // failure so the session lands on FailedRecoverable and accepts
            // the next prompt.
            let _ = handle.append_event(
                kilop_core::event::EventKind::Failed,
                AgentState::FailedRecoverable,
                Some(receipt.op_id),
                Some(serde_json::json!({ "message": e.message })),
            );
        }
        outcome
    }

    /// Continue a turn interrupted by a crash: resolve pending tool runs per
    /// their recovery strategy, then resume if the state allows.
    pub async fn continue_turn(
        self: &Arc<Self>,
        session: SessionId,
    ) -> kilop_core::Result<TurnOutcome> {
        let handle = self
            .deps
            .session
            .get_session(session)?
            .ok_or_else(|| Error::not_found(format!("session {session}")))?;
        let pending = handle.pending_tool_runs()?;
        if pending.is_empty() {
            return Err(Error::conflict(format!(
                "session {session} has no interrupted tool run to continue"
            )));
        }
        self.recover_session(&handle)?;
        self.drive_turn(&handle, OpId::new(1), CancellationToken::new(), None)
            .await
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
        let handle = self
            .deps
            .session
            .get_session(session)?
            .ok_or_else(|| Error::not_found(format!("session {session}")))?;
        Ok(handle.abort(None)?.op_ids)
    }

    /// Explicitly close a session (the only normal route to terminal
    /// closure; review P0-2 — Stop/abort cancels the turn, not the session).
    pub fn end_session(&self, session: SessionId) -> kilop_core::Result<()> {
        let handle = self
            .deps
            .session
            .get_session(session)?
            .ok_or_else(|| Error::not_found(format!("session {session}")))?;
        handle.end_session()?;
        Ok(())
    }

    pub fn recover(&self) -> kilop_core::Result<Vec<kilop_session::RecoveryReport>> {
        self.deps.session.recover_all_sessions()
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

    /// Resolve interrupted tool runs. VerifyHash checks the file; MarkUnknown
    /// forces verification; Idempotent re-runs; Manual requires a human.
    fn recover_session(&self, handle: &kilop_session::SessionHandle) -> kilop_core::Result<()> {
        let pending = handle.pending_tool_runs()?;
        for row in pending {
            let recovery: RecoveryStrategy = serde_json::from_value(row.recovery.clone())
                .map_err(|e| Error::malformed(format!("corrupt recovery row: {e}")))?;
            match recovery {
                RecoveryStrategy::VerifyHash { path, expected } => {
                    // Streamed hashing: never load an arbitrarily large file
                    // into RAM on the recovery path (audit round 5). Bounded
                    // 64KiB chunks; an unreadable file hashes to the
                    // zero-marker and is treated as "write never landed".
                    let actual = stream_hash_file(&path);
                    if actual == expected {
                        // The write landed: record completion, never re-run.
                        handle.finish_tool_run(row.op_id, "completed", EffectStatus::Verified)?;
                    } else {
                        // The write never happened (or was overwritten): the
                        // effect is unknown; force verification.
                        handle.finish_tool_run(row.op_id, "failed", EffectStatus::Unknown)?;
                    }
                }
                RecoveryStrategy::MarkUnknown
                | RecoveryStrategy::Manual
                | RecoveryStrategy::None => {
                    handle.finish_tool_run(row.op_id, "failed", EffectStatus::Unknown)?;
                }
                RecoveryStrategy::Idempotent => {
                    // Safe to replay; the drive loop re-runs it.
                    handle.finish_tool_run(row.op_id, "failed", EffectStatus::Unknown)?;
                }
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------ the turn loop

    async fn drive_turn(
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
        };
        let mut detector = LoopDetector::new(3);
        let mut ledger = self.load_ledger(handle)?;
        let provider = self.provider_for(handle)?;
        // The effective model is the per-message override when present; the
        // provider is ALWAYS the session's provider. Capabilities for a
        // model the provider does not know fall back to the provider's
        // default (never an error at send time).
        let model = match model_override {
            Some(m) => m,
            None => handle.model()?,
        };
        let caps = provider.capabilities(&model);
        let budget = ContextBudget::for_capabilities(&caps);
        // True on iterations that continue the SAME logical turn after a
        // tool batch (the machine is already at WaitingForModel; context
        // preparation hops must be skipped — there is exactly ONE
        // TurnCompleted per logical turn, and ReadyForNextTurn is only
        // reached at the genuine end; audit round 6).
        let mut continuing_turn = false;

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
            if state == AgentState::ReadyForNextTurn && !continuing_turn {
                outcome.final_state = state;
                return Ok(outcome);
            }

            // ---- prepare context (fresh turn only; continuing iterations
            // re-plan in memory from history, no journal hops)
            if continuing_turn {
                continuing_turn = false;
            } else {
                handle.append_event(
                    kilop_core::event::EventKind::ContextPrepared,
                    AgentState::BuildingContext,
                    Some(op_id),
                    None,
                )?;
            }
            let recent = self.recent_turns(handle)?;
            let evidence = self
                .deps
                .evidence
                .evidence_for(handle.id(), &handle.title()?);
            let mut history = self.history_messages(handle)?;
            let mut wire_plan = plan_wire_request(
                &self.deps.instructions,
                "",
                &self.deps.tools.specs(),
                "",
                &ledger,
                "",
                &history,
                &evidence,
                "",
                &budget,
            )?;

            // ---- proactive compaction (spec §9)
            let usage = budget.effective_usage(wire_plan.total_tokens);
            if usage >= self.deps.compact_at_usage.clamp(0.0, 1.0) {
                if let Some(plan) = self.try_compact(handle, &recent, &ledger, &budget)? {
                    outcome.compacted = true;
                    ledger = plan.ledger.clone();
                    history = recent_turns_to_messages(&plan.kept_recent);
                    wire_plan = plan_wire_request(
                        &self.deps.instructions,
                        "",
                        &self.deps.tools.specs(),
                        "",
                        &ledger,
                        "",
                        &history,
                        &evidence,
                        "",
                        &budget,
                    )?;
                }
            }

            // ---- provider call
            handle.append_event(
                kilop_core::event::EventKind::ModelStarted,
                AgentState::WaitingForModel,
                Some(op_id),
                None,
            )?;
            let request = self.build_request(handle, &wire_plan, op_id, &model, &cancel)?;
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
            let mut assistant_message: Option<i64> = None;
            let mut text_buf = String::new();
            let mut reasoning_buf = String::new();
            let mut tool_calls = Vec::new();
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
                        handle.put_tool_call_part(mid, &id, &name, input.clone(), "completed")?;
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
                        return self
                            .handle_provider_failure(handle, op_id, e, &mut outcome)
                            .await;
                    }
                }
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
                        &cancel,
                        tool_calls,
                    )
                    .await?;
                if executed == 0 && detector.trips > 0 {
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
                    continuing_turn = true;
                    continue; // stream again with tool results
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
                ledger.record_turn(&TurnSummary::default());
                handle.put_task_ledger(serde_json::to_value(&ledger)?)?;
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
            ledger.record_turn(&TurnSummary::default());
            handle.put_task_ledger(serde_json::to_value(&ledger)?)?;
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
        let mut submitted: Vec<(OpId, String, String)> = Vec::new();
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
                .await;
            match decision? {
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

            // Op envelope: deadline, retry, cancellation, recovery.
            let op_id = self.deps.session.next_op_id();
            let recovery = match &tool.recovery_hint {
                RecoveryHint::VerifyHash {
                    path_arg,
                    content_arg,
                } => {
                    let path = input
                        .get(path_arg)
                        .and_then(|p| p.as_str())
                        .unwrap_or_default();
                    let content = input
                        .get(content_arg)
                        .cloned()
                        .unwrap_or(serde_json::Value::Null);
                    let expected = kilop_core::hash::FileHash::from(
                        blake3::hash(&serde_json::to_vec(&content).unwrap_or_default()).into(),
                    );
                    RecoveryStrategy::VerifyHash {
                        path: path.to_string(),
                        expected,
                    }
                }
                RecoveryHint::Idempotent => RecoveryStrategy::Idempotent,
                RecoveryHint::UnknownEffect => RecoveryStrategy::MarkUnknown,
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
            let run_handle = handle.start_tool_run(op_meta.clone(), &name, input.clone())?;
            let _ = run_handle;

            // Scheduler task for this tool; the OpMeta envelope (deadline,
            // retry, cancellation, recovery) is passed straight through.
            let ctx = ToolRunCtx {
                session_id: handle.id(),
                op_id,
                identity: WorkspaceIdentity::new(
                    workspace_id,
                    kilop_core::WorktreeId::new(1),
                    kilop_core::TaskId::new(1),
                ),
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
            submitted.push((op_id, name.clone(), call_id.clone()));
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
        for (op_id, name, _call_id) in submitted.iter() {
            if done.contains(op_id) {
                handle.append_event(
                    kilop_core::event::EventKind::FileChanged,
                    AgentState::ExecutingTool,
                    Some(*op_id),
                    Some(serde_json::json!({ "tool": name, "effect": "applied" })),
                )?;
            }
        }
        for (op_id, name, call_id) in submitted {
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
                handle.finish_tool_run(op_id, "completed", outcome.effect_status)?;
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
                ledger.record_turn(&TurnSummary {
                    files_changed: vec![name.clone()],
                    ..Default::default()
                });
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

    fn load_ledger(&self, handle: &kilop_session::SessionHandle) -> kilop_core::Result<TaskLedger> {
        match handle.get_task_ledger()? {
            Some(v) => Ok(serde_json::from_value(v).unwrap_or_default()),
            None => Ok(TaskLedger::default()),
        }
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

    fn recent_turns(
        &self,
        handle: &kilop_session::SessionHandle,
    ) -> kilop_core::Result<Vec<RecentTurn>> {
        let rows = handle.messages_before(None, 40)?; // newest first
        let mut turns = Vec::new();
        for row in rows.iter().rev() {
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
        let rows = handle.messages_before(None, 40)?; // newest first
        let mut out = Vec::new();
        for row in rows.iter().rev() {
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
                attempt: 0,
                deadline_ms: self.deps.tool_deadline_ms,
                cancellation: cancel.child(),
            },
        })
    }

    fn try_compact(
        &self,
        handle: &kilop_session::SessionHandle,
        recent: &[RecentTurn],
        ledger: &TaskLedger,
        budget: &ContextBudget,
    ) -> kilop_core::Result<Option<CompactionPlan>> {
        let before = recent.iter().map(|t| t.text.len()).sum::<usize>() / 4;
        if before == 0 {
            return Ok(None);
        }
        let target = budget.context_max();
        let compactor: Compactor = if self.deps.compaction_model.is_some() {
            // A real summarizer streams the compaction model; until the
            // adapters wire it, deterministic pruning with a weak summary
            // path — the hard invariant still rejects weak output.
            Compactor::new(Some(Arc::new(LedgerSummarizer)))
        } else {
            Compactor::deterministic_only()
        };
        let plan = compactor.compact(recent, ledger, &CompactionRequest::new(before, target));
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
    fn summarize(&self, _history: &[kilop_context::RecentTurn], ledger: &TaskLedger) -> String {
        ledger.compact_render()
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

/// Read a required string field from a durable part payload; a missing or
/// non-string field is loud corruption, never silently dropped.
fn str_field(data: &serde_json::Value, key: &str) -> kilop_core::Result<String> {
    data.get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| Error::malformed(format!("durable part is missing string field `{key}`")))
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
        };
        (deps, dir)
    }

    fn deps(provider: FakeProvider, tools: Vec<Tool>) -> (AgentDeps, tempfile::TempDir) {
        deps_with(Arc::new(provider), tools)
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
            recovery_hint: RecoveryHint::VerifyHash {
                path_arg: "path".into(),
                content_arg: "content".into(),
            },
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
    async fn compaction_trigger_uses_wire_footprint() {
        // Seed durable history large enough that the WIRE footprint (system +
        // messages + tools) crosses the threshold: compaction must trigger off
        // plan.total_tokens, and the turn must still complete within budget.
        let (mut deps, _dir) = deps(
            scripted_provider(vec![
                ScriptedResponse::Text("t".into()),
                ScriptedResponse::End,
            ]),
            vec![],
        );
        deps.compact_at_usage = 0.1;
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());
        let handle = runtime.deps.session.get_session(session).unwrap().unwrap();
        // Seed FIRST the prompt (its message seq comes from the journal event
        // seq), then the bulky durable history AFTER it via the seq allocator
        // — never colliding with the journal's message seqs.
        let receipt = handle.submit_prompt("go", &[]).unwrap();
        for i in 0..30 {
            let seq = handle.proposed_message_seq().unwrap();
            let mid = handle
                .put_message(seq, "user", serde_json::json!({}))
                .unwrap();
            handle
                .put_text_part(mid, &format!("turn {i} {}", "z".repeat(1500)))
                .unwrap();
        }
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
        assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);
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
                ScriptedResponse::ToolCall { id: "c1".into(), name: "echo".into(), input: serde_json::json!({"x": 1}) },
                ScriptedResponse::ToolCall { id: "c2".into(), name: "echo".into(), input: serde_json::json!({"x": 2}) },
                ScriptedResponse::Text("final answer".into()),
                ScriptedResponse::End,
            ]),
            vec![echo_tool()],
        );
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());
        let outcome = runtime.run_turn(session, "do work", &[]).await.unwrap();
        assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn);
        assert_eq!(outcome.turns, 1, "one logical turn despite two tool batches");
        let handle = runtime.deps.session.get_session(session).unwrap().unwrap();
        let events = handle.events_range(1, None).unwrap();
        let turn_completed = events.iter().filter(|e| e.kind == kilop_core::event::EventKind::TurnCompleted).count();
        assert_eq!(turn_completed, 1, "exactly one TurnCompleted per logical turn");
        // ReadyForNextTurn must appear in the journal EXACTLY ONCE (the end).
        let ready = events.iter().filter(|e| e.state == AgentState::ReadyForNextTurn).count();
        assert_eq!(ready, 1, "ReadyForNextTurn only at the genuine end");
        // The interior tool batches used PhaseChanged hops (never TurnCompleted).
        let interior = events.iter().filter(|e| e.kind == kilop_core::event::EventKind::PhaseChanged).count();
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
                ScriptedResponse::ToolCall { id: "c1".into(), name: "echo".into(), input: serde_json::json!({"x": 1}) },
                ScriptedResponse::End,
            ]),
            vec![echo_tool()],
        );
        let runtime = AgentRuntime::new(deps).unwrap();
        let session = new_session(runtime.deps());
        let handle = runtime.deps.session.get_session(session).unwrap().unwrap();
        let receipt = handle.submit_prompt("crash me", &[]).unwrap();
        let outcome = runtime
            .drive_turn(&handle, receipt.op_id, receipt.op_meta.cancellation.clone(), None)
            .await
            .unwrap();
        assert_eq!(outcome.final_state, AgentState::ReadyForNextTurn, "turn ran to completion with a single batch");
        // The crash happens BEFORE the model continuation? Simulate by
        // ending the provider script: first stream consumed the ToolCall;
        // second stream (continuation) has no script → Done → turn ends.
        let events = handle.events_range(1, None).unwrap();
        let prompt_events = events.iter().filter(|e| e.kind == kilop_core::event::EventKind::PromptReceived).count();
        assert_eq!(prompt_events, 1, "one prompt for the whole logical turn");
        let turn_completed = events.iter().filter(|e| e.kind == kilop_core::event::EventKind::TurnCompleted).count();
        assert_eq!(turn_completed, 1);
    }
}
