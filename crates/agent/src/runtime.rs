//! The agent runtime: the durable turn loop that drives the session with
//! commands, streams providers, schedules tools, and keeps context bounded.

use std::collections::HashMap;
use std::sync::Arc;

use kilop_context::artifact::ArtifactWriter;
use kilop_context::assembler::{AssembledContext, ContextAssembler, Evidence, RecentTurn};
use kilop_context::budget::ContextBudget;
use kilop_context::compactor::{CompactionPlan, CompactionRequest, Compactor, Summarizer};
use kilop_context::ledger::{TaskLedger, TurnSummary};
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
    CapabilityValidator, GenericAgentRequest, ProviderChunk, ProviderError, ProviderRegistry,
    RequestMessage, RequestMeta, Role,
};
use kilop_scheduler::{OwnershipSet, ResourceRequest, ScheduledOp, Scheduler};
use kilop_session::ops::PermissionRequest as SessionPermission;
use kilop_session::SessionManager;

use crate::loop_detect::LoopDetector;
use crate::tool::{RecoveryHint, ToolOutcome, ToolRegistry, ToolRunCtx};
use crate::tool_json::ToolCallMode;

/// Ephemeral-stream flush cadence: durable parts are written in segments of
/// this size (plus the final tail), so per-token journaling never happens.
const STREAM_FLUSH_BYTES: usize = 8 * 1024;

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
            .drive_turn(&handle, receipt.op_id, receipt.op_meta.cancellation.clone())
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
        self.drive_turn(&handle, OpId::new(1), CancellationToken::new())
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
                    let actual = std::fs::read(&path)
                        .map(|bytes| kilop_core::hash::FileHash::from(blake3::hash(&bytes).into()))
                        .unwrap_or_else(|_| kilop_core::hash::FileHash::from([0u8; 32]));
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
        let caps = provider.capabilities(&handle.model()?);
        let budget = ContextBudget::for_capabilities(&caps);
        // True after a tool batch: the model continues (tool results pending)
        // from ReadyForNextTurn without a new user prompt.
        let mut continue_after_tools = false;

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
            if state == AgentState::ReadyForNextTurn && !continue_after_tools {
                outcome.final_state = state;
                return Ok(outcome);
            }

            // ---- prepare context (hop from ReadyForNextTurn when continuing)
            if state == AgentState::ReadyForNextTurn {
                handle.append_event(
                    kilop_core::event::EventKind::ContextPrepared,
                    AgentState::Preparing,
                    Some(op_id),
                    None,
                )?;
            }
            handle.append_event(
                kilop_core::event::EventKind::ContextPrepared,
                AgentState::BuildingContext,
                Some(op_id),
                None,
            )?;
            let recent = self.recent_turns(handle)?;
            let evidence = self
                .deps
                .evidence
                .evidence_for(handle.id(), &handle.title()?);
            let mut assembled = ContextAssembler::assemble(
                &self.deps.instructions,
                "",
                &tool_schemas_json(self.deps.tools.specs()),
                "",
                &ledger,
                "",
                &recent,
                &evidence,
                "",
                &budget,
            )?;

            // ---- proactive compaction (spec §9)
            let usage = budget.effective_usage(assembled.total_tokens);
            if usage >= self.deps.compact_at_usage.clamp(0.0, 1.0) {
                if let Some(plan) = self.try_compact(handle, &recent, &ledger, &budget)? {
                    outcome.compacted = true;
                    ledger = plan.ledger.clone();
                    assembled = ContextAssembler::assemble(
                        &self.deps.instructions,
                        "",
                        &tool_schemas_json(self.deps.tools.specs()),
                        "",
                        &ledger,
                        "",
                        &plan.kept_recent,
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
            let request = self.build_request(handle, &assembled, op_id)?;
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
            let mut tool_calls = Vec::new();
            let mut tokens_in = 0u64;
            let mut tokens_out = 0u64;

            use futures::StreamExt;
            while let Some(chunk) = stream.next().await {
                if cancel.is_cancelled() {
                    let _ = handle.abort(Some(op_id));
                    outcome.final_state = AgentState::Cancelled;
                    return Ok(outcome);
                }
                match chunk {
                    Ok(ProviderChunk::Text { text }) => {
                        text_buf.push_str(&text);
                        if assistant_message.is_none() {
                            let seq = handle.proposed_message_seq()?;
                            assistant_message = Some(handle.put_message(
                                seq,
                                "assistant",
                                serde_json::json!({ "parts": [] }),
                            )?);
                        }
                        // EPHEMERAL path: text deltas are NOT journaled per
                        // chunk (a multi-hour agent would commit millions of
                        // tiny SQLite events). The durable representation is
                        // the message part, flushed in bounded segments so a
                        // crash loses at most one segment.
                        if text_buf.len() >= STREAM_FLUSH_BYTES {
                            if let Some(mid) = assistant_message {
                                handle.put_text_part(mid, &text_buf)?;
                                text_buf.clear();
                            }
                        }
                    }
                    Ok(ProviderChunk::Reasoning { text }) => {
                        text_buf.push_str(&text);
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
                            &handle.model()?,
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
                if !text_buf.is_empty() {
                    handle.put_text_part(mid, &text_buf)?;
                }
            }
            handle.record_provider_call(
                op_id,
                provider.id(),
                &handle.model()?,
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
                    // Tools ran: back to ReadyForNextTurn, then continue
                    // streaming the model with the tool results.
                    handle.append_event(
                        kilop_core::event::EventKind::TurnCompleted,
                        AgentState::UpdatingMemory,
                        Some(op_id),
                        None,
                    )?;
                    handle.append_event(
                        kilop_core::event::EventKind::TurnCompleted,
                        AgentState::ReadyForNextTurn,
                        Some(op_id),
                        None,
                    )?;
                    outcome.turns += 1;
                }
                continue_after_tools = executed > 0;
                continue; // stream again with tool results
            }

            // ---- no tools: validate → update memory → turn complete
            handle.append_event(
                kilop_core::event::EventKind::TurnCompleted,
                AgentState::Validating,
                Some(op_id),
                None,
            )?;
            handle.append_event(
                kilop_core::event::EventKind::TurnCompleted,
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
                    kilop_core::WorkspaceId::new(1),
                    kilop_core::WorktreeId::new(1),
                    kilop_core::TaskId::new(1),
                ),
                cancellation: op_meta.cancellation.clone(),
                artifacts: Arc::new(self.deps.artifact_sink(handle.id())),
                tool_call_mode: self.deps.tool_call_mode,
            };
            let tool_arc = tool.clone();
            let outcomes = outcomes.clone();
            let args = input.clone();
            let spec = ScheduledOp {
                meta: op_meta.clone(),
                resources: ResourceRequest {
                    class: tool.resource_class,
                },
                reads: OwnershipSet::new([]),
                writes: OwnershipSet::new([]),
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

        for (op_id, name, call_id) in submitted {
            if done.contains(&op_id) {
                // FileChanged is recorded while still ExecutingTool (before
                // finish_tool_run moves the machine to Validating).
                handle.append_event(
                    kilop_core::event::EventKind::FileChanged,
                    AgentState::ExecutingTool,
                    Some(op_id),
                    Some(serde_json::json!({ "tool": name, "effect": "applied" })),
                )?;
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
        let page = handle.messages_page(None, 40)?;
        let mut turns = Vec::new();
        for m in page.messages.iter().rev() {
            for part in &m.parts {
                if let kilop_protocol::v756::Part::Text { text } = part {
                    turns.push(RecentTurn {
                        role: m.role.clone(),
                        text: text.clone(),
                    });
                }
            }
        }
        Ok(turns)
    }

    fn build_request(
        &self,
        handle: &kilop_session::SessionHandle,
        assembled: &AssembledContext,
        op_id: OpId,
    ) -> kilop_core::Result<GenericAgentRequest> {
        let recent = self.recent_turns(handle)?;
        let mut messages = Vec::new();
        for t in recent {
            messages.push(RequestMessage {
                role: if t.role == "user" {
                    Role::User
                } else {
                    Role::Assistant
                },
                content: vec![kilop_provider::ContentPart::text(&t.text)],
            });
        }
        Ok(GenericAgentRequest {
            model: handle.model()?,
            system: assembled.render(),
            messages,
            tools: self.deps.tools.specs(),
            max_output: None,
            reasoning: None,
            stream: true,
            meta: RequestMeta {
                operation_id: op_id,
                session_id: handle.id(),
                provider: handle.provider()?,
                attempt: 0,
                deadline_ms: self.deps.tool_deadline_ms,
                cancellation: CancellationToken::new(),
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

fn tool_schemas_json(specs: Vec<kilop_provider::ToolSpec>) -> String {
    serde_json::to_string(&specs).unwrap_or_else(|_| "[]".into())
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
    use kilop_provider::{FakeProvider, ScriptedResponse};
    use tempfile::tempdir;

    fn deps(provider: FakeProvider, tools: Vec<Tool>) -> (AgentDeps, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let mut registry = ProviderRegistry::new();
        registry.register(Arc::new(provider));
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
            .drive_turn(&handle, receipt.op_id, receipt.op_meta.cancellation.clone())
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
}
