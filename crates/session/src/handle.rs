//! `SessionHandle`: every durable command a session accepts, journaled and
//! state-machine-validated.

use std::sync::Arc;

use kilop_core::event::{Event, EventKind};
use kilop_core::id::{EventSeq, OpId, SessionId};
use kilop_core::op::{OpMeta, RecoveryStrategy};
use kilop_core::state::{AgentState, SessionLifecycle};
use kilop_core::time::Deadline;
use kilop_store::{SessionRow, SessionTransition};

use crate::journal::{replay, ReplayOutcome};
use crate::manager::SessionManager;
use crate::ops::OpRegistry;
use crate::process::ProcessRegistry;
use crate::recovery::SystemFileHasher;
use crate::{SessionError, TURN_DEADLINE_MS};

/// A handle to one session. Cheap to clone; all handles to the same session
/// share one cancellation/process registry through the manager.
#[derive(Debug, Clone)]
pub struct SessionHandle {
    pub(crate) manager: Arc<SessionManager>,
    pub(crate) id: SessionId,
    /// Shared per-session registries; every handle clone sees the same ops
    /// and process ownership.
    pub(crate) resources: Arc<crate::manager::SessionResources>,
    pub(crate) system_hasher: Arc<SystemFileHasher>,
}

/// Receipt of an accepted (or queued) prompt.
#[derive(Debug, Clone)]
pub struct PromptReceipt {
    pub op_id: OpId,
    /// The full operation envelope for the turn (deadline, retry, recovery,
    /// cancellation token).
    pub op_meta: OpMeta,
    /// Sequence of the `PromptReceived` journal event.
    pub event_seq: EventSeq,
    /// Durable message row id carrying the prompt text.
    pub message_id: i64,
    pub accepted: bool,
    pub queued: bool,
}

/// Receipt of an abort.
#[derive(Debug, Clone)]
pub struct AbortReceipt {
    pub op_ids: Vec<OpId>,
    pub event_seq: EventSeq,
    pub cancelled_all: bool,
}

/// States from which a new prompt may start a fresh turn (every transition to
/// `Preparing` is legal in core).
const PROMPTABLE: &[AgentState] = &[
    AgentState::Idle,
    AgentState::ReadyForNextTurn,
    AgentState::NeedsUserInput,
    AgentState::FailedRecoverable,
    // A cancelled TURN leaves the chat usable (Stop cancels the turn).
    AgentState::Cancelled,
    AgentState::Suspended,
];

/// States that mean "an operation is in flight" (used by crash recovery).
pub(crate) fn is_op_active(s: AgentState) -> bool {
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

impl SessionHandle {
    pub(crate) fn new(
        manager: Arc<SessionManager>,
        id: SessionId,
        resources: Arc<crate::manager::SessionResources>,
        system_hasher: Arc<SystemFileHasher>,
    ) -> Self {
        Self {
            manager,
            id,
            resources,
            system_hasher,
        }
    }

    // ---------------------------------------------------------------- identity

    pub fn id(&self) -> SessionId {
        self.id
    }

    /// The session's durable workspace/worktree/task identity (v8), read
    /// fresh from the row. The agent runtime builds every `ToolRunCtx`
    /// identity from this — never a hardcoded worktree 1/task 1. Standalone
    /// sessions read 1/1 (the documented default) until
    /// `SessionManager::adopt_identity` moves them onto a real worktree.
    pub fn identity(&self) -> kilop_core::Result<kilop_core::WorkspaceIdentity> {
        let row = self.row()?;
        Ok(kilop_core::WorkspaceIdentity::new(
            row.workspace_id,
            row.worktree_id,
            row.task_id,
        ))
    }

    pub fn now_ms(&self) -> i64 {
        self.manager.now_ms()
    }

    pub(crate) fn manager(&self) -> &SessionManager {
        &self.manager
    }

    pub(crate) fn ops(&self) -> &OpRegistry {
        &self.resources.ops
    }

    pub(crate) fn processes(&self) -> &ProcessRegistry {
        &self.resources.processes
    }

    /// The per-session command lock: serializes read-validate-append
    /// transition sequences so two callers cannot validate against the same
    /// durable state and both append.
    pub(crate) fn command_guard(&self) -> std::sync::MutexGuard<'_, ()> {
        self.resources
            .command_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Validated append: transition rules are checked against the durable
    /// state while holding the command lock; an illegal transition returns
    /// `InvalidState` and leaves no trace.
    pub(crate) fn transition(
        &self,
        kind: EventKind,
        to_state: AgentState,
        op_id: Option<OpId>,
        payload: Option<serde_json::Value>,
    ) -> kilop_core::Result<EventSeq> {
        let _guard = self.command_guard();
        self.transition_locked(kind, to_state, op_id, payload)
    }

    pub(crate) fn transition_locked(
        &self,
        kind: EventKind,
        to_state: AgentState,
        op_id: Option<OpId>,
        payload: Option<serde_json::Value>,
    ) -> kilop_core::Result<EventSeq> {
        let current = self.state()?;
        crate::journal::validate_transition(current, kind, to_state)?;
        Ok(self
            .manager
            .store()
            .append_event(self.id, op_id, kind, to_state, self.now_ms(), payload)
            .map_err(crate::map_store_err)?)
    }

    pub(crate) fn system_hasher(&self) -> &Arc<SystemFileHasher> {
        &self.system_hasher
    }

    // ---------------------------------------------------------------- read state

    /// The fresh session row (title/provider/model/state) from durable store.
    pub fn row(&self) -> kilop_core::Result<SessionRow> {
        match self
            .manager
            .store()
            .get_session(self.id)
            .map_err(crate::map_store_err)?
        {
            Some(r) => Ok(r),
            None => Err(SessionError::NotFound(format!("session {}", self.id)).into()),
        }
    }

    pub fn state(&self) -> kilop_core::Result<AgentState> {
        Ok(self.row()?.state)
    }

    /// The session LIFETIME machine (Open/Suspended/Closing/Closed/...).
    /// Orthogonal to the per-turn state machine.
    pub fn lifecycle(&self) -> kilop_core::Result<kilop_core::state::SessionLifecycle> {
        Ok(self.row()?.lifecycle)
    }

    pub fn title(&self) -> kilop_core::Result<String> {
        Ok(self.row()?.title)
    }

    pub fn provider(&self) -> kilop_core::Result<String> {
        Ok(self.row()?.provider)
    }

    pub fn model(&self) -> kilop_core::Result<String> {
        Ok(self.row()?.model)
    }

    // ---------------------------------------------------------------- journal

    /// Append an event with a validated state transition. The transition is
    /// validated against the durable session state **before** anything is
    /// written: an illegal transition returns `InvalidState` and leaves no
    /// trace in the journal or the session row.
    pub fn append_event(
        &self,
        kind: EventKind,
        to_state: AgentState,
        op_id: Option<OpId>,
        payload: Option<serde_json::Value>,
    ) -> kilop_core::Result<EventSeq> {
        self.transition(kind, to_state, op_id, payload)
    }

    /// Append an event without transition validation. **Recovery/replay only**
    /// (the journal already records the sequence; validation would deadlock
    /// reconstruction). Callers must have validated against the journal first.
    pub fn force_append_event(
        &self,
        kind: EventKind,
        state: AgentState,
        op_id: Option<OpId>,
        payload: Option<serde_json::Value>,
    ) -> kilop_core::Result<EventSeq> {
        Ok(self
            .manager
            .store()
            .append_event(self.id, op_id, kind, state, self.now_ms(), payload)
            .map_err(crate::map_store_err)?)
    }

    pub fn events_after(&self, after: EventSeq) -> kilop_core::Result<Vec<Event>> {
        self.manager
            .store()
            .events_after(self.id, after)
            .map_err(|e| crate::map_store_err(e).into())
    }

    pub fn events_range(
        &self,
        from_seq: u64,
        limit: Option<u64>,
    ) -> kilop_core::Result<Vec<Event>> {
        self.manager
            .store()
            .events_range(self.id, from_seq, limit)
            .map_err(|e| crate::map_store_err(e).into())
    }

    pub fn last_event_seq(&self) -> kilop_core::Result<Option<EventSeq>> {
        self.manager
            .store()
            .last_event_seq(self.id)
            .map_err(|e| crate::map_store_err(e).into())
    }

    /// Replay the journal from durable state, enforcing the same transition
    /// rules as the live path. Corruption is a loud error. O(n) in journal
    /// length; diagnostic / startup-verification use only.
    pub fn replay_journal(&self) -> kilop_core::Result<ReplayOutcome> {
        let events = self.events_range(1, None)?;
        replay(&events).map_err(|e| e.into())
    }

    // ---------------------------------------------------------------- prompts

    /// Accept a user prompt: journals `PromptReceived`, stores the user
    /// message, registers the turn operation and transitions to `Preparing`
    /// (or records the prompt as queued when another turn is in flight).
    ///
    /// Bounds are enforced before any write: `MAX_PROMPT_BYTES`,
    /// `MAX_FILES_PER_PROMPT`, `MAX_FILE_PATH_BYTES`. Terminal sessions
    /// reject prompts with `Conflict`.
    pub fn submit_prompt(
        &self,
        prompt: &str,
        files: &[String],
    ) -> kilop_core::Result<PromptReceipt> {
        if prompt.len() > crate::MAX_PROMPT_BYTES {
            return Err(SessionError::Oversized(format!(
                "prompt of {} bytes exceeds MAX_PROMPT_BYTES",
                prompt.len()
            ))
            .into());
        }
        if files.len() > crate::MAX_FILES_PER_PROMPT {
            return Err(SessionError::Oversized(format!(
                "{} files exceed MAX_FILES_PER_PROMPT",
                files.len()
            ))
            .into());
        }
        for f in files {
            if f.len() > crate::MAX_FILE_PATH_BYTES {
                return Err(SessionError::Oversized(format!(
                    "file path of {} bytes exceeds MAX_FILE_PATH_BYTES",
                    f.len()
                ))
                .into());
            }
        }

        let _guard = self.command_guard();
        let current = self.state()?;
        if current.is_terminal() {
            return Err(SessionError::Conflict(format!(
                "session {} is {:?}; cannot accept prompts",
                self.id, current
            ))
            .into());
        }
        // The session LIFETIME machine gates prompts too: only Open sessions
        // accept them (a prompt on a Suspended session resumes it).
        let lifecycle = self.lifecycle()?;
        if !lifecycle.can_accept_prompts() {
            if lifecycle == SessionLifecycle::Suspended {
                // Conditional single UPDATE (WHERE lifecycle = Suspended);
                // the journal is intentionally untouched — auto-resume on
                // prompt is not a new event. The command lock makes the
                // expectation true in-process; the conditional write is the
                // atomic guard against anything else.
                let resumed = self
                    .manager
                    .store()
                    .set_lifecycle_if(self.id, SessionLifecycle::Suspended, SessionLifecycle::Open)
                    .map_err(crate::map_store_err)?;
                if !resumed {
                    return Err(SessionError::Conflict(format!(
                        "session {} lifecycle changed while auto-resuming; expected Suspended",
                        self.id
                    ))
                    .into());
                }
            } else {
                return Err(SessionError::Conflict(format!(
                    "session {} lifecycle is {:?}; cannot accept prompts",
                    self.id, lifecycle
                ))
                .into());
            }
        }

        let (queued, to_state) = if PROMPTABLE.contains(&current) {
            (false, AgentState::Preparing)
        } else {
            (true, current)
        };
        let op_id = self.manager.next_op_id();
        let op_meta = OpMeta::new(
            op_id,
            self.id,
            Deadline::at(self.now_ms().saturating_add(TURN_DEADLINE_MS as i64)),
            kilop_core::retry::RetryPolicy::default(),
            kilop_core::cancellation::CancellationToken::new(),
            RecoveryStrategy::None, // turns are reconstructed from the journal; never re-run blindly
            self.now_ms(),
        );

        let event_seq = self.transition_locked(
            EventKind::PromptReceived,
            to_state,
            Some(op_id),
            Some(serde_json::json!({ "queued": queued })),
        )?;
        // Deferred materialization (audit round 7): queued prompts do NOT
        // enter the conversation timeline yet — the message is created at
        // admission so chronology stays the insertion order. Immediate
        // prompts keep the existing message creation (seq == event seq).
        let message_id = if queued {
            -1
        } else {
            self.manager
                .store()
                .put_message(
                    self.id,
                    event_seq.raw() as i64,
                    "user",
                    serde_json::json!({ "text": prompt, "files": files }),
                )
                .map_err(crate::map_store_err)?
        };
        if queued {
            // Durable queue entry with the FULL execution envelope (audit
            // round 7). The user conversation message is NOT materialized
            // now — deferred materialization happens at admission so the
            // conversation chronology stays the insertion order. A queued
            // prompt is NOT a tracked machine turn and gets NO turn record
            // yet: the record is created only when the prompt is admitted as
            // the ACTIVE logical turn (never for a never-started prompt).
            self.manager
                .store()
                .enqueue_prompt(
                    self.id,
                    op_id,
                    prompt,
                    files,
                    None,
                    None,
                    None,
                    self.now_ms(),
                )
                .map_err(crate::map_store_err)?;
        } else {
            self.ops()
                .register_turn(op_id, op_meta.cancellation.clone());
            // Durable per-turn identity (v7): the moment the prompt becomes
            // the ACTIVE logical turn its exact operation id and effective
            // envelope (session provider/model; a per-message override is
            // recorded when the runtime drives the turn) are fixed. Crash
            // recovery resumes THIS record — never a synthesized op.
            let row = self.row()?;
            self.start_turn_record(
                op_id,
                None,
                Some(event_seq.raw() as i64),
                &row.provider,
                &row.model,
                None,
            )?;
        }

        Ok(PromptReceipt {
            op_id,
            op_meta,
            event_seq,
            message_id,
            accepted: true,
            queued,
        })
    }

    pub fn queue_status_counts(&self) -> kilop_core::Result<serde_json::Value> {
        Ok(self
            .manager
            .store()
            .queue_status_counts(self.id)
            .map_err(SessionError::from)?)
    }

    pub fn queued_prompt_count(&self) -> kilop_core::Result<i64> {
        let counts = self
            .manager
            .store()
            .queue_status_counts(self.id)
            .map_err(SessionError::from)?;
        let mut n = 0i64;
        for status in ["pending", "claimed", "running"] {
            n += counts.get(status).and_then(|v| v.as_i64()).unwrap_or(0);
        }
        Ok(n)
    }

    /// Atomic admission of the queue head (audit round 7): the store claims
    /// the row, materializes the user message at the conversation tail, and
    /// moves the session to Preparing in ONE transaction. The caller then
    /// journals the admission event for the same gapless sequence.
    pub fn admit_next_queued(&self) -> kilop_core::Result<Option<crate::AdmittedQueuedPrompt>> {
        let admitted = self
            .manager
            .store()
            .admit_queue_head(
                self.id,
                &[
                    "idle",
                    "ready_for_next_turn",
                    "cancelled",
                    "failed_recoverable",
                ],
                "preparing",
            )
            .map_err(SessionError::from)?;
        let Some((a, _event_seq)) = admitted else {
            return Ok(None);
        };
        // A fresh in-process token for the SAME durable op identity.
        let token = kilop_core::cancellation::CancellationToken::new();
        self.ops().register_turn(a.op_id, token);
        // Durable per-turn identity (v7): the queue row just became the
        // ACTIVE logical turn — open its record NOW with the queue's stored
        // envelope (a crash before this point leaves no phantom record; a
        // re-admission after recovery upserts the SAME record).
        let row = self.row()?;
        let model = a.model.as_deref().unwrap_or(&row.model);
        let provider = row.provider.clone();
        let variant = a.variant.clone();
        let queue_seq = Some(a.queue_seq);
        let message_seq = Some(a.message_seq);
        let op_id = a.op_id;
        let _ = self.start_turn_record(
            op_id,
            queue_seq,
            message_seq,
            &provider,
            model,
            variant.as_deref(),
        )?;
        Ok(Some(crate::AdmittedQueuedPrompt {
            queue_seq: a.queue_seq,
            op_id: a.op_id,
            prompt: a.prompt,
            files: a.files,
            model: a.model,
            variant: a.variant,
            agent: a.agent,
            message_seq: a.message_seq,
        }))
    }

    pub fn mark_queued_status(&self, queue_seq: i64, status: &str) -> kilop_core::Result<()> {
        Ok(self
            .manager
            .store()
            .mark_queue_status(self.id, queue_seq, status)
            .map_err(SessionError::from)?)
    }

    /// Durable cancellation of queued rows for the aborted ops.
    pub fn cancel_queued_ops(&self, ops: &[OpId]) -> kilop_core::Result<i64> {
        Ok(self
            .manager
            .store()
            .cancel_queued_ops(self.id, ops)
            .map_err(SessionError::from)?)
    }

    /// Recovery: claimed rows crash back to pending (re-admit later).
    pub fn recover_queued_rows(&self) -> kilop_core::Result<i64> {
        Ok(self
            .manager
            .store()
            .recover_claimed_queue_rows(self.id)
            .map_err(SessionError::from)?)
    }

    /// Abort one operation or (with `None`) every tracked operation and the
    /// session. Durable rows are updated first (tool runs become
    /// `cancelled`/`unknown`), then one journal event per affected op.
    ///
    /// Event-kind convention: tool ops journal `ToolCancelled`; turn ops
    /// journal `Failed` with `{"error": "aborted"}` (no dedicated kind exists
    /// in the frozen set). The state column is authoritative: `Cancelled`.
    pub fn abort(&self, op_id: Option<OpId>) -> kilop_core::Result<AbortReceipt> {
        let _guard = self.command_guard();
        let current = self.state()?;
        if current.is_terminal() {
            return Err(SessionError::Conflict(format!(
                "session {} is already {:?}",
                self.id, current
            ))
            .into());
        }
        // Queued prompts are NOT tracked machine turns (they are durable
        // queue rows). Killing one must durably cancel its row WITHOUT any
        // machine transition — the active turn (or idle session) is
        // untouched (audit round 7 follow-up).
        if let Some(o) = op_id {
            if self.ops().tracked(o).is_none() {
                let n = self
                    .manager
                    .store()
                    .cancel_queued_ops(self.id, &[o])
                    .map_err(crate::map_store_err)?;
                if n == 0 {
                    return Err(SessionError::NotFound(format!("operation {o}")).into());
                }
                // No journal event: the machine never saw the queued row;
                // the durable row status IS the audit trail. The receipt's
                // event_seq points at the last real journal entry. The row
                // exists, so at least its PromptReceived event exists.
                // A queue row exists, so its PromptReceived event exists
                // too; the fallback is unreachable except on an empty store.
                let last = self.last_event_seq()?.unwrap_or(EventSeq::new(1));
                return Ok(AbortReceipt {
                    op_ids: vec![o],
                    event_seq: last,
                    cancelled_all: false,
                });
            }
        }

        let affected: Vec<OpId> = match op_id {
            Some(o) => vec![o],
            None => self.ops().all(),
        };
        // abort(None) also durably cancels every queued prompt of the
        // session (pending/claimed rows).
        let queued_ids = if op_id.is_none() {
            self.manager
                .store()
                .queue_op_ids(self.id)
                .map_err(crate::map_store_err)?
        } else {
            vec![]
        };

        for o in &affected {
            self.ops().cancel(*o);
        }
        // Durable rows first; events last so a failure leaves no journal trace.
        for o in &affected {
            if self.ops().kind(*o) == Some(crate::ops::OpKind::Tool) {
                self.manager
                    .store()
                    .finish_tool_run(self.id, *o, "cancelled", "unknown")
                    .map_err(crate::map_store_err)?;
            }
        }
        let mut event_seq = None;
        for o in &affected {
            let kind = if self.ops().kind(*o) == Some(crate::ops::OpKind::Tool) {
                EventKind::ToolCancelled
            } else {
                EventKind::Failed
            };
            let payload = if kind == EventKind::ToolCancelled {
                serde_json::json!({ "op_id": o.raw() })
            } else {
                serde_json::json!({ "error": "aborted", "op_id": o.raw() })
            };
            event_seq = Some(self.transition_locked(
                kind,
                AgentState::Cancelled,
                Some(*o),
                Some(payload),
            )?);
        }
        // Nothing tracked (abort with None on an idle session) still ends it.
        if affected.is_empty() {
            event_seq = Some(self.transition_locked(
                EventKind::Failed,
                AgentState::Cancelled,
                None,
                Some(serde_json::json!({ "error": "aborted" })),
            )?);
        }
        // A cancelled turn leaves the session READY for the next prompt
        // (Stop in Kilo cancels the turn, never the session).
        if let Some(seq) = event_seq {
            event_seq = Some(self.transition_locked(
                EventKind::TurnCompleted,
                AgentState::ReadyForNextTurn,
                affected.first().copied(),
                Some(serde_json::json!({ "aborted": true })),
            )?);
            let _ = seq;
        }
        for o in &affected {
            self.ops().unregister(*o);
        }
        // abort(None): durably cancel all queued prompts too (no machine
        // transition; their row status is the audit trail).
        if !queued_ids.is_empty() {
            self.manager
                .store()
                .cancel_queued_ops(self.id, &queued_ids)
                .map_err(crate::map_store_err)?;
        }
        let mut op_ids = affected;
        for q in queued_ids {
            if !op_ids.contains(&q) {
                op_ids.push(q);
            }
        }
        Ok(AbortReceipt {
            op_ids,
            event_seq: event_seq.expect("at least one event appended"),
            cancelled_all: op_id.is_none(),
        })
    }

    /// Durable cross-turn loop signal (spec §28): same-key repeats count
    /// across logical turns and daemon restarts. True when the threshold
    /// trips (the window closes; drive_turn stops the task).
    pub fn bump_loop_signal(&self, key: &str, threshold: u32) -> kilop_core::Result<bool> {
        if key.is_empty() || key.len() > 1024 {
            return Err(
                SessionError::Oversized("loop signal key exceeds 1024 bytes".into()).into(),
            );
        }
        if !(2..=64).contains(&threshold) {
            return Err(SessionError::Malformed("loop threshold must be in [2, 64]".into()).into());
        }
        self.manager
            .store()
            .bump_loop_signal(self.id, key, threshold, self.now_ms())
            .map_err(|e| crate::map_store_err(e).into())
    }

    /// The task made progress: clear every durable loop signal.
    pub fn reset_loop_signals(&self) -> kilop_core::Result<()> {
        self.manager
            .store()
            .reset_loop_signals(self.id)
            .map_err(|e| crate::map_store_err(e).into())
    }

    // ---------------------------------------------------------------- lifecycle

    /// Suspend the session (user-initiated pause). Active states may suspend.
    /// The lifecycle move and the `Suspended` journal event are ONE atomic
    /// store transaction: a crash can never leave lifecycle Suspended without
    /// the event (or the event without the lifecycle).
    pub fn suspend(&self) -> kilop_core::Result<EventSeq> {
        let _guard = self.command_guard();
        let current = self.state()?;
        crate::journal::validate_transition(current, EventKind::Suspended, AgentState::Suspended)?;
        Ok(self
            .manager
            .store()
            .transition_session(
                self.id,
                None,
                SessionTransition {
                    expected_lifecycle: Some(SessionLifecycle::Open),
                    new_lifecycle: Some(SessionLifecycle::Suspended),
                    expected_state: Some(current),
                    new_state: AgentState::Suspended,
                    event_kind: EventKind::Suspended,
                    event_payload: None,
                },
            )
            .map_err(crate::map_store_err)?)
    }

    /// Resume a suspended session into `to` (`Idle` or `Preparing`).
    pub fn resume(&self, to: AgentState) -> kilop_core::Result<EventSeq> {
        if to != AgentState::Idle && to != AgentState::Preparing {
            return Err(
                SessionError::Malformed("resume target must be Idle or Preparing".into()).into(),
            );
        }
        let _guard = self.command_guard();
        let current = self.state()?;
        crate::journal::validate_transition(current, EventKind::Resumed, to)?;
        Ok(self
            .manager
            .store()
            .transition_session(
                self.id,
                None,
                SessionTransition {
                    expected_lifecycle: Some(SessionLifecycle::Suspended),
                    new_lifecycle: Some(SessionLifecycle::Open),
                    expected_state: Some(current),
                    new_state: to,
                    event_kind: EventKind::Resumed,
                    event_payload: None,
                },
            )
            .map_err(crate::map_store_err)?)
    }

    /// Return a failed-recoverable session to `Idle` (documented recovery
    /// outcome; legal only from `FailedRecoverable`).
    pub fn reset(&self) -> kilop_core::Result<EventSeq> {
        self.append_event(EventKind::RecoveryApplied, AgentState::Idle, None, None)
    }

    /// The in-process cancellation token registered for a turn op (None
    /// after a restart — the durable queue/registry is the source; the
    /// runner reconstructs a fresh token at admission).
    pub fn turn_cancellation(
        &self,
        op: OpId,
    ) -> Option<kilop_core::cancellation::CancellationToken> {
        self.ops().tracked(op).map(|t| t.token)
    }

    /// End the session — the ONLY normal route to terminal closure
    /// (review P0-2): journals `SessionEnded`, moves the lifecycle to Closed
    /// and the turn machine to `Completed` — all in ONE atomic store
    /// transaction. Refuses while child processes are still registered
    /// (Commandment 8 — zero orphans); ownership must transfer first.
    /// Prompts are rejected afterwards.
    pub fn end_session(&self) -> kilop_core::Result<EventSeq> {
        if !self.processes().all().is_empty() {
            return Err(SessionError::Conflict(format!(
                "session {} still owns {} child process(es); release or transfer them first",
                self.id,
                self.processes().all().len()
            ))
            .into());
        }
        let _guard = self.command_guard();
        let lifecycle = self.lifecycle()?;
        if lifecycle.is_terminal() {
            return Err(SessionError::Conflict(format!(
                "session {} is already {:?}",
                self.id, lifecycle
            ))
            .into());
        }
        let current = self.state()?;
        // Same journal legality as before (SessionEnded must land on
        // Completed from the current turn state) — validated BEFORE any
        // write; the store's expected_* checks are the atomic guard.
        crate::journal::validate_transition(
            current,
            EventKind::SessionEnded,
            AgentState::Completed,
        )?;
        Ok(self
            .manager
            .store()
            .transition_session(
                self.id,
                None,
                SessionTransition {
                    expected_lifecycle: Some(lifecycle),
                    new_lifecycle: Some(SessionLifecycle::Closed),
                    expected_state: Some(current),
                    new_state: AgentState::Completed,
                    event_kind: EventKind::SessionEnded,
                    event_payload: None,
                },
            )
            .map_err(crate::map_store_err)?)
    }

    /// Mark the session failed. `permanent: true` uses the documented
    /// two-step escalation (`FailedRecoverable` legality, then force).
    pub fn mark_failed(&self, permanent: bool, message: &str) -> kilop_core::Result<EventSeq> {
        let to = if permanent {
            AgentState::FailedPermanent
        } else {
            AgentState::FailedRecoverable
        };
        self.append_event(
            EventKind::Failed,
            to,
            None,
            Some(serde_json::json!({ "message": message })),
        )
    }

    /// Record a file change (journals `FileChanged`, state stays
    /// `ExecutingTool`).
    pub fn record_file_change(
        &self,
        path: &str,
        hash: Option<&kilop_core::hash::FileHash>,
    ) -> kilop_core::Result<EventSeq> {
        let payload = match hash {
            Some(h) => serde_json::json!({ "path": path, "hash": h.to_hex() }),
            None => serde_json::json!({ "path": path }),
        };
        self.append_event(
            EventKind::FileChanged,
            AgentState::ExecutingTool,
            None,
            Some(payload),
        )
    }

    // --------------------------------------------- durable metadata + removal

    /// Max session-title length in chars (after control stripping).
    pub const MAX_TITLE_CHARS: usize = 200;

    /// Durable session-title update (session.update, P1). Control characters
    /// are stripped first; the result must be 1..=`MAX_TITLE_CHARS` chars or
    /// the update refuses BEFORE any write (malformed when empty, oversized
    /// beyond the bound). Unknown sessions are `NotFound`. The store row is
    /// updated with a bumped `updated_ms`; the journal is untouched (a title
    /// is session metadata, not a state-machine transition).
    pub fn update_session_title(&self, title: &str) -> kilop_core::Result<()> {
        self.row()?; // existence re-verified against the durable row
        let cleaned: String = title.chars().filter(|c| !c.is_control()).collect();
        let chars = cleaned.chars().count();
        if chars == 0 {
            return Err(SessionError::Malformed(
                "session title must be 1..=200 chars after control characters are stripped".into(),
            )
            .into());
        }
        if chars > Self::MAX_TITLE_CHARS {
            return Err(SessionError::Oversized(format!(
                "session title of {chars} chars exceeds the {} char bound",
                Self::MAX_TITLE_CHARS
            ))
            .into());
        }
        let updated = self
            .manager
            .store()
            .update_session_title(self.id, &cleaned)
            .map_err(crate::map_store_err)?;
        if !updated {
            return Err(SessionError::NotFound(format!("session {}", self.id)).into());
        }
        Ok(())
    }

    /// One page of the other-message reference scan (bounded everything).
    const REFERENCE_SCAN_PAGE: u64 = 100;

    /// Durably remove ONE message and its parts (deleteMessage, P1) with the
    /// documented safety rules:
    ///
    /// - `NotFound` when the message (identity = durable sequence) does not
    ///   exist.
    /// - `Conflict` when the session has an ACTIVE turn (the machine state is
    ///   active or an active durable turn record exists) and the message is
    ///   the session's NEWEST message — that is the in-flight message being
    ///   streamed (or the active turn's own just-materialized prompt).
    /// - `Conflict` when any part of the message is a `tool_result`, or any
    ///   of its `tool_call` parts is referenced by another message's
    ///   `tool_result` (a tool_call/tool_result pairing must never be torn).
    ///
    /// The store deletes the message row and its part rows in ONE
    /// transaction; message sequences stay stable (paging skips the hole,
    /// never renumbers).
    pub fn delete_message(&self, seq: i64) -> kilop_core::Result<()> {
        let store = self.manager.store();
        // Existence first (identity = durable sequence, same surface as
        // revert/diff/deleteMessage).
        let mut rows = store
            .messages_before(self.id, Some(seq + 1), 1)
            .map_err(crate::map_store_err)?;
        let row = rows.pop().filter(|r| r.seq == seq);
        let Some(row) = row else {
            return Err(
                SessionError::NotFound(format!("message {seq} of session {}", self.id)).into(),
            );
        };
        // In-flight refusal: an active turn owns the newest message.
        let session_row = self.row()?;
        let turn_active = session_row.state.is_active()
            || store
                .active_turn_record(self.id)
                .map_err(crate::map_store_err)?
                .is_some();
        if turn_active {
            let newest_seq = store
                .messages_before(self.id, None, 1)
                .map_err(crate::map_store_err)?
                .first()
                .map(|r| r.seq);
            if newest_seq == Some(seq) {
                return Err(SessionError::Conflict(format!(
                    "deleteMessage refused: message {seq} is the active turn's newest message (in flight); stop the turn before removing it"
                ))
                .into());
            }
        }
        // Tool pairing: a result part on the message, or a call part that
        // some other message's tool_result references.
        let parts = store.parts_of(row.id).map_err(crate::map_store_err)?;
        let has_result = parts.iter().any(|p| p.kind == "tool_result");
        let call_ids: Vec<String> = parts
            .iter()
            .filter(|p| p.kind == "tool_call")
            .filter_map(|p| {
                p.data
                    .get("tool_call_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
            })
            .collect();
        let referenced_elsewhere = if call_ids.is_empty() {
            false
        } else {
            let mut cursor: Option<i64> = None;
            let mut referenced = false;
            'scan: loop {
                let page = store
                    .messages_before(self.id, cursor, Self::REFERENCE_SCAN_PAGE)
                    .map_err(crate::map_store_err)?;
                if page.is_empty() {
                    break;
                }
                for m in &page {
                    if m.id == row.id {
                        continue;
                    }
                    for p in store.parts_of(m.id).map_err(crate::map_store_err)? {
                        if p.kind == "tool_result" {
                            if let Some(tc) = p.data.get("tool_call_id").and_then(|v| v.as_str()) {
                                if call_ids.iter().any(|c| c == tc) {
                                    referenced = true;
                                    break 'scan;
                                }
                            }
                        }
                    }
                }
                cursor = Some(page.last().map(|m| m.seq).unwrap_or(0));
            }
            referenced
        };
        if has_result || referenced_elsewhere {
            return Err(SessionError::Conflict(format!(
                "deleteMessage refused: message {seq} has tool-result dependencies (a part references another part)"
            ))
            .into());
        }
        let removed = store
            .delete_message(self.id, seq)
            .map_err(crate::map_store_err)?;
        if !removed {
            return Err(
                SessionError::NotFound(format!("message {seq} of session {}", self.id)).into(),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    pub(crate) fn test_manager() -> (tempfile::TempDir, Arc<SessionManager>) {
        let dir = tempfile::tempdir().unwrap();
        let m =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        (dir, m)
    }

    pub(crate) fn session(m: &Arc<SessionManager>) -> SessionHandle {
        let ws = m.create_workspace("/w").unwrap();
        m.create_session(ws, "t", "ollama", "qwen3.8").unwrap()
    }

    #[test]
    fn illegal_transition_rejected_without_journal_trace() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let err = s
            .append_event(EventKind::ModelStarted, AgentState::Streaming, None, None)
            .unwrap_err();
        assert!(matches!(
            err.kind,
            kilop_core::ErrorKind::InvalidState { .. }
        ));
        // Nothing was written: journal still has only SessionCreated, state Idle.
        assert_eq!(s.last_event_seq().unwrap().unwrap().raw(), 1);
        assert_eq!(s.state().unwrap(), AgentState::Idle);
    }

    #[test]
    fn prompt_after_terminal_conflicts() {
        let (_d, m) = test_manager();
        let s = session(&m);
        s.append_event(EventKind::TurnCompleted, AgentState::Completed, None, None)
            .unwrap();
        let err = s.submit_prompt("hello", &[]).unwrap_err();
        assert_eq!(err.kind, kilop_core::ErrorKind::Conflict);
        // No trace: no PromptReceived event, no message.
        assert_eq!(s.events_after(EventSeq::new(1)).unwrap().len(), 1);
        assert_eq!(s.message_count().unwrap(), 0);
    }

    #[test]
    fn oversized_prompt_and_files_rejected_before_write() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let big = "x".repeat(crate::MAX_PROMPT_BYTES + 1);
        let err = s.submit_prompt(&big, &[]).unwrap_err();
        assert_eq!(err.kind, kilop_core::ErrorKind::Oversized);
        let files: Vec<String> = (0..crate::MAX_FILES_PER_PROMPT + 1)
            .map(|i| format!("/f{i}"))
            .collect();
        let err = s.submit_prompt("ok", &files).unwrap_err();
        assert_eq!(err.kind, kilop_core::ErrorKind::Oversized);
        let huge_path = vec!["x".repeat(crate::MAX_FILE_PATH_BYTES + 1)];
        let err = s.submit_prompt("ok", &huge_path).unwrap_err();
        assert_eq!(err.kind, kilop_core::ErrorKind::Oversized);
        // Journal untouched.
        assert_eq!(s.last_event_seq().unwrap().unwrap().raw(), 1);
    }

    #[test]
    fn abort_unknown_op_is_not_found() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let err = s.abort(Some(OpId::new(999))).unwrap_err();
        assert_eq!(err.kind, kilop_core::ErrorKind::NotFound);
    }

    #[test]
    fn abort_cancels_turn_and_session_stays_usable() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let receipt = s.submit_prompt("do it", &[]).unwrap();
        let aborted = s.abort(Some(receipt.op_id)).unwrap();
        assert_eq!(aborted.op_ids, vec![receipt.op_id]);
        assert!(!aborted.cancelled_all);
        // The turn is cancelled but the SESSION lands ReadyForNextTurn:
        // Stop cancels the turn, never the session (review P0-2).
        assert_eq!(s.state().unwrap(), AgentState::ReadyForNextTurn);
        // The session accepts a new prompt after the abort.
        let r2 = s.submit_prompt("continue", &[]).unwrap();
        assert!(r2.accepted);
        // Abort remains idempotent for the new turn.
        let aborted2 = s.abort(Some(r2.op_id)).unwrap();
        assert_eq!(aborted2.op_ids, vec![r2.op_id]);
        assert_eq!(s.state().unwrap(), AgentState::ReadyForNextTurn);
    }

    #[test]
    fn abort_without_ops_leaves_session_ready() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let r = s.abort(None).unwrap();
        assert!(r.op_ids.is_empty());
        assert!(r.cancelled_all);
        // Idle abort: the session stays usable (ReadyForNextTurn).
        assert_eq!(s.state().unwrap(), AgentState::ReadyForNextTurn);
    }

    #[test]
    fn concurrent_prompts_serialize_with_single_transition() {
        let (_d, m) = test_manager();
        let s = Arc::new(session(&m));
        let n = 16;
        let mut handles = Vec::new();
        for i in 0..n {
            let s = s.clone();
            handles.push(thread::spawn(move || {
                s.submit_prompt(&format!("prompt {i}"), &[]).unwrap()
            }));
        }
        let receipts: Vec<PromptReceipt> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        // Exactly one prompt transitioned the machine; the rest queued.
        let non_queued = receipts.iter().filter(|r| !r.queued).count();
        assert_eq!(non_queued, 1, "exactly one prompt must transition");
        let events = s.events_range(1, None).unwrap();
        assert_eq!(events.len(), 1 + n, "SessionCreated + N prompts");
        assert!(events
            .iter()
            .skip(1)
            .all(|e| e.kind == EventKind::PromptReceived && e.state == AgentState::Preparing));
        let queued_flags: Vec<bool> = events
            .iter()
            .skip(1)
            .map(|e| e.payload.as_ref().unwrap()["queued"].as_bool().unwrap())
            .collect();
        assert_eq!(queued_flags.iter().filter(|q| !**q).count(), 1);
        // Every receipt has a distinct op id.
        let mut ops = std::collections::HashSet::new();
        for r in &receipts {
            assert!(ops.insert(r.op_id.raw()), "op ids must be unique");
        }
        // Deferred materialization (audit round 7): only the admitted prompt
        // enters the timeline; the rest are durable queue rows, not messages.
        assert_eq!(s.message_count().unwrap(), 1);
        assert_eq!(s.queued_prompt_count().unwrap(), (n - 1) as i64);
        // Gapless journal.
        for (i, e) in events.iter().enumerate() {
            assert_eq!(e.seq.raw(), (i + 1) as u64);
        }
    }

    #[test]
    fn concurrent_chunk_events_gapless_and_state_consistent() {
        let (_d, m) = test_manager();
        let s = Arc::new(session(&m));
        s.submit_prompt("run", &[]).unwrap();
        s.append_event(
            EventKind::ContextPrepared,
            AgentState::BuildingContext,
            None,
            None,
        )
        .unwrap();
        s.append_event(
            EventKind::ModelStarted,
            AgentState::WaitingForModel,
            None,
            None,
        )
        .unwrap();
        s.append_event(
            EventKind::ModelChunkReceived,
            AgentState::Streaming,
            None,
            None,
        )
        .unwrap();
        let mut handles = Vec::new();
        for t in 0..8 {
            let s = s.clone();
            handles.push(thread::spawn(move || {
                for i in 0..50 {
                    s.append_event(
                        EventKind::ModelChunkReceived,
                        AgentState::Streaming,
                        Some(OpId::new(1 + t * 100 + i)),
                        Some(serde_json::json!({ "i": i })),
                    )
                    .unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let events = s.events_range(1, None).unwrap();
        assert_eq!(
            events.len(),
            405,
            "created + prompt + 3 chain events + 400 chunks"
        );
        for (i, e) in events.iter().enumerate() {
            assert_eq!(e.seq.raw(), (i + 1) as u64, "gapless seq at {i}");
            if e.seq.raw() > 4 {
                assert_eq!(e.state, AgentState::Streaming);
                assert_eq!(e.kind, EventKind::ModelChunkReceived);
            }
        }
        assert_eq!(s.state().unwrap(), AgentState::Streaming);
        assert_eq!(s.row().unwrap().state, AgentState::Streaming);
    }

    #[test]
    fn reopen_replays_state_from_journal() {
        let dir = tempfile::tempdir().unwrap();
        let chain = |m: &Arc<SessionManager>, s: &SessionHandle| {
            s.submit_prompt("build it", &[]).unwrap();
            s.append_event(
                EventKind::ContextPrepared,
                AgentState::BuildingContext,
                None,
                None,
            )
            .unwrap();
            s.append_event(
                EventKind::ModelStarted,
                AgentState::WaitingForModel,
                None,
                None,
            )
            .unwrap();
            s.append_event(
                EventKind::ModelChunkReceived,
                AgentState::Streaming,
                None,
                None,
            )
            .unwrap();
            s.append_event(
                EventKind::ToolRequested,
                AgentState::WaitingForPermission,
                None,
                None,
            )
            .unwrap();
            let op = m.next_op_id();
            let meta = OpMeta::new(
                op,
                s.id(),
                Deadline::at(m.now_ms() + 1000),
                kilop_core::retry::RetryPolicy::default(),
                kilop_core::cancellation::CancellationToken::new(),
                RecoveryStrategy::None,
                m.now_ms(),
            );
            s.start_tool_run(meta, "write_file", serde_json::json!({"path": "a.txt"}))
                .unwrap();
            s.finish_tool_run(op, "completed", kilop_core::op::EffectStatus::Verified)
                .unwrap();
            s.append_event(EventKind::TurnCompleted, AgentState::Completed, None, None)
                .unwrap();
        };
        let sid = {
            let m = SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                .unwrap();
            let s = session(&m);
            chain(&m, &s);
            s.id()
        };
        // "Daemon restart": a fresh manager over the same root.
        let m2 =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let s2 = m2
            .get_session(sid)
            .unwrap()
            .expect("session survived restart");
        let replay = s2.replay_journal().unwrap();
        assert_eq!(replay.state, AgentState::Completed);
        assert_eq!(replay.event_count, 9);
        assert_eq!(s2.state().unwrap(), AgentState::Completed);
        assert_eq!(s2.title().unwrap(), "t");
    }

    #[test]
    fn submit_prompt_records_message_and_receipt() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let r = s
            .submit_prompt("hello world", &["/a.rs".to_string(), "/b.rs".to_string()])
            .unwrap();
        assert!(r.accepted);
        assert!(!r.queued);
        assert_eq!(r.event_seq.raw(), 2);
        let msgs = s.messages_before(None, 10).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[0].data["text"], "hello world");
        assert_eq!(msgs[0].data["files"][0], "/a.rs");
        assert_eq!(msgs[0].seq, 2, "message seq aligns with the journal");
        assert_eq!(r.message_id, msgs[0].id);
        // The envelope is alive at creation.
        r.op_meta.ensure_alive(m.now_ms()).unwrap();
    }

    #[test]
    fn suspend_resume_and_reset_respect_the_machine() {
        let (_d, m) = test_manager();
        let s = session(&m);
        s.submit_prompt("x", &[]).unwrap();
        // Suspending mid-turn is legal; resuming targets only Idle/Preparing.
        s.suspend().unwrap();
        assert_eq!(s.state().unwrap(), AgentState::Suspended);
        assert!(
            s.resume(AgentState::Preparing).is_ok(),
            "Suspended -> Preparing is legal"
        );
        // Resuming from an active (non-Suspended) state is not.
        assert!(s.resume(AgentState::Idle).is_err());
        assert!(s.resume(AgentState::Streaming).is_err(), "malformed target");
        // A prompt from Preparing queues; from Suspended it resumes the turn.
        s.suspend().unwrap();
        assert!(
            s.submit_prompt("y", &[]).is_ok(),
            "a prompt resumes a suspended session"
        );
        assert_eq!(s.state().unwrap(), AgentState::Preparing);
        // FailedRecoverable -> Idle only via reset.
        s.mark_failed(false, "boom").unwrap();
        assert_eq!(s.state().unwrap(), AgentState::FailedRecoverable);
        s.reset().unwrap();
        assert_eq!(s.state().unwrap(), AgentState::Idle);
        // end_session closes the LIFETIME machine; only after that the
        // turn machine reaches Completed and nothing more is promptable.
        assert_eq!(
            s.lifecycle().unwrap(),
            kilop_core::state::SessionLifecycle::Open
        );
        s.end_session().unwrap();
        assert_eq!(
            s.lifecycle().unwrap(),
            kilop_core::state::SessionLifecycle::Closed
        );
        assert_eq!(s.state().unwrap(), AgentState::Completed);
        assert!(s.reset().is_err(), "Completed is terminal");
        assert!(
            s.submit_prompt("late", &[]).is_err(),
            "closed session rejects prompts"
        );
        assert!(s.end_session().is_err(), "double close conflicts");
    }

    #[test]
    fn mark_failed_permanent_is_documented_two_step() {
        let (_d, m) = test_manager();
        let s = session(&m);
        // From Idle, marking failed is illegal (Idle cannot reach
        // FailedRecoverable); the session must be mid-work first.
        assert!(s.mark_failed(false, "x").is_err());
        s.submit_prompt("work", &[]).unwrap();
        s.mark_failed(true, "unrecoverable").unwrap();
        assert_eq!(s.state().unwrap(), AgentState::FailedPermanent);
        // The journal records the escalation; replay accepts it.
        assert_eq!(
            s.replay_journal().unwrap().state,
            AgentState::FailedPermanent
        );
        // Terminal: nothing else works.
        assert!(s.submit_prompt("x", &[]).is_err());
        assert!(s.end_session().is_err());
    }

    #[test]
    fn file_change_event_requires_executing_state() {
        let (_d, m) = test_manager();
        let s = session(&m);
        // From Idle, FileChanged -> ExecutingTool is illegal.
        assert!(s.record_file_change("/a.rs", None).is_err());
        s.submit_prompt("x", &[]).unwrap();
        assert!(
            s.record_file_change("/a.rs", None).is_err(),
            "Preparing cannot record a file change"
        );
    }

    #[test]
    fn queued_prompt_state_is_documented_self_transition() {
        let (_d, m) = test_manager();
        let s = session(&m);
        s.submit_prompt("first", &[]).unwrap();
        // Second prompt while Preparing must queue, not transition.
        let r = s.submit_prompt("second", &[]).unwrap();
        assert!(r.queued);
        assert_eq!(s.state().unwrap(), AgentState::Preparing);
    }

    #[test]
    fn turn_record_created_only_at_admission_and_tracks_the_envelope() {
        // v7 (requirement 1): a durable per-turn record exists the moment a
        // prompt is admitted as the ACTIVE logical turn; a queued (never
        // admitted) prompt has NO record — no phantom identity.
        let (_d, m) = test_manager();
        let s = session(&m);
        let r1 = s.submit_prompt("one", &[]).unwrap();
        assert!(!r1.queued);
        let records = s.turn_records().unwrap();
        assert_eq!(records.len(), 1, "admitted prompt opens exactly one record");
        assert_eq!(records[0].turn_op_id, r1.op_id);
        assert_eq!(records[0].status, "active");
        assert_eq!(records[0].effective_provider, "ollama");
        assert_eq!(records[0].effective_model, "qwen3.8");
        assert_eq!(
            records[0].prompt_message_id,
            Some(2),
            "message seq of the prompt"
        );
        // A queued prompt while the machine is mid-turn: no record at all.
        let r2 = s.submit_prompt("two", &[]).unwrap();
        assert!(r2.queued);
        assert_eq!(
            s.turn_records().unwrap().len(),
            1,
            "queued prompt must not open a record"
        );
        assert!(s.turn_record(r2.op_id).unwrap().is_none());
        assert_eq!(s.queued_prompt_count().unwrap(), 1);
        // The effective envelope is fixed at drive start (model override).
        s.set_turn_envelope(
            r1.op_id,
            "ollama",
            "override-model",
            Some("v1"),
            Some("native"),
        )
        .unwrap();
        let rec = s.turn_record(r1.op_id).unwrap().unwrap();
        assert_eq!(rec.effective_model, "override-model");
        assert_eq!(rec.tool_mode.as_deref(), Some("native"));
        assert_eq!(rec.variant.as_deref(), Some("v1"));
        // Finish is idempotent and the record closes.
        assert!(s.finish_turn_record(r1.op_id, "completed").unwrap());
        assert!(!s.finish_turn_record(r1.op_id, "completed").unwrap());
        assert_eq!(
            s.turn_record(r1.op_id).unwrap().unwrap().status,
            "completed"
        );
        assert!(s.active_turn_record().unwrap().is_none());
        // A fresh prompt on a promptable machine opens a NEW active record
        // and the old one stays closed (only one active logical turn ever).
        s.mark_failed(false, "sim").unwrap();
        s.reset().unwrap();
        let r3 = s.submit_prompt("three", &[]).unwrap();
        assert!(!r3.queued);
        let recs = s.turn_records().unwrap();
        // r1 (completed) + r3 (active): the queued r2 NEVER got a record.
        assert_eq!(recs.len(), 2, "queued prompts never open a record");
        assert_eq!(
            s.active_turn_record().unwrap().unwrap().turn_op_id,
            r3.op_id
        );
        assert_eq!(
            s.turn_record(r1.op_id).unwrap().unwrap().status,
            "completed",
            "a new admission never resurrects an old record"
        );
    }

    // -------------------------------------------------- title update (P1)

    #[test]
    fn update_session_title_bounds_strip_and_persist() {
        let dir = tempfile::tempdir().unwrap();
        let m =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let s = session(&m);
        // Control characters are stripped; the result persists durably.
        s.update_session_title("clean\n\tname\u{7f}done").unwrap();
        assert_eq!(s.title().unwrap(), "cleannamedone");
        assert!(s.row().unwrap().updated_ms > 0);
        // Hostile titles refuse before any write.
        let err = s.update_session_title("\r\n\u{0}").unwrap_err();
        assert_eq!(err.kind, kilop_core::error::ErrorKind::Malformed, "{err}");
        let err = s.update_session_title(&"x".repeat(201)).unwrap_err();
        assert_eq!(err.kind, kilop_core::error::ErrorKind::Oversized, "{err}");
        // 200 chars exactly is accepted; control chars do not count.
        s.update_session_title(&format!("{}x", "y".repeat(199)))
            .unwrap();
        assert_eq!(s.title().unwrap().len(), 200);
        // The title survives a full reopen.
        drop(m);
        let m2 =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let row = m2.get_session(s.id()).unwrap().unwrap().row().unwrap();
        assert_eq!(row.title, format!("{}x", "y".repeat(199)));
    }

    // -------------------------------------------------- message removal (P1)

    #[test]
    fn delete_message_removes_durably_and_keeps_seqs_stable() {
        let dir = tempfile::tempdir().unwrap();
        let sid = {
            let m = SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                .unwrap();
            let s = session(&m);
            let mid1 = s
                .put_message(1, "user", serde_json::json!({"text": "a"}))
                .unwrap();
            s.put_text_part(mid1, "hello").unwrap();
            let mid2 = s
                .put_message(2, "assistant", serde_json::json!({"parts": []}))
                .unwrap();
            s.put_text_part(mid2, "world").unwrap();
            s.put_message(3, "user", serde_json::json!({"text": "b"}))
                .unwrap();
            s.delete_message(2).unwrap();
            // Rows removed; paging skips the hole with stable seqs.
            assert_eq!(s.message_count().unwrap(), 2);
            let page = s.messages_page(None, 10).unwrap();
            let seqs: Vec<i64> = page.messages.iter().map(|m| m.seq).collect();
            assert_eq!(seqs, vec![3, 1]);
            assert!(s.parts_of(mid2).unwrap().is_empty());
            assert_eq!(s.proposed_message_seq().unwrap(), 4);
            s.id()
        };
        // Durable across reopen: gone from the page.
        let m2 =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let s2 = m2.get_session(sid).unwrap().unwrap();
        assert_eq!(s2.message_count().unwrap(), 2);
        let seqs: Vec<i64> = s2
            .messages_page(None, 10)
            .unwrap()
            .messages
            .iter()
            .map(|m| m.seq)
            .collect();
        assert_eq!(seqs, vec![3, 1]);
    }

    #[test]
    fn delete_message_unknown_seq_is_not_found() {
        let (_d, m) = test_manager();
        let s = session(&m);
        s.put_message(1, "user", serde_json::json!({"text": "a"}))
            .unwrap();
        let err = s.delete_message(9).unwrap_err();
        assert_eq!(err.kind, kilop_core::error::ErrorKind::NotFound, "{err}");
        assert!(err.message.contains("9"), "{err}");
    }

    #[test]
    fn delete_message_refuses_tool_paired_messages() {
        let (_d, m) = test_manager();
        let s = session(&m);
        s.put_message(1, "user", serde_json::json!({"text": "run"}))
            .unwrap();
        let mid2 = s
            .put_message(2, "assistant", serde_json::json!({"parts": []}))
            .unwrap();
        s.put_tool_call_part(mid2, "c1", "echo", serde_json::json!({}), "completed")
            .unwrap();
        let mid3 = s
            .put_message(3, "assistant", serde_json::json!({"parts": []}))
            .unwrap();
        s.put_tool_result_part(
            mid3,
            "c1",
            &kilop_protocol::v756::ToolResultBody {
                excerpt: "out".into(),
                exit_code: Some(0),
                artifact: None,
                slice_hint: None,
            },
        )
        .unwrap();
        // The result message references the call: refuse.
        let err = s.delete_message(3).unwrap_err();
        assert_eq!(err.kind, kilop_core::error::ErrorKind::Conflict, "{err}");
        assert!(err.message.contains("tool-result dependencies"), "{err}");
        // The call message is referenced by that result: refuse.
        let err = s.delete_message(2).unwrap_err();
        assert_eq!(err.kind, kilop_core::error::ErrorKind::Conflict, "{err}");
        assert!(err.message.contains("tool-result dependencies"), "{err}");
        // The unrelated prompt is still removable.
        s.delete_message(1).unwrap();
        assert_eq!(s.message_count().unwrap(), 2);
        // The pairing refusal is bidirectional and durable: even deleting
        // the call message whose result sits in a DIFFERENT session-owned
        // message refuses (scan is over the whole session).
        let err = s.delete_message(2).unwrap_err();
        assert_eq!(err.kind, kilop_core::error::ErrorKind::Conflict);
        let err = s.delete_message(3).unwrap_err();
        assert_eq!(err.kind, kilop_core::error::ErrorKind::Conflict);
    }

    #[test]
    fn delete_message_refuses_in_flight_newest_and_allows_after_turn() {
        let (_d, m) = test_manager();
        let s = session(&m);
        s.submit_prompt("stream", &[]).unwrap();
        let mid = s
            .put_message(3, "assistant", serde_json::json!({"parts": []}))
            .unwrap();
        s.put_text_part(mid, "partial").unwrap();
        s.append_event(
            kilop_core::event::EventKind::ContextPrepared,
            kilop_core::state::AgentState::BuildingContext,
            None,
            None,
        )
        .unwrap();
        s.append_event(
            kilop_core::event::EventKind::ModelStarted,
            kilop_core::state::AgentState::WaitingForModel,
            None,
            None,
        )
        .unwrap();
        s.append_event(
            kilop_core::event::EventKind::ModelChunkReceived,
            kilop_core::state::AgentState::Streaming,
            None,
            None,
        )
        .unwrap();
        // The newest (streamed) message of the active turn refuses.
        let err = s.delete_message(3).unwrap_err();
        assert_eq!(err.kind, kilop_core::error::ErrorKind::Conflict, "{err}");
        assert!(err.message.contains("in flight"), "{err}");
        assert_eq!(s.message_count().unwrap(), 2);
        // An OLDER message (the prompt, seq 2) is not the newest: deleting
        // it while the turn streams is still allowed? No — keep the safety
        // rule minimal: the prompt is not in flight, but removing it would
        // tear the active turn's materialized prompt; the rule refuses only
        // the newest message, so this older prompt is removable.
        s.delete_message(2).unwrap();
        assert_eq!(s.message_count().unwrap(), 1);
        // Once the turn completes, the (new) newest message is removable.
        s.put_message(4, "assistant", serde_json::json!({"parts": []}))
            .unwrap();
        s.append_event(
            kilop_core::event::EventKind::ToolCompleted,
            kilop_core::state::AgentState::Validating,
            None,
            None,
        )
        .unwrap();
        s.append_event(
            kilop_core::event::EventKind::TurnCompleted,
            kilop_core::state::AgentState::Completed,
            None,
            None,
        )
        .unwrap();
        assert!(!s.state().unwrap().is_active());
        // The agent runtime finalizes the durable turn record when a turn
        // ends; do the same here so the "active turn" gate opens.
        let record = m.store().active_turn_record(s.id()).unwrap().unwrap();
        m.store()
            .finish_turn_record(
                s.id(),
                record.turn_op_id,
                kilop_store::TURN_RECORD_COMPLETED,
            )
            .unwrap();
        s.delete_message(4).unwrap();
        // Only the earlier seq-3 assistant row remains.
        assert_eq!(s.message_count().unwrap(), 1);
        let seqs: Vec<i64> = s
            .messages_page(None, 10)
            .unwrap()
            .messages
            .iter()
            .map(|m| m.seq)
            .collect();
        assert_eq!(seqs, vec![3]);
    }
}
