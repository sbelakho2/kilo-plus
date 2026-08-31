//! `SessionHandle`: every durable command a session accepts, journaled and
//! state-machine-validated.

use std::sync::Arc;

use kilop_core::event::{Event, EventKind};
use kilop_core::id::{EventSeq, OpId, SessionId};
use kilop_core::op::{OpMeta, RecoveryStrategy};
use kilop_core::state::AgentState;
use kilop_core::time::Deadline;
use kilop_store::SessionRow;

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
            .map_err(|e| crate::map_store_err(e))?)
    }

    pub(crate) fn system_hasher(&self) -> &Arc<SystemFileHasher> {
        &self.system_hasher
    }

    // ---------------------------------------------------------------- read state

    /// The fresh session row (title/provider/model/state) from durable store.
    pub fn row(&self) -> kilop_core::Result<SessionRow> {
        match self.manager.store().get_session(self.id).map_err(|e| crate::map_store_err(e))? {
            Some(r) => Ok(r),
            None => Err(SessionError::NotFound(format!("session {}", self.id)).into()),
        }
    }

    pub fn state(&self) -> kilop_core::Result<AgentState> {
        Ok(self.row()?.state)
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
        Ok(self.manager
            .store()
            .append_event(self.id, op_id, kind, state, self.now_ms(), payload)
            .map_err(|e| crate::map_store_err(e))?)
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
    pub fn submit_prompt(&self, prompt: &str, files: &[String]) -> kilop_core::Result<PromptReceipt> {
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
                self.id,
                current
            ))
            .into());
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
        let message_id = self
            .manager
            .store()
            .put_message(
                self.id,
                event_seq.raw() as i64,
                "user",
                serde_json::json!({ "text": prompt, "files": files }),
            )
            .map_err(|e| crate::map_store_err(e))?;
        self.ops().register_turn(op_id, op_meta.cancellation.clone());

        Ok(PromptReceipt {
            op_id,
            op_meta,
            event_seq,
            message_id,
            accepted: true,
            queued,
        })
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
        let affected: Vec<OpId> = match op_id {
            Some(o) => {
                if self.ops().tracked(o).is_none() {
                    return Err(SessionError::NotFound(format!("operation {o}")).into());
                }
                vec![o]
            }
            None => self.ops().all(),
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
                    .map_err(|e| crate::map_store_err(e))?;
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
            event_seq = Some(self.transition_locked(kind, AgentState::Cancelled, Some(*o), Some(payload))?);
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
        for o in &affected {
            self.ops().unregister(*o);
        }
        Ok(AbortReceipt {
            op_ids: affected,
            event_seq: event_seq.expect("at least one event appended"),
            cancelled_all: op_id.is_none(),
        })
    }

    // ---------------------------------------------------------------- lifecycle

    /// Suspend the session (user-initiated pause). Active states may suspend.
    pub fn suspend(&self) -> kilop_core::Result<EventSeq> {
        self.append_event(EventKind::Suspended, AgentState::Suspended, None, None)
    }

    /// Resume a suspended session into `to` (`Idle` or `Preparing`).
    pub fn resume(&self, to: AgentState) -> kilop_core::Result<EventSeq> {
        if to != AgentState::Idle && to != AgentState::Preparing {
            return Err(SessionError::Malformed(
                "resume target must be Idle or Preparing".into(),
            )
            .into());
        }
        self.append_event(EventKind::Resumed, to, None, None)
    }

    /// Return a failed-recoverable session to `Idle` (documented recovery
    /// outcome; legal only from `FailedRecoverable`).
    pub fn reset(&self) -> kilop_core::Result<EventSeq> {
        self.append_event(EventKind::RecoveryApplied, AgentState::Idle, None, None)
    }

    /// End the session: journals `SessionEnded` and transitions to
    /// `Completed`. Refuses while child processes are still registered
    /// (Commandment 8 — zero orphans); ownership must transfer first.
    pub fn end_session(&self) -> kilop_core::Result<EventSeq> {
        if !self.processes().all().is_empty() {
            return Err(SessionError::Conflict(format!(
                "session {} still owns {} child process(es); release or transfer them first",
                self.id,
                self.processes().all().len()
            ))
            .into());
        }
        self.append_event(EventKind::SessionEnded, AgentState::Completed, None, None)
    }

    /// Mark the session failed. `permanent: true` uses the documented
    /// two-step escalation (`FailedRecoverable` legality, then force).
    pub fn mark_failed(
        &self,
        permanent: bool,
        message: &str,
    ) -> kilop_core::Result<EventSeq> {
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
        self.append_event(EventKind::FileChanged, AgentState::ExecutingTool, None, Some(payload))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    pub(crate) fn test_manager() -> (tempfile::TempDir, Arc<SessionManager>) {
        let dir = tempfile::tempdir().unwrap();
        let m = SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
            .unwrap();
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
    fn abort_ends_session_and_second_abort_conflicts() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let receipt = s.submit_prompt("do it", &[]).unwrap();
        let aborted = s.abort(Some(receipt.op_id)).unwrap();
        assert_eq!(aborted.op_ids, vec![receipt.op_id]);
        assert!(!aborted.cancelled_all);
        assert_eq!(s.state().unwrap(), AgentState::Cancelled);
        // The turn op is no longer tracked, and the session is terminal.
        assert!(s.abort(None).is_err());
    }

    #[test]
    fn abort_without_ops_still_ends_session() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let r = s.abort(None).unwrap();
        assert!(r.op_ids.is_empty());
        assert!(r.cancelled_all);
        assert_eq!(s.state().unwrap(), AgentState::Cancelled);
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
        let receipts: Vec<PromptReceipt> = handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect();
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
        // Every receipt has a distinct op id; messages carry distinct seqs.
        let mut ops = std::collections::HashSet::new();
        for r in &receipts {
            assert!(ops.insert(r.op_id.raw()), "op ids must be unique");
        }
        assert_eq!(s.message_count().unwrap(), n as i64);
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
        s.append_event(EventKind::ContextPrepared, AgentState::BuildingContext, None, None)
            .unwrap();
        s.append_event(EventKind::ModelStarted, AgentState::WaitingForModel, None, None)
            .unwrap();
        s.append_event(EventKind::ModelChunkReceived, AgentState::Streaming, None, None)
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
        assert_eq!(events.len(), 405, "created + prompt + 3 chain events + 400 chunks");
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
            s.append_event(EventKind::ContextPrepared, AgentState::BuildingContext, None, None)
                .unwrap();
            s.append_event(EventKind::ModelStarted, AgentState::WaitingForModel, None, None)
                .unwrap();
            s.append_event(EventKind::ModelChunkReceived, AgentState::Streaming, None, None)
                .unwrap();
            s.append_event(EventKind::ToolRequested, AgentState::WaitingForPermission, None, None)
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
            s.start_tool_run(meta, "write_file", serde_json::json!({"path": "a.txt"})).unwrap();
            s.finish_tool_run(op, "completed", kilop_core::op::EffectStatus::Verified).unwrap();
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
        let m2 = SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
            .unwrap();
        let s2 = m2.get_session(sid).unwrap().expect("session survived restart");
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
        let r = s.submit_prompt("hello world", &["/a.rs".to_string(), "/b.rs".to_string()]).unwrap();
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
        assert!(s.resume(AgentState::Preparing).is_ok(), "Suspended -> Preparing is legal");
        // Resuming from an active (non-Suspended) state is not.
        assert!(s.resume(AgentState::Idle).is_err());
        assert!(s.resume(AgentState::Streaming).is_err(), "malformed target");
        // A prompt from Preparing queues; from Suspended it resumes the turn.
        s.suspend().unwrap();
        assert!(s.submit_prompt("y", &[]).is_ok(), "a prompt resumes a suspended session");
        assert_eq!(s.state().unwrap(), AgentState::Preparing);
        // FailedRecoverable -> Idle only via reset.
        s.mark_failed(false, "boom").unwrap();
        assert_eq!(s.state().unwrap(), AgentState::FailedRecoverable);
        assert!(s.end_session().is_err(), "FailedRecoverable cannot end");
        s.reset().unwrap();
        assert_eq!(s.state().unwrap(), AgentState::Idle);
        s.end_session().unwrap();
        assert_eq!(s.state().unwrap(), AgentState::Completed);
        assert!(s.reset().is_err(), "Completed is terminal");
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
        assert_eq!(s.replay_journal().unwrap().state, AgentState::FailedPermanent);
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
        assert!(s.record_file_change("/a.rs", None).is_err(), "Preparing cannot record a file change");
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
}
