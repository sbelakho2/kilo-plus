//! Operation ledger: tool runs, provider calls, permission requests, and the
//! in-memory registry that gives the session ownership of every in-flight op's
//! cancellation token.

use std::collections::HashMap;
use std::sync::Mutex;

use kilop_core::cancellation::CancellationToken;
use kilop_core::capability::Capability;
use kilop_core::id::OpId;
use kilop_core::op::{EffectStatus, OpMeta};
use kilop_core::state::AgentState;
use kilop_store::ToolRunRow;

use crate::handle::SessionHandle;
use crate::{SessionError, json_bytes, effect_str, MAX_TOOL_ARGS_BYTES};

/// What kind of operation an id refers to (drives abort's event kind).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpKind {
    /// A user prompt turn.
    Turn,
    /// A tool execution.
    Tool,
}

#[derive(Debug, Clone)]
pub(crate) struct TrackedOp {
    pub kind: OpKind,
    pub token: CancellationToken,
}

/// In-memory ownership registry: every registered op holds its cancellation
/// token here so `abort` can fan cancellation out. Shared per session through
/// the manager (all handle clones cancel the same ops).
#[derive(Debug, Default)]
pub(crate) struct OpRegistry {
    inner: Mutex<HashMap<OpId, TrackedOp>>,
}

impl OpRegistry {
    pub fn register(&self, op: OpId, kind: OpKind, token: CancellationToken) {
        self.inner
            .lock()
            .expect("op registry poisoned")
            .insert(op, TrackedOp { kind, token });
    }

    pub fn register_turn(&self, op: OpId, token: CancellationToken) {
        self.register(op, OpKind::Turn, token);
    }

    pub fn unregister(&self, op: OpId) {
        self.inner.lock().expect("op registry poisoned").remove(&op);
    }

    pub fn tracked(&self, op: OpId) -> Option<TrackedOp> {
        self.inner
            .lock()
            .expect("op registry poisoned")
            .get(&op)
            .cloned()
    }

    pub fn kind(&self, op: OpId) -> Option<OpKind> {
        self.tracked(op).map(|t| t.kind)
    }

    pub fn cancel(&self, op: OpId) {
        if let Some(t) = self.tracked(op) {
            t.token.cancel();
        }
    }

    pub fn all(&self) -> Vec<OpId> {
        self.inner
            .lock()
            .expect("op registry poisoned")
            .keys()
            .copied()
            .collect()
    }

}

/// Handle to a started tool run.
#[derive(Debug, Clone)]
pub struct ToolRunHandle {
    pub op_id: OpId,
    pub tool: String,
    pub row_id: i64,
    pub started_ms: i64,
}

/// A pending permission request awaiting the user.
#[derive(Debug, Clone)]
pub struct PermissionRequest {
    pub id: i64,
    pub op_id: OpId,
    pub capability: Capability,
    pub event_seq: kilop_core::id::EventSeq,
}

fn capability_tag(cap: &Capability) -> String {
    serde_json::to_value(cap)
        .ok()
        .and_then(|v| v.get("capability").and_then(|t| t.as_str()).map(String::from))
        .unwrap_or_else(|| "unknown".into())
}

impl SessionHandle {
    /// Start a tool run. The op envelope is the caller's (it owns the
    /// deadline/retry/cancellation/recovery); this command journals
    /// `ToolStarted` and registers the op so abort can cancel it. Requires the
    /// session to be in `WaitingForPermission`/`ToolRequested`/`ExecutingTool`
    /// (auto-allowed tools must still pass through the `ToolRequested` hop so
    /// the journal is complete).
    pub fn start_tool_run(
        &self,
        op: OpMeta,
        tool: &str,
        args: serde_json::Value,
    ) -> kilop_core::Result<ToolRunHandle> {
        if op.session_id != self.id {
            return Err(SessionError::NotFound(format!(
                "op {} belongs to session {}, not {}",
                op.operation_id, op.session_id, self.id
            ))
            .into());
        }
        op.ensure_alive(self.now_ms()).map_err(SessionError::from)?;
        if tool.is_empty() || tool.len() > 256 {
            return Err(SessionError::Malformed(format!("invalid tool name {tool:?}")).into());
        }
        if json_bytes(&args) > MAX_TOOL_ARGS_BYTES {
            return Err(SessionError::Oversized(format!(
                "tool args of {} bytes exceed MAX_TOOL_ARGS_BYTES",
                json_bytes(&args)
            ))
            .into());
        }
        let _guard = self.command_guard();
        let recovery = serde_json::to_value(&op.recovery)
            .map_err(|e| SessionError::Malformed(format!("recovery serialization: {e}")))?;
        let expected_hash = match &op.recovery {
            kilop_core::op::RecoveryStrategy::VerifyHash { expected, .. } => {
                Some(expected.to_hex())
            }
            _ => None,
        };
        // Validate the transition before any durable write.
        crate::journal::validate_transition(
            self.state()?,
            kilop_core::event::EventKind::ToolStarted,
            AgentState::ExecutingTool,
        )?;
        let row_id = self
            .manager
            .store()
            .start_tool_run(self.id, op.operation_id, tool, args, recovery, expected_hash)
            .map_err(crate::map_store_err)?;
        self.transition_locked(
            kilop_core::event::EventKind::ToolStarted,
            AgentState::ExecutingTool,
            Some(op.operation_id),
            Some(serde_json::json!({ "tool": tool })),
        )?;
        self.ops().register(op.operation_id, OpKind::Tool, op.cancellation.clone());
        Ok(ToolRunHandle {
            op_id: op.operation_id,
            tool: tool.to_string(),
            row_id,
            started_ms: op.start_time_ms,
        })
    }

    /// Finish a tool run durably and journal the outcome. `completed` lands
    /// on `Validating`; `failed` on `FailedRecoverable`; `cancelled` on
    /// `Cancelled` (the session ends — cancel is terminal by the core
    /// machine). Finishing an unknown op is `NotFound` (loud).
    pub fn finish_tool_run(
        &self,
        op: OpId,
        status: &str,
        effect: EffectStatus,
    ) -> kilop_core::Result<()> {
        if !matches!(status, "completed" | "failed" | "cancelled") {
            return Err(SessionError::Malformed(format!("invalid tool status {status:?}")).into());
        }
        let _guard = self.command_guard();
        // Must exist before journaling anything.
        let pending = self.pending_tool_runs()?;
        if !pending.iter().any(|r| r.op_id == op) {
            return Err(SessionError::NotFound(format!("tool run {op} is not running")).into());
        }
        self.manager
            .store()
            .finish_tool_run(self.id, op, status, effect_str(effect))
            .map_err(crate::map_store_err)?;
        let (kind, state) = match status {
            "completed" => (kilop_core::event::EventKind::ToolCompleted, AgentState::Validating),
            "failed" => (kilop_core::event::EventKind::ToolCompleted, AgentState::FailedRecoverable),
            "cancelled" => (kilop_core::event::EventKind::ToolCancelled, AgentState::Cancelled),
            _ => unreachable!("validated above"),
        };
        self.transition_locked(
            kind,
            state,
            Some(op),
            Some(serde_json::json!({ "status": status, "effect": effect_str(effect) })),
        )?;
        self.ops().unregister(op);
        Ok(())
    }

    pub fn set_tool_run_effect(&self, op: OpId, effect: EffectStatus) -> kilop_core::Result<()> {
        self.manager
            .store()
            .set_tool_run_effect(self.id, op, effect_str(effect))
            .map_err(|e| crate::map_store_err(e).into())
    }

    pub fn pending_tool_runs(&self) -> kilop_core::Result<Vec<ToolRunRow>> {
        self.manager
            .store()
            .pending_tool_runs(self.id)
            .map_err(|e| crate::map_store_err(e).into())
    }

    /// Record a provider wire call (never cancels the session; provider calls
    /// are sub-operations of a turn).
    #[allow(clippy::too_many_arguments)]
    pub fn record_provider_call(
        &self,
        op: OpId,
        provider: &str,
        model: &str,
        status: &str,
        tokens_in: Option<u64>,
        tokens_out: Option<u64>,
        error: Option<&str>,
    ) -> kilop_core::Result<i64> {
        if provider.len() > 256 || model.len() > 256 {
            return Err(SessionError::Oversized("provider/model name too long".into()).into());
        }
        Ok(self.manager
            .store()
            .record_provider_call(
                self.id,
                op,
                provider,
                model,
                status,
                tokens_in,
                tokens_out,
                error,
            )
            .map_err(crate::map_store_err)?)
    }

    /// Request permission to use `capability` for `op`. Journals
    /// `ToolRequested` (recorded with state `WaitingForPermission` — the
    /// documented two-hop) and inserts the durable pending row.
    pub fn request_permission(
        &self,
        op: OpId,
        capability: &Capability,
    ) -> kilop_core::Result<PermissionRequest> {
        let _guard = self.command_guard();
        if self.ops().tracked(op).is_none() {
            return Err(
                SessionError::NotFound(format!("operation {op} is not tracked")).into(),
            );
        }
        let cap_json = serde_json::to_string(capability)
            .map_err(|e| SessionError::Malformed(format!("capability serialization: {e}")))?;
        if cap_json.len() > 4096 {
            return Err(SessionError::Oversized("capability too large".into()).into());
        }
        let id = self
            .manager
            .store()
            .insert_permission(self.id, op, &cap_json)
            .map_err(crate::map_store_err)?;
        let event_seq = self.transition_locked(
            kilop_core::event::EventKind::ToolRequested,
            AgentState::WaitingForPermission,
            Some(op),
            Some(serde_json::json!({
                "permission_id": id,
                "capability": capability_tag(capability),
                "detail": capability,
            })),
        )?;
        Ok(PermissionRequest {
            id,
            op_id: op,
            capability: capability.clone(),
            event_seq,
        })
    }

    /// Resolve a pending permission. `Allow` journals `PermissionGranted`
    /// (state `ExecutingTool`); `Deny` journals `PermissionDenied` and returns
    /// the session to `ReadyForNextTurn` when nothing else is running (stays
    /// `ExecutingTool` under parallel tools). A double resolve loses the race
    /// with `Conflict`; the journal never records two resolutions.
    pub fn resolve_permission(
        &self,
        id: i64,
        decision: kilop_core::capability::PermissionDecision,
    ) -> kilop_core::Result<kilop_core::id::EventSeq> {
        let (decision_str, kind, target) = match decision {
            kilop_core::capability::PermissionDecision::Allow => {
                ("allow", kilop_core::event::EventKind::PermissionGranted, AgentState::ExecutingTool)
            }
            kilop_core::capability::PermissionDecision::Deny => {
                ("deny", kilop_core::event::EventKind::PermissionDenied, AgentState::ReadyForNextTurn)
            }
            kilop_core::capability::PermissionDecision::Ask => {
                return Err(SessionError::Malformed(
                    "a permission cannot be resolved with Ask".into(),
                )
                .into());
            }
        };
        let _guard = self.command_guard();
        // Pre-check: unknown or already-resolved rows are loud conflicts.
        let op = match self
            .manager
            .store()
            .pending_permission(id)
            .map_err(crate::map_store_err)?
        {
            Some((_, op, _)) => op,
            None => {
                return Err(SessionError::Conflict(format!(
                    "permission {id} is not pending (unknown or already resolved)"
                ))
                .into());
            }
        };
        self.manager
            .store()
            .resolve_permission(id, decision_str)
            .map_err(crate::map_store_err)?;
        // Post-check: whoever lost the race must not journal.
        if self.manager.store().pending_permission(id).map_err(crate::map_store_err)?.is_some() {
            return Err(SessionError::Conflict(format!(
                "permission {id} was resolved concurrently"
            ))
            .into());
        }
        // Deny under a parallel tool: staying ExecutingTool is the honest
        // machine outcome (ExecutingTool cannot go to ReadyForNextTurn).
        let target = if kind == kilop_core::event::EventKind::PermissionDenied {
            let current = self.state()?;
            let mut m = kilop_core::state::StateMachine::new(current);
            if m.transition(AgentState::ReadyForNextTurn).is_err() {
                current
            } else {
                AgentState::ReadyForNextTurn
            }
        } else {
            target
        };
        self.transition_locked(
            kind,
            target,
            Some(op),
            Some(serde_json::json!({ "permission_id": id, "decision": decision_str })),
        )
    }

    pub fn pending_permission(
        &self,
        id: i64,
    ) -> kilop_core::Result<Option<(kilop_core::id::SessionId, OpId, String)>> {
        self.manager
            .store()
            .pending_permission(id)
            .map_err(|e| crate::map_store_err(e).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::tests::{session, test_manager};
    use kilop_core::event::EventKind;
    use kilop_core::id::SessionId;
    use kilop_core::time::Deadline;

    fn op_meta(m: &crate::SessionManager, s: SessionId, recovery: kilop_core::op::RecoveryStrategy) -> OpMeta {
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

    fn to_streaming(s: &SessionHandle) {
        s.submit_prompt("x", &[]).unwrap();
        s.append_event(EventKind::ContextPrepared, AgentState::BuildingContext, None, None).unwrap();
        s.append_event(EventKind::ModelStarted, AgentState::WaitingForModel, None, None).unwrap();
        s.append_event(EventKind::ModelChunkReceived, AgentState::Streaming, None, None).unwrap();
    }

    fn to_waiting(s: &SessionHandle) {
        to_streaming(s);
        let turn_op = s.ops().all()[0];
        s.request_permission(turn_op, &Capability::ReadWorkspace { path: "/w/a".into() })
            .unwrap();
    }

    #[test]
    fn permission_double_resolve_race_single_event() {
        let (_d, m) = test_manager();
        let s = session(&m);
        to_streaming(&s);
        let turn_op = s.ops().all()[0];
        let req = s
            .request_permission(
                turn_op,
                &Capability::ExecuteShell { command: "cargo test".into() },
            )
            .unwrap();
        // Two resolvers race: allow vs deny.
        let s = std::sync::Arc::new(s);
        let s1 = s.clone();
        let s2 = s.clone();
        let t1 = std::thread::spawn(move || {
            s1.resolve_permission(req.id, kilop_core::capability::PermissionDecision::Allow)
        });
        let t2 = std::thread::spawn(move || {
            s2.resolve_permission(req.id, kilop_core::capability::PermissionDecision::Deny)
        });
        let r1 = t1.join().unwrap();
        let r2 = t2.join().unwrap();
        assert!(
            r1.is_ok() != r2.is_ok(),
            "exactly one resolver must win; got {r1:?} / {r2:?}"
        );
        // Exactly one resolution event, and it matches the winner's decision.
        let events = s.events_range(1, None).unwrap();
        let resolutions: Vec<_> = events
            .iter()
            .filter(|e| {
                matches!(
                    e.kind,
                    EventKind::PermissionGranted | EventKind::PermissionDenied
                )
            })
            .collect();
        assert_eq!(resolutions.len(), 1);
        let winning_decision = if r1.is_ok() { "allow" } else { "deny" };
        let expected_kind = if winning_decision == "allow" {
            EventKind::PermissionGranted
        } else {
            EventKind::PermissionDenied
        };
        assert_eq!(resolutions[0].kind, expected_kind);
    }

    #[test]
    fn start_tool_run_requires_tool_request_hop() {
        let (_d, m) = test_manager();
        let s = session(&m);
        s.submit_prompt("x", &[]).unwrap();
        // From Preparing, ExecutingTool is illegal: the ToolRequested hop is
        // mandatory and the command leaves no trace.
        let meta = op_meta(&m, s.id(), kilop_core::op::RecoveryStrategy::None);
        let err = s
            .start_tool_run(meta, "read_file", serde_json::json!({}))
            .unwrap_err();
        assert!(matches!(err.kind, kilop_core::ErrorKind::InvalidState { .. }));
        assert!(s.pending_tool_runs().unwrap().is_empty(), "no tool_run row");
        assert_eq!(s.last_event_seq().unwrap().unwrap().raw(), 2);
    }

    #[test]
    fn foreign_op_meta_rejected_by_start_tool_run() {
        let (_d, m) = test_manager();
        let s = session(&m);
        s.submit_prompt("x", &[]).unwrap();
        s.append_event(EventKind::ContextPrepared, AgentState::BuildingContext, None, None).unwrap();
        s.append_event(EventKind::ModelStarted, AgentState::WaitingForModel, None, None).unwrap();
        s.append_event(EventKind::ModelChunkReceived, AgentState::Streaming, None, None).unwrap();
        s.append_event(EventKind::ToolRequested, AgentState::WaitingForPermission, None, None).unwrap();
        // An op envelope pointed at another session is rejected loudly.
        let mut meta = op_meta(&m, s.id(), kilop_core::op::RecoveryStrategy::None);
        meta.session_id = SessionId::new(999);
        assert!(s.start_tool_run(meta, "read", serde_json::json!({})).is_err());
    }

    #[test]
    fn cancelled_tool_run_ends_session_and_unknown_ops_are_loud() {
        let (_d, m) = test_manager();
        let s = session(&m);
        to_waiting(&s);
        let meta = op_meta(&m, s.id(), kilop_core::op::RecoveryStrategy::None);
        let op = meta.operation_id;
        s.start_tool_run(meta, "write_file", serde_json::json!({"path": "a"})).unwrap();
        // Cancelling the run ends the session (terminal by the core machine).
        s.finish_tool_run(op, "cancelled", EffectStatus::Unknown).unwrap();
        assert_eq!(s.state().unwrap(), AgentState::Cancelled);
        assert!(s.pending_tool_runs().unwrap().is_empty());
        // Finishing a finished op is a loud NotFound (no double finish).
        assert!(s.finish_tool_run(op, "completed", EffectStatus::Verified).is_err());
        // The registry forgot the op.
        assert!(s.abort(Some(op)).is_err());
    }

    #[test]
    fn completed_tool_run_moves_to_validating_and_unregisters() {
        let (_d, m) = test_manager();
        let s = session(&m);
        to_waiting(&s);
        let meta = op_meta(&m, s.id(), kilop_core::op::RecoveryStrategy::VerifyHash {
            path: "/w/a.txt".into(),
            expected: kilop_core::hash::FileHash::from([7; 32]),
        });
        let op = meta.operation_id;
        let handle = s.start_tool_run(meta, "write_file", serde_json::json!({"path": "a"})).unwrap();
        assert_eq!(handle.op_id, op);
        assert_eq!(s.state().unwrap(), AgentState::ExecutingTool);
        // The recovery strategy is durable in the row.
        let rows = s.pending_tool_runs().unwrap();
        assert_eq!(rows[0].recovery["strategy"], "verify_hash");
        assert_eq!(
            rows[0].expected_hash.as_deref(),
            Some(kilop_core::hash::FileHash::from([7; 32]).to_hex().as_str())
        );
        s.finish_tool_run(op, "completed", EffectStatus::Verified).unwrap();
        assert_eq!(s.state().unwrap(), AgentState::Validating);
        assert!(s.pending_tool_runs().unwrap().is_empty());
        // Duplicate finish is now a loud NotFound.
        assert!(s.finish_tool_run(op, "completed", EffectStatus::Verified).is_err());
    }

    #[test]
    fn request_permission_requires_tracked_op_and_persists() {
        let (_d, m) = test_manager();
        let s = session(&m);
        // No tracked op yet: loud NotFound.
        let err = s
            .request_permission(
                m.next_op_id(),
                &Capability::Network { destination: "https://x".into() },
            )
            .unwrap_err();
        assert_eq!(err.kind, kilop_core::ErrorKind::NotFound);
        // After a prompt the op is tracked; the machine must be at the tool
        // request point before the permission request journals.
        s.submit_prompt("go", &[]).unwrap();
        s.append_event(EventKind::ContextPrepared, AgentState::BuildingContext, None, None).unwrap();
        s.append_event(EventKind::ModelStarted, AgentState::WaitingForModel, None, None).unwrap();
        s.append_event(EventKind::ModelChunkReceived, AgentState::Streaming, None, None).unwrap();
        let op = s.ops().all()[0];
        let req = s
            .request_permission(op, &Capability::Git { operation: "push".into() })
            .unwrap();
        assert_eq!(req.op_id, op);
        assert_eq!(s.state().unwrap(), AgentState::WaitingForPermission);
        // The pending row round-trips to a Capability.
        let (_, rop, cap_str) = s.pending_permission(req.id).unwrap().unwrap();
        assert_eq!(rop, op);
        let cap: Capability = serde_json::from_str(&cap_str).unwrap();
        assert_eq!(cap, Capability::Git { operation: "push".into() });
        // Deny returns the session to ready and records PermissionDenied.
        s.resolve_permission(req.id, kilop_core::capability::PermissionDecision::Deny)
            .unwrap();
        assert_eq!(s.state().unwrap(), AgentState::ReadyForNextTurn);
        // Resolving again conflicts.
        assert!(s
            .resolve_permission(req.id, kilop_core::capability::PermissionDecision::Allow)
            .is_err());
        // The event's payload carries the frozen permission_id + capability.
        let events = s.events_range(1, None).unwrap();
        let req_ev = events
            .iter()
            .find(|e| e.kind == EventKind::ToolRequested)
            .unwrap();
        assert_eq!(req_ev.payload.as_ref().unwrap()["permission_id"], req.id);
        assert_eq!(req_ev.payload.as_ref().unwrap()["capability"], "git");
    }

    #[test]
    fn denied_parallel_tool_stays_executing() {
        let (_d, m) = test_manager();
        let s = session(&m);
        to_streaming(&s);
        let turn_op = s.ops().all()[0];
        // Request permission for a tool, then start it before the resolution
        // lands (the agent's auto-approval path): the session is ExecutingTool
        // while the permission is still pending. Denying it now cannot jump
        // ExecutingTool -> ReadyForNextTurn, so it stays executing.
        let req = s
            .request_permission(turn_op, &Capability::ReadWorkspace { path: "/w/b".into() })
            .unwrap();
        let meta = op_meta(&m, s.id(), kilop_core::op::RecoveryStrategy::None);
        s.start_tool_run(meta, "read_file", serde_json::json!({"path": "b"})).unwrap();
        s.resolve_permission(req.id, kilop_core::capability::PermissionDecision::Deny)
            .unwrap();
        assert_eq!(s.state().unwrap(), AgentState::ExecutingTool);
        // The denial was still journaled, with the state the machine actually
        // has.
        let ev = s
            .events_range(1, None)
            .unwrap()
            .into_iter()
            .find(|e| e.kind == EventKind::PermissionDenied)
            .expect("denial journaled");
        assert_eq!(ev.state, AgentState::ExecutingTool);
        // A clean deny from WaitingForPermission returns to ready.
        let req2 = s
            .request_permission(turn_op, &Capability::ReadWorkspace { path: "/w/d".into() })
            .unwrap();
        s.resolve_permission(req2.id, kilop_core::capability::PermissionDecision::Deny)
            .unwrap();
        assert_eq!(s.state().unwrap(), AgentState::ReadyForNextTurn);
    }

    #[test]
    fn ask_cannot_resolve_permission() {
        let (_d, m) = test_manager();
        let s = session(&m);
        to_streaming(&s);
        let op = s.ops().all()[0];
        let req = s
            .request_permission(op, &Capability::ReadWorkspace { path: "/w/a".into() })
            .unwrap();
        let err = s
            .resolve_permission(req.id, kilop_core::capability::PermissionDecision::Ask)
            .unwrap_err();
        assert_eq!(err.kind, kilop_core::ErrorKind::Malformed);
        // The permission stays pending and untouched.
        assert!(s.pending_permission(req.id).unwrap().is_some());
    }

    #[test]
    fn oversized_tool_args_rejected_before_write() {
        let (_d, m) = test_manager();
        let s = session(&m);
        s.submit_prompt("x", &[]).unwrap();
        for (k, st) in [
            (EventKind::ContextPrepared, AgentState::BuildingContext),
            (EventKind::ModelStarted, AgentState::WaitingForModel),
            (EventKind::ModelChunkReceived, AgentState::Streaming),
            (EventKind::ToolRequested, AgentState::WaitingForPermission),
        ] {
            s.append_event(k, st, None, None).unwrap();
        }
        let meta = op_meta(&m, s.id(), kilop_core::op::RecoveryStrategy::None);
        let big_args = serde_json::json!({ "blob": "x".repeat(MAX_TOOL_ARGS_BYTES + 1) });
        assert!(s.start_tool_run(meta, "run", big_args).is_err());
        assert!(s.pending_tool_runs().unwrap().is_empty());
    }

    #[test]
    fn expired_or_cancelled_op_meta_is_rejected_before_start() {
        let (_d, m) = test_manager();
        let s = session(&m);
        s.submit_prompt("x", &[]).unwrap();
        for (k, st) in [
            (EventKind::ContextPrepared, AgentState::BuildingContext),
            (EventKind::ModelStarted, AgentState::WaitingForModel),
            (EventKind::ModelChunkReceived, AgentState::Streaming),
            (EventKind::ToolRequested, AgentState::WaitingForPermission),
        ] {
            s.append_event(k, st, None, None).unwrap();
        }
        // Deadline already in the past.
        let mut meta = op_meta(&m, s.id(), kilop_core::op::RecoveryStrategy::None);
        meta.deadline = Deadline::at(m.now_ms() - 1);
        assert!(s.start_tool_run(meta, "read", serde_json::json!({})).is_err());
        // Cancelled token.
        let token = CancellationToken::new();
        token.cancel();
        let mut meta = op_meta(&m, s.id(), kilop_core::op::RecoveryStrategy::None);
        meta.cancellation = token;
        assert!(s.start_tool_run(meta, "read", serde_json::json!({})).is_err());
        assert!(s.pending_tool_runs().unwrap().is_empty());
    }

    #[test]
    fn provider_call_records_are_durable_and_bounded() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let op = m.next_op_id();
        let id = s
            .record_provider_call(op, "ollama", "qwen3.8", "ok", Some(100), Some(50), None)
            .unwrap();
        assert!(id > 0);
        let huge = "p".repeat(300);
        assert!(s.record_provider_call(op, &huge, "m", "ok", None, None, None).is_err());
    }

    #[test]
    fn abort_cancels_tool_token_before_durable_updates() {
        let (_d, m) = test_manager();
        let s = session(&m);
        to_waiting(&s);
        let meta = op_meta(&m, s.id(), kilop_core::op::RecoveryStrategy::None);
        let op = meta.operation_id;
        s.start_tool_run(meta, "run", serde_json::json!({})).unwrap();
        let tracked = s.ops().tracked(op).unwrap();
        assert!(!tracked.token.is_cancelled());
        let receipt = s.abort(Some(op)).unwrap();
        assert_eq!(receipt.op_ids, vec![op]);
        assert!(tracked.token.is_cancelled(), "abort must cancel the op token");
        // The durable row is finished with cancelled/unknown.
        let rows = s.pending_tool_runs().unwrap();
        assert!(rows.is_empty());
        // Aborting the tool op journals exactly one ToolCancelled event; the
        // turn op is untouched (a tool abort does not kill the session turn).
        let kinds: Vec<_> = s
            .events_range(1, None)
            .unwrap()
            .into_iter()
            .map(|e| e.kind)
            .collect();
        assert_eq!(kinds.iter().filter(|k| **k == EventKind::ToolCancelled).count(), 1);
        assert!(!kinds.contains(&EventKind::Failed));
        // A tool abort cancels the tool, not the session: the machine lands
        // ready for the next prompt (review P0-2).
        assert_eq!(s.state().unwrap(), AgentState::ReadyForNextTurn);
    }
}
