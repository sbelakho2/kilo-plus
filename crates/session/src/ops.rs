//! Operation ledger: tool runs, provider calls, permission requests, and the
//! in-memory registry that gives the session ownership of every in-flight op's
//! cancellation token.

use std::collections::HashMap;
use std::sync::Mutex;

use faktor_core::cancellation::CancellationToken;
use faktor_core::capability::Capability;
use faktor_core::id::OpId;
use faktor_core::op::{EffectStatus, OpMeta};
use faktor_core::state::AgentState;
use faktor_store::ToolRunRow;

use crate::handle::SessionHandle;
use crate::{effect_str, json_bytes, SessionError, MAX_TOOL_ARGS_BYTES};

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
    pub event_seq: faktor_core::id::EventSeq,
}

fn capability_tag(cap: &Capability) -> String {
    serde_json::to_value(cap)
        .ok()
        .and_then(|v| {
            v.get("capability")
                .and_then(|t| t.as_str())
                .map(String::from)
        })
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
    ) -> faktor_core::Result<ToolRunHandle> {
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
            faktor_core::op::RecoveryStrategy::VerifyHash { expected, .. } => {
                Some(expected.to_hex())
            }
            _ => None,
        };
        // Validate the transition before any durable write.
        crate::journal::validate_transition(
            self.state()?,
            faktor_core::event::EventKind::ToolStarted,
            AgentState::ExecutingTool,
        )?;
        let row_id = self
            .manager
            .store()
            .start_tool_run(
                self.id,
                op.operation_id,
                tool,
                args,
                recovery,
                expected_hash,
                op.replay.clone(),
            )
            .map_err(crate::map_store_err)?;
        self.transition_locked(
            faktor_core::event::EventKind::ToolStarted,
            AgentState::ExecutingTool,
            Some(op.operation_id),
            Some(serde_json::json!({ "tool": tool })),
        )?;
        self.ops()
            .register(op.operation_id, OpKind::Tool, op.cancellation.clone());
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
    ) -> faktor_core::Result<()> {
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
            "completed" => (
                faktor_core::event::EventKind::ToolCompleted,
                AgentState::Validating,
            ),
            "failed" => (
                faktor_core::event::EventKind::ToolCompleted,
                AgentState::FailedRecoverable,
            ),
            "cancelled" => (
                faktor_core::event::EventKind::ToolCancelled,
                AgentState::Cancelled,
            ),
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

    pub fn set_tool_run_effect(&self, op: OpId, effect: EffectStatus) -> faktor_core::Result<()> {
        self.manager
            .store()
            .set_tool_run_effect(self.id, op, effect_str(effect))
            .map_err(|e| crate::map_store_err(e).into())
    }

    pub fn pending_tool_runs(&self) -> faktor_core::Result<Vec<ToolRunRow>> {
        self.manager
            .store()
            .pending_tool_runs(self.id)
            .map_err(|e| crate::map_store_err(e).into())
    }

    /// Record the workspace-write postcondition a tool reported at execution
    /// end (v7): crash recovery verifies the CURRENT file bytes against it
    /// through the workspace file service. Only a still-running row may be
    /// annotated (loud otherwise).
    pub fn record_tool_postcondition(
        &self,
        op: OpId,
        postcondition: &serde_json::Value,
    ) -> faktor_core::Result<()> {
        self.manager
            .store()
            .record_tool_postcondition(self.id, op, postcondition)
            .map_err(|e| crate::map_store_err(e).into())
    }

    /// Bump the physical-attempt counter of one still-running tool run (a
    /// crash-recovery replay is a new physical attempt of the same logical
    /// operation). Returns the new attempt number.
    pub fn bump_tool_attempt(&self, op: OpId) -> faktor_core::Result<i64> {
        self.manager
            .store()
            .bump_tool_run_attempt(self.id, op)
            .map_err(|e| crate::map_store_err(e).into())
    }

    /// Durably open the record of an admitted logical turn (v7). The runtime
    /// drives the turn with exactly this identity; recovery never synthesizes
    /// one. Re-admission of the same turn op upserts the same record; other
    /// active records of the session are finalized as failed (at most one
    /// active logical turn per session).
    #[allow(clippy::too_many_arguments)]
    pub fn start_turn_record(
        &self,
        turn_op: OpId,
        queue_seq: Option<i64>,
        prompt_message_id: Option<i64>,
        provider: &str,
        model: &str,
        variant: Option<&str>,
    ) -> faktor_core::Result<i64> {
        if provider.len() > 256 || model.len() > 256 {
            return Err(SessionError::Oversized("provider/model name too long".into()).into());
        }
        self.manager
            .store()
            .start_turn_record(
                self.id,
                turn_op,
                queue_seq,
                prompt_message_id,
                provider,
                model,
                variant,
            )
            .map_err(|e| crate::map_store_err(e).into())
    }

    /// Finalize the recorded envelope at logical-turn start: the effective
    /// provider/model (per-message override wins over the session default),
    /// the reasoning variant and the tool-call mode. Only an active record
    /// is updated.
    pub fn set_turn_envelope(
        &self,
        turn_op: OpId,
        provider: &str,
        model: &str,
        variant: Option<&str>,
        tool_mode: Option<&str>,
    ) -> faktor_core::Result<()> {
        self.manager
            .store()
            .set_turn_record_envelope(self.id, turn_op, provider, model, variant, tool_mode)
            .map_err(|e| crate::map_store_err(e).into())
            .map(|_| ())
    }

    /// Close an active turn record. Idempotent: absent or already-closed
    /// records are a no-op (returns whether anything was updated).
    pub fn finish_turn_record(&self, turn_op: OpId, status: &str) -> faktor_core::Result<bool> {
        self.manager
            .store()
            .finish_turn_record(self.id, turn_op, status)
            .map_err(|e| crate::map_store_err(e).into())
    }

    /// The session's single active logical-turn record (v7). `None` when no
    /// prompt was admitted as an active turn (e.g. everything is queued).
    pub fn active_turn_record(&self) -> faktor_core::Result<Option<faktor_store::TurnRecordRow>> {
        self.manager
            .store()
            .active_turn_record(self.id)
            .map_err(|e| crate::map_store_err(e).into())
    }

    pub fn turn_record(
        &self,
        turn_op: OpId,
    ) -> faktor_core::Result<Option<faktor_store::TurnRecordRow>> {
        self.manager
            .store()
            .turn_record_of(self.id, turn_op)
            .map_err(|e| crate::map_store_err(e).into())
    }

    /// Every turn record of the session (oldest first; diagnostics/tests).
    pub fn turn_records(&self) -> faktor_core::Result<Vec<faktor_store::TurnRecordRow>> {
        self.manager
            .store()
            .turn_records_of(self.id)
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
    ) -> faktor_core::Result<i64> {
        if provider.len() > 256 || model.len() > 256 {
            return Err(SessionError::Oversized("provider/model name too long".into()).into());
        }
        Ok(self
            .manager
            .store()
            .record_provider_call(
                self.id, op, provider, model, status, tokens_in, tokens_out, error,
            )
            .map_err(crate::map_store_err)?)
    }

    /// Async twin of [`SessionHandle::record_provider_call`] (usage
    /// settlement, audit 13/42): one provider usage frame lands in the
    /// durable `provider_call` rows through the manager's DbActor, and the
    /// caller awaits the fsynced response.
    #[allow(clippy::too_many_arguments)]
    pub async fn settle_usage(
        &self,
        op: OpId,
        provider: &str,
        model: &str,
        status: &str,
        tokens_in: Option<u64>,
        tokens_out: Option<u64>,
        error: Option<&str>,
    ) -> faktor_core::Result<i64> {
        if provider.len() > 256 || model.len() > 256 {
            return Err(SessionError::Oversized("provider/model name too long".into()).into());
        }
        let handle = self.manager.actor().handle();
        Ok(handle
            .settle_usage(
                self.id, op, provider, model, status, tokens_in, tokens_out, error,
            )
            .await
            .map_err(crate::map_store_err)?)
    }

    /// Prefix-cache settlement twin (audits 65-66 fill site, architecture
    /// §8.4): the usage row of a completed provider call additionally
    /// records the byte-truth prefix observation the runtime measured —
    /// `prompt_prefix_hash` (digest of the exact cacheable-prefix bytes the
    /// sent request carried: StaticPrefix + SemiStable) and `prompt_tokens`
    /// (that prefix's estimated token count) — and the row's per-turn
    /// `prefix_stability` against the session's PREVIOUS observation
    /// (1.0 for the first observation and for byte-identical prefixes;
    /// prev/cur under append-consistent growth; 0.0 on a rewrite), mirroring
    /// the documented pair rule in `faktor-router::stability` that the
    /// routing layer applies over these durable rows.
    ///
    /// Unlike [`SessionHandle::settle_usage`] this twin is synchronous and
    /// writes DIRECTLY through the store: the DbActor's hot-write batch
    /// shape (`HotWrite::RecordProviderCall`) predates the v13 prefix
    /// columns, and the prefix fill happens once per completed call, not on
    /// a hot chunk path — durability semantics are identical (one
    /// transaction, fsynced). Prefix-less calls (both `None`) land exactly
    /// like a plain `record_provider_call` with the stability columns NULL.
    ///
    /// A row that carries a prefix hash but NO stability never exists: the
    /// stability is derived here from byte truth (previous observation
    /// hash/count), never accepted from callers, and corrupt previous rows
    /// fail loudly (`Malformed`) instead of silently mispairing.
    #[allow(clippy::too_many_arguments)]
    pub fn settle_usage_with_prefix(
        &self,
        op: OpId,
        provider: &str,
        model: &str,
        status: &str,
        tokens_in: Option<u64>,
        tokens_out: Option<u64>,
        error: Option<&str>,
        prompt_prefix_hash: Option<[u8; 32]>,
        prompt_tokens: Option<u64>,
    ) -> faktor_core::Result<i64> {
        if provider.len() > 256 || model.len() > 256 {
            return Err(SessionError::Oversized("provider/model name too long".into()).into());
        }
        let stability = match (prompt_prefix_hash, prompt_tokens) {
            (Some(hash), Some(tokens)) => {
                let prev = self.last_prefix_observation()?;
                Some(prefix_pair_stability(
                    prev.as_ref()
                        .map(|p| (p.prompt_prefix_hash, p.prompt_tokens)),
                    hash,
                    tokens,
                ))
            }
            _ => None,
        };
        Ok(self
            .manager
            .store()
            .record_provider_call_with_prefix(
                self.id,
                op,
                provider,
                model,
                status,
                tokens_in,
                tokens_out,
                error,
                prompt_prefix_hash,
                prompt_tokens,
                stability,
            )
            .map_err(crate::map_store_err)?)
    }

    /// The session's NEWEST durable prefix observation (v13): the last
    /// `provider_call` row of this session carrying a prefix hash, `None`
    /// when none was recorded yet (or the session predates v13). Read-time
    /// store validation is loud: a corrupt previous observation fails the
    /// settlement instead of silently mispairing the stability.
    fn last_prefix_observation(
        &self,
    ) -> faktor_core::Result<Option<faktor_store::ProviderCallPrefixRow>> {
        Ok(self
            .manager
            .store()
            .provider_call_prefix_rows(self.id)
            .map_err(crate::map_store_err)?
            .into_iter()
            .next_back())
    }

    /// The session's stored prefix-stability aggregate (v13): mean/std-dev
    /// over the recorded per-row prefix stabilities, `None` while no
    /// observation row carries one (fresh session or pre-v13 rows only).
    /// Read-only; hostile stored values surface as loud `Malformed`.
    pub fn stored_prefix_stability(
        &self,
    ) -> faktor_core::Result<Option<faktor_store::PrefixStabilityAggregate>> {
        self.manager
            .store()
            .session_stored_prefix_stability(self.id)
            .map_err(crate::map_store_err)
            .map_err(Into::into)
    }

    /// Request permission to use `capability` for `op`. Journals
    /// `ToolRequested` (recorded with state `WaitingForPermission` — the
    /// documented two-hop) and inserts the durable pending row.
    pub fn request_permission(
        &self,
        op: OpId,
        capability: &Capability,
    ) -> faktor_core::Result<PermissionRequest> {
        let _guard = self.command_guard();
        if self.ops().tracked(op).is_none() {
            return Err(SessionError::NotFound(format!("operation {op} is not tracked")).into());
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
            faktor_core::event::EventKind::ToolRequested,
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
        decision: faktor_core::capability::PermissionDecision,
    ) -> faktor_core::Result<faktor_core::id::EventSeq> {
        let (decision_str, kind, target) = match decision {
            faktor_core::capability::PermissionDecision::Allow => (
                "allow",
                faktor_core::event::EventKind::PermissionGranted,
                AgentState::ExecutingTool,
            ),
            faktor_core::capability::PermissionDecision::Deny => (
                "deny",
                faktor_core::event::EventKind::PermissionDenied,
                AgentState::ReadyForNextTurn,
            ),
            faktor_core::capability::PermissionDecision::Ask => {
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
        if self
            .manager
            .store()
            .pending_permission(id)
            .map_err(crate::map_store_err)?
            .is_some()
        {
            return Err(SessionError::Conflict(format!(
                "permission {id} was resolved concurrently"
            ))
            .into());
        }
        // Deny under a parallel tool: staying ExecutingTool is the honest
        // machine outcome (ExecutingTool cannot go to ReadyForNextTurn).
        let target = if kind == faktor_core::event::EventKind::PermissionDenied {
            let current = self.state()?;
            let mut m = faktor_core::state::StateMachine::new(current);
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
    ) -> faktor_core::Result<Option<(faktor_core::id::SessionId, OpId, String)>> {
        self.manager
            .store()
            .pending_permission(id)
            .map_err(|e| crate::map_store_err(e).into())
    }
}

/// Per-row prefix-cache stability of one new observation against the
/// session's previous one (audits 65-66). Deterministic mirror of the
/// documented pair rule in `faktor-router::stability`, which the routing
/// layer applies over the durable rows this twin fills — keep both in
/// lockstep:
///
/// ```text
/// stability = 1.0     first observation (nothing precedes it)
///           = 1.0     either prefix is EMPTY (0 tokens): an empty prefix
///                     destabilizes nothing (documented convention)
///           = 1.0     byte-identical digests: full cache coverage
///           = p / c   strict token growth with different bytes: the only
///                     byte relation consistent with append-only growth —
///                     the previous prefix is fully covered by the current
///           = 0.0     otherwise: same-length or shorter prefix with
///                     different bytes is a REWRITE (reorder/churn)
/// ```
///
/// Always finite in [0, 1]; `cur_tokens` beyond the store's u32 bound is
/// rejected loudly by the insert itself, never silently stored.
fn prefix_pair_stability(
    prev: Option<([u8; 32], u32)>,
    cur_hash: [u8; 32],
    cur_tokens: u64,
) -> f64 {
    let Some((prev_hash, prev_tokens)) = prev else {
        return 1.0;
    };
    if prev_tokens == 0 || cur_tokens == 0 {
        return 1.0;
    }
    if prev_hash == cur_hash {
        return 1.0;
    }
    let (p, c) = (f64::from(prev_tokens), cur_tokens as f64);
    if cur_tokens > u64::from(prev_tokens) {
        p / c
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::tests::{session, test_manager};
    use crate::SessionManager;
    use faktor_core::event::EventKind;
    use faktor_core::id::SessionId;
    use faktor_core::time::Deadline;

    fn op_meta(
        m: &crate::SessionManager,
        s: SessionId,
        recovery: faktor_core::op::RecoveryStrategy,
    ) -> OpMeta {
        let op = m.next_op_id();
        OpMeta::new(
            op,
            s,
            Deadline::at(m.now_ms() + 60_000),
            faktor_core::retry::RetryPolicy::default(),
            CancellationToken::new(),
            recovery,
            m.now_ms(),
        )
    }

    fn to_streaming(s: &SessionHandle) {
        s.submit_prompt("x", &[]).unwrap();
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
    }

    fn to_waiting(s: &SessionHandle) {
        to_streaming(s);
        let turn_op = s.ops().all()[0];
        s.request_permission(
            turn_op,
            &Capability::ReadWorkspace {
                path: "/w/a".into(),
            },
        )
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
                &Capability::ExecuteShell {
                    command: "cargo test".into(),
                },
            )
            .unwrap();
        // Two resolvers race: allow vs deny.
        let s = std::sync::Arc::new(s);
        let s1 = s.clone();
        let s2 = s.clone();
        let t1 = std::thread::spawn(move || {
            s1.resolve_permission(req.id, faktor_core::capability::PermissionDecision::Allow)
        });
        let t2 = std::thread::spawn(move || {
            s2.resolve_permission(req.id, faktor_core::capability::PermissionDecision::Deny)
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
        let meta = op_meta(&m, s.id(), faktor_core::op::RecoveryStrategy::None);
        let err = s
            .start_tool_run(meta, "read_file", serde_json::json!({}))
            .unwrap_err();
        assert!(matches!(
            err.kind,
            faktor_core::ErrorKind::InvalidState { .. }
        ));
        assert!(s.pending_tool_runs().unwrap().is_empty(), "no tool_run row");
        assert_eq!(s.last_event_seq().unwrap().unwrap().raw(), 2);
    }

    #[test]
    fn foreign_op_meta_rejected_by_start_tool_run() {
        let (_d, m) = test_manager();
        let s = session(&m);
        s.submit_prompt("x", &[]).unwrap();
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
        // An op envelope pointed at another session is rejected loudly.
        let mut meta = op_meta(&m, s.id(), faktor_core::op::RecoveryStrategy::None);
        meta.session_id = SessionId::new(999);
        assert!(s
            .start_tool_run(meta, "read", serde_json::json!({}))
            .is_err());
    }

    #[test]
    fn cancelled_tool_run_ends_session_and_unknown_ops_are_loud() {
        let (_d, m) = test_manager();
        let s = session(&m);
        to_waiting(&s);
        let meta = op_meta(&m, s.id(), faktor_core::op::RecoveryStrategy::None);
        let op = meta.operation_id;
        s.start_tool_run(meta, "write_file", serde_json::json!({"path": "a"}))
            .unwrap();
        // Cancelling the run ends the session (terminal by the core machine).
        s.finish_tool_run(op, "cancelled", EffectStatus::Unknown)
            .unwrap();
        assert_eq!(s.state().unwrap(), AgentState::Cancelled);
        assert!(s.pending_tool_runs().unwrap().is_empty());
        // Finishing a finished op is a loud NotFound (no double finish).
        assert!(s
            .finish_tool_run(op, "completed", EffectStatus::Verified)
            .is_err());
        // The registry forgot the op.
        assert!(s.abort(Some(op)).is_err());
    }

    #[test]
    fn completed_tool_run_moves_to_validating_and_unregisters() {
        let (_d, m) = test_manager();
        let s = session(&m);
        to_waiting(&s);
        let meta = op_meta(
            &m,
            s.id(),
            faktor_core::op::RecoveryStrategy::VerifyHash {
                path: "/w/a.txt".into(),
                expected: faktor_core::hash::FileHash::from([7; 32]),
            },
        );
        let op = meta.operation_id;
        let handle = s
            .start_tool_run(meta, "write_file", serde_json::json!({"path": "a"}))
            .unwrap();
        assert_eq!(handle.op_id, op);
        assert_eq!(s.state().unwrap(), AgentState::ExecutingTool);
        // The recovery strategy is durable in the row.
        let rows = s.pending_tool_runs().unwrap();
        assert_eq!(rows[0].recovery["strategy"], "verify_hash");
        assert_eq!(
            rows[0].expected_hash.as_deref(),
            Some(faktor_core::hash::FileHash::from([7; 32]).to_hex().as_str())
        );
        s.finish_tool_run(op, "completed", EffectStatus::Verified)
            .unwrap();
        assert_eq!(s.state().unwrap(), AgentState::Validating);
        assert!(s.pending_tool_runs().unwrap().is_empty());
        // Duplicate finish is now a loud NotFound.
        assert!(s
            .finish_tool_run(op, "completed", EffectStatus::Verified)
            .is_err());
    }

    #[test]
    fn request_permission_requires_tracked_op_and_persists() {
        let (_d, m) = test_manager();
        let s = session(&m);
        // No tracked op yet: loud NotFound.
        let err = s
            .request_permission(
                m.next_op_id(),
                &Capability::Network {
                    destination: "https://x".into(),
                },
            )
            .unwrap_err();
        assert_eq!(err.kind, faktor_core::ErrorKind::NotFound);
        // After a prompt the op is tracked; the machine must be at the tool
        // request point before the permission request journals.
        s.submit_prompt("go", &[]).unwrap();
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
        let op = s.ops().all()[0];
        let req = s
            .request_permission(
                op,
                &Capability::Git {
                    operation: "push".into(),
                },
            )
            .unwrap();
        assert_eq!(req.op_id, op);
        assert_eq!(s.state().unwrap(), AgentState::WaitingForPermission);
        // The pending row round-trips to a Capability.
        let (_, rop, cap_str) = s.pending_permission(req.id).unwrap().unwrap();
        assert_eq!(rop, op);
        let cap: Capability = serde_json::from_str(&cap_str).unwrap();
        assert_eq!(
            cap,
            Capability::Git {
                operation: "push".into()
            }
        );
        // Deny returns the session to ready and records PermissionDenied.
        s.resolve_permission(req.id, faktor_core::capability::PermissionDecision::Deny)
            .unwrap();
        assert_eq!(s.state().unwrap(), AgentState::ReadyForNextTurn);
        // Resolving again conflicts.
        assert!(s
            .resolve_permission(req.id, faktor_core::capability::PermissionDecision::Allow)
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
            .request_permission(
                turn_op,
                &Capability::ReadWorkspace {
                    path: "/w/b".into(),
                },
            )
            .unwrap();
        let meta = op_meta(&m, s.id(), faktor_core::op::RecoveryStrategy::None);
        s.start_tool_run(meta, "read_file", serde_json::json!({"path": "b"}))
            .unwrap();
        s.resolve_permission(req.id, faktor_core::capability::PermissionDecision::Deny)
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
            .request_permission(
                turn_op,
                &Capability::ReadWorkspace {
                    path: "/w/d".into(),
                },
            )
            .unwrap();
        s.resolve_permission(req2.id, faktor_core::capability::PermissionDecision::Deny)
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
            .request_permission(
                op,
                &Capability::ReadWorkspace {
                    path: "/w/a".into(),
                },
            )
            .unwrap();
        let err = s
            .resolve_permission(req.id, faktor_core::capability::PermissionDecision::Ask)
            .unwrap_err();
        assert_eq!(err.kind, faktor_core::ErrorKind::Malformed);
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
        let meta = op_meta(&m, s.id(), faktor_core::op::RecoveryStrategy::None);
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
        let mut meta = op_meta(&m, s.id(), faktor_core::op::RecoveryStrategy::None);
        meta.deadline = Deadline::at(m.now_ms() - 1);
        assert!(s
            .start_tool_run(meta, "read", serde_json::json!({}))
            .is_err());
        // Cancelled token.
        let token = CancellationToken::new();
        token.cancel();
        let mut meta = op_meta(&m, s.id(), faktor_core::op::RecoveryStrategy::None);
        meta.cancellation = token;
        assert!(s
            .start_tool_run(meta, "read", serde_json::json!({}))
            .is_err());
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
        assert!(s
            .record_provider_call(op, &huge, "m", "ok", None, None, None)
            .is_err());
    }

    #[test]
    fn abort_cancels_tool_token_before_durable_updates() {
        let (_d, m) = test_manager();
        let s = session(&m);
        to_waiting(&s);
        let meta = op_meta(&m, s.id(), faktor_core::op::RecoveryStrategy::None);
        let op = meta.operation_id;
        s.start_tool_run(meta, "run", serde_json::json!({}))
            .unwrap();
        let tracked = s.ops().tracked(op).unwrap();
        assert!(!tracked.token.is_cancelled());
        let receipt = s.abort(Some(op)).unwrap();
        assert_eq!(receipt.op_ids, vec![op]);
        assert!(
            tracked.token.is_cancelled(),
            "abort must cancel the op token"
        );
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
        assert_eq!(
            kinds
                .iter()
                .filter(|k| **k == EventKind::ToolCancelled)
                .count(),
            1
        );
        assert!(!kinds.contains(&EventKind::Failed));
        // A tool abort cancels the tool, not the session: the machine lands
        // ready for the next prompt (review P0-2).
        assert_eq!(s.state().unwrap(), AgentState::ReadyForNextTurn);
    }

    #[test]
    fn tool_run_carries_replay_descriptor_attempt_and_postcondition() {
        // v7: an idempotent run's row stores its replay descriptor; the
        // attempt counter starts at 0 and a recovery replay bumps it; the
        // workspace-write postcondition is annotated before the finish.
        let (_d, m) = test_manager();
        let s = session(&m);
        to_waiting(&s);
        let mut meta = op_meta(&m, s.id(), faktor_core::op::RecoveryStrategy::Idempotent);
        meta = meta.with_replay(serde_json::json!({
            "tool_name": "echo",
            "validated_args": {"x": 1},
        }));
        let op = meta.operation_id;
        let handle = s
            .start_tool_run(meta, "echo", serde_json::json!({"x": 1}))
            .unwrap();
        let rows = s.pending_tool_runs().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].attempt, 0, "original physical attempt is 0");
        assert_eq!(
            rows[0].replay_descriptor.as_ref().unwrap()["tool_name"],
            "echo"
        );
        assert_eq!(s.bump_tool_attempt(op).unwrap(), 1, "a replay is attempt 1");
        let pc = serde_json::json!({
            "workspace_id": 1,
            "worktree_id": 1,
            "relative_path": "a.txt",
            "expected_hash": "ab".repeat(32),
        });
        s.record_tool_postcondition(op, &pc).unwrap();
        assert_eq!(
            s.pending_tool_runs().unwrap()[0]
                .postcondition
                .as_ref()
                .unwrap()["relative_path"],
            "a.txt"
        );
        // Annotation of a finished row is loud (never a silent ignore).
        s.finish_tool_run(op, "completed", EffectStatus::Applied)
            .unwrap();
        assert!(s.record_tool_postcondition(op, &pc).is_err());
        assert!(s.bump_tool_attempt(op).is_err());
        assert_eq!(handle.op_id, op);
        // The row kept ONE identity throughout: same op, no duplicates.
        assert!(s.pending_tool_runs().unwrap().is_empty());
    }

    // ------------------------------------------------- prefix-cache rows
    // (audits 65-66 fill site): settle_usage_with_prefix records
    // byte-truth prefix observations and derives the per-row stability
    // against the session's previous observation.

    /// Test-only digest: distinct byte strings map to distinct 32-byte
    /// digests (FNV-1a lanes, the same shape the router stability tests
    /// use — no external hash dependency needed for row math).
    fn test_digest(bytes: &[u8]) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (k, basis) in [
            0xcbf29ce484222325u64,
            0x84222325,
            0x9e3779b97f4a7c15,
            0x100000001b3,
        ]
        .iter()
        .enumerate()
        {
            let mut h = *basis ^ 0xdead_beef_1234_5678u64.wrapping_mul(k as u64 + 1);
            for &b in bytes {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x100000001b3);
            }
            let lane = h.to_le_bytes();
            out[k * 8..k * 8 + 8].copy_from_slice(&lane);
        }
        out
    }

    fn prefix_rows(s: &SessionHandle) -> Vec<faktor_store::ProviderCallPrefixRow> {
        s.manager.store().provider_call_prefix_rows(s.id()).unwrap()
    }

    fn settle_bytes(s: &SessionHandle, op: OpId, bytes: &[u8]) -> i64 {
        // Byte-truth settle: hash the EXACT prefix bytes and count the
        // bytes as the token proxy — equal bytes → equal count, so
        // consecutive identical states are byte-identical observations.
        s.settle_usage_with_prefix(
            op,
            "fake",
            "m",
            "completed",
            Some(100),
            Some(50),
            None,
            Some(test_digest(bytes)),
            Some(bytes.len().max(1) as u64),
        )
        .unwrap()
    }

    #[test]
    fn prefix_observations_record_stability_over_turns_and_survive_reopen() {
        // The fill contract end to end at the session chokepoint: after
        // three byte-identical turns the rows carry hash/tokens/stability
        // (1.0 — byte truth), a reordering turn (same token count, rewritten
        // bytes) records 0.0, an append-consistent growth records the
        // coverage ratio, and every observation survives a store reopen.
        let dir = tempfile::tempdir().unwrap();
        let m =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let s = session(&m);
        let sid = s.id();
        let op = || m.next_op_id();
        // Turn 1-3: identical prompt heads.
        settle_bytes(&s, op(), b"static prefix bytes");
        settle_bytes(&s, op(), b"static prefix bytes");
        settle_bytes(&s, op(), b"static prefix bytes");
        // Turn 4: a REWRITE of the head — same estimated token count (the
        // estimator counts chars/3 vs chars/3.4; these differ by one byte,
        // same token estimate... choose equal-length distinct bytes).
        settle_bytes(&s, op(), b"static prefix bxtes");
        // Turn 5: append-consistent growth (old head is a byte-prefix).
        settle_bytes(&s, op(), b"static prefix bytes plus more");

        let rows = prefix_rows(&s);
        assert_eq!(rows.len(), 5, "one observation per settled call");
        assert!(
            rows.iter()
                .all(|r| r.prompt_tokens > 0 && r.prefix_stability.is_some()),
            "every observation row must carry tokens and stability: {rows:?}"
        );
        assert_eq!(
            rows[0].prefix_stability,
            Some(1.0),
            "first observation is stable by definition"
        );
        assert_eq!(rows[1].prefix_stability, Some(1.0), "byte-identical head");
        assert_eq!(rows[2].prefix_stability, Some(1.0), "byte-identical head");
        assert_eq!(
            rows[3].prefix_stability,
            Some(0.0),
            "same-count rewrite is 0.0"
        );
        // Row 5: growth with a different digest is append-consistent.
        let p = rows[3].prompt_tokens as f64;
        let c = rows[4].prompt_tokens as f64;
        assert_eq!(rows[4].prefix_stability, Some(p / c));
        // Hashes are byte truth: equal bytes → equal digests.
        assert_eq!(rows[0].prompt_prefix_hash, rows[1].prompt_prefix_hash);
        assert_eq!(rows[0].prompt_prefix_hash, rows[2].prompt_prefix_hash);
        assert_ne!(rows[3].prompt_prefix_hash, rows[4].prompt_prefix_hash);
        // Store aggregate mirrors the recorded series (mean incl. row 1).
        let agg = s.stored_prefix_stability().unwrap().unwrap();
        assert_eq!(agg.observations, 5);
        let mean = (1.0 + 1.0 + 1.0 + 0.0 + p / c) / 5.0;
        assert!((agg.mean - mean).abs() < 1e-12);

        // Reopen: the rows are durable — release every handle, then a fresh
        // manager over the same dir reads the identical observations and
        // aggregate.
        drop(s);
        drop(m);
        let m2 =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let s2 = m2.get_session(sid).unwrap().unwrap();
        let rows2 = prefix_rows(&s2);
        assert_eq!(rows2, rows, "observations must survive the reopen");
        let agg2 = s2.stored_prefix_stability().unwrap().unwrap();
        assert_eq!(agg2.observations, agg.observations);
        assert!((agg2.mean - agg.mean).abs() < 1e-12);
        // A new observation on the reopened session chains off the last row.
        settle_bytes(&s2, m2.next_op_id(), b"static prefix bytes");
        let rows3 = prefix_rows(&s2);
        assert_eq!(rows3.len(), 6);
        assert_eq!(
            rows3[5].prefix_stability,
            Some(0.0),
            "row 6 vs the grown row 5 is a shrink-rewrite"
        );
    }

    #[test]
    fn prefix_observations_are_per_session_and_prefix_less_rows_never_chain() {
        // Two sessions interleave: each session's stability chains against
        // ITS OWN previous observation only, and rows settled without a
        // prefix (started/failed frames, pre-v13 shapes) never join the
        // observation series nor move the chain.
        let (_d, m) = test_manager();
        let a = session(&m);
        let b = session(&m);
        let op = || m.next_op_id();
        settle_bytes(&a, op(), b"session A head");
        settle_bytes(&b, op(), b"session B head");
        settle_bytes(&a, op(), b"session A head");
        // Prefix-less frame between A's observations: a NULL-prefix row that
        // must not appear in the series nor reset the chain.
        a.record_provider_call(op(), "fake", "m", "started", None, None, None)
            .unwrap();
        settle_bytes(&a, op(), b"session A head");
        let ra = prefix_rows(&a);
        assert_eq!(ra.len(), 3);
        assert!(
            ra.iter().all(|r| r.prefix_stability == Some(1.0)),
            "A's chain must stay stable: {ra:?}"
        );
        let rb = prefix_rows(&b);
        assert_eq!(rb.len(), 1);
        assert_eq!(rb[0].prefix_stability, Some(1.0));
        // A's rows never leak into B's chain: rewrite A, B stays 1.0.
        settle_bytes(&a, op(), b"session A heab");
        let rb = prefix_rows(&b);
        assert_eq!(rb.len(), 1);
        assert_eq!(rb[0].prefix_stability, Some(1.0));
        let ra = prefix_rows(&a);
        assert_eq!(ra.len(), 4);
        assert_eq!(ra[3].prefix_stability, Some(0.0));
    }

    #[test]
    fn prefix_pair_stability_rule_is_deterministic_and_total() {
        // Adversarial row math: every branch of the mirror rule, including
        // hostile inputs that must never panic or escape [0, 1].
        let h = test_digest(b"head");
        assert_eq!(prefix_pair_stability(None, h, 10), 1.0);
        // Empty prefixes (0 tokens) destabilize nothing.
        assert_eq!(prefix_pair_stability(Some((h, 0)), h, 10), 1.0);
        assert_eq!(prefix_pair_stability(Some((h, 10)), h, 0), 1.0);
        // Identical digests win over any count difference.
        assert_eq!(prefix_pair_stability(Some((h, 10)), h, 5), 1.0);
        assert_eq!(prefix_pair_stability(Some((h, 10)), h, 10), 1.0);
        assert_eq!(prefix_pair_stability(Some((h, 10)), h, u64::MAX), 1.0);
        // Different digest, strict growth: coverage ratio.
        let g = test_digest(b"head plus");
        assert_eq!(prefix_pair_stability(Some((h, 10)), g, 20), 0.5);
        assert_eq!(prefix_pair_stability(Some((h, 3)), g, 4), 0.75);
        assert_eq!(prefix_pair_stability(Some((h, 1)), g, 2), 0.5);
        // Different digest, equal or shorter: rewrite → 0.0.
        assert_eq!(prefix_pair_stability(Some((h, 10)), g, 10), 0.0);
        assert_eq!(prefix_pair_stability(Some((h, 10)), g, 9), 0.0);
        // A hostile u64 current count with a u32-max previous count is
        // strict growth: the append-consistent coverage ratio, never 0/NaN.
        assert_eq!(
            prefix_pair_stability(Some((h, u32::MAX)), g, u64::MAX),
            f64::from(u32::MAX) / u64::MAX as f64
        );
        // Always finite and in [0, 1].
        for (p, cur) in [
            (Some((h, 1)), 3u64),
            (Some((h, u32::MAX)), 3u64),
            (None, 0u64),
        ] {
            let v = prefix_pair_stability(p, g, cur);
            assert!(v.is_finite() && (0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn settle_usage_with_prefix_rejects_hostile_inputs_loudly() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let h = test_digest(b"head");
        // Oversized provider/model names are rejected before any write.
        let huge = "p".repeat(300);
        assert!(s
            .settle_usage_with_prefix(
                m.next_op_id(),
                &huge,
                "m",
                "completed",
                None,
                None,
                None,
                Some(h),
                Some(1),
            )
            .is_err());
        assert!(s
            .settle_usage_with_prefix(
                m.next_op_id(),
                "fake",
                &"m".repeat(300),
                "completed",
                None,
                None,
                None,
                Some(h),
                Some(1),
            )
            .is_err());
        // A prompt token count beyond the u32 bound fails loudly (the
        // store's oversized guard surfaces through the session twin's
        // error mapping) and writes nothing.
        let err = s
            .settle_usage_with_prefix(
                m.next_op_id(),
                "fake",
                "m",
                "completed",
                None,
                None,
                None,
                Some(h),
                Some(1 << 33),
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("u32 prefix-token bound"),
            "the store's loud oversized guard must surface: {err}"
        );
        // Nothing landed (the first hostile settle was also rejected).
        assert!(prefix_rows(&s).is_empty());
        // Prefix-less settles land with NULL prefix columns and are excluded
        // from the observation series (like plain settle_usage rows).
        s.record_provider_call(m.next_op_id(), "fake", "m", "started", None, None, None)
            .unwrap();
        assert!(prefix_rows(&s).is_empty());
        assert!(s.stored_prefix_stability().unwrap().is_none());
    }
}
