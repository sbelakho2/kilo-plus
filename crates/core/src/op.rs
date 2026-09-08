//! Operation metadata. Every asynchronous operation carries the full envelope
//! from the spec: `operation_id, session_id, state, start_time, deadline,
//! retry_policy, cancellation_token, recovery_strategy`.

use crate::cancellation::CancellationToken;
use crate::error::{Error, Result};
use crate::hash::FileHash;
use crate::id::{OpId, SessionId};
use crate::retry::RetryPolicy;
use crate::time::Deadline;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpState {
    Pending,
    Running,
    Done,
    Failed,
    Cancelled,
}

/// What to do with an unfinished operation after a crash.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "strategy", content = "detail", rename_all = "snake_case")]
pub enum RecoveryStrategy {
    /// Deterministic FS op: verify the file now hashes to `expected`;
    /// if so, mark the op complete; if not, the op truly never ran.
    VerifyHash { path: String, expected: FileHash },
    /// Command with unknown external effects: record `effect_status = unknown`
    /// and force verification instead of re-running.
    MarkUnknown,
    /// Safe to re-run (reads, idempotent calls).
    Idempotent,
    /// Never re-run automatically; require a human.
    Manual,
    /// No recovery action.
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffectStatus {
    Unknown,
    Verified,
    Applied,
    Failed,
}

/// Identity of one PHYSICAL attempt of a logical model call (attempt
/// accounting, phase-1 audit items D/E/F): every physical network attempt
/// must carry a fresh durable global [`OpId`] instead of reusing the shared
/// turn/model-call op, so reservations and provider-call rows can key by
/// exactly which wire attempt caused them. `logical_op_id` is the shared
/// logical op all attempts of one call belong to (today the turn/model-call
/// op stored in `provider_call.op_id` / `cost_reservation.op_id`);
/// `attempt_op_id` is THIS attempt's own fresh id — never equal to the
/// logical id (an attempt that reuses its parent's op id is the exact audit
/// hole this type exists to close); `ordinal` is the 0-based physical-attempt
/// ordinal within the logical call (0 = the original wire call, 1 = the
/// first retry/replay, ...).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ModelCallAttempt {
    /// The shared logical model-call op id (parent of every attempt).
    pub logical_op_id: OpId,
    /// The fresh durable global op id of THIS physical attempt. Must never
    /// equal `logical_op_id` — validated loudly by the writers.
    pub attempt_op_id: OpId,
    /// 0-based physical-attempt ordinal inside the logical call.
    pub ordinal: u32,
}

impl ModelCallAttempt {
    /// Build an attempt identity. `attempt_op_id` must differ from
    /// `logical_op_id`: a physical attempt that shares its parent's op id
    /// would collide with every other attempt of the same call in the
    /// attempt-keyed ledger rows. `None` when the caller passes the same id
    /// twice (validated, never silently accepted).
    pub fn new(logical_op_id: OpId, attempt_op_id: OpId, ordinal: u32) -> Option<Self> {
        if logical_op_id == attempt_op_id {
            return None;
        }
        Some(Self {
            logical_op_id,
            attempt_op_id,
            ordinal,
        })
    }
}

/// The full metadata envelope every async operation must carry.
#[derive(Debug, Clone)]
pub struct OpMeta {
    pub operation_id: OpId,
    pub session_id: SessionId,
    pub state: OpState,
    pub start_time_ms: i64,
    pub deadline: Deadline,
    pub retry_policy: RetryPolicy,
    pub cancellation: CancellationToken,
    pub recovery: RecoveryStrategy,
    /// Recovery descriptor for idempotent tool runs (JSON, validated by the
    /// agent runtime): the stored invocation that crash recovery replays
    /// ONCE as a new physical attempt of this same logical operation.
    /// `None` for every non-replayable operation.
    pub replay: Option<serde_json::Value>,
}

impl OpMeta {
    pub fn new(
        operation_id: OpId,
        session_id: SessionId,
        deadline: Deadline,
        retry_policy: RetryPolicy,
        cancellation: CancellationToken,
        recovery: RecoveryStrategy,
        now_ms: i64,
    ) -> Self {
        Self {
            operation_id,
            session_id,
            state: OpState::Pending,
            start_time_ms: now_ms,
            deadline,
            retry_policy,
            cancellation,
            recovery,
            replay: None,
        }
    }

    /// Attach the durable replay descriptor (idempotent tools only).
    pub fn with_replay(mut self, replay: serde_json::Value) -> Self {
        self.replay = Some(replay);
        self
    }

    /// Fail fast if the deadline has passed or cancellation was requested.
    pub fn ensure_alive(&self, now_ms: i64) -> Result<()> {
        if self.cancellation.is_cancelled() {
            return Err(Error::cancelled());
        }
        if self.deadline.is_expired(now_ms) {
            return Err(Error::timeout(format!(
                "operation {} deadline exceeded at {}",
                self.operation_id, now_ms
            )));
        }
        Ok(())
    }

    /// True if this op may be retried at all.
    pub fn retryable(&self) -> bool {
        self.retry_policy.max_attempts > 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::{Clock, SystemClock, TestClock};

    #[test]
    fn deadline_and_cancel_are_enforced_before_any_work() {
        let clock = TestClock::new(1000);
        let token = CancellationToken::new();
        let mut meta = OpMeta::new(
            OpId::new(1),
            SessionId::new(2),
            Deadline::at(clock.now_ms() + 500),
            RetryPolicy::default(),
            token.clone(),
            RecoveryStrategy::None,
            clock.now_ms(),
        );
        // healthy
        meta.ensure_alive(clock.now_ms()).unwrap();
        // deadline passes
        clock.advance(600);
        let err = meta.ensure_alive(clock.now_ms()).unwrap_err();
        assert!(err.kind == crate::ErrorKind::Timeout);
        // cancellation wins even before deadline
        let token = CancellationToken::new();
        meta.cancellation = token.clone();
        token.cancel();
        let err = meta.ensure_alive(clock.now_ms()).unwrap_err();
        assert!(err.kind == crate::ErrorKind::Cancelled);
        // fresh op again
        clock.advance(-600);
        let token = CancellationToken::new();
        let meta = OpMeta::new(
            OpId::new(1),
            SessionId::new(2),
            Deadline::at(clock.now_ms() + 500),
            RetryPolicy::default(),
            token,
            RecoveryStrategy::None,
            clock.now_ms(),
        );
        meta.ensure_alive(clock.now_ms()).unwrap();
    }

    #[test]
    fn metadata_serde_and_recovery_tagging() {
        let v = serde_json::to_value(RecoveryStrategy::VerifyHash {
            path: "/x".into(),
            expected: FileHash::from([3; 32]),
        })
        .unwrap();
        assert_eq!(v["strategy"], "verify_hash");
        let back: RecoveryStrategy = serde_json::from_value(v).unwrap();
        assert!(matches!(back, RecoveryStrategy::VerifyHash { .. }));

        // Unknown variant must be rejected, not silently defaulted.
        let bad = serde_json::json!({"strategy": "delete_everything"});
        assert!(serde_json::from_value::<RecoveryStrategy>(bad).is_err());
    }

    #[test]
    fn system_clock_is_monotonic_enough() {
        let c = SystemClock;
        let a = c.now_ms();
        let b = c.now_ms();
        assert!(b >= a);
    }

    #[test]
    fn attempt_identity_requires_a_fresh_attempt_op_id() {
        // Distinct logical/attempt ids are the point of the type: two
        // physical attempts of the same logical op carry two distinct
        // attempt ids and distinct ordinals, and the logical parent is
        // recoverable from either.
        let a = ModelCallAttempt::new(OpId::new(1), OpId::new(2), 0).unwrap();
        let b = ModelCallAttempt::new(OpId::new(1), OpId::new(3), 1).unwrap();
        assert_eq!(a.logical_op_id, b.logical_op_id);
        assert_ne!(
            a.attempt_op_id, b.attempt_op_id,
            "attempts never share an id"
        );
        assert_eq!(a.ordinal, 0);
        assert_eq!(b.ordinal, 1);
        // An attempt that reuses its parent's op id is rejected by the
        // constructor (never silently accepted, never serialized).
        assert!(ModelCallAttempt::new(OpId::new(1), OpId::new(1), 0).is_none());
        // An attempt with the SAME id as a DIFFERENT logical op's attempt is
        // legal at this level — the durable global allocator guarantees
        // global uniqueness, and both rows keep distinct logical parents.
        let c = ModelCallAttempt::new(OpId::new(5), OpId::new(2), 0).unwrap();
        assert_eq!(c.attempt_op_id, a.attempt_op_id);
    }

    #[test]
    fn attempt_identity_serde_roundtrips_and_rejects_collisions() {
        let a = ModelCallAttempt::new(OpId::new(10), OpId::new(11), 2).unwrap();
        let json = serde_json::to_string(&a).unwrap();
        let back: ModelCallAttempt = serde_json::from_str(&json).unwrap();
        assert_eq!(back, a);
        assert_eq!(back.ordinal, 2);
        // A hostile payload naming the same id twice must parse as a value
        // the writer validation refuses, never as two different things.
        let collision = serde_json::json!({
            "logical_op_id": 7,
            "attempt_op_id": 7,
            "ordinal": 0
        });
        let parsed: ModelCallAttempt = serde_json::from_value(collision).unwrap();
        assert_eq!(parsed.logical_op_id, parsed.attempt_op_id);
        assert!(ModelCallAttempt::new(parsed.logical_op_id, parsed.attempt_op_id, 0).is_none());
        // Zero op ids are rejected by the id contract itself.
        let zero = serde_json::json!({
            "logical_op_id": 0,
            "attempt_op_id": 1,
            "ordinal": 0
        });
        assert!(serde_json::from_value::<ModelCallAttempt>(zero).is_err());
    }
}
