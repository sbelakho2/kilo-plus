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
        }
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
}
