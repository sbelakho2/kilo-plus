//! Compaction records with the hard invariant from section 9 of the
//! architecture spec: a successful compaction must achieve the configured
//! minimum reduction. A "summary" that reduces context by ~1% is rejected.

use faktor_core::event::EventKind;

use crate::handle::SessionHandle;
use crate::SessionError;

/// Compaction acceptance policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompactionPolicy {
    /// Minimum reduction ratio (1 - after/before) a compaction must achieve
    /// to be accepted. Must be in `[0.0, 1.0)`.
    pub min_reduction_ratio: f64,
}

impl Default for CompactionPolicy {
    fn default() -> Self {
        Self {
            min_reduction_ratio: 0.25,
        }
    }
}

/// The outcome of a compaction attempt.
#[derive(Debug, Clone, PartialEq)]
pub struct CompactionRecord {
    pub accepted: bool,
    pub before: i64,
    pub after: i64,
    pub target: i64,
    pub reduction_ratio: f64,
    /// Present when rejected: why.
    pub reason: Option<String>,
}

impl SessionHandle {
    /// Record a compaction attempt and enforce the hard invariant. Accepted
    /// iff `after <= target` and `reduction >= min_reduction_ratio`; anything
    /// else journals `CompactRejected` and reports the reason. The session
    /// state does not move (compaction is an interior event).
    pub fn record_compaction(
        &self,
        before: i64,
        after: i64,
        target: i64,
        strategy: &str,
        policy: &CompactionPolicy,
    ) -> faktor_core::Result<CompactionRecord> {
        if before <= 0 {
            return Err(SessionError::Malformed(format!(
                "compaction `before` must be > 0, got {before}"
            ))
            .into());
        }
        if after < 0 || target < 0 {
            return Err(SessionError::Malformed(format!(
                "compaction sizes must be >= 0 (before={before}, after={after}, target={target})"
            ))
            .into());
        }
        if after > before {
            return Err(SessionError::Malformed(format!(
                "compaction cannot grow the context: after={after} > before={before}"
            ))
            .into());
        }
        if !(0.0..1.0).contains(&policy.min_reduction_ratio) {
            return Err(SessionError::Malformed(format!(
                "min_reduction_ratio must be in [0, 1), got {}",
                policy.min_reduction_ratio
            ))
            .into());
        }
        if strategy.is_empty() || strategy.len() > 128 {
            return Err(SessionError::Malformed("invalid compaction strategy".into()).into());
        }

        let reduction_ratio = 1.0 - (after as f64 / before as f64);
        let accepted = after <= target && reduction_ratio >= policy.min_reduction_ratio;
        let reason = if !accepted {
            let why = if after > target {
                format!("after ({after}) exceeds target ({target})")
            } else {
                format!(
                    "reduction ({:.3}) below minimum ({:.3})",
                    reduction_ratio, policy.min_reduction_ratio
                )
            };
            Some(why)
        } else {
            None
        };

        let _guard = self.command_guard();
        self.manager
            .store()
            .record_compaction(self.id, before, after, target, accepted, strategy)
            .map_err(crate::map_store_err)?;
        let kind = if accepted {
            EventKind::ContextCompacted
        } else {
            EventKind::CompactRejected
        };
        let current = self.state()?;
        self.transition_locked(
            kind,
            current,
            None,
            Some(serde_json::json!({
                "before": before,
                "after": after,
                "target": target,
                "accepted": accepted,
                "strategy": strategy,
                "reduction": reduction_ratio,
            })),
        )?;
        Ok(CompactionRecord {
            accepted,
            before,
            after,
            target,
            reduction_ratio,
            reason,
        })
    }

    /// Record a compaction with the default policy.
    pub fn record_compaction_defaults(
        &self,
        before: i64,
        after: i64,
        target: i64,
        strategy: &str,
    ) -> faktor_core::Result<CompactionRecord> {
        self.record_compaction(
            before,
            after,
            target,
            strategy,
            &CompactionPolicy::default(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::tests::{session, test_manager};
    use faktor_core::event::EventKind;

    #[test]
    fn compaction_rejects_one_percent_reduction() {
        // The archetypal death-spiral case: 180k -> 178k is "done" by the
        // summarizer but saves ~1%. Reject.
        let (_d, m) = test_manager();
        let s = session(&m);
        let r = s
            .record_compaction_defaults(100_000, 99_000, 99_000, "summarize")
            .unwrap();
        assert!(!r.accepted);
        assert!(r.reason.as_ref().unwrap().contains("minimum"));
        let kinds: Vec<_> = s
            .events_range(1, None)
            .unwrap()
            .into_iter()
            .map(|e| e.kind)
            .collect();
        assert!(kinds.contains(&EventKind::CompactRejected));
        assert!(!kinds.contains(&EventKind::ContextCompacted));
    }

    #[test]
    fn compaction_accepts_meaningful_reduction() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let r = s
            .record_compaction_defaults(100_000, 60_000, 90_000, "deterministic_prune")
            .unwrap();
        assert!(r.accepted);
        assert!(r.reason.is_none());
        assert!((r.reduction_ratio - 0.4).abs() < 1e-9);
        let kinds: Vec<_> = s
            .events_range(1, None)
            .unwrap()
            .into_iter()
            .map(|e| e.kind)
            .collect();
        assert!(kinds.contains(&EventKind::ContextCompacted));
    }

    #[test]
    fn compaction_after_exceeding_target_is_rejected() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let r = s
            .record_compaction_defaults(100_000, 95_000, 90_000, "summarize")
            .unwrap();
        assert!(!r.accepted);
        assert!(r.reason.as_ref().unwrap().contains("target"));
        // A summary that *grows* context is malformed, not just rejected.
        assert!(s
            .record_compaction_defaults(100_000, 120_000, 90_000, "x")
            .is_err());
        assert!(s.record_compaction_defaults(0, 0, 0, "x").is_err());
        assert!(s.record_compaction_defaults(-5, 0, 0, "x").is_err());
    }

    #[test]
    fn compaction_hard_invariant_rejects_zero_reduction_even_at_target() {
        // before == target, after == before: nothing to compact.
        let (_d, m) = test_manager();
        let s = session(&m);
        let r = s
            .record_compaction_defaults(90_000, 90_000, 90_000, "x")
            .unwrap();
        assert!(!r.accepted);
        assert!((r.reduction_ratio).abs() < 1e-12);
    }

    #[test]
    fn compaction_policy_is_configurable_and_validated() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let lax = CompactionPolicy {
            min_reduction_ratio: 0.01,
        };
        // With a 1% floor, a 5% reduction is accepted.
        let r = s
            .record_compaction(100_000, 95_000, 95_000, "x", &lax)
            .unwrap();
        assert!(r.accepted);
        // An insane floor is malformed.
        assert!(s
            .record_compaction(
                100_000,
                95_000,
                95_000,
                "x",
                &CompactionPolicy {
                    min_reduction_ratio: 1.5
                }
            )
            .is_err());
        assert!(s
            .record_compaction(
                100_000,
                95_000,
                95_000,
                "x",
                &CompactionPolicy {
                    min_reduction_ratio: -0.1
                }
            )
            .is_err());
    }

    #[test]
    fn compaction_events_are_interior_and_state_preserving() {
        let (_d, m) = test_manager();
        let s = session(&m);
        s.submit_prompt("x", &[]).unwrap();
        s.record_compaction_defaults(100_000, 50_000, 80_000, "prune")
            .unwrap();
        assert_eq!(
            s.state().unwrap(),
            faktor_core::state::AgentState::Preparing
        );
        assert_eq!(
            s.replay_journal().unwrap().state,
            faktor_core::state::AgentState::Preparing
        );
    }
}
