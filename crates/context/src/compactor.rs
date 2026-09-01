//! Compaction that cannot enter a death spiral (spec §9).
//!
//! Hard invariant: a successful compaction must achieve the configured
//! minimum reduction. An LLM "summary" that would leave the context above
//! the target (or shave only ~1%) is REJECTED and deterministic pruning
//! takes over. Incremental by design: the task ledger is preserved whole,
//! only old recent turns are archived.

use std::sync::Arc;

use crate::artifact::ArtifactRef;
use crate::assembler::RecentTurn;
use crate::estimator::Estimator;
use crate::ledger::TaskLedger;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompactionRequest {
    pub before_tokens: usize,
    pub target_tokens: usize,
    /// Minimum fraction the context must shrink by (0.25 = 25%).
    pub min_reduction_ratio: f64,
}

impl Default for CompactionRequest {
    fn default() -> Self {
        Self {
            before_tokens: 0,
            target_tokens: 0,
            min_reduction_ratio: 0.25,
        }
    }
}

impl CompactionRequest {
    pub fn new(before_tokens: usize, target_tokens: usize) -> Self {
        Self {
            before_tokens,
            target_tokens,
            min_reduction_ratio: 0.25,
        }
    }

    /// The maximum acceptable after_tokens: min(target, before*(1-ratio)).
    pub fn hard_cap(&self) -> usize {
        let by_ratio = if self.min_reduction_ratio <= 0.0 {
            self.before_tokens
        } else {
            let factor = (1.0 - self.min_reduction_ratio).clamp(0.0, 1.0);
            (self.before_tokens as f64 * factor) as usize
        };
        self.target_tokens.min(by_ratio)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionStrategy {
    LlmSummary,
    DeterministicPruning,
    Rejected,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompactionPlan {
    pub accepted: bool,
    pub before_tokens: usize,
    pub after_tokens: usize,
    pub target_tokens: usize,
    pub strategy: CompactionStrategy,
    pub ledger: TaskLedger,
    pub kept_recent: Vec<RecentTurn>,
    pub archived: Vec<ArtifactRef>,
}

/// Produces the LLM-written summary (injected from the agent; None in
/// deterministic-only operation).
pub trait Summarizer: Send + Sync {
    fn summarize(&self, history: &[RecentTurn], ledger: &TaskLedger) -> String;
}

pub struct Compactor {
    summarizer: Option<Arc<dyn Summarizer>>,
}

impl Compactor {
    pub fn new(summarizer: Option<Arc<dyn Summarizer>>) -> Self {
        Self { summarizer }
    }

    pub fn deterministic_only() -> Self {
        Self::new(None)
    }

    /// Compaction pipeline:
    /// 1. If a summarizer exists, try the LLM summary; accept it ONLY if it
    ///    satisfies the hard invariant (after <= hard_cap).
    /// 2. Otherwise (or when rejected), deterministic pruning.
    /// 3. If even deterministic pruning cannot reach the cap (pathological),
    ///    the plan is marked rejected with strategy Rejected — callers must
    ///    surface it (CompactRejected) instead of pretending success.
    pub fn compact(
        &self,
        history: &[RecentTurn],
        ledger: &TaskLedger,
        req: &CompactionRequest,
    ) -> CompactionPlan {
        if let Some(summarizer) = &self.summarizer {
            let summary = summarizer.summarize(history, ledger);
            let after = Estimator.estimate_tokens(&summary);
            if after <= req.hard_cap() {
                return CompactionPlan {
                    accepted: true,
                    before_tokens: req.before_tokens,
                    after_tokens: after,
                    target_tokens: req.target_tokens,
                    strategy: CompactionStrategy::LlmSummary,
                    ledger: ledger.clone(),
                    kept_recent: history.to_vec(),
                    archived: vec![],
                };
            }
            // REJECT the liar summary; fall through to deterministic.
            let mut plan = self.deterministic_prune(history, ledger, req);
            plan.strategy = CompactionStrategy::Rejected;
            plan
        } else {
            self.deterministic_prune(history, ledger, req)
        }
    }

    /// Deterministic pruning: keep the ledger in full, keep the newest
    /// recent turns that fit under the cap, archive the rest as references.
    pub fn deterministic_prune(
        &self,
        history: &[RecentTurn],
        ledger: &TaskLedger,
        req: &CompactionRequest,
    ) -> CompactionPlan {
        let ledger_render = ledger.compact_render();
        let ledger_tokens = Estimator.estimate_tokens(&ledger_render);
        let cap = req.hard_cap();
        let mut kept_recent = Vec::new();
        let mut archived = Vec::new();
        let mut used = ledger_tokens.saturating_add(32); // headers/overhead
        if cap > ledger_tokens {
            // Newest first; each turn's cost = role + text.
            for turn in history.iter().rev() {
                let t = Estimator.estimate_tokens(&format!("{}: {}", turn.role, turn.text));
                if used + t > cap {
                    archived.push(ArtifactRef {
                        inline: None,
                        artifact: None,
                        summary: format!("archived turn ({} chars)", turn.text.len()),
                        size: turn.text.len(),
                    });
                    continue;
                }
                used += t;
                kept_recent.push(turn.clone());
            }
            kept_recent.reverse();
        }
        let after = used;
        let accepted = after <= cap;
        CompactionPlan {
            accepted,
            before_tokens: req.before_tokens,
            after_tokens: after,
            target_tokens: req.target_tokens,
            strategy: if accepted {
                CompactionStrategy::DeterministicPruning
            } else {
                CompactionStrategy::Rejected
            },
            ledger: ledger.clone(),
            kept_recent,
            archived,
        }
    }

    /// Would a new compaction still be required immediately after `plan`?
    /// The death-spiral guard: if true after an *accepted* plan, the engine
    /// must not loop forever — this must converge to false within bounded
    /// steps because deterministic pruning is monotonically non-increasing.
    pub fn would_compact_again(&self, plan: &CompactionPlan, req: &CompactionRequest) -> bool {
        if !plan.accepted {
            return true;
        }
        let new_req = CompactionRequest {
            before_tokens: plan.after_tokens,
            target_tokens: req.target_tokens,
            min_reduction_ratio: req.min_reduction_ratio,
        };
        // If before <= target there is nothing left to do.
        plan.after_tokens
            > new_req
                .hard_cap()
                .max(new_req.target_tokens.min(plan.after_tokens))
            && plan.after_tokens > req.target_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn history(n: usize) -> Vec<RecentTurn> {
        (0..n)
            .map(|i| RecentTurn {
                role: "assistant".into(),
                text: format!("turn {i}: {}", "z".repeat(400)),
            })
            .collect()
    }

    fn ledger() -> TaskLedger {
        TaskLedger {
            goal: "g".into(),
            open_steps: vec!["s".into()],
            ..Default::default()
        }
    }

    /// The adversary: a "summarizer" that returns the whole history verbatim
    /// (reduces context by ~0%).
    struct LiarSummarizer;
    impl Summarizer for LiarSummarizer {
        fn summarize(&self, history: &[RecentTurn], _ledger: &TaskLedger) -> String {
            history
                .iter()
                .map(|t| format!("{}: {}", t.role, t.text))
                .collect::<Vec<_>>()
                .join("\n")
        }
    }

    #[test]
    fn one_percent_summary_rejected_and_deterministic_fallback() {
        let history = history(200);
        let e = Estimator;
        let before = e.estimate_tokens(
            &history
                .iter()
                .map(|t| t.text.clone())
                .collect::<Vec<_>>()
                .join("\n"),
        );
        let req = CompactionRequest::new(before, before / 2);
        let compactor = Compactor::new(Some(Arc::new(LiarSummarizer)));
        let plan = compactor.compact(&history, &ledger(), &req);
        assert!(
            !matches!(plan.strategy, CompactionStrategy::LlmSummary),
            "liar summary must never be accepted"
        );
        assert!(plan.accepted, "deterministic fallback must succeed");
        assert_eq!(
            plan.strategy,
            CompactionStrategy::Rejected,
            "the summary attempt was rejected, then pruned"
        );
        assert!(plan.after_tokens <= req.hard_cap(), "hard invariant");
        // ~1% reduction never accepted: verify explicitly.
        let one_pct = CompactionRequest {
            before_tokens: 180_000,
            target_tokens: 178_200, // 1% reduction
            min_reduction_ratio: 0.25,
        };
        let cap = one_pct.hard_cap();
        assert!(cap <= 135_000, "cap enforces the 25% floor, not the 1% ask");
    }

    #[test]
    fn good_summary_accepted() {
        struct GoodSummarizer;
        impl Summarizer for GoodSummarizer {
            fn summarize(&self, _h: &[RecentTurn], ledger: &TaskLedger) -> String {
                format!("SUMMARY: {}", ledger.compact_render())
            }
        }
        let history = history(200);
        let before = 100_000;
        let req = CompactionRequest::new(before, 30_000);
        let compactor = Compactor::new(Some(Arc::new(GoodSummarizer)));
        let plan = compactor.compact(&history, &ledger(), &req);
        assert!(plan.accepted);
        assert_eq!(plan.strategy, CompactionStrategy::LlmSummary);
        assert!(plan.after_tokens <= req.hard_cap());
    }

    #[test]
    fn death_spiral_converges_with_liar_summarizer() {
        // The classic failure: compaction keeps "succeeding" by tiny margins
        // and never reaches the target. Our invariant must force the
        // deterministic path, and repeated compactions must converge.
        let mut current = history(400);
        let compactor = Compactor::new(Some(Arc::new(LiarSummarizer)));
        let target = 40_000usize;
        let mut steps = 0;
        let e = Estimator;
        let mut before_tokens = e.estimate_tokens(
            &current
                .iter()
                .map(|t| t.text.clone())
                .collect::<Vec<_>>()
                .join("\n"),
        );
        let mut converged = false;
        while steps < 20 {
            let req = CompactionRequest::new(before_tokens, target);
            let plan = compactor.compact(&current, &ledger(), &req);
            assert!(plan.accepted, "deterministic path must accept");
            assert!(
                plan.after_tokens <= req.hard_cap(),
                "step {steps}: after {} > cap {}",
                plan.after_tokens,
                req.hard_cap()
            );
            if compactor.would_compact_again(&plan, &req) {
                // Simulate the next context built from the plan.
                current = plan.kept_recent.clone();
                before_tokens = plan.after_tokens;
                steps += 1;
                continue;
            }
            converged = true;
            break;
        }
        assert!(converged, "did not converge within 20 steps");
    }

    #[test]
    fn none_summarizer_incremental_compaction_preserves_ledger() {
        let history = history(300);
        let before = 200_000;
        let req = CompactionRequest::new(before, 60_000);
        let compactor = Compactor::deterministic_only();
        let plan = compactor.compact(&history, &ledger(), &req);
        assert!(plan.accepted);
        assert_eq!(plan.strategy, CompactionStrategy::DeterministicPruning);
        assert_eq!(plan.ledger, ledger(), "ledger preserved in full");
        assert!(!plan.kept_recent.is_empty(), "newest turns kept");
        assert!(plan.after_tokens <= req.hard_cap());
    }

    #[test]
    fn archived_artifacts_track_evicted_turns() {
        let history = history(100);
        let req = CompactionRequest::new(200_000, 10_000);
        let compactor = Compactor::deterministic_only();
        let plan = compactor.compact(&history, &ledger(), &req);
        assert!(!plan.archived.is_empty(), "most turns archived");
        // Every evicted turn is accounted for: kept + archived = total.
        assert_eq!(plan.kept_recent.len() + plan.archived.len(), history.len());
        // Newest turns survive, oldest archived.
        assert_eq!(
            plan.kept_recent.last().map(|t| &t.text),
            history.last().map(|t| &t.text)
        );
    }

    #[test]
    fn tiny_history_fits_without_archiving() {
        let history = history(2);
        let req = CompactionRequest::new(10_000, 8_000);
        let compactor = Compactor::deterministic_only();
        let plan = compactor.compact(&history, &ledger(), &req);
        assert!(plan.accepted);
        assert!(plan.archived.is_empty());
        assert_eq!(plan.kept_recent.len(), 2);
    }

    #[test]
    fn zero_reduction_never_accepted_even_at_target() {
        // before == target: any "compaction" is a zero reduction → rejected.
        let history = history(10);
        let before = 5_000;
        let req = CompactionRequest::new(before, before);
        let compactor = Compactor::deterministic_only();
        let plan = compactor.compact(&history, &ledger(), &req);
        // hard_cap = before * 0.75 < before, so after (>= ledger) may or may
        // not fit; what must NEVER happen is `accepted` with after == before.
        if plan.accepted {
            assert!(
                plan.after_tokens < plan.before_tokens,
                "zero-reduction compaction accepted!"
            );
        }
    }

    #[test]
    fn would_compact_again_boundaries() {
        let compactor = Compactor::deterministic_only();
        let plan = CompactionPlan {
            accepted: true,
            before_tokens: 100_000,
            after_tokens: 80_000,
            target_tokens: 80_000,
            strategy: CompactionStrategy::DeterministicPruning,
            ledger: ledger(),
            kept_recent: vec![],
            archived: vec![],
        };
        let req = CompactionRequest::new(100_000, 80_000);
        // After == target: done.
        assert!(!compactor.would_compact_again(&plan, &req));
        // Rejected plans always need attention.
        let rejected = CompactionPlan {
            accepted: false,
            ..plan.clone()
        };
        assert!(compactor.would_compact_again(&rejected, &req));
        // Above target: must compact again.
        let above = CompactionPlan {
            after_tokens: 90_000,
            ..plan
        };
        let req = CompactionRequest::new(100_000, 70_000);
        assert!(compactor.would_compact_again(&above, &req));
    }

    #[test]
    fn hostile_ratio_values_are_clamped() {
        let req = CompactionRequest {
            before_tokens: 100,
            target_tokens: 10,
            min_reduction_ratio: 5.0, // hostile
        };
        let cap = req.hard_cap();
        assert!(cap <= 10);
        let req = CompactionRequest {
            before_tokens: 100,
            target_tokens: 10,
            min_reduction_ratio: -1.0, // hostile
        };
        assert!(req.hard_cap() <= 10);
    }
}
