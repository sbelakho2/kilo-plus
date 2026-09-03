//! Compaction that cannot enter a death spiral (spec §9).
//!
//! Hard invariant: a successful compaction must achieve the configured
//! minimum reduction. An LLM "summary" that would leave the context above
//! the target (or shave only ~1%) is REJECTED and deterministic pruning
//! takes over. Incremental by design: the task ledger is preserved whole,
//! only old recent turns are archived.

use std::sync::Arc;

use crate::artifact::ArtifactRef;

/// Bound on ONE chunk of the rendered text of evicted turns carried out of
/// the compactor for the content store. There is NO total archive cap: every
/// evicted turn is preserved across as many chunks as needed, and the runtime
/// stores each chunk in the CAS behind a JSON manifest (the recent history in
/// RAM is itself bounded, so the chunked archive never exceeds it).
const ARCHIVE_CHUNK_BYTES: usize = 512 * 1024;
const DIGEST_MAX_CHARS: usize = 600;
const DIGEST_MAX_LINES: usize = 8;
const DIGEST_TEMPLATE: &str = "[Earlier context archived: …<artifact://hash>]";
const ARTIFACT_PLACEHOLDER: &str = "<artifact://hash>";

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        &s[..max]
    }
}

/// Chunk the rendered text of evicted turns (`"{role}: {text}\n"` per turn,
/// OLDEST-first iteration) into `Vec<String>` chunks each <= `chunk_max`
/// bytes. Chunks preserve order and NEVER split mid-turn: a single turn
/// larger than the bound occupies its own whole chunk (the recent history in
/// RAM is bounded, so an oversize chunk is bounded too). Nothing is dropped.
fn fill_chunks<'a>(evicted: impl Iterator<Item = &'a RecentTurn>, chunk_max: usize) -> Vec<String> {
    let mut chunks: Vec<String> = Vec::new();
    for turn in evicted {
        let line = format!("{}: {}", turn.role, turn.text);
        if let Some(last) = chunks.last_mut() {
            if !last.is_empty() && last.len() + line.len() < chunk_max {
                last.push_str(&line);
                last.push('\n');
                continue;
            }
        }
        let mut chunk = String::new();
        chunk.push_str(&line);
        chunk.push('\n');
        chunks.push(chunk);
    }
    chunks
}

/// The eviction digest that rides the wire inside kept_recent[0]: the NEWEST
/// evicted material (the tail of the archive — chunks are ordered oldest
/// first), bounded to a few lines and `DIGEST_MAX_CHARS`. A single line
/// larger than the whole budget is truncated to fit, never dropped entirely,
/// so the marker always carries a glimpse of what was archived. Empty only
/// when nothing was archived.
fn archive_digest(chunks: &[String]) -> String {
    let mut newest_first: Vec<&str> = Vec::new();
    for chunk in chunks.iter().rev() {
        for line in chunk.lines().rev() {
            newest_first.push(line);
            if newest_first.len() >= DIGEST_MAX_LINES {
                break;
            }
        }
        if newest_first.len() >= DIGEST_MAX_LINES {
            break;
        }
    }
    // Render chronologically (oldest of the sampled lines first).
    let mut digest_text = String::new();
    for line in newest_first.into_iter().rev() {
        let remaining = DIGEST_MAX_CHARS.saturating_sub(digest_text.len());
        if remaining == 0 {
            break;
        }
        if line.len() > remaining {
            digest_text.push_str(truncate(line, remaining));
            digest_text.push('\n');
            break;
        }
        digest_text.push_str(line);
        digest_text.push('\n');
    }
    digest_text
}
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
    /// The rendered text of every turn evicted from the wire, chunked
    /// (`"{role}: {text}\n"` per turn, oldest first, each chunk <=
    /// `ARCHIVE_CHUNK_BYTES`, never split mid-turn, NO total cap — nothing
    /// evicted is ever omitted). This is the durable archive material for
    /// the content store: the runtime writes each chunk to the CAS, then a
    /// JSON manifest, and replaces the digest placeholder with the manifest
    /// artifact ref.
    pub archive_chunks: Vec<String>,
}

/// Produces the LLM-written summary (injected from the agent; None in
/// deterministic-only operation). Async: real summarizers stream a
/// provider request; the deterministic ledger summarizer resolves
/// immediately.
pub trait Summarizer: Send + Sync {
    fn summarize<'a>(
        &'a self,
        history: &'a [RecentTurn],
        ledger: &'a TaskLedger,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = String> + Send + 'a>>;
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
    pub async fn compact(
        &self,
        history: &[RecentTurn],
        ledger: &TaskLedger,
        req: &CompactionRequest,
    ) -> CompactionPlan {
        if let Some(summarizer) = &self.summarizer {
            let summary = summarizer.summarize(history, ledger).await;
            let after = Estimator.estimate_tokens(&summary);
            if after <= req.hard_cap() {
                // Accepted summary: it REPLACES the history on the wire (a
                // summary branch that kept the full history verbatim shrank
                // the bookkeeping numbers but not the actual context — the
                // audit-round fix). EVERY evicted turn is archived for the
                // CAS, chunked — never truncated at 1 MiB.
                let archived = history
                    .iter()
                    .map(|t| ArtifactRef {
                        inline: None,
                        artifact: None,
                        summary: format!("archived turn ({} chars)", t.text.len()),
                        size: t.text.len(),
                    })
                    .collect();
                return CompactionPlan {
                    accepted: true,
                    before_tokens: req.before_tokens,
                    after_tokens: after,
                    target_tokens: req.target_tokens,
                    strategy: CompactionStrategy::LlmSummary,
                    ledger: ledger.clone(),
                    kept_recent: vec![RecentTurn {
                        role: "assistant".into(),
                        text: summary,
                    }],
                    archived,
                    archive_chunks: fill_chunks(history.iter(), ARCHIVE_CHUNK_BYTES),
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
        // The eviction digest rides the wire in kept_recent[0]; reserve its
        // budget up front so the after_tokens figure stays honest.
        let digest_tokens = Estimator.estimate_tokens(DIGEST_TEMPLATE);
        let mut kept_recent = Vec::new();
        let mut archived = Vec::new();
        // Evicted turns as collected (newest first — the scan runs newest to
        // oldest); reversed into chronological order for the chunk fill.
        let mut evicted: Vec<&RecentTurn> = Vec::new();
        let mut used = ledger_tokens
            .saturating_add(32)
            .saturating_add(digest_tokens);
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
                    evicted.push(turn);
                    continue;
                }
                used += t;
                kept_recent.push(turn.clone());
            }
            kept_recent.reverse();
        }
        // Every evicted turn is archived, chunked, oldest first — nothing
        // beyond the kept window is ever omitted (the old 1 MiB cap silently
        // dropped cold history).
        let archive_chunks = fill_chunks(evicted.iter().rev().copied(), ARCHIVE_CHUNK_BYTES);
        let after = used;
        let accepted = after <= cap;
        // Digest of what was archived: the newest evicted lines (tail of the
        // last chunk), bounded — the marker tells the model history was
        // dropped and where the durable material lives.
        let digest_text = archive_digest(&archive_chunks);
        if !digest_text.is_empty() && accepted {
            let digest = RecentTurn {
                role: "assistant".into(),
                text: format!(
                    "[Earlier context archived: {}…{}]",
                    truncate(&digest_text, DIGEST_MAX_CHARS),
                    ARTIFACT_PLACEHOLDER
                ),
            };
            kept_recent.insert(0, digest);
        }
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
            archive_chunks,
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
        fn summarize<'a>(
            &'a self,
            history: &'a [RecentTurn],
            _ledger: &'a TaskLedger,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = String> + Send + 'a>> {
            Box::pin(async move {
                history
                    .iter()
                    .map(|t| format!("{}: {}", t.role, t.text))
                    .collect::<Vec<_>>()
                    .join("\n")
            })
        }
    }

    #[tokio::test]
    async fn one_percent_summary_rejected_and_deterministic_fallback() {
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
        let plan = compactor.compact(&history, &ledger(), &req).await;
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

    #[tokio::test]
    async fn good_summary_accepted() {
        struct GoodSummarizer;
        impl Summarizer for GoodSummarizer {
            fn summarize<'a>(
                &'a self,
                _h: &'a [RecentTurn],
                ledger: &'a TaskLedger,
            ) -> std::pin::Pin<Box<dyn std::future::Future<Output = String> + Send + 'a>>
            {
                Box::pin(async move { format!("SUMMARY: {}", ledger.compact_render()) })
            }
        }
        let history = history(200);
        let before = 100_000;
        let req = CompactionRequest::new(before, 30_000);
        let compactor = Compactor::new(Some(Arc::new(GoodSummarizer)));
        let plan = compactor.compact(&history, &ledger(), &req).await;
        assert!(plan.accepted);
        assert_eq!(plan.strategy, CompactionStrategy::LlmSummary);
        assert!(plan.after_tokens <= req.hard_cap());
    }

    #[tokio::test]
    async fn death_spiral_converges_with_liar_summarizer() {
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
            let plan = compactor.compact(&current, &ledger(), &req).await;
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

    #[tokio::test]
    async fn none_summarizer_incremental_compaction_preserves_ledger() {
        let history = history(300);
        let before = 200_000;
        let req = CompactionRequest::new(before, 60_000);
        let compactor = Compactor::deterministic_only();
        let plan = compactor.compact(&history, &ledger(), &req).await;
        assert!(plan.accepted);
        assert_eq!(plan.strategy, CompactionStrategy::DeterministicPruning);
        assert_eq!(plan.ledger, ledger(), "ledger preserved in full");
        assert!(!plan.kept_recent.is_empty(), "newest turns kept");
        assert!(plan.after_tokens <= req.hard_cap());
    }

    #[tokio::test]
    async fn archived_artifacts_track_evicted_turns() {
        let history = history(100);
        let req = CompactionRequest::new(200_000, 10_000);
        let compactor = Compactor::deterministic_only();
        let plan = compactor.compact(&history, &ledger(), &req).await;
        assert!(!plan.archived.is_empty(), "most turns archived");
        // Every evicted turn is accounted for: kept (minus the eviction
        // digest at index 0) + archived = total.
        let kept_excluding_digest = if plan
            .kept_recent
            .first()
            .is_some_and(|t| t.text.starts_with("[Earlier context archived:"))
        {
            plan.kept_recent.len() - 1
        } else {
            plan.kept_recent.len()
        };
        assert_eq!(
            kept_excluding_digest + plan.archived.len(),
            history.len(),
            "every evicted turn must be accounted for"
        );
        // Newest turns survive, oldest archived.
        assert_eq!(
            plan.kept_recent.last().map(|t| &t.text),
            history.last().map(|t| &t.text)
        );
        // The durable archive material is non-empty, chunked, and every
        // chunk respects the bound.
        assert!(!plan.archive_chunks.is_empty());
        assert!(plan
            .archive_chunks
            .iter()
            .all(|c| !c.is_empty() && c.len() <= ARCHIVE_CHUNK_BYTES));
        // The digest rides the wire so the model knows history was dropped.
        assert!(plan
            .kept_recent
            .first()
            .is_some_and(|t| t.text.contains("<artifact://hash>")));
    }

    #[tokio::test]
    async fn accepted_llm_summary_actually_replaces_history() {
        struct GoodSummarizer;
        impl Summarizer for GoodSummarizer {
            fn summarize<'a>(
                &'a self,
                _h: &'a [RecentTurn],
                ledger: &'a TaskLedger,
            ) -> std::pin::Pin<Box<dyn std::future::Future<Output = String> + Send + 'a>>
            {
                Box::pin(async move { format!("COMPACT SUMMARY: {}", ledger.compact_render()) })
            }
        }
        let history = history(50);
        let req = CompactionRequest::new(100_000, 5_000);
        let compactor = Compactor::new(Some(Arc::new(GoodSummarizer)));
        let plan = compactor.compact(&history, &ledger(), &req).await;
        assert!(plan.accepted);
        assert_eq!(plan.strategy, CompactionStrategy::LlmSummary);
        // The wire content after an accepted summary is the summary ALONE,
        // not the full history (the old code kept history verbatim and the
        // shrink was bookkeeping-only).
        assert_eq!(plan.kept_recent.len(), 1);
        assert!(plan.kept_recent[0].text.starts_with("COMPACT SUMMARY:"));
        // All evicted turns are accounted for as archives + durable text.
        assert_eq!(plan.archived.len(), history.len());
        let archive_len: usize = plan.archive_chunks.iter().map(String::len).sum();
        assert!(archive_len >= 10_000, "archive holds real text");
    }

    #[tokio::test]
    async fn tiny_history_fits_without_archiving() {
        let history = history(2);
        let req = CompactionRequest::new(10_000, 8_000);
        let compactor = Compactor::deterministic_only();
        let plan = compactor.compact(&history, &ledger(), &req).await;
        assert!(plan.accepted);
        assert!(plan.archived.is_empty());
        assert_eq!(plan.kept_recent.len(), 2);
    }

    /// Rendered evicted-turn text exactly as the compactor archives it.
    fn rendered(evicted: &[RecentTurn]) -> String {
        let mut out = String::new();
        for t in evicted {
            out.push_str(&format!("{}: {}\n", t.role, t.text));
        }
        out
    }

    #[tokio::test]
    async fn deterministic_evictions_over_1mib_archive_chunked_nothing_lost() {
        // P0: the old 1 MiB archive cap silently dropped cold history. A
        // ~2.5 MiB eviction must now come back as MULTIPLE ordered chunks
        // whose concatenation equals the evicted turns EXACTLY — oldest and
        // newest evicted text included, nothing truncated.
        let history = history(6000); // ≈ 2.5 MiB of turn text
        let before = Estimator.estimate_tokens(&rendered(&history));
        let req = CompactionRequest::new(before, before / 5);
        let compactor = Compactor::deterministic_only();
        let plan = compactor.compact(&history, &ledger(), &req).await;
        assert!(plan.accepted, "deterministic pruning must fit the cap");
        assert!(
            plan.archive_chunks.len() >= 2,
            "> 1 MiB of evicted text must produce multiple chunks, got {}",
            plan.archive_chunks.len()
        );
        // Every chunk respects the bound; no empty chunks.
        assert!(plan
            .archive_chunks
            .iter()
            .all(|c| !c.is_empty() && c.len() <= ARCHIVE_CHUNK_BYTES));
        // The kept turns are the newest suffix (minus the digest at index 0).
        let kept: Vec<&RecentTurn> = plan
            .kept_recent
            .iter()
            .filter(|t| !t.text.starts_with("[Earlier context archived:"))
            .collect();
        assert_eq!(
            kept,
            history[history.len() - kept.len()..]
                .iter()
                .collect::<Vec<_>>()
        );
        let evicted = &history[..history.len() - kept.len()];
        assert!(!evicted.is_empty());
        // Concatenation (oldest first) equals the evicted text EXACTLY.
        let concat: String = plan.archive_chunks.concat();
        assert_eq!(
            concat,
            rendered(evicted),
            "archived material must be lossless"
        );
        // The oldest AND newest evicted text both survive.
        assert!(concat.starts_with(&format!("{}: ", evicted[0].role)));
        assert_eq!(concat.len(), rendered(evicted).len());
        assert_eq!(
            concat[concat.len() - evicted.last().unwrap().text.len() - 1..].trim_end(),
            evicted.last().unwrap().text
        );
    }

    #[tokio::test]
    async fn accepted_summary_archives_whole_history_chunked() {
        // The LlmSummary branch archives EVERY evicted turn (the whole
        // history) — over 1 MiB it must chunk, never truncate.
        struct GoodSummarizer;
        impl Summarizer for GoodSummarizer {
            fn summarize<'a>(
                &'a self,
                _h: &'a [RecentTurn],
                ledger: &'a TaskLedger,
            ) -> std::pin::Pin<Box<dyn std::future::Future<Output = String> + Send + 'a>>
            {
                Box::pin(async move { format!("COMPACT SUMMARY: {}", ledger.compact_render()) })
            }
        }
        let history = history(6000); // ≈ 2.5 MiB of turn text
        let before = Estimator.estimate_tokens(&rendered(&history));
        let req = CompactionRequest::new(before, before / 5);
        let compactor = Compactor::new(Some(Arc::new(GoodSummarizer)));
        let plan = compactor.compact(&history, &ledger(), &req).await;
        assert!(plan.accepted);
        assert_eq!(plan.strategy, CompactionStrategy::LlmSummary);
        assert!(
            plan.archive_chunks.len() >= 2,
            "> 1 MiB of evicted text must produce multiple chunks, got {}",
            plan.archive_chunks.len()
        );
        assert!(plan
            .archive_chunks
            .iter()
            .all(|c| !c.is_empty() && c.len() <= ARCHIVE_CHUNK_BYTES));
        assert_eq!(
            plan.archive_chunks.concat(),
            rendered(&history),
            "the whole evicted history must be archived losslessly"
        );
    }

    #[test]
    fn fill_chunks_never_splits_mid_turn_and_preserves_order() {
        // Chunk boundaries fall ONLY between turns; a single turn larger
        // than the bound occupies its own WHOLE chunk (never a fragment);
        // oldest-first order is preserved end to end.
        let turn = |text: &str| RecentTurn {
            role: "assistant".into(),
            text: text.to_string(),
        };
        let turns = vec![turn("aa"), turn("bbbb"), turn("cc"), turn("dddddddddd")];
        // Bound 12 bytes: "assistant: aa\n" (13) already exceeds it.
        let chunks = fill_chunks(turns.iter(), 12);
        // Every turn whole in its own chunk: the rendered lines are larger
        // than the bound, so nothing may be joined.
        assert_eq!(
            chunks,
            vec![
                "assistant: aa\n".to_string(),
                "assistant: bbbb\n".to_string(),
                "assistant: cc\n".to_string(),
                "assistant: dddddddddd\n".to_string(),
            ]
        );
        // A bound that fits two whole turns joins them, oldest first, and
        // never lets a third straddle a boundary ("cc" would push chunk 0
        // to 43 > 30, so it opens its own chunk).
        let chunks = fill_chunks(turns.iter(), 30);
        assert_eq!(
            chunks,
            vec![
                "assistant: aa\nassistant: bbbb\n".to_string(),
                "assistant: cc\n".to_string(),
                "assistant: dddddddddd\n".to_string(),
            ]
        );
        // Concatenation is always the lossless rendered text, in order.
        assert_eq!(chunks.concat(), rendered(&turns));
    }

    #[tokio::test]
    async fn zero_reduction_never_accepted_even_at_target() {
        // before == target: any "compaction" is a zero reduction → rejected.
        let history = history(10);
        let before = 5_000;
        let req = CompactionRequest::new(before, before);
        let compactor = Compactor::deterministic_only();
        let plan = compactor.compact(&history, &ledger(), &req).await;
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
            archive_chunks: vec![],
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
