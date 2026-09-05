//! faktor-context — bounded context construction, the durable task ledger, and
//! a compaction engine that cannot death-spiral.
//!
//! Five memory classes (spec §8): immutable instructions, durable task state,
//! repository knowledge, recent conversation, historical artifacts. The
//! budget is enforced BEFORE anything is sent to a provider (spec §9); a
//! successful compaction must achieve the configured minimum reduction.

pub mod artifact;
pub mod assembler;
pub mod budget;
pub mod compactor;
pub mod estimator;
pub mod ledger;
pub mod wire_plan;

pub use artifact::{ArtifactRef, ArtifactWriter};
pub use assembler::{Evidence, RecentTurn};
pub use budget::ContextBudget;
pub use compactor::{CompactionPlan, CompactionRequest, CompactionStrategy, Compactor, Summarizer};
pub use estimator::{Estimator, GenericConservativeEstimator, TokenEstimator};
pub use ledger::{TaskLedger, TurnSummary};
pub use selection::{
    message_candidates_from_rows, select_by_utility, CandidateKind, ContextCandidate,
};
pub use wire_plan::{plan_wire_request, WirePlan};

/// Context selection by utility per token (audit 28): the planner chooses
/// what a turn's context window includes by utility-per-token instead of a
/// wholesale trim, and the conversation section (Message candidates) is
/// never dropped as a whole — evidence is.
///
/// [`select_by_utility`] is a pure, deterministic planner over
/// [`ContextCandidate`]s; [`message_candidates_from_rows`] seeds Message
/// candidates from the durable rows a bounded loader returns (newest-first
/// by contract), sized from their stored payload bytes.
pub mod selection {
    use faktor_store::MessageRow;

    /// One candidate for the turn's context window.
    #[derive(Debug, Clone, PartialEq)]
    pub struct ContextCandidate {
        /// Stable identity; tie-break key for equal-utility candidates.
        pub id: String,
        pub kind: CandidateKind,
        /// Payload size in bytes (for Message candidates: the stored JSON
        /// payload bytes of the durable row — the loader's byte accounting).
        pub bytes: usize,
        /// Conservative token estimate of the payload.
        pub estimate_tokens: u32,
        /// Utility in [0,1]; candidates at or below 0 are never selected.
        pub utility: f64,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub enum CandidateKind {
        /// Durable conversation rows; handled by the never-droppable
        /// conversation phases, never by the evidence ratio sort.
        Message,
        Symbol,
        FileNote,
        RepoRule,
        ToolNote,
        SubagentSummary,
    }

    /// Seed Message candidates from a bounded loader's rows (audit 28):
    /// the rows arrive newest-first (the store's
    /// `messages_backwards_bounded` contract), ids are the durable row ids,
    /// and the byte/token sizes come from the stored JSON payload — the
    /// same accounting the store's byte bound uses, so a token budget and a
    /// loader byte budget stay comparable. Utility of a message is 1.0:
    /// the conversation is the never-droppable class; selection ordering
    /// among messages is recency, decided by the phases below, not by this
    /// utility value.
    pub fn message_candidates_from_rows(rows: &[MessageRow]) -> Vec<ContextCandidate> {
        rows.iter()
            .map(|r| {
                let data_len = serde_json::to_string(&r.data).unwrap_or_default().len();
                ContextCandidate {
                    id: r.id.to_string(),
                    kind: CandidateKind::Message,
                    bytes: data_len,
                    // Conservative chars/3-style floor on the stored bytes;
                    // never 0 for a real row.
                    estimate_tokens: u32::try_from((data_len / 3).saturating_add(1))
                        .unwrap_or(u32::MAX),
                    utility: 1.0,
                }
            })
            .collect()
    }

    /// Select the context window under `token_budget`, greedy by utility
    /// per token with the conversation never droppable. Deterministic:
    /// the same input always yields the same output. Never exceeds
    /// `token_budget`; never duplicates a candidate.
    ///
    /// Budget classes, in order:
    ///
    /// 1. **Conversation cap.** Message candidates (in the caller's order —
    ///    newest-first, as the bounded loader produces them) are taken while
    ///    they fit `token_budget - evidence_reserve`, where the reserve is
    ///    `max(token_budget / 10, 1)` when at least one eligible
    ///    (utility > 0, `utility >= min_utility`) evidence candidate exists
    ///    and 0 otherwise. The reserve is what guarantees evidence is never
    ///    dropped wholesale: the conversation may claim most of the budget
    ///    but never all of it while evidence competes.
    /// 2. **Evidence.** All non-message candidates with `utility > 0` and
    ///    `utility >= min_utility`, greedy by `utility / estimate_tokens`
    ///    descending, ties broken by `id` ascending, while the total stays
    ///    under `token_budget`. A zero-token candidate has no price and is
    ///    never selected.
    /// 3. **Refill.** Unselected Message candidates (still newest-first) are
    ///    re-added while they fit the budget left by evidence — the
    ///    conversation is the budget absorber, so unused evidence reserve is
    ///    never wasted.
    ///
    /// Last-resort guarantee (conversation never droppable): when NO Message
    /// candidate was selected yet and messages exist, the selection is
    /// recomputed with the reserve set to zero (a single message that alone
    /// exceeds the conversation cap — an oversized exchange — displaces
    /// evidence rather than dropping the whole conversation). A zero budget
    /// yields the empty window: with no tokens even the conversation is
    /// excluded — nothing is ever selected beyond what the budget prices.
    pub fn select_by_utility(
        candidates: &[ContextCandidate],
        token_budget: u32,
        min_utility: f64,
    ) -> Vec<ContextCandidate> {
        if token_budget == 0 {
            return Vec::new();
        }
        let budget = u64::from(token_budget);
        let eligible = |c: &ContextCandidate| {
            c.kind != CandidateKind::Message
                && c.estimate_tokens > 0
                && c.utility > 0.0
                && c.utility >= min_utility
        };

        let has_evidence = candidates.iter().any(eligible);
        let reserve = if has_evidence {
            (budget / 10).max(1)
        } else {
            0
        };
        let selected = select_phases(candidates, budget, reserve, &eligible);
        let has_message = selected.iter().any(|c| c.kind == CandidateKind::Message);
        let messages_exist = candidates.iter().any(|c| c.kind == CandidateKind::Message);
        if !has_message && messages_exist {
            // Conversation would be absent entirely: recompute with no
            // evidence reserve — the conversation wins the whole budget.
            return select_phases(candidates, budget, 0, &eligible);
        }
        selected
    }

    fn select_phases(
        candidates: &[ContextCandidate],
        budget: u64,
        reserve: u64,
        eligible: &impl Fn(&ContextCandidate) -> bool,
    ) -> Vec<ContextCandidate> {
        // Phase 1: messages newest-first (caller order) up to the cap.
        // Phase 3 (refill) resumes after phase 2, so remember the first
        // message that no longer fit the cap.
        let cap = budget.saturating_sub(reserve);
        let mut out: Vec<ContextCandidate> = Vec::new();
        let mut used: u64 = 0;
        let mut refill_from = candidates.len();
        for (idx, c) in candidates.iter().enumerate() {
            if c.kind != CandidateKind::Message {
                continue;
            }
            let t = u64::from(c.estimate_tokens);
            if used.saturating_add(t) <= cap {
                out.push(c.clone());
                used += t;
            } else {
                refill_from = idx;
                break;
            }
        }

        // Phase 2: evidence greedy by utility/token, ties by id ascending.
        let mut evidence: Vec<&ContextCandidate> =
            candidates.iter().filter(|c| eligible(c)).collect();
        evidence.sort_by(|a, b| {
            let ra = a.utility / f64::from(a.estimate_tokens);
            let rb = b.utility / f64::from(b.estimate_tokens);
            rb.partial_cmp(&ra)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        for c in evidence {
            let t = u64::from(c.estimate_tokens);
            if used.saturating_add(t) > budget {
                continue;
            }
            out.push(c.clone());
            used += t;
        }

        // Phase 3: refill messages newest-first with whatever evidence left
        // unused (including the reserve it did not consume). The refill
        // STOPS at the first message that no longer fits — the conversation
        // must stay a contiguous newest prefix, never a hole-riddled
        // selection of older survivors.
        for c in candidates.iter().skip(refill_from) {
            if c.kind != CandidateKind::Message {
                continue;
            }
            let t = u64::from(c.estimate_tokens);
            if used.saturating_add(t) > budget {
                break;
            }
            out.push(c.clone());
            used += t;
        }
        out
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn msg(id: u32, tokens: u32) -> ContextCandidate {
            ContextCandidate {
                id: format!("m{id:05}"),
                kind: CandidateKind::Message,
                bytes: (tokens as usize).saturating_mul(3),
                estimate_tokens: tokens,
                utility: 1.0,
            }
        }

        fn evidence(id: &str, kind: CandidateKind, tokens: u32, utility: f64) -> ContextCandidate {
            ContextCandidate {
                id: id.into(),
                kind,
                bytes: (tokens as usize).saturating_mul(3),
                estimate_tokens: tokens,
                utility,
            }
        }

        /// Newest-first message seeds: ids mirror durable row ids, strictly
        /// increasing with recency, so descending ids = newest first.
        fn msgs_newest_first(n: u32, tokens: u32) -> Vec<ContextCandidate> {
            (1..=n).rev().map(|i| msg(i, tokens)).collect()
        }

        fn ids(sel: &[ContextCandidate]) -> Vec<String> {
            sel.iter().map(|c| c.id.clone()).collect()
        }

        /// (a) Budget 100 tokens, 50 messages of 10 tokens each plus one
        /// Symbol of utility .95 costing 5 tokens: the symbol is selected
        /// ahead of the 11th message — evidence survives by utility instead
        /// of being dropped wholesale (a plain messages-first planner would
        /// fill the whole budget with the newest 10 messages and lose the
        /// symbol; selection trades the 10th message for it).
        #[test]
        fn symbol_survives_ahead_of_the_eleventh_message() {
            let mut candidates = msgs_newest_first(50, 10);
            candidates.push(evidence("sym::parser", CandidateKind::Symbol, 5, 0.95));
            let sel = select_by_utility(&candidates, 100, 0.0);
            assert!(
                sel.iter().any(|c| c.id == "sym::parser"),
                "the symbol must survive selection (no wholesale evidence drop)"
            );
            // 9 messages (90 tokens) + the symbol (5) = 95 <= 100; the 10th
            // message (m00050, the oldest of the surviving candidates'
            // alternatives) is dropped to admit the evidence.
            let symbol = sel.iter().find(|c| c.id == "sym::parser").unwrap();
            assert_eq!(symbol.kind, CandidateKind::Symbol);
            assert_eq!(symbol.utility, 0.95);
            let selected_msgs: Vec<&ContextCandidate> = sel
                .iter()
                .filter(|c| c.kind == CandidateKind::Message)
                .collect();
            assert_eq!(selected_msgs.len(), 9);
            // The kept messages are exactly the newest 9 — a contiguous
            // newest prefix, the conversation never gains holes.
            assert_eq!(selected_msgs[0].id, "m00050");
            assert_eq!(selected_msgs[8].id, "m00042");
            let total: u64 = sel.iter().map(|c| u64::from(c.estimate_tokens)).sum();
            assert!(total <= 100, "budget respected: {total}");
            // Determinism: a second run is identical.
            let again = select_by_utility(&candidates, 100, 0.0);
            assert_eq!(sel, again);
            // Counterfactual: without the symbol there is room for the
            // newest 10 messages (100 tokens exactly) — evidence is what
            // gives the symbol its slot.
            let msgs_only: Vec<ContextCandidate> = candidates
                .into_iter()
                .filter(|c| c.kind == CandidateKind::Message)
                .collect();
            let plain = select_by_utility(&msgs_only, 100, 0.0);
            assert_eq!(plain.len(), 10, "messages alone fill the budget");
        }

        /// (b) 10k candidates against a small budget: selection is bounded
        /// in time (far under 50 ms), fully deterministic, duplicate-free,
        /// and never exceeds the budget.
        #[test]
        fn ten_thousand_candidates_select_in_bounded_time_without_dupes() {
            let mut candidates = msgs_newest_first(5_000, 10);
            for i in 0..5_000u32 {
                candidates.push(evidence(
                    &format!("note-{i:05}"),
                    CandidateKind::FileNote,
                    10,
                    (i % 100) as f64 / 100.0 + 0.001,
                ));
            }
            assert_eq!(candidates.len(), 10_000);
            let start = std::time::Instant::now();
            let sel = select_by_utility(&candidates, 300, 0.5);
            let elapsed = start.elapsed();
            assert!(
                elapsed.as_millis() < 50,
                "selection over 10k candidates took {elapsed:?}"
            );
            let total: u64 = sel.iter().map(|c| u64::from(c.estimate_tokens)).sum();
            assert!(total <= 300);
            let mut seen = std::collections::HashSet::new();
            for c in &sel {
                assert!(seen.insert(c.id.clone()), "duplicate {}", c.id);
            }
            let run2 = select_by_utility(&candidates, 300, 0.5);
            assert_eq!(sel, run2, "deterministic order");
            // Every message in the window is part of a contiguous newest
            // prefix.
            let msgs: Vec<&ContextCandidate> = sel
                .iter()
                .filter(|c| c.kind == CandidateKind::Message)
                .collect();
            assert!(msgs.windows(2).all(|w| w[0].id > w[1].id), "newest first");
        }

        /// (c) Zero budget: nothing is selected — evidence AND messages are
        /// excluded; only an empty window is legal.
        #[test]
        fn zero_budget_selects_nothing_at_all() {
            let mut candidates = msgs_newest_first(50, 10);
            candidates.push(evidence("sym::hot", CandidateKind::Symbol, 1, 1.0));
            let sel = select_by_utility(&candidates, 0, 0.0);
            assert!(sel.is_empty());
        }

        /// (d) All messages are kept when they fit; no evidence means no
        /// reserve is carved out (no wasted tokens).
        #[test]
        fn all_messages_kept_when_they_fit() {
            let candidates = msgs_newest_first(3, 10);
            let sel = select_by_utility(&candidates, 40, 0.0);
            assert_eq!(ids(&sel), vec!["m00003", "m00002", "m00001"]);
            // With enough budget the full transcript is preserved.
            let wide = select_by_utility(&candidates, 1000, 0.0);
            assert_eq!(wide.len(), 3);
        }

        /// (e) A candidate with utility 0 is never selected even while the
        /// budget has tokens left after the messages.
        #[test]
        fn zero_utility_candidate_never_selected() {
            let mut candidates = msgs_newest_first(3, 10);
            candidates.push(evidence("junk::noise", CandidateKind::ToolNote, 20, 0.0));
            let sel = select_by_utility(&candidates, 100, 0.0);
            assert_eq!(sel.len(), 3);
            assert!(!sel.iter().any(|c| c.id == "junk::noise"));
            // A hostile explicit minimum below zero cannot smuggle utility-0
            // candidates in either: 0 is excluded by the utility > 0 rule
            // regardless of min_utility.
            let sel = select_by_utility(&candidates, 100, f64::MIN);
            assert!(!sel.iter().any(|c| c.id == "junk::noise"));
        }

        /// min_utility prunes low-utility evidence even when the budget
        /// remains after messages.
        #[test]
        fn min_utility_filters_low_utility_evidence() {
            let mut candidates = msgs_newest_first(2, 10);
            candidates.push(evidence("a::dull", CandidateKind::FileNote, 10, 0.1));
            candidates.push(evidence("b::hot", CandidateKind::Symbol, 10, 0.9));
            let sel = select_by_utility(&candidates, 100, 0.5);
            assert!(sel.iter().any(|c| c.id == "b::hot"));
            assert!(!sel.iter().any(|c| c.id == "a::dull"));
        }

        /// Oversized single exchange: one message larger than the
        /// conversation cap alone — evidence yields to it instead of the
        /// whole conversation disappearing (never-droppable guarantee).
        #[test]
        fn oversized_conversation_displaces_evidence_not_itself() {
            let candidates = vec![
                msg(2, 95),
                msg(1, 5),
                evidence("sym::big", CandidateKind::Symbol, 20, 1.0),
            ];
            // Budget 100: the conversation cap is 90, and the newest message
            // alone costs 95 — under the cap it does not fit. The
            // never-droppable guarantee must still keep the conversation.
            let sel = select_by_utility(&candidates, 100, 0.0);
            let selected_msgs: Vec<&ContextCandidate> = sel
                .iter()
                .filter(|c| c.kind == CandidateKind::Message)
                .collect();
            assert!(!selected_msgs.is_empty(), "conversation never droppable");
            assert_eq!(selected_msgs.len(), 2, "newest message + refill");
            let total: u64 = sel.iter().map(|c| u64::from(c.estimate_tokens)).sum();
            assert!(total <= 100);
        }

        /// Messages arrive newest-first per the bounded-loader contract; the
        /// seed function derives sizes from real durable rows.
        #[test]
        fn message_seeding_from_durable_rows_is_sized_by_payload() {
            // Newest-first as `messages_backwards_bounded` yields them:
            // seq 4 (id 104) first.
            let rows: Vec<MessageRow> = (1..=4u64)
                .rev()
                .map(|seq| MessageRow {
                    id: 100 + seq as i64,
                    session_id: faktor_core::id::SessionId::new(1),
                    seq: seq as i64,
                    role: "user".into(),
                    data: serde_json::json!({"text": "a".repeat(seq as usize * 30)}),
                    created_ms: 0,
                })
                .collect();
            let seeded = message_candidates_from_rows(&rows);
            assert_eq!(seeded.len(), 4);
            for c in &seeded {
                assert_eq!(c.kind, CandidateKind::Message);
                assert_eq!(c.utility, 1.0);
                assert!(c.bytes > 0 && c.estimate_tokens > 0);
            }
            // Newest row (seq 4) has the largest payload.
            assert_eq!(seeded[0].id, "104");
            assert!(seeded[0].bytes > seeded[3].bytes);
            // Token estimate stays conservative vs the estimator's ~3
            // chars/token profile on dense ASCII.
            assert!(u64::from(seeded[0].estimate_tokens) >= (seeded[0].bytes as u64) / 3);
            // A full pipeline over real sizes: budget 100 tokens keeps the
            // newest messages by payload and can still admit evidence.
            let mut candidates = seeded;
            candidates.push(evidence("sym::x", CandidateKind::Symbol, 5, 0.99));
            let sel = select_by_utility(&candidates, 100, 0.0);
            let total: u64 = sel.iter().map(|c| u64::from(c.estimate_tokens)).sum();
            assert!(total <= 100);
            assert!(sel.iter().any(|c| c.kind == CandidateKind::Message));
        }

        /// Evidence keeps a reserved slice when it competes (the audit-28
        /// property), and refill never wastes budget when it is absent.
        #[test]
        fn evidence_gets_a_reserved_slice_and_refill_never_wastes_budget() {
            // 12 messages of 10 tokens, no evidence: all 10 that fit the
            // whole 100-token budget are kept (reserve = 0 without evidence).
            let candidates = msgs_newest_first(12, 10);
            assert_eq!(select_by_utility(&candidates, 100, 0.0).len(), 10);
            // With one tiny high-utility symbol competing, messages give up
            // a slot to it (reserve), then refill only with what remains.
            let mut candidates = msgs_newest_first(12, 10);
            candidates.push(evidence("sym::k", CandidateKind::Symbol, 5, 0.95));
            let sel = select_by_utility(&candidates, 100, 0.0);
            let total: u64 = sel.iter().map(|c| u64::from(c.estimate_tokens)).sum();
            assert!(total <= 100);
            assert!(sel.iter().any(|c| c.id == "sym::k"));
        }
    }
}
