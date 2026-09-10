//! faktor-context — bounded context construction, the durable task ledger, and
//! a compaction engine that cannot death-spiral.
//!
//! Five memory classes (spec §8): immutable instructions, durable task state,
//! repository knowledge, recent conversation, historical artifacts. The
//! budget is enforced BEFORE anything is sent to a provider (spec §9); a
//! successful compaction must achieve the configured minimum reduction.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use faktor_core::hash::FileHash;
use faktor_provider::{tokenizer_for, TokenizerId};

pub mod artifact;
pub mod assembler;
pub mod budget;
pub mod compactor;
pub mod estimator;
pub mod information;
pub mod ledger;
pub mod planner;
pub mod selection;
pub mod tokenizer;
pub mod wire_plan;

pub use artifact::{ArtifactRef, ArtifactWriter};
pub use assembler::{Evidence, RecentTurn};
pub use budget::ContextBudget;
pub use compactor::{
    CompactionPlan, CompactionRequest, CompactionStrategy, Compactor, EvidenceArchive, EvidenceRef,
    Summarizer, DEFAULT_SUMMARIZER_RESIDUE_BUDGET_TOKENS, EVIDENCE_BACKING_CAP_BYTES,
    EVIDENCE_COMPACT_BODY_MAX_BYTES, TOOL_OUTPUT_EVIDENCE_THRESHOLD_BYTES,
};
pub use estimator::{Estimator, GenericConservativeEstimator, TokenEstimator};
pub use information::{
    candidate_coverage, marginal_gain, remaining_coverage, required_candidates,
    select_by_information, InformationBudget, InformationError, InformationSelection, Need,
};
pub use ledger::{
    DurableTaskRows, ProjectedCheck, ProjectedChild, ProjectedDecision, TaskContextProjection,
    TaskLedger, TurnSummary,
};
pub use planner::{plan_context, plan_context_with_information};
pub use selection::{
    message_candidates_from_rows, select_by_utility, CandidateKind, CandidateRequirement,
    ContextCandidate, EvidenceLevel, NeedCoverage,
};
pub use tokenizer::{
    global_registry, ConservativeEstimatorTokenizer, TiktokenTokenizer, Tokenizer,
    TokenizerRegistry, MAX_EXACT_BYTES,
};
pub use wire_plan::{
    classify_prefix_cache, plan_wire_request, recompact_stable_prefix, PrefixCachePolicy,
    PrefixCacheState, PrefixObservation, PromptSegment, PromptSegments, PromptStability,
    SectionCosts, StablePrefix, WirePlan, WirePlanError, MAX_PROMPT_OBSERVATION_SEGMENTS,
    PROMPT_CACHEABLE_PREFIX_SEGMENTS, PROMPT_SEGMENT_COUNT,
};

// ======================================================================
// Token-count identity + cache (P0-81, audits 72/73)
//
// The estimator stays generic/conservative (estimator.rs). This section
// adds the *identity* of the tokenizer a model targets
// (`faktor_provider::tokenizer_for`, a pure prefix mapping — never a
// remote API), the registry of real local exact tokenizers
// (`tokenizer.rs`, audits 72/73), and a deterministic content-hash cache
// so the SAME (tokenizer id + version, content) pair is never counted
// twice:
//
// ```text
// count(text) = registry.count(tokenizer_id, text)
//               registry.resolve(id) = Some(t)  -> t.count(text)   // Exact
//               registry.resolve(id) = None     -> estimator(text) // UpperBound
// ```
//
// Cache entries are keyed by `(TokenizerId { family, version },
// FileHash)` (blake3 content hash — deterministic across processes and
// cache instances; identical to the workspace's content-hash contract).
// The version is part of the key, so bumping a vocabulary version
// invalidates every stale count. Content bytes are NEVER stored.
// Boundedness: the LRU caps entries at (identity, 32-byte hash) + count,
// hits cost only a hash, and the real BPE backends refuse inputs above
// `MAX_EXACT_BYTES`, falling back to the conservative estimator labeled
// UpperBound instead of allocating tokens proportional to hostile text.
// ======================================================================

/// Classification of a [`TokenEstimate`]: EXACT counts are produced only by
/// a real local tokenizer registered in the [`TokenizerRegistry`];
/// everything else is the conservative generic estimator's UPPER BOUND.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenEstimateKind {
    /// A real local tokenizer implementation counted the content exactly.
    Exact,
    /// The conservative generic estimator's value (never below the
    /// estimator's chars/3.4 floor), an upper bound on the real count.
    UpperBound,
}

/// A counted token total with its classification. `Exact`/`UpperBound` is
/// stored WITH the count so a cache hit returns exactly what the first
/// count produced — an exact result never degrades to an estimate and an
/// estimate is never mistaken for an exact one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenEstimate {
    pub count: u64,
    pub kind: TokenEstimateKind,
}

impl TokenEstimate {
    /// An EXACT count, constructible only by a real tokenizer
    /// implementation. Never use this for an estimator value.
    pub const fn exact(count: u64) -> Self {
        Self {
            count,
            kind: TokenEstimateKind::Exact,
        }
    }

    /// The conservative generic estimator's value, explicitly labeled as a
    /// bound — never an exact count.
    pub const fn upper_bound(count: u64) -> Self {
        Self {
            count,
            kind: TokenEstimateKind::UpperBound,
        }
    }
}

/// Default LRU entry cap of [`TokenCache`] (configurable via
/// [`TokenCache::with_capacity`]).
pub const DEFAULT_TOKEN_CACHE_CAPACITY: usize = 4096;

/// Deterministic content digest of a cache key (blake3 over the UTF-8
/// bytes): identical across processes and independent cache instances.
fn content_hash(text: &str) -> FileHash {
    FileHash::from(*blake3::hash(text.as_bytes()).as_bytes())
}

#[derive(Debug, Clone, Copy)]
struct CacheEntry {
    estimate: TokenEstimate,
    last_used: u64,
}

struct TokenCacheInner {
    entries: HashMap<(TokenizerId, FileHash), CacheEntry>,
    cap: usize,
    clock: u64,
    hits: u64,
    misses: u64,
}

impl TokenCacheInner {
    fn new(cap: usize) -> Self {
        Self {
            entries: HashMap::new(),
            cap: cap.max(1),
            clock: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Evict the least-recently-used entry (scan of ≤ cap entries). Ties
    /// (identical stamps are impossible — every insert bumps) are broken
    /// arbitrarily; only the evicted KEY choice can vary, never counts.
    fn evict_lru(&mut self) {
        let victim = self
            .entries
            .iter()
            .min_by_key(|(_, v)| v.last_used)
            .map(|(k, _)| *k);
        if let Some(k) = victim {
            self.entries.remove(&k);
        }
    }
}

/// Recover from a poisoned mutex instead of panicking: our critical
/// sections cannot panic, and a cache must never take the runtime down
/// because some unrelated panic poisoned the lock.
fn lock(inner: &Mutex<TokenCacheInner>) -> std::sync::MutexGuard<'_, TokenCacheInner> {
    inner
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Bounded LRU cache of token counts keyed by
/// `(TokenizerId { family, version }, blake3(content hash))` (P0-81,
/// audits 72/73). Thread-safe (interior mutex); misses run the registered
/// tokenizer / estimator OUTSIDE the lock so a huge hostile text never
/// stalls a concurrent planner.
pub struct TokenCache {
    inner: Mutex<TokenCacheInner>,
    registry: Arc<TokenizerRegistry>,
}

impl TokenCache {
    /// A cache with the default cap ([`DEFAULT_TOKEN_CACHE_CAPACITY`]) and
    /// the process-wide [`global_registry`] (real `o200k_base` /
    /// `cl100k_base` backends; every other identity falls back to the
    /// conservative estimator labeled `UpperBound`).
    pub fn new() -> Self {
        Self::with_registry(global_registry())
    }

    /// A cache with a configurable LRU cap (`0` is clamped to `1`) and the
    /// process-wide registry.
    pub fn with_capacity(cap: usize) -> Self {
        Self::with_registry_capacity(global_registry(), cap)
    }

    /// A cache over an explicit tokenizer registry (isolated tests; a
    /// future runtime that ships additional vocabularies).
    pub fn with_registry(registry: Arc<TokenizerRegistry>) -> Self {
        Self::with_registry_capacity(registry, DEFAULT_TOKEN_CACHE_CAPACITY)
    }

    /// A cache over an explicit registry with a configurable LRU cap (`0`
    /// is clamped to `1`).
    pub fn with_registry_capacity(registry: Arc<TokenizerRegistry>, cap: usize) -> Self {
        Self {
            inner: Mutex::new(TokenCacheInner::new(cap)),
            registry,
        }
    }

    /// Count `text` under `tokenizer` through the cache. The key is
    /// `(family, version, blake3(content))`, so a vocabulary version bump
    /// is a NEW entry (old counts are never silently reused). First call
    /// for a key is a miss → the registered exact backend, else the
    /// conservative estimator (labeled UpperBound); repeat calls with
    /// byte-identical content are hits returning the stored estimate with
    /// its kind preserved.
    pub fn count_tokenizer(&self, tokenizer: TokenizerId, text: &str) -> TokenEstimate {
        let key = (tokenizer, content_hash(text));
        {
            let mut g = lock(&self.inner);
            let inner = &mut *g; // direct field projections below: disjoint
            if let Some(entry) = inner.entries.get_mut(&key) {
                inner.hits = inner.hits.saturating_add(1);
                let now = inner.clock.wrapping_add(1);
                inner.clock = now;
                entry.last_used = now;
                return entry.estimate;
            }
            inner.misses = inner.misses.saturating_add(1);
        }
        // Miss: tokenizer/estimator run outside the lock.
        let estimate = self.registry.count(tokenizer, text);
        let mut g = lock(&self.inner);
        let inner = &mut *g;
        let now = inner.clock.wrapping_add(1);
        inner.clock = now;
        if let Some(entry) = inner.entries.get_mut(&key) {
            // Lost an insert race to another thread: the value is identical
            // (same deterministic inputs); keep the newest stamp.
            entry.last_used = now;
        } else {
            if inner.entries.len() >= inner.cap {
                inner.evict_lru();
            }
            inner.entries.insert(
                key,
                CacheEntry {
                    estimate,
                    last_used: now,
                },
            );
        }
        estimate
    }

    /// Count `text` under the tokenizer the MODEL maps to
    /// (`faktor_provider::tokenizer_for`, conservative fallback =
    /// GenericEstimator). The cache key never stores the model string —
    /// two models sharing one tokenizer identity share entries.
    pub fn count_for_model(&self, model: &str, text: &str) -> TokenEstimate {
        self.count_tokenizer(tokenizer_for(model, None), text)
    }

    /// Cache-only probe: `Some(entry)` when the (tokenizer, content) pair
    /// is cached. No estimation, no hit/miss counter effect, no LRU touch.
    pub fn peek(&self, tokenizer: TokenizerId, text: &str) -> Option<TokenEstimate> {
        let key = (tokenizer, content_hash(text));
        lock(&self.inner).entries.get(&key).map(|e| e.estimate)
    }

    /// LRU cap (entries).
    pub fn capacity(&self) -> usize {
        lock(&self.inner).cap
    }

    /// Current entry count (never exceeds `capacity()`).
    pub fn len(&self) -> usize {
        lock(&self.inner).entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Cache hits since construction (telemetry).
    pub fn hits(&self) -> u64 {
        lock(&self.inner).hits
    }

    /// Cache misses since construction (telemetry).
    pub fn misses(&self) -> u64 {
        lock(&self.inner).misses
    }
}

impl Default for TokenCache {
    fn default() -> Self {
        Self::new()
    }
}

/// The model-aware count entry the agent wire planning uses: routes the
/// text through the tokenizer the plan's model targets and returns the
/// count only. Models whose family has a registered real backend (`gpt-*`
/// → o200k/cl100k) count EXACTLY; every other family (Anthropic, Gemini,
/// Llama, unknown) returns the conservative generic estimator's count,
/// labeled `UpperBound` in the cache.
pub fn estimate_for_model(model: &str, text: &str, cache: &TokenCache) -> u64 {
    cache.count_for_model(model, text).count
}

#[cfg(test)]
mod token_cache_tests {
    use super::*;
    use faktor_provider::TokenFamily;

    /// A deterministic local "tokenizer" used to prove the Exact plumbing
    /// without depending on the real BPE vocabulary: counts
    /// `chars / divisor`. Registered under identities no model mapping ever
    /// returns (versions far from the frozen v1), so tests control exactly
    /// which ids are exact.
    struct FixtureTokenizer {
        id: TokenizerId,
        divisor: u64,
    }

    impl Tokenizer for FixtureTokenizer {
        fn id(&self) -> TokenizerId {
            self.id
        }

        fn count(&self, text: &str) -> TokenEstimate {
            TokenEstimate::exact((text.chars().count() as u64) / self.divisor)
        }
    }

    const FIXTURE_TOKENIZER: TokenizerId = TokenizerId {
        family: TokenFamily::O200kBase,
        version: 0x5EED_0001,
    };
    /// Same family as [`FIXTURE_TOKENIZER`], different vocabulary version:
    /// the cache key must treat it as a different tokenizer entirely.
    const FIXTURE_V2: TokenizerId = TokenizerId {
        family: TokenFamily::O200kBase,
        version: 0x5EED_0002,
    };

    fn fixture_registry() -> Arc<TokenizerRegistry> {
        let mut registry = TokenizerRegistry::new();
        registry.register(Arc::new(FixtureTokenizer {
            id: FIXTURE_TOKENIZER,
            divisor: 1,
        }));
        registry.register(Arc::new(FixtureTokenizer {
            id: FIXTURE_V2,
            divisor: 2,
        }));
        Arc::new(registry)
    }

    fn est(text: &str) -> u64 {
        u64::try_from(Estimator.estimate_tokens(text)).unwrap_or(u64::MAX)
    }

    #[test]
    fn hit_after_second_call_with_identical_content_miss_after_change() {
        let cache = TokenCache::new();
        let text = "fn main() { let x = 1; } // ".repeat(40);
        let first = cache.count_for_model("gpt-5", &text);
        assert_eq!((cache.hits(), cache.misses()), (0, 1));
        assert_eq!(
            first.kind,
            TokenEstimateKind::Exact,
            "gpt-5 maps to a registered real o200k backend"
        );
        assert!(first.count > 0);

        let second = cache.count_for_model("gpt-5", &text);
        assert_eq!(first, second);
        assert_eq!(
            (cache.hits(), cache.misses()),
            (1, 1),
            "hit after second call"
        );

        // Content change: the KEY is the content hash, so the pair misses
        // even when the count happens to coincide.
        let changed = cache.count_for_model("gpt-5", &format!("{text}x"));
        assert_eq!(changed.kind, TokenEstimateKind::Exact);
        assert_eq!(
            (cache.hits(), cache.misses()),
            (1, 2),
            "content change must miss"
        );
        assert_eq!(cache.len(), 2, "changed content is a distinct entry");
        assert!(
            cache.peek(TokenizerId::O200K_BASE, &text).is_some(),
            "the original entry survives"
        );
    }

    #[test]
    fn entries_are_keyed_by_tokenizer_identity_and_content() {
        let cache = TokenCache::with_registry(fixture_registry());
        let text = "the same content under different tokenizers".repeat(3);
        let o200k = TokenizerId::O200K_BASE;
        let anthropic = TokenizerId::ANTHROPIC;

        let a = cache.count_tokenizer(o200k, &text);
        let b = cache.count_tokenizer(anthropic, &text);
        assert_eq!(a.kind, TokenEstimateKind::UpperBound);
        assert_eq!(a.count, est(&text));
        assert_eq!(b.count, a.count, "estimator is tokenizer-agnostic");
        assert_eq!(
            cache.len(),
            2,
            "same content, different tokenizer → distinct entries"
        );

        // The fixture identity has a registered backend: Exact.
        let exact = cache.count_tokenizer(FIXTURE_TOKENIZER, &text);
        assert_eq!(exact.kind, TokenEstimateKind::Exact);
        assert_ne!(exact.count, a.count, "exact fixture count is distinct");
        assert_eq!(exact.count, text.chars().count() as u64);
        assert_eq!(cache.len(), 3);

        // The same tokenizer + a materially different byte string is a
        // different entry (a 1-char change may not move the estimate — the
        // KEY is the content hash, so entries are distinct regardless of
        // count equality).
        let different = format!("{text}\n{}", "x".repeat(1000));
        let c = cache.count_tokenizer(o200k, &different);
        assert_eq!(cache.len(), 4);
        assert_ne!(c.count, a.count);
        assert_ne!(cache.peek(o200k, &different), Some(a));
    }

    #[test]
    fn lru_eviction_at_cap_never_drops_recent_entries() {
        let cache = TokenCache::with_capacity(2);
        let t = TokenizerId::GENERIC_ESTIMATOR;
        let (ta, tb, tc) = ("alpha content", "beta content", "gamma content");
        cache.count_tokenizer(t, ta);
        cache.count_tokenizer(t, tb);
        // Touch `ta` so it is the most-recently-used entry.
        assert_eq!(cache.count_tokenizer(t, ta).count, est(ta));
        cache.count_tokenizer(t, tc); // evicts tb (LRU)
        assert_eq!(cache.len(), 2, "eviction keeps the cache at cap");
        assert!(cache.peek(t, ta).is_some(), "touched entry survives");
        assert!(cache.peek(t, tc).is_some());
        assert!(
            cache.peek(t, tb).is_none(),
            "least-recently-used entry evicted"
        );
        // Evicted content misses again and is re-counted deterministically.
        let again = cache.count_tokenizer(t, tb);
        assert_eq!(again.count, est(tb));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn exact_backend_counts_are_stored_and_kind_is_preserved_on_hits() {
        let cache = TokenCache::with_registry(fixture_registry());
        let text = "exact backend content 汉字 😀".repeat(5);
        let first = cache.count_tokenizer(FIXTURE_TOKENIZER, &text);
        assert_eq!(first.kind, TokenEstimateKind::Exact);
        assert_eq!(first.count, text.chars().count() as u64);
        assert_eq!(cache.peek(FIXTURE_TOKENIZER, &text), Some(first));
        // Hit returns the stored EXACT row (never degrades to UpperBound).
        let hit = cache.count_tokenizer(FIXTURE_TOKENIZER, &text);
        assert_eq!(hit.kind, TokenEstimateKind::Exact);
        assert_eq!(hit.count, first.count);
        assert_eq!((cache.hits(), cache.misses()), (1, 1));
        // Unregistered tokenizers → UpperBound fallback, stored under a
        // DIFFERENT entry.
        let fallback = cache.count_tokenizer(TokenizerId::GEMINI, &text);
        assert_eq!(fallback.kind, TokenEstimateKind::UpperBound);
        assert_eq!(fallback.count, est(&text));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.peek(TokenizerId::GEMINI, &text), Some(fallback));
    }

    #[test]
    fn unregistered_families_and_versions_fall_back_to_conservative_upper_bound() {
        // Registered exact backends: only o200k@v1 and cl100k@v1. Every
        // other named family (Anthropic/Gemini/Llama) and every unknown
        // version falls back to the estimator, labeled UpperBound. The
        // labels are part of the contract, so assert the exact kind too.
        let cache = TokenCache::new();
        for id in [
            TokenizerId::ANTHROPIC,
            TokenizerId::GEMINI,
            TokenizerId::LLAMA,
            TokenizerId::GENERIC_ESTIMATOR,
            TokenizerId {
                family: TokenFamily::O200kBase,
                version: 2,
            },
            TokenizerId {
                family: TokenFamily::Cl100kBase,
                version: 0xDEAD_BEEF,
            },
            TokenizerId {
                family: TokenFamily::GenericEstimator,
                version: 0,
            },
        ] {
            let got = cache.count_tokenizer(id, "anything");
            assert_eq!(got.kind, TokenEstimateKind::UpperBound, "{id}");
            assert_eq!(got.count, est("anything"), "{id}");
        }
        // ... while the two real backends are Exact on the SAME content.
        assert_eq!(
            cache
                .count_tokenizer(TokenizerId::O200K_BASE, "anything")
                .kind,
            TokenEstimateKind::Exact
        );
        assert_eq!(
            cache
                .count_tokenizer(TokenizerId::CL100K_BASE, "anything")
                .kind,
            TokenEstimateKind::Exact
        );
    }

    #[test]
    fn same_prompt_two_tokenizers_reports_distinct_counts_exact_only_where_registered() {
        // The audit lock: the SAME prompt under two real vocabularies must
        // report different exact counts; families without a local
        // vocabulary must stay UpperBound, never exact.
        let cache = TokenCache::new();
        let prompt = "fn main() { let x = 1; } // the quick brown fox 汉字 😀".repeat(12);
        let o200k = cache.count_tokenizer(TokenizerId::O200K_BASE, &prompt);
        let cl100k = cache.count_tokenizer(TokenizerId::CL100K_BASE, &prompt);
        assert_eq!(o200k.kind, TokenEstimateKind::Exact);
        assert_eq!(cl100k.kind, TokenEstimateKind::Exact);
        assert_ne!(
            o200k.count, cl100k.count,
            "o200k and cl100k are different vocabularies"
        );
        for id in [TokenizerId::ANTHROPIC, TokenizerId::GEMINI] {
            let got = cache.count_tokenizer(id, &prompt);
            assert_eq!(got.kind, TokenEstimateKind::UpperBound, "{id}");
            assert_eq!(got.count, est(&prompt), "{id}");
        }
        // Labels survive a cache hit (stored WITH the count).
        assert_eq!(
            cache.count_tokenizer(TokenizerId::O200K_BASE, &prompt).kind,
            TokenEstimateKind::Exact
        );
        assert_eq!(
            cache.count_tokenizer(TokenizerId::GEMINI, &prompt).kind,
            TokenEstimateKind::UpperBound
        );
    }

    #[test]
    fn cache_key_includes_tokenizer_version_so_a_bump_invalidates() {
        // Same family, same content, different vocabulary version: the ids
        // are DIFFERENT keys, so the version bump re-counts instead of
        // reusing the old vocabulary's count.
        let cache = TokenCache::with_registry(fixture_registry());
        let text = "six six";
        let v1 = cache.count_tokenizer(FIXTURE_TOKENIZER, text);
        let v2 = cache.count_tokenizer(FIXTURE_V2, text);
        assert_eq!(v1.kind, TokenEstimateKind::Exact);
        assert_eq!(v2.kind, TokenEstimateKind::Exact);
        assert_eq!(v1.count, 7, "v1 fixture: chars / 1");
        assert_eq!(v2.count, 3, "v2 fixture: chars / 2, a NEW vocabulary");
        assert_ne!(v1.count, v2.count);
        assert_eq!(cache.len(), 2, "versions are distinct cache entries");
        assert_eq!((cache.hits(), cache.misses()), (0, 2));
        // Re-asking v1 is a hit of the v1 row only; v2 never poisons it.
        assert_eq!(cache.count_tokenizer(FIXTURE_TOKENIZER, text), v1);
        assert_eq!((cache.hits(), cache.misses()), (1, 2));
        // A cache whose registry only knows v2 cannot answer v1 exactly:
        // the old count is not silently reused under the new version.
        let mut only_v2 = TokenizerRegistry::new();
        only_v2.register(Arc::new(FixtureTokenizer {
            id: FIXTURE_V2,
            divisor: 2,
        }));
        let bumped = TokenCache::with_registry(Arc::new(only_v2));
        assert_eq!(bumped.count_tokenizer(FIXTURE_V2, text).count, 3);
        assert_eq!(
            bumped.count_tokenizer(FIXTURE_TOKENIZER, text).kind,
            TokenEstimateKind::UpperBound,
            "v1 has no backend in the bumped registry"
        );
    }

    #[test]
    fn two_cache_instances_agree_on_every_count() {
        // Cross-instance (and thereby cross-process) determinism: cache
        // contents, caps and model strings never change the COUNT of a
        // (tokenizer, content) pair — blake3 keys + pure mapping + the
        // real vocabularies / estimator formula are the only inputs.
        let a = TokenCache::with_capacity(1); // hostile tiny cap: constant eviction
        let b = TokenCache::with_capacity(4096);
        let samples = [
            ("gpt-5", "fn main() {}"),
            ("gpt-4", "fn main() {}"),
            ("claude-opus-4-1", "fn main() {}"),
            ("qwen3.8", "qwen over ollama"),
            ("no-such-model-anywhere", "anything at all"),
            ("gpt-4o", "汉字与😀混排的样本"),
        ];
        for (model, text) in samples {
            assert_eq!(
                a.count_for_model(model, text),
                b.count_for_model(model, text),
                "{model} counts must not depend on the cache instance"
            );
        }
        assert_eq!(
            a.count_for_model("gpt-5", "fn main() {}"),
            b.count_for_model("gpt-5", "fn main() {}")
        );
        // Two independently built fixture registries agree as well.
        let c = TokenCache::with_registry(fixture_registry());
        let d = TokenCache::with_registry(fixture_registry());
        assert_eq!(
            c.count_tokenizer(FIXTURE_TOKENIZER, "determinism"),
            d.count_tokenizer(FIXTURE_TOKENIZER, "determinism")
        );
    }

    #[test]
    fn models_sharing_a_tokenizer_share_entries_but_models_do_not() {
        // Identity-level dedup: gpt-5 and GPT-4o both map to o200k_base@v1,
        // so the second model's count of identical content is a HIT.
        let cache = TokenCache::new();
        let text = "dedup me please";
        let first = cache.count_for_model("gpt-5", text);
        let again = cache.count_for_model("GPT-4o", text);
        assert_eq!(again, first);
        assert_eq!(
            (cache.hits(), cache.misses()),
            (1, 1),
            "identity-level dedup"
        );
        // Unknown models map to GenericEstimator and never share entries
        // with a named tokenizer family.
        cache.count_for_model("totally-unknown-model", text);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.hits(), 1, "generic rows still miss");
    }

    #[test]
    fn huge_content_is_capped_bounded_and_cached_without_repeated_work() {
        // 10 MiB hostile content: above MAX_EXACT_BYTES the real backend
        // refuses the BPE pass and the conservative estimator answers,
        // honestly labeled UpperBound (an upper bound is never called
        // exact). Boundedness: one entry per (tokenizer, content) — bytes
        // are never stored — and a repeat is a pure hash hit.
        let cache = TokenCache::new();
        let ascii = "x".repeat(10 << 20);
        let first = cache.count_for_model("gpt-5", &ascii);
        assert_eq!(first.kind, TokenEstimateKind::UpperBound);
        assert_eq!(first.count, est(&ascii), "estimator's documented contract");
        assert!(
            first.count < ascii.len() as u64,
            "count stays far below bytes"
        );
        let second = cache.count_for_model("gpt-5", &ascii);
        assert_eq!(second.count, first.count);
        assert_eq!((cache.hits(), cache.misses()), (1, 1));
        assert_eq!(cache.len(), 1, "one entry regardless of content size");

        // Multibyte hostile content: identical contract, no panic, and an
        // exact 0 for empty input stays consistent with the estimator.
        let cjk = "汉字汉字汉字汉字😀😀".repeat(1 << 20);
        let _ = cache.count_for_model("claude", &cjk);
        assert!(cache.len() <= 2);
        assert_eq!(cache.count_for_model("gemini-2.5-pro", "").count, 0);
    }

    #[test]
    fn hostile_keys_and_content_never_panic_and_counters_stay_coherent() {
        let cache = TokenCache::with_capacity(3);
        let hostile: Vec<String> = vec![
            String::new(),
            " ".into(),
            "\n\t".into(),
            "\0".into(),
            "😀".repeat(64),
            "## markers".into(),
            "a".repeat(4096),
        ];
        for (i, h) in hostile.iter().enumerate() {
            let t = cache.count_for_model("gpt-4o", h);
            assert_eq!(
                t.kind,
                TokenEstimateKind::Exact,
                "row {i}: gpt-4o has a registered real backend"
            );
            assert_eq!(t.count > 0, !h.is_empty(), "row {i}: zero only for empty");
            // peek never mutates: counters must reflect ONLY count calls.
            let before = cache.hits() + cache.misses();
            assert!(cache.peek(TokenizerId::O200K_BASE, h).is_some());
            assert_eq!(cache.hits() + cache.misses(), before);
        }
        assert!(cache.len() <= 3, "bounded by the LRU cap");
        assert_eq!(cache.hits() + cache.misses(), hostile.len() as u64);
        // Model strings are never part of the key: hostile MODEL names
        // route to the conservative generic identity and cannot panic.
        for model in ["", "\0model", "a/b/c", "..", "😀"] {
            let t = cache.count_for_model(model, "text");
            assert!(t.count > 0);
            assert_eq!(t.kind, TokenEstimateKind::UpperBound);
        }
    }

    #[test]
    fn telemetry_counters_track_every_lookup() {
        let cache = TokenCache::with_capacity(8);
        let mut lookups = 0u64;
        let mut hits = 0u64;
        let mut misses = 0u64;
        let texts = ["one", "two", "one", "three", "one", "two", "four", "four"];
        for (i, text) in texts.iter().enumerate() {
            cache.count_for_model("gpt-5", text);
            lookups += 1;
            if texts[..i].contains(text) {
                hits += 1;
            } else {
                misses += 1;
            }
            assert_eq!(cache.hits(), hits, "telemetry drift at row {i}");
            assert_eq!(cache.misses(), misses, "telemetry drift at row {i}");
        }
        assert_eq!(cache.hits() + cache.misses(), lookups);
        assert_eq!(cache.hits(), 4);
        assert_eq!(cache.misses(), 4);
    }

    #[test]
    fn concurrent_counts_are_consistent_and_never_mix_tokenizers() {
        // Adversarial concurrency: many threads hammer one cache with the
        // same and different (tokenizer, content) pairs. Every observed
        // value must equal an independent reference, and the counters must
        // add up exactly.
        let cache = TokenCache::new();
        let reference_cache = TokenCache::new();
        let texts: Vec<String> = (0..8)
            .map(|i| format!("concurrent content {i} {}", "y".repeat(i * 17)))
            .collect();
        let ids = [
            TokenizerId::O200K_BASE,
            TokenizerId::CL100K_BASE,
            TokenizerId::ANTHROPIC,
            TokenizerId::GEMINI,
        ];
        let cases: Vec<(TokenizerId, String, TokenEstimate)> = ids
            .iter()
            .flat_map(|id| texts.iter().map(move |t| (*id, t.clone())))
            .map(|(id, text)| {
                let want = reference_cache.count_tokenizer(id, &text);
                (id, text, want)
            })
            .collect();
        let calls = cases.len() as u64;
        let cache_ref = &cache;
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let cases = &cases;
                scope.spawn(move || {
                    for (id, text, want) in cases {
                        assert_eq!(
                            cache_ref.count_tokenizer(*id, text),
                            *want,
                            "{id} count mixed under concurrency"
                        );
                    }
                });
            }
        });
        assert_eq!(
            cache.hits() + cache.misses(),
            calls * 8,
            "every lookup is counted exactly once"
        );
        // The exact/UpperBound split never crosses under concurrency.
        assert_eq!(
            cache
                .count_tokenizer(TokenizerId::O200K_BASE, &texts[0])
                .kind,
            TokenEstimateKind::Exact
        );
        assert_eq!(
            cache.count_tokenizer(TokenizerId::GEMINI, &texts[0]).kind,
            TokenEstimateKind::UpperBound
        );
    }
}
