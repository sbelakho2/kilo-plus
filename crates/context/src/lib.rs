//! faktor-context — bounded context construction, the durable task ledger, and
//! a compaction engine that cannot death-spiral.
//!
//! Five memory classes (spec §8): immutable instructions, durable task state,
//! repository knowledge, recent conversation, historical artifacts. The
//! budget is enforced BEFORE anything is sent to a provider (spec §9); a
//! successful compaction must achieve the configured minimum reduction.

use std::collections::HashMap;
use std::sync::Mutex;

use faktor_core::hash::FileHash;
use faktor_provider::{tokenizer_for, TokenizerId};

pub mod artifact;
pub mod assembler;
pub mod budget;
pub mod compactor;
pub mod estimator;
pub mod ledger;
pub mod planner;
pub mod selection;
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

// ======================================================================
// Token-count identity + cache (P0-81)
//
// The estimator stays generic/conservative (estimator.rs). This section
// adds the *identity* of the tokenizer a model targets
// (`faktor_provider::tokenizer_for`, a pure prefix mapping — never a
// remote API) and a deterministic content-hash cache so the SAME
// (tokenizer id/version, content) pair is never counted twice:
//
// ```text
// count(text) = exact_count(tokenizer, text)      // a real local
//               .unwrap_or_else(                  // tokenizer impl;
//                  estimator(text))               // today: None -> the
//                                                  // conservative generic
//                                                  // estimator, labeled
//                                                  // UpperBound
// ```
//
// Cache entries are keyed by `(TokenizerId, FileHash)` (blake3 content
// hash — deterministic across processes and cache instances; identical to
// the workspace's content-hash contract). Content bytes are NEVER stored.
// The estimator is a single pass over the in-RAM `&str` with no copy, so
// the huge-content row follows the estimator's OWN documented contract
// (its tests lock the exact full-text formula at 8 MiB): no prefix cap,
// because a cap would under-count the tail and break the estimator's
// upper-bound property that the wire-plan accounting lockstep relies on.
// Boundedness comes from the LRU (entries are (identity, 32-byte hash) +
// count) and from hits costing only a hash.
// ======================================================================

/// Classification of a [`TokenEstimate`]: EXACT counts are produced only by
/// a real local tokenizer behind the seam; everything else is the
/// conservative generic estimator's UPPER BOUND.
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
    fn exact(count: u64) -> Self {
        Self {
            count,
            kind: TokenEstimateKind::Exact,
        }
    }

    fn upper_bound(count: u64) -> Self {
        Self {
            count,
            kind: TokenEstimateKind::UpperBound,
        }
    }
}

/// The seam a local tokenizer implementation plugs into. A future
/// tokenizer implementation names its [`TokenizerId`] and returns
/// `Some(exact)`; the cache then stores the exact count under that
/// identity. Fixture tokenizers used by the adversarial tests register
/// here too.
pub type ExactCountFn = fn(&TokenizerId, &str) -> Option<u64>;

/// The default seam: no local tokenizer exists in the workspace today, so
/// every request falls back to the conservative generic estimator and is
/// labeled [`TokenEstimateKind::UpperBound`]. Never consults a remote
/// tokenizer API.
pub fn exact_count(_tokenizer: &TokenizerId, _text: &str) -> Option<u64> {
    None
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
/// `(TokenizerId, blake3(content hash))` (P0-81). Thread-safe (interior
/// mutex); misses run the seam/estimator OUTSIDE the lock so a huge
/// hostile text never stalls a concurrent planner.
pub struct TokenCache {
    inner: Mutex<TokenCacheInner>,
    exact: ExactCountFn,
}

impl TokenCache {
    /// A cache with the default cap ([`DEFAULT_TOKEN_CACHE_CAPACITY`]) and
    /// the default seam ([`exact_count`] — no local tokenizer yet).
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_TOKEN_CACHE_CAPACITY)
    }

    /// A cache with a configurable LRU cap (`0` is clamped to `1`).
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            inner: Mutex::new(TokenCacheInner::new(cap)),
            exact: exact_count,
        }
    }

    /// A cache whose exact-count seam is `exact` instead of the default.
    /// Reserved for real tokenizer implementations and fixture tokenizers
    /// in tests.
    pub fn with_exact_seam(exact: ExactCountFn) -> Self {
        Self {
            inner: Mutex::new(TokenCacheInner::new(DEFAULT_TOKEN_CACHE_CAPACITY)),
            exact,
        }
    }

    /// Count `text` under `tokenizer` through the cache. First call for a
    /// (tokenizer, content-hash) pair is a miss → exact seam, else the
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
        // Miss: seam + estimator run outside the lock.
        let estimate = match (self.exact)(&tokenizer, text) {
            Some(count) => TokenEstimate::exact(count),
            None => TokenEstimate::upper_bound(
                u64::try_from(Estimator.estimate_tokens(text)).unwrap_or(u64::MAX),
            ),
        };
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
/// count only (an UpperBound from the conservative estimator until a real
/// local tokenizer lands behind the seam). Falls back exactly to
/// [`Estimator::estimate_tokens`] values today, so budget accounting is
/// unchanged by the cache layer.
pub fn estimate_for_model(model: &str, text: &str, cache: &TokenCache) -> u64 {
    cache.count_for_model(model, text).count
}

#[cfg(test)]
mod token_cache_tests {
    use super::*;
    use faktor_provider::TokenFamily;

    /// A tokenizer identity no model mapping ever returns (version is far
    /// from the mapping's frozen v1): the fixture seam counts every char as
    /// exactly one token — a deterministic "local tokenizer" that exists
    /// ONLY for tests, proving Exact flows through the seam end-to-end.
    const FIXTURE_TOKENIZER: TokenizerId = TokenizerId {
        family: TokenFamily::O200kBase,
        version: 0x5EED_0001,
    };

    fn fixture_exact(tokenizer: &TokenizerId, text: &str) -> Option<u64> {
        if *tokenizer == FIXTURE_TOKENIZER {
            Some(text.chars().count() as u64)
        } else {
            None
        }
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
        assert_eq!(first.kind, TokenEstimateKind::UpperBound);
        assert_eq!(first.count, est(&text), "fallback = the estimator, exactly");

        let second = cache.count_for_model("gpt-5", &text);
        assert_eq!(first, second);
        assert_eq!(
            (cache.hits(), cache.misses()),
            (1, 1),
            "hit after second call"
        );

        // Content change: the KEY is the content hash, so the pair misses
        // even when the estimator's rounded value happens to coincide.
        let changed = cache.count_for_model("gpt-5", &format!("{text}x"));
        assert_eq!(changed.kind, TokenEstimateKind::UpperBound);
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
        let cache = TokenCache::with_exact_seam(fixture_exact);
        let text = "the same content under different tokenizers".repeat(3);
        let o200k = TokenizerId::O200K_BASE;
        let anthropic = TokenizerId::ANTHROPIC;

        let a = cache.count_tokenizer(o200k, &text);
        let b = cache.count_tokenizer(anthropic, &text);
        assert_eq!(a.kind, TokenEstimateKind::UpperBound);
        assert_eq!(a.count, est(&text));
        assert_eq!(b.count, a.count, "estimator is tokenizer-agnostic today");
        assert_eq!(
            cache.len(),
            2,
            "same content, different tokenizer → distinct entries"
        );

        let exact = cache.count_tokenizer(FIXTURE_TOKENIZER, &text);
        assert_eq!(exact.kind, TokenEstimateKind::Exact);
        assert_ne!(exact.count, a.count, "exact fixture count is distinct");
        assert_eq!(exact.count, text.chars().count() as u64);
        assert_eq!(cache.len(), 3);

        // The same tokenizer + a materially different byte string is a
        // different entry (a 1-char change may not move the estimator's
        // value — the KEY is the content hash, so entries are distinct
        // regardless of count equality).
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
    fn exact_seam_counts_are_stored_and_kind_is_preserved_on_hits() {
        let cache = TokenCache::with_exact_seam(fixture_exact);
        let text = "exact seam content 汉字 😀".repeat(5);
        let first = cache.count_tokenizer(FIXTURE_TOKENIZER, &text);
        assert_eq!(first.kind, TokenEstimateKind::Exact);
        assert_eq!(first.count, text.chars().count() as u64);
        assert_eq!(cache.peek(FIXTURE_TOKENIZER, &text), Some(first));
        // Hit returns the stored EXACT row (never degrades to UpperBound).
        let hit = cache.count_tokenizer(FIXTURE_TOKENIZER, &text);
        assert_eq!(hit.kind, TokenEstimateKind::Exact);
        assert_eq!(hit.count, first.count);
        assert_eq!((cache.hits(), cache.misses()), (1, 1));
        // The seam returns None for non-fixture tokenizers → UpperBound
        // fallback, stored under a DIFFERENT entry.
        let fallback = cache.count_tokenizer(TokenizerId::GEMINI, &text);
        assert_eq!(fallback.kind, TokenEstimateKind::UpperBound);
        assert_eq!(fallback.count, est(&text));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.peek(TokenizerId::GEMINI, &text), Some(fallback));
    }

    #[test]
    fn default_seam_is_none_and_generic_estimator_is_the_fallback() {
        // Today the workspace has no local tokenizer: EVERY tokenizer id
        // (named families included) falls back to the estimator, labeled
        // UpperBound — exactness is reserved for when a real implementation
        // lands behind the seam.
        for id in [
            TokenizerId::O200K_BASE,
            TokenizerId::CL100K_BASE,
            TokenizerId::ANTHROPIC,
            TokenizerId::GEMINI,
            TokenizerId::LLAMA,
            TokenizerId::GENERIC_ESTIMATOR,
        ] {
            assert_eq!(exact_count(&id, "anything"), None);
            let cache = TokenCache::new();
            let got = cache.count_tokenizer(id, "anything");
            assert_eq!(got.kind, TokenEstimateKind::UpperBound);
            assert_eq!(got.count, est("anything"));
        }
    }

    #[test]
    fn two_cache_instances_agree_on_every_count() {
        // Cross-instance (and thereby cross-process) determinism: cache
        // contents, caps and model strings never change the COUNT of a
        // (tokenizer, content) pair — blake3 keys + pure mapping + the
        // estimator's formula are the only inputs.
        let a = TokenCache::with_capacity(1); // hostile tiny cap: constant eviction
        let b = TokenCache::with_capacity(4096);
        let samples = [
            ("gpt-5", "fn main() {}"),
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
    }

    #[test]
    fn models_sharing_a_tokenizer_share_entries_but_models_do_not() {
        // Identity-level dedup: gpt-5 and GPT-4o both map to o200k_base@v1,
        // so the second model's count of identical content is a HIT.
        let cache = TokenCache::new();
        let text = "dedup me please";
        cache.count_for_model("gpt-5", text);
        let again = cache.count_for_model("GPT-4o", text);
        assert_eq!(again.count, est(text));
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
    fn huge_content_follows_the_estimator_contract_and_is_bounded_and_cached() {
        // 10 MiB hostile content (the P0-81 row): the estimator is a
        // single-pass formula over the in-RAM &str (no copy, no allocation)
        // and its tests lock the exact full-text formula at 8 MiB — so the
        // cache honors that contract and reports the estimator's exact
        // value as an UpperBound. No prefix cap exists: capping would
        // silently under-count the tail and break the upper-bound property
        // the wire-plan accounting relies on. Boundedness: one entry per
        // (tokenizer, content) — bytes are never stored.
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
            assert_eq!(t.count, est(h), "row {i}: estimator equality");
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
}
