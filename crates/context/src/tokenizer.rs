//! Exact tokenizer backends (audits 72/73).
//!
//! The generic [`Estimator`](crate::Estimator) is a conservative HEURISTIC
//! and stays the labeled fallback; this module adds the *real* local
//! tokenizer surface the wire accounting was missing:
//!
//! ```text
//! Tokenizer          id() -> TokenizerId        // family + version identity
//!                    count(text) -> TokenEstimate { count, kind }
//!
//! TokenizerRegistry  TokenizerId -> Arc<dyn Tokenizer>   (exact backends)
//!                    unknown id  -> conservative UpperBound (never exact)
//! ```
//!
//! The registered exact backends are the real OpenAI BPE vocabularies via
//! `tiktoken-rs` ([`TiktokenTokenizer`]): `o200k_base` and `cl100k_base`
//! load their vocabularies from `include_str!`-embedded assets compiled into
//! the binary — no network, no file system, no remote tokenizer API at
//! runtime. Anthropic/Gemini/Llama have NO local vocabulary in the
//! workspace, so they are deliberately NOT registered: they resolve to the
//! conservative generic estimator and are labeled
//! [`TokenEstimateKind::UpperBound`](crate::TokenEstimateKind::UpperBound).
//! Registering a real backend is the documented hook
//! ([`TokenizerRegistry::register`]) — an adapter that ships a vocabulary
//! file implements [`Tokenizer`] and registers under its own versioned
//! [`TokenizerId`], bumping `version` whenever the vocabulary changes so
//! every cached count keyed by the old identity is invalidated.
//!
//! Boundedness: an exact BPE pass allocates tokens proportional to the
//! input, so [`TiktokenTokenizer::count`] only runs BPE for inputs up to
//! [`MAX_EXACT_BYTES`]; hostile/oversized input falls back to the
//! conservative estimator labeled `UpperBound` — an upper bound is NEVER
//! relabeled `Exact`, and the fallback's over-count keeps budgeting safe.
//! The cache layer ([`TokenCache`](crate::TokenCache)) keys by
//! `(TokenizerId family + version, blake3(content))`, so repeated huge
//! content costs one hash and zero re-tokenization.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use faktor_provider::TokenizerId;

use crate::estimator::Estimator;
use crate::TokenEstimate;

/// Maximum input size (UTF-8 bytes) handed to a real BPE backend by
/// [`TiktokenTokenizer`]. 1 MiB is ≈ 250k–350k tokens — comfortably above
/// any single real request text the runtime sends — while keeping an exact
/// pass's transient allocation bounded on adversarial input. Inputs above
/// the cap are counted by the conservative generic estimator and labeled
/// [`TokenEstimateKind::UpperBound`](crate::TokenEstimateKind::UpperBound).
pub const MAX_EXACT_BYTES: usize = 1 << 20;

/// A local tokenizer implementation with a stable versioned identity.
///
/// `count` returns a [`TokenEstimate`] whose `kind` must be
/// [`TokenEstimateKind::Exact`](crate::TokenEstimateKind::Exact) ONLY when
/// a real tokenizer counted the text; an implementation that cannot count
/// exactly MUST fall back to a conservative estimator and label it
/// `UpperBound`.
pub trait Tokenizer: Send + Sync {
    /// The identity (family + version) this implementation implements.
    fn id(&self) -> TokenizerId;
    /// Count `text` exactly, or return a label-honest conservative bound.
    fn count(&self, text: &str) -> TokenEstimate;
}

/// Conservative fallback for every tokenizer identity without a registered
/// exact backend: the generic estimator's value, explicitly labeled
/// [`TokenEstimateKind::UpperBound`](crate::TokenEstimateKind::UpperBound).
pub struct ConservativeEstimatorTokenizer {
    id: TokenizerId,
}

impl ConservativeEstimatorTokenizer {
    pub fn new(id: TokenizerId) -> Self {
        Self { id }
    }
}

impl Tokenizer for ConservativeEstimatorTokenizer {
    fn id(&self) -> TokenizerId {
        self.id
    }

    fn count(&self, text: &str) -> TokenEstimate {
        TokenEstimate::upper_bound(estimator_count(text))
    }
}

/// The generic estimator's count widened to `u64` (saturating at the absurd
/// upper end). Never zero for non-empty input, never panics.
fn estimator_count(text: &str) -> u64 {
    u64::try_from(Estimator.estimate_tokens(text)).unwrap_or(u64::MAX)
}

/// Real local OpenAI BPE backend (`tiktoken-rs`): `o200k_base` or
/// `cl100k_base`, vocabulary embedded in the binary at compile time and
/// read from memory at runtime (fully offline).
pub struct TiktokenTokenizer {
    id: TokenizerId,
    max_exact_bytes: usize,
    bpe: tiktoken_rs::CoreBPE,
}

impl TiktokenTokenizer {
    /// Exact backend for the frozen v1 identities
    /// ([`TokenizerId::O200K_BASE`], [`TokenizerId::CL100K_BASE`]) with the
    /// default [`MAX_EXACT_BYTES`] cap. Any other family or version returns
    /// `None` (unknown identities must fall back conservatively — a version
    /// bump never silently reuses the old vocabulary's counts).
    pub fn new(id: TokenizerId) -> Option<Self> {
        Self::with_max_exact_bytes(id, MAX_EXACT_BYTES)
    }

    /// [`TiktokenTokenizer::new`] with an explicit exact-pass byte cap.
    /// Exposed so adversarial tests can exercise the cap boundary without
    /// multi-megabyte inputs.
    pub fn with_max_exact_bytes(id: TokenizerId, max_exact_bytes: usize) -> Option<Self> {
        let bpe = match id {
            TokenizerId::O200K_BASE => tiktoken_rs::o200k_base().ok()?,
            TokenizerId::CL100K_BASE => tiktoken_rs::cl100k_base().ok()?,
            _ => return None,
        };
        Some(Self {
            id,
            max_exact_bytes,
            bpe,
        })
    }
}

impl Tokenizer for TiktokenTokenizer {
    fn id(&self) -> TokenizerId {
        self.id
    }

    fn count(&self, text: &str) -> TokenEstimate {
        if text.len() > self.max_exact_bytes {
            // Oversized input: BPE would allocate tokens proportional to
            // the input. Return the conservative estimator labeled as what
            // it is — an upper bound, never an "exact" count.
            return TokenEstimate::upper_bound(estimator_count(text));
        }
        let tokens = self.bpe.encode_with_special_tokens(text);
        TokenEstimate::exact(u64::try_from(tokens.len()).unwrap_or(u64::MAX))
    }
}

/// Registry resolving a versioned [`TokenizerId`] to the local backend that
/// implements it. Unregistered identities resolve to `None` and their count
/// is the conservative estimator, labeled `UpperBound`.
#[derive(Default)]
pub struct TokenizerRegistry {
    tokenizers: HashMap<TokenizerId, Arc<dyn Tokenizer>>,
}

impl TokenizerRegistry {
    /// An empty registry: every identity falls back conservatively.
    pub fn new() -> Self {
        Self::default()
    }

    /// A registry with the real local backends registered: `o200k_base@v1`
    /// and `cl100k_base@v1` via [`TiktokenTokenizer`]. A backend that fails
    /// to initialize is simply left unregistered (its counts fall back,
    /// labeled `UpperBound` — construction never panics).
    pub fn with_builtin_backends() -> Self {
        let mut registry = Self::new();
        for id in [TokenizerId::O200K_BASE, TokenizerId::CL100K_BASE] {
            if let Some(tokenizer) = TiktokenTokenizer::new(id) {
                registry.register(Arc::new(tokenizer));
            }
        }
        registry
    }

    /// Register `tokenizer` under its [`Tokenizer::id`]. Replaces (and
    /// returns) any previous backend for the same identity — the versioned
    /// identity is the whole key, so a replacement never leaks old counts.
    pub fn register(&mut self, tokenizer: Arc<dyn Tokenizer>) -> Option<Arc<dyn Tokenizer>> {
        self.tokenizers.insert(tokenizer.id(), tokenizer)
    }

    /// The backend registered for `id`, if any.
    pub fn resolve(&self, id: TokenizerId) -> Option<Arc<dyn Tokenizer>> {
        self.tokenizers.get(&id).cloned()
    }

    /// Whether an exact backend is registered for `id`.
    pub fn is_registered(&self, id: TokenizerId) -> bool {
        self.tokenizers.contains_key(&id)
    }

    /// Count `text` under `id`: the registered exact backend when one
    /// exists, otherwise the conservative generic estimator labeled
    /// `UpperBound`.
    pub fn count(&self, id: TokenizerId, text: &str) -> TokenEstimate {
        match self.resolve(id) {
            Some(tokenizer) => tokenizer.count(text),
            None => ConservativeEstimatorTokenizer::new(id).count(text),
        }
    }

    /// Number of registered exact backends.
    pub fn len(&self) -> usize {
        self.tokenizers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokenizers.is_empty()
    }
}

/// The process-wide registry: built-in real backends, constructed once on
/// first use (the BPE vocabulary build is the only startup cost). Callers
/// that need an isolated registry (tests, future vocabulary extensions)
/// build their own with [`TokenizerRegistry::new`] +
/// [`TokenizerRegistry::register`].
pub fn global_registry() -> Arc<TokenizerRegistry> {
    static GLOBAL: OnceLock<Arc<TokenizerRegistry>> = OnceLock::new();
    GLOBAL
        .get_or_init(|| Arc::new(TokenizerRegistry::with_builtin_backends()))
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TokenEstimateKind;
    use faktor_provider::TokenFamily;

    const CODE_CORPUS: &str = r#"
fn main() {
    let cache = TokenCache::new();
    let text = "the quick brown fox jumps over the lazy dog";
    // deterministic corpus for the upper-bound honesty lock
    println!("total = {}", cache.count_for_model("gpt-5", text).count);
}
"#;

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

    #[test]
    fn builtin_backends_are_local_exact_and_family_distinct() {
        let registry = TokenizerRegistry::with_builtin_backends();
        assert_eq!(registry.len(), 2, "o200k + cl100k registered");
        assert!(registry.is_registered(TokenizerId::O200K_BASE));
        assert!(registry.is_registered(TokenizerId::CL100K_BASE));

        let o200k = registry.count(TokenizerId::O200K_BASE, CODE_CORPUS);
        let cl100k = registry.count(TokenizerId::CL100K_BASE, CODE_CORPUS);
        assert_eq!(o200k.kind, TokenEstimateKind::Exact);
        assert_eq!(cl100k.kind, TokenEstimateKind::Exact);
        assert!(
            o200k.count > 0 && cl100k.count > 0,
            "real BPE never reports zero for non-empty text"
        );
        // The two families are genuinely DIFFERENT vocabularies: a
        // tokenizer-registry bug that returned one shared count would be a
        // silent mis-accounting on every wire plan. No single short string
        // is guaranteed to disagree, so scan a small corpus.
        let corpus = [
            CODE_CORPUS,
            "hello world 12345",
            "fn x() { println!(\"hi\"); }",
            "汉字😀 mixed 123",
            "def calculate_total(items, tax_rate=0.2):",
        ];
        assert!(
            corpus.iter().any(|text| {
                registry.count(TokenizerId::O200K_BASE, text).count
                    != registry.count(TokenizerId::CL100K_BASE, text).count
            }),
            "o200k and cl100k must not be one shared count on the corpus"
        );

        // Empty input is an exact 0 under a real tokenizer.
        assert_eq!(
            registry.count(TokenizerId::O200K_BASE, ""),
            TokenEstimate::exact(0)
        );
        // Unicode/emoji never panics and stays exact.
        let unicode = registry.count(TokenizerId::O200K_BASE, "汉字😀 mixed 123");
        assert_eq!(unicode.kind, TokenEstimateKind::Exact);
        assert!(unicode.count > 0);
    }

    #[test]
    fn exact_counts_are_deterministic_across_independent_registries() {
        // Two independently built registries must agree byte-for-byte:
        // the BPE assets are frozen and there is no clock/rng/network input.
        let a = TokenizerRegistry::with_builtin_backends();
        let b = TokenizerRegistry::with_builtin_backends();
        for text in [
            "",
            "x",
            CODE_CORPUS,
            "fn main() { let x = 1; } // repeated repeated repeated",
            "汉字与😀混排的样本",
        ] {
            for id in [TokenizerId::O200K_BASE, TokenizerId::CL100K_BASE] {
                assert_eq!(
                    a.count(id, text),
                    b.count(id, text),
                    "{id} count drifted for {text:?}"
                );
            }
        }
    }

    #[test]
    fn unregistered_families_and_versions_are_conservative_upper_bounds() {
        let registry = TokenizerRegistry::with_builtin_backends();
        let text = "counted without a vocabulary";
        let want = TokenEstimate::upper_bound(estimator_count(text));
        for id in [
            TokenizerId::ANTHROPIC,
            TokenizerId::GEMINI,
            TokenizerId::LLAMA,
            TokenizerId::GENERIC_ESTIMATOR,
            TokenizerId {
                family: TokenFamily::O200kBase,
                version: 2, // not implemented: v1 vocab only
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
            assert!(!registry.is_registered(id), "{id} must not be registered");
            let got = registry.count(id, text);
            assert_eq!(
                got, want,
                "{id} must fall back to the conservative estimator"
            );
            assert_eq!(got.kind, TokenEstimateKind::UpperBound);
        }
    }

    #[test]
    fn oversized_input_is_capped_and_never_relabeled_exact() {
        // A tiny cap proves the boundary cheaply: at the cap the count is
        // exact; one byte over, the backend refuses the BPE pass and the
        // result is honestly labeled UpperBound (estimator value).
        let exact = TiktokenTokenizer::with_max_exact_bytes(TokenizerId::O200K_BASE, 64).unwrap();
        let at_cap = "a".repeat(64);
        let over_cap = "a".repeat(65);
        let within = exact.count(&at_cap);
        assert_eq!(within.kind, TokenEstimateKind::Exact);
        assert!(within.count > 0);
        let over = exact.count(&over_cap);
        assert_eq!(over.kind, TokenEstimateKind::UpperBound);
        assert_eq!(over.count, estimator_count(&over_cap));
        assert_ne!(over.kind, TokenEstimateKind::Exact);
    }

    #[test]
    fn replacing_a_backend_replaces_its_counts_keyed_by_identity() {
        let id = TokenizerId {
            family: TokenFamily::Llama,
            version: 7,
        };
        let mut registry = TokenizerRegistry::new();
        registry.register(Arc::new(FixtureTokenizer { id, divisor: 1 }));
        assert_eq!(registry.count(id, "abcdef").count, 6);
        // Same identity, new backend: the old backend is returned and the
        // new one owns the identity (registry replace is explicit).
        let old = registry.register(Arc::new(FixtureTokenizer { id, divisor: 2 }));
        assert!(old.is_some());
        assert_eq!(registry.count(id, "abcdef").count, 3);
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn conservative_estimator_tokenizer_is_never_exact() {
        for id in [TokenizerId::ANTHROPIC, TokenizerId::GEMINI] {
            let t = ConservativeEstimatorTokenizer::new(id);
            assert_eq!(t.id(), id);
            for text in ["", "x", "arbitrary text", "汉字😀"] {
                assert_eq!(t.count(text).kind, TokenEstimateKind::UpperBound);
                assert_eq!(t.count(text).count, estimator_count(text));
            }
        }
    }

    #[test]
    fn fallback_estimator_result_is_never_below_fixture_exact_on_corpus() {
        // Upper-bound honesty (audits 72/73): whatever the conservative
        // fallback reports for a corpus, a real exact count from the same
        // corpus must not exceed it. The fixture is a deterministic local
        // BPE-free "tokenizer" at 4 chars/token — strictly denser than the
        // estimator's chars/3+1 contract — so a fallback that undercounts
        // the fixture would fail here.
        let registry = TokenizerRegistry::with_builtin_backends();
        let id = TokenizerId {
            family: TokenFamily::Llama,
            version: 0x5EED,
        };
        let mut fixture_registry = TokenizerRegistry::new();
        fixture_registry.register(Arc::new(FixtureTokenizer { id, divisor: 4 }));
        for text in [
            CODE_CORPUS,
            "hello world, tokens please",
            "a b c d",
            &"dense".repeat(1000),
        ] {
            let exact = fixture_registry.count(id, text);
            assert_eq!(exact.kind, TokenEstimateKind::Exact);
            let fallback = registry.count(id, text);
            assert_eq!(fallback.kind, TokenEstimateKind::UpperBound);
            assert!(
                fallback.count >= exact.count,
                "fallback {} < exact {} for {text:?}",
                fallback.count,
                exact.count
            );
            assert_eq!(fallback.count, estimator_count(text));
        }
    }
}
