//! Model economics catalog (audit P0-1): one real [`ModelCatalogEntry`] per
//! provider/model, replacing the graph's fabricated
//! `ModelEconomics::default()` rows that routed every production model with
//! zero prices and 50/50/50 priors.
//!
//! The three pricing states are the audit's core distinction:
//!
//! - [`PricingState::Known`] — real price knowledge (adapter catalog row or
//!   user override). Carries the V1 [`ModelEconomics`] price fields the
//!   router reads.
//! - [`PricingState::LocalZero`] — a LOCAL runtime (Ollama): zero monetary
//!   cost is the measured truth, never a stand-in for "no price".
//! - [`PricingState::Unknown`] — no price knowledge. **Unknown is never
//!   mapped to zero and never to the 1-microUSD runtime fallback**; the
//!   routing graph excludes Unknown-priced REMOTE entries from the economy
//!   candidate set unless a conservative user-configured ceiling prices
//!   them ([`PricingOverrides`]).
//!
//! Every entry carries a `source_epoch` ([`ModelCatalogEntry::pricing_epoch`])
//! so settlement can reference WHICH catalog generation priced a model
//! (user overrides bump the epoch; provenance records why the price is what
//! it is).
//!
//! Priors: the V1 [`ModelEconomics`] carries reliability priors (u8 0..=100
//! per dimension, 50 = neutral). [`QualityPrior`] mirrors that field shape
//! as an inspectable catalog value; its default is the same conservative
//! generic prior the graph used before (so router-visible quality behavior
//! is unchanged) but it is now a named, documented default instead of an
//! invisible struct-literal side effect.

use std::sync::Arc;

use faktor_core::model::{MicroUsdPerToken, ModelCapabilities, ModelEconomics};

use super::Provider;

/// Epoch of the very first catalog row of any provider (built-in default
/// catalog entries and adapter-declared rows both start here).
pub const CATALOG_FIRST_EPOCH: u64 = 1;

/// Why a catalog entry carries the price/prior it does. Ranked so the
/// entry's derived [`Ord`] is stable across processes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum Provenance {
    /// Conservative defaults baked into the runtime (the trait's default
    /// [`Provider::catalog_entry`]): no provider data, no user config.
    BuiltIn,
    /// Declared by the provider adapter from real knowledge (e.g. Ollama
    /// rows are local-zero because the runtime IS local).
    ProviderCatalog,
    /// The user declared exact prices for this endpoint.
    UserOverride,
    /// The user's conservative ceiling priced a model whose adapter
    /// knowledge was [`PricingState::Unknown`]. The price is a budget
    /// ceiling, NOT a measured price.
    Composite,
}

/// Reliability/availability priors of one catalog row. Field shape and
/// semantics mirror the reliability fields of the V1
/// [`faktor_core::model::ModelEconomics`] the router consumes
/// (`coding_quality()`/`context_reliability`); priors live there today, so
/// this type is a catalog-side view of the same dimensions, not a new
/// invented prior model.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(default)]
pub struct QualityPrior {
    pub tool_reliability: u8,
    pub reasoning_reliability: u8,
    pub coding_reliability: u8,
    pub context_reliability: u8,
    pub availability: u8,
}

impl QualityPrior {
    /// The conservative generic prior: exactly the reliability/availability
    /// numbers the routing graph produced before catalogs existed
    /// (`ModelEconomics::default()` — neutral 50 reliability on every
    /// dimension, 100 availability). Exposed as a named constant so the
    /// default is inspectable and never re-invented per call site.
    pub const fn conservative_generic() -> Self {
        Self {
            tool_reliability: 50,
            reasoning_reliability: 50,
            coding_reliability: 50,
            context_reliability: 50,
            availability: 100,
        }
    }
}

impl Default for QualityPrior {
    fn default() -> Self {
        Self::conservative_generic()
    }
}

/// Price knowledge of one provider/model. **The states are NOT numeric
/// prices** — `Unknown` must never be flattened to 0 microUSD, and
/// `LocalZero` is a measured zero, not a missing price.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PricingState {
    /// Real price knowledge. Only the four price fields of the wrapped
    /// [`ModelEconomics`] are authoritative (`input/output/cache_read/
    /// cache_write`, microUSD per token); its reliability/availability
    /// fields are the neutral defaults and are ignored — priors live in
    /// [`ModelCatalogEntry::quality_prior`].
    Known(ModelEconomics),
    /// Local runtime with zero monetary cost (Ollama). The zero is
    /// measured truth: latency and reliability still count.
    LocalZero,
    /// No price knowledge. Never zero, never a fabricated 1-microUSD
    /// fallback; the graph excludes Unknown-priced remote candidates
    /// unless a configured ceiling prices them.
    Unknown,
}

impl PricingState {
    /// True only for a measured local-zero price (the router's
    /// local-cost marker semantics).
    pub fn is_local_zero(&self) -> bool {
        matches!(self, PricingState::LocalZero)
    }

    /// The authoritative price fields when the state is [`PricingState::Known`].
    pub fn known_prices(&self) -> Option<&ModelEconomics> {
        match self {
            PricingState::Known(e) => Some(e),
            PricingState::LocalZero | PricingState::Unknown => None,
        }
    }
}

/// One real catalog row: provider/model identity, capabilities, pricing
/// state, priors, and the epoch/provenance of the row's price.
///
/// [`Ord`] is a full-field deterministic total order (every field,
/// declaration order) so sorted entry lists are stable across processes
/// and runs; [`PricingState`] ranks `LocalZero < Known < Unknown` (a
/// fixed, documented order — it carries no policy meaning).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ModelCatalogEntry {
    /// Provider instance id the row belongs to (the registry key the
    /// daemon resolves; adapters fill their family id, instance wrapping
    /// rewrites it).
    pub provider: String,
    pub model: String,
    pub capabilities: ModelCapabilities,
    pub pricing: PricingState,
    pub quality_prior: QualityPrior,
    /// Generation of the price knowledge. `1` = first catalog row; user
    /// overrides bump it (`+1`) so settlement can tell WHICH generation
    /// priced a model. Never decreases.
    pub source_epoch: u64,
    pub provenance: Provenance,
}

impl ModelCatalogEntry {
    /// The pricing epoch of this row (the source generation its price
    /// came from) — the settlement reference for route decisions built
    /// over the row.
    pub const fn pricing_epoch(&self) -> u64 {
        self.source_epoch
    }

    /// The router-consumed V1 economics of this row: the four price fields
    /// from [`PricingState::Known`] (zero for `LocalZero`/`Unknown`), the
    /// latency when the Known row declares one, and the reliability priors
    /// from [`ModelCatalogEntry::quality_prior`]. `Unknown` rows map to
    /// zero prices ONLY through this lossy projection — callers that need
    /// the price state must match on [`PricingState`], never on the zeros.
    pub fn economics(&self) -> ModelEconomics {
        let mut e = ModelEconomics::default();
        if let Some(k) = self.pricing.known_prices() {
            e.input_price_per_mtok = k.input_price_per_mtok;
            e.output_price_per_mtok = k.output_price_per_mtok;
            e.cache_read_price_per_mtok = k.cache_read_price_per_mtok;
            e.cache_write_price_per_mtok = k.cache_write_price_per_mtok;
            if k.estimated_latency_ms != 0 {
                e.estimated_latency_ms = k.estimated_latency_ms;
            }
        }
        e.tool_reliability = self.quality_prior.tool_reliability;
        e.reasoning_reliability = self.quality_prior.reasoning_reliability;
        e.coding_reliability = self.quality_prior.coding_reliability;
        e.context_reliability = self.quality_prior.context_reliability;
        e.availability = self.quality_prior.availability;
        e
    }
}

// ------------------------------------------------------------- Ord (full field)

fn cmp_usize(a: usize, b: usize) -> std::cmp::Ordering {
    a.cmp(&b)
}

fn cmp_economics(a: &ModelEconomics, b: &ModelEconomics) -> std::cmp::Ordering {
    a.input_price_per_mtok
        .cmp(&b.input_price_per_mtok)
        .then_with(|| a.output_price_per_mtok.cmp(&b.output_price_per_mtok))
        .then_with(|| {
            a.cache_read_price_per_mtok
                .cmp(&b.cache_read_price_per_mtok)
        })
        .then_with(|| {
            a.cache_write_price_per_mtok
                .cmp(&b.cache_write_price_per_mtok)
        })
        .then_with(|| a.estimated_latency_ms.cmp(&b.estimated_latency_ms))
        .then_with(|| a.tool_reliability.cmp(&b.tool_reliability))
        .then_with(|| a.reasoning_reliability.cmp(&b.reasoning_reliability))
        .then_with(|| a.coding_reliability.cmp(&b.coding_reliability))
        .then_with(|| a.context_reliability.cmp(&b.context_reliability))
        .then_with(|| a.availability.cmp(&b.availability))
        .then_with(|| rate_rank(a.rate_limit_state).cmp(&rate_rank(b.rate_limit_state)))
}

fn rate_rank(s: faktor_core::model::RateLimitState) -> u8 {
    use faktor_core::model::RateLimitState as R;
    match s {
        R::Healthy => 0,
        R::Soft => 1,
        R::Hard => 2,
    }
}

fn cmp_caps(a: &ModelCapabilities, b: &ModelCapabilities) -> std::cmp::Ordering {
    cmp_usize(a.context, b.context)
        .then_with(|| cmp_usize(a.max_output, b.max_output))
        .then_with(|| a.tools.cmp(&b.tools))
        .then_with(|| a.parallel_tools.cmp(&b.parallel_tools))
        .then_with(|| a.thinking.cmp(&b.thinking))
        .then_with(|| a.vision.cmp(&b.vision))
        .then_with(|| a.json_schema.cmp(&b.json_schema))
        .then_with(|| a.streaming.cmp(&b.streaming))
        .then_with(|| a.embeddings.cmp(&b.embeddings))
        .then_with(|| a.reasoning.cmp(&b.reasoning))
}

/// Fixed, documented pricing-state rank: `LocalZero < Known < Unknown`.
fn pricing_rank(s: &PricingState) -> u8 {
    match s {
        PricingState::LocalZero => 0,
        PricingState::Known(_) => 1,
        PricingState::Unknown => 2,
    }
}

impl PartialOrd for PricingState {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PricingState {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        pricing_rank(self)
            .cmp(&pricing_rank(other))
            .then_with(|| match (self, other) {
                (PricingState::Known(a), PricingState::Known(b)) => cmp_economics(a, b),
                _ => std::cmp::Ordering::Equal,
            })
    }
}

impl PartialOrd for ModelCatalogEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for ModelCatalogEntry {}

impl Ord for ModelCatalogEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.provider
            .cmp(&other.provider)
            .then_with(|| self.model.cmp(&other.model))
            .then_with(|| cmp_caps(&self.capabilities, &other.capabilities))
            .then_with(|| self.pricing.cmp(&other.pricing))
            .then_with(|| self.quality_prior.cmp(&other.quality_prior))
            .then_with(|| self.source_epoch.cmp(&other.source_epoch))
            .then_with(|| self.provenance.cmp(&other.provenance))
    }
}

// ---------------------------------------------------------------- overrides

/// User-configured pricing policy for ONE provider instance (the
/// `[providers.<id>.pricing]` config surface). Applied per model through
/// [`PricingOverrideProvider`], which rewrites catalog entries:
///
/// - `exact` (all four price fields set) prices EVERY model of the
///   instance at the declared microUSD-per-token values
///   ([`Provenance::UserOverride`]) — for custom OpenAI-compatible
///   endpoints whose real prices the operator knows;
/// - `ceiling_micro_per_token` prices ONLY models the adapter itself
///   leaves [`PricingState::Unknown`], at the conservative ceiling on all
///   four fields ([`Provenance::Composite`]) — the ceiling is a budget
///   bound, never a measured price;
/// - [`PricingState::LocalZero`] models are never touched (a local runtime
///   does not start costing money because a config table exists);
/// - [`PricingState::Known`] models with only a ceiling configured keep
///   their real price.
///
/// Both overrides bump `source_epoch` by one so settlement can tell that
/// the price generation changed.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PricingOverrides {
    /// Exact per-token prices for every model of the instance. When set,
    /// ALL four price fields are authoritative (callers validate before
    /// constructing: input/output must be >= 1 microUSD — a zero input
    /// price on a remote endpoint is the local-zero lie the audit kills).
    pub exact: Option<ModelEconomics>,
    /// Conservative ceiling in microUSD per token, applied to every price
    /// field of Unknown-priced models only.
    pub ceiling_micro_per_token: Option<u64>,
}

impl PricingOverrides {
    /// Ceiling economics: every price field at the ceiling.
    pub fn ceiling_economics(&self) -> Option<ModelEconomics> {
        self.ceiling_micro_per_token.map(|c| ModelEconomics {
            input_price_per_mtok: MicroUsdPerToken(c),
            output_price_per_mtok: MicroUsdPerToken(c),
            cache_read_price_per_mtok: MicroUsdPerToken(c),
            cache_write_price_per_mtok: MicroUsdPerToken(c),
            ..Default::default()
        })
    }

    /// Apply this policy to one adapter-produced entry.
    pub fn apply(&self, entry: ModelCatalogEntry) -> ModelCatalogEntry {
        if entry.pricing.is_local_zero() {
            // LocalZero is measured truth; overrides never touch it.
            return entry;
        }
        if let Some(exact) = self.exact {
            return ModelCatalogEntry {
                pricing: PricingState::Known(exact),
                source_epoch: entry.source_epoch.saturating_add(1),
                provenance: Provenance::UserOverride,
                ..entry
            };
        }
        if entry.pricing == PricingState::Unknown {
            if let Some(ceiling) = self.ceiling_economics() {
                return ModelCatalogEntry {
                    pricing: PricingState::Known(ceiling),
                    source_epoch: entry.source_epoch.saturating_add(1),
                    provenance: Provenance::Composite,
                    ..entry
                };
            }
        }
        entry
    }
}

/// An instance-wrapped provider whose [`Provider::catalog_entry`] results
/// run through a [`PricingOverrides`] policy. Identity/capabilities/
/// streaming delegate to the wrapped adapter; only the catalog rows are
/// rewritten (same role `InstanceProvider` plays for instance identity).
pub struct PricingOverrideProvider {
    inner: Arc<dyn Provider>,
    instance_id: String,
    overrides: PricingOverrides,
}

impl PricingOverrideProvider {
    pub fn wrap(
        inner: Arc<dyn Provider>,
        instance_id: impl Into<String>,
        overrides: PricingOverrides,
    ) -> Arc<dyn Provider> {
        Arc::new(Self {
            inner,
            instance_id: instance_id.into(),
            overrides,
        })
    }
}

impl Provider for PricingOverrideProvider {
    fn id(&self) -> &str {
        self.inner.id()
    }

    fn identity(&self) -> super::ProviderIdentity {
        super::ProviderIdentity::new(self.instance_id.clone(), self.id())
    }

    fn capabilities(&self, model: &str) -> ModelCapabilities {
        self.inner.capabilities(model)
    }

    fn known_models(&self) -> Vec<String> {
        self.inner.known_models()
    }

    fn runtime_context_limit(&self, model: &str) -> Option<usize> {
        self.inner.runtime_context_limit(model)
    }

    fn catalog_entry(&self, model: &str) -> ModelCatalogEntry {
        let mut entry = self.inner.catalog_entry(model);
        entry.provider = self.instance_id.clone();
        self.overrides.apply(entry)
    }

    fn stream(&self, req: super::GenericAgentRequest) -> super::ProviderStream {
        self.inner.stream(req)
    }
}

// ------------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn entry(provider: &str, model: &str) -> ModelCatalogEntry {
        ModelCatalogEntry {
            provider: provider.into(),
            model: model.into(),
            capabilities: ModelCapabilities::small_local(),
            pricing: PricingState::Unknown,
            quality_prior: QualityPrior::default(),
            source_epoch: CATALOG_FIRST_EPOCH,
            provenance: Provenance::BuiltIn,
        }
    }

    fn priced(entry: &mut ModelCatalogEntry, input: u64, output: u64) {
        entry.pricing = PricingState::Known(ModelEconomics {
            input_price_per_mtok: MicroUsdPerToken::from_dollars_per_million(input),
            output_price_per_mtok: MicroUsdPerToken::from_dollars_per_million(output),
            cache_read_price_per_mtok: MicroUsdPerToken::from_dollars_per_million(input / 5),
            cache_write_price_per_mtok: MicroUsdPerToken::from_dollars_per_million(input / 2),
            estimated_latency_ms: 800,
            ..Default::default()
        });
    }

    #[test]
    fn conservative_generic_prior_matches_the_v1_default_semantics() {
        // The graph used to build descriptors from ModelEconomics::default()
        // (neutral 50 reliability everywhere, 100 availability). The catalog
        // exposes that SAME prior as a named inspectable default — router
        // quality behavior of an unpriced candidate is unchanged.
        let p = QualityPrior::default();
        assert_eq!(p.tool_reliability, 50);
        assert_eq!(p.reasoning_reliability, 50);
        assert_eq!(p.coding_reliability, 50);
        assert_eq!(p.context_reliability, 50);
        assert_eq!(p.availability, 100);
        assert_eq!(p, QualityPrior::conservative_generic());
        let e = ModelEconomics::default();
        assert_eq!(p.coding_reliability, e.coding_reliability);
        assert_eq!(p.context_reliability, e.context_reliability);
        assert_eq!(p.availability, e.availability);
    }

    #[test]
    fn unknown_is_a_distinct_state_never_a_zero_or_one_micro_fake() {
        // The audit core: Unknown != zero and != the 1-microUSD runtime
        // fallback. The lossy economics projection may read zero, but the
        // STATE must stay observable.
        let e = entry("openai", "gpt-5");
        assert_eq!(e.pricing, PricingState::Unknown);
        assert!(!e.pricing.is_local_zero());
        assert!(e.pricing.known_prices().is_none());
        assert_eq!(e.pricing_epoch(), CATALOG_FIRST_EPOCH);
        // LocalZero is the ONLY zero-cost state.
        let local = ModelCatalogEntry {
            pricing: PricingState::LocalZero,
            ..entry("ollama", "qwen3.8")
        };
        assert!(local.pricing.is_local_zero());
        let known = ModelCatalogEntry {
            pricing: PricingState::Known(ModelEconomics {
                output_price_per_mtok: MicroUsdPerToken(1),
                ..Default::default()
            }),
            ..entry("openai", "gpt-5")
        };
        assert!(!known.pricing.is_local_zero());
        assert!(!known.pricing.known_prices().is_none());
    }

    #[test]
    fn economics_projection_merges_prices_priors_and_latency() {
        let mut k = entry("openai", "gpt-5");
        priced(&mut k, 15, 60);
        k.quality_prior = QualityPrior {
            tool_reliability: 90,
            reasoning_reliability: 91,
            coding_reliability: 92,
            context_reliability: 93,
            availability: 99,
        };
        let e = k.economics();
        assert_eq!(e.input_price_per_mtok, MicroUsdPerToken(15));
        assert_eq!(e.output_price_per_mtok, MicroUsdPerToken(60));
        assert_eq!(e.cache_read_price_per_mtok, MicroUsdPerToken(3));
        assert_eq!(e.cache_write_price_per_mtok, MicroUsdPerToken(7));
        assert_eq!(e.estimated_latency_ms, 800);
        assert_eq!(e.tool_reliability, 90);
        assert_eq!(e.coding_reliability, 92);
        assert_eq!(e.context_reliability, 93);
        assert_eq!(e.availability, 99);
        assert!(!e.is_local_zero_cost());
        // Known rows whose latency field is unset (0) fall back to the
        // default latency, never to a zero-latency lie.
        let mut no_latency = entry("openai", "gpt-5");
        no_latency.pricing = PricingState::Known(ModelEconomics {
            input_price_per_mtok: MicroUsdPerToken(15),
            ..Default::default()
        });
        assert_eq!(no_latency.economics().estimated_latency_ms, 1000);
        // LocalZero rows project to the local marker the router knows.
        let local = ModelCatalogEntry {
            pricing: PricingState::LocalZero,
            ..entry("ollama", "qwen3.8")
        };
        let e = local.economics();
        assert!(e.is_local_zero_cost());
        assert_eq!(e.availability, 100);
    }

    #[test]
    fn serde_roundtrip_is_total_and_stable() {
        let mut a = entry("openai", "gpt-5");
        priced(&mut a, 15, 60);
        let mut b = entry("ollama", "qwen3.8");
        b.pricing = PricingState::LocalZero;
        b.provenance = Provenance::ProviderCatalog;
        let mut c = entry("corp-proxy", "my-model");
        c.source_epoch = 2;
        c.provenance = Provenance::UserOverride;
        let mut d = entry("corp-proxy", "other-model");
        d.source_epoch = 2;
        d.provenance = Provenance::Composite;
        for e in [a.clone(), b.clone(), c.clone(), d.clone()] {
            let v = serde_json::to_value(&e).unwrap();
            let back: ModelCatalogEntry = serde_json::from_value(v).unwrap();
            assert_eq!(back, e);
        }
        // Wire names are frozen snake_case.
        let v = serde_json::to_value(&b).unwrap();
        assert_eq!(v["pricing"], serde_json::json!("local_zero"));
        assert_eq!(v["provenance"], serde_json::json!("provider_catalog"));
        // Hostile unknown fields are ignored (never fatal); hostile enum
        // values are rejected.
        let mut hostile = serde_json::to_value(&c).unwrap();
        hostile["intruder"] = serde_json::json!("x");
        assert!(serde_json::from_value::<ModelCatalogEntry>(hostile).is_ok());
        let bad = serde_json::json!({"pricing": "free_beer"});
        assert!(serde_json::from_value::<PricingState>(bad).is_err());
    }

    #[test]
    fn ord_is_full_field_deterministic_and_consistent_with_eq() {
        // Identical entries compare Equal; sorting is stable across runs.
        let mut a = entry("openai", "gpt-5");
        priced(&mut a, 15, 60);
        let mut a2 = a.clone();
        a2.source_epoch += 1;
        let mut local = entry("ollama", "qwen3.8");
        local.pricing = PricingState::LocalZero;
        let mut unknown = entry("openai", "gpt-5-mini");
        unknown.pricing = PricingState::Unknown;
        assert_eq!(a.cmp(&a), std::cmp::Ordering::Equal);
        assert_eq!(a.partial_cmp(&a2), Some(std::cmp::Ordering::Less));
        // Every pair decides (total order) and is antisymmetric: however
        // the vector is seeded, both stable and unstable sorts converge on
        // ONE canonical order (identical entries cannot diverge: cmp == Eq).
        let mut v = vec![a.clone(), a2.clone(), local.clone(), unknown.clone()];
        v.sort();
        for _ in 0..3 {
            let mut shuffled = v.clone();
            shuffled.reverse();
            shuffled.sort();
            assert_eq!(shuffled, v, "sort order is deterministic");
        }
        let mut unstable = v.clone();
        unstable.sort_unstable();
        assert_eq!(unstable, v, "no equal-key collisions in the sample");
        assert!(PricingState::LocalZero < PricingState::Unknown);
        // Ord and Eq agree on the pairwise results (contract).
        for x in &v {
            for y in &v {
                let ord_eq = x.cmp(y) == std::cmp::Ordering::Equal;
                let eq = x == y;
                assert_eq!(ord_eq, eq, "Ord must agree with Eq for {x:?} vs {y:?}");
            }
        }
    }

    #[test]
    fn pricing_rank_is_documented_and_stable() {
        let econ = ModelEconomics {
            input_price_per_mtok: MicroUsdPerToken(15),
            ..Default::default()
        };
        assert!(PricingState::LocalZero < PricingState::Known(econ));
        assert!(PricingState::Known(econ) < PricingState::Unknown);
        let json = serde_json::to_string(&[
            PricingState::LocalZero,
            PricingState::Known(ModelEconomics::default()),
            PricingState::Unknown,
        ])
        .unwrap();
        let back: Vec<PricingState> = serde_json::from_str(&json).unwrap();
        assert_eq!(back.len(), 3);
    }

    #[test]
    fn overrides_never_touch_local_zero_and_bump_epoch_once() {
        let local = ModelCatalogEntry {
            pricing: PricingState::LocalZero,
            source_epoch: CATALOG_FIRST_EPOCH,
            ..entry("ollama", "qwen3.8")
        };
        let overrides = PricingOverrides {
            exact: Some(ModelEconomics {
                input_price_per_mtok: MicroUsdPerToken(15),
                output_price_per_mtok: MicroUsdPerToken(60),
                ..Default::default()
            }),
            ceiling_micro_per_token: Some(50),
        };
        assert_eq!(overrides.apply(local.clone()), local, "locals are exempt");
        // LocalZero stays local even with only a ceiling configured.
        let ceiling_only = PricingOverrides {
            exact: None,
            ceiling_micro_per_token: Some(50),
        };
        assert_eq!(ceiling_only.apply(local.clone()), local);
    }

    #[test]
    fn user_override_prices_every_model_and_epoch_increments() {
        let overrides = PricingOverrides {
            exact: Some(ModelEconomics {
                input_price_per_mtok: MicroUsdPerToken::from_dollars_per_million(2),
                output_price_per_mtok: MicroUsdPerToken::from_dollars_per_million(8),
                cache_read_price_per_mtok: MicroUsdPerToken(0),
                cache_write_price_per_mtok: MicroUsdPerToken(0),
                ..Default::default()
            }),
            ceiling_micro_per_token: None,
        };
        for base in [
            PricingState::Unknown,
            PricingState::Known(ModelEconomics::default()),
        ] {
            let before = ModelCatalogEntry {
                pricing: base,
                ..entry("corp-proxy", "m")
            };
            let after = overrides.apply(before.clone());
            assert_eq!(after.provenance, Provenance::UserOverride);
            assert_eq!(
                after.source_epoch,
                before.source_epoch + 1,
                "override bumps the pricing epoch"
            );
            match after.pricing {
                PricingState::Known(e) => {
                    assert_eq!(e.input_price_per_mtok, MicroUsdPerToken(2));
                    assert_eq!(e.output_price_per_mtok, MicroUsdPerToken(8));
                }
                other => panic!("override must produce Known, got {other:?}"),
            }
        }
    }

    #[test]
    fn ceiling_prices_only_unknown_models_as_composite() {
        let ceiling = PricingOverrides {
            exact: None,
            ceiling_micro_per_token: Some(42),
        };
        // Unknown -> Known at exactly the ceiling on every price field,
        // provenance Composite, epoch bumped.
        let before = entry("corp-proxy", "renamed-model");
        let after = ceiling.apply(before);
        assert_eq!(after.provenance, Provenance::Composite);
        assert_eq!(after.source_epoch, CATALOG_FIRST_EPOCH + 1);
        match after.pricing {
            PricingState::Known(e) => {
                for p in [
                    e.input_price_per_mtok,
                    e.output_price_per_mtok,
                    e.cache_read_price_per_mtok,
                    e.cache_write_price_per_mtok,
                ] {
                    assert_eq!(p, MicroUsdPerToken(42), "priced at exactly the ceiling");
                }
            }
            other => panic!("ceiling must produce Known, got {other:?}"),
        }
        // Known models keep their REAL price under a ceiling.
        let mut known = entry("corp-proxy", "known-model");
        priced(&mut known, 7, 21);
        let after = ceiling.apply(known.clone());
        assert_eq!(after, known);
        assert_eq!(after.provenance, Provenance::BuiltIn);
        assert_eq!(after.source_epoch, CATALOG_FIRST_EPOCH);
    }

    /// Legacy provider double with NO catalog override: exercises the
    /// trait default implementation (which provider tests must not break).
    #[derive(Clone)]
    struct LegacyTestProvider {
        id: String,
    }

    impl Provider for LegacyTestProvider {
        fn id(&self) -> &str {
            &self.id
        }

        fn capabilities(&self, _model: &str) -> ModelCapabilities {
            ModelCapabilities::small_local()
        }

        fn known_models(&self) -> Vec<String> {
            vec!["default".into(), "probed-a".into()]
        }

        fn stream(&self, _req: super::super::GenericAgentRequest) -> super::super::ProviderStream {
            Box::pin(futures::stream::empty())
        }
    }

    #[test]
    fn default_catalog_entries_are_unknown_builtin_first_epoch() {
        // Legacy adapters compile against the default impl and must NOT
        // silently read as zero-priced: Unknown + BuiltIn + epoch 1.
        let p = LegacyTestProvider {
            id: "legacy".into(),
        };
        for model in p.known_models() {
            let e = p.catalog_entry(&model);
            assert_eq!(e.provider, "legacy");
            assert_eq!(e.pricing, PricingState::Unknown, "{model} must be Unknown");
            assert_eq!(e.provenance, Provenance::BuiltIn);
            assert_eq!(e.pricing_epoch(), CATALOG_FIRST_EPOCH);
            assert_eq!(e.quality_prior, QualityPrior::conservative_generic());
            assert_eq!(e.capabilities, p.capabilities(&model));
        }
        // The default entry economics read zero ONLY through the lossy
        // projection; the state stays Unknown.
        let e = p.catalog_entry("probed-a");
        assert!(e.economics().is_local_zero_cost());
        assert_ne!(e.pricing, PricingState::LocalZero);
    }

    #[test]
    fn override_wrapper_delegates_identity_and_rewrites_provider() {
        use super::super::ProviderRegistry;
        let inner = Arc::new(LegacyTestProvider {
            id: "openai".into(),
        });
        let wrapped = PricingOverrideProvider::wrap(
            inner,
            "corp-proxy",
            PricingOverrides {
                exact: None,
                ceiling_micro_per_token: Some(99),
            },
        );
        assert_eq!(wrapped.id(), "openai", "family id for capability queries");
        assert_eq!(
            wrapped.identity(),
            super::super::ProviderIdentity::new("corp-proxy", "openai")
        );
        assert_eq!(
            wrapped.known_models(),
            vec!["default".to_string(), "probed-a".to_string()],
            "known_models delegates through the wrapper"
        );
        let e = wrapped.catalog_entry("probed-a");
        assert_eq!(e.provider, "corp-proxy", "rows name the instance id");
        assert_eq!(e.provenance, Provenance::Composite);
        assert_eq!(e.source_epoch, CATALOG_FIRST_EPOCH + 1);

        let mut reg = ProviderRegistry::new();
        reg.try_register(wrapped).unwrap();
        assert_eq!(reg.ids(), vec!["corp-proxy"]);
        assert_eq!(reg.get("corp-proxy").unwrap().id(), "openai");
    }
}
