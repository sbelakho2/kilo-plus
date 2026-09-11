//! Model economics catalog (audit wave-B item C): one real
//! [`ModelCatalogEntry`] per provider/model, replacing the graph's
//! fabricated `ModelEconomics::default()` rows that routed every production
//! model with zero prices and 50/50/50 priors.
//!
//! The pricing states are the audit's core distinction, and they NEVER
//! flatten through a lossy economics projection (the wave-A
//! `entry.economics()` projection is gone):
//!
//! - [`PricingState::Known`] — real, exact price knowledge (built-in list
//!   price table or user-exact override). Wraps the full
//!   [`PricingSnapshot`] (per-million-token quote, `Exact` authority,
//!   epoch + source id).
//! - [`PricingState::ConservativeCeiling`] — the user's conservative
//!   budget ceiling priced a model whose adapter knowledge was Unknown.
//!   Same snapshot shape with `ConservativeCeiling` authority: the quote
//!   is a budget bound, never a measured price.
//! - [`PricingState::LocalZero`] — a LOCAL runtime (Ollama): zero monetary
//!   cost is the measured truth. Unit-variant (the Ollama adapter
//!   constructs it); the row's own `source_epoch`/`provenance` are its
//!   epoch/source identity and [`ModelCatalogEntry::pricing_snapshot`]
//!   materializes the authoritative zero snapshot.
//! - [`PricingState::Unknown`] — no price knowledge. **Unknown is never
//!   mapped to zero and never to a 1-microUSD runtime fallback**; the
//!   routing graph excludes Unknown-priced REMOTE entries from economy
//!   candidate sets unless admission says otherwise
//!   ([`admissible`], configured ceilings, or the pin itself).
//! - [`PricingState::Stale`] — last-known prices with the wall-clock
//!   observation that aged them; admission and candidate building treat a
//!   stale row by its last-known authority.
//!
//! Every entry carries a `source_epoch` ([`ModelCatalogEntry::pricing_epoch`])
//! so settlement can reference WHICH catalog generation priced a model
//! (user overrides bump the epoch; provenance records why the price is what
//! it is). [`PricingProvenance`] names the source/catalog-version/clock of
//! a price statement.
//!
//! Priors: [`QualityPrior`] carries the reliability/availability priors
//! (u8 0..=100 per dimension, 50 = neutral) as an inspectable catalog
//! value; its default is the same conservative generic prior the graph
//! used before (so router-visible quality behavior is unchanged) but it is
//! now a named, documented default instead of an invisible struct-literal
//! side effect. Price and performance never share one blob here: rows keep
//! their per-million-token [`PriceQuote`] (exact, sub-$1/M included) in the
//! pricing state, and reliability/latency live in `quality_prior` + the
//! core [`faktor_core::model::ModelPerformance`] shape.

use std::sync::Arc;

use faktor_core::model::{
    BillingOrigin, EffectivePriceState, MicroUsdPerMillionTokens, ModelCapabilities,
    ModelPerformance, PriceAuthority, PriceQuote, PricingSnapshot, RoutingMode,
};

use crate::{GenericAgentRequest, Provider, ProviderIdentity, ProviderStream};

pub mod builtin;

/// The router's priced unit state (pricing-path audit): the pricing-state
/// type MOVED to `faktor_core::model` so router candidates can carry it,
/// and is re-exported here unchanged for every catalog-facing caller.
pub use faktor_core::model::PricingState;

/// Epoch of the very first catalog row of any provider (built-in default
/// catalog entries and adapter-declared rows both start here).
pub const CATALOG_FIRST_EPOCH: u64 = 1;

/// Source id of the built-in list-price table (audit item C: every built-in
/// row's snapshot names `"faktor-builtin-v1"`).
pub use builtin::BUILTIN_SOURCE_ID;

/// Source id stamped onto user-override snapshots (exact tables AND
/// conservative ceilings; the entry's [`Provenance`] still distinguishes
/// `UserOverride` from `Composite`).
pub const USER_OVERRIDE_SOURCE_ID: &str = "faktor-user-override-v1";

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
/// semantics mirror the reliability fields of the legacy V1
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
    /// Estimated one-call latency (ms); the conservative default is the
    /// legacy 1000 ms, built-in performance priors carry their documented
    /// Faktor estimate.
    pub estimated_latency_ms: u64,
}

impl QualityPrior {
    /// The conservative generic prior: exactly the reliability/availability
    /// numbers the routing graph produced before catalogs existed
    /// (`ModelEconomics::default()` — neutral 50 reliability on every
    /// dimension, 100 availability, 1000 ms). Exposed as a named constant
    /// so the default is inspectable and never re-invented per call site.
    pub const fn conservative_generic() -> Self {
        Self {
            tool_reliability: 50,
            reasoning_reliability: 50,
            coding_reliability: 50,
            context_reliability: 50,
            availability: 100,
            estimated_latency_ms: 1000,
        }
    }

    /// The non-monetary performance projection of this prior.
    pub fn performance(&self) -> ModelPerformance {
        ModelPerformance {
            context_reliability: self.context_reliability,
            coding_reliability: self.coding_reliability,
            estimated_latency_ms: self.estimated_latency_ms,
            rate_limit_state: faktor_core::model::RateLimitState::Healthy,
        }
    }
}

impl Default for QualityPrior {
    fn default() -> Self {
        Self::conservative_generic()
    }
}

/// Provenance of ONE price statement (audit item C): who priced it, when it
/// took effect, when it was observed, until when it stays authoritative, and
/// which catalog version produced it. The versioned built-in table stamps
/// `source_id = "faktor-builtin-v2"`, `catalog_version = "builtin-v2"` with
/// its row/table dates ([`builtin::provenance_of`]). `None` means the clock
/// is unknown (legacy rows); it is NEVER 0-as-unknown — 0 is a real instant
/// (1970-01-01). Settlement-relevant identity (epoch/source id) also rides
/// the route-time [`PricingSnapshot`]; the validity window rides it too so
/// routing can derive the effective price state at route time.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PricingProvenance {
    pub source_id: String,
    /// Wall-clock ms when the price took effect (`None` = unknown).
    pub effective_at_ms: Option<u64>,
    /// Wall-clock ms when the price was observed (`None` = never observed
    /// on the wire; static tables and user configs).
    pub observed_at_ms: Option<u64>,
    /// Wall-clock ms after which the price stops being authoritative
    /// (`None` = no published expiry).
    pub valid_until_ms: Option<u64>,
    /// Version of the pricing catalog that produced the statement
    /// (`"builtin-v2"` today; user overrides carry `"user-v1"`).
    pub catalog_version: String,
}

/// One real catalog row: provider/model identity, capabilities, pricing
/// state, priors, and the epoch/provenance of the row's price.
///
/// [`Ord`] is a full-field deterministic total order (every field,
/// declaration order) so sorted entry lists are stable across processes
/// and runs; [`PricingState`] ranks `LocalZero < Known < ConservativeCeiling
/// < Unknown < Stale` (a fixed, documented order — it carries no policy
/// meaning).
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

    /// Static tokenizer identity of this row's model (P0-81, audits 72/73;
    /// additive mapping, no new stored field): the same pure model →
    /// tokenizer mapping the wire accounting uses
    /// ([`crate::tokenizer_for`]), with the row's provider as the
    /// deployment hint. Never probes the provider, never consults a remote
    /// tokenizer API; a model without a known family maps to the
    /// conservative generic estimator.
    pub fn tokenizer_id(&self) -> crate::TokenizerId {
        crate::tokenizer_for(&self.model, Some(&self.provider))
    }

    /// The route-time [`PricingSnapshot`] this row's price knowledge cuts
    /// (audit item B/C — the ONLY projection out of the catalog, and it is
    /// not lossy): `Known`/`ConservativeCeiling` rows forward their frozen
    /// snapshot (authority from the VARIANT, quote/epoch/source id from the
    /// snapshot); `LocalZero` rows materialize the authoritative zero
    /// snapshot (`quote: Some(PriceQuote::ZERO)`, epoch = the row's
    /// `source_epoch`, source = the row's source id); `Unknown` rows
    /// materialize the no-quote Unknown snapshot — settlement refuses
    /// every fabricated number; `Stale` rows forward their last-known
    /// snapshot.
    pub fn pricing_snapshot(&self) -> PricingSnapshot {
        match &self.pricing {
            PricingState::Known(s) => PricingSnapshot {
                quote: s.quote,
                authority: PriceAuthority::Exact,
                epoch: s.epoch,
                source_id: s.source_id.clone(),
                valid_until_ms: s.valid_until_ms,
                conservative_ceiling: s.conservative_ceiling,
            },
            PricingState::ConservativeCeiling(s) => PricingSnapshot {
                quote: s.quote,
                authority: PriceAuthority::ConservativeCeiling,
                epoch: s.epoch,
                source_id: s.source_id.clone(),
                valid_until_ms: s.valid_until_ms,
                conservative_ceiling: None,
            },
            PricingState::Stale { last_known, .. } => last_known.clone(),
            PricingState::LocalZero => {
                PricingSnapshot::local_zero(self.source_epoch, self.source_id())
            }
            PricingState::Unknown => PricingSnapshot::unknown(self.source_epoch, self.source_id()),
        }
    }

    /// The source id a row without its own snapshot names: the built-in
    /// table for [`Provenance::BuiltIn`] rows, the provider instance id
    /// otherwise (adapter-declared local/unknown knowledge IS the
    /// instance).
    fn source_id(&self) -> String {
        match self.provenance {
            Provenance::BuiltIn => BUILTIN_SOURCE_ID.to_string(),
            _ => self.provider.clone(),
        }
    }
}

// ------------------------------------------------------------- Ord (full field)

fn cmp_usize(a: usize, b: usize) -> std::cmp::Ordering {
    a.cmp(&b)
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

// ------------------------------------------------------------- admission

/// Admission policy of one pricing state under one routing mode (audit
/// item C matrix). The decision is: MAY this (provider, model) row join the
/// router candidate set / be routed at all?
///
/// ```text
/// authority of the state (Exact / Ceiling / LocalZero / Unknown)
///
/// mode          | Exact | Ceiling | LocalZero | Unknown
/// --------------+-------+---------+-----------+-------------------
/// Economy       | admit | admit   | admit     | REFUSE (never a
///               |       |         |           | fabricated 0/1-micro
///               |       |         |           | price)
/// Balanced      | admit | admit   | admit     | admit ONLY when
///               |       |         |           | allow_unknown_balanced
///               |       |         |           | AND no hard cost cap
/// MaximumQuality| admit | admit   | admit     | admit with NO hard cost
///               |       |         |           | cap (quality decides;
///               |       |         |           | the unknown spend is
///               |       |         |           | recorded as Unknown);
///               |       |         |           | refuse under a hard cap
///               |       |         |           | (no honest bound)
/// Pinned        | admit | admit   | admit     | admit only without a
///               |       |         |           | hard cost cap (with a
///               |       |         |           | cap an unknown price
///               |       |         |           | cannot reserve honestly
///               |       |         |           | -> fail closed)
/// ```
///
/// `hard_cost_cap` = the task carries a durable monetary cap: priced rows
/// (Exact/Ceiling/LocalZero) reserve honestly against their quote; an
/// Unknown row has no bound and fails closed. `allow_unknown_balanced` is
/// the operator's explicit opt-in for unknown-priced models in Balanced
/// mode WITHOUT a hard cap (spend then settles as a documented Unknown
/// amount). A [`PricingState::Stale`] row is judged by its last-known
/// authority. Unknown NEVER becomes a numeric zero in any admitted row.
pub fn admissible(
    mode: &RoutingMode,
    state: &PricingState,
    hard_cost_cap: bool,
    allow_unknown_balanced: bool,
) -> bool {
    admissible_authority(
        mode,
        state.authority(),
        hard_cost_cap,
        allow_unknown_balanced,
    )
}

/// The ROUTE-TIME admission twin of [`admissible`]: judges an
/// [`EffectivePriceState`] derived at route (fresh Known → Exact, expired
/// Known → Unknown unless a documented ceiling exists, Stale → Unknown,
/// ConservativeCeiling → Conservative). Candidate sets MUST be filtered
/// through this form so an expired/stale exact row cannot be admitted as if
/// its old price were current.
pub fn admissible_effective(
    mode: &RoutingMode,
    state: &EffectivePriceState,
    hard_cost_cap: bool,
    allow_unknown_balanced: bool,
) -> bool {
    admissible_authority(
        mode,
        state.authority(),
        hard_cost_cap,
        allow_unknown_balanced,
    )
}

fn admissible_authority(
    mode: &RoutingMode,
    authority: PriceAuthority,
    hard_cost_cap: bool,
    allow_unknown_balanced: bool,
) -> bool {
    match mode {
        RoutingMode::Pinned { .. } => match authority {
            PriceAuthority::Unknown => !hard_cost_cap,
            _ => true,
        },
        RoutingMode::Balanced => match authority {
            PriceAuthority::Unknown => allow_unknown_balanced && !hard_cost_cap,
            _ => true,
        },
        // Economy refuses Unknown outright (cost minimization cannot price
        // it). MaximumQuality maximizes verified quality subject to the
        // HARD caps: with no hard cost cap an unknown-priced model is
        // admitted (its spend settles as documented Unknown), under a hard
        // cap it fails closed (no honest numeric bound).
        RoutingMode::Economy => !matches!(authority, PriceAuthority::Unknown),
        RoutingMode::MaximumQuality => match authority {
            PriceAuthority::Unknown => !hard_cost_cap,
            _ => true,
        },
    }
}

// ---------------------------------------------------------------- overrides

/// User-configured pricing policy for ONE provider instance (the
/// `[providers.<id>.pricing]` config surface). Applied per model through
/// [`PricingOverrideProvider`], which rewrites catalog entries:
///
/// - `exact` (a full per-million-token [`PriceQuote`]) prices EVERY model
///   of the instance at the declared values (state
///   [`PricingState::Known`], provenance [`Provenance::UserOverride`]) —
///   for custom OpenAI-compatible endpoints whose real prices the operator
///   knows;
/// - `ceiling_micro_usd_per_million_tokens` prices ONLY models the adapter
///   itself leaves [`PricingState::Unknown`], at the conservative ceiling
///   on every price line (state
///   [`PricingState::ConservativeCeiling`], provenance
///   [`Provenance::Composite`]) — the ceiling is a budget bound, never a
///   measured price;
/// - [`PricingState::LocalZero`] models are never touched (a local runtime
///   does not start costing money because a config table exists);
/// - [`PricingState::Known`]/[`PricingState::ConservativeCeiling`] models
///   with only a ceiling configured keep their real price.
///
/// Both overrides bump `source_epoch` by one so settlement can tell that
/// the price generation changed.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PricingOverrides {
    /// Exact per-million-token prices for every model of the instance.
    /// When set, ALL four price lines are authoritative (callers validate
    /// before constructing: input/output must be >= 1 microUSD per million
    /// tokens — a zero input price on a remote endpoint is the local-zero
    /// lie the audit kills).
    pub exact: Option<PriceQuote>,
    /// Conservative ceiling in microUSD per million tokens, applied to
    /// every price line of Unknown-priced models only.
    pub ceiling_micro_usd_per_million_tokens: Option<MicroUsdPerMillionTokens>,
}

impl PricingOverrides {
    /// The ceiling quote: every price line at the ceiling (a conservative
    /// bound on every category).
    pub fn ceiling_quote(&self) -> Option<PriceQuote> {
        self.ceiling_micro_usd_per_million_tokens
            .map(|c| PriceQuote {
                input: c,
                output: c,
                cache_read: c,
                cache_write: c,
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
                pricing: PricingState::Known(PricingSnapshot::exact(
                    exact,
                    entry.source_epoch.saturating_add(1),
                    USER_OVERRIDE_SOURCE_ID.to_string(),
                )),
                source_epoch: entry.source_epoch.saturating_add(1),
                provenance: Provenance::UserOverride,
                ..entry
            };
        }
        if entry.pricing == PricingState::Unknown {
            if let Some(quote) = self.ceiling_quote() {
                return ModelCatalogEntry {
                    pricing: PricingState::ConservativeCeiling(
                        PricingSnapshot::conservative_ceiling(
                            quote,
                            entry.source_epoch.saturating_add(1),
                            USER_OVERRIDE_SOURCE_ID.to_string(),
                        ),
                    ),
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

    fn identity(&self) -> ProviderIdentity {
        ProviderIdentity::new(self.instance_id.clone(), self.id())
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

    fn stream(&self, req: GenericAgentRequest) -> ProviderStream {
        self.inner.stream(req)
    }
}

/// An instance-wrapped provider whose catalog rows resolve pricing by the
/// endpoint's strict [`BillingOrigin`] (billing-origin audit), then apply
/// the user's [`PricingOverrides`]:
///
/// - [`BillingOrigin::Local`] → [`PricingState::LocalZero`] (measured zero);
/// - an OFFICIAL origin → the origin+model built-in row when documented,
///   else the adapter's own declared non-default knowledge
///   (`Provenance::ProviderCatalog`/`UserOverride`), else
///   [`PricingState::Unknown`] — official list prices NEVER leak through a
///   transport family id;
/// - [`BillingOrigin::CustomEndpoint`]/[`BillingOrigin::Gateway`] →
///   [`PricingState::Unknown`] (then the user's exact quote/ceiling may
///   price it), NEVER a built-in price inherited by wire protocol;
/// - the built-in Faktor routing prior fills the row's `quality_prior` when
///   the adapter left its conservative default.
///
/// The wrapper is the daemon's production instance wrapper (config `build`
/// always wraps): identity/capabilities/streaming delegate, only catalog
/// rows are rewritten.
pub struct BillingOriginProvider {
    inner: Arc<dyn Provider>,
    instance_id: String,
    origin: BillingOrigin,
    overrides: PricingOverrides,
}

impl BillingOriginProvider {
    pub fn wrap(
        inner: Arc<dyn Provider>,
        instance_id: impl Into<String>,
        origin: BillingOrigin,
        overrides: PricingOverrides,
    ) -> Arc<dyn Provider> {
        Arc::new(Self {
            inner,
            instance_id: instance_id.into(),
            origin,
            overrides,
        })
    }

    /// The strictly-resolved billing origin of this wrapped endpoint.
    pub fn billing_origin(&self) -> BillingOrigin {
        self.origin
    }

    fn resolve_pricing(&self, inner: &ModelCatalogEntry, model: &str) -> PricingState {
        match self.origin {
            BillingOrigin::Local => PricingState::LocalZero,
            BillingOrigin::CustomEndpoint | BillingOrigin::Gateway => PricingState::Unknown,
            origin => match builtin::lookup_by_origin(origin, model) {
                // The row's class + metadata decide the state: an exact row
                // becomes Known at the conservative max applicable tariff
                // with its validity window attached; a
                // ConservativeSchedule row (undocumented V4-era facts)
                // becomes ConservativeCeiling — a bound, never a claimed
                // list price; retired/expired rows become Unknown.
                Some(row) => builtin::pricing_state_of(&row),
                None => match inner.provenance {
                    // Adapter-declared knowledge (not the family-keyed
                    // trait default) is real and survives; everything the
                    // adapter could only have inherited from a transport
                    // family stays Unknown.
                    Provenance::ProviderCatalog
                    | Provenance::UserOverride
                    | Provenance::Composite
                        if !matches!(inner.pricing, PricingState::Unknown) =>
                    {
                        inner.pricing.clone()
                    }
                    _ => PricingState::Unknown,
                },
            },
        }
    }
}

impl Provider for BillingOriginProvider {
    fn id(&self) -> &str {
        self.inner.id()
    }

    fn identity(&self) -> ProviderIdentity {
        ProviderIdentity::new(self.instance_id.clone(), self.id())
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
        entry.pricing = self.resolve_pricing(&entry, model);
        // The built-in Faktor routing prior fills the row only when the
        // adapter left the conservative generic default; declared priors
        // win. Durable verified outcomes dominate both at scoring time.
        let adapter_declared = entry.provenance == Provenance::ProviderCatalog;
        if !adapter_declared {
            if let Some(profile) = builtin::performance_prior(self.origin, model) {
                entry.quality_prior = QualityPrior {
                    tool_reliability: profile.prior.coding_reliability,
                    reasoning_reliability: profile.prior.coding_reliability,
                    coding_reliability: profile.prior.coding_reliability,
                    context_reliability: profile.prior.context_reliability,
                    availability: entry.quality_prior.availability,
                    estimated_latency_ms: profile.prior.estimated_latency_ms,
                };
            }
        }
        self.overrides.apply(entry)
    }

    fn stream(&self, req: GenericAgentRequest) -> ProviderStream {
        self.inner.stream(req)
    }
}

// ------------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use faktor_core::model::{ModelCapabilities, ModelEconomics};
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

    #[test]
    fn catalog_rows_map_to_their_tokenizer_identity_additively() {
        // Additive P0-81/audits-72-73 mapping: every row names the static
        // tokenizer identity its model routes to. Real local backends exist
        // for o200k_cl100k only; every other family falls back at the
        // context-layer registry (never probed here).
        assert_eq!(
            entry("openai", "gpt-5").tokenizer_id(),
            crate::TokenizerId::O200K_BASE
        );
        assert_eq!(
            entry("openai", "gpt-4").tokenizer_id(),
            crate::TokenizerId::CL100K_BASE
        );
        assert_eq!(
            entry("anthropic", "claude-opus-4-1").tokenizer_id(),
            crate::TokenizerId::ANTHROPIC
        );
        assert_eq!(
            entry("google", "gemini-2.5-pro").tokenizer_id(),
            crate::TokenizerId::GEMINI
        );
        // The deployment hint only disambiguates llama-family weights:
        // ollama-hosted deepseek maps to Llama; the official deepseek API
        // keeps its own (unknown) tokenizer and stays generic.
        assert_eq!(
            entry("ollama", "llama3.8").tokenizer_id(),
            crate::TokenizerId::LLAMA
        );
        assert_eq!(
            entry("ollama", "deepseek-r1").tokenizer_id(),
            crate::TokenizerId::LLAMA
        );
        assert_eq!(
            entry("deepseek", "deepseek-chat").tokenizer_id(),
            crate::TokenizerId::GENERIC_ESTIMATOR
        );
        assert_eq!(
            entry("corp", "my-model").tokenizer_id(),
            crate::TokenizerId::GENERIC_ESTIMATOR
        );
    }

    fn exact_snapshot(input: u64, output: u64) -> PricingSnapshot {
        PricingSnapshot::exact(
            PriceQuote {
                input: MicroUsdPerMillionTokens(input),
                output: MicroUsdPerMillionTokens(output),
                cache_read: MicroUsdPerMillionTokens(input / 5),
                cache_write: MicroUsdPerMillionTokens(input / 2),
            },
            CATALOG_FIRST_EPOCH,
            "row-src".to_string(),
        )
    }

    fn priced(entry: &mut ModelCatalogEntry, input: u64, output: u64) {
        entry.pricing = PricingState::Known(exact_snapshot(input, output));
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
        // fallback. A state carries NO numeric price for Unknown — the
        // settlement-relevant snapshot materializes a no-quote Unknown.
        let e = entry("openai", "gpt-5");
        assert_eq!(e.pricing, PricingState::Unknown);
        assert!(!e.pricing.is_local_zero());
        assert_eq!(e.pricing.quote(), None);
        assert_eq!(e.pricing.authority(), PriceAuthority::Unknown);
        assert_eq!(e.pricing_epoch(), CATALOG_FIRST_EPOCH);
        let snap = e.pricing_snapshot();
        assert_eq!(snap.authority, PriceAuthority::Unknown);
        assert_eq!(snap.quote, None);
        assert_eq!(snap.settle_cost(100_000, 0, 0, 2_000), None);
        // LocalZero is the ONLY zero-cost state: quote Some(ZERO) +
        // LocalZero, never inferred.
        let local = ModelCatalogEntry {
            pricing: PricingState::LocalZero,
            ..entry("ollama", "qwen3.8")
        };
        assert!(local.pricing.is_local_zero());
        assert_eq!(local.pricing.authority(), PriceAuthority::LocalZero);
        assert_eq!(local.pricing.quote(), Some(&PriceQuote::ZERO));
        let known = ModelCatalogEntry {
            pricing: PricingState::Known(exact_snapshot(15, 60)),
            ..entry("openai", "gpt-5")
        };
        assert!(!known.pricing.is_local_zero());
        assert_eq!(known.pricing.authority(), PriceAuthority::Exact);
        assert!(known.pricing.quote().is_some());
    }

    #[test]
    fn state_authority_is_never_inferred_from_numbers() {
        // A zero numeric quote under Known is still Exact (a real zero-
        // price statement is a price statement, not a local runtime); a
        // quote of Some(ZERO) under Unknown is still Unknown and settles
        // to NOTHING.
        let zero_known =
            PricingState::Known(PricingSnapshot::exact(PriceQuote::ZERO, 1, "x".into()));
        assert_eq!(zero_known.authority(), PriceAuthority::Exact);
        assert!(!zero_known.is_local_zero());
        let forged_unknown = PricingState::Unknown;
        assert_eq!(forged_unknown.authority(), PriceAuthority::Unknown);
        assert_eq!(forged_unknown.quote(), None);
        // ConservativeCeiling rows are ceilings, never exact measurements.
        let ceiling = PricingState::ConservativeCeiling(PricingSnapshot::conservative_ceiling(
            PriceQuote::ZERO,
            2,
            "user".into(),
        ));
        assert_eq!(ceiling.authority(), PriceAuthority::ConservativeCeiling);
        assert!(!ceiling.is_local_zero());
    }

    #[test]
    fn pricing_snapshot_materializes_state_without_loss_or_inference() {
        let mut k = entry("openai", "gpt-5");
        priced(&mut k, 2_500_000, 10_000_000);
        let snap = k.pricing_snapshot();
        assert_eq!(snap.authority, PriceAuthority::Exact);
        assert_eq!(
            snap.quote.unwrap().input,
            MicroUsdPerMillionTokens(2_500_000)
        );
        assert_eq!(snap.epoch, CATALOG_FIRST_EPOCH);
        assert_eq!(snap.source_id, "row-src");
        // Ceiling rows keep their ceiling authority.
        let c = ModelCatalogEntry {
            pricing: PricingState::ConservativeCeiling(PricingSnapshot::conservative_ceiling(
                PriceQuote {
                    input: MicroUsdPerMillionTokens(42_000_000),
                    ..PriceQuote::ZERO
                },
                2,
                USER_OVERRIDE_SOURCE_ID.to_string(),
            )),
            source_epoch: 2,
            provenance: Provenance::Composite,
            ..entry("corp", "m")
        };
        let snap = c.pricing_snapshot();
        assert_eq!(snap.authority, PriceAuthority::ConservativeCeiling);
        assert_eq!(snap.epoch, 2);
        // LocalZero materializes the authoritative zero at the row's epoch
        // and source identity.
        let local = ModelCatalogEntry {
            pricing: PricingState::LocalZero,
            source_epoch: 7,
            provenance: Provenance::ProviderCatalog,
            ..entry("ollama", "qwen3.8")
        };
        let snap = local.pricing_snapshot();
        assert_eq!(snap.authority, PriceAuthority::LocalZero);
        assert_eq!(snap.quote, Some(PriceQuote::ZERO));
        assert_eq!(snap.epoch, 7);
        assert_eq!(snap.source_id, "ollama");
        assert_eq!(snap.settle_cost(u64::MAX, 0, 0, 0), Some(0));
        // Unknown materializes the no-quote Unknown snapshot at the row's
        // epoch; the BUILT-IN source id names the built-in catalog miss.
        let u = ModelCatalogEntry {
            source_epoch: 3,
            ..entry("openai", "gpt-5")
        };
        let snap = u.pricing_snapshot();
        assert_eq!(snap.authority, PriceAuthority::Unknown);
        assert_eq!(snap.quote, None);
        assert_eq!(snap.epoch, 3);
        assert_eq!(snap.source_id, BUILTIN_SOURCE_ID);
        // Stale rows forward their last-known snapshot verbatim.
        let stale = PricingState::Stale {
            last_known: exact_snapshot(15, 60),
            observed_at_ms: 99,
        };
        let s = ModelCatalogEntry {
            pricing: stale,
            ..entry("openai", "gpt-5")
        };
        let snap = s.pricing_snapshot();
        assert_eq!(snap.authority, PriceAuthority::Exact);
        assert_eq!(snap.quote.unwrap().input, MicroUsdPerMillionTokens(15));
        assert_eq!(snap.settle_cost(1_000_000, 0, 0, 0), Some(15));
    }

    #[test]
    fn known_rows_settle_exact_per_million_prices_never_free() {
        // gpt-4o-mini-class list prices ($0.15/M input) never truncate:
        // settle_cost of 1M input tokens is 150_000 microUSD, and a
        // 1-token call costs 1 micro — the quote math of item A.
        let mut k = entry("openai", "gpt-4o-mini");
        k.pricing = PricingState::Known(exact_snapshot(150_000, 600_000));
        let snap = k.pricing_snapshot();
        assert_eq!(snap.settle_cost(1_000_000, 0, 0, 0), Some(150_000));
        assert_eq!(snap.settle_cost(1, 0, 0, 0), Some(1));
        assert_eq!(snap.settle_cost(1_000_000, 0, 0, 1_000_000), Some(750_000));
    }

    #[test]
    fn serde_roundtrip_is_total_and_stable() {
        let mut a = entry("openai", "gpt-5");
        priced(&mut a, 15, 60);
        let mut b = entry("ollama", "qwen3.8");
        b.pricing = PricingState::LocalZero;
        b.provenance = Provenance::ProviderCatalog;
        let mut c = entry("corp-proxy", "my-model");
        c.pricing = PricingState::ConservativeCeiling(PricingSnapshot::conservative_ceiling(
            PriceQuote {
                input: MicroUsdPerMillionTokens(42_000_000),
                ..PriceQuote::ZERO
            },
            2,
            USER_OVERRIDE_SOURCE_ID.to_string(),
        ));
        c.source_epoch = 2;
        c.provenance = Provenance::Composite;
        let mut d = entry("corp-proxy", "stale-model");
        d.pricing = PricingState::Stale {
            last_known: exact_snapshot(3, 6),
            observed_at_ms: 1234,
        };
        for e in [a.clone(), b.clone(), c.clone(), d.clone(), entry("x", "y")] {
            let v = serde_json::to_value(&e).unwrap();
            let back: ModelCatalogEntry = serde_json::from_value(v).unwrap();
            assert_eq!(back, e);
        }
        // Wire names are frozen snake_case; quotes serialize as plain
        // per-million integers.
        let v = serde_json::to_value(&b).unwrap();
        assert_eq!(v["pricing"], serde_json::json!("local_zero"));
        assert_eq!(v["provenance"], serde_json::json!("provider_catalog"));
        let v = serde_json::to_value(&c).unwrap();
        assert_eq!(
            v["pricing"]["conservative_ceiling"]["authority"],
            serde_json::json!("conservative_ceiling")
        );
        let v = serde_json::to_value(&d).unwrap();
        assert_eq!(
            v["pricing"]["stale"]["observed_at_ms"],
            serde_json::json!(1234)
        );
        // Hostile unknown fields are ignored (never fatal); hostile enum
        // values are rejected.
        let mut hostile = serde_json::to_value(&c).unwrap();
        hostile["intruder"] = serde_json::json!("x");
        assert!(serde_json::from_value::<ModelCatalogEntry>(hostile).is_ok());
        for bad in ["free_beer", "measured", "null"] {
            assert!(
                serde_json::from_value::<PricingState>(serde_json::json!(bad)).is_err(),
                "{bad} must be rejected"
            );
        }
    }

    #[test]
    fn ord_is_full_field_deterministic_and_consistent_with_eq() {
        let mut a = entry("openai", "gpt-5");
        priced(&mut a, 15, 60);
        let mut a2 = a.clone();
        a2.source_epoch += 1;
        let mut local = entry("ollama", "qwen3.8");
        local.pricing = PricingState::LocalZero;
        let mut ceiling = entry("corp", "m");
        ceiling.pricing = PricingState::ConservativeCeiling(exact_snapshot(15, 60));
        let mut stale = entry("corp", "stale");
        stale.pricing = PricingState::Stale {
            last_known: exact_snapshot(15, 60),
            observed_at_ms: 1,
        };
        let mut unknown = entry("openai", "gpt-5-mini");
        unknown.pricing = PricingState::Unknown;
        assert_eq!(a.cmp(&a), std::cmp::Ordering::Equal);
        assert_eq!(a.partial_cmp(&a2), Some(std::cmp::Ordering::Less));
        let mut v = vec![
            a.clone(),
            a2.clone(),
            local.clone(),
            ceiling.clone(),
            stale.clone(),
            unknown.clone(),
        ];
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
        let snap = exact_snapshot(15, 60);
        assert!(PricingState::LocalZero < PricingState::Known(snap.clone()));
        assert!(
            PricingState::Known(snap.clone()) < PricingState::ConservativeCeiling(snap.clone())
        );
        assert!(PricingState::ConservativeCeiling(snap.clone()) < PricingState::Unknown);
        assert!(
            PricingState::Unknown
                < PricingState::Stale {
                    last_known: snap,
                    observed_at_ms: 0,
                }
        );
    }

    #[test]
    fn overrides_never_touch_local_zero_and_bump_epoch_once() {
        let local = ModelCatalogEntry {
            pricing: PricingState::LocalZero,
            source_epoch: CATALOG_FIRST_EPOCH,
            ..entry("ollama", "qwen3.8")
        };
        let overrides = PricingOverrides {
            exact: Some(PriceQuote {
                input: MicroUsdPerMillionTokens::from_dollars_per_million(15),
                output: MicroUsdPerMillionTokens::from_dollars_per_million(60),
                ..PriceQuote::ZERO
            }),
            ceiling_micro_usd_per_million_tokens: None,
        };
        assert_eq!(overrides.apply(local.clone()), local, "locals are exempt");
        let ceiling_only = PricingOverrides {
            exact: None,
            ceiling_micro_usd_per_million_tokens: Some(MicroUsdPerMillionTokens(50)),
        };
        assert_eq!(ceiling_only.apply(local.clone()), local);
    }

    #[test]
    fn user_override_prices_every_model_as_exact_and_epoch_increments() {
        let overrides = PricingOverrides {
            exact: Some(PriceQuote {
                input: MicroUsdPerMillionTokens::from_dollars_per_million(2),
                output: MicroUsdPerMillionTokens::from_dollars_per_million(8),
                ..PriceQuote::ZERO
            }),
            ceiling_micro_usd_per_million_tokens: None,
        };
        for base in [
            PricingState::Unknown,
            PricingState::Known(PricingSnapshot::exact(PriceQuote::ZERO, 1, "x".into())),
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
                PricingState::Known(snap) => {
                    assert_eq!(snap.authority, PriceAuthority::Exact);
                    let q = snap.quote.expect("exact override quotes");
                    assert_eq!(q.input, MicroUsdPerMillionTokens(2_000_000));
                    assert_eq!(q.output, MicroUsdPerMillionTokens(8_000_000));
                    assert_eq!(snap.epoch, before.source_epoch + 1);
                    assert_eq!(snap.source_id, USER_OVERRIDE_SOURCE_ID);
                }
                other => panic!("override must produce Known, got {other:?}"),
            }
        }
    }

    #[test]
    fn ceiling_prices_only_unknown_models_as_conservative_ceiling() {
        let ceiling = PricingOverrides {
            exact: None,
            ceiling_micro_usd_per_million_tokens: Some(MicroUsdPerMillionTokens(42_000_000)),
        };
        // Unknown -> ConservativeCeiling at exactly the ceiling on every
        // price line, provenance Composite, epoch bumped, authority never
        // Exact (a ceiling is a bound, not a measurement).
        let before = entry("corp-proxy", "renamed-model");
        let after = ceiling.apply(before);
        assert_eq!(after.provenance, Provenance::Composite);
        assert_eq!(after.source_epoch, CATALOG_FIRST_EPOCH + 1);
        match after.pricing {
            PricingState::ConservativeCeiling(snap) => {
                assert_eq!(snap.authority, PriceAuthority::ConservativeCeiling);
                let q = snap.quote.expect("ceiling quotes");
                for line in [q.input, q.output, q.cache_read, q.cache_write] {
                    assert_eq!(
                        line,
                        MicroUsdPerMillionTokens(42_000_000),
                        "ceiling on every line"
                    );
                }
                assert_eq!(snap.epoch, CATALOG_FIRST_EPOCH + 1);
            }
            other => panic!("ceiling must produce ConservativeCeiling, got {other:?}"),
        }
        // Known models keep their REAL price under a ceiling.
        let mut known = entry("corp-proxy", "known-model");
        priced(&mut known, 7_000_000, 21_000_000);
        let after = ceiling.apply(known.clone());
        assert_eq!(after, known);
        assert_eq!(after.provenance, Provenance::BuiltIn);
        assert_eq!(after.source_epoch, CATALOG_FIRST_EPOCH);
        // ... and so do ConservativeCeiling rows (never double-ceilinged).
        let mut ceilinged = entry("corp-proxy", "ceilinged-model");
        ceilinged.pricing =
            PricingState::ConservativeCeiling(PricingSnapshot::conservative_ceiling(
                PriceQuote {
                    input: MicroUsdPerMillionTokens(1_000),
                    ..PriceQuote::ZERO
                },
                2,
                "s".into(),
            ));
        let after = ceiling.apply(ceilinged.clone());
        assert_eq!(after, ceilinged);
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

        fn stream(&self, _req: crate::GenericAgentRequest) -> crate::ProviderStream {
            Box::pin(futures::stream::empty())
        }
    }

    #[test]
    fn default_catalog_entries_consult_builtin_then_stay_unknown() {
        // Adapters compile against the default impl: family+model rows
        // documented in the built-in table come back Known at epoch 1 with
        // the built-in source id; everything else must NOT read as
        // zero-priced: Unknown + BuiltIn + epoch 1.
        let p = LegacyTestProvider {
            id: "openai".into(),
        };
        let e = p.catalog_entry("gpt-4o");
        assert_eq!(e.pricing.authority(), PriceAuthority::Exact);
        assert_eq!(e.source_epoch, CATALOG_FIRST_EPOCH);
        assert_eq!(e.provenance, Provenance::BuiltIn);
        let snap = e.pricing_snapshot();
        let q = snap.quote.unwrap();
        assert_eq!(q.input, MicroUsdPerMillionTokens(2_500_000));
        assert_eq!(q.output, MicroUsdPerMillionTokens(10_000_000));
        assert_eq!(q.cache_read, MicroUsdPerMillionTokens(1_250_000));
        assert_eq!(q.cache_write, MicroUsdPerMillionTokens(0));
        assert_eq!(snap.epoch, 1);
        assert_eq!(snap.source_id, BUILTIN_SOURCE_ID);
        // The 500k microUSD/M row settles exactly (never a truncated free).
        assert_eq!(snap.settle_cost(1_000_000, 0, 0, 0), Some(2_500_000));
        // Undocumented models of the SAME family stay Unknown.
        for model in p.known_models() {
            let e = p.catalog_entry(&model);
            assert_eq!(e.pricing, PricingState::Unknown, "{model} must be Unknown");
            assert_eq!(e.provenance, Provenance::BuiltIn);
            assert_eq!(e.pricing_epoch(), CATALOG_FIRST_EPOCH);
            assert_eq!(e.quality_prior, QualityPrior::conservative_generic());
            assert_eq!(e.capabilities, p.capabilities(&model));
        }
        let e = p.catalog_entry("probed-a");
        assert!(!e.pricing.is_local_zero());
        assert_eq!(e.pricing_snapshot().quote, None);
        // Legacy (unlisted) families never read as priced.
        let legacy = LegacyTestProvider {
            id: "legacy".into(),
        };
        for model in legacy.known_models() {
            assert_eq!(legacy.catalog_entry(&model).pricing, PricingState::Unknown);
        }
    }

    // ------------------------------------------------------ admission matrix

    fn state_for(authority: PriceAuthority) -> PricingState {
        match authority {
            PriceAuthority::Exact => PricingState::Known(exact_snapshot(15, 60)),
            PriceAuthority::ConservativeCeiling => {
                PricingState::ConservativeCeiling(exact_snapshot(15, 60))
            }
            PriceAuthority::LocalZero => PricingState::LocalZero,
            PriceAuthority::Unknown => PricingState::Unknown,
        }
    }

    fn expected_admission(
        mode: &RoutingMode,
        authority: PriceAuthority,
        hard_cost_cap: bool,
        allow_unknown_balanced: bool,
    ) -> bool {
        let free_priced = !matches!(authority, PriceAuthority::Unknown);
        match mode {
            RoutingMode::Economy => free_priced,
            // MaximumQuality maximizes verified quality subject to hard
            // caps: Unknown admitted only without a hard cost cap.
            RoutingMode::MaximumQuality => free_priced || !hard_cost_cap,
            RoutingMode::Balanced => free_priced || (allow_unknown_balanced && !hard_cost_cap),
            RoutingMode::Pinned { .. } => {
                !matches!((authority, hard_cost_cap), (PriceAuthority::Unknown, true))
            }
        }
    }

    #[test]
    fn admission_matrix_is_exact_for_every_mode_x_state_x_cap_row() {
        // The audit matrix: Economy/Balanced/MaximumQuality/Pinned x
        // Exact/Ceiling/LocalZero/Unknown, with the hard-cost-cap rows.
        let modes = [
            RoutingMode::Economy,
            RoutingMode::MaximumQuality,
            RoutingMode::Balanced,
            RoutingMode::Pinned {
                provider: "p".into(),
                model: "m".into(),
            },
        ];
        for mode in &modes {
            for authority in [
                PriceAuthority::Exact,
                PriceAuthority::ConservativeCeiling,
                PriceAuthority::LocalZero,
                PriceAuthority::Unknown,
            ] {
                for hard_cost_cap in [false, true] {
                    for allow_unknown_balanced in [false, true] {
                        let state = state_for(authority);
                        let got = admissible(mode, &state, hard_cost_cap, allow_unknown_balanced);
                        let want = expected_admission(
                            mode,
                            authority,
                            hard_cost_cap,
                            allow_unknown_balanced,
                        );
                        assert_eq!(
                            got, want,
                            "admissible({mode:?}, {authority:?}, cap={hard_cost_cap},                              allow_unknown_balanced={allow_unknown_balanced})"
                        );
                    }
                }
            }
        }
        // Stale rows are judged by their last-known authority: a stale
        // exact row admits everywhere an exact row does and refuses the
        // Unknown rows.
        let stale_exact = PricingState::Stale {
            last_known: exact_snapshot(1, 2),
            observed_at_ms: 1,
        };
        assert!(admissible(
            &RoutingMode::Economy,
            &stale_exact,
            false,
            false
        ));
        assert!(admissible(
            &RoutingMode::Pinned {
                provider: "p".into(),
                model: "m".into()
            },
            &stale_exact,
            true,
            false
        ));
        let stale_unknown = PricingState::Stale {
            last_known: PricingSnapshot::unknown(1, "s".into()),
            observed_at_ms: 1,
        };
        assert!(!admissible(
            &RoutingMode::Economy,
            &stale_unknown,
            false,
            false
        ));
        assert!(!admissible(
            &RoutingMode::Pinned {
                provider: "p".into(),
                model: "m".into()
            },
            &stale_unknown,
            true,
            false
        ));
        assert!(admissible(
            &RoutingMode::Pinned {
                provider: "p".into(),
                model: "m".into()
            },
            &stale_unknown,
            false,
            false
        ));
    }

    #[test]
    fn override_wrapper_delegates_identity_and_rewrites_provider() {
        use crate::ProviderRegistry;
        let inner = Arc::new(LegacyTestProvider {
            id: "openai".into(),
        });
        let wrapped = PricingOverrideProvider::wrap(
            inner,
            "corp-proxy",
            PricingOverrides {
                exact: None,
                ceiling_micro_usd_per_million_tokens: Some(MicroUsdPerMillionTokens(99_000_000)),
            },
        );
        assert_eq!(wrapped.id(), "openai", "family id for capability queries");
        assert_eq!(
            wrapped.identity(),
            ProviderIdentity::new("corp-proxy", "openai")
        );
        assert_eq!(
            wrapped.known_models(),
            vec!["default".to_string(), "probed-a".to_string()],
            "known_models delegates through the wrapper"
        );
        // Ceiling applies ONLY to rows the adapter leaves Unknown; a row
        // the built-in table prices keeps its exact built-in price.
        let e = wrapped.catalog_entry("probed-a");
        assert_eq!(e.provider, "corp-proxy", "rows name the instance id");
        assert_eq!(e.provenance, Provenance::Composite);
        assert_eq!(e.source_epoch, CATALOG_FIRST_EPOCH + 1);
        assert_eq!(e.pricing.authority(), PriceAuthority::ConservativeCeiling);
        let e = wrapped.catalog_entry("gpt-4o");
        assert_eq!(e.provenance, Provenance::BuiltIn);
        assert_eq!(e.source_epoch, CATALOG_FIRST_EPOCH);
        assert_eq!(e.pricing.authority(), PriceAuthority::Exact);

        let mut reg = ProviderRegistry::new();
        reg.try_register(wrapped).unwrap();
        assert_eq!(reg.ids(), vec!["corp-proxy"]);
        assert_eq!(reg.get("corp-proxy").unwrap().id(), "openai");
    }

    #[test]
    fn custom_openai_does_not_inherit_builtin_price() {
        // The SAME OpenAI-wire adapter resolves two different billing
        // origins: the official canonical endpoint gets the documented
        // built-in list price, while a custom OpenAI-compatible endpoint —
        // whose family-keyed trait default WOULD have found that same row —
        // is stripped to Unknown. Wire protocol is not a billing contract.
        let openai = || {
            Arc::new(LegacyTestProvider {
                id: "openai".into(),
            }) as Arc<dyn Provider>
        };
        // The trap: the raw adapter DOES inherit the built-in price.
        assert_eq!(
            openai().catalog_entry("gpt-4o").pricing.authority(),
            PriceAuthority::Exact
        );
        let official = BillingOriginProvider::wrap(
            openai(),
            "openai-official",
            BillingOrigin::OfficialOpenAi,
            PricingOverrides::default(),
        );
        let custom = BillingOriginProvider::wrap(
            openai(),
            "corp-proxy",
            BillingOrigin::CustomEndpoint,
            PricingOverrides::default(),
        );
        let gateway = BillingOriginProvider::wrap(
            openai(),
            "gw",
            BillingOrigin::Gateway,
            PricingOverrides::default(),
        );
        let e = official.catalog_entry("gpt-4o");
        assert_eq!(e.provider, "openai-official");
        assert_eq!(e.pricing.authority(), PriceAuthority::Exact);
        let q = e.pricing_snapshot().quote.expect("built-in row quotes");
        assert_eq!(q.input, MicroUsdPerMillionTokens(2_500_000));
        assert_eq!(q.output, MicroUsdPerMillionTokens(10_000_000));
        assert_eq!(e.pricing_snapshot().source_id, BUILTIN_SOURCE_ID);
        assert_eq!(
            e.quality_prior.coding_reliability, 90,
            "official rows carry the documented Faktor routing prior"
        );
        for endpoint in [&custom, &gateway] {
            let e = endpoint.catalog_entry("gpt-4o");
            assert_eq!(
                e.pricing,
                PricingState::Unknown,
                "a custom/gateway endpoint must never inherit official list prices"
            );
            assert_eq!(e.pricing_snapshot().settle_cost(1_000_000, 0, 0, 0), None);
            assert_eq!(
                e.quality_prior,
                QualityPrior::default(),
                "official priors do not leak across billing origins either"
            );
        }
        // A user exact quote is the sanctioned way to price a custom
        // endpoint; it survives the origin stripping.
        let priced_custom = BillingOriginProvider::wrap(
            openai(),
            "corp-proxy",
            BillingOrigin::CustomEndpoint,
            PricingOverrides {
                exact: Some(PriceQuote {
                    input: MicroUsdPerMillionTokens(500_000),
                    output: MicroUsdPerMillionTokens(1_500_000),
                    ..PriceQuote::ZERO
                }),
                ceiling_micro_usd_per_million_tokens: None,
            },
        );
        let e = priced_custom.catalog_entry("gpt-4o");
        assert_eq!(e.pricing.authority(), PriceAuthority::Exact);
        assert_eq!(
            e.pricing.quote().unwrap().input,
            MicroUsdPerMillionTokens(500_000)
        );
        assert_eq!(e.provenance, Provenance::UserOverride);

        // DeepSeek: the V4-era schedule is NOT documented as exact, so the
        // official endpoint resolves to a CONSERVATIVE CEILING (a bound,
        // never a claimed list price), while a custom endpoint stays
        // Unknown (no inheritance).
        let deepseek = || {
            Arc::new(LegacyTestProvider {
                id: "deepseek".into(),
            }) as Arc<dyn Provider>
        };
        let official_ds = BillingOriginProvider::wrap(
            deepseek(),
            "deepseek-official",
            BillingOrigin::OfficialDeepSeek,
            PricingOverrides::default(),
        );
        let custom_ds = BillingOriginProvider::wrap(
            deepseek(),
            "corp-ds",
            BillingOrigin::CustomEndpoint,
            PricingOverrides::default(),
        );
        let e = official_ds.catalog_entry("deepseek-chat");
        assert_eq!(
            e.pricing.authority(),
            PriceAuthority::ConservativeCeiling,
            "an undocumented V4-era schedule is a bound, not Exact"
        );
        assert_eq!(
            e.pricing.quote().unwrap().input,
            MicroUsdPerMillionTokens(560_000)
        );
        assert_eq!(
            custom_ds.catalog_entry("deepseek-chat").pricing,
            PricingState::Unknown
        );
        // Origin scoping is exact: an official OpenAI endpoint never sees
        // another origin's row, even for a model name it might serve.
        assert_eq!(
            official.catalog_entry("claude-sonnet-4").pricing,
            PricingState::Unknown
        );
        assert_eq!(
            official_ds.catalog_entry("gpt-4o").pricing,
            PricingState::Unknown
        );
    }

    #[test]
    fn billed_cache_writes_and_validity_windows_reach_catalog_snapshots() {
        // The catalog snapshot of an active Anthropic row reserves the MAX
        // documented cache-write tariff (1h = 2x input) and carries the row's
        // validity window, so a hard budget can never under-reserve cache
        // creation.
        let anthropic = || {
            Arc::new(LegacyTestProvider {
                id: "anthropic".into(),
            }) as Arc<dyn Provider>
        };
        let official = BillingOriginProvider::wrap(
            anthropic(),
            "anthropic-official",
            BillingOrigin::OfficialAnthropic,
            PricingOverrides::default(),
        );
        let e = official.catalog_entry("claude-opus-4");
        assert_eq!(e.pricing.authority(), PriceAuthority::Exact);
        let snap = e.pricing_snapshot();
        let q = snap.quote.expect("active row quotes");
        assert_eq!(q.cache_write, MicroUsdPerMillionTokens(30_000_000));
        assert_eq!(q.cache_read, MicroUsdPerMillionTokens(1_500_000));
        assert_eq!(snap.settle_cost(0, 0, 1_000_000, 0), Some(30_000_000));
        assert_eq!(
            snap.valid_until_ms,
            Some(builtin::BUILTIN_VALID_UNTIL_MS),
            "the validity window reaches the route-time snapshot"
        );
        assert_eq!(snap.source_id, BUILTIN_SOURCE_ID);
        // Retired 2026 IDs resolve to Unknown, never their last price.
        for retired in ["claude-sonnet-4", "claude-haiku-3.5"] {
            let e = official.catalog_entry(retired);
            assert_eq!(e.pricing, PricingState::Unknown, "{retired} is retired");
            assert_eq!(e.pricing_snapshot().settle_cost(1_000_000, 0, 0, 0), None);
        }
    }

    #[test]
    fn billing_origin_change_cannot_inherit_an_unrelated_catalog() {
        // The SAME adapter (family id "openai") resolved under three
        // billing origins: official OpenAI sees the OpenAI row, official
        // DeepSeek sees only its own (conservative) row, and a model with
        // no row under the resolved origin stays Unknown. Origin, not wire
        // family or adapter id, selects the catalog.
        let adapter = || {
            Arc::new(LegacyTestProvider {
                id: "openai".into(),
            }) as Arc<dyn Provider>
        };
        let openai = BillingOriginProvider::wrap(
            adapter(),
            "same-instance",
            BillingOrigin::OfficialOpenAi,
            PricingOverrides::default(),
        );
        let deepseek = BillingOriginProvider::wrap(
            adapter(),
            "same-instance",
            BillingOrigin::OfficialDeepSeek,
            PricingOverrides::default(),
        );
        let google = BillingOriginProvider::wrap(
            adapter(),
            "same-instance",
            BillingOrigin::OfficialGoogle,
            PricingOverrides::default(),
        );
        assert_eq!(
            openai.catalog_entry("gpt-4o").pricing.authority(),
            PriceAuthority::Exact
        );
        assert_eq!(
            deepseek.catalog_entry("gpt-4o").pricing,
            PricingState::Unknown,
            "a DeepSeek origin never inherits the OpenAI catalog"
        );
        assert_eq!(
            google.catalog_entry("gpt-4o").pricing,
            PricingState::Unknown
        );
        assert_eq!(
            openai.catalog_entry("deepseek-chat").pricing,
            PricingState::Unknown,
            "and OpenAI never inherits DeepSeek's conservative row"
        );
        assert_eq!(
            deepseek.catalog_entry("deepseek-chat").pricing.authority(),
            PriceAuthority::ConservativeCeiling
        );
    }

    #[test]
    fn effective_admission_downgrades_expired_exact_rows() {
        let expired_exact = PricingState::Known(
            PricingSnapshot::exact(
                PriceQuote {
                    input: MicroUsdPerMillionTokens(1_000_000),
                    ..PriceQuote::ZERO
                },
                1,
                "expired".into(),
            )
            .with_valid_until(100),
        );
        let no_cap = expired_exact.effective_at(200, false);
        assert!(no_cap.is_unknown());
        assert!(!admissible_effective(
            &RoutingMode::Economy,
            &no_cap,
            false,
            false
        ));
        // MaximumQuality admits Unknown WITHOUT a hard cap (spend settles
        // documented Unknown), but never with a cap.
        assert!(admissible_effective(
            &RoutingMode::MaximumQuality,
            &no_cap,
            false,
            false
        ));
        assert!(!admissible_effective(
            &RoutingMode::MaximumQuality,
            &no_cap,
            true,
            false
        ));
        // A documented ceiling keeps a hard-capped request admissible at
        // the bound.
        let bounded = PricingState::Known(
            PricingSnapshot::exact(
                PriceQuote {
                    input: MicroUsdPerMillionTokens(1_000_000),
                    ..PriceQuote::ZERO
                },
                1,
                "expired".into(),
            )
            .with_valid_until(100)
            .with_conservative_ceiling(PriceQuote {
                input: MicroUsdPerMillionTokens(9_000_000),
                ..PriceQuote::ZERO
            }),
        )
        .effective_at(200, true);
        assert_eq!(bounded.authority(), PriceAuthority::ConservativeCeiling);
        assert!(admissible_effective(
            &RoutingMode::Economy,
            &bounded,
            true,
            false
        ));
        // Stale rows are Unknown on the effective path even under a cap.
        let stale = PricingState::Stale {
            last_known: exact_snapshot(1, 2),
            observed_at_ms: 7,
        }
        .effective_at(200, true);
        assert!(!admissible_effective(
            &RoutingMode::Pinned {
                provider: "p".into(),
                model: "m".into()
            },
            &stale,
            true,
            false
        ));
    }
}
