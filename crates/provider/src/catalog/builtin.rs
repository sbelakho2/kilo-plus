//! Versioned built-in list-price table (audit wave-B item C, billing-origin
//! audit, and the hard-budget-safe pricing audit): documented list prices of
//! officially-known models ONLY — nothing invented, nothing extrapolated.
//! Prices are microUSD per MILLION tokens (the published $/M reading x 1e6)
//! so sub-$1/M list prices ($0.15/M gpt-4o-mini input, $0.10/M gemini flash
//! input, ...) stay exact and can never truncate to a free lie.
//!
//! Rows are keyed by [`BillingOrigin`] + model, NOT by transport family id:
//! a custom OpenAI-compatible proxy speaks the same wire protocol but bills
//! somewhere else, so it must never inherit official OpenAI list prices. The
//! legacy family-keyed [`lookup`] documents the raw-adapter compatibility
//! path (and is only reachable for adapters used OUTSIDE the config-built
//! daemon graph); the daemon resolves the real origin in config and uses
//! [`lookup_by_origin`].
//!
//! # builtin-v2: metadata, validity windows and conditional tariffs
//!
//! Every row carries its own `effective_at_ms` / `observed_at_ms` /
//! `valid_until_ms`, `source_id` and `catalog_version`
//! ([`BUILTIN_CATALOG_VERSION`] = `"builtin-v2"`): a price statement is
//! dated, and a quote past `valid_until_ms` stops being exact at route time
//! (see `PricingState::effective_at`). Rows whose official model ID was
//! RETIRED carry `retired_at_ms` and are excluded from every Exact lookup —
//! a withdrawn ID never routes at its last-known price.
//!
//! Tariffs are conditional, not a single number:
//!
//! - Anthropic charges prompt-cache WRITES, and the write rate depends on
//!   the cache TTL: [`BuiltinCacheWrite::TtlDependent`] carries the 5-minute
//!   and 1-hour rates, and [`BuiltinRowSchedule`] — a [`PriceSchedule`] —
//!   applies the known TTL exactly, or the MAXIMUM of the two for a hard
//!   budget whose applicable tariff is unknown (never the minimum);
//! - cache-read rows are included where documented; `cache_write` is never
//!   zero-by-omission on a provider that bills writes;
//! - DeepSeek moved to V4-era schedules the audited table does not document
//!   as exact: [`BuiltinPriceClass::ConservativeSchedule`] marks a row whose
//!   numbers are a documented CONSERVATIVE BOUND (a ceiling), never a
//!   claimed exact list price.
//!
//! # Performance profiles
//!
//! [`performance_prior`] exposes the documented FAKTOR ROUTING PRIORS for
//! the same models — reliability/latency estimates the economic router
//! uses to compare candidates. These are deliberate, conservative routing
//! priors, NOT vendor truths: every profile carries
//! [`QualityAuthority::ConservativeUnknown`] and the
//! [`PERFORMANCE_PRIOR_VERSION`] benchmark string, and durable verified
//! outcomes dominate them whenever verified history exists.

use faktor_core::model::{
    unix_now_ms, BillingOrigin, CacheWriteTtl, MicroUsdPerMillionTokens, ModelPerformance,
    ModelPerformanceProfile, PriceAuthority, PriceQuote, PriceSchedule, PriceUnavailable,
    PricingContext, PricingSnapshot, PricingState, QualityAuthority, ServiceTier,
};

use crate::catalog::PricingProvenance;

/// Version of the frozen built-in table (bump the table AND this string
/// whenever a price changes, so old snapshots never claim a new version).
pub const BUILTIN_CATALOG_VERSION: &str = "builtin-v2";

/// Source id stamped on every built-in row's snapshot.
pub const BUILTIN_SOURCE_ID: &str = "faktor-builtin-v2";

/// Documented effective date of the builtin-v2 rows (2026-08-01T00:00:00Z).
pub const BUILTIN_EFFECTIVE_AT_MS: u64 = 1_785_542_400_000;

/// Documented observation date of the builtin-v2 rows (2026-09-01T00:00:00Z).
pub const BUILTIN_OBSERVED_AT_MS: u64 = 1_788_220_800_000;

/// Validity window of the builtin-v2 rows (2027-02-01T00:00:00Z): after
/// this instant a row's exact quote stops being authoritative at route time
/// and must be re-observed or treated as Unknown / a declared ceiling.
pub const BUILTIN_VALID_UNTIL_MS: u64 = 1_801_440_000_000;

/// Version string of the built-in Faktor routing priors (the performance
/// profiles below). Bump whenever a prior changes.
pub const PERFORMANCE_PRIOR_VERSION: &str = "faktor-routing-priors-v1";

/// Source id of the built-in performance priors (inspectable identity of a
/// conservative-unknown prior).
pub const PERFORMANCE_PRIOR_SOURCE_ID: &str = "faktor-builtin-priors-v1";

/// How a row's cache-write line is billed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinCacheWrite {
    /// The provider publishes no cache-write charge.
    NotBilled,
    /// One published write rate (microUSD per million tokens).
    Fixed(u64),
    /// TTL-dependent published write rates (the Anthropic shape).
    TtlDependent { five_minutes: u64, one_hour: u64 },
}

impl BuiltinCacheWrite {
    /// The maximum documented write rate: the conservative tariff when the
    /// applicable TTL is unknown.
    pub const fn max_rate(self) -> u64 {
        match self {
            BuiltinCacheWrite::NotBilled => 0,
            BuiltinCacheWrite::Fixed(v) => v,
            BuiltinCacheWrite::TtlDependent {
                five_minutes,
                one_hour,
            } => {
                if five_minutes > one_hour {
                    five_minutes
                } else {
                    one_hour
                }
            }
        }
    }

    /// The write rate for a KNOWN TTL, or `None` when that tariff does not
    /// exist / the tariff is not known and `hard_budget` is false.
    pub const fn rate_for(self, ttl: Option<CacheWriteTtl>, hard_budget: bool) -> Option<u64> {
        match self {
            BuiltinCacheWrite::NotBilled => Some(0),
            BuiltinCacheWrite::Fixed(v) => Some(v),
            BuiltinCacheWrite::TtlDependent {
                five_minutes,
                one_hour,
            } => match ttl {
                Some(CacheWriteTtl::FiveMinutes) => Some(five_minutes),
                Some(CacheWriteTtl::OneHour) => Some(one_hour),
                None if hard_budget => Some(if five_minutes > one_hour {
                    five_minutes
                } else {
                    one_hour
                }),
                None => None,
            },
        }
    }
}

/// What a row's numbers REST on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinPriceClass {
    /// Documented current list prices (exact statement with a validity
    /// window).
    Exact,
    /// No exact V4-era facts are documented: the numbers are a documented
    /// CONSERVATIVE BOUND and the resolved state is
    /// [`PricingState::ConservativeCeiling`], admissible under a hard cap
    /// but never claimed as a measured price.
    ConservativeSchedule,
}

/// One documented list-price row (per-million-token microUSD) with its
/// catalog metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuiltinPrice {
    /// The BILLING origin the row belongs to (never a transport family id).
    pub origin: BillingOrigin,
    /// The officially-known model name (matched case-insensitively).
    pub model: &'static str,
    /// What the numeric lines rest on.
    pub class: BuiltinPriceClass,
    pub input_micro_usd_per_million: u64,
    pub output_micro_usd_per_million: u64,
    /// Published cache-read line, when the audit records one; `None`
    /// otherwise (the line is 0 — never billed).
    pub cache_read_micro_usd_per_million: Option<u64>,
    /// Published cache-write tariff(s); never silently zero on a provider
    /// that bills writes.
    pub cache_write: BuiltinCacheWrite,
    /// Wall-clock ms the row's prices took effect.
    pub effective_at_ms: u64,
    /// Wall-clock ms the row's prices were observed.
    pub observed_at_ms: u64,
    /// Wall-clock ms after which the exact quote stops being authoritative
    /// at route time (`None` = no published expiry).
    pub valid_until_ms: Option<u64>,
    /// Wall-clock ms the official model ID was retired (`None` = active).
    /// Retired rows are documented here but excluded from every Exact
    /// lookup: a withdrawn ID never routes at its last-known price.
    pub retired_at_ms: Option<u64>,
    /// Source id stamped on snapshots cut from this row.
    pub source_id: &'static str,
    /// Catalog version of this row.
    pub catalog_version: &'static str,
}

impl BuiltinPrice {
    /// True once the official model ID was retired.
    pub const fn is_retired(&self) -> bool {
        self.retired_at_ms.is_some()
    }

    /// True once the validity window elapsed at `at_ms` (inclusive
    /// boundary). A row without a window never expires.
    pub const fn is_expired_at(&self, at_ms: u64) -> bool {
        match self.valid_until_ms {
            Some(until) => at_ms >= until,
            None => false,
        }
    }
}

/// The frozen, versioned price table.
pub const TABLE: &[BuiltinPrice] = &[
    // Official OpenAI.
    BuiltinPrice {
        origin: BillingOrigin::OfficialOpenAi,
        model: "gpt-4o",
        class: BuiltinPriceClass::Exact,
        input_micro_usd_per_million: 2_500_000,
        output_micro_usd_per_million: 10_000_000,
        cache_read_micro_usd_per_million: Some(1_250_000),
        cache_write: BuiltinCacheWrite::NotBilled,
        effective_at_ms: BUILTIN_EFFECTIVE_AT_MS,
        observed_at_ms: BUILTIN_OBSERVED_AT_MS,
        valid_until_ms: Some(BUILTIN_VALID_UNTIL_MS),
        retired_at_ms: None,
        source_id: BUILTIN_SOURCE_ID,
        catalog_version: BUILTIN_CATALOG_VERSION,
    },
    BuiltinPrice {
        origin: BillingOrigin::OfficialOpenAi,
        model: "gpt-4o-mini",
        class: BuiltinPriceClass::Exact,
        input_micro_usd_per_million: 150_000,
        output_micro_usd_per_million: 600_000,
        cache_read_micro_usd_per_million: Some(75_000),
        cache_write: BuiltinCacheWrite::NotBilled,
        effective_at_ms: BUILTIN_EFFECTIVE_AT_MS,
        observed_at_ms: BUILTIN_OBSERVED_AT_MS,
        valid_until_ms: Some(BUILTIN_VALID_UNTIL_MS),
        retired_at_ms: None,
        source_id: BUILTIN_SOURCE_ID,
        catalog_version: BUILTIN_CATALOG_VERSION,
    },
    // o3 was repriced: $2/M input, $0.50/M cached input, $8/M output.
    BuiltinPrice {
        origin: BillingOrigin::OfficialOpenAi,
        model: "o3",
        class: BuiltinPriceClass::Exact,
        input_micro_usd_per_million: 2_000_000,
        output_micro_usd_per_million: 8_000_000,
        cache_read_micro_usd_per_million: Some(500_000),
        cache_write: BuiltinCacheWrite::NotBilled,
        effective_at_ms: BUILTIN_EFFECTIVE_AT_MS,
        observed_at_ms: BUILTIN_OBSERVED_AT_MS,
        valid_until_ms: Some(BUILTIN_VALID_UNTIL_MS),
        retired_at_ms: None,
        source_id: BUILTIN_SOURCE_ID,
        catalog_version: BUILTIN_CATALOG_VERSION,
    },
    // o4-mini has a separate published cached-input rate ($0.275/M).
    BuiltinPrice {
        origin: BillingOrigin::OfficialOpenAi,
        model: "o4-mini",
        class: BuiltinPriceClass::Exact,
        input_micro_usd_per_million: 1_100_000,
        output_micro_usd_per_million: 4_400_000,
        cache_read_micro_usd_per_million: Some(275_000),
        cache_write: BuiltinCacheWrite::NotBilled,
        effective_at_ms: BUILTIN_EFFECTIVE_AT_MS,
        observed_at_ms: BUILTIN_OBSERVED_AT_MS,
        valid_until_ms: Some(BUILTIN_VALID_UNTIL_MS),
        retired_at_ms: None,
        source_id: BUILTIN_SOURCE_ID,
        catalog_version: BUILTIN_CATALOG_VERSION,
    },
    // Official Anthropic. Cache writes are billed: 1.25x input for the
    // 5-minute TTL, 2x input for the 1-hour TTL; cache reads are 0.1x.
    BuiltinPrice {
        origin: BillingOrigin::OfficialAnthropic,
        model: "claude-opus-4",
        class: BuiltinPriceClass::Exact,
        input_micro_usd_per_million: 15_000_000,
        output_micro_usd_per_million: 75_000_000,
        cache_read_micro_usd_per_million: Some(1_500_000),
        cache_write: BuiltinCacheWrite::TtlDependent {
            five_minutes: 18_750_000,
            one_hour: 30_000_000,
        },
        effective_at_ms: BUILTIN_EFFECTIVE_AT_MS,
        observed_at_ms: BUILTIN_OBSERVED_AT_MS,
        valid_until_ms: Some(BUILTIN_VALID_UNTIL_MS),
        retired_at_ms: None,
        source_id: BUILTIN_SOURCE_ID,
        catalog_version: BUILTIN_CATALOG_VERSION,
    },
    // Retired 2026-06-01; documented, never routed at the old price.
    BuiltinPrice {
        origin: BillingOrigin::OfficialAnthropic,
        model: "claude-sonnet-4",
        class: BuiltinPriceClass::Exact,
        input_micro_usd_per_million: 3_000_000,
        output_micro_usd_per_million: 15_000_000,
        cache_read_micro_usd_per_million: Some(300_000),
        cache_write: BuiltinCacheWrite::TtlDependent {
            five_minutes: 3_750_000,
            one_hour: 6_000_000,
        },
        effective_at_ms: BUILTIN_EFFECTIVE_AT_MS,
        observed_at_ms: BUILTIN_OBSERVED_AT_MS,
        valid_until_ms: Some(1_780_272_000_000),
        retired_at_ms: Some(1_780_272_000_000),
        source_id: BUILTIN_SOURCE_ID,
        catalog_version: BUILTIN_CATALOG_VERSION,
    },
    // Retired 2026-02-01; documented, never routed at the old price.
    BuiltinPrice {
        origin: BillingOrigin::OfficialAnthropic,
        model: "claude-haiku-3.5",
        class: BuiltinPriceClass::Exact,
        input_micro_usd_per_million: 800_000,
        output_micro_usd_per_million: 4_000_000,
        cache_read_micro_usd_per_million: Some(80_000),
        cache_write: BuiltinCacheWrite::TtlDependent {
            five_minutes: 1_000_000,
            one_hour: 1_600_000,
        },
        effective_at_ms: BUILTIN_EFFECTIVE_AT_MS,
        observed_at_ms: BUILTIN_OBSERVED_AT_MS,
        valid_until_ms: Some(1_769_904_000_000),
        retired_at_ms: Some(1_769_904_000_000),
        source_id: BUILTIN_SOURCE_ID,
        catalog_version: BUILTIN_CATALOG_VERSION,
    },
    // Official Google.
    BuiltinPrice {
        origin: BillingOrigin::OfficialGoogle,
        model: "gemini-2.0-flash",
        class: BuiltinPriceClass::Exact,
        input_micro_usd_per_million: 100_000,
        output_micro_usd_per_million: 400_000,
        cache_read_micro_usd_per_million: None,
        cache_write: BuiltinCacheWrite::NotBilled,
        effective_at_ms: BUILTIN_EFFECTIVE_AT_MS,
        observed_at_ms: BUILTIN_OBSERVED_AT_MS,
        valid_until_ms: Some(BUILTIN_VALID_UNTIL_MS),
        retired_at_ms: None,
        source_id: BUILTIN_SOURCE_ID,
        catalog_version: BUILTIN_CATALOG_VERSION,
    },
    // Official DeepSeek. The V4-era schedule is not documented as exact in
    // this table: the row is a CONSERVATIVE BOUND (2x the greatest last
    // published cache-miss/output rates across the audited V3-era schedules,
    // with write tokens bounded at the cache-miss rate) and resolves to a
    // ConservativeCeiling state. A hard cap may use the bound; no caller may
    // read it as a list price.
    BuiltinPrice {
        origin: BillingOrigin::OfficialDeepSeek,
        model: "deepseek-chat",
        class: BuiltinPriceClass::ConservativeSchedule,
        input_micro_usd_per_million: 560_000,
        output_micro_usd_per_million: 2_200_000,
        cache_read_micro_usd_per_million: Some(140_000),
        cache_write: BuiltinCacheWrite::Fixed(560_000),
        effective_at_ms: BUILTIN_EFFECTIVE_AT_MS,
        observed_at_ms: BUILTIN_OBSERVED_AT_MS,
        valid_until_ms: Some(BUILTIN_VALID_UNTIL_MS),
        retired_at_ms: None,
        source_id: BUILTIN_SOURCE_ID,
        catalog_version: BUILTIN_CATALOG_VERSION,
    },
];

/// One documented FAKTOR ROUTING PRIOR row (not a vendor claim).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuiltinPerformance {
    pub origin: BillingOrigin,
    pub model: &'static str,
    pub context_reliability: u8,
    pub coding_reliability: u8,
    pub estimated_latency_ms: u64,
}

/// The documented Faktor routing priors. Values are conservative relative
/// ordering estimates: frontier coding models sit in the high 80s/90s, small
/// flash/mini models lower, with latency roughly anti-correlated. They exist
/// so a fresh Balanced configuration has a real candidate above its default
/// quality floor instead of silently failing every route; durable verified
/// outcomes replace them per (provider, model, phase) once recorded.
pub const PERFORMANCE_TABLE: &[BuiltinPerformance] = &[
    BuiltinPerformance {
        origin: BillingOrigin::OfficialOpenAi,
        model: "gpt-4o",
        context_reliability: 90,
        coding_reliability: 90,
        estimated_latency_ms: 700,
    },
    BuiltinPerformance {
        origin: BillingOrigin::OfficialOpenAi,
        model: "gpt-4o-mini",
        context_reliability: 78,
        coding_reliability: 74,
        estimated_latency_ms: 500,
    },
    BuiltinPerformance {
        origin: BillingOrigin::OfficialOpenAi,
        model: "o3",
        context_reliability: 92,
        coding_reliability: 94,
        estimated_latency_ms: 2_000,
    },
    BuiltinPerformance {
        origin: BillingOrigin::OfficialOpenAi,
        model: "o4-mini",
        context_reliability: 86,
        coding_reliability: 88,
        estimated_latency_ms: 1_200,
    },
    BuiltinPerformance {
        origin: BillingOrigin::OfficialAnthropic,
        model: "claude-opus-4",
        context_reliability: 92,
        coding_reliability: 95,
        estimated_latency_ms: 1_500,
    },
    BuiltinPerformance {
        origin: BillingOrigin::OfficialAnthropic,
        model: "claude-sonnet-4",
        context_reliability: 92,
        coding_reliability: 93,
        estimated_latency_ms: 900,
    },
    BuiltinPerformance {
        origin: BillingOrigin::OfficialAnthropic,
        model: "claude-haiku-3.5",
        context_reliability: 82,
        coding_reliability: 78,
        estimated_latency_ms: 450,
    },
    BuiltinPerformance {
        origin: BillingOrigin::OfficialGoogle,
        model: "gemini-2.0-flash",
        context_reliability: 88,
        coding_reliability: 84,
        estimated_latency_ms: 350,
    },
    BuiltinPerformance {
        origin: BillingOrigin::OfficialDeepSeek,
        model: "deepseek-chat",
        context_reliability: 88,
        coding_reliability: 90,
        estimated_latency_ms: 1_100,
    },
];

/// The provenance of the whole frozen table (the table's own dates).
pub fn provenance() -> PricingProvenance {
    PricingProvenance {
        source_id: BUILTIN_SOURCE_ID.to_string(),
        effective_at_ms: Some(BUILTIN_EFFECTIVE_AT_MS),
        observed_at_ms: Some(BUILTIN_OBSERVED_AT_MS),
        valid_until_ms: Some(BUILTIN_VALID_UNTIL_MS),
        catalog_version: BUILTIN_CATALOG_VERSION.to_string(),
    }
}

/// The provenance of ONE row (every row carries its own dates, source id
/// and catalog version).
pub fn provenance_of(row: &BuiltinPrice) -> PricingProvenance {
    PricingProvenance {
        source_id: row.source_id.to_string(),
        effective_at_ms: Some(row.effective_at_ms),
        observed_at_ms: Some(row.observed_at_ms),
        valid_until_ms: row.valid_until_ms,
        catalog_version: row.catalog_version.to_string(),
    }
}

/// The ROW as a conditional [`PriceSchedule`]: applies the documented
/// validity window, service tier and cache-write TTL (unknown TTL + hard
/// budget → the maximum documented write tariff). Conservative-schedule
/// rows quote with [`PriceAuthority::ConservativeCeiling`] authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuiltinRowSchedule {
    pub row: BuiltinPrice,
    pub epoch: u64,
}

impl BuiltinRowSchedule {
    pub const fn new(row: BuiltinPrice, epoch: u64) -> Self {
        Self { row, epoch }
    }

    fn priced_quote(
        &self,
        ctx: &PricingContext,
        hard_budget: bool,
    ) -> Result<(PriceQuote, PriceAuthority), PriceUnavailable> {
        let row = &self.row;
        if row.is_retired() {
            return Err(PriceUnavailable::Retired);
        }
        if ctx.at_ms < row.effective_at_ms {
            // The row is not yet in force: no applicable tariff.
            return Err(PriceUnavailable::NoApplicableTariff);
        }
        if row.is_expired_at(ctx.at_ms) {
            return Err(PriceUnavailable::Expired);
        }
        if let Some(tier) = ctx.service_tier {
            if tier != ServiceTier::Standard {
                return Err(PriceUnavailable::NoApplicableTariff);
            }
        }
        let Some(cache_write) = row.cache_write.rate_for(ctx.cache_write_ttl, hard_budget) else {
            return Err(PriceUnavailable::NoApplicableTariff);
        };
        let quote = PriceQuote {
            input: MicroUsdPerMillionTokens(row.input_micro_usd_per_million),
            output: MicroUsdPerMillionTokens(row.output_micro_usd_per_million),
            cache_read: MicroUsdPerMillionTokens(row.cache_read_micro_usd_per_million.unwrap_or(0)),
            cache_write: MicroUsdPerMillionTokens(cache_write),
        };
        let authority = match row.class {
            BuiltinPriceClass::Exact => PriceAuthority::Exact,
            BuiltinPriceClass::ConservativeSchedule => PriceAuthority::ConservativeCeiling,
        };
        Ok((quote, authority))
    }

    fn snapshot(&self, quote: PriceQuote, authority: PriceAuthority) -> PricingSnapshot {
        let snap = match authority {
            PriceAuthority::ConservativeCeiling => PricingSnapshot::conservative_ceiling(
                quote,
                self.epoch,
                self.row.source_id.to_string(),
            ),
            _ => PricingSnapshot::exact(quote, self.epoch, self.row.source_id.to_string()),
        };
        match self.row.valid_until_ms {
            Some(until) => snap.with_valid_until(until),
            None => snap,
        }
    }
}

impl PriceSchedule for BuiltinRowSchedule {
    fn quote(&self, ctx: &PricingContext) -> Result<PricingSnapshot, PriceUnavailable> {
        let (quote, authority) = self.priced_quote(ctx, false)?;
        Ok(self.snapshot(quote, authority))
    }

    fn quote_for_hard_budget(
        &self,
        ctx: &PricingContext,
    ) -> Result<PricingSnapshot, PriceUnavailable> {
        let (quote, authority) = self.priced_quote(ctx, true)?;
        Ok(self.snapshot(quote, authority))
    }
}

/// Exact list-price lookup keyed by BILLING ORIGIN + model (the production
/// path): case-insensitive on the model, exact on the origin, and RETIRED
/// rows excluded. `None` for anything not in the documented table — callers
/// then stay [`PricingState::Unknown`].
pub fn lookup_by_origin(origin: BillingOrigin, model: &str) -> Option<BuiltinPrice> {
    lookup_documented_by_origin(origin, model).filter(|row| !row.is_retired())
}

/// Documented-row lookup INCLUDING retired rows (metadata/tests only; the
/// routing paths use [`lookup_by_origin`]).
pub fn lookup_documented_by_origin(origin: BillingOrigin, model: &str) -> Option<BuiltinPrice> {
    let model = model.trim().to_ascii_lowercase();
    TABLE
        .iter()
        .copied()
        .find(|row| row.origin == origin && row.model.to_ascii_lowercase() == model)
}

/// LEGACY family-keyed lookup kept for the raw `Provider::catalog_entry`
/// trait default (untouched adapters construct rows without config): maps the
/// canonical adapter family names onto their official billing origins and
/// otherwise misses. Config-built daemon providers NEVER use this — they
/// resolve [`BillingOrigin`] strictly and call [`lookup_by_origin`], so a
/// custom OpenAI-compatible endpoint can never inherit official prices.
///
/// The raw trait default can only build an EXACT row, so this shim also
/// excludes conservative-schedule and retired rows: those must resolve
/// through the billing-origin wrapper (which understands the pricing state).
pub fn lookup(family: &str, model: &str) -> Option<BuiltinPrice> {
    family_origin(family)
        .and_then(|origin| lookup_by_origin(origin, model))
        .filter(|row| row.class == BuiltinPriceClass::Exact)
}

fn family_origin(family: &str) -> Option<BillingOrigin> {
    match family.trim().to_ascii_lowercase().as_str() {
        "openai" => Some(BillingOrigin::OfficialOpenAi),
        "anthropic" => Some(BillingOrigin::OfficialAnthropic),
        "google" => Some(BillingOrigin::OfficialGoogle),
        "deepseek" => Some(BillingOrigin::OfficialDeepSeek),
        _ => None,
    }
}

/// Resolve one documented row into its route-time [`PricingState`] at
/// `at_ms`, using the HARD-BUDGET tariff selection (unknown TTL → maximum
/// documented write tariff) so the catalog never under-reserves:
///
/// - [`BuiltinPriceClass::Exact`] → `Known` at the max applicable tariff,
///   with the row's validity window attached;
/// - [`BuiltinPriceClass::ConservativeSchedule`] → `ConservativeCeiling`
///   (a documented bound, never a claimed exact price);
/// - retired/expired/not-yet-effective rows → `Unknown`.
pub fn pricing_state_at(row: &BuiltinPrice, at_ms: u64) -> PricingState {
    let schedule = BuiltinRowSchedule::new(*row, crate::catalog::CATALOG_FIRST_EPOCH);
    let ctx = PricingContext::at(at_ms);
    match schedule.quote_for_hard_budget(&ctx) {
        Ok(snapshot) => match snapshot.authority {
            PriceAuthority::Exact => PricingState::Known(snapshot),
            PriceAuthority::ConservativeCeiling => PricingState::ConservativeCeiling(snapshot),
            _ => PricingState::Unknown,
        },
        Err(_) => PricingState::Unknown,
    }
}

/// [`pricing_state_at`] at the current wall clock (the catalog resolution
/// path).
pub fn pricing_state_of(row: &BuiltinPrice) -> PricingState {
    pricing_state_at(row, unix_now_ms())
}

/// The exact per-million-token quote of one built-in row, with the
/// conservative max cache-write tariff (`cache_write` is max(TTL tariffs)
/// because the raw compatibility snapshot carries no context). DeepSeek
/// rows are conservative bounds; callers that need the authority use
/// [`pricing_state_at`].
pub fn quote_of(row: &BuiltinPrice) -> PriceQuote {
    PriceQuote {
        input: MicroUsdPerMillionTokens(row.input_micro_usd_per_million),
        output: MicroUsdPerMillionTokens(row.output_micro_usd_per_million),
        cache_read: MicroUsdPerMillionTokens(row.cache_read_micro_usd_per_million.unwrap_or(0)),
        cache_write: MicroUsdPerMillionTokens(row.cache_write.max_rate()),
    }
}

/// The documented Faktor routing prior of one (origin, model), when the
/// table documents it: a [`ModelPerformanceProfile`] carrying the
/// conservative-unknown authority and the frozen benchmark version.
pub fn performance_prior(origin: BillingOrigin, model: &str) -> Option<ModelPerformanceProfile> {
    let model = model.trim().to_ascii_lowercase();
    PERFORMANCE_TABLE
        .iter()
        .copied()
        .find(|row| row.origin == origin && row.model.to_ascii_lowercase() == model)
        .map(|row| ModelPerformanceProfile {
            prior: ModelPerformance {
                context_reliability: row.context_reliability,
                coding_reliability: row.coding_reliability,
                estimated_latency_ms: row.estimated_latency_ms,
                rate_limit_state: faktor_core::model::RateLimitState::Healthy,
            },
            authority: QualityAuthority::ConservativeUnknown,
            benchmark_version: PERFORMANCE_PRIOR_VERSION.to_string(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_rows_are_locked_with_quote_provenance_and_never_free() {
        // Every documented row: exact input/output lines, cache lines only
        // where published, cache_write never silently zero on a
        // write-billing provider, sub-$1/M prices kept exact (no
        // truncation to zero), metadata dated, provenance frozen.
        assert_eq!(TABLE.len(), 9);
        let mut seen: Vec<(BillingOrigin, &str)> = Vec::new();
        for row in TABLE {
            assert!(!seen.contains(&(row.origin, row.model)), "duplicate row");
            seen.push((row.origin, row.model));
            let q = quote_of(row);
            assert_eq!(
                q.input.0, row.input_micro_usd_per_million,
                "{} input line exact",
                row.model
            );
            assert_eq!(
                q.output.0, row.output_micro_usd_per_million,
                "{} output line exact",
                row.model
            );
            assert_eq!(
                q.cache_read.0,
                row.cache_read_micro_usd_per_million.unwrap_or(0),
                "{} cache_read line exact",
                row.model
            );
            assert!(!q.is_zero(), "{} must never read as free", row.model);
            assert_ne!(q.input.0, 0, "{} input price nonzero", row.model);
            assert_ne!(q.output.0, 0, "{} output price nonzero", row.model);
            // Every row carries its metadata dates + source identity.
            assert_eq!(row.source_id, BUILTIN_SOURCE_ID);
            assert_eq!(row.catalog_version, BUILTIN_CATALOG_VERSION);
            assert_eq!(row.catalog_version, "builtin-v2");
            assert!(row.effective_at_ms > 0);
            assert!(row.observed_at_ms >= row.effective_at_ms);
            assert!(row.valid_until_ms.is_some());
            let p = provenance_of(row);
            assert_eq!(p.effective_at_ms, Some(row.effective_at_ms));
            assert_eq!(p.observed_at_ms, Some(row.observed_at_ms));
            assert_eq!(p.valid_until_ms, row.valid_until_ms);
            assert_eq!(p.source_id, row.source_id);
            // Retired rows are documented but never returned as Exact.
            if row.is_retired() {
                assert_eq!(lookup_by_origin(row.origin, row.model), None);
                assert_eq!(
                    lookup_by_origin(row.origin, &row.model.to_ascii_uppercase()),
                    None
                );
                continue;
            }
            assert_eq!(
                lookup_documented_by_origin(row.origin, row.model),
                Some(*row),
                "documented lookup finds every row"
            );
            match row.class {
                BuiltinPriceClass::Exact => {
                    assert_eq!(
                        lookup_by_origin(row.origin, row.model),
                        Some(*row),
                        "active exact row is offered by origin"
                    );
                    assert_eq!(
                        lookup_by_origin(row.origin, &row.model.to_ascii_uppercase()),
                        Some(*row),
                        "origin lookup is case-insensitive on the model"
                    );
                    // The legacy family shim agrees for canonical family
                    // names (exact rows only).
                    let family = family_of(row.origin);
                    assert_eq!(lookup(family, row.model), Some(*row));
                }
                BuiltinPriceClass::ConservativeSchedule => {
                    // Conservative rows are offered by origin (the wrapper
                    // maps them to a ceiling) but NEVER through the raw
                    // exact-only family shim.
                    assert_eq!(
                        lookup_by_origin(row.origin, row.model),
                        Some(*row),
                        "active conservative row is offered by origin"
                    );
                    assert_eq!(lookup(family_of(row.origin), row.model), None);
                }
            }
        }
        // The audited goldens settle exactly from the stored rows.
        let gpt4o = quote_of(TABLE.iter().find(|r| r.model == "gpt-4o").unwrap());
        assert_eq!(gpt4o.input.0, 2_500_000, "$2.50/M input, exact");
        assert_eq!(gpt4o.cache_read.0, 1_250_000, "$1.25/M cache read");
        let mini = quote_of(TABLE.iter().find(|r| r.model == "gpt-4o-mini").unwrap());
        assert_eq!(mini.input.0, 150_000, "$0.15/M must NOT truncate to zero");
        assert_eq!(mini.output.0, 600_000, "$0.60/M must NOT truncate to zero");
        // o3's CURRENT repricing is locked (was $10/$40, then $2/$8 with a
        // $0.50/M cached-input line).
        let o3 = quote_of(TABLE.iter().find(|r| r.model == "o3").unwrap());
        assert_eq!(o3.input.0, 2_000_000);
        assert_eq!(o3.output.0, 8_000_000);
        assert_eq!(o3.cache_read.0, 500_000);
        // o4-mini has its separate published cached rate, never zero.
        let o4 = quote_of(TABLE.iter().find(|r| r.model == "o4-mini").unwrap());
        assert_eq!(o4.cache_read.0, 275_000);
    }

    fn family_of(origin: BillingOrigin) -> &'static str {
        match origin {
            BillingOrigin::OfficialOpenAi => "openai",
            BillingOrigin::OfficialAnthropic => "anthropic",
            BillingOrigin::OfficialGoogle => "google",
            BillingOrigin::OfficialDeepSeek => "deepseek",
            _ => panic!("builtin rows are official"),
        }
    }

    #[test]
    fn o3_and_o4_mini_cached_input_is_not_free() {
        // A cached input token is a priced category: 1M cached o3 input
        // tokens cost 500_000 microUSD (never 0), and o4-mini's published
        // cached rate bills 275_000. Zero cache pricing (the builtin-v1
        // lie) would understate both.
        let mut o3 = PricingSnapshot::exact(
            quote_of(TABLE.iter().find(|r| r.model == "o3").unwrap()),
            2,
            BUILTIN_SOURCE_ID.into(),
        );
        o3.valid_until_ms = Some(BUILTIN_VALID_UNTIL_MS);
        assert_eq!(o3.settle_cost(0, 1_000_000, 0, 0), Some(500_000));
        let o4 = quote_of(TABLE.iter().find(|r| r.model == "o4-mini").unwrap());
        assert_eq!(o4.cache_read.0, 275_000);
        assert_eq!(
            PricingSnapshot::exact(o4, 2, BUILTIN_SOURCE_ID.into()).settle_cost(0, 1_000_000, 0, 0),
            Some(275_000)
        );
    }

    #[test]
    fn anthropic_cache_write_is_priced_ttl_dependent_and_max_under_hard_budget() {
        // Anthropic bills cache CREATION. claude-opus-4: 5m write 1.25x
        // input ($18.75/M), 1h write 2x input ($30/M); read 0.1x.
        let row = *TABLE
            .iter()
            .find(|r| r.model == "claude-opus-4")
            .expect("documented row");
        let sched = BuiltinRowSchedule::new(row, 1);
        let ctx = PricingContext::at(BUILTIN_OBSERVED_AT_MS);
        // A plain quote with an unknown TTL cannot pick a tariff.
        assert_eq!(
            sched.quote(&ctx).unwrap_err(),
            PriceUnavailable::NoApplicableTariff
        );
        // A KNOWN TTL uses exactly its tariff.
        assert_eq!(
            sched
                .quote(&ctx.with_cache_write_ttl(CacheWriteTtl::FiveMinutes))
                .unwrap()
                .quote
                .unwrap()
                .cache_write,
            MicroUsdPerMillionTokens(18_750_000)
        );
        assert_eq!(
            sched
                .quote(&ctx.with_cache_write_ttl(CacheWriteTtl::OneHour))
                .unwrap()
                .quote
                .unwrap()
                .cache_write,
            MicroUsdPerMillionTokens(30_000_000)
        );
        // Unknown tariff + hard budget: the MAXIMUM documented tariff.
        let hard = sched.quote_for_hard_budget(&ctx).unwrap();
        assert_eq!(
            hard.quote.unwrap().cache_write,
            MicroUsdPerMillionTokens(30_000_000),
            "cache creation must never reserve zero or the cheaper TTL"
        );
        // The catalog state reserves the same conservative write tariff.
        let state = pricing_state_at(&row, BUILTIN_OBSERVED_AT_MS);
        assert_eq!(state.authority(), PriceAuthority::Exact);
        let snap = state.snapshot();
        assert_eq!(
            snap.quote.unwrap().cache_write,
            MicroUsdPerMillionTokens(30_000_000)
        );
        assert_eq!(snap.settle_cost(0, 0, 1_000_000, 0), Some(30_000_000));
        assert!(snap.valid_until_ms.is_some());
        // A non-standard service tier has no documented tariff: refuse.
        assert_eq!(
            sched
                .quote(&ctx.with_service_tier(ServiceTier::Priority))
                .unwrap_err(),
            PriceUnavailable::NoApplicableTariff
        );
    }

    #[test]
    fn retired_rows_and_expired_rows_are_unknown() {
        for model in ["claude-sonnet-4", "claude-haiku-3.5"] {
            let row = *TABLE.iter().find(|r| r.model == model).unwrap();
            assert!(row.is_retired());
            assert_eq!(
                pricing_state_at(&row, BUILTIN_OBSERVED_AT_MS),
                PricingState::Unknown,
                "{model} retired in 2026 must never resolve exact"
            );
        }
        // An ACTIVE row past its window is Unknown too (no ceiling on
        // exact rows: a stale exact quote is not a bound).
        let row = *TABLE.iter().find(|r| r.model == "gpt-4o").unwrap();
        assert_eq!(
            pricing_state_at(&row, row.valid_until_ms.unwrap()),
            PricingState::Unknown
        );
        assert_eq!(
            pricing_state_at(&row, row.valid_until_ms.unwrap() - 1).authority(),
            PriceAuthority::Exact
        );
    }

    #[test]
    fn deepseek_v4_row_is_a_conservative_bound_not_an_exact_claim() {
        let row = *TABLE.iter().find(|r| r.model == "deepseek-chat").unwrap();
        assert_eq!(row.class, BuiltinPriceClass::ConservativeSchedule);
        assert_eq!(
            pricing_state_at(&row, BUILTIN_OBSERVED_AT_MS).authority(),
            PriceAuthority::ConservativeCeiling,
            "an undocumented V4-era schedule never claims Exact"
        );
        // The bound is a bound: every line is at/above the last published
        // V3.2 rates (270k/1.1M with a 70k cache hit), never below.
        let q = quote_of(&row);
        assert!(q.input.0 >= 270_000);
        assert!(q.output.0 >= 1_100_000);
        assert!(q.cache_read.0 >= 70_000);
        assert!(q.cache_write.0 >= q.input.0, "write bounded at input");
        // The raw exact-only family shim never offers it.
        assert_eq!(lookup("deepseek", "deepseek-chat"), None);
        assert_eq!(
            lookup_by_origin(BillingOrigin::OfficialDeepSeek, "deepseek-chat").map(|r| r.class),
            Some(BuiltinPriceClass::ConservativeSchedule)
        );
    }

    #[test]
    fn lookup_never_invents_rows_and_origin_scoping_is_exact() {
        for (family, model) in [
            ("openai", "gpt-5"),
            ("openai", "gpt-4o-2024-08-06"), // dated snapshots are not rows
            ("ollama", "gpt-4o"),
            ("openai", "my-private-model"),
            ("openai", "claude-sonnet-4"), // origin-scoped: no leak
            ("", ""),
            ("deepseek", "deepseek-reasoner"),
            ("deepseek", "deepseek-chat"), // conservative class: not exact
            ("corp-proxy", "gpt-4o"),      // custom endpoint: never official
        ] {
            assert_eq!(
                lookup(family, model),
                None,
                "{family}/{model} must stay Unknown"
            );
        }
        // The origin-keyed path is scoped too: a custom endpoint or a
        // different official origin never sees another origin's row.
        assert_eq!(
            lookup_by_origin(BillingOrigin::OfficialDeepSeek, "gpt-4o"),
            None
        );
        assert_eq!(
            lookup_by_origin(BillingOrigin::CustomEndpoint, "gpt-4o"),
            None
        );
        assert_eq!(
            lookup_by_origin(BillingOrigin::Gateway, "deepseek-chat"),
            None
        );
        // Retired Anthropic IDs are never Exact lookups.
        for retired in ["claude-sonnet-4", "claude-haiku-3.5"] {
            assert_eq!(
                lookup_by_origin(BillingOrigin::OfficialAnthropic, retired),
                None,
                "{retired} is retired"
            );
        }
        let row = lookup_by_origin(BillingOrigin::OfficialAnthropic, "claude-opus-4")
            .expect("opus-4 is active");
        assert_eq!(row.cache_write.max_rate(), 30_000_000);
    }

    #[test]
    fn performance_priors_are_conservative_unknown_and_never_vendor_truths() {
        assert_eq!(PERFORMANCE_TABLE.len(), TABLE.len());
        for row in PERFORMANCE_TABLE {
            let profile = performance_prior(row.origin, row.model).expect("documented prior");
            assert_eq!(
                profile.authority,
                QualityAuthority::ConservativeUnknown,
                "{} prior authority",
                row.model
            );
            assert_eq!(profile.benchmark_version, PERFORMANCE_PRIOR_VERSION);
            assert!(profile.prior.context_reliability <= 100);
            assert!(profile.prior.coding_reliability <= 100);
            assert!(profile.prior.estimated_latency_ms > 0);
        }
        // A fresh Balanced configuration must find at least one documented
        // candidate at/above the 88 quality floor for every official origin
        // whose table row exists.
        for origin in [
            BillingOrigin::OfficialOpenAi,
            BillingOrigin::OfficialAnthropic,
            BillingOrigin::OfficialGoogle,
            BillingOrigin::OfficialDeepSeek,
        ] {
            let best = PERFORMANCE_TABLE
                .iter()
                .filter(|r| r.origin == origin)
                .map(|r| r.coding_reliability.max(r.context_reliability))
                .max();
            assert!(
                best.is_some_and(|q| q >= 88),
                "{origin:?} must have a documented prior above the balanced floor"
            );
        }
        // Custom/gateway/local never inherit an official prior.
        assert_eq!(
            performance_prior(BillingOrigin::CustomEndpoint, "gpt-4o"),
            None
        );
        assert_eq!(performance_prior(BillingOrigin::Gateway, "gpt-4o"), None);
    }

    #[test]
    fn provenance_roundtrips_on_the_wire_with_dated_rows() {
        let p = provenance();
        let back: PricingProvenance =
            serde_json::from_value(serde_json::to_value(&p).unwrap()).unwrap();
        assert_eq!(back, p);
        assert_eq!(back.catalog_version, "builtin-v2");
        assert_eq!(back.source_id, "faktor-builtin-v2");
        assert_eq!(back.effective_at_ms, Some(BUILTIN_EFFECTIVE_AT_MS));
        assert_eq!(back.observed_at_ms, Some(BUILTIN_OBSERVED_AT_MS));
        assert_eq!(back.valid_until_ms, Some(BUILTIN_VALID_UNTIL_MS));
        assert!(back.observed_at_ms >= back.effective_at_ms);
        // Legacy provenance JSON without the new/optional fields still
        // decodes (Option defaults).
        let legacy: PricingProvenance = serde_json::from_value(serde_json::json!({
            "source_id": "old", "catalog_version": "builtin-v1"
        }))
        .unwrap();
        assert_eq!(legacy.effective_at_ms, None);
        assert_eq!(legacy.observed_at_ms, None);
        assert_eq!(legacy.valid_until_ms, None);
    }
}
