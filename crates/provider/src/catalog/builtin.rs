//! Versioned built-in list-price table (audit wave-B item C + billing-origin
//! audit): documented list prices of officially-known models ONLY — nothing
//! invented, nothing extrapolated. Prices are microUSD per MILLION tokens
//! (the published $/M reading x 1e6) so sub-$1/M list prices ($0.15/M
//! gpt-4o-mini input, $0.10/M gemini flash input, ...) stay exact and can
//! never truncate to a free lie.
//!
//! Rows are keyed by [`BillingOrigin`] + model, NOT by transport family id:
//! a custom OpenAI-compatible proxy speaks the same wire protocol but bills
//! somewhere else, so it must never inherit official OpenAI list prices. The
//! legacy family-keyed [`lookup`] documents the raw-adapter compatibility
//! path (and is only reachable for adapters used OUTSIDE the config-built
//! daemon graph); the daemon resolves the real origin in config and uses
//! [`lookup_by_origin`].
//!
//! Cache-read rows are included ONLY where the audit records them as widely
//! published: gpt-4o 1_250_000, gpt-4o-mini 75_000, claude-sonnet-4
//! 300_000, claude-haiku-3.5 80_000, deepseek-chat 70_000 (five rows).
//! cache_write lines are published nowhere in the audited table: every row
//! carries `cache_write = 0` (the quote line simply never bills).
//!
//! Identity is versioned: `catalog_version = "builtin-v1"`,
//! `source_id = "faktor-builtin-v1"`. A static table is never observed on
//! the wire, so `effective_at_ms`/`observed_at_ms` stay 0; epoch 1 is the
//! catalog first-epoch (adapter default rows cut their snapshots at
//! [`crate::catalog::CATALOG_FIRST_EPOCH`]).
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
    BillingOrigin, MicroUsdPerMillionTokens, ModelPerformance, ModelPerformanceProfile, PriceQuote,
    QualityAuthority,
};

use crate::catalog::PricingProvenance;

/// Version of the frozen built-in table (bump the table AND this string
/// whenever a price changes, so old snapshots never claim a new version).
pub const BUILTIN_CATALOG_VERSION: &str = "builtin-v1";

/// Source id stamped on every built-in row's snapshot.
pub const BUILTIN_SOURCE_ID: &str = "faktor-builtin-v1";

/// Version string of the built-in Faktor routing priors (the performance
/// profiles below). Bump whenever a prior changes.
pub const PERFORMANCE_PRIOR_VERSION: &str = "faktor-routing-priors-v1";

/// Source id of the built-in performance priors (inspectable identity of a
/// conservative-unknown prior).
pub const PERFORMANCE_PRIOR_SOURCE_ID: &str = "faktor-builtin-priors-v1";

/// One documented list-price row (per-million-token microUSD).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuiltinPrice {
    /// The BILLING origin the row belongs to (never a transport family id).
    pub origin: BillingOrigin,
    /// The officially-known model name (matched case-insensitively).
    pub model: &'static str,
    pub input_micro_usd_per_million: u64,
    pub output_micro_usd_per_million: u64,
    /// Published cache-read line, when the audit records one; `None`
    /// otherwise (the line is 0 — never billed).
    pub cache_read_micro_usd_per_million: Option<u64>,
}

/// The frozen, versioned price table.
pub const TABLE: &[BuiltinPrice] = &[
    // Official OpenAI.
    BuiltinPrice {
        origin: BillingOrigin::OfficialOpenAi,
        model: "gpt-4o",
        input_micro_usd_per_million: 2_500_000,
        output_micro_usd_per_million: 10_000_000,
        cache_read_micro_usd_per_million: Some(1_250_000),
    },
    BuiltinPrice {
        origin: BillingOrigin::OfficialOpenAi,
        model: "gpt-4o-mini",
        input_micro_usd_per_million: 150_000,
        output_micro_usd_per_million: 600_000,
        cache_read_micro_usd_per_million: Some(75_000),
    },
    BuiltinPrice {
        origin: BillingOrigin::OfficialOpenAi,
        model: "o3",
        input_micro_usd_per_million: 10_000_000,
        output_micro_usd_per_million: 40_000_000,
        cache_read_micro_usd_per_million: None,
    },
    BuiltinPrice {
        origin: BillingOrigin::OfficialOpenAi,
        model: "o4-mini",
        input_micro_usd_per_million: 1_100_000,
        output_micro_usd_per_million: 4_400_000,
        cache_read_micro_usd_per_million: None,
    },
    // Official Anthropic.
    BuiltinPrice {
        origin: BillingOrigin::OfficialAnthropic,
        model: "claude-opus-4",
        input_micro_usd_per_million: 15_000_000,
        output_micro_usd_per_million: 75_000_000,
        cache_read_micro_usd_per_million: None,
    },
    BuiltinPrice {
        origin: BillingOrigin::OfficialAnthropic,
        model: "claude-sonnet-4",
        input_micro_usd_per_million: 3_000_000,
        output_micro_usd_per_million: 15_000_000,
        cache_read_micro_usd_per_million: Some(300_000),
    },
    BuiltinPrice {
        origin: BillingOrigin::OfficialAnthropic,
        model: "claude-haiku-3.5",
        input_micro_usd_per_million: 800_000,
        output_micro_usd_per_million: 4_000_000,
        cache_read_micro_usd_per_million: Some(80_000),
    },
    // Official Google.
    BuiltinPrice {
        origin: BillingOrigin::OfficialGoogle,
        model: "gemini-2.0-flash",
        input_micro_usd_per_million: 100_000,
        output_micro_usd_per_million: 400_000,
        cache_read_micro_usd_per_million: None,
    },
    // Official DeepSeek.
    BuiltinPrice {
        origin: BillingOrigin::OfficialDeepSeek,
        model: "deepseek-chat",
        input_micro_usd_per_million: 270_000,
        output_micro_usd_per_million: 1_100_000,
        cache_read_micro_usd_per_million: Some(70_000),
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

/// The provenance of the whole frozen table (identical for every row).
pub fn provenance() -> PricingProvenance {
    PricingProvenance {
        source_id: BUILTIN_SOURCE_ID.to_string(),
        effective_at_ms: 0,
        observed_at_ms: 0,
        catalog_version: BUILTIN_CATALOG_VERSION.to_string(),
    }
}

/// Exact list-price lookup keyed by BILLING ORIGIN + model (the production
/// path): case-insensitive on the model, exact on the origin. `None` for
/// anything not in the documented table — callers then stay
/// [`PricingState::Unknown`](faktor_core::model::PricingState::Unknown).
pub fn lookup_by_origin(origin: BillingOrigin, model: &str) -> Option<BuiltinPrice> {
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
pub fn lookup(family: &str, model: &str) -> Option<BuiltinPrice> {
    family_origin(family).and_then(|origin| lookup_by_origin(origin, model))
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

/// The exact per-million-token quote of one built-in row
/// (`cache_write` is never published in the audited table: always zero).
pub fn quote_of(row: &BuiltinPrice) -> PriceQuote {
    PriceQuote {
        input: MicroUsdPerMillionTokens(row.input_micro_usd_per_million),
        output: MicroUsdPerMillionTokens(row.output_micro_usd_per_million),
        cache_read: MicroUsdPerMillionTokens(row.cache_read_micro_usd_per_million.unwrap_or(0)),
        cache_write: MicroUsdPerMillionTokens(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_rows_are_locked_with_quote_provenance_and_never_free() {
        // Every documented row: exact input/output lines, cache lines only
        // where published, cache_write NEVER billed, sub-$1/M prices kept
        // exact (no truncation to zero), provenance frozen.
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
            assert_eq!(q.cache_write.0, 0, "{} never bills cache writes", row.model);
            assert!(!q.is_zero(), "{} must never read as free", row.model);
            assert_ne!(q.input.0, 0, "{} input price nonzero", row.model);
            assert_ne!(q.output.0, 0, "{} output price nonzero", row.model);
            let p = provenance();
            assert_eq!(p.source_id, BUILTIN_SOURCE_ID);
            assert_eq!(p.catalog_version, BUILTIN_CATALOG_VERSION);
            assert_eq!(p.effective_at_ms, 0);
            assert_eq!(p.observed_at_ms, 0);
            assert_eq!(
                lookup_by_origin(row.origin, row.model),
                Some(*row),
                "origin lookup finds every documented row"
            );
            assert_eq!(
                lookup_by_origin(row.origin, &row.model.to_ascii_uppercase()),
                Some(*row),
                "origin lookup is case-insensitive on the model"
            );
            // The legacy family shim agrees for canonical family names.
            let family = match row.origin {
                BillingOrigin::OfficialOpenAi => "openai",
                BillingOrigin::OfficialAnthropic => "anthropic",
                BillingOrigin::OfficialGoogle => "google",
                BillingOrigin::OfficialDeepSeek => "deepseek",
                _ => panic!("builtin rows are official"),
            };
            assert_eq!(lookup(family, row.model), Some(*row));
        }
        // The audited goldens settle exactly from the stored rows.
        let gpt4o = quote_of(TABLE.iter().find(|r| r.model == "gpt-4o").unwrap());
        assert_eq!(gpt4o.input.0, 2_500_000, "$2.50/M input, exact");
        assert_eq!(gpt4o.cache_read.0, 1_250_000, "$1.25/M cache read");
        let mini = quote_of(TABLE.iter().find(|r| r.model == "gpt-4o-mini").unwrap());
        assert_eq!(mini.input.0, 150_000, "$0.15/M must NOT truncate to zero");
        assert_eq!(mini.output.0, 600_000, "$0.60/M must NOT truncate to zero");
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
            ("corp-proxy", "gpt-4o"), // custom endpoint: never official prices
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
        assert_eq!(
            lookup_by_origin(BillingOrigin::OfficialDeepSeek, "deepseek-chat")
                .map(|r| r.output_micro_usd_per_million),
            Some(1_100_000)
        );
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
    fn provenance_roundtrips_on_the_wire() {
        let p = provenance();
        let back: PricingProvenance =
            serde_json::from_value(serde_json::to_value(&p).unwrap()).unwrap();
        assert_eq!(back, p);
        assert_eq!(back.catalog_version, "builtin-v1");
        assert_eq!(back.source_id, "faktor-builtin-v1");
    }
}
