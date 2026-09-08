//! Versioned built-in list-price table (audit wave-B item C): documented
//! list prices of officially-known models ONLY — nothing invented, nothing
//! extrapolated. Prices are microUSD per MILLION tokens (the published
//! $/M reading x 1e6) so sub-$1/M list prices ($0.15/M gpt-4o-mini input,
//! $0.10/M gemini flash input, ...) stay exact and can never truncate to a
//! free lie.
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

use faktor_core::model::{MicroUsdPerMillionTokens, PriceQuote};

use crate::catalog::PricingProvenance;

/// Version of the frozen built-in table (bump the table AND this string
/// whenever a price changes, so old snapshots never claim a new version).
pub const BUILTIN_CATALOG_VERSION: &str = "builtin-v1";

/// Source id stamped on every built-in row's snapshot.
pub const BUILTIN_SOURCE_ID: &str = "faktor-builtin-v1";

/// One documented list-price row (per-million-token microUSD).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuiltinPrice {
    /// Provider FAMILY the row belongs to (the adapter's `id()`).
    pub family: &'static str,
    /// The officially-known model name (matched case-insensitively).
    pub model: &'static str,
    pub input_micro_usd_per_million: u64,
    pub output_micro_usd_per_million: u64,
    /// Published cache-read line, when the audit records one; `None`
    /// otherwise (the line is 0 — never billed).
    pub cache_read_micro_usd_per_million: Option<u64>,
}

/// The frozen, versioned table.
pub const TABLE: &[BuiltinPrice] = &[
    // OpenAI (family "openai").
    BuiltinPrice {
        family: "openai",
        model: "gpt-4o",
        input_micro_usd_per_million: 2_500_000,
        output_micro_usd_per_million: 10_000_000,
        cache_read_micro_usd_per_million: Some(1_250_000),
    },
    BuiltinPrice {
        family: "openai",
        model: "gpt-4o-mini",
        input_micro_usd_per_million: 150_000,
        output_micro_usd_per_million: 600_000,
        cache_read_micro_usd_per_million: Some(75_000),
    },
    BuiltinPrice {
        family: "openai",
        model: "o3",
        input_micro_usd_per_million: 10_000_000,
        output_micro_usd_per_million: 40_000_000,
        cache_read_micro_usd_per_million: None,
    },
    BuiltinPrice {
        family: "openai",
        model: "o4-mini",
        input_micro_usd_per_million: 1_100_000,
        output_micro_usd_per_million: 4_400_000,
        cache_read_micro_usd_per_million: None,
    },
    // Anthropic (family "anthropic").
    BuiltinPrice {
        family: "anthropic",
        model: "claude-opus-4",
        input_micro_usd_per_million: 15_000_000,
        output_micro_usd_per_million: 75_000_000,
        cache_read_micro_usd_per_million: None,
    },
    BuiltinPrice {
        family: "anthropic",
        model: "claude-sonnet-4",
        input_micro_usd_per_million: 3_000_000,
        output_micro_usd_per_million: 15_000_000,
        cache_read_micro_usd_per_million: Some(300_000),
    },
    BuiltinPrice {
        family: "anthropic",
        model: "claude-haiku-3.5",
        input_micro_usd_per_million: 800_000,
        output_micro_usd_per_million: 4_000_000,
        cache_read_micro_usd_per_million: Some(80_000),
    },
    // Google (family "google").
    BuiltinPrice {
        family: "google",
        model: "gemini-2.0-flash",
        input_micro_usd_per_million: 100_000,
        output_micro_usd_per_million: 400_000,
        cache_read_micro_usd_per_million: None,
    },
    // DeepSeek (family "deepseek").
    BuiltinPrice {
        family: "deepseek",
        model: "deepseek-chat",
        input_micro_usd_per_million: 270_000,
        output_micro_usd_per_million: 1_100_000,
        cache_read_micro_usd_per_million: Some(70_000),
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

/// Exact list-price lookup by (provider-family, model): case-insensitive
/// on both keys (adapter family ids and model strings are compared
/// verbatim apart from casing/trim). `None` for anything not in the
/// documented table — callers then stay [`crate::catalog::PricingState::Unknown`].
pub fn lookup(family: &str, model: &str) -> Option<BuiltinPrice> {
    let family = family.trim().to_ascii_lowercase();
    let model = model.trim().to_ascii_lowercase();
    TABLE
        .iter()
        .copied()
        .find(|row| row.family == family && row.model.to_ascii_lowercase() == model)
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
        let mut seen: Vec<(&str, &str)> = Vec::new();
        for row in TABLE {
            assert!(!seen.contains(&(row.family, row.model)), "duplicate row");
            seen.push((row.family, row.model));
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
                lookup(row.family, row.model),
                Some(*row),
                "lookup finds every documented row"
            );
            assert_eq!(
                lookup(
                    &row.family.to_ascii_uppercase(),
                    &row.model.to_ascii_uppercase()
                ),
                Some(*row),
                "lookup is case-insensitive"
            );
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
    fn lookup_never_invents_rows() {
        for (family, model) in [
            ("openai", "gpt-5"),
            ("openai", "gpt-4o-2024-08-06"), // dated snapshots are not rows
            ("ollama", "gpt-4o"),
            ("openai", "my-private-model"),
            ("openai", "claude-sonnet-4"), // family-scoped: no leak
            ("", ""),
            ("deepseek", "deepseek-reasoner"),
        ] {
            assert_eq!(
                lookup(family, model),
                None,
                "{family}/{model} must stay Unknown"
            );
        }
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
