//! Model capabilities. Provider behavior is decided by these flags, never by
//! string-matching provider names. The agent reads capabilities; the adapters
//! set them; provider quirks stay inside adapters.
//!
//! # Pricing unit model (audit wave-B items A/B)
//!
//! Real list prices are **microUSD per million tokens**
//! ([`MicroUsdPerMillionTokens`]), kept exactly as providers publish them
//! ($/M x 1e6), so sub-$1/M prices ($0.50/M = 500_000) are representable and
//! never truncate to zero. Cost is computed as one ceiling division over the
//! summed category numerators ([`PriceQuote::quote_cost_micro`]).
//!
//! Prices and performance are separate types: [`PriceQuote`] is money,
//! [`ModelPerformance`] is reliability/latency, and every price state is
//! decided by an explicit [`PriceAuthority`] — a zero quote is never inferred
//! to be a "local free model", and "unknown price" never collapses to zero.
//! [`PricingSnapshot`] is the route-time freeze settlement prices usage
//! against.
//!
//! [`ModelEconomics`]/[`MicroUsdPerToken`] remain as the LEGACY per-token
//! estimate surface the router's internal candidate scoring and untouched
//! callers compile against (whole-microUSD-per-token values, numerically
//! equal to USD per million tokens only for prices that are exact multiples
//! of $1/M). New price knowledge must enter through
//! [`MicroUsdPerMillionTokens`]/[`PriceQuote`], never through the legacy
//! per-token ingestion (which cannot represent sub-$1/M prices).

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningMode {
    Off,
    Low,
    Medium,
    High,
}

/// Discovered per model. Unknown fields default conservatively (no tools,
/// no thinking, small context) so an unprobed model fails safe, not loud.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ModelCapabilities {
    /// Total context window in tokens.
    pub context: usize,
    /// Max output tokens.
    pub max_output: usize,
    pub tools: bool,
    pub parallel_tools: bool,
    pub thinking: bool,
    pub vision: bool,
    pub json_schema: bool,
    pub streaming: bool,
    pub embeddings: bool,
    pub reasoning: bool,
}

impl Default for ModelCapabilities {
    fn default() -> Self {
        Self {
            context: 32_768,
            max_output: 4_096,
            tools: false,
            parallel_tools: false,
            thinking: false,
            vision: false,
            json_schema: false,
            streaming: true,
            embeddings: false,
            reasoning: false,
        }
    }
}

impl ModelCapabilities {
    /// A deliberately conservative small-model profile (Ollama 32K class).
    pub fn small_local() -> Self {
        Self {
            context: 32_768,
            max_output: 4_096,
            tools: true,
            parallel_tools: false,
            thinking: true,
            vision: false,
            json_schema: false,
            streaming: true,
            embeddings: true,
            reasoning: false,
        }
    }

    pub fn supports_tools(&self) -> bool {
        self.tools
    }

    pub fn supports_parallel_tools(&self) -> bool {
        self.tools && self.parallel_tools
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_conservative_fail_safe() {
        let c = ModelCapabilities::default();
        assert!(!c.tools);
        assert!(!c.thinking);
        assert!(c.context > 0);
        assert!(!c.supports_parallel_tools());
    }

    #[test]
    fn capabilities_decide_agent_behavior_without_provider_names() {
        // The agent must branch on capabilities; this test locks the shape.
        let with_tools = ModelCapabilities {
            tools: true,
            parallel_tools: true,
            ..Default::default()
        };
        assert!(with_tools.supports_tools());
        assert!(with_tools.supports_parallel_tools());
        let no_tools = ModelCapabilities {
            tools: false,
            parallel_tools: true,
            ..Default::default()
        };
        assert!(!no_tools.supports_tools());
        assert!(
            !no_tools.supports_parallel_tools(),
            "parallel requires tools"
        );
    }

    #[test]
    fn small_local_profile_matches_32k_budget_math() {
        let c = ModelCapabilities::small_local();
        assert_eq!(c.context, 32_768);
        assert!(c.tools);
        assert!(c.embeddings);
        assert!(c.context > c.max_output);
    }

    #[test]
    fn json_roundtrip_preserves_all_fields() {
        let c = ModelCapabilities::small_local();
        let v = serde_json::to_value(&c).unwrap();
        let back: ModelCapabilities = serde_json::from_value(v).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn unknown_capability_flags_default_to_conservative() {
        // A provider that omits fields must yield a safe profile.
        let v = serde_json::json!({"context": 8192});
        let c: ModelCapabilities = serde_json::from_value(v).unwrap();
        assert_eq!(c.context, 8192);
        assert!(!c.tools);
        assert!(!c.vision);
        assert!(!c.thinking);
        assert_eq!(c.max_output, 4096);
    }

    #[test]
    fn huge_context_values_roundtrip_and_are_clamped_by_callers() {
        // A hostile/misconfigured provider advertising usize::MAX context is
        // preserved on the wire; the context engine is responsible for
        // clamping (tested in faktor-context).
        let raw = serde_json::json!({"context": u64::MAX});
        let c: ModelCapabilities = serde_json::from_value(raw).unwrap();
        assert_eq!(c.context, usize::MAX);
        let back = serde_json::to_value(c).unwrap();
        assert_eq!(back["context"], serde_json::Value::from(u64::MAX));
        // Negative context is rejected outright (must not wrap).
        let raw = serde_json::json!({"context": -1});
        assert!(serde_json::from_value::<ModelCapabilities>(raw).is_err());
    }

    #[test]
    fn task_class_and_risk_bucket_wire_roundtrip_and_cover_all_values() {
        // Additive outcome-learning dimensions: every variant serializes to
        // its snake_case string and roundtrips; ALL arrays are exhaustive.
        for c in TaskClass::ALL {
            let v = serde_json::to_value(c).unwrap();
            assert!(v.is_string());
            assert_eq!(serde_json::from_value::<TaskClass>(v).unwrap(), c);
        }
        for r in RiskBucket::ALL {
            let v = serde_json::to_value(r).unwrap();
            assert!(v.is_string());
            assert_eq!(serde_json::from_value::<RiskBucket>(v).unwrap(), r);
        }
        assert_eq!(serde_json::to_value(TaskClass::Hard).unwrap(), "hard");
        assert_eq!(serde_json::to_value(RiskBucket::High).unwrap(), "high");
        assert_eq!(TaskClass::ALL.len(), 3);
        assert_eq!(RiskBucket::ALL.len(), 3);
        // Unknown wire values are rejected loudly (a corrupt stat row must
        // never deserialize into a guessed bucket).
        assert!(serde_json::from_str::<TaskClass>("\"impossible\"").is_err());
        assert!(serde_json::from_str::<RiskBucket>("\"impossible\"").is_err());
    }

    #[test]
    fn routing_mode_wire_roundtrip_and_defaults() {
        // Economy serializes as the bare string "economy"; Pinned as the
        // tagged {"pinned": {provider, model}} object — a config field of
        // this type accepts both JSON shapes.
        let economy = RoutingMode::Economy;
        let v = serde_json::to_value(&economy).unwrap();
        assert_eq!(v, serde_json::json!("economy"));
        assert_eq!(serde_json::from_value::<RoutingMode>(v).unwrap(), economy);
        let pinned = RoutingMode::Pinned {
            provider: "deepseek".into(),
            model: "deepseek-chat".into(),
        };
        let v = serde_json::to_value(&pinned).unwrap();
        assert_eq!(
            v,
            serde_json::json!({"pinned": {"provider": "deepseek", "model": "deepseek-chat"}})
        );
        assert_eq!(serde_json::from_value::<RoutingMode>(v).unwrap(), pinned);
        assert!(economy.is_economy());
        assert!(!pinned.is_economy());
        assert_eq!(pinned.pinned(), Some(("deepseek", "deepseek-chat")));
        assert_eq!(economy.pinned(), None);
    }

    #[test]
    fn hostile_routing_mode_values_are_rejected() {
        for bad in [
            serde_json::json!("auto"),
            serde_json::json!({"pinned": {"provider": "p"}}), // model missing
            serde_json::json!({"economy": {}}),
            serde_json::json!(42),
        ] {
            assert!(
                serde_json::from_value::<RoutingMode>(bad).is_err(),
                "hostile routing mode must be rejected"
            );
        }
    }

    // ---- price wrapper (audit wave-B item A): microUSD PER MILLION tokens
    // is the money unit of every real list price; sub-$1/M prices never
    // truncate to zero and every cost is one ceiling division. ----

    #[test]
    fn half_dollar_per_million_is_representable_and_costs_500k_at_one_million() {
        // The wave-A unit (microUSD per token) truncated 500_000 microUSD/
        // million tokens ($0.50/M) to ZERO at ingestion: a paid model read
        // as free. The per-million-token wrapper keeps the list price
        // EXACTLY, and 1M tokens at $0.50/M must cost $0.50 = 500_000
        // microUSD (the audit's price_050 golden).
        let half = MicroUsdPerMillionTokens(500_000);
        assert!(!half.is_zero(), "price_050 must never read as free");
        assert_eq!(half.cost_ceil_micro(1_000_000), 500_000);
        assert_eq!(
            MicroUsdPerMillionTokens::from_dollars_per_million(15).cost_ceil_micro(1_000_000),
            15_000_000,
            "$15/M x 1M tokens = $15 exactly"
        );
    }

    #[test]
    fn one_cent_per_million_is_representable_and_costs_10k_at_one_million() {
        // price_001: $0.01/M = 10_000 microUSD per million tokens. Under
        // the per-token unit this truncated to zero (free); under the
        // per-million unit 1M tokens cost exactly 10_000 microUSD.
        let cent = MicroUsdPerMillionTokens(10_000);
        assert!(!cent.is_zero(), "price_001 must never read as free");
        assert_eq!(cent.cost_ceil_micro(1_000_000), 10_000);
        // A 1-token call at any positive price costs at least 1 microUSD
        // (ceiling, never a free lie, never understated).
        assert_eq!(cent.cost_ceil_micro(1), 1);
        assert_eq!(cent.cost_ceil_micro(999_999), 10_000);
        assert_eq!(cent.cost_ceil_micro(1_000), 10);
    }

    #[test]
    fn per_line_cost_is_a_single_ceiling_division_never_truncation() {
        // $15/M x 100k tokens = 1_500_000 microUSD — exact (the audit's
        // third golden), and non-multiple usage rounds UP once, per line.
        let p = MicroUsdPerMillionTokens::from_dollars_per_million(15);
        assert_eq!(p.cost_ceil_micro(100_000), 1_500_000);
        assert_eq!(p.cost_ceil_micro(999_999), 14_999_985, "exact multiple");
        assert_eq!(p.cost_ceil_micro(1), 15, "1 token at 15 micro/token");
        // A $1/M price over 500_001 tokens costs ceil(500_001 x 1e6 / 1e6)
        // = 500_001 micro; the per-token reading must agree exactly when
        // the price is a whole-dollar-per-M multiple.
        assert_eq!(
            MicroUsdPerMillionTokens::from_dollars_per_million(1).cost_ceil_micro(500_001),
            500_001
        );
    }

    #[test]
    fn hostile_magnitudes_saturate_never_panic_or_understate() {
        let max = MicroUsdPerMillionTokens(u64::MAX);
        assert_eq!(
            max.cost_ceil_micro(u64::MAX),
            u64::MAX,
            "hostile magnitude saturates, never panics"
        );
        assert_eq!(max.cost_ceil_micro(0), 0, "zero usage costs zero");
        assert_eq!(MicroUsdPerMillionTokens::ZERO.cost_ceil_micro(u64::MAX), 0);
        // u128 numerator keeps u64 x u64 exact before the single division.
        assert_eq!(
            MicroUsdPerMillionTokens(u64::MAX).cost_ceil_micro(1_000_000),
            u64::MAX
        );
    }

    #[test]
    fn money_and_latency_units_are_distinct_types() {
        // MicroUsdPerMillionTokens is a distinct nominal type, never an
        // alias, so a price can never reach latency math silently.
        let price_name = std::any::type_name::<MicroUsdPerMillionTokens>();
        assert_ne!(price_name, std::any::type_name::<u64>());
        assert_eq!(
            std::mem::size_of::<MicroUsdPerMillionTokens>(),
            std::mem::size_of::<u64>()
        );
        let price = MicroUsdPerMillionTokens::from_dollars_per_million(15);
        let latency_ms: u64 = 500;
        let _ = (price, latency_ms);
    }

    #[test]
    fn per_million_price_serde_is_transparent_and_roundtrips() {
        let p = MicroUsdPerMillionTokens(500_000);
        let v = serde_json::to_value(p).unwrap();
        assert_eq!(v, serde_json::json!(500_000));
        assert_eq!(
            serde_json::from_value::<MicroUsdPerMillionTokens>(v).unwrap(),
            p
        );
        // Hostile magnitudes survive the wire (clamping is caller-side).
        assert_eq!(
            serde_json::from_value::<MicroUsdPerMillionTokens>(serde_json::json!(u64::MAX))
                .unwrap(),
            MicroUsdPerMillionTokens(u64::MAX)
        );
        assert!(serde_json::from_value::<MicroUsdPerMillionTokens>(serde_json::json!(-1)).is_err());
        // Named constructors document the reading at the boundary.
        assert_eq!(
            MicroUsdPerMillionTokens::from_dollars_per_million(2),
            MicroUsdPerMillionTokens(2_000_000)
        );
        assert_eq!(
            MicroUsdPerMillionTokens::from_dollars_per_million(15).to_string(),
            "15000000"
        );
    }

    // ---- PriceQuote (audit item A): category numerators are summed in
    // u128 BEFORE ONE division with ceil; cache lines bill at their own
    // lines. ----

    fn quote(input: u64, output: u64) -> PriceQuote {
        PriceQuote {
            input: MicroUsdPerMillionTokens(input),
            output: MicroUsdPerMillionTokens(output),
            cache_read: MicroUsdPerMillionTokens::ZERO,
            cache_write: MicroUsdPerMillionTokens::ZERO,
        }
    }

    #[test]
    fn quote_settles_mixed_categories_with_one_rounding_at_the_total() {
        // 100k input at $15/M + 2k output at $60/M == 1_500_000 + 120_000
        // == 1_620_000 microUSD — the audit's settlement-truth identity,
        // now via the summed-numerator ceiling formula.
        let q = quote(15_000_000, 60_000_000);
        assert_eq!(
            q.quote_cost_micro(TokenUsage {
                uncached_input_tokens: 100_000,
                cache_read_tokens: 0,
                cache_write_tokens: 0,
                output_tokens: 2_000,
            }),
            1_620_000
        );
        assert_eq!(q.quote_cost_micro(TokenUsage::ZERO), 0);
    }

    #[test]
    fn quote_bills_cache_lines_at_their_own_price_lines() {
        let q = PriceQuote {
            input: quote(15_000_000, 60_000_000).input,
            output: quote(15_000_000, 60_000_000).output,
            cache_read: MicroUsdPerMillionTokens(3_000_000),
            cache_write: MicroUsdPerMillionTokens(7_000_000),
        };
        // 10k cache reads at $3/1M + 4k writes at $7/1M + 2k out at $60/1M.
        assert_eq!(
            q.quote_cost_micro(TokenUsage::new(0, 10_000, 4_000, 2_000)),
            30_000 + 28_000 + 120_000
        );
        // Every line stacks: uncached input + cache lines + output.
        assert_eq!(
            q.quote_cost_micro(TokenUsage::new(1_000, 500, 200, 100)),
            15_000 + 1_500 + 1_400 + 6_000
        );
    }

    #[test]
    fn quote_rounds_once_at_the_total_not_per_line() {
        // Two lines whose exact costs are both below one microUSD sum to
        // one microUSD total: per-line ceil would charge 2, the summed-
        // numerator ceiling charges exactly 1 — the audit's "round once at
        // the total" rule (never overcharge by accumulating ceilings).
        let q = PriceQuote {
            input: MicroUsdPerMillionTokens(1),
            cache_read: MicroUsdPerMillionTokens(1),
            ..quote(0, 0)
        };
        assert_eq!(
            q.quote_cost_micro(TokenUsage::new(1, 1, 0, 0)),
            1,
            "ceil(2 microUSD-numerator / 1e6) == 1, never 2"
        );
        // And a single 1-micro line over one token still costs 1 (never 0).
        assert_eq!(quote(1, 0).quote_cost_micro(TokenUsage::new(1, 0, 0, 0)), 1);
        assert_eq!(
            quote(1, 0).quote_cost_micro(TokenUsage::new(999_999, 0, 0, 0)),
            1,
            "ceil(999_999/1e6) == 1"
        );
    }

    #[test]
    fn quote_huge_magnitudes_saturate_never_panic() {
        let q = PriceQuote {
            input: MicroUsdPerMillionTokens(u64::MAX),
            output: MicroUsdPerMillionTokens(u64::MAX),
            cache_read: MicroUsdPerMillionTokens(u64::MAX),
            cache_write: MicroUsdPerMillionTokens(u64::MAX),
        };
        assert_eq!(
            q.quote_cost_micro(TokenUsage::new(u64::MAX, u64::MAX, u64::MAX, u64::MAX)),
            u64::MAX
        );
        assert_eq!(
            q.quote_cost_micro(TokenUsage::new(1, 0, 0, 0)),
            18_446_744_073_710,
            "u64::MAX micro/M over one token == ceil((2^64-1)/1e6)"
        );
    }

    // ---- PricingSnapshot (audit items A/B): settlement math is category-
    // exact over an explicit quote, and authority is NEVER inferred from
    // numbers. ----

    fn known_snapshot() -> PricingSnapshot {
        PricingSnapshot::exact(quote(15_000_000, 60_000_000), 7, "test-source".to_string())
    }

    #[test]
    fn known_snapshot_settles_exactly() {
        let snap = known_snapshot();
        assert_eq!(snap.authority, PriceAuthority::Exact);
        assert_eq!(
            snap.settle_cost(100_000, 0, 0, 2_000),
            Some(1_620_000),
            "the audit's settlement-truth identity"
        );
        assert_eq!(snap.settle_cost(0, 0, 0, 0), Some(0));
    }

    #[test]
    fn snapshot_settles_sub_dollar_list_prices_exactly() {
        // price_050: $0.50/M x 1M tokens == 500_000 microUSD — the wave-A
        // per-token truncation made this settle to ZERO; the quote model
        // settles the real number.
        let half = PricingSnapshot::exact(quote(500_000, 0), 1, "s".into());
        assert_eq!(half.settle_cost(1_000_000, 0, 0, 0), Some(500_000));
        // price_001: $0.01/M x 1M tokens == 10_000 microUSD.
        let cent = PricingSnapshot::exact(quote(10_000, 0), 1, "s".into());
        assert_eq!(cent.settle_cost(1_000_000, 0, 0, 0), Some(10_000));
        assert_eq!(cent.settle_cost(1, 0, 0, 0), Some(1));
        // $15/M x 100k tokens == 1_500_000 microUSD (audit golden).
        let fifteen = PricingSnapshot::exact(quote(15_000_000, 0), 1, "s".into());
        assert_eq!(fifteen.settle_cost(100_000, 0, 0, 0), Some(1_500_000));
    }

    #[test]
    fn local_zero_snapshot_is_an_authoritative_zero_quote() {
        // LocalZero = quote Some(PriceQuote::ZERO) + authority LocalZero.
        let snap = PricingSnapshot::local_zero(1, "ollama".into());
        assert_eq!(snap.authority, PriceAuthority::LocalZero);
        assert_eq!(snap.quote, Some(PriceQuote::ZERO));
        assert!(snap.is_local_zero());
        assert_eq!(
            snap.settle_cost(u64::MAX, u64::MAX, u64::MAX, u64::MAX),
            Some(0),
            "an authoritative local-zero profile settles to an honest 0"
        );
        let roundtrip: PricingSnapshot =
            serde_json::from_value(serde_json::to_value(&snap).unwrap()).unwrap();
        assert!(roundtrip.is_local_zero());
        assert_eq!(roundtrip.settle_cost(5, 0, 0, 0), Some(0));
    }

    #[test]
    fn unknown_snapshot_never_produces_a_number() {
        // Unknown = quote None + authority Unknown; settlement must refuse
        // every fabricated total — zero included.
        let snap = PricingSnapshot::unknown(1, "no-catalog".into());
        assert_eq!(snap.authority, PriceAuthority::Unknown);
        assert_eq!(snap.quote, None);
        assert!(!snap.is_local_zero());
        assert_eq!(snap.settle_cost(100_000, 0, 0, 2_000), None);
        assert_eq!(snap.settle_cost(0, 0, 0, 0), None, "zero usage too");
    }

    #[test]
    fn zero_numeric_quote_never_becomes_local_zero_and_unknown_stays_unknown() {
        // An EXACT quote of all-zero prices is a real (if odd) price
        // statement, NOT evidence of a local runtime: authority stays
        // Exact and is_local_zero() is false. Only an explicit
        // PriceAuthority::LocalZero names a local zero.
        let zero_exact = PricingSnapshot::exact(PriceQuote::ZERO, 1, "s".into());
        assert_eq!(zero_exact.authority, PriceAuthority::Exact);
        assert!(
            !zero_exact.is_local_zero(),
            "zero numbers are not LocalZero"
        );
        assert_eq!(zero_exact.settle_cost(u64::MAX, 0, 0, 0), Some(0));
        // Hostile: an UNKNOWN authority carrying a zero quote (a JSON
        // fabricator's best attempt at free) must still refuse every total:
        // authority decides, numbers never do.
        let forged = PricingSnapshot {
            quote: Some(PriceQuote::ZERO),
            authority: PriceAuthority::Unknown,
            epoch: 0,
            source_id: "forged".into(),
            valid_until_ms: None,
            conservative_ceiling: None,
        };
        assert!(!forged.is_local_zero());
        assert_eq!(
            forged.settle_cost(u64::MAX, u64::MAX, u64::MAX, u64::MAX),
            None
        );
        assert_eq!(forged.settle_cost(0, 0, 0, 0), None);
        // And a hostile JSON body that drops the quote under LocalZero can
        // never fabricate a zero either: LocalZero without a quote settles
        // to None, not to Some(0).
        let hostile: PricingSnapshot = serde_json::from_value(serde_json::json!({
            "authority": "local_zero", "epoch": 1, "source_id": "x",
        }))
        .unwrap();
        assert!(hostile.is_local_zero());
        assert_eq!(hostile.quote, None);
        assert_eq!(hostile.settle_cost(1_000_000, 0, 0, 0), None);
    }

    #[test]
    fn hostile_magnitudes_saturate_never_panic() {
        let snap = PricingSnapshot::exact(
            PriceQuote {
                input: MicroUsdPerMillionTokens(u64::MAX),
                output: MicroUsdPerMillionTokens(u64::MAX),
                cache_read: MicroUsdPerMillionTokens(u64::MAX),
                cache_write: MicroUsdPerMillionTokens(u64::MAX),
            },
            0,
            "hostile".into(),
        );
        assert_eq!(
            snap.settle_cost(u64::MAX, u64::MAX, u64::MAX, u64::MAX),
            Some(u64::MAX)
        );
    }

    #[test]
    fn authority_and_quote_roundtrip_without_any_inference() {
        for authority in [
            PriceAuthority::Exact,
            PriceAuthority::ConservativeCeiling,
            PriceAuthority::LocalZero,
            PriceAuthority::Unknown,
        ] {
            let snap = PricingSnapshot {
                quote: (authority != PriceAuthority::Unknown).then_some(quote(1, 2)),
                authority,
                epoch: 9,
                source_id: "src".into(),
                valid_until_ms: None,
                conservative_ceiling: None,
            };
            let v = serde_json::to_value(&snap).unwrap();
            let back: PricingSnapshot = serde_json::from_value(v).unwrap();
            assert_eq!(back, snap, "{authority:?} must round-trip untouched");
            assert_eq!(back.authority, authority);
        }
        // Unknown is the wire-default authority (fail closed): a legacy
        // wave-A snapshot JSON (six per-token fields, no quote/authority/
        // source_id) decodes as Unknown + no quote — settlement refuses,
        // and an old "local_zero" source NEVER becomes an authoritative
        // zero (that would be inference from a numeric/source field).
        for legacy in [
            serde_json::json!({
                "input_micro_per_token": 0, "output_micro_per_token": 0,
                "cache_read_micro_per_token": 0, "cache_write_micro_per_token": 0,
                "pricing_epoch": 0, "source": "local_zero",
            }),
            serde_json::json!({
                "input_micro_per_token": 15, "output_micro_per_token": 60,
                "cache_read_micro_per_token": 3, "cache_write_micro_per_token": 7,
                "pricing_epoch": 1, "source": "known",
            }),
        ] {
            let snap: PricingSnapshot = serde_json::from_value(legacy).unwrap();
            assert_eq!(snap.authority, PriceAuthority::Unknown);
            assert_eq!(snap.quote, None);
            assert_eq!(snap.settle_cost(100_000, 0, 0, 2_000), None);
        }
        // Authority wire spelling is frozen snake_case.
        let v = serde_json::to_value(PriceAuthority::ConservativeCeiling).unwrap();
        assert_eq!(v, serde_json::json!("conservative_ceiling"));
        assert!(serde_json::from_value::<PriceAuthority>(serde_json::json!("free!")).is_err());
    }

    #[test]
    fn route_decision_json_roundtrip_with_and_without_snapshot() {
        let decision = RouteDecision {
            provider: "p".into(),
            model: "m".into(),
            estimated_cost_micro: 10,
            estimated_latency_ms: 5,
            reasoning: "r".into(),
            considered: 2,
            source: ModelSource::ProviderCatalog,
            pricing_snapshot: Some(known_snapshot()),
        };
        let v = serde_json::to_value(&decision).unwrap();
        assert_eq!(
            v["pricing_snapshot"]["authority"],
            serde_json::json!("exact")
        );
        let back: RouteDecision = serde_json::from_value(v.clone()).unwrap();
        assert_eq!(back, decision);
        // Additive wire default: a pre-snapshot decision JSON (no snapshot
        // field) parses to None — never an error, never a fabricated price.
        let mut legacy = v.clone();
        legacy.as_object_mut().unwrap().remove("pricing_snapshot");
        let back: RouteDecision = serde_json::from_value(legacy).unwrap();
        assert_eq!(back.pricing_snapshot, None);
    }

    // ---- performance/price split (audit item B) ----

    #[test]
    fn economics_projects_performance_without_any_price() {
        // The legacy economics blob carries both money and performance; the
        // named performance projection exposes ONLY the performance dims
        // the audit keeps (context/coding reliability + latency), so price
        // and performance stop being one untyped integer soup.
        let e = ModelEconomics {
            input_price_per_mtok: MicroUsdPerToken(15),
            output_price_per_mtok: MicroUsdPerToken(60),
            context_reliability: 88,
            coding_reliability: 92,
            estimated_latency_ms: 700,
            ..Default::default()
        };
        let perf = e.performance();
        assert_eq!(perf.context_reliability, 88);
        assert_eq!(perf.coding_reliability, 92);
        assert_eq!(perf.estimated_latency_ms, 700);
        let p = ModelPerformance::default();
        assert_eq!(p.context_reliability, 50);
        assert_eq!(p.coding_reliability, 50);
        assert_eq!(p.estimated_latency_ms, 1000);
        // A zero-price economics (the old local marker) projects the SAME
        // conservative performance — performance never depends on price.
        let zero = ModelEconomics::default();
        assert_eq!(zero.performance(), ModelPerformance::default());
        // And the projection round-trips onto the economics blob.
        let back = e.with_performance(perf);
        assert_eq!(back.context_reliability, 88);
        assert_eq!(back.coding_reliability, 92);
        assert_eq!(back.estimated_latency_ms, 700);
        assert_eq!(back.input_price_per_mtok, e.input_price_per_mtok);
    }

    #[test]
    fn performance_json_defaults_are_conservative() {
        let v = serde_json::json!({"coding_reliability": 90});
        let p: ModelPerformance = serde_json::from_value(v).unwrap();
        assert_eq!(p.coding_reliability, 90);
        assert_eq!(p.context_reliability, 50);
        assert_eq!(p.estimated_latency_ms, 1000);
        let back: ModelPerformance =
            serde_json::from_value(serde_json::to_value(p).unwrap()).unwrap();
        assert_eq!(back, p);
    }

    // ---- hard-budget-safe pricing (audit): validity windows, effective
    // route-time state, conditional tariffs. Every fixture is a fixed
    // integer; no live data. ----

    const VALID_UNTIL_MS: u64 = 1_800_000_000_000;

    fn dated_exact(valid_until_ms: Option<u64>, ceiling: Option<PriceQuote>) -> PricingSnapshot {
        let mut s = PricingSnapshot::exact(quote(2_000_000, 8_000_000), 7, "fixture".into());
        s.valid_until_ms = valid_until_ms;
        s.conservative_ceiling = ceiling;
        s
    }

    #[test]
    fn validity_window_boundary_is_inclusive_and_none_never_expires() {
        let s = dated_exact(Some(VALID_UNTIL_MS), None);
        assert!(!s.is_expired_at(VALID_UNTIL_MS - 1));
        assert!(s.is_expired_at(VALID_UNTIL_MS), "the boundary is expired");
        assert!(s.is_expired_at(VALID_UNTIL_MS + 1));
        assert!(!dated_exact(None, None).is_expired_at(u64::MAX));
    }

    #[test]
    fn expired_exact_is_unknown_without_cap_and_fails_closed_under_one() {
        let state = PricingState::Known(dated_exact(Some(100), None));
        // Without a cap the honest state is Unknown: no number is trusted.
        let no_cap = state.effective_at(200, false);
        assert!(no_cap.is_unknown());
        assert_eq!(no_cap.authority(), PriceAuthority::Unknown);
        assert_eq!(no_cap.snapshot().settle_cost(1_000_000, 0, 0, 0), None);
        // Under a hard cap with NO documented ceiling the exact quote is
        // not a bound: fail closed, still Unknown.
        let capped = state.effective_at(200, true);
        assert!(
            capped.is_unknown(),
            "a stale exact quote is not a conservative bound"
        );
        // A future window keeps the quote exact on both paths.
        let fresh = PricingState::Known(dated_exact(Some(100), None));
        assert!(matches!(
            fresh.effective_at(99, true),
            EffectivePriceState::Exact(_)
        ));
    }

    #[test]
    fn expired_exact_with_a_documented_ceiling_is_conservative_under_hard_cap() {
        let ceiling = quote(6_000_000, 24_000_000);
        let state = PricingState::Known(dated_exact(Some(100), Some(ceiling)));
        let no_cap = state.effective_at(200, false);
        assert!(
            no_cap.is_unknown(),
            "without a cap the expired quote is Unknown"
        );
        let capped = state.effective_at(200, true);
        assert_eq!(
            capped.authority(),
            PriceAuthority::ConservativeCeiling,
            "a documented ceiling is the hard-cap bound"
        );
        assert_eq!(capped.quote(), Some(&ceiling));
        assert_eq!(
            capped.snapshot().settle_cost(1_000_000, 0, 0, 0),
            Some(6_000_000)
        );
    }

    #[test]
    fn stale_last_known_exact_is_unknown_even_under_a_hard_cap() {
        let stale = PricingState::Stale {
            last_known: PricingSnapshot::exact(quote(1, 2), 3, "aged".into())
                .with_conservative_ceiling(quote(4, 8)),
            observed_at_ms: 10,
        };
        for hard in [false, true] {
            let e = stale.effective_at(20, hard);
            assert!(
                e.is_unknown(),
                "stale exact != conservative bound (hard_cap={hard})"
            );
        }
    }

    #[test]
    fn conservative_ceiling_state_stays_a_bound_under_hard_cap() {
        let state = PricingState::ConservativeCeiling(PricingSnapshot::conservative_ceiling(
            quote(42, 42),
            2,
            "user".into(),
        ));
        let capped = state.effective_at(u64::MAX, true);
        assert_eq!(capped.authority(), PriceAuthority::ConservativeCeiling);
        assert_eq!(capped.quote(), Some(&quote(42, 42)));
    }

    #[test]
    fn local_zero_and_unknown_effective_states_are_never_numbers() {
        assert!(PricingState::LocalZero
            .effective_at(u64::MAX, true)
            .is_local_zero());
        assert_eq!(
            PricingState::LocalZero.effective_at(0, false).quote(),
            Some(&PriceQuote::ZERO)
        );
        let unknown = PricingState::Unknown.effective_at(0, false);
        assert!(unknown.is_unknown());
        assert_eq!(unknown.quote(), None);
        assert_eq!(unknown.snapshot().settle_cost(0, 0, 0, 0), None);
    }

    /// Fixed two-tariff test schedule: the 1-hour write tariff is strictly
    /// larger than the 5-minute one (the Anthropic shape).
    struct TwoTariffSchedule {
        five_minutes: PriceQuote,
        one_hour: PriceQuote,
        valid_until_ms: Option<u64>,
    }

    impl PriceSchedule for TwoTariffSchedule {
        fn quote(&self, ctx: &PricingContext) -> Result<PricingSnapshot, PriceUnavailable> {
            if self.valid_until_ms.map(|u| ctx.at_ms >= u).unwrap_or(false) {
                return Err(PriceUnavailable::Expired);
            }
            let q = match ctx.cache_write_ttl {
                Some(CacheWriteTtl::FiveMinutes) => self.five_minutes,
                Some(CacheWriteTtl::OneHour) => self.one_hour,
                // Unknown applicable tariff: a plain quote cannot pick one.
                None => return Err(PriceUnavailable::NoApplicableTariff),
            };
            Ok(PricingSnapshot::exact(q, 3, "sched".into()))
        }

        fn quote_for_hard_budget(
            &self,
            ctx: &PricingContext,
        ) -> Result<PricingSnapshot, PriceUnavailable> {
            if self.valid_until_ms.map(|u| ctx.at_ms >= u).unwrap_or(false) {
                return Err(PriceUnavailable::Expired);
            }
            let q = match ctx.cache_write_ttl {
                Some(CacheWriteTtl::FiveMinutes) => self.five_minutes,
                Some(CacheWriteTtl::OneHour) => self.one_hour,
                // MAXIMUM legitimate applicable tariff, never the minimum.
                None => self.five_minutes.max_per_line(self.one_hour),
            };
            Ok(PricingSnapshot::exact(q, 3, "sched".into()))
        }
    }

    #[test]
    fn schedule_with_unknown_tariff_and_hard_cap_picks_the_max_tariff() {
        let five = quote(15_000_000, 75_000_000).max_per_line(PriceQuote {
            cache_write: MicroUsdPerMillionTokens(18_750_000),
            ..quote(0, 0)
        });
        let one = PriceQuote {
            cache_write: MicroUsdPerMillionTokens(30_000_000),
            ..five
        };
        let sched = TwoTariffSchedule {
            five_minutes: five,
            one_hour: one,
            valid_until_ms: Some(VALID_UNTIL_MS),
        };
        let at = PricingContext::at(1_000);
        // Unknown TTL: plain quote refuses, hard budget picks the max.
        assert_eq!(
            sched.quote(&at).unwrap_err(),
            PriceUnavailable::NoApplicableTariff
        );
        let hard = sched.quote_for_hard_budget(&at).unwrap();
        assert_eq!(
            hard.quote.unwrap().cache_write,
            MicroUsdPerMillionTokens(30_000_000),
            "the unknown applicable tariff must reserve the maximum"
        );
        // A KNOWN TTL uses exactly its tariff (never the other one).
        assert_eq!(
            sched
                .quote_for_hard_budget(&at.with_cache_write_ttl(CacheWriteTtl::FiveMinutes))
                .unwrap()
                .quote
                .unwrap()
                .cache_write,
            MicroUsdPerMillionTokens(18_750_000)
        );
        assert_eq!(
            sched
                .quote(&at.with_cache_write_ttl(CacheWriteTtl::OneHour))
                .unwrap()
                .quote
                .unwrap()
                .cache_write,
            MicroUsdPerMillionTokens(30_000_000)
        );
        // The schedule's own validity window refuses after expiry, on both
        // entry points.
        let expired = PricingContext::at(VALID_UNTIL_MS);
        assert_eq!(
            sched.quote(&expired).unwrap_err(),
            PriceUnavailable::Expired
        );
        assert_eq!(
            sched.quote_for_hard_budget(&expired).unwrap_err(),
            PriceUnavailable::Expired
        );
    }

    #[test]
    fn fixed_price_quote_is_the_context_independent_fallback_schedule() {
        let q = quote(1_000_000, 3_000_000);
        let snapshot = PriceSchedule::quote(&q, &PricingContext::at(0)).unwrap();
        assert_eq!(snapshot.authority, PriceAuthority::Exact);
        assert_eq!(snapshot.quote, Some(q));
        assert_eq!(snapshot.authority, PriceAuthority::Exact);
        assert!(snapshot.valid_until_ms.is_none());
    }

    #[test]
    fn max_per_line_bounds_each_category_independently() {
        let a = PriceQuote {
            input: MicroUsdPerMillionTokens(1),
            output: MicroUsdPerMillionTokens(9),
            cache_read: MicroUsdPerMillionTokens(5),
            cache_write: MicroUsdPerMillionTokens(2),
        };
        let b = PriceQuote {
            input: MicroUsdPerMillionTokens(7),
            output: MicroUsdPerMillionTokens(3),
            cache_read: MicroUsdPerMillionTokens(4),
            cache_write: MicroUsdPerMillionTokens(8),
        };
        assert_eq!(
            a.max_per_line(b),
            PriceQuote {
                input: MicroUsdPerMillionTokens(7),
                output: MicroUsdPerMillionTokens(9),
                cache_read: MicroUsdPerMillionTokens(5),
                cache_write: MicroUsdPerMillionTokens(8),
            }
        );
    }

    #[test]
    fn snapshot_expiry_metadata_roundtrips_on_the_wire() {
        let snap = dated_exact(Some(VALID_UNTIL_MS), Some(quote(9, 9)));
        let back: PricingSnapshot =
            serde_json::from_value(serde_json::to_value(&snap).unwrap()).unwrap();
        assert_eq!(back, snap);
        // Legacy JSON without the new fields decodes with None/None.
        let legacy: PricingSnapshot = serde_json::from_value(serde_json::json!({
            "quote": {"input": 1, "output": 2, "cache_read": 3, "cache_write": 4},
            "authority": "exact", "epoch": 1, "source_id": "legacy"
        }))
        .unwrap();
        assert!(legacy.valid_until_ms.is_none());
        assert!(legacy.conservative_ceiling.is_none());
        assert!(!legacy.is_expired_at(u64::MAX));
    }
}

// ------------------------------------------------------------ economics

/// LEGACY per-token money unit (microUSD per token) — the numeric reading of
/// the router's internal per-token estimate fields (audit wave-B item A:
/// per-token microUSD cannot represent sub-$1/M prices, which truncate to
/// zero at ingestion, so REAL list prices must not enter through this type).
///
/// The name records the true dimension of the stored integer. It is NOT a
/// per-million-token price: the field keeps the per-token microUSD value,
/// which is *numerically equal* to USD per million tokens only when the
/// price is an exact multiple of $1/M (proof below). The router's candidate
/// scoring consumes this legacy estimate surface; catalog list prices and
/// settlement travel as [`MicroUsdPerMillionTokens`]/[`PriceQuote`].
///
/// # Proof that dollars-per-million == microUSD-per-token (identity)
///
/// `P` USD per million tokens, converted once at ingestion:
///
/// ```text
/// P USD       1_000_000 microUSD
/// -------  *  ------------------  =  P microUSD per token
/// 1e6 tokens       1 USD
/// ```
///
/// `P` dollars per million tokens therefore occupies the same integer as
/// `P` microUSD per token — the conversion is the identity — but ONLY when
/// the ingestion site SAYS SO through [`MicroUsdPerToken::from_dollars_per_million`].
/// The alternative spelling
/// [`MicroUsdPerToken::from_micros_per_million_tokens`] divides by `1e6`
/// (rounding DOWN, at ingestion only), so the two never silently collide.
///
/// Sub-integer microUSD per token cannot be represented: any price below
/// $1/Mtok truncates to 0 here at ingestion. **New price knowledge must
/// therefore use [`MicroUsdPerMillionTokens`]**, which keeps $0.50/M =
/// 500_000 microUSD per million tokens exact.
///
/// # Examples
///
/// ```
/// use faktor_core::model::{MicroUsdPerToken, MicroUsdPerMillionTokens};
///
/// // $15 per million tokens is 15 microUSD per token (identity conversion
/// // only at exact whole-dollar prices).
/// let input = MicroUsdPerToken::from_dollars_per_million(15);
/// assert_eq!(input, MicroUsdPerToken(15));
/// // 1M tokens at 15 microUSD/token cost exactly $15 in microUSD.
/// assert_eq!(input.saturating_mul(1_000_000), 15_000_000);
///
/// // Real list prices keep the per-million reading instead (no truncation
/// // below $1/M: $0.50/M stays 500_000 microUSD per million tokens).
/// let half = MicroUsdPerMillionTokens(500_000);
/// assert_eq!(half.cost_ceil_micro(1_000_000), 500_000);
/// assert_eq!(MicroUsdPerMillionTokens::ZERO.cost_ceil_micro(1_000_000), 0);
/// ```
///
/// The wrapper is deliberately NOT interchangeable with `u64`: mixing a
/// price with a latency value is a compile error, not a silent mis-read.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(transparent)]
pub struct MicroUsdPerToken(pub u64);

impl MicroUsdPerToken {
    /// `P` USD per million tokens -> `P` microUSD per token.
    ///
    /// The identity conversion above is expressed through this named
    /// constructor so a raw `P` can never be read as *microUSD per million
    /// tokens* by accident. Value = `dollars_per_million` exactly. Valid
    /// only for whole-dollar-per-million prices; prefer
    /// [`MicroUsdPerMillionTokens`] for real list prices.
    pub const fn from_dollars_per_million(dollars_per_million: u64) -> Self {
        Self(dollars_per_million)
    }

    /// MicroUSD per million tokens -> microUSD per token.
    ///
    /// `micros_per_million / 1_000_000`, rounding DOWN. Rounding happens
    /// ONLY at this ingestion boundary: every later arithmetic step
    /// ([`MicroUsdPerToken::saturating_mul`]) is exact integer math.
    /// Sub-integer microUSD per token (any price below $1/Mtok) truncates
    /// to 0 here — the reason the audit's wave B moved real list prices to
    /// [`MicroUsdPerMillionTokens`].
    pub const fn from_micros_per_million_tokens(micros_per_million: u64) -> Self {
        Self(micros_per_million / 1_000_000)
    }

    /// Total cost in microUSD of `tokens` at this per-token price,
    /// saturating at `u64::MAX` (hostile magnitudes never panic).
    pub const fn saturating_mul(self, tokens: u64) -> u64 {
        self.0.saturating_mul(tokens)
    }

    /// Zero price = unknown/local model (monetary cost zero; latency and
    /// reliability still count).
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }
}

impl From<u64> for MicroUsdPerToken {
    /// Wrap a value already expressed as microUSD PER TOKEN (the raw
    /// reading of the old bare fields). Catalog/config ingestion that
    /// starts from USD-per-million must use
    /// [`MicroUsdPerToken::from_dollars_per_million`] instead.
    fn from(microusd_per_token: u64) -> Self {
        Self(microusd_per_token)
    }
}

impl std::fmt::Display for MicroUsdPerToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// LEGACY typed price fields (microUSD per token, see
/// [`MicroUsdPerToken`]); 0 = the legacy zero marker (local models: zero
/// monetary cost, latency still counts).
///
/// This blob conflates money with performance (latency, reliability,
/// rate-limit state) — the audit wave-B split: real price knowledge now
/// travels as [`PriceQuote`] with an explicit [`PriceAuthority`], and the
/// performance dims the router consumes are named in
/// [`ModelPerformance`]. [`ModelEconomics`] survives ONLY as the router's
/// internal per-token estimate surface and for untouched callers.
///
/// ```compile_fail
/// // A latency value in milliseconds must never be accepted where a price
/// // is expected: the wrapper is a DISTINCT type, so the mixed addition
/// // below fails to compile instead of silently mis-budgeting.
/// use faktor_core::model::MicroUsdPerToken;
/// let latency_ms = 500u64;
/// let _budget = MicroUsdPerToken(15) + latency_ms;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelEconomics {
    pub input_price_per_mtok: MicroUsdPerToken,
    pub output_price_per_mtok: MicroUsdPerToken,
    pub cache_read_price_per_mtok: MicroUsdPerToken,
    pub cache_write_price_per_mtok: MicroUsdPerToken,
    pub estimated_latency_ms: u64,
    pub tool_reliability: u8,
    pub reasoning_reliability: u8,
    pub coding_reliability: u8,
    pub context_reliability: u8,
    pub availability: u8,
    pub rate_limit_state: RateLimitState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RateLimitState {
    Healthy,
    Soft,
    Hard,
}

impl Default for ModelEconomics {
    fn default() -> Self {
        Self {
            input_price_per_mtok: MicroUsdPerToken(0),
            output_price_per_mtok: MicroUsdPerToken(0),
            cache_read_price_per_mtok: MicroUsdPerToken(0),
            cache_write_price_per_mtok: MicroUsdPerToken(0),
            estimated_latency_ms: 1000,
            tool_reliability: 50,
            reasoning_reliability: 50,
            coding_reliability: 50,
            context_reliability: 50,
            availability: 100,
            rate_limit_state: RateLimitState::Healthy,
        }
    }
}

impl ModelEconomics {
    pub fn is_local_zero_cost(&self) -> bool {
        self.input_price_per_mtok.is_zero()
            && self.output_price_per_mtok.is_zero()
            && self.cache_read_price_per_mtok.is_zero()
            && self.cache_write_price_per_mtok.is_zero()
    }

    /// Mean of the reliability dimensions most relevant to a coding turn.
    pub fn coding_quality(&self) -> u8 {
        let sum = u32::from(self.tool_reliability)
            + u32::from(self.coding_reliability)
            + u32::from(self.context_reliability);
        (sum / 3) as u8
    }

    /// The named performance projection (audit item B): the split-out
    /// performance dims (reliability + latency + rate-limit state), carrying
    /// NO price fields. The routing path reads performance through THIS
    /// projection; the price fields below are the legacy per-token
    /// compatibility surface only.
    pub fn performance(&self) -> ModelPerformance {
        ModelPerformance {
            context_reliability: self.context_reliability,
            coding_reliability: self.coding_reliability,
            estimated_latency_ms: self.estimated_latency_ms,
            rate_limit_state: self.rate_limit_state,
        }
    }

    /// Write the named performance projection back onto the estimate blob
    /// (the price fields are untouched).
    pub fn with_performance(&self, p: ModelPerformance) -> Self {
        let mut e = *self;
        e.context_reliability = p.context_reliability;
        e.coding_reliability = p.coding_reliability;
        e.estimated_latency_ms = p.estimated_latency_ms;
        e.rate_limit_state = p.rate_limit_state;
        e
    }
}

// ---------------------------------------------------------------- pricing

/// MicroUSD per MILLION tokens — the money unit of every REAL list price
/// (audit wave-B item A). Providers publish prices as $/M; the published
/// integer is `dollars_per_million x 1_000_000` microUSD per million
/// tokens. Storing the raw per-million number (instead of truncating it to
/// microUSD per token) keeps EVERY price exact: $0.50/M = 500_000, $0.01/M
/// = 10_000, $15/M = 15_000_000. Nothing below $1/M truncates to zero, so a
/// paid model can never read as free.
///
/// Cost of `tokens` at this price is ONE ceiling division:
/// `ceil(price x tokens / 1_000_000)` in u128 — the exact integer formula,
/// never floats, never per-line rounding.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(transparent)]
pub struct MicroUsdPerMillionTokens(pub u64);

impl MicroUsdPerMillionTokens {
    /// The honest zero price: $0/M (free/local semantics — an authoritative
    /// zero only when a [`PriceAuthority`] says so).
    pub const ZERO: Self = Self(0);

    /// Ingestion from a published dollars-per-million price: `$15/M` ->
    /// `15_000_000` microUSD per million tokens (saturating).
    pub const fn from_dollars_per_million(dollars_per_million: u64) -> Self {
        Self(dollars_per_million.saturating_mul(1_000_000))
    }

    /// Exact ceiling cost of `tokens` at this price: category numerator in
    /// u128 (u64 x u64 cannot overflow it), ONE division with ceil, result
    /// saturated into u64. A positive price over a positive token count
    /// always costs at least 1 microUSD; zero usage costs zero.
    pub fn cost_ceil_micro(self, tokens: u64) -> u64 {
        let numerator = (self.0 as u128).saturating_mul(tokens as u128);
        let ceiling = numerator.saturating_add(999_999) / 1_000_000;
        u64::try_from(ceiling).unwrap_or(u64::MAX)
    }

    /// Zero price (the free/local line).
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }
}

impl From<u64> for MicroUsdPerMillionTokens {
    /// Wrap a value already expressed as microUSD PER MILLION TOKENS (the
    /// raw published list-price reading). USD-per-million ingestion must
    /// use [`MicroUsdPerMillionTokens::from_dollars_per_million`].
    fn from(microusd_per_million_tokens: u64) -> Self {
        Self(microusd_per_million_tokens)
    }
}

impl std::fmt::Display for MicroUsdPerMillionTokens {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// One usage frame settlement prices: the four token categories a model
/// call reports (uncached input, cache reads, cache writes, output). The
/// fields mirror the settlement signature so category numerators can be
/// summed BEFORE the single ceiling division of
/// [`PriceQuote::quote_cost_micro`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct TokenUsage {
    pub uncached_input_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    pub output_tokens: u64,
}

impl TokenUsage {
    pub const ZERO: Self = Self {
        uncached_input_tokens: 0,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        output_tokens: 0,
    };

    pub const fn new(
        uncached_input_tokens: u64,
        cache_read_tokens: u64,
        cache_write_tokens: u64,
        output_tokens: u64,
    ) -> Self {
        Self {
            uncached_input_tokens,
            cache_read_tokens,
            cache_write_tokens,
            output_tokens,
        }
    }
}

impl Default for TokenUsage {
    fn default() -> Self {
        Self::ZERO
    }
}

/// One model's price lines: microUSD per MILLION tokens per category
/// (audit wave-B item A). Every line keeps the published list price EXACTLY
/// (sub-$1/M included), and settlement sums the four category numerators in
/// u128 BEFORE one ceiling division — the exact cost formula of the audit.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct PriceQuote {
    pub input: MicroUsdPerMillionTokens,
    pub output: MicroUsdPerMillionTokens,
    pub cache_read: MicroUsdPerMillionTokens,
    pub cache_write: MicroUsdPerMillionTokens,
}

impl PriceQuote {
    /// The all-zero quote (the honest content of an authoritative
    /// [`PriceAuthority::LocalZero`] snapshot — zero by AUTHORITY, never by
    /// numeric accident).
    pub const ZERO: Self = Self {
        input: MicroUsdPerMillionTokens::ZERO,
        output: MicroUsdPerMillionTokens::ZERO,
        cache_read: MicroUsdPerMillionTokens::ZERO,
        cache_write: MicroUsdPerMillionTokens::ZERO,
    };

    /// The exact cost of one usage frame: sum the four category numerators
    /// (`price x tokens`, each exact in u128), then apply ONE ceiling
    /// division by 1e6. Rounding happens once at the total — never per
    /// line, never understated, never truncated to a free lie.
    pub fn quote_cost_micro(&self, usage: TokenUsage) -> u64 {
        let numerator = (self.input.0 as u128)
            .saturating_mul(usage.uncached_input_tokens as u128)
            .saturating_add(
                (self.cache_read.0 as u128).saturating_mul(usage.cache_read_tokens as u128),
            )
            .saturating_add(
                (self.cache_write.0 as u128).saturating_mul(usage.cache_write_tokens as u128),
            )
            .saturating_add((self.output.0 as u128).saturating_mul(usage.output_tokens as u128));
        let ceiling = numerator.saturating_add(999_999) / 1_000_000;
        u64::try_from(ceiling).unwrap_or(u64::MAX)
    }

    /// The per-line maximum of two quotes: a conservative upper bound when
    /// the applicable tariff is unknown. Each category is bounded
    /// INDEPENDENTLY (never a sum of tariffs: summing would charge two
    /// schedules at once).
    pub fn max_per_line(self, other: Self) -> Self {
        let max = |a: MicroUsdPerMillionTokens, b: MicroUsdPerMillionTokens| {
            MicroUsdPerMillionTokens(a.0.max(b.0))
        };
        Self {
            input: max(self.input, other.input),
            output: max(self.output, other.output),
            cache_read: max(self.cache_read, other.cache_read),
            cache_write: max(self.cache_write, other.cache_write),
        }
    }

    /// True when every line is zero (an all-free price statement — its
    /// MEANING still comes from the authority, never from this number).
    pub const fn is_zero(self) -> bool {
        self.input.is_zero()
            && self.output.is_zero()
            && self.cache_read.is_zero()
            && self.cache_write.is_zero()
    }
}

// ------------------------------------------------------- price schedules

/// Wall-clock Unix time in milliseconds (0 when the host clock is before
/// the epoch — a hostile clock must never panic a route).
pub fn unix_now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

/// Cache-write TTL dimension of a price schedule. Anthropic publishes
/// DISTINCT write rates for 5-minute and 1-hour prompt caches; the write
/// rate is not a single number.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum CacheWriteTtl {
    FiveMinutes,
    OneHour,
}

/// Service-tier dimension of a price schedule (e.g. OpenAI's flex/priority
/// tiers, Anthropic batch). A tier the schedule does not document has NO
/// applicable tariff: quoting must refuse rather than silently use the
/// standard price (which may be either above or below the real tariff).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum ServiceTier {
    Standard,
    Flex,
    Priority,
}

/// The tariff dimensions a quote depends on: wall-clock time (validity
/// windows), the prompt-cache TTL actually in use, and the service tier.
/// `None` means the caller does not know (or did not specify) that
/// dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct PricingContext {
    pub at_ms: u64,
    pub cache_write_ttl: Option<CacheWriteTtl>,
    pub service_tier: Option<ServiceTier>,
}

impl PricingContext {
    pub const fn at(at_ms: u64) -> Self {
        Self {
            at_ms,
            cache_write_ttl: None,
            service_tier: None,
        }
    }

    pub const fn with_cache_write_ttl(mut self, ttl: CacheWriteTtl) -> Self {
        self.cache_write_ttl = Some(ttl);
        self
    }

    pub const fn with_service_tier(mut self, tier: ServiceTier) -> Self {
        self.service_tier = Some(tier);
        self
    }
}

/// Why a [`PriceSchedule`] could not produce a quote. These are NOT
/// numeric fallbacks: a schedule that cannot price the context refuses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PriceUnavailable {
    /// The context asked for a tariff dimension (TTL/tier) the schedule
    /// does not document: there is no legitimate tariff to use.
    NoApplicableTariff,
    /// The quote's validity window has elapsed at `ctx.at_ms`.
    Expired,
    /// The model's price was retired (an official ID withdrawn): the last
    /// known quote must never be routed as if current.
    Retired,
    /// No price knowledge for this (provider, model).
    Unknown,
}

/// A conditional tariff: quotes a [`PricingSnapshot`] for a
/// [`PricingContext`] (validity window, cache-write TTL, service tier).
/// [`PriceQuote`] itself implements this trait as the fixed fallback: a
/// context-independent quote.
///
/// The hard-budget entry point is [`PriceSchedule::quote_for_hard_budget`]:
/// when the applicable tariff is unknown it must choose the MAXIMUM
/// legitimate applicable tariff — never the minimum — so a hard cap can
/// never be under-reserved.
pub trait PriceSchedule {
    fn quote(&self, ctx: &PricingContext) -> Result<PricingSnapshot, PriceUnavailable>;

    /// Hard-budget quote: conservative tariff selection. The default
    /// implementation is exactly [`PriceSchedule::quote`] — schedules with
    /// several legitimate tariffs override it to pick the maximum when the
    /// context leaves the applicable tariff unknown.
    fn quote_for_hard_budget(
        &self,
        ctx: &PricingContext,
    ) -> Result<PricingSnapshot, PriceUnavailable> {
        self.quote(ctx)
    }
}

impl PriceSchedule for PriceQuote {
    /// The fixed per-million quote, context-independent: the fallback
    /// schedule. It never expires (no metadata), so it quotes on every
    /// context.
    fn quote(&self, _ctx: &PricingContext) -> Result<PricingSnapshot, PriceUnavailable> {
        Ok(PricingSnapshot::exact(*self, 0, "fixed-quote".to_string()))
    }
}

/// The named NON-MONETARY performance dims the router consumes (audit
/// wave-B item B + the pricing-path audit): reliability ratings (0..=100,
/// 50 = neutral), estimated latency, and the observed rate-limit state.
/// This type carries NO price fields — money lives in [`PriceQuote`] and
/// reaches the router only through [`PricingState`]/[`PricingSnapshot`].
///
/// `ModelEconomics` remains the legacy per-token compatibility surface for
/// untouched callers, but the routing path reads performance through THIS
/// projection ([`ModelEconomics::performance`] / [`ModelDescriptor::performance`])
/// so money and performance can never be conflated again.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ModelPerformance {
    pub context_reliability: u8,
    pub coding_reliability: u8,
    pub estimated_latency_ms: u64,
    pub rate_limit_state: RateLimitState,
}

impl Default for ModelPerformance {
    /// The conservative generic prior (mirror of the legacy
    /// [`ModelEconomics::default`] reliability dims): neutral 50
    /// reliability, 1000 ms estimated latency, a healthy rate limiter.
    fn default() -> Self {
        Self {
            context_reliability: 50,
            coding_reliability: 50,
            estimated_latency_ms: 1000,
            rate_limit_state: RateLimitState::Healthy,
        }
    }
}

/// What a [`ModelPerformanceProfile`]'s prior RESTS on (quality-authority
/// audit): a prior is never a vendor truth by default — built-in priors are
/// documented Faktor routing priors, durable verified outcomes dominate them
/// when present, and a user may override them explicitly.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum QualityAuthority {
    /// The prior came from the durable VERIFIED-outcome corpus: measured
    /// success/rework history, the strongest authority.
    VerifiedCorpus,
    /// The user explicitly configured this performance prior.
    UserOverride,
    /// A built-in CONSERVATIVE-UNKNOWN routing prior (documented Faktor
    /// scoring priors, not vendor claims), the weakest authority.
    #[serde(rename = "conservative_unknown")]
    ConservativeUnknown,
}

/// One model's performance prior with its authority and the benchmark
/// version that produced it (quality-authority audit): built-in priors
/// carry [`QualityAuthority::ConservativeUnknown`] and a version string so
/// a prior is always inspectable and never silently treated as measured.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelPerformanceProfile {
    /// The non-monetary prior itself.
    pub prior: ModelPerformance,
    /// Why the prior is what it is (never inferred from numbers).
    pub authority: QualityAuthority,
    /// The benchmark/prior-table version (e.g. `"faktor-routing-priors-v1"`).
    pub benchmark_version: String,
}

/// The billing origin of a configured endpoint (billing-origin audit):
/// WHERE money is owed, decided from the strict endpoint configuration
/// (official canonical endpoint vs custom `base_url`, gateway kind, local
/// runtime) — NEVER from the adapter's transport family id, so a custom
/// OpenAI-compatible proxy can never inherit official OpenAI list prices
/// just because it speaks the OpenAI wire protocol.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum BillingOrigin {
    /// The canonical official OpenAI endpoint.
    OfficialOpenAi,
    /// The canonical official Anthropic endpoint.
    OfficialAnthropic,
    /// The canonical official Google endpoint.
    OfficialGoogle,
    /// The canonical official DeepSeek endpoint.
    OfficialDeepSeek,
    /// A user-configured / self-hosted OpenAI-compatible endpoint: prices
    /// stay Unknown unless the user supplies an exact quote or ceiling.
    CustomEndpoint,
    /// A gateway/aggregator endpoint (e.g. the Kilo gateway): prices stay
    /// Unknown unless the user supplies an exact quote or ceiling.
    Gateway,
    /// A local runtime (Ollama): an authoritative zero price.
    Local,
}

impl BillingOrigin {
    /// True for the four officially-priced origins whose built-in list
    /// prices the catalog may apply.
    pub const fn is_official(self) -> bool {
        matches!(
            self,
            BillingOrigin::OfficialOpenAi
                | BillingOrigin::OfficialAnthropic
                | BillingOrigin::OfficialGoogle
                | BillingOrigin::OfficialDeepSeek
        )
    }
}

/// What a price statement RESTS on (audit wave-B item B). Authority is
/// decided where the price is created (builtin table, user override,
/// adapter declaration, catalog miss) and is NEVER inferred from numbers: a
/// zero quote is not `LocalZero`, and an absent quote is not a zero.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Default,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum PriceAuthority {
    /// Real, authoritative list prices (a built-in or user-declared exact
    /// table row).
    Exact,
    /// The user's conservative budget ceiling priced the model: the quote
    /// is a bound, never a measured price.
    ConservativeCeiling,
    /// Authoritative zero-price profile of a LOCAL runtime (Ollama): $0 is
    /// the measured truth. Always carried together with
    /// `quote: Some(PriceQuote::ZERO)`.
    LocalZero,
    /// A pricing authority was consulted but has no price for this model:
    /// settlement must never pretend zero or one.
    #[default]
    Unknown,
}

/// The route-time price capture (audit wave-B items A/B): an immutable
/// [`PriceQuote`] (per-million-token microUSD lines, exact) plus the
/// authority/epoch/source that produced it. Captured ONCE at routing so a
/// later catalog repricing can never rewrite what an already-paid call
/// should have cost, and persisted on the cost reservation row so
/// settlement survives daemon restarts.
///
/// The invariants of the audit:
///
/// - `LocalZero` == `authority: LocalZero` AND `quote: Some(PriceQuote::ZERO)`;
/// - `Unknown` == `authority: Unknown` AND `quote: None`;
/// - authority is NEVER inferred from the quote numbers; a snapshot that
///   decodes without an explicit authority (legacy wave-A JSON rows) reads
///   as `Unknown`/no-quote and refuses settlement (fail closed).
///
/// [`PricingSnapshot::settle_cost`] is the settlement math: sum the usage
/// categories' numerators against the quote's per-million lines, one
/// ceiling division. It returns `None` for `Unknown` or a missing quote —
/// no number is honest then (the caller refuses under a hard budget and
/// records an Unknown spend otherwise). `None` from a hostile
/// `LocalZero`-without-quote JSON is also correct: nothing is fabricated.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PricingSnapshot {
    /// The frozen price lines; `None` only for `Unknown` (or a corrupt/
    /// legacy row, which then behaves as Unknown).
    #[serde(default)]
    pub quote: Option<PriceQuote>,
    /// Why the quote is what it is (or why there is none).
    #[serde(default)]
    pub authority: PriceAuthority,
    /// Catalog/pricing generation this snapshot was cut from (0 for a row
    /// that carried no epoch — decoded legacy JSON).
    #[serde(default)]
    pub epoch: u64,
    /// The pricing source that produced this snapshot (e.g.
    /// `"faktor-builtin-v1"`, a provider instance id, a user-override
    /// marker). Empty for decoded legacy rows.
    #[serde(default)]
    pub source_id: String,
    /// Wall-clock ms at which the exact quote stops being an exact
    /// statement (`None` = no published expiry: static fixtures and legacy
    /// rows never expire). At the boundary the quote counts as EXPIRED (the
    /// same inclusive boundary rule as operation deadlines).
    #[serde(default)]
    pub valid_until_ms: Option<u64>,
    /// A documented conservative UPPER BOUND usable after the exact quote
    /// expires when a hard budget still needs an honest bound. `None`
    /// means "no honest bound": an expired quote then becomes Unknown and
    /// fails closed under a hard cap — a stale exact quote is NOT itself a
    /// conservative bound.
    #[serde(default)]
    pub conservative_ceiling: Option<PriceQuote>,
}

impl PricingSnapshot {
    /// A real list-price snapshot (built-in or exact user table). No
    /// expiry and no ceiling by default; [`PricingSnapshot::with_valid_until`]
    /// / [`PricingSnapshot::with_conservative_ceiling`] attach the catalog
    /// metadata.
    pub fn exact(quote: PriceQuote, epoch: u64, source_id: String) -> Self {
        Self {
            quote: Some(quote),
            authority: PriceAuthority::Exact,
            epoch,
            source_id,
            valid_until_ms: None,
            conservative_ceiling: None,
        }
    }

    /// A conservative-ceiling snapshot (the user's budget bound).
    pub fn conservative_ceiling(quote: PriceQuote, epoch: u64, source_id: String) -> Self {
        Self {
            quote: Some(quote),
            authority: PriceAuthority::ConservativeCeiling,
            epoch,
            source_id,
            valid_until_ms: None,
            conservative_ceiling: None,
        }
    }

    /// The authoritative-zero profile of a LOCAL runtime (Ollama): the
    /// zero quote is the honest local-model price, never a fabricated
    /// missing number.
    pub fn local_zero(epoch: u64, source_id: String) -> Self {
        Self {
            quote: Some(PriceQuote::ZERO),
            authority: PriceAuthority::LocalZero,
            epoch,
            source_id,
            valid_until_ms: None,
            conservative_ceiling: None,
        }
    }

    /// No price knowledge: `quote: None` + `authority: Unknown`. Settlement
    /// must never pretend zero or one.
    pub fn unknown(epoch: u64, source_id: String) -> Self {
        Self {
            quote: None,
            authority: PriceAuthority::Unknown,
            epoch,
            source_id,
            valid_until_ms: None,
            conservative_ceiling: None,
        }
    }

    /// Attach the catalog validity window (builder form; additive).
    pub fn with_valid_until(mut self, valid_until_ms: u64) -> Self {
        self.valid_until_ms = Some(valid_until_ms);
        self
    }

    /// Attach a documented conservative upper bound usable after expiry
    /// (builder form; additive).
    pub fn with_conservative_ceiling(mut self, ceiling: PriceQuote) -> Self {
        self.conservative_ceiling = Some(ceiling);
        self
    }

    /// True when the validity window has elapsed at `now_ms`. The boundary
    /// is inclusive: `now_ms == valid_until_ms` is expired. A snapshot
    /// WITHOUT a validity window never expires.
    pub fn is_expired_at(&self, now_ms: u64) -> bool {
        self.valid_until_ms
            .map(|until| now_ms >= until)
            .unwrap_or(false)
    }

    /// True only for an explicit [`PriceAuthority::LocalZero`] — never
    /// inferred from zero quote numbers.
    pub const fn is_local_zero(&self) -> bool {
        matches!(self.authority, PriceAuthority::LocalZero)
    }

    /// The exact settlement cost of one usage frame at this snapshot's
    /// lines (summed numerators, one ceiling division; saturating; never
    /// panics on hostile magnitudes). `None` when the snapshot is
    /// [`PriceAuthority::Unknown`] or carries no quote: no number is honest
    /// then — the caller refuses under a hard budget and records Unknown
    /// spend otherwise.
    pub fn settle_cost(
        &self,
        uncached_input_tokens: u64,
        cache_read_tokens: u64,
        cache_write_tokens: u64,
        output_tokens: u64,
    ) -> Option<u64> {
        if self.authority == PriceAuthority::Unknown {
            return None;
        }
        self.quote.map(|q| {
            q.quote_cost_micro(TokenUsage::new(
                uncached_input_tokens,
                cache_read_tokens,
                cache_write_tokens,
                output_tokens,
            ))
        })
    }
}

/// The price knowledge that is actually usable AT ROUTE TIME, derived from
/// a [`PricingState`] plus the wall clock and the presence of a hard cost
/// cap. A [`PricingState::Known`] quote is no longer sufficient on its own:
///
/// - a fresh Known quote is [`EffectivePriceState::Exact`];
/// - a Known quote past its `valid_until_ms` becomes
///   [`EffectivePriceState::Unknown`] WITHOUT a hard cap (no fabricated
///   number), and under a hard cap becomes
///   [`EffectivePriceState::Conservative`] ONLY when the snapshot carries a
///   documented `conservative_ceiling`; otherwise it stays Unknown and the
///   hard-capped request fails closed;
/// - a [`PricingState::Stale`] row is Unknown even when its last-known
///   snapshot was Exact: a stale exact quote is NOT a conservative bound;
/// - [`PricingState::ConservativeCeiling`] stays Conservative (a declared
///   bound), LocalZero stays the authoritative zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EffectivePriceState {
    Exact(PricingSnapshot),
    Conservative(PricingSnapshot),
    LocalZero,
    Unknown(PricingSnapshot),
}

impl EffectivePriceState {
    /// The settlement authority of the effective state (the VARIANT
    /// decides, never the numbers).
    pub fn authority(&self) -> PriceAuthority {
        match self {
            EffectivePriceState::Exact(_) => PriceAuthority::Exact,
            EffectivePriceState::Conservative(_) => PriceAuthority::ConservativeCeiling,
            EffectivePriceState::LocalZero => PriceAuthority::LocalZero,
            EffectivePriceState::Unknown(_) => PriceAuthority::Unknown,
        }
    }

    /// The frozen route-time snapshot this effective state settles with.
    pub fn snapshot(&self) -> PricingSnapshot {
        match self {
            EffectivePriceState::Exact(s)
            | EffectivePriceState::Conservative(s)
            | EffectivePriceState::Unknown(s) => s.clone(),
            EffectivePriceState::LocalZero => {
                PricingSnapshot::local_zero(0, "effective-local-zero".into())
            }
        }
    }

    /// The usable price lines, when a numeric quote exists (the
    /// authoritative local zero included). Unknown has none — never a
    /// fabricated zero.
    pub fn quote(&self) -> Option<&PriceQuote> {
        match self {
            EffectivePriceState::Exact(s) | EffectivePriceState::Conservative(s) => {
                s.quote.as_ref()
            }
            EffectivePriceState::LocalZero => Some(&PriceQuote::ZERO),
            EffectivePriceState::Unknown(_) => None,
        }
    }

    pub const fn is_unknown(&self) -> bool {
        matches!(self, EffectivePriceState::Unknown(_))
    }

    pub const fn is_local_zero(&self) -> bool {
        matches!(self, EffectivePriceState::LocalZero)
    }
}

/// Price knowledge of one provider/model, the ROUTER's priced unit
/// (pricing-path audit). The states are NOT numeric prices: `Unknown` must
/// never be flattened to 0 microUSD, and `LocalZero` is a measured zero, not
/// a missing price. Authority is the STATE VARIANT, never inferred from the
/// quote numbers.
///
/// This type lives in core (moved from the provider catalog) so the router
/// can carry it on a `RouteCandidate` without depending on provider code;
/// `faktor_provider::catalog` re-exports it unchanged.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PricingState {
    /// Real, exact price knowledge: the wrapped snapshot's per-million
    /// quote is authoritative with [`PriceAuthority::Exact`].
    Known(PricingSnapshot),
    /// A user-configured conservative ceiling priced the model: the quote
    /// is a budget bound with [`PriceAuthority::ConservativeCeiling`],
    /// never a measured price.
    ConservativeCeiling(PricingSnapshot),
    /// Local runtime with zero monetary cost (Ollama). The zero is
    /// measured truth: latency and reliability still count.
    LocalZero,
    /// No price knowledge. Never zero, never a fabricated 1-microUSD
    /// fallback.
    Unknown,
    /// Last-known price knowledge that aged out of freshness: admission
    /// and candidate building keep treating the row by its last-known
    /// authority while `observed_at_ms` records when the knowledge stopped
    /// being current.
    Stale {
        last_known: PricingSnapshot,
        observed_at_ms: u64,
    },
}

impl PricingState {
    /// True only for a measured local-zero price.
    pub fn is_local_zero(&self) -> bool {
        matches!(self, PricingState::LocalZero)
    }

    /// The state's authority: the VARIANT decides, never the numbers.
    /// A [`PricingState::Stale`] row speaks with its last-known authority.
    pub fn authority(&self) -> PriceAuthority {
        match self {
            PricingState::Known(_) => PriceAuthority::Exact,
            PricingState::ConservativeCeiling(_) => PriceAuthority::ConservativeCeiling,
            PricingState::LocalZero => PriceAuthority::LocalZero,
            PricingState::Unknown => PriceAuthority::Unknown,
            PricingState::Stale { last_known, .. } => last_known.authority,
        }
    }

    /// The authoritative price lines of the state, when any exist:
    /// `Known`/`ConservativeCeiling`/`Stale` quote their snapshot,
    /// [`PricingState::LocalZero`] is the honest `Some(PriceQuote::ZERO)`
    /// (its definition), `Unknown` has none. Never a fabricated zero.
    pub fn quote(&self) -> Option<&PriceQuote> {
        match self {
            PricingState::Known(s) | PricingState::ConservativeCeiling(s) => s.quote.as_ref(),
            PricingState::Stale { last_known, .. } => last_known.quote.as_ref(),
            PricingState::LocalZero => Some(&PriceQuote::ZERO),
            PricingState::Unknown => None,
        }
    }

    /// The route-time [`PricingSnapshot`] this state cuts. `Known`/
    /// `ConservativeCeiling`/`Stale` forward their frozen snapshot;
    /// `LocalZero`/`Unknown` materialize the matching no-identity snapshot
    /// (their unit variants carry no epoch/source; callers that need the
    /// catalog row's real identity use the provider catalog's
    /// `ModelCatalogEntry::pricing_snapshot`). Settlement semantics depend
    /// on the AUTHORITY, never on the identity fields.
    pub fn snapshot(&self) -> PricingSnapshot {
        match self {
            PricingState::Known(s)
            | PricingState::ConservativeCeiling(s)
            | PricingState::Stale { last_known: s, .. } => s.clone(),
            PricingState::LocalZero => PricingSnapshot::local_zero(0, "pricing-state".into()),
            PricingState::Unknown => PricingSnapshot::unknown(0, "pricing-state".into()),
        }
    }

    /// Derive the route-time [`EffectivePriceState`] at `now_ms` under an
    /// optional hard cost cap (hard-budget-safe pricing audit):
    ///
    /// - `Known` fresh → `Exact`;
    /// - `Known` expired + no cap → `Unknown` (the honest state: no number
    ///   is trusted);
    /// - `Known` expired + hard cap → `Conservative` at the snapshot's
    ///   documented `conservative_ceiling` when it has one, else `Unknown`
    ///   (fail closed: a stale exact quote is not a bound);
    /// - `ConservativeCeiling` → `Conservative` (a declared bound);
    /// - `LocalZero` → `LocalZero`;
    /// - `Stale { .. }` → `Unknown` ALWAYS: staleness is not a bound, even
    ///   when the last-known snapshot was exact;
    /// - `Unknown` → `Unknown`.
    pub fn effective_at(&self, now_ms: u64, hard_cost_cap: bool) -> EffectivePriceState {
        match self {
            PricingState::Known(s) if !s.is_expired_at(now_ms) => {
                EffectivePriceState::Exact(s.clone())
            }
            PricingState::Known(s) => match (hard_cost_cap, s.conservative_ceiling) {
                (true, Some(ceiling)) => {
                    let mut bound = s.clone();
                    bound.quote = Some(ceiling);
                    bound.authority = PriceAuthority::ConservativeCeiling;
                    bound.conservative_ceiling = None;
                    bound.valid_until_ms = None;
                    EffectivePriceState::Conservative(bound)
                }
                _ => EffectivePriceState::Unknown(PricingSnapshot::unknown(
                    s.epoch,
                    s.source_id.clone(),
                )),
            },
            PricingState::ConservativeCeiling(s) => EffectivePriceState::Conservative(s.clone()),
            PricingState::LocalZero => EffectivePriceState::LocalZero,
            PricingState::Unknown => {
                EffectivePriceState::Unknown(PricingSnapshot::unknown(0, "pricing-state".into()))
            }
            PricingState::Stale { last_known, .. } => EffectivePriceState::Unknown(
                PricingSnapshot::unknown(last_known.epoch, last_known.source_id.clone()),
            ),
        }
    }
}

fn pricing_rank(s: &PricingState) -> u8 {
    match s {
        PricingState::LocalZero => 0,
        PricingState::Known(_) => 1,
        PricingState::ConservativeCeiling(_) => 2,
        PricingState::Unknown => 3,
        PricingState::Stale { .. } => 4,
    }
}

fn cmp_price_snapshot(a: &PricingSnapshot, b: &PricingSnapshot) -> std::cmp::Ordering {
    let cmp_quote = |q: Option<PriceQuote>| {
        q.map(|quote| {
            (
                quote.input,
                quote.output,
                quote.cache_read,
                quote.cache_write,
            )
        })
    };
    cmp_quote(a.quote)
        .cmp(&cmp_quote(b.quote))
        .then_with(|| a.authority.cmp(&b.authority))
        .then_with(|| a.epoch.cmp(&b.epoch))
        .then_with(|| a.source_id.cmp(&b.source_id))
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
                (PricingState::Known(a), PricingState::Known(b))
                | (PricingState::ConservativeCeiling(a), PricingState::ConservativeCeiling(b)) => {
                    cmp_price_snapshot(a, b)
                }
                (
                    PricingState::Stale { last_known: a, .. },
                    PricingState::Stale { last_known: b, .. },
                ) => cmp_price_snapshot(a, b),
                _ => std::cmp::Ordering::Equal,
            })
    }
}

/// What a request is FOR (audit economic router phases).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouterPhase {
    Plan,
    Explore,
    Retrieve,
    Implement,
    Review,
    TestAnalysis,
    Debug,
    Compact,
    Summarize,
    Title,
    Embed,
}

impl RouterPhase {
    pub const ALL: [RouterPhase; 11] = [
        RouterPhase::Plan,
        RouterPhase::Explore,
        RouterPhase::Retrieve,
        RouterPhase::Implement,
        RouterPhase::Review,
        RouterPhase::TestAnalysis,
        RouterPhase::Debug,
        RouterPhase::Compact,
        RouterPhase::Summarize,
        RouterPhase::Title,
        RouterPhase::Embed,
    ];
}

/// The class of the work a routable request carries — an OUTCOME-LEARNING
/// dimension of the verified-outcome registry (audit items 13/14/L), never
/// a provider-behavior switch.
///
/// A router consult folds every class recorded for one
/// (provider, model, phase) because a route request carries no class of its
/// own; settlement-time recording appends each verified sample against the
/// FULL key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskClass {
    Easy,
    Medium,
    Hard,
}

impl TaskClass {
    pub const ALL: [TaskClass; 3] = [TaskClass::Easy, TaskClass::Medium, TaskClass::Hard];
}

impl Default for TaskClass {
    /// The neutral class for additive wire/config defaults.
    fn default() -> Self {
        TaskClass::Medium
    }
}

/// The semantic risk bucket of the operation a routable request carries —
/// an OUTCOME-LEARNING dimension of the verified-outcome registry (audit
/// items 13/14/L). It is decided by the runtime's own risk model at
/// settlement time; it must NEVER be inferred from provider/model names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskBucket {
    Low,
    Medium,
    High,
}

impl RiskBucket {
    pub const ALL: [RiskBucket; 3] = [RiskBucket::Low, RiskBucket::Medium, RiskBucket::High];
}

impl Default for RiskBucket {
    /// The neutral (lowest) risk for additive wire/config defaults.
    fn default() -> Self {
        RiskBucket::Low
    }
}

/// One routable model with its provenance (audit ModelRegistry-lite).
///
/// `economics` is the LEGACY per-token estimate surface the router's
/// internal candidate scoring reads (prices rounded to whole microUSD per
/// token — lossy below $1/M by construction). Real catalog price knowledge
/// rides the model catalog's [`PricingSnapshot`] (per-million-token quotes,
/// exact), which the routing graph attaches to route decisions; the
/// descriptor itself carries no authority and a descriptor-only route
/// produces `pricing_snapshot: None`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelDescriptor {
    pub provider: String,
    pub model: String,
    pub context: u64,
    pub max_output: u64,
    pub tools: bool,
    pub parallel_tools: bool,
    pub reasoning: bool,
    pub thinking: bool,
    pub vision: bool,
    pub structured_output: bool,
    pub embeddings: bool,
    pub streaming: bool,
    pub economics: ModelEconomics,
    pub source: ModelSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelSource {
    LiveProbe,
    ProviderCatalog,
    CuratedRegistry,
    UserOverride,
    ConservativeDefault,
}

impl ModelDescriptor {
    /// The NON-MONETARY performance view of this descriptor. The router's
    /// qualification/scoring paths read performance through this projection
    /// so a price can never sneak into quality/latency math.
    pub fn performance(&self) -> ModelPerformance {
        self.economics.performance()
    }

    /// Fixed capability-id table (audit: capability filtering is explicit).
    pub fn capability_ok(&self, required: &[String]) -> bool {
        required.iter().all(|c| match c.as_str() {
            "tools" => self.tools,
            "parallel_tools" => self.parallel_tools,
            "reasoning" => self.reasoning,
            "thinking" => self.thinking,
            "vision" => self.vision,
            "structured_output" => self.structured_output,
            "embeddings" => self.embeddings,
            "streaming" => self.streaming,
            _ => false, // unknown capability: fail closed (never assume)
        })
    }
}

/// One routing decision (audit: every decision recorded and auditable).
///
/// `pricing_snapshot` is the immutable route-time price capture the
/// settlement path prices actual usage against (audit wave-B items A/B):
/// `Some` = a pricing authority was consulted and this is its frozen word
/// (an exact quote, a ceiling, an authoritative local zero, or an explicit
/// Unknown with no quote — the authority is never inferred); `None` = NO
/// pricing authority was consulted (the descriptor-only router path,
/// passthrough / RouterUnavailable degradation, the test graph's empty
/// pin) and the runtime must NOT invent one — an unpriced decision under a
/// hard task cost budget fails closed at reserve time, and without a
/// budget its spend is recorded as Unknown, never as a fabricated number.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RouteDecision {
    pub provider: String,
    pub model: String,
    pub estimated_cost_micro: u64,
    pub estimated_latency_ms: u64,
    pub reasoning: String,
    pub considered: usize,
    pub source: ModelSource,
    /// `None` (additive, wire-default) = no pricing authority consulted.
    #[serde(default)]
    pub pricing_snapshot: Option<PricingSnapshot>,
}

/// How the daemon's economic routing policy treats every model call
/// (P0-2/85/87/88). The mode is fixed at graph build from the daemon config
/// (Economy is the default) and never re-decided per turn — the policy is a
/// single authority over the whole daemon.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingMode {
    /// Every model call is routed through the RouterService: the decision's
    /// provider/model override the session-configured defaults (the former
    /// "auto" sentinel semantics, now unconditional). A session whose model
    /// the router cannot serve is refused with a typed failure — never
    /// silently replaced.
    Economy,
    /// Maximum verified-success routing: every request is routed to the
    /// highest phase-quality tier that clears the router's hard caps
    /// (capabilities, context/output fit, quality floor, budget, latency
    /// preference, rate-limit/cooldown health — the router's single
    /// qualification pass is the authority). Cost is the tie-break within
    /// that top tier, never the primary objective: the decision may exceed
    /// what an Economy evaluation would spend.
    MaximumQuality,
    /// Balanced routing: the same expected-cost-to-success evaluation as
    /// Economy, but at a higher quality floor — the policy never routes a
    /// model below the balanced band while a band model can serve the
    /// request (cheap-band models keep winning only when nothing at the
    /// band clears the request's hard caps).
    Balanced,
    /// One explicit (provider, model) pin. Every model call still passes
    /// through the routing policy for capability/fit/budget/health
    /// validation; when validation passes the pin wins even if the router
    /// would have picked a cheaper model (fail closed, never a silent
    /// switch). An EMPTY provider or model means "the session's own
    /// configured side" (the pin of the test graph's passthrough policy).
    Pinned { provider: String, model: String },
}

impl RoutingMode {
    /// The configured pin, if any.
    pub fn pinned(&self) -> Option<(&str, &str)> {
        match self {
            RoutingMode::Economy | RoutingMode::MaximumQuality | RoutingMode::Balanced => None,
            RoutingMode::Pinned { provider, model } => Some((provider, model)),
        }
    }

    pub fn is_economy(&self) -> bool {
        matches!(self, RoutingMode::Economy)
    }
}
