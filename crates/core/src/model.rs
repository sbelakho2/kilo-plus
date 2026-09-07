//! Model capabilities. Provider behavior is decided by these flags, never by
//! string-matching provider names. The agent reads capabilities; the adapters
//! set them; provider quirks stay inside adapters.

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

    // ---- price wrapper (audit P0-5): units are typed, conversions are
    // named, the wire representation is a plain number. ----

    #[test]
    fn dollars_per_million_conversion_is_the_proven_identity() {
        // $15/Mtok == 15 microUSD per token: 1M tokens at 15 micro each is
        // exactly $15 = 15_000_000 microUSD, with NO division anywhere.
        let price = MicroUsdPerToken::from_dollars_per_million(15);
        assert_eq!(price.0, 15);
        assert_eq!(price.saturating_mul(1_000_000), 15_000_000);
        assert_eq!(price.saturating_mul(999_999), 14_999_985);
        assert_eq!(
            MicroUsdPerToken::from_dollars_per_million(60).saturating_mul(2048),
            122_880
        );
        // Boundary magnitude: $0/Mtok (free/local) is a valid zero price.
        assert!(MicroUsdPerToken::from_dollars_per_million(0).is_zero());
    }

    #[test]
    fn micros_per_million_tokens_conversion_rounds_down_only_at_ingestion() {
        // 15e6 microUSD per million tokens == $15/Mtok == 15 microUSD/token.
        let a = MicroUsdPerToken::from_micros_per_million_tokens(15_000_000);
        assert_eq!(a, MicroUsdPerToken::from_dollars_per_million(15));
        // Truncation is downward, never bankers' rounding, never upward.
        assert_eq!(
            MicroUsdPerToken::from_micros_per_million_tokens(15_999_999).0,
            15
        );
        assert_eq!(
            MicroUsdPerToken::from_micros_per_million_tokens(14_999_999).0,
            14
        );
    }

    #[test]
    fn loud_zero_sub_integer_prices_truncate_to_zero_at_ingestion() {
        // $0.50/Mtok = 500_000 microUSD/Mtok = 0.5 microUSD/token: sub-
        // integer microUSD per token is impossible, so ingestion truncates
        // DOWN to 0. This is loud on purpose: any price below $1/Mtok
        // silently behaves as zero-cost (local-model semantics), which
        // callers must not mistake for a measured $0 price.
        let half = MicroUsdPerToken::from_micros_per_million_tokens(500_000);
        assert_eq!(half, MicroUsdPerToken(0), "0.5 microUSD/token -> 0");
        assert!(
            half.is_zero(),
            "a truncated sub-integer price is indistinguishable from free"
        );
        // One microUSD per million tokens truncates to zero as well; the
        // first representable price above $0 is exactly $1/Mtok.
        assert_eq!(
            MicroUsdPerToken::from_micros_per_million_tokens(999_999),
            MicroUsdPerToken(0)
        );
        assert_eq!(
            MicroUsdPerToken::from_micros_per_million_tokens(1_000_000),
            MicroUsdPerToken(1)
        );
    }

    #[test]
    fn saturating_price_arithmetic_never_overflows_or_understates() {
        let max = MicroUsdPerToken::from_dollars_per_million(u64::MAX);
        assert_eq!(
            max.saturating_mul(u64::MAX),
            u64::MAX,
            "hostile magnitude saturates, never panics"
        );
        // Tiny calls still cost at least one micro where a price is set:
        // per-token prices are >= 1 microUSD, so 1 token costs >= 1 micro.
        assert_eq!(MicroUsdPerToken(1).saturating_mul(1), 1);
        assert_eq!(MicroUsdPerToken(0).saturating_mul(1), 0);
    }

    #[test]
    fn price_and_latency_units_are_distinct_types() {
        // The P0-5 hazard was one integer being readable as two units.
        // MicroUsdPerToken is a distinct nominal type (not an alias): its
        // type name never equals u64's, so a price can only reach latency
        // math through an explicit `.0`/From (the wrapper implements no
        // unit-mixing arithmetic; see the compile_fail doc example).
        let price_name = std::any::type_name::<MicroUsdPerToken>();
        let u64_name = std::any::type_name::<u64>();
        assert_ne!(
            price_name, u64_name,
            "MicroUsdPerToken must not be an alias of u64"
        );
        // Same storage layout, different meaning — the compiler enforces
        // the meaning, not the size.
        assert_eq!(
            std::mem::size_of::<MicroUsdPerToken>(),
            std::mem::size_of::<u64>()
        );
        let price = MicroUsdPerToken::from_dollars_per_million(15);
        let latency_ms: u64 = 500;
        let _ = (price, latency_ms);
    }

    #[test]
    fn economics_json_stays_a_plain_number_and_roundtrips() {
        let e = ModelEconomics {
            input_price_per_mtok: MicroUsdPerToken::from_dollars_per_million(15),
            output_price_per_mtok: MicroUsdPerToken(60),
            cache_read_price_per_mtok: MicroUsdPerToken(3),
            cache_write_price_per_mtok: MicroUsdPerToken(7),
            ..Default::default()
        };
        // Transparent serde: the wire representation is the plain integer
        // the u64 fields used to serialize — configs/fixtures are compatible.
        let v = serde_json::to_value(e).unwrap();
        assert_eq!(v["input_price_per_mtok"], serde_json::json!(15));
        assert_eq!(v["output_price_per_mtok"], serde_json::json!(60));
        assert_eq!(v["cache_read_price_per_mtok"], serde_json::json!(3));
        assert_eq!(v["cache_write_price_per_mtok"], serde_json::json!(7));
        let back: ModelEconomics = serde_json::from_value(v).unwrap();
        assert_eq!(back, e);
        // Huge hostile magnitudes survive the wire (clamping is caller-side);
        // unknown fields are ignored, never fatal.
        let mut hostile = serde_json::to_value(e).unwrap();
        hostile["input_price_per_mtok"] = serde_json::json!(u64::MAX);
        hostile["intruder_field"] = serde_json::json!("nope");
        let econ: ModelEconomics = serde_json::from_value(hostile).unwrap();
        assert_eq!(econ.input_price_per_mtok, MicroUsdPerToken(u64::MAX));
        assert!(!econ.is_local_zero_cost());
    }

    #[test]
    fn zero_prices_are_local_cost_but_latency_still_counts() {
        let e = ModelEconomics::default();
        assert!(e.is_local_zero_cost());
        let mut priced = e;
        priced.output_price_per_mtok = MicroUsdPerToken::from_dollars_per_million(1);
        assert!(!priced.is_local_zero_cost());
        // A price of 0 on every field is the local-model marker; setting
        // any single field to a real price flips the marker.
        let mut one_cache = e;
        one_cache.cache_read_price_per_mtok = MicroUsdPerToken(1);
        assert!(!one_cache.is_local_zero_cost());
    }

    #[test]
    fn display_and_from_are_plain() {
        assert_eq!(
            MicroUsdPerToken::from_dollars_per_million(15).to_string(),
            "15"
        );
        let raw = u64::from(15u8);
        assert_eq!(MicroUsdPerToken::from(raw).0, 15);
        assert_eq!(
            MicroUsdPerToken::from(15),
            MicroUsdPerToken::from_dollars_per_million(15)
        );
    }

    // ---- PricingSnapshot (P0-1): settlement math is category-exact and
    // never fabricates a price. ----

    fn known(in_p: u64, out_p: u64) -> PricingSnapshot {
        PricingSnapshot {
            input_micro_per_token: in_p,
            output_micro_per_token: out_p,
            cache_read_micro_per_token: 0,
            cache_write_micro_per_token: 0,
            pricing_epoch: 7,
            source: PriceSource::Known,
        }
    }

    #[test]
    fn known_snapshot_settles_100k_in_2k_out_exactly() {
        // $15/1M in + $60/1M out == 15/60 microUSD per token: 100k x 15 +
        // 2k x 60 == 1_500_000 + 120_000 == 1_620_000 microUSD == $1.62 —
        // no division anywhere.
        let snap = known(15, 60);
        assert_eq!(
            snap.settle_cost(100_000, 0, 0, 2_000),
            Some(1_620_000),
            "the audit's settlement-truth identity"
        );
        assert_eq!(snap.settle_cost(0, 0, 0, 0), Some(0));
    }

    #[test]
    fn cache_lines_are_billed_at_their_own_price_lines() {
        let snap = PricingSnapshot {
            input_micro_per_token: 15,
            output_micro_per_token: 60,
            cache_read_micro_per_token: 3,
            cache_write_micro_per_token: 7,
            pricing_epoch: 1,
            source: PriceSource::Known,
        };
        // 10k cache reads at $3/1M + 4k writes at $7/1M + 2k out at $60/1M.
        assert_eq!(
            snap.settle_cost(0, 10_000, 4_000, 2_000),
            Some(30_000 + 28_000 + 120_000)
        );
        // Every line stacks: uncached input + cache lines + output.
        assert_eq!(
            snap.settle_cost(1_000, 500, 200, 100),
            Some(15_000 + 1_500 + 1_400 + 6_000)
        );
    }

    #[test]
    fn local_zero_snapshot_settles_to_an_honest_zero() {
        let snap = PricingSnapshot::local_zero();
        assert_eq!(snap.source, PriceSource::LocalZero);
        // An authoritative $0 profile settles to exactly 0 — the honest
        // local-model price, never a fabricated missing number.
        assert_eq!(
            snap.settle_cost(u64::MAX, u64::MAX, u64::MAX, u64::MAX),
            Some(0)
        );
    }

    #[test]
    fn unknown_snapshot_never_produces_a_number() {
        let snap = PricingSnapshot {
            input_micro_per_token: 15,
            output_micro_per_token: 60,
            cache_read_micro_per_token: 0,
            cache_write_micro_per_token: 0,
            pricing_epoch: 1,
            source: PriceSource::Unknown,
        };
        assert_eq!(
            snap.settle_cost(100_000, 0, 0, 2_000),
            None,
            "Unknown must refuse every fabricated total — zero included"
        );
    }

    #[test]
    fn hostile_magnitudes_saturate_never_panic() {
        let snap = PricingSnapshot {
            input_micro_per_token: u64::MAX,
            output_micro_per_token: u64::MAX,
            cache_read_micro_per_token: u64::MAX,
            cache_write_micro_per_token: u64::MAX,
            pricing_epoch: 0,
            source: PriceSource::Known,
        };
        assert_eq!(
            snap.settle_cost(u64::MAX, u64::MAX, u64::MAX, u64::MAX),
            Some(u64::MAX)
        );
    }

    #[test]
    fn snapshot_from_zero_economics_is_local_zero_and_priced_is_known() {
        let zero = PricingSnapshot::from_economics(&ModelEconomics::default());
        assert_eq!(zero.source, PriceSource::LocalZero);
        let priced = ModelEconomics {
            output_price_per_mtok: MicroUsdPerToken::from_dollars_per_million(60),
            ..Default::default()
        };
        let snap = PricingSnapshot::from_economics(&priced);
        assert_eq!(snap.source, PriceSource::Known);
        assert_eq!(snap.output_micro_per_token, 60);
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
            pricing_snapshot: Some(known(15, 60)),
        };
        let v = serde_json::to_value(&decision).unwrap();
        assert_eq!(v["pricing_snapshot"]["source"], serde_json::json!("known"));
        let back: RouteDecision = serde_json::from_value(v.clone()).unwrap();
        assert_eq!(back, decision);
        // Additive wire default: a pre-P0-1 decision JSON (no snapshot
        // field) parses to None — never an error, never a fabricated price.
        let mut legacy = v.clone();
        legacy.as_object_mut().unwrap().remove("pricing_snapshot");
        let back: RouteDecision = serde_json::from_value(legacy).unwrap();
        assert_eq!(back.pricing_snapshot, None);
    }

    #[test]
    fn snapshot_source_wire_spelling_is_snake_case() {
        let v = serde_json::to_value(PriceSource::LocalZero).unwrap();
        assert_eq!(v, serde_json::json!("local_zero"));
        let v = serde_json::to_value(PriceSource::Unknown).unwrap();
        assert_eq!(v, serde_json::json!("unknown"));
        let snap: PricingSnapshot = serde_json::from_value(serde_json::json!({
            "input_micro_per_token": 1, "output_micro_per_token": 2,
            "cache_read_micro_per_token": 0, "cache_write_micro_per_token": 0,
            "pricing_epoch": 0, "source": "known",
        }))
        .unwrap();
        assert_eq!(snap.source, PriceSource::Known);
        // Hostile source values are rejected loudly.
        let mut hostile = serde_json::to_value(snap).unwrap();
        hostile["source"] = serde_json::json!("free!");
        assert!(serde_json::from_value::<PricingSnapshot>(hostile).is_err());
    }
}

// ------------------------------------------------------------ economics

/// MicroUSD per token — the money unit of every price field and of every
/// cost estimate (audit P0-5: the bare-`u64` price field was a dimensional
/// hazard because `microUSD-per-token` and `milliseconds` are both integers
/// and nothing stopped arithmetic from mixing them).
///
/// The name records the true dimension of the stored integer. It is NOT a
/// per-million-token price: the field keeps the per-Token microUSD value,
/// which is *numerically equal* to USD per million tokens (proof below), so
/// the "per_mtok" naming of the price fields stays accurate while the unit
/// is unambiguous.
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
/// `P` microUSD per token — the conversion is the identity — but only when
/// the ingestion site SAYS SO through [`MicroUsdPerToken::from_dollars_per_million`].
/// The alternative spelling
/// [`MicroUsdPerToken::from_micros_per_million_tokens`] divides by `1e6`
/// (rounding DOWN, at ingestion only), so the two never silently collide.
///
/// Sub-integer microUSD per token cannot be represented: any price below
/// $1/Mtok (i.e. `< 1e6` microUSD per million tokens) truncates to 0 at
/// ingestion, which the conversion tests flag loudly.
///
/// # Examples
///
/// ```
/// use faktor_core::model::{MicroUsdPerToken, ModelEconomics};
///
/// // $15 per million tokens is 15 microUSD per token (identity conversion).
/// let input = MicroUsdPerToken::from_dollars_per_million(15);
/// assert_eq!(input, MicroUsdPerToken(15));
/// // 1M tokens at 15 microUSD/token cost exactly $15 in microUSD.
/// assert_eq!(input.saturating_mul(1_000_000), 15_000_000);
///
/// let econ = ModelEconomics {
///     input_price_per_mtok: input,
///     ..Default::default()
/// };
/// assert!(!econ.is_local_zero_cost());
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
    /// tokens* by accident. Value = `dollars_per_million` exactly.
    pub const fn from_dollars_per_million(dollars_per_million: u64) -> Self {
        Self(dollars_per_million)
    }

    /// MicroUSD per million tokens -> microUSD per token.
    ///
    /// `micros_per_million / 1_000_000`, rounding DOWN. Rounding happens
    /// ONLY at this ingestion boundary: every later arithmetic step
    /// ([`MicroUsdPerToken::saturating_mul`]) is exact integer math.
    /// Sub-integer microUSD per token (any price below $1/Mtok) truncates
    /// to 0 here — see the `loud_zero` conversion test.
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

/// Typed price fields (microUSD per token, see [`MicroUsdPerToken`]);
/// 0 = unknown (local models: monetary cost zero, latency still counts).
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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

/// One routable model with its provenance (audit ModelRegistry-lite).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
/// settlement path prices actual usage against (P0-1): `Some` = a pricing
/// authority was consulted and this is its frozen word; `None` = NO pricing
/// authority was consulted (passthrough / RouterUnavailable degradation /
/// the test graph's empty pin) and the runtime must NOT invent one — an
/// unpriced decision under a hard task cost budget fails closed at reserve
/// time, and without a budget its spend is recorded as Unknown, never as a
/// fabricated number.
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

/// What a [`PricingSnapshot`]'s prices rest on (P0-1 settlement truth).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PriceSource {
    /// Real, authoritative price lines (a priced catalog entry).
    Known,
    /// An authoritative zero-price profile (local/free model semantics —
    /// [`ModelEconomics::is_local_zero_cost`]): $0 is the honest price, not
    /// a missing one.
    LocalZero,
    /// A pricing authority was consulted but produced no price (no catalog
    /// entry): settlement must never pretend zero or one.
    Unknown,
}

/// An immutable, route-time capture of the per-token price lines (microUSD
/// per token — numerically equal to USD per million tokens, see
/// [`MicroUsdPerToken`]) the settlement path prices a call's usage against.
/// Captured ONCE at routing so a later catalog repricing can never rewrite
/// what an already-paid call should have cost, and persisted on the cost
/// reservation row so settlement survives daemon restarts.
///
/// Money math is exactly "each reported token category at its own line":
/// `uncached input x input`, `cache reads x cache_read`, `cache writes x
/// cache_write`, output x output (reasoning tokens are counted at the
/// output line — this snapshot carries no reasoning line of its own).
/// [`PricingSnapshot::settle_cost`] returns `None` for [`PriceSource::Unknown`]
/// (the caller refuses or records Unknown — never zero, never one).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PricingSnapshot {
    pub input_micro_per_token: u64,
    pub output_micro_per_token: u64,
    pub cache_read_micro_per_token: u64,
    pub cache_write_micro_per_token: u64,
    /// Catalog pricing epoch the snapshot was cut from; 0 = unversioned
    /// (local/curated prices carry no catalog epoch yet).
    pub pricing_epoch: u64,
    pub source: PriceSource,
}

impl PricingSnapshot {
    /// The authoritative-zero profile of a [`ModelEconomics`] whose prices
    /// are all zero (local/free model semantics, P0-1).
    pub fn local_zero() -> Self {
        Self {
            input_micro_per_token: 0,
            output_micro_per_token: 0,
            cache_read_micro_per_token: 0,
            cache_write_micro_per_token: 0,
            pricing_epoch: 0,
            source: PriceSource::LocalZero,
        }
    }

    /// The snapshot the router cuts from a chosen candidate's economics
    /// (P0-1): zero-price economics are the documented local/free profile
    /// (LocalZero — an authoritative $0, never "no price"); any real price
    /// line makes the snapshot Known.
    pub fn from_economics(economics: &ModelEconomics) -> Self {
        if economics.is_local_zero_cost() {
            return Self::local_zero();
        }
        Self {
            input_micro_per_token: economics.input_price_per_mtok.0,
            output_micro_per_token: economics.output_price_per_mtok.0,
            cache_read_micro_per_token: economics.cache_read_price_per_mtok.0,
            cache_write_micro_per_token: economics.cache_write_price_per_mtok.0,
            pricing_epoch: 0,
            source: PriceSource::Known,
        }
    }

    /// The exact settlement cost of one usage frame at this snapshot's
    /// lines (saturating; never panics on hostile magnitudes). `None` when
    /// the snapshot is [`PriceSource::Unknown`]: no number is honest then —
    /// the caller refuses under a hard budget and records Unknown spend
    /// otherwise.
    pub fn settle_cost(
        &self,
        uncached_input_tokens: u64,
        cache_read_tokens: u64,
        cache_write_tokens: u64,
        output_tokens: u64,
    ) -> Option<u64> {
        if self.source == PriceSource::Unknown {
            return None;
        }
        let mut total = self
            .input_micro_per_token
            .saturating_mul(uncached_input_tokens);
        total = total.saturating_add(
            self.cache_read_micro_per_token
                .saturating_mul(cache_read_tokens),
        );
        total = total.saturating_add(
            self.cache_write_micro_per_token
                .saturating_mul(cache_write_tokens),
        );
        total = total.saturating_add(self.output_micro_per_token.saturating_mul(output_tokens));
        Some(total)
    }
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
