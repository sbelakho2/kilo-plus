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
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RouteDecision {
    pub provider: String,
    pub model: String,
    pub estimated_cost_micro: u64,
    pub estimated_latency_ms: u64,
    pub reasoning: String,
    pub considered: usize,
    pub source: ModelSource,
}
