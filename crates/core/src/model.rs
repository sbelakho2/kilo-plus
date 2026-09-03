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
}

// ------------------------------------------------------------ economics

/// Per-million-token prices in the same integer currency; 0 = unknown
/// (local models: monetary cost zero, latency still counts).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelEconomics {
    pub input_price_per_mtok: u64,
    pub output_price_per_mtok: u64,
    pub cache_read_price_per_mtok: u64,
    pub cache_write_price_per_mtok: u64,
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
            input_price_per_mtok: 0,
            output_price_per_mtok: 0,
            cache_read_price_per_mtok: 0,
            cache_write_price_per_mtok: 0,
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
        self.input_price_per_mtok == 0
            && self.output_price_per_mtok == 0
            && self.cache_read_price_per_mtok == 0
            && self.cache_write_price_per_mtok == 0
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
