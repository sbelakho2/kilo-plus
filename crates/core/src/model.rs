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
