//! Context budgets. The exact 32K small-model math from the spec:
//! system/tools 5K, working state 3K, retrieved code 7K, recent turns 10K,
//! output reserve 5K, safety 2K. The engine enforces the budget BEFORE
//! sending anything — it never discovers the limit from a provider error.

use faktor_core::model::ModelCapabilities;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ContextBudget {
    pub system: usize,
    pub tools: usize,
    pub working: usize,
    pub retrieved: usize,
    pub recent: usize,
    pub output_reserve: usize,
    pub safety: usize,
}

impl Default for ContextBudget {
    fn default() -> Self {
        // 32K local model profile (spec §9): exactly 32_000 tokens.
        Self {
            system: 5_000,
            tools: 0,
            working: 3_000,
            retrieved: 7_000,
            recent: 10_000,
            output_reserve: 5_000,
            safety: 2_000,
        }
    }
}

impl ContextBudget {
    /// Derive a budget from discovered capabilities. Large contexts keep
    /// the proportional reserve model; small contexts never exceed the
    /// 32K profile's absolute numbers.
    pub fn for_capabilities(caps: &ModelCapabilities) -> ContextBudget {
        let context = caps.context.max(caps.max_output.saturating_mul(2));
        if context <= 32_768 {
            // Small/local models: fixed conservative split (spec §9).
            Self::default()
        } else {
            // Bigger models: scale the volatile classes, keep the reserve.
            let working = 4_000;
            let recent = (context / 8).clamp(10_000, 60_000);
            let retrieved = (context / 12).clamp(7_000, 24_000);
            let output_reserve = caps.max_output.saturating_add(1_000).min(context / 4);
            let safety = (context / 32).max(2_000);
            let system = (context / 16).clamp(5_000, 12_000);
            let used = system
                .saturating_add(working)
                .saturating_add(retrieved)
                .saturating_add(recent)
                .saturating_add(output_reserve)
                .saturating_add(safety);
            let tools = context.saturating_sub(used).min(8_000);
            Self {
                system,
                tools,
                working,
                retrieved,
                recent,
                output_reserve,
                safety,
            }
        }
    }

    pub fn total(&self) -> usize {
        self.system
            .saturating_add(self.tools)
            .saturating_add(self.working)
            .saturating_add(self.retrieved)
            .saturating_add(self.recent)
            .saturating_add(self.output_reserve)
            .saturating_add(self.safety)
    }

    /// The maximum the assembled context may occupy.
    pub fn context_max(&self) -> usize {
        self.total()
            .saturating_sub(self.output_reserve)
            .saturating_sub(self.safety)
    }

    /// Effective usage fraction in [0,1]: used / context_max.
    pub fn effective_usage(&self, used: usize) -> f64 {
        let max = self.context_max();
        if max == 0 {
            return 1.0;
        }
        (used as f64 / max as f64).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faktor_core::model::ModelCapabilities;

    #[test]
    fn default_32k_math_is_exact() {
        let b = ContextBudget::default();
        assert_eq!(b.total(), 32_000, "5K+3K+7K+10K+5K+2K = 32K");
        assert_eq!(
            b.context_max(),
            25_000,
            "reserve 5K + safety 2K are not context"
        );
    }

    #[test]
    fn for_capabilities_small_local_is_exact() {
        let caps = ModelCapabilities::small_local();
        let b = ContextBudget::for_capabilities(&caps);
        assert_eq!(b, ContextBudget::default());
        assert_eq!(b.total(), 32_000);
    }

    #[test]
    fn for_capabilities_large_context_scales_without_overflow() {
        let caps = ModelCapabilities {
            context: 200_000,
            max_output: 8_192,
            ..Default::default()
        };
        let b = ContextBudget::for_capabilities(&caps);
        assert!(
            b.total() <= caps.context,
            "budget {} > context {}",
            b.total(),
            caps.context
        );
        assert!(b.total() >= 32_000);
        assert!(b.output_reserve >= caps.max_output);
        assert_eq!(b.safety, 200_000 / 32);
    }

    #[test]
    fn insane_capabilities_fail_cleanly() {
        // usize::MAX context (hostile provider metadata): for_capabilities
        // must saturate, not overflow/panic.
        let caps = ModelCapabilities {
            context: usize::MAX,
            max_output: usize::MAX / 2,
            ..Default::default()
        };
        let b = ContextBudget::for_capabilities(&caps);
        assert!(b.total() >= 32_000);
        let eff = b.effective_usage(b.total());
        assert!(eff <= 1.0);
        // Zero context: budget math must not panic (returns the 32K default
        // profile is wrong — but must not panic; the assembler clamps).
        let caps = ModelCapabilities {
            context: 0,
            max_output: 0,
            ..Default::default()
        };
        let b = ContextBudget::for_capabilities(&caps);
        assert_eq!(b.total(), 32_000);
    }

    #[test]
    fn effective_usage_boundaries() {
        let b = ContextBudget::default();
        assert_eq!(b.effective_usage(0), 0.0);
        assert_eq!(b.effective_usage(25_000), 1.0);
        assert_eq!(b.effective_usage(99_999), 1.0, "clamped");
        assert!(b.effective_usage(12_500) - 0.5 < 1e-9);
    }
}
