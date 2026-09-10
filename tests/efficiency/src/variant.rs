//! The audit 84-88 efficiency flag profile and its explicit wiring honesty.

use serde::{Deserialize, Serialize};

/// One efficiency variant: the five audit flags. Two arms of a pair differ
/// ONLY by these booleans — same corpus, same starting revisions, same
/// provider seed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct EfficiencyVariant {
    /// Compressed Context Representation: tool/evidence payloads pass
    /// through the real `faktor-evidence` compression policy before being
    /// sent.
    pub ccr: bool,
    /// Semantic context: the real `faktor-context` information-gain
    /// selection picks which evidence candidates fit the token budget.
    pub semantic_context: bool,
    /// Failure learning: a repeated known failure is recognized from durable
    /// `faktor-memory` facts and its learned fix is applied without another
    /// repair round.
    pub failure_learning: bool,
    /// Typed handoff: re-sent history is replaced by the real
    /// `faktor-context` `TaskContextProjection` render of the durable task
    /// rows.
    pub typed_handoff: bool,
    /// Rework-aware routing: repair calls are chosen with the real
    /// `faktor-router` verified-outcome rework math over durable
    /// `model_outcome_stats`.
    pub rework_aware_routing: bool,
}

/// One named flag, for sweeps and reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EfficiencyFlag {
    Ccr,
    SemanticContext,
    FailureLearning,
    TypedHandoff,
    ReworkAwareRouting,
}

/// All five flags in a fixed order (stable reporting).
pub const FLAGS: [EfficiencyFlag; 5] = [
    EfficiencyFlag::Ccr,
    EfficiencyFlag::SemanticContext,
    EfficiencyFlag::FailureLearning,
    EfficiencyFlag::TypedHandoff,
    EfficiencyFlag::ReworkAwareRouting,
];

impl EfficiencyFlag {
    pub fn label(self) -> &'static str {
        match self {
            EfficiencyFlag::Ccr => "ccr",
            EfficiencyFlag::SemanticContext => "semantic_context",
            EfficiencyFlag::FailureLearning => "failure_learning",
            EfficiencyFlag::TypedHandoff => "typed_handoff",
            EfficiencyFlag::ReworkAwareRouting => "rework_aware_routing",
        }
    }

    pub fn get(self, variant: EfficiencyVariant) -> bool {
        match self {
            EfficiencyFlag::Ccr => variant.ccr,
            EfficiencyFlag::SemanticContext => variant.semantic_context,
            EfficiencyFlag::FailureLearning => variant.failure_learning,
            EfficiencyFlag::TypedHandoff => variant.typed_handoff,
            EfficiencyFlag::ReworkAwareRouting => variant.rework_aware_routing,
        }
    }

    /// The variant with exactly this flag set to `on`.
    pub fn with(self, variant: EfficiencyVariant, on: bool) -> EfficiencyVariant {
        let mut out = variant;
        match self {
            EfficiencyFlag::Ccr => out.ccr = on,
            EfficiencyFlag::SemanticContext => out.semantic_context = on,
            EfficiencyFlag::FailureLearning => out.failure_learning = on,
            EfficiencyFlag::TypedHandoff => out.typed_handoff = on,
            EfficiencyFlag::ReworkAwareRouting => out.rework_aware_routing = on,
        }
        out
    }
}

impl EfficiencyVariant {
    /// The control arm: every flag off.
    pub const BASELINE: Self = Self {
        ccr: false,
        semantic_context: false,
        failure_learning: false,
        typed_handoff: false,
        rework_aware_routing: false,
    };

    pub const fn is_baseline(self) -> bool {
        !self.ccr
            && !self.semantic_context
            && !self.failure_learning
            && !self.typed_handoff
            && !self.rework_aware_routing
    }

    pub fn all_on() -> Self {
        Self {
            ccr: true,
            semantic_context: true,
            failure_learning: true,
            typed_handoff: true,
            rework_aware_routing: true,
        }
    }

    /// Stable 5-bit encoding (flag order = [`FLAGS`]); useful as a map key.
    pub fn bits(self) -> u8 {
        let mut bits = 0u8;
        for (i, flag) in FLAGS.iter().enumerate() {
            if flag.get(self) {
                bits |= 1 << i;
            }
        }
        bits
    }

    /// Stable report name: `baseline` or the enabled flag labels joined by
    /// `+`.
    pub fn name(self) -> String {
        if self.is_baseline() {
            return "baseline".to_string();
        }
        FLAGS
            .iter()
            .filter(|flag| flag.get(self))
            .map(|flag| flag.label())
            .collect::<Vec<_>>()
            .join("+")
    }
}

/// How one variant flag is actually wired, stated explicitly so no report
/// can silently claim production behavior. The global wiring mode is
/// `WiredIntoHarness`: the flag's real component runs in the harness
/// pipeline and is measured through durable rows; the production daemon does
/// not consult the flag (there is no production feature switch yet).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WiringStatus {
    /// The flag's real production component runs in the harness and its
    /// effect is measured from durable rows. The production runtime does not
    /// consult the flag itself.
    WiredIntoHarness,
}

impl WiringStatus {
    pub fn label(self) -> &'static str {
        match self {
            WiringStatus::WiredIntoHarness => "wired-into-harness",
        }
    }
}

/// One flag's honesty row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VariantHonesty {
    pub flag: EfficiencyFlag,
    pub status: WiringStatus,
    /// The exact production component the harness drives for this flag.
    pub component: &'static str,
}

/// The explicit wiring ledger of all five flags. FR-84-88 review gate: any
/// future change that wires a flag into production must update its row here.
pub fn wiring_honesty() -> [VariantHonesty; 5] {
    [
        VariantHonesty {
            flag: EfficiencyFlag::Ccr,
            status: WiringStatus::WiredIntoHarness,
            component: "faktor-evidence::compress (real CCR policy + backing records)",
        },
        VariantHonesty {
            flag: EfficiencyFlag::SemanticContext,
            status: WiringStatus::WiredIntoHarness,
            component: "faktor-context::information::select_by_information (real information-gain selection)",
        },
        VariantHonesty {
            flag: EfficiencyFlag::FailureLearning,
            status: WiringStatus::WiredIntoHarness,
            component: "faktor-memory::SessionMemory (real durable memory_fact read/write)",
        },
        VariantHonesty {
            flag: EfficiencyFlag::TypedHandoff,
            status: WiringStatus::WiredIntoHarness,
            component: "faktor-context::ledger::TaskContextProjection (real durable-row projection render)",
        },
        VariantHonesty {
            flag: EfficiencyFlag::ReworkAwareRouting,
            status: WiringStatus::WiredIntoHarness,
            component: "faktor-router::outcomes::work_cost_estimate over durable model_outcome_stats",
        },
    ]
}
