//! Risk computation over semantic proposals (audit 54/58/59).
//!
//! Risk data is provider DATA like everything else: a provider may say what
//! it believes, but it can never grant capability. The only capability
//! movement semantic data can cause is a REDUCTION, enforced by
//! [`capability_intersection`]: the result is always a subset of the parent
//! set, so provider input claiming `Network` on a parent without it stays
//! without it.
//!
//! `Unknown` poisons joins by default: not knowing is not safety. Callers
//! with an explicit policy may opt into treating `Unknown` as absent.

use faktor_core::CapabilitySet;

use crate::types::{RiskLevel, SemanticRisk};

/// How risk joins treat [`RiskLevel::Unknown`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RiskPolicy {
    /// When true, `Unknown` is treated as absent in joins (the caller has
    /// explicitly accepted the unknowns). Default false: Unknown poisons.
    pub allow_unknown: bool,
}

impl RiskPolicy {
    pub const STRICT: Self = Self {
        allow_unknown: false,
    };
    pub const ALLOW_UNKNOWN: Self = Self {
        allow_unknown: true,
    };

    pub const fn new(allow_unknown: bool) -> Self {
        Self { allow_unknown }
    }
}

impl RiskLevel {
    /// True when the level is [`RiskLevel::Unknown`].
    pub const fn is_unknown(self) -> bool {
        matches!(self, Self::Unknown)
    }

    /// True only for [`RiskLevel::Safe`]. An unknown level is never safe.
    pub const fn is_safe(self) -> bool {
        matches!(self, Self::Safe)
    }

    /// Total order over known levels; `None` for [`RiskLevel::Unknown`].
    pub const fn severity_rank(self) -> Option<u8> {
        match self {
            Self::Unknown => None,
            Self::Safe => Some(0),
            Self::Low => Some(1),
            Self::Medium => Some(2),
            Self::High => Some(3),
        }
    }

    /// Join two levels. Unknown poisons the join unless the policy allows
    /// unknown; joining two unknowns stays Unknown even then.
    pub const fn join(self, other: Self, policy: &RiskPolicy) -> Self {
        match (self.severity_rank(), other.severity_rank()) {
            (Some(a), Some(b)) => {
                if a >= b {
                    self
                } else {
                    other
                }
            }
            (None, None) => Self::Unknown,
            (None, Some(_)) | (Some(_), None) => {
                if policy.allow_unknown {
                    match (self, other) {
                        (Self::Unknown, known) => known,
                        (known, Self::Unknown) => known,
                        _ => Self::Unknown,
                    }
                } else {
                    Self::Unknown
                }
            }
        }
    }
}

impl SemanticRisk {
    /// The conservative default: every axis Unknown.
    pub const fn unknown() -> Self {
        Self {
            blast_radius: RiskLevel::Unknown,
            public_surface_delta: RiskLevel::Unknown,
            security_delta: RiskLevel::Unknown,
            unsafe_delta: RiskLevel::Unknown,
            external_effect_delta: RiskLevel::Unknown,
            capability_delta: RiskLevel::Unknown,
            contract_delta: RiskLevel::Unknown,
            concurrency_delta: RiskLevel::Unknown,
            verification_gap: RiskLevel::Unknown,
            resource_constraint_delta: RiskLevel::Unknown,
        }
    }

    /// Field-wise join with another risk under the policy.
    pub fn join(&self, other: &Self, policy: &RiskPolicy) -> Self {
        macro_rules! join_field {
            ($field:ident) => {
                self.$field.join(other.$field, policy)
            };
        }
        Self {
            blast_radius: join_field!(blast_radius),
            public_surface_delta: join_field!(public_surface_delta),
            security_delta: join_field!(security_delta),
            unsafe_delta: join_field!(unsafe_delta),
            external_effect_delta: join_field!(external_effect_delta),
            capability_delta: join_field!(capability_delta),
            contract_delta: join_field!(contract_delta),
            concurrency_delta: join_field!(concurrency_delta),
            verification_gap: join_field!(verification_gap),
            resource_constraint_delta: join_field!(resource_constraint_delta),
        }
    }

    /// The worst level across all axes under the policy. Folding starts from
    /// the first axis, so an all-unknown risk stays Unknown even when the
    /// policy allows unknowns: explicitly accepting unknowns is not evidence.
    pub fn worst(&self, policy: &RiskPolicy) -> RiskLevel {
        let axes = [
            self.blast_radius,
            self.public_surface_delta,
            self.security_delta,
            self.unsafe_delta,
            self.external_effect_delta,
            self.capability_delta,
            self.contract_delta,
            self.concurrency_delta,
            self.verification_gap,
            self.resource_constraint_delta,
        ];
        let mut worst: Option<RiskLevel> = None;
        for level in axes {
            worst = Some(match worst {
                None => level,
                Some(current) => current.join(level, policy),
            });
        }
        worst.unwrap_or(RiskLevel::Unknown)
    }

    /// True only when every axis is known-safe under the policy.
    pub fn is_safe(&self, policy: &RiskPolicy) -> bool {
        self.worst(policy).is_safe()
    }
}

/// Semantic data can only ever REDUCE a capability set: the result is the
/// intersection with the parent (session-granted) set. A semantic provider
/// claiming `Network` can never grant network to a parent that lacks it.
pub const fn capability_intersection(
    parent: CapabilitySet,
    claimed: CapabilitySet,
) -> CapabilitySet {
    parent.intersection(claimed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RiskLevel;
    use faktor_core::{CapabilityKind, CapabilitySet};

    fn all_levels(risk: &SemanticRisk, level: RiskLevel) -> SemanticRisk {
        let mut out = *risk;
        for field in [
            &mut out.blast_radius,
            &mut out.public_surface_delta,
            &mut out.security_delta,
            &mut out.unsafe_delta,
            &mut out.external_effect_delta,
            &mut out.capability_delta,
            &mut out.contract_delta,
            &mut out.concurrency_delta,
            &mut out.verification_gap,
            &mut out.resource_constraint_delta,
        ] {
            *field = level;
        }
        out
    }

    #[test]
    fn unknown_poisons_joins_unless_policy_allows() {
        let strict = RiskPolicy::STRICT;
        let loose = RiskPolicy::ALLOW_UNKNOWN;
        assert_eq!(
            RiskLevel::Unknown.join(RiskLevel::Safe, &strict),
            RiskLevel::Unknown
        );
        assert_eq!(
            RiskLevel::High.join(RiskLevel::Unknown, &strict),
            RiskLevel::Unknown
        );
        assert_eq!(
            RiskLevel::Unknown.join(RiskLevel::Safe, &loose),
            RiskLevel::Safe
        );
        assert_eq!(
            RiskLevel::Unknown.join(RiskLevel::High, &loose),
            RiskLevel::High
        );
        assert_eq!(
            RiskLevel::Unknown.join(RiskLevel::Unknown, &loose),
            RiskLevel::Unknown
        );
        assert_eq!(
            RiskLevel::Medium.join(RiskLevel::High, &strict),
            RiskLevel::High
        );
        assert_eq!(RiskLevel::Low.join(RiskLevel::Low, &strict), RiskLevel::Low);
    }

    #[test]
    fn worst_and_field_join_are_conservative() {
        let unknown = SemanticRisk::unknown();
        assert_eq!(unknown.worst(&RiskPolicy::STRICT), RiskLevel::Unknown);
        assert!(!unknown.is_safe(&RiskPolicy::ALLOW_UNKNOWN));

        let safe = all_levels(&unknown, RiskLevel::Safe);
        assert_eq!(safe.worst(&RiskPolicy::STRICT), RiskLevel::Safe);
        assert!(safe.is_safe(&RiskPolicy::STRICT));

        let mut one_low = safe;
        one_low.security_delta = RiskLevel::Low;
        assert_eq!(one_low.worst(&RiskPolicy::STRICT), RiskLevel::Low);
        assert!(!one_low.is_safe(&RiskPolicy::STRICT));

        let joined = one_low.join(&all_levels(&unknown, RiskLevel::High), &RiskPolicy::STRICT);
        assert_eq!(joined.worst(&RiskPolicy::STRICT), RiskLevel::High);
    }

    #[test]
    fn capability_intersection_only_reduces_and_never_grants() {
        let parent =
            CapabilitySet::of(CapabilityKind::Read).union(CapabilitySet::of(CapabilityKind::Write));
        // Semantic says Network — parent lacks it; result stays without.
        let semantic_network = CapabilitySet::of(CapabilityKind::Network);
        let result = capability_intersection(parent, semantic_network);
        assert!(!result.contains(CapabilityKind::Network));
        assert!(result.is_subset_of(parent));
        assert!(result.is_empty());

        // Semantic claiming what the parent already grants keeps it.
        let semantic_read_network = CapabilitySet::of(CapabilityKind::Read)
            .union(CapabilitySet::of(CapabilityKind::Network));
        let result = capability_intersection(parent, semantic_read_network);
        assert!(result.contains(CapabilityKind::Read));
        assert!(!result.contains(CapabilityKind::Network));
        assert!(result.is_subset_of(parent));

        // Never grants: every combination stays a subset of the parent,
        // including a semantic claim of ALL.
        for claimed in [
            CapabilitySet::EMPTY,
            CapabilitySet::ALL,
            semantic_read_network,
        ] {
            assert!(capability_intersection(parent, claimed).is_subset_of(parent));
        }
        // Unknown capability sets cannot widen anything either.
        assert!(capability_intersection(CapabilitySet::EMPTY, CapabilitySet::ALL).is_empty());
    }
}
