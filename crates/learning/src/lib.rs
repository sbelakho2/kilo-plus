//! Faktor project learning (audit items 65-67/92/106).
//!
//! Verified failure/recovery episodes are mined into **project learnings**:
//! structured advice that is
//!
//! - backed ONLY by a completed recovery chain and a durable verification
//!   record (`FailureEpisode::eventual_verified_success`) — "the model said
//!   it fixed it" is not an input this API even accepts;
//! - project/workspace scoped, so identical symbol names in another project
//!   can never collide with a pattern;
//! - DATA, never instruction authority: advice reuses the evidence
//!   provenance rule ([`faktor_evidence::provenance`]), and rendering wraps
//!   every advice field in a data block with marker-neutralizing escaping;
//! - explicitly invalidated (`Never` / `SourceHashChanged` /
//!   `EvidenceStale` / `TaskEnd`), with confidence shrinking conservatively
//!   under a documented Beta prior instead of trusting 1-2 samples;
//! - bounded at every layer: bounded descriptors, bounded fingerprint
//!   input, a bounded in-memory store, and newest-first rendering that
//!   stops at the token budget without loading the corpus.
//!
//! The failure-aware omission prior ([`omission_risk_of`],
//! [`LearningService::omission_risk_index`]) is clamped to `[1, 2]` and
//! always finite: it can only ever protect context, never demote it.
//!
//! Persistence is a trait seam ([`LearningStore`]): this crate owns no
//! schema and adds no migrations. [`MemoryLearningStore`] is the in-process
//! implementation; [`SessionLearningStore`] is the durable adapter over the
//! session's typed `learning_record` ledger entries (episodes, stored
//! learnings and removal tombstones — bounded payloads, strict decode,
//! loud on corruption).

#![forbid(unsafe_code)]

use std::fmt;

pub mod episode;
pub mod miner;
pub mod service;
pub mod store;

pub use episode::{
    ActionDescriptor, ActionFingerprint, AssumptionDelta, EnvironmentFingerprint, EpisodeId,
    FailureDescriptor, FailureEpisode, FailureFingerprint, ProjectScope, TaskClass,
};
pub use miner::{
    bayesian_confidence_ppm, mine, InvalidationContext, InvalidationRule, LearningPattern,
    ProjectLearning, StructuredAdvice, CONFIDENCE_TRUSTED_PPM, PRIOR_ALPHA, PRIOR_BETA,
};
pub use service::{
    adjusted_gain, context_prior, omission_risk_for_keys, omission_risk_of, ContextNecessity,
    LearningService, RenderOutcome, OMISSION_RISK_MAX, OMISSION_RISK_NEUTRAL, RENDER_PAGE,
    RENDER_SCAN_CAP,
};
pub use store::{
    LearningId, LearningStore, MemoryLearningStore, SessionLearningStore, DEFAULT_MEMORY_CAPACITY,
    MAX_SESSION_EPISODES,
};

/// Typed learning error. Every rejection is structural: malformed input,
/// an exceeded explicit bound, a refused invariant (advice that would carry
/// instruction authority), or a store-seam failure. Nothing here panics on
/// hostile input and nothing silently defaults.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LearningError {
    /// Input could not be turned into valid learning state (empty class,
    /// empty tool, ...).
    Malformed(String),
    /// A bounded field or list exceeded its explicit bound.
    Oversized {
        what: String,
        max: usize,
        actual: usize,
    },
    /// A hard invariant was violated (e.g. rendering instruction-authority
    /// advice through the data channel).
    Refused(String),
    /// The persistence seam failed (durable adapter I/O, row corruption).
    Store(String),
}

impl fmt::Display for LearningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LearningError::Malformed(message) => write!(f, "malformed learning input: {message}"),
            LearningError::Oversized { what, max, actual } => {
                write!(
                    f,
                    "learning {what} oversized: {actual} exceeds the {max} bound"
                )
            }
            LearningError::Refused(message) => write!(f, "learning refused: {message}"),
            LearningError::Store(message) => write!(f, "learning store failed: {message}"),
        }
    }
}

impl std::error::Error for LearningError {}

/// Non-zero `u64` id newtypes in the style of `faktor-core`: zero is
/// rejected by the constructor and by deserialization, never silently
/// accepted as a valid id.
macro_rules! learning_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[repr(transparent)]
        pub struct $name(u64);

        impl $name {
            /// Construct from a non-zero raw id.
            #[inline]
            pub const fn new(raw: u64) -> Self {
                assert!(raw != 0, concat!(stringify!($name), " cannot be 0"));
                Self(raw)
            }

            /// The raw `u64`.
            #[inline]
            pub const fn raw(self) -> u64 {
                self.0
            }
        }

        impl ::std::fmt::Display for $name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        impl From<$name> for u64 {
            fn from(value: $name) -> u64 {
                value.0
            }
        }

        impl ::serde::Serialize for $name {
            fn serialize<S: ::serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.serialize_u64(self.0)
            }
        }

        impl<'de> ::serde::Deserialize<'de> for $name {
            fn deserialize<D: ::serde::Deserializer<'de>>(
                deserializer: D,
            ) -> Result<Self, D::Error> {
                let raw: u64 = ::serde::Deserialize::deserialize(deserializer)?;
                if raw == 0 {
                    Err(::serde::de::Error::custom(concat!(
                        stringify!($name),
                        " cannot be 0"
                    )))
                } else {
                    Ok(Self(raw))
                }
            }
        }
    };
}

pub(crate) use learning_id;
