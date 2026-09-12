//! Shared, bounded child-execution vocabulary.
//!
//! This module is the ONE definition of two durable child concepts used by
//! the session projection and the orchestrator runtime:
//!
//! - [`ChildBlocker`]: why a non-terminal child is not running. The `kind` is
//!   a bounded machine enum (never free-form text); every string field is
//!   strictly length-bounded and rejected BEFORE any durable write. The
//!   session v23 child-runtime columns serialize this type's fields
//!   one-for-one, and decode re-parses the kind through the enum — a
//!   hostile/unknown kind is a loud decode failure, never a silent string.
//! - [`ExecutionPhase`]: the coarse phase of a child drive, persisted at
//!   safe boundaries into the child's durable drive-state row. Additive:
//!   absent rows decode as [`ExecutionPhase::Planning`].
//!
//! Decode validators (additive): [`validate_child_runtime_state`] enforces
//! the durable invariant `state == "blocked" <=> blocker.is_some()` plus the
//! closed lifecycle-tag vocabulary, so a corrupt child-runtime row fails
//! loudly at decode instead of fabricating a state.

use serde::{Deserialize, Serialize};

/// Strict bounds of the durable child blocker text (v23 typed projection).
pub const MAX_CHILD_BLOCKER_REASON_CHARS: usize = 512;
pub const MAX_CHILD_BLOCKER_DEPENDENCY_CHARS: usize = 64;
pub const MAX_CHILD_BLOCKER_RESOLUTION_CHARS: usize = 512;

/// The bounded vocabulary of a durable child blocker kind. A closed enum:
/// an unknown durable kind string is corrupt and decode-rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockerKind {
    /// The child waits on an unfinished work-item dependency.
    Dependency,
    /// The child waits on a pending permission decision.
    Permission,
    /// The child waits on token/monetary capacity.
    Budget,
    /// The child waits on a blocking verification condition.
    Verification,
    /// The child waits on an external dependency outside this run (a remote
    /// service, another machine, a human-owned process).
    External,
    /// The child is blocked on a condition the runtime could not classify.
    /// Distinct from an UNKNOWN DURABLE TAG: this is a valid, decoded
    /// blocker whose cause is genuinely undetermined.
    Unknown,
}

impl BlockerKind {
    pub const ALL: [BlockerKind; 6] = [
        BlockerKind::Dependency,
        BlockerKind::Permission,
        BlockerKind::Budget,
        BlockerKind::Verification,
        BlockerKind::External,
        BlockerKind::Unknown,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            BlockerKind::Dependency => "dependency",
            BlockerKind::Permission => "permission",
            BlockerKind::Budget => "budget",
            BlockerKind::Verification => "verification",
            BlockerKind::External => "external",
            BlockerKind::Unknown => "unknown",
        }
    }

    /// Parse a durable kind tag; `None` for anything outside the vocabulary
    /// (the caller turns that into a loud decode error). The six known tags
    /// are exactly the legacy string values the pre-typed rows wrote.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "dependency" => Some(BlockerKind::Dependency),
            "permission" => Some(BlockerKind::Permission),
            "budget" => Some(BlockerKind::Budget),
            "verification" => Some(BlockerKind::Verification),
            "external" => Some(BlockerKind::External),
            "unknown" => Some(BlockerKind::Unknown),
            _ => None,
        }
    }
}

impl std::fmt::Display for BlockerKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The typed durable blocker of one blocked child: WHY it is not running.
/// Bounded and validated — hostile oversized text is a typed reject before
/// any durable write, and an unknown kind never decodes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChildBlocker {
    pub kind: BlockerKind,
    pub reason: String,
    #[serde(default)]
    pub dependency: Option<String>,
    #[serde(default)]
    pub resolution: Option<String>,
    #[serde(default)]
    pub last_progress_ms: Option<i64>,
}

impl ChildBlocker {
    /// A blocker with a kind tag, bounded reason and suggested resolution.
    pub fn new(
        kind: BlockerKind,
        reason: impl Into<String>,
        resolution: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            reason: reason.into(),
            dependency: None,
            resolution: Some(resolution.into()),
            last_progress_ms: None,
        }
    }

    /// A dependency blocker naming the work item it waits on.
    pub fn dependency(
        item_id: &str,
        reason: impl Into<String>,
        resolution: impl Into<String>,
    ) -> Self {
        Self {
            kind: BlockerKind::Dependency,
            reason: reason.into(),
            dependency: Some(item_id.to_string()),
            resolution: Some(resolution.into()),
            last_progress_ms: None,
        }
    }

    /// The durable ledger reason string this blocker records (bounded by
    /// [`MAX_CHILD_BLOCKER_REASON_CHARS`], so every ledger append accepts it).
    pub fn ledger_reason(&self) -> String {
        self.reason.clone()
    }

    /// Structural validation with strict bounds (hostile text is a typed
    /// reject BEFORE anything durable is written — never a silent truncate).
    pub fn validate(&self) -> Result<(), crate::Error> {
        check_blocker_text(
            "blocker reason",
            &self.reason,
            MAX_CHILD_BLOCKER_REASON_CHARS,
        )?;
        if let Some(dep) = &self.dependency {
            check_blocker_text(
                "blocker dependency",
                dep,
                MAX_CHILD_BLOCKER_DEPENDENCY_CHARS,
            )?;
        }
        if let Some(res) = &self.resolution {
            check_blocker_text(
                "blocker resolution",
                res,
                MAX_CHILD_BLOCKER_RESOLUTION_CHARS,
            )?;
        }
        if self.last_progress_ms.is_some_and(|ms| ms < 0) {
            return Err(crate::Error::malformed(
                "blocker last_progress_ms must be non-negative",
            ));
        }
        Ok(())
    }
}

fn check_blocker_text(field: &str, value: &str, max: usize) -> Result<(), crate::Error> {
    if value.trim().is_empty() {
        return Err(crate::Error::malformed(format!(
            "{field} must not be empty or whitespace-only"
        )));
    }
    if value.chars().count() > max {
        return Err(crate::Error::oversized(format!(
            "{field} of {} characters exceeds {max}",
            value.chars().count()
        )));
    }
    if value.chars().any(|c| c.is_control()) {
        return Err(crate::Error::malformed(format!(
            "{field} carries control characters"
        )));
    }
    Ok(())
}

/// The closed durable child lifecycle-tag vocabulary of the v23
/// `child_runtime.state` column (mirrors the orchestrator's `ChildState`).
pub const CHILD_LIFECYCLE_TAGS: [&str; 7] = [
    "running",
    "paused",
    "waiting",
    "blocked",
    "done",
    "cancelled",
    "failed",
];

/// True when `state` is one of [`CHILD_LIFECYCLE_TAGS`].
pub fn child_lifecycle_tag_is_known(state: &str) -> bool {
    CHILD_LIFECYCLE_TAGS.contains(&state)
}

/// The durable decode invariant of one child-runtime projection row:
/// the state tag is from the closed vocabulary AND
/// `state == "blocked" <=> blocker.is_some()`. A blocker that is present is
/// validated (bounded text, non-negative progress). Corrupt rows are a loud
/// typed error — the caller must never fabricate a state from them.
pub fn validate_child_runtime_state(
    state: &str,
    blocker: Option<&ChildBlocker>,
) -> Result<(), crate::Error> {
    if !child_lifecycle_tag_is_known(state) {
        return Err(crate::Error::malformed(format!(
            "child runtime state {state:?} is not one of running|paused|waiting|blocked|done|cancelled|failed"
        )));
    }
    match (state, blocker) {
        ("blocked", None) => Err(crate::Error::malformed(
            "child runtime row is state=blocked without a blocker; refusing the corrupt row",
        )),
        ("blocked", Some(b)) => b.validate(),
        (_, Some(_)) => Err(crate::Error::malformed(format!(
            "child runtime row is state={state} but carries a blocker; refusing the corrupt row"
        ))),
        (_, None) => Ok(()),
    }
}

/// The coarse execution phase of one child drive, persisted additively at
/// safe boundaries into the child's durable drive-state row. Closed enum:
/// an unknown durable value is a loud decode failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionPhase {
    /// Before the first route/plan of the drive.
    #[default]
    Planning,
    /// Context/evidence preparation.
    Context,
    /// Provider reasoning (model call).
    Reasoning,
    /// Mutating tool/edit work.
    Coding,
    /// Non-mutating tool work.
    Tool,
    /// Verification of the turn's change.
    Verifying,
    /// Settlement/integration of a finished child.
    Integrating,
    /// Independent review.
    Review,
}

impl ExecutionPhase {
    pub const ALL: [ExecutionPhase; 8] = [
        ExecutionPhase::Planning,
        ExecutionPhase::Context,
        ExecutionPhase::Reasoning,
        ExecutionPhase::Coding,
        ExecutionPhase::Tool,
        ExecutionPhase::Verifying,
        ExecutionPhase::Integrating,
        ExecutionPhase::Review,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            ExecutionPhase::Planning => "planning",
            ExecutionPhase::Context => "context",
            ExecutionPhase::Reasoning => "reasoning",
            ExecutionPhase::Coding => "coding",
            ExecutionPhase::Tool => "tool",
            ExecutionPhase::Verifying => "verifying",
            ExecutionPhase::Integrating => "integrating",
            ExecutionPhase::Review => "review",
        }
    }

    /// Parse a durable phase tag; `None` for anything outside the closed
    /// vocabulary.
    pub fn parse(value: &str) -> Option<Self> {
        ExecutionPhase::ALL
            .into_iter()
            .find(|p| p.as_str() == value)
    }
}

impl std::fmt::Display for ExecutionPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocker_kind_is_closed_and_roundtrips_its_durable_tag() {
        for kind in BlockerKind::ALL {
            assert_eq!(BlockerKind::parse(kind.as_str()), Some(kind));
            let json = serde_json::to_string(&kind).unwrap();
            assert_eq!(json, format!("\"{}\"", kind.as_str()));
            assert_eq!(serde_json::from_str::<BlockerKind>(&json).unwrap(), kind);
        }
        for hostile in ["", "Blocked", "dependency ", "zombie", "permission:1"] {
            assert!(BlockerKind::parse(hostile).is_none(), "{hostile:?}");
            assert!(serde_json::from_str::<BlockerKind>(&format!("\"{hostile}\"")).is_err());
        }
    }

    #[test]
    fn legacy_string_kinds_delegate_to_the_typed_shape_additively() {
        // The pre-typed durable projection stored the kind as free text. Every
        // tag the old writers produced (plus the two bounded additions) must
        // decode to its typed variant through the SAME string surface the
        // legacy rows carry — no migration, no re-encode.
        for (tag, kind) in [
            ("dependency", BlockerKind::Dependency),
            ("permission", BlockerKind::Permission),
            ("budget", BlockerKind::Budget),
            ("verification", BlockerKind::Verification),
            ("external", BlockerKind::External),
            ("unknown", BlockerKind::Unknown),
        ] {
            assert_eq!(BlockerKind::parse(tag), Some(kind), "{tag}");
            assert_eq!(kind.as_str(), tag);
        }
        // A legacy JSON row (string enum) decodes through serde directly.
        for kind in BlockerKind::ALL {
            let body = format!(
                r#"{{"kind":"{}","reason":"r","dependency":null,"resolution":null,"last_progress_ms":null}}"#,
                kind.as_str()
            );
            assert_eq!(
                serde_json::from_str::<ChildBlocker>(&body).unwrap().kind,
                kind
            );
        }
    }

    #[test]
    fn blocker_decodes_hostile_values_loudly() {
        // Unknown kind is a decode failure, never a fallback.
        assert!(
            serde_json::from_str::<ChildBlocker>(
                r#"{"kind":"hostile","reason":"r","dependency":null,"resolution":null,"last_progress_ms":null}"#
            )
            .is_err()
        );
        // Extra fields are refused.
        assert!(serde_json::from_str::<ChildBlocker>(
            r#"{"kind":"budget","reason":"r","resolution":null,"extra":1}"#
        )
        .is_err());
        // Empty/whitespace/control text and negative stamps are typed rejects.
        let empty = ChildBlocker::new(BlockerKind::Budget, "  ", "raise it");
        assert_eq!(
            empty.validate().unwrap_err().kind,
            crate::ErrorKind::Malformed
        );
        let huge = ChildBlocker::new(
            BlockerKind::Budget,
            "x".repeat(MAX_CHILD_BLOCKER_REASON_CHARS + 1),
            "raise it",
        );
        assert_eq!(
            huge.validate().unwrap_err().kind,
            crate::ErrorKind::Oversized
        );
        let negative = ChildBlocker {
            kind: BlockerKind::Dependency,
            reason: "waiting".into(),
            dependency: Some("a".into()),
            resolution: Some("wait".into()),
            last_progress_ms: Some(-1),
        };
        assert_eq!(
            negative.validate().unwrap_err().kind,
            crate::ErrorKind::Malformed
        );
        assert!(
            ChildBlocker::dependency("a", "waiting on work item \"a\"", "wait")
                .validate()
                .is_ok()
        );
    }

    #[test]
    fn child_runtime_state_blocker_equivalence_is_enforced() {
        let blocker = ChildBlocker::new(BlockerKind::Permission, "waiting", "resolve it");
        // Blocked <=> blocker present.
        assert!(validate_child_runtime_state("blocked", None).is_err());
        assert!(validate_child_runtime_state("blocked", Some(&blocker)).is_ok());
        for state in [
            "running",
            "paused",
            "waiting",
            "done",
            "cancelled",
            "failed",
        ] {
            assert!(validate_child_runtime_state(state, None).is_ok(), "{state}");
            assert!(
                validate_child_runtime_state(state, Some(&blocker)).is_err(),
                "{state} with a blocker must fail"
            );
        }
        // Unknown lifecycle tags are corrupt.
        assert!(validate_child_runtime_state("zombie", None).is_err());
        // A present blocker is still validated.
        let corrupt = ChildBlocker {
            kind: BlockerKind::Budget,
            reason: String::new(),
            dependency: None,
            resolution: None,
            last_progress_ms: None,
        };
        assert!(validate_child_runtime_state("blocked", Some(&corrupt)).is_err());
    }

    #[test]
    fn execution_phase_is_closed_and_defaults_to_planning() {
        for phase in ExecutionPhase::ALL {
            assert_eq!(ExecutionPhase::parse(phase.as_str()), Some(phase));
            assert_eq!(
                serde_json::from_str::<ExecutionPhase>(&format!("\"{}\"", phase.as_str())).unwrap(),
                phase
            );
        }
        assert_eq!(ExecutionPhase::default(), ExecutionPhase::Planning);
        assert!(ExecutionPhase::parse("hostile").is_none());
        assert!(serde_json::from_str::<ExecutionPhase>("\"blocked\"").is_err());
    }
}
