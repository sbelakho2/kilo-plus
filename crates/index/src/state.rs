//! The durable per-workspace repository index state machine (audits 30/64).
//!
//! The machine is persisted in the store's `index_state` row, one row per
//! workspace, with every legal transition journaled atomically with the row
//! ([`crate::service::IndexService`]). Generation numbers are per-workspace
//! and monotone; a generation becomes visible to readers only through an
//! atomic file swap (the service's publish step), never through the state
//! row alone.
//!
//! Legal machine:
//!
//! ```text
//! NotStarted --build(1)--> Building{1} --publish--> Ready{1}
//! Ready{g} --watcher event--> Dirty{g} --build(g+1)--> Building{g+1} --publish--> Ready{g+1}
//! Building{g} --build failure--> Failed{message}   (row generation = g, the retry target)
//! Failed --retry/next change--> Building{g}        (same target; no renumbering)
//! Building{g} --crash residue--> resume: Building{g}
//! ```
//!
//! `Ready -> Ready(N+1)` is only reachable through `Building`; a rebuild of a
//! ready generation therefore REQUIRES the durable `Ready -> Dirty` hop
//! first. `Failed` is recoverable from every state by a new build. `Building`
//! or `Dirty` persisted at restart "resumes to Building": the row's target
//! generation survives the crash, so the next build produces the same number
//! it was killed on (never a skip, never a reuse).

use serde::{Deserialize, Serialize};

/// Journal kinds written to `index_state_log` (opaque to the store; the
/// machine owns their meaning).
pub const JOURNAL_NOT_STARTED: &str = "not_started";
pub const JOURNAL_BUILDING: &str = "building";
pub const JOURNAL_READY: &str = "ready";
pub const JOURNAL_DIRTY: &str = "dirty";
pub const JOURNAL_FAILED: &str = "failed";
pub const JOURNAL_RESUME: &str = "resume";
pub const JOURNAL_CORRUPT: &str = "corrupt";
pub const JOURNAL_TORN_READY: &str = "torn_ready";

/// One workspace's persisted index state. The enum is stored as opaque JSON
/// in the `index_state.state_json` column; the numeric generation of the
/// row lives in the column alongside it (for `Failed` the column carries the
/// generation the failed build attempted — the retry target — which the enum
/// shape deliberately does not repeat).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum WorkspaceIndexState {
    /// No build has ever completed for this workspace (generation 0).
    #[serde(rename = "not_started")]
    NotStarted,
    /// A build of `generation` is in flight. Durable on purpose: a crash
    /// mid-build leaves this row, and the next attach resumes the SAME
    /// generation.
    #[serde(rename = "building")]
    Building { generation: u64 },
    /// `generation` is complete, durable, and queryable.
    #[serde(rename = "ready")]
    Ready { generation: u64 },
    /// A watcher event arrived after `generation` was published; a rebuild
    /// of `generation + 1` is due. `generation`'s data stays readable while
    /// it is stale.
    #[serde(rename = "dirty")]
    Dirty { generation: u64 },
    /// The last build attempt failed (`message` explains why). Recoverable
    /// by building again from ANY state — on the next change or on an
    /// explicit retry; never silently retried on a timer.
    #[serde(rename = "failed")]
    Failed { message: String },
}

/// Typed transition/decoding errors of the index state machine.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StateError {
    #[error("illegal index state transition: {from} -> {to}")]
    Illegal { from: String, to: String },
    #[error("corrupt persisted index state: {0}")]
    Corrupt(String),
}

impl WorkspaceIndexState {
    /// The generation the state names (Building/Ready/Dirty), if any.
    pub fn generation(&self) -> Option<u64> {
        match self {
            WorkspaceIndexState::Building { generation }
            | WorkspaceIndexState::Ready { generation }
            | WorkspaceIndexState::Dirty { generation } => Some(*generation),
            _ => None,
        }
    }

    /// A published generation exists (`Ready` or `Dirty`, both of which keep
    /// their generation readable while stale).
    pub fn has_published_generation(&self) -> bool {
        matches!(
            self,
            WorkspaceIndexState::Ready { .. } | WorkspaceIndexState::Dirty { .. }
        )
    }

    /// The opaque JSON the store row keeps for this state.
    pub fn to_row_json(&self) -> String {
        // In-process invariant: the state is always serde-serializable
        // (plain data); never reachable from untrusted input.
        serde_json::to_string(self).expect("index state serializes")
    }

    /// Parse a persisted row payload. A corrupt payload is a typed
    /// [`StateError::Corrupt`] — the caller fails OPEN (durable `Failed`
    /// row), never a silent default.
    pub fn from_row_json(raw: &str) -> Result<Self, StateError> {
        serde_json::from_str(raw).map_err(|e| StateError::Corrupt(format!("{e}")))
    }

    /// The generation a build started from this state would target.
    ///
    /// `row_generation` is the numeric generation of the persisted row
    /// (for `Failed` it is the attempted target; the enum does not repeat
    /// it). `Ready` has no legal build target: a rebuild requires the
    /// watcher-driven `Ready -> Dirty` hop first.
    pub fn next_build_target(&self, row_generation: u64) -> Result<u64, StateError> {
        match self {
            WorkspaceIndexState::NotStarted => Ok(1),
            WorkspaceIndexState::Dirty { generation } => Ok(generation.saturating_add(1)),
            WorkspaceIndexState::Building { generation } => Ok(*generation),
            WorkspaceIndexState::Failed { .. } => Ok(row_generation.max(1)),
            WorkspaceIndexState::Ready { .. } => Err(StateError::Illegal {
                from: format!("{self:?} ({row_generation})"),
                to: "Building".to_string(),
            }),
        }
    }

    /// The legal-transition predicate. `row_generation` is the numeric
    /// generation of the CURRENT persisted row; `to_row_generation` is the
    /// numeric generation of the row the transition would produce. Returns
    /// [`StateError::Illegal`] for every hop the machine above forbids.
    pub fn check_transition(
        &self,
        row_generation: u64,
        to: &WorkspaceIndexState,
        to_row_generation: u64,
    ) -> Result<(), StateError> {
        let legal = match (self, to) {
            // NotStarted -> Building(1) is the only way out of NotStarted.
            (WorkspaceIndexState::NotStarted, WorkspaceIndexState::Building { generation: g }) => {
                *g == 1
            }
            // Building -> Ready only for the exact generation being built.
            (
                WorkspaceIndexState::Building { generation: x },
                WorkspaceIndexState::Ready { generation: y },
            ) => x == y,
            // Building -> Building (same generation): crash-resume journal.
            (
                WorkspaceIndexState::Building { generation: x },
                WorkspaceIndexState::Building { generation: y },
            ) => x == y,
            // Any in-flight build may fail; the row then carries the target.
            (WorkspaceIndexState::Building { .. }, WorkspaceIndexState::Failed { .. }) => true,
            // Ready -> Dirty keeps the same generation (stale-but-readable).
            (
                WorkspaceIndexState::Ready { generation: x },
                WorkspaceIndexState::Dirty { generation: y },
            ) => x == y,
            // Dirty -> Building(N+1).
            (
                WorkspaceIndexState::Dirty { generation: x },
                WorkspaceIndexState::Building { generation: y },
            ) => *y == x.saturating_add(1),
            // Failed -> Building toward the SAME attempted target (the row's
            // generation): recovery never renumbers, so a retry that
            // eventually succeeds produces the generation the row promised.
            (
                WorkspaceIndexState::Failed { .. },
                WorkspaceIndexState::Building { generation: y },
            ) => *y == row_generation.max(1) && *y == to_row_generation,
            _ => false,
        };
        if legal {
            Ok(())
        } else {
            Err(StateError::Illegal {
                from: format!("{self:?} (row generation {row_generation})"),
                to: format!("{to:?} (row generation {to_row_generation})"),
            })
        }
    }
}

/// The mirrored persisted row the service keeps in memory: the state plus
/// the row's numeric generation column (authoritative for `Failed` retry
/// targets) and the exact JSON the store row holds (CAS equality must
/// compare byte-exact strings, never re-serializations).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistedIndexState {
    pub state: WorkspaceIndexState,
    pub row_generation: u64,
    /// The exact `state_json` text currently in the store row.
    pub state_json: String,
}

impl PersistedIndexState {
    pub fn parse(state_json: String, row_generation: i64) -> Result<Self, StateError> {
        let state = WorkspaceIndexState::from_row_json(&state_json)?;
        Ok(Self {
            state,
            row_generation: row_generation.max(0) as u64,
            state_json,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(gen: u64) -> WorkspaceIndexState {
        WorkspaceIndexState::Building { generation: gen }
    }
    fn r(gen: u64) -> WorkspaceIndexState {
        WorkspaceIndexState::Ready { generation: gen }
    }
    fn d(gen: u64) -> WorkspaceIndexState {
        WorkspaceIndexState::Dirty { generation: gen }
    }

    #[test]
    fn legal_hop_chain_builds_to_ready_and_renumbers() {
        let start = WorkspaceIndexState::NotStarted;
        start.check_transition(0, &b(1), 1).unwrap();
        assert!(
            start.check_transition(0, &b(2), 2).is_err(),
            "first build is gen 1"
        );
        assert!(
            start.check_transition(0, &r(1), 1).is_err(),
            "no Ready without Building"
        );
        b(1).check_transition(1, &r(1), 1).unwrap();
        b(1).check_transition(1, &r(2), 2).unwrap_err();
        // Ready -> Ready(N+1) only through Building: direct hop illegal.
        r(1).check_transition(1, &r(2), 2).unwrap_err();
        r(1).check_transition(1, &b(2), 2).unwrap_err();
        // Watcher event: Ready -> Dirty same gen; then Dirty -> Building N+1.
        r(1).check_transition(1, &d(1), 1).unwrap();
        assert!(r(1).check_transition(1, &d(2), 2).is_err());
        d(1).check_transition(1, &b(2), 2).unwrap();
        d(2).check_transition(2, &b(3), 3).unwrap();
        d(2).check_transition(2, &b(2), 2).unwrap_err();
        d(2).check_transition(2, &b(4), 4).unwrap_err();
    }

    #[test]
    fn failed_recovers_by_building_the_same_attempted_target() {
        // A build toward gen 3 failed; the row keeps target 3.
        let f = WorkspaceIndexState::Failed {
            message: "boom".into(),
        };
        // Retry target == row generation (attempted target): legal.
        f.check_transition(3, &b(3), 3).unwrap();
        // Never renumber past the failed attempt on recovery.
        assert!(f.check_transition(3, &b(4), 4).is_err());
        // Failed cannot leap straight to Ready.
        assert!(f.check_transition(3, &r(3), 3).is_err());
        // Ready/Dirty may never follow Failed directly.
        assert!(f.check_transition(3, &d(3), 3).is_err());
    }

    #[test]
    fn build_failure_journaling_and_resume_are_legal() {
        b(4).check_transition(
            4,
            &WorkspaceIndexState::Failed {
                message: "io".into(),
            },
            4,
        )
        .unwrap();
        // Crash residue: Building{4} resumed as Building{4}.
        b(4).check_transition(4, &b(4), 4).unwrap();
    }

    #[test]
    fn next_build_targets_are_monotone_and_crash_faithful() {
        assert_eq!(
            WorkspaceIndexState::NotStarted
                .next_build_target(0)
                .unwrap(),
            1
        );
        assert_eq!(d(1).next_build_target(1).unwrap(), 2);
        assert_eq!(d(7).next_build_target(7).unwrap(), 8);
        // Building keeps the exact interrupted target.
        assert_eq!(b(3).next_build_target(3).unwrap(), 3);
        // Failed retries its attempted target (row generation), never 0.
        let f = WorkspaceIndexState::Failed {
            message: "x".into(),
        };
        assert_eq!(f.next_build_target(2).unwrap(), 2);
        assert_eq!(f.next_build_target(0).unwrap(), 1);
        assert!(
            r(5).next_build_target(5).is_err(),
            "ready requires the dirty hop"
        );
    }

    #[test]
    fn row_json_roundtrip_and_corrupt_payload() {
        for s in [
            WorkspaceIndexState::NotStarted,
            b(1),
            r(9),
            d(3),
            WorkspaceIndexState::Failed {
                message: "disk full".into(),
            },
        ] {
            let json = s.to_row_json();
            assert_eq!(WorkspaceIndexState::from_row_json(&json).unwrap(), s);
        }
        // A future/unknown state tag must fail loudly, never silently
        // default to NotStarted.
        let err = WorkspaceIndexState::from_row_json(r#"{ "state": "definitely_future" }"#);
        assert!(matches!(err, Err(StateError::Corrupt(_))));
        let err = WorkspaceIndexState::from_row_json("not json at all");
        assert!(matches!(err, Err(StateError::Corrupt(_))));
        let err = WorkspaceIndexState::from_row_json("");
        assert!(matches!(err, Err(StateError::Corrupt(_))));
    }

    #[test]
    fn corrupt_state_json_never_deserializes_to_a_silent_default() {
        // The adversarial contract: garbage in the store row fails open as
        // an error (the service persists Failed), never as NotStarted.
        let evil = vec![
            "{}",
            r#"{"state":"ready","generation":-1}"#,
            r#"{"state":"ready"}"#,
            r#"{"state":"failed"}"#,
            "[]",
            "null",
            r#"{"state":42}"#,
        ];
        for e in evil {
            let parsed = WorkspaceIndexState::from_row_json(e);
            assert!(
                matches!(parsed, Err(StateError::Corrupt(_))),
                "hostile payload {e:?} must fail open, got {parsed:?}"
            );
        }
    }

    #[test]
    fn max_generation_cannot_wrap() {
        // Dirty{u64::MAX} may never build toward a wrapped 0: the machine
        // rejects the hop, so a generation counter can never roll over.
        assert!(d(u64::MAX).check_transition(u64::MAX, &b(0), 0).is_err());
        // NotStarted targets exactly 1, never 0.
        assert_eq!(
            WorkspaceIndexState::NotStarted
                .next_build_target(0)
                .unwrap(),
            1
        );
    }
}
