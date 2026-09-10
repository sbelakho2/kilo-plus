//! The explicit session state machine.
//!
//! There is **no generic `await Promise` that determines application state**.
//! Every session is an explicit state machine; every transition is validated.

use crate::error::{Error, ErrorKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentState {
    Idle,
    Preparing,
    BuildingContext,
    WaitingForModel,
    Streaming,
    ToolRequested,
    WaitingForPermission,
    ExecutingTool,
    Validating,
    UpdatingMemory,
    ReadyForNextTurn,
    Completed,
    Cancelled,
    FailedRecoverable,
    FailedPermanent,
    NeedsUserInput,
    Suspended,
}

impl AgentState {
    /// All states a session may legally transition into from `self`.
    /// The state machine is intentionally conservative: unknown or
    /// ambiguous transitions are rejected loudly instead of silently
    /// corrupting the session.
    pub fn allowed_transitions(self) -> &'static [AgentState] {
        use AgentState::*;
        match self {
            Idle => &[Preparing, Suspended, Completed, Cancelled],
            Preparing => &[BuildingContext, FailedRecoverable, Cancelled, Suspended],
            BuildingContext => &[WaitingForModel, FailedRecoverable, Cancelled, Suspended],
            WaitingForModel => &[
                Streaming,
                FailedRecoverable,
                Cancelled,
                Suspended,
                NeedsUserInput,
            ],
            Streaming => &[
                ToolRequested,
                Validating,
                WaitingForModel,
                FailedRecoverable,
                Cancelled,
                Suspended,
            ],
            ToolRequested => &[
                WaitingForPermission,
                ExecutingTool,
                Validating,
                Cancelled,
                Suspended,
            ],
            WaitingForPermission => &[
                ExecutingTool,
                ReadyForNextTurn,
                Cancelled,
                Suspended,
                NeedsUserInput,
            ],
            ExecutingTool => &[
                Validating,
                ToolRequested,
                FailedRecoverable,
                Cancelled,
                Suspended,
            ],
            Validating => &[
                UpdatingMemory,
                ToolRequested,
                FailedRecoverable,
                Completed,
                Cancelled,
                Suspended,
            ],
            UpdatingMemory => &[
                ReadyForNextTurn,
                WaitingForModel,
                Completed,
                FailedRecoverable,
                Cancelled,
                Suspended,
            ],
            ReadyForNextTurn => &[Preparing, Completed, Cancelled, Suspended, NeedsUserInput],
            Completed => &[],
            // A cancelled TURN does not end the session: the chat stays
            // usable (Stop in Kilo cancels the turn, not the session).
            Cancelled => &[Preparing, ReadyForNextTurn, Suspended],
            FailedPermanent => &[],
            FailedRecoverable => &[Preparing, Idle, Cancelled, Suspended, NeedsUserInput],
            NeedsUserInput => &[ReadyForNextTurn, Preparing, Cancelled, Suspended],
            Suspended => &[Idle, Preparing, Cancelled, Completed],
        }
    }

    pub fn is_terminal(self) -> bool {
        // Cancelled is a turn outcome, not a session outcome: the session
        // remains promptable after an abort.
        matches!(self, AgentState::Completed | AgentState::FailedPermanent)
    }

    pub fn is_active(self) -> bool {
        !self.is_terminal() && self != AgentState::Idle && self != AgentState::Suspended
    }

    /// Human label used by the frozen UI's state display.
    pub fn label(self) -> &'static str {
        match self {
            AgentState::Idle => "idle",
            AgentState::Preparing => "preparing",
            AgentState::BuildingContext => "building context",
            AgentState::WaitingForModel => "waiting for model",
            AgentState::Streaming => "streaming",
            AgentState::ToolRequested => "tool requested",
            AgentState::WaitingForPermission => "waiting for permission",
            AgentState::ExecutingTool => "executing tool",
            AgentState::Validating => "validating",
            AgentState::UpdatingMemory => "updating memory",
            AgentState::ReadyForNextTurn => "ready",
            AgentState::Completed => "completed",
            AgentState::Cancelled => "cancelled",
            AgentState::FailedRecoverable => "failed, retrying",
            AgentState::FailedPermanent => "failed",
            AgentState::NeedsUserInput => "needs input",
            AgentState::Suspended => "suspended",
        }
    }
}

/// The session LIFETIME machine — orthogonal to the per-turn `AgentState`
/// machine (spec §6 + review P0-2). A session is Open for days across many
/// turns; only `end_session()` moves it toward Closed. `AgentState` alone
/// cannot express session lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionLifecycle {
    Open,
    Suspended,
    Closing,
    Closed,
    FailedPermanent,
}

impl SessionLifecycle {
    pub fn allowed_transitions(self) -> &'static [SessionLifecycle] {
        use SessionLifecycle::*;
        match self {
            Open => &[Suspended, Closing, Closed, FailedPermanent],
            Suspended => &[Open, Closing, Closed, FailedPermanent],
            Closing => &[Closed, FailedPermanent, Open],
            Closed => &[],
            FailedPermanent => &[],
        }
    }

    pub fn can_accept_prompts(self) -> bool {
        matches!(self, SessionLifecycle::Open)
    }

    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            SessionLifecycle::Closed | SessionLifecycle::FailedPermanent
        )
    }

    pub fn label(self) -> &'static str {
        match self {
            SessionLifecycle::Open => "open",
            SessionLifecycle::Suspended => "suspended",
            SessionLifecycle::Closing => "closing",
            SessionLifecycle::Closed => "closed",
            SessionLifecycle::FailedPermanent => "failed",
        }
    }
}

/// Write ownership of ONE work item of a plan (audits 7/8/21/22: ownership
/// is per work item — never per whole plan). A plan may keep a default/
/// ceiling, but the ACTUAL assignment of write authority is always one of
/// these, recorded per item on the durable work-item→child assignment rows
/// before any child spawns.
///
/// Typed, never prose. `NoWrites` is the read-only default: items of
/// read-only kinds (Analysis/Exploration/Review) may ONLY ever hold it —
/// a write-capable variant on a read-only item is structurally invalid.
/// Mutating items (Implementation/Verification) need one of the three
/// write-capable variants; mutating items whose writes collide (overlapping
/// path sets) are rejected before any spawn.
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OwnershipSpec {
    /// No write authority at all (the only legal ownership of a read-only
    /// work item; a mutating item under this spec is invalid).
    #[default]
    NoWrites,
    /// An explicit disjoint normalized path set inside the owner worktree.
    /// Overlapping path sets of TWO mutating items are a structural
    /// violation — rejected at plan compile, before any child spawn.
    Paths { paths: Vec<String> },
    /// A mutating item on its own isolated worktree/workspace: it can never
    /// collide with any path mutator of the shared owner worktree.
    IsolatedWorktree,
    /// A mutating item whose writes are provider-scoped SEMANTIC ENTITIES
    /// inside one immutable provider snapshot: two semantic mutators
    /// collide exactly when they target the same provider snapshot with
    /// overlapping entity sets. The provider snapshot (not the shared
    /// filesystem) bounds what the child may touch.
    SemanticEntities {
        provider_id: String,
        snapshot_id: String,
        entities: Vec<String>,
    },
}

/// Hard cap on the per-spec path/entity lists (bounded everything; the
/// durable assignment row itself is capped at 4096 bytes on write).
pub const MAX_OWNERSHIP_SPEC_ENTRIES: usize = 64;
/// Hard cap on one path / entity / id string inside an ownership spec.
pub const MAX_OWNERSHIP_SPEC_ENTRY_CHARS: usize = 256;

/// A single offending value of an ownership spec (typed violations).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnershipSpecViolation {
    pub what: &'static str,
    pub value: String,
    pub message: String,
}

impl OwnershipSpec {
    /// Whether this spec grants ANY write authority.
    pub fn allows_writes(&self) -> bool {
        !matches!(self, Self::NoWrites)
    }

    /// The exclusive path set of this spec (`None` when the spec carries
    /// none: read-only, isolated, or semantic-entity ownership).
    pub fn exclusive_paths(&self) -> Option<&[String]> {
        match self {
            Self::Paths { paths } => Some(paths),
            _ => None,
        }
    }

    /// Lexical normalization of one ownership path: trailing `/` is
    /// trimmed; an empty result is a violation (reported by
    /// [`Self::validate`]).
    fn norm_path(p: &str) -> String {
        let mut s = p.trim_end_matches('/');
        while s.ends_with("/.") {
            s = s.trim_end_matches('/').trim_end_matches("/.");
        }
        s.to_string()
    }

    fn path_overlaps(a: &str, b: &str) -> bool {
        a == b
            || (a.starts_with(b) && a.as_bytes().get(b.len()) == Some(&b'/'))
            || (b.starts_with(a) && b.as_bytes().get(a.len()) == Some(&b'/'))
    }

    /// Structural sanity of one spec (bounded entries, sane strings, and —
    /// for a path list — no self-overlapping entries). Every violation is
    /// reported; a spec with no write authority validates trivially.
    pub fn validate(&self) -> Result<(), Vec<OwnershipSpecViolation>> {
        let mut errs = Vec::new();
        fn sane(s: &str) -> bool {
            !s.is_empty()
                && s.chars().count() <= MAX_OWNERSHIP_SPEC_ENTRY_CHARS
                && !s
                    .chars()
                    .any(|c| c.is_control() || c.is_whitespace() || !c.is_ascii())
        }
        /// Provider/snapshot ids are path-component-safe: no `/`, no `\`
        /// (they may be embedded in store keys and directory names).
        fn sane_id(s: &str) -> bool {
            sane(s)
                && !s.contains('/')
                && !s.contains('\\')
                && s.chars().all(|c| c.is_ascii_graphic())
        }
        match self {
            Self::NoWrites => {}
            Self::Paths { paths } => {
                if paths.is_empty() {
                    errs.push(OwnershipSpecViolation {
                        what: "paths",
                        value: String::new(),
                        message: "Paths ownership with no paths".to_string(),
                    });
                }
                if paths.len() > MAX_OWNERSHIP_SPEC_ENTRIES {
                    errs.push(OwnershipSpecViolation {
                        what: "paths",
                        value: String::new(),
                        message: format!(
                            "{MAX_OWNERSHIP_SPEC_ENTRIES} max paths per spec ({} given)",
                            paths.len()
                        ),
                    });
                }
                let norm: Vec<String> = paths.iter().map(|p| Self::norm_path(p)).collect();
                for (i, p) in paths.iter().enumerate() {
                    if norm[i].is_empty() {
                        errs.push(OwnershipSpecViolation {
                            what: "paths",
                            value: p.clone(),
                            message: format!("empty ownership path {p:?}"),
                        });
                    } else if !sane(p) {
                        errs.push(OwnershipSpecViolation {
                            what: "paths",
                            value: p.clone(),
                            message: format!(
                                "ownership path {p:?} is empty, overlong, or not ASCII-printable"
                            ),
                        });
                    }
                }
                for i in 0..norm.len() {
                    for j in (i + 1)..norm.len() {
                        if norm[i].is_empty() || norm[j].is_empty() {
                            continue;
                        }
                        if Self::path_overlaps(&norm[i], &norm[j]) {
                            errs.push(OwnershipSpecViolation {
                                what: "paths",
                                value: p_display(&paths[i], &paths[j]),
                                message: format!(
                                    "ownership paths {p:?} and {q:?} overlap",
                                    p = paths[i],
                                    q = paths[j]
                                ),
                            });
                        }
                    }
                }
            }
            Self::IsolatedWorktree => {}
            Self::SemanticEntities {
                provider_id,
                snapshot_id,
                entities,
            } => {
                if !sane_id(provider_id) {
                    errs.push(OwnershipSpecViolation {
                        what: "provider_id",
                        value: provider_id.clone(),
                        message: format!("provider id {provider_id:?} is empty, overlong, or not ASCII-printable"),
                    });
                }
                if !sane_id(snapshot_id) {
                    errs.push(OwnershipSpecViolation {
                        what: "snapshot_id",
                        value: snapshot_id.clone(),
                        message: format!("snapshot id {snapshot_id:?} is empty, overlong, or not ASCII-printable"),
                    });
                }
                if entities.is_empty() {
                    errs.push(OwnershipSpecViolation {
                        what: "entities",
                        value: String::new(),
                        message: "SemanticEntities ownership with no entities".to_string(),
                    });
                }
                if entities.len() > MAX_OWNERSHIP_SPEC_ENTRIES {
                    errs.push(OwnershipSpecViolation {
                        what: "entities",
                        value: String::new(),
                        message: format!(
                            "{MAX_OWNERSHIP_SPEC_ENTRIES} max entities per spec ({} given)",
                            entities.len()
                        ),
                    });
                }
                let mut seen = std::collections::HashSet::new();
                for e in entities {
                    if !sane(e) {
                        errs.push(OwnershipSpecViolation {
                            what: "entities",
                            value: e.clone(),
                            message: format!(
                                "entity {e:?} is empty, overlong, or not ASCII-printable"
                            ),
                        });
                    }
                    if !seen.insert(e.clone()) {
                        errs.push(OwnershipSpecViolation {
                            what: "entities",
                            value: e.clone(),
                            message: format!("duplicate entity {e:?}"),
                        });
                    }
                }
            }
        }
        if errs.is_empty() {
            Ok(())
        } else {
            Err(errs)
        }
    }

    /// Whether two mutating ownership specs COLLIDE (may never both be the
    /// write authority of two mutating work items of one plan):
    ///
    /// - two path sets overlap on a normalized path boundary;
    /// - two semantic-entity specs over the SAME provider snapshot share an
    ///   entity;
    /// - an isolated worktree never collides with anything; a path mutator
    ///   and a semantic-entity mutator write through different channels and
    ///   never collide statically (the snapshot bounds the semantic side).
    pub fn overlaps(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::NoWrites, _) | (_, Self::NoWrites) => false,
            (Self::IsolatedWorktree, _) | (_, Self::IsolatedWorktree) => false,
            (Self::Paths { paths: a }, Self::Paths { paths: b }) => {
                let na: Vec<String> = a.iter().map(|p| Self::norm_path(p)).collect();
                let nb: Vec<String> = b.iter().map(|p| Self::norm_path(p)).collect();
                na.iter().any(|p| {
                    !p.is_empty()
                        && nb
                            .iter()
                            .any(|q| !q.is_empty() && Self::path_overlaps(p, q))
                })
            }
            (Self::SemanticEntities { .. }, Self::Paths { .. })
            | (Self::Paths { .. }, Self::SemanticEntities { .. }) => false,
            (
                Self::SemanticEntities {
                    provider_id: pa,
                    snapshot_id: sa,
                    entities: ea,
                },
                Self::SemanticEntities {
                    provider_id: pb,
                    snapshot_id: sb,
                    entities: eb,
                },
            ) => pa == pb && sa == sb && ea.iter().any(|e| eb.iter().any(|f| f == e)),
        }
    }
}

fn p_display(a: &str, b: &str) -> String {
    format!("{a} vs {b}")
}

/// The write-capability vocabulary of a work item: one of these ride the
/// durable plan rows (`OwnershipSpec`) — never free-form prose.
#[cfg(test)]
mod ownership_spec_tests {
    use super::*;

    #[test]
    fn no_writes_is_the_default_and_allows_nothing() {
        assert_eq!(OwnershipSpec::default(), OwnershipSpec::NoWrites);
        assert!(!OwnershipSpec::NoWrites.allows_writes());
        assert!(OwnershipSpec::NoWrites.validate().is_ok());
    }

    #[test]
    fn write_capable_variants_validate_and_sane_lists_pass() {
        let paths = OwnershipSpec::Paths {
            paths: vec!["src/a".into(), "src/b".into()],
        };
        assert!(paths.allows_writes());
        assert_eq!(paths.exclusive_paths().unwrap().len(), 2);
        assert!(paths.validate().is_ok());
        assert!(OwnershipSpec::IsolatedWorktree.validate().is_ok());
        let semantic = OwnershipSpec::SemanticEntities {
            provider_id: "fake".into(),
            snapshot_id: "s-1".into(),
            entities: vec!["e1".into(), "e2".into()],
        };
        assert!(semantic.allows_writes());
        assert!(semantic.validate().is_ok());
    }

    #[test]
    fn hostile_specs_fail_loudly() {
        assert!(OwnershipSpec::Paths { paths: vec![] }.validate().is_err());
        assert!(OwnershipSpec::Paths {
            paths: vec!["a b".into()],
        }
        .validate()
        .is_err());
        assert!(OwnershipSpec::Paths {
            paths: vec!["caf\u{e9}".into()],
        }
        .validate()
        .is_err());
        assert!(OwnershipSpec::Paths {
            paths: vec!["a/".into()],
        }
        .validate()
        .is_ok());
        // Self-overlapping path lists are invalid.
        assert!(OwnershipSpec::Paths {
            paths: vec!["src".into(), "src/a".into()],
        }
        .validate()
        .is_err());
        assert!(OwnershipSpec::Paths {
            paths: vec!["src".into(), "src".into()],
        }
        .validate()
        .is_err());
        // Overlong entries and lists are refused.
        let long: String = "a".repeat(MAX_OWNERSHIP_SPEC_ENTRY_CHARS + 1);
        assert!(OwnershipSpec::Paths { paths: vec![long] }
            .validate()
            .is_err());
        let many: Vec<String> = (0..=MAX_OWNERSHIP_SPEC_ENTRIES)
            .map(|i| format!("p{i}"))
            .collect();
        assert!(OwnershipSpec::Paths { paths: many }.validate().is_err());
        // Semantic hostility: empty ids, no entities, duplicates.
        assert!(OwnershipSpec::SemanticEntities {
            provider_id: String::new(),
            snapshot_id: "s".into(),
            entities: vec!["e".into()],
        }
        .validate()
        .is_err());
        assert!(OwnershipSpec::SemanticEntities {
            provider_id: "p".into(),
            snapshot_id: "s/../evil".into(),
            entities: vec!["e".into()],
        }
        .validate()
        .is_err());
        assert!(OwnershipSpec::SemanticEntities {
            provider_id: "p".into(),
            snapshot_id: "s".into(),
            entities: vec![],
        }
        .validate()
        .is_err());
        assert!(OwnershipSpec::SemanticEntities {
            provider_id: "p".into(),
            snapshot_id: "s".into(),
            entities: vec!["e".into(), "e".into()],
        }
        .validate()
        .is_err());
    }

    #[test]
    fn overlap_matrix_is_exact() {
        use OwnershipSpec::*;
        // Paths overlap on normalized boundaries, including duplicates and
        // parent/child spellings.
        assert!(Paths {
            paths: vec!["src".into()],
        }
        .overlaps(&Paths {
            paths: vec!["src/a".into()],
        }));
        assert!(Paths {
            paths: vec!["src/a".into()],
        }
        .overlaps(&Paths {
            paths: vec!["src".into(), "other".into()],
        }));
        assert!(!Paths {
            paths: vec!["src/a".into()],
        }
        .overlaps(&Paths {
            paths: vec!["src/b".into()],
        }));
        assert!(!Paths {
            paths: vec!["src".into()],
        }
        .overlaps(&Paths {
            paths: vec!["src2".into()],
        }));
        // Isolated never collides; NoWrites never collides.
        assert!(!IsolatedWorktree.overlaps(&Paths {
            paths: vec!["src".into()],
        }));
        assert!(!Paths {
            paths: vec!["src".into()],
        }
        .overlaps(&NoWrites));
        // Semantic entities collide only over the same provider snapshot.
        let sem_a = SemanticEntities {
            provider_id: "p".into(),
            snapshot_id: "s".into(),
            entities: vec!["e1".into(), "e2".into()],
        };
        assert!(sem_a.overlaps(&SemanticEntities {
            provider_id: "p".into(),
            snapshot_id: "s".into(),
            entities: vec!["e2".into()],
        }));
        assert!(!sem_a.overlaps(&SemanticEntities {
            provider_id: "p".into(),
            snapshot_id: "s".into(),
            entities: vec!["e9".into()],
        }));
        assert!(!sem_a.overlaps(&SemanticEntities {
            provider_id: "p".into(),
            snapshot_id: "s2".into(),
            entities: vec!["e1".into()],
        }));
        // Path and semantic channels never collide statically.
        assert!(!sem_a.overlaps(&Paths {
            paths: vec!["src".into()],
        }));
    }
}
///
/// This machine tracks the TASK's verification lifecycle, orthogonal to the
/// per-turn `AgentState`: a turn can end `ReadyForNextTurn` while the task is
/// `NeedsVerification`, `Verifying`, `Blocked` or `Failed`.
///
/// **Hard invariant**: only `Verifying -> VerifiedComplete` may produce task
/// success, and that transition is legal only with a PASSING durable
/// verification record (every required check the project type derives ran
/// and passed). No other path ever claims the task completed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum TaskState {
    #[default]
    Pending,
    Planning,
    Running,
    Waiting,
    Blocked,
    NeedsVerification,
    Verifying,
    VerifiedComplete,
    Failed,
    Cancelled,
}

impl TaskState {
    /// The legal one-step edges out of `self` (audit P0-7, machine
    /// semantics). Completion-relevant states are reachable only through the
    /// dedicated transition/complete paths, so `VerifiedComplete` has no
    /// outgoing edge here and no incoming edge either — it is produced
    /// exclusively by `complete_verified_task` against a passing durable
    /// record, never by a generic state assignment.
    pub fn allowed_transitions(self) -> &'static [TaskState] {
        use TaskState::*;
        match self {
            Pending => &[Planning, Running, Cancelled],
            Planning => &[Running, Cancelled],
            Running => &[Waiting, Blocked, NeedsVerification, Failed, Cancelled],
            Waiting => &[Running, Blocked, Cancelled],
            Blocked => &[Running, Failed, Cancelled],
            NeedsVerification => &[Verifying, Cancelled],
            Verifying => &[Failed, NeedsVerification, Cancelled],
            // Terminal: VerifiedComplete is written ONLY by the completion
            // transaction (proof + revision CAS), never by a transition.
            VerifiedComplete => &[],
            Failed => &[],
            Cancelled => &[],
        }
    }

    /// True when the machine allows `from -> to` as one legal step. A
    /// self-transition is legal and idempotent (replay/no-op writes must not
    /// fail), which is why the comparison happens before the edge lookup.
    pub fn transition_legal(from: TaskState, to: TaskState) -> bool {
        from == to || from.allowed_transitions().contains(&to)
    }

    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            TaskState::VerifiedComplete | TaskState::Failed | TaskState::Cancelled
        )
    }

    /// States that certify or claim completion. They may NEVER be assigned
    /// through a generic task patch; the transition API
    /// (`NeedsVerification`/`Verifying`) and the completion transaction
    /// (`VerifiedComplete`) are the only legal producers.
    pub fn is_completion_relevant(self) -> bool {
        matches!(
            self,
            TaskState::NeedsVerification | TaskState::Verifying | TaskState::VerifiedComplete
        )
    }

    /// States `create_task` may seed a fresh row with. Creating a task
    /// already "complete", "verifying" or "verified" would mint completion
    /// proof from nothing, so every other state is rejected at creation.
    pub fn is_creatable(self) -> bool {
        matches!(
            self,
            TaskState::Pending | TaskState::Planning | TaskState::Running
        )
    }

    pub fn label(self) -> &'static str {
        use TaskState::*;
        match self {
            Pending => "pending",
            Planning => "planning",
            Running => "running",
            Waiting => "waiting",
            Blocked => "blocked",
            NeedsVerification => "needs_verification",
            Verifying => "verifying",
            VerifiedComplete => "verified_complete",
            Failed => "failed",
            Cancelled => "cancelled",
        }
    }
}

/// One legal task-state edge of the machine (audit P0-7). The variants are
/// exactly the machine's edges; `VerifiedComplete` is deliberately absent —
/// the ONLY producer of `VerifiedComplete` is
/// `complete_verified_task`, which additionally requires a passing durable
/// verification record for THIS revision, NOT a bare transition.
///
/// `Cancel` encodes "any -> Cancelled", legal from every non-terminal state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskTransition {
    /// Pending -> Planning
    StartPlanning,
    /// Pending -> Running
    StartRunning,
    /// Planning -> Running
    PlanComplete,
    /// Running -> Waiting
    Wait,
    /// Running -> Blocked
    BlockFromRunning,
    /// Running -> NeedsVerification (claim: work finished, proof required)
    RequestVerification,
    /// Running -> Failed
    FailFromRunning,
    /// Waiting -> Running
    ResumeFromWaiting,
    /// Waiting -> Blocked
    BlockFromWaiting,
    /// Blocked -> Running
    Unblock,
    /// Blocked -> Failed
    FailFromBlocked,
    /// NeedsVerification -> Verifying (a verifier picked the claim up)
    StartVerification,
    /// Verifying -> NeedsVerification (verification needs another iteration)
    Reverify,
    /// Verifying -> Failed
    FailFromVerifying,
    /// any non-terminal -> Cancelled
    Cancel,
}

impl TaskTransition {
    /// The state this edge lands on.
    pub fn to_state(self) -> TaskState {
        use TaskState::*;
        use TaskTransition::*;
        match self {
            StartPlanning => Planning,
            StartRunning => Running,
            PlanComplete => Running,
            Wait => Waiting,
            BlockFromRunning | BlockFromWaiting => Blocked,
            RequestVerification | Reverify => NeedsVerification,
            ResumeFromWaiting | Unblock => Running,
            StartVerification => Verifying,
            FailFromRunning | FailFromBlocked | FailFromVerifying => Failed,
            Cancel => Cancelled,
        }
    }

    /// Whether this edge may be taken from `from`. Cancel is legal from
    /// every non-terminal state; every other edge names its one source.
    pub fn legal_from(self, from: TaskState) -> bool {
        use TaskTransition::*;
        match self {
            Cancel => !from.is_terminal(),
            StartPlanning => from == TaskState::Pending,
            StartRunning => from == TaskState::Pending,
            PlanComplete => from == TaskState::Planning,
            Wait | BlockFromRunning | RequestVerification | FailFromRunning => {
                from == TaskState::Running
            }
            ResumeFromWaiting | BlockFromWaiting => from == TaskState::Waiting,
            Unblock | FailFromBlocked => from == TaskState::Blocked,
            StartVerification => from == TaskState::NeedsVerification,
            Reverify | FailFromVerifying => from == TaskState::Verifying,
        }
    }

    /// Every edge of the machine, used by the exhaustive table test.
    pub const ALL: [TaskTransition; 15] = [
        TaskTransition::StartPlanning,
        TaskTransition::StartRunning,
        TaskTransition::PlanComplete,
        TaskTransition::Wait,
        TaskTransition::BlockFromRunning,
        TaskTransition::RequestVerification,
        TaskTransition::FailFromRunning,
        TaskTransition::ResumeFromWaiting,
        TaskTransition::BlockFromWaiting,
        TaskTransition::Unblock,
        TaskTransition::FailFromBlocked,
        TaskTransition::StartVerification,
        TaskTransition::Reverify,
        TaskTransition::FailFromVerifying,
        TaskTransition::Cancel,
    ];
}

/// The durable verification-engine status of the task's last genuine turn
/// end (audits 4/6/7). Distinct from the completion gate: verification may
/// be `Passed` while the completion gate is `Blocked` (skeptical review),
/// and `Unavailable` means no objective mechanism ran at all.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum VerificationStatus {
    #[default]
    Pending,
    Running,
    Passed,
    Failed,
    Unavailable,
}

/// One acceptance-criterion verdict inside a durable verification record
/// (audit P0-8). `criterion_key` is the criterion's identity: the exact text
/// of one entry of the task row's `acceptance_criteria` list (the typed
/// wave-10 rows store criteria as bounded text, so the text IS the key).
/// Completion requires every CURRENT task criterion to be present here with
/// `passed = true`; extra entries (criteria the verification also certified)
/// are allowed.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CriterionVerification {
    pub criterion_key: String,
    pub passed: bool,
    pub evidence: Option<String>,
}

/// Who authored one acceptance criterion (audits 56/57): the origin decides
/// what a re-derivation may do to the criterion. Only [`Self::User`]
/// criteria are sticky — a re-derivation may NEVER remove them; every other
/// origin is derived and is replaced/refreshed by its own derivation.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum CriterionOrigin {
    /// Stated by the user (or the user's goal). Never removed by
    /// re-derivation.
    User,
    /// Derived from the project's policy (project-type detection, the
    /// configured/derived required checks).
    ProjectPolicy,
    /// Derived from the verification policy contract itself.
    VerificationPolicy,
    /// Derived from a semantic provider snapshot: such criteria are tied to
    /// the snapshot id and MUST be re-derived when the source snapshot is
    /// stale.
    SemanticProvider,
}

impl CriterionOrigin {
    /// True only for [`CriterionOrigin::User`] — the sticky origin.
    pub fn is_user(self) -> bool {
        matches!(self, Self::User)
    }

    /// The stable machine label (equals the serde spelling).
    pub fn label(self) -> &'static str {
        match self {
            CriterionOrigin::User => "user",
            CriterionOrigin::ProjectPolicy => "project_policy",
            CriterionOrigin::VerificationPolicy => "verification_policy",
            CriterionOrigin::SemanticProvider => "semantic_provider",
        }
    }
}

impl std::fmt::Display for CriterionOrigin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// How binding one acceptance criterion is (audit 56): a `Required`
/// criterion must be covered by a passing verification; `Preferred` is
/// advisory and never blocks completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CriterionRequirement {
    Required,
    Preferred,
}

impl CriterionRequirement {
    /// True only for [`CriterionRequirement::Required`].
    pub fn is_required(self) -> bool {
        matches!(self, Self::Required)
    }

    /// The stable machine label (equals the serde spelling).
    pub fn label(self) -> &'static str {
        match self {
            CriterionRequirement::Required => "required",
            CriterionRequirement::Preferred => "preferred",
        }
    }
}

impl std::fmt::Display for CriterionRequirement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// Hard cap on one [`ChangeBudget`] list (`allowed_paths` /
/// `allowed_semantic_entities`), mirroring the ownership-spec cap.
pub const MAX_CHANGE_BUDGET_ENTRIES: usize = 64;
/// Hard cap on one [`ChangeBudget`] path / entity string.
pub const MAX_CHANGE_BUDGET_ENTRY_CHARS: usize = 256;

/// The change-scope budget of a task (audit 57 + 105): bounds WHAT one
/// mutating run may change, independent of how much it may cost (that is the
/// token/cost budget). An ABSENT budget (`None` where it is stored) means
/// today's behavior: no change-scope enforcement at all.
///
/// Semantics:
/// - `allowed_paths`: path prefixes. Empty = no path restriction. A changed
///   path is allowed when it equals an entry or lives under it at a path
///   component boundary (`src/` covers `src/a.rs`, never `src2/a.rs`).
/// - `allowed_semantic_entities`: semantic entity ids. Empty = no semantic
///   restriction. When non-empty, a run that reports NO semantic data
///   (Unknown) is refused: the policy requires stronger verification before
///   the change can be certified.
/// - `allow_*`: `false` forbids that class of increase. The class is
///   enforced only when the run REPORTS the observation; an unknown
///   observation is not a violation of this type (see
///   `crate::state::ChangeBudget` consumers for the strict path).
/// - `max_blast_radius`: maximum number of distinct changed paths; `None` =
///   unlimited.
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ChangeBudget {
    pub allowed_paths: Vec<String>,
    pub allowed_semantic_entities: Vec<String>,
    pub allow_public_surface_increase: bool,
    pub allow_security_risk_increase: bool,
    pub allow_unsafe_increase: bool,
    pub allow_new_external_effects: bool,
    pub allow_contract_weakening: bool,
    pub max_blast_radius: Option<u64>,
}

impl ChangeBudget {
    /// Whether one already-normalized changed path is inside the allowed
    /// path set (empty set = unrestricted). Component-boundary prefix match:
    /// `src` covers `src/a.rs` and `src` but never `src2/a.rs`.
    pub fn allows_path(&self, path: &str) -> bool {
        if self.allowed_paths.is_empty() {
            return true;
        }
        let path = norm_budget_path(path);
        self.allowed_paths.iter().any(|allowed| {
            let allowed = norm_budget_path(allowed);
            !allowed.is_empty()
                && (path == allowed
                    || (path.starts_with(&allowed)
                        && path.as_bytes().get(allowed.len()) == Some(&b'/')))
        })
    }

    /// Whether one semantic entity is inside the allowed set (empty set =
    /// unrestricted; an empty entity is never allowed by a non-empty set).
    pub fn allows_semantic_entity(&self, entity: &str) -> bool {
        self.allowed_semantic_entities.is_empty()
            || self.allowed_semantic_entities.iter().any(|e| e == entity)
    }
}

/// Normalize one change-budget path: backslashes to `/`, leading `./`
/// dropped, trailing `/` trimmed. Empty entries are never allowed.
pub fn norm_budget_path(p: &str) -> String {
    let mut s = p.replace('\\', "/");
    while s.starts_with("./") {
        s = s[2..].to_string();
    }
    while s.ends_with('/') {
        s.pop();
    }
    s
}

/// One executed end-of-turn check inside a durable verification record
/// (audit P0-8). Mirrors the runtime's check rows (`verification`/`<id>`
/// facts and typed-ledger `CheckRun`s) as an immutable, bounded,
/// machine-readable row.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckExecution {
    pub check: String,
    pub program: String,
    pub args: Vec<String>,
    pub category: String,
    pub required: bool,
    pub status: VerificationStatus,
    pub started_ms: i64,
    pub finished_ms: Option<i64>,
    pub exit: Option<i32>,
    pub summary: Option<String>,
}

/// One workspace-file state observation inside a durable verification record
/// (audit P0-8): a content-addressed digest of the file as verified, so a
/// record's "these files changed" claim is bound to bytes, not prose.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileStateEvidence {
    pub path: String,
    pub digest_hex: String,
    pub size: u64,
}

/// Machine-readable reason codes for terminal/blocked outcomes (audit 94):
/// every outcome that previously carried ONLY prose now carries
/// `(ReasonCode, detail)` pairs — prose stays for humans, codes exist for
/// machines. Codes are snake_case and unique (the code-table test in this
/// file locks both). Additive by design: new outcomes may add codes, never
/// reuse them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasonCode {
    /// A required check RAN and failed (`FailedVerification`).
    CheckFailed,
    /// A required check could not run: the execution infra delivered no
    /// verdict (`BlockedVerification`).
    CheckUnavailable,
    /// The completion review blocked the gate (verdict `block`, or — in
    /// Strict quality — any non-clean advisory verdict on a mutating turn).
    ReviewBlocked,
    /// A request was denied before it reached the provider because the hard
    /// per-session budget was already exceeded.
    BudgetExceeded,
    /// The durable task row's spend exceeds its caps: VerifiedComplete is
    /// refused at the genuine end.
    SpendOverBudget,
    /// The mutating run left the task's [`ChangeBudget`] (a changed path or
    /// semantic entity outside the allowed set, a forbidden increase, or an
    /// exceeded blast radius): VerifiedComplete is refused.
    ChangeBudgetExceeded,
    /// The turn stopped because no output/progress/op-completion arrived
    /// within the silence budget.
    Stalled,
    /// The turn stopped because repeated identical failing iterations/calls
    /// tripped the loop detector.
    LoopDetected,
    /// The turn/operation was cancelled.
    Cancelled,
    /// Durable acceptance-criteria rows are missing where the gate requires
    /// them.
    CriteriaMissing,
    /// The durable criteria fact (`criteria`/`0`) and the typed task row's
    /// acceptance criteria disagree (crash residue or a hostile write).
    CriteriaInconsistent,
    /// The loop detector stopped the turn because the repo state oscillated
    /// patch → revert → patch (file digests alternating A,B,A,B) while every
    /// other state dimension was unchanged (P0-78).
    PatchRevertPatch,
    /// The loop detector stopped the turn because ≥3 consecutive DIFFERENT
    /// search commands returned the identical evidence set (P0-78).
    RepeatedEvidenceSet,
}

impl ReasonCode {
    /// The complete, ordered code table. The uniqueness test iterates this
    /// array: adding a variant without extending it (or vice versa) fails.
    pub const ALL: [ReasonCode; 13] = [
        ReasonCode::CheckFailed,
        ReasonCode::CheckUnavailable,
        ReasonCode::ReviewBlocked,
        ReasonCode::BudgetExceeded,
        ReasonCode::SpendOverBudget,
        ReasonCode::ChangeBudgetExceeded,
        ReasonCode::Stalled,
        ReasonCode::LoopDetected,
        ReasonCode::Cancelled,
        ReasonCode::CriteriaMissing,
        ReasonCode::CriteriaInconsistent,
        ReasonCode::PatchRevertPatch,
        ReasonCode::RepeatedEvidenceSet,
    ];

    /// The stable machine code (snake_case; equals the serde spelling).
    pub fn code(self) -> &'static str {
        match self {
            ReasonCode::CheckFailed => "check_failed",
            ReasonCode::CheckUnavailable => "check_unavailable",
            ReasonCode::ReviewBlocked => "review_blocked",
            ReasonCode::BudgetExceeded => "budget_exceeded",
            ReasonCode::SpendOverBudget => "spend_over_budget",
            ReasonCode::ChangeBudgetExceeded => "change_budget_exceeded",
            ReasonCode::Stalled => "stalled",
            ReasonCode::LoopDetected => "loop_detected",
            ReasonCode::Cancelled => "cancelled",
            ReasonCode::CriteriaMissing => "criteria_missing",
            ReasonCode::CriteriaInconsistent => "criteria_inconsistent",
            ReasonCode::PatchRevertPatch => "patch_revert_patch",
            ReasonCode::RepeatedEvidenceSet => "repeated_evidence_set",
        }
    }

    /// A short human label for the code.
    pub fn label(self) -> &'static str {
        match self {
            ReasonCode::CheckFailed => "a required check failed",
            ReasonCode::CheckUnavailable => "a required check could not run",
            ReasonCode::ReviewBlocked => "the completion review blocked the change",
            ReasonCode::BudgetExceeded => "the request exceeded the hard budget",
            ReasonCode::SpendOverBudget => "the task spent over its durable budget",
            ReasonCode::ChangeBudgetExceeded => "the mutating run left the task's change budget",
            ReasonCode::Stalled => "the turn stalled (no progress evidence)",
            ReasonCode::LoopDetected => "the loop detector stopped the turn",
            ReasonCode::Cancelled => "the turn was cancelled",
            ReasonCode::CriteriaMissing => "durable acceptance criteria are missing",
            ReasonCode::CriteriaInconsistent => "durable criteria rows disagree",
            ReasonCode::PatchRevertPatch => {
                "the repo state oscillated patch/revert/patch without other change"
            }
            ReasonCode::RepeatedEvidenceSet => {
                "different searches returned the identical evidence set repeatedly"
            }
        }
    }
}

impl std::fmt::Display for ReasonCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.code())
    }
}

impl TryFrom<&str> for ReasonCode {
    type Error = ();
    fn try_from(s: &str) -> Result<Self, ()> {
        for code in ReasonCode::ALL {
            if code.code() == s {
                return Ok(code);
            }
        }
        Err(())
    }
}

/// One machine-readable outcome reason: a stable [`ReasonCode`] plus the
/// human detail prose. Every gate/stall/loop/cancel reason rides this shape
/// (audit 94) so downstream machines can branch on codes and humans still
/// get the prose. Core-internal: no wire protocol serializes this yet (the
/// protocol crates never touch it); unknown codes or extra fields fail
/// loudly at parse time — a hostile payload can never silently become a
/// different reason.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeReason {
    pub code: ReasonCode,
    pub detail: String,
}

impl OutcomeReason {
    pub fn new(code: ReasonCode, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }
}

/// Wraps the state machine and rejects illegal transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateMachine(pub AgentState);

impl StateMachine {
    pub const fn new(initial: AgentState) -> Self {
        Self(initial)
    }

    pub fn state(&self) -> AgentState {
        self.0
    }

    /// Attempt a transition. Returns `Err(InvalidState)` if illegal.
    /// Terminal states cannot transition at all.
    pub fn transition(&mut self, to: AgentState) -> crate::Result<()> {
        if self.0 == to {
            // Self-transitions are allowed and idempotent (re-emitted events
            // during replay must not fail).
            return Ok(());
        }
        if !self.0.allowed_transitions().contains(&to) {
            return Err(Error::new(
                ErrorKind::InvalidState { from: self.0, to },
                format!(
                    "illegal state transition: {} -> {}",
                    self.0.label(),
                    to.label()
                ),
            ));
        }
        self.0 = to;
        Ok(())
    }

    /// Force a state (recovery/replay only, never called from normal flow).
    /// Rejecting force-set during replay would deadlock recovery, but setting
    /// a terminal state that has recorded later events is a corruption sign —
    /// callers must validate against the journal before forcing.
    pub fn force(&mut self, to: AgentState) {
        self.0 = to;
    }
}

// ---------------------------------------------------------------- durable edit transactions

/// Identifies one durable multi-file edit transaction (P0-53). Minted by the
/// durable caller (the session tool-run record) so a post-crash re-execution
/// can resume the SAME transaction instead of re-beginning it. Zero is
/// rejected by contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct EditTxnId(u64);

impl EditTxnId {
    #[inline]
    pub const fn new(raw: u64) -> Self {
        assert!(raw != 0, "EditTxnId cannot be 0");
        Self(raw)
    }

    #[inline]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for EditTxnId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<EditTxnId> for u64 {
    fn from(v: EditTxnId) -> u64 {
        v.0
    }
}

impl serde::Serialize for EditTxnId {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(self.0)
    }
}

impl<'de> serde::Deserialize<'de> for EditTxnId {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let raw = u64::deserialize(d)?;
        if raw == 0 {
            Err(serde::de::Error::custom("EditTxnId cannot be 0"))
        } else {
            Ok(Self(raw))
        }
    }
}

/// Crash-recovery policy of a durable edit transaction (P0-53).
///
/// - `RollForward`: after a crash the unfinished files are committed (CAS
///   replay; already-committed files are detected and never rewritten).
/// - `RollBack`: after a crash the already-committed files of a conflicted
///   transaction are restored to their staged before-content (CAS-restore,
///   never a clobber).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EditTxnStrategy {
    RollForward,
    RollBack,
}

impl EditTxnStrategy {
    /// The durable tag stored in typed ledger rows.
    pub const fn as_tag(self) -> &'static str {
        match self {
            EditTxnStrategy::RollForward => "roll_forward",
            EditTxnStrategy::RollBack => "roll_back",
        }
    }

    /// Parse a durable ledger tag; hostile tags return `None` (never
    /// silently accepted).
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "roll_forward" => Some(EditTxnStrategy::RollForward),
            "roll_back" => Some(EditTxnStrategy::RollBack),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorKind;

    #[test]
    fn happy_chain_is_legal() {
        let mut m = StateMachine::new(AgentState::Idle);
        let chain = [
            AgentState::Preparing,
            AgentState::BuildingContext,
            AgentState::WaitingForModel,
            AgentState::Streaming,
            AgentState::ToolRequested,
            AgentState::WaitingForPermission,
            AgentState::ExecutingTool,
            AgentState::Validating,
            AgentState::UpdatingMemory,
            AgentState::ReadyForNextTurn,
            AgentState::Completed,
        ];
        for s in chain {
            m.transition(s).unwrap();
        }
        assert!(m.state().is_terminal());
    }

    #[test]
    fn illegal_transition_rejected_with_both_ends() {
        let mut m = StateMachine::new(AgentState::Completed);
        let err = m.transition(AgentState::Preparing).unwrap_err();
        match err.kind {
            ErrorKind::InvalidState { from, to } => {
                assert_eq!(from, AgentState::Completed);
                assert_eq!(to, AgentState::Preparing);
            }
            other => panic!("wrong kind {other:?}"),
        }
    }

    #[test]
    fn terminal_states_are_truly_terminal() {
        // Completed and FailedPermanent end the SESSION (only reachable via
        // end_session / permanent failure). Cancelled is a TURN outcome and
        // the session stays usable.
        for t in [AgentState::Completed, AgentState::FailedPermanent] {
            let mut m = StateMachine::new(t);
            for s in [
                AgentState::Idle,
                AgentState::Preparing,
                AgentState::Streaming,
                AgentState::Suspended,
            ] {
                assert!(m.transition(s).is_err(), "{t:?} -> {s:?} must fail");
            }
        }
    }

    #[test]
    fn cancelled_turn_keeps_session_usable() {
        // Stop in Kilo cancels the turn; the chat must accept the next
        // prompt. Cancelled → Preparing is legal; Cancelled → ReadyForNextTurn
        // is legal (abort lands the session ready).
        let mut m = StateMachine::new(AgentState::Cancelled);
        m.transition(AgentState::Preparing).unwrap();
        let mut m = StateMachine::new(AgentState::Cancelled);
        m.transition(AgentState::ReadyForNextTurn).unwrap();
    }

    #[test]
    fn session_lifecycle_machine() {
        use SessionLifecycle::*;
        assert!(Open.can_accept_prompts());
        assert!(!Closed.can_accept_prompts());
        assert!(!Suspended.can_accept_prompts());
        // Legal: open → suspended → open → closing → closed.
        let mut l = Open;
        l = *l
            .allowed_transitions()
            .iter()
            .find(|t| **t == Suspended)
            .unwrap();
        assert_eq!(l, Suspended);
        l = *l
            .allowed_transitions()
            .iter()
            .find(|t| **t == Open)
            .unwrap();
        assert_eq!(l, Open);
        l = *l
            .allowed_transitions()
            .iter()
            .find(|t| **t == Closing)
            .unwrap();
        assert_eq!(l, Closing);
        l = *l
            .allowed_transitions()
            .iter()
            .find(|t| **t == Closed)
            .unwrap();
        assert_eq!(l, Closed);
        assert!(Closed.is_terminal());
        assert!(Closed.allowed_transitions().is_empty());
        assert_eq!(Open.label(), "open");
        assert_eq!(Closed.label(), "closed");
        assert_eq!(FailedPermanent.label(), "failed");
    }

    #[test]
    fn skipping_states_is_rejected() {
        // Streaming -> UpdatingMemory skips Validating: illegal.
        let mut m = StateMachine::new(AgentState::Streaming);
        assert!(m.transition(AgentState::UpdatingMemory).is_err());
        // WaitingForModel -> ExecutingTool skips Streaming: illegal.
        let mut m = StateMachine::new(AgentState::WaitingForModel);
        assert!(m.transition(AgentState::ExecutingTool).is_err());
    }

    #[test]
    fn every_state_has_a_nonempty_label() {
        for s in [
            AgentState::Idle,
            AgentState::Preparing,
            AgentState::BuildingContext,
            AgentState::WaitingForModel,
            AgentState::Streaming,
            AgentState::ToolRequested,
            AgentState::WaitingForPermission,
            AgentState::ExecutingTool,
            AgentState::Validating,
            AgentState::UpdatingMemory,
            AgentState::ReadyForNextTurn,
            AgentState::Completed,
            AgentState::Cancelled,
            AgentState::FailedRecoverable,
            AgentState::FailedPermanent,
            AgentState::NeedsUserInput,
            AgentState::Suspended,
        ] {
            assert!(!s.label().is_empty());
        }
    }

    #[test]
    fn self_transition_is_idempotent_for_replay() {
        let mut m = StateMachine::new(AgentState::Streaming);
        m.transition(AgentState::Streaming).unwrap();
        assert_eq!(m.state(), AgentState::Streaming);
    }

    #[test]
    fn exhaustive_transition_matrix_is_defined_for_all_states() {
        // Every state must declare an allowed set (even if empty) — no
        // unhandled states can silently enter the machine.
        for s in [
            AgentState::Idle,
            AgentState::Preparing,
            AgentState::BuildingContext,
            AgentState::WaitingForModel,
            AgentState::Streaming,
            AgentState::ToolRequested,
            AgentState::WaitingForPermission,
            AgentState::ExecutingTool,
            AgentState::Validating,
            AgentState::UpdatingMemory,
            AgentState::ReadyForNextTurn,
            AgentState::Completed,
            AgentState::Cancelled,
            AgentState::FailedRecoverable,
            AgentState::FailedPermanent,
            AgentState::NeedsUserInput,
            AgentState::Suspended,
        ] {
            let _ = s.allowed_transitions();
            let _ = s.is_terminal();
            let _ = s.is_active();
        }
    }

    #[test]
    fn reason_code_table_codes_are_unique_snake_case_and_stable() {
        // Audit 94 code table: every code string is unique (a machine
        // branching on codes must never see two meanings), snake_case, and
        // the table equals the serde spelling (the wire/hostile-payload
        // path and the machine path can never drift apart).
        let mut seen: std::collections::HashSet<&'static str> = std::collections::HashSet::new();
        for code in ReasonCode::ALL {
            let s = code.code();
            assert!(
                seen.insert(s),
                "reason code {s:?} is duplicated in the table"
            );
            assert!(
                s.chars().all(|c| c.is_ascii_lowercase() || c == '_')
                    && s.chars().next().is_some_and(|c| c.is_ascii_lowercase()),
                "code {s:?} must be snake_case"
            );
            assert_ne!(code.code(), code.label(), "label must not equal code");
            // The serde spelling must agree with the machine code string.
            let json = serde_json::to_string(&code).unwrap();
            assert_eq!(json, format!("\"{s}\""), "serde spelling drifted");
            // TryFrom round trip: the canonical string resolves back.
            assert_eq!(ReasonCode::try_from(s), Ok(code));
        }
        // Every variant of the enum is in the table: an enum variant added
        // without a table row breaks ALL (serde deserializes it but no
        // machine code exists). Exhaustive via a manual listing — adding a
        // variant here without a row above fails the next match arm.
        assert_eq!(ReasonCode::ALL.len(), 13);
    }

    #[test]
    fn hostile_reason_payloads_fail_loudly_at_parse() {
        // Unknown codes and extra/missing fields must ERROR — a hostile or
        // corrupted payload can never silently decode into a different
        // reason than the one that was durably recorded.
        assert!(
            serde_json::from_str::<ReasonCode>("\"not_a_code\"").is_err(),
            "an unknown code must fail loudly"
        );
        assert!(
            serde_json::from_str::<ReasonCode>("\"review_blocked\"").is_ok(),
            "a canonical code must parse"
        );
        let reason = serde_json::json!({
            "code": "check_failed",
            "detail": "required check 'cargo check' failed",
        });
        let parsed: OutcomeReason = serde_json::from_value(reason.clone()).unwrap();
        assert_eq!(parsed.code, ReasonCode::CheckFailed);
        assert_eq!(parsed.detail, "required check 'cargo check' failed");
        let mut hostile = reason.clone();
        hostile["detail"] = serde_json::Value::Null;
        assert!(
            serde_json::from_value::<OutcomeReason>(hostile).is_err(),
            "a missing detail must fail loudly"
        );
        let mut hostile = reason.clone();
        hostile["extra"] = serde_json::json!(true);
        assert!(
            serde_json::from_value::<OutcomeReason>(hostile).is_err(),
            "deny_unknown_fields: an extra field must fail loudly"
        );
        let mut hostile = reason;
        hostile["code"] = serde_json::json!("mystery");
        assert!(
            serde_json::from_value::<OutcomeReason>(hostile).is_err(),
            "an unknown code inside a reason must fail loudly"
        );
    }

    #[test]
    fn task_state_machine_edges_are_exactly_the_transition_table() {
        // Every allowed_transitions edge must be encodable as a
        // TaskTransition, and every TaskTransition must be a legal
        // allowed_transitions edge — the two tables can never drift.
        use TaskState::*;
        let all = [
            Pending,
            Planning,
            Running,
            Waiting,
            Blocked,
            NeedsVerification,
            Verifying,
            VerifiedComplete,
            Failed,
            Cancelled,
        ];
        for from in all {
            for to in from.allowed_transitions() {
                assert!(
                    TaskTransition::ALL
                        .iter()
                        .any(|t| t.legal_from(from) && t.to_state() == *to),
                    "machine edge {from:?} -> {to:?} has no TaskTransition"
                );
            }
        }
        for t in TaskTransition::ALL {
            for from in all {
                if t.legal_from(from) {
                    assert!(
                        from.allowed_transitions().contains(&t.to_state()),
                        "TaskTransition {t:?} from {from:?} -> {:?} is not a machine edge",
                        t.to_state()
                    );
                }
            }
        }
    }

    #[test]
    fn verified_complete_is_unreachable_by_transition_and_patch() {
        // P0-7: the ONLY producer of VerifiedComplete is the completion
        // transaction with a passing record — no machine edge and no legal
        // patch assignment may reach it.
        use TaskState::*;
        for from in [
            Pending,
            Planning,
            Running,
            Waiting,
            Blocked,
            NeedsVerification,
            Verifying,
        ] {
            assert!(
                !from.allowed_transitions().contains(&VerifiedComplete),
                "{from:?} must not lead to VerifiedComplete"
            );
            assert!(!TaskTransition::ALL
                .iter()
                .any(|t| t.legal_from(from) && t.to_state() == VerifiedComplete));
        }
        assert!(VerifiedComplete.is_terminal());
        assert!(VerifiedComplete.is_completion_relevant());
        assert!(!VerifiedComplete.is_creatable());
        // Generic patch legality: self-transition idempotent; illegal jump
        // rejected even when both ends are ordinary states.
        assert!(TaskState::transition_legal(Running, Running));
        assert!(!TaskState::transition_legal(Pending, Failed));
        assert!(!TaskState::transition_legal(VerifiedComplete, Running));
        // Cancelling is legal from every non-terminal state and from none of
        // the terminals.
        for from in all_ten() {
            assert_eq!(
                TaskTransition::Cancel.legal_from(from),
                !from.is_terminal(),
                "Cancel from {from:?}"
            );
        }
    }

    #[test]
    fn task_state_creation_gate_and_completion_relevance_are_stable() {
        // create_task admits exactly {Pending, Planning, Running}; the
        // completion-relevant set is exactly the three states a generic
        // patch must never assign.
        for s in all_ten() {
            match s {
                TaskState::Pending | TaskState::Planning | TaskState::Running => {
                    assert!(s.is_creatable(), "{s:?} must be creatable")
                }
                other => assert!(!other.is_creatable(), "{other:?} must not be creatable"),
            }
            match s {
                TaskState::NeedsVerification
                | TaskState::Verifying
                | TaskState::VerifiedComplete => {
                    assert!(
                        s.is_completion_relevant(),
                        "{s:?} must be completion-relevant"
                    )
                }
                other => assert!(
                    !other.is_completion_relevant(),
                    "{other:?} must be patchable"
                ),
            }
            // Terminal means no further mutation at all.
            match s {
                TaskState::VerifiedComplete | TaskState::Failed | TaskState::Cancelled => {
                    assert!(s.is_terminal(), "{s:?} must be terminal")
                }
                other => assert!(!other.is_terminal(), "{other:?} must not be terminal"),
            }
            assert!(!s.label().is_empty());
        }
        // The serde spellings are the durable labels (they are persisted in
        // the task row's state column).
        for s in all_ten() {
            let json = serde_json::to_string(&s).unwrap();
            assert_eq!(json.trim_matches('"'), s.label());
        }
    }

    #[test]
    fn evidence_types_parse_strictly_and_roundtrip() {
        let criterion = CriterionVerification {
            criterion_key: "cargo check passes".into(),
            passed: true,
            evidence: Some("exit 0".into()),
        };
        let check = CheckExecution {
            check: "compile".into(),
            program: "cargo".into(),
            args: vec!["check".into()],
            category: "required".into(),
            required: true,
            status: VerificationStatus::Passed,
            started_ms: 1,
            finished_ms: Some(2),
            exit: Some(0),
            summary: Some("ok".into()),
        };
        let file = FileStateEvidence {
            path: "src/main.rs".into(),
            digest_hex: "ab".repeat(32),
            size: 42,
        };
        for v in [
            serde_json::to_value(&criterion).unwrap(),
            serde_json::to_value(&check).unwrap(),
            serde_json::to_value(&file).unwrap(),
        ] {
            let s = v.to_string();
            let parsed: serde_json::Value = serde_json::from_str(&s).unwrap();
            assert_eq!(parsed, v, "serde roundtrip must be byte-stable");
        }
        // Hostile payloads fail loudly: unknown fields and nulls are
        // rejected, never silently defaulted.
        let mut hostile = serde_json::to_value(&criterion).unwrap();
        hostile["extra"] = serde_json::json!(true);
        assert!(serde_json::from_value::<CriterionVerification>(hostile).is_err());
        let hostile = serde_json::json!({"criterion_key": null, "passed": false});
        assert!(serde_json::from_value::<CriterionVerification>(hostile).is_err());
        // Evidence status roundtrip agrees with the durable spelling.
        assert_eq!(
            serde_json::to_string(&check.status)
                .unwrap()
                .trim_matches('"'),
            "passed"
        );
    }

    fn all_ten() -> [TaskState; 10] {
        use TaskState::*;
        [
            Pending,
            Planning,
            Running,
            Waiting,
            Blocked,
            NeedsVerification,
            Verifying,
            VerifiedComplete,
            Failed,
            Cancelled,
        ]
    }

    #[test]
    fn criterion_origin_and_requirement_are_typed_and_hostile_safe() {
        // The vocabulary is stable snake_case; the sticky origin is exactly
        // User (re-derivation may never drop it) and only Required binds.
        for origin in [
            CriterionOrigin::User,
            CriterionOrigin::ProjectPolicy,
            CriterionOrigin::VerificationPolicy,
            CriterionOrigin::SemanticProvider,
        ] {
            let json = serde_json::to_string(&origin).unwrap();
            assert_eq!(json, format!("\"{}\"", origin.label()));
            assert_eq!(
                serde_json::from_str::<CriterionOrigin>(&json).unwrap(),
                origin
            );
        }
        assert!(CriterionOrigin::User.is_user());
        assert!(!CriterionOrigin::ProjectPolicy.is_user());
        assert!(CriterionRequirement::Required.is_required());
        assert!(!CriterionRequirement::Preferred.is_required());
        // Hostile variants fail loudly, never default to a guessed origin.
        assert!(serde_json::from_str::<CriterionOrigin>("\"provider\"").is_err());
        assert!(serde_json::from_str::<CriterionOrigin>("\"USER\"").is_err());
        assert!(serde_json::from_str::<CriterionRequirement>("\"must\"").is_err());
    }

    #[test]
    fn change_budget_paths_are_component_bounded_and_entries_are_bounded() {
        let empty = ChangeBudget::default();
        assert!(
            empty.allows_path("anything/at/all.rs"),
            "empty = unrestricted"
        );
        assert!(empty.allows_semantic_entity("e1"), "empty = unrestricted");
        assert_eq!(empty.max_blast_radius, None);
        let budget = ChangeBudget {
            allowed_paths: vec!["src".into(), "docs/".into(), "./tests/x.rs".into()],
            allowed_semantic_entities: vec!["entity-a".into()],
            allow_public_surface_increase: false,
            allow_security_risk_increase: false,
            allow_unsafe_increase: false,
            allow_new_external_effects: false,
            allow_contract_weakening: false,
            max_blast_radius: Some(2),
        };
        assert!(budget.allows_path("src/a.rs"), "component prefix");
        assert!(budget.allows_path("src"), "exact entry");
        assert!(budget.allows_path("docs/readme.md"), "normalized prefix");
        assert!(budget.allows_path("tests/x.rs"), "leading ./ normalized");
        assert!(
            !budget.allows_path("src2/a.rs"),
            "no partial component match"
        );
        assert!(!budget.allows_path("other/a.rs"));
        assert!(!budget.allows_path(""), "empty path never inside a set");
        assert!(budget.allows_semantic_entity("entity-a"));
        assert!(!budget.allows_semantic_entity("entity-b"));
        assert!(
            !budget.allows_semantic_entity(""),
            "empty entity is not a member"
        );
        // Hostile unknown fields fail loudly (no silent widening).
        let hostile = serde_json::json!({"allowed_paths": ["src"], "bogus": true});
        assert!(serde_json::from_value::<ChangeBudget>(hostile).is_err());
        // Missing fields default (forward/backward compatible).
        let partial: ChangeBudget = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(partial, ChangeBudget::default());
    }

    #[test]
    fn change_budget_reason_code_is_additive_and_stable() {
        assert_eq!(
            ReasonCode::ChangeBudgetExceeded.code(),
            "change_budget_exceeded"
        );
        assert_eq!(
            ReasonCode::try_from("change_budget_exceeded"),
            Ok(ReasonCode::ChangeBudgetExceeded)
        );
        assert_ne!(
            ReasonCode::ChangeBudgetExceeded.code(),
            ReasonCode::SpendOverBudget.code(),
            "the change-scope refusal is distinct from the spend refusal"
        );
    }
}
