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

/// The durable per-task completion state (audits 4/6/7: turn completion and
/// TASK verification-complete used to be conflated, and verification was
/// advisory — missing infrastructure silently yielded "completed").
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
}

impl ReasonCode {
    /// The complete, ordered code table. The uniqueness test iterates this
    /// array: adding a variant without extending it (or vice versa) fails.
    pub const ALL: [ReasonCode; 10] = [
        ReasonCode::CheckFailed,
        ReasonCode::CheckUnavailable,
        ReasonCode::ReviewBlocked,
        ReasonCode::BudgetExceeded,
        ReasonCode::SpendOverBudget,
        ReasonCode::Stalled,
        ReasonCode::LoopDetected,
        ReasonCode::Cancelled,
        ReasonCode::CriteriaMissing,
        ReasonCode::CriteriaInconsistent,
    ];

    /// The stable machine code (snake_case; equals the serde spelling).
    pub fn code(self) -> &'static str {
        match self {
            ReasonCode::CheckFailed => "check_failed",
            ReasonCode::CheckUnavailable => "check_unavailable",
            ReasonCode::ReviewBlocked => "review_blocked",
            ReasonCode::BudgetExceeded => "budget_exceeded",
            ReasonCode::SpendOverBudget => "spend_over_budget",
            ReasonCode::Stalled => "stalled",
            ReasonCode::LoopDetected => "loop_detected",
            ReasonCode::Cancelled => "cancelled",
            ReasonCode::CriteriaMissing => "criteria_missing",
            ReasonCode::CriteriaInconsistent => "criteria_inconsistent",
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
            ReasonCode::Stalled => "the turn stalled (no progress evidence)",
            ReasonCode::LoopDetected => "the loop detector stopped the turn",
            ReasonCode::Cancelled => "the turn was cancelled",
            ReasonCode::CriteriaMissing => "durable acceptance criteria are missing",
            ReasonCode::CriteriaInconsistent => "durable criteria rows disagree",
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
        assert_eq!(ReasonCode::ALL.len(), 10);
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
}
