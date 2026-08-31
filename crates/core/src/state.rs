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
            ReadyForNextTurn => &[
                Preparing,
                Completed,
                Cancelled,
                Suspended,
                NeedsUserInput,
            ],
            Completed => &[],
            Cancelled => &[],
            FailedPermanent => &[],
            FailedRecoverable => &[
                Preparing,
                Idle,
                Cancelled,
                Suspended,
                NeedsUserInput,
            ],
            NeedsUserInput => &[ReadyForNextTurn, Preparing, Cancelled, Suspended],
            Suspended => &[Idle, Preparing, Cancelled, Completed],
        }
    }

    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            AgentState::Completed | AgentState::Cancelled | AgentState::FailedPermanent
        )
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
                ErrorKind::InvalidState {
                    from: self.0,
                    to,
                },
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
        for t in [
            AgentState::Completed,
            AgentState::Cancelled,
            AgentState::FailedPermanent,
        ] {
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
}
