//! Journaling rules: every event's state must be a legal `StateMachine`
//! transition from the previous state, and replay must detect corruption.

use kilop_core::event::EventKind;
use kilop_core::state::{AgentState, StateMachine};

use crate::SessionError;

/// Validate that an event of `kind` may land on `to` from `current`.
///
/// Two kinds carry documented sub-chains because the *event kind* describes a
/// step that the *state column* records as its result:
///
/// - `ToolRequested` events are recorded with state `WaitingForPermission`
///   (the machine hops `ToolRequested` then `WaitingForPermission`; both hops
///   must be legal).
/// - `Failed` events recorded with state `FailedPermanent` are legal only via
///   the documented two-step `FailedRecoverable` then force — no state's
///   `allowed_transitions` lists `FailedPermanent` by design, so entering it
///   is a deliberate, recorded escalation.
///
/// Self-transitions are legal and idempotent (replay must not fail on
/// re-emitted events).
pub(crate) fn validate_transition(
    current: AgentState,
    kind: EventKind,
    to: AgentState,
) -> Result<(), SessionError> {
    let mut m = StateMachine::new(current);
    let hop = |m: &mut StateMachine, target: AgentState| -> Result<(), SessionError> {
        m.transition(target).map_err(|_| SessionError::illegal(current, to))
    };
    match kind {
        EventKind::ToolRequested => {
            if to != AgentState::WaitingForPermission {
                return Err(SessionError::Malformed(
                    "ToolRequested events must record state WaitingForPermission".into(),
                ));
            }
            hop(&mut m, AgentState::ToolRequested)?;
            hop(&mut m, AgentState::WaitingForPermission)?;
            Ok(())
        }
        EventKind::Failed if to == AgentState::FailedPermanent => {
            // Documented two-step: FailedRecoverable must be reachable first.
            hop(&mut m, AgentState::FailedRecoverable)?;
            Ok(())
        }
        _ => hop(&mut m, to),
    }
}

/// The result of replaying a session journal from durable state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayOutcome {
    /// Reconstructed machine state after the final event.
    pub state: AgentState,
    /// Sequence of the final event.
    pub last_seq: kilop_core::id::EventSeq,
    /// Number of events replayed.
    pub event_count: u64,
}

/// Replay a session's journal, enforcing the same transition rules the live
/// append path uses. Any violation is journal corruption and a loud error —
/// never a silent skip.
pub(crate) fn replay(
    events: &[kilop_core::event::Event],
) -> Result<ReplayOutcome, SessionError> {
    let first = events.first().ok_or_else(|| {
        SessionError::Internal("session has no events; journal is missing SessionCreated".into())
    })?;
    if first.kind != EventKind::SessionCreated {
        return Err(SessionError::Internal(format!(
            "journal corruption: first event is {:?}, not SessionCreated",
            first.kind
        )));
    }
    let mut m = StateMachine::new(first.state);
    for e in &events[1..] {
        validate_transition(m.state(), e.kind, e.state).map_err(|err| {
            SessionError::Internal(format!(
                "journal corruption at seq {}: {}",
                e.seq, err
            ))
        })?;
        m.force(e.state);
    }
    let last = events.last().expect("non-empty by construction");
    Ok(ReplayOutcome {
        state: m.state(),
        last_seq: last.seq,
        event_count: events.len() as u64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kilop_core::event::Event;
    use kilop_core::id::{EventSeq, OpId, SessionId};

    fn ev(seq: u64, kind: EventKind, state: AgentState) -> Event {
        Event::new(
            EventSeq::new(seq),
            SessionId::new(1),
            Some(OpId::new(1)),
            kind,
            state,
            0,
            None,
        )
    }

    #[test]
    fn replay_rejects_skipped_states() {
        // Preparing -> Streaming skips BuildingContext/WaitingForModel.
        let events = vec![
            ev(1, EventKind::SessionCreated, AgentState::Idle),
            ev(2, EventKind::PromptReceived, AgentState::Preparing),
            ev(3, EventKind::ModelStarted, AgentState::Streaming),
        ];
        assert!(replay(&events).is_err(), "skipped states are corruption");
    }

    #[test]
    fn replay_rejects_transition_from_terminal() {
        let events = vec![
            ev(1, EventKind::SessionCreated, AgentState::Idle),
            ev(2, EventKind::TurnCompleted, AgentState::Completed),
            ev(3, EventKind::PromptReceived, AgentState::Preparing),
        ];
        assert!(replay(&events).is_err(), "terminal must be terminal");
    }

    #[test]
    fn replay_accepts_the_full_legal_chain() {
        let events = vec![
            ev(1, EventKind::SessionCreated, AgentState::Idle),
            ev(2, EventKind::PromptReceived, AgentState::Preparing),
            ev(3, EventKind::ContextPrepared, AgentState::BuildingContext),
            ev(4, EventKind::ModelStarted, AgentState::WaitingForModel),
            ev(5, EventKind::ModelChunkReceived, AgentState::Streaming),
            ev(6, EventKind::ToolRequested, AgentState::WaitingForPermission),
            ev(7, EventKind::ToolStarted, AgentState::ExecutingTool),
            ev(8, EventKind::ToolCompleted, AgentState::Validating),
            ev(9, EventKind::TurnCompleted, AgentState::Completed),
        ];
        let out = replay(&events).unwrap();
        assert_eq!(out.state, AgentState::Completed);
        assert_eq!(out.event_count, 9);
    }

    #[test]
    fn replay_accepts_documented_failed_permanent_escalation() {
        let events = vec![
            ev(1, EventKind::SessionCreated, AgentState::Idle),
            ev(2, EventKind::PromptReceived, AgentState::Preparing),
            ev(3, EventKind::Failed, AgentState::FailedRecoverable),
            ev(4, EventKind::Failed, AgentState::FailedPermanent),
        ];
        let out = replay(&events).unwrap();
        assert_eq!(out.state, AgentState::FailedPermanent);
    }

    #[test]
    fn validation_rejects_tool_requested_with_wrong_state() {
        let err = validate_transition(
            AgentState::Streaming,
            EventKind::ToolRequested,
            AgentState::ToolRequested,
        )
        .unwrap_err();
        assert!(matches!(err, SessionError::Malformed(_)));
    }
}
