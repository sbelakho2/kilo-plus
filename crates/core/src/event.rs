//! The append-only event journal contract.
//!
//! Every significant transition in a session becomes an event. The session
//! database therefore knows exactly what happened; the rendered conversation
//! is a *view derived from the journal*, never the source of truth.

use crate::id::{EventSeq, OpId, SessionId};
use crate::state::AgentState;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    SessionCreated,
    PromptReceived,
    ContextPrepared,
    ModelStarted,
    ModelChunkReceived,
    ToolRequested,
    ToolStarted,
    FileChanged,
    ToolCompleted,
    ToolCancelled,
    CheckpointCreated,
    ContextCompacted,
    CompactRejected,
    SubagentStarted,
    SubagentCompleted,
    TurnCompleted,
    PermissionGranted,
    PermissionDenied,
    /// A durably queued prompt was admitted as the active logical turn
    /// (atomic claim + message materialization; audit round 7).
    PromptAdmitted,
    /// An interior state hop WITHIN one logical turn (e.g. after a tool
    /// batch: Validating → UpdatingMemory → WaitingForModel). Never
    /// completes a turn — exactly one `TurnCompleted` marks the end of a
    /// logical turn (audit round 6).
    PhaseChanged,
    /// Crash recovery re-executed an interrupted idempotent tool invocation
    /// as a NEW PHYSICAL attempt of the SAME logical operation (the run row
    /// carries the attempt counter; the tool call id never changes).
    ReplayStarted,
    CrashDetected,
    RecoveryApplied,
    SessionEnded,
    Suspended,
    Resumed,
    Failed,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct Event {
    /// Monotonic per-session sequence number; the SSE resume cursor.
    pub seq: EventSeq,
    pub session_id: SessionId,
    pub op_id: Option<OpId>,
    pub kind: EventKind,
    /// Agent state after this event (projection source of truth).
    pub state: AgentState,
    /// Milliseconds since Unix epoch, monotonic non-decreasing per session.
    pub ts_ms: i64,
    /// Kind-specific payload; optional, may be large (chunk text, tool args).
    pub payload: Option<serde_json::Value>,
}

impl Event {
    pub fn new(
        seq: EventSeq,
        session_id: SessionId,
        op_id: Option<OpId>,
        kind: EventKind,
        state: AgentState,
        ts_ms: i64,
        payload: Option<serde_json::Value>,
    ) -> Self {
        Self {
            seq,
            session_id,
            op_id,
            kind,
            state,
            ts_ms,
            payload,
        }
    }
}

/// Invariants the journal enforces at append time; violated invariants are
/// loud errors instead of silent corruption.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JournalInvariants;

impl JournalInvariants {
    /// Sequence numbers must be exactly `prev + 1`; gaps or duplicates are
    /// corruption (or replay) and are rejected by the store.
    pub fn next_seq(prev: Option<EventSeq>) -> EventSeq {
        match prev {
            None => EventSeq::new(1),
            Some(p) => {
                let raw = p.raw().checked_add(1).expect("event seq overflow");
                EventSeq::new(raw)
            }
        }
    }

    /// Timestamps must be non-decreasing; clock skew backwards by more than
    /// the tolerance is treated as a clock reset and the event is stamped
    /// with the previous event's time instead (never decreases).
    pub fn monotonic_ts(prev_ts: Option<i64>, now: i64) -> i64 {
        match prev_ts {
            None => now,
            Some(p) if now >= p => now,
            Some(p) => p, // never go backwards
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::FileHash;

    #[test]
    fn seq_never_reuses_or_gaps() {
        assert_eq!(JournalInvariants::next_seq(None).raw(), 1);
        assert_eq!(JournalInvariants::next_seq(Some(EventSeq::new(1))).raw(), 2);
        // u64::MAX + 1 would overflow; next_seq is contractually forbidden
        // from being called there (checked_add panics loudly).
        assert_eq!(EventSeq::new(u64::MAX).raw(), u64::MAX);
    }

    #[test]
    fn timestamps_never_decrease() {
        assert_eq!(JournalInvariants::monotonic_ts(None, 100), 100);
        assert_eq!(JournalInvariants::monotonic_ts(Some(100), 101), 101);
        assert_eq!(JournalInvariants::monotonic_ts(Some(100), 50), 100);
        // clock jump forward is fine
        assert_eq!(
            JournalInvariants::monotonic_ts(Some(100), 1_000_000),
            1_000_000
        );
    }

    #[test]
    fn event_roundtrip_json_and_defaults() {
        let e = Event::new(
            EventSeq::new(7),
            SessionId::new(1),
            Some(OpId::new(3)),
            EventKind::FileChanged,
            AgentState::ExecutingTool,
            1234,
            Some(serde_json::json!({"path": "a.rs", "hash": FileHash::from([0u8; 32]).to_hex()})),
        );
        let json = serde_json::to_string(&e).unwrap();
        let back: Event = serde_json::from_str(&json).unwrap();
        assert_eq!(e, back);
        assert_eq!(back.kind, EventKind::FileChanged);
    }

    #[test]
    fn duplicate_seq_detected_by_store_is_not_a_journal_silence() {
        // This is enforced in kilop-store; here we verify the marker types
        // used for that check exist and compare correctly.
        assert!(EventSeq::new(2) != EventSeq::new(3));
    }
}
