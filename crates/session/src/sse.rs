//! SSE resume: journal events projected onto the frozen SSE surface, using
//! the protocol crate's frame contract. The cursor is the event sequence.

use faktor_core::id::EventSeq;
use faktor_protocol::sse::SseEvent;

use crate::handle::SessionHandle;

/// One projected journal event ready for framing.
#[derive(Debug, Clone, PartialEq)]
pub struct JournalFrame {
    /// SSE `id:` cursor (journal sequence).
    pub seq: EventSeq,
    pub event: SseEvent,
}

impl SessionHandle {
    /// All journal events strictly after `after` that have an SSE projection
    /// (interior chunk events without message context are skipped by the
    /// projection — the state endpoints carry that information).
    pub fn journal_events_after(&self, after: EventSeq) -> faktor_core::Result<Vec<JournalFrame>> {
        let events = self.events_after(after)?;
        let mut frames = Vec::with_capacity(events.len());
        for e in events {
            if let Some((event, _kind)) = faktor_protocol::sse::project_event(&e) {
                frames.push(JournalFrame { seq: e.seq, event });
            }
        }
        Ok(frames)
    }

    /// The complete journal projected to SSE from the beginning.
    pub fn journal_events(&self) -> faktor_core::Result<Vec<JournalFrame>> {
        let events = self.events_range(1, None)?;
        let mut frames = Vec::with_capacity(events.len());
        for e in events {
            if let Some((event, _kind)) = faktor_protocol::sse::project_event(&e) {
                frames.push(JournalFrame { seq: e.seq, event });
            }
        }
        Ok(frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::tests::{session, test_manager};
    use faktor_core::event::EventKind;
    use faktor_core::state::AgentState;

    #[test]
    fn sse_resume_cursor_resumes_after_given_seq() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let r = s.submit_prompt("hello", &[]).unwrap();
        // Resume after seq 1: only the PromptReceived event (seq 2) remains,
        // projected to agent_state_changed. SessionCreated is behind us.
        let frames = s.journal_events_after(EventSeq::new(1)).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].seq.raw(), 2);
        assert_eq!(frames[0].event.event_type(), "agent_state_changed");
        // Resume from the prompt event: nothing new.
        assert!(s.journal_events_after(r.event_seq).unwrap().is_empty());
        // A later event resumes fine.
        s.append_event(
            EventKind::ContextPrepared,
            AgentState::BuildingContext,
            None,
            None,
        )
        .unwrap();
        let frames = s.journal_events_after(r.event_seq).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].seq.raw(), 3);
        // The frame encodes the event id the protocol parser reads back.
        let frame = frames[0].event.to_frame(frames[0].seq.raw());
        let (id, _ev) = SseEvent::from_frame(&frame).unwrap();
        assert_eq!(id, 3);
    }

    #[test]
    fn full_journal_projection_is_bounded_to_the_session() {
        let (_d, m) = test_manager();
        let s = session(&m);
        s.submit_prompt("x", &[]).unwrap();
        let frames = s.journal_events().unwrap();
        // Every frame carries this session's id and an ascending cursor.
        let mut last = 0u64;
        for f in &frames {
            assert!(f.seq.raw() > last);
            last = f.seq.raw();
            if let SseEvent::SessionUpdated { session_id, .. } = &f.event {
                assert_eq!(session_id, &s.id().to_string());
            }
        }
        assert!(frames.len() >= 2);
    }
}
