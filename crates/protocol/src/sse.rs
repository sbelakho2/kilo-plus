//! SSE framing for the v7.5.6 event stream. Each event is a standard SSE
//! frame with an explicit `event:` type, an `id:` cursor (event sequence),
//! and a JSON `data:` payload. Frames are the frozen byte contract.

use std::fmt::Write as _;

use kilop_core::event::{Event, EventKind};
use kilop_core::state::AgentState;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum SseEvent {
    SessionUpdated {
        session_id: String,
        state: String,
    },
    MessageCreated {
        session_id: String,
        message: crate::v756::Message,
    },
    MessagePartUpdated {
        session_id: String,
        message_id: String,
        part: crate::v756::Part,
    },
    ToolCallState {
        session_id: String,
        tool_call_id: String,
        state: String,
    },
    PermissionRequested {
        permission_id: String,
        session_id: String,
        capability: String,
        detail: serde_json::Value,
    },
    AgentStateChanged {
        session_id: String,
        state: String,
        label: String,
    },
    AgentManagerUpdate {
        update: serde_json::Value,
    },
    Compaction {
        session_id: String,
        before_tokens: i64,
        after_tokens: i64,
        accepted: bool,
    },
    Error {
        session_id: Option<String>,
        code: String,
        message: String,
    },
}

impl SseEvent {
    /// The SSE `event:` type name (frozen).
    pub fn event_type(&self) -> &'static str {
        match self {
            SseEvent::SessionUpdated { .. } => "session_updated",
            SseEvent::MessageCreated { .. } => "message_created",
            SseEvent::MessagePartUpdated { .. } => "message_part_updated",
            SseEvent::ToolCallState { .. } => "tool_call_state",
            SseEvent::PermissionRequested { .. } => "permission_requested",
            SseEvent::AgentStateChanged { .. } => "agent_state_changed",
            SseEvent::AgentManagerUpdate { .. } => "agent_manager_update",
            SseEvent::Compaction { .. } => "compaction",
            SseEvent::Error { .. } => "error",
        }
    }

    /// Encode one frame. `seq` becomes the SSE `id:` (resume cursor).
    pub fn to_frame(&self, seq: u64) -> String {
        let data = serde_json::to_string(self).expect("sse event serializes");
        let mut frame = String::with_capacity(data.len() + 64);
        let _ = write!(
            frame,
            "event: {}\nid: {}\ndata: {}\n\n",
            self.event_type(),
            seq,
            data
        );
        frame
    }

    /// Parse one frame (including the trailing blank line).
    pub fn from_frame(frame: &str) -> Option<(u64, SseEvent)> {
        let mut event = None;
        let mut id = None;
        let mut data = String::new();
        for line in frame.lines() {
            if let Some(v) = line.strip_prefix("event:") {
                event = Some(v.trim());
            } else if let Some(v) = line.strip_prefix("id:") {
                id = Some(v.trim().parse::<u64>().ok()?);
            } else if let Some(v) = line.strip_prefix("data:") {
                data.push_str(v.trim());
            }
        }
        let ev = event?;
        let id = id?;
        let sse: SseEvent = serde_json::from_str(&data).ok()?;
        if sse.event_type() != ev {
            return None;
        }
        Some((id, sse))
    }
}

/// Maps a journal event to the SSE projection the frozen UI expects.
/// The rendered conversation is a view derived from the journal.
pub fn project_event(e: &Event) -> Option<(SseEvent, EventKind)> {
    let sid = e.session_id.to_string();
    let payload = e.payload.as_ref();
    match e.kind {
        EventKind::SessionCreated => Some((
            SseEvent::SessionUpdated {
                session_id: sid,
                state: serde_json::to_string(&e.state).unwrap(),
            },
            e.kind,
        )),
        EventKind::PromptReceived => Some((
            SseEvent::AgentStateChanged {
                session_id: sid,
                state: e.state.label().into(),
                label: e.state.label().into(),
            },
            e.kind,
        )),
        EventKind::PromptAdmitted => Some((
            SseEvent::AgentStateChanged {
                session_id: sid,
                state: e.state.label().into(),
                label: e.state.label().into(),
            },
            e.kind,
        )),
        EventKind::ContextPrepared | EventKind::ModelStarted => Some((
            SseEvent::AgentStateChanged {
                session_id: sid,
                state: e.state.label().into(),
                label: e.state.label().into(),
            },
            e.kind,
        )),
        EventKind::ModelChunkReceived => payload.and_then(|p| {
            p.get("message_id")
                .and_then(|m| m.as_str())
                .and_then(|message_id| {
                    p.get("part")
                        .cloned()
                        .and_then(|part| serde_json::from_value(part).ok())
                        .map(|part| {
                            (
                                SseEvent::MessagePartUpdated {
                                    session_id: sid.clone(),
                                    message_id: message_id.to_string(),
                                    part,
                                },
                                e.kind,
                            )
                        })
                })
        }),
        EventKind::ToolRequested => payload.map(|p| {
            (
                SseEvent::PermissionRequested {
                    permission_id: p
                        .get("permission_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    session_id: sid.clone(),
                    capability: p
                        .get("capability")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    detail: p.clone(),
                },
                e.kind,
            )
        }),
        EventKind::PhaseChanged | EventKind::PermissionGranted | EventKind::PermissionDenied => {
            Some((
                SseEvent::AgentStateChanged {
                    session_id: sid,
                    state: e.state.label().into(),
                    label: e.state.label().into(),
                },
                e.kind,
            ))
        }
        EventKind::ContextCompacted | EventKind::CompactRejected => payload.map(|p| {
            (
                SseEvent::Compaction {
                    session_id: sid.clone(),
                    before_tokens: p.get("before").and_then(|v| v.as_i64()).unwrap_or(0),
                    after_tokens: p.get("after").and_then(|v| v.as_i64()).unwrap_or(0),
                    accepted: p.get("accepted").and_then(|v| v.as_bool()).unwrap_or(false),
                },
                e.kind,
            )
        }),
        EventKind::Failed => Some((
            SseEvent::Error {
                session_id: Some(sid),
                code: "agent_failed".into(),
                message: payload
                    .and_then(|p| p.get("message"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("agent failed")
                    .to_string(),
            },
            e.kind,
        )),
        // Interior events without a dedicated projection; the UI learns the
        // new state through the session state endpoint instead.
        EventKind::ToolStarted
        | EventKind::ToolCompleted
        | EventKind::ToolCancelled
        | EventKind::FileChanged
        | EventKind::CheckpointCreated
        | EventKind::SubagentStarted
        | EventKind::SubagentCompleted
        | EventKind::TurnCompleted
        | EventKind::CrashDetected
        | EventKind::RecoveryApplied
        | EventKind::SessionEnded
        | EventKind::Suspended
        | EventKind::Resumed => Some((
            SseEvent::AgentStateChanged {
                session_id: sid,
                state: e.state.label().into(),
                label: e.state.label().into(),
            },
            e.kind,
        )),
    }
}

/// An agent state change the UI can render without full re-fetch.
pub fn state_event(session_id: &str, state: AgentState) -> SseEvent {
    SseEvent::AgentStateChanged {
        session_id: session_id.to_string(),
        state: state.label().to_string(),
        label: state.label().to_string(),
    }
}

impl crate::v756::GlobalEvent {
    /// Encode one global-event frame. The `event:` field is optional in the
    /// real stream — the payload's `type` field carries the discriminator —
    /// so frames carry only `id:` (resume cursor) and `data:`.
    pub fn to_frame(&self, seq: u64) -> String {
        let data = serde_json::to_string(self).expect("global event serializes");
        let mut frame = String::with_capacity(data.len() + 24);
        let _ = write!(frame, "id: {seq}\ndata: {data}\n\n");
        frame
    }

    /// Parse one global-event frame (including the trailing blank line).
    /// Strict: every line must be `id:` or `data:`, both must appear, and
    /// the data must parse with the frozen `deny_unknown_fields` envelope.
    pub fn from_frame(frame: &str) -> Option<(u64, crate::v756::GlobalEvent)> {
        let mut id = None;
        let mut data = String::new();
        for line in frame.lines() {
            if let Some(v) = line.strip_prefix("id:") {
                if id.is_some() {
                    return None;
                }
                id = Some(v.trim().parse::<u64>().ok()?);
            } else if let Some(v) = line.strip_prefix("data:") {
                data.push_str(v.trim());
            } else if !line.trim().is_empty() {
                // Unknown/event lines are rejected: the frame shape is frozen.
                return None;
            }
        }
        let id = id?;
        let ge: crate::v756::GlobalEvent = serde_json::from_str(&data).ok()?;
        Some((id, ge))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_roundtrip_is_byte_stable() {
        let ev = SseEvent::AgentStateChanged {
            session_id: "s1".into(),
            state: "streaming".into(),
            label: "streaming".into(),
        };
        let frame = ev.to_frame(7);
        assert_eq!(
            frame,
            "event: agent_state_changed\nid: 7\ndata: {\"event\":\"agent_state_changed\",\"session_id\":\"s1\",\"state\":\"streaming\",\"label\":\"streaming\"}\n\n"
        );
        let (id, back) = SseEvent::from_frame(&frame).unwrap();
        assert_eq!(id, 7);
        assert_eq!(back, ev);
    }

    #[test]
    fn frame_with_mismatched_event_type_rejected() {
        let ev = SseEvent::SessionUpdated {
            session_id: "s".into(),
            state: "idle".into(),
        };
        let mut frame = ev.to_frame(1);
        // Tamper with the event type field.
        frame = frame.replace("event: session_updated", "event: error");
        assert!(SseEvent::from_frame(&frame).is_none());
    }

    #[test]
    fn truncated_and_garbage_frames_rejected() {
        assert!(SseEvent::from_frame("").is_none());
        assert!(SseEvent::from_frame("event: error").is_none());
        assert!(SseEvent::from_frame("data: {").is_none());
        assert!(SseEvent::from_frame("id: nope\ndata: {}").is_none());
        assert!(SseEvent::from_frame(
            "event: error\nid: 1\ndata: {\"event\":\"agent_state_changed\"}"
        )
        .is_none());
    }

    #[test]
    fn project_event_covers_every_journal_kind() {
        // Every EventKind must produce a deterministic projection (Some or
        // None with a sensible payload), never panic.
        let kinds = [
            EventKind::SessionCreated,
            EventKind::PromptReceived,
            EventKind::ContextPrepared,
            EventKind::ModelStarted,
            EventKind::ModelChunkReceived,
            EventKind::ToolRequested,
            EventKind::ToolStarted,
            EventKind::FileChanged,
            EventKind::ToolCompleted,
            EventKind::ToolCancelled,
            EventKind::CheckpointCreated,
            EventKind::ContextCompacted,
            EventKind::CompactRejected,
            EventKind::SubagentStarted,
            EventKind::SubagentCompleted,
            EventKind::TurnCompleted,
            EventKind::PermissionGranted,
            EventKind::PermissionDenied,
            EventKind::CrashDetected,
            EventKind::RecoveryApplied,
            EventKind::SessionEnded,
            EventKind::Suspended,
            EventKind::Resumed,
            EventKind::Failed,
        ];
        for kind in kinds {
            let e = Event::new(
                kilop_core::id::EventSeq::new(1),
                kilop_core::id::SessionId::new(1),
                None,
                kind,
                AgentState::Idle,
                0,
                Some(serde_json::json!({"message_id": "m1", "part": {"type":"text","text":"x"}})),
            );
            let _ = project_event(&e);
        }
    }

    #[test]
    fn adversarial_oversized_data_field_roundtrips() {
        let big = "x".repeat(1 << 20);
        let ev = SseEvent::MessagePartUpdated {
            session_id: "s".into(),
            message_id: "m".into(),
            part: crate::v756::Part::Text { text: big },
        };
        let frame = ev.to_frame(1);
        let (_, back) = SseEvent::from_frame(&frame).unwrap();
        match back {
            SseEvent::MessagePartUpdated { part, .. } => match part {
                crate::v756::Part::Text { text } => assert_eq!(text.len(), 1 << 20),
                _ => panic!("wrong part"),
            },
            _ => panic!("wrong event"),
        }
    }

    #[test]
    fn newline_injection_in_fields_cannot_break_framing() {
        // A hostile text containing \n\n must not allow frame injection:
        // JSON escaping keeps \n inside the data payload.
        let evil = "data: fake\n\n" /* injection attempt */;
        let ev = SseEvent::MessagePartUpdated {
            session_id: "s".into(),
            message_id: "m".into(),
            part: crate::v756::Part::Text {
                text: format!("legit {evil}"),
            },
        };
        let frame = ev.to_frame(1);
        // The frame must contain exactly one blank-line terminator at the end.
        let frames: Vec<&str> = frame.split("\n\n").collect();
        assert_eq!(frames.len(), 2, "injection would create extra frame breaks");
        // And parsing must restore the original text.
        let (_, back) = SseEvent::from_frame(&frame).unwrap();
        match back {
            SseEvent::MessagePartUpdated { part, .. } => match part {
                crate::v756::Part::Text { text } => assert_eq!(text, format!("legit {evil}")),
                _ => panic!("wrong part"),
            },
            _ => panic!("wrong event"),
        }
    }

    #[test]
    fn global_event_frame_roundtrip_is_byte_stable() {
        use crate::v756::{GlobalEvent, GlobalEventPayload};
        let ge = GlobalEvent {
            directory: Some("/w".into()),
            project: None,
            workspace: None,
            payload: GlobalEventPayload::SessionCreated {
                session_id: "s1".into(),
            },
        };
        let frame = ge.to_frame(7);
        assert_eq!(
            frame,
            "id: 7\ndata: {\"directory\":\"/w\",\"project\":null,\"workspace\":null,\"payload\":{\"type\":\"session_created\",\"session_id\":\"s1\"}}\n\n"
        );
        let (id, back) = GlobalEvent::from_frame(&frame).unwrap();
        assert_eq!(id, 7);
        assert_eq!(back, ge);
    }

    #[test]
    fn global_event_malformed_and_tampered_frames_rejected() {
        use crate::v756::GlobalEvent;
        // Missing id, missing data, truncated JSON, garbage lines, duplicate
        // id fields: all rejected loudly.
        assert!(GlobalEvent::from_frame("").is_none());
        assert!(GlobalEvent::from_frame("data: {}").is_none());
        assert!(GlobalEvent::from_frame("id: 1").is_none());
        assert!(GlobalEvent::from_frame("id: 1\ndata: {").is_none());
        assert!(GlobalEvent::from_frame("id: nope\ndata: {}").is_none());
        assert!(GlobalEvent::from_frame("event: session_created\nid: 1\ndata: {}").is_none());
        assert!(GlobalEvent::from_frame("id: 1\nid: 2\ndata: {}").is_none());
        // Unknown envelope fields (deny_unknown_fields) are rejected.
        let evil = "id: 1\ndata: {\"directory\":null,\"project\":null,\"workspace\":null,\"payload\":{\"type\":\"session_created\",\"session_id\":\"s\"},\"smuggled\":1}";
        assert!(GlobalEvent::from_frame(evil).is_none());
        // Unknown payload type rejected.
        let evil = "id: 1\ndata: {\"directory\":null,\"project\":null,\"workspace\":null,\"payload\":{\"type\":\"hax\",\"session_id\":\"s\"}}";
        assert!(GlobalEvent::from_frame(evil).is_none());
    }

    #[test]
    fn global_event_frame_newline_injection_is_contained() {
        use crate::v756::{GlobalEvent, GlobalEventPayload};
        let evil = "id: 99\n\n";
        let ge = GlobalEvent {
            directory: None,
            project: None,
            workspace: None,
            payload: GlobalEventPayload::SessionNextTextDelta {
                session_id: "s".into(),
                delta: format!("legit {evil}"),
            },
        };
        let frame = ge.to_frame(3);
        assert_eq!(
            frame.split("\n\n").count(),
            2,
            "injection creates frame breaks"
        );
        let (id, back) = GlobalEvent::from_frame(&frame).unwrap();
        assert_eq!(id, 3);
        match back.payload {
            GlobalEventPayload::SessionNextTextDelta { delta, .. } => {
                assert_eq!(delta, format!("legit {evil}"));
            }
            other => panic!("wrong payload {other:?}"),
        }
    }

    #[test]
    fn global_event_frame_oversized_data_roundtrips() {
        use crate::v756::{GlobalEvent, GlobalEventPayload};
        let big = "x".repeat(1 << 20);
        let ge = GlobalEvent {
            directory: None,
            project: None,
            workspace: None,
            payload: GlobalEventPayload::SessionNextTextDelta {
                session_id: "s".into(),
                delta: big,
            },
        };
        let frame = ge.to_frame(1);
        let (_, back) = GlobalEvent::from_frame(&frame).unwrap();
        match back.payload {
            GlobalEventPayload::SessionNextTextDelta { delta, .. } => {
                assert_eq!(delta.len(), 1 << 20)
            }
            other => panic!("wrong payload {other:?}"),
        }
    }
}
