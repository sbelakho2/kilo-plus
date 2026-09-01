//! The conversation view: messages and parts, mapped to the frozen
//! `kilop-protocol::v756` shapes. Paging never loads more than one page.

use kilop_protocol::v756::{
    Message as WireMessage, MessagesPage, Part as WirePart, ToolResultBody,
};
use kilop_store::{MessageRow, PartRow};

use crate::handle::SessionHandle;
use crate::{json_bytes, SessionError, MAX_MESSAGE_BYTES, MAX_PAGE_SIZE, MAX_PART_BYTES};

/// The frozen set of part kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PartKind {
    Text,
    Reasoning,
    ToolCall,
    ToolResult,
    Summary,
}

impl PartKind {
    pub const ALL: [PartKind; 5] = [
        PartKind::Text,
        PartKind::Reasoning,
        PartKind::ToolCall,
        PartKind::ToolResult,
        PartKind::Summary,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            PartKind::Text => "text",
            PartKind::Reasoning => "reasoning",
            PartKind::ToolCall => "tool_call",
            PartKind::ToolResult => "tool_result",
            PartKind::Summary => "summary",
        }
    }

    pub fn from_name(s: &str) -> Option<PartKind> {
        match s {
            "text" => Some(PartKind::Text),
            "reasoning" => Some(PartKind::Reasoning),
            "tool_call" => Some(PartKind::ToolCall),
            "tool_result" => Some(PartKind::ToolResult),
            "summary" => Some(PartKind::Summary),
            _ => None,
        }
    }
}

/// Validate a part payload against its kind's documented shape and bound it.
/// Rejects malformed shapes *before* any write.
pub(crate) fn validate_part(kind: PartKind, data: &serde_json::Value) -> Result<(), SessionError> {
    if json_bytes(data) > MAX_PART_BYTES {
        return Err(SessionError::Oversized(format!(
            "part payload of {} bytes exceeds MAX_PART_BYTES",
            json_bytes(data)
        )));
    }
    let get_str = |key: &str| -> Result<String, SessionError> {
        data.get(key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| {
                SessionError::Malformed(format!(
                    "part of kind {:?} is missing string field `{key}`",
                    kind
                ))
            })
    };
    match kind {
        PartKind::Text | PartKind::Reasoning | PartKind::Summary => {
            get_str("text")?;
        }
        PartKind::ToolCall => {
            get_str("tool_call_id")?;
            get_str("name")?;
            get_str("state")?;
            if !data.get("input").is_some() {
                return Err(SessionError::Malformed(
                    "tool_call part is missing `input`".into(),
                ));
            }
        }
        PartKind::ToolResult => {
            get_str("tool_call_id")?;
            get_str("excerpt")?;
            if let Some(code) = data.get("exit_code") {
                if !code.is_null() && !code.is_i64() {
                    return Err(SessionError::Malformed(
                        "tool_result `exit_code` must be an integer or null".into(),
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Map a durable part row to the frozen wire shape. Unknown kinds are loud
/// errors (corruption), never silently dropped.
pub(crate) fn wire_part(row: &PartRow) -> Result<WirePart, SessionError> {
    let kind = PartKind::from_name(&row.kind)
        .ok_or_else(|| SessionError::Malformed(format!("unknown part kind {:?}", row.kind)))?;
    let s = |key: &str| -> Result<String, SessionError> {
        row.data
            .get(key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| {
                SessionError::Malformed(format!(
                    "part row {} kind {:?} is missing `{key}`",
                    row.id, row.kind
                ))
            })
    };
    Ok(match kind {
        PartKind::Text => WirePart::Text { text: s("text")? },
        PartKind::Reasoning => WirePart::Reasoning { text: s("text")? },
        PartKind::Summary => WirePart::Summary { text: s("text")? },
        PartKind::ToolCall => WirePart::ToolCall {
            tool_call_id: s("tool_call_id")?,
            name: s("name")?,
            input: row
                .data
                .get("input")
                .cloned()
                .unwrap_or(serde_json::Value::Null),
            state: s("state")?,
        },
        PartKind::ToolResult => WirePart::ToolResult {
            tool_call_id: s("tool_call_id")?,
            result: ToolResultBody {
                excerpt: s("excerpt")?,
                exit_code: row
                    .data
                    .get("exit_code")
                    .and_then(|v| if v.is_null() { None } else { v.as_i64() })
                    .map(|v| v as i32),
                artifact: row
                    .data
                    .get("artifact")
                    .and_then(|v| v.as_str())
                    .map(String::from),
                slice_hint: row
                    .data
                    .get("slice_hint")
                    .and_then(|v| v.as_str())
                    .map(String::from),
            },
        },
    })
}

impl SessionHandle {
    /// Put a message with an explicit sequence (the journal event seq is the
    /// natural choice). Duplicate `(session, seq)` is a `Conflict`. Role must
    /// be one of `user`, `assistant`, `system`. Returns the durable row id.
    pub fn put_message(
        &self,
        seq: i64,
        role: &str,
        data: serde_json::Value,
    ) -> kilop_core::Result<i64> {
        if !matches!(role, "user" | "assistant" | "system") {
            return Err(SessionError::Malformed(format!("invalid role {role:?}")).into());
        }
        if seq <= 0 {
            return Err(
                SessionError::Malformed(format!("message seq must be > 0, got {seq}")).into(),
            );
        }
        if json_bytes(&data) > MAX_MESSAGE_BYTES {
            return Err(SessionError::Oversized(format!(
                "message payload of {} bytes exceeds MAX_MESSAGE_BYTES",
                json_bytes(&data)
            ))
            .into());
        }
        Ok(self
            .manager
            .store()
            .put_message(self.id, seq, role, data)
            .map_err(crate::map_store_err)?)
    }

    /// The next free message sequence: one past the newest stored message
    /// (or 1 when empty). The journal is the master order; message rows align
    /// with it, but message creation must not depend on journal growth.
    pub fn proposed_message_seq(&self) -> kilop_core::Result<i64> {
        let newest = self
            .manager
            .store()
            .messages_before(self.id, None, 1)
            .map_err(crate::map_store_err)?;
        Ok(newest.first().map(|r| r.seq + 1).unwrap_or(1))
    }

    /// Put one part on an existing message. Payload shape is validated against
    /// [`PartKind`] before any write; unknown kinds are rejected.
    pub fn put_part(
        &self,
        message_id: i64,
        kind: PartKind,
        data: serde_json::Value,
    ) -> kilop_core::Result<i64> {
        validate_part(kind, &data)?;
        Ok(self
            .manager
            .store()
            .put_part(message_id, kind.as_str(), data)
            .map_err(crate::map_store_err)?)
    }

    pub fn parts_of(&self, message_id: i64) -> kilop_core::Result<Vec<PartRow>> {
        self.manager
            .store()
            .parts_of(message_id)
            .map_err(|e| crate::map_store_err(e).into())
    }

    /// Convenience: a `text` part.
    pub fn put_text_part(&self, message_id: i64, text: &str) -> kilop_core::Result<i64> {
        self.put_part(
            message_id,
            PartKind::Text,
            serde_json::json!({ "text": text }),
        )
    }

    /// Convenience: a `tool_result` part from a frozen wire body.
    pub fn put_tool_result_part(
        &self,
        message_id: i64,
        tool_call_id: &str,
        body: &ToolResultBody,
    ) -> kilop_core::Result<i64> {
        self.put_part(
            message_id,
            PartKind::ToolResult,
            serde_json::json!({
                "tool_call_id": tool_call_id,
                "excerpt": body.excerpt,
                "exit_code": body.exit_code,
                "artifact": body.artifact,
                "slice_hint": body.slice_hint,
            }),
        )
    }

    pub fn message_count(&self) -> kilop_core::Result<i64> {
        self.manager
            .store()
            .message_count(self.id)
            .map_err(|e| crate::map_store_err(e).into())
    }

    /// Newest-first messages with `seq < before` (None = latest page). Never
    /// loads more than one page. The limit is clamped into `[1, MAX_PAGE_SIZE]`.
    pub fn messages_before(
        &self,
        before: Option<i64>,
        limit: i64,
    ) -> kilop_core::Result<Vec<MessageRow>> {
        let limit = limit.clamp(1, MAX_PAGE_SIZE);
        self.manager
            .store()
            .messages_before(self.id, before, limit as u64)
            .map_err(|e| crate::map_store_err(e).into())
    }

    /// The frozen protocol page for the webview: metadata + one page, with the
    /// cursor for older pages. Parts are loaded per message *in the page only*.
    pub fn messages_page(
        &self,
        before: Option<i64>,
        limit: i64,
    ) -> kilop_core::Result<MessagesPage> {
        let limit = limit.clamp(1, MAX_PAGE_SIZE);
        let mut rows = self
            .manager
            .store()
            .messages_before(self.id, before, limit as u64 + 1)
            .map_err(crate::map_store_err)?;
        let has_more = rows.len() as i64 > limit;
        if has_more {
            rows.truncate(limit as usize);
        }
        let next_before = if has_more {
            rows.last().map(|r| r.seq)
        } else {
            None
        };

        let mut messages = Vec::with_capacity(rows.len());
        for row in rows {
            let part_rows = self
                .manager
                .store()
                .parts_of(row.id)
                .map_err(crate::map_store_err)?;
            let mut parts = Vec::with_capacity(part_rows.len());
            for p in &part_rows {
                parts.push(wire_part(p)?);
            }
            messages.push(WireMessage {
                id: row.id.to_string(),
                role: row.role.clone(),
                session_id: row.session_id.to_string(),
                seq: row.seq,
                created_ms: row.created_ms,
                parts,
            });
        }
        Ok(MessagesPage {
            session_id: self.id.to_string(),
            messages,
            has_more,
            next_before,
        })
    }

    /// Latest message page (the webview's initial load).
    pub fn latest_messages_page(&self, limit: i64) -> kilop_core::Result<MessagesPage> {
        self.messages_page(None, limit)
    }

    /// The frozen `SessionState` projection the UI polls on reconnect.
    pub fn session_state_view(&self) -> kilop_core::Result<kilop_protocol::v756::SessionState> {
        let row = self.row()?;
        let last_event_seq = self.last_event_seq()?.map(|s| s.raw() as i64).unwrap_or(0);
        let ledger = self.get_task_ledger()?;
        Ok(kilop_protocol::v756::SessionState {
            session_id: self.id.to_string(),
            state: crate::state_tag(row.state),
            title: row.title,
            last_event_seq,
            agent_state: kilop_protocol::v756::AgentStateView {
                state: crate::state_tag(row.state),
                label: row.state.label().to_string(),
                active: row.state.is_active(),
                terminal: row.state.is_terminal(),
            },
            task_ledger: ledger,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::tests::{session, test_manager};

    fn seeded_session() -> (tempfile::TempDir, SessionHandle) {
        let (_d, m) = test_manager();
        let s = session(&m);
        s.submit_prompt("first", &[]).unwrap();
        // The user prompt message occupies seq 2; the assistant reply goes next.
        s.put_message(
            s.proposed_message_seq().unwrap(),
            "assistant",
            serde_json::json!({"text": "ok"}),
        )
        .unwrap();
        (_d, s)
    }

    #[test]
    fn message_paging_single_page_bounds_and_full_walk() {
        let (_d, s) = seeded_session();
        let mut mids = Vec::new();
        for i in 0..250 {
            let seq = s.proposed_message_seq().unwrap();
            let mid = s
                .put_message(seq, "assistant", serde_json::json!({"i": i}))
                .unwrap();
            mids.push(mid);
        }
        for mid in mids {
            s.put_part(mid, PartKind::Text, serde_json::json!({"text": "p"}))
                .unwrap();
        }
        let page = s.messages_page(None, 10).unwrap();
        assert_eq!(page.messages.len(), 10);
        assert!(page.has_more);
        let next = page.next_before.unwrap();
        assert!(next < page.messages[0].seq, "cursor moves backwards");
        // Every page is bounded by the requested limit; the full walk reaches
        // every message exactly once (1 user + 1 assistant + 250 = 252).
        let mut seen = Vec::new();
        let mut cursor = None;
        loop {
            let p = s.messages_page(cursor, 7).unwrap();
            assert!(p.messages.len() <= 7, "never exceed one page");
            if p.messages.is_empty() {
                break;
            }
            for m in &p.messages {
                assert!(!seen.contains(&m.seq), "duplicate message in walk");
                seen.push(m.seq);
            }
            // next_before == None is the end-of-history marker: the walk must
            // stop there, never rewind to the newest page.
            match p.next_before {
                Some(b) => cursor = Some(b),
                None => break,
            }
        }
        assert_eq!(seen.len(), 252);
    }

    #[test]
    fn paging_limit_clamped_and_latest_page() {
        let (_d, s) = seeded_session();
        let page = s.messages_page(None, -5).unwrap();
        assert_eq!(page.messages.len(), 1, "negative limit clamps to 1");
        let page = s.messages_page(None, i64::MAX).unwrap();
        assert!(
            page.messages.len() <= MAX_PAGE_SIZE as usize,
            "limit clamps at MAX_PAGE_SIZE"
        );
        let latest = s.latest_messages_page(1).unwrap();
        assert!(latest.has_more);
        // before=0 yields nothing, not an error.
        assert!(s.messages_page(Some(0), 10).unwrap().messages.is_empty());
    }

    #[test]
    fn duplicate_message_seq_conflicts() {
        let (_d, s) = seeded_session();
        s.put_message(99, "assistant", serde_json::json!({"t": "a"}))
            .unwrap();
        let err = s
            .put_message(99, "assistant", serde_json::json!({"t": "b"}))
            .unwrap_err();
        assert_eq!(
            err.kind,
            kilop_core::ErrorKind::Conflict,
            "unique (session, seq)"
        );
    }

    #[test]
    fn malformed_role_and_seq_rejected() {
        let (_d, s) = seeded_session();
        assert!(s.put_message(1, "root", serde_json::json!({})).is_err());
        assert!(s.put_message(0, "user", serde_json::json!({})).is_err());
        assert!(s.put_message(-1, "user", serde_json::json!({})).is_err());
    }

    #[test]
    fn oversized_message_rejected_before_write() {
        let (_d, s) = seeded_session();
        let big = serde_json::json!({ "blob": "x".repeat(MAX_MESSAGE_BYTES + 1) });
        assert!(s.put_message(7, "user", big).is_err());
        assert_eq!(
            s.message_count().unwrap(),
            2,
            "no trace of the rejected write"
        );
    }

    #[test]
    fn part_kind_validation_rejects_malformed_shapes() {
        let (_d, s) = seeded_session();
        let mid = s
            .put_message(50, "assistant", serde_json::json!({"text": "m"}))
            .unwrap();
        assert!(s
            .put_part(mid, PartKind::Text, serde_json::json!({"missing": true}))
            .is_err());
        assert!(s
            .put_part(
                mid,
                PartKind::ToolCall,
                serde_json::json!({"tool_call_id": "c"})
            )
            .is_err());
        assert!(s
            .put_part(
                mid,
                PartKind::ToolResult,
                serde_json::json!({"tool_call_id": "c", "excerpt": "x", "exit_code": "not-an-int"})
            )
            .is_err());
        // Rejected parts leave no rows.
        assert!(s.parts_of(mid).unwrap().is_empty());
        // A valid shape round-trips onto the frozen wire type.
        s.put_part(
            mid,
            PartKind::ToolCall,
            serde_json::json!({
                "tool_call_id": "c1",
                "name": "read_file",
                "input": {"path": "a.rs"},
                "state": "pending"
            }),
        )
        .unwrap();
        let parts = s.parts_of(mid).unwrap();
        match wire_part(&parts[0]).unwrap() {
            WirePart::ToolCall { name, input, .. } => {
                assert_eq!(name, "read_file");
                assert_eq!(input["path"], "a.rs");
            }
            other => panic!("wrong part {other:?}"),
        }
    }

    #[test]
    fn tool_result_body_roundtrip_and_unknown_kind_row_is_loud() {
        let (_d, s) = seeded_session();
        let body = ToolResultBody {
            excerpt: "1 | fn main".into(),
            exit_code: Some(0),
            artifact: Some("artifact://abc".into()),
            slice_hint: Some("artifact://abc?slice=0".into()),
        };
        let mid = s
            .put_message(60, "assistant", serde_json::json!({"text": "r"}))
            .unwrap();
        s.put_tool_result_part(mid, "c1", &body).unwrap();
        let rows = s.parts_of(mid).unwrap();
        match wire_part(&rows[0]).unwrap() {
            WirePart::ToolResult { result, .. } => assert_eq!(result, body),
            other => panic!("wrong part {other:?}"),
        }
        // A corrupted row with an unknown kind must error when mapped.
        let _ = s
            .manager
            .store()
            .put_part(mid, "escape_hatch", serde_json::json!({"text": "x"}))
            .unwrap();
        let rows = s.parts_of(mid).unwrap();
        assert!(rows.iter().any(|r| wire_part(r).is_err()));
    }

    #[test]
    fn session_state_view_matches_durable_projection() {
        let (_d, s) = seeded_session();
        let v = s.session_state_view().unwrap();
        assert_eq!(v.session_id, s.id().to_string());
        assert_eq!(v.state, "preparing");
        assert!(v.agent_state.active);
        assert!(!v.agent_state.terminal);
        assert_eq!(v.agent_state.label, "preparing");
        assert!(v.last_event_seq >= 2);
        assert_eq!(v.task_ledger, None);
        // Complete the turn along a legal machine path, then re-read the view.
        s.append_event(
            kilop_core::event::EventKind::ContextPrepared,
            kilop_core::state::AgentState::BuildingContext,
            None,
            None,
        )
        .unwrap();
        s.append_event(
            kilop_core::event::EventKind::ModelStarted,
            kilop_core::state::AgentState::WaitingForModel,
            None,
            None,
        )
        .unwrap();
        s.append_event(
            kilop_core::event::EventKind::ModelChunkReceived,
            kilop_core::state::AgentState::Streaming,
            None,
            None,
        )
        .unwrap();
        s.append_event(
            kilop_core::event::EventKind::ToolCompleted,
            kilop_core::state::AgentState::Validating,
            None,
            None,
        )
        .unwrap();
        s.append_event(
            kilop_core::event::EventKind::TurnCompleted,
            kilop_core::state::AgentState::Completed,
            None,
            None,
        )
        .unwrap();
        let v = s.session_state_view().unwrap();
        assert_eq!(v.state, "completed");
        assert!(v.agent_state.terminal);
    }

    #[test]
    fn session_ids_are_isolated_for_paging_and_seqs() {
        let (_d, m) = test_manager();
        let s1 = session(&m);
        let s2 = {
            let ws = m.create_workspace("/w2").unwrap();
            m.create_session(ws, "t2", "p", "m").unwrap()
        };
        s1.put_message(5, "user", serde_json::json!({"text": "one"}))
            .unwrap();
        s2.put_message(5, "user", serde_json::json!({"text": "two"}))
            .unwrap();
        assert_eq!(s1.message_count().unwrap(), 1);
        assert_eq!(s2.message_count().unwrap(), 1);
        let p1 = s1.messages_page(None, 10).unwrap();
        assert_eq!(p1.session_id, s1.id().to_string());
        assert_eq!(p1.messages[0].session_id, s1.id().to_string());
    }

    #[test]
    fn parts_never_load_beyond_the_page() {
        let (_d, s) = seeded_session();
        let mut mids = Vec::new();
        for i in 0..50 {
            let seq = s.proposed_message_seq().unwrap();
            let mid = s
                .put_message(seq, "assistant", serde_json::json!({"i": i}))
                .unwrap();
            for p in 0..5 {
                s.put_part(
                    mid,
                    PartKind::Text,
                    serde_json::json!({"text": format!("{i}.{p}")}),
                )
                .unwrap();
            }
            mids.push(mid);
        }
        // A page of 10 carries parts of exactly those 10 messages: ≤ 50 parts,
        // never the 250 parts of all 50 messages.
        let page = s.messages_page(None, 10).unwrap();
        let part_count: usize = page.messages.iter().map(|m| m.parts.len()).sum();
        assert!(
            part_count <= 50,
            "page fetched too many parts: {part_count}"
        );
        assert_eq!(page.messages.len(), 10);
    }
}
