//! The conversation view: messages and parts, mapped to the frozen
//! `faktor-protocol::v756` shapes. Paging never loads more than one page.

use faktor_protocol::v756::{
    Message as WireMessage, MessagesPage, PageMeta, Part as WirePart, ToolResultBody,
};
use faktor_store::{MessageRow, PartRow};

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
    ) -> faktor_core::Result<i64> {
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
    pub fn proposed_message_seq(&self) -> faktor_core::Result<i64> {
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
    ) -> faktor_core::Result<i64> {
        validate_part(kind, &data)?;
        Ok(self
            .manager
            .store()
            .put_part(message_id, kind.as_str(), data)
            .map_err(crate::map_store_err)?)
    }

    pub fn parts_of(&self, message_id: i64) -> faktor_core::Result<Vec<PartRow>> {
        self.manager
            .store()
            .parts_of(message_id)
            .map_err(|e| crate::map_store_err(e).into())
    }

    /// Convenience: a `text` part.
    pub fn put_text_part(&self, message_id: i64, text: &str) -> faktor_core::Result<i64> {
        self.put_part(
            message_id,
            PartKind::Text,
            serde_json::json!({ "text": text }),
        )
    }

    /// Convenience: a `reasoning` part (model thinking; never merged into
    /// the `text` parts).
    pub fn put_reasoning_part(&self, message_id: i64, text: &str) -> faktor_core::Result<i64> {
        self.put_part(
            message_id,
            PartKind::Reasoning,
            serde_json::json!({ "text": text }),
        )
    }

    /// Convenience: a `tool_call` part with an explicit state.
    pub fn put_tool_call_part(
        &self,
        message_id: i64,
        tool_call_id: &str,
        name: &str,
        input: serde_json::Value,
        state: &str,
    ) -> faktor_core::Result<i64> {
        self.put_part(
            message_id,
            PartKind::ToolCall,
            serde_json::json!({
                "tool_call_id": tool_call_id,
                "name": name,
                "input": input,
                "state": state,
            }),
        )
    }

    /// Convenience: a `tool_result` part from a frozen wire body.
    pub fn put_tool_result_part(
        &self,
        message_id: i64,
        tool_call_id: &str,
        body: &ToolResultBody,
    ) -> faktor_core::Result<i64> {
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

    pub fn message_count(&self) -> faktor_core::Result<i64> {
        self.manager
            .store()
            .message_count(self.id)
            .map_err(|e| crate::map_store_err(e).into())
    }

    // -------------------------------------------------- async hot append twins

    // The synchronous put_* twins above stay for non-hot callers (direct
    // store, no actor). The async append_* twins below run the SAME
    // validation and bounds, but route the write through the manager's
    // DbActor (audit 42): a dedicated store thread executes the append, the
    // caller's tokio worker only enqueues and awaits the fsynced response.

    /// Async twin of [`SessionHandle::put_message`] (hot append path).
    /// Returns the durable row id after the actor fsynced the append.
    pub async fn append_message(
        &self,
        seq: i64,
        role: &str,
        data: serde_json::Value,
    ) -> faktor_core::Result<i64> {
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
        let handle = self.manager.actor().handle();
        Ok(handle
            .append_message(self.id, seq, role, data)
            .await
            .map_err(crate::map_store_err)?)
    }

    /// Async twin of [`SessionHandle::put_part`] (hot append path).
    pub async fn append_part(
        &self,
        message_id: i64,
        kind: PartKind,
        data: serde_json::Value,
    ) -> faktor_core::Result<i64> {
        validate_part(kind, &data)?;
        let handle = self.manager.actor().handle();
        Ok(handle
            .append_part(message_id, kind.as_str(), data)
            .await
            .map_err(crate::map_store_err)?)
    }

    /// Async twin of [`SessionHandle::put_text_part`].
    pub async fn append_text_part(&self, message_id: i64, text: &str) -> faktor_core::Result<i64> {
        self.append_part(
            message_id,
            PartKind::Text,
            serde_json::json!({ "text": text }),
        )
        .await
    }

    /// Async twin of [`SessionHandle::put_reasoning_part`].
    pub async fn append_reasoning_part(
        &self,
        message_id: i64,
        text: &str,
    ) -> faktor_core::Result<i64> {
        self.append_part(
            message_id,
            PartKind::Reasoning,
            serde_json::json!({ "text": text }),
        )
        .await
    }

    /// Async twin of [`SessionHandle::put_tool_call_part`].
    #[allow(clippy::too_many_arguments)]
    pub async fn append_tool_call_part(
        &self,
        message_id: i64,
        tool_call_id: &str,
        name: &str,
        input: serde_json::Value,
        state: &str,
    ) -> faktor_core::Result<i64> {
        self.append_part(
            message_id,
            PartKind::ToolCall,
            serde_json::json!({
                "tool_call_id": tool_call_id,
                "name": name,
                "input": input,
                "state": state,
            }),
        )
        .await
    }

    /// Async twin of [`SessionHandle::put_tool_result_part`].
    pub async fn append_tool_result_part(
        &self,
        message_id: i64,
        tool_call_id: &str,
        body: &ToolResultBody,
    ) -> faktor_core::Result<i64> {
        self.append_part(
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
        .await
    }

    /// Newest-first messages with `seq < before` (None = latest page). Never
    /// loads more than one page. The limit is clamped into `[1, MAX_PAGE_SIZE]`.
    pub fn messages_before(
        &self,
        before: Option<i64>,
        limit: i64,
    ) -> faktor_core::Result<Vec<MessageRow>> {
        let limit = limit.clamp(1, MAX_PAGE_SIZE);
        self.manager
            .store()
            .messages_before(self.id, before, limit as u64)
            .map_err(|e| crate::map_store_err(e).into())
    }

    /// Bounded newest-first loader twin of
    /// [`faktor_store::Store::messages_backwards_bounded`] (audit 29): one
    /// conversation window for a turn, newest-first, stopping when EITHER
    /// `max_messages` or `max_bytes` (stored message-payload bytes) is hit.
    /// The store walks the backward index lazily and never reads rows beyond
    /// the returned window — history is selected by the budget-aware
    /// planner, not loaded wholesale and trimmed afterward. Message
    /// granularity is absolute: a single oversized message counts as one
    /// message and may exceed `max_bytes` alone (never a partial row).
    /// `before` cuts strictly (`seq < before`); `u64::MAX` behaves like the
    /// newest page.
    pub fn messages_backwards_bounded(
        &self,
        before: Option<u64>,
        max_messages: u64,
        max_bytes: u64,
    ) -> faktor_core::Result<Vec<MessageRow>> {
        self.manager
            .store()
            .messages_backwards_bounded(self.id, before, max_messages, max_bytes)
            .map_err(|e| crate::map_store_err(e).into())
    }

    /// The frozen protocol page for the webview: metadata + one page, with the
    /// cursor for older pages. Parts are loaded per message *in the page only*.
    pub fn messages_page(
        &self,
        before: Option<i64>,
        limit: i64,
    ) -> faktor_core::Result<MessagesPage> {
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
            page: PageMeta {
                size: limit,
                cursor: next_before,
                has_more,
                total_estimate: None,
            },
        })
    }

    /// Latest message page (the webview's initial load).
    pub fn latest_messages_page(&self, limit: i64) -> faktor_core::Result<MessagesPage> {
        self.messages_page(None, limit)
    }

    /// The frozen `SessionState` projection the UI polls on reconnect.
    pub fn session_state_view(&self) -> faktor_core::Result<faktor_protocol::v756::SessionState> {
        let row = self.row()?;
        let last_event_seq = self.last_event_seq()?.map(|s| s.raw() as i64).unwrap_or(0);
        let ledger = self.get_task_ledger()?;
        Ok(faktor_protocol::v756::SessionState {
            session_id: self.id.to_string(),
            state: crate::state_tag(row.state),
            title: row.title,
            last_event_seq,
            agent_state: faktor_protocol::v756::AgentStateView {
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
            faktor_core::ErrorKind::Conflict,
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
            faktor_core::event::EventKind::ContextPrepared,
            faktor_core::state::AgentState::BuildingContext,
            None,
            None,
        )
        .unwrap();
        s.append_event(
            faktor_core::event::EventKind::ModelStarted,
            faktor_core::state::AgentState::WaitingForModel,
            None,
            None,
        )
        .unwrap();
        s.append_event(
            faktor_core::event::EventKind::ModelChunkReceived,
            faktor_core::state::AgentState::Streaming,
            None,
            None,
        )
        .unwrap();
        s.append_event(
            faktor_core::event::EventKind::ToolCompleted,
            faktor_core::state::AgentState::Validating,
            None,
            None,
        )
        .unwrap();
        s.append_event(
            faktor_core::event::EventKind::TurnCompleted,
            faktor_core::state::AgentState::Completed,
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

    /// A bare session with no conversation rows: the empty-store paging
    /// contract (metadata must still be explicit and truthful).
    #[test]
    fn empty_store_page_carries_full_metadata() {
        let (_d, m) = test_manager();
        let s = session(&m);
        assert_eq!(s.message_count().unwrap(), 0);
        let page = s.messages_page(None, 25).unwrap();
        assert!(page.messages.is_empty(), "no rows on an empty store");
        assert!(!page.has_more);
        assert_eq!(page.next_before, None);
        // The paging metadata object proves boundedness even when empty:
        // the applied page size is echoed, has_more=false, no cursor.
        assert_eq!(page.page.size, 25);
        assert!(!page.page.has_more);
        assert_eq!(page.page.cursor, None);
        // An explicit malicious cursor on the empty store is also a clean
        // empty page — never an error and never a repeat.
        for hostile in [Some(0i64), Some(-1), Some(i64::MIN), Some(i64::MAX)] {
            let p = s.messages_page(hostile, 5).unwrap();
            assert!(p.messages.is_empty());
            assert!(!p.has_more);
            assert_eq!(p.page.size, 5);
        }
    }

    /// Exact-multiple totals must not force a client into a phantom page
    /// loop: the walk ends on a full final page with has_more=false, and a
    /// probe past the end is the final EMPTY page (has_more=false) — never
    /// a rewind to the newest page. Replaying a consumed cursor returns the
    /// identical page (deterministic idempotence, not a duplicate).
    #[test]
    fn exact_multiple_boundary_ends_with_final_empty_page() {
        let (_d, m) = test_manager();
        let s = session(&m);
        for seq in 1..=20i64 {
            s.put_message(seq, "assistant", serde_json::json!({"i": seq}))
                .unwrap();
        }
        let p1 = s.messages_page(None, 10).unwrap();
        assert_eq!(p1.messages.len(), 10);
        assert!(p1.has_more, "20 messages / 10 per page: page 1 has more");
        assert_eq!(p1.page.size, 10);
        assert_eq!(p1.page.cursor, p1.next_before);
        let cursor = p1.next_before.unwrap();
        let p2 = s.messages_page(Some(cursor), 10).unwrap();
        assert_eq!(p2.messages.len(), 10, "exact multiple: final full page");
        assert!(!p2.has_more, "final full page must close the walk");
        assert_eq!(p2.next_before, None);
        assert_eq!(p2.page.cursor, None);
        // Cursor idempotence: replaying page 1's cursor re-yields page 2
        // byte-identical, never a duplicate in a fresh walk.
        let p2b = s.messages_page(Some(cursor), 10).unwrap();
        let a: Vec<i64> = p2.messages.iter().map(|m| m.seq).collect();
        let b: Vec<i64> = p2b.messages.iter().map(|m| m.seq).collect();
        assert_eq!(a, b, "replaying a cursor is deterministic");
        // The probe one row past the oldest retained message is the final
        // EMPTY page: has_more=false, still sized, terminates the walk.
        let oldest = p2.messages.last().unwrap().seq;
        let tail = s.messages_page(Some(oldest), 10).unwrap();
        assert!(tail.messages.is_empty());
        assert!(!tail.has_more);
        assert_eq!(tail.next_before, None);
        assert_eq!(tail.page.size, 10);
        assert_eq!(tail.page.cursor, None);
        // A full clean walk reaches every one of the 20 exactly once.
        let mut seen = Vec::new();
        let mut cursor = None;
        loop {
            let p = s.messages_page(cursor, 10).unwrap();
            if p.messages.is_empty() {
                break;
            }
            for m in &p.messages {
                assert!(!seen.contains(&m.seq), "duplicate in walk");
                seen.push(m.seq);
            }
            if !p.has_more {
                break;
            }
            cursor = p.next_before;
        }
        assert_eq!(seen.len(), 20);
        assert!(seen.windows(2).all(|w| w[0] > w[1]), "newest-first order");
    }

    /// The bounded backward loader twin (audit 29): newest-first window
    /// stopped by the byte bound or the message cap, `before` cuts strictly,
    /// and an oversized newest message is returned whole.
    #[test]
    fn bounded_backward_twin_stops_at_bytes_or_messages() {
        let (_d, m) = test_manager();
        let s = session(&m);
        for seq in 1..=6i64 {
            s.put_message(seq, "user", serde_json::json!({"text": "a".repeat(80)}))
                .unwrap();
        }
        let size = serde_json::to_string(&serde_json::json!({"text": "a".repeat(80)}))
            .unwrap()
            .len() as u64;
        // Exactly the byte boundary: three rows fit a 3-row budget, the
        // fourth stops the walk.
        let win = s.messages_backwards_bounded(None, 10, size * 3).unwrap();
        assert_eq!(win.len(), 3);
        assert_eq!(win[0].seq, 6);
        assert_eq!(win[2].seq, 4);
        // Message cap binds before bytes.
        let capped = s.messages_backwards_bounded(None, 2, u64::MAX).unwrap();
        assert_eq!(capped.iter().map(|r| r.seq).collect::<Vec<_>>(), vec![6, 5]);
        // before cuts strictly between messages.
        let cut = s.messages_backwards_bounded(Some(4), 10, u64::MAX).unwrap();
        assert_eq!(cut.iter().map(|r| r.seq).collect::<Vec<_>>(), vec![3, 2, 1]);
        // Oversized newest message returned whole under a tiny byte budget.
        let big = s
            .put_message(7, "user", serde_json::json!({"text": "z".repeat(500)}))
            .unwrap();
        let _ = big;
        let win = s.messages_backwards_bounded(None, 10, 64).unwrap();
        assert_eq!(
            win.len(),
            1,
            "oversized newest message alone, never partial"
        );
        assert_eq!(win[0].seq, 7);
        assert_eq!(win[0].data["text"].as_str().unwrap().len(), 500);
        // Zero cap is the empty contract.
        assert!(s
            .messages_backwards_bounded(None, 0, u64::MAX)
            .unwrap()
            .is_empty());
    }

    /// Malicious cursors (past the oldest row, negative, absurd) are empty
    /// final pages — has_more=false, never a panic and never a rewind.
    #[test]
    fn malicious_offset_beyond_end_returns_empty_page_not_panic() {
        let (_d, m) = test_manager();
        let s = session(&m);
        for seq in 1..=3i64 {
            s.put_message(seq, "assistant", serde_json::json!({"i": seq}))
                .unwrap();
        }
        // before = seq 1: nothing is older; before ≤ 0 is below any seq.
        for hostile in [Some(1i64), Some(0), Some(-5), Some(i64::MIN)] {
            let p = s.messages_page(hostile, 4).unwrap();
            assert!(p.messages.is_empty(), "cursor {hostile:?}");
            assert!(!p.has_more);
            assert_eq!(p.next_before, None);
            assert_eq!(p.page.size, 4, "empty page keeps the page-size proof");
            assert_eq!(p.page.cursor, None);
        }
        // Absurd large cursors behave like the newest page (clamped SQL), not
        // an error.
        let p = s.messages_page(Some(i64::MAX), 2).unwrap();
        assert_eq!(p.messages.len(), 2, "requested page size wins");
    }

    /// Concurrent appends while three readers page: cursor semantics are
    /// deterministic — a cursor identifies a (journal_seq, message) position,
    /// appends after the cursor are visible on LATER pages only when they
    /// were already durable at the reader's first fetch, and a walk can
    /// never observe a duplicate or a gap.
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn concurrent_appends_never_duplicate_or_gap_across_three_readers() {
        use std::collections::BTreeSet;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc as StdArc;

        let (_d, m) = test_manager();
        let s = StdArc::new(session(&m));
        // 60 durable messages exist before the race starts (seq 1..=60).
        for seq in 1..=60i64 {
            s.put_message(seq, "assistant", serde_json::json!({ "i": seq }))
                .unwrap();
        }
        // One appender adds seq 61..=100 (each new seq is strictly newer, so
        // it can only ever land on a reader's FIRST page).
        let done = StdArc::new(AtomicBool::new(false));
        let appender = {
            let s = s.clone();
            let done = done.clone();
            tokio::spawn(async move {
                for seq in 61..=100i64 {
                    s.put_message(seq, "assistant", serde_json::json!({ "i": seq }))
                        .unwrap();
                    tokio::task::yield_now().await;
                }
                done.store(true, Ordering::SeqCst);
            })
        };
        // Each reader walks full pages until the walk terminates (has_more
        // false or an empty page), repeatedly, while the appender runs.
        // Every completed walk must satisfy: strictly descending seqs (no
        // duplicates), full coverage of every message at-or-below its first
        // page cutoff (no gaps), and a terminal page that reports
        // has_more=false.
        #[derive(Debug)]
        struct Walk {
            cutoff: i64,
            first_page: BTreeSet<i64>,
            seen: Vec<i64>,
        }
        let reader = |s: StdArc<SessionHandle>, done: StdArc<AtomicBool>| async move {
            let mut walks: Vec<Walk> = Vec::new();
            loop {
                if done.load(Ordering::SeqCst) {
                    break;
                }
                let mut cursor: Option<i64> = None;
                let mut seen: Vec<i64> = Vec::new();
                let mut cutoff: Option<i64> = None;
                let mut first_page: BTreeSet<i64> = BTreeSet::new();
                loop {
                    let p = s.messages_page(cursor, 11).unwrap();
                    if p.messages.is_empty() {
                        assert!(!p.has_more, "empty page must close the walk");
                        assert_eq!(p.page.cursor, None);
                        break;
                    }
                    if cutoff.is_none() {
                        cutoff = Some(p.messages.last().unwrap().seq);
                        first_page = p.messages.iter().map(|m| m.seq).collect();
                    }
                    for msg in &p.messages {
                        assert!(
                            seen.last().map(|l| *l > msg.seq).unwrap_or(true),
                            "walk must be strictly newest-first (duplicate or rewind)"
                        );
                        seen.push(msg.seq);
                    }
                    if !p.has_more {
                        break;
                    }
                    cursor = p.next_before;
                }
                // Cooperative yield between walks: the readers and the
                // appender share one runtime, and the appender must
                // always make progress no matter how loaded the box is
                // (a reader that never yields can starve it).
                tokio::task::yield_now().await;
                // Every walk terminates on a closed page: an empty page or a
                // final page with has_more=false (never an unbounded loop —
                // that is exactly what paging-is-fundamental forbids).
                let cutoff = cutoff.unwrap_or(i64::MAX);
                walks.push(Walk {
                    cutoff,
                    first_page,
                    seen,
                });
            }
            walks
        };
        let readers: Vec<_> = (0..3)
            .map(|_| tokio::spawn(reader(s.clone(), done.clone())))
            .collect();
        appender.await.unwrap();
        let all_walks: Vec<Vec<Walk>> = {
            let mut out = Vec::new();
            for r in readers {
                out.push(r.await.unwrap());
            }
            out
        };
        // All 100 messages exist once the appender joined.
        assert_eq!(s.message_count().unwrap(), 100);
        // Every completed walk is duplicate-free AND gap-free below its own
        // cutoff: any message at or below the first page's cutoff already
        // existed when the walk started (the appender only adds strictly
        // newer seqs), so the walk must have seen each one exactly once.
        let total_walks: usize = all_walks.iter().map(|w| w.len()).sum();
        assert!(
            total_walks >= 3,
            "readers must have completed walks: {all_walks:?}"
        );
        for (r, walks) in all_walks.iter().enumerate() {
            assert!(!walks.is_empty(), "reader {r} completed no walk");
            for w in walks {
                let seen: BTreeSet<i64> = w.seen.iter().copied().collect();
                assert_eq!(seen.len(), w.seen.len(), "duplicate inside a walk");
                assert!(
                    w.seen.windows(2).all(|pair| pair[0] > pair[1]),
                    "walk must stay strictly newest-first"
                );
                // No gap: every message at-or-below the first page's cutoff
                // already existed when the walk started (the appender only
                // adds strictly newer seqs), so the walk must have seen each
                // one exactly once.
                let expected: BTreeSet<i64> = (1..=w.cutoff).collect();
                assert!(
                    expected.is_subset(&seen),
                    "reader {r} walk with cutoff {} has a gap: missing {:?}",
                    w.cutoff,
                    expected.difference(&seen).collect::<Vec<_>>()
                );
                // And the only rows above the cutoff a walk may report are
                // the ones it saw on its FIRST page (appends after that can
                // never enter older pages).
                let extras: BTreeSet<i64> =
                    seen.iter().copied().filter(|s| *s > w.cutoff).collect();
                assert!(
                    extras.is_subset(&w.first_page),
                    "reader {r} walk leaked newer rows beyond its first page: {extras:?}"
                );
            }
        }
    }

    /// Paging across a compaction boundary (older rows pruned, a summary
    /// row inserted at a NEWER seq): cursors must not be corrupted — the
    /// retained window is returned with no duplicate and no gap, and the
    /// walk terminates exactly at the boundary.
    #[test]
    fn paging_across_compaction_boundary_no_dup_or_gap_for_retained() {
        use std::collections::BTreeSet;
        let (_d, m) = test_manager();
        let s = session(&m);
        for seq in 1..=50i64 {
            s.put_message(seq, "assistant", serde_json::json!({ "i": seq }))
                .unwrap();
        }
        // The compaction pass: prune rows 10..=40 and insert the summary
        // message at seq 51 (newer than every pruned row). Deletes leave
        // holes; nothing is renumbered.
        for seq in 10..=40i64 {
            s.delete_message(seq).unwrap();
        }
        s.put_message(51, "assistant", serde_json::json!({ "summary": true }))
            .unwrap();
        let retained: BTreeSet<i64> = (1..=51i64).filter(|seq| !(10..=40).contains(seq)).collect();
        assert_eq!(retained.len(), 20, "9 kept + 10 kept + 1 summary");
        // Full walk: exactly the retained messages, newest first, once each.
        let mut seen: Vec<i64> = Vec::new();
        let mut cursor: Option<i64> = None;
        loop {
            let p = s.messages_page(cursor, 13).unwrap();
            if p.messages.is_empty() {
                assert!(!p.has_more);
                break;
            }
            for m in &p.messages {
                assert!(retained.contains(&m.seq), "pruned row {} leaked", m.seq);
                assert!(!seen.contains(&m.seq), "duplicate {}", m.seq);
                seen.push(m.seq);
            }
            if !p.has_more {
                break;
            }
            cursor = p.next_before;
        }
        assert_eq!(seen.len(), retained.len(), "gap: walk did not cover all");
        assert_eq!(
            seen.iter().copied().collect::<BTreeSet<i64>>(),
            retained,
            "retained window must appear exactly once"
        );
        assert!(seen.windows(2).all(|w| w[0] > w[1]));
        // The cursor boundary across the prune: a request just past the
        // oldest retained row is the final empty page (has_more=false).
        let tail = s.messages_page(Some(1), 13).unwrap();
        assert!(tail.messages.is_empty());
        assert!(!tail.has_more);
        assert_eq!(tail.page.cursor, None);
    }
}
