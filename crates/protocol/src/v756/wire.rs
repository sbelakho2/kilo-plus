//! The v7.5.6 wire compatibility surface (subset): the exact JSON DTOs the
//! real Kilo VS Code extension sends and receives. These shapes are the
//! frozen contract — field names are camelCase on the wire, every id is a
//! string (`sessionID`, `messageID`, `callID`, `providerID`, `modelID`), and
//! every request/response type rejects unknown fields (an unknown field is a
//! protocol drift signal and must fail loudly, never be ignored).
//!
//! These types are the *wire* side only. The internal domain model lives in
//! `kilop-core`/`kilop-store` rows and `super::{Part, Message}`; `super::mapper`
//! translates strictly between the two surfaces.

use serde::{Deserialize, Serialize};

/// `POST /session` — create a session.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SessionCreateRequest {
    #[serde(rename = "parentID")]
    pub parent_id: Option<String>,
    pub title: Option<String>,
    pub agent: Option<String>,
    pub model: Option<SessionModel>,
    pub metadata: Option<serde_json::Value>,
    pub permission: Option<serde_json::Value>,
    pub platform: Option<String>,
    #[serde(rename = "workspaceID")]
    pub workspace_id: Option<String>,
    pub sandbox_inheritance_token: Option<String>,
}

/// The model selection inside `SessionCreateRequest`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SessionModel {
    pub id: String,
    #[serde(rename = "providerID")]
    pub provider_id: String,
    pub variant: Option<String>,
}

/// `POST /session` response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SessionCreateResponse {
    #[serde(rename = "sessionID")]
    pub session_id: String,
    pub title: String,
    pub created_ms: i64,
}

/// `GET /session` — list sessions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SessionListResponse {
    pub sessions: Vec<SessionSummary>,
}

/// One row of `SessionListResponse` and the `GET /session/{sessionID}` body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SessionSummary {
    #[serde(rename = "sessionID")]
    pub session_id: String,
    pub title: String,
    pub state: String,
    pub created_ms: i64,
    pub updated_ms: i64,
}

/// `POST /session/{sessionID}/message` — send one message (a full turn).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MessageSendRequest {
    #[serde(rename = "messageID")]
    pub message_id: Option<String>,
    pub model: MessageModel,
    pub agent: Option<String>,
    pub no_reply: Option<bool>,
    pub tools: Option<Vec<String>>,
    pub format: Option<String>,
    pub system: Option<String>,
    pub variant: Option<String>,
    pub snapshot_initialization: Option<bool>,
    pub editor_context: Option<serde_json::Value>,
    pub parts: Vec<WirePart>,
}

/// The per-message model selection inside `MessageSendRequest`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MessageModel {
    #[serde(rename = "providerID")]
    pub provider_id: String,
    #[serde(rename = "modelID")]
    pub model_id: String,
}

/// `POST /session/{sessionID}/message` response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MessageSendResponse {
    #[serde(rename = "messageID")]
    pub message_id: String,
    pub accepted: bool,
    pub queued: bool,
}

/// One part of a message. The discriminator is the `type` field with the
/// exact v7.5.6 kind names; unknown kinds are rejected at parse time.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(
    tag = "type",
    rename_all = "camelCase",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
pub enum WirePart {
    Text {
        text: String,
    },
    Subtask {
        label: Option<String>,
        note: Option<String>,
    },
    Reasoning {
        text: String,
    },
    File {
        path: String,
        content: Option<String>,
        mode: Option<String>,
    },
    Tool {
        #[serde(rename = "callID")]
        call_id: String,
        name: String,
        input: serde_json::Value,
        state: Option<String>,
        output: Option<serde_json::Value>,
    },
    StepStart {
        id: Option<String>,
        title: Option<String>,
    },
    StepFinish {
        id: Option<String>,
        title: Option<String>,
        outcome: Option<String>,
    },
    Snapshot {
        #[serde(rename = "sessionID")]
        session_id: Option<String>,
        #[serde(rename = "messageID")]
        message_id: Option<String>,
    },
    Patch {
        #[serde(rename = "sessionID")]
        session_id: Option<String>,
        #[serde(rename = "messageID")]
        message_id: Option<String>,
        path: Option<String>,
        diff: Option<String>,
    },
    Agent {
        id: Option<String>,
        name: Option<String>,
        state: Option<String>,
    },
    Retry {
        reason: Option<String>,
    },
    Compaction {
        before_tokens: Option<u64>,
        after_tokens: Option<u64>,
    },
}

/// One message as served by `GET /session/{sessionID}/message`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WireMessage {
    #[serde(rename = "sessionID")]
    pub session_id: String,
    #[serde(rename = "messageID")]
    pub message_id: String,
    pub role: String,
    pub parts: Vec<WirePart>,
    pub created_ms: i64,
    #[serde(rename = "providerID")]
    pub provider_id: Option<String>,
    #[serde(rename = "modelID")]
    pub model_id: Option<String>,
}

/// A page of messages, newest first.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WireMessagesPage {
    #[serde(rename = "sessionID")]
    pub session_id: String,
    pub messages: Vec<WireMessage>,
    pub has_more: bool,
}

/// `POST /session/{sessionID}/abort` body. May be empty.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AbortBody {
    #[serde(rename = "messageID")]
    pub message_id: Option<String>,
}

/// `POST /session/{sessionID}/revert` body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RevertBody {
    #[serde(rename = "messageID")]
    pub message_id: String,
}

/// `POST /session/{sessionID}/revert` / `unrevert` response. `message` is
/// present only when the operation was refused (honest failure).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RevertResponse {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// `GET /session/{sessionID}/diff` — frozen stub shape.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DiffResponse {
    pub diff: Option<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(v: &serde_json::Value) -> Vec<String> {
        let mut k: Vec<String> = v.as_object().unwrap().keys().cloned().collect();
        k.sort_unstable();
        k
    }

    #[test]
    fn session_create_request_wire_names_are_camel_case() {
        let req = SessionCreateRequest {
            parent_id: Some("sess-1".into()),
            title: Some("t".into()),
            agent: Some("a".into()),
            model: Some(SessionModel {
                id: "qwen3.8".into(),
                provider_id: "ollama".into(),
                variant: Some("fast".into()),
            }),
            metadata: Some(serde_json::json!({"k": 1})),
            permission: None,
            platform: Some("darwin".into()),
            workspace_id: Some("/w".into()),
            sandbox_inheritance_token: Some("tok".into()),
        };
        let v = serde_json::to_value(&req).unwrap();
        assert_eq!(
            keys(&v),
            vec![
                "agent",
                "metadata",
                "model",
                "parentID",
                "permission",
                "platform",
                "sandboxInheritanceToken",
                "title",
                "workspaceID"
            ]
        );
        // Roundtrip is exact (nulls are null, absent stays absent).
        let back: SessionCreateRequest = serde_json::from_value(v.clone()).unwrap();
        assert_eq!(back, req);
        // The model object carries providerID (never provider_id).
        assert_eq!(v["model"]["providerID"], "ollama");
        // Unknown fields are drift and must fail loudly.
        let evil = serde_json::json!({"model": {"id": "m", "providerID": "p"}, "smuggled": 1});
        assert!(serde_json::from_value::<SessionCreateRequest>(evil).is_err());
        // model is optional (the daemon falls back to "default").
        assert!(serde_json::from_value::<SessionCreateRequest>(serde_json::json!({})).is_ok());
    }

    #[test]
    fn session_create_response_wire_names_are_camel_case() {
        let resp = SessionCreateResponse {
            session_id: "sess-1001".into(),
            title: "t".into(),
            created_ms: 1750000000000,
        };
        let v = serde_json::to_value(&resp).unwrap();
        assert_eq!(keys(&v), vec!["createdMs", "sessionID", "title"]);
        let back: SessionCreateResponse = serde_json::from_value(v).unwrap();
        assert_eq!(back, resp);
    }

    #[test]
    fn session_summary_and_list_are_strict() {
        let s = SessionSummary {
            session_id: "1".into(),
            title: "t".into(),
            state: "ready_for_next_turn".into(),
            created_ms: 1,
            updated_ms: 2,
        };
        let v = serde_json::to_value(&s).unwrap();
        assert_eq!(
            keys(&v),
            vec!["createdMs", "sessionID", "state", "title", "updatedMs"]
        );
        let list = SessionListResponse {
            sessions: vec![s.clone()],
        };
        let back: SessionListResponse =
            serde_json::from_value(serde_json::to_value(&list).unwrap()).unwrap();
        assert_eq!(back, list);
        // Unknown list fields are rejected.
        let evil = serde_json::json!({"sessions": [], "total": 5});
        assert!(serde_json::from_value::<SessionListResponse>(evil).is_err());
        // Unknown summary fields are rejected.
        let evil = serde_json::json!({"sessionID": "1", "title": "t", "state": "s",
            "createdMs": 1, "updatedMs": 2, "extra": true});
        assert!(serde_json::from_value::<SessionSummary>(evil).is_err());
    }

    #[test]
    fn message_send_request_wire_names_are_camel_case() {
        let req = MessageSendRequest {
            message_id: Some("msg-1".into()),
            model: MessageModel {
                provider_id: "ollama".into(),
                model_id: "qwen3.8".into(),
            },
            agent: None,
            no_reply: Some(false),
            tools: Some(vec!["read_file".into()]),
            format: None,
            system: None,
            variant: None,
            snapshot_initialization: Some(false),
            editor_context: Some(serde_json::json!({"file": "a.rs"})),
            parts: vec![WirePart::Text { text: "hi".into() }],
        };
        let v = serde_json::to_value(&req).unwrap();
        assert_eq!(
            keys(&v),
            vec![
                "agent",
                "editorContext",
                "format",
                "messageID",
                "model",
                "noReply",
                "parts",
                "snapshotInitialization",
                "system",
                "tools",
                "variant"
            ]
        );
        assert_eq!(v["model"]["providerID"], "ollama");
        assert_eq!(v["model"]["modelID"], "qwen3.8");
        let back: MessageSendRequest = serde_json::from_value(v).unwrap();
        assert_eq!(back, req);
        // Unknown fields are drift.
        let evil = serde_json::json!({"model": {"providerID": "p", "modelID": "m"},
            "parts": [], "secret": 1});
        assert!(serde_json::from_value::<MessageSendRequest>(evil).is_err());
        // model and parts are required.
        assert!(serde_json::from_value::<MessageSendRequest>(serde_json::json!({})).is_err());
        assert!(
            serde_json::from_value::<MessageSendRequest>(serde_json::json!({
                "model": {"providerID": "p", "modelID": "m"}
            }))
            .is_err()
        );
    }

    #[test]
    fn wire_part_type_tags_are_the_exact_v756_kinds() {
        let cases: Vec<(&str, WirePart)> = vec![
            ("text", WirePart::Text { text: "x".into() }),
            (
                "subtask",
                WirePart::Subtask {
                    label: Some("l".into()),
                    note: Some("n".into()),
                },
            ),
            ("reasoning", WirePart::Reasoning { text: "r".into() }),
            (
                "file",
                WirePart::File {
                    path: "a.rs".into(),
                    content: Some("c".into()),
                    mode: Some("edit".into()),
                },
            ),
            (
                "tool",
                WirePart::Tool {
                    call_id: "c1".into(),
                    name: "read_file".into(),
                    input: serde_json::json!({"p": 1}),
                    state: Some("running".into()),
                    output: None,
                },
            ),
            (
                "stepStart",
                WirePart::StepStart {
                    id: Some("s1".into()),
                    title: Some("t".into()),
                },
            ),
            (
                "stepFinish",
                WirePart::StepFinish {
                    id: Some("s1".into()),
                    title: Some("t".into()),
                    outcome: Some("ok".into()),
                },
            ),
            (
                "snapshot",
                WirePart::Snapshot {
                    session_id: Some("1".into()),
                    message_id: Some("2".into()),
                },
            ),
            (
                "patch",
                WirePart::Patch {
                    session_id: None,
                    message_id: None,
                    path: Some("a.rs".into()),
                    diff: Some("@@".into()),
                },
            ),
            (
                "agent",
                WirePart::Agent {
                    id: Some("a1".into()),
                    name: Some("n".into()),
                    state: Some("idle".into()),
                },
            ),
            (
                "retry",
                WirePart::Retry {
                    reason: Some("r".into()),
                },
            ),
            (
                "compaction",
                WirePart::Compaction {
                    before_tokens: Some(100),
                    after_tokens: Some(50),
                },
            ),
        ];
        for (tag, part) in cases {
            let v = serde_json::to_value(&part).unwrap();
            assert_eq!(v["type"], tag, "type tag drift for {tag}");
            let back: WirePart = serde_json::from_value(v).unwrap();
            assert_eq!(back, part, "roundtrip drift for {tag}");
        }
        // Tool wire names: callID, not call_id.
        let v = serde_json::to_value(&WirePart::Tool {
            call_id: "c".into(),
            name: "n".into(),
            input: serde_json::Value::Null,
            state: None,
            output: None,
        })
        .unwrap();
        assert_eq!(
            keys(&v),
            vec!["callID", "input", "name", "output", "state", "type"]
        );
    }

    #[test]
    fn wire_part_rejects_unknown_types_and_missing_fields() {
        // Unknown kind: loud parse failure (frozen discriminator set).
        let evil = serde_json::json!({"type": "escape_hatch", "text": "x"});
        assert!(serde_json::from_value::<WirePart>(evil).is_err());
        // Missing type tag.
        assert!(serde_json::from_value::<WirePart>(serde_json::json!({"text": "x"})).is_err());
        // Missing required fields per variant.
        assert!(serde_json::from_value::<WirePart>(serde_json::json!({"type": "text"})).is_err());
        assert!(
            serde_json::from_value::<WirePart>(serde_json::json!({"type": "reasoning"})).is_err()
        );
        assert!(serde_json::from_value::<WirePart>(serde_json::json!({"type": "file"})).is_err());
        assert!(serde_json::from_value::<WirePart>(serde_json::json!({"type": "tool"})).is_err());
        // Unknown fields inside a known variant are rejected.
        let evil = serde_json::json!({"type": "text", "text": "x", "smuggled": 1});
        assert!(serde_json::from_value::<WirePart>(evil).is_err());
        // Nulls are allowed only where declared Option.
        let ok = serde_json::json!({"type": "tool", "callID": "c", "name": "n",
            "input": null, "state": null, "output": null});
        assert!(serde_json::from_value::<WirePart>(ok).is_ok());
        // Non-Option fields may not be null.
        let evil = serde_json::json!({"type": "text", "text": null});
        assert!(serde_json::from_value::<WirePart>(evil).is_err());
    }

    #[test]
    fn wire_message_and_page_shapes_are_strict() {
        let m = WireMessage {
            session_id: "1".into(),
            message_id: "2".into(),
            role: "assistant".into(),
            parts: vec![WirePart::Text { text: "hi".into() }],
            created_ms: 5,
            provider_id: Some("ollama".into()),
            model_id: Some("qwen3.8".into()),
        };
        let v = serde_json::to_value(&m).unwrap();
        assert_eq!(
            keys(&v),
            vec![
                "createdMs",
                "messageID",
                "modelID",
                "parts",
                "providerID",
                "role",
                "sessionID"
            ]
        );
        let page = WireMessagesPage {
            session_id: "1".into(),
            messages: vec![m],
            has_more: false,
        };
        let v = serde_json::to_value(&page).unwrap();
        assert_eq!(keys(&v), vec!["hasMore", "messages", "sessionID"]);
        let back: WireMessagesPage = serde_json::from_value(v).unwrap();
        assert_eq!(back, page);
        // Unknown fields rejected.
        let evil = serde_json::json!({"sessionID": "1", "messages": [], "hasMore": false,
            "cursor": 5});
        assert!(serde_json::from_value::<WireMessagesPage>(evil).is_err());
    }

    #[test]
    fn abort_revert_diff_bodies_are_strict_and_camel_case() {
        let a: AbortBody = serde_json::from_str(r#"{"messageID": "1"}"#).unwrap();
        assert_eq!(a.message_id.as_deref(), Some("1"));
        let a: AbortBody = serde_json::from_str("{}").unwrap();
        assert_eq!(a.message_id, None);
        assert!(serde_json::from_str::<AbortBody>(r#"{"nope": 1}"#).is_err());

        let r: RevertBody = serde_json::from_str(r#"{"messageID": "3"}"#).unwrap();
        assert_eq!(r.message_id, "3");
        assert!(serde_json::from_str::<RevertBody>("{}").is_err());
        assert!(serde_json::from_str::<RevertBody>(r#"{"messageID": "3", "x": 1}"#).is_err());

        let ok = RevertResponse {
            ok: true,
            message: None,
        };
        let v = serde_json::to_value(&ok).unwrap();
        assert_eq!(
            keys(&v),
            vec!["ok"],
            "success body is exactly {{\"ok\":true}}"
        );
        let refused = RevertResponse {
            ok: false,
            message: Some("no".into()),
        };
        let v = serde_json::to_value(&refused).unwrap();
        assert_eq!(keys(&v), vec!["message", "ok"]);

        let d = DiffResponse { diff: None };
        let v = serde_json::to_value(&d).unwrap();
        assert_eq!(keys(&v), vec!["diff"]);
        assert!(v["diff"].is_null());
    }
}
