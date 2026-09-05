//! Typed, versioned journal event payloads (audits 71-72).
//!
//! Every `event` row carries an explicit payload schema version column
//! (`payload_ver`, store schema v11). Writers stamp the schema version of
//! the payload shape they write; readers decode payloads through
//! [`decode_payload`] and **never** assume v1 — an unknown version is a
//! loud error, never a silent parse of a future shape.
//!
//! Kinds with a REGISTERED v1 schema are decoded strictly (shape violations
//! error loudly at append time too). Kinds without a registered schema are
//! opaque: nothing semantically parses them today, so an opaque payload can
//! never be misread — but its version is still checked, so a payload row
//! stamped with an unknown version refuses to decode.
//!
//! Version bumps: bump [`PAYLOAD_SCHEMA_V`] and register the new shape
//! (keeping the v1 decoder) whenever a registered payload's shape changes.
//! Unregistered kinds that gain a typed reader must register their CURRENT
//! shape as v1 at that moment and bump on every later change.

use faktor_core::event::EventKind;
use serde::{Deserialize, Serialize};

use crate::SessionError;

/// The payload schema version every writer in this crate currently stamps.
/// v1 = the original unversioned payload shapes (writers that existed
/// before versioning keep v1; nothing changed shape this wave).
pub const PAYLOAD_SCHEMA_V: i64 = 1;

/// The typed decode result of one journal event payload.
#[derive(Debug, Clone, PartialEq)]
pub enum TypedEventPayload {
    /// `Failed` events with the `{"message": string}` v1 shape.
    Failed { message: String },
    /// `Failed` events with the abort v1 shape `{"error": string,
    /// "op_id": integer}` (the abort path has always journaled this shape).
    FailedAborted { error: String, op_id: Option<u64> },
    /// A kind with no registered typed schema: carried verbatim, never
    /// semantically parsed. The version gate still applies.
    Opaque(Option<serde_json::Value>),
}

/// Strict v1 schema of a `Failed` payload: `{"message": string}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailedPayload {
    pub message: String,
}

/// Strict v1 schema of the abort `Failed` payload:
/// `{"error": string, "op_id": integer}` (`op_id` absent on idle aborts).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailedAbortPayload {
    pub error: String,
    #[serde(default)]
    pub op_id: Option<u64>,
}

/// Decode one journal payload through its schema version. `ver` unknown for
/// the kind (registered or not) => loud `Malformed` (corrupt), never a
/// silent parse; registered kinds additionally decode strictly.
pub fn decode_payload(
    kind: EventKind,
    ver: i64,
    payload: Option<&serde_json::Value>,
) -> Result<TypedEventPayload, SessionError> {
    if ver != PAYLOAD_SCHEMA_V {
        return Err(SessionError::Malformed(format!(
            "journal payload schema version {ver} of kind {kind:?} is unknown \
             (this reader understands v{PAYLOAD_SCHEMA_V}); refusing to parse a future payload shape"
        )));
    }
    match kind {
        EventKind::Failed => {
            let value = payload.ok_or_else(|| {
                SessionError::Malformed(
                    "Failed event without a payload violates the v1 schema".into(),
                )
            })?;
            if let Ok(abort) = serde_json::from_value::<FailedAbortPayload>(value.clone()) {
                return Ok(TypedEventPayload::FailedAborted {
                    error: abort.error,
                    op_id: abort.op_id,
                });
            }
            let parsed: FailedPayload = serde_json::from_value(value.clone()).map_err(|e| {
                SessionError::Malformed(format!("Failed event payload violates the v1 schema: {e}"))
            })?;
            Ok(TypedEventPayload::Failed {
                message: parsed.message,
            })
        }
        _ => Ok(TypedEventPayload::Opaque(payload.cloned())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_payload_decodes_strictly() {
        let v = serde_json::json!({ "message": "boom" });
        let decoded = decode_payload(EventKind::Failed, 1, Some(&v)).unwrap();
        assert_eq!(
            decoded,
            TypedEventPayload::Failed {
                message: "boom".into()
            }
        );
        // The abort v1 shape decodes too (both shapes share v1: they have
        // coexisted since before versioning).
        let abort = serde_json::json!({ "error": "aborted", "op_id": 7 });
        assert_eq!(
            decode_payload(EventKind::Failed, 1, Some(&abort)).unwrap(),
            TypedEventPayload::FailedAborted {
                error: "aborted".into(),
                op_id: Some(7)
            }
        );
        let idle_abort = serde_json::json!({ "error": "aborted" });
        assert_eq!(
            decode_payload(EventKind::Failed, 1, Some(&idle_abort)).unwrap(),
            TypedEventPayload::FailedAborted {
                error: "aborted".into(),
                op_id: None
            }
        );
        // Shape violations are loud, never a silent fallback parse.
        for hostile in [
            serde_json::json!({ "nope": 1 }),
            serde_json::json!("boom"),
            serde_json::json!(42),
            serde_json::json!({}),
        ] {
            assert!(
                decode_payload(EventKind::Failed, 1, Some(&hostile)).is_err(),
                "{hostile}"
            );
        }
        assert!(decode_payload(EventKind::Failed, 1, None).is_err());
    }

    #[test]
    fn unknown_payload_version_errors_loudly_for_every_kind() {
        // A payload row crafted with schema_ver 999 must make EVERY typed
        // reader error loudly — registered kinds and opaque kinds alike.
        for kind in [
            EventKind::Failed,
            EventKind::TurnCompleted,
            EventKind::ModelChunkReceived,
            EventKind::SessionCreated,
        ] {
            let err = decode_payload(kind, 999, Some(&serde_json::json!({ "x": 1 }))).unwrap_err();
            assert!(
                matches!(err, SessionError::Malformed(_)),
                "{kind:?}: {err:?}"
            );
            let msg = format!("{err}");
            assert!(msg.contains("999"), "{kind:?}: {msg}");
        }
    }

    #[test]
    fn opaque_payloads_are_carried_but_never_parsed() {
        let v = serde_json::json!({ "anything": [1, 2, 3] });
        assert_eq!(
            decode_payload(EventKind::ContextCompacted, 1, Some(&v)).unwrap(),
            TypedEventPayload::Opaque(Some(v))
        );
        assert_eq!(
            decode_payload(EventKind::TurnCompleted, 1, None).unwrap(),
            TypedEventPayload::Opaque(None)
        );
    }
}
