//! Harness 5: journal event payload decode.
//!
//! `faktor_session::decode_payload` decodes one journal event payload
//! strictly by `(EventKind, schema version)`: an unknown version or a
//! shape violation is a loud typed error, never a silent parse. Adversarial
//! properties: arbitrary JSON (and non-JSON text) across every `EventKind`
//! and hostile schema versions must never panic, and an accepted decode
//! must be reproducible (decoding the identical input again yields the
//! identical outcome).

use faktor_core::event::EventKind;

use super::{Lcg, Outcome};

const KINDS: &[EventKind] = &[
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
    EventKind::PromptAdmitted,
    EventKind::PhaseChanged,
    EventKind::ReplayStarted,
    EventKind::CrashDetected,
    EventKind::RecoveryApplied,
    EventKind::SessionEnded,
    EventKind::Suspended,
    EventKind::Resumed,
    EventKind::Failed,
];

fn decode(
    kind: EventKind,
    version: i64,
    payload: Option<&serde_json::Value>,
) -> Result<String, String> {
    faktor_session::decode_payload(kind, version, payload)
        .map(|d| format!("{d:?}"))
        .map_err(|e| format!("{e}"))
}

/// Fuzz the event payload decoder. `bytes` is interpreted as JSON when
/// possible and decoded against every kind × a hostile version set.
pub fn harness_event_payload_decode(bytes: &[u8]) -> Outcome {
    let mut lcg = Lcg::new(bytes.len() as u64 ^ 0xEAC);
    let Ok(text) = std::str::from_utf8(bytes) else {
        return Outcome::NotApplicable;
    };
    let value: Option<serde_json::Value> = serde_json::from_str(text).ok();
    let versions = [
        1,
        2,
        3,
        faktor_session::PAYLOAD_SCHEMA_V,
        faktor_session::PAYLOAD_SCHEMA_V + 1,
        0,
        -1,
        i64::MAX,
    ];
    for kind in KINDS {
        for &version in &versions {
            let payload = if lcg.chance(15) {
                None
            } else {
                value.as_ref().or(Some(&serde_json::Value::Null))
            };
            let first = decode(*kind, version, payload);
            // Reproducibility: decoding the same input again yields the
            // identical outcome.
            let second = decode(*kind, version, payload);
            if first != second {
                return Outcome::Violation(format!(
                    "decode of {kind:?} v{version} is not reproducible: {first:?} vs {second:?}"
                ));
            }
        }
    }
    Outcome::Clean
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Deterministic hostile JSON value from the seed.
    fn json_value(mut lcg: Lcg) -> serde_json::Value {
        match lcg.below(9) {
            0 => serde_json::Value::Null,
            1 => serde_json::json!(lcg.next_u64()),
            2 => serde_json::json!(lcg.next_u64() as i64),
            3 => serde_json::json!(lcg.next_u64() as f64 / 3.0),
            4 => serde_json::Value::Bool(lcg.chance(50)),
            5 => serde_json::json!({}),
            6 => {
                let n = 1 + lcg.below(8) as usize;
                let mut arr = Vec::new();
                for _ in 0..n {
                    arr.push(serde_json::json!(lcg.next_u64()));
                }
                serde_json::Value::Array(arr)
            }
            _ => {
                // Deeply nested hostile object with reserved-ish keys.
                let mut obj = serde_json::Map::new();
                obj.insert(
                    "path".into(),
                    serde_json::json!({ "depth": 1 + lcg.below(9) }),
                );
                obj.insert("hash".into(), serde_json::json!("h".repeat(64)));
                obj.insert("bytes".into(), serde_json::json!(lcg.next_u64()));
                serde_json::Value::Object(obj)
            }
        }
    }

    #[test]
    fn seeded_pseudo_fuzz_event_payload_decode_2000() {
        let mut lcg = Lcg::new(0x5EED_0000_0000_0005);
        let mut clean = 0;
        for i in 0..2000u64 {
            let mut bytes = Vec::new();
            match i % 5 {
                0 => {
                    let v = json_value(Lcg::new(lcg.next_u64() ^ 0xF00D));
                    bytes.extend_from_slice(serde_json::to_string(&v).unwrap().as_bytes());
                }
                1 => {
                    // Raw hostile bytes.
                    let n = 1 + lcg.below(400) as usize;
                    for _ in 0..n {
                        bytes.push(lcg.next_u64() as u8);
                    }
                }
                2 => {
                    // JSON that starts valid then breaks.
                    let s = format!("{{\"a\": [1, 2, {}], \"b\": \"unterminated", lcg.next_u64());
                    bytes.extend_from_slice(s.as_bytes());
                }
                3 => {
                    // Shapes known to be decoded per kind.
                    let s = match i % 4 {
                        0 => "{\"path\": \"/x\", \"hash\": \"h\"}".to_string(),
                        1 => {
                            "{\"title\": \"t\", \"provider\": \"p\", \"model\": \"m\"}".to_string()
                        }
                        2 => "[1, 2, 3]".to_string(),
                        _ => "\"a plain string\"".to_string(),
                    };
                    bytes.extend_from_slice(s.as_bytes());
                }
                _ => {
                    // Empty and whitespace.
                    bytes.push(b'\n');
                }
            }
            match harness_event_payload_decode(&bytes) {
                Outcome::Clean => clean += 1,
                Outcome::NotApplicable => {}
                v => panic!("iteration {i}: {v}"),
            }
        }
        assert!(clean > 0);
    }
}
