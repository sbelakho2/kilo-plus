//! Harness: ACP Content-Length frame decoder (`faktor_acp::protocol`).
//!
//! Adversarial properties for arbitrary byte streams: never panic, and the
//! incremental decoder stays bounded and honest:
//!
//! - a returned frame consumes `1..=bytes.len()` bytes (never more than the
//!   caller fed);
//! - a returned frame re-encodes to a frame that decodes back to the
//!   IDENTICAL JSON-RPC value, consuming exactly its own length;
//! - any strict prefix of the input either stays incomplete or decodes a
//!   frame that fits inside that prefix — never a phantom frame longer than
//!   what was fed;
//! - decoding is deterministic.
//!
//! Hostile headers (duplicate/oversized/non-numeric `Content-Length`,
//! over-long header blocks) surface as typed `Err` and are exercised here.

use super::Outcome;

fn parse(bytes: &[u8]) -> Result<Option<(usize, serde_json::Value)>, String> {
    faktor_acp::protocol::parse_frame(bytes)
}

/// Fuzz the ACP frame decoder.
pub fn harness_acp_frame_decoder(bytes: &[u8]) -> Outcome {
    let first = parse(bytes);
    if first != parse(bytes) {
        return Outcome::Violation("ACP frame decoding is not deterministic".into());
    }
    let Some((consumed, value)) = (match first {
        Ok(found) => found,
        // Typed framing rejection: hostile header/body, still panic-free.
        Err(_) => return Outcome::Clean,
    }) else {
        // Incomplete frame: callers keep feeding; nothing to assert yet.
        return Outcome::Clean;
    };
    if consumed == 0 || consumed > bytes.len() {
        return Outcome::Violation(format!(
            "ACP decoder consumed {consumed} bytes of a {}-byte input",
            bytes.len()
        ));
    }
    let encoded = match faktor_acp::protocol::encode(&value) {
        Ok(encoded) => encoded,
        Err(e) => {
            return Outcome::Violation(format!("a decoded ACP value failed to re-encode: {e}"))
        }
    };
    match parse(&encoded) {
        Ok(Some((consumed2, value2))) if consumed2 == encoded.len() && value2 == value => {}
        other => {
            return Outcome::Violation(format!(
                "ACP frame round-trip broke: decoded {value:?}, re-encoded frame re-decodes to {other:?}"
            ))
        }
    }
    // Incremental safety: no strict prefix may decode a frame that does not
    // fit inside it. Deterministic cut set, never mid-logic randomness.
    let mut cuts: Vec<usize> = (1..=8.min(bytes.len())).collect();
    for divisor in [2u64, 3, 4] {
        let cut = (bytes.len() as u64 / divisor) as usize;
        if cut > 0 && cut < bytes.len() {
            cuts.push(cut);
        }
    }
    if bytes.len() > 1 {
        cuts.push(bytes.len() - 1);
    }
    for cut in cuts {
        if cut == 0 || cut >= bytes.len() {
            continue;
        }
        if let Ok(Some((c, _))) = parse(&bytes[..cut]) {
            if c > cut {
                return Outcome::Violation(format!(
                    "a {cut}-byte prefix decoded a {c}-byte frame (phantom frame)"
                ));
            }
        }
    }
    Outcome::Clean
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Lcg;

    fn framed(method: &str, id: u64, params: serde_json::Value) -> Vec<u8> {
        faktor_acp::protocol::frame(method.to_string(), id, params)
    }

    /// Deterministic corpus: canonical frames, hostile headers, truncated
    /// bodies, raw bytes and byte mutations — 2000 iterations.
    #[test]
    fn seeded_pseudo_fuzz_acp_frame_decoder_2000() {
        let mut lcg = Lcg::new(0x5EED_0000_0000_0006);
        let mut clean = 0;
        for i in 0..2000u64 {
            let mut bytes = Vec::new();
            match i % 6 {
                0 => {
                    // Canonical request/notification frames.
                    let params = serde_json::json!({
                        "sessionId": format!("sess-{i}"),
                        "prompt": [{"type": "text", "text": "hello λ"}],
                    });
                    if lcg.chance(50) {
                        bytes = framed("session/prompt", i + 1, params);
                    } else {
                        bytes = faktor_acp::protocol::notification_frame(
                            "session/update".to_string(),
                            params,
                        );
                    }
                }
                1 => {
                    // Hostile headers: bad/duplicate/oversized lengths.
                    let header = match lcg.below(5) {
                        0 => "Content-Length: nope\r\n\r\n",
                        1 => "Content-Length: -3\r\n\r\n",
                        2 => "Content-Length: 99999999999999999999\r\n\r\n",
                        3 => "Content-Length: 2\r\nContent-Length: 2\r\n\r\n{}",
                        _ => "Content-Length: 33554432\r\n\r\n",
                    };
                    bytes.extend_from_slice(header.as_bytes());
                }
                2 => {
                    // Truncated canonical frame (header intact, body cut).
                    let frame = framed(
                        "initialize",
                        lcg.next_u64() % 1000,
                        serde_json::json!({"protocolVersion": 1}),
                    );
                    let cut = 1 + (lcg.next_u64() as usize) % frame.len();
                    bytes.extend_from_slice(&frame[..cut]);
                }
                3 => {
                    // A complete frame followed by trailing garbage.
                    bytes = framed(
                        "session/new",
                        lcg.next_u64() % 1000,
                        serde_json::json!({"cwd": "/tmp"}),
                    );
                    bytes.extend_from_slice(b"\r\nnot-a-frame");
                }
                4 => {
                    // Raw random bytes.
                    let n = 1 + lcg.below(700) as usize;
                    for _ in 0..n {
                        bytes.push(lcg.next_u64() as u8);
                    }
                }
                _ => {
                    // JSON body with a valid header but hostile payload.
                    let body = match lcg.below(3) {
                        0 => "{\"jsonrpc\":\"2.0\"".to_string(),
                        1 => format!("[\"{}\"]", "x".repeat(1 + lcg.below(50) as usize)),
                        _ => format!("{{\"method\":\"m{}\"}}", lcg.next_u64() % 100),
                    };
                    bytes.extend_from_slice(
                        format!("Content-Length: {}\r\n\r\n{}", body.len(), body).as_bytes(),
                    );
                }
            }
            match harness_acp_frame_decoder(&bytes) {
                Outcome::Clean => clean += 1,
                v => panic!("iteration {i}: {v}"),
            }
        }
        assert!(clean > 0, "canonical frames must be exercised");
    }

    #[test]
    fn acp_frame_harness_is_deterministic() {
        let mut lcg = Lcg::new(0xACF);
        for _ in 0..50 {
            let n = 1 + lcg.below(600) as usize;
            let bytes: Vec<u8> = (0..n).map(|_| lcg.next_u64() as u8).collect();
            assert_eq!(
                harness_acp_frame_decoder(&bytes),
                harness_acp_frame_decoder(&bytes)
            );
        }
    }

    /// Canonical frames decode and round-trip (sanity of the property).
    #[test]
    fn canonical_acp_frames_roundtrip() {
        let frame = framed("session/cancel", 7, serde_json::json!({"sessionId": "s1"}));
        match faktor_acp::protocol::parse_frame(&frame).unwrap() {
            Some((consumed, value)) => {
                assert_eq!(consumed, frame.len());
                assert_eq!(value["method"], "session/cancel");
            }
            None => panic!("canonical frame must decode"),
        }
    }
}
