//! Harness: frozen v7.5.6 compat DTO decode.
//!
//! The compat surface is a frozen wire contract: field presence, naming and
//! nullability are locked (`deny_unknown_fields` on the request shapes), so
//! arbitrary JSON must decode to a typed accept or a typed reject — never a
//! panic, never a partial parse. Adversarial properties:
//!
//! - hostile JSON is panic-free across every public compat DTO;
//! - any accepted decode `x` satisfies `decode(encode(x)) == x` (a value must
//!   not mutate through its own serialization);
//! - the legacy `Handshake` line parser never panics and a parsed handshake
//!   round-trips through `to_line`/`from_line`.
//!
//! Seeds: the checked-in `compat/kilo-v756/` goldens (`crate::fixtures`)
//! are replayed by the deterministic test and by the fuzz corpus script.

use super::{json_roundtrip, Outcome};

/// Fuzz the compat DTO decoders.
pub fn harness_compat_dto_decode(bytes: &[u8]) -> Outcome {
    macro_rules! decode_all {
        ($($ty:ty),+ $(,)?) => {
            $(
                if let Err(evidence) = json_roundtrip::<$ty>(bytes) {
                    return Outcome::Violation(format!(
                        "{}: {evidence}",
                        stringify!($ty)
                    ));
                }
            )+
        };
    }

    decode_all!(
        faktor_protocol::v756::GlobalEvent,
        faktor_protocol::v756::GlobalEventPayload,
        faktor_protocol::v756::Part,
        faktor_protocol::v756::Message,
        faktor_protocol::v756::CreateSessionRequest,
        faktor_protocol::v756::CreateSessionResponse,
        faktor_protocol::v756::SdkPromptRequest,
        faktor_protocol::v756::SdkAbortRequest,
        faktor_protocol::v756::ProviderConfig,
        faktor_protocol::v756::ProviderList,
        faktor_protocol::v756::wire::SessionCreateRequest,
        faktor_protocol::v756::wire::SessionCreateResponse,
        faktor_protocol::v756::wire::SessionModel,
        faktor_protocol::v756::wire::MessageSendRequest,
        faktor_protocol::v756::wire::MessageSendResponse,
        faktor_protocol::v756::wire::WirePart,
        faktor_protocol::v756::wire::WireMessage,
        faktor_protocol::v756::wire::WireMessageEntry,
        faktor_protocol::v756::wire::DiffStatus,
        faktor_protocol::v756::wire::SessionUpdateRequest,
        faktor_protocol::v756::wire::SessionUpdateResponse,
        faktor_protocol::v756::wire::RevertResponse,
    );

    // Legacy handshake line: a prefix + JSON. Hostile text never panics; a
    // parsed handshake must survive its own canonical line.
    if let Ok(text) = std::str::from_utf8(bytes) {
        if let Some(handshake) = faktor_protocol::v756::Handshake::from_line(text) {
            match faktor_protocol::v756::Handshake::from_line(&handshake.to_line()) {
                Some(again) if again == handshake => {}
                other => {
                    return Outcome::Violation(format!(
                        "handshake line round-trip broke: {other:?}"
                    ))
                }
            }
        }
    }
    Outcome::Clean
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Lcg;

    /// Deterministic corpus: real compat fixture bytes (mutated) plus raw
    /// hostile JSON and truncations — 2000 iterations.
    #[test]
    fn seeded_pseudo_fuzz_compat_dto_decode_2000() {
        let seeds = crate::fixtures::fixture_seeds();
        assert!(!seeds.is_empty(), "compat/provider fixtures must exist");
        let mut lcg = Lcg::new(0x5EED_0000_0000_0007);
        let mut clean = 0;
        for i in 0..2000u64 {
            let mut bytes = Vec::new();
            match i % 6 {
                0 => {
                    // A real fixture, as-is.
                    let seed = &seeds[(lcg.next_u64() as usize) % seeds.len()];
                    bytes.clone_from(&seed.bytes);
                }
                1 => {
                    // A real fixture with deterministic byte flips.
                    let seed = &seeds[(lcg.next_u64() as usize) % seeds.len()];
                    bytes.clone_from(&seed.bytes);
                    if !bytes.is_empty() {
                        for _ in 0..=lcg.below(6) {
                            let pos = (lcg.next_u64() as usize) % bytes.len();
                            bytes[pos] ^= 1 << (lcg.below(8) as u8);
                        }
                    }
                }
                2 => {
                    // Truncated fixture (mid-scalar cuts are fine here: the
                    // decoder sees raw bytes).
                    let seed = &seeds[(lcg.next_u64() as usize) % seeds.len()];
                    let cut = if seed.bytes.is_empty() {
                        0
                    } else {
                        (lcg.next_u64() as usize) % seed.bytes.len()
                    };
                    bytes.extend_from_slice(&seed.bytes[..cut]);
                }
                3 => {
                    // Hostile JSON shapes.
                    let s = match lcg.below(5) {
                        0 => format!(
                            "{{\"type\":\"{}\",\"x\":[]}}",
                            "e".repeat(1 + lcg.below(20) as usize)
                        ),
                        1 => format!("[{}]", lcg.next_u64()),
                        2 => format!("{{\"sessionID\":null,\"parts\":[{}]}}", lcg.next_u64()),
                        3 => format!("{{\"directory\":{},\"payload\":null}}", lcg.next_u64()),
                        _ => format!("\"unterminated{}", "λ".repeat(1 + lcg.below(9) as usize)),
                    };
                    bytes.extend_from_slice(s.as_bytes());
                }
                4 => {
                    // Raw bytes (often invalid UTF-8 by construction).
                    let n = 1 + lcg.below(800) as usize;
                    for _ in 0..n {
                        bytes.push(lcg.next_u64() as u8);
                    }
                }
                _ => {
                    // Handshake-line shaped text.
                    let s = format!(
                        "FAKTOR_PLUS_HANDSHAKE {{\"version\":\"0.1.0\",\"protocol\":\"v756\",\"pid\":{},\"auth_token\":\"{}\",\"port\":{}}}",
                        lcg.next_u64() % 100000,
                        "t".repeat(1 + lcg.below(16) as usize),
                        lcg.below(65536)
                    );
                    bytes.extend_from_slice(s.as_bytes());
                }
            }
            match harness_compat_dto_decode(&bytes) {
                Outcome::Clean => clean += 1,
                v => panic!("iteration {i}: {v}"),
            }
        }
        assert!(clean > 0);
    }

    /// Every checked-in compat golden must exercise the harness panic-free.
    #[test]
    fn compat_fixture_corpus_is_clean() {
        let mut compat = 0;
        for seed in crate::fixtures::fixture_seeds() {
            if !seed.name.starts_with("compat/kilo-v756/") {
                continue;
            }
            compat += 1;
            match harness_compat_dto_decode(&seed.bytes) {
                Outcome::Clean => {}
                v => panic!("fixture {}: {v}", seed.name),
            }
        }
        assert!(compat >= 10, "expected the compat corpus, found {compat}");
    }

    #[test]
    fn compat_harness_is_deterministic() {
        let mut lcg = Lcg::new(0xC0DE);
        for _ in 0..50 {
            let n = 1 + lcg.below(600) as usize;
            let bytes: Vec<u8> = (0..n).map(|_| lcg.next_u64() as u8).collect();
            assert_eq!(
                harness_compat_dto_decode(&bytes),
                harness_compat_dto_decode(&bytes)
            );
        }
    }
}
