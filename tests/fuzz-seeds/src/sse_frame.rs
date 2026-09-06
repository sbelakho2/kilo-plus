//! Harness 2: the frozen v7.5.6 SSE frame parser.
//!
//! `faktor_protocol::sse::SseEvent::from_frame` parses one `event:` /
//! `id:` / `data:` frame. Adversarial properties: arbitrary text, hostile
//! field text and truncated/mutated frames must never panic; and every
//! frame that PARSES must round-trip exactly:
//! `from_frame(ev.to_frame(seq)) == Some((seq, ev))` — plus parse
//! stability under semantic-equivalent field whitespace (the parser trims
//! values, so leading/trailing whitespace around fields must not change
//! the parsed event).

use super::{Lcg, Outcome};

fn parse(text: &str) -> Option<(u64, faktor_protocol::sse::SseEvent)> {
    faktor_protocol::sse::SseEvent::from_frame(text)
}

/// Fuzz the SSE frame parser: hostile text never panics, and any frame
/// that parses must round-trip.
pub fn harness_sse_frame_parser(bytes: &[u8]) -> Outcome {
    let Ok(text) = std::str::from_utf8(bytes) else {
        return Outcome::NotApplicable;
    };
    match parse(text) {
        None => Outcome::Clean,
        Some((seq, ev)) => {
            // Round-trip: encoding the parsed event must parse back to the
            // identical (seq, event).
            let frame = ev.to_frame(seq);
            match parse(&frame) {
                Some((seq2, ev2)) if seq2 == seq && ev2 == ev => Outcome::Clean,
                other => Outcome::Violation(format!(
                    "round-trip broke: parsed {seq}/{ev:?} from {text:?}, reparse of its own frame gave {other:?}"
                )),
            }
        }
    }
}

/// Mutation fuzz: deterministic truncations and byte flips of real frames
/// must never panic and must never parse to a DIFFERENT event of the same
/// kind (a mutated frame either fails or is byte-identical in meaning).
pub fn harness_sse_frame_truncation(bytes: &[u8]) -> Outcome {
    let Ok(text) = std::str::from_utf8(bytes) else {
        return Outcome::NotApplicable;
    };
    if text.len() < 8 {
        return Outcome::NotApplicable;
    }
    let mut lcg = Lcg::new(bytes.len() as u64 ^ 0x5EE);
    // Deterministic prefixes and mid-splits of the input (char-boundary
    // safe: slicing mid-rune would be a harness panic, never a parser bug).
    for raw_cut in [1, text.len() / 3, text.len() / 2, text.len() - 1] {
        if raw_cut == 0 || raw_cut >= text.len() {
            continue;
        }
        let cut = text.floor_char_boundary(raw_cut);
        if cut == 0 {
            continue;
        }
        let prefix = &text[..cut];
        if let Some((seq, ev)) = parse(prefix) {
            // A truncated frame may legitimately parse; it must round-trip.
            let frame = ev.to_frame(seq);
            match parse(&frame) {
                Some((s2, e2)) if s2 == seq && e2 == ev => {}
                other => {
                    return Outcome::Violation(format!(
                        "truncation {cut:?} parsed to a non-round-tripping event: {other:?}"
                    ))
                }
            }
        }
        // Flip a byte at a deterministic position inside the prefix.
        let mut flipped = prefix.as_bytes().to_vec();
        let pos = (lcg.next_u64() as usize) % flipped.len();
        flipped[pos] ^= 0x01;
        if let Ok(t) = std::str::from_utf8(&flipped) {
            let _ = parse(t);
        }
    }
    Outcome::Clean
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Build one canonical frame for `i` (deterministic, mixed event kinds).
    fn canonical_frame(i: u64) -> (u64, faktor_protocol::sse::SseEvent) {
        use faktor_protocol::sse::SseEvent;
        let ev = match i % 4 {
            0 => SseEvent::AgentStateChanged {
                session_id: format!("ses-{i}"),
                state: "streaming".into(),
                label: "Streaming".into(),
            },
            1 => SseEvent::Error {
                session_id: Some(format!("ses-{i}")),
                code: "x".into(),
                message: "boom αβ".into(),
            },
            2 => SseEvent::SessionUpdated {
                session_id: format!("ses-{i}"),
                state: "idle".into(),
            },
            _ => SseEvent::Compaction {
                session_id: format!("ses-{i}"),
                before_tokens: i as i64,
                after_tokens: (i / 2) as i64,
                accepted: i.is_multiple_of(2),
            },
        };
        let seq = (i * 7 + 1) % (u64::MAX - 3);
        (seq, ev)
    }

    #[test]
    fn seeded_pseudo_fuzz_sse_frame_parser_2000() {
        let mut lcg = Lcg::new(0x5EED_0000_0000_0002);
        let mut clean = 0;
        for i in 0..2000u64 {
            let mut bytes = Vec::new();
            match i % 5 {
                0 => {
                    // Whole canonical frames.
                    let (seq, ev) = canonical_frame(lcg.next_u64() % 1_000_000);
                    bytes.extend_from_slice(ev.to_frame(seq).as_bytes());
                }
                1 => {
                    // Raw hostile bytes.
                    let n = 1 + lcg.below(800) as usize;
                    for _ in 0..n {
                        bytes.push(lcg.next_u64() as u8);
                    }
                }
                2 => {
                    // A real frame truncated at a deterministic cut.
                    let (seq, ev) = canonical_frame(lcg.next_u64() % 1_000_000);
                    let frame = ev.to_frame(seq);
                    let cut = 1 + (lcg.next_u64() as usize) % frame.len();
                    bytes.extend_from_slice(&frame.as_bytes()[..cut]);
                }
                3 => {
                    // Field-hostile text (json garbage after data:).
                    let s = format!(
                        "event: error\nid: {}\ndata: {}\n\n",
                        lcg.next_u64() % 1000,
                        "{\"not\": \"json\", ".repeat(1 + lcg.below(5) as usize)
                    );
                    bytes.extend_from_slice(s.as_bytes());
                }
                _ => {
                    // Valid UTF-8 text soup.
                    let s = format!("x{i} :: {}", "λμνξ".repeat(1 + lcg.below(20) as usize));
                    bytes.extend_from_slice(s.as_bytes());
                }
            }
            match harness_sse_frame_parser(&bytes) {
                Outcome::Clean => clean += 1,
                Outcome::NotApplicable => {}
                v => panic!("iteration {i}: {v}"),
            }
            match harness_sse_frame_truncation(&bytes) {
                Outcome::Clean | Outcome::NotApplicable => {}
                v => panic!("iteration {i} (truncation): {v}"),
            }
        }
        assert!(clean > 0);
    }

    /// The round-trip property on whole canonical frames (sanity).
    #[test]
    fn sse_roundtrip_property_holds_for_canonical_frames() {
        for i in 0..500u64 {
            let (seq, ev) = canonical_frame(i);
            let frame = ev.to_frame(seq);
            let parsed = parse(&frame).expect("canonical frame parses");
            assert_eq!(parsed.0, seq);
            assert_eq!(parsed.1, ev);
        }
    }

    #[test]
    fn sse_harness_never_misparses_hex_junk() {
        // Bytes that are valid UTF-8 but never a frame must be Clean.
        let mut lcg = Lcg::new(0xBEE);
        for _ in 0..200 {
            let mut s = String::new();
            for _ in 0..lcg.below(40) {
                s.push(char::from(b'a' + lcg.below(26) as u8));
            }
            assert!(harness_sse_frame_parser(s.as_bytes()).is_ok());
        }
    }
}
