//! Harness 1: provider line framing (byte-split independence).
//!
//! `faktor-provider`'s `utf8_line_stream` assembles lines at the BYTE
//! level, so an SSE/NDJSON line or multibyte rune split across two network
//! chunks must never corrupt the stream. Adversarial property: for ANY
//! byte input, driving the REAL stream implementation with DIFFERENT
//! deterministic chunk splits must yield the IDENTICAL item stream — a
//! framing bug that is sensitive to chunk boundaries changes the observable
//! protocol. Strict UTF-8 rejection, the line cap and the final
//! unterminated line must all be split-independent too.

use futures::StreamExt;

use super::{Lcg, Outcome};

/// Deterministic split points of `bytes`: cut at `stride`-ish intervals
/// offset by `start`. Never returns empty cuts for a non-empty input.
fn splits(bytes: &[u8], start: u64, stride: u64) -> Vec<usize> {
    let stride = stride.max(1) as usize;
    let mut cuts = Vec::new();
    let mut pos = stride + (start as usize % 7);
    while pos < bytes.len() {
        cuts.push(pos);
        pos += stride;
    }
    cuts.push(bytes.len());
    cuts
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Drive the REAL `faktor_provider::transport::utf8_line_stream` over the
/// byte chunks cut at deterministic boundaries. Every emitted item is
/// rendered as `L:<hex>` (a decoded line) or `E:<err>` (a typed framing
/// rejection), so two runs are comparable byte-for-byte.
fn framing_items_real(bytes: &[u8], start: u64, stride: u64) -> Result<Vec<String>, String> {
    let cuts = splits(bytes, start, stride);
    let mut chunks = Vec::new();
    let mut prev = 0usize;
    for &c in &cuts {
        if c <= prev {
            return Err("degenerate split produced an empty-or-reversed chunk".into());
        }
        chunks.push(bytes[prev..c].to_vec());
        prev = c;
    }
    let stream = futures::stream::iter(chunks.into_iter().map(Ok::<_, std::convert::Infallible>));
    let typed = faktor_provider::transport::utf8_line_stream(stream, 4096);
    let mut items = Vec::new();
    let mut typed = Box::pin(typed);
    futures::executor::block_on(async {
        while let Some(item) = typed.next().await {
            match item {
                Ok(line) => items.push(format!("L:{}", hex(line.as_bytes()))),
                Err(e) => items.push(format!("E:{e}")),
            }
        }
    });
    Ok(items)
}

/// Fuzz the REAL provider line framing for byte-split independence and the
/// line cap. Deterministic: the two split strategies derive from the input
/// content itself (a small LCG keyed by the input length).
pub fn harness_provider_line_framing(bytes: &[u8]) -> Outcome {
    if bytes.is_empty() {
        return Outcome::Clean;
    }
    let mut lcg = Lcg::new(bytes.len() as u64 ^ 0xF0E);
    let a = (lcg.next_u64() % 5, 1 + lcg.next_u64() % 9);
    let b = (lcg.next_u64() % 5, 1 + lcg.next_u64() % 9);
    if a == b {
        return Outcome::NotApplicable;
    }
    let s1 = match framing_items_real(bytes, a.0, a.1) {
        Ok(s) => s,
        Err(v) => return Outcome::Violation(v),
    };
    let s2 = match framing_items_real(bytes, b.0, b.1) {
        Ok(s) => s,
        Err(v) => return Outcome::Violation(v),
    };
    if s1 != s2 {
        return Outcome::Violation(format!(
            "byte-split dependence in the real framing: split {a:?} emitted {:?} while split {b:?} emitted {:?}",
            s1, s2
        ));
    }
    // Every decoded line must respect the cap (4096 in the harness drive).
    for item in &s1 {
        if let Some(rest) = item.strip_prefix("L:") {
            if rest.len() / 2 > 4096 {
                return Outcome::Violation(format!(
                    "framing emitted a line of {} bytes over the 4096 byte cap",
                    rest.len() / 2
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

    /// Deterministic corpus: 2000 iterations mixing structured frames
    /// (multibyte text, CRLF, giant lines) with hostile raw bytes.
    #[test]
    fn seeded_pseudo_fuzz_line_framing_2000() {
        let mut lcg = Lcg::new(0x5EED_0000_0000_0001);
        let mut clean = 0;
        for i in 0..2000u64 {
            let mut bytes = Vec::new();
            match i % 6 {
                0 => {
                    // Structured NDJSON-ish lines with multibyte content.
                    let n = 1 + lcg.below(24);
                    for _ in 0..n {
                        let line = format!(
                            "{{\"seed\":{i},\"txt\":\"{}\"}}\n",
                            "αβγδ".repeat(1 + lcg.below(6) as usize)
                        );
                        bytes.extend_from_slice(line.as_bytes());
                    }
                }
                1 => {
                    // CRLF plus a final unterminated multibyte line.
                    let s = format!(
                        "data: x{i}\r\ndata: y\r\npartial-no-newline-{}",
                        "λ".repeat(1 + lcg.below(9) as usize)
                    );
                    bytes.extend_from_slice(s.as_bytes());
                }
                2 => {
                    // A single giant line (over-cap hostile input).
                    let s = format!("x{}", "a".repeat(3000 + lcg.below(3000) as usize));
                    bytes.extend_from_slice(s.as_bytes());
                }
                3 => {
                    // Raw random bytes (invalid UTF-8 likely).
                    let n = 1 + lcg.below(600) as usize;
                    for _ in 0..n {
                        bytes.push(lcg.next_u64() as u8);
                    }
                }
                _ => {
                    // Valid text with mixed ASCII/UTF-8 boundaries.
                    let n = 1 + lcg.below(40) as usize;
                    for _ in 0..n {
                        match lcg.below(3) {
                            0 => bytes.push(b'\n'),
                            1 => bytes.extend_from_slice("é".as_bytes()),
                            _ => bytes.push(b'a' + lcg.below(26) as u8),
                        }
                    }
                }
            }
            match harness_provider_line_framing(&bytes) {
                Outcome::Clean => clean += 1,
                Outcome::NotApplicable => {}
                v => panic!("iteration {i}: {v}"),
            }
        }
        assert!(clean > 0, "structured inputs must be exercised");
    }

    /// Determinism: same input, same outcome.
    #[test]
    fn line_framing_harness_is_deterministic() {
        let mut lcg = Lcg::new(0xD06);
        for _ in 0..50 {
            let n = 1 + lcg.below(500) as usize;
            let bytes: Vec<u8> = (0..n).map(|_| lcg.next_u64() as u8).collect();
            let a = harness_provider_line_framing(&bytes);
            let b = harness_provider_line_framing(&bytes);
            assert_eq!(a, b);
        }
    }
}
