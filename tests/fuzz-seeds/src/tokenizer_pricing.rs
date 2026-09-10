//! Harness: tokenizer identity mapping + pricing parse (P0-81).
//!
//! Two pure parsers are exercised together because both turn wire/config
//! strings into settlement-relevant identity:
//!
//! - `faktor_provider::tokenizer_for` (model string + deployment hint ->
//!   [`TokenizerId`]) must be total, deterministic and versioned — an
//!   unknown model conservatively maps to the generic estimator, never to a
//!   fake exact vocabulary;
//! - the persisted pricing shapes ([`PricingSnapshot`], the catalog's
//!   [`PricingState`]/[`ModelCatalogEntry`]) decode from hostile JSON as
//!   typed accept/reject, accepted values round-trip, and settlement math
//!   never fabricates a number: an `Unknown` authority always settles to
//!   `None`.

use faktor_core::model::{PriceAuthority, PricingSnapshot, TokenUsage};
use faktor_provider::catalog::{ModelCatalogEntry, PricingState};
use faktor_provider::{tokenizer_for, TokenizerId};

use super::{json_roundtrip, Lcg, Outcome};

/// Fuzz the tokenizer mapping and the pricing decoders.
pub fn harness_tokenizer_pricing_parse(bytes: &[u8]) -> Outcome {
    let text = String::from_utf8_lossy(bytes);

    // 1. Model → tokenizer mapping: total, deterministic, versioned.
    for hint in [None, Some("ollama"), Some("openai"), Some("llama.cpp")] {
        let id = tokenizer_for(&text, hint);
        if id.version == 0 {
            return Outcome::Violation(format!(
                "tokenizer_for returned a zero version for {text:?} (hint {hint:?})"
            ));
        }
        if id.to_string().is_empty() {
            return Outcome::Violation("TokenizerId displays as an empty string".into());
        }
        if tokenizer_for(&text, hint) != id {
            return Outcome::Violation(format!(
                "tokenizer_for is not deterministic for {text:?} (hint {hint:?})"
            ));
        }
    }

    // 2. Persisted pricing/tokenizer JSON: typed accept/reject + round-trip.
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
        TokenizerId,
        PricingSnapshot,
        PricingState,
        ModelCatalogEntry,
    );

    // 3. Settlement honesty on whatever decoded: Unknown never fabricates,
    //    and hostile magnitudes must not panic the saturating math.
    if let Ok(snapshot) = serde_json::from_slice::<PricingSnapshot>(bytes) {
        let mut lcg = Lcg::new(bytes.len() as u64 ^ 0x0009_C1CE);
        let usage = TokenUsage::new(
            lcg.next_u64(),
            lcg.next_u64(),
            lcg.next_u64(),
            lcg.next_u64(),
        );
        let cost = snapshot.settle_cost(
            usage.uncached_input_tokens,
            usage.cache_read_tokens,
            usage.cache_write_tokens,
            usage.output_tokens,
        );
        if snapshot.authority == PriceAuthority::Unknown && cost.is_some() {
            return Outcome::Violation(format!(
                "an Unknown-authority snapshot fabricated a settled cost: {cost:?}"
            ));
        }
        if cost.is_some() && snapshot.quote.is_none() {
            return Outcome::Violation("a quote-less snapshot fabricated a settled cost".into());
        }
    }
    Outcome::Clean
}

#[cfg(test)]
mod tests {
    use super::*;
    use faktor_core::model::{MicroUsdPerMillionTokens, PriceQuote};

    fn quote(input: u64, output: u64) -> PriceQuote {
        PriceQuote {
            input: MicroUsdPerMillionTokens::from_dollars_per_million(input),
            output: MicroUsdPerMillionTokens::from_dollars_per_million(output),
            cache_read: MicroUsdPerMillionTokens::ZERO,
            cache_write: MicroUsdPerMillionTokens::ZERO,
        }
    }

    /// Deterministic corpus: model-name space, real snapshot encodings and
    /// hostile pricing JSON — 2000 iterations.
    #[test]
    fn seeded_pseudo_fuzz_tokenizer_pricing_parse_2000() {
        let mut lcg = Lcg::new(0x5EED_0000_0000_0009);
        let mut clean = 0;
        for i in 0..2000u64 {
            let mut bytes = Vec::new();
            match i % 6 {
                0 => {
                    // Model strings across every documented prefix + suffix.
                    let base = [
                        "gpt-4o",
                        "gpt-4.1-mini",
                        "gpt-5",
                        "o1-preview",
                        "o3",
                        "gpt-4",
                        "gpt-3.5-turbo",
                        "claude-3-5-sonnet",
                        "gemini-2.0-flash",
                        "llama-3.3-70b",
                        "qwen2.5-coder",
                        "deepseek-r1",
                        "provider/",
                        "unknown-model",
                    ][lcg.below(14) as usize];
                    let s = format!(
                        "{base}{}",
                        if lcg.chance(40) {
                            format!("-{}", lcg.next_u64() % 1000)
                        } else {
                            String::new()
                        }
                    );
                    bytes.extend_from_slice(s.as_bytes());
                }
                1 => {
                    // Real encodings of every authority state.
                    let snap = match lcg.below(4) {
                        0 => PricingSnapshot::exact(
                            quote(5 + lcg.below(30), 10 + lcg.below(60)),
                            1 + lcg.below(9),
                            "test-source".into(),
                        ),
                        1 => PricingSnapshot::conservative_ceiling(
                            quote(50, 90),
                            1,
                            "ceiling".into(),
                        ),
                        2 => PricingSnapshot::local_zero(1, "ollama".into()),
                        _ => PricingSnapshot::unknown(0, "no-catalog".into()),
                    };
                    bytes.extend_from_slice(&serde_json::to_vec(&snap).unwrap());
                }
                2 => {
                    // Hostile numbers and shapes.
                    let s = match lcg.below(6) {
                        0 => format!(
                            "{{\"quote\":{{\"input\":{},\"output\":{},\"cache_read\":0,\"cache_write\":0}},\"authority\":\"exact\",\"epoch\":{},\"source_id\":\"s\"}}",
                            lcg.next_u64(),
                            lcg.next_u64(),
                            lcg.next_u64()
                        ),
                        1 => "{\"quote\":null,\"authority\":\"local_zero\",\"epoch\":0,\"source_id\":\"\"}".to_string(),
                        2 => format!("{{\"authority\":\"unknown\",\"epoch\":{}}}", lcg.next_u64()),
                        3 => format!(
                            "{{\"input\":{},\"output\":-1,\"cache_read\":1.5,\"cache_write\":[]}}",
                            lcg.next_u64()
                        ),
                        4 => "{\"quote\":{\"input\":18446744073709551615,\"output\":18446744073709551615,\"cache_read\":18446744073709551615,\"cache_write\":18446744073709551615}}".to_string(),
                        _ => format!(
                            "{{\"provider\":\"p\",\"model\":\"m\",\"capabilities\":{{}},\"pricing\":\"unknown\",\"quality_prior\":{{}},\"source_epoch\":{},\"provenance\":\"built_in\"}}",
                            lcg.next_u64()
                        ),
                    };
                    bytes.extend_from_slice(s.as_bytes());
                }
                3 => {
                    // Truncated real snapshot JSON at a deterministic cut.
                    let snap = PricingSnapshot::exact(quote(12, 34), 2, "cut".into());
                    let encoded = serde_json::to_vec(&snap).unwrap();
                    let cut = (lcg.next_u64() as usize) % encoded.len();
                    bytes.extend_from_slice(&encoded[..cut]);
                }
                4 => {
                    // Raw bytes.
                    let n = 1 + lcg.below(500) as usize;
                    for _ in 0..n {
                        bytes.push(lcg.next_u64() as u8);
                    }
                }
                _ => {
                    // Long/deep hostile text.
                    let s = format!(
                        "{}model-{}",
                        "x".repeat(1 + lcg.below(3000) as usize),
                        lcg.next_u64() % 100
                    );
                    bytes.extend_from_slice(s.as_bytes());
                }
            }
            match harness_tokenizer_pricing_parse(&bytes) {
                Outcome::Clean => clean += 1,
                v => panic!("iteration {i}: {v}"),
            }
        }
        assert!(clean > 0);
    }

    /// Documented mapping rows hold under suffixes and provider prefixes.
    #[test]
    fn tokenizer_mapping_rows_are_stable() {
        assert_eq!(
            tokenizer_for("openai/gpt-4o-2024", None),
            TokenizerId::O200K_BASE
        );
        assert_eq!(tokenizer_for("gpt-4-turbo", None), TokenizerId::CL100K_BASE);
        assert_eq!(
            tokenizer_for("anthropic/claude-3-opus", None),
            TokenizerId::ANTHROPIC
        );
        assert_eq!(
            tokenizer_for("deepseek-r1", Some("llama.cpp")),
            TokenizerId::LLAMA
        );
        assert_eq!(
            tokenizer_for("deepseek-r1", Some("deepseek")),
            TokenizerId::GENERIC_ESTIMATOR
        );
        assert_eq!(tokenizer_for("", None), TokenizerId::GENERIC_ESTIMATOR);
    }

    #[test]
    fn tokenizer_pricing_harness_is_deterministic() {
        let mut lcg = Lcg::new(0x81);
        for _ in 0..50 {
            let n = 1 + lcg.below(500) as usize;
            let bytes: Vec<u8> = (0..n).map(|_| lcg.next_u64() as u8).collect();
            assert_eq!(
                harness_tokenizer_pricing_parse(&bytes),
                harness_tokenizer_pricing_parse(&bytes)
            );
        }
    }
}
