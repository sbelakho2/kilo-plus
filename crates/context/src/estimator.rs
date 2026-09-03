//! Token estimation. Conservative heuristic that the budget engine uses;
//! provider-specific tokenizers are exact, this is safe.
//!
//! Audit round 12: the old chars/4 factor was optimistic — real tokenizers
//! average ≈3.4 chars/token on code, so chars/4 under-budgeted text. The
//! default estimator now never rates text below the chars/3.4 floor:
//!
//! ```text
//! estimate(text) = max(chars/3 + 1, ceil(chars/3.4))     // chars > 0
//!                = 0                                      // empty
//! ```
//!
//! `chars/3 + 1` carries the +1 structural adder for the message/part
//! envelope on top of a strictly conservative chars-per-token ratio; the
//! `ceil(chars/3.4)` term is the documented lower bound kept explicit so
//! the factor can never silently drop below the 3.4 floor again. This is
//! the "safety factor" of the wire-plan token math (`wire_plan.rs` counts
//! every text run through this estimator); budget semantics otherwise
//! unchanged.

/// The audit-named conservative estimator. Implemented by the default
/// [`Estimator`]; keep the alias so reviews find the guaranteed-safe
/// profile without renaming the pervasive unit struct.
pub type GenericConservativeEstimator = Estimator;

/// Anything that can conservatively estimate the token cost of text.
pub trait TokenEstimator {
    /// A conservative token count for `text`; 0 only for empty input.
    fn estimate(&self, text: &str) -> usize;
}

pub struct Estimator;

/// ceil(a / b) for positive `b`, in integer math.
fn ceil_div(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

impl Estimator {
    /// Conservative estimate: `max(chars/3 + 1, ceil(chars/3.4))` — never
    /// below the chars/3.4 floor, plus a +1 structural envelope adder.
    /// Never returns 0 for non-empty input; never panics on any input.
    pub fn estimate_tokens(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        let chars = text.chars().count();
        let floor_34 = ceil_div(chars.saturating_mul(10), 34);
        let dense = chars.saturating_div(3).saturating_add(1);
        // For chars >= 1 the dense term dominates, but the max stays
        // explicit: it IS the chars/3.4 lower bound contract.
        core::cmp::max(dense, floor_34)
    }

    pub fn estimate_json(&self, value: &serde_json::Value) -> usize {
        match value {
            serde_json::Value::Null => 1,
            serde_json::Value::Bool(_) => 1,
            serde_json::Value::Number(n) => n.to_string().len().saturating_add(1) / 2 + 1,
            serde_json::Value::String(s) => self.estimate_tokens(s).max(1),
            serde_json::Value::Array(items) => items
                .iter()
                .map(|i| self.estimate_json(i))
                .sum::<usize>()
                .saturating_add(1),
            serde_json::Value::Object(map) => map
                .iter()
                .map(|(k, v)| {
                    self.estimate_tokens(k)
                        .saturating_add(self.estimate_json(v))
                })
                .sum::<usize>()
                .saturating_add(1),
        }
    }

    /// Clamp text to a token budget, never splitting UTF-8 and never
    /// exceeding the bound by more than the single trailing ellipsis
    /// character (an empty result is possible). Cuts at the estimator's
    /// own chars-per-token ratio (3 chars/token) so the clamped text's
    /// estimate cannot drift above `max_tokens + 1`.
    pub fn clamp_to(&self, text: &str, max_tokens: usize) -> String {
        if max_tokens == 0 {
            return String::new();
        }
        let max_chars = max_tokens.saturating_mul(3);
        if text.chars().count() <= max_chars {
            return text.to_string();
        }
        let mut end = max_chars;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        let mut out: String = text[..end].to_string();
        out.push('…');
        out
    }
}

impl TokenEstimator for Estimator {
    fn estimate(&self, text: &str) -> usize {
        self.estimate_tokens(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_whitespace() {
        assert_eq!(Estimator.estimate_tokens(""), 0);
        assert_eq!(Estimator.estimate(""), 0, "trait estimate agrees");
        assert!(Estimator.estimate_tokens("   \n\t ") > 0);
    }

    #[test]
    fn huge_string_is_finite() {
        let big = "x".repeat(8 << 20);
        let t = Estimator.estimate_tokens(&big);
        let chars = big.chars().count();
        let expected = chars.saturating_mul(10).div_ceil(34);
        assert_eq!(
            t,
            core::cmp::max(chars.saturating_div(3).saturating_add(1), expected),
            "exact formula lock"
        );
        assert!(t < big.len());
    }

    #[test]
    fn estimate_is_never_below_the_chars_34_floor() {
        // Audit round 12: the old chars/4 factor under-budgeted text (real
        // tokenizers average ~3.4 chars/token). The estimator must rate
        // BOTH an ASCII code sample and a heavy CJK sample at or above
        // ceil(chars/3.4) — the strict lower bound of the contract — and
        // must never report 0 for non-empty input.
        let e = Estimator;
        let ascii =
            "fn main() { let x = \"The quick brown fox jumps over the lazy dog\"; } ".repeat(64);
        let cjk =
            "汉字测试用例：上下文预算必须保守，不能低于每三点四个字符一个词的底线。".repeat(40);
        for sample in [ascii, cjk] {
            let chars = sample.chars().count();
            let floor = chars.saturating_mul(10).div_ceil(34);
            let got = e.estimate(&sample);
            assert!(got > 0, "non-empty input never estimates to 0");
            assert!(
                got >= floor,
                "estimate {got} dropped below the chars/3.4 floor {floor} \
                 (chars={chars})"
            );
            // Dense term dominates for chars >= 1: exact formula equality.
            assert_eq!(got, chars.saturating_div(3).saturating_add(1));
        }
        // The floor contract is strict on the boundary too: a 340-char run
        // must never be rated below 100 tokens.
        let text = "x".repeat(340);
        assert!(e.estimate(&text) >= 100);
    }

    #[test]
    fn generic_estimator_alias_and_trait_agree() {
        // TokenEstimator is the trait face; GenericConservativeEstimator is
        // the audit-named default — both must agree with estimate_tokens.
        let text = "some code sample ".repeat(50);
        let via_trait: &dyn TokenEstimator = &Estimator;
        assert_eq!(via_trait.estimate(&text), Estimator.estimate_tokens(&text));
        // The alias names the SAME type as the pervasive Estimator (no
        // drift possible between the audit name and the default estimator).
        let via_alias: &dyn TokenEstimator = &{
            let e: GenericConservativeEstimator = Estimator;
            e
        };
        assert_eq!(via_alias.estimate(&text), Estimator.estimate_tokens(&text));
    }

    #[test]
    fn deep_nested_json_is_finite() {
        let mut v = serde_json::Value::Null;
        for _ in 0..100 {
            v = serde_json::json!([v]);
        }
        let t = Estimator.estimate_json(&v);
        assert!(t > 0 && t < 10_000);
    }

    #[test]
    fn invalid_utf8_bytes_are_handled() {
        let bytes = vec![0xFF, 0xFE, 0x00, 0x80, 0xC3, 0x28];
        let s = String::from_utf8_lossy(&bytes);
        let t = Estimator.estimate_tokens(&s);
        assert!(t > 0);
    }

    #[test]
    fn clamp_respects_token_bound() {
        let e = Estimator;
        assert_eq!(e.clamp_to("", 100), "");
        assert_eq!(e.clamp_to("ab", 100), "ab");
        let text = "a".repeat(1000);
        let clamped = e.clamp_to(&text, 10);
        assert!(Estimator.estimate_tokens(&clamped) <= 10 + 1); // +1 for the ellipsis
        assert!(clamped.ends_with('…'));
        assert_eq!(e.clamp_to(&text, 0), "");
    }

    #[test]
    fn clamp_never_splits_multibyte() {
        let e = Estimator;
        // 100 emoji = 400 bytes but 100 chars.
        let text = "😀".repeat(100);
        let clamped = e.clamp_to(&text, 3);
        assert!(clamped.is_char_boundary(clamped.len()));
        assert!(clamped.chars().count() <= 12 + 1);
        // The cut lands mid-codepoint: is_char_boundary loop must fix it.
        let s = "aé😀b";
        let c = e.clamp_to(s, 1);
        assert!(c.is_char_boundary(c.len()));
    }

    #[test]
    fn json_estimate_never_underflows() {
        assert_eq!(Estimator.estimate_json(&serde_json::Value::Null), 1);
        assert_eq!(Estimator.estimate_json(&serde_json::json!(true)), 1);
        assert!(
            Estimator.estimate_json(&serde_json::json!({"a": [1,2,{"b":"x".repeat(400)}]})) > 0
        );
        // Absurdly deep array (5k levels) — recursion is bounded by serde
        // parse limits; construct programmatically at depth 1000.
        let mut v = serde_json::Value::Null;
        for _ in 0..1000 {
            v = serde_json::Value::Array(vec![v]);
        }
        assert!(Estimator.estimate_json(&v) > 0);
    }
}
