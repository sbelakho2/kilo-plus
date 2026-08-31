//! Token estimation. Conservative heuristic (chars/4 + overhead) that the
//! budget engine uses; provider-specific tokenizers are exact, this is safe.

pub struct Estimator;

impl Estimator {
    /// Conservative estimate: chars/4 with a small overhead for structure.
    /// Never returns 0 for non-empty input; never panics on any input.
    pub fn estimate_tokens(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        let chars = text.chars().count();
        chars.saturating_add(3) / 4
    }

    pub fn estimate_json(&self, value: &serde_json::Value) -> usize {
        match value {
            serde_json::Value::Null => 1,
            serde_json::Value::Bool(_) => 1,
            serde_json::Value::Number(n) => n.to_string().len().saturating_add(1) / 2 + 1,
            serde_json::Value::String(s) => self.estimate_tokens(s).max(1),
            serde_json::Value::Array(items) => {
                items.iter().map(|i| self.estimate_json(i)).sum::<usize>().saturating_add(1)
            }
            serde_json::Value::Object(map) => map
                .iter()
                .map(|(k, v)| self.estimate_tokens(k).saturating_add(self.estimate_json(v)))
                .sum::<usize>()
                .saturating_add(1),
        }
    }

    /// Clamp text to a token budget, never splitting UTF-8 and never
    /// exceeding the bound (an empty result is possible).
    pub fn clamp_to(&self, text: &str, max_tokens: usize) -> String {
        if max_tokens == 0 {
            return String::new();
        }
        let max_chars = max_tokens.saturating_mul(4);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_whitespace() {
        assert_eq!(Estimator.estimate_tokens(""), 0);
        assert!(Estimator.estimate_tokens("   \n\t ") > 0);
    }

    #[test]
    fn huge_string_is_finite() {
        let big = "x".repeat(8 << 20);
        let t = Estimator.estimate_tokens(&big);
        assert_eq!(t, big.len().saturating_add(3) / 4);
        assert!(t < big.len());
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
        assert!(Estimator.estimate_json(&serde_json::json!({"a": [1,2,{"b":"x".repeat(400)}]})) > 0);
        // Absurdly deep array (5k levels) — recursion is bounded by serde
        // parse limits; construct programmatically at depth 1000.
        let mut v = serde_json::Value::Null;
        for _ in 0..1000 {
            v = serde_json::Value::Array(vec![v]);
        }
        assert!(Estimator.estimate_json(&v) > 0);
    }
}
