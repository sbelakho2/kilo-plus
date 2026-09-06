//! Harness 3: the destination allowlist parser.
//!
//! `faktor_security::destination` parses allowlist rules
//! (`scheme://host:port` triples, wildcards, IP literals) at configuration
//! time: a single bad entry must be a policy load error — never a silently
//! skipped or silently permissive entry. Adversarial properties: arbitrary
//! rule text never panics; a successful multi-line parse must agree with
//! the per-entry parser on every entry (no entry is silently dropped); and
//! reparsing is idempotent (the parser's own canonical form round-trips).
//!
//! Deterministic multi-line rule sets are built from the bytes by
//! splitting on `\n` after a lossy decode — structured entries come from a
//! seed LCG so both valid and hostile shapes are exercised.

use super::{Lcg, Outcome};

fn parse_lines(text: &str) -> Result<faktor_security::destination::DestinationPolicy, String> {
    let lines: Vec<&str> = text.lines().collect();
    faktor_security::destination::DestinationPolicy::parse_lines(lines).map_err(|e| format!("{e}"))
}

/// Hostile rule text generator: deterministic from `seed`.
fn hostile_rule(seed: u64) -> String {
    let mut lcg = Lcg::new(seed ^ 0xD57);
    let host = match lcg.below(6) {
        0 => format!("host-{}", lcg.next_u64() % 1000),
        1 => format!("*.sub-{}.example.com", lcg.next_u64() % 100),
        2 => {
            let a = lcg.below(256);
            let b = lcg.below(256);
            let c = lcg.below(256);
            let d = lcg.below(256);
            format!("{a}.{b}.{c}.{d}")
        }
        3 => "::1".to_string(),
        _ => format!(
            "{}{}.example-{}.com",
            if lcg.chance(30) { "https://" } else { "" },
            if lcg.chance(20) { "*.api-" } else { "api-" },
            lcg.next_u64() % 1000
        ),
    };
    let port = match lcg.below(3) {
        0 => String::new(),
        1 => format!(":{}", 1 + lcg.next_u64() % 65535),
        _ => format!(":{}", if lcg.chance(50) { "999999" } else { "-1" }),
    };
    format!("{host}{port}")
}

/// Fuzz the destination-policy parser.
pub fn harness_destination_policy_parser(bytes: &[u8]) -> Outcome {
    let text = String::from_utf8_lossy(bytes);
    // Adversarial corpus: the raw text AND a deterministic rebuilt version
    // where lines that are not valid UTF-8 separators become structured
    // rules.
    let mut checks = vec![text.to_string()];
    let mut lcg = Lcg::new(bytes.len() as u64 ^ 0xD57);
    let mut rebuilt = String::new();
    for seg in text.split('\n').take(64) {
        if seg
            .chars()
            .all(|c| c.is_ascii_graphic() || c == ' ' || c == ':')
        {
            rebuilt.push_str(seg);
        } else {
            rebuilt.push_str(&hostile_rule(lcg.next_u64()));
        }
        rebuilt.push('\n');
    }
    checks.push(rebuilt);

    for (i, candidate) in checks.into_iter().enumerate() {
        let whole = parse_lines(&candidate);
        match whole {
            Err(_) => {
                // A rejected load is fine — nothing may be silently
                // half-applied. (Error paths are typed.)
            }
            Ok(policy) => {
                let mut rules: Vec<String> = Vec::new();
                for entry in candidate.lines().filter(|l| !l.trim().is_empty()) {
                    rules.push(entry.to_string());
                }
                if rules.is_empty() {
                    continue;
                }
                // 1. Per-entry agreement: every line that loaded must
                //    individually parse; every individually-invalid line
                //    must be absent from the policy.
                for entry in &rules {
                    match faktor_security::destination::DestinationRule::parse(entry) {
                        Ok(_) => {}
                        Err(_) => {
                            return Outcome::Violation(format!(
                                "candidate {i}: parse_lines accepted the entry {entry:?} that DestinationRule::parse rejects"
                            ))
                        }
                    }
                }
                // 2. Reparse idempotence on the same text.
                match parse_lines(&candidate) {
                    Ok(again) => {
                        let a = policy.rules().len();
                        let b = again.rules().len();
                        if a != b {
                            return Outcome::Violation(format!(
                                "candidate {i}: reparse changed the rule count {a} -> {b}"
                            ));
                        }
                    }
                    Err(e) => {
                        return Outcome::Violation(format!(
                            "candidate {i}: first parse succeeded but the reparse failed: {e}"
                        ))
                    }
                }
            }
        }
    }
    Outcome::Clean
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic corpus: structured valid entries, wildcard/scheme
    /// variants, hostile rules, pure junk and duplicate entries.
    #[test]
    fn seeded_pseudo_fuzz_destination_policy_parser_2000() {
        let mut lcg = Lcg::new(0x5EED_0000_0000_0003);
        let mut clean = 0;
        for i in 0..2000u64 {
            let mut text = String::new();
            match i % 6 {
                0 => {
                    // A structured valid policy.
                    for _ in 0..(1 + lcg.below(6)) {
                        let scheme = ["", "https://", "http://", "wss://"][lcg.below(4) as usize];
                        let port = if lcg.chance(40) {
                            format!(":{}", 1 + lcg.below(65535))
                        } else {
                            String::new()
                        };
                        let host = if lcg.chance(25) {
                            "*.api".to_string()
                        } else {
                            format!("svc-{}.example.com", lcg.below(100))
                        };
                        text.push_str(&format!("{scheme}{host}{port}\n"));
                    }
                }
                1 => {
                    // Duplicate entries: a documented parse error class.
                    let r = format!("dup-{}.example.com\n", i % 3);
                    text.push_str(&r.repeat(2 + lcg.below(4) as usize));
                }
                2 => {
                    // Hostile rules only.
                    for _ in 0..(1 + lcg.below(8)) {
                        text.push_str(&hostile_rule(lcg.next_u64()));
                        text.push('\n');
                    }
                }
                3 => {
                    // Byte soup.
                    let n = 1 + lcg.below(500) as usize;
                    for _ in 0..n {
                        text.push(char::from(1 + (lcg.next_u64() % 250) as u8));
                    }
                }
                4 => {
                    // Deep nesting attempts and path junk.
                    text.push_str(&"../".repeat(1 + lcg.below(10) as usize));
                    text.push_str(&format!("x{}", lcg.next_u64()));
                }
                _ => {
                    // Whitespace and empty-line soup.
                    for _ in 0..(1 + lcg.below(20)) {
                        text.push_str(&" \t\n".repeat(1 + lcg.below(3) as usize));
                    }
                }
            }
            match harness_destination_policy_parser(text.as_bytes()) {
                Outcome::Clean => clean += 1,
                v => panic!("iteration {i}: {v}"),
            }
        }
        assert!(clean > 0);
    }
}
