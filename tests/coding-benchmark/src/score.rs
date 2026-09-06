//! Deterministic criteria scoring.
//!
//! `criteria.md` is machine-parseable by construction:
//!
//! ```text
//! # comment lines are skipped
//! - crit-01: the repository test suite passes end to end
//! ```
//!
//! A criterion is `(key, text)` with `key = crit-NN` (two digits). The
//! benchmark records the assistant's final summary and the durable
//! verification record of the task, then scores deterministically:
//!
//! - `summary_hit`: the criterion key appears as its own token in the
//!   final summary text (the prompt instructs the model to report each
//!   criterion as `crit-NN: PASS/FAIL …`);
//! - `record_pass`: corroboration from the daemon's durable
//!   verification-record rows when they exist (`criterionKey` equal to the
//!   key or to the full `key: text` line, with `passed = true`).
//!
//! No LLM judging, no probabilistic heuristics: two runs of the checker on
//! the same inputs return the same verdicts.

use crate::corpus::{MAX_CRITERIA, MAX_META_BYTES};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Criterion {
    /// `crit-01`, `crit-02`, … (unique within one task).
    pub key: String,
    /// The single-line criterion text after the key.
    pub text: String,
}

impl Criterion {
    /// The full canonical line `crit-01: <text>` — the identity a durable
    /// verification record would carry if the daemon's acceptance
    /// criteria were seeded from `criteria.md`.
    pub fn canonical_line(&self) -> String {
        format!("{}: {}", self.key, self.text)
    }
}

/// Parse the deterministic criteria format. Adversarial inputs (unknown
/// keys, duplicate keys, non-comment junk, oversized anything) are
/// rejected with the offending detail.
pub fn parse_criteria(text: &str) -> Result<Vec<Criterion>, String> {
    if text.len() > MAX_META_BYTES as usize {
        return Err(format!("{MAX_META_BYTES}-byte cap exceeded"));
    }
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (idx, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let rest = line
            .strip_prefix('-')
            .ok_or_else(|| format!("line {}: expected `- crit-NN: <text>`", idx + 1))?
            .trim();
        let (key, text) = rest.split_once(':').ok_or_else(|| {
            format!(
                "line {}: expected `crit-NN: <text>`, found {rest:?}",
                idx + 1
            )
        })?;
        let key = key.trim();
        let text = text.trim();
        if !is_criterion_key(key) {
            return Err(format!("line {}: invalid criterion key {key:?}", idx + 1));
        }
        if text.is_empty() {
            return Err(format!("line {}: empty criterion text", idx + 1));
        }
        if text.len() > 512 {
            return Err(format!("line {}: criterion text over 512 bytes", idx + 1));
        }
        if !seen.insert(key.to_string()) {
            return Err(format!("duplicate criterion key {key:?}"));
        }
        out.push(Criterion {
            key: key.to_string(),
            text: text.to_string(),
        });
    }
    if out.is_empty() {
        return Err("no criteria found".into());
    }
    if out.len() > MAX_CRITERIA {
        return Err(format!("more than {MAX_CRITERIA} criteria"));
    }
    Ok(out)
}

fn is_criterion_key(key: &str) -> bool {
    let Some(rest) = key.strip_prefix("crit-") else {
        return false;
    };
    rest.len() == 2 && rest.bytes().all(|b| b.is_ascii_digit())
}

/// One scored criterion.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CriterionResult {
    pub key: String,
    /// The key appears as its own token in the final assistant summary.
    pub summary_hit: bool,
    /// A durable verification record corroborates the criterion (its key
    /// or canonical line appears with `passed = true`).
    pub record_pass: bool,
}

/// The durable-record evidence the real runner extracts from the daemon's
/// verification endpoint (`/native/session/{id}/tasks/{task_id}/
/// verification`), reduced to what the deterministic checker needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordCriteria {
    /// Record status text (`Passed`/`Failed`/… as reported by the wire).
    pub status: String,
    /// `(criterionKey, passed)` pairs of the record's criteria rows.
    pub criteria: Vec<(String, bool)>,
}

fn token_hit(text: &str, key: &str) -> bool {
    text.split_whitespace()
        .any(|word| word.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-') == key)
}

/// Score every criterion against the final summary text (deterministic).
pub fn score_summary(criteria: &[Criterion], summary: &str) -> Vec<CriterionResult> {
    criteria
        .iter()
        .map(|c| CriterionResult {
            key: c.key.clone(),
            summary_hit: token_hit(summary, &c.key),
            record_pass: false,
        })
        .collect()
}

/// Corroborate summary hits with the durable verification-record evidence.
pub fn corroborate_records(
    mut results: Vec<CriterionResult>,
    criteria: &[Criterion],
    records: &[RecordCriteria],
) -> Vec<CriterionResult> {
    let mut canon: std::collections::HashMap<String, Criterion> = criteria
        .iter()
        .map(|c| (c.canonical_line(), c.clone()))
        .collect();
    for c in criteria {
        canon.entry(c.key.clone()).or_insert_with(|| c.clone());
    }
    for record in records {
        for (key, passed) in &record.criteria {
            if !passed {
                continue;
            }
            let hit = canon.get(key).map(|c| c.key.clone());
            if let Some(k) = hit {
                if let Some(r) = results.iter_mut().find(|r| r.key == k) {
                    r.record_pass = true;
                }
            }
        }
    }
    results
}

/// Criteria met = the criteria whose key the summary names (the objective
/// verified-success signal stays with the repository's own `verify.sh`).
pub fn criteria_met(results: &[CriterionResult]) -> usize {
    results.iter().filter(|r| r.summary_hit).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_accepts_the_canonical_format_and_skips_comments() {
        let criteria = parse_criteria(
            "# header\ntext before is junk\n- crit-01: a\n\n- crit-02: b b b\n# trailing\n",
        )
        .unwrap_err();
        assert!(criteria.contains("line 2"), "{criteria}");

        let criteria =
            parse_criteria("# h\n- crit-01: suite passes\n- crit-02: behavior x\n").unwrap();
        assert_eq!(criteria.len(), 2);
        assert_eq!(criteria[0].canonical_line(), "crit-01: suite passes");
    }

    #[test]
    fn parser_rejects_hostile_and_sloppy_metadata() {
        for bad in [
            "",                                 // nothing
            "- x-01: text",                     // wrong prefix
            "- crit-1: text",                   // one digit
            "- crit-01:",                       // no text
            "- crit-01: a\n- crit-01: b",       // duplicate
            "- crit-01: a\njunk\n- crit-02: b", // junk line
            "- CRIT-01: a",                     // case
        ] {
            assert!(parse_criteria(bad).is_err(), "{bad:?} must be rejected");
        }
    }

    #[test]
    fn summary_scoring_is_token_exact_and_deterministic() {
        let criteria = parse_criteria("- crit-01: a\n- crit-02: b\n- crit-03: c\n").unwrap();
        let summary =
            "Done. crit-01: PASS — suite green. crit-03 FAIL see logs. crit-010 is unrelated.";
        let results = score_summary(&criteria, summary);
        assert!(results[0].summary_hit, "crit-01 named");
        assert!(!results[1].summary_hit, "crit-02 never named");
        assert!(results[2].summary_hit, "crit-03 named");
        assert_eq!(criteria_met(&results), 2);

        // Deterministic: same input, same verdicts.
        assert_eq!(score_summary(&criteria, summary), results);
    }

    #[test]
    fn records_corroborate_by_key_or_canonical_line() {
        let criteria = parse_criteria("- crit-01: suite passes\n- crit-02: order kept\n").unwrap();
        let results = score_summary(&criteria, "crit-02 reached");
        let records = vec![
            RecordCriteria {
                status: "Passed".into(),
                criteria: vec![("crit-02: order kept".into(), true)],
            },
            RecordCriteria {
                status: "Failed".into(),
                criteria: vec![("crit-01: suite passes".into(), false)],
            },
        ];
        let results = corroborate_records(results, &criteria, &records);
        assert!(
            !results[0].record_pass,
            "failed record rows never corroborate (crit-01 has passed=false)"
        );
        assert!(results[1].record_pass, "canonical-line hit corroborates");
    }
}
