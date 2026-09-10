//! Bounded typed normalization of raw tool/process output.
//!
//! Each normalizer exposes `try_from_text` and returns
//! `Result<_, EvidenceError>`; nothing here performs I/O, and no input can
//! panic a normalizer. Inputs and outputs are bounded:
//!
//! - [`MAX_INPUT_LINES`]: more raw input lines than this is rejected up front
//!   with [`EvidenceError::Oversized`]. Blank rows count toward the cap and
//!   are ignored afterwards.
//! - [`MAX_FIELD_CHARS`]: every string field written into the output is
//!   capped at this many Unicode scalar values. Truncation is never silent:
//!   the owning value reports `field_truncated = true`.
//! - Whitespace-only input yields an empty value (`Ok`), never an error.
//!
//! Row policies are documented per normalizer and are deliberately
//! asymmetric:
//!
//! - [`DiagnosticSet`] and [`SearchResults`] are **strict**: every non-blank
//!   row must match the documented grammar. A single row that fails to parse
//!   makes the whole input [`EvidenceError::Malformed`] — there is no "skip
//!   the bad row" mode, because a silently dropped row can hide a real
//!   failure.
//! - [`TestReport`] recognizes summary `key=value` tokens and `FAIL ...`
//!   rows; unrecognized rows are ignored (test runner output is noisy), but
//!   a recognized key with an unparsable value is
//!   [`EvidenceError::Malformed`].
//! - [`ProcessLogSummary`] accepts arbitrary log text; no row is an error.
//!
//! Output is deterministic: no hash-map iteration, no clocks, and
//! repeated-log fingerprints are stable blake3 prefixes.

use crate::types::{EvidenceError, Severity};
use serde::{Deserialize, Serialize};

/// Hard cap on raw input lines accepted by any `try_from_text`. Exceeding it
/// is [`EvidenceError::Oversized`], never silent truncation.
pub const MAX_INPUT_LINES: usize = 500_000;

/// Hard cap on Unicode scalar values kept per output string field. Exceeding
/// it truncates the field and sets `field_truncated = true` on the owning
/// value.
pub const MAX_FIELD_CHARS: usize = 4096;

/// Maximum number of keyword-prioritized lines kept by
/// [`ProcessLogSummary`]. The final log line always occupies one slot.
const MAX_IMPORTANT_LINES: usize = 256;

/// Characters of hostile input embedded in an error message before eliding.
const ERROR_PREVIEW_CHARS: usize = 120;

/// One parsed compiler/linter diagnostic row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticEvidence {
    /// Source path, whitespace-trimmed; may contain `:` (e.g. `C:\...`).
    pub path: String,
    /// 1-based line number.
    pub line: u32,
    /// 1-based column number.
    pub column: u32,
    /// Parsed severity; defaults to [`Severity::Error`] when the row has no
    /// recognizable severity token.
    pub severity: Severity,
    /// Rule/error code when written as `severity[code]:`, otherwise `None`.
    pub code: Option<String>,
    /// Human-readable message.
    pub message: String,
}

/// A set of diagnostics normalized from compiler/linter output.
///
/// # Grammar
///
/// Each non-blank row must be `path:line:column: rest`. The location is found
/// by scanning for the first `:<digits>:<digits>:` group, so Windows drive
/// letters and colons inside paths are tolerated. `rest` is interpreted as
/// `severity: message` or `severity[code]: message` when it begins with the
/// case-insensitive word `error`, `warning` or `info`; otherwise `rest` is the
/// message and the severity defaults to [`Severity::Error`]. `line` and
/// `column` must parse as `u32`.
///
/// # Failure rule (typed)
///
/// **A row that fails to parse makes the whole input
/// [`EvidenceError::Malformed`].** Rows are never silently dropped: a skipped
/// diagnostic can hide a real failure, so the caller must fix or strip the
/// offending row before retrying.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticSet {
    /// Parsed rows, in input order.
    pub diagnostics: Vec<DiagnosticEvidence>,
    /// True when any string field was truncated at [`MAX_FIELD_CHARS`].
    pub field_truncated: bool,
}

impl DiagnosticSet {
    /// Parses `text` into a diagnostic set. See the type docs for the row
    /// grammar and the strict failure rule.
    pub fn try_from_text(text: &str) -> Result<Self, EvidenceError> {
        let lines = bounded_lines(text)?;
        let mut diagnostics = Vec::new();
        let mut field_truncated = false;
        for (index, raw) in lines.iter().enumerate() {
            if raw.trim().is_empty() {
                continue;
            }
            let Some((path, line, column, rest)) = split_location(raw) else {
                return Err(malformed_row("diagnostic_set", index + 1, raw));
            };
            let (severity, code, message) = split_severity(rest);
            if path.trim().is_empty() || message.trim().is_empty() {
                return Err(malformed_row("diagnostic_set", index + 1, raw));
            }
            diagnostics.push(DiagnosticEvidence {
                path: cap_field(path.trim(), &mut field_truncated),
                line,
                column,
                severity,
                code: code.map(|code| cap_field(&code, &mut field_truncated)),
                message: cap_field(message, &mut field_truncated),
            });
        }
        Ok(Self {
            diagnostics,
            field_truncated,
        })
    }
}

/// One failed test case.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TestFailure {
    /// Test name.
    pub name: String,
    /// Failure message (empty when the runner only reported a name).
    pub message: String,
}

/// A test-run report normalized from runner output.
///
/// # Grammar
///
/// - Summary rows are whitespace/comma separated `key=value` tokens with keys
///   `total`, `passed`, `skipped` and `duration_ms` (`u32`/`u64`); the last
///   occurrence of a key wins. When `total` is absent it is derived as
///   `passed + skipped + failed.len()`.
/// - Failure rows start with a case-insensitive `fail ` or `failed ` prefix,
///   followed by `name: message` split at the first `": "`; without that
///   separator the whole remainder is the name and the message is empty.
/// - Unknown rows are ignored, but a recognized key with a value that does
///   not parse as a number is [`EvidenceError::Malformed`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TestReport {
    /// Total tests; explicit `total=` when present, otherwise derived.
    pub total: u32,
    /// Passing test count from `passed=`.
    pub passed: u32,
    /// Failed tests, in input order.
    pub failed: Vec<TestFailure>,
    /// Skipped test count from `skipped=`.
    pub skipped: u32,
    /// Duration from `duration_ms=`.
    pub duration_ms: u64,
    /// True when any string field was truncated at [`MAX_FIELD_CHARS`].
    pub field_truncated: bool,
}

impl TestReport {
    /// Parses `text` into a test report. See the type docs for the grammar.
    pub fn try_from_text(text: &str) -> Result<Self, EvidenceError> {
        let lines = bounded_lines(text)?;
        let mut explicit_total: Option<u32> = None;
        let mut passed = 0u32;
        let mut skipped = 0u32;
        let mut duration_ms = 0u64;
        let mut failed = Vec::new();
        let mut field_truncated = false;
        for (index, raw) in lines.iter().enumerate() {
            let line = raw.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(rest) =
                strip_prefix_ci(line, "fail ").or_else(|| strip_prefix_ci(line, "failed "))
            {
                let (name, message) = split_failure(rest);
                if name.is_empty() {
                    return Err(malformed_row("test_report", index + 1, raw));
                }
                failed.push(TestFailure {
                    name: cap_field(name, &mut field_truncated),
                    message: cap_field(message, &mut field_truncated),
                });
                continue;
            }
            for token in line.split(|c: char| c == ',' || c.is_whitespace()) {
                let Some((key, value)) = token.split_once('=') else {
                    continue;
                };
                match key {
                    "total" => {
                        explicit_total =
                            Some(parse_number::<u32>("test_report", index + 1, key, value)?);
                    }
                    "passed" => {
                        passed = parse_number::<u32>("test_report", index + 1, key, value)?;
                    }
                    "skipped" => {
                        skipped = parse_number::<u32>("test_report", index + 1, key, value)?;
                    }
                    "duration_ms" => {
                        duration_ms = parse_number::<u64>("test_report", index + 1, key, value)?;
                    }
                    _ => {}
                }
            }
        }
        let total = explicit_total.unwrap_or_else(|| {
            passed
                .saturating_add(skipped)
                .saturating_add(failed.len() as u32)
        });
        Ok(Self {
            total,
            passed,
            failed,
            skipped,
            duration_ms,
            field_truncated,
        })
    }
}

/// One search hit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchHit {
    /// Source path, whitespace-trimmed; may contain `:` (e.g. `C:\...`).
    pub path: String,
    /// 1-based line number.
    pub line: u32,
    /// Matched line text, verbatim after `path:line:`.
    pub text: String,
}

/// Search hits normalized from grep-style output.
///
/// # Grammar
///
/// Every non-blank row must be `path:line:text`, where the location is the
/// first `:<digits>:` group (so colons and Windows drive letters inside paths
/// are tolerated). `text` may be empty. A row that fails to parse makes the
/// whole input [`EvidenceError::Malformed`], matching the strict
/// [`DiagnosticSet`] policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchResults {
    /// Parsed hits, in input order.
    pub hits: Vec<SearchHit>,
    /// True when any string field was truncated at [`MAX_FIELD_CHARS`].
    pub field_truncated: bool,
}

impl SearchResults {
    /// Parses `text` into search results. See the type docs for the grammar.
    pub fn try_from_text(text: &str) -> Result<Self, EvidenceError> {
        let lines = bounded_lines(text)?;
        let mut hits = Vec::new();
        let mut field_truncated = false;
        for (index, raw) in lines.iter().enumerate() {
            if raw.trim().is_empty() {
                continue;
            }
            let Some((path, line, text)) = split_search_hit(raw) else {
                return Err(malformed_row("search_results", index + 1, raw));
            };
            if path.trim().is_empty() {
                return Err(malformed_row("search_results", index + 1, raw));
            }
            hits.push(SearchHit {
                path: cap_field(path.trim(), &mut field_truncated),
                line,
                text: cap_field(text, &mut field_truncated),
            });
        }
        Ok(Self {
            hits,
            field_truncated,
        })
    }
}

/// One retained log line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogEvent {
    /// Whitespace-trimmed line text, capped at [`MAX_FIELD_CHARS`].
    pub text: String,
}

/// A run of consecutive identical (trimmed) log lines.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepeatedLogGroup {
    /// First 8 bytes of `blake3(trimmed line)`, big-endian, as `u64`.
    /// Stable across processes and platforms.
    pub fingerprint: u64,
    /// Number of consecutive occurrences (always `>= 2`; singletons are not
    /// groups).
    pub count: u64,
    /// The trimmed line, capped at [`MAX_FIELD_CHARS`].
    pub sample: String,
}

/// A bounded summary of a process/tool log.
///
/// # Builder rules
///
/// - Whitespace-only rows are dropped before analysis.
/// - `exit` is taken from the first row that states one, in any of the
///   case-insensitive shapes `exit code: N`, `exit code=N`, `exit_code=N`,
///   `exit status: N`, `exit: N`, `exit N`, `exited with code N`,
///   `process exited with status N`; later exit rows are ignored.
/// - `repeated_groups` collapses runs of consecutive rows with identical
///   trimmed text; only runs of length `>= 2` become groups, in first-run
///   order. `fingerprint = u64::from_be_bytes(blake3(trimmed)[0..8])`.
/// - `important_lines` keeps up to [`MAX_IMPORTANT_LINES`] rows: rows
///   containing `error`, `fail`, `panic` or `assert` (ASCII case-insensitive)
///   in input order, plus the final non-blank row, which is always present.
///   Selection is deterministic and independent of timing.
/// - Arbitrary log text is never an error; there is no row grammar to break.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessLogSummary {
    /// Reported process exit code, when a row states one.
    pub exit: Option<i32>,
    /// Keyword-prioritized lines plus the final line, capped at 256 entries.
    pub important_lines: Vec<LogEvent>,
    /// Consecutive-repetition groups, in first-run order.
    pub repeated_groups: Vec<RepeatedLogGroup>,
    /// True when any string field was truncated at [`MAX_FIELD_CHARS`].
    pub field_truncated: bool,
}

impl ProcessLogSummary {
    /// Builds a bounded summary of arbitrary log `text`. See the type docs
    /// for the builder rules.
    pub fn try_from_text(text: &str) -> Result<Self, EvidenceError> {
        let lines = bounded_lines(text)?;
        let entries: Vec<&str> = lines
            .iter()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect();
        let mut field_truncated = false;
        let mut exit = None;
        for line in &entries {
            if let Some(code) = parse_exit_code(line) {
                exit = Some(code);
                break;
            }
        }
        let important_lines = select_important_lines(&entries, &mut field_truncated);
        let repeated_groups = group_repeats(&entries, &mut field_truncated);
        Ok(Self {
            exit,
            important_lines,
            repeated_groups,
            field_truncated,
        })
    }
}

/// Rejects inputs above [`MAX_INPUT_LINES`] raw lines before any parsing.
fn bounded_lines(text: &str) -> Result<Vec<&str>, EvidenceError> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() > MAX_INPUT_LINES {
        return Err(EvidenceError::Oversized {
            max: MAX_INPUT_LINES,
            actual: lines.len(),
        });
    }
    Ok(lines)
}

/// Copies `raw` into the output, stopping one char past [`MAX_FIELD_CHARS`] and
/// flagging the truncation. The single-pass loop is O(cap), even for a
/// multi-megabyte single-line field.
fn cap_field(raw: &str, truncated: &mut bool) -> String {
    let mut out = String::new();
    for (index, ch) in raw.chars().enumerate() {
        if index == MAX_FIELD_CHARS {
            *truncated = true;
            break;
        }
        out.push(ch);
    }
    out
}

/// Typed row error carrying a bounded preview of the offending input.
fn malformed_row(kind: &str, line_no: usize, raw: &str) -> EvidenceError {
    let preview: String = raw.trim().chars().take(ERROR_PREVIEW_CHARS).collect();
    EvidenceError::Malformed(format!("{kind}: line {line_no}: {preview:?}"))
}

/// Typed numeric parse error for a recognized `key=value` token.
fn parse_number<T: std::str::FromStr>(
    kind: &str,
    line_no: usize,
    key: &str,
    value: &str,
) -> Result<T, EvidenceError> {
    value.parse::<T>().map_err(|_| {
        let preview: String = value.chars().take(ERROR_PREVIEW_CHARS).collect();
        EvidenceError::Malformed(format!(
            "{kind}: line {line_no}: {key}={preview:?} is not a valid number"
        ))
    })
}

/// Scans for the first `:<digits>:<digits>:` group. Returns the path before it,
/// the two numbers and the remainder after it. Colons inside the path (Windows
/// drive letters included) are tolerated; a location that does not start at
/// byte 0 or does not parse as `u32` yields `None`.
fn split_location(line: &str) -> Option<(&str, u32, u32, &str)> {
    let bytes = line.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b':' && index > 0 {
            let line_start = index + 1;
            let mut cursor = line_start;
            while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
                cursor += 1;
            }
            if cursor > line_start && cursor < bytes.len() && bytes[cursor] == b':' {
                let column_start = cursor + 1;
                let mut column_end = column_start;
                while column_end < bytes.len() && bytes[column_end].is_ascii_digit() {
                    column_end += 1;
                }
                if column_end > column_start
                    && column_end < bytes.len()
                    && bytes[column_end] == b':'
                {
                    if let (Ok(line_no), Ok(column)) = (
                        line[line_start..cursor].parse::<u32>(),
                        line[column_start..column_end].parse::<u32>(),
                    ) {
                        return Some((&line[..index], line_no, column, &line[column_end + 1..]));
                    }
                }
            }
        }
        index += 1;
    }
    None
}

/// Splits the `rest` after `path:line:column:` into severity, optional code and
/// message. Only the exact tokens `error`, `warning` and `info` (ASCII
/// case-insensitive, optionally followed by `[code]`) are severity markers;
/// anything else is message text and the severity defaults to `Error`.
fn split_severity(rest: &str) -> (Severity, Option<String>, &str) {
    let body = rest.strip_prefix(' ').unwrap_or(rest);
    if let Some(colon) = body.find(':') {
        let head = &body[..colon];
        let (word, code) = match head.find('[') {
            Some(open) if head.ends_with(']') => {
                let inner = &head[open + 1..head.len() - 1];
                (
                    &head[..open],
                    if inner.is_empty() {
                        None
                    } else {
                        Some(inner.to_string())
                    },
                )
            }
            _ => (head, None),
        };
        if let Some(severity) = severity_word(word) {
            let tail = &body[colon + 1..];
            let message = tail.strip_prefix(' ').unwrap_or(tail);
            return (severity, code, message);
        }
    }
    (Severity::Error, None, body)
}

fn severity_word(word: &str) -> Option<Severity> {
    if word.eq_ignore_ascii_case("error") {
        Some(Severity::Error)
    } else if word.eq_ignore_ascii_case("warning") {
        Some(Severity::Warning)
    } else if word.eq_ignore_ascii_case("info") {
        Some(Severity::Info)
    } else {
        None
    }
}

/// Scans for the first `:<digits>:` group and returns path, line and the
/// verbatim text after it.
fn split_search_hit(line: &str) -> Option<(&str, u32, &str)> {
    let bytes = line.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b':' && index > 0 {
            let number_start = index + 1;
            let mut cursor = number_start;
            while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
                cursor += 1;
            }
            if cursor > number_start && cursor < bytes.len() && bytes[cursor] == b':' {
                if let Ok(number) = line[number_start..cursor].parse::<u32>() {
                    return Some((&line[..index], number, &line[cursor + 1..]));
                }
            }
        }
        index += 1;
    }
    None
}

/// Splits a failure row at the first `": "`; without it the whole remainder is
/// the name and the message is empty.
fn split_failure(rest: &str) -> (&str, &str) {
    match rest.find(": ") {
        Some(index) => (rest[..index].trim(), &rest[index + 2..]),
        None => (rest.trim(), ""),
    }
}

/// Case-insensitive ASCII prefix strip. The prefix is ASCII, so the returned
/// slice is always on a char boundary.
fn strip_prefix_ci<'a>(haystack: &'a str, prefix_lower: &str) -> Option<&'a str> {
    let prefix = prefix_lower.as_bytes();
    if haystack.len() >= prefix.len()
        && haystack.as_bytes()[..prefix.len()].eq_ignore_ascii_case(prefix)
    {
        Some(&haystack[prefix.len()..])
    } else {
        None
    }
}

/// Case-insensitive ASCII substring test. Multibyte continuation bytes are
/// `>= 0x80`, so an ASCII needle can never match inside a multibyte char.
fn contains_ci(haystack: &str, needle_lower: &str) -> bool {
    let needle = needle_lower.as_bytes();
    if needle.is_empty() {
        return true;
    }
    let haystack = haystack.as_bytes();
    haystack.len() >= needle.len()
        && haystack
            .windows(needle.len())
            .any(|w| w.eq_ignore_ascii_case(needle))
}

fn is_important_line(line: &str) -> bool {
    ["error", "fail", "panic", "assert"]
        .iter()
        .any(|needle| contains_ci(line, needle))
}

/// Keyword-prioritized selection, capped at [`MAX_IMPORTANT_LINES`]. The scan
/// reserves one slot so the final row is always present, even when keyword
/// rows overflow the cap.
fn select_important_lines(entries: &[&str], field_truncated: &mut bool) -> Vec<LogEvent> {
    let mut important = Vec::new();
    if entries.is_empty() {
        return important;
    }
    let last_index = entries.len() - 1;
    let mut last_captured = false;
    for (index, line) in entries.iter().enumerate() {
        if important.len() >= MAX_IMPORTANT_LINES - 1 {
            break;
        }
        if is_important_line(line) {
            if index == last_index {
                last_captured = true;
            }
            important.push(LogEvent {
                text: cap_field(line, field_truncated),
            });
        }
    }
    if !last_captured {
        important.push(LogEvent {
            text: cap_field(entries[last_index], field_truncated),
        });
    }
    important
}

/// Collapses consecutive runs of identical trimmed lines. Only runs of length
/// `>= 2` are reported, in first-run order.
fn group_repeats(entries: &[&str], field_truncated: &mut bool) -> Vec<RepeatedLogGroup> {
    let mut groups = Vec::new();
    let mut index = 0;
    while index < entries.len() {
        let sample = entries[index];
        let mut count = 1u64;
        while index + (count as usize) < entries.len()
            && entries[index + (count as usize)] == sample
        {
            count += 1;
        }
        if count > 1 {
            groups.push(RepeatedLogGroup {
                fingerprint: line_fingerprint(sample),
                count,
                sample: cap_field(sample, field_truncated),
            });
        }
        index += count as usize;
    }
    groups
}

/// Stable repetition fingerprint: first 8 bytes of `blake3(trimmed line)`,
/// big-endian, as `u64`.
fn line_fingerprint(trimmed: &str) -> u64 {
    let hash = blake3::hash(trimmed.as_bytes());
    let bytes = hash.as_bytes();
    u64::from_be_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

/// First row that states an exit code wins; later exit rows are ignored.
/// Recognized (case-insensitive) shapes include `exit code: N`, `exit code=N`,
/// `exit_code=N`, `exit status: N`, `exit: N`, `exit N`,
/// `exited with code N` and `process exited with status N`.
fn parse_exit_code(line: &str) -> Option<i32> {
    let lower = line.to_ascii_lowercase();
    let mut search_from = 0;
    while let Some(offset) = lower[search_from..].find("exit") {
        let index = search_from + offset;
        let at_word_start = index == 0 || !line.as_bytes()[index - 1].is_ascii_alphanumeric();
        if at_word_start {
            if let Some(code) = exit_code_after(&line[index + 4..]) {
                return Some(code);
            }
        }
        search_from = index + 4;
    }
    None
}

fn exit_code_after(rest: &str) -> Option<i32> {
    let rest = strip_prefix_ci(rest, "ed with code")
        .or_else(|| strip_prefix_ci(rest, "ed with status"))
        .or_else(|| strip_prefix_ci(rest, " code"))
        .or_else(|| strip_prefix_ci(rest, " status"))
        .or_else(|| strip_prefix_ci(rest, "_code"))
        .or_else(|| strip_prefix_ci(rest, "_status"))
        .unwrap_or(rest);
    let token = rest
        .trim_start_matches(|c: char| c == ':' || c == '=' || c.is_whitespace())
        .split_whitespace()
        .next()?;
    let token = token.trim_start_matches('+');
    let token = token.trim_end_matches(|c: char| !c.is_ascii_digit());
    token.parse::<i32>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expected_fingerprint(trimmed: &str) -> u64 {
        let hash = blake3::hash(trimmed.as_bytes());
        let b = hash.as_bytes();
        u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
    }

    #[test]
    fn diagnostics_parse_both_forms_and_skip_blank_rows() {
        let raw = "\n  \nsrc/main.rs:10:5: error[E0308]: mismatched types\n\
                   lib.rs:3:1: warning: unused variable\n\
                   build.rs:1:2: something went wrong\n\
                   \t\n";
        let set = DiagnosticSet::try_from_text(raw).unwrap();
        assert!(!set.field_truncated);
        assert_eq!(
            set.diagnostics,
            vec![
                DiagnosticEvidence {
                    path: "src/main.rs".to_string(),
                    line: 10,
                    column: 5,
                    severity: Severity::Error,
                    code: Some("E0308".to_string()),
                    message: "mismatched types".to_string(),
                },
                DiagnosticEvidence {
                    path: "lib.rs".to_string(),
                    line: 3,
                    column: 1,
                    severity: Severity::Warning,
                    code: None,
                    message: "unused variable".to_string(),
                },
                DiagnosticEvidence {
                    path: "build.rs".to_string(),
                    line: 1,
                    column: 2,
                    severity: Severity::Error,
                    code: None,
                    message: "something went wrong".to_string(),
                },
            ]
        );
    }

    #[test]
    fn diagnostics_accept_case_insensitive_severity_and_coloned_paths() {
        let raw = "UPPER.rs:3:4: ERROR: nope\n\
                   C:\\work\\src\\main.rs:7:9: info[HINT]: check this\n\
                   /tmp/a:b:c.rs:2:4: WARNING: odd path\n";
        let set = DiagnosticSet::try_from_text(raw).unwrap();
        assert_eq!(set.diagnostics.len(), 3);
        assert_eq!(set.diagnostics[0].path, "UPPER.rs");
        assert_eq!(set.diagnostics[0].severity, Severity::Error);
        assert_eq!(set.diagnostics[0].message, "nope");
        assert_eq!(set.diagnostics[1].path, r"C:\work\src\main.rs");
        assert_eq!(set.diagnostics[1].severity, Severity::Info);
        assert_eq!(set.diagnostics[1].code.as_deref(), Some("HINT"));
        assert_eq!(set.diagnostics[2].path, "/tmp/a:b:c.rs");
        assert_eq!(set.diagnostics[2].severity, Severity::Warning);
    }

    #[test]
    fn diagnostics_malformed_row_fails_whole_input_typed() {
        let hostile = [
            "just some text without a location",
            "src/main.rs:x:5: error: bad line number",
            "src/main.rs:1:y: error: bad column",
            "src/main.rs:4294967296:5: error: line overflow",
            "src/main.rs:1:4294967296: error: column overflow",
            "src/main.rs:1:2:",
            "src/main.rs:1:2:   ",
            "src/main.rs:1:2: error:",
            ":1:2: error: no path",
            "src/main.rs:1: error: missing column",
        ];
        for raw in hostile {
            let err = DiagnosticSet::try_from_text(raw).unwrap_err();
            assert!(
                matches!(err, EvidenceError::Malformed(_)),
                "{raw:?} must be a typed Malformed error, got {err:?}"
            );
        }
        // A single bad row poisons the whole input, even after a good row.
        let err =
            DiagnosticSet::try_from_text("a.rs:1:1: error: fine\nthis row is junk").unwrap_err();
        assert!(matches!(err, EvidenceError::Malformed(_)), "{err:?}");
        assert!(err.to_string().contains("line 2"), "{err}");
    }

    #[test]
    fn diagnostics_field_cap_sets_flag_never_silently() {
        let exact_message = "m".repeat(MAX_FIELD_CHARS);
        let raw = format!("a.rs:1:1: error: {exact_message}");
        let set = DiagnosticSet::try_from_text(&raw).unwrap();
        assert!(!set.field_truncated);
        assert_eq!(set.diagnostics[0].message.chars().count(), MAX_FIELD_CHARS);

        let long_path = "p".repeat(MAX_FIELD_CHARS + 100);
        let long_message = "m".repeat(MAX_FIELD_CHARS * 2);
        let long_code = "c".repeat(MAX_FIELD_CHARS + 1);
        let raw = format!("{long_path}:7:9: warning[{long_code}]: {long_message}");
        let set = DiagnosticSet::try_from_text(&raw).unwrap();
        assert!(set.field_truncated);
        let diagnostic = &set.diagnostics[0];
        assert_eq!(diagnostic.path.chars().count(), MAX_FIELD_CHARS);
        assert_eq!(diagnostic.message.chars().count(), MAX_FIELD_CHARS);
        assert_eq!(
            diagnostic
                .code
                .as_deref()
                .map(str::chars)
                .map(Iterator::count),
            Some(MAX_FIELD_CHARS)
        );
        assert_eq!(diagnostic.line, 7);
        assert_eq!(diagnostic.column, 9);

        // Cap counts chars, not bytes.
        let emoji = "\u{1F4A5}".repeat(MAX_FIELD_CHARS + 10);
        let set = DiagnosticSet::try_from_text(&format!("a.rs:1:1: error: {emoji}")).unwrap();
        assert!(set.field_truncated);
        assert_eq!(set.diagnostics[0].message.chars().count(), MAX_FIELD_CHARS);
    }

    #[test]
    fn test_report_parses_summary_and_failures() {
        let raw = "total=4 passed=2 skipped=1 duration_ms=1234\n\
                   FAIL tests::a: assertion failed\n\
                   fail tests::b: fell over\n";
        let report = TestReport::try_from_text(raw).unwrap();
        assert!(!report.field_truncated);
        assert_eq!(report.total, 4);
        assert_eq!(report.passed, 2);
        assert_eq!(report.skipped, 1);
        assert_eq!(report.duration_ms, 1234);
        assert_eq!(
            report.failed,
            vec![
                TestFailure {
                    name: "tests::a".to_string(),
                    message: "assertion failed".to_string(),
                },
                TestFailure {
                    name: "tests::b".to_string(),
                    message: "fell over".to_string(),
                },
            ]
        );

        // total is derived when absent; the last explicit key wins.
        let derived = TestReport::try_from_text("passed=3 skipped=1\nFAIL x: y\n").unwrap();
        assert_eq!((derived.total, derived.passed, derived.skipped), (5, 3, 1));
        let last_wins = TestReport::try_from_text("total=1 total=2,passed=2").unwrap();
        assert_eq!((last_wins.total, last_wins.passed), (2, 2));

        // Unrecognized rows are ignored; a broken recognized value is typed.
        let noisy = TestReport::try_from_text("noise\nmore noise=ignored\npassed=1").unwrap();
        assert_eq!(noisy.passed, 1);
        for bad in [
            "total=many",
            "passed=-1",
            "skipped=1.5",
            "duration_ms=99999999999999999999",
        ] {
            let err = TestReport::try_from_text(bad).unwrap_err();
            assert!(
                matches!(err, EvidenceError::Malformed(_)),
                "{bad:?} must be Malformed, got {err:?}"
            );
        }
    }

    #[test]
    fn test_report_field_cap_sets_flag() {
        let raw = format!(
            "FAIL {}: {}",
            "n".repeat(MAX_FIELD_CHARS + 7),
            "m".repeat(MAX_FIELD_CHARS + 7)
        );
        let report = TestReport::try_from_text(&raw).unwrap();
        assert!(report.field_truncated);
        assert_eq!(report.failed[0].name.chars().count(), MAX_FIELD_CHARS);
        assert_eq!(report.failed[0].message.chars().count(), MAX_FIELD_CHARS);
    }

    #[test]
    fn search_results_parse_hits_strictly() {
        let raw = "src/lib.rs:42:fn main() {}\n\
                   C:\\repo\\a.rs:7:let x = 1;\n\
                   /tmp/a:b.rs:1:\n";
        let results = SearchResults::try_from_text(raw).unwrap();
        assert!(!results.field_truncated);
        assert_eq!(
            results.hits,
            vec![
                SearchHit {
                    path: "src/lib.rs".to_string(),
                    line: 42,
                    text: "fn main() {}".to_string(),
                },
                SearchHit {
                    path: r"C:\repo\a.rs".to_string(),
                    line: 7,
                    text: "let x = 1;".to_string(),
                },
                SearchHit {
                    path: "/tmp/a:b.rs".to_string(),
                    line: 1,
                    text: String::new(),
                },
            ]
        );

        for bad in [
            "no colon at all",
            "file.rs:abc:text",
            "file.rs:4294967296:text",
            "file.rs::text",
            ":1:text",
            "file.rs:1 text",
        ] {
            let err = SearchResults::try_from_text(bad).unwrap_err();
            assert!(
                matches!(err, EvidenceError::Malformed(_)),
                "{bad:?} must be Malformed, got {err:?}"
            );
        }
    }

    #[test]
    fn search_field_cap_sets_flag() {
        let raw = format!(
            "{}:1:{}",
            "p".repeat(MAX_FIELD_CHARS + 1),
            "t".repeat(MAX_FIELD_CHARS + 1)
        );
        let results = SearchResults::try_from_text(&raw).unwrap();
        assert!(results.field_truncated);
        assert_eq!(results.hits[0].path.chars().count(), MAX_FIELD_CHARS);
        assert_eq!(results.hits[0].text.chars().count(), MAX_FIELD_CHARS);
    }

    #[test]
    fn process_log_collapses_runs_with_golden_fingerprints() {
        let raw = "boot\nspam\n  spam  \nSPAM\nstart\nstart\nend";
        let summary = ProcessLogSummary::try_from_text(raw).unwrap();
        assert!(!summary.field_truncated);
        assert_eq!(summary.exit, None);
        assert_eq!(
            summary.repeated_groups,
            vec![
                RepeatedLogGroup {
                    fingerprint: 5_434_107_346_319_650_050,
                    count: 2,
                    sample: "spam".to_string(),
                },
                RepeatedLogGroup {
                    fingerprint: 9_435_337_193_356_032_396,
                    count: 2,
                    sample: "start".to_string(),
                },
            ]
        );
        // Trimming happens before hashing: whitespace variants collapse into
        // one group with a stable fingerprint, and parsing is deterministic.
        let spaced = ProcessLogSummary::try_from_text("spam\n  spam  ").unwrap();
        assert_eq!(spaced.repeated_groups.len(), 1);
        assert_eq!(
            spaced.repeated_groups[0].fingerprint,
            expected_fingerprint("spam")
        );
        assert_eq!(summary, ProcessLogSummary::try_from_text(raw).unwrap());

        // Non-consecutive duplicates are separate runs, never groups.
        let scattered = ProcessLogSummary::try_from_text("a\nb\na").unwrap();
        assert!(scattered.repeated_groups.is_empty());
    }

    #[test]
    fn process_log_important_lines_prioritize_keywords_and_keep_final() {
        let mut raw = String::new();
        for index in 0..260 {
            raw.push_str(&format!("error number {index}\n"));
        }
        for index in 260..300 {
            raw.push_str(&format!("noise {index}\n"));
        }
        raw.push_str("done\n");
        let summary = ProcessLogSummary::try_from_text(&raw).unwrap();
        assert_eq!(summary.important_lines.len(), MAX_IMPORTANT_LINES);
        assert_eq!(
            summary.important_lines.last().unwrap().text,
            "done".to_string()
        );
        assert!(summary.important_lines[..MAX_IMPORTANT_LINES - 1]
            .iter()
            .all(|event| event.text.starts_with("error number")));

        // A final keyword line is kept exactly once, and the final row is
        // retained even when nothing matches.
        let small = ProcessLogSummary::try_from_text("error a\nerror b").unwrap();
        assert_eq!(small.important_lines.len(), 2);
        assert_eq!(small.important_lines[1].text, "error b");
        let quiet = ProcessLogSummary::try_from_text("nothing to see\nlast").unwrap();
        assert_eq!(quiet.important_lines.len(), 1);
        assert_eq!(quiet.important_lines[0].text, "last");
    }

    #[test]
    fn process_log_exit_code_parsing_is_typed_and_first_wins() {
        let cases = [
            ("exit code: 1", Some(1)),
            ("Exit Status = -1", Some(-1)),
            ("exit_code=4", Some(4)),
            ("exit: 2", Some(2)),
            ("exit 7", Some(7)),
            ("process exited with code 3", Some(3)),
            ("exited with status 5.", Some(5)),
            ("no exit information", None),
            ("exit code: not-a-number", None),
            ("myexit code: 3", None),
        ];
        for (raw, expected) in cases {
            assert_eq!(
                ProcessLogSummary::try_from_text(raw).unwrap().exit,
                expected,
                "{raw:?}"
            );
        }
        let first_wins = ProcessLogSummary::try_from_text("exit: 2\nexit: 9").unwrap();
        assert_eq!(first_wins.exit, Some(2));
    }

    #[test]
    fn process_log_field_cap_sets_flag_on_samples_and_events() {
        let long = "e".repeat(MAX_FIELD_CHARS + 10);
        let summary = ProcessLogSummary::try_from_text(&format!("{long}\n{long}\n")).unwrap();
        assert!(summary.field_truncated);
        assert_eq!(
            summary.repeated_groups[0].sample.chars().count(),
            MAX_FIELD_CHARS
        );
        assert_eq!(
            summary.important_lines[0].text.chars().count(),
            MAX_FIELD_CHARS
        );
    }

    #[test]
    fn oversized_inputs_are_typed_for_every_normalizer() {
        let oversized = "x\n".repeat(MAX_INPUT_LINES + 1);
        let results = [
            DiagnosticSet::try_from_text(&oversized).map(|_| ()),
            TestReport::try_from_text(&oversized).map(|_| ()),
            SearchResults::try_from_text(&oversized).map(|_| ()),
            ProcessLogSummary::try_from_text(&oversized).map(|_| ()),
        ];
        for result in results {
            match result {
                Err(EvidenceError::Oversized { max, actual }) => {
                    assert_eq!(max, MAX_INPUT_LINES);
                    assert_eq!(actual, MAX_INPUT_LINES + 1);
                }
                other => panic!("expected Oversized, got {other:?}"),
            }
        }
        // Exactly at the cap is still admissible.
        let at_cap = "\n".repeat(MAX_INPUT_LINES);
        let set = DiagnosticSet::try_from_text(&at_cap).unwrap();
        assert!(set.diagnostics.is_empty());
        assert!(!set.field_truncated);
    }

    #[test]
    fn empty_and_blank_inputs_yield_empty_values() {
        for raw in ["", " ", "\n", "\r\n", "\t\n  \n"] {
            let set = DiagnosticSet::try_from_text(raw).unwrap();
            assert!(set.diagnostics.is_empty(), "{raw:?}");
            assert!(!set.field_truncated, "{raw:?}");

            let report = TestReport::try_from_text(raw).unwrap();
            assert_eq!(
                report,
                TestReport {
                    total: 0,
                    passed: 0,
                    failed: Vec::new(),
                    skipped: 0,
                    duration_ms: 0,
                    field_truncated: false,
                },
                "{raw:?}"
            );

            let results = SearchResults::try_from_text(raw).unwrap();
            assert!(results.hits.is_empty(), "{raw:?}");

            let summary = ProcessLogSummary::try_from_text(raw).unwrap();
            assert_eq!(summary.exit, None, "{raw:?}");
            assert!(summary.important_lines.is_empty(), "{raw:?}");
            assert!(summary.repeated_groups.is_empty(), "{raw:?}");
            assert!(!summary.field_truncated, "{raw:?}");
        }
    }

    #[test]
    fn junk_never_panics() {
        let junk = [
            "\u{0}\u{1}\u{7f}",
            "\u{1F4A5}\u{1F4A5}\u{1F4A5}",
            "a:b:c",
            "::::",
            "path:1:2: error: ",
            "= = =",
            "----",
            "exit code: not-a-number",
            "path:4294967295:4294967295: info[x]: ok",
            "line with trailing colon:",
        ];
        for raw in junk {
            let _ = DiagnosticSet::try_from_text(raw);
            let _ = TestReport::try_from_text(raw);
            let _ = SearchResults::try_from_text(raw);
            let _ = ProcessLogSummary::try_from_text(raw);
        }
        let huge_line = "x".repeat(200_000);
        let _ = DiagnosticSet::try_from_text(&huge_line);
        let _ = TestReport::try_from_text(&huge_line);
        let _ = SearchResults::try_from_text(&huge_line);
        let _ = ProcessLogSummary::try_from_text(&huge_line);
    }

    #[test]
    fn normalizer_outputs_round_trip_through_json() {
        let diagnostics = DiagnosticSet::try_from_text("a.rs:1:2: error[E1]: boom").unwrap();
        let report = TestReport::try_from_text("passed=1\nFAIL x: y").unwrap();
        let search = SearchResults::try_from_text("a.rs:1:code").unwrap();
        let log = ProcessLogSummary::try_from_text("spam\nspam\nexit code: 0").unwrap();

        for json in [
            serde_json::to_string(&diagnostics).unwrap(),
            serde_json::to_string(&report).unwrap(),
            serde_json::to_string(&search).unwrap(),
            serde_json::to_string(&log).unwrap(),
        ] {
            assert!(json.contains("field_truncated"), "{json}");
        }
        assert_eq!(
            serde_json::from_str::<DiagnosticSet>(&serde_json::to_string(&diagnostics).unwrap())
                .unwrap(),
            diagnostics
        );
        assert_eq!(
            serde_json::from_str::<TestReport>(&serde_json::to_string(&report).unwrap()).unwrap(),
            report
        );
        assert_eq!(
            serde_json::from_str::<SearchResults>(&serde_json::to_string(&search).unwrap())
                .unwrap(),
            search
        );
        assert_eq!(
            serde_json::from_str::<ProcessLogSummary>(&serde_json::to_string(&log).unwrap())
                .unwrap(),
            log
        );
    }
}
