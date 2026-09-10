//! Compact line grammars for evidence bodies: `DIAGv1`, `TESTSv1`, `SEARCHv1`
//! and `LOGv1`.
//!
//! Every body is line-oriented and every line is field-oriented. Fields are
//! joined with `|` and escaped by [`escape_field`] so the field separator and
//! the row separator can never appear raw inside a field:
//!
//! - backslash -> `\\`
//! - pipe -> `\|`
//! - newline -> `\n` (backslash followed by `n`)
//! - carriage return -> `\r`
//!
//! Parsing is the exact inverse: fields are only split on *unescaped* pipes,
//! and [`unescape_field`] refuses any byte sequence the escaping rules could
//! not have produced (unknown escapes, a trailing backslash, a raw delimiter).
//! Nothing here panics on junk: every rejection is a typed
//! [`EvidenceError::Malformed`].
//!
//! One marker is reserved and can never round-trip as data: an absent
//! diagnostic `code` renders as `-`, so a literal `-` code parses back as
//! `None`.

use std::collections::HashSet;

use crate::types::{CompactRepresentation, EvidenceError};

/// The `DIAGv1` grammar version this module writes.
pub const DIAG_GRAMMAR: &str = "DIAGv1";
/// The `TESTSv1` grammar version this module writes.
pub const TESTS_GRAMMAR: &str = "TESTSv1";
/// The `SEARCHv1` grammar version this module writes.
pub const SEARCH_GRAMMAR: &str = "SEARCHv1";
/// The `LOGv1` grammar version this module writes.
pub const LOG_GRAMMAR: &str = "LOGv1";

/// One diagnostic row in the self-contained `DIAGv1` grammar.
///
/// Deliberately independent of `normalize.rs`: this module owns its row shape
/// so rendering and parsing stay testable in isolation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosticRow {
    pub path: String,
    pub line: u32,
    pub column: u32,
    pub severity: String,
    pub code: Option<String>,
    pub message: String,
}

/// Escape one field so it can be embedded between `|` separators and inside
/// `\n`-joined rows without ever being mistaken for grammar.
pub fn escape_field(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for c in raw.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '|' => out.push_str("\\|"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            other => out.push(other),
        }
    }
    out
}

/// Inverse of [`escape_field`]. Rejects anything the escaper could not have
/// produced: an unknown escape sequence, a trailing backslash, or a raw
/// (unescaped) pipe delimiter.
pub fn unescape_field(raw: &str) -> Result<String, EvidenceError> {
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => match chars.next() {
                Some('\\') => out.push('\\'),
                Some('|') => out.push('|'),
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some(other) => {
                    return Err(EvidenceError::Malformed(format!(
                        "invalid escape sequence \\{other} in field {raw:?}"
                    )))
                }
                None => {
                    return Err(EvidenceError::Malformed(format!(
                        "trailing backslash in field {raw:?}"
                    )))
                }
            },
            '|' => {
                return Err(EvidenceError::Malformed(format!(
                    "unescaped pipe delimiter in field {raw:?}"
                )))
            }
            other => out.push(other),
        }
    }
    Ok(out)
}

/// Split one row into raw fields, treating `\` plus the next character as an
/// atomic escape pair so `\|` never acts as a separator. The resulting fields
/// still need [`unescape_field`] to be validated and decoded.
fn split_escaped(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                current.push(c);
                if let Some(escaped) = chars.next() {
                    current.push(escaped);
                }
            }
            '|' => fields.push(std::mem::take(&mut current)),
            other => current.push(other),
        }
    }
    fields.push(current);
    fields
}

/// Strict decimal `u32`: no sign, no whitespace, no radix prefix, no overflow.
fn parse_u32_strict(raw: &str) -> Result<u32, EvidenceError> {
    let malformed = || EvidenceError::Malformed(format!("{raw:?} is not a plain u32"));
    if raw.is_empty() || !raw.bytes().all(|b| b.is_ascii_digit()) {
        return Err(malformed());
    }
    raw.parse::<u32>().map_err(|_| malformed())
}

fn parse_diagnostic_row(line: &str) -> Result<DiagnosticRow, EvidenceError> {
    let fields = split_escaped(line);
    if fields.len() != 5 {
        return Err(EvidenceError::Malformed(format!(
            "diagnostic row must have exactly 5 pipe-separated fields, got {}",
            fields.len()
        )));
    }
    let path = unescape_field(&fields[0])?;
    let location = unescape_field(&fields[1])?;
    let severity = unescape_field(&fields[2])?;
    let code = unescape_field(&fields[3])?;
    let message = unescape_field(&fields[4])?;

    let (line_raw, column_raw) = location.split_once(':').ok_or_else(|| {
        EvidenceError::Malformed(format!("{location:?} is not a line:column location"))
    })?;
    if column_raw.contains(':') {
        return Err(EvidenceError::Malformed(format!(
            "{location:?} is not a line:column location"
        )));
    }
    let line = parse_u32_strict(line_raw)?;
    let column = parse_u32_strict(column_raw)?;

    let code = if code == "-" { None } else { Some(code) };
    Ok(DiagnosticRow {
        path,
        line,
        column,
        severity,
        code,
        message,
    })
}

/// Render diagnostics as `path|line:col|severity|code|message` rows joined by
/// newlines, with every field escaped and an absent `code` rendered as `-`.
pub fn render_diagnostics(rows: &[DiagnosticRow]) -> CompactRepresentation {
    let mut body = String::new();
    for (index, row) in rows.iter().enumerate() {
        if index > 0 {
            body.push('\n');
        }
        body.push_str(&escape_field(&row.path));
        body.push('|');
        body.push_str(&row.line.to_string());
        body.push(':');
        body.push_str(&row.column.to_string());
        body.push('|');
        body.push_str(&escape_field(&row.severity));
        body.push('|');
        match &row.code {
            Some(code) => body.push_str(&escape_field(code)),
            None => body.push('-'),
        }
        body.push('|');
        body.push_str(&escape_field(&row.message));
    }
    CompactRepresentation {
        grammar: DIAG_GRAMMAR.to_string(),
        body,
    }
}

/// Parse a `DIAGv1` body back into rows. An empty body is zero rows; every
/// other malformed shape is a typed [`EvidenceError::Malformed`].
pub fn parse_diagnostics(body: &str) -> Result<Vec<DiagnosticRow>, EvidenceError> {
    if body.is_empty() {
        return Ok(Vec::new());
    }
    body.split('\n')
        .enumerate()
        .map(|(index, line)| {
            parse_diagnostic_row(line)
                .map_err(|err| EvidenceError::Malformed(format!("row {index}: {err}")))
        })
        .collect()
}

/// Render a test run summary plus one escaped `FAIL|name|message` row per
/// failed case.
pub fn render_tests(
    failed: &[(String, String)],
    total: usize,
    passed: usize,
    skipped: usize,
    duration_ms: u64,
) -> CompactRepresentation {
    let mut body = format!(
        "total={total} passed={passed} skipped={skipped} failed={} duration_ms={duration_ms}",
        failed.len()
    );
    for (name, message) in failed {
        body.push('\n');
        body.push_str("FAIL|");
        body.push_str(&escape_field(name));
        body.push('|');
        body.push_str(&escape_field(message));
    }
    CompactRepresentation {
        grammar: TESTS_GRAMMAR.to_string(),
        body,
    }
}

/// Render search hits as escaped `path|line` rows, deduplicated by exact
/// `(path, line)` while preserving first-occurrence order.
pub fn render_search(hits: &[(String, u32)]) -> CompactRepresentation {
    let mut seen: HashSet<(&str, u32)> = HashSet::new();
    let mut body = String::new();
    for (path, line) in hits {
        if !seen.insert((path.as_str(), *line)) {
            continue;
        }
        if !body.is_empty() {
            body.push('\n');
        }
        body.push_str(&escape_field(path));
        body.push('|');
        body.push_str(&line.to_string());
    }
    CompactRepresentation {
        grammar: SEARCH_GRAMMAR.to_string(),
        body,
    }
}

/// Render a process log: an `exit=..`/group-count header, one escaped
/// `G|start|end|text` row per output group, then one escaped
/// `I|-|-|text` row per important line.
pub fn render_log(
    groups: &[(u64, u64, String)],
    important: &[String],
    exit: Option<i32>,
) -> CompactRepresentation {
    let mut body = String::new();
    body.push_str("exit=");
    match exit {
        Some(code) => body.push_str(&code.to_string()),
        None => body.push('-'),
    }
    body.push_str(&format!(
        " groups={} important={}",
        groups.len(),
        important.len()
    ));
    for (start, end, text) in groups {
        body.push('\n');
        body.push_str(&format!("G|{start}|{end}|"));
        body.push_str(&escape_field(text));
    }
    for text in important {
        body.push('\n');
        body.push_str("I|-|-|");
        body.push_str(&escape_field(text));
    }
    CompactRepresentation {
        grammar: LOG_GRAMMAR.to_string(),
        body,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diag(
        path: &str,
        line: u32,
        column: u32,
        severity: &str,
        code: Option<&str>,
        message: &str,
    ) -> DiagnosticRow {
        DiagnosticRow {
            path: path.to_string(),
            line,
            column,
            severity: severity.to_string(),
            code: code.map(str::to_string),
            message: message.to_string(),
        }
    }

    fn has_unescaped_pipe(s: &str) -> bool {
        let mut chars = s.chars();
        while let Some(c) = chars.next() {
            match c {
                '\\' => {
                    let _ = chars.next();
                }
                '|' => return true,
                _ => {}
            }
        }
        false
    }

    #[test]
    fn escape_unescape_round_trips_hostile_payloads() {
        let hostile = [
            "",
            "plain text",
            "|",
            "|||",
            "\\",
            "\\\\",
            "\\|",
            "\n",
            "\r",
            "\r\n",
            "line one\nline two\r\nline three",
            "a\\|b\nc\\d\re",
            "prefix|middle\\suffix\n",
            "日本語/ファイル.rs",
            "emoji 🎉 with | and \\ and \n",
            "-",
            "a-b-c",
        ];
        for raw in hostile {
            let escaped = escape_field(raw);
            assert!(!escaped.contains('\n'), "raw newline leaked for {raw:?}");
            assert!(!escaped.contains('\r'), "raw CR leaked for {raw:?}");
            assert!(
                !has_unescaped_pipe(&escaped),
                "raw pipe leaked for {raw:?} -> {escaped:?}"
            );
            assert_eq!(unescape_field(&escaped).unwrap(), raw);
        }
    }

    #[test]
    fn unescape_field_rejects_broken_grammar() {
        for bad in [
            "\\", "abc\\", "\\x", "a\\qb", "a|b", "|", "a\\n\\", "\\\\\\",
        ] {
            match unescape_field(bad) {
                Err(EvidenceError::Malformed(message)) => assert!(!message.is_empty()),
                other => panic!("{bad:?} must be typed Malformed, got {other:?}"),
            }
        }
    }

    #[test]
    fn diagnostics_render_parse_round_trip() {
        let rows = vec![
            diag(
                "src/main.rs",
                12,
                5,
                "error",
                Some("E0308"),
                "mismatched types",
            ),
            diag(
                "src/weird|name\\with\nnewline.rs",
                1,
                1,
                "warning",
                None,
                "first\nsecond | pipe \\ slash \r carriage",
            ),
            diag("", 0, 0, "", Some(""), ""),
            diag(
                "日本語/ファイル.rs",
                u32::MAX,
                u32::MAX,
                "hint",
                Some("clippy::all"),
                "🎉",
            ),
        ];
        let rendered = render_diagnostics(&rows);
        assert_eq!(rendered.grammar, DIAG_GRAMMAR);
        assert!(!rendered.body.ends_with('\n'));
        assert_eq!(rendered.body.lines().count(), rows.len());
        assert_eq!(parse_diagnostics(&rendered.body).unwrap(), rows);

        let empty = render_diagnostics(&[]);
        assert_eq!(empty.grammar, DIAG_GRAMMAR);
        assert_eq!(empty.body, "");
        assert_eq!(parse_diagnostics("").unwrap(), Vec::new());
    }

    #[test]
    fn malformed_diagnostics_are_typed() {
        let bad = [
            "single-field",
            "a|1:2|error|-",
            "a|1:2|error|-|message|extra",
            "a|1:2:3|error|-|message",
            "a|:2|error|-|message",
            "a|1:|error|-|message",
            "a|1.5:2|error|-|message",
            "a|+1:2|error|-|message",
            "a| 1:2|error|-|message",
            "a|1:2 |error|-|message",
            "a|4294967296:1|error|-|message",
            "a|1:2|error|-|msg\\",
            "a|1:2|err\\q|-|msg",
            "a|1:2|error|-|m|sg",
            "a|1:2|error|-|msg\n",
            "\n",
            "||||",
        ];
        for raw in bad {
            match parse_diagnostics(raw) {
                Err(EvidenceError::Malformed(message)) => assert!(!message.is_empty()),
                other => panic!("{raw:?} must be typed Malformed, got {other:?}"),
            }
        }
    }

    #[test]
    fn dash_code_marker_is_reserved_for_absent_code() {
        let literal_dash = vec![diag("a.rs", 1, 1, "error", Some("-"), "literal dash")];
        let parsed = parse_diagnostics(&render_diagnostics(&literal_dash).body).unwrap();
        assert_eq!(parsed[0].code, None);

        let absent = vec![diag("a.rs", 1, 1, "error", None, "no code")];
        assert_eq!(
            parse_diagnostics(&render_diagnostics(&absent).body).unwrap(),
            absent
        );
    }

    #[test]
    fn search_render_dedupes_preserving_first_occurrence() {
        let hits = [
            ("src/a.rs".to_string(), 1u32),
            ("src/b|c.rs".to_string(), 2),
            ("src/a.rs".to_string(), 1),
            ("src/a.rs".to_string(), 3),
            ("src/b|c.rs".to_string(), 2),
        ];
        let rendered = render_search(&hits);
        assert_eq!(rendered.grammar, SEARCH_GRAMMAR);
        assert_eq!(rendered.body, "src/a.rs|1\nsrc/b\\|c.rs|2\nsrc/a.rs|3");
        assert_eq!(render_search(&[]).body, "");
    }

    #[test]
    fn tests_and_logs_render_escaped_bodies() {
        let failed = [("case|one".to_string(), "boom\nnext".to_string())];
        let tests = render_tests(&failed, 3, 2, 0, 41);
        assert_eq!(tests.grammar, TESTS_GRAMMAR);
        assert_eq!(
            tests.body,
            "total=3 passed=2 skipped=0 failed=1 duration_ms=41\nFAIL|case\\|one|boom\\nnext"
        );

        let groups = [(1u64, 4u64, "a\nb".to_string())];
        let important = ["careful|now".to_string()];
        let log = render_log(&groups, &important, Some(2));
        assert_eq!(log.grammar, LOG_GRAMMAR);
        assert_eq!(
            log.body,
            "exit=2 groups=1 important=1\nG|1|4|a\\nb\nI|-|-|careful\\|now"
        );
        assert_eq!(
            render_log(&[], &[], None).body,
            "exit=- groups=0 important=0"
        );
    }

    #[test]
    fn rendering_is_byte_deterministic_across_calls() {
        let rows = vec![diag("p|q", 3, 4, "error", None, "m\\n")];
        let first = render_diagnostics(&rows);
        let second = render_diagnostics(&rows);
        assert_eq!(first, second);
        assert_eq!(first.body.as_bytes(), second.body.as_bytes());

        let hits = [("p".to_string(), 1u32), ("p".to_string(), 1)];
        assert_eq!(render_search(&hits), render_search(&hits));

        let failed = [("t".to_string(), "m".to_string())];
        assert_eq!(
            render_tests(&failed, 1, 0, 0, 0),
            render_tests(&failed, 1, 0, 0, 0)
        );

        let groups = [(0u64, 1u64, "x".to_string())];
        let important = ["y".to_string()];
        assert_eq!(
            render_log(&groups, &important, None),
            render_log(&groups, &important, None)
        );
    }

    #[test]
    fn junk_never_panics() {
        let junk: Vec<String> = vec![
            String::new(),
            "\0\u{1}\u{7f}".to_string(),
            "🦀|🦀\n🦀\\".to_string(),
            "\\".repeat(4097),
            "|".repeat(4097),
            "\\n\\r\\|".repeat(1000),
            format!(
                "{}|1:1|error|-|{}",
                "p".repeat(64 * 1024),
                "m".repeat(64 * 1024)
            ),
            format!("{}\\", "x".repeat(1000)),
        ];
        for raw in &junk {
            let _ = parse_diagnostics(raw);
            let _ = unescape_field(raw);
            let _ = escape_field(raw);
        }
        assert!(parse_diagnostics(&"|".repeat(10_000)).is_err());
        assert!(parse_diagnostics(&"\\".repeat(10_001)).is_err());
        let odd = "\\".repeat(10_000);
        assert_eq!(unescape_field(&odd).unwrap().len(), 5_000);
        let round = escape_field(&"\\".repeat(10_001));
        assert_eq!(unescape_field(&round).unwrap().len(), 10_001);
    }
}
