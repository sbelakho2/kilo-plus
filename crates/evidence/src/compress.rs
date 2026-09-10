//! Deterministic, bounded compaction of raw evidence into compact
//! representations.
//!
//! Compression policy is hardcoded per [`EvidenceKind`] via
//! [`EvidenceKind::default_compressibility`]; call sites never pick it by
//! mood. The public surface is [`compress_for`], [`compress`] and the
//! explicit [`compress_never_identity`] escape hatch for `Never` payloads.
//!
//! Hard rules enforced here:
//!
//! - `Never` refuses every compaction attempt; verbatim bytes are only
//!   produced when the caller asks for `compress_never_identity` by name, so
//!   policy input can never be silently rewritten.
//! - `LosslessOnly` copies bytes verbatim and records `Lossiness::None` with
//!   `original_bytes == compact_bytes`.
//! - `Reversible` requires [`BackingCompleteness::Complete`]: truncated
//!   backing can never be re-expanded, so it is refused with
//!   [`EvidenceError::Refused`], never downgraded.
//! - `Aggressive` and `Reversible` always stamp a blake3 digest of the
//!   backing bytes into the [`CompressionRecord`].
//! - Every compact body produced here is bounded by `MAX_COMPACT_BYTES`
//!   (16 KiB). Transforms stream line by line, cap retained lines, and bound
//!   every "seen" set, so hostile multi-megabyte inputs cannot blow up RAM.
//!
//! All transforms are deterministic: identical input bytes always yield
//! byte-identical compact bodies (no hash-map iteration, no timing).

use std::borrow::Cow;
use std::collections::BTreeSet;

use crate::types::{
    BackingCompleteness, CompactRepresentation, Compressibility, CompressionRecord, EvidenceError,
    EvidenceKind, Lossiness,
};

/// Algorithm tag stamped on every [`CompressionRecord`] this module produces.
const COMPRESSION_ALGORITHM: &str = "faktor-compress-v1";
/// Grammar/version stamped on every [`CompressionRecord`].
const COMPRESSION_VERSION: u32 = 1;
/// Hard ceiling for any compact body produced by a structural or aggressive
/// transform.
const MAX_COMPACT_BYTES: usize = 16 * 1024;
/// Maximum bytes kept per retained line (truncated on a char boundary).
const MAX_LINE_BYTES: usize = 512;
/// Budget for the leading (verbatim) section of one compact body.
const MAX_HEAD_BYTES: usize = 6 * 1024;
/// Budget for the retained failure lines of one compact body.
const MAX_FAILURE_BYTES: usize = 6 * 1024;
/// Maximum bytes of one line fed into repetition normalization.
const NORMALIZE_CAP: usize = 256;
/// Maximum whitespace columns kept per table row.
const MAX_TABLE_COLUMNS: usize = 6;
/// Maximum distinct keys remembered for dedupe (memory is bounded, not just
/// output).
const MAX_SEEN_KEYS: usize = 4096;
/// Content lines kept after each diff hunk header.
const FIRST_HUNK_LINES: usize = 3;
/// Lines kept verbatim by the generic structural summary.
const STRUCTURAL_FIRST_LINES: usize = 24;
/// Marker appended to a line that exceeded [`MAX_LINE_BYTES`].
const TRUNCATION_MARKER: &str = "\u{2026}[truncated]";

/// Returns the hardcoded compression policy for `kind`.
///
/// Thin, explicit delegation to [`EvidenceKind::default_compressibility`] so
/// every call site reads through one evidence-owned function rather than
/// reaching into the type table directly.
pub fn compress_for(kind: &EvidenceKind) -> Compressibility {
    kind.default_compressibility()
}

/// The explicit verbatim path for `Never` payloads.
///
/// Deliberately separate from [`compress`]: a `Never` payload is copied only
/// when the caller asks for it by name, so no call site can "compact" policy
/// input by accident. Infallible, because an identity copy has no failure
/// mode; the body is exactly `raw` and the record is lossless.
pub fn compress_never_identity(raw: &str) -> (CompactRepresentation, CompressionRecord) {
    let compact = CompactRepresentation {
        grammar: "identity-v1".to_string(),
        body: raw.to_string(),
    };
    let record = make_record(raw.len(), compact.body.len(), Lossiness::None, None);
    (compact, record)
}

/// Compacts `raw` under `kind`'s hardcoded policy.
///
/// Returns the compact representation plus the record that ties it back to
/// the backing bytes. See the module docs for the exact per-policy rules;
/// violations of a hard invariant are [`EvidenceError::Refused`], never a
/// silent downgrade.
pub fn compress(
    kind: &EvidenceKind,
    backing_completeness: BackingCompleteness,
    raw: &str,
) -> Result<(CompactRepresentation, CompressionRecord), EvidenceError> {
    compress_with(kind, compress_for(kind), backing_completeness, raw)
}

/// Policy-driven core, split out so tests can exercise policies that no
/// current [`EvidenceKind`] maps to (notably `Never`).
fn compress_with(
    kind: &EvidenceKind,
    policy: Compressibility,
    backing_completeness: BackingCompleteness,
    raw: &str,
) -> Result<(CompactRepresentation, CompressionRecord), EvidenceError> {
    let digest = || *blake3::hash(raw.as_bytes()).as_bytes();
    let (grammar, body, lossiness, backing_digest) = match policy {
        Compressibility::Never => {
            return Err(EvidenceError::Refused(format!(
                "{kind:?} uses Compressibility::Never: compaction is forbidden; \
                 call compress_never_identity for an explicit verbatim copy"
            )))
        }
        Compressibility::LosslessOnly => (
            "identity-v1".to_string(),
            raw.to_string(),
            Lossiness::None,
            None,
        ),
        Compressibility::Reversible => {
            if backing_completeness != BackingCompleteness::Complete {
                return Err(EvidenceError::Refused(format!(
                    "{kind:?} claims Reversible but backing is Truncated: truncated \
                     backing can never be re-expanded"
                )));
            }
            (
                grammar_for(kind).to_string(),
                structural_summary_body(kind, raw),
                Lossiness::Structural,
                Some(digest()),
            )
        }
        Compressibility::Aggressive => (
            grammar_for(kind).to_string(),
            aggressive_body(kind, raw),
            Lossiness::Content,
            Some(digest()),
        ),
    };
    let compact = CompactRepresentation { grammar, body };
    let record = make_record(raw.len(), compact.body.len(), lossiness, backing_digest);
    Ok((compact, record))
}

fn make_record(
    original_bytes: usize,
    compact_bytes: usize,
    lossiness: Lossiness,
    backing_digest: Option<[u8; 32]>,
) -> CompressionRecord {
    CompressionRecord {
        algorithm: COMPRESSION_ALGORITHM.to_string(),
        version: COMPRESSION_VERSION,
        original_bytes: original_bytes as u64,
        compact_bytes: compact_bytes as u64,
        lossiness,
        backing_digest,
    }
}

fn grammar_for(kind: &EvidenceKind) -> &'static str {
    match kind {
        EvidenceKind::DiagnosticSet => "diag-v1",
        EvidenceKind::TestReport => "test-v1",
        EvidenceKind::SearchResults => "search-v1",
        EvidenceKind::FileMap => "filemap-v1",
        EvidenceKind::Diff => "diff-v1",
        EvidenceKind::ProcessLog => "log-v1",
        EvidenceKind::SymbolSet => "symbol-v1",
        EvidenceKind::StructuredRows => "rows-v1",
        EvidenceKind::ChildHandoff => "handoff-v1",
        EvidenceKind::SemanticContext => "semantic-v1",
        EvidenceKind::GenericText => "text-v1",
    }
}

fn kind_label(kind: &EvidenceKind) -> &'static str {
    match kind {
        EvidenceKind::DiagnosticSet => "diagnostic_set",
        EvidenceKind::TestReport => "test_report",
        EvidenceKind::SearchResults => "search_results",
        EvidenceKind::FileMap => "file_map",
        EvidenceKind::Diff => "diff",
        EvidenceKind::ProcessLog => "process_log",
        EvidenceKind::SymbolSet => "symbol_set",
        EvidenceKind::StructuredRows => "structured_rows",
        EvidenceKind::ChildHandoff => "child_handoff",
        EvidenceKind::SemanticContext => "semantic_context",
        EvidenceKind::GenericText => "generic_text",
    }
}

/// Aggressive transforms: each is kind-specific, deterministic and bounded.
fn aggressive_body(kind: &EvidenceKind, raw: &str) -> String {
    match kind {
        EvidenceKind::ProcessLog => process_log_body(raw),
        EvidenceKind::TestReport => test_report_body(raw),
        EvidenceKind::DiagnosticSet => columnar_dedupe_body(raw, "diagnostic_set"),
        EvidenceKind::SearchResults => columnar_dedupe_body(raw, "search_results"),
        EvidenceKind::FileMap | EvidenceKind::SymbolSet | EvidenceKind::StructuredRows => {
            column_table_body(kind, raw)
        }
        EvidenceKind::Diff => diff_summary_body(raw),
        EvidenceKind::ChildHandoff | EvidenceKind::SemanticContext | EvidenceKind::GenericText => {
            first_lines_summary_body(kind, raw)
        }
    }
}

/// Reversible transforms: structure survives, so they are summaries with a
/// backing digest rather than verbatim bodies.
fn structural_summary_body(kind: &EvidenceKind, raw: &str) -> String {
    match kind {
        EvidenceKind::Diff => diff_summary_body(raw),
        _ => first_lines_summary_body(kind, raw),
    }
}

/// Repetition collapse for process logs: consecutive identical or
/// normalized-identical lines become `line xN`, failure lines are always
/// kept, and the final input line is always retained.
fn process_log_body(raw: &str) -> String {
    let mut bound = Bound::new();
    let mut total: u64 = 0;
    let mut collapsed_runs: u64 = 0;
    let mut failure_lines: u64 = 0;
    let mut run_repr: Option<String> = None;
    let mut run_norm: Option<String> = None;
    let mut run_len: u64 = 0;
    let mut scratch = String::new();
    let mut last_raw: Option<&str> = None;

    for line in raw.lines() {
        total += 1;
        let capped = cap_line(line);
        if is_failure_line(&capped) {
            flush_run(
                &mut bound,
                &mut run_repr,
                &mut run_norm,
                &mut run_len,
                &mut collapsed_runs,
            );
            failure_lines += 1;
            bound.push_failure(&capped);
        } else {
            normalize_into(&capped, &mut scratch);
            if run_norm.as_deref() == Some(scratch.as_str()) {
                run_len += 1;
            } else {
                flush_run(
                    &mut bound,
                    &mut run_repr,
                    &mut run_norm,
                    &mut run_len,
                    &mut collapsed_runs,
                );
                run_norm = Some(scratch.clone());
                run_repr = Some(capped.to_string());
                run_len = 1;
            }
        }
        last_raw = Some(line);
    }
    flush_run(
        &mut bound,
        &mut run_repr,
        &mut run_norm,
        &mut run_len,
        &mut collapsed_runs,
    );
    if let Some(line) = last_raw {
        bound.observe_last(&cap_line(line));
    }

    let header = format!(
        "# kind=process_log lines={total} collapsed_runs={collapsed_runs} \
         failures={failure_lines}"
    );
    bound.render(&header)
}

fn flush_run(
    bound: &mut Bound,
    run_repr: &mut Option<String>,
    run_norm: &mut Option<String>,
    run_len: &mut u64,
    collapsed_runs: &mut u64,
) {
    if let Some(line) = run_repr.take() {
        if *run_len >= 2 {
            bound.push_head(&format!("{line} x{run_len}"));
            *collapsed_runs += 1;
        } else {
            bound.push_head(&line);
        }
    }
    *run_norm = None;
    *run_len = 0;
}

/// Keeps failed rows (and any failure-looking line) and aggregates the
/// pass/fail/ignored counts of the report.
fn test_report_body(raw: &str) -> String {
    let mut bound = Bound::new();
    let mut total: u64 = 0;
    let mut passed: u64 = 0;
    let mut failed: u64 = 0;
    let mut ignored: u64 = 0;

    for line in raw.lines() {
        total += 1;
        let capped = cap_line(line);
        let text = capped.as_ref();
        if is_failure_line(text) {
            failed += 1;
            bound.push_failure(text);
        } else if contains_ci(text, "ignored") {
            ignored += 1;
        } else if contains_ci(text, " ok") || text.trim_start().starts_with("ok ") {
            passed += 1;
        }
    }

    let header = format!(
        "# kind=test_report lines={total} rows={} passed={passed} failed={failed} ignored={ignored}",
        passed + failed + ignored
    );
    bound.render(&header)
}

/// Columnar dedupe: one retained row per `path:line` key (the first two
/// columns), first occurrence wins, later duplicates counted.
fn columnar_dedupe_body(raw: &str, label: &str) -> String {
    let mut bound = Bound::new();
    let mut total: u64 = 0;
    let mut unique: u64 = 0;
    let mut duplicates: u64 = 0;
    let mut seen: BTreeSet<String> = BTreeSet::new();

    for line in raw.lines() {
        total += 1;
        let capped = cap_line(line);
        let key = row_key(&capped);
        if seen.contains(&key) {
            duplicates += 1;
            continue;
        }
        if seen.len() < MAX_SEEN_KEYS {
            seen.insert(key);
        }
        unique += 1;
        bound.push_head(&capped);
    }

    let header = format!("# kind={label} lines={total} unique={unique} duplicates={duplicates}");
    bound.render(&header)
}

/// Column table for file maps / symbol sets / structured rows: first
/// [`MAX_TABLE_COLUMNS`] whitespace columns per row, tab separated, exact
/// duplicate rows collapsed.
fn column_table_body(kind: &EvidenceKind, raw: &str) -> String {
    let mut bound = Bound::new();
    let mut total: u64 = 0;
    let mut kept: u64 = 0;
    let mut duplicates: u64 = 0;
    let mut seen: BTreeSet<String> = BTreeSet::new();

    for line in raw.lines() {
        total += 1;
        let capped = cap_line(line);
        let columns: Vec<&str> = capped.split_whitespace().take(MAX_TABLE_COLUMNS).collect();
        if columns.is_empty() {
            continue;
        }
        let row = columns.join("\t");
        if seen.contains(&row) {
            duplicates += 1;
            continue;
        }
        if seen.len() < MAX_SEEN_KEYS {
            seen.insert(row.clone());
        }
        kept += 1;
        bound.push_head(&row);
    }

    let header = format!(
        "# kind={} lines={total} kept={kept} duplicates={duplicates}",
        kind_label(kind)
    );
    bound.render(&header)
}

/// Unified-diff structure: hunk headers, file headers, per-hunk added and
/// removed counts, and the first few content lines of each hunk.
fn diff_summary_body(raw: &str) -> String {
    let mut bound = Bound::new();
    let mut total: u64 = 0;
    let mut hunks: u64 = 0;
    let mut added: u64 = 0;
    let mut removed: u64 = 0;
    let mut lines_in_hunk: usize = 0;

    for line in raw.lines() {
        total += 1;
        let capped = cap_line(line);
        let text = capped.as_ref();
        if text.starts_with("@@") {
            hunks += 1;
            lines_in_hunk = 0;
            bound.push_head(text);
        } else if text.starts_with("+++")
            || text.starts_with("---")
            || text.starts_with("diff ")
            || text.starts_with("index ")
        {
            bound.push_head(text);
        } else if lines_in_hunk < FIRST_HUNK_LINES {
            lines_in_hunk += 1;
            bound.push_head(text);
        }
        if text.starts_with('+') && !text.starts_with("+++") {
            added += 1;
        } else if text.starts_with('-') && !text.starts_with("---") {
            removed += 1;
        }
    }

    let header = format!("# kind=diff lines={total} hunks={hunks} added={added} removed={removed}");
    bound.render(&header)
}

/// Generic structural summary: line count plus the first
/// [`STRUCTURAL_FIRST_LINES`] lines, each bounded.
fn first_lines_summary_body(kind: &EvidenceKind, raw: &str) -> String {
    let mut bound = Bound::new();
    let mut total: u64 = 0;

    for line in raw.lines() {
        total += 1;
        if bound.head_len() < STRUCTURAL_FIRST_LINES {
            bound.push_head(&cap_line(line));
        } else {
            bound.omit();
        }
    }

    let header = format!(
        "# kind={} lines={total} kept={}",
        kind_label(kind),
        bound.head_len()
    );
    bound.render(&header)
}

/// Bounded accumulator shared by every transform: a sized head section, a
/// sized failure section, an optional final line, and omission counters.
/// Nothing here is unbounded; `render` re-checks the global ceiling.
struct Bound {
    head: Vec<String>,
    head_bytes: usize,
    failures: Vec<String>,
    failure_bytes: usize,
    dropped_failures: u64,
    omitted: u64,
    last: Option<String>,
}

impl Bound {
    fn new() -> Self {
        Self {
            head: Vec::new(),
            head_bytes: 0,
            failures: Vec::new(),
            failure_bytes: 0,
            dropped_failures: 0,
            omitted: 0,
            last: None,
        }
    }

    fn head_len(&self) -> usize {
        self.head.len()
    }

    fn omit(&mut self) {
        self.omitted += 1;
    }

    fn push_head(&mut self, line: &str) {
        if self.head_bytes + line.len() + 1 > MAX_HEAD_BYTES {
            self.omitted += 1;
            return;
        }
        self.head_bytes += line.len() + 1;
        self.head.push(line.to_string());
    }

    fn push_failure(&mut self, line: &str) {
        if self.failure_bytes + line.len() + 1 > MAX_FAILURE_BYTES {
            self.dropped_failures += 1;
            return;
        }
        self.failure_bytes += line.len() + 1;
        self.failures.push(line.to_string());
    }

    fn observe_last(&mut self, line: &str) {
        self.last = Some(line.to_string());
    }

    fn render(&self, header: &str) -> String {
        let mut out =
            String::with_capacity(header.len() + self.head_bytes + self.failure_bytes + 256);
        out.push_str(header);
        for line in &self.head {
            out.push('\n');
            out.push_str(line);
        }
        if self.omitted > 0 {
            out.push('\n');
            out.push_str(&format!("... {} more lines omitted ...", self.omitted));
        }
        if self.dropped_failures > 0 {
            out.push('\n');
            out.push_str(&format!(
                "... {} more failure lines omitted ...",
                self.dropped_failures
            ));
        }
        for line in &self.failures {
            out.push('\n');
            out.push_str(line);
        }
        if let Some(last) = &self.last {
            let duplicated =
                self.head.iter().any(|l| l == last) || self.failures.iter().any(|l| l == last);
            if !duplicated {
                out.push('\n');
                out.push_str(last);
            }
        }
        truncate_body(out)
    }
}

/// Cap one line to [`MAX_LINE_BYTES`], appending the truncation marker on a
/// char boundary. O(cap) even for a multi-megabyte single-line input.
fn cap_line(line: &str) -> Cow<'_, str> {
    if line.len() <= MAX_LINE_BYTES {
        return Cow::Borrowed(line);
    }
    let budget = MAX_LINE_BYTES - TRUNCATION_MARKER.len();
    let end = floor_char_boundary(line, budget);
    let mut out = String::with_capacity(end + TRUNCATION_MARKER.len());
    out.push_str(&line[..end]);
    out.push_str(TRUNCATION_MARKER);
    Cow::Owned(out)
}

fn floor_char_boundary(s: &str, mut index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    while index > 0 && !s.is_char_boundary(index) {
        index -= 1;
    }
    index
}

/// Last-resort global ceiling; budgets make it unreachable, but no code path
/// may ever hand back an oversized body.
fn truncate_body(body: String) -> String {
    if body.len() <= MAX_COMPACT_BYTES {
        return body;
    }
    let mut truncated = body;
    let mut end = MAX_COMPACT_BYTES;
    while end > 0 && !truncated.is_char_boundary(end) {
        end -= 1;
    }
    truncated.truncate(end);
    truncated
}

/// Whitespace-normalized form of one line, bounded to [`NORMALIZE_CAP`]
/// bytes. Used only for equality, never emitted.
fn normalize_into(line: &str, out: &mut String) {
    out.clear();
    let mut pending_space = false;
    for ch in line.trim().chars() {
        if ch.is_whitespace() {
            if !out.is_empty() {
                pending_space = true;
            }
            continue;
        }
        if pending_space {
            if out.len() + 1 > NORMALIZE_CAP {
                break;
            }
            out.push(' ');
            pending_space = false;
        }
        if out.len() + ch.len_utf8() > NORMALIZE_CAP {
            break;
        }
        out.push(ch);
    }
}

fn normalize_line(line: &str) -> String {
    let mut out = String::new();
    normalize_into(line, &mut out);
    out
}

/// `path:line` key when the second colon column is numeric; otherwise the
/// normalized whole line. Non-UTF8-neighboring data never panics: this is
/// plain byte/char slicing on valid `&str`.
fn row_key(line: &str) -> String {
    let mut columns = line.splitn(3, ':');
    if let (Some(path), Some(line_no)) = (columns.next(), columns.next()) {
        if line_no.trim().parse::<u64>().is_ok() {
            return format!("{}:{}", path.trim(), line_no.trim());
        }
    }
    normalize_line(line)
}

/// Case-insensitive ASCII substring test. Multibyte UTF-8 continuation bytes
/// are >= 0x80 and ASCII needles are < 0x80, so byte scanning cannot match
/// inside a multibyte character.
fn contains_ci(haystack: &str, needle_lower: &str) -> bool {
    let needle = needle_lower.as_bytes();
    if needle.is_empty() {
        return true;
    }
    let hay = haystack.as_bytes();
    if hay.len() < needle.len() {
        return false;
    }
    let first = needle[0];
    for (index, &byte) in hay.iter().enumerate() {
        if index + needle.len() > hay.len() {
            break;
        }
        if byte.to_ascii_lowercase() == first
            && hay[index..index + needle.len()].eq_ignore_ascii_case(needle)
        {
            return true;
        }
    }
    false
}

/// Failure predicate: case-insensitive `error|fail|panic|assert`.
fn is_failure_line(line: &str) -> bool {
    let bytes = line.as_bytes();
    for (index, &byte) in bytes.iter().enumerate() {
        let ok = match byte.to_ascii_lowercase() {
            b'e' => starts_with_ci(&bytes[index..], b"error"),
            b'f' => starts_with_ci(&bytes[index..], b"fail"),
            b'p' => starts_with_ci(&bytes[index..], b"panic"),
            b'a' => starts_with_ci(&bytes[index..], b"assert"),
            _ => false,
        };
        if ok {
            return true;
        }
    }
    false
}

fn starts_with_ci(haystack: &[u8], needle_lower: &[u8]) -> bool {
    haystack.len() >= needle_lower.len()
        && haystack[..needle_lower.len()].eq_ignore_ascii_case(needle_lower)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Compressibility, Lossiness};

    const ALL_KINDS: [EvidenceKind; 11] = [
        EvidenceKind::DiagnosticSet,
        EvidenceKind::TestReport,
        EvidenceKind::SearchResults,
        EvidenceKind::FileMap,
        EvidenceKind::Diff,
        EvidenceKind::ProcessLog,
        EvidenceKind::SymbolSet,
        EvidenceKind::StructuredRows,
        EvidenceKind::ChildHandoff,
        EvidenceKind::SemanticContext,
        EvidenceKind::GenericText,
    ];

    fn digest(raw: &str) -> Option<[u8; 32]> {
        Some(*blake3::hash(raw.as_bytes()).as_bytes())
    }

    #[test]
    fn compress_for_delegates_to_kind_policy() {
        for kind in ALL_KINDS {
            assert_eq!(compress_for(&kind), kind.default_compressibility());
        }
        assert_eq!(
            compress_for(&EvidenceKind::ProcessLog),
            Compressibility::Aggressive
        );
        assert_eq!(
            compress_for(&EvidenceKind::Diff),
            Compressibility::Reversible
        );
        assert_eq!(
            compress_for(&EvidenceKind::GenericText),
            Compressibility::LosslessOnly
        );
    }

    #[test]
    fn lossy_compressor_rejects_never_class() {
        assert_eq!(Compressibility::for_policy_inputs(), Compressibility::Never);
        for completeness in [
            BackingCompleteness::Complete,
            BackingCompleteness::Truncated,
        ] {
            for raw in ["", "goal: do the thing", "instructions\nstep 1\n"] {
                let err = compress_with(
                    &EvidenceKind::GenericText,
                    Compressibility::Never,
                    completeness,
                    raw,
                )
                .unwrap_err();
                assert!(matches!(err, EvidenceError::Refused(_)), "{err:?}");
                assert!(err.to_string().contains("Never"), "{err}");
            }
        }
        // The only authorised Never path is the explicit identity copy.
        let (compact, record) = compress_never_identity("goal: do the thing");
        assert_eq!(compact.body, "goal: do the thing");
        assert_eq!(compact.grammar, "identity-v1");
        assert_eq!(record.lossiness, Lossiness::None);
        assert_eq!(record.backing_digest, None);
    }

    #[test]
    fn reversible_requires_complete_backing() {
        let diff = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1,3 +1,3 @@\n-a\n+b\n c\n";
        let err = compress_with(
            &EvidenceKind::Diff,
            Compressibility::Reversible,
            BackingCompleteness::Truncated,
            diff,
        )
        .unwrap_err();
        assert!(matches!(err, EvidenceError::Refused(_)), "{err:?}");
        assert!(err.to_string().contains("truncated"), "{err}");

        let (compact, record) = compress_with(
            &EvidenceKind::Diff,
            Compressibility::Reversible,
            BackingCompleteness::Complete,
            diff,
        )
        .unwrap();
        assert_eq!(compact.grammar, "diff-v1");
        assert_eq!(record.lossiness, Lossiness::Structural);
        assert_eq!(record.backing_digest, digest(diff));
        assert_eq!(record.original_bytes, diff.len() as u64);
        assert_eq!(record.compact_bytes, compact.body.len() as u64);
        assert!(compact.body.contains("@@ -1,3 +1,3 @@"), "{}", compact.body);
    }

    #[test]
    fn truncated_backing_cannot_claim_reversible() {
        let raw = "@@ -1 +1 @@\n-old\n+new\n";
        for kind in [
            EvidenceKind::Diff,
            EvidenceKind::ChildHandoff,
            EvidenceKind::SemanticContext,
        ] {
            let err = compress(&kind, BackingCompleteness::Truncated, raw).unwrap_err();
            assert!(
                matches!(err, EvidenceError::Refused(_)),
                "{kind:?}: {err:?}"
            );
        }
        // Aggressive kinds may compact truncated backing, but must still tie
        // the lossy body to the backing bytes via a digest.
        let log = "spam\nspam\nERROR: boom\n";
        let (compact, record) = compress(
            &EvidenceKind::ProcessLog,
            BackingCompleteness::Truncated,
            log,
        )
        .unwrap();
        assert_eq!(record.lossiness, Lossiness::Content);
        assert_eq!(record.backing_digest, digest(log));
        assert!(compact.body.contains("ERROR: boom"));
    }

    #[test]
    fn lossless_identity_exact() {
        let raw = "generic text\nwith  weird\t spacing\r\nand snowman \u{2603}\n";
        let (compact, record) = compress(
            &EvidenceKind::GenericText,
            BackingCompleteness::Complete,
            raw,
        )
        .unwrap();
        assert_eq!(compact.body, raw);
        assert_eq!(compact.grammar, "identity-v1");
        assert_eq!(record.algorithm, "faktor-compress-v1");
        assert_eq!(record.version, 1);
        assert_eq!(record.original_bytes, raw.len() as u64);
        assert_eq!(record.compact_bytes, raw.len() as u64);
        assert_eq!(record.lossiness, Lossiness::None);
        assert_eq!(record.backing_digest, None);

        // Truncated backing does not block an honest lossless copy.
        let (compact, record) = compress(
            &EvidenceKind::GenericText,
            BackingCompleteness::Truncated,
            raw,
        )
        .unwrap();
        assert_eq!(compact.body, raw);
        assert_eq!(record.lossiness, Lossiness::None);

        let (compact, record) = compress_never_identity(raw);
        assert_eq!(compact.body, raw);
        assert_eq!(record.original_bytes, raw.len() as u64);
        assert_eq!(record.compact_bytes, raw.len() as u64);
        assert_eq!(record.lossiness, Lossiness::None);
    }

    #[test]
    fn empty_input_ok() {
        for kind in ALL_KINDS {
            let policy = compress_for(&kind);
            let completeness = if policy == Compressibility::Reversible {
                BackingCompleteness::Complete
            } else {
                BackingCompleteness::Truncated
            };
            let (compact, record) = compress(&kind, completeness, "").unwrap();
            assert_eq!(record.original_bytes, 0, "{kind:?}");
            assert_eq!(record.compact_bytes, compact.body.len() as u64, "{kind:?}");
            assert!(compact.body.len() <= MAX_COMPACT_BYTES, "{kind:?}");
            assert!(!compact.grammar.is_empty(), "{kind:?}");
            match policy {
                Compressibility::LosslessOnly => {
                    assert!(compact.body.is_empty(), "{kind:?}");
                    assert_eq!(record.backing_digest, None, "{kind:?}");
                }
                Compressibility::Aggressive => {
                    assert_eq!(record.lossiness, Lossiness::Content, "{kind:?}");
                    assert_eq!(record.backing_digest, digest(""), "{kind:?}");
                }
                Compressibility::Reversible => {
                    assert_eq!(record.lossiness, Lossiness::Structural, "{kind:?}");
                    assert_eq!(record.backing_digest, digest(""), "{kind:?}");
                }
                Compressibility::Never => unreachable!("no kind defaults to Never"),
            }
        }
    }

    #[test]
    fn process_log_collapses_runs_and_keeps_failures() {
        let raw = "spam\nspam\nspam\nINFO ok\nboom ERROR: kaboom\nboom ERROR: kaboom\n";
        let (compact, record) = compress(
            &EvidenceKind::ProcessLog,
            BackingCompleteness::Complete,
            raw,
        )
        .unwrap();
        assert!(compact.body.contains("spam x3"), "{}", compact.body);
        assert!(compact.body.contains("INFO ok"), "{}", compact.body);
        assert!(compact.body.contains("ERROR: kaboom"), "{}", compact.body);
        assert_eq!(record.lossiness, Lossiness::Content);
        assert_eq!(record.backing_digest, digest(raw));
        assert_eq!(record.compact_bytes, compact.body.len() as u64);
    }

    #[test]
    fn test_report_keeps_failed_rows_and_counts() {
        let raw = "test a ... ok\ntest b ... ok\ntest c ... FAILED\n\
                   test result: FAILED. 2 passed; 1 failed\n";
        let (compact, _) = compress(
            &EvidenceKind::TestReport,
            BackingCompleteness::Complete,
            raw,
        )
        .unwrap();
        assert!(
            compact.body.contains("test c ... FAILED"),
            "{}",
            compact.body
        );
        assert!(compact.body.contains("passed=2"), "{}", compact.body);
        assert!(compact.body.contains("failed=2"), "{}", compact.body);
        assert!(!compact.body.contains("test a ... ok"), "{}", compact.body);
    }

    #[test]
    fn diagnostic_and_search_rows_dedupe_on_path_line() {
        let raw = "src/a.rs:1:1: error: nope\nsrc/a.rs:1:1: error: nope\n\
                   src/b.rs:2:1: warning: hmm\n";
        let (compact, _) = compress(
            &EvidenceKind::DiagnosticSet,
            BackingCompleteness::Truncated,
            raw,
        )
        .unwrap();
        assert!(compact.body.contains("unique=2"), "{}", compact.body);
        assert!(compact.body.contains("duplicates=1"), "{}", compact.body);
        assert_eq!(
            compact.body.matches("src/a.rs").count(),
            1,
            "{}",
            compact.body
        );

        let hits = "src/a.rs:10:hit one\nsrc/a.rs:10:hit one\nsrc/a.rs:11:hit two\n";
        let (compact, _) = compress(
            &EvidenceKind::SearchResults,
            BackingCompleteness::Truncated,
            hits,
        )
        .unwrap();
        assert!(compact.body.contains("unique=2"), "{}", compact.body);
        assert_eq!(
            compact.body.matches("src/a.rs:10").count(),
            1,
            "{}",
            compact.body
        );
    }

    #[test]
    fn column_table_caps_columns_and_dedupes_rows() {
        let raw = "src/a.rs 10 20 30 40 50 60 70\nsrc/a.rs 10 20 30 40 50 60 70\n\
                   src/b.rs 30 40 50 60 70 80 90\n";
        let (compact, _) =
            compress(&EvidenceKind::FileMap, BackingCompleteness::Truncated, raw).unwrap();
        assert!(compact.body.contains("kept=2"), "{}", compact.body);
        assert!(compact.body.contains("duplicates=1"), "{}", compact.body);
        assert!(
            compact.body.contains("src/a.rs\t10\t20\t30\t40\t50"),
            "{}",
            compact.body
        );
        // Only the first MAX_TABLE_COLUMNS columns are kept.
        assert!(
            !compact.body.contains("src/a.rs\t10\t20\t30\t40\t50\t60"),
            "{}",
            compact.body
        );
    }

    #[test]
    fn diff_summary_counts_hunks_and_first_lines() {
        let raw = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1,2 +1,2 @@\n-a\n+b\n c\n\
                   @@ -9 +9 @@\n-d\n+e\n";
        let (compact, record) =
            compress(&EvidenceKind::Diff, BackingCompleteness::Complete, raw).unwrap();
        assert!(compact.body.contains("hunks=2"), "{}", compact.body);
        assert!(compact.body.contains("added=2"), "{}", compact.body);
        assert!(compact.body.contains("removed=2"), "{}", compact.body);
        assert!(compact.body.contains("-a"), "{}", compact.body);
        assert_eq!(record.lossiness, Lossiness::Structural);
        assert_eq!(record.backing_digest, digest(raw));
    }

    #[test]
    fn huge_single_line_is_capped_on_a_char_boundary() {
        let huge = "e".repeat(8 * 1024 * 1024 - 1);
        let (compact, record) = compress(
            &EvidenceKind::ProcessLog,
            BackingCompleteness::Complete,
            &huge,
        )
        .unwrap();
        assert!(compact.body.len() <= MAX_COMPACT_BYTES);
        assert!(
            compact.body.contains(TRUNCATION_MARKER),
            "body was not capped"
        );
        assert_eq!(record.original_bytes, huge.len() as u64);
        assert_eq!(record.compact_bytes, compact.body.len() as u64);

        let multibyte = "é".repeat(2 * 1024 * 1024);
        let (compact, _) = compress(
            &EvidenceKind::DiagnosticSet,
            BackingCompleteness::Truncated,
            &multibyte,
        )
        .unwrap();
        assert!(compact.body.len() <= MAX_COMPACT_BYTES);
        assert!(compact.body.contains('\u{e9}'));
    }

    #[test]
    fn hostile_50mb_log_is_bounded_and_deterministic() {
        const TARGET: usize = 50 * 1024 * 1024;
        let spam = format!("spam {}\n", "x".repeat(80));
        let mut raw = String::with_capacity(TARGET + 256);
        while raw.len() < TARGET / 2 {
            raw.push_str(&spam);
        }
        raw.push_str("ERROR: explosion near the midpoint\n");
        while raw.len() < TARGET {
            raw.push_str(&spam);
        }
        raw.push_str("FATAL: final error: pipeline aborted\n");
        assert!(raw.len() >= TARGET);
        assert!(raw.len() >= 50 * 1024 * 1024);

        let (first_compact, first_record) = compress(
            &EvidenceKind::ProcessLog,
            BackingCompleteness::Complete,
            &raw,
        )
        .unwrap();
        let (second_compact, second_record) = compress(
            &EvidenceKind::ProcessLog,
            BackingCompleteness::Complete,
            &raw,
        )
        .unwrap();

        assert_eq!(first_compact, second_compact);
        assert_eq!(first_record, second_record);
        assert!(
            first_compact.body.len() <= MAX_COMPACT_BYTES,
            "compact body {} exceeds the {} byte bound",
            first_compact.body.len(),
            MAX_COMPACT_BYTES
        );
        assert!(
            first_compact.body.contains("explosion near the midpoint"),
            "midpoint failure was dropped"
        );
        assert!(
            first_compact.body.contains("final error: pipeline aborted"),
            "final error line was dropped"
        );
        assert_eq!(first_record.original_bytes, raw.len() as u64);
        assert_eq!(first_record.compact_bytes, first_compact.body.len() as u64);
        assert_eq!(first_record.lossiness, Lossiness::Content);
        assert_eq!(first_record.backing_digest, digest(&raw));
    }

    #[test]
    fn every_kind_is_deterministic_across_runs() {
        let samples = [
            (EvidenceKind::ProcessLog, "a\na\nb\nERROR: x\nc\n"),
            (
                EvidenceKind::TestReport,
                "test a ... ok\ntest b ... FAILED\n",
            ),
            (
                EvidenceKind::DiagnosticSet,
                "src/a.rs:1:2: error: nope\nsrc/a.rs:1:2: error: nope\n",
            ),
            (
                EvidenceKind::SearchResults,
                "src/a.rs:10:hit\nsrc/a.rs:10:hit\n",
            ),
            (EvidenceKind::FileMap, "src/a.rs 120\nsrc/a.rs 120\n"),
            (EvidenceKind::SymbolSet, "fn a src/a.rs:3\n"),
            (EvidenceKind::StructuredRows, "a 1 2\n"),
            (EvidenceKind::Diff, "@@ -1 +1 @@\n-old\n+new\n"),
            (EvidenceKind::ChildHandoff, "handoff\nbody\n"),
            (EvidenceKind::SemanticContext, "chunk\nbody\n"),
            (EvidenceKind::GenericText, "just text\n"),
        ];
        for (kind, raw) in samples {
            let first = compress(&kind, BackingCompleteness::Complete, raw).unwrap();
            let second = compress(&kind, BackingCompleteness::Complete, raw).unwrap();
            assert_eq!(first, second, "{kind:?} must be byte-deterministic");
            assert!(first.0.body.len() <= MAX_COMPACT_BYTES, "{kind:?}");
            assert!(first.0.grammar.ends_with("-v1"), "{kind:?}");
            assert_eq!(first.1.compact_bytes, first.0.body.len() as u64, "{kind:?}");
        }
    }
}
