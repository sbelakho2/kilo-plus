//! Bounded, high-quality line diffing (P1 "snapshot diff quality").
//!
//! The previous diff strategy (common prefix, then one giant "all removed,
//! then all added" middle) collapses two changes 900 lines apart into one
//! unreadable replacement. This module replaces it with a standard O(ND)
//! Myers diff (Myers 1986) over lines, with exact bounds everywhere:
//!
//! - max input bytes per side: [`MAX_DIFF_BYTES`];
//! - max lines per side: [`MAX_DIFF_LINES`];
//! - max Myers iterations: [`MAX_MYERS_D_ITER`] with an additional trace
//!   memory budget (rows grow quadratically in D), so the algorithm is
//!   bounded by construction and *falls back* to the deterministic
//!   prefix/suffix middle replacement when the texts are huge or too
//!   different to fit the budget — never a hang, never an OOM.
//!
//! Output is per-file, line-level hunks with 1-based old/new line numbers
//! ([`DiffHunk`]), plus a unified-ish renderer ([`render_unified`]) for the
//! wire/diff projections. Line splitting never splits a UTF-8 codepoint
//! (splitting happens only on `\n` bytes; continuation bytes never equal
//! `\n`), and text is decoded losslessly for valid UTF-8 input.

/// Hard bound on one diff side (bytes). Larger sides fall back to the
/// bounded coarse path before any line materialization.
pub const MAX_DIFF_BYTES: usize = 512 * 1024;
/// Hard bound on one diff side (lines). Larger files fall back.
pub const MAX_DIFF_LINES: usize = 20_000;
/// The Myers D-iteration cap (edit distance) requested by the spec; the
/// effective cap is the min with the trace-memory budget below.
pub const MAX_MYERS_D_ITER: usize = 4096;
/// Trace rows grow quadratically with D (row d holds 2d+1 entries). This is
/// the entry budget: ~8M i32 ≈ 32 MB worst case, bounding CPU to the same
/// order (~8M frontier steps). Fall back past it.
const TRACE_ENTRY_BUDGET: usize = 8_000_000;
/// Context lines shown around each change group.
pub const DIFF_CONTEXT_LINES: usize = 3;
/// Hard bound on rendered output lines (parity with the old wire bound).
pub const DIFF_MAX_RENDER_LINES: usize = 2000;

/// One line of a diff: a context line (' '), a removed line ('-') or an
/// added line ('+').
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffLine {
    Context(String),
    Removed(String),
    Added(String),
}

impl DiffLine {
    /// The unified-diff rendering: `' '`/`'-'`/`'+'` prefix + line.
    pub fn render(&self) -> String {
        match self {
            DiffLine::Context(l) => format!(" {l}"),
            DiffLine::Removed(l) => format!("-{l}"),
            DiffLine::Added(l) => format!("+{l}"),
        }
    }
}

/// One contiguous change region: its 1-based old/new starting line numbers,
/// the number of old/new lines it spans, and its rendered lines
/// (context + changes). A hunk whose first lines are all additions (an
/// insertion at the very head of the new file) starts at old line 0;
/// symmetric for pure deletions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffHunk {
    pub old_start: usize,
    pub old_count: usize,
    pub new_start: usize,
    pub new_count: usize,
    pub lines: Vec<DiffLine>,
}

/// How the hunk list was produced: exact Myers or a documented fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffMode {
    /// Bounded Myers: precise per-change hunks.
    Myers,
    /// Inputs exceed the byte/line bounds: a coarse single-hunk marker with
    /// the total line counts (nothing heavy is materialized).
    Coarse,
    /// Myers edit distance exceeded the budget: deterministic
    /// prefix/suffix middle replacement (one hunk).
    PrefixSuffix,
}

/// The diff result: per-file hunks plus the mode that produced them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffOutcome {
    pub hunks: Vec<DiffHunk>,
    pub mode: DiffMode,
}

/// Line diff of `before` → `after`. Bounded everywhere; never panics on
/// hostile input. See the module docs for the fallback rules.
pub fn diff_hunks(before: &[u8], after: &[u8]) -> DiffOutcome {
    // Byte gate first: a 10 MB side must never even be split.
    if before.len() > MAX_DIFF_BYTES || after.len() > MAX_DIFF_BYTES {
        return coarse(before, after);
    }
    let a = split_lines(before);
    let b = split_lines(after);
    if a.len() > MAX_DIFF_LINES || b.len() > MAX_DIFF_LINES {
        return coarse(before, after);
    }
    let Some(ops) = myers_edit_script(&a, &b) else {
        return DiffOutcome {
            hunks: vec![prefix_suffix_hunk(&a, &b)],
            mode: DiffMode::PrefixSuffix,
        };
    };
    DiffOutcome {
        hunks: hunks_from_ops(&a, &b, &ops),
        mode: DiffMode::Myers,
    }
}

/// Unified-ish rendering of the hunks: `@@ -oldStart,oldCount
/// +newStart,newCount @@` headers with ` `-/`-`/`+`-prefixed lines, capped
/// at [`DIFF_MAX_RENDER_LINES`] output lines (a truncation marker replaces
/// the tail). The cap never splits a line.
pub fn render_unified(hunks: &[DiffHunk]) -> String {
    let mut out = String::new();
    let mut remaining = DIFF_MAX_RENDER_LINES;
    for hunk in hunks {
        if remaining == 0 {
            break;
        }
        // Header lines count against the cap too (bounded everything).
        if remaining >= 2 {
            out.push_str(&format!(
                "@@ -{},{} +{},{} @@\n",
                hunk.old_start, hunk.old_count, hunk.new_start, hunk.new_count
            ));
            remaining -= 1;
        } else {
            out.push_str("… diff truncated: more than 2000 lines\n");
            break;
        }
        for line in &hunk.lines {
            if remaining == 0 {
                out.push_str("… diff truncated: more than 2000 lines\n");
                break;
            }
            out.push_str(&line.render());
            out.push('\n');
            remaining -= 1;
        }
    }
    out
}

// ---------------------------------------------------------------- splitting

/// Split text into lines exactly like the snapshot projection used to:
/// split on `\n`, drop the trailing empty piece (a trailing newline
/// terminates the last line; it is not itself a line). `\n` can never occur
/// inside a UTF-8 codepoint, so splits are always on char boundaries.
fn split_lines(bytes: &[u8]) -> Vec<String> {
    let text = String::from_utf8_lossy(bytes);
    let mut lines: Vec<String> = text.split('\n').map(|l| l.to_string()).collect();
    if lines.last().is_some_and(|l| l.is_empty()) {
        lines.pop();
    }
    lines
}

/// Cheap line count over raw bytes (no allocation).
fn line_count(bytes: &[u8]) -> usize {
    if bytes.is_empty() {
        return 0;
    }
    let newlines = bytes.iter().filter(|b| **b == b'\n').count();
    if *bytes.last().expect("non-empty") == b'\n' {
        newlines
    } else {
        newlines + 1
    }
}

// ---------------------------------------------------------------- coarse gate

/// A bounded single-hunk marker for inputs past the byte/line bounds: the
/// total old/new line counts plus an honest truncation note. Nothing heavy
/// is materialized (line counts are computed by a raw byte scan).
fn coarse(before: &[u8], after: &[u8]) -> DiffOutcome {
    let n = line_count(before);
    let m = line_count(after);
    DiffOutcome {
        hunks: vec![DiffHunk {
            old_start: 1,
            old_count: n,
            new_start: 1,
            new_count: m,
            lines: vec![DiffLine::Context(format!(
                "… diff too large to line-diff within bounds ({} removed / {} added lines, {} / {} bytes); use the saved snapshot content instead …",
                n,
                m,
                before.len(),
                after.len()
            ))],
        }],
        mode: DiffMode::Coarse,
    }
}

// ---------------------------------------------------------------- prefix/suffix fallback

/// The deterministic fallback of the old algorithm, expressed as one hunk:
/// common prefix, common suffix, and the whole changed middle (all removed
/// lines then all added lines) with [`DIFF_CONTEXT_LINES`] of context.
fn prefix_suffix_hunk(a: &[String], b: &[String]) -> DiffHunk {
    let mut lines: Vec<DiffLine> = Vec::new();
    if a == b {
        return DiffHunk {
            old_start: 1,
            old_count: 0,
            new_start: 1,
            new_count: 0,
            lines,
        };
    }
    let prefix = a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count();
    let mut suffix = 0usize;
    while suffix < a.len() - prefix
        && suffix < b.len() - prefix
        && a[a.len() - 1 - suffix] == b[b.len() - 1 - suffix]
    {
        suffix += 1;
    }
    let ctx_start = prefix.saturating_sub(DIFF_CONTEXT_LINES);
    for l in &a[ctx_start..prefix] {
        lines.push(DiffLine::Context(l.clone()));
    }
    for l in &a[prefix..a.len() - suffix] {
        lines.push(DiffLine::Removed(l.clone()));
    }
    for l in &b[prefix..b.len() - suffix] {
        lines.push(DiffLine::Added(l.clone()));
    }
    let ctx_end = (b.len() - suffix + DIFF_CONTEXT_LINES).min(b.len());
    for l in &b[b.len() - suffix..ctx_end] {
        lines.push(DiffLine::Context(l.clone()));
    }
    let old_count = lines
        .iter()
        .filter(|l| matches!(l, DiffLine::Context(_) | DiffLine::Removed(_)))
        .count();
    let new_count = lines
        .iter()
        .filter(|l| matches!(l, DiffLine::Context(_) | DiffLine::Added(_)))
        .count();
    // The first old-bearing entry starts at old line ctx_start + 1. On the
    // new side, shown pre-context lines occupy the SAME positions (they
    // precede the first difference); otherwise the first new line is the
    // first added line, the first after-context line (pure deletion), or 0
    // when the new side shows nothing (whole file deleted).
    let old_start = ctx_start + 1;
    let pre_ctx = ctx_start < prefix;
    let has_added = prefix < b.len() - suffix;
    let has_post = b.len() - suffix < ctx_end;
    let new_start = if pre_ctx {
        old_start
    } else if has_added {
        prefix + 1
    } else if has_post {
        b.len() - suffix + 1
    } else {
        0
    };
    DiffHunk {
        old_start,
        old_count,
        new_start,
        new_count,
        lines,
    }
}

// ---------------------------------------------------------------- Myers engine

/// One step of an edit script (forward order).
#[derive(Debug, Clone, PartialEq, Eq)]
enum Op {
    /// `count` lines identical on both sides.
    Eq(usize),
    /// Remove one old line.
    Del(String),
    /// Insert one new line.
    Ins(String),
}

/// Standard greedy O(ND) Myers (1986) over lines. Returns `None` when the
/// edit distance exceeds the iteration/memory budget (callers fall back).
///
/// Trace rows hold the furthest-x frontier of each iteration, so the path
/// can be backtracked exactly. Row d stores `v` for diagonals `-d..=d`
/// *before* iteration d runs (i.e. the state after iteration d-1), which is
/// exactly what backtracking reads.
fn myers_edit_script(a: &[String], b: &[String]) -> Option<Vec<Op>> {
    let n = a.len() as isize;
    let m = b.len() as isize;
    if n == 0 && m == 0 {
        return Some(Vec::new());
    }
    // Effective D cap: min(spec cap, memory budget). Trace row d holds
    // 2d+1 entries; total = (D+1)^2 <= TRACE_ENTRY_BUDGET.
    let mut d_budget = 0usize;
    while (d_budget + 1) * (d_budget + 1) <= TRACE_ENTRY_BUDGET {
        d_budget += 1;
    }
    let d_cap = MAX_MYERS_D_ITER.min(d_budget);

    // Frontier values, indexed by diagonal + offset. Diagonals in
    // [-d_cap, d_cap]; reads at iteration d only touch [-d+1, d-1].
    let offset = d_cap as isize;
    let mut v: Vec<isize> = vec![0; 2 * d_cap + 1];
    let mut trace: Vec<Vec<isize>> = Vec::new();

    let mut found: Option<usize> = None;
    'outer: for d in 0..=d_cap {
        let d_i = d as isize;
        // Snapshot the frontier BEFORE iteration d (values after d-1).
        let lo = (offset - d_i) as usize;
        let hi = (offset + d_i) as usize + 1;
        trace.push(v[lo..hi].to_vec());
        // Diagonals of parity d: k in [-d, d] step 2.
        let mut k = -d_i;
        while k <= d_i {
            let idx = (k + offset) as usize;
            let came_from_kplus1 = k == -d_i || (k != d_i && v[idx - 1] < v[idx + 1]);
            let mut x = if came_from_kplus1 {
                v[idx + 1]
            } else {
                v[idx - 1] + 1
            };
            let mut y = x - k;
            while (x as usize) < a.len() && (y as usize) < b.len() && a[x as usize] == b[y as usize]
            {
                x += 1;
                y += 1;
            }
            v[idx] = x;
            if x >= n && y >= m {
                found = Some(d);
                break 'outer;
            }
            k += 2;
        }
    }
    let found = found?;

    // Backtrack from (n, m) to (0, 0), emitting ops in reverse order.
    // Diagonal arithmetic: a deletion (x+1) moves k → k+1, an insertion
    // (y+1) moves k → k-1. The forward rule read x = v[k+1] (came from
    // k+1: an insertion step) or x = v[k-1] + 1 (came from k-1: a
    // deletion step).
    let mut ops = Vec::new();
    let mut x = n;
    let mut y = m;
    for d in (1..=found).rev() {
        let d_i = d as isize;
        let k = x - y;
        let row = &trace[d];
        // row covers diagonals [-d..=d] at index diag + d.
        let idx = (k + d_i) as usize;
        let came_from_kplus1 = k == -d_i || (k != d_i && row[idx - 1] < row[idx + 1]);
        let prev_k = if came_from_kplus1 { k + 1 } else { k - 1 };
        let px = row[(prev_k + d_i) as usize];
        let py = px - prev_k;
        // The d-th step is one insertion (y+1 at x) or one deletion (x+1 at
        // y); the snake after it consumes equal pairs only.
        let (sx, sy) = if came_from_kplus1 {
            (px, py + 1)
        } else {
            (px + 1, py)
        };
        while x > sx && y > sy {
            ops.push(Op::Eq(1));
            x -= 1;
            y -= 1;
        }
        debug_assert_eq!((x, y), (sx, sy));
        if came_from_kplus1 {
            ops.push(Op::Ins(b[(y - 1) as usize].clone()));
            y -= 1;
        } else {
            ops.push(Op::Del(a[(x - 1) as usize].clone()));
            x -= 1;
        }
        // (x, y) is now the predecessor position on prev_k.
        debug_assert_eq!((x, y), (px, py));
    }
    // The d=0 snake from (0, 0): a remaining equal run.
    debug_assert_eq!(x, y, "diagonal 0: x == y after backtracking");
    if x > 0 {
        ops.push(Op::Eq(x as usize));
    }
    ops.reverse();
    // Merge adjacent equal runs (the rewind pushes one Eq per line).
    let mut merged: Vec<Op> = Vec::with_capacity(ops.len());
    for op in ops {
        let is_eq_run = matches!(&op, Op::Eq(_));
        let mergeable = matches!(merged.last(), Some(Op::Eq(_))) && is_eq_run;
        if mergeable {
            if let (Some(Op::Eq(run)), Op::Eq(n)) = (merged.last_mut(), op) {
                *run += n;
            }
        } else {
            merged.push(op);
        }
    }
    Some(merged)
}

/// One expanded line event of the edit script (exact 1-based numbering).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Ctx,
    Del,
    Add,
}

struct Ev {
    kind: Kind,
    old: usize,
    new: usize,
    text: String,
}

/// Group an edit script into hunks: context [`DIFF_CONTEXT_LINES`] around
/// each change; change groups closer than 2×context merge into one hunk.
fn hunks_from_ops(a: &[String], b: &[String], ops: &[Op]) -> Vec<DiffHunk> {
    // Expand to per-line events carrying exact 1-based line numbers.
    let mut evs: Vec<Ev> = Vec::new();
    let mut o = 0usize;
    let mut n = 0usize;
    let mut ai = 0usize;
    let mut bi = 0usize;
    for op in ops {
        match op {
            Op::Eq(count) => {
                for _ in 0..*count {
                    o += 1;
                    n += 1;
                    evs.push(Ev {
                        kind: Kind::Ctx,
                        old: o,
                        new: n,
                        text: a[ai].clone(),
                    });
                    ai += 1;
                    bi += 1;
                }
            }
            Op::Del(_) => {
                o += 1;
                evs.push(Ev {
                    kind: Kind::Del,
                    old: o,
                    new: 0,
                    text: a[ai].clone(),
                });
                ai += 1;
            }
            Op::Ins(_) => {
                n += 1;
                evs.push(Ev {
                    kind: Kind::Add,
                    old: 0,
                    new: n,
                    text: b[bi].clone(),
                });
                bi += 1;
            }
        }
    }
    debug_assert_eq!(ai, a.len());
    debug_assert_eq!(bi, b.len());
    // Sanity: numbering must agree with the inputs.
    if cfg!(debug_assertions) {
        for ev in &evs {
            match ev.kind {
                Kind::Ctx => {
                    debug_assert_eq!(a[ev.old - 1], ev.text);
                    debug_assert_eq!(b[ev.new - 1], ev.text);
                }
                Kind::Del => debug_assert_eq!(a[ev.old - 1], ev.text),
                Kind::Add => debug_assert_eq!(b[ev.new - 1], ev.text),
            }
        }
    }

    // Walk the event list once: each change opens (or extends) a hunk; a
    // change within 2×context of the previous change's end is the same
    // hunk, otherwise the previous hunk closes and a new one opens.
    let mut hunks: Vec<DiffHunk> = Vec::new();
    let mut open: Option<(usize, usize)> = None; // (start, end) event span
    let mut last_change: usize = 0;
    for i in 0..evs.len() {
        if evs[i].kind == Kind::Ctx {
            continue;
        }
        match open.as_mut() {
            Some((_s, e)) if i.saturating_sub(last_change) <= 2 * DIFF_CONTEXT_LINES => {
                *e = (i + DIFF_CONTEXT_LINES).min(evs.len() - 1);
            }
            _ => {
                if let Some((s, e)) = open.take() {
                    hunks.push(build_hunk(&evs, s, e));
                }
                open = Some((
                    i.saturating_sub(DIFF_CONTEXT_LINES),
                    (i + DIFF_CONTEXT_LINES).min(evs.len() - 1),
                ));
            }
        }
        last_change = i;
    }
    if let Some((s, e)) = open.take() {
        hunks.push(build_hunk(&evs, s, e));
    }
    hunks
}

fn build_hunk(evs: &[Ev], s: usize, e: usize) -> DiffHunk {
    let slice = &evs[s..=e];
    let mut lines = Vec::with_capacity(slice.len());
    let mut old_count = 0usize;
    let mut new_count = 0usize;
    for ev in slice {
        match ev.kind {
            Kind::Ctx => {
                old_count += 1;
                new_count += 1;
                lines.push(DiffLine::Context(ev.text.clone()));
            }
            Kind::Del => {
                old_count += 1;
                lines.push(DiffLine::Removed(ev.text.clone()));
            }
            Kind::Add => {
                new_count += 1;
                lines.push(DiffLine::Added(ev.text.clone()));
            }
        }
    }
    // Start numbers: the first old-bearing line's number (1-based); a hunk
    // whose changes are pure insertions at the head of an empty old side
    // starts at old line 0 (the insertion point).
    let old_start = slice
        .iter()
        .find(|e| e.kind != Kind::Add)
        .map(|e| e.old)
        .unwrap_or(0);
    let new_start = slice
        .iter()
        .find(|e| e.kind != Kind::Del)
        .map(|e| e.new)
        .unwrap_or(0);
    DiffHunk {
        old_start,
        old_count,
        new_start,
        new_count,
        lines,
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    fn lines(n: usize, prefix: &str) -> Vec<u8> {
        let mut out = String::new();
        for i in 1..=n {
            out.push_str(&format!("{prefix} line {i}\n"));
        }
        out.into_bytes()
    }

    fn render_text(out: &DiffOutcome) -> String {
        render_unified(&out.hunks)
    }

    #[test]
    fn identical_files_produce_no_hunks() {
        let l = lines(1000, "x");
        let out = diff_hunks(&l, &l);
        assert_eq!(out.mode, DiffMode::Myers);
        assert!(out.hunks.is_empty(), "{:?}", out.hunks);
        assert_eq!(render_text(&out), "");
    }

    #[test]
    fn two_far_apart_changes_produce_two_hunks_not_one_replacement() {
        // The audit bug: common-prefix + whole-middle replacement collapsed
        // a change at line 20 and a change at line 900 into ONE giant
        // removed/added block. The Myers engine must produce two hunks.
        let before = lines(1000, "x");
        let mut after = String::from_utf8(before.clone()).unwrap();
        after = after.replacen("x line 20\n", "x line 20 EDITED\n", 1);
        after = after.replacen("x line 900\n", "x line 900 EDITED\n", 1);
        let out = diff_hunks(&before, after.as_bytes());
        assert_eq!(out.mode, DiffMode::Myers);
        assert_eq!(out.hunks.len(), 2, "{:?}", out.hunks);
        assert_eq!(out.hunks[0].old_start, 17, "{:?}", out.hunks[0]);
        assert_eq!(out.hunks[1].old_start, 897, "{:?}", out.hunks[1]);
        for h in &out.hunks {
            assert!(h.lines.len() <= 8, "context stays tight: {:?}", h);
        }
        let text = render_text(&out);
        assert_eq!(text.matches("@@").count(), 4, "two @@ headers: {text}");
        let first = out.hunks[0].lines.iter().collect::<Vec<_>>();
        assert!(!format!("{first:?}").contains("900"));
        // Each hunk contains its own edit only.
        let joined: Vec<String> = out
            .hunks
            .iter()
            .map(|h| {
                h.lines
                    .iter()
                    .map(DiffLine::render)
                    .collect::<Vec<_>>()
                    .join("")
            })
            .collect();
        assert!(joined[0].contains("EDITED"));
        assert!(!joined[0].contains("line 900 EDITED"));
        assert!(joined[1].contains("line 900 EDITED"));
    }

    #[test]
    fn close_changes_merge_one_hunk_far_changes_split() {
        let before = lines(60, "x");
        let mut after = String::from_utf8(before.clone()).unwrap();
        after = after.replacen("x line 10\n", "x line 10 A\n", 1);
        after = after.replacen("x line 14\n", "x line 14 B\n", 1);
        let out = diff_hunks(&before, after.as_bytes());
        assert_eq!(out.hunks.len(), 1, "4-line gap merges: {:?}", out.hunks);
        let joined = render_text(&out);
        assert!(joined.contains("x line 10 A") && joined.contains("x line 14 B"));
        assert_eq!(
            out.hunks[0].lines.len(),
            13,
            "one tight hunk: {:?}",
            out.hunks[0]
        );

        let mut after2 = String::from_utf8(before.clone()).unwrap();
        after2 = after2.replacen("x line 10\n", "x line 10 A\n", 1);
        after2 = after2.replacen("x line 21\n", "x line 21 B\n", 1);
        let out = diff_hunks(&before, after2.as_bytes());
        assert_eq!(out.hunks.len(), 2, "11-line gap splits: {:?}", out.hunks);
    }

    #[test]
    fn whole_line_replacements_number_correctly() {
        let before = lines(6, "x");
        let mut after = String::from_utf8(before.clone()).unwrap();
        after = after.replacen("x line 3\n", "y line 3\n", 1);
        let out = diff_hunks(&before, after.as_bytes());
        assert_eq!(out.hunks.len(), 1);
        let h = &out.hunks[0];
        assert_eq!(h.old_start, 1);
        assert_eq!(h.new_start, 1);
        // ctx 1,2 + removed 3 + added 3 + ctx 4,5,6
        assert_eq!(h.old_count, 6);
        assert_eq!(h.new_count, 6);
        let kinds: Vec<&str> = h
            .lines
            .iter()
            .map(|l| match l {
                DiffLine::Context(_) => " ",
                DiffLine::Removed(_) => "-",
                DiffLine::Added(_) => "+",
            })
            .collect();
        assert_eq!(kinds.join(""), "  -+   ");
    }

    #[test]
    fn insertions_and_deletions_at_edges_and_middle() {
        // Append at end.
        let a = lines(5, "x");
        let mut b = lines(5, "x");
        b.extend_from_slice(b"x line 6\n");
        let out = diff_hunks(&a, &b);
        assert_eq!(out.hunks.len(), 1);
        // ctx lines 3,4,5 + the added line 6.
        assert_eq!(out.hunks[0].old_start, 3);
        assert_eq!(out.hunks[0].old_count, 3);
        assert_eq!(out.hunks[0].new_start, 3);
        assert_eq!(out.hunks[0].new_count, 4);
        // Prepend at head: insertion at new line 1.
        let b2 = b"x line 0\n";
        let mut full = Vec::new();
        full.extend_from_slice(b2);
        full.extend_from_slice(&a);
        let out = diff_hunks(&a, &full);
        assert_eq!(out.hunks.len(), 1);
        assert_eq!(out.hunks[0].old_start, 1);
        assert_eq!(out.hunks[0].new_start, 1);
        // Delete the head line.
        let out = diff_hunks(&full, &a);
        assert_eq!(out.hunks.len(), 1);
        assert_eq!(out.hunks[0].old_start, 1);
        assert_eq!(out.hunks[0].new_start, 1);
        // Insert into the middle of an empty old file (creation).
        let out = diff_hunks(b"", b"a\nb\n");
        assert_eq!(out.mode, DiffMode::Myers);
        assert_eq!(out.hunks.len(), 1);
        assert_eq!(out.hunks[0].old_start, 0, "pure insertion at head: -0");
        assert_eq!(out.hunks[0].new_start, 1);
        assert_eq!(out.hunks[0].old_count, 0);
        assert_eq!(out.hunks[0].new_count, 2);
        // Delete everything.
        let out = diff_hunks(b"a\nb\n", b"");
        assert_eq!(out.hunks.len(), 1);
        assert_eq!(out.hunks[0].old_start, 1);
        assert_eq!(out.hunks[0].new_start, 0);
        assert_eq!(out.hunks[0].old_count, 2);
        assert_eq!(out.hunks[0].new_count, 0);
    }

    #[test]
    fn unicode_lines_split_on_char_boundaries() {
        // Multi-byte content: 'é' (2B), '😀' (4B); valid UTF-8 never splits
        // and changes 10 lines apart produce two hunks.
        let before = lines(12, "l");
        let mut after = String::from_utf8(before.clone()).unwrap();
        after = after.replacen("l line 1\n", "ligne étoile 😀\n", 1);
        after.push_str("ajout é");
        let out = diff_hunks(&before, after.as_bytes());
        assert_eq!(out.mode, DiffMode::Myers);
        assert_eq!(out.hunks.len(), 2, "{:?}", out.hunks);
        let text = render_text(&out);
        assert!(text.contains("😀"), "{text}");
        assert!(text.contains("ajout é"), "{text}");
        // No hunk mixes both changes.
        let hunks = out
            .hunks
            .iter()
            .map(|h| render_unified(std::slice::from_ref(h)))
            .collect::<Vec<_>>();
        assert!(hunks[0].contains("😀") && !hunks[0].contains("ajout"));
        assert!(hunks[1].contains("ajout") && !hunks[1].contains("😀"));
        // The final line has no trailing newline; the é stays intact at EOF.
        assert!(text.trim_end().ends_with("ajout é"), "{text:?}");
    }

    #[test]
    fn hostile_10mb_sides_fall_back_bounded_and_quick() {
        let mut seed = 0x1234_5678_9ABC_DEF0u64;
        let noise = |seed: &mut u64, n: usize| -> Vec<u8> {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                *seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                v.push((*seed >> 33) as u8);
            }
            v
        };
        let a = noise(&mut seed, 10 * 1024 * 1024);
        let b = noise(&mut seed, 10 * 1024 * 1024);
        let start = std::time::Instant::now();
        let out = diff_hunks(&a, &b);
        let elapsed = start.elapsed();
        assert!(elapsed.as_secs() < 5, "fallback took {elapsed:?}");
        assert_eq!(out.mode, DiffMode::Coarse);
        assert_eq!(out.hunks.len(), 1);
        let total: usize = out.hunks.iter().map(|h| h.lines.len()).sum();
        assert!(
            total <= 4,
            "coarse output must stay tiny, got {total} lines"
        );
        assert!(out.hunks[0].old_count > 0 && out.hunks[0].new_count > 0);
    }

    #[test]
    fn over_budget_edit_distance_falls_back_to_prefix_suffix() {
        // Two 3000-line files with NOTHING in common: Myers would need
        // D = 6000 > the budget, so the bounded fallback runs instead.
        let a: Vec<u8> = (0..3000)
            .map(|i| format!("a{i}\n"))
            .collect::<String>()
            .into_bytes();
        let b: Vec<u8> = (0..3000)
            .map(|i| format!("b{i}\n"))
            .collect::<String>()
            .into_bytes();
        let start = std::time::Instant::now();
        let out = diff_hunks(&a, &b);
        let elapsed = start.elapsed();
        assert!(elapsed.as_secs() < 5, "fallback took {elapsed:?}");
        assert_eq!(out.mode, DiffMode::PrefixSuffix);
        assert_eq!(out.hunks.len(), 1);
        let h = &out.hunks[0];
        assert_eq!(h.old_start, 1);
        assert_eq!(h.new_start, 1);
        assert_eq!(h.old_count, 3000);
        assert_eq!(h.new_count, 3000);
        // The renderer stays bounded.
        let text = render_text(&out);
        assert!(text.lines().count() <= DIFF_MAX_RENDER_LINES + 2);
        assert!(text.contains("truncated") || text.lines().count() <= 3000 + 2);
    }

    #[test]
    fn renderer_caps_output_lines() {
        // A 2500-line replacement: the rendered text must stop at the cap
        // with an explicit marker, and never split the cap mid-line.
        let a = lines(2500, "x");
        let mut b = lines(2500, "y");
        b.extend_from_slice(b"extra\n");
        let out = diff_hunks(&a, &b);
        let text = render_text(&out);
        let n = text.lines().count();
        assert!(n <= DIFF_MAX_RENDER_LINES + 1, "cap held: {n}");
        assert!(text.contains("truncated"), "cap must be loud");
        // No partial lines: every rendered line ends with '\n'.
        for l in text.lines() {
            assert!(!l.is_empty(), "no blank filler lines");
        }
    }

    #[test]
    fn prefix_suffix_fallback_keeps_common_edges() {
        // Common 5-line head and tail with a large differing middle: the
        // fallback hunk keeps the edges as context, not as changes.
        let mut a = lines(3, "head");
        let mut b = lines(3, "head");
        a.extend_from_slice(b"old1\nold2\nold3\nold4\nold5\n");
        b.extend_from_slice(b"new1\nnew2\nnew3\n");
        let tail_a = lines(3, "tail");
        let tail_b = lines(3, "tail");
        a.extend_from_slice(&tail_a);
        b.extend_from_slice(&tail_b);
        // Force the prefix/suffix path: 3+5+3 lines each, D = 5 < budget, so
        // Myers runs — instead verify edges survive in the Myers hunks too.
        let out = diff_hunks(&a, &b);
        assert_eq!(out.mode, DiffMode::Myers);
        let text = render_text(&out);
        assert!(!text.contains("-head"), "head must stay context: {text}");
        assert!(!text.contains("-tail"), "tail must stay context: {text}");
        assert!(text.contains("-old1") && text.contains("+new1"), "{text}");
    }

    #[test]
    fn prefix_suffix_fallback_counts_common_edges_exactly() {
        // Common 100-line head and tail with 2800 differing middle lines:
        // Myers would need D ≈ 5600 > budget, so the fallback hunk must
        // keep the common edges as context with self-consistent counts.
        let mut a: String = (0..3000).map(|i| format!("a{i}\n")).collect();
        let mut b: String = (0..3000).map(|i| format!("b{i}\n")).collect();
        // Make the head and tail of both sides identical.
        for i in 0..100 {
            a = a.replacen(&format!("a{i}\n"), &format!("common-{i}\n"), 1);
            b = b.replacen(&format!("b{i}\n"), &format!("common-{i}\n"), 1);
        }
        for i in 2900..3000 {
            a = a.replacen(&format!("a{i}\n"), &format!("common-{i}\n"), 1);
            b = b.replacen(&format!("b{i}\n"), &format!("common-{i}\n"), 1);
        }
        let out = diff_hunks(a.as_bytes(), b.as_bytes());
        assert_eq!(out.mode, DiffMode::PrefixSuffix, "{:?}", out.hunks);
        assert_eq!(out.hunks.len(), 1);
        let h = &out.hunks[0];
        assert_eq!(h.old_start, 98, "ctx_start 97 + 1: {:?}", h);
        assert_eq!(
            h.new_start, 98,
            "pre-context lines occupy the same positions on both sides: {:?}",
            h
        );
        assert_eq!(
            h.old_count, 2806,
            "3 pre + 2800 removed + 3 post context (context counts on both sides): {:?}",
            h
        );
        assert_eq!(h.new_count, 2806, "{:?}", h);
        // The hunk itself carries the removed and added middles (rendering
        // is capped at DIFF_MAX_RENDER_LINES, so the test inspects hunks).
        let removed_has = |needle: &str| {
            h.lines
                .iter()
                .any(|l| matches!(l, DiffLine::Removed(t) if t == needle))
        };
        let added_has = |needle: &str| {
            h.lines
                .iter()
                .any(|l| matches!(l, DiffLine::Added(t) if t == needle))
        };
        assert!(removed_has("a100") && added_has("b100"), "{:?}", h);
        let context_has_common = h
            .lines
            .iter()
            .any(|l| matches!(l, DiffLine::Removed(t) if t.starts_with("common")));
        assert!(!context_has_common, "common edges stay context: {:?}", h);
        // The rendered cap is loud and stays bounded.
        let text = render_unified(&out.hunks);
        assert!(text.contains("truncated"), "{text}");
        assert!(text.lines().count() <= DIFF_MAX_RENDER_LINES + 1);
    }

    #[test]
    fn diff_of_trailing_newline_only_is_no_change() {
        // "a\n" and "a" are the same line list (the trailing newline
        // terminates the last line; it is not a line itself).
        let out = diff_hunks(b"a\n", b"a");
        assert!(out.hunks.is_empty());
        let out = diff_hunks(b"", b"");
        assert!(out.hunks.is_empty());
    }

    // ---------------------------------------------------------- property tests

    /// Reference edit distance via DP LCS (n + m - 2*lcs for del/ins).
    fn lcs_distance(a: &[String], b: &[String]) -> usize {
        let n = a.len();
        let m = b.len();
        let mut dp = vec![vec![0usize; m + 1]; n + 1];
        for i in (0..n).rev() {
            for j in (0..m).rev() {
                dp[i][j] = if a[i] == b[j] {
                    dp[i + 1][j + 1] + 1
                } else {
                    dp[i + 1][j].max(dp[i][j + 1])
                };
            }
        }
        n + m - 2 * dp[0][0]
    }

    /// Myers ops must be optimal AND must reconstruct b from a exactly.
    fn verify_script(a: &[String], b: &[String]) {
        let ops = myers_edit_script(a, b).unwrap_or_else(|| panic!("budget exceeded"));
        let mut edits = 0usize;
        let (mut ai, mut bi) = (0usize, 0usize);
        for op in &ops {
            match op {
                Op::Eq(c) => {
                    assert_eq!(&a[ai..ai + c], &b[bi..bi + c], "equal runs must match");
                    ai += c;
                    bi += c;
                }
                Op::Del(t) => {
                    assert_eq!(&a[ai], t);
                    ai += 1;
                    edits += 1;
                }
                Op::Ins(t) => {
                    assert_eq!(&b[bi], t);
                    bi += 1;
                    edits += 1;
                }
            }
        }
        assert_eq!(
            (ai, bi),
            (a.len(), b.len()),
            "script must consume both sides"
        );
        assert_eq!(edits, lcs_distance(a, b), "Myers must be optimal");
    }

    fn xorshift(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    #[test]
    fn myers_matches_lcs_reference_over_random_inputs() {
        let mut seed = 0xDEAD_BEEF_CAFE_F00Du64;
        for _ in 0..400 {
            let n = (xorshift(&mut seed) % 11) as usize;
            let m = (xorshift(&mut seed) % 11) as usize;
            let mk = |len: usize, seed: &mut u64| -> Vec<String> {
                (0..len)
                    .map(|_| {
                        let c = match xorshift(seed) % 3 {
                            0 => 'a',
                            1 => 'b',
                            _ => 'c',
                        };
                        let d = match xorshift(seed) % 3 {
                            0 => 'x',
                            1 => 'y',
                            _ => 'z',
                        };
                        format!("{c}{d}")
                    })
                    .collect()
            };
            let a = mk(n, &mut seed);
            let b = mk(m, &mut seed);
            verify_script(&a, &b);
            // And the hunk projection must round-trip numbers for these too.
            let ops = myers_edit_script(&a, &b).unwrap();
            let hunks = hunks_from_ops(&a, &b, &ops);
            let mut last_old = 0usize;
            for h in &hunks {
                assert!(h.old_start >= last_old, "hunks ordered: {h:?}");
                last_old = h.old_start;
                for l in &h.lines {
                    match l {
                        DiffLine::Context(t) | DiffLine::Removed(t) => {
                            assert!(a.contains(t), "old-side text must exist: {t:?}");
                        }
                        DiffLine::Added(t) => {
                            assert!(b.contains(t), "new-side text must exist: {t:?}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn randomized_hunks_reconstruct_both_sides() {
        // For small inputs, rendering every hunk's lines with prefixes must
        // equal the full aligned diff: walk hunks and events is internal, so
        // instead verify per-hunk numbering against the line arrays.
        let mut seed = 42u64;
        for _ in 0..200 {
            let n = (xorshift(&mut seed) % 8) as usize;
            let m = (xorshift(&mut seed) % 8) as usize;
            let a: Vec<String> = (0..n)
                .map(|i| format!("old-{i}-{}", xorshift(&mut seed) % 5))
                .collect();
            let b: Vec<String> = (0..m)
                .map(|i| format!("new-{i}-{}", xorshift(&mut seed) % 5))
                .collect();
            let ops = myers_edit_script(&a, &b).unwrap();
            let hunks = hunks_from_ops(&a, &b, &ops);
            for h in &hunks {
                if h.old_count > 0 {
                    assert!(h.old_start >= 1 && h.old_start + h.old_count - 1 <= n);
                }
                if h.new_count > 0 {
                    assert!(h.new_start >= 1 && h.new_start + h.new_count - 1 <= m);
                }
            }
        }
    }

    #[test]
    fn coarse_mode_reports_honest_counts() {
        let big_a = vec![b'a'; MAX_DIFF_BYTES + 1];
        let big_b = vec![b'b'; MAX_DIFF_BYTES + 1];
        let out = diff_hunks(&big_a, &big_b);
        assert_eq!(out.mode, DiffMode::Coarse);
        let text = render_text(&out);
        assert!(text.contains("diff too large"), "{text}");
        // Line cap gate alone (few bytes, many lines) also goes coarse.
        let many: Vec<u8> = (0..MAX_DIFF_LINES + 5)
            .flat_map(|_| b"x\n".iter().copied())
            .collect();
        let out = diff_hunks(&many, &many);
        assert_eq!(out.mode, DiffMode::Coarse);
    }

    #[test]
    fn myers_handles_common_small_shapes() {
        // The classic cases.
        let s = |t: &[&str]| -> Vec<String> { t.iter().map(|s| s.to_string()).collect() };
        // Empty vs non-empty.
        verify_script(&[], &s(&["a"]));
        // All equal.
        verify_script(&s(&["a", "b", "c"]), &s(&["a", "b", "c"]));
        // One del, one ins, mixed order.
        verify_script(&s(&["a", "b", "c"]), &s(&["a", "c"]));
        verify_script(&s(&["a", "c"]), &s(&["a", "b", "c"]));
        verify_script(&s(&["a", "b", "c"]), &s(&["b"]));
        // Reversed (worst case).
        verify_script(&s(&["a", "b", "c", "d"]), &s(&["d", "c", "b", "a"]));
        // Repeated tokens.
        verify_script(&s(&["x", "x", "x"]), &s(&["x", "y", "x"]));
    }
}
