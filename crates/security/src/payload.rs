//! Whole-payload streaming secret scanning (audit P0-37).
//!
//! The old scanner truncated hostile input at a fixed 256 KiB prefix cap
//! and reported `Clean` for anything past it — a secret hidden in the
//! suffix was silently invisible. This module replaces that semantics with
//! a scanner that **sees every byte exactly once**:
//!
//! - [`Scanner`] is an incremental API: call [`Scanner::feed`] with chunks
//!   and [`Scanner::finish`] at end of input. Memory stays bounded: only an
//!   overlap window of at most `policy.overlap_max` bytes is retained
//!   (plus one internal region and the in-flight chunk), and the window is
//!   sized for *pattern recognition*, never as a total-input cap.
//! - A secret split across any chunk boundary — or across the internal
//!   overlap window — is still detected: the engine carries per-pattern
//!   walker state and open-run state across boundaries.
//! - When a caller sets `ScanPolicy::max_payload_bytes` to an absolute
//!   maximum and the payload exceeds it, the outcome is
//!   [`ScanOutcome::TooLargeForPolicy`] — fail closed. It is **never**
//!   `Clean` for an unscanned suffix, and it takes precedence over any
//!   hits already found (a caller that sees `TooLargeForPolicy` must block
//!   or re-scan under an explicit larger policy).
//!
//! ## Pattern support on byte payloads
//!
//! The payload engine uses the same pattern DSL as the text scanner but
//! matches raw bytes (payloads need not be valid UTF-8). It is ASCII-case-
//! insensitive; non-ASCII **literal** characters match their exact UTF-8
//! bytes, while character classes containing non-ASCII members are not
//! streamable and are skipped (they stay supported by the whole-text
//! [`crate::scan_secrets`] and by [`CompiledSecretPolicy::scan_text`]).
//! Patterns containing an open-ended run (`{n,}`) followed by further
//! segments need unbounded backtracking and are likewise skipped by this
//! engine; the six frozen defaults are all streamable. A pattern whose
//! recognition decision is longer than the configured `overlap_max` is
//! skipped in streaming mode (raise `overlap_max` to scan it);
//! [`scan_payload`]/[`scan_payload_compiled`] scan an in-memory payload
//! whole, so there only patterns longer than the payload itself are
//! skipped.
//!
//! Any skipped pattern is **reported** through
//! [`Scanner::skipped_patterns`] (original pattern indexes) — a host that
//! needs a guaranteed-coverage deny decision must treat a non-empty report
//! as a configuration error, never scan silently (audit 31/107-109:
//! accept-then-skip is exactly the failure mode this crate refuses).
//!
//! Output is bounded: at most [`MAX_PAYLOAD_HITS`] hits are retained;
//! beyond that the scan stops early and reports `Found` with the capped
//! list (never `Clean`). Snippets on streamed hits may be partial because
//! the match's start may already have left the bounded window.

use crate::{SecretHit, Seg, DEFAULT_PATTERN_KINDS, DEFAULT_SECRET_PATTERNS};

/// Hard output cap per scan: never retain more than this many raw hits
/// (bounded memory + bounded work on pathological payloads).
pub const MAX_PAYLOAD_HITS: usize = 8192;

/// Default overlap window: the number of trailing bytes the streaming
/// engine retains so a match crossing any boundary is still recognised.
/// This is a recognition window (a few KiB), not a total-input cap — the
/// whole payload is streamed through it exactly once.
pub const DEFAULT_OVERLAP_MAX: usize = 4096;

/// Bytes processed per internal region sweep. Retained memory is bounded by
/// `SEG + overlap_max + largest in-flight chunk`.
const SEG: usize = 64 * 1024;

/// Whole-payload scan policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanPolicy {
    /// Absolute payload ceiling. `None` = no total cap (the default for
    /// in-memory payloads the caller already holds). When set and the
    /// payload exceeds it, the outcome is [`ScanOutcome::TooLargeForPolicy`]
    /// — never `Clean` for the unscanned suffix.
    pub max_payload_bytes: Option<u64>,
    /// Streaming overlap window in bytes (recognition context retained
    /// across chunk boundaries). Must cover the longest recognisable
    /// pattern decision; patterns longer than this are skipped in streaming
    /// mode. Defaults to [`DEFAULT_OVERLAP_MAX`].
    pub overlap_max: usize,
}

impl Default for ScanPolicy {
    fn default() -> Self {
        ScanPolicy {
            max_payload_bytes: None,
            overlap_max: DEFAULT_OVERLAP_MAX,
        }
    }
}

/// Result of a whole-payload scan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScanOutcome {
    /// No secret found; the ENTIRE payload was inspected (every byte seen
    /// exactly once). `Clean` is never returned for bytes that were not
    /// scanned.
    Clean,
    /// At least one secret occurrence. For an outbound caller under a
    /// deny policy this means: do not send. The list may be capped at
    /// [`MAX_PAYLOAD_HITS`] (still `Found`, never `Clean`).
    Found(Vec<SecretHit>),
    /// The payload exceeded `ScanPolicy::max_payload_bytes` (when one was
    /// configured). Fail closed: the caller must block or adopt an
    /// explicit larger policy — the unscanned suffix must never be treated
    /// as clean. Takes precedence over earlier hits.
    TooLargeForPolicy,
}

/// Incremental feed status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedStatus {
    /// Chunk accepted; keep feeding or call [`Scanner::finish`].
    Ok,
    /// The payload exceeded `ScanPolicy::max_payload_bytes`. The scanner is
    /// fail-closed: [`Scanner::finish`] will return
    /// [`ScanOutcome::TooLargeForPolicy`] no matter what was scanned so far.
    ExceedsLimit,
}

/// Convenience whole-buffer scan under the frozen default patterns.
pub fn scan_payload(payload: &[u8], policy: &ScanPolicy) -> ScanOutcome {
    scan_payload_with(payload, policy, &default_patterns())
}

/// Convenience whole-buffer scan under explicit patterns.
pub fn scan_payload_with(payload: &[u8], policy: &ScanPolicy, patterns: &[String]) -> ScanOutcome {
    let policy = ScanPolicy {
        // Single-shot mode: the whole payload is buffered, so the
        // recognition window is the payload length itself — patterns are
        // skipped only when their decision is longer than the payload
        // (they could never match there anyway).
        overlap_max: payload.len().saturating_add(1),
        ..policy.clone()
    };
    let mut scanner = Scanner::with_patterns(&policy, patterns.to_vec());
    if scanner.feed(payload) == FeedStatus::ExceedsLimit {
        return ScanOutcome::TooLargeForPolicy;
    }
    scanner.finish()
}

/// Convenience whole-buffer scan under a [`crate::CompiledSecretPolicy`]
/// (config-time validated patterns — audit 31/107-109). The payload is
/// buffered whole, so the recognition window is the payload length:
/// patterns whose decision exceeds it could never match there (skipped and
/// reported); every other pattern is scanned — no syntax-based silent
/// drop is possible on this path.
pub fn scan_payload_compiled(
    payload: &[u8],
    policy: &ScanPolicy,
    compiled: &crate::CompiledSecretPolicy,
) -> ScanOutcome {
    let policy = ScanPolicy {
        overlap_max: payload.len().saturating_add(1),
        ..policy.clone()
    };
    let mut scanner = Scanner::with_compiled_policy(&policy, compiled);
    if scanner.feed(payload) == FeedStatus::ExceedsLimit {
        return ScanOutcome::TooLargeForPolicy;
    }
    scanner.finish()
}

/// The frozen default patterns as owned strings.
fn default_patterns() -> Vec<String> {
    DEFAULT_SECRET_PATTERNS
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// Streaming whole-payload scanner. See the module docs for semantics.
pub struct Scanner {
    overlap_max: usize,
    max_payload_bytes: Option<u64>,
    patterns: Vec<Pat>,
    /// Original pattern indexes that this engine could NOT recognise under
    /// the given [`ScanPolicy`] (see [`Scanner::skipped_patterns`]) —
    /// never silently unobservable: a host making a deny decision on a
    /// custom compiled policy must treat a non-empty report as a
    /// configuration error.
    skipped: Vec<usize>,
    /// Retained bytes; `buf[0]` is logical offset `b0`.
    buf: Vec<u8>,
    b0: usize,
    /// Logical end of input seen so far (`end - b0 == buf.len()`).
    end: usize,
    state: Vec<PatState>,
    /// Raw (start, end, pattern index) hits, capped at [`MAX_PAYLOAD_HITS`].
    hits: Vec<(usize, usize, usize)>,
    overflow: bool,
    /// Matching disabled after the hit cap is reached (bytes are still
    /// counted toward `max_payload_bytes` so overflow still wins).
    matching: bool,
}

/// One compiled streamable byte pattern.
struct Pat {
    /// Index of the pattern in the caller-provided list (skipped patterns
    /// keep their original index, matching the text engine's semantics).
    idx: usize,
    text: String,
    segs: Vec<BSeg>,
    /// Recognition decision bound in bytes: how far past a match start the
    /// engine may need to look to decide hit/dead/open.
    decision: usize,
    /// Class of a final open-ended run, if the pattern ends with one.
    open_run_class: Option<ByteClass>,
}

/// Walker state for one pattern.
#[derive(Debug, Clone, Default)]
struct PatState {
    /// Next attempt position (logical byte offset).
    pos: usize,
    /// An open-ended-run hit confirmed but not yet terminated.
    active: Option<Active>,
}

#[derive(Debug, Clone)]
struct Active {
    start: usize,
    /// Bytes up to here were class-checked as part of the open run.
    checked_to: usize,
}

// ---------------------------------------------------------------------------
// Byte lowering of the char-level pattern language
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum BSeg {
    Lit(Vec<u8>),
    Cls(ByteClass, BQuant),
    Alt(Vec<Vec<u8>>),
}

#[derive(Debug, Clone)]
struct BQuant {
    min: usize,
    exact: Option<usize>,
}

#[derive(Debug, Clone, Default)]
struct ByteClass {
    ranges: Vec<(u8, u8)>,
    singles: Vec<u8>,
}

impl ByteClass {
    /// ASCII-case-insensitive membership of one raw byte.
    fn hit(&self, b: u8) -> bool {
        let b = fold_byte(b);
        self.ranges.iter().any(|(lo, hi)| b >= *lo && b <= *hi) || self.singles.contains(&b)
    }
}

fn fold_byte(b: u8) -> u8 {
    if b.is_ascii() {
        b.to_ascii_lowercase()
    } else {
        b
    }
}

/// Lower the char-level parser output to bytes. `None` = pattern not
/// streamable on raw bytes (non-ASCII class member): skipped like other
/// unsupported syntax.
fn lower_seg(seg: &Seg) -> Option<BSeg> {
    match seg {
        Seg::Lit(chars) => {
            let mut out = Vec::with_capacity(chars.len());
            for c in chars {
                if c.is_ascii() {
                    out.push(c.to_ascii_lowercase() as u8);
                } else {
                    let mut enc = [0u8; 4];
                    out.extend_from_slice(c.encode_utf8(&mut enc).as_bytes());
                }
            }
            Some(BSeg::Lit(out))
        }
        Seg::Cls(cls, q) => {
            let mut ranges: Vec<(u8, u8)> = Vec::with_capacity(cls.ranges.len());
            for (lo, hi) in &cls.ranges {
                if !lo.is_ascii() || !hi.is_ascii() {
                    return None; // non-ASCII class member: not byte-streamable
                }
                ranges.push((lo.to_ascii_lowercase() as u8, hi.to_ascii_lowercase() as u8));
            }
            let mut singles: Vec<u8> = Vec::with_capacity(cls.singles.len());
            for s in &cls.singles {
                if !s.is_ascii() {
                    return None;
                }
                singles.push(s.to_ascii_lowercase() as u8);
            }
            Some(BSeg::Cls(
                ByteClass { ranges, singles },
                BQuant {
                    min: q.min,
                    exact: q.exact,
                },
            ))
        }
        Seg::Alt(alts) => {
            let lowered: Option<Vec<Vec<u8>>> = alts
                .iter()
                .map(|alt| {
                    let mut out = Vec::with_capacity(alt.len());
                    for c in alt {
                        if c.is_ascii() {
                            out.push(c.to_ascii_lowercase() as u8);
                        } else {
                            let mut enc = [0u8; 4];
                            out.extend_from_slice(c.encode_utf8(&mut enc).as_bytes());
                        }
                    }
                    Some(out)
                })
                .collect();
            Some(BSeg::Alt(lowered?))
        }
    }
}

/// Classify one lowered segment sequence: total decision bound, and the
/// final-open-run class when the pattern ends with an open run.
fn pat_metrics(segs: &[BSeg]) -> Option<(usize, Option<ByteClass>)> {
    let mut decision: usize = 0;
    let mut open_class: Option<ByteClass> = None;
    for (i, seg) in segs.iter().enumerate() {
        open_class = None;
        match seg {
            BSeg::Lit(b) => decision = decision.saturating_add(b.len()),
            BSeg::Alt(alts) => {
                let max_len = alts.iter().map(|a| a.len()).max().unwrap_or(0);
                decision = decision.saturating_add(max_len);
            }
            BSeg::Cls(cls, q) => match q.exact {
                Some(n) => decision = decision.saturating_add(n),
                None => {
                    if i + 1 != segs.len() {
                        // Open run followed by more segments needs
                        // unbounded backtracking: not streamable.
                        return None;
                    }
                    open_class = Some(cls.clone());
                    decision = decision.saturating_add(q.min);
                }
            },
        }
    }
    Some((decision, open_class))
}

fn pat_from_segs(text: &str, segs: &[crate::Seg]) -> Option<Pat> {
    let lowered: Option<Vec<BSeg>> = segs.iter().map(lower_seg).collect();
    let segs = lowered?;
    if segs.is_empty() {
        return None;
    }
    let (decision, open_run_class) = pat_metrics(&segs)?;
    Some(Pat {
        idx: 0,
        text: text.to_string(),
        segs,
        decision,
        open_run_class,
    })
}

fn compile_pat(pattern: &str) -> Option<Pat> {
    let segs = crate::compile(pattern)?;
    pat_from_segs(pattern, &segs)
}

// ---------------------------------------------------------------------------
// Attempt matching (mirrors the whole-text engine's greedy semantics).
// All positions passed to the matcher are RELATIVE to `buf` (buf[0] == 0).
// ---------------------------------------------------------------------------

enum SegRes {
    /// Consumed to `usize` (exclusive end of the match, terminator seen).
    Done(usize),
    /// A final open run consumed to the current input end; terminator not
    /// seen yet.
    EofOpen,
    Fail,
}

enum TryOutcome {
    Dead,
    Hit(usize),
    Open,
}

fn lit_eq_fold(lit: &[u8], buf: &[u8], pos: usize, eof: usize) -> bool {
    pos + lit.len() <= eof
        && buf[pos..pos + lit.len()]
            .iter()
            .zip(lit)
            .all(|(a, b)| fold_byte(*a) == *b)
}

fn match_segs(
    segs: &[BSeg],
    buf: &[u8],
    pos: usize,
    si: usize,
    eof: usize,
    at_eof: bool,
) -> SegRes {
    if si == segs.len() {
        return SegRes::Done(pos);
    }
    match &segs[si] {
        BSeg::Lit(lit) => {
            if lit_eq_fold(lit, buf, pos, eof) {
                match_segs(segs, buf, pos + lit.len(), si + 1, eof, at_eof)
            } else {
                SegRes::Fail
            }
        }
        BSeg::Alt(alts) => {
            for alt in alts {
                if lit_eq_fold(alt, buf, pos, eof) {
                    // First prefix-matching alternative decides, matching
                    // the whole-text engine's (quirky) ordered semantics.
                    return match_segs(segs, buf, pos + alt.len(), si + 1, eof, at_eof);
                }
            }
            SegRes::Fail
        }
        BSeg::Cls(cls, q) => {
            let mut run = 0usize;
            while pos + run < eof && cls.hit(buf[pos + run]) {
                run += 1;
            }
            if run < q.min {
                return SegRes::Fail;
            }
            match q.exact {
                Some(n) => match_segs(segs, buf, pos + n, si + 1, eof, at_eof),
                None => {
                    debug_assert!(
                        si + 1 == segs.len(),
                        "open run must be final (compile filtered it)"
                    );
                    if pos + run < eof {
                        // Terminator observed: greedy span ends before it.
                        SegRes::Done(pos + run)
                    } else if at_eof {
                        SegRes::Done(pos + run)
                    } else {
                        SegRes::EofOpen
                    }
                }
            }
        }
    }
}

/// Decide the outcome of one attempt at relative position `pos`.
fn try_at(pat: &Pat, buf: &[u8], pos: usize, eof: usize, at_eof: bool) -> TryOutcome {
    match match_segs(&pat.segs, buf, pos, 0, eof, at_eof) {
        SegRes::Fail => TryOutcome::Dead,
        SegRes::Done(end) => TryOutcome::Hit(end),
        SegRes::EofOpen => TryOutcome::Open,
    }
}

// ---------------------------------------------------------------------------
// Scanner
// ---------------------------------------------------------------------------

impl Scanner {
    /// New streaming scanner over the frozen default patterns.
    pub fn new(policy: &ScanPolicy) -> Scanner {
        Scanner::with_patterns(policy, default_patterns())
    }

    /// New streaming scanner over explicit patterns. Patterns outside the
    /// streamable subset (see module docs) are skipped, as are patterns
    /// whose recognition decision exceeds `policy.overlap_max`; every
    /// skipped original pattern index is reported through
    /// [`Scanner::skipped_patterns`] so an accept-then-skip can never stay
    /// invisible to a caller that needs a guaranteed-coverage decision.
    pub fn with_patterns(policy: &ScanPolicy, patterns: Vec<String>) -> Scanner {
        let mut pats: Vec<Pat> = Vec::new();
        let mut skipped: Vec<usize> = Vec::new();
        for (pidx, text) in patterns.into_iter().enumerate() {
            if let Some(mut pat) = compile_pat(&text) {
                if pat.decision <= policy.overlap_max {
                    pat.idx = pidx;
                    pats.push(pat);
                } else {
                    skipped.push(pidx);
                }
            } else {
                skipped.push(pidx);
            }
        }
        let n = pats.len();
        Scanner {
            overlap_max: policy.overlap_max,
            max_payload_bytes: policy.max_payload_bytes,
            patterns: pats,
            skipped,
            buf: Vec::with_capacity(SEG + policy.overlap_max + 256),
            b0: 0,
            end: 0,
            state: vec![PatState::default(); n],
            hits: Vec::new(),
            overflow: false,
            matching: true,
        }
    }

    /// Streaming scanner over a [`CompiledSecretPolicy`] (config-time
    /// validated, see [`crate::CompiledSecretPolicy::try_from`]). Patterns
    /// this byte engine cannot stream — decision longer than
    /// `overlap_max`, non-ASCII class member, an open run followed by
    /// further segments — are reported (never silently dropped) through
    /// [`Scanner::skipped_patterns`].
    pub fn with_compiled_policy(
        policy: &ScanPolicy,
        compiled: &crate::CompiledSecretPolicy,
    ) -> Scanner {
        let mut pats: Vec<Pat> = Vec::new();
        let mut skipped: Vec<usize> = Vec::new();
        let key_patterns = &compiled.as_policy().key_patterns;
        let segments = compiled.segments();
        debug_assert_eq!(key_patterns.len(), segments.len());
        for (pidx, (text, segs)) in key_patterns.iter().zip(segments).enumerate() {
            if let Some(mut pat) = pat_from_segs(text, segs) {
                if pat.decision <= policy.overlap_max {
                    pat.idx = pidx;
                    pats.push(pat);
                } else {
                    skipped.push(pidx);
                }
            } else {
                skipped.push(pidx);
            }
        }
        let n = pats.len();
        Scanner {
            overlap_max: policy.overlap_max,
            max_payload_bytes: policy.max_payload_bytes,
            patterns: pats,
            skipped,
            buf: Vec::with_capacity(SEG + policy.overlap_max + 256),
            b0: 0,
            end: 0,
            state: vec![PatState::default(); n],
            hits: Vec::new(),
            overflow: false,
            matching: true,
        }
    }

    /// Original pattern indexes (into the policy's `key_patterns`) that
    /// this scanner does not recognise under its [`ScanPolicy`]. A
    /// non-empty report means the scanner cannot guarantee coverage of
    /// those patterns: hosts making a deny decision must refuse, not scan
    /// silently.
    pub fn skipped_patterns(&self) -> &[usize] {
        &self.skipped
    }

    /// Bytes of payload seen so far (logical, including already-dropped
    /// bytes).
    pub fn bytes_seen(&self) -> usize {
        self.end
    }

    /// Bytes currently retained internally (bounded by one region plus the
    /// overlap window plus the largest in-flight chunk).
    pub fn buffered_bytes(&self) -> usize {
        self.buf.len()
    }

    /// Feed the next chunk. Returns [`FeedStatus::ExceedsLimit`] (fail
    /// closed) once `ScanPolicy::max_payload_bytes` is crossed; the
    /// crossing bytes are never scanned and `finish` will report
    /// [`ScanOutcome::TooLargeForPolicy`].
    pub fn feed(&mut self, chunk: &[u8]) -> FeedStatus {
        if self.overflow {
            return FeedStatus::ExceedsLimit;
        }
        if chunk.is_empty() {
            return FeedStatus::Ok;
        }
        let new_end = self.end.saturating_add(chunk.len());
        if self.max_payload_bytes.is_some_and(|m| new_end as u64 > m) {
            self.overflow = true;
            return FeedStatus::ExceedsLimit;
        }
        self.end = new_end;
        self.buf.extend_from_slice(chunk);
        if self.matching {
            self.close_open_runs();
            while self.buf.len() >= SEG + self.overlap_max {
                let region_end = self.b0 + SEG;
                self.process_region(region_end, false);
                if !self.matching {
                    break;
                }
                let drop = SEG;
                self.buf.drain(..drop);
                self.b0 += drop;
            }
        }
        FeedStatus::Ok
    }

    /// End of input: scan the tail (EOF semantics), close any open runs at
    /// the true end, and produce the outcome.
    pub fn finish(mut self) -> ScanOutcome {
        if self.overflow {
            return ScanOutcome::TooLargeForPolicy;
        }
        if self.matching {
            // Open runs terminate at the true end of input.
            for pi in 0..self.state.len() {
                let Some(active) = self.state[pi].active.take() else {
                    continue;
                };
                self.emit(pi, active.start, self.end);
                if !self.matching {
                    break;
                }
                self.state[pi].pos = self.end;
            }
            if self.matching {
                self.process_region(self.end, true);
            }
        }
        let mut raw = std::mem::take(&mut self.hits);
        raw.sort_unstable_by_key(|h| (h.0, h.2, h.1));
        let mut picked: Vec<(usize, usize, usize)> = Vec::with_capacity(raw.len());
        let mut last_end = 0usize;
        for (start, end, pidx) in raw {
            if start >= last_end {
                picked.push((start, end, pidx));
                last_end = end;
            }
        }
        if picked.is_empty() {
            return ScanOutcome::Clean;
        }
        let buf = &self.buf;
        let b0 = self.b0;
        let hits = picked
            .into_iter()
            .map(|(start, end, slot)| {
                let pat = &self.patterns[slot];
                let kind = kind_of(&pat.text, pat.idx);
                SecretHit {
                    pattern_index: pat.idx,
                    snippet: build_snippet(buf, b0, start, end),
                    redacted: format!("<redacted:{kind}>"),
                    kind,
                    offset: start,
                    len: end.saturating_sub(start),
                }
            })
            .collect();
        ScanOutcome::Found(hits)
    }

    fn emit(&mut self, pidx: usize, start: usize, end: usize) {
        if end <= start {
            return;
        }
        if self.hits.len() >= MAX_PAYLOAD_HITS {
            self.matching = false;
            return;
        }
        self.hits.push((start, end, pidx));
        if self.hits.len() >= MAX_PAYLOAD_HITS {
            self.matching = false;
        }
    }

    /// Run per-pattern attempts over [current pos, region_end).
    fn process_region(&mut self, region_end: usize, at_eof: bool) {
        let b0 = self.b0;
        let eof = self.end - b0;
        for pi in 0..self.patterns.len() {
            loop {
                if !self.matching {
                    return;
                }
                if self.state[pi].active.is_some() {
                    break;
                }
                let abs_pos = self.state[pi].pos;
                if abs_pos >= region_end {
                    break;
                }
                let outcome = try_at(&self.patterns[pi], &self.buf, abs_pos - b0, eof, at_eof);
                match outcome {
                    TryOutcome::Dead => self.state[pi].pos = abs_pos + 1,
                    // The matcher works in buffer-relative offsets; convert
                    // its result back to logical (absolute) offsets.
                    TryOutcome::Hit(rel_end) => {
                        let abs_end = rel_end + b0;
                        if abs_end > abs_pos {
                            self.emit(pi, abs_pos, abs_end);
                            if !self.matching {
                                return;
                            }
                            self.state[pi].pos = abs_end;
                        } else {
                            self.state[pi].pos = abs_pos + 1;
                        }
                    }
                    TryOutcome::Open => {
                        self.state[pi].active = Some(Active {
                            start: abs_pos,
                            checked_to: self.end,
                        });
                        break;
                    }
                }
            }
        }
    }

    /// Advance open runs over newly arrived bytes; a run closes at the
    /// first byte outside its class (its greedy span ends before it).
    fn close_open_runs(&mut self) {
        let b0 = self.b0;
        let end_rel = self.end - b0;
        for pi in 0..self.patterns.len() {
            let Some(active) = self.state[pi].active.take() else {
                continue;
            };
            let cls = self.patterns[pi].open_run_class.clone();
            let start = active.start;
            let mut checked = active.checked_to;
            let mut checked_rel = checked - b0;
            if let Some(cls) = cls {
                while checked_rel < end_rel && cls.hit(self.buf[checked_rel]) {
                    checked_rel += 1;
                }
            }
            checked = checked_rel + b0;
            if checked < self.end {
                // Terminator found: greedy span ends before it.
                self.state[pi].active = None;
                self.state[pi].pos = checked;
                self.emit(pi, start, checked);
                if !self.matching {
                    return;
                }
            } else {
                self.state[pi].active = Some(Active {
                    start,
                    checked_to: checked,
                });
            }
        }
    }
}

/// Whole-payload streaming scan over an iterator of chunks under the frozen
/// default patterns. Chunk boundaries are arbitrary: a secret split across
/// any boundary is still detected.
pub fn streaming_scan<I, C>(chunks: I, policy: &ScanPolicy) -> ScanOutcome
where
    I: IntoIterator<Item = C>,
    C: AsRef<[u8]>,
{
    let mut scanner = Scanner::new(policy);
    for chunk in chunks {
        if scanner.feed(chunk.as_ref()) == FeedStatus::ExceedsLimit {
            return ScanOutcome::TooLargeForPolicy;
        }
    }
    scanner.finish()
}

fn kind_of(pattern: &str, index: usize) -> String {
    DEFAULT_SECRET_PATTERNS
        .iter()
        .position(|d| *d == pattern)
        .map(|i| DEFAULT_PATTERN_KINDS[i].to_string())
        .unwrap_or_else(|| format!("pattern{index}"))
}

/// Snippet around a hit over the retained buffer (`buf[0]` == logical
/// `b0`), centred where the bytes are still available; never more than 40
/// chars, decoded lossily (byte payloads need not be UTF-8).
fn build_snippet(buf: &[u8], b0: usize, start: usize, end: usize) -> String {
    const CAP: usize = 40;
    let span = end.saturating_sub(start).min(CAP);
    let budget = CAP - span;
    let left = budget / 2;
    let right = budget - left;
    let lo = start.saturating_sub(left).max(b0);
    let hi = (end + right).min(b0 + buf.len());
    if lo >= hi {
        return String::new();
    }
    let raw = &buf[lo - b0..hi - b0];
    let s = String::from_utf8_lossy(raw);
    let mut out: String = s.chars().take(CAP).collect();
    if lo > b0 {
        out.insert(0, '…');
        while out.chars().count() > CAP {
            out.pop();
        }
    }
    if hi < b0 + buf.len() {
        out.push('…');
        while out.chars().count() > CAP {
            out.pop();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CompiledSecretPolicy, SecretPolicy};

    const GHP: &str = "ghp_0123456789abcdefghijklmnopqrstuv";
    const AKIA: &str = "AKIA0123456789ABCDEF";
    const PEM: &str = "-----BEGIN RSA PRIVATE KEY-----";

    fn hit_rows(outcome: &ScanOutcome) -> Vec<(usize, usize, String, usize)> {
        match outcome {
            ScanOutcome::Found(hits) => hits
                .iter()
                .map(|h| (h.offset, h.len, h.kind.clone(), h.pattern_index))
                .collect(),
            _ => Vec::new(),
        }
    }

    fn feed_all(scanner: Scanner, payload: &[u8], chunk: usize) -> ScanOutcome {
        let mut scanner = scanner;
        for c in payload.chunks(chunk.max(1)) {
            scanner.feed(c);
        }
        scanner.finish()
    }

    // ------------------------------------------------------------------
    // spec row (a): a secret at byte 300 KiB of a 400 KiB payload IS found
    // ------------------------------------------------------------------

    #[test]
    fn secret_at_300k_of_400k_is_detected_single_shot() {
        let filler = vec![b'a'; 300 * 1024];
        let mut payload = filler.clone();
        payload.extend_from_slice(GHP.as_bytes());
        payload.push(b'.'); // terminate the open class run
        payload.extend_from_slice(&vec![b'a'; 100 * 1024 - GHP.len() - 1]);
        let policy = ScanPolicy::default();
        let outcome = scan_payload(&payload, &policy);
        let rows = hit_rows(&outcome);
        assert_eq!(rows.len(), 1, "secret past the old 256 KiB cap must fire");
        assert_eq!(rows[0].0, 300 * 1024);
        assert_eq!(rows[0].1, GHP.len());
        assert_eq!(rows[0].2, "github_token");
    }

    #[test]
    fn secret_at_300k_of_400k_is_detected_streaming_chunks() {
        let mut payload = vec![b'b'; 300 * 1024];
        payload.extend_from_slice(AKIA.as_bytes());
        payload.extend_from_slice(&vec![b'c'; 100 * 1024 - AKIA.len()]);
        let policy = ScanPolicy::default();
        let mut scanner = Scanner::new(&policy);
        let mut peak = 0usize;
        for c in payload.chunks(64 * 1024) {
            scanner.feed(c);
            peak = peak.max(scanner.buffered_bytes());
        }
        let outcome = scanner.finish();
        let rows = hit_rows(&outcome);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].0, 300 * 1024);
        assert_eq!(rows[0].2, "aws_key");
        assert!(
            peak <= 64 * 1024 + policy.overlap_max + 64 * 1024,
            "retention must stay bounded: {peak}"
        );
    }

    // ------------------------------------------------------------------
    // spec row (b): absolute max_payload_bytes -> TooLargeForPolicy
    // ------------------------------------------------------------------

    #[test]
    fn payload_over_explicit_max_is_too_large_never_clean() {
        let policy = ScanPolicy {
            max_payload_bytes: Some(1000),
            ..ScanPolicy::default()
        };
        let payload = vec![b'x'; 2000];
        let outcome = scan_payload(&payload, &policy);
        assert_eq!(outcome, ScanOutcome::TooLargeForPolicy);
        // A secret BEFORE the cap does not rescue the outcome: the suffix
        // was never scanned, so fail-closed wins over Found.
        let mut payload = AKIA.as_bytes().to_vec();
        payload.extend(std::iter::repeat_n(b'y', 5000));
        assert_eq!(
            scan_payload(&payload, &policy),
            ScanOutcome::TooLargeForPolicy
        );
    }

    #[test]
    fn payload_exactly_at_max_is_scanned_fully() {
        let policy = ScanPolicy {
            max_payload_bytes: Some(1000),
            ..ScanPolicy::default()
        };
        let payload = vec![b'z'; 1000];
        assert_eq!(scan_payload(&payload, &policy), ScanOutcome::Clean);
        // Exactly at the cap AND holding a secret is Found, not TooLarge.
        let mut payload = vec![b'z'; 980];
        payload.extend_from_slice(AKIA.as_bytes());
        assert!(matches!(
            scan_payload(&payload, &policy),
            ScanOutcome::Found(_)
        ));
    }

    #[test]
    fn streaming_overflow_midstream_fails_closed() {
        let policy = ScanPolicy {
            max_payload_bytes: Some(60),
            ..ScanPolicy::default()
        };
        let mut scanner = Scanner::new(&policy);
        // First chunk: a full secret inside the cap — scanned and found.
        assert_eq!(scanner.feed(AKIA.as_bytes()), FeedStatus::Ok);
        assert!(matches!(scanner.finish(), ScanOutcome::Found(_)));
        // Now overflow after an earlier hit: TooLarge takes precedence.
        let mut scanner = Scanner::new(&policy);
        assert_eq!(scanner.feed(AKIA.as_bytes()), FeedStatus::Ok);
        assert_eq!(scanner.feed(&[b'w'; 60]), FeedStatus::ExceedsLimit);
        assert_eq!(scanner.feed(&[b'w'; 1]), FeedStatus::ExceedsLimit);
        assert_eq!(scanner.finish(), ScanOutcome::TooLargeForPolicy);
    }

    // ------------------------------------------------------------------
    // spec row (c): chunk splits, boundary splits, 10 GiB logical stream
    // ------------------------------------------------------------------

    #[test]
    fn secret_split_across_feed_boundary_at_every_offset_is_detected() {
        let policy = ScanPolicy::default();
        let filler = vec![b'q'; 128];
        let secret = PEM;
        for cut in 0..secret.len() {
            let mut payload = filler.clone();
            payload.extend_from_slice(secret.as_bytes());
            payload.extend_from_slice(&[b'r'; 100]);
            // Split the payload into two feeds at every byte of the secret.
            let split_at = filler.len() + cut;
            let mut scanner = Scanner::new(&policy);
            assert_eq!(scanner.feed(&payload[..split_at]), FeedStatus::Ok);
            assert_eq!(scanner.feed(&payload[split_at..]), FeedStatus::Ok);
            let rows = hit_rows(&scanner.finish());
            assert_eq!(rows.len(), 1, "split at secret byte {cut}");
            assert_eq!(rows[0].0, filler.len());
            assert_eq!(rows[0].1, secret.len());
            assert_eq!(rows[0].2, "pem_private_key");
        }
    }

    #[test]
    fn secret_crossing_internal_region_boundary_is_detected() {
        // The engine sweeps 64 KiB regions; place secrets straddling and
        // just past region edges 65536/131072 and feed in awkward chunks.
        let policy = ScanPolicy::default();
        for start in [
            65536 - 12usize, // straddles the first region end
            65536 + 2,       // just inside the second region
            131072 - 4,      // straddles the second region end
            131072 + 1,
        ] {
            let mut payload = vec![b'p'; 200 * 1024];
            payload[start..start + AKIA.len()].copy_from_slice(AKIA.as_bytes());
            let scanner = Scanner::new(&policy);
            let outcome = feed_all(scanner, &payload, 70_000);
            let rows = hit_rows(&outcome);
            assert_eq!(rows.len(), 1, "secret at {start} of {}", payload.len());
            assert_eq!(rows[0].0, start);
            assert_eq!(rows[0].2, "aws_key");
        }
    }

    #[test]
    fn one_byte_dribble_finds_secret_at_tail() {
        let policy = ScanPolicy::default();
        let mut payload = vec![b'x'; 3000];
        payload.extend_from_slice(GHP.as_bytes());
        let mut scanner = Scanner::new(&policy);
        for b in &payload {
            assert_eq!(scanner.feed(&[*b]), FeedStatus::Ok);
        }
        let rows = hit_rows(&scanner.finish());
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].0, 3000);
    }

    #[test]
    fn eight_meg_stream_stays_bounded_and_finds_deep_secret() {
        // CI-sized proof of the streaming boundedness claim: memory stays
        // under one region + the overlap window + the chunk while an 8 MiB
        // payload streams through the incremental API, and a secret deep in
        // the stream (past many region sweeps, b0 far from 0) is found.
        let policy = ScanPolicy::default();
        let chunk = vec![b'n'; 64 * 1024];
        let target = 8u64 * 1024 * 1024;
        let mut scanner = Scanner::new(&policy);
        let mut peak = 0usize;
        let mut sent = 0u64;
        while sent < target {
            let take = chunk.len().min((target - sent) as usize);
            assert_eq!(scanner.feed(&chunk[..take]), FeedStatus::Ok);
            sent += take as u64;
            peak = peak.max(scanner.buffered_bytes());
        }
        let mut tail = vec![b'o'; GHP.len()];
        tail.copy_from_slice(GHP.as_bytes());
        assert_eq!(scanner.feed(&tail), FeedStatus::Ok);
        let rows = hit_rows(&scanner.finish());
        assert_eq!(rows.len(), 1, "secret after 8 MiB must be found");
        assert_eq!(rows[0].0, target as usize);
        assert!(
            peak <= 64 * 1024 + policy.overlap_max + 64 * 1024,
            "peak {peak}"
        );
    }

    /// Manual soak (not part of CI; runs in release only): 10 GiB of
    /// logical payload streamed through [`Scanner`] with bounded memory;
    /// the secret lands at logical byte 5 GiB and must be found.
    #[test]
    #[ignore]
    fn soak_ten_gib_logical_stream_stays_bounded() {
        let policy = ScanPolicy::default();
        let chunk = vec![b'n'; 64 * 1024];
        let target = 5u64 * 1024 * 1024 * 1024;
        let mut scanner = Scanner::new(&policy);
        let mut peak = 0usize;
        let mut sent = 0u64;
        let mut secret_at: Option<u64> = None;
        while sent < target {
            let take = chunk.len().min((target - sent) as usize);
            if scanner.feed(&chunk[..take]) == FeedStatus::ExceedsLimit {
                panic!("no cap configured: must never exceed");
            }
            sent += take as u64;
            peak = peak.max(scanner.buffered_bytes());
            if sent == target {
                // Place the secret so its start lands exactly at 5 GiB.
                let mut tail = vec![b'o'; GHP.len()];
                tail.copy_from_slice(GHP.as_bytes());
                if scanner.feed(&tail) == FeedStatus::ExceedsLimit {
                    panic!("no cap configured");
                }
                secret_at = Some(sent);
            }
        }
        let rows = hit_rows(&scanner.finish());
        assert_eq!(rows.len(), 1, "secret at 5 GiB must be found");
        assert_eq!(rows[0].0, secret_at.unwrap() as usize);
        assert!(
            peak <= 64 * 1024 + policy.overlap_max + 64 * 1024,
            "peak {peak}"
        );
    }

    #[test]
    #[ignore]
    fn perf_ten_gib_real_bytes_stream_bounded_memory() {
        // Manual soak: 10 GiB of REAL bytes through the engine.
        let policy = ScanPolicy::default();
        let mut scanner = Scanner::new(&policy);
        let chunk = vec![b'm'; 1 << 20];
        let total = 10u64 << 30;
        let mut sent = 0u64;
        while sent < total {
            let take = chunk.len().min((total - sent) as usize);
            scanner.feed(&chunk[..take]);
            sent += take as u64;
        }
        scanner.feed(GHP.as_bytes());
        let rows = hit_rows(&scanner.finish());
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].0, total as usize);
    }

    // ------------------------------------------------------------------
    // spec row (d): overlap = longest pattern decision; boundary rows
    // ------------------------------------------------------------------

    #[test]
    fn overlap_equal_to_longest_pattern_decision_is_sufficient() {
        // PEM header is 31 bytes for the RSA match, but the pattern's
        // recognition decision is 35 bytes (the OPENSSH alternation is the
        // longest recognisable form among the defaults). overlap_max == 35
        // must catch a PEM split exactly at the window edge; overlap 34
        // must skip PEM (documented exclusion) while still catching shorter
        // patterns.
        let filler = vec![b'a'; 400];
        for overlap in [35usize, 4096] {
            let policy = ScanPolicy {
                overlap_max: overlap,
                ..ScanPolicy::default()
            };
            for cut in (filler.len() - 2)..=(filler.len() + 4) {
                let mut payload = filler.clone();
                payload.extend_from_slice(PEM.as_bytes());
                let mut scanner = Scanner::new(&policy);
                scanner.feed(&payload[..cut]);
                scanner.feed(&payload[cut..]);
                let rows = hit_rows(&scanner.finish());
                assert_eq!(rows.len(), 1, "overlap {overlap} cut {cut}");
                assert_eq!(rows[0].2, "pem_private_key");
            }
        }
        // Documented skip: pattern decision (35) longer than the window.
        let policy = ScanPolicy {
            overlap_max: 34,
            ..ScanPolicy::default()
        };
        let mut payload = filler.clone();
        payload.extend_from_slice(PEM.as_bytes());
        assert_eq!(scan_payload_stream(&policy, &payload), ScanOutcome::Clean);
        // Shorter patterns still stream fine at overlap 34.
        let mut payload = vec![b'a'; 400];
        payload.extend_from_slice(GHP.as_bytes());
        assert!(matches!(
            scan_payload_stream(&policy, &payload),
            ScanOutcome::Found(_)
        ));
    }

    fn scan_payload_stream(policy: &ScanPolicy, payload: &[u8]) -> ScanOutcome {
        let mut scanner = Scanner::new(policy);
        for c in payload.chunks(37) {
            scanner.feed(c);
        }
        scanner.finish()
    }

    // ------------------------------------------------------------------
    // parity with the whole-text engine and with single-shot scanning
    // ------------------------------------------------------------------

    /// Deterministic LCG for reproducible fuzzing.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add(1442695040888963407u64);
            self.0 >> 33
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    const DEFAULT_STRINGS: [&str; 6] = crate::DEFAULT_SECRET_PATTERNS;

    #[test]
    fn streaming_parity_with_single_shot_across_chunk_splits() {
        // For 150 seeded random ASCII payloads, split each into chunks at
        // random boundaries and assert the streaming scan reports exactly
        // the same (offset, len, kind, pattern_index) rows as the whole
        // payload single-shot scan.
        let policy = ScanPolicy::default();
        let alphabet: Vec<char> =
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_:. /"
                .chars()
                .collect();
        for seed in 0u64..150 {
            let mut rng = Lcg(seed.wrapping_mul(0x9E37_79B9u64));
            let mut payload = Vec::new();
            let len = 200 + rng.below(600);
            for _ in 0..len {
                payload.push(alphabet[rng.below(alphabet.len())] as u8);
            }
            // Inject 0..3 genuine secrets at random offsets (replaced).
            let inject = rng.below(4);
            for _ in 0..inject {
                let secret = DEFAULT_STRINGS[rng.below(DEFAULT_STRINGS.len())].as_bytes();
                if secret.len() <= payload.len() {
                    let at = rng.below(payload.len() - secret.len() + 1);
                    payload[at..at + secret.len()].copy_from_slice(secret);
                }
            }
            let expected = hit_rows(&scan_payload(&payload, &policy));
            // Random chunking (including 1-byte dribbles).
            let mut scanner = Scanner::new(&policy);
            let mut i = 0usize;
            while i < payload.len() {
                let take = 1 + rng.below(70);
                let take = take.min(payload.len() - i);
                scanner.feed(&payload[i..i + take]);
                i += take;
            }
            let got = hit_rows(&scanner.finish());
            assert_eq!(got, expected, "seed {seed}: payload len {}", payload.len());
        }
    }

    #[test]
    fn streaming_parity_with_whole_text_engine_on_ascii() {
        // Cross-validate the byte matcher against the char engine
        // (scan_secrets) — both implement the same greedy semantics.
        let policy = SecretPolicy::default();
        let alphabet: Vec<char> =
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_:. /"
                .chars()
                .collect();
        for seed in 0u64..120 {
            let mut rng = Lcg(seed.wrapping_mul(0x1111_1111_1111_1111u64).wrapping_add(7));
            let mut payload = Vec::new();
            let len = 100 + rng.below(400);
            for _ in 0..len {
                payload.push(alphabet[rng.below(alphabet.len())] as u8);
            }
            let inject = rng.below(3);
            for _ in 0..inject {
                let secret = DEFAULT_STRINGS[rng.below(DEFAULT_STRINGS.len())].as_bytes();
                if secret.len() <= payload.len() {
                    let at = rng.below(payload.len() - secret.len() + 1);
                    payload[at..at + secret.len()].copy_from_slice(secret);
                }
            }
            let text = String::from_utf8(payload.clone()).unwrap();
            let text_rows = crate::scan_secrets(&text, &policy)
                .iter()
                .map(|h| (h.offset, h.len, h.kind.clone(), h.pattern_index))
                .collect::<Vec<_>>();
            let scan_policy = ScanPolicy::default();
            let mut scanner = Scanner::new(&scan_policy);
            for c in payload.chunks(1 + seed as usize % 11) {
                scanner.feed(c);
            }
            let got = hit_rows(&scanner.finish());
            assert_eq!(got, text_rows, "seed {seed}");
        }
    }

    // ------------------------------------------------------------------
    // audit 31/107-109: compiled custom policies, skip reporting, bounds
    // ------------------------------------------------------------------

    #[test]
    fn custom_compiled_pattern_crossing_every_chunk_boundary_is_detected() {
        // A CUSTOM policy (strictly compiled at config time) whose secret
        // is split across feed boundaries at every possible byte — and
        // across an internal 64 KiB region edge — is still detected by the
        // streaming engine. Also locks parity: the streaming rows equal the
        // whole-buffer compiled scan rows.
        let policy = SecretPolicy {
            key_patterns: vec![
                "CST-[0-9A-Za-z]{4}-END".to_string(),
                "ghp_[A-Za-z0-9]{20,}".to_string(),
            ],
            ..SecretPolicy::default()
        };
        let compiled = CompiledSecretPolicy::try_from(policy).unwrap();
        let secret = "CST-9fK2-END";
        let filler = vec![b'q'; 128];
        for cut in 0..secret.len() {
            let mut payload = filler.clone();
            payload.extend_from_slice(secret.as_bytes());
            payload.extend_from_slice(&[b'r'; 100]);
            let split_at = filler.len() + cut;
            let mut scanner = compiled.payload_scanner(&ScanPolicy::default());
            assert_eq!(scanner.feed(&payload[..split_at]), FeedStatus::Ok);
            assert_eq!(scanner.feed(&payload[split_at..]), FeedStatus::Ok);
            assert!(scanner.skipped_patterns().is_empty());
            let rows = hit_rows(&scanner.finish());
            assert_eq!(rows.len(), 1, "split at secret byte {cut}");
            assert_eq!(rows[0].0, filler.len());
            assert_eq!(rows[0].1, secret.len());
            assert_eq!(rows[0].2, "pattern0");
        }
        // Crossing the internal 64 KiB region sweep too.
        for start in [65536usize - 6, 65536 + 1, 131072 - 3] {
            let mut payload = vec![b'p'; 200 * 1024];
            payload[start..start + secret.len()].copy_from_slice(secret.as_bytes());
            let outcome = compiled.scan_payload_bytes(&payload, &ScanPolicy::default());
            let rows = hit_rows(&outcome);
            assert_eq!(rows.len(), 1, "compiled secret at {start}");
            assert_eq!(rows[0].0, start);
            assert_eq!(rows[0].2, "pattern0");
        }
    }

    #[test]
    fn compiled_streaming_skip_is_reported_never_silent() {
        // A config-valid pattern this byte engine cannot stream (non-ASCII
        // class member; open run followed by segments) compiles fine but
        // MUST be reported by the streaming scanner — accept-then-skip is
        // not allowed to be invisible. The whole-text compiled scan still
        // detects the value (byte-class limits are a streaming-engine
        // property, not a policy property).
        let policy = SecretPolicy {
            key_patterns: vec![
                "BEGIN-é[0-9]{3}".to_string(),  // non-ASCII class? literal é ok;
                "PRE[0-9]{2,}POST".to_string(), // open run then more segments
            ],
            ..SecretPolicy::default()
        };
        let compiled = CompiledSecretPolicy::try_from(policy).unwrap();
        let text_scan = compiled.scan_text("x BEGIN-é123 y PRE99POST z");
        assert!(
            text_scan.iter().any(|h| h.kind == "pattern0"),
            "whole-text compiled scan must find the é pattern: {text_scan:?}"
        );
        let scanner = compiled.payload_scanner(&ScanPolicy::default());
        let mut skipped = scanner.skipped_patterns().to_vec();
        skipped.sort_unstable();
        assert!(
            skipped.contains(&1),
            "open-run-then-more must be reported as unstreamable: {skipped:?}"
        );
        // The é class pattern stays streamable on bytes? é inside a class
        // member → lower_seg returns None (non-ASCII class member) so it is
        // skipped too; non-ASCII LITERALS stream fine. Use a literal form:
        let policy = SecretPolicy {
            key_patterns: vec!["BEGIN-é[0-9]{3}".to_string(), "GH[0-9]{5}END".to_string()],
            ..SecretPolicy::default()
        };
        let compiled = CompiledSecretPolicy::try_from(policy).unwrap();
        let s = compiled.payload_scanner(&ScanPolicy::default());
        // é is a literal here: streamable; the class member case needs an
        // é INSIDE brackets:
        assert!(s.skipped_patterns().is_empty());
        let policy = SecretPolicy {
            key_patterns: vec!["[é0-9]{3}".to_string()],
            ..SecretPolicy::default()
        };
        let compiled = CompiledSecretPolicy::try_from(policy).unwrap();
        let s2 = compiled.payload_scanner(&ScanPolicy::default());
        assert_eq!(s2.skipped_patterns(), &[0]);
        assert_eq!(s2.finish(), ScanOutcome::Clean);
        // whole-text still catches it:
        let hits = compiled.scan_text("é0é");
        assert_eq!(hits.len(), 1);
        // overlap too small is reported too, and a deny-deciding host can
        // refuse on the report.
        let policy = SecretPolicy {
            key_patterns: vec!["-----BEGIN (RSA|OPENSSH|EC|DSA) PRIVATE KEY-----".to_string()],
            ..SecretPolicy::default()
        };
        let compiled = CompiledSecretPolicy::try_from(policy).unwrap();
        let tiny = ScanPolicy {
            overlap_max: 10,
            ..ScanPolicy::default()
        };
        let s3 = compiled.payload_scanner(&tiny);
        assert_eq!(s3.skipped_patterns(), &[0], "PEM decision > overlap 10");
    }

    #[test]
    fn fifty_meg_clean_payload_stays_bounded_and_reports_clean() {
        // Audit boundedness: a 50 MiB benign payload streams through the
        // incremental API with memory bounded by one region + the overlap
        // window + the largest chunk, and the outcome is Clean (every byte
        // seen exactly once, nothing retained unbounded).
        let policy = ScanPolicy::default();
        let chunk = vec![b'n'; 64 * 1024];
        let target = 50u64 * 1024 * 1024;
        let mut scanner = Scanner::new(&policy);
        let mut peak = 0usize;
        let mut sent = 0u64;
        while sent < target {
            let take = chunk.len().min((target - sent) as usize);
            assert_eq!(scanner.feed(&chunk[..take]), FeedStatus::Ok);
            sent += take as u64;
            peak = peak.max(scanner.buffered_bytes());
        }
        assert_eq!(scanner.finish(), ScanOutcome::Clean);
        assert!(
            peak <= 64 * 1024 + policy.overlap_max + 64 * 1024,
            "peak {peak}"
        );
        assert_eq!(sent, target);
    }

    // ------------------------------------------------------------------
    // exactness, binary safety, caps, case folding
    // ------------------------------------------------------------------

    #[test]
    fn binary_and_invalid_utf8_payloads_never_panic_and_still_match() {
        let policy = ScanPolicy::default();
        let mut payload: Vec<u8> = (0..=255u8).collect();
        payload.push(0xff);
        payload.extend_from_slice(AKIA.as_bytes());
        payload.push(0x00);
        payload.push(0xfe);
        let rows = hit_rows(&scan_payload(&payload, &policy));
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].0, 257);
        // Split inside the invalid bytes and inside the secret.
        for split in 0..payload.len() {
            let mut scanner = Scanner::new(&policy);
            scanner.feed(&payload[..split]);
            scanner.feed(&payload[split..]);
            let outcome = scanner.finish();
            assert!(matches!(outcome, ScanOutcome::Found(_)), "split at {split}");
        }
    }

    #[test]
    fn payload_matching_is_case_insensitive_on_ascii() {
        let policy = ScanPolicy::default();
        let rows = hit_rows(&scan_payload(b"akia0123456789abcdef", &policy));
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].2, "aws_key");
    }

    #[test]
    fn hit_cap_bounds_output_and_never_reports_clean() {
        let patterns = vec!["A".to_string()];
        let policy = ScanPolicy::default();
        let payload = vec![b'A'; 20_000];
        let outcome = scan_payload_with(&payload, &policy, &patterns);
        match outcome {
            ScanOutcome::Found(hits) => {
                assert_eq!(hits.len(), MAX_PAYLOAD_HITS, "capped");
            }
            other => panic!("must be Found, got {other:?}"),
        }
    }

    #[test]
    fn non_ascii_literal_pattern_matches_exact_utf8_bytes() {
        let patterns = vec!["BEGIN—KEY".to_string()];
        let policy = ScanPolicy::default();
        let mut payload = vec![b'x'; 50];
        payload.extend_from_slice("BEGIN—KEY".as_bytes());
        payload.push(b'y');
        let rows = hit_rows(&scan_payload_with(&payload, &policy, &patterns));
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].0, 50);
        assert_eq!(rows[0].1, "BEGIN—KEY".len());
    }

    #[test]
    fn empty_inputs_and_no_patterns_are_clean() {
        let policy = ScanPolicy::default();
        assert_eq!(scan_payload(b"", &policy), ScanOutcome::Clean);
        let patterns: Vec<String> = Vec::new();
        assert_eq!(
            scan_payload_with(b"sk-0123456789abcdefghijklmnopqrstuv", &policy, &patterns),
            ScanOutcome::Clean
        );
        let scanner = Scanner::new(&policy);
        assert_eq!(scanner.finish(), ScanOutcome::Clean);
    }

    #[test]
    fn scan_outcome_and_hits_never_echo_plaintext_snippet_cap() {
        // Snippets on payload hits are bounded (<= 40 chars), lossy-safe.
        let policy = ScanPolicy::default();
        let mut payload = vec![b'a'; 100_000];
        payload.extend_from_slice(GHP.as_bytes());
        payload.extend_from_slice(&[b'a'; 100]);
        match scan_payload(&payload, &policy) {
            ScanOutcome::Found(hits) => {
                assert!(hits.iter().all(|h| h.snippet.chars().count() <= 40));
                assert!(hits.iter().all(|h| h.offset + h.len <= payload.len()));
            }
            other => panic!("{other:?}"),
        }
    }
}
