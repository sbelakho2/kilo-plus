//! Time-based stall vs progress (spec §28) — DISTINCT from loop detection
//! (`crate::loop_detect`). Loop detection stops REPEATING identical calls;
//! stall detection answers "is the runtime making progress at all?".
//!
//! A stall is: **no output AND no progress AND no pending-work heartbeat
//! AND (no in-flight op OR the in-flight op is itself stuck)**. A
//! long-running legitimate op that emits periodic progress updates
//! (heartbeats: tool events, op completions, iteration completions) must
//! NEVER be marked stalled — only true silence may.
//!
//! # Evidence ≠ activity (P0-79)
//!
//! A tool event occurring does NOT mean the task is closer to completion —
//! an op can churn tools forever without moving its task (same compiler
//! failure, patch/revert/patch, the same evidence set re-retrieved). The
//! tracker therefore carries TWO notions of alive:
//!
//! - **activity** (wave-15): output, heartbeats/tool events, op
//!   completions. This is the pure-silence predicate ([`StallTracker::check`])
//!   that guards *stuck streams* — an op streaming chunks is never dead.
//! - **semantic evidence** (P0-79): only the [`ProgressEvidence`] classes —
//!   criterion status changed, failure fingerprint changed, repo state
//!   changed, new evidence admitted, plan step completed, verification
//!   improved. [`StallTracker::check_evidence`] consults BOTH the evidence
//!   log and durable output/op-completion stamps (but never bare
//!   heartbeats) — it is the predicate for *expensive-cycle decisions*
//!   (each iteration costs a model call). An op emitting only tool events
//!   for 2x the stall threshold WITHOUT any evidence class stalls; one
//!   emitting an evidence class every 30 s never stalls, even when output
//!   is silent.
//!
//! The record is bounded: constant memory per session (no unbounded event
//! history is retained): the time stamps, a 32-entry evidence ring, a
//! 256-path repo-digest LRU, a 32-entry criteria-status map and a 32-entry
//! failure-fingerprint map, plus the in-flight op's start stamps.
//!
//! Time source: all callers pass `now_ms` (the runtime's injectable
//! [`faktor_core::time::Clock`]). The tracker MONOTONICIZES the reading
//! (`max(now, last_seen)`), so a wall-clock regression (NTP skew, test
//! clock set backwards) can neither stall nor un-stall a live op: it can
//! never produce a false stall, and a genuine stall keeps aging.

/// How long total silence (no output, no progress, no op completion) must
/// last before the predicate trips. Tunable per tracker.
pub const DEFAULT_STALL_SILENCE_MS: u64 = 10 * 60 * 1000;

/// Semantic progress classes (P0-79). **Evidence ≠ activity**: only these
/// classes count toward "progress" for expensive-cycle decisions (whether
/// to buy another model call / stop-and-replan). A tool event, heartbeat or
/// iteration boundary is *activity*, not evidence — activity alone for 2x
/// the stall threshold without any evidence class stalls the op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressEvidence {
    /// The verification/criteria fact rows were rewritten (a genuine turn
    /// end wrote `task_state`/`verification`/`criteria` memory facts).
    CriterionStatusChanged,
    /// A verification Failed outcome carried a different failure
    /// fingerprint (check id + summary hash) than the previous failure of
    /// the same check — the task moved through the failure space.
    FailureFingerprintChanged,
    /// A tool outcome mutated a file whose digest differs from the last
    /// digest observed for that path (bounded per-path digest LRU).
    RepoStateChanged,
    /// Evidence retrieval admitted a NEW evidence set into the context
    /// (rows were added vs the previous retrieval).
    NewEvidenceAdmitted,
    /// The durable plan grew: a PlanStepAdded entry was appended.
    PlanStepCompleted,
    /// A durable verification record finalized Passed after an earlier
    /// Failed record — verification of the work improved.
    VerificationImproved,
}

/// Cap of the bounded evidence log (ring of `(time, class)` entries).
pub const EVIDENCE_LOG_CAP: usize = 32;
/// Cap of the per-path last-digest LRU map (repo-state change detection +
/// loop fingerprints).
pub const REPO_DIGEST_CAP: usize = 256;
/// Cap of the per-check criteria-status map (criterion-state fingerprints).
pub const CRITERIA_STATUS_CAP: usize = 32;
/// Cap of the per-check failure-fingerprint map (failure-space progress).
pub const FAILURE_FINGERPRINT_CAP: usize = 32;

/// Bounded LRU map (insert/touch moves the key to the back; overflow
/// evicts the front = least recently used). O(cap) per touch — caps are
/// small constants.
#[derive(Debug, Clone)]
struct LruMap<V: Copy> {
    map: HashMap<String, V>,
    order: VecDeque<String>,
    cap: usize,
}

impl<V: Copy> LruMap<V> {
    fn with_cap(cap: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            cap: cap.max(1),
        }
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    fn get(&self, key: &str) -> Option<V> {
        self.map.get(key).copied()
    }

    /// Insert (or re-touch) `key`; returns the previous value. The key is
    /// capped to 1 KiB — hostile oversized paths never grow the map.
    fn insert(&mut self, key: &str, value: V) -> Option<V> {
        let key = truncate_key(key);
        let prev = self.map.insert(key.clone(), value);
        self.order.retain(|k| k != &key);
        self.order.push_back(key);
        while self.order.len() > self.cap {
            if let Some(evicted) = self.order.pop_front() {
                self.map.remove(&evicted);
            }
        }
        prev
    }

    /// Stable sorted entries `(key, value)` for hashing (sorted by key —
    /// keys are unique, so no value ordering is needed).
    fn sorted(&self) -> Vec<(&str, V)> {
        let mut v: Vec<(&str, V)> = self.map.iter().map(|(k, val)| (k.as_str(), *val)).collect();
        v.sort_by(|a, b| a.0.cmp(b.0));
        v
    }
}

/// Keys are bounded before entering the maps (a hostile path/check id can
/// never inflate the bounded per-session record).
fn truncate_key(key: &str) -> String {
    key.chars().take(1024).collect()
}

/// Cheap deterministic content hash (blake3) over length-prefixed parts,
/// folded to u64. `hash_u64(&[])` is never used for non-empty maps: an
/// EMPTY semantic map hashes to 0 by convention (stable constant).
fn hash_u64(parts: &[Vec<u8>]) -> u64 {
    let mut hasher = blake3::Hasher::new();
    for p in parts {
        hasher.update(&(p.len() as u64).to_le_bytes());
        hasher.update(p);
    }
    let out = hasher.finalize();
    u64::from_le_bytes(out.as_bytes()[..8].try_into().expect("8 bytes"))
}

/// Bounded per-session progress record + semantic evidence store + the two
/// stalled predicates.
///
/// Feeding methods (`output`, `progress`, `begin_op`, `end_op`,
/// `note_evidence`, `note_repo_digest`, ...) record a single timestamp /
/// bounded entry each — O(1)-bounded memory, no unbounded history. The
/// predicates are:
///
/// ```text
/// stalled                = now - alive_at(now)              > threshold
/// stalled_expensive_cycle = now - evidence_alive_at(now)    > threshold
/// ```
/// where `alive_at` is the newest of `{last_output_at, last_progress_at,
/// last_op_completed_at}` and `evidence_alive_at` the newest of
/// `{last_output_at, last_evidence_at, last_op_completed_at}` — both
/// widened to the in-flight op's start while an op is running (a fresh op
/// gets the full threshold of grace; an op with no evidence for a full
/// threshold is stuck). Heartbeats (tool events, iteration completions)
/// keep `alive_at` fresh but NEVER `evidence_alive_at`: an op that only
/// churns tools is stalled for expensive-cycle decisions (P0-79).
#[derive(Debug, Clone)]
pub struct StallTracker {
    threshold_ms: u64,
    /// Newest durable output (text/reasoning/tool bytes emitted).
    last_output_at: Option<i64>,
    /// Newest progress evidence: op heartbeats, tool events, iteration
    /// completions — AND every semantic-evidence note (P0-79 feeds the
    /// same stamp, so evidence is also activity).
    last_progress_at: Option<i64>,
    /// When the most recent operation completed.
    last_op_completed_at: Option<i64>,
    /// The in-flight operation id, if any (bounded string).
    in_flight_op: Option<String>,
    /// When the in-flight op started (grace anchor).
    op_started_at: Option<i64>,
    /// Monotonic floor of observed time: regressions are absorbed.
    monotonic_now: i64,
    /// Last predicate verdict.
    pub stalled: bool,
    // ---- semantic evidence (P0-79): evidence ≠ activity ----
    /// Newest semantic-evidence stamp (one of [`ProgressEvidence`]).
    last_evidence_at: Option<i64>,
    /// Bounded evidence log (ring, `EVIDENCE_LOG_CAP`).
    evidence_log: VecDeque<(i64, ProgressEvidence)>,
    /// Per-path last-digest LRU (`REPO_DIGEST_CAP`): the repo state the
    /// runtime has observed through tool outcomes that mutate files.
    repo_digests: LruMap<FileHash>,
    /// Per-check criteria statuses (`CRITERIA_STATUS_CAP`): the current
    /// verification status set the gate facts last wrote.
    criteria_statuses: LruMap<bool>,
    /// Per-check failure fingerprints (`FAILURE_FINGERPRINT_CAP`): the last
    /// `(check id, summary hash)` fingerprint of a Failed verification
    /// record, per check id.
    failure_fingerprints: LruMap<u64>,
    /// Status of the last finalized durable verification record (Failed →
    /// later Passed = verification improved).
    last_verification_status: Option<VerificationStatus>,
}

use std::collections::{HashMap, VecDeque};

use faktor_core::hash::FileHash;
use faktor_core::state::VerificationStatus;

impl StallTracker {
    pub fn new(threshold_ms: u64) -> Self {
        Self {
            threshold_ms: threshold_ms.max(1),
            last_output_at: None,
            last_progress_at: None,
            last_op_completed_at: None,
            in_flight_op: None,
            op_started_at: None,
            monotonic_now: i64::MIN,
            stalled: false,
            last_evidence_at: None,
            evidence_log: VecDeque::with_capacity(EVIDENCE_LOG_CAP),
            repo_digests: LruMap::with_cap(REPO_DIGEST_CAP),
            criteria_statuses: LruMap::with_cap(CRITERIA_STATUS_CAP),
            failure_fingerprints: LruMap::with_cap(FAILURE_FINGERPRINT_CAP),
            last_verification_status: None,
        }
    }

    pub fn threshold_ms(&self) -> u64 {
        self.threshold_ms
    }

    /// Monotonic absorption: time may never go backwards inside the
    /// tracker, so a clock regression can neither stall nor un-stall.
    fn absorb(&mut self, now_ms: i64) -> i64 {
        if now_ms > self.monotonic_now {
            self.monotonic_now = now_ms;
        }
        self.monotonic_now
    }

    /// Durable output bytes reached the client/stream.
    pub fn output(&mut self, now_ms: i64) {
        let now = self.absorb(now_ms);
        self.last_output_at = Some(now);
        self.stalled = false;
    }

    /// Progress evidence not tied to output: op heartbeats, tool events,
    /// iteration completions.
    pub fn progress(&mut self, now_ms: i64) {
        let now = self.absorb(now_ms);
        self.last_progress_at = Some(now);
        self.stalled = false;
    }

    /// A semantic evidence class occurred (P0-79). **Evidence ≠ activity**:
    /// `note_evidence` feeds the SAME `last_progress_at` stamp as a
    /// heartbeat (evidence is also activity — it keeps the pure-silence
    /// predicate alive) but additionally stamps `last_evidence_at` and
    /// pushes a bounded ring entry consulted by
    /// [`StallTracker::check_evidence`] — the expensive-cycle predicate.
    /// Only the six [`ProgressEvidence`] classes may be fed here; tool
    /// events alone are NOT evidence.
    pub fn note_evidence(&mut self, now_ms: i64, evidence: ProgressEvidence) {
        let now = self.absorb(now_ms);
        self.last_progress_at = Some(now);
        self.last_evidence_at = Some(now);
        self.evidence_log.push_back((now, evidence));
        while self.evidence_log.len() > EVIDENCE_LOG_CAP {
            self.evidence_log.pop_front();
        }
        self.stalled = false;
    }

    /// Observe the digest of a file a tool outcome mutated. Records the
    /// per-path digest in the bounded LRU and returns true — feeding the
    /// [`ProgressEvidence::RepoStateChanged`] evidence class — exactly when
    /// the repo state actually changed (new path, or a digest different
    /// from the last digest seen for that path). An identical re-write is
    /// NOT progress: the file is byte-identical.
    pub fn note_repo_digest(&mut self, now_ms: i64, path: &str, digest: FileHash) -> bool {
        let changed = self.repo_digests.get(path) != Some(digest);
        self.repo_digests.insert(path, digest);
        if changed {
            self.note_evidence(now_ms, ProgressEvidence::RepoStateChanged);
        }
        changed
    }

    /// Fold the newest per-check criteria statuses (the `(check id,
    /// passed)` results a genuine end wrote to the gate facts) into the
    /// bounded criteria-status map. The status set is what the loop
    /// fingerprint's `criterion_state` hashes over. The caller notes
    /// [`ProgressEvidence::CriterionStatusChanged`] at the fact-write site.
    pub fn note_criteria_statuses(&mut self, results: &[(String, bool)]) {
        for (id, passed) in results.iter().take(CRITERIA_STATUS_CAP) {
            self.criteria_statuses.insert(id, *passed);
        }
    }

    /// Store the failure fingerprint `(check id → summary hash)` of the
    /// current verification attempt. Returns true when ANY check's
    /// fingerprint differs from the fingerprint of its previous failure —
    /// the task moved through the failure space (a changed fingerprint is
    /// [`ProgressEvidence::FailureFingerprintChanged`] evidence; the caller
    /// notes it when this returns true).
    pub fn note_failure_fingerprints(&mut self, fingerprints: &[(String, u64)]) -> bool {
        let mut changed = false;
        for (id, fp) in fingerprints.iter().take(FAILURE_FINGERPRINT_CAP) {
            if self.failure_fingerprints.get(id) != Some(*fp) {
                changed = true;
            }
            self.failure_fingerprints.insert(id, *fp);
        }
        changed
    }

    /// The status of the most recent finalized durable verification record.
    pub fn last_verification_status(&self) -> Option<VerificationStatus> {
        self.last_verification_status
    }

    pub fn set_last_verification_status(&mut self, status: VerificationStatus) {
        self.last_verification_status = Some(status);
    }

    /// An operation started. Replaces any previous in-flight op (the prior
    /// op was abandoned); its completion stamp, if any, stays on the record.
    pub fn begin_op(&mut self, now_ms: i64, op_id: impl Into<String>) {
        let now = self.absorb(now_ms);
        self.in_flight_op = Some(op_id.into());
        self.op_started_at = Some(now);
        self.stalled = false;
    }

    /// The in-flight operation completed (successfully or not — completion
    /// is progress evidence).
    pub fn end_op(&mut self, now_ms: i64) {
        let now = self.absorb(now_ms);
        if self.in_flight_op.is_some() {
            self.last_op_completed_at = Some(now);
            self.in_flight_op = None;
            self.op_started_at = None;
        }
        self.stalled = false;
    }

    pub fn in_flight_op(&self) -> Option<&str> {
        self.in_flight_op.as_deref()
    }

    pub fn last_output_at(&self) -> Option<i64> {
        self.last_output_at
    }

    pub fn last_progress_at(&self) -> Option<i64> {
        self.last_progress_at
    }

    pub fn last_op_completed_at(&self) -> Option<i64> {
        self.last_op_completed_at
    }

    /// Newest semantic-evidence stamp (None when no [`ProgressEvidence`]
    /// class ever occurred — heartbeats and tool events do NOT set it).
    pub fn last_evidence_at(&self) -> Option<i64> {
        self.last_evidence_at
    }

    /// The bounded evidence ring (oldest first; `EVIDENCE_LOG_CAP`).
    pub fn evidence_log(&self) -> impl Iterator<Item = &(i64, ProgressEvidence)> {
        self.evidence_log.iter()
    }

    pub fn evidence_log_len(&self) -> usize {
        self.evidence_log.len()
    }

    /// Number of paths with a recorded last-digest (`REPO_DIGEST_CAP`).
    pub fn repo_digest_len(&self) -> usize {
        self.repo_digests.len()
    }

    /// The loop fingerprint's `repo_state`: a stable hash over the bounded
    /// per-path digest map (sorted by path — insertion order never leaks
    /// into the fingerprint). 0 when no digest was ever observed.
    pub fn repo_state_hash(&self) -> u64 {
        let entries = self.repo_digests.sorted();
        if entries.is_empty() {
            return 0;
        }
        let mut parts: Vec<Vec<u8>> = Vec::with_capacity(entries.len() * 2);
        for (path, digest) in &entries {
            parts.push(path.as_bytes().to_vec());
            parts.push(digest.bytes().to_vec());
        }
        hash_u64(&parts)
    }

    /// The loop fingerprint's `criterion_state`: a stable hash over the
    /// current per-check criteria statuses. 0 when no status was recorded.
    pub fn criteria_state_hash(&self) -> u64 {
        let entries = self.criteria_statuses.sorted();
        if entries.is_empty() {
            return 0;
        }
        let mut parts: Vec<Vec<u8>> = Vec::with_capacity(entries.len() * 2);
        for (id, passed) in &entries {
            parts.push(id.as_bytes().to_vec());
            parts.push(if *passed {
                b"1".to_vec()
            } else {
                b"0".to_vec()
            });
        }
        hash_u64(&parts)
    }

    /// The loop fingerprint's `failure_fingerprint`: a stable hash over the
    /// per-check failure fingerprints. 0 when no failure was recorded.
    pub fn failure_state_hash(&self) -> u64 {
        let entries = self.failure_fingerprints.sorted();
        if entries.is_empty() {
            return 0;
        }
        let mut parts: Vec<Vec<u8>> = Vec::with_capacity(entries.len() * 2);
        for (id, fp) in &entries {
            parts.push(id.as_bytes().to_vec());
            parts.push(fp.to_le_bytes().to_vec());
        }
        hash_u64(&parts)
    }

    /// The newest moment at which the runtime was provably alive: output,
    /// progress, op completion — widened to the in-flight op's start while
    /// one runs (a freshly started op is alive by definition).
    fn alive_at(&self) -> Option<i64> {
        let mut alive = None;
        for stamp in [
            self.last_output_at,
            self.last_progress_at,
            self.last_op_completed_at,
        ]
        .into_iter()
        .flatten()
        {
            alive = Some(alive.map_or(stamp, |a: i64| a.max(stamp)));
        }
        if self.in_flight_op.is_some() {
            if let Some(started) = self.op_started_at {
                alive = Some(alive.map_or(started, |a| a.max(started)));
            }
        }
        alive
    }

    /// The newest moment at which the op was provably PROGRESSING in the
    /// semantic sense (P0-79): durable output, an evidence class, or an op
    /// completion — widened to the in-flight op's start. Heartbeat/tool
    /// events (`progress`) are deliberately NOT in this set: activity ≠
    /// evidence.
    fn evidence_alive_at(&self) -> Option<i64> {
        let mut alive = None;
        for stamp in [
            self.last_output_at,
            self.last_evidence_at,
            self.last_op_completed_at,
        ]
        .into_iter()
        .flatten()
        {
            alive = Some(alive.map_or(stamp, |a: i64| a.max(stamp)));
        }
        if self.in_flight_op.is_some() {
            if let Some(started) = self.op_started_at {
                alive = Some(alive.map_or(started, |a| a.max(started)));
            }
        }
        alive
    }

    /// Milliseconds of silence since the newest alive evidence (0 when
    /// nothing was ever recorded).
    pub fn silence_ms(&mut self, now_ms: i64) -> u64 {
        let now = self.absorb(now_ms);
        match self.alive_at() {
            None => 0,
            Some(alive) => now.saturating_sub(alive).max(0) as u64,
        }
    }

    /// Milliseconds since the newest semantic evidence (0 when no evidence
    /// class was ever recorded). Consulted by the expensive-cycle
    /// predicate.
    pub fn evidence_silence_ms(&mut self, now_ms: i64) -> u64 {
        let now = self.absorb(now_ms);
        match self.evidence_alive_at() {
            None => 0,
            Some(alive) => now.saturating_sub(alive).max(0) as u64,
        }
    }

    /// Evaluate the stalled predicate at `now_ms`. Pure wrt. the record;
    /// absorbs the reading into the monotonic floor. `stalled` stays true
    /// across clock regressions and is cleared only by real evidence.
    pub fn check(&mut self, now_ms: i64) -> bool {
        let silence = self.silence_ms(now_ms);
        // Nothing was ever recorded AND no op is in flight: there is no
        // basis to claim a stall (a fresh, never-active session is not
        // stalled).
        let verdict = self.alive_at().is_some() && silence > self.threshold_ms;
        if verdict {
            self.stalled = true;
        }
        verdict
    }

    /// The EVIDENCE-aware stalled predicate (P0-79) for expensive-cycle
    /// decisions (each cycle costs a model call): consult BOTH the
    /// time-based record and the semantic evidence log. An op whose ONLY
    /// liveness is tool events/heartbeats stalls once no
    /// [`ProgressEvidence`] class (and no durable output / op completion)
    /// arrived for a full threshold — even when a tool fires every 100 ms —
    /// while an op emitting one evidence class per interval shorter than
    /// the threshold NEVER stalls, even with zero output. Same grace /
    /// regression rules as [`StallTracker::check`].
    pub fn check_evidence(&mut self, now_ms: i64) -> bool {
        let silence = self.evidence_silence_ms(now_ms);
        let verdict = self.evidence_alive_at().is_some() && silence > self.threshold_ms;
        if verdict {
            self.stalled = true;
        }
        verdict
    }

    /// `stalled` without evaluating (the last [`StallTracker::check`]
    /// verdict; feeds clear it).
    pub fn is_stalled(&self) -> bool {
        self.stalled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicI64, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    /// Controllable clock that may be set backwards (regression tests).
    #[derive(Debug, Clone, Default)]
    struct SkewClock(Arc<AtomicI64>);

    impl SkewClock {
        fn new(now: i64) -> Self {
            let c = SkewClock::default();
            c.0.store(now, Ordering::SeqCst);
            c
        }
        fn now(&self) -> i64 {
            self.0.load(Ordering::SeqCst)
        }
        fn advance(&self, ms: i64) {
            self.0.fetch_add(ms, Ordering::SeqCst);
        }
        fn set(&self, now: i64) {
            self.0.store(now, Ordering::SeqCst);
        }
    }

    const T: u64 = 1000; // threshold (1 s) for the unit tests below

    #[test]
    fn silence_with_no_evidence_is_not_stalled() {
        let clock = SkewClock::new(0);
        let mut t = StallTracker::new(T);
        clock.advance(100 * T as i64);
        assert!(
            !t.check(clock.now()),
            "a session that never did anything is not stalled"
        );
        assert!(!t.is_stalled());
    }

    #[test]
    fn long_running_op_with_periodic_heartbeats_never_stalls() {
        // The instruction's core case: an op runs 3x the stall threshold
        // while emitting periodic progress updates — it must never stall.
        let clock = SkewClock::new(0);
        let mut t = StallTracker::new(T);
        t.begin_op(clock.now(), "op-1");
        // Heartbeat every T/4 for 3x the threshold.
        for i in 0..(3 * T as i64) {
            clock.advance(T as i64 / 4);
            let now = clock.now();
            assert!(
                !t.check(now),
                "heartbeat at tick {i} must keep the op alive"
            );
            t.progress(now);
        }
        assert!(!t.is_stalled(), "op with heartbeats is never stalled");
        t.end_op(clock.now());
        assert!(!t.check(clock.now()));
    }

    #[test]
    fn silence_stalls_and_progress_resumes() {
        let clock = SkewClock::new(0);
        let mut t = StallTracker::new(T);
        t.begin_op(clock.now(), "op-1");
        assert!(!t.check(clock.now()), "fresh op has grace");
        clock.advance(T as i64 - 1);
        assert!(
            !t.check(clock.now()),
            "silence below the threshold is not a stall"
        );
        clock.advance(2);
        assert!(
            t.check(clock.now()),
            "silence past the threshold with an in-flight op stalls (stuck op)"
        );
        assert!(t.is_stalled());
        // Progress resumes -> unstalled immediately; the predicate must
        // stay false while evidence keeps arriving.
        t.progress(clock.now());
        assert!(!t.is_stalled(), "evidence clears the stall");
        assert!(!t.check(clock.now()));
        clock.advance(2 * T as i64);
        assert!(
            t.check(clock.now()),
            "silence after the resume stalls again"
        );
    }

    #[test]
    fn idle_session_with_old_op_completion_stalls() {
        // in_flight_op is None; the last op completed a long time ago:
        // nothing is running and nothing is happening -> stalled.
        let clock = SkewClock::new(0);
        let mut t = StallTracker::new(T);
        t.begin_op(clock.now(), "op-1");
        t.output(clock.now());
        clock.advance(T as i64 / 2);
        t.end_op(clock.now()); // completed at t = T/2
        assert!(!t.check(clock.now()), "just completed");
        clock.advance(T as i64);
        assert!(
            !t.check(clock.now()),
            "exactly at the boundary is not yet stalled (strict >)"
        );
        clock.advance(1);
        assert!(t.check(clock.now()), "past the boundary stalls");
    }

    #[test]
    fn op_completing_exactly_at_threshold_boundary_never_stalls() {
        // Adversarial (c): an op that completes exactly when its silence
        // reaches the threshold must NOT be stalled — completion is the
        // newest evidence and resets the age to zero.
        let clock = SkewClock::new(0);
        let mut t = StallTracker::new(T);
        t.begin_op(clock.now(), "op-1");
        clock.advance(T as i64);
        assert!(
            !t.check(clock.now()),
            "silence exactly AT the threshold with an in-flight op is not stuck yet"
        );
        t.end_op(clock.now());
        assert!(
            !t.check(clock.now()),
            "op completed exactly at the boundary: never stalled"
        );
        assert!(!t.is_stalled());
        // Only AFTER the completion does the clock start over.
        clock.advance(T as i64 + 1);
        assert!(t.check(clock.now()));
    }

    #[test]
    fn clock_regression_cannot_stall_or_false_stall() {
        // Adversarial (a): time goes backwards — the monotonic floor must
        // absorb it. A regression can neither manufacture a stall (the
        // silence age never jumps) nor un-stall a genuine one.
        let clock = SkewClock::new(10_000);
        let mut t = StallTracker::new(T);
        t.begin_op(clock.now(), "op-1");
        clock.advance(T as i64 / 2); // progress at 10_500
        t.progress(clock.now());
        // Regression: set the clock 10x the threshold into the past.
        clock.set(500);
        assert!(
            !t.check(clock.now()),
            "clock regression must not false-stall: age is frozen, not inflated"
        );
        assert!(!t.is_stalled());
        // Real silence still ages from the pre-regression floor (10_500):
        // at floor + threshold + 1 the stall is due even though the raw
        // clock only reads 10_501.
        clock.set(10_500 + T as i64 + 1);
        assert!(
            t.check(clock.now()),
            "genuine silence past the threshold stalls even after a regression"
        );
        assert!(t.is_stalled());
        // A second regression cannot clear a true stall (no evidence).
        clock.set(0);
        assert!(
            t.is_stalled(),
            "a backward clock must not clear a genuine stall"
        );
        assert!(t.check(clock.now()), "verdict survives the regression");
        // Only real evidence clears it.
        t.output(clock.now());
        assert!(!t.is_stalled());
    }

    #[test]
    fn begin_op_replaces_abandoned_op_and_keeps_record_bounded() {
        let clock = SkewClock::new(0);
        let mut t = StallTracker::new(T);
        t.begin_op(clock.now(), "op-a");
        clock.advance(10);
        t.output(clock.now());
        t.begin_op(clock.now(), "op-b"); // op-a abandoned mid-flight
        assert_eq!(t.in_flight_op(), Some("op-b"));
        assert_eq!(t.last_output_at(), Some(10), "output record survives");
        assert!(t.last_op_completed_at().is_none(), "op-a never completed");
        clock.advance(T as i64 + 1);
        assert!(
            t.check(clock.now()),
            "op-b with no evidence past the threshold is stuck"
        );
    }

    #[test]
    fn real_time_op_with_heartbeats_runs_three_thresholds_without_stalling() {
        // End-to-end with the real monotonic clock: an async op emits a
        // progress heartbeat repeatedly for 3x the stall threshold; a
        // watchdog samples the predicate mid-interval (between beats) and
        // the op must NEVER be stalled while it keeps heartbeating. A
        // second op that goes totally silent must stall.
        const STALL: Duration = Duration::from_millis(200);
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let t0 = std::time::Instant::now();
            // Instant-elapsed is monotonic by construction.
            let now_ms = move || t0.elapsed().as_millis() as i64;
            let tracker = std::sync::Arc::new(std::sync::Mutex::new(StallTracker::new(
                STALL.as_millis() as u64,
            )));
            {
                let mut t = tracker.lock().unwrap();
                t.begin_op(now_ms(), "op-live");
            }
            let (beat_tx, mut beat_rx) = tokio::sync::mpsc::unbounded_channel::<()>();
            let hb_tracker = tracker.clone();
            let hb_now = now_ms;
            let hb = tokio::spawn(async move {
                // 3x the stall threshold of beats at STALL/4 cadence.
                let end = std::time::Instant::now() + STALL * 3;
                while std::time::Instant::now() < end {
                    tokio::time::sleep(STALL / 4).await;
                    hb_tracker.lock().unwrap().progress(hb_now());
                    let _ = beat_tx.send(());
                }
                hb_tracker.lock().unwrap().end_op(hb_now());
            });
            // The watchdog samples mid-interval: after every beat, it waits
            // STALL/6 and checks — the age at that moment is at most
            // STALL/6 + scheduling slack, far below the threshold, so the
            // op must never stall for the whole 3x-threshold run.
            let wd_now = now_ms;
            let wd_tracker = tracker.clone();
            let wd = tokio::spawn(async move {
                let end = std::time::Instant::now() + STALL * 3;
                let mut tripped = false;
                while std::time::Instant::now() < end {
                    if beat_rx.recv().await.is_none() {
                        break;
                    }
                    tokio::time::sleep(STALL / 6).await;
                    if wd_tracker.lock().unwrap().check(wd_now()) {
                        tripped = true;
                    }
                }
                tripped
            });
            hb.await.unwrap();
            let tripped = wd.await.unwrap();
            assert!(
                !tripped,
                "an op heartbeating every STALL/4 for 3x STALL must never stall"
            );
            // Total silence stalls under the real clock too.
            let s_t0 = std::time::Instant::now();
            let s_now = move || s_t0.elapsed().as_millis() as i64;
            let mut silent = StallTracker::new(STALL.as_millis() as u64);
            silent.begin_op(s_now(), "op-silent");
            let end = std::time::Instant::now() + STALL * 2;
            let mut tripped = false;
            while std::time::Instant::now() < end {
                tokio::time::sleep(STALL / 4).await;
                if silent.check(s_now()) {
                    tripped = true;
                    break;
                }
            }
            assert!(tripped, "total silence past the threshold must stall");
        });
    }

    #[test]
    fn output_is_evidence_and_keeps_large_threshold_ops_alive() {
        let clock = SkewClock::new(0);
        let mut t = StallTracker::new(T);
        t.begin_op(clock.now(), "op");
        for i in 0..5 {
            clock.advance(T as i64 / 2);
            t.output(clock.now());
            assert!(!t.check(clock.now()), "output tick {i} keeps it alive");
        }
        t.end_op(clock.now());
        assert!(!t.is_stalled());
    }

    #[test]
    fn completed_op_is_alive_at_its_completion_time() {
        let clock = SkewClock::new(1000);
        let mut t = StallTracker::new(T);
        t.begin_op(clock.now(), "op");
        clock.advance(5 * T as i64);
        t.end_op(clock.now());
        assert!(!t.check(clock.now()), "completion happened just now");
        // ... and the age restarts from the completion stamp.
        clock.advance(T as i64 + 1);
        assert!(t.check(clock.now()));
    }

    // ---- P0-79: semantic evidence (evidence ≠ activity) ----

    #[test]
    fn heartbeats_alone_are_activity_never_evidence() {
        let clock = SkewClock::new(0);
        let mut t = StallTracker::new(T);
        t.begin_op(clock.now(), "op");
        clock.advance(500);
        t.progress(clock.now());
        assert!(
            t.last_evidence_at().is_none(),
            "a heartbeat/tool event must never stamp evidence"
        );
        assert_eq!(t.evidence_log_len(), 0);
        t.output(clock.now());
        assert!(
            t.last_evidence_at().is_none(),
            "output is durable activity, not a semantic evidence class"
        );
    }

    #[test]
    fn tool_activity_without_evidence_stalls_expensive_cycles() {
        // (a) An op fires a tool event every T/10 for 3x the stall
        // threshold but emits NO evidence class. Activity keeps the
        // pure-silence (stuck-stream) predicate quiet, yet the
        // evidence-aware predicate must stall: tools every 100 ms do not
        // mean the task is closer to completion.
        let clock = SkewClock::new(0);
        let mut t = StallTracker::new(T);
        t.begin_op(clock.now(), "op-tools");
        let ticks = (3 * T as i64) / (T as i64 / 10);
        for _ in 0..ticks {
            clock.advance(T as i64 / 10);
            let now = clock.now();
            assert!(
                !t.check(now),
                "tool activity is still activity: pure silence never trips"
            );
            if now <= T as i64 {
                assert!(
                    !t.check_evidence(now),
                    "a fresh op gets the full threshold of grace"
                );
            }
            t.progress(now); // heartbeat at each tool event
        }
        assert!(clock.now() >= 3 * T as i64);
        assert!(
            t.check_evidence(clock.now()),
            "3x the threshold of tool-only activity with zero evidence classes must stall"
        );
        assert!(t.is_stalled());
        assert!(
            !t.check(clock.now()),
            "the same instant is NOT a pure-silence stall: activity happened"
        );
        // Evidence resumes -> the expensive-cycle predicate clears.
        t.note_evidence(clock.now(), ProgressEvidence::CriterionStatusChanged);
        assert!(!t.is_stalled());
        assert!(!t.check_evidence(clock.now()));
        // ... and ages again when evidence stops.
        clock.advance(T as i64 + 1);
        assert!(t.check_evidence(clock.now()));
    }

    #[test]
    fn evidence_cadence_keeps_a_silent_op_alive_for_five_thresholds() {
        // (b) Output is silent for 5x the threshold, but a semantic
        // evidence class fires every 40 s while the stall threshold is
        // 60 s: the op must NEVER stall — evidence every 40 s keeps the
        // predicate alive even with zero output.
        const EVIDENCE_EVERY_MS: i64 = 40_000;
        let clock = SkewClock::new(0);
        let mut t = StallTracker::new(60_000);
        t.begin_op(clock.now(), "op-evidence");
        let mut at = 0i64;
        while at < 5 * 60_000 {
            clock.advance(EVIDENCE_EVERY_MS);
            at += EVIDENCE_EVERY_MS;
            let now = clock.now();
            assert!(
                !t.check_evidence(now),
                "evidence {at} ms in with a 60 s budget must never stall"
            );
            assert!(
                !t.check(now),
                "evidence feeds the same last_progress_at: pure silence stays quiet"
            );
            t.note_evidence(now, ProgressEvidence::CriterionStatusChanged);
        }
        assert!(
            !t.is_stalled(),
            "5x threshold of 40 s evidence cadence: never stalled"
        );
        // The cadence STOPPED: silence ages past the threshold eventually.
        clock.advance(60_001);
        assert!(
            t.check_evidence(clock.now()),
            "an evidence cadence that stops must stall after one budget"
        );
    }

    #[test]
    fn output_counts_as_progress_but_tool_heartbeats_do_not() {
        // The semantic boundary the predicates encode: durable output
        // (text/reasoning the user sees) is progress for expensive cycles —
        // a 10x-budget stream of chunks must never stall — while a
        // heartbeat/tool event at the same cadence buys nothing.
        let clock = SkewClock::new(0);
        let mut out = StallTracker::new(T);
        out.begin_op(clock.now(), "op");
        let mut beats = StallTracker::new(T);
        beats.begin_op(clock.now(), "op");
        for _ in 0..(8 * T as i64 / (T as i64 / 4)) {
            clock.advance(T as i64 / 4);
            let now = clock.now();
            out.output(now);
            beats.progress(now);
        }
        assert!(clock.now() >= 8 * T as i64);
        assert!(
            !out.check_evidence(clock.now()),
            "output cadence keeps cycles alive"
        );
        assert!(
            beats.check_evidence(clock.now()),
            "identical cadence of bare tool heartbeats stalls the expensive cycle"
        );
    }

    #[test]
    fn repo_digest_observation_is_change_sensitive() {
        // RepoStateChanged evidence fires ONLY when the digest for a path
        // actually differs from the last digest seen for it: re-writing a
        // file byte-identically is not progress.
        let clock = SkewClock::new(0);
        let mut t = StallTracker::new(T);
        let d1 = FileHash::from([1u8; 32]);
        let d2 = FileHash::from([2u8; 32]);
        assert!(
            t.note_repo_digest(clock.now(), "src/a.rs", d1),
            "a new path with a digest is a repo-state change"
        );
        assert!(
            !t.note_repo_digest(clock.now(), "src/a.rs", d1),
            "an identical re-write is not a change"
        );
        assert!(
            t.note_repo_digest(clock.now(), "src/a.rs", d2),
            "a different digest is a change"
        );
        assert!(
            t.note_repo_digest(clock.now(), "src/b.rs", d2),
            "a second path is a change"
        );
        // RepoStateChanged evidence entries: 3 (new path, changed digest,
        // second path) — the identical re-write left no entry.
        let repo_notes = t
            .evidence_log()
            .filter(|(_, e)| *e == ProgressEvidence::RepoStateChanged)
            .count();
        assert_eq!(repo_notes, 3);
    }

    #[test]
    fn repo_state_hash_is_stable_insertion_order_insensitive_and_sensitive_to_content() {
        let a = FileHash::from([1u8; 32]);
        let b = FileHash::from([2u8; 32]);
        let feed = |t: &mut StallTracker, order: &[(&str, FileHash)]| {
            for (p, d) in order {
                let _ = t.note_repo_digest(0, p, *d);
            }
        };
        let mut t1 = StallTracker::new(T);
        feed(&mut t1, &[("z.rs", a), ("a.rs", b), ("m.rs", a)]);
        let mut t2 = StallTracker::new(T);
        feed(&mut t2, &[("m.rs", a), ("z.rs", a), ("a.rs", b)]);
        assert_eq!(
            t1.repo_state_hash(),
            t2.repo_state_hash(),
            "insertion order must never leak into the fingerprint"
        );
        assert_ne!(
            t1.repo_state_hash(),
            0,
            "a non-empty digest map never hashes to the empty sentinel"
        );
        assert_eq!(StallTracker::new(T).repo_state_hash(), 0, "empty map = 0");
        let mut t3 = StallTracker::new(T);
        feed(&mut t3, &[("a.rs", a)]);
        assert_ne!(
            t1.repo_state_hash(),
            t3.repo_state_hash(),
            "different digest sets must differ"
        );
        // The patch->revert->patch alternation is visible as A,B,A.
        let mut t4 = StallTracker::new(T);
        let _ = t4.note_repo_digest(0, "src/x.rs", a);
        let h_a1 = t4.repo_state_hash();
        let _ = t4.note_repo_digest(0, "src/x.rs", b);
        let h_b = t4.repo_state_hash();
        let _ = t4.note_repo_digest(0, "src/x.rs", a);
        assert_eq!(t4.repo_state_hash(), h_a1, "re-applied patch: A again");
        assert_ne!(h_b, h_a1);
    }

    #[test]
    fn failure_fingerprints_and_criteria_statuses_feed_state_hashes() {
        let mut t = StallTracker::new(T);
        assert_eq!(t.criteria_state_hash(), 0);
        assert_eq!(t.failure_state_hash(), 0);
        assert!(t.last_verification_status().is_none());
        t.note_criteria_statuses(&[("cargo check".into(), true)]);
        assert_ne!(t.criteria_state_hash(), 0);
        // Fingerprint change detection: same check, different summary hash.
        assert!(t.note_failure_fingerprints(&[("rust_check".into(), 11)]));
        assert!(
            !t.note_failure_fingerprints(&[("rust_check".into(), 11)]),
            "the identical failure fingerprint is not a change"
        );
        assert!(
            t.note_failure_fingerprints(&[("rust_check".into(), 22)]),
            "a moved failure fingerprint is a change (progress through the failure space)"
        );
        // Order-insensitive aggregate.
        let mut u = StallTracker::new(T);
        u.note_failure_fingerprints(&[("rust_check".into(), 11), ("lint_check".into(), 33)]);
        u.note_failure_fingerprints(&[("lint_check".into(), 33), ("rust_check".into(), 11)]);
        let mut t2 = StallTracker::new(T);
        t2.note_failure_fingerprints(&[("lint_check".into(), 33), ("rust_check".into(), 11)]);
        assert_eq!(
            t2.failure_state_hash(),
            u.failure_state_hash(),
            "the same fingerprint set folded in any order hashes identically"
        );
        assert_ne!(
            t.failure_state_hash(),
            u.failure_state_hash(),
            "rust_check 22 ≠ 11: the aggregates differ"
        );
        let mut t3 = StallTracker::new(T);
        t3.note_failure_fingerprints(&[("lint_check".into(), 33)]);
        t3.note_failure_fingerprints(&[("rust_check".into(), 11)]);
        assert_eq!(
            t3.failure_state_hash(),
            u.failure_state_hash(),
            "incremental folds converge to the same aggregate"
        );
    }

    #[test]
    fn heavy_evidence_and_digest_churn_stay_within_the_bounded_caps() {
        // (f) Bounded memory: 10k evidence notes + 10k digest observations
        // across 300 paths must leave the ring at 32 and the LRU at 256.
        let clock = SkewClock::new(0);
        let mut t = StallTracker::new(T);
        t.begin_op(clock.now(), "op");
        let kinds = [
            ProgressEvidence::CriterionStatusChanged,
            ProgressEvidence::FailureFingerprintChanged,
            ProgressEvidence::RepoStateChanged,
            ProgressEvidence::NewEvidenceAdmitted,
            ProgressEvidence::PlanStepCompleted,
            ProgressEvidence::VerificationImproved,
        ];
        for i in 0..10_000 {
            clock.advance(1);
            t.note_evidence(clock.now(), kinds[i % kinds.len()]);
        }
        assert_eq!(
            t.evidence_log_len(),
            EVIDENCE_LOG_CAP,
            "the evidence ring never exceeds 32 entries"
        );
        assert_eq!(
            t.last_evidence_at(),
            Some(clock.now()),
            "the newest note is the newest evidence stamp"
        );
        assert_eq!(
            t.evidence_log().next().unwrap().0,
            clock.now() - EVIDENCE_LOG_CAP as i64 + 1,
            "the ring holds the NEWEST 32 entries"
        );
        // Digest churn across 300 distinct paths: the LRU keeps exactly the
        // newest 256.
        for i in 0..300 {
            let _ = t.note_repo_digest(
                clock.now(),
                &format!("src/f{i:03}.rs"),
                FileHash::from([(i % 251) as u8; 32]),
            );
        }
        assert_eq!(t.repo_digest_len(), REPO_DIGEST_CAP);
        assert!(
            t.repo_digests.get("src/f299.rs").is_some(),
            "the newest path survives"
        );
        assert!(
            t.repo_digests.get("src/f044.rs").is_some(),
            "kept = the newest 256 (044..=299)"
        );
        assert!(
            t.repo_digests.get("src/f043.rs").is_none(),
            "the 44 least-recent paths (000..=043) are evicted"
        );
        // Re-touching an evicted path evicts the least-recent survivor and
        // stays within the cap.
        assert!(t.note_repo_digest(clock.now(), "src/f000.rs", FileHash::from([9u8; 32])));
        assert_eq!(t.repo_digest_len(), REPO_DIGEST_CAP);
        assert!(
            t.repo_digests.get("src/f000.rs").is_some(),
            "re-touched path is back"
        );
        assert!(
            t.repo_digests.get("src/f044.rs").is_none(),
            "the re-touch evicted the least-recent survivor (044)"
        );
        // ... and the criteria/failure maps are capped too.
        for i in 0..10_000 {
            t.note_criteria_statuses(&[(format!("check-{}", i % 200), i % 2 == 0)]);
            let _ = t.note_failure_fingerprints(&[(format!("fail-{}", i % 200), i as u64)]);
        }
        assert_eq!(t.criteria_statuses.len(), CRITERIA_STATUS_CAP);
        assert_eq!(t.failure_fingerprints.len(), FAILURE_FINGERPRINT_CAP);
    }

    #[test]
    fn evidence_predicate_never_false_stalls_a_fresh_session() {
        let clock = SkewClock::new(0);
        let mut t = StallTracker::new(T);
        clock.advance(100 * T as i64);
        assert!(
            !t.check_evidence(clock.now()),
            "a session that never did anything is not stalled for expensive cycles"
        );
        assert!(!t.is_stalled());
    }

    #[test]
    fn clock_regression_cannot_clear_an_evidence_stall() {
        let clock = SkewClock::new(10_000);
        let mut t = StallTracker::new(T);
        t.begin_op(clock.now(), "op");
        clock.advance(2 * T as i64);
        assert!(t.check_evidence(clock.now()));
        assert!(t.is_stalled());
        clock.set(0);
        assert!(
            t.is_stalled(),
            "a backward clock must not clear a genuine evidence stall"
        );
        assert!(
            t.check_evidence(clock.now()),
            "verdict survives the regression"
        );
        t.note_evidence(clock.now(), ProgressEvidence::NewEvidenceAdmitted);
        assert!(!t.is_stalled());
    }
}
