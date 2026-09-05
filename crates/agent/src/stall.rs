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
//! The record is bounded: constant memory per session (no event history is
//! retained), exactly
//! `{last_output_at, last_progress_at, in_flight_op, last_op_completed_at}`
//! plus the in-flight op's start/evidence stamps.
//!
//! Time source: all callers pass `now_ms` (the runtime's injectable
//! [`faktor_core::time::Clock`]). The tracker MONOTONICIZES the reading
//! (`max(now, last_seen)`), so a wall-clock regression (NTP skew, test
//! clock set backwards) can neither stall nor un-stall a live op: it can
//! never produce a false stall, and a genuine stall keeps aging.

/// How long total silence (no output, no progress, no op completion) must
/// last before the predicate trips. Tunable per tracker.
pub const DEFAULT_STALL_SILENCE_MS: u64 = 10 * 60 * 1000;

/// Bounded per-session progress record + stalled predicate.
///
/// Feeding methods (`output`, `progress`, `begin_op`, `end_op`) record a
/// single timestamp each — O(1) memory, no history. The predicate is:
///
/// ```text
/// stalled = now - alive_at(now) > threshold
/// ```
/// where `alive_at` is the newest of `{last_output_at, last_progress_at,
/// last_op_completed_at}`, widened to the in-flight op's start while an op
/// is running (a fresh op gets the full threshold of grace; an op with no
/// evidence for a full threshold is stuck).
#[derive(Debug, Clone)]
pub struct StallTracker {
    threshold_ms: u64,
    /// Newest durable output (text/reasoning/tool bytes emitted).
    last_output_at: Option<i64>,
    /// Newest progress evidence: op heartbeats, tool events, iteration
    /// completions.
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
}

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

    /// Milliseconds of silence since the newest alive evidence (0 when
    /// nothing was ever recorded).
    pub fn silence_ms(&mut self, now_ms: i64) -> u64 {
        let now = self.absorb(now_ms);
        match self.alive_at() {
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
}
