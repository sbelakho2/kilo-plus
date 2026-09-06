//! Loop detection (spec §28): repeating sequences — same command, same
//! failure, same patch, same tool-call arguments, same retrieval query —
//! must stop and re-plan instead of repeating for 40 turns.
//!
//! # State-based loop fingerprints (P0-78)
//!
//! The text-level detectors (identical calls, A/B alternation, identical
//! errors) cannot see *state-based* loops — same compiler-failure
//! fingerprint, patch → revert → patch, different greps returning the same
//! evidence set. The detector therefore also consumes cheap state hashes
//! (never an LLM):
//!
//! - [`Fingerprint`] steps ([`LoopDetector::record_state_step`]): repo
//!   state + failure fingerprint + criterion state per step. Repo digests
//!   alternating A,B,A,B while the other dimensions stay unchanged is a
//!   patch→revert→patch loop ([`ReasonCode::PatchRevertPatch`]). A changed
//!   failure/criterion dimension resets the window: the task is moving
//!   through the failure space, not looping.
//! - evidence-set observations ([`LoopDetector::record_tool_evidence`]):
//!   ≥3 CONSECUTIVE tool calls with DIFFERENT commands returning the
//!   identical evidence hash is a repeated-evidence loop
//!   ([`ReasonCode::RepeatedEvidenceSet`]); the evidence-hash window is a
//!   bounded 16-entry ring.
//! - fingerprint-aware failures ([`LoopDetector::record_failure`]):
//!   identical failure text with the SAME fingerprint counts toward the
//!   repeat threshold; the same text with a CHANGED fingerprint is a new
//!   state, never a repeat (progress through the failure space).
//!
//! Every fingerprint trip reports its typed [`ReasonCode`] so the runtime
//! can surface it on the turn outcome like the existing loop codes.

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};

use faktor_core::state::ReasonCode;

/// Bounded evidence-hash ring cap (P0-78): at most this many recent
/// `(command key, evidence hash)` observations are retained.
pub const EVIDENCE_RING_CAP: usize = 16;
/// Bounded per-message failure-fingerprint map cap (P0-78).
pub const FAILURE_FP_CAP: usize = 64;
/// Consecutive different-command calls with an identical evidence set that
/// trip [`ReasonCode::RepeatedEvidenceSet`].
pub const REPEATED_EVIDENCE_MIN: usize = 3;

/// Cheap deterministic key/evidence/content hash folded to u64.
/// (Test-only today; kept generic so production fingerprinting can use it.)
#[allow(dead_code)]
fn hash_u64<T: Hash>(value: &T) -> u64 {
    let mut h = DefaultHasher::new();
    value.hash(&mut h);
    h.finish()
}

/// One step's state fingerprint (P0-78): cheap hashes the CALLER computes
/// from the bounded per-session state (repo digests, failure fingerprints,
/// criteria statuses — see `crate::stall::StallTracker`) plus the action
/// class of the step. All hashes are u64 — cheap, never an LLM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fingerprint {
    /// Hash of the per-path last-digest map (repo state).
    pub repo_state: u64,
    /// Hash of the per-check failure fingerprints (failure space position).
    pub failure_fingerprint: u64,
    /// Hash of the current criteria statuses.
    pub criterion_state: u64,
    /// What the step was ("write_file", "tool_batch", "patch", ...).
    pub action_class: &'static str,
}

impl Fingerprint {
    pub fn new(
        repo_state: u64,
        failure_fingerprint: u64,
        criterion_state: u64,
        action_class: &'static str,
    ) -> Self {
        Self {
            repo_state,
            failure_fingerprint,
            criterion_state,
            action_class,
        }
    }

    /// The state dimensions that must stay UNCHANGED for a repo oscillation
    /// to be a loop: failure fingerprints + criteria statuses (the "other
    /// state" of the patch→revert→patch rule).
    fn context(&self) -> (u64, u64) {
        (self.failure_fingerprint, self.criterion_state)
    }
}

/// Tracks normalized keys of tool calls / errors. When the same key is seen
/// `threshold` times the detector trips.
#[derive(Debug, Clone)]
pub struct LoopDetector {
    threshold: usize,
    seen: HashMap<String, usize>,
    pub trips: u32,
    // Alternation window (audit: A->B->A->B oscillation detection).
    last: Option<String>,
    prev_last: Option<String>,
    alt_run: usize,
    // Stall window (audit: expensive cycles with no new durable state).
    stall_run: usize,
    pub stalled: bool,
    // ---- state-based loop fingerprints (P0-78) ----
    /// Typed code of the most recent fingerprint trip (consumed by the
    /// runtime to surface the code on the turn outcome).
    last_trip: Option<ReasonCode>,
    /// Patch→revert→patch alternation window: the two previous steps.
    fp_prev2: Option<Fingerprint>,
    fp_prev1: Option<Fingerprint>,
    /// Consecutive repo-state alternations under an unchanged context.
    patch_run: usize,
    /// Evidence-hash ring (bounded `EVIDENCE_RING_CAP`).
    evidence_ring: VecDeque<(u64, u64)>,
    /// Last observation for the consecutive-different-commands run.
    ev_last: Option<(u64, u64)>,
    /// Consecutive steps where the evidence hash repeated under a
    /// DIFFERENT command key.
    ev_run: usize,
    /// Fingerprint-aware failure repeats: message -> (last fingerprint,
    /// repeat count). Bounded `FAILURE_FP_CAP`.
    fail_state: HashMap<String, (u64, usize)>,
}

impl LoopDetector {
    pub fn new(threshold: usize) -> Self {
        Self {
            threshold: threshold.max(2),
            seen: HashMap::new(),
            trips: 0,
            last: None,
            prev_last: None,
            alt_run: 0,
            stall_run: 0,
            stalled: false,
            last_trip: None,
            fp_prev2: None,
            fp_prev1: None,
            patch_run: 0,
            evidence_ring: VecDeque::with_capacity(EVIDENCE_RING_CAP),
            ev_last: None,
            ev_run: 0,
            fail_state: HashMap::new(),
        }
    }

    /// Stall signal: callers feed whether the iteration added durable new
    /// state; `max_stall` consecutive no-progress iterations report a stall.
    pub fn record_progress(&mut self, made_progress: bool, max_stall: usize) -> bool {
        if made_progress {
            self.stall_run = 0;
            return false;
        }
        self.stall_run += 1;
        if self.stall_run >= max_stall.max(2) {
            self.stall_run = 0;
            self.stalled = true;
            return true;
        }
        false
    }

    /// Normalize a tool call to a stable key: name + sorted args JSON.
    /// Whitespace/order-insensitive so "the same call" is recognized.
    pub fn tool_key(name: &str, args: &serde_json::Value) -> String {
        let normalized = normalize_value(args);
        format!("{name} {normalized}")
    }

    /// Register a tool call; returns true when the loop threshold trips.
    pub fn record_tool_call(&mut self, name: &str, args: &serde_json::Value) -> bool {
        let key = Self::tool_key(name, args);
        self.record(key)
    }

    /// Register an error; returns true when the same error repeats.
    pub fn record_error(&mut self, message: &str) -> bool {
        let key = format!("err {}", message.trim());
        self.record(key)
    }

    /// Fingerprint-aware failure recording (P0-78): identical failure text
    /// with an UNCHANGED failure fingerprint counts toward the repeat
    /// threshold (the same state failed again — a loop); the same text with
    /// a CHANGED fingerprint is a NEW state (progress through the failure
    /// space) and never counts toward a repeat. Returns the typed code when
    /// `threshold` repeats of the same fingerprint occur.
    pub fn record_failure(&mut self, message: &str, fingerprint: u64) -> Option<ReasonCode> {
        let key = message.trim();
        if key.is_empty() || key.len() > 4096 {
            return None; // hostile oversized/blank failures never trip
        }
        if !self.fail_state.contains_key(key) && self.fail_state.len() >= FAILURE_FP_CAP {
            // Bounded: evict one arbitrary entry instead of growing forever.
            if let Some(old) = self.fail_state.keys().next().cloned() {
                self.fail_state.remove(&old);
            }
        }
        let entry = self
            .fail_state
            .entry(key.to_string())
            .or_insert((fingerprint, 0));
        if entry.0 != fingerprint {
            // The failure space moved: this is not a repeat, restart fresh.
            *entry = (fingerprint, 1);
            return None;
        }
        entry.1 += 1;
        if entry.1 >= self.threshold {
            *entry = (fingerprint, 0); // a trip resets the window
            self.trips += 1;
            self.last_trip = Some(ReasonCode::LoopDetected);
            return Some(ReasonCode::LoopDetected);
        }
        None
    }

    /// Feed one state step (P0-78): the caller folds the session's current
    /// repo/failure/criterion state hashes after each state-changing turn
    /// or tool batch. Detects the patch → revert → patch loop: the repo
    /// state alternating A,B,A,B while the failure fingerprint and
    /// criterion state stay UNCHANGED across the run. A changed
    /// failure/criterion dimension resets the window (state progressed).
    /// Returns the typed code when the alternation run reaches the
    /// threshold.
    pub fn record_state_step(&mut self, fp: &Fingerprint) -> Option<ReasonCode> {
        if let (Some(prev2), Some(prev1)) = (self.fp_prev2, self.fp_prev1) {
            let context_held = prev2.context() == fp.context()
                && prev1.context() == fp.context()
                && prev2.action_class == fp.action_class
                && prev1.action_class == fp.action_class;
            if context_held
                && fp.repo_state == prev2.repo_state
                && fp.repo_state != prev1.repo_state
            {
                // A -> (any change) -> A under the same context: one more
                // alternation step of the same cycle.
                self.patch_run += 1;
                if self.patch_run >= self.threshold {
                    self.patch_run = 0;
                    self.fp_prev1 = None;
                    self.fp_prev2 = None;
                    self.trips += 1;
                    let code = ReasonCode::PatchRevertPatch;
                    self.last_trip = Some(code);
                    return Some(code);
                }
            } else if fp.repo_state != prev1.repo_state || !context_held {
                // The repo moved somewhere new, or the context changed:
                // neither continues an alternation cycle.
                self.patch_run = 0;
            }
        }
        self.fp_prev2 = self.fp_prev1;
        self.fp_prev1 = Some(*fp);
        None
    }

    /// Feed one tool call's evidence observation (P0-78): the tool's
    /// normalized command key and the hash of its returned evidence set.
    /// When ≥[`REPEATED_EVIDENCE_MIN`] CONSECUTIVE calls used DIFFERENT
    /// commands yet returned the IDENTICAL evidence hash, the detector
    /// trips [`ReasonCode::RepeatedEvidenceSet`] (different greps, same
    /// evidence — the model is not converging). The observation window is
    /// the bounded `EVIDENCE_RING_CAP` ring. Identical commands repeating
    /// an evidence set are the existing identical-call detector's business
    /// and never count here.
    pub fn record_tool_evidence(
        &mut self,
        command_key: u64,
        evidence_hash: u64,
    ) -> Option<ReasonCode> {
        self.evidence_ring.push_back((command_key, evidence_hash));
        while self.evidence_ring.len() > EVIDENCE_RING_CAP {
            self.evidence_ring.pop_front();
        }
        // Run length of consecutive calls where EVERY call differs from its
        // immediate predecessor AND the evidence hash never changed across
        // the run: this call qualifies as a member when its command differs
        // from the previous one and the evidence is unchanged (grow); an
        // evidence change starts a fresh one-member run; a same-command
        // repeat (identical-call territory) ends the run empty.
        let run = match self.ev_last {
            Some((last_key, last_evidence)) => {
                if command_key == last_key {
                    0
                } else if evidence_hash == last_evidence {
                    self.ev_run + 1
                } else {
                    1
                }
            }
            None => 1,
        };
        self.ev_run = run;
        if run >= REPEATED_EVIDENCE_MIN {
            self.ev_run = 0;
            self.evidence_ring.clear();
            self.ev_last = None;
            self.trips += 1;
            let code = ReasonCode::RepeatedEvidenceSet;
            self.last_trip = Some(code);
            return Some(code);
        }
        self.ev_last = Some((command_key, evidence_hash));
        None
    }

    /// The typed code of the most recent fingerprint trip
    /// (`record_state_step` / `record_tool_evidence` / `record_failure`),
    /// for the runtime to surface on the turn outcome. None when the last
    /// trip was a text-level one (identical call/alternation) or nothing
    /// tripped.
    pub fn last_trip_code(&self) -> Option<ReasonCode> {
        self.last_trip
    }

    /// Bounded window sizes (audit: hostile churn never grows the
    /// detector).
    pub fn evidence_ring_len(&self) -> usize {
        self.evidence_ring.len()
    }

    pub fn failure_state_len(&self) -> usize {
        self.fail_state.len()
    }

    fn record(&mut self, key: String) -> bool {
        if key.len() > 4096 {
            return false; // hostile oversized keys never trip the detector
        }
        // Alternation detection (A->B->A->B oscillation): a key that
        // matches the one TWO steps back (and differs from the immediate
        // predecessor) continues a 2-cycle; `threshold` consecutive cycle
        // steps trip.
        if let Some(prev) = self.last.clone() {
            if let Some(prev2) = self.prev_last.clone() {
                if key != prev && key == prev2 {
                    self.alt_run += 1;
                    if self.alt_run >= self.threshold {
                        self.trips += 1;
                        self.alt_run = 0;
                        self.last = None;
                        self.prev_last = None;
                        self.seen.clear();
                        return true;
                    }
                } else if key != prev2 {
                    self.alt_run = 0;
                }
            }
        }
        self.prev_last = self.last.clone();
        self.last = Some(key.clone());
        let count = self.seen.entry(key).or_insert(0);
        *count += 1;
        if *count >= self.threshold {
            self.trips += 1;
            self.seen.clear(); // a trip resets the window
            true
        } else {
            false
        }
    }

    pub fn stalled(&self) -> bool {
        self.stalled
    }

    pub fn threshold(&self) -> usize {
        self.threshold
    }

    pub fn count(&self, name: &str, args: &serde_json::Value) -> usize {
        self.seen
            .get(&Self::tool_key(name, args))
            .copied()
            .unwrap_or(0)
    }

    pub fn len(&self) -> usize {
        self.seen.len()
    }

    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }
}

/// Canonical sort of JSON keys (recursive) for stable keys.
fn normalize_value(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Object(map) => {
            let mut pairs: Vec<(String, String)> = map
                .iter()
                .map(|(k, val)| (k.clone(), normalize_value(val)))
                .collect();
            pairs.sort();
            let inner: Vec<String> = pairs
                .into_iter()
                .map(|(k, val)| format!("{k}:{val}"))
                .collect();
            format!("{{{}}}", inner.join(","))
        }
        serde_json::Value::Array(items) => {
            let inner: Vec<String> = items.iter().map(normalize_value).collect();
            format!("[{}]", inner.join(","))
        }
        serde_json::Value::String(s) => format!("\"{s}\""),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_calls_trip_at_threshold() {
        let mut d = LoopDetector::new(3);
        let args = serde_json::json!({"path": "a.rs"});
        assert!(!d.record_tool_call("read_file", &args));
        assert!(!d.record_tool_call("read_file", &args));
        assert!(
            d.record_tool_call("read_file", &args),
            "3rd identical call trips"
        );
        assert_eq!(d.trips, 1);
    }

    #[test]
    fn whitespace_and_key_order_are_insensitive() {
        let a = serde_json::json!({"path": "a.rs", "limit": 10});
        let b = serde_json::json!({"limit": 10, "path": "a.rs"});
        assert_eq!(
            LoopDetector::tool_key("read_file", &a),
            LoopDetector::tool_key("read_file", &b)
        );
        let mut d = LoopDetector::new(2);
        assert!(!d.record_tool_call("read_file", &a));
        assert!(
            d.record_tool_call("read_file", &b),
            "same call, different key order"
        );
    }

    #[test]
    fn different_args_do_not_trip() {
        let mut d = LoopDetector::new(3);
        for i in 0..20 {
            let args = serde_json::json!({"path": format!("f{i}.rs")});
            assert!(
                !d.record_tool_call("read_file", &args),
                "distinct calls must never trip"
            );
        }
        assert_eq!(d.trips, 0);
        assert_eq!(d.len(), 20);
    }

    #[test]
    fn same_call_different_tool_does_not_trip() {
        let mut d = LoopDetector::new(3);
        let args = serde_json::json!({"path": "x"});
        d.record_tool_call("read_file", &args);
        d.record_tool_call("grep", &args);
        d.record_tool_call("write_file", &args);
        assert_eq!(d.trips, 0);
    }

    #[test]
    fn error_repeats_trip() {
        let mut d = LoopDetector::new(2);
        assert!(!d.record_error("cargo build failed: E0308"));
        assert!(d.record_error("cargo build failed: E0308"));
        // Slightly different error text does not trip.
        let mut d = LoopDetector::new(2);
        d.record_error("E0308");
        assert!(!d.record_error("E0425"), "different error = different loop");
    }

    #[test]
    fn hostile_oversized_keys_never_trip() {
        let mut d = LoopDetector::new(2);
        let big = serde_json::json!({"payload": "x".repeat(5000)});
        assert!(
            !d.record_tool_call("t", &big),
            "oversized key must not trip"
        );
        assert_eq!(d.trips, 0);
    }

    #[test]
    fn trip_resets_window() {
        let mut d = LoopDetector::new(3);
        let args = serde_json::json!({"a": 1});
        d.record_tool_call("t", &args);
        d.record_tool_call("t", &args);
        assert!(d.record_tool_call("t", &args));
        assert_eq!(d.count("t", &args), 0, "window reset after trip");
        assert!(
            !d.record_tool_call("t", &args),
            "fresh window needs 3 again"
        );
    }

    #[test]
    fn json_normalization_is_stable() {
        let a = serde_json::json!({"z": [1, 2], "a": {"y": "s", "x": null}});
        let b = serde_json::json!({"a": {"x": null, "y": "s"}, "z": [1, 2]});
        assert_eq!(normalize_value(&a), normalize_value(&b));
    }

    #[test]
    fn alternation_a_b_a_b_trips() {
        // A->B->A->B oscillation: pure repeats never occur yet the cycle
        // must stop. With threshold 3 the alternation trips at the third
        // step back to A.
        let mut d = LoopDetector::new(3);
        assert!(!d.record_tool_call("A", &serde_json::json!({})));
        assert!(!d.record_tool_call("B", &serde_json::json!({})));
        assert!(!d.record_tool_call("A", &serde_json::json!({}))); // alt 1
        assert!(!d.record_tool_call("B", &serde_json::json!({}))); // alt 2
        assert!(
            d.record_tool_call("A", &serde_json::json!({})),
            "the fifth step completes the alternation cycle"
        );
        assert!(d.trips >= 1);
    }

    #[test]
    fn noisy_sequence_never_trips_alternation() {
        let mut d = LoopDetector::new(5);
        // A long distinct prefix (no key repeats near the threshold), then
        // exactly FOUR clean alternation steps: below threshold, no trip.
        for name in [
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "A", "B", "A", "B",
        ] {
            let _ = d.record_tool_call(name, &serde_json::json!({}));
        }
        assert_eq!(d.trips, 0, "noise + short cycle must not trip: {}", d.trips);
        // Three more clean steps (A,B,A) complete the 5-step alternation.
        let _ = d.record_tool_call("A", &serde_json::json!({})); // alt 3
        let _ = d.record_tool_call("B", &serde_json::json!({})); // alt 4
        assert!(
            d.record_tool_call("A", &serde_json::json!({})), // alt 5 -> trip
            "the 5-step A/B alternation must trip"
        );
    }

    #[test]
    fn stall_after_max_consecutive_no_progress() {
        let mut d = LoopDetector::new(2);
        assert!(!d.record_progress(false, 3), "step 1");
        assert!(!d.record_progress(false, 3), "step 2");
        assert!(d.record_progress(false, 3), "third consecutive stall trips");
        assert!(d.stalled());
        // Progress resets the window.
        let mut d2 = LoopDetector::new(2);
        assert!(!d2.record_progress(false, 3));
        assert!(!d2.record_progress(true, 3), "progress resets");
        for _ in 0..3 {
            let _ = d2.record_progress(false, 3);
        }
        assert!(d2.stalled());
    }

    // ---- P0-78: state-based loop fingerprints ----

    /// Convenience: a fingerprint over a synthetic repo state.
    fn fp(repo: u64, failure: u64, criteria: u64, action: &'static str) -> Fingerprint {
        Fingerprint::new(repo, failure, criteria, action)
    }

    #[test]
    fn patch_revert_patch_over_six_turns_trips_patch_revert_patch() {
        // (d) The same file digest alternates A,B,A,B across turns while
        // every other state dimension is unchanged: a loop. With threshold
        // 3 the alternation trips at its fifth step (well within 6 turns).
        let mut d = LoopDetector::new(3);
        let turns = [
            fp(1, 7, 9, "patch"), // A: apply
            fp(2, 7, 9, "patch"), // B: revert
            fp(1, 7, 9, "patch"), // A: re-apply the same patch
            fp(2, 7, 9, "patch"), // B: revert again
            fp(1, 7, 9, "patch"), // A ...
            fp(2, 7, 9, "patch"),
        ];
        for (i, step) in turns.iter().enumerate() {
            let code = d.record_state_step(step);
            assert_eq!(
                code,
                if i == 4 {
                    Some(ReasonCode::PatchRevertPatch)
                } else {
                    None
                },
                "the alternation run trips at its fifth step (well within 6 turns)"
            );
        }
        assert_eq!(d.last_trip_code(), Some(ReasonCode::PatchRevertPatch));
    }

    #[test]
    fn patch_revert_patch_resets_when_the_failure_space_moves() {
        // The repo oscillates BUT the failure fingerprint moves on EVERY
        // step (the task keeps progressing through the failure space):
        // never a loop signal — a run can only build under a context that
        // stays still.
        let mut d = LoopDetector::new(3);
        let turns = [
            fp(1, 101, 9, "patch"),
            fp(2, 102, 9, "patch"),
            fp(1, 103, 9, "patch"),
            fp(2, 104, 9, "patch"),
            fp(1, 105, 9, "patch"),
            fp(2, 106, 9, "patch"),
            fp(1, 107, 9, "patch"),
            fp(2, 108, 9, "patch"),
        ];
        for step in &turns {
            assert_eq!(
                d.record_state_step(step),
                None,
                "a moving failure fingerprint is progress, never a patch-revert-patch loop"
            );
        }
        assert_eq!(d.trips, 0);
        assert_eq!(d.last_trip_code(), None);
        // ... and the criteria state moving on every step resets it too.
        let mut d2 = LoopDetector::new(3);
        for step in [
            fp(1, 7, 901, "patch"),
            fp(2, 7, 902, "patch"),
            fp(1, 7, 903, "patch"),
            fp(2, 7, 904, "patch"),
            fp(1, 7, 905, "patch"),
            fp(2, 7, 906, "patch"),
            fp(1, 7, 907, "patch"),
            fp(2, 7, 908, "patch"),
            fp(1, 7, 909, "patch"),
            fp(2, 7, 910, "patch"),
        ] {
            assert_eq!(
                d2.record_state_step(&step),
                None,
                "a moving criterion state is progress, never a loop"
            );
        }
        assert_eq!(d2.trips, 0);
        // A single mid-run change pauses the run; a settled context that
        // resumes oscillating is a loop again (the failure stopped moving).
        let mut d3 = LoopDetector::new(3);
        for (i, step) in [
            fp(1, 7, 9, "patch"),
            fp(2, 7, 9, "patch"),
            fp(1, 8, 9, "patch"), // fingerprint moved: run resets here
            fp(2, 8, 9, "patch"),
            fp(1, 8, 9, "patch"),
            fp(2, 8, 9, "patch"),
            fp(1, 8, 9, "patch"),
        ]
        .iter()
        .enumerate()
        {
            let code = d3.record_state_step(step);
            if i == 6 {
                assert_eq!(
                    code,
                    Some(ReasonCode::PatchRevertPatch),
                    "once the context settles at fp 8, the oscillation is a loop again"
                );
            } else {
                assert_eq!(code, None);
            }
        }
    }

    #[test]
    fn monotonic_repo_progression_never_trips_patch_revert_patch() {
        let mut d = LoopDetector::new(3);
        for repo in 0..40u64 {
            assert_eq!(
                d.record_state_step(&fp(repo, 7, 9, "patch")),
                None,
                "a monotonic repo trajectory (even failing) never alternates"
            );
        }
        assert_eq!(d.trips, 0);
    }

    #[test]
    fn failure_fingerprint_changes_across_identical_failure_turns_never_loop() {
        // (c) Four turns with the SAME failure text but a CHANGED failure
        // fingerprint: the state is progressing through the failure space —
        // never loop-signaled.
        let mut d = LoopDetector::new(3);
        for turn in 1u64..=4 {
            // One identical-text failure per turn, fingerprint moving every
            // turn: the state is progressing through the failure space.
            assert_eq!(
                d.record_failure("cargo check failed: E0308", 100 + turn),
                None,
                "turn {turn}: a moved failure fingerprint must never count as a repeat"
            );
        }
        assert_eq!(d.trips, 0);
        assert_eq!(d.last_trip_code(), None);
        // Identical failures with an UNCHANGED fingerprint loop at the
        // existing threshold.
        let mut d2 = LoopDetector::new(3);
        assert_eq!(d2.record_failure("cargo check failed: E0308", 5), None);
        assert_eq!(d2.record_failure("cargo check failed: E0308", 5), None);
        assert_eq!(
            d2.record_failure("cargo check failed: E0308", 5),
            Some(ReasonCode::LoopDetected),
            "3 identical failures of the same fingerprint state trip at the threshold"
        );
        // A fingerprint change between repeats resets that failure's count.
        let mut d3 = LoopDetector::new(3);
        assert_eq!(d3.record_failure("boom", 1), None);
        assert_eq!(d3.record_failure("boom", 1), None);
        assert_eq!(
            d3.record_failure("boom", 2),
            None,
            "changed fingerprint resets"
        );
        assert_eq!(d3.record_failure("boom", 2), None);
        assert_eq!(
            d3.record_failure("boom", 2),
            Some(ReasonCode::LoopDetected),
            "fresh same-fingerprint run reaches the threshold again"
        );
    }

    #[test]
    fn hostile_oversized_failures_never_trip_the_fingerprint_path() {
        let mut d = LoopDetector::new(2);
        let huge = "x".repeat(5000);
        assert_eq!(d.record_failure(&huge, 1), None);
        assert_eq!(d.record_failure(&huge, 1), None);
        assert_eq!(d.record_failure("", 1), None);
        assert_eq!(d.trips, 0);
    }

    #[test]
    fn different_greps_returning_the_same_evidence_set_trip() {
        // (e) Three CONSECUTIVE tool calls with DIFFERENT commands that all
        // return the identical evidence set: a retrieval loop.
        let ev = 4242u64;
        let mut d = LoopDetector::new(3);
        assert_eq!(
            d.record_tool_evidence(hash_u64(&"grep pattern-a"), ev),
            None
        );
        assert_eq!(
            d.record_tool_evidence(hash_u64(&"grep -i pattern-a"), ev),
            None
        );
        assert_eq!(
            d.record_tool_evidence(hash_u64(&"rg pattern-a src/"), ev),
            Some(ReasonCode::RepeatedEvidenceSet),
            "3rd different command with the same evidence set trips"
        );
        assert_eq!(d.last_trip_code(), Some(ReasonCode::RepeatedEvidenceSet));
        assert!(d.trips >= 1);
        // ... and only ≥3 do: two different commands with the same evidence
        // are a coincidence, not a loop.
        let mut d2 = LoopDetector::new(3);
        assert_eq!(d2.record_tool_evidence(1, ev), None);
        assert_eq!(d2.record_tool_evidence(2, ev), None);
        assert_eq!(d2.trips, 0);
    }

    #[test]
    fn evidence_churn_and_identical_commands_never_false_trip() {
        // A moving evidence set resets the run.
        let mut d = LoopDetector::new(3);
        for i in 0..20u64 {
            assert_eq!(
                d.record_tool_evidence(i, i * 7),
                None,
                "distinct evidence per command never trips"
            );
        }
        assert_eq!(d.trips, 0);
        // The SAME command repeating the same evidence set is the existing
        // identical-call detector's business — never a repeated-evidence
        // code.
        let mut d2 = LoopDetector::new(3);
        for _ in 0..10 {
            assert_eq!(
                d2.record_tool_evidence(hash_u64(&"grep same"), ev()),
                None,
                "identical commands with identical evidence never trip the evidence code"
            );
        }
        assert_eq!(d2.trips, 0);
        // Interleaved same-command repeats do not trip by themselves and do
        // not break the consecutive-different-command chain: a,a,b,c,d
        // still has the three consecutive different commands b,c,d.
        let mut d3 = LoopDetector::new(3);
        let ev3 = ev();
        assert_eq!(d3.record_tool_evidence(hash_u64(&"a"), ev3), None);
        assert_eq!(d3.record_tool_evidence(hash_u64(&"a"), ev3), None);
        assert_eq!(d3.record_tool_evidence(hash_u64(&"b"), ev3), None);
        assert_eq!(d3.record_tool_evidence(hash_u64(&"c"), ev3), None);
        assert_eq!(
            d3.record_tool_evidence(hash_u64(&"d"), ev3),
            Some(ReasonCode::RepeatedEvidenceSet),
            "b,c,d are three consecutive different commands with the identical evidence set"
        );
        // ... and even a,b,a (three consecutive calls, each a different
        // command from its predecessor, all with the identical evidence
        // set) trips: the evidence is not converging.
        let mut d4 = LoopDetector::new(3);
        let ev4 = ev();
        assert_eq!(d4.record_tool_evidence(hash_u64(&"a"), ev4), None);
        assert_eq!(d4.record_tool_evidence(hash_u64(&"b"), ev4), None);
        assert_eq!(
            d4.record_tool_evidence(hash_u64(&"a"), ev4),
            Some(ReasonCode::RepeatedEvidenceSet)
        );
    }

    fn ev() -> u64 {
        hash_u64(&"same evidence set: file.rs:12 fn parse() {}")
    }

    #[test]
    fn bounded_windows_survive_hostile_churn() {
        // (f) Bounded memory: 10k evidence observations stay within the
        // 16-entry ring; 10k fingerprint-aware failures across 200 messages
        // stay within the failure map cap.
        let mut d = LoopDetector::new(3);
        for i in 0..10_000u64 {
            let _ = d.record_tool_evidence(i, i.wrapping_mul(31) % 7);
            let _ = d.record_failure(&format!("failure {i}"), i % 200);
            let _ = d.record_state_step(&fp(i % 5, i % 3, i % 2, "patch"));
        }
        assert!(
            d.evidence_ring_len() <= EVIDENCE_RING_CAP,
            "the evidence ring is bounded: {}",
            d.evidence_ring_len()
        );
        assert!(
            d.failure_state_len() <= FAILURE_FP_CAP,
            "the failure-fingerprint map is bounded: {}",
            d.failure_state_len()
        );
        assert!(
            d.trips < 10_000 / 10,
            "no pathological tripping under churn"
        );
    }

    #[test]
    fn record_state_step_does_not_disturb_text_level_detection() {
        // Fingerprint feeds and text feeds are independent bookkeeping: an
        // A/B/A repo cycle below the trip count never trips the text
        // detectors, and identical text calls still trip at the threshold.
        let mut d = LoopDetector::new(3);
        for step in [
            fp(1, 7, 9, "patch"),
            fp(2, 7, 9, "patch"),
            fp(1, 7, 9, "patch"),
        ] {
            assert_eq!(d.record_state_step(&step), None);
        }
        let args = serde_json::json!({"path": "a.rs"});
        assert!(!d.record_tool_call("read_file", &args));
        assert!(!d.record_tool_call("read_file", &args));
        assert!(
            d.record_tool_call("read_file", &args),
            "identical calls still trip at the threshold"
        );
    }
}
