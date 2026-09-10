//! Verified-outcome learning for the economic router (audit items 13/14/L).
//!
//! Expected success is a PER-CALL probabilistic prior; the router must
//! learn *verified rework* from durable history, not from "the model said
//! it was done". This module owns the estimate math and the outcome
//! registry surface:
//!
//! - [`OutcomeKey`] — provider / model / phase / task_class / risk_bucket;
//! - [`VerifiedOutcomeStats`] — durable per-key accumulators;
//! - conservative estimators: [`verified_success_confidence_ppm`]
//!   (MaximumQuality consumption) and [`rework_probability_ppm`]
//!   (Economy consumption). Both are Wilson one-sided bounds with z = 2
//!   (~97.5% confidence): low samples stay conservative and a model is
//!   NEVER "excellent after two successes" (two clean samples sit at
//!   ≈ 33% verified-success confidence / ≈ 67% rework risk). The bounds
//!   converge to the observed rates as the sample grows;
//! - [`work_cost_estimate`] — economic ranking = immediate cost +
//!   P(rework) x expected downstream spend;
//! - [`OutcomeStore`] — the additive registry handle a
//!   [`RouterService`](crate::RouterService) consults when stats exist,
//!   with the in-process [`MemoryOutcomeStore`] and the default
//!   [`EmptyOutcomeStore`]. Durable backing is the store crate's v18
//!   `model_outcome_stats` table; a store-backed `OutcomeStore` impl is a
//!   wiring-crate concern (see the router crate docs / residual risks).
//!
//! Verified-only attribution: a sample is recorded as a FIRST-PASS SUCCESS
//! only when the caller holds an explicit verified-success signal (the
//! task ultimately passed deterministic verification AND attribution
//! identified this model call/phase). `append_sample` is the ONLY
//! success-recording entry point and it takes that signal explicitly — a
//! caller that only knows "the model said done" must pass
//! `verified_success = false`, which counts a FAILURE sample (rework was
//! needed), never a success.

use std::collections::HashMap;
use std::sync::Mutex;

use faktor_core::model::{RiskBucket, RouterPhase, TaskClass};

/// Default verified first-pass success prior in ppm (0.8) — mirrors the
/// router's telemetry `DEFAULT_SUCCESS_PPM`; a fresh key inherits the same
/// prior the attempt-level telemetry uses so the two surfaces agree when
/// no verified history exists.
pub const DEFAULT_VERIFIED_SUCCESS_PPM: u32 = 800_000;

/// Wilson z for the conservative one-sided bounds: z = 2 ≈ 97.5% one-sided
/// confidence. Larger z = more conservative small-sample behavior; the
/// whole point is that two clean samples must not clear an "excellent"
/// bar (see [`EXCELLENT_CONFIDENCE_PPM`]).
pub const WILSON_Z: f64 = 2.0;

/// The documented "excellent" bar for verified-success confidence in ppm:
/// a key clears it only after enough clean samples (36 clean successes
/// out of 36 sits exactly at the bar; fewer stays below it). MaximumQuality
/// wiring consumes the conservative confidence against this bar.
pub const EXCELLENT_CONFIDENCE_PPM: u32 = 900_000;

/// One outcome dimension key. Every verified sample is recorded against
/// the FULL key (the runtime knows its task class and semantic risk at
/// settlement time); routing consults the hierarchy through
/// [`OutcomeStore::lookup_stats`] — exact key, then the task-level fold,
/// then the per-phase fold, then the caller's telemetry/the global prior —
/// never collapsing straight to the prior while a narrower key holds
/// evidence.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct OutcomeKey {
    pub provider: String,
    pub model: String,
    pub phase: RouterPhase,
    pub task_class: TaskClass,
    pub risk_bucket: RiskBucket,
}

/// Durable per-key verified-outcome accumulators (store v18
/// `model_outcome_stats` projection columns).
///
/// Invariant (enforced by [`VerifiedOutcomeStats::absorb`] and mirrored by
/// the store's append): `sample_count = successes_first_pass +
/// failures_first_pass`, and rework sums accumulate ONLY on failure
/// samples — a verified first-pass success cannot cause rework.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct VerifiedOutcomeStats {
    /// Attempts whose call the task ultimately passed deterministic
    /// verification on its first pass (verified signal + attribution).
    pub successes_first_pass: u64,
    /// Attempts that did NOT pass verification (they needed downstream
    /// rework spend before the task verified).
    pub failures_first_pass: u64,
    /// Total downstream spend (microUSD) the recorded failures caused
    /// until their tasks verified (0 per success sample, always).
    pub rework_cost_micro_sum: u64,
    /// Total downstream TURNS the recorded failures caused.
    pub rework_turns_sum: u64,
    /// Total recorded samples = successes_first_pass + failures_first_pass.
    pub sample_count: u64,
}

impl VerifiedOutcomeStats {
    /// Absorb one verified sample, saturating every accumulator.
    /// `verified_success = false` counts a FAILURE sample and is the only
    /// way rework sums grow; `true` counts a first-pass success and never
    /// contributes rework (a success sample that carries a nonzero
    /// `rework_cost_micro`/`rework_turns` in the sample is ignored for the
    /// rework sums — a verified first-pass success cannot cause rework).
    pub fn absorb(&mut self, sample: OutcomeSample) {
        self.sample_count = self.sample_count.saturating_add(1);
        if sample.verified_success {
            self.successes_first_pass = self.successes_first_pass.saturating_add(1);
        } else {
            self.failures_first_pass = self.failures_first_pass.saturating_add(1);
            self.rework_cost_micro_sum = self
                .rework_cost_micro_sum
                .saturating_add(sample.rework_cost_micro);
            self.rework_turns_sum = self.rework_turns_sum.saturating_add(sample.rework_turns);
        }
    }
}

/// ONE verified outcome fact: the attempt's explicit verified-success
/// signal plus the rework its failure eventually caused (recorded when the
/// task closes and the downstream spend is attributable). Success samples
/// carry zero rework.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct OutcomeSample {
    /// The explicit verified-success signal: the task ultimately passed
    /// deterministic verification AND attribution identified this model
    /// call/phase. "The model said done" is NOT this signal.
    pub verified_success: bool,
    /// Downstream spend this attempt's failure caused until its task
    /// verified (microUSD). Ignored on success samples.
    pub rework_cost_micro: u64,
    /// Downstream turns this attempt's failure caused. Ignored on success
    /// samples.
    pub rework_turns: u64,
}

fn wilson_terms(successes: u64, failures: u64, total: u64, upper: bool) -> f64 {
    let s = successes as f64;
    let f = failures as f64;
    let n = total as f64;
    let z = WILSON_Z;
    let z2 = z * z;
    let inner = (s * f / n) + (z2 / 4.0);
    let root = z * inner.sqrt();
    if upper {
        // Upper bound on the FAILURE rate f/n.
        ((f + z2 / 2.0 + root) / (n + z2)).min(1.0)
    } else {
        // Lower bound on the SUCCESS rate s/n.
        ((s + z2 / 2.0 - root) / (n + z2)).max(0.0)
    }
}

fn ppm_of(rate: f64) -> u32 {
    rate.clamp(0.0, 1.0).mul_add(1_000_000.0, 0.0).round() as u32
}

/// Conservative verified-success confidence in ppm: the one-sided Wilson
/// LOWER bound (z = [`WILSON_Z`]) on the true first-pass verified-success
/// rate. Low samples shrink the confidence far below the observed rate
/// (never "excellent after two successes": 2/2 sits at ≈ 333k ppm, below
/// the [`EXCELLENT_CONFIDENCE_PPM`] bar of 900k; 36/36 clean reaches the
/// bar exactly, and large clean samples converge toward the observed
/// rate). `sample_count == 0` carries no evidence: 0 ppm.
pub fn verified_success_confidence_ppm(stats: &VerifiedOutcomeStats) -> u32 {
    if stats.sample_count == 0 {
        return 0;
    }
    let successes = stats.successes_first_pass.min(stats.sample_count);
    let failures = stats.sample_count.saturating_sub(successes);
    ppm_of(wilson_terms(successes, failures, stats.sample_count, false))
}

/// Conservative rework probability in ppm for the ECONOMY ranking: the
/// one-sided Wilson UPPER bound (z = [`WILSON_Z`]) on the observed
/// failure rate `failures_first_pass / sample_count`. Under small samples
/// the bound stays high (two clean successes still imply ≈ 667k ppm of
/// rework risk — a two-sample track record is not a zero-rework license);
/// it converges to the observed rate as samples grow. `sample_count == 0`
/// carries no evidence: 0 ppm.
pub fn rework_probability_ppm(stats: &VerifiedOutcomeStats) -> u32 {
    if stats.sample_count == 0 {
        return 0;
    }
    let failures = stats.failures_first_pass.min(stats.sample_count);
    let successes = stats.sample_count.saturating_sub(failures);
    ppm_of(wilson_terms(successes, failures, stats.sample_count, true))
}

/// One candidate's expected-cost estimate: economic ranking =
/// `immediate_cost_micro + P(rework) x expected downstream spend`.
///
/// - `immediate_cost_micro` — the candidate's own base per-call cost;
/// - `stats = Some(..)` — measured history exists: `rework_probability_ppm`
///   is the conservative Wilson upper bound and the expected downstream
///   spend per rework is the MEASURED mean
///   (`rework_cost_micro_sum / failures_first_pass`, rounded UP — never
///   understated) when any failure with measured spend exists, else the
///   caller's `unmeasured_rework_spend_micro` fallback (the routing graph
///   passes its escalation cost: a rework costs at least one full
///   escalation call);
/// - `stats = None` — no history: the documented default failure prior
///   (200k ppm, mirroring the 800k success prior) against the same
///   fallback spend.
///
/// Integer-exact, saturating: the probability term rounds UP to the next
/// whole micro and hostile magnitudes saturate instead of panicking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WorkCostEstimate {
    pub immediate_cost_micro: u64,
    /// Conservative rework probability in ppm (0..=1_000_000).
    pub rework_probability_ppm: u32,
    /// P(rework) x expected downstream spend per rework, rounded up.
    pub expected_rework_micro: u64,
    /// immediate + expected rework (saturating).
    pub total_expected_micro: u64,
}

/// One work-cost estimate per the documented ranking. See
/// [`WorkCostEstimate`] for the parameter semantics.
pub fn work_cost_estimate(
    immediate_cost_micro: u64,
    stats: Option<&VerifiedOutcomeStats>,
    unmeasured_rework_spend_micro: u64,
) -> WorkCostEstimate {
    let (rework_probability_ppm, mean_rework_spend_micro) = match stats {
        None => (
            1_000_000 - DEFAULT_VERIFIED_SUCCESS_PPM,
            unmeasured_rework_spend_micro,
        ),
        Some(st) => {
            let ppm = rework_probability_ppm(st);
            let measured = if st.failures_first_pass > 0 && st.rework_cost_micro_sum > 0 {
                Some(st.rework_cost_micro_sum.div_ceil(st.failures_first_pass))
            } else {
                None
            };
            (ppm, measured.unwrap_or(unmeasured_rework_spend_micro))
        }
    };
    let expected = u128::from(mean_rework_spend_micro)
        .saturating_mul(u128::from(rework_probability_ppm))
        .saturating_add(999_999);
    let expected_rework_micro = u64::try_from(expected / 1_000_000).unwrap_or(u64::MAX);
    WorkCostEstimate {
        immediate_cost_micro,
        rework_probability_ppm,
        expected_rework_micro,
        total_expected_micro: immediate_cost_micro.saturating_add(expected_rework_micro),
    }
}

/// The additive outcome-registry handle a
/// [`RouterService`](crate::RouterService) is built with
/// ([`with_outcomes`](crate::RouterService::with_outcomes), defaulting to
/// an empty registry — scoring is byte-identical while no stats exist).
///
/// The durable implementation is a wiring-crate concern over the store
/// crate's v18 `model_outcome_stats` table (append/read fns); the trait
/// keeps the router independent of the store crate.
pub trait OutcomeStore: Send + Sync {
    /// Append ONE verified sample (event fact) for a key. The caller holds
    /// the explicit verified-success signal; see [`OutcomeSample`].
    fn append_sample(&self, key: &OutcomeKey, sample: OutcomeSample);

    /// Per-key stats.
    fn stats(&self, key: &OutcomeKey) -> Option<VerifiedOutcomeStats>;

    /// Stats summed over every task_class/risk_bucket recorded for one
    /// (provider, model, phase) — the broadest (provider, model, phase)
    /// fold. The sums are saturating and the invariant columns stay
    /// consistent (`sample_count = successes + failures`).
    fn phase_stats(
        &self,
        provider: &str,
        model: &str,
        phase: RouterPhase,
    ) -> Option<VerifiedOutcomeStats>;

    /// Stats summed over every risk_bucket recorded for one
    /// (provider, model, phase, task_class). The default is `None` (a
    /// store with no class-level index degrades to the phase fold, never
    /// to a fabricated row).
    fn task_stats(
        &self,
        _provider: &str,
        _model: &str,
        _phase: RouterPhase,
        _task_class: TaskClass,
    ) -> Option<VerifiedOutcomeStats> {
        None
    }

    /// THE outcome consult hierarchy (audit items 13/14/L): narrowest key
    /// first, NEVER collapsing straight to the global prior while a
    /// narrower key holds evidence:
    ///
    /// 1. `(provider, model, phase, task_class, risk_bucket)` — exact key;
    /// 2. `(provider, model, phase, task_class)` — every risk bucket folded;
    /// 3. `(provider, model, phase)` — every class/risk bucket folded;
    /// 4. `None` — no evidence: the caller falls back to its telemetry/global
    ///    prior (never a fabricated zero-rework row).
    fn lookup_stats(
        &self,
        provider: &str,
        model: &str,
        phase: RouterPhase,
        task_class: TaskClass,
        risk_bucket: RiskBucket,
    ) -> Option<VerifiedOutcomeStats> {
        let exact = OutcomeKey {
            provider: provider.to_string(),
            model: model.to_string(),
            phase,
            task_class,
            risk_bucket,
        };
        if let Some(stats) = self.stats(&exact) {
            return Some(stats);
        }
        if let Some(stats) = self.task_stats(provider, model, phase, task_class) {
            return Some(stats);
        }
        self.phase_stats(provider, model, phase)
    }
}

/// Default empty registry: every consult misses, every append is a no-op.
/// Scoring over this registry is byte-identical to the pre-outcome router.
#[derive(Debug, Clone, Copy, Default)]
pub struct EmptyOutcomeStore;

impl OutcomeStore for EmptyOutcomeStore {
    fn append_sample(&self, _key: &OutcomeKey, _sample: OutcomeSample) {}

    fn stats(&self, _key: &OutcomeKey) -> Option<VerifiedOutcomeStats> {
        None
    }

    fn phase_stats(
        &self,
        _provider: &str,
        _model: &str,
        _phase: RouterPhase,
    ) -> Option<VerifiedOutcomeStats> {
        None
    }
}

/// In-process outcome registry (Mutex-protected). The corpus gates and
/// router tests feed and consult verified history through this; the
/// production daemon swaps in a store-backed implementation over the same
/// trait.
#[derive(Debug, Default)]
pub struct MemoryOutcomeStore {
    inner: Mutex<HashMap<OutcomeKey, VerifiedOutcomeStats>>,
}

impl MemoryOutcomeStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of distinct keys currently holding at least one sample.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl OutcomeStore for MemoryOutcomeStore {
    fn append_sample(&self, key: &OutcomeKey, sample: OutcomeSample) {
        let mut inner = self.inner.lock().unwrap();
        let entry = inner.entry(key.clone()).or_default();
        entry.absorb(sample);
    }

    fn stats(&self, key: &OutcomeKey) -> Option<VerifiedOutcomeStats> {
        self.inner.lock().unwrap().get(key).copied()
    }

    fn phase_stats(
        &self,
        provider: &str,
        model: &str,
        phase: RouterPhase,
    ) -> Option<VerifiedOutcomeStats> {
        let inner = self.inner.lock().unwrap();
        let mut acc: Option<VerifiedOutcomeStats> = None;
        for (key, stats) in inner.iter() {
            if key.provider == provider && key.model == model && key.phase == phase {
                match acc.as_mut() {
                    Some(a) => a.absorb_aggregate(*stats),
                    None => acc = Some(*stats),
                }
            }
        }
        acc
    }

    fn task_stats(
        &self,
        provider: &str,
        model: &str,
        phase: RouterPhase,
        task_class: TaskClass,
    ) -> Option<VerifiedOutcomeStats> {
        let inner = self.inner.lock().unwrap();
        let mut acc: Option<VerifiedOutcomeStats> = None;
        for (key, stats) in inner.iter() {
            if key.provider == provider
                && key.model == model
                && key.phase == phase
                && key.task_class == task_class
            {
                match acc.as_mut() {
                    Some(a) => a.absorb_aggregate(*stats),
                    None => acc = Some(*stats),
                }
            }
        }
        acc
    }
}

impl VerifiedOutcomeStats {
    /// Saturating column-wise sum of two per-key accumulator rows (the
    /// registry's phase consult folds every class/risk bucket of one
    /// (provider, model, phase) into one row).
    fn absorb_aggregate(&mut self, other: VerifiedOutcomeStats) {
        self.successes_first_pass = self
            .successes_first_pass
            .saturating_add(other.successes_first_pass);
        self.failures_first_pass = self
            .failures_first_pass
            .saturating_add(other.failures_first_pass);
        self.rework_cost_micro_sum = self
            .rework_cost_micro_sum
            .saturating_add(other.rework_cost_micro_sum);
        self.rework_turns_sum = self.rework_turns_sum.saturating_add(other.rework_turns_sum);
        self.sample_count = self.sample_count.saturating_add(other.sample_count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(
        successes: u64,
        failures: u64,
        rework_cost: u64,
        rework_turns: u64,
    ) -> VerifiedOutcomeStats {
        VerifiedOutcomeStats {
            successes_first_pass: successes,
            failures_first_pass: failures,
            rework_cost_micro_sum: rework_cost,
            rework_turns_sum: rework_turns,
            sample_count: successes + failures,
        }
    }

    #[test]
    fn two_clean_samples_never_clear_the_excellent_bar() {
        // The audit's core small-sample rule: after TWO verified first-pass
        // successes the conservative confidence stays far below "excellent"
        // (and far below the observed 100%).
        let two = stats(2, 0, 0, 0);
        let conf = verified_success_confidence_ppm(&two);
        assert!(
            conf < EXCELLENT_CONFIDENCE_PPM,
            "2/2 confidence {conf} must stay below the excellent bar"
        );
        assert!(
            (300_000..=370_000).contains(&conf),
            "2/2 lower bound must sit near 333k ppm, got {conf}"
        );
        // The economy side mirrors the same conservatism: a two-sample
        // track record is NOT a zero-rework license.
        let rp = rework_probability_ppm(&two);
        assert!(
            (600_000..=700_000).contains(&rp),
            "2 clean samples must keep >= 60% conservative rework risk, got {rp}"
        );
        // The observed rate (100%) is NOT the estimate — the bound is.
        assert!(conf < 800_000);
    }

    #[test]
    fn confidence_converges_only_with_samples_and_clears_the_bar_exactly_at_36_clean() {
        // Monotone in the number of clean samples: 2 -> 36 -> 100.
        let prev = verified_success_confidence_ppm(&stats(2, 0, 0, 0));
        let at_bar = verified_success_confidence_ppm(&stats(36, 0, 0, 0));
        let saturated = verified_success_confidence_ppm(&stats(100, 0, 0, 0));
        assert!(prev < at_bar && at_bar < saturated);
        // 36/36 sits exactly at the documented excellent bar (s/(s+4) when
        // no failures were observed, 36/40 = 0.9) — the boundary the audit
        // demands: only a substantial clean record clears it.
        assert_eq!(at_bar, EXCELLENT_CONFIDENCE_PPM);
        assert!(
            saturated > EXCELLENT_CONFIDENCE_PPM,
            "100 clean samples converge past the bar"
        );
        // Failures bite: 90/100 clean is far less confident than 100/100.
        let with_failures = verified_success_confidence_ppm(&stats(90, 10, 0, 0));
        assert!(with_failures < saturated);
        // No samples: no evidence at all.
        assert_eq!(verified_success_confidence_ppm(&stats(0, 0, 0, 0)), 0);
        // Hostile: inconsistent rows clamp, never panic.
        let hostile = VerifiedOutcomeStats {
            successes_first_pass: u64::MAX,
            failures_first_pass: u64::MAX,
            sample_count: 1,
            ..Default::default()
        };
        assert!(verified_success_confidence_ppm(&hostile) <= 1_000_000);
        assert!(rework_probability_ppm(&hostile) <= 1_000_000);
    }

    #[test]
    fn rework_estimate_is_conservative_and_converges() {
        // The corpus shape: 45% observed rework over 200 samples vs 6% over
        // 100 — the conservative bounds keep the flaky model's risk above
        // its observed rate and the reliable model's risk below the flaky
        // one by a wide margin.
        let flaky = stats(110, 90, 90 * 2_400_000, 90 * 3);
        let reliable = stats(94, 6, 6 * 960_000, 6 * 2);
        let flaky_ppm = rework_probability_ppm(&flaky);
        let reliable_ppm = rework_probability_ppm(&reliable);
        assert!(
            flaky_ppm > 500_000 && flaky_ppm < 560_000,
            "flaky 45% observed must stay conservative ~52%, got {flaky_ppm}"
        );
        assert!(
            (110_000..=150_000).contains(&reliable_ppm),
            "reliable 6% observed must stay conservative ~12-13%, got {reliable_ppm}"
        );
        assert!(flaky_ppm > reliable_ppm);
        // Zero observed failures on a large sample still leaves a floor
        // (the classic rule-of-three shape: z^2/(n+z^2)).
        let clean_1000 = rework_probability_ppm(&stats(1000, 0, 0, 0));
        assert!(
            clean_1000 > 0 && clean_1000 < 10_000,
            "1000 clean samples: rework floor stays small but nonzero, got {clean_1000}"
        );
    }

    #[test]
    fn work_cost_is_integer_exact_saturating_and_never_understates() {
        // 2 clean successes: 666_667 ppm x 500_000 unmeasured escalation
        // spend -> expected rework 333_334 (rounded UP), total 391_334.
        let two = stats(2, 0, 0, 0);
        let est = work_cost_estimate(58_000, Some(&two), 500_000);
        assert_eq!(est.immediate_cost_micro, 58_000);
        assert_eq!(est.rework_probability_ppm, 666_667);
        assert_eq!(est.expected_rework_micro, 333_334);
        assert_eq!(est.total_expected_micro, 391_334);
        // Measured mean spend dominates the fallback once failures carry
        // spend: 90 failures x 2.4M each -> mean 2_400_000 exactly.
        let flaky = stats(110, 90, 90 * 2_400_000, 90 * 3);
        let est = work_cost_estimate(58_000, Some(&flaky), 500_000);
        assert_eq!(est.rework_probability_ppm, 520_650);
        assert_eq!(est.expected_rework_micro, 1_249_560);
        assert_eq!(est.total_expected_micro, 1_307_560);
        // No stats: the documented 200k default failure prior against the
        // fallback spend.
        let est = work_cost_estimate(58_000, None, 500_000);
        assert_eq!(est.rework_probability_ppm, 200_000);
        assert_eq!(est.expected_rework_micro, 100_000);
        assert_eq!(est.total_expected_micro, 158_000);
        // Hostile magnitudes saturate, never panic.
        let hostile = VerifiedOutcomeStats {
            failures_first_pass: 1,
            rework_cost_micro_sum: u64::MAX,
            sample_count: 1,
            ..Default::default()
        };
        let est = work_cost_estimate(u64::MAX, Some(&hostile), u64::MAX);
        assert_eq!(est.total_expected_micro, u64::MAX);
        let est = work_cost_estimate(u64::MAX, None, u64::MAX);
        assert_eq!(est.total_expected_micro, u64::MAX);
    }

    #[test]
    fn verified_only_attribution_and_success_samples_never_carry_rework() {
        let mut acc = VerifiedOutcomeStats::default();
        // "The model said done" but verification failed: this is a FAILURE
        // sample, never a success, and it carries the eventual rework.
        for _ in 0..3 {
            acc.absorb(OutcomeSample {
                verified_success: false,
                rework_cost_micro: 900_000,
                rework_turns: 2,
            });
        }
        assert_eq!(acc.successes_first_pass, 0, "no success may be learned");
        assert_eq!(acc.failures_first_pass, 3);
        assert_eq!(acc.rework_cost_micro_sum, 2_700_000);
        assert_eq!(acc.sample_count, 3);
        // A genuine verified first-pass success never contributes rework,
        // even if a hostile caller hands a nonzero cost/turn count.
        acc.absorb(OutcomeSample {
            verified_success: true,
            rework_cost_micro: u64::MAX,
            rework_turns: u64::MAX,
        });
        assert_eq!(acc.successes_first_pass, 1);
        assert_eq!(acc.failures_first_pass, 3);
        assert_eq!(acc.rework_cost_micro_sum, 2_700_000);
        assert_eq!(acc.rework_turns_sum, 6);
        assert_eq!(acc.sample_count, 4);
        // Saturating absorption of hostile accumulators never panics.
        let mut maxed = VerifiedOutcomeStats {
            sample_count: u64::MAX - 1,
            ..Default::default()
        };
        maxed.absorb(OutcomeSample {
            verified_success: true,
            rework_cost_micro: u64::MAX,
            rework_turns: u64::MAX,
        });
        assert_eq!(maxed.sample_count, u64::MAX);
    }

    #[test]
    fn lookup_walks_exact_then_task_then_phase_and_never_fabricates() {
        // The consult hierarchy of audit items 13/14/L: an exact key
        // dominates a task-level fold, which dominates the phase fold; with
        // no evidence at any level the caller's prior is used (None), never
        // a fabricated zero-rework row.
        let store = MemoryOutcomeStore::new();
        let key = |class: TaskClass, bucket: RiskBucket| OutcomeKey {
            provider: "p".into(),
            model: "m".into(),
            phase: RouterPhase::Implement,
            task_class: class,
            risk_bucket: bucket,
        };
        let sample = |verified_success: bool, rework: u64| OutcomeSample {
            verified_success,
            rework_cost_micro: rework,
            rework_turns: 1,
        };
        // Exact: Hard/High (one clean). Task-level: Medium folded over two
        // risk buckets; Easy carries NO row of its own.
        store.append_sample(&key(TaskClass::Hard, RiskBucket::High), sample(true, 0));
        store.append_sample(&key(TaskClass::Medium, RiskBucket::Low), sample(false, 100));
        store.append_sample(&key(TaskClass::Medium, RiskBucket::High), sample(true, 0));

        // 1. Exact key wins and matches its own row exactly.
        let exact = store
            .lookup_stats(
                "p",
                "m",
                RouterPhase::Implement,
                TaskClass::Hard,
                RiskBucket::High,
            )
            .expect("exact key");
        assert_eq!(exact.sample_count, 1);
        assert_eq!(exact.successes_first_pass, 1);
        // 2. Missing exact key but a class-level record exists: the task
        //    fold answers (both risk buckets), never the wider phase fold.
        let task = store
            .lookup_stats(
                "p",
                "m",
                RouterPhase::Implement,
                TaskClass::Medium,
                RiskBucket::Medium,
            )
            .expect("task fold");
        assert_eq!(task.sample_count, 2);
        assert_eq!(task.failures_first_pass, 1);
        assert_eq!(task.rework_cost_micro_sum, 100);
        // 3. Missing class-level record: the phase fold answers.
        let phase_only = store
            .lookup_stats(
                "p",
                "m",
                RouterPhase::Implement,
                TaskClass::Easy,
                RiskBucket::Low,
            )
            .expect("phase fold");
        assert_eq!(phase_only.sample_count, 3, "every Implement bucket folds");
        assert_eq!(phase_only.successes_first_pass, 2);
        // 4. No evidence for the (provider, model, phase): None — the
        //    caller falls back to its prior, never a fabricated row.
        assert!(store
            .lookup_stats(
                "p",
                "m",
                RouterPhase::Review,
                TaskClass::Medium,
                RiskBucket::Low
            )
            .is_none());
        assert!(store
            .lookup_stats(
                "other",
                "m",
                RouterPhase::Implement,
                TaskClass::Medium,
                RiskBucket::Low
            )
            .is_none());
    }

    #[test]
    fn memory_store_folds_class_and_risk_buckets_per_phase() {
        let store = MemoryOutcomeStore::new();
        let key = |class: TaskClass, bucket: RiskBucket| OutcomeKey {
            provider: "p".into(),
            model: "m".into(),
            phase: RouterPhase::Implement,
            task_class: class,
            risk_bucket: bucket,
        };
        let sample = |verified_success: bool, rework: u64| OutcomeSample {
            verified_success,
            rework_cost_micro: rework,
            rework_turns: 1,
        };
        // Medium/Low: 1 clean + 1 failure; Hard/High: 2 clean; a different
        // phase stays out of the Implement fold.
        store.append_sample(&key(TaskClass::Medium, RiskBucket::Low), sample(true, 0));
        store.append_sample(&key(TaskClass::Medium, RiskBucket::Low), sample(false, 100));
        store.append_sample(&key(TaskClass::Hard, RiskBucket::High), sample(true, 0));
        store.append_sample(&key(TaskClass::Hard, RiskBucket::High), sample(true, 0));
        let other = OutcomeKey {
            phase: RouterPhase::Review,
            ..key(TaskClass::Medium, RiskBucket::Low)
        };
        store.append_sample(&other, sample(false, 5));
        assert_eq!(store.len(), 3, "two Implement buckets + one Review key");
        let folded = store
            .phase_stats("p", "m", RouterPhase::Implement)
            .expect("Implement rows exist");
        assert_eq!(folded.successes_first_pass, 3);
        assert_eq!(folded.failures_first_pass, 1);
        assert_eq!(folded.rework_cost_micro_sum, 100);
        assert_eq!(folded.sample_count, 4);
        // Per-key read stays exact.
        let hard = store
            .stats(&key(TaskClass::Hard, RiskBucket::High))
            .unwrap();
        assert_eq!(hard.successes_first_pass, 2);
        assert_eq!(hard.sample_count, 2);
        // Empty store consults miss; appending to it is a documented no-op.
        assert!(EmptyOutcomeStore
            .stats(&key(TaskClass::Easy, RiskBucket::Low))
            .is_none());
        assert!(EmptyOutcomeStore
            .phase_stats("p", "m", RouterPhase::Implement)
            .is_none());
    }
}
