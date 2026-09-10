//! Efficiency A/B harness and KPI derivation (audit 84-88).
//!
//! What this crate is
//! ------------------
//! Audit 84-88 define a per-task efficiency measurement derived FROM DURABLE
//! ROWS (never from component counters) and paired A/B runs over an
//! [`EfficiencyVariant`] flag profile whose gains may only be claimed AFTER
//! verified-success parity is proven. This crate implements both:
//!
//! - [`metrics::TaskEfficiencyMetrics::derive`] sums only durable rows:
//!   `provider_call` (usage + v13 prefix observations),
//!   `cost_reservation` (settled folds), `verification_record` + task state
//!   (verified), typed-ledger edit entries (`failed_edits`, `reverted_edits`),
//!   `failure_recorded` entries (`repair_turns`), `prompt_admitted` events
//!   (`human_interventions`), and durable task-row lifetime (`wall_ms`).
//!   Each field's source is documented on the field.
//! - [`runner::run_paired`] executes a scripted corpus through the REAL
//!   `faktor-store` schema (all rows are written through the production
//!   store APIs) for a baseline arm and a candidate arm on identical
//!   starting revisions and seeds, asserts verified parity first, and then
//!   reports per-KPI deltas over the paired verified set.
//! - [`work::verified_work_score`] computes the weighted Verified Work per
//!   Token KPI. Work is weighted per TASK (`weight x verified`); criterion
//!   counts are recorded for coverage reporting but are deliberately NOT in
//!   the numerator, so splitting a criterion into several cannot inflate the
//!   score.
//!
//! Honest wiring (read this before trusting a gain)
//! ------------------------------------------------
//! The production agent runtime has no feature switch that consults an
//! `EfficiencyVariant` at run time. What IS real here is every *component*
//! each flag names: the harness drives the actual production functions and
//! measures their output through durable rows. [`variant::wiring_honesty`]
//! returns the explicit per-flag status, and every arm outcome carries it.
//! The A/B statement this crate makes is therefore: "with the flag's real
//! component in the harness pipeline, these durable KPIs changed, and
//! verified success did not regress" — not "the production daemon changes
//! behavior when the flag is set".

pub mod harness;
pub mod metrics;
pub mod runner;
pub mod variant;
pub mod work;

pub use harness::{run_scripted_task, scripted_corpus, ScriptedTask, TaskPlan, TaskRun};
pub use metrics::{
    EfficiencyError, TaskEfficiencyMetrics, EVIDENCE_OBSERVATION_KEY, MAX_TASK_CALL_ROWS,
    MAX_TASK_EVENTS, MAX_TASK_RESERVATIONS,
};
pub use runner::{run_arm, run_paired, ArmOutcome, KpiDelta, PairedReport, TaskArmResult};
pub use variant::{
    wiring_honesty, EfficiencyFlag, EfficiencyVariant, VariantHonesty, WiringStatus, FLAGS,
};
pub use work::{verified_work_score, TaskWorkRow, VerifiedWorkScore};

/// Deterministic corpus/provider seed used by the paired runner tests. Both
/// arms of a pair MUST use the same seed: that is part of the pairing
/// contract.
pub const PAIR_SEED: u64 = 42;
