//! The scripted paired runner: a small deterministic corpus executed through
//! the REAL production store schema, with the REAL component each variant
//! flag names driven in-process.
//!
//! Everything a run measures is written through production store APIs —
//! `create_session` / `adopt_session_identity` / `upsert_task` /
//! `cost_reserve*` / `cost_mark_dispatched` / `cost_settle` /
//! `record_provider_call*` / `start_turn_record` / `append_event_v` /
//! `append_ledger_entry` / `verification_record_put` /
//! `task_complete_verified` — and then read back by
//! [`TaskEfficiencyMetrics::derive`](crate::metrics::TaskEfficiencyMetrics::derive).
//! No component counter is ever read.
//!
//! Determinism/pairing contract:
//! - every task carries a fixed `base_revision`; both arms record the same
//!   one and the runner refuses a pair whose revisions differ;
//! - both arms share the provider seed; the only seed-derived quantity is a
//!   small per-task jitter applied identically to both arms;
//! - the executor never sleeps, spawns, or contacts a network.

use std::sync::{Arc, LazyLock};

use faktor_context::information::{
    select_by_information, InformationBudget, InformationSelection, Need,
};
use faktor_context::selection::{
    CandidateKind, CandidateRequirement, ContextCandidate, EvidenceLevel, NeedCoverage,
};
use faktor_context::{DurableTaskRows, Estimator, ProjectedDecision, TaskContextProjection};
use faktor_core::event::EventKind;
use faktor_core::id::{OpId, SessionId, TaskId, TaskRevision, WorkspaceId, WorktreeId};
use faktor_core::model::{RiskBucket, RouterPhase, TaskClass};
use faktor_core::op::ModelCallAttempt;
use faktor_core::state::{AgentState, CriterionVerification, TaskState, VerificationStatus};
use faktor_evidence::compress::compress;
use faktor_evidence::types::{BackingCompleteness, EvidenceKind};
use faktor_memory::SessionMemory;
use faktor_router::outcomes::{work_cost_estimate, VerifiedOutcomeStats};
use faktor_router::stability::{turn_stabilities, TurnPrefix};
use faktor_store::{ModelOutcomeSample, Store, TaskRow, VerificationRecordRow};
use serde_json::json;

use crate::metrics::EfficiencyError;
use crate::variant::EfficiencyVariant;

/// The stub provider name every harness row records.
pub const STUB_PROVIDER: &str = "efficiency-stub";
/// The cheap scripted model the baseline routes to.
pub const WEAK_MODEL: &str = "stub-weak";
/// The stronger scripted model rework-aware routing escalates repair calls to.
pub const STRONG_MODEL: &str = "stub-strong";
/// Scripted microUSD per input token of the weak model.
pub const WEAK_INPUT_MICRO: u64 = 1;
/// Scripted microUSD per output token of the weak model.
pub const WEAK_OUTPUT_MICRO: u64 = 3;
/// Scripted microUSD per input token of the strong model.
pub const STRONG_INPUT_MICRO: u64 = 4;
/// Scripted microUSD per output token of the strong model.
pub const STRONG_OUTPUT_MICRO: u64 = 12;
/// The unmeasured-rework fallback the router's work-cost estimate uses.
pub const REWORK_FALLBACK_MICRO: u64 = 30_000;

const BASE_TS: i64 = 1_000_000_000;
const OP_BASE: u64 = 10_000_000;
const MEMORY_FAILURE_KIND: &str = "efficiency_failure";

// ---------------------------------------------------------------- corpus

/// One scripted corpus task. The fields are the deterministic inputs of both
/// arms; the variant flags transform them only through the real components
/// documented in [`crate::variant::wiring_honesty`].
#[derive(Debug, Clone, Copy)]
pub struct ScriptedTask {
    pub id: &'static str,
    /// Benchmark weight (the Verified Work per Token numerator unit).
    pub weight: u64,
    pub criteria: &'static [&'static str],
    /// Starting repository revision. Both arms MUST record the same value.
    pub base_revision: &'static str,
    /// Raw evidence payloads retrieved during the task.
    pub evidence: &'static [&'static str],
    /// Implementation calls the baseline makes (including the first).
    pub baseline_calls: u32,
    /// Repair rounds the baseline needs (one `failure_recorded` + one call
    /// each).
    pub baseline_repairs: u32,
    /// Commit-time edit conflicts the baseline records.
    pub baseline_failed_edits: u32,
    /// Files one roll-back edit transaction restores.
    pub rollback_paths: u32,
    /// Human steers during the task (journal `prompt_admitted` events beyond
    /// the first).
    pub human_interventions: u32,
    /// Fixed (history-independent) prompt tokens per call.
    pub fixed_input_tokens_per_call: u64,
    /// History tokens re-sent per call under the baseline.
    pub history_resend_tokens: u64,
    /// Cacheable-prefix tokens observed per call.
    pub prefix_tokens_per_call: u64,
    pub output_tokens_per_call: u64,
    /// Whether the task reaches its durable completion proof. `false` is the
    /// corpus's unverified control: its KPIs never enter a gain claim.
    pub verifies: bool,
}

static LOG_OVERFLOW: LazyLock<&'static str> =
    LazyLock::new(|| synthetic_log("ringbuf", "src/ringbuf.rs", 700));
static LOG_SUMRANGES: LazyLock<&'static str> =
    LazyLock::new(|| synthetic_log("sumranges", "sumranges.go", 500));
static LOG_LEAP: LazyLock<&'static str> =
    LazyLock::new(|| synthetic_log("leapyears", "LeapYears.java", 400));
static LOG_REVERSE: LazyLock<&'static str> =
    LazyLock::new(|| synthetic_log("reverse-words", "src/lib.rs", 350));
static LOG_FLAKY: LazyLock<&'static str> =
    LazyLock::new(|| synthetic_log("flaky", "src/flush.rs", 450));

/// A deterministic tool log: a long run of repetition (the evidence layer's
/// process-log policy collapses it) plus one failure line (always kept).
fn synthetic_log(task: &str, file: &str, ticks: usize) -> &'static str {
    let mut out = String::with_capacity(ticks * 48 + 256);
    for _ in 0..ticks {
        out.push_str(&format!(
            "info: {task} heartbeat tick, queue empty, elapsed 1ms\n"
        ));
    }
    out.push_str(&format!("warning: {task} retrying stale handle\n"));
    out.push_str(&format!(
        "error: {task} mismatch in {file}:88 (expected 0, found 1)\n"
    ));
    out.push_str(&format!("info: {task} retry scheduled\n"));
    Box::leak(out.into_boxed_str())
}

fn evidence(items: Vec<&'static str>) -> &'static [&'static str] {
    Box::leak(items.into_boxed_slice())
}

/// The corpus in a fixed order (stable reporting). Built once; the evidence
/// payloads are deterministic and leak intentionally for the process
/// lifetime (a test binary, bounded at construction).
pub fn scripted_corpus() -> &'static [ScriptedTask] {
    static CORPUS: LazyLock<Vec<ScriptedTask>> = LazyLock::new(|| {
        vec![
            ScriptedTask {
                id: "ringbuf",
                weight: 3,
                criteria: &["push/pop roundtrip", "wrap-around preserves order"],
                base_revision: "rev-ringbuf-1a2b3c",
                evidence: evidence(vec![*LOG_OVERFLOW]),
                baseline_calls: 3,
                baseline_repairs: 1,
                baseline_failed_edits: 1,
                rollback_paths: 2,
                human_interventions: 0,
                fixed_input_tokens_per_call: 220,
                history_resend_tokens: 900,
                prefix_tokens_per_call: 300,
                output_tokens_per_call: 160,
                verifies: true,
            },
            ScriptedTask {
                id: "sumranges",
                weight: 2,
                criteria: &[
                    "prefix sums correct",
                    "range query inclusive",
                    "empty input",
                ],
                base_revision: "rev-sumranges-4d5e6f",
                evidence: evidence(vec![*LOG_SUMRANGES]),
                baseline_calls: 2,
                baseline_repairs: 0,
                baseline_failed_edits: 0,
                rollback_paths: 0,
                human_interventions: 0,
                fixed_input_tokens_per_call: 170,
                history_resend_tokens: 700,
                prefix_tokens_per_call: 210,
                output_tokens_per_call: 120,
                verifies: true,
            },
            ScriptedTask {
                id: "leapyears",
                weight: 1,
                criteria: &["divisible by 4", "century rule"],
                base_revision: "rev-leap-7a8b9c",
                evidence: evidence(vec![*LOG_LEAP, *LOG_OVERFLOW]),
                baseline_calls: 4,
                baseline_repairs: 2,
                baseline_failed_edits: 1,
                rollback_paths: 1,
                human_interventions: 1,
                fixed_input_tokens_per_call: 190,
                history_resend_tokens: 1100,
                prefix_tokens_per_call: 250,
                output_tokens_per_call: 200,
                verifies: true,
            },
            ScriptedTask {
                id: "reverse-words",
                weight: 3,
                criteria: &["word order reversed", "whitespace preserved"],
                base_revision: "rev-reverse-0f1e2d",
                evidence: evidence(vec![*LOG_REVERSE]),
                baseline_calls: 2,
                baseline_repairs: 0,
                baseline_failed_edits: 0,
                rollback_paths: 0,
                human_interventions: 0,
                fixed_input_tokens_per_call: 160,
                history_resend_tokens: 600,
                prefix_tokens_per_call: 220,
                output_tokens_per_call: 130,
                verifies: true,
            },
            ScriptedTask {
                id: "flaky-control",
                weight: 2,
                criteria: &["flush settles", "no lost writes"],
                base_revision: "rev-flaky-998877",
                evidence: evidence(vec![*LOG_FLAKY]),
                baseline_calls: 3,
                baseline_repairs: 3,
                baseline_failed_edits: 2,
                rollback_paths: 1,
                human_interventions: 0,
                fixed_input_tokens_per_call: 180,
                history_resend_tokens: 800,
                prefix_tokens_per_call: 240,
                output_tokens_per_call: 140,
                verifies: false,
            },
        ]
    });
    CORPUS.as_slice()
}

// ---------------------------------------------------------------- plan

/// The concrete row plan one arm executed for one task (recorded for report
/// and test assertions; the KPIs themselves always come from the durable
/// rows, never from this struct).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskPlan {
    pub implementation_calls: u32,
    pub repair_calls: u32,
    pub total_calls: u32,
    pub failures_recorded: u32,
    pub failed_edits: u32,
    pub reverted_edits: u32,
    pub human_interventions: u32,
    pub evidence_original_bytes: u64,
    pub evidence_sent_tokens: u64,
    pub evidence_retrievals: u64,
    pub repair_model: &'static str,
    pub verified: bool,
}

/// One executed scripted task: identities + the executed plan.
#[derive(Debug, Clone)]
pub struct TaskRun {
    pub task_id: &'static str,
    pub base_revision: String,
    pub session: SessionId,
    pub task: TaskId,
    pub plan: TaskPlan,
}

// ---------------------------------------------------------------- helpers

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic small per-task jitter derived from the shared provider
/// seed. Both arms call this with the SAME seed, so pairing is preserved.
fn seed_jitter(seed: u64, task_id: &str) -> u64 {
    let mut h = seed;
    for byte in task_id.bytes() {
        h = splitmix64(h ^ u64::from(byte));
    }
    h % 7
}

fn raw_evidence_tokens(task: &ScriptedTask) -> u64 {
    task.evidence
        .iter()
        .map(|raw| Estimator.estimate_tokens(raw) as u64)
        .sum()
}

/// Evidence candidates for the real information-gain selection.
fn evidence_candidates(bodies: &[String]) -> Vec<ContextCandidate> {
    bodies
        .iter()
        .enumerate()
        .map(|(i, body)| {
            let (need, requirement) = if i == 0 {
                (
                    vec![NeedCoverage {
                        need_id: "task".to_string(),
                        coverage_ppm: 1_000_000,
                    }],
                    CandidateRequirement::Required,
                )
            } else {
                (
                    vec![NeedCoverage {
                        need_id: "context".to_string(),
                        coverage_ppm: 1_000_000,
                    }],
                    CandidateRequirement::Optional,
                )
            };
            ContextCandidate {
                id: format!("ev-{i}"),
                kind: CandidateKind::ToolNote,
                bytes: body.len(),
                estimate_tokens: Estimator.estimate_tokens(body).max(1) as u32,
                utility: 1.0,
                evidence: Some(i as u64),
                requirement,
                confidence_ppm: 1_000_000,
                freshness_ppm: 1_000_000,
                need_coverage: need,
                expected_error_reduction_ppm: 0,
                level: EvidenceLevel::Exact,
                omission_keys: Vec::new(),
            }
        })
        .collect()
}

/// Run the REAL information-gain selection under a 3/4 budget, with the
/// required candidate always affordable (a required need that cannot fit is
/// a typed `Oversized` refusal, and the harness propagates it loudly).
fn select_evidence(
    bodies: &[String],
    budget_tokens: u64,
) -> Result<InformationSelection, EfficiencyError> {
    if bodies.is_empty() {
        return Ok(InformationSelection {
            selected: Vec::new(),
            selected_tokens: 0,
            required_tokens: 0,
        });
    }
    let candidates = evidence_candidates(bodies);
    let required_tokens = candidates
        .iter()
        .filter(|c| c.requirement == CandidateRequirement::Required)
        .map(|c| c.estimate_tokens)
        .sum::<u32>();
    let scaled = u32::try_from(budget_tokens.saturating_mul(3) / 4).unwrap_or(u32::MAX);
    let budget = InformationBudget {
        token_budget: scaled.max(required_tokens),
        needs: vec![
            Need {
                id: "task".to_string(),
                weight: 1.0,
                required: true,
            },
            Need {
                id: "context".to_string(),
                weight: 0.2,
                required: false,
            },
        ],
    };
    select_by_information(&candidates, &budget)
        .map_err(|e| EfficiencyError::Malformed(format!("information selection refused: {e}")))
}

/// The durable task rows the real typed-handoff projection is built from.
fn durable_rows_for(task: &ScriptedTask) -> DurableTaskRows {
    DurableTaskRows {
        task_state: "running".to_string(),
        goal: format!("complete {}", task.id),
        criteria: task.criteria.iter().map(|c| (*c).to_string()).collect(),
        plan_steps: (0..6)
            .map(|i| format!("{}: step {i} keeps the change minimal", task.id))
            .collect(),
        decisions: vec![ProjectedDecision {
            step: "approach".to_string(),
            choice: "scripted deterministic edit".to_string(),
            rationale: "efficiency harness".to_string(),
        }],
        checks: Vec::new(),
        children: Vec::new(),
        known_failures: Vec::new(),
        changed_files: vec!["src/lib.rs".to_string()],
    }
}

fn verified_stats(row: &faktor_store::ModelOutcomeStatsRow) -> VerifiedOutcomeStats {
    VerifiedOutcomeStats {
        successes_first_pass: row.successes_first_pass,
        failures_first_pass: row.failures_first_pass,
        rework_cost_micro_sum: row.rework_cost_micro_sum,
        rework_turns_sum: row.rework_turns_sum,
        sample_count: row.sample_count,
    }
}

/// Seed the durable verified-outcome history BOTH arms receive (identical
/// setup); only the rework-aware flag consults it. The weak model carries
/// three failures with heavy rework spend.
pub fn seed_router_history(store: &Store) -> Result<(), EfficiencyError> {
    for _ in 0..3 {
        store.model_outcome_stats_append(
            STUB_PROVIDER,
            WEAK_MODEL,
            RouterPhase::Implement,
            TaskClass::Medium,
            RiskBucket::High,
            ModelOutcomeSample {
                verified_success: false,
                rework_cost_micro: 30_000,
                rework_turns: 1,
            },
        )?;
    }
    Ok(())
}

fn scripted_cost(model: &str, input: u64, output: u64) -> u64 {
    let (input_price, output_price) = if model == STRONG_MODEL {
        (STRONG_INPUT_MICRO, STRONG_OUTPUT_MICRO)
    } else {
        (WEAK_INPUT_MICRO, WEAK_OUTPUT_MICRO)
    };
    input
        .saturating_mul(input_price)
        .saturating_add(output.saturating_mul(output_price))
}

fn set_task_state(
    store: &Store,
    session: SessionId,
    task_id: TaskId,
    state: TaskState,
    updated_ms: i64,
) -> Result<(), EfficiencyError> {
    let mut row = store
        .get_task(session, task_id)?
        .ok_or(EfficiencyError::TaskMissing {
            session,
            task: task_id,
        })?;
    row.state = state;
    row.updated_ms = updated_ms;
    store.upsert_task(&row)?;
    Ok(())
}

// ---------------------------------------------------------------- executor

/// Execute one scripted task for one arm: compute the variant effects through
/// the real components, write every durable row through production store
/// APIs, and return the executed plan + identities. The caller derives the
/// KPIs from the store.
pub fn run_scripted_task(
    store: &Arc<Store>,
    workspace: WorkspaceId,
    index: usize,
    task: &ScriptedTask,
    variant: EfficiencyVariant,
    seed: u64,
) -> Result<TaskRun, EfficiencyError> {
    let session = store.create_session(workspace, task.id, STUB_PROVIDER, WEAK_MODEL)?;
    let task_row_id = TaskId::new(index as u64 + 1);
    let worktree = WorktreeId::new(index as u64 + 1);
    store.adopt_session_identity(session.id, worktree, task_row_id)?;

    let jitter = seed_jitter(seed, task.id);
    let output_tokens = task.output_tokens_per_call + jitter;
    let prefix_tokens = u32::try_from(task.prefix_tokens_per_call + jitter).unwrap_or(u32::MAX);

    // ---- evidence: real CCR compression, then real information selection ---
    let mut bodies: Vec<String> = task.evidence.iter().map(|raw| (*raw).to_string()).collect();
    let evidence_original_bytes: u64 = task
        .evidence
        .iter()
        .map(|raw| raw.len() as u64)
        .fold(0u64, u64::saturating_add);
    let mut evidence_sent_tokens = raw_evidence_tokens(task);
    let mut evidence_retrievals = task.evidence.len() as u64;
    if variant.ccr {
        let mut compact_tokens = 0u64;
        let mut compacted = Vec::with_capacity(bodies.len());
        for raw in task.evidence {
            let (compact, _record) = compress(
                &EvidenceKind::ProcessLog,
                BackingCompleteness::Complete,
                raw,
            )
            .map_err(|e| EfficiencyError::Malformed(format!("ccr compression: {e}")))?;
            compact_tokens =
                compact_tokens.saturating_add(Estimator.estimate_tokens(&compact.body) as u64);
            compacted.push(compact.body);
        }
        evidence_sent_tokens = evidence_sent_tokens.min(compact_tokens);
        bodies = compacted;
    }
    if variant.semantic_context {
        let selection = select_evidence(&bodies, evidence_sent_tokens)?;
        evidence_sent_tokens = evidence_sent_tokens.min(u64::from(selection.selected_tokens));
        evidence_retrievals = selection.selected.len() as u64;
    }

    // ---- typed handoff: real projection render replaces re-sent history ----
    let history_tokens = if variant.typed_handoff {
        TaskContextProjection::from_durable_rows(durable_rows_for(task)).token_estimate() as u64
    } else {
        task.history_resend_tokens
    };
    let input_tokens_per_call = task
        .fixed_input_tokens_per_call
        .saturating_add(history_tokens);

    // ---- durable task row --------------------------------------------------
    let created_ms = BASE_TS + index as i64 * 100_000 + (seed % 1_000) as i64;
    let acceptance_criteria: Vec<String> = task.criteria.iter().map(|c| (*c).to_string()).collect();
    store.upsert_task(&TaskRow {
        task_id: task_row_id,
        session_id: session.id,
        goal: format!("efficiency task {}", task.id),
        acceptance_criteria: acceptance_criteria.clone(),
        plan: Vec::new(),
        attachments: Vec::new(),
        max_tokens: None,
        max_turns: None,
        spent_tokens: 0,
        spent_turns: 0,
        state: TaskState::Running,
        revision: TaskRevision::new(1),
        created_ms,
        updated_ms: created_ms,
    })?;

    // ---- journal: prompt admissions + the evidence observation -------------
    for prompt in 0..=task.human_interventions {
        store.append_event_v(
            session.id,
            None,
            EventKind::PromptAdmitted,
            AgentState::Idle,
            created_ms + i64::from(prompt),
            Some(json!({ "source": "efficiency-harness" })),
            1,
        )?;
    }
    store.append_event_v(
        session.id,
        None,
        EventKind::ContextPrepared,
        AgentState::BuildingContext,
        created_ms + 100,
        Some(json!({
            crate::metrics::EVIDENCE_OBSERVATION_KEY: {
                "original_bytes": evidence_original_bytes,
                "sent_tokens": evidence_sent_tokens,
                "retrievals": evidence_retrievals,
            }
        })),
        1,
    )?;

    // ---- failure learning: real SessionMemory read/write --------------------
    let memory = SessionMemory::new(Arc::clone(store), session.id);
    let failure_key = format!("{}:repeat", task.id);
    let mut repairs = 0u32;
    for round in 0..task.baseline_repairs {
        let known =
            variant.failure_learning && memory.latest(MEMORY_FAILURE_KIND, &failure_key)?.is_some();
        if known {
            // The durable fact says this failure already happened and its fix
            // is known: the repair round is skipped.
            break;
        }
        if variant.failure_learning {
            memory.remember(
                MEMORY_FAILURE_KIND,
                &failure_key,
                &format!("failure while repairing {} (round {round})", task.id),
            )?;
        }
        store.append_ledger_entry(
            session.id,
            "failure_recorded",
            1,
            json!({
                "kind": "failure_recorded",
                "failure": format!("{} repair round {round}", task.id)
            }),
        )?;
        repairs += 1;
    }

    // ---- edit transactions: one committed txn, per-conflict rollbacks -------
    let txn_base = (index as u64 + 1) * 100;
    store.append_ledger_entry(
        session.id,
        "edit_txn_prepared",
        1,
        json!({
            "kind": "edit_txn_prepared",
            "txn_id": txn_base,
            "session": task.id,
            "files": [{
                "path": "src/lib.rs",
                "base_digest": task.base_revision,
                "base_bytes_len": 128
            }],
            "strategy": "roll_forward"
        }),
    )?;
    store.append_ledger_entry(
        session.id,
        "edit_txn_committed",
        1,
        json!({
            "kind": "edit_txn_committed",
            "txn_id": txn_base,
            "committed": ["src/lib.rs"],
            "conflicted": [],
            "skipped": []
        }),
    )?;

    // ---- rework-aware routing: real work-cost math over durable stats -------
    let mut repair_model = WEAK_MODEL;
    let mut avoided_conflict = false;
    if variant.rework_aware_routing && task.baseline_failed_edits > 0 && task.baseline_repairs > 0 {
        let weak_row =
            store.model_outcome_stats_phase(STUB_PROVIDER, WEAK_MODEL, RouterPhase::Implement)?;
        let strong_row =
            store.model_outcome_stats_phase(STUB_PROVIDER, STRONG_MODEL, RouterPhase::Implement)?;
        // Repair calls happen after the first call, so they never carry the
        // one-time evidence payload: the estimate prices the per-call input.
        let repair_input = input_tokens_per_call;
        let weak_est = work_cost_estimate(
            scripted_cost(WEAK_MODEL, repair_input, output_tokens),
            weak_row.as_ref().map(verified_stats).as_ref(),
            REWORK_FALLBACK_MICRO,
        );
        let strong_est = work_cost_estimate(
            scripted_cost(STRONG_MODEL, repair_input, output_tokens),
            strong_row.as_ref().map(verified_stats).as_ref(),
            REWORK_FALLBACK_MICRO,
        );
        if strong_est.total_expected_micro < weak_est.total_expected_micro {
            repair_model = STRONG_MODEL;
            avoided_conflict = true;
        }
    }
    let failed_edits = task
        .baseline_failed_edits
        .saturating_sub(u32::from(avoided_conflict));
    let reverted_edits = failed_edits.saturating_mul(task.rollback_paths);
    for conflict in 0..failed_edits {
        let txn_id = txn_base + 1 + u64::from(conflict);
        store.append_ledger_entry(
            session.id,
            "edit_txn_progress",
            1,
            json!({
                "kind": "edit_txn_progress",
                "txn_id": txn_id,
                "seq": conflict,
                "path": "src/lib.rs",
                "outcome": "conflicted"
            }),
        )?;
        let restored: Vec<String> = (0..task.rollback_paths)
            .map(|p| format!("src/restored_{p}.rs"))
            .collect();
        store.append_ledger_entry(
            session.id,
            "edit_txn_rolled_back",
            1,
            json!({
                "kind": "edit_txn_rolled_back",
                "txn_id": txn_id,
                "rolled_back": restored,
                "rollback_conflicts": []
            }),
        )?;
    }

    // ---- calls: reservations, provider rows, turn records, prefix rows ------
    let total_calls = task.baseline_calls + repairs;
    let mut prefix_hashes: Vec<([u8; 32], u32)> = Vec::with_capacity(total_calls as usize);
    for call in 0..total_calls {
        let is_repair = call >= task.baseline_calls;
        let model = if is_repair { repair_model } else { WEAK_MODEL };
        let mut input = input_tokens_per_call;
        if call == 0 {
            input = input.saturating_add(evidence_sent_tokens);
        }
        let cost = scripted_cost(model, input, output_tokens);
        let created = created_ms + 1_000 + i64::from(call) * 10;
        let logical_op = OpId::new(OP_BASE + (index as u64) * 10_000 + u64::from(call) * 2 + 1);
        let attempt_op = OpId::new(logical_op.raw() + 1);
        store.start_turn_record(
            session.id,
            logical_op,
            None,
            None,
            STUB_PROVIDER,
            model,
            None,
        )?;
        if call % 3 == 0 {
            // Attempt-keyed path (v18): reservation by attempt identity.
            let attempt = ModelCallAttempt::new(logical_op, attempt_op, call)
                .ok_or_else(|| EfficiencyError::Malformed("attempt op ids collided".into()))?;
            let reservation =
                match store.cost_reserve_attempt(session.id, task_row_id, &attempt, cost, created, None)? {
                    faktor_store::CostReserveOutcome::Granted(id) => id,
                    faktor_store::CostReserveOutcome::Exceeded { free } => {
                        return Err(EfficiencyError::Malformed(format!(
                            "scripted reservation exceeded (free {free}); the corpus must stay within budget"
                        )))
                    }
                };
            store.cost_mark_dispatched(reservation, created + 1)?;
            store.record_provider_call_attempt(
                session.id,
                &attempt,
                Some(reservation),
                STUB_PROVIDER,
                model,
                "completed",
                Some(input),
                Some(output_tokens),
                None,
            )?;
            store.cost_settle(reservation, cost, Some(cost), None, None, created + 2)?;
        } else {
            // Legacy logical-op path: reservation + usage row share the op.
            let reservation = match store.cost_reserve(
                session.id,
                task_row_id,
                logical_op,
                cost,
                created,
            )? {
                faktor_store::CostReserveOutcome::Granted(id) => id,
                faktor_store::CostReserveOutcome::Exceeded { free } => {
                    return Err(EfficiencyError::Malformed(format!(
                        "scripted reservation exceeded (free {free}); the corpus must stay within budget"
                    )))
                }
            };
            store.cost_mark_dispatched(reservation, created + 1)?;
            store.record_provider_call(
                session.id,
                logical_op,
                STUB_PROVIDER,
                model,
                "completed",
                Some(input),
                Some(output_tokens),
                None,
            )?;
            store.cost_settle(reservation, cost, Some(cost), None, None, created + 2)?;
        }
        store.finish_turn_record(session.id, logical_op, "completed")?;

        let prefix_bytes = format!("{}:{}:cacheable-head", task.id, task.base_revision);
        prefix_hashes.push((
            *blake3::hash(prefix_bytes.as_bytes()).as_bytes(),
            prefix_tokens,
        ));
    }

    // Prefix observations: the per-turn stability is the REAL router pair
    // rule over the observed prefix digests/tokens.
    let turns: Vec<TurnPrefix> = prefix_hashes
        .iter()
        .enumerate()
        .map(|(i, (hash, tokens))| TurnPrefix::new(i as u64, *hash, *tokens))
        .collect();
    let stabilities = turn_stabilities(&turns);
    for (call, (hash, tokens)) in prefix_hashes.iter().enumerate() {
        let logical_op = OpId::new(OP_BASE + (index as u64) * 10_000 + call as u64 * 2 + 1);
        store.record_provider_call_with_prefix(
            session.id,
            logical_op,
            STUB_PROVIDER,
            WEAK_MODEL,
            "completed",
            None,
            None,
            None,
            Some(*hash),
            Some(u64::from(*tokens)),
            Some(stabilities[call]),
        )?;
    }

    // ---- verification: the durable completion proof (or its failure) --------
    let completed_ms = created_ms + 2_000 + i64::from(total_calls) * 10;
    set_task_state(
        store,
        session.id,
        task_row_id,
        TaskState::NeedsVerification,
        completed_ms - 20,
    )?;
    set_task_state(
        store,
        session.id,
        task_row_id,
        TaskState::Verifying,
        completed_ms - 10,
    )?;
    let record = VerificationRecordRow {
        id: faktor_core::id::VerificationRecordId::new(1),
        task_id: task_row_id,
        revision: TaskRevision::new(1),
        workspace_id: workspace,
        worktree_id: worktree,
        tree_hash: Some(task.base_revision.to_string()),
        criteria: acceptance_criteria
            .iter()
            .map(|criterion| CriterionVerification {
                criterion_key: criterion.clone(),
                passed: task.verifies,
                evidence: Some("efficiency harness deterministic verification".to_string()),
            })
            .collect(),
        checks: Vec::new(),
        changed_files: Vec::new(),
        unrelated_changes: Vec::new(),
        reviewer: None,
        status: VerificationStatus::Running,
        started_ms: completed_ms - 5,
        completed_ms: None,
    };
    let record_id = store.verification_record_put(&record)?;
    if task.verifies {
        if let Err(refusal) = store.verification_record_finalize(
            record_id,
            VerificationStatus::Passed,
            completed_ms,
        )? {
            return Err(EfficiencyError::Malformed(format!(
                "scripted record finalize refused: {refusal:?}"
            )));
        }
        let outcome = store.task_complete_verified(
            session.id,
            task_row_id,
            TaskRevision::new(1),
            record_id,
            completed_ms,
        )?;
        if let Err(refusal) = outcome {
            return Err(EfficiencyError::Malformed(format!(
                "scripted completion refused: {refusal:?}"
            )));
        }
    } else {
        if let Err(refusal) = store.verification_record_finalize(
            record_id,
            VerificationStatus::Failed,
            completed_ms,
        )? {
            return Err(EfficiencyError::Malformed(format!(
                "scripted record finalize refused: {refusal:?}"
            )));
        }
        set_task_state(
            store,
            session.id,
            task_row_id,
            TaskState::Failed,
            completed_ms,
        )?;
    }

    Ok(TaskRun {
        task_id: task.id,
        base_revision: task.base_revision.to_string(),
        session: session.id,
        task: task_row_id,
        plan: TaskPlan {
            implementation_calls: task.baseline_calls,
            repair_calls: repairs,
            total_calls,
            failures_recorded: repairs,
            failed_edits,
            reverted_edits,
            human_interventions: task.human_interventions,
            evidence_original_bytes,
            evidence_sent_tokens,
            evidence_retrievals,
            repair_model,
            verified: task.verifies,
        },
    })
}
