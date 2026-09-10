//! `TaskEfficiencyMetrics` — the audit 84-88 per-task KPI set, DERIVED FROM
//! DURABLE ROWS ONLY.
//!
//! Each field documents the exact durable read it sums. Nothing here reads
//! an in-memory component counter: re-running [`derive`](TaskEfficiencyMetrics::derive)
//! after a store reopen yields the identical value, which the KPI tests
//! assert.
//!
//! Two measurements are intentionally bounded and loud about their
//! incompleteness: the provider-call and cost-reservation reads page at
//! [`MAX_TASK_CALL_ROWS`] / [`MAX_TASK_RESERVATIONS`], and exceeding the
//! bound is a typed `Truncated` error, never a silently partial total.

use faktor_core::event::EventKind;
use faktor_core::id::{SessionId, TaskId};
use faktor_core::state::TaskState;
use faktor_session::ledger::{LEDGER_ENTRY_SCHEMA_V, MAX_LEDGER_PAGE};
use faktor_session::LedgerPayload;
use faktor_store::Store;

/// Hard bound on the provider-call rows one task KPI derivation may sum.
pub const MAX_TASK_CALL_ROWS: i64 = 100_000;
/// Hard bound on the cost-reservation rows one task KPI derivation may sum.
pub const MAX_TASK_RESERVATIONS: i64 = 100_000;
/// Hard bound on the journal events one task KPI derivation may scan.
pub const MAX_TASK_EVENTS: u64 = 100_000;

/// The payload key under which a durable journal event records one
/// evidence-pipeline observation. This is the ONLY producer boundary for the
/// three evidence KPIs, and it is currently written by the harness's
/// evidence adapter, not by the production runtime (see
/// [`crate::variant::wiring_honesty`]): a run without such an event reads
/// honest zeros, never estimated bytes.
///
/// Payload shape (event kind `ContextPrepared`, payload schema v1):
///
/// ```json
/// { "efficiency_evidence": {
///     "original_bytes": 4096,
///     "sent_tokens": 512,
///     "retrievals": 2
/// } }
/// ```
pub const EVIDENCE_OBSERVATION_KEY: &str = "efficiency_evidence";

/// Typed failure of a KPI derivation. Every variant is structural: a missing
/// durable row, a session/task identity mismatch, a bounded read that would
/// have to truncate, or a corrupt durable shape.
#[derive(Debug, thiserror::Error)]
pub enum EfficiencyError {
    #[error("store failure: {0}")]
    Store(#[from] faktor_store::StoreError),
    #[error("session {0} has no durable row")]
    SessionMissing(SessionId),
    #[error(
        "session {session} is scoped to task {session_task}, not the requested task {requested}"
    )]
    TaskSessionMismatch {
        session: SessionId,
        session_task: TaskId,
        requested: TaskId,
    },
    #[error("task {session}/{task} has no durable row")]
    TaskMissing { session: SessionId, task: TaskId },
    #[error("bounded read of {what} hit its {limit}-row limit: refusing a partial KPI total")]
    Truncated { what: &'static str, limit: i64 },
    #[error("malformed durable KPI row: {0}")]
    Malformed(String),
}

/// One task's audit 84-88 efficiency metrics, derived exclusively from
/// durable rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct TaskEfficiencyMetrics {
    /// Durable sum of `provider_call.tokens_in` over the task's usage rows
    /// (attempt rows linked by `reservation_id`/`attempt_op_id`, legacy rows
    /// linked by the shared logical op). The row's `tokens_in` is the
    /// canonical input fold the runtime persists (uncached + cache reads +
    /// cache writes).
    pub input_tokens_total: u64,
    /// Durable prefix-cache attribution: `sum(floor(prompt_tokens *
    /// prefix_stability))` over the task's v13 prefix-observation rows,
    /// capped at `input_tokens_total`. `prompt_tokens` is the measured
    /// cacheable-prefix size and `prefix_stability` is the router's per-turn
    /// cache-coverage fraction, so the product is the covered (cached) part
    /// of that prefix. Zero when no observation was recorded — the runtime
    /// persists no provider-reported cache split (a documented limitation),
    /// so this is the only durable attribution that exists.
    pub input_tokens_cached: u64,
    /// `input_tokens_total - input_tokens_cached` (saturating).
    pub input_tokens_noncached: u64,
    /// Durable sum of `provider_call.tokens_out` over the task's usage rows.
    pub output_tokens: u64,
    /// Durable settled spend of the task: per `cost_reservation` row with
    /// status `settled`, the folded amount
    /// (`settled_cost_micro`, falling back to the pre-v18
    /// `provider_reported_cost_micro` / `provider_cost_micro` /
    /// `provider_reported_micro` for migrated rows; an Unknown close folds
    /// nothing and contributes 0). Reserved/dispatched/uncertain rows have
    /// not been folded into the task's spend and contribute 0.
    pub model_cost_micro: u64,
    /// Durable evidence bytes captured, summed from
    /// `efficiency_evidence.original_bytes` observations in the session's
    /// journal events (see [`EVIDENCE_OBSERVATION_KEY`]).
    pub evidence_original_bytes: u64,
    /// Durable evidence tokens sent, summed from
    /// `efficiency_evidence.sent_tokens` observations.
    pub evidence_sent_tokens: u64,
    /// Durable evidence retrievals, summed from
    /// `efficiency_evidence.retrievals` observations.
    pub evidence_retrieval_count: u64,
    /// Physical/logical call count: provider-call rows that carry usage
    /// counters or a non-`completed` status. Prefix-observation rows (no
    /// counters, `completed`) describe a call already counted and are
    /// excluded, so the count never doubles.
    pub model_calls: u64,
    /// Repair turns: durable typed-ledger `failure_recorded` entries. Each
    /// entry is one recorded failed attempt that a subsequent turn repaired.
    pub repair_turns: u64,
    /// Failed edits: typed-ledger `edit_txn_progress` entries whose durable
    /// per-file outcome is `conflicted` (the commit-time CAS refused the
    /// write; the file was never clobbered).
    pub failed_edits: u64,
    /// Reverted edits: paths listed in typed-ledger `edit_txn_rolled_back`
    /// terminal entries (files CAS-restored by a roll-back edit
    /// transaction). `rollback_conflicts` are NOT counted: those files were
    /// never restored.
    pub reverted_edits: u64,
    /// Human interventions: journal `prompt_admitted` events beyond the
    /// first. The first admitted prompt starts the task; every subsequent
    /// one is a human steer during the task's lifetime.
    pub human_interventions: u64,
    /// Durability span of the task row (`updated_ms - created_ms`, clamped at
    /// 0). Completion (and every task-row write) stamps `updated_ms`, so a
    /// completed task's wall is stable across reopens.
    pub wall_ms: u64,
    /// The task's durable completion proof: the task row is
    /// `VerifiedComplete`, a state only
    /// `Store::task_complete_verified` can produce, and only against a
    /// `Passed` verification record covering every criterion.
    pub verified: bool,
}

/// The strict decoded shape of one durable evidence observation payload.
#[derive(Debug, serde::Deserialize)]
struct EvidenceObservation {
    original_bytes: u64,
    sent_tokens: u64,
    retrievals: u64,
}

fn decode_edit_entry(
    entry_type: &str,
    schema_ver: i64,
    seq: i64,
    payload: &serde_json::Value,
) -> Result<LedgerPayload, EfficiencyError> {
    if schema_ver != LEDGER_ENTRY_SCHEMA_V {
        return Err(EfficiencyError::Malformed(format!(
            "ledger entry seq {seq} ({entry_type}) has schema_ver {schema_ver}, \
             this reader understands v{LEDGER_ENTRY_SCHEMA_V}"
        )));
    }
    let decoded: LedgerPayload = serde_json::from_value(payload.clone()).map_err(|e| {
        EfficiencyError::Malformed(format!(
            "ledger entry seq {seq} ({entry_type}) payload violates its v1 schema: {e}"
        ))
    })?;
    let tag = match &decoded {
        LedgerPayload::FailureRecorded { .. } => "failure_recorded",
        LedgerPayload::EditTxnProgress { .. } => "edit_txn_progress",
        LedgerPayload::EditTxnRolledBack { .. } => "edit_txn_rolled_back",
        _ => entry_type,
    };
    if tag != entry_type {
        return Err(EfficiencyError::Malformed(format!(
            "ledger entry seq {seq} column says {entry_type:?} but its payload \
             decodes as {tag:?}"
        )));
    }
    Ok(decoded)
}

impl TaskEfficiencyMetrics {
    /// Derive one task's KPIs from durable rows.
    ///
    /// The session must be durably scoped to `task_id`
    /// (`session.task_id == task_id`): task-less durable rows (journal
    /// events, edit-ledger entries, turn records) are attributed to the task
    /// through that durable identity, and a multi-task session would make
    /// that attribution ambiguous. The read refuses rather than guess.
    pub fn derive(
        store: &Store,
        session_id: SessionId,
        task_id: TaskId,
    ) -> Result<Self, EfficiencyError> {
        let session = store
            .get_session(session_id)?
            .ok_or(EfficiencyError::SessionMissing(session_id))?;
        if session.task_id != task_id {
            return Err(EfficiencyError::TaskSessionMismatch {
                session: session_id,
                session_task: session.task_id,
                requested: task_id,
            });
        }
        let task = store
            .get_task(session_id, task_id)?
            .ok_or(EfficiencyError::TaskMissing {
                session: session_id,
                task: task_id,
            })?;
        let mut metrics = Self::default();

        // ---- provider_call rows (usage + prefix observations) -------------
        let (calls, truncated) =
            store.provider_call_task_rows(session_id, task_id, MAX_TASK_CALL_ROWS)?;
        if truncated {
            return Err(EfficiencyError::Truncated {
                what: "provider_call rows",
                limit: MAX_TASK_CALL_ROWS,
            });
        }
        let mut cached_attributed: u64 = 0;
        for row in &calls {
            let usage_row =
                row.tokens_in.is_some() || row.tokens_out.is_some() || row.status != "completed";
            if usage_row {
                metrics.model_calls = metrics.model_calls.saturating_add(1);
                metrics.input_tokens_total = metrics
                    .input_tokens_total
                    .saturating_add(row.tokens_in.unwrap_or(0));
                metrics.output_tokens = metrics
                    .output_tokens
                    .saturating_add(row.tokens_out.unwrap_or(0));
            }
            if let (Some(prompt_tokens), Some(stability)) =
                (row.prompt_tokens, row.prefix_stability)
            {
                // The store read already validated stability into [0, 1] and
                // counters as non-negative; floor() makes the attribution
                // exact and deterministic (never rounds a cache hit up).
                let attributed = (prompt_tokens as f64 * stability).floor();
                if attributed.is_finite() && attributed > 0.0 {
                    cached_attributed =
                        cached_attributed.saturating_add(attributed.min(u64::MAX as f64) as u64);
                }
            }
        }
        metrics.input_tokens_cached = cached_attributed.min(metrics.input_tokens_total);
        metrics.input_tokens_noncached = metrics
            .input_tokens_total
            .saturating_sub(metrics.input_tokens_cached);

        // ---- cost_reservation settled folds -------------------------------
        let reservations =
            store.cost_reservations_of(session_id, task_id, MAX_TASK_RESERVATIONS)?;
        if reservations.len() as i64 >= MAX_TASK_RESERVATIONS {
            return Err(EfficiencyError::Truncated {
                what: "cost_reservation rows",
                limit: MAX_TASK_RESERVATIONS,
            });
        }
        for row in &reservations {
            if row.status != "settled" {
                continue;
            }
            let folded = row
                .settled_cost_micro
                .or(row.provider_reported_cost_micro)
                .or(row.provider_cost_micro)
                .or(row.provider_reported_micro)
                .unwrap_or(0);
            metrics.model_cost_micro = metrics.model_cost_micro.saturating_add(folded);
        }

        // ---- journal events: evidence observations + human steering -------
        let events = store.events_versioned_range(session_id, 1, Some(MAX_TASK_EVENTS + 1))?;
        if events.len() as u64 > MAX_TASK_EVENTS {
            return Err(EfficiencyError::Truncated {
                what: "journal events",
                limit: MAX_TASK_EVENTS as i64,
            });
        }
        let mut prompts = 0u64;
        for (event, _payload_ver) in &events {
            if event.kind == EventKind::PromptAdmitted {
                prompts = prompts.saturating_add(1);
            }
            let Some(observation) = event
                .payload
                .as_ref()
                .and_then(|payload| payload.get(EVIDENCE_OBSERVATION_KEY))
            else {
                continue;
            };
            let observation: EvidenceObservation = serde_json::from_value(observation.clone())
                .map_err(|e| {
                    EfficiencyError::Malformed(format!(
                        "event seq {}: {EVIDENCE_OBSERVATION_KEY} payload violates its v1 \
                         shape: {e}",
                        event.seq.raw()
                    ))
                })?;
            metrics.evidence_original_bytes = metrics
                .evidence_original_bytes
                .saturating_add(observation.original_bytes);
            metrics.evidence_sent_tokens = metrics
                .evidence_sent_tokens
                .saturating_add(observation.sent_tokens);
            metrics.evidence_retrieval_count = metrics
                .evidence_retrieval_count
                .saturating_add(observation.retrievals);
        }
        metrics.human_interventions = prompts.saturating_sub(1);

        // ---- typed ledger: repairs, failed edits, reverted edits ----------
        let mut cursor: Option<i64> = None;
        loop {
            let page = store.ledger_entries(session_id, cursor, MAX_LEDGER_PAGE)?;
            let page_len = page.len() as u64;
            for row in &page {
                cursor = Some(row.seq);
                match row.entry_type.as_str() {
                    "failure_recorded" => {
                        decode_edit_entry(&row.entry_type, row.schema_ver, row.seq, &row.payload)?;
                        metrics.repair_turns = metrics.repair_turns.saturating_add(1);
                    }
                    "edit_txn_progress" => {
                        let decoded = decode_edit_entry(
                            &row.entry_type,
                            row.schema_ver,
                            row.seq,
                            &row.payload,
                        )?;
                        if let LedgerPayload::EditTxnProgress { outcome, .. } = decoded {
                            if outcome == "conflicted" {
                                metrics.failed_edits = metrics.failed_edits.saturating_add(1);
                            }
                        }
                    }
                    "edit_txn_rolled_back" => {
                        let decoded = decode_edit_entry(
                            &row.entry_type,
                            row.schema_ver,
                            row.seq,
                            &row.payload,
                        )?;
                        if let LedgerPayload::EditTxnRolledBack { rolled_back, .. } = decoded {
                            metrics.reverted_edits = metrics
                                .reverted_edits
                                .saturating_add(rolled_back.len() as u64);
                        }
                    }
                    _ => {}
                }
            }
            if page_len < MAX_LEDGER_PAGE {
                break;
            }
        }

        // ---- durable task lifetime + completion proof ---------------------
        metrics.wall_ms = task.updated_ms.saturating_sub(task.created_ms).max(0) as u64;
        metrics.verified = task.state == TaskState::VerifiedComplete;
        Ok(metrics)
    }
}
