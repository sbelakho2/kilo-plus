//! The typed durable session ledger (audits 27 / 71-72).
//!
//! The opaque one-row-per-session ledger JSON blob (`task_ledger`) stays as
//! the runtime's working copy; the RICH ledger lives here: a typed,
//! versioned, append-only entry stream (`ledger_entry`) plus a per-session
//! materialized head checkpoint (`ledger_head`) that compaction rewrites.
//!
//! Never-lose contract
//! -------------------
//! Compaction deletes entries ONLY below a durability watermark computed
//! from the fully decoded entry stream, and the watermark NEVER evicts:
//! the LAST `GoalSet`/`CriteriaSet`/`Decision` entry and EVERY unresolved
//! `BlockerOpened` entry (the fold is open-blocks-state, so a resolved
//! opener is deletable, an unresolved one is not). Everything compaction
//! prunes is folded into the head checkpoint first: plan DAG, child-agent
//! records, the routing tail + count, epochs, the last verify run — so
//! compaction is projection-preserving, never FIFO-evicting of durable
//! meaning. The rule is enforced in code and locked by tests.
//!
//! Every payload row carries its own `schema_ver`; decoding is strict:
//! an unknown version or a shape violation is a loud error (corrupt), never
//! a silent parse. A session whose entry stream fails decode FAILS TO OPEN
//! loudly. A head checkpoint that is missing or undecodable is rebuilt by
//! replaying the surviving entries; entries are the authority.
//!
//! Sequence allocation never goes backwards (new seqs stay above the head's
//! checkpoint_seq even after compaction prunes), so "fold entries after the
//! checkpoint" is a correct crash-recovery cursor in both crash orders:
//! appended-then-not-checkpointed and checkpointed-then-crashed.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::handle::SessionHandle;
use crate::{json_bytes, map_store_err, SessionError, MAX_LEDGER_BYTES};

// ---------------------------------------------------------------- bounds

/// Payload schema version of every `ledger_entry` row this crate writes.
pub const LEDGER_ENTRY_SCHEMA_V: i64 = 1;
/// Schema version of the `ledger_head` checkpoint JSON.
pub const LEDGER_HEAD_SCHEMA_V: i64 = 1;
/// Max concurrently OPEN blockers a session may hold (bounded everything);
/// opening beyond the cap is a loud error, never a silent drop.
pub const MAX_LEDGER_OPEN_BLOCKERS: usize = 256;
/// Plan-mirror bound: mirrors the durable task row's plan cap
/// (`MAX_TASK_PLAN_STEPS`), so the mirror never exceeds the real plan.
pub const MAX_LEDGER_PLAN_STEPS: u32 = 256;
/// Routing-history tail retained in the materialized head across
/// compactions (the count is exact; the tail is the newest entries).
pub const MAX_LEDGER_ROUTING_TAIL: usize = 128;
/// Bounded page size for typed ledger reads (paging is fundamental).
pub const MAX_LEDGER_PAGE: u64 = 500;
/// Hard bound on ONE entry payload (serialized bytes).
pub const MAX_LEDGER_ENTRY_BYTES: usize = 16 * 1024;
/// Hard bound on one text field inside an entry payload.
pub const MAX_LEDGER_TEXT: usize = 4096;
/// Hard bound on the head checkpoint JSON (the legacy blob's bound).
pub const MAX_LEDGER_HEAD_BYTES: usize = MAX_LEDGER_BYTES;
/// Max criteria rows in one CriteriaSet (the task row's criterion cap).
pub const MAX_LEDGER_CRITERIA: usize = 64;
/// Max child-agent records tracked at once (bounded everything).
pub const MAX_LEDGER_CHILDREN: usize = 256;

// entry_type tags (the explicit schema tag column of every row)
pub const ENTRY_GOAL_SET: &str = "goal_set";
pub const ENTRY_CRITERIA_SET: &str = "criteria_set";
pub const ENTRY_BLOCKER_OPENED: &str = "blocker_opened";
pub const ENTRY_BLOCKER_RESOLVED: &str = "blocker_resolved";
pub const ENTRY_DECISION: &str = "decision";
pub const ENTRY_PLAN_STEP_ADDED: &str = "plan_step_added";
pub const ENTRY_CHILD_AGENT_STARTED: &str = "child_agent_started";
pub const ENTRY_CHILD_AGENT_FINISHED: &str = "child_agent_finished";
pub const ENTRY_ROUTING_DECISION: &str = "routing_decision";
pub const ENTRY_EPOCH_BUMPED: &str = "epoch_bumped";
pub const ENTRY_FAILURE_RECORDED: &str = "failure_recorded";
pub const ENTRY_VERIFY_RUN: &str = "verify_run";
pub const ENTRY_TURN_COMPLETED: &str = "turn_completed";

// ---------------------------------------------------------------- typed payloads

/// One `(check id, passed)` row of a VerifyRun.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerCheckRun {
    pub id: String,
    pub passed: bool,
}

/// The typed payload of ONE ledger entry. The serde `kind` field is the
/// per-payload schema tag inside the row JSON (schema v1); the row's
/// `entry_type` column is the same tag, stored redundantly so the stream
/// is decodable without touching the payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum LedgerPayload {
    /// The task goal (last wins). `{goal}`.
    GoalSet { goal: String },
    /// The wave-8 acceptance criteria as derived: full row list plus the
    /// canonical joined text (last wins). `{criteria, canonical}`.
    CriteriaSet {
        criteria: Vec<String>,
        canonical: String,
    },
    /// A verification/completion gate opened with this reason.
    /// `{reason}`. Unresolved openers are never evicted.
    BlockerOpened { reason: String },
    /// The blocker with this reason is cleared. `{reason}`.
    BlockerResolved { reason: String },
    /// One decision at a decision point. `{step, choice, rationale}`.
    Decision {
        step: String,
        choice: String,
        rationale: String,
    },
    /// One plan DAG node; `parent_index` links to the prior step it extends
    /// (None for the first step). Mirror of the typed task row's plan.
    PlanStepAdded {
        step_index: u32,
        text: String,
        parent_index: Option<u32>,
    },
    /// A child agent started. `{agent_id, task_id, worktree_id, purpose}`.
    ChildAgentStarted {
        agent_id: u64,
        task_id: u64,
        worktree_id: u64,
        purpose: String,
    },
    /// A child agent finished. `{agent_id, outcome}`. Append-time typed
    /// error when no matching running child exists.
    ChildAgentFinished { agent_id: u64, outcome: String },
    /// One economic routing decision of one turn. `{turn, provider, model,
    /// reasoning, cost_micro}`.
    RoutingDecision {
        turn: u64,
        provider: String,
        model: String,
        reasoning: String,
        cost_micro: u64,
    },
    /// Instruction epoch changed (rule files changed across a reload).
    /// `{from, to}`; `from` is None when no epoch was recorded yet.
    EpochBumped { from: Option<u64>, to: u64 },
    /// One recorded failure of a turn.
    FailureRecorded { failure: String },
    /// One genuine end-of-turn verification run: every check that ran with
    /// its pass/fail plus the outcome tag
    /// (`passed|failed|blocked|pending|unverified`).
    VerifyRun {
        checks: Vec<LedgerCheckRun>,
        outcome: String,
    },
    /// One genuine logical-turn completion. `{turn}` (op-based turn id).
    TurnCompleted { turn: u64 },
}

/// One decoded ledger row.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedLedgerEntry {
    pub seq: i64,
    pub entry_type: String,
    pub schema_ver: i64,
    pub payload: LedgerPayload,
    pub created_ms: i64,
}

/// The typed fold projection of the ledger: what compaction prunes is
/// folded HERE before it may delete entries.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct LedgerHead {
    /// Head JSON schema version ([`LEDGER_HEAD_SCHEMA_V`]).
    pub schema_ver: i64,
    pub goal: String,
    pub criteria: Vec<String>,
    pub canonical_criteria: String,
    pub open_blockers: Vec<String>,
    /// Decisions newest-first, bounded (the last Decision entry is also
    /// pinned in the stream, so the head's oldest entries age out only
    /// below the pinned newest).
    pub decisions: Vec<LedgerDecision>,
    /// Plan DAG: one row per step_index, ascending.
    pub plan_steps: Vec<LedgerPlanStep>,
    /// Child-agent records (running + finished), keyed by agent_id.
    pub children: Vec<LedgerChild>,
    /// Exact routing-decision count; `routing_tail` is the bounded newest
    /// tail that survives compaction.
    pub routing_count: u64,
    pub routing_tail: Vec<LedgerRouting>,
    /// The current instruction epoch (last EpochBumped target).
    pub epoch: Option<u64>,
    /// The last genuine VerifyRun (bounded summary).
    pub last_verify: Option<LedgerVerifySummary>,
    /// The materialized checkpoint: seq of the newest folded entry (0 when
    /// nothing is folded yet). Compaction rewrites it to the pre-prune max;
    /// appends always allocate ABOVE it, so the fold cursor never rewinds.
    pub checkpoint_seq: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct LedgerDecision {
    pub step: String,
    pub choice: String,
    pub rationale: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct LedgerPlanStep {
    pub step_index: u32,
    pub text: String,
    pub parent_index: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct LedgerChild {
    pub agent_id: u64,
    pub task_id: u64,
    pub worktree_id: u64,
    pub purpose: String,
    /// Some when the child finished; the entry stream records the moment.
    pub outcome: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct LedgerRouting {
    pub turn: u64,
    pub provider: String,
    pub model: String,
    pub reasoning: String,
    pub cost_micro: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct LedgerVerifySummary {
    pub checks: Vec<LedgerCheckRun>,
    pub outcome: String,
}

/// A typed, paged read of the entry stream (bounded).
#[derive(Debug, Clone, PartialEq)]
pub struct LedgerEntryPage {
    pub entries: Vec<TypedLedgerEntry>,
    pub has_more: bool,
}

/// Report of one watermark compaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LedgerCompactReport {
    pub entries_before: i64,
    pub deleted: i64,
    pub kept: i64,
    /// The pinned never-evict seqs (last GoalSet/CriteriaSet/Decision +
    /// every unresolved BlockerOpened).
    pub pinned: Vec<i64>,
    /// New head checkpoint seq (the pre-prune max).
    pub checkpoint_seq: i64,
}

// ---------------------------------------------------------------- decode/encode

fn entry_tag_of(payload: &LedgerPayload) -> &'static str {
    match payload {
        LedgerPayload::GoalSet { .. } => ENTRY_GOAL_SET,
        LedgerPayload::CriteriaSet { .. } => ENTRY_CRITERIA_SET,
        LedgerPayload::BlockerOpened { .. } => ENTRY_BLOCKER_OPENED,
        LedgerPayload::BlockerResolved { .. } => ENTRY_BLOCKER_RESOLVED,
        LedgerPayload::Decision { .. } => ENTRY_DECISION,
        LedgerPayload::PlanStepAdded { .. } => ENTRY_PLAN_STEP_ADDED,
        LedgerPayload::ChildAgentStarted { .. } => ENTRY_CHILD_AGENT_STARTED,
        LedgerPayload::ChildAgentFinished { .. } => ENTRY_CHILD_AGENT_FINISHED,
        LedgerPayload::RoutingDecision { .. } => ENTRY_ROUTING_DECISION,
        LedgerPayload::EpochBumped { .. } => ENTRY_EPOCH_BUMPED,
        LedgerPayload::FailureRecorded { .. } => ENTRY_FAILURE_RECORDED,
        LedgerPayload::VerifyRun { .. } => ENTRY_VERIFY_RUN,
        LedgerPayload::TurnCompleted { .. } => ENTRY_TURN_COMPLETED,
    }
}

/// Decode one typed payload from its row. Unknown `entry_type` or unknown
/// `schema_ver` => loud `Malformed` (corrupt), never a silent parse.
/// Schema-shape violations are equally loud.
fn decode_payload(
    entry_type: &str,
    schema_ver: i64,
    json: &serde_json::Value,
) -> Result<LedgerPayload, SessionError> {
    if schema_ver != LEDGER_ENTRY_SCHEMA_V {
        return Err(SessionError::Malformed(format!(
            "ledger entry {entry_type:?} has unknown schema version {schema_ver} \
             (this reader understands v{LEDGER_ENTRY_SCHEMA_V}); refusing to parse"
        )));
    }
    let decode = |tag: &str| -> Result<LedgerPayload, SessionError> {
        serde_json::from_value(json.clone()).map_err(|e| {
            SessionError::Malformed(format!(
                "ledger entry {tag} payload violates its v1 schema: {e}"
            ))
        })
    };
    match entry_type {
        ENTRY_GOAL_SET => decode(entry_type),
        ENTRY_CRITERIA_SET => decode(entry_type),
        ENTRY_BLOCKER_OPENED => decode(entry_type),
        ENTRY_BLOCKER_RESOLVED => decode(entry_type),
        ENTRY_DECISION => decode(entry_type),
        ENTRY_PLAN_STEP_ADDED => decode(entry_type),
        ENTRY_CHILD_AGENT_STARTED => decode(entry_type),
        ENTRY_CHILD_AGENT_FINISHED => decode(entry_type),
        ENTRY_ROUTING_DECISION => decode(entry_type),
        ENTRY_EPOCH_BUMPED => decode(entry_type),
        ENTRY_FAILURE_RECORDED => decode(entry_type),
        ENTRY_VERIFY_RUN => decode(entry_type),
        ENTRY_TURN_COMPLETED => decode(entry_type),
        other => Err(SessionError::Malformed(format!(
            "ledger entry type {other:?} is unknown to this reader"
        ))),
    }
}

fn check_text(value: &str, what: &str) -> Result<(), SessionError> {
    if value.is_empty() {
        return Err(SessionError::Malformed(format!(
            "ledger entry {what} must be non-empty"
        )));
    }
    if value.len() > MAX_LEDGER_TEXT {
        return Err(SessionError::Oversized(format!(
            "ledger entry {what} of {} bytes exceeds {MAX_LEDGER_TEXT}",
            value.len()
        )));
    }
    Ok(())
}

fn check_payload_bytes(payload: &LedgerPayload) -> Result<(), SessionError> {
    let bytes = json_bytes(&serde_json::to_value(payload).unwrap_or_default());
    if bytes > MAX_LEDGER_ENTRY_BYTES {
        return Err(SessionError::Oversized(format!(
            "ledger entry payload of {bytes} bytes exceeds MAX_LEDGER_ENTRY_BYTES"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------- head fold

fn fold(head: &mut LedgerHead, payload: &LedgerPayload) -> Result<(), SessionError> {
    match payload {
        LedgerPayload::GoalSet { goal } => {
            head.goal = goal.clone();
        }
        LedgerPayload::CriteriaSet {
            criteria,
            canonical,
        } => {
            head.criteria = criteria.clone();
            head.canonical_criteria = canonical.clone();
        }
        LedgerPayload::BlockerOpened { reason } => {
            if !head.open_blockers.iter().any(|r| r == reason) {
                head.open_blockers.push(reason.clone());
            }
        }
        LedgerPayload::BlockerResolved { reason } => {
            // Tolerant: resolving a blocker that is no longer open (its
            // opener was already pruned below the watermark) is a no-op,
            // never an error — the fold is a projection of what happened.
            head.open_blockers.retain(|r| r != reason);
        }
        LedgerPayload::Decision {
            step,
            choice,
            rationale,
        } => {
            let entry = LedgerDecision {
                step: step.clone(),
                choice: choice.clone(),
                rationale: rationale.clone(),
            };
            if head.decisions.first() != Some(&entry) {
                head.decisions.insert(0, entry);
                head.decisions.truncate(MAX_LEDGER_OPEN_BLOCKERS);
            }
        }
        LedgerPayload::PlanStepAdded {
            step_index,
            text,
            parent_index,
        } => {
            let row = LedgerPlanStep {
                step_index: *step_index,
                text: text.clone(),
                parent_index: *parent_index,
            };
            match head
                .plan_steps
                .iter_mut()
                .find(|p| p.step_index == *step_index)
            {
                Some(existing) => *existing = row,
                None => {
                    head.plan_steps.push(row);
                    head.plan_steps.sort_by_key(|p| p.step_index);
                }
            }
        }
        LedgerPayload::ChildAgentStarted {
            agent_id,
            task_id,
            worktree_id,
            purpose,
        } => {
            let row = LedgerChild {
                agent_id: *agent_id,
                task_id: *task_id,
                worktree_id: *worktree_id,
                purpose: purpose.clone(),
                outcome: None,
            };
            match head.children.iter_mut().find(|c| c.agent_id == *agent_id) {
                Some(existing) => *existing = row,
                None => head.children.push(row),
            }
        }
        LedgerPayload::ChildAgentFinished { agent_id, outcome } => {
            if let Some(child) = head
                .children
                .iter_mut()
                .find(|c| c.agent_id == *agent_id && c.outcome.is_none())
            {
                child.outcome = Some(outcome.clone());
            }
            // Tolerant when the running record was already pruned with its
            // start below the watermark (the fold is a projection).
        }
        LedgerPayload::RoutingDecision {
            turn,
            provider,
            model,
            reasoning,
            cost_micro,
        } => {
            head.routing_count = head.routing_count.saturating_add(1);
            head.routing_tail.push(LedgerRouting {
                turn: *turn,
                provider: provider.clone(),
                model: model.clone(),
                reasoning: reasoning.clone(),
                cost_micro: *cost_micro,
            });
            if head.routing_tail.len() > MAX_LEDGER_ROUTING_TAIL {
                head.routing_tail.remove(0);
            }
        }
        LedgerPayload::EpochBumped { from: _, to } => {
            head.epoch = Some(*to);
        }
        LedgerPayload::FailureRecorded { .. } => {}
        LedgerPayload::VerifyRun { checks, outcome } => {
            head.last_verify = Some(LedgerVerifySummary {
                checks: checks.clone(),
                outcome: outcome.clone(),
            });
        }
        LedgerPayload::TurnCompleted { .. } => {}
    }
    Ok(())
}

fn fold_entries(head: &mut LedgerHead, entries: &[TypedLedgerEntry]) -> Result<(), SessionError> {
    for entry in entries {
        fold(head, &entry.payload)?;
        head.checkpoint_seq = entry.seq;
    }
    Ok(())
}

fn head_to_json(head: &LedgerHead) -> Result<String, SessionError> {
    let json = serde_json::to_value(head)
        .map_err(|e| SessionError::Internal(format!("ledger head serialization: {e}")))?;
    let bytes = json_bytes(&json);
    if bytes > MAX_LEDGER_HEAD_BYTES {
        return Err(SessionError::Oversized(format!(
            "ledger head of {bytes} bytes exceeds MAX_LEDGER_HEAD_BYTES; \
             the ledger cannot be compacted to a larger head"
        )));
    }
    serde_json::to_string(&json)
        .map_err(|e| SessionError::Internal(format!("ledger head serialization: {e}")))
}

fn head_from_json(raw: &serde_json::Value) -> Result<LedgerHead, SessionError> {
    let head: LedgerHead = serde_json::from_value(raw.clone()).map_err(|e| {
        SessionError::Malformed(format!("ledger head JSON is corrupt (undecodable): {e}"))
    })?;
    if head.schema_ver != LEDGER_HEAD_SCHEMA_V {
        return Err(SessionError::Malformed(format!(
            "ledger head has unknown schema version {} (this reader understands v{LEDGER_HEAD_SCHEMA_V})",
            head.schema_ver
        )));
    }
    Ok(head)
}

// ---------------------------------------------------------------- handle API

impl SessionHandle {
    fn decode_row(
        &self,
        row: &faktor_store::LedgerEntryRow,
    ) -> Result<TypedLedgerEntry, SessionError> {
        let payload = decode_payload(&row.entry_type, row.schema_ver, &row.payload)?;
        Ok(TypedLedgerEntry {
            seq: row.seq,
            entry_type: row.entry_type.clone(),
            schema_ver: row.schema_ver,
            payload,
            created_ms: row.created_ms,
        })
    }

    /// Read the session's typed ledger from durable rows, ascending by seq
    /// (bounded page; the caller pages with the cursor seq). Every row is
    /// decoded strictly: an unknown version or a shape violation is a loud
    /// error, never a silent drop.
    pub fn ledger_entries_page(
        &self,
        after_seq: Option<i64>,
        limit: u64,
    ) -> faktor_core::Result<LedgerEntryPage> {
        let limit = limit.clamp(1, MAX_LEDGER_PAGE);
        let rows = self
            .manager
            .store()
            .ledger_entries(self.id, after_seq, limit + 1)
            .map_err(map_store_err)?;
        let has_more = rows.len() as u64 > limit;
        let mut entries = Vec::with_capacity(rows.len());
        for row in rows.into_iter().take(limit as usize) {
            entries.push(self.decode_row(&row)?);
        }
        Ok(LedgerEntryPage { entries, has_more })
    }

    fn all_entries_decoded(&self) -> Result<Vec<TypedLedgerEntry>, SessionError> {
        let store = self.manager.store();
        let mut out = Vec::new();
        let mut cursor: Option<i64> = None;
        loop {
            let page = store
                .ledger_entries(self.id, cursor, 1000)
                .map_err(map_store_err)?;
            if page.is_empty() {
                break;
            }
            cursor = page.last().map(|r| r.seq);
            for row in page {
                out.push(self.decode_row(&row)?);
            }
            if cursor.is_none() {
                break;
            }
        }
        Ok(out)
    }

    /// Fold every entry after the materialized head's checkpoint onto the
    /// head (crash recovery between an entry append and its head
    /// checkpoint) and persist the refreshed checkpoint. A missing or
    /// undecodable head is rebuilt by replaying the surviving entries
    /// (entries are the authority). An ENTRY that fails its schema decode
    /// is a loud error — the ledger never silently drops one.
    pub fn ledger_ensure_head(&self) -> faktor_core::Result<LedgerHead> {
        let store = self.manager.store();
        let max_seq = store.ledger_max_seq(self.id).map_err(map_store_err)?;
        // A ledger_head row whose JSON is undecodable is a hand-corrupted
        // head: recover by replay from entries (never fail the open).
        let head_row = store
            .ledger_head(self.id)
            .map_err(map_store_err)
            .unwrap_or_default();
        let mut head: LedgerHead = LedgerHead {
            schema_ver: LEDGER_HEAD_SCHEMA_V,
            ..Default::default()
        };
        let mut after: Option<i64> = None;
        if let Some(row) = &head_row {
            if row.schema_ver != LEDGER_HEAD_SCHEMA_V {
                // Unknown FUTURE head schema: silently rebuilding would
                // discard head-only content compaction folded; refuse loudly.
                return Err(SessionError::Malformed(format!(
                    "ledger head has unknown schema version {}; refusing to misread it \
                     (delete the head row to force a rebuild from entries)",
                    row.schema_ver
                ))
                .into());
            }
            match head_from_json(&row.head_json) {
                Ok(h) => {
                    head = h;
                    after = Some(head.checkpoint_seq);
                }
                // Corrupt head JSON: replay the surviving entries from
                // scratch (the rebuilt checkpoint may lose content that
                // compaction had pruned below the entries — the entries
                // are the authority on what survived).
                Err(SessionError::Malformed(_)) => after = None,
                Err(e) => return Err(e.into()),
            }
        }
        let stale = match after {
            Some(checkpoint) => checkpoint < max_seq,
            None => max_seq > 0 || head_row.is_some(),
        };
        if !stale {
            return Ok(head);
        }
        let rows = store
            .ledger_entries(self.id, after, u64::MAX)
            .map_err(map_store_err)?;
        let mut entries = Vec::with_capacity(rows.len());
        for row in rows {
            entries.push(self.decode_row(&row)?);
        }
        fold_entries(&mut head, &entries)?;
        let json = head_to_json(&head)?;
        let head_value = serde_json::from_str(&json)
            .map_err(|e| SessionError::Internal(format!("head json: {e}")))?;
        store
            .put_ledger_head(
                self.id,
                head_value,
                head.checkpoint_seq,
                LEDGER_HEAD_SCHEMA_V,
            )
            .map_err(map_store_err)?;
        Ok(head)
    }

    // ---------------------------------------------------- typed appenders

    /// Append `GoalSet {goal}` (bounded; a fresh goal replaces the old in
    /// the projection).
    pub fn ledger_goal_set(&self, goal: &str) -> faktor_core::Result<Option<i64>> {
        check_text(goal, "goal")?;
        self.append_entry(LedgerPayload::GoalSet {
            goal: goal.to_string(),
        })
    }

    /// Append `CriteriaSet` with the wave-8 canonical criteria: the derived
    /// row list and its canonical joined text.
    pub fn ledger_criteria_set(
        &self,
        criteria: &[String],
        canonical: &str,
    ) -> faktor_core::Result<Option<i64>> {
        if criteria.is_empty() {
            return Err(SessionError::Malformed(
                "criteria_set requires at least one criterion".into(),
            )
            .into());
        }
        if criteria.len() > MAX_LEDGER_CRITERIA {
            return Err(SessionError::Oversized(format!(
                "{} criteria exceed MAX_LEDGER_CRITERIA",
                criteria.len()
            ))
            .into());
        }
        for c in criteria {
            check_text(c, "criterion")?;
        }
        if canonical.is_empty() || canonical.len() > MAX_LEDGER_TEXT * 4 {
            return Err(SessionError::Malformed(
                "criteria canonical text must be 1..=16384 bytes".into(),
            )
            .into());
        }
        self.append_entry(LedgerPayload::CriteriaSet {
            criteria: criteria.to_vec(),
            canonical: canonical.to_string(),
        })
    }

    /// Append `BlockerOpened {reason}`. A reason that is already open is a
    /// no-op (Ok(None)) — the gate is open, not re-opened. Opening beyond
    /// [`MAX_LEDGER_OPEN_BLOCKERS`] is a loud error.
    pub fn ledger_blocker_opened(&self, reason: &str) -> faktor_core::Result<Option<i64>> {
        check_text(reason, "blocker reason")?;
        let head = self.ledger_ensure_head()?;
        if head.open_blockers.iter().any(|r| r == reason) {
            return Ok(None);
        }
        if head.open_blockers.len() >= MAX_LEDGER_OPEN_BLOCKERS {
            return Err(SessionError::Oversized(format!(
                "{} open blockers exceed MAX_LEDGER_OPEN_BLOCKERS; resolve blockers before opening more",
                head.open_blockers.len()
            ))
            .into());
        }
        self.append_entry(LedgerPayload::BlockerOpened {
            reason: reason.to_string(),
        })
    }

    /// Append `BlockerResolved {reason}`. A typed error when the reason is
    /// not open (nothing to resolve) — loud, never a silent no-op.
    pub fn ledger_blocker_resolved(&self, reason: &str) -> faktor_core::Result<Option<i64>> {
        check_text(reason, "blocker reason")?;
        let head = self.ledger_ensure_head()?;
        if !head.open_blockers.iter().any(|r| r == reason) {
            return Err(SessionError::Conflict(format!(
                "blocker {reason:?} is not open; nothing to resolve"
            ))
            .into());
        }
        self.append_entry(LedgerPayload::BlockerResolved {
            reason: reason.to_string(),
        })
    }

    /// Append one `Decision {step, choice, rationale}`.
    pub fn ledger_decision(
        &self,
        step: &str,
        choice: &str,
        rationale: &str,
    ) -> faktor_core::Result<Option<i64>> {
        check_text(step, "decision step")?;
        check_text(choice, "decision choice")?;
        check_text(rationale, "decision rationale")?;
        self.append_entry(LedgerPayload::Decision {
            step: step.to_string(),
            choice: choice.to_string(),
            rationale: rationale.to_string(),
        })
    }

    /// Mirror one plan step as `PlanStepAdded {step_index, text,
    /// parent_index}` (`parent_index` = the prior step it extends). The
    /// mirror is idempotent: re-recording an index replaces it.
    pub fn ledger_plan_step_added(
        &self,
        step_index: u32,
        text: &str,
        parent_index: Option<u32>,
    ) -> faktor_core::Result<Option<i64>> {
        check_text(text, "plan step")?;
        if step_index >= MAX_LEDGER_PLAN_STEPS {
            return Err(SessionError::Oversized(format!(
                "plan step index {step_index} exceeds MAX_LEDGER_PLAN_STEPS"
            ))
            .into());
        }
        if parent_index.is_some_and(|p| p >= step_index) {
            return Err(SessionError::Malformed(format!(
                "plan step {step_index} cannot have parent {parent_index:?} (parents precede their step)"
            ))
            .into());
        }
        self.append_entry(LedgerPayload::PlanStepAdded {
            step_index,
            text: text.to_string(),
            parent_index,
        })
    }

    /// Record a child agent start. A second start of the same RUNNING
    /// agent is a typed error; a finished agent may start again.
    pub fn ledger_child_started(
        &self,
        agent_id: u64,
        task_id: u64,
        worktree_id: u64,
        purpose: &str,
    ) -> faktor_core::Result<Option<i64>> {
        if agent_id == 0 || task_id == 0 || worktree_id == 0 {
            return Err(SessionError::Malformed("child agent ids must be non-zero".into()).into());
        }
        check_text(purpose, "child purpose")?;
        let head = self.ledger_ensure_head()?;
        if head
            .children
            .iter()
            .any(|c| c.agent_id == agent_id && c.outcome.is_none())
        {
            return Err(SessionError::Conflict(format!(
                "child agent {agent_id} is already running"
            ))
            .into());
        }
        if head.children.len() >= MAX_LEDGER_CHILDREN {
            return Err(SessionError::Oversized(format!(
                "{} child records exceed MAX_LEDGER_CHILDREN",
                head.children.len()
            ))
            .into());
        }
        self.append_entry(LedgerPayload::ChildAgentStarted {
            agent_id,
            task_id,
            worktree_id,
            purpose: purpose.to_string(),
        })
    }

    /// Record a child agent finish. A finish WITHOUT a running start is a
    /// typed error at append time (out-of-order stream is corruption).
    pub fn ledger_child_finished(
        &self,
        agent_id: u64,
        outcome: &str,
    ) -> faktor_core::Result<Option<i64>> {
        check_text(outcome, "child outcome")?;
        let head = self.ledger_ensure_head()?;
        if !head
            .children
            .iter()
            .any(|c| c.agent_id == agent_id && c.outcome.is_none())
        {
            return Err(SessionError::Malformed(format!(
                "child agent finish without a running start: agent {agent_id} has no open start"
            ))
            .into());
        }
        self.append_entry(LedgerPayload::ChildAgentFinished {
            agent_id,
            outcome: outcome.to_string(),
        })
    }

    /// Record one routing decision (`turn` = the op id of the routed
    /// logical turn), with the router's reasoning and estimated cost.
    pub fn ledger_routing_decision(
        &self,
        turn: u64,
        provider: &str,
        model: &str,
        reasoning: &str,
        cost_micro: u64,
    ) -> faktor_core::Result<Option<i64>> {
        if provider.is_empty() || provider.len() > 256 {
            return Err(SessionError::Malformed("provider must be 1..=256 bytes".into()).into());
        }
        if model.is_empty() || model.len() > 256 {
            return Err(SessionError::Malformed("model must be 1..=256 bytes".into()).into());
        }
        if reasoning.len() > MAX_LEDGER_TEXT {
            return Err(SessionError::Oversized(format!(
                "routing reasoning of {} bytes exceeds {MAX_LEDGER_TEXT}",
                reasoning.len()
            ))
            .into());
        }
        self.append_entry(LedgerPayload::RoutingDecision {
            turn,
            provider: provider.to_string(),
            model: model.to_string(),
            reasoning: reasoning.to_string(),
            cost_micro,
        })
    }

    /// Record an instruction-epoch bump `from -> to` (`from` = the epoch
    /// recorded in the head, None when nothing was recorded yet). A no-op
    /// when the head already records `to`.
    pub fn ledger_epoch_bumped(
        &self,
        from: Option<u64>,
        to: u64,
    ) -> faktor_core::Result<Option<i64>> {
        if from == Some(to) {
            return Err(SessionError::Malformed("epoch_bumped requires from != to".into()).into());
        }
        let head = self.ledger_ensure_head()?;
        if head.epoch == Some(to) {
            return Ok(None);
        }
        let from = from.or(head.epoch);
        self.append_entry(LedgerPayload::EpochBumped { from, to })
    }

    /// Record one turn failure (bounded; mirrors the ledger blob's
    /// known-failures list).
    pub fn ledger_failure_recorded(&self, failure: &str) -> faktor_core::Result<Option<i64>> {
        check_text(failure, "failure")?;
        self.append_entry(LedgerPayload::FailureRecorded {
            failure: failure.to_string(),
        })
    }

    /// Record one genuine end-of-turn verification run. `outcome` is one of
    /// `passed|failed|blocked|pending|unverified`; a typo is a typed error.
    pub fn ledger_verify_run(
        &self,
        checks: &[LedgerCheckRun],
        outcome: &str,
    ) -> faktor_core::Result<Option<i64>> {
        if checks.is_empty() {
            return Err(SessionError::Malformed(
                "verify_run requires at least one executed check".into(),
            )
            .into());
        }
        if checks.len() > 64 {
            return Err(SessionError::Oversized("too many checks in one verify_run".into()).into());
        }
        for c in checks {
            if c.id.is_empty() || c.id.len() > 128 {
                return Err(
                    SessionError::Malformed("check id must be 1..=128 bytes".into()).into(),
                );
            }
        }
        if !matches!(
            outcome,
            "passed" | "failed" | "blocked" | "pending" | "unverified"
        ) {
            return Err(SessionError::Malformed(format!(
                "verify_run outcome {outcome:?} is not one of passed|failed|blocked|pending|unverified"
            ))
            .into());
        }
        self.append_entry(LedgerPayload::VerifyRun {
            checks: checks.to_vec(),
            outcome: outcome.to_string(),
        })
    }

    /// Record one genuine logical-turn completion.
    pub fn ledger_turn_completed(&self, turn: u64) -> faktor_core::Result<Option<i64>> {
        if turn == 0 {
            return Err(SessionError::Malformed(
                "turn_completed requires a non-zero turn id".into(),
            )
            .into());
        }
        self.append_entry(LedgerPayload::TurnCompleted { turn })
    }

    /// The shared typed append tail: bounds the payload, maps its entry
    /// type, and writes the single row (gapless, always above the head's
    /// checkpoint so the fold cursor never rewinds).
    fn append_entry(&self, payload: LedgerPayload) -> faktor_core::Result<Option<i64>> {
        check_payload_bytes(&payload)?;
        let tag = entry_tag_of(&payload);
        let json = serde_json::to_value(&payload)
            .map_err(|e| SessionError::Internal(format!("ledger payload serialization: {e}")))?;
        let seq = self
            .manager
            .store()
            .append_ledger_entry(self.id, tag, LEDGER_ENTRY_SCHEMA_V, json)
            .map_err(map_store_err)?;
        Ok(Some(seq))
    }

    // ------------------------------------------------------------ compaction

    /// Watermark compaction of the typed ledger. Reads and strictly decodes
    /// the ENTIRE entry stream first (an undecodable row refuses the
    /// compaction loudly — nothing is ever silently deleted), then deletes
    /// every entry below the watermark EXCEPT the pinned never-evict set:
    /// the last GoalSet, the last CriteriaSet, the last Decision and every
    /// unresolved BlockerOpened. The head checkpoint is rewritten in the
    /// SAME transaction as the deletion, folded from the pre-prune stream
    /// (compaction is projection-preserving, never FIFO-evicting).
    pub fn compact_typed_ledger(&self) -> faktor_core::Result<LedgerCompactReport> {
        let entries = self.all_entries_decoded()?;
        let entries_before = entries.len() as i64;
        if entries.is_empty() {
            return Ok(LedgerCompactReport {
                entries_before: 0,
                deleted: 0,
                kept: 0,
                pinned: Vec::new(),
                checkpoint_seq: 0,
            });
        }
        // Current materialized head FIRST: it already folds everything the
        // stream holds plus what EARLIER compactions pruned (epochs, child
        // records, routing tails...). Compaction folds onto it — a
        // full-stream replay would silently forget head-only content.
        let mut head = self.ledger_ensure_head()?;
        let checkpoint_seq = entries.last().expect("non-empty").seq;
        if head.checkpoint_seq < checkpoint_seq {
            let after = Some(head.checkpoint_seq);
            let rows = self
                .manager
                .store()
                .ledger_entries(self.id, after, u64::MAX)
                .map_err(map_store_err)?;
            let mut extra = Vec::with_capacity(rows.len());
            for row in rows {
                extra.push(self.decode_row(&row)?);
            }
            fold_entries(&mut head, &extra)?;
        }
        // Pinned never-evict set: the LAST GoalSet / CriteriaSet / Decision
        // entry and EVERY unresolved BlockerOpened's own opener entry.
        let mut pinned: Vec<i64> = Vec::new();
        let mut last: BTreeMap<&'static str, i64> = BTreeMap::new();
        let mut open_opener_seqs: Vec<i64> = Vec::new();
        for entry in &entries {
            match &entry.payload {
                LedgerPayload::GoalSet { .. } => {
                    last.insert(ENTRY_GOAL_SET, entry.seq);
                }
                LedgerPayload::CriteriaSet { .. } => {
                    last.insert(ENTRY_CRITERIA_SET, entry.seq);
                }
                LedgerPayload::Decision { .. } => {
                    last.insert(ENTRY_DECISION, entry.seq);
                }
                LedgerPayload::BlockerOpened { reason } => open_opener_seqs.extend(
                    head.open_blockers
                        .iter()
                        .any(|r| r == reason)
                        .then_some(entry.seq),
                ),
                _ => {}
            }
        }
        for seq in last.values() {
            pinned.push(*seq);
        }
        pinned.extend(open_opener_seqs);
        pinned.sort_unstable();
        pinned.dedup();
        let head_json = head_to_json(&head)?;
        let head_value = serde_json::from_str(&head_json)
            .map_err(|e| SessionError::Internal(format!("head json: {e}")))?;
        let deleted = self
            .manager
            .store()
            .compact_ledger(
                self.id,
                checkpoint_seq + 1,
                &pinned,
                head_value,
                checkpoint_seq,
                LEDGER_HEAD_SCHEMA_V,
            )
            .map_err(map_store_err)? as i64;
        Ok(LedgerCompactReport {
            entries_before,
            deleted,
            kept: entries_before - deleted,
            pinned,
            checkpoint_seq,
        })
    }

    /// The materialized typed view: the head fold + entry count. This is
    /// the read surface the runtime and the never-lose tests assert on.
    pub fn ledger_view(&self) -> faktor_core::Result<crate::ledger::LedgerView> {
        let head = self.ledger_ensure_head()?;
        let entry_count = self
            .manager
            .store()
            .ledger_max_seq(self.id)
            .map_err(map_store_err)?;
        Ok(LedgerView { head, entry_count })
    }

    /// Session-open ledger verification (audit 27 never-lose contract): the
    /// typed entry stream is strictly decoded in FULL and the head is
    /// brought current. An entry that fails its schema decode (unknown
    /// version, unknown type, shape violation) FAILS THE OPEN loudly — the
    /// ledger never silently drops a row. Called on every handle open
    /// (manager get/list); no entries => a fast no-op.
    pub fn ledger_verify_open(&self) -> faktor_core::Result<()> {
        if self
            .manager
            .store()
            .ledger_max_seq(self.id)
            .map_err(map_store_err)?
            == 0
        {
            return Ok(());
        }
        let _ = self.all_entries_decoded()?;
        let _ = self.ledger_ensure_head()?;
        Ok(())
    }
}

/// The session's typed ledger view: the materialized head + entry count.
#[derive(Debug, Clone, PartialEq)]
pub struct LedgerView {
    pub head: LedgerHead,
    /// Newest entry seq (0 = empty ledger). After a compaction this can sit
    /// below `head.checkpoint_seq` (the head is folded AHEAD of the pruned
    /// stream by design).
    pub entry_count: i64,
}

/// True when `reason` names a currently open blocker (view read helper used
/// by the runtime gate integration).
pub fn blocker_is_open(head: &LedgerHead, reason: &str) -> bool {
    head.open_blockers.iter().any(|r| r == reason)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::tests::{session, test_manager};
    use std::sync::Arc;

    fn raw_sql(m: &crate::SessionManager, sql: &str) {
        m.store().sql_execute(sql).unwrap();
    }

    fn turn_entries(s: &SessionHandle, turn: u64) {
        s.ledger_decision(
            &format!("step-{turn}"),
            &format!("choice-{turn}"),
            &format!("rationale-{turn}"),
        )
        .unwrap();
        s.ledger_routing_decision(
            turn,
            &format!("prov-{turn}"),
            &format!("model-{turn}"),
            "quality fit",
            42,
        )
        .unwrap();
        s.ledger_plan_step_added(
            turn as u32,
            &format!("do step {turn}"),
            Some(turn as u32 - 1),
        )
        .unwrap();
        s.ledger_verify_run(
            &[LedgerCheckRun {
                id: format!("check-{turn}"),
                passed: true,
            }],
            "passed",
        )
        .unwrap();
        s.ledger_turn_completed(turn).unwrap();
    }

    fn assert_never_lost(s: &SessionHandle, goal: &str, open_blockers: &[&str]) {
        let view = s.ledger_view().unwrap();
        assert_eq!(view.head.goal, goal, "goal survives compaction");
        assert!(
            !view.head.criteria.is_empty(),
            "criteria survive compaction"
        );
        for b in open_blockers {
            assert!(
                view.head.open_blockers.iter().any(|o| o == b),
                "open blocker {b} must survive compaction"
            );
        }
        assert!(
            !view.head.decisions.is_empty(),
            "last decision must survive compaction"
        );
        // Entry-level: the pinned rows still exist in the stream.
        let mut page = s.ledger_entries_page(None, 500).unwrap();
        let mut entries = Vec::new();
        loop {
            let cursor = page.entries.last().map(|e| e.seq);
            entries.extend(page.entries);
            if !page.has_more {
                break;
            }
            page = s
                .ledger_entries_page(cursor, 500)
                .expect("paged entries decode");
        }
        assert!(
            entries
                .iter()
                .any(|e| matches!(&e.payload, LedgerPayload::GoalSet { goal: g } if g == goal)),
            "GoalSet entry must survive"
        );
        assert!(
            entries
                .iter()
                .any(|e| matches!(&e.payload, LedgerPayload::CriteriaSet { .. })),
            "CriteriaSet entry must survive"
        );
        assert!(
            entries
                .iter()
                .any(|e| matches!(&e.payload, LedgerPayload::Decision { .. })),
            "last Decision entry must survive"
        );
        for b in open_blockers {
            assert!(
                entries.iter().any(|e| matches!(
                    &e.payload,
                    LedgerPayload::BlockerOpened { reason } if reason == b
                )),
                "unresolved BlockerOpened {b} entry must survive"
            );
        }
        let head = s.manager.store().ledger_head(s.id).unwrap();
        assert!(
            head.is_some() && head.unwrap().checkpoint_seq > 0,
            "latest checkpoint head must be present"
        );
    }

    #[test]
    fn typed_accessors_roundtrip_and_view_folds() {
        let (_d, m) = test_manager();
        let s = session(&m);
        assert!(s.ledger_view().unwrap().head.goal.is_empty());
        s.ledger_goal_set("implement the ledger").unwrap();
        s.ledger_criteria_set(
            &["cargo test".into(), "no warnings".into()],
            "cargo test + no warnings",
        )
        .unwrap();
        s.ledger_blocker_opened("check alpha failed").unwrap();
        s.ledger_decision("pick", "rust", "ecosystem").unwrap();
        s.ledger_plan_step_added(0, "read the spec", None).unwrap();
        s.ledger_plan_step_added(1, "write code", Some(0)).unwrap();
        s.ledger_routing_decision(9, "ollama", "qwen3.8", "cost", 3)
            .unwrap();
        s.ledger_epoch_bumped(None, 7).unwrap();
        s.ledger_failure_recorded("test failed: x").unwrap();
        s.ledger_turn_completed(9).unwrap();
        let view = s.ledger_view().unwrap();
        assert_eq!(view.head.goal, "implement the ledger");
        assert_eq!(view.head.criteria, vec!["cargo test", "no warnings"]);
        assert_eq!(view.head.open_blockers, vec!["check alpha failed"]);
        assert_eq!(view.head.epoch, Some(7));
        assert_eq!(view.head.routing_count, 1);
        assert_eq!(view.head.routing_tail[0].provider, "ollama");
        assert_eq!(view.head.plan_steps[1].parent_index, Some(0));
        // Blocker resolution removes the reason from the fold.
        s.ledger_blocker_resolved("check alpha failed").unwrap();
        let view = s.ledger_view().unwrap();
        assert!(view.head.open_blockers.is_empty());
        // Duplicate open of the same reason is a no-op, never an error.
        s.ledger_blocker_opened("again").unwrap();
        assert!(s.ledger_blocker_opened("again").unwrap().is_none());
        assert_eq!(
            s.ledger_view().unwrap().head.open_blockers,
            vec!["again".to_string()]
        );
        // Resolving a reason that is not open is a typed error.
        let err = s.ledger_blocker_resolved("never-opened").unwrap_err();
        assert_eq!(err.kind, faktor_core::ErrorKind::Conflict);
    }

    #[test]
    fn out_of_order_child_finish_is_a_typed_append_error() {
        let (_d, m) = test_manager();
        let s = session(&m);
        // Finish without start: typed error at append time.
        let err = s.ledger_child_finished(77, "done").unwrap_err();
        assert!(err.to_string().contains("no open start"), "{err}");
        // Start -> finish is legal.
        s.ledger_child_started(77, 3, 2, "verify the change")
            .unwrap();
        s.ledger_child_finished(77, "verified").unwrap();
        // Double finish is again a typed error.
        let err = s.ledger_child_finished(77, "again").unwrap_err();
        assert!(err.to_string().contains("no open start"), "{err}");
        // Double START of the same running agent conflicts.
        s.ledger_child_started(78, 3, 2, "second").unwrap();
        let err = s.ledger_child_started(78, 3, 2, "dup").unwrap_err();
        assert_eq!(err.kind, faktor_core::ErrorKind::Conflict);
        // A finished agent may start again.
        s.ledger_child_finished(78, "done").unwrap();
        s.ledger_child_started(78, 3, 2, "restart").unwrap();
        assert!(s.ledger_child_finished(78, "done again").is_ok());
        // Zero ids are malformed.
        assert!(s.ledger_child_started(0, 1, 1, "x").is_err());
        // The fold keeps children with their outcomes.
        let view = s.ledger_view().unwrap();
        assert_eq!(view.head.children.len(), 2);
        assert_eq!(
            view.head
                .children
                .iter()
                .find(|c| c.agent_id == 77)
                .unwrap()
                .outcome
                .as_deref(),
            Some("verified")
        );
    }

    #[test]
    fn never_lose_survives_20_turns_and_6_compactions() {
        // Requirement 4a: 20 turns with decisions/blocks/children, 6
        // compactions, then GoalSet + CriteriaSet + the last Decision +
        // every unresolved BlockerOpened and the latest head are all there.
        let (_d, m) = test_manager();
        let s = session(&m);
        s.ledger_goal_set("goal-0").unwrap();
        s.ledger_criteria_set(&["c0".into(), "c1".into()], "c0 c1")
            .unwrap();
        s.ledger_epoch_bumped(None, 1).unwrap();
        for turn in 1..=20u64 {
            turn_entries(&s, turn);
            if turn == 5 {
                s.ledger_child_started(500 + turn, turn, 1, "subagent verify")
                    .unwrap();
                s.ledger_child_finished(500 + turn, "done").unwrap();
            }
            if turn == 7 {
                s.ledger_blocker_opened("blocker-seven").unwrap();
            }
            if turn == 13 {
                s.ledger_blocker_opened("blocker-thirteen").unwrap();
            }
            if turn == 9 {
                // A blocker opened and later resolved must NOT linger.
                s.ledger_blocker_opened("resolved-nine").unwrap();
                s.ledger_blocker_resolved("resolved-nine").unwrap();
            }
            if turn % 3 == 0 {
                s.compact_typed_ledger().unwrap();
            }
        }
        // Three more compactions beyond the turns.
        for _ in 0..3 {
            s.compact_typed_ledger().unwrap();
        }
        assert_never_lost(&s, "goal-0", &["blocker-seven", "blocker-thirteen"]);
        let view = s.ledger_view().unwrap();
        // The last Decision entry is the decision of turn 20.
        let last = view.head.decisions.first().unwrap();
        assert_eq!(last.step, "step-20");
        assert_eq!(view.head.epoch, Some(1));
        assert_eq!(view.head.routing_count, 20);
        // Head-only content that compaction pruned from the entry stream is
        // still folded in the head: the child-agent records of turn 5 and
        // the plan DAG of the compacted steps.
        assert_eq!(view.head.children.len(), 1, "children survive in the head");
        assert_eq!(view.head.children[0].agent_id, 505);
        assert!(
            !view.head.plan_steps.is_empty(),
            "plan DAG survives in the head"
        );
        // Compaction pruned the stream: entries are bounded to the pinned
        // set + nothing newer (goal, criteria, 2 blockers, 1 decision).
        let entries = collect_all(&s);
        assert_eq!(entries.len(), 5, "only pinned entries survive: {entries:?}");
        // Resolved blockers do not linger in the fold.
        assert!(!view.head.open_blockers.iter().any(|r| r == "resolved-nine"));
    }

    fn collect_all(s: &SessionHandle) -> Vec<TypedLedgerEntry> {
        let mut out = Vec::new();
        let mut cursor = None;
        loop {
            let page = s.ledger_entries_page(cursor, 500).unwrap();
            let c = page.entries.last().map(|e| e.seq);
            out.extend(page.entries);
            if !page.has_more {
                break;
            }
            cursor = c;
        }
        out
    }

    #[test]
    fn watermark_holds_across_five_compacting_turns() {
        // Requirement 3's watermark test: five turns, each compacting.
        let (_d, m) = test_manager();
        let s = session(&m);
        s.ledger_goal_set("the-goal").unwrap();
        s.ledger_criteria_set(&["must compile".into()], "must compile")
            .unwrap();
        for turn in 1..=5u64 {
            s.ledger_blocker_opened(&format!("blocker-{turn}")).unwrap();
            turn_entries(&s, turn);
            s.compact_typed_ledger().unwrap();
            // After EVERY compaction all protected content is present.
            assert_never_lost(&s, "the-goal", &[&format!("blocker-{turn}")]);
        }
        // With five open blockers, all five survive compaction.
        s.compact_typed_ledger().unwrap();
        assert_never_lost(
            &s,
            "the-goal",
            &[
                "blocker-1",
                "blocker-2",
                "blocker-3",
                "blocker-4",
                "blocker-5",
            ],
        );
    }

    #[test]
    fn crafted_unknown_schema_version_makes_every_typed_reader_error_loudly() {
        // Requirement 4b: a row written with schema_ver 999 makes every
        // typed reader error loudly (Corrupt), and compaction refuses.
        let (_d, m) = test_manager();
        let s = session(&m);
        s.ledger_goal_set("g").unwrap();
        raw_sql(
            &m,
            &format!(
                "INSERT INTO ledger_entry(session_id, seq, entry_type, schema_ver, payload, created_ms)
                 VALUES ({}, 2, 'goal_set', 999, '{{\"goal\": \"future\"}}', 1)",
                s.id().raw()
            ),
        );
        // Every typed reader fails loudly.
        let err = s.ledger_view().unwrap_err();
        assert!(err.to_string().contains("999"), "{err}");
        let err = s.ledger_verify_open().unwrap_err();
        assert!(err.to_string().contains("999"), "{err}");
        let err = s.ledger_entries_page(None, 10).unwrap_err();
        assert!(err.to_string().contains("999"), "{err}");
        // Compaction refuses to checkpoint it: nothing deleted, head intact.
        let head_before = s.manager.store().ledger_head(s.id()).unwrap();
        let err = s.compact_typed_ledger().unwrap_err();
        assert!(err.to_string().contains("999"), "{err}");
        let head_after = s.manager.store().ledger_head(s.id()).unwrap();
        assert_eq!(head_before, head_after, "compaction must not checkpoint");
        let rows = s.manager.store().ledger_entries(s.id(), None, 10).unwrap();
        assert_eq!(rows.len(), 2, "nothing was deleted around the corrupt row");
        // Reopening the session fails loudly too.
        let err = m.get_session(s.id()).unwrap_err();
        assert!(err.to_string().contains("999"), "{err}");
    }

    #[test]
    fn payload_shape_violation_is_loud_never_silent() {
        let (_d, m) = test_manager();
        let s = session(&m);
        s.ledger_goal_set("g").unwrap();
        // Valid JSON, wrong shape for its tag: strict decode fails loudly.
        raw_sql(
            &m,
            &format!(
                "INSERT INTO ledger_entry(session_id, seq, entry_type, schema_ver, payload, created_ms)
                 VALUES ({}, 2, 'goal_set', 1, '{{\"nonsense\": true}}', 1)",
                s.id().raw()
            ),
        );
        let err = s.ledger_view().unwrap_err();
        assert!(err.to_string().contains("v1 schema"), "{err}");
        assert!(m.get_session(s.id()).is_err());
    }

    #[test]
    fn concurrent_appends_interleave_without_losing_entries() {
        // Requirement 4d: two handles append concurrently; seqs stay unique
        // and every entry lands.
        let (_d, m) = test_manager();
        let s = Arc::new(session(&m));
        let s1 = s.clone();
        let s2 = s.clone();
        let t1 = std::thread::spawn(move || {
            for i in 0..50u32 {
                // Even plan indexes (0..98); index 0 is the root step.
                let parent = if i == 0 { None } else { Some(2 * i - 1) };
                s1.ledger_plan_step_added(2 * i, &format!("a{i}"), parent)
                    .unwrap();
            }
        });
        let t2 = std::thread::spawn(move || {
            for i in 0..50u32 {
                s2.ledger_plan_step_added(2 * i + 1, &format!("b{i}"), Some(2 * i))
                    .unwrap();
            }
        });
        t1.join().unwrap();
        t2.join().unwrap();
        let entries = collect_all(&s);
        assert_eq!(entries.len(), 100, "every concurrent append must land");
        let mut seqs: Vec<i64> = entries.iter().map(|e| e.seq).collect();
        seqs.sort_unstable();
        seqs.dedup();
        assert_eq!(seqs.len(), 100, "seqs must be unique");
        let mut indexes: Vec<u32> = Vec::new();
        for e in &entries {
            if let LedgerPayload::PlanStepAdded { step_index, .. } = &e.payload {
                indexes.push(*step_index);
            }
        }
        indexes.sort_unstable();
        indexes.dedup();
        assert_eq!(indexes.len(), 100, "every planned step present");
        // The fold is coherent after the interleaving.
        let view = s.ledger_view().unwrap();
        assert_eq!(view.head.plan_steps.len(), 100);
    }

    #[test]
    fn crash_between_append_and_checkpoint_recovers_both_orders() {
        // Requirement 4e: a crash between an entry append and the head
        // checkpoint rebuilds the head from entries — in both orders.
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        // First "process": appends, NO head checkpoint yet, "crash".
        let (store, cas) = (root.join("store"), root.join("cas"));
        {
            let m = crate::SessionManager::open(&store, &cas, true).unwrap();
            let ws = m.create_workspace("/w").unwrap();
            let s = m.create_session(ws, "t", "p", "m").unwrap();
            let id = s.id();
            s.ledger_goal_set("goal-batch-1").unwrap();
            s.ledger_decision("d", "c", "r").unwrap();
            assert!(
                m.store().ledger_head(id).unwrap().is_none(),
                "crash before the first checkpoint"
            );
            // Second "process" reopens and must rebuild the head.
            let m2 = crate::SessionManager::open(&store, &cas, true).unwrap();
            let s2 = m2.get_session(id).unwrap().unwrap();
            let head = m2.store().ledger_head(id).unwrap().unwrap();
            assert_eq!(head.checkpoint_seq, 2);
            let view = s2.ledger_view().unwrap();
            assert_eq!(view.head.goal, "goal-batch-1");
            // Third "process": head EXISTS, more appends land, crash again
            // before the checkpoint folds them.
            s2.ledger_decision("d2", "c2", "r2").unwrap();
            s2.ledger_routing_decision(2, "p", "m", "r", 1).unwrap();
            let m3 = crate::SessionManager::open(&store, &cas, true).unwrap();
            let s3 = m3.get_session(id).unwrap().unwrap();
            let view = s3.ledger_view().unwrap();
            assert_eq!(view.head.goal, "goal-batch-1");
            assert_eq!(view.head.decisions.len(), 2);
            assert_eq!(view.head.routing_count, 1);
            assert_eq!(
                m3.store().ledger_head(id).unwrap().unwrap().checkpoint_seq,
                4
            );
        }
    }

    #[test]
    fn corrupted_head_recovers_but_corrupted_entry_fails_the_open() {
        // Requirement 4f: a hand-corrupted head recovers by replay from
        // entries; an entry whose JSON fails its schema decode fails the
        // session open loudly — never silently dropped.
        let dir = tempfile::tempdir().unwrap();
        let (store, cas) = (dir.path().join("store"), dir.path().join("cas"));
        let id = {
            let m = crate::SessionManager::open(&store, &cas, true).unwrap();
            let ws = m.create_workspace("/w").unwrap();
            let s = m.create_session(ws, "t", "p", "m").unwrap();
            s.ledger_goal_set("the-goal").unwrap();
            s.ledger_criteria_set(&["c".into()], "c").unwrap();
            s.ledger_decision("d", "c", "r").unwrap();
            s.ledger_blocker_opened("open-b").unwrap();
            // Head materialized.
            let _ = s.ledger_view().unwrap();
            // Hand-corrupt the head JSON.
            raw_sql(
                &m,
                &format!(
                    "UPDATE ledger_head SET head_json = 'garbage{{{{' WHERE session_id = {}",
                    s.id().raw()
                ),
            );
            s.id()
        };
        // Open recovers by replaying the surviving entries.
        let m2 = crate::SessionManager::open(&store, &cas, true).unwrap();
        let h2 = m2.get_session(id).unwrap().unwrap();
        let view = h2.ledger_view().unwrap();
        assert_eq!(view.head.goal, "the-goal");
        assert_eq!(
            view.head.open_blockers,
            vec!["open-b".to_string()],
            "entry replay must rebuild the full head"
        );
        assert_eq!(
            m2.store().ledger_head(id).unwrap().unwrap().checkpoint_seq,
            4
        );
        // Now corrupt an ENTRY payload: the next open fails loudly.
        h2.ledger_decision("d2", "c2", "r2").unwrap();
        let last_seq = m2.store().ledger_max_seq(id).unwrap();
        raw_sql(
            &m2,
            &format!(
                "UPDATE ledger_entry SET payload = 'not-json{{{{' WHERE session_id = {} AND seq = {last_seq}",
                id.raw()
            ),
        );
        let err = m2.get_session(id).unwrap_err();
        assert!(
            !err.to_string().is_empty(),
            "corrupt entry must fail the open loudly"
        );
        let err = h2.ledger_view().unwrap_err();
        assert!(err.to_string().contains("payload"), "{err}");
    }

    #[test]
    fn journal_replay_fails_loudly_on_crafted_future_payload_version() {
        // A journal event row crafted with payload_ver 999 fails the typed
        // replay loudly (never a silent v1 parse).
        let (_d, m) = test_manager();
        let s = session(&m);
        s.submit_prompt("work", &[]).unwrap();
        raw_sql(
            &m,
            &format!(
                "UPDATE event SET payload_ver = 999 WHERE session_id = {} AND kind = 'prompt_received'",
                s.id().raw()
            ),
        );
        let err = s.replay_journal().unwrap_err();
        assert!(err.to_string().contains("999"), "{err}");
    }

    #[test]
    fn typed_ledger_is_per_session() {
        let (_d, m) = test_manager();
        let s1 = session(&m);
        let ws = m.create_workspace("/w2").unwrap();
        let s2 = m.create_session(ws, "t2", "p", "m").unwrap();
        s1.ledger_goal_set("one").unwrap();
        assert!(s2.ledger_view().unwrap().head.goal.is_empty());
        s2.ledger_goal_set("two").unwrap();
        assert_eq!(s1.ledger_view().unwrap().head.goal, "one");
        assert_eq!(s2.ledger_view().unwrap().head.goal, "two");
    }

    #[test]
    fn hostile_entry_bounds_are_rejected_before_write() {
        let (_d, m) = test_manager();
        let s = session(&m);
        assert!(s.ledger_goal_set("").is_err());
        assert!(s.ledger_goal_set(&"x".repeat(MAX_LEDGER_TEXT + 1)).is_err());
        assert!(s.ledger_blocker_opened("").is_err());
        assert!(s.ledger_decision("s", "", "r").is_err());
        assert!(s.ledger_verify_run(&[], "passed").is_err());
        assert!(s
            .ledger_verify_run(
                &[LedgerCheckRun {
                    id: "c".into(),
                    passed: true
                }],
                "typo"
            )
            .is_err());
        assert!(s.ledger_epoch_bumped(Some(3), 3).is_err());
        assert!(s.ledger_plan_step_added(300, "x", None).is_err());
        assert!(s.ledger_plan_step_added(4, "x", Some(4)).is_err());
        // None of the rejections wrote anything.
        assert_eq!(collect_all(&s).len(), 0);
    }
}
