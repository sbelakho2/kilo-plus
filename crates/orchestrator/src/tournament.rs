//! Multi-candidate implementation tournaments (N = 2..=4), the deterministic
//! winner-selection engine behind `TaskExecutor::start_tournament`.
//!
//! One tournament fans the SAME goal and the SAME criteria out to N
//! ISOLATED candidate worktrees through the existing executor/assignment
//! machinery (`TaskExecutor::start_tournament` builds an N-item plan of
//! `Implementation` items under `OwnershipSpec::IsolatedWorktree` and starts
//! it through the one task-start authority). Each candidate is driven to a
//! terminal child state; every settlement then carries
//!
//! - the deterministic verification spec (the SAME derived check set for
//!   every candidate — a drift is a typed refusal, never silently ranked),
//! - the durable verification record id + its pass/fail verdict,
//! - an INDEPENDENT review verdict (the reviewer may not be any candidate
//!   child), and
//! - the candidate's measured cost (microUSD) and wall time.
//!
//! Winner ordering is documented and total:
//!
//! ```text
//! verification pass  >  review verdict rank  >  lower cost_micro  >  candidate ordinal
//! ```
//!
//! A candidate that failed verification (or has no independent review, or
//! is not Done) is NEVER eligible — even when it is the cheapest. Among
//! eligible candidates the highest review rank wins; ties break on the
//! lower cost; remaining ties break on the lower candidate ordinal (the
//! `child-0..child-{N-1}` plan order), so a tournament is reproducible.
//!
//! Durability: the whole lifecycle is a typed durable ledger stream on the
//! PARENT session — `TournamentStarted`, zero or more `CandidateSettled`,
//! one `TournamentDecided` — pinned across compaction. [`Tournament::reopen`]
//! folds those rows back into tournaments, so a crashed or restarted
//! executor reconstructs the candidate set and the winner exactly.
//!
//! Integration is NEVER automatic: the winner's worktree is only PROPOSED
//! (it is the candidate id the caller may hand to the explicit
//! approved-merge path). Losers are discarded — their registry rows are
//! settled terminal and their isolated worktree directories/rows are
//! removed — with the why recorded on the decision ledger row.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use faktor_core::id::{SessionId, VerificationRecordId, WorkspaceId};
use faktor_session::ledger::{
    TournamentCandidateRow, TournamentCheckSpec, TournamentCriterionRow, TournamentSettlementRow,
    MAX_TOURNAMENT_CANDIDATES, MAX_TOURNAMENT_CRITERIA, MAX_TOURNAMENT_ID, MAX_TOURNAMENT_OUTCOME,
    MAX_TOURNAMENT_TEXT, MIN_TOURNAMENT_CANDIDATES, TOURNAMENT_OUTCOME_ABORTED,
    TOURNAMENT_REVIEW_BLOCK, TOURNAMENT_REVIEW_CLEAN, TOURNAMENT_REVIEW_CONCERN,
    TOURNAMENT_STATE_CANCELLED, TOURNAMENT_STATE_DONE, TOURNAMENT_STATE_FAILED,
};
use faktor_session::{LedgerPayload, SessionHandle, SessionManager};

use crate::runtime::{OrchestratorRuntime, REGISTRY_ROW_KIND};
use crate::{ChildState, WorkItem};

/// The supported candidate band of one tournament (typed, enforced).
pub const MIN_CANDIDATES: usize = MIN_TOURNAMENT_CANDIDATES;
pub const MAX_CANDIDATES: usize = MAX_TOURNAMENT_CANDIDATES;
/// Max criteria per tournament.
pub const MAX_CRITERIA: usize = MAX_TOURNAMENT_CRITERIA;
/// Max bytes of one criterion id/spec.
pub const MAX_CRITERION_BYTES: usize = MAX_TOURNAMENT_TEXT;
/// Max bytes of a tournament goal.
pub const MAX_GOAL_BYTES: usize = MAX_TOURNAMENT_TEXT;
/// Max bytes of a tournament id / run family / child id.
pub const MAX_ID_BYTES: usize = MAX_TOURNAMENT_ID;
/// Max bytes of a decision rationale.
pub const MAX_RATIONALE_BYTES: usize = MAX_TOURNAMENT_OUTCOME;

/// Typed refusals of the tournament engine/logic. Every boundary (DTO,
/// executor, ledger decode) maps these onto its own error space; nothing in
/// this module silently clamps, sorts by accident, or ranks a candidate the
/// rules exclude.
#[derive(Debug, thiserror::Error)]
pub enum TournamentError {
    #[error("tournament candidate count {requested} is outside the supported band {min}..={max}")]
    InvalidCandidateCount {
        requested: usize,
        min: usize,
        max: usize,
    },
    #[error("tournament criteria must be 1..={max} (got {got})")]
    InvalidCriteriaCount { got: usize, max: usize },
    #[error("invalid criterion: {0}")]
    InvalidCriterion(String),
    #[error("duplicate criterion id {0:?}")]
    DuplicateCriterion(String),
    #[error("invalid tournament id: {0}")]
    InvalidId(String),
    #[error("tournament {0:?} was not found")]
    NotFound(String),
    #[error("candidate {0:?} is unknown to the tournament")]
    UnknownCandidate(String),
    #[error(
        "candidate {0:?} already settled; a duplicate settlement is corruption, not a re-rank"
    )]
    DuplicateSettlement(String),
    #[error("candidate settlement carries an illegal state ({0:?})")]
    IllegalSettlementState(CandidateState),
    #[error(
        "candidate settlement verification spec differs byte-for-byte from the derived check set"
    )]
    VerificationSpecDrift,
    #[error("reviewer {0:?} is not independent from the candidate band")]
    ReviewNotIndependent(String),
    #[error("tournament is {0:?}; the operation needs an open tournament")]
    NotOpen(TournamentState),
    #[error("no eligible winner: {0}")]
    NoEligibleWinner(String),
    #[error("durable tournament {tournament_id:?} is corrupt: {detail}")]
    Corrupt {
        tournament_id: String,
        detail: String,
    },
    #[error("oversized: {0}")]
    Oversized(String),
    #[error("session ledger: {0}")]
    Ledger(String),
    #[error("candidate worktree cleanup failed: {0}")]
    Cleanup(String),
}

fn map_ledger(e: faktor_core::Error) -> TournamentError {
    // A corrupt row and a lost store are different truths; the `kind` tag
    // keeps them distinguishable for callers.
    match e.kind {
        faktor_core::ErrorKind::NotFound => TournamentError::NotFound(e.message),
        _ => TournamentError::Ledger(e.message),
    }
}

// ---------------------------------------------------------------- criteria

/// One tournament acceptance criterion: a deterministic content id plus the
/// exact specification text. The id is derived from the spec, so two
/// candidates handed the same spec carry byte-identical ids; a drift is
/// detectable without trusting the caller.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Criterion {
    pub id: String,
    pub spec: String,
}

impl Criterion {
    /// Derive a criterion from its specification text (bounded, non-empty,
    /// deterministic id).
    pub fn derive(spec: &str) -> Result<Self, TournamentError> {
        let spec = spec.trim();
        if spec.is_empty() {
            return Err(TournamentError::InvalidCriterion(
                "criterion spec is empty".into(),
            ));
        }
        if spec.len() > MAX_CRITERION_BYTES {
            return Err(TournamentError::Oversized(format!(
                "criterion of {} bytes exceeds MAX_CRITERION_BYTES ({MAX_CRITERION_BYTES})",
                spec.len()
            )));
        }
        Ok(Self {
            id: stable_criterion_id(spec),
            spec: spec.to_string(),
        })
    }
}

/// FNV-1a (64-bit) content id of one criterion spec. Deterministic across
/// processes and platforms; collision-resistant enough for candidate
/// identity (a collision merely makes two distinct specs share an id, which
/// the duplicate-id refusal then rejects loudly).
fn stable_criterion_id(spec: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in spec.as_bytes() {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("c-{hash:016x}")
}

/// The deterministic verification spec derived from a criterion set: one
/// check per criterion, in criterion order. Every candidate of a tournament
/// is verified against EXACTLY this set (a settlement carrying anything
/// else is [`TournamentError::VerificationSpecDrift`]).
pub fn derive_check_specs(criteria: &[Criterion]) -> Vec<TournamentCheckSpec> {
    criteria
        .iter()
        .map(|c| TournamentCheckSpec {
            id: c.id.clone(),
            spec: c.spec.clone(),
        })
        .collect()
}

/// The AGGREGATE root check set of one orchestrated run: the tournament
/// derivation applied to the run's OWN acceptance criteria. The post-run
/// settlement verifies the whole run against EXACTLY this set (one
/// deterministic check per criterion, in order) — never per candidate.
/// Every spec is validated with the tournament criterion bounds; an empty
/// criterion list yields an empty (vacuously passing) set.
pub fn aggregate_check_specs(
    criteria_specs: &[String],
) -> Result<Vec<TournamentCheckSpec>, TournamentError> {
    let mut criteria = Vec::with_capacity(criteria_specs.len());
    for spec in criteria_specs {
        criteria.push(Criterion::derive(spec)?);
    }
    Ok(derive_check_specs(&criteria))
}

fn specs_byte_equal(a: &[TournamentCheckSpec], b: &[TournamentCheckSpec]) -> bool {
    match (serde_json::to_vec(a), serde_json::to_vec(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

/// The canonical criteria text every candidate item carries (and the
/// durable task row mirrors). Pure function of the criterion set.
pub fn canonical_criteria_text(criteria: &[Criterion]) -> String {
    criteria
        .iter()
        .map(|c| c.spec.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Assert that EVERY candidate work item carries byte-identical criterion
/// ids/specs and the same goal-facing summary. Returns a typed refusal on
/// the first drift — a tournament never fans out divergent goals.
pub fn assert_candidates_identical(
    items: &[WorkItem],
    criteria: &[Criterion],
) -> Result<(), TournamentError> {
    let expected: Vec<String> = criteria.iter().map(|c| c.spec.clone()).collect();
    let expected_bytes = serde_json::to_vec(&expected).unwrap_or_default();
    let mut first_summary: Option<&str> = None;
    for item in items {
        if item.acceptance_checks != expected {
            return Err(TournamentError::VerificationSpecDrift);
        }
        let bytes = serde_json::to_vec(&item.acceptance_checks).unwrap_or_default();
        if bytes != expected_bytes {
            return Err(TournamentError::VerificationSpecDrift);
        }
        match first_summary {
            None => first_summary = Some(item.summary.as_str()),
            Some(s) if s != item.summary => {
                return Err(TournamentError::InvalidCriterion(format!(
                    "candidate item {} does not carry the byte-identical goal summary",
                    item.id
                )));
            }
            _ => {}
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- reviews

/// Independent-review verdict ranks, ascending: `Block < Concern < Clean`.
/// A `Block` review can never win; `Clean` outranks `Concern`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewRank {
    Block,
    Concern,
    Clean,
}

impl ReviewRank {
    pub fn tag(self) -> &'static str {
        match self {
            Self::Block => TOURNAMENT_REVIEW_BLOCK,
            Self::Concern => TOURNAMENT_REVIEW_CONCERN,
            Self::Clean => TOURNAMENT_REVIEW_CLEAN,
        }
    }

    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            TOURNAMENT_REVIEW_BLOCK => Some(Self::Block),
            TOURNAMENT_REVIEW_CONCERN => Some(Self::Concern),
            TOURNAMENT_REVIEW_CLEAN => Some(Self::Clean),
            _ => None,
        }
    }
}

/// One independent review verdict: the rank plus the reviewer's identity
/// (a reviewer child/agent that is NOT any candidate of the tournament).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReviewVerdict {
    pub rank: ReviewRank,
    pub reviewer: String,
}

// ---------------------------------------------------------------- candidate

/// Lifecycle of one tournament candidate. `Running` is the initial state;
/// `Done`/`Failed`/`Cancelled` are the observed terminal child states;
/// `Discarded` is the engine's post-decision loser mark.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateState {
    Running,
    Done,
    Failed,
    Cancelled,
    Discarded,
}

impl CandidateState {
    pub fn tag(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Done => TOURNAMENT_STATE_DONE,
            Self::Failed => TOURNAMENT_STATE_FAILED,
            Self::Cancelled => TOURNAMENT_STATE_CANCELLED,
            Self::Discarded => "discarded",
        }
    }
}

/// One candidate of a tournament. `child_id` is the child id the existing
/// assignment machinery minted (`child-0..child-{N-1}` in plan order),
/// `worktree` the isolated candidate worktree path, `base_revision` the
/// base snapshot it started from. `verification` is the durable
/// verification record id and `verification_pass` its verdict;
/// `review` is the independent review; `cost_micro`/`wall_ms` the measured
/// cost axes the deterministic ordering uses.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Candidate {
    pub child_id: String,
    pub worktree: String,
    pub base_revision: String,
    pub state: CandidateState,
    pub verification: Option<VerificationRecordId>,
    pub verification_pass: Option<bool>,
    pub review: Option<ReviewVerdict>,
    pub cost_micro: u64,
    pub wall_ms: u64,
    /// Engine-internal: a settlement landed for this candidate. Not part of
    /// the wire shape (rebuilt from the durable settlement rows).
    #[serde(default, skip_serializing)]
    pub settled: bool,
}

impl Candidate {
    fn fresh(child_id: String) -> Self {
        Self {
            child_id,
            worktree: String::new(),
            base_revision: String::new(),
            state: CandidateState::Running,
            verification: None,
            verification_pass: None,
            review: None,
            cost_micro: 0,
            wall_ms: 0,
            settled: false,
        }
    }
}

// ---------------------------------------------------------------- tournament

/// Lifecycle of one tournament: `Open` once started, `Deciding` while the
/// deterministic comparison runs, `Decided` with a winner, `Aborted` with
/// every candidate discarded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TournamentState {
    Open,
    Deciding,
    Decided,
    Aborted,
}

impl TournamentState {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Decided | Self::Aborted)
    }
}

/// The durable multi-candidate tournament object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tournament {
    pub id: String,
    /// The run id whose child rows are the candidate drives.
    pub run_family: String,
    pub goal: String,
    pub criteria: Vec<Criterion>,
    pub candidates: Vec<Candidate>,
    pub winner: Option<String>,
    pub state: TournamentState,
}

/// One listing summary of a durable tournament (additive read surface): the
/// identity, lifecycle state, candidate count, proposed winner and the
/// durable decision timestamp, folded from the session's typed ledger. The
/// candidate COUNT is the durable anchor's seed count; running candidates
/// are not refreshed here (the per-id state endpoint does that).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TournamentSummary {
    pub id: String,
    pub state: TournamentState,
    pub candidate_count: usize,
    pub winner: Option<String>,
    /// `created_ms` of the durable `TournamentDecided` row; `None` while the
    /// tournament is still open.
    pub decided_ms: Option<i64>,
}

/// The input evidence of ONE candidate settlement (verification spec +
/// record + review + measured axes). Built by the executor/operator from
/// the driven candidate; persisted verbatim as a `CandidateSettled` row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateSettlement {
    pub child_id: String,
    pub worktree: String,
    pub base_revision: String,
    pub state: CandidateState,
    pub verification: Option<VerificationRecordId>,
    pub verification_pass: Option<bool>,
    /// The DERIVED check set the candidate was verified against. Must equal
    /// [`derive_check_specs`] of the tournament criteria byte-for-byte.
    pub checks: Vec<TournamentCheckSpec>,
    pub review: Option<ReviewVerdict>,
    pub cost_micro: u64,
    pub wall_ms: u64,
    /// Bounded audit text (why this settlement happened).
    pub reason: String,
}

impl CandidateSettlement {
    pub fn to_row(&self) -> TournamentSettlementRow {
        TournamentSettlementRow {
            child_id: self.child_id.clone(),
            worktree: self.worktree.clone(),
            base_revision: self.base_revision.clone(),
            state: self.state.tag().to_string(),
            verification: self.verification.map(|v| v.raw()),
            verification_pass: self.verification_pass,
            checks: self.checks.clone(),
            review: self.review.as_ref().map(|r| r.rank.tag().to_string()),
            reviewer: self.review.as_ref().map(|r| r.reviewer.clone()),
            cost_micro: self.cost_micro,
            wall_ms: self.wall_ms,
            reason: self.reason.clone(),
        }
    }

    pub fn from_row(row: &TournamentSettlementRow) -> Result<Self, TournamentError> {
        let state = match row.state.as_str() {
            TOURNAMENT_STATE_DONE => CandidateState::Done,
            TOURNAMENT_STATE_FAILED => CandidateState::Failed,
            TOURNAMENT_STATE_CANCELLED => CandidateState::Cancelled,
            other => {
                return Err(TournamentError::Corrupt {
                    tournament_id: String::new(),
                    detail: format!("settlement state {other:?} is not done|failed|cancelled"),
                })
            }
        };
        let verification =
            match row.verification {
                Some(raw) => Some(VerificationRecordId::try_from(raw).map_err(|e| {
                    TournamentError::Corrupt {
                        tournament_id: String::new(),
                        detail: format!("verification record id: {}", e.message),
                    }
                })?),
                None => None,
            };
        let review = match (&row.review, &row.reviewer) {
            (Some(rank), Some(reviewer)) => Some(ReviewVerdict {
                rank: ReviewRank::from_tag(rank).ok_or_else(|| TournamentError::Corrupt {
                    tournament_id: String::new(),
                    detail: format!("review rank {rank:?} is not block|concern|clean"),
                })?,
                reviewer: reviewer.clone(),
            }),
            (None, None) => None,
            _ => {
                return Err(TournamentError::Corrupt {
                    tournament_id: String::new(),
                    detail: "settlement has a partial review".into(),
                })
            }
        };
        Ok(Self {
            child_id: row.child_id.clone(),
            worktree: row.worktree.clone(),
            base_revision: row.base_revision.clone(),
            state,
            verification,
            verification_pass: row.verification_pass,
            checks: row.checks.clone(),
            review,
            cost_micro: row.cost_micro,
            wall_ms: row.wall_ms,
            reason: row.reason.clone(),
        })
    }
}

/// The deterministic decision of one tournament.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TournamentDecision {
    pub tournament_id: String,
    pub winner: Candidate,
    /// Why the winner won and every loser was discarded (bounded).
    pub rationale: String,
    /// `(child_id, reason)` of every discarded candidate.
    pub discarded: Vec<(String, String)>,
}

impl Tournament {
    /// Build an OPEN tournament over `n` candidates. Enforces the typed band
    /// `2..=4`, the criterion bounds and unique criterion ids. Candidate
    /// child ids are the deterministic plan order ids the existing
    /// assignment machinery mints (`child-0..child-{n-1}`).
    pub fn new(
        id: impl Into<String>,
        run_family: impl Into<String>,
        goal: impl Into<String>,
        criteria: Vec<Criterion>,
        n: usize,
    ) -> Result<Self, TournamentError> {
        let id = id.into();
        let run_family = run_family.into();
        let goal = goal.into();
        check_id(&id, "tournament id")?;
        check_id(&run_family, "tournament run family")?;
        if goal.trim().is_empty() || goal.len() > MAX_GOAL_BYTES {
            return Err(TournamentError::Oversized(format!(
                "tournament goal must be 1..={MAX_GOAL_BYTES} bytes"
            )));
        }
        if !(MIN_CANDIDATES..=MAX_CANDIDATES).contains(&n) {
            return Err(TournamentError::InvalidCandidateCount {
                requested: n,
                min: MIN_CANDIDATES,
                max: MAX_CANDIDATES,
            });
        }
        if criteria.is_empty() || criteria.len() > MAX_CRITERIA {
            return Err(TournamentError::InvalidCriteriaCount {
                got: criteria.len(),
                max: MAX_CRITERIA,
            });
        }
        let mut seen: Vec<&str> = Vec::with_capacity(criteria.len());
        for c in &criteria {
            check_id(&c.id, "criterion id")?;
            if c.spec.trim().is_empty() || c.spec.len() > MAX_CRITERION_BYTES {
                return Err(TournamentError::InvalidCriterion(format!(
                    "criterion {:?} must be 1..={MAX_CRITERION_BYTES} bytes",
                    c.id
                )));
            }
            if seen.contains(&c.id.as_str()) {
                return Err(TournamentError::DuplicateCriterion(c.id.clone()));
            }
            seen.push(&c.id);
        }
        let candidates = (0..n)
            .map(|i| Candidate::fresh(format!("child-{i}")))
            .collect();
        Ok(Self {
            id,
            run_family,
            goal,
            criteria,
            candidates,
            winner: None,
            state: TournamentState::Open,
        })
    }

    pub fn criterion_rows(&self) -> Vec<TournamentCriterionRow> {
        self.criteria
            .iter()
            .map(|c| TournamentCriterionRow {
                id: c.id.clone(),
                spec: c.spec.clone(),
            })
            .collect()
    }

    pub fn candidate_rows(&self) -> Vec<TournamentCandidateRow> {
        self.candidates
            .iter()
            .map(|c| TournamentCandidateRow {
                child_id: c.child_id.clone(),
                worktree: c.worktree.clone(),
                base_revision: c.base_revision.clone(),
            })
            .collect()
    }

    pub fn check_specs(&self) -> Vec<TournamentCheckSpec> {
        derive_check_specs(&self.criteria)
    }

    /// The eligible candidates, best first, under the documented ordering:
    /// verification pass > review rank > lower cost > candidate ordinal.
    /// Ineligible candidates (not Done, not verified-passing, no
    /// independent review) are excluded — never ranked last.
    pub fn ranking(&self) -> Vec<&Candidate> {
        let mut eligible: Vec<(usize, &Candidate)> = self
            .candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                c.state == CandidateState::Done
                    && c.verification_pass == Some(true)
                    && c.review.is_some()
            })
            .collect();
        eligible.sort_by(|(ia, a), (ib, b)| {
            review_rank_of(b)
                .cmp(&review_rank_of(a))
                .then_with(|| a.cost_micro.cmp(&b.cost_micro))
                .then_with(|| ia.cmp(ib))
        });
        eligible.into_iter().map(|(_, c)| c).collect()
    }

    /// Settle one candidate with the evidence the comparison consumes.
    /// Refusals: unknown/duplicate candidates, non-terminal settlement
    /// states, a verification spec that drifted from the derived check set
    /// (byte compare), or a reviewer inside the candidate band.
    pub fn settle_candidate(
        &mut self,
        settlement: CandidateSettlement,
    ) -> Result<(), TournamentError> {
        if self.state != TournamentState::Open {
            return Err(TournamentError::NotOpen(self.state));
        }
        if !specs_byte_equal(&settlement.checks, &self.check_specs()) {
            return Err(TournamentError::VerificationSpecDrift);
        }
        if matches!(
            settlement.state,
            CandidateState::Running | CandidateState::Discarded
        ) {
            return Err(TournamentError::IllegalSettlementState(settlement.state));
        }
        if let Some(review) = &settlement.review {
            if review.reviewer.trim().is_empty()
                || review.reviewer.len() > MAX_ID_BYTES
                || self
                    .candidates
                    .iter()
                    .any(|c| c.child_id == review.reviewer)
                || review.reviewer == settlement.child_id
            {
                return Err(TournamentError::ReviewNotIndependent(
                    review.reviewer.clone(),
                ));
            }
        }
        let candidate = self
            .candidates
            .iter_mut()
            .find(|c| c.child_id == settlement.child_id)
            .ok_or_else(|| TournamentError::UnknownCandidate(settlement.child_id.clone()))?;
        if candidate.settled {
            return Err(TournamentError::DuplicateSettlement(
                settlement.child_id.clone(),
            ));
        }
        candidate.worktree = settlement.worktree;
        candidate.base_revision = settlement.base_revision;
        candidate.state = settlement.state;
        candidate.verification = settlement.verification;
        candidate.verification_pass = settlement.verification_pass;
        candidate.review = settlement.review;
        candidate.cost_micro = settlement.cost_micro;
        candidate.wall_ms = settlement.wall_ms;
        candidate.settled = true;
        Ok(())
    }

    /// The deterministic comparison. Requires an OPEN tournament and at
    /// least one eligible candidate; on success marks every other candidate
    /// `Discarded` (with the audit reason), records the winner and returns
    /// the decision. The winner is only PROPOSED — this call never merges.
    pub fn decide(&mut self) -> Result<TournamentDecision, TournamentError> {
        if self.state != TournamentState::Open {
            return Err(TournamentError::NotOpen(self.state));
        }
        let ranking: Vec<(usize, Candidate)> = self
            .ranking()
            .into_iter()
            .map(|c| {
                let idx = self
                    .candidates
                    .iter()
                    .position(|x| x.child_id == c.child_id)
                    .expect("ranked candidate exists");
                (idx, c.clone())
            })
            .collect();
        let Some((winner_idx, winner)) = ranking.first().cloned() else {
            return Err(TournamentError::NoEligibleWinner(
                "no candidate is Done with a passing verification and an independent review".into(),
            ));
        };
        self.state = TournamentState::Deciding;
        let mut discarded: Vec<(String, String)> = Vec::new();
        for (idx, c) in self.candidates.iter_mut().enumerate() {
            if idx == winner_idx {
                continue;
            }
            let reason = discard_reason(c, idx);
            c.state = CandidateState::Discarded;
            discarded.push((c.child_id.clone(), reason));
        }
        self.winner = Some(winner.child_id.clone());
        self.state = TournamentState::Decided;
        let rationale = truncate_chars(
            &format!(
                "winner {} (verification=pass, review={}, cost_micro={}, ordinal={}); losers discarded: {}",
                winner.child_id,
                winner.review.as_ref().map(|r| r.rank.tag()).unwrap_or("none"),
                winner.cost_micro,
                winner_idx,
                discarded
                    .iter()
                    .map(|(id, why)| format!("{id} ({why})"))
                    .collect::<Vec<_>>()
                    .join("; ")
            ),
            MAX_RATIONALE_BYTES,
        );
        Ok(TournamentDecision {
            tournament_id: self.id.clone(),
            winner,
            rationale,
            discarded,
        })
    }

    /// Abort an OPEN tournament: every candidate is discarded, no winner is
    /// proposed, and the terminal state is `Aborted`.
    pub fn abort(&mut self, _reason: &str) -> Result<(), TournamentError> {
        if self.state.is_terminal() {
            return Err(TournamentError::NotOpen(self.state));
        }
        self.state = TournamentState::Aborted;
        self.winner = None;
        for c in &mut self.candidates {
            c.state = CandidateState::Discarded;
        }
        Ok(())
    }

    /// Persist the `TournamentStarted` anchor (idempotent by ledger
    /// semantics: the caller persists a tournament exactly once).
    pub fn persist_started(&self, handle: &SessionHandle) -> Result<(), TournamentError> {
        handle
            .ledger_tournament_started(
                &self.id,
                &self.run_family,
                &self.goal,
                &self.criterion_rows(),
                &self.candidate_rows(),
            )
            .map(|_| ())
            .map_err(map_ledger)
    }

    /// Persist one `CandidateSettled` row.
    pub fn persist_settlement(
        &self,
        handle: &SessionHandle,
        settlement: &CandidateSettlement,
    ) -> Result<(), TournamentError> {
        handle
            .ledger_candidate_settled(&self.id, &settlement.to_row())
            .map(|_| ())
            .map_err(map_ledger)
    }

    /// Persist the `TournamentDecided` terminal row (`decided` with a
    /// winner, or `aborted`) plus its bounded rationale.
    pub fn persist_decision(
        &self,
        handle: &SessionHandle,
        outcome: &str,
        rationale: &str,
    ) -> Result<(), TournamentError> {
        handle
            .ledger_tournament_decided(&self.id, self.winner.as_deref(), outcome, rationale)
            .map(|_| ())
            .map_err(map_ledger)
    }

    /// Fold every durable tournament row of the session into reconstructed
    /// tournaments, oldest tournament first. Row-level corruption (a
    /// settlement before its start, a duplicate settlement, a decision
    /// before any settlement, a drifted check set) is LOUD — never a
    /// silently partial tournament.
    pub fn reopen(handle: &SessionHandle) -> Result<Vec<Tournament>, TournamentError> {
        let mut by_id: BTreeMap<String, Tournament> = BTreeMap::new();
        let mut cursor: Option<i64> = None;
        loop {
            let page = handle
                .ledger_entries_page(cursor, faktor_session::MAX_LEDGER_PAGE)
                .map_err(map_ledger)?;
            for entry in &page.entries {
                match &entry.payload {
                    LedgerPayload::TournamentStarted {
                        tournament_id,
                        run_family,
                        goal,
                        criteria,
                        candidates,
                    } => {
                        if by_id.contains_key(tournament_id) {
                            return Err(TournamentError::Corrupt {
                                tournament_id: tournament_id.clone(),
                                detail: "duplicate TournamentStarted row".into(),
                            });
                        }
                        let criteria: Vec<Criterion> = criteria
                            .iter()
                            .map(|c| Criterion {
                                id: c.id.clone(),
                                spec: c.spec.clone(),
                            })
                            .collect();
                        let candidates = candidates
                            .iter()
                            .map(|c| Candidate {
                                child_id: c.child_id.clone(),
                                worktree: c.worktree.clone(),
                                base_revision: c.base_revision.clone(),
                                ..Candidate::fresh(c.child_id.clone())
                            })
                            .collect();
                        by_id.insert(
                            tournament_id.clone(),
                            Tournament {
                                id: tournament_id.clone(),
                                run_family: run_family.clone(),
                                goal: goal.clone(),
                                criteria,
                                candidates,
                                winner: None,
                                state: TournamentState::Open,
                            },
                        );
                    }
                    LedgerPayload::CandidateSettled {
                        tournament_id,
                        settlement,
                    } => {
                        let tournament = by_id.get_mut(tournament_id).ok_or_else(|| {
                            TournamentError::Corrupt {
                                tournament_id: tournament_id.clone(),
                                detail: "CandidateSettled before TournamentStarted".into(),
                            }
                        })?;
                        if tournament.state != TournamentState::Open {
                            return Err(TournamentError::Corrupt {
                                tournament_id: tournament_id.clone(),
                                detail: "CandidateSettled after the decision".into(),
                            });
                        }
                        let settlement =
                            CandidateSettlement::from_row(settlement).map_err(|e| {
                                TournamentError::Corrupt {
                                    tournament_id: tournament_id.clone(),
                                    detail: e.to_string(),
                                }
                            })?;
                        tournament.settle_candidate(settlement).map_err(|e| {
                            TournamentError::Corrupt {
                                tournament_id: tournament_id.clone(),
                                detail: e.to_string(),
                            }
                        })?;
                    }
                    LedgerPayload::TournamentDecided {
                        tournament_id,
                        winner,
                        outcome,
                        rationale: _,
                    } => {
                        let tournament = by_id.get_mut(tournament_id).ok_or_else(|| {
                            TournamentError::Corrupt {
                                tournament_id: tournament_id.clone(),
                                detail: "TournamentDecided before TournamentStarted".into(),
                            }
                        })?;
                        if tournament.state != TournamentState::Open {
                            return Err(TournamentError::Corrupt {
                                tournament_id: tournament_id.clone(),
                                detail: "duplicate TournamentDecided row".into(),
                            });
                        }
                        if outcome == TOURNAMENT_OUTCOME_ABORTED {
                            tournament.state = TournamentState::Aborted;
                            tournament.winner = None;
                            for c in &mut tournament.candidates {
                                c.state = CandidateState::Discarded;
                            }
                        } else {
                            let Some(winner_id) = winner.clone() else {
                                return Err(TournamentError::Corrupt {
                                    tournament_id: tournament_id.clone(),
                                    detail: "decided row without a winner".into(),
                                });
                            };
                            if !tournament
                                .candidates
                                .iter()
                                .any(|c| c.child_id == winner_id)
                            {
                                return Err(TournamentError::Corrupt {
                                    tournament_id: tournament_id.clone(),
                                    detail: format!(
                                        "decided row names unknown winner {winner_id:?}"
                                    ),
                                });
                            }
                            for c in &mut tournament.candidates {
                                if c.child_id != winner_id {
                                    c.state = CandidateState::Discarded;
                                }
                            }
                            tournament.winner = Some(winner_id);
                            tournament.state = TournamentState::Decided;
                        }
                    }
                    _ => {}
                }
            }
            if !page.has_more {
                break;
            }
            cursor = page.entries.last().map(|e| e.seq);
        }
        Ok(by_id.into_values().collect())
    }

    /// Reconstruct ONE tournament by id (`NotFound` when no durable row
    /// names it).
    pub fn load(
        handle: &SessionHandle,
        tournament_id: &str,
    ) -> Result<Tournament, TournamentError> {
        Self::reopen(handle)?
            .into_iter()
            .find(|t| t.id == tournament_id)
            .ok_or_else(|| TournamentError::NotFound(tournament_id.to_string()))
    }

    /// The durable summaries of every tournament of one session, folded
    /// from the typed ledger (the same `reopen` fold the per-id read uses,
    /// plus the `TournamentDecided` row's `created_ms` as `decided_ms`).
    /// Deterministic id order; an unknown/empty ledger is an empty list.
    /// Corrupt rows are loud — never a silently partial listing.
    pub fn summaries(handle: &SessionHandle) -> Result<Vec<TournamentSummary>, TournamentError> {
        let tournaments = Self::reopen(handle)?;
        let mut decided: BTreeMap<String, i64> = BTreeMap::new();
        let mut cursor: Option<i64> = None;
        loop {
            let page = handle
                .ledger_entries_page(cursor, faktor_session::MAX_LEDGER_PAGE)
                .map_err(map_ledger)?;
            for entry in &page.entries {
                if let LedgerPayload::TournamentDecided { tournament_id, .. } = &entry.payload {
                    // Entries ascend by seq: the last terminal row wins.
                    decided.insert(tournament_id.clone(), entry.created_ms);
                }
            }
            if !page.has_more {
                break;
            }
            cursor = page.entries.last().map(|e| e.seq);
        }
        Ok(tournaments
            .into_iter()
            .map(|t| TournamentSummary {
                decided_ms: decided.get(&t.id).copied(),
                id: t.id,
                state: t.state,
                candidate_count: t.candidates.len(),
                winner: t.winner,
            })
            .collect())
    }
}

fn review_rank_of(c: &Candidate) -> u8 {
    match c.review.as_ref().map(|r| r.rank) {
        Some(ReviewRank::Clean) => 2,
        Some(ReviewRank::Concern) => 1,
        Some(ReviewRank::Block) | None => 0,
    }
}

fn discard_reason(c: &Candidate, ordinal: usize) -> String {
    match (c.state, c.verification_pass, c.review.is_some()) {
        (CandidateState::Discarded, _, _) => {
            format!("candidate was already discarded (ordinal {ordinal})")
        }
        (state, _, _) if state != CandidateState::Done => {
            format!("candidate ended {} (ordinal {ordinal})", state.tag())
        }
        (_, Some(false), _) => format!("verification failed (ordinal {ordinal})"),
        (_, None, _) => format!("no verification verdict (ordinal {ordinal})"),
        (_, Some(true), false) => format!("no independent review (ordinal {ordinal})"),
        (_, Some(true), true) => format!(
            "lost the deterministic comparison (review {}, cost_micro {}, ordinal {ordinal})",
            c.review.as_ref().map(|r| r.rank.tag()).unwrap_or("none"),
            c.cost_micro
        ),
    }
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut out = String::new();
    for ch in s.chars() {
        if out.len() + ch.len_utf8() > max {
            break;
        }
        out.push(ch);
    }
    out
}

fn check_id(value: &str, what: &str) -> Result<(), TournamentError> {
    if value.is_empty() || value.len() > MAX_ID_BYTES {
        return Err(TournamentError::InvalidId(format!(
            "{what} must be 1..={MAX_ID_BYTES} bytes"
        )));
    }
    if !value.is_ascii() || value.contains('/') || value.contains('\\') {
        return Err(TournamentError::InvalidId(format!(
            "{what} must be ASCII without '/' or '\\\\'"
        )));
    }
    Ok(())
}

// ------------------------------------------------- candidate location/cleanup

/// Fill RUNNING candidates' locations and observed states from the run's
/// durable registry rows (worktree path, base snapshot id, child state).
/// Settled/discarded candidates are never touched. Read-only projection.
pub fn refresh_candidates_from_registry(
    manager: &Arc<SessionManager>,
    parent: SessionId,
    tournament: &mut Tournament,
) -> Result<(), TournamentError> {
    let rows = OrchestratorRuntime::registry_rows(manager.clone(), parent, &tournament.run_family)
        .map_err(map_ledger)?;
    for row in rows {
        let Some(candidate) = tournament
            .candidates
            .iter_mut()
            .find(|c| c.child_id == row.child_id)
        else {
            continue;
        };
        if candidate.settled {
            continue;
        }
        candidate.base_revision = row.base_snapshot_id.clone().unwrap_or_default();
        if let Ok(Some(path)) = worktree_path(manager, row.workspace_id, row.worktree_id) {
            candidate.worktree = path.to_string_lossy().into_owned();
        }
        if candidate.state == CandidateState::Running {
            candidate.state = match row.state {
                ChildState::Done => CandidateState::Done,
                ChildState::Failed => CandidateState::Failed,
                ChildState::Cancelled => CandidateState::Cancelled,
                _ => CandidateState::Running,
            };
        }
    }
    Ok(())
}

fn worktree_path(
    manager: &Arc<SessionManager>,
    workspace_id: u64,
    worktree_id: u64,
) -> Result<Option<PathBuf>, TournamentError> {
    let rows = manager
        .worktrees_of(WorkspaceId::new(workspace_id))
        .map_err(map_ledger)?;
    Ok(rows
        .into_iter()
        .find(|w| (w.id.max(0) as u64) == worktree_id)
        .map(|w| PathBuf::from(w.path)))
}

/// Discard ONE loser candidate durably and physically:
///
/// 1. its registry row (when present and non-terminal) is settled
///    `Cancelled` FIRST — the zero-orphan registry invariant never sees a
///    LIVE child pointing at a removed worktree;
/// 2. its isolated worktree directory is removed (when it exists);
/// 3. its worktree row is removed (idempotent: an absent row is fine).
///
/// The caller records WHY on the tournament decision ledger row before (or
/// atomically with) this cleanup; this function only removes.
pub fn discard_candidate_worktree(
    manager: &Arc<SessionManager>,
    handle: &SessionHandle,
    run_family: &str,
    candidate: &Candidate,
) -> Result<(), TournamentError> {
    if let Ok(rows) = OrchestratorRuntime::registry_rows(manager.clone(), handle.id(), run_family) {
        if let Some(mut row) = rows.into_iter().find(|r| r.child_id == candidate.child_id) {
            if !row.state.is_terminal() {
                row.state = ChildState::Cancelled;
                row.updated_ms = manager.now_ms();
                let value = serde_json::to_string(&row).map_err(|e| {
                    TournamentError::Cleanup(format!("registry row serialization: {e}"))
                })?;
                let key = format!("{run_family}/{}", row.child_id);
                handle
                    .upsert_memory_fact(REGISTRY_ROW_KIND, &key, &value)
                    .map_err(|e| {
                        TournamentError::Cleanup(format!("registry row settle: {}", e.message))
                    })?;
            }
        }
    }
    if candidate.worktree.is_empty() {
        return Ok(());
    }
    let path = Path::new(&candidate.worktree);
    if path.exists() {
        std::fs::remove_dir_all(path).map_err(|e| {
            TournamentError::Cleanup(format!(
                "candidate {} worktree {:?}: {e}",
                candidate.child_id, candidate.worktree
            ))
        })?;
    }
    manager
        .remove_worktree(&candidate.worktree)
        .map_err(|e| TournamentError::Cleanup(format!("worktree row: {}", e.message)))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use faktor_core::id::TaskId;
    use faktor_core::id::WorktreeId;

    fn criteria_of(specs: &[&str]) -> Vec<Criterion> {
        specs
            .iter()
            .map(|s| Criterion::derive(s).expect("criterion"))
            .collect()
    }

    fn open_handle() -> (tempfile::TempDir, Arc<SessionManager>, SessionHandle) {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(dir.path().join("owner")).expect("owner dir");
        let manager = SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
            .expect("manager");
        let ws = manager
            .create_workspace(dir.path().join("owner").to_str().unwrap())
            .expect("workspace");
        let wt = WorktreeId::new(
            manager
                .put_worktree(ws, dir.path().join("owner").to_str().unwrap(), "main")
                .expect("worktree") as u64,
        );
        let handle = manager
            .create_session(ws, "tournament-test", "fake", "m")
            .expect("session");
        manager
            .adopt_identity(handle.id(), wt, TaskId::new(1))
            .expect("identity");
        (dir, manager, handle)
    }

    fn settlement(t: &Tournament, child_id: &str, pass: bool) -> CandidateSettlement {
        CandidateSettlement {
            child_id: child_id.to_string(),
            worktree: format!("/tmp/isolated/{child_id}"),
            base_revision: format!("base-{child_id}"),
            state: CandidateState::Done,
            verification: Some(VerificationRecordId::new(7)),
            verification_pass: Some(pass),
            checks: t.check_specs(),
            review: Some(ReviewVerdict {
                rank: ReviewRank::Clean,
                reviewer: "review-0".into(),
            }),
            cost_micro: 10,
            wall_ms: 100,
            reason: "settled".into(),
        }
    }

    fn settle_with(
        t: &mut Tournament,
        child_id: &str,
        pass: bool,
        rank: ReviewRank,
        cost_micro: u64,
    ) {
        let mut s = settlement(t, child_id, pass);
        s.review = Some(ReviewVerdict {
            rank,
            reviewer: "review-0".into(),
        });
        s.cost_micro = cost_micro;
        t.settle_candidate(s).expect("settlement");
    }

    #[test]
    fn candidate_band_is_enforced_typed() {
        for n in [0usize, 1, 5, 99] {
            let err = Tournament::new("tour-1", "run-1", "goal", criteria_of(&["a"]), n)
                .expect_err("out-of-band N must be refused");
            match err {
                TournamentError::InvalidCandidateCount {
                    requested,
                    min,
                    max,
                } => {
                    assert_eq!(requested, n);
                    assert_eq!(min, 2);
                    assert_eq!(max, 4);
                }
                other => panic!("wrong refusal for N={n}: {other}"),
            }
        }
        for n in [2usize, 3, 4] {
            Tournament::new("tour-1", "run-1", "goal", criteria_of(&["a"]), n)
                .expect("in-band N accepted");
        }
    }

    #[test]
    fn identical_criteria_and_check_specs_are_byte_verified() {
        let criteria = criteria_of(&["cargo test", "no clippy warnings", "docs updated"]);
        let mut t =
            Tournament::new("tour-1", "run-1", "goal", criteria.clone(), 3).expect("tournament");
        // Every candidate's derived check set is identical byte-for-byte.
        let expected = serde_json::to_vec(&derive_check_specs(&criteria)).unwrap();
        for c in &t.candidates {
            let _ = c;
            assert_eq!(
                serde_json::to_vec(&t.check_specs()).unwrap(),
                expected,
                "derived check set must not depend on the candidate"
            );
        }
        // A settlement whose spec drifted (one byte / one order swap) is a
        // typed refusal, not a silently ranked candidate.
        let mut drifted = settlement(&t, "child-0", true);
        drifted.checks = t.check_specs();
        drifted.checks[0].spec.push('!');
        assert!(matches!(
            t.settle_candidate(drifted),
            Err(TournamentError::VerificationSpecDrift)
        ));
        let mut swapped = settlement(&t, "child-0", true);
        swapped.checks = t.check_specs();
        swapped.checks.swap(0, 1);
        assert!(matches!(
            t.settle_candidate(swapped),
            Err(TournamentError::VerificationSpecDrift)
        ));
        // Candidate items built for a fan-out must carry the identical
        // criterion bytes and the identical goal summary.
        let mut a = WorkItem::new(
            "candidate-0",
            "do the goal",
            crate::WorkKind::Implementation,
        );
        let mut b = WorkItem::new(
            "candidate-1",
            "do the goal",
            crate::WorkKind::Implementation,
        );
        let specs: Vec<String> = criteria.iter().map(|c| c.spec.clone()).collect();
        a.acceptance_checks = specs.clone();
        b.acceptance_checks = specs;
        assert_candidates_identical(&[a.clone(), b.clone()], &criteria).expect("identical");
        b.summary = "a different goal".into();
        assert!(assert_candidates_identical(&[a, b], &criteria).is_err());
    }

    #[test]
    fn deterministic_winner_verification_then_review_then_cost_then_ordinal() {
        let criteria = criteria_of(&["cargo test"]);
        // Same verification + review, different costs: the cheapest wins.
        let mut t = Tournament::new("tour-1", "run-1", "goal", criteria.clone(), 3).unwrap();
        settle_with(&mut t, "child-0", true, ReviewRank::Clean, 500);
        settle_with(&mut t, "child-1", true, ReviewRank::Clean, 100);
        settle_with(&mut t, "child-2", true, ReviewRank::Clean, 300);
        let d = t.decide().expect("decision");
        assert_eq!(d.winner.child_id, "child-1");
        assert_eq!(
            d.discarded
                .iter()
                .map(|(id, _)| id.as_str())
                .collect::<Vec<_>>(),
            vec!["child-0", "child-2"]
        );
        // Equal verification + equal review + equal cost: the lower
        // candidate ordinal wins, deterministically.
        let mut t = Tournament::new("tour-2", "run-2", "goal", criteria.clone(), 3).unwrap();
        settle_with(&mut t, "child-0", true, ReviewRank::Clean, 42);
        settle_with(&mut t, "child-1", true, ReviewRank::Clean, 42);
        settle_with(&mut t, "child-2", true, ReviewRank::Clean, 42);
        let d = t.decide().expect("decision");
        assert_eq!(d.winner.child_id, "child-0");
        // A higher review rank outranks a cheaper candidate.
        let mut t = Tournament::new("tour-3", "run-3", "goal", criteria, 2).unwrap();
        settle_with(&mut t, "child-0", true, ReviewRank::Concern, 1);
        settle_with(&mut t, "child-1", true, ReviewRank::Clean, 999);
        let d = t.decide().expect("decision");
        assert_eq!(d.winner.child_id, "child-1");
    }

    #[test]
    fn failing_verification_cannot_win_even_if_cheapest() {
        let criteria = criteria_of(&["cargo test"]);
        let mut t = Tournament::new("tour-1", "run-1", "goal", criteria, 2).unwrap();
        // child-0 is free but failed verification; child-1 passed and is
        // expensive: leaf node must be child-1.
        settle_with(&mut t, "child-0", false, ReviewRank::Clean, 0);
        settle_with(&mut t, "child-1", true, ReviewRank::Clean, 1_000_000);
        let d = t.decide().expect("decision");
        assert_eq!(d.winner.child_id, "child-1");
        assert!(d
            .discarded
            .iter()
            .any(|(id, why)| id == "child-0" && why.contains("verification failed")));
    }

    #[test]
    fn review_must_be_independent() {
        let criteria = criteria_of(&["cargo test"]);
        let mut t = Tournament::new("tour-1", "run-1", "goal", criteria, 2).unwrap();
        let mut s = settlement(&t, "child-0", true);
        // A candidate reviewing itself (or a peer candidate) is refused.
        s.review = Some(ReviewVerdict {
            rank: ReviewRank::Clean,
            reviewer: "child-0".into(),
        });
        assert!(matches!(
            t.settle_candidate(s),
            Err(TournamentError::ReviewNotIndependent(_))
        ));
        let mut s = settlement(&t, "child-0", true);
        s.review = Some(ReviewVerdict {
            rank: ReviewRank::Clean,
            reviewer: "child-1".into(),
        });
        assert!(matches!(
            t.settle_candidate(s),
            Err(TournamentError::ReviewNotIndependent(_))
        ));
        // An external reviewer is accepted.
        let s = settlement(&t, "child-0", true);
        t.settle_candidate(s).expect("external review accepted");
    }

    #[test]
    fn decide_without_eligible_candidate_is_refused() {
        let criteria = criteria_of(&["cargo test"]);
        let mut t = Tournament::new("tour-1", "run-1", "goal", criteria, 2).unwrap();
        t.settle_candidate(CandidateSettlement {
            review: None,
            ..settlement(&t, "child-0", true)
        })
        .unwrap();
        t.settle_candidate(CandidateSettlement {
            review: None,
            ..settlement(&t, "child-1", true)
        })
        .unwrap();
        assert!(matches!(
            t.decide(),
            Err(TournamentError::NoEligibleWinner(_))
        ));
        // An unprepared tournament is refused too.
        let mut fresh = Tournament::new("tour-2", "run-2", "goal", criteria_of(&["x"]), 2).unwrap();
        assert!(matches!(
            fresh.decide(),
            Err(TournamentError::NoEligibleWinner(_))
        ));
        assert_eq!(fresh.state, TournamentState::Open);
    }

    #[test]
    fn abort_discards_every_candidate() {
        let criteria = criteria_of(&["cargo test"]);
        let mut t = Tournament::new("tour-1", "run-1", "goal", criteria, 3).unwrap();
        settle_with(&mut t, "child-0", true, ReviewRank::Clean, 1);
        t.abort("operator cancelled the tournament").unwrap();
        assert_eq!(t.state, TournamentState::Aborted);
        assert!(t.winner.is_none());
        for c in &t.candidates {
            assert_eq!(c.state, CandidateState::Discarded);
        }
        // A terminal tournament is never re-aborted or decided.
        assert!(matches!(t.abort("again"), Err(TournamentError::NotOpen(_))));
        assert!(matches!(t.decide(), Err(TournamentError::NotOpen(_))));
    }

    #[test]
    fn duplicate_settlement_is_refused() {
        let criteria = criteria_of(&["cargo test"]);
        let mut t = Tournament::new("tour-1", "run-1", "goal", criteria, 2).unwrap();
        let s = settlement(&t, "child-0", true);
        t.settle_candidate(s.clone()).unwrap();
        assert!(matches!(
            t.settle_candidate(s),
            Err(TournamentError::DuplicateSettlement(_))
        ));
        assert!(matches!(
            t.settle_candidate(settlement(&t, "child-9", true)),
            Err(TournamentError::UnknownCandidate(_))
        ));
    }

    #[test]
    fn reopen_reconstructs_decided_tournament_with_same_winner() {
        let (_dir, _manager, handle) = open_handle();
        let criteria = criteria_of(&["cargo test", "docs updated"]);
        let mut t = Tournament::new("tour-1", "run-1", "goal", criteria, 3).unwrap();
        t.persist_started(&handle).unwrap();
        settle_with(&mut t, "child-0", true, ReviewRank::Clean, 900);
        settle_with(&mut t, "child-1", true, ReviewRank::Clean, 100);
        settle_with(&mut t, "child-2", true, ReviewRank::Clean, 100);
        for id in ["child-0", "child-1", "child-2"] {
            let s = settlement(&t, id, true);
            // Re-derive the settlement exactly as `settle_with` did.
            let mut s = s;
            s.review = Some(ReviewVerdict {
                rank: ReviewRank::Clean,
                reviewer: "review-0".into(),
            });
            s.cost_micro = match id {
                "child-0" => 900,
                _ => 100,
            };
            t.persist_settlement(&handle, &s).unwrap();
        }
        let d = t.decide().unwrap();
        assert_eq!(d.winner.child_id, "child-1");
        t.persist_decision(
            &handle,
            faktor_session::TOURNAMENT_OUTCOME_DECIDED,
            &d.rationale,
        )
        .unwrap();
        // Compact the ledger: tournament rows are pinned, so the reopen
        // below must still see every row.
        let report = handle.compact_typed_ledger().unwrap();
        assert!(report.entries_before >= 5);
        let reopened = Tournament::load(&handle, "tour-1").unwrap();
        assert_eq!(reopened.state, TournamentState::Decided);
        assert_eq!(reopened.winner.as_deref(), Some("child-1"));
        assert_eq!(reopened.candidates.len(), 3);
        // The listing summary folds the same durable rows and carries the
        // decision timestamp from the terminal entry.
        let summaries = Tournament::summaries(&handle).unwrap();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].id, "tour-1");
        assert_eq!(summaries[0].state, TournamentState::Decided);
        assert_eq!(summaries[0].candidate_count, 3);
        assert_eq!(summaries[0].winner.as_deref(), Some("child-1"));
        assert!(summaries[0].decided_ms.is_some());
        let settled = reopened
            .candidates
            .iter()
            .find(|c| c.child_id == "child-1")
            .unwrap();
        assert_eq!(settled.verification_pass, Some(true));
        assert_eq!(settled.cost_micro, 100);
        assert_eq!(
            settled.review.as_ref().map(|r| r.rank),
            Some(ReviewRank::Clean)
        );
        // The loser rows carry the discard mark on reopen.
        assert_eq!(
            reopened
                .candidates
                .iter()
                .find(|c| c.child_id == "child-0")
                .unwrap()
                .state,
            CandidateState::Discarded
        );
    }

    #[test]
    fn reopen_rejects_settlement_before_start() {
        let (_dir, _manager, handle) = open_handle();
        let criteria = criteria_of(&["cargo test"]);
        let t = Tournament::new("tour-orphan", "run-orphan", "goal", criteria, 2).unwrap();
        let s = settlement(&t, "child-0", true);
        handle
            .ledger_candidate_settled("tour-orphan", &s.to_row())
            .unwrap();
        assert!(matches!(
            Tournament::load(&handle, "tour-orphan"),
            Err(TournamentError::Corrupt { .. })
        ));
    }
}
