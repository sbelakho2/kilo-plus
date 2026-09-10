//! The first-class durable Task object (audit 25) + the locked-down task
//! state machine (audit P0-7) + the first-class VerificationRecord (audit
//! P0-8).
//!
//! A `Task` is the durable object a session works on: a goal, the
//! acceptance criteria derived from that goal (goal + project checks,
//! seeded once), an append-only ordered step plan, and a durable budget
//! envelope (`max_tokens`/`max_turns` vs crash-safe `spent_tokens`/
//! `spent_turns`). It lives in typed store rows (`task`, schema v10) that
//! survive IDE close, daemon restart, OS restart, provider switch and
//! context compaction — compaction never rewrites them and they are never
//! FIFO-evicted.
//!
//! # Revisions (schema v14)
//!
//! Every effective state/criteria/plan/budget mutation bumps the row's
//! monotonic `revision` exactly once, in the same transaction as the
//! mutation. The revision is the row's optimistic-lock token: the
//! transition and completion APIs take the caller's `expected_revision`
//! and refuse with a typed error when the row moved on.
//!
//! # State machine enforcement (audit P0-7)
//!
//! `VerifiedComplete` (and the states that lead to it, `NeedsVerification`
//! and `Verifying`) can NEVER be assigned through a generic patch:
//! [`SessionHandle::update_task`] rejects any patch carrying a
//! completion-relevant state with [`TaskError::CompletionStateViaPatch`]
//! and any patch that would jump a non-completion machine edge with
//! [`TaskError::IllegalTransition`]. The legal producers are exactly:
//!
//! - [`SessionHandle::transition_task`] — one named [`TaskTransition`]
//!   edge; `VerifiedComplete` has no edge and is unreachable here;
//! - [`SessionHandle::complete_verified_task`] — the ONLY path to
//!   `VerifiedComplete`, validated in ONE store transaction against a
//!   passing [`VerificationRecord`] that certifies the task's current
//!   revision and covers every current acceptance criterion.
//!
//! The store backstops the same rule at the row level: a raw
//! `task` upsert that would move a row INTO a completion-relevant state is
//! refused unless the machine allows the edge (or the row already holds
//! that state), so no generic write — API or raw — can mint completion.
//!
//! Bounds are enforced HERE, before any write, and oversized input is
//! REJECTED with an error — never silently truncated: goal <=
//! [`MAX_TASK_GOAL_BYTES`] bytes, criteria <= [`MAX_TASK_CRITERIA`] entries
//! (each <= [`MAX_TASK_CRITERION_BYTES`]), plan <= [`MAX_TASK_PLAN_STEPS`]
//! steps (each <= [`MAX_TASK_STEP_BYTES`]). A patch with a `None` field
//! keeps the row's current value, so updates are read-modify-write safe
//! under the session command lock.
//!
//! # Error typing
//!
//! The task operations that can fail with machine/proof rejections return
//! the crate's typed [`TaskError`] (an audit requirement: distinct typed
//! causes, never prose-only). The legacy `create_task` keeps returning
//! [`faktor_core::Error`] because the agent runtime performs a direct
//! `return` of its mapped result; its rejections carry stable
//! `create_task refused: ...` messages. Every `TaskError` converts into
//! [`faktor_core::Error`], so `?` interop with core-`Result` callers is
//! unchanged.

use faktor_core::id::{
    SessionId, TaskId, TaskRevision, VerificationRecordId, WorkspaceId, WorktreeId,
};
use faktor_core::state::{
    CheckExecution, CriterionOrigin, CriterionRequirement, CriterionVerification,
    FileStateEvidence, TaskState, TaskTransition, VerificationStatus,
};

use crate::handle::SessionHandle;
use crate::SessionError;

/// Hard bound on one task goal (UTF-8 bytes).
pub const MAX_TASK_GOAL_BYTES: usize = 16 * 1024;
/// Hard bound on the acceptance-criteria entry count.
pub const MAX_TASK_CRITERIA: usize = 32;
/// Hard bound on ONE criterion (mirrors the memory-fact value cap).
pub const MAX_TASK_CRITERION_BYTES: usize = 3000;
/// Hard bound on the append-only plan's step count.
pub const MAX_TASK_PLAN_STEPS: usize = 256;
/// Hard bound on ONE plan step.
pub const MAX_TASK_STEP_BYTES: usize = 3000;

// ------------------------------------------------------ record bounds (P0-8)

/// Hard bound on the number of criterion verdicts in one record.
pub const MAX_VERIFICATION_RECORD_CRITERIA: usize = 64;
/// Serialized bound of the record's criteria JSON (128 KiB).
pub const MAX_VERIFICATION_CRITERIA_JSON_BYTES: usize = 128 * 1024;
/// Hard bound on the executed checks in one record.
pub const MAX_VERIFICATION_RECORD_CHECKS: usize = 256;
/// Serialized bound of the record's checks JSON (256 KiB).
pub const MAX_VERIFICATION_CHECKS_JSON_BYTES: usize = 256 * 1024;
/// Hard bound on the changed-file evidence entries in one record.
pub const MAX_VERIFICATION_CHANGED_FILES: usize = 4096;
/// Serialized bound of the record's changed-files JSON.
pub const MAX_VERIFICATION_CHANGED_FILES_JSON_BYTES: usize = 128 * 1024;
/// Hard bound on the unrelated-change path entries in one record.
pub const MAX_VERIFICATION_UNRELATED_CHANGES: usize = 4096;
/// Serialized bound of the record's unrelated-changes JSON.
pub const MAX_VERIFICATION_UNRELATED_JSON_BYTES: usize = 128 * 1024;
/// Serialized bound of the opaque reviewer JSON.
pub const MAX_VERIFICATION_REVIEWER_JSON_BYTES: usize = 16 * 1024;
/// Bound on the tree-hash hex text.
pub const MAX_VERIFICATION_TREE_HASH_BYTES: usize = 128;
/// Bound on one criterion key (equal texts live on the task row at
/// <= MAX_TASK_CRITERION_BYTES; the bound must not be tighter than that).
pub const MAX_VERIFICATION_CRITERION_KEY_BYTES: usize = MAX_TASK_CRITERION_BYTES;
/// Bound on one criterion's evidence prose.
pub const MAX_VERIFICATION_EVIDENCE_BYTES: usize = 4096;
/// Bound on one check name.
pub const MAX_VERIFICATION_CHECK_NAME_BYTES: usize = 2048;
/// Bound on one check program.
pub const MAX_VERIFICATION_PROGRAM_BYTES: usize = 4096;
/// Bound on the per-argument count of one check.
pub const MAX_VERIFICATION_CHECK_ARGS: usize = 32;
/// Bound on ONE check argument.
pub const MAX_VERIFICATION_CHECK_ARG_BYTES: usize = 1024;
/// Bound on one check category.
pub const MAX_VERIFICATION_CATEGORY_BYTES: usize = 128;
/// Bound on one check summary prose.
pub const MAX_VERIFICATION_SUMMARY_BYTES: usize = 8192;
/// Bound on one file path (mirrors the message-payload path bound).
pub const MAX_VERIFICATION_PATH_BYTES: usize = 4096;
/// Bound on one file digest hex text.
pub const MAX_VERIFICATION_DIGEST_BYTES: usize = 128;

/// The in-band marker of a V2 typed-criterion entry in the existing criteria
/// row values (task row `acceptance_criteria` strings). Legacy plain-text
/// entries carry no marker and keep working: they are migrated
/// deterministically on read (see [`Criterion::decode`]).
const CRITERION_V2_PREFIX: &str = "v2:";
/// The V2 envelope version (a different version is a legacy/foreign entry,
/// never a silently re-interpreted criterion).
const CRITERION_V2_VERSION: u8 = 2;

/// Hard bound on the human TEXT of one typed criterion. The V2 JSON
/// envelope must still fit the existing per-entry value bound
/// ([`MAX_TASK_CRITERION_BYTES`]), and it must stay a legal verification
/// criterion key ([`MAX_VERIFICATION_CRITERION_KEY_BYTES`]), so the text cap
/// reserves room for the encoding.
pub const MAX_TASK_CRITERION_TEXT_BYTES: usize = MAX_TASK_CRITERION_BYTES - 512;
/// Hard bound on a derived criterion's source snapshot id.
pub const MAX_CRITERION_SNAPSHOT_BYTES: usize = 256;

/// The opaque, deterministic content id of one acceptance criterion.
///
/// The id is derived from the criterion's identity fields (`origin`,
/// `requirement`, `text`, `semantic_snapshot`) with a stable FNV-1a 64
/// content hash — `faktor-session` has no blake3 dependency and the id must
/// be reproducible across restarts, re-derivations and legacy migrations
/// without a durable counter. Zero is folded to 1 so the id is never 0.
/// A 64-bit hash collision is handled structurally: a criteria set carrying
/// two equal ids is rejected loudly by [`validate_criteria`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct CriterionId(u64);

impl CriterionId {
    /// Build an id from a raw value (0 is rejected by contract).
    pub const fn new(raw: u64) -> Self {
        assert!(raw != 0, "CriterionId cannot be 0");
        Self(raw)
    }

    /// The raw id value.
    pub const fn raw(self) -> u64 {
        self.0
    }

    /// The deterministic content id of a criterion identity. FNV-1a 64 over
    /// the four identity fields (NUL-separated); deterministic across
    /// process restarts, insertion orders and legacy migration.
    pub fn for_content(
        origin: CriterionOrigin,
        requirement: CriterionRequirement,
        text: &str,
        semantic_snapshot: Option<&str>,
    ) -> Self {
        const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut hash = OFFSET;
        let mut feed = |bytes: &[u8]| {
            for b in bytes {
                hash ^= u64::from(*b);
                hash = hash.wrapping_mul(PRIME);
            }
            hash ^= 0x1f;
            hash = hash.wrapping_mul(PRIME);
        };
        feed(origin.label().as_bytes());
        feed(requirement.label().as_bytes());
        feed(text.as_bytes());
        feed(semantic_snapshot.unwrap_or("").as_bytes());
        if hash == 0 {
            hash = 1;
        }
        Self(hash)
    }
}

impl std::fmt::Display for CriterionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<CriterionId> for u64 {
    fn from(v: CriterionId) -> u64 {
        v.0
    }
}

impl serde::Serialize for CriterionId {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(self.0)
    }
}

impl<'de> serde::Deserialize<'de> for CriterionId {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let raw = u64::deserialize(d)?;
        if raw == 0 {
            Err(serde::de::Error::custom("CriterionId cannot be 0"))
        } else {
            Ok(Self(raw))
        }
    }
}

/// One typed acceptance criterion (audits 56/57/105): exactly the
/// `Criterion{id, text, origin, requirement, evidence_source,
/// semantic_snapshot}` shape. Criteria are persisted through the EXISTING
/// criteria row values (a V2 JSON envelope inside each
/// `acceptance_criteria` string); no store schema change.
///
/// `evidence_source` is the durable raw `EvidenceId` (crates/evidence) of
/// the evidence that certifies the criterion; `semantic_snapshot` is the
/// provider snapshot id the criterion was derived from (derived criteria
/// are tied to their source snapshot — a stale snapshot forces
/// re-derivation).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Criterion {
    pub id: CriterionId,
    pub text: String,
    pub origin: CriterionOrigin,
    pub requirement: CriterionRequirement,
    pub evidence_source: Option<u64>,
    pub semantic_snapshot: Option<String>,
}

/// The on-disk V2 envelope (private: the in-band representation is an
/// implementation detail; absent optional fields are omitted so the encoding
/// is compact and byte-deterministic).
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CriterionEnvelope {
    v: u8,
    id: CriterionId,
    text: String,
    origin: CriterionOrigin,
    requirement: CriterionRequirement,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    evidence_source: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    semantic_snapshot: Option<String>,
}

impl Criterion {
    /// A user criterion (sticky origin, `Required`, no evidence/snapshot).
    pub fn user(text: impl Into<String>) -> Self {
        let text = text.into();
        let id = CriterionId::for_content(
            CriterionOrigin::User,
            CriterionRequirement::Required,
            &text,
            None,
        );
        Self {
            id,
            text,
            origin: CriterionOrigin::User,
            requirement: CriterionRequirement::Required,
            evidence_source: None,
            semantic_snapshot: None,
        }
    }

    /// A derived criterion tied to its source (non-user origin; the id is
    /// content-addressed over origin/requirement/text/snapshot).
    pub fn derived(
        text: impl Into<String>,
        origin: CriterionOrigin,
        requirement: CriterionRequirement,
        semantic_snapshot: Option<String>,
    ) -> Self {
        let text = text.into();
        let id = CriterionId::for_content(origin, requirement, &text, semantic_snapshot.as_deref());
        Self {
            id,
            text,
            origin,
            requirement,
            evidence_source: None,
            semantic_snapshot,
        }
    }

    /// Re-bind the criterion's evidence source (the criterion's id does not
    /// change: evidence is a verification binding, not identity).
    pub fn with_evidence(mut self, evidence_source: u64) -> Self {
        self.evidence_source = Some(evidence_source);
        self
    }

    /// Structural validation: bounded text/snapshot and an id that IS the
    /// deterministic content id (a hostile hand-crafted id can never pass).
    pub fn validate(&self) -> Result<(), TaskError> {
        if self.text.len() > MAX_TASK_CRITERION_TEXT_BYTES {
            return Err(TaskError::Oversized(format!(
                "criterion text of {} bytes exceeds MAX_TASK_CRITERION_TEXT_BYTES ({MAX_TASK_CRITERION_TEXT_BYTES})",
                self.text.len()
            )));
        }
        if let Some(snapshot) = &self.semantic_snapshot {
            if snapshot.len() > MAX_CRITERION_SNAPSHOT_BYTES {
                return Err(TaskError::Oversized(format!(
                    "criterion snapshot of {} bytes exceeds MAX_CRITERION_SNAPSHOT_BYTES ({MAX_CRITERION_SNAPSHOT_BYTES})",
                    snapshot.len()
                )));
            }
        }
        let expected = CriterionId::for_content(
            self.origin,
            self.requirement,
            &self.text,
            self.semantic_snapshot.as_deref(),
        );
        if self.id != expected {
            return Err(TaskError::Malformed(format!(
                "criterion id {} is not the deterministic content id {expected} of origin={} requirement={} snapshot={:?}",
                self.id, self.origin, self.requirement, self.semantic_snapshot
            )));
        }
        // A text within the text bound can still encode beyond the per-entry
        // bound (escape-heavy content): reject loudly here rather than let
        // `encoded_entry` silently demote the criterion to plain text and
        // drop its typed metadata on write.
        let encoded_len = self.encode().len();
        if encoded_len > MAX_TASK_CRITERION_BYTES {
            return Err(TaskError::Oversized(format!(
                "criterion {} encodes to {encoded_len} bytes, beyond MAX_TASK_CRITERION_BYTES ({MAX_TASK_CRITERION_BYTES})",
                self.id
            )));
        }
        Ok(())
    }

    /// The V2 in-band encoding (`v2:` + compact JSON).
    pub fn encode(&self) -> String {
        let envelope = CriterionEnvelope {
            v: CRITERION_V2_VERSION,
            id: self.id,
            text: self.text.clone(),
            origin: self.origin,
            requirement: self.requirement,
            evidence_source: self.evidence_source,
            semantic_snapshot: self.semantic_snapshot.clone(),
        };
        format!(
            "{CRITERION_V2_PREFIX}{}",
            serde_json::to_string(&envelope).unwrap_or_default()
        )
    }

    /// The entry to persist for this criterion: the V2 encoding when it fits
    /// the existing per-entry bound, otherwise the plain text (a legacy
    /// over-bound entry stays lossless and un-typed — never truncated).
    pub fn encoded_entry(&self) -> String {
        let encoded = self.encode();
        if encoded.len() <= MAX_TASK_CRITERION_BYTES {
            encoded
        } else {
            self.text.clone()
        }
    }

    /// Decode one persisted entry. `None` means "legacy/foreign plain text"
    /// (no marker, wrong version, or malformed JSON) — never a guessed
    /// criterion.
    pub fn decode(entry: &str) -> Option<Self> {
        let json = entry.strip_prefix(CRITERION_V2_PREFIX)?;
        let envelope: CriterionEnvelope = serde_json::from_str(json).ok()?;
        if envelope.v != CRITERION_V2_VERSION {
            return None;
        }
        Some(Self {
            id: envelope.id,
            text: envelope.text,
            origin: envelope.origin,
            requirement: envelope.requirement,
            evidence_source: envelope.evidence_source,
            semantic_snapshot: envelope.semantic_snapshot,
        })
    }

    /// Migrate one legacy plain-text criterion deterministically. The legacy
    /// writer was always the system derivation, so only the canonical goal
    /// prefix is a sticky user criterion (`goal: ` -> User); any other
    /// legacy text migrates as a replaceable policy derivation
    /// (`ProjectPolicy`). Genuinely user-authored criteria survive
    /// re-derivation by being written through the typed API with
    /// [`CriterionOrigin::User`]. The id is the same content id used for
    /// typed criteria, so the migration is stable across restarts and
    /// repeated reads.
    pub fn legacy(entry: &str) -> Self {
        let origin = if entry.starts_with("goal: ") {
            CriterionOrigin::User
        } else {
            CriterionOrigin::ProjectPolicy
        };
        Self::derived(
            entry.to_string(),
            origin,
            CriterionRequirement::Required,
            None,
        )
    }

    /// The human text of one persisted entry (typed entries decode to their
    /// text; legacy entries are already text). Read-only helper for
    /// consumers that must not see the encoding envelope.
    pub fn text_of(entry: &str) -> String {
        Self::decode(entry)
            .map(|c| c.text)
            .unwrap_or_else(|| entry.to_string())
    }
}

/// Decode a full criteria row: typed V2 entries decode; every other entry
/// migrates deterministically through [`Criterion::legacy`]. Deterministic:
/// repeated reads of the same row yield identical ids.
pub fn decode_criteria(entries: &[String]) -> Vec<Criterion> {
    entries
        .iter()
        .map(|entry| Criterion::decode(entry).unwrap_or_else(|| Criterion::legacy(entry)))
        .collect()
}

/// Encode a full criteria row into the existing per-entry values.
pub fn encode_criteria(criteria: &[Criterion]) -> Vec<String> {
    criteria.iter().map(Criterion::encoded_entry).collect()
}

/// Re-derive a criteria set (audits 56/57/105), deterministically:
///
/// - every existing USER criterion survives verbatim (a re-derivation may
///   never remove a user criterion);
/// - the `derived` set is authoritative for every non-user origin: a
///   derived criterion whose source snapshot moved is replaced by the newly
///   derived one (its content-addressed id changes with the snapshot, so the
///   stale criterion is re-derived, never silently kept);
/// - existing non-user criteria absent from `derived` are dropped
///   (superseded);
/// - identical content is deduplicated by id (and a derived criterion whose
///   text duplicates a user criterion is skipped: the user criterion wins).
pub fn merge_derived_criteria(existing: &[Criterion], derived: &[Criterion]) -> Vec<Criterion> {
    let mut out: Vec<Criterion> = Vec::new();
    for criterion in existing.iter().filter(|c| c.origin.is_user()) {
        if !out.iter().any(|c| c.id == criterion.id) {
            out.push(criterion.clone());
        }
    }
    for criterion in derived {
        if out.iter().any(|c| c.id == criterion.id) {
            continue;
        }
        if out.iter().any(|c| c.text == criterion.text) {
            continue;
        }
        out.push(criterion.clone());
    }
    out
}

/// Validate a whole criteria set: bounds, content ids and id uniqueness.
fn validate_criteria(criteria: &[Criterion]) -> Result<(), TaskError> {
    if criteria.len() > MAX_TASK_CRITERIA {
        return Err(TaskError::Oversized(format!(
            "{} acceptance criteria exceed MAX_TASK_CRITERIA ({MAX_TASK_CRITERIA})",
            criteria.len()
        )));
    }
    let mut ids = std::collections::HashSet::new();
    for criterion in criteria {
        criterion.validate()?;
        if !ids.insert(criterion.id) {
            return Err(TaskError::Malformed(format!(
                "duplicate criterion id {} (content hash collision or a hostile id): the criteria set is ambiguous",
                criterion.id
            )));
        }
    }
    Ok(())
}

/// The durable budget envelope of a Task. `None` max fields mean unlimited;
/// `spent_*` fields grow monotonically from durable sources (provider-call
/// rows + `turn_completed` journal events), so spend survives crashes.
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct TaskBudget {
    pub max_tokens: Option<u64>,
    pub max_turns: Option<u32>,
    pub spent_tokens: u64,
    pub spent_turns: u32,
}

/// The durable Task object (audit 25). One row per `(session_id, task_id)`;
/// `task_id` is the session's adopted durable task identity.
///
/// The row's `revision` counter is NOT a field of this struct (the agent
/// runtime constructs `Task` literals by field, and adding a field would
/// break every literal site in a crate this wave may not touch); it rides
/// the store row and is read through
/// [`SessionHandle::task_revision`].
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Task {
    pub task_id: TaskId,
    pub session_id: SessionId,
    pub goal: String,
    /// Goal + project-derived required checks, seeded once when first seen.
    pub acceptance_criteria: Vec<String>,
    /// Ordered steps; append-only durable.
    pub plan: Vec<String>,
    pub budget: TaskBudget,
    pub state: TaskState,
    pub created_ms: i64,
    pub updated_ms: i64,
}

impl Default for Task {
    fn default() -> Self {
        Self {
            task_id: TaskId::new(1),
            session_id: SessionId::new(1),
            goal: String::new(),
            acceptance_criteria: Vec::new(),
            plan: Vec::new(),
            budget: TaskBudget::default(),
            state: TaskState::Pending,
            created_ms: 0,
            updated_ms: 0,
        }
    }
}

impl From<faktor_store::TaskRow> for Task {
    fn from(r: faktor_store::TaskRow) -> Self {
        Self {
            task_id: r.task_id,
            session_id: r.session_id,
            goal: r.goal,
            acceptance_criteria: r.acceptance_criteria,
            plan: r.plan,
            budget: TaskBudget {
                max_tokens: r.max_tokens,
                max_turns: r.max_turns,
                spent_tokens: r.spent_tokens,
                spent_turns: r.spent_turns,
            },
            state: r.state,
            created_ms: r.created_ms,
            updated_ms: r.updated_ms,
        }
    }
}

/// Build a store row for a caller-constructed revision (private: the
/// revision is never derived from a `Task`, which deliberately does not
/// carry one).
fn task_row(task: Task, revision: TaskRevision) -> faktor_store::TaskRow {
    faktor_store::TaskRow {
        task_id: task.task_id,
        session_id: task.session_id,
        goal: task.goal,
        acceptance_criteria: task.acceptance_criteria,
        plan: task.plan,
        max_tokens: task.budget.max_tokens,
        max_turns: task.budget.max_turns,
        spent_tokens: task.budget.spent_tokens,
        spent_turns: task.budget.spent_turns,
        state: task.state,
        revision,
        created_ms: task.created_ms,
        updated_ms: task.updated_ms,
    }
}

impl Task {
    /// The typed criteria view of this row (audits 56/57): V2 entries decode
    /// to their criterion; legacy plain-text entries deterministically
    /// migrate (stable content ids, inferred origin). Repeated reads of the
    /// same row always yield identical ids.
    pub fn criteria(&self) -> Vec<Criterion> {
        decode_criteria(&self.acceptance_criteria)
    }
}

/// The session-facing view of one durable verification record (audit P0-8).
/// Records are immutable after creation except the single CAS finalize
/// `Running -> Passed|Failed` ([`SessionHandle::finalize_verification_record`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationRecord {
    pub record_id: VerificationRecordId,
    pub task_id: TaskId,
    /// The task revision this record certifies (its creation-time task
    /// revision). Completion requires the task to STILL be at this revision.
    pub revision: TaskRevision,
    pub workspace_id: WorkspaceId,
    pub worktree_id: WorktreeId,
    pub tree_hash: Option<String>,
    pub criteria: Vec<CriterionVerification>,
    pub checks: Vec<CheckExecution>,
    pub changed_files: Vec<FileStateEvidence>,
    pub unrelated_changes: Vec<String>,
    pub reviewer: Option<serde_json::Value>,
    pub status: VerificationStatus,
    pub started_ms: i64,
    pub completed_ms: Option<i64>,
}

impl From<faktor_store::VerificationRecordRow> for VerificationRecord {
    fn from(r: faktor_store::VerificationRecordRow) -> Self {
        Self {
            record_id: r.id,
            task_id: r.task_id,
            revision: r.revision,
            workspace_id: r.workspace_id,
            worktree_id: r.worktree_id,
            tree_hash: r.tree_hash,
            criteria: r.criteria,
            checks: r.checks,
            changed_files: r.changed_files,
            unrelated_changes: r.unrelated_changes,
            reviewer: r.reviewer,
            status: r.status,
            started_ms: r.started_ms,
            completed_ms: r.completed_ms,
        }
    }
}

/// Typed failure of the task machine / completion-proof operations
/// (audits P0-7/P0-8). Every rejection cause is its own variant, so
/// callers distinguish a missing record from a wrong-revision record from
/// an uncovered criterion without parsing prose. Conversions into
/// [`faktor_core::Error`] keep the crate's `?` interop with core-`Result`
/// callers (the agent runtime) intact.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TaskError {
    #[error("task {0} not found")]
    NotFound(TaskId),
    #[error("task {task_id} already exists: a task row is created once and mutated through update_task/transition_task/complete_verified_task, never recreated")]
    AlreadyExists { task_id: TaskId },
    #[error("create_task refused: state {state:?} cannot seed a task; only Pending, Planning or Running may (VerifiedComplete requires a passing verification record via complete_verified_task)")]
    IllegalCreateState { state: TaskState },
    #[error("illegal task state transition {from:?} -> {to:?}: {detail}")]
    IllegalTransition {
        from: TaskState,
        to: TaskState,
        detail: String,
    },
    #[error("completion-relevant task state {state:?} cannot be assigned through update_task; use transition_task (NeedsVerification/Verifying) or complete_verified_task (VerifiedComplete)")]
    CompletionStateViaPatch { state: TaskState },
    #[error(
        "task {task_id} is terminal ({state:?}): its row is frozen, no further mutation is legal"
    )]
    TerminalTask { task_id: TaskId, state: TaskState },
    #[error("task {task_id} revision mismatch: expected {expected}, actual {actual} (re-read the row and retry with the current revision)")]
    RevisionMismatch {
        task_id: TaskId,
        expected: TaskRevision,
        actual: TaskRevision,
    },
    #[error("task is not Verifying (actual state {actual:?}); VerifiedComplete requires Verifying plus a passing record via complete_verified_task")]
    NotVerifying { actual: TaskState },
    #[error("verification record {0} does not exist")]
    RecordNotFound(VerificationRecordId),
    #[error(
        "verification record {record} certifies task {record_task}, not task {requested_task}"
    )]
    RecordWrongTask {
        record: VerificationRecordId,
        record_task: TaskId,
        requested_task: TaskId,
    },
    #[error("verification record {record} certifies task revision {record_revision}, but the task is at revision {expected}")]
    RecordWrongRevision {
        record: VerificationRecordId,
        record_revision: TaskRevision,
        expected: TaskRevision,
    },
    #[error(
        "verification record {record} has status {status:?}; only Passed certifies completion"
    )]
    RecordNotPassed {
        record: VerificationRecordId,
        status: VerificationStatus,
    },
    #[error("verification record {record} does not cover {missing:?} (present with passed=true is required for every current acceptance criterion)")]
    CriteriaNotCovered {
        record: VerificationRecordId,
        missing: Vec<String>,
    },
    #[error("verification record {record} was certified against worktree {record_workspace}/{record_worktree}; the task's base worktree is {task_workspace}/{task_worktree}")]
    WorktreeMismatch {
        record: VerificationRecordId,
        record_workspace: WorkspaceId,
        record_worktree: WorktreeId,
        task_workspace: WorkspaceId,
        task_worktree: WorktreeId,
    },
    #[error("verification record {record} cannot be finalized: it is {current:?}; only a Running record finalizes, exactly once")]
    RecordNotFinalizable {
        record: VerificationRecordId,
        current: VerificationStatus,
    },
    #[error(
        "completion accounting incomplete for task {task_id}: {open_count} open reservation(s) \
         ({open_micro} micro held; {dispatched_count} already dispatched) and {uncertain_count} \
         UNCERTAIN ({uncertain_micro} micro held) still consume budget; VerifiedComplete requires \
         every reservation settled, refunded or conservatively finalized — the task STAYS Verifying \
         until a later completion pass converges"
    )]
    AccountingIncomplete {
        task_id: TaskId,
        open_count: usize,
        open_micro: u64,
        dispatched_count: usize,
        uncertain_count: usize,
        uncertain_micro: u64,
    },
    #[error("completion accounting failed for task {task_id}: {detail} (nothing was transitioned; the task STAYS Verifying)")]
    AccountingFailure { task_id: TaskId, detail: String },
    #[error("the mutating run left task {task_id}'s change budget: {violations:?}")]
    ChangeBudgetRefused {
        task_id: TaskId,
        violations: Vec<crate::budget::ChangeBudgetViolation>,
    },
    #[error("input exceeds bound: {0}")]
    Oversized(String),
    #[error("malformed input: {0}")]
    Malformed(String),
    #[error("store failure: {0}")]
    Store(String),
}

impl From<faktor_store::StoreError> for TaskError {
    fn from(e: faktor_store::StoreError) -> Self {
        TaskError::Store(e.to_string())
    }
}

impl From<TaskError> for SessionError {
    fn from(e: TaskError) -> Self {
        match e {
            TaskError::NotFound(m) => SessionError::NotFound(format!("task {m}")),
            TaskError::Store(m) => SessionError::Store(faktor_store::StoreError::Conflict(m)),
            TaskError::Oversized(m) => SessionError::Oversized(m),
            TaskError::Malformed(m) => SessionError::Malformed(m),
            other => SessionError::Conflict(other.to_string()),
        }
    }
}

impl From<TaskError> for faktor_core::Error {
    fn from(e: TaskError) -> Self {
        SessionError::from(e).into()
    }
}

/// A bounded update over an existing durable Task. Every field is optional:
/// `None` keeps the row's current value, so `update_task` never clobbers a
/// field its caller did not intend to change (e.g. the runtime preserves a
/// caller-set budget when it patches the gate state).
///
/// `state` assignments are machine-checked (audit P0-7): completion-relevant
/// states (`NeedsVerification`/`Verifying`/`VerifiedComplete`) are rejected
/// with [`TaskError::CompletionStateViaPatch`], and any non-completion
/// assignment that is not a legal edge from the row's current state is
/// rejected with [`TaskError::IllegalTransition`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TaskPatch {
    pub goal: Option<String>,
    pub acceptance_criteria: Option<Vec<String>>,
    pub plan: Option<Vec<String>>,
    pub budget: Option<TaskBudget>,
    pub state: Option<TaskState>,
}

impl SessionHandle {
    /// The session's durable task identity (the adopted `task_id` on the
    /// session row; standalone sessions default to 1).
    pub fn task_id(&self) -> faktor_core::Result<TaskId> {
        Ok(self.row()?.task_id)
    }

    /// Create ONE durable task row (audit P0-7). Creation is the ONLY place
    /// a row may seed with `Pending`/`Planning`/`Running`; every other
    /// state is refused (creating a "verified" task would mint completion
    /// proof from nothing), and an existing row for `(session_id, task_id)`
    /// is refused — a task row is created once and mutated afterwards.
    /// Fresh rows start at revision 1. Oversized fields are rejected with
    /// an error BEFORE any write — never truncated silently.
    pub fn create_task(&self, task: Task) -> faktor_core::Result<Task> {
        if task.task_id.raw() == 0 || task.session_id.raw() == 0 {
            return Err(
                SessionError::Malformed("task_id and session_id must be non-zero".into()).into(),
            );
        }
        if !task.state.is_creatable() {
            return Err(faktor_core::Error::new(
                faktor_core::ErrorKind::Conflict,
                format!(
                    "create_task refused: state {:?} cannot seed a task; only Pending, Planning or Running may (VerifiedComplete requires a passing verification record via complete_verified_task)",
                    task.state
                ),
            ));
        }
        validate_task_fields(&task)?;
        if task.created_ms == 0 {
            return Err(SessionError::Malformed("created_ms must be set".into()).into());
        }
        let _guard = self.command_guard();
        let store = self.manager.store();
        if store
            .get_task(self.id, task.task_id)
            .map_err(crate::map_store_err)?
            .is_some()
        {
            return Err(faktor_core::Error::new(
                faktor_core::ErrorKind::Conflict,
                format!(
                    "create_task refused: task {} already exists for this session; a task row is created once and mutated through update_task/transition_task/complete_verified_task, never recreated",
                    task.task_id
                ),
            ));
        }
        store
            .upsert_task(&task_row(task.clone(), TaskRevision::new(1)))
            .map_err(crate::map_store_err)?;
        Ok(task)
    }

    /// Patch ONE durable task row under the task state machine (audit
    /// P0-7). Fields that validate and are present are applied; the other
    /// fields keep their current values; `created_ms` is preserved by
    /// construction.
    ///
    /// Enforcement, in order:
    /// 1. a patch whose `state` is completion-relevant
    ///    (`NeedsVerification`/`Verifying`/`VerifiedComplete`) is rejected
    ///    with [`TaskError::CompletionStateViaPatch`] — those states are
    ///    produced only by `transition_task`/`complete_verified_task`;
    /// 2. a patch whose `state` would jump an edge the machine does not
    ///    allow from the row's current state is rejected with
    ///    [`TaskError::IllegalTransition`] (self-assignment is an
    ///    idempotent no-op);
    /// 3. a patch that effectively changes the content of a TERMINAL row
    ///    (VerifiedComplete/Failed/Cancelled) is rejected with
    ///    [`TaskError::TerminalTask`].
    ///
    /// Every effective change bumps the row revision exactly once. A no-op
    /// patch (nothing actually changes — e.g. a spend heal that found
    /// nothing to heal) writes nothing and does not bump.
    pub fn update_task(&self, task_id: TaskId, patch: TaskPatch) -> Result<Task, TaskError> {
        if task_id.raw() == 0 {
            return Err(TaskError::Malformed("task_id must be non-zero".into()));
        }
        let _guard = self.command_guard();
        let store = self.manager.store();
        let row = store
            .get_task(self.id, task_id)?
            .ok_or(TaskError::NotFound(task_id))?;
        let current = Task::from(row.clone());
        let mut next = current.clone();
        if let Some(goal) = patch.goal {
            next.goal = goal;
        }
        if let Some(criteria) = patch.acceptance_criteria {
            next.acceptance_criteria = criteria;
        }
        if let Some(plan) = patch.plan {
            next.plan = plan;
        }
        if let Some(budget) = patch.budget {
            // The durable budget cap: spent counters only ever move
            // forward. A patch that would rewind spend is a corruption sign
            // (two writers cannot both gate on a rewindable counter).
            next.budget.spent_tokens = budget.spent_tokens.max(next.budget.spent_tokens);
            next.budget.spent_turns = budget.spent_turns.max(next.budget.spent_turns);
            next.budget.max_tokens = budget.max_tokens;
            next.budget.max_turns = budget.max_turns;
        }
        if let Some(state) = patch.state {
            if state.is_completion_relevant() {
                return Err(TaskError::CompletionStateViaPatch { state });
            }
            if state != row.state && !row.state.allowed_transitions().contains(&state) {
                return Err(TaskError::IllegalTransition {
                    from: row.state,
                    to: state,
                    detail: "update_task only drives ordinary machine edges; completion states (NeedsVerification/Verifying/VerifiedComplete) require transition_task/complete_verified_task".into(),
                });
            }
            next.state = state;
        }
        validate_task_fields(&next)?;
        if next == current {
            // Idempotent no-op (replay / heal): nothing to bump, no write.
            return Ok(current);
        }
        if row.state.is_terminal() {
            return Err(TaskError::TerminalTask {
                task_id,
                state: row.state,
            });
        }
        let revision = row
            .revision
            .checked_next()
            .ok_or_else(|| TaskError::Malformed("task revision overflow".into()))?;
        let mut out = task_row(next, revision);
        out.updated_ms = self.manager.now_ms();
        store.upsert_task(&out)?;
        Ok(Task::from(out))
    }

    /// Drive ONE legal machine edge (audit P0-7) — the single chokepoint
    /// for `Pending`/`Planning`/`Running`/`Waiting`/`Blocked`/`Verifying`
    /// state changes plus cancellation. `expected_revision` must equal the
    /// row's current revision (typed [`TaskError::RevisionMismatch`]
    /// otherwise); the edge must be legal from the row's current state
    /// (typed [`TaskError::IllegalTransition`]). Success bumps the revision
    /// exactly once. `proof` is refused here with a typed error: only
    /// [`SessionHandle::complete_verified_task`] consumes a verification
    /// record, and `VerifiedComplete` has no `TaskTransition` edge at all.
    pub fn transition_task(
        &self,
        task_id: TaskId,
        expected_revision: TaskRevision,
        transition: TaskTransition,
        proof: Option<VerificationRecordId>,
    ) -> Result<Task, TaskError> {
        if task_id.raw() == 0 {
            return Err(TaskError::Malformed("task_id must be non-zero".into()));
        }
        if let Some(proof) = proof {
            return Err(TaskError::Malformed(format!(
                "transition_task does not consume a proof ({proof}); a completion proof belongs to complete_verified_task, and VerifiedComplete is unreachable by transition"
            )));
        }
        let _guard = self.command_guard();
        let store = self.manager.store();
        let row = store
            .get_task(self.id, task_id)?
            .ok_or(TaskError::NotFound(task_id))?;
        if row.revision != expected_revision {
            return Err(TaskError::RevisionMismatch {
                task_id,
                expected: expected_revision,
                actual: row.revision,
            });
        }
        if !transition.legal_from(row.state) {
            return Err(TaskError::IllegalTransition {
                from: row.state,
                to: transition.to_state(),
                detail: format!(
                    "transition_task({transition:?}): the task is {:?}; re-read the row and pick a legal edge",
                    row.state
                ),
            });
        }
        let revision = expected_revision
            .checked_next()
            .ok_or_else(|| TaskError::Malformed("task revision overflow".into()))?;
        let mut out = task_row(Task::from(row), revision);
        out.state = transition.to_state();
        out.updated_ms = self.manager.now_ms();
        store.upsert_task(&out)?;
        Ok(Task::from(out))
    }

    /// Complete a verified task (audit P0-7/P0-8 + attempt-accounting
    /// extension): the ONLY path to `VerifiedComplete`. The completion runs
    /// as one ordered, crash-resumable sequence — every step is its own
    /// durable unit, so a crash at any seam leaves the task Verifying and a
    /// later completion pass converges (all monetary steps are idempotent):
    ///
    /// 1. **Proof load + validation** (read-only mirror of the store checks
    ///    (a)-(g) below — a lying proof never triggers a monetary step);
    /// 2. **Accounting-before-completion**: (a) reconcile every UNCERTAIN
    ///    provider attempt whose exact usage is known (a completed durable
    ///    provider-call row of that attempt); (b) conservatively settle any
    ///    remaining UNCERTAIN attempt AT ITS RESERVED ESTIMATE; (c) assert
    ///    ZERO open (reserved/dispatched) and ZERO UNCERTAIN reservations
    ///    remain — the final monetary spend is folded by the settlements
    ///    themselves (the task row's spent columns are written in the same
    ///    store transactions). Any failure is a typed
    ///    [`TaskError::AccountingFailure`]/[`TaskError::AccountingIncomplete`]
    ///    and the task STAYS Verifying;
    /// 3. **Transition** — the whole proof validation runs AGAIN inside the
    ///    ONE store transaction: (a) the task is `Verifying` (a
    ///    `NeedsVerification` task must transition to `Verifying` first —
    ///    completion never skips the verifier), (b) the record exists,
    ///    (c) it certifies THIS task, (d) it certifies exactly
    ///    `expected_revision` == the task's current revision, (e) its status
    ///    is `Passed`, (f) it covers every current acceptance criterion of
    ///    the task (extra record criteria are fine), (g) its
    ///    workspace/worktree equal the task's current base worktree.
    ///    Success writes `VerifiedComplete` and bumps the revision exactly
    ///    once.
    ///
    /// Invariant (locked by fault tests): a row in `VerifiedComplete` has
    /// zero open + zero uncertain reservations and its final monetary
    /// totals folded — the accounting gate ran in the same logical
    /// completion before the transition CAS.
    pub fn complete_verified_task(
        &self,
        task_id: TaskId,
        expected_revision: TaskRevision,
        proof: VerificationRecordId,
    ) -> Result<Task, TaskError> {
        self.complete_verified_task_crashable(task_id, expected_revision, proof, None)
            .map(|t| t.expect("the full completion sequence always reaches the transition"))
    }

    /// Crash-seamed twin of [`SessionHandle::complete_verified_task`]
    /// (adversarial fault tests): `crash` simulates process death at one
    /// seam of the sequence — the steps up to (not including) the seam ran
    /// and committed durably, everything after it did not. `Ok(None)` =
    /// the simulated crash point; the caller reopens the store and asserts
    /// the task still reads `Verifying` and the accounting invariant, then
    /// re-runs the FULL completion (which converges — every monetary step
    /// is idempotent). `None` runs the whole sequence.
    fn complete_verified_task_crashable(
        &self,
        task_id: TaskId,
        expected_revision: TaskRevision,
        proof: VerificationRecordId,
        crash: Option<CompletionCrashPoint>,
    ) -> Result<Option<Task>, TaskError> {
        if task_id.raw() == 0 {
            return Err(TaskError::Malformed("task_id must be non-zero".into()));
        }
        let _guard = self.command_guard();
        // ---- step 1: verification proof load + validation (read-only) ----
        self.validate_completion_proof(task_id, expected_revision, proof)?;
        if crash == Some(CompletionCrashPoint::AfterProofLoad) {
            return Ok(None);
        }
        // ---- step 2: accounting-before-completion (sync, idempotent).
        // A seam inside the pass stops the whole sequence there.
        if self.run_completion_accounting(task_id, crash)? {
            return Ok(None);
        }
        // ---- step 3: the atomic transition CAS (re-validates + writes
        // VerifiedComplete exactly once; the ONLY completion writer) ----
        if crash == Some(CompletionCrashPoint::BeforeTransition) {
            return Ok(None);
        }
        let outcome = self.manager.store().task_complete_verified(
            self.id,
            task_id,
            expected_revision,
            proof,
            self.manager.now_ms(),
        )?;
        match outcome {
            Ok(row) => Ok(Some(Task::from(row))),
            Err(refusal) => Err(match refusal {
                faktor_store::TaskCompletionRefusal::TaskMissing { .. } => {
                    TaskError::NotFound(task_id)
                }
                faktor_store::TaskCompletionRefusal::RevisionMismatch { expected, actual } => {
                    TaskError::RevisionMismatch {
                        task_id,
                        expected,
                        actual,
                    }
                }
                faktor_store::TaskCompletionRefusal::NotVerifying { actual } => {
                    TaskError::NotVerifying { actual }
                }
                faktor_store::TaskCompletionRefusal::RecordMissing { record_id } => {
                    TaskError::RecordNotFound(record_id)
                }
                faktor_store::TaskCompletionRefusal::RecordWrongTask {
                    record_id,
                    record_task,
                    ..
                } => TaskError::RecordWrongTask {
                    record: record_id,
                    record_task,
                    requested_task: task_id,
                },
                faktor_store::TaskCompletionRefusal::RecordWrongRevision {
                    record_id,
                    record_revision,
                    ..
                } => TaskError::RecordWrongRevision {
                    record: record_id,
                    record_revision,
                    expected: expected_revision,
                },
                faktor_store::TaskCompletionRefusal::RecordNotPassed { record_id, status } => {
                    TaskError::RecordNotPassed {
                        record: record_id,
                        status,
                    }
                }
                faktor_store::TaskCompletionRefusal::CriteriaNotCovered { record_id, missing } => {
                    TaskError::CriteriaNotCovered {
                        record: record_id,
                        missing,
                    }
                }
                faktor_store::TaskCompletionRefusal::WorktreeMismatch {
                    record_id,
                    record_workspace,
                    record_worktree,
                    task_workspace,
                    task_worktree,
                } => TaskError::WorktreeMismatch {
                    record: record_id,
                    record_workspace,
                    record_worktree,
                    task_workspace,
                    task_worktree,
                },
            }),
        }
    }

    /// Read-only mirror of the store completion transaction's proof checks
    /// (a)-(g) — run BEFORE any monetary step so a lying proof never moves
    /// money. The store re-validates the same checks atomically at the
    /// transition CAS; drift here only changes WHEN accounting runs, never
    /// the final gate (the store stays authoritative).
    fn validate_completion_proof(
        &self,
        task_id: TaskId,
        expected_revision: TaskRevision,
        proof: VerificationRecordId,
    ) -> Result<(), TaskError> {
        let store = self.manager.store();
        let row = store
            .get_task(self.id, task_id)?
            .ok_or(TaskError::NotFound(task_id))?;
        if row.revision != expected_revision {
            return Err(TaskError::RevisionMismatch {
                task_id,
                expected: expected_revision,
                actual: row.revision,
            });
        }
        if row.state != TaskState::Verifying {
            return Err(TaskError::NotVerifying { actual: row.state });
        }
        let rec = store
            .verification_record_get(proof)?
            .ok_or(TaskError::RecordNotFound(proof))?;
        if rec.task_id != task_id {
            return Err(TaskError::RecordWrongTask {
                record: proof,
                record_task: rec.task_id,
                requested_task: task_id,
            });
        }
        if rec.revision != expected_revision {
            return Err(TaskError::RecordWrongRevision {
                record: proof,
                record_revision: rec.revision,
                expected: expected_revision,
            });
        }
        if rec.status != VerificationStatus::Passed {
            return Err(TaskError::RecordNotPassed {
                record: proof,
                status: rec.status,
            });
        }
        let missing: Vec<String> = row
            .acceptance_criteria
            .iter()
            .filter(|c| {
                !rec.criteria
                    .iter()
                    .any(|cv| cv.passed && &cv.criterion_key == *c)
            })
            .cloned()
            .collect();
        if !missing.is_empty() {
            return Err(TaskError::CriteriaNotCovered {
                record: proof,
                missing,
            });
        }
        let base = store
            .get_session(self.id)?
            .ok_or(TaskError::NotFound(task_id))?;
        if rec.workspace_id != base.workspace_id || rec.worktree_id != base.worktree_id {
            return Err(TaskError::WorktreeMismatch {
                record: proof,
                record_workspace: rec.workspace_id,
                record_worktree: rec.worktree_id,
                task_workspace: base.workspace_id,
                task_worktree: base.worktree_id,
            });
        }
        Ok(())
    }

    /// The accounting-before-completion pass (step 2 of
    /// [`SessionHandle::complete_verified_task`]): reconcile → conservative
    /// finalize → assert ZERO. Sync over the durable ledger (each monetary
    /// step is its own store transaction; all are idempotent, so a crash
    /// anywhere and a later re-run converge). Any failure is typed and the
    /// task row is untouched (it stays Verifying).
    /// `Ok(true)` = the crash seam fired inside the pass (the caller stops
    /// the whole completion sequence there — the simulated process death).
    fn run_completion_accounting(
        &self,
        task_id: TaskId,
        crash: Option<CompletionCrashPoint>,
    ) -> Result<bool, TaskError> {
        let ledger = crate::budget::DurableBudgetLedger::new(self.manager.clone());
        let session_id = self.id;
        let account_err = |e: crate::budget::BudgetError| TaskError::AccountingFailure {
            task_id,
            detail: e.to_string(),
        };
        // (a) reconcile provider attempts whose exact usage is known: every
        // UNCERTAIN reservation whose ATTEMPT has a completed durable
        // provider-call row settles FROM that row's tokens at the
        // reservation's frozen snapshot.
        ledger
            .reconcile_uncertain_now(session_id, task_id)
            .map_err(account_err)?;
        if crash == Some(CompletionCrashPoint::AfterReconcile) {
            return Ok(true);
        }
        // (b) conservatively settle every still-UNCERTAIN attempt AT ITS
        // RESERVED ESTIMATE (the provider may have billed a dispatched
        // attempt whose actual never reconciled) — the final monetary spend
        // folds into the task row in the same transaction.
        ledger
            .finalize_uncertain_now(session_id, task_id)
            .map_err(account_err)?;
        if crash == Some(CompletionCrashPoint::AfterCostFold) {
            return Ok(true);
        }
        // (c) assert ZERO open (reserved/dispatched) and ZERO UNCERTAIN
        // reservations remain. A nonzero balance refuses completion — the
        // row never transitions and stays Verifying.
        let balance = ledger
            .completion_accounting_balance(session_id, task_id)
            .map_err(account_err)?;
        if !balance.is_zero() {
            return Err(TaskError::AccountingIncomplete {
                task_id,
                open_count: balance.open_count,
                open_micro: balance.open_micro,
                dispatched_count: balance.dispatched_count,
                uncertain_count: balance.uncertain_count,
                uncertain_micro: balance.uncertain_micro,
            });
        }
        Ok(false)
    }

    /// The row's current revision — the `expected_revision` a transition
    /// or completion must be called with.
    pub fn task_revision(&self, task_id: TaskId) -> Result<TaskRevision, TaskError> {
        let row = self
            .manager
            .store()
            .get_task(self.id, task_id)?
            .ok_or(TaskError::NotFound(task_id))?;
        Ok(row.revision)
    }

    /// The durable task row identified by `task_id` (session-scoped).
    pub fn get_task(&self, task_id: TaskId) -> faktor_core::Result<Option<Task>> {
        self.manager
            .store()
            .get_task(self.id, task_id)
            .map_err(|e| crate::map_store_err(e).into())
            .map(|r| r.map(Task::from))
    }

    /// Every durable task row of this session (oldest-created first).
    pub fn list_tasks(&self) -> faktor_core::Result<Vec<Task>> {
        self.manager
            .store()
            .list_tasks(self.id)
            .map_err(|e| crate::map_store_err(e).into())
            .map(|rows| rows.into_iter().map(Task::from).collect())
    }

    /// The typed acceptance criteria of one durable task row (audits
    /// 56/57/105). Legacy plain-text entries migrate deterministically on
    /// read (stable content ids, inferred origin) and are NEVER rewritten by
    /// this read.
    pub fn task_criteria(&self, task_id: TaskId) -> Result<Vec<Criterion>, TaskError> {
        let row = self
            .manager
            .store()
            .get_task(self.id, task_id)?
            .ok_or(TaskError::NotFound(task_id))?;
        Ok(Task::from(row).criteria())
    }

    /// Replace the task's acceptance criteria with an explicit typed set
    /// (audits 56/57/105). The set is validated (bounds, deterministic
    /// content ids, unique ids) and serialized as V2 JSON into the EXISTING
    /// criteria row values; an effective change bumps the row revision
    /// exactly once through [`SessionHandle::update_task`] — which is what
    /// invalidates any prior verification (its record pins the old revision).
    pub fn set_task_criteria(
        &self,
        task_id: TaskId,
        criteria: Vec<Criterion>,
    ) -> Result<Task, TaskError> {
        if task_id.raw() == 0 {
            return Err(TaskError::Malformed("task_id must be non-zero".into()));
        }
        validate_criteria(&criteria)?;
        self.update_task(
            task_id,
            TaskPatch {
                acceptance_criteria: Some(encode_criteria(&criteria)),
                ..Default::default()
            },
        )
    }

    /// Re-derive the task's criteria: merge the freshly derived set with the
    /// durable row under the audit-56 rules (user criteria survive
    /// verbatim; derived criteria are authoritative for their origin;
    /// snapshot-stale derived criteria are re-derived), then persist through
    /// [`SessionHandle::set_task_criteria`] — one revision bump on any
    /// effective change.
    pub fn rederive_task_criteria(
        &self,
        task_id: TaskId,
        derived: Vec<Criterion>,
    ) -> Result<Task, TaskError> {
        if task_id.raw() == 0 {
            return Err(TaskError::Malformed("task_id must be non-zero".into()));
        }
        validate_criteria(&derived)?;
        let row = self
            .manager
            .store()
            .get_task(self.id, task_id)?
            .ok_or(TaskError::NotFound(task_id))?;
        let merged = merge_derived_criteria(&Task::from(row).criteria(), &derived);
        self.set_task_criteria(task_id, merged)
    }

    /// Crash-safe token spend of the session: the durable sum of every
    /// recorded provider call (input + output tokens).
    pub fn spent_tokens(&self) -> faktor_core::Result<u64> {
        self.manager
            .store()
            .session_usage_tokens(self.id)
            .map_err(|e| crate::map_store_err(e).into())
    }

    /// Crash-safe logical-turn count of the session: durable
    /// `turn_completed` journal events.
    pub fn spent_turns(&self) -> faktor_core::Result<u64> {
        self.manager
            .store()
            .turn_completed_count(self.id)
            .map_err(|e| crate::map_store_err(e).into())
    }

    /// The configured wall-clock budget of one logical turn (ms; 0 =
    /// unbounded). The runtime caps every turn slice with this value.
    pub fn turn_budget_ms(&self) -> u64 {
        self.manager.turn_budget_ms()
    }

    // -------------------------------------------------- verification records

    /// Create one durable verification record certifying `task_id` at its
    /// CURRENT revision (audit P0-8). The record is written once and is
    /// immutable afterwards except the single CAS finalize
    /// (`Running -> Passed|Failed`); a record created directly as `Passed`
    /// is complete from birth. Every bound is enforced BEFORE the write —
    /// oversized criteria/checks/evidence JSON is rejected with a typed
    /// [`TaskError::Oversized`], never truncated.
    #[allow(clippy::too_many_arguments)]
    pub fn create_verification_record(
        &self,
        task_id: TaskId,
        tree_hash: Option<String>,
        criteria: Vec<CriterionVerification>,
        checks: Vec<CheckExecution>,
        changed_files: Vec<FileStateEvidence>,
        unrelated_changes: Vec<String>,
        reviewer: Option<serde_json::Value>,
        status: VerificationStatus,
        started_ms: i64,
    ) -> Result<VerificationRecordId, TaskError> {
        if task_id.raw() == 0 {
            return Err(TaskError::Malformed("task_id must be non-zero".into()));
        }
        validate_verification_record(
            &criteria,
            &checks,
            &changed_files,
            &unrelated_changes,
            reviewer.as_ref(),
            tree_hash.as_deref(),
        )?;
        let _guard = self.command_guard();
        let store = self.manager.store();
        let task = store
            .get_task(self.id, task_id)?
            .ok_or(TaskError::NotFound(task_id))?;
        let session = store
            .get_session(self.id)?
            .ok_or(TaskError::NotFound(task_id))?;
        let rec = faktor_store::VerificationRecordRow {
            id: VerificationRecordId::new(1), // ignored by put; a fresh id is minted
            task_id,
            revision: task.revision,
            workspace_id: session.workspace_id,
            worktree_id: session.worktree_id,
            tree_hash,
            criteria,
            checks,
            changed_files,
            unrelated_changes,
            reviewer,
            status,
            started_ms,
            completed_ms: None,
        };
        Ok(store.verification_record_put(&rec)?)
    }

    /// One verification record by id, or `None`.
    pub fn get_verification_record(
        &self,
        record_id: VerificationRecordId,
    ) -> Result<Option<VerificationRecord>, TaskError> {
        self.manager
            .store()
            .verification_record_get(record_id)
            .map(|r| r.map(VerificationRecord::from))
            .map_err(TaskError::from)
    }

    /// Every verification record of `task_id`, in deterministic creation
    /// order (record id ascending).
    ///
    /// NOTE: `verification_record` rows are keyed by the NUMERIC task id
    /// (the approved schema carries no session column), so records of a
    /// standalone session (task id 1) of another session in the same
    /// workspace are visible here. Completion itself is protected by the
    /// record's workspace/worktree content check against the completing
    /// session's base worktree.
    pub fn list_verification_records(
        &self,
        task_id: TaskId,
    ) -> Result<Vec<VerificationRecord>, TaskError> {
        self.manager
            .store()
            .verification_record_list_by_task(task_id)
            .map(|rows| rows.into_iter().map(VerificationRecord::from).collect())
            .map_err(TaskError::from)
    }

    /// The record's single allowed status write (audit P0-8): a CAS from
    /// `Running` to `Passed` or `Failed`, written exactly once. A second
    /// completion attempt on an already-final record is a typed
    /// [`TaskError::RecordNotFinalizable`] carrying the current status.
    pub fn finalize_verification_record(
        &self,
        record_id: VerificationRecordId,
        status: VerificationStatus,
        completed_ms: i64,
    ) -> Result<(), TaskError> {
        if !matches!(
            status,
            VerificationStatus::Passed | VerificationStatus::Failed
        ) {
            return Err(TaskError::Malformed(format!(
                "finalize status must be Passed or Failed, got {status:?}"
            )));
        }
        let _guard = self.command_guard();
        match self
            .manager
            .store()
            .verification_record_finalize(record_id, status, completed_ms)?
        {
            Ok(()) => Ok(()),
            Err(faktor_store::RecordFinalizeRefusal::Missing { .. }) => {
                Err(TaskError::RecordNotFound(record_id))
            }
            Err(faktor_store::RecordFinalizeRefusal::NotRunning { record_id, current }) => {
                Err(TaskError::RecordNotFinalizable {
                    record: record_id,
                    current,
                })
            }
        }
    }
}

/// Crash seams of the completion sequence (adversarial fault tests): each
/// seam sits between two durable steps of
/// [`SessionHandle::complete_verified_task`]. A crash AT a seam means every
/// step before it committed and nothing after it ran — the store reopens
/// with exactly that prefix, and a re-run of the full completion converges
/// (every monetary step is idempotent).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompletionCrashPoint {
    /// After proof load + validation, before the accounting pass.
    AfterProofLoad,
    /// After the exact-usage reconcile, before the conservative finalize.
    AfterReconcile,
    /// After the conservative cost fold, before the ZERO-open assertion.
    AfterCostFold,
    /// After the ZERO-open assertion, before the transition CAS.
    BeforeTransition,
}

fn validate_task_fields(t: &Task) -> Result<(), TaskError> {
    if t.goal.len() > MAX_TASK_GOAL_BYTES {
        return Err(TaskError::Oversized(format!(
            "task goal of {} bytes exceeds MAX_TASK_GOAL_BYTES ({MAX_TASK_GOAL_BYTES})",
            t.goal.len()
        )));
    }
    if t.acceptance_criteria.len() > MAX_TASK_CRITERIA {
        return Err(TaskError::Oversized(format!(
            "{} acceptance criteria exceed MAX_TASK_CRITERIA ({MAX_TASK_CRITERIA})",
            t.acceptance_criteria.len()
        )));
    }
    let mut typed_ids = std::collections::HashSet::new();
    for c in &t.acceptance_criteria {
        if c.len() > MAX_TASK_CRITERION_BYTES {
            return Err(TaskError::Oversized(format!(
                "a criterion of {} bytes exceeds MAX_TASK_CRITERION_BYTES ({MAX_TASK_CRITERION_BYTES})",
                c.len()
            )));
        }
        // A V2 typed criterion is validated structurally (bounds +
        // deterministic content id + uniqueness); a legacy plain-text entry
        // keeps the historical per-entry bound only.
        if let Some(typed) = Criterion::decode(c) {
            typed.validate()?;
            if !typed_ids.insert(typed.id) {
                return Err(TaskError::Malformed(format!(
                    "duplicate criterion id {} in the acceptance criteria row",
                    typed.id
                )));
            }
        }
    }
    if t.plan.len() > MAX_TASK_PLAN_STEPS {
        return Err(TaskError::Oversized(format!(
            "{} plan steps exceed MAX_TASK_PLAN_STEPS ({MAX_TASK_PLAN_STEPS})",
            t.plan.len()
        )));
    }
    for s in &t.plan {
        if s.len() > MAX_TASK_STEP_BYTES {
            return Err(TaskError::Oversized(format!(
                "a plan step of {} bytes exceeds MAX_TASK_STEP_BYTES ({MAX_TASK_STEP_BYTES})",
                s.len()
            )));
        }
    }
    Ok(())
}

/// Bounded-field contract of one verification record (audit P0-8): every
/// bound is enforced before ANY write; oversized input is rejected with a
/// typed error, never truncated.
#[allow(clippy::too_many_lines)]
fn validate_verification_record(
    criteria: &[CriterionVerification],
    checks: &[CheckExecution],
    changed_files: &[FileStateEvidence],
    unrelated_changes: &[String],
    reviewer: Option<&serde_json::Value>,
    tree_hash: Option<&str>,
) -> Result<(), TaskError> {
    let reject = |what: &str| TaskError::Oversized(what.to_string());
    let malformed = |what: String| TaskError::Malformed(what);
    if criteria.len() > MAX_VERIFICATION_RECORD_CRITERIA {
        return Err(reject(&format!(
            "{} criterion verdicts exceed MAX_VERIFICATION_RECORD_CRITERIA ({MAX_VERIFICATION_RECORD_CRITERIA})",
            criteria.len()
        )));
    }
    for c in criteria {
        if c.criterion_key.is_empty() {
            return Err(malformed(
                "a criterion verdict has an empty criterion_key".into(),
            ));
        }
        if c.criterion_key.len() > MAX_VERIFICATION_CRITERION_KEY_BYTES {
            return Err(reject(&format!(
                "a criterion key of {} bytes exceeds MAX_VERIFICATION_CRITERION_KEY_BYTES ({MAX_VERIFICATION_CRITERION_KEY_BYTES})",
                c.criterion_key.len()
            )));
        }
        if let Some(evidence) = &c.evidence {
            if evidence.len() > MAX_VERIFICATION_EVIDENCE_BYTES {
                return Err(reject(&format!(
                    "criterion evidence of {} bytes exceeds MAX_VERIFICATION_EVIDENCE_BYTES ({MAX_VERIFICATION_EVIDENCE_BYTES})",
                    evidence.len()
                )));
            }
        }
    }
    let criteria_json =
        serde_json::to_vec(criteria).map_err(|e| malformed(format!("criteria json: {e}")))?;
    if criteria_json.len() > MAX_VERIFICATION_CRITERIA_JSON_BYTES {
        return Err(reject(&format!(
            "criteria JSON of {} bytes exceeds MAX_VERIFICATION_CRITERIA_JSON_BYTES ({MAX_VERIFICATION_CRITERIA_JSON_BYTES})",
            criteria_json.len()
        )));
    }
    if checks.len() > MAX_VERIFICATION_RECORD_CHECKS {
        return Err(reject(&format!(
            "{} checks exceed MAX_VERIFICATION_RECORD_CHECKS ({MAX_VERIFICATION_RECORD_CHECKS})",
            checks.len()
        )));
    }
    for ch in checks {
        if ch.check.is_empty() || ch.check.len() > MAX_VERIFICATION_CHECK_NAME_BYTES {
            return Err(reject(&format!(
                "check name {} exceeds MAX_VERIFICATION_CHECK_NAME_BYTES ({MAX_VERIFICATION_CHECK_NAME_BYTES})",
                ch.check.len()
            )));
        }
        if ch.program.is_empty() || ch.program.len() > MAX_VERIFICATION_PROGRAM_BYTES {
            return Err(reject(&format!(
                "check program {} exceeds MAX_VERIFICATION_PROGRAM_BYTES ({MAX_VERIFICATION_PROGRAM_BYTES})",
                ch.program.len()
            )));
        }
        if ch.args.len() > MAX_VERIFICATION_CHECK_ARGS {
            return Err(reject(&format!(
                "{} check args exceed MAX_VERIFICATION_CHECK_ARGS ({MAX_VERIFICATION_CHECK_ARGS})",
                ch.args.len()
            )));
        }
        for a in &ch.args {
            if a.len() > MAX_VERIFICATION_CHECK_ARG_BYTES {
                return Err(reject(&format!(
                    "a check arg of {} bytes exceeds MAX_VERIFICATION_CHECK_ARG_BYTES ({MAX_VERIFICATION_CHECK_ARG_BYTES})",
                    a.len()
                )));
            }
        }
        if ch.category.len() > MAX_VERIFICATION_CATEGORY_BYTES {
            return Err(reject(&format!(
                "check category {} exceeds MAX_VERIFICATION_CATEGORY_BYTES ({MAX_VERIFICATION_CATEGORY_BYTES})",
                ch.category.len()
            )));
        }
        if let Some(summary) = &ch.summary {
            if summary.len() > MAX_VERIFICATION_SUMMARY_BYTES {
                return Err(reject(&format!(
                    "check summary of {} bytes exceeds MAX_VERIFICATION_SUMMARY_BYTES ({MAX_VERIFICATION_SUMMARY_BYTES})",
                    summary.len()
                )));
            }
        }
    }
    let checks_json =
        serde_json::to_vec(checks).map_err(|e| malformed(format!("checks json: {e}")))?;
    if checks_json.len() > MAX_VERIFICATION_CHECKS_JSON_BYTES {
        return Err(reject(&format!(
            "checks JSON of {} bytes exceeds MAX_VERIFICATION_CHECKS_JSON_BYTES ({MAX_VERIFICATION_CHECKS_JSON_BYTES})",
            checks_json.len()
        )));
    }
    if changed_files.len() > MAX_VERIFICATION_CHANGED_FILES {
        return Err(reject(&format!(
            "{} changed files exceed MAX_VERIFICATION_CHANGED_FILES ({MAX_VERIFICATION_CHANGED_FILES})",
            changed_files.len()
        )));
    }
    for f in changed_files {
        if f.path.is_empty() || f.path.len() > MAX_VERIFICATION_PATH_BYTES {
            return Err(reject(&format!(
                "file path of {} bytes exceeds MAX_VERIFICATION_PATH_BYTES ({MAX_VERIFICATION_PATH_BYTES})",
                f.path.len()
            )));
        }
        if f.digest_hex.is_empty()
            || f.digest_hex.len() > MAX_VERIFICATION_DIGEST_BYTES
            || !f.digest_hex.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err(malformed(format!(
                "file {} digest {:?} must be non-empty lowercase/uppercase hex of at most {MAX_VERIFICATION_DIGEST_BYTES} chars",
                f.path, f.digest_hex
            )));
        }
    }
    let files_json = serde_json::to_vec(changed_files)
        .map_err(|e| malformed(format!("changed_files json: {e}")))?;
    if files_json.len() > MAX_VERIFICATION_CHANGED_FILES_JSON_BYTES {
        return Err(reject(&format!(
            "changed_files JSON of {} bytes exceeds MAX_VERIFICATION_CHANGED_FILES_JSON_BYTES ({MAX_VERIFICATION_CHANGED_FILES_JSON_BYTES})",
            files_json.len()
        )));
    }
    if unrelated_changes.len() > MAX_VERIFICATION_UNRELATED_CHANGES {
        return Err(reject(&format!(
            "{} unrelated changes exceed MAX_VERIFICATION_UNRELATED_CHANGES ({MAX_VERIFICATION_UNRELATED_CHANGES})",
            unrelated_changes.len()
        )));
    }
    for u in unrelated_changes {
        if u.is_empty() || u.len() > MAX_VERIFICATION_PATH_BYTES {
            return Err(reject(&format!(
                "an unrelated-change path of {} bytes exceeds MAX_VERIFICATION_PATH_BYTES ({MAX_VERIFICATION_PATH_BYTES})",
                u.len()
            )));
        }
    }
    let unrelated_json = serde_json::to_vec(unrelated_changes)
        .map_err(|e| malformed(format!("unrelated_changes json: {e}")))?;
    if unrelated_json.len() > MAX_VERIFICATION_UNRELATED_JSON_BYTES {
        return Err(reject(&format!(
            "unrelated_changes JSON of {} bytes exceeds MAX_VERIFICATION_UNRELATED_JSON_BYTES ({MAX_VERIFICATION_UNRELATED_JSON_BYTES})",
            unrelated_json.len()
        )));
    }
    if let Some(reviewer) = reviewer {
        let len = serde_json::to_vec(reviewer)
            .map_err(|e| malformed(format!("reviewer json: {e}")))?
            .len();
        if len > MAX_VERIFICATION_REVIEWER_JSON_BYTES {
            return Err(reject(&format!(
                "reviewer JSON of {len} bytes exceeds MAX_VERIFICATION_REVIEWER_JSON_BYTES ({MAX_VERIFICATION_REVIEWER_JSON_BYTES})"
            )));
        }
    }
    if let Some(hash) = tree_hash {
        if hash.is_empty()
            || hash.len() > MAX_VERIFICATION_TREE_HASH_BYTES
            || !hash.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err(malformed(format!(
                "tree_hash {:?} must be non-empty hex of at most {MAX_VERIFICATION_TREE_HASH_BYTES} chars",
                hash
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::BudgetAuthority;
    use crate::handle::tests::{session, test_manager};
    use faktor_core::ErrorKind;
    use std::sync::Arc;
    use std::thread;

    fn task(s: &SessionHandle) -> Task {
        Task {
            task_id: s.task_id().unwrap(),
            session_id: s.id,
            goal: "implement durable tasks".into(),
            acceptance_criteria: vec!["goal: implement durable tasks".into()],
            plan: vec!["schema".into(), "repo".into()],
            budget: TaskBudget {
                max_tokens: Some(100_000),
                max_turns: Some(10),
                spent_tokens: 0,
                spent_turns: 0,
            },
            state: TaskState::Pending,
            created_ms: 1,
            updated_ms: 1,
        }
    }

    fn criteria_task(s: &SessionHandle, task_id: TaskId, criteria: Vec<String>) -> Task {
        Task {
            task_id,
            session_id: s.id,
            goal: "gated goal".into(),
            acceptance_criteria: criteria,
            plan: vec![],
            budget: TaskBudget::default(),
            state: TaskState::Pending,
            created_ms: 1,
            updated_ms: 1,
        }
    }

    /// Drive a task Pending -> Running -> NeedsVerification -> Verifying
    /// through legal transitions; returns the revision at Verifying.
    fn drive_to_verifying(s: &SessionHandle, task_id: TaskId) -> TaskRevision {
        let r1 = s.task_revision(task_id).unwrap();
        s.transition_task(task_id, r1, TaskTransition::StartRunning, None)
            .unwrap();
        let r2 = s.task_revision(task_id).unwrap();
        s.transition_task(task_id, r2, TaskTransition::RequestVerification, None)
            .unwrap();
        let r3 = s.task_revision(task_id).unwrap();
        s.transition_task(task_id, r3, TaskTransition::StartVerification, None)
            .unwrap();
        s.task_revision(task_id).unwrap()
    }

    fn passed_record(
        s: &SessionHandle,
        task_id: TaskId,
        criteria: &[String],
    ) -> VerificationRecordId {
        let criteria: Vec<CriterionVerification> = criteria
            .iter()
            .map(|c| CriterionVerification {
                criterion_key: c.clone(),
                passed: true,
                evidence: Some("exit 0".into()),
            })
            .collect();
        s.create_verification_record(
            task_id,
            None,
            criteria,
            vec![],
            vec![],
            vec![],
            None,
            VerificationStatus::Passed,
            1,
        )
        .unwrap()
    }

    #[test]
    fn task_create_get_update_list_roundtrip() {
        let (_d, m) = test_manager();
        let s = session(&m);
        assert!(s.list_tasks().unwrap().is_empty());
        let t = task(&s);
        let created = s.create_task(t.clone()).unwrap();
        assert_eq!(created.state, TaskState::Pending);
        assert_eq!(s.get_task(created.task_id).unwrap(), Some(created.clone()));
        assert_eq!(
            s.task_revision(created.task_id).unwrap(),
            TaskRevision::new(1)
        );
        // Patch: state + spend move forward; untouched fields survive; the
        // effective change bumps the revision exactly once.
        let patched = s
            .update_task(
                created.task_id,
                TaskPatch {
                    state: Some(TaskState::Running),
                    budget: Some(TaskBudget {
                        max_tokens: Some(100_000),
                        max_turns: Some(10),
                        spent_tokens: 40,
                        spent_turns: 1,
                    }),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(patched.state, TaskState::Running);
        assert_eq!(patched.budget.spent_tokens, 40);
        assert_eq!(patched.goal, created.goal, "unpatched fields survive");
        assert_eq!(patched.created_ms, created.created_ms);
        assert_eq!(
            s.task_revision(created.task_id).unwrap(),
            TaskRevision::new(2)
        );
        assert_eq!(s.get_task(created.task_id).unwrap(), Some(patched.clone()));
        assert_eq!(s.list_tasks().unwrap().len(), 1);
    }

    #[test]
    fn update_on_missing_task_is_not_found() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let err = s
            .update_task(TaskId::new(999), TaskPatch::default())
            .unwrap_err();
        assert_eq!(err, TaskError::NotFound(TaskId::new(999)));
    }

    #[test]
    fn oversized_goal_criteria_and_plan_are_rejected_never_truncated() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let mut t = task(&s);
        t.goal = "g".repeat(MAX_TASK_GOAL_BYTES + 1);
        assert!(matches!(
            s.create_task(t.clone()).unwrap_err().kind,
            ErrorKind::Oversized
        ));
        t.goal = "ok".into();
        t.acceptance_criteria = (0..=MAX_TASK_CRITERIA)
            .map(|i| format!("criterion {i}"))
            .collect();
        assert!(matches!(
            s.create_task(t.clone()).unwrap_err().kind,
            ErrorKind::Oversized
        ));
        t.acceptance_criteria = vec!["c".repeat(MAX_TASK_CRITERION_BYTES + 1)];
        assert!(matches!(
            s.create_task(t.clone()).unwrap_err().kind,
            ErrorKind::Oversized
        ));
        t.acceptance_criteria = vec!["c".into()];
        t.plan = (0..=MAX_TASK_PLAN_STEPS)
            .map(|i| format!("step {i}"))
            .collect();
        assert!(matches!(
            s.create_task(t.clone()).unwrap_err().kind,
            ErrorKind::Oversized
        ));
        // Rejection leaves NO trace: the store stays empty, and the
        // rejected values were never silently truncated into the row.
        assert!(s.list_tasks().unwrap().is_empty());
    }

    #[test]
    fn oversized_patch_fields_are_rejected() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let created = s.create_task(task(&s)).unwrap();
        let err = s
            .update_task(
                created.task_id,
                TaskPatch {
                    goal: Some("x".repeat(MAX_TASK_GOAL_BYTES + 1)),
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(matches!(err, TaskError::Oversized(_)));
        let row = s.get_task(created.task_id).unwrap().unwrap();
        assert_eq!(row.goal, created.goal, "rejected patch left no trace");
        let err = s
            .update_task(
                created.task_id,
                TaskPatch {
                    plan: Some((0..=MAX_TASK_PLAN_STEPS).map(|i| format!("s{i}")).collect()),
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(matches!(err, TaskError::Oversized(_)));
        // A no-op patch (nothing changes) writes nothing and bumps nothing.
        let rev_before = s.task_revision(created.task_id).unwrap();
        let noop = s
            .update_task(created.task_id, TaskPatch::default())
            .unwrap();
        assert_eq!(noop, created);
        assert_eq!(s.task_revision(created.task_id).unwrap(), rev_before);
    }

    #[test]
    fn budget_spend_only_moves_forward() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let created = s.create_task(task(&s)).unwrap();
        let bump = |spent_tokens: u64| TaskPatch {
            budget: Some(TaskBudget {
                max_tokens: Some(100_000),
                max_turns: Some(10),
                spent_tokens,
                spent_turns: 3,
            }),
            ..Default::default()
        };
        let p1 = s.update_task(created.task_id, bump(50)).unwrap();
        assert_eq!(p1.budget.spent_tokens, 50);
        // A rewind attempt is refused by construction: the effective content
        // is unchanged, so NOTHING is written and the revision does not move.
        let rev = s.task_revision(created.task_id).unwrap();
        let p2 = s.update_task(created.task_id, bump(10)).unwrap();
        assert_eq!(p2.budget.spent_tokens, 50, "spend is monotone");
        assert_eq!(s.task_revision(created.task_id).unwrap(), rev);
        // max fields DO update (they are not counters) and bump.
        let p3 = s
            .update_task(
                created.task_id,
                TaskPatch {
                    budget: Some(TaskBudget {
                        max_tokens: Some(5),
                        max_turns: Some(1),
                        spent_tokens: 0,
                        spent_turns: 0,
                    }),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(p3.budget.max_tokens, Some(5));
        assert_eq!(p3.budget.spent_tokens, 50);
        assert_eq!(
            s.task_revision(created.task_id).unwrap(),
            TaskRevision::new(3)
        );
    }

    #[test]
    fn tasks_are_session_scoped() {
        let (_d, m) = test_manager();
        let s1 = session(&m);
        let s2 = {
            let ws = m.create_workspace("/w2").unwrap();
            m.create_session(ws, "t2", "p", "m").unwrap()
        };
        let t1 = s1.create_task(task(&s1)).unwrap();
        assert!(s2.list_tasks().unwrap().is_empty());
        assert!(s2.get_task(t1.task_id).unwrap().is_none());
    }

    // (a) the direct patch cannot reach completion states or jump edges.
    #[test]
    fn patch_cannot_reach_completion_states_or_jump_edges() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let created = s.create_task(task(&s)).unwrap();
        for forbidden in [
            TaskState::VerifiedComplete,
            TaskState::Verifying,
            TaskState::NeedsVerification,
        ] {
            let err = s
                .update_task(
                    created.task_id,
                    TaskPatch {
                        state: Some(forbidden),
                        ..Default::default()
                    },
                )
                .unwrap_err();
            assert_eq!(
                err,
                TaskError::CompletionStateViaPatch { state: forbidden },
                "{forbidden:?} must be patch-unreachable"
            );
        }
        let row = s.get_task(created.task_id).unwrap().unwrap();
        assert_eq!(row.state, TaskState::Pending, "state unchanged");
        assert_eq!(
            s.task_revision(created.task_id).unwrap(),
            TaskRevision::new(1)
        );
        // A non-completion edge the machine forbids (Pending -> Failed).
        let err = s
            .update_task(
                created.task_id,
                TaskPatch {
                    state: Some(TaskState::Failed),
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(matches!(
            err,
            TaskError::IllegalTransition {
                from: TaskState::Pending,
                to: TaskState::Failed,
                ..
            }
        ));
        assert_eq!(
            s.task_revision(created.task_id).unwrap(),
            TaskRevision::new(1)
        );
        // Legal ordinary edges apply and bump exactly once.
        s.update_task(
            created.task_id,
            TaskPatch {
                state: Some(TaskState::Running),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            s.task_revision(created.task_id).unwrap(),
            TaskRevision::new(2)
        );
    }

    // (b) create_task admits only Pending/Planning/Running and never
    // recreates an existing row.
    #[test]
    fn create_task_rejects_uncreatable_states_and_recreation() {
        let (_d, m) = test_manager();
        let s = session(&m);
        for forbidden in [
            TaskState::VerifiedComplete,
            TaskState::Verifying,
            TaskState::NeedsVerification,
            TaskState::Failed,
            TaskState::Cancelled,
            TaskState::Waiting,
            TaskState::Blocked,
        ] {
            let mut t = task(&s);
            t.state = forbidden;
            let err = s.create_task(t.clone()).unwrap_err();
            assert_eq!(err.kind, ErrorKind::Conflict);
            assert!(
                err.message.contains("create_task refused"),
                "{forbidden:?}: {}",
                err.message
            );
        }
        assert!(
            s.list_tasks().unwrap().is_empty(),
            "no trace of refused creates"
        );
        let t = task(&s);
        let created = s.create_task(t.clone()).unwrap();
        assert_eq!(created.state, TaskState::Pending);
        let again = s.create_task(t.clone()).unwrap_err();
        assert_eq!(again.kind, ErrorKind::Conflict);
        assert!(again.message.contains("already exists"));
        assert_eq!(s.list_tasks().unwrap().len(), 1);
        assert_eq!(
            s.task_revision(created.task_id).unwrap(),
            TaskRevision::new(1)
        );
    }

    // (c)+(d) the completion transaction: typed refusals for every broken
    // proof facet, state and revision untouched, happy path bumps exactly
    // once, and VerifiedComplete is terminal.
    #[test]
    fn completion_requires_proof_records_revision_criteria_and_worktree() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let main = s
            .create_task(criteria_task(
                &s,
                s.task_id().unwrap(),
                vec!["c1".into(), "c2".into()],
            ))
            .unwrap();
        let t1 = main.task_id;
        let rev_at_verifying = drive_to_verifying(&s, t1);
        let record = passed_record(&s, t1, &["c1".into(), "c2".into()]);
        let done = s
            .complete_verified_task(t1, rev_at_verifying, record)
            .unwrap();
        assert_eq!(done.state, TaskState::VerifiedComplete);
        assert_eq!(
            s.task_revision(t1).unwrap(),
            rev_at_verifying.checked_next().unwrap(),
            "revision bumped exactly once"
        );
        // VerifiedComplete is terminal: content edits are frozen, no
        // transition is legal, a second completion refuses.
        let frozen = s
            .update_task(
                t1,
                TaskPatch {
                    goal: Some("tamper".into()),
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert_eq!(
            frozen,
            TaskError::TerminalTask {
                task_id: t1,
                state: TaskState::VerifiedComplete
            }
        );
        let rev_done = s.task_revision(t1).unwrap();
        let frozen2 = s
            .transition_task(t1, rev_done, TaskTransition::Cancel, None)
            .unwrap_err();
        assert!(matches!(frozen2, TaskError::IllegalTransition { .. }));
        let frozen3 = s.complete_verified_task(t1, rev_done, record).unwrap_err();
        assert_eq!(
            frozen3,
            TaskError::NotVerifying {
                actual: TaskState::VerifiedComplete
            }
        );
        assert_eq!(s.task_revision(t1).unwrap(), rev_done, "no double bump");

        let keep_verifying = |s: &SessionHandle, id: TaskId, rev: TaskRevision| {
            let row = s.get_task(id).unwrap().unwrap();
            assert_eq!(row.state, TaskState::Verifying);
            assert_eq!(s.task_revision(id).unwrap(), rev, "refusal must not bump");
        };

        // Missing record.
        let t2 = s
            .create_task(criteria_task(&s, TaskId::new(2), vec!["c1".into()]))
            .unwrap();
        let r2 = drive_to_verifying(&s, t2.task_id);
        let err = s
            .complete_verified_task(t2.task_id, r2, VerificationRecordId::new(42_000))
            .unwrap_err();
        assert_eq!(
            err,
            TaskError::RecordNotFound(VerificationRecordId::new(42_000))
        );
        keep_verifying(&s, t2.task_id, r2);
        // Wrong task: the record certifies task 1, not task 2.
        let err = s
            .complete_verified_task(t2.task_id, r2, record)
            .unwrap_err();
        assert_eq!(
            err,
            TaskError::RecordWrongTask {
                record,
                record_task: t1,
                requested_task: t2.task_id
            }
        );
        keep_verifying(&s, t2.task_id, r2);

        // Wrong revision: a record certifies the revision before a legal
        // content bump (budget while Verifying) and cannot complete the
        // moved task.
        let t3 = s
            .create_task(criteria_task(&s, TaskId::new(3), vec!["c1".into()]))
            .unwrap();
        let r3 = drive_to_verifying(&s, t3.task_id);
        let rec3 = passed_record(&s, t3.task_id, &["c1".into()]);
        s.update_task(
            t3.task_id,
            TaskPatch {
                budget: Some(TaskBudget {
                    max_tokens: Some(1),
                    max_turns: None,
                    spent_tokens: 0,
                    spent_turns: 0,
                }),
                ..Default::default()
            },
        )
        .unwrap();
        let r3_after = s.task_revision(t3.task_id).unwrap();
        assert_ne!(r3_after, r3);
        let err = s
            .complete_verified_task(t3.task_id, r3_after, rec3)
            .unwrap_err();
        assert_eq!(
            err,
            TaskError::RecordWrongRevision {
                record: rec3,
                record_revision: r3,
                expected: r3_after
            }
        );
        keep_verifying(&s, t3.task_id, r3_after);
        // The STALE expected revision reports the task-side mismatch first.
        let err = s.complete_verified_task(t3.task_id, r3, rec3).unwrap_err();
        assert_eq!(
            err,
            TaskError::RevisionMismatch {
                task_id: t3.task_id,
                expected: r3,
                actual: r3_after
            }
        );
        keep_verifying(&s, t3.task_id, r3_after);

        // Record status Failed refuses with full coverage.
        let t4 = s
            .create_task(criteria_task(&s, TaskId::new(4), vec!["c1".into()]))
            .unwrap();
        let r4 = drive_to_verifying(&s, t4.task_id);
        let failed_rec = s
            .create_verification_record(
                t4.task_id,
                None,
                vec![CriterionVerification {
                    criterion_key: "c1".into(),
                    passed: true,
                    evidence: None,
                }],
                vec![],
                vec![],
                vec![],
                None,
                VerificationStatus::Failed,
                1,
            )
            .unwrap();
        let err = s
            .complete_verified_task(t4.task_id, r4, failed_rec)
            .unwrap_err();
        assert_eq!(
            err,
            TaskError::RecordNotPassed {
                record: failed_rec,
                status: VerificationStatus::Failed
            }
        );
        keep_verifying(&s, t4.task_id, r4);

        // A record missing one current criterion refuses with the missing
        // list; full coverage completes; a passed=false entry is NOT
        // coverage.
        let t5 = s
            .create_task(criteria_task(
                &s,
                TaskId::new(5),
                vec!["c1".into(), "c2".into()],
            ))
            .unwrap();
        let r5 = drive_to_verifying(&s, t5.task_id);
        let partial = passed_record(&s, t5.task_id, &["c1".into()]);
        let err = s
            .complete_verified_task(t5.task_id, r5, partial)
            .unwrap_err();
        assert_eq!(
            err,
            TaskError::CriteriaNotCovered {
                record: partial,
                missing: vec!["c2".to_string()]
            }
        );
        keep_verifying(&s, t5.task_id, r5);
        let full = passed_record(&s, t5.task_id, &["c1".into(), "c2".into()]);
        let done5 = s.complete_verified_task(t5.task_id, r5, full).unwrap();
        assert_eq!(done5.state, TaskState::VerifiedComplete);
        let t6 = s
            .create_task(criteria_task(&s, TaskId::new(6), vec!["c1".into()]))
            .unwrap();
        let r6 = drive_to_verifying(&s, t6.task_id);
        let lying = s
            .create_verification_record(
                t6.task_id,
                None,
                vec![CriterionVerification {
                    criterion_key: "c1".into(),
                    passed: false,
                    evidence: None,
                }],
                vec![],
                vec![],
                vec![],
                None,
                VerificationStatus::Passed,
                1,
            )
            .unwrap();
        let err = s.complete_verified_task(t6.task_id, r6, lying).unwrap_err();
        assert_eq!(
            err,
            TaskError::CriteriaNotCovered {
                record: lying,
                missing: vec!["c1".to_string()]
            }
        );
        keep_verifying(&s, t6.task_id, r6);

        // (g) worktree mismatch: the record's base worktree must equal the
        // completing session's. Standalone sessions share the numeric task
        // id 1 but live in different workspaces: a record certified in /wb
        // cannot complete task 1 of a session in /wc.
        let wb = m.create_workspace("/wb").unwrap();
        let sb = m.create_session(wb, "b", "p", "m").unwrap();
        let tb = sb
            .create_task(criteria_task(&sb, sb.task_id().unwrap(), vec!["c1".into()]))
            .unwrap();
        let rb = drive_to_verifying(&sb, tb.task_id);
        let rec_b = passed_record(&sb, tb.task_id, &["c1".into()]);
        let wc = m.create_workspace("/wc").unwrap();
        let sc = m.create_session(wc, "c", "p", "m").unwrap();
        assert_eq!(
            sc.task_id().unwrap(),
            tb.task_id,
            "both standalone at task 1"
        );
        let tc = sc
            .create_task(criteria_task(&sc, sc.task_id().unwrap(), vec!["c1".into()]))
            .unwrap();
        let rc = drive_to_verifying(&sc, tc.task_id);
        let err = sc
            .complete_verified_task(tc.task_id, rc, rec_b)
            .unwrap_err();
        assert!(matches!(err, TaskError::WorktreeMismatch { .. }));
        assert_eq!(sc.task_revision(tc.task_id).unwrap(), rc);
        // And the same-workspace record completes it.
        let rec_c = passed_record(&sc, tc.task_id, &["c1".into()]);
        let done_c = sc.complete_verified_task(tc.task_id, rc, rec_c).unwrap();
        assert_eq!(done_c.state, TaskState::VerifiedComplete);
        let _ = (rb, wb, sb, tb);
    }

    #[test]
    fn transition_task_is_the_only_way_into_completion_relevant_states() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let created = s.create_task(task(&s)).unwrap();
        let t = created.task_id;
        // Wait on a Pending task: illegal edge.
        let err = s
            .transition_task(t, TaskRevision::new(1), TaskTransition::Wait, None)
            .unwrap_err();
        assert!(matches!(err, TaskError::IllegalTransition { .. }));
        // A stale revision refuses even for a legal edge.
        let err = s
            .transition_task(t, TaskRevision::new(9), TaskTransition::StartRunning, None)
            .unwrap_err();
        assert_eq!(
            err,
            TaskError::RevisionMismatch {
                task_id: t,
                expected: TaskRevision::new(9),
                actual: TaskRevision::new(1)
            }
        );
        // Proofs do not ride transitions.
        let err = s
            .transition_task(
                t,
                TaskRevision::new(1),
                TaskTransition::StartRunning,
                Some(VerificationRecordId::new(7)),
            )
            .unwrap_err();
        assert!(matches!(err, TaskError::Malformed(_)));
        // The full chain into Verifying bumps revision every hop.
        let r1 = s.task_revision(t).unwrap();
        s.transition_task(t, r1, TaskTransition::StartRunning, None)
            .unwrap();
        let r2 = s.task_revision(t).unwrap();
        s.transition_task(t, r2, TaskTransition::Wait, None)
            .unwrap();
        let r3 = s.task_revision(t).unwrap();
        s.transition_task(t, r3, TaskTransition::ResumeFromWaiting, None)
            .unwrap();
        let r4 = s.task_revision(t).unwrap();
        s.transition_task(t, r4, TaskTransition::RequestVerification, None)
            .unwrap();
        let r5 = s.task_revision(t).unwrap();
        s.transition_task(t, r5, TaskTransition::StartVerification, None)
            .unwrap();
        let r6 = s.task_revision(t).unwrap();
        assert_eq!(r6, TaskRevision::new(6));
        assert_eq!(s.get_task(t).unwrap().unwrap().state, TaskState::Verifying);
        // Reverify and fail edges from Verifying.
        s.transition_task(t, r6, TaskTransition::Reverify, None)
            .unwrap();
        let r7 = s.task_revision(t).unwrap();
        assert_eq!(
            s.get_task(t).unwrap().unwrap().state,
            TaskState::NeedsVerification
        );
        s.transition_task(t, r7, TaskTransition::StartVerification, None)
            .unwrap();
        let r8 = s.task_revision(t).unwrap();
        s.transition_task(t, r8, TaskTransition::FailFromVerifying, None)
            .unwrap();
        assert_eq!(s.get_task(t).unwrap().unwrap().state, TaskState::Failed);
        // Failed is terminal: Cancel is illegal.
        let r9 = s.task_revision(t).unwrap();
        let err = s
            .transition_task(t, r9, TaskTransition::Cancel, None)
            .unwrap_err();
        assert!(matches!(err, TaskError::IllegalTransition { .. }));
        // Cancellation from a non-terminal state is legal.
        let t2 = s
            .create_task(criteria_task(&s, TaskId::new(7), vec![]))
            .unwrap();
        let c = s
            .transition_task(
                t2.task_id,
                TaskRevision::new(1),
                TaskTransition::Cancel,
                None,
            )
            .unwrap();
        assert_eq!(c.state, TaskState::Cancelled);
    }

    // (e) racing completions and finalizes: exactly one winner each.
    #[test]
    fn concurrent_completions_and_finalizes_win_exactly_once() {
        let (_d, m) = test_manager();
        let s = Arc::new(session(&m));
        let t = s
            .create_task(criteria_task(&s, s.task_id().unwrap(), vec!["c1".into()]))
            .unwrap();
        let tid = t.task_id;
        let rev = drive_to_verifying(&s, tid);
        let rec = passed_record(&s, tid, &["c1".into()]);
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let mut handles = Vec::new();
        for _ in 0..2 {
            let s = s.clone();
            let barrier = barrier.clone();
            handles.push(thread::spawn(move || {
                barrier.wait();
                s.complete_verified_task(tid, rev, rec)
            }));
        }
        barrier.wait();
        let results: Vec<Result<Task, TaskError>> =
            handles.into_iter().map(|h| h.join().unwrap()).collect();
        let wins = results.iter().filter(|r| r.is_ok()).count();
        assert_eq!(wins, 1, "exactly one completion winner: {results:?}");
        for r in &results {
            if let Err(e) = r {
                assert!(
                    matches!(
                        e,
                        TaskError::NotVerifying { .. } | TaskError::RevisionMismatch { .. }
                    ),
                    "loser must fail typed, got {e:?}"
                );
            }
        }
        assert_eq!(s.task_revision(tid).unwrap(), rev.checked_next().unwrap());
        assert_eq!(
            s.get_task(tid).unwrap().unwrap().state,
            TaskState::VerifiedComplete
        );

        // Record-finalize CAS: a Running record finalizes exactly once.
        let t2 = s
            .create_task(criteria_task(&s, TaskId::new(2), vec![]))
            .unwrap();
        let running = s
            .create_verification_record(
                t2.task_id,
                None,
                vec![],
                vec![],
                vec![],
                vec![],
                None,
                VerificationStatus::Running,
                1,
            )
            .unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let mut handles = Vec::new();
        for _ in 0..2 {
            let s = s.clone();
            let barrier = barrier.clone();
            handles.push(thread::spawn(move || {
                barrier.wait();
                s.finalize_verification_record(running, VerificationStatus::Passed, 99)
            }));
        }
        barrier.wait();
        let results: Vec<Result<(), TaskError>> =
            handles.into_iter().map(|h| h.join().unwrap()).collect();
        let wins = results.iter().filter(|r| r.is_ok()).count();
        assert_eq!(wins, 1, "exactly one finalize winner: {results:?}");
        for r in &results {
            if let Err(e) = r {
                assert!(matches!(
                    e,
                    TaskError::RecordNotFinalizable {
                        current: VerificationStatus::Passed,
                        ..
                    }
                ));
            }
        }
        assert_eq!(
            s.get_verification_record(running).unwrap().unwrap().status,
            VerificationStatus::Passed
        );
    }

    // (f)+(i) crash between record creation and completion: reopen shows a
    // consistent store; the completion applies exactly once afterwards, and
    // records survive.
    #[test]
    fn records_and_state_survive_reopen_and_completion_after_crash() {
        let (dir, m) = test_manager();
        let s = session(&m);
        let sid = s.id;
        let t = s
            .create_task(criteria_task(&s, s.task_id().unwrap(), vec!["c1".into()]))
            .unwrap();
        let tid = t.task_id;
        let rev = drive_to_verifying(&s, tid);
        let rec = passed_record(&s, tid, &["c1".into()]);
        let listed = s.list_verification_records(tid).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].record_id, rec);
        assert_eq!(listed[0].revision, rev);
        // Crash (drop the manager) before the completion transaction.
        drop(s);
        drop(m);
        let m2 =
            crate::SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                .unwrap();
        let s2 = m2.get_session(sid).unwrap().unwrap();
        let row = s2.get_task(tid).unwrap().unwrap();
        assert_eq!(row.state, TaskState::Verifying, "crashed mid-verification");
        assert_eq!(s2.task_revision(tid).unwrap(), rev);
        assert_eq!(
            s2.get_verification_record(rec).unwrap().unwrap().revision,
            rev,
            "the record survived the crash"
        );
        // Reopen again after a second crash between record and completion…
        drop(s2);
        drop(m2);
        let m3 =
            crate::SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                .unwrap();
        let s3 = m3.get_session(sid).unwrap().unwrap();
        let done = s3.complete_verified_task(tid, rev, rec).unwrap();
        assert_eq!(done.state, TaskState::VerifiedComplete);
        assert_eq!(s3.task_revision(tid).unwrap(), rev.checked_next().unwrap());
        // Repeat completion after ANOTHER reopen refuses and does not bump.
        drop(s3);
        drop(m3);
        let m4 =
            crate::SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                .unwrap();
        let s4 = m4.get_session(sid).unwrap().unwrap();
        let err = s4
            .complete_verified_task(tid, rev.checked_next().unwrap(), rec)
            .unwrap_err();
        assert_eq!(
            err,
            TaskError::NotVerifying {
                actual: TaskState::VerifiedComplete
            }
        );
        assert_eq!(s4.task_revision(tid).unwrap(), rev.checked_next().unwrap());
        let frozen = s4
            .update_task(
                tid,
                TaskPatch {
                    goal: Some("tamper".into()),
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert_eq!(
            frozen,
            TaskError::TerminalTask {
                task_id: tid,
                state: TaskState::VerifiedComplete
            }
        );
        let records = s4.list_verification_records(tid).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].record_id, rec);
    }

    // (g) oversized record JSON is rejected with a typed error and NO row
    // is written (never truncated).
    #[test]
    fn oversized_record_json_is_rejected_typed_and_never_truncated() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let created = s.create_task(task(&s)).unwrap();
        let tid = created.task_id;
        // criteria_json over 128 KiB: 64 keys near the per-key cap.
        let huge_criteria: Vec<CriterionVerification> = (0..MAX_VERIFICATION_RECORD_CRITERIA)
            .map(|i| CriterionVerification {
                criterion_key: format!("{i:03}").repeat(MAX_VERIFICATION_CRITERION_KEY_BYTES / 4),
                passed: true,
                evidence: None,
            })
            .collect();
        assert!(
            serde_json::to_vec(&huge_criteria).unwrap().len()
                > MAX_VERIFICATION_CRITERIA_JSON_BYTES
        );
        let err = s
            .create_verification_record(
                tid,
                None,
                huge_criteria,
                vec![],
                vec![],
                vec![],
                None,
                VerificationStatus::Passed,
                1,
            )
            .unwrap_err();
        assert!(matches!(err, TaskError::Oversized(_)));
        // checks_json over 256 KiB.
        let huge_checks: Vec<CheckExecution> = (0..MAX_VERIFICATION_RECORD_CHECKS)
            .map(|i| CheckExecution {
                check: format!("check {i}").repeat(MAX_VERIFICATION_CHECK_NAME_BYTES / 10),
                program: "cargo".into(),
                args: vec![],
                category: "required".into(),
                required: true,
                status: VerificationStatus::Passed,
                started_ms: 1,
                finished_ms: Some(2),
                exit: Some(0),
                summary: Some("s".repeat(MAX_VERIFICATION_SUMMARY_BYTES)),
            })
            .collect();
        assert!(
            serde_json::to_vec(&huge_checks).unwrap().len() > MAX_VERIFICATION_CHECKS_JSON_BYTES
        );
        let err = s
            .create_verification_record(
                tid,
                None,
                vec![],
                huge_checks,
                vec![],
                vec![],
                None,
                VerificationStatus::Passed,
                1,
            )
            .unwrap_err();
        assert!(matches!(err, TaskError::Oversized(_)));
        // A hostile non-hex digest is malformed (typed), a giant evidence is
        // oversized.
        let err = s
            .create_verification_record(
                tid,
                None,
                vec![CriterionVerification {
                    criterion_key: "c1".into(),
                    passed: true,
                    evidence: None,
                }],
                vec![],
                vec![FileStateEvidence {
                    path: "src/main.rs".into(),
                    digest_hex: "not-hex!!".into(),
                    size: 1,
                }],
                vec![],
                None,
                VerificationStatus::Passed,
                1,
            )
            .unwrap_err();
        assert!(matches!(err, TaskError::Malformed(_)));
        let err = s
            .create_verification_record(
                tid,
                None,
                vec![CriterionVerification {
                    criterion_key: "c1".into(),
                    passed: true,
                    evidence: Some("e".repeat(MAX_VERIFICATION_EVIDENCE_BYTES + 1)),
                }],
                vec![],
                vec![],
                vec![],
                None,
                VerificationStatus::Passed,
                1,
            )
            .unwrap_err();
        assert!(matches!(err, TaskError::Oversized(_)));
        assert!(
            s.list_verification_records(tid).unwrap().is_empty(),
            "refused records left no trace"
        );
    }

    // (h) revisions are strictly monotone under 20 mixed updates — no
    // reuse, no backwards gap — also across a reopen.
    #[test]
    fn revisions_stay_monotone_under_mixed_updates_and_reopen() {
        let (dir, m) = test_manager();
        let s = session(&m);
        let sid = s.id;
        let created = s.create_task(task(&s)).unwrap();
        let tid = created.task_id;
        // Start the machine so the Wait/Resume round trips are legal.
        let r0 = s.task_revision(tid).unwrap();
        s.transition_task(tid, r0, TaskTransition::StartRunning, None)
            .unwrap();
        let mut last = s.task_revision(tid).unwrap();
        let mut seen = vec![last];
        for i in 0..20u64 {
            if i % 3 == 0 {
                s.update_task(
                    tid,
                    TaskPatch {
                        goal: Some(format!("goal iteration {i}")),
                        ..Default::default()
                    },
                )
                .unwrap();
            } else if i % 3 == 1 {
                s.update_task(
                    tid,
                    TaskPatch {
                        plan: Some(vec![format!("step {i}")]),
                        budget: Some(TaskBudget {
                            max_tokens: Some(i),
                            max_turns: Some(i as u32),
                            spent_tokens: 0,
                            spent_turns: 0,
                        }),
                        ..Default::default()
                    },
                )
                .unwrap();
            } else {
                // Ordinary state machine round trip Running -> Waiting ->
                // Running.
                let r = s.task_revision(tid).unwrap();
                s.transition_task(tid, r, TaskTransition::Wait, None)
                    .unwrap();
                let r = s.task_revision(tid).unwrap();
                s.transition_task(tid, r, TaskTransition::ResumeFromWaiting, None)
                    .unwrap();
            }
            let now = s.task_revision(tid).unwrap();
            assert!(
                now > last,
                "revision must strictly increase: {now:?} after {last:?}"
            );
            assert!(
                !seen.contains(&now),
                "revision {now:?} reused after {seen:?}"
            );
            seen.push(now);
            last = now;
        }
        assert!(last.raw() >= 2 + 20, "each iteration bumped at least once");
        // No-op patches do not move the revision.
        let before = last;
        s.update_task(tid, TaskPatch::default()).unwrap();
        assert_eq!(s.task_revision(tid).unwrap(), before);
        // Reopen: the sequence continues, never resetting or reusing.
        drop(s);
        drop(m);
        let m2 =
            crate::SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                .unwrap();
        let s2 = m2.get_session(sid).unwrap().unwrap();
        let after_reopen = s2.task_revision(tid).unwrap();
        assert_eq!(after_reopen, before, "revision survives reopen untouched");
        s2.update_task(
            tid,
            TaskPatch {
                goal: Some("after reopen".into()),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            s2.task_revision(tid).unwrap(),
            after_reopen.checked_next().unwrap(),
            "no reset, no reuse after reopen"
        );
    }

    // (j) record listing is deterministic (creation order) across calls.
    #[test]
    fn record_listing_is_deterministic() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let created = s.create_task(task(&s)).unwrap();
        let tid = created.task_id;
        let r1 = passed_record(&s, tid, &[]);
        let r2 = passed_record(&s, tid, &[]);
        let r3 = passed_record(&s, tid, &[]);
        let first = s.list_verification_records(tid).unwrap();
        let ids: Vec<VerificationRecordId> = first.iter().map(|r| r.record_id).collect();
        assert_eq!(ids, vec![r1, r2, r3], "creation order");
        assert_eq!(s.list_verification_records(tid).unwrap(), first, "stable");
        // Records of another task do not leak into this task's list.
        let t2 = s
            .create_task(criteria_task(&s, TaskId::new(2), vec![]))
            .unwrap();
        assert!(s.list_verification_records(t2.task_id).unwrap().is_empty());
    }

    // ---------------------------------------------------------------- accounting-before-completion
    // (attempt-accounting completion invariant): VerifiedComplete requires
    // zero OPEN + UNCERTAIN reservations, exact usage reconciled first, the
    // rest charged conservatively at reserved estimates, and the task row
    // only then transitioning. Fault tests crash at every seam and reopen.

    fn ledger_for(m: &Arc<crate::SessionManager>) -> Arc<crate::budget::DurableBudgetLedger> {
        crate::budget::DurableBudgetLedger::new(m.clone())
    }

    fn snapshot() -> faktor_core::model::PricingSnapshot {
        use faktor_core::model::{MicroUsdPerMillionTokens, PriceQuote};
        faktor_core::model::PricingSnapshot::exact(
            PriceQuote {
                input: MicroUsdPerMillionTokens(10_000_000),
                output: MicroUsdPerMillionTokens(20_000_000),
                cache_read: MicroUsdPerMillionTokens(2_000_000),
                cache_write: MicroUsdPerMillionTokens(4_000_000),
            },
            7,
            "task-accounting-test".into(),
        )
    }

    /// A Verifying task with a Passed record at the current revision, and a
    /// task row under a hard cost cap.
    fn verifying_task_with_record(
        s: &SessionHandle,
        cap: Option<u64>,
    ) -> (TaskId, TaskRevision, VerificationRecordId) {
        let t = s
            .create_task(criteria_task(s, s.task_id().unwrap(), vec!["c1".into()]))
            .unwrap();
        let tid = t.task_id;
        ledger_for(&s.manager)
            .set_task_max_cost(s.id, tid, cap)
            .unwrap();
        let rev = drive_to_verifying(s, tid);
        let rec = passed_record(s, tid, &["c1".into()]);
        (tid, rev, rec)
    }

    #[tokio::test]
    async fn open_reserved_reservation_refuses_completion_and_task_stays_verifying() {
        let (dir, m) = test_manager();
        let s = session(&m);
        let (tid, rev, rec) = verifying_task_with_record(&s, None);
        // A reservation that was never dispatched (a lost pre-dispatch row):
        // still RESERVED, refundable — but the completion gate must refuse
        // while it holds budget.
        let ledger = ledger_for(&m);
        let r = ledger
            .reserve(s.id, tid, m.next_op_id(), 5_000, None)
            .await
            .unwrap();
        let err = s.complete_verified_task(tid, rev, rec).unwrap_err();
        assert!(
            matches!(
                err,
                TaskError::AccountingIncomplete {
                    open_count: 1,
                    open_micro: 5_000,
                    dispatched_count: 0,
                    uncertain_count: 0,
                    uncertain_micro: 0,
                    ..
                }
            ),
            "{err:?}"
        );
        assert_eq!(
            s.get_task(tid).unwrap().unwrap().state,
            TaskState::Verifying,
            "the refusal never transitions the row"
        );
        // Refunding the lost reservation lets the SAME completion pass land
        // (nothing was charged, nothing was written by the refusal).
        let sid = s.id;
        drop(s);
        let s2 = m.get_session(sid).unwrap().unwrap();
        ledger.refund(sid, r).await.unwrap();
        let done = s2.complete_verified_task(tid, rev, rec).unwrap();
        assert_eq!(done.state, TaskState::VerifiedComplete);
        let _ = dir;
    }

    #[tokio::test]
    async fn dispatched_open_row_refuses_completion_until_uncertain_or_settled() {
        let (_dir, m) = test_manager();
        let s = session(&m);
        let (tid, rev, rec) = verifying_task_with_record(&s, Some(1_000_000));
        let ledger = ledger_for(&m);
        // A DISPATCHED row (durable marker written, provider may have
        // billed): never refundable; the completion gate must refuse while
        // it sits open — the conservative close is mark_uncertain (or a
        // settle), never a refund.
        let r = ledger
            .reserve(s.id, tid, m.next_op_id(), 8_000, None)
            .await
            .unwrap();
        ledger.mark_dispatched(s.id, r).await.unwrap();
        let err = s.complete_verified_task(tid, rev, rec).unwrap_err();
        assert!(
            matches!(
                err,
                TaskError::AccountingIncomplete {
                    open_count: 1,
                    open_micro: 8_000,
                    dispatched_count: 1,
                    ..
                }
            ),
            "{err:?}"
        );
        assert_eq!(
            s.get_task(tid).unwrap().unwrap().state,
            TaskState::Verifying
        );
        // A refund is SQL-refused on the dispatched row.
        assert!(matches!(
            ledger.refund(s.id, r).await,
            Err(crate::budget::BudgetError::CannotRefundDispatched { .. })
        ));
        // Closing it as UNCERTAIN lets completion charge the estimate and
        // land.
        ledger
            .mark_uncertain(s.id, r, "test_dispatch_left_open".into(), None)
            .await
            .unwrap();
        let done = s.complete_verified_task(tid, rev, rec).unwrap();
        assert_eq!(done.state, TaskState::VerifiedComplete);
        let balance = ledger.completion_accounting_balance(s.id, tid).unwrap();
        assert!(balance.is_zero());
        assert_eq!(balance.spent_cost_micro, 8_000, "charged the estimate");
    }

    #[tokio::test]
    async fn uncertain_attempt_is_charged_conservatively_at_its_reserved_estimate() {
        let (_dir, m) = test_manager();
        let s = session(&m);
        let (tid, rev, rec) = verifying_task_with_record(&s, Some(1_000_000));
        let ledger = ledger_for(&m);
        // A crashed dispatched attempt with no completed provider row: exact
        // usage unknown — finalize charges the reserved estimate.
        let r = ledger
            .reserve(s.id, tid, m.next_op_id(), 12_000, Some(snapshot()))
            .await
            .unwrap();
        ledger.mark_dispatched(s.id, r).await.unwrap();
        ledger
            .mark_uncertain(s.id, r, "crash".into(), None)
            .await
            .unwrap();
        let done = s.complete_verified_task(tid, rev, rec).unwrap();
        assert_eq!(done.state, TaskState::VerifiedComplete);
        let balance = ledger.completion_accounting_balance(s.id, tid).unwrap();
        assert!(
            balance.is_zero(),
            "VerifiedComplete => zero open + uncertain"
        );
        assert_eq!(balance.spent_cost_micro, 12_000);
    }

    #[tokio::test]
    async fn uncertain_attempt_with_known_usage_reconciles_exactly_before_charging() {
        let (_dir, m) = test_manager();
        let s = session(&m);
        let (tid, rev, rec) = verifying_task_with_record(&s, Some(1_000_000));
        let ledger = ledger_for(&m);
        // The crashed attempt DID complete at the provider (a completed
        // attempt-keyed provider_call row exists): reconcile settles FROM
        // that exact usage — 900 input + 100 output tokens x the frozen
        // snapshot (10 micro / 1M tokens x ... pricing snapshot fields are
        // microUSD per MILLION tokens here) — never the 60_000 estimate.
        let logical = m.next_op_id();
        let attempt = faktor_core::op::ModelCallAttempt::new(logical, m.next_op_id(), 0).unwrap();
        let r = ledger
            .reserve_attempt(s.id, tid, attempt, 60_000, Some(snapshot()))
            .await
            .unwrap();
        ledger.mark_dispatched(s.id, r).await.unwrap();
        ledger
            .mark_uncertain(s.id, r, "crash_after_completion".into(), None)
            .await
            .unwrap();
        // The completed durable row of THIS attempt (op id = the logical op,
        // attempt keyed; the exact usage is durable truth).
        s.record_provider_call_attempt(
            attempt,
            Some(r),
            "fake",
            "m",
            "completed",
            Some(900),
            Some(100),
            None,
        )
        .unwrap();
        let done = s.complete_verified_task(tid, rev, rec).unwrap();
        assert_eq!(done.state, TaskState::VerifiedComplete);
        let balance = ledger.completion_accounting_balance(s.id, tid).unwrap();
        assert!(balance.is_zero());
        // 900 input x 10 + 100 output x 20 = 9_000 + 2_000 = 11_000 micro.
        assert_eq!(
            balance.spent_cost_micro, 11_000,
            "reconcile settles the EXACT usage, not the 60k estimate"
        );
    }

    /// Crash at EVERY seam of the completion sequence, reopen, and assert:
    /// the task STAYS Verifying, the accounting prefix is durable and
    /// idempotent, and the re-run converges to VerifiedComplete with the
    /// final invariant (zero open + uncertain, totals folded, revision proof
    /// exact).
    #[tokio::test]
    async fn crash_at_every_completion_seam_reopens_verifying_and_reruns_converge() {
        for seam in [
            CompletionCrashPoint::AfterProofLoad,
            CompletionCrashPoint::AfterReconcile,
            CompletionCrashPoint::AfterCostFold,
            CompletionCrashPoint::BeforeTransition,
        ] {
            let (dir, m) = test_manager();
            let s = session(&m);
            let sid = s.id;
            let (tid, rev, rec) = verifying_task_with_record(&s, Some(1_000_000));
            let ledger = ledger_for(&m);
            // Two crashed dispatched attempts: one with exact usage known,
            // one without.
            let logical = m.next_op_id();
            let exact = faktor_core::op::ModelCallAttempt::new(logical, m.next_op_id(), 0).unwrap();
            let r1 = ledger
                .reserve_attempt(s.id, tid, exact, 60_000, Some(snapshot()))
                .await
                .unwrap();
            ledger.mark_dispatched(s.id, r1).await.unwrap();
            s.record_provider_call_attempt(
                exact,
                Some(r1),
                "fake",
                "m",
                "completed",
                Some(900),
                Some(100),
                None,
            )
            .unwrap();
            ledger
                .mark_uncertain(s.id, r1, "crash_a".into(), None)
                .await
                .unwrap();
            let r2 = ledger
                .reserve(s.id, tid, m.next_op_id(), 7_000, None)
                .await
                .unwrap();
            ledger.mark_dispatched(s.id, r2).await.unwrap();
            ledger
                .mark_uncertain(s.id, r2, "crash_b".into(), None)
                .await
                .unwrap();
            // "Crash" at the seam: every step before it committed.
            let crashed = s
                .complete_verified_task_crashable(tid, rev, rec, Some(seam))
                .unwrap();
            assert_eq!(crashed, None, "the seam simulates process death");
            drop(s);
            drop(m);
            // Reopen: the task STAYS Verifying at the exact proof revision.
            let m2 =
                crate::SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                    .unwrap();
            let s2 = m2.get_session(sid).unwrap().unwrap();
            let row = s2.get_task(tid).unwrap().unwrap();
            assert_eq!(
                row.state,
                TaskState::Verifying,
                "no seam may transition before accounting is provably closed: {seam:?}"
            );
            assert_eq!(s2.task_revision(tid).unwrap(), rev);
            assert_eq!(
                s2.get_verification_record(rec).unwrap().unwrap().status,
                VerificationStatus::Passed
            );
            // The re-run converges: reconcile idempotent, finalize charges
            // only what reconcile left, the balance asserts zero, and the
            // transition CAS lands exactly once.
            let done = s2.complete_verified_task(tid, rev, rec).unwrap();
            assert_eq!(done.state, TaskState::VerifiedComplete, "{seam:?}");
            assert_eq!(
                s2.task_revision(tid).unwrap(),
                rev.checked_next().unwrap(),
                "the transition bumped exactly once: {seam:?}"
            );
            let balance = ledger_for(&m2)
                .completion_accounting_balance(sid, tid)
                .unwrap();
            assert!(
                balance.is_zero(),
                "VerifiedComplete => zero open + uncertain ({seam:?}): {balance:?}"
            );
            // Exact-usage attempt reconciled at 11_000; the usage-less
            // crashed attempt charged its 7_000 estimate.
            assert_eq!(balance.spent_cost_micro, 11_000 + 7_000, "{seam:?}");
            let _ = dir;
        }
    }

    // ------------------------------------------- typed criteria (56/57/105)

    #[test]
    fn typed_criteria_round_trip_and_revision_bump_on_change() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let tid = s.task_id().unwrap();
        s.create_task(criteria_task(&s, tid, vec!["goal: ship".into()]))
            .unwrap();
        let rev = s.task_revision(tid).unwrap();
        let criteria = vec![
            Criterion::user("goal: ship"),
            Criterion::derived(
                "required check: cargo check",
                CriterionOrigin::VerificationPolicy,
                CriterionRequirement::Required,
                Some("snap-1".into()),
            )
            .with_evidence(7),
            Criterion::derived(
                "no public API churn",
                CriterionOrigin::SemanticProvider,
                CriterionRequirement::Preferred,
                Some("provider-snap-9".into()),
            ),
        ];
        let updated = s.set_task_criteria(tid, criteria.clone()).unwrap();
        // The typed criteria ride the EXISTING row values as V2 JSON.
        assert!(
            updated
                .acceptance_criteria
                .iter()
                .all(|e| e.starts_with("v2:")),
            "{:?}",
            updated.acceptance_criteria
        );
        assert_eq!(s.task_criteria(tid).unwrap(), criteria, "round trip exact");
        assert_eq!(
            s.task_revision(tid).unwrap(),
            rev.checked_next().unwrap(),
            "a criterion change bumps the revision exactly once"
        );
        // Idempotent re-set: byte-identical, no bump.
        let again = s.set_task_criteria(tid, criteria.clone()).unwrap();
        assert_eq!(again, updated);
        assert_eq!(
            s.task_revision(tid).unwrap(),
            rev.checked_next().unwrap(),
            "an identical criteria set writes nothing"
        );
        // A metadata-only change (evidence binding) is content: it bumps.
        let mut with_evidence = criteria.clone();
        with_evidence[0].evidence_source = Some(42);
        let bumped = s.set_task_criteria(tid, with_evidence.clone()).unwrap();
        assert_eq!(bumped.acceptance_criteria, encode_criteria(&with_evidence));
        assert_eq!(
            s.task_revision(tid).unwrap(),
            rev.checked_next().unwrap().checked_next().unwrap()
        );
        // Hostile hand-crafted ids and duplicate sets are refused loudly
        // before any write.
        let mut hostile = criteria.clone();
        hostile[0].id = CriterionId::new(7);
        assert!(matches!(
            s.set_task_criteria(tid, hostile).unwrap_err(),
            TaskError::Malformed(_)
        ));
        let rev_before = s.task_revision(tid).unwrap();
        let mut dup = criteria.clone();
        dup.push(criteria[0].clone());
        assert!(matches!(
            s.set_task_criteria(tid, dup).unwrap_err(),
            TaskError::Malformed(_)
        ));
        let too_many: Vec<Criterion> = (0..=MAX_TASK_CRITERIA)
            .map(|i| Criterion::user(format!("criterion {i}")))
            .collect();
        assert!(matches!(
            s.set_task_criteria(tid, too_many).unwrap_err(),
            TaskError::Oversized(_)
        ));
        assert_eq!(s.task_revision(tid).unwrap(), rev_before, "no trace");
        // Escape-heavy text inside the text bound encodes beyond the entry
        // bound: refused loudly, never silently demoted to plain text.
        let mut escape_heavy = Criterion::user("goal: ok");
        escape_heavy.text = "\\".repeat(MAX_TASK_CRITERION_TEXT_BYTES);
        escape_heavy.id = CriterionId::for_content(
            escape_heavy.origin,
            escape_heavy.requirement,
            &escape_heavy.text,
            None,
        );
        assert!(matches!(
            s.set_task_criteria(tid, vec![escape_heavy]).unwrap_err(),
            TaskError::Oversized(_)
        ));
        assert_eq!(s.task_revision(tid).unwrap(), rev_before, "no trace");
    }

    #[test]
    fn legacy_string_migration_is_stable_and_coverage_still_validates() {
        // (a) An untouched legacy row keeps completing: the record keys are
        // the raw legacy strings the row carries.
        let (_d, m) = test_manager();
        let s = session(&m);
        let tid = s.task_id().unwrap();
        let legacy = vec![
            "goal: gated goal".to_string(),
            "required check: cargo check".to_string(),
        ];
        s.create_task(criteria_task(&s, tid, legacy.clone()))
            .unwrap();
        let first = s.task_criteria(tid).unwrap();
        let second = s.task_criteria(tid).unwrap();
        assert_eq!(first, second, "repeated legacy reads are identical");
        assert_eq!(first[0].origin, CriterionOrigin::User);
        assert_eq!(first[1].origin, CriterionOrigin::ProjectPolicy);
        for criterion in &first {
            criterion.validate().unwrap();
        }
        let rev = drive_to_verifying(&s, tid);
        let record = passed_record(&s, tid, &legacy);
        let done = s.complete_verified_task(tid, rev, record).unwrap();
        assert_eq!(done.state, TaskState::VerifiedComplete);

        // (a2) A legacy entry that cannot fit the V2 envelope stays plain
        // (lossless) instead of being truncated or demoted with data loss.
        let s_big = session(&m);
        let tid_big = s_big.task_id().unwrap();
        let long = "x".repeat(MAX_TASK_CRITERION_BYTES);
        s_big
            .create_task(criteria_task(&s_big, tid_big, vec![long.clone()]))
            .unwrap();
        let decoded = s_big.task_criteria(tid_big).unwrap();
        assert_eq!(decoded[0].text, long, "the over-bound text is preserved");
        assert_eq!(
            encode_criteria(&decoded),
            vec![long],
            "an over-bound legacy criterion keeps its plain representation"
        );

        // (b) The migrated (V2) row completes against a record keyed by the
        // migrated row values — coverage never silently drifts.
        let s2 = session(&m);
        let tid2 = s2.task_id().unwrap();
        s2.create_task(criteria_task(&s2, tid2, legacy.clone()))
            .unwrap();
        let migrated = s2.task_criteria(tid2).unwrap();
        let migrated_row = s2.set_task_criteria(tid2, migrated.clone()).unwrap();
        let rev2 = drive_to_verifying(&s2, tid2);
        let record2 = passed_record(&s2, tid2, &migrated_row.acceptance_criteria);
        let done2 = s2.complete_verified_task(tid2, rev2, record2).unwrap();
        assert_eq!(done2.state, TaskState::VerifiedComplete);
    }

    #[test]
    fn user_criteria_survive_rederivation_and_stale_snapshot_rederives() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let tid = s.task_id().unwrap();
        s.create_task(criteria_task(&s, tid, vec![])).unwrap();
        let v1 = vec![
            Criterion::user("goal: first"),
            Criterion::derived(
                "required check: cargo check",
                CriterionOrigin::ProjectPolicy,
                CriterionRequirement::Required,
                Some("checks:v1".into()),
            ),
        ];
        s.rederive_task_criteria(tid, v1.clone()).unwrap();
        let rev1 = s.task_revision(tid).unwrap();
        // The source snapshot moved: the derived criterion is re-derived
        // (new content id, new snapshot), a new user goal joins, the ORIGINAL
        // user criterion survives verbatim.
        let v2 = vec![
            Criterion::user("goal: second"),
            Criterion::derived(
                "required check: cargo check",
                CriterionOrigin::ProjectPolicy,
                CriterionRequirement::Required,
                Some("checks:v2".into()),
            ),
            Criterion::derived(
                "required check: cargo test",
                CriterionOrigin::ProjectPolicy,
                CriterionRequirement::Required,
                Some("checks:v2".into()),
            ),
        ];
        let row = s.rederive_task_criteria(tid, v2.clone()).unwrap();
        assert!(s.task_revision(tid).unwrap() != rev1);
        let criteria = s.task_criteria(tid).unwrap();
        assert_eq!(criteria.len(), 4, "{criteria:?}");
        let sticky = criteria
            .iter()
            .find(|c| c.text == "goal: first")
            .expect("the user criterion survives re-derivation");
        assert_eq!(sticky.id, v1[0].id);
        assert!(criteria.iter().any(|c| c.text == "goal: second"));
        let check = criteria
            .iter()
            .find(|c| c.text == "required check: cargo check")
            .unwrap();
        assert_eq!(check.semantic_snapshot.as_deref(), Some("checks:v2"));
        assert_ne!(check.id, v1[1].id, "the stale snapshot re-derived");
        assert!(criteria
            .iter()
            .any(|c| c.text == "required check: cargo test"));
        // Re-running the SAME derivation is idempotent.
        let rev2 = s.task_revision(tid).unwrap();
        let again = s.rederive_task_criteria(tid, v2).unwrap();
        assert_eq!(again.acceptance_criteria, row.acceptance_criteria);
        assert_eq!(s.task_revision(tid).unwrap(), rev2);
    }

    #[test]
    fn criterion_change_invalidates_prior_verification() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let tid = s.task_id().unwrap();
        s.create_task(criteria_task(&s, tid, vec!["goal: gated goal".into()]))
            .unwrap();
        let certified_rev = drive_to_verifying(&s, tid);
        let legacy_keys = s.get_task(tid).unwrap().unwrap().acceptance_criteria;
        let stale_record = passed_record(&s, tid, &legacy_keys);
        // A criterion change (here: adding a user criterion) bumps the row.
        s.set_task_criteria(
            tid,
            vec![
                Criterion::user("goal: gated goal"),
                Criterion::user("the new seam must be named"),
            ],
        )
        .unwrap();
        let moved = s.task_revision(tid).unwrap();
        assert!(moved != certified_rev);
        // The prior PASSING record pins the old revision: completion is
        // refused typed, never silently certified.
        let err = s
            .complete_verified_task(tid, certified_rev, stale_record)
            .unwrap_err();
        assert!(
            matches!(err, TaskError::RevisionMismatch { .. }),
            "criterion change must invalidate the prior verification: {err:?}"
        );
        assert_eq!(
            s.get_task(tid).unwrap().unwrap().state,
            TaskState::Verifying
        );
        // A fresh record certifying the CURRENT row completes.
        let current_keys = s.get_task(tid).unwrap().unwrap().acceptance_criteria;
        let fresh = passed_record(&s, tid, &current_keys);
        let done = s.complete_verified_task(tid, moved, fresh).unwrap();
        assert_eq!(done.state, TaskState::VerifiedComplete);
    }
}
