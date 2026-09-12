//! Durable rows for orchestrated child agents (audits 20-24 wiring).
//!
//! Everything a child drive needs to be crash-safe lives as durable
//! memory-fact rows scoped to the CHILD session (the drive side reads only
//! its own session's rows) plus one registry row per child scoped to the
//! PARENT session (the executor side enumerates its children from the
//! parent's row space — restart re-attaches through the parent rows, never
//! through a process-local map).
//!
//! Row layout (all values are bounded JSON; the memory-fact store enforces
//! the 4096-byte value cap and the 64-byte kind cap):
//!
//! - child session row space
//!   - kind `orchestrator`, key `identity`:
//!     [`ChildIdentity`] — parent session, workspace/worktree ids, the work
//!     item id, ownership mode, model policy and the child's operation id.
//!   - kind `orchestrator`, key `drive_state`:
//!     [`DriveState`] — the drive-side phase (`running` | `waiting`), the
//!     current steering note and the current model override. Written by the
//!     drive hook; read by the executor mirror and by re-attach.
//!   - kind `orchestrator_ctl`, key = `seq-<seq>`:
//!     one [`ControlRow`] — the durable control queue.
//!
//! - parent session row space
//!   - kind `orchestrator_registry`, key = child session id:
//!     the executor's durable [`ChildRow`] (ownership, worktree, state,
//!     budget, effective capability set, model policy) — the zero-orphan
//!     anchor: every registered worktree of an executor-created child is
//!     referenced from exactly one child row in this registry and every
//!     registry row names a live session row.
//!
//! Exactly-once control application: a control row carries
//! `applied_ms: NULL` until the applier acks it; [`SessionHandle::orchestrator_ctl_ack`]
//! is idempotent (the FIRST ack's timestamp sticks). Effects are ordered
//! so that a crash between effect and ack is harmless (the effect is
//! re-applied idempotently) and a crash between ack and effect is harmless
//! too (the row is acked, the effect row — waiting phase, current note,
//! current model — is the durable truth the drive re-reads).

use faktor_core::id::SessionId;
use serde::{Deserialize, Serialize};

use crate::handle::SessionHandle;
use crate::SessionError;

/// Hard cap on the durable control queue per child (bounded everything).
pub const MAX_CHILD_CONTROL_ROWS: usize = 128;
/// Hard cap on the durable control note payload (chars), matching the
/// orchestrator's steering-note bound.
pub const MAX_CHILD_CONTROL_NOTE_CHARS: usize = 500;
/// Hard cap on a model selector stored in a control row.
pub const MAX_CHILD_CONTROL_MODEL_CHARS: usize = 128;
/// Rows are JSON values under these kinds; keys must stay bounded.
const ROW_KIND_IDENTITY: &str = "orchestrator";
const KEY_IDENTITY: &str = "identity";
const KEY_DRIVE_STATE: &str = "drive_state";
const ROW_KIND_CONTROL: &str = "orchestrator_ctl";
/// A memory-fact scan walks pages of this size and stops after
/// [`MAX_FACT_SCAN_PAGES`] pages (a hostile row space cannot stall a drive
/// boundary with an unbounded scan).
const FACT_PAGE: i64 = 200;
const MAX_FACT_SCAN_PAGES: usize = 25;

/// Ownership mode of a child, durably recorded on the child row and the
/// child identity. Typed — never a free-form string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChildOwnership {
    /// Read-only child sharing the parent's worktree.
    ReadOnlyShared,
    /// Mutating child with an EXPLICIT disjoint normalized path set inside
    /// the parent worktree.
    ExclusivePaths,
    /// Mutating child on its own isolated worktree/workspace.
    IsolatedWorktree,
}

/// The durable identity of one child session, stored in the child's own row
/// space (written once at creation, extended after the first submit).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ChildIdentity {
    /// The orchestrator session that spawned this child.
    pub parent_session_id: SessionId,
    pub workspace_id: u64,
    pub worktree_id: u64,
    /// Work item id inside the parent plan.
    pub item_id: String,
    /// The task goal this child drives toward (also its initial prompt).
    pub task_goal: String,
    /// The child's durable turn operation id once submitted (0 before).
    pub operation_id: u64,
    pub ownership: ChildOwnership,
    /// The effective model the daemon should select for this child
    /// (empty = session default).
    pub model: String,
    pub created_ms: i64,
}

impl Default for ChildIdentity {
    fn default() -> Self {
        Self {
            parent_session_id: SessionId::new(1),
            workspace_id: 1,
            worktree_id: 1,
            item_id: String::new(),
            task_goal: String::new(),
            operation_id: 0,
            ownership: ChildOwnership::ReadOnlyShared,
            model: String::new(),
            created_ms: 0,
        }
    }
}

/// Drive-side phase of an orchestrated child (payload-tagged additive state:
/// no new journal `EventKind` variant, so protocol golden fixtures stay
/// untouched).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChildPhase {
    #[default]
    Running,
    /// The drive yielded at a safe reasoning boundary and is parked until a
    /// Resume control arrives.
    Waiting,
}

/// Durable drive-side state written by the boundary hook and read by the
/// executor mirror.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct DriveState {
    pub phase: ChildPhase,
    /// The currently applied steering note (empty = none); survives crashes.
    pub current_note: String,
    /// The currently applied model override (empty = session default).
    pub current_model: String,
    pub updated_ms: i64,
}

/// Strict bounds of the durable child blocker text (v23 typed projection).
pub const MAX_CHILD_BLOCKER_KIND_CHARS: usize = 64;
pub const MAX_CHILD_BLOCKER_REASON_CHARS: usize = 512;
pub const MAX_CHILD_BLOCKER_DEPENDENCY_CHARS: usize = 64;
pub const MAX_CHILD_BLOCKER_RESOLUTION_CHARS: usize = 512;

/// The typed durable blocker of one blocked child: WHY it is not running.
/// Bounded and validated — hostile oversized text is a typed reject before
/// any durable write.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ChildBlocker {
    pub kind: String,
    pub reason: String,
    pub dependency: Option<String>,
    pub resolution: Option<String>,
    pub last_progress_ms: Option<i64>,
}

impl ChildBlocker {
    pub fn validate(&self) -> Result<(), SessionError> {
        check_blocker_text("blocker kind", &self.kind, MAX_CHILD_BLOCKER_KIND_CHARS)?;
        check_blocker_text(
            "blocker reason",
            &self.reason,
            MAX_CHILD_BLOCKER_REASON_CHARS,
        )?;
        if let Some(dep) = &self.dependency {
            check_blocker_text(
                "blocker dependency",
                dep,
                MAX_CHILD_BLOCKER_DEPENDENCY_CHARS,
            )?;
        }
        if let Some(res) = &self.resolution {
            check_blocker_text(
                "blocker resolution",
                res,
                MAX_CHILD_BLOCKER_RESOLUTION_CHARS,
            )?;
        }
        if self.last_progress_ms.is_some_and(|ms| ms < 0) {
            return Err(SessionError::Malformed(
                "blocker last_progress_ms must be non-negative".into(),
            ));
        }
        Ok(())
    }
}

fn check_blocker_text(field: &str, value: &str, max: usize) -> Result<(), SessionError> {
    if value.trim().is_empty() {
        return Err(SessionError::Malformed(format!(
            "{field} must not be empty or whitespace-only"
        )));
    }
    if value.chars().count() > max {
        return Err(SessionError::Oversized(format!(
            "{field} of {} characters exceeds {max}",
            value.chars().count()
        )));
    }
    if value.chars().any(|c| c.is_control()) {
        return Err(SessionError::Malformed(format!(
            "{field} carries control characters"
        )));
    }
    Ok(())
}

/// One durable child-runtime projection row (v23 `child_runtime` table):
/// the child's durable state and blocker truth, keyed by the child session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChildRuntimeBlockerRow {
    pub child_id: String,
    pub state: String,
    pub blocker: Option<ChildBlocker>,
    pub updated_ms: i64,
}

impl ChildRuntimeBlockerRow {
    pub fn validate(&self) -> Result<(), SessionError> {
        if self.child_id.is_empty() || self.child_id.len() > 64 {
            return Err(SessionError::Malformed(
                "child_id must be 1..=64 bytes".into(),
            ));
        }
        if self.state.is_empty() || self.state.len() > 32 {
            return Err(SessionError::Malformed("state must be 1..=32 bytes".into()));
        }
        if let Some(b) = &self.blocker {
            b.validate()?;
        }
        Ok(())
    }
}

/// One durable control message of a child.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ChildControl {
    Pause,
    Resume,
    Cancel,
    Steer { note: String },
    Retry,
    ChangeModel { model: String },
    ChangeBudget { max_tokens: u64 },
}

/// The TYPED budget-change vocabulary of a child (audit 9/H): the historic
/// single `ChildControl::ChangeBudget { max_tokens }` wire is split into two
/// typed changes — token capacity ([`ChildBudgetChange::ChangeTokenBudget`])
/// and monetary capacity ([`ChildBudgetChange::ChangeCostBudget`], in
/// microUSD). The vocabulary is ADDITIVE and deliberately separate from
/// [`ChildControl`]: the frozen drive-boundary applier (agent runtime) and
/// the executor's synchronous effects (orchestrator runtime) match
/// `ChildControl` exhaustively and cannot change shape in this revision, so
/// the typed split lives on the surfaces this layer owns — the native
/// budget-change endpoint maps its one-of body onto these variants and
/// applies each as a synchronous durable effect:
///
/// - `ChangeTokenBudget` → the existing token path (the child's durable
///   task-row `max_tokens` patch, delivered exactly once through the
///   `ChangeBudget` control queue);
/// - `ChangeCostBudget` → the durable cost-cap effect (the child's task-row
///   `max_cost_micro` + its budget subscope rows, see
///   [`crate::budget::DurableBudgetLedger::change_child_scope_cap`]).
///
/// Serde shape is tagged + snake_case (`kind: "change_token_budget"`,
/// `"change_cost_budget"`), mirroring the camelCase native DTOs one-for-one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ChildBudgetChange {
    ChangeTokenBudget { max_tokens: u64 },
    ChangeCostBudget { max_cost_micro: u64 },
}

impl ChildBudgetChange {
    /// Structural validation (bounded everything): token budgets are
    /// 1..=u64, monetary budgets 1..=u64 — 0 is refused as ambiguous on both
    /// axes (the store treats a NULL/0 cap as "unlimited"; an explicit
    /// change that means "remove the cap" must go through the None-cap
    /// surface, never a silent 0).
    pub fn validate(&self) -> Result<(), SessionError> {
        match self {
            ChildBudgetChange::ChangeTokenBudget { max_tokens } if *max_tokens == 0 => {
                Err(SessionError::Malformed(
                    "max_tokens must be >= 1 (0 is ambiguous; a missing key means unlimited)"
                        .into(),
                ))
            }
            ChildBudgetChange::ChangeCostBudget { max_cost_micro } if *max_cost_micro == 0 => {
                Err(SessionError::Malformed(
                    "max_cost_micro must be >= 1 (0 is ambiguous; a missing key means unlimited)"
                        .into(),
                ))
            }
            _ => Ok(()),
        }
    }
}

/// The durable control row (audit 23 shape: seq, kind, created_ms,
/// applied_ms NULL until applied exactly once).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlRow {
    pub seq: u64,
    pub control: ChildControl,
    pub created_ms: i64,
    pub applied_ms: Option<i64>,
}

impl ControlRow {
    pub fn applied(&self) -> bool {
        self.applied_ms.is_some()
    }
}

fn ctl_key(seq: u64) -> String {
    format!("seq-{seq:016x}")
}

fn parse_ctl_key(key: &str) -> Option<u64> {
    key.strip_prefix("seq-")
        .and_then(|hex| u64::from_str_radix(hex, 16).ok())
}

/// Collect every memory fact of `handle` with a bounded page scan.
fn all_facts(handle: &SessionHandle) -> faktor_core::Result<Vec<(String, String, String)>> {
    let mut out = Vec::new();
    let mut after: Option<(i64, String, String)> = None;
    for _ in 0..MAX_FACT_SCAN_PAGES {
        let page = handle.memory_facts_page(after.as_ref(), FACT_PAGE)?;
        out.extend(page.facts);
        match page.cursor {
            Some(c) => after = Some(c),
            None => break,
        }
    }
    if out.is_empty() {
        return Ok(out);
    }
    // A scan that hit the page cap without an end is a hostile/broken row
    // space: loud, never a silent truncation of control rows.
    let total = out.len();
    let ended = handle
        .memory_facts_page(after.as_ref(), FACT_PAGE)?
        .cursor
        .is_none();
    if !ended {
        return Err(SessionError::Internal(format!(
            "memory-fact scan of session {} exceeded {MAX_FACT_SCAN_PAGES} pages ({total} rows); refusing a partial control read",
            handle.id()
        ))
        .into());
    }
    Ok(out)
}

/// Public, bounded control-queue and child-row surface (session layer,
/// additive).
impl SessionHandle {
    /// Read one value row by kind/key from this session's fact space.
    fn fact_get(&self, kind: &str, key: &str) -> faktor_core::Result<Option<String>> {
        for (k, kk, v) in all_facts(self)? {
            if k == kind && kk == key {
                return Ok(Some(v));
            }
        }
        Ok(None)
    }

    // ------------------------------------------------------- child identity

    /// Durable (parent, worktree, item) identity of this child session.
    pub fn orchestrator_child_identity_put(&self, id: &ChildIdentity) -> faktor_core::Result<()> {
        if id.item_id.len() > 64 || id.task_goal.len() > 2000 {
            return Err(
                SessionError::Oversized("child identity item/goal exceed bounds".into()).into(),
            );
        }
        if id.model.len() > MAX_CHILD_CONTROL_MODEL_CHARS {
            return Err(
                SessionError::Oversized("child identity model exceeds bound".into()).into(),
            );
        }
        let value = serde_json::to_string(id)
            .map_err(|e| SessionError::Internal(format!("child identity serialization: {e}")))?;
        self.upsert_memory_fact(ROW_KIND_IDENTITY, KEY_IDENTITY, &value)
    }

    /// The durable identity row; `None` when this session is not an
    /// orchestrated child.
    pub fn orchestrator_child_identity_get(&self) -> faktor_core::Result<Option<ChildIdentity>> {
        let Some(raw) = self.fact_get(ROW_KIND_IDENTITY, KEY_IDENTITY)? else {
            return Ok(None);
        };
        serde_json::from_str(&raw)
            .map(Some)
            .map_err(|e| SessionError::Internal(format!("child identity decode: {e}")).into())
    }

    // ---------------------------------------------------------- drive state

    pub fn orchestrator_drive_state_put(&self, state: &DriveState) -> faktor_core::Result<()> {
        let value = serde_json::to_string(state)
            .map_err(|e| SessionError::Internal(format!("drive state serialization: {e}")))?;
        self.upsert_memory_fact(ROW_KIND_IDENTITY, KEY_DRIVE_STATE, &value)
    }

    pub fn orchestrator_drive_state_get(&self) -> faktor_core::Result<DriveState> {
        let Some(raw) = self.fact_get(ROW_KIND_IDENTITY, KEY_DRIVE_STATE)? else {
            return Ok(DriveState {
                phase: ChildPhase::Running,
                ..Default::default()
            });
        };
        serde_json::from_str(&raw)
            .map_err(|e| SessionError::Internal(format!("drive state decode: {e}")).into())
    }

    // ------------------------------------------ durable blocker truth (v23)

    /// Persist the typed child-runtime projection row (state + blocker) of
    /// this child session. Strict bounds are validated BEFORE any durable
    /// write; a hostile oversized blocker is a typed reject, never a
    /// truncation.
    pub fn orchestrator_child_runtime_put(
        &self,
        row: &ChildRuntimeBlockerRow,
    ) -> faktor_core::Result<()> {
        row.validate()?;
        let store_row = faktor_store::ChildRuntimeRow {
            session_id: self.id,
            child_id: row.child_id.clone(),
            state: row.state.clone(),
            blocker_kind: row.blocker.as_ref().map(|b| b.kind.clone()),
            blocker_reason: row.blocker.as_ref().map(|b| b.reason.clone()),
            blocker_dependency: row.blocker.as_ref().and_then(|b| b.dependency.clone()),
            blocker_resolution: row.blocker.as_ref().and_then(|b| b.resolution.clone()),
            last_progress_ms: row.blocker.as_ref().and_then(|b| b.last_progress_ms),
            updated_ms: if row.updated_ms > 0 {
                row.updated_ms
            } else {
                self.now_ms()
            },
        };
        self.manager
            .store()
            .child_runtime_put(&store_row)
            .map_err(crate::map_store_err)?;
        Ok(())
    }

    /// The typed child-runtime projection row of this session (None before
    /// the first write).
    pub fn orchestrator_child_runtime_get(
        &self,
    ) -> faktor_core::Result<Option<ChildRuntimeBlockerRow>> {
        let Some(row) = self
            .manager
            .store()
            .child_runtime_get(self.id)
            .map_err(crate::map_store_err)?
        else {
            return Ok(None);
        };
        let blocker = row.blocker_kind.map(|kind| ChildBlocker {
            kind,
            reason: row.blocker_reason.unwrap_or_default(),
            dependency: row.blocker_dependency,
            resolution: row.blocker_resolution,
            last_progress_ms: row.last_progress_ms,
        });
        Ok(Some(ChildRuntimeBlockerRow {
            child_id: row.child_id,
            state: row.state,
            blocker,
            updated_ms: row.updated_ms,
        }))
    }

    /// Mark this child BLOCKED with the given blocker (validated, bounded).
    pub fn orchestrator_child_blocker_put(
        &self,
        child_id: &str,
        blocker: &ChildBlocker,
    ) -> faktor_core::Result<()> {
        self.orchestrator_child_runtime_put(&ChildRuntimeBlockerRow {
            child_id: child_id.to_string(),
            state: "blocked".into(),
            blocker: Some(blocker.clone()),
            updated_ms: self.now_ms(),
        })
    }

    /// Drop this child's durable blocker truth (a transition to Running).
    pub fn orchestrator_child_blocker_clear(&self) -> faktor_core::Result<()> {
        self.manager
            .store()
            .child_runtime_delete(self.id)
            .map_err(crate::map_store_err)?;
        Ok(())
    }

    /// The durable blocker of this child, when one is recorded.
    pub fn orchestrator_child_blocker_get(&self) -> faktor_core::Result<Option<ChildBlocker>> {
        Ok(self
            .orchestrator_child_runtime_get()?
            .and_then(|r| r.blocker))
    }

    // -------------------------------------------------------- control queue

    /// Read every durable control row of this child, oldest first.
    pub fn orchestrator_ctl_all(&self) -> faktor_core::Result<Vec<ControlRow>> {
        let mut rows = Vec::new();
        for (kind, key, value) in all_facts(self)? {
            if kind != ROW_KIND_CONTROL {
                continue;
            }
            let Some(seq) = parse_ctl_key(&key) else {
                return Err(SessionError::Internal(format!(
                    "child control row with hostile key {key:?} on session {}",
                    self.id()
                ))
                .into());
            };
            let row: ControlRow = serde_json::from_str(&value).map_err(|e| {
                SessionError::Internal(format!("child control row {seq} decode: {e}"))
            })?;
            if row.seq != seq {
                return Err(SessionError::Internal(format!(
                    "child control row key/seq mismatch ({key:?} vs {})",
                    row.seq
                ))
                .into());
            }
            rows.push(row);
        }
        rows.sort_by_key(|r| r.seq);
        Ok(rows)
    }

    /// The unapplied control rows (audit 23 delivery queue), oldest first.
    pub fn orchestrator_ctl_pending(&self) -> faktor_core::Result<Vec<ControlRow>> {
        Ok(self
            .orchestrator_ctl_all()?
            .into_iter()
            .filter(|r| !r.applied())
            .collect())
    }

    /// Enqueue one durable control message. Bounded: at most
    /// [`MAX_CHILD_CONTROL_ROWS`] rows exist per child; the queue is capped
    /// (the executor is responsible for acking consumed rows).
    pub fn orchestrator_ctl_enqueue(
        &self,
        control: ChildControl,
    ) -> faktor_core::Result<ControlRow> {
        match &control {
            ChildControl::Steer { note } if note.chars().count() > MAX_CHILD_CONTROL_NOTE_CHARS => {
                return Err(SessionError::Oversized(format!(
                    "steering note exceeds {MAX_CHILD_CONTROL_NOTE_CHARS} characters"
                ))
                .into());
            }
            ChildControl::ChangeModel { model }
                if model.is_empty() || model.chars().count() > MAX_CHILD_CONTROL_MODEL_CHARS =>
            {
                return Err(SessionError::Oversized(format!(
                    "model selector must be 1..={MAX_CHILD_CONTROL_MODEL_CHARS} characters"
                ))
                .into());
            }
            _ => {}
        }
        let existing = self.orchestrator_ctl_all()?;
        if existing.len() >= MAX_CHILD_CONTROL_ROWS {
            return Err(SessionError::Conflict(format!(
                "child control queue of session {} holds {} rows (cap {MAX_CHILD_CONTROL_ROWS}); ack before enqueueing more",
                self.id(),
                existing.len()
            ))
            .into());
        }
        let seq = existing.last().map(|r| r.seq).unwrap_or(0) + 1;
        if seq > 0x0000_ffff_ffff_ffff {
            return Err(SessionError::Oversized("control queue seq overflow".into()).into());
        }
        let row = ControlRow {
            seq,
            control,
            created_ms: self.now_ms(),
            applied_ms: None,
        };
        let value = serde_json::to_string(&row)
            .map_err(|e| SessionError::Internal(format!("control row serialization: {e}")))?;
        self.upsert_memory_fact(ROW_KIND_CONTROL, &ctl_key(seq), &value)?;
        Ok(row)
    }

    /// Ack one control row. Idempotent: the FIRST ack's `applied_ms`
    /// sticks — a re-ack after a crash never rewrites the timestamp, so a
    /// restart mid-queue applies each message exactly once.
    pub fn orchestrator_ctl_ack(&self, seq: u64) -> faktor_core::Result<()> {
        let rows = self.orchestrator_ctl_all()?;
        let Some(mut row) = rows.into_iter().find(|r| r.seq == seq) else {
            return Err(SessionError::NotFound(format!(
                "control row {seq} of session {}",
                self.id()
            ))
            .into());
        };
        if row.applied_ms.is_none() {
            row.applied_ms = Some(self.now_ms());
            let value = serde_json::to_string(&row)
                .map_err(|e| SessionError::Internal(format!("control row serialization: {e}")))?;
            self.upsert_memory_fact(ROW_KIND_CONTROL, &ctl_key(seq), &value)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SessionManager;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn fixture() -> (tempfile::TempDir, Arc<SessionManager>, SessionHandle) {
        let dir = tempdir().unwrap();
        let m =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let ws = m.create_workspace("/root").unwrap();
        let s = m.create_session(ws, "child", "fake", "m").unwrap();
        (dir, m, s)
    }

    #[test]
    fn enqueue_assigns_monotonic_seq_and_roundtrips() {
        let (_d, _m, s) = fixture();
        let a = s.orchestrator_ctl_enqueue(ChildControl::Pause).unwrap();
        let b = s
            .orchestrator_ctl_enqueue(ChildControl::Steer {
                note: "go slow".into(),
            })
            .unwrap();
        assert_eq!(a.seq, 1);
        assert_eq!(b.seq, 2);
        let rows = s.orchestrator_ctl_all().unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].control, ChildControl::Pause);
        assert_eq!(
            rows[1].control,
            ChildControl::Steer {
                note: "go slow".into()
            }
        );
        assert!(rows[0].applied_ms.is_none());
        assert_eq!(s.orchestrator_ctl_pending().unwrap().len(), 2);
    }

    #[test]
    fn hostile_and_oversized_inputs_rejected_before_write() {
        let (_d, _m, s) = fixture();
        let long_note = "n".repeat(MAX_CHILD_CONTROL_NOTE_CHARS + 1);
        let err = s
            .orchestrator_ctl_enqueue(ChildControl::Steer { note: long_note })
            .unwrap_err();
        assert_eq!(err.kind, faktor_core::error::ErrorKind::Oversized);
        assert!(s.orchestrator_ctl_all().unwrap().is_empty());
        assert!(s
            .orchestrator_ctl_enqueue(ChildControl::ChangeModel {
                model: String::new()
            })
            .is_err());
        assert!(s
            .orchestrator_ctl_enqueue(ChildControl::ChangeModel {
                model: "x".repeat(129)
            })
            .is_err());
        assert!(s.orchestrator_ctl_all().unwrap().is_empty());
    }

    #[test]
    fn ack_is_exactly_once_and_sticks_across_reopen() {
        let dir = tempdir().unwrap();
        let (sid, seq) = {
            let m = SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                .unwrap();
            let ws = m.create_workspace("/root").unwrap();
            let s = m.create_session(ws, "child", "fake", "m").unwrap();
            let row = s
                .orchestrator_ctl_enqueue(ChildControl::Steer { note: "x".into() })
                .unwrap();
            s.orchestrator_ctl_ack(row.seq).unwrap();
            s.orchestrator_ctl_ack(row.seq).unwrap();
            let rows = s.orchestrator_ctl_all().unwrap();
            assert_eq!(rows.len(), 1);
            let first = rows[0].applied_ms;
            assert!(first.is_some());
            (s.id(), row.seq)
        };
        // Reopen: the row is durable and the ack timestamp was never
        // rewritten (applied exactly once).
        let m =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let s = m.get_session(sid).unwrap().unwrap();
        s.orchestrator_ctl_ack(seq).unwrap();
        let rows = s.orchestrator_ctl_all().unwrap();
        assert_eq!(rows[0].seq, seq);
        assert!(rows[0].applied_ms.is_some());
        assert!(s.orchestrator_ctl_pending().unwrap().is_empty());
    }

    #[test]
    fn identity_and_drive_state_survive_reopen() {
        let dir = tempdir().unwrap();
        let (sid, _parent, wt) = {
            let m = SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                .unwrap();
            let ws = m.create_workspace("/root").unwrap();
            let s = m.create_session(ws, "child", "fake", "m").unwrap();
            let id = ChildIdentity {
                parent_session_id: SessionId::new(9),
                workspace_id: ws.raw(),
                worktree_id: 7,
                item_id: "impl-a".into(),
                task_goal: "do the thing".into(),
                operation_id: 42,
                ownership: ChildOwnership::IsolatedWorktree,
                model: "m2".into(),
                created_ms: 1,
            };
            s.orchestrator_child_identity_put(&id).unwrap();
            s.orchestrator_drive_state_put(&DriveState {
                phase: ChildPhase::Waiting,
                current_note: "slower".into(),
                current_model: String::new(),
                updated_ms: 2,
            })
            .unwrap();
            let wt = id.worktree_id;
            (s.id(), id, wt)
        };
        let m =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let s = m.get_session(sid).unwrap().unwrap();
        let id = s.orchestrator_child_identity_get().unwrap().unwrap();
        assert_eq!(id.parent_session_id, SessionId::new(9));
        assert_eq!(id.worktree_id, wt);
        assert_eq!(id.item_id, "impl-a");
        assert_eq!(id.operation_id, 42);
        let state = s.orchestrator_drive_state_get().unwrap();
        assert_eq!(state.phase, ChildPhase::Waiting);
        assert_eq!(state.current_note, "slower");
    }

    #[test]
    fn queue_is_capped_and_hostile_keys_fail_loud() {
        let (_d, _m, s) = fixture();
        for _ in 0..MAX_CHILD_CONTROL_ROWS {
            s.orchestrator_ctl_enqueue(ChildControl::Pause).unwrap();
        }
        let err = s.orchestrator_ctl_enqueue(ChildControl::Pause).unwrap_err();
        assert_eq!(err.kind, faktor_core::error::ErrorKind::Conflict);
        assert!(err.message.contains("cap"), "{}", err.message);
        // Unknown seqs are not found.
        assert!(s.orchestrator_ctl_ack(9999).is_err());
    }

    #[test]
    fn non_child_sessions_read_empty_rows() {
        let (_d, _m, s) = fixture();
        assert!(s.orchestrator_child_identity_get().unwrap().is_none());
        assert_eq!(
            s.orchestrator_drive_state_get().unwrap().phase,
            ChildPhase::Running
        );
        assert!(s.orchestrator_ctl_pending().unwrap().is_empty());
    }

    #[test]
    fn budget_change_wire_split_is_typed_and_hostile_inputs_fail_loud() {
        // The typed token/cost split keeps its own tagged snake_case wire and
        // rejects hostile kinds, missing fields and ambiguous zero caps.
        let token = ChildBudgetChange::ChangeTokenBudget { max_tokens: 5 };
        let cost = ChildBudgetChange::ChangeCostBudget {
            max_cost_micro: 500_000,
        };
        let token_json = serde_json::to_string(&token).unwrap();
        assert_eq!(
            token_json,
            r#"{"kind":"change_token_budget","max_tokens":5}"#
        );
        assert_eq!(
            serde_json::to_string(&cost).unwrap(),
            r#"{"kind":"change_cost_budget","max_cost_micro":500000}"#
        );
        assert_eq!(token, serde_json::from_str(&token_json).unwrap());
        // A hostile body under the tag is a typed decode failure, never a
        // silent fallback to the other budget axis.
        for hostile in [
            r#"{"kind":"change_budget","max_tokens":5}"#,
            r#"{"kind":"change_cost_budget"}"#,
            r#"{"kind":"change_token_budget","max_cost_micro":5}"#,
            r#"{"kind":"steer","note":"x"}"#,
            r#"{"kind":"change_token_budget","max_tokens":"5"}"#,
        ] {
            assert!(
                serde_json::from_str::<ChildBudgetChange>(hostile).is_err(),
                "{hostile}"
            );
        }
        // Zero caps are refused typed on both axes (0 = unlimited in the
        // store; an explicit "remove the cap" is a different surface).
        assert!(ChildBudgetChange::ChangeTokenBudget { max_tokens: 0 }
            .validate()
            .is_err());
        assert!(ChildBudgetChange::ChangeCostBudget { max_cost_micro: 0 }
            .validate()
            .is_err());
        assert!(ChildBudgetChange::ChangeCostBudget { max_cost_micro: 1 }
            .validate()
            .is_ok());
    }

    #[test]
    fn durable_child_blocker_roundtrips_bounds_hostile_text_and_clears() {
        let dir = tempdir().unwrap();
        let sid = {
            let m = SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                .unwrap();
            let ws = m.create_workspace("/root").unwrap();
            let s = m.create_session(ws, "child", "fake", "m").unwrap();
            let blocker = ChildBlocker {
                kind: "budget".into(),
                reason: "task budget exhausted: max_tokens=1 spent_tokens=5".into(),
                dependency: None,
                resolution: Some("raise the child budget, then resume".into()),
                last_progress_ms: Some(7),
            };
            s.orchestrator_child_blocker_put("child-0", &blocker)
                .unwrap();
            assert_eq!(
                s.orchestrator_child_blocker_get().unwrap().unwrap(),
                blocker
            );
            let row = s.orchestrator_child_runtime_get().unwrap().unwrap();
            assert_eq!(row.child_id, "child-0");
            assert_eq!(row.state, "blocked");
            // Hostile oversized text is typed-rejected BEFORE any write and
            // changes nothing.
            let huge = ChildBlocker {
                kind: "budget".into(),
                reason: "x".repeat(MAX_CHILD_BLOCKER_REASON_CHARS + 1),
                ..Default::default()
            };
            let err = s
                .orchestrator_child_blocker_put("child-0", &huge)
                .unwrap_err();
            assert_eq!(err.kind, faktor_core::error::ErrorKind::Oversized);
            assert_eq!(
                s.orchestrator_child_blocker_get().unwrap().unwrap(),
                blocker,
                "a rejected hostile blocker must not mutate the durable row"
            );
            // Empty/whitespace/control text is Malformed.
            for hostile in [
                ChildBlocker {
                    kind: "  ".into(),
                    reason: "x".into(),
                    ..Default::default()
                },
                ChildBlocker {
                    kind: "budget".into(),
                    reason: "\t\n".into(),
                    ..Default::default()
                },
            ] {
                assert_eq!(
                    s.orchestrator_child_blocker_put("child-0", &hostile)
                        .unwrap_err()
                        .kind,
                    faktor_core::error::ErrorKind::Malformed
                );
            }
            s.id()
        };
        // Reopen: the v23 typed row is durable across a manager restart.
        let m =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let s = m.get_session(sid).unwrap().unwrap();
        let got = s.orchestrator_child_blocker_get().unwrap().unwrap();
        assert_eq!(got.kind, "budget");
        assert_eq!(got.last_progress_ms, Some(7));
        // Running clears the durable blocker (row gone).
        s.orchestrator_child_blocker_clear().unwrap();
        assert!(s.orchestrator_child_blocker_get().unwrap().is_none());
        assert!(s.orchestrator_child_runtime_get().unwrap().is_none());
    }
}
