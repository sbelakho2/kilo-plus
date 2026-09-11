//! Shadow mutation roots (P0-48): the strongest remaining filesystem
//! guarantee for single-agent MUTATING tasks.
//!
//! When the config gate `[tasks] shadow_mutation` is ON, a single-item
//! mutating task does NOT work in the user checkout: the executor begins a
//! daemon-owned SHADOW — a bounded copy of the checkout under
//! `<data dir>/shadows/<session>/<shadow id>` (never inside the checkout) —
//! and only a conflict-aware commit copies the shadow's changed files back
//! into the user checkout through the wave-10/13 commit-time CAS primitives.
//!
//! Semantics (decided and documented here):
//! - The user checkout is the INTEGRATION TARGET, never the worktree of a
//!   shadowed drive. Every applied file is CAS-guarded against the digest it
//!   had at `begin_shadow` (the durable base manifest); a user file that
//!   changed meanwhile is a per-file CONFLICT and is never overwritten.
//! - Integration is automatic on a verified-complete run (every changed file
//!   approved) — wave-13's direct-integration semantics for children,
//!   re-expressed here for the single-agent case. The wave-13
//!   `approve_and_merge` API itself does not fit this case (it resolves the
//!   owner root and child worktree through durable child/plan rows, which an
//!   in-session run has none of), so the same record-first envelope +
//!   CAS-apply discipline is implemented against the shadow's own durable
//!   rows; the durable base/change-set rows reuse the wave-13 chunked row
//!   helpers under the shadow's own run namespace.
//! - On integration CONFLICTS the shadow is NOT discarded: the durable
//!   shadow row moves to `IntegrationBlocked`, the envelope records status
//!   Conflicted with the conflict list durably, and re-running the commit
//!   with the same (auto) decision after the user resolves the drift resumes
//!   the apply phase — each file apply is CAS-idempotent
//!   ([`faktor_fs::CasMergeResult::AlreadyCurrent`] on replay). The task
//!   row itself is agent-owned: a task whose verification certified the
//!   SHADOW state keeps its certified state while the durable shadow rows
//!   carry the integration_conflict reason — exactly the wave-13 model where
//!   a Done child can still have a Failed merge envelope.
//! - On failure/cancel (or an empty diff) the shadow is discarded.
//! - Zero orphans: the active-shadow row is a DURABLE session fact
//!   ([`faktor_session::ShadowRow`]), so a crashed daemon's shadows are
//!   recoverable/cleanable after reopen ([`ShadowRoots::reconcile`]); a
//!   graceful daemon shutdown removes every shadow ([`ShadowRoots::shutdown`]
//!   and the service's `Drop`).
//! - Bounded everything: the base copy and the staged diff are capped
//!   (typed Oversized refusals; nothing is ever silently truncated), shadow
//!   rows live under bounded keys/values, and the git plumbing of the
//!   checkout (`.git`) is never copied.
//!
//! Git note: `faktor-git` exposes no worktree/archive-at-revision API
//! outside the repository (its only creation path, `WorktreeManager::create`,
//! adds a branch-worktree INSIDE the user repo — forbidden for shadows — and
//! needs a supervisor + async). All roots therefore use the bounded fs copy;
//! a git root (a tree with a `.git` entry) is detected and its `.git` is
//! skipped so no plumbing is ever materialized into a shadow.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(test)]
use std::sync::Mutex;

use faktor_core::hash::FileHash;
use faktor_core::id::SessionId;
use faktor_fs::CasMergeResult;
use faktor_session::{SessionManager, ShadowRow, ShadowRowState};

use crate::runtime::merge::{
    self, base_id_of, compute_change_entries, parent_handle, put_base_map, put_change_set,
    read_base_map, read_change_set, scan_facts, validate_decision, ChangeEntry, ChangeSet,
    MAX_BASE_ENTRIES,
};
use crate::runtime::{ExecError, MAX_RUN_ID_CHARS};

/// Default entry cap of one shadow base copy (matches the wave-13 base-map
/// cap; trees beyond it are typed Oversized refusals).
pub const SHADOW_MAX_BASE_ENTRIES: usize = MAX_BASE_ENTRIES;
/// Default total-byte cap of one shadow base copy.
pub const SHADOW_MAX_COPY_BYTES: u64 = 1024 * 1024 * 1024;
/// The daemon-owned subdirectory every shadow lives under.
pub const SHADOWS_DIR_NAME: &str = "shadows";
/// Deterministic shadow-run namespace inside the session's fact space (all
/// base-map/change-set rows of one shadow are keyed under its shadow id,
/// which is itself derived from the durable op-id sequence — generations
/// never collide).
const CHILD_ID: &str = "shadow";

/// The durable envelope of ONE shadow integration attempt (P0-48, modeled
/// on the wave-13 merge record): written (in-flight) BEFORE any file apply,
/// finalized after every apply. Kind [`SHADOW_MERGE_KIND`] under
/// `"<shadow id>/merge/<seq>"`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShadowMergeStatus {
    /// Every approved file applied with no conflicts.
    Applied,
    /// Conflicts surfaced (or the in-flight pre-apply marker); the shadow is
    /// retained and the conflict list is recorded durably. The machine
    /// reason of the blocked outcome is `integration_conflict` (the turn
    /// itself was verified against the shadow world — the conflict is an
    /// integration-time event, never a turn outcome).
    Conflicted,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ShadowMergeEnvelope {
    pub seq: u64,
    pub shadow_id: String,
    pub cs_id: String,
    pub status: ShadowMergeStatus,
    pub approved_count: usize,
    pub rejected_count: usize,
    pub merged_count: usize,
    pub conflict_count: usize,
    pub created_ms: i64,
    /// Set only once the apply phase fully finished; `None` = in-flight
    /// (crash-safe replay resumes from the durable decision rows).
    pub finished_ms: Option<i64>,
    /// Bounded detail (first conflicts / integration_conflict reason).
    pub details: String,
}

impl ShadowMergeEnvelope {
    pub fn in_flight(&self) -> bool {
        self.finished_ms.is_none()
    }
}

/// The durable part rows of one shadow integration (approved/rejected
/// decisions written BEFORE the first apply; merged/conflicts after).
pub(crate) const SHADOW_MERGE_KIND: &str = "shadow_merge";
pub(crate) const SHADOW_MERGE_PART_KIND: &str = "shadow_merge_part";
const ENVELOPE_BUDGET: usize = 3900;

/// Outcome of one shadow integration attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShadowIntegration {
    pub merged: Vec<PathBuf>,
    pub rejected: Vec<PathBuf>,
    pub conflicts: Vec<(PathBuf, String)>,
}

impl ShadowIntegration {
    pub fn clean(&self) -> bool {
        self.conflicts.is_empty()
    }
}

/// A stored approved/rejected decision pair of one integration attempt.
type StoredDecision = (Vec<PathBuf>, Vec<PathBuf>);

/// One begun shadow: the daemon-owned work root of a shadowed drive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shadow {
    pub session_id: SessionId,
    pub shadow_id: String,
    /// The user checkout (integration target of `commit_back`).
    pub base_root: PathBuf,
    /// The shadow's own root (daemon data dir; never inside the checkout).
    pub root: PathBuf,
}

/// Copy caps of one shadow base (bounded everything: caps are typed
/// Oversized refusals — the service never silently truncates a base copy).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShadowCopyLimits {
    pub max_entries: usize,
    pub max_total_bytes: u64,
}

impl Default for ShadowCopyLimits {
    fn default() -> Self {
        Self {
            max_entries: SHADOW_MAX_BASE_ENTRIES,
            max_total_bytes: SHADOW_MAX_COPY_BYTES,
        }
    }
}

/// The shadow service: one per daemon data dir. Begins/commits/discards
/// shadows and keeps their durable registry rows consistent. A graceful
/// shutdown (Drop or [`ShadowRoots::shutdown`]) removes every shadow dir;
/// a crash leaves rows that a reopen reconciles deterministically.
pub struct ShadowRoots {
    manager: Arc<SessionManager>,
    shadows_root: PathBuf,
    limits: ShadowCopyLimits,
    #[cfg(test)]
    apply_seam: Arc<Mutex<Option<usize>>>,
}

impl std::fmt::Debug for ShadowRoots {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShadowRoots")
            .field("shadows_root", &self.shadows_root)
            .finish_non_exhaustive()
    }
}

impl ShadowRoots {
    /// A service rooted at `<data dir>/<SHADOWS_DIR_NAME>`. The root is
    /// created on first use; `shadows_root` must never be a user checkout.
    pub fn new(manager: Arc<SessionManager>, shadows_root: PathBuf) -> Arc<Self> {
        Self::new_with_limits(manager, shadows_root, ShadowCopyLimits::default())
    }

    pub fn new_with_limits(
        manager: Arc<SessionManager>,
        shadows_root: PathBuf,
        limits: ShadowCopyLimits,
    ) -> Arc<Self> {
        if limits.max_entries == 0 || limits.max_total_bytes == 0 {
            panic!("shadow copy caps must be >= 1");
        }
        if limits.max_entries > MAX_BASE_ENTRIES {
            panic!("shadow copy entries cap exceeds the base-map cap");
        }
        Arc::new(Self {
            manager,
            shadows_root,
            limits,
            #[cfg(test)]
            apply_seam: Arc::new(Mutex::new(None)),
        })
    }

    pub fn manager(&self) -> Arc<SessionManager> {
        self.manager.clone()
    }

    pub fn shadows_root(&self) -> &Path {
        &self.shadows_root
    }

    /// The durable active-shadow row of a session (reopen-safe read).
    pub fn active_shadow(&self, session: SessionId) -> Result<Option<ShadowRow>, ExecError> {
        self.manager
            .shadow_row(session)
            .map_err(|e| ExecError::Internal(format!("shadow row read: {e}")))
    }

    // ------------------------------------------------------------ begin

    /// Begin a shadow of the session's workspace: a fresh daemon-owned
    /// bounded copy of `base_root` (the user checkout) plus the durable
    /// active-shadow row and the base manifest (the CAS anchors of every
    /// later apply). Refuses while the session already carries a LIVE
    /// shadow (crash residue or a drive in flight — resume/discard first).
    ///
    /// Copy order is deterministic and zero-orphan-safe: directory, bounded
    /// copy (`.git` plumbing skipped; symlink escapes, unreadable files and
    /// trees beyond the caps fail LOUDLY with typed errors), durable base
    /// manifest, durable active row. A failure at any point removes the
    /// partial directory; a crash between steps leaves a row-less directory
    /// (removed by [`ShadowRoots::reconcile`]) or a manifest without a row
    /// (harmless chunk rows, same crash semantics as the wave-13 base
    /// records).
    pub fn begin_shadow(&self, session: SessionId, base_root: &Path) -> Result<Shadow, ExecError> {
        parent_handle(&self.manager, session)?;
        let shadows_root = self.ensure_shadows_root()?;
        if let Some(existing) = self
            .manager
            .shadow_row(session)
            .map_err(|e| ExecError::Internal(format!("shadow row read: {e}")))?
        {
            if existing.state.is_live() {
                return Err(ExecError::Conflict(format!(
                    "session {session} has a live shadow {} (root {}); resume its integration or discard() it before beginning a new one",
                    existing.shadow_id,
                    existing.root
                )));
            }
            // A retired shadow row: its directory must be gone already; a
            // residue is removed below with the fresh directory.
        }
        let base = base_root.canonicalize().map_err(|e| {
            ExecError::NotFound(format!("shadow base {}: {e}", base_root.display()))
        })?;
        if !base.is_dir() {
            return Err(ExecError::NotFound(format!(
                "shadow base {} is not a directory",
                base.display()
            )));
        }
        if base.starts_with(&shadows_root) || base == shadows_root {
            return Err(ExecError::Conflict(format!(
                "shadow base {} lives inside the daemon shadow root {}; a user checkout can never be a shadow",
                base.display(),
                shadows_root.display()
            )));
        }
        let shadow_id = format!("sh-{:016x}", self.manager.next_op_id().raw());
        let dir = shadows_root
            .join(session.raw().to_string())
            .join(&shadow_id);
        if dir.exists() {
            // Crash residue of an earlier generation: the daemon-owned dir
            // is removed wholesale (never inside the user checkout).
            std::fs::remove_dir_all(&dir).map_err(|e| {
                ExecError::Internal(format!("shadow residue removal {}: {e}", dir.display()))
            })?;
        }
        std::fs::create_dir_all(&dir)
            .map_err(|e| ExecError::Internal(format!("shadow dir {}: {e}", dir.display())))?;
        let copied = match faktor_fs::copy_tree_skip(
            &base,
            &dir,
            self.limits.max_entries,
            self.limits.max_total_bytes,
            &[".git"],
        ) {
            Ok(m) => m,
            Err(e) => {
                let _ = std::fs::remove_dir_all(&dir);
                return Err(ExecError::from_fs("shadow base copy", &base, e));
            }
        };
        let manifest: Vec<(PathBuf, FileHash)> =
            copied.iter().map(|e| (e.path.clone(), e.hash)).collect();
        // Durable base manifest FIRST (crash before the row leaves a
        // row-less dir removed by reconcile), then the durable active row.
        if let Err(e) = put_base_map(
            &self.manager,
            session,
            &shadow_id,
            CHILD_ID,
            "base",
            &manifest,
        ) {
            let _ = std::fs::remove_dir_all(&dir);
            return Err(e);
        }
        let total: u64 = copied.iter().map(|e| e.size).sum();
        let row = ShadowRow {
            session_id: session.raw(),
            shadow_id: shadow_id.clone(),
            base_root: base.to_string_lossy().into_owned(),
            root: dir.to_string_lossy().into_owned(),
            state: ShadowRowState::Active,
            base_entries: copied.len() as u64,
            base_bytes: total,
            created_ms: self.manager.now_ms(),
        };
        if let Err(e) = self
            .manager
            .put_shadow_row(session, &row)
            .map_err(|e| ExecError::Internal(format!("shadow row write: {e}")))
        {
            let _ = std::fs::remove_dir_all(&dir);
            return Err(e);
        }
        Ok(Shadow {
            session_id: session,
            shadow_id,
            base_root: base,
            root: dir,
        })
    }

    // ------------------------------------------------------------ staging

    /// Compute and durably store the change set of the session's shadow:
    /// every file whose current shadow content differs from the base
    /// manifest (with the base digest as the CAS anchor of the user path),
    /// plus recorded deletions. Entry count beyond [`MAX_CHANGES`] is a
    /// typed Oversized error and NOTHING is stored (the merge never
    /// silently truncates). Deterministic and idempotent for an unchanged
    /// shadow.
    pub fn stage_change_set(&self, session: SessionId) -> Result<ChangeSet, ExecError> {
        let row = self.live_row(session)?;
        let shadow = Shadow {
            session_id: session,
            shadow_id: row.shadow_id.clone(),
            base_root: PathBuf::from(&row.base_root),
            root: PathBuf::from(&row.root),
        };
        let base_map = self.base_manifest(&shadow)?;
        let now_snap = faktor_fs::snapshot_tree(&shadow.root, MAX_BASE_ENTRIES)
            .map_err(|e| ExecError::from_fs("shadow tree snapshot", &shadow.root, e))?;
        let now: Vec<(PathBuf, FileHash)> =
            now_snap.iter().map(|e| (e.path.clone(), e.hash)).collect();
        let files = compute_change_entries(CHILD_ID, &base_map, &now, &base_map)?;
        let cs = ChangeSet {
            child_id: CHILD_ID.to_string(),
            base_id: base_id_of(&shadow.shadow_id),
            files,
            created_ms: self.manager.now_ms(),
        };
        put_change_set(&self.manager, session, &shadow.shadow_id, &cs)?;
        Ok(cs)
    }

    // -------------------------------------------------------- integration

    /// Present the session's staged change set (staging it first when
    /// absent). The structured candidate a drive's verified-complete end
    /// presents for approval-merge: [`commit_back`] consumes it.
    pub fn present_change_set(&self, session: SessionId) -> Result<ChangeSet, ExecError> {
        let row = self.live_row(session)?;
        let cs_id = format!("{}-cs", base_id_of(&row.shadow_id));
        match read_change_set(&self.manager, session, &row.shadow_id, CHILD_ID, &cs_id) {
            Ok(cs) => Ok(cs),
            Err(ExecError::NotFound(_)) => self.stage_change_set(session),
            Err(e) => Err(e),
        }
    }

    /// Commit the shadow's changes back into the USER checkout. `approved`
    /// and `rejected` must decide every changed file ([`validate_decision`];
    /// [`ShadowRoots::commit_all`] auto-approves everything). Each apply is
    /// a commit-time CAS write against the base manifest digest of the user
    /// path (expected = the digest at begin): a user file that changed
    /// meanwhile is a CONFLICT and is never overwritten; a path absent at
    /// begin is an exclusive create.
    ///
    /// Record-first durability (wave-13 discipline): the in-flight envelope
    /// and the durable decision rows are written BEFORE any apply, the
    /// merged/conflict parts and the finalized envelope after. A crash at
    /// any point is resumed by calling this again with the SAME decision:
    /// every apply is CAS-idempotent ([`faktor_fs::CasMergeResult::AlreadyCurrent`]).
    ///
    /// Outcome semantics:
    /// - no conflicts → the user checkout holds the new content; the shadow
    ///   directory is removed and the durable row is marked `Integrated`;
    /// - conflicts → the shadow is RETAINED, the row moves to
    ///   `IntegrationBlocked` and the envelope carries the durable conflict
    ///   list (reason `integration_conflict`). Re-run after the user
    ///   resolves the drift with the same decision to resume.
    pub fn commit_back(
        &self,
        session: SessionId,
        approved: &[PathBuf],
        rejected: &[PathBuf],
    ) -> Result<ShadowIntegration, ExecError> {
        let row = self.live_row(session)?;
        let shadow = Shadow {
            session_id: session,
            shadow_id: row.shadow_id.clone(),
            base_root: PathBuf::from(&row.base_root),
            root: PathBuf::from(&row.root),
        };
        let base_root = shadow.base_root.canonicalize().map_err(|e| {
            ExecError::NotFound(format!(
                "integration target {} vanished since begin_shadow: {e}",
                shadow.base_root.display()
            ))
        })?;
        if !shadow.root.is_dir() {
            return Err(ExecError::NotFound(format!(
                "shadow root {} is gone; discard() and begin_shadow() again",
                shadow.root.display()
            )));
        }
        let cs = self.present_change_set(session)?;
        let (approved_v, rejected_v) = validate_decision(&cs, approved, rejected)?;
        let base_map = self.base_manifest(&shadow)?;
        let base_idx: BTreeMap<PathBuf, FileHash> = base_map.iter().cloned().collect();

        // ---- durable record FIRST (crash between record and applies is
        // replay-safe because every apply is CAS-idempotent).
        let seq = self.existing_seq(session, &shadow.shadow_id, &cs.id())?;
        let decision_stored = self
            .decision_rows(session, &shadow.shadow_id, &cs.id(), seq)?
            .is_some();
        if decision_stored {
            let stored = self
                .decision_rows(session, &shadow.shadow_id, &cs.id(), seq)?
                .expect("checked above");
            if stored.0 != approved_v || stored.1 != rejected_v {
                return Err(ExecError::Conflict(format!(
                    "the durable integration record of shadow {} records a different decision; replay must carry the identical approved/rejected sets",
                    shadow.shadow_id
                )));
            }
        } else {
            // In-flight envelope + durable decision rows BEFORE any apply.
            if self.envelope(session, &shadow.shadow_id, seq)?.is_none() {
                self.write_envelope(
                    session,
                    &ShadowMergeEnvelope {
                        seq,
                        shadow_id: shadow.shadow_id.clone(),
                        cs_id: cs.id(),
                        status: ShadowMergeStatus::Conflicted,
                        approved_count: approved_v.len(),
                        rejected_count: rejected_v.len(),
                        merged_count: 0,
                        conflict_count: 0,
                        created_ms: self.manager.now_ms(),
                        finished_ms: None,
                        details:
                            "in-flight: durable integration record written before any file apply"
                                .into(),
                    },
                )?;
            }
            self.write_part(
                session,
                &shadow.shadow_id,
                seq,
                "approved",
                &approved_v,
                &[],
            )?;
            self.write_part(
                session,
                &shadow.shadow_id,
                seq,
                "rejected",
                &rejected_v,
                &[],
            )?;
        }

        // ---- apply phase (deterministic order = staged path order).
        let mut merged: Vec<PathBuf> = Vec::new();
        let mut conflicts: Vec<(PathBuf, String)> = Vec::new();
        let mut processed = 0usize;
        for entry in &cs.files {
            if rejected_v.contains(&entry.path) {
                processed += 1;
                continue;
            }
            match self.apply_one(&base_root, &shadow, entry, &base_idx) {
                Ok(()) => merged.push(entry.path.clone()),
                Err(ApplyFailure::Conflict(detail)) => conflicts.push((entry.path.clone(), detail)),
                Err(ApplyFailure::Hard(e)) => {
                    // Prior applies stay (each individually CAS-committed);
                    // the in-flight durable record lets the caller resume.
                    return Err(e);
                }
            }
            processed += 1;
            self.check_apply_seam(processed)?;
        }
        merged.sort();
        // ---- durable outcome rows, then the FINAL envelope.
        self.write_part(session, &shadow.shadow_id, seq, "merged", &merged, &[])?;
        self.write_part(
            session,
            &shadow.shadow_id,
            seq,
            "conflicts",
            &[],
            &conflicts,
        )?;
        let conflicted = !conflicts.is_empty();
        let detail = if conflicts.is_empty() {
            format!("all {} approved file(s) integrated", merged.len())
        } else {
            let (first_path, first_detail) = &conflicts[0];
            format!(
                "integration_conflict: {} conflict(s); first: {} — {}",
                conflicts.len(),
                first_path.display(),
                first_detail.chars().take(160).collect::<String>()
            )
        };
        self.write_envelope(
            session,
            &ShadowMergeEnvelope {
                seq,
                shadow_id: shadow.shadow_id.clone(),
                cs_id: cs.id(),
                status: if conflicted {
                    ShadowMergeStatus::Conflicted
                } else {
                    ShadowMergeStatus::Applied
                },
                approved_count: approved_v.len(),
                rejected_count: rejected_v.len(),
                merged_count: merged.len(),
                conflict_count: conflicts.len(),
                created_ms: self.manager.now_ms(),
                finished_ms: Some(self.manager.now_ms()),
                details: detail.chars().take(300).collect(),
            },
        )?;
        let outcome = ShadowIntegration {
            merged,
            rejected: rejected_v,
            conflicts,
        };
        if outcome.clean() {
            // Clean integration: the user checkout holds the new content.
            // Record-first: the durable row reaches Integrated BEFORE the
            // directory is removed, so a reader (or a crash) can never see
            // filesystem cleanup with the row still Active. Windows CI
            // exposed the inverse ordering as an integration race.
            self.mark_state(session, ShadowRowState::Integrated)?;
            let _ = std::fs::remove_dir_all(&shadow.root);
        } else {
            // Conflicts: retain the shadow; the durable conflict list is the
            // integration_conflict record. The row's IntegrationBlocked
            // state keeps the run from completing on the user side.
            self.mark_state(session, ShadowRowState::IntegrationBlocked)?;
        }
        Ok(outcome)
    }

    /// Auto-approve every changed file and commit (wave-13 direct
    /// integration semantics). Conflicts surface in the returned outcome;
    /// the shadow is retained on conflict ([`commit_back`] docs).
    pub fn commit_all(&self, session: SessionId) -> Result<ShadowIntegration, ExecError> {
        let cs = self.present_change_set(session)?;
        let all: Vec<PathBuf> = cs.files.iter().map(|f| f.path.clone()).collect();
        self.commit_back(session, &all, &[])
    }

    // ------------------------------------------------------------ discard

    /// Remove the session's shadow directory and durably mark the row
    /// `Discarded` (the row itself stays: a tombstone, never silently
    /// dropped). Idempotent for a missing directory.
    pub fn discard(&self, session: SessionId) -> Result<(), ExecError> {
        let Some(row) = self
            .manager
            .shadow_row(session)
            .map_err(|e| ExecError::Internal(format!("shadow row read: {e}")))?
        else {
            return Err(ExecError::NotFound(format!(
                "session {session} has no shadow row to discard"
            )));
        };
        let dir = PathBuf::from(&row.root);
        if dir.exists() {
            std::fs::remove_dir_all(&dir).map_err(|e| {
                ExecError::Internal(format!("shadow removal {}: {e}", dir.display()))
            })?;
        }
        self.mark_state(session, ShadowRowState::Discarded)
    }

    // ------------------------------------------------------------ lifecycle

    /// Reopen recovery (crash residue, deterministic): every durable shadow
    /// row of every session is checked against the filesystem and the
    /// session table.
    /// - a LIVE row whose directory is gone → the shadow cannot integrate;
    ///   marked `Discarded` (deterministic: resume is impossible, discard
    ///   is the only path — a fresh shadow is begun by the next run);
    /// - a live row of a CLOSED session → the drive can never resume;
    ///   directory removed, row marked `Discarded`;
    /// - a row marked Integrated/Discarded whose directory still exists →
    ///   removed;
    /// - daemon-owned shadow directories without any row (a crash between
    ///   the copy and the row write) → removed.
    ///
    /// Returns a human-readable list of every action taken (empty = already
    /// consistent). Read/write access to rows goes through the session
    /// registry; nothing outside the daemon's shadow root is ever touched.
    pub fn reconcile(&self) -> Result<Vec<String>, ExecError> {
        let mut actions: Vec<String> = Vec::new();
        let Ok(sessions) = self.manager.list_sessions(None) else {
            return Err(ExecError::Internal("session list failed".into()));
        };
        let mut rowed_dirs: Vec<PathBuf> = Vec::new();
        for handle in sessions {
            let session = handle.id();
            let row = self
                .manager
                .shadow_row(session)
                .map_err(|e| ExecError::Internal(format!("shadow row read: {e}")))?;
            let Ok(srow) = handle.row() else {
                continue;
            };
            let session_terminal = srow.lifecycle.is_terminal();
            let Some(row) = row else {
                continue;
            };
            let dir = PathBuf::from(&row.root);
            rowed_dirs.push(dir.clone());
            if row.state.is_live() && !dir.is_dir() {
                // Shadow directory gone: integration is impossible.
                self.mark_state(session, ShadowRowState::Discarded)?;
                actions.push(format!(
                    "session {session}: shadow {} directory gone; marked discarded",
                    row.shadow_id
                ));
                continue;
            }
            if row.state.is_live() && session_terminal {
                // Record-first, like integration: the durable terminal state
                // precedes filesystem cleanup so no observer can catch a
                // live row with its directory already gone.
                self.mark_state(session, ShadowRowState::Discarded)?;
                if dir.is_dir() {
                    let _ = std::fs::remove_dir_all(&dir);
                }
                actions.push(format!(
                    "session {session}: closed with a live shadow {}; discarded",
                    row.shadow_id
                ));
                continue;
            }
            if !row.state.is_live() && dir.is_dir() {
                let _ = std::fs::remove_dir_all(&dir);
                actions.push(format!(
                    "session {session}: retired shadow {} directory removed",
                    row.shadow_id
                ));
            }
        }
        // Row-less daemon-owned shadow directories (crash between the copy
        // and the row write) are removed. The scan root is canonicalized so
        // its entries compare equal to the canonical row roots.
        let scan_root = self
            .shadows_root
            .canonicalize()
            .unwrap_or_else(|_| self.shadows_root.clone());
        if let Ok(entries) = std::fs::read_dir(&scan_root) {
            for e in entries.flatten() {
                let session_dir = e.path();
                let Ok(entries2) = std::fs::read_dir(&session_dir) else {
                    continue;
                };
                for e2 in entries2.flatten() {
                    let dir = e2.path();
                    if dir.is_dir() && !rowed_dirs.contains(&dir) {
                        let _ = std::fs::remove_dir_all(&dir);
                        actions.push(format!(
                            "row-less shadow directory {} removed",
                            dir.display()
                        ));
                    }
                }
            }
        }
        Ok(actions)
    }

    /// Graceful daemon shutdown: remove every shadow directory of every
    /// live row and mark the rows `Discarded`. Best effort (a shutdown must
    /// never fail the process); [`ShadowRoots::reconcile`] at the next
    /// daemon start handles whatever a crash left behind. Also invoked by
    /// the service's `Drop`.
    pub fn shutdown(&self) {
        let Ok(sessions) = self.manager.list_sessions(None) else {
            return;
        };
        for handle in sessions {
            let session = handle.id();
            let Ok(Some(row)) = self.manager.shadow_row(session) else {
                continue;
            };
            if row.state.is_live() {
                let dir = PathBuf::from(&row.root);
                if dir.exists() {
                    let _ = std::fs::remove_dir_all(&dir);
                }
                let mut retired = row;
                retired.state = ShadowRowState::Discarded;
                let _ = self.manager.put_shadow_row(session, &retired);
            }
        }
    }

    // ------------------------------------------------------------ internals

    fn ensure_shadows_root(&self) -> Result<PathBuf, ExecError> {
        std::fs::create_dir_all(&self.shadows_root).map_err(|e| {
            ExecError::Internal(format!("shadow root {}: {e}", self.shadows_root.display()))
        })?;
        // Canonical: the "base inside the shadow root" escape check compares
        // canonical paths — a `/var` vs `/private/var` spelling mismatch
        // would defeat starts_with.
        self.shadows_root
            .canonicalize()
            .map_err(|e| ExecError::Internal(format!("shadow root canonical: {e}")))
    }

    fn live_row(&self, session: SessionId) -> Result<ShadowRow, ExecError> {
        let Some(row) = self
            .manager
            .shadow_row(session)
            .map_err(|e| ExecError::Internal(format!("shadow row read: {e}")))?
        else {
            return Err(ExecError::NotFound(format!(
                "session {session} has no shadow; begin_shadow() first"
            )));
        };
        if !row.state.is_live() {
            return Err(ExecError::InvalidState(format!(
                "shadow {} of session {session} is {:?}; only Active/IntegrationBlocked shadows integrate or stage",
                row.shadow_id, row.state
            )));
        }
        Ok(row)
    }

    fn mark_state(&self, session: SessionId, state: ShadowRowState) -> Result<(), ExecError> {
        let Some(mut row) = self
            .manager
            .shadow_row(session)
            .map_err(|e| ExecError::Internal(format!("shadow row read: {e}")))?
        else {
            return Err(ExecError::NotFound(format!(
                "session {session} has no shadow row"
            )));
        };
        row.state = state;
        self.manager
            .put_shadow_row(session, &row)
            .map_err(|e| ExecError::Internal(format!("shadow row write: {e}")))
    }

    /// The durable base manifest of one shadow (the CAS anchors of the user
    /// checkout at begin). Missing = a crashed begin: refuse loudly — the
    /// only deterministic paths are discard (then begin again) or, when the
    /// manifest simply never got written, retrying begin after discard.
    fn base_manifest(&self, shadow: &Shadow) -> Result<Vec<(PathBuf, FileHash)>, ExecError> {
        match read_base_map(&self.manager, shadow.session_id, &shadow.shadow_id, CHILD_ID, "base")
        {
            Ok(Some(map)) => Ok(map),
            Ok(None) => Err(ExecError::InvalidState(format!(
                "shadow {} has no durable base manifest (crashed begin); discard() it and begin_shadow() again",
                shadow.shadow_id
            ))),
            Err(e) => Err(e),
        }
    }

    fn envelope_key(shadow_id: &str, seq: u64) -> String {
        format!("{shadow_id}/merge/{seq}")
    }

    fn part_key(shadow_id: &str, seq: u64, part: &str) -> String {
        format!("{shadow_id}/merge/{seq}/part/{part}")
    }

    fn envelope(
        &self,
        session: SessionId,
        shadow_id: &str,
        seq: u64,
    ) -> Result<Option<ShadowMergeEnvelope>, ExecError> {
        let handle = parent_handle(&self.manager, session)?;
        let key = Self::envelope_key(shadow_id, seq);
        for (kind, k, value) in scan_facts(&handle)? {
            if kind == SHADOW_MERGE_KIND && k == key {
                let env: ShadowMergeEnvelope = serde_json::from_str(&value).map_err(|e| {
                    ExecError::Internal(format!("integration envelope decode {key}: {e}"))
                })?;
                return Ok(Some(env));
            }
        }
        Ok(None)
    }

    fn existing_seq(
        &self,
        session: SessionId,
        shadow_id: &str,
        cs_id: &str,
    ) -> Result<u64, ExecError> {
        let handle = parent_handle(&self.manager, session)?;
        let prefix = format!("{shadow_id}/merge/");
        let mut envs: Vec<(u64, ShadowMergeEnvelope)> = Vec::new();
        for (kind, key, value) in scan_facts(&handle)? {
            if kind != SHADOW_MERGE_KIND {
                continue;
            }
            let Some(rest) = key.strip_prefix(&prefix) else {
                continue;
            };
            let seq: u64 = rest
                .parse()
                .map_err(|_| ExecError::Internal(format!("hostile envelope key {key:?}")))?;
            let env: ShadowMergeEnvelope = serde_json::from_str(&value).map_err(|e| {
                ExecError::Internal(format!("integration envelope decode {key}: {e}"))
            })?;
            envs.push((seq, env));
        }
        envs.sort_by_key(|(seq, _)| *seq);
        let existing = envs.into_iter().find(|(_, e)| e.cs_id == cs_id);
        match existing {
            Some((seq, _)) => Ok(seq),
            None => {
                // A fresh attempt: seq 1. A hostile leftover at seq 1 with a
                // different cs id (never produced by this service) is
                // refused loudly.
                Ok(1)
            }
        }
    }

    fn decision_rows(
        &self,
        session: SessionId,
        shadow_id: &str,
        _cs_id: &str,
        seq: u64,
    ) -> Result<Option<StoredDecision>, ExecError> {
        let handle = parent_handle(&self.manager, session)?;
        let approved = self.read_part(&handle, shadow_id, seq, "approved")?;
        let rejected = self.read_part(&handle, shadow_id, seq, "rejected")?;
        match (approved, rejected) {
            (Some(a), Some(r)) => Ok(Some((a, r))),
            _ => Ok(None),
        }
    }

    fn write_envelope(
        &self,
        session: SessionId,
        env: &ShadowMergeEnvelope,
    ) -> Result<(), ExecError> {
        let handle = parent_handle(&self.manager, session)?;
        if !env.shadow_id.is_ascii()
            || env.shadow_id.is_empty()
            || env.shadow_id.len() > MAX_RUN_ID_CHARS
            || env.shadow_id.contains('/')
        {
            return Err(ExecError::Oversized(format!(
                "shadow id must be 1..={MAX_RUN_ID_CHARS} ASCII characters without '/'"
            )));
        }
        let value = serde_json::to_string(env)
            .map_err(|e| ExecError::Internal(format!("envelope serialization: {e}")))?;
        if value.len() > ENVELOPE_BUDGET {
            return Err(ExecError::Oversized(format!(
                "integration envelope of {} bytes exceeds the durable row budget",
                value.len()
            )));
        }
        handle
            .upsert_memory_fact(
                SHADOW_MERGE_KIND,
                &Self::envelope_key(&env.shadow_id, env.seq),
                &value,
            )
            .map_err(|e| ExecError::Internal(format!("integration record write: {}", e.message)))
    }

    fn write_part(
        &self,
        session: SessionId,
        shadow_id: &str,
        seq: u64,
        part: &str,
        paths: &[PathBuf],
        conflicts: &[(PathBuf, String)],
    ) -> Result<(), ExecError> {
        let handle = parent_handle(&self.manager, session)?;
        let items: Vec<serde_json::Value> = match part {
            "conflicts" => conflicts
                .iter()
                .map(|(p, d)| {
                    serde_json::json!([
                        p.to_string_lossy(),
                        d.chars().take(400).collect::<String>()
                    ])
                })
                .collect(),
            _ => paths
                .iter()
                .map(|p| serde_json::json!(p.to_string_lossy()))
                .collect(),
        };
        let chunks = merge::pack_chunks(&items)?;
        let header = serde_json::json!({ "part": part, "chunks": chunks.len(), "created_ms": self.manager.now_ms() });
        let header = serde_json::to_string(&header)
            .map_err(|e| ExecError::Internal(format!("part header: {e}")))?;
        merge::put_chunks(
            &handle,
            SHADOW_MERGE_PART_KIND,
            &Self::part_key(shadow_id, seq, part),
            &header,
            &chunks,
        )
    }

    fn read_part(
        &self,
        handle: &faktor_session::SessionHandle,
        shadow_id: &str,
        seq: u64,
        part: &str,
    ) -> Result<Option<Vec<PathBuf>>, ExecError> {
        let key = Self::part_key(shadow_id, seq, part);
        let Some((_h, chunks)) = merge::read_chunks(handle, SHADOW_MERGE_PART_KIND, &key)? else {
            return Ok(None);
        };
        let rows: Vec<String> = merge::unpack_chunks(&chunks)?;
        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            out.push(merge::validate_rel_path_str(&r)?);
        }
        Ok(Some(out))
    }

    /// Apply ONE approved entry with the wave-10/13 commit-time CAS
    /// primitives; per-file drift of the user file is a Conflict (the file
    /// is not merged; the rest of the decision still applies), anything
    /// else fails the whole commit loudly.
    fn apply_one(
        &self,
        base_root: &Path,
        shadow: &Shadow,
        entry: &ChangeEntry,
        base_idx: &BTreeMap<PathBuf, FileHash>,
    ) -> Result<(), ApplyFailure> {
        // The base anchor of the user path: the digest at begin. A path
        // absent from the base manifest (the user checkout had no such file)
        // is an exclusive create.
        let base_hash = entry
            .base_hash
            .or_else(|| base_idx.get(&entry.path).copied());
        let res = match entry.child_hash {
            Some(child_hash) => faktor_fs::merge_apply_content(
                base_root,
                &entry.path,
                &shadow.root,
                &entry.path,
                child_hash,
                base_hash,
            ),
            None => match base_hash {
                Some(base) => faktor_fs::merge_delete(base_root, &entry.path, base),
                None => {
                    return Err(ApplyFailure::Hard(ExecError::Internal(format!(
                        "staged deletion {:?} has no base anchor",
                        entry.path.display()
                    ))))
                }
            },
        };
        match res {
            Ok(CasMergeResult::Applied | CasMergeResult::AlreadyCurrent) => Ok(()),
            Err(e)
                if matches!(
                    e.kind,
                    faktor_core::ErrorKind::Conflict
                        | faktor_core::ErrorKind::Permission
                        | faktor_core::ErrorKind::NotFound
                ) =>
            {
                Err(ApplyFailure::Conflict(e.message))
            }
            Err(e) => Err(ApplyFailure::Hard(ExecError::from_fs(
                "shadow integration apply",
                base_root,
                e,
            ))),
        }
    }

    #[cfg(test)]
    fn check_apply_seam(&self, processed: usize) -> Result<(), ExecError> {
        let mut guard = self.apply_seam.lock().expect("seam poisoned");
        if let Some(after) = *guard {
            if processed > after {
                // Fire-and-clear: the seam trips ONE apply run of THIS
                // service instance (other instances in parallel tests are
                // never affected).
                *guard = None;
                return Err(ExecError::InjectedCrashSeam(format!(
                    "shadow integration apply seam after {after} files"
                )));
            }
        }
        Ok(())
    }

    /// Test-only: arm the deterministic apply crash seam of THIS instance.
    #[cfg(test)]
    pub fn arm_apply_seam(&self, after: usize) {
        *self.apply_seam.lock().expect("seam poisoned") = Some(after);
    }

    #[cfg(not(test))]
    fn check_apply_seam(&self, _processed: usize) -> Result<(), ExecError> {
        Ok(())
    }
}

impl Drop for ShadowRoots {
    fn drop(&mut self) {
        // Zero orphans on graceful teardown: remove every shadow of every
        // live row. Best effort by design (Drop cannot fail); reconcile()
        // on the next daemon start is the deterministic recovery of a
        // crash that skipped this.
        self.shutdown();
    }
}

/// Per-file outcome of one CAS apply.
enum ApplyFailure {
    Conflict(String),
    Hard(ExecError),
}

/// Public helper: an empty decision is `([], [])` — never valid for a
/// non-empty change set ([`validate_decision`] refuses undecided paths).
pub fn auto_approve(cs: &ChangeSet) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let mut all: Vec<PathBuf> = cs.files.iter().map(|f| f.path.clone()).collect();
    all.sort();
    (all, Vec::new())
}

/// Convenience: the auto-approve decision of a staged change set (used by
/// the executor's direct-integration path; exposed for the presentation
/// surface).
pub fn shadow_decision_of(cs: &ChangeSet) -> (Vec<PathBuf>, Vec<PathBuf>) {
    auto_approve(cs)
}

#[cfg(test)]
#[path = "shadow_tests.rs"]
mod shadow_tests;
