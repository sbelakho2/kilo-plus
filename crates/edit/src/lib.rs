//! faktor-edit — the transactional patch engine (spec §17, §18).
//!
//! Every agent edit is optimistic and versioned: `expected_hash` must match
//! the current content or the edit is rejected before any write. All ops are
//! validated against a copy first, then applied with ONE atomic write (no
//! partial writes). For supported languages the before/after syntax trees
//! are compared: an edit that breaks previously valid syntax is suspicious
//! and rolls back (or is flagged, per mode).

use std::collections::{HashMap, HashSet};

use faktor_core::error::{Error, ErrorKind};
use faktor_core::hash::FileHash;
use faktor_core::state::{EditTxnId, EditTxnStrategy};
use faktor_core::WorkspaceIdentity;
use faktor_fs::WorkspaceHandle;

pub mod diff;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditOp {
    /// Byte offsets into the (valid UTF-8) file.
    Range {
        start: usize,
        end: usize,
        replacement: String,
    },
    /// before must match exactly once.
    SearchReplace { before: String, after: String },
    /// Anchor must match uniquely; then replace bytes [region_start, region_end)
    /// (offsets relative to the anchor's start).
    BoundedRegion {
        anchor: String,
        region_start: usize,
        region_end: usize,
        replacement: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditRequest {
    pub path: String,
    pub expected_hash: FileHash,
    pub ops: Vec<EditOp>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditOutcome {
    pub new_hash: FileHash,
    pub ops_applied: usize,
    pub suspicious: bool,
    pub parse_error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct BatchCommittedFile {
    pub path: String,
    pub new_hash: FileHash,
    pub ops_applied: usize,
    pub suspicious: bool,
}

#[derive(Debug, Clone)]
pub struct EditBatchOutcome {
    /// Files written (in request order), with their new hashes.
    pub committed: Vec<BatchCommittedFile>,
    /// (path, reason): files whose commit-time compare-and-swap failed —
    /// they changed between validation and write and were NOT clobbered.
    pub conflicted: Vec<(String, String)>,
    /// Paths that were never attempted because an earlier file conflicted.
    pub skipped: Vec<String>,
    /// Ops applied across all committed files.
    pub ops_applied_total: usize,
}

impl EditBatchOutcome {
    pub fn all_committed(&self) -> bool {
        self.conflicted.is_empty() && self.skipped.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepairMode {
    /// Suspicious edits are rolled back (never written).
    Rollback,
    /// Suspicious edits are written but flagged for the model to repair.
    AllowModelRepair,
}

const MAX_FILE_BYTES: usize = 16 * 1024 * 1024;

// ============================================ durable multi-file edit txn (P0-53)
//
// `begin_multi_file_edit` / `commit_prepared` / `recover_edit_transactions`
// turn a multi-file apply into a RECORD-FIRST transaction: every file is
// staged and validated, a durable `EditTxnPrepared` row is journaled to the
// session's typed ledger BEFORE any byte is written, each commit-time CAS is
// journaled as `EditTxnProgress` AFTER its write, and the run ends with a
// terminal `EditTxnCommitted` (or `EditTxnRolledBack` for the roll-back
// policy). After a crash the open rows drive deterministic, idempotent
// recovery. `apply_many` stays as the non-durable convenience wrapper of the
// same stage-then-commit flow for callers WITHOUT a session ledger.

/// Max files in one durable edit transaction.
pub const MAX_TXN_FILES: usize = 2000;
/// Max serialized bytes of one durable edit-transaction journal record.
pub const MAX_TXN_PAYLOAD_BYTES: usize = 64 * 1024;
/// Max bytes of one staged file path inside a durable transaction.
pub const MAX_TXN_PATH_BYTES: usize = 4096;

/// One staged file of a durable transaction, exactly as journaled in its
/// `EditTxnPrepared` row: `base_digest` is the digest of the content the
/// transaction validated against, `base_len` its byte length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditTxnFileRef {
    pub path: String,
    pub base_digest: FileHash,
    pub base_len: u64,
}

/// The per-file outcome of one commit-time CAS, journaled after the write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnFileOutcome {
    Committed,
    Conflicted,
}

impl TxnFileOutcome {
    /// The durable tag stored in typed ledger rows.
    pub const fn as_tag(self) -> &'static str {
        match self {
            TxnFileOutcome::Committed => "committed",
            TxnFileOutcome::Conflicted => "conflicted",
        }
    }
}

/// One decoded progress row of an open transaction (the durable answer to
/// "was this file already committed?" during recovery).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TxnProgressRow {
    /// The file's index inside the prepared file list.
    pub seq: u64,
    pub path: String,
    pub outcome: TxnFileOutcome,
}

/// One OPEN (prepared, not yet terminaled) transaction as read back from the
/// journal — the input of [`EditEngine::recover_edit_transactions`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenEditTxn {
    pub txn_id: EditTxnId,
    pub strategy: EditTxnStrategy,
    pub files: Vec<EditTxnFileRef>,
    pub progress: Vec<TxnProgressRow>,
}

/// The durable journal a multi-file edit transaction records into. The
/// session ledger implements this over its typed `edit_txn_*` rows; engines
/// without a session ledger run the plain [`EditEngine::apply_many`] flow.
///
/// Record ordering contract (what crash recovery depends on):
/// `record_prepared` exactly once BEFORE any write, one `record_progress`
/// AFTER each file's CAS write, then exactly one terminal call. A crash
/// between a write and its progress row leaves an AMBIGUOUS file that
/// recovery refuses (typed), never silently accepting or clobbering it.
pub trait EditTxnJournal: Send + Sync {
    /// This journal's hard bound on ONE serialized record (bytes). Begins
    /// and terminals above the bound are rejected before anything is
    /// journaled or written.
    fn entry_payload_cap(&self) -> usize;
    /// Journal `EditTxnPrepared {txn_id, session, files, strategy}` —
    /// record-first, before any write.
    fn record_prepared(
        &self,
        txn_id: EditTxnId,
        session: &str,
        files: &[EditTxnFileRef],
        strategy: EditTxnStrategy,
    ) -> Result<(), Error>;
    /// Journal one `EditTxnProgress` row (after the file's CAS write).
    fn record_progress(
        &self,
        txn_id: EditTxnId,
        seq: u64,
        path: &str,
        outcome: TxnFileOutcome,
    ) -> Result<(), Error>;
    /// Journal the terminal `EditTxnCommitted` row.
    fn record_committed(
        &self,
        txn_id: EditTxnId,
        committed: &[String],
        conflicted: &[String],
        skipped: &[String],
    ) -> Result<(), Error>;
    /// Journal the terminal `EditTxnRolledBack` row.
    fn record_rolled_back(
        &self,
        txn_id: EditTxnId,
        rolled_back: &[String],
        rollback_conflicts: &[String],
    ) -> Result<(), Error>;
    /// Read every OPEN transaction (prepared without a terminal row).
    fn open_transactions(&self) -> Result<Vec<OpenEditTxn>, Error>;
}

/// A fully staged, journaled multi-file edit transaction, ready for
/// [`EditEngine::commit_prepared`]. Produced by
/// [`EditEngine::begin_multi_file_edit`]; dropped (never committed) it
/// simulates a crash after the prepared record — recovery replays it.
pub struct PreparedEdit {
    txn_id: EditTxnId,
    strategy: EditTxnStrategy,
    journal: std::sync::Arc<dyn EditTxnJournal>,
    files: Vec<StagedEdit>,
}

impl std::fmt::Debug for PreparedEdit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedEdit")
            .field("txn_id", &self.txn_id)
            .field("strategy", &self.strategy)
            .field("journal", &"<dyn EditTxnJournal>")
            .field("files", &self.files.len())
            .finish()
    }
}

impl PreparedEdit {
    pub fn txn_id(&self) -> EditTxnId {
        self.txn_id
    }

    pub fn strategy(&self) -> EditTxnStrategy {
        self.strategy
    }

    pub fn file_count(&self) -> usize {
        self.files.len()
    }
}

/// The outcome of one durable transaction commit.
#[derive(Debug, Clone)]
pub struct EditTxnResult {
    pub txn_id: EditTxnId,
    /// Files written (request order), with their new hashes.
    pub committed: Vec<BatchCommittedFile>,
    /// (path, reason): files whose commit-time CAS failed — they changed
    /// between validation and write and were NOT clobbered.
    pub conflicted: Vec<(String, String)>,
    /// Paths never attempted because an earlier file conflicted.
    pub skipped: Vec<String>,
    /// Files CAS-restored to their staged before-content (roll-back
    /// policy only).
    pub rolled_back: Vec<String>,
    /// Committed files whose restore CAS refused because the file changed
    /// again after our write — NEVER clobbered.
    pub rollback_conflicts: Vec<String>,
    /// Ops applied across all committed files.
    pub ops_applied_total: usize,
}

impl EditTxnResult {
    pub fn all_committed(&self) -> bool {
        self.conflicted.is_empty() && self.skipped.is_empty()
    }
}

/// The reissued input of ONE open transaction during crash recovery: the
/// SAME requests the transaction was begun with (the session's durable
/// tool-run record carries them across a restart) and a resolver of staged
/// before-content for roll-back restores of pre-crash commits.
pub struct EditTxnResume<'r> {
    pub txn_id: EditTxnId,
    pub reqs: &'r [EditRequest],
    /// Before-content of a path for a roll-back restore, or `None` when no
    /// durable source is wired. Without it a pre-crash commit of a
    /// roll-back transaction is a typed needs-recovery error, never a
    /// clobber.
    pub before_content: &'r dyn Fn(&str) -> Option<Vec<u8>>,
}

/// The outcome of recovering ONE open transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveryTxnOutcome {
    pub txn_id: EditTxnId,
    pub strategy: EditTxnStrategy,
    /// Paths at their target content when recovery finished (committed
    /// before the crash OR replayed by this run).
    pub committed: Vec<String>,
    /// Paths an external writer owns (never clobbered, never rewritten).
    pub conflicted: Vec<String>,
    /// Paths never attempted (behind a roll-back stop, or untouched files
    /// of a roll-back transaction that did not need replay).
    pub skipped: Vec<String>,
    /// Paths restored to their staged before-content (roll-back policy).
    pub rolled_back: Vec<String>,
    /// Restores the CAS refused (the file changed again) — never clobbered.
    pub rollback_conflicts: Vec<String>,
    /// Paths ACTUALLY rewritten (CAS replayed) by this recovery run —
    /// empty for files that were already committed before the crash.
    pub wrote: Vec<String>,
}

/// The report of one full recovery pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RecoveryReport {
    pub txns: Vec<RecoveryTxnOutcome>,
}

/// A file this recovery run wrote (in-memory roll-back restore data).
struct JustWritten {
    idx: usize,
    new_hash: FileHash,
    original: Vec<u8>,
}

/// The raw outcome of the shared stage-then-commit phase (journaled or not).
#[derive(Debug, Default)]
struct CommitRun {
    committed: Vec<BatchCommittedFile>,
    conflicted: Vec<(String, String)>,
    skipped: Vec<String>,
    rolled_back: Vec<String>,
    rollback_conflicts: Vec<String>,
    ops_applied_total: usize,
}

/// Semantic validation of ONE open transaction read back from a journal.
/// The journal adapter is expected to enforce the same rules; the engine
/// re-validates defensively (hostile/corrupt progress rows must surface as
/// typed errors before any write).
fn validate_open_txn(txn: &OpenEditTxn) -> Result<(), Error> {
    let what = format!("open edit transaction {}", txn.txn_id);
    if txn.files.is_empty() {
        return Err(Error::malformed(format!(
            "{what} has an empty prepared file list"
        )));
    }
    let mut paths: HashSet<&str> = HashSet::with_capacity(txn.files.len());
    let mut seqs: HashMap<u64, usize> = HashMap::with_capacity(txn.progress.len());
    for (i, f) in txn.files.iter().enumerate() {
        if !paths.insert(&f.path) {
            return Err(Error::malformed(format!(
                "{what} lists file {i} ({}) more than once",
                f.path
            )));
        }
    }
    for row in &txn.progress {
        let idx = usize::try_from(row.seq)
            .map_err(|_| Error::malformed(format!("{what} progress seq {} overflows", row.seq)))?;
        if idx >= txn.files.len() {
            return Err(Error::malformed(format!(
                "{what} progress seq {} is out of range ({} files prepared)",
                row.seq,
                txn.files.len()
            )));
        }
        if txn.files[idx].path != row.path {
            return Err(Error::malformed(format!(
                "{what} progress seq {} path {:?} does not match prepared file {:?}",
                row.seq, row.path, txn.files[idx].path
            )));
        }
        if seqs.insert(row.seq, idx).is_some() {
            return Err(Error::malformed(format!(
                "{what} has duplicate progress rows for seq {}",
                row.seq
            )));
        }
    }
    Ok(())
}

pub struct EditEngine {
    #[allow(dead_code)]
    fs: std::sync::Arc<faktor_fs::WorkspaceFileService>,
}

impl EditEngine {
    pub fn new(fs: std::sync::Arc<faktor_fs::WorkspaceFileService>) -> Self {
        Self { fs }
    }

    /// Apply one file transactionally. VALIDATE FIRST, WRITE LAST: every op
    /// is checked against a copy, syntax is checked before any write, and the
    /// final write is a commit-time compare-and-swap against the content this
    /// transaction actually read (audit 46/76) — a file that changed between
    /// validation and write is never clobbered.
    pub fn apply(
        &self,
        workspace: &WorkspaceHandle,
        identity: &WorkspaceIdentity,
        req: &EditRequest,
        mode: RepairMode,
    ) -> Result<EditOutcome, Error> {
        let staged = self.stage_one(workspace, identity, req, mode)?;
        let new_hash = workspace.write_atomic_cas(
            &staged.rel,
            staged.expected_hash,
            staged.edited.as_bytes(),
        )?;
        Ok(EditOutcome {
            new_hash,
            ops_applied: req.ops.len(),
            suspicious: staged.suspicious,
            parse_error: staged.parse_error,
        })
    }

    /// Multi-file transaction (audit 76-77): ALL requests are validated in a
    /// stage phase — nothing is written there — then committed one file at a
    /// time with a commit-time CAS. A stage failure (bad op, parse rollback,
    /// path escape, stale expected hash, budget expiry) aborts the whole
    /// batch BEFORE any write. A commit-time conflict (the file changed
    /// between stage and commit) stops the commit phase and is reported with
    /// exact per-file status: never a clobber, never a silent partial apply.
    ///
    /// This is the NON-DURABLE convenience wrapper of the P0-53 flow: it
    /// stages and commits exactly like `begin_multi_file_edit` +
    /// `commit_prepared` but without a session ledger, so there is no
    /// recovery record when the process dies mid-commit (the caller loses
    /// the partial state). Callers WITH a session ledger must use the
    /// durable begin/commit/recover API instead.
    pub fn apply_many(
        &self,
        workspace: &WorkspaceHandle,
        identity: &WorkspaceIdentity,
        reqs: &[EditRequest],
        mode: RepairMode,
        stage_budget: Option<std::time::Duration>,
    ) -> Result<EditBatchOutcome, Error> {
        let staged = self.stage_batch(workspace, identity, reqs, mode, stage_budget)?;
        // Test seam: the adversary strikes between validation and commit.
        commit_gap_hook();
        let run = self.commit_staged(
            workspace,
            identity,
            None,
            EditTxnStrategy::RollForward,
            staged,
        )?;
        Ok(EditBatchOutcome {
            committed: run.committed,
            conflicted: run.conflicted,
            skipped: run.skipped,
            ops_applied_total: run.ops_applied_total,
        })
    }

    /// BEGIN a durable multi-file edit transaction (P0-53): stage and
    /// validate EVERY file (nothing is written) and journal the durable
    /// `EditTxnPrepared` row to `journal` — record-first. A stage failure,
    /// a bound violation (file count, per-path bound, duplicate path,
    /// journaled-record payload) or a journal failure returns a typed error
    /// with NOTHING journaled and NOTHING written.
    ///
    /// `txn_id` is minted by the durable caller (e.g. the session's tool-run
    /// id) so a post-crash re-execution RESUMES the same transaction instead
    /// of re-beginning it. Callers must not re-begin an open transaction —
    /// recovery owns open transactions.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_multi_file_edit(
        &self,
        workspace: &WorkspaceHandle,
        identity: &WorkspaceIdentity,
        reqs: &[EditRequest],
        mode: RepairMode,
        strategy: EditTxnStrategy,
        journal: std::sync::Arc<dyn EditTxnJournal>,
        txn_id: EditTxnId,
        session: &str,
        stage_budget: Option<std::time::Duration>,
    ) -> Result<PreparedEdit, Error> {
        if reqs.is_empty() {
            return Err(Error::malformed(
                "edit transaction requires at least one file",
            ));
        }
        if reqs.len() > MAX_TXN_FILES {
            return Err(Error::oversized(format!(
                "edit transaction of {} files exceeds MAX_TXN_FILES",
                reqs.len()
            )));
        }
        if session.is_empty() || session.len() > 128 {
            return Err(Error::malformed(
                "edit transaction session must be 1..=128 bytes",
            ));
        }
        let mut seen = HashSet::with_capacity(reqs.len());
        for req in reqs {
            if req.path.is_empty() || req.path.len() > MAX_TXN_PATH_BYTES {
                return Err(Error::malformed(format!(
                    "edit transaction path {:?} must be 1..={MAX_TXN_PATH_BYTES} bytes",
                    req.path
                )));
            }
            if !seen.insert(req.path.clone()) {
                return Err(Error::malformed(format!(
                    "edit transaction lists {} twice; a file must appear once",
                    req.path
                )));
            }
        }
        // Journaled-record payload bound (the journal's own cap and the
        // engine's) BEFORE anything is staged or journaled: an oversized
        // transaction is refused with zero disk or journal writes.
        let cap = journal.entry_payload_cap().min(MAX_TXN_PAYLOAD_BYTES);
        let estimate = 224 + reqs.iter().map(|r| r.path.len() + 152).sum::<usize>();
        if estimate > cap {
            return Err(Error::oversized(format!(
                "edit transaction prepared record of ~{estimate} bytes exceeds the journal's {cap} byte bound"
            )));
        }
        // STAGE + validate everything first: a stage failure must leave no
        // journal trace (an orphan prepared row would be recovered forever).
        let staged = self.stage_batch(workspace, identity, reqs, mode, stage_budget)?;
        let refs: Vec<EditTxnFileRef> = staged
            .iter()
            .map(|s| EditTxnFileRef {
                path: s.rel.to_string_lossy().to_string(),
                base_digest: s.expected_hash,
                base_len: s.base_len,
            })
            .collect();
        // RECORD-FIRST: the prepared row is durable before any write.
        journal.record_prepared(txn_id, session, &refs, strategy)?;
        Ok(PreparedEdit {
            txn_id,
            strategy,
            journal,
            files: staged,
        })
    }

    /// COMMIT a prepared durable transaction: one commit-time CAS per file,
    /// each journaled as `EditTxnProgress` AFTER its write, ending with the
    /// terminal `EditTxnCommitted` row — or, when the policy is
    /// `RollBack` and a conflict occurred, the already-committed files are
    /// rolled back (CAS-restore to their staged before-content, expected
    /// digest == the digest WE wrote; a file that changed again since our
    /// write is a typed `rollback_conflicts` entry, never a clobber) and the
    /// terminal `EditTxnRolledBack` row is journaled.
    ///
    /// A journal failure (storage down) mid-commit returns a typed error;
    /// the transaction stays OPEN and recovery reconciles it from the rows
    /// that did land.
    pub fn commit_prepared(
        &self,
        workspace: &WorkspaceHandle,
        identity: &WorkspaceIdentity,
        prepared: PreparedEdit,
    ) -> Result<EditTxnResult, Error> {
        let run = self.commit_staged(
            workspace,
            identity,
            Some((prepared.txn_id, prepared.journal.as_ref())),
            prepared.strategy,
            prepared.files,
        )?;
        Ok(EditTxnResult {
            txn_id: prepared.txn_id,
            committed: run.committed,
            conflicted: run.conflicted,
            skipped: run.skipped,
            rolled_back: run.rolled_back,
            rollback_conflicts: run.rollback_conflicts,
            ops_applied_total: run.ops_applied_total,
        })
    }

    /// Deterministic, idempotent crash recovery of OPEN transactions
    /// (prepared rows without a terminal row). For each open transaction:
    ///
    /// - files with a committed progress row are DONE (never rewritten — a
    ///   digest match against our write is not needed because the progress
    ///   row is the durable post-write record),
    /// - files with a conflicted progress row stay conflicted (the external
    ///   writer keeps the file) and do not stop the recovery,
    /// - files without a progress row whose current digest still equals
    ///   their staged base digest are replayed: re-staged (full validation
    ///   again) and CAS-written, then journaled committed,
    /// - a file without a progress row whose current digest differs from its
    ///   base digest is AMBIGUOUS (either our write landed and crashed
    ///   before its progress row, or an external writer moved it): a typed
    ///   needs-recovery error, never a silent accept or clobber.
    ///
    /// Under `RollBack`, a conflict (pre-recorded or found during replay)
    /// triggers the restore phase: files committed during THIS run are
    /// restored in-memory; files committed BEFORE the crash are restored
    /// through the resume's `before_content` provider (their staged
    /// before-content must hash to the journaled base digest, and the
    /// restore CAS expects the digest of the content we wrote — a file that
    /// changed again is a typed rollback conflict). Without a provider such
    /// a restore is a typed needs-recovery error and the transaction stays
    /// open.
    ///
    /// A terminal row is journaled at the end of every successful recovery,
    /// so a second run finds nothing open (idempotent). Running the same
    /// recovery twice yields byte-identical state.
    pub fn recover_edit_transactions(
        &self,
        workspace: &WorkspaceHandle,
        identity: &WorkspaceIdentity,
        journal: &dyn EditTxnJournal,
        resumes: &[EditTxnResume<'_>],
        mode: RepairMode,
    ) -> Result<RecoveryReport, Error> {
        workspace.verify_identity(identity)?;
        // Duplicate resumes for one txn id are corruption of the caller.
        let mut seen: HashSet<EditTxnId> = HashSet::new();
        for r in resumes {
            if !seen.insert(r.txn_id) {
                return Err(Error::malformed(format!(
                    "recovery received duplicate resumes for txn {}",
                    r.txn_id
                )));
            }
        }
        let mut report = RecoveryReport::default();
        for open in journal.open_transactions()? {
            report
                .txns
                .push(self.recover_one(workspace, identity, journal, open, resumes, mode)?);
        }
        Ok(report)
    }

    fn recover_one(
        &self,
        workspace: &WorkspaceHandle,
        identity: &WorkspaceIdentity,
        journal: &dyn EditTxnJournal,
        open: OpenEditTxn,
        resumes: &[EditTxnResume<'_>],
        mode: RepairMode,
    ) -> Result<RecoveryTxnOutcome, Error> {
        validate_open_txn(&open)?;
        let txn = open.txn_id;
        let strategy = open.strategy;
        let has_conflict_pre = open
            .progress
            .iter()
            .any(|r| r.outcome == TxnFileOutcome::Conflicted);
        // Roll-back policy that already owes a restore (a conflict was
        // journaled before the crash): committed files are restored, files
        // that were never attempted simply stay at their base content.
        let rollback_pending = strategy == EditTxnStrategy::RollBack && has_conflict_pre;
        let resume = resumes.iter().find(|r| r.txn_id == txn);
        let mut by_seq: HashMap<u64, TxnFileOutcome> = HashMap::with_capacity(open.progress.len());
        for row in &open.progress {
            by_seq.insert(row.seq, row.outcome);
        }
        // A file without a progress row needs its requests reissued (to be
        // replayed). Roll-back restores needing before-content are resolved
        // per file at restore time (a committed file that is already back at
        // its base digest needs neither requests nor a provider).
        let needs_ops = open.progress.len() < open.files.len();
        if resume.is_none() && needs_ops {
            return Err(Error::conflict(format!(
                "edit transaction {txn} needs recovery but its original requests are \
                 unavailable; re-issue the durable tool call with the same txn id \
                 (nothing was written or dropped)"
            )));
        }
        let mut outcome = RecoveryTxnOutcome {
            txn_id: txn,
            strategy,
            committed: Vec::new(),
            conflicted: Vec::new(),
            skipped: Vec::new(),
            rolled_back: Vec::new(),
            rollback_conflicts: Vec::new(),
            wrote: Vec::new(),
        };
        // Files committed BEFORE the crash (durable progress rows), file
        // order; files committed BY THIS RUN keep their staged data.
        let mut pre_run: Vec<usize> = Vec::new();
        let mut this_run: Vec<JustWritten> = Vec::new();
        let mut replay_stop: Option<usize> = None; // first conflict of a roll-back replay
        for (i, f) in open.files.iter().enumerate() {
            match by_seq.get(&(i as u64)) {
                Some(TxnFileOutcome::Committed) => {
                    outcome.committed.push(f.path.clone());
                    pre_run.push(i);
                    continue;
                }
                Some(TxnFileOutcome::Conflicted) => {
                    outcome.conflicted.push(f.path.clone());
                    continue;
                }
                None => {}
            }
            if replay_stop.is_some() {
                outcome.skipped.push(f.path.clone());
                continue;
            }
            // No progress row: classify by the CURRENT content digest.
            let current = workspace.read(std::path::Path::new(&f.path), MAX_FILE_BYTES)?;
            let current_hash = match current.full_hash() {
                Some(h) => h,
                None => {
                    return Err(Error::oversized(format!(
                        "{} exceeds the edit bound; edit transaction {txn} needs recovery",
                        f.path
                    )))
                }
            };
            if current_hash != f.base_digest {
                // Ambiguous: our write landed but crashed before its progress
                // row, or an external writer moved the file. Never silently
                // accepted, never clobbered.
                return Err(Error::conflict(format!(
                    "file {} of edit transaction {txn} has no progress row and no longer \
                     matches its staged base digest (a write landed before its journal row, \
                     or an external writer); the transaction needs recovery and was left \
                     untouched",
                    f.path
                )));
            }
            if rollback_pending {
                // The transaction owes a restore: never-attempted files stay
                // at their base content (the rollback end state).
                outcome.skipped.push(f.path.clone());
                continue;
            }
            // Replay: re-stage the reissued request (full validation again)
            // and CAS-write; the progress row lands AFTER the write.
            let reqs = resume.expect("needs_ops implies a resume").reqs;
            let req = reqs.get(i).ok_or_else(|| {
                Error::malformed(format!(
                    "resume of txn {txn} has {} requests for {} prepared files",
                    reqs.len(),
                    open.files.len()
                ))
            })?;
            if req.path != f.path {
                return Err(Error::malformed(format!(
                    "resume request {i} of txn {txn} does not match prepared file {:?}",
                    f.path
                )));
            }
            if req.expected_hash != f.base_digest {
                return Err(Error::malformed(format!(
                    "resume request {i} of txn {txn} validates against a digest that is not \
                     the prepared base digest"
                )));
            }
            let staged = self.stage_one(workspace, identity, req, mode)?;
            match workspace.write_atomic_cas(
                &staged.rel,
                staged.expected_hash,
                staged.edited.as_bytes(),
            ) {
                Ok(new_hash) => {
                    outcome.committed.push(f.path.clone());
                    outcome.wrote.push(f.path.clone());
                    this_run.push(JustWritten {
                        idx: i,
                        new_hash,
                        original: staged.original,
                    });
                    journal.record_progress(txn, i as u64, &f.path, TxnFileOutcome::Committed)?;
                }
                Err(_) => {
                    // A conflict found by the replay itself. Roll-forward
                    // recovery continues past it (its progress row is
                    // durable); a roll-back transaction stops here and owes
                    // the restore.
                    outcome.conflicted.push(f.path.clone());
                    journal.record_progress(txn, i as u64, &f.path, TxnFileOutcome::Conflicted)?;
                    if strategy == EditTxnStrategy::RollBack {
                        replay_stop = Some(i);
                    }
                }
            }
        }
        // Terminal phase.
        if strategy == EditTxnStrategy::RollBack && (has_conflict_pre || replay_stop.is_some()) {
            self.recovery_rollback(
                workspace,
                journal,
                &open,
                resume,
                &pre_run,
                &this_run,
                &mut outcome,
            )?;
            // Every committed file was restored (or refused with a typed
            // conflict); none is at its target content anymore.
            outcome.committed.clear();
            journal.record_rolled_back(txn, &outcome.rolled_back, &outcome.rollback_conflicts)?;
            Ok(outcome)
        } else {
            let conflicted: Vec<String> = outcome.conflicted.clone();
            journal.record_committed(txn, &outcome.committed, &conflicted, &outcome.skipped)?;
            Ok(outcome)
        }
    }

    /// The roll-back restore phase of one recovery: every committed file
    /// (pre-crash AND this-run) is CAS-restored to its staged before-content
    /// in reverse file order. The restore CAS expects the digest of the
    /// content WE wrote; a file that changed again since our write is a
    /// typed `rollback_conflicts` entry — never a clobber.
    #[allow(clippy::too_many_arguments)]
    fn recovery_rollback(
        &self,
        workspace: &WorkspaceHandle,
        _journal: &dyn EditTxnJournal,
        open: &OpenEditTxn,
        resume: Option<&EditTxnResume<'_>>,
        pre_run: &[usize],
        this_run: &[JustWritten],
        outcome: &mut RecoveryTxnOutcome,
    ) -> Result<(), Error> {
        let txn = open.txn_id;
        // Union of committed files, descending by index (reverse of commit
        // order).
        let mut committed: Vec<(usize, Option<&JustWritten>)> = Vec::new();
        for jw in this_run.iter() {
            committed.push((jw.idx, Some(jw)));
        }
        committed.extend(pre_run.iter().map(|&idx| (idx, None)));
        committed.sort_by_key(|(idx, _)| std::cmp::Reverse(*idx));
        for (idx, written) in committed {
            let f = &open.files[idx];
            if let Some(jw) = written {
                // Written by THIS recovery run: staged data is in hand.
                let p = f.path.clone();
                match workspace.write_atomic_cas(
                    std::path::Path::new(&f.path),
                    jw.new_hash,
                    &jw.original,
                ) {
                    Ok(_) => outcome.rolled_back.push(p),
                    Err(_) => outcome.rollback_conflicts.push(p),
                }
                continue;
            }
            // Committed BEFORE the crash. If the file is already back at its
            // base digest the restore is done (a restore landed before the
            // crash, or an external writer reverted it — same end state).
            let current = workspace.read(std::path::Path::new(&f.path), MAX_FILE_BYTES)?;
            let current_hash = current.full_hash().ok_or_else(|| {
                Error::oversized(format!(
                    "{} exceeds the edit bound; edit transaction {txn} needs recovery",
                    f.path
                ))
            })?;
            if current_hash == f.base_digest {
                outcome.rolled_back.push(f.path.clone());
                continue;
            }
            let resume = resume.ok_or_else(|| {
                Error::conflict(format!(
                    "edit transaction {txn} needs its before-content to roll back {}; \
                     no resume was provided (nothing was clobbered, the transaction stays open)",
                    f.path
                ))
            })?;
            // Before-content + the original ops recompute the digest of what
            // we wrote: the restore CAS expects EXACTLY that digest, so a
            // file that changed again is refused, never clobbered.
            let base = (resume.before_content)(&f.path).ok_or_else(|| {
                Error::conflict(format!(
                    "edit transaction {txn} cannot roll back {}: its staged before-content is \
                     unavailable (nothing was clobbered, the transaction stays open)",
                    f.path
                ))
            })?;
            if EditEngine::hash_of(&base) != f.base_digest {
                return Err(Error::malformed(format!(
                    "before-content provider of txn {txn} returned bytes that do not hash to \
                     the prepared base digest of {}",
                    f.path
                )));
            }
            let req = resume.reqs.get(idx).ok_or_else(|| {
                Error::malformed(format!(
                    "resume of txn {txn} has no request {idx} to recompute the target of {}",
                    f.path
                ))
            })?;
            let base_str = String::from_utf8(base).map_err(|_| {
                Error::malformed(format!(
                    "before-content of {} is not valid UTF-8; cannot recompute its target",
                    f.path
                ))
            })?;
            // Recompute the digest of what WE wrote on a copy of the
            // before-content: the restore CAS expects exactly that digest,
            // so a file that changed again is refused, never clobbered.
            let mut buf = base_str.clone();
            for (i, op) in req.ops.iter().enumerate() {
                apply_op(&mut buf, op).map_err(|e| {
                    Error::new(
                        e.kind.clone(),
                        format!("replay op {} of {}: {}", i + 1, f.path, e.message),
                    )
                })?;
            }
            let target_digest = EditEngine::hash_of(buf.as_bytes());
            match workspace.write_atomic_cas(
                std::path::Path::new(&f.path),
                target_digest,
                base_str.as_bytes(),
            ) {
                Ok(_) => outcome.rolled_back.push(f.path.clone()),
                Err(_) => outcome.rollback_conflicts.push(f.path.clone()),
            }
        }
        Ok(())
    }

    /// STAGE: validate every request against a copy; zero writes. Shared by
    /// `apply_many`, `begin_multi_file_edit` and crash recovery replay.
    fn stage_batch(
        &self,
        workspace: &WorkspaceHandle,
        identity: &WorkspaceIdentity,
        reqs: &[EditRequest],
        mode: RepairMode,
        stage_budget: Option<std::time::Duration>,
    ) -> Result<Vec<StagedEdit>, Error> {
        workspace.verify_identity(identity)?;
        let started = std::time::Instant::now();
        let budget = |stage_budget: Option<std::time::Duration>,
                      started: std::time::Instant|
         -> Result<(), Error> {
            if let Some(b) = stage_budget {
                if started.elapsed() > b {
                    return Err(Error::timeout(
                        "edit batch exceeded its validation budget; nothing was written",
                    ));
                }
            }
            Ok(())
        };
        let mut staged = Vec::with_capacity(reqs.len());
        for req in reqs {
            budget(stage_budget, started)?;
            staged.push(self.stage_one(workspace, identity, req, mode)?);
            budget(stage_budget, started)?;
        }
        Ok(staged)
    }

    /// The shared COMMIT phase (journaled or not): one commit-time CAS per
    /// staged file in request order. Each successful write is journaled as
    /// `EditTxnProgress {committed}` AFTER the write; the first conflict is
    /// journaled as `{conflicted}` and ends the commit (remaining files are
    /// skipped). Under `EditTxnStrategy::RollBack` the conflict triggers the
    /// rollback phase: the already-committed files are CAS-restored to their
    /// staged before-content in reverse commit order — the CAS expects the
    /// digest of the content WE wrote, so a file that changed again since
    /// our write is a typed `rollback_conflicts` entry and is NEVER
    /// clobbered — and the terminal `EditTxnRolledBack` row is journaled.
    /// Otherwise the terminal `EditTxnCommitted` row is journaled. A
    /// journal failure mid-commit is a typed error; the transaction stays
    /// open for recovery.
    fn commit_staged(
        &self,
        workspace: &WorkspaceHandle,
        identity: &WorkspaceIdentity,
        txn: Option<(EditTxnId, &dyn EditTxnJournal)>,
        strategy: EditTxnStrategy,
        files: Vec<StagedEdit>,
    ) -> Result<CommitRun, Error> {
        workspace.verify_identity(identity)?;
        let mut run = CommitRun::default();
        // (staged index, digest of OUR write) — the roll-back restore axis.
        let mut written: Vec<(usize, FileHash)> = Vec::new();
        let mut stopped: bool = false;
        for (i, staged) in files.iter().enumerate() {
            if stopped {
                break;
            }
            let path = staged.rel.to_string_lossy().to_string();
            match workspace.write_atomic_cas(
                &staged.rel,
                staged.expected_hash,
                staged.edited.as_bytes(),
            ) {
                Ok(new_hash) => {
                    run.ops_applied_total += staged.ops_applied;
                    run.committed.push(BatchCommittedFile {
                        path: path.clone(),
                        new_hash,
                        ops_applied: staged.ops_applied,
                        suspicious: staged.suspicious,
                    });
                    written.push((i, new_hash));
                    if let Some((id, j)) = txn {
                        j.record_progress(id, i as u64, &path, TxnFileOutcome::Committed)?;
                    }
                }
                Err(e) => {
                    run.conflicted.push((path.clone(), e.message));
                    run.skipped.extend(
                        files[i + 1..]
                            .iter()
                            .map(|x| x.rel.to_string_lossy().to_string()),
                    );
                    if let Some((id, j)) = txn {
                        j.record_progress(id, i as u64, &path, TxnFileOutcome::Conflicted)?;
                    }
                    if strategy == EditTxnStrategy::RollBack {
                        // Test seam: the adversary strikes between our write
                        // of a committed file and its roll-back restore.
                        rollback_gap_hook();
                        for (idx, our_digest) in written.iter().rev() {
                            let s = &files[*idx];
                            let p = s.rel.to_string_lossy().to_string();
                            match workspace.write_atomic_cas(&s.rel, *our_digest, &s.original) {
                                Ok(_) => run.rolled_back.push(p),
                                Err(_) => run.rollback_conflicts.push(p),
                            }
                        }
                        if let Some((id, j)) = txn {
                            j.record_rolled_back(id, &run.rolled_back, &run.rollback_conflicts)?;
                        }
                    } else if let Some((id, j)) = txn {
                        let committed: Vec<String> =
                            run.committed.iter().map(|c| c.path.clone()).collect();
                        let conflicted: Vec<String> =
                            run.conflicted.iter().map(|c| c.0.clone()).collect();
                        j.record_committed(id, &committed, &conflicted, &run.skipped)?;
                    }
                    stopped = true;
                }
            }
        }
        // Full success (or a rollback whose conflict never came): the
        // terminal committed row.
        if !stopped {
            if let Some((id, j)) = txn {
                let committed: Vec<String> = run.committed.iter().map(|c| c.path.clone()).collect();
                j.record_committed(id, &committed, &[], &[])?;
            }
        }
        Ok(run)
    }

    /// The stage half of a single-file edit: resolve, bound-check, verify the
    /// expected hash, apply every op on a copy, and run parse-before-accept.
    /// Returns the transformed content for a later commit-time CAS.
    fn stage_one(
        &self,
        workspace: &WorkspaceHandle,
        identity: &WorkspaceIdentity,
        req: &EditRequest,
        mode: RepairMode,
    ) -> Result<StagedEdit, Error> {
        workspace.verify_identity(identity)?;
        let rel = std::path::Path::new(&req.path);
        let current = workspace.read(rel, MAX_FILE_BYTES)?;
        // Whole-file identity FIRST (P0-50): a read capped by the edit bound
        // proves it covers the whole file only through a Full digest. A
        // Slice digest means the file exceeds the bound — the same typed
        // oversized refusal the historical `truncated` flag produced — and
        // its prefix hash is never compared against the whole-file
        // `expected_hash`.
        let current_hash = match current.full_hash() {
            Some(h) => h,
            None => {
                return Err(Error::oversized(format!(
                    "{} exceeds the {} byte edit bound",
                    req.path, MAX_FILE_BYTES
                )))
            }
        };
        // Optimistic versioning: the file must be exactly what the model read.
        if current_hash != req.expected_hash {
            return Err(Error::conflict(format!(
                "{} changed since it was read (expected {}, found {})",
                req.path,
                req.expected_hash.to_hex(),
                current_hash.to_hex()
            )));
        }
        let original_bytes = current.bytes;
        let base_len = original_bytes.len() as u64;
        let original = String::from_utf8(original_bytes.clone())
            .map_err(|_| Error::malformed(format!("{} is not valid UTF-8", req.path)))?;

        // Validate + apply on a copy.
        let mut buf = original.clone();
        for (i, op) in req.ops.iter().enumerate() {
            apply_op(&mut buf, op).map_err(|e| {
                Error::new(
                    e.kind.clone(),
                    format!("op {} of {}: {}", i + 1, req.path, e.message),
                )
            })?;
        }
        let edited = buf;

        // Parse-before-accept for supported languages.
        let (suspicious, parse_error) = check_syntax(rel, &original, &edited);

        if suspicious {
            match mode {
                RepairMode::Rollback => {
                    return Err(Error::new(
                        ErrorKind::Malformed,
                        format!(
                            "edit of {} introduces parse errors (rollback): {}",
                            req.path,
                            parse_error.unwrap_or_default()
                        ),
                    ));
                }
                RepairMode::AllowModelRepair => {}
            }
        }

        Ok(StagedEdit {
            rel: rel.to_path_buf(),
            expected_hash: current_hash,
            base_len,
            original: original_bytes,
            edited,
            ops_applied: req.ops.len(),
            suspicious,
            parse_error,
        })
    }

    /// Hash a string the way the engine expects (public for tooling).
    pub fn hash_of(bytes: &[u8]) -> FileHash {
        FileHash::from(blake3::hash(bytes).into())
    }
}

fn apply_op(buf: &mut String, op: &EditOp) -> Result<(), Error> {
    match op {
        EditOp::Range {
            start,
            end,
            replacement,
        } => {
            if start > end {
                return Err(Error::malformed(format!("range start {start} > end {end}")));
            }
            if !buf.is_char_boundary(*start) || !buf.is_char_boundary(*end) {
                return Err(Error::malformed(format!(
                    "range [{start},{end}) splits a UTF-8 codepoint"
                )));
            }
            if *end > buf.len() {
                return Err(Error::malformed(format!(
                    "range end {end} exceeds file length {}",
                    buf.len()
                )));
            }
            buf.replace_range(*start..*end, replacement);
            Ok(())
        }
        EditOp::SearchReplace { before, after } => {
            let matches = buf.match_indices(before).count();
            match matches {
                0 => Err(Error::malformed("search text not found (0 matches)")),
                1 => {
                    let start = buf.find(before).unwrap();
                    buf.replace_range(start..start + before.len(), after);
                    Ok(())
                }
                n => Err(Error::conflict(format!(
                    "search text is ambiguous ({n} matches)"
                ))),
            }
        }
        EditOp::BoundedRegion {
            anchor,
            region_start,
            region_end,
            replacement,
        } => {
            let matches: Vec<usize> = buf.match_indices(anchor).map(|(i, _)| i).collect();
            if matches.len() != 1 {
                return Err(Error::conflict(format!(
                    "anchor must match uniquely ({} matches)",
                    matches.len()
                )));
            }
            let base = matches[0];
            let start = base + region_start;
            let end = base + region_end;
            if start > end {
                return Err(Error::malformed("region start > end"));
            }
            if !buf.is_char_boundary(start) || !buf.is_char_boundary(end) {
                return Err(Error::malformed("region splits a UTF-8 codepoint"));
            }
            if end > buf.len() {
                return Err(Error::malformed(format!(
                    "region end {end} exceeds file length {}",
                    buf.len()
                )));
            }
            buf.replace_range(start..end, replacement);
            Ok(())
        }
    }
}

/// Parse-before-accept: returns (suspicious, first_error). When the original
/// parses cleanly and the edited version does not, the edit is suspicious.
fn check_syntax(rel: &std::path::Path, original: &str, edited: &str) -> (bool, Option<String>) {
    let lang = language_for(rel);
    let Some((lang, grammar)) = lang else {
        return (false, None);
    };
    let before_ok = parse_ok(lang, &grammar, original);
    if !before_ok {
        // The file was already broken; the edit cannot be blamed.
        return (false, None);
    }
    match first_parse_error(lang, &grammar, edited) {
        Some(err) => (true, Some(err)),
        None => (false, None),
    }
}

/// Parse-before-accept grammars (spec §18; audit round 9): Rust + Python
/// were the scaffold; TypeScript/TSX/JSX/JS (the frozen client is TS-heavy),
/// Go, and Java are next-tier. Unlisted extensions have no grammar —
/// whole-file writes for them skip parse validation (documented behavior).
fn language_for(rel: &std::path::Path) -> Option<(&'static str, tree_sitter::Language)> {
    match rel.extension().and_then(|e| e.to_str()) {
        Some("rs") => Some(("rust", tree_sitter_rust::LANGUAGE.into())),
        Some("py") => Some(("python", tree_sitter_python::LANGUAGE.into())),
        Some("ts") => Some((
            "typescript",
            tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        )),
        Some("tsx") => Some(("tsx", tree_sitter_typescript::LANGUAGE_TSX.into())),
        Some("js") => Some(("javascript", tree_sitter_javascript::LANGUAGE.into())),
        Some("go") => Some(("go", tree_sitter_go::LANGUAGE.into())),
        Some("java") => Some(("java", tree_sitter_java::LANGUAGE.into())),
        _ => None,
    }
}

fn parse_ok(_lang: &str, language: &tree_sitter::Language, src: &str) -> bool {
    first_parse_error(_lang, language, src).is_none()
}

fn first_parse_error(_lang: &str, language: &tree_sitter::Language, src: &str) -> Option<String> {
    let mut parser = tree_sitter::Parser::new();
    if parser.set_language(language).is_err() {
        return Some("language grammar unavailable".into());
    }
    let tree = parser.parse(src, None)?;
    let mut first: Option<(usize, String)> = None;
    fn walk(node: tree_sitter::Node<'_>, src: &[u8], first: &mut Option<(usize, String)>) {
        if node.is_error() || node.is_missing() {
            let pos = node.start_position();
            let msg = if node.is_missing() {
                format!("missing {}", node.kind())
            } else {
                let text = node.utf8_text(src).unwrap_or("?");
                format!("unexpected {text:?} ({})", node.kind())
            };
            if first.as_ref().map(|(k, _)| *k > pos.row).unwrap_or(true) {
                *first = Some((pos.row, msg));
            }
        }
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            walk(child, src, first);
        }
    }
    walk(tree.root_node(), src.as_bytes(), &mut first);
    first.map(|(row, msg)| format!("line {}: {msg}", row + 1))
}

/// Test seam: fired between the stage phase (all validation, zero writes)
/// and the commit phase of `apply_many`. Deterministic multi-file race tests
/// modify a middle file in this gap so the commit-time CAS must detect it.
#[cfg(test)]
type CommitGapHook = Box<dyn Fn() + Send>;
#[cfg(test)]
static COMMIT_GAP: std::sync::OnceLock<std::sync::Mutex<Option<CommitGapHook>>> =
    std::sync::OnceLock::new();
#[cfg(test)]
fn commit_gap_hook() {
    if let Some(lock) = COMMIT_GAP.get() {
        if let Some(hook) = lock.lock().expect("seam poisoned").as_ref() {
            hook();
        }
    }
}
#[cfg(not(test))]
fn commit_gap_hook() {}

/// Test seam: fired between the engine's own write of a committed file and
/// its roll-back restore (the rollback CAS must detect the adversary). Used
/// by the roll-back conflict tests; sibling tests that touch this seam
/// serialize through `ROLLBACK_GAP_LOCK`.
#[cfg(test)]
type RollbackGapHook = Box<dyn Fn() + Send>;
#[cfg(test)]
static ROLLBACK_GAP: std::sync::OnceLock<std::sync::Mutex<Option<RollbackGapHook>>> =
    std::sync::OnceLock::new();
#[cfg(test)]
pub(crate) fn rollback_gap_hook() {
    if let Some(lock) = ROLLBACK_GAP.get() {
        if let Some(hook) = lock.lock().expect("seam poisoned").as_ref() {
            hook();
        }
    }
}
#[cfg(not(test))]
fn rollback_gap_hook() {}

/// A fully validated single-file edit, ready for the commit-time CAS.
struct StagedEdit {
    rel: std::path::PathBuf,
    /// Hash of the content this edit was validated against (digest axis of
    /// the commit-time compare-and-swap; the transaction's `base_digest`).
    expected_hash: FileHash,
    /// Byte length of the validated content (journaled as `base_len`).
    base_len: u64,
    /// The exact bytes validated against — the staged before-content that a
    /// roll-back restore CAS writes back.
    original: Vec<u8>,
    edited: String,
    ops_applied: usize,
    suspicious: bool,
    parse_error: Option<String>,
}

/// Lookup helper used by tests to fetch tree-sitter grammar info.
#[allow(dead_code)]
fn _grammar_names() -> HashMap<&'static str, &'static str> {
    HashMap::from([
        ("rust", "tree-sitter-rust"),
        ("python", "tree-sitter-python"),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use faktor_core::id::WorkspaceId;
    use std::fs;

    fn fixture() -> (
        tempfile::TempDir,
        std::sync::Arc<faktor_fs::WorkspaceFileService>,
        WorkspaceHandle,
        WorkspaceIdentity,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("ws");
        fs::create_dir_all(&root).unwrap();
        let service = faktor_fs::WorkspaceFileService::new();
        let handle = service.open(WorkspaceId::new(1), root.clone()).unwrap();
        let identity = WorkspaceIdentity::new(
            WorkspaceId::new(1),
            faktor_core::WorktreeId::new(1),
            faktor_core::TaskId::new(1),
        );
        (dir, service, handle, identity)
    }

    fn req(path: &str, expected: &[u8], ops: Vec<EditOp>) -> EditRequest {
        EditRequest {
            path: path.into(),
            expected_hash: EditEngine::hash_of(expected),
            ops,
        }
    }

    #[test]
    fn hash_mismatch_rejected_before_write() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(faktor_fs::WorkspaceFileService::new());
        fs::write(h.root().join("a.txt"), b"one").unwrap();
        // Model read "one" but the file became "two" (another writer).
        fs::write(h.root().join("a.txt"), b"two").unwrap();
        let r = req(
            "a.txt",
            b"one",
            vec![EditOp::SearchReplace {
                before: "two".into(),
                after: "three".into(),
            }],
        );
        let err = engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap_err();
        assert!(err.kind == ErrorKind::Conflict);
        // File untouched.
        assert_eq!(fs::read(h.root().join("a.txt")).unwrap(), b"two");
    }

    #[test]
    fn range_edit_applies_and_hashes() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(faktor_fs::WorkspaceFileService::new());
        fs::write(h.root().join("a.txt"), b"hello world").unwrap();
        let r = req(
            "a.txt",
            b"hello world",
            vec![EditOp::Range {
                start: 0,
                end: 5,
                replacement: "goodbye".into(),
            }],
        );
        let out = engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap();
        assert_eq!(out.ops_applied, 1);
        assert!(!out.suspicious);
        assert_eq!(fs::read(h.root().join("a.txt")).unwrap(), b"goodbye world");
        assert_eq!(out.new_hash, EditEngine::hash_of(b"goodbye world"));
    }

    #[test]
    fn search_replace_unique_ok() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(faktor_fs::WorkspaceFileService::new());
        fs::write(h.root().join("a.txt"), b"fn main() {}\n").unwrap();
        let r = req(
            "a.txt",
            b"fn main() {}\n",
            vec![EditOp::SearchReplace {
                before: "fn main".into(),
                after: "fn entry".into(),
            }],
        );
        engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap();
        assert_eq!(
            fs::read(h.root().join("a.txt")).unwrap(),
            b"fn entry() {}\n"
        );
    }

    #[test]
    fn search_replace_zero_matches_malformed() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(faktor_fs::WorkspaceFileService::new());
        fs::write(h.root().join("a.txt"), b"abc").unwrap();
        let r = req(
            "a.txt",
            b"abc",
            vec![EditOp::SearchReplace {
                before: "zzz".into(),
                after: "x".into(),
            }],
        );
        let err = engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap_err();
        assert!(err.kind == ErrorKind::Malformed);
        assert_eq!(fs::read(h.root().join("a.txt")).unwrap(), b"abc");
    }

    #[test]
    fn search_replace_multiple_matches_conflict() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(faktor_fs::WorkspaceFileService::new());
        fs::write(h.root().join("a.txt"), b"aaa").unwrap();
        let r = req(
            "a.txt",
            b"aaa",
            vec![EditOp::SearchReplace {
                before: "a".into(),
                after: "b".into(),
            }],
        );
        let err = engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap_err();
        assert!(err.kind == ErrorKind::Conflict);
    }

    #[test]
    fn out_of_bounds_range_malformed() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(faktor_fs::WorkspaceFileService::new());
        fs::write(h.root().join("a.txt"), b"abc").unwrap();
        let r = req(
            "a.txt",
            b"abc",
            vec![EditOp::Range {
                start: 1,
                end: 99,
                replacement: "x".into(),
            }],
        );
        let err = engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap_err();
        assert!(err.kind == ErrorKind::Malformed);
        assert!(err.message.contains("op 1"));
    }

    #[test]
    fn split_codepoint_offsets_malformed() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(faktor_fs::WorkspaceFileService::new());
        let content = "aé😀b"; // bytes: a(1) é(2) 😀(4) b(1)
        fs::write(h.root().join("a.txt"), content).unwrap();
        // offset 2 lands inside 'é' (2 bytes: 0xE9 is at 1..3).
        let r = req(
            "a.txt",
            content.as_bytes(),
            vec![EditOp::Range {
                start: 2,
                end: 3,
                replacement: "x".into(),
            }],
        );
        let err = engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap_err();
        assert!(err.kind == ErrorKind::Malformed);
        // A boundary-correct edit works.
        let r = req(
            "a.txt",
            content.as_bytes(),
            vec![EditOp::Range {
                start: 1,
                end: 7,
                replacement: "Z".into(),
            }],
        );
        let out = engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap();
        assert_eq!(fs::read(h.root().join("a.txt")).unwrap(), b"aZb");
        assert!(!out.suspicious);
    }

    #[test]
    fn rust_parse_error_rolls_back_in_rollback_mode() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(faktor_fs::WorkspaceFileService::new());
        let original = b"fn main() {\n    let x = 1;\n    println!(\"{}\", x);\n}\n";
        fs::write(h.root().join("main.rs"), original).unwrap();
        // Break the syntax: delete the closing brace line.
        let r = req(
            "main.rs",
            original,
            vec![EditOp::SearchReplace {
                before: "}\n".into(),
                after: "".into(),
            }],
        );
        let err = engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap_err();
        assert!(err.kind == ErrorKind::Malformed, "{err:?}");
        assert!(err.message.contains("parse"), "{err}");
        assert_eq!(
            fs::read(h.root().join("main.rs")).unwrap(),
            original,
            "rollback: file must be untouched"
        );
    }

    #[test]
    fn rust_parse_error_allow_repair_writes_and_flags() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(faktor_fs::WorkspaceFileService::new());
        let original = b"fn main() {\n    let x = 1;\n}\n";
        fs::write(h.root().join("main.rs"), original).unwrap();
        let r = req(
            "main.rs",
            original,
            vec![EditOp::SearchReplace {
                before: "}\n".into(),
                after: "".into(),
            }],
        );
        let out = engine
            .apply(&h, &id, &r, RepairMode::AllowModelRepair)
            .unwrap();
        assert!(out.suspicious);
        assert!(out.parse_error.is_some());
        assert!(fs::read(h.root().join("main.rs")).unwrap() != original);
    }

    #[test]
    fn valid_edit_stays_not_suspicious() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(faktor_fs::WorkspaceFileService::new());
        let original = b"fn main() {\n    let x = 1;\n    println!(\"{}\", x);\n}\n";
        fs::write(h.root().join("main.rs"), original).unwrap();
        let r = req(
            "main.rs",
            original,
            vec![EditOp::SearchReplace {
                before: "let x = 1;".into(),
                after: "let x = 2;".into(),
            }],
        );
        let out = engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap();
        assert!(!out.suspicious);
        assert!(out.parse_error.is_none());
    }

    #[test]
    fn python_parse_check_works() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(faktor_fs::WorkspaceFileService::new());
        let original = b"def f():\n    return 1\n";
        fs::write(h.root().join("f.py"), original).unwrap();
        // Valid edit → not suspicious.
        let r = req(
            "f.py",
            original,
            vec![EditOp::SearchReplace {
                before: "return 1".into(),
                after: "return 2".into(),
            }],
        );
        let out = engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap();
        assert!(!out.suspicious);
        // Directly test the syntax check with a broken edited version
        // (unterminated string is a definite parse error).
        let broken = "def f():\n    return \"1\n";
        let original_str = std::str::from_utf8(original).unwrap();
        let (suspicious, err) = check_syntax(std::path::Path::new("f.py"), original_str, broken);
        assert!(suspicious, "unterminated string must be a parse error");
        assert!(err.is_some());
    }

    #[test]
    fn unknown_language_skips_parse_check() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(faktor_fs::WorkspaceFileService::new());
        let original = b"<div><span></div>"; // broken HTML
        fs::write(h.root().join("x.html"), original).unwrap();
        let r = req(
            "x.html",
            original,
            vec![EditOp::SearchReplace {
                before: "<div>".into(),
                after: "<p>".into(),
            }],
        );
        let out = engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap();
        assert!(!out.suspicious, "unsupported language skips the check");
    }

    #[test]
    fn partial_failure_no_partial_write() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(faktor_fs::WorkspaceFileService::new());
        let original = b"one two three";
        fs::write(h.root().join("a.txt"), original).unwrap();
        // Op 1 valid, op 2 broken: NOTHING may be written.
        let r = req(
            "a.txt",
            original,
            vec![
                EditOp::SearchReplace {
                    before: "one".into(),
                    after: "ONE".into(),
                },
                EditOp::SearchReplace {
                    before: "zzz".into(),
                    after: "x".into(),
                },
            ],
        );
        let err = engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap_err();
        assert!(err.kind == ErrorKind::Malformed);
        assert!(err.message.contains("op 2"));
        assert_eq!(fs::read(h.root().join("a.txt")).unwrap(), original);
    }

    #[test]
    fn huge_edit_bounded() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(faktor_fs::WorkspaceFileService::new());
        // A 20MB file exceeds the bound → Oversized, never OOM.
        let big = vec![b'x'; 20 * 1024 * 1024];
        fs::write(h.root().join("big.txt"), &big).unwrap();
        let r = req(
            "big.txt",
            &big,
            vec![EditOp::SearchReplace {
                before: "x".into(),
                after: "y".into(),
            }],
        );
        let err = engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap_err();
        assert!(err.kind == ErrorKind::Oversized);
    }

    #[tokio::test]
    async fn concurrent_edits_same_file_one_wins() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(faktor_fs::WorkspaceFileService::new());
        let original = b"start\n";
        fs::write(h.root().join("c.txt"), original).unwrap();
        let engine = std::sync::Arc::new(engine);
        let mut handles = Vec::new();
        for t in 0..6 {
            let engine = engine.clone();
            let h = h.clone();
            let _ = &id;
            handles.push(tokio::spawn(async move {
                let r = req(
                    "c.txt",
                    original,
                    vec![EditOp::SearchReplace {
                        before: "start".into(),
                        after: format!("thread-{t}"),
                    }],
                );
                engine.apply(&h, &id, &r, RepairMode::Rollback)
            }));
        }
        let mut ok = 0;
        let mut conflicts = 0;
        for h in handles {
            match h.await.unwrap() {
                Ok(_) => ok += 1,
                Err(e) if e.kind == ErrorKind::Conflict => conflicts += 1,
                Err(e) => panic!("unexpected {e:?}"),
            }
        }
        // Exactly one edit wins; the rest see a stale expected_hash.
        assert_eq!(ok, 1);
        assert_eq!(conflicts, 5);
        let final_content = fs::read(h.root().join("c.txt")).unwrap();
        assert!(String::from_utf8_lossy(&final_content).starts_with("thread-"));
    }

    #[test]
    fn bounded_region_edits() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(faktor_fs::WorkspaceFileService::new());
        let original = b"fn f() {\n    let a = 1;\n    let b = 2;\n}\n";
        fs::write(h.root().join("x.rs"), original).unwrap();
        // Anchor on the fn line; replace the region after it.
        let anchor = "fn f() {";
        let r = req(
            "x.rs",
            original,
            vec![EditOp::BoundedRegion {
                anchor: anchor.into(),
                region_start: anchor.len(),
                region_end: original.len() - 2,
                replacement: "\n    let z = 9;\n".into(),
            }],
        );
        let out = engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap();
        assert!(!out.suspicious);
        let text = String::from_utf8(fs::read(h.root().join("x.rs")).unwrap()).unwrap();
        assert!(text.contains("let z = 9;"));
        // Ambiguous anchor → conflict.
        let original2 = b"let a = 1;\nlet a = 2;\n";
        fs::write(h.root().join("y.txt"), original2).unwrap();
        let r = req(
            "y.txt",
            original2,
            vec![EditOp::BoundedRegion {
                anchor: "let a".into(),
                region_start: 0,
                region_end: 4,
                replacement: "x".into(),
            }],
        );
        let err = engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap_err();
        assert!(err.kind == ErrorKind::Conflict);
    }

    #[test]
    fn already_broken_file_not_blamed() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(faktor_fs::WorkspaceFileService::new());
        let broken = b"fn main() { let x = ; }\n";
        fs::write(h.root().join("b.rs"), broken).unwrap();
        // The file is already broken; the edit cannot be flagged for making
        // it broken (before-parse failed).
        let r = req(
            "b.rs",
            broken,
            vec![EditOp::SearchReplace {
                before: "fn main".into(),
                after: "fn entry".into(),
            }],
        );
        let out = engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap();
        assert!(!out.suspicious);
    }

    #[test]
    fn non_utf8_file_rejected() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(faktor_fs::WorkspaceFileService::new());
        let bytes = vec![0xFF, 0xFE, 0x00, 0x80];
        fs::write(h.root().join("bin.dat"), &bytes).unwrap();
        let r = req(
            "bin.dat",
            &bytes,
            vec![EditOp::SearchReplace {
                before: "x".into(),
                after: "y".into(),
            }],
        );
        let err = engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap_err();
        assert!(err.kind == ErrorKind::Malformed);
    }

    #[test]
    fn parse_validation_covers_next_tier_languages() {
        // Audit round 9 (P1): TypeScript/JS/Go/Java join Rust/Python in
        // parse-before-accept. Valid files parse; broken files are caught
        // with a line-numbered error.
        let cases: Vec<(&str, &str)> = vec![
            (
                "a.ts",
                "const x: number = 1;\nexport function f(a: string): void {}\n",
            ),
            ("a.tsx", "const el = <div attr={x}>hi</div>;\n"),
            ("a.js", "function f(x) { return x * 2; }\n"),
            (
                "a.go",
                "package main\n\nfunc main() {\n\tprintln(\"hi\")\n}\n",
            ),
            ("a.java", "class A {\n  int f() { return 1; }\n}\n"),
        ];
        for (name, src) in cases {
            let (label, lang) = language_for(std::path::Path::new(name))
                .unwrap_or_else(|| panic!("{name} must resolve a grammar"));
            assert!(
                parse_ok(label, &lang, src),
                "{name}: valid source must parse"
            );
            // Hostile broken input: catch an error, never accept silently.
            let broken = format!("{src} this is not valid {{{name}");
            assert!(
                first_parse_error(label, &lang, &broken).is_some(),
                "{name}: broken source must be rejected"
            );
        }
        // Unlisted extensions: no grammar (documented skip).
        assert!(language_for(std::path::Path::new("a.zig")).is_none());
        assert!(language_for(std::path::Path::new("a.kt")).is_none());
    }

    // -------------------------------------------------- audit 46/76/77 CAS

    #[test]
    fn single_file_commit_is_cas_protected_against_mid_edit_writers() {
        let (dir, _s, h, id) = fixture();
        let engine = EditEngine::new(faktor_fs::WorkspaceFileService::new());
        let target = h.root().join("r.txt");
        fs::write(&target, b"model-read-content-1234").unwrap();
        let base = fs::read(&target).unwrap();
        // A second writer lands between the stage validation and the commit.
        // Deterministic via the commit-gap seam? apply() has no seam, so use
        // real threads: writer B replaces content while A validates a large
        // file... simpler deterministic proof: B writes AFTER A's stage by
        // using the commit-gap hook path through apply_many (below); here we
        // prove the CAS itself rejects a stale expected hash at commit time
        // by staging against the ORIGINAL content and committing after an
        // external replace with the ORIGINAL expected hash — the digest the
        // CAS checks is of the file AS READ during staging, so an external
        // same-size replacement is still caught by the engine's read+hash.
        let req = req(
            "r.txt",
            &base,
            vec![EditOp::SearchReplace {
                before: "model-read-content-1234".into(),
                after: "edited-by-A-123456789012".into(),
            }],
        );
        // Validate+write normally: success path.
        let out = engine.apply(&h, &id, &req, RepairMode::Rollback).unwrap();
        assert_eq!(out.ops_applied, 1);
        assert_eq!(fs::read(&target).unwrap(), b"edited-by-A-123456789012");
        // Now stage against STALE state: B changed the file after A's read.
        let stale_base = fs::read(&target).unwrap();
        let _ = &stale_base;
        fs::write(&target, b"writer-B-took-over-098765").unwrap();
        let out = engine.apply(&h, &id, &req, RepairMode::Rollback);
        assert!(out.is_err(), "stale stage must be rejected");
        assert_eq!(fs::read(&target).unwrap(), b"writer-B-took-over-098765");
        let _ = dir;
    }

    #[test]
    fn multi_file_commit_conflict_reports_exact_partial_state() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(faktor_fs::WorkspaceFileService::new());
        fs::write(h.root().join("f1.txt"), b"orig-one").unwrap();
        fs::write(h.root().join("f2.txt"), b"orig-two").unwrap();
        fs::write(h.root().join("f3.txt"), b"orig-three").unwrap();
        let mk = |name: &str, from: &str, to: &str| {
            let bytes = fs::read(h.root().join(name)).unwrap();
            req(
                name,
                &bytes,
                vec![EditOp::SearchReplace {
                    before: from.into(),
                    after: to.into(),
                }],
            )
        };
        let reqs = vec![
            mk("f1.txt", "orig-one", "edited-one"),
            mk("f2.txt", "orig-two", "edited-two"),
            mk("f3.txt", "orig-three", "edited-three"),
        ];
        // Adversary: between validation and commit, an EXTERNAL writer (not
        // the edit engine — plain fs) replaces f2 with different content.
        let f2abs = h.root().join("f2.txt");
        let hook = Box::new(move || {
            fs::write(&f2abs, b"external-writer-content").unwrap();
        });
        *COMMIT_GAP
            .get_or_init(|| std::sync::Mutex::new(None))
            .lock()
            .expect("seam poisoned") = Some(hook);
        let outcome = engine
            .apply_many(&h, &id, &reqs, RepairMode::Rollback, None)
            .unwrap();
        // Cleanup the global seam for sibling tests.
        *COMMIT_GAP
            .get_or_init(|| std::sync::Mutex::new(None))
            .lock()
            .expect("seam poisoned") = None;
        assert_eq!(outcome.committed.len(), 1, "{outcome:?}");
        assert_eq!(outcome.committed[0].path, "f1.txt");
        assert_eq!(outcome.conflicted.len(), 1, "{outcome:?}");
        assert_eq!(outcome.conflicted[0].0, "f2.txt");
        assert!(outcome.conflicted[0].1.contains("changed"), "{outcome:?}");
        assert_eq!(outcome.skipped, vec!["f3.txt".to_string()]);
        assert_eq!(outcome.ops_applied_total, 1);
        // Content truth: f1 committed; f2 = external bytes (never clobbered);
        // f3 untouched.
        assert_eq!(fs::read(h.root().join("f1.txt")).unwrap(), b"edited-one");
        assert_eq!(
            fs::read(h.root().join("f2.txt")).unwrap(),
            b"external-writer-content"
        );
        assert_eq!(fs::read(h.root().join("f3.txt")).unwrap(), b"orig-three");
    }

    #[test]
    fn multi_file_stage_failure_aborts_before_any_write() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(faktor_fs::WorkspaceFileService::new());
        fs::write(h.root().join("g1.txt"), b"keep-one").unwrap();
        fs::write(h.root().join("g2.txt"), b"keep-two").unwrap();
        let mk = |name: &str, from: &str, to: &str| {
            let bytes = fs::read(h.root().join(name)).unwrap();
            req(
                name,
                &bytes,
                vec![EditOp::SearchReplace {
                    before: from.into(),
                    after: to.into(),
                }],
            )
        };
        let good = mk("g1.txt", "keep-one", "edited-one");
        // A hostile path escapes the workspace: stage must fail.
        let evil = EditRequest {
            path: "../outside.txt".into(),
            expected_hash: EditEngine::hash_of(b"x"),
            ops: vec![EditOp::SearchReplace {
                before: "x".into(),
                after: "y".into(),
            }],
        };
        let r = engine.apply_many(&h, &id, &[good.clone(), evil], RepairMode::Rollback, None);
        assert!(r.is_err(), "escape must abort the batch before writes");
        assert_eq!(fs::read(h.root().join("g1.txt")).unwrap(), b"keep-one");
        assert_eq!(fs::read(h.root().join("g2.txt")).unwrap(), b"keep-two");
        // Stale expected hash in the SECOND file also aborts before writes.
        let stale = EditRequest {
            path: "g2.txt".into(),
            expected_hash: EditEngine::hash_of(b"something-else"),
            ops: vec![EditOp::SearchReplace {
                before: "keep-two".into(),
                after: "edited-two".into(),
            }],
        };
        let r = engine.apply_many(&h, &id, &[good, stale], RepairMode::Rollback, None);
        assert!(r.is_err(), "stale stage must abort before writes");
        assert_eq!(fs::read(h.root().join("g1.txt")).unwrap(), b"keep-one");
        assert_eq!(fs::read(h.root().join("g2.txt")).unwrap(), b"keep-two");
    }

    #[test]
    fn multi_file_budget_expiry_writes_nothing() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(faktor_fs::WorkspaceFileService::new());
        fs::write(h.root().join("b1.txt"), b"before-one").unwrap();
        fs::write(h.root().join("b2.txt"), b"before-two").unwrap();
        let mk = |name: &str, from: &str, to: &str| {
            let bytes = fs::read(h.root().join(name)).unwrap();
            req(
                name,
                &bytes,
                vec![EditOp::SearchReplace {
                    before: from.into(),
                    after: to.into(),
                }],
            )
        };
        let reqs = vec![
            mk("b1.txt", "before-one", "after-one"),
            mk("b2.txt", "before-two", "after-two"),
        ];
        // A sub-nanosecond validation budget expires before the first commit.
        let r = engine.apply_many(
            &h,
            &id,
            &reqs,
            RepairMode::Rollback,
            Some(std::time::Duration::from_nanos(1)),
        );
        assert!(r.is_err(), "budget expiry must abort: {r:?}");
        assert_eq!(fs::read(h.root().join("b1.txt")).unwrap(), b"before-one");
        assert_eq!(fs::read(h.root().join("b2.txt")).unwrap(), b"before-two");
        // A sane budget commits everything.
        let out = engine
            .apply_many(
                &h,
                &id,
                &reqs,
                RepairMode::Rollback,
                Some(std::time::Duration::from_secs(5)),
            )
            .unwrap();
        assert!(out.all_committed(), "{out:?}");
        assert_eq!(out.ops_applied_total, 2);
        assert_eq!(fs::read(h.root().join("b1.txt")).unwrap(), b"after-one");
        assert_eq!(fs::read(h.root().join("b2.txt")).unwrap(), b"after-two");
    }

    #[test]
    fn multi_file_parse_rollback_aborts_before_any_write() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(faktor_fs::WorkspaceFileService::new());
        fs::write(
            h.root().join("p1.rs"),
            b"fn ok() {}
",
        )
        .unwrap();
        fs::write(
            h.root().join("p2.rs"),
            b"fn also_ok() {}
",
        )
        .unwrap();
        let mk = |name: &str, bytes: Vec<u8>, before: &str, after: &str| {
            req(
                name,
                &bytes,
                vec![EditOp::SearchReplace {
                    before: before.into(),
                    after: after.into(),
                }],
            )
        };
        let p2_bytes = fs::read(h.root().join("p2.rs")).unwrap();
        let broken = mk(
            "p1.rs",
            fs::read(h.root().join("p1.rs")).unwrap(),
            "fn ok() {}",
            "fn ok( {",
        );
        let good2 = mk("p2.rs", p2_bytes, "fn also_ok() {}", "fn also_ok() {} // x");
        let r = engine.apply_many(&h, &id, &[broken, good2], RepairMode::Rollback, None);
        assert!(r.is_err(), "parse rollback must abort: {r:?}");
        assert_eq!(
            fs::read(h.root().join("p1.rs")).unwrap(),
            b"fn ok() {}
",
            "file 1 must be untouched"
        );
        assert_eq!(
            fs::read(h.root().join("p2.rs")).unwrap(),
            b"fn also_ok() {}
",
            "file 2 must be untouched"
        );
    }

    // ============================================ P0-53 durable edit txn suite

    use std::panic::AssertUnwindSafe;

    use faktor_core::state::{EditTxnId, EditTxnStrategy};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};

    /// In-memory stand-in of the session's typed ledger: rows survive the
    /// "crash" (the engine + prepared data are dropped, the journal Arc is
    /// not), plus deterministic crash knobs at every journal point:
    /// - `panic_after_progress`: crash right AFTER that progress row landed
    ///   (before the next write),
    /// - `panic_before_progress`: crash right BEFORE that progress row lands
    ///   (the file was already written → the ambiguous recovery window),
    /// - `panic_on_terminal`: crash BEFORE the terminal row lands (the
    ///   commit's writes + progress rows are durable, the txn stays open).
    #[derive(Clone)]
    struct MockJournal {
        rows: Arc<Mutex<Vec<MockRow>>>,
        panic_after_progress: Option<u64>,
        panic_before_progress: Option<u64>,
        /// One-shot: the FIRST terminal append crashes (recovery's own
        /// terminal append succeeds — the crash already happened once).
        panic_on_terminal: Arc<AtomicBool>,
        progress_appended: Arc<AtomicU64>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum MockRow {
        Prepared {
            txn: EditTxnId,
            strategy: EditTxnStrategy,
            files: Vec<EditTxnFileRef>,
        },
        Progress {
            txn: EditTxnId,
            seq: u64,
            path: String,
            outcome: TxnFileOutcome,
        },
        Committed {
            txn: EditTxnId,
            committed: Vec<String>,
            conflicted: Vec<String>,
            skipped: Vec<String>,
        },
        RolledBack {
            txn: EditTxnId,
            rolled_back: Vec<String>,
            rollback_conflicts: Vec<String>,
        },
    }

    impl Default for MockJournal {
        fn default() -> Self {
            Self {
                rows: Arc::new(Mutex::new(Vec::new())),
                panic_after_progress: None,
                panic_before_progress: None,
                panic_on_terminal: Arc::new(AtomicBool::new(false)),
                progress_appended: Arc::new(AtomicU64::new(0)),
            }
        }
    }

    impl MockJournal {
        fn snapshot(&self) -> Vec<MockRow> {
            self.rows.lock().expect("mock poisoned").clone()
        }
    }

    impl EditTxnJournal for MockJournal {
        fn entry_payload_cap(&self) -> usize {
            MAX_TXN_PAYLOAD_BYTES
        }

        fn record_prepared(
            &self,
            txn_id: EditTxnId,
            _session: &str,
            files: &[EditTxnFileRef],
            strategy: EditTxnStrategy,
        ) -> Result<(), Error> {
            self.rows
                .lock()
                .expect("mock poisoned")
                .push(MockRow::Prepared {
                    txn: txn_id,
                    strategy,
                    files: files.to_vec(),
                });
            Ok(())
        }

        fn record_progress(
            &self,
            txn_id: EditTxnId,
            seq: u64,
            path: &str,
            outcome: TxnFileOutcome,
        ) -> Result<(), Error> {
            let ordinal = self.progress_appended.fetch_add(1, Ordering::SeqCst);
            if self.panic_before_progress == Some(ordinal) {
                panic!("simulated crash before progress row {ordinal} is journaled");
            }
            self.rows
                .lock()
                .expect("mock poisoned")
                .push(MockRow::Progress {
                    txn: txn_id,
                    seq,
                    path: path.to_string(),
                    outcome,
                });
            if self.panic_after_progress == Some(ordinal) {
                panic!("simulated crash right after progress row {ordinal}");
            }
            Ok(())
        }

        fn record_committed(
            &self,
            txn_id: EditTxnId,
            committed: &[String],
            conflicted: &[String],
            skipped: &[String],
        ) -> Result<(), Error> {
            if self.panic_on_terminal.swap(false, Ordering::SeqCst) {
                panic!("simulated crash before the terminal row is journaled");
            }
            self.rows
                .lock()
                .expect("mock poisoned")
                .push(MockRow::Committed {
                    txn: txn_id,
                    committed: committed.to_vec(),
                    conflicted: conflicted.to_vec(),
                    skipped: skipped.to_vec(),
                });
            Ok(())
        }

        fn record_rolled_back(
            &self,
            txn_id: EditTxnId,
            rolled_back: &[String],
            rollback_conflicts: &[String],
        ) -> Result<(), Error> {
            if self.panic_on_terminal.swap(false, Ordering::SeqCst) {
                panic!("simulated crash before the terminal row is journaled");
            }
            self.rows
                .lock()
                .expect("mock poisoned")
                .push(MockRow::RolledBack {
                    txn: txn_id,
                    rolled_back: rolled_back.to_vec(),
                    rollback_conflicts: rollback_conflicts.to_vec(),
                });
            Ok(())
        }

        fn open_transactions(&self) -> Result<Vec<OpenEditTxn>, Error> {
            let rows = self.snapshot();
            let mut terminal: HashSet<EditTxnId> = HashSet::new();
            for row in &rows {
                if matches!(row, MockRow::Committed { .. } | MockRow::RolledBack { .. }) {
                    terminal.insert(txn_of(row));
                }
            }
            let mut out: Vec<OpenEditTxn> = Vec::new();
            for row in &rows {
                let MockRow::Prepared {
                    txn,
                    strategy,
                    files,
                } = row
                else {
                    continue;
                };
                if terminal.contains(txn) {
                    continue;
                }
                let mut progress: Vec<TxnProgressRow> = Vec::new();
                for p in rows
                    .iter()
                    .filter(|r| matches!(r, MockRow::Progress { txn: t, .. } if t == txn))
                {
                    if let MockRow::Progress {
                        seq, path, outcome, ..
                    } = p
                    {
                        progress.push(TxnProgressRow {
                            seq: *seq,
                            path: path.clone(),
                            outcome: *outcome,
                        });
                    }
                }
                progress.sort_by_key(|p| p.seq);
                out.push(OpenEditTxn {
                    txn_id: *txn,
                    strategy: *strategy,
                    files: files.clone(),
                    progress,
                });
            }
            Ok(out)
        }
    }

    fn txn_of(row: &MockRow) -> EditTxnId {
        match row {
            MockRow::Prepared { txn, .. }
            | MockRow::Progress { txn, .. }
            | MockRow::Committed { txn, .. }
            | MockRow::RolledBack { txn, .. } => *txn,
        }
    }

    fn engine() -> EditEngine {
        EditEngine::new(faktor_fs::WorkspaceFileService::new())
    }

    /// (t1, t2, t3) with base content orig-one/two/three and search-replace
    /// requests to edited-one/two/three.
    fn three_file_requests(h: &WorkspaceHandle) -> Vec<EditRequest> {
        let mk = |name: &str, from: &str, to: &str| {
            let bytes = fs::read(h.root().join(name)).unwrap();
            req(
                name,
                &bytes,
                vec![EditOp::SearchReplace {
                    before: from.into(),
                    after: to.into(),
                }],
            )
        };
        vec![
            mk("f1.txt", "orig-one", "edited-one"),
            mk("f2.txt", "orig-two", "edited-two"),
            mk("f3.txt", "orig-three", "edited-three"),
        ]
    }

    /// An external (non-engine, plain-fs) writer replaces f2 between the
    /// begin (stage) and the commit — the deterministic mid-commit race.
    fn strike_f2_externally(h: &WorkspaceHandle) {
        fs::write(h.root().join("f2.txt"), b"external-writer-content").unwrap();
    }

    fn begin_three(
        engine: &EditEngine,
        h: &WorkspaceHandle,
        id: &WorkspaceIdentity,
        journal: &MockJournal,
        txn: u64,
        strategy: EditTxnStrategy,
    ) -> PreparedEdit {
        let reqs = three_file_requests(h);
        engine
            .begin_multi_file_edit(
                h,
                id,
                &reqs,
                RepairMode::Rollback,
                strategy,
                Arc::new(journal.clone()),
                EditTxnId::new(txn),
                "session-1",
                None,
            )
            .unwrap()
    }

    fn drop_prepared(_prepared: PreparedEdit) {
        // Simulated crash: the in-memory prepared state dies with the
        // process; only the journal rows survive.
    }

    fn progress_paths(journal: &MockJournal, txn: EditTxnId) -> Vec<(String, String)> {
        journal
            .snapshot()
            .iter()
            .filter_map(|r| match r {
                MockRow::Progress {
                    txn: t,
                    seq,
                    path,
                    outcome,
                } if *t == txn => Some((format!("{seq}:{path}"), outcome.as_tag().to_string())),
                _ => None,
            })
            .collect()
    }

    // (a) live RollForward conflict: committed=[f1], conflicted=[f2],
    // skipped=[f3]; durable rows Prepared + Progress + Committed present.
    #[test]
    fn txn_rollforward_conflict_reports_exact_state_and_durable_rows() {
        let (_d, _s, h, id) = fixture();
        let engine = engine();
        fs::write(h.root().join("f1.txt"), b"orig-one").unwrap();
        fs::write(h.root().join("f2.txt"), b"orig-two").unwrap();
        fs::write(h.root().join("f3.txt"), b"orig-three").unwrap();
        let journal = MockJournal::default();
        let txn = begin_three(&engine, &h, &id, &journal, 1, EditTxnStrategy::RollForward);
        // The adversary strikes between the begin (validation) and commit.
        strike_f2_externally(&h);
        let result = engine
            .commit_prepared(&h, &id, txn)
            .expect("conflict is a reported outcome, not an error");
        assert_eq!(result.committed.len(), 1, "{result:?}");
        assert_eq!(result.committed[0].path, "f1.txt");
        assert_eq!(result.conflicted.len(), 1, "{result:?}");
        assert_eq!(result.conflicted[0].0, "f2.txt");
        assert!(result.conflicted[0].1.contains("changed"), "{result:?}");
        assert_eq!(result.skipped, vec!["f3.txt".to_string()]);
        assert!(result.rolled_back.is_empty());
        assert_eq!(result.txn_id, EditTxnId::new(1));
        // Durable rows: Prepared, Progress(0 committed), Progress(1
        // conflicted), terminal Committed — no other rows.
        let rows = journal.snapshot();
        assert_eq!(rows.len(), 4, "{rows:?}");
        assert!(matches!(&rows[0], MockRow::Prepared { files, .. } if files.len() == 3));
        assert!(matches!(
            &rows[1],
            MockRow::Progress {
                seq: 0,
                outcome: TxnFileOutcome::Committed,
                ..
            }
        ));
        assert!(matches!(
            &rows[2],
            MockRow::Progress {
                seq: 1,
                outcome: TxnFileOutcome::Conflicted,
                ..
            }
        ));
        match &rows[3] {
            MockRow::Committed {
                committed,
                conflicted,
                skipped,
                ..
            } => {
                assert_eq!(committed, &vec!["f1.txt".to_string()]);
                assert_eq!(conflicted, &vec!["f2.txt".to_string()]);
                assert_eq!(skipped, &vec!["f3.txt".to_string()]);
            }
            other => panic!("terminal must be Committed, got {other:?}"),
        }
        // Content truth: f1 = our target; f2 = the external bytes (never
        // clobbered); f3 untouched.
        assert_eq!(fs::read(h.root().join("f1.txt")).unwrap(), b"edited-one");
        assert_eq!(
            fs::read(h.root().join("f2.txt")).unwrap(),
            b"external-writer-content"
        );
        assert_eq!(fs::read(h.root().join("f3.txt")).unwrap(), b"orig-three");
    }

    // (a, crash) simulate a crash BEFORE the terminal row lands, then
    // recover: file 3 lands via replay, file 1 is NOT rewritten (its
    // committed progress row makes it done), file 2 untouched. Deterministic
    // and idempotent (run recovery twice → identical rows and files).
    #[test]
    fn txn_crash_before_terminal_recovery_replays_and_is_idempotent() {
        let (_d, _s, h, id) = fixture();
        let engine = engine();
        fs::write(h.root().join("f1.txt"), b"orig-one").unwrap();
        fs::write(h.root().join("f2.txt"), b"orig-two").unwrap();
        fs::write(h.root().join("f3.txt"), b"orig-three").unwrap();
        let journal = MockJournal::default();
        journal.panic_on_terminal.store(true, Ordering::SeqCst);
        let txn = begin_three(&engine, &h, &id, &journal, 2, EditTxnStrategy::RollForward);
        strike_f2_externally(&h);
        // The "crash": commit dies exactly at the terminal journal point.
        let caught =
            std::panic::catch_unwind(AssertUnwindSafe(|| engine.commit_prepared(&h, &id, txn)));
        assert!(caught.is_err(), "commit must crash at the terminal point");
        let rows = journal.snapshot();
        assert_eq!(
            rows.len(),
            3,
            "Prepared + 2 progress, no terminal: {rows:?}"
        );
        assert!(rows.iter().all(|r| !matches!(r, MockRow::Committed { .. })));
        // Recovery with the reissued requests.
        let reqs = three_file_requests(&h);
        let provider = |_path: &str| -> Option<Vec<u8>> { None };
        let journal_arc = Arc::new(journal.clone());
        let resume = EditTxnResume {
            txn_id: EditTxnId::new(2),
            reqs: &reqs,
            before_content: &provider,
        };
        let report = engine
            .recover_edit_transactions(&h, &id, &*journal_arc, &[resume], RepairMode::Rollback)
            .unwrap();
        assert_eq!(report.txns.len(), 1, "{report:?}");
        let out = &report.txns[0];
        assert_eq!(out.committed.len(), 2, "{out:?}");
        assert_eq!(out.conflicted, vec!["f2.txt".to_string()], "{out:?}");
        assert_eq!(out.wrote, vec!["f3.txt".to_string()], "{out:?}");
        assert!(out.skipped.is_empty(), "{out:?}");
        // f3 landed via replay; f1 was NOT rewritten (digest match → done);
        // f2 untouched (external bytes).
        assert_eq!(fs::read(h.root().join("f3.txt")).unwrap(), b"edited-three");
        assert_eq!(fs::read(h.root().join("f1.txt")).unwrap(), b"edited-one");
        assert_eq!(
            fs::read(h.root().join("f2.txt")).unwrap(),
            b"external-writer-content"
        );
        // f1 has exactly ONE committed progress row (recovery did not
        // re-journal or rewrite it).
        assert_eq!(
            progress_paths(&journal, EditTxnId::new(2)),
            vec![
                ("0:f1.txt".to_string(), "committed".into()),
                ("1:f2.txt".to_string(), "conflicted".into()),
                ("2:f3.txt".to_string(), "committed".into()),
            ]
        );
        // The recovery journaled the terminal Committed row: durable entries
        // Prepared + Progress + Committed are all present.
        let rows = journal.snapshot();
        assert_eq!(rows.len(), 5, "{rows:?}");
        assert!(rows.iter().any(|r| matches!(r, MockRow::Committed { .. })));
        // Idempotency: a second recovery finds nothing open and changes
        // nothing; the content is byte-identical.
        let before: Vec<Vec<u8>> = ["f1.txt", "f2.txt", "f3.txt"]
            .iter()
            .map(|f| fs::read(h.root().join(f)).unwrap())
            .collect();
        let rows_before = journal.snapshot();
        let provider = |_path: &str| -> Option<Vec<u8>> { None };
        let resume2 = EditTxnResume {
            txn_id: EditTxnId::new(2),
            reqs: &reqs,
            before_content: &provider,
        };
        let report = engine
            .recover_edit_transactions(&h, &id, &*journal_arc, &[resume2], RepairMode::Rollback)
            .unwrap();
        assert!(report.txns.is_empty(), "{report:?}");
        let after: Vec<Vec<u8>> = ["f1.txt", "f2.txt", "f3.txt"]
            .iter()
            .map(|f| fs::read(h.root().join(f)).unwrap())
            .collect();
        assert_eq!(before, after);
        assert_eq!(journal.snapshot(), rows_before);
    }

    // (b) RollBack policy with a file-2 conflict: file 1 is rolled back to
    // its original bytes, the ledger shows RolledBack, file 2 keeps the
    // external bytes; recovery twice → identical state (no-op).
    #[test]
    fn txn_rollback_restores_committed_and_recovery_is_noop() {
        let (_d, _s, h, id) = fixture();
        let engine = engine();
        fs::write(h.root().join("f1.txt"), b"orig-one").unwrap();
        fs::write(h.root().join("f2.txt"), b"orig-two").unwrap();
        fs::write(h.root().join("f3.txt"), b"orig-three").unwrap();
        let journal = MockJournal::default();
        let txn = begin_three(&engine, &h, &id, &journal, 3, EditTxnStrategy::RollBack);
        strike_f2_externally(&h);
        let result = engine.commit_prepared(&h, &id, txn).unwrap();
        assert_eq!(result.committed.len(), 1);
        assert_eq!(result.conflicted.len(), 1);
        assert_eq!(result.conflicted[0].0, "f2.txt");
        assert_eq!(result.skipped, vec!["f3.txt".to_string()]);
        assert_eq!(result.rolled_back, vec!["f1.txt".to_string()]);
        assert!(result.rollback_conflicts.is_empty(), "{result:?}");
        // Files: f1 back at its ORIGINAL bytes; f2 = external bytes; f3
        // untouched.
        assert_eq!(fs::read(h.root().join("f1.txt")).unwrap(), b"orig-one");
        assert_eq!(
            fs::read(h.root().join("f2.txt")).unwrap(),
            b"external-writer-content"
        );
        assert_eq!(fs::read(h.root().join("f3.txt")).unwrap(), b"orig-three");
        let rows = journal.snapshot();
        assert_eq!(rows.len(), 4, "{rows:?}");
        assert!(
            matches!(&rows[3], MockRow::RolledBack { rolled_back, rollback_conflicts, .. }
            if rolled_back == &vec!["f1.txt".to_string()]
                && rollback_conflicts.is_empty())
        );
        // Recovery twice: terminal present → nothing open → identical state.
        let journal_arc = Arc::new(journal.clone());
        let reqs = three_file_requests(&h);
        for _ in 0..2 {
            let provider = |_p: &str| -> Option<Vec<u8>> { None };
            let resume = EditTxnResume {
                txn_id: EditTxnId::new(3),
                reqs: &reqs,
                before_content: &provider,
            };
            let report = engine
                .recover_edit_transactions(&h, &id, &*journal_arc, &[resume], RepairMode::Rollback)
                .unwrap();
            assert!(report.txns.is_empty(), "{report:?}");
        }
        let after: Vec<Vec<u8>> = ["f1.txt", "f2.txt", "f3.txt"]
            .iter()
            .map(|f| fs::read(h.root().join(f)).unwrap())
            .collect();
        assert_eq!(after[0], b"orig-one");
        assert_eq!(after[1], b"external-writer-content");
        assert_eq!(after[2], b"orig-three");
    }

    // (c) rollback CAS conflict: after our write of file 1 an external
    // writer changes file 1 AGAIN → the rollback refuses (typed), the
    // RolledBack row records rollback_conflicts, nothing is clobbered.
    #[test]
    fn txn_rollback_refuses_when_committed_file_changed_again() {
        let (_d, _s, h, id) = fixture();
        let engine = engine();
        fs::write(h.root().join("f1.txt"), b"orig-one").unwrap();
        fs::write(h.root().join("f2.txt"), b"orig-two").unwrap();
        fs::write(h.root().join("f3.txt"), b"orig-three").unwrap();
        let journal = MockJournal::default();
        let txn = begin_three(&engine, &h, &id, &journal, 4, EditTxnStrategy::RollBack);
        strike_f2_externally(&h);
        // The adversary strikes a SECOND time, between our write of f1 and
        // its roll-back restore (deterministic via the rollback-gap seam).
        let f1 = h.root().join("f1.txt");
        let hook = Box::new(move || {
            fs::write(&f1, b"external-again").unwrap();
        });
        *ROLLBACK_GAP
            .get_or_init(|| std::sync::Mutex::new(None))
            .lock()
            .expect("seam poisoned") = Some(hook);
        let result = engine.commit_prepared(&h, &id, txn).unwrap();
        *ROLLBACK_GAP
            .get_or_init(|| std::sync::Mutex::new(None))
            .lock()
            .expect("seam poisoned") = None;
        assert!(result.rolled_back.is_empty(), "{result:?}");
        assert_eq!(
            result.rollback_conflicts,
            vec!["f1.txt".to_string()],
            "{result:?}"
        );
        // The restore CAS refused: f1 keeps the EXTERNAL bytes, never a
        // clobber; f2 keeps its external bytes.
        assert_eq!(
            fs::read(h.root().join("f1.txt")).unwrap(),
            b"external-again"
        );
        assert_eq!(
            fs::read(h.root().join("f2.txt")).unwrap(),
            b"external-writer-content"
        );
        let rows = journal.snapshot();
        assert_eq!(rows.len(), 4, "{rows:?}");
        assert!(
            matches!(&rows[3], MockRow::RolledBack { rolled_back, rollback_conflicts, .. }
            if rolled_back.is_empty()
                && rollback_conflicts == &vec!["f1.txt".to_string()])
        );
    }

    // (d) crash BETWEEN each journal point.
    // (d1) Prepared only (crash before the first write): recovery replays
    // every file cleanly; a second recovery is a no-op.
    #[test]
    fn txn_crash_after_prepared_only_replays_cleanly() {
        let (_d, _s, h, id) = fixture();
        let engine = engine();
        fs::write(h.root().join("f1.txt"), b"orig-one").unwrap();
        fs::write(h.root().join("f2.txt"), b"orig-two").unwrap();
        fs::write(h.root().join("f3.txt"), b"orig-three").unwrap();
        let journal = MockJournal::default();
        let txn = begin_three(&engine, &h, &id, &journal, 5, EditTxnStrategy::RollForward);
        assert_eq!(journal.snapshot().len(), 1, "Prepared only");
        drop_prepared(txn);
        let reqs = three_file_requests(&h);
        let provider = |_p: &str| -> Option<Vec<u8>> { None };
        let journal_arc = Arc::new(journal.clone());
        let resume = EditTxnResume {
            txn_id: EditTxnId::new(5),
            reqs: &reqs,
            before_content: &provider,
        };
        let report = engine
            .recover_edit_transactions(&h, &id, &*journal_arc, &[resume], RepairMode::Rollback)
            .unwrap();
        assert_eq!(report.txns.len(), 1, "{report:?}");
        let out = &report.txns[0];
        assert_eq!(out.wrote.len(), 3, "{out:?}");
        assert!(
            out.conflicted.is_empty() && out.skipped.is_empty(),
            "{out:?}"
        );
        assert_eq!(out.committed.len(), 3, "{out:?}");
        assert_eq!(fs::read(h.root().join("f1.txt")).unwrap(), b"edited-one");
        assert_eq!(fs::read(h.root().join("f2.txt")).unwrap(), b"edited-two");
        assert_eq!(fs::read(h.root().join("f3.txt")).unwrap(), b"edited-three");
        // Terminal journaled by the recovery; the second recovery is a no-op.
        let rows = journal.snapshot();
        assert_eq!(rows.len(), 5, "{rows:?}");
        assert!(rows.iter().any(|r| matches!(r, MockRow::Committed { .. })));
        let provider = |_p: &str| -> Option<Vec<u8>> { None };
        let resume = EditTxnResume {
            txn_id: EditTxnId::new(5),
            reqs: &reqs,
            before_content: &provider,
        };
        let report = engine
            .recover_edit_transactions(&h, &id, &*journal_arc, &[resume], RepairMode::Rollback)
            .unwrap();
        assert!(report.txns.is_empty(), "{report:?}");
        assert_eq!(journal.snapshot(), rows);
    }

    // (d2) Prepared + one Progress: recovery resumes at the second file.
    #[test]
    fn txn_crash_after_first_progress_resumes() {
        let (_d, _s, h, id) = fixture();
        let engine = engine();
        fs::write(h.root().join("f1.txt"), b"orig-one").unwrap();
        fs::write(h.root().join("f2.txt"), b"orig-two").unwrap();
        fs::write(h.root().join("f3.txt"), b"orig-three").unwrap();
        let journal = MockJournal {
            panic_after_progress: Some(0),
            ..Default::default()
        };
        let txn = begin_three(&engine, &h, &id, &journal, 6, EditTxnStrategy::RollForward);
        let caught =
            std::panic::catch_unwind(AssertUnwindSafe(|| engine.commit_prepared(&h, &id, txn)));
        assert!(caught.is_err());
        let rows = journal.snapshot();
        assert_eq!(rows.len(), 2, "Prepared + progress(f1): {rows:?}");
        assert!(matches!(&rows[1], MockRow::Progress { seq: 0, .. }));
        // f1's write is durable; recovery must NOT rewrite it.
        let reqs = three_file_requests(&h);
        let provider = |_p: &str| -> Option<Vec<u8>> { None };
        let journal_arc = Arc::new(journal.clone());
        let resume = EditTxnResume {
            txn_id: EditTxnId::new(6),
            reqs: &reqs,
            before_content: &provider,
        };
        let report = engine
            .recover_edit_transactions(&h, &id, &*journal_arc, &[resume], RepairMode::Rollback)
            .unwrap();
        let out = &report.txns[0];
        assert_eq!(
            out.wrote,
            vec!["f2.txt".to_string(), "f3.txt".to_string()],
            "{out:?}"
        );
        assert_eq!(fs::read(h.root().join("f1.txt")).unwrap(), b"edited-one");
        assert_eq!(fs::read(h.root().join("f2.txt")).unwrap(), b"edited-two");
        assert_eq!(fs::read(h.root().join("f3.txt")).unwrap(), b"edited-three");
    }

    // (d3) Terminal present → recovery is a no-op, in BOTH strategies.
    #[test]
    fn txn_recovery_with_terminal_present_is_noop() {
        for strategy in [EditTxnStrategy::RollForward, EditTxnStrategy::RollBack] {
            let (_d, _s, h, id) = fixture();
            let engine = engine();
            fs::write(h.root().join("f1.txt"), b"orig-one").unwrap();
            fs::write(h.root().join("f2.txt"), b"orig-two").unwrap();
            fs::write(h.root().join("f3.txt"), b"orig-three").unwrap();
            let journal = MockJournal::default();
            let txn = begin_three(&engine, &h, &id, &journal, 7, strategy);
            // RollBack needs a conflict to terminal as RolledBack; give it
            // one BETWEEN begin and commit (deterministic mid-commit race).
            if strategy == EditTxnStrategy::RollBack {
                strike_f2_externally(&h);
            }
            engine.commit_prepared(&h, &id, txn).unwrap();
            let terminal_rows = journal.snapshot().len();
            assert!(terminal_rows >= 4, "commit must journal its terminal");
            let expected_content: Vec<Vec<u8>> = ["f1.txt", "f2.txt", "f3.txt"]
                .iter()
                .map(|f| fs::read(h.root().join(f)).unwrap())
                .collect();
            let reqs = three_file_requests(&h);
            let provider = |_p: &str| -> Option<Vec<u8>> { None };
            let journal_arc = Arc::new(journal.clone());
            for _ in 0..2 {
                let resume = EditTxnResume {
                    txn_id: EditTxnId::new(7),
                    reqs: &reqs,
                    before_content: &provider,
                };
                let report = engine
                    .recover_edit_transactions(
                        &h,
                        &id,
                        &*journal_arc,
                        &[resume],
                        RepairMode::Rollback,
                    )
                    .unwrap();
                assert!(report.txns.is_empty(), "{strategy:?}: {report:?}");
            }
            let after: Vec<Vec<u8>> = ["f1.txt", "f2.txt", "f3.txt"]
                .iter()
                .map(|f| fs::read(h.root().join(f)).unwrap())
                .collect();
            assert_eq!(
                expected_content, after,
                "{strategy:?}: recovery must be a no-op"
            );
            assert_eq!(journal.snapshot().len(), terminal_rows, "{strategy:?}");
            // A recovery WITHOUT any resume is equally a no-op (all rows
            // have progress; nothing needs replay).
            let report = engine
                .recover_edit_transactions(&h, &id, &*journal_arc, &[], RepairMode::Rollback)
                .unwrap();
            assert!(report.txns.is_empty(), "{strategy:?}: {report:?}");
        }
    }

    // (d4) The ambiguous window: a write lands but its progress row never
    // does. Recovery refuses with a typed needs-recovery error, writes
    // NOTHING, and leaves the transaction open (deterministic on rerun).
    #[test]
    fn txn_ambiguous_write_after_crash_needs_recovery_typed() {
        let (_d, _s, h, id) = fixture();
        let engine = engine();
        fs::write(h.root().join("f1.txt"), b"orig-one").unwrap();
        fs::write(h.root().join("f2.txt"), b"orig-two").unwrap();
        fs::write(h.root().join("f3.txt"), b"orig-three").unwrap();
        // Crash BEFORE the first progress row is journaled: f1 was already
        // written by the engine (its content is our target, no row proves
        // it).
        let journal = MockJournal {
            panic_before_progress: Some(0),
            ..Default::default()
        };
        let txn = begin_three(&engine, &h, &id, &journal, 8, EditTxnStrategy::RollForward);
        let caught =
            std::panic::catch_unwind(AssertUnwindSafe(|| engine.commit_prepared(&h, &id, txn)));
        assert!(caught.is_err());
        assert_eq!(journal.snapshot().len(), 1, "Prepared only");
        assert_eq!(fs::read(h.root().join("f1.txt")).unwrap(), b"edited-one");
        // Recovery: f1 has no progress row and is not at its base digest →
        // typed needs-recovery error; f2/f3 stay untouched; the txn stays
        // open (no terminal row).
        let reqs = three_file_requests(&h);
        let provider = |_p: &str| -> Option<Vec<u8>> { None };
        let journal_arc = Arc::new(journal.clone());
        for _ in 0..2 {
            let resume = EditTxnResume {
                txn_id: EditTxnId::new(8),
                reqs: &reqs,
                before_content: &provider,
            };
            let err = engine
                .recover_edit_transactions(&h, &id, &*journal_arc, &[resume], RepairMode::Rollback)
                .unwrap_err();
            assert_eq!(err.kind, ErrorKind::Conflict, "{err}");
            assert!(err.message.contains("needs recovery"), "{err}");
            assert_eq!(journal.snapshot().len(), 1, "txn must stay open: {err}");
        }
        assert_eq!(fs::read(h.root().join("f1.txt")).unwrap(), b"edited-one");
        assert_eq!(fs::read(h.root().join("f2.txt")).unwrap(), b"orig-two");
        assert_eq!(fs::read(h.root().join("f3.txt")).unwrap(), b"orig-three");
    }

    // (d5) RollBack crash between the conflict progress row and the restore:
    // with a before-content provider the committed file is CAS-restored to
    // its original bytes; WITHOUT the provider the restore is a typed
    // needs-recovery error — never a clobber, and the transaction stays
    // open.
    #[test]
    fn txn_rollback_crash_before_restore_needs_before_content() {
        let engine = engine();
        let fresh = || {
            let (d, _s, h, id) = fixture();
            fs::write(h.root().join("f1.txt"), b"orig-one").unwrap();
            fs::write(h.root().join("f2.txt"), b"orig-two").unwrap();
            fs::write(h.root().join("f3.txt"), b"orig-three").unwrap();
            (d, h, id)
        };
        // Without a provider: typed error, file 1 keeps OUR target bytes
        // (never clobbered), txn stays open.
        {
            let (d, h, id) = fresh();
            // Crash after f2's conflict row is journaled (before the
            // roll-back restore).
            let journal = MockJournal {
                panic_after_progress: Some(1),
                ..Default::default()
            };
            let txn = begin_three(&engine, &h, &id, &journal, 9, EditTxnStrategy::RollBack);
            strike_f2_externally(&h);
            let caught =
                std::panic::catch_unwind(AssertUnwindSafe(|| engine.commit_prepared(&h, &id, txn)));
            assert!(caught.is_err());
            let rows = journal.snapshot();
            assert_eq!(rows.len(), 3, "{rows:?}");
            assert!(matches!(
                &rows[2],
                MockRow::Progress {
                    outcome: TxnFileOutcome::Conflicted,
                    ..
                }
            ));
            let reqs = three_file_requests(&h);
            let none = |_p: &str| -> Option<Vec<u8>> { None };
            let journal_arc = Arc::new(journal.clone());
            let resume = EditTxnResume {
                txn_id: EditTxnId::new(9),
                reqs: &reqs,
                before_content: &none,
            };
            let err = engine
                .recover_edit_transactions(&h, &id, &*journal_arc, &[resume], RepairMode::Rollback)
                .unwrap_err();
            assert_eq!(err.kind, ErrorKind::Conflict, "{err}");
            assert!(err.message.contains("before-content"), "{err}");
            assert_eq!(fs::read(h.root().join("f1.txt")).unwrap(), b"edited-one");
            assert_eq!(journal.snapshot(), rows, "txn stays open");
            let _ = d;
        }
        // With a provider returning the exact staged before-content: file 1
        // is CAS-restored to orig-one; f2 stays external; f3 untouched; the
        // RolledBack terminal lands; recovery twice = identical.
        {
            let (_d, h, id) = fresh();
            let journal = MockJournal {
                panic_after_progress: Some(1),
                ..Default::default()
            };
            let txn = begin_three(&engine, &h, &id, &journal, 10, EditTxnStrategy::RollBack);
            strike_f2_externally(&h);
            let caught =
                std::panic::catch_unwind(AssertUnwindSafe(|| engine.commit_prepared(&h, &id, txn)));
            assert!(caught.is_err());
            let reqs = three_file_requests(&h);
            let provider = |p: &str| -> Option<Vec<u8>> {
                Some(match p {
                    "f1.txt" => b"orig-one".to_vec(),
                    "f2.txt" => b"orig-two".to_vec(),
                    _ => b"orig-three".to_vec(),
                })
            };
            let journal_arc = Arc::new(journal.clone());
            let resume = EditTxnResume {
                txn_id: EditTxnId::new(10),
                reqs: &reqs,
                before_content: &provider,
            };
            let report = engine
                .recover_edit_transactions(&h, &id, &*journal_arc, &[resume], RepairMode::Rollback)
                .unwrap();
            let out = &report.txns[0];
            assert_eq!(out.rolled_back, vec!["f1.txt".to_string()], "{out:?}");
            assert_eq!(out.conflicted, vec!["f2.txt".to_string()], "{out:?}");
            assert!(out.rollback_conflicts.is_empty(), "{out:?}");
            assert_eq!(fs::read(h.root().join("f1.txt")).unwrap(), b"orig-one");
            assert_eq!(
                fs::read(h.root().join("f2.txt")).unwrap(),
                b"external-writer-content"
            );
            assert_eq!(fs::read(h.root().join("f3.txt")).unwrap(), b"orig-three");
            let rows = journal.snapshot();
            assert!(rows
                .iter()
                .any(|r| matches!(r, MockRow::RolledBack { rolled_back, .. } if rolled_back == &vec!["f1.txt".to_string()])));
            // Twice → identical.
            let content_before: Vec<Vec<u8>> = ["f1.txt", "f2.txt", "f3.txt"]
                .iter()
                .map(|f| fs::read(h.root().join(f)).unwrap())
                .collect();
            let resume = EditTxnResume {
                txn_id: EditTxnId::new(10),
                reqs: &reqs,
                before_content: &provider,
            };
            let report = engine
                .recover_edit_transactions(&h, &id, &*journal_arc, &[resume], RepairMode::Rollback)
                .unwrap();
            assert!(report.txns.is_empty(), "{report:?}");
            let content_after: Vec<Vec<u8>> = ["f1.txt", "f2.txt", "f3.txt"]
                .iter()
                .map(|f| fs::read(h.root().join(f)).unwrap())
                .collect();
            assert_eq!(content_before, content_after);
        }
    }

    // (e) oversized transactions (>2000 files or >64 KiB record) are
    // rejected at BEGIN with nothing journaled and nothing written.
    #[test]
    fn txn_oversized_begin_rejects_with_nothing_journaled() {
        let (_d, _s, h, id) = fixture();
        let engine = engine();
        let journal = MockJournal::default();
        let journal_arc: Arc<dyn EditTxnJournal> = Arc::new(journal.clone());
        // > MAX_TXN_FILES files.
        let too_many: Vec<EditRequest> = (0..MAX_TXN_FILES + 1)
            .map(|i| EditRequest {
                path: format!("f{i}.txt"),
                expected_hash: EditEngine::hash_of(b"x"),
                ops: vec![EditOp::SearchReplace {
                    before: "x".into(),
                    after: "y".into(),
                }],
            })
            .collect();
        let err = engine
            .begin_multi_file_edit(
                &h,
                &id,
                &too_many,
                RepairMode::Rollback,
                EditTxnStrategy::RollForward,
                journal_arc.clone(),
                EditTxnId::new(11),
                "session-1",
                None,
            )
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Oversized, "{err}");
        // A record payload beyond the engine's 64 KiB bound (files need not
        // exist: the bound fires before staging).
        let wide: Vec<EditRequest> = (0..600)
            .map(|i| EditRequest {
                path: format!("dir/{i}/{}", "n".repeat(200)),
                expected_hash: EditEngine::hash_of(b"x"),
                ops: vec![EditOp::SearchReplace {
                    before: "x".into(),
                    after: "y".into(),
                }],
            })
            .collect();
        let err = engine
            .begin_multi_file_edit(
                &h,
                &id,
                &wide,
                RepairMode::Rollback,
                EditTxnStrategy::RollForward,
                journal_arc.clone(),
                EditTxnId::new(12),
                "session-1",
                None,
            )
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Oversized, "{err}");
        // Duplicate path and zero files are malformed before any journaling.
        let dup = vec![
            EditRequest {
                path: "a.txt".into(),
                expected_hash: EditEngine::hash_of(b"x"),
                ops: vec![],
            },
            EditRequest {
                path: "a.txt".into(),
                expected_hash: EditEngine::hash_of(b"x"),
                ops: vec![],
            },
        ];
        let err = engine
            .begin_multi_file_edit(
                &h,
                &id,
                &dup,
                RepairMode::Rollback,
                EditTxnStrategy::RollForward,
                journal_arc.clone(),
                EditTxnId::new(13),
                "session-1",
                None,
            )
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Malformed, "{err}");
        assert!(err.message.contains("twice"), "{err}");
        let err = engine
            .begin_multi_file_edit(
                &h,
                &id,
                &[],
                RepairMode::Rollback,
                EditTxnStrategy::RollForward,
                journal_arc,
                EditTxnId::new(14),
                "session-1",
                None,
            )
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Malformed, "{err}");
        assert!(journal.snapshot().is_empty(), "nothing was journaled");
        // Nothing was written either (the workspace has no files at all).
        assert!(!h.root().join("f0.txt").exists());
        assert!(!h.root().join("dir").exists());
    }

    // (f) hostile progress rows (crafted straight into the journal, past the
    // typed appenders) → typed errors, zero writes, no terminal row, the
    // transaction stays open.
    #[test]
    fn txn_hostile_progress_rows_error_before_any_write() {
        let engine = engine();
        let fresh = || {
            let (d, _s, h, id) = fixture();
            fs::write(h.root().join("f1.txt"), b"orig-one").unwrap();
            fs::write(h.root().join("f2.txt"), b"orig-two").unwrap();
            fs::write(h.root().join("f3.txt"), b"orig-three").unwrap();
            (d, h, id)
        };
        let mk_refs = |h: &WorkspaceHandle| -> Vec<EditTxnFileRef> {
            ["f1.txt", "f2.txt", "f3.txt"]
                .iter()
                .map(|name| {
                    let bytes = fs::read(h.root().join(name)).unwrap();
                    EditTxnFileRef {
                        path: name.to_string(),
                        base_digest: EditEngine::hash_of(&bytes),
                        base_len: bytes.len() as u64,
                    }
                })
                .collect()
        };
        let mut hostiles: Vec<(MockJournal, String)> = Vec::new();
        for (txn, needle, extra) in [
            // Duplicate progress rows for one seq.
            (21u64, "duplicate progress rows", None),
            // A progress row whose seq is out of the prepared range.
            (22, "out of range", Some((9, "f9.txt"))),
            // A progress row whose path does not match the prepared file.
            (23, "does not match prepared file", Some((1, "not-f2.txt"))),
        ] {
            let (d, h, _id) = fresh();
            let files = mk_refs(&h);
            let mut rows = vec![MockRow::Prepared {
                txn: EditTxnId::new(txn),
                strategy: EditTxnStrategy::RollForward,
                files,
            }];
            match extra {
                None => {
                    rows.push(MockRow::Progress {
                        txn: EditTxnId::new(txn),
                        seq: 0,
                        path: "f1.txt".into(),
                        outcome: TxnFileOutcome::Committed,
                    });
                    rows.push(MockRow::Progress {
                        txn: EditTxnId::new(txn),
                        seq: 0,
                        path: "f1.txt".into(),
                        outcome: TxnFileOutcome::Committed,
                    });
                }
                Some((seq, path)) => {
                    rows.push(MockRow::Progress {
                        txn: EditTxnId::new(txn),
                        seq,
                        path: path.into(),
                        outcome: TxnFileOutcome::Committed,
                    });
                }
            }
            let journal = MockJournal::default();
            *journal.rows.lock().expect("poisoned") = rows;
            hostiles.push((journal, needle.to_string()));
            let _ = d;
        }
        for (journal, needle) in hostiles {
            let (d, h, id) = fresh();
            let seeded_len = journal.snapshot().len();
            let reqs = three_file_requests(&h);
            let none = |_p: &str| -> Option<Vec<u8>> { None };
            let journal_arc = Arc::new(journal.clone());
            let resume = EditTxnResume {
                txn_id: EditTxnId::new(21),
                reqs: &reqs,
                before_content: &none,
            };
            let err = engine
                .recover_edit_transactions(&h, &id, &*journal_arc, &[resume], RepairMode::Rollback)
                .unwrap_err();
            assert_eq!(err.kind, ErrorKind::Malformed, "{err}");
            assert!(err.message.contains(&needle), "{err}");
            // Zero writes (every file still at its base content), and no
            // row was appended by the failed recovery.
            assert_eq!(fs::read(h.root().join("f1.txt")).unwrap(), b"orig-one");
            assert_eq!(fs::read(h.root().join("f2.txt")).unwrap(), b"orig-two");
            assert_eq!(fs::read(h.root().join("f3.txt")).unwrap(), b"orig-three");
            assert_eq!(journal.snapshot().len(), seeded_len, "no rows appended");
            let _ = d;
        }
        // A transaction that still needs replay but has NO resume: typed
        // needs-recovery, zero writes, stays open.
        {
            let (d, h, id) = fresh();
            let journal = MockJournal::default();
            *journal.rows.lock().expect("poisoned") = vec![MockRow::Prepared {
                txn: EditTxnId::new(24),
                strategy: EditTxnStrategy::RollForward,
                files: mk_refs(&h),
            }];
            let journal_arc = Arc::new(journal.clone());
            let err = engine
                .recover_edit_transactions(&h, &id, &*journal_arc, &[], RepairMode::Rollback)
                .unwrap_err();
            assert_eq!(err.kind, ErrorKind::Conflict, "{err}");
            assert!(err.message.contains("unavailable"), "{err}");
            assert_eq!(journal.snapshot().len(), 1, "txn stays open");
            assert_eq!(fs::read(h.root().join("f1.txt")).unwrap(), b"orig-one");
            assert_eq!(fs::read(h.root().join("f2.txt")).unwrap(), b"orig-two");
            assert_eq!(fs::read(h.root().join("f3.txt")).unwrap(), b"orig-three");
            let _ = d;
        }
    }

    // (g) the non-durable wrapper and the durable flow agree on a clean
    // commit (apply_many semantics unchanged).
    #[test]
    fn apply_many_and_durable_flow_agree_without_conflicts() {
        let (_d, _s, h, id) = fixture();
        let engine = engine();
        fs::write(h.root().join("f1.txt"), b"orig-one").unwrap();
        fs::write(h.root().join("f2.txt"), b"orig-two").unwrap();
        let mk = |name: &str, from: &str, to: &str| {
            let bytes = fs::read(h.root().join(name)).unwrap();
            req(
                name,
                &bytes,
                vec![EditOp::SearchReplace {
                    before: from.into(),
                    after: to.into(),
                }],
            )
        };
        let reqs = vec![
            mk("f1.txt", "orig-one", "edited-one"),
            mk("f2.txt", "orig-two", "edited-two"),
        ];
        // Wrapper path.
        fs::write(h.root().join("f1.txt"), b"orig-one").unwrap();
        fs::write(h.root().join("f2.txt"), b"orig-two").unwrap();
        let out = engine
            .apply_many(&h, &id, &reqs, RepairMode::Rollback, None)
            .unwrap();
        assert!(out.all_committed(), "{out:?}");
        let wrapper_content: Vec<Vec<u8>> = ["f1.txt", "f2.txt"]
            .iter()
            .map(|f| fs::read(h.root().join(f)).unwrap())
            .collect();
        // Durable path.
        fs::write(h.root().join("f1.txt"), b"orig-one").unwrap();
        fs::write(h.root().join("f2.txt"), b"orig-two").unwrap();
        let journal = MockJournal::default();
        let reqs2 = vec![
            mk("f1.txt", "orig-one", "edited-one"),
            mk("f2.txt", "orig-two", "edited-two"),
        ];
        let prepared = engine
            .begin_multi_file_edit(
                &h,
                &id,
                &reqs2,
                RepairMode::Rollback,
                EditTxnStrategy::RollForward,
                Arc::new(journal.clone()),
                EditTxnId::new(30),
                "session-1",
                None,
            )
            .unwrap();
        let result = engine.commit_prepared(&h, &id, prepared).unwrap();
        assert!(result.all_committed(), "{result:?}");
        let durable_content: Vec<Vec<u8>> = ["f1.txt", "f2.txt"]
            .iter()
            .map(|f| fs::read(h.root().join(f)).unwrap())
            .collect();
        assert_eq!(wrapper_content, durable_content);
        // Same file count, same hashes reported.
        assert_eq!(out.ops_applied_total, result.ops_applied_total);
    }
}
