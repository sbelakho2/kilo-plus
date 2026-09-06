//! Campaign (c): durable multi-file edit transaction (wave-19 API) crashed
//! at every journal point (P0-76, 300 seeds).
//!
//! The op is `begin_multi_file_edit` (stages everything, journals the
//! `Prepared` row record-first) then `commit_prepared` (per-file CAS write
//! with a journaled `Progress` row AFTER each write, terminal `Committed`
//! row at the end) — the wave-19 seam semantics over REAL workspace files.
//! The journal is the campaign's typed-ledger stand-in (the role the
//! session ledger plays in production), with the wave-19 crash knobs: a
//! panic BEFORE a record push = the record never became durable; AFTER the
//! push = it did. Rows survive the "crash" (engine + prepared data die,
//! the journal Arc does not).
//!
//! Isolation: every (seed, boundary) world writes its files under its OWN
//! subdirectory of one process-wide watched workspace root (the notify
//! watcher is created exactly once per process — creating it per tempdir
//! costs seconds each on macOS FSEvents and would dominate the campaign).
//! No world ever reads another world's files or journal.
//!
//! Boundaries (three files, seed-derived content):
//! - `begin.before_prepared` (PreOp): nothing journaled, files untouched;
//!   recovery sees nothing open, the caller re-issues the whole op.
//! - `begin.after_prepared` (FullyCommitted): the Prepared row is durable,
//!   commit never ran; recovery replays all three files.
//! - `commit.progress_before#j` (Ambiguous): file j's write landed but its
//!   Progress row never became durable. Recovery must REFUSE with the
//!   engine's typed needs-recovery error naming file j and must not clobber
//!   anything. The campaign then plays the durable layer's verification
//!   decision (the file's bytes hash to the intended target; the progress
//!   row is journaled) and recovery converges to the reference.
//! - `commit.progress_after#j` (FullyCommitted): file j committed and
//!   journaled; recovery skips it and replays the rest.
//! - `commit.terminal_before` (FullyCommitted): every write + progress row
//!   is durable; only the terminal row crashed — recovery appends it.
//! - `recover.*` (crash DURING recovery, after the commit crashed after
//!   progress row 0 so recovery replays files 1..2): the recovery epoch's
//!   own writes crash before/after their rows (`recover.progress_before` is
//!   Ambiguous at file 1; the others finish on the second recovery run —
//!   recovery is idempotent and crash-re-runnable).
//!
//! Final certification per (seed, boundary): file bytes AND journal rows
//! EQUAL the uninterrupted reference world, and every armed knob fired
//! exactly once.

use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use faktor_core::error::Error as EditError;
use faktor_core::id::{TaskId, WorkspaceId, WorktreeId};
use faktor_core::state::{EditTxnId, EditTxnStrategy};
use faktor_core::WorkspaceIdentity;
use faktor_edit::{
    EditEngine, EditOp, EditRequest, EditTxnFileRef, EditTxnJournal, EditTxnResume, OpenEditTxn,
    PreparedEdit, RecoveryReport, TxnFileOutcome,
};
use faktor_fs::{WorkspaceFileService, WorkspaceHandle};

use super::{check_equals, BoundarySpec, Campaign, CrashClass, Lcg, WorldState};

/// Seeds of the full [fault]-gated campaign.
pub const FULL_SEEDS: u64 = 300;
/// Seeds of the normal-mode smoke run.
pub const SMOKE_SEEDS: u64 = 3;

const FILES: usize = 3;

// ---------------------------------------------------------------------------
// Process-wide workspace kit: ONE notify watcher, one root, per-world
// subdirectories (notify watcher creation costs seconds on macOS; a fresh
// watcher per world would dominate every run).
// ---------------------------------------------------------------------------

struct SharedKit {
    _dir: &'static tempfile::TempDir,
    service: Arc<WorkspaceFileService>,
    handle: WorkspaceHandle,
    identity: WorkspaceIdentity,
}

fn shared_kit() -> &'static SharedKit {
    static KIT: OnceLock<SharedKit> = OnceLock::new();
    KIT.get_or_init(|| {
        let dir = Box::leak(Box::new(tempfile::tempdir().expect("tempdir")));
        let root = dir.path().join("ws");
        fs::create_dir_all(&root).unwrap();
        let service = WorkspaceFileService::new();
        let handle = service.open(WorkspaceId::new(7), root).unwrap();
        let identity =
            WorkspaceIdentity::new(WorkspaceId::new(7), WorktreeId::new(1), TaskId::new(1));
        SharedKit {
            _dir: dir,
            service,
            handle,
            identity,
        }
    })
}

// ---------------------------------------------------------------------------
// The typed-ledger stand-in journal with the wave-19 crash knobs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
enum Row {
    Prepared {
        txn: u64,
        files: Vec<EditTxnFileRef>,
    },
    Progress {
        txn: u64,
        seq: u64,
        path: String,
        outcome: TxnFileOutcome,
    },
    Committed {
        txn: u64,
        committed: Vec<String>,
    },
    RolledBack {
        txn: u64,
        rolled_back: Vec<String>,
    },
}

// One knob per crash boundary phase; the Crash- prefix reads as the
// phase ordering in the seam's own vocabulary.
#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Knob {
    CrashBeforePrepared(u64),
    CrashAfterPrepared(u64),
    CrashBeforeProgress(u64),
    CrashAfterProgress(u64),
    CrashBeforeTerminal(u64),
    CrashAfterTerminal(u64),
}

/// One journal instance per world. Every knob fires at most once.
#[derive(Clone)]
struct Journal {
    rows: Arc<Mutex<Vec<Row>>>,
    knobs: Arc<Mutex<Vec<Knob>>>,
    prepared_pushes: Arc<AtomicU64>,
    progress_pushes: Arc<AtomicU64>,
    terminal_pushes: Arc<AtomicU64>,
}

impl std::fmt::Debug for Journal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Journal")
            .field("rows", &self.snapshot())
            .field("armed_knobs", &self.knobs.lock().expect("poisoned").clone())
            .finish()
    }
}

impl Journal {
    fn new(knobs: Vec<Knob>) -> Self {
        Self {
            rows: Arc::new(Mutex::new(Vec::new())),
            knobs: Arc::new(Mutex::new(knobs)),
            prepared_pushes: Arc::new(AtomicU64::new(0)),
            progress_pushes: Arc::new(AtomicU64::new(0)),
            terminal_pushes: Arc::new(AtomicU64::new(0)),
        }
    }

    fn snapshot(&self) -> Vec<Row> {
        self.rows.lock().expect("journal poisoned").clone()
    }

    fn armed(&self) -> bool {
        !self.knobs.lock().expect("poisoned").is_empty()
    }

    fn fire(&self, knob: Knob) {
        let mut knobs = self.knobs.lock().expect("knobs poisoned");
        if let Some(pos) = knobs.iter().position(|k| *k == knob) {
            knobs.remove(pos);
            drop(knobs);
            panic!("[fault-seam] simulated edit-txn journal crash at {knob:?}");
        }
    }
}

impl EditTxnJournal for Journal {
    fn entry_payload_cap(&self) -> usize {
        64 * 1024
    }

    fn record_prepared(
        &self,
        txn_id: EditTxnId,
        _session: &str,
        files: &[EditTxnFileRef],
        _strategy: EditTxnStrategy,
    ) -> Result<(), EditError> {
        let ordinal = self.prepared_pushes.fetch_add(1, Ordering::SeqCst);
        self.fire(Knob::CrashBeforePrepared(ordinal));
        self.rows.lock().expect("poisoned").push(Row::Prepared {
            txn: txn_id.raw(),
            files: files.to_vec(),
        });
        self.fire(Knob::CrashAfterPrepared(ordinal));
        Ok(())
    }

    fn record_progress(
        &self,
        txn_id: EditTxnId,
        seq: u64,
        path: &str,
        outcome: TxnFileOutcome,
    ) -> Result<(), EditError> {
        let ordinal = self.progress_pushes.fetch_add(1, Ordering::SeqCst);
        self.fire(Knob::CrashBeforeProgress(ordinal));
        self.rows.lock().expect("poisoned").push(Row::Progress {
            txn: txn_id.raw(),
            seq,
            path: path.to_string(),
            outcome,
        });
        self.fire(Knob::CrashAfterProgress(ordinal));
        Ok(())
    }

    fn record_committed(
        &self,
        txn_id: EditTxnId,
        committed: &[String],
        _conflicted: &[String],
        _skipped: &[String],
    ) -> Result<(), EditError> {
        let ordinal = self.terminal_pushes.fetch_add(1, Ordering::SeqCst);
        self.fire(Knob::CrashBeforeTerminal(ordinal));
        self.rows.lock().expect("poisoned").push(Row::Committed {
            txn: txn_id.raw(),
            committed: committed.to_vec(),
        });
        self.fire(Knob::CrashAfterTerminal(ordinal));
        Ok(())
    }

    fn record_rolled_back(
        &self,
        txn_id: EditTxnId,
        rolled_back: &[String],
        _rollback_conflicts: &[String],
    ) -> Result<(), EditError> {
        let ordinal = self.terminal_pushes.fetch_add(1, Ordering::SeqCst);
        self.fire(Knob::CrashBeforeTerminal(ordinal));
        self.rows.lock().expect("poisoned").push(Row::RolledBack {
            txn: txn_id.raw(),
            rolled_back: rolled_back.to_vec(),
        });
        self.fire(Knob::CrashAfterTerminal(ordinal));
        Ok(())
    }

    fn open_transactions(&self) -> Result<Vec<OpenEditTxn>, EditError> {
        let rows = self.snapshot();
        let mut by_txn: Vec<u64> = Vec::new();
        for r in &rows {
            let txn = row_txn(r);
            if !by_txn.contains(&txn) {
                by_txn.push(txn);
            }
        }
        let mut out = Vec::new();
        for txn in by_txn {
            let txn_rows: Vec<&Row> = rows.iter().filter(|r| row_txn(r) == txn).collect();
            let terminal = txn_rows
                .iter()
                .any(|r| matches!(r, Row::Committed { .. } | Row::RolledBack { .. }));
            if terminal {
                continue;
            }
            let mut files = Vec::new();
            let mut progress = Vec::new();
            for r in txn_rows {
                match r {
                    Row::Prepared { files: f, .. } => files.extend(f.iter().cloned()),
                    Row::Progress {
                        seq, path, outcome, ..
                    } => progress.push(faktor_edit::TxnProgressRow {
                        seq: *seq,
                        path: path.clone(),
                        outcome: *outcome,
                    }),
                    _ => {}
                }
            }
            out.push(OpenEditTxn {
                txn_id: EditTxnId::new(txn),
                strategy: EditTxnStrategy::RollForward,
                files,
                progress,
            });
        }
        Ok(out)
    }
}

fn row_txn(r: &Row) -> u64 {
    match r {
        Row::Prepared { txn, .. }
        | Row::Progress { txn, .. }
        | Row::Committed { txn, .. }
        | Row::RolledBack { txn, .. } => *txn,
    }
}

// ---------------------------------------------------------------------------
// Seed-derived three-file worlds
// ---------------------------------------------------------------------------

struct EditWorld {
    /// Subdirectory of the shared root this world owns (isolation unit).
    tag: String,
    engine: EditEngine,
    handle: WorkspaceHandle,
    identity: WorkspaceIdentity,
    /// Pre-op content of each file.
    base: Vec<Vec<u8>>,
    /// Content of each file after the intended edits.
    target: Vec<Vec<u8>>,
    reqs: Vec<EditRequest>,
    journal: Journal,
    txn_id: u64,
}

fn file_names() -> [&'static str; FILES] {
    ["f0.txt", "f1.txt", "f2.txt"]
}

/// Base content: deterministic random letters around ONE unique sentinel
/// (the sentinel is the single search-replace anchor). Unicode prefixes
/// exercise multibyte digests; the sentinel itself is pure ASCII.
fn base_content(seed: u64, i: usize) -> Vec<u8> {
    let mut lcg = Lcg::new(seed ^ (i as u64).wrapping_mul(0xBEEF_0000_0000_0001));
    let n = lcg.below(1500) as usize;
    let mut head = String::from("αβγ");
    while head.len() < n {
        head.push(char::from(b'a' + lcg.below(26) as u8));
    }
    let sentinel = sentinel_of(seed, i);
    let mut out = head.into_bytes();
    out.extend_from_slice(sentinel.as_bytes());
    out.extend_from_slice(b"\n");
    out
}

fn sentinel_of(seed: u64, i: usize) -> String {
    format!("@@anchor-{}-{}@@", seed % 7919, i)
}

fn replacement_of(seed: u64, i: usize) -> String {
    let mut lcg = Lcg::new(seed ^ (i as u64).wrapping_mul(0xCAFE_0000_0000_0003));
    format!(
        ">>edited-{}-{}<<{}",
        seed % 7919,
        i,
        "λ".repeat(1 + lcg.below(6) as usize)
    )
}

fn target_content(seed: u64, i: usize) -> Vec<u8> {
    // target == base with the unique sentinel replaced exactly once.
    let base = base_content(seed, i);
    let text = String::from_utf8_lossy(&base);
    let out = text.replacen(&sentinel_of(seed, i), &replacement_of(seed, i), 1);
    out.into_bytes()
}

fn request_for(name: &str, seed: u64, i: usize) -> EditRequest {
    // expected_hash validates against the CURRENT (base) content; the op
    // replaces the unique sentinel with the replacement, yielding exactly
    // `target_content`.
    EditRequest {
        path: name.to_string(),
        expected_hash: EditEngine::hash_of(&base_content(seed, i)),
        ops: vec![EditOp::SearchReplace {
            before: sentinel_of(seed, i),
            after: replacement_of(seed, i),
        }],
    }
}

impl EditWorld {
    /// `tag` uniquely names this world's subdirectory (seed + boundary),
    /// so parallel tests and repeated runs never share files.
    fn new(seed: u64, tag: &str, knobs: Vec<Knob>) -> EditWorld {
        let kit = shared_kit();
        let rel = |name: &str| format!("{tag}/{name}");
        fs::create_dir_all(kit.handle.root().join(tag)).expect("world dir");
        let mut base = Vec::new();
        let mut target = Vec::new();
        let mut reqs = Vec::new();
        for (i, name) in file_names().iter().enumerate() {
            let b = base_content(seed, i);
            fs::write(kit.handle.root().join(rel(name)), &b).expect("base file");
            let t = target_content(seed, i);
            base.push(b);
            target.push(t);
            let rel_path = rel(name);
            reqs.push(request_for(&rel_path, seed, i));
        }
        let journal = Journal::new(knobs);
        EditWorld {
            tag: tag.to_string(),
            engine: EditEngine::new(kit.service.clone()),
            handle: kit.handle.clone(),
            identity: kit.identity,
            base,
            target,
            reqs,
            journal,
            txn_id: (seed % (u64::MAX - 2)) + 1,
        }
    }

    fn begin(&self) -> Result<PreparedEdit, EditError> {
        self.engine.begin_multi_file_edit(
            &self.handle,
            &self.identity,
            &self.reqs,
            faktor_edit::RepairMode::Rollback,
            EditTxnStrategy::RollForward,
            Arc::new(self.journal.clone()),
            EditTxnId::new(self.txn_id),
            "session-1",
            None,
        )
    }

    fn resume(&self) -> EditTxnResume<'_> {
        EditTxnResume {
            txn_id: EditTxnId::new(self.txn_id),
            reqs: &self.reqs,
            before_content: &|_| None,
        }
    }

    fn commit(&self, prepared: PreparedEdit) -> Result<faktor_edit::EditTxnResult, EditError> {
        self.engine
            .commit_prepared(&self.handle, &self.identity, prepared)
    }

    /// Drive `recover_edit_transactions` to a terminal outcome. Recovery
    /// itself may crash (its own armed journal knobs) — re-run until the
    /// knobs are consumed, exactly like a daemon restarting after a crash
    /// DURING recovery. Returns `Done` when the epoch completed and
    /// `Refused` when the engine declared an ambiguous residue.
    fn drive_recovery(&self) -> Result<RecoveryEnd, String> {
        let max_attempts = 2 + self.journal.knobs.lock().expect("poisoned").len();
        for _ in 0..max_attempts {
            let dyn_journal: Arc<dyn EditTxnJournal> = Arc::new(self.journal.clone());
            let caught = catch_unwind(AssertUnwindSafe(|| {
                self.engine.recover_edit_transactions(
                    &self.handle,
                    &self.identity,
                    dyn_journal.as_ref(),
                    &[self.resume()],
                    faktor_edit::RepairMode::Rollback,
                )
            }));
            match caught {
                Ok(Ok(report)) => return Ok(RecoveryEnd::Done(report)),
                Ok(Err(e)) => return Ok(RecoveryEnd::Refused(e.to_string())),
                // A recovery crash knob fired: restart the recovery run.
                Err(_) => continue,
            }
        }
        Err("recovery kept crashing beyond its armed knobs".into())
    }

    fn file_path(&self, i: usize) -> std::path::PathBuf {
        self.handle
            .root()
            .join(format!("{}/{}", self.tag, file_names()[i]))
    }

    fn files(&self) -> Vec<Vec<u8>> {
        (0..FILES)
            .map(|i| fs::read(self.file_path(i)).expect("file readable"))
            .collect()
    }

    fn dump(&self) -> WorldState {
        let mut lines = Vec::new();
        for (i, bytes) in self.files().iter().enumerate() {
            lines.push(format!("file:{i}:{}", hex(bytes)));
        }
        // Journal paths carry the world's isolation tag (subdirectory of
        // the shared root) — that is harness plumbing, not transaction
        // state. Strip it so worlds of one seed compare equal.
        let strip = format!("{}/", self.tag);
        let norm = |p: &str| p.strip_prefix(&strip).unwrap_or(p).to_string();
        for row in self.journal.snapshot() {
            lines.push(format!(
                "row:{}",
                match row {
                    Row::Prepared { txn, files } => format!(
                        "Prepared {{ txn: {txn}, files: {:?} }}",
                        files
                            .iter()
                            .map(|f| EditTxnFileRef {
                                path: norm(&f.path),
                                base_digest: f.base_digest,
                                base_len: f.base_len,
                            })
                            .collect::<Vec<_>>()
                    ),
                    Row::Progress {
                        txn,
                        seq,
                        path,
                        outcome,
                    } => format!(
                        "Progress {{ txn: {txn}, seq: {seq}, path: {:?}, outcome: {outcome:?} }}",
                        norm(&path)
                    ),
                    Row::Committed { txn, committed } => format!(
                        "Committed {{ txn: {txn}, committed: {:?} }}",
                        committed.iter().map(|p| norm(p)).collect::<Vec<_>>()
                    ),
                    Row::RolledBack { txn, rolled_back } => format!(
                        "RolledBack {{ txn: {txn}, rolled_back: {:?} }}",
                        rolled_back.iter().map(|p| norm(p)).collect::<Vec<_>>()
                    ),
                }
            ));
        }
        WorldState { lines }
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

// ---------------------------------------------------------------------------
// The campaign
// ---------------------------------------------------------------------------

/// (name, class, ambiguous file index when applicable, crash knobs)
/// Progress ordinals: the commit pushes rows 0,1,2 (files 0,1,2); a commit
/// that crashes after progress row 0 leaves files 1..2 base, so recovery
/// REPLAYS them (pushing rows 1,2) — the `recover.*` boundaries arm
/// against exactly those pushes.
const BOUNDARIES: &[(&str, CrashClass, Option<usize>, &[Knob])] = &[
    (
        "begin.before_prepared",
        CrashClass::PreOp,
        None,
        &[Knob::CrashBeforePrepared(0)],
    ),
    (
        "begin.after_prepared",
        CrashClass::FullyCommitted,
        None,
        &[Knob::CrashAfterPrepared(0)],
    ),
    (
        "commit.progress_before#0",
        CrashClass::Ambiguous,
        Some(0),
        &[Knob::CrashBeforeProgress(0)],
    ),
    (
        "commit.progress_after#0",
        CrashClass::FullyCommitted,
        None,
        &[Knob::CrashAfterProgress(0)],
    ),
    (
        "commit.progress_before#1",
        CrashClass::Ambiguous,
        Some(1),
        &[Knob::CrashBeforeProgress(1)],
    ),
    (
        "commit.progress_after#1",
        CrashClass::FullyCommitted,
        None,
        &[Knob::CrashAfterProgress(1)],
    ),
    (
        "commit.progress_before#2",
        CrashClass::Ambiguous,
        Some(2),
        &[Knob::CrashBeforeProgress(2)],
    ),
    (
        "commit.terminal_before",
        CrashClass::FullyCommitted,
        None,
        &[Knob::CrashBeforeTerminal(0)],
    ),
    // The commit crashes after file 0's progress row; the recovery epoch
    // replays file 1 (ordinal 1) and file 2 (ordinal 2), then appends the
    // terminal row (ordinal 0).
    (
        "recover.progress_before",
        CrashClass::Ambiguous,
        Some(1),
        &[Knob::CrashAfterProgress(0), Knob::CrashBeforeProgress(1)],
    ),
    (
        "recover.progress_after",
        CrashClass::FullyCommitted,
        None,
        &[Knob::CrashAfterProgress(0), Knob::CrashAfterProgress(1)],
    ),
    (
        "recover.terminal_before",
        CrashClass::FullyCommitted,
        None,
        &[Knob::CrashAfterProgress(0), Knob::CrashBeforeTerminal(0)],
    ),
];

static BOUNDARY_SPECS: &[BoundarySpec] = &[
    BoundarySpec {
        name: "begin.before_prepared",
        class: CrashClass::PreOp,
    },
    BoundarySpec {
        name: "begin.after_prepared",
        class: CrashClass::FullyCommitted,
    },
    BoundarySpec {
        name: "commit.progress_before#0",
        class: CrashClass::Ambiguous,
    },
    BoundarySpec {
        name: "commit.progress_after#0",
        class: CrashClass::FullyCommitted,
    },
    BoundarySpec {
        name: "commit.progress_before#1",
        class: CrashClass::Ambiguous,
    },
    BoundarySpec {
        name: "commit.progress_after#1",
        class: CrashClass::FullyCommitted,
    },
    BoundarySpec {
        name: "commit.progress_before#2",
        class: CrashClass::Ambiguous,
    },
    BoundarySpec {
        name: "commit.terminal_before",
        class: CrashClass::FullyCommitted,
    },
    BoundarySpec {
        name: "recover.progress_before",
        class: CrashClass::Ambiguous,
    },
    BoundarySpec {
        name: "recover.progress_after",
        class: CrashClass::FullyCommitted,
    },
    BoundarySpec {
        name: "recover.terminal_before",
        class: CrashClass::FullyCommitted,
    },
];

fn reference(seed: u64) -> Result<WorldState, String> {
    let tag = format!("s{seed:x}-ref");
    let w = EditWorld::new(seed, &tag, Vec::new());
    let prepared = w.begin().map_err(|e| e.to_string())?;
    w.commit(prepared).map_err(|e| e.to_string())?;
    let world = w.dump();
    assert!(
        world.lines.iter().any(|l| l.starts_with("row:Committed")),
        "reference must end committed"
    );
    Ok(world)
}

fn crash_run(seed: u64, boundary: &BoundarySpec) -> Result<WorldState, String> {
    let (_, class, ambiguous_file, knobs) = BOUNDARIES
        .iter()
        .find(|(name, ..)| *name == boundary.name)
        .ok_or_else(|| format!("unknown boundary {:?}", boundary.name))?;
    let tag = format!("s{seed:x}-{}", boundary.name.replace('#', "_"));
    let w = EditWorld::new(seed, &tag, knobs.to_vec());

    // -------- interrupted begin/commit with the armed knobs --------------
    let begin_caught = catch_unwind(AssertUnwindSafe(|| w.begin()));
    match begin_caught {
        Err(_) if w.journal.snapshot().is_empty() => {
            // begin.before_prepared: nothing durable. Assert the PreOp
            // residue, then re-issue the whole op (the caller's contract).
            if *class != CrashClass::PreOp {
                return Err("a non-PreOp boundary crashed before the prepared row".into());
            }
            if w.files() != w.base {
                return Err("PreOp begin crash touched files".into());
            }
            let prepared = w.begin().map_err(|e| e.to_string())?;
            w.commit(prepared).map_err(|e| e.to_string())?;
            if w.journal.armed() {
                return Err("armed knobs were never all consumed".into());
            }
            return Ok(w.dump());
        }
        Err(_) => {
            // begin.after_prepared: the Prepared row is durable; the
            // commit never ran. Fall through to recovery below.
            if *class != CrashClass::FullyCommitted {
                return Err(
                    "begin crashed after the prepared row outside its declared class".into(),
                );
            }
        }
        Ok(Ok(prepared)) => {
            let commit_caught = catch_unwind(AssertUnwindSafe(|| w.commit(prepared)));
            match commit_caught {
                Err(_) => {}
                Ok(Ok(_result)) => {
                    // No crash fired anywhere: a certification gap.
                    return Err(
                        "seam boundary was never reached: begin+commit completed without crashing"
                            .into(),
                    );
                }
                Ok(Err(e)) => return Err(format!("commit failed without crashing: {e}")),
            }
        }
        Ok(Err(e)) => {
            return Err(format!("begin failed without crashing: {e}"));
        }
    }

    // -------- recovery ----------------------------------------------------
    let end = w.drive_recovery()?;
    match end {
        RecoveryEnd::Done(_report) => {
            if *class == CrashClass::Ambiguous {
                return Err(format!(
                    "boundary {:?} must refuse recovery with a typed needs-recovery error, but recovery succeeded",
                    boundary.name
                ));
            }
            if w.journal.armed() {
                return Err("armed knobs were never all consumed".into());
            }
            Ok(w.dump())
        }
        RecoveryEnd::Refused(refusal) => {
            if *class != CrashClass::Ambiguous {
                return Err(format!(
                    "recovery refused a {} boundary with: {refusal}",
                    class_name(class)
                ));
            }
            let seq =
                ambiguous_file.ok_or_else(|| "ambiguous boundary has no file index".to_string())?;
            let name = format!("{}/{}", w.tag, file_names()[seq]);
            if !refusal.contains(file_names()[seq]) || !refusal.contains("needs recovery") {
                return Err(format!(
                    "recovery must name {} in its typed needs-recovery refusal, got: {refusal}",
                    file_names()[seq]
                ));
            }
            // Durable residue: files before the ambiguity are committed,
            // the ambiguous file holds OUR OWN crashed write (no external
            // writer exists in this campaign), files behind it are base —
            // nothing was clobbered and nothing was silently accepted.
            let current = w.files();
            for (i, cur) in current.iter().enumerate().take(seq) {
                if *cur != w.target[i] {
                    return Err(format!(
                        "file {i} must be committed before the ambiguous residue"
                    ));
                }
            }
            if current[seq] != w.target[seq] {
                return Err("the ambiguous file must hold the crashed write's bytes".into());
            }
            for (i, cur) in current.iter().enumerate().take(FILES).skip(seq + 1) {
                if *cur != w.base[i] {
                    return Err(format!("file {i} must still be base behind the ambiguity"));
                }
            }
            // The durable layer's verification decision: the file hashes to
            // the intended target, so the write is declared committed.
            let bytes = fs::read(w.file_path(seq)).map_err(|e| e.to_string())?;
            let actual = EditEngine::hash_of(&bytes);
            let intended = EditEngine::hash_of(&w.target[seq]);
            if actual != intended {
                return Err(format!(
                    "verification of the ambiguous file {name} failed: bytes are not the intended write"
                ));
            }
            w.journal
                .record_progress(
                    EditTxnId::new(w.txn_id),
                    seq as u64,
                    &name,
                    TxnFileOutcome::Committed,
                )
                .map_err(|e| format!("reconciliation journal write failed: {e}"))?;
            match w.drive_recovery()? {
                RecoveryEnd::Done(_) => {}
                RecoveryEnd::Refused(e) => {
                    return Err(format!("post-verification recovery refused again: {e}"))
                }
            }
            if w.journal.armed() {
                return Err("armed knobs were never all consumed".into());
            }
            Ok(w.dump())
        }
    }
}

/// Terminal outcome of one recovery drive.
enum RecoveryEnd {
    Done(RecoveryReport),
    Refused(String),
}

fn class_name(c: &CrashClass) -> &'static str {
    match c {
        CrashClass::PreOp => "PreOp",
        CrashClass::FullyCommitted => "FullyCommitted",
        CrashClass::Ambiguous => "Ambiguous",
    }
}

pub fn campaign() -> Campaign {
    Campaign {
        name: "edit-txn-journal-crash",
        boundaries: BOUNDARY_SPECS,
        reference,
        crash_run,
        check: check_equals,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn smoke_edit_txn_journal_crash() {
    let c = campaign();
    let checks = super::run_campaign(&c, SMOKE_SEEDS).expect("smoke must pass");
    assert_eq!(checks, SMOKE_SEEDS * c.boundaries.len() as u64);
}

#[test]
#[ignore = "[fault] edit txn begin/commit/recover crash at every journal point, 300 seeds"]
fn full_edit_txn_journal_crash() {
    let c = campaign();
    let checks = super::run_campaign(&c, FULL_SEEDS).expect("full campaign must pass");
    assert_eq!(checks, FULL_SEEDS * c.boundaries.len() as u64);
}
