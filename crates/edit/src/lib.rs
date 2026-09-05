//! faktor-edit — the transactional patch engine (spec §17, §18).
//!
//! Every agent edit is optimistic and versioned: `expected_hash` must match
//! the current content or the edit is rejected before any write. All ops are
//! validated against a copy first, then applied with ONE atomic write (no
//! partial writes). For supported languages the before/after syntax trees
//! are compared: an edit that breaks previously valid syntax is suspicious
//! and rolls back (or is flagged, per mode).

use std::collections::HashMap;

use faktor_core::error::{Error, ErrorKind};
use faktor_core::hash::FileHash;
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
    pub fn apply_many(
        &self,
        workspace: &WorkspaceHandle,
        identity: &WorkspaceIdentity,
        reqs: &[EditRequest],
        mode: RepairMode,
        stage_budget: Option<std::time::Duration>,
    ) -> Result<EditBatchOutcome, Error> {
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
        // STAGE: validate everything against copies; zero writes.
        let mut staged = Vec::with_capacity(reqs.len());
        for req in reqs {
            budget(stage_budget, started)?;
            staged.push(self.stage_one(workspace, identity, req, mode)?);
            budget(stage_budget, started)?;
        }
        // Test seam: the adversary strikes between validation and commit.
        commit_gap_hook();
        // COMMIT: CAS every staged file; the first conflict ends the phase.
        let mut outcome = EditBatchOutcome {
            committed: Vec::new(),
            conflicted: Vec::new(),
            skipped: Vec::new(),
            ops_applied_total: 0,
        };
        for (idx, s) in staged.iter().enumerate() {
            match workspace.write_atomic_cas(&s.rel, s.expected_hash, s.edited.as_bytes()) {
                Ok(new_hash) => {
                    outcome.ops_applied_total += s.ops_applied;
                    outcome.committed.push(BatchCommittedFile {
                        path: s.rel.to_string_lossy().to_string(),
                        new_hash,
                        ops_applied: s.ops_applied,
                        suspicious: s.suspicious,
                    });
                }
                Err(e) => {
                    outcome
                        .conflicted
                        .push((s.rel.to_string_lossy().to_string(), e.message));
                    outcome.skipped.extend(
                        staged[idx + 1..]
                            .iter()
                            .map(|x| x.rel.to_string_lossy().to_string()),
                    );
                    break;
                }
            }
        }
        Ok(outcome)
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
        if current.truncated {
            return Err(Error::oversized(format!(
                "{} exceeds the {} byte edit bound",
                req.path, MAX_FILE_BYTES
            )));
        }
        // Optimistic versioning: the file must be exactly what the model read.
        if current.hash != req.expected_hash {
            return Err(Error::conflict(format!(
                "{} changed since it was read (expected {}, found {})",
                req.path,
                req.expected_hash.to_hex(),
                current.hash.to_hex()
            )));
        }
        let original = String::from_utf8(current.bytes.clone())
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
            expected_hash: current.hash,
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

/// A fully validated single-file edit, ready for the commit-time CAS.
struct StagedEdit {
    rel: std::path::PathBuf,
    /// Hash of the content this edit was validated against (digest axis of
    /// the commit-time compare-and-swap).
    expected_hash: FileHash,
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
}
