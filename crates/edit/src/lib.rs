//! kilop-edit — the transactional patch engine (spec §17, §18).
//!
//! Every agent edit is optimistic and versioned: `expected_hash` must match
//! the current content or the edit is rejected before any write. All ops are
//! validated against a copy first, then applied with ONE atomic write (no
//! partial writes). For supported languages the before/after syntax trees
//! are compared: an edit that breaks previously valid syntax is suspicious
//! and rolls back (or is flagged, per mode).

use std::collections::HashMap;

use kilop_core::error::{Error, ErrorKind};
use kilop_core::hash::FileHash;
use kilop_core::WorkspaceIdentity;
use kilop_fs::WorkspaceHandle;

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
    fs: std::sync::Arc<kilop_fs::WorkspaceFileService>,
}

impl EditEngine {
    pub fn new(fs: std::sync::Arc<kilop_fs::WorkspaceFileService>) -> Self {
        Self { fs }
    }

    /// Apply ops transactionally. Validates EVERYTHING against a copy first:
    /// a failing op N means ops 1..N-1 must not be written either.
    pub fn apply(
        &self,
        workspace: &WorkspaceHandle,
        identity: &WorkspaceIdentity,
        req: &EditRequest,
        mode: RepairMode,
    ) -> Result<EditOutcome, Error> {
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

        let new_hash = workspace.write_atomic(rel, edited.as_bytes())?;
        Ok(EditOutcome {
            new_hash,
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
        EditOp::Range { start, end, replacement } => {
            if start > end {
                return Err(Error::malformed(format!(
                    "range start {start} > end {end}"
                )));
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
                0 => Err(Error::malformed(format!(
                    "search text not found (0 matches)"
                ))),
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

fn language_for(rel: &std::path::Path) -> Option<(&'static str, tree_sitter::Language)> {
    match rel.extension().and_then(|e| e.to_str()) {
        Some("rs") => Some(("rust", tree_sitter_rust::LANGUAGE.into())),
        Some("py") => Some(("python", tree_sitter_python::LANGUAGE.into())),
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
    fn walk(
        node: tree_sitter::Node<'_>,
        src: &[u8],
        first: &mut Option<(usize, String)>,
    ) {
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

/// Lookup helper used by tests to fetch tree-sitter grammar info.
#[allow(dead_code)]
fn _grammar_names() -> HashMap<&'static str, &'static str> {
    HashMap::from([("rust", "tree-sitter-rust"), ("python", "tree-sitter-python")])
}

#[cfg(test)]
mod tests {
    use super::*;
    use kilop_core::id::WorkspaceId;
    use std::fs;

    fn fixture() -> (tempfile::TempDir, std::sync::Arc<kilop_fs::WorkspaceFileService>, WorkspaceHandle, WorkspaceIdentity) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("ws");
        fs::create_dir_all(&root).unwrap();
        let service = kilop_fs::WorkspaceFileService::new();
        let handle = service.open(WorkspaceId::new(1), root.clone()).unwrap();
        let identity = WorkspaceIdentity::new(
            WorkspaceId::new(1),
            kilop_core::WorktreeId::new(1),
            kilop_core::TaskId::new(1),
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
        let engine = EditEngine::new(kilop_fs::WorkspaceFileService::new());
        fs::write(h.root().join("a.txt"), b"one").unwrap();
        // Model read "one" but the file became "two" (another writer).
        fs::write(h.root().join("a.txt"), b"two").unwrap();
        let r = req("a.txt", b"one", vec![EditOp::SearchReplace {
            before: "two".into(),
            after: "three".into(),
        }]);
        let err = engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap_err();
        assert!(err.kind == ErrorKind::Conflict);
        // File untouched.
        assert_eq!(fs::read(h.root().join("a.txt")).unwrap(), b"two");
    }

    #[test]
    fn range_edit_applies_and_hashes() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(kilop_fs::WorkspaceFileService::new());
        fs::write(h.root().join("a.txt"), b"hello world").unwrap();
        let r = req("a.txt", b"hello world", vec![EditOp::Range {
            start: 0,
            end: 5,
            replacement: "goodbye".into(),
        }]);
        let out = engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap();
        assert_eq!(out.ops_applied, 1);
        assert!(!out.suspicious);
        assert_eq!(fs::read(h.root().join("a.txt")).unwrap(), b"goodbye world");
        assert_eq!(out.new_hash, EditEngine::hash_of(b"goodbye world"));
    }

    #[test]
    fn search_replace_unique_ok() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(kilop_fs::WorkspaceFileService::new());
        fs::write(h.root().join("a.txt"), b"fn main() {}\n").unwrap();
        let r = req("a.txt", b"fn main() {}\n", vec![EditOp::SearchReplace {
            before: "fn main".into(),
            after: "fn entry".into(),
        }]);
        engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap();
        assert_eq!(fs::read(h.root().join("a.txt")).unwrap(), b"fn entry() {}\n");
    }

    #[test]
    fn search_replace_zero_matches_malformed() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(kilop_fs::WorkspaceFileService::new());
        fs::write(h.root().join("a.txt"), b"abc").unwrap();
        let r = req("a.txt", b"abc", vec![EditOp::SearchReplace {
            before: "zzz".into(),
            after: "x".into(),
        }]);
        let err = engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap_err();
        assert!(err.kind == ErrorKind::Malformed);
        assert_eq!(fs::read(h.root().join("a.txt")).unwrap(), b"abc");
    }

    #[test]
    fn search_replace_multiple_matches_conflict() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(kilop_fs::WorkspaceFileService::new());
        fs::write(h.root().join("a.txt"), b"aaa").unwrap();
        let r = req("a.txt", b"aaa", vec![EditOp::SearchReplace {
            before: "a".into(),
            after: "b".into(),
        }]);
        let err = engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap_err();
        assert!(err.kind == ErrorKind::Conflict);
    }

    #[test]
    fn out_of_bounds_range_malformed() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(kilop_fs::WorkspaceFileService::new());
        fs::write(h.root().join("a.txt"), b"abc").unwrap();
        let r = req("a.txt", b"abc", vec![EditOp::Range {
            start: 1,
            end: 99,
            replacement: "x".into(),
        }]);
        let err = engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap_err();
        assert!(err.kind == ErrorKind::Malformed);
        assert!(err.message.contains("op 1"));
    }

    #[test]
    fn split_codepoint_offsets_malformed() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(kilop_fs::WorkspaceFileService::new());
        let content = "aé😀b"; // bytes: a(1) é(2) 😀(4) b(1)
        fs::write(h.root().join("a.txt"), content).unwrap();
        // offset 2 lands inside 'é' (2 bytes: 0xE9 is at 1..3).
        let r = req("a.txt", content.as_bytes(), vec![EditOp::Range {
            start: 2,
            end: 3,
            replacement: "x".into(),
        }]);
        let err = engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap_err();
        assert!(err.kind == ErrorKind::Malformed);
        // A boundary-correct edit works.
        let r = req("a.txt", content.as_bytes(), vec![EditOp::Range {
            start: 1,
            end: 7,
            replacement: "Z".into(),
        }]);
        let out = engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap();
        assert_eq!(fs::read(h.root().join("a.txt")).unwrap(), b"aZb");
        assert!(!out.suspicious);
    }

    #[test]
    fn rust_parse_error_rolls_back_in_rollback_mode() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(kilop_fs::WorkspaceFileService::new());
        let original = b"fn main() {\n    let x = 1;\n    println!(\"{}\", x);\n}\n";
        fs::write(h.root().join("main.rs"), original).unwrap();
        // Break the syntax: delete the closing brace line.
        let r = req("main.rs", original, vec![EditOp::SearchReplace {
            before: "}\n".into(),
            after: "".into(),
        }]);
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
        let engine = EditEngine::new(kilop_fs::WorkspaceFileService::new());
        let original = b"fn main() {\n    let x = 1;\n}\n";
        fs::write(h.root().join("main.rs"), original).unwrap();
        let r = req("main.rs", original, vec![EditOp::SearchReplace {
            before: "}\n".into(),
            after: "".into(),
        }]);
        let out = engine.apply(&h, &id, &r, RepairMode::AllowModelRepair).unwrap();
        assert!(out.suspicious);
        assert!(out.parse_error.is_some());
        assert!(fs::read(h.root().join("main.rs")).unwrap() != original);
    }

    #[test]
    fn valid_edit_stays_not_suspicious() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(kilop_fs::WorkspaceFileService::new());
        let original = b"fn main() {\n    let x = 1;\n    println!(\"{}\", x);\n}\n";
        fs::write(h.root().join("main.rs"), original).unwrap();
        let r = req("main.rs", original, vec![EditOp::SearchReplace {
            before: "let x = 1;".into(),
            after: "let x = 2;".into(),
        }]);
        let out = engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap();
        assert!(!out.suspicious);
        assert!(out.parse_error.is_none());
    }

    #[test]
    fn python_parse_check_works() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(kilop_fs::WorkspaceFileService::new());
        let original = b"def f():\n    return 1\n";
        fs::write(h.root().join("f.py"), original).unwrap();
        // Valid edit → not suspicious.
        let r = req("f.py", original, vec![EditOp::SearchReplace {
            before: "return 1".into(),
            after: "return 2".into(),
        }]);
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
        let engine = EditEngine::new(kilop_fs::WorkspaceFileService::new());
        let original = b"<div><span></div>"; // broken HTML
        fs::write(h.root().join("x.html"), original).unwrap();
        let r = req("x.html", original, vec![EditOp::SearchReplace {
            before: "<div>".into(),
            after: "<p>".into(),
        }]);
        let out = engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap();
        assert!(!out.suspicious, "unsupported language skips the check");
    }

    #[test]
    fn partial_failure_no_partial_write() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(kilop_fs::WorkspaceFileService::new());
        let original = b"one two three";
        fs::write(h.root().join("a.txt"), original).unwrap();
        // Op 1 valid, op 2 broken: NOTHING may be written.
        let r = req("a.txt", original, vec![
            EditOp::SearchReplace {
                before: "one".into(),
                after: "ONE".into(),
            },
            EditOp::SearchReplace {
                before: "zzz".into(),
                after: "x".into(),
            },
        ]);
        let err = engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap_err();
        assert!(err.kind == ErrorKind::Malformed);
        assert!(err.message.contains("op 2"));
        assert_eq!(fs::read(h.root().join("a.txt")).unwrap(), original);
    }

    #[test]
    fn huge_edit_bounded() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(kilop_fs::WorkspaceFileService::new());
        // A 20MB file exceeds the bound → Oversized, never OOM.
        let big = vec![b'x'; 20 * 1024 * 1024];
        fs::write(h.root().join("big.txt"), &big).unwrap();
        let r = req("big.txt", &big, vec![EditOp::SearchReplace {
            before: "x".into(),
            after: "y".into(),
        }]);
        let err = engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap_err();
        assert!(err.kind == ErrorKind::Oversized);
    }

    #[tokio::test]
    async fn concurrent_edits_same_file_one_wins() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(kilop_fs::WorkspaceFileService::new());
        let original = b"start\n";
        fs::write(h.root().join("c.txt"), original).unwrap();
        let engine = std::sync::Arc::new(engine);
        let mut handles = Vec::new();
        for t in 0..6 {
            let engine = engine.clone();
            let h = h.clone();
            let id = id.clone();
            handles.push(tokio::spawn(async move {
                let r = req("c.txt", original, vec![EditOp::SearchReplace {
                    before: "start".into(),
                    after: format!("thread-{t}"),
                }]);
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
        let engine = EditEngine::new(kilop_fs::WorkspaceFileService::new());
        let original = b"fn f() {\n    let a = 1;\n    let b = 2;\n}\n";
        fs::write(h.root().join("x.rs"), original).unwrap();
        // Anchor on the fn line; replace the region after it.
        let anchor = "fn f() {";
        let r = req("x.rs", original, vec![EditOp::BoundedRegion {
            anchor: anchor.into(),
            region_start: anchor.len(),
            region_end: original.len() - 2,
            replacement: "\n    let z = 9;\n".into(),
        }]);
        let out = engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap();
        assert!(!out.suspicious);
        let text = String::from_utf8(fs::read(h.root().join("x.rs")).unwrap()).unwrap();
        assert!(text.contains("let z = 9;"));
        // Ambiguous anchor → conflict.
        let original2 = b"let a = 1;\nlet a = 2;\n";
        fs::write(h.root().join("y.txt"), original2).unwrap();
        let r = req("y.txt", original2, vec![EditOp::BoundedRegion {
            anchor: "let a".into(),
            region_start: 0,
            region_end: 4,
            replacement: "x".into(),
        }]);
        let err = engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap_err();
        assert!(err.kind == ErrorKind::Conflict);
    }

    #[test]
    fn already_broken_file_not_blamed() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(kilop_fs::WorkspaceFileService::new());
        let broken = b"fn main() { let x = ; }\n";
        fs::write(h.root().join("b.rs"), broken).unwrap();
        // The file is already broken; the edit cannot be flagged for making
        // it broken (before-parse failed).
        let r = req("b.rs", broken, vec![EditOp::SearchReplace {
            before: "fn main".into(),
            after: "fn entry".into(),
        }]);
        let out = engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap();
        assert!(!out.suspicious);
    }

    #[test]
    fn non_utf8_file_rejected() {
        let (_d, _s, h, id) = fixture();
        let engine = EditEngine::new(kilop_fs::WorkspaceFileService::new());
        let bytes = vec![0xFF, 0xFE, 0x00, 0x80];
        fs::write(h.root().join("bin.dat"), &bytes).unwrap();
        let r = req("bin.dat", &bytes, vec![EditOp::SearchReplace {
            before: "x".into(),
            after: "y".into(),
        }]);
        let err = engine.apply(&h, &id, &r, RepairMode::Rollback).unwrap_err();
        assert!(err.kind == ErrorKind::Malformed);
    }
}
