//! Harness 4: workspace path normalization / traversal-safe resolution.
//!
//! `faktor-fs::WorkspaceHandle::resolve` normalizes a relative path under
//! a workspace root with traversal/symlink discipline. Adversarial
//! properties for arbitrary path text: never panic; a successful
//! resolution ALWAYS lands inside the workspace root (canonical), never
//! outside; absolute paths and `..`/symlink escapes are typed denials,
//! never silently accepted.
//!
//! The workspace (and its notify watcher) is created ONCE per process and
//! kept for the process lifetime — the harness is otherwise pure.

use std::path::Path;
use std::sync::OnceLock;

use faktor_core::id::WorkspaceId;
use faktor_fs::{WorkspaceFileService, WorkspaceHandle};

use super::Outcome;

struct Kit {
    _dir: &'static tempfile::TempDir,
    handle: WorkspaceHandle,
}

fn kit() -> &'static Kit {
    static KIT: OnceLock<Kit> = OnceLock::new();
    KIT.get_or_init(|| {
        let dir = Box::leak(Box::new(tempfile::tempdir().expect("tempdir")));
        let root = dir.path().join("ws");
        std::fs::create_dir_all(&root).unwrap();
        // A few real files so canonicalization paths are exercised.
        std::fs::write(root.join("real.txt"), b"x").unwrap();
        std::fs::create_dir_all(root.join("sub")).unwrap();
        std::fs::write(root.join("sub").join("deep.txt"), b"y").unwrap();
        let service = WorkspaceFileService::new();
        let handle = service.open(WorkspaceId::new(9), root).unwrap();
        Kit { _dir: dir, handle }
    })
}

/// Fuzz workspace-relative path normalization.
pub fn harness_path_normalization(bytes: &[u8]) -> Outcome {
    let kit = kit();
    if bytes.is_empty() {
        return Outcome::Clean;
    }
    if bytes.contains(&0) {
        // NUL cannot exist in a path: exercised, must stay typed.
        return Outcome::NotApplicable;
    }
    let Ok(text) = std::str::from_utf8(bytes) else {
        return Outcome::NotApplicable;
    };
    let rel = Path::new(text);
    match kit.handle.resolve(rel) {
        Ok(absolute) => {
            let canon_root = kit.handle.root();
            if !absolute.starts_with(canon_root) {
                return Outcome::Violation(format!(
                    "resolve accepted {text:?} and landed OUTSIDE the workspace root at {absolute:?}"
                ));
            }
        }
        Err(_) => {
            // Typed denial (traversal, escape, symlink, malformed). The
            // error type is not part of the harness comparison scope.
        }
    }
    Outcome::Clean
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Lcg;

    #[test]
    fn seeded_pseudo_fuzz_path_normalization_2000() {
        let mut lcg = Lcg::new(0x5EED_0000_0000_0004);
        let mut clean = 0;
        for i in 0..2000u64 {
            let mut bytes = Vec::new();
            match i % 6 {
                0 => {
                    // Traversal attempts.
                    let depth = 1 + lcg.below(12);
                    for _ in 0..depth {
                        bytes.extend_from_slice(b"../");
                    }
                    bytes.extend_from_slice(format!("x{}", i).as_bytes());
                }
                1 => {
                    // Absolute hostile paths.
                    let s = format!("/etc/passwd{}", i % 3);
                    bytes.extend_from_slice(s.as_bytes());
                }
                2 => {
                    // Deep relative nesting.
                    let n = 1 + lcg.below(30) as usize;
                    for _ in 0..n {
                        bytes.extend_from_slice(b"a/");
                    }
                    bytes.push(b'f');
                }
                3 => {
                    // Raw bytes.
                    let n = 1 + lcg.below(300) as usize;
                    for _ in 0..n {
                        bytes.push(lcg.next_u64() as u8);
                    }
                }
                4 => {
                    // Real-ish relative files.
                    let s = if lcg.chance(50) {
                        "sub/deep.txt".to_string()
                    } else {
                        "real.txt".to_string()
                    };
                    bytes.extend_from_slice(s.as_bytes());
                }
                _ => {
                    // Unicode + dots.
                    let s = format!(".{}/δ/..", "λ".repeat(1 + lcg.below(6) as usize));
                    bytes.extend_from_slice(s.as_bytes());
                }
            }
            match harness_path_normalization(&bytes) {
                Outcome::Clean => clean += 1,
                Outcome::NotApplicable => {}
                v => panic!("iteration {i}: {v}"),
            }
        }
        assert!(clean > 0);
    }

    /// A successfully resolved path is always under the root (spot check on
    /// the harness invariant itself).
    #[test]
    fn resolve_never_escapes_for_known_good_paths() {
        let k = kit();
        for p in ["real.txt", "sub/deep.txt", "./real.txt"] {
            let out = k.handle.resolve(Path::new(p)).expect("good path resolves");
            assert!(out.starts_with(k.handle.root()), "{p} escaped");
        }
    }
}
