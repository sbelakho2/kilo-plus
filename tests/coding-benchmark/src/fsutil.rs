//! Filesystem primitives for workspace copies and immutability snapshots.
//!
//! The checked-in corpus is NEVER written by the harness: every run copies
//! the task directory into a fresh temp workspace first. Copies reject
//! symlinks (a checked-in symlink pointing outside the task would escape
//! the workspace) and never follow anything.

use std::path::{Path, PathBuf};

use crate::corpus::CorpusError;

/// Recursively copy `src` into `dst` (which must not exist yet). Regular
/// files and directories only; symlinks and special files are loud errors.
/// Returns the number of files copied.
pub fn copy_tree(src: &Path, dst: &Path) -> Result<u64, CorpusError> {
    let meta = std::fs::symlink_metadata(src).map_err(|e| CorpusError::Io {
        path: src.display().to_string(),
        detail: e.to_string(),
    })?;
    let kind = meta.file_type();
    if kind.is_symlink() {
        return Err(CorpusError::Symlink {
            path: src.display().to_string(),
        });
    }
    if kind.is_dir() {
        std::fs::create_dir_all(dst).map_err(|e| CorpusError::Io {
            path: dst.display().to_string(),
            detail: e.to_string(),
        })?;
        let mut count = 0u64;
        let mut names: Vec<PathBuf> = std::fs::read_dir(src)
            .map_err(|e| CorpusError::Io {
                path: src.display().to_string(),
                detail: e.to_string(),
            })?
            .flatten()
            .map(|e| e.path())
            .collect();
        names.sort();
        for child in names {
            let name = child
                .file_name()
                .ok_or_else(|| CorpusError::UnexpectedEntry {
                    path: child.display().to_string(),
                })?;
            count = count.saturating_add(copy_tree(&child, &dst.join(name))?);
        }
        Ok(count)
    } else if kind.is_file() {
        std::fs::copy(src, dst).map_err(|e| CorpusError::Io {
            path: src.display().to_string(),
            detail: e.to_string(),
        })?;
        Ok(1)
    } else {
        Err(CorpusError::UnexpectedEntry {
            path: src.display().to_string(),
        })
    }
}

/// Byte snapshot of every file under `root` (relative paths, sorted,
/// symlink-free) — used by the corpus-immutability tests to prove a
/// harness run changed nothing in the checked-in corpus.
pub fn snapshot_tree(root: &Path) -> Result<Vec<(PathBuf, Vec<u8>)>, CorpusError> {
    crate::corpus::walk_files(root, 10_000, u64::MAX).map(|files| {
        files
            .into_iter()
            .map(|(abs, bytes)| {
                let rel = abs
                    .strip_prefix(root)
                    .map(|r| r.to_path_buf())
                    .unwrap_or_else(|_| abs.clone());
                (rel, bytes)
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tree(root: &Path) {
        std::fs::create_dir_all(root.join("a/b")).unwrap();
        std::fs::write(root.join("a/top.txt"), b"one").unwrap();
        std::fs::write(root.join("a/b/nested.txt"), b"two").unwrap();
    }

    #[test]
    fn copy_tree_copies_content_and_rejects_symlinks() {
        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();
        tree(src.path());
        let target = dst.path().join("copy");
        assert_eq!(copy_tree(src.path(), &target).unwrap(), 2);
        assert_eq!(std::fs::read(target.join("a/top.txt")).unwrap(), b"one");
        assert_eq!(
            std::fs::read(target.join("a/b/nested.txt")).unwrap(),
            b"two"
        );
        assert!(snapshot_tree(&target).unwrap().len() == 2);
    }

    #[cfg(unix)]
    #[test]
    fn copy_tree_refuses_a_symlink_inside_the_tree() {
        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();
        std::fs::write(src.path().join("real.txt"), b"x").unwrap();
        std::os::unix::fs::symlink(src.path().join("real.txt"), src.path().join("link.txt"))
            .unwrap();
        let err = copy_tree(src.path(), &dst.path().join("copy")).unwrap_err();
        assert!(err.to_string().contains("symlink"), "{err}");
    }

    #[test]
    fn snapshot_is_deterministic_sorted_bytes() {
        let dir = tempfile::tempdir().unwrap();
        tree(dir.path());
        let first = snapshot_tree(dir.path()).unwrap();
        // Reorder mtimes / rewrite content to the same bytes: the snapshot
        // must be identical (sorted paths, content only).
        let mut f = std::fs::File::create(dir.path().join("a/top.txt")).unwrap();
        f.write_all(b"one").unwrap();
        f.sync_all().unwrap();
        let second = snapshot_tree(dir.path()).unwrap();
        assert_eq!(first, second);
        assert_eq!(first[0].0, PathBuf::from("a/b/nested.txt"));
        assert_eq!(first[1].0, PathBuf::from("a/top.txt"));
    }
}
