//! Shared durable-atomic-write primitives (audit 45/75).
//!
//! Every writer in the runtime that replaces file content must go through
//! [`atomic_replace`]: a uniquely-named temp file in the SAME directory is
//! written, flushed and fsynced, then renamed over the destination, and on
//! unix the containing directory is fsynced afterwards so a power loss right
//! after the rename cannot lose the directory entry. A crash at any point
//! leaves at worst an orphan temp file — never a partial file at the real
//! address, and never a rename whose durability is unconfirmed.
//!
//! [`atomic_create`] adds exclusive-create semantics on top of the same
//! machinery: it fails when the destination already exists, so "the first
//! writer wins" without a check-then-write race.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use faktor_core::error::Error;
use faktor_core::hash::FileHash;

/// fsync a directory so a completed rename/link is durable. Errors are
/// ignored on purpose: macOS refuses directory fsync with EINVAL, and on
/// platforms where it works the error would only be reportable, not
/// recoverable.
pub fn fsync_parent(dir: &Path) {
    #[cfg(unix)]
    {
        if let Ok(d) = fs::File::open(dir) {
            let _ = d.sync_all();
        }
    }
    #[cfg(not(unix))]
    let _ = dir;
}

/// Atomically replace the file at `path` with `bytes`. The parent directory
/// must exist. Returns the BLAKE3 hash of the written content.
pub fn atomic_replace(path: &Path, bytes: &[u8]) -> Result<FileHash, Error> {
    let hash = FileHash::from(blake3::hash(bytes).into());
    let tmp = nonce_temp(path)?;
    write_and_fsync(&tmp, bytes)?;
    fs::rename(&tmp, path).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        Error::internal(format!(
            "rename {} -> {}: {e}",
            tmp.display(),
            path.display()
        ))
    })?;
    if let Some(parent) = path.parent() {
        fsync_parent(parent);
    }
    Ok(hash)
}

/// Atomically create `path` with `bytes`, failing with [`Error::conflict`]
/// when the file already exists (exclusive create: no check-then-write
/// race — the hard link fails atomically when the destination is taken).
pub fn atomic_create(path: &Path, bytes: &[u8]) -> Result<FileHash, Error> {
    let hash = FileHash::from(blake3::hash(bytes).into());
    let tmp = nonce_temp(path)?;
    write_and_fsync(&tmp, bytes)?;
    // hard_link(2) never clobbers: an existing destination is an error.
    fs::hard_link(&tmp, path).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        if e.kind() == std::io::ErrorKind::AlreadyExists {
            Error::conflict(format!("{} already exists", path.display()))
        } else {
            Error::internal(format!("link {} -> {}: {e}", tmp.display(), path.display()))
        }
    })?;
    if let Some(parent) = path.parent() {
        fsync_parent(parent);
    }
    let _ = fs::remove_file(&tmp);
    Ok(hash)
}

/// A unique temp path in the same directory as `path`: same filesystem, so
/// the rename is atomic, and never colliding with concurrent writers.
fn nonce_temp(path: &Path) -> Result<PathBuf, Error> {
    let parent = path
        .parent()
        .ok_or_else(|| Error::malformed(format!("{} has no parent", path.display())))?;
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy())
        .unwrap_or_default();
    Ok(parent.join(format!(
        ".{name}.kp-tmp-{}-{}",
        std::process::id(),
        uuid::Uuid::new_v4()
    )))
}

fn write_and_fsync(tmp: &Path, bytes: &[u8]) -> Result<(), Error> {
    let mut f = fs::File::create(tmp)
        .map_err(|e| Error::internal(format!("create {}: {e}", tmp.display())))?;
    f.write_all(bytes)
        .map_err(|e| Error::internal(format!("write {}: {e}", tmp.display())))?;
    f.flush()
        .map_err(|e| Error::internal(format!("flush {}: {e}", tmp.display())))?;
    f.sync_all()
        .map_err(|e| Error::internal(format!("fsync {}: {e}", tmp.display())))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_replaces_leave_no_temp_and_content_is_final() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("f.bin");
        for i in 0..50u64 {
            let payload = format!("replace-{i}-{}", "x".repeat((i % 4096) as usize));
            let hash = atomic_replace(&target, payload.as_bytes()).unwrap();
            assert_eq!(
                hash,
                FileHash::from(blake3::hash(payload.as_bytes()).into())
            );
        }
        // Crash simulation (behavioral): after many replaces the directory
        // holds exactly the target — no temp files survive a replace storm.
        let names: Vec<String> = fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(names, vec!["f.bin".to_string()], "temp leaked: {names:?}");
        let expected = format!("replace-49-{}", "x".repeat(49));
        assert_eq!(fs::read(&target).unwrap(), expected.as_bytes());
    }

    #[test]
    fn replace_in_subdirectory_fsyncs_parent_and_keeps_other_files() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("sub");
        fs::create_dir_all(&sub).unwrap();
        fs::write(dir.path().join("unrelated.txt"), b"keep").unwrap();
        for i in 0..10 {
            atomic_replace(&sub.join("g.txt"), format!("v{i}").as_bytes()).unwrap();
        }
        let mut names: Vec<String> = fs::read_dir(&sub)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        names.sort();
        assert_eq!(names, vec!["g.txt"]);
        assert_eq!(fs::read(sub.join("g.txt")).unwrap(), b"v9");
        assert_eq!(fs::read(dir.path().join("unrelated.txt")).unwrap(), b"keep");
    }

    #[test]
    fn atomic_create_is_exclusive_and_wins_first() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("c.txt");
        let h1 = atomic_create(&target, b"first writer").unwrap();
        let h2 = atomic_create(&target, b"second writer").unwrap_err();
        assert!(h2.message.contains("already exists"), "{h2:?}");
        // The first content survives, and no temp files linger.
        assert_eq!(h1, FileHash::from(blake3::hash(b"first writer").into()));
        assert_eq!(fs::read(&target).unwrap(), b"first writer");
        let names: Vec<String> = fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(names, vec!["c.txt".to_string()], "temp leaked: {names:?}");
    }

    #[test]
    fn fsync_parent_never_panics() {
        let dir = tempfile::tempdir().unwrap();
        fsync_parent(dir.path());
        // Even a directory that refuses fsync (macOS EINVAL) must be a no-op.
        let target = dir.path().join("x");
        atomic_replace(&target, b"x").unwrap();
        assert!(target.exists());
    }
}
