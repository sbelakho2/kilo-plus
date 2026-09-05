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

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

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

/// A point-in-time snapshot of a file's identity-relevant state (audit 46:
/// commit-time compare-and-swap). Every check is optional: `None` means the
/// axis is not part of the precondition. The digest is only populated when
/// the caller (or [`FileState::now_with_digest`]) actually hashed the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileState {
    /// Whether the file existed at capture time.
    pub exists: bool,
    /// Byte length (only meaningful when `exists`).
    pub size: Option<u64>,
    /// Content digest of the whole file (only meaningful when `exists`).
    pub digest: Option<FileHash>,
    /// Last modification ms since the UNIX epoch (best-effort; `None` when
    /// the platform cannot report it).
    pub modified_ms: Option<i64>,
}

impl FileState {
    pub fn absent() -> Self {
        Self {
            exists: false,
            size: None,
            digest: None,
            modified_ms: None,
        }
    }

    /// Capture size/mtime without hashing (cheap rejection path).
    pub fn now(path: &Path) -> Result<Self, Error> {
        match fs::symlink_metadata(path) {
            Ok(meta) if meta.is_file() => Ok(FileState {
                exists: true,
                size: Some(meta.len()),
                digest: None,
                modified_ms: meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_millis() as i64),
            }),
            Ok(_) => Ok(Self {
                exists: true,
                size: None,
                digest: None,
                modified_ms: None,
            }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::absent()),
            Err(e) => Err(Error::internal(format!("stat {}: {e}", path.display()))),
        }
    }

    /// Capture size/mtime AND the whole-content digest (streamed through a
    /// bounded buffer — never materialized). The digest is the strong check;
    /// size/mtime are the cheap pre-check.
    pub fn now_with_digest(path: &Path) -> Result<Self, Error> {
        let mut state = Self::now(path)?;
        if !state.exists {
            return Ok(state);
        }
        let f = fs::File::open(path)
            .map_err(|e| Error::internal(format!("open {}: {e}", path.display())))?;
        let mut hasher = blake3::Hasher::new();
        let mut buf = [0u8; 64 * 1024];
        let mut reader = std::io::BufReader::new(f);
        use std::io::Read;
        loop {
            let n = reader
                .read(&mut buf)
                .map_err(|e| Error::internal(format!("read {}: {e}", path.display())))?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        state.digest = Some(FileHash::from(hasher.finalize().into()));
        Ok(state)
    }

    /// Does the current state satisfy this precondition on every axis the
    /// precondition pins? Never the inverse: a precondition with no axes set
    /// matches anything (documented caller error).
    pub fn satisfied_by(&self, actual: &FileState) -> bool {
        if self.exists != actual.exists {
            return false;
        }
        if let (Some(a), Some(b)) = (self.size, actual.size) {
            if a != b {
                return false;
            }
        }
        if let (Some(a), Some(b)) = (self.digest, actual.digest) {
            if a != b {
                return false;
            }
        }
        if let (Some(a), Some(b)) = (self.modified_ms, actual.modified_ms) {
            if a != b {
                return false;
            }
        }
        true
    }
}

/// Per-path cooperative mutation locks (audit 46): every runtime writer that
/// replaces `path` holds the lock for its whole stage+check+rename sequence,
/// so two cooperative writers can never clobber each other between one
/// writer's recheck and its rename. Non-cooperative external writers cannot
/// be excluded by advisory locks — the CAS digest recheck immediately before
/// the rename shrinks that window to the rename syscall itself, which is the
/// strongest guarantee POSIX rename offers (no compare-and-swap rename).
fn path_lock(path: &Path) -> Arc<Mutex<()>> {
    static REGISTRY: OnceLock<Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>> = OnceLock::new();
    let registry = REGISTRY.get_or_init(|| Mutex::new(HashMap::new()));
    let key = path.to_path_buf();
    let mut guard = registry.lock().expect("fs path-lock registry poisoned");
    guard
        .entry(key)
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

/// Commit-time compare-and-swap replace (audit 46): like
/// [`atomic_replace`], but the destination must still match `expected`
/// immediately before the rename — a file that changed (digest, size,
/// mtime, or existence) since the caller staged its work is never
/// clobbered; the write fails with [`Error::conflict`] instead.
pub fn atomic_replace_cas(
    path: &Path,
    expected: &FileState,
    bytes: &[u8],
) -> Result<FileHash, Error> {
    let lock = path_lock(path);
    let _guard = lock.lock().expect("fs path lock poisoned");
    // Cheap rejection before staging anything.
    let quick = FileState::now(path)?;
    if !expected.satisfied_by(&quick) {
        return Err(mismatch(path, expected, &quick));
    }
    let hash = FileHash::from(blake3::hash(bytes).into());
    let tmp = nonce_temp(path)?;
    write_and_fsync(&tmp, bytes)?;
    // Strong recheck (whole-file digest) immediately before the rename.
    let actual = FileState::now_with_digest(path)?;
    if !expected.satisfied_by(&actual) {
        let _ = fs::remove_file(&tmp);
        return Err(mismatch(path, expected, &actual));
    }
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

fn mismatch(path: &Path, expected: &FileState, actual: &FileState) -> Error {
    Error::conflict(format!(
        "{} changed since it was staged (cas mismatch; expected exists={} size={:?} digest={:?}, found exists={} size={:?} digest={:?})",
        path.display(),
        expected.exists,
        expected.size,
        expected.digest.map(|d| d.to_hex()),
        actual.exists,
        actual.size,
        actual.digest.map(|d| d.to_hex())
    ))
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

    // ---------------------------------------------------------- audit 46 CAS

    #[test]
    fn cas_matches_exact_state_and_rejects_any_change() {
        let dir = tempfile::tempdir().unwrap();
        let t = dir.path().join("cas.bin");
        atomic_replace(&t, b"original-original").unwrap();
        let base = FileState::now_with_digest(&t).unwrap();
        // Same content: passes.
        atomic_replace_cas(&t, &base, b"new-content-here!!").unwrap();
        assert_eq!(fs::read(&t).unwrap(), b"new-content-here!!");
        // Stale digest: fails, content untouched, no temp left.
        let err = atomic_replace_cas(&t, &base, b"should-not-land").unwrap_err();
        assert!(err.message.contains("cas mismatch"), "{err:?}");
        assert_eq!(fs::read(&t).unwrap(), b"new-content-here!!");
        let names: Vec<String> = fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(names, vec!["cas.bin".to_string()], "temp leaked: {names:?}");
    }

    #[test]
    fn cas_detects_same_size_different_digest_and_delete_recreate() {
        let dir = tempfile::tempdir().unwrap();
        let t = dir.path().join("same.bin");
        let a: Vec<u8> = vec![b'a'; 4096];
        let b: Vec<u8> = vec![b'b'; 4096];
        atomic_replace(&t, &a).unwrap();
        let expected = FileState::now_with_digest(&t).unwrap();
        // Same size, different digest (the hard TOCTOU case).
        atomic_replace(&t, &b).unwrap();
        let err = atomic_replace_cas(&t, &expected, b"nope").unwrap_err();
        assert!(err.message.contains("cas mismatch"), "{err:?}");
        assert_eq!(fs::read(&t).unwrap(), b);
        // Delete + recreate with the ORIGINAL content: size+mtime axis differs
        // even though the digest would match — existence identity is pinned.
        fs::remove_file(&t).unwrap();
        atomic_replace(&t, &a).unwrap();
        let err = atomic_replace_cas(&t, &expected, b"nope").unwrap_err();
        assert!(err.message.contains("cas mismatch"), "{err:?}");
        assert_eq!(fs::read(&t).unwrap(), a);
    }

    #[test]
    fn cas_rejects_writes_to_a_missing_file_and_allows_absent_expectation() {
        let dir = tempfile::tempdir().unwrap();
        let t = dir.path().join("ghost.bin");
        // Expected a file, found none.
        let ghost = FileState {
            exists: true,
            size: Some(4),
            digest: None,
            modified_ms: None,
        };
        let err = atomic_replace_cas(&t, &ghost, b"x").unwrap_err();
        assert!(err.message.contains("cas mismatch"), "{err:?}");
        assert!(!t.exists());
        // Expected absence, file absent: create succeeds (first writer wins).
        let h = atomic_replace_cas(&t, &FileState::absent(), b"first").unwrap();
        assert_eq!(fs::read(&t).unwrap(), b"first");
        assert_eq!(h, FileHash::from(blake3::hash(b"first").into()));
    }

    #[test]
    fn cooperative_cas_writers_never_clobber_under_race() {
        // Two cooperative writers race the same path 200 times: the per-path
        // lock serializes stage+recheck+rename, so exactly one CAS wins each
        // round and the loser reports conflict — the file is always one of
        // the two full payloads, never a torn mix.
        let dir = tempfile::tempdir().unwrap();
        let t = dir.path().join("race.bin");
        atomic_replace(&t, vec![b'0'; 64 * 1024].as_slice()).unwrap();
        for round in 0..200u64 {
            let w1 = format!("writer-one-{round}-{}", "x".repeat(4096));
            let w2 = format!("writer-two-{round}-{}", "y".repeat(4096));
            let t1 = t.clone();
            let t2 = t.clone();
            let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
            let b1 = barrier.clone();
            let b2 = barrier.clone();
            let h1 = std::thread::spawn(move || {
                b1.wait();
                let expected = FileState::now_with_digest(&t1).unwrap();
                atomic_replace_cas(&t1, &expected, w1.as_bytes())
            });
            let h2 = std::thread::spawn(move || {
                b2.wait();
                let expected = FileState::now_with_digest(&t2).unwrap();
                atomic_replace_cas(&t2, &expected, w2.as_bytes())
            });
            barrier.wait();
            let r1 = h1.join().unwrap();
            let r2 = h2.join().unwrap();
            let wins = [&r1, &r2].iter().filter(|r| r.is_ok()).count();
            assert_eq!(
                wins, 1,
                "round {round}: both CAS must not win: {r1:?} {r2:?}"
            );
            let content = fs::read(&t).unwrap();
            let text = String::from_utf8_lossy(&content);
            assert!(
                text.starts_with("writer-one-") || text.starts_with("writer-two-"),
                "round {round}: torn content"
            );
            // Reset to a known base for the next round.
            atomic_replace(&t, vec![b'0'; 64 * 1024].as_slice()).unwrap();
        }
    }
}
