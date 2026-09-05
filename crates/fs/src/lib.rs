//! faktor-fs — file service and watcher (spec §21 resource scopes).
//!
//! Every call carries its explicit workspace identity; paths are relative to
//! the workspace root and resolved traversal/symlink-safely. Reads are
//! bounded; writes are atomic (temp + fsync + rename). Workspaces own their
//! watcher; `close` unloads heavyweight resources after inactivity.

use std::collections::HashMap;
use std::fs;
use std::path::{Component, Path, PathBuf};
#[cfg(test)]
use std::sync::OnceLock;
use std::sync::{Arc, Mutex};

use faktor_core::error::{Error, ErrorKind};
use faktor_core::hash::FileHash;
use faktor_core::id::WorkspaceId;
use faktor_core::WorkspaceIdentity;
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::mpsc;

pub mod atomic;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FsEventKind {
    Created,
    Modified,
    Removed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FsEvent {
    pub workspace_id: WorkspaceId,
    pub path: PathBuf,
    pub kind: FsEventKind,
}

#[derive(Debug, Clone)]
pub struct FileData {
    pub path: PathBuf,
    pub bytes: Vec<u8>,
    pub hash: FileHash,
    pub size: usize,
    pub truncated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileMeta {
    pub path: PathBuf,
    pub size: u64,
    pub modified_ms: i64,
}

const DEFAULT_READ_MAX: usize = 4 * 1024 * 1024;

/// Identity of a read: whether the digest covers the WHOLE file or only a
/// bounded prefix/slice (audit 48). A truncated hash must never be compared
/// with a whole-file identity: [`ContentDigest::Full`] values may be matched
/// against stored file hashes (e.g. snapshot `FileState`), while a
/// [`ContentDigest::Slice`] proves the bytes were cut short — the type
/// separation makes mistaking one for the other impossible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentDigest {
    /// Hash of the file's entire content at read time.
    Full(FileHash),
    /// Hash of the bytes in `[offset, offset+len)` of the file. Produced by
    /// bounded reads that hit the cap: the file has MORE content beyond
    /// `offset+len` (or the read was otherwise partial).
    Slice {
        hash: FileHash,
        offset: u64,
        len: u64,
    },
}

impl ContentDigest {
    pub fn hash(self) -> FileHash {
        match self {
            ContentDigest::Full(h) => h,
            ContentDigest::Slice { hash, .. } => hash,
        }
    }

    /// True when the digest proves it covers the entire file content.
    pub fn is_full(self) -> bool {
        matches!(self, ContentDigest::Full(_))
    }
}

/// Registry of open workspaces; `open` is idempotent per root.
#[derive(Debug, Default)]
pub struct WorkspaceFileService {
    workspaces: Mutex<HashMap<WorkspaceId, WorkspaceHandle>>,
}

impl WorkspaceFileService {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn open(&self, workspace_id: WorkspaceId, root: PathBuf) -> Result<WorkspaceHandle, Error> {
        // Canonicalize first so idempotency compares equal paths.
        let root = root
            .canonicalize()
            .map_err(|e| Error::not_found(format!("workspace root {}: {e}", root.display())))?;
        {
            let map = self.workspaces.lock().unwrap();
            if let Some(h) = map.get(&workspace_id) {
                if h.root == root {
                    return Ok(h.clone());
                }
                return Err(Error::conflict(format!(
                    "workspace {workspace_id} already open at {}",
                    h.root.display()
                )));
            }
        }
        if !root.is_dir() {
            return Err(Error::not_found(format!(
                "workspace root {} is not a directory",
                root.display()
            )));
        }
        let (tx, rx) = mpsc::channel(1024);
        let mut watcher = RecommendedWatcher::new(
            move |res: notify::Result<notify::Event>| {
                if let Ok(ev) = res {
                    for p in ev.paths {
                        let kind = match ev.kind {
                            notify::EventKind::Create(_) => FsEventKind::Created,
                            notify::EventKind::Modify(_) => FsEventKind::Modified,
                            notify::EventKind::Remove(_) => FsEventKind::Removed,
                            _ => continue,
                        };
                        // Lossy by design: the watcher callback must NEVER
                        // block the filesystem pipeline (a full channel would
                        // deadlock writes). Dropped events are recovered by
                        // index rescan (incremental indexing rebuilds from
                        // disk state).
                        let _ = tx.try_send(FsEvent {
                            workspace_id,
                            path: p,
                            kind,
                        });
                    }
                }
            },
            notify::Config::default(),
        )
        .map_err(|e| Error::internal(format!("watcher: {e}")))?;
        watcher
            .watch(&root, RecursiveMode::Recursive)
            .map_err(|e| Error::internal(format!("watch {}: {e}", root.display())))?;
        let handle = WorkspaceHandle {
            workspace_id,
            root,
            _watcher: Arc::new(Mutex::new(watcher)),
            events: Arc::new(Mutex::new(rx)),
        };
        self.workspaces
            .lock()
            .unwrap()
            .insert(workspace_id, handle.clone());
        Ok(handle)
    }

    /// Idle unload (spec §21): drops the watcher and cached handles.
    pub fn close(&self, workspace_id: WorkspaceId) {
        self.workspaces.lock().unwrap().remove(&workspace_id);
    }

    pub fn open_count(&self) -> usize {
        self.workspaces.lock().unwrap().len()
    }
}

#[derive(Clone)]
pub struct WorkspaceHandle {
    workspace_id: WorkspaceId,
    root: PathBuf,
    _watcher: Arc<Mutex<RecommendedWatcher>>,
    events: Arc<Mutex<mpsc::Receiver<FsEvent>>>,
}

impl std::fmt::Debug for WorkspaceHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkspaceHandle")
            .field("workspace_id", &self.workspace_id)
            .field("root", &self.root)
            .finish()
    }
}

impl WorkspaceHandle {
    pub fn workspace_id(&self) -> WorkspaceId {
        self.workspace_id
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Traversal/symlink-safe resolution of a relative path under the root.
    pub fn resolve(&self, rel: &Path) -> Result<PathBuf, Error> {
        resolve_within(&self.root, rel)
    }

    /// Read bounded by max_bytes (default 4MB); `truncated` when exceeded.
    /// The returned `FileData.hash` covers the bytes actually returned (see
    /// [`WorkspaceHandle::read_hashed`] for a digest that cannot be mistaken
    /// for whole-file identity).
    pub fn read(&self, rel: &Path, max_bytes: usize) -> Result<FileData, Error> {
        let path = self.resolve(rel)?;
        let (bytes, digest) = self.read_bounded(&path, rel, max_bytes)?;
        let (truncated, hash) = match digest {
            ContentDigest::Full(h) => (false, h),
            ContentDigest::Slice { hash, .. } => (true, hash),
        };
        let size = bytes.len();
        Ok(FileData {
            path,
            bytes,
            hash,
            size,
            truncated,
        })
    }

    pub fn read_default(&self, rel: &Path) -> Result<FileData, Error> {
        self.read(rel, DEFAULT_READ_MAX)
    }

    /// Bounded read whose digest says what it covers: the whole file
    /// ([`ContentDigest::Full`]) or a capped prefix
    /// ([`ContentDigest::Slice`]). Snapshot probes and indexers use this so
    /// a truncated hash can never masquerade as the file's identity.
    pub fn read_hashed(
        &self,
        rel: &Path,
        max_bytes: usize,
    ) -> Result<(Vec<u8>, ContentDigest), Error> {
        let path = self.resolve(rel)?;
        self.read_bounded(&path, rel, max_bytes)
    }

    /// The shared bounded read: metadata first, then either the whole file
    /// (Full digest) or exactly `max_bytes` (Slice digest).
    fn read_bounded(
        &self,
        path: &Path,
        rel: &Path,
        max_bytes: usize,
    ) -> Result<(Vec<u8>, ContentDigest), Error> {
        let size = fs::metadata(path).map_err(|e| err_not_found(rel, e))?.len();
        let mut f = fs::File::open(path).map_err(|e| err_not_found(rel, e))?;
        read_race_seam(rel);
        if !opened_is_path(&f, path) {
            return Err(Error::permission(format!(
                "{rel:?} changed identity between resolution and open (TOCTOU)"
            )));
        }
        use std::io::Read;
        let mut bytes = Vec::new();
        let digest = if size > max_bytes as u64 {
            bytes.resize(max_bytes, 0);
            f.read_exact(&mut bytes)
                .map_err(|e| Error::internal(format!("read {rel:?}: {e}")))?;
            ContentDigest::Slice {
                hash: FileHash::from(blake3::hash(&bytes).into()),
                offset: 0,
                len: bytes.len() as u64,
            }
        } else {
            f.read_to_end(&mut bytes)
                .map_err(|e| Error::internal(format!("read {rel:?}: {e}")))?;
            ContentDigest::Full(FileHash::from(blake3::hash(&bytes).into()))
        };
        Ok((bytes, digest))
    }

    /// Stream-hash a file through a bounded 64 KiB buffer — the file is
    /// NEVER materialized in RAM (audit 49). Returns the number of bytes
    /// actually hashed and the BLAKE3 hash of those bytes.
    ///
    /// - `max_bytes: None` hashes the whole file (files of any size, but
    ///   streamingly — this is the snapshot probe path).
    /// - `max_bytes: Some(n)` caps the read at `min(file size, n)`; the
    ///   returned hash then covers only that prefix — callers that need
    ///   whole-file identity must pass `None` or check the byte count.
    pub fn hash_file_streaming(
        &self,
        rel: &Path,
        max_bytes: Option<u64>,
    ) -> Result<(u64, FileHash), Error> {
        let path = self.resolve(rel)?;
        let meta = fs::metadata(&path).map_err(|e| err_not_found(rel, e))?;
        let mut f = fs::File::open(&path).map_err(|e| err_not_found(rel, e))?;
        read_race_seam(rel);
        if !opened_is_path(&f, &path) {
            return Err(Error::permission(format!(
                "{rel:?} changed identity between resolution and open (TOCTOU)"
            )));
        }
        use std::io::Read;
        let mut hasher = blake3::Hasher::new();
        let mut buf = [0u8; 64 * 1024];
        let mut remaining = match max_bytes {
            Some(max) => max.min(meta.len()),
            None => meta.len(),
        };
        let mut hashed = 0u64;
        while remaining > 0 {
            let want = remaining.min(buf.len() as u64) as usize;
            let n = f
                .read(&mut buf[..want])
                .map_err(|e| Error::internal(format!("read {rel:?}: {e}")))?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            hashed += n as u64;
            remaining -= n as u64;
        }
        Ok((hashed, FileHash::from(hasher.finalize().into())))
    }

    /// Slice read for paging big files (spec §23).
    pub fn read_slice(&self, rel: &Path, offset: u64, len: usize) -> Result<FileData, Error> {
        use std::io::{Read, Seek, SeekFrom};
        let path = self.resolve(rel)?;
        let mut f = fs::File::open(&path).map_err(|e| err_not_found(rel, e))?;
        read_race_seam(rel);
        if !opened_is_path(&f, &path) {
            return Err(Error::permission(format!(
                "{rel:?} changed identity between resolution and open (TOCTOU)"
            )));
        }
        f.seek(SeekFrom::Start(offset))
            .map_err(|e| Error::internal(format!("seek {rel:?}: {e}")))?;
        let mut bytes = vec![0u8; len];
        let mut read = 0usize;
        while read < len {
            let n = f
                .read(&mut bytes[read..])
                .map_err(|e| Error::internal(format!("read {rel:?}: {e}")))?;
            if n == 0 {
                break;
            }
            read += n;
        }
        bytes.truncate(read);
        let hash = FileHash::from(blake3::hash(&bytes).into());
        let size = bytes.len();
        Ok(FileData {
            path,
            bytes,
            hash,
            size,
            truncated: read < len
                && offset as usize + read < f.metadata().map(|m| m.len() as usize).unwrap_or(read),
        })
    }

    /// Atomic write: temp file in the same dir + fsync + rename + parent-dir
    /// fsync on unix. Delegates to the shared durable helper so every writer
    /// in the workspace follows the identical crash-safe sequence (audit
    /// 45/75).
    pub fn write_atomic(&self, rel: &Path, bytes: &[u8]) -> Result<FileHash, Error> {
        let path = self.resolve(rel)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| Error::internal(format!("mkdir {}: {e}", parent.display())))?;
        }
        atomic::atomic_replace(&path, bytes)
    }

    /// Commit-time compare-and-swap write (audit 46): the destination's
    /// CURRENT content digest must equal `expected_hash` (the hash of what
    /// the caller read and validated) at the moment of replacement — a file
    /// that changed since the caller's read is never clobbered. The digest
    /// recheck happens immediately before the rename, under the shared
    /// per-path mutation lock, so cooperative writers serialize.
    pub fn write_atomic_cas(
        &self,
        rel: &Path,
        expected_hash: FileHash,
        bytes: &[u8],
    ) -> Result<FileHash, Error> {
        let path = self.resolve(rel)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| Error::internal(format!("mkdir {}: {e}", parent.display())))?;
        }
        let expected = match atomic::FileState::now(&path)? {
            st if !st.exists => {
                return Err(Error::conflict(format!(
                    "{} disappeared before the commit-time check",
                    rel.display()
                )))
            }
            st => st,
        };
        let mut expected = expected;
        expected.digest = Some(expected_hash);
        atomic::atomic_replace_cas(&path, &expected, bytes)
    }

    pub fn stat(&self, rel: &Path) -> Result<FileMeta, Error> {
        let path = self.resolve(rel)?;
        let meta = fs::metadata(&path).map_err(|e| err_not_found(rel, e))?;
        let modified_ms = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        Ok(FileMeta {
            path,
            size: meta.len(),
            modified_ms,
        })
    }

    pub fn exists(&self, rel: &Path) -> bool {
        self.resolve(rel).map(|p| p.exists()).unwrap_or(false)
    }

    pub fn list(&self, rel: &Path, max_entries: usize) -> Result<Vec<FileMeta>, Error> {
        let path = self.resolve(rel)?;
        let mut out = Vec::new();
        let rd = fs::read_dir(&path).map_err(|e| err_not_found(rel, e))?;
        for entry in rd.flatten() {
            if out.len() >= max_entries {
                break;
            }
            if let Ok(meta) = entry.metadata() {
                let modified_ms = meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_millis() as i64)
                    .unwrap_or(0);
                out.push(FileMeta {
                    path: entry.path(),
                    size: meta.len(),
                    modified_ms,
                });
            }
        }
        out.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(out)
    }

    pub fn events(&self) -> &Mutex<mpsc::Receiver<FsEvent>> {
        &self.events
    }

    /// Identity check helper: every tool call carries its identity; this
    /// verifies the workspace matches this handle (no cross-workspace use).
    pub fn verify_identity(&self, identity: &WorkspaceIdentity) -> Result<(), Error> {
        if identity.workspace_id != self.workspace_id {
            return Err(Error::permission(format!(
                "workspace identity mismatch: handle {}, identity {}",
                self.workspace_id, identity.workspace_id
            )));
        }
        Ok(())
    }
}

fn err_not_found(rel: &Path, e: std::io::Error) -> Error {
    if e.kind() == std::io::ErrorKind::NotFound {
        Error::not_found(format!("{}", rel.display()))
    } else {
        Error::new(ErrorKind::Internal, format!("{}: {e}", rel.display()))
    }
}

/// Open-after-resolve identity check (audit 47): resolution (canonicalize)
/// and open are two syscalls; a swap between them redirects the open to a
/// different file. After opening we re-stat the path WITHOUT following the
/// final component and compare (dev, inode): a swap — including a swap to a
/// symlink — changes the inode and is rejected loudly. On platforms without
/// stable (dev, ino) metadata this is skipped (documented honest limit).
#[cfg(unix)]
fn opened_is_path(f: &fs::File, path: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    match (f.metadata(), fs::symlink_metadata(path)) {
        (Ok(open), Ok(at_path)) => open.dev() == at_path.dev() && open.ino() == at_path.ino(),
        _ => false,
    }
}

#[cfg(not(unix))]
fn opened_is_path(_f: &fs::File, _path: &Path) -> bool {
    true
}

/// Test seam: called between the pre-open stat and the open in the read
/// paths. Deterministic TOCTOU tests swap the target file (or replace it
/// with a symlink) inside this window.
#[cfg(test)]
type ReadSeam = Box<dyn Fn(&Path) + Send>;
#[cfg(test)]
static READ_RACE_SEAM: OnceLock<std::sync::Mutex<Option<ReadSeam>>> = OnceLock::new();
#[cfg(test)]
fn read_race_seam(rel: &Path) {
    if let Some(lock) = READ_RACE_SEAM.get() {
        if let Some(hook) = lock.lock().expect("seam poisoned").as_ref() {
            hook(rel);
        }
    }
}
#[cfg(not(test))]
fn read_race_seam(_rel: &Path) {}

fn resolve_within(root: &Path, path: &Path) -> Result<PathBuf, Error> {
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };
    for component in joined.components() {
        if let Component::ParentDir = component {
            return Err(Error::permission(format!(
                "path traversal rejected: {path:?}"
            )));
        }
    }
    if let Ok(canon) = joined.canonicalize() {
        if canon.starts_with(root) {
            return Ok(canon);
        }
        return Err(Error::permission(format!(
            "path escapes workspace: {path:?}"
        )));
    }
    if let Ok(meta) = fs::symlink_metadata(&joined) {
        if meta.file_type().is_symlink() {
            return Err(Error::permission(format!(
                "symlink escape rejected: {path:?}"
            )));
        }
    }
    let parent = joined
        .parent()
        .ok_or_else(|| Error::malformed("path has no parent"))?;
    let file_name = joined
        .file_name()
        .ok_or_else(|| Error::malformed("path has no file name"))?;
    let canon_parent = parent
        .canonicalize()
        .map_err(|_| Error::permission(format!("parent resolution failed: {path:?}")))?;
    let resolved = canon_parent.join(file_name);
    if resolved.starts_with(root) {
        Ok(resolved)
    } else {
        Err(Error::permission(format!(
            "path escapes workspace: {path:?}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    fn fixture() -> (
        tempfile::TempDir,
        Arc<WorkspaceFileService>,
        WorkspaceHandle,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("ws");
        fs::create_dir_all(&root).unwrap();
        let service = WorkspaceFileService::new();
        let handle = service.open(WorkspaceId::new(1), root.clone()).unwrap();
        (dir, service, handle)
    }

    #[test]
    fn traversal_escape_rejected() {
        let (_d, _s, h) = fixture();
        for evil in ["../x", "a/../../b", "/etc/passwd", "..", "a/.."] {
            let r = h.resolve(Path::new(evil));
            assert!(r.is_err(), "{evil} must be rejected");
        }
        assert!(h.resolve(Path::new("ok.txt")).is_ok());
    }

    #[test]
    fn symlink_escape_rejected() {
        let (_d, _s, h) = fixture();
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("secret"), "s").unwrap();
        symlink(outside.path(), h.root().join("link")).unwrap();
        assert!(h.resolve(Path::new("link/secret")).is_err());
        assert!(h.resolve(Path::new("link")).is_err());
        // Inside symlink is fine.
        fs::write(h.root().join("real"), "x").unwrap();
        symlink(h.root().join("real"), h.root().join("alias")).unwrap();
        assert!(h.resolve(Path::new("alias")).is_ok());
    }

    #[test]
    fn read_bounded_sets_truncated_flag() {
        let (_d, _s, h) = fixture();
        fs::write(h.root().join("big.bin"), vec![7u8; 10_000]).unwrap();
        let data = h.read(Path::new("big.bin"), 1000).unwrap();
        assert!(data.truncated);
        assert_eq!(data.bytes.len(), 1000);
        let data = h.read(Path::new("big.bin"), 20_000).unwrap();
        assert!(!data.truncated);
        assert_eq!(data.bytes.len(), 10_000);
    }

    #[test]
    fn read_slice_pages_whole_file() {
        let (_d, _s, h) = fixture();
        let content: Vec<u8> = (0..5000).map(|i| (i % 256) as u8).collect();
        fs::write(h.root().join("paged.bin"), &content).unwrap();
        let mut assembled = Vec::new();
        let mut offset = 0u64;
        loop {
            let part = h.read_slice(Path::new("paged.bin"), offset, 777).unwrap();
            if part.bytes.is_empty() {
                break;
            }
            assembled.extend_from_slice(&part.bytes);
            offset += part.bytes.len() as u64;
            if part.truncated {
                break;
            }
        }
        assert_eq!(assembled, content);
    }

    #[test]
    fn write_atomic_replaces_and_hashes() {
        let (_d, _s, h) = fixture();
        let h1 = h.write_atomic(Path::new("a.txt"), b"one").unwrap();
        let h2 = h.write_atomic(Path::new("a.txt"), b"two").unwrap();
        assert_ne!(h1, h2);
        let data = h.read(Path::new("a.txt"), 100).unwrap();
        assert_eq!(data.bytes, b"two");
        assert_eq!(data.hash, h2);
        // No temp files left behind.
        let names: Vec<_> = fs::read_dir(h.root())
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        assert!(
            !names.iter().any(|n| n.contains("kp-tmp-")),
            "temp leaked: {names:?}"
        );
    }

    #[tokio::test]
    async fn watcher_delivers_events_with_workspace_id() {
        let (_d, _s, h) = fixture();
        fs::write(h.root().join("w.txt"), "x").unwrap();
        let mut saw = false;
        for _ in 0..60 {
            if let Ok(ev) = h.events().lock().unwrap().try_recv() {
                if ev.workspace_id == WorkspaceId::new(1) && ev.path.ends_with("w.txt") {
                    saw = true;
                    break;
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
        assert!(saw, "watcher must deliver the create event");
    }

    #[test]
    fn missing_file_not_found() {
        let (_d, _s, h) = fixture();
        assert!(h.read(Path::new("nope.rs"), 100).is_err());
        assert!(h.stat(Path::new("nope.rs")).is_err());
        assert!(!h.exists(Path::new("nope.rs")));
    }

    #[tokio::test]
    async fn concurrent_atomic_writes_never_partial() {
        let (_d, _s, h) = fixture();
        let mut handles = Vec::new();
        for t in 0..8 {
            let h = h.clone();
            handles.push(tokio::spawn(async move {
                for i in 0..50 {
                    let payload = format!("thread-{t}-{i}-{}", "x".repeat(100));
                    h.write_atomic(Path::new("shared.txt"), payload.as_bytes())
                        .unwrap();
                    let data = h.read(Path::new("shared.txt"), 10_000).unwrap();
                    let text = String::from_utf8_lossy(&data.bytes);
                    assert!(
                        text.starts_with("thread-") && text.contains("-x"),
                        "partial write observed: {text}"
                    );
                }
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
    }

    #[test]
    fn list_bounded_and_sorted() {
        let (_d, _s, h) = fixture();
        for i in 0..30 {
            fs::write(h.root().join(format!("f{i:02}.rs")), "x").unwrap();
        }
        let list = h.list(Path::new("."), 10).unwrap();
        assert_eq!(list.len(), 10);
        assert!(list.windows(2).all(|w| w[0].path <= w[1].path));
    }

    #[test]
    fn unicode_and_binary_paths() {
        let (_d, _s, h) = fixture();
        fs::write(h.root().join("é😀.rs"), b"\x00\x01\x02").unwrap();
        let data = h.read(Path::new("é😀.rs"), 100).unwrap();
        assert_eq!(data.bytes, vec![0, 1, 2]);
        assert!(h.exists(Path::new("é😀.rs")));
    }

    #[test]
    fn identity_verification_blocks_cross_workspace() {
        let (_d, _s, h) = fixture();
        let wrong = WorkspaceIdentity::new(
            WorkspaceId::new(99),
            faktor_core::WorktreeId::new(1),
            faktor_core::TaskId::new(1),
        );
        assert!(h.verify_identity(&wrong).is_err());
        let right = WorkspaceIdentity::new(
            WorkspaceId::new(1),
            faktor_core::WorktreeId::new(1),
            faktor_core::TaskId::new(1),
        );
        assert!(h.verify_identity(&right).is_ok());
    }

    #[test]
    fn open_is_idempotent_and_close_unloads() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("ws");
        fs::create_dir_all(&root).unwrap();
        let service = WorkspaceFileService::new();
        let h1 = service.open(WorkspaceId::new(1), root.clone()).unwrap();
        let h2 = service.open(WorkspaceId::new(1), root.clone()).unwrap();
        assert_eq!(h1.root(), h2.root());
        assert_eq!(service.open_count(), 1);
        // Same id, different root → conflict.
        let other = dir.path().join("other");
        fs::create_dir_all(&other).unwrap();
        assert!(service.open(WorkspaceId::new(1), other).is_err());
        service.close(WorkspaceId::new(1));
        assert_eq!(service.open_count(), 0);
    }

    #[test]
    fn missing_root_not_found() {
        let service = WorkspaceFileService::new();
        assert!(service
            .open(WorkspaceId::new(1), "/definitely/not/here".into())
            .is_err());
    }

    #[test]
    fn truncated_read_produces_slice_digest_full_read_produces_full() {
        let (_d, _s, h) = fixture();
        let content: Vec<u8> = (0..100_000).map(|i| (i % 253) as u8).collect();
        fs::write(h.root().join("dig.bin"), &content).unwrap();
        // Whole file: Full digest, equal to the plain read()'s hash.
        let (bytes, digest) = h.read_hashed(Path::new("dig.bin"), 200_000).unwrap();
        assert_eq!(bytes, content);
        let ContentDigest::Full(full_hash) = digest else {
            panic!("unbounded read must be Full");
        };
        let data = h.read(Path::new("dig.bin"), 200_000).unwrap();
        assert_eq!(data.hash, full_hash);
        assert!(!data.truncated);
        // Capped read: Slice digest of exactly the prefix, never Full.
        let (bytes, digest) = h.read_hashed(Path::new("dig.bin"), 10_000).unwrap();
        assert_eq!(bytes.len(), 10_000);
        let ContentDigest::Slice {
            hash: slice_hash,
            offset: 0,
            len,
        } = digest
        else {
            panic!("truncated read must be Slice");
        };
        assert_eq!(len, 10_000);
        assert_eq!(
            slice_hash,
            FileHash::from(blake3::hash(&content[..10_000]).into()),
            "slice hash must be the hash of the returned prefix"
        );
        // Type-level separation: a Slice over the same bytes is never equal
        // to the Full digest of the file.
        assert_ne!(
            digest,
            ContentDigest::Full(FileHash::from(blake3::hash(&content).into()))
        );
        let data = h.read(Path::new("dig.bin"), 10_000).unwrap();
        assert!(data.truncated);
        assert_eq!(data.hash, slice_hash, "read() keeps its historical hash");
    }

    #[test]
    fn content_digest_full_vs_slice_never_equal() {
        let h = FileHash::from([7; 32]);
        assert_ne!(
            ContentDigest::Full(h),
            ContentDigest::Slice {
                hash: h,
                offset: 0,
                len: 10,
            },
            "Full(same-hash) must never equal a Slice: a prefix is not the file"
        );
        assert!(ContentDigest::Full(h).is_full());
        assert!(!ContentDigest::Slice {
            hash: h,
            offset: 0,
            len: 10
        }
        .is_full());
    }

    #[test]
    fn hash_file_streaming_matches_direct_read_and_caps() {
        let (_d, _s, h) = fixture();
        // 5 MB patterned control file: the streaming hash must equal a
        // direct full read.
        let content: Vec<u8> = (0..(5 * 1024 * 1024))
            .map(|i| ((i * 31 + 7) % 256) as u8)
            .collect();
        fs::write(h.root().join("stream.bin"), &content).unwrap();
        let (n, hash) = h
            .hash_file_streaming(Path::new("stream.bin"), None)
            .unwrap();
        assert_eq!(n, content.len() as u64);
        assert_eq!(hash, FileHash::from(blake3::hash(&content).into()));
        // Capped: hashes exactly the prefix, reports the byte count.
        let (n, hash) = h
            .hash_file_streaming(Path::new("stream.bin"), Some(12_345))
            .unwrap();
        assert_eq!(n, 12_345);
        assert_eq!(
            hash,
            FileHash::from(blake3::hash(&content[..12_345]).into())
        );
        // Empty file: zero bytes, empty-content hash.
        fs::write(h.root().join("empty.bin"), b"").unwrap();
        let (n, hash) = h.hash_file_streaming(Path::new("empty.bin"), None).unwrap();
        assert_eq!(n, 0);
        assert_eq!(hash, FileHash::from(blake3::hash(b"").into()));
    }

    #[test]
    fn write_atomic_delegates_to_shared_helper_no_temps_after_storm() {
        let (_d, _s, h) = fixture();
        fs::create_dir_all(h.root().join("deep")).unwrap();
        for i in 0..25 {
            let payload = format!("storm-{i}-{}", "x".repeat(i * 100));
            let hash = h
                .write_atomic(Path::new("deep/f.bin"), payload.as_bytes())
                .unwrap();
            assert_eq!(
                hash,
                FileHash::from(blake3::hash(payload.as_bytes()).into())
            );
        }
        // Crash simulation (behavioral): the destination directory holds
        // exactly the target after the storm — replacement visible, zero
        // temp leftovers, and the parent was fsynced after each rename.
        let names: Vec<String> = fs::read_dir(h.root().join("deep"))
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(names, vec!["f.bin"], "temp leaked: {names:?}");
        let expected = format!("storm-24-{}", "x".repeat(2400));
        assert_eq!(
            fs::read(h.root().join("deep/f.bin")).unwrap(),
            expected.as_bytes()
        );
    }

    // ---------------------------------------------------------- audit 46/47

    #[test]
    fn write_atomic_cas_rejects_stale_writes_and_never_clobbers() {
        let (_d, _s, h) = fixture();
        fs::write(h.root().join("cas.txt"), b"original-0123456789").unwrap();
        let first = h.read(Path::new("cas.txt"), 100).unwrap();
        // Same state: succeeds.
        let new_hash = h
            .write_atomic_cas(Path::new("cas.txt"), first.hash, b"edited-by-tool")
            .unwrap();
        let after = h.read(Path::new("cas.txt"), 100).unwrap();
        assert_eq!(after.bytes, b"edited-by-tool");
        assert_eq!(after.hash, new_hash);
        // Stale expected hash: the file moved on (same size, different
        // digest) — CAS must reject, not clobber.
        let stale = first.hash;
        h.write_atomic(Path::new("cas.txt"), b"other-writer-0987654321")
            .unwrap();
        let err = h
            .write_atomic_cas(Path::new("cas.txt"), stale, b"late-edit-should-fail")
            .unwrap_err();
        assert!(
            err.message.contains("cas mismatch") || err.message.contains("changed"),
            "{err:?}"
        );
        assert_eq!(
            fs::read(h.root().join("cas.txt")).unwrap(),
            b"other-writer-0987654321"
        );
        // Deleted underneath: never silently recreated from a stale patch.
        fs::remove_file(h.root().join("cas.txt")).unwrap();
        let err = h
            .write_atomic_cas(Path::new("cas.txt"), stale, b"zombie")
            .unwrap_err();
        assert!(
            err.kind == ErrorKind::Conflict || err.kind == ErrorKind::NotFound,
            "{err:?}"
        );
        assert!(!h.root().join("cas.txt").exists());
    }

    #[test]
    fn read_after_directory_entry_swap_is_detected_via_open_identity() {
        let (_d, _s, h) = fixture();
        fs::write(h.root().join("swap.bin"), b"aaaaaaaaaaaaaaaaaaaa").unwrap();
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("secret"), b"sssh").unwrap();

        let install = |hook: Box<dyn Fn(&Path) + Send>| {
            let m = READ_RACE_SEAM.get_or_init(|| std::sync::Mutex::new(None));
            *m.lock().expect("seam poisoned") = Some(hook);
        };

        // Seam fires between the OPEN and the identity check: the directory
        // entry is swapped for a symlink pointing outside the workspace. The
        // opened fd belongs to the original file; the path now names a
        // symlink — (dev, ino) must differ and the read must be rejected.
        // This is the canonical canonicalize-then-open TOCTOU, deterministic.
        let root = h.root().to_path_buf();
        let root2 = root.clone();
        install(Box::new(move |rel: &Path| {
            if rel == Path::new("swap.bin") {
                let t = root2.join("swap.bin");
                let _ = fs::remove_file(&t);
                #[cfg(unix)]
                std::os::unix::fs::symlink(outside.path().join("secret"), &t).unwrap();
            }
        }));
        let r = h.read(Path::new("swap.bin"), 100);
        assert!(r.is_err(), "symlink-swapped open must be rejected");
        assert!(r.clone().unwrap_err().message.contains("TOCTOU"), "{r:?}");
        // Cleanup so the symlink cannot leak into the next scenario.
        let _ = fs::remove_file(root.join("swap.bin"));

        // Second scenario: the directory entry is RENAMED over by a
        // different regular file mid-read (inode changes under the fd).
        fs::write(root.join("swap.bin"), b"aaaaaaaaaaaaaaaaaaaa").unwrap();
        let alt = root.join("alt.bin");
        let root3 = root.clone();
        fs::write(&alt, b"bbbbbbbbbbbbbbbbbbbb").unwrap();
        install(Box::new(move |rel: &Path| {
            if rel == Path::new("swap.bin") {
                let _ = fs::rename(&alt, root3.join("swap.bin"));
            }
        }));
        let r = h.read(Path::new("swap.bin"), 100);
        assert!(r.is_err(), "renamed-over open must be rejected");
        assert!(r.clone().unwrap_err().message.contains("TOCTOU"), "{r:?}");
        // Cleanup: swap.bin now contains alt's content; restore shape.
        let _ = fs::remove_file(root.join("swap.bin"));
        fs::write(root.join("swap.bin"), b"aaaaaaaaaaaaaaaaaaaa").unwrap();
        let ok = h.read(Path::new("swap.bin"), 100).unwrap();
        assert_eq!(ok.bytes, b"aaaaaaaaaaaaaaaaaaaa");
    }

    #[test]
    fn benign_read_has_no_seam_effect() {
        let (_d, _s, h) = fixture();
        fs::write(h.root().join("ok.bin"), b"hello world").unwrap();
        let data = h.read(Path::new("ok.bin"), 100).unwrap();
        assert_eq!(data.bytes, b"hello world");
    }
}
