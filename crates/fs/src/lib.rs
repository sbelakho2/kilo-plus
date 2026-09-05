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

// ======================================================================
// Wave-13 controlled-merge primitives (audits 70/98/99) — additive.
//
// Bounded tree snapshot/copy + commit-time CAS merge applies over RAW
// paths (roots are canonicalized; every relative path is resolved with
// the same traversal/symlink discipline as [`WorkspaceHandle::resolve`]).
// Nothing is ever skipped silently: an un-copyable or unhashable file
// fails the whole operation loudly.
// ======================================================================

/// Hard depth bound of the bounded tree walks below (a hostile tree that
/// nests deeper fails loudly — never an unbounded recursion).
pub const MAX_TREE_WALK_DEPTH: usize = 256;
/// Per-file cap of one merge content apply. Merging streams the child's
/// file content through the wave-10 commit-time CAS helpers, which take the
/// whole payload; files beyond this bound fail with a typed [`ErrorKind::Oversized`]
/// error — the merge NEVER silently truncates a file.
pub const MAX_MERGE_FILE_BYTES: u64 = 256 * 1024 * 1024;

/// One file of a bounded tree snapshot: relative path, whole-content hash
/// and byte size. The hash always covers the WHOLE file (streamed, never
/// materialized) — a snapshot entry can never masquerade as a partial read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotEntry {
    pub path: PathBuf,
    pub hash: FileHash,
    pub size: u64,
}

/// Outcome of one CAS merge operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CasMergeResult {
    /// The file content/absence was written now.
    Applied,
    /// The destination already holds the merge's end state (an earlier,
    /// crashed run applied it — or the state converged); treated as merged.
    AlreadyCurrent,
}

/// Bounded snapshot of every file under `root` as (relative path -> hash).
/// The walk is traversal-safe (symlinks must resolve inside `root`), reads
/// are whole-file streaming (bounded RAM), never skips an unreadable file
/// and refuses trees beyond `max_entries` with a typed Oversized error.
/// Sorted by relative path (deterministic replay order).
pub fn snapshot_tree(root: &Path, max_entries: usize) -> Result<Vec<SnapshotEntry>, Error> {
    if max_entries == 0 {
        return Err(Error::malformed("snapshot max_entries must be >= 1"));
    }
    let root = root
        .canonicalize()
        .map_err(|e| Error::not_found(format!("snapshot root {}: {e}", root.display())))?;
    if !root.is_dir() {
        return Err(Error::not_found(format!(
            "snapshot root {} is not a directory",
            root.display()
        )));
    }
    let mut out = Vec::new();
    let mut count = 0usize;
    walk_files(
        &root,
        &root,
        Path::new(""),
        0,
        max_entries,
        &mut count,
        &mut |rel, abs, f| {
            let size = f
                .metadata()
                .map_err(|e| Error::internal(format!("metadata {}: {e}", abs.display())))?
                .len();
            let (_, hash) = hash_open_file(f)?;
            out.push(SnapshotEntry {
                path: rel.to_path_buf(),
                hash,
                size,
            });
            Ok(())
        },
    )?;
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
}

/// Bounded copy of every file under `src_root` into `dst_root` (which must
/// already exist as a directory, distinct from `src_root`). Each file is
/// copied with the durable atomic sequence (unique temp + fsync + rename +
/// parent fsync), so a crash leaves whole files, never torn ones. Every
/// source file is opened exactly once and read whole; an atomic-rename
/// writer on the source therefore yields either the pre- or the post-state
/// of each file — never a mix. Errors fail the WHOLE copy loudly (nothing
/// is skipped); caps (entries / total bytes) are typed Oversized errors.
/// Returns the copied manifest (relative path, hash, size), sorted.
pub fn copy_tree(
    src_root: &Path,
    dst_root: &Path,
    max_entries: usize,
    max_total_bytes: u64,
) -> Result<Vec<SnapshotEntry>, Error> {
    if max_entries == 0 || max_total_bytes == 0 {
        return Err(Error::malformed("copy caps must be >= 1"));
    }
    let src = src_root
        .canonicalize()
        .map_err(|e| Error::not_found(format!("copy source {}: {e}", src_root.display())))?;
    let dst = dst_root
        .canonicalize()
        .map_err(|e| Error::not_found(format!("copy destination {}: {e}", dst_root.display())))?;
    if !src.is_dir() || !dst.is_dir() {
        return Err(Error::not_found(
            "copy source and destination must be directories",
        ));
    }
    if src == dst {
        return Err(Error::conflict(
            "copy source and destination are the same tree",
        ));
    }
    let mut out = Vec::new();
    let mut count = 0usize;
    let mut total = 0u64;
    let out_ref = &mut out;
    let total_ref = &mut total;
    walk_files(
        &src,
        &src,
        Path::new(""),
        0,
        max_entries,
        &mut count,
        &mut |rel, abs, f| {
            let meta = f
                .metadata()
                .map_err(|e| Error::internal(format!("metadata {}: {e}", abs.display())))?;
            let size = meta.len();
            if total_ref.saturating_add(size) > max_total_bytes {
                return Err(Error::oversized(format!(
                    "copy of {abs:?} would exceed the {max_total_bytes}-byte total bound"
                )));
            }
            *total_ref += size;
            let target = dst.join(rel);
            let parent = target
                .parent()
                .ok_or_else(|| Error::malformed(format!("copy target {target:?} has no parent")))?;
            fs::create_dir_all(parent)
                .map_err(|e| Error::internal(format!("mkdir {}: {e}", parent.display())))?;
            let (n, hash) = copy_open_file(f, &target)?;
            if n != size {
                // The file changed size mid-copy (an in-place, non-atomic
                // writer): loud — never a silently torn copy.
                return Err(Error::conflict(format!(
                    "{rel:?} changed size while being copied ({n} of {size} bytes)"
                )));
            }
            out_ref.push(SnapshotEntry {
                path: rel.to_path_buf(),
                hash,
                size,
            });
            Ok(())
        },
    )?;
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
}

/// Apply ONE approved merge: overwrite `dst_root/dst_rel` with the whole
/// content of `src_root/src_rel`, guarded by commit-time CAS on BOTH sides.
///
/// - `expected_src_hash`: the staged digest of the source (the change-set
///   `child_hash`). If the source drifted since staging, the apply fails
///   with a Conflict — a changed candidate is never merged.
/// - `expected_dst`: the base-snapshot hash of the destination path
///   (`None` = the parent had no such file at the base snapshot, so the
///   destination must not exist; the write is an exclusive create).
///   If the destination moved on since the base snapshot, the apply fails
///   with a Conflict and the destination stays byte-identical — never
///   overwritten.
///
/// Replay semantics: when the destination already equals the source digest
/// (a crashed run applied it, or the state converged), the apply is
/// [`CasMergeResult::AlreadyCurrent`] — idempotent, never an error. The
/// destination write itself goes through `atomic::atomic_replace_cas` /
/// `atomic::atomic_create` (the shared per-path commit discipline), so a
/// crash at any point leaves either the old or the new whole file.
pub fn merge_apply_content(
    dst_root: &Path,
    dst_rel: &Path,
    src_root: &Path,
    src_rel: &Path,
    expected_src_hash: FileHash,
    expected_dst: Option<FileHash>,
) -> Result<CasMergeResult, Error> {
    let dst_root = dst_root.canonicalize().map_err(|e| {
        Error::not_found(format!(
            "merge destination root {}: {e}",
            dst_root.display()
        ))
    })?;
    let src_root = src_root
        .canonicalize()
        .map_err(|e| Error::not_found(format!("merge source root {}: {e}", src_root.display())))?;
    // An EXCLUSIVE create may need to materialize parent directories that
    // the parent tree never had (the child created the file inside its own
    // tree); every intermediate directory is created and verified inside
    // the canonical root — a symlink escape fails the file loudly. For an
    // existing-file apply the parent directory must already exist (its
    // disappearance is a per-file conflict, surfaced below).
    let dst = if expected_dst.is_none() {
        resolve_or_create_within(&dst_root, dst_rel)?
    } else {
        resolve_within(&dst_root, dst_rel)?
    };
    let src = resolve_within(&src_root, src_rel)?;
    // Classify the destination BEFORE reading the source (cheap reject).
    let dst_state = atomic::FileState::now_with_digest(&dst).map_err(|e| {
        Error::new(
            e.kind,
            format!("destination recheck {}: {}", dst.display(), e.message),
        )
    })?;
    match (dst_state.exists, expected_dst, dst_state.digest) {
        (false, Some(_), _) => {
            return Err(Error::conflict(format!(
                "{} vanished after the base snapshot; refusing to recreate it from a stale patch",
                dst.display()
            )));
        }
        (true, None, Some(cur)) if cur == expected_src_hash => {
            // The destination already holds exactly the child's content: a
            // crashed earlier apply (or convergent state) — idempotent.
            return Ok(CasMergeResult::AlreadyCurrent);
        }
        (true, None, _) => {
            return Err(Error::conflict(format!(
                "{} exists although the base snapshot had no such file (parent-side creation conflicts with the merge)",
                dst.display()
            )));
        }
        (true, Some(_base), Some(cur)) if cur == expected_src_hash => {
            // The destination already holds exactly the child's content: a
            // crashed earlier apply (or convergent state) — idempotent.
            return Ok(CasMergeResult::AlreadyCurrent);
        }
        (true, Some(base), Some(cur)) if cur != base => {
            return Err(Error::conflict(format!(
                "{} changed since the base snapshot (expected {}, found {}); the parent file was not touched",
                dst.display(),
                base.to_hex(),
                cur.to_hex()
            )));
        }
        (true, Some(_), None) => {
            return Err(Error::conflict(format!(
                "{} is not a readable file; refusing to overwrite it",
                dst.display()
            )));
        }
        _ => {}
    }
    // Read the source ONCE into the wave-10 CAS payload (bounded: files
    // beyond the cap fail loudly, never truncated).
    let meta = fs::metadata(&src).map_err(|e| err_not_found(src_rel, e))?;
    if meta.len() > MAX_MERGE_FILE_BYTES {
        return Err(Error::oversized(format!(
            "merge file {src_rel:?} has {} bytes (cap {MAX_MERGE_FILE_BYTES}); refusing to merge it whole",
            meta.len()
        )));
    }
    let mut f = fs::File::open(&src).map_err(|e| err_not_found(src_rel, e))?;
    if !opened_is_path(&f, &src) {
        return Err(Error::permission(format!(
            "{src_rel:?} changed identity between resolution and open (TOCTOU)"
        )));
    }
    use std::io::Read;
    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes)
        .map_err(|e| Error::internal(format!("read {src_rel:?}: {e}")))?;
    let hash = FileHash::from(blake3::hash(&bytes).into());
    if hash != expected_src_hash {
        return Err(Error::conflict(format!(
            "{} drifted since it was staged (expected {}, found {}); nothing was merged",
            src.display(),
            expected_src_hash.to_hex(),
            hash.to_hex()
        )));
    }
    match expected_dst {
        Some(base) => {
            let expected = atomic::FileState {
                exists: true,
                size: None,
                digest: Some(base),
                modified_ms: None,
            };
            atomic::atomic_replace_cas(&dst, &expected, &bytes).map_err(|e| {
                if e.kind == ErrorKind::Conflict {
                    Error::conflict(format!(
                        "{}: {}",
                        dst.display(),
                        e.message
                            .strip_prefix("cas mismatch; ")
                            .unwrap_or(&e.message)
                    ))
                } else {
                    e
                }
            })?;
        }
        None => {
            atomic::atomic_create(&dst, &bytes).map_err(|e| {
                if e.kind == ErrorKind::Conflict {
                    Error::conflict(format!(
                        "{} appeared since the base snapshot; the merge did not overwrite it",
                        dst.display()
                    ))
                } else {
                    e
                }
            })?;
        }
    }
    Ok(CasMergeResult::Applied)
}

/// Apply ONE approved DELETION: remove `dst_root/dst_rel` only when its
/// current content digest still equals the base-snapshot hash `expected`
/// (the parent must be unchanged since the base snapshot). A destination
/// that moved on fails with a Conflict and stays untouched; an already
/// absent destination is [`CasMergeResult::AlreadyCurrent`] (the merge's
/// end state — absent — already holds). POSIX offers no atomic
/// compare-and-unlink: the digest recheck happens immediately before the
/// unlink, the same recheck-rename window wave-10 documents for CAS
/// content writes.
pub fn merge_delete(
    dst_root: &Path,
    dst_rel: &Path,
    expected: FileHash,
) -> Result<CasMergeResult, Error> {
    let dst_root = dst_root.canonicalize().map_err(|e| {
        Error::not_found(format!(
            "merge destination root {}: {e}",
            dst_root.display()
        ))
    })?;
    let dst = resolve_within(&dst_root, dst_rel)?;
    let state = atomic::FileState::now_with_digest(&dst).map_err(|e| {
        Error::new(
            e.kind,
            format!("delete recheck {}: {}", dst.display(), e.message),
        )
    })?;
    if !state.exists {
        return Ok(CasMergeResult::AlreadyCurrent);
    }
    match state.digest {
        Some(cur) if cur == expected => {
            fs::remove_file(&dst)
                .map_err(|e| Error::internal(format!("remove {}: {e}", dst.display())))?;
            if let Some(parent) = dst.parent() {
                atomic::fsync_parent(parent);
            }
            Ok(CasMergeResult::Applied)
        }
        Some(cur) => Err(Error::conflict(format!(
            "{} changed since the base snapshot (expected {}, found {}); the deletion was refused",
            dst.display(),
            expected.to_hex(),
            cur.to_hex()
        ))),
        None => Err(Error::conflict(format!(
            "{} is not a readable file; the deletion was refused",
            dst.display()
        ))),
    }
}

/// Bounded recursive walk over the canonical tree `root` (already
/// canonicalized, non-symlink). Yields every FILE as (relative path,
/// absolute path, open verified handle) to `visit`. Directory symlinks are
/// followed only when their canonical target stays inside `root`; symlink
/// escapes, unreadable entries and non-file specials fail loudly. The walk
/// never silently skips anything and caps the entry count at `max_entries`.
/// Outcome of processing ONE directory entry: either fully handled or
/// confirmed-gone (it vanished mid-walk because an atomic writer renamed it
/// away — the file does not exist in the tree's current state, so skipping
/// it is honest, never a silent skip of an unreadable file).
enum EntryHandled {
    Done,
    Gone,
}

/// How many times a listed entry may race an atomic writer before the walk
/// gives up loudly (an entry that keeps vanishing/reappearing is hostile).
const WALK_ENTRY_ATTEMPTS: usize = 64;

fn walk_files<F>(
    root: &Path,
    dir: &Path,
    rel: &Path,
    depth: usize,
    max_entries: usize,
    count: &mut usize,
    visit: &mut F,
) -> Result<(), Error>
where
    F: FnMut(&Path, &Path, &fs::File) -> Result<(), Error>,
{
    if depth > MAX_TREE_WALK_DEPTH {
        return Err(Error::oversized(format!(
            "tree walk exceeded depth {MAX_TREE_WALK_DEPTH} at {rel:?}"
        )));
    }
    let rd = match fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()), // dir vanished
        Err(e) => {
            return Err(Error::internal(format!("read_dir {}: {e}", dir.display())));
        }
    };
    for entry in rd {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                return Err(Error::internal(format!("read_dir {}: {e}", dir.display())));
            }
        };
        let name = entry.file_name();
        let full = dir.join(&name);
        let rel_child = rel.join(&name);
        // Retry window: a listed entry may be renamed away (atomic writer)
        // or replaced between the listing and its open. The entry is
        // handled when it is processed while present; an entry that is
        // confirmed-absent at metadata time is skipped (it no longer is
        // part of the tree); an entry that keeps racing for
        // WALK_ENTRY_ATTEMPTS is hostile and fails loudly.
        let mut handled = EntryHandled::Gone;
        for attempt in 0..WALK_ENTRY_ATTEMPTS {
            let count_before = *count;
            match try_walk_entry(root, &full, &rel_child, depth, max_entries, count, visit) {
                Ok(h) => {
                    handled = h;
                    break;
                }
                Err(e) if attempt + 1 < WALK_ENTRY_ATTEMPTS => {
                    *count = count_before; // a raced attempt must not count
                                           // Transient race (vanished/replaced mid-open): retry;
                                           // the next attempt re-metadatas and either processes a
                                           // stable image or confirms the entry gone.
                    let _ = e;
                    continue;
                }
                Err(e) => {
                    *count = count_before;
                    return Err(e);
                }
            }
        }
        match handled {
            EntryHandled::Done => {}
            EntryHandled::Gone => {}
        }
    }
    Ok(())
}

/// Process one directory entry. Ok(Done) = fully visited;
/// Ok(Gone) = confirmed absent (the file no longer exists); Err = real
/// failure (escape, unsupported type, cap) or a transient race the caller
/// retries.
fn try_walk_entry<F>(
    root: &Path,
    full: &Path,
    rel_child: &Path,
    depth: usize,
    max_entries: usize,
    count: &mut usize,
    visit: &mut F,
) -> Result<EntryHandled, Error>
where
    F: FnMut(&Path, &Path, &fs::File) -> Result<(), Error>,
{
    let meta = match fs::symlink_metadata(full) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(EntryHandled::Gone),
        Err(e) => return Err(Error::internal(format!("metadata {}: {e}", full.display()))),
    };
    if meta.is_dir() {
        walk_files(root, full, rel_child, depth + 1, max_entries, count, visit)?;
        return Ok(EntryHandled::Done);
    }
    if meta.is_file() {
        if *count >= max_entries {
            return Err(Error::oversized(format!(
                "tree at {root:?} exceeds the {max_entries}-entry snapshot cap"
            )));
        }
        *count += 1;
        let f = match fs::File::open(full) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                *count -= 1;
                return Ok(EntryHandled::Gone);
            }
            Err(e) => {
                return Err(Error::internal(format!("open {}: {e}", full.display())));
            }
        };
        if !opened_is_path(&f, full) {
            return Err(Error::permission(format!(
                "{rel_child:?} changed identity between resolution and open (TOCTOU)"
            )));
        }
        visit(rel_child, full, &f)?;
        return Ok(EntryHandled::Done);
    }
    if meta.file_type().is_symlink() {
        let target = fs::canonicalize(full).map_err(|e| {
            Error::permission(format!(
                "symlink {rel_child:?} cannot be resolved safely: {e}"
            ))
        })?;
        if !target.starts_with(root) {
            return Err(Error::permission(format!(
                "symlink escape rejected: {rel_child:?} -> {}",
                target.display()
            )));
        }
        let tmeta = match fs::symlink_metadata(&target) {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(EntryHandled::Gone);
            }
            Err(e) => {
                return Err(Error::internal(format!(
                    "metadata {}: {e}",
                    target.display()
                )));
            }
        };
        if tmeta.is_dir() {
            walk_files(
                root,
                &target,
                rel_child,
                depth + 1,
                max_entries,
                count,
                visit,
            )?;
            return Ok(EntryHandled::Done);
        }
        if tmeta.is_file() {
            if *count >= max_entries {
                return Err(Error::oversized(format!(
                    "tree at {root:?} exceeds the {max_entries}-entry snapshot cap"
                )));
            }
            *count += 1;
            let f = fs::File::open(&target)
                .map_err(|e| Error::internal(format!("open {}: {e}", target.display())))?;
            if !opened_is_path(&f, &target) {
                return Err(Error::permission(format!(
                    "{rel_child:?} changed identity between resolution and open (TOCTOU)"
                )));
            }
            visit(rel_child, &target, &f)?;
            return Ok(EntryHandled::Done);
        }
        return Err(Error::internal(format!(
            "symlink {rel_child:?} resolves to an unsupported file type"
        )));
    }
    Err(Error::internal(format!(
        "refusing to walk unsupported file type at {rel_child:?}"
    )))
}

/// Stream-hash an already-open file handle. Returns (bytes read, hash).
fn hash_open_file(f: &fs::File) -> Result<(u64, FileHash), Error> {
    use std::io::Read;
    let mut hasher = blake3::Hasher::new();
    let mut buf = [0u8; 64 * 1024];
    let mut reader = std::io::BufReader::new(f);
    let mut hashed = 0u64;
    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| Error::internal(format!("read: {e}")))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        hashed += n as u64;
    }
    Ok((hashed, FileHash::from(hasher.finalize().into())))
}

/// Stream-copy an already-open source handle into `target` with the durable
/// atomic sequence: unique temp + fsync + rename + parent fsync. Returns
/// (bytes copied, hash of the copied content). Never skips, never tears.
fn copy_open_file(f: &fs::File, target: &Path) -> Result<(u64, FileHash), Error> {
    let parent = target
        .parent()
        .ok_or_else(|| Error::malformed(format!("{target:?} has no parent")))?;
    fs::create_dir_all(parent)
        .map_err(|e| Error::internal(format!("mkdir {}: {e}", parent.display())))?;
    let tmp = parent.join(format!(
        ".{}.kp-tmp-{}-{}",
        target
            .file_name()
            .map(|n| n.to_string_lossy())
            .unwrap_or_default(),
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    let (n, hash) = {
        let mut out = fs::File::create(&tmp)
            .map_err(|e| Error::internal(format!("create {}: {e}", tmp.display())))?;
        let mut hasher = blake3::Hasher::new();
        use std::io::{Read, Write};
        let mut reader = std::io::BufReader::new(f);
        let mut buf = [0u8; 64 * 1024];
        let mut copied = 0u64;
        loop {
            let r = reader
                .read(&mut buf)
                .map_err(|e| Error::internal(format!("read: {e}")))?;
            if r == 0 {
                break;
            }
            out.write_all(&buf[..r])
                .map_err(|e| Error::internal(format!("write {}: {e}", tmp.display())))?;
            hasher.update(&buf[..r]);
            copied += r as u64;
        }
        out.flush()
            .map_err(|e| Error::internal(format!("flush {}: {e}", tmp.display())))?;
        out.sync_all()
            .map_err(|e| Error::internal(format!("fsync {}: {e}", tmp.display())))?;
        (copied, FileHash::from(hasher.finalize().into()))
    };
    fs::rename(&tmp, target).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        Error::internal(format!(
            "rename {} -> {}: {e}",
            tmp.display(),
            target.display()
        ))
    })?;
    atomic::fsync_parent(parent);
    Ok((n, hash))
}

/// Resolve `rel` under the CANONICAL root, creating every missing
/// INTERMEDIATE directory along the way (used for exclusive merge creates
/// whose parent directories the parent tree never had). Every step is
/// verified: an existing prefix must canonicalize to a directory INSIDE the
/// root (symlink escapes and swapped prefixes fail loudly), and a created
/// prefix is re-verified immediately after `mkdir` (a racer cannot smuggle
/// a symlink past the check). The final component may name an absent file.
fn resolve_or_create_within(root: &Path, rel: &Path) -> Result<PathBuf, Error> {
    let comps: Vec<Component> = rel.components().collect();
    if comps.is_empty() {
        return Err(Error::malformed(format!("{rel:?} is empty")));
    }
    let mut prefix = root.to_path_buf();
    for (i, comp) in comps.iter().enumerate() {
        let Component::Normal(name) = comp else {
            return Err(Error::permission(format!(
                "path traversal rejected: {rel:?}"
            )));
        };
        prefix.push(name);
        let last = i + 1 == comps.len();
        match fs::symlink_metadata(&prefix) {
            Ok(meta) => {
                let canon = prefix.canonicalize().map_err(|_| {
                    Error::permission(format!("parent resolution failed: {rel:?} ({name:?})"))
                })?;
                if !canon.starts_with(root) {
                    return Err(Error::permission(format!(
                        "path escapes workspace: {rel:?}"
                    )));
                }
                if !last && !meta.is_dir() {
                    return Err(Error::conflict(format!(
                        "{:?} exists and is not a directory; {:?} cannot be merged",
                        canon.display(),
                        rel
                    )));
                }
                prefix = canon;
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                if last {
                    // The final file simply does not exist yet.
                    return Ok(prefix);
                }
                fs::create_dir(&prefix).map_err(|ce| {
                    if ce.kind() == std::io::ErrorKind::AlreadyExists {
                        Error::conflict(format!(
                            "{:?} appeared while {:?} was being staged",
                            prefix.display(),
                            rel
                        ))
                    } else {
                        Error::internal(format!("mkdir {}: {ce}", prefix.display()))
                    }
                })?;
                let canon = prefix.canonicalize().map_err(|_| {
                    Error::permission(format!(
                        "parent resolution failed after mkdir: {rel:?} ({name:?})"
                    ))
                })?;
                if !canon.starts_with(root) {
                    return Err(Error::permission(format!(
                        "path escapes workspace: {rel:?}"
                    )));
                }
                prefix = canon;
            }
            Err(e) => {
                return Err(Error::internal(format!(
                    "metadata {}: {e}",
                    prefix.display()
                )));
            }
        }
    }
    Ok(prefix)
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

    // ------------------------------------------------ wave-13 merge/snapshot

    fn tree_fixture(root: &Path) {
        fs::create_dir_all(root.join("src/deep")).unwrap();
        fs::write(root.join("a.txt"), b"alpha").unwrap();
        fs::write(root.join("src/b.rs"), b"mod b;").unwrap();
        fs::write(root.join("src/deep/c.rs"), b"fn c() {}").unwrap();
    }

    #[test]
    fn snapshot_hashes_whole_files_sorted_and_bounded() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("ws");
        fs::create_dir_all(&root).unwrap();
        tree_fixture(&root);
        let snap = snapshot_tree(&root, 100).unwrap();
        let paths: Vec<PathBuf> = snap.iter().map(|e| e.path.clone()).collect();
        assert_eq!(
            paths,
            vec![
                PathBuf::from("a.txt"),
                PathBuf::from("src/b.rs"),
                PathBuf::from("src/deep/c.rs"),
            ]
        );
        for e in &snap {
            let whole = fs::read(root.join(&e.path)).unwrap();
            assert_eq!(
                e.hash,
                FileHash::from(blake3::hash(&whole).into()),
                "hash of {} must cover the whole file",
                e.path.display()
            );
            assert_eq!(e.size, whole.len() as u64);
        }
        // Entry cap: typed Oversized, never a silent truncation.
        let err = snapshot_tree(&root, 2).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Oversized);
        assert!(err.message.contains("cap"), "{err:?}");
    }

    #[test]
    fn snapshot_refuses_symlink_escape_and_unsupported_entries_loudly() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("ws");
        let outside = dir.path().join("outside");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("secret.txt"), b"s").unwrap();
        fs::write(root.join("ok.txt"), b"ok").unwrap();
        // File symlink pointing OUTSIDE the tree: the snapshot must fail
        // loudly — never silently skip the escaping file.
        symlink(outside.join("secret.txt"), root.join("evil-link")).unwrap();
        let err = snapshot_tree(&root, 100).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Permission);
        assert!(err.message.contains("escape"), "{err:?}");
        fs::remove_file(root.join("evil-link")).unwrap();
        // A FIFO is not a file: loud, never skipped.
        #[cfg(unix)]
        {
            let fifo = root.join("fifo");
            std::process::Command::new("mkfifo")
                .arg(&fifo)
                .status()
                .unwrap();
            let err = snapshot_tree(&root, 100).unwrap_err();
            assert!(err.message.contains("unsupported"), "{err:?}");
            fs::remove_file(&fifo).unwrap();
        }
        // Inside-root file symlink is fine (same discipline as resolve()).
        symlink(root.join("ok.txt"), root.join("alias")).unwrap();
        let snap = snapshot_tree(&root, 100).unwrap();
        let paths: Vec<PathBuf> = snap.iter().map(|e| e.path.clone()).collect();
        assert_eq!(paths, vec![PathBuf::from("alias"), PathBuf::from("ok.txt")]);
        assert_eq!(snap[0].hash, snap[1].hash, "alias hashes the target");
    }

    #[test]
    fn copy_tree_replicates_content_and_respects_both_caps() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        let dst = dir.path().join("dst");
        fs::create_dir_all(&src).unwrap();
        fs::create_dir_all(&dst).unwrap();
        tree_fixture(&src);
        let manifest = copy_tree(&src, &dst, 100, 1_000_000).unwrap();
        assert_eq!(manifest.len(), 3);
        let src_snap = snapshot_tree(&src, 100).unwrap();
        assert_eq!(
            manifest, src_snap,
            "copy manifest equals the source snapshot"
        );
        let dst_snap = snapshot_tree(&dst, 100).unwrap();
        assert_eq!(dst_snap, manifest, "destination is byte-identical");
        for e in &manifest {
            assert_eq!(
                fs::read(src.join(&e.path)).unwrap(),
                fs::read(dst.join(&e.path)).unwrap()
            );
        }
        // Entry cap and byte cap are loud Oversized.
        let err = copy_tree(&src, &dst, 2, 1_000_000).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Oversized);
        assert!(err.message.contains("entry"), "{err:?}");
        let err = copy_tree(&src, &dst, 100, 3).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Oversized);
        assert!(err.message.contains("byte"), "{err:?}");
        // Same tree is refused; a missing root is not found.
        assert_eq!(
            copy_tree(&src, &src, 100, 1_000_000).unwrap_err().kind,
            ErrorKind::Conflict
        );
        assert_eq!(
            copy_tree(&dir.path().join("ghost"), &dst, 100, 1_000_000)
                .unwrap_err()
                .kind,
            ErrorKind::NotFound
        );
    }

    #[test]
    fn copy_under_concurrent_atomic_writers_never_sees_a_torn_file() {
        // A writer repeatedly replaces two files with write_atomic_cas
        // payloads A/B while copy_tree runs: every copied file must be a
        // WHOLE A or a WHOLE B image — never a mix — and the copy must
        // succeed (per-file atomic snapshots are the contract, cross-file
        // mixing is allowed).
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();
        let payload_a = format!("AAAAAAAA-{}", "x".repeat(4096));
        let payload_b = format!("BBBBBBBB-{}", "y".repeat(4096));
        let service = WorkspaceFileService::new();
        let h = service.open(WorkspaceId::new(1), src.clone()).unwrap();
        h.write_atomic(Path::new("f1.bin"), payload_a.as_bytes())
            .unwrap();
        h.write_atomic(Path::new("f2.bin"), payload_a.as_bytes())
            .unwrap();
        for round in 0..6 {
            let dst = dir.path().join(format!("dst{round}"));
            fs::create_dir_all(&dst).unwrap();
            let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let h1 = h.clone();
            let s1 = stop.clone();
            let pa = payload_a.clone();
            let pb = payload_b.clone();
            let writer = std::thread::spawn(move || {
                let mut turn = 0usize;
                while !s1.load(std::sync::atomic::Ordering::SeqCst) {
                    let (first, second) = if turn.is_multiple_of(2) {
                        (pa.as_bytes(), pb.as_bytes())
                    } else {
                        (pb.as_bytes(), pa.as_bytes())
                    };
                    for (name, content) in [("f1.bin", first), ("f2.bin", second)] {
                        let _ = h1.read(Path::new(name), 1_000_000).map(|d| {
                            let _ = h1.write_atomic_cas(Path::new(name), d.hash, content);
                        });
                    }
                    turn += 1;
                }
            });
            let result = copy_tree(&src, &dst, 100, 1_000_000);
            stop.store(true, std::sync::atomic::Ordering::SeqCst);
            writer.join().unwrap();
            let manifest = result.expect("copy must succeed under the writer");
            assert_eq!(manifest.len(), 2);
            for e in &manifest {
                let content = fs::read(dst.join(&e.path)).unwrap();
                let text = String::from_utf8(content).unwrap();
                assert!(
                    text == payload_a || text == payload_b,
                    "round {round}: torn copy of {}: {text:?}",
                    e.path.display()
                );
                assert_eq!(
                    e.hash,
                    FileHash::from(blake3::hash(text.as_bytes()).into()),
                    "manifest hash must match the copied bytes"
                );
            }
        }
    }

    #[test]
    fn merge_apply_classifies_conflict_already_current_and_applied() {
        let dir = tempfile::tempdir().unwrap();
        let parent = dir.path().join("parent");
        let child = dir.path().join("child");
        fs::create_dir_all(&parent).unwrap();
        fs::create_dir_all(&child).unwrap();
        // Parent file at the base snapshot.
        fs::write(parent.join("f.rs"), b"base-content").unwrap();
        let base_hash = FileHash::from(blake3::hash(b"base-content").into());
        // Child modified it.
        fs::write(child.join("f.rs"), b"child-edit").unwrap();
        let child_hash = FileHash::from(blake3::hash(b"child-edit").into());
        // 1. Parent unchanged since base: applied, digest preserved.
        let r = merge_apply_content(
            &parent,
            Path::new("f.rs"),
            &child,
            Path::new("f.rs"),
            child_hash,
            Some(base_hash),
        )
        .unwrap();
        assert_eq!(r, CasMergeResult::Applied);
        assert_eq!(fs::read(parent.join("f.rs")).unwrap(), b"child-edit");
        // 2. Replay: already merged content -> AlreadyCurrent, idempotent.
        let r = merge_apply_content(
            &parent,
            Path::new("f.rs"),
            &child,
            Path::new("f.rs"),
            child_hash,
            Some(base_hash),
        )
        .unwrap();
        assert_eq!(r, CasMergeResult::AlreadyCurrent);
        assert_eq!(fs::read(parent.join("f.rs")).unwrap(), b"child-edit");
        // 3. Parent moved on (different content): conflict, parent intact.
        fs::write(parent.join("f.rs"), b"parent-edit-after-base").unwrap();
        let err = merge_apply_content(
            &parent,
            Path::new("f.rs"),
            &child,
            Path::new("f.rs"),
            child_hash,
            Some(base_hash),
        )
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Conflict);
        assert_eq!(
            fs::read(parent.join("f.rs")).unwrap(),
            b"parent-edit-after-base",
            "the conflicting parent file must stay byte-identical"
        );
        // 4. Parent file vanished: never recreated from a stale patch.
        fs::remove_file(parent.join("f.rs")).unwrap();
        let err = merge_apply_content(
            &parent,
            Path::new("f.rs"),
            &child,
            Path::new("f.rs"),
            child_hash,
            Some(base_hash),
        )
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Conflict);
        assert!(!parent.join("f.rs").exists());
        // 5. Child content drifted since staging: conflict, nothing merged.
        fs::write(parent.join("f.rs"), b"base-content").unwrap();
        fs::write(child.join("f.rs"), b"child-changed-again").unwrap();
        let err = merge_apply_content(
            &parent,
            Path::new("f.rs"),
            &child,
            Path::new("f.rs"),
            child_hash,
            Some(base_hash),
        )
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Conflict);
        assert!(err.message.contains("drifted"), "{err:?}");
        assert_eq!(fs::read(parent.join("f.rs")).unwrap(), b"base-content");
    }

    #[test]
    fn merge_create_is_exclusive_and_replay_safe() {
        let dir = tempfile::tempdir().unwrap();
        let parent = dir.path().join("parent");
        let child = dir.path().join("child");
        fs::create_dir_all(&parent).unwrap();
        fs::create_dir_all(&child).unwrap();
        fs::write(child.join("new.txt"), b"child-new-file").unwrap();
        let child_hash = FileHash::from(blake3::hash(b"child-new-file").into());
        // Base had no such file (expected None): exclusive create.
        let r = merge_apply_content(
            &parent,
            Path::new("new.txt"),
            &child,
            Path::new("new.txt"),
            child_hash,
            None,
        )
        .unwrap();
        assert_eq!(r, CasMergeResult::Applied);
        assert_eq!(fs::read(parent.join("new.txt")).unwrap(), b"child-new-file");
        // Replay: file already holds the child digest -> AlreadyCurrent.
        let r = merge_apply_content(
            &parent,
            Path::new("new.txt"),
            &child,
            Path::new("new.txt"),
            child_hash,
            None,
        )
        .unwrap();
        assert_eq!(r, CasMergeResult::AlreadyCurrent);
        // A parent-side file appeared since the base: conflict, never an
        // overwrite of a file the merge did not create.
        fs::remove_file(parent.join("new.txt")).unwrap();
        fs::write(parent.join("new.txt"), b"parent-made-this").unwrap();
        let err = merge_apply_content(
            &parent,
            Path::new("new.txt"),
            &child,
            Path::new("new.txt"),
            child_hash,
            None,
        )
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Conflict);
        assert_eq!(
            fs::read(parent.join("new.txt")).unwrap(),
            b"parent-made-this"
        );
    }

    #[test]
    fn merge_create_materializes_missing_parent_dirs_inside_the_root_only() {
        let dir = tempfile::tempdir().unwrap();
        let parent = dir.path().join("parent");
        let child = dir.path().join("child");
        fs::create_dir_all(&parent).unwrap();
        fs::create_dir_all(&child).unwrap();
        // The child created a nested file whose parent directories never
        // existed in the parent tree: the merge creates them.
        fs::create_dir_all(child.join("sub/deep")).unwrap();
        fs::write(child.join("sub/deep/new.rs"), b"child-nested").unwrap();
        let child_hash = FileHash::from(blake3::hash(b"child-nested").into());
        let r = merge_apply_content(
            &parent,
            Path::new("sub/deep/new.rs"),
            &child,
            Path::new("sub/deep/new.rs"),
            child_hash,
            None,
        )
        .unwrap();
        assert_eq!(r, CasMergeResult::Applied);
        assert_eq!(
            fs::read(parent.join("sub/deep/new.rs")).unwrap(),
            b"child-nested"
        );
        // Replay after the dirs exist: AlreadyCurrent, dirs not duplicated.
        let r = merge_apply_content(
            &parent,
            Path::new("sub/deep/new.rs"),
            &child,
            Path::new("sub/deep/new.rs"),
            child_hash,
            None,
        )
        .unwrap();
        assert_eq!(r, CasMergeResult::AlreadyCurrent);
        // A hostile create that must walk THROUGH an existing file for an
        // intermediate component is a conflict, never a clobber.
        fs::write(parent.join("blocker"), b"i-am-a-file").unwrap();
        fs::create_dir_all(child.join("blocker/x")).unwrap();
        fs::write(child.join("blocker/x/y.rs"), b"z").unwrap();
        let hz = FileHash::from(blake3::hash(b"z").into());
        let err = merge_apply_content(
            &parent,
            Path::new("blocker/x/y.rs"),
            &child,
            Path::new("blocker/x/y.rs"),
            hz,
            None,
        )
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Conflict, "{err:?}");
        assert_eq!(fs::read(parent.join("blocker")).unwrap(), b"i-am-a-file");
    }

    #[test]
    fn merge_delete_removes_only_the_base_content() {
        let dir = tempfile::tempdir().unwrap();
        let parent = dir.path().join("parent");
        fs::create_dir_all(&parent).unwrap();
        fs::write(parent.join("gone.rs"), b"base-content").unwrap();
        let base_hash = FileHash::from(blake3::hash(b"base-content").into());
        assert_eq!(
            merge_delete(&parent, Path::new("gone.rs"), base_hash).unwrap(),
            CasMergeResult::Applied
        );
        assert!(!parent.join("gone.rs").exists());
        // Replay after the crashed run deleted it: end state (absent) holds.
        assert_eq!(
            merge_delete(&parent, Path::new("gone.rs"), base_hash).unwrap(),
            CasMergeResult::AlreadyCurrent
        );
        // Parent content moved on: the deletion is refused, file intact.
        fs::write(parent.join("gone.rs"), b"base-content").unwrap();
        fs::write(parent.join("gone.rs"), b"parent-recreated").unwrap();
        let err = merge_delete(&parent, Path::new("gone.rs"), base_hash).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Conflict);
        assert_eq!(
            fs::read(parent.join("gone.rs")).unwrap(),
            b"parent-recreated"
        );
    }

    #[test]
    fn merge_paths_are_resolved_traversal_and_escape_safe() {
        let dir = tempfile::tempdir().unwrap();
        let parent = dir.path().join("parent");
        let child = dir.path().join("child");
        fs::create_dir_all(&parent).unwrap();
        fs::create_dir_all(&child).unwrap();
        fs::write(child.join("x.txt"), b"x").unwrap();
        let h = FileHash::from(blake3::hash(b"x").into());
        for evil in ["../x.txt", "a/../../x.txt", "/etc/x.txt", ".."] {
            let err = merge_apply_content(
                &parent,
                Path::new(evil),
                &child,
                Path::new("x.txt"),
                h,
                None,
            )
            .unwrap_err();
            assert_eq!(err.kind, ErrorKind::Permission, "{evil} must be refused");
        }
        let err = merge_apply_content(
            &parent,
            Path::new("x.txt"),
            &child,
            Path::new("x.txt"),
            h,
            None,
        )
        .unwrap();
        assert_eq!(err, CasMergeResult::Applied);
        // An inside-root symlink destination that escapes the parent root
        // must never be followed for the write.
        let outside = dir.path().join("outside-root");
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("victim.txt"), b"v").unwrap();
        fs::create_dir_all(parent.join("sub")).unwrap();
        symlink(&outside, parent.join("sub/link-out")).unwrap();
        fs::write(child.join("payload.txt"), b"p").unwrap();
        let hp = FileHash::from(blake3::hash(b"p").into());
        let err = merge_apply_content(
            &parent,
            Path::new("sub/link-out/victim.txt"),
            &child,
            Path::new("payload.txt"),
            hp,
            None,
        )
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Permission);
        assert_eq!(fs::read(outside.join("victim.txt")).unwrap(), b"v");
        assert_eq!(
            merge_delete(&parent, Path::new("../x"), h)
                .unwrap_err()
                .kind,
            ErrorKind::Permission
        );
    }
}
