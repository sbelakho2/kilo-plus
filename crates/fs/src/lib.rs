//! kilop-fs — file service and watcher (spec §21 resource scopes).
//!
//! Every call carries its explicit workspace identity; paths are relative to
//! the workspace root and resolved traversal/symlink-safely. Reads are
//! bounded; writes are atomic (temp + fsync + rename). Workspaces own their
//! watcher; `close` unloads heavyweight resources after inactivity.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

use kilop_core::error::{Error, ErrorKind};
use kilop_core::hash::FileHash;
use kilop_core::id::WorkspaceId;
use kilop_core::WorkspaceIdentity;
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::mpsc;

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
    pub fn read(&self, rel: &Path, max_bytes: usize) -> Result<FileData, Error> {
        let path = self.resolve(rel)?;
        let size = fs::metadata(&path)
            .map_err(|e| err_not_found(rel, e))?
            .len();
        let mut f = fs::File::open(&path).map_err(|e| err_not_found(rel, e))?;
        use std::io::Read;
        let mut bytes = Vec::new();
        let mut truncated = false;
        if size > max_bytes as u64 {
            bytes.resize(max_bytes, 0);
            f.read_exact(&mut bytes)
                .map_err(|e| Error::internal(format!("read {rel:?}: {e}")))?;
            truncated = true;
        } else {
            f.read_to_end(&mut bytes)
                .map_err(|e| Error::internal(format!("read {rel:?}: {e}")))?;
        }
        let hash = FileHash::from(blake3::hash(&bytes).into());
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

    /// Slice read for paging big files (spec §23).
    pub fn read_slice(&self, rel: &Path, offset: u64, len: usize) -> Result<FileData, Error> {
        use std::io::{Read, Seek, SeekFrom};
        let path = self.resolve(rel)?;
        let mut f = fs::File::open(&path).map_err(|e| err_not_found(rel, e))?;
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

    /// Atomic write: temp file in the same dir + fsync + rename.
    pub fn write_atomic(&self, rel: &Path, bytes: &[u8]) -> Result<FileHash, Error> {
        let path = self.resolve(rel)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| Error::internal(format!("mkdir {}: {e}", parent.display())))?;
        }
        let tmp = path.with_extension(format!(
            "kp-tmp-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        {
            let mut f =
                fs::File::create(&tmp).map_err(|e| Error::internal(format!("create tmp: {e}")))?;
            f.write_all(bytes)
                .map_err(|e| Error::internal(format!("write tmp: {e}")))?;
            f.sync_all()
                .map_err(|e| Error::internal(format!("fsync tmp: {e}")))?;
        }
        fs::rename(&tmp, &path).map_err(|e| {
            let _ = fs::remove_file(&tmp);
            Error::internal(format!("rename {}: {e}", path.display()))
        })?;
        Ok(FileHash::from(blake3::hash(bytes).into()))
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
            kilop_core::WorktreeId::new(1),
            kilop_core::TaskId::new(1),
        );
        assert!(h.verify_identity(&wrong).is_err());
        let right = WorkspaceIdentity::new(
            WorkspaceId::new(1),
            kilop_core::WorktreeId::new(1),
            kilop_core::TaskId::new(1),
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
}
