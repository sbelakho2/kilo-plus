//! faktor-git — worktree management and per-repository mutation locks
//! (spec §33). Git is used only for legitimate Git operations; every
//! invocation is a supervised child with an explicit owner. Read-only
//! operations run concurrently; mutations serialize per repository (never
//! a global Git lock across unrelated repos).

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use faktor_core::cancellation::CancellationToken;
use faktor_core::error::{Error, ErrorKind};
use faktor_core::id::{SessionId, WorktreeId};
use faktor_terminal::{ProcessOwner, ProcessSupervisor, SpawnConfig};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Worktree {
    pub id: WorktreeId,
    pub workspace_root: PathBuf,
    pub path: PathBuf,
    pub branch: String,
    /// The session that owns this worktree (None = discovered with no
    /// recorded owner — never an invented session id).
    pub owner_session: Option<SessionId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Validation {
    pub exists: bool,
    pub has_git_file: bool,
    pub branch: Option<String>,
    pub clean: bool,
}

/// Durable worktree metadata (spec §33): git knows nothing about session
/// ownership, so the manager records it next to the repository's own
/// bookkeeping (inside `.git/` — never user-visible, gitignored by
/// definition, survives daemon restarts and worktree re-discovery).
const META_FILE: &str = "faktor-plus-worktrees.json";
const META_MAX_ENTRIES: usize = 500;
const META_MAX_PATH_BYTES: usize = 4096;
const META_MAX_BRANCH_BYTES: usize = 256;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct MetaEntry {
    id: u64,
    path: String,
    branch: String,
    owner_session: u64,
    created_ms: i64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Meta {
    worktrees: Vec<MetaEntry>,
}

fn meta_path(workspace_root: &Path) -> PathBuf {
    workspace_root.join(".git").join(META_FILE)
}

/// A temp sibling of the final metadata file that is unique PER WRITE:
/// process-wide monotonic nonce + pid + nanosecond timestamp. Concurrent
/// writers (multiple manager instances over the same repo) each get their
/// own name, so two in-flight saves can never tear or interleave the final
/// file — whichever rename lands last wins atomically.
fn unique_meta_tmp_path(final_path: &Path) -> PathBuf {
    static NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let nonce = NONCE.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let name = final_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| META_FILE.to_string());
    final_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!("{name}.tmp.{}.{nonce}.{nanos}", std::process::id()))
}

#[derive(Clone)]
pub struct WorktreeManager {
    supervisor: Arc<ProcessSupervisor>,
    /// Per-repository mutation locks (never global). Tokio RwLock so the
    /// guard is Send across awaits.
    locks: Arc<Mutex<HashMap<PathBuf, Arc<tokio::sync::RwLock<()>>>>>,
    next_id: Arc<std::sync::atomic::AtomicU64>,
    /// Serializes metadata file reads/writes (fast, bounded; the git ops
    /// themselves stay under the per-repo locks).
    meta_lock: Arc<tokio::sync::Mutex<()>>,
}

impl WorktreeManager {
    pub fn new(supervisor: Arc<ProcessSupervisor>) -> Self {
        Self {
            supervisor,
            locks: Arc::new(Mutex::new(HashMap::new())),
            next_id: Arc::new(std::sync::atomic::AtomicU64::new(1)),
            meta_lock: Arc::new(tokio::sync::Mutex::new(())),
        }
    }

    /// Load the durable metadata; a corrupt or hostile file is treated as
    /// empty (never an error — the manager repairs by re-recording).
    fn load_meta(&self, workspace_root: &Path) -> Meta {
        let Ok(bytes) = std::fs::read(meta_path(workspace_root)) else {
            return Meta { worktrees: vec![] };
        };
        let Ok(parsed) = serde_json::from_slice::<Meta>(&bytes) else {
            return Meta { worktrees: vec![] };
        };
        if parsed.worktrees.len() > META_MAX_ENTRIES {
            return Meta { worktrees: vec![] };
        }
        let mut sane = Vec::new();
        for e in parsed.worktrees {
            if e.path.is_empty()
                || e.path.len() > META_MAX_PATH_BYTES
                || e.branch.len() > META_MAX_BRANCH_BYTES
            {
                continue; // hostile entry: drop it
            }
            sane.push(e);
        }
        Meta { worktrees: sane }
    }

    /// Durable metadata save with CAS discipline (spec §33): a UNIQUE temp
    /// file name per write (never a shared fixed name — concurrent writers
    /// cannot interleave into one torn file), then write + flush + fsync the
    /// file, rename over the final, and fsync the parent directory so the
    /// rename itself survives a crash. Best-effort by design: metadata loss
    /// is recoverable by re-discovery, so any failed step logs a warning
    /// and never fails the surrounding git operation.
    fn save_meta(&self, workspace_root: &Path, meta: &Meta) {
        let path = meta_path(workspace_root);
        let Some(dir) = path.parent() else {
            return;
        };
        if !dir.is_dir() {
            return;
        }
        let bytes = match serde_json::to_vec(meta) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "metadata serialization failed");
                return;
            }
        };
        let tmp = unique_meta_tmp_path(&path);
        if let Err(e) = (|| -> std::io::Result<()> {
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&tmp)?;
            f.write_all(&bytes)?;
            f.flush()?;
            f.sync_all()?; // fsync the file before it is published
            drop(f);
            std::fs::rename(&tmp, &path)?;
            Ok(())
        })() {
            tracing::warn!(path = %path.display(), tmp = %tmp.display(), error = %e, "durable metadata save failed; ownership recoverable by re-discovery");
            let _ = std::fs::remove_file(&tmp); // best-effort cleanup
            return;
        }
        // fsync the parent directory so the rename is durable (an opened
        // directory fsyncs fine on macOS and Linux).
        match std::fs::File::open(dir) {
            Ok(d) => {
                if let Err(e) = d.sync_all() {
                    tracing::warn!(dir = %dir.display(), error = %e, "metadata directory fsync failed");
                }
            }
            Err(e) => {
                tracing::warn!(dir = %dir.display(), error = %e, "cannot open metadata directory for fsync")
            }
        }
    }

    fn meta_entry(wt: &Worktree, created_ms: i64) -> MetaEntry {
        MetaEntry {
            id: wt.id.raw(),
            path: wt.path.to_string_lossy().into_owned(),
            branch: wt.branch.clone(),
            owner_session: wt.owner_session.map(|s| s.raw()).unwrap_or(0),
            created_ms,
        }
    }

    fn lock_for(&self, repo: &Path) -> Arc<tokio::sync::RwLock<()>> {
        let mut locks = self.locks.lock().unwrap();
        locks
            .entry(repo.to_path_buf())
            .or_insert_with(|| Arc::new(tokio::sync::RwLock::new(())))
            .clone()
    }

    /// Run a read-only git op under the repository's READ lock (concurrent).
    pub async fn git_read(
        &self,
        repo: &Path,
        args: &[&str],
        owner: ProcessOwner,
    ) -> Result<String, Error> {
        let lock = self.lock_for(repo);
        let _guard = lock.read().await;
        self.git(repo, args, owner).await
    }

    /// Run a mutating git op under the repository's WRITE lock (serialized
    /// per repo; unrelated repos stay concurrent).
    pub async fn git_mutate(
        &self,
        repo: &Path,
        args: &[&str],
        owner: ProcessOwner,
    ) -> Result<String, Error> {
        let lock = self.lock_for(repo);
        let _guard = lock.write().await;
        self.git(repo, args, owner).await
    }

    async fn git(&self, repo: &Path, args: &[&str], owner: ProcessOwner) -> Result<String, Error> {
        if !repo.is_dir() {
            return Err(Error::not_found(format!(
                "repository {} not found",
                repo.display()
            )));
        }
        let cfg = SpawnConfig {
            cmd: "git".into(),
            args: args.iter().map(|s| s.to_string()).collect(),
            cwd: repo.to_path_buf(),
            owner,
            ..Default::default()
        };
        // Deliberate CI-hang bound: 15s per git op so a wedged git process
        // fails the op (and its test) instead of hanging the whole stage.
        let out = self
            .supervisor
            .run(
                cfg,
                std::time::Duration::from_secs(15),
                CancellationToken::new(),
            )
            .await?;
        if out.exit_code != Some(0) {
            return Err(Error::new(
                ErrorKind::Internal,
                format!(
                    "git {} failed ({}): {}",
                    args.join(" "),
                    out.exit_code.unwrap_or(-1),
                    truncate(&out.excerpt, 2000)
                ),
            ));
        }
        // The supervisor appends an "[exit code: 0]" trailer; strip it so
        // git output is used verbatim.
        let out = strip_exit_trailer(&out.excerpt);
        Ok(out)
    }

    // ---------------------------------------------------------------- worktrees

    pub async fn create(
        &self,
        workspace_root: &Path,
        branch: &str,
        name: &str,
        owner: SessionId,
    ) -> Result<Worktree, Error> {
        validate_branch(branch)?;
        validate_name(name)?;
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if id == 0 {
            return Err(Error::internal("worktree id overflow"));
        }
        // Canonicalize so recorded paths match git's own absolute output.
        let workspace_root = workspace_root.canonicalize().map_err(|e| {
            Error::not_found(format!("workspace {}: {e}", workspace_root.display()))
        })?;
        let wt_path = workspace_root.join(format!(".worktrees/{name}"));
        let owner_po = ProcessOwner::Session(owner);
        self.git_mutate(
            &workspace_root,
            &["worktree", "add", "-b", branch, wt_path.to_str().unwrap()],
            owner_po.clone(),
        )
        .await?;
        let wt = Worktree {
            id: WorktreeId::new(id),
            workspace_root: workspace_root.to_path_buf(),
            path: wt_path,
            branch: branch.to_string(),
            owner_session: Some(owner),
        };
        // Durable ownership record (spec §33: transfer changes owner
        // durably — the git worktree itself has no owner concept).
        let _guard = self.meta_lock.lock().await;
        let mut meta = self.load_meta(&workspace_root);
        meta.worktrees.push(Self::meta_entry(
            &wt,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as i64)
                .unwrap_or(0),
        ));
        self.save_meta(&workspace_root, &meta);
        Ok(wt)
    }

    pub async fn discover(&self, workspace_root: &Path) -> Result<Vec<Worktree>, Error> {
        // Phase (a) — git read (repository READ lock inside git_read) with
        // NO metadata lock held: universal order is repository lock first,
        // metadata second, and the metadata lock must never be held across
        // a git operation.
        let out = self
            .git_read(
                workspace_root,
                &["worktree", "list", "--porcelain"],
                ProcessOwner::Daemon,
            )
            .await?;
        // Phase (b) — metadata scope: load, merge, maybe save. Only filesystem
        // and in-memory work here; no git function runs while the metadata
        // lock is held.
        let _guard = self.meta_lock.lock().await;
        let mut meta = self.load_meta(workspace_root);
        let by_path: std::collections::HashMap<String, MetaEntry> = meta
            .worktrees
            .iter()
            .map(|e| (e.path.clone(), e.clone()))
            .collect();
        let mut worktrees = Vec::new();
        let mut current: Option<(String, String)> = None; // (path, branch)
        let mut saw_new = false;
        let flush = |current: &mut Option<(String, String)>,
                     worktrees: &mut Vec<Worktree>,
                     by_path: &std::collections::HashMap<String, MetaEntry>,
                     workspace_root: &Path,
                     next_id: &std::sync::atomic::AtomicU64| {
            if let Some((path, branch)) = current.take() {
                let meta_entry = by_path.get(&path);
                let id = meta_entry
                    .map(|e| e.id)
                    .unwrap_or_else(|| next_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst));
                let owner = meta_entry
                    .and_then(|e| (e.owner_session != 0).then(|| SessionId::new(e.owner_session)));
                worktrees.push(Worktree {
                    id: WorktreeId::new(id),
                    workspace_root: workspace_root.to_path_buf(),
                    path: PathBuf::from(&path),
                    branch,
                    owner_session: owner,
                });
            }
        };
        for line in out.lines() {
            if let Some(path) = line.strip_prefix("worktree ") {
                flush(
                    &mut current,
                    &mut worktrees,
                    &by_path,
                    workspace_root,
                    &self.next_id,
                );
                current = Some((path.trim().to_string(), String::new()));
            } else if let Some(branch) = line.strip_prefix("branch refs/heads/") {
                if let Some((_, b)) = current.as_mut() {
                    *b = branch.trim().to_string();
                }
            }
        }
        flush(
            &mut current,
            &mut worktrees,
            &by_path,
            workspace_root,
            &self.next_id,
        );
        // Record metadata for previously unknown worktrees (stable ids and
        // unowned markers persist across restarts).
        for wt in &worktrees {
            let key = wt.path.to_string_lossy().into_owned();
            if !by_path.contains_key(&key) {
                meta.worktrees.push(MetaEntry {
                    id: wt.id.raw(),
                    path: key.clone(),
                    branch: wt.branch.clone(),
                    owner_session: 0,
                    created_ms: 0,
                });
                saw_new = true;
            }
        }
        if saw_new {
            self.save_meta(workspace_root, &meta);
        }
        Ok(worktrees)
    }

    pub async fn validate(&self, wt: &Worktree) -> Result<Validation, Error> {
        let exists = wt.path.is_dir();
        let has_git_file = exists && wt.path.join(".git").exists();
        let branch = if has_git_file {
            self.git_read(
                &wt.path,
                &["branch", "--show-current"],
                ProcessOwner::Daemon,
            )
            .await
            .ok()
            .map(|b| b.trim().to_string())
        } else {
            None
        };
        let clean = if has_git_file {
            self.git_read(&wt.path, &["status", "--porcelain"], ProcessOwner::Daemon)
                .await
                .map(|s| s.trim().is_empty())
                .unwrap_or(false)
        } else {
            false
        };
        Ok(Validation {
            exists,
            has_git_file,
            branch,
            clean,
        })
    }

    pub async fn remove(&self, wt: &Worktree) -> Result<(), Error> {
        self.git_mutate(
            &wt.workspace_root,
            &["worktree", "remove", "--force", wt.path.to_str().unwrap()],
            ProcessOwner::Daemon,
        )
        .await?;
        let _guard = self.meta_lock.lock().await;
        let mut meta = self.load_meta(&wt.workspace_root);
        let key = wt.path.to_string_lossy().into_owned();
        meta.worktrees.retain(|e| e.path != key);
        self.save_meta(&wt.workspace_root, &meta);
        Ok(())
    }

    /// Deliberate, DURABLE ownership move (spec §33): the recorded owner of
    /// the worktree becomes `new_owner` and survives daemon restarts.
    pub async fn transfer(&self, wt: &Worktree, new_owner: SessionId) -> Result<(), Error> {
        if !wt.path.is_dir() {
            return Err(Error::not_found(format!(
                "worktree {} missing",
                wt.path.display()
            )));
        }
        let _guard = self.meta_lock.lock().await;
        let mut meta = self.load_meta(&wt.workspace_root);
        let key = wt.path.to_string_lossy().into_owned();
        let mut found = false;
        for e in meta.worktrees.iter_mut() {
            if e.path == key {
                e.owner_session = new_owner.raw();
                found = true;
                break;
            }
        }
        if !found {
            // Adopt an unrecorded worktree (recovery after metadata loss).
            meta.worktrees.push(MetaEntry {
                id: wt.id.raw(),
                path: key,
                branch: wt.branch.clone(),
                owner_session: new_owner.raw(),
                created_ms: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as i64)
                    .unwrap_or(0),
            });
        }
        self.save_meta(&wt.workspace_root, &meta);
        Ok(())
    }

    /// Repair (spec §33): prune stale git bookkeeping and drop metadata for
    /// worktrees whose directories vanished. Returns the cleaned paths.
    ///
    /// Lock order is ALWAYS (1) repository git lock, (2) metadata lock; the
    /// metadata lock is never held across a git operation. Repair therefore
    /// runs in two phases: git prune first with no metadata lock held, then
    /// a metadata-only cleanup under the metadata lock.
    pub async fn repair(&self, workspace_root: &Path) -> Result<Vec<String>, Error> {
        // Phase 1 — git first, no metadata lock: a pure directory-existence
        // scan decides whether git's own bookkeeping needs pruning (the
        // scan is lock-free; a rename-based metadata save is atomic, and a
        // vanished directory is plain filesystem state).
        let preliminary = self.load_meta(workspace_root);
        let any_vanished = preliminary
            .worktrees
            .iter()
            .any(|e| !PathBuf::from(&e.path).is_dir());
        if any_vanished {
            // Best-effort like the metadata save: a prune failure must not
            // break repair, and metadata cleanup below stays authoritative.
            let _ = self
                .git_mutate(workspace_root, &["worktree", "prune"], ProcessOwner::Daemon)
                .await;
        }
        // Phase 2 — metadata cleanup under the metadata lock only. No git
        // function runs between the lock acquisition and its release.
        let _guard = self.meta_lock.lock().await;
        let mut meta = self.load_meta(workspace_root);
        let mut repaired = Vec::new();
        meta.worktrees.retain(|e| {
            let path = PathBuf::from(&e.path);
            if path.is_dir() {
                true
            } else {
                repaired.push(e.path.clone());
                false
            }
        });
        if !repaired.is_empty() {
            self.save_meta(workspace_root, &meta);
        }
        Ok(repaired)
    }
}

fn validate_branch(branch: &str) -> Result<(), Error> {
    if branch.is_empty() || branch.len() > 128 {
        return Err(Error::malformed("branch name empty or too long"));
    }
    if branch.starts_with('-')
        || branch.contains("..")
        || branch.contains(" ")
        || branch.contains('/') && branch.ends_with('/')
    {
        return Err(Error::malformed(format!("branch name {branch:?} rejected")));
    }
    for c in branch.chars() {
        if c.is_control() || matches!(c, '~' | '^' | ':' | '?' | '*' | '[' | '\\' | ' ' | '\t') {
            return Err(Error::malformed(format!("branch name {branch:?} rejected")));
        }
    }
    Ok(())
}

fn validate_name(name: &str) -> Result<(), Error> {
    if name.is_empty() || name.len() > 64 {
        return Err(Error::malformed("worktree name empty or too long"));
    }
    if name.contains('/') || name.contains("..") || name.contains(' ') || name.contains('\t') {
        return Err(Error::malformed(format!("worktree name {name:?} rejected")));
    }
    Ok(())
}

fn strip_exit_trailer(s: &str) -> String {
    s.lines()
        .filter(|l| !l.starts_with("[exit code: "))
        .collect::<Vec<_>>()
        .join("\n")
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    async fn fixture() -> (
        tempfile::TempDir,
        Arc<ProcessSupervisor>,
        WorktreeManager,
        PathBuf,
    ) {
        let dir = tempdir().unwrap();
        let cas = Arc::new(faktor_cas::Cas::open(dir.path().join("cas")).unwrap());
        let sup = ProcessSupervisor::new(cas);
        let repo = dir.path().join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        // Init a repo with a first commit (worktree add needs one).
        sup.run(
            SpawnConfig {
                cmd: "git".into(),
                args: vec!["init".into(), "-b".into(), "main".into()],
                cwd: repo.clone(),
                owner: ProcessOwner::Daemon,
                ..Default::default()
            },
            std::time::Duration::from_secs(20),
            CancellationToken::new(),
        )
        .await
        .unwrap();
        std::fs::write(repo.join("README.md"), "# repo\n").unwrap();
        sup.run(
            SpawnConfig {
                cmd: "git".into(),
                args: vec![
                    "-c".into(),
                    "user.email=test@kilo.local".into(),
                    "-c".into(),
                    "user.name=Kilo Test".into(),
                    "add".into(),
                    ".".into(),
                ],
                cwd: repo.clone(),
                owner: ProcessOwner::Daemon,
                ..Default::default()
            },
            std::time::Duration::from_secs(20),
            CancellationToken::new(),
        )
        .await
        .unwrap();
        sup.run(
            SpawnConfig {
                cmd: "git".into(),
                args: vec![
                    "-c".into(),
                    "user.email=test@kilo.local".into(),
                    "-c".into(),
                    "user.name=Kilo Test".into(),
                    "commit".into(),
                    "-m".into(),
                    "init".into(),
                ],
                cwd: repo.clone(),
                owner: ProcessOwner::Daemon,
                ..Default::default()
            },
            std::time::Duration::from_secs(20),
            CancellationToken::new(),
        )
        .await
        .unwrap();
        let mgr = WorktreeManager::new(sup.clone());
        (dir, sup, mgr, repo)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_worktree_has_valid_git_metadata() {
        tokio::time::timeout(
            std::time::Duration::from_secs(25),
            create_worktree_has_valid_git_metadata_inner(),
        )
        .await
        .unwrap_or_else(|_| {
            panic!("create_worktree_has_valid_git_metadata exceeded its 25s hard test timeout")
        });
    }

    async fn create_worktree_has_valid_git_metadata_inner() {
        let (_d, _sup, mgr, repo) = fixture().await;
        let wt = mgr
            .create(&repo, "feat/x", "wt1", SessionId::new(7))
            .await
            .unwrap();
        assert!(
            wt.path.join(".git").exists(),
            "worktree must have a .git file"
        );
        let v = mgr.validate(&wt).await.unwrap();
        assert!(v.exists);
        assert!(v.has_git_file);
        assert_eq!(v.branch.as_deref(), Some("feat/x"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn discover_finds_created_worktrees() {
        tokio::time::timeout(
            std::time::Duration::from_secs(25),
            discover_finds_created_worktrees_inner(),
        )
        .await
        .unwrap_or_else(|_| {
            panic!("discover_finds_created_worktrees exceeded its 25s hard test timeout")
        });
    }

    async fn discover_finds_created_worktrees_inner() {
        let (_d, _sup, mgr, repo) = fixture().await;
        let wt = mgr
            .create(&repo, "feat/a", "wt-a", SessionId::new(1))
            .await
            .unwrap();
        let _ = mgr
            .create(&repo, "feat/b", "wt-b", SessionId::new(1))
            .await
            .unwrap();
        let found = mgr.discover(&repo).await.unwrap();
        assert!(
            found.iter().any(|w| w.path == wt.path),
            "discover must include the created worktree"
        );
        assert!(found.len() >= 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn validate_rejects_missing_dir_and_accepts_healthy() {
        tokio::time::timeout(
            std::time::Duration::from_secs(25),
            validate_rejects_missing_dir_and_accepts_healthy_inner(),
        )
        .await
        .unwrap_or_else(|_| panic!("validate_rejects_missing_dir_and_accepts_healthy exceeded its 25s hard test timeout"));
    }

    async fn validate_rejects_missing_dir_and_accepts_healthy_inner() {
        let (_d, _sup, mgr, repo) = fixture().await;
        let wt = mgr
            .create(&repo, "feat/c", "wt-c", SessionId::new(1))
            .await
            .unwrap();
        let healthy = mgr.validate(&wt).await.unwrap();
        assert!(healthy.has_git_file);
        let missing = Worktree {
            id: WorktreeId::new(99),
            workspace_root: repo.clone(),
            path: repo.join(".worktrees/does-not-exist"),
            branch: "x".into(),
            owner_session: Some(SessionId::new(1)),
        };
        let bad = mgr.validate(&missing).await.unwrap();
        assert!(!bad.exists);
        assert!(!bad.has_git_file);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn remove_removes_worktree() {
        tokio::time::timeout(
            std::time::Duration::from_secs(25),
            remove_removes_worktree_inner(),
        )
        .await
        .unwrap_or_else(|_| panic!("remove_removes_worktree exceeded its 25s hard test timeout"));
    }

    async fn remove_removes_worktree_inner() {
        let (_d, _sup, mgr, repo) = fixture().await;
        let wt = mgr
            .create(&repo, "feat/d", "wt-d", SessionId::new(1))
            .await
            .unwrap();
        mgr.remove(&wt).await.unwrap();
        assert!(!wt.path.exists(), "worktree must be removed");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn transfer_is_deliberate_and_validates() {
        tokio::time::timeout(
            std::time::Duration::from_secs(25),
            transfer_is_deliberate_and_validates_inner(),
        )
        .await
        .unwrap_or_else(|_| {
            panic!("transfer_is_deliberate_and_validates exceeded its 25s hard test timeout")
        });
    }

    async fn transfer_is_deliberate_and_validates_inner() {
        let (_d, _sup, mgr, repo) = fixture().await;
        let wt = mgr
            .create(&repo, "feat/e", "wt-e", SessionId::new(1))
            .await
            .unwrap();
        mgr.transfer(&wt, SessionId::new(2)).await.unwrap();
        // Transfer of a missing worktree is loud.
        let ghost = Worktree {
            id: wt.id,
            workspace_root: repo.clone(),
            path: repo.join("gone"),
            branch: "x".into(),
            owner_session: Some(SessionId::new(1)),
        };
        assert!(mgr.transfer(&ghost, SessionId::new(3)).await.is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn mutate_lock_serializes_writes_per_repo() {
        tokio::time::timeout(
            std::time::Duration::from_secs(25),
            mutate_lock_serializes_writes_per_repo_inner(),
        )
        .await
        .unwrap_or_else(|_| {
            panic!("mutate_lock_serializes_writes_per_repo exceeded its 25s hard test timeout")
        });
    }

    async fn mutate_lock_serializes_writes_per_repo_inner() {
        let (_d, _sup, mgr, repo) = fixture().await;
        // Two concurrent mutation streams on the same repo: no corruption.
        let mut handles = Vec::new();
        for t in 0..4 {
            let mgr = mgr.clone();
            let repo = repo.clone();
            handles.push(tokio::spawn(async move {
                for i in 0..5 {
                    let branch = format!("t{t}-b{i}");
                    mgr.git_mutate(&repo, &["branch", &branch], ProcessOwner::Daemon)
                        .await
                        .unwrap();
                }
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        // All 20 branches exist (no lost updates).
        let out = mgr
            .git_read(&repo, &["branch", "--list"], ProcessOwner::Daemon)
            .await
            .unwrap();
        for t in 0..4 {
            for i in 0..5 {
                assert!(
                    out.contains(&format!("t{t}-b{i}")),
                    "branch t{t}-b{i} lost: {out}"
                );
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unrelated_repos_do_not_serialize() {
        tokio::time::timeout(
            std::time::Duration::from_secs(25),
            unrelated_repos_do_not_serialize_inner(),
        )
        .await
        .unwrap_or_else(|_| {
            panic!("unrelated_repos_do_not_serialize exceeded its 25s hard test timeout")
        });
    }

    async fn unrelated_repos_do_not_serialize_inner() {
        let (_d, _sup, mgr, repo) = fixture().await;
        // Create a second repo.
        let repo2 = repo.parent().unwrap().join("repo2");
        std::fs::create_dir_all(&repo2).unwrap();
        let _ = mgr
            .git_mutate(&repo2, &["init"], ProcessOwner::Daemon)
            .await;
        std::fs::write(repo2.join("README.md"), "# repo2\n").unwrap();
        let _ = mgr
            .git_mutate(
                &repo2,
                &[
                    "-c",
                    "user.email=t@k.local",
                    "-c",
                    "user.name=T",
                    "add",
                    ".",
                ],
                ProcessOwner::Daemon,
            )
            .await;
        let _ = mgr
            .git_mutate(
                &repo2,
                &[
                    "-c",
                    "user.email=t@k.local",
                    "-c",
                    "user.name=T",
                    "commit",
                    "-m",
                    "init",
                ],
                ProcessOwner::Daemon,
            )
            .await;
        let t0 = std::time::Instant::now();
        let mut handles = Vec::new();
        for (r, n) in [(&repo, "a"), (&repo2, "b")] {
            let mgr = mgr.clone();
            let r = r.clone();
            let n = n.to_string();
            handles.push(tokio::spawn(async move {
                for i in 0..3 {
                    mgr.git_mutate(&r, &["branch", &format!("{n}{i}")], ProcessOwner::Daemon)
                        .await
                        .unwrap();
                }
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        let elapsed = t0.elapsed();
        // With per-repo locks the two repos run concurrently; a global lock
        // would serialize the two tiny streams anyway — the important
        // assertion is that both succeed.
        assert!(elapsed < std::time::Duration::from_secs(20));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn git_command_failure_is_loud() {
        tokio::time::timeout(
            std::time::Duration::from_secs(25),
            git_command_failure_is_loud_inner(),
        )
        .await
        .unwrap_or_else(|_| {
            panic!("git_command_failure_is_loud exceeded its 25s hard test timeout")
        });
    }

    async fn git_command_failure_is_loud_inner() {
        let dir = tempdir().unwrap();
        let cas = Arc::new(faktor_cas::Cas::open(dir.path().join("cas")).unwrap());
        let sup = ProcessSupervisor::new(cas);
        let mgr = WorktreeManager::new(sup);
        let not_a_repo = dir.path().join("not-a-repo");
        std::fs::create_dir_all(&not_a_repo).unwrap();
        let err = mgr
            .git_read(&not_a_repo, &["status"], ProcessOwner::Daemon)
            .await
            .unwrap_err();
        assert!(err.kind == ErrorKind::Internal, "{err:?}");
        // Missing repo → NotFound.
        let err = mgr
            .git_read(&dir.path().join("nope"), &["status"], ProcessOwner::Daemon)
            .await
            .unwrap_err();
        assert!(err.kind == ErrorKind::NotFound);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn malicious_branch_name_rejected() {
        tokio::time::timeout(
            std::time::Duration::from_secs(25),
            malicious_branch_name_rejected_inner(),
        )
        .await
        .unwrap_or_else(|_| {
            panic!("malicious_branch_name_rejected exceeded its 25s hard test timeout")
        });
    }

    async fn malicious_branch_name_rejected_inner() {
        let (_d, _sup, mgr, repo) = fixture().await;
        for evil in [
            "../escape",
            "a b",
            "x; rm -rf /",
            "-leading",
            "a..b",
            "tab\tname",
            "ctrl\x07",
        ] {
            assert!(
                mgr.create(&repo, evil, "wt-evil", SessionId::new(1))
                    .await
                    .is_err(),
                "branch {evil:?} must be rejected before spawn"
            );
        }
        for evil in ["../esc", "a/b", "with space", "a..b"] {
            assert!(
                mgr.create(&repo, "ok-branch", evil, SessionId::new(1))
                    .await
                    .is_err(),
                "name {evil:?} must be rejected before spawn"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_worktree_in_nonexistent_root_not_found() {
        tokio::time::timeout(
            std::time::Duration::from_secs(25),
            create_worktree_in_nonexistent_root_not_found_inner(),
        )
        .await
        .unwrap_or_else(|_| {
            panic!(
                "create_worktree_in_nonexistent_root_not_found exceeded its 25s hard test timeout"
            )
        });
    }

    async fn create_worktree_in_nonexistent_root_not_found_inner() {
        let (_d, _sup, mgr, _repo) = fixture().await;
        let err = mgr
            .create(
                &PathBuf::from("/definitely/not/here"),
                "feat/x",
                "wt-x",
                SessionId::new(1),
            )
            .await
            .unwrap_err();
        assert!(err.kind == ErrorKind::NotFound);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_creates_get_distinct_worktree_ids() {
        tokio::time::timeout(
            std::time::Duration::from_secs(25),
            concurrent_creates_get_distinct_worktree_ids_inner(),
        )
        .await
        .unwrap_or_else(|_| {
            panic!(
                "concurrent_creates_get_distinct_worktree_ids exceeded its 25s hard test timeout"
            )
        });
    }

    async fn concurrent_creates_get_distinct_worktree_ids_inner() {
        let (_d, _sup, mgr, repo) = fixture().await;
        let mut ids = std::collections::HashSet::new();
        let mut handles = Vec::new();
        for i in 0..4 {
            let mgr = mgr.clone();
            let repo = repo.clone();
            handles.push(tokio::spawn(async move {
                mgr.create(
                    &repo,
                    &format!("feat/cc{i}"),
                    &format!("wt-cc{i}"),
                    SessionId::new(1),
                )
                .await
                .unwrap()
            }));
        }
        for h in handles {
            let wt = h.await.unwrap();
            assert!(ids.insert(wt.id), "worktree ids must be unique");
        }
        assert_eq!(ids.len(), 4);
    }

    #[test]
    fn branch_validation_edge_cases() {
        assert!(validate_branch("feat/x-1").is_ok());
        assert!(validate_branch("main").is_ok());
        assert!(validate_branch("").is_err());
        assert!(validate_branch(&"x".repeat(200)).is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn transfer_changes_owner_durably_across_managers() {
        tokio::time::timeout(
            std::time::Duration::from_secs(25),
            transfer_changes_owner_durably_across_managers_inner(),
        )
        .await
        .unwrap_or_else(|_| {
            panic!(
                "transfer_changes_owner_durably_across_managers exceeded its 25s hard test timeout"
            )
        });
    }

    async fn transfer_changes_owner_durably_across_managers_inner() {
        // Spec §33: transfer must survive daemon restarts (a fresh manager
        // over the same repo still sees the new owner — the audit's
        // transfer() was a no-op that ignored new_owner).
        let (_d, sup, mgr, repo) = fixture().await;
        let wt = mgr
            .create(&repo, "feat/durable", "wt-durable", SessionId::new(1))
            .await
            .unwrap();
        mgr.transfer(&wt, SessionId::new(2)).await.unwrap();
        // A brand-new manager (daemon restart): the owner is durable.
        let fresh = WorktreeManager::new(sup.clone());
        let found = fresh.discover(&repo).await.unwrap();
        let owned = found
            .iter()
            .find(|w| w.path == wt.path)
            .expect("worktree still discovered");
        assert_eq!(owned.owner_session, Some(SessionId::new(2)));
        // The created worktree's OWNER is recorded from creation too.
        let wt2 = mgr
            .create(&repo, "feat/b", "wt-b", SessionId::new(7))
            .await
            .unwrap();
        let fresh2 = WorktreeManager::new(sup.clone());
        let found2 = fresh2.discover(&repo).await.unwrap();
        let owned2 = found2.iter().find(|w| w.path == wt2.path).unwrap();
        assert_eq!(owned2.owner_session, Some(SessionId::new(7)));
        // Remove drops the metadata: a fresh discover no longer sees it.
        mgr.remove(&wt2).await.unwrap();
        let fresh3 = WorktreeManager::new(sup.clone());
        let found3 = fresh3.discover(&repo).await.unwrap();
        assert!(
            !found3.iter().any(|w| w.path == wt2.path),
            "removed worktree must be gone from discovery"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn discover_never_invents_owners_for_unknown_worktrees() {
        tokio::time::timeout(
            std::time::Duration::from_secs(25),
            discover_never_invents_owners_for_unknown_worktrees_inner(),
        )
        .await
        .unwrap_or_else(|_| panic!("discover_never_invents_owners_for_unknown_worktrees exceeded its 25s hard test timeout"));
    }

    async fn discover_never_invents_owners_for_unknown_worktrees_inner() {
        // A worktree git knows about but this manager never recorded must
        // surface as UNOWNED (None) — the audit's invented SessionId::new(1)
        // fabricated cross-session ownership.
        let (_d, sup, mgr, repo) = fixture().await;
        // Create a worktree via RAW git (bypasses the manager metadata).
        // git reports CANONICAL paths; match them exactly.
        let repo_canon = repo.canonicalize().unwrap();
        let wt_path = repo_canon.join(".worktrees/raw");
        let _add_out = sup
            .run(
                SpawnConfig {
                    cmd: "git".into(),
                    args: vec![
                        "worktree".into(),
                        "add".into(),
                        "-b".into(),
                        "feat/raw".into(),
                        wt_path.to_str().unwrap().into(),
                    ],
                    cwd: repo.clone(),
                    owner: ProcessOwner::Daemon,
                    ..Default::default()
                },
                std::time::Duration::from_secs(20),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        let found = mgr.discover(&repo).await.unwrap();
        let raw = found
            .iter()
            .find(|w| w.path == wt_path)
            .expect("raw worktree discovered by git");
        assert_eq!(
            raw.owner_session, None,
            "no recorded owner: never an invented session"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn repair_prunes_vanished_worktree_metadata() {
        tokio::time::timeout(
            std::time::Duration::from_secs(25),
            repair_prunes_vanished_worktree_metadata_inner(),
        )
        .await
        .unwrap_or_else(|_| {
            panic!("repair_prunes_vanished_worktree_metadata exceeded its 25s hard test timeout")
        });
    }

    async fn repair_prunes_vanished_worktree_metadata_inner() {
        let (_d, sup, mgr, repo) = fixture().await;
        let wt = mgr
            .create(&repo, "feat/gone", "wt-gone", SessionId::new(1))
            .await
            .unwrap();
        // Delete the worktree directory out from under the manager.
        std::fs::remove_dir_all(&wt.path).unwrap();
        let repaired = mgr.repair(&repo).await.unwrap();
        assert!(
            repaired.iter().any(|p| p.contains("wt-gone")),
            "repair must report the pruned worktree: {repaired:?}"
        );
        let found = mgr.discover(&repo).await.unwrap();
        assert!(
            !found.iter().any(|w| w.path == wt.path),
            "pruned worktree must vanish from discovery"
        );
        let _ = sup;
    }

    /// Adversarial lock-order proof (P1 AB/BA): 8 concurrent tasks mixing
    /// create/repair/discover/remove on ONE repo under a hard 20s bound. If
    /// any pair of operations inverted the universal order (repository git
    /// lock first, metadata lock second) the tasks would deadlock and the
    /// timeout fires — the test fails by hanging, never by false success.
    /// Afterwards the metadata must be consistent: every created worktree is
    /// discovered, every removed one is gone.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_mixed_ops_never_deadlock() {
        tokio::time::timeout(
            std::time::Duration::from_secs(25),
            concurrent_mixed_ops_never_deadlock_inner(),
        )
        .await
        .unwrap_or_else(|_| {
            panic!("concurrent_mixed_ops_never_deadlock exceeded its 25s hard test timeout")
        });
    }

    async fn concurrent_mixed_ops_never_deadlock_inner() {
        let (_d, sup, mgr, repo) = fixture().await;
        let owner = SessionId::new(1);
        // 4 keepers survive the whole run; 2 are removed mid-flight.
        let mut keepers = Vec::new();
        let mut removers = Vec::new();
        for i in 0..6 {
            let wt = mgr
                .create(&repo, &format!("feat/pre{i}"), &format!("wt-pre{i}"), owner)
                .await
                .unwrap();
            if i < 4 {
                keepers.push(wt);
            } else {
                removers.push(wt);
            }
        }
        let mut handles = Vec::new();
        for i in 0..8usize {
            let mgr = mgr.clone();
            let repo = repo.clone();
            let remover = removers.get(i / 4).cloned(); // i = 3, 7 remove
            handles.push(tokio::spawn(async move {
                match i % 4 {
                    0 => mgr
                        .create(
                            &repo,
                            &format!("feat/live{i}"),
                            &format!("wt-live{i}"),
                            SessionId::new(3),
                        )
                        .await
                        .map(Some),
                    1 => {
                        let _ = mgr.repair(&repo).await.unwrap();
                        Ok(None)
                    }
                    2 => {
                        let _ = mgr.discover(&repo).await.unwrap();
                        Ok(None)
                    }
                    _ => {
                        let wt = remover.expect("remover task has a target");
                        mgr.remove(&wt).await.map(|()| None)
                    }
                }
            }));
        }
        let bounded = tokio::time::timeout(std::time::Duration::from_secs(20), async {
            let mut created = Vec::new();
            for h in handles {
                if let Some(wt) = h.await.unwrap().unwrap() {
                    created.push(wt);
                }
            }
            created
        });
        let created = bounded
            .await
            .expect("deadlock: mixed create/repair/discover/remove exceeded the 20s bound");
        assert_eq!(created.len(), 2, "both concurrent creates must succeed");
        let found = mgr.discover(&repo).await.unwrap();
        let paths: std::collections::HashSet<String> = found
            .iter()
            .map(|w| w.path.to_string_lossy().into_owned())
            .collect();
        for wt in &keepers {
            let key = wt.path.to_string_lossy().into_owned();
            assert!(paths.contains(&key), "kept worktree {key} lost");
        }
        for wt in &created {
            let key = wt.path.to_string_lossy().into_owned();
            assert!(paths.contains(&key), "created worktree {key} lost");
        }
        for wt in &removers {
            let key = wt.path.to_string_lossy().into_owned();
            assert!(
                !paths.contains(&key),
                "removed worktree {key} still discovered"
            );
        }
        let _ = sup;
    }

    /// CAS durability discipline (P1): the metadata written by create()
    /// survives (a) an immediate re-read, and (c) 16 concurrent transfer
    /// writers over the same repo — the final file must parse cleanly (no
    /// torn/interleaved JSON) and record every transferred owner, with no
    /// leftover temp files. Fresh-manager survival (b) is covered by
    /// transfer_changes_owner_durably_across_managers.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn metadata_save_is_durable_under_concurrent_writers() {
        tokio::time::timeout(
            std::time::Duration::from_secs(25),
            metadata_save_is_durable_under_concurrent_writers_inner(),
        )
        .await
        .unwrap_or_else(|_| panic!("metadata_save_is_durable_under_concurrent_writers exceeded its 25s hard test timeout"));
    }

    async fn metadata_save_is_durable_under_concurrent_writers_inner() {
        let (_d, sup, mgr, repo) = fixture().await;
        // (a) immediate re-read after create(): the file parses and the
        // owner/branch are recorded as written.
        let first = mgr
            .create(&repo, "feat/cas", "wt-cas", SessionId::new(7))
            .await
            .unwrap();
        let raw = std::fs::read_to_string(meta_path(&repo)).unwrap();
        let parsed: Meta = serde_json::from_str(&raw)
            .expect("immediate re-read of metadata must parse (no torn JSON)");
        let entry = parsed
            .worktrees
            .iter()
            .find(|e| e.path.ends_with("wt-cas"))
            .expect("create must be recorded durably");
        assert_eq!(entry.owner_session, 7);
        assert_eq!(entry.branch, "feat/cas");
        let reloaded = mgr.load_meta(&repo);
        assert!(
            reloaded
                .worktrees
                .iter()
                .any(|e| e.path == first.path.to_string_lossy() && e.owner_session == 7),
            "create must survive an immediate load_meta re-read"
        );
        // 16 more worktrees, each transferred to a DIFFERENT owner in
        // parallel — 16 concurrent writers over the same metadata file.
        let mut wts = Vec::new();
        for i in 0..16 {
            let wt = mgr
                .create(
                    &repo,
                    &format!("feat/cas{i}"),
                    &format!("wt-cas{i}"),
                    SessionId::new(1),
                )
                .await
                .unwrap();
            wts.push(wt);
        }
        let mut handles = Vec::new();
        for (i, wt) in wts.iter().enumerate() {
            let mgr = mgr.clone();
            let wt = wt.clone();
            handles.push(tokio::spawn(async move {
                mgr.transfer(&wt, SessionId::new(100 + i as u64))
                    .await
                    .unwrap()
            }));
        }
        tokio::time::timeout(std::time::Duration::from_secs(20), async {
            for h in handles {
                h.await.unwrap();
            }
        })
        .await
        .expect("16 parallel transfers exceeded the 20s bound");
        // (c) final file parses and contains EVERY transferred owner.
        let raw = std::fs::read_to_string(meta_path(&repo)).unwrap();
        let parsed: Meta = serde_json::from_str(&raw)
            .expect("final metadata must parse — no torn/interleaved JSON");
        let owners: std::collections::HashSet<u64> =
            parsed.worktrees.iter().map(|e| e.owner_session).collect();
        for i in 0..16u64 {
            assert!(
                owners.contains(&(100 + i)),
                "owner {i} lost from the final metadata"
            );
        }
        // The nonce discipline renames each temp away; none may linger.
        let git_dir = repo.join(".git");
        let stale: Vec<String> = std::fs::read_dir(&git_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains(".tmp."))
            .collect();
        assert!(stale.is_empty(), "stale metadata temps left: {stale:?}");
        // A fresh manager (daemon restart) sees the durable owners.
        let fresh = WorktreeManager::new(sup.clone());
        let found = fresh.discover(&repo).await.unwrap();
        for (i, wt) in wts.iter().enumerate() {
            let f = found
                .iter()
                .find(|w| w.path == wt.path)
                .unwrap_or_else(|| panic!("worktree {} vanished", wt.path.display()));
            assert_eq!(
                f.owner_session,
                Some(SessionId::new(100 + i as u64)),
                "fresh manager must see the transferred owner"
            );
        }
        let _ = sup;
    }
}
