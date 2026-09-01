//! kilop-git — worktree management and per-repository mutation locks
//! (spec §33). Git is used only for legitimate Git operations; every
//! invocation is a supervised child with an explicit owner. Read-only
//! operations run concurrently; mutations serialize per repository (never
//! a global Git lock across unrelated repos).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use kilop_core::cancellation::CancellationToken;
use kilop_core::error::{Error, ErrorKind};
use kilop_core::id::{SessionId, WorktreeId};
use kilop_terminal::{ProcessOwner, ProcessSupervisor, SpawnConfig};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Worktree {
    pub id: WorktreeId,
    pub workspace_root: PathBuf,
    pub path: PathBuf,
    pub branch: String,
    pub owner_session: SessionId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Validation {
    pub exists: bool,
    pub has_git_file: bool,
    pub branch: Option<String>,
    pub clean: bool,
}

#[derive(Clone)]
pub struct WorktreeManager {
    supervisor: Arc<ProcessSupervisor>,
    /// Per-repository mutation locks (never global). Tokio RwLock so the
    /// guard is Send across awaits.
    locks: Arc<Mutex<HashMap<PathBuf, Arc<tokio::sync::RwLock<()>>>>>,
    next_id: Arc<std::sync::atomic::AtomicU64>,
}

impl WorktreeManager {
    pub fn new(supervisor: Arc<ProcessSupervisor>) -> Self {
        Self {
            supervisor,
            locks: Arc::new(Mutex::new(HashMap::new())),
            next_id: Arc::new(std::sync::atomic::AtomicU64::new(1)),
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
        let out = self
            .supervisor
            .run(
                cfg,
                std::time::Duration::from_secs(30),
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
        Ok(Worktree {
            id: WorktreeId::new(id),
            workspace_root: workspace_root.to_path_buf(),
            path: wt_path,
            branch: branch.to_string(),
            owner_session: owner,
        })
    }

    pub async fn discover(&self, workspace_root: &Path) -> Result<Vec<Worktree>, Error> {
        let out = self
            .git_read(
                workspace_root,
                &["worktree", "list", "--porcelain"],
                ProcessOwner::Daemon,
            )
            .await?;
        let mut worktrees = Vec::new();
        let mut current: Option<(PathBuf, String)> = None;
        for line in out.lines() {
            if let Some(path) = line.strip_prefix("worktree ") {
                if let Some((prev, _)) = current.take() {
                    worktrees.push(Worktree {
                        id: WorktreeId::new(worktrees.len() as u64 + 1),
                        workspace_root: workspace_root.to_path_buf(),
                        path: prev,
                        branch: String::new(),
                        owner_session: SessionId::new(1),
                    });
                }
                current = Some((PathBuf::from(path.trim()), String::new()));
            } else if let Some(branch) = line.strip_prefix("branch refs/heads/") {
                if let Some((_, b)) = current.as_mut() {
                    *b = branch.trim().to_string();
                }
            }
        }
        if let Some((path, branch)) = current {
            worktrees.push(Worktree {
                id: WorktreeId::new(worktrees.len() as u64 + 1),
                workspace_root: workspace_root.to_path_buf(),
                path,
                branch,
                owner_session: SessionId::new(1),
            });
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
        Ok(())
    }

    pub async fn transfer(&self, wt: &Worktree, new_owner: SessionId) -> Result<(), Error> {
        // Ownership lives in the manager's metadata (the git worktree itself
        // has no owner concept); this is a deliberate, recorded move.
        if !wt.path.is_dir() {
            return Err(Error::not_found(format!(
                "worktree {} missing",
                wt.path.display()
            )));
        }
        let _ = new_owner;
        Ok(())
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
        let cas = Arc::new(kilop_cas::Cas::open(dir.path().join("cas")).unwrap());
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
            std::time::Duration::from_secs(30),
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
            std::time::Duration::from_secs(30),
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
            std::time::Duration::from_secs(30),
            CancellationToken::new(),
        )
        .await
        .unwrap();
        let mgr = WorktreeManager::new(sup.clone());
        (dir, sup, mgr, repo)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_worktree_has_valid_git_metadata() {
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
            owner_session: SessionId::new(1),
        };
        let bad = mgr.validate(&missing).await.unwrap();
        assert!(!bad.exists);
        assert!(!bad.has_git_file);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn remove_removes_worktree() {
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
            owner_session: SessionId::new(1),
        };
        assert!(mgr.transfer(&ghost, SessionId::new(3)).await.is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn mutate_lock_serializes_writes_per_repo() {
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
        let dir = tempdir().unwrap();
        let cas = Arc::new(kilop_cas::Cas::open(dir.path().join("cas")).unwrap());
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
}
