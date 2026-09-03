# faktor-snapshot + faktor-git specs (spec §16, §33)

## Part A — faktor-snapshot: native content-addressed checkpoints

Crate: crates/snapshot (faktor-core, faktor-cas, faktor-store, faktor-fs, blake3).

```rust
pub struct CheckpointStore { cas: Arc<Cas>, store: Arc<Store>, fs: Arc<WorkspaceFileService> }
impl CheckpointStore {
    pub fn new(cas, store, fs) -> Self;
    /// Capture before-change content: store original bytes in CAS once
    /// (dedup is free), record Checkpoint{before_hash, after_hash?}.
    pub fn before_write(&self, session: SessionId, path: &str, content: &[u8]) -> Result<FileHash>;
    /// Record the after-write hash (called by the edit engine on success).
    pub fn after_write(&self, session: SessionId, path: &str, after: FileHash, sequence: i64) -> Result<i64>;
    /// Rollback: verify current disk hash == after_hash, then atomically
    /// write before content; mismatch → Error::conflict (never overwrite
    /// unrelated user edits).
    pub fn rollback(&self, identity: &WorkspaceIdentity, checkpoint_id: i64) -> Result<RollbackOutcome>;
    pub fn checkpoints(&self, session: SessionId) -> Result<Vec<CheckpointRow>>;
}
pub enum RollbackOutcome { Restored { path: String, hash: FileHash }, Conflict { path: String, current: FileHash, expected_after: FileHash } }
```
Rules: 10 checkpoints of the same unchanged file consume one CAS blob (assert blob_count). Rollback verifies after_hash first; writes via fs write_atomic (identity-scoped). mark_checkpoint_restored on success.

## Part B — faktor-git: worktree manager + per-repo mutation lock

Crate: crates/git (faktor-core, faktor-terminal, tokio). Uses ProcessSupervisor for every git invocation (no orphans).

```rust
pub struct WorktreeManager { supervisor: Arc<ProcessSupervisor>, repos: Mutex<HashMap<PathBuf, Arc<RwLock<()>>>> }
impl WorktreeManager {
    pub fn new(supervisor: Arc<ProcessSupervisor>) -> Self;
    pub fn create(&self, workspace_root: &Path, branch: &str, name: &str) -> Result<Worktree>;
    pub fn discover(&self, workspace_root: &Path) -> Result<Vec<Worktree>>; // `git worktree list --porcelain`
    pub fn validate(&self, wt: &Worktree) -> Result<Validation>;           // dir exists + .git file + clean state
    pub fn repair(&self, wt: &Worktree) -> Result<()>;                     // re-register missing metadata
    pub fn remove(&self, wt: &Worktree) -> Result<()>;
    pub fn transfer(&self, wt: &Worktree, new_session: SessionId) -> Result<()>; // deliberate ownership move
    pub fn mutate<F, T>(&self, repo: &Path, f: F) -> T where F: FnOnce() -> T;  // per-repo mutation lock
}
pub struct Worktree { pub id: WorktreeId, pub workspace_root: PathBuf, pub path: PathBuf, pub branch: String, pub owner_session: SessionId }
```
Rules:
- `mutate` takes the per-repository lock (RwLock write); read-only ops (discover/validate) use the read lock and stay concurrent. No global git lock across unrelated repos.
- Every git command runs through supervisor with an owner (Session or Daemon), a deadline, and env `GIT_TERMINAL_PROMPT=0`.
- Adversarial tests (all real git in tempdirs; git is available):
  1. create_worktree_has_valid_git_metadata (.git file + branch)
  2. discover_finds_created_worktrees (incl. after reopen)
  3. validate_rejects_missing_dir_and_accepts_healthy
  4. remove_removes_and_releases_lock
  5. transfer_changes_owner_durably (roundtrip via discover/validate)
  6. mutate_lock_serializes_writes_per_repo (2 threads hammering `git branch` mutations: no interleaving corruption)
  7. unrelated_repos_do_not_serialize (two repos, parallel mutate both, wall time < serial)
  8. git_command_failure_is_loud (git in non-repo → error with stderr)
  9. malicious_branch_name_rejected (branch with `../` or spaces → Malformed before spawn)
  10. orphan_safety (kill supervisor child via kill_all_for — no git processes left: assert via ps)
  11. create_worktree_in_nonexistent_root_not_found
  12. concurrent_creates_get_distinct_worktree_ids
- Note: sandbox/traversal checks are faktor-sandbox's job; here only arg construction must be injection-safe (no shell; exec args directly; `--` separators).

Build/test each crate green, zero warnings. Do NOT modify other crates. Do NOT commit.
