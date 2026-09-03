# faktor-fs + faktor-edit specs (spec §17, §18, §21)

## Part A — faktor-fs: file service and watcher

Crate: crates/fs (faktor-core, notify, tokio).

```rust
pub struct FileData { pub path: PathBuf, pub bytes: Vec<u8>, pub hash: FileHash, pub size: usize, pub truncated: bool }
pub struct FsEvent { pub workspace_id: WorkspaceId, pub path: PathBuf, pub kind: FsEventKind } // Created/Modified/Removed
pub struct WorkspaceFileService { /* registry of workspace → handle */ }
impl WorkspaceFileService {
    pub fn new() -> Arc<Self>;
    pub fn open(&self, workspace_id: WorkspaceId, root: PathBuf) -> Result<WorkspaceHandle>;
    pub fn close(&self, workspace_id: WorkspaceId); // idle unload (spec §21)
}
pub struct WorkspaceHandle { root, workspace_id, watcher_rx: mpsc::Receiver<FsEvent> }
impl WorkspaceHandle {
    pub fn workspace_id(&self) -> WorkspaceId;
    pub fn resolve(&self, rel: &Path) -> Result<PathBuf>;  // canonicalize + prefix check: traversal/symlink-safe
    pub fn read(&self, rel: &Path, max_bytes: usize) -> Result<FileData>;  // truncated flag when over
    pub fn read_slice(&self, rel: &Path, offset: u64, len: usize) -> Result<FileData>;
    pub fn write_atomic(&self, rel: &Path, bytes: &[u8]) -> Result<FileHash>; // temp+fsync+rename
    pub fn stat(&self, rel: &Path) -> Result<FileMeta>;
    pub fn exists(&self, rel: &Path) -> bool;
    pub fn list(&self, rel: &Path, max_entries: usize) -> Result<Vec<FileMeta>>;
    pub fn events(&self) -> &mpsc::Receiver<FsEvent>;
    pub fn root(&self) -> &Path;
}
pub struct FileMeta { pub path: PathBuf, pub size: u64, pub modified_ms: i64 }
```
Rules: every call takes explicit rel path resolved under root; `resolve` rejects `..` escapes and symlinks pointing outside root (canonicalize the PARENT, then append). Watcher via notify (add dep `notify = "8"`): forward events with workspace_id. read is bounded by max_bytes (default 4MB) with truncated flag; read_slice for paging big files.

## Part B — faktor-edit: transactional patch engine (spec §17, §18)

Crate: crates/edit (faktor-core, faktor-fs, tree-sitter 0.27, tree-sitter-rust, tree-sitter-python, blake3).

```rust
pub enum EditOp {
    Range { start: usize, end: usize, replacement: String },          // byte offsets
    SearchReplace { before: String, after: String },                  // must match uniquely
    BoundedRegion { anchor: String, region_start: usize, region_end: usize, replacement: String },
}
pub struct EditRequest { pub path: String, pub expected_hash: FileHash, pub ops: Vec<EditOp> }
pub struct EditOutcome { pub new_hash: FileHash, pub ops_applied: usize, pub suspicious: bool,
    pub parse_error: Option<String> }
pub struct EditEngine { service: Arc<WorkspaceFileService> }
impl EditEngine {
    pub fn new(service: Arc<WorkspaceFileService>) -> Self;
    /// Optimistic + versioned: expected_hash must match current content or
    /// the edit is REJECTED (Error::conflict). Apply ops in hierarchy order;
    /// atomic write; verify after-write hash; parse-before-accept.
    pub fn apply(&self, identity: &WorkspaceIdentity, req: EditRequest, mode: RepairMode) -> Result<EditOutcome>;
}
pub enum RepairMode { Rollback, AllowModelRepair }
```
Rules:
1. expected_hash mismatch → Conflict error BEFORE any write. Never apply an old patch to unexpected contents.
2. Ops applied in order; Range clamps to file bounds (out-of-bounds → Malformed). SearchReplace requires exactly one match (0 → Malformed with the count; >1 → Conflict listing ambiguity). BoundedRegion: anchor must match uniquely, region bounds clamped.
3. Whole-file regeneration is NOT an op here (last resort lives in the agent).
4. Atomic write via fs.write_atomic.
5. Parse-before-accept: for .rs (tree-sitter-rust) and .py (tree-sitter-python) files, parse the ORIGINAL; if it parsed cleanly and the EDIT result has syntax errors → mark `suspicious = true` with the first error; per RepairMode: Rollback → do not write, return Err(Malformed); AllowModelRepair → write but flag suspicious. Unknown languages skip the check (not suspicious).
6. UTF-8: Range offsets are byte offsets into a validated-UTF-8 buffer; splitting inside a codepoint → Malformed (never panic).

## Adversarial tests (name every one)
Part A:
1. traversal_escape_rejected (../, /abs, encoded %2e%2e — all rejected)
2. symlink_escape_rejected (symlink inside root → outside; resolve must reject)
3. read_bounded_sets_truncated_flag (10MB file, max 1MB)
4. read_slice_roundtrip_paging (read whole via slices == read whole)
5. write_atomic_replaces_and_hashes (crash-safe: temp not left behind; write 100x same path)
6. watcher_delivers_events_with_workspace_id (create/modify/remove; may need a small sleep — keep <2s)
7. missing_file_not_found
8. concurrent_writes_same_path_are_atomic (8 threads, file always a complete variant)
9. list_bounded_and_sorted
10. unicode_and_binary_paths

Part B:
1. hash_mismatch_rejected_before_write (file changed on disk → Conflict, file untouched)
2. range_edit_applies_and_hashes
3. search_replace_unique_ok / zero_matches_malformed / multiple_matches_conflict
4. out_of_bounds_range_malformed
5. split_codepoint_offsets_malformed_never_panic
6. rust_parse_error_marks_suspicious_and_rolls_back (edit that breaks syntax, RepairMode::Rollback → error, file unchanged)
7. rust_parse_error_allow_repair_writes_and_flags (file changed, suspicious=true)
8. valid_edit_stays_not_suspicious
9. py_parse_check_works (python grammar)
10. unknown_language_skips_parse_check
11. adversarial_ops_in_one_request_partial_failure (op2 fails after op1 → NO partial write: validate ALL ops against a copy first, then single atomic write)
12. huge_edit_bounded (10MB replacement → Oversized or works within bounds, never OOM)
13. concurrent_edits_same_file_conflict_one (two engines, same expected_hash, one wins, other Conflicts)

Build/test each crate green, zero warnings. Do NOT modify other crates. Do NOT commit.
