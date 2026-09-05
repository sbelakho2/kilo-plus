//! Controlled child-worktree merge + reviewer-worktree semantics
//! (audits 69/70/98/99).
//!
//! A mutating child owns an ISOLATED worktree (wave-12). Its changes must
//! NOT auto-land in the parent: when the child reaches terminal success the
//! executor stages a durable [`ChangeSet`] (diff of the base snapshot the
//! child started from vs its worktree now, capped at [`MAX_CHANGES`] — a
//! larger diff is a typed [`ExecError::Oversized`], never a silent
//! truncation), the parent explicitly approves or rejects EVERY changed
//! file, and only approved files are merged into the parent worktree, each
//! through the wave-10 commit-time CAS primitives (`faktor_fs`):
//!
//! - expected = the base-snapshot hash of the parent path; a parent file
//!   that changed since the base snapshot is a CONFLICT — never overwritten;
//! - a file the parent lacked at the base snapshot must still be absent
//!   (exclusive create); a parent-side creation since then is a conflict;
//! - replay is idempotent: a destination that already holds the child
//!   digest is [`faktor_fs::CasMergeResult::AlreadyCurrent`] (a crashed
//!   earlier run applied it) and is treated as merged, never an error.
//!
//! Durability (audit 99: record-first, notify-after): the durable
//! [`MergeEnvelope`] row (kind [`KIND_MERGE`], "status applied|failed")
//! plus the durable approved/rejected decision rows (kind
//! [`KIND_MERGE_PART`]) are written BEFORE any file is touched, and the
//! outcome (merged/conflicts parts + the finalized envelope) only after
//! every apply. A crash between the record and the applies, or between
//! applies, is replayed by calling [`OrchestratorRuntime::approve_and_merge`]
//! again with the same decision — the CAS makes each apply idempotent.
//!
//! Reviewer worktrees (audit 70): [`OrchestratorRuntime::spawn_reviewer`]
//! creates a FRESH isolated worktree that is a bounded, atomic-per-file
//! copy of the CURRENT parent state (never the reviewed child's dirty
//! worktree), and records the copy as the reviewer's base snapshot
//! (`base_snapshot_id` on the reviewer's durable child row).
//!
//! All rows live in the PARENT session's fact space under the run id, so
//! they survive manager reopens exactly like the registry rows do; the
//! read paths scan pages and refuse a partial view loudly.

use std::collections::HashMap;
use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};

use faktor_core::hash::FileHash;
use faktor_core::id::{SessionId, WorkspaceId};
use faktor_fs::CasMergeResult;
use serde::{Deserialize, Serialize};

use super::*;

/// Hard cap on one change set (audit 98). A child diff beyond this bound is
/// a typed [`ExecError::Oversized`] and nothing is staged — the merge never
/// silently truncates; the parent must handle the oversized set explicitly.
pub const MAX_CHANGES: usize = 2000;
/// Hard cap on ONE base-snapshot/copy tree walk. Trees beyond the cap fail
/// loudly (typed Oversized) — snapshots never silently drop files.
pub const MAX_BASE_ENTRIES: usize = 100_000;
/// Total-byte bound of one reviewer tree copy (audit 70: bounded copy with
/// typed Oversized beyond; nothing is ever skipped silently).
pub const MAX_REVIEW_COPY_BYTES: u64 = 512 * 1024 * 1024;
/// Decision paths must stay within this many characters (bounded inputs).
pub const MAX_DECISION_PATH_CHARS: usize = 512;

pub(crate) const KIND_BASE: &str = "orchestrator_base";
pub(crate) const KIND_CS: &str = "orchestrator_cs";
pub(crate) const KIND_MERGE: &str = "orchestrator_merge";
pub(crate) const KIND_MERGE_PART: &str = "orchestrator_merge_part";

/// Payload budget of ONE chunk row (values are capped at 4096 bytes by the
/// fact store; staying under this keeps one row per chunk with headroom).
const CHUNK_BUDGET: usize = 3000;
/// Max pages of one bounded fact scan (200 rows per page) before the scan
/// refuses loudly instead of returning a partial view.
const MAX_FACT_PAGES: usize = 256;

/// One changed file: the child's current hash (`None` = the child deleted
/// it) and the parent path's hash at the base snapshot (`None` = the parent
/// had no such file then; the merge is an exclusive create).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChangeEntry {
    pub path: PathBuf,
    #[serde(default)]
    pub child_hash: Option<FileHash>,
    #[serde(default)]
    pub base_hash: Option<FileHash>,
}

/// The staged merge candidate (audit 98) — the STRUCTURED result of a
/// finished child (audit 69): files + digests, stored durably, replacing
/// prose-only results.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChangeSet {
    pub child_id: String,
    /// The durable base snapshot (id) the diff was computed against.
    pub base_id: String,
    pub files: Vec<ChangeEntry>,
    pub created_ms: i64,
}

impl ChangeSet {
    /// Deterministic change-set id (one per child base — re-staging after a
    /// crash re-upserts the SAME rows, so it is idempotent).
    pub fn id(&self) -> String {
        format!("{}-cs", self.base_id)
    }
}

/// Durable merge record (audit 99): `merge_records(child_id, seq, status
/// applied|failed, details)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MergeStatus {
    /// Every approved file applied with no conflicts.
    Applied,
    /// Conflicts were surfaced (or the record is the crash-safe in-flight
    /// marker written BEFORE any apply: `finished_ms: None`).
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MergeEnvelope {
    pub seq: u64,
    pub child_id: String,
    pub cs_id: String,
    pub status: MergeStatus,
    pub approved_count: usize,
    pub rejected_count: usize,
    pub merged_count: usize,
    pub conflict_count: usize,
    pub created_ms: i64,
    /// Set only once the apply phase fully finished; `None` = in-flight
    /// (crash-safe replay resumes from the durable decision rows).
    pub finished_ms: Option<i64>,
    /// Bounded human-readable detail (conflict summary / rejection cause).
    pub details: String,
}

impl MergeEnvelope {
    pub fn in_flight(&self) -> bool {
        self.finished_ms.is_none()
    }
}

/// The result of one controlled merge (audit 99 signature).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeOutcome {
    pub merged: Vec<PathBuf>,
    pub rejected: Vec<PathBuf>,
    pub conflicts: Vec<(PathBuf, String)>,
}

/// The structured result of one child (audit 69): durable child row + the
/// staged merge candidate (when one exists) + a summary.
#[derive(Debug, Clone)]
pub struct ChildResult {
    pub child: ChildRuntime,
    pub change_set: Option<ChangeSet>,
    pub summary: String,
    pub merges: Vec<MergeEnvelope>,
}

// ------------------------------------------------------------------ rows

/// All facts of a session via a bounded page walk; a scan that cannot end
/// within MAX_FACT_PAGES refuses loudly (never a silently partial read).
pub(crate) fn scan_facts(
    handle: &faktor_session::SessionHandle,
) -> Result<Vec<(String, String, String)>, ExecError> {
    let mut out = Vec::new();
    let mut after: Option<(i64, String, String)> = None;
    let mut pages = 0usize;
    loop {
        let page = handle
            .memory_facts_page(after.as_ref(), 200)
            .map_err(|e| ExecError::Internal(format!("fact page scan: {}", e.message)))?;
        pages += 1;
        out.extend(page.facts);
        match page.cursor {
            Some(c) => after = Some(c),
            None => return Ok(out),
        }
        if pages >= MAX_FACT_PAGES {
            return Err(ExecError::Oversized(format!(
                "memory-fact scan of session {} exceeded {MAX_FACT_PAGES} pages; refusing a partial merge read",
                handle.id()
            )));
        }
    }
}

fn parent_handle(
    manager: &Arc<faktor_session::SessionManager>,
    parent: SessionId,
) -> Result<faktor_session::SessionHandle, ExecError> {
    manager
        .get_session(parent)?
        .ok_or_else(|| ExecError::NotFound(format!("parent session {parent}")))
}

/// Write one chunked durable row set: `header` under `key`, the chunks
/// under `key/cNNN`. Idempotent (upserts overwrite) — a crash mid-write
/// leaves partial rows that a re-run overwrites in full.
fn put_chunks(
    handle: &faktor_session::SessionHandle,
    kind: &str,
    key: &str,
    header: &str,
    chunks: &[String],
) -> Result<(), ExecError> {
    if key.len() > 480 {
        return Err(ExecError::Oversized(format!(
            "row key of {} bytes",
            key.len()
        )));
    }
    handle.upsert_memory_fact(kind, key, header).map_err(|e| {
        ExecError::Internal(format!("row header write {kind}/{key}: {}", e.message))
    })?;
    for (i, chunk) in chunks.iter().enumerate() {
        handle
            .upsert_memory_fact(kind, &format!("{key}/c{:03}", i + 1), chunk)
            .map_err(|e| {
                ExecError::Internal(format!("row chunk write {kind}/{key}: {}", e.message))
            })?;
    }
    Ok(())
}

fn read_chunks(
    handle: &faktor_session::SessionHandle,
    kind: &str,
    key: &str,
) -> Result<Option<(serde_json::Value, Vec<String>)>, ExecError> {
    let mut header: Option<serde_json::Value> = None;
    let mut chunks: Vec<(usize, String)> = Vec::new();
    for (k, kk, v) in scan_facts(handle)? {
        if k != kind {
            continue;
        }
        if kk == key {
            header = Some(serde_json::from_str(&v).map_err(|e| {
                ExecError::Internal(format!("row header decode {kind}/{key}: {e}"))
            })?);
        } else if let Some(rest) = kk.strip_prefix(key).and_then(|r| r.strip_prefix("/c")) {
            let idx: usize = rest
                .parse()
                .map_err(|_| ExecError::Internal(format!("hostile chunk key {kind}/{kk}")))?;
            chunks.push((idx, v));
        }
    }
    let Some(header) = header else {
        return Ok(None);
    };
    chunks.sort_by_key(|(i, _)| *i);
    let n = header.get("chunks").and_then(|c| c.as_u64()).unwrap_or(0) as usize;
    if chunks.len() != n {
        return Err(ExecError::Internal(format!(
            "chunked row {kind}/{key} is incomplete ({} of {n} chunks; crash mid-write — re-run overwrites it)",
            chunks.len()
        )));
    }
    Ok(Some((header, chunks.into_iter().map(|(_, v)| v).collect())))
}

/// Generic chunker: serializes `items` into ≤CHUNK_BUDGET rows (at least
/// one row, `[]` when empty).
fn pack_chunks<T: Serialize>(items: &[T]) -> Result<Vec<String>, ExecError> {
    let mut chunks: Vec<String> = Vec::new();
    let mut cur: Vec<serde_json::Value> = Vec::new();
    let mut cur_bytes = 0usize;
    for item in items {
        let v = serde_json::to_value(item)
            .map_err(|e| ExecError::Internal(format!("chunk serialization: {e}")))?;
        let s = serde_json::to_string(&v)
            .map_err(|e| ExecError::Internal(format!("chunk serialization: {e}")))?;
        if !cur.is_empty() && cur_bytes + s.len() + 2 > CHUNK_BUDGET {
            let text = serde_json::to_string(&cur)
                .map_err(|e| ExecError::Internal(format!("chunk serialization: {e}")))?;
            chunks.push(text);
            cur.clear();
            cur_bytes = 0;
        }
        cur.push(v);
        cur_bytes += s.len() + 2;
    }
    let text = serde_json::to_string(&cur)
        .map_err(|e| ExecError::Internal(format!("chunk serialization: {e}")))?;
    chunks.push(text);
    Ok(chunks)
}

fn unpack_chunks<T: for<'de> Deserialize<'de>>(chunks: &[String]) -> Result<Vec<T>, ExecError> {
    let mut out = Vec::new();
    for c in chunks {
        let batch: Vec<T> = serde_json::from_str(c)
            .map_err(|e| ExecError::Internal(format!("row chunk decode: {e}")))?;
        out.extend(batch);
    }
    Ok(out)
}

// ------------------------------------------------------- base snapshots

fn base_key(run: &str, child_id: &str, map: &str) -> String {
    format!("{run}/{child_id}/base/{map}")
}

pub(crate) fn base_id_of(child_id: &str) -> String {
    format!("base-{child_id}")
}

/// Record one base map (path -> hash) of a child's spawn-time world under
/// the parent session's fact space. Idempotent re-upsert per
/// (run, child, map). Used for the PARENT map (merge CAS anchors) and the
/// START map (what the child's own worktree contained at spawn).
pub(crate) fn put_base_map(
    manager: &Arc<faktor_session::SessionManager>,
    parent: SessionId,
    run: &str,
    child_id: &str,
    map: &str,
    entries: &[(PathBuf, FileHash)],
) -> Result<(), ExecError> {
    if entries.len() > MAX_BASE_ENTRIES {
        return Err(ExecError::Oversized(format!(
            "base map of {child_id} has {} entries (cap {MAX_BASE_ENTRIES})",
            entries.len()
        )));
    }
    let handle = parent_handle(manager, parent)?;
    let now = manager.now_ms();
    let header = serde_json::json!({
        "child_id": child_id,
        "map": map,
        "base_id": base_id_of(child_id),
        "chunks": pack_chunks(entries)?.len(),
        "created_ms": now,
    });
    let header = serde_json::to_string(&header)
        .map_err(|e| ExecError::Internal(format!("base header: {e}")))?;
    let chunks = pack_chunks(entries)?;
    put_chunks(
        &handle,
        KIND_BASE,
        &base_key(run, child_id, map),
        &header,
        &chunks,
    )
}

/// Read one recorded base map; `Ok(None)` when the map was never recorded
/// (e.g. an empty start tree that was skipped at spawn).
pub(crate) fn read_base_map(
    manager: &Arc<faktor_session::SessionManager>,
    parent: SessionId,
    run: &str,
    child_id: &str,
    map: &str,
) -> Result<Option<Vec<(PathBuf, FileHash)>>, ExecError> {
    let handle = parent_handle(manager, parent)?;
    let Some((_header, chunks)) = read_chunks(&handle, KIND_BASE, &base_key(run, child_id, map))?
    else {
        return Ok(None);
    };
    let rows: Vec<[String; 2]> = unpack_chunks(&chunks)?;
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let path = validate_rel_path_str(&row[0])?;
        let hash = FileHash::from_hex(&row[1]).ok_or_else(|| {
            ExecError::Internal(format!("base row of {child_id} carries a hostile hash"))
        })?;
        out.push((path, hash));
    }
    Ok(Some(out))
}

// ------------------------------------------------------------ change set

fn cs_key(run: &str, child_id: &str, cs_id: &str) -> String {
    format!("{run}/{child_id}/cs/{cs_id}")
}

/// Store (or idempotently re-store) the staged change set of one child.
pub(crate) fn put_change_set(
    manager: &Arc<faktor_session::SessionManager>,
    parent: SessionId,
    run: &str,
    cs: &ChangeSet,
) -> Result<(), ExecError> {
    if cs.files.len() > MAX_CHANGES {
        return Err(ExecError::Oversized(format!(
            "change set of {} has {} files (cap {MAX_CHANGES})",
            cs.child_id,
            cs.files.len()
        )));
    }
    let handle = parent_handle(manager, parent)?;
    let key = cs_key(run, &cs.child_id, &cs.id());
    let header = serde_json::json!({
        "child_id": cs.child_id,
        "base_id": cs.base_id,
        "files": cs.files.len(),
        "chunks": pack_chunks(&cs.files)?.len(),
        "created_ms": cs.created_ms,
    });
    let header = serde_json::to_string(&header)
        .map_err(|e| ExecError::Internal(format!("change-set header: {e}")))?;
    let chunks = pack_chunks(&cs.files)?;
    put_chunks(&handle, KIND_CS, &key, &header, &chunks)
}

/// Read the stored change set of one child (by change-set id).
pub(crate) fn read_change_set(
    manager: &Arc<faktor_session::SessionManager>,
    parent: SessionId,
    run: &str,
    child_id: &str,
    cs_id: &str,
) -> Result<ChangeSet, ExecError> {
    let handle = parent_handle(manager, parent)?;
    let key = cs_key(run, child_id, cs_id);
    let Some((header, chunks)) = read_chunks(&handle, KIND_CS, &key)? else {
        return Err(ExecError::NotFound(format!(
            "change set {cs_id} of child {child_id}"
        )));
    };
    let files: Vec<ChangeEntry> = unpack_chunks(&chunks)?;
    for f in &files {
        // Defense in depth: stored paths must be sane relative paths.
        validate_rel_path_str(&f.path.to_string_lossy())?;
    }
    if files.len() > MAX_CHANGES {
        return Err(ExecError::Internal(format!(
            "stored change set {cs_id} holds {} files (cap {MAX_CHANGES})",
            files.len()
        )));
    }
    Ok(ChangeSet {
        child_id: header
            .get("child_id")
            .and_then(|c| c.as_str())
            .unwrap_or(child_id)
            .to_string(),
        base_id: header
            .get("base_id")
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string(),
        files,
        created_ms: header
            .get("created_ms")
            .and_then(|c| c.as_i64())
            .unwrap_or(0),
    })
}

fn cs_id_of(base_id: &str) -> String {
    format!("{base_id}-cs")
}

// -------------------------------------------------------------- decision

/// Validate one decision path: relative, sane components, bounded length.
/// Anything else (absolute, `..`, `.`, empty, overlong, control bytes) is a
/// typed [`ExecError::InvalidApproval`] — such a path is NEVER applied and
/// NEVER resolved against the filesystem.
pub fn validate_rel_path_str(s: &str) -> Result<PathBuf, ExecError> {
    if s.is_empty() {
        return Err(ExecError::InvalidApproval("empty path".into()));
    }
    if s.chars().count() > MAX_DECISION_PATH_CHARS {
        return Err(ExecError::InvalidApproval(format!(
            "path of {} characters exceeds {MAX_DECISION_PATH_CHARS}",
            s.chars().count()
        )));
    }
    let p = PathBuf::from(s);
    for c in p.components() {
        match c {
            Component::Normal(_) => {}
            Component::CurDir => {
                return Err(ExecError::InvalidApproval(format!(
                    "path {s:?} contains a '.' component; decisions must name change-set paths exactly"
                )));
            }
            _ => {
                return Err(ExecError::InvalidApproval(format!(
                    "path {s:?} is not a plain relative path (traversal/absolute rejected)"
                )));
            }
        }
    }
    if p.as_os_str().is_empty() {
        return Err(ExecError::InvalidApproval("empty path".into()));
    }
    Ok(p)
}

fn rel_str(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

fn find_entry(cs: &ChangeSet, wanted: &Path) -> Result<Option<usize>, ExecError> {
    if let Some(i) = cs.files.iter().position(|e| e.path == wanted) {
        return Ok(Some(i));
    }
    // Exact membership failed: distinguish a CASE variant of a staged path
    // (a friendly typed error) from a path that is simply not part of the
    // change set.
    let wanted_lc = rel_str(wanted).to_lowercase();
    for e in &cs.files {
        if rel_str(&e.path).to_lowercase() == wanted_lc {
            return Err(ExecError::InvalidApproval(format!(
                "path {:?} differs in case from the staged path {:?}",
                wanted.display(),
                e.path.display()
            )));
        }
    }
    Ok(None)
}

/// Validate an approval/rejection decision against the change set. Every
/// changed file must be decided (approved XOR rejected): a path that is
/// neither is an explicit error and NOTHING is merged. Returns the sorted,
/// deduplicated approved and rejected lists.
pub fn validate_decision(
    cs: &ChangeSet,
    approved: &[PathBuf],
    rejected: &[PathBuf],
) -> Result<(Vec<PathBuf>, Vec<PathBuf>), ExecError> {
    let mut approved_set: Vec<PathBuf> = Vec::new();
    let mut rejected_set: Vec<PathBuf> = Vec::new();
    let mut seen_approved: HashSet<PathBuf> = HashSet::new();
    let mut seen_rejected: HashSet<PathBuf> = HashSet::new();
    for p in approved {
        let p = validate_rel_path_str(&rel_str(p))?;
        if seen_rejected.contains(&p) {
            return Err(ExecError::InvalidApproval(format!(
                "path {} is both approved and rejected",
                p.display()
            )));
        }
        if !seen_approved.insert(p.clone()) {
            return Err(ExecError::InvalidApproval(format!(
                "path {} appears twice in the approved list",
                p.display()
            )));
        }
        match find_entry(cs, &p)? {
            Some(_) => approved_set.push(p),
            None => {
                return Err(ExecError::InvalidApproval(format!(
                    "path {} is not part of the change set (nothing was merged)",
                    p.display()
                )));
            }
        }
    }
    for p in rejected {
        let p = validate_rel_path_str(&rel_str(p))?;
        if seen_approved.contains(&p) {
            return Err(ExecError::InvalidApproval(format!(
                "path {} is both approved and rejected",
                p.display()
            )));
        }
        if !seen_rejected.insert(p.clone()) {
            return Err(ExecError::InvalidApproval(format!(
                "path {} appears twice in the rejected list",
                p.display()
            )));
        }
        match find_entry(cs, &p)? {
            Some(_) => rejected_set.push(p),
            None => {
                return Err(ExecError::InvalidApproval(format!(
                    "path {} is not part of the change set (nothing was merged)",
                    p.display()
                )));
            }
        }
    }
    approved_set.sort();
    rejected_set.sort();
    // The parent must decide EVERY changed file — no silent default.
    let mut undecided: Vec<String> = cs
        .files
        .iter()
        .map(|e| rel_str(&e.path))
        .filter(|s| {
            let p = PathBuf::from(s);
            !approved_set.contains(&p) && !rejected_set.contains(&p)
        })
        .collect();
    undecided.sort();
    if !undecided.is_empty() {
        let shown: Vec<String> = undecided.iter().take(8).cloned().collect();
        let more = if undecided.len() > shown.len() {
            format!(" (+{} more)", undecided.len() - shown.len())
        } else {
            String::new()
        };
        return Err(ExecError::UndecidedPaths(format!(
            "{} unchanged file(s) were neither approved nor rejected: {}{} — decide every changed file",
            undecided.len(),
            shown.join(", "),
            more
        )));
    }
    Ok((approved_set, rejected_set))
}

// ---------------------------------------------------------- merge parts

fn part_key(run: &str, child_id: &str, cs_id: &str, seq: u64, part: &str) -> String {
    format!("{run}/{child_id}/merge/{cs_id}/{seq}/part/{part}")
}

fn envelope_key(run: &str, child_id: &str, cs_id: &str, seq: u64) -> String {
    format!("{run}/{child_id}/merge/{cs_id}/{seq}")
}

#[allow(clippy::too_many_arguments)]
fn put_part(
    manager: &Arc<faktor_session::SessionManager>,
    parent: SessionId,
    run: &str,
    child_id: &str,
    cs_id: &str,
    seq: u64,
    part: &str,
    paths: &[PathBuf],
    conflicts: &[(PathBuf, String)],
) -> Result<(), ExecError> {
    let handle = parent_handle(manager, parent)?;
    let key = part_key(run, child_id, cs_id, seq, part);
    let items: Vec<serde_json::Value> = match part {
        "conflicts" => conflicts
            .iter()
            .map(|(p, d)| serde_json::json!([rel_str(p), d.chars().take(400).collect::<String>()]))
            .collect(),
        _ => paths
            .iter()
            .map(|p| serde_json::json!(rel_str(p)))
            .collect(),
    };
    let chunks = pack_chunks(&items)?;
    let header =
        serde_json::json!({ "part": part, "chunks": chunks.len(), "created_ms": manager.now_ms() });
    let header = serde_json::to_string(&header)
        .map_err(|e| ExecError::Internal(format!("part header: {e}")))?;
    put_chunks(&handle, KIND_MERGE_PART, &key, &header, &chunks)
}

fn read_part_paths(
    manager: &Arc<faktor_session::SessionManager>,
    parent: SessionId,
    run: &str,
    child_id: &str,
    cs_id: &str,
    seq: u64,
    part: &str,
) -> Result<Option<Vec<PathBuf>>, ExecError> {
    let handle = parent_handle(manager, parent)?;
    let key = part_key(run, child_id, cs_id, seq, part);
    let Some((_h, chunks)) = read_chunks(&handle, KIND_MERGE_PART, &key)? else {
        return Ok(None);
    };
    let rows: Vec<String> = unpack_chunks(&chunks)?;
    let mut out = Vec::with_capacity(rows.len());
    for r in rows {
        out.push(validate_rel_path_str(&r)?);
    }
    Ok(Some(out))
}

/// Every durable merge envelope of one child, oldest first.
pub(crate) fn merge_envelopes(
    manager: &Arc<faktor_session::SessionManager>,
    parent: SessionId,
    run: &str,
    child_id: &str,
) -> Result<Vec<MergeEnvelope>, ExecError> {
    let handle = parent_handle(manager, parent)?;
    let prefix = format!("{run}/{child_id}/merge/");
    let mut envs: Vec<(u64, MergeEnvelope)> = Vec::new();
    for (kind, key, value) in scan_facts(&handle)? {
        if kind != KIND_MERGE {
            continue;
        }
        let Some(rest) = key.strip_prefix(&prefix) else {
            continue;
        };
        // rest = "<cs_id>/<seq>" — cs ids never contain '/' (built from
        // base ids), so the seq is the LAST numeric segment.
        let Some((_cs, seq_s)) = rest.rsplit_once('/') else {
            return Err(ExecError::Internal(format!(
                "hostile merge row key {key:?}"
            )));
        };
        let seq: u64 = seq_s
            .parse()
            .map_err(|_| ExecError::Internal(format!("hostile merge row key {key:?}")))?;
        let env: MergeEnvelope = serde_json::from_str(&value)
            .map_err(|e| ExecError::Internal(format!("merge envelope decode {key}: {e}")))?;
        envs.push((seq, env));
    }
    envs.sort_by_key(|(seq, _)| *seq);
    Ok(envs.into_iter().map(|(_, e)| e).collect())
}

fn write_envelope(
    manager: &Arc<faktor_session::SessionManager>,
    parent: SessionId,
    run: &str,
    env: &MergeEnvelope,
) -> Result<(), ExecError> {
    let handle = parent_handle(manager, parent)?;
    let value = serde_json::to_string(env)
        .map_err(|e| ExecError::Internal(format!("envelope serialization: {e}")))?;
    if value.len() > 3900 {
        return Err(ExecError::Oversized(format!(
            "merge envelope of {} bytes exceeds the durable row budget",
            value.len()
        )));
    }
    handle
        .upsert_memory_fact(
            KIND_MERGE,
            &envelope_key(run, &env.child_id, &env.cs_id, env.seq),
            &value,
        )
        .map_err(|e| ExecError::Internal(format!("merge record write: {}", e.message)))
}

// ------------------------------------------------------------ machinery

/// Map `kind -> path` of one recorded base map.
fn base_map_index(map: &[(PathBuf, FileHash)]) -> HashMap<PathBuf, FileHash> {
    map.iter().cloned().collect()
}

/// Compute the staged change entries of a child: every file of the child's
/// current tree that differs from its START map (files it began with are
/// skipped when unchanged; files it began with and removed are DELETION
/// entries) with `base_hash` anchored to the recorded PARENT map (the CAS
/// expectation against the parent worktree). Sorted by path. Entry count
/// beyond [`MAX_CHANGES`] is a typed Oversized error — the diff NEVER
/// silently truncates.
pub(crate) fn compute_change_entries(
    child_id: &str,
    start: &[(PathBuf, FileHash)],
    now: &[(PathBuf, FileHash)],
    parent_base: &[(PathBuf, FileHash)],
) -> Result<Vec<ChangeEntry>, ExecError> {
    let start_idx = base_map_index(start);
    let now_idx = base_map_index(now);
    let parent_idx = base_map_index(parent_base);
    let mut entries: Vec<ChangeEntry> = Vec::new();
    let mut paths: Vec<PathBuf> = now_idx.keys().cloned().collect();
    for p in start_idx.keys() {
        if !paths.contains(p) {
            paths.push(p.clone());
        }
    }
    paths.sort();
    for p in paths {
        let child = now_idx.get(&p).copied();
        let base_start = start_idx.get(&p).copied();
        let entry = match (child, base_start) {
            (Some(c), Some(s)) if c == s => continue, // untouched copy content
            (Some(c), _) => {
                let base = parent_idx.get(&p).copied();
                ChangeEntry {
                    path: p,
                    child_hash: Some(c),
                    base_hash: base,
                }
            }
            (None, Some(s)) => {
                // The child deleted a file it started from. The deletion
                // anchor must exist in the parent base map, otherwise the
                // file the child removed was never part of the parent tree
                // at spawn — refusing loudly beats guessing.
                let base = parent_idx.get(&p).copied().ok_or_else(|| {
                    ExecError::InvalidState(format!(
                        "child {child_id} removed {p:?} which its base snapshot held but the parent tree never had at the base snapshot; refusing to stage an unsound deletion"
                    ))
                })?;
                let _ = s;
                ChangeEntry {
                    path: p,
                    child_hash: None,
                    base_hash: Some(base),
                }
            }
            (None, None) => continue, // unreachable: p came from a map
        };
        entries.push(entry);
        if entries.len() > MAX_CHANGES {
            return Err(ExecError::Oversized(format!(
                "the change set of child {child_id} exceeds {MAX_CHANGES} files; refusing to stage a truncated diff (the parent must handle the oversized set explicitly)"
            )));
        }
    }
    Ok(entries)
}

// ================================================================ runtime

impl OrchestratorRuntime {
    /// Locate one child durably: the live exec mirror first, then every
    /// parent session's registry rows (post-run / reopened flows). A child
    /// id found in several runs is ambiguous and refused.
    fn locate_child(&self, child_id: &str) -> Result<(SessionId, String, ChildRuntime), ExecError> {
        {
            let guard = self.exec.lock().expect("exec lock");
            if let Some(exec) = guard.as_ref() {
                if let Some(row) = exec.children.get(child_id) {
                    return Ok((exec.parent_session, exec.run_id.clone(), row.clone()));
                }
            }
        }
        let mut found: Vec<(SessionId, String, ChildRuntime)> = Vec::new();
        for handle in self.manager.list_sessions(None)? {
            for (kind, key, value) in scan_facts(&handle)? {
                if kind != REGISTRY_ROW_KIND {
                    continue;
                }
                let Some((run, id)) = key.rsplit_once('/') else {
                    continue;
                };
                if id != child_id {
                    continue;
                }
                let row: ChildRuntime = serde_json::from_str(&value)
                    .map_err(|e| ExecError::Internal(format!("registry row decode: {e}")))?;
                found.push((handle.id(), run.to_string(), row));
            }
        }
        found.sort_by_key(|(_, run, _)| run.clone());
        found.dedup_by_key(|(_, run, row)| (run.clone(), row.child_id.clone()));
        match found.len() {
            0 => Err(ExecError::NotFound(format!("unknown child {child_id}"))),
            1 => Ok(found.remove(0)),
            _ => Err(ExecError::Conflict(format!(
                "child id {child_id} exists in {} runs; name the run explicitly",
                found.len()
            ))),
        }
    }

    /// The real directory of a child's worktree (from the durable rows).
    fn child_worktree_dir(&self, row: &ChildRuntime) -> Result<PathBuf, ExecError> {
        let worktrees = self
            .manager
            .worktrees_of(WorkspaceId::new(row.workspace_id))?
            .into_iter()
            .filter(|w| w.id as u64 == row.worktree_id)
            .map(|w| PathBuf::from(w.path))
            .collect::<Vec<_>>();
        let dir = worktrees.first().cloned().ok_or_else(|| {
            ExecError::NotFound(format!(
                "child {} has no registered worktree row",
                row.child_id
            ))
        })?;
        if !dir.is_dir() {
            return Err(ExecError::NotFound(format!(
                "child worktree {} of {} is not a directory",
                dir.display(),
                row.child_id
            )));
        }
        Ok(dir)
    }

    /// Record the base snapshot of a freshly spawned isolated child: the
    /// PARENT worktree map (merge CAS anchors; the tree the child would
    /// have been seeded from) and, when the child's own worktree already
    /// holds content, its START map. Returns the durable base id.
    pub(crate) fn record_spawn_base(
        &self,
        parent_session: SessionId,
        run_id: &str,
        owner_root: &Path,
        child: &ChildRuntime,
    ) -> Result<String, ExecError> {
        if child.ownership != ChildOwnership::IsolatedWorktree {
            return Ok(base_id_of(&child.child_id));
        }
        let parent_snap = faktor_fs::snapshot_tree(owner_root, MAX_BASE_ENTRIES).map_err(|e| {
            ExecError::from_fs("base snapshot of the parent worktree", owner_root, e)
        })?;
        let parent_map: Vec<(PathBuf, FileHash)> = parent_snap
            .iter()
            .map(|e| (e.path.clone(), e.hash))
            .collect();
        put_base_map(
            &self.manager,
            parent_session,
            run_id,
            &child.child_id,
            "parent",
            &parent_map,
        )?;
        // The child's own worktree at spawn (isolated dirs are created
        // empty by wave-12 spawn; reviewer copies are recorded by
        // spawn_reviewer directly).
        let child_dir = self.child_worktree_dir(child)?;
        let start_snap = faktor_fs::snapshot_tree(&child_dir, MAX_BASE_ENTRIES).map_err(|e| {
            ExecError::from_fs("base snapshot of the child worktree", &child_dir, e)
        })?;
        if !start_snap.is_empty() {
            let start_map: Vec<(PathBuf, FileHash)> = start_snap
                .iter()
                .map(|e| (e.path.clone(), e.hash))
                .collect();
            put_base_map(
                &self.manager,
                parent_session,
                run_id,
                &child.child_id,
                "start",
                &start_map,
            )?;
        }
        Ok(base_id_of(&child.child_id))
    }

    /// Audit 69/98: the structured result of a finished child — its durable
    /// row, the staged merge candidate (files + digests + summary) and any
    /// durable merge records. Prose-only results are replaced by this.
    pub fn child_result(&self, child_id: &str) -> Result<ChildResult, ExecError> {
        let (parent, run, child) = self.locate_child(child_id)?;
        let change_set = self.read_change_set_for_child(&parent, &run, &child)?;
        let summary = self
            .manager
            .get_session(SessionId::new(child.session_id))?
            .and_then(|h| h.orchestrator_child_identity_get().ok().flatten())
            .map(|i| i.task_goal)
            .unwrap_or_default();
        let merges = merge_envelopes(&self.manager, parent, &run, child_id)?;
        Ok(ChildResult {
            child,
            change_set,
            summary,
            merges,
        })
    }

    fn read_change_set_for_child(
        &self,
        parent: &SessionId,
        run: &str,
        child: &ChildRuntime,
    ) -> Result<Option<ChangeSet>, ExecError> {
        let Some(base_id) = child.base_snapshot_id.as_deref() else {
            return Ok(None);
        };
        match read_change_set(
            &self.manager,
            *parent,
            run,
            &child.child_id,
            &cs_id_of(base_id),
        ) {
            Ok(cs) => Ok(Some(cs)),
            Err(ExecError::NotFound(_)) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Audit 98: stage the merge candidate of a terminal child — compute
    /// its changed files against the base snapshot it started from (both
    /// recorded durably) and store the [`ChangeSet`]. A diff beyond
    /// [`MAX_CHANGES`] is a typed Oversized error and NOTHING is stored:
    /// the merge never silently truncates. Idempotent (re-staging after a
    /// crash re-upserts the same rows).
    pub fn stage_child_changes(&self, child_id: &str) -> Result<ChangeSet, ExecError> {
        let (parent, run, child) = self.locate_child(child_id)?;
        if child.state != ChildState::Done {
            return Err(ExecError::InvalidState(format!(
                "cannot stage child {child_id}: only terminal-success (Done) children stage (state {:?})",
                child.state
            )));
        }
        if child.ownership != ChildOwnership::IsolatedWorktree {
            return Err(ExecError::InvalidState(format!(
                "cannot stage child {child_id}: only IsolatedWorktree children own a change set (ownership {:?}); ExclusivePaths changes already live in the shared parent tree by wave-12 semantics",
                child.ownership
            )));
        }
        let parent_map = read_base_map(&self.manager, parent, &run, child_id, "parent")?;
        let parent_map = parent_map.unwrap_or_default();
        let start_map = read_base_map(&self.manager, parent, &run, child_id, "start")?;
        let start_map = start_map.unwrap_or_default();
        let child_dir = self.child_worktree_dir(&child)?;
        let now_snap = faktor_fs::snapshot_tree(&child_dir, MAX_BASE_ENTRIES)
            .map_err(|e| ExecError::from_fs("snapshot of the child worktree", &child_dir, e))?;
        let now: Vec<(PathBuf, FileHash)> =
            now_snap.iter().map(|e| (e.path.clone(), e.hash)).collect();
        let files = compute_change_entries(child_id, &start_map, &now, &parent_map)?;
        let cs = ChangeSet {
            child_id: child_id.to_string(),
            base_id: child
                .base_snapshot_id
                .clone()
                .unwrap_or_else(|| base_id_of(child_id)),
            files,
            created_ms: self.manager.now_ms(),
        };
        put_change_set(&self.manager, parent, &run, &cs)?;
        Ok(cs)
    }

    /// Audit 99: approve and merge. Applies ONLY the approved paths of the
    /// change set into the parent worktree, each via the fs commit-time CAS
    /// primitives (expected = the base-snapshot hash of the parent path —
    /// a parent that moved on since the base snapshot is a CONFLICT and is
    /// never overwritten). Rejected paths are recorded durably and are
    /// never merged later. Every changed file must be decided: an
    /// undecided path is an explicit error and NOTHING merges. Replay-safe:
    /// the durable merge record is written before any apply; a crash at any
    /// point is resumed by calling this again with the SAME decision (a
    /// path whose parent already holds the child digest is AlreadyCurrent,
    /// a mismatch with different content is surfaced as a conflict).
    pub fn approve_and_merge(
        &self,
        child_id: &str,
        change_set_id: &str,
        approved: &[PathBuf],
        rejected: &[PathBuf],
    ) -> Result<MergeOutcome, ExecError> {
        let (parent, run, child) = self.locate_child(child_id)?;
        if child.state != ChildState::Done {
            return Err(ExecError::InvalidState(format!(
                "cannot merge child {child_id}: only terminal-success (Done) children merge (state {:?})",
                child.state
            )));
        }
        if child.ownership != ChildOwnership::IsolatedWorktree {
            return Err(ExecError::InvalidState(format!(
                "cannot merge child {child_id}: only IsolatedWorktree children own a change set (ownership {:?})",
                child.ownership
            )));
        }
        let cs = read_change_set(&self.manager, parent, &run, child_id, change_set_id)?;
        if cs.id() != change_set_id {
            return Err(ExecError::NotFound(format!(
                "change set {change_set_id} does not belong to child {child_id}"
            )));
        }
        let (approved_v, rejected_v) = validate_decision(&cs, approved, rejected)?;
        let owner_root = self.plan_row(parent, &run)?.owner.root;
        let child_dir = self.child_worktree_dir(&child)?;

        // ---- durable record FIRST (crash between record and applies is
        // replay-safe because every apply is CAS-idempotent).
        let existing = merge_envelopes(&self.manager, parent, &run, child_id)?;
        let existing_env = existing.into_iter().find(|e| e.cs_id == cs.id());
        if let Some(env) = &existing_env {
            if env.cs_id != cs.id() {
                return Err(ExecError::Conflict(format!(
                    "child {child_id} already has a merge record for change set {}",
                    env.cs_id
                )));
            }
        }
        let seq = existing_env.as_ref().map(|e| e.seq).unwrap_or(1);
        let stored_approved = read_part_paths(
            &self.manager,
            parent,
            &run,
            child_id,
            &cs.id(),
            seq,
            "approved",
        )?;
        let stored_rejected = read_part_paths(
            &self.manager,
            parent,
            &run,
            child_id,
            &cs.id(),
            seq,
            "rejected",
        )?;
        let decision_stored = stored_approved.is_some() && stored_rejected.is_some();
        match (&existing_env, &stored_approved, &stored_rejected) {
            // A finalized record that lost its decision rows is corruption.
            (Some(env), _, _) if !env.in_flight() && !decision_stored => {
                return Err(ExecError::Internal(format!(
                    "durable merge record {seq} of child {child_id} is finalized but lost its decision rows"
                )));
            }
            // Both sides decided already: replay must carry the IDENTICAL
            // decision (decisions are durable, never silently revised).
            (Some(_), Some(a), Some(r)) => {
                if a != &approved_v || r != &rejected_v {
                    return Err(ExecError::Conflict(format!(
                        "the durable merge record {seq} of child {child_id} records a different decision; replay must carry the identical approved/rejected sets"
                    )));
                }
            }
            // Decision rows without an envelope: corrupt.
            (None, Some(a), Some(r)) if !a.is_empty() || !r.is_empty() => {
                return Err(ExecError::Internal(format!(
                    "decision rows of child {child_id} exist without a merge record"
                )));
            }
            // An in-flight record whose decision rows are missing or
            // partial: the crash happened before any apply (parts are
            // written before the first apply), so the current decision is
            // written fresh.
            (Some(env), _, _) if env.in_flight() && !decision_stored => {}
            (None, _, _) => {}
            _ => {}
        }
        // In-flight envelope + durable decision rows BEFORE any file apply.
        let now = self.manager.now_ms();
        if existing_env.is_none() {
            write_envelope(
                &self.manager,
                parent,
                &run,
                &MergeEnvelope {
                    seq,
                    child_id: child_id.to_string(),
                    cs_id: cs.id(),
                    status: MergeStatus::Failed,
                    approved_count: approved_v.len(),
                    rejected_count: rejected_v.len(),
                    merged_count: 0,
                    conflict_count: 0,
                    created_ms: now,
                    finished_ms: None,
                    details: "in-flight: durable merge record written before any file apply".into(),
                },
            )?;
        }
        if !decision_stored {
            put_part(
                &self.manager,
                parent,
                &run,
                child_id,
                &cs.id(),
                seq,
                "approved",
                &approved_v,
                &[],
            )?;
            put_part(
                &self.manager,
                parent,
                &run,
                child_id,
                &cs.id(),
                seq,
                "rejected",
                &rejected_v,
                &[],
            )?;
        }
        self.check_merge_seam(CrashSeam::AfterMergeRecord)?;

        // ---- apply phase (deterministic order = staged path order).
        let mut merged: Vec<PathBuf> = Vec::new();
        let mut conflicts: Vec<(PathBuf, String)> = Vec::new();
        let mut processed = 0usize;
        for entry in &cs.files {
            if rejected_v.contains(&entry.path) {
                processed += 1;
                continue;
            }
            let outcome = self.apply_one_merge(&owner_root, &child_dir, entry);
            match outcome {
                Ok(()) => merged.push(entry.path.clone()),
                Err(MergeFailure::Conflict(c)) => conflicts.push((entry.path.clone(), c)),
                Err(MergeFailure::Hard(e)) => {
                    // Prior applies stay (each individually CAS-committed);
                    // the in-flight durable record lets the parent resume.
                    return Err(e);
                }
            }
            processed += 1;
            self.check_merge_seam(CrashSeam::MergeApply { after: processed })?;
        }
        merged.sort();
        // ---- durable outcome rows, then the FINAL envelope: the parent is
        // notified only after the durable merge record exists in full.
        put_part(
            &self.manager,
            parent,
            &run,
            child_id,
            &cs.id(),
            seq,
            "merged",
            &merged,
            &[],
        )?;
        put_part(
            &self.manager,
            parent,
            &run,
            child_id,
            &cs.id(),
            seq,
            "conflicts",
            &[],
            &conflicts,
        )?;
        let failed = !conflicts.is_empty();
        let detail = if conflicts.is_empty() {
            format!("all {} approved file(s) merged", merged.len())
        } else {
            let (first_path, first_detail) = &conflicts[0];
            format!(
                "{} conflict(s); first: {} — {}",
                conflicts.len(),
                first_path.display(),
                first_detail.chars().take(160).collect::<String>()
            )
        };
        write_envelope(
            &self.manager,
            parent,
            &run,
            &MergeEnvelope {
                seq,
                child_id: child_id.to_string(),
                cs_id: cs.id(),
                status: if failed {
                    MergeStatus::Failed
                } else {
                    MergeStatus::Applied
                },
                approved_count: approved_v.len(),
                rejected_count: rejected_v.len(),
                merged_count: merged.len(),
                conflict_count: conflicts.len(),
                created_ms: now,
                finished_ms: Some(self.manager.now_ms()),
                details: detail.chars().take(300).collect(),
            },
        )?;
        Ok(MergeOutcome {
            merged,
            rejected: rejected_v,
            conflicts,
        })
    }

    fn apply_one_merge(
        &self,
        owner_root: &Path,
        child_dir: &Path,
        entry: &ChangeEntry,
    ) -> Result<(), MergeFailure> {
        let res = match entry.child_hash {
            Some(child_hash) => faktor_fs::merge_apply_content(
                owner_root,
                &entry.path,
                child_dir,
                &entry.path,
                child_hash,
                entry.base_hash,
            ),
            None => match entry.base_hash {
                Some(base) => faktor_fs::merge_delete(owner_root, &entry.path, base),
                None => {
                    return Err(MergeFailure::Hard(ExecError::Internal(format!(
                        "staged deletion {:?} has no base anchor",
                        entry.path
                    ))));
                }
            },
        };
        match res {
            Ok(CasMergeResult::Applied | CasMergeResult::AlreadyCurrent) => Ok(()),
            Err(e)
                if matches!(
                    e.kind,
                    faktor_core::ErrorKind::Conflict
                        | faktor_core::ErrorKind::Permission
                        | faktor_core::ErrorKind::NotFound
                ) =>
            {
                // A parent path that cannot be resolved safely, a vanished
                // file or a child-source drift are all PER-FILE conflicts
                // (the file is not merged; the rest of the decision still
                // applies). Anything else aborts the merge loudly.
                Err(MergeFailure::Conflict(e.message))
            }
            Err(e) => Err(MergeFailure::Hard(ExecError::from_fs(
                "controlled merge apply",
                owner_root,
                e,
            ))),
        }
    }

    fn check_merge_seam(&self, seam: CrashSeam) -> Result<(), ExecError> {
        let mut guard = self.exec.lock().expect("exec lock");
        let Some(exec) = guard.as_mut() else {
            return Ok(());
        };
        self.check_crash(exec, seam)
    }

    /// Audit 70: spawn a REVIEWER child whose worktree is a FRESH, bounded,
    /// atomic-per-file copy of the CURRENT parent state at review spawn —
    /// never the reviewed child's dirty worktree. The copy is recorded as
    /// the reviewer's durable base snapshot (`base_snapshot_id` on the
    /// reviewer child row). Nothing is skipped silently: an un-copyable
    /// file fails the spawn; trees beyond the entry/byte caps are typed
    /// Oversized errors.
    pub fn spawn_reviewer(&self, plan_child_id: &str) -> Result<ChildRuntime, ExecError> {
        let (parent, run, plan_child) = self.locate_child(plan_child_id)?;
        if plan_child.state != ChildState::Done {
            return Err(ExecError::InvalidState(format!(
                "cannot review child {plan_child_id}: only terminal-success (Done) children are reviewed (state {:?})",
                plan_child.state
            )));
        }
        let plan = self.plan_row(parent, &run)?;
        let seq = self.next_spawn_seq(&parent, &run)?;
        let reviewer_id = format!("review-{seq}");
        let dir = plan
            .isolated_root
            .join(sanitize_run_id(&run))
            .join(&reviewer_id);
        std::fs::create_dir_all(&dir)
            .map_err(|e| ExecError::Internal(format!("reviewer dir {dir:?}: {e}")))?;
        // Bounded copy of the CURRENT parent state (per-file atomic).
        let manifest = faktor_fs::copy_tree(
            &plan.owner.root,
            &dir,
            MAX_BASE_ENTRIES,
            MAX_REVIEW_COPY_BYTES,
        )
        .map_err(|e| {
            let _ = std::fs::remove_dir_all(&dir);
            ExecError::from_fs("reviewer worktree copy", &plan.owner.root, e)
        })?;
        let ws = self
            .manager
            .create_workspace(&dir.to_string_lossy())
            .map_err(|e| ExecError::Internal(format!("reviewer workspace row: {e}")))?;
        let wt_raw = self
            .manager
            .put_worktree(ws, &dir.to_string_lossy(), &format!("review-{seq}"))
            .map_err(|e| ExecError::Internal(format!("reviewer worktree row: {e}")))?;
        let title = truncate(
            &format!(
                "review of child {plan_child_id} — {}",
                truncate(&plan.plan.goal, 300)
            ),
            2000,
        );
        let session = self
            .manager
            .create_child_session(
                parent,
                ws,
                WorktreeId::new(wt_raw as u64),
                TaskId::new(1),
                &plan.provider,
                &plan.default_model,
                &title,
                ChildOwnership::IsolatedWorktree,
            )
            .map_err(|e| ExecError::Internal(format!("create_child_session: {e}")))?;
        let now = self.manager.now_ms();
        let mut row = ChildRuntime {
            child_id: reviewer_id.clone(),
            parent_session_id: parent.raw(),
            run_id: run.clone(),
            item_id: format!("review-of-{plan_child_id}"),
            kind: WorkKind::Review,
            session_id: session.id().raw(),
            operation_id: 0,
            workspace_id: ws.raw(),
            worktree_id: wt_raw as u64,
            ownership: ChildOwnership::IsolatedWorktree,
            ownership_paths: Vec::new(),
            state: ChildState::Running,
            budget_max_tokens: plan_child.budget_max_tokens,
            permissions: plan_child.permissions.clone(),
            model_policy: ModelPolicy {
                model: plan_child.model_policy.model.clone(),
            },
            created_ms: now,
            updated_ms: now,
            base_snapshot_id: None,
        };
        // Base rows: the reviewer's own fresh copy IS its base (audit 70:
        // "base snapshot at review start"); the manifest returned by the
        // copy is the exact copied content (single-pass, hash-verified).
        let manifest_map: Vec<(PathBuf, FileHash)> =
            manifest.iter().map(|e| (e.path.clone(), e.hash)).collect();
        let base_id = base_id_of(&reviewer_id);
        put_base_map(
            &self.manager,
            parent,
            &run,
            &reviewer_id,
            "parent",
            &manifest_map,
        )?;
        put_base_map(
            &self.manager,
            parent,
            &run,
            &reviewer_id,
            "start",
            &manifest_map,
        )?;
        row.base_snapshot_id = Some(base_id);
        let mut guard = self.exec.lock().expect("exec lock");
        let Some(exec) = guard.as_mut() else {
            // No active execution: only the durable rows exist; the
            // executor drives reviewers through the same submit path in
            // later waves (they are not plan items).
            return Ok(row);
        };
        {
            let value = serde_json::to_string(&row)
                .map_err(|e| ExecError::Internal(format!("registry row serialization: {e}")))?;
            exec_manager_row_write(
                &self.manager,
                exec.parent_session,
                &run,
                &row.child_id,
                &value,
            )?;
        }
        exec.children.insert(row.child_id.clone(), row.clone());
        if seq >= exec.next_child_seq {
            exec.next_child_seq = seq + 1;
        }
        Ok(row)
    }

    /// Next free spawn sequence number across the mirror and the durable
    /// registry rows (`child-N` and `review-N` share one sequence).
    fn next_spawn_seq(&self, parent: &SessionId, run: &str) -> Result<u64, ExecError> {
        let mut max: u64 = 0;
        {
            let guard = self.exec.lock().expect("exec lock");
            if let Some(exec) = guard.as_ref() {
                if exec.parent_session == *parent && exec.run_id == run {
                    max = max.max(exec.next_child_seq.saturating_sub(1));
                }
            }
        }
        for row in OrchestratorRuntime::registry_rows(self.manager.clone(), *parent, run)? {
            if let Some(n) = parse_seq_id(&row.child_id) {
                max = max.max(n);
            }
        }
        Ok(max + 1)
    }
}

/// Registry-row write shared by spawn paths that run outside
/// `persist_row`'s exec borrow.
fn exec_manager_row_write(
    manager: &Arc<faktor_session::SessionManager>,
    parent: SessionId,
    run: &str,
    child_id: &str,
    value: &str,
) -> Result<(), ExecError> {
    let handle = manager
        .get_session(parent)?
        .ok_or_else(|| ExecError::NotFound(format!("parent session {parent}")))?;
    handle
        .upsert_memory_fact(REGISTRY_ROW_KIND, &format!("{run}/{child_id}"), value)
        .map_err(|e| ExecError::Internal(format!("registry row write: {}", e.message)))
}

fn parse_seq_id(child_id: &str) -> Option<u64> {
    for prefix in ["child-", "review-"] {
        if let Some(n) = child_id.strip_prefix(prefix) {
            if let Ok(n) = n.parse::<u64>() {
                return Some(n);
            }
        }
    }
    None
}

/// Hard (abort-the-whole-merge) failure vs a per-file conflict.
enum MergeFailure {
    Conflict(String),
    Hard(ExecError),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(path: &str, child: Option<&str>, base: Option<&str>) -> ChangeEntry {
        let hex = |h: &str| FileHash::from_hex(h).expect("hex");
        ChangeEntry {
            path: PathBuf::from(path),
            child_hash: child.map(hex),
            base_hash: base.map(hex),
        }
    }

    fn cs(files: Vec<ChangeEntry>) -> ChangeSet {
        ChangeSet {
            child_id: "child-0".into(),
            base_id: "base-child-0".into(),
            files,
            created_ms: 1,
        }
    }

    const H: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const H2: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    #[test]
    fn compute_entries_skips_unchanged_and_anchors_deletes_and_creates() {
        // Seeded start (copy of the parent): modify + delete + unchanged.
        let start = vec![
            (PathBuf::from("a.rs"), FileHash::from_hex(H).unwrap()),
            (PathBuf::from("gone.rs"), FileHash::from_hex(H2).unwrap()),
            (PathBuf::from("same.rs"), FileHash::from_hex(H).unwrap()),
        ];
        let now = vec![
            (PathBuf::from("a.rs"), FileHash::from_hex(H2).unwrap()),
            (PathBuf::from("same.rs"), FileHash::from_hex(H).unwrap()),
            (PathBuf::from("new.rs"), FileHash::from_hex(H).unwrap()),
        ];
        let parent_base = vec![
            (PathBuf::from("a.rs"), FileHash::from_hex(H).unwrap()),
            (PathBuf::from("gone.rs"), FileHash::from_hex(H2).unwrap()),
            (PathBuf::from("same.rs"), FileHash::from_hex(H).unwrap()),
        ];
        let entries = compute_change_entries("c", &start, &now, &parent_base).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].path, PathBuf::from("a.rs"));
        assert_eq!(entries[0].child_hash, Some(FileHash::from_hex(H2).unwrap()));
        assert_eq!(entries[0].base_hash, Some(FileHash::from_hex(H).unwrap()));
        assert_eq!(entries[1].path, PathBuf::from("gone.rs"));
        assert_eq!(entries[1].child_hash, None);
        assert_eq!(entries[1].base_hash, Some(FileHash::from_hex(H2).unwrap()));
        assert_eq!(entries[2].path, PathBuf::from("new.rs"));
        assert_eq!(entries[2].child_hash, Some(FileHash::from_hex(H).unwrap()));
        assert_eq!(entries[2].base_hash, None);
    }

    #[test]
    fn compute_entries_refuses_unsound_deletion_and_caps_loudly() {
        // The child deleted a file its start map had, but the parent base
        // map never contained it: staging an unsound deletion is refused.
        let start = vec![(PathBuf::from("x"), FileHash::from_hex(H).unwrap())];
        let now: Vec<(PathBuf, FileHash)> = vec![];
        let parent_base: Vec<(PathBuf, FileHash)> = vec![];
        let err = compute_change_entries("c", &start, &now, &parent_base).unwrap_err();
        assert!(matches!(err, ExecError::InvalidState(_)), "{err:?}");
        // An empty-seeded wave-12 child that wrote one file beyond the cap
        // must fail loudly at MAX_CHANGES, never truncate.
        let many: Vec<(PathBuf, FileHash)> = (0..(MAX_CHANGES + 1))
            .map(|i| {
                (
                    PathBuf::from(format!("f{i:05}.rs")),
                    FileHash::from_hex(H).unwrap(),
                )
            })
            .collect();
        let err = compute_change_entries("c", &[], &many, &[]).unwrap_err();
        assert!(matches!(err, ExecError::Oversized(_)), "{err:?}");
        let exactly: Vec<(PathBuf, FileHash)> = (0..MAX_CHANGES)
            .map(|i| {
                (
                    PathBuf::from(format!("f{i:05}.rs")),
                    FileHash::from_hex(H).unwrap(),
                )
            })
            .collect();
        assert_eq!(
            compute_change_entries("c", &[], &exactly, &[])
                .unwrap()
                .len(),
            MAX_CHANGES
        );
    }

    #[test]
    fn decision_must_cover_every_file_and_reject_hostile_paths() {
        let c = cs(vec![
            entry("src/a.rs", Some(H), Some(H2)),
            entry("src/b.rs", Some(H), None),
        ]);
        // Undecided b: explicit error, nothing merges.
        let err = validate_decision(&c, &[PathBuf::from("src/a.rs")], &[]).unwrap_err();
        assert!(matches!(err, ExecError::UndecidedPaths(_)), "{err:?}");
        assert!(format!("{err:?}").contains("src/b.rs"));
        // Traversal / absolute / dot components are typed InvalidApproval.
        for evil in ["../evil", "/etc/passwd", "src/../a.rs", "./src/a.rs", ".."] {
            let err = validate_decision(
                &c,
                &[PathBuf::from(evil), PathBuf::from("src/b.rs")],
                &[PathBuf::from("src/a.rs")],
            )
            .unwrap_err();
            assert!(
                matches!(err, ExecError::InvalidApproval(_)),
                "{evil:?} must be a typed invalid approval: {err:?}"
            );
        }
        // Case-variant approval is a typed error with a case hint.
        let err = validate_decision(
            &c,
            &[PathBuf::from("SRC/A.RS"), PathBuf::from("src/b.rs")],
            &[PathBuf::from("src/a.rs")],
        )
        .unwrap_err();
        assert!(
            matches!(err, ExecError::InvalidApproval(_)) && format!("{err:?}").contains("case"),
            "{err:?}"
        );
        // Unknown paths and approve/reject overlap are typed errors.
        let err = validate_decision(
            &c,
            &[PathBuf::from("nope.rs"), PathBuf::from("src/b.rs")],
            &[PathBuf::from("src/a.rs")],
        )
        .unwrap_err();
        assert!(matches!(err, ExecError::InvalidApproval(_)), "{err:?}");
        let err = validate_decision(
            &c,
            &[PathBuf::from("src/a.rs")],
            &[PathBuf::from("src/a.rs"), PathBuf::from("src/b.rs")],
        )
        .unwrap_err();
        assert!(matches!(err, ExecError::InvalidApproval(_)), "{err:?}");
        // Duplicates are refused.
        let err = validate_decision(
            &c,
            &[
                PathBuf::from("src/a.rs"),
                PathBuf::from("src/a.rs"),
                PathBuf::from("src/b.rs"),
            ],
            &[],
        )
        .unwrap_err();
        assert!(matches!(err, ExecError::InvalidApproval(_)), "{err:?}");
        // A complete, disjoint decision passes and round-trips sorted.
        let (a, r) = validate_decision(
            &c,
            &[PathBuf::from("src/b.rs"), PathBuf::from("src/a.rs")],
            &[],
        )
        .unwrap();
        assert_eq!(
            a,
            vec![PathBuf::from("src/a.rs"), PathBuf::from("src/b.rs")]
        );
        assert!(r.is_empty());
    }

    #[test]
    fn pack_chunks_roundtrips_and_empty_payloads_stay_readable() {
        let items: Vec<ChangeEntry> = (0..1000)
            .map(|i| entry(&format!("dir/f{i:04}.rs"), Some(H), None))
            .collect();
        let chunks = pack_chunks(&items).unwrap();
        assert!(chunks.len() > 1, "chunking splits big payloads");
        for c in &chunks {
            assert!(c.len() <= CHUNK_BUDGET, "chunk of {} bytes", c.len());
        }
        let back: Vec<ChangeEntry> = unpack_chunks(&chunks).unwrap();
        assert_eq!(back, items);
        // Empty payload: one readable `[]` chunk.
        let chunks = pack_chunks::<ChangeEntry>(&[]).unwrap();
        assert_eq!(chunks, vec!["[]".to_string()]);
        let back: Vec<ChangeEntry> = unpack_chunks(&chunks).unwrap();
        assert!(back.is_empty());
    }

    #[test]
    fn stored_rows_decode_never_accepts_hostile_values() {
        // Unpacked values pass through typed validators: a hostile path in
        // a stored row must be caught by validate_rel_path_str.
        let err = validate_rel_path_str("a/../../b").unwrap_err();
        assert!(matches!(err, ExecError::InvalidApproval(_)));
        let err = validate_rel_path_str("/abs").unwrap_err();
        assert!(matches!(err, ExecError::InvalidApproval(_)));
        let err = validate_rel_path_str("").unwrap_err();
        assert!(matches!(err, ExecError::InvalidApproval(_)));
        assert!(validate_rel_path_str("src/a.rs").is_ok());
        assert!(validate_rel_path_str(&"p".repeat(MAX_DECISION_PATH_CHARS)).is_ok());
        let err = validate_rel_path_str(&"p".repeat(MAX_DECISION_PATH_CHARS + 1)).unwrap_err();
        assert!(matches!(err, ExecError::InvalidApproval(_)));
    }
}
