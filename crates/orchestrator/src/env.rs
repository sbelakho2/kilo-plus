//! Durable env-snapshot rows + epoch-pinned context reads (audit 97).
//!
//! A child/dependent session that inherits instructions/environment from
//! its parent binds to an IMMUTABLE [`EnvSnapshot`] taken at spawn:
//!
//! - **Snapshot rows** (kind [`KIND_ENV_SNAPSHOT`], key
//!   `<run_id>/env-<child_id>`) hold the snapshot JSON: id, instruction
//!   epoch, capture root, ordered workspace-relative paths with their
//!   recorded `rules_hash` (path + content) and `bytes_hash` (content),
//!   taken time.
//! - **Content rows** (kind [`KIND_ENV_CONTENT`], key = `rules_hash` hex —
//!   GLOBAL across runs) hold the captured file bytes, chunked like every
//!   other durable row. Unchanged environments between two spawns produce
//!   the SAME rules hashes, so their content is stored ONCE (dedupe);
//!   snapshot rows are per child and cheap.
//!
//! Content rows are written BEFORE the snapshot row (a snapshot row never
//! references missing content; a crash mid-write leaves only deduped,
//! bounded garbage that a later bind overwrites). Every write is an
//! idempotent upsert.
//!
//! Context building for a bound child reads ONLY its snapshot through
//! [`OrchestratorRuntime::pinned_context_instructions`]: rows are joined,
//! re-verified against the recorded hashes (missing or tampered rows are
//! loud typed errors), and assembled with
//! `faktor_instructions::Instructions::from_snapshot` — the live
//! filesystem is NEVER consulted. A child row without a binding (or with a
//! binding whose rows vanished) refuses loudly instead of silently falling
//! back to the live environment.

use std::collections::BTreeMap;
use std::path::Path;

use faktor_core::id::SessionId;
use faktor_instructions::{
    EnvSnapshot, EnvSnapshotError, Instructions, MAX_RULE_BYTES, MAX_SNAPSHOT_TOTAL_BYTES,
};

use super::merge::{
    pack_chunks, parent_handle, put_chunks, read_chunks, scan_facts, unpack_chunks,
};
use super::*;

/// Durable snapshot row kind (parent session fact space).
pub(crate) const KIND_ENV_SNAPSHOT: &str = "orchestrator_env";
/// Durable content row kind: key = rules-hash hex (global dedupe).
pub(crate) const KIND_ENV_CONTENT: &str = "orchestrator_env_content";

fn env_key(run: &str, snapshot_id: &str) -> String {
    format!("{run}/{snapshot_id}")
}

fn content_key(rules_hash: u64) -> String {
    format!("{rules_hash:016x}")
}

/// Map a pure snapshot error onto the typed orchestrator error space.
fn map_env_error(e: EnvSnapshotError) -> ExecError {
    match e {
        EnvSnapshotError::Oversized(m) => ExecError::Oversized(m),
        EnvSnapshotError::Missing(m) => ExecError::NotFound(m),
        EnvSnapshotError::Malformed(m) | EnvSnapshotError::Tampered(m) => {
            ExecError::InvalidState(m)
        }
        EnvSnapshotError::EpochMismatch { expected, actual } => ExecError::InvalidState(format!(
            "live instruction epoch {actual} differs from the required epoch {expected}; the pinned snapshot must be read, never the live env"
        )),
    }
}

impl OrchestratorRuntime {
    /// Capture the current rule environment of `env_root` and persist it
    /// durably under the parent session as the child's env binding
    /// (audit 97). Content rows are deduplicated by rules hash; the
    /// snapshot row is written last so it never references missing
    /// content. Returns the snapshot id (`env-<child_id>`). A hostile
    /// environment beyond the snapshot caps (64 paths / bounded total
    /// bytes) is a typed [`ExecError::Oversized`] — the spawn fails
    /// loudly, the binding is never silently truncated or skipped.
    pub(crate) fn bind_child_env(
        &self,
        parent: SessionId,
        run_id: &str,
        child_id: &str,
        env_root: &Path,
    ) -> Result<String, ExecError> {
        let snapshot_id = format!("env-{child_id}");
        let captured = faktor_instructions::EnvSnapshot::capture(
            env_root,
            &snapshot_id,
            self.manager.now_ms(),
        )
        .map_err(map_env_error)?;
        self.put_env_snapshot(parent, run_id, &captured)?;
        Ok(snapshot_id)
    }

    /// Persist one captured env: content rows first (deduped by rules
    /// hash, idempotent upserts), then the snapshot row.
    pub(crate) fn put_env_snapshot(
        &self,
        parent: SessionId,
        run_id: &str,
        captured: &faktor_instructions::CapturedEnv,
    ) -> Result<(), ExecError> {
        let handle = parent_handle(&self.manager, parent)?;
        for (rel, rec) in &captured.snapshot.workspace_paths {
            let Some(text) = captured.content.get(rel) else {
                return Err(ExecError::Internal(format!(
                    "captured env of {:?} lacks content for {rel:?}",
                    captured.snapshot.root
                )));
            };
            let chunks = pack_chunks(&[text.as_str()])?;
            let header = serde_json::to_string(&serde_json::json!({
                "chunks": chunks.len(),
                "bytes": text.len(),
            }))
            .map_err(|e| ExecError::Internal(format!("env content header: {e}")))?;
            put_chunks(
                &handle,
                KIND_ENV_CONTENT,
                &content_key(rec.rules_hash),
                &header,
                &chunks,
            )?;
        }
        let chunks = pack_chunks(std::slice::from_ref(&captured.snapshot))?;
        let header = serde_json::to_string(&serde_json::json!({
            "chunks": chunks.len(),
            "paths": captured.snapshot.workspace_paths.len(),
        }))
        .map_err(|e| ExecError::Internal(format!("env snapshot header: {e}")))?;
        put_chunks(
            &handle,
            KIND_ENV_SNAPSHOT,
            &env_key(run_id, &captured.snapshot.snapshot_id),
            &header,
            &chunks,
        )?;
        Ok(())
    }

    /// Read one durable snapshot row by id; `None` when never recorded.
    fn read_env_snapshot_row(
        &self,
        parent: SessionId,
        run_id: &str,
        snapshot_id: &str,
    ) -> Result<Option<EnvSnapshot>, ExecError> {
        let handle = parent_handle(&self.manager, parent)?;
        let Some((_header, chunks)) =
            read_chunks(&handle, KIND_ENV_SNAPSHOT, &env_key(run_id, snapshot_id))?
        else {
            return Ok(None);
        };
        let mut snaps: Vec<EnvSnapshot> = unpack_chunks(&chunks)?;
        if snaps.len() != 1 {
            return Err(ExecError::Internal(format!(
                "env snapshot row {run_id}/{snapshot_id} holds {} snapshots",
                snaps.len()
            )));
        }
        let snap = snaps.remove(0);
        if snap.snapshot_id != snapshot_id {
            return Err(ExecError::Internal(format!(
                "env snapshot row key/ids mismatch (key {snapshot_id} vs value {})",
                snap.snapshot_id
            )));
        }
        Ok(Some(snap))
    }

    /// Join the durable content rows of one snapshot, verifying row-level
    /// bounds (hostile oversize is loud; the hash re-verification happens
    /// in the pinned assembly).
    fn read_snapshot_content(
        &self,
        parent: SessionId,
        snap: &EnvSnapshot,
    ) -> Result<BTreeMap<String, String>, ExecError> {
        let handle = parent_handle(&self.manager, parent)?;
        let mut content = BTreeMap::new();
        let mut total = 0usize;
        for (rel, rec) in &snap.workspace_paths {
            let key = content_key(rec.rules_hash);
            let Some((_header, chunks)) = read_chunks(&handle, KIND_ENV_CONTENT, &key)? else {
                return Err(ExecError::NotFound(format!(
                    "env snapshot {}: content row for {rel:?} (hash {key}) is missing; refusing a silent live fallback",
                    snap.snapshot_id
                )));
            };
            let rows: Vec<String> = unpack_chunks(&chunks)?;
            if rows.len() != 1 {
                return Err(ExecError::Internal(format!(
                    "env content row {key} holds {} values",
                    rows.len()
                )));
            }
            let text = rows.into_iter().next().expect("len checked");
            if text.len() > MAX_RULE_BYTES {
                return Err(ExecError::Internal(format!(
                    "env content row {key} of {} bytes exceeds the {MAX_RULE_BYTES} rule bound",
                    text.len()
                )));
            }
            total = total.saturating_add(text.len());
            if total > MAX_SNAPSHOT_TOTAL_BYTES {
                return Err(ExecError::Oversized(format!(
                    "env snapshot {} joined content exceeds {MAX_SNAPSHOT_TOTAL_BYTES} bytes; refusing a partial env",
                    snap.snapshot_id
                )));
            }
            content.insert(rel.clone(), text);
        }
        Ok(content)
    }

    /// The durable env binding (snapshot row only) of one child of a run.
    /// `None` when the child row carries no binding (legacy/tampered rows).
    pub fn env_snapshot_of(
        &self,
        parent: SessionId,
        run_id: &str,
        child_id: &str,
    ) -> Result<Option<EnvSnapshot>, ExecError> {
        let rows = Self::registry_rows(self.manager.clone(), parent, run_id)?;
        let row = rows
            .iter()
            .find(|r| r.child_id == child_id)
            .ok_or_else(|| ExecError::NotFound(format!("unknown child {child_id} of {run_id}")))?;
        let Some(binding) = row.env_snapshot_id.as_deref() else {
            return Ok(None);
        };
        self.read_env_snapshot_row(parent, run_id, binding)
    }

    /// The durable env binding of one child wherever it lives (locate
    /// semantics; ambiguous child ids across runs are a Conflict).
    pub fn env_snapshot_by_child(&self, child_id: &str) -> Result<Option<EnvSnapshot>, ExecError> {
        let (parent, run, row) = self.locate_child(child_id)?;
        let Some(binding) = row.env_snapshot_id else {
            return Ok(None);
        };
        self.read_env_snapshot_row(parent, &run, &binding)
    }

    /// The epoch-pinned context instructions of one bound child (audit 97).
    /// Reads ONLY the child's durable spawn-time env snapshot: rows are
    /// joined and re-verified, then assembled without ever touching the
    /// live filesystem. Refusals are loud and typed:
    ///
    /// - a child row WITHOUT a binding is an incomplete spawn
    ///   ([`ExecError::InvalidState`]) — never a silent live fallback;
    /// - a binding whose snapshot/content rows are missing is
    ///   [`ExecError::NotFound`];
    /// - tampered content or hostile row shapes are [`ExecError::InvalidState`].
    pub fn pinned_context_instructions(
        &self,
        parent: SessionId,
        run_id: &str,
        child_id: &str,
    ) -> Result<Instructions, ExecError> {
        let rows = Self::registry_rows(self.manager.clone(), parent, run_id)?;
        let row = rows
            .iter()
            .find(|r| r.child_id == child_id)
            .ok_or_else(|| ExecError::NotFound(format!("unknown child {child_id} of {run_id}")))?;
        let Some(binding) = row.env_snapshot_id.as_deref() else {
            return Err(ExecError::InvalidState(format!(
                "child {child_id} of {run_id} has no durable env-snapshot binding; its spawn was incomplete or its rows were tampered — refusing a silent live-env fallback"
            )));
        };
        self.pinned_context_instructions_bound(parent, run_id, binding)
    }

    /// The same read addressed by child id alone (locate semantics).
    pub fn pinned_context_instructions_by_child(
        &self,
        child_id: &str,
    ) -> Result<Instructions, ExecError> {
        let (parent, run, row) = self.locate_child(child_id)?;
        let Some(binding) = row.env_snapshot_id else {
            return Err(ExecError::InvalidState(format!(
                "child {child_id} has no durable env-snapshot binding; its spawn was incomplete or its rows were tampered — refusing a silent live-env fallback"
            )));
        };
        self.pinned_context_instructions_bound(parent, &run, &binding)
    }

    fn pinned_context_instructions_bound(
        &self,
        parent: SessionId,
        run_id: &str,
        binding: &str,
    ) -> Result<Instructions, ExecError> {
        let snap = self
            .read_env_snapshot_row(parent, run_id, binding)?
            .ok_or_else(|| {
                ExecError::NotFound(format!(
                    "child binding names env snapshot {run_id}/{binding} but its durable row is missing — refusing a silent live-env fallback"
                ))
            })?;
        let content = self.read_snapshot_content(parent, &snap)?;
        Instructions::from_snapshot(&snap, &content).map_err(map_env_error)
    }
}

/// Enumerate the durable env-snapshot rows of one run (bounded scan).
#[allow(dead_code)]
pub(crate) fn env_snapshot_rows(
    manager: &Arc<faktor_session::SessionManager>,
    parent: SessionId,
    run_id: &str,
) -> Result<Vec<(String, String, String)>, ExecError> {
    let handle = parent_handle(manager, parent)?;
    let prefix = format!("{run_id}/");
    let mut out = Vec::new();
    for (kind, key, value) in scan_facts(&handle)? {
        if kind == KIND_ENV_SNAPSHOT && key.strip_prefix(&prefix).is_some() {
            out.push((kind, key, value));
        }
    }
    Ok(out)
}
