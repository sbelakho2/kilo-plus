//! In-memory evidence storage with scoped reads and bounded backing retention.
//!
//! Two invariants are structural in this module:
//!
//! 1. **Knowing an id is not authorization.** [`EvidenceStore::get_scoped`]
//!    compares the caller's session/workspace (and task, when both sides name
//!    one) against the envelope and returns [`EvidenceError::AccessDenied`] on
//!    any mismatch — even when the caller presents a valid [`EvidenceId`].
//! 2. **Backing is optional and bounded.** Bytes are retained only for
//!    reversible/aggressive evidence whose capture is complete and whose
//!    length fits the store's `backing_cap`; otherwise the envelope (including
//!    its `backing_hash`) is kept and only the bytes are dropped. Reading
//!    dropped backing fails loudly, never by substitution.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::types::{
    BackingCompleteness, Compressibility, EvidenceEnvelope, EvidenceError, EvidenceId,
};

/// The identity a reader presents. A read is permitted only when session and
/// workspace match the stored envelope; task ids are compared only when both
/// the envelope and the context name one. A task-less envelope stays visible to
/// task-scoped readers, and a task-less reader sees every task in its
/// session/workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvidenceAccessContext {
    pub session_id: u64,
    pub workspace_id: u64,
    pub task_id: Option<u64>,
}

impl EvidenceAccessContext {
    pub const fn new(session_id: u64, workspace_id: u64, task_id: Option<u64>) -> Self {
        Self {
            session_id,
            workspace_id,
            task_id,
        }
    }
}

/// One stored envelope plus the backing bytes the store chose to retain.
/// `backing == None` means the bytes were never provided or were dropped by
/// the store policy; the envelope and its `backing_hash` remain authoritative.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredEvidence {
    pub envelope: EvidenceEnvelope,
    pub backing: Option<Vec<u8>>,
}

impl StoredEvidence {
    /// Length of the retained backing, or `None` when no backing is retained.
    pub fn backing_len(&self) -> Option<usize> {
        self.backing.as_ref().map(Vec::len)
    }
}

/// Evidence storage contract. `insert` never overwrites: re-inserting a live
/// id is refused, so a replay can never silently replace evidence bytes.
pub trait EvidenceStore {
    fn insert(
        &mut self,
        env: EvidenceEnvelope,
        backing: Option<Vec<u8>>,
    ) -> Result<(), EvidenceError>;

    /// Scope-checked read. The default implementation performs the scope
    /// comparison on top of [`EvidenceStore::get`] so every store enforces the
    /// same rule.
    fn get_scoped(
        &self,
        id: EvidenceId,
        ctx: &EvidenceAccessContext,
    ) -> Result<StoredEvidence, EvidenceError> {
        let stored = self
            .get(id)
            .ok_or_else(|| EvidenceError::Malformed(format!("unknown evidence id {id}")))?;
        ensure_scope(&stored.envelope, ctx)?;
        Ok(stored)
    }

    /// Raw, unscoped lookup used by storage internals. Callers acting for a
    /// session MUST use [`EvidenceStore::get_scoped`].
    fn get(&self, id: EvidenceId) -> Option<StoredEvidence>;
}

fn ensure_scope(env: &EvidenceEnvelope, ctx: &EvidenceAccessContext) -> Result<(), EvidenceError> {
    let session_ok = env.session_id.raw() == ctx.session_id;
    let workspace_ok = env.workspace_id.raw() == ctx.workspace_id;
    let task_ok = match (env.task_id, ctx.task_id) {
        (Some(env_task), Some(ctx_task)) => env_task == ctx_task,
        _ => true,
    };
    if session_ok && workspace_ok && task_ok {
        return Ok(());
    }
    Err(EvidenceError::AccessDenied(format!(
        "evidence {} belongs to session {} workspace {}; caller is session {} workspace {}",
        env.id, env.session_id, env.workspace_id, ctx.session_id, ctx.workspace_id,
    )))
}

/// In-memory [`EvidenceStore`] with a hard per-envelope backing cap.
#[derive(Debug)]
pub struct MemoryEvidenceStore {
    entries: HashMap<EvidenceId, StoredEvidence>,
    backing_cap: usize,
}

impl MemoryEvidenceStore {
    /// Create a store that retains backing bytes only up to `backing_cap` per
    /// envelope. A cap of `0` keeps every envelope and drops all non-empty
    /// backing.
    pub fn new(backing_cap: usize) -> Self {
        Self {
            entries: HashMap::new(),
            backing_cap,
        }
    }

    pub const fn backing_cap(&self) -> usize {
        self.backing_cap
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn contains(&self, id: EvidenceId) -> bool {
        self.entries.contains_key(&id)
    }

    /// Owned, scope-checked read (the compiler seam): a cross-session or
    /// cross-workspace id is [`EvidenceError::AccessDenied`], exactly like
    /// the borrowed [`EvidenceStore::get_scoped`].
    pub fn stored_scoped(
        &self,
        id: EvidenceId,
        ctx: &EvidenceAccessContext,
    ) -> Result<StoredEvidence, EvidenceError> {
        let stored = self
            .get(id)
            .ok_or_else(|| EvidenceError::Malformed(format!("unknown evidence id {id}")))?;
        ensure_scope(&stored.envelope, ctx)?;
        Ok(stored.clone())
    }

    /// Every envelope visible to `ctx` (bounded by `limit`, ascending id),
    /// scope-checked exactly like the durable listing. Ordering is by id, so
    /// two runs over the same store are deterministic even though the
    /// backing map is hash-ordered.
    pub fn list_scoped_envelopes(
        &self,
        ctx: &EvidenceAccessContext,
        limit: usize,
    ) -> Vec<EvidenceEnvelope> {
        let mut visible: Vec<&EvidenceEnvelope> = self
            .entries
            .values()
            .map(|stored| &stored.envelope)
            .filter(|env| ensure_scope(env, ctx).is_ok())
            .collect();
        visible.sort_by_key(|env| env.id);
        visible.truncate(limit);
        visible.into_iter().cloned().collect()
    }

    /// Backing is retained only when it can actually be re-expanded: the
    /// envelope claims `Reversible`/`Aggressive`, the capture is `Complete`,
    /// and the bytes fit the cap. Every other case drops the bytes (never the
    /// envelope, never the recorded `backing_hash`).
    fn retain_backing(
        env: &EvidenceEnvelope,
        backing: Option<Vec<u8>>,
        cap: usize,
    ) -> Option<Vec<u8>> {
        let bytes = backing?;
        let compactable = matches!(
            env.compressibility,
            Compressibility::Reversible | Compressibility::Aggressive
        );
        if compactable
            && env.backing_completeness == BackingCompleteness::Complete
            && bytes.len() <= cap
        {
            Some(bytes)
        } else {
            None
        }
    }
}

impl EvidenceStore for MemoryEvidenceStore {
    fn insert(
        &mut self,
        env: EvidenceEnvelope,
        backing: Option<Vec<u8>>,
    ) -> Result<(), EvidenceError> {
        if self.entries.contains_key(&env.id) {
            return Err(EvidenceError::Refused(format!(
                "evidence {} already exists; refusing to overwrite stored evidence",
                env.id
            )));
        }
        let backing = Self::retain_backing(&env, backing, self.backing_cap);
        self.entries.insert(
            env.id,
            StoredEvidence {
                envelope: env,
                backing,
            },
        );
        Ok(())
    }

    fn get(&self, id: EvidenceId) -> Option<StoredEvidence> {
        self.entries.get(&id).cloned()
    }
}

// ---------------------------------------------------------------------------
// Durable evidence store (schema v21)
// ---------------------------------------------------------------------------

/// Convert a durable-store failure into the evidence layer's typed error:
/// a conflict is a refused write (never an overwrite), an oversized value
/// stays oversized, and every read-back decode failure is malformed evidence
/// (the caller must not silently substitute).
impl From<faktor_store::StoreError> for EvidenceError {
    fn from(err: faktor_store::StoreError) -> Self {
        match err {
            faktor_store::StoreError::Conflict(m) => EvidenceError::Refused(m),
            faktor_store::StoreError::Oversized(m) => EvidenceError::Malformed(m),
            other => EvidenceError::Malformed(other.to_string()),
        }
    }
}

/// The durable [`EvidenceStore`] over the `faktor-store` v21 `evidence`
/// table: the production evidence authority. Ids are assigned by SQLite's
/// `AUTOINCREMENT` high-water mark, so a daemon restart can never reissue an
/// id; scope checks are the SAME rule the in-memory store enforces (knowing
/// an id — or a backing CAS digest — never grants a cross-session read).
///
/// Backing bytes are content-addressed on disk under `backing_root` by their
/// BLAKE3 hex digest, so identical backing deduplicates and the recorded
/// `backing_cas_hash` is verifiable: a read whose file digest disagrees with
/// the recorded hash is `Malformed`, never silently served.
pub struct DurableEvidenceStore<'a> {
    store: &'a faktor_store::Store,
    backing_root: PathBuf,
    backing_cap: usize,
}

impl std::fmt::Debug for DurableEvidenceStore<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DurableEvidenceStore")
            .field("backing_root", &self.backing_root)
            .field("backing_cap", &self.backing_cap)
            .finish()
    }
}

/// The JSON payload of the audited `compact` column: the grammar-tagged
/// compact body plus the envelope's source revision (the row's `revision`
/// column is the row revision, 1 at insert). Kept local and strict: unknown
/// fields are ignored on read, missing/garbage fields refuse.
#[derive(serde::Serialize, serde::Deserialize)]
struct CompactColumn {
    grammar: String,
    body: String,
    #[serde(default)]
    source_revision: Option<String>,
}

impl<'a> DurableEvidenceStore<'a> {
    /// A durable evidence store over `store` whose backing bytes live under
    /// `backing_root` and whose compact bodies are retained only when the
    /// backing fits `backing_cap` bytes (same retention rule as
    /// [`MemoryEvidenceStore`]).
    pub fn new(
        store: &'a faktor_store::Store,
        backing_root: impl Into<PathBuf>,
        backing_cap: usize,
    ) -> Self {
        Self {
            store,
            backing_root: backing_root.into(),
            backing_cap,
        }
    }

    pub const fn backing_cap(&self) -> usize {
        self.backing_cap
    }

    pub fn backing_root(&self) -> &Path {
        &self.backing_root
    }

    /// Insert a NEW envelope, assigning a fresh globally-unique id. The
    /// returned envelope is the durable truth: its id is the assigned row
    /// id and its `backing_hash` is the digest of the bytes actually written
    /// (recomputed — a stale caller hash is replaced, never trusted). An
    /// envelope whose id is non-zero is inserted explicitly; a collision is
    /// a typed refusal, never an overwrite.
    pub fn insert_new(
        &self,
        envelope: &EvidenceEnvelope,
        backing: Option<&[u8]>,
    ) -> Result<EvidenceEnvelope, EvidenceError> {
        let (row_backing, stored_hash) = self.persist_backing(envelope, backing)?;
        let row = faktor_store::EvidenceRow {
            id: envelope.id.0,
            session_id: envelope.session_id,
            workspace_id: envelope.workspace_id,
            task_id: envelope.task_id,
            kind: kind_str(envelope.kind)?,
            revision: 1,
            provenance_json: serde_json::to_string(&envelope.provenance)
                .map_err(|e| EvidenceError::Malformed(e.to_string()))?,
            compressibility: serde_json::to_string(&envelope.compressibility)
                .map_err(|e| EvidenceError::Malformed(e.to_string()))?
                .trim_matches('"')
                .to_string(),
            compression_json: serde_json::to_string(&envelope.compression)
                .map_err(|e| EvidenceError::Malformed(e.to_string()))?,
            retrieval_json: serde_json::to_string(&envelope.retrieval)
                .map_err(|e| EvidenceError::Malformed(e.to_string()))?,
            compact_json: serde_json::to_string(&CompactColumn {
                grammar: envelope.compact.grammar.clone(),
                body: envelope.compact.body.clone(),
                source_revision: envelope.source_revision.clone(),
            })
            .map_err(|e| EvidenceError::Malformed(e.to_string()))?,
            backing_cas_hash: row_backing,
            completeness: serde_json::to_string(&envelope.backing_completeness)
                .map_err(|e| EvidenceError::Malformed(e.to_string()))?
                .trim_matches('"')
                .to_string(),
            created_ms: now_ms(),
        };
        let id = self.store.evidence_insert(&row)?;
        let mut stored = envelope.clone();
        stored.id = EvidenceId(id);
        if let Some(hash) = stored_hash {
            stored.backing_hash = Some(hash);
        }
        Ok(stored)
    }

    /// Unscoped durable read. Callers acting for a session MUST use
    /// [`DurableEvidenceStore::get_scoped`].
    pub fn get(&self, id: EvidenceId) -> Result<Option<StoredEvidence>, EvidenceError> {
        let Some(row) = self.store.evidence_get(id.0)? else {
            return Ok(None);
        };
        Ok(Some(self.stored_from_row(&row)?))
    }

    /// Scope-checked read: session and workspace must match (task compared
    /// only when both sides name one), exactly like the in-memory store.
    pub fn get_scoped(
        &self,
        id: EvidenceId,
        ctx: &EvidenceAccessContext,
    ) -> Result<StoredEvidence, EvidenceError> {
        let stored = self
            .get(id)?
            .ok_or_else(|| EvidenceError::Malformed(format!("unknown evidence id {id}")))?;
        ensure_scope(&stored.envelope, ctx)?;
        Ok(stored)
    }

    /// Read backing bytes by their CAS digest for `ctx`. The digest is an
    /// INDEX, not a capability: every referencing envelope's scope is
    /// checked and a digest whose owners all belong to another session is
    /// [`EvidenceError::AccessDenied`] — a known digest never crosses the
    /// session boundary.
    pub fn read_backing_by_digest(
        &self,
        digest_hex: &str,
        ctx: &EvidenceAccessContext,
    ) -> Result<Vec<u8>, EvidenceError> {
        let normalized = normalize_digest(digest_hex)?;
        let ids = self.store.evidence_ids_by_backing(&normalized, 1024)?;
        if ids.is_empty() {
            return Err(EvidenceError::Malformed(format!(
                "unknown backing digest {normalized}"
            )));
        }
        let mut denied: Option<EvidenceError> = None;
        for id in ids {
            let Some(row) = self.store.evidence_get(id)? else {
                continue;
            };
            let env = envelope_from_row(&row)?;
            match ensure_scope(&env, ctx) {
                Ok(()) => {
                    return self.read_backing_file(&normalized)?.ok_or_else(|| {
                        EvidenceError::Malformed(format!(
                            "backing {normalized} is recorded but its bytes are absent"
                        ))
                    })
                }
                Err(err) => denied = Some(err),
            }
        }
        Err(denied.unwrap_or_else(|| {
            EvidenceError::AccessDenied(format!(
                "backing digest {normalized} has no envelope visible to this scope"
            ))
        }))
    }

    /// The dedupe probe producers use before archiving: an envelope visible
    /// to `ctx` whose backing digest, kind and source revision match is the
    /// SAME evidence (re-archiving identical output returns it instead of
    /// minting an unbounded stream of duplicate rows).
    pub fn find_scoped_by_backing(
        &self,
        digest_hex: &str,
        kind: crate::types::EvidenceKind,
        source_revision: Option<&str>,
        ctx: &EvidenceAccessContext,
    ) -> Result<Option<EvidenceEnvelope>, EvidenceError> {
        let normalized = normalize_digest(digest_hex)?;
        for id in self.store.evidence_ids_by_backing(&normalized, 64)? {
            let Some(row) = self.store.evidence_get(id)? else {
                continue;
            };
            let env = envelope_from_row(&row)?;
            if env.kind == kind
                && env.source_revision.as_deref() == source_revision
                && ensure_scope(&env, ctx).is_ok()
            {
                return Ok(Some(env));
            }
        }
        Ok(None)
    }

    /// Normalize + compress + store one producer payload (run_command
    /// output, verification output, diagnostics, tests, search hits, MCP
    /// output, semantic context, child handoff, a large read, compaction
    /// backing) as one evidence envelope whose compact body is BOUNDED by
    /// `compact_body_cap` and whose backing is retrievable through the CAS.
    ///
    /// Identical bytes for the same kind/revision and scope are the SAME
    /// evidence: the pre-insert probe returns the existing envelope, so a
    /// producer that re-runs every turn cannot grow the table unboundedly.
    #[allow(clippy::too_many_arguments)]
    pub fn archive_text(
        &self,
        session_id: faktor_core::SessionId,
        workspace_id: faktor_core::WorkspaceId,
        task_id: Option<u64>,
        kind: crate::types::EvidenceKind,
        source_revision: Option<&str>,
        provenance: crate::types::ProvenanceSource,
        raw: &str,
        compact_body_cap: usize,
    ) -> Result<EvidenceEnvelope, EvidenceError> {
        let digest = blake3::hash(raw.as_bytes());
        let digest_hex = hex(digest.as_bytes());
        let ctx = EvidenceAccessContext::new(session_id.raw(), workspace_id.raw(), task_id);
        if let Some(existing) =
            self.find_scoped_by_backing(&digest_hex, kind, source_revision, &ctx)?
        {
            return Ok(existing);
        }
        let (compact, record) =
            crate::compress::compress(&kind, BackingCompleteness::Complete, raw)?;
        if compact.body.len() > compact_body_cap {
            return Err(EvidenceError::Oversized {
                max: compact_body_cap,
                actual: compact.body.len(),
            });
        }
        let envelope = EvidenceEnvelope::new(
            EvidenceId(0),
            kind,
            session_id,
            workspace_id,
            task_id,
            source_revision.map(str::to_string),
            crate::types::ProvenanceSet::new([provenance]),
            kind.default_compressibility(),
            compact,
            Some(*digest.as_bytes()),
            BackingCompleteness::Complete,
            record,
            crate::types::RetrievalPolicy::new(true, true, compact_body_cap.max(4096)),
        )?;
        self.insert_new(&envelope, Some(raw.as_bytes()))
    }

    /// Every evidence id visible to `ctx` (bounded listing), oldest first.
    pub fn list_scoped(
        &self,
        ctx: &EvidenceAccessContext,
        limit: usize,
    ) -> Result<Vec<EvidenceId>, EvidenceError> {
        let rows = self.store.evidence_list_by_scope(
            faktor_core::SessionId::new(ctx.session_id),
            faktor_core::WorkspaceId::new(ctx.workspace_id),
            limit,
        )?;
        let mut out = Vec::new();
        for row in rows {
            let env = envelope_from_row(&row)?;
            if ensure_scope(&env, ctx).is_ok() {
                out.push(env.id);
            }
        }
        Ok(out)
    }

    /// Every envelope visible to `ctx` (bounded, oldest first). The scoped
    /// listing never crosses the session/workspace boundary: a task-scoped
    /// caller sees its own task plus task-less envelopes of its scope.
    pub fn list_scoped_envelopes(
        &self,
        ctx: &EvidenceAccessContext,
        limit: usize,
    ) -> Result<Vec<EvidenceEnvelope>, EvidenceError> {
        let rows = self.store.evidence_list_by_scope(
            faktor_core::SessionId::new(ctx.session_id),
            faktor_core::WorkspaceId::new(ctx.workspace_id),
            limit,
        )?;
        let mut out = Vec::new();
        for row in rows {
            let env = envelope_from_row(&row)?;
            if ensure_scope(&env, ctx).is_ok() {
                out.push(env);
            }
        }
        Ok(out)
    }

    /// Backing retention + CAS write, mirroring the in-memory rule: bytes
    /// are written only for compactable, complete captures within the cap.
    /// Returns `(backing_cas_hash, actual_digest)`.
    fn persist_backing(
        &self,
        envelope: &EvidenceEnvelope,
        backing: Option<&[u8]>,
    ) -> Result<(Option<String>, Option<[u8; 32]>), EvidenceError> {
        let Some(bytes) = backing else {
            // No bytes presented: record the envelope's declared hash only
            // when it has one (an honest reference to backing captured
            // elsewhere); never invent one.
            return Ok((
                envelope.backing_hash.as_ref().map(hex),
                envelope.backing_hash,
            ));
        };
        let digest = *blake3::hash(bytes).as_bytes();
        let compactable = matches!(
            envelope.compressibility,
            Compressibility::Reversible | Compressibility::Aggressive
        );
        if !compactable
            || envelope.backing_completeness != BackingCompleteness::Complete
            || bytes.len() > self.backing_cap
        {
            // Dropped by policy: the row still records the digest of what
            // was presented (the envelope's own hash when set), so a later
            // audit can tell "dropped" from "never captured".
            let recorded = envelope.backing_hash.unwrap_or(digest);
            return Ok((Some(hex(&recorded)), Some(recorded)));
        }
        let hex_digest = hex(&digest);
        let path = self.backing_root.join(&hex_digest);
        if !path.exists() {
            std::fs::create_dir_all(&self.backing_root)
                .map_err(|e| EvidenceError::Malformed(format!("backing dir: {e}")))?;
            // Atomic-enough durable capture: write a sibling temp then
            // rename, so a crash never leaves a torn blob at the final name.
            let tmp = self.backing_root.join(format!(".{hex_digest}.tmp"));
            std::fs::write(&tmp, bytes)
                .map_err(|e| EvidenceError::Malformed(format!("backing write: {e}")))?;
            std::fs::rename(&tmp, &path)
                .map_err(|e| EvidenceError::Malformed(format!("backing rename: {e}")))?;
        }
        Ok((Some(hex_digest), Some(digest)))
    }

    fn stored_from_row(
        &self,
        row: &faktor_store::EvidenceRow,
    ) -> Result<StoredEvidence, EvidenceError> {
        let envelope = envelope_from_row(row)?;
        let backing = match &row.backing_cas_hash {
            Some(hash)
                if matches!(
                    envelope.compressibility,
                    Compressibility::Reversible | Compressibility::Aggressive
                ) && envelope.backing_completeness == BackingCompleteness::Complete =>
            {
                // Missing/dropped backing is the memory store's `None`
                // contract; a PRESENT file that fails its digest is a loud
                // corruption (propagated), never silently `None`.
                self.read_backing_file(&normalize_digest(hash)?)?
            }
            _ => None,
        };
        Ok(StoredEvidence { envelope, backing })
    }

    fn read_backing_file(&self, digest_hex: &str) -> Result<Option<Vec<u8>>, EvidenceError> {
        let path = self.backing_root.join(digest_hex);
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            // A missing file is the documented "dropped/absent backing"
            // result; every other read failure is loud.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => {
                return Err(EvidenceError::Malformed(format!(
                    "backing {digest_hex} unreadable: {e}"
                )))
            }
        };
        let actual = hex(blake3::hash(&bytes).as_bytes());
        if actual != digest_hex {
            return Err(EvidenceError::Malformed(format!(
                "backing {digest_hex} hash mismatch: file hashes to {actual}"
            )));
        }
        Ok(Some(bytes))
    }
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn hex(bytes: &[u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// Strict 64-char lowercase-hex digest normalization (uppercase is folded —
/// the same digest spelled differently is the same address).
fn normalize_digest(raw: &str) -> Result<String, EvidenceError> {
    let folded = raw.to_ascii_lowercase();
    if folded.len() != 64 || !folded.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(EvidenceError::Malformed(format!(
            "{raw:?} is not a 64-char hex backing digest"
        )));
    }
    Ok(folded)
}

fn kind_str(kind: crate::types::EvidenceKind) -> Result<String, EvidenceError> {
    serde_json::to_string(&kind)
        .map(|s| s.trim_matches('"').to_string())
        .map_err(|e| EvidenceError::Malformed(e.to_string()))
}

/// Decode one durable row back into a validated envelope. Every field is
/// parsed loudly; a row that cannot produce a valid envelope (unknown enum
/// tag, invariant violation, missing compact shape) is `Malformed`, never a
/// guessed default.
fn envelope_from_row(row: &faktor_store::EvidenceRow) -> Result<EvidenceEnvelope, EvidenceError> {
    let kind: crate::types::EvidenceKind =
        serde_json::from_value(serde_json::Value::String(row.kind.clone()))
            .map_err(|e| EvidenceError::Malformed(format!("evidence {} kind: {e}", row.id)))?;
    let compressibility: Compressibility = serde_json::from_value(serde_json::Value::String(
        row.compressibility.clone(),
    ))
    .map_err(|e| EvidenceError::Malformed(format!("evidence {} compressibility: {e}", row.id)))?;
    let completeness: BackingCompleteness = serde_json::from_value(serde_json::Value::String(
        row.completeness.clone(),
    ))
    .map_err(|e| EvidenceError::Malformed(format!("evidence {} completeness: {e}", row.id)))?;
    let provenance: crate::types::ProvenanceSet = serde_json::from_str(&row.provenance_json)
        .map_err(|e| EvidenceError::Malformed(format!("evidence {} provenance: {e}", row.id)))?;
    let compression: crate::types::CompressionRecord = serde_json::from_str(&row.compression_json)
        .map_err(|e| EvidenceError::Malformed(format!("evidence {} compression: {e}", row.id)))?;
    let retrieval: crate::types::RetrievalPolicy = serde_json::from_str(&row.retrieval_json)
        .map_err(|e| EvidenceError::Malformed(format!("evidence {} retrieval: {e}", row.id)))?;
    let compact_payload: CompactColumn = serde_json::from_str(&row.compact_json)
        .map_err(|e| EvidenceError::Malformed(format!("evidence {} compact: {e}", row.id)))?;
    let backing_hash =
        match &row.backing_cas_hash {
            Some(raw) => Some(decode_digest_hex(raw).map_err(|e| {
                EvidenceError::Malformed(format!("evidence {} backing: {e}", row.id))
            })?),
            None => None,
        };
    EvidenceEnvelope::new(
        EvidenceId(row.id),
        kind,
        row.session_id,
        row.workspace_id,
        row.task_id,
        compact_payload.source_revision,
        provenance,
        compressibility,
        crate::types::CompactRepresentation {
            grammar: compact_payload.grammar,
            body: compact_payload.body,
        },
        backing_hash,
        completeness,
        compression,
        retrieval,
    )
}

fn decode_digest_hex(raw: &str) -> Result<[u8; 32], String> {
    let folded = normalize_digest(raw).map_err(|e| e.to_string())?;
    let mut out = [0u8; 32];
    for (i, chunk) in folded.as_bytes().chunks(2).enumerate() {
        let s = std::str::from_utf8(chunk).map_err(|e| e.to_string())?;
        out[i] = u8::from_str_radix(s, 16).map_err(|e| e.to_string())?;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Owned durable evidence authority
// ---------------------------------------------------------------------------

/// Owned handle to the durable evidence authority (schema v21): the
/// production daemon graph holds this `Arc`. [`DurableEvidenceStore`] is a
/// borrowed VIEW reconstructed per call over the SAME rows, so reopening
/// the same store directory reopens the same ids and backing bytes — and a
/// second daemon instance cannot mint an id this one handed out (the
/// AUTOINCREMENT high-water mark is durable).
#[derive(Clone)]
pub struct DurableEvidenceAuthority {
    store: std::sync::Arc<faktor_store::Store>,
    backing_root: PathBuf,
    backing_cap: usize,
}

impl std::fmt::Debug for DurableEvidenceAuthority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DurableEvidenceAuthority")
            .field("backing_root", &self.backing_root)
            .field("backing_cap", &self.backing_cap)
            .finish()
    }
}

impl DurableEvidenceAuthority {
    /// The default backing root beside a store: `<store root>/evidence-cas`.
    /// Deterministic, so a restart finds the SAME files without extra state.
    pub fn for_store(store: std::sync::Arc<faktor_store::Store>, backing_cap: usize) -> Self {
        let root = store.root().join("evidence-cas");
        Self::new(store, root, backing_cap)
    }

    pub fn new(
        store: std::sync::Arc<faktor_store::Store>,
        backing_root: impl Into<PathBuf>,
        backing_cap: usize,
    ) -> Self {
        Self {
            store,
            backing_root: backing_root.into(),
            backing_cap,
        }
    }

    pub fn store(&self) -> &std::sync::Arc<faktor_store::Store> {
        &self.store
    }

    pub fn backing_root(&self) -> &Path {
        &self.backing_root
    }

    pub const fn backing_cap(&self) -> usize {
        self.backing_cap
    }

    fn view(&self) -> DurableEvidenceStore<'_> {
        DurableEvidenceStore::new(&self.store, &self.backing_root, self.backing_cap)
    }

    /// Insert a NEW envelope, assigning a fresh durable id. Delegates to the
    /// borrowed view; the returned envelope is the durable truth.
    pub fn insert_new(
        &self,
        envelope: &EvidenceEnvelope,
        backing: Option<&[u8]>,
    ) -> Result<EvidenceEnvelope, EvidenceError> {
        self.view().insert_new(envelope, backing)
    }

    /// Scope-checked read (session + workspace must match; task compared
    /// only when both sides name one).
    pub fn get_scoped(
        &self,
        id: EvidenceId,
        ctx: &EvidenceAccessContext,
    ) -> Result<StoredEvidence, EvidenceError> {
        self.view().get_scoped(id, ctx)
    }

    /// Every envelope visible to `ctx`, oldest first, bounded by `limit`.
    pub fn list_scoped_envelopes(
        &self,
        ctx: &EvidenceAccessContext,
        limit: usize,
    ) -> Result<Vec<EvidenceEnvelope>, EvidenceError> {
        self.view().list_scoped_envelopes(ctx, limit)
    }

    /// Scope-checked backing read by CAS digest (a known digest is an INDEX,
    /// never a capability: a digest whose owners all belong elsewhere is
    /// [`EvidenceError::AccessDenied`]).
    pub fn read_backing_by_digest(
        &self,
        digest_hex: &str,
        ctx: &EvidenceAccessContext,
    ) -> Result<Vec<u8>, EvidenceError> {
        self.view().read_backing_by_digest(digest_hex, ctx)
    }

    /// Normalize + compress + store one producer payload as durable evidence
    /// (deduplicated by digest/kind/revision within the scope). The compact
    /// body is bounded; the backing bytes are retrievable through the CAS.
    #[allow(clippy::too_many_arguments)]
    pub fn archive_text(
        &self,
        session_id: faktor_core::SessionId,
        workspace_id: faktor_core::WorkspaceId,
        task_id: Option<u64>,
        kind: crate::types::EvidenceKind,
        source_revision: Option<&str>,
        provenance: crate::types::ProvenanceSource,
        raw: &str,
        compact_body_cap: usize,
    ) -> Result<EvidenceEnvelope, EvidenceError> {
        self.view().archive_text(
            session_id,
            workspace_id,
            task_id,
            kind,
            source_revision,
            provenance,
            raw,
            compact_body_cap,
        )
    }
}

/// The owned durable authority satisfies the same [`EvidenceStore`] contract
/// as the in-memory store (server/native handlers hold it as a trait
/// object), so the daemon's evidence surface is the durable store of record:
/// ids are globally unique across restart and scope checks are identical.
impl EvidenceStore for DurableEvidenceAuthority {
    fn insert(
        &mut self,
        env: EvidenceEnvelope,
        backing: Option<Vec<u8>>,
    ) -> Result<(), EvidenceError> {
        self.insert_new(&env, backing.as_deref()).map(|_| ())
    }

    fn get(&self, id: EvidenceId) -> Option<StoredEvidence> {
        self.view().get(id).ok().flatten()
    }

    fn get_scoped(
        &self,
        id: EvidenceId,
        ctx: &EvidenceAccessContext,
    ) -> Result<StoredEvidence, EvidenceError> {
        DurableEvidenceAuthority::get_scoped(self, id, ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        CompactRepresentation, CompressionRecord, EvidenceKind, ProvenanceSet, ProvenanceSource,
        RetrievalPolicy,
    };
    use faktor_core::{SessionId, WorkspaceId};

    fn envelope(
        id: u64,
        session: u64,
        workspace: u64,
        task: Option<u64>,
        compressibility: Compressibility,
        completeness: BackingCompleteness,
    ) -> EvidenceEnvelope {
        let backing_hash = matches!(
            compressibility,
            Compressibility::Reversible | Compressibility::Aggressive
        )
        .then_some([3u8; 32]);
        EvidenceEnvelope::new(
            EvidenceId(id),
            EvidenceKind::ProcessLog,
            SessionId::new(session),
            WorkspaceId::new(workspace),
            task,
            None,
            ProvenanceSet::new([ProvenanceSource::Tool]),
            compressibility,
            CompactRepresentation {
                grammar: "log-v1".to_string(),
                body: "bounded compact body".to_string(),
            },
            backing_hash,
            completeness,
            CompressionRecord::identity(17),
            RetrievalPolicy::new(true, true, 64),
        )
        .unwrap()
    }

    #[test]
    fn scope_is_checked_even_when_the_id_is_known() {
        let mut store = MemoryEvidenceStore::new(1024);
        store
            .insert(
                envelope(
                    7,
                    1,
                    2,
                    Some(3),
                    Compressibility::Aggressive,
                    BackingCompleteness::Complete,
                ),
                Some(vec![1, 2, 3]),
            )
            .unwrap();

        let owner = EvidenceAccessContext::new(1, 2, Some(3));
        assert_eq!(
            store
                .get_scoped(EvidenceId(7), &owner)
                .unwrap()
                .backing_len(),
            Some(3)
        );

        // Same session, different workspace: denied despite knowing id 7.
        let err = store
            .get_scoped(EvidenceId(7), &EvidenceAccessContext::new(1, 9, Some(3)))
            .unwrap_err();
        assert!(matches!(err, EvidenceError::AccessDenied(_)), "{err:?}");

        // Same session+workspace, different task: denied.
        let err = store
            .get_scoped(EvidenceId(7), &EvidenceAccessContext::new(1, 2, Some(4)))
            .unwrap_err();
        assert!(matches!(err, EvidenceError::AccessDenied(_)), "{err:?}");

        // Different session: denied.
        let err = store
            .get_scoped(EvidenceId(7), &EvidenceAccessContext::new(8, 2, Some(3)))
            .unwrap_err();
        assert!(matches!(err, EvidenceError::AccessDenied(_)), "{err:?}");

        // Task is only compared when both sides name one: a task-less reader
        // sees the tasked envelope.
        let taskless = EvidenceAccessContext::new(1, 2, None);
        assert!(store.get_scoped(EvidenceId(7), &taskless).is_ok());

        // A task-scoped reader sees task-less evidence.
        store
            .insert(
                envelope(
                    8,
                    1,
                    2,
                    None,
                    Compressibility::Aggressive,
                    BackingCompleteness::Complete,
                ),
                None,
            )
            .unwrap();
        assert!(store.get_scoped(EvidenceId(8), &owner).is_ok());

        // `get` is raw and unscoped; unknown ids are None, not an error.
        assert!(store.get(EvidenceId(7)).is_some());
        assert!(store.get(EvidenceId(999)).is_none());
        assert_eq!(store.len(), 2);
        assert!(!store.is_empty());
        assert!(store.contains(EvidenceId(8)));
    }

    #[test]
    fn unknown_scoped_id_is_malformed_not_a_denial() {
        let store = MemoryEvidenceStore::new(8);
        let err = store
            .get_scoped(EvidenceId(1), &EvidenceAccessContext::new(1, 2, None))
            .unwrap_err();
        assert!(matches!(err, EvidenceError::Malformed(_)), "{err:?}");
        assert!(err.to_string().contains('1'), "{err}");
    }

    #[test]
    fn backing_cap_keeps_envelope_and_drops_oversized_bytes() {
        let mut store = MemoryEvidenceStore::new(4);

        // Exactly at the cap is retained.
        store
            .insert(
                envelope(
                    1,
                    1,
                    2,
                    None,
                    Compressibility::Aggressive,
                    BackingCompleteness::Complete,
                ),
                Some(vec![0u8; 4]),
            )
            .unwrap();
        assert_eq!(
            store.get(EvidenceId(1)).unwrap().backing.as_deref(),
            Some(&[0u8; 4][..])
        );

        // Over the cap: envelope kept, bytes dropped, hash still recorded.
        let env = envelope(
            2,
            1,
            2,
            None,
            Compressibility::Reversible,
            BackingCompleteness::Complete,
        );
        assert!(env.backing_hash.is_some());
        store.insert(env, Some(vec![0u8; 5])).unwrap();
        let stored = store.get(EvidenceId(2)).unwrap();
        assert!(
            stored.backing.is_none(),
            "oversized backing must be dropped"
        );
        assert!(stored.envelope.backing_hash.is_some(), "hash must survive");
        assert_eq!(stored.envelope.id, EvidenceId(2));

        // Never/LosslessOnly evidence never retains bytes.
        store
            .insert(
                envelope(
                    3,
                    1,
                    2,
                    None,
                    Compressibility::LosslessOnly,
                    BackingCompleteness::Complete,
                ),
                Some(vec![1]),
            )
            .unwrap();
        assert!(store.get(EvidenceId(3)).unwrap().backing.is_none());
        store
            .insert(
                envelope(
                    4,
                    1,
                    2,
                    None,
                    Compressibility::Never,
                    BackingCompleteness::Complete,
                ),
                Some(vec![1]),
            )
            .unwrap();
        assert!(store.get(EvidenceId(4)).unwrap().backing.is_none());

        // Truncated capture never retains bytes even when aggressive + small.
        store
            .insert(
                envelope(
                    5,
                    1,
                    2,
                    None,
                    Compressibility::Aggressive,
                    BackingCompleteness::Truncated,
                ),
                Some(vec![1]),
            )
            .unwrap();
        assert!(store.get(EvidenceId(5)).unwrap().backing.is_none());

        // No backing provided stays None.
        store
            .insert(
                envelope(
                    6,
                    1,
                    2,
                    None,
                    Compressibility::Aggressive,
                    BackingCompleteness::Complete,
                ),
                None,
            )
            .unwrap();
        assert!(store.get(EvidenceId(6)).unwrap().backing.is_none());

        // Cap zero retains only the empty backing (0 bytes fit by definition).
        let mut zero = MemoryEvidenceStore::new(0);
        zero.insert(
            envelope(
                1,
                1,
                2,
                None,
                Compressibility::Aggressive,
                BackingCompleteness::Complete,
            ),
            Some(vec![1]),
        )
        .unwrap();
        assert!(zero.get(EvidenceId(1)).unwrap().backing.is_none());
        zero.insert(
            envelope(
                2,
                1,
                2,
                None,
                Compressibility::Aggressive,
                BackingCompleteness::Complete,
            ),
            Some(Vec::new()),
        )
        .unwrap();
        assert_eq!(
            zero.get(EvidenceId(2)).unwrap().backing.as_deref(),
            Some(&[][..])
        );
        assert_eq!(zero.backing_cap(), 0);
    }

    #[test]
    fn duplicate_insert_is_refused_never_a_silent_overwrite() {
        let mut store = MemoryEvidenceStore::new(16);
        store
            .insert(
                envelope(
                    1,
                    1,
                    2,
                    None,
                    Compressibility::Aggressive,
                    BackingCompleteness::Complete,
                ),
                Some(b"first".to_vec()),
            )
            .unwrap();
        let err = store
            .insert(
                envelope(
                    1,
                    1,
                    2,
                    None,
                    Compressibility::Aggressive,
                    BackingCompleteness::Complete,
                ),
                Some(b"second".to_vec()),
            )
            .unwrap_err();
        assert!(matches!(err, EvidenceError::Refused(_)), "{err:?}");
        assert_eq!(
            store.get(EvidenceId(1)).unwrap().backing.as_deref(),
            Some(&b"first"[..]),
            "the original bytes must survive a duplicate insert"
        );
        assert_eq!(store.len(), 1);
    }

    // -----------------------------------------------------------------
    // Durable evidence store (v21)
    // -----------------------------------------------------------------

    struct DurableFixture {
        _dir: tempfile::TempDir,
        store: faktor_store::Store,
    }

    fn durable() -> DurableFixture {
        let dir = tempfile::tempdir().unwrap();
        let store = faktor_store::Store::open(dir.path().join("db"), true).unwrap();
        DurableFixture { _dir: dir, store }
    }

    fn scoped_session(store: &faktor_store::Store, root: &str) -> (SessionId, WorkspaceId) {
        let ws = store.create_workspace(root).unwrap();
        let sid = store.create_session(ws, "t", "p", "m").unwrap().id;
        (sid, ws)
    }

    fn durable_envelope(
        session: SessionId,
        workspace: WorkspaceId,
        body: &str,
        compressibility: Compressibility,
        completeness: BackingCompleteness,
    ) -> EvidenceEnvelope {
        EvidenceEnvelope::new(
            EvidenceId(0),
            EvidenceKind::ProcessLog,
            session,
            workspace,
            Some(7),
            Some("rev-1".to_string()),
            ProvenanceSet::new([ProvenanceSource::Tool]),
            compressibility,
            CompactRepresentation {
                grammar: "log-v1".to_string(),
                body: body.to_string(),
            },
            None,
            completeness,
            CompressionRecord::identity(17),
            RetrievalPolicy::new(true, true, 64),
        )
        .unwrap()
    }

    #[test]
    fn durable_ids_survive_reopen_and_never_repeat() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("db");
        let cas = dir.path().join("cas");
        let (first, second);
        {
            let store = faktor_store::Store::open(&db, true).unwrap();
            let (sid, ws) = scoped_session(&store, "/w");
            let durable = DurableEvidenceStore::new(&store, &cas, 1024);
            let a = durable
                .insert_new(
                    &durable_envelope(
                        sid,
                        ws,
                        "first",
                        Compressibility::Aggressive,
                        BackingCompleteness::Complete,
                    ),
                    Some(b"backing-one"),
                )
                .unwrap();
            let b = durable
                .insert_new(
                    &durable_envelope(
                        sid,
                        ws,
                        "second",
                        Compressibility::Aggressive,
                        BackingCompleteness::Complete,
                    ),
                    Some(b"backing-two"),
                )
                .unwrap();
            first = a.id;
            second = b.id;
            assert!(first.0 >= 1 && second > first, "ids are monotonic");
        }
        // "Daemon restart": ids resolve, the high-water mark is durable, and
        // a new insert continues above every id ever issued.
        {
            let store = faktor_store::Store::open(&db, true).unwrap();
            let durable = DurableEvidenceStore::new(&store, &cas, 1024);
            let reopened = durable.get(first).unwrap().expect("first id survives");
            assert_eq!(reopened.envelope.compact.body, "first");
            assert_eq!(reopened.backing.as_deref(), Some(&b"backing-one"[..]));
            let third = durable
                .insert_new(
                    &durable_envelope(
                        SessionId::new(1),
                        WorkspaceId::new(1),
                        "third",
                        Compressibility::Aggressive,
                        BackingCompleteness::Complete,
                    ),
                    Some(b"backing-three"),
                )
                .unwrap();
            assert!(
                third.id > second,
                "reopen must never reissue {second}: got {}",
                third.id
            );
            assert_eq!(durable.backing_cap(), 1024);
        }
    }

    #[test]
    fn cross_session_retrieval_is_denied_even_with_the_known_backing_digest() {
        let f = durable();
        let cas = f._dir.path().join("cas");
        let (sid_a, ws) = scoped_session(&f.store, "/w");
        let sid_b = f.store.create_session(ws, "other", "p", "m").unwrap().id;
        let durable = DurableEvidenceStore::new(&f.store, &cas, 1024);
        let stored = durable
            .insert_new(
                &durable_envelope(
                    sid_a,
                    ws,
                    "secret body",
                    Compressibility::Aggressive,
                    BackingCompleteness::Complete,
                ),
                Some(b"private backing bytes"),
            )
            .unwrap();
        let digest = hex(&stored.backing_hash.expect("backing recorded"));

        // The owner can read the bytes by digest.
        let owner = EvidenceAccessContext::new(sid_a.raw(), ws.raw(), Some(7));
        assert_eq!(
            durable.read_backing_by_digest(&digest, &owner).unwrap(),
            b"private backing bytes"
        );
        assert_eq!(
            durable
                .get_scoped(stored.id, &owner)
                .unwrap()
                .backing
                .as_deref(),
            Some(&b"private backing bytes"[..])
        );

        // A different session that KNOWS the digest is denied — both by id
        // and by digest. Knowing the address is not authorization.
        let intruder = EvidenceAccessContext::new(sid_b.raw(), ws.raw(), Some(7));
        match durable.read_backing_by_digest(&digest, &intruder) {
            Err(EvidenceError::AccessDenied(_)) => {}
            other => panic!("known digest must not cross sessions, got {other:?}"),
        }
        match durable.get_scoped(stored.id, &intruder) {
            Err(EvidenceError::AccessDenied(_)) => {}
            other => panic!("known id must not cross sessions, got {other:?}"),
        }
        // The listing is scoped too.
        assert_eq!(durable.list_scoped(&owner, 100).unwrap(), vec![stored.id]);
        assert!(durable.list_scoped(&intruder, 100).unwrap().is_empty());

        // Hostile digests are typed malformed, not denials.
        for bad in ["", "zz", &"A".repeat(63), &"g".repeat(64)] {
            assert!(matches!(
                durable.read_backing_by_digest(bad, &owner),
                Err(EvidenceError::Malformed(_))
            ));
        }
        // Unknown (well-formed) digest is malformed too.
        assert!(matches!(
            durable.read_backing_by_digest(&"ab".repeat(32), &owner),
            Err(EvidenceError::Malformed(_))
        ));
    }

    #[test]
    fn durable_backing_policy_and_corruption_are_honest() {
        let f = durable();
        let cas = f._dir.path().join("cas");
        let (sid, ws) = scoped_session(&f.store, "/w");
        let durable = DurableEvidenceStore::new(&f.store, &cas, 8);
        let owner = EvidenceAccessContext::new(sid.raw(), ws.raw(), Some(7));

        // At the cap: bytes are written and exactly retrievable.
        let kept = durable
            .insert_new(
                &durable_envelope(
                    sid,
                    ws,
                    "kept",
                    Compressibility::Aggressive,
                    BackingCompleteness::Complete,
                ),
                Some(b"12345678"),
            )
            .unwrap();
        assert_eq!(
            durable.get(kept.id).unwrap().unwrap().backing.as_deref(),
            Some(&b"12345678"[..])
        );

        // Over the cap: recorded digest survives, bytes are dropped.
        let dropped = durable
            .insert_new(
                &durable_envelope(
                    sid,
                    ws,
                    "dropped",
                    Compressibility::Aggressive,
                    BackingCompleteness::Complete,
                ),
                Some(b"123456789"),
            )
            .unwrap();
        assert!(dropped.backing_hash.is_some());
        assert!(durable.get(dropped.id).unwrap().unwrap().backing.is_none());

        // Truncated capture never retains bytes even when small.
        let truncated = durable
            .insert_new(
                &durable_envelope(
                    sid,
                    ws,
                    "truncated",
                    Compressibility::Aggressive,
                    BackingCompleteness::Truncated,
                ),
                Some(b"1"),
            )
            .unwrap();
        assert!(durable
            .get(truncated.id)
            .unwrap()
            .unwrap()
            .backing
            .is_none());

        // A duplicate explicit id is refused; the original row survives.
        let mut dup = kept.clone();
        dup.compact.body = "overwrite".into();
        match durable.insert_new(&dup, Some(b"other")) {
            Err(EvidenceError::Refused(_)) => {}
            other => panic!("duplicate id must be refused, got {other:?}"),
        }
        assert_eq!(
            durable.get(kept.id).unwrap().unwrap().envelope.compact.body,
            "kept"
        );

        // Envelope provenance/compression/retrieval/source_revision survive
        // the durable round trip byte-for-byte (the row is a lossless view).
        let stored = durable.get(kept.id).unwrap().unwrap();
        let original = durable_envelope(
            sid,
            ws,
            "kept",
            Compressibility::Aggressive,
            BackingCompleteness::Complete,
        );
        assert_eq!(stored.envelope.provenance, original.provenance);
        assert_eq!(stored.envelope.source_revision, original.source_revision);
        assert_eq!(stored.envelope.retrieval, original.retrieval);
        assert_eq!(stored.envelope.compressibility, original.compressibility);
        assert_eq!(
            stored.envelope.backing_completeness,
            original.backing_completeness
        );

        // Corruption injection LAST (every later read is a loud hash
        // mismatch, never silently wrong content).
        let digest = hex(&kept.backing_hash.unwrap());
        std::fs::write(cas.join(&digest), b"tampered").unwrap();
        match durable.get_scoped(kept.id, &owner) {
            Err(EvidenceError::Malformed(m)) => assert!(m.contains("mismatch"), "{m}"),
            other => panic!("tampered backing must be malformed, got {other:?}"),
        }
        match durable.read_backing_by_digest(&digest, &owner) {
            Err(EvidenceError::Malformed(m)) => assert!(m.contains("mismatch"), "{m}"),
            other => panic!("tampered digest read must be malformed, got {other:?}"),
        }
    }

    #[test]
    fn owned_authority_reopens_ids_and_denies_cross_session_digest_reads() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("db");
        let store = std::sync::Arc::new(faktor_store::Store::open(&db, true).unwrap());
        let (sid_a, ws) = scoped_session(&store, "/w");
        let sid_b = store.create_session(ws, "other", "p", "m").unwrap().id;
        let first;
        let digest;
        {
            // A first daemon instance (authority dropped at scope end).
            let authority = DurableEvidenceAuthority::for_store(store.clone(), 4096);
            let stored = authority
                .insert_new(
                    &durable_envelope(
                        sid_a,
                        ws,
                        "authority body",
                        Compressibility::Aggressive,
                        BackingCompleteness::Complete,
                    ),
                    Some(b"authority backing"),
                )
                .unwrap();
            first = stored.id;
            digest = hex(&stored.backing_hash.expect("backing recorded"));
            assert!(authority.backing_root().ends_with("evidence-cas"));
            // The intruder cannot even list the owner's evidence.
            let intruder = EvidenceAccessContext::new(sid_b.raw(), ws.raw(), Some(7));
            assert!(authority
                .list_scoped_envelopes(&intruder, 100)
                .unwrap()
                .is_empty());
            for bad in ["", "AB", &"A".repeat(63), &"g".repeat(64)] {
                assert!(matches!(
                    authority.read_backing_by_digest(bad, &intruder),
                    Err(EvidenceError::Malformed(_))
                ));
            }
        }
        // "Restart": a fresh authority over the SAME store directory keeps
        // the id and a known backing digest still never crosses sessions.
        let reopened_store = std::sync::Arc::new(faktor_store::Store::open(&db, true).unwrap());
        let authority = DurableEvidenceAuthority::for_store(reopened_store, 4096);
        let owner = EvidenceAccessContext::new(sid_a.raw(), ws.raw(), Some(7));
        assert_eq!(
            authority
                .get_scoped(first, &owner)
                .unwrap()
                .envelope
                .compact
                .body,
            "authority body"
        );
        assert_eq!(
            authority.read_backing_by_digest(&digest, &owner).unwrap(),
            b"authority backing"
        );
        let intruder = EvidenceAccessContext::new(sid_b.raw(), ws.raw(), Some(7));
        match authority.get_scoped(first, &intruder) {
            Err(EvidenceError::AccessDenied(_)) => {}
            other => panic!("known id must not cross sessions after reopen, got {other:?}"),
        }
        match authority.read_backing_by_digest(&digest, &intruder) {
            Err(EvidenceError::AccessDenied(_)) => {}
            other => panic!("known digest must not cross sessions after reopen, got {other:?}"),
        }
        // A brand-new id issued after reopen is ABOVE every id ever issued.
        let after = authority
            .insert_new(
                &durable_envelope(
                    sid_a,
                    ws,
                    "after reopen",
                    Compressibility::Aggressive,
                    BackingCompleteness::Complete,
                ),
                Some(b"more backing"),
            )
            .unwrap();
        assert!(after.id > first, "reopen minted a stale id {}", after.id);
    }
}
