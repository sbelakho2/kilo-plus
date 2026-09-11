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
use std::path::Path;
use std::sync::Arc;

use faktor_cas::{Cas, CasError};
use faktor_core::hash::FileHash;

use crate::types::{
    BackingCompleteness, Compressibility, EvidenceEnvelope, EvidenceError, EvidenceId,
};

/// The DECLARED scope a reader presents. There is no task-less sentinel: a
/// caller is either acting for exactly one task (`Task`) or is the
/// authenticated session administrator (`SessionAdmin`). A `Task` reader
/// sees its own task's envelopes plus task-less envelopes of its
/// session/workspace; a `SessionAdmin` reader sees every task of its
/// session/workspace. No authorization decision is derived from an absent
/// (None) task id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceAccessScope {
    /// Acting for one durable task: envelope task ids must match (a
    /// task-less envelope is shared session/workspace context, visible to
    /// every task of the scope).
    Task(faktor_core::id::TaskId),
    /// The authenticated session administrator: sees every task of the
    /// session/workspace. Only an authenticated UI may request this.
    SessionAdmin,
}

/// The identity a reader presents: session + workspace + the declared
/// [`EvidenceAccessScope`]. A read is permitted only when all three match
/// the stored envelope's scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvidenceAccessContext {
    pub session_id: u64,
    pub workspace_id: u64,
    pub scope: EvidenceAccessScope,
}

impl EvidenceAccessContext {
    /// Build a context from an explicit scope.
    pub const fn scoped(session_id: u64, workspace_id: u64, scope: EvidenceAccessScope) -> Self {
        Self {
            session_id,
            workspace_id,
            scope,
        }
    }

    /// A task-scoped reader (runtime/model path: exactly one durable task).
    pub const fn for_task(
        session_id: u64,
        workspace_id: u64,
        task_id: faktor_core::id::TaskId,
    ) -> Self {
        Self::scoped(session_id, workspace_id, EvidenceAccessScope::Task(task_id))
    }

    /// The explicit session-administrator reader (authenticated native UI).
    pub const fn admin(session_id: u64, workspace_id: u64) -> Self {
        Self::scoped(session_id, workspace_id, EvidenceAccessScope::SessionAdmin)
    }

    /// Compatibility/test constructor for the historic `Option` task form:
    /// `Some(task)` is the task scope; `None` maps to the EXPLICIT admin
    /// scope (a caller asking for no task boundary must mean admin, never an
    /// accidental wildcard). Production runtime paths use [`Self::for_task`].
    pub const fn new(session_id: u64, workspace_id: u64, task_id: Option<u64>) -> Self {
        match task_id {
            Some(task) => Self::scoped(
                session_id,
                workspace_id,
                EvidenceAccessScope::Task(faktor_core::id::TaskId::new(task)),
            ),
            None => Self::admin(session_id, workspace_id),
        }
    }

    /// The declared scope of this context.
    pub const fn scope(&self) -> EvidenceAccessScope {
        self.scope
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

    /// Unscoped existence probe that never materializes backing bytes. The
    /// default implementation is the honest `get`-based fallback; durable
    /// stores override it so a 404/403 decision never pays a CAS read.
    fn exists(&self, id: EvidenceId) -> Result<bool, EvidenceError> {
        Ok(self.get(id).is_some())
    }

    /// Certification seam: the allocation identity of the sharing authority
    /// behind this store, when it has one. Two handles with the same
    /// non-`None` value are the SAME authority instance. Never a security
    /// decision input; `None` for stores with no shared authority identity.
    #[doc(hidden)]
    fn authority_ptr(&self) -> Option<usize> {
        None
    }
}

fn ensure_scope(env: &EvidenceEnvelope, ctx: &EvidenceAccessContext) -> Result<(), EvidenceError> {
    let session_ok = env.session_id.raw() == ctx.session_id;
    let workspace_ok = env.workspace_id.raw() == ctx.workspace_id;
    let task_ok = match ctx.scope {
        EvidenceAccessScope::Task(task) => env
            .task_id
            .map(|env_task| env_task == task.raw())
            .unwrap_or(true),
        EvidenceAccessScope::SessionAdmin => true,
    };
    if session_ok && workspace_ok && task_ok {
        return Ok(());
    }
    Err(EvidenceError::AccessDenied(format!(
        "evidence {} belongs to session {} workspace {} task {}; caller is session {} workspace {} scope {:?}",
        env.id, env.session_id, env.workspace_id,
        env.task_id.map(|t| t.to_string()).unwrap_or_else(|| "none".to_string()),
        ctx.session_id, ctx.workspace_id, ctx.scope,
    )))
}

/// One bounded newest-first page of scoped evidence envelopes. `rows` are
/// ordered `created_ms DESC, id DESC` (newest first); `next_before` is the
/// exclusive id cursor for the next page, or `None` when the page was the
/// last (empty input, or the caller consumed the whole scope).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EvidencePage {
    pub rows: Vec<EvidenceEnvelope>,
    pub next_before: Option<u64>,
}

// ---------------------------------------------------------------------------
// Deterministic evidence-insert crash seams (fault-certification campaigns)
//
// One-shot fault injection at the two durable boundaries of one evidence
// insert: after the backing blob is persisted in the CAS but BEFORE the row
// exists, and after the row is committed but BEFORE the caller sees the
// response. The seam is inert unless armed and fires at most once per arm.
// The CAS's own crash seam covers the boundary inside the put.
// ---------------------------------------------------------------------------

/// One-shot evidence-insert fault target: panic at the `ordinal`-th
/// crossing (0-based) of durability boundary `point`.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvidenceCrashArm {
    /// `evidence_blob_persisted` (CAS put returned, row not yet inserted) or
    /// `evidence_row_inserted` (row committed, response not yet returned).
    pub point: &'static str,
    pub ordinal: u64,
}

#[derive(Default)]
struct EvidenceSeamState {
    armed: Option<EvidenceCrashArm>,
    crossings: u64,
}

/// Per-authority deterministic evidence-insert crash seam. Additive and
/// default-off: while unarmed every `trip` is a single uncontended mutex
/// check.
#[doc(hidden)]
#[derive(Default)]
pub struct EvidenceCrashSeam {
    state: std::sync::Mutex<EvidenceSeamState>,
}

impl std::fmt::Debug for EvidenceCrashSeam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let armed = self.state.lock().map(|s| s.armed).unwrap_or(None);
        f.debug_struct("EvidenceCrashSeam")
            .field("armed", &armed)
            .finish()
    }
}

impl EvidenceCrashSeam {
    /// Arm ONE crossing, replacing any previous arm and resetting the
    /// crossing counter.
    pub fn arm(&self, arm: EvidenceCrashArm) {
        let mut s = self.state.lock().unwrap_or_else(|p| p.into_inner());
        *s = EvidenceSeamState {
            armed: Some(arm),
            crossings: 0,
        };
    }

    /// Trip the seam at `point`. Panics when the armed crossing is hit.
    fn trip(&self, point: &'static str) {
        let mut s = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let Some(arm) = s.armed else {
            return;
        };
        if arm.point != point {
            return;
        }
        s.crossings += 1;
        if s.crossings - 1 != arm.ordinal {
            return;
        }
        s.armed = None;
        drop(s);
        panic!(
            "[fault-seam] simulated crash at evidence durability boundary `{point}` (crossing {})",
            arm.ordinal
        );
    }
}

/// In-memory [`EvidenceStore`] with a hard per-envelope backing cap.
#[derive(Debug)]
pub struct MemoryEvidenceStore {
    entries: HashMap<EvidenceId, StoredEvidence>,
    /// Insertion order (oldest first) so the newest-first page is a
    /// deterministic reverse walk, exactly like the durable
    /// `created_ms DESC, id DESC` order.
    order: Vec<EvidenceId>,
    backing_cap: usize,
}

impl MemoryEvidenceStore {
    /// Create a store that retains backing bytes only up to `backing_cap` per
    /// envelope. A cap of `0` keeps every envelope and drops all non-empty
    /// backing.
    pub fn new(backing_cap: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: Vec::new(),
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

    /// The newest-first scoped page (mirror of the durable
    /// `ORDER BY created_ms DESC, id DESC`): a reverse walk of the insertion
    /// order with the exclusive `before` id cursor. `next_before` is the
    /// last returned id, or `None` when the scope was exhausted or the page
    /// was empty.
    pub fn list_scoped_newest(
        &self,
        ctx: &EvidenceAccessContext,
        before: Option<u64>,
        limit: usize,
    ) -> EvidencePage {
        let mut rows = Vec::new();
        if limit == 0 {
            return EvidencePage::default();
        }
        for id in self.order.iter().rev() {
            if let Some(before) = before {
                if id.0 >= before {
                    continue;
                }
            }
            let Some(stored) = self.entries.get(id) else {
                continue;
            };
            if ensure_scope(&stored.envelope, ctx).is_err() {
                continue;
            }
            rows.push(stored.envelope.clone());
            if rows.len() == limit {
                break;
            }
        }
        let next_before = rows.last().map(|env| env.id.0);
        EvidencePage { rows, next_before }
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
        self.order.push(env.id);
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
/// table: the production evidence authority's borrowed view. Ids are
/// assigned by SQLite's `AUTOINCREMENT` high-water mark, so a daemon restart
/// can never reissue an id; scope checks are the SAME rule the in-memory
/// store enforces (knowing an id — or a backing CAS digest — never grants a
/// cross-session read).
///
/// Backing bytes are stored in the shared [`faktor_cas::Cas`] (BLAKE3
/// identity + fsync + atomic rename + verified reads); the SQL row stores
/// the canonical [`faktor_core::hash::FileHash`] hex. Retrieval goes through
/// [`Cas::get_verified_now`], so a torn or corrupt blob is a loud typed
/// error, never silently served content.
pub struct DurableEvidenceStore<'a> {
    store: &'a faktor_store::Store,
    cas: &'a Arc<Cas>,
    backing_cap: usize,
    seam: &'a EvidenceCrashSeam,
}

impl std::fmt::Debug for DurableEvidenceStore<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DurableEvidenceStore")
            .field("backing_root", &self.cas.root())
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
    /// A durable evidence store view over `store` whose backing bytes live
    /// in `cas` and whose compact bodies are retained only when the backing
    /// fits `backing_cap` bytes (same retention rule as
    /// [`MemoryEvidenceStore`]).
    pub fn new(
        store: &'a faktor_store::Store,
        cas: &'a Arc<Cas>,
        seam: &'a EvidenceCrashSeam,
        backing_cap: usize,
    ) -> Self {
        Self {
            store,
            cas,
            backing_cap,
            seam,
        }
    }

    pub const fn backing_cap(&self) -> usize {
        self.backing_cap
    }

    pub fn backing_root(&self) -> &Path {
        self.cas.root()
    }

    /// The shared content-addressed backing store.
    pub fn cas(&self) -> &Arc<Cas> {
        self.cas
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
        // Durability boundary: the blob is fully persisted (or policy-dropped)
        // and the row does not exist yet. A crash here leaves an unreferenced
        // blob — tolerated by the CAS — and NEVER a row without its bytes.
        self.seam.trip("evidence_blob_persisted");
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
        // Durability boundary: the row is committed against a blob that was
        // persisted first, so retrieval after a crash here always resolves.
        self.seam.trip("evidence_row_inserted");
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
    ///
    /// Kept for compatibility/audit listings; the compiler's retrieval path
    /// uses [`DurableEvidenceStore::list_scoped_newest`], because an
    /// oldest-first capped page can never surface recent evidence under a
    /// large table.
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

    /// One newest-first scoped page (`ORDER BY created_ms DESC, id DESC`),
    /// bounded by `limit`; `before` is the exclusive keyset cursor taken from
    /// the previous page's `next_before`. Scope filtering (session, workspace
    /// and the declared task/admin scope) is applied to every row, and
    /// `next_before` advances past the last FETCHED row so a task-filtered
    /// page still makes progress.
    pub fn list_scoped_newest(
        &self,
        ctx: &EvidenceAccessContext,
        before: Option<u64>,
        limit: usize,
    ) -> Result<EvidencePage, EvidenceError> {
        if limit == 0 {
            return Ok(EvidencePage::default());
        }
        let rows = self.store.evidence_list_by_scope_newest(
            faktor_core::SessionId::new(ctx.session_id),
            faktor_core::WorkspaceId::new(ctx.workspace_id),
            before,
            limit,
        )?;
        let next_before = rows.last().map(|row| row.id);
        let mut out = Vec::new();
        for row in rows {
            let env = envelope_from_row(&row)?;
            if ensure_scope(&env, ctx).is_ok() {
                out.push(env);
            }
        }
        Ok(EvidencePage {
            rows: out,
            next_before,
        })
    }

    /// Backing retention + CAS write, mirroring the in-memory rule: bytes
    /// are written only for compactable, complete captures within the cap.
    /// Returns `(backing_cas_hash, actual_digest)`. The CAS owns temp naming,
    /// fsync and atomic publication, so a torn blob can never appear at a
    /// valid address and an absent address is a typed `NotFound` on read.
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
        let hash = self
            .cas
            .put(bytes)
            .map_err(|e| EvidenceError::Malformed(format!("backing cas put: {e}")))?;
        Ok((Some(hash.to_hex()), Some(hash.bytes())))
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
                // contract; a PRESENT blob that fails CAS verification is a
                // loud corruption (propagated), never silently `None`.
                self.read_backing_file(&normalize_digest(hash)?)?
            }
            _ => None,
        };
        Ok(StoredEvidence { envelope, backing })
    }

    /// Verified CAS read by canonical digest hex. A missing blob is the
    /// documented "dropped/absent backing" result; a PRESENT blob that fails
    /// decompression or its address hash is a loud typed corruption — the
    /// caller must never silently substitute content.
    fn read_backing_file(&self, digest_hex: &str) -> Result<Option<Vec<u8>>, EvidenceError> {
        let normalized = normalize_digest(digest_hex)?;
        let hash = FileHash::from_hex(&normalized).ok_or_else(|| {
            EvidenceError::Malformed(format!("{digest_hex:?} is not a BLAKE3 file hash"))
        })?;
        match self.cas.get_verified_now(hash) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(CasError::NotFound(_)) => Ok(None),
            Err(e) => Err(EvidenceError::Malformed(format!(
                "backing {normalized} failed verification: {e}"
            ))),
        }
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
/// production daemon holds ONE `Arc` of this authority, built once and
/// shared by the runtime's `ContextCompiler` and the native server surface.
/// [`DurableEvidenceStore`] is a borrowed VIEW reconstructed per call over
/// the SAME rows and the SAME [`faktor_cas::Cas`], so reopening the same
/// store directory reopens the same ids and backing bytes — and a second
/// daemon instance cannot mint an id this one handed out (the AUTOINCREMENT
/// high-water mark is durable).
#[derive(Clone)]
pub struct DurableEvidenceAuthority {
    store: std::sync::Arc<faktor_store::Store>,
    /// The ONE content-addressed backing store. The SQL row records the
    /// canonical [`FileHash`] hex of the blob written here.
    cas: std::sync::Arc<Cas>,
    backing_cap: usize,
    /// Deterministic fault seam (inert unless armed; tests only).
    seam: std::sync::Arc<EvidenceCrashSeam>,
}

impl std::fmt::Debug for DurableEvidenceAuthority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DurableEvidenceAuthority")
            .field("backing_root", &self.cas.root())
            .field("backing_cap", &self.backing_cap)
            .finish()
    }
}

impl DurableEvidenceAuthority {
    /// The default backing CAS beside a store: `<store root>/evidence-cas`.
    /// Deterministic, so a restart finds the SAME blobs without extra state.
    pub fn for_store(store: std::sync::Arc<faktor_store::Store>, backing_cap: usize) -> Self {
        let root = store.root().join("evidence-cas");
        let cas = Cas::open(root.clone()).unwrap_or_else(|_| Cas::new(root));
        Self::new(store, std::sync::Arc::new(cas), backing_cap)
    }

    pub fn new(
        store: std::sync::Arc<faktor_store::Store>,
        cas: std::sync::Arc<Cas>,
        backing_cap: usize,
    ) -> Self {
        Self {
            store,
            cas,
            backing_cap,
            seam: std::sync::Arc::new(EvidenceCrashSeam::default()),
        }
    }

    pub fn store(&self) -> &std::sync::Arc<faktor_store::Store> {
        &self.store
    }

    /// The shared content-addressed backing store.
    pub fn cas(&self) -> &std::sync::Arc<Cas> {
        &self.cas
    }

    /// The backing root (the CAS root; `<store root>/evidence-cas` by
    /// default).
    pub fn backing_root(&self) -> &Path {
        self.cas.root()
    }

    pub const fn backing_cap(&self) -> usize {
        self.backing_cap
    }

    /// Arm the deterministic evidence-insert crash seam (fault
    /// certification only; see [`EvidenceCrashArm`]). Inert when unarmed.
    #[doc(hidden)]
    pub fn crash_arm(&self, arm: EvidenceCrashArm) {
        self.seam.arm(arm);
    }

    fn view(&self) -> DurableEvidenceStore<'_> {
        DurableEvidenceStore::new(&self.store, &self.cas, &self.seam, self.backing_cap)
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

    /// Scope-checked read (session + workspace must match; the declared
    /// task/admin scope decides task visibility).
    pub fn get_scoped(
        &self,
        id: EvidenceId,
        ctx: &EvidenceAccessContext,
    ) -> Result<StoredEvidence, EvidenceError> {
        self.view().get_scoped(id, ctx)
    }

    /// Unscoped durable read; callers acting for a session MUST use
    /// [`DurableEvidenceAuthority::get_scoped`].
    pub fn get(&self, id: EvidenceId) -> Result<Option<StoredEvidence>, EvidenceError> {
        self.view().get(id)
    }

    /// Every envelope visible to `ctx`, oldest first, bounded by `limit`
    /// (compatibility/audit listing; retrieval uses
    /// [`DurableEvidenceAuthority::list_scoped_newest`]).
    pub fn list_scoped_envelopes(
        &self,
        ctx: &EvidenceAccessContext,
        limit: usize,
    ) -> Result<Vec<EvidenceEnvelope>, EvidenceError> {
        self.view().list_scoped_envelopes(ctx, limit)
    }

    /// One newest-first scoped page (`ORDER BY created_ms DESC, id DESC`).
    pub fn list_scoped_newest(
        &self,
        ctx: &EvidenceAccessContext,
        before: Option<u64>,
        limit: usize,
    ) -> Result<EvidencePage, EvidenceError> {
        self.view().list_scoped_newest(ctx, before, limit)
    }

    /// Every evidence id visible to `ctx`, oldest first, bounded by `limit`.
    pub fn list_scoped(
        &self,
        ctx: &EvidenceAccessContext,
        limit: usize,
    ) -> Result<Vec<EvidenceId>, EvidenceError> {
        self.view().list_scoped(ctx, limit)
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

    /// Dedupe probe used by producers: an envelope visible to `ctx` whose
    /// backing digest, kind and source revision match is the SAME evidence.
    pub fn find_scoped_by_backing(
        &self,
        digest_hex: &str,
        kind: crate::types::EvidenceKind,
        source_revision: Option<&str>,
        ctx: &EvidenceAccessContext,
    ) -> Result<Option<EvidenceEnvelope>, EvidenceError> {
        self.view()
            .find_scoped_by_backing(digest_hex, kind, source_revision, ctx)
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
        DurableEvidenceAuthority::get(self, id).ok().flatten()
    }

    fn get_scoped(
        &self,
        id: EvidenceId,
        ctx: &EvidenceAccessContext,
    ) -> Result<StoredEvidence, EvidenceError> {
        DurableEvidenceAuthority::get_scoped(self, id, ctx)
    }

    fn exists(&self, id: EvidenceId) -> Result<bool, EvidenceError> {
        Ok(self.store.evidence_get(id.0)?.is_some())
    }

    fn authority_ptr(&self) -> Option<usize> {
        Some(std::sync::Arc::as_ptr(&self.cas) as usize)
    }
}

/// A shared `Arc<DurableEvidenceAuthority>` is itself an [`EvidenceStore`]
/// (the native server's boxed handle wraps the SAME allocation the daemon
/// graph and the runtime's compiler hold). `insert` needs a unique handle:
/// the server handle is locked exclusively when it is called, so the
/// `Arc::get_mut` failure path is a typed refusal, never a silent no-op.
impl EvidenceStore for Arc<DurableEvidenceAuthority> {
    fn insert(
        &mut self,
        env: EvidenceEnvelope,
        backing: Option<Vec<u8>>,
    ) -> Result<(), EvidenceError> {
        std::sync::Arc::get_mut(self)
            .ok_or_else(|| {
                EvidenceError::Refused(
                    "evidence authority has other live handles; refusing an aliased insert"
                        .to_string(),
                )
            })?
            .insert_new(&env, backing.as_deref())
            .map(|_| ())
    }

    fn get(&self, id: EvidenceId) -> Option<StoredEvidence> {
        DurableEvidenceAuthority::get(self, id).ok().flatten()
    }

    fn get_scoped(
        &self,
        id: EvidenceId,
        ctx: &EvidenceAccessContext,
    ) -> Result<StoredEvidence, EvidenceError> {
        DurableEvidenceAuthority::get_scoped(self, id, ctx)
    }

    fn exists(&self, id: EvidenceId) -> Result<bool, EvidenceError> {
        Ok(self.store.evidence_get(id.0)?.is_some())
    }

    fn authority_ptr(&self) -> Option<usize> {
        Some(std::sync::Arc::as_ptr(self) as usize)
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
    // Durable evidence store (v21), CAS-backed
    // -----------------------------------------------------------------

    struct DurableFixture {
        _dir: tempfile::TempDir,
        store: Arc<faktor_store::Store>,
        cas: Arc<Cas>,
    }

    fn durable() -> DurableFixture {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(faktor_store::Store::open(dir.path().join("db"), true).unwrap());
        let cas = Arc::new(Cas::open(dir.path().join("evidence-cas")).unwrap());
        DurableFixture {
            _dir: dir,
            store,
            cas,
        }
    }

    fn durable_authority(f: &DurableFixture, cap: usize) -> DurableEvidenceAuthority {
        DurableEvidenceAuthority::new(f.store.clone(), f.cas.clone(), cap)
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
        let cas_root = dir.path().join("evidence-cas");
        let (first, second);
        {
            let store = Arc::new(faktor_store::Store::open(&db, true).unwrap());
            let (sid, ws) = scoped_session(&store, "/w");
            let durable = DurableEvidenceAuthority::new(
                store,
                Arc::new(Cas::open(cas_root.clone()).unwrap()),
                1024,
            );
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
            let store = Arc::new(faktor_store::Store::open(&db, true).unwrap());
            let durable = DurableEvidenceAuthority::new(
                store,
                Arc::new(Cas::open(cas_root.clone()).unwrap()),
                1024,
            );
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
        let (sid_a, ws) = scoped_session(&f.store, "/w");
        let sid_b = f.store.create_session(ws, "other", "p", "m").unwrap().id;
        let durable = durable_authority(&f, 1024);
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
        let (sid, ws) = scoped_session(&f.store, "/w");
        let durable = durable_authority(&f, 8);
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

        // Corruption injection LAST: the stored blob is overwritten in
        // place, so every later read goes through `get_verified_now` and is
        // a loud verification failure, never silently wrong content.
        let digest = hex(&kept.backing_hash.unwrap());
        let blob = f
            .cas
            .root()
            .join(FileHash::from_hex(&digest).unwrap().cas_path());
        std::fs::write(&blob, b"tampered").unwrap();
        match durable.get_scoped(kept.id, &owner) {
            Err(EvidenceError::Malformed(m)) => assert!(m.contains("verification"), "{m}"),
            other => panic!("tampered backing must be malformed, got {other:?}"),
        }
        match durable.read_backing_by_digest(&digest, &owner) {
            Err(EvidenceError::Malformed(m)) => assert!(m.contains("verification"), "{m}"),
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
        let cas_root = dir.path().join("evidence-cas");
        let first;
        let digest;
        {
            // A first daemon instance (authority dropped at scope end).
            let authority = DurableEvidenceAuthority::new(
                store.clone(),
                Arc::new(Cas::open(cas_root.clone()).unwrap()),
                4096,
            );
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
        let authority = DurableEvidenceAuthority::new(
            reopened_store,
            Arc::new(Cas::open(cas_root.clone()).unwrap()),
            4096,
        );
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

    #[test]
    fn task_scope_a_b_and_explicit_admin_are_distinct() {
        let mut store = MemoryEvidenceStore::new(64);
        store
            .insert(
                envelope(
                    1,
                    1,
                    9,
                    Some(1),
                    Compressibility::Aggressive,
                    BackingCompleteness::Complete,
                ),
                None,
            )
            .unwrap();
        store
            .insert(
                envelope(
                    2,
                    1,
                    9,
                    Some(2),
                    Compressibility::Aggressive,
                    BackingCompleteness::Complete,
                ),
                None,
            )
            .unwrap();
        store
            .insert(
                envelope(
                    3,
                    1,
                    9,
                    None,
                    Compressibility::Aggressive,
                    BackingCompleteness::Complete,
                ),
                None,
            )
            .unwrap();

        let task_a = EvidenceAccessContext::for_task(1, 9, faktor_core::id::TaskId::new(1));
        let task_b = EvidenceAccessContext::for_task(1, 9, faktor_core::id::TaskId::new(2));
        let admin = EvidenceAccessContext::admin(1, 9);
        let visible = |ctx: &EvidenceAccessContext| -> Vec<u64> {
            store
                .list_scoped_newest(ctx, None, 16)
                .rows
                .iter()
                .map(|env| env.id.0)
                .collect()
        };
        // Newest-first insertion order; each task sees its own + shared.
        assert_eq!(visible(&task_a), vec![3, 1]);
        assert_eq!(visible(&task_b), vec![3, 2]);
        assert_eq!(visible(&admin), vec![3, 2, 1]);
        // Knowing task B's id is not authorization for task A; the explicit
        // admin scope is the only cross-task reader.
        assert!(matches!(
            store.get_scoped(EvidenceId(2), &task_a),
            Err(EvidenceError::AccessDenied(_))
        ));
        assert!(store.get_scoped(EvidenceId(2), &admin).is_ok());
        // The legacy Option constructor maps Some -> Task, None -> admin;
        // no decision anywhere is derived from an absent task id.
        assert_eq!(EvidenceAccessContext::new(1, 9, Some(1)), task_a);
        assert_eq!(EvidenceAccessContext::new(1, 9, None), admin);
    }

    #[test]
    fn newest_pages_are_ordered_and_cursor_bounded() {
        let mut store = MemoryEvidenceStore::new(64);
        for id in 1..=10u64 {
            store
                .insert(
                    envelope(
                        id,
                        1,
                        2,
                        None,
                        Compressibility::Aggressive,
                        BackingCompleteness::Complete,
                    ),
                    None,
                )
                .unwrap();
        }
        let ctx = EvidenceAccessContext::admin(1, 2);
        let ids =
            |page: &EvidencePage| -> Vec<u64> { page.rows.iter().map(|env| env.id.0).collect() };
        let first = store.list_scoped_newest(&ctx, None, 4);
        assert_eq!(ids(&first), vec![10, 9, 8, 7]);
        assert_eq!(first.next_before, Some(7));
        let second = store.list_scoped_newest(&ctx, first.next_before, 4);
        assert_eq!(ids(&second), vec![6, 5, 4, 3]);
        let third = store.list_scoped_newest(&ctx, second.next_before, 4);
        assert_eq!(ids(&third), vec![2, 1]);
        assert_eq!(third.next_before, Some(1));
        let exhausted = store.list_scoped_newest(&ctx, third.next_before, 4);
        assert!(exhausted.rows.is_empty());
        assert_eq!(exhausted.next_before, None);
        assert!(ids(&store.list_scoped_newest(&ctx, None, 0)).is_empty());
    }

    #[test]
    fn durable_newest_listing_orders_by_id_desc_and_honors_the_cursor() {
        let f = durable();
        let (sid, ws) = scoped_session(&f.store, "/w");
        let authority = durable_authority(&f, 4096);
        let owner = EvidenceAccessContext::new(sid.raw(), ws.raw(), Some(7));
        for i in 0..10 {
            authority
                .insert_new(
                    &durable_envelope(
                        sid,
                        ws,
                        &format!("row-{i}"),
                        Compressibility::Aggressive,
                        BackingCompleteness::Complete,
                    ),
                    None,
                )
                .unwrap();
        }
        let page = authority.list_scoped_newest(&owner, None, 4).unwrap();
        assert_eq!(
            page.rows.iter().map(|e| e.id.0).collect::<Vec<_>>(),
            vec![10, 9, 8, 7]
        );
        assert_eq!(page.next_before, Some(7));
        let rest = authority
            .list_scoped_newest(&owner, page.next_before, 100)
            .unwrap();
        assert_eq!(
            rest.rows.iter().map(|e| e.id.0).collect::<Vec<_>>(),
            vec![6, 5, 4, 3, 2, 1]
        );
        let exhausted = authority
            .list_scoped_newest(&owner, rest.next_before, 10)
            .unwrap();
        assert!(exhausted.rows.is_empty());
        // A task scope that owns no rows still advances the cursor past the
        // last fetched row (progress, never a stall).
        let foreign_task =
            EvidenceAccessContext::for_task(sid.raw(), ws.raw(), faktor_core::id::TaskId::new(999));
        let hidden = authority
            .list_scoped_newest(&foreign_task, None, 4)
            .unwrap();
        assert!(hidden.rows.is_empty());
        assert!(hidden.next_before.is_some(), "the fetched cursor advances");
    }

    #[test]
    fn cas_backed_backing_dedups_across_concurrent_stores_and_leaves_no_temps() {
        // 100 concurrent stores of the SAME 8 MiB payload: exactly one CAS
        // blob, 100 valid references, zero temp debris.
        let f = durable();
        let (sid, ws) = scoped_session(&f.store, "/w");
        let authority = Arc::new(durable_authority(&f, 8 * 1024 * 1024));
        let payload: Vec<u8> = (0..(8 << 20)).map(|i| ((i * 31 + 7) % 256) as u8).collect();
        let mut handles = Vec::new();
        for _ in 0..100 {
            let authority = authority.clone();
            let payload = payload.clone();
            handles.push(std::thread::spawn(move || {
                authority
                    .insert_new(
                        &durable_envelope(
                            sid,
                            ws,
                            "concurrent",
                            Compressibility::Aggressive,
                            BackingCompleteness::Complete,
                        ),
                        Some(&payload),
                    )
                    .expect("concurrent store")
                    .id
            }));
        }
        let mut ids = Vec::new();
        for handle in handles {
            ids.push(handle.join().expect("store thread must not panic"));
        }
        assert_eq!(ids.len(), 100);
        assert_eq!(f.cas.blob_count(), 1, "identical bytes are ONE CAS blob");
        for id in ids {
            let stored = DurableEvidenceAuthority::get(&authority, id)
                .unwrap()
                .expect("row survives");
            assert_eq!(
                stored.backing.as_deref(),
                Some(&payload[..]),
                "every reference resolves to the exact bytes"
            );
        }
        let temps = std::fs::read_dir(f.cas.root().join("tmp")).unwrap().count();
        assert_eq!(temps, 0, "no temp file may survive the concurrent storm");
    }

    #[test]
    fn evidence_fault_seams_never_leave_a_row_without_its_blob() {
        use faktor_cas::CrashArm;

        let f = durable();
        let (sid, ws) = scoped_session(&f.store, "/w");
        let authority = durable_authority(&f, 4096);
        let owner = EvidenceAccessContext::new(sid.raw(), ws.raw(), Some(7));
        let env = |body: &str| {
            durable_envelope(
                sid,
                ws,
                body,
                Compressibility::Aggressive,
                BackingCompleteness::Complete,
            )
        };
        let store = authority.store().clone();

        // (a) Crash INSIDE the CAS put, before the rename: no blob at the
        // address and no row.
        authority.cas().crash_arm(CrashArm {
            point: "cas_tmp",
            ordinal: 0,
        });
        let crashed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            authority.insert_new(&env("crash-in-put"), Some(b"put bytes"))
        }));
        assert!(crashed.is_err(), "the armed seam must fire");
        assert_eq!(
            store.evidence_high_water().unwrap(),
            0,
            "no row after a put crash"
        );
        assert_eq!(
            f.cas.blob_count(),
            0,
            "no blob at a valid address after a put crash"
        );

        // (b) Crash AFTER the blob, BEFORE the row: an unreferenced blob is
        // tolerable; no row references it, and a retry dedupes to it.
        authority.crash_arm(EvidenceCrashArm {
            point: "evidence_blob_persisted",
            ordinal: 0,
        });
        let crashed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            authority.insert_new(&env("crash-after-blob"), Some(b"blob first bytes"))
        }));
        assert!(crashed.is_err(), "the blob-persisted seam must fire");
        assert_eq!(
            store.evidence_high_water().unwrap(),
            0,
            "no row may reference an absent blob"
        );
        assert_eq!(
            f.cas.blob_count(),
            1,
            "the unreferenced blob is the tolerated residue"
        );
        let retry = authority
            .insert_new(&env("crash-after-blob"), Some(b"blob first bytes"))
            .unwrap();
        assert_eq!(f.cas.blob_count(), 1, "the retry reuses the blob");
        assert_eq!(
            authority
                .get_scoped(retry.id, &owner)
                .unwrap()
                .backing
                .as_deref(),
            Some(&b"blob first bytes"[..])
        );

        // (c) Crash AFTER the row, BEFORE the response: retrieval resolves
        // (row + blob), so a row referencing an absent blob is impossible by
        // ordering.
        authority.crash_arm(EvidenceCrashArm {
            point: "evidence_row_inserted",
            ordinal: 0,
        });
        let crashed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            authority.insert_new(&env("crash-after-row"), Some(b"row second bytes"))
        }));
        assert!(crashed.is_err(), "the row-inserted seam must fire");
        let rows = authority.list_scoped(&owner, 100).unwrap();
        assert_eq!(rows.len(), 2, "both committed rows survive");
        let committed = rows[1];
        assert_eq!(
            authority
                .get_scoped(committed, &owner)
                .unwrap()
                .backing
                .as_deref(),
            Some(&b"row second bytes"[..])
        );
        // A CAS crash right AFTER its rename (blob in place, row never
        // inserted) leaves an unreferenced blob, never a torn one.
        authority.cas().crash_arm(CrashArm {
            point: "cas_renamed",
            ordinal: 0,
        });
        let crashed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            authority.insert_new(&env("crash-after-rename"), Some(b"renamed bytes"))
        }));
        assert!(crashed.is_err(), "the cas_renamed seam must fire");
        for id in authority.list_scoped(&owner, 100).unwrap() {
            authority
                .get_scoped(id, &owner)
                .expect("every committed row resolves its blob");
        }
    }

    #[test]
    fn corrupt_blob_post_persist_fails_loudly_on_retrieval() {
        let f = durable();
        let (sid, ws) = scoped_session(&f.store, "/w");
        let authority = durable_authority(&f, 4096);
        let owner = EvidenceAccessContext::new(sid.raw(), ws.raw(), Some(7));
        let stored = authority
            .insert_new(
                &durable_envelope(
                    sid,
                    ws,
                    "corrupt",
                    Compressibility::Aggressive,
                    BackingCompleteness::Complete,
                ),
                Some(b"verified once"),
            )
            .unwrap();
        let hash = FileHash::from(stored.backing_hash.expect("hash recorded"));
        let blob = f.cas.root().join(hash.cas_path());
        assert_eq!(f.cas.get_verified_now(hash).unwrap(), b"verified once");
        std::fs::write(&blob, b"not a zstd frame at all").unwrap();
        assert!(
            f.cas.get_verified_now(hash).is_err(),
            "the strict read must fail loudly"
        );
        assert!(matches!(
            authority.get_scoped(stored.id, &owner),
            Err(EvidenceError::Malformed(_))
        ));
        assert!(matches!(
            authority.read_backing_by_digest(&hash.to_hex(), &owner),
            Err(EvidenceError::Malformed(_))
        ));
    }

    #[test]
    fn arc_authority_shares_the_authority_identity_and_delegates_reads() {
        let f = durable();
        let (sid, ws) = scoped_session(&f.store, "/w");
        let authority = Arc::new(durable_authority(&f, 4096));
        let owner = EvidenceAccessContext::new(sid.raw(), ws.raw(), Some(7));
        let stored = authority
            .insert_new(
                &durable_envelope(
                    sid,
                    ws,
                    "aliased",
                    Compressibility::Aggressive,
                    BackingCompleteness::Complete,
                ),
                Some(b"aliased bytes"),
            )
            .unwrap();
        let handle: Box<dyn EvidenceStore + Send + Sync> = Box::new(authority.clone());
        assert_eq!(
            handle.authority_ptr(),
            Some(Arc::as_ptr(&authority) as usize),
            "the boxed handle wraps the SAME allocation the graph holds"
        );
        assert!(!handle.exists(EvidenceId(999)).unwrap());
        assert!(handle.exists(stored.id).unwrap());
        assert_eq!(
            handle
                .get_scoped(stored.id, &owner)
                .unwrap()
                .backing
                .as_deref(),
            Some(&b"aliased bytes"[..])
        );
    }
}
