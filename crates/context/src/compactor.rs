//! Compaction that cannot enter a death spiral (spec §9) and cannot let a
//! model rewrite the user's ask (audits 70/71).
//!
//! Hard invariant: a successful compaction must achieve the configured
//! minimum reduction. An LLM "summary" that would leave the context above
//! the target (or shave only ~1%) is REJECTED and deterministic pruning
//! takes over. Incremental by design: the durable task projection is
//! preserved whole, only old recent turns are archived.
//!
//! # Structured compaction pipeline (audit 70)
//!
//! BEFORE any LLM summarization runs, deterministic stages run in order:
//!
//! 1. **Evidence extraction.** A huge historical tool output (a `tool`
//!    turn above [`TOOL_OUTPUT_EVIDENCE_THRESHOLD_BYTES`]) is
//!    normalized/compressed/stored through `faktor-evidence`; the wire
//!    carries an [`EvidenceRef`] whose compact body is bounded by the
//!    evidence crate's 16 KiB ceiling, and the original stays retrievable
//!    from the [`EvidenceArchive`]. Identical outputs content-address to
//!    ONE evidence id.
//! 2. **Exact dedupe.** Byte-identical (role, text) rows collapse to their
//!    first occurrence before summarization or pruning.
//! 3. **Durable truth.** Goal, criteria, decisions, checks, plan and child
//!    progress come from a [`TaskContextProjection`] built from durable
//!    rows (typed Task + typed ledger + VerificationRecord rows), never
//!    from the transcript. Goal/criteria ride the post-compaction wire
//!    VERBATIM ([`TaskContextProjection::identity_render`]) and are
//!    excluded from the summarizer input entirely.
//! 4. **Gated summarization.** The LLM summarizer is invoked only when the
//!    post-extraction conversational residue exceeds
//!    [`CompactionRequest::summarizer_residue_budget_tokens`]; below the
//!    budget the deterministic pipeline compacts with no model call.

use std::sync::{Arc, Mutex, MutexGuard};

use crate::artifact::ArtifactRef;
use crate::ledger::TaskContextProjection;
use faktor_evidence::compress::compress;
use faktor_evidence::normalize::ProcessLogSummary;
use faktor_evidence::store::{
    EvidenceAccessContext, EvidenceStore, MemoryEvidenceStore, StoredEvidence,
};
use faktor_evidence::types::{
    BackingCompleteness, CompactRepresentation, CompressionRecord, EvidenceEnvelope, EvidenceError,
    EvidenceId, EvidenceKind, ProvenanceSet, ProvenanceSource, RetrievalPolicy,
};

/// Bound on ONE chunk of the rendered text of evicted turns carried out of
/// the compactor for the content store. There is NO total archive cap: every
/// evicted turn is preserved across as many chunks as needed, and the runtime
/// stores each chunk in the CAS behind a JSON manifest (the recent history in
/// RAM is itself bounded, so the chunked archive never exceeds it).
const ARCHIVE_CHUNK_BYTES: usize = 512 * 1024;
const DIGEST_MAX_CHARS: usize = 600;
const DIGEST_MAX_LINES: usize = 8;
const DIGEST_TEMPLATE: &str = "[Earlier context archived: …<artifact://hash>]";
const ARTIFACT_PLACEHOLDER: &str = "<artifact://hash>";

/// A `tool`-role turn larger than this many bytes is "huge historical tool
/// output": it leaves the wire as a bounded evidence ref instead of riding
/// inline. User/assistant turns are never extracted this way.
pub const TOOL_OUTPUT_EVIDENCE_THRESHOLD_BYTES: usize = 16 * 1024;
/// The evidence crate's hard compact-body ceiling; re-stated here so the
/// wire bound is visible at the compaction contract (compress() enforces
/// it at 16 KiB).
pub const EVIDENCE_COMPACT_BODY_MAX_BYTES: usize = 16 * 1024;
/// Backing bytes the default archive retains per evidence id (the original
/// is retrievable below this cap; above it the envelope + hash survive and
/// retrieval fails loudly, never silently).
pub const EVIDENCE_BACKING_CAP_BYTES: usize = 16 * 1024 * 1024;
/// Ceiling for any single evidence retrieval response.
pub const EVIDENCE_RETRIEVAL_MAX_BYTES: usize = EVIDENCE_BACKING_CAP_BYTES;
/// Default conversational-residue budget: the LLM summarizer runs only when
/// the post-extraction residue exceeds this many estimated tokens.
pub const DEFAULT_SUMMARIZER_RESIDUE_BUDGET_TOKENS: usize = 128;

/// Insertion-ordered rendering of turns (`"{role}: {text}\n"`), the same
/// shape the archive chunks use.
fn render_turns<'a>(turns: impl Iterator<Item = &'a RecentTurn>) -> String {
    let mut out = String::new();
    for t in turns {
        out.push_str(&format!("{}: {}\n", t.role, t.text));
    }
    out
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max.min(s.len());
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// (b) Exact-fact dedupe: byte-identical (role, text) rows collapse to their
/// FIRST occurrence, preserving order. Applied after evidence extraction, so
/// identical huge tool outputs also collapse (they render one ref).
fn dedupe_exact_rows(turns: &[RecentTurn]) -> Vec<RecentTurn> {
    let mut seen: std::collections::HashSet<(&str, &str)> = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(turns.len());
    for turn in turns {
        if seen.insert((turn.role.as_str(), turn.text.as_str())) {
            out.push(turn.clone());
        }
    }
    out
}

/// Only a TOOL-role turn is evidence-eligible. Tool output has no
/// instruction authority and no user-policy bytes, so it may be moved
/// behind an evidence ref; user and assistant text stay exactly where they
/// were (the user's ask is policy; assistant residue keeps its own archival
/// path through the CAS manifest).
fn is_evidence_eligible(turn: &RecentTurn) -> bool {
    turn.text.len() > TOOL_OUTPUT_EVIDENCE_THRESHOLD_BYTES && turn.role.eq_ignore_ascii_case("tool")
}

/// (c) The synthesized wire turns that carry the durable truth OUTSIDE the
/// summarization input: the verbatim identity block (goal + criteria) and
/// the durable facts render. Empty blocks produce no turn, so a compaction
/// with no durable content stays turn-for-turn identical to the old shape.
fn synthesized_turns(identity: &str, durable_facts: &str) -> Vec<RecentTurn> {
    let mut out = Vec::new();
    if !identity.is_empty() {
        out.push(RecentTurn {
            role: "system".into(),
            text: format!("TASK IDENTITY (verbatim):\n{identity}"),
        });
    }
    if !durable_facts.is_empty() {
        out.push(RecentTurn {
            role: "system".into(),
            text: format!("DURABLE TASK PROJECTION:\n{durable_facts}"),
        });
    }
    out
}

/// One reference to archived historical tool output: the compact body is
/// wire-visible and bounded by [`EVIDENCE_COMPACT_BODY_MAX_BYTES`]; the
/// original bytes are retrievable through the [`EvidenceArchive`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceRef {
    pub id: EvidenceId,
    pub kind: EvidenceKind,
    pub original_bytes: usize,
    pub compact_bytes: usize,
    pub backing_hash: [u8; 32],
    /// Bounded structural summary produced by the evidence compressor.
    pub compact_body: String,
    /// Exit code normalized from the raw output, when recognized.
    pub normalized_exit: Option<i32>,
    /// Lines the normalizer kept as important, when normalization ran.
    pub normalized_important_lines: usize,
}

impl EvidenceRef {
    /// The wire text that replaces the raw output.
    pub fn render(&self) -> String {
        format!(
            "[tool output archived: evidence://{} kind={:?} original_bytes={} compact_bytes={}{}]\n{}",
            self.id,
            self.kind,
            self.original_bytes,
            self.compact_bytes,
            self.normalized_exit
                .map(|e| format!(" exit={e}"))
                .unwrap_or_default(),
            self.compact_body,
        )
    }
}

/// Deterministic archive of huge historical tool output as evidence
/// envelopes + backing, built on `faktor-evidence` (normalize + compress +
/// store). Content-addressed by blake3: re-archiving identical bytes returns
/// the SAME ref (duplicate insert can never overwrite stored evidence).
///
/// Bounded: backing is retained per envelope up to `backing_cap`; beyond it
/// the envelope + hash survive and retrieval fails loudly. Scope reads go
/// through the evidence store's session/workspace check.
#[derive(Debug)]
pub struct EvidenceArchive {
    store: MemoryEvidenceStore,
    session_id: faktor_core::SessionId,
    workspace_id: faktor_core::WorkspaceId,
    task_id: Option<u64>,
}

impl Default for EvidenceArchive {
    fn default() -> Self {
        // Standalone session defaults (ids may never be 0).
        Self::new(1, 1, None)
    }
}

impl EvidenceArchive {
    pub fn new(session_id: u64, workspace_id: u64, task_id: Option<u64>) -> Self {
        Self::with_backing_cap(
            session_id,
            workspace_id,
            task_id,
            EVIDENCE_BACKING_CAP_BYTES,
        )
    }

    pub fn with_backing_cap(
        session_id: u64,
        workspace_id: u64,
        task_id: Option<u64>,
        backing_cap: usize,
    ) -> Self {
        // Identity types refuse 0; a hostile 0 scope is clamped to the
        // standalone default rather than panicking compaction.
        Self {
            store: MemoryEvidenceStore::new(backing_cap),
            session_id: faktor_core::SessionId::new(session_id.max(1)),
            workspace_id: faktor_core::WorkspaceId::new(workspace_id.max(1)),
            task_id,
        }
    }

    pub fn len(&self) -> usize {
        self.store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    fn access(&self) -> EvidenceAccessContext {
        EvidenceAccessContext::new(self.session_id.raw(), self.workspace_id.raw(), self.task_id)
    }

    /// Normalize + compress + store `raw` under `kind`. Idempotent for
    /// identical bytes (content-addressed id). The compact body is bounded
    /// by the evidence crate; the backing is retained only under the
    /// store's cap policy.
    pub fn archive(&mut self, kind: EvidenceKind, raw: &str) -> Result<EvidenceRef, EvidenceError> {
        let hash = *blake3::hash(raw.as_bytes()).as_bytes();
        // normalize step (bounded typed view; a >MAX_INPUT_LINES hostile
        // line count still compresses below via the streaming transform).
        let normalized = ProcessLogSummary::try_from_text(raw).ok();
        let normalized_exit = normalized.as_ref().and_then(|n| n.exit);
        let normalized_important_lines = normalized
            .as_ref()
            .map(|n| n.important_lines.len())
            .unwrap_or(0);

        let mut id_raw = u64::from_be_bytes(hash[..8].try_into().expect("8 bytes"));
        loop {
            let candidate = EvidenceId(id_raw);
            match self.store.get(candidate) {
                Some(stored) if stored.envelope.backing_hash == Some(hash) => {
                    // Exact duplicate: the SAME evidence ref, never a second
                    // copy (audit 70 dedupe at the evidence layer).
                    return Ok(EvidenceRef {
                        id: candidate,
                        kind: stored.envelope.kind,
                        original_bytes: stored.envelope.compression.original_bytes as usize,
                        compact_bytes: stored.envelope.compression.compact_bytes as usize,
                        backing_hash: hash,
                        compact_body: stored.envelope.compact.body.clone(),
                        normalized_exit,
                        normalized_important_lines,
                    });
                }
                Some(_) => id_raw = id_raw.wrapping_add(1),
                None => break,
            }
        }
        let id = EvidenceId(id_raw);
        // compress step: kind-hardcoded policy; ProcessLog is Aggressive, so
        // the body is structurally compacted and ties back to the digest.
        // A kind whose policy copies verbatim (LosslessOnly) could exceed
        // the evidence bound — refuse it rather than promise a 16 KiB body
        // and deliver more.
        let (compact, record): (CompactRepresentation, CompressionRecord) =
            compress(&kind, BackingCompleteness::Complete, raw)?;
        if compact.body.len() > EVIDENCE_COMPACT_BODY_MAX_BYTES {
            return Err(EvidenceError::Oversized {
                max: EVIDENCE_COMPACT_BODY_MAX_BYTES,
                actual: compact.body.len(),
            });
        }
        let compressibility = kind.default_compressibility();
        let envelope = EvidenceEnvelope::new(
            id,
            kind,
            self.session_id,
            self.workspace_id,
            self.task_id,
            None,
            ProvenanceSet::new([ProvenanceSource::Tool]),
            compressibility,
            compact.clone(),
            Some(hash),
            BackingCompleteness::Complete,
            record,
            RetrievalPolicy::new(true, true, EVIDENCE_RETRIEVAL_MAX_BYTES),
        )?;
        // store step: backing retained under the store's cap policy.
        self.store.insert(envelope, Some(raw.as_bytes().to_vec()))?;
        Ok(EvidenceRef {
            id,
            kind,
            original_bytes: raw.len(),
            compact_bytes: compact.body.len(),
            backing_hash: hash,
            compact_body: compact.body,
            normalized_exit,
            normalized_important_lines,
        })
    }

    /// Scope-checked retrieval of one archived envelope (backing included
    /// when the store retained it).
    pub fn retrieve(&self, id: EvidenceId) -> Result<StoredEvidence, EvidenceError> {
        self.store.get_scoped(id, &self.access())
    }
}

/// Chunk the rendered text of evicted turns (`"{role}: {text}\n"` per turn,
/// OLDEST-first iteration) into `Vec<String>` chunks each <= `chunk_max`
/// bytes. Chunks preserve order and NEVER split mid-turn: a single turn
/// larger than the bound occupies its own whole chunk (the recent history in
/// RAM is bounded, so an oversize chunk is bounded too). Nothing is dropped.
fn fill_chunks<'a>(evicted: impl Iterator<Item = &'a RecentTurn>, chunk_max: usize) -> Vec<String> {
    let mut chunks: Vec<String> = Vec::new();
    for turn in evicted {
        let line = format!("{}: {}", turn.role, turn.text);
        if let Some(last) = chunks.last_mut() {
            if !last.is_empty() && last.len() + line.len() < chunk_max {
                last.push_str(&line);
                last.push('\n');
                continue;
            }
        }
        let mut chunk = String::new();
        chunk.push_str(&line);
        chunk.push('\n');
        chunks.push(chunk);
    }
    chunks
}

/// The eviction digest that rides the wire inside kept_recent[0]: the NEWEST
/// evicted material (the tail of the archive — chunks are ordered oldest
/// first), bounded to a few lines and `DIGEST_MAX_CHARS`. A single line
/// larger than the whole budget is truncated to fit, never dropped entirely,
/// so the marker always carries a glimpse of what was archived. Empty only
/// when nothing was archived.
fn archive_digest(chunks: &[String]) -> String {
    let mut newest_first: Vec<&str> = Vec::new();
    for chunk in chunks.iter().rev() {
        for line in chunk.lines().rev() {
            newest_first.push(line);
            if newest_first.len() >= DIGEST_MAX_LINES {
                break;
            }
        }
        if newest_first.len() >= DIGEST_MAX_LINES {
            break;
        }
    }
    // Render chronologically (oldest of the sampled lines first).
    let mut digest_text = String::new();
    for line in newest_first.into_iter().rev() {
        let remaining = DIGEST_MAX_CHARS.saturating_sub(digest_text.len());
        if remaining == 0 {
            break;
        }
        if line.len() > remaining {
            digest_text.push_str(truncate(line, remaining));
            digest_text.push('\n');
            break;
        }
        digest_text.push_str(line);
        digest_text.push('\n');
    }
    digest_text
}
use crate::assembler::RecentTurn;
use crate::estimator::Estimator;
use crate::ledger::TaskLedger;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompactionRequest {
    pub before_tokens: usize,
    pub target_tokens: usize,
    /// Minimum fraction the context must shrink by (0.25 = 25%).
    pub min_reduction_ratio: f64,
    /// (f) Conversational residue (AFTER evidence extraction and exact
    /// dedupe) above which the LLM summarizer may run. At or below the
    /// budget the deterministic pipeline compacts without any model call.
    pub summarizer_residue_budget_tokens: usize,
}

impl Default for CompactionRequest {
    fn default() -> Self {
        Self {
            before_tokens: 0,
            target_tokens: 0,
            min_reduction_ratio: 0.25,
            summarizer_residue_budget_tokens: DEFAULT_SUMMARIZER_RESIDUE_BUDGET_TOKENS,
        }
    }
}

impl CompactionRequest {
    pub fn new(before_tokens: usize, target_tokens: usize) -> Self {
        Self {
            before_tokens,
            target_tokens,
            min_reduction_ratio: 0.25,
            summarizer_residue_budget_tokens: DEFAULT_SUMMARIZER_RESIDUE_BUDGET_TOKENS,
        }
    }

    /// The maximum acceptable after_tokens: min(target, before*(1-ratio)).
    pub fn hard_cap(&self) -> usize {
        let by_ratio = if self.min_reduction_ratio <= 0.0 {
            self.before_tokens
        } else {
            let factor = (1.0 - self.min_reduction_ratio).clamp(0.0, 1.0);
            (self.before_tokens as f64 * factor) as usize
        };
        self.target_tokens.min(by_ratio)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionStrategy {
    LlmSummary,
    DeterministicPruning,
    Rejected,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompactionPlan {
    pub accepted: bool,
    pub before_tokens: usize,
    pub after_tokens: usize,
    pub target_tokens: usize,
    pub strategy: CompactionStrategy,
    pub ledger: TaskLedger,
    pub kept_recent: Vec<RecentTurn>,
    pub archived: Vec<ArtifactRef>,
    /// The rendered text of every turn evicted from the wire, chunked
    /// (`"{role}: {text}\n"` per turn, oldest first, each chunk <=
    /// `ARCHIVE_CHUNK_BYTES`, never split mid-turn, NO total cap — nothing
    /// evicted is ever omitted). This is the durable archive material for
    /// the content store: the runtime writes each chunk to the CAS, then a
    /// JSON manifest, and replaces the digest placeholder with the manifest
    /// artifact ref.
    pub archive_chunks: Vec<String>,
    /// (a) Huge historical tool output that was normalized, compressed and
    /// stored as evidence before any summarization; each ref's compact body
    /// is bounded and the original is retrievable from the archive.
    pub evidence_refs: Vec<EvidenceRef>,
}

/// Produces the LLM-written summary (injected from the agent; None in
/// deterministic-only operation). Async: real summarizers stream a
/// provider request; the deterministic ledger summarizer resolves
/// immediately.
///
/// The input is the POST-EXTRACTION residue (huge tool output already
/// replaced by evidence refs, exact duplicates collapsed) plus the
/// durable non-goal facts render — the goal and acceptance criteria are
/// NEVER passed here (audit 70: the user's ask cannot go through a lossy
/// model).
pub trait Summarizer: Send + Sync {
    fn summarize<'a>(
        &'a self,
        residue: &'a [RecentTurn],
        durable_facts: &'a str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = String> + Send + 'a>>;
}

/// Compactor with a per-instance evidence archive. Production wires one
/// compactor per compaction call; tests/soaks may reuse one instance and
/// read the archive back through [`Compactor::evidence`].
pub struct Compactor {
    summarizer: Option<Arc<dyn Summarizer>>,
    evidence: Mutex<EvidenceArchive>,
}

impl Compactor {
    pub fn new(summarizer: Option<Arc<dyn Summarizer>>) -> Self {
        Self {
            summarizer,
            evidence: Mutex::new(EvidenceArchive::default()),
        }
    }

    pub fn deterministic_only() -> Self {
        Self::new(None)
    }

    /// Scope the archive to a session/workspace/task (retrieval then runs
    /// through the evidence store's scope check).
    pub fn with_evidence_scope(
        mut self,
        session_id: u64,
        workspace_id: u64,
        task_id: Option<u64>,
    ) -> Self {
        self.evidence = Mutex::new(EvidenceArchive::new(session_id, workspace_id, task_id));
        self
    }

    /// Recover from a poisoned lock instead of panicking: the archive
    /// critical sections cannot panic, and compaction must never take the
    /// runtime down over an unrelated poison.
    fn lock_evidence(&self) -> MutexGuard<'_, EvidenceArchive> {
        self.evidence
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The per-instance evidence archive (bounded backing, scope checks).
    pub fn evidence(&self) -> MutexGuard<'_, EvidenceArchive> {
        self.lock_evidence()
    }

    /// (a)+(b)+(c) Deterministic preparation, run BEFORE any summarization:
    /// huge non-user tool output → evidence refs (normalize/compress/store),
    /// then byte-identical rows collapse. Returns the residue plus the refs
    /// that replaced archived output.
    fn prepare_history(&self, history: &[RecentTurn]) -> (Vec<RecentTurn>, Vec<EvidenceRef>) {
        let mut prepared: Vec<RecentTurn> = Vec::with_capacity(history.len());
        let mut refs: Vec<EvidenceRef> = Vec::new();
        {
            let mut archive = self.lock_evidence();
            for turn in history {
                if is_evidence_eligible(turn) {
                    if let Ok(ev) = archive.archive(EvidenceKind::ProcessLog, &turn.text) {
                        // Identical outputs content-address to ONE ref.
                        if !refs.iter().any(|r| r.id == ev.id) {
                            refs.push(ev.clone());
                        }
                        prepared.push(RecentTurn {
                            role: turn.role.clone(),
                            text: ev.render(),
                        });
                        continue;
                    }
                    // Archive failure (refused/oversized invariant): the raw
                    // stays inline — evidence extraction never drops bytes.
                }
                prepared.push(turn.clone());
            }
        }
        (dedupe_exact_rows(&prepared), refs)
    }

    /// Compatibility entry point (existing callers/tests): the read-only
    /// projection is built from the legacy durable ledger rows and the
    /// instance's own evidence archive is used.
    pub async fn compact(
        &self,
        history: &[RecentTurn],
        ledger: &TaskLedger,
        req: &CompactionRequest,
    ) -> CompactionPlan {
        let projection = TaskContextProjection::from_task_ledger(ledger);
        self.compact_projected(history, &projection, ledger, req)
            .await
    }

    /// Compaction pipeline (audits 70/71):
    /// 1. Evidence extraction + exact dedupe (deterministic, model-free).
    /// 2. Goal/criteria identity is rendered VERBATIM outside the
    ///    summarization input.
    /// 3. If a summarizer exists AND the residue exceeds the configured
    ///    budget, try the LLM summary; accept it ONLY if it satisfies the
    ///    hard invariant (after <= hard_cap) with identity + durable facts
    ///    counted.
    /// 4. Otherwise (or when rejected), deterministic pruning.
    /// 5. If even deterministic pruning cannot reach the cap
    ///    (pathological), the plan is marked rejected with strategy
    ///    `Rejected` — callers must surface it (CompactRejected) instead of
    ///    pretending success.
    pub async fn compact_projected(
        &self,
        history: &[RecentTurn],
        projection: &TaskContextProjection,
        working_ledger: &TaskLedger,
        req: &CompactionRequest,
    ) -> CompactionPlan {
        // Stages (a)+(b): move huge historical tool output to evidence refs
        // and collapse exact duplicate rows BEFORE any model sees anything.
        let (residue, evidence_refs) = self.prepare_history(history);
        // (c): the user's ask, verbatim, and the durable facts, both kept
        // OUT of the summarizer input.
        let identity = projection.identity_render();
        let durable_facts = projection.summarizable_render();
        let identity_tokens = Estimator.estimate_tokens(&identity);
        let facts_tokens = Estimator.estimate_tokens(&durable_facts);
        let residue_tokens = Estimator.estimate_tokens(&render_turns(residue.iter()));

        // (f): the LLM is invoked only above the residue budget.
        if let Some(summarizer) = &self.summarizer {
            if residue_tokens > req.summarizer_residue_budget_tokens {
                let summary = summarizer.summarize(&residue, &durable_facts).await;
                let after = identity_tokens
                    .saturating_add(facts_tokens)
                    .saturating_add(Estimator.estimate_tokens(&summary));
                if after <= req.hard_cap() {
                    // Accepted summary: it REPLACES the conversational
                    // residue on the wire. Identity (verbatim) and durable
                    // facts ride alongside; every evicted residue turn is
                    // archived for the CAS, chunked, never truncated.
                    let archived = residue
                        .iter()
                        .map(|t| ArtifactRef {
                            inline: None,
                            artifact: None,
                            summary: format!("archived turn ({} chars)", t.text.len()),
                            size: t.text.len(),
                        })
                        .collect();
                    let mut kept_recent = synthesized_turns(&identity, &durable_facts);
                    kept_recent.push(RecentTurn {
                        role: "assistant".into(),
                        text: summary,
                    });
                    return CompactionPlan {
                        accepted: true,
                        before_tokens: req.before_tokens,
                        after_tokens: after,
                        target_tokens: req.target_tokens,
                        strategy: CompactionStrategy::LlmSummary,
                        ledger: working_ledger.clone(),
                        kept_recent,
                        archived,
                        archive_chunks: fill_chunks(residue.iter(), ARCHIVE_CHUNK_BYTES),
                        evidence_refs,
                    };
                }
                // REJECT the liar summary; fall through to deterministic.
                let mut plan = self.deterministic_core(
                    &residue,
                    &identity,
                    &durable_facts,
                    working_ledger,
                    req,
                    evidence_refs,
                );
                plan.strategy = CompactionStrategy::Rejected;
                return plan;
            }
        }
        self.deterministic_core(
            &residue,
            &identity,
            &durable_facts,
            working_ledger,
            req,
            evidence_refs,
        )
    }

    /// Deterministic pruning: keep the durable projection in full, keep the
    /// newest residue turns that fit under the cap, archive the rest as
    /// references. `identity` and `durable_facts` are reserved up front.
    fn deterministic_core(
        &self,
        residue: &[RecentTurn],
        identity: &str,
        durable_facts: &str,
        working_ledger: &TaskLedger,
        req: &CompactionRequest,
        evidence_refs: Vec<EvidenceRef>,
    ) -> CompactionPlan {
        let identity_tokens = Estimator.estimate_tokens(identity);
        let facts_tokens = Estimator.estimate_tokens(durable_facts);
        let cap = req.hard_cap();
        // The eviction digest rides the wire in kept_recent[0]; reserve its
        // budget up front so the after_tokens figure stays honest.
        let digest_tokens = Estimator.estimate_tokens(DIGEST_TEMPLATE);
        let mut kept_recent = Vec::new();
        let mut archived = Vec::new();
        // Evicted turns as collected (newest first — the scan runs newest to
        // oldest); reversed into chronological order for the chunk fill.
        let mut evicted: Vec<&RecentTurn> = Vec::new();
        let reserved = identity_tokens
            .saturating_add(facts_tokens)
            .saturating_add(32)
            .saturating_add(digest_tokens);
        let mut used = reserved;
        if cap > reserved {
            // Newest first; each turn's cost = role + text.
            for turn in residue.iter().rev() {
                let t = Estimator.estimate_tokens(&format!("{}: {}", turn.role, turn.text));
                if used + t > cap {
                    archived.push(ArtifactRef {
                        inline: None,
                        artifact: None,
                        summary: format!("archived turn ({} chars)", turn.text.len()),
                        size: turn.text.len(),
                    });
                    evicted.push(turn);
                    continue;
                }
                used += t;
                kept_recent.push(turn.clone());
            }
            kept_recent.reverse();
        }
        // Every evicted turn is archived, chunked, oldest first — nothing
        // beyond the kept window is ever omitted (the old 1 MiB cap silently
        // dropped cold history).
        let archive_chunks = fill_chunks(evicted.iter().rev().copied(), ARCHIVE_CHUNK_BYTES);
        let after = used;
        let accepted = after <= cap;
        // Digest of what was archived: the newest evicted lines (tail of the
        // last chunk), bounded — the marker tells the model history was
        // dropped and where the durable material lives. The digest stays
        // FIRST: the runtime rewrites its placeholder with the CAS manifest.
        let digest_text = archive_digest(&archive_chunks);
        let mut wire_recent = Vec::new();
        if !digest_text.is_empty() && accepted {
            wire_recent.push(RecentTurn {
                role: "assistant".into(),
                text: format!(
                    "[Earlier context archived: {}…{}]",
                    truncate(&digest_text, DIGEST_MAX_CHARS),
                    ARTIFACT_PLACEHOLDER
                ),
            });
        }
        wire_recent.extend(synthesized_turns(identity, durable_facts));
        wire_recent.extend(kept_recent);
        CompactionPlan {
            accepted,
            before_tokens: req.before_tokens,
            after_tokens: after,
            target_tokens: req.target_tokens,
            strategy: if accepted {
                CompactionStrategy::DeterministicPruning
            } else {
                CompactionStrategy::Rejected
            },
            ledger: working_ledger.clone(),
            kept_recent: wire_recent,
            archived,
            archive_chunks,
            evidence_refs,
        }
    }

    /// Compatibility wrapper: builds the read-only projection from the
    /// legacy durable ledger rows and prunes deterministically.
    pub fn deterministic_prune(
        &self,
        history: &[RecentTurn],
        ledger: &TaskLedger,
        req: &CompactionRequest,
    ) -> CompactionPlan {
        let projection = TaskContextProjection::from_task_ledger(ledger);
        let (residue, evidence_refs) = self.prepare_history(history);
        let identity = projection.identity_render();
        let durable_facts = projection.summarizable_render();
        self.deterministic_core(
            &residue,
            &identity,
            &durable_facts,
            ledger,
            req,
            evidence_refs,
        )
    }

    /// Would a new compaction still be required immediately after `plan`?
    /// The death-spiral guard: if true after an *accepted* plan, the engine
    /// must not loop forever — this must converge to false within bounded
    /// steps because deterministic pruning is monotonically non-increasing.
    pub fn would_compact_again(&self, plan: &CompactionPlan, req: &CompactionRequest) -> bool {
        if !plan.accepted {
            return true;
        }
        let new_req = CompactionRequest {
            before_tokens: plan.after_tokens,
            target_tokens: req.target_tokens,
            min_reduction_ratio: req.min_reduction_ratio,
            summarizer_residue_budget_tokens: req.summarizer_residue_budget_tokens,
        };
        // If before <= target there is nothing left to do.
        plan.after_tokens
            > new_req
                .hard_cap()
                .max(new_req.target_tokens.min(plan.after_tokens))
            && plan.after_tokens > req.target_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::{DurableTaskRows, ProjectedCheck, ProjectedDecision};
    use std::sync::Arc;

    fn history(n: usize) -> Vec<RecentTurn> {
        (0..n)
            .map(|i| RecentTurn {
                role: "assistant".into(),
                text: format!("turn {i}: {}", "z".repeat(400)),
            })
            .collect()
    }

    fn ledger() -> TaskLedger {
        TaskLedger {
            goal: "g".into(),
            open_steps: vec!["s".into()],
            ..Default::default()
        }
    }

    /// The adversary: a "summarizer" that returns the whole history verbatim
    /// (reduces context by ~0%).
    struct LiarSummarizer;
    impl Summarizer for LiarSummarizer {
        fn summarize<'a>(
            &'a self,
            residue: &'a [RecentTurn],
            _durable_facts: &'a str,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = String> + Send + 'a>> {
            Box::pin(async move { render_turns(residue.iter()) })
        }
    }

    #[tokio::test]
    async fn one_percent_summary_rejected_and_deterministic_fallback() {
        let history = history(200);
        let e = Estimator;
        let before = e.estimate_tokens(
            &history
                .iter()
                .map(|t| t.text.clone())
                .collect::<Vec<_>>()
                .join("\n"),
        );
        let req = CompactionRequest::new(before, before / 2);
        let compactor = Compactor::new(Some(Arc::new(LiarSummarizer)));
        let plan = compactor.compact(&history, &ledger(), &req).await;
        assert!(
            !matches!(plan.strategy, CompactionStrategy::LlmSummary),
            "liar summary must never be accepted"
        );
        assert!(plan.accepted, "deterministic fallback must succeed");
        assert_eq!(
            plan.strategy,
            CompactionStrategy::Rejected,
            "the summary attempt was rejected, then pruned"
        );
        assert!(plan.after_tokens <= req.hard_cap(), "hard invariant");
        // ~1% reduction never accepted: verify explicitly.
        let one_pct = CompactionRequest {
            before_tokens: 180_000,
            target_tokens: 178_200, // 1% reduction
            min_reduction_ratio: 0.25,
            ..Default::default()
        };
        let cap = one_pct.hard_cap();
        assert!(cap <= 135_000, "cap enforces the 25% floor, not the 1% ask");
    }

    #[tokio::test]
    async fn good_summary_accepted() {
        struct GoodSummarizer;
        impl Summarizer for GoodSummarizer {
            fn summarize<'a>(
                &'a self,
                _residue: &'a [RecentTurn],
                durable_facts: &'a str,
            ) -> std::pin::Pin<Box<dyn std::future::Future<Output = String> + Send + 'a>>
            {
                Box::pin(async move { format!("SUMMARY: {durable_facts}") })
            }
        }
        let history = history(200);
        let before = 100_000;
        let req = CompactionRequest::new(before, 30_000);
        let compactor = Compactor::new(Some(Arc::new(GoodSummarizer)));
        let plan = compactor.compact(&history, &ledger(), &req).await;
        assert!(plan.accepted);
        assert_eq!(plan.strategy, CompactionStrategy::LlmSummary);
        assert!(plan.after_tokens <= req.hard_cap());
    }

    #[tokio::test]
    async fn death_spiral_converges_with_liar_summarizer() {
        // The classic failure: compaction keeps "succeeding" by tiny margins
        // and never reaches the target. Our invariant must force the
        // deterministic path, and repeated compactions must converge.
        let mut current = history(400);
        let compactor = Compactor::new(Some(Arc::new(LiarSummarizer)));
        let target = 40_000usize;
        let mut steps = 0;
        let e = Estimator;
        let mut before_tokens = e.estimate_tokens(
            &current
                .iter()
                .map(|t| t.text.clone())
                .collect::<Vec<_>>()
                .join("\n"),
        );
        let mut converged = false;
        while steps < 20 {
            let req = CompactionRequest::new(before_tokens, target);
            let plan = compactor.compact(&current, &ledger(), &req).await;
            assert!(plan.accepted, "deterministic path must accept");
            assert!(
                plan.after_tokens <= req.hard_cap(),
                "step {steps}: after {} > cap {}",
                plan.after_tokens,
                req.hard_cap()
            );
            if compactor.would_compact_again(&plan, &req) {
                // Simulate the next context built from the plan.
                current = plan.kept_recent.clone();
                before_tokens = plan.after_tokens;
                steps += 1;
                continue;
            }
            converged = true;
            break;
        }
        assert!(converged, "did not converge within 20 steps");
    }

    #[tokio::test]
    async fn none_summarizer_incremental_compaction_preserves_ledger() {
        let history = history(300);
        let before = 200_000;
        let req = CompactionRequest::new(before, 60_000);
        let compactor = Compactor::deterministic_only();
        let plan = compactor.compact(&history, &ledger(), &req).await;
        assert!(plan.accepted);
        assert_eq!(plan.strategy, CompactionStrategy::DeterministicPruning);
        assert_eq!(plan.ledger, ledger(), "ledger preserved in full");
        assert!(!plan.kept_recent.is_empty(), "newest turns kept");
        assert!(plan.after_tokens <= req.hard_cap());
    }

    #[tokio::test]
    async fn archived_artifacts_track_evicted_turns() {
        let history = history(100);
        let req = CompactionRequest::new(200_000, 10_000);
        let compactor = Compactor::deterministic_only();
        let plan = compactor.compact(&history, &ledger(), &req).await;
        assert!(!plan.archived.is_empty(), "most turns archived");
        // Every evicted turn is accounted for: kept history turns (minus
        // the synthesized identity/durable-facts turns and the eviction
        // digest) + archived = total.
        let synthesized = plan
            .kept_recent
            .iter()
            .filter(|t| t.role == "system")
            .count();
        let digest = usize::from(
            plan.kept_recent
                .first()
                .is_some_and(|t| t.text.starts_with("[Earlier context archived:")),
        );
        let kept_excluding_synthesized = plan.kept_recent.len() - synthesized - digest;
        assert_eq!(
            kept_excluding_synthesized + plan.archived.len(),
            history.len(),
            "every evicted turn must be accounted for"
        );
        // Newest turns survive, oldest archived.
        assert_eq!(
            plan.kept_recent.last().map(|t| &t.text),
            history.last().map(|t| &t.text)
        );
        // The durable archive material is non-empty, chunked, and every
        // chunk respects the bound.
        assert!(!plan.archive_chunks.is_empty());
        assert!(plan
            .archive_chunks
            .iter()
            .all(|c| !c.is_empty() && c.len() <= ARCHIVE_CHUNK_BYTES));
        // The digest rides the wire so the model knows history was dropped.
        assert!(plan
            .kept_recent
            .first()
            .is_some_and(|t| t.text.contains("<artifact://hash>")));
    }

    #[tokio::test]
    async fn accepted_llm_summary_actually_replaces_history() {
        struct GoodSummarizer;
        impl Summarizer for GoodSummarizer {
            fn summarize<'a>(
                &'a self,
                _residue: &'a [RecentTurn],
                durable_facts: &'a str,
            ) -> std::pin::Pin<Box<dyn std::future::Future<Output = String> + Send + 'a>>
            {
                Box::pin(async move { format!("COMPACT SUMMARY: {durable_facts}") })
            }
        }
        let history = history(50);
        let req = CompactionRequest::new(100_000, 5_000);
        let compactor = Compactor::new(Some(Arc::new(GoodSummarizer)));
        let plan = compactor.compact(&history, &ledger(), &req).await;
        assert!(plan.accepted);
        assert_eq!(plan.strategy, CompactionStrategy::LlmSummary);
        // The wire content after an accepted summary is the summary plus
        // the synthesized durable turns, NOT the full history (the old code
        // kept history verbatim and the shrink was bookkeeping-only).
        assert_eq!(
            plan.kept_recent
                .iter()
                .filter(|t| t.role != "system")
                .count(),
            1,
            "exactly one summary turn replaces the whole history"
        );
        assert!(plan
            .kept_recent
            .last()
            .is_some_and(|t| t.text.starts_with("COMPACT SUMMARY:")));
        // All evicted turns are accounted for as archives + durable text.
        assert_eq!(plan.archived.len(), history.len());
        let archive_len: usize = plan.archive_chunks.iter().map(String::len).sum();
        assert!(archive_len >= 10_000, "archive holds real text");
    }

    #[tokio::test]
    async fn tiny_history_fits_without_archiving() {
        let history = history(2);
        let req = CompactionRequest::new(10_000, 8_000);
        let compactor = Compactor::deterministic_only();
        let plan = compactor.compact(&history, &ledger(), &req).await;
        assert!(plan.accepted);
        assert!(plan.archived.is_empty());
        // Two history turns + the synthesized identity + durable-facts
        // turns (goal `g` and open step `s` exist in the fixture).
        assert_eq!(plan.kept_recent.len(), 4);
        assert_eq!(
            plan.kept_recent
                .iter()
                .filter(|t| t.role != "system")
                .count(),
            2
        );
    }

    /// Rendered evicted-turn text exactly as the compactor archives it.
    fn rendered(evicted: &[RecentTurn]) -> String {
        let mut out = String::new();
        for t in evicted {
            out.push_str(&format!("{}: {}\n", t.role, t.text));
        }
        out
    }

    #[tokio::test]
    async fn deterministic_evictions_over_1mib_archive_chunked_nothing_lost() {
        // P0: the old 1 MiB archive cap silently dropped cold history. A
        // ~2.5 MiB eviction must now come back as MULTIPLE ordered chunks
        // whose concatenation equals the evicted turns EXACTLY — oldest and
        // newest evicted text included, nothing truncated.
        let history = history(6000); // ≈ 2.5 MiB of turn text
        let before = Estimator.estimate_tokens(&rendered(&history));
        let req = CompactionRequest::new(before, before / 5);
        let compactor = Compactor::deterministic_only();
        let plan = compactor.compact(&history, &ledger(), &req).await;
        assert!(plan.accepted, "deterministic pruning must fit the cap");
        assert!(
            plan.archive_chunks.len() >= 2,
            "> 1 MiB of evicted text must produce multiple chunks, got {}",
            plan.archive_chunks.len()
        );
        // Every chunk respects the bound; no empty chunks.
        assert!(plan
            .archive_chunks
            .iter()
            .all(|c| !c.is_empty() && c.len() <= ARCHIVE_CHUNK_BYTES));
        // The kept turns are the newest suffix (minus the synthesized
        // identity/facts turns and the digest at index 0).
        let kept: Vec<&RecentTurn> = plan
            .kept_recent
            .iter()
            .filter(|t| t.role != "system" && !t.text.starts_with("[Earlier context archived:"))
            .collect();
        assert_eq!(
            kept,
            history[history.len() - kept.len()..]
                .iter()
                .collect::<Vec<_>>()
        );
        let evicted = &history[..history.len() - kept.len()];
        assert!(!evicted.is_empty());
        // Concatenation (oldest first) equals the evicted text EXACTLY.
        let concat: String = plan.archive_chunks.concat();
        assert_eq!(
            concat,
            rendered(evicted),
            "archived material must be lossless"
        );
        // The oldest AND newest evicted text both survive.
        assert!(concat.starts_with(&format!("{}: ", evicted[0].role)));
        assert_eq!(concat.len(), rendered(evicted).len());
        assert_eq!(
            concat[concat.len() - evicted.last().unwrap().text.len() - 1..].trim_end(),
            evicted.last().unwrap().text
        );
    }

    #[tokio::test]
    async fn accepted_summary_archives_whole_history_chunked() {
        // The LlmSummary branch archives EVERY evicted turn (the whole
        // history) — over 1 MiB it must chunk, never truncate.
        struct GoodSummarizer;
        impl Summarizer for GoodSummarizer {
            fn summarize<'a>(
                &'a self,
                _residue: &'a [RecentTurn],
                durable_facts: &'a str,
            ) -> std::pin::Pin<Box<dyn std::future::Future<Output = String> + Send + 'a>>
            {
                Box::pin(async move { format!("COMPACT SUMMARY: {durable_facts}") })
            }
        }
        let history = history(6000); // ≈ 2.5 MiB of turn text
        let before = Estimator.estimate_tokens(&rendered(&history));
        let req = CompactionRequest::new(before, before / 5);
        let compactor = Compactor::new(Some(Arc::new(GoodSummarizer)));
        let plan = compactor.compact(&history, &ledger(), &req).await;
        assert!(plan.accepted);
        assert_eq!(plan.strategy, CompactionStrategy::LlmSummary);
        assert!(
            plan.archive_chunks.len() >= 2,
            "> 1 MiB of evicted text must produce multiple chunks, got {}",
            plan.archive_chunks.len()
        );
        assert!(plan
            .archive_chunks
            .iter()
            .all(|c| !c.is_empty() && c.len() <= ARCHIVE_CHUNK_BYTES));
        assert_eq!(
            plan.archive_chunks.concat(),
            rendered(&history),
            "the whole evicted history must be archived losslessly"
        );
    }

    #[test]
    fn fill_chunks_never_splits_mid_turn_and_preserves_order() {
        // Chunk boundaries fall ONLY between turns; a single turn larger
        // than the bound occupies its own WHOLE chunk (never a fragment);
        // oldest-first order is preserved end to end.
        let turn = |text: &str| RecentTurn {
            role: "assistant".into(),
            text: text.to_string(),
        };
        let turns = vec![turn("aa"), turn("bbbb"), turn("cc"), turn("dddddddddd")];
        // Bound 12 bytes: "assistant: aa\n" (13) already exceeds it.
        let chunks = fill_chunks(turns.iter(), 12);
        // Every turn whole in its own chunk: the rendered lines are larger
        // than the bound, so nothing may be joined.
        assert_eq!(
            chunks,
            vec![
                "assistant: aa\n".to_string(),
                "assistant: bbbb\n".to_string(),
                "assistant: cc\n".to_string(),
                "assistant: dddddddddd\n".to_string(),
            ]
        );
        // A bound that fits two whole turns joins them, oldest first, and
        // never lets a third straddle a boundary ("cc" would push chunk 0
        // to 43 > 30, so it opens its own chunk).
        let chunks = fill_chunks(turns.iter(), 30);
        assert_eq!(
            chunks,
            vec![
                "assistant: aa\nassistant: bbbb\n".to_string(),
                "assistant: cc\n".to_string(),
                "assistant: dddddddddd\n".to_string(),
            ]
        );
        // Concatenation is always the lossless rendered text, in order.
        assert_eq!(chunks.concat(), rendered(&turns));
    }

    #[tokio::test]
    async fn zero_reduction_never_accepted_even_at_target() {
        // before == target: any "compaction" is a zero reduction → rejected.
        let history = history(10);
        let before = 5_000;
        let req = CompactionRequest::new(before, before);
        let compactor = Compactor::deterministic_only();
        let plan = compactor.compact(&history, &ledger(), &req).await;
        // hard_cap = before * 0.75 < before, so after (>= ledger) may or may
        // not fit; what must NEVER happen is `accepted` with after == before.
        if plan.accepted {
            assert!(
                plan.after_tokens < plan.before_tokens,
                "zero-reduction compaction accepted!"
            );
        }
    }

    #[test]
    fn would_compact_again_boundaries() {
        let compactor = Compactor::deterministic_only();
        let plan = CompactionPlan {
            accepted: true,
            before_tokens: 100_000,
            after_tokens: 80_000,
            target_tokens: 80_000,
            strategy: CompactionStrategy::DeterministicPruning,
            ledger: ledger(),
            kept_recent: vec![],
            archived: vec![],
            archive_chunks: vec![],
            evidence_refs: vec![],
        };
        let req = CompactionRequest::new(100_000, 80_000);
        // After == target: done.
        assert!(!compactor.would_compact_again(&plan, &req));
        // Rejected plans always need attention.
        let rejected = CompactionPlan {
            accepted: false,
            ..plan.clone()
        };
        assert!(compactor.would_compact_again(&rejected, &req));
        // Above target: must compact again.
        let above = CompactionPlan {
            after_tokens: 90_000,
            ..plan
        };
        let req = CompactionRequest::new(100_000, 70_000);
        assert!(compactor.would_compact_again(&above, &req));
    }

    #[test]
    fn hostile_ratio_values_are_clamped() {
        let req = CompactionRequest {
            before_tokens: 100,
            target_tokens: 10,
            min_reduction_ratio: 5.0, // hostile
            ..Default::default()
        };
        let cap = req.hard_cap();
        assert!(cap <= 10);
        let req = CompactionRequest {
            before_tokens: 100,
            target_tokens: 10,
            min_reduction_ratio: -1.0, // hostile
            ..Default::default()
        };
        assert!(req.hard_cap() <= 10);
    }

    // ===================================================================
    // Audit 70/71 adversarial rows
    // ===================================================================

    /// Records every summarizer invocation and the exact inputs it saw.
    #[derive(Default)]
    struct SpySummarizer {
        calls: std::sync::atomic::AtomicUsize,
        residue: Mutex<Vec<RecentTurn>>,
        facts: Mutex<String>,
    }

    impl SpySummarizer {
        fn new() -> Self {
            Self::default()
        }

        fn calls(&self) -> usize {
            self.calls.load(std::sync::atomic::Ordering::SeqCst)
        }

        fn residue(&self) -> Vec<RecentTurn> {
            self.residue.lock().expect("spy residue lock").clone()
        }

        fn facts(&self) -> String {
            self.facts.lock().expect("spy facts lock").clone()
        }
    }

    impl Summarizer for SpySummarizer {
        fn summarize<'a>(
            &'a self,
            residue: &'a [RecentTurn],
            durable_facts: &'a str,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = String> + Send + 'a>> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            *self.residue.lock().expect("spy residue lock") = residue.to_vec();
            *self.facts.lock().expect("spy facts lock") = durable_facts.to_string();
            Box::pin(async move { "SHORT SUMMARY".to_string() })
        }
    }

    #[tokio::test]
    async fn ten_megabyte_tool_output_becomes_evidence_ref_and_original_retrievable() {
        // (a) A 10 MiB historical tool output must leave the wire as an
        // evidence ref whose compact body is <= 16 KiB, with the original
        // retrievable byte-for-byte from the evidence store.
        let line = "compile warning: unused variable `x` in module foo\n";
        let target = 10 * 1024 * 1024;
        let mut raw = String::with_capacity(target + line.len());
        while raw.len() < target {
            raw.push_str(line);
        }
        assert!(raw.len() >= target);
        let history = vec![
            RecentTurn {
                role: "tool".into(),
                text: raw.clone(),
            },
            RecentTurn {
                role: "assistant".into(),
                text: "analysis complete".into(),
            },
        ];
        let compactor = Compactor::deterministic_only();
        let req = CompactionRequest::new(3_000_000, 200_000);
        let plan = compactor.compact(&history, &ledger(), &req).await;
        assert!(plan.accepted, "the compacted plan must fit the cap");
        assert_eq!(
            plan.evidence_refs.len(),
            1,
            "the huge tool output must become exactly one evidence ref"
        );
        let ev = &plan.evidence_refs[0];
        assert!(ev.original_bytes >= target);
        assert!(
            ev.compact_body.len() <= EVIDENCE_COMPACT_BODY_MAX_BYTES,
            "compact body {} exceeds the 16 KiB evidence bound",
            ev.compact_body.len()
        );
        assert!(ev.compact_bytes <= EVIDENCE_COMPACT_BODY_MAX_BYTES);
        // The wire carries the ref, never the 10 MiB payload.
        assert!(plan
            .kept_recent
            .iter()
            .all(|t| t.text.len() <= EVIDENCE_COMPACT_BODY_MAX_BYTES + 512));
        // The original is retrievable via the evidence store.
        let archive = compactor.evidence();
        let stored = archive
            .retrieve(ev.id)
            .expect("evidence must be retrievable");
        assert_eq!(stored.backing.as_deref(), Some(raw.as_bytes()));
        assert_eq!(
            stored.envelope.compression.original_bytes as usize,
            raw.len()
        );
        assert_eq!(stored.envelope.backing_hash, Some(ev.backing_hash));
    }

    #[tokio::test]
    async fn identical_huge_tool_outputs_content_address_to_one_evidence_ref() {
        // Two identical huge outputs (exact-fact dedupe at the evidence
        // layer and at the row layer): ONE evidence id, ONE ref row.
        let raw = "same tool output line\n".repeat(20_000); // ~440 KiB
        let history = vec![
            RecentTurn {
                role: "tool".into(),
                text: raw.clone(),
            },
            RecentTurn {
                role: "tool".into(),
                text: raw.clone(),
            },
        ];
        let compactor = Compactor::deterministic_only();
        let req = CompactionRequest::new(500_000, 50_000);
        let plan = compactor.compact(&history, &ledger(), &req).await;
        assert_eq!(plan.evidence_refs.len(), 1, "content-addressed dedupe");
        assert_eq!(
            plan.kept_recent.iter().filter(|t| t.role == "tool").count(),
            1,
            "identical ref rows collapse to one"
        );
        assert_eq!(compactor.evidence().len(), 1);
    }

    #[tokio::test]
    async fn duplicate_exact_facts_collapse_before_summarization_and_pruning() {
        // (b) Byte-identical (role, text) rows collapse to their first
        // occurrence before the residue reaches the model.
        let fact = RecentTurn {
            role: "assistant".into(),
            text: "FACT: parser is recursive descent".into(),
        };
        let history = vec![
            fact.clone(),
            fact.clone(),
            fact.clone(),
            RecentTurn {
                role: "assistant".into(),
                text: "unique tail".into(),
            },
        ];
        let spy = Arc::new(SpySummarizer::new());
        let compactor = Compactor::new(Some(spy.clone()));
        let req = CompactionRequest {
            summarizer_residue_budget_tokens: 0,
            ..CompactionRequest::new(100_000, 50_000)
        };
        let plan = compactor.compact(&history, &ledger(), &req).await;
        assert_eq!(spy.calls(), 1, "the non-empty residue was summarized");
        let residue = spy.residue();
        assert_eq!(
            residue.len(),
            2,
            "identical rows must collapse: {residue:?}"
        );
        assert_eq!(
            residue
                .iter()
                .filter(|t| t.text.contains("FACT: parser"))
                .count(),
            1
        );
        assert_eq!(plan.strategy, CompactionStrategy::LlmSummary);
    }

    #[tokio::test]
    async fn goal_and_criteria_stay_verbatim_and_never_enter_summarizer_input() {
        // (c) The user's ask is excluded from the summarization input
        // entirely and rides the post-compaction wire byte-for-byte.
        let goal = "GOAL-Ω-ship-the-parser-😀";
        let criterion = "CRITERION-Ω-tests-pass-verbatim";
        let projection = TaskContextProjection::from_durable_rows(DurableTaskRows {
            goal: goal.into(),
            criteria: vec![criterion.into()],
            ..Default::default()
        });
        let spy = Arc::new(SpySummarizer::new());
        let compactor = Compactor::new(Some(spy.clone()));
        let history = history(200);
        let req = CompactionRequest::new(100_000, 30_000);
        let plan = compactor
            .compact_projected(&history, &projection, &TaskLedger::default(), &req)
            .await;
        assert_eq!(spy.calls(), 1, "large residue invokes the summarizer");
        assert!(
            !spy.facts().contains(goal),
            "goal must never reach the summarizer"
        );
        assert!(!spy.facts().contains(criterion));
        assert!(
            !render_turns(spy.residue().iter()).contains(goal),
            "goal must not be smuggled in the residue"
        );
        assert!(
            plan.kept_recent.iter().any(|t| t.text.contains(goal)),
            "post-compaction wire must carry the goal verbatim"
        );
        assert!(plan.kept_recent.iter().any(|t| t.text.contains(criterion)));
    }

    #[tokio::test]
    async fn decisions_and_checks_come_from_durable_rows_not_the_transcript() {
        // (d) The transcript lies; the projection reports durable truth.
        let projection = TaskContextProjection::from_durable_rows(DurableTaskRows {
            task_state: "Verifying".into(),
            goal: "durable goal".into(),
            decisions: vec![ProjectedDecision {
                step: "3".into(),
                choice: "DURABLE-CHOICE-X".into(),
                rationale: "durable rationale".into(),
            }],
            checks: vec![ProjectedCheck {
                name: "cargo test".into(),
                status: "passed".into(),
                summary: "all green".into(),
            }],
            ..Default::default()
        });
        let history = vec![RecentTurn {
            role: "assistant".into(),
            text: "we decided to use TRANSCRIPT-LIE and the checks FAILED".into(),
        }];
        let compactor = Compactor::deterministic_only();
        let req = CompactionRequest::new(50_000, 20_000);
        let plan = compactor
            .compact_projected(&history, &projection, &TaskLedger::default(), &req)
            .await;
        assert!(plan.accepted);
        let facts_turn = plan
            .kept_recent
            .iter()
            .find(|t| t.role == "system" && t.text.contains("DURABLE TASK PROJECTION"))
            .expect("the durable facts turn must ride the wire");
        assert!(facts_turn.text.contains("DURABLE-CHOICE-X"));
        assert!(facts_turn.text.contains("cargo test"));
        assert!(facts_turn.text.contains("passed"));
        assert!(
            !facts_turn.text.contains("TRANSCRIPT-LIE"),
            "the durable projection cannot be rewritten by the transcript"
        );
        // Reinforce the source of truth: the projection itself is rows-only.
        assert!(projection
            .summarizable_render()
            .contains("DURABLE-CHOICE-X"));
        assert!(!projection.summarizable_render().contains("TRANSCRIPT-LIE"));
    }

    #[tokio::test]
    async fn llm_summarization_only_above_the_residue_budget() {
        // (f) A spy counter proves the model is invoked only when the
        // post-extraction residue exceeds the configured budget.
        let spy = Arc::new(SpySummarizer::new());
        let compactor = Compactor::new(Some(spy.clone()));
        let small = vec![
            RecentTurn {
                role: "user".into(),
                text: "hi".into(),
            },
            RecentTurn {
                role: "assistant".into(),
                text: "hello".into(),
            },
        ];
        let roomy = CompactionRequest {
            summarizer_residue_budget_tokens: 1_000_000,
            ..CompactionRequest::new(10_000, 5_000)
        };
        let plan = compactor.compact(&small, &ledger(), &roomy).await;
        assert_eq!(spy.calls(), 0, "small residue must not call the model");
        assert_eq!(plan.strategy, CompactionStrategy::DeterministicPruning);
        assert!(plan.accepted);

        let big = history(200);
        let tight = CompactionRequest {
            summarizer_residue_budget_tokens: 128,
            ..CompactionRequest::new(100_000, 30_000)
        };
        let plan = compactor.compact(&big, &ledger(), &tight).await;
        assert_eq!(
            spy.calls(),
            1,
            "residue above budget invokes the model once"
        );
        assert_eq!(plan.strategy, CompactionStrategy::LlmSummary);
    }
}
