//! The production context compiler (audit 2/3/41/42/70): the ONE content
//! authority that turns durable task facts into the evidence a turn sends.
//!
//! ```text
//! Task + active WorkItem + criteria + failures + verification state
//!   => NeedSet      (criterion => REQUIRED need; failed test => high;
//!                     compile-error symbol => Need(symbol); owned paths =>
//!                     Need; recently changed file => moderate)
//!   => retrieval    (scoped durable evidence + semantic DATA + learning
//!                     omission priors)
//!   => information-gain selection (required coverage first; redundant
//!                     S/M/L variants of one evidence group never both)
//!   => S/M/L        (Summary/Structure/Exact variants, one level per group)
//!   => CCR          (bounded compact bodies; backing stays retrievable)
//!   => PromptSegments (measured segment accounting of the compiled render)
//! ```
//!
//! # One selector, one rerun
//!
//! Selection runs [`select_by_information_with_prior`] — the crate's single
//! selector — and nothing here post-filters the result. When the REQUIRED
//! set alone exceeds the envelope the compiler does not drop evidence: it
//! re-runs deterministically under a corrected envelope (cheaper levels,
//! smaller CCR body caps) and only if even the smallest fully-covering
//! variants overflow returns the typed [`CompilerError::EnvelopeOverflow`]
//! with the section costs and the exact overage. Required needs are never
//! silently uncovered on an overflow.
//!
//! # Parity
//!
//! A compiler with no evidence source (or an empty store, or no needs)
//! selects nothing and renders an empty evidence segment: the caller keeps
//! its baseline content byte-for-byte. Provider retrieval failures are a
//! typed error the caller treats as "no compiled evidence", never a panic.

use std::sync::Arc;

use faktor_core::state::{CriterionOrigin, CriterionRequirement};
use faktor_core::{SessionId, WorkspaceId};
use faktor_evidence::store::{EvidencePage, MemoryEvidenceStore, StoredEvidence};
use faktor_evidence::types::{EvidenceEnvelope, EvidenceError, ProvenanceSet};

use crate::estimator::Estimator;
use crate::information::{
    required_candidates, select_by_information, select_by_information_with_prior, FailurePrior,
    InformationBudget, InformationError, Need,
};
use crate::selection::{
    CandidateKind, CandidateRequirement, ContextCandidate, EvidenceLevel, NeedCoverage,
};
use crate::wire_plan::{PromptSegment, PromptSegments, PromptStability, SectionCosts};

// Re-exports: the runtime wires the durable authority and names evidence
// kinds/provenance through THIS module, so faktor-agent needs no direct
// dependency on faktor-evidence (the context crate is already its seam).
pub use faktor_evidence::store::{DurableEvidenceAuthority, EvidenceAccessContext};
pub use faktor_evidence::types::{EvidenceId, EvidenceKind, ProvenanceSource};

/// Hard bound on the envelopes one compile lists from the durable authority.
pub const MAX_COMPILED_ENVELOPES: usize = 256;
/// Hard bound on newest pages one compile may walk while filling candidates
/// after the directly referenced evidence was fetched by id.
pub const MAX_EVIDENCE_PAGES: usize = 64;
/// Hard bound on the compact body carried per compiled evidence item.
pub const MAX_COMPILED_BODY_BYTES: usize = 4096;
/// Deterministic envelope corrections before an overflow is terminal.
pub const MAX_COMPILE_RETRIES: u32 = 4;
/// Maximum keywords carried per need (bounded matching).
const MAX_NEED_KEYWORDS: usize = 12;

/// The scoped read seam the compiler retrieves through. Implemented by the
/// durable authority (production) and the in-memory store (tests); both
/// enforce the identical session/workspace/task scope rule.
pub trait ScopedEvidenceSource: Send + Sync {
    /// One newest-first scoped page (`created_ms DESC, id DESC`); `before` is
    /// the exclusive cursor from the previous page's `next_before`.
    fn list_scoped_newest(
        &self,
        ctx: &EvidenceAccessContext,
        before: Option<u64>,
        limit: usize,
    ) -> Result<EvidencePage, EvidenceError>;

    fn list_scoped_envelopes(
        &self,
        ctx: &EvidenceAccessContext,
        limit: usize,
    ) -> Result<Vec<EvidenceEnvelope>, EvidenceError>;

    fn get_scoped(
        &self,
        id: EvidenceId,
        ctx: &EvidenceAccessContext,
    ) -> Result<StoredEvidence, EvidenceError>;
}

impl ScopedEvidenceSource for DurableEvidenceAuthority {
    fn list_scoped_newest(
        &self,
        ctx: &EvidenceAccessContext,
        before: Option<u64>,
        limit: usize,
    ) -> Result<EvidencePage, EvidenceError> {
        DurableEvidenceAuthority::list_scoped_newest(self, ctx, before, limit)
    }

    fn list_scoped_envelopes(
        &self,
        ctx: &EvidenceAccessContext,
        limit: usize,
    ) -> Result<Vec<EvidenceEnvelope>, EvidenceError> {
        DurableEvidenceAuthority::list_scoped_envelopes(self, ctx, limit)
    }

    fn get_scoped(
        &self,
        id: EvidenceId,
        ctx: &EvidenceAccessContext,
    ) -> Result<StoredEvidence, EvidenceError> {
        DurableEvidenceAuthority::get_scoped(self, id, ctx)
    }
}

impl ScopedEvidenceSource for MemoryEvidenceStore {
    fn list_scoped_newest(
        &self,
        ctx: &EvidenceAccessContext,
        before: Option<u64>,
        limit: usize,
    ) -> Result<EvidencePage, EvidenceError> {
        Ok(MemoryEvidenceStore::list_scoped_newest(
            self, ctx, before, limit,
        ))
    }

    fn list_scoped_envelopes(
        &self,
        ctx: &EvidenceAccessContext,
        limit: usize,
    ) -> Result<Vec<EvidenceEnvelope>, EvidenceError> {
        Ok(MemoryEvidenceStore::list_scoped_envelopes(self, ctx, limit))
    }

    fn get_scoped(
        &self,
        id: EvidenceId,
        ctx: &EvidenceAccessContext,
    ) -> Result<StoredEvidence, EvidenceError> {
        MemoryEvidenceStore::stored_scoped(self, id, ctx)
    }
}

/// One active work item of the durable plan.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WorkItem {
    pub id: String,
    pub title: String,
    pub state: String,
    pub paths: Vec<String>,
    pub criteria: Vec<String>,
}

/// Durable verification state of the task, as the compiler sees it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum VerificationState {
    #[default]
    Unknown,
    Pending,
    Passed,
    Failed,
}

/// One TYPED acceptance criterion as the compiler sees it (audits
/// 56/57/105): the durable criterion's stable content id, text, binding
/// requirement, origin, optional explicit evidence edge and the semantic
/// snapshot it was derived from.
///
/// `evidence_source` is the durable evidence id that certifies the
/// criterion. It is an EXPLICIT edge: the compiler fetches that id by id
/// and treats the matching envelope as full, required coverage — keyword
/// matching is never needed for it (a criterion with opaque text still
/// retrieves its evidence). When the criterion's `semantic_snapshot` differs
/// from the task's current `semantic_snapshot`, the derived criterion's
/// evidence edge is STALE: the edge is not fetched and not attributed (the
/// criterion must be re-derived from the new snapshot), while keyword
/// matching on the text still applies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CriterionFact {
    /// The criterion's stable content id (`criterion:<id>` is the need id).
    pub id: String,
    pub text: String,
    pub requirement: CriterionRequirement,
    pub origin: CriterionOrigin,
    pub evidence_source: Option<EvidenceId>,
    pub semantic_snapshot: Option<String>,
}

/// The durable task facts one compile is generated from. This is the
/// "Task + active WorkItem + criteria + failures + verification state" input
/// of the audit: every field is read from durable state by the runtime.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TaskFacts {
    pub session_id: SessionId,
    pub workspace_id: WorkspaceId,
    pub task_id: Option<u64>,
    pub goal: String,
    pub active_work_item: Option<WorkItem>,
    /// The task's TYPED acceptance criteria (never raw strings: the id,
    /// requirement, origin and evidence edge are durable facts).
    pub criteria: Vec<CriterionFact>,
    pub failures: Vec<String>,
    pub verification_state: VerificationState,
    pub owned_paths: Vec<String>,
    pub changed_files: Vec<String>,
    /// The CURRENT semantic provider snapshot of the task, when one is
    /// known. A derived criterion carrying a DIFFERENT `semantic_snapshot`
    /// is stale. `None` means "unknown" — never a positive mismatch, so an
    /// absent provider never invalidates edges.
    pub semantic_snapshot: Option<String>,
}

impl TaskFacts {
    /// The scope context every retrieval runs under. A task-scoped fact set
    /// resolves to an explicit `Task(task_id)`; the historic task-less test
    /// shape resolves to the explicit admin scope (never a wildcard derived
    /// from a missing value).
    pub fn access(&self) -> EvidenceAccessContext {
        EvidenceAccessContext::new(self.session_id.raw(), self.workspace_id.raw(), self.task_id)
    }

    /// True when `criterion`'s derived semantic snapshot provably moved (both
    /// snapshots known and different). An unknown current snapshot is never a
    /// positive mismatch.
    pub fn criterion_snapshot_stale(&self, criterion: &CriterionFact) -> bool {
        match (&criterion.semantic_snapshot, &self.semantic_snapshot) {
            (Some(derived), Some(current)) => derived != current,
            _ => false,
        }
    }
}

/// Every evidence id the durable facts DIRECTLY reference: the explicit
/// [`CompilerInput::evidence_refs`] plus every TYPED criterion's
/// `evidence_source` (a stale derived edge is NOT referenced: retrieving it
/// would attribute coverage to evidence derived from a moved snapshot), plus
/// `evidence://<id>` / `evidence_source=<id>` / `evidence_id=<id>` tokens
/// embedded in failure and active work-item strings. These ids are fetched
/// by id in pass 1; the newest-page walk can never hide them.
pub fn referenced_evidence_ids(input: &CompilerInput) -> Vec<EvidenceId> {
    let mut out: Vec<EvidenceId> = Vec::new();
    let mut push = |id: EvidenceId| {
        if !out.contains(&id) {
            out.push(id);
        }
    };
    for id in &input.evidence_refs {
        push(*id);
    }
    let facts = &input.facts;
    // Explicit typed criterion edges: full-strength required retrieval with
    // NO keyword matching; a provably stale derived edge is skipped.
    for criterion in &facts.criteria {
        if let Some(id) = criterion.evidence_source {
            if !facts.criterion_snapshot_stale(criterion) {
                push(id);
            }
        }
    }
    let mut texts: Vec<&str> = Vec::new();
    texts.extend(facts.failures.iter().map(String::as_str));
    if let Some(item) = &facts.active_work_item {
        texts.push(item.title.as_str());
        texts.push(item.state.as_str());
        texts.extend(item.paths.iter().map(String::as_str));
        texts.extend(item.criteria.iter().map(String::as_str));
    }
    for text in texts {
        extract_evidence_refs(text, &mut push);
    }
    out
}

/// Scan one fact string for the bounded `evidence` id token forms. A token
/// that does not parse as a u64 is ignored (a revision like
/// `evidence://src/x.rs` is prose, never an id fabricated from a hash).
fn extract_evidence_refs(text: &str, push: &mut impl FnMut(EvidenceId)) {
    for token in text.split(|c: char| {
        c.is_whitespace() || matches!(c, ',' | ';' | '(' | ')' | '[' | ']' | '"' | '\'' | '`')
    }) {
        let raw = token
            .strip_prefix("evidence://")
            .or_else(|| token.strip_prefix("evidence_source="))
            .or_else(|| token.strip_prefix("evidence_id="))
            .or_else(|| token.strip_prefix("evidence="));
        if let Some(raw) = raw {
            if let Ok(n) = raw.parse::<u64>() {
                push(EvidenceId(n));
            }
        }
    }
}

/// Where one generated need came from (telemetry + deterministic ordering).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeedSource {
    Criterion,
    WorkItem,
    FailedTest,
    CompileSymbol,
    OwnedPath,
    RecentChange,
}

impl NeedSource {
    /// Base weight of a need from this source. Criteria are REQUIRED and
    /// therefore never gain-ranked; the weights below only drive the
    /// non-required gain ranking.
    pub const fn weight(self) -> f64 {
        match self {
            NeedSource::Criterion => 1.0,
            NeedSource::WorkItem => 0.8,
            NeedSource::FailedTest => 0.9,
            NeedSource::CompileSymbol => 0.85,
            NeedSource::OwnedPath => 0.6,
            NeedSource::RecentChange => 0.4,
        }
    }

    pub const fn required(self) -> bool {
        matches!(self, NeedSource::Criterion)
    }
}

/// One generated need plus the bounded keyword set that matches it against
/// evidence bodies and the explicit evidence edge that needs no matching.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledNeed {
    pub need: Need,
    pub keywords: Vec<String>,
    pub source: NeedSource,
    /// The typed criterion's explicit `evidence_source` edge, when it is
    /// fresh. The envelope with this id is REQUIRED coverage for the need
    /// regardless of keywords (opaque criterion text still retrieves it).
    pub evidence_source: Option<EvidenceId>,
    /// True when a derived criterion's semantic snapshot provably moved: the
    /// declared edge is stale and deliberately NOT in `evidence_source`.
    pub stale: bool,
}

/// The needs one turn must satisfy, in generation order (criteria first,
/// sorted by stable need id so a permutation of the criteria array is
/// bit-identical).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NeedSet {
    pub entries: Vec<CompiledNeed>,
}

impl NeedSet {
    pub fn from_facts(facts: &TaskFacts) -> Self {
        // Typed criteria. The need id is the criterion's STABLE content id
        // (`criterion:<id>`), never its array index: swapping two criteria
        // changes nothing in the generated needs, their ids or attribution.
        // A criterion with an explicit evidence edge survives even when its
        // text yields no keywords (the edge is retrieved by id).
        let mut criterion_needs: Vec<CompiledNeed> = Vec::new();
        for criterion in &facts.criteria {
            if criterion.id.is_empty() {
                continue; // no stable identity: cannot be a durable need
            }
            let stale = facts.criterion_snapshot_stale(criterion);
            let evidence_source = criterion.evidence_source.filter(|_| !stale);
            let keywords = keywords(&criterion.text);
            if keywords.is_empty() && evidence_source.is_none() {
                continue; // neither searchable nor explicitly referenced
            }
            criterion_needs.push(CompiledNeed {
                need: Need {
                    id: format!("criterion:{}", criterion.id),
                    weight: NeedSource::Criterion.weight(),
                    required: criterion.requirement.is_required(),
                },
                keywords,
                source: NeedSource::Criterion,
                evidence_source,
                stale,
            });
        }
        criterion_needs.sort_by(|a, b| a.need.id.cmp(&b.need.id));
        let mut entries: Vec<CompiledNeed> = criterion_needs;
        let mut push = |id: String, keywords: Vec<String>, source: NeedSource, required: bool| {
            if keywords.is_empty() {
                return; // nothing searchable: a need no evidence can match
            }
            entries.push(CompiledNeed {
                need: Need {
                    id,
                    weight: source.weight(),
                    required,
                },
                keywords,
                source,
                evidence_source: None,
                stale: false,
            });
        };
        if let Some(item) = &facts.active_work_item {
            let mut keys = keywords(&item.title);
            for path in &item.paths {
                keys.extend(path_keywords(path));
            }
            for criterion in &item.criteria {
                keys.extend(keywords(criterion));
            }
            push(
                format!("workitem:{}", item.id),
                dedupe(keys),
                NeedSource::WorkItem,
                false,
            );
        }
        for (i, failure) in facts.failures.iter().enumerate() {
            let lower = failure.to_ascii_lowercase();
            let is_test = lower.contains("test") || lower.contains("failed");
            let source = if is_test {
                NeedSource::FailedTest
            } else {
                NeedSource::CompileSymbol
            };
            let symbol = if is_test {
                None
            } else {
                extract_symbol(failure)
            };
            match symbol {
                Some(symbol) => push(
                    format!("symbol:{symbol}"),
                    vec![symbol.to_ascii_lowercase()],
                    NeedSource::CompileSymbol,
                    false,
                ),
                None => push(format!("failure:{i}"), keywords(failure), source, false),
            }
        }
        for path in &facts.owned_paths {
            push(
                format!("path:{}", path.trim()),
                path_keywords(path),
                NeedSource::OwnedPath,
                false,
            );
        }
        for path in &facts.changed_files {
            push(
                format!("recent:{}", path.trim()),
                path_keywords(path),
                NeedSource::RecentChange,
                false,
            );
        }
        Self { entries }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// The planner-facing needs in generation order.
    pub fn needs(&self) -> Vec<Need> {
        self.entries.iter().map(|e| e.need.clone()).collect()
    }
}

/// Explicit upper cap on the evidence envelope: a CEILING, never an
/// allocation. The old fixed split allocated `(context/3).clamp(256..32768)`
/// up front; the adaptive allocator below only ever treats this as a bound.
pub const MAX_VOLATILE_EVIDENCE_TOKENS: u32 = 32_768;
/// Explicit upper cap on the whole volatile competition (memory/runtime
/// ceiling): history, evidence, handoff summaries and tool notes together
/// can never claim more, however large the context window is.
pub const MAX_VOLATILE_TOTAL_TOKENS: u32 = 131_072;

/// The token demand of every volatile region that competes for the turn's
/// marginal-information budget. All values are caller-bounded estimates:
/// history is the loaded conversation's token estimate, evidence the
/// produced candidate evidence, handoff the child-session summaries and
/// tool-notes the current tool observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VolatileClaims {
    pub history_tokens: u32,
    pub evidence_tokens: u32,
    pub handoff_tokens: u32,
    pub tool_notes_tokens: u32,
}

/// Marginal-information weight of each volatile region, in ppm. The
/// allocation is pro-rata over `demand * weight`: a region whose marginal
/// token carries more information wins a larger share of the leftover
/// budget, up to its demand (a region can never be allocated more than it
/// can use). Required evidence is reserved BEFORE this competition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VolatileWeights {
    pub history_ppm: u32,
    pub evidence_ppm: u32,
    pub handoff_ppm: u32,
    pub tool_notes_ppm: u32,
}

impl Default for VolatileWeights {
    fn default() -> Self {
        Self {
            history_ppm: 1_000_000,
            evidence_ppm: 1_000_000,
            handoff_ppm: 600_000,
            tool_notes_ppm: 400_000,
        }
    }
}

impl VolatileWeights {
    /// The weights implied by the durable facts: a repair-heavy task (known
    /// failures, failed verification, or required criteria with explicit
    /// evidence edges) raises evidence's marginal information; a task with
    /// none keeps the neutral baseline.
    pub fn for_facts(facts: &TaskFacts) -> Self {
        let mut weights = Self::default();
        let mut evidence: u64 = u64::from(weights.evidence_ppm);
        if !facts.failures.is_empty() {
            evidence = evidence.saturating_add(1_000_000);
        }
        if facts.verification_state == VerificationState::Failed {
            evidence = evidence.saturating_add(500_000);
        }
        if facts.criteria.iter().any(|c| {
            c.requirement.is_required()
                && c.evidence_source.is_some()
                && !facts.criterion_snapshot_stale(c)
        }) {
            evidence = evidence.saturating_add(500_000);
        }
        weights.evidence_ppm = u32::try_from(evidence).unwrap_or(u32::MAX);
        weights
    }
}

/// One region's share of the volatile budget after the marginal-information
/// competition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VolatileBudget {
    /// The total volatile ceiling the competition ran under.
    pub ceiling: u32,
    /// The evidence tokens hard-reserved before any competition.
    pub required_evidence: u32,
    /// The evidence token envelope handed to information selection
    /// (`required_evidence` plus the evidence share of the leftover).
    pub evidence: u32,
    pub history: u32,
    pub handoff: u32,
    pub tool_notes: u32,
}

/// One region of the volatile competition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VolatileRegion {
    History,
    Evidence,
    Handoff,
    ToolNotes,
}

impl VolatileRegion {
    fn weight(self, weights: &VolatileWeights) -> u64 {
        u64::from(match self {
            Self::History => weights.history_ppm,
            Self::Evidence => weights.evidence_ppm,
            Self::Handoff => weights.handoff_ppm,
            Self::ToolNotes => weights.tool_notes_ppm,
        })
    }
}

/// Allocate the volatile budget by marginal information:
///
/// 1. REQUIRED evidence is hard-reserved first (`min(required, ceiling)`);
///    when the required set alone exceeds the ceiling the reservation is the
///    whole ceiling and selection surfaces the typed overflow — required
///    content is never silently dropped;
/// 2. the leftover is distributed pro-rata over `demand * weight` (integer
///    math), each region capped at its demand, in at most one pass per
///    region (bounded; the remainder stays unallocated — a ceiling, never a
///    guarantee);
/// 3. evidence can never exceed `evidence_ceiling` (the explicit
///    memory/runtime cap), and every other region is bounded by its demand.
pub fn allocate_volatile_budget(
    ceiling: u32,
    evidence_ceiling: u32,
    required_evidence: u32,
    claims: &VolatileClaims,
    weights: &VolatileWeights,
) -> VolatileBudget {
    let ceiling = ceiling.min(MAX_VOLATILE_TOTAL_TOKENS);
    let evidence_ceiling = evidence_ceiling.min(ceiling);
    let reserved = required_evidence.min(evidence_ceiling);
    let mut budget = VolatileBudget {
        ceiling,
        required_evidence: reserved,
        evidence: reserved,
        ..VolatileBudget::default()
    };
    let mut left = ceiling.saturating_sub(reserved);
    if left == 0 {
        return budget;
    }
    // (region, remaining demand); evidence demand excludes its reservation.
    let mut demand: [(VolatileRegion, u32); 4] = [
        (VolatileRegion::History, claims.history_tokens),
        (
            VolatileRegion::Evidence,
            claims
                .evidence_tokens
                .saturating_sub(reserved)
                .min(evidence_ceiling.saturating_sub(reserved)),
        ),
        (VolatileRegion::Handoff, claims.handoff_tokens),
        (VolatileRegion::ToolNotes, claims.tool_notes_tokens),
    ];
    for _ in 0..demand.len() {
        if left == 0 {
            break;
        }
        let total: u64 = demand
            .iter()
            .map(|(region, tokens)| u64::from(*tokens) * region.weight(weights) / 1_000_000)
            .sum();
        if total == 0 {
            break;
        }
        let mut distributed: u64 = 0;
        let mut all_saturated = true;
        for (region, tokens) in demand.iter_mut() {
            if *tokens == 0 {
                continue;
            }
            let weighted = u64::from(*tokens) * region.weight(weights) / 1_000_000;
            let share = u64::from(left) * weighted / total;
            let give = u32::try_from(share).unwrap_or(u32::MAX).min(*tokens);
            if give == 0 {
                all_saturated = false;
                continue;
            }
            match region {
                VolatileRegion::History => {
                    budget.history = budget.history.saturating_add(give);
                }
                VolatileRegion::Evidence => {
                    budget.evidence = budget.evidence.saturating_add(give);
                }
                VolatileRegion::Handoff => {
                    budget.handoff = budget.handoff.saturating_add(give);
                }
                VolatileRegion::ToolNotes => {
                    budget.tool_notes = budget.tool_notes.saturating_add(give);
                }
            }
            *tokens -= give;
            distributed = distributed.saturating_add(u64::from(give));
            if *tokens > 0 {
                all_saturated = false;
            }
        }
        left = left.saturating_sub(u32::try_from(distributed).unwrap_or(u32::MAX));
        if distributed == 0 || all_saturated {
            break;
        }
    }
    budget
}

/// One compile input: the durable facts, the volatile token ceiling the
/// allocation runs under, the competing region claims, and any
/// caller-supplied supplemental envelopes (producers that could not persist,
/// and unit tests).
#[derive(Debug, Clone)]
pub struct CompilerInput {
    pub facts: TaskFacts,
    /// The TOTAL volatile ceiling (memory/runtime bound), not the evidence
    /// envelope: the compiler allocates the evidence share from it.
    pub budget_tokens: u32,
    pub supplemental: Vec<EvidenceEnvelope>,
    /// Evidence ids the durable facts directly reference (criteria rows,
    /// failure rows, work-item state). They are fetched BY ID before any
    /// newest page walk, so a capped listing can never hide required
    /// evidence. `evidence://<id>` / `evidence_source=<id>` tokens embedded
    /// in the fact strings are picked up automatically as well.
    pub evidence_refs: Vec<EvidenceId>,
    /// The competing volatile claims (history/evidence/handoff/tool-notes)
    /// the marginal-information allocation runs over.
    pub volatile: VolatileClaims,
}

impl CompilerInput {
    pub fn new(facts: TaskFacts, budget_tokens: u32) -> Self {
        Self {
            facts,
            budget_tokens,
            supplemental: Vec::new(),
            evidence_refs: Vec::new(),
            volatile: VolatileClaims::default(),
        }
    }

    pub fn with_supplemental(mut self, envelopes: Vec<EvidenceEnvelope>) -> Self {
        self.supplemental = envelopes;
        self
    }

    pub fn with_evidence_refs(mut self, refs: Vec<EvidenceId>) -> Self {
        self.evidence_refs = refs;
        self
    }

    pub fn with_volatile(mut self, claims: VolatileClaims) -> Self {
        self.volatile = claims;
        self
    }
}

/// One compiled evidence item: the S/M/L variant that survived selection.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledEvidence {
    pub id: EvidenceId,
    pub kind: EvidenceKind,
    pub level: EvidenceLevel,
    pub path: String,
    pub body: String,
    pub score: f64,
    pub required: bool,
    pub tokens: u32,
    pub provenance: ProvenanceSet,
}

impl CompiledEvidence {
    /// The renderer-shaped block (`### path` + body), byte-identical to the
    /// wire renderer's evidence section shape.
    pub fn render(&self) -> String {
        format!("\n### {}\n{}\n", self.path, self.body)
    }

    /// The compact, retrievable reference line: CCR keeps the compact body
    /// in the prompt and the full backing behind the evidence id.
    pub fn reference(&self) -> String {
        format!(
            "[evidence://{} kind={:?} level={:?} required={}]",
            self.id, self.kind, self.level, self.required
        )
    }
}

/// The compiled turn content: needs, selected evidence and the measured
/// segment accounting of the compiled render.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledContext {
    pub needs: Vec<Need>,
    pub selected: Vec<CompiledEvidence>,
    pub total_tokens: u32,
    pub required_tokens: u32,
    pub retries: u32,
    pub segments: PromptSegments,
    pub render: String,
}

impl CompiledContext {
    fn empty(needs: Vec<Need>) -> Self {
        Self {
            needs,
            selected: Vec::new(),
            total_tokens: 0,
            required_tokens: 0,
            retries: 0,
            segments: PromptSegments::default(),
            render: String::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.selected.is_empty()
    }
}

/// Typed compiler failure. Retrieval/selection failures are distinct from
/// the envelope overflow so callers can react structurally (a provider
/// failure falls back to producer content; an overflow is a hard budget
/// verdict, never a silent drop).
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CompilerError {
    #[error("evidence retrieval failed: {0}")]
    Evidence(String),
    #[error(
        "compiled envelope overflow: {over_by} tokens over the {budget} token envelope after {retries} deterministic replans"
    )]
    EnvelopeOverflow {
        section_costs: SectionCosts,
        over_by: u32,
        budget: u32,
        retries: u32,
    },
    #[error("information selection failed: {0}")]
    Selection(String),
}

/// Bounds the compiler runs under (bounded everything). The two `max_*`
/// token fields are explicit CEILINGS for the adaptive volatile allocation
/// (memory/runtime caps), never pre-reserved allocations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompilerLimits {
    pub max_envelopes: usize,
    pub compact_body_cap: usize,
    /// Ceiling of the whole volatile competition.
    pub max_volatile_tokens: u32,
    /// Ceiling of the evidence envelope specifically.
    pub max_evidence_tokens: u32,
}

impl Default for CompilerLimits {
    fn default() -> Self {
        Self {
            max_envelopes: MAX_COMPILED_ENVELOPES,
            compact_body_cap: MAX_COMPILED_BODY_BYTES,
            max_volatile_tokens: MAX_VOLATILE_TOTAL_TOKENS,
            max_evidence_tokens: MAX_VOLATILE_EVIDENCE_TOKENS,
        }
    }
}

/// The deterministic envelope of one retry attempt: which S/M/L variants are
/// offered, the CCR body cap, and whether every matched variant declares
/// full coverage (so a cheap full-coverage variant can cover required needs
/// when the exact one no longer fits).
#[derive(Debug, Clone, Copy)]
struct AttemptPolicy {
    levels: &'static [EvidenceLevel],
    body_cap: usize,
    full_coverage_floor: bool,
}

impl AttemptPolicy {
    fn for_attempt(attempt: u32, limits: &CompilerLimits) -> Self {
        const ALL: &[EvidenceLevel] = &[
            EvidenceLevel::Summary,
            EvidenceLevel::Structure,
            EvidenceLevel::Exact,
        ];
        const SUMMARY: &[EvidenceLevel] = &[EvidenceLevel::Summary];
        match attempt {
            0 => Self {
                levels: ALL,
                body_cap: limits.compact_body_cap,
                full_coverage_floor: false,
            },
            1 => Self {
                levels: ALL,
                body_cap: (limits.compact_body_cap / 2).max(64),
                full_coverage_floor: false,
            },
            2 => Self {
                levels: ALL,
                body_cap: (limits.compact_body_cap / 4).max(64),
                full_coverage_floor: true,
            },
            3 => Self {
                levels: SUMMARY,
                body_cap: (limits.compact_body_cap / 8).max(64),
                full_coverage_floor: true,
            },
            _ => Self {
                levels: SUMMARY,
                body_cap: 256.min((limits.compact_body_cap / 16).max(64)),
                full_coverage_floor: true,
            },
        }
    }
}

/// The production compiler. `evidence` is the durable authority's scoped
/// read seam (None = no store: neutral parity), `learning` the failure
/// prior (applied to non-Required candidates only), `tokens` the shared
/// model-targeted token cache.
pub struct ContextCompiler {
    evidence: Option<Arc<dyn ScopedEvidenceSource>>,
    learning: Option<Arc<dyn FailurePrior + Send + Sync>>,
    limits: CompilerLimits,
}

impl std::fmt::Debug for ContextCompiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContextCompiler")
            .field("has_evidence", &self.evidence.is_some())
            .field("has_learning", &self.learning.is_some())
            .field("limits", &self.limits)
            .finish()
    }
}

impl ContextCompiler {
    pub fn new(
        evidence: Option<Arc<dyn ScopedEvidenceSource>>,
        learning: Option<Arc<dyn FailurePrior + Send + Sync>>,
    ) -> Self {
        Self {
            evidence,
            learning,
            limits: CompilerLimits::default(),
        }
    }

    pub fn with_limits(mut self, limits: CompilerLimits) -> Self {
        self.limits = limits;
        self
    }

    pub fn has_evidence(&self) -> bool {
        self.evidence.is_some()
    }

    /// Compile one turn's evidence. Deterministic: identical input and store
    /// state produce identical output (ids order the tie-breaks; retries are
    /// a fixed sequence).
    pub fn compile(&self, input: &CompilerInput) -> Result<CompiledContext, CompilerError> {
        let need_set = NeedSet::from_facts(&input.facts);
        let mut envelopes = input.supplemental.clone();
        // Deduplicate by evidence id: supplemental first, then fetched order.
        let mut seen = std::collections::HashSet::new();
        envelopes.retain(|env| seen.insert(env.id));
        if let Some(source) = &self.evidence {
            let ctx = input.facts.access();
            // PASS 1 — directly referenced required evidence. Every id the
            // durable facts name (criterion/failure/work-item state) is
            // fetched BY ID, so the newest-page cap can never hide it.
            for id in referenced_evidence_ids(input) {
                if seen.contains(&id) {
                    continue;
                }
                match source.get_scoped(id, &ctx) {
                    Ok(stored) => {
                        seen.insert(stored.envelope.id);
                        envelopes.push(stored.envelope);
                    }
                    Err(err) => {
                        tracing::debug!("direct evidence {id} not retrievable: {err}");
                    }
                }
            }
            // PASS 2 — fill the remaining candidate slots from the newest
            // pages (bounded by MAX_COMPILED_ENVELOPES and MAX_EVIDENCE_PAGES).
            let mut before = None;
            let mut pages = 0usize;
            while envelopes.len() < self.limits.max_envelopes && pages < MAX_EVIDENCE_PAGES {
                let page = source
                    .list_scoped_newest(&ctx, before, self.limits.max_envelopes)
                    .map_err(|e| CompilerError::Evidence(e.to_string()))?;
                if page.rows.is_empty() {
                    break;
                }
                let progress = page.next_before;
                for env in page.rows {
                    if envelopes.len() >= self.limits.max_envelopes {
                        break;
                    }
                    if seen.insert(env.id) {
                        envelopes.push(env);
                    }
                }
                match progress {
                    Some(next) if Some(next) != before => before = Some(next),
                    _ => break,
                }
                pages += 1;
            }
        }
        if envelopes.is_empty() || need_set.is_empty() {
            return Ok(CompiledContext::empty(need_set.needs()));
        }
        let needs = need_set.needs();
        // The adaptive volatile allocation: explicit ceilings bound the
        // competition, required evidence is hard-reserved inside it, and the
        // leftover is shared by marginal information across
        // history/evidence/handoff/tool-notes — never a fixed third.
        let ceiling = input.budget_tokens.min(self.limits.max_volatile_tokens);
        let evidence_ceiling = self.limits.max_evidence_tokens.min(ceiling);
        let weights = VolatileWeights::for_facts(&input.facts);
        let mut last_required = 0u64;
        let mut last_budget = evidence_ceiling;
        for attempt in 0..=MAX_COMPILE_RETRIES {
            let policy = AttemptPolicy::for_attempt(attempt, &self.limits);
            let (candidates, metas) = build_candidates(&envelopes, &need_set, &policy);
            // The minimal required set is reserved BEFORE any competition;
            // a corrected retry shrinks the variants, so the reservation
            // follows the cheapest full-coverage forms.
            let required_tokens: u64 = required_candidates(&candidates, &needs)
                .iter()
                .map(|candidate| u64::from(candidate.estimate_tokens))
                .sum();
            let allocation = allocate_volatile_budget(
                ceiling,
                evidence_ceiling,
                u32::try_from(required_tokens).unwrap_or(u32::MAX),
                &input.volatile,
                &weights,
            );
            last_budget = allocation.evidence;
            let budget = InformationBudget {
                token_budget: allocation.evidence,
                needs: needs.clone(),
            };
            let selection = match &self.learning {
                Some(prior) => {
                    select_by_information_with_prior(&candidates, &budget, prior.as_ref())
                }
                None => select_by_information(&candidates, &budget),
            };
            match selection {
                Ok(selected) => {
                    return Ok(finish(need_set, selected, &metas, &envelopes, attempt));
                }
                Err(InformationError::Oversized {
                    required_tokens, ..
                }) => {
                    last_required = required_tokens;
                    continue;
                }
            }
        }
        let over_by = last_required
            .saturating_sub(u64::from(last_budget))
            .min(u64::from(u32::MAX)) as u32;
        Err(CompilerError::EnvelopeOverflow {
            section_costs: SectionCosts {
                evidence: usize::try_from(last_required).unwrap_or(usize::MAX),
                ..SectionCosts::default()
            },
            over_by,
            budget: last_budget,
            retries: MAX_COMPILE_RETRIES,
        })
    }
}

/// The per-candidate back-reference the compiler keeps beside each
/// [`ContextCandidate`] (the planner value type carries no body).
struct CandidateMeta {
    envelope: usize,
    level: EvidenceLevel,
    body: String,
    required: bool,
}

/// Build the S/M/L candidates of every envelope under `policy`, plus the
/// meta map that maps each candidate id back to its envelope/level/body.
#[allow(clippy::type_complexity)]
fn build_candidates(
    envelopes: &[EvidenceEnvelope],
    need_set: &NeedSet,
    policy: &AttemptPolicy,
) -> (
    Vec<ContextCandidate>,
    std::collections::HashMap<String, CandidateMeta>,
) {
    let est = Estimator;
    let mut out: Vec<ContextCandidate> = Vec::new();
    let mut metas: std::collections::HashMap<String, CandidateMeta> =
        std::collections::HashMap::new();
    // The typed explicit edges: need id -> evidence id. The matching
    // envelope declares FULL required coverage with no keyword matching.
    let explicit: std::collections::HashMap<&str, u64> = need_set
        .entries
        .iter()
        .filter_map(|entry| {
            entry
                .evidence_source
                .map(|id| (entry.need.id.as_str(), id.0))
        })
        .collect();
    for (env_idx, env) in envelopes.iter().enumerate() {
        let text = searchable_text(env);
        let mut coverages: Vec<(usize, u64)> = Vec::new();
        for (idx, compiled) in need_set.entries.iter().enumerate() {
            let matched = if explicit.get(compiled.need.id.as_str()) == Some(&env.id.0) {
                1.0
            } else {
                match_fraction(&text, &compiled.keywords)
            };
            if matched <= 0.0 {
                continue;
            }
            coverages.push((idx, (matched * 1_000_000.0).min(1_000_000.0) as u64));
        }
        if coverages.is_empty() {
            continue;
        }
        // Attribution is keyed by the need's STABLE id, never by position:
        // a criteria array permutation yields bit-identical candidate
        // coverage.
        coverages.sort_by(|a, b| {
            need_set.entries[a.0]
                .need
                .id
                .cmp(&need_set.entries[b.0].need.id)
        });
        let required = coverages
            .iter()
            .any(|(idx, _)| need_set.entries[*idx].need.required);
        let max_raw = coverages.iter().map(|(_, ppm)| *ppm).max().unwrap_or(0);
        for &level in policy.levels {
            let body = bounded_body(env, level, policy.body_cap);
            if body.is_empty() {
                continue;
            }
            let tokens = est
                .estimate_tokens(&body)
                .saturating_add(2)
                .min(u32::MAX as usize) as u32;
            if tokens == 0 {
                continue;
            }
            // Under a corrected envelope every level declares FULL coverage
            // of every matched need (the corrected envelope trades detail
            // for required coverage, it never drops required content).
            let level_factor: u64 = if policy.full_coverage_floor {
                1_000_000
            } else {
                match level {
                    EvidenceLevel::Summary => 600_000,
                    EvidenceLevel::Structure => 800_000,
                    EvidenceLevel::Exact => 1_000_000,
                }
            };
            let need_coverage = coverages
                .iter()
                .map(|(idx, ppm)| NeedCoverage {
                    need_id: need_set.entries[*idx].need.id.clone(),
                    coverage_ppm: (ppm.saturating_mul(level_factor) / 1_000_000).min(1_000_000)
                        as u32,
                })
                .collect();
            let id = format!("ev:{}:{}", env.id.0, level_tag(level));
            out.push(ContextCandidate {
                id: id.clone(),
                kind: candidate_kind(env.kind),
                bytes: body.len(),
                estimate_tokens: tokens,
                utility: (max_raw as f64 / 1_000_000.0).clamp(0.0, 1.0),
                evidence: Some(env.id.0),
                requirement: if required {
                    CandidateRequirement::Required
                } else if max_raw >= 800_000 {
                    CandidateRequirement::Preferred
                } else {
                    CandidateRequirement::Optional
                },
                confidence_ppm: 900_000,
                freshness_ppm: 900_000,
                need_coverage,
                expected_error_reduction_ppm: max_raw.min(1_000_000) as u32,
                level,
                omission_keys: env.source_revision.clone().into_iter().collect(),
            });
            metas.insert(
                id,
                CandidateMeta {
                    envelope: env_idx,
                    level,
                    body,
                    required,
                },
            );
        }
    }
    (out, metas)
}

/// The measured compiled context: selected items in planner order plus the
/// segment accounting of their render. Bodies come from the meta map, so
/// the compiled render is the exact bounded body the selection priced.
fn finish(
    need_set: NeedSet,
    selection: crate::information::InformationSelection,
    metas: &std::collections::HashMap<String, CandidateMeta>,
    envelopes: &[EvidenceEnvelope],
    retries: u32,
) -> CompiledContext {
    let est = Estimator;
    let mut selected: Vec<CompiledEvidence> = Vec::new();
    for candidate in &selection.selected {
        if candidate.kind == CandidateKind::Message {
            continue; // the compiler compiles evidence; history is the renderer's
        }
        let Some(meta) = metas.get(candidate.id.as_str()) else {
            continue;
        };
        let env = &envelopes[meta.envelope];
        selected.push(CompiledEvidence {
            id: env.id,
            kind: env.kind,
            level: meta.level,
            path: env
                .source_revision
                .clone()
                .unwrap_or_else(|| format!("evidence://{}", env.id)),
            body: meta.body.clone(),
            score: candidate.utility,
            required: meta.required,
            tokens: candidate.estimate_tokens,
            provenance: env.provenance.clone(),
        });
    }
    let mut render = String::new();
    for item in &selected {
        render.push_str(&item.render());
    }
    let segment = PromptSegment {
        class: PromptStability::Volatile,
        bytes: render.len(),
        hash: faktor_core::hash::FileHash::from(*blake3::hash(render.as_bytes()).as_bytes()),
        tokens: u64::try_from(est.estimate_tokens(&render)).unwrap_or(u64::MAX),
    };
    let segments = PromptSegments {
        evidence: segment,
        ..PromptSegments::default()
    };
    CompiledContext {
        needs: need_set.needs(),
        selected,
        total_tokens: selection.selected_tokens,
        required_tokens: selection.required_tokens,
        retries,
        segments,
        render,
    }
}

fn level_tag(level: EvidenceLevel) -> &'static str {
    match level {
        EvidenceLevel::Summary => "s",
        EvidenceLevel::Structure => "m",
        EvidenceLevel::Exact => "l",
    }
}

fn candidate_kind(kind: EvidenceKind) -> CandidateKind {
    match kind {
        EvidenceKind::SymbolSet | EvidenceKind::SemanticContext => CandidateKind::Symbol,
        EvidenceKind::FileMap => CandidateKind::FileNote,
        EvidenceKind::ChildHandoff => CandidateKind::SubagentSummary,
        EvidenceKind::GenericText => CandidateKind::ToolNote,
        _ => CandidateKind::ToolNote,
    }
}

/// The searchable, lowercased text of one envelope: the compact body plus
/// the source revision (file path / learning digest) — never the backing
/// (which may be huge).
fn searchable_text(env: &EvidenceEnvelope) -> String {
    let mut text = String::with_capacity(env.compact.body.len() + 64);
    text.push_str(&env.compact.body.to_ascii_lowercase());
    text.push(' ');
    if let Some(revision) = &env.source_revision {
        text.push_str(&revision.to_ascii_lowercase());
    }
    text
}

/// Fraction of `keywords` present in `text` (`[0,1]`); empty keywords never
/// match. Bounded: keyword and text lengths are caller-bounded.
fn match_fraction(text: &str, keywords: &[String]) -> f64 {
    if keywords.is_empty() || text.is_empty() {
        return 0.0;
    }
    let hits = keywords
        .iter()
        .filter(|keyword| !keyword.is_empty() && text.contains(keyword.as_str()))
        .count();
    (hits as f64) / (keywords.len() as f64)
}

/// The bounded per-level body of one envelope: Summary carries a small
/// prefix, Structure a larger structural prefix, Exact the full bounded
/// compact body. Always on a char boundary, never above `cap`.
fn bounded_body(env: &EvidenceEnvelope, level: EvidenceLevel, cap: usize) -> String {
    let body = &env.compact.body;
    let limit = match level {
        EvidenceLevel::Summary => cap.min(512),
        EvidenceLevel::Structure => cap.min(2048),
        EvidenceLevel::Exact => cap,
    };
    truncate_chars(body, limit)
}

fn truncate_chars(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let mut end = max;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_string()
}

/// Lowercased alphanumeric keywords (len >= 3), deduplicated, bounded.
fn keywords(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            current.extend(ch.to_lowercase());
        } else if !current.is_empty() {
            if current.len() >= 3 {
                out.push(std::mem::take(&mut current));
            } else {
                current.clear();
            }
        }
    }
    if current.len() >= 3 {
        out.push(current);
    }
    dedupe(out)
}

/// Path keywords: the basename (with and without extension) plus each path
/// segment of length >= 3, lowercased.
fn path_keywords(path: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let trimmed = path.trim().trim_matches('"');
    let basename = trimmed.rsplit(['/', '\\']).next().unwrap_or(trimmed);
    if !basename.is_empty() {
        out.push(basename.to_ascii_lowercase());
        if let Some((stem, _)) = basename.rsplit_once('.') {
            if stem.len() >= 3 {
                out.push(stem.to_ascii_lowercase());
            }
        }
    }
    for segment in trimmed.split(['/', '\\']) {
        if segment.len() >= 3 {
            out.push(segment.to_ascii_lowercase());
        }
    }
    dedupe(out)
}

/// The longest identifier of a compiler-error failure line, if any: quoted
/// names are unwrapped by the identifier scan and `E####` error codes are
/// filtered out, so `error[E0308]: ... \`compile_unit\`` yields
/// `compile_unit`.
fn extract_symbol(failure: &str) -> Option<String> {
    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();
    for ch in failure.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            current.push(ch);
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
        .into_iter()
        .filter(|token| {
            let lower = token.to_ascii_lowercase();
            let error_code = lower.starts_with('e')
                && lower.len() > 1
                && lower[1..].chars().all(|c| c.is_ascii_digit());
            token.len() >= 3 && !error_code
        })
        .max_by_key(String::len)
}

fn dedupe(values: Vec<String>) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut out: Vec<String> = Vec::new();
    for value in values {
        if value.is_empty() || !seen.insert(value.clone()) {
            continue;
        }
        out.push(value);
        if out.len() >= MAX_NEED_KEYWORDS {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use faktor_evidence::store::{EvidenceStore, MemoryEvidenceStore};
    use faktor_evidence::types::{
        CompactRepresentation, CompressionRecord, ProvenanceSource, RetrievalPolicy,
    };

    fn envelope(
        id: u64,
        session: u64,
        workspace: u64,
        body: &str,
        revision: &str,
        kind: EvidenceKind,
    ) -> EvidenceEnvelope {
        EvidenceEnvelope::new(
            EvidenceId(id),
            kind,
            SessionId::new(session),
            WorkspaceId::new(workspace),
            None,
            Some(revision.to_string()),
            ProvenanceSet::new([ProvenanceSource::Tool]),
            kind.default_compressibility(),
            CompactRepresentation {
                grammar: "test-v1".to_string(),
                body: body.to_string(),
            },
            None,
            faktor_evidence::types::BackingCompleteness::Complete,
            CompressionRecord::identity(body.len() as u64),
            RetrievalPolicy::new(true, true, 4096),
        )
        .unwrap()
    }

    fn criterion(text: &str) -> CriterionFact {
        CriterionFact {
            id: text.to_string(),
            text: text.to_string(),
            requirement: CriterionRequirement::Required,
            origin: CriterionOrigin::User,
            evidence_source: None,
            semantic_snapshot: None,
        }
    }

    fn facts(criteria: &[&str], failures: &[&str]) -> TaskFacts {
        TaskFacts {
            session_id: SessionId::new(1),
            workspace_id: WorkspaceId::new(2),
            task_id: Some(3),
            goal: "implement the change".to_string(),
            active_work_item: None,
            criteria: criteria.iter().map(|c| criterion(c)).collect(),
            failures: failures.iter().map(|f| f.to_string()).collect(),
            verification_state: VerificationState::Unknown,
            owned_paths: Vec::new(),
            changed_files: Vec::new(),
            semantic_snapshot: None,
        }
    }

    fn store_with(envelopes: Vec<EvidenceEnvelope>) -> MemoryEvidenceStore {
        let mut store = MemoryEvidenceStore::new(64 * 1024);
        for envelope in envelopes {
            store.insert(envelope, None).unwrap();
        }
        store
    }

    #[test]
    fn required_needs_keep_one_variant_and_drop_redundant_ones() {
        // A1/A2 both cover required need A; B1 covers REQUIRED need B.
        // Both A variants are exact duplicates of the SAME evidence group?
        // No: A1 and A2 are two PRODUCERS of the same knowledge — the
        // selection must keep exactly one (redundancy dropped) and B1.
        let facts = facts(&["alpha requirement", "beta requirement"], &[]);
        let store = store_with(vec![
            envelope(
                1,
                1,
                2,
                "alpha requirement satisfied by implementation one",
                "src/a1.rs",
                EvidenceKind::SymbolSet,
            ),
            envelope(
                2,
                1,
                2,
                "alpha requirement satisfied by implementation two",
                "src/a2.rs",
                EvidenceKind::SymbolSet,
            ),
            envelope(
                3,
                1,
                2,
                "beta requirement satisfied by implementation",
                "src/b1.rs",
                EvidenceKind::SymbolSet,
            ),
        ]);
        let compiler = ContextCompiler::new(Some(Arc::new(store)), None);
        let compiled = compiler
            .compile(&CompilerInput::new(facts, 100_000))
            .expect("compile");
        let ids: Vec<u64> = compiled.selected.iter().map(|e| e.id.0).collect();
        assert_eq!(ids, vec![1, 3], "A1 and B1 must be selected: {ids:?}");
        assert!(compiled.selected.iter().all(|e| e.required));
        // The two required needs are covered; the redundant producer is gone.
        assert_eq!(compiled.needs.len(), 2);
        assert!(!compiled.is_empty());
    }

    #[test]
    fn no_source_is_neutral_parity() {
        let compiler = ContextCompiler::new(None, None);
        let compiled = compiler
            .compile(&CompilerInput::new(facts(&["must pass"], &[]), 4096))
            .expect("compile");
        assert!(compiled.is_empty());
        assert_eq!(compiled.render, "");
        assert_eq!(compiled.segments, PromptSegments::default());
        // Even with a source, an empty store selects nothing.
        let compiler = ContextCompiler::new(Some(Arc::new(store_with(vec![]))), None);
        let compiled = compiler
            .compile(&CompilerInput::new(facts(&["must pass"], &[]), 4096))
            .expect("compile");
        assert!(compiled.is_empty());
    }

    #[test]
    fn crafted_overflow_replans_and_keeps_required_evidence() {
        // One required envelope whose Exact body alone exceeds a tiny
        // envelope: the compiler must re-run under a corrected envelope and
        // keep the required need covered via a smaller variant — never drop
        // required evidence.
        let big = "need a ".to_string() + &"x".repeat(64 * 1024);
        let store = store_with(vec![envelope(
            1,
            1,
            2,
            &big,
            "src/big.rs",
            EvidenceKind::GenericText,
        )]);
        let compiler = ContextCompiler::new(Some(Arc::new(store)), None);
        let compiled = compiler
            .compile(&CompilerInput::new(facts(&["need A"], &[]), 400))
            .expect("a corrected envelope must fit the required need");
        assert!(!compiled.is_empty(), "required evidence must survive");
        assert!(compiled.retries > 0, "the compiler must have replanned");
        assert!(compiled.selected.iter().any(|e| e.required));
        assert!(
            compiled.total_tokens <= 400,
            "selection must fit the envelope: {}",
            compiled.total_tokens
        );
    }

    #[test]
    fn impossible_envelope_is_a_typed_overflow_never_a_drop() {
        let store = store_with(vec![envelope(
            1,
            1,
            2,
            &("need a ".to_string() + &"y".repeat(4096)),
            "src/big.rs",
            EvidenceKind::GenericText,
        )]);
        let compiler = ContextCompiler::new(Some(Arc::new(store)), None);
        let err = compiler
            .compile(&CompilerInput::new(facts(&["need A"], &[]), 1))
            .expect_err("a 1-token envelope cannot fit required evidence");
        match err {
            CompilerError::EnvelopeOverflow {
                section_costs,
                over_by,
                budget,
                ..
            } => {
                assert!(section_costs.evidence > 0);
                assert!(over_by > 0);
                assert_eq!(budget, 1);
            }
            other => panic!("expected typed EnvelopeOverflow, got {other:?}"),
        }
    }

    #[test]
    fn cross_session_evidence_is_invisible_to_the_compiler() {
        let store = store_with(vec![
            envelope(
                1,
                1,
                2,
                "need a satisfied for session one only",
                "src/a.rs",
                EvidenceKind::SymbolSet,
            ),
            envelope(
                2,
                9,
                2,
                "need a satisfied for session nine",
                "src/a9.rs",
                EvidenceKind::SymbolSet,
            ),
        ]);
        let compiler = ContextCompiler::new(Some(Arc::new(store)), None);
        let compiled = compiler
            .compile(&CompilerInput::new(facts(&["need A"], &[]), 4096))
            .expect("compile");
        let ids: Vec<u64> = compiled.selected.iter().map(|e| e.id.0).collect();
        assert_eq!(ids, vec![1], "session 9 evidence must be invisible");
    }

    #[test]
    fn failed_tests_and_symbols_generate_weighted_needs() {
        let facts = TaskFacts {
            session_id: SessionId::new(1),
            workspace_id: WorkspaceId::new(2),
            task_id: None,
            goal: "fix".into(),
            active_work_item: None,
            criteria: vec![criterion("the parser must accept trailing commas")],
            failures: vec![
                "test parser::trailing_comma ... FAILED".into(),
                "error[E0308]: mismatched types in `compile_unit`".into(),
            ],
            verification_state: VerificationState::Failed,
            owned_paths: vec!["src/parser/mod.rs".into()],
            changed_files: vec!["src/parser/mod.rs".into()],
            semantic_snapshot: None,
        };
        let set = NeedSet::from_facts(&facts);
        assert!(set
            .entries
            .iter()
            .any(|e| e.source == NeedSource::Criterion && e.need.required));
        assert!(set
            .entries
            .iter()
            .any(|e| e.source == NeedSource::FailedTest && e.need.weight >= 0.9));
        assert!(set
            .entries
            .iter()
            .any(|e| e.source == NeedSource::CompileSymbol && e.need.id.contains("compile_unit")));
        assert!(set
            .entries
            .iter()
            .any(|e| e.source == NeedSource::OwnedPath && !e.need.required));
        assert!(set
            .entries
            .iter()
            .any(|e| e.source == NeedSource::RecentChange));
        // Determinism: same facts, same needs (ids and order).
        assert_eq!(set, NeedSet::from_facts(&facts));
    }

    #[test]
    fn hostile_inputs_never_panic_and_never_inflate() {
        // Empty/whitespace criteria generate nothing searchable.
        let empty = facts(&["", "   ", ":::"], &[]);
        assert!(NeedSet::from_facts(&empty).is_empty());
        // A gigantic budget, a hostile symbol, unicode paths: total.
        let hostile = TaskFacts {
            session_id: SessionId::new(1),
            workspace_id: WorkspaceId::new(1),
            task_id: Some(1),
            goal: "x".repeat(10_000),
            active_work_item: Some(WorkItem {
                id: "w".into(),
                title: "t".into(),
                state: "running".into(),
                paths: vec!["../../../etc/passwd".into(), "日本語/ファイル.rs".into()],
                criteria: vec![],
            }),
            criteria: vec![criterion(&"é".repeat(5000))],
            failures: vec!["error[E0999]: \u{0}\u{1} `sym`".into()],
            verification_state: VerificationState::Pending,
            owned_paths: vec!["/".into()],
            changed_files: vec!["\\".into()],
            semantic_snapshot: None,
        };
        let compiler = ContextCompiler::new(Some(Arc::new(store_with(vec![]))), None);
        let first = compiler
            .compile(&CompilerInput::new(hostile.clone(), u32::MAX))
            .expect("compile");
        let second = compiler
            .compile(&CompilerInput::new(hostile, u32::MAX))
            .expect("compile");
        assert_eq!(first, second, "compilation must be deterministic");
    }

    #[test]
    fn learning_prior_can_only_protect_and_never_touches_required() {
        struct RiskAll(f64);
        impl FailurePrior for RiskAll {
            fn omission_risk(&self, _candidate: &ContextCandidate) -> f64 {
                self.0
            }
        }
        let base = facts(&["alpha requirement", "beta requirement"], &[]);
        let store = store_with(vec![
            envelope(
                1,
                1,
                2,
                "alpha requirement one",
                "src/a1.rs",
                EvidenceKind::SymbolSet,
            ),
            envelope(
                2,
                1,
                2,
                "alpha requirement two",
                "src/a2.rs",
                EvidenceKind::SymbolSet,
            ),
            envelope(
                3,
                1,
                2,
                "beta requirement three",
                "src/b1.rs",
                EvidenceKind::SymbolSet,
            ),
        ]);
        // A prior that tries to protect EVERYTHING cannot change required
        // selection (Required candidates are never consulted) and cannot
        // demote A1 below A2's group conflict.
        let compiler = ContextCompiler::new(Some(Arc::new(store)), Some(Arc::new(RiskAll(2.0))));
        let compiled = compiler
            .compile(&CompilerInput::new(base, 100_000))
            .expect("compile");
        let ids: Vec<u64> = compiled.selected.iter().map(|e| e.id.0).collect();
        assert_eq!(
            ids,
            vec![1, 3],
            "priors must not change required selection: {:?} needs={:?}",
            compiled.selected,
            compiled.needs
        );
        // A hostile risk (NaN/inf) is sanitized by the planner, not here.
        let compiler = ContextCompiler::new(None, Some(Arc::new(RiskAll(f64::NAN))));
        assert!(compiler
            .compile(&CompilerInput::new(facts(&["need A"], &[]), 4096))
            .expect("compile")
            .is_empty());
    }

    /// Auditor recency case: 400 envelopes where 390 is the current
    /// criterion's evidence and 400 the latest failed verification's; a
    /// 256-candidate compile must include both.
    #[test]
    fn recency_compile_includes_current_criterion_and_latest_failure() {
        let mut envelopes = Vec::new();
        for id in 1..=400u64 {
            let (body, revision) = match id {
                390 => (
                    "current criterion alpha beta gamma satisfied",
                    "src/current.rs",
                ),
                400 => ("test latest failed parser run observed", "src/failure.rs"),
                _ => ("irrelevant filler row", "src/filler.rs"),
            };
            envelopes.push(envelope(
                id,
                1,
                2,
                body,
                revision,
                EvidenceKind::GenericText,
            ));
        }
        let store = store_with(envelopes);
        let criterion = CriterionFact {
            id: "current-criterion".into(),
            text: "current criterion alpha beta gamma".into(),
            requirement: CriterionRequirement::Required,
            origin: CriterionOrigin::User,
            evidence_source: Some(EvidenceId(390)),
            semantic_snapshot: None,
        };
        let input_facts = TaskFacts {
            criteria: vec![criterion],
            failures: vec!["test latest failed parser run evidence_source=400".into()],
            ..facts(&[], &[])
        };
        let compiler = ContextCompiler::new(Some(Arc::new(store)), None);
        let compiled = compiler
            .compile(
                &CompilerInput::new(input_facts, 1_000_000).with_volatile(VolatileClaims {
                    evidence_tokens: 1_000_000,
                    ..VolatileClaims::default()
                }),
            )
            .expect("compile");
        let ids: Vec<u64> = compiled.selected.iter().map(|e| e.id.0).collect();
        assert!(
            ids.contains(&390),
            "the current criterion's evidence must survive the cap: {ids:?}"
        );
        assert!(
            ids.contains(&400),
            "the latest failed verification's evidence must survive the cap: {ids:?}"
        );
        assert!(compiled.total_tokens <= 1_000_000);
    }

    /// Auditor cap case: a directly referenced required envelope keeps its
    /// place even when 10,000 newer irrelevant rows exist (the newest pages
    /// can never reach it).
    #[test]
    fn direct_required_reference_survives_ten_thousand_later_rows() {
        let mut store = MemoryEvidenceStore::new(1024);
        store
            .insert(
                envelope(
                    390,
                    1,
                    2,
                    "required criterion alpha satisfied",
                    "src/req.rs",
                    EvidenceKind::GenericText,
                ),
                None,
            )
            .unwrap();
        for id in 391..=10_390u64 {
            store
                .insert(
                    envelope(
                        id,
                        1,
                        2,
                        "irrelevant filler row",
                        "src/filler.rs",
                        EvidenceKind::GenericText,
                    ),
                    None,
                )
                .unwrap();
        }
        let referenced = TaskFacts {
            criteria: vec![CriterionFact {
                id: "req".into(),
                text: "required criterion alpha".into(),
                requirement: CriterionRequirement::Required,
                origin: CriterionOrigin::User,
                evidence_source: Some(EvidenceId(390)),
                semantic_snapshot: None,
            }],
            ..facts(&[], &[])
        };
        let compiler = ContextCompiler::new(Some(Arc::new(store)), None);
        let compiled = compiler
            .compile(&CompilerInput::new(referenced.clone(), 1_000_000))
            .expect("compile");
        assert!(
            compiled.selected.iter().any(|item| item.id.0 == 390),
            "a directly referenced required id must be fetched by id: {:?}",
            compiled.selected.iter().map(|e| e.id.0).collect::<Vec<_>>()
        );
        // The reference is what saves it: without the explicit edge the
        // newest pages contain only irrelevant rows and the old evidence is
        // gone.
        let unreferenced = TaskFacts {
            criteria: vec![criterion("required criterion alpha satisfied")],
            ..facts(&[], &[])
        };
        let compiled = compiler
            .compile(&CompilerInput::new(unreferenced, 1_000_000))
            .expect("compile");
        assert!(
            compiled.selected.is_empty(),
            "without a direct reference the capped newest pages cannot reach 390"
        );
    }

    /// Typed criterion edges are collected before token forms; criterion
    /// PROSE tokens are never scanned (only the typed field is an edge), and
    /// hostile tokens never fabricate an id.
    #[test]
    fn referenced_evidence_ids_parses_typed_edges_and_bounded_token_forms() {
        let mut facts = facts(&["criterion evidence://7"], &["failure evidence_source=9"]);
        facts.criteria = vec![
            criterion("criterion evidence://7"),
            CriterionFact {
                id: "typed-edge".into(),
                text: "opaque".into(),
                requirement: CriterionRequirement::Required,
                origin: CriterionOrigin::User,
                evidence_source: Some(EvidenceId(7)),
                semantic_snapshot: None,
            },
            criterion("prose evidence://src/x.rs"),
            criterion("dup evidence_id=7"),
            criterion("foreign evidence_source=99"),
        ];
        let input = CompilerInput::new(facts, 64).with_evidence_refs(vec![EvidenceId(3)]);
        let ids: Vec<u64> = referenced_evidence_ids(&input)
            .iter()
            .map(|id| id.0)
            .collect();
        assert_eq!(
            ids,
            vec![3, 7, 9],
            "explicit refs, typed criterion edges and failure tokens; criterion prose is never scanned"
        );
    }

    /// Swapping two criteria's array positions changes NOTHING: the need ids
    /// are the criteria's stable content ids, the needs are sorted by id, and
    /// candidate coverage is attributed by need id — the whole compiled
    /// context is bit-identical.
    #[test]
    fn criteria_order_swap_is_bit_identical() {
        let alpha = criterion("alpha requirement");
        let beta = criterion("beta requirement");
        let forward = TaskFacts {
            criteria: vec![alpha.clone(), beta.clone()],
            ..facts(&[], &[])
        };
        let swapped = TaskFacts {
            criteria: vec![beta, alpha],
            ..facts(&[], &[])
        };
        let forward_set = NeedSet::from_facts(&forward);
        let swapped_set = NeedSet::from_facts(&swapped);
        assert_eq!(forward_set, swapped_set, "NeedSet is order-invariant");
        let need_ids: Vec<&str> = forward_set
            .entries
            .iter()
            .map(|e| e.need.id.as_str())
            .collect();
        assert!(
            need_ids.iter().all(|id| id.starts_with("criterion:")),
            "need identity is criterion:<stable id>, never an index: {need_ids:?}"
        );
        let store = store_with(vec![
            envelope(
                1,
                1,
                2,
                "alpha requirement satisfied",
                "src/a.rs",
                EvidenceKind::SymbolSet,
            ),
            envelope(
                2,
                1,
                2,
                "beta requirement satisfied",
                "src/b.rs",
                EvidenceKind::SymbolSet,
            ),
        ]);
        let compiler = ContextCompiler::new(Some(Arc::new(store)), None);
        let first = compiler
            .compile(&CompilerInput::new(forward, 100_000))
            .expect("compile");
        let second = compiler
            .compile(&CompilerInput::new(swapped, 100_000))
            .expect("compile");
        assert_eq!(first, second, "the compiled context is bit-identical");
        assert_eq!(
            first.selected.iter().filter(|e| e.required).count(),
            2,
            "both required criteria stay attributed"
        );
    }

    /// A criterion whose text yields NO keywords still retrieves its
    /// evidence when the typed edge names it: the edge is a required
    /// coverage declaration, not a keyword.
    #[test]
    fn opaque_criterion_with_explicit_edge_still_retrieves_by_id() {
        let store = store_with(vec![
            envelope(
                7,
                1,
                2,
                "zzz qqq unrelated body",
                "src/opaque.rs",
                EvidenceKind::GenericText,
            ),
            envelope(
                8,
                1,
                2,
                "xxx yyy newer filler",
                "src/filler.rs",
                EvidenceKind::GenericText,
            ),
        ]);
        let facts = TaskFacts {
            criteria: vec![CriterionFact {
                id: "opaque-criterion".into(),
                text: "***".into(),
                requirement: CriterionRequirement::Required,
                origin: CriterionOrigin::User,
                evidence_source: Some(EvidenceId(7)),
                semantic_snapshot: None,
            }],
            ..facts(&[], &[])
        };
        let set = NeedSet::from_facts(&facts);
        assert_eq!(set.entries.len(), 1, "the edge keeps the need alive");
        let compiler = ContextCompiler::new(Some(Arc::new(store)), None);
        let compiled = compiler
            .compile(&CompilerInput::new(facts, 100_000))
            .expect("compile");
        assert!(
            compiled.selected.iter().any(|e| e.id.0 == 7 && e.required),
            "the explicitly referenced evidence is retrieved by id and required: {:?}",
            compiled.selected.iter().map(|e| e.id.0).collect::<Vec<_>>()
        );
    }

    /// A derived criterion whose semantic snapshot moved is STALE: its
    /// explicit edge is not fetched nor attributed (the criterion must be
    /// re-derived), while a fresh snapshot restores the edge.
    #[test]
    fn stale_semantic_snapshot_marks_the_derived_edge_stale() {
        let edge = |snapshot: Option<&str>, current: Option<&str>| TaskFacts {
            semantic_snapshot: current.map(str::to_string),
            criteria: vec![CriterionFact {
                id: "derived".into(),
                text: "gamma requirement".into(),
                requirement: CriterionRequirement::Required,
                origin: CriterionOrigin::SemanticProvider,
                evidence_source: Some(EvidenceId(42)),
                semantic_snapshot: snapshot.map(str::to_string),
            }],
            ..facts(&[], &[])
        };
        let stale_facts = edge(Some("snap-1"), Some("snap-2"));
        let set = NeedSet::from_facts(&stale_facts);
        assert_eq!(set.entries.len(), 1);
        assert!(set.entries[0].stale, "the snapshot provably moved");
        assert!(
            set.entries[0].evidence_source.is_none(),
            "a stale edge is not an evidence_source"
        );
        assert!(
            referenced_evidence_ids(&CompilerInput::new(stale_facts.clone(), 4096))
                .iter()
                .all(|id| id.0 != 42)
        );
        // The stored envelope does NOT keyword-match gamma: a stale edge can
        // never resurrect it.
        let store = store_with(vec![envelope(
            42,
            1,
            2,
            "zzz unrelated body",
            "src/old.rs",
            EvidenceKind::GenericText,
        )]);
        let compiler = ContextCompiler::new(Some(Arc::new(store)), None);
        let stale = compiler
            .compile(&CompilerInput::new(stale_facts, 100_000))
            .expect("compile");
        assert!(
            stale.selected.iter().all(|e| e.id.0 != 42),
            "stale evidence is never attributed: {:?}",
            stale.selected.iter().map(|e| e.id.0).collect::<Vec<_>>()
        );
        // Same criterion, current snapshot = the criterion's snapshot: the
        // edge is fresh and retrieves.
        let fresh_facts = edge(Some("snap-1"), Some("snap-1"));
        let set = NeedSet::from_facts(&fresh_facts);
        assert!(!set.entries[0].stale);
        assert_eq!(set.entries[0].evidence_source, Some(EvidenceId(42)));
        let fresh = compiler
            .compile(&CompilerInput::new(fresh_facts, 100_000))
            .expect("compile");
        assert!(fresh.selected.iter().any(|e| e.id.0 == 42 && e.required));
    }

    /// The volatile budget is a marginal-information competition, not a
    /// fixed third: repair pressure gives evidence MORE than a third while a
    /// history-heavy turn gives it LESS, and required evidence is
    /// hard-reserved in both.
    #[test]
    fn volatile_budget_adapts_between_repair_and_history_with_required_reserved() {
        let repair_facts = TaskFacts {
            failures: vec!["test repair::case ... FAILED".into()],
            criteria: vec![CriterionFact {
                id: "repair".into(),
                text: "the repair must hold".into(),
                requirement: CriterionRequirement::Required,
                origin: CriterionOrigin::User,
                evidence_source: Some(EvidenceId(1)),
                semantic_snapshot: None,
            }],
            verification_state: VerificationState::Failed,
            ..facts(&[], &[])
        };
        let repair_weights = VolatileWeights::for_facts(&repair_facts);
        assert!(
            repair_weights.evidence_ppm > repair_weights.history_ppm,
            "repair pressure raises evidence's marginal information"
        );
        let claims = VolatileClaims {
            history_tokens: 20_000,
            evidence_tokens: 10_000,
            handoff_tokens: 0,
            tool_notes_tokens: 0,
        };
        let repair = allocate_volatile_budget(
            9_000,
            MAX_VOLATILE_EVIDENCE_TOKENS,
            1_000,
            &claims,
            &repair_weights,
        );
        assert!(
            repair.evidence > 3_000,
            "repair-heavy evidence gets more than a third: {repair:?}"
        );
        assert_eq!(repair.required_evidence, 1_000, "reserved exactly");
        assert!(repair.evidence >= repair.required_evidence);

        let history_weights = VolatileWeights::for_facts(&facts(&[], &[]));
        let history_claims = VolatileClaims {
            history_tokens: 60_000,
            evidence_tokens: 10_000,
            handoff_tokens: 0,
            tool_notes_tokens: 0,
        };
        let history = allocate_volatile_budget(
            9_000,
            MAX_VOLATILE_EVIDENCE_TOKENS,
            1_000,
            &history_claims,
            &history_weights,
        );
        assert!(
            history.evidence < 3_000,
            "history-heavy evidence gets less than a third: {history:?}"
        );
        assert_eq!(history.required_evidence, 1_000, "unchanged reservation");
        assert!(history.evidence >= history.required_evidence);

        // Required evidence alone over the ceiling is never silently
        // dropped: the reservation is the whole ceiling and selection
        // surfaces the typed overflow.
        let tiny = allocate_volatile_budget(
            100,
            MAX_VOLATILE_EVIDENCE_TOKENS,
            1_000,
            &history_claims,
            &history_weights,
        );
        assert_eq!(tiny.evidence, 100);
        assert_eq!(tiny.required_evidence, 100);

        // Handoff/tool-note claims compete too: a dominant tool-note
        // marginal weight wins the leftover over evidence.
        let notes = allocate_volatile_budget(
            9_000,
            MAX_VOLATILE_EVIDENCE_TOKENS,
            0,
            &VolatileClaims {
                history_tokens: 0,
                evidence_tokens: 1_000,
                handoff_tokens: 0,
                tool_notes_tokens: 5_000,
            },
            &VolatileWeights {
                tool_notes_ppm: 2_000_000,
                ..VolatileWeights::default()
            },
        );
        assert!(
            notes.tool_notes > notes.evidence && notes.tool_notes > 0,
            "tool notes compete for the same budget: {notes:?}"
        );
        // The explicit evidence ceiling is a ceiling, never an allocation.
        let capped = allocate_volatile_budget(
            MAX_VOLATILE_TOTAL_TOKENS,
            512,
            0,
            &VolatileClaims {
                history_tokens: 0,
                evidence_tokens: 1_000_000,
                handoff_tokens: 0,
                tool_notes_tokens: 0,
            },
            &history_weights,
        );
        assert_eq!(capped.evidence, 512, "{capped:?}");
    }

    /// The compiler end of the adaptive budget: the SAME facts and store
    /// produce a materially larger evidence selection under repair pressure
    /// than under a history-heavy claim, and the required envelope is
    /// selected in both (hard-reserved, unchanged).
    #[test]
    fn compiler_allocates_evidence_adaptively_from_competing_claims() {
        let mut envelopes = vec![envelope(
            1,
            1,
            2,
            "required criterion alpha satisfied",
            "src/req.rs",
            EvidenceKind::GenericText,
        )];
        let mut changed = Vec::new();
        for i in 0..20 {
            let path = format!("src/file{i:02}.rs");
            envelopes.push(envelope(
                2 + i,
                1,
                2,
                "recent change observed in the working tree",
                &path,
                EvidenceKind::GenericText,
            ));
            changed.push(path);
        }
        let facts = TaskFacts {
            criteria: vec![CriterionFact {
                id: "required-alpha".into(),
                text: "required criterion alpha".into(),
                requirement: CriterionRequirement::Required,
                origin: CriterionOrigin::User,
                evidence_source: None,
                semantic_snapshot: None,
            }],
            changed_files: changed,
            ..facts(&[], &[])
        };
        let compiler = ContextCompiler::new(Some(Arc::new(store_with(envelopes))), None);
        // Repair-heavy: failures raise evidence's marginal information, so
        // nearly every candidate fits.
        let repair_claims = VolatileClaims {
            history_tokens: 0,
            evidence_tokens: 400,
            ..VolatileClaims::default()
        };
        let repair_facts = TaskFacts {
            failures: vec!["test repair::case ... FAILED".into()],
            ..facts.clone()
        };
        let repair = compiler
            .compile(&CompilerInput::new(repair_facts, 200).with_volatile(repair_claims))
            .expect("compile");
        // History-heavy: a huge conversation claim crowds evidence down to
        // its hard-reserved required part.
        let history_claims = VolatileClaims {
            history_tokens: 100_000,
            evidence_tokens: 400,
            ..VolatileClaims::default()
        };
        let history = compiler
            .compile(&CompilerInput::new(facts, 200).with_volatile(history_claims))
            .expect("compile");
        assert!(
            repair.selected.len() > history.selected.len(),
            "repair evidence must win more of the volatile budget: repair={} history={}",
            repair.selected.len(),
            history.selected.len()
        );
        for (label, compiled) in [("repair", &repair), ("history", &history)] {
            assert!(
                compiled.selected.iter().any(|e| e.id.0 == 1 && e.required),
                "{label} must still select the required envelope"
            );
        }
    }

    /// Cursor paging: the newest-first walk never repeats or skips an
    /// envelope, even when a page boundary lands between two ids.
    #[test]
    fn newest_page_walk_is_complete_and_ordered() {
        let envelopes: Vec<EvidenceEnvelope> = (1..=7)
            .map(|id| {
                envelope(
                    id,
                    1,
                    2,
                    "criterion alpha",
                    "src/x.rs",
                    EvidenceKind::GenericText,
                )
            })
            .collect();
        let mut store = MemoryEvidenceStore::new(1024);
        for env in envelopes {
            store.insert(env, None).unwrap();
        }
        let ctx = EvidenceAccessContext::new(1, 2, Some(3));
        let mut seen = Vec::new();
        let mut before = None;
        loop {
            let page = store.list_scoped_newest(&ctx, before, 3);
            if page.rows.is_empty() {
                break;
            }
            seen.extend(page.rows.iter().map(|env| env.id.0));
            before = page.next_before;
        }
        assert_eq!(seen, vec![7, 6, 5, 4, 3, 2, 1]);
    }
}
