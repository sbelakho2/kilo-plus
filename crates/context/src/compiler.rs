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

use faktor_core::{SessionId, WorkspaceId};
use faktor_evidence::store::{MemoryEvidenceStore, StoredEvidence};
use faktor_evidence::types::{EvidenceEnvelope, EvidenceError, ProvenanceSet};

use crate::estimator::Estimator;
use crate::information::{
    select_by_information, select_by_information_with_prior, FailurePrior, InformationBudget,
    InformationError, Need,
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
    pub criteria: Vec<String>,
    pub failures: Vec<String>,
    pub verification_state: VerificationState,
    pub owned_paths: Vec<String>,
    pub changed_files: Vec<String>,
}

impl TaskFacts {
    /// The scope context every retrieval runs under.
    pub fn access(&self) -> EvidenceAccessContext {
        EvidenceAccessContext::new(self.session_id.raw(), self.workspace_id.raw(), self.task_id)
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
/// evidence bodies.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledNeed {
    pub need: Need,
    pub keywords: Vec<String>,
    pub source: NeedSource,
}

/// The needs one turn must satisfy, in generation order (criteria first).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NeedSet {
    pub entries: Vec<CompiledNeed>,
}

impl NeedSet {
    pub fn from_facts(facts: &TaskFacts) -> Self {
        let mut entries: Vec<CompiledNeed> = Vec::new();
        let mut push = |id: String, keywords: Vec<String>, source: NeedSource| {
            if keywords.is_empty() {
                return; // nothing searchable: a need no evidence can match
            }
            entries.push(CompiledNeed {
                need: Need {
                    id,
                    weight: source.weight(),
                    required: source.required(),
                },
                keywords,
                source,
            });
        };
        for (i, criterion) in facts.criteria.iter().enumerate() {
            push(
                format!("criterion:{i}"),
                keywords(criterion),
                NeedSource::Criterion,
            );
        }
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
                ),
                None => push(format!("failure:{i}"), keywords(failure), source),
            }
        }
        for path in &facts.owned_paths {
            push(
                format!("path:{}", path.trim()),
                path_keywords(path),
                NeedSource::OwnedPath,
            );
        }
        for path in &facts.changed_files {
            push(
                format!("recent:{}", path.trim()),
                path_keywords(path),
                NeedSource::RecentChange,
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

/// One compile input: the durable facts, the token envelope the selection
/// runs under, and any caller-supplied supplemental envelopes (producers
/// that could not persist, and unit tests).
#[derive(Debug, Clone)]
pub struct CompilerInput {
    pub facts: TaskFacts,
    pub budget_tokens: u32,
    pub supplemental: Vec<EvidenceEnvelope>,
}

impl CompilerInput {
    pub fn new(facts: TaskFacts, budget_tokens: u32) -> Self {
        Self {
            facts,
            budget_tokens,
            supplemental: Vec::new(),
        }
    }

    pub fn with_supplemental(mut self, envelopes: Vec<EvidenceEnvelope>) -> Self {
        self.supplemental = envelopes;
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

/// Bounds the compiler runs under (bounded everything).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompilerLimits {
    pub max_envelopes: usize,
    pub compact_body_cap: usize,
}

impl Default for CompilerLimits {
    fn default() -> Self {
        Self {
            max_envelopes: MAX_COMPILED_ENVELOPES,
            compact_body_cap: MAX_COMPILED_BODY_BYTES,
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
        if let Some(source) = &self.evidence {
            let ctx = input.facts.access();
            let retrieved = source
                .list_scoped_envelopes(&ctx, self.limits.max_envelopes)
                .map_err(|e| CompilerError::Evidence(e.to_string()))?;
            envelopes.extend(retrieved);
        }
        // Deduplicate by evidence id: supplemental first, then store order.
        let mut seen = std::collections::HashSet::new();
        envelopes.retain(|env| seen.insert(env.id));
        if envelopes.is_empty() || need_set.is_empty() {
            return Ok(CompiledContext::empty(need_set.needs()));
        }
        let needs = need_set.needs();
        let mut last_required = 0u64;
        let mut last_budget = input.budget_tokens;
        for attempt in 0..=MAX_COMPILE_RETRIES {
            let policy = AttemptPolicy::for_attempt(attempt, &self.limits);
            let (candidates, metas) = build_candidates(&envelopes, &need_set, &policy);
            let budget = InformationBudget {
                token_budget: input.budget_tokens,
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
                    required_tokens,
                    token_budget,
                }) => {
                    last_required = required_tokens;
                    last_budget = token_budget;
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
    for (env_idx, env) in envelopes.iter().enumerate() {
        let text = searchable_text(env);
        let mut coverages: Vec<(usize, u64)> = Vec::new();
        for (idx, compiled) in need_set.entries.iter().enumerate() {
            let matched = match_fraction(&text, &compiled.keywords);
            if matched <= 0.0 {
                continue;
            }
            coverages.push((idx, (matched * 1_000_000.0).min(1_000_000.0) as u64));
        }
        if coverages.is_empty() {
            continue;
        }
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

    fn facts(criteria: &[&str], failures: &[&str]) -> TaskFacts {
        TaskFacts {
            session_id: SessionId::new(1),
            workspace_id: WorkspaceId::new(2),
            task_id: Some(3),
            goal: "implement the change".to_string(),
            active_work_item: None,
            criteria: criteria.iter().map(|c| c.to_string()).collect(),
            failures: failures.iter().map(|f| f.to_string()).collect(),
            verification_state: VerificationState::Unknown,
            owned_paths: Vec::new(),
            changed_files: Vec::new(),
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
            criteria: vec!["the parser must accept trailing commas".into()],
            failures: vec![
                "test parser::trailing_comma ... FAILED".into(),
                "error[E0308]: mismatched types in `compile_unit`".into(),
            ],
            verification_state: VerificationState::Failed,
            owned_paths: vec!["src/parser/mod.rs".into()],
            changed_files: vec!["src/parser/mod.rs".into()],
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
            criteria: vec!["é".repeat(5000)],
            failures: vec!["error[E0999]: \u{0}\u{1} `sym`".into()],
            verification_state: VerificationState::Pending,
            owned_paths: vec!["/".into()],
            changed_files: vec!["\\".into()],
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
}
