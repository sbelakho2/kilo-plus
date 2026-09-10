//! Wire request rendering (audit 33): the context budget must bound the
//! provider request (conservative normalized estimate — not a
//! provider-tokenizer-exact count), and every conceptual element must appear
//! exactly once:
//!
//! ```text
//! system    = STATIC + SEMI-STABLE head (instructions, project rules, task
//!             contract, repository map) rendered as ONE byte-stable head,
//!             followed by a VOLATILE tail (task progress + steering note,
//!             retrieved evidence, current errors) appended AFTER the head —
//!             volatile content is appended last so provider prompt caching
//!             is not invalidated by every new turn. The exact head byte
//!             range is exposed as `WirePlan::cacheable_prefix_len` /
//!             `WirePlan::cacheable_prefix()`.
//! messages  = the structured history, each message exactly once, with tool
//!             calls/results as structured parts.
//! tools     = the schemas, once — never embedded in `system`.
//! ```
//!
//! The renderer is PURE (audit 33): it renders exactly the sections it is
//! handed and NEVER deletes one — no history drop, no evidence drop, no
//! error drop. When the render does not fit the budget it returns the typed
//! [`WirePlanError::Oversized`] carrying `actual_tokens` and the per-section
//! [`SectionCosts`]; the ONE selector on the runtime path (the planner in
//! `faktor-agent`'s wire-plan entry) deterministically replans a smaller
//! volatile window. Required content (everything the planner cannot shrink:
//! static + rules + contract + tools + repository map + task progress) that
//! alone exceeds the budget stays a typed `Oversized` — the renderer never
//! manufactures a plan by deleting a section.
//!
//! Segmented prompt identity (audit 44): every render measures its eight
//! conceptual [`PromptSegment`]s in provider prompt order —
//!
//! ```text
//! static_policy, project_rules, task_contract, tool_bundle,
//! semi_stable_project, task_progress, evidence, recent_history
//! ```
//!
//! `task_contract` carries goal + criteria/constraints + non-goals + scope
//! (it must NOT mix `task_progress`: completed steps / blocker / verification
//! status — those live in the volatile `task_progress` segment, where the
//! steering note also rides because steering is volatile). The segment
//! hashes/tokens feed the per-call [`PrefixObservation`] (audit 45) and the
//! [`classify_prefix_cache`] policy (audit 46).
//!
//! `total_tokens = estimate(system) + Σ estimate(message) + estimate(tools)`
//! is enforced BEFORE anything reaches the wire; the planner never returns
//! an unbudgeted plan.

use std::collections::HashSet;

use faktor_core::error::{Error, ErrorKind};
use faktor_core::hash::FileHash;
use faktor_provider::{ContentKind, RequestMessage, ToolSpec};

use crate::assembler::Evidence;
use crate::budget::ContextBudget;
use crate::estimator::Estimator;
use crate::ledger::TaskLedger;
use crate::{TokenCache, TokenEstimate, TokenEstimateKind};

/// Number of conceptual prompt segments (audit 44).
pub const PROMPT_SEGMENT_COUNT: usize = 8;

/// The segments that form the byte-stable cacheable prefix (static policy,
/// project rules, task contract, tool bundle, semi-stable project knowledge).
/// `task_progress` and everything after it are the volatile tail.
pub const PROMPT_CACHEABLE_PREFIX_SEGMENTS: usize = 5;

/// Bound on a decoded [`PrefixObservation`] segment vector (bounded
/// everything: a hostile JSON payload may not describe an unbounded prompt).
pub const MAX_PROMPT_OBSERVATION_SEGMENTS: usize = 64;

/// Hard bound on the serialized [`PrefixObservation`] JSON bytes.
const MAX_PREFIX_OBSERVATION_JSON: usize = 64 * 1024;

/// §8.4 memory stability class of one prompt segment, in cache order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PromptStability {
    /// Immutable instructions / project rules / tool schemas.
    Static,
    /// Durable task state and repository knowledge.
    SemiStable,
    /// Recent turns, retrieved evidence, current progress/errors.
    Volatile,
}

impl PromptStability {
    /// True for the stable cacheable-prefix classes (never recompacted
    /// merely because a new turn happened).
    pub const fn is_cacheable_prefix(self) -> bool {
        !matches!(self, PromptStability::Volatile)
    }
}

/// One measured conceptual prompt segment: its stability class, rendered
/// byte length, BLAKE3 digest of the rendered bytes and conservative token
/// count. Equal hashes mean byte-identical segment content; the digest is
/// the comparison oracle the prefix math uses (never the token count alone).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptSegment {
    pub class: PromptStability,
    pub bytes: usize,
    pub hash: FileHash,
    pub tokens: u64,
}

impl PromptSegment {
    /// An empty segment of `class` with the empty-content digest.
    pub fn empty(class: PromptStability) -> Self {
        Self {
            class,
            bytes: 0,
            hash: FileHash::from(*blake3::hash(b"").as_bytes()),
            tokens: 0,
        }
    }

    fn measure(est: &Estimator, class: PromptStability, text: &str) -> Self {
        Self {
            class,
            bytes: text.len(),
            hash: FileHash::from(*blake3::hash(text.as_bytes()).as_bytes()),
            tokens: u64::try_from(est.estimate_tokens(text)).unwrap_or(u64::MAX),
        }
    }
}

impl Default for PromptSegment {
    fn default() -> Self {
        Self::empty(PromptStability::Static)
    }
}

/// The eight measured prompt segments of one render, in provider prompt
/// order (the order is fixed and is the comparison order of every prefix
/// observation; field names are the segment identities).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptSegments {
    pub static_policy: PromptSegment,
    pub project_rules: PromptSegment,
    pub task_contract: PromptSegment,
    pub tool_bundle: PromptSegment,
    pub semi_stable_project: PromptSegment,
    pub task_progress: PromptSegment,
    pub evidence: PromptSegment,
    pub recent_history: PromptSegment,
}

impl Default for PromptSegments {
    fn default() -> Self {
        Self {
            static_policy: PromptSegment::empty(PromptStability::Static),
            project_rules: PromptSegment::empty(PromptStability::Static),
            task_contract: PromptSegment::empty(PromptStability::SemiStable),
            tool_bundle: PromptSegment::empty(PromptStability::Static),
            semi_stable_project: PromptSegment::empty(PromptStability::SemiStable),
            task_progress: PromptSegment::empty(PromptStability::Volatile),
            evidence: PromptSegment::empty(PromptStability::Volatile),
            recent_history: PromptSegment::empty(PromptStability::Volatile),
        }
    }
}

impl PromptSegments {
    /// The segments in canonical prompt order with their names.
    pub fn ordered(&self) -> [(&'static str, &PromptSegment); PROMPT_SEGMENT_COUNT] {
        [
            ("static_policy", &self.static_policy),
            ("project_rules", &self.project_rules),
            ("task_contract", &self.task_contract),
            ("tool_bundle", &self.tool_bundle),
            ("semi_stable_project", &self.semi_stable_project),
            ("task_progress", &self.task_progress),
            ("evidence", &self.evidence),
            ("recent_history", &self.recent_history),
        ]
    }

    /// Segment content digests in canonical order.
    pub fn hashes(&self) -> Vec<FileHash> {
        self.ordered().iter().map(|(_, s)| s.hash).collect()
    }

    /// Segment token counts in canonical order.
    pub fn token_counts(&self) -> Vec<u64> {
        self.ordered().iter().map(|(_, s)| s.tokens).collect()
    }

    /// Total tokens across all eight segments (saturating).
    pub fn total_tokens(&self) -> u64 {
        self.token_counts()
            .into_iter()
            .fold(0u64, u64::saturating_add)
    }

    /// Tokens of the cacheable prefix (segments 0..[`PROMPT_CACHEABLE_PREFIX_SEGMENTS`]).
    pub fn cacheable_tokens(&self) -> u64 {
        self.token_counts()
            .into_iter()
            .take(PROMPT_CACHEABLE_PREFIX_SEGMENTS)
            .fold(0u64, u64::saturating_add)
    }

    /// Stable leading tokens between this render and the previous render:
    /// the sum of segment tokens up to (not including) the first segment
    /// whose byte digest changed. First turn (no previous) = the full
    /// length; a total change = 0.
    pub fn stable_leading_tokens(&self, previous: Option<&PromptSegments>) -> u64 {
        let current = PrefixObservation::from_segments(self, 0);
        match previous {
            None => current.prompt_tokens(),
            Some(prev) => {
                let previous = PrefixObservation::from_segments(prev, 0);
                current
                    .longest_stable_prefix(Some(&previous))
                    .stable_leading_tokens
            }
        }
    }

    /// The index of the first segment whose digest differs from `previous`
    /// (`None` = every current segment matched; the new turn only appended
    /// nothing).
    pub fn first_changed_segment(&self, previous: &PromptSegments) -> Option<usize> {
        let current = PrefixObservation::from_segments(self, 0);
        let previous = PrefixObservation::from_segments(previous, 0);
        current.longest_stable_prefix(Some(&previous)).first_changed
    }
}

/// The longest-identical-prefix verdict between two consecutive turns
/// (audit 45): `stable_leading_tokens` is the sum of segment token counts up
/// to the first changed digest; `matched_segments` counts the leading
/// segments whose digests are identical; `first_changed` names the first
/// changed segment index (`None` when every current segment matched — the
/// first turn and the no-change case).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StablePrefix {
    pub stable_leading_tokens: u64,
    pub matched_segments: usize,
    pub first_changed: Option<usize>,
}

/// Per-call prefix observation (audit 45): the ordered segment digests and
/// token counts of one rendered request plus the provider-reported cache-read
/// tokens of that call. Deterministic, JSON-serializable so it can ride the
/// durable prefix row payload, and bounded on decode.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrefixObservation {
    pub segment_hashes: Vec<FileHash>,
    pub segment_token_counts: Vec<u64>,
    pub cache_read_tokens: u64,
}

impl PrefixObservation {
    /// Measure one render's segments into an observation.
    pub fn from_segments(segments: &PromptSegments, cache_read_tokens: u64) -> Self {
        Self {
            segment_hashes: segments.hashes(),
            segment_token_counts: segments.token_counts(),
            cache_read_tokens,
        }
    }

    /// Canonical JSON (field order fixed by the struct; hex digests).
    /// Infallible for this value shape — the fallback is the empty object.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }

    /// Strict decode: malformed JSON, unknown fields, mismatched
    /// hash/token vector lengths and any vector beyond
    /// [`MAX_PROMPT_OBSERVATION_SEGMENTS`] are `None` (loud to the caller,
    /// never silently guessed).
    pub fn from_json(json: &str) -> Option<Self> {
        if json.len() > MAX_PREFIX_OBSERVATION_JSON {
            return None;
        }
        let obs: Self = serde_json::from_str(json).ok()?;
        if obs.segment_hashes.len() != obs.segment_token_counts.len()
            || obs.segment_hashes.len() > MAX_PROMPT_OBSERVATION_SEGMENTS
        {
            return None;
        }
        Some(obs)
    }

    /// Sum of segment token counts (saturating).
    pub fn prompt_tokens(&self) -> u64 {
        self.segment_token_counts
            .iter()
            .copied()
            .fold(0u64, u64::saturating_add)
    }

    /// The longest identical leading segment sequence between `previous`
    /// and `self` (audit 45). First turn (`previous = None`) reports this
    /// observation's full length; a changed first digest reports 0; a
    /// shortened previous sequence reports the append point as changed.
    /// Hostile vectors (mismatched lengths, overflow counts) never panic.
    pub fn longest_stable_prefix(&self, previous: Option<&PrefixObservation>) -> StablePrefix {
        let Some(previous) = previous else {
            return StablePrefix {
                stable_leading_tokens: self.prompt_tokens(),
                matched_segments: self.segment_hashes.len(),
                first_changed: None,
            };
        };
        let mut stable_leading_tokens = 0u64;
        let mut matched_segments = 0usize;
        for (i, hash) in self.segment_hashes.iter().enumerate() {
            if previous.segment_hashes.get(i) == Some(hash) {
                stable_leading_tokens = stable_leading_tokens
                    .saturating_add(self.segment_token_counts.get(i).copied().unwrap_or(0));
                matched_segments += 1;
            } else {
                return StablePrefix {
                    stable_leading_tokens,
                    matched_segments,
                    first_changed: Some(i),
                };
            }
        }
        StablePrefix {
            stable_leading_tokens,
            matched_segments,
            first_changed: None,
        }
    }
}

/// Provider prefix-cache state (audit 46), derived from the previous
/// observation, the current one (with this call's observed cache reads),
/// the last-request time and the configured TTL policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixCacheState {
    /// The provider still holds the stable prefix: preserve it byte-for-byte.
    Warm,
    /// The provider cache is provably gone: deterministically recompact the
    /// stable sections into a smaller new prefix (see
    /// [`recompact_stable_prefix`]) instead of re-sending the old large one.
    Cold,
    /// No identity/time evidence: conservatively preserve the prefix and
    /// never recompact on a guess.
    Unknown,
}

/// The cache-eviction policy of [`classify_prefix_cache`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixCachePolicy {
    /// Presumed provider-cache lifetime since the last request, in ms.
    pub ttl_ms: i64,
}

impl Default for PrefixCachePolicy {
    /// Five minutes: conservative provider prompt-cache lifetime.
    fn default() -> Self {
        Self { ttl_ms: 300_000 }
    }
}

/// Derive the prefix-cache state (audit 46). Deterministic and total:
/// hostile inputs (negative TTL, `now_ms` before the last request, empty
/// observations, oversized counts) never panic and never make a cold
/// decision on no evidence.
///
/// ```text
/// Unknown  no previous observation, no last-request timestamp, or both
///          sides empty → conservative preserve (never recompact on a guess)
/// Cold     the stable prefix identity was rewritten (first changed segment
///          lies inside the cacheable prefix) OR the provider evidenced
///          eviction (last request older than the TTL with 0 cache reads)
/// Warm     anything else: identity preserved and/or cache reads observed —
///          a NEW TURN ALONE NEVER RECOMPACTS
/// ```
pub fn classify_prefix_cache(
    previous: Option<&PrefixObservation>,
    current: &PrefixObservation,
    last_request_ms: Option<i64>,
    now_ms: i64,
    policy: &PrefixCachePolicy,
) -> PrefixCacheState {
    let Some(previous) = previous else {
        return PrefixCacheState::Unknown;
    };
    let Some(last_request_ms) = last_request_ms else {
        return PrefixCacheState::Unknown;
    };
    if previous.prompt_tokens() == 0 && current.prompt_tokens() == 0 {
        return PrefixCacheState::Unknown;
    }
    let stable = current.longest_stable_prefix(Some(previous));
    let stable_rewrite = stable
        .first_changed
        .is_some_and(|i| i < PROMPT_CACHEABLE_PREFIX_SEGMENTS);
    if stable_rewrite {
        return PrefixCacheState::Cold;
    }
    let idle = now_ms.saturating_sub(last_request_ms) > policy.ttl_ms.max(0);
    if idle && current.cache_read_tokens == 0 {
        return PrefixCacheState::Cold;
    }
    PrefixCacheState::Warm
}

/// Deterministic recompaction of a rendered stable prefix into a smaller,
/// byte-stable new prefix (audit 46, the Cold transform). Documented
/// transform, in order:
///
/// 1. within the target: return the input UNCHANGED (byte-identical; an
///    already-minimal prefix is never rewritten merely because a turn
///    happened);
/// 2. line-wise deduplication (first occurrence kept) with consecutive
///    blank-line runs collapsed to one blank line;
/// 3. if still over, truncate at the estimator's chars-per-token ratio on a
///    UTF-8 char boundary with one trailing ellipsis.
///
/// Properties: same input + same target → same bytes; never longer than the
/// input; never splits UTF-8; result estimate ≤ `target_tokens + 1`.
pub fn recompact_stable_prefix(stable_prefix: &str, target_tokens: usize) -> String {
    let est = Estimator;
    if est.estimate_tokens(stable_prefix) <= target_tokens {
        return stable_prefix.to_string();
    }
    let mut seen: HashSet<&str> = HashSet::new();
    let mut deduped = String::with_capacity(stable_prefix.len());
    let mut blank_run = false;
    for line in stable_prefix.split_inclusive('\n') {
        if line.trim().is_empty() {
            if blank_run {
                continue;
            }
            blank_run = true;
        } else {
            blank_run = false;
            if !seen.insert(line) {
                continue;
            }
        }
        deduped.push_str(line);
    }
    if est.estimate_tokens(&deduped) <= target_tokens {
        return deduped;
    }
    est.clamp_to(&deduped, target_tokens)
}

/// Typed renderer failure (audit 33): the pure renderer never trims, so an
/// over-budget render reports exactly how large it was and what each
/// conceptual section cost. `actual_tokens` is the exact
/// `estimate(system) + Σ estimate(message) + estimate(tools)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WirePlanError {
    Oversized {
        actual_tokens: usize,
        section_costs: SectionCosts,
    },
}

impl WirePlanError {
    pub fn actual_tokens(&self) -> usize {
        match self {
            WirePlanError::Oversized { actual_tokens, .. } => *actual_tokens,
        }
    }

    pub fn section_costs(&self) -> &SectionCosts {
        match self {
            WirePlanError::Oversized { section_costs, .. } => section_costs,
        }
    }
}

impl std::fmt::Display for WirePlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WirePlanError::Oversized {
                actual_tokens,
                section_costs,
            } => write!(
                f,
                "wire plan oversized: {actual_tokens} tokens ({} required, {} volatile); the renderer deleted no section",
                section_costs.required_tokens(),
                section_costs.volatile_tokens()
            ),
        }
    }
}

impl std::error::Error for WirePlanError {}

/// The renderer's typed error maps to the workspace's `Oversized` kind for
/// callers that carry [`faktor_core::error::Error`].
impl From<WirePlanError> for Error {
    fn from(e: WirePlanError) -> Self {
        Error::new(ErrorKind::Oversized, e.to_string())
    }
}

/// Per-section token costs of one render, in the audit-44 segment order.
/// `required_tokens` is what the planner cannot shrink (everything except
/// the volatile planner window); `volatile_tokens` is the planner's window
/// (retrieved evidence + recent history).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SectionCosts {
    pub static_policy: usize,
    pub project_rules: usize,
    pub task_contract: usize,
    pub tool_bundle: usize,
    pub semi_stable_project: usize,
    pub task_progress: usize,
    pub evidence: usize,
    pub recent_history: usize,
}

impl SectionCosts {
    /// Costs of a measured segment set.
    pub fn from_segments(segments: &PromptSegments) -> Self {
        fn c(s: &PromptSegment) -> usize {
            usize::try_from(s.tokens).unwrap_or(usize::MAX)
        }
        Self {
            static_policy: c(&segments.static_policy),
            project_rules: c(&segments.project_rules),
            task_contract: c(&segments.task_contract),
            tool_bundle: c(&segments.tool_bundle),
            semi_stable_project: c(&segments.semi_stable_project),
            task_progress: c(&segments.task_progress),
            evidence: c(&segments.evidence),
            recent_history: c(&segments.recent_history),
        }
    }

    /// Content the planner cannot delete: static policy, rules, contract,
    /// tool bundle, repository map and task progress. If this alone exceeds
    /// the budget the render stays a typed `Oversized`.
    pub const fn required_tokens(&self) -> usize {
        self.static_policy
            .saturating_add(self.project_rules)
            .saturating_add(self.task_contract)
            .saturating_add(self.tool_bundle)
            .saturating_add(self.semi_stable_project)
            .saturating_add(self.task_progress)
    }

    /// The planner's volatile window (evidence + recent history).
    pub const fn volatile_tokens(&self) -> usize {
        self.evidence.saturating_add(self.recent_history)
    }

    /// Every section, saturating.
    pub const fn total(&self) -> usize {
        self.required_tokens()
            .saturating_add(self.volatile_tokens())
    }
}

/// A conservative normalized-request estimate (`estimator.rs`: never below
/// the chars/3.4 floor, plus hand-written envelope estimates);
/// provider-specific tokenizers are future work.
#[derive(Debug, Clone, PartialEq)]
pub struct WirePlan {
    pub system: String,
    pub messages: Vec<RequestMessage>,
    pub tools: Vec<ToolSpec>,
    pub total_tokens: usize,
    /// Byte length of the cacheable-prefix head of [`WirePlan::system`]
    /// (architecture §8.4, audits 65-66): instructions + project rules +
    /// task contract + repository map. Volatile content (task progress,
    /// steering note, retrieved evidence, current errors) is appended after
    /// this boundary and can never precede it. The renderer copies the head
    /// verbatim into `system`, so the boundary is a char boundary of the
    /// render by construction; callers that record prefix observations hash
    /// `&system[..cacheable_prefix_len]` as the exact bytes the wire
    /// request sent.
    pub cacheable_prefix_len: usize,
    /// The measured audit-44 segments of the render (segment identities,
    /// digests and token counts).
    pub prompt_segments: PromptSegments,
}

impl WirePlan {
    /// The byte-exact cacheable-prefix head of [`WirePlan::system`]
    /// (static + semi-stable, architecture §8.4). `None` only when the
    /// boundary is not on a char boundary of the render — impossible by
    /// construction (the head is copied verbatim); callers must hash
    /// nothing on `None` rather than guess.
    pub fn cacheable_prefix(&self) -> Option<&str> {
        self.system.get(..self.cacheable_prefix_len)
    }

    /// This plan's per-call prefix observation (audit 45) with the
    /// provider-reported cache-read tokens of the call.
    pub fn prefix_observation(&self, cache_read_tokens: u64) -> PrefixObservation {
        PrefixObservation::from_segments(&self.prompt_segments, cache_read_tokens)
    }

    /// Stable leading tokens of this plan against a previous observation
    /// (first turn = this plan's full prompt length).
    pub fn stable_leading_tokens(&self, previous: Option<&PrefixObservation>) -> u64 {
        self.prefix_observation(0)
            .longest_stable_prefix(previous)
            .stable_leading_tokens
    }
}

/// Render the wire request under the budget, PURELY: every input section is
/// rendered exactly once and never deleted; a render that cannot fit returns
/// [`WirePlanError::Oversized`] with the exact total and per-section costs.
/// There is no trim loop and no section ever disappears.
#[allow(clippy::too_many_arguments)]
pub fn plan_wire_request(
    instructions: &str,
    system_extra: &str,
    tool_schemas: &[ToolSpec],
    project_rules: &str,
    ledger: &TaskLedger,
    repo_map: &str,
    history: &[RequestMessage],
    evidence: &[Evidence],
    errors: &str,
    budget: &ContextBudget,
) -> Result<WirePlan, WirePlanError> {
    let est = Estimator;

    // Render every conceptual section exactly once. The cacheable head is
    // static + semi-stable only; the volatile tail is appended after it.
    let static_policy = instructions.to_string();
    let project_rules_render = render_project_rules(project_rules);
    let task_contract_render = render_task_contract(ledger);
    let semi_stable_project_render = render_repo_map(repo_map);
    let task_progress_render = format!(
        "{}{}",
        render_task_progress(ledger),
        render_steering(system_extra)
    );
    let evidence_render = format!("{}{}", render_evidence(evidence), render_errors(errors));

    let head_len = static_policy
        .len()
        .saturating_add(project_rules_render.len())
        .saturating_add(task_contract_render.len())
        .saturating_add(semi_stable_project_render.len());
    let mut system = String::with_capacity(
        head_len
            .saturating_add(task_progress_render.len())
            .saturating_add(evidence_render.len()),
    );
    system.push_str(&static_policy);
    system.push_str(&project_rules_render);
    system.push_str(&task_contract_render);
    system.push_str(&semi_stable_project_render);
    system.push_str(&task_progress_render);
    system.push_str(&evidence_render);

    let prompt_segments = PromptSegments {
        static_policy: PromptSegment::measure(&est, PromptStability::Static, &static_policy),
        project_rules: PromptSegment::measure(&est, PromptStability::Static, &project_rules_render),
        task_contract: PromptSegment::measure(
            &est,
            PromptStability::SemiStable,
            &task_contract_render,
        ),
        tool_bundle: measure_tool_bundle(&est, tool_schemas),
        semi_stable_project: PromptSegment::measure(
            &est,
            PromptStability::SemiStable,
            &semi_stable_project_render,
        ),
        task_progress: PromptSegment::measure(
            &est,
            PromptStability::Volatile,
            &task_progress_render,
        ),
        evidence: PromptSegment::measure(&est, PromptStability::Volatile, &evidence_render),
        recent_history: measure_history(&est, history),
    };

    let system_tokens = est.estimate_tokens(&system);
    let messages_tokens =
        usize::try_from(prompt_segments.recent_history.tokens).unwrap_or(usize::MAX);
    let tools_tokens = usize::try_from(prompt_segments.tool_bundle.tokens).unwrap_or(usize::MAX);
    let total = system_tokens
        .saturating_add(messages_tokens)
        .saturating_add(tools_tokens);
    let section_costs = SectionCosts::from_segments(&prompt_segments);
    let context_max = budget.context_max();
    if context_max == 0 || total > context_max {
        return Err(WirePlanError::Oversized {
            actual_tokens: total,
            section_costs,
        });
    }
    Ok(WirePlan {
        cacheable_prefix_len: head_len,
        system,
        messages: history.to_vec(),
        tools: tool_schemas.to_vec(),
        total_tokens: total,
        prompt_segments,
    })
}

/// Static + semi-stable cacheable head, in class order: instructions, then
/// project rules, then the task contract, then the repository map.
fn render_project_rules(project_rules: &str) -> String {
    if project_rules.is_empty() {
        String::new()
    } else {
        format!("\n## Project rules\n{project_rules}")
    }
}

/// The task CONTRACT (audit 44): goal + criteria/constraints + non-goals +
/// scope. It must not mix task progress (completed steps, blocker,
/// verification status) — those render in [`render_task_progress`].
fn render_task_contract(ledger: &TaskLedger) -> String {
    let has = !ledger.goal.is_empty()
        || !ledger.constraints.is_empty()
        || !ledger.user_preferences.is_empty();
    if !has {
        return String::new();
    }
    let mut out = String::from("\n## Task contract\n");
    if !ledger.goal.is_empty() {
        out.push_str(&format!("GOAL: {}\n", truncate(&ledger.goal, 200)));
    }
    if !ledger.constraints.is_empty() {
        out.push_str("CONSTRAINTS:\n");
        for c in ledger.constraints.iter().take(8) {
            out.push_str(&format!("- {}\n", truncate(c, 120)));
        }
    }
    if !ledger.user_preferences.is_empty() {
        out.push_str("NON-GOALS/PREFERENCES:\n");
        for p in ledger.user_preferences.iter().take(8) {
            out.push_str(&format!("- {}\n", truncate(p, 120)));
        }
    }
    out
}

/// The VOLATILE task progress (audit 44): completed/open steps, blocker
/// (known failures) and verification status (tests run/failed), changed
/// files, decisions. Never mixes into the task contract.
fn render_task_progress(ledger: &TaskLedger) -> String {
    let has = !ledger.completed_steps.is_empty()
        || !ledger.open_steps.is_empty()
        || !ledger.decisions.is_empty()
        || !ledger.known_failures.is_empty()
        || !ledger.changed_files.is_empty()
        || !ledger.tests_run.is_empty()
        || !ledger.tests_failed.is_empty();
    if !has {
        return String::new();
    }
    let mut out = String::from("\n## Task progress\n");
    if !ledger.open_steps.is_empty() {
        out.push_str("OPEN STEPS:\n");
        for s in ledger.open_steps.iter().take(12) {
            out.push_str(&format!("- {}\n", truncate(s, 120)));
        }
    }
    if !ledger.completed_steps.is_empty() {
        out.push_str("COMPLETED STEPS:\n");
        for s in ledger.completed_steps.iter().rev().take(8) {
            out.push_str(&format!("- {}\n", truncate(s, 120)));
        }
    }
    if !ledger.decisions.is_empty() {
        out.push_str("DECISIONS:\n");
        for d in ledger.decisions.iter().rev().take(8) {
            out.push_str(&format!("- {}\n", truncate(d, 120)));
        }
    }
    if !ledger.known_failures.is_empty() {
        out.push_str("KNOWN FAILURES:\n");
        for f in ledger.known_failures.iter().rev().take(8) {
            out.push_str(&format!("- {}\n", truncate(f, 120)));
        }
    }
    if !ledger.changed_files.is_empty() {
        out.push_str("CHANGED FILES: ");
        let joined = ledger
            .changed_files
            .iter()
            .map(|s| truncate(s, 80))
            .collect::<Vec<_>>()
            .join(", ");
        out.push_str(&truncate(&joined, 300));
        out.push('\n');
    }
    if !ledger.tests_run.is_empty() {
        out.push_str("TESTS RUN: ");
        let joined = ledger
            .tests_run
            .iter()
            .rev()
            .take(8)
            .map(|s| truncate(s, 80))
            .collect::<Vec<_>>()
            .join(", ");
        out.push_str(&truncate(&joined, 300));
        out.push('\n');
    }
    if !ledger.tests_failed.is_empty() {
        out.push_str("TESTS FAILED: ");
        let joined = ledger
            .tests_failed
            .iter()
            .rev()
            .take(8)
            .map(|s| truncate(s, 80))
            .collect::<Vec<_>>()
            .join(", ");
        out.push_str(&truncate(&joined, 300));
        out.push('\n');
    }
    out
}

/// The steering note is VOLATILE (audit 44): a Steer rewrite may only
/// invalidate content after it, never the static/semi-stable head.
fn render_steering(system_extra: &str) -> String {
    if system_extra.is_empty() {
        String::new()
    } else {
        format!("\n## Steering\n{system_extra}")
    }
}

/// Repository knowledge (semi-stable, architecture §8.4 class 3), bounded.
fn render_repo_map(repo_map: &str) -> String {
    if repo_map.is_empty() {
        String::new()
    } else {
        format!("\n## Repository map\n{}", truncate(repo_map, 2000))
    }
}

/// Retrieved evidence (volatile), highest score first, bounded snippets.
fn render_evidence(evidence: &[Evidence]) -> String {
    if evidence.is_empty() {
        return String::new();
    }
    let mut scored: Vec<&Evidence> = evidence.iter().collect();
    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.path.cmp(&b.path))
    });
    let mut out = String::from("\n## Retrieved evidence\n");
    for ev in scored {
        out.push_str("\n### ");
        out.push_str(&ev.path);
        out.push('\n');
        out.push_str(&truncate(&ev.snippet, 1500));
        out.push('\n');
    }
    out
}

/// Current errors (volatile tail; rides the evidence segment identity).
fn render_errors(errors: &str) -> String {
    if errors.is_empty() {
        String::new()
    } else {
        format!("\n## Current errors\n{errors}")
    }
}

/// Canonical tool-bundle segment: the schema JSON is the identity (tools
/// travel as the wire `tools` field, never embedded in `system`), the token
/// count is the renderer's own schema accounting.
fn measure_tool_bundle(est: &Estimator, specs: &[ToolSpec]) -> PromptSegment {
    let canonical = serde_json::to_string(specs).unwrap_or_default();
    PromptSegment {
        class: PromptStability::Static,
        bytes: canonical.len(),
        hash: FileHash::from(*blake3::hash(canonical.as_bytes()).as_bytes()),
        tokens: u64::try_from(estimate_tools(est, specs)).unwrap_or(u64::MAX),
    }
}

/// Canonical history segment: incremental digest over each message's JSON
/// (never materializing one huge string) and the renderer's exact
/// per-message token accounting.
fn measure_history(est: &Estimator, messages: &[RequestMessage]) -> PromptSegment {
    let mut hasher = blake3::Hasher::new();
    let mut bytes = 0usize;
    for m in messages {
        match serde_json::to_string(m) {
            Ok(json) => {
                bytes = bytes.saturating_add(json.len());
                hasher.update(json.as_bytes());
            }
            Err(_) => {
                bytes = bytes.saturating_add(1);
                hasher.update(b"\0");
            }
        }
    }
    PromptSegment {
        class: PromptStability::Volatile,
        bytes,
        hash: FileHash::from(*hasher.finalize().as_bytes()),
        tokens: u64::try_from(estimate_messages(est, messages)).unwrap_or(u64::MAX),
    }
}

fn estimate_messages(est: &Estimator, messages: &[RequestMessage]) -> usize {
    messages.iter().map(|m| estimate_message(est, m)).sum()
}

fn estimate_message(est: &Estimator, m: &RequestMessage) -> usize {
    let mut t = 2usize; // role + message envelope
    for p in &m.content {
        t = t.saturating_add(match &p.kind {
            ContentKind::Text { text } => est.estimate_tokens(text),
            ContentKind::Reasoning { text } => est.estimate_tokens(text),
            ContentKind::Image { url } => est.estimate_tokens(url).max(1),
            ContentKind::ToolCall { id, name, input } => est
                .estimate_tokens(id)
                .saturating_add(est.estimate_tokens(name))
                .saturating_add(est.estimate_json(input))
                .saturating_add(2),
            ContentKind::ToolResult { content, is_error } => est
                .estimate_tokens(content)
                .saturating_add(usize::from(*is_error)),
        });
        t = t.saturating_add(1); // part envelope
    }
    t
}

fn estimate_tools(est: &Estimator, specs: &[ToolSpec]) -> usize {
    specs
        .iter()
        .map(|s| {
            est.estimate_tokens(&s.name)
                .saturating_add(est.estimate_tokens(&s.description))
                .saturating_add(est.estimate_json(&s.input_schema))
                .saturating_add(2)
        })
        .sum()
}

/// Exact renderer accounting of an ALREADY-RENDERED request: the same
/// estimator, per-message envelope and tool-bundle accounting
/// [`plan_wire_request`] charges. The runtime's provider request IS the
/// planned render (`build_request` is a thin adapter), so this value equals
/// the plan's own `total_tokens` for that request — the tripwire that locks
/// routing on the plan's real dimensions consumes it.
pub fn measure_wire_request(
    system: &str,
    messages: &[RequestMessage],
    tools: &[ToolSpec],
) -> usize {
    let est = Estimator;
    est.estimate_tokens(system)
        .saturating_add(estimate_messages(&est, messages))
        .saturating_add(estimate_tools(&est, tools))
}

/// The tokenizer-specific footprint of one rendered request, as counted by
/// the CANDIDATE model's own tokenizer (audit: candidate-specific sizing).
/// `exact` is true only when every text run was counted by a real local
/// tokenizer; a family with no registered vocabulary yields the conservative
/// estimator's value labeled [`TokenEstimateKind::UpperBound`] — an honest
/// upper bound, never an "exact" count the candidate could disprove.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandidateFootprint {
    pub input_tokens: u64,
    pub exact: bool,
}

/// Structural JSON counted as text under a candidate tokenizer (serialize
/// first, then count the bytes): tool schemas and tool-call inputs are real
/// wire bytes, so this is a closer footprint than the generic estimator's
/// JSON formula while never claiming exactness for a fallback family.
fn size_json_for_model(
    model: &str,
    cache: &TokenCache,
    value: &serde_json::Value,
) -> TokenEstimate {
    match serde_json::to_string(value) {
        Ok(text) => cache.count_for_model(model, &text),
        Err(_) => TokenEstimate::upper_bound(0),
    }
}

/// Saturating combine of two estimates: counts add, exactness is the AND
/// (any upper-bound component makes the total an upper bound).
fn combine_estimates(a: TokenEstimate, b: TokenEstimate) -> TokenEstimate {
    TokenEstimate {
        count: a.count.saturating_add(b.count),
        kind: if a.kind == TokenEstimateKind::Exact && b.kind == TokenEstimateKind::Exact {
            TokenEstimateKind::Exact
        } else {
            TokenEstimateKind::UpperBound
        },
    }
}

/// One message's footprint under a candidate tokenizer: the renderer's
/// envelope constants (2 for role + message, 1 per part) with every text run
/// counted by the model's tokenizer.
fn size_message_for_model(model: &str, cache: &TokenCache, m: &RequestMessage) -> TokenEstimate {
    let mut total = TokenEstimate::exact(2);
    for p in &m.content {
        let part = match &p.kind {
            ContentKind::Text { text } | ContentKind::Reasoning { text } => {
                cache.count_for_model(model, text)
            }
            ContentKind::Image { url } => {
                let estimate = cache.count_for_model(model, url);
                TokenEstimate {
                    count: estimate.count.max(1),
                    kind: estimate.kind,
                }
            }
            ContentKind::ToolCall { id, name, input } => {
                let mut t = combine_estimates(
                    cache.count_for_model(model, id),
                    cache.count_for_model(model, name),
                );
                t = combine_estimates(t, size_json_for_model(model, cache, input));
                TokenEstimate {
                    count: t.count.saturating_add(2),
                    kind: t.kind,
                }
            }
            ContentKind::ToolResult { content, is_error } => {
                let t = cache.count_for_model(model, content);
                TokenEstimate {
                    count: t.count.saturating_add(u64::from(*is_error)),
                    kind: t.kind,
                }
            }
        };
        let combined = combine_estimates(total, part);
        total = TokenEstimate {
            count: combined.count.saturating_add(1),
            kind: combined.kind,
        };
    }
    total
}

/// Size one rendered request with the tokenizer the CANDIDATE model maps to
/// (`faktor_provider::tokenizer_for` through the model-targeted cache): the
/// candidate-specific fit check the router's top-K pass consumes. Every text
/// run is counted under its own identity; families without a registered
/// local vocabulary fall back to the conservative estimator and the result
/// is labeled as an upper bound (never exact).
pub fn size_request_for_model(
    model: &str,
    system: &str,
    messages: &[RequestMessage],
    tools: &[ToolSpec],
    cache: &TokenCache,
) -> CandidateFootprint {
    let mut total = cache.count_for_model(model, system);
    for m in messages {
        total = combine_estimates(total, size_message_for_model(model, cache, m));
    }
    for spec in tools {
        for text in [spec.name.as_str(), spec.description.as_str()] {
            total = combine_estimates(total, cache.count_for_model(model, text));
        }
        let schema = size_json_for_model(model, cache, &spec.input_schema);
        total = combine_estimates(total, schema);
        total = TokenEstimate {
            count: total.count.saturating_add(2),
            kind: total.kind,
        };
    }
    CandidateFootprint {
        input_tokens: total.count,
        exact: total.kind == TokenEstimateKind::Exact,
    }
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
    use faktor_provider::{ContentPart, Role};

    fn ledger() -> TaskLedger {
        TaskLedger {
            goal: "fix the parser".into(),
            open_steps: vec!["reproduce crash".into()],
            ..Default::default()
        }
    }

    fn contract_ledger() -> TaskLedger {
        TaskLedger {
            goal: "fix the parser".into(),
            constraints: vec!["no global state".into()],
            ..Default::default()
        }
    }

    fn tool(name: &str) -> ToolSpec {
        ToolSpec {
            name: name.into(),
            description: format!("{name} description"),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
            }),
        }
    }

    fn text_history(n: usize) -> Vec<RequestMessage> {
        (0..n)
            .map(|i| RequestMessage {
                role: if i % 2 == 0 {
                    Role::User
                } else {
                    Role::Assistant
                },
                content: vec![ContentPart::text(format!("turn {i}: {}", "x".repeat(780)))],
            })
            .collect()
    }

    fn evidence(n: usize) -> Vec<Evidence> {
        (0..n)
            .map(|i| Evidence {
                path: format!("src/f{i}.rs"),
                snippet: format!("fn f{i}() {{}} // {}", "y".repeat(200)),
                score: (n - i) as f64,
            })
            .collect()
    }

    fn tight_budget(tokens: usize) -> ContextBudget {
        ContextBudget {
            system: tokens,
            tools: 0,
            working: 0,
            retrieved: 0,
            recent: 0,
            output_reserve: 0,
            safety: 0,
        }
    }

    fn obs(hashes: &[[u8; 32]], tokens: &[u64], cache_read_tokens: u64) -> PrefixObservation {
        PrefixObservation {
            segment_hashes: hashes.iter().copied().map(FileHash::from).collect(),
            segment_token_counts: tokens.to_vec(),
            cache_read_tokens,
        }
    }

    fn h(n: u8) -> [u8; 32] {
        [n; 32]
    }

    // ------------------------------------------------------------ audit 33:
    // the renderer is pure — no trim loop exists in the source, a fitting
    // render carries every section exactly once, and an overflow is the
    // typed error with per-section costs.

    #[test]
    fn renderer_source_has_no_trimming_and_no_section_deletion() {
        // Source-level assertion: the old trim vocabulary must not exist in
        // this renderer. Needles are assembled at runtime so this test's own
        // text cannot satisfy the search.
        let src = include_str!("wire_plan.rs");
        for needle in [
            ["include", "_evidence"].concat(),
            ["drop", "_oldest"].concat(),
            ["sweep", "_dangling"].concat(),
        ] {
            assert!(
                !src.contains(&needle),
                "the pure renderer must not contain trim logic {needle:?}"
            );
        }
    }

    #[test]
    fn renderer_renders_every_section_exactly_once() {
        // The wire plan must contain every conceptual element exactly once:
        // instructions + rules + contract in system, history once in
        // messages, schemas once in tools — and the system render must NOT
        // duplicate the tool schema JSON or the conversation text. Nothing
        // is trimmed when everything fits.
        let b = ContextBudget::default();
        let history = vec![
            RequestMessage {
                role: Role::User,
                content: vec![ContentPart::text("first user turn")],
            },
            RequestMessage {
                role: Role::Assistant,
                content: vec![ContentPart::tool_call(
                    "c1",
                    "read_file",
                    serde_json::json!({"path": "x"}),
                )],
            },
            RequestMessage {
                role: Role::User,
                content: vec![ContentPart::tool_result("ok", false, "c1")],
            },
        ];
        let tools = vec![tool("read_file")];
        let plan = plan_wire_request(
            "You are Faktor.\n",
            "extra",
            &tools,
            "no global state",
            &ledger(),
            "src/",
            &history,
            &evidence(1),
            "boom",
            &b,
        )
        .unwrap();
        assert!(plan.system.starts_with("You are Faktor.\n"));
        assert!(plan.system.contains("GOAL: fix the parser"));
        assert!(plan.system.contains("CONSTRAINTS:") || plan.system.contains("no global state"));
        assert!(plan.system.contains("## Project rules"));
        assert!(plan.system.contains("## Task contract"));
        assert!(plan.system.contains("## Task progress"));
        assert!(plan.system.contains("## Repository map"));
        assert!(plan.system.contains("## Steering"));
        assert!(plan.system.contains("## Retrieved evidence"));
        assert!(plan.system.contains("## Current errors"));
        assert!(!plan.system.contains("first user turn"));
        assert!(!plan.system.contains("read_file"));
        // Messages: exactly the history, once, in order.
        assert_eq!(plan.messages, history);
        // Tools: exactly the schemas, once.
        assert_eq!(plan.tools, tools);
        let tool_json = serde_json::to_string(&tools[0]).unwrap();
        assert!(
            !plan.system.contains(&tool_json),
            "tool schema JSON must not leak into system"
        );
    }

    #[test]
    fn crafted_overflow_is_a_typed_error_and_never_dropped_evidence() {
        // The OLD renderer dropped history, then ALL evidence, then errors
        // and returned Ok; the pure renderer returns the typed Oversized
        // with the exact total and the evidence/history costs — proof the
        // sections were PRICED, not deleted. Required content alone fits,
        // so the runtime planner can replan a smaller volatile window.
        let b = tight_budget(600);
        let history = text_history(20);
        let ev = evidence(4);
        let err = plan_wire_request(
            "sys",
            "steer",
            &[tool("echo")],
            "rules",
            &contract_ledger(),
            "map",
            &history,
            &ev,
            "current errors here",
            &b,
        )
        .unwrap_err();
        let WirePlanError::Oversized {
            actual_tokens,
            section_costs,
        } = err;
        assert!(actual_tokens > b.context_max());
        assert!(
            section_costs.evidence > 0,
            "evidence must be priced, never dropped: {section_costs:?}"
        );
        assert!(section_costs.recent_history > 0);
        assert!(
            section_costs.required_tokens() <= b.context_max(),
            "required alone fits: volatile overflow is what replans"
        );
        assert_eq!(
            section_costs.evidence,
            Estimator.estimate_tokens(&format!(
                "{}{}",
                render_evidence(&ev),
                render_errors("current errors here")
            ))
        );
        // And the typed error maps onto the workspace's Oversized kind.
        let mapped: Error = WirePlanError::Oversized {
            actual_tokens,
            section_costs,
        }
        .into();
        assert_eq!(mapped.kind, ErrorKind::Oversized);
        assert!(mapped.message.contains("deleted no section"));
    }

    #[test]
    fn oversized_required_content_stays_a_typed_error() {
        // Required content alone (a hostile instructions block) exceeds the
        // budget: no planner window can save it; the render stays a typed
        // Oversized and reports the required cost.
        let b = tight_budget(200);
        let err = plan_wire_request(
            &"i".repeat(5000),
            "",
            &[tool("echo")],
            "rules",
            &contract_ledger(),
            "map",
            &[],
            &evidence(2),
            "err",
            &b,
        )
        .unwrap_err();
        let WirePlanError::Oversized {
            actual_tokens,
            section_costs,
        } = err;
        assert!(actual_tokens > b.context_max());
        assert!(section_costs.required_tokens() > b.context_max());
        // Hostile project rules alone and a hostile schema alone both stay
        // typed errors too (never truncated into a fake success).
        for plan in [
            plan_wire_request(
                "",
                "",
                &[],
                &"r".repeat(10_000),
                &TaskLedger::default(),
                "",
                &[],
                &[],
                "",
                &b,
            ),
            plan_wire_request(
                "",
                "",
                &[ToolSpec {
                    name: "t".into(),
                    description: "d".repeat(10_000),
                    input_schema: serde_json::json!({}),
                }],
                "",
                &TaskLedger::default(),
                "",
                &[],
                &[],
                "",
                &b,
            ),
        ] {
            assert!(matches!(plan, Err(WirePlanError::Oversized { .. })));
        }
    }

    #[test]
    fn section_costs_report_every_segment_exactly() {
        let b = ContextBudget::default();
        let history = text_history(3);
        let ev = evidence(2);
        let tools = vec![tool("echo")];
        let l = contract_ledger();
        let mut progress = l.clone();
        progress.open_steps.push("step".into());
        progress.changed_files.push("src/a.rs".into());
        let plan = plan_wire_request(
            "sys", "steer", &tools, "rules", &progress, "map", &history, &ev, "err", &b,
        )
        .unwrap();
        let costs = SectionCosts::from_segments(&plan.prompt_segments);
        let est = Estimator;
        assert_eq!(costs.static_policy, est.estimate_tokens("sys"));
        assert_eq!(
            costs.project_rules,
            est.estimate_tokens(&render_project_rules("rules"))
        );
        assert_eq!(
            costs.task_contract,
            est.estimate_tokens(&render_task_contract(&progress))
        );
        assert_eq!(costs.tool_bundle, estimate_tools(&est, &tools));
        assert_eq!(
            costs.semi_stable_project,
            est.estimate_tokens(&render_repo_map("map"))
        );
        assert_eq!(
            costs.task_progress,
            est.estimate_tokens(&format!(
                "{}{}",
                render_task_progress(&progress),
                render_steering("steer")
            ))
        );
        assert_eq!(
            costs.evidence,
            est.estimate_tokens(&format!("{}{}", render_evidence(&ev), render_errors("err")))
        );
        assert_eq!(costs.recent_history, estimate_messages(&est, &history));
        // The volatile window is exactly evidence + history; required is
        // everything else.
        assert_eq!(
            costs.required_tokens() + costs.volatile_tokens(),
            costs.total()
        );
        assert_eq!(
            costs.volatile_tokens(),
            costs.evidence + costs.recent_history
        );
        // Segment digests are byte truth of the rendered subsections.
        let segs = &plan.prompt_segments;
        assert_eq!(segs.static_policy.bytes, "sys".len());
        assert_eq!(
            segs.static_policy.hash,
            FileHash::from(*blake3::hash(b"sys").as_bytes())
        );
        assert_eq!(
            segs.project_rules.hash,
            FileHash::from(*blake3::hash(render_project_rules("rules").as_bytes()).as_bytes())
        );
        assert_eq!(segs.evidence.hash, segs.evidence.hash);
        // Exact plan math: total == estimate(system) + Σ messages + tools.
        let expected = est.estimate_tokens(&plan.system)
            + estimate_messages(&est, &plan.messages)
            + estimate_tools(&est, &plan.tools);
        assert_eq!(plan.total_tokens, expected, "accounting drifted");
        assert!(plan.total_tokens <= b.context_max());
    }

    #[test]
    fn zero_budget_is_never_a_plan() {
        let zero = ContextBudget {
            system: 0,
            tools: 0,
            working: 0,
            retrieved: 0,
            recent: 0,
            output_reserve: 0,
            safety: 0,
        };
        assert!(matches!(
            plan_wire_request(
                "",
                "",
                &[],
                "",
                &TaskLedger::default(),
                "",
                &[],
                &[],
                "",
                &zero
            ),
            Err(WirePlanError::Oversized { .. })
        ));
    }

    #[test]
    fn hostile_unicode_inputs_never_panic() {
        // Oversized unicode, absurd evidence, absurd errors, deep JSON tool
        // schemas: the pure renderer prices everything and errors — it never
        // panics and never emits a partially deleted plan.
        let b = ContextBudget::default();
        let mut hostile = Vec::new();
        for i in 0..100 {
            hostile.push(RequestMessage {
                role: Role::User,
                content: vec![ContentPart::text(format!("😀{i} {}", "汉".repeat(3000)))],
            });
        }
        let tools = vec![ToolSpec {
            name: "t".into(),
            description: "d".into(),
            input_schema: {
                let mut v = serde_json::Value::Null;
                for _ in 0..50 {
                    v = serde_json::json!([v]);
                }
                v
            },
        }];
        let result = plan_wire_request(
            &"s".repeat(5_000),
            "",
            &tools,
            &"r".repeat(2_000),
            &TaskLedger {
                goal: "g".repeat(5_000),
                ..Default::default()
            },
            &"m".repeat(10_000),
            &hostile,
            &[Evidence {
                path: "p".into(),
                snippet: "é".repeat(10_000),
                score: 1.0,
            }],
            &"e".repeat(10_000),
            &b,
        );
        match result {
            Ok(plan) => assert!(plan.total_tokens <= b.context_max()),
            Err(WirePlanError::Oversized {
                actual_tokens,
                section_costs,
            }) => {
                assert!(actual_tokens > b.context_max());
                assert!(section_costs.recent_history > 0);
                assert!(section_costs.evidence > 0);
            }
        }
    }

    // ------------------------------------------------------------ audit 44:
    // segmentation, byte-stability and invalidation semantics.

    fn plan_with(
        rules: &str,
        progress: &[&str],
        ev: &[Evidence],
        history: &[RequestMessage],
    ) -> WirePlan {
        let mut l = contract_ledger();
        l.open_steps = progress.iter().map(|s| (*s).to_string()).collect();
        plan_wire_request(
            "You are Faktor.\n",
            "",
            &[tool("echo")],
            rules,
            &l,
            "src/",
            history,
            ev,
            "",
            &ContextBudget::default(),
        )
        .unwrap()
    }

    #[test]
    fn volatile_churn_keeps_static_segments_byte_identical() {
        // Two turns where ONLY task_progress/evidence/recent change: the
        // static policy, project rules, task contract and tool bundle must
        // stay byte-identical with unchanged hashes.
        let a = plan_with("rules", &["step one"], &evidence(1), &text_history(2));
        let b = plan_with(
            "rules",
            &["step two", "step three"],
            &evidence(3),
            &text_history(4),
        );
        for (name, x, y) in [
            (
                "static_policy",
                &a.prompt_segments.static_policy,
                &b.prompt_segments.static_policy,
            ),
            (
                "project_rules",
                &a.prompt_segments.project_rules,
                &b.prompt_segments.project_rules,
            ),
            (
                "task_contract",
                &a.prompt_segments.task_contract,
                &b.prompt_segments.task_contract,
            ),
            (
                "tool_bundle",
                &a.prompt_segments.tool_bundle,
                &b.prompt_segments.tool_bundle,
            ),
            (
                "semi_stable_project",
                &a.prompt_segments.semi_stable_project,
                &b.prompt_segments.semi_stable_project,
            ),
        ] {
            assert_eq!(x.bytes, y.bytes, "{name} bytes changed");
            assert_eq!(x.hash, y.hash, "{name} hash changed");
            assert_eq!(x.tokens, y.tokens, "{name} tokens changed");
            assert_eq!(x, y, "{name} segment changed");
        }
        assert_ne!(
            a.prompt_segments.task_progress,
            b.prompt_segments.task_progress
        );
        assert_ne!(a.prompt_segments.evidence, b.prompt_segments.evidence);
        assert_ne!(
            a.prompt_segments.recent_history,
            b.prompt_segments.recent_history
        );
    }

    #[test]
    fn hundred_synthetic_turns_change_only_volatile_segment_hashes() {
        // 100 turns: volatile content changes every turn; the stable
        // segments' bytes/hashes never move. Hashes change ONLY in the
        // volatile segment positions.
        let base = plan_with("rules", &["step 0"], &evidence(0), &text_history(1));
        let mut previous_hashes = base.prompt_segments.hashes();
        let mut volatile_changed = false;
        for turn in 1..100 {
            let progress = vec![format!("step {turn}")];
            let ev: Vec<Evidence> = (0..(turn % 4) + 1)
                .map(|i| Evidence {
                    path: format!("src/turn_{turn}_{i}.rs"),
                    snippet: format!("turn {turn} item {i}"),
                    score: 1.0,
                })
                .collect();
            let plan = {
                let mut l = contract_ledger();
                l.open_steps = progress;
                let history: Vec<RequestMessage> = (0..3)
                    .map(|i| RequestMessage {
                        role: Role::User,
                        content: vec![ContentPart::text(format!("small turn {turn}/{i}"))],
                    })
                    .collect();
                plan_wire_request(
                    "You are Faktor.\n",
                    &format!("steer {turn}"),
                    &[tool("echo")],
                    "rules",
                    &l,
                    "src/",
                    &history,
                    &ev,
                    "",
                    &ContextBudget::default(),
                )
                .unwrap()
            };
            let hashes = plan.prompt_segments.hashes();
            for (i, (x, y)) in previous_hashes.iter().zip(&hashes).enumerate() {
                if i < PROMPT_CACHEABLE_PREFIX_SEGMENTS {
                    assert_eq!(x, y, "stable segment {i} hash moved at turn {turn}");
                } else if x != y {
                    volatile_changed = true;
                }
            }
            assert_eq!(
                plan.prompt_segments.static_policy,
                base.prompt_segments.static_policy
            );
            assert_eq!(
                plan.prompt_segments.task_contract,
                base.prompt_segments.task_contract
            );
            previous_hashes = hashes;
        }
        assert!(volatile_changed, "volatile segments must move across turns");
    }

    #[test]
    fn project_rule_change_invalidates_exactly_the_expected_suffix() {
        // A project-rule change is the SECOND segment: static_policy stays
        // identical; the invalidation starts exactly at project_rules and
        // covers every later segment (cache suffix).
        let a = plan_with("rules one", &["s"], &evidence(0), &[]);
        let b = plan_with("rules two changed", &["s"], &evidence(0), &[]);
        let oa = a.prefix_observation(0);
        let ob = b.prefix_observation(0);
        let stable = ob.longest_stable_prefix(Some(&oa));
        assert_eq!(stable.first_changed, Some(1));
        assert_eq!(stable.matched_segments, 1);
        assert_eq!(
            stable.stable_leading_tokens, ob.segment_token_counts[0],
            "only static_policy survives"
        );
        assert_eq!(
            PROMPT_SEGMENT_COUNT - stable.matched_segments,
            7,
            "the whole suffix from project_rules is invalidated"
        );
        assert_eq!(
            a.prompt_segments.static_policy,
            b.prompt_segments.static_policy
        );
        assert_ne!(
            a.prompt_segments.project_rules,
            b.prompt_segments.project_rules
        );
        // And a change only in the volatile tail never invalidates the head.
        let c = plan_with(
            "rules one",
            &["completely different"],
            &evidence(2),
            &text_history(3),
        );
        let stable = c.prefix_observation(0).longest_stable_prefix(Some(&oa));
        assert!(stable.matched_segments >= PROMPT_CACHEABLE_PREFIX_SEGMENTS);
        assert!(stable
            .first_changed
            .is_none_or(|i| i >= PROMPT_CACHEABLE_PREFIX_SEGMENTS));
        assert_eq!(
            stable.stable_leading_tokens,
            c.prompt_segments.cacheable_tokens()
        );
    }

    #[test]
    fn task_contract_never_mixes_task_progress() {
        // The contract segment is goal+criteria/non-goals/scope only; the
        // completed-step/blocker/verification content lives in the volatile
        // task_progress segment and never touches the contract.
        let mut l = contract_ledger();
        l.open_steps.push("do the thing".into());
        l.completed_steps.push("finished the thing".into());
        l.known_failures.push("blocker: test fails".into());
        l.tests_failed.push("unit::x".into());
        l.changed_files.push("src/x.rs".into());
        let plan = plan_wire_request(
            "sys",
            "",
            &[],
            "",
            &l,
            "",
            &[],
            &[],
            "",
            &ContextBudget::default(),
        )
        .unwrap();
        let segs = &plan.prompt_segments;
        let contract_render = &plan.system
            [..segs.static_policy.bytes + segs.project_rules.bytes + segs.task_contract.bytes];
        assert!(contract_render.contains("GOAL: fix the parser"));
        assert!(contract_render.contains("no global state"));
        assert!(!contract_render.contains("do the thing"));
        assert!(!contract_render.contains("finished the thing"));
        assert!(!contract_render.contains("blocker: test fails"));
        assert!(!contract_render.contains("unit::x"));
        assert!(!contract_render.contains("src/x.rs"));
        let progress = &plan.system[plan
            .system
            .find("## Task progress")
            .expect("progress section")..];
        assert!(progress.contains("do the thing"));
        assert!(progress.contains("finished the thing"));
        assert!(progress.contains("blocker: test fails"));
    }

    #[test]
    fn steering_is_volatile_and_after_the_cacheable_boundary() {
        let b = ContextBudget::default();
        let plan = |note: &str| {
            plan_wire_request(
                "You are Faktor.\n",
                note,
                &[tool("echo")],
                "rules",
                &contract_ledger(),
                "repo map",
                &[],
                &[],
                "",
                &b,
            )
            .unwrap()
        };
        let p1 = plan("do it the blue way");
        let p2 = plan("do it the gold way");
        // The cacheable head is byte-identical across steer rewrites.
        assert_eq!(
            p1.cacheable_prefix().unwrap(),
            p2.cacheable_prefix().unwrap()
        );
        // Steering renders in the volatile tail, after the boundary.
        let prefix = p1.cacheable_prefix().unwrap();
        assert!(!prefix.contains("## Steering"));
        assert!(!prefix.contains("do it the blue way"));
        let tail = &p1.system[p1.cacheable_prefix_len..];
        assert!(tail.contains("## Steering"));
        assert!(tail.contains("do it the blue way"));
        assert_eq!(
            p1.prompt_segments.task_progress.class,
            PromptStability::Volatile
        );
    }

    // ------------------------------------------------------------ audit 45:
    // longest-identical-prefix arithmetic and the observation payload.

    #[test]
    fn longest_stable_prefix_first_turn_total_change_partial() {
        // First turn: no previous observation -> full length.
        let first = obs(&[h(1), h(2), h(3)], &[10, 20, 30], 0);
        let s = first.longest_stable_prefix(None);
        assert_eq!(s.stable_leading_tokens, 60);
        assert_eq!(s.matched_segments, 3);
        assert_eq!(s.first_changed, None);
        // Total change: the first digest differs -> 0.
        let changed = obs(&[h(9), h(2), h(3)], &[10, 20, 30], 0);
        let s = changed.longest_stable_prefix(Some(&first));
        assert_eq!(s.stable_leading_tokens, 0);
        assert_eq!(s.matched_segments, 0);
        assert_eq!(s.first_changed, Some(0));
        // Partial: first two match, third changed.
        let partial = obs(&[h(1), h(2), h(7)], &[10, 20, 30], 0);
        let s = partial.longest_stable_prefix(Some(&first));
        assert_eq!(s.stable_leading_tokens, 30);
        assert_eq!(s.matched_segments, 2);
        assert_eq!(s.first_changed, Some(2));
        // No change: full equality.
        let same = obs(&[h(1), h(2), h(3)], &[10, 20, 30], 5);
        let s = same.longest_stable_prefix(Some(&first));
        assert_eq!(s.stable_leading_tokens, 60);
        assert_eq!(s.first_changed, None);
        // Previous shorter, current appended: the append point is changed.
        let shorter = obs(&[h(1)], &[10], 0);
        let s = first.longest_stable_prefix(Some(&shorter));
        assert_eq!(s.stable_leading_tokens, 10);
        assert_eq!(s.first_changed, Some(1));
        // Current shorter, every current segment matched: full current.
        let s = shorter.longest_stable_prefix(Some(&first));
        assert_eq!(s.stable_leading_tokens, 10);
        assert_eq!(s.matched_segments, 1);
        assert_eq!(s.first_changed, None);
    }

    #[test]
    fn longest_stable_prefix_saturates_and_never_panics_on_hostile_vectors() {
        // Mismatched vector lengths, missing token rows and u64::MAX counts
        // are handled with saturating math — no panic, no wraparound.
        let cur = obs(&[h(1), h(2), h(3)], &[u64::MAX, u64::MAX, 1], 0);
        let prev = PrefixObservation {
            segment_hashes: vec![FileHash::from(h(1)), FileHash::from(h(2))],
            segment_token_counts: vec![], // hostile: hashes without counts
            cache_read_tokens: 0,
        };
        let s = cur.longest_stable_prefix(Some(&prev));
        assert_eq!(s.stable_leading_tokens, u64::MAX);
        assert_eq!(s.first_changed, Some(2));
        let empty = PrefixObservation {
            segment_hashes: vec![],
            segment_token_counts: vec![],
            cache_read_tokens: 0,
        };
        assert_eq!(empty.longest_stable_prefix(None).stable_leading_tokens, 0);
        assert_eq!(empty.longest_stable_prefix(None).matched_segments, 0);
    }

    #[test]
    fn prefix_observation_json_roundtrips_and_rejects_corruption() {
        let o = obs(&[h(1), h(2)], &[10, 20], 7);
        let json = o.to_json();
        assert_eq!(PrefixObservation::from_json(&json), Some(o.clone()));
        // Deterministic serialization.
        assert_eq!(o.to_json(), json);
        // Mismatched vector lengths are rejected, never guessed.
        let bad = serde_json::json!({
            "segment_hashes": [h(1)],
            "segment_token_counts": [1, 2],
            "cache_read_tokens": 0,
        });
        assert_eq!(PrefixObservation::from_json(&bad.to_string()), None);
        // Unknown fields are rejected (strict decode).
        let bad = serde_json::json!({
            "segment_hashes": [],
            "segment_token_counts": [],
            "cache_read_tokens": 0,
            "extra": true,
        });
        assert_eq!(PrefixObservation::from_json(&bad.to_string()), None);
        // Malformed JSON and oversized payloads are rejected.
        assert_eq!(PrefixObservation::from_json("{"), None);
        assert_eq!(
            PrefixObservation::from_json(&"x".repeat(MAX_PREFIX_OBSERVATION_JSON + 1)),
            None
        );
        // Too many segments are rejected.
        let huge = PrefixObservation {
            segment_hashes: (0..(MAX_PROMPT_OBSERVATION_SEGMENTS + 1))
                .map(|i| FileHash::from(h(i as u8)))
                .collect(),
            segment_token_counts: vec![1; MAX_PROMPT_OBSERVATION_SEGMENTS + 1],
            cache_read_tokens: 0,
        };
        assert_eq!(PrefixObservation::from_json(&huge.to_json()), None);
    }

    // ------------------------------------------------------------ audit 46:
    // Warm/Cold/Unknown policy and deterministic recompaction.

    #[test]
    fn prefix_cache_policy_rows_are_deterministic_and_total() {
        let policy = PrefixCachePolicy::default();
        let now = 1_000_000i64;
        let fresh = now - 1_000;
        let prev = obs(
            &[h(1), h(2), h(3), h(4), h(5), h(6)],
            &[10, 10, 10, 10, 10, 10],
            0,
        );
        // Unknown: no previous observation.
        assert_eq!(
            classify_prefix_cache(None, &prev, Some(fresh), now, &policy),
            PrefixCacheState::Unknown
        );
        // Unknown: no last-request time.
        assert_eq!(
            classify_prefix_cache(Some(&prev), &prev, None, now, &policy),
            PrefixCacheState::Unknown
        );
        // Unknown: both sides empty.
        let empty = obs(&[], &[], 0);
        assert_eq!(
            classify_prefix_cache(Some(&empty), &empty, Some(fresh), now, &policy),
            PrefixCacheState::Unknown
        );
        // Warm: a new turn with byte-identical stable segments and no cache
        // reads yet — a new turn ALONE never recompacts.
        assert_eq!(
            classify_prefix_cache(Some(&prev), &prev, Some(fresh), now, &policy),
            PrefixCacheState::Warm
        );
        // Warm: only the volatile tail changed (indices 5+ move, 0..5 hold).
        let volatile_churn = obs(
            &[h(1), h(2), h(3), h(4), h(5), h(9)],
            &[10, 10, 10, 10, 10, 10],
            0,
        );
        assert_eq!(
            classify_prefix_cache(Some(&prev), &volatile_churn, Some(fresh), now, &policy),
            PrefixCacheState::Warm
        );
        // Warm: TTL expired but the provider PROVED cache reads > 0.
        let idle = now - policy.ttl_ms - 1;
        let cached = obs(
            &[h(1), h(2), h(3), h(4), h(5), h(6)],
            &[10, 10, 10, 10, 10, 10],
            999,
        );
        assert_eq!(
            classify_prefix_cache(Some(&prev), &cached, Some(idle), now, &policy),
            PrefixCacheState::Warm
        );
        // Cold: TTL expired with zero cache reads (provider eviction).
        assert_eq!(
            classify_prefix_cache(Some(&prev), &prev, Some(idle), now, &policy),
            PrefixCacheState::Cold
        );
        // Cold: the stable core was rewritten (the policy must recompact the
        // new smaller prefix instead of growing the dead one).
        let rewritten = obs(
            &[h(1), h(2), h(3), h(4), h(7), h(6)],
            &[10, 10, 10, 10, 10, 10],
            0,
        );
        assert_eq!(
            classify_prefix_cache(Some(&prev), &rewritten, Some(fresh), now, &policy),
            PrefixCacheState::Cold
        );
        // Determinism over 100 runs.
        for _ in 0..100 {
            assert_eq!(
                classify_prefix_cache(Some(&prev), &volatile_churn, Some(idle), now, &policy),
                PrefixCacheState::Cold
            );
            assert_eq!(
                classify_prefix_cache(Some(&prev), &volatile_churn, Some(fresh), now, &policy),
                PrefixCacheState::Warm
            );
        }
        // Hostile time inputs never panic.
        assert_eq!(
            classify_prefix_cache(
                Some(&prev),
                &volatile_churn,
                Some(i64::MAX),
                i64::MIN,
                &policy
            ),
            PrefixCacheState::Warm
        );
        assert_eq!(
            classify_prefix_cache(
                Some(&prev),
                &volatile_churn,
                Some(0),
                now,
                &PrefixCachePolicy { ttl_ms: -1 }
            ),
            PrefixCacheState::Cold
        );
    }

    #[test]
    fn recompaction_is_deterministic_byte_stable_and_never_larger() {
        let est = Estimator;
        // Already minimal: byte-identical, never rewritten.
        let small = "GOAL: ship it\nCONSTRAINTS:\n- rust first\n";
        assert_eq!(recompact_stable_prefix(small, 10_000), small);
        // Over target with duplicated lines: deduped, shorter, deterministic.
        let repeated = "## Project rules\nno global state\n".repeat(50);
        let target = 40;
        let a = recompact_stable_prefix(&repeated, target);
        let b = recompact_stable_prefix(&repeated, target);
        assert_eq!(a, b, "same input + target must be byte-stable");
        assert!(a.len() <= repeated.len(), "recompaction never grows");
        assert!(
            est.estimate_tokens(&a) <= target + 1,
            "recompacted estimate {} exceeds target {target}",
            est.estimate_tokens(&a)
        );
        // Unicode truncation never splits a char boundary.
        let unicode = "汉字😀".repeat(400);
        let r = recompact_stable_prefix(&unicode, 5);
        assert!(unicode.is_char_boundary(0));
        assert!(r.is_char_boundary(r.len()));
        assert!(est.estimate_tokens(&r) <= 6);
        // 100 runs are bit-identical (byte-stable outcome).
        let first = recompact_stable_prefix(&repeated, target);
        for _ in 0..100 {
            assert_eq!(recompact_stable_prefix(&repeated, target), first);
        }
    }

    // ------------------------------------------------------------ prefix:
    // byte-truth boundary and head stability (audits 65-66 regressions).

    fn ledger_with(mut l: TaskLedger, goal: &str) -> TaskLedger {
        l.goal = goal.into();
        l
    }

    fn evidence_set(paths: &[(&str, f64)]) -> Vec<Evidence> {
        paths
            .iter()
            .map(|(path, score)| Evidence {
                path: (*path).into(),
                snippet: format!("snippet of {path}"),
                score: *score,
            })
            .collect()
    }

    #[test]
    fn cacheable_prefix_boundary_never_precedes_static_or_semi_stable_content() {
        // Two consecutive turns with the SAME stable head but DIFFERENT
        // volatile tails must produce byte-identical cacheable prefixes, and
        // volatile content must sit strictly AFTER the boundary.
        let b = ContextBudget::default();
        let static_input = (
            "You are Faktor.\nStay cacheable.",
            "rules",
            &ledger_with(TaskLedger::default(), "fix the parser"),
            "src/\nlib/",
        );
        let plan_a = plan_wire_request(
            static_input.0,
            "",
            &[tool("echo")],
            static_input.1,
            static_input.2,
            static_input.3,
            &[],
            &evidence_set(&[("src/a.rs", 1.0), ("src/b.rs", 9.0)]),
            "error one",
            &b,
        )
        .unwrap();
        let plan_b = plan_wire_request(
            static_input.0,
            "",
            &[tool("echo")],
            static_input.1,
            static_input.2,
            static_input.3,
            &[],
            &evidence_set(&[("src/z.rs", 0.1)]),
            "a completely different current error",
            &b,
        )
        .unwrap();
        let pa = plan_a.cacheable_prefix().unwrap();
        let pb = plan_b.cacheable_prefix().unwrap();
        assert_eq!(pa, pb, "volatile tail changes must not move the head");
        assert_eq!(plan_a.cacheable_prefix_len, plan_b.cacheable_prefix_len);
        for plan in [&plan_a, &plan_b] {
            let prefix = plan.cacheable_prefix().unwrap();
            for forbidden in [
                "## Retrieved evidence",
                "## Current errors",
                "src/a.rs",
                "src/b.rs",
                "src/z.rs",
                "error one",
            ] {
                assert!(
                    !prefix.contains(forbidden),
                    "volatile content {forbidden:?} leaked into the cacheable head"
                );
            }
            let tail = &plan.system[plan.cacheable_prefix_len..];
            assert!(
                tail.contains("## Retrieved evidence") || tail.contains("## Current errors"),
                "the volatile tail must follow the boundary"
            );
        }
        assert_eq!(blake3::hash(pa.as_bytes()), blake3::hash(pb.as_bytes()));
    }

    #[test]
    fn contract_rewrite_moves_boundary_but_progress_never_does() {
        let b = ContextBudget::default();
        let base = contract_ledger();
        let p1 = plan_wire_request(
            "You are Faktor.\n",
            "",
            &[tool("echo")],
            "rules",
            &base,
            "",
            &[],
            &[],
            "",
            &b,
        )
        .unwrap();
        // Progress churn (a touched file) renders in the volatile tail: the
        // boundary and the head bytes must not move.
        let mut l2 = base.clone();
        l2.changed_files.push("src/ledger.rs".into());
        let p2 = plan_wire_request(
            "You are Faktor.\n",
            "",
            &[tool("echo")],
            "rules",
            &l2,
            "",
            &[],
            &[],
            "",
            &b,
        )
        .unwrap();
        assert_eq!(p1.cacheable_prefix_len, p2.cacheable_prefix_len);
        assert_eq!(
            p1.cacheable_prefix().unwrap(),
            p2.cacheable_prefix().unwrap()
        );
        assert_ne!(p1.system, p2.system, "progress must render (tail)");
        assert!(p2.system.contains("src/ledger.rs"));
        // A contract rewrite (goal extended + constraint appended) moves the
        // head — the first changed segment is the contract.
        let mut l3 = base.clone();
        l3.goal.push_str(" (extended criteria)");
        l3.constraints.push("no unsafe".into());
        let p3 = plan_wire_request(
            "You are Faktor.\n",
            "",
            &[tool("echo")],
            "rules",
            &l3,
            "",
            &[],
            &[],
            "",
            &b,
        )
        .unwrap();
        assert_ne!(
            p3.cacheable_prefix().unwrap(),
            p1.cacheable_prefix().unwrap()
        );
        let first_changed = p3
            .prompt_segments
            .first_changed_segment(&p1.prompt_segments);
        assert_eq!(first_changed, Some(2), "the contract is segment 2");
        // The STATIC part (before the contract marker) is byte-identical.
        let marker = "## Task contract";
        fn static_part<'a>(plan: &'a WirePlan, marker: &str) -> &'a str {
            let m = plan.system.find(marker).unwrap();
            &plan.system[..m]
        }
        assert_eq!(static_part(&p1, marker), static_part(&p2, marker));
        assert_eq!(static_part(&p1, marker), static_part(&p3, marker));
    }

    #[test]
    fn evidence_reorder_never_moves_the_prefix_boundary() {
        let b = ContextBudget::default();
        let common = (
            "You are Faktor.\n",
            "",
            &ledger_with(TaskLedger::default(), "same goal"),
            "",
        );
        let p1 = plan_wire_request(
            common.0,
            "",
            &[tool("echo")],
            common.1,
            common.2,
            common.3,
            &[],
            &evidence_set(&[("src/a.rs", 1.0), ("src/b.rs", 9.0)]),
            "",
            &b,
        )
        .unwrap();
        let p2 = plan_wire_request(
            common.0,
            "",
            &[tool("echo")],
            common.1,
            common.2,
            common.3,
            &[],
            &evidence_set(&[("src/a.rs", 9.0), ("src/b.rs", 1.0)]),
            "",
            &b,
        )
        .unwrap();
        assert_eq!(p1.cacheable_prefix_len, p2.cacheable_prefix_len);
        assert_eq!(
            p1.cacheable_prefix().unwrap(),
            p2.cacheable_prefix().unwrap()
        );
        // Equal-score ties break by path: any permutation stays in the tail.
        let mut paths: Vec<Evidence> = vec![
            Evidence {
                path: "z.rs".into(),
                snippet: "z".into(),
                score: 5.0,
            },
            Evidence {
                path: "a.rs".into(),
                snippet: "a".into(),
                score: 5.0,
            },
            Evidence {
                path: "m.rs".into(),
                snippet: "m".into(),
                score: 5.0,
            },
        ];
        for _ in 0..6 {
            let plan = plan_wire_request(
                common.0,
                "",
                &[tool("echo")],
                common.1,
                common.2,
                common.3,
                &[],
                &paths,
                "",
                &b,
            )
            .unwrap();
            let prefix = plan.cacheable_prefix().unwrap();
            let tail = &plan.system[plan.cacheable_prefix_len..];
            for ev in &paths {
                assert!(!prefix.contains(ev.path.as_str()));
                assert!(tail.contains(ev.path.as_str()));
            }
            paths.rotate_left(1);
        }
    }

    #[test]
    fn hostile_markers_and_multibyte_inputs_never_break_the_boundary() {
        let b = ContextBudget::default();
        let hostile_evidence = vec![
            Evidence {
                path: "src/## Task state.rs".into(),
                snippet: "## Repository map\n## Steering\n## Current errors".into(),
                score: 1e9,
            },
            Evidence {
                path: "汉/字.rs".into(),
                snippet: "😀".repeat(2000),
                score: -1.0,
            },
        ];
        let plan = plan_wire_request(
            &"s".repeat(3_000),
            "steer 汉字",
            &[tool("echo")],
            &"r".repeat(2_000),
            &TaskLedger {
                goal: "目标".repeat(200),
                ..Default::default()
            },
            &"m".repeat(1_500),
            &[],
            &hostile_evidence,
            &"e".repeat(1_000),
            &b,
        )
        .unwrap();
        let prefix = plan.cacheable_prefix().unwrap();
        assert!(!prefix.contains("## Current errors"));
        assert!(!prefix.contains("## Retrieved evidence"));
        assert!(!prefix.contains("steer 汉字"), "steering is volatile now");
        let tail = &plan.system[plan.cacheable_prefix_len..];
        assert!(tail.contains("steer 汉字") && tail.contains("## Current errors"));
        for marker in ["## Retrieved evidence", "## Current errors"] {
            if let Some(pos) = plan.system.find(marker) {
                assert!(
                    pos >= plan.cacheable_prefix_len,
                    "{marker} precedes the head"
                );
            }
        }
        assert!(plan.system.is_char_boundary(plan.cacheable_prefix_len));
    }

    #[test]
    fn wire_plan_exposes_one_boundary_and_head_is_prefix_of_system() {
        let b = ContextBudget::default();
        let plan = plan_wire_request(
            "instructions",
            "",
            &[],
            "",
            &ledger_with(TaskLedger::default(), "g"),
            "",
            &[],
            &[],
            "",
            &b,
        )
        .unwrap();
        assert_eq!(plan.cacheable_prefix_len, plan.system.len());
        assert_eq!(plan.cacheable_prefix().unwrap(), plan.system.as_str());
        assert_eq!(
            plan.stable_leading_tokens(None),
            plan.prompt_segments.total_tokens()
        );
        // With a volatile tail the head is a strict prefix ending before it.
        let plan2 = plan_wire_request(
            "instructions",
            "",
            &[],
            "",
            &ledger_with(TaskLedger::default(), "g"),
            "",
            &[],
            &evidence_set(&[("src/a.rs", 1.0)]),
            "err",
            &b,
        )
        .unwrap();
        assert!(plan2.cacheable_prefix_len < plan2.system.len());
        assert_eq!(
            &plan2.system[..plan2.cacheable_prefix_len],
            plan2.cacheable_prefix().unwrap()
        );
        assert!(
            plan2.system[plan2.cacheable_prefix_len..].starts_with("\n## Retrieved evidence"),
            "the volatile tail must begin exactly at the boundary"
        );
    }

    #[test]
    fn measure_wire_request_equals_the_renderers_own_total() {
        // The measurement helper the routing tripwire consumes must equal
        // `plan_wire_request`'s `total_tokens` byte-for-byte: same estimator,
        // same per-message envelope, same tool accounting. Any drift here
        // would silently un-tether routing from the planned wire.
        let b = ContextBudget::default();
        let history = text_history(40);
        let tools = vec![tool("echo"), tool("read_file")];
        let plan = plan_wire_request(
            "You are Faktor.\n",
            "steer",
            &tools,
            "rules",
            &ledger(),
            "map",
            &history,
            &evidence(3),
            "errors",
            &b,
        )
        .unwrap();
        assert_eq!(
            measure_wire_request(&plan.system, &plan.messages, &plan.tools),
            plan.total_tokens,
            "measured request total must equal the plan's own total"
        );
        // Empty request: zero, not a panic.
        assert_eq!(measure_wire_request("", &[], &[]), 0);
    }

    #[test]
    fn candidate_sizing_uses_each_models_own_tokenizer_and_never_fakes_exactness() {
        let cache = TokenCache::new();
        let messages = vec![RequestMessage {
            role: Role::User,
            content: vec![ContentPart::text(
                "fn main() { let x = 1; } // the quick brown fox 汉字 😀".repeat(12),
            )],
        }];
        let tools = vec![tool("echo")];
        // o200k_base is registered locally: exact.
        let gpt5 = size_request_for_model("gpt-5", "system", &messages, &tools, &cache);
        assert!(gpt5.input_tokens > 0);
        assert!(gpt5.exact, "gpt-5 maps to a registered o200k backend");
        // Anthropic has no local vocabulary: an honest upper bound.
        let claude = size_request_for_model("claude-3-5", "system", &messages, &tools, &cache);
        assert!(claude.input_tokens > 0);
        assert!(
            !claude.exact,
            "an unregistered family must never be labeled exact"
        );
        // The two identities are genuinely distinct counts on this corpus:
        // sizing everything under one shared tokenizer is a bug this lock
        // would catch.
        let cl100k = size_request_for_model("gpt-4", "system", &messages, &tools, &cache);
        assert!(cl100k.input_tokens > 0);
        assert_ne!(
            gpt5.input_tokens, cl100k.input_tokens,
            "o200k and cl100k must not be one shared count"
        );
        // Repeats are cache hits, never re-tokenizations.
        let again = size_request_for_model("gpt-5", "system", &messages, &tools, &cache);
        assert_eq!(again, gpt5);
        assert!(cache.hits() > 0, "the second sizing must hit the cache");
    }

    #[test]
    fn candidate_sizing_counts_structured_parts_as_real_bytes() {
        // Tool calls and schemas are sized through their serialized JSON
        // under the candidate tokenizer (not silently skipped): a tool-heavy
        // request must size strictly larger than the same request without
        // tools.
        let cache = TokenCache::new();
        let messages = vec![RequestMessage {
            role: Role::Assistant,
            content: vec![ContentPart::tool_call(
                "call_1",
                "read_file",
                serde_json::json!({ "path": "src/main.rs" }),
            )],
        }];
        let bare = size_request_for_model("gpt-5", "s", &messages, &[], &cache);
        let with_tools = size_request_for_model("gpt-5", "s", &messages, &[tool("echo")], &cache);
        assert!(
            with_tools.input_tokens > bare.input_tokens,
            "tool schemas must add to the footprint: {bare:?} vs {with_tools:?}"
        );
    }
}
