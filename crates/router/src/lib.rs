//! Economic model router (audit: minimize expected cost to VERIFIED
//! success, not price per token).
//!
//! Two candidate surfaces share one qualification/scoring ladder:
//! the PRICED production path ([`RouteCandidate`] = descriptor + a
//! [`PricingState`]) prices every call through the exact per-million
//! [`faktor_core::model::PriceQuote`] and never touches a per-token field;
//! the legacy descriptor-only constructors remain for untouched callers.
//!
//! Deterministic algorithm:
//! 1. ONE authoritative qualification pass (`qualified_candidates` /
//!    `qualified_priced_candidates`) applies every axis — capability filter
//!    (fail closed on unknown capabilities), context/output fit, phase
//!    quality floor, request budget, latency preference, and live
//!    rate-limit/cooldown state (static Hard state plus telemetry cooldowns
//!    fed in as a `LiveHealth` snapshot). Every selection path consumes
//!    ONLY `QualifiedCandidate` vectors; there is exactly one place that
//!    filters;
//! 2. scoring is explicit on `ScoredCandidate`/`PricedScoredCandidate`:
//!    expected cost-to-success in microUSD (prompt-cache-aware, integer,
//!    rounded up) with the documented tie-break ladder — no tuple-slot
//!    overloading, no unit-confused comparisons (base microUSD is never
//!    compared against milliseconds), and an unpriced/Unknown candidate is
//!    NON-numeric (ranked after every number, never read as zero);
//! 3. money is computed only from a [`faktor_core::model::PricingSnapshot`]
//!    quote ([`faktor_core::model::PriceQuote::quote_cost_micro`]) and
//!    carried as [`CostEstimate`]; the legacy per-token projection is
//!    confined to the compatibility constructors and is never consulted by
//!    the priced path;
//! 4. local zero-cost models count as cost-free but latency-weighted;
//! 5. every decision carries an audit string (phase, considered,
//!    qualified, chosen, cost, latency, floor) — no hidden choices;
//! 6. verified-outcome history is ADDITIVE (audit items 13/14/L): a
//!    [`RouterService`] may be built with an [`OutcomeStore`]
//!    ([`RouterService::with_outcomes`]) whose per-key verified stats, when
//!    they exist, replace the telemetry-prior expected-cost term with the
//!    conservative [`WorkCostEstimate`] — expected cost to VERIFIED
//!    completion = immediate cost + P(rework) x downstream spend, with
//!    rework probability as a Wilson upper bound and the conservative
//!    verified-success confidence as the ladder's success prior
//!    (MaximumQuality's consumption). The default registry is empty and
//!    scoring over it is byte-identical to the legacy math.
//!
//! Pinned routing ([`RouterService::with_pinned_route_candidates`]) calls
//! ONLY [`qualify_specific`]: the pin never competes, and no other
//! candidate can reject it.

use std::collections::HashMap;
use std::sync::Arc;

use faktor_core::model::{
    ModelDescriptor, ModelEconomics, ModelPerformance, PriceAuthority, PricingSnapshot,
    PricingState, RateLimitState, RiskBucket, RouteDecision, RouterPhase, TaskClass, TokenUsage,
};

/// Journaled task-budget ledger (micro-units) with reservations,
/// settlements, refunds and crash reconstruction from the denial journal —
/// model-checked against the router's hard-budget semantics (audit 79-80).
pub mod budget;

/// Prefix-cache stability measurement: per-turn stability, session mean/std,
/// the churn advisory detector and the churn cost premium (audits 65-66).
/// Pure and deterministic; inputs are per-turn prefix observations the
/// settlement layer persists.
pub mod stability;

/// Verified-outcome learning (audit items 13/14/L): conservative Bayesian
/// rework estimates from verified-only history, plus the outcome-registry
/// surface a [`RouterService`] consults when stats exist.
pub mod outcomes;

pub use outcomes::{
    rework_probability_ppm, verified_success_confidence_ppm, work_cost_estimate, EmptyOutcomeStore,
    MemoryOutcomeStore, OutcomeKey, OutcomeSample, OutcomeStore, VerifiedOutcomeStats,
    WorkCostEstimate,
};

/// One routing request.
///
/// `task_class`/`risk_bucket` are the OUTCOME-LEARNING dimensions of the
/// request (audit items 13/14/L): the verified-outcome consult walks the
/// hierarchy (provider, model, phase, task, risk) -> (provider, model,
/// phase, task) -> (provider, model, phase) -> global telemetry prior —
/// it never collapses straight to the global prior while a narrower key
/// holds evidence. Both default to the neutral bucket for additive
/// compatibility.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RouteRequest {
    pub phase: RouterPhase,
    pub required_capabilities: Vec<String>,
    pub context_tokens: u64,
    pub estimated_output_tokens: u64,
    /// 0..=100; the mean coding quality must clear this floor.
    pub quality_floor: u8,
    /// Remaining task budget in micro-units (0 = unlimited).
    pub task_budget_remaining_micro: u64,
    pub latency_preference_ms: Option<u64>,
    /// The class of work this request carries (outcome-learning dimension).
    #[serde(default)]
    pub task_class: TaskClass,
    /// The semantic risk bucket of the operation (outcome-learning
    /// dimension), decided by the runtime's own risk model.
    #[serde(default)]
    pub risk_bucket: RiskBucket,
}

impl Default for RouteRequest {
    fn default() -> Self {
        Self {
            phase: RouterPhase::Implement,
            required_capabilities: vec!["tools".into(), "streaming".into()],
            context_tokens: 16_384,
            estimated_output_tokens: 2048,
            quality_floor: 60,
            task_budget_remaining_micro: 0,
            latency_preference_ms: None,
            task_class: TaskClass::Medium,
            risk_bucket: RiskBucket::Low,
        }
    }
}

/// Observed cache state for (provider, model): tokens already cached
/// (read hits) and tokens this call writes into the cache.
#[derive(Debug, Clone, PartialEq)]
pub struct CacheState {
    pub provider: String,
    pub model: String,
    pub cached_input_tokens: u64,
    pub will_write_tokens: u64,
}

/// The router's monetary estimate of ONE candidate call (pricing-path
/// audit): the EXACT per-million-token quote evaluated over the request's
/// token categories. The variant carries the pricing AUTHORITY, never a
/// number inferred from one:
///
/// - [`CostEstimate::Known`] — an exact quote priced the call;
/// - [`CostEstimate::Conservative`] — a conservative ceiling (or the
///   legacy descriptor projection on compatibility paths) priced it;
/// - [`CostEstimate::LocalZero`] — an authoritative local zero (exactly 0);
/// - [`CostEstimate::Unknown`] — NO number is honest; the router never
///   treats it as 0 and never lets it into numeric comparisons.
///
/// [`CostEstimate::numeric`] is the ONLY way to reach a number: `Unknown`
/// returns `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CostEstimate {
    Known(u64),
    Conservative(u64),
    LocalZero,
    Unknown,
}

impl CostEstimate {
    /// Evaluate a frozen [`PricingSnapshot`] against one usage frame: the
    /// summed-numerator, one-ceiling-division quote math, keyed by the
    /// snapshot's AUTHORITY. A hostile inconsistency (Unknown authority
    /// with a quote, LocalZero without one) resolves to `Unknown` — no
    /// fabricated number.
    pub fn from_snapshot(snapshot: &PricingSnapshot, usage: TokenUsage) -> Self {
        match snapshot.authority {
            PriceAuthority::Exact => snapshot
                .quote
                .map(|q| CostEstimate::Known(q.quote_cost_micro(usage)))
                .unwrap_or(CostEstimate::Unknown),
            PriceAuthority::ConservativeCeiling => snapshot
                .quote
                .map(|q| CostEstimate::Conservative(q.quote_cost_micro(usage)))
                .unwrap_or(CostEstimate::Unknown),
            PriceAuthority::LocalZero => snapshot
                .quote
                .map(|_| CostEstimate::LocalZero)
                .unwrap_or(CostEstimate::Unknown),
            PriceAuthority::Unknown => CostEstimate::Unknown,
        }
    }

    /// Evaluate a [`PricingState`] against one usage frame.
    pub fn from_state(state: &PricingState, usage: TokenUsage) -> Self {
        Self::from_snapshot(&state.snapshot(), usage)
    }

    /// The honest number, when one exists: exact/ceiling costs and the
    /// authoritative local zero. `Unknown` is NON-numeric — it is never 0.
    pub const fn numeric(self) -> Option<u64> {
        match self {
            CostEstimate::Known(m) | CostEstimate::Conservative(m) => Some(m),
            CostEstimate::LocalZero => Some(0),
            CostEstimate::Unknown => None,
        }
    }

    pub const fn is_unknown(self) -> bool {
        matches!(self, CostEstimate::Unknown)
    }

    pub const fn is_local_zero(self) -> bool {
        matches!(self, CostEstimate::LocalZero)
    }
}

/// The router's PRICED unit (pricing-path audit): a model descriptor plus
/// the pricing state the catalog authority resolved for it. Every priced
/// qualification/scoring path consumes these; the descriptor alone can no
/// longer smuggle a lossy per-token projection into a decision.
#[derive(Debug, Clone, PartialEq)]
pub struct RouteCandidate {
    pub descriptor: ModelDescriptor,
    pub pricing: PricingState,
}

impl RouteCandidate {
    pub fn new(descriptor: ModelDescriptor, pricing: PricingState) -> Self {
        Self {
            descriptor,
            pricing,
        }
    }

    /// The candidate's non-monetary performance view.
    pub fn performance(&self) -> ModelPerformance {
        self.descriptor.performance()
    }

    /// The candidate's exact cost estimate for this request/cache state,
    /// evaluated through the per-million-token quote (never through a
    /// per-token projection).
    pub fn cost_estimate(&self, req: &RouteRequest, cache: &[CacheState]) -> CostEstimate {
        CostEstimate::from_state(&self.pricing, request_usage(req, cache, &self.descriptor))
    }
}

/// The token categories one request/cache state projects for a candidate.
fn request_usage(req: &RouteRequest, cache: &[CacheState], d: &ModelDescriptor) -> TokenUsage {
    let cs = cache
        .iter()
        .find(|c| c.provider == d.provider && c.model == d.model);
    let (cached, will_write) = cs
        .map(|c| {
            (
                c.cached_input_tokens.min(req.context_tokens),
                c.will_write_tokens,
            )
        })
        .unwrap_or((0, 0));
    TokenUsage::new(
        req.context_tokens.saturating_sub(cached),
        cached,
        will_write,
        req.estimated_output_tokens,
    )
}

/// LEGACY descriptor-only cost estimate (microUSD) over the per-token
/// compatibility fields of [`ModelEconomics`]. **The priced routing path
/// never calls this**: it prices calls through
/// [`CostEstimate::from_state`] over the exact per-million-token quote.
/// This function survives for untouched descriptor-only callers (agent
/// budget fallbacks, compatibility constructors and their tests); values
/// below $1/M cannot be represented here, which is exactly why new price
/// knowledge must never enter through it.
///
/// UNITS (audit 9 + P0-5): the price fields are
/// [`faktor_core::model::MicroUsdPerToken`] — microUSD PER TOKEN, typed so
/// money can never mix with latency. The per-token microUSD value is
/// numerically equal to USD per million tokens (1e6 microUSD per USD over
/// 1e6 tokens), so a $15/Mtok price occupies the same integer as 15
/// microUSD/token and one integer serves both readings: cost_micro =
/// sum(tokens x price) with NO division. Saturating arithmetic makes
/// hostile magnitudes safe.
pub fn estimated_call_cost(
    ec: &ModelEconomics,
    input_tokens: u64,
    output_tokens: u64,
    cache_read_tokens: u64,
    cache_write_tokens: u64,
) -> u64 {
    let uncached = input_tokens.saturating_sub(cache_read_tokens);
    let mut cost = ec
        .input_price_per_mtok
        .saturating_mul(uncached)
        .saturating_add(
            ec.cache_read_price_per_mtok
                .saturating_mul(cache_read_tokens),
        )
        .saturating_add(
            ec.cache_write_price_per_mtok
                .saturating_mul(cache_write_tokens),
        )
        .saturating_add(ec.output_price_per_mtok.saturating_mul(output_tokens));
    // Cost is never understated: any priced byte carries at least 1 micro.
    if cost == 0 && !ec.is_local_zero_cost() {
        cost = 1;
    }
    cost
}

// ---------------------------------------------------------- qualification

/// Blended success prior when nothing was ever recorded for a key:
/// [`PRIOR_SUCCESS`] (0.8) expressed in parts-per-million.
const DEFAULT_SUCCESS_PPM: u32 = 800_000;

/// Point-in-time live-health snapshot consumed by the single qualification
/// pass ([`qualified_candidates`]).
///
/// [`RouterService`] captures one from its [`RouterTelemetry`] at the start
/// of every route; the plain [`Router`] path uses the default view (no
/// live cooldowns, prior success everywhere). Capturing once per decision
/// keeps a route deterministic within its own evaluation.
#[derive(Debug, Clone, Default)]
pub struct LiveHealth {
    /// Providers inside an active rate-limit cooldown window.
    cooldown: std::collections::HashSet<String>,
    /// Telemetry-blended success prior in ppm, keyed
    /// `(provider, model, phase)`.
    success_ppm: std::collections::HashMap<(String, String, RouterPhase), u32>,
}

impl LiveHealth {
    fn cooldown_active(&self, provider: &str) -> bool {
        self.cooldown.contains(provider)
    }

    fn success_ppm(&self, provider: &str, model: &str, phase: RouterPhase) -> u32 {
        self.success_ppm
            .get(&(provider.to_string(), model.to_string(), phase))
            .copied()
            .unwrap_or(DEFAULT_SUCCESS_PPM)
    }
}

/// One candidate that cleared EVERY qualification axis for a request:
/// live rate-limit/cooldown state, capabilities, context/output fit, the
/// phase quality floor, the request budget and the latency preference.
///
/// Every selection path — plain cheapest ([`Router::route`]),
/// expected-cost ([`RouterService::route`]), maximum-quality requests and
/// the escalation pool — consumes ONLY vectors of these. There is exactly
/// one filtering pass in the router; a candidate absent here can never be
/// chosen, costed as an escalation target, or tie-broken into a decision.
#[derive(Debug, Clone, Copy)]
pub struct QualifiedCandidate<'a> {
    pub descriptor: &'a ModelDescriptor,
    /// Cache-aware single-call cost in microUSD (the base cost: integer,
    /// rounded up to >= 1 micro for any priced model). For an
    /// [`CostEstimate::Unknown`] priced candidate this carries `u64::MAX`
    /// (unbounded) — [`QualifiedCandidate::cost`] is the authoritative
    /// estimate; unknown is NEVER a numeric zero.
    pub call_cost_micro: u64,
    /// Telemetry-blended success prior in ppm (1_000_000 = certain).
    pub success_ppm: u32,
    /// The authoritative cost estimate of this candidate (pricing-path
    /// audit): exact quote cost, conservative ceiling, authoritative local
    /// zero, or non-numeric Unknown.
    pub cost: CostEstimate,
}

impl QualifiedCandidate<'_> {
    /// The numeric base cost, when one exists (`None` for Unknown).
    pub const fn numeric_cost(&self) -> Option<u64> {
        self.cost.numeric()
    }
}

/// Why [`qualified_candidates`] returned an empty vector, staged so each
/// path reproduces its distinct denial message (capability/fit, quality
/// floor, or budget/latency).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualificationFailure {
    pub phase: RouterPhase,
    /// The floor actually applied: `quality_floor` clamped to 100.
    pub quality_floor: u8,
    /// Required capabilities supported by no candidate that cleared the
    /// live-health axis.
    pub unsupported_capabilities: Vec<String>,
    /// Survivors of capability + context/output fit.
    pub fit_survivors: usize,
    /// Survivors of the phase quality floor as well.
    pub quality_survivors: usize,
}

impl QualificationFailure {
    /// The distinct denial strings of the legacy plain router, derived from
    /// the staged survivor counts (never from a second filtering loop).
    pub fn route_error(&self) -> String {
        if self.fit_survivors == 0 {
            format!(
                "no candidate clears capability/fit filtering (missing: {})",
                self.unsupported_capabilities.join(",")
            )
        } else if self.quality_survivors == 0 {
            format!(
                "no candidate clears the quality floor {} for phase {:?}",
                self.quality_floor, self.phase
            )
        } else {
            "no candidate within the remaining budget/latency constraints".to_string()
        }
    }
}

/// Heavy phases trust the coding reliability; cheap phases
/// (summarize/title/embed/plan/...) compare the floor against context
/// reliability only. Documented per the audit's cheap-phase guidance; both
/// metrics come from the NON-MONETARY [`ModelPerformance`] view.
fn floor_metric_heavy(phase: RouterPhase) -> bool {
    matches!(
        phase,
        RouterPhase::Implement | RouterPhase::Review | RouterPhase::Debug
    )
}

fn clears_quality_floor(p: &ModelPerformance, phase: RouterPhase, floor: u8) -> bool {
    if floor_metric_heavy(phase) {
        p.coding_reliability >= floor
    } else {
        p.context_reliability >= floor
    }
}

/// Cache-aware base cost of one call, shared by the qualification budget
/// axis and the scored candidates (computed exactly once per candidate).
fn base_call_cost(d: &ModelDescriptor, req: &RouteRequest, cache: &[CacheState]) -> u64 {
    let cs = cache
        .iter()
        .find(|c| c.provider == d.provider && c.model == d.model);
    let (cached, will_write) = cs
        .map(|c| {
            (
                c.cached_input_tokens.min(req.context_tokens),
                c.will_write_tokens,
            )
        })
        .unwrap_or((0, 0));
    estimated_call_cost(
        &d.economics,
        req.context_tokens,
        req.estimated_output_tokens,
        cached,
        will_write,
    )
}

/// THE single authoritative qualification pass (audit P0-3/P0-89).
///
/// Applied once per route, in this order, with every axis evaluated in one
/// loop over the candidates:
/// 1. live rate-limit/cooldown state: static [`RateLimitState::Hard`] or an
///    active cooldown in `health` (replaces the inert
///    `RouteRequest::rate_limit_blocks` — P0-89);
/// 2. required capabilities (fail closed on unknown capabilities);
/// 3. context window and max-output fit;
/// 4. the phase quality floor (coding mean for Implement/Review/Debug,
///    context reliability otherwise);
/// 5. the request budget (`task_budget_remaining_micro`, 0 = unlimited);
/// 6. the latency preference (a hard filter when set).
///
/// The returned vector contains only candidates that passed every axis; it
/// is the ONLY input every selection path is allowed to choose from.
pub fn qualified_candidates<'a>(
    candidates: &'a [ModelDescriptor],
    req: &RouteRequest,
    cache: &[CacheState],
    health: &LiveHealth,
) -> Result<Vec<QualifiedCandidate<'a>>, QualificationFailure> {
    let floor = req.quality_floor.min(100);
    let mut fit_survivors = 0usize;
    let mut quality_survivors = 0usize;
    let mut supported: Vec<String> = Vec::new();
    let mut qualified: Vec<QualifiedCandidate<'a>> = Vec::new();
    for d in candidates {
        // Axis 1: live rate-limit/cooldown state (static + telemetry).
        if d.performance().rate_limit_state == RateLimitState::Hard
            || health.cooldown_active(&d.provider)
        {
            continue;
        }
        for cap in &req.required_capabilities {
            if !supported.iter().any(|s| s == cap) && d.capability_ok(std::slice::from_ref(cap)) {
                supported.push(cap.clone());
            }
        }
        // Axis 2: capabilities (fail closed).
        if !d.capability_ok(&req.required_capabilities) {
            continue;
        }
        // Axis 3: context/output fit.
        if d.context < req.context_tokens || d.max_output < req.estimated_output_tokens {
            continue;
        }
        fit_survivors += 1;
        // Axis 4: quality floor.
        if !clears_quality_floor(&d.performance(), req.phase, floor) {
            continue;
        }
        quality_survivors += 1;
        // Axis 5: request budget.
        let cost = base_call_cost(d, req, cache);
        if req.task_budget_remaining_micro > 0 && cost > req.task_budget_remaining_micro {
            continue;
        }
        // Axis 6: latency preference (hard filter when set).
        if let Some(lp) = req.latency_preference_ms {
            if d.performance().estimated_latency_ms > lp {
                continue;
            }
        }
        qualified.push(QualifiedCandidate {
            descriptor: d,
            call_cost_micro: cost,
            success_ppm: health.success_ppm(&d.provider, &d.model, req.phase),
            // The legacy descriptor path has no exact quote: its rounded-up
            // per-token projection is a CONSERVATIVE estimate, never a
            // claimed exact price.
            cost: CostEstimate::Conservative(cost),
        });
    }
    if qualified.is_empty() {
        let unsupported_capabilities = req
            .required_capabilities
            .iter()
            .filter(|c| !supported.iter().any(|s| s == *c))
            .cloned()
            .collect();
        return Err(QualificationFailure {
            phase: req.phase,
            quality_floor: floor,
            unsupported_capabilities,
            fit_survivors,
            quality_survivors,
        });
    }
    Ok(qualified)
}

/// THE priced qualification pass (pricing-path audit): the same axes as
/// [`qualified_candidates`], but every monetary decision is made on the
/// candidate's [`CostEstimate`] from its exact per-million quote — never on
/// a per-token projection.
///
/// The budget axis is where "unknown is nonnumeric" bites: with a positive
/// `task_budget_remaining_micro`, an [`CostEstimate::Unknown`] candidate is
/// excluded (no honest bound fits the cap — fail closed); without a cap it
/// is admitted and its spend settles as a documented Unknown amount.
/// Numeric costs (exact, conservative, local zero) compare against the
/// remaining budget exactly.
pub fn qualified_priced_candidates<'a>(
    candidates: &'a [RouteCandidate],
    req: &RouteRequest,
    cache: &[CacheState],
    health: &LiveHealth,
) -> Result<Vec<QualifiedCandidate<'a>>, QualificationFailure> {
    let refs: Vec<&'a RouteCandidate> = candidates.iter().collect();
    qualify_priced_refs(&refs, req, cache, health)
}

fn qualify_priced_refs<'a>(
    candidates: &[&'a RouteCandidate],
    req: &RouteRequest,
    cache: &[CacheState],
    health: &LiveHealth,
) -> Result<Vec<QualifiedCandidate<'a>>, QualificationFailure> {
    let floor = req.quality_floor.min(100);
    let mut fit_survivors = 0usize;
    let mut quality_survivors = 0usize;
    let mut supported: Vec<String> = Vec::new();
    let mut qualified: Vec<QualifiedCandidate<'a>> = Vec::new();
    for c in candidates {
        let d = &c.descriptor;
        // Axis 1: live rate-limit/cooldown state (static + telemetry).
        if d.performance().rate_limit_state == RateLimitState::Hard
            || health.cooldown_active(&d.provider)
        {
            continue;
        }
        for cap in &req.required_capabilities {
            if !supported.iter().any(|s| s == cap) && d.capability_ok(std::slice::from_ref(cap)) {
                supported.push(cap.clone());
            }
        }
        // Axis 2: capabilities (fail closed).
        if !d.capability_ok(&req.required_capabilities) {
            continue;
        }
        // Axis 3: context/output fit.
        if d.context < req.context_tokens || d.max_output < req.estimated_output_tokens {
            continue;
        }
        fit_survivors += 1;
        // Axis 4: quality floor.
        if !clears_quality_floor(&d.performance(), req.phase, floor) {
            continue;
        }
        quality_survivors += 1;
        // Axis 5: the request budget, over the EXACT cost estimate. Unknown
        // + positive budget fails closed; Unknown + no cap is admitted.
        let cost = c.cost_estimate(req, cache);
        if req.task_budget_remaining_micro > 0 {
            match cost.numeric() {
                Some(n) if n <= req.task_budget_remaining_micro => {}
                Some(_) => continue,
                None => continue,
            }
        }
        // Axis 6: latency preference (hard filter when set).
        if let Some(lp) = req.latency_preference_ms {
            if d.performance().estimated_latency_ms > lp {
                continue;
            }
        }
        qualified.push(QualifiedCandidate {
            descriptor: d,
            call_cost_micro: cost.numeric().unwrap_or(u64::MAX),
            success_ppm: health.success_ppm(&d.provider, &d.model, req.phase),
            cost,
        });
    }
    if qualified.is_empty() {
        let unsupported_capabilities = req
            .required_capabilities
            .iter()
            .filter(|c| !supported.iter().any(|s| s == *c))
            .cloned()
            .collect();
        return Err(QualificationFailure {
            phase: req.phase,
            quality_floor: floor,
            unsupported_capabilities,
            fit_survivors,
            quality_survivors,
        });
    }
    Ok(qualified)
}

/// PINNED qualification (audit item: pinned routing never competes): check
/// exactly ONE (provider, model) candidate against every hard axis and
/// return it, or a staged [`QualificationFailure`]. No other candidate is
/// evaluated, scored or allowed to win — "another model was cheaper" can
/// never reject a pin that clears the request's own hard caps.
///
/// The failure staging matches [`qualified_priced_candidates`] so the
/// pinned denial maps to the same typed failures (a pin missing tools is a
/// capability failure, an unknown pin under a hard cap is a budget
/// failure).
pub fn qualify_specific<'a>(
    candidates: &'a [RouteCandidate],
    provider: &str,
    model: &str,
    req: &RouteRequest,
    cache: &[CacheState],
    health: &LiveHealth,
) -> Result<QualifiedCandidate<'a>, QualificationFailure> {
    let refs: Vec<&'a RouteCandidate> = candidates
        .iter()
        .filter(|c| c.descriptor.provider == provider && c.descriptor.model == model)
        .collect();
    let mut qualified = qualify_priced_refs(&refs, req, cache, health)?;
    Ok(qualified.remove(0))
}

// ------------------------------------------- candidate-specific fit sizing

/// Hard bound on how many seriously-considered candidates one route may size
/// with their REAL tokenizer (audit: candidate-specific token accounting).
/// Only the top of the already qualified/ranked order is ever measured, so
/// the tokenizer work is bounded no matter how large the catalog is.
pub const MAX_SIZED_CANDIDATES: usize = 8;

/// One ranked candidate after the candidate-specific fit re-check: its REAL
/// input footprint (the rendered request counted under ITS tokenizer, or a
/// conservative upper bound) and whether that count was exact.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SizedCandidate<'a> {
    pub candidate: &'a ModelDescriptor,
    /// The rendered request's real input footprint under this candidate's
    /// tokenizer (honest upper bound when no local vocabulary exists).
    pub input_tokens: u64,
    /// True only when a real local tokenizer counted every run.
    pub exact: bool,
}

/// THE candidate-specific fit pass: before the final selection, size each of
/// the top `max_sized` (hard-capped to [`MAX_SIZED_CANDIDATES`]) candidates
/// in the caller's already-ranked, already-CHEAPLY-qualified order with the
/// caller's candidate sizer — which must count the rendered request under
/// that candidate's own tokenizer (or return a conservative upper bound).
/// A candidate whose REAL footprint exceeds its own context window is
/// filtered out; the survivors keep the ranked order and carry their
/// measured footprints.
///
/// Bounded work: the sizer runs at most `min(max_sized, 8)` times per route
/// (each sized candidate may warm the shared token cache, never iterate the
/// catalog). Candidates beyond the top-K are never sized and therefore never
/// returned — callers must treat an empty survivor set as "no candidate
/// cleared the real fit", never fall back to an unsized pick.
pub fn size_candidates_top_k<'a>(
    ranked: &[QualifiedCandidate<'a>],
    max_sized: usize,
    mut size: impl FnMut(&ModelDescriptor) -> (u64, bool),
) -> Vec<SizedCandidate<'a>> {
    let take = max_sized.min(MAX_SIZED_CANDIDATES).min(ranked.len());
    let mut survivors = Vec::with_capacity(take);
    for q in &ranked[..take] {
        let descriptor = q.descriptor;
        let (input_tokens, exact) = size(descriptor);
        if input_tokens <= descriptor.context {
            survivors.push(SizedCandidate {
                candidate: descriptor,
                input_tokens,
                exact,
            });
        }
    }
    survivors
}

// ---------------------------------------------------------------- scoring

/// One fully qualified, fully scored candidate.
///
/// All money fields are microUSD; all time fields are milliseconds. The
/// ranking ladder in [`ScoredCandidate::compare`] is explicit and ordered —
/// no tuple-slot overloading and never a comparison between a microUSD
/// figure and a millisecond figure (audit P0-4: the old tie-break compared
/// a base cost against the previous candidate's latency).
#[derive(Debug, Clone, Copy)]
pub struct ScoredCandidate<'a> {
    pub candidate: &'a ModelDescriptor,
    /// Expected cost-to-success in microUSD: base call cost plus the
    /// probabilistic retry and escalation terms (integer, rounded up), or —
    /// when per-phase verified-outcome history exists — the conservative
    /// [`WorkCostEstimate::total_expected_micro`] (immediate + P(rework) x
    /// downstream spend to VERIFIED completion).
    pub expected_cost_micro: u64,
    /// Estimated latency of one call in milliseconds.
    pub expected_latency_ms: u64,
    /// Success prior in ppm (1_000_000 = certain): the telemetry-blended
    /// prior when no verified history exists; otherwise the conservative
    /// one-sided lower-bound verified-success confidence (the prior
    /// MaximumQuality semantics consume — a two-sample track record stays
    /// far below the documented "excellent" bar).
    pub success_ppm: u32,
    /// Base per-call cost in microUSD (cache-aware).
    pub call_cost_micro: u64,
    /// The conservative work-cost estimate when verified history exists for
    /// this candidate's (provider, model, phase); `None` = legacy
    /// telemetry-prior scoring was used.
    pub work_estimate: Option<WorkCostEstimate>,
}

impl<'a> ScoredCandidate<'a> {
    /// The documented total order over scored candidates. Returns
    /// `Ordering::Less` when `self` ranks ahead of `other`. Ladder:
    /// 1. `expected_cost_micro` (microUSD cost-to-success) ascending,
    /// 2. `success_ppm` (success prior, ppm) descending,
    /// 3. `expected_latency_ms` (milliseconds) ascending,
    /// 4. `call_cost_micro` (microUSD) ascending,
    /// 5. `(provider, model)` lexicographic ascending — the deterministic
    ///    final key (identical candidates resolve identically every run).
    pub fn compare(&self, other: &Self) -> std::cmp::Ordering {
        self.expected_cost_micro
            .cmp(&other.expected_cost_micro)
            .then_with(|| other.success_ppm.cmp(&self.success_ppm))
            .then_with(|| self.expected_latency_ms.cmp(&other.expected_latency_ms))
            .then_with(|| self.call_cost_micro.cmp(&other.call_cost_micro))
            .then_with(|| {
                (&self.candidate.provider, &self.candidate.model)
                    .cmp(&(&other.candidate.provider, &other.candidate.model))
            })
    }
}

/// Expected cost-to-success of one qualified candidate (audit-9 integer
/// math): base + P(fail)/2 * base (retry) + P(fail)/2 * escalation, with
/// P(fail) = 1 - success_ppm/1e6. Each probabilistic term rounds the exact
/// product UP to the next integer micro, so the estimate never understates.
/// `escalation_cost_micro` is the base cost of the best OTHER qualified
/// candidate (or 3x the candidate's own base when none exists).
fn expected_cost_to_success(
    success_ppm: u32,
    call_cost_micro: u64,
    escalation_cost_micro: u128,
) -> u64 {
    let base = u128::from(call_cost_micro);
    let p_fail_ppm = u128::from(1_000_000u64 - u64::from(success_ppm.min(1_000_000)));
    // ceil(x * p_fail_ppm / 2_000_000) == (x * p_fail_ppm + 1_999_999) / 2_000_000.
    let retry = (base.saturating_mul(p_fail_ppm).saturating_add(1_999_999)) / 2_000_000;
    let escalate = (escalation_cost_micro
        .saturating_mul(p_fail_ppm)
        .saturating_add(1_999_999))
        / 2_000_000;
    let total = base + retry + escalate;
    u64::try_from(total).unwrap_or(u64::MAX)
}

/// The two cheapest DISTINCT (provider, model) qualified candidates,
/// ordered deterministically by `(cost, provider, model)`. Every candidate
/// escalates to the cheapest candidate that is not itself — the top two
/// suffice for that lookup.
#[derive(Debug, Clone, Copy)]
struct CheapestTwo<'a> {
    cheapest: Option<(u64, &'a str, &'a str)>,
    second: Option<(u64, &'a str, &'a str)>,
}

impl<'a> CheapestTwo<'a> {
    /// Base cost of the best OTHER candidate, or `fallback` (3x the
    /// candidate's own base) when it is the only qualified candidate.
    fn escalation_for(&self, provider: &str, model: &str, fallback: u128) -> u128 {
        match self.cheapest {
            None => fallback,
            Some((cost, p, m)) => {
                if p == provider && m == model {
                    self.second
                        .map(|(c, _, _)| u128::from(c))
                        .unwrap_or(fallback)
                } else {
                    u128::from(cost)
                }
            }
        }
    }
}

fn two_cheapest_distinct<'a>(qualified: &[QualifiedCandidate<'a>]) -> CheapestTwo<'a> {
    let mut first: Option<(u64, &'a str, &'a str)> = None;
    let mut second: Option<(u64, &'a str, &'a str)> = None;
    for q in qualified {
        let d = q.descriptor;
        let entry = (q.call_cost_micro, d.provider.as_str(), d.model.as_str());
        let same_key_as =
            |current: &(u64, &str, &str)| current.1 == entry.1 && current.2 == entry.2;
        let overtakes = |current: &(u64, &str, &str)| {
            entry.0 < current.0
                || (entry.0 == current.0 && (entry.1, entry.2) < (current.1, current.2))
        };
        let take_first = match &first {
            None => true,
            Some(current) => overtakes(current),
        };
        if take_first {
            // The displaced minimum is the new second-best — but never when
            // it is the SAME (provider, model) as the new leader (that
            // would let a candidate escalate to itself).
            if let Some(old) = first {
                if !same_key_as(&old) {
                    second = Some(old);
                }
            }
            first = Some(entry);
        } else {
            let same_key = first.as_ref().map(same_key_as).unwrap_or(false);
            if !same_key {
                let take_second = match &second {
                    None => true,
                    Some(current) => overtakes(current),
                };
                if take_second {
                    second = Some(entry);
                }
            }
        }
    }
    CheapestTwo {
        cheapest: first,
        second,
    }
}

/// Score every qualified candidate: expected cost-to-success with the
/// escalation pool restricted to OTHER QUALIFIED candidates — escalation
/// can never target a model that could not serve the request itself.
///
/// Verified-outcome consult (audit items 13/14/L): when `outcomes` holds
/// per-phase history for a candidate, the legacy telemetry-prior terms are
/// REPLACED by the conservative [`WorkCostEstimate`] — expected cost to
/// VERIFIED completion = immediate cost + P(rework) x downstream spend. The
/// unmeasured rework spend fallback is the candidate's escalation cost (a
/// rework costs at least one full escalation call); measured rework spend
/// from the history dominates it once any failure with spend exists. The
/// scored success prior becomes the conservative verified-success
/// confidence (lower Wilson bound). An empty registry misses every consult,
/// so scoring is byte-identical to the pre-outcome math.
fn score_candidates<'a>(
    qualified: &[QualifiedCandidate<'a>],
    phase: RouterPhase,
    outcomes: &dyn outcomes::OutcomeStore,
) -> Vec<ScoredCandidate<'a>> {
    let two = two_cheapest_distinct(qualified);
    qualified
        .iter()
        .map(|q| {
            let d = q.descriptor;
            let escalation = two.escalation_for(
                &d.provider,
                &d.model,
                u128::from(q.call_cost_micro).saturating_mul(3),
            );
            let escalation_micro = u64::try_from(escalation).unwrap_or(u64::MAX);
            let stats = outcomes.phase_stats(&d.provider, &d.model, phase);
            match stats {
                Some(st) => {
                    let estimate =
                        work_cost_estimate(q.call_cost_micro, Some(&st), escalation_micro);
                    ScoredCandidate {
                        candidate: d,
                        expected_cost_micro: estimate.total_expected_micro,
                        expected_latency_ms: d.performance().estimated_latency_ms,
                        success_ppm: verified_success_confidence_ppm(&st),
                        call_cost_micro: q.call_cost_micro,
                        work_estimate: Some(estimate),
                    }
                }
                None => ScoredCandidate {
                    candidate: d,
                    expected_cost_micro: expected_cost_to_success(
                        q.success_ppm,
                        q.call_cost_micro,
                        escalation,
                    ),
                    expected_latency_ms: d.performance().estimated_latency_ms,
                    success_ppm: q.success_ppm,
                    call_cost_micro: q.call_cost_micro,
                    work_estimate: None,
                },
            }
        })
        .collect()
}

/// One fully qualified PRICED candidate, scored for selection. The money
/// field is a [`CostEstimate`]-derived option: `expected_cost_micro = None`
/// means the candidate's cost is Unknown (nonnumeric); numeric candidates
/// always rank ahead of unknown ones at the same tier, so "we cannot price
/// it" is never silently read as "free".
#[derive(Debug, Clone, Copy)]
pub struct PricedScoredCandidate<'a> {
    pub candidate: &'a RouteCandidate,
    /// Expected cost-to-success in microUSD, or `None` for Unknown pricing
    /// (no honest number exists).
    pub expected_cost_micro: Option<u64>,
    pub expected_latency_ms: u64,
    pub success_ppm: u32,
    /// The candidate's exact single-call cost estimate.
    pub call_cost: CostEstimate,
    /// The conservative verified work-cost estimate when history exists.
    pub work_estimate: Option<WorkCostEstimate>,
}

impl PricedScoredCandidate<'_> {
    /// Explicit total order: numeric expected cost ascending first,
    /// Unknown cost last; then success descending, latency ascending,
    /// numeric call cost ascending (LocalZero cheapest), Unknown last, and
    /// finally the deterministic (provider, model) key.
    pub fn compare(&self, other: &Self) -> std::cmp::Ordering {
        cmp_optional_cost(self.expected_cost_micro, other.expected_cost_micro)
            .then_with(|| other.success_ppm.cmp(&self.success_ppm))
            .then_with(|| self.expected_latency_ms.cmp(&other.expected_latency_ms))
            .then_with(|| cmp_optional_cost(self.call_cost.numeric(), other.call_cost.numeric()))
            .then_with(|| {
                (
                    &self.candidate.descriptor.provider,
                    &self.candidate.descriptor.model,
                )
                    .cmp(&(
                        &other.candidate.descriptor.provider,
                        &other.candidate.descriptor.model,
                    ))
            })
    }
}

/// Ascending numeric order with `None` (Unknown) always AFTER every number
/// — never treated as zero.
fn cmp_optional_cost(a: Option<u64>, b: Option<u64>) -> std::cmp::Ordering {
    match (a, b) {
        (Some(x), Some(y)) => x.cmp(&y),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    }
}

/// Score every qualified PRICED candidate. Escalation targets are the
/// cheapest OTHER numeric qualified candidates (an unknown cost can never
/// seed a numeric escalation); an Unknown candidate itself keeps
/// `expected_cost_micro = None`. Verified history is consulted through the
/// [`OutcomeStore::lookup_stats`] hierarchy for this request's task class
/// and risk bucket, so an exact/class-level record dominates the phase
/// fold and the global prior.
pub fn score_priced_candidates<'a>(
    qualified: &[QualifiedCandidate<'a>],
    candidates: &'a [RouteCandidate],
    req: &RouteRequest,
    outcomes: &dyn outcomes::OutcomeStore,
) -> Vec<PricedScoredCandidate<'a>> {
    // Numeric base costs of every qualified candidate, cheapest first —
    // the escalation pool (unknown costs never enter it).
    let mut numeric: Vec<(u64, &str, &str)> = qualified
        .iter()
        .filter_map(|q| {
            q.cost.numeric().map(|n| {
                (
                    n,
                    q.descriptor.provider.as_str(),
                    q.descriptor.model.as_str(),
                )
            })
        })
        .collect();
    numeric.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| (a.1, a.2).cmp(&(b.1, b.2))));
    numeric.dedup_by(|a, b| a.1 == b.1 && a.2 == b.2);
    let escalation_for = |provider: &str, model: &str, own: u64| -> u64 {
        numeric
            .iter()
            .find(|(_, p, m)| *p != provider || *m != model)
            .map(|(n, _, _)| *n)
            .unwrap_or_else(|| own.saturating_mul(3))
    };

    qualified
        .iter()
        .filter_map(|q| {
            let c = candidates
                .iter()
                .find(|c| std::ptr::eq(&c.descriptor, q.descriptor))?;
            let d = &c.descriptor;
            let stats = outcomes.lookup_stats(
                &d.provider,
                &d.model,
                req.phase,
                req.task_class,
                req.risk_bucket,
            );
            let (expected_cost_micro, success_ppm, work_estimate) = match q.cost.numeric() {
                Some(immediate) => {
                    let escalation = escalation_for(&d.provider, &d.model, immediate);
                    match stats {
                        Some(st) => {
                            let estimate = work_cost_estimate(immediate, Some(&st), escalation);
                            (
                                Some(estimate.total_expected_micro),
                                verified_success_confidence_ppm(&st),
                                Some(estimate),
                            )
                        }
                        None => (
                            Some(expected_cost_to_success(
                                q.success_ppm,
                                immediate,
                                u128::from(escalation),
                            )),
                            q.success_ppm,
                            None,
                        ),
                    }
                }
                // Unknown pricing: no honest expected-cost number; the
                // ladder ranks it after every numeric candidate.
                None => (None, q.success_ppm, None),
            };
            Some(PricedScoredCandidate {
                candidate: c,
                expected_cost_micro,
                expected_latency_ms: d.performance().estimated_latency_ms,
                success_ppm,
                call_cost: q.cost,
                work_estimate,
            })
        })
        .collect()
}

pub struct Router {
    pub candidates: Vec<ModelDescriptor>,
}

impl Router {
    pub fn new(candidates: Vec<ModelDescriptor>) -> Self {
        Self { candidates }
    }

    /// Plain cheapest-above-floor routing. Uses the SAME single
    /// qualification pass as every other path ([`qualified_candidates`])
    /// with the default (empty) live-health view: static
    /// [`RateLimitState::Hard`] models are excluded, live telemetry
    /// cooldowns do not exist here — they arrive through the service.
    pub fn route(&self, req: &RouteRequest, cache: &[CacheState]) -> Result<RouteDecision, String> {
        let qualified = qualified_candidates(&self.candidates, req, cache, &LiveHealth::default())
            .map_err(|f| f.route_error())?;
        // Cheapest above the floor wins; ties by latency, then candidate
        // order (deterministic). Both units are explicit: microUSD cost,
        // then milliseconds latency — never a cross-unit comparison.
        let mut best: Option<(&ModelDescriptor, u64, u64)> = None;
        for q in &qualified {
            let d = q.descriptor;
            let cost = q.call_cost_micro;
            let latency = d.performance().estimated_latency_ms;
            let better = match best {
                None => true,
                Some((_, bc, bl)) => cost < bc || (cost == bc && latency < bl),
            };
            if better {
                best = Some((d, cost, latency));
            }
        }
        let (chosen, cost, latency) = best.expect("qualified_candidates is non-empty on Ok");
        let phase_tag = serde_json::to_string(&req.phase)
            .unwrap_or_default()
            .trim_matches('"')
            .to_string();
        let quality = chosen.performance().coding_reliability;
        let floor = req.quality_floor.min(100);
        let reasoning = format!(
            "phase={phase_tag} considered={} qualified={} chosen={}/{} cost_micro={cost} latency_ms={latency} quality={quality} floor={floor}",
            self.candidates.len(),
            qualified.len(),
            chosen.provider,
            chosen.model,
        );
        Ok(RouteDecision {
            provider: chosen.provider.clone(),
            model: chosen.model.clone(),
            estimated_cost_micro: cost,
            estimated_latency_ms: latency,
            reasoning,
            considered: self.candidates.len(),
            source: chosen.source,
            // Wave-B item B: this descriptor-only path consults NO pricing
            // authority (a ModelDescriptor carries a lossy per-token
            // estimate, never a quote + authority), so the decision says so
            // — `None` — instead of inferring a snapshot from numbers. The
            // catalog-aware RouterService path stamps the entry's real
            // snapshot (see `RouterService::with_pricing`).
            pricing_snapshot: None,
        })
    }
}

// ---------------------------------------------------------------- service

/// Per (provider, model, phase) exponentially weighted observations with
/// prior blending: new models start near sane priors and a small sample can
/// never wreck a reputation.
pub struct RouterTelemetry {
    inner: std::sync::Mutex<std::collections::HashMap<(String, String, RouterPhase), Ewma>>,
    cooldown: std::sync::Mutex<std::collections::HashMap<String, std::time::Instant>>,
}

const PRIOR_SUCCESS: f64 = 0.8;
const ALPHA: f64 = 0.1;

#[derive(Clone, Copy)]
struct Ewma {
    success: f64,
    retry: f64,
    rate_limit: f64,
    latency_ms: f64,
}

impl Default for Ewma {
    fn default() -> Self {
        Self {
            success: PRIOR_SUCCESS,
            retry: 0.1,
            rate_limit: 0.0,
            latency_ms: 0.0,
        }
    }
}

impl Ewma {
    fn update(&mut self, obs: bool) {
        let target = if obs { 1.0 } else { 0.0 };
        self.success = ALPHA * target + (1.0 - ALPHA) * self.success;
    }
}

impl RouterTelemetry {
    pub fn new() -> Self {
        Self {
            inner: Default::default(),
            cooldown: Default::default(),
        }
    }

    pub fn record(
        &self,
        provider: &str,
        model: &str,
        phase: RouterPhase,
        success: bool,
        retried: bool,
        rate_limited: bool,
    ) {
        self.record_outcome(provider, model, phase, success, retried, rate_limited, 0);
    }

    /// The outcome record entry (P0-28 residuals): one settled model call
    /// with its measured latency. Same (provider, model, phase) EWMA
    /// reliability update as [`RouterTelemetry::record`] — latency rides
    /// the same observation so a caller that only ever saw `record` keeps
    /// identical priors.
    #[allow(clippy::too_many_arguments)]
    pub fn record_outcome(
        &self,
        provider: &str,
        model: &str,
        phase: RouterPhase,
        success: bool,
        retried: bool,
        rate_limited: bool,
        latency_ms: u64,
    ) {
        let mut m = self.inner.lock().unwrap();
        let e = m.entry((provider.into(), model.into(), phase)).or_default();
        e.update(success);
        let t = if retried { 1.0 } else { 0.0 };
        e.retry = ALPHA * t + (1.0 - ALPHA) * e.retry;
        if rate_limited {
            e.rate_limit = ALPHA + (1.0 - ALPHA) * e.rate_limit;
        } else {
            e.rate_limit *= 1.0 - ALPHA;
        }
        if latency_ms > 0 {
            e.latency_ms = ALPHA * latency_ms as f64 + (1.0 - ALPHA) * e.latency_ms;
        }
    }

    /// Rate-limit cooldown: providers stay Hard-excluded until `secs` pass.
    pub fn record_rate_limit(&self, provider: &str, secs: u64) {
        self.cooldown.lock().unwrap().insert(
            provider.to_string(),
            std::time::Instant::now() + std::time::Duration::from_secs(secs),
        );
    }

    pub fn cooldown_active(&self, provider: &str) -> bool {
        self.cooldown
            .lock()
            .unwrap()
            .get(provider)
            .map(|t| *t > std::time::Instant::now())
            .unwrap_or(false)
    }

    /// Blended success prior for (provider, model, phase).
    pub fn success_estimate(&self, provider: &str, model: &str, phase: RouterPhase) -> f64 {
        let m = self.inner.lock().unwrap();
        match m.get(&(provider.into(), model.into(), phase)) {
            Some(e) => {
                // Prior blend: pull toward PRIOR_SUCCESS as n is small.
                // n is implicit via variance; use a fixed light blend.
                0.7 * e.success + 0.3 * PRIOR_SUCCESS
            }
            None => PRIOR_SUCCESS,
        }
    }

    /// EWMA of the recorded settled-call latencies (0.0 when nothing was
    /// recorded through [`RouterTelemetry::record_outcome`] yet).
    pub fn avg_latency_ms(&self, provider: &str, model: &str, phase: RouterPhase) -> f64 {
        let m = self.inner.lock().unwrap();
        m.get(&(provider.into(), model.into(), phase))
            .map(|e| e.latency_ms)
            .unwrap_or(0.0)
    }

    /// Point-in-time live-health snapshot for one routing decision
    /// (audit P0-89: the qualification pass reads REAL rate-limit state,
    /// never an inert always-false helper). Cooldown expiry is evaluated
    /// once, here, so a single decision is deterministic.
    pub fn snapshot(&self) -> LiveHealth {
        let now = std::time::Instant::now();
        let cooldown = self
            .cooldown
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, until)| **until > now)
            .map(|(p, _)| p.clone())
            .collect();
        let inner = self.inner.lock().unwrap();
        let success_ppm = inner
            .iter()
            .map(|((p, m, phase), e)| {
                let blend = 0.7 * e.success + 0.3 * PRIOR_SUCCESS;
                let ppm = (blend * 1_000_000.0).round().clamp(0.0, 1_000_000.0) as u32;
                ((p.clone(), m.clone(), *phase), ppm)
            })
            .collect();
        LiveHealth {
            cooldown,
            success_ppm,
        }
    }
}

impl Default for RouterTelemetry {
    fn default() -> Self {
        Self::new()
    }
}

/// Production router: expected-cost-to-verified-success selection with
/// telemetry priors, rate-limit cooldowns and full audit strings.
///
/// `pricing` maps (provider, model) to the route-time [`PricingSnapshot`]
/// the routing graph cut from each candidate's REAL catalog row (wave-B
/// item B: authority + exact per-million quote, never inferred from the
/// descriptor's lossy estimate). Decisions over candidates absent from the
/// map carry `pricing_snapshot: None` — no pricing authority was
/// consulted — which settlement treats as unpriced (fail closed under a
/// hard cap, documented Unknown spend without one).
pub struct RouterService {
    pub router: Router,
    pub telemetry: RouterTelemetry,
    /// Catalog-cut route-time snapshots keyed (provider, model); the legacy
    /// compatibility path ([`RouterService::with_pricing`]). The PRICED
    /// path ([`RouterService::with_route_candidates`]) carries each
    /// candidate's [`PricingState`] directly instead.
    pub pricing: HashMap<(String, String), PricingSnapshot>,
    /// THE priced candidate set (pricing-path audit): descriptors paired
    /// with their catalog-resolved pricing state. When non-empty, every
    /// route runs the priced qualification/scoring path and no per-token
    /// projection is ever consulted.
    pub priced: Vec<RouteCandidate>,
    /// The pinned (provider, model) of a Pinned-mode service: `route()`
    /// then calls ONLY [`RouterService::qualify_specific`] — the pinned
    /// candidate never competes and can never be rejected because another
    /// model won.
    pub pinned: Option<(String, String)>,
    /// Verified-outcome registry (audit items 13/14/L): per-phase verified
    /// history the scoring consult reads when it exists. Every constructor
    /// defaults to an [`EmptyOutcomeStore`], so a service built without
    /// outcomes is byte-identical to the pre-outcome router; wiring builds
    /// the service through [`RouterService::with_outcomes`] /
    /// [`RouterService::with_pricing_and_outcomes`].
    pub outcomes: Arc<dyn OutcomeStore>,
}

impl RouterService {
    pub fn new(candidates: Vec<ModelDescriptor>) -> Self {
        Self::build(candidates, HashMap::new(), Arc::new(EmptyOutcomeStore))
    }

    /// Build the service over candidates AND the catalog pricing authority
    /// behind them: the routing graph passes the snapshot each candidate's
    /// catalog entry cut (exact quote / ceiling / authoritative local zero
    /// / explicit Unknown), so every decision freezes the real price
    /// lines at route time — settlement prices usage against this, never
    /// against later catalog repricing.
    pub fn with_pricing(
        candidates: Vec<ModelDescriptor>,
        pricing: HashMap<(String, String), PricingSnapshot>,
    ) -> Self {
        Self::build(candidates, pricing, Arc::new(EmptyOutcomeStore))
    }

    /// Build the service over candidates AND a verified-outcome registry
    /// (no catalog pricing map; decisions carry `pricing_snapshot: None`).
    /// Scoring consults the registry's per-phase verified stats when they
    /// exist ([`WorkCostEstimate`]); an empty registry keeps every decision
    /// byte-identical to [`RouterService::new`].
    pub fn with_outcomes(
        candidates: Vec<ModelDescriptor>,
        outcomes: Arc<dyn OutcomeStore>,
    ) -> Self {
        Self::build(candidates, HashMap::new(), outcomes)
    }

    /// The full wiring constructor: catalog pricing authority AND the
    /// verified-outcome registry in one additive build step.
    pub fn with_pricing_and_outcomes(
        candidates: Vec<ModelDescriptor>,
        pricing: HashMap<(String, String), PricingSnapshot>,
        outcomes: Arc<dyn OutcomeStore>,
    ) -> Self {
        Self::build(candidates, pricing, outcomes)
    }

    /// THE production constructor (pricing-path audit): the daemon graph
    /// passes [`RouteCandidate`]s (descriptor + catalog-resolved
    /// [`PricingState`]) and every route prices calls through the exact
    /// per-million quote, never through a per-token projection.
    pub fn with_route_candidates(
        candidates: Vec<RouteCandidate>,
        outcomes: Arc<dyn OutcomeStore>,
    ) -> Self {
        let descriptors: Vec<ModelDescriptor> =
            candidates.iter().map(|c| c.descriptor.clone()).collect();
        Self {
            router: Router::new(descriptors),
            telemetry: RouterTelemetry::new(),
            pricing: HashMap::new(),
            priced: candidates,
            pinned: None,
            outcomes,
        }
    }

    /// A Pinned-mode production service: exactly the pin's
    /// [`RouteCandidate`] set plus the pin identity, so `route()` runs ONLY
    /// the pinned qualification. The caller must have resolved
    /// (provider, model) against the registered provider already (the graph
    /// does, loudly, at build).
    pub fn with_pinned_route_candidates(
        candidates: Vec<RouteCandidate>,
        provider: impl Into<String>,
        model: impl Into<String>,
        outcomes: Arc<dyn OutcomeStore>,
    ) -> Self {
        let mut service = Self::with_route_candidates(candidates, outcomes);
        service.pinned = Some((provider.into(), model.into()));
        service
    }

    fn build(
        candidates: Vec<ModelDescriptor>,
        pricing: HashMap<(String, String), PricingSnapshot>,
        outcomes: Arc<dyn OutcomeStore>,
    ) -> Self {
        Self {
            router: Router::new(candidates),
            telemetry: RouterTelemetry::new(),
            pricing,
            priced: Vec::new(),
            pinned: None,
            outcomes,
        }
    }

    /// PINNED qualification over this service's priced candidates (pricing-
    /// path audit): checks the single pinned (provider, model) against
    /// every hard axis and returns it. No competition, no score.
    pub fn qualify_specific(
        &self,
        provider: &str,
        model: &str,
        req: &RouteRequest,
        cache: &[CacheState],
    ) -> Result<QualifiedCandidate<'_>, QualificationFailure> {
        let health = self.telemetry.snapshot();
        qualify_specific(&self.priced, provider, model, req, cache, &health)
    }

    /// Expected cost = base + P(retry)*base + (1-P(success))*escalation,
    /// where escalation = cost of the best OTHER QUALIFIED candidate
    /// (or base*3 when the candidate is the only qualified option) — or,
    /// for candidates whose per-phase VERIFIED-outcome history exists, the
    /// conservative [`WorkCostEstimate`]: expected cost to VERIFIED
    /// completion = immediate cost + P(rework) x downstream spend (measured
    /// from durable history; the escalation cost stands in until failures
    /// with measured spend exist). Selection picks the minimum EXPECTED
    /// cost by the documented [`ScoredCandidate::compare`] ladder; the
    /// decision's `estimated_cost_micro` stays the BASE cost so downstream
    /// budget math is conservative.
    ///
    /// Qualification is the single [`qualified_candidates`] pass — the
    /// same one the plain router uses — with the live telemetry snapshot
    /// (cooldowns + success priors) fed in. There is deliberately NO
    /// second loop over the unfiltered candidate list (audit P0-3): a
    /// model that fails any qualification axis can never be chosen, no
    /// matter how cheap its expected cost looks.
    pub fn route(&self, req: &RouteRequest, cache: &[CacheState]) -> Result<RouteDecision, String> {
        if let Some((provider, model)) = &self.pinned {
            return self.route_pinned(provider, model, req, cache);
        }
        if !self.priced.is_empty() {
            return self.route_priced(req, cache);
        }
        self.route_legacy(req, cache)
    }

    /// The priced production route (pricing-path audit): qualification over
    /// [`RouteCandidate`]s, exact per-million cost estimates, the
    /// outcome-hierarchy consult, and a decision carrying the candidate's
    /// frozen [`PricingState`].
    fn route_priced(
        &self,
        req: &RouteRequest,
        cache: &[CacheState],
    ) -> Result<RouteDecision, String> {
        let health = self.telemetry.snapshot();
        let qualified = qualified_priced_candidates(&self.priced, req, cache, &health)
            .map_err(|f| f.route_error())?;
        let scored = score_priced_candidates(&qualified, &self.priced, req, self.outcomes.as_ref());
        let winner = scored.iter().min_by(|a, b| a.compare(b)).ok_or_else(|| {
            "no candidate clears capability/fit filtering (missing: )".to_string()
        })?;
        let chosen = winner.candidate;
        let base = winner.call_cost.numeric().unwrap_or(0);
        let ps = f64::from(winner.success_ppm) / 1_000_000.0;
        let verified_tag = match winner.work_estimate {
            Some(est) => format!(
                " verified rework_ppm={} exp_rework_micro={} total_expected_micro={}",
                est.rework_probability_ppm, est.expected_rework_micro, est.total_expected_micro
            ),
            None => String::new(),
        };
        let cost_tag = match winner.call_cost {
            CostEstimate::Unknown => " cost=unknown",
            _ => "",
        };
        let reasoning = format!(
            "phase={:?} expected-cost chosen={}/{} base_micro={base}{} p_success={ps:.2}{}",
            req.phase, chosen.descriptor.provider, chosen.descriptor.model, cost_tag, verified_tag,
        );
        Ok(RouteDecision {
            provider: chosen.descriptor.provider.clone(),
            model: chosen.descriptor.model.clone(),
            estimated_cost_micro: base,
            estimated_latency_ms: winner.expected_latency_ms,
            reasoning,
            considered: self.priced.len(),
            source: chosen.descriptor.source,
            pricing_snapshot: Some(chosen.pricing.snapshot()),
        })
    }

    /// PINNED route (pricing-path audit): calls ONLY
    /// [`qualify_specific`] — the pin never competes, and no other
    /// candidate can reject it.
    /// Public pinned decision (pricing-path audit): qualifies ONLY the
    /// pinned (provider, model) — no competition — and builds the decision
    /// from that candidate. Used by pinned routing policies whose service
    /// was not constructed with [`Self::with_pinned_route_candidates`]
    /// (test/embedded callers): a pinned service never competes.
    pub fn route_pinned_decision(
        &self,
        provider: &str,
        model: &str,
        req: &RouteRequest,
        cache: &[CacheState],
    ) -> Result<RouteDecision, String> {
        if self.priced.is_empty() {
            // Descriptor-only (legacy/embedded) service: pin by qualifying
            // a RESTRICTED single-candidate router — never the full
            // competition (pinning must not depend on what else exists).
            let descriptor = self
                .router
                .candidates
                .iter()
                .find(|d| d.provider == provider && d.model == model)
                .cloned()
                .ok_or_else(|| format!("pinned model {provider}/{model} is not registered"))?;
            let single = Router::new(vec![descriptor]);
            let qualified =
                qualified_candidates(&single.candidates, req, cache, &self.telemetry.snapshot())
                    .map_err(|f| f.route_error())?;
            let chosen = qualified
                .first()
                .ok_or_else(|| "pinned candidate failed qualification".to_string())?;
            return Ok(RouteDecision {
                provider: provider.to_string(),
                model: model.to_string(),
                estimated_cost_micro: chosen.call_cost_micro,
                estimated_latency_ms: chosen.descriptor.performance().estimated_latency_ms,
                reasoning: format!(
                    "phase={:?} pinned chosen={}/{} qualified=1 considered=1",
                    req.phase, provider, model
                ),
                considered: 1,
                source: chosen.descriptor.source,
                pricing_snapshot: None,
            });
        }
        self.route_pinned(provider, model, req, cache)
    }

    fn route_pinned(
        &self,
        provider: &str,
        model: &str,
        req: &RouteRequest,
        cache: &[CacheState],
    ) -> Result<RouteDecision, String> {
        let q = self
            .qualify_specific(provider, model, req, cache)
            .map_err(|f| f.route_error())?;
        let cost = q.cost;
        let base = cost.numeric().unwrap_or(0);
        let reasoning = format!(
            "phase={:?} pinned chosen={}/{} cost_micro={base} qualified=1 considered=1",
            req.phase, provider, model
        );
        Ok(RouteDecision {
            provider: provider.to_string(),
            model: model.to_string(),
            estimated_cost_micro: base,
            estimated_latency_ms: self
                .priced
                .iter()
                .find(|c| c.descriptor.provider == provider && c.descriptor.model == model)
                .map(|c| c.performance().estimated_latency_ms)
                .unwrap_or(0),
            reasoning,
            considered: self.priced.len(),
            source: self
                .priced
                .iter()
                .find(|c| c.descriptor.provider == provider && c.descriptor.model == model)
                .map(|c| c.descriptor.source)
                .unwrap_or(faktor_core::model::ModelSource::ConservativeDefault),
            pricing_snapshot: Some(
                self.priced
                    .iter()
                    .find(|c| c.descriptor.provider == provider && c.descriptor.model == model)
                    .map(|c| c.pricing.snapshot())
                    .unwrap_or_else(|| PricingSnapshot::unknown(0, "pinned-missing".into())),
            ),
        })
    }

    /// The legacy descriptor-only route (compatibility constructors; no
    /// pricing authority consulted — decisions carry no snapshot).
    fn route_legacy(
        &self,
        req: &RouteRequest,
        cache: &[CacheState],
    ) -> Result<RouteDecision, String> {
        let health = self.telemetry.snapshot();
        let qualified = qualified_candidates(&self.router.candidates, req, cache, &health)
            .map_err(|f| f.route_error())?;
        let scored = score_candidates(&qualified, req.phase, self.outcomes.as_ref());
        let winner = scored
            .iter()
            .min_by(|a, b| a.compare(b))
            .expect("qualified_candidates is non-empty on Ok");
        // The plain-router pick over the SAME qualified set (audit string).
        let plain = qualified
            .iter()
            .min_by(|a, b| {
                a.call_cost_micro.cmp(&b.call_cost_micro).then_with(|| {
                    a.descriptor
                        .economics
                        .estimated_latency_ms
                        .cmp(&b.descriptor.performance().estimated_latency_ms)
                })
            })
            .expect("qualified_candidates is non-empty on Ok");
        let chosen = winner.candidate;
        let base = winner.call_cost_micro;
        let ps = f64::from(winner.success_ppm) / 1_000_000.0;
        // Verified-outcome audit (audit 13/14/L): when the winner's score
        // rode a conservative WorkCostEstimate the reasoning names its
        // conservative rework probability and rework term explicitly.
        let verified_tag = match winner.work_estimate {
            Some(est) => format!(
                " verified rework_ppm={} exp_rework_micro={} total_expected_micro={}",
                est.rework_probability_ppm, est.expected_rework_micro, est.total_expected_micro
            ),
            None => String::new(),
        };
        let reasoning = format!(
            "phase={:?} expected-cost chosen={}/{} base_micro={base} p_success={ps:.2} plain={}/{}",
            req.phase,
            chosen.provider,
            chosen.model,
            plain.descriptor.provider,
            plain.descriptor.model,
        ) + &verified_tag;
        Ok(RouteDecision {
            provider: chosen.provider.clone(),
            model: chosen.model.clone(),
            estimated_cost_micro: base,
            estimated_latency_ms: winner.expected_latency_ms,
            reasoning,
            considered: self.router.candidates.len(),
            source: chosen.source,
            // Wave-B item B: route-time price capture of the CHOSEN
            // candidate (the winner, never the plain-cheapest comparison
            // pick) — the frozen snapshot the catalog authority cut for it,
            // or None when this service was built without a pricing map
            // (no authority consulted: settlement fails closed under a
            // hard cap and records a documented Unknown spend otherwise).
            pricing_snapshot: self
                .pricing
                .get(&(chosen.provider.clone(), chosen.model.clone()))
                .cloned(),
        })
    }

    pub fn record(
        &self,
        provider: &str,
        model: &str,
        phase: RouterPhase,
        success: bool,
        retried: bool,
        rate_limited: bool,
    ) {
        if rate_limited {
            self.telemetry.record_rate_limit(provider, 30);
        }
        self.telemetry
            .record(provider, model, phase, success, retried, rate_limited);
    }

    /// Outcome record entry with the call's measured latency (P0-28): the
    /// settlement path feeds settled model calls here — attempted/resolved
    /// outcome, latency, provider/model, and the reliability signals
    /// (retried/rate-limited) — so the next route's priors see reality.
    #[allow(clippy::too_many_arguments)]
    pub fn record_outcome(
        &self,
        provider: &str,
        model: &str,
        phase: RouterPhase,
        success: bool,
        retried: bool,
        rate_limited: bool,
        latency_ms: u64,
    ) {
        if rate_limited {
            self.telemetry.record_rate_limit(provider, 30);
        }
        self.telemetry.record_outcome(
            provider,
            model,
            phase,
            success,
            retried,
            rate_limited,
            latency_ms,
        );
    }

    /// Churn-aware routing (audits 65-66 + P0-82): the same expected-cost
    /// selection as [`RouterService::route`], plus the prefix-churn risk
    /// premium. When the session's LAST recorded prefix stability (the
    /// final per-turn stability of `prefix_history`, i.e. the most recent
    /// completed turn) sits below `floor`, the decision's
    /// `estimated_cost_micro` is scaled by `(1 + churn_penalty)` with the
    /// penalty bounded in [0, 0.25] — unstable prefixes defeat
    /// provider-side caches, so the per-turn cost prediction must not
    /// pretend they hit.
    ///
    /// Cache economics under churn: provider-side prompt caches are exactly
    /// what churn invalidates. With a penalty in effect, every cache
    /// READ discount (`CacheState.cached_input_tokens`) is zeroed for the
    /// route — a churning session must be priced as if its cache misses —
    /// while `will_write_tokens` (the call's own cache write) survives.
    /// The budget axis then sees the honest uncached costs, so a candidate
    /// that was only cheap through cache reads drops out of qualification
    /// and the CHOICE can change, not just the price.
    ///
    /// The premium is a per-SESSION factor (identical for every candidate),
    /// so candidate ORDER for equally-cached candidates is unchanged; the
    /// estimate the decision returns — and therefore downstream budget
    /// math — carries the risk premium, and the audit string records
    /// `prefix_stability` and `churn_penalty` explicitly. `None`/empty
    /// history means no recorded stability: no premium is charged and the
    /// result is identical to `route`.
    pub fn route_with_prefix_stability(
        &self,
        req: &RouteRequest,
        cache: &[CacheState],
        floor: f64,
        prefix_history: Option<&[stability::TurnPrefix]>,
    ) -> Result<RouteDecision, String> {
        let Some(history) = prefix_history else {
            return self.route(req, cache);
        };
        let last = stability::turn_stabilities(history).pop();
        let Some(last_stability) = last else {
            return self.route(req, cache);
        };
        let penalty = stability::churn_penalty(last_stability, floor);
        if penalty == 0.0 {
            return self.route(req, cache);
        }
        // Churn invalidates provider-side caches: price the route with
        // every cache-read discount zeroed (the cache write this call
        // performs still stands).
        let cache_without_reads: Vec<CacheState> = cache
            .iter()
            .map(|c| CacheState {
                provider: c.provider.clone(),
                model: c.model.clone(),
                cached_input_tokens: 0,
                will_write_tokens: c.will_write_tokens,
            })
            .collect();
        let mut decision = self.route(req, &cache_without_reads)?;
        decision.estimated_cost_micro =
            stability::apply_churn_penalty(decision.estimated_cost_micro, last_stability, floor);
        decision.reasoning = format!(
            "{} prefix_stability={last_stability:.3} churn_penalty={penalty:.4}",
            decision.reasoning
        );
        Ok(decision)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faktor_core::model::{
        MicroUsdPerToken, ModelEconomics, ModelSource, RateLimitState, RiskBucket, TaskClass,
    };

    fn desc(
        provider: &str,
        model: &str,
        tools: bool,
        context: u64,
        out: u64,
        econ: ModelEconomics,
    ) -> ModelDescriptor {
        ModelDescriptor {
            provider: provider.into(),
            model: model.into(),
            context,
            max_output: out,
            tools,
            parallel_tools: true,
            reasoning: true,
            thinking: true,
            vision: false,
            structured_output: true,
            embeddings: false,
            streaming: true,
            economics: econ,
            source: ModelSource::ProviderCatalog,
        }
    }

    fn econ(input: u64, output: u64, tool: u8, code: u8) -> ModelEconomics {
        ModelEconomics {
            // Test helper arguments are microUSD per token (= dollars per
            // million tokens numerically); the typed constructor documents
            // the reading at the boundary.
            input_price_per_mtok: MicroUsdPerToken::from_dollars_per_million(input),
            output_price_per_mtok: MicroUsdPerToken::from_dollars_per_million(output),
            cache_read_price_per_mtok: MicroUsdPerToken::from(input / 5),
            cache_write_price_per_mtok: MicroUsdPerToken::from(input / 2),
            estimated_latency_ms: 500,
            tool_reliability: tool,
            reasoning_reliability: tool,
            coding_reliability: code,
            context_reliability: code,
            availability: 100,
            rate_limit_state: RateLimitState::Healthy,
        }
    }

    fn caps(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn capability_filter_rejects_tool_less_models() {
        let r = Router::new(vec![
            desc("a", "plain", false, 100_000, 4096, econ(2, 8, 80, 80)),
            desc("a", "tooly", true, 100_000, 4096, econ(4, 16, 80, 80)),
        ]);
        let req = RouteRequest {
            required_capabilities: caps(&["tools"]),
            ..Default::default()
        };
        let d = r.route(&req, &[]).unwrap();
        assert_eq!(d.model, "tooly");
    }

    #[test]
    fn empty_capability_set_errors_naming_the_blocker() {
        let r = Router::new(vec![desc(
            "a",
            "m",
            false,
            100_000,
            4096,
            econ(1, 1, 90, 90),
        )]);
        let err = r.route(&RouteRequest::default(), &[]).unwrap_err();
        assert!(err.contains("tools"), "{err}");
    }

    #[test]
    fn cheapest_above_floor_wins() {
        let r = Router::new(vec![
            desc(
                "frontier",
                "big",
                true,
                200_000,
                32_000,
                econ(15, 60, 95, 95),
            ),
            desc("cheap", "fast", true, 100_000, 4096, econ(1, 2, 80, 78)),
        ]);
        let req = RouteRequest {
            quality_floor: 70,
            ..Default::default()
        };
        let d = r.route(&req, &[]).unwrap();
        assert_eq!(d.provider, "cheap", "{}", d.reasoning);
        assert!(d.reasoning.contains("chosen=cheap/fast"));
    }

    #[test]
    fn cheap_below_floor_loses() {
        let r = Router::new(vec![
            desc(
                "frontier",
                "big",
                true,
                200_000,
                32_000,
                econ(15, 60, 95, 95),
            ),
            desc("cheap", "fast", true, 100_000, 4096, econ(1, 2, 40, 40)),
        ]);
        let req = RouteRequest {
            quality_floor: 70,
            ..Default::default()
        };
        let d = r.route(&req, &[]).unwrap();
        assert_eq!(d.model, "big");
    }

    #[test]
    fn local_zero_cost_beats_paid_when_latency_ok() {
        let local = {
            let mut m = econ(0, 0, 82, 82);
            m.estimated_latency_ms = 3000;
            m
        };
        let r = Router::new(vec![
            desc("paid", "fast", true, 100_000, 4096, econ(5, 15, 90, 90)),
            desc("ollama", "qwen3.8", true, 256_000, 8192, local),
        ]);
        let d = r.route(&RouteRequest::default(), &[]).unwrap();
        assert_eq!(d.provider, "ollama");
        assert_eq!(
            d.estimated_cost_micro, 0,
            "local zero-cost models cost zero micro"
        );
    }

    #[test]
    fn tight_latency_preference_excludes_local() {
        let local = {
            let mut m = econ(0, 0, 82, 82);
            m.estimated_latency_ms = 5000;
            m
        };
        let r = Router::new(vec![
            desc("paid", "fast", true, 100_000, 4096, econ(5, 15, 90, 90)),
            desc("ollama", "qwen3.8", true, 256_000, 8192, local),
        ]);
        let req = RouteRequest {
            latency_preference_ms: Some(800),
            ..Default::default()
        };
        let d = r.route(&req, &[]).unwrap();
        assert_eq!(d.provider, "paid");
    }

    #[test]
    fn cache_read_economics_reduces_cost() {
        let e = econ(10, 30, 90, 90);
        let r = Router::new(vec![desc("p", "m", true, 100_000, 4096, e)]);
        let req = RouteRequest {
            context_tokens: 10_000,
            ..Default::default()
        };
        let no_cache = r.route(&req, &[]).unwrap().estimated_cost_micro;
        let cache = CacheState {
            provider: "p".into(),
            model: "m".into(),
            cached_input_tokens: 9_000,
            will_write_tokens: 0,
        };
        let with_cache = r.route(&req, &[cache]).unwrap().estimated_cost_micro;
        assert!(
            with_cache < no_cache,
            "cached call must cost less: {with_cache} vs {no_cache}"
        );
    }

    #[test]
    fn decisions_are_deterministic_and_auditable() {
        let r = Router::new(vec![
            desc("a", "x", true, 100_000, 4096, econ(3, 9, 80, 80)),
            desc("b", "y", true, 100_000, 4096, econ(2, 6, 80, 80)),
            desc("c", "z", true, 100_000, 4096, econ(1, 3, 80, 80)),
        ]);
        let d1 = r.route(&RouteRequest::default(), &[]).unwrap();
        let d2 = r.route(&RouteRequest::default(), &[]).unwrap();
        assert_eq!(d1, d2);
        assert!(d1.reasoning.contains("phase=implement"));
        assert!(d1
            .reasoning
            .contains(&format!("chosen={}/{}", d1.provider, d1.model)));
    }

    #[test]
    fn hard_budget_is_not_overshot() {
        let r = Router::new(vec![
            desc("a", "x", true, 100_000, 4096, econ(100, 300, 90, 90)),
            desc("a", "cheap", true, 100_000, 4096, econ(1, 3, 90, 90)),
        ]);
        let req = RouteRequest {
            task_budget_remaining_micro: 500,
            context_tokens: 100,
            estimated_output_tokens: 10,
            ..Default::default()
        };
        // cheap: 100*1 + 10*3 = 130 micro; big: 100*100 + 10*300 = 13,000.
        let d = r.route(&req, &[]).unwrap();
        assert_eq!(d.model, "cheap");
        assert!(d.estimated_cost_micro <= 500);
    }

    #[test]
    fn micro_rounding_never_understates() {
        let e = econ(1, 1, 80, 80);
        let cost = estimated_call_cost(&e, 1, 1, 0, 0);
        assert!(cost >= 1, "tiny call costs at least 1 micro");
        let big = estimated_call_cost(&e, u64::MAX, u64::MAX, 0, 0);
        assert_eq!(big, u64::MAX, "saturating math never overflows");
    }

    // ---- RouterService / telemetry / units (audit 9-11) ----

    #[test]
    fn price_units_equivalence_is_exact() {
        // 1M tokens at $15/Mtok costs $15 = 15_000_000 microUSD, and the
        // formula tokens x price(=15 microUSD/token) yields exactly that.
        let e = econ(15, 60, 90, 90);
        assert_eq!(
            estimated_call_cost(&e, 1_000_000, 0, 0, 0),
            15_000_000,
            "$15/Mtok x 1M tokens = $15 exactly"
        );
        assert_eq!(
            estimated_call_cost(&e, 999_999, 0, 0, 0),
            14_999_985,
            "linear in tokens"
        );
        // No division: tokens x per-token-microUSD is the correct microUSD
        // total; dividing by 1e6 would understate by a factor of a million.
        assert!(
            estimated_call_cost(&e, 1, 0, 0, 0) >= 1,
            "a single token costs at least 1 micro (ceiling)"
        );
    }

    #[test]
    fn service_picks_reliable_over_flaky_cheap() {
        let reliable = {
            let mut e = econ(12, 48, 92, 92);
            e.estimated_latency_ms = 600;
            e
        };
        // Flaky is 3.2x cheaper per call but near-zero success: with
        // once-then-escalate economics, reliability wins only when
        // p_success < cost_flaky/cost_reliable ~ 0.31 at these prices.
        let flaky_e = {
            let mut e = econ(10, 30, 84, 84);
            e.estimated_latency_ms = 300;
            e
        };
        let svc = RouterService::new(vec![
            desc("reliable", "r", true, 512_000, 64_000, reliable),
            desc("flaky", "f", true, 512_000, 64_000, flaky_e),
        ]);
        // The flaky model catastrophically fails 24 of its last 25 phase
        // observations (12.5% record would still win under pure retry
        // economics; near-zero success must not).
        for _ in 0..80 {
            svc.record("flaky", "f", RouterPhase::Implement, false, false, false);
        }
        svc.record("flaky", "f", RouterPhase::Implement, true, false, false);
        let d = svc
            .route(
                &RouteRequest {
                    phase: RouterPhase::Implement,
                    context_tokens: 40_000,
                    estimated_output_tokens: 4_000,
                    quality_floor: 70,
                    ..Default::default()
                },
                &[],
            )
            .unwrap();
        assert_eq!(
            d.provider, "reliable",
            "expected-cost must favor reliability over raw price: {}",
            d.reasoning
        );
        assert!(d.reasoning.contains("expected-cost"));
    }

    #[test]
    fn static_hard_provider_is_excluded() {
        let mut hard = econ(5, 15, 95, 95);
        hard.rate_limit_state = RateLimitState::Hard;
        let svc = RouterService::new(vec![
            desc("a", "m", true, 512_000, 64_000, econ(5, 15, 95, 95)),
            desc("b", "n", true, 512_000, 64_000, hard),
        ]);
        let d = svc.route(&RouteRequest::default(), &[]).unwrap();
        assert_eq!(d.provider, "a", "Hard provider must be excluded");
    }

    #[test]
    fn runtime_rate_limit_routes_around_after_cooldown() {
        // Both healthy; a live 429 puts the winner into cooldown so the
        // next route picks the other provider (same behavior, different
        // model context preserved where possible).
        let svc = RouterService::new(vec![
            desc("a", "m", true, 512_000, 64_000, econ(5, 15, 95, 95)),
            desc("b", "n", true, 512_000, 64_000, econ(5, 15, 95, 95)),
        ]);
        let d1 = svc.route(&RouteRequest::default(), &[]).unwrap();
        // Both cost the same: deterministic tie-break -> 'a'.
        assert_eq!(d1.provider, "a");
        svc.record("a", "m", RouterPhase::Implement, false, false, true);
        assert!(svc.telemetry.cooldown_active("a"));
        let d2 = svc.route(&RouteRequest::default(), &[]).unwrap();
        assert_eq!(d2.provider, "b", "cooldown routes around the limiter");
        assert!(d2.reasoning.contains("expected-cost"));
    }

    // ---- prefix-cache stability integration (audits 65-66) ----

    fn prefix_turn(id: u64, bytes: &[u8]) -> stability::TurnPrefix {
        // FNV-1a-derived deterministic content digest (router has no hash
        // dep): identical bytes must hash identically regardless of turn id.
        let mut h = [0u8; 32];
        let mut acc = 0xcbf29ce484222325u64;
        for &b in bytes {
            acc ^= u64::from(b);
            acc = acc.wrapping_mul(0x100000001b3);
        }
        h[..8].copy_from_slice(&acc.to_le_bytes());
        h[8..16].copy_from_slice(&acc.wrapping_mul(31).to_le_bytes());
        h[16..24].copy_from_slice(&acc.wrapping_mul(97).to_le_bytes());
        h[24..].copy_from_slice(&acc.wrapping_mul(211).to_le_bytes());
        stability::TurnPrefix::new(id, h, bytes.len() as u32)
    }

    #[test]
    fn churn_premium_scales_estimate_and_stays_auditable_and_deterministic() {
        let svc = RouterService::new(vec![desc(
            "p",
            "m",
            true,
            100_000,
            4096,
            econ(1, 1, 90, 90),
        )]);
        let req = RouteRequest {
            context_tokens: 100,
            estimated_output_tokens: 10,
            ..Default::default()
        };
        let base = svc.route(&req, &[]).unwrap();
        assert_eq!(base.estimated_cost_micro, 110);
        // Stable history (identical prefixes): no premium, decision equal.
        let stable: Vec<u8> = vec![b's'; 40];
        let healthy = [
            prefix_turn(1, &stable),
            prefix_turn(2, &stable),
            prefix_turn(3, &stable),
        ];
        let d = svc
            .route_with_prefix_stability(&req, &[], 0.8, Some(&healthy))
            .unwrap();
        assert_eq!(d, base);
        // Churning history: same-length rewritten prefix -> stability 0.0 ->
        // penalty 0.25 -> estimate *= 1.25, rounded up (110 + ceil(27.5) = 138).
        let rewritten: Vec<u8> = vec![b'r'; 40];
        assert_eq!(stable.len(), rewritten.len());
        assert_ne!(
            prefix_turn(2, &rewritten).prefix_hash,
            prefix_turn(1, &stable).prefix_hash
        );
        let churny = [prefix_turn(1, &stable), prefix_turn(2, &rewritten)];
        let d = svc
            .route_with_prefix_stability(&req, &[], 0.8, Some(&churny))
            .unwrap();
        assert_eq!(d.estimated_cost_micro, 138);
        assert!(
            d.reasoning.contains("prefix_stability=0.000"),
            "{}",
            d.reasoning
        );
        assert!(
            d.reasoning.contains("churn_penalty=0.2500"),
            "{}",
            d.reasoning
        );
        // Deterministic: identical history reproduces the decision exactly.
        let d2 = svc
            .route_with_prefix_stability(&req, &[], 0.8, Some(&churny))
            .unwrap();
        assert_eq!(d, d2);
        // No history / empty history: identical to the plain route.
        assert_eq!(
            svc.route_with_prefix_stability(&req, &[], 0.8, None)
                .unwrap(),
            base
        );
        assert_eq!(
            svc.route_with_prefix_stability(&req, &[], 0.8, Some(&[]))
                .unwrap(),
            base
        );
        // A floor of 0 disables the premium (nothing is below it).
        let d = svc
            .route_with_prefix_stability(&req, &[], 0.0, Some(&churny))
            .unwrap();
        assert_eq!(d, base);
        // Mid-churn stability scales proportionally and stays in (0, 0.25].
        let mut grown = stable.clone();
        grown.extend_from_slice(&vec![b'z'; 300]);
        let partial = [prefix_turn(1, &stable), prefix_turn(2, &stable), {
            let mut tp = prefix_turn(3, &grown);
            tp.prefix_tokens = 300; // growth ratio 40/300 -> very low
            tp
        }];
        let d = svc
            .route_with_prefix_stability(&req, &[], 0.8, Some(&partial))
            .unwrap();
        assert!(d.estimated_cost_micro > base.estimated_cost_micro);
        assert!(d.estimated_cost_micro < 138);
        assert!(d.reasoning.contains("churn_penalty=0."));
    }

    #[test]
    fn churn_premium_never_leaks_into_plain_routing() {
        // The plain surface is byte-stable under the new machinery.
        let svc = RouterService::new(vec![desc(
            "p",
            "m",
            true,
            100_000,
            4096,
            econ(1, 1, 90, 90),
        )]);
        let a = svc.route(&RouteRequest::default(), &[]).unwrap();
        let b = svc.route(&RouteRequest::default(), &[]).unwrap();
        assert_eq!(a, b);
        assert!(!a.reasoning.contains("churn_penalty"));
    }

    #[test]
    fn churn_zeroes_cache_read_discounts_and_can_flip_the_choice() {
        // P0-82 adversarial: a candidate whose cheapness comes ONLY from
        // provider-side cache reads wins a stable session and LOSES a
        // churning one (stability 0.3 < floor 0.8) — the route is priced
        // as if its cache misses and the choice flips to the flat-price
        // candidate, never just a scaled price tag.
        let mut x_econ = econ(10, 1, 90, 90);
        x_econ.cache_read_price_per_mtok = MicroUsdPerToken::from(1);
        x_econ.cache_write_price_per_mtok = MicroUsdPerToken::from(1);
        x_econ.estimated_latency_ms = 200;
        let mut y_econ = econ(5, 1, 90, 90);
        y_econ.estimated_latency_ms = 300;
        let svc = RouterService::new(vec![
            desc("cache", "cx", true, 100_000, 4096, x_econ),
            desc("flat", "fy", true, 100_000, 4096, y_econ),
        ]);
        let req = RouteRequest {
            context_tokens: 1000,
            estimated_output_tokens: 10,
            ..Default::default()
        };
        // The session's own last turn wrote a cache; with a STABLE prefix
        // 900 of the 1000 input tokens hit it.
        let cache = [CacheState {
            provider: "cache".into(),
            model: "cx".into(),
            cached_input_tokens: 900,
            will_write_tokens: 1000,
        }];
        let stable = [
            prefix_turn(1, b"stable-prefix-bytes"),
            prefix_turn(2, b"stable-prefix-bytes"),
        ];
        let d = svc
            .route_with_prefix_stability(&req, &cache, 0.8, Some(&stable))
            .unwrap();
        assert_eq!(
            (d.provider.as_str(), d.model.as_str()),
            ("cache", "cx"),
            "cache-read discount wins a stable session: {}",
            d.reasoning
        );
        // Churning history: the last pair REWRITES to a longer, different
        // prefix (40 -> 130 tokens): growth scores 40/130 ~ 0.154 (below
        // the 0.8 floor) — the churn premium applies.
        let churny = [
            prefix_turn(1, b"stable-prefix-bytes"),
            prefix_turn(2, b"stable-prefix-bytes"),
            {
                let mut tp = prefix_turn(
                    3,
                    b"stable-prefix-bytes-grown-longer-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                );
                tp.prefix_tokens = 130;
                tp
            },
        ];
        let d2 = svc
            .route_with_prefix_stability(&req, &cache, 0.8, Some(&churny))
            .unwrap();
        assert_eq!(
            (d2.provider.as_str(), d2.model.as_str()),
            ("flat", "fy"),
            "a churning session must be priced without its cache reads: {}",
            d2.reasoning
        );
        assert!(
            d2.reasoning.contains("prefix_stability=0.146"),
            "{}",
            d2.reasoning
        );
        assert!(d2.reasoning.contains("churn_penalty="), "{}", d2.reasoning);
        // The chosen decision still carries the churn premium on its cost
        // (flat base 5010 * (1 + penalty) with penalty ~0.204).
        let flat_base = 1000 * 5 + 10;
        assert!(
            d2.estimated_cost_micro > flat_base,
            "premium must inflate the estimate: {} > {flat_base}",
            d2.estimated_cost_micro
        );
        // Determinism.
        let d3 = svc
            .route_with_prefix_stability(&req, &cache, 0.8, Some(&churny))
            .unwrap();
        assert_eq!(d2, d3);
    }

    #[test]
    fn segment_observations_replace_the_binary_rule_and_legacy_rows_stay_identical() {
        // Audits 45/82 end at the routing consult: when the durable rows
        // carry segment observations, cache economics consume the MEASURED
        // longest stable prefix — a same-length volatile-tail rewrite that
        // the binary digest pair would score 0.0 scores its true coverage
        // (1.0), and a rewrite inside the cacheable prefix scores exactly
        // the surviving share. Legacy rows keep the binary verdict.
        let svc = RouterService::new(vec![desc(
            "p",
            "m",
            true,
            100_000,
            4096,
            econ(1, 1, 90, 90),
        )]);
        let req = RouteRequest {
            context_tokens: 100,
            estimated_output_tokens: 10,
            ..Default::default()
        };
        let base = svc.route(&req, &[]).unwrap();
        assert_eq!(base.estimated_cost_micro, 110);

        let tokens = [10u64, 20, 30, 40, 50, 5, 4, 3];
        let ids: Vec<u8> = (0..8).collect();
        let json = |ids: &[u8]| {
            let hashes: Vec<String> = ids.iter().map(|&i| format!("{i:02x}").repeat(32)).collect();
            serde_json::json!({
                "segment_hashes": hashes,
                "segment_token_counts": tokens,
                "cache_read_tokens": 0u64,
            })
            .to_string()
        };
        let history = |second_ids: &[u8]| {
            stability::TurnPrefix::history_from_persisted(vec![
                (1, [1u8; 32], 150, Some(json(&ids))),
                (2, [2u8; 32], 150, Some(json(second_ids))),
            ])
        };

        // Volatile-only change (first volatile segment): fully covered. The
        // binary hashes DIFFER with equal token counts — the old rule would
        // score 0.0 and charge the premium; the measured rule scores 1.0.
        let mut volatile = ids.clone();
        volatile[5] = 90;
        let h = history(&volatile);
        assert_eq!(h[1].stable_leading_tokens, Some(150));
        let d = svc
            .route_with_prefix_stability(&req, &[], 0.8, Some(&h))
            .unwrap();
        assert_eq!(d, base, "measured full coverage must not charge churn");
        assert!(!d.reasoning.contains("churn_penalty"));

        // Rewrite inside the cacheable prefix (segment 2): 10+20 of the
        // 150 prefix tokens survive -> stability 0.2 -> premium applies.
        let mut mid = ids.clone();
        mid[2] = 91;
        let h = history(&mid);
        assert_eq!(h[1].stable_leading_tokens, Some(30));
        let d = svc
            .route_with_prefix_stability(&req, &[], 0.8, Some(&h))
            .unwrap();
        assert!(
            d.reasoning.contains("prefix_stability=0.200"),
            "{}",
            d.reasoning
        );
        assert!(d.reasoning.contains("churn_penalty="), "{}", d.reasoning);
        assert_eq!(d.estimated_cost_micro, 131); // ceil(110 * 1.1875)

        // Legacy fallback: no segment payloads anywhere -> byte-identical
        // to the pre-v19 binary history (same-length rewrite -> 0.0 -> 138).
        let legacy = vec![prefix_turn(1, b"same-length-old"), {
            let mut tp = prefix_turn(2, b"same-length-new");
            tp.prefix_tokens = prefix_turn(1, b"same-length-old").prefix_tokens;
            tp
        }];
        let legacy_decision = svc
            .route_with_prefix_stability(&req, &[], 0.8, Some(&legacy))
            .unwrap();
        assert!(legacy_decision.reasoning.contains("prefix_stability=0.000"));
        assert_eq!(legacy_decision.estimated_cost_micro, 138);
        let no_payloads = stability::TurnPrefix::history_from_persisted(vec![
            (1, legacy[0].prefix_hash, legacy[0].prefix_tokens, None),
            (2, legacy[1].prefix_hash, legacy[1].prefix_tokens, None),
        ]);
        assert_eq!(
            svc.route_with_prefix_stability(&req, &[], 0.8, Some(&no_payloads))
                .unwrap(),
            legacy_decision,
            "rows without the v19 column route byte-identically"
        );
    }

    #[test]
    fn telemetry_outcome_records_latency_priors_and_isolated_reliability() {
        // The outcome record entry: settled calls feed success + latency
        // per (provider, model, phase); reliability priors move ONLY for
        // the failing instance pair and rate-limit cooldown touches only
        // the limited provider.
        let svc = RouterService::new(vec![
            desc("a", "am", true, 100_000, 4096, econ(5, 15, 95, 95)),
            desc("b", "bm", true, 100_000, 4096, econ(5, 15, 95, 95)),
        ]);
        let before = svc.telemetry.snapshot();
        let a_before = before.success_ppm("a", "am", RouterPhase::Implement);
        // N settled calls against (a, am): 9 failures + 1 success with
        // latency; (b, bm) untouched.
        for _ in 0..9 {
            svc.record_outcome("a", "am", RouterPhase::Implement, false, false, false, 400);
        }
        svc.record_outcome("a", "am", RouterPhase::Implement, true, false, false, 200);
        let after = svc.telemetry.snapshot();
        let a_after = after.success_ppm("a", "am", RouterPhase::Implement);
        assert!(
            a_after < a_before,
            "(a, am) reliability must fall: {a_before} -> {a_after}"
        );
        // Every OTHER pair keeps its prior exactly (the map only gains the
        // observed key; unobserved keys still resolve to DEFAULT_SUCCESS_PPM).
        let b_after = after.success_ppm("b", "bm", RouterPhase::Implement);
        let b_before = before.success_ppm("b", "bm", RouterPhase::Implement);
        assert_eq!(a_before, 800_000);
        assert_eq!(b_before, 800_000);
        assert_eq!(b_after, 800_000, "unobserved pair untouched");
        let review_after = after.success_ppm("a", "am", RouterPhase::Review);
        assert_eq!(review_after, 800_000, "other phases untouched");
        // Latency EWMA tracked the recorded outcome.
        let avg = svc
            .telemetry
            .avg_latency_ms("a", "am", RouterPhase::Implement);
        assert!(avg > 200.0 && avg < 400.0, "latency ewma: {avg}");
        // Rate limit: cooldown ONLY for the limited provider.
        svc.record_outcome("a", "am", RouterPhase::Implement, false, false, true, 500);
        assert!(svc.telemetry.cooldown_active("a"));
        assert!(!svc.telemetry.cooldown_active("b"));
    }

    // ==================================================================
    // Single-authoritative-qualification (audit P0-3/P0-89), typed-price
    // units (P0-5) and the explicit scoring ladder (P0-4). Adversarial:
    // every axis tries to smuggle an unqualified ultra-cheap model into a
    // decision, and every path must refuse it.
    // ==================================================================

    fn tiny_req(names: &[&str], floor: u8) -> RouteRequest {
        RouteRequest {
            required_capabilities: caps(names),
            context_tokens: 2,
            estimated_output_tokens: 1,
            quality_floor: floor,
            ..Default::default()
        }
    }

    fn good_desc() -> ModelDescriptor {
        desc("good", "gm", true, 100_000, 4096, econ(300, 1, 90, 90))
    }

    fn cheap_bad_desc() -> ModelDescriptor {
        // 2 tokens x 1 microUSD/token: a ~2-microUSD competitor.
        desc("bad", "bm", true, 100_000, 4096, econ(1, 1, 90, 90))
    }

    /// Both selection paths must choose `good`; `bad` (whatever axis it
    /// violates) must never be chosen — the single qualification pass is
    /// the ONLY filter, so cost can never re-admit a filtered candidate.
    fn assert_paths_never_select_bad(
        good: &ModelDescriptor,
        bad: &ModelDescriptor,
        req: &RouteRequest,
        label: &str,
    ) {
        let plain = Router::new(vec![good.clone(), bad.clone()]);
        let d = plain
            .route(req, &[])
            .unwrap_or_else(|e| panic!("{label}: plain route failed: {e}"));
        assert_eq!(
            (d.provider.as_str(), d.model.as_str()),
            (good.provider.as_str(), good.model.as_str()),
            "{label}: plain route selected an unqualified candidate: {}",
            d.reasoning
        );
        let svc = RouterService::new(vec![good.clone(), bad.clone()]);
        let d = svc
            .route(req, &[])
            .unwrap_or_else(|e| panic!("{label}: service route failed: {e}"));
        assert_eq!(
            (d.provider.as_str(), d.model.as_str()),
            (good.provider.as_str(), good.model.as_str()),
            "{label}: service route selected an unqualified candidate: {}",
            d.reasoning
        );
        assert!(
            d.reasoning
                .contains(&format!("chosen={}/{}", good.provider, good.model)),
            "{label}: reasoning must name the qualified winner: {}",
            d.reasoning
        );
    }

    #[test]
    fn unqualified_ultra_cheap_candidates_never_win_any_axis() {
        let tools_req = tiny_req(&["tools", "streaming"], 60);
        // 1. Missing the tools capability (the canonical P0-3 shape: the
        //    cheapest eligible model cannot serve the request at all).
        let bad = {
            let mut d = cheap_bad_desc();
            d.tools = false;
            d
        };
        assert_paths_never_select_bad(&good_desc(), &bad, &tools_req, "tools axis");
        // 2. Missing parallel tools.
        let bad = {
            let mut d = cheap_bad_desc();
            d.parallel_tools = false;
            d
        };
        assert_paths_never_select_bad(
            &good_desc(),
            &bad,
            &tiny_req(&["parallel_tools"], 60),
            "parallel_tools axis",
        );
        // 3. Context window too small for the request.
        let bad = {
            let mut d = cheap_bad_desc();
            d.context = 1;
            d
        };
        assert_paths_never_select_bad(&good_desc(), &bad, &tools_req, "context axis");
        // 4. Max output too small.
        let bad = {
            let mut d = cheap_bad_desc();
            d.max_output = 0;
            d
        };
        assert_paths_never_select_bad(&good_desc(), &bad, &tools_req, "output axis");
        // 5. Below the quality floor (cheap garbage must lose to floor).
        let good = desc("good", "gm", true, 100_000, 4096, econ(300, 1, 85, 85));
        let bad = {
            let mut d = cheap_bad_desc();
            d.economics = econ(1, 1, 40, 40);
            d
        };
        assert_paths_never_select_bad(
            &good,
            &bad,
            &tiny_req(&["tools", "streaming"], 80),
            "quality-floor axis",
        );
        // 6. Exceeds the latency preference.
        let bad = {
            let mut d = cheap_bad_desc();
            d.economics.estimated_latency_ms = 5000;
            d
        };
        let req = RouteRequest {
            latency_preference_ms: Some(700),
            ..tiny_req(&["tools", "streaming"], 60)
        };
        assert_paths_never_select_bad(&good_desc(), &bad, &req, "latency axis");
        // 7. Over the hard request budget (only the expensive one may fit).
        let good = desc("good", "gm", true, 100_000, 4096, econ(1, 1, 90, 90));
        let bad = desc("bad", "bm", true, 100_000, 4096, econ(300, 1, 90, 90));
        let req = RouteRequest {
            task_budget_remaining_micro: 3,
            ..tiny_req(&["tools", "streaming"], 60)
        };
        assert_paths_never_select_bad(&good, &bad, &req, "budget axis");
        // 8. Static Hard rate-limit state — excluded by BOTH paths (the
        //    plain router now consumes the same qualification pass).
        let bad = {
            let mut d = cheap_bad_desc();
            d.economics.rate_limit_state = RateLimitState::Hard;
            d
        };
        assert_paths_never_select_bad(&good_desc(), &bad, &tools_req, "hard-rate-limit axis");
    }

    #[test]
    fn soft_rate_limit_is_not_a_qualification_blocker() {
        // Only Hard state and an ACTIVE cooldown exclude; Soft is an
        // advisory signal and must not disable a cheap qualified model.
        let mut soft = cheap_bad_desc();
        soft.economics.rate_limit_state = RateLimitState::Soft;
        let good = desc("good", "gm", true, 100_000, 4096, econ(300, 1, 90, 90));
        let req = tiny_req(&["tools", "streaming"], 60);
        let d = RouterService::new(vec![good, soft.clone()])
            .route(&req, &[])
            .unwrap();
        assert_eq!(
            (d.provider.as_str(), d.model.as_str()),
            (soft.provider.as_str(), soft.model.as_str()),
            "Soft rate-limit state is not a Hard exclusion: {}",
            d.reasoning
        );
    }

    #[test]
    fn live_cooldown_excludes_only_via_the_service_health_view() {
        // P0-89: live cooldown state is REAL state fed into the single
        // qualification pass, not the old inert always-false helper.
        let svc = RouterService::new(vec![good_desc(), cheap_bad_desc()]);
        let req = tiny_req(&["tools", "streaming"], 60);
        // The cheap bad provider takes a 429: it enters the telemetry
        // cooldown window and the NEXT service decision must route around.
        svc.record("bad", "bm", RouterPhase::Implement, false, false, true);
        assert!(svc.telemetry.cooldown_active("bad"));
        let d = svc.route(&req, &[]).unwrap();
        assert_eq!(
            d.provider, "good",
            "cooldown must route around bad: {}",
            d.reasoning
        );
        // The plain path has no live view by design: it legitimately still
        // sees bad (static axes only). This documents the boundary.
        let d = Router::new(vec![good_desc(), cheap_bad_desc()])
            .route(&req, &[])
            .unwrap();
        assert_eq!(d.provider, "bad", "plain routing carries no live cooldown");
    }

    #[test]
    fn escalation_pool_contains_only_qualified_candidates_after_two_strikes() {
        // A = cheap qualified workhorse that is failing; B = ultra-cheap
        // (1 microUSD/token) but WITHOUT tools — the old service loop would
        // happily escalate against B's price and could select it. C = the
        // expensive qualified frontier model.
        let flaky = {
            let mut e = econ(50, 1, 90, 90);
            e.estimated_latency_ms = 500;
            e
        };
        let mut b_no_tools = desc("bad", "bm", true, 100_000, 4096, econ(1, 1, 99, 99));
        b_no_tools.tools = false;
        let premium = {
            let mut e = econ(400, 1, 95, 95);
            e.estimated_latency_ms = 700;
            e
        };
        let candidates = vec![
            desc("a", "am", true, 100_000, 4096, flaky),
            b_no_tools,
            desc("c", "cm", true, 100_000, 4096, premium),
        ];
        let svc = RouterService::new(candidates.clone());
        let req = RouteRequest {
            // 1 input token, 0 output: per-token prices are the base costs.
            context_tokens: 1,
            estimated_output_tokens: 0,
            ..tiny_req(&["tools", "streaming"], 60)
        };
        // Two strikes plus: the cheap workhorse keeps failing.
        for _ in 0..30 {
            svc.record("a", "am", RouterPhase::Implement, false, false, false);
        }
        // The single qualification pass admits exactly the tooled pair.
        let health = svc.telemetry.snapshot();
        let qualified = qualified_candidates(&candidates, &req, &[], &health).unwrap();
        assert_eq!(qualified.len(), 2, "only a and c qualify: {qualified:?}");
        assert!(
            qualified.iter().all(|q| q.descriptor.provider != "bad"),
            "the unqualified cheap model must not appear in the qualified vector"
        );
        // Escalation math must reference C (400 micro), never B (1 micro):
        // two_cheapest_distinct over the QUALIFIED set.
        let two = two_cheapest_distinct(&qualified);
        assert_eq!(
            two.cheapest.map(|(c, p, _)| (c, p.to_string())),
            Some((50, "a".into()))
        );
        assert_eq!(
            two.second.map(|(c, p, _)| (c, p.to_string())),
            Some((400, "c".into()))
        );
        let a_scored = score_candidates(&qualified, RouterPhase::Implement, &EmptyOutcomeStore)
            .into_iter()
            .find(|s| s.candidate.provider == "a")
            .expect("a is qualified");
        let esc_c = 400u128;
        assert_eq!(
            a_scored.expected_cost_micro,
            expected_cost_to_success(a_scored.success_ppm, 50, esc_c),
            "expected cost must escalate to the qualified C, never to B"
        );
        // Routing under the strikes: A stays the expected-cost winner and
        // B is never chosen, on every re-route (service and plain).
        for _ in 0..3 {
            let d = svc.route(&req, &[]).unwrap();
            assert_ne!(
                d.provider, "bad",
                "never escalate INTO the unqualified model"
            );
            assert!(d.provider == "a" || d.provider == "c");
        }
        let d = Router::new(candidates).route(&req, &[]).unwrap();
        assert_eq!(d.provider, "a");
    }

    #[test]
    fn maximum_quality_path_consumes_only_qualified_candidates() {
        // A 95 floor: only the frontier model qualifies; the 99-quality
        // tool-less ultra-cheap model and the 90-quality workhorse must
        // both lose on BOTH paths.
        let mut b_no_tools = desc("bad", "bm", true, 100_000, 4096, econ(1, 1, 99, 99));
        b_no_tools.tools = false;
        let candidates = vec![
            desc("a", "am", true, 100_000, 4096, econ(50, 1, 90, 90)),
            b_no_tools,
            desc("c", "cm", true, 100_000, 4096, econ(400, 1, 95, 95)),
        ];
        let req = tiny_req(&["tools", "streaming"], 95);
        let d = Router::new(candidates.clone()).route(&req, &[]).unwrap();
        assert_eq!(
            (d.provider.as_str(), d.model.as_str()),
            ("c", "cm"),
            "{}",
            d.reasoning
        );
        let d = RouterService::new(candidates).route(&req, &[]).unwrap();
        assert_eq!(
            (d.provider.as_str(), d.model.as_str()),
            ("c", "cm"),
            "{}",
            d.reasoning
        );
    }

    #[test]
    fn qualified_candidates_stages_its_denials() {
        // Direct API: denials carry staged counts so each path reproduces
        // its own denial message; qualification is a single pass.
        let tools_req = tiny_req(&["tools", "streaming"], 60);
        let good = good_desc();
        let mut no_tools = cheap_bad_desc();
        no_tools.tools = false;
        let mut too_small_ctx = cheap_bad_desc();
        too_small_ctx.context = 1;
        let mut low_quality = cheap_bad_desc();
        low_quality.economics = econ(1, 1, 40, 40);
        let mut hard = cheap_bad_desc();
        hard.economics.rate_limit_state = RateLimitState::Hard;
        // Stage 1: capability/fit empties the set -> the capability error,
        // naming exactly the capability NO health-passing candidate has.
        let f = qualified_candidates(&[no_tools.clone()], &tools_req, &[], &LiveHealth::default())
            .unwrap_err();
        assert_eq!(f.fit_survivors, 0, "no candidate clears capability/fit");
        assert_eq!(f.quality_survivors, 0);
        assert_eq!(f.unsupported_capabilities, vec!["tools".to_string()]);
        assert!(f.route_error().contains("missing: tools"));
        // Stage 2: fit passes but the floor empties the set -> quality error.
        let f = qualified_candidates(
            &[too_small_ctx, low_quality.clone()],
            &tiny_req(&["tools", "streaming"], 80),
            &[],
            &LiveHealth::default(),
        )
        .unwrap_err();
        assert_eq!(f.fit_survivors, 1);
        assert_eq!(f.quality_survivors, 0);
        assert!(f.route_error().contains("quality floor 80"));
        // Stage 3: fit and quality pass but budget/latency empties the set
        // -> budget error, with both earlier counts still positive.
        let over_budget_req = RouteRequest {
            task_budget_remaining_micro: 1,
            ..tools_req.clone()
        };
        let f = qualified_candidates(&[good], &over_budget_req, &[], &LiveHealth::default())
            .unwrap_err();
        assert_eq!(f.fit_survivors, 1);
        assert_eq!(f.quality_survivors, 1);
        assert!(f.route_error().contains("budget"));
        // Health empties the set BEFORE capability counting: a Hard model
        // does not "support" a capability it could never serve on.
        let f = qualified_candidates(&[hard], &tools_req, &[], &LiveHealth::default()).unwrap_err();
        assert_eq!(f.fit_survivors, 0);
        assert_eq!(
            f.unsupported_capabilities,
            vec!["tools".to_string(), "streaming".to_string()]
        );
    }

    #[test]
    fn plain_denials_name_the_same_stages_as_the_service() {
        let tools_req = tiny_req(&["tools", "streaming"], 60);
        let mut no_tools = cheap_bad_desc();
        no_tools.tools = false;
        let r = Router::new(vec![no_tools.clone()]);
        let err = r.route(&tools_req, &[]).unwrap_err();
        assert!(err.contains("missing: tools"), "{err}");
        let mut low = cheap_bad_desc();
        low.economics = econ(1, 1, 40, 40);
        let r = Router::new(vec![low]);
        let err = r
            .route(&tiny_req(&["tools", "streaming"], 80), &[])
            .unwrap_err();
        assert!(err.contains("quality floor 80"), "{err}");
        let r = Router::new(vec![good_desc()]);
        let err = r
            .route(
                &RouteRequest {
                    task_budget_remaining_micro: 1,
                    ..tools_req.clone()
                },
                &[],
            )
            .unwrap_err();
        assert!(err.contains("budget"), "denial must name the budget: {err}");
        // The service maps the SAME staged failure to the same strings.
        let svc = RouterService::new(vec![no_tools]);
        let err = svc.route(&tools_req, &[]).unwrap_err();
        assert!(err.contains("missing: tools"), "{err}");
    }

    // ---- P0-4: the explicit scoring ladder, with documented units ----

    fn scored<'a>(
        d: &'a ModelDescriptor,
        expected_cost_micro: u64,
        expected_latency_ms: u64,
        success_ppm: u32,
        call_cost_micro: u64,
    ) -> ScoredCandidate<'a> {
        ScoredCandidate {
            candidate: d,
            expected_cost_micro,
            expected_latency_ms,
            success_ppm,
            call_cost_micro,
            work_estimate: None,
        }
    }

    #[test]
    fn scoring_ladder_is_explicit_and_unit_honest() {
        let a = desc("a", "x", true, 100_000, 4096, econ(300, 1, 90, 90));
        let b = desc("b", "y", true, 100_000, 4096, econ(300, 1, 90, 90));
        // (1) Expected cost-to-success asc dominates everything.
        let sa = scored(&a, 100, 500, 800_000, 300);
        let sb = scored(&b, 101, 1, 900_000, 1);
        assert_eq!(sa.compare(&sb), std::cmp::Ordering::Less);
        // (2) Equal expected cost: higher success ppm wins.
        let sa = scored(&a, 100, 500, 900_000, 300);
        let sb = scored(&b, 100, 500, 500_000, 300);
        assert_eq!(sa.compare(&sb), std::cmp::Ordering::Less);
        // (3) Equal expected + ppm: lower latency (ms) wins.
        let sa = scored(&a, 100, 250, 800_000, 300);
        let sb = scored(&b, 100, 900, 800_000, 300);
        assert_eq!(sa.compare(&sb), std::cmp::Ordering::Less);
        // (4) Equal expected + ppm + latency: lower call cost (microUSD) wins.
        let sa = scored(&a, 100, 500, 800_000, 200);
        let sb = scored(&b, 100, 500, 800_000, 40);
        assert_eq!(sa.compare(&sb), std::cmp::Ordering::Greater);
        // (5) Fully equal scores: (provider, model) lex asc is the
        // deterministic final key.
        let sa = scored(&a, 100, 500, 800_000, 300);
        let sb = scored(&b, 100, 500, 800_000, 300);
        assert_eq!(sa.compare(&sb), std::cmp::Ordering::Less, "a/x < b/y");
        assert_eq!(sb.compare(&sa), std::cmp::Ordering::Greater);
        let sa2 = scored(&a, 100, 500, 800_000, 300);
        assert_eq!(sa.compare(&sa2), std::cmp::Ordering::Equal);
    }

    #[test]
    fn regression_old_tie_break_compared_cost_to_latency() {
        // Old RouterService tie-break: expected costs equal -> better iff
        // `base < previous_candidate.estimated_latency_ms` — comparing a
        // microUSD cost against MILLISECONDS. A and B share a 40-micro base
        // (identical economics, latency differs), hence exactly equal
        // expected costs; the old rule then judged B "better" because
        // `base_B (40 micro) < latency_A (100 ms)`. The explicit ladder
        // must descend deterministically to latency: A (100 ms) over B.
        let mut e_a = econ(40, 1, 90, 90);
        e_a.estimated_latency_ms = 100;
        let mut e_b = econ(40, 1, 90, 90);
        e_b.estimated_latency_ms = 1000;
        let candidates = vec![
            desc("a", "am", true, 100_000, 4096, e_a),
            desc("b", "bm", true, 100_000, 4096, e_b),
        ];
        let req = RouteRequest {
            context_tokens: 1,
            estimated_output_tokens: 0,
            ..tiny_req(&["tools", "streaming"], 60)
        };
        let d = RouterService::new(candidates).route(&req, &[]).unwrap();
        assert_eq!(
            (d.provider.as_str(), d.model.as_str()),
            ("a", "am"),
            "the ladder picks lower latency on equal expected cost: {}",
            d.reasoning
        );
        // Equal-everything candidates resolve by (provider, model) key.
        let candidates = vec![
            desc("z", "zm", true, 100_000, 4096, econ(300, 1, 90, 90)),
            desc("a", "am", true, 100_000, 4096, econ(300, 1, 90, 90)),
        ];
        let d = RouterService::new(candidates).route(&req, &[]).unwrap();
        assert_eq!(
            d.provider, "a",
            "deterministic key tie-break: {}",
            d.reasoning
        );
        // Equal-cost, differing-latency: repeated routes stay deterministic.
        let repeated = RouterService::new(vec![
            desc("a", "am", true, 100_000, 4096, {
                let mut e = econ(40, 1, 90, 90);
                e.estimated_latency_ms = 100;
                e
            }),
            desc("b", "bm", true, 100_000, 4096, {
                let mut e = econ(40, 1, 90, 90);
                e.estimated_latency_ms = 1000;
                e
            }),
        ]);
        for _ in 0..3 {
            let d = repeated
                .route(
                    &RouteRequest {
                        context_tokens: 1,
                        estimated_output_tokens: 0,
                        ..tiny_req(&["tools", "streaming"], 60)
                    },
                    &[],
                )
                .unwrap();
            assert_eq!(d.provider, "a", "repeated routes stay deterministic");
        }
    }

    #[test]
    fn expected_cost_is_integer_exact_and_saturating() {
        // P(fail)/2 terms round UP (never understate), sums saturate.
        let e = expected_cost_to_success(800_000, 1_000, 3_000u128);
        // base 1000 + ceil(1000*0.2/2)=100 + ceil(3000*0.1)=300.
        assert_eq!(e, 1_400);
        assert_eq!(expected_cost_to_success(1_000_000, 500, 1u128), 500);
        assert_eq!(
            expected_cost_to_success(0, u64::MAX, u128::MAX),
            u64::MAX,
            "hostile magnitudes saturate, never panic"
        );
        assert_eq!(
            expected_cost_to_success(500_000, 7, 7u128),
            // ceil(7*0.25)=2 and ceil(7*0.25)=2 -> 11
            11
        );
    }

    // ---- P0-5: typed price units flow through routing untouched ----

    #[test]
    fn wrapper_arithmetic_equals_the_legacy_interpretation() {
        // The typed field keeps the exact legacy numbers: tokens x price
        // with no division, no floats, no rounding mid-route.
        let e = econ(15, 60, 90, 90);
        assert_eq!(e.input_price_per_mtok.0, 15);
        assert_eq!(e.output_price_per_mtok.0, 60);
        assert_eq!(
            estimated_call_cost(&e, 1_000_000, 0, 0, 0),
            15_000_000,
            "$15/Mtok x 1M tokens = $15 exactly (typed fields)"
        );
        // The conversion constructor used by catalog ingestion documents
        // the dollars-per-million reading at the boundary.
        let from_usd = MicroUsdPerToken::from_dollars_per_million(15);
        assert_eq!(from_usd.saturating_mul(1_000_000), 15_000_000);
    }

    #[test]
    fn budget_denial_never_confuses_units() {
        // A latency value can no longer BE a price: the fields reject the
        // mix at compile time (see the core compile_fail doc example), and
        // at runtime the budget axis only ever compares microUSD to
        // microUSD.
        let mut e = econ(100, 100, 90, 90);
        e.estimated_latency_ms = 5; // ms, unrelated to the 100-micro price
        let r = Router::new(vec![desc("p", "m", true, 100_000, 4096, e)]);
        let req = RouteRequest {
            context_tokens: 1,
            estimated_output_tokens: 0,
            task_budget_remaining_micro: 99,
            ..Default::default()
        };
        let err = r.route(&req, &[]).unwrap_err();
        assert!(err.contains("budget"), "{err}");
    }

    // ==================================================================
    // Verified-outcome learning integration (audit items 13/14/L): scoring
    // consults conservative WorkCostEstimates only when verified history
    // exists; an empty registry is byte-identical to the legacy router.
    // ==================================================================

    fn implement_req(tokens_in: u64, tokens_out: u64, floor: u8) -> RouteRequest {
        RouteRequest {
            phase: RouterPhase::Implement,
            required_capabilities: caps(&["tools", "streaming"]),
            context_tokens: tokens_in,
            estimated_output_tokens: tokens_out,
            quality_floor: floor,
            task_budget_remaining_micro: 0,
            latency_preference_ms: None,
            task_class: TaskClass::Medium,
            risk_bucket: RiskBucket::Low,
        }
    }

    fn sample(verified_success: bool, rework: u64) -> OutcomeSample {
        OutcomeSample {
            verified_success,
            rework_cost_micro: rework,
            rework_turns: 1,
        }
    }

    #[test]
    fn empty_outcome_registry_is_byte_identical_to_legacy_routing() {
        // The additive guarantee: a service built with an empty outcome
        // store makes EXACTLY the decisions of the pre-outcome constructors
        // (same candidate set, same request, including the audit strings).
        let candidates = vec![
            desc("cheap", "fast", true, 512_000, 64_000, econ(1, 3, 82, 82)),
            desc("f1", "big", true, 512_000, 64_000, econ(15, 60, 95, 95)),
        ];
        let legacy = RouterService::new(candidates.clone());
        let empty_registry = RouterService::with_outcomes(candidates, Arc::new(EmptyOutcomeStore));
        let req = implement_req(40_000, 6_000, 80);
        for _ in 0..3 {
            let a = legacy.route(&req, &[]).unwrap();
            let b = empty_registry.route(&req, &[]).unwrap();
            assert_eq!(a, b, "empty outcome registry must not change decisions");
            assert!(!a.reasoning.contains("rework_ppm"), "{}", a.reasoning);
        }
    }

    #[test]
    fn verified_rework_history_flips_economy_to_the_strong_model() {
        // Cheap model: base 58_000 micro on the request, observed 45%
        // rework over 200 verified samples (each failure's downstream spend
        // measured at 2.4M). Strong model: base 960_000, observed 6%
        // rework over 100 samples (measured 960k per failure). Totals favor
        // the STRONG model after the verified history exists; the legacy
        // expected-cost prior (no stats) favors the cheap model.
        let cheap = desc("e1", "cheap", true, 512_000, 64_000, econ(1, 3, 82, 82));
        let strong = desc("f1", "big", true, 512_000, 64_000, econ(15, 60, 95, 95));
        let candidates = vec![cheap, strong];
        let req = implement_req(40_000, 6_000, 80);
        let before = RouterService::new(candidates.clone())
            .route(&req, &[])
            .unwrap();
        assert_eq!(
            (before.provider.as_str(), before.model.as_str()),
            ("e1", "cheap"),
            "without verified history the cheap model's expected cost wins: {}",
            before.reasoning
        );
        // Verified samples under several class/risk keys of the Implement
        // phase: the route consult folds them per phase.
        let store = MemoryOutcomeStore::new();
        for (class, bucket, ok, fail) in [
            (TaskClass::Medium, RiskBucket::Low, 55u64, 45u64),
            (TaskClass::Hard, RiskBucket::High, 55, 45),
        ] {
            for _ in 0..ok {
                store.append_sample(
                    &OutcomeKey {
                        provider: "e1".into(),
                        model: "cheap".into(),
                        phase: RouterPhase::Implement,
                        task_class: class,
                        risk_bucket: bucket,
                    },
                    sample(true, 0),
                );
            }
            for _ in 0..fail {
                store.append_sample(
                    &OutcomeKey {
                        provider: "e1".into(),
                        model: "cheap".into(),
                        phase: RouterPhase::Implement,
                        task_class: class,
                        risk_bucket: bucket,
                    },
                    sample(false, 2_400_000),
                );
            }
        }
        for _ in 0..94 {
            store.append_sample(
                &OutcomeKey {
                    provider: "f1".into(),
                    model: "big".into(),
                    phase: RouterPhase::Implement,
                    task_class: TaskClass::Medium,
                    risk_bucket: RiskBucket::Low,
                },
                sample(true, 0),
            );
        }
        for _ in 0..6 {
            store.append_sample(
                &OutcomeKey {
                    provider: "f1".into(),
                    model: "big".into(),
                    phase: RouterPhase::Implement,
                    task_class: TaskClass::Medium,
                    risk_bucket: RiskBucket::Low,
                },
                sample(false, 960_000),
            );
        }
        let svc = RouterService::with_outcomes(candidates, Arc::new(store));
        let after = svc.route(&req, &[]).unwrap();
        assert_eq!(
            (after.provider.as_str(), after.model.as_str()),
            ("f1", "big"),
            "verified rework history must flip Economy to the strong model: {}",
            after.reasoning
        );
        assert!(
            after.reasoning.contains("rework_ppm="),
            "the audit string must name the verified estimate: {}",
            after.reasoning
        );
        // Deterministic over the same history.
        let again = svc.route(&req, &[]).unwrap();
        assert_eq!(after, again);
    }

    #[test]
    fn two_verified_successes_stay_conservative_and_never_license_cheap() {
        // The audit's small-sample rule at the ROUTER level: a candidate
        // with TWO verified first-pass successes must NOT be treated as a
        // zero-rework license. Trusting 2/2 as excellent would make the
        // 100k-micro candidate win (100k < the fresh candidate's ~175k
        // legacy expected cost); the conservative upper rework bound keeps
        // its expected verified cost at ~200k, so the fresh 150k candidate
        // wins.
        let two_time = desc("p1", "proven2x", true, 512_000, 64_000, {
            let mut e = econ(4, 30, 88, 88);
            e.estimated_latency_ms = 500;
            e
        });
        let fresh = desc("p2", "fresh", true, 512_000, 64_000, {
            let mut e = econ(10, 25, 88, 88);
            e.estimated_latency_ms = 500;
            e
        });
        let candidates = vec![two_time.clone(), fresh.clone()];
        let req = implement_req(10_000, 2_000, 60);
        // 100k base (10k x 4 + 2k x 30) vs 150k base (10k x 10 + 2k x 25).
        assert_eq!(base_call_cost(&two_time, &req, &[]), 100_000);
        assert_eq!(base_call_cost(&fresh, &req, &[]), 150_000);
        let store = MemoryOutcomeStore::new();
        for _ in 0..2 {
            store.append_sample(
                &OutcomeKey {
                    provider: "p1".into(),
                    model: "proven2x".into(),
                    phase: RouterPhase::Implement,
                    task_class: TaskClass::Medium,
                    risk_bucket: RiskBucket::Low,
                },
                sample(true, 0),
            );
        }
        let two_stats = store
            .phase_stats("p1", "proven2x", RouterPhase::Implement)
            .unwrap();
        assert!(
            verified_success_confidence_ppm(&two_stats) < outcomes::EXCELLENT_CONFIDENCE_PPM,
            "2/2 must stay below the excellent bar"
        );
        let svc = RouterService::with_outcomes(candidates, Arc::new(store));
        let d = svc.route(&req, &[]).unwrap();
        assert_eq!(
            (d.provider.as_str(), d.model.as_str()),
            ("p2", "fresh"),
            "two verified successes must NOT license the cheaper candidate: {}",
            d.reasoning
        );
        // Prove the licensing arithmetic: under a zero-rework reading the
        // 2/2 candidate would win (100k base), but the conservative Wilson
        // upper rework bound keeps its expected verified cost above the
        // fresh candidate's legacy expected cost.
        let health = LiveHealth::default();
        let qualified = qualified_candidates(&svc.router.candidates, &req, &[], &health).unwrap();
        let scored = score_candidates(&qualified, req.phase, svc.outcomes.as_ref());
        let proven = scored
            .iter()
            .find(|s| s.candidate.provider == "p1")
            .unwrap();
        let fresh_scored = scored
            .iter()
            .find(|s| s.candidate.provider == "p2")
            .unwrap();
        assert_eq!(proven.call_cost_micro, 100_000);
        assert_eq!(proven.expected_cost_micro, 200_001);
        assert_eq!(fresh_scored.expected_cost_micro, 175_000);
        assert!(
            proven.expected_cost_micro > fresh_scored.expected_cost_micro,
            "2/2 must not clear the excellent bar: proven {} vs fresh {}",
            proven.expected_cost_micro,
            fresh_scored.expected_cost_micro
        );
    }

    #[test]
    fn failed_verification_is_never_learned_as_a_success_and_hostile_stats_saturate() {
        // Verified-only attribution at the registry level: three "model
        // said done" calls that failed verification record three FAILURES
        // with their rework, and zero successes — routing over that history
        // keeps the strong candidate's cost honest (never zeroed).
        let strong = desc("f1", "big", true, 512_000, 64_000, econ(15, 60, 95, 95));
        let cheap = desc("e1", "cheap", true, 512_000, 64_000, econ(1, 3, 82, 82));
        let store = MemoryOutcomeStore::new();
        for _ in 0..3 {
            store.append_sample(
                &OutcomeKey {
                    provider: "f1".into(),
                    model: "big".into(),
                    phase: RouterPhase::Implement,
                    task_class: TaskClass::Hard,
                    risk_bucket: RiskBucket::High,
                },
                sample(false, 960_000),
            );
        }
        let st = store
            .stats(&OutcomeKey {
                provider: "f1".into(),
                model: "big".into(),
                phase: RouterPhase::Implement,
                task_class: TaskClass::Hard,
                risk_bucket: RiskBucket::High,
            })
            .unwrap();
        assert_eq!(st.successes_first_pass, 0, "no success may be learned");
        assert_eq!(st.failures_first_pass, 3);
        assert_eq!(st.rework_cost_micro_sum, 3 * 960_000);
        // Hostile store rows must saturate, never panic or understate: a
        // registry that reports near-maximum rework spend keeps the
        // candidate's estimate at the saturation ceiling.
        struct Hostile;
        impl OutcomeStore for Hostile {
            fn append_sample(&self, _key: &OutcomeKey, _s: OutcomeSample) {}
            fn stats(&self, _key: &OutcomeKey) -> Option<VerifiedOutcomeStats> {
                Some(VerifiedOutcomeStats {
                    failures_first_pass: 1,
                    rework_cost_micro_sum: u64::MAX,
                    sample_count: 1,
                    ..Default::default()
                })
            }
            fn phase_stats(
                &self,
                _provider: &str,
                _model: &str,
                _phase: RouterPhase,
            ) -> Option<VerifiedOutcomeStats> {
                Some(VerifiedOutcomeStats {
                    failures_first_pass: 1,
                    rework_cost_micro_sum: u64::MAX,
                    sample_count: 1,
                    ..Default::default()
                })
            }
        }
        let svc = RouterService::with_outcomes(vec![cheap, strong], Arc::new(Hostile));
        let d = svc.route(&implement_req(40_000, 6_000, 80), &[]).unwrap();
        assert_eq!(
            d.estimated_cost_micro, 58_000,
            "decision cost stays the base"
        );
        assert!(
            d.reasoning.contains("total_expected_micro="),
            "saturating hostile stats must not panic and must stay auditable: {}",
            d.reasoning
        );
    }

    // ------------------------------------------ candidate-specific sizing

    /// The two-candidate fixture of the audit: the SAME prompt is 10k tokens
    /// under A (20k window) and 13k under B (12k window) — the cheap
    /// qualification pass admits both, but B's REAL footprint overflows its
    /// own window and must be filtered before final selection.
    #[test]
    fn real_footprint_filter_drops_a_candidate_that_overflows_its_window() {
        let a = desc("a", "am", true, 20_000, 4096, econ(1, 2, 90, 90));
        let b = desc("b", "bm", true, 12_000, 4096, econ(1, 2, 90, 90));
        let qualified = vec![
            QualifiedCandidate {
                descriptor: &a,
                call_cost_micro: 10,
                success_ppm: 900_000,
                cost: CostEstimate::Conservative(10),
            },
            QualifiedCandidate {
                descriptor: &b,
                call_cost_micro: 10,
                success_ppm: 900_000,
                cost: CostEstimate::Conservative(10),
            },
        ];
        let sizes: std::collections::HashMap<&str, u64> =
            [("am", 10_000u64), ("bm", 13_000u64)].into_iter().collect();
        let mut sized_calls = 0usize;
        let survivors = size_candidates_top_k(&qualified, 8, |d| {
            sized_calls += 1;
            (sizes[d.model.as_str()], true)
        });
        assert_eq!(
            sized_calls, 2,
            "each seriously-considered candidate sized once"
        );
        assert_eq!(
            survivors.len(),
            1,
            "B's 13k footprint overflows its 12k window"
        );
        assert_eq!(survivors[0].candidate.model, "am");
        assert_eq!(survivors[0].input_tokens, 10_000);
        assert!(survivors[0].exact);
    }

    #[test]
    fn sizing_work_is_bounded_to_the_top_k_and_order_is_preserved() {
        let models: Vec<ModelDescriptor> = (0..20)
            .map(|i| {
                desc(
                    "p",
                    &format!("m{i:02}"),
                    true,
                    100_000,
                    4096,
                    econ(1, 2, 90, 90),
                )
            })
            .collect();
        let qualified: Vec<QualifiedCandidate<'_>> = models
            .iter()
            .map(|d| QualifiedCandidate {
                descriptor: d,
                call_cost_micro: 10,
                success_ppm: 900_000,
                cost: CostEstimate::Conservative(10),
            })
            .collect();
        let mut seen: Vec<String> = Vec::new();
        let survivors = size_candidates_top_k(&qualified, 64, |d| {
            seen.push(d.model.clone());
            (1_000, true)
        });
        assert_eq!(
            seen.len(),
            MAX_SIZED_CANDIDATES,
            "the sizer must never run more than the hard top-K cap"
        );
        assert_eq!(survivors.len(), MAX_SIZED_CANDIDATES);
        assert_eq!(survivors[0].candidate.model, "m00");
        assert_eq!(survivors[MAX_SIZED_CANDIDATES - 1].candidate.model, "m07");
        // A caller asking for fewer sizes fewer (bound includes 0).
        assert!(size_candidates_top_k(&qualified, 0, |_| (1, true)).is_empty());
        let three = size_candidates_top_k(&qualified, 3, |_| (1, true));
        assert_eq!(three.len(), 3);
    }

    #[test]
    fn every_survivor_that_is_not_sized_is_never_returned_unsized() {
        // An EMPTY survivor set is the honest end state when the top-K all
        // overflow; the helper must not fall through to an unsized pick.
        let a = desc("a", "am", true, 5_000, 4096, econ(1, 2, 90, 90));
        let qualified = vec![QualifiedCandidate {
            descriptor: &a,
            call_cost_micro: 10,
            success_ppm: 900_000,
            cost: CostEstimate::Conservative(10),
        }];
        let survivors = size_candidates_top_k(&qualified, 8, |_| (5_001, false));
        assert!(survivors.is_empty());
        // Exactly at the window boundary fits (<=, not <).
        let at_boundary = size_candidates_top_k(&qualified, 8, |_| (5_000, true));
        assert_eq!(at_boundary.len(), 1);
    }

    // ==================================================================
    // Priced-path integration (pricing-path audit): RouteCandidate +
    // PricingState as the router's priced unit, the exact per-million quote
    // as the only money math, and unknown as a NON-numeric state.
    // ==================================================================

    use faktor_core::model::{MicroUsdPerMillionTokens, PriceQuote};

    fn perf_desc(
        provider: &str,
        model: &str,
        tools: bool,
        context: u64,
        out: u64,
        coding: u8,
        context_rel: u8,
    ) -> ModelDescriptor {
        let mut e = econ(0, 0, coding, coding);
        e.context_reliability = context_rel;
        desc(provider, model, tools, context, out, e)
    }

    fn candidate(d: ModelDescriptor, pricing: PricingState) -> RouteCandidate {
        RouteCandidate::new(d, pricing)
    }

    fn exact_quote(input: u64, output: u64) -> PriceQuote {
        PriceQuote {
            input: MicroUsdPerMillionTokens(input),
            output: MicroUsdPerMillionTokens(output),
            cache_read: MicroUsdPerMillionTokens(0),
            cache_write: MicroUsdPerMillionTokens(0),
        }
    }

    fn exact_pricing(input: u64, output: u64) -> PricingState {
        PricingState::Known(PricingSnapshot::exact(
            exact_quote(input, output),
            1,
            "priced-test".into(),
        ))
    }

    fn priced_req(context_tokens: u64, output_tokens: u64, floor: u8) -> RouteRequest {
        RouteRequest {
            phase: RouterPhase::Implement,
            required_capabilities: caps(&["tools"]),
            context_tokens,
            estimated_output_tokens: output_tokens,
            quality_floor: floor,
            task_budget_remaining_micro: 0,
            latency_preference_ms: None,
            task_class: TaskClass::Medium,
            risk_bucket: RiskBucket::Low,
        }
    }

    #[test]
    fn router_uses_exact_sub_dollar_quotes() {
        // $0.10/M vs $0.15/M input are BOTH 1 microUSD by the legacy
        // per-token ceil projection — indistinguishable. The exact
        // per-million quote prices them 100_000 vs 150_000 micro on 1M
        // tokens, so the router MUST rank them distinctly through the
        // exact math, never through the lossy projection.
        let cheap = candidate(
            perf_desc("a", "cheap", true, 2_000_000, 64_000, 90, 90),
            exact_pricing(100_000, 0),
        );
        let rich = candidate(
            perf_desc("b", "rich", true, 2_000_000, 64_000, 90, 90),
            exact_pricing(150_000, 0),
        );
        let svc = RouterService::with_route_candidates(
            vec![cheap.clone(), rich.clone()],
            Arc::new(EmptyOutcomeStore),
        );
        let req = priced_req(1_000_000, 0, 50);
        let decision = svc.route(&req, &[]).unwrap();
        assert_eq!(
            (decision.provider.as_str(), decision.model.as_str()),
            ("a", "cheap"),
            "exact quotes must rank $0.10/M below $0.15/M: {}",
            decision.reasoning
        );
        assert_eq!(decision.estimated_cost_micro, 100_000);
        let snap = decision
            .pricing_snapshot
            .expect("priced decisions carry snapshots");
        assert_eq!(snap.authority, PriceAuthority::Exact);
        assert_eq!(snap.settle_cost(1_000_000, 0, 0, 0), Some(100_000));

        // The sub-dollar regime at small token counts: 100 micro vs 150
        // micro — the projection would round both to 1 micro/token.
        let small = priced_req(1_000, 0, 50);
        assert_eq!(
            CostEstimate::from_state(&cheap.pricing, TokenUsage::new(1_000, 0, 0, 0)),
            CostEstimate::Known(100)
        );
        assert_eq!(
            CostEstimate::from_state(&rich.pricing, TokenUsage::new(1_000, 0, 0, 0)),
            CostEstimate::Known(150)
        );
        let d = svc.route(&small, &[]).unwrap();
        assert_eq!(d.provider, "a");
        assert_eq!(d.estimated_cost_micro, 100, "exact sub-dollar quote");
    }

    /// Seeded numerical-recipe LCG (identical on every platform).
    struct QuoteGen {
        state: u64,
    }

    impl QuoteGen {
        fn new(seed: u64) -> Self {
            Self {
                state: seed
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(0x1234_5678_9ABC_DEF0),
            }
        }

        fn next(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.state >> 33) as u32 as u64
        }

        fn range(&mut self, lo: u64, hi_exclusive: u64) -> u64 {
            assert!(hi_exclusive > lo);
            lo + self.next() % (hi_exclusive - lo)
        }
    }

    #[test]
    fn router_budget_matches_quote_oracle() {
        // 10,000 randomized cases: the router's budget admission is checked
        // against an ORACLE re-implementation of the exact per-million quote
        // (`PriceQuote::quote_cost_micro`). exact == cap admits, a cap
        // above the exact cost admits, a cap below it rejects; Unknown is
        // admitted ONLY without a cap; LocalZero is exactly zero.
        let mut g = QuoteGen::new(0xB0D6_E7A5);
        let mut examined = 0usize;
        for _ in 0..10_000 {
            let quote = PriceQuote {
                input: MicroUsdPerMillionTokens(g.range(0, 5_000_000)),
                output: MicroUsdPerMillionTokens(g.range(0, 20_000_000)),
                cache_read: MicroUsdPerMillionTokens(g.range(0, 1_000_000)),
                cache_write: MicroUsdPerMillionTokens(g.range(0, 1_000_000)),
            };
            let ctx = g.range(0, 200_000);
            let out = g.range(0, 20_000);
            let usage = TokenUsage::new(ctx, 0, 0, out);
            let oracle = quote.quote_cost_micro(usage);
            let pricing = match g.range(0, 4) {
                0 => PricingState::Known(PricingSnapshot::exact(quote, 1, "oracle".into())),
                1 => PricingState::ConservativeCeiling(PricingSnapshot::conservative_ceiling(
                    quote,
                    1,
                    "oracle".into(),
                )),
                2 => PricingState::LocalZero,
                _ => PricingState::Unknown,
            };
            let c = candidate(
                perf_desc("p", "m", true, 1_000_000, 64_000, 90, 90),
                pricing.clone(),
            );
            let admit = |budget: u64| {
                qualified_priced_candidates(
                    std::slice::from_ref(&c),
                    &RouteRequest {
                        task_budget_remaining_micro: budget,
                        ..priced_req(ctx, out, 50)
                    },
                    &[],
                    &LiveHealth::default(),
                )
                .is_ok()
            };
            match &pricing {
                PricingState::Unknown => {
                    assert!(admit(0), "Unknown is admitted while no cap exists");
                    assert!(
                        !admit(1),
                        "Unknown fails closed the moment a hard cap exists"
                    );
                }
                PricingState::LocalZero => {
                    assert!(admit(1), "the authoritative zero fits every positive cap");
                    let q = qualified_priced_candidates(
                        std::slice::from_ref(&c),
                        &RouteRequest {
                            task_budget_remaining_micro: 1,
                            ..priced_req(ctx, out, 50)
                        },
                        &[],
                        &LiveHealth::default(),
                    )
                    .unwrap();
                    assert_eq!(q[0].cost, CostEstimate::LocalZero);
                    assert_eq!(q[0].cost.numeric(), Some(0));
                }
                _ => {
                    let estimate = CostEstimate::from_state(&pricing, usage);
                    assert_eq!(
                        estimate.numeric(),
                        Some(oracle),
                        "router estimate diverged from the quote oracle"
                    );
                    assert!(admit(oracle), "cost == cap must admit");
                    assert!(admit(oracle + 1), "cap above the exact cost must admit");
                    if oracle > 0 {
                        assert!(
                            !admit(oracle - 1),
                            "a cap one micro below the exact cost must reject"
                        );
                    }
                    examined += 1;
                }
            }
        }
        assert!(examined > 4_000, "the case mix must exercise real quotes");
    }

    #[test]
    fn maximum_quality_unknown_without_cap() {
        // The best model is Unknown-priced: without a hard cap it must not
        // be excluded (quality decides), and under a hard cap it must fail
        // closed rather than be treated as free.
        let best = candidate(
            perf_desc("best", "bm", true, 256_000, 64_000, 95, 95),
            PricingState::Unknown,
        );
        let affordable = candidate(
            perf_desc("ok", "om", true, 256_000, 64_000, 80, 80),
            exact_pricing(1_000_000, 3_000_000),
        );
        let svc = RouterService::with_route_candidates(
            vec![best.clone(), affordable.clone()],
            Arc::new(EmptyOutcomeStore),
        );
        let req = priced_req(1_000, 100, 90);
        let both = [best.clone(), affordable.clone()];
        let qualified =
            qualified_priced_candidates(&both, &req, &[], &LiveHealth::default()).unwrap();
        assert_eq!(qualified.len(), 1, "only the 95-quality model clears 90");
        assert_eq!(qualified[0].cost, CostEstimate::Unknown);
        let d = svc.route(&req, &[]).unwrap();
        assert_eq!((d.provider.as_str(), d.model.as_str()), ("best", "bm"));
        let snap = d.pricing_snapshot.expect("decision carries the snapshot");
        assert_eq!(snap.authority, PriceAuthority::Unknown);
        assert_eq!(snap.settle_cost(1_000, 0, 0, 100), None);
        assert!(
            d.reasoning.contains("cost=unknown"),
            "the decision must SAY it is unpriced: {}",
            d.reasoning
        );

        // Hard cap: the Unknown best is excluded; the affordable tier clears
        // the floor and serves.
        let mut capped = req.clone();
        capped.quality_floor = 50;
        capped.task_budget_remaining_micro = 1_000_000_000;
        let d = svc.route(&capped, &[]).unwrap();
        assert_eq!(
            (d.provider.as_str(), d.model.as_str()),
            ("ok", "om"),
            "under a hard cap the unpriced best is not treated as free: {}",
            d.reasoning
        );

        // Hard cap and NO affordable above-floor candidate: typed refusal.
        let only_best =
            RouterService::with_route_candidates(vec![best], Arc::new(EmptyOutcomeStore));
        let mut capped_high_floor = capped.clone();
        capped_high_floor.quality_floor = 90;
        assert!(
            only_best.route(&capped_high_floor, &[]).is_err(),
            "an unpriced candidate must never fit a hard cap"
        );
    }

    #[test]
    fn pinned_never_competes() {
        // Pinned routing calls ONLY qualify_specific: the pin wins even
        // when the free evaluation would pick a much cheaper model.
        let pin = candidate(
            perf_desc("pin", "pm", true, 256_000, 64_000, 90, 90),
            exact_pricing(50_000_000, 150_000_000),
        );
        let cheap = candidate(
            perf_desc("cheap", "cm", true, 256_000, 64_000, 90, 90),
            exact_pricing(1_000, 2_000),
        );
        let free = RouterService::with_route_candidates(
            vec![pin.clone(), cheap.clone()],
            Arc::new(EmptyOutcomeStore),
        );
        let req = priced_req(1_000, 100, 50);
        assert_eq!(free.route(&req, &[]).unwrap().provider, "cheap");

        let pinned = RouterService::with_pinned_route_candidates(
            vec![pin, cheap],
            "pin",
            "pm",
            Arc::new(EmptyOutcomeStore),
        );
        let d = pinned.route(&req, &[]).unwrap();
        assert_eq!((d.provider.as_str(), d.model.as_str()), ("pin", "pm"));
        let snap = d.pricing_snapshot.expect("pin snapshots");
        assert_eq!(snap.authority, PriceAuthority::Exact);
        assert_eq!(snap.settle_cost(1_000_000, 0, 0, 0), Some(50_000_000));
        assert!(
            d.reasoning.contains("pinned"),
            "the pinned decision is auditable: {}",
            d.reasoning
        );
    }

    #[test]
    fn pin_lacking_tools_is_a_no_capable_model_refusal() {
        // Pinned qualification checks the ONE pin; a pin missing a required
        // capability stages the capability failure (NoCapableModel), never
        // a silent substitution or a lowered requirement.
        let pin = candidate(
            perf_desc("pin", "pm", false, 256_000, 64_000, 95, 95),
            PricingState::Unknown,
        );
        let req = priced_req(1_000, 100, 50);
        let failure = qualify_specific(
            std::slice::from_ref(&pin),
            "pin",
            "pm",
            &req,
            &[],
            &LiveHealth::default(),
        )
        .expect_err("a tool-less pin cannot serve a tools request");
        assert_eq!(failure.fit_survivors, 0);
        assert!(
            failure
                .unsupported_capabilities
                .iter()
                .any(|c| c == "tools"),
            "{failure:?}"
        );
        assert!(
            failure
                .route_error()
                .contains("no candidate clears capability/fit"),
            "{}",
            failure.route_error()
        );
        let svc = RouterService::with_pinned_route_candidates(
            vec![pin],
            "pin",
            "pm",
            Arc::new(EmptyOutcomeStore),
        );
        let err = svc.route(&req, &[]).unwrap_err();
        assert!(err.contains("no candidate clears capability/fit"), "{err}");
    }

    #[test]
    fn pin_unknown_is_accepted_without_cap_and_refused_under_a_hard_cap() {
        let pin = candidate(
            perf_desc("pin", "pm", true, 256_000, 64_000, 90, 90),
            PricingState::Unknown,
        );
        let svc = RouterService::with_pinned_route_candidates(
            vec![pin],
            "pin",
            "pm",
            Arc::new(EmptyOutcomeStore),
        );
        let req = priced_req(1_000, 100, 50);
        let d = svc.route(&req, &[]).unwrap();
        assert_eq!((d.provider.as_str(), d.model.as_str()), ("pin", "pm"));
        assert_eq!(
            d.pricing_snapshot.expect("snapshot").authority,
            PriceAuthority::Unknown
        );
        // A hard cap cannot reserve an unknown price: the pin fails closed
        // (no fabricated number), never "treated as free".
        let mut capped = req;
        capped.task_budget_remaining_micro = 10_000_000_000;
        let err = svc.route(&capped, &[]).unwrap_err();
        assert!(
            err.contains("budget/latency"),
            "unknown under a hard cap is a budget refusal: {err}"
        );
    }

    #[test]
    fn local_zero_is_exactly_zero_by_authority() {
        let local = candidate(
            perf_desc("local", "lm", true, 2_000_000, 64_000, 90, 90),
            PricingState::LocalZero,
        );
        let req = priced_req(1_000_000, 10_000, 50);
        let qualified = qualified_priced_candidates(
            std::slice::from_ref(&local),
            &req,
            &[],
            &LiveHealth::default(),
        )
        .unwrap();
        assert_eq!(qualified[0].cost, CostEstimate::LocalZero);
        assert_eq!(qualified[0].cost.numeric(), Some(0));
        let svc = RouterService::with_route_candidates(vec![local], Arc::new(EmptyOutcomeStore));
        let d = svc.route(&req, &[]).unwrap();
        assert_eq!(d.estimated_cost_micro, 0, "an authoritative zero is zero");
        let snap = d.pricing_snapshot.expect("snapshot");
        assert_eq!(snap.authority, PriceAuthority::LocalZero);
        assert_eq!(
            snap.settle_cost(u64::MAX, u64::MAX, u64::MAX, u64::MAX),
            Some(0)
        );
    }

    #[test]
    fn unknown_cost_is_never_a_numeric_zero() {
        let usage = TokenUsage::new(1_000, 0, 0, 100);
        assert_eq!(CostEstimate::Unknown.numeric(), None);
        assert_ne!(CostEstimate::Unknown, CostEstimate::Known(0));
        assert_ne!(CostEstimate::Unknown, CostEstimate::LocalZero);
        assert!(
            !matches!(CostEstimate::Unknown.numeric(), Some(0)),
            "unknown must not read as free"
        );
        // Unknown always ranks AFTER every numeric cost (including 0).
        assert_eq!(
            cmp_optional_cost(CostEstimate::Unknown.numeric(), Some(0)),
            std::cmp::Ordering::Greater
        );
        // A hostile zero-by-number quote under Unknown authority stays
        // Unknown and settles to NOTHING.
        let mut hostile = PricingSnapshot::unknown(1, "hostile".into());
        hostile.quote = Some(PriceQuote::ZERO);
        assert_eq!(
            CostEstimate::from_snapshot(&hostile, usage),
            CostEstimate::Unknown
        );
        // A `Known` snapshot with a ZERO quote is a real (exact) zero-price
        // statement, not a local runtime: authority is the variant.
        let zero_known = PricingSnapshot::exact(PriceQuote::ZERO, 1, "zero-price".into());
        assert_eq!(
            CostEstimate::from_snapshot(&zero_known, usage),
            CostEstimate::Known(0)
        );
    }

    #[test]
    fn priced_outcomes_dominate_priors_when_present() {
        // Durable verified history dominates the conservative priors: a
        // cheap model with measured rework loses to a reliable model once
        // the history exists, on the priced path too.
        let cheap = candidate(
            perf_desc("e1", "cheap", true, 512_000, 64_000, 82, 82),
            exact_pricing(1_000_000, 3_000_000),
        );
        let strong = candidate(
            perf_desc("f1", "big", true, 512_000, 64_000, 95, 95),
            exact_pricing(15_000_000, 60_000_000),
        );
        let store = Arc::new(MemoryOutcomeStore::new());
        let key = |provider: &str, model: &str| OutcomeKey {
            provider: provider.into(),
            model: model.into(),
            phase: RouterPhase::Implement,
            task_class: TaskClass::Medium,
            risk_bucket: RiskBucket::Low,
        };
        for _ in 0..20 {
            store.append_sample(
                &key("e1", "cheap"),
                OutcomeSample {
                    verified_success: false,
                    rework_cost_micro: 2_400_000,
                    rework_turns: 3,
                },
            );
        }
        for _ in 0..100 {
            store.append_sample(
                &key("f1", "big"),
                OutcomeSample {
                    verified_success: true,
                    rework_cost_micro: 0,
                    rework_turns: 0,
                },
            );
        }
        let req = priced_req(40_000, 6_000, 80);
        let before = RouterService::with_route_candidates(
            vec![cheap.clone(), strong.clone()],
            Arc::new(EmptyOutcomeStore),
        )
        .route(&req, &[])
        .unwrap();
        assert_eq!(
            before.provider, "e1",
            "without history the cheap quote wins"
        );
        let after = RouterService::with_route_candidates(vec![cheap, strong], store)
            .route(&req, &[])
            .unwrap();
        assert_eq!(
            after.provider, "f1",
            "verified outcomes must dominate the priors: {}",
            after.reasoning
        );
        assert!(after.reasoning.contains("verified"), "{}", after.reasoning);
        assert!(
            after.reasoning.contains("expected-cost"),
            "{}",
            after.reasoning
        );
    }
}
