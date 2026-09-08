//! Economic model router (audit: minimize expected cost to VERIFIED
//! success, not price per token).
//!
//! Deterministic algorithm:
//! 1. ONE authoritative qualification pass (`qualified_candidates`) applies
//!    every axis — capability filter (fail closed on unknown capabilities),
//!    context/output fit, phase quality floor, request budget, latency
//!    preference, and live rate-limit/cooldown state (static Hard state
//!    plus telemetry cooldowns fed in as a `LiveHealth` snapshot). Every
//!    selection path consumes ONLY `QualifiedCandidate` vectors; there is
//!    exactly one place that filters;
//! 2. scoring is explicit on `ScoredCandidate`: expected cost-to-success
//!    in microUSD (prompt-cache-aware, integer, rounded up) with the
//!    documented tie-break ladder — no tuple-slot overloading, no
//!    unit-confused comparisons (base microUSD is never compared against
//!    milliseconds);
//! 3. prices are typed [`faktor_core::model::MicroUsdPerToken`] so money
//!    can never silently mix with latency or any other bare `u64`;
//! 4. local zero-cost models count as cost-free but latency-weighted;
//! 5. every decision carries an audit string (phase, considered,
//!    qualified, chosen, cost, latency, floor) — no hidden choices.

use std::collections::HashMap;

use faktor_core::model::{
    ModelDescriptor, ModelEconomics, PricingSnapshot, RateLimitState, RouteDecision, RouterPhase,
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

/// One routing request.
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

/// Estimated total cost of one call in MICRO-units (integer math).
///
/// UNITS (audit 9 + P0-5): the price fields are
/// [`faktor_core::model::MicroUsdPerToken`] — microUSD PER TOKEN, typed so
/// money can never mix with latency. The per-token microUSD value is
/// numerically equal to USD per million tokens (1e6 microUSD per USD over
/// 1e6 tokens), so a $15/Mtok price occupies the same integer as 15
/// microUSD/token and one integer serves both readings: cost_micro =
/// sum(tokens x price) with NO division. (Dividing by 1_000_000 would
/// double-count: $15/Mtok = 15 microUSD/token, so 1M tokens cost 1M x 15
/// microUSD = $15 exactly.) The equivalence is locked by the property tests
/// below. Saturating arithmetic makes hostile magnitudes safe.
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
    /// rounded up to >= 1 micro for any priced model).
    pub call_cost_micro: u64,
    /// Telemetry-blended success prior in ppm (1_000_000 = certain).
    pub success_ppm: u32,
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

/// Heavy phases trust the coding-relevant reliability mean; cheap phases
/// (summarize/title/embed/plan/...) compare the floor against context
/// reliability only. Documented per the audit's cheap-phase guidance.
fn floor_metric_heavy(phase: RouterPhase) -> bool {
    matches!(
        phase,
        RouterPhase::Implement | RouterPhase::Review | RouterPhase::Debug
    )
}

fn clears_quality_floor(d: &ModelDescriptor, req: &RouteRequest, floor: u8) -> bool {
    if floor_metric_heavy(req.phase) {
        d.economics.coding_quality() >= floor
    } else {
        d.economics.context_reliability >= floor
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
        if d.economics.rate_limit_state == RateLimitState::Hard
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
        if !clears_quality_floor(d, req, floor) {
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
            if d.economics.estimated_latency_ms > lp {
                continue;
            }
        }
        qualified.push(QualifiedCandidate {
            descriptor: d,
            call_cost_micro: cost,
            success_ppm: health.success_ppm(&d.provider, &d.model, req.phase),
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
    /// probabilistic retry and escalation terms (integer, rounded up).
    pub expected_cost_micro: u64,
    /// Estimated latency of one call in milliseconds.
    pub expected_latency_ms: u64,
    /// Telemetry-blended success prior in ppm (1_000_000 = certain).
    pub success_ppm: u32,
    /// Base per-call cost in microUSD (cache-aware).
    pub call_cost_micro: u64,
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
fn score_candidates<'a>(qualified: &[QualifiedCandidate<'a>]) -> Vec<ScoredCandidate<'a>> {
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
            ScoredCandidate {
                candidate: d,
                expected_cost_micro: expected_cost_to_success(
                    q.success_ppm,
                    q.call_cost_micro,
                    escalation,
                ),
                expected_latency_ms: d.economics.estimated_latency_ms,
                success_ppm: q.success_ppm,
                call_cost_micro: q.call_cost_micro,
            }
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
            let latency = d.economics.estimated_latency_ms;
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
        let quality = chosen.economics.coding_quality();
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
    /// Catalog-cut route-time snapshots keyed (provider, model); public so
    /// wiring code (and certification harnesses that build services from
    /// scratch) can construct and inspect it.
    pub pricing: HashMap<(String, String), PricingSnapshot>,
}

impl RouterService {
    pub fn new(candidates: Vec<ModelDescriptor>) -> Self {
        Self {
            router: Router::new(candidates),
            telemetry: RouterTelemetry::new(),
            pricing: HashMap::new(),
        }
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
        Self {
            router: Router::new(candidates),
            telemetry: RouterTelemetry::new(),
            pricing,
        }
    }

    /// Expected cost = base + P(retry)*base + (1-P(success))*escalation,
    /// where escalation = cost of the best OTHER QUALIFIED candidate
    /// (or base*3 when the candidate is the only qualified option).
    /// Selection picks the minimum EXPECTED cost by the documented
    /// [`ScoredCandidate::compare`] ladder; the decision's
    /// estimated_cost_micro stays the BASE cost so downstream budget math
    /// is conservative.
    ///
    /// Qualification is the single [`qualified_candidates`] pass — the
    /// same one the plain router uses — with the live telemetry snapshot
    /// (cooldowns + success priors) fed in. There is deliberately NO
    /// second loop over the unfiltered candidate list (audit P0-3): a
    /// model that fails any qualification axis can never be chosen, no
    /// matter how cheap its expected cost looks.
    pub fn route(&self, req: &RouteRequest, cache: &[CacheState]) -> Result<RouteDecision, String> {
        let health = self.telemetry.snapshot();
        let qualified = qualified_candidates(&self.router.candidates, req, cache, &health)
            .map_err(|f| f.route_error())?;
        let scored = score_candidates(&qualified);
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
                        .cmp(&b.descriptor.economics.estimated_latency_ms)
                })
            })
            .expect("qualified_candidates is non-empty on Ok");
        let chosen = winner.candidate;
        let base = winner.call_cost_micro;
        let ps = f64::from(winner.success_ppm) / 1_000_000.0;
        let reasoning = format!(
            "phase={:?} expected-cost chosen={}/{} base_micro={base} p_success={ps:.2} plain={}/{}",
            req.phase,
            chosen.provider,
            chosen.model,
            plain.descriptor.provider,
            plain.descriptor.model,
        );
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
    use faktor_core::model::{MicroUsdPerToken, ModelEconomics, ModelSource, RateLimitState};

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
        stability::TurnPrefix {
            turn_id: id,
            prefix_hash: h,
            prefix_tokens: bytes.len() as u32,
        }
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
        let a_scored = score_candidates(&qualified)
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
}
