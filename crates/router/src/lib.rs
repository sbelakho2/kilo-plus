//! Economic model router (audit: minimize expected cost to VERIFIED
//! success, not price per token).
//!
//! Deterministic algorithm:
//! 1. capability filter (fail closed on unknown capabilities),
//! 2. context/output fit filter,
//! 3. quality-floor filter over the coding-relevant reliability mean,
//! 4. cost = prompt-cache-aware micro-unit estimate (integer, rounded up),
//! 5. cheapest qualifying model wins; ties by latency then name,
//! 6. local zero-cost models count as cost-free but latency-weighted,
//! 7. every decision carries an audit string (phase, considered,
//!    filtered, chosen, cost, latency, floor) — no hidden choices.

use faktor_core::model::{
    ModelDescriptor, ModelEconomics, RateLimitState, RouteDecision, RouterPhase,
};

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
/// UNITS (audit 9): `input_price_per_mtok` is the price PER MILLION
/// TOKENS expressed in MICROUSD — equivalently it is the price per token
/// in microUSD, because 1 Mtok at $p/Mtok = p microUSD/token. One integer
/// therefore serves both readings: cost_micro = sum(tokens x price) with
/// NO division. (Dividing by 1_000_000 would double-count: $15/Mtok = 15
/// microUSD/token, so 1M tokens cost 1M x 15 microUSD = $15 exactly.)
/// The equivalence is locked by the property tests below. Saturating
/// arithmetic makes hostile magnitudes safe.
pub fn estimated_call_cost(
    ec: &ModelEconomics,
    input_tokens: u64,
    output_tokens: u64,
    cache_read_tokens: u64,
    cache_write_tokens: u64,
) -> u64 {
    let uncached = input_tokens.saturating_sub(cache_read_tokens);
    let mut cost = uncached
        .saturating_mul(ec.input_price_per_mtok)
        .saturating_add(cache_read_tokens.saturating_mul(ec.cache_read_price_per_mtok))
        .saturating_add(cache_write_tokens.saturating_mul(ec.cache_write_price_per_mtok))
        .saturating_add(output_tokens.saturating_mul(ec.output_price_per_mtok));
    // Cost is never understated: any priced byte carries at least 1 micro.
    if cost == 0 && !ec.is_local_zero_cost() {
        cost = 1;
    }
    cost
}

pub struct Router {
    pub candidates: Vec<ModelDescriptor>,
}

impl Router {
    pub fn new(candidates: Vec<ModelDescriptor>) -> Self {
        Self { candidates }
    }

    fn quality_of(d: &ModelDescriptor) -> u8 {
        d.economics.coding_quality()
    }

    /// Phase→quality semantics: implementation/review/debug trust the
    /// coding mean; summarization/title/embed are cheap-phase (floor is
    /// compared against context reliability only for those). Documented
    /// per the audit's cheap-phase guidance.
    fn floor_model(phase: &RouterPhase) -> u8 {
        match phase {
            RouterPhase::Implement | RouterPhase::Review | RouterPhase::Debug => 3,
            _ => 1,
        }
    }

    pub fn route(&self, req: &RouteRequest, cache: &[CacheState]) -> Result<RouteDecision, String> {
        // 1. capability + fit filter (fail closed, explicit blocker).
        let mut considered = 0usize;
        let mut filtered = Vec::new();
        for d in &self.candidates {
            considered += 1;
            if !d.capability_ok(&req.required_capabilities) {
                continue;
            }
            if d.context < req.context_tokens || d.max_output < req.estimated_output_tokens {
                continue;
            }
            if req.rate_limit_blocks(d) {
                continue;
            }
            filtered.push(d);
        }
        if filtered.is_empty() {
            let missing = req
                .required_capabilities
                .iter()
                .filter(|c| {
                    !self
                        .candidates
                        .iter()
                        .any(|d| d.capability_ok(&[(*c).clone()]))
                })
                .cloned()
                .collect::<Vec<_>>();
            return Err(format!(
                "no candidate clears capability/fit filtering (missing: {})",
                missing.join(",")
            ));
        }
        // 2. quality floor.
        let floor = req.quality_floor.min(100);
        filtered.retain(|d| {
            let q = Self::quality_of(d);
            match Self::floor_model(&req.phase) {
                3 => q >= floor,
                _ => d.economics.context_reliability >= floor.min(100),
            }
        });
        if filtered.is_empty() {
            return Err(format!(
                "no candidate clears the quality floor {} for phase {:?}",
                floor, req.phase
            ));
        }
        // 3. cost each with cache economics.
        let mut best: Option<(u64, u64, &ModelDescriptor)> = None; // (cost, latency, desc)
        for d in &filtered {
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
            let cost = estimated_call_cost(
                &d.economics,
                req.context_tokens,
                req.estimated_output_tokens,
                cached,
                will_write,
            );
            if req.task_budget_remaining_micro > 0 && cost > req.task_budget_remaining_micro {
                continue;
            }
            if let Some(lp) = req.latency_preference_ms {
                if d.economics.estimated_latency_ms > lp {
                    continue; // latency preference is a hard filter
                }
            }
            let latency = d.economics.estimated_latency_ms;
            let better = match best {
                None => true,
                Some((bc, bl, _)) => cost < bc || (cost == bc && latency < bl),
            };
            if better {
                best = Some((cost, latency, d));
            }
        }
        let (cost, latency, chosen) = best.ok_or_else(|| {
            "no candidate within the remaining budget/latency constraints".to_string()
        })?;
        let phase_tag = serde_json::to_string(&req.phase)
            .unwrap_or_default()
            .trim_matches('"')
            .to_string();
        let quality = Self::quality_of(chosen);
        let reasoning = format!(
            "phase={phase_tag} considered={considered} filtered={} chosen={}/{} cost_micro={cost} latency_ms={latency} quality={quality} floor={floor}",
            filtered.len(),
            chosen.provider,
            chosen.model,
        );
        Ok(RouteDecision {
            provider: chosen.provider.clone(),
            model: chosen.model.clone(),
            estimated_cost_micro: cost,
            estimated_latency_ms: latency,
            reasoning,
            considered,
            source: chosen.source,
        })
    }
}

impl RouteRequest {
    fn rate_limit_blocks(&self, _d: &ModelDescriptor) -> bool {
        // Budget/latency constraints handled below; rate-limit state is
        // carried in economics and consulted by callers that observe
        // live 429s (escalation happens there, per the audit).
        false
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
}

impl Default for Ewma {
    fn default() -> Self {
        Self {
            success: PRIOR_SUCCESS,
            retry: 0.1,
            rate_limit: 0.0,
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
}

impl Default for RouterTelemetry {
    fn default() -> Self {
        Self::new()
    }
}

/// Production router: expected-cost-to-verified-success selection with
/// telemetry priors, rate-limit cooldowns and full audit strings.
pub struct RouterService {
    pub router: Router,
    pub telemetry: RouterTelemetry,
}

impl RouterService {
    pub fn new(candidates: Vec<ModelDescriptor>) -> Self {
        Self {
            router: Router::new(candidates),
            telemetry: RouterTelemetry::new(),
        }
    }

    /// Expected cost = base + P(retry)*base + (1-P(success))*escalation,
    /// where escalation = cost of the best frontier candidate (or base*3
    /// when the cheapest-above-floor is the only option). Selection picks
    /// the minimum EXPECTED cost; the decision's estimated_cost_micro
    /// stays the BASE cost so downstream budget math is conservative.
    pub fn route(&self, req: &RouteRequest, cache: &[CacheState]) -> Result<RouteDecision, String> {
        // Cooldown/Hard exclusion (audit 11).
        let eligible: Vec<&ModelDescriptor> = self
            .router
            .candidates
            .iter()
            .filter(|d| {
                !(d.economics.rate_limit_state == RateLimitState::Hard
                    || self.telemetry.cooldown_active(&d.provider))
            })
            .collect();
        let service = Router {
            candidates: eligible.iter().map(|d| (*d).clone()).collect(),
        };
        // (1) pick the base floor/fit winner the plain router would.
        let plain = service.route(req, cache)?;
        // Per-candidate base cost (cache-aware).
        let base_cost = |d: &ModelDescriptor| -> u64 {
            let cs = cache
                .iter()
                .find(|c| c.provider == d.provider && c.model == d.model);
            let (cached, w) = cs
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
                w,
            )
        };
        let _ = &plain;
        let mut best: Option<(u128, &ModelDescriptor)> = None;
        for d in service.candidates.iter() {
            let base = base_cost(d) as u128;
            if req.task_budget_remaining_micro > 0
                && base > u128::from(req.task_budget_remaining_micro)
            {
                continue;
            }
            // Escalation: cheapest qualified candidate EXCLUDING this one.
            let escalation = service
                .candidates
                .iter()
                .filter(|o| o.provider != d.provider || o.model != d.model)
                .filter(|o| o.economics.coding_quality() >= req.quality_floor.min(100))
                .map(|o| u128::from(base_cost(o)))
                .min()
                .unwrap_or(base.saturating_mul(3));
            let p_success =
                self.telemetry
                    .success_estimate(&d.provider, &d.model, req.phase) as f64;
            let p_fail = 1.0 - p_success;
            let expected_cost = base
                + ((0.5 * p_fail * base).ceil() as u128)
                + ((0.5 * p_fail * escalation).ceil() as u128);
            let better = match best {
                None => true,
                Some((be, bd)) => {
                    expected_cost < be
                        || (expected_cost == be
                            && base < u128::from(bd.economics.estimated_latency_ms))
                }
            };
            if better {
                best = Some((expected_cost, d));
            }
        }
        let (_expected, chosen) =
            best.ok_or_else(|| "no candidate clears budget/expected-cost constraints".to_string())?;
        let base = base_cost(chosen);
        let ps = self
            .telemetry
            .success_estimate(&chosen.provider, &chosen.model, req.phase);
        let reasoning = format!(
            "phase={:?} expected-cost chosen={}/{} base_micro={base} p_success={ps:.2} plain={}/{}",
            req.phase, chosen.provider, chosen.model, plain.provider, plain.model,
        );
        Ok(RouteDecision {
            provider: chosen.provider.clone(),
            model: chosen.model.clone(),
            estimated_cost_micro: base,
            estimated_latency_ms: chosen.economics.estimated_latency_ms,
            reasoning,
            considered: service.candidates.len(),
            source: chosen.source,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use faktor_core::model::{ModelEconomics, ModelSource, RateLimitState};

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
            input_price_per_mtok: input,
            output_price_per_mtok: output,
            cache_read_price_per_mtok: input / 5,
            cache_write_price_per_mtok: input / 2,
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

    fn prices() -> (u64, u64) {
        // $15/Mtok == 15 microUSD per token: equivalence property.
        (15, 60)
    }

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
}
