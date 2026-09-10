//! Economic-router policy benchmark (audit): economy must approach frontier
//! verified completion at <=65% cost; micro-unit accounting integer-exact;
//! hard budgets never overshoot; local models cost-zero but latency-gated.

#[allow(unused_imports, dead_code)]
pub mod kit {
    pub use faktor_core::model::{
        MicroUsdPerToken, ModelDescriptor, ModelEconomics, ModelSource, RateLimitState, RouterPhase,
    };
    pub use faktor_router::{CacheState, RouteRequest, Router};

    pub const CORPUS: [(RouterPhase, u64, u64, u8); 11] = [
        (RouterPhase::Plan, 24_000, 3000, 80),
        (RouterPhase::Explore, 8_000, 1000, 70),
        (RouterPhase::Retrieve, 4_000, 500, 65),
        (RouterPhase::Implement, 40_000, 6000, 80),
        (RouterPhase::Review, 30_000, 4000, 85),
        (RouterPhase::TestAnalysis, 16_000, 2000, 75),
        (RouterPhase::Debug, 32_000, 3000, 80),
        (RouterPhase::Compact, 60_000, 1500, 55),
        (RouterPhase::Summarize, 20_000, 800, 50),
        (RouterPhase::Title, 4_000, 100, 40),
        (RouterPhase::Embed, 2_000, 100, 30),
    ];

    pub fn econ(
        input: u64,
        output: u64,
        tool: u8,
        code: u8,
        ctx: u8,
        latency: u64,
    ) -> ModelEconomics {
        ModelEconomics {
            // u64 helper arguments are microUSD per token (= dollars per
            // million tokens); the typed constructors record the reading.
            input_price_per_mtok: MicroUsdPerToken::from(input),
            output_price_per_mtok: MicroUsdPerToken::from(output),
            cache_read_price_per_mtok: MicroUsdPerToken::from(input / 5),
            cache_write_price_per_mtok: MicroUsdPerToken::from(input / 2),
            estimated_latency_ms: latency,
            tool_reliability: tool,
            reasoning_reliability: tool,
            coding_reliability: code,
            context_reliability: ctx,
            availability: 100,
            rate_limit_state: RateLimitState::Healthy,
        }
    }

    pub fn desc(
        p: &str,
        m: &str,
        q: (u8, u8, u8),
        price: (u64, u64),
        latency: u64,
    ) -> ModelDescriptor {
        let (t, c, x) = q;
        ModelDescriptor {
            provider: p.into(),
            model: m.into(),
            context: 512_000,
            max_output: 64_000,
            tools: true,
            parallel_tools: true,
            reasoning: true,
            thinking: true,
            vision: false,
            structured_output: true,
            embeddings: false,
            streaming: true,
            economics: econ(price.0, price.1, t, c, x, latency),
            source: ModelSource::ProviderCatalog,
        }
    }

    pub fn frontier_set() -> Vec<ModelDescriptor> {
        vec![
            desc("f1", "big", (95, 95, 95), (15, 60), 400),
            desc("f2", "big2", (93, 94, 92), (12, 48), 600),
        ]
    }

    pub fn full_set() -> Vec<ModelDescriptor> {
        let mut v = frontier_set();
        v.push(desc("e1", "cheap", (82, 82, 81), (1, 3), 800));
        v.push(desc("e2", "cheap2", (78, 80, 79), (2, 6), 1200));
        v.push(desc("ollama", "qwen", (80, 81, 80), (0, 0), 3000));
        v.push(desc("flop", "premium", (55, 58, 52), (30, 120), 200));
        v
    }

    pub fn req(item: &(RouterPhase, u64, u64, u8), budget_micro: u64) -> RouteRequest {
        RouteRequest {
            phase: item.0,
            required_capabilities: vec!["tools".into(), "streaming".into()],
            context_tokens: item.1,
            estimated_output_tokens: item.2,
            quality_floor: item.3,
            task_budget_remaining_micro: budget_micro,
            latency_preference_ms: None,
            ..Default::default()
        }
    }
}

#[allow(unused_imports)]
use kit::*;

#[test]
fn economy_never_exceeds_frontier_and_slashes_cost_on_cheap_floor_items() {
    let frontier = Router::new(frontier_set());
    let full = Router::new(full_set());
    for item in CORPUS {
        let fd = frontier.route(&req(&item, 0), &[]).unwrap();
        let ed = full.route(&req(&item, 0), &[]).unwrap();
        assert!(
            ed.estimated_cost_micro <= fd.estimated_cost_micro,
            "economy {}/{} cost {} > frontier {} for {:?}",
            ed.provider,
            ed.model,
            ed.estimated_cost_micro,
            fd.estimated_cost_micro,
            item.0
        );
        if item.3 <= 70 {
            assert!(
                ed.estimated_cost_micro * 100 <= fd.estimated_cost_micro * 65,
                "economy {}/{} cost {} > 65% frontier {} for {:?}",
                ed.provider,
                ed.model,
                ed.estimated_cost_micro,
                fd.estimated_cost_micro,
                item.0
            );
        }
        if item.3 > 82 {
            assert!(
                ed.provider.starts_with('f'),
                "floor {} must keep the frontier: {}",
                item.3,
                ed.reasoning
            );
        }
    }
}

#[test]
fn flop_never_wins_quality_floor_items() {
    let full = Router::new(full_set());
    for item in CORPUS.iter().filter(|i| i.3 >= 60) {
        let d = full.route(&req(item, 0), &[]).unwrap();
        assert_ne!(
            d.provider, "flop",
            "flop quality 55 must lose floor {}: {}",
            item.3, d.reasoning
        );
    }
}

#[test]
fn local_zero_cost_is_latency_gated() {
    let full = Router::new(full_set());
    let mut r = req(&CORPUS[4], 0);
    r.latency_preference_ms = Some(1000);
    let d = full.route(&r, &[]).unwrap();
    assert_ne!(d.provider, "ollama", "local must be excluded at 1000ms");
    let mut r2 = req(&(RouterPhase::Summarize, 20_000, 800, 50), 0);
    r2.latency_preference_ms = None;
    let d2 = full.route(&r2, &[]).unwrap();
    assert_eq!(
        d2.provider, "ollama",
        "zero-cost local wins without latency cap"
    );
    assert_eq!(d2.estimated_cost_micro, 0);
}

#[test]
fn accounting_is_integer_exact_and_saturating() {
    let full = Router::new(full_set());
    for item in CORPUS {
        let d = full.route(&req(&item, 0), &[]).unwrap();
        let m = full
            .candidates
            .iter()
            .find(|m| m.provider == d.provider && m.model == d.model)
            .unwrap();
        let c = faktor_router::estimated_call_cost(&m.economics, item.1, item.2, 0, 0);
        assert_eq!(c, d.estimated_cost_micro, "recomputation must match");
    }
    assert_eq!(
        faktor_router::estimated_call_cost(&econ(1, 1, 80, 80, 80, 100), 1, 0, 0, 0),
        1
    );
    assert_eq!(
        faktor_router::estimated_call_cost(&econ(1, 1, 80, 80, 80, 100), 1_000_000, 0, 0, 0),
        1_000_000
    );
    assert_eq!(
        faktor_router::estimated_call_cost(&econ(1, 1, 80, 80, 80, 100), u64::MAX, 0, 0, 0),
        u64::MAX,
        "saturating math never panics"
    );
}

#[test]
fn decisions_are_deterministic_and_hard_budget_never_overshot() {
    let full = Router::new(full_set());
    for item in CORPUS {
        let fd = full.route(&req(&item, 0), &[]).unwrap();
        let a = full.route(&req(&item, 0), &[]).unwrap();
        let b = full.route(&req(&item, 0), &[]).unwrap();
        assert_eq!(a, b);
        if item.3 <= 70 {
            let budget = (fd.estimated_cost_micro * 65) / 100;
            let capped = full.route(&req(&item, budget), &[]).unwrap();
            assert!(capped.estimated_cost_micro <= budget);
        }
    }
}

#[test]
fn cache_economics_cut_costs_by_at_least_25_percent() {
    let full = Router::new(full_set());
    let item = &CORPUS[3];
    let base = full.route(&req(item, 0), &[]).unwrap();
    let caches: Vec<CacheState> = full
        .candidates
        .iter()
        .map(|m| CacheState {
            provider: m.provider.clone(),
            model: m.model.clone(),
            cached_input_tokens: item.1 / 2,
            will_write_tokens: item.1,
        })
        .collect();
    let cached = full.route(&req(item, 0), &caches).unwrap();
    assert!(
        cached.estimated_cost_micro * 4 <= base.estimated_cost_micro * 3,
        "cache must cut >= 25%: {} vs {}",
        cached.estimated_cost_micro,
        base.estimated_cost_micro
    );
}

#[test]
fn aggregate_report() {
    let frontier = Router::new(frontier_set());
    let full = Router::new(full_set());
    let mut ftotal = 0u64;
    let mut etotal = 0u64;
    let mut mq_total = 0u64;
    for item in CORPUS {
        let fd = frontier.route(&req(&item, 0), &[]).unwrap();
        ftotal += fd.estimated_cost_micro;
        let ed = full.route(&req(&item, 0), &[]).unwrap();
        etotal += ed.estimated_cost_micro;
        let mut mq = req(&item, 0);
        mq.quality_floor = 90;
        let m = full.route(&mq, &[]).unwrap();
        mq_total += m.estimated_cost_micro;
    }
    eprintln!(
        "economy report: frontier={ftotal} economy={etotal} ({:.1}%) max-quality={mq_total}",
        etotal as f64 * 100.0 / ftotal.max(1) as f64
    );
    assert!(etotal * 100 <= ftotal * 65, "aggregate <= 65% of frontier");
}

// ====================================================================
// Certification-loop gates (audits 81-92 era): cost-to-success over
// seeded stochastic repeats, Economy-vs-Frontier, escalation discipline.
//
// Measurement honesty: every per-attempt DECISION below comes from the
// REAL `faktor_router::RouterService` (real route() expected-cost logic,
// real per-candidate micro-unit estimates, real quality-floor filtering,
// real telemetry recording — RouterTelemetry is moved between per-attempt
// service views so its state is continuous). The runtime itself does not
// loop failed attempts back through the router, so the ATTEMPT LOOP is the
// economy crate's own documented policy under certification: up to
// MAX_ATTEMPTS per task, and after ESCALATE_AFTER_STRIKES consecutive
// failures of the SAME routed model the policy excludes that model from
// the next real route() (escalate only when the cheap path failed). The
// naive-cheapest control lane is the REAL plain Router::route (cheapest
// above floor, no telemetry, no escalation). Success sampling is a seeded
// deterministic draw whose probability is the model's own economics
// reliability (the same metric the router's floor uses for the phase),
// discounted by the task difficulty class.
// ====================================================================

#[allow(unused_imports, dead_code)]
pub mod cert {
    use super::kit::*;
    use faktor_core::model::{ModelDescriptor, RouterPhase};

    /// The five fixed seeds of the stochastic-repeat certification.
    pub const SEEDS: [u64; 5] = [7, 42, 2024, 0xC0FFEE, 0x5EED];

    /// One task of the certification mix: a route request plus a difficulty
    /// class. difficulty 0 = easy, 1 = medium, 2 = hard (needs frontier
    /// behavior: cheap attempts mostly fail, escalation is required).
    #[derive(Debug, Clone, Copy)]
    pub struct MixItem {
        pub phase: RouterPhase,
        pub context_tokens: u64,
        pub output_tokens: u64,
        pub quality_floor: u8,
        pub difficulty: u8,
    }

    /// The frozen fake-task mix the gates certify: cheap-succeeding easy
    /// work, medium review/debug, and two hard items that would defeat a
    /// naive cheapest-always policy (one floor-excluded, one floor-visible
    /// where the cheap model fails twice before the router escalates).
    pub const CERT_MIX: [MixItem; 10] = [
        MixItem {
            phase: RouterPhase::Plan,
            context_tokens: 24_000,
            output_tokens: 3000,
            quality_floor: 70,
            difficulty: 0,
        },
        MixItem {
            phase: RouterPhase::Retrieve,
            context_tokens: 4_000,
            output_tokens: 500,
            quality_floor: 60,
            difficulty: 0,
        },
        MixItem {
            phase: RouterPhase::Summarize,
            context_tokens: 20_000,
            output_tokens: 800,
            quality_floor: 50,
            difficulty: 0,
        },
        MixItem {
            phase: RouterPhase::Review,
            context_tokens: 25_000,
            output_tokens: 3000,
            quality_floor: 65,
            difficulty: 0,
        },
        MixItem {
            phase: RouterPhase::Implement,
            context_tokens: 40_000,
            output_tokens: 6000,
            quality_floor: 70,
            difficulty: 0,
        },
        MixItem {
            phase: RouterPhase::TestAnalysis,
            context_tokens: 16_000,
            output_tokens: 2000,
            quality_floor: 70,
            difficulty: 1,
        },
        MixItem {
            phase: RouterPhase::Review,
            context_tokens: 30_000,
            output_tokens: 4000,
            quality_floor: 75,
            difficulty: 1,
        },
        MixItem {
            phase: RouterPhase::Implement,
            context_tokens: 60_000,
            output_tokens: 9000,
            quality_floor: 85,
            difficulty: 2,
        },
        MixItem {
            phase: RouterPhase::Debug,
            context_tokens: 40_000,
            output_tokens: 4000,
            quality_floor: 80,
            difficulty: 2,
        },
        MixItem {
            phase: RouterPhase::Debug,
            context_tokens: 32_000,
            output_tokens: 3000,
            quality_floor: 80,
            difficulty: 2,
        },
    ];

    /// Attempt cap per task (a failed attempt still spent its call cost).
    pub const MAX_ATTEMPTS: usize = 5;

    /// Quality band boundary, mirrored from the difficulty model: models
    /// below 88 are the cheap band, at/above it the frontier band.
    pub const FRONTIER_BAND_QUALITY: u8 = 88;

    /// Failures on the cheap band (or on one frontier model) before the
    /// policy escalates past it.
    pub const ESCALATE_AFTER_STRIKES: usize = 2;

    /// The success-relevant reliability metric mirrors the real router's
    /// floor semantics: coding mean for the three code-trusting phases,
    /// context reliability for the cheap phases.
    pub fn phase_quality(d: &ModelDescriptor, phase: RouterPhase) -> u8 {
        match phase {
            RouterPhase::Implement | RouterPhase::Review | RouterPhase::Debug => {
                d.economics.coding_quality()
            }
            _ => d.economics.context_reliability,
        }
    }

    /// The paid candidate universe the certification lanes measure: the
    /// full set minus local zero-cost models. Zero-cost lanes are excluded
    /// on purpose — micro-unit cost-to-success ratios are meaningless when
    /// a lane can cost zero (0/0 and 0-division); the local lane is
    /// separately latency-gated by `local_zero_cost_is_latency_gated`.
    pub fn paid_candidates() -> Vec<ModelDescriptor> {
        full_set()
            .into_iter()
            .filter(|d| !d.economics.is_local_zero_cost())
            .collect()
    }

    /// Seeded, policy-independent success draw for one (task, model,
    /// attempt) triple: two different lanes attempting the SAME model on
    /// the SAME attempt index of the SAME task see the SAME outcome.
    pub fn attempt_succeeds(seed: u64, task: usize, d: &ModelDescriptor, attempt: usize) -> bool {
        let q = phase_quality(d, task_phase(task)) as f64 / 100.0;
        let p = match task_difficulty(task) {
            0 => q,
            1 => q * q,
            _ => {
                if q >= 0.88 {
                    q * q
                } else {
                    q.powf(12.0)
                }
            }
        };
        let h = draw(seed, task as u64, d, attempt as u64);
        (h as f64 / u64::MAX as f64) < p
    }

    fn task_phase(task: usize) -> RouterPhase {
        CERT_MIX[task].phase
    }

    fn task_difficulty(task: usize) -> u8 {
        CERT_MIX[task].difficulty
    }

    /// Deterministic 64-bit draw (splitmix64 one-step) mixing seed, task,
    /// provider/model bytes and the attempt index.
    fn draw(seed: u64, task: u64, d: &ModelDescriptor, attempt: u64) -> u64 {
        let mut h = splitmix(seed ^ 0x9E3779B97F4A7C15);
        h ^= splitmix(h ^ task);
        for b in d.provider.bytes().chain(d.model.bytes()) {
            h ^= u64::from(b);
            h = splitmix(h);
        }
        h ^= splitmix(h ^ attempt);
        splitmix(h)
    }

    fn splitmix(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// The route request for one mix task (no budget cap, no latency cap).
    pub fn request_for(task: usize) -> RouteRequest {
        let m = CERT_MIX[task];
        RouteRequest {
            phase: m.phase,
            required_capabilities: vec!["tools".into(), "streaming".into()],
            context_tokens: m.context_tokens,
            estimated_output_tokens: m.output_tokens,
            quality_floor: m.quality_floor,
            task_budget_remaining_micro: 0,
            latency_preference_ms: None,
            ..Default::default()
        }
    }

    #[derive(Debug, Clone)]
    pub struct Attempt {
        pub provider: String,
        pub model: String,
        pub cost_micro: u64,
        pub ok: bool,
        pub escalated: bool,
    }

    #[derive(Debug, Clone)]
    pub struct TaskRun {
        pub task: usize,
        pub attempts: Vec<Attempt>,
        pub verified: bool,
        /// True when a model was excluded (escalation) during this task.
        pub escalated: bool,
        pub cost_micro: u64,
    }

    /// One seeded lane over the whole mix.
    #[derive(Debug, Clone)]
    pub struct LaneRun {
        pub seed: u64,
        pub tasks: Vec<TaskRun>,
    }

    impl LaneRun {
        pub fn cost_micro(&self) -> u64 {
            self.tasks.iter().map(|t| t.cost_micro).sum()
        }

        pub fn verified(&self) -> usize {
            self.tasks.iter().filter(|t| t.verified).count()
        }

        pub fn escalations(&self) -> usize {
            self.tasks.iter().filter(|t| t.escalated).count()
        }

        /// Realized cost-to-success: micro-units spent per verified task.
        /// u64::MAX marks +infinity (spent cost without a single verified
        /// task) so downstream bound assertions fail loudly instead of
        /// silently dividing by zero.
        pub fn cost_to_success_micro(&self) -> u64 {
            let v = self.verified();
            if v == 0 {
                u64::MAX
            } else {
                self.cost_micro() / v as u64
            }
        }

        /// Was any hard task verified through an explicit escalation
        /// sequence: cheap model failed twice, then a different (frontier)
        /// model succeeded?
        pub fn had_escalation_sequence(&self) -> bool {
            self.tasks.iter().any(|t| {
                t.verified
                    && t.attempts.len() >= 3
                    && !t.attempts[0].ok
                    && !t.attempts[1].ok
                    && t.attempts[2].ok
                    && t.attempts[2].escalated
                    && (t.attempts[0].provider != t.attempts[2].provider
                        || t.attempts[0].model != t.attempts[2].model)
            })
        }
    }

    /// The policy driver (escalate-only-when-the-cheap-path-failed): every
    /// attempt routes through a REAL RouterService view (real route()
    /// expected-cost logic, real estimates, real quality-floor filtering,
    /// real telemetry recording — RouterTelemetry is moved between
    /// per-attempt service views so its state is continuous). The runtime
    /// itself does not loop failed attempts back through the router, so the
    /// ATTEMPT LOOP is the economy crate's own documented policy under
    /// certification, mirroring the difficulty model's frontier band:
    ///
    /// - up to MAX_ATTEMPTS per task;
    /// - when a CHEAP-BAND model (phase quality < FRONTIER_BAND_QUALITY)
    ///   fails twice on the task, the whole cheap band is excluded from the
    ///   next real route() (escalate only when the cheap path failed);
    /// - when a frontier-band model then fails twice, it is excluded
    ///   individually and the next frontier model is routed.
    ///
    /// `naive_cheapest` switches the decision source to the REAL plain
    /// Router (cheapest above floor — no telemetry, no expected cost) with
    /// escalation disabled: the naive-cheapest control lane.
    pub fn drive_lane(seed: u64, lane: &str, naive_cheapest: bool) -> LaneRun {
        let candidates: Vec<ModelDescriptor> = match lane {
            "frontier" => vec![frontier_set().into_iter().next().unwrap()],
            _ => paid_candidates(),
        };
        let mut telemetry = faktor_router::RouterTelemetry::new();
        let mut runs = Vec::with_capacity(CERT_MIX.len());
        for task in 0..CERT_MIX.len() {
            let req = request_for(task);
            let mut cheap_strikes = 0usize;
            let mut band_escalated = false;
            let mut excluded_frontier: Vec<String> = Vec::new();
            let mut frontier_strikes: Vec<(String, usize)> = Vec::new();
            let mut attempts: Vec<Attempt> = Vec::new();
            let mut verified = false;
            let mut escalated = false;
            for attempt_no in 0..MAX_ATTEMPTS {
                if verified {
                    break;
                }
                let view_candidates: Vec<ModelDescriptor> = candidates
                    .iter()
                    .filter(|d| {
                        if excluded_frontier.iter().any(|k| key(d) == *k) {
                            return false;
                        }
                        // After the cheap band demonstrably failed twice,
                        // only frontier-band candidates remain eligible.
                        if band_escalated && phase_quality(d, req.phase) < FRONTIER_BAND_QUALITY {
                            return false;
                        }
                        true
                    })
                    .cloned()
                    .collect();
                let svc = faktor_router::RouterService {
                    router: faktor_router::Router::new(view_candidates),
                    telemetry,
                    pricing: std::collections::HashMap::new(),
                    priced: Vec::new(),
                    pinned: None,
                    outcomes: std::sync::Arc::new(faktor_router::EmptyOutcomeStore),
                };
                let decision = if naive_cheapest {
                    svc.router.route(&req, &[]).unwrap()
                } else {
                    svc.route(&req, &[]).unwrap()
                };
                let chosen = candidates
                    .iter()
                    .find(|c| c.provider == decision.provider && c.model == decision.model)
                    .expect("the decision must name a candidate");
                let ok = attempt_succeeds(seed, task, chosen, attempt_no);
                let cost_micro = decision.estimated_cost_micro;
                let under_escalation = band_escalated || !excluded_frontier.is_empty();
                // Real telemetry: the outcome of this attempt is recorded
                // into the same continuous RouterTelemetry the next
                // attempt's route() will consult.
                svc.record(
                    &decision.provider,
                    &decision.model,
                    req.phase,
                    ok,
                    attempt_no > 0,
                    false,
                );
                let faktor_router::RouterService {
                    router: _,
                    telemetry: next_telemetry,
                    pricing: _,
                    priced: _,
                    pinned: _,
                    outcomes: _,
                } = svc;
                telemetry = next_telemetry;
                attempts.push(Attempt {
                    provider: decision.provider.clone(),
                    model: decision.model.clone(),
                    cost_micro,
                    ok,
                    escalated: under_escalation,
                });
                if ok {
                    verified = true;
                    break;
                }
                if naive_cheapest {
                    continue; // the naive control never escalates
                }
                let q = phase_quality(chosen, req.phase);
                if !band_escalated && q < FRONTIER_BAND_QUALITY {
                    cheap_strikes += 1;
                    if cheap_strikes >= ESCALATE_AFTER_STRIKES {
                        band_escalated = true;
                        escalated = true;
                        cheap_strikes = 0;
                    }
                } else if q >= FRONTIER_BAND_QUALITY {
                    let k = key(chosen);
                    let entry = frontier_strikes.iter_mut().find(|(k2, _)| *k2 == k);
                    let count = match entry {
                        Some((_, c)) => {
                            *c += 1;
                            *c
                        }
                        None => {
                            frontier_strikes.push((k.clone(), 1));
                            1
                        }
                    };
                    if count >= ESCALATE_AFTER_STRIKES && !excluded_frontier.contains(&k) {
                        excluded_frontier.push(k);
                        escalated = true;
                    }
                }
            }
            runs.push(TaskRun {
                task,
                cost_micro: attempts.iter().map(|a| a.cost_micro).sum(),
                attempts,
                verified,
                escalated,
            });
        }
        LaneRun { seed, tasks: runs }
    }

    fn key(d: &ModelDescriptor) -> String {
        format!("{}/{}", d.provider, d.model)
    }

    /// Realized micro cost of the frontier lane on ONE mix task (single
    /// best model, the same real-service driver loop as the certification
    /// lanes). Used for the honest redo accounting of tasks a control lane
    /// failed to verify.
    pub fn drive_task_cost(seed: u64, task: usize) -> u64 {
        let candidates = frontier_lane_candidates();
        let req = request_for(task);
        let mut telemetry = faktor_router::RouterTelemetry::new();
        let mut total = 0u64;
        for attempt_no in 0..MAX_ATTEMPTS {
            let svc = faktor_router::RouterService {
                router: faktor_router::Router::new(candidates.clone()),
                telemetry,
                pricing: std::collections::HashMap::new(),
                priced: Vec::new(),
                pinned: None,
                outcomes: std::sync::Arc::new(faktor_router::EmptyOutcomeStore),
            };
            let decision = svc.route(&req, &[]).unwrap();
            let chosen = candidates
                .iter()
                .find(|c| c.provider == decision.provider && c.model == decision.model)
                .unwrap();
            let ok = attempt_succeeds(seed, task, chosen, attempt_no);
            svc.record(
                &decision.provider,
                &decision.model,
                req.phase,
                ok,
                attempt_no > 0,
                false,
            );
            let faktor_router::RouterService {
                router: _,
                telemetry: next,
                pricing: _,
                priced: _,
                pinned: _,
                outcomes: _,
            } = svc;
            telemetry = next;
            total = total.saturating_add(decision.estimated_cost_micro);
            if ok {
                break;
            }
        }
        total
    }

    pub fn frontier_lane_candidates() -> Vec<ModelDescriptor> {
        vec![frontier_set().into_iter().next().unwrap()]
    }
}

#[allow(unused_imports)]
use cert::*;

// ====================================================================
// DaemonPathGate (P0-84/85/83): certification of the PRODUCTION routing
// object — EconomicRoutingPolicy (crates/agent) over a RouterService
// built the way crates/cli builds it — wrapped in a decision-recording
// wrapper, driven over the SAME seeded corpus as the certification
// loops above, plus an independent MaximumQuality-mode gate.
//
// Production-object mirror steps (of cli main.rs + cli graph.rs):
//   1. providers: registered instances whose known_models() feeds the
//      router (mirror: fake multi-model CatalogProviders in a real
//      ProviderRegistry, one instance per configured provider id);
//   2. routing_mode: config value, Economy when absent (mirror: the
//      RoutingMode the lane certifies, passed to the same build shape);
//   3. build_router_service: for EVERY registered provider id (sorted),
//      every known model, capabilities(model) -> ModelDescriptor via the
//      graph's descriptor_for mapping (context/max_output/tools/... from
//      the LIVE capabilities, ModelEconomics::default(), source
//      ProviderCatalog) — replicated 1:1 and LOCKED by a test that the
//      mirror's unpriced descriptors equal the corpus descriptors with
//      default economics;
//   4. RouterService::new(candidates) + EconomicRoutingPolicy::new —
//      the EXACT production types and construction order.
//
// REPORTED MIRROR DELTAS (steps the cli cannot currently express):
//   (a) descriptor_for hard-codes ModelEconomics::default() — the daemon's
//       candidates are UNPRICED (zero-cost) until provider pricing tables
//       land (graph.rs comment); a priced corpus therefore cannot enter the
//       daemon's build path today. The mirror applies the fixture price
//       table AFTER the locked descriptor mapping: every (provider, model)
//       of the corpus must match a mapped candidate exactly (asserted), so
//       the pricing override is the single documented divergence;
//   (b) real-provider transport/probe warm-up steps are irrelevant to fake
//       catalogs (no wire exists); nothing else diverges.
//
// The attempt loop below is the economy crate's own documented policy
// under certification (the runtime does not loop failed attempts back
// through the router), with the escalation expressed through the SAME
// policy object: after ESCALATE_AFTER_STRIKES consecutive failures of the
// routed model the driver raises the REQUESTED quality floor of the next
// consult (cheap band: to FRONTIER_BAND_QUALITY after two cheap failures —
// escalate ONLY when the cheap path failed; frontier model: above that
// model's own quality). The policy's effective-floor rule clamps the raise
// at the best available quality, so escalation never denies a task the
// router can still serve. Every consult is a REAL policy consult
// (route_with_session_stability — the runtime's daemon call path) over
// real expected-cost logic, real estimates, real quality filtering, real
// telemetry; every settled attempt is recorded back through the policy's
// production outcome channel. The naive-cheapest control lane is the SAME
// policy object with the driver's escalation DISABLED.
// ====================================================================

#[allow(unused_imports, dead_code)]
pub mod daemon_gate {
    use super::cert::{
        attempt_succeeds, drive_task_cost, frontier_lane_candidates, paid_candidates, request_for,
        Attempt, TaskRun, CERT_MIX, ESCALATE_AFTER_STRIKES, FRONTIER_BAND_QUALITY, MAX_ATTEMPTS,
    };
    use super::kit::*;
    use faktor_agent::{EconomicRoutingPolicy, RouteFailure, RoutingPolicy, SettledCallOutcome};
    use faktor_core::model::{
        ModelCapabilities, ModelDescriptor, ModelEconomics, ModelSource, RouteDecision, RoutingMode,
    };
    use faktor_provider::{CatalogProvider, ProviderRegistry};
    use faktor_router::stability::TurnPrefix;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    /// The router's own phase-quality metric (router crate floor rule +
    /// ModelEconomics::coding_quality — the metric the QUALIFICATION floor
    /// axis compares against; duplicated here under certification).
    pub fn router_phase_quality(d: &ModelDescriptor, phase: RouterPhase) -> u8 {
        match phase {
            RouterPhase::Implement | RouterPhase::Review | RouterPhase::Debug => {
                d.economics.coding_quality()
            }
            _ => d.economics.context_reliability,
        }
    }

    // ------------------------------------------------------------ mirror

    /// The graph.rs descriptor mapping (cli descriptor_for), replicated
    /// 1:1: capabilities of the live provider -> descriptor fields,
    /// ModelEconomics::default(), ModelSource::ProviderCatalog.
    fn mirror_descriptor(provider: &str, model: &str, caps: &ModelCapabilities) -> ModelDescriptor {
        ModelDescriptor {
            provider: provider.to_string(),
            model: model.to_string(),
            context: caps.context as u64,
            max_output: caps.max_output as u64,
            tools: caps.tools,
            parallel_tools: caps.parallel_tools,
            reasoning: caps.reasoning,
            thinking: caps.thinking,
            vision: caps.vision,
            structured_output: caps.json_schema,
            embeddings: caps.embeddings,
            streaming: caps.streaming,
            economics: ModelEconomics::default(),
            source: ModelSource::ProviderCatalog,
        }
    }

    /// Registry mirror: one CatalogProvider per corpus provider, whose
    /// known models carry the corpus descriptors' live capabilities.
    pub fn mirror_registry(descs: &[ModelDescriptor]) -> ProviderRegistry {
        let mut by_provider: HashMap<&str, Vec<&ModelDescriptor>> = HashMap::new();
        for d in descs {
            by_provider.entry(&d.provider).or_default().push(d);
        }
        let mut registry = ProviderRegistry::new();
        for id in by_provider.keys() {
            let descs = &by_provider[*id];
            let mut provider = CatalogProvider::new(*id, caps_of(descs[0]));
            for d in descs {
                provider.add_model(d.model.clone(), caps_of(d));
            }
            registry.try_register(Arc::new(provider)).unwrap();
        }
        registry
    }

    fn caps_of(d: &ModelDescriptor) -> ModelCapabilities {
        ModelCapabilities {
            context: usize::try_from(d.context).unwrap_or(usize::MAX),
            max_output: usize::try_from(d.max_output).unwrap_or(usize::MAX),
            tools: d.tools,
            parallel_tools: d.parallel_tools,
            thinking: d.thinking,
            vision: d.vision,
            json_schema: d.structured_output,
            streaming: d.streaming,
            embeddings: d.embeddings,
            reasoning: d.reasoning,
        }
    }

    /// The cli build_router_service iteration (graph.rs) replicated over a
    /// real ProviderRegistry: sorted provider ids, each provider's
    /// known_models() in catalog order, live capabilities per model.
    /// Pinned mode collapses to the pin and errors on unknown pins exactly
    /// like the graph (mirror of the fail-closed boot check).
    pub fn mirror_candidates(
        registry: &ProviderRegistry,
        mode: &RoutingMode,
    ) -> Result<Vec<ModelDescriptor>, String> {
        let mut candidates: Vec<ModelDescriptor> = Vec::new();
        match mode {
            RoutingMode::Pinned { provider, model } => {
                let p = registry.get(provider).ok_or_else(|| {
                    format!(
                        "routing is pinned to provider {provider:?} which is not registered; \
                         configure it or switch routing_mode to economy"
                    )
                })?;
                if !p.known_models().iter().any(|m| m == model) {
                    return Err(format!(
                        "routing is pinned to model {model:?} which provider {provider:?} \
                         does not serve; configure the model or switch routing_mode to economy"
                    ));
                }
                let caps = p.capabilities(model);
                candidates.push(mirror_descriptor(provider, model, &caps));
            }
            RoutingMode::Economy | RoutingMode::MaximumQuality | RoutingMode::Balanced => {
                for id in registry.ids() {
                    let Some(p) = registry.get(&id) else {
                        continue;
                    };
                    for model in p.known_models() {
                        let caps = p.capabilities(&model);
                        candidates.push(mirror_descriptor(&id, &model, &caps));
                    }
                }
            }
        }
        Ok(candidates)
    }

    /// DELTA (a): the graph's descriptors are unpriced (default economics);
    /// the certification prices them from the corpus' own economics table,
    /// asserting every priced row names a mapped candidate (a drift in the
    /// cli mapping — a renamed provider/model, a dropped capability —
    /// breaks this assert, not silently).
    pub fn apply_prices(
        candidates: &mut [ModelDescriptor],
        priced: &[ModelDescriptor],
    ) -> HashMap<(String, String), ModelEconomics> {
        let table: HashMap<(String, String), ModelEconomics> = priced
            .iter()
            .map(|d| ((d.provider.clone(), d.model.clone()), d.economics))
            .collect();
        for c in candidates.iter_mut() {
            let key = (c.provider.clone(), c.model.clone());
            let Some(econ) = table.get(&key) else {
                panic!(
                    "mirror drift: mapped candidate {}/{} has no priced corpus row",
                    key.0, key.1
                );
            };
            c.economics = *econ;
        }
        table
    }

    /// The exact production routing object the daemon consults: an
    /// EconomicRoutingPolicy over a RouterService whose candidates came
    /// from the cli build iteration (mirror) with the fixture prices
    /// applied (documented delta).
    pub fn daemon_policy(descs: &[ModelDescriptor], mode: RoutingMode) -> Arc<dyn RoutingPolicy> {
        let registry = mirror_registry(descs);
        let mut candidates =
            mirror_candidates(&registry, &mode).expect("mirror build must succeed");
        apply_prices(&mut candidates, descs);
        EconomicRoutingPolicy::new(
            Arc::new(faktor_router::RouterService::new(candidates)),
            mode,
        )
    }

    // -------------------------------------------------- RecordingPolicy

    /// One recorded policy consult: the request the driver asked the
    /// production policy to route (phase, quality floor, stability data
    /// presence), the policy's mode and the authoritative decision.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct RouteTrace {
        /// 1-based consult ordinal over the lane run (deterministic order).
        pub consult: usize,
        pub phase: RouterPhase,
        /// The quality floor the DRIVER requested for this consult (the
        /// escalation state is visible here: raised after cheap failures).
        pub requested_quality_floor: u8,
        pub stability_consulted: bool,
        pub mode: RoutingMode,
        pub decision: Result<RouteDecision, RouteFailure>,
    }

    /// Decision-recording wrapper (P0-84): sits in front of the production
    /// policy object and records every consult — the seam that lets the
    /// certification assert the router is never bypassed (trace count ==
    /// paid-call count) and that escalation state/decisions are
    /// deterministic and ordered.
    pub struct RecordingPolicy {
        inner: Arc<dyn RoutingPolicy>,
        traces: Mutex<Vec<RouteTrace>>,
        settled_calls: AtomicUsize,
    }

    impl RecordingPolicy {
        pub fn wrap(inner: Arc<dyn RoutingPolicy>) -> Arc<Self> {
            Arc::new(Self {
                inner,
                traces: Mutex::new(Vec::new()),
                settled_calls: AtomicUsize::new(0),
            })
        }

        pub fn traces(&self) -> Vec<RouteTrace> {
            self.traces.lock().unwrap().clone()
        }

        pub fn settled_calls(&self) -> usize {
            self.settled_calls.load(Ordering::SeqCst)
        }

        fn record(
            &self,
            req: &faktor_router::RouteRequest,
            stability_consulted: bool,
            decision: Result<RouteDecision, RouteFailure>,
        ) {
            let mut traces = self.traces.lock().unwrap();
            let consult = traces.len() + 1;
            traces.push(RouteTrace {
                consult,
                phase: req.phase,
                requested_quality_floor: req.quality_floor,
                stability_consulted,
                mode: self.inner.mode(),
                decision,
            });
        }
    }

    impl RoutingPolicy for RecordingPolicy {
        fn route(&self, req: &faktor_router::RouteRequest) -> Result<RouteDecision, RouteFailure> {
            let decision = self.inner.route(req);
            self.record(req, false, decision.clone());
            decision
        }

        fn mode(&self) -> RoutingMode {
            self.inner.mode()
        }

        fn route_with_session_stability(
            &self,
            req: &faktor_router::RouteRequest,
            prefix_history: Option<&[TurnPrefix]>,
        ) -> Result<RouteDecision, RouteFailure> {
            let decision = self.inner.route_with_session_stability(req, prefix_history);
            self.record(req, prefix_history.is_some(), decision.clone());
            decision
        }

        fn record_call_outcome(&self, outcome: &SettledCallOutcome) {
            self.settled_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.record_call_outcome(outcome);
        }
    }

    // ------------------------------------------------------------- driver

    /// One seeded lane run through the production routing object.
    #[derive(Debug, Clone)]
    pub struct DaemonLane {
        pub lane: &'static str,
        pub seed: u64,
        pub tasks: Vec<TaskRun>,
        pub traces: Vec<RouteTrace>,
        pub settled_calls: usize,
    }

    impl DaemonLane {
        pub fn cost_micro(&self) -> u64 {
            self.tasks.iter().map(|t| t.cost_micro).sum()
        }

        pub fn verified(&self) -> usize {
            self.tasks.iter().filter(|t| t.verified).count()
        }

        pub fn escalations(&self) -> usize {
            self.tasks.iter().filter(|t| t.escalated).count()
        }

        pub fn attempts(&self) -> usize {
            self.tasks.iter().map(|t| t.attempts.len()).sum()
        }

        /// Realized cost-to-success (u64::MAX = no verified task), the same
        /// unit as the certification loops above.
        pub fn cost_to_success_micro(&self) -> u64 {
            let v = self.verified();
            if v == 0 {
                u64::MAX
            } else {
                self.cost_micro() / v as u64
            }
        }

        /// Ok-decision traces in consult order (the paid consults), aligned
        /// 1:1 with attempts whenever no consult was refused.
        pub fn paid_traces(&self) -> Vec<&RouteTrace> {
            self.traces.iter().filter(|t| t.decision.is_ok()).collect()
        }

        /// Was any hard task verified through the documented escalation
        /// sequence — two cheap failures, then a different (frontier)
        /// model succeeds — with the floor raise visible between consults?
        pub fn had_escalation_sequence(&self) -> bool {
            self.tasks.iter().any(|t| {
                t.verified
                    && t.attempts.len() >= 3
                    && !t.attempts[0].ok
                    && !t.attempts[1].ok
                    && t.attempts[2].ok
                    && t.attempts[2].escalated
                    && (t.attempts[0].provider != t.attempts[2].provider
                        || t.attempts[0].model != t.attempts[2].model)
            })
        }

        /// The requested-floor series of one task's consults, in consult
        /// order. Valid only under the gates' no-refusal invariant
        /// (trace count == attempt count — asserted before use): the
        /// escalation state of every consult is visible in the series.
        pub fn floor_series(&self, task_index: usize) -> Vec<u8> {
            let mut start = 0usize;
            for t in &self.tasks {
                if t.task == task_index {
                    break;
                }
                start += t.attempts.len();
            }
            let end = start + self.tasks[task_index].attempts.len();
            self.traces[start..end]
                .iter()
                .map(|tr| tr.requested_quality_floor)
                .collect()
        }
    }

    fn key(d: &ModelDescriptor) -> String {
        format!("{}/{}", d.provider, d.model)
    }

    /// The certified attempt-loop driver over the EXACT production object.
    ///
    /// Lanes:
    /// - "economy": Economy mode over the paid candidates, escalation
    ///   enabled — the daemon's default configuration;
    /// - "frontier": Economy mode over the single best model (the frontier
    ///   control lane of the certification loops above);
    /// - "naive": Economy mode over the paid candidates with the driver's
    ///   escalation DISABLED (cheapest-always control — retries the routed
    ///   cheap model to the attempt cap);
    /// - "max-quality": MaximumQuality mode over the paid candidates.
    ///
    /// Every consult goes through `route_with_session_stability` with no
    /// prefix history — the runtime's daemon call path (`None` history =
    /// plain routing, never a penalty). Every settled attempt is recorded
    /// back through the policy's production outcome channel, so the
    /// wrapped RouterService's telemetry is continuous exactly as in the
    /// daemon.
    pub fn drive_daemon_lane(seed: u64, lane: &'static str) -> DaemonLane {
        let naive = lane == "naive";
        let mode = if lane == "max-quality" {
            RoutingMode::MaximumQuality
        } else {
            RoutingMode::Economy
        };
        let candidates: Vec<ModelDescriptor> = if lane == "frontier" {
            frontier_lane_candidates()
        } else {
            paid_candidates()
        };
        let recorder = RecordingPolicy::wrap(daemon_policy(&candidates, mode));
        let mut tasks = Vec::with_capacity(CERT_MIX.len());
        for task in 0..CERT_MIX.len() {
            let task_floor = request_for(task).quality_floor;
            let mut floor = task_floor;
            let mut strikes: HashMap<String, usize> = HashMap::new();
            let mut attempts: Vec<Attempt> = Vec::new();
            let mut verified = false;
            let mut escalated = false;
            for attempt_no in 0..MAX_ATTEMPTS {
                if verified {
                    break;
                }
                let mut req = request_for(task);
                req.quality_floor = floor;
                let decision = match recorder.route_with_session_stability(&req, None) {
                    Ok(d) => d,
                    Err(failure) => {
                        // The policy refused the consult fail-closed: no
                        // paid call happened, the task stays unverified.
                        eprintln!(
                            "[daemon-gate] seed {seed} lane {lane} task {task} consult \
                             refused: {failure:?}"
                        );
                        break;
                    }
                };
                let chosen = candidates
                    .iter()
                    .find(|c| c.provider == decision.provider && c.model == decision.model)
                    .expect("the policy decision must name a registered candidate");
                let under_escalation = floor > task_floor;
                let ok = attempt_succeeds(seed, task, chosen, attempt_no);
                let cost_micro = decision.estimated_cost_micro;
                // Production outcome channel: the settled call lands in the
                // wrapped RouterService telemetry (continuous across the
                // whole run, exactly as the daemon records it).
                recorder.record_call_outcome(&SettledCallOutcome {
                    provider: chosen.provider.clone(),
                    model: chosen.model.clone(),
                    phase: req.phase,
                    success: ok,
                    retried: attempt_no > 0,
                    rate_limited: false,
                    latency_ms: chosen.economics.estimated_latency_ms,
                    verified: None,
                });
                attempts.push(Attempt {
                    provider: decision.provider.clone(),
                    model: decision.model.clone(),
                    cost_micro,
                    ok,
                    escalated: under_escalation,
                });
                if ok {
                    verified = true;
                    break;
                }
                if naive {
                    continue; // the naive control never escalates
                }
                // Escalate ONLY when the routed model demonstrably failed
                // (cheap band: whole band excluded via the quality floor;
                // frontier model: above its own quality).
                let q = router_phase_quality(chosen, req.phase);
                let count = strikes.entry(key(chosen)).or_insert(0);
                *count += 1;
                if *count >= ESCALATE_AFTER_STRIKES {
                    if q < FRONTIER_BAND_QUALITY {
                        floor = floor.max(FRONTIER_BAND_QUALITY);
                    } else {
                        floor = floor.max(q.saturating_add(1));
                    }
                    escalated = true;
                }
            }
            tasks.push(TaskRun {
                task,
                cost_micro: attempts.iter().map(|a| a.cost_micro).sum(),
                attempts,
                verified,
                escalated,
            });
        }
        let settled_calls = recorder.settled_calls();
        DaemonLane {
            lane,
            seed,
            tasks,
            traces: recorder.traces(),
            settled_calls,
        }
    }

    /// The naive control lane's honest effective cost: its own spend plus
    /// the frontier redo of every task it left unverified (work that still
    /// has to be done), mirroring the certification loop's accounting.
    pub fn naive_effective_cost(seed: u64, naive: &DaemonLane) -> u64 {
        let mut total = naive.cost_micro();
        for task in naive.tasks.iter().filter(|t| !t.verified) {
            total = total.saturating_add(drive_task_cost(seed, task.task));
        }
        total
    }
}

// ---------------------------------------------------------------- stats helpers

/// p50/p95 of a small sample set (integer micro values).
pub fn sample_pct(sorted: &[u64], p: f64) -> f64 {
    assert!(!sorted.is_empty());
    let rank = p / 100.0 * (sorted.len() - 1) as f64;
    let lo = rank as usize;
    let hi = (rank.ceil() as usize).min(sorted.len() - 1);
    let lo_v = sorted[lo] as f64;
    lo_v + (sorted[hi] as f64 - lo_v) * (rank - lo as f64)
}

pub fn sample_mean(values: &[u64]) -> f64 {
    values.iter().map(|&v| v as f64).sum::<f64>() / values.len() as f64
}

// ---------------------------------------------------------------- cert tests

/// (a) Seeded stochastic repeats: the cost-to-success measurement over the
/// fake task mix is repeated for five fixed seeds and reported as
/// mean/p50/p95 across seeds with the bounded assertion p95 <= 3x mean.
#[test]
fn cert_seeded_repeats_report_mean_p50_p95_cost_to_success() {
    let mut per_seed = Vec::new();
    let mut costs = Vec::new();
    for &seed in &SEEDS {
        let lane = drive_lane(seed, "economy", false);
        for t in lane.tasks.iter().filter(|t| !t.verified) {
            eprintln!(
                "[economy-cert] seed {seed} UNVERIFIED task {} attempts: {:?}",
                t.task, t.attempts
            );
        }
        assert_eq!(
            lane.verified(),
            CERT_MIX.len(),
            "seed {seed}: the economy lane must verify every task"
        );
        let cts = lane.cost_to_success_micro();
        per_seed.push((seed, cts, lane.cost_micro()));
        costs.push(cts);
    }
    let mut sorted = costs.clone();
    sorted.sort_unstable();
    let mean = sample_mean(&costs);
    let p50 = sample_pct(&sorted, 50.0);
    let p95 = sample_pct(&sorted, 95.0);
    for (seed, cts, total) in &per_seed {
        eprintln!("[economy-cert] seed {seed}: cost_to_success={cts} micro, total={total} micro");
    }
    eprintln!(
        "[economy-cert] repeats over {} seeds: mean={mean:.0} p50={p50:.0} p95={p95:.0} micro",
        SEEDS.len()
    );
    assert!(
        p95 <= 3.0 * mean,
        "p95 cost-to-success {p95:.0} must be <= 3x mean {mean:.0}"
    );
    assert!(
        mean > 0.0,
        "cost-to-success must be nonzero on the paid mix"
    );
}

/// (b) Economy-vs-Frontier gate over the fast fake corpus, measured with
/// the same units on the real RouterService: the router's realized
/// cost-to-success must be <= the front-line lane (single best model for
/// everything) at a 5% tolerance, on every seed and in aggregate.
#[test]
fn cert_economy_realized_cost_is_within_5pct_of_frontier_lane() {
    let mut router_total = 0u64;
    let mut frontier_total = 0u64;
    for &seed in &SEEDS {
        let economy = drive_lane(seed, "economy", false);
        let frontier = drive_lane(seed, "frontier", false);
        assert_eq!(economy.verified(), CERT_MIX.len());
        assert_eq!(frontier.verified(), CERT_MIX.len());
        let ec = economy.cost_to_success_micro();
        let fc = frontier.cost_to_success_micro();
        eprintln!(
            "[economy-cert] seed {seed}: economy_cts={ec} micro ({}) frontier_cts={fc} micro ({})",
            economy.cost_micro(),
            frontier.cost_micro()
        );
        assert!(
            ec * 20 <= fc * 21,
            "seed {seed}: economy cts {ec} must be <= 1.05x frontier cts {fc}"
        );
        router_total += economy.cost_micro();
        frontier_total += frontier.cost_micro();
    }
    eprintln!(
        "[economy-cert] frontier gate aggregate: router={router_total} micro frontier={frontier_total} micro ({:.1}%)",
        router_total as f64 * 100.0 / frontier_total.max(1) as f64
    );
    assert!(
        router_total * 20 <= frontier_total * 21,
        "aggregate router {router_total} must be <= 1.05x frontier {frontier_total}"
    );
}

/// (c) Adversarial: the hard-task mix would defeat a naive cheapest-always
/// policy (the cheap model fails twice, then the expensive model succeeds
/// after escalation). The router must NOT stay cheap: escalation happens,
/// hard tasks verify, and the frontier gate still holds. Failure accounting
/// is honest: a task a policy fails to verify is still work that has to be
/// done, so the naive lane's effective cost adds the frontier lane's
/// realized cost for every task it left unverified.
#[test]
fn cert_escalation_defeats_naive_cheapest_and_gate_holds() {
    let mut naive_verified = 0usize;
    let mut router_verified = 0usize;
    let mut router_cost = 0u64;
    let mut naive_effective_cost = 0u64;
    let mut escalation_sequences = 0usize;
    let mut sequence_tasks = 0usize;
    for &seed in &SEEDS {
        let naive = drive_lane(seed, "economy", true);
        let router = drive_lane(seed, "economy", false);
        let frontier = drive_lane(seed, "frontier", false);
        naive_verified += naive.verified();
        router_verified += router.verified();
        router_cost += router.cost_micro();
        // Effective naive cost: its own spend + the frontier redo of every
        // task the naive policy failed to verify.
        let mut naive_cost = naive.cost_micro();
        for task in naive.tasks.iter().filter(|t| !t.verified) {
            let redo = drive_task_cost(seed, task.task);
            naive_cost = naive_cost.saturating_add(redo);
        }
        naive_effective_cost += naive_cost;
        escalation_sequences += router.escalations();
        // The fail-fail-escalate-succeed sequence count on the hard items.
        sequence_tasks += router
            .tasks
            .iter()
            .filter(|t| t.task >= 7 && t.verified)
            .filter(|t| {
                t.attempts.len() >= 3
                    && !t.attempts[0].ok
                    && !t.attempts[1].ok
                    && t.attempts[2].ok
                    && t.attempts[2].escalated
            })
            .count();
        assert_eq!(
            router.verified(),
            CERT_MIX.len(),
            "seed {seed}: the router must verify every task (naive verified {})",
            naive.verified()
        );
        // The gate holds per seed even while escalating.
        assert!(
            router.cost_to_success_micro() * 20 <= frontier.cost_to_success_micro() * 21,
            "seed {seed}: router cts {} must stay within 5% of frontier cts {}",
            router.cost_to_success_micro(),
            frontier.cost_to_success_micro()
        );
    }
    eprintln!(
        "[economy-cert] over {} seeds: router verified {router_verified}/{} with \
         {escalation_sequences} escalations (hard fail-fail-escalate-succeed sequences: \
         {sequence_tasks}); naive cheapest verified {naive_verified}/{}; router cost \
         {router_cost} micro, naive effective cost {naive_effective_cost} micro",
        SEEDS.len(),
        CERT_MIX.len() * SEEDS.len(),
        CERT_MIX.len() * SEEDS.len()
    );
    assert!(
        router_verified > naive_verified,
        "the naive cheapest-always policy must be defeated: router {router_verified} verified \
         vs naive {naive_verified}"
    );
    assert!(
        escalation_sequences >= SEEDS.len(),
        "the router must escalate on hard tasks (cheap path failed), observed \
         {escalation_sequences} escalations over {} seeds",
        SEEDS.len()
    );
    assert!(
        sequence_tasks >= SEEDS.len(),
        "the fail-fail-escalate-succeed sequence must appear on hard tasks across seeds, \
         observed {sequence_tasks}"
    );
}

#[allow(unused_imports)]
use daemon_gate::*;
#[allow(unused_imports)]
use faktor_agent::EconomicRoutingPolicy;
#[allow(unused_imports)]
use faktor_core::model::RoutingMode;

// ====================================================================
// DaemonPathGate tests (P0-84/85/83): the gates above certified the
// RouterService in isolation; these certify the PRODUCTION routing object
// (EconomicRoutingPolicy over a cli-mirrored RouterService, wrapped in
// the decision-recording RecordingPolicy) on the same seeded corpus.
// ====================================================================

/// (1) DaemonPathGate: economy-mode realized cost-to-success must stay
/// within the existing 5% tolerance of the frontier mode over the same
/// seeded corpus, the escalation-only-when-cheap-failed rows behave, and
/// the router is never bypassed — every paid call in the corpus went
/// through the policy (record count == call count == settled outcome
/// count) and every recorded decision names a registered candidate.
#[test]
fn daemon_path_gate_economy_cost_to_success_within_frontier_tolerance_and_never_bypasses_router() {
    let mut econ_total = 0u64;
    let mut frontier_total = 0u64;
    let mut econ_calls = 0usize;
    let mut naive_unverified_hard = 0usize;
    let mut naive_verified_total = 0usize;
    for &seed in &SEEDS {
        let economy = drive_daemon_lane(seed, "economy");
        let frontier = drive_daemon_lane(seed, "frontier");
        assert_eq!(
            economy.verified(),
            CERT_MIX.len(),
            "seed {seed}: the daemon-path economy lane must verify every task"
        );
        assert_eq!(
            frontier.verified(),
            CERT_MIX.len(),
            "seed {seed}: the frontier control must verify every task"
        );
        // The router is never bypassed: every settled paid call of the
        // corpus is exactly one recorded policy consult, exactly one
        // settled-outcome record, and every recorded decision names a
        // candidate the RouterService was built over.
        assert_eq!(
            economy.attempts(),
            economy.traces.len(),
            "seed {seed}: record count {} must equal paid call count {}",
            economy.traces.len(),
            economy.attempts()
        );
        assert_eq!(
            economy.settled_calls,
            economy.attempts(),
            "seed {seed}: settled outcomes {} must equal paid calls {}",
            economy.settled_calls,
            economy.attempts()
        );
        assert_eq!(
            economy.traces.len(),
            economy.paid_traces().len(),
            "seed {seed}: no consult may be refused on the corpus (fail-closed rows \
             would mean an unpaid task)"
        );
        let paid = economy.paid_traces();
        for (attempt, trace) in economy.tasks.iter().flat_map(|t| &t.attempts).zip(paid) {
            let decision = trace.decision.as_ref().expect("paid traces are Ok");
            assert_eq!(decision.provider, attempt.provider);
            assert_eq!(decision.model, attempt.model);
            assert_eq!(decision.estimated_cost_micro, attempt.cost_micro);
            assert!(
                decision
                    .reasoning
                    .contains(&format!("chosen={}/{}", attempt.provider, attempt.model))
                    || decision.reasoning.contains(&format!(
                        "expected-cost chosen={}/{}",
                        attempt.provider, attempt.model
                    )),
                "the recorded decision must be the router's audited choice: {}",
                decision.reasoning
            );
        }
        assert!(
            economy.escalations() >= 1,
            "seed {seed}: hard rows must escalate through the policy"
        );
        assert!(
            economy.had_escalation_sequence(),
            "seed {seed}: the fail-fail-escalate-succeed sequence must appear"
        );
        let ec = economy.cost_to_success_micro();
        let fc = frontier.cost_to_success_micro();
        eprintln!(
            "[daemon-gate] seed {seed}: economy_cts={ec} micro ({}) frontier_cts={fc} micro ({}) \
             economy_calls={}",
            economy.cost_micro(),
            frontier.cost_micro(),
            economy.attempts()
        );
        assert!(
            ec * 20 <= fc * 21,
            "seed {seed}: daemon-path economy cts {ec} must be <= 1.05x frontier cts {fc}"
        );
        econ_total += economy.cost_micro();
        frontier_total += frontier.cost_micro();
        econ_calls += economy.attempts();
        // Escalation-only-when-cheap-failed rows behave: the naive control
        // (same policy object, driver escalation disabled) loses the hard
        // rows the economy lane verifies.
        let naive = drive_daemon_lane(seed, "naive");
        assert_eq!(naive.traces.len(), naive.attempts());
        assert_eq!(naive.settled_calls, naive.attempts());
        assert_eq!(naive.traces.len(), naive.paid_traces().len());
        naive_unverified_hard += naive
            .tasks
            .iter()
            .filter(|t| !t.verified && CERT_MIX[t.task].difficulty == 2)
            .count();
        naive_verified_total += naive.verified();
        assert_eq!(naive.escalations(), 0, "the naive control never escalates");
    }
    eprintln!(
        "[daemon-gate] aggregate over {} seeds: economy={econ_total} micro frontier={frontier_total} \
         micro ({:.1}%) paid_calls={econ_calls} naive_verified={naive_verified_total}/{}",
        SEEDS.len(),
        econ_total as f64 * 100.0 / frontier_total.max(1) as f64,
        CERT_MIX.len() * SEEDS.len()
    );
    assert!(
        econ_total * 20 <= frontier_total * 21,
        "aggregate daemon-path economy {econ_total} must be <= 1.05x frontier {frontier_total}"
    );
    assert!(
        naive_verified_total < CERT_MIX.len() * SEEDS.len(),
        "cheapest-always must fail the naive policy gate in aggregate: {naive_verified_total} \
         verified vs the economy lane's {}",
        CERT_MIX.len() * SEEDS.len()
    );
    assert!(
        naive_unverified_hard >= 1,
        "the naive cheapest-always control must leave hard rows unverified: \
         observed {naive_unverified_hard} over {} seeds",
        SEEDS.len()
    );
}

/// (2) MaximumQuality mode gate: the mode exists on the config surface
/// (RoutingMode serde), the policy routes every request to the top
/// phase-quality tier (never below the request floor, hard caps honored)
/// and the lane's verified completion must be >= 99% of the frontier
/// lane's verified completion. Cost is NOT required to beat the frontier
/// — reported in both directions.
#[test]
fn daemon_maximum_quality_mode_gate_holds_independently() {
    let mut mq_total = 0u64;
    let mut frontier_total = 0u64;
    let mut mq_verified = 0usize;
    let mut frontier_verified = 0usize;
    for &seed in &SEEDS {
        let mq = drive_daemon_lane(seed, "max-quality");
        let frontier = drive_daemon_lane(seed, "frontier");
        assert_eq!(mq.verified(), CERT_MIX.len());
        assert_eq!(frontier.verified(), CERT_MIX.len());
        assert_eq!(mq.attempts(), mq.traces.len());
        assert_eq!(mq.settled_calls, mq.attempts());
        // Every recorded decision sits at the maximum phase quality any
        // paid candidate offers for that phase (the top tier — the mode
        // never trades quality down while the top tier clears the caps).
        let paid = paid_candidates();
        for trace in mq.paid_traces() {
            let decision = trace.decision.as_ref().expect("Ok trace");
            let chosen = paid
                .iter()
                .find(|c| c.provider == decision.provider && c.model == decision.model)
                .expect("decision names a candidate");
            let max_q = paid
                .iter()
                .map(|c| router_phase_quality(c, trace.phase))
                .max()
                .unwrap();
            assert_eq!(
                router_phase_quality(chosen, trace.phase),
                max_q,
                "max-quality must route the top quality tier for {:?}: {}/{}",
                trace.phase,
                chosen.provider,
                chosen.model
            );
        }
        mq_total += mq.cost_micro();
        frontier_total += frontier.cost_micro();
        mq_verified += mq.verified();
        frontier_verified += frontier.verified();
        eprintln!(
            "[daemon-gate] seed {seed}: max-quality verified {}/{} cost {} micro; \
             frontier verified {}/{} cost {} micro",
            mq.verified(),
            CERT_MIX.len(),
            mq.cost_micro(),
            frontier.verified(),
            CERT_MIX.len(),
            frontier.cost_micro()
        );
    }
    // Quality gate: >= 99% of the frontier lane's verified completion.
    assert!(
        mq_verified * 100 >= frontier_verified * 99,
        "max-quality verified {mq_verified}/{} must be >= 99% of frontier verified \
         {frontier_verified}/{}",
        CERT_MIX.len() * SEEDS.len(),
        CERT_MIX.len() * SEEDS.len()
    );
    // Cost direction: NOT required to beat the frontier (asserted in both
    // directions, reported — the gate is the quality floor, not economy).
    let ratio = if frontier_total == 0 {
        f64::MAX
    } else {
        mq_total as f64 * 100.0 / frontier_total as f64
    };
    eprintln!(
        "[daemon-gate] maximum-quality aggregate: cost {mq_total} micro vs frontier \
         {frontier_total} micro ({ratio:.1}%); verified {mq_verified} vs {frontier_verified}"
    );
    assert!(
        mq_verified >= frontier_verified,
        "max-quality may never verify FEWER tasks than the frontier lane"
    );
}

/// (3a) Adversarial: a corpus row that defeats cheapest-always (hard task:
/// the cheap model fails twice, the expensive model succeeds after the
/// escalation) still passes the economy-vs-frontier gate, and fails a
/// naive policy gate (the naive control never escalates and leaves hard
/// rows unverified, whose honest redo cost exceeds the economy lane's
/// spend).
#[test]
fn daemon_adversarial_hard_row_defeats_naive_cheapest_but_passes_economy_gate() {
    let mut naive_verified_total = 0usize;
    let mut naive_unverified_hard = 0usize;
    let mut naive_lucky_seeds = 0usize;
    let mut sequence_tasks = 0usize;
    for &seed in &SEEDS {
        let economy = drive_daemon_lane(seed, "economy");
        let naive = drive_daemon_lane(seed, "naive");
        let frontier = drive_daemon_lane(seed, "frontier");
        assert_eq!(economy.verified(), CERT_MIX.len());
        assert_eq!(frontier.verified(), CERT_MIX.len());
        assert_eq!(economy.traces.len(), economy.paid_traces().len());
        // The naive lane leaves ONLY hard rows unverified (difficulty 2) —
        // the corpus rows that defeat cheapest-always. On a lucky seed the
        // cheap draws may all land and the naive lane verifies everything;
        // the gate counts that as a defeated-check across seeds below.
        for task in naive.tasks.iter().filter(|t| !t.verified) {
            assert_eq!(
                CERT_MIX[task.task].difficulty, 2,
                "seed {seed}: naive cheapest-always only loses hard rows, task {} \
                 (difficulty {})",
                task.task, CERT_MIX[task.task].difficulty
            );
        }
        naive_verified_total += naive.verified();
        naive_unverified_hard += naive
            .tasks
            .iter()
            .filter(|t| !t.verified && CERT_MIX[t.task].difficulty == 2)
            .count();
        if naive.verified() == CERT_MIX.len() {
            naive_lucky_seeds += 1;
        }
        // The economy lane's hard rows show the fail-fail-escalate-succeed
        // shape in its recorded consults, and the escalation is visible in
        // the trace stream as a requested-floor jump AFTER the two failed
        // cheap consults (the naive lane's floor series never moves).
        for t in economy.tasks.iter().filter(|t| t.task >= 7 && t.verified) {
            if t.attempts.len() >= 3
                && !t.attempts[0].ok
                && !t.attempts[1].ok
                && t.attempts[2].ok
                && t.attempts[2].escalated
            {
                sequence_tasks += 1;
                let series = economy.floor_series(t.task);
                assert_eq!(series.len(), t.attempts.len());
                assert_eq!(series[0], request_for(t.task).quality_floor);
                assert!(
                    series[2] > series[0],
                    "seed {seed} task {}: escalation must raise the requested floor after \
                     two cheap failures: {series:?}",
                    t.task
                );
                assert!(
                    series.windows(2).all(|w| w[0] <= w[1]),
                    "seed {seed} task {}: floors never drop mid-task: {series:?}",
                    t.task
                );
            }
        }
        let naive_series_constant = naive.tasks.iter().all(|t| {
            let series = naive.floor_series(t.task);
            series
                .iter()
                .all(|&f| f == request_for(t.task).quality_floor)
        });
        assert!(
            naive_series_constant,
            "seed {seed}: the naive control's floor series must never move"
        );
        let effective = naive_effective_cost(seed, &naive);
        eprintln!(
            "[daemon-gate] seed {seed}: economy spend {} micro vs naive cheapest-always \
             effective spend {effective} micro (naive verified {}/{})",
            economy.cost_micro(),
            naive.verified(),
            CERT_MIX.len()
        );
    }
    eprintln!(
        "[daemon-gate] adversarial aggregate over {} seeds: naive cheapest-always verified \
         {naive_verified_total}/{} ({naive_unverified_hard} hard rows unverified, \
         {naive_lucky_seeds} lucky seeds); economy lane verified {}/{}; hard \
         fail-fail-escalate-succeed sequences {sequence_tasks}",
        SEEDS.len(),
        CERT_MIX.len() * SEEDS.len(),
        CERT_MIX.len() * SEEDS.len(),
        CERT_MIX.len() * SEEDS.len()
    );
    assert!(
        naive_verified_total < CERT_MIX.len() * SEEDS.len(),
        "cheapest-always must fail the naive policy gate in aggregate: \
         {naive_verified_total} verified vs the economy lane's {} over {} seeds",
        CERT_MIX.len() * SEEDS.len(),
        SEEDS.len()
    );
    assert!(
        naive_unverified_hard > 0,
        "cheapest-always must leave at least one hard row unverified over the seeds"
    );
    assert!(
        sequence_tasks >= SEEDS.len(),
        "the fail-fail-escalate-succeed shape must appear on hard rows across seeds, \
         observed {sequence_tasks}"
    );
}

/// (3b) Hostile routing-mode values, typed: MaximumQuality/Balanced are
/// valid config strings on the same serde surface the daemon config parses
/// (`Option<RoutingMode>`), and hostile shapes are rejected — wrong case,
/// object-shaped unit variants, wrong types, incomplete pins, unknown
/// sentinels.
#[test]
fn routing_mode_wire_shapes_typed_and_hostile_values_rejected() {
    let good: [(RoutingMode, serde_json::Value); 4] = [
        (RoutingMode::Economy, serde_json::json!("economy")),
        (
            RoutingMode::MaximumQuality,
            serde_json::json!("maximum_quality"),
        ),
        (RoutingMode::Balanced, serde_json::json!("balanced")),
        (
            RoutingMode::Pinned {
                provider: "p".into(),
                model: "m".into(),
            },
            serde_json::json!({"pinned": {"provider": "p", "model": "m"}}),
        ),
    ];
    for (mode, wire) in good {
        let back: RoutingMode = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(back, mode, "wire {wire} must parse");
        let again = serde_json::to_value(&mode).unwrap();
        assert_eq!(again, wire, "wire shape must round-trip");
    }
    assert!(RoutingMode::MaximumQuality.pinned().is_none());
    assert!(RoutingMode::Balanced.pinned().is_none());
    assert!(!RoutingMode::MaximumQuality.is_economy());
    assert!(!RoutingMode::Balanced.is_economy());
    // Unknown extra fields inside a variant payload are IGNORED by the
    // derived serde surface (the daemon config's Option<RoutingMode>
    // behaves identically — hostile configs are rejected on the axes that
    // matter: the variant tag, the field presence and the value types).
    let extra = serde_json::from_value::<RoutingMode>(serde_json::json!({
        "pinned": {"provider": "p", "model": "m", "extra": 1}
    }))
    .unwrap();
    assert_eq!(
        extra,
        RoutingMode::Pinned {
            provider: "p".into(),
            model: "m".into(),
        }
    );
    for bad in [
        serde_json::json!("MaximumQuality"),
        serde_json::json!("max_quality"),
        serde_json::json!("BALANCED"),
        serde_json::json!("auto"),
        serde_json::json!(42),
        serde_json::json!(true),
        serde_json::json!({"maximum_quality": {}}),
        serde_json::json!({"balanced": {}}),
        serde_json::json!({"economy": {}}),
        serde_json::json!({"pinned": {"provider": "p"}}),
        serde_json::json!({"mode": "maximum_quality"}),
    ] {
        let parsed: Result<RoutingMode, _> = serde_json::from_value(bad.clone());
        assert!(
            parsed.is_err(),
            "hostile routing mode must be rejected: {bad}"
        );
    }
}

/// (3c) RecordingPolicy ordering is deterministic: two identical lane runs
/// through two identical production policy objects produce byte-identical
/// trace streams, the consult ordinals are strictly sequential, and every
/// task's requested-floor series is monotone (escalation state only ever
/// demands MORE quality, never less).
#[test]
fn daemon_recording_policy_ordering_is_deterministic() {
    let a = drive_daemon_lane(SEEDS[0], "economy");
    let b = drive_daemon_lane(SEEDS[0], "economy");
    assert_eq!(
        a.traces, b.traces,
        "identical runs must record identical consult traces"
    );
    assert_eq!(
        a.traces.len(),
        a.attempts(),
        "no refused consults on the corpus"
    );
    for (i, trace) in a.traces.iter().enumerate() {
        assert_eq!(
            trace.consult,
            i + 1,
            "consult ordinals must be strictly sequential"
        );
    }
    for task in &a.tasks {
        let series = a.floor_series(task.task);
        assert_eq!(series.len(), task.attempts.len());
        assert!(
            series.windows(2).all(|w| w[0] <= w[1]),
            "task {}: requested floors must be monotone within a task: {series:?}",
            task.task
        );
    }
    // Trace content is fully determined by (seed, lane): the mode and the
    // stability-consult flag are recorded identically on every consult.
    assert!(
        a.traces.iter().all(|t| !t.stability_consulted),
        "no prefix history is supplied on the corpus lane"
    );
    assert!(
        a.traces.iter().all(|t| t.mode == RoutingMode::Economy),
        "the lane's policy mode is recorded per consult"
    );
}

/// The cli mirror is exact: the graph.rs descriptor mapping (capabilities
/// -> descriptor with DEFAULT economics) reproduces the corpus descriptors
/// field-for-field apart from the pricing table, and the pin-collapse
/// build refuses unknown pins like the cli does.
#[test]
fn daemon_mirror_locks_the_cli_descriptor_mapping_and_pin_fail_closed() {
    let corpus = paid_candidates();
    let registry = mirror_registry(&corpus);
    let mapped = mirror_candidates(&registry, &RoutingMode::Economy).unwrap();
    assert_eq!(mapped.len(), corpus.len());
    let mut mapped_sorted = mapped.clone();
    mapped_sorted.sort_by(|a, b| (&a.provider, &a.model).cmp(&(&b.provider, &b.model)));
    let mut corpus_sorted = corpus.clone();
    corpus_sorted.sort_by(|a, b| (&a.provider, &a.model).cmp(&(&b.provider, &b.model)));
    for (m, c) in mapped_sorted.iter().zip(&corpus_sorted) {
        assert_eq!(m.provider, c.provider);
        assert_eq!(m.model, c.model);
        assert_eq!(m.context, c.context);
        assert_eq!(m.max_output, c.max_output);
        assert_eq!(m.tools, c.tools);
        assert_eq!(m.parallel_tools, c.parallel_tools);
        assert_eq!(m.reasoning, c.reasoning);
        assert_eq!(m.thinking, c.thinking);
        assert_eq!(m.vision, c.vision);
        assert_eq!(m.structured_output, c.structured_output);
        assert_eq!(m.embeddings, c.embeddings);
        assert_eq!(m.streaming, c.streaming);
        assert_eq!(m.source, c.source);
        assert_eq!(
            m.economics,
            ModelEconomics::default(),
            "the cli mapping prices nothing (default economics) — the corpus pricing \
             is the documented mirror delta"
        );
        assert!(!c.economics.is_local_zero_cost());
        let _ = &c.economics;
    }
    // Pricing application is total over the mapped set (no orphan rows).
    let mut priced = mapped.clone();
    apply_prices(&mut priced, &corpus);
    for (m, c) in priced.iter().zip(&corpus_sorted) {
        assert_eq!(m.economics, c.economics);
    }
    // The Pinned mirror refuses unknown providers/models exactly like the
    // cli boot check (fail closed, never a silent Economy).
    let bad = mirror_candidates(
        &registry,
        &RoutingMode::Pinned {
            provider: "ghost".into(),
            model: "m".into(),
        },
    );
    assert!(bad.is_err());
    let bad_model = mirror_candidates(
        &registry,
        &RoutingMode::Pinned {
            provider: "e1".into(),
            model: "not-a-model".into(),
        },
    );
    assert!(bad_model.is_err());
    let pinned = mirror_candidates(
        &registry,
        &RoutingMode::Pinned {
            provider: "e1".into(),
            model: "cheap".into(),
        },
    )
    .unwrap();
    assert_eq!(pinned.len(), 1);
    assert_eq!(
        (pinned[0].provider.as_str(), pinned[0].model.as_str()),
        ("e1", "cheap")
    );
}

/// Balanced mode semantics: expected-cost routing at the quality band —
/// on cheap-floor rows where Economy routes the cheap model, Balanced
/// routes the band tier (never below BALANCED_QUALITY_FLOOR while a band
/// candidate clears the caps), deterministic, and MaximumQuality/Balanced
/// hard caps hold under a tight budget (nothing below the request floor
/// and nothing over the budget).
#[test]
fn daemon_balanced_and_maximum_quality_mode_semantics_under_caps() {
    let corpus = paid_candidates();
    let balanced = daemon_policy(&corpus, RoutingMode::Balanced);
    assert_eq!(balanced.mode(), RoutingMode::Balanced);
    let economy = daemon_policy(&corpus, RoutingMode::Economy);
    let mut balanced_total = 0u64;
    let mut economy_total = 0u64;
    for task in 0..CERT_MIX.len() {
        let req = request_for(task);
        let eb = balanced.route(&req).unwrap();
        let ee = economy.route(&req).unwrap();
        let bq = router_phase_quality(
            corpus
                .iter()
                .find(|c| c.provider == eb.provider && c.model == eb.model)
                .unwrap(),
            req.phase,
        );
        assert!(
            bq >= EconomicRoutingPolicy::BALANCED_QUALITY_FLOOR.min(95),
            "balanced must stay at the quality band: chosen {}/{} quality {bq}",
            eb.provider,
            eb.model
        );
        balanced_total += eb.estimated_cost_micro;
        economy_total += ee.estimated_cost_micro;
        let again = balanced.route(&req).unwrap();
        assert_eq!(eb, again, "balanced decisions must be deterministic");
    }
    eprintln!(
        "[daemon-gate] balanced aggregate {balanced_total} micro vs economy {economy_total} \
         micro ({:.1}%)",
        balanced_total as f64 * 100.0 / economy_total.max(1) as f64
    );
    assert!(
        balanced_total >= economy_total,
        "balanced may cost more than economy (band floor), never less overall"
    );
    // Hard caps under MaximumQuality: a budget that excludes the top tier
    // must drop the decision down the quality ladder — never above the
    // budget, never below the request floor.
    let mq = daemon_policy(&corpus, RoutingMode::MaximumQuality);
    assert_eq!(mq.mode(), RoutingMode::MaximumQuality);
    let mut budget_req = request_for(8); // Debug 40k ctx / 4k out, floor 80
    let full = mq.route(&budget_req).unwrap();
    let full_q = router_phase_quality(
        corpus
            .iter()
            .find(|c| c.provider == full.provider && c.model == full.model)
            .unwrap(),
        budget_req.phase,
    );
    let max_q = corpus
        .iter()
        .map(|c| router_phase_quality(c, budget_req.phase))
        .max()
        .unwrap();
    assert_eq!(
        full_q, max_q,
        "unbounded maximum-quality routes the top tier"
    );
    // The top tier (f1 at 840_000 micro on this row) is over budget.
    budget_req.task_budget_remaining_micro = 500_000;
    let capped = mq.route(&budget_req).unwrap();
    assert!(
        capped.estimated_cost_micro <= budget_req.task_budget_remaining_micro,
        "hard budget caps hold under maximum-quality: {} > {}",
        capped.estimated_cost_micro,
        budget_req.task_budget_remaining_micro
    );
    let capped_q = router_phase_quality(
        corpus
            .iter()
            .find(|c| c.provider == capped.provider && c.model == capped.model)
            .unwrap(),
        budget_req.phase,
    );
    let best_in_budget = corpus
        .iter()
        .map(|c| router_phase_quality(c, budget_req.phase))
        .filter(|&q| {
            let cost = faktor_router::estimated_call_cost(
                &corpus
                    .iter()
                    .find(|d| router_phase_quality(d, budget_req.phase) == q)
                    .unwrap()
                    .economics,
                budget_req.context_tokens,
                budget_req.estimated_output_tokens,
                0,
                0,
            );
            cost <= budget_req.task_budget_remaining_micro
        })
        .max()
        .unwrap_or(0);
    assert_eq!(
        capped_q, best_in_budget,
        "maximum-quality under a cap must route the best quality the budget admits"
    );
    assert!(
        capped_q >= budget_req.quality_floor,
        "never below the request floor"
    );
    // Balanced mode never over-runs the band when the request floor demands
    // even more: a 95-floor request stays at 95+ quality.
    let mut hard = req(&CORPUS[3], 0);
    hard.quality_floor = 95;
    let d = balanced.route(&hard).unwrap();
    assert!(
        router_phase_quality(
            corpus
                .iter()
                .find(|c| c.provider == d.provider && c.model == d.model)
                .unwrap(),
            hard.phase,
        ) >= 95
    );
}

// ====================================================================
// Verified-outcome learning (audit items 13/14/L): Economy consumes
// conservative WorkCostEstimates from durable verified history. The same
// paid corpus routes the CHEAP model while no verified stats exist and the
// STRONG model once per-phase rework history does — expected cost to
// VERIFIED completion = immediate + P(rework) x downstream spend, where a
// two-sample track record stays far below the "excellent" bar and failed
// verification is never learned as a success.
// ====================================================================

#[allow(unused_imports)]
use faktor_core::model::{RiskBucket, TaskClass};
#[allow(unused_imports)]
use faktor_router::OutcomeStore;

// The scenario helpers are exercised by the #[cfg(test)] gates below; the
// lib target itself only hosts them (like the kit/cert/daemon_gate mods).
#[allow(dead_code)]
fn economy_key(
    provider: &str,
    model: &str,
    phase: RouterPhase,
    class: TaskClass,
    bucket: RiskBucket,
) -> faktor_router::OutcomeKey {
    faktor_router::OutcomeKey {
        provider: provider.into(),
        model: model.into(),
        phase,
        task_class: class,
        risk_bucket: bucket,
    }
}

#[allow(dead_code)]
fn economy_sample(verified_success: bool, rework: u64) -> faktor_router::OutcomeSample {
    faktor_router::OutcomeSample {
        verified_success,
        rework_cost_micro: rework,
        rework_turns: 1,
    }
}

// ====================================================================
// Verified-outcome learning (audit items 13/14/L): Economy consumes
// conservative WorkCostEstimates from durable verified history. The same
// paid corpus routes the CHEAP model while no verified stats exist and the
// STRONG model once per-phase rework history does — expected cost to
// VERIFIED completion = immediate + P(rework) x downstream spend, where a
// two-sample track record stays far below the "excellent" bar and failed
// verification is never learned as a success.
// ====================================================================

#[test]
fn verified_rework_history_flips_economy_to_strong_after_cheap_before_stats() {
    let cheap = desc("e1", "cheap", (82, 82, 81), (1, 3), 800);
    let strong = desc("f1", "big", (95, 95, 95), (15, 60), 400);
    let candidates = vec![cheap, strong];
    let req = req(&(RouterPhase::Implement, 40_000, 6_000, 80), 0);
    // No verified stats exist: the legacy expected-cost prior routes cheap
    // (58k base + small retry/escalation terms vs the strong 960k base).
    let before = faktor_router::RouterService::new(candidates.clone())
        .route(&req, &[])
        .unwrap();
    assert_eq!(
        (before.provider.as_str(), before.model.as_str()),
        ("e1", "cheap"),
        "cheap must win BEFORE verified stats exist: {}",
        before.reasoning
    );
    assert!(
        !before.reasoning.contains("rework_ppm"),
        "no history, no verified tag: {}",
        before.reasoning
    );
    // Seed verified history under several class/risk buckets of the
    // Implement phase (the route consult folds them per phase):
    // cheap 45% rework (each failure's downstream spend measured at 2.4M),
    // strong 6% rework (each at 960k).
    let store = faktor_router::MemoryOutcomeStore::new();
    for (class, bucket) in [
        (
            faktor_core::model::TaskClass::Medium,
            faktor_core::model::RiskBucket::Low,
        ),
        (
            faktor_core::model::TaskClass::Hard,
            faktor_core::model::RiskBucket::High,
        ),
    ] {
        for _ in 0..55u64 {
            store.append_sample(
                &economy_key("e1", "cheap", RouterPhase::Implement, class, bucket),
                economy_sample(true, 0),
            );
        }
        for _ in 0..45u64 {
            store.append_sample(
                &economy_key("e1", "cheap", RouterPhase::Implement, class, bucket),
                economy_sample(false, 2_400_000),
            );
        }
    }
    for _ in 0..94u64 {
        store.append_sample(
            &economy_key(
                "f1",
                "big",
                RouterPhase::Implement,
                faktor_core::model::TaskClass::Medium,
                faktor_core::model::RiskBucket::Low,
            ),
            economy_sample(true, 0),
        );
    }
    for _ in 0..6u64 {
        store.append_sample(
            &economy_key(
                "f1",
                "big",
                RouterPhase::Implement,
                faktor_core::model::TaskClass::Medium,
                faktor_core::model::RiskBucket::Low,
            ),
            economy_sample(false, 960_000),
        );
    }
    // The registry sees the exact folded totals the routing consult reads.
    let cheap_stats = store
        .phase_stats("e1", "cheap", RouterPhase::Implement)
        .unwrap();
    assert_eq!(cheap_stats.sample_count, 200);
    assert_eq!(cheap_stats.successes_first_pass, 110);
    assert_eq!(cheap_stats.failures_first_pass, 90);
    // With verified stats existing, Economy must pick the strong model:
    // cheap ≈ 58k + 520_650ppm x 2.4M ≈ 1.31M vs strong ≈ 960k +
    // 126_477ppm x 960k ≈ 1.08M.
    let svc = faktor_router::RouterService::with_outcomes(candidates, std::sync::Arc::new(store));
    let after = svc.route(&req, &[]).unwrap();
    assert_eq!(
        (after.provider.as_str(), after.model.as_str()),
        ("f1", "big"),
        "verified rework history must flip Economy to the strong model: {}",
        after.reasoning
    );
    assert!(
        after.reasoning.contains("verified rework_ppm="),
        "the audit string names the conservative verified estimate: {}",
        after.reasoning
    );
    assert!(
        after.reasoning.contains("total_expected_micro=1081"),
        "the audit string names the verified total: {}",
        after.reasoning
    );
    // Deterministic over the same history.
    let again = svc.route(&req, &[]).unwrap();
    assert_eq!(after, again);
    // The budget axis still sees the BASE cost (conservative downstream
    // budget math), never the inflated verified expectation.
    assert_eq!(after.estimated_cost_micro, 960_000);
}

/// Two verified first-pass successes must NOT license the cheaper model:
/// the conservative rework bound keeps its expected verified cost above a
/// fresh model's legacy expected cost even when trusting 2/2 as excellent
/// would flip the decision the other way.
#[test]
fn two_verified_successes_remain_conservative_in_economy() {
    let proven = desc("p1", "proven2x", (88, 88, 88), (4, 30), 500);
    let fresh = desc("p2", "fresh", (88, 88, 88), (10, 25), 500);
    let candidates = vec![proven, fresh];
    let req = RouteRequest {
        phase: RouterPhase::Implement,
        required_capabilities: vec!["tools".into(), "streaming".into()],
        context_tokens: 10_000,
        estimated_output_tokens: 2_000,
        quality_floor: 60,
        task_budget_remaining_micro: 0,
        latency_preference_ms: None,
        ..Default::default()
    };
    // 100k base (10k x 4 + 2k x 30) vs 150k base (10k x 10 + 2k x 25).
    let store = faktor_router::MemoryOutcomeStore::new();
    for _ in 0..2u64 {
        store.append_sample(
            &economy_key(
                "p1",
                "proven2x",
                RouterPhase::Implement,
                faktor_core::model::TaskClass::Medium,
                faktor_core::model::RiskBucket::Low,
            ),
            economy_sample(true, 0),
        );
    }
    let svc = faktor_router::RouterService::with_outcomes(candidates, std::sync::Arc::new(store));
    let d = svc.route(&req, &[]).unwrap();
    assert_eq!(
        (d.provider.as_str(), d.model.as_str()),
        ("p2", "fresh"),
        "2/2 verified successes must NOT license the 100k candidate over the 150k fresh one: {}",
        d.reasoning
    );
    // Conservative confidence: two clean samples sit ~333k ppm, far below
    // the documented 900k "excellent" bar.
    let stats = svc
        .outcomes
        .phase_stats("p1", "proven2x", RouterPhase::Implement)
        .unwrap();
    assert!(
        faktor_router::verified_success_confidence_ppm(&stats) < 400_000,
        "two successes stay conservative: {:?}",
        stats
    );
    // And the ECONOMY estimate mirrors it: 100k base + 666_667ppm of the
    // 150k escalation spend keeps the proven candidate above the fresh one.
    let est = faktor_router::work_cost_estimate(100_000, Some(&stats), 150_000);
    assert_eq!(est.rework_probability_ppm, 666_667);
    assert_eq!(est.total_expected_micro, 200_001);
}

/// Verified-only attribution end to end: three calls that "the model said
/// were done" but failed deterministic verification record FAILURES (with
/// their rework), never successes — and Economy's next consult prices the
/// model's honest rework risk instead of trusting it.
#[test]
fn failed_verification_records_no_success_and_economy_prices_the_rework() {
    let cheap = desc("e1", "cheap", (82, 82, 81), (1, 3), 800);
    let strong = desc("f1", "big", (95, 95, 95), (15, 60), 400);
    let candidates = vec![cheap, strong];
    let req = req(&(RouterPhase::Implement, 40_000, 6_000, 80), 0);
    let store = faktor_router::MemoryOutcomeStore::new();
    for _ in 0..3u64 {
        store.append_sample(
            &economy_key(
                "e1",
                "cheap",
                RouterPhase::Implement,
                faktor_core::model::TaskClass::Hard,
                faktor_core::model::RiskBucket::High,
            ),
            economy_sample(false, 2_400_000),
        );
    }
    let stats = store
        .stats(&economy_key(
            "e1",
            "cheap",
            RouterPhase::Implement,
            faktor_core::model::TaskClass::Hard,
            faktor_core::model::RiskBucket::High,
        ))
        .expect("the failed-verification key must have history");
    assert_eq!(
        stats.successes_first_pass, 0,
        "failed verification is never learned as a success"
    );
    assert_eq!(stats.failures_first_pass, 3);
    assert_eq!(stats.rework_cost_micro_sum, 3 * 2_400_000);
    assert_eq!(stats.sample_count, 3);
    // The Economy consult prices the never-verified model at the saturated
    // conservative rework bound: cheap = 58k + 1_000_000ppm x 2.4M measured
    // rework spend = 2.458M — far above the strong model's ~1.06M legacy
    // expected cost, so the strong model wins (a model that never verified
    // must not keep winning because it is cheap).
    let fold = store
        .phase_stats("e1", "cheap", RouterPhase::Implement)
        .unwrap();
    let cheap_verified = faktor_router::work_cost_estimate(58_000, Some(&fold), 960_000);
    assert_eq!(cheap_verified.rework_probability_ppm, 1_000_000);
    assert_eq!(cheap_verified.total_expected_micro, 2_458_000);
    let svc = faktor_router::RouterService::with_outcomes(candidates, std::sync::Arc::new(store));
    let d = svc.route(&req, &[]).unwrap();
    assert_eq!(
        (d.provider.as_str(), d.model.as_str()),
        ("f1", "big"),
        "a never-verified cheap model must lose: {}",
        d.reasoning
    );
    // The strong winner's own audit stays legacy (no verified history for
    // it): the flip came from the cheap model's honest verified record.
    assert!(!d.reasoning.contains("rework_ppm="), "{}", d.reasoning);
}
