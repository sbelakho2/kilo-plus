//! Economic-router policy benchmark (audit): economy must approach frontier
//! verified completion at <=65% cost; micro-unit accounting integer-exact;
//! hard budgets never overshoot; local models cost-zero but latency-gated.

#[allow(unused_imports, dead_code)]
pub mod kit {
    pub use faktor_core::model::{
        ModelDescriptor, ModelEconomics, ModelSource, RateLimitState, RouterPhase,
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
            input_price_per_mtok: input,
            output_price_per_mtok: output,
            cache_read_price_per_mtok: input / 5,
            cache_write_price_per_mtok: input / 2,
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
