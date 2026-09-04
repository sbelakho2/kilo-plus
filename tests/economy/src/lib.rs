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

    pub fn econ(input: u64, output: u64, tool: u8, code: u8, ctx: u8, latency: u64) -> ModelEconomics {
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

    pub fn desc(p: &str, m: &str, q: (u8, u8, u8), price: (u64, u64), latency: u64) -> ModelDescriptor {
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
            ed.provider, ed.model, ed.estimated_cost_micro, fd.estimated_cost_micro, item.0
        );
        if item.3 <= 70 {
            assert!(
                ed.estimated_cost_micro * 100 <= fd.estimated_cost_micro * 65,
                "economy {}/{} cost {} > 65% frontier {} for {:?}",
                ed.provider, ed.model, ed.estimated_cost_micro, fd.estimated_cost_micro, item.0
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
        assert_ne!(d.provider, "flop", "flop quality 55 must lose floor {}: {}", item.3, d.reasoning);
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
    assert_eq!(d2.provider, "ollama", "zero-cost local wins without latency cap");
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
    assert_eq!(faktor_router::estimated_call_cost(&econ(1, 1, 80, 80, 80, 100), 1, 0, 0, 0), 1);
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
