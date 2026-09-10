//! Randomized routing economic property tests (audit item 101).
//!
//! Every test generates seeded LCG cases over prices (exact, local-zero,
//! conservative-ceiling, unknown), token shapes, cache hits, quality,
//! verified success posteriors, rework costs and hard caps, and asserts the
//! audit's economic invariants against the real router / policy surfaces:
//!
//! - unknown price is NEVER treated as free;
//! - a hard cap never admits a known estimate above the free balance;
//! - price is never compared against latency (structurally distinct types;
//!   the scored ladder is cost -> success -> latency -> base -> name);
//! - a hard quality floor is NEVER lowered by any mode (a floor with no
//!   candidate above it is a typed `NoCapableModel`, never a decision);
//! - Economy minimizes expected verified-work cost on generated pairs;
//! - MaximumQuality does not discard unknown-price high-quality models when
//!   there is no cap.

#![cfg_attr(
    not(test),
    allow(dead_code, unused_imports, unused_variables, unused_mut)
)]

use std::any::type_name;
use std::cmp::Ordering;
use std::sync::Arc;

use faktor_agent::{EconomicRoutingPolicy, RouteFailure, RoutingPolicy};
use faktor_core::model::{
    MicroUsdPerMillionTokens, MicroUsdPerToken, ModelDescriptor, ModelEconomics, ModelSource,
    PriceAuthority, PriceQuote, PricingSnapshot, RateLimitState, RouterPhase, RoutingMode,
};
use faktor_router::outcomes::{
    rework_probability_ppm, verified_success_confidence_ppm, work_cost_estimate,
    MemoryOutcomeStore, OutcomeKey, OutcomeSample, OutcomeStore, VerifiedOutcomeStats,
};
use faktor_router::{
    qualified_candidates, CacheState, LiveHealth, QualifiedCandidate, RouteRequest, Router,
    RouterService, ScoredCandidate,
};

/// All tests use fixed seeds: deterministic across platforms.
const SEEDS: u64 = 400;

/// True for the phases the router judges on the coding-quality mean.
fn heavy_phase(phase: RouterPhase) -> bool {
    matches!(
        phase,
        RouterPhase::Implement | RouterPhase::Review | RouterPhase::Debug
    )
}

fn phase_quality(d: &ModelDescriptor, phase: RouterPhase) -> u8 {
    if heavy_phase(phase) {
        d.economics.coding_quality()
    } else {
        d.economics.context_reliability
    }
}

/// Seeded numerical-recipe LCG (identical on every platform).
struct Gen {
    state: u64,
}

impl Gen {
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

    fn chance(&mut self, percent: u64) -> bool {
        self.range(0, 100) < percent
    }
}

fn descriptor(
    provider: &str,
    model: &str,
    quality: u8,
    input: u64,
    output: u64,
    latency: u64,
    rate_limit: RateLimitState,
) -> ModelDescriptor {
    ModelDescriptor {
        provider: provider.into(),
        model: model.into(),
        context: 256_000,
        max_output: 16_384,
        tools: true,
        parallel_tools: true,
        reasoning: true,
        thinking: true,
        vision: false,
        structured_output: true,
        embeddings: false,
        streaming: true,
        economics: ModelEconomics {
            input_price_per_mtok: MicroUsdPerToken::from(input),
            output_price_per_mtok: MicroUsdPerToken::from(output),
            cache_read_price_per_mtok: MicroUsdPerToken::from(input / 4),
            cache_write_price_per_mtok: MicroUsdPerToken::from(input / 2),
            estimated_latency_ms: latency,
            tool_reliability: quality,
            reasoning_reliability: quality,
            coding_reliability: quality,
            context_reliability: quality,
            availability: 100,
            rate_limit_state: rate_limit,
        },
        source: ModelSource::ProviderCatalog,
    }
}

fn random_descriptor(g: &mut Gen, idx: u64) -> ModelDescriptor {
    let quality = g.range(30, 101) as u8;
    let input = g.range(0, 80);
    let output = g.range(0, 240);
    let latency = g.range(50, 5000);
    let rate_limit = match g.range(0, 12) {
        0 => RateLimitState::Hard,
        1..=2 => RateLimitState::Soft,
        _ => RateLimitState::Healthy,
    };
    descriptor(
        &format!("p{idx}"),
        &format!("m{idx}"),
        quality,
        input,
        output,
        latency,
        rate_limit,
    )
}

fn random_request(g: &mut Gen) -> RouteRequest {
    let latency_preference_ms = if g.chance(25) {
        Some(g.range(100, 4000))
    } else {
        None
    };
    RouteRequest {
        phase: RouterPhase::ALL[(g.next() % 11) as usize],
        required_capabilities: vec!["tools".into()],
        context_tokens: g.range(1, 200_000),
        estimated_output_tokens: g.range(1, 8192),
        quality_floor: g.range(0, 111) as u8,
        task_budget_remaining_micro: if g.chance(40) { g.range(1, 60_000) } else { 0 },
        latency_preference_ms,
        ..Default::default()
    }
}

fn random_quote(g: &mut Gen) -> PriceQuote {
    PriceQuote {
        input: MicroUsdPerMillionTokens(g.range(0, 5_000_000)),
        output: MicroUsdPerMillionTokens(g.range(0, 20_000_000)),
        cache_read: MicroUsdPerMillionTokens(g.range(0, 1_000_000)),
        cache_write: MicroUsdPerMillionTokens(g.range(0, 3_000_000)),
    }
}

fn key(provider: &str, model: &str, phase: RouterPhase) -> OutcomeKey {
    OutcomeKey {
        provider: provider.into(),
        model: model.into(),
        phase,
        task_class: faktor_core::model::TaskClass::Medium,
        risk_bucket: faktor_core::model::RiskBucket::Low,
    }
}

/// The documented legacy expected-cost formula (the oracle re-implementation
/// of the router's integer math; the property test checks the router against
/// it).
fn legacy_expected(success_ppm: u32, call_cost_micro: u64, escalation: u128) -> u64 {
    let base = u128::from(call_cost_micro);
    let p_fail_ppm = u128::from(1_000_000u64 - u64::from(success_ppm.min(1_000_000)));
    let retry = (base.saturating_mul(p_fail_ppm).saturating_add(1_999_999)) / 2_000_000;
    let escalate = (escalation
        .saturating_mul(p_fail_ppm)
        .saturating_add(1_999_999))
        / 2_000_000;
    u64::try_from(base + retry + escalate).unwrap_or(u64::MAX)
}

/// The oracle escalation cost: the cheapest OTHER distinct qualified
/// candidate's base cost, or 3x the candidate's own base when alone.
fn oracle_escalation(q: &QualifiedCandidate<'_>, qualified: &[QualifiedCandidate<'_>]) -> u128 {
    let mut entries: Vec<(u64, &str, &str)> = qualified
        .iter()
        .map(|c| {
            (
                c.call_cost_micro,
                c.descriptor.provider.as_str(),
                c.descriptor.model.as_str(),
            )
        })
        .collect();
    entries.sort_by_key(|e| (e.0, e.1.to_string(), e.2.to_string()));
    entries.dedup_by(|a, b| a.1 == b.1 && a.2 == b.2);
    match entries
        .iter()
        .find(|e| e.1 != q.descriptor.provider || e.2 != q.descriptor.model)
    {
        Some((cost, _, _)) => u128::from(*cost),
        None => u128::from(q.call_cost_micro).saturating_mul(3),
    }
}

struct OracleRow {
    provider: String,
    model: String,
    expected: u64,
    success_ppm: u32,
    latency_ms: u64,
    base: u64,
}

fn oracle_rows(
    qualified: &[QualifiedCandidate<'_>],
    phase: RouterPhase,
    store: &dyn OutcomeStore,
) -> Vec<OracleRow> {
    qualified
        .iter()
        .map(|q| {
            let escalation = oracle_escalation(q, qualified);
            match store.phase_stats(&q.descriptor.provider, &q.descriptor.model, phase) {
                Some(st) => {
                    let estimate = work_cost_estimate(
                        q.call_cost_micro,
                        Some(&st),
                        u64::try_from(escalation).unwrap_or(u64::MAX),
                    );
                    OracleRow {
                        provider: q.descriptor.provider.clone(),
                        model: q.descriptor.model.clone(),
                        expected: estimate.total_expected_micro,
                        success_ppm: verified_success_confidence_ppm(&st),
                        latency_ms: q.descriptor.economics.estimated_latency_ms,
                        base: q.call_cost_micro,
                    }
                }
                None => OracleRow {
                    provider: q.descriptor.provider.clone(),
                    model: q.descriptor.model.clone(),
                    expected: legacy_expected(q.success_ppm, q.call_cost_micro, escalation),
                    success_ppm: q.success_ppm,
                    latency_ms: q.descriptor.economics.estimated_latency_ms,
                    base: q.call_cost_micro,
                },
            }
        })
        .collect()
}

/// The documented tie-break ladder over oracle rows.
fn oracle_cmp(a: &OracleRow, b: &OracleRow) -> Ordering {
    a.expected
        .cmp(&b.expected)
        .then_with(|| b.success_ppm.cmp(&a.success_ppm))
        .then_with(|| a.latency_ms.cmp(&b.latency_ms))
        .then_with(|| a.base.cmp(&b.base))
        .then_with(|| (&a.provider, &a.model).cmp(&(&b.provider, &b.model)))
}

// ------------------------------------------------------------- properties

/// Unknown price is never treated as free; a zero-by-number snapshot with
/// Unknown authority is still not free; LocalZero is free BY AUTHORITY.
#[test]
fn unknown_price_is_never_free() {
    let mut g = Gen::new(0xE100);
    for _ in 0..4_000 {
        let quote = random_quote(&mut g);
        let (tin, tcr, tcw, tout) = (
            g.range(0, 50_000),
            g.range(0, 50_000),
            g.range(0, 50_000),
            g.range(0, 5_000),
        );
        let unknown = PricingSnapshot::unknown(1, "property".into());
        assert_eq!(unknown.settle_cost(tin, tcr, tcw, tout), None);
        assert_ne!(unknown.authority, PriceAuthority::LocalZero);

        let local = PricingSnapshot::local_zero(1, "property".into());
        assert_eq!(local.settle_cost(tin, tcr, tcw, tout), Some(0));
        assert!(local.is_local_zero());

        let exact = PricingSnapshot::exact(quote, 1, "property".into());
        let ceiling = PricingSnapshot::conservative_ceiling(quote, 1, "property".into());
        assert_eq!(
            ceiling.settle_cost(tin, tcr, tcw, tout),
            exact.settle_cost(tin, tcr, tcw, tout),
            "a ceiling snapshot prices exactly like the bound it states"
        );

        // A hostile zero quote under Unknown authority must stay unknown.
        let mut hostile = PricingSnapshot::unknown(1, "property".into());
        hostile.quote = Some(PriceQuote::ZERO);
        assert_eq!(
            hostile.settle_cost(1, 0, 0, 0),
            None,
            "a zero-by-number unknown-authority snapshot is NOT free"
        );
    }
}

/// A hard cap never admits a known estimate above the free balance; the
/// chosen decision's base cost never exceeds the remaining budget.
#[test]
fn hard_cap_never_admits_known_estimate_over_free() {
    for seed in 0..SEEDS {
        let mut g = Gen::new(0xCA9 ^ seed);
        let count = 2 + g.range(0, 4);
        let candidates: Vec<ModelDescriptor> =
            (0..count).map(|i| random_descriptor(&mut g, i)).collect();
        let mut req = random_request(&mut g);
        req.task_budget_remaining_micro = g.range(1, 80_000);
        let qualified = match qualified_candidates(&candidates, &req, &[], &LiveHealth::default()) {
            Ok(q) => q,
            Err(_) => continue,
        };
        for q in &qualified {
            assert!(
                q.call_cost_micro <= req.task_budget_remaining_micro,
                "seed {seed}: qualified candidate above the free budget"
            );
        }
        if let Ok(d) = Router::new(candidates.clone()).route(&req, &[]) {
            assert!(
                d.estimated_cost_micro <= req.task_budget_remaining_micro,
                "seed {seed}: plain route overshot the hard cap"
            );
        }
        if let Ok(d) = RouterService::new(candidates.clone()).route(&req, &[]) {
            assert!(
                d.estimated_cost_micro <= req.task_budget_remaining_micro,
                "seed {seed}: expected-cost route overshot the hard cap"
            );
        }
    }

    // Adversarial: a top-tier candidate known to cost above the free balance
    // is never admitted, while an affordable lower tier still serves.
    let expensive = descriptor("premium", "big", 99, 40, 120, 200, RateLimitState::Healthy);
    let cheap = descriptor("budget", "small", 70, 1, 3, 800, RateLimitState::Healthy);
    let cap = faktor_router::estimated_call_cost(&cheap.economics, 4000, 500, 0, 0) + 1;
    assert!(
        faktor_router::estimated_call_cost(&expensive.economics, 4000, 500, 0, 0) > cap,
        "the test needs the expensive candidate above the cap"
    );
    let req = RouteRequest {
        phase: RouterPhase::Implement,
        required_capabilities: vec!["tools".into()],
        context_tokens: 4000,
        estimated_output_tokens: 500,
        quality_floor: 60,
        task_budget_remaining_micro: cap,
        latency_preference_ms: None,
        ..Default::default()
    };
    let plain = Router::new(vec![expensive.clone(), cheap.clone()]);
    assert_eq!(plain.route(&req, &[]).unwrap().provider, "budget");
    let service = RouterService::new(vec![expensive.clone(), cheap.clone()]);
    assert_eq!(service.route(&req, &[]).unwrap().provider, "budget");
    let policy = EconomicRoutingPolicy::new(
        Arc::new(RouterService::new(vec![expensive.clone(), cheap.clone()])),
        RoutingMode::MaximumQuality,
    );
    let d = policy.route(&req).unwrap();
    assert_eq!(
        d.provider, "budget",
        "MaximumQuality never admits a known estimate above the free balance"
    );
    assert_eq!(d.model, "small");
}

/// Money and latency are distinct types, and the scored ladder can never
/// compare one against the other.
#[test]
fn price_and_latency_are_structurally_distinct() {
    assert_ne!(
        type_name::<MicroUsdPerToken>(),
        type_name::<u64>(),
        "money must be a distinct type from a bare counter"
    );
    assert_ne!(
        type_name::<MicroUsdPerMillionTokens>(),
        type_name::<u64>(),
        "per-million money must be a distinct type from a bare counter"
    );

    let da = descriptor("a", "a", 80, 0, 0, 500, RateLimitState::Healthy);
    let db = descriptor("b", "b", 80, 0, 0, 1, RateLimitState::Healthy);
    fn scored<'a>(
        candidate: &'a ModelDescriptor,
        expected: u64,
        latency: u64,
        success: u32,
        base: u64,
    ) -> ScoredCandidate<'a> {
        ScoredCandidate {
            candidate,
            expected_cost_micro: expected,
            expected_latency_ms: latency,
            success_ppm: success,
            call_cost_micro: base,
            work_estimate: None,
        }
    }
    // Expected cost is the primary key: a 1ms latency cannot outrank it.
    let low_expected_slow = scored(&da, 100, 5_000, 900_000, 100);
    let high_expected_fast = scored(&db, 101, 1, 900_000, 100);
    assert_eq!(
        low_expected_slow.compare(&high_expected_fast),
        Ordering::Less
    );
    // Success prior next.
    let high_success = scored(&da, 100, 5_000, 900_000, 100);
    let low_success = scored(&db, 100, 1, 800_000, 100);
    assert_eq!(high_success.compare(&low_success), Ordering::Less);
    // Latency next (same expected cost + success).
    let quick = scored(&da, 100, 10, 900_000, 1_000);
    let slow = scored(&db, 100, 20, 900_000, 10);
    assert_eq!(quick.compare(&slow), Ordering::Less);
    // Base cost next, then the deterministic name key.
    let cheap_base = scored(&db, 100, 10, 900_000, 10);
    let rich_base = scored(&da, 100, 10, 900_000, 20);
    assert_eq!(cheap_base.compare(&rich_base), Ordering::Less);

    // Behavioral: the plain router picks by cost first.
    let cheap_slow = descriptor("cheap", "slow", 80, 1, 2, 4_000, RateLimitState::Healthy);
    let rich_fast = descriptor("rich", "fast", 80, 50, 150, 1, RateLimitState::Healthy);
    let r = Router::new(vec![rich_fast, cheap_slow]);
    let req = RouteRequest {
        quality_floor: 60,
        ..Default::default()
    };
    let d = r.route(&req, &[]).unwrap();
    assert_eq!(d.provider, "cheap", "latency can never outrank price");

    // Equal cost: latency decides (still no cross-unit comparison).
    let fast = descriptor("fast", "m", 80, 5, 15, 200, RateLimitState::Healthy);
    let slow = descriptor("slow", "m", 80, 5, 15, 900, RateLimitState::Healthy);
    let r2 = Router::new(vec![slow, fast]);
    assert_eq!(r2.route(&req, &[]).unwrap().provider, "fast");
}

/// A hard quality floor is never lowered by any policy mode.
#[test]
fn hard_quality_floor_is_never_lowered() {
    for seed in 0..SEEDS {
        let mut g = Gen::new(0xF100 ^ seed);
        let count = 2 + g.range(0, 4);
        let candidates: Vec<ModelDescriptor> =
            (0..count).map(|i| random_descriptor(&mut g, i)).collect();
        let req = random_request(&mut g);
        let floor = req.quality_floor.min(100);
        let check = |d: &faktor_core::model::RouteDecision| {
            let chosen = candidates
                .iter()
                .find(|c| c.provider == d.provider && c.model == d.model)
                .expect("decisions name a candidate");
            assert!(
                phase_quality(chosen, req.phase) >= floor,
                "seed {seed}: decision quality below the hard floor {floor}"
            );
        };
        if let Ok(d) = Router::new(candidates.clone()).route(&req, &[]) {
            check(&d);
        }
        if let Ok(d) = RouterService::new(candidates.clone()).route(&req, &[]) {
            check(&d);
        }
        for mode in [
            RoutingMode::Economy,
            RoutingMode::Balanced,
            RoutingMode::MaximumQuality,
        ] {
            let service = Arc::new(RouterService::new(candidates.clone()));
            let policy = EconomicRoutingPolicy::new(service, mode.clone());
            if let Ok(d) = policy.route(&req) {
                check(&d);
                if mode == RoutingMode::Balanced {
                    let balanced_floor = floor.max(EconomicRoutingPolicy::BALANCED_QUALITY_FLOOR);
                    let chosen = candidates
                        .iter()
                        .find(|c| c.provider == d.provider && c.model == d.model)
                        .expect("decisions name a candidate");
                    assert!(
                        phase_quality(chosen, req.phase) >= balanced_floor,
                        "seed {seed}: Balanced routed below its band"
                    );
                }
            }
        }
    }

    // Adversarial: a floor above every candidate is a typed refusal, never a
    // decision at a lowered floor.
    let only = descriptor("weak", "m", 50, 1, 3, 100, RateLimitState::Healthy);
    let req = RouteRequest {
        phase: RouterPhase::Implement,
        required_capabilities: vec!["tools".into()],
        context_tokens: 1000,
        estimated_output_tokens: 100,
        quality_floor: 60,
        task_budget_remaining_micro: 0,
        latency_preference_ms: None,
        ..Default::default()
    };
    assert!(Router::new(vec![only.clone()]).route(&req, &[]).is_err());
    assert!(RouterService::new(vec![only.clone()])
        .route(&req, &[])
        .is_err());
    for mode in [
        RoutingMode::Economy,
        RoutingMode::Balanced,
        RoutingMode::MaximumQuality,
    ] {
        let policy = EconomicRoutingPolicy::new(
            Arc::new(RouterService::new(vec![only.clone()])),
            mode.clone(),
        );
        assert_eq!(
            policy.route(&req),
            Err(RouteFailure::NoCapableModel),
            "mode {mode:?} lowered the hard floor"
        );
    }
    // A candidate above the floor serves.
    let strong = descriptor("strong", "m", 90, 5, 15, 100, RateLimitState::Healthy);
    let service = Arc::new(RouterService::new(vec![only, strong]));
    let policy = EconomicRoutingPolicy::new(service, RoutingMode::MaximumQuality);
    let d = policy.route(&req).unwrap();
    assert_eq!(d.provider, "strong");
}

/// Economy minimizes the expected verified-work cost over generated pairs
/// with verified-outcome histories (the reference oracle re-implements the
/// documented work-cost math with the public estimators).
#[test]
fn economy_minimizes_expected_verified_work_cost() {
    for seed in 0..SEEDS {
        let mut g = Gen::new(0xEC0 ^ seed);
        let count = 2 + g.range(0, 3);
        let candidates: Vec<ModelDescriptor> =
            (0..count).map(|i| random_descriptor(&mut g, i)).collect();
        let mut req = random_request(&mut g);
        req.task_budget_remaining_micro = 0;
        let qualified = match qualified_candidates(&candidates, &req, &[], &LiveHealth::default()) {
            Ok(q) => q,
            Err(_) => continue,
        };
        let store = Arc::new(MemoryOutcomeStore::new());
        for (i, q) in qualified.iter().enumerate() {
            if i % 4 == 3 {
                continue;
            }
            let successes = g.range(0, 50);
            let failures = g.range(0, 50);
            for _ in 0..successes {
                store.append_sample(
                    &key(&q.descriptor.provider, &q.descriptor.model, req.phase),
                    OutcomeSample {
                        verified_success: true,
                        rework_cost_micro: 0,
                        rework_turns: 0,
                    },
                );
            }
            for _ in 0..failures {
                store.append_sample(
                    &key(&q.descriptor.provider, &q.descriptor.model, req.phase),
                    OutcomeSample {
                        verified_success: false,
                        rework_cost_micro: g.range(0, 50_000),
                        rework_turns: g.range(1, 4),
                    },
                );
            }
        }
        let service = RouterService::with_outcomes(candidates.clone(), store.clone());
        let decision = service.route(&req, &[]).unwrap();
        let rows = oracle_rows(&qualified, req.phase, store.as_ref());
        let best = rows
            .iter()
            .min_by(|a, b| oracle_cmp(a, b))
            .expect("qualified is non-empty");
        assert_eq!(
            (&decision.provider, &decision.model),
            (&best.provider, &best.model),
            "seed {seed}: Economy did not minimize the expected verified-work cost"
        );

        // The agent's Economy mode rides the same engine.
        let policy = EconomicRoutingPolicy::new(
            Arc::new(RouterService::with_outcomes(
                candidates.clone(),
                store.clone(),
            )),
            RoutingMode::Economy,
        );
        let d = policy.route(&req).unwrap();
        assert_eq!((&d.provider, &d.model), (&best.provider, &best.model));

        // Sanity: the public estimators obey their documented invariants.
        for q in &qualified {
            if let Some(st) =
                store.phase_stats(&q.descriptor.provider, &q.descriptor.model, req.phase)
            {
                assert!(rework_probability_ppm(&st) <= 1_000_000);
                let est = work_cost_estimate(q.call_cost_micro, Some(&st), u64::MAX / 4);
                assert_eq!(
                    est.total_expected_micro,
                    est.immediate_cost_micro
                        .saturating_add(est.expected_rework_micro)
                );
            }
        }
    }
}

/// MaximumQuality keeps an unknown-price high-quality model when there is no
/// cap; the same model is excluded (not treated as free) under a hard cap
/// that its estimate cannot fit, and an affordable tier still serves.
#[test]
fn maximum_quality_keeps_unpriced_high_quality_model_without_cap() {
    let unpriced = descriptor("unpriced", "hi", 99, 10, 30, 900, RateLimitState::Healthy);
    let priced = descriptor("priced", "lo", 80, 1, 3, 900, RateLimitState::Healthy);
    let mut pricing = std::collections::HashMap::new();
    pricing.insert(
        ("priced".to_string(), "lo".to_string()),
        PricingSnapshot::exact(
            PriceQuote {
                input: MicroUsdPerMillionTokens(1_000_000),
                output: MicroUsdPerMillionTokens(3_000_000),
                cache_read: MicroUsdPerMillionTokens(250_000),
                cache_write: MicroUsdPerMillionTokens(500_000),
            },
            1,
            "property".into(),
        ),
    );
    let service = Arc::new(RouterService::with_pricing(
        vec![unpriced.clone(), priced.clone()],
        pricing,
    ));
    let policy = EconomicRoutingPolicy::new(service, RoutingMode::MaximumQuality);
    let req = RouteRequest {
        phase: RouterPhase::Implement,
        required_capabilities: vec!["tools".into()],
        context_tokens: 4_000,
        estimated_output_tokens: 500,
        quality_floor: 60,
        task_budget_remaining_micro: 0,
        latency_preference_ms: None,
        ..Default::default()
    };
    let d = policy.route(&req).unwrap();
    assert_eq!(
        d.provider, "unpriced",
        "MaximumQuality discarded a high-quality unpriced model with no cap"
    );
    assert!(
        d.pricing_snapshot.is_none(),
        "an unpriced candidate must never carry a fabricated snapshot"
    );

    // A hard cap that the unpriced candidate's known estimate cannot fit:
    // the affordable tier serves, the unpriced one is NOT treated as free.
    let unpriced_cost = faktor_router::estimated_call_cost(&unpriced.economics, 4_000, 500, 0, 0);
    let priced_cost = faktor_router::estimated_call_cost(&priced.economics, 4_000, 500, 0, 0);
    assert!(priced_cost < unpriced_cost);
    let mut capped = req.clone();
    capped.task_budget_remaining_micro = priced_cost + 1;
    let d2 = policy.route(&capped).unwrap();
    assert_eq!(d2.provider, "priced");
    assert!(
        d2.pricing_snapshot.is_some(),
        "the priced winner carries its snapshot"
    );

    // Unknown never treated as free: an actually unknown snapshot settles
    // nothing, so the cap axis can never see it as a zero-cost candidate.
    let unpriced_snapshot = PricingSnapshot::unknown(1, "property".into());
    assert_eq!(
        unpriced_snapshot.settle_cost(10, 0, 0, 10),
        None,
        "unknown pricing can never produce a free settlement"
    );
    let zeroed = RouterService::new(vec![unpriced]);
    let d3 = zeroed.route(&req, &[]).unwrap();
    assert!(
        d3.pricing_snapshot.is_none(),
        "descriptor-only routing carries no authority (no fabricated snapshot)"
    );
}

/// The cache-hit term never understates and never fabricates a discount for
/// an unknown-priced candidate; a cache hit can only lower a KNOWN cost.
#[test]
fn cache_hits_only_lower_known_estimates() {
    let mut g = Gen::new(0xCAC4E);
    for _ in 0..2_000 {
        let mut d = random_descriptor(&mut g, 0);
        d.economics.rate_limit_state = RateLimitState::Healthy;
        let req = RouteRequest {
            context_tokens: g.range(1, 100_000),
            estimated_output_tokens: g.range(1, 2_000),
            quality_floor: 0,
            ..Default::default()
        };
        let cached = g.range(0, req.context_tokens + 1);
        let hit = CacheState {
            provider: d.provider.clone(),
            model: d.model.clone(),
            cached_input_tokens: cached,
            will_write_tokens: 0,
        };
        let plain = Router::new(vec![d.clone()]);
        let without = plain.route(&req, &[]).unwrap().estimated_cost_micro;
        let with = plain.route(&req, &[hit]).unwrap().estimated_cost_micro;
        assert!(
            with <= without,
            "a cache hit must never raise a known estimate"
        );
        if d.economics.is_local_zero_cost() {
            assert_eq!(with, 0, "local zero stays zero");
        }
    }
    // A cache state for a DIFFERENT model cannot discount this decision.
    let a = descriptor("a", "one", 80, 10, 30, 100, RateLimitState::Healthy);
    let other = CacheState {
        provider: "b".into(),
        model: "two".into(),
        cached_input_tokens: 1_000_000,
        will_write_tokens: 0,
    };
    let req = RouteRequest {
        context_tokens: 10_000,
        ..Default::default()
    };
    let r = Router::new(vec![a]);
    assert_eq!(
        r.route(&req, &[other]).unwrap().estimated_cost_micro,
        r.route(&req, &[]).unwrap().estimated_cost_micro
    );
}

/// A zero-by-number snapshot with Unknown authority is not free, and a
/// LocalZero authority is; `VerifiedOutcomeStats` absorb never lets a
/// success contribute rework.
#[test]
fn zero_authority_and_rework_absorb_invariants() {
    let mut zero_unknown = PricingSnapshot::unknown(1, "property".into());
    zero_unknown.quote = Some(PriceQuote::ZERO);
    assert_eq!(zero_unknown.settle_cost(1_000, 0, 0, 500), None);
    let local = PricingSnapshot::local_zero(1, "property".into());
    assert_eq!(local.settle_cost(1_000, 0, 0, 500), Some(0));

    let mut stats = VerifiedOutcomeStats::default();
    stats.absorb(OutcomeSample {
        verified_success: true,
        rework_cost_micro: 999_999,
        rework_turns: 9,
    });
    assert_eq!(stats.sample_count, 1);
    assert_eq!(stats.successes_first_pass, 1);
    assert_eq!(
        stats.rework_cost_micro_sum, 0,
        "a verified first-pass success cannot cause rework"
    );
    stats.absorb(OutcomeSample {
        verified_success: false,
        rework_cost_micro: 7,
        rework_turns: 1,
    });
    assert_eq!(stats.sample_count, 2);
    assert_eq!(stats.rework_cost_micro_sum, 7);
}

/// Keep the helper honest: the cached-cost recomputation uses the router's
/// own public integer estimate.
#[test]
fn cached_estimate_matches_the_public_cost_function() {
    for seed in 0..200 {
        let mut g = Gen::new(0xC057 ^ seed);
        let mut d = random_descriptor(&mut g, 0);
        d.economics.rate_limit_state = RateLimitState::Healthy;
        let req = RouteRequest {
            context_tokens: g.range(1, 100_000),
            estimated_output_tokens: g.range(1, 2_000),
            quality_floor: 0,
            ..Default::default()
        };
        let cached = g.range(0, req.context_tokens + 1);
        let hit = CacheState {
            provider: d.provider.clone(),
            model: d.model.clone(),
            cached_input_tokens: cached,
            will_write_tokens: 0,
        };
        let expected = faktor_router::estimated_call_cost(
            &d.economics,
            req.context_tokens,
            req.estimated_output_tokens,
            cached,
            0,
        );
        let decision = Router::new(vec![d]).route(&req, &[hit]).unwrap();
        assert_eq!(decision.estimated_cost_micro, expected, "seed {seed}");
    }
}

/// The empty outcome registry keeps scoring byte-identical to the legacy
/// math (so Economy's verified-cost minimization degrades honestly to the
/// telemetry prior when no history exists).
#[test]
fn empty_registry_matches_legacy_prior_scoring() {
    for seed in 0..SEEDS {
        let mut g = Gen::new(0xE49 ^ seed);
        let count = 2 + g.range(0, 3);
        let candidates: Vec<ModelDescriptor> =
            (0..count).map(|i| random_descriptor(&mut g, i)).collect();
        let mut req = random_request(&mut g);
        req.task_budget_remaining_micro = 0;
        let qualified = match qualified_candidates(&candidates, &req, &[], &LiveHealth::default()) {
            Ok(q) => q,
            Err(_) => continue,
        };
        let empty = MemoryOutcomeStore::new();
        let rows = oracle_rows(&qualified, req.phase, &empty);
        let best = rows
            .iter()
            .min_by(|a, b| oracle_cmp(a, b))
            .expect("non-empty");
        let service = RouterService::new(candidates.clone());
        let d = service.route(&req, &[]).unwrap();
        assert_eq!(
            (&d.provider, &d.model),
            (&best.provider, &best.model),
            "seed {seed}: empty-registry scoring diverged from the legacy prior math"
        );
    }
}
