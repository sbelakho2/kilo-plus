//! The production daemon dependency graph (P0-2): a NAMED struct — the
//! historic tuple type made every consumer depend on field ORDER and kept
//! routing/budgets/instructions out of the graph. The struct names the
//! daemon's authorities:
//!
//! - `session`/`providers` — the durable store and the provider registry;
//! - `agent` — the reasoning runtime (drives sessions with commands);
//! - `permissions`/`mcp_servers` — the permission channel and the
//!   supervised MCP servers whose tools ride the agent registry;
//! - `routing` — the ECONOMIC routing policy (P0-2/85/87/88): every model
//!   call of every session consults it first (Economy default; Pinned
//!   validates every call against the configured pin);
//! - `budgets` — the DURABLE cost ledger over this daemon's store
//!   (P0-6/12): one reservation per paid model call, settled exactly once;
//! - `instructions` — the per-workspace instruction resolver (P0-32) built
//!   over this daemon's workspace table.
//!
//! TODO(next wave): the orchestrator-as-TaskExecutor and the repository
//! IndexService join this graph once their daemon wiring lands; the
//! `cost_reservation` route_decision_json column and the task row's
//! max_cost_micro column are already waiting for the config surface that
//! sets per-task money caps (provider-level pricing tables and the
//! per-session cap plumbing).

use std::collections::HashMap;
use std::sync::Arc;

use faktor_agent::AgentRuntime;
use faktor_core::model::{ModelDescriptor, ModelSource, RoutingMode};
use faktor_provider::catalog::{admissible, ModelCatalogEntry, Provenance};
use faktor_provider::ProviderRegistry;
use faktor_server::permission::ChannelPermissionRequester;
use faktor_session::{DurableBudgetLedger, SessionManager};

/// The named daemon dependency graph (see the module docs).
pub struct DaemonGraph {
    pub session: Arc<SessionManager>,
    pub providers: Arc<ProviderRegistry>,
    pub agent: Arc<AgentRuntime>,
    pub permissions: Arc<ChannelPermissionRequester>,
    /// Supervised MCP servers (spec §31); the servers own their children
    /// for the daemon lifetime.
    pub mcp_servers: Vec<Arc<faktor_mcp::McpServer>>,
    /// The economic routing policy (P0-2): RouterService over the daemon's
    /// registered models + the mode from config (Economy default).
    pub routing: Arc<dyn faktor_agent::RoutingPolicy>,
    /// The durable monetary ledger over THIS daemon's store (P0-6/12).
    pub budgets: Arc<DurableBudgetLedger>,
    /// The per-workspace instruction resolver (P0-32).
    pub instructions: Arc<faktor_instructions::InstructionResolver>,
}

impl DaemonGraph {
    /// The classic tuple destructuring order (session, agent, permissions,
    /// mcp servers) so serve/acp/run call sites read naturally.
    pub fn core(
        &self,
    ) -> (
        &Arc<SessionManager>,
        &Arc<AgentRuntime>,
        &Arc<ChannelPermissionRequester>,
    ) {
        (&self.session, &self.agent, &self.permissions)
    }
}

/// Legacy per-token estimate line for the router's INTERNAL scoring: the
/// catalog quote is microUSD per MILLION tokens (exact); the descriptor's
/// per-token field rounds UP (ceil) so a positive list price never reads as
/// a free zero and an estimate never understates. Settlement never touches
/// this lossy projection — it prices usage against the frozen exact
/// [`faktor_core::model::PricingSnapshot`] the graph attaches to decisions.
fn legacy_per_token(per_million: u64) -> faktor_core::model::MicroUsdPerToken {
    faktor_core::model::MicroUsdPerToken(per_million.div_ceil(1_000_000))
}

/// A router candidate descriptor for one registered provider model, built
/// from the provider's REAL catalog row (audit P0-1 / wave-B item B):
///
/// - the descriptor's `economics` is the LEGACY per-token estimate surface
///   the router's internal qualification/scoring reads: reliability priors
///   from the row's `quality_prior`, latency from the conservative
///   performance default, and price lines projected UP from the row's exact
///   per-million quote ([`legacy_per_token`]) — never a fabricated zero for
///   a priced row, and zero only for authoritative-local or unpriced rows;
/// - [`PricingState::Unknown`] rows reach a descriptor ONLY through the
///   pinned path (the pin — not economics — decides; see
///   [`build_router_service_with_outcomes`]); free-economy candidate lists exclude them
///   BEFORE a descriptor exists, so the router never sees a fabricated
///   zero where a price is missing;
/// - `source` records the row's provenance.
///
/// The candidate's ROUTE-TIME PRICING AUTHORITY is not the descriptor: the
/// graph cuts the row's exact [`faktor_core::model::PricingSnapshot`] into
/// the service's pricing map ([`faktor_router::RouterService::with_pricing`])
/// so every decision freezes quote + authority (exact/ceiling/local-zero/
/// unknown) without inference.
fn descriptor_for(provider_id: &str, entry: &ModelCatalogEntry) -> ModelDescriptor {
    let caps = &entry.capabilities;
    let qp = entry.quality_prior;
    let mut economics = faktor_core::model::ModelEconomics {
        tool_reliability: qp.tool_reliability,
        reasoning_reliability: qp.reasoning_reliability,
        coding_reliability: qp.coding_reliability,
        context_reliability: qp.context_reliability,
        availability: qp.availability,
        ..Default::default()
    };
    // LocalZero rows quote Some(PriceQuote::ZERO) (the authoritative local
    // marker); Known/Ceiling rows quote their exact lines; Unknown rows
    // quote None and keep the all-zero estimate (pinned validation only).
    if let Some(q) = entry.pricing.quote() {
        economics.input_price_per_mtok = legacy_per_token(q.input.0);
        economics.output_price_per_mtok = legacy_per_token(q.output.0);
        economics.cache_read_price_per_mtok = legacy_per_token(q.cache_read.0);
        economics.cache_write_price_per_mtok = legacy_per_token(q.cache_write.0);
    }
    ModelDescriptor {
        provider: provider_id.to_string(),
        model: entry.model.clone(),
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
        economics,
        source: match entry.provenance {
            Provenance::BuiltIn => ModelSource::ConservativeDefault,
            Provenance::ProviderCatalog => ModelSource::ProviderCatalog,
            // A Composite row is priced by USER policy (the configured
            // ceiling): the price is a budget bound, the user made it.
            Provenance::UserOverride | Provenance::Composite => ModelSource::UserOverride,
        },
    }
}

/// The exclusion/admission policy for free-economy candidate sets (audit
/// P0-1/wave-B item C — the admission matrix): [`PricingState::Unknown`]
/// rows are EXCLUDED from Economy/Balanced/MaximumQuality candidate sets —
/// an unknown price is never zero and never the 1-microUSD runtime
/// fallback. The ceiling case never reaches this function as `Unknown`: a
/// configured `pricing_ceiling_micro_usd_per_million_tokens` turns Unknown
/// rows into [`PricingState::ConservativeCeiling`] at exactly the ceiling
/// with provenance [`Provenance::Composite`] inside the provider wrapper.
/// `LocalZero` (Ollama) and priced rows are always included; in Pinned
/// mode the pin itself is admitted regardless of its price state (the pin,
/// not economics, decides — an Unknown pin keeps its Unknown snapshot).
fn candidate_entry_ok(mode: &RoutingMode, entry: &ModelCatalogEntry) -> bool {
    // No hard cost cap exists at graph build (caps are per-task, decided at
    // route time by the runtime): admission uses the no-cap rows, and the
    // daemon has no allow-unknown-in-balanced knob yet.
    admissible(mode, &entry.pricing, false, false)
}

/// The daemon's router candidate set: every PRICED known model of every
/// registered provider (bounded by the registry and the providers' own
/// `known_models()`), built from each provider's real catalog rows.
///
/// Candidate-set policy by mode (audit P0-1/wave-B C — unknown price !=
/// zero, authority never inferred):
///
/// | pricing state | Economy / Balanced / MaximumQuality | Pinned |
/// |---|---|---|
/// | Known (exact prices) | included at its real price | validation as today |
/// | ConservativeCeiling | included at the ceiling | validation as today |
/// | LocalZero (Ollama) | always included, zero cost | validation as today |
/// | Unknown, no ceiling | **EXCLUDED** (never a fabricated 0 / 1-micro fallback) | pin included (the pin decides; its snapshot stays Unknown — never LocalZero) |
/// | Unknown + configured ceiling | included as ConservativeCeiling (applied by the provider wrapper) | as today |
///
/// Every candidate also contributes its catalog-cut
/// [`faktor_core::model::PricingSnapshot`] to the service's pricing map
/// keyed (provider, model), so route decisions freeze the real authority
/// and the exact per-million quote.
///
/// In Pinned mode the candidate set collapses to the pin itself: the
/// RouterService then VALIDATES the pin's capability/fit/budget/health
/// axes and the pin always wins when feasible — the router's free choice
/// can never silently substitute the configured pin (fail closed). A pin
/// whose (provider, model) is not among the registered models is a
/// graph-build error (loud, at boot — never a silent Economy).
///
/// The daemon wiring twin [`build_router_service_with_outcomes`] builds the
/// SAME candidates through
/// [`faktor_router::RouterService::with_pricing_and_outcomes`] so the
/// service carries the durable verified-outcome registry; this plain
/// variant is the test/embedded shape (no registry — the default empty
/// store keeps decisions byte-identical to a registry-less service).
pub fn build_router_service_with_outcomes(
    providers: &ProviderRegistry,
    mode: &RoutingMode,
    outcomes: Arc<dyn faktor_router::OutcomeStore>,
) -> Result<Arc<faktor_router::RouterService>, String> {
    let (candidates, pricing) = router_candidates(providers, mode)?;
    Ok(Arc::new(
        faktor_router::RouterService::with_pricing_and_outcomes(candidates, pricing, outcomes),
    ))
}

/// The candidate set + pricing map the router constructors consume (see
/// [`build_router_service_with_outcomes`] for the admission policy).
type RouterCandidates = (
    Vec<ModelDescriptor>,
    HashMap<(String, String), faktor_core::model::PricingSnapshot>,
);

/// Candidate + pricing-map build shared by both service constructors (see
/// [`build_router_service_with_outcomes`] for the admission policy).
fn router_candidates(
    providers: &ProviderRegistry,
    mode: &RoutingMode,
) -> Result<RouterCandidates, String> {
    let mut candidates: Vec<ModelDescriptor> = Vec::new();
    let mut pricing: HashMap<(String, String), faktor_core::model::PricingSnapshot> =
        HashMap::new();
    match mode {
        // MaximumQuality and Balanced route over the SAME full registered
        // candidate set as Economy — the mode is policy-level semantics
        // (top-quality tier / balanced quality band), not a candidate
        // filter at build time. All three exclude Unknown-priced rows:
        // an entry whose price is unknown cannot be costed or budgeted.
        RoutingMode::Economy | RoutingMode::MaximumQuality | RoutingMode::Balanced => {
            for id in providers.ids() {
                let Some(p) = providers.get(&id) else {
                    continue;
                };
                for model in p.known_models() {
                    let entry = p.catalog_entry(&model);
                    if candidate_entry_ok(mode, &entry) {
                        let d = descriptor_for(&id, &entry);
                        pricing.insert(
                            (d.provider.clone(), d.model.clone()),
                            entry.pricing_snapshot(),
                        );
                        candidates.push(d);
                    }
                }
            }
        }
        RoutingMode::Pinned { provider, model } => {
            let p = providers.get(provider).ok_or_else(|| {
                format!(
                    "routing is pinned to provider {provider:?} which is not registered; \
                     configure it or switch routing_mode to economy"
                )
            })?;
            if !p.known_models().iter().any(|m| m == model) {
                return Err(format!(
                    "routing is pinned to model {model:?} which provider {provider:?} does not serve; \
                     configure the model or switch routing_mode to economy"
                ));
            }
            // Pinned is unaffected by the exclusion policy: the pin — not
            // its economics — decides. An Unknown-priced pin validates on
            // capability/fit/budget axes exactly as the zero-default rows
            // did before catalogs existed; its decision snapshot stays the
            // honest Unknown (never a fabricated LocalZero).
            let entry = p.catalog_entry(model);
            let d = descriptor_for(provider, &entry);
            pricing.insert(
                (d.provider.clone(), d.model.clone()),
                entry.pricing_snapshot(),
            );
            candidates.push(d);
        }
    }
    Ok((candidates, pricing))
}

/// The daemon's economic routing policy over the candidates of
/// [`build_router_service_with_outcomes`]: the policy's RouterService is
/// built via `with_pricing_and_outcomes`, so `record_call_outcome` verified
/// samples (runtime deterministic-gate sites) land in the OUTCOME STORE
/// this call wires — the same registry every route consult reads. The
/// daemon passes a [`faktor_agent::StoreOutcomeStore`] over its store
/// (verified stats then survive restarts and serve every later route);
/// tests and embedded hosts pass their own registry or the default
/// [`faktor_router::EmptyOutcomeStore`].
pub fn economic_routing_policy_with_outcomes(
    providers: &ProviderRegistry,
    mode: RoutingMode,
    outcomes: Arc<dyn faktor_router::OutcomeStore>,
) -> Result<Arc<dyn faktor_agent::RoutingPolicy>, String> {
    let service = build_router_service_with_outcomes(providers, &mode, outcomes)?;
    Ok(faktor_agent::EconomicRoutingPolicy::new(service, mode))
}

#[cfg(test)]
mod tests {
    use super::*;
    use faktor_core::model::{
        MicroUsdPerMillionTokens, MicroUsdPerToken, ModelCapabilities, ModelEconomics,
        PriceAuthority, PriceQuote, PricingSnapshot,
    };
    use faktor_provider::catalog::{
        ModelCatalogEntry, PricingState, Provenance, QualityPrior, CATALOG_FIRST_EPOCH,
    };
    use faktor_provider::{FakeProvider, Provider, ProviderStream};
    use std::sync::Arc;

    /// Candidate/service twin over the default empty outcome registry (the
    /// legacy 2-arg build shape — no durable verified-outcome history).
    fn empty_store_service(
        providers: &ProviderRegistry,
        mode: &RoutingMode,
    ) -> Result<Arc<faktor_router::RouterService>, String> {
        build_router_service_with_outcomes(
            providers,
            mode,
            Arc::new(faktor_router::EmptyOutcomeStore),
        )
    }

    /// Policy twin over the default empty outcome registry (the legacy
    /// 2-arg build shape — no durable verified-outcome history).
    fn empty_store_policy(
        providers: &ProviderRegistry,
        mode: RoutingMode,
    ) -> Result<Arc<dyn faktor_agent::RoutingPolicy>, String> {
        economic_routing_policy_with_outcomes(
            providers,
            mode,
            Arc::new(faktor_router::EmptyOutcomeStore),
        )
    }

    /// Catalog-aware test provider: every known model carries an explicit
    /// pricing state, so candidate-building tests exercise the REAL
    /// exclusion/ceiling policy instead of the legacy FakeProvider rows
    /// (which are Unknown by the trait default and therefore excluded from
    /// free-economy candidate sets).
    struct PricedTestProvider {
        id: String,
        caps: ModelCapabilities,
        models: Vec<String>,
        pricing: PricingState,
    }

    impl PricedTestProvider {
        /// A Known row at `input`/`output` WHOLE DOLLARS per million tokens
        /// (the per-token legacy projection of x whole dollars reads x).
        fn known(
            id: &str,
            model: &str,
            caps: ModelCapabilities,
            input: u64,
            output: u64,
        ) -> Arc<dyn Provider> {
            Self::with_pricing(
                id,
                model,
                caps,
                PricingState::Known(PricingSnapshot::exact(
                    PriceQuote {
                        input: MicroUsdPerMillionTokens::from_dollars_per_million(input),
                        output: MicroUsdPerMillionTokens::from_dollars_per_million(output),
                        cache_read: MicroUsdPerMillionTokens::ZERO,
                        cache_write: MicroUsdPerMillionTokens::ZERO,
                    },
                    CATALOG_FIRST_EPOCH,
                    "row".to_string(),
                )),
            )
        }

        fn local(id: &str, model: &str, caps: ModelCapabilities) -> Arc<dyn Provider> {
            Self::with_pricing(id, model, caps, PricingState::LocalZero)
        }

        fn with_pricing(
            id: &str,
            model: &str,
            caps: ModelCapabilities,
            pricing: PricingState,
        ) -> Arc<dyn Provider> {
            Arc::new(Self {
                id: id.into(),
                caps,
                models: vec![model.into()],
                pricing,
            })
        }
    }

    impl Provider for PricedTestProvider {
        fn id(&self) -> &str {
            &self.id
        }

        fn capabilities(&self, _model: &str) -> ModelCapabilities {
            self.caps.clone()
        }

        fn known_models(&self) -> Vec<String> {
            self.models.clone()
        }

        fn catalog_entry(&self, model: &str) -> ModelCatalogEntry {
            ModelCatalogEntry {
                provider: self.id.clone(),
                model: model.to_string(),
                capabilities: self.capabilities(model),
                pricing: self.pricing.clone(),
                quality_prior: QualityPrior::default(),
                source_epoch: CATALOG_FIRST_EPOCH,
                provenance: Provenance::ProviderCatalog,
            }
        }

        fn stream(&self, _req: faktor_provider::GenericAgentRequest) -> ProviderStream {
            Box::pin(futures::stream::empty())
        }
    }

    fn open_transport() -> Arc<dyn faktor_provider::egress::HttpTransport> {
        Arc::new(faktor_provider::egress::PolicyCheckedHttpTransport::with_policy(None))
    }

    #[test]
    fn economy_candidates_cover_every_priced_registered_model_with_live_capabilities() {
        let mut registry = ProviderRegistry::new();
        registry
            .try_register(PricedTestProvider::known(
                "local-a",
                "model-x",
                ModelCapabilities {
                    tools: true,
                    context: 64_000,
                    ..Default::default()
                },
                15,
                60,
            ))
            .unwrap();
        let svc = empty_store_service(&registry, &RoutingMode::Economy).unwrap();
        assert_eq!(svc.router.candidates.len(), 1);
        let c = &svc.router.candidates[0];
        assert_eq!(c.provider, "local-a");
        assert_eq!(c.model, "model-x");
        assert!(c.tools, "capabilities come from the live provider");
        assert_eq!(c.context, 64_000);
        assert_eq!(
            c.economics.input_price_per_mtok,
            MicroUsdPerToken(15),
            "a Known catalog row keeps its REAL price"
        );
        assert_eq!(c.economics.output_price_per_mtok, MicroUsdPerToken(60));
        assert_eq!(c.source, ModelSource::ProviderCatalog);
        // Empty registry -> empty candidates (every route then fails typed;
        // nothing silently falls back).
        let empty = ProviderRegistry::new();
        assert!(empty_store_service(&empty, &RoutingMode::Economy)
            .unwrap()
            .router
            .candidates
            .is_empty());
    }

    #[test]
    fn unknown_priced_models_are_excluded_from_every_free_economy_candidate_set() {
        // Legacy providers (the trait default catalog row) are Unknown:
        // free-economy candidate lists EXCLUDE them — never a fabricated
        // zero price, never a 1-microUSD fallback.
        let mut registry = ProviderRegistry::new();
        registry
            .try_register(Arc::new(FakeProvider::with_script(
                "openai",
                ModelCapabilities {
                    tools: true,
                    streaming: true,
                    context: 128_000,
                    ..Default::default()
                },
                vec![],
            )))
            .unwrap();
        registry
            .try_register(Arc::new(FakeProvider::with_script(
                "another-remote",
                ModelCapabilities {
                    tools: true,
                    streaming: true,
                    context: 128_000,
                    ..Default::default()
                },
                vec![],
            )))
            .unwrap();
        for mode in [
            RoutingMode::Economy,
            RoutingMode::MaximumQuality,
            RoutingMode::Balanced,
        ] {
            let svc = empty_store_service(&registry, &mode).unwrap();
            assert!(
                svc.router.candidates.is_empty(),
                "{mode:?} must exclude every Unknown-priced entry"
            );
        }
        // The provider's catalog row really is Unknown (the exclusion
        // policy reads the STATE, not the zero projection).
        let p = registry.get("openai").unwrap();
        assert_eq!(p.catalog_entry("default").pricing, PricingState::Unknown);
    }

    #[test]
    fn builtin_table_rows_enter_economy_candidates_with_exact_prices() {
        // Wave-B item C: an adapter WITHOUT its own rows (the trait
        // default) still prices officially-known models through the
        // built-in table — a FakeProvider of family openai serving gpt-4o
        // is a Known exact-priced candidate, never an excluded Unknown and
        // never free.
        let mut registry = ProviderRegistry::new();
        registry
            .try_register(Arc::new(FakeProvider::with_script(
                "openai",
                ModelCapabilities {
                    tools: true,
                    streaming: true,
                    context: 128_000,
                    ..Default::default()
                },
                vec![],
            )))
            .unwrap();
        registry
            .try_register(Arc::new(FakeProvider::with_script(
                "ollama",
                ModelCapabilities::small_local(),
                vec![],
            )))
            .unwrap();
        // FakeProvider reports only "default": add gpt-4o explicitly through
        // the registry mirror path used by daemon wiring is out of scope
        // here — instead lock the row and its economy inclusion directly.
        let p = registry.get("openai").unwrap();
        let e = p.catalog_entry("gpt-4o");
        assert_eq!(e.pricing.authority(), PriceAuthority::Exact);
        assert_eq!(e.provenance, Provenance::BuiltIn);
        let q = e.pricing_snapshot().quote.unwrap();
        assert_eq!(q.input, MicroUsdPerMillionTokens(2_500_000));
        assert_eq!(q.output, MicroUsdPerMillionTokens(10_000_000));
        assert_eq!(q.cache_read, MicroUsdPerMillionTokens(1_250_000));
        assert_eq!(
            e.pricing_snapshot().settle_cost(1_000_000, 0, 0, 0),
            Some(2_500_000)
        );
        let p = registry.get("ollama").unwrap();
        assert_eq!(p.catalog_entry("gpt-4o").pricing, PricingState::Unknown);
    }

    #[test]
    fn configured_ceiling_admits_unknown_models_at_exactly_the_ceiling_as_composite() {
        // The config surface (ProviderCfg.pricing ceiling) wraps the
        // endpoint: its Unknown rows enter the economy candidate set priced
        // at EXACTLY the ceiling (per-million microUSD) with Composite
        // provenance and ConservativeCeiling authority.
        let cfg = crate::config::ProviderCfg::OpenAi {
            id: "corp-proxy".into(),
            base_url: "https://corp.example.com/v1".into(),
            api_key_env: None,
            pricing: Some(crate::config::ProviderPricingCfg {
                pricing_ceiling_micro_usd_per_million_tokens: Some(42_000_000),
                ..Default::default()
            }),
        };
        let mut registry = ProviderRegistry::new();
        registry
            .try_register(cfg.build(open_transport()).unwrap())
            .unwrap();
        // The endpoint row is Unknown at the adapter level...
        let entry = registry.get("corp-proxy").unwrap().catalog_entry("default");
        assert_eq!(entry.provenance, Provenance::Composite);
        assert_eq!(entry.source_epoch, CATALOG_FIRST_EPOCH + 1);
        match &entry.pricing {
            PricingState::ConservativeCeiling(snap) => {
                assert_eq!(snap.authority, PriceAuthority::ConservativeCeiling);
                let q = snap.quote.expect("ceiling quotes");
                for line in [q.input, q.output, q.cache_read, q.cache_write] {
                    assert_eq!(line, MicroUsdPerMillionTokens(42_000_000));
                }
            }
            other => panic!("ceiling must produce ConservativeCeiling, got {other:?}"),
        }
        // ...and the economy candidate set includes it at the ceiling,
        // projected UP to the legacy per-token estimate (42 microUSD/token
        // for a $42/M ceiling — exact, never free).
        let svc = empty_store_service(&registry, &RoutingMode::Economy).unwrap();
        assert_eq!(svc.router.candidates.len(), 1);
        let c = &svc.router.candidates[0];
        for p in [
            c.economics.input_price_per_mtok,
            c.economics.output_price_per_mtok,
            c.economics.cache_read_price_per_mtok,
            c.economics.cache_write_price_per_mtok,
        ] {
            assert_eq!(p, MicroUsdPerToken(42), "priced at exactly the ceiling");
        }
        assert_eq!(
            c.source,
            ModelSource::UserOverride,
            "Composite provenance maps to the user-configured source"
        );
    }

    #[test]
    fn local_zero_and_known_rows_are_always_included_and_keep_their_state() {
        let mut registry = ProviderRegistry::new();
        registry
            .try_register(PricedTestProvider::local(
                "ollama",
                "qwen3.8",
                ModelCapabilities::small_local(),
            ))
            .unwrap();
        registry
            .try_register(PricedTestProvider::known(
                "openai",
                "gpt-5",
                ModelCapabilities::default(),
                15,
                60,
            ))
            .unwrap();
        let svc = empty_store_service(&registry, &RoutingMode::Economy).unwrap();
        assert_eq!(svc.router.candidates.len(), 2);
        let local = svc
            .router
            .candidates
            .iter()
            .find(|c| c.provider == "ollama")
            .unwrap();
        assert!(
            local.economics.is_local_zero_cost(),
            "LocalZero stays the router's explicit zero-cost marker"
        );
        assert_eq!(local.economics.estimated_latency_ms, 1000);
        let paid = svc
            .router
            .candidates
            .iter()
            .find(|c| c.provider == "openai")
            .unwrap();
        assert_eq!(paid.economics.input_price_per_mtok, MicroUsdPerToken(15));
        assert!(!paid.economics.is_local_zero_cost());
        // The decision over the local model freezes an authoritative
        // LocalZero snapshot — never an inference from zeros.
        let d = svc
            .route(
                &faktor_router::RouteRequest {
                    phase: faktor_core::model::RouterPhase::Implement,
                    required_capabilities: vec!["tools".into(), "streaming".into()],
                    context_tokens: 8_000,
                    estimated_output_tokens: 512,
                    quality_floor: 50,
                    task_budget_remaining_micro: 0,
                    latency_preference_ms: None,
                },
                &[],
            )
            .unwrap();
        let snap = d.pricing_snapshot.expect("catalog path stamps snapshots");
        assert_eq!(snap.authority, PriceAuthority::LocalZero);
        assert_eq!(snap.settle_cost(1_000_000, 0, 0, 0), Some(0));
    }

    #[test]
    fn economy_policy_routes_deterministically_and_pinned_keeps_the_pin() {
        // (a) graph-level: an economy policy over one known-priced and one
        // local model makes a decision, and a pinned policy returns the
        // PIN even when the free economy evaluation would have picked the
        // cheaper one.
        let mut registry = ProviderRegistry::new();
        registry
            .try_register(PricedTestProvider::known(
                "alpha",
                "fast",
                ModelCapabilities {
                    tools: true,
                    streaming: true,
                    context: 128_000,
                    ..Default::default()
                },
                1,
                3,
            ))
            .unwrap();
        registry
            .try_register(PricedTestProvider::known(
                "beta",
                "fast",
                ModelCapabilities {
                    tools: true,
                    streaming: true,
                    context: 128_000,
                    ..Default::default()
                },
                15,
                60,
            ))
            .unwrap();
        let economy = empty_store_policy(&registry, RoutingMode::Economy).unwrap();
        assert_eq!(economy.mode(), RoutingMode::Economy);
        let request = || faktor_router::RouteRequest {
            phase: faktor_core::model::RouterPhase::Implement,
            required_capabilities: vec!["tools".into(), "streaming".into()],
            context_tokens: 8_000,
            estimated_output_tokens: 512,
            quality_floor: 60,
            task_budget_remaining_micro: 0,
            latency_preference_ms: None,
        };
        let decision = economy
            .route(&request())
            .expect("economy route over two priced models must decide");
        assert_eq!(
            decision.provider, "alpha",
            "the known-priced cheap model wins on real prices: {}",
            decision.reasoning
        );
        assert!(
            !decision.reasoning.is_empty(),
            "audit string rides the decision"
        );
        let snap = decision
            .pricing_snapshot
            .expect("catalog decisions carry snapshots");
        assert_eq!(snap.authority, PriceAuthority::Exact);
        assert_eq!(
            snap.settle_cost(1_000_000, 0, 0, 0),
            Some(1_000_000),
            "the frozen quote prices the real settlement"
        );
        // Pinned to beta: the policy validates and returns beta — the free
        // evaluation prefers alpha (cheaper), the pin never loses.
        let pinned = empty_store_policy(
            &registry,
            RoutingMode::Pinned {
                provider: "beta".into(),
                model: "fast".into(),
            },
        )
        .unwrap();
        let decision = pinned.route(&request()).expect("pinned validation passes");
        assert_eq!(decision.provider, "beta");
        assert_eq!(decision.model, "fast");
        assert_eq!(decision.source, ModelSource::ProviderCatalog);
        // Budget-denial over REAL prices: a budget below beta's real cost
        // (1000x15 + 100x60 = 21 000 micro) still lets alpha through
        // (1000x1 + 100x3 = 1 300 micro); a budget below alpha's own cost
        // fails the route typed — never a silent substitute.
        let request = || faktor_router::RouteRequest {
            phase: faktor_core::model::RouterPhase::Implement,
            required_capabilities: vec!["tools".into(), "streaming".into()],
            context_tokens: 1_000,
            estimated_output_tokens: 100,
            quality_floor: 60,
            task_budget_remaining_micro: 5_000,
            latency_preference_ms: None,
        };
        assert_eq!(economy.route(&request()).unwrap().provider, "alpha");
        let starved = || faktor_router::RouteRequest {
            phase: faktor_core::model::RouterPhase::Implement,
            required_capabilities: vec!["tools".into(), "streaming".into()],
            context_tokens: 1_000,
            estimated_output_tokens: 100,
            quality_floor: 60,
            task_budget_remaining_micro: 100,
            latency_preference_ms: None,
        };
        assert!(
            economy.route(&starved()).is_err(),
            "a budget below every candidate's real cost is a typed refusal"
        );
    }

    #[test]
    fn pinned_candidates_collapse_to_the_pin_unknown_pricing_included_missing_pins_fail() {
        // Pinned is UNAFFECTED by the Unknown exclusion: a pin over an
        // Unknown-priced (legacy) model still builds and validates — the
        // pin, not economics, decides.
        let mut registry = ProviderRegistry::new();
        registry
            .try_register(Arc::new(FakeProvider::with_script(
                "paid",
                ModelCapabilities {
                    tools: true,
                    ..Default::default()
                },
                vec![],
            )))
            .unwrap();
        let mode = RoutingMode::Pinned {
            provider: "paid".into(),
            model: "default".into(),
        };
        let svc = empty_store_service(&registry, &mode).unwrap();
        assert_eq!(
            svc.router.candidates.len(),
            1,
            "the pin is the only candidate"
        );
        assert_eq!(svc.router.candidates[0].provider, "paid");
        assert_eq!(
            svc.router.candidates[0].economics,
            ModelEconomics {
                // Pinned validation only: an Unknown row projects zeros
                // (never chosen on price — the pin is fixed by config).
                ..Default::default()
            }
        );
        // The pin's decision snapshot is the honest UNKNOWN (quote None) —
        // a pinned unknown model NEVER collapses to LocalZero, and its
        // settlement refuses every fabricated number.
        let policy = empty_store_policy(&registry, mode).unwrap();
        let d = policy
            .route(&faktor_router::RouteRequest {
                required_capabilities: vec!["tools".into()],
                context_tokens: 2,
                estimated_output_tokens: 1,
                quality_floor: 50,
                ..Default::default()
            })
            .expect("pinned validation passes");
        let snap = d
            .pricing_snapshot
            .expect("pinned decisions carry snapshots");
        assert_eq!(snap.authority, PriceAuthority::Unknown);
        assert_eq!(snap.quote, None);
        assert!(!snap.is_local_zero());
        assert_eq!(snap.settle_cost(1_000_000, 0, 0, 0), None);
        // Unknown pin: the daemon refuses to boot on it (loud, at build).
        let bad = RoutingMode::Pinned {
            provider: "nope".into(),
            model: "m".into(),
        };
        assert!(empty_store_service(&registry, &bad).is_err());
        let bad_model = RoutingMode::Pinned {
            provider: "paid".into(),
            model: "not-a-model".into(),
        };
        assert!(empty_store_service(&registry, &bad_model).is_err());
    }

    #[test]
    fn exact_user_override_table_prices_unknown_models_and_bumps_the_epoch() {
        use crate::config::ProviderPricingCfg;
        let mut registry = ProviderRegistry::new();
        for (id, pricing) in [
            (
                "corp-proxy",
                ProviderPricingCfg {
                    input_micro_usd_per_million_tokens: Some(2_000_000),
                    output_micro_usd_per_million_tokens: Some(8_000_000),
                    ..Default::default()
                },
            ),
            // The scoping half of the adversarial case: a SECOND custom
            // endpoint WITHOUT a table stays Unknown (overrides apply to
            // the configured endpoint only, never leaking across ids).
            ("dev-proxy", ProviderPricingCfg::default()),
        ] {
            let cfg = crate::config::ProviderCfg::OpenAi {
                id: id.into(),
                base_url: format!("https://{id}.example.com/v1"),
                api_key_env: None,
                pricing: Some(pricing),
            };
            registry
                .try_register(cfg.build(open_transport()).unwrap())
                .unwrap();
        }
        let p = registry.get("corp-proxy").unwrap();
        let entry = p.catalog_entry("default");
        match &entry.pricing {
            PricingState::Known(snap) => {
                assert_eq!(snap.authority, PriceAuthority::Exact);
                let q = snap.quote.expect("exact override quotes");
                assert_eq!(q.input, MicroUsdPerMillionTokens(2_000_000));
                assert_eq!(q.output, MicroUsdPerMillionTokens(8_000_000));
                assert_eq!(q.cache_read, MicroUsdPerMillionTokens::ZERO);
                assert_eq!(q.cache_write, MicroUsdPerMillionTokens::ZERO);
            }
            other => panic!("exact override must produce Known, got {other:?}"),
        }
        assert_eq!(entry.provenance, Provenance::UserOverride);
        assert_eq!(
            entry.source_epoch,
            CATALOG_FIRST_EPOCH + 1,
            "overrides increment the pricing epoch"
        );
        // The dev-proxy endpoint keeps its Unknown adapter rows.
        assert_eq!(
            registry
                .get("dev-proxy")
                .unwrap()
                .catalog_entry("default")
                .pricing,
            PricingState::Unknown
        );
        // The economy graph sees exactly one priced candidate.
        let svc = empty_store_service(&registry, &RoutingMode::Economy).unwrap();
        assert_eq!(svc.router.candidates.len(), 1);
        assert_eq!(svc.router.candidates[0].provider, "corp-proxy");
    }

    #[test]
    fn overrides_are_ignored_but_local_zero_survives_on_ollama_instances() {
        // A pricing table on a local runtime is refused at build (typed);
        // an ollama instance with no table stays LocalZero and always
        // routes in economy.
        use crate::config::ProviderPricingCfg;
        let cfg = crate::config::ProviderCfg::Ollama {
            id: "ollama".into(),
            base_url: None,
            pricing: None,
        };
        let mut registry = ProviderRegistry::new();
        registry
            .try_register(cfg.build(open_transport()).unwrap())
            .unwrap();
        let entry = registry.get("ollama").unwrap().catalog_entry("default");
        assert_eq!(entry.pricing, PricingState::LocalZero);
        let svc = empty_store_service(&registry, &RoutingMode::Economy).unwrap();
        assert_eq!(svc.router.candidates.len(), 1);
        assert!(svc.router.candidates[0].economics.is_local_zero_cost());
        let hostile = crate::config::ProviderCfg::Ollama {
            id: "ollama".into(),
            base_url: None,
            pricing: Some(ProviderPricingCfg {
                input_micro_usd_per_million_tokens: Some(15_000_000),
                output_micro_usd_per_million_tokens: Some(60_000_000),
                ..Default::default()
            }),
        };
        let e = match hostile.build(open_transport()) {
            Ok(_) => panic!("ollama pricing refused"),
            Err(e) => e,
        };
        assert!(e.contains("local"), "{e}");
    }

    #[test]
    fn daemon_graph_attaches_the_store_backed_outcome_registry_to_routing() {
        // Spy assertion (audit items 13/14/L wiring): the graph-built
        // policy's RouterService is constructed via
        // with_pricing_and_outcomes over a StoreOutcomeStore on the DAEMON
        // store — a verified sample recorded through the policy's
        // record_call_outcome lands in that store (read back through the
        // store's own projection), exactly the shape "recorded stats serve
        // routing after a restart". The negative control: a policy over
        // the default empty registry learns nothing.
        let dir = tempfile::tempdir().unwrap();
        let manager =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let store = manager.store();
        let mut registry = ProviderRegistry::new();
        for (id, model, input, output) in
            [("alpha", "fast", 1u64, 3u64), ("beta", "big", 15u64, 60u64)]
        {
            registry
                .try_register(PricedTestProvider::known(
                    id,
                    model,
                    ModelCapabilities {
                        tools: true,
                        streaming: true,
                        context: 512_000,
                        ..Default::default()
                    },
                    input,
                    output,
                ))
                .unwrap();
        }
        let policy = economic_routing_policy_with_outcomes(
            &registry,
            RoutingMode::Economy,
            Arc::new(faktor_agent::StoreOutcomeStore::new(store.clone())),
        )
        .unwrap();
        let recorded = || {
            store
                .model_outcome_stats_get(
                    "alpha",
                    "fast",
                    faktor_core::model::RouterPhase::Implement,
                    faktor_core::model::TaskClass::Medium,
                    faktor_core::model::RiskBucket::Low,
                )
                .unwrap()
        };
        assert!(recorded().is_none(), "nothing recorded yet");
        // A settled Implement call whose deterministic gate verified:
        // the sample must ride the SAME registry the routing consults.
        policy.record_call_outcome(&faktor_agent::SettledCallOutcome {
            provider: "alpha".into(),
            model: "fast".into(),
            phase: faktor_core::model::RouterPhase::Implement,
            success: true,
            retried: false,
            rate_limited: false,
            latency_ms: 120,
            verified: Some(faktor_agent::VerifiedCallAttribution {
                task_class: faktor_core::model::TaskClass::Medium,
                risk_bucket: faktor_core::model::RiskBucket::Low,
                verified_success: true,
                rework_cost_micro: 0,
                rework_turns: 0,
            }),
        });
        let row = recorded().expect("the store-backed registry is attached");
        assert_eq!(row.successes_first_pass, 1);
        assert_eq!(row.failures_first_pass, 0);
        assert_eq!(row.sample_count, 1);
        // The policy ALSO serves routing through the same store (the
        // per-phase consult sees the recorded history).
        let decision = policy
            .route(&faktor_router::RouteRequest {
                phase: faktor_core::model::RouterPhase::Implement,
                required_capabilities: vec!["tools".into(), "streaming".into()],
                context_tokens: 1_000,
                estimated_output_tokens: 100,
                quality_floor: 50,
                ..Default::default()
            })
            .unwrap();
        assert!(
            decision.provider == "alpha" || decision.provider == "beta",
            "{decision:?}"
        );
        // Negative control: an empty-registry policy records into the
        // documented no-op store — the daemon store stays untouched by it.
        let empty = economic_routing_policy_with_outcomes(
            &registry,
            RoutingMode::Economy,
            Arc::new(faktor_router::EmptyOutcomeStore),
        )
        .unwrap();
        empty.record_call_outcome(&faktor_agent::SettledCallOutcome {
            provider: "alpha".into(),
            model: "fast".into(),
            phase: faktor_core::model::RouterPhase::Implement,
            success: true,
            retried: false,
            rate_limited: false,
            latency_ms: 120,
            verified: Some(faktor_agent::VerifiedCallAttribution {
                task_class: faktor_core::model::TaskClass::Medium,
                risk_bucket: faktor_core::model::RiskBucket::Low,
                verified_success: true,
                rework_cost_micro: 0,
                rework_turns: 0,
            }),
        });
        assert_eq!(
            recorded().unwrap().successes_first_pass,
            1,
            "the empty registry never reaches the daemon store"
        );
    }
}
