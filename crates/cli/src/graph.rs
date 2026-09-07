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

use std::sync::Arc;

use faktor_agent::AgentRuntime;
use faktor_core::model::{ModelDescriptor, ModelSource, RoutingMode};
use faktor_provider::catalog::{ModelCatalogEntry, PricingState, Provenance};
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

/// A router candidate descriptor for one registered provider model, built
/// from the provider's REAL catalog row (audit P0-1):
///
/// - [`PricingState::Known`] rows project their price fields onto the
///   V1 [`ModelEconomics`] the router reads (real prices, never defaults);
/// - [`PricingState::LocalZero`] rows project the explicit zero-cost
///   local marker the router already understands (`is_local_zero_cost`);
/// - [`PricingState::Unknown`] rows reach a descriptor ONLY through the
///   pinned path (the pin — not economics — decides; see
///   [`build_router_service`]); free-economy candidate lists exclude them
///   BEFORE a descriptor exists, so the router never sees a fabricated
///   zero where a price is missing.
///
/// Reliability priors come from the row's inspectable
/// `quality_prior` (conservative generic by default — the same numbers the
/// graph used to emit via `ModelEconomics::default()`, now named and
/// explicit), and `source` records the row's provenance.
fn descriptor_for(provider_id: &str, entry: &ModelCatalogEntry) -> ModelDescriptor {
    let caps = &entry.capabilities;
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
        economics: entry.economics(),
        source: match entry.provenance {
            Provenance::BuiltIn => ModelSource::ConservativeDefault,
            Provenance::ProviderCatalog => ModelSource::ProviderCatalog,
            // A Composite row is priced by USER policy (the configured
            // ceiling): the price is a budget bound, the user made it.
            Provenance::UserOverride | Provenance::Composite => ModelSource::UserOverride,
        },
    }
}

/// The exclusion/ceiling policy for free-economy candidate sets (audit
/// P0-1): [`PricingState::Unknown`] rows are EXCLUDED — an unknown price
/// is never zero and never the 1-microUSD runtime fallback. The ceiling
/// case never reaches this function as `Unknown`: a configured
/// `pricing_ceiling_micro_per_token` turns Unknown rows into
/// [`PricingState::Known`] at exactly the ceiling with provenance
/// [`Provenance::Composite`] inside the provider wrapper. `LocalZero`
/// (Ollama) and `Known` rows are always included.
fn candidate_entry_ok(entry: &ModelCatalogEntry) -> bool {
    !matches!(entry.pricing, PricingState::Unknown)
}

/// The daemon's router candidate set: every PRICED known model of every
/// registered provider (bounded by the registry and the providers' own
/// `known_models()`), built from each provider's real catalog rows.
///
/// Candidate-set policy by mode (audit P0-1 — unknown price != zero):
///
/// | pricing state | Economy / Balanced / MaximumQuality | Pinned |
/// |---|---|---|
/// | Known (real prices) | included at its real price | validation as today |
/// | LocalZero (Ollama) | always included, zero cost | validation as today |
/// | Unknown, no ceiling | **EXCLUDED** (never a fabricated 0 / 1-micro fallback) | pin included (the pin decides, economics only validates) |
/// | Unknown + configured ceiling | included at exactly the ceiling, provenance Composite (the ceiling is applied by the provider wrapper) | as today |
///
/// In Pinned mode the candidate set collapses to the pin itself: the
/// RouterService then VALIDATES the pin's capability/fit/budget/health
/// axes and the pin always wins when feasible — the router's free choice
/// can never silently substitute the configured pin (fail closed). A pin
/// whose (provider, model) is not among the registered models is a
/// graph-build error (loud, at boot — never a silent Economy).
pub fn build_router_service(
    providers: &ProviderRegistry,
    mode: &RoutingMode,
) -> Result<Arc<faktor_router::RouterService>, String> {
    let mut candidates: Vec<ModelDescriptor> = Vec::new();
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
                    if candidate_entry_ok(&entry) {
                        candidates.push(descriptor_for(&id, &entry));
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
            // did before catalogs existed.
            let entry = p.catalog_entry(model);
            candidates.push(descriptor_for(provider, &entry));
        }
    }
    Ok(Arc::new(faktor_router::RouterService::new(candidates)))
}

/// The daemon's economic routing policy over [`build_router_service`]'s
/// candidates.
pub fn economic_routing_policy(
    providers: &ProviderRegistry,
    mode: RoutingMode,
) -> Result<Arc<dyn faktor_agent::RoutingPolicy>, String> {
    let service = build_router_service(providers, &mode)?;
    Ok(faktor_agent::EconomicRoutingPolicy::new(service, mode))
}

#[cfg(test)]
mod tests {
    use super::*;
    use faktor_core::model::{MicroUsdPerToken, ModelCapabilities, ModelEconomics};
    use faktor_provider::catalog::{
        ModelCatalogEntry, PricingState, Provenance, QualityPrior, CATALOG_FIRST_EPOCH,
    };
    use faktor_provider::{FakeProvider, Provider, ProviderStream};
    use std::sync::Arc;

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
                PricingState::Known(ModelEconomics {
                    input_price_per_mtok: MicroUsdPerToken::from_dollars_per_million(input),
                    output_price_per_mtok: MicroUsdPerToken::from_dollars_per_million(output),
                    cache_read_price_per_mtok: MicroUsdPerToken(0),
                    cache_write_price_per_mtok: MicroUsdPerToken(0),
                    estimated_latency_ms: 800,
                    ..Default::default()
                }),
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
        let svc = build_router_service(&registry, &RoutingMode::Economy).unwrap();
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
        assert!(build_router_service(&empty, &RoutingMode::Economy)
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
            let svc = build_router_service(&registry, &mode).unwrap();
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
    fn configured_ceiling_admits_unknown_models_at_exactly_the_ceiling_as_composite() {
        // The config surface (ProviderCfg.pricing ceiling) wraps the
        // endpoint: its Unknown rows enter the economy candidate set priced
        // at EXACTLY the ceiling with Composite provenance.
        let cfg = crate::config::ProviderCfg::OpenAi {
            id: "corp-proxy".into(),
            base_url: "https://corp.example.com/v1".into(),
            api_key_env: None,
            pricing: Some(crate::config::ProviderPricingCfg {
                pricing_ceiling_micro_per_token: Some(42),
                ..Default::default()
            }),
        };
        let mut registry = ProviderRegistry::new();
        registry
            .try_register(cfg.build(open_transport()).unwrap())
            .unwrap();
        // The endpoint row is Unknown at the adapter level...
        assert_eq!(
            registry
                .get("corp-proxy")
                .unwrap()
                .catalog_entry("default")
                .pricing,
            PricingState::Known(ModelEconomics {
                input_price_per_mtok: MicroUsdPerToken(42),
                output_price_per_mtok: MicroUsdPerToken(42),
                cache_read_price_per_mtok: MicroUsdPerToken(42),
                cache_write_price_per_mtok: MicroUsdPerToken(42),
                ..Default::default()
            })
        );
        // ...and the economy candidate set includes it at the ceiling.
        let svc = build_router_service(&registry, &RoutingMode::Economy).unwrap();
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
        let svc = build_router_service(&registry, &RoutingMode::Economy).unwrap();
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
        let economy = economic_routing_policy(&registry, RoutingMode::Economy).unwrap();
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
        // Pinned to beta: the policy validates and returns beta — the free
        // evaluation prefers alpha (cheaper), the pin never loses.
        let pinned = economic_routing_policy(
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
        let svc = build_router_service(&registry, &mode).unwrap();
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
        // Unknown pin: the daemon refuses to boot on it (loud, at build).
        let bad = RoutingMode::Pinned {
            provider: "nope".into(),
            model: "m".into(),
        };
        assert!(build_router_service(&registry, &bad).is_err());
        let bad_model = RoutingMode::Pinned {
            provider: "paid".into(),
            model: "not-a-model".into(),
        };
        assert!(build_router_service(&registry, &bad_model).is_err());
    }

    #[test]
    fn exact_user_override_table_prices_unknown_models_and_bumps_the_epoch() {
        use crate::config::ProviderPricingCfg;
        let mut registry = ProviderRegistry::new();
        for (id, pricing) in [
            (
                "corp-proxy",
                ProviderPricingCfg {
                    input_micro_usd_per_token: Some(2),
                    output_micro_usd_per_token: Some(8),
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
            PricingState::Known(e) => {
                assert_eq!(e.input_price_per_mtok, MicroUsdPerToken(2));
                assert_eq!(e.output_price_per_mtok, MicroUsdPerToken(8));
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
        let svc = build_router_service(&registry, &RoutingMode::Economy).unwrap();
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
        let svc = build_router_service(&registry, &RoutingMode::Economy).unwrap();
        assert_eq!(svc.router.candidates.len(), 1);
        assert!(svc.router.candidates[0].economics.is_local_zero_cost());
        let hostile = crate::config::ProviderCfg::Ollama {
            id: "ollama".into(),
            base_url: None,
            pricing: Some(ProviderPricingCfg {
                input_micro_usd_per_token: Some(15),
                output_micro_usd_per_token: Some(60),
                ..Default::default()
            }),
        };
        let e = match hostile.build(open_transport()) {
            Ok(_) => panic!("ollama pricing refused"),
            Err(e) => e,
        };
        assert!(e.contains("local"), "{e}");
    }
}
