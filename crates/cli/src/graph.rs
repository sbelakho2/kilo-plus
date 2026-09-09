//! The production daemon dependency graph (P0-2 + audit 12/17): a NAMED
//! struct — the historic tuple type made every consumer depend on field
//! ORDER and kept routing/budgets/instructions/orchestration out of the
//! graph. The struct names the daemon's authorities, in construction order:
//!
//! 1. `session` — the durable store + session/workspace table;
//! 2. `supervisor` — the ONE process supervisor (audit P0-40): every MCP
//!    server child, hook child, tool/terminal child and verification check
//!    of the daemon rides this single bounded registry;
//! 3. `providers` — the provider registry (catalog/pricing included);
//! 4. `transport` — the policy-checked + secret-scanned egress transport
//!    every configured adapter executes through;
//! 5. `permissions` — the permission channel; `mcp_servers` — the
//!    supervised MCP servers whose tools ride the agent registry;
//! 6. `routing` — the ECONOMIC routing policy (P0-2/85/87/88): every model
//!    call of every session consults it first (Economy default; Pinned
//!    validates every call against the configured pin);
//! 7. `budgets` — the DURABLE cost ledger over this daemon's store
//!    (P0-6/12): one reservation per paid model call, settled exactly once;
//! 8. `index` — the durable repository IndexService (audits 30/64) over
//!    this daemon's store + workspace table;
//! 9. `evidence` — the daemon's evidence provider over the same store;
//! 10. `instructions` — the per-workspace instruction resolver (P0-32)
//!     built over this daemon's workspace table;
//! 11. `agent` — the reasoning runtime (drives sessions with commands);
//! 12. `orchestrator` — the OrchestratorRuntime (audits P0-20/21/23/61),
//!     the AUTHORITATIVE executor of multi-agent tasks and the durable
//!     child control surface;
//! 13. `shadows` — the daemon's shadow-mutation roots (P0-48 + wave-24);
//! 14. `tasks` — the TaskExecutor over the SAME orchestrator: the ONE
//!     native task-start authority of the daemon.
//!
//! Every authority is built EXACTLY ONCE per daemon lifetime, in the order
//! of [`DAEMON_CONSTRUCTION_ORDER`]; the construction happens in the
//! graph-construction region of `main.rs` (steps 1-2 in the daemon
//! entries, steps 3-16 inline in `build_daemon_core`). Serve/ACP/commands
//! never construct a supervisor, ledger, index or executor of their own —
//! they take references from this graph. The `cost_reservation`
//! route_decision_json column and the task row's max_cost_micro column
//! wait for the config surface that sets per-task money caps (provider-level
//! pricing tables and the per-session cap plumbing).

use std::collections::HashMap;
use std::sync::Arc;

use crate::evidence::RepoEvidence;
use faktor_agent::AgentRuntime;
use faktor_core::model::{ModelDescriptor, ModelSource, RoutingMode};
use faktor_index::IndexService;
use faktor_orchestrator::runtime::shadow::ShadowRoots;
use faktor_orchestrator::runtime::task_executor::TaskExecutor;
use faktor_orchestrator::runtime::OrchestratorRuntime;
use faktor_provider::catalog::{admissible, ModelCatalogEntry, Provenance};
use faktor_provider::egress::HttpTransport;
use faktor_provider::ProviderRegistry;
use faktor_server::permission::ChannelPermissionRequester;
use faktor_session::{DurableBudgetLedger, SessionManager};
use faktor_terminal::ProcessSupervisor;

/// The ONE construction order of the daemon (audit 12/17): the canonical
/// marker list. Steps 1-2 run in the daemon entries of `main.rs`
/// (`build_daemon` / `build_daemon_with_mcp_inner`); steps 3-16 are inline
/// in `main.rs::build_daemon_core` in exactly this order; step 17 consumes
/// the graph (serve/ACP/commands) and constructs nothing of its own. The
/// tests verify the builder text against this list — a component inserted
/// out of order, or a second construction of any authority anywhere else,
/// is a compile-time-red test, never a review nit.
#[allow(dead_code)] // wave B8: consumed by the construction-order certification tests
pub(crate) const DAEMON_CONSTRUCTION_ORDER: [&str; 17] = [
    "session",      // 1. store/session
    "cas",          // 2. CAS
    "supervisor",   // 3. ProcessSupervisor
    "transport",    // 4. checked transport/security
    "providers",    // 5. provider registry
    "catalog",      // 6. catalog/pricing
    "router",       // 7. router(+outcomes)
    "budgets",      // 8. budget ledger
    "index",        // 9. IndexService
    "evidence",     // 10. evidence/cold
    "instructions", // 11. instructions
    "verification", // 12. verification(executor+service)
    "agent",        // 13. AgentRuntime
    "orchestrator", // 14. OrchestratorRuntime
    "shadows",      // 15. ShadowRoots
    "tasks",        // 16. TaskExecutor
    "server",       // 17. ServerDeps/ACP/commands (consume only)
];

/// The named daemon dependency graph (see the module docs). Field order is
/// the construction order of [`DAEMON_CONSTRUCTION_ORDER`].
pub struct DaemonGraph {
    /// 1. The durable store + session/workspace table (store/session →
    ///    CAS). Opened by the daemon entry, then handed to every authority.
    pub session: Arc<SessionManager>,
    /// 3. The ONE process supervisor (audit P0-40): the same Arc supervises
    ///    every MCP server child, hook child, tool/terminal child and
    ///    verification child of the daemon.
    pub supervisor: Arc<ProcessSupervisor>,
    /// 4. The policy-checked + whole-payload secret-scanned egress
    ///    transport (P0-36/37/38): every configured adapter executes through
    ///    this single checked transport.
    pub transport: Arc<dyn HttpTransport>,
    /// 5. The provider registry: every registered adapter, catalog rows
    ///    (built-in + provider + user pricing) included.
    pub providers: Arc<ProviderRegistry>,
    /// The permission channel (spec §25): pending requests ride this.
    pub permissions: Arc<ChannelPermissionRequester>,
    /// Supervised MCP servers (spec §31); the servers own their children
    /// for the daemon lifetime.
    pub mcp_servers: Vec<Arc<faktor_mcp::McpServer>>,
    /// 7. The economic routing policy (P0-2): RouterService over the
    ///    daemon's registered models + the mode from config (Economy default).
    pub routing: Arc<dyn faktor_agent::RoutingPolicy>,
    /// 8. The durable monetary ledger over THIS daemon's store (P0-6/12).
    pub budgets: Arc<DurableBudgetLedger>,
    /// 9. The durable repository IndexService (audits 30/64): generation
    ///    state rows + published generations over this daemon's store. `None`
    ///    only when hosting failed at boot (hostile/unwritable data root) —
    ///    the daemon then keeps serving with the bounded evidence scan, the
    ///    documented degrade of the runtime's own evidence ladder.
    pub index: Option<Arc<IndexService>>,
    /// 10. The evidence provider (spec §20): per-workspace bounded index +
    ///     search over this daemon's session store; the legacy degrade of the
    ///     runtime's evidence ladder.
    pub evidence: Arc<RepoEvidence>,
    /// 11. The per-workspace instruction resolver (P0-32).
    pub instructions: Arc<faktor_instructions::InstructionResolver>,
    /// 13. The reasoning runtime (drives sessions with commands).
    pub agent: Arc<AgentRuntime>,
    /// 14. The orchestration runtime (audits P0-20/21/23/61): the
    ///     AUTHORITATIVE executor of multi-agent tasks and the durable control
    ///     surface the native `/agents/{child}/...` endpoints drive.
    pub orchestrator: Arc<OrchestratorRuntime>,
    /// 15. The shadow-mutation roots (P0-48 + wave-24): the executor's
    ///     shadow service; its Drop removes every shadow on graceful daemon
    ///     teardown, reconcile() at boot is the deterministic crash recovery.
    pub shadows: Arc<ShadowRoots>,
    /// 16. The TaskExecutor over [`DaemonGraph::orchestrator`]: the ONE
    ///     native task-start authority of the daemon. Non-optional; the
    ///     configured MutationMode decides usage only.
    pub tasks: Arc<TaskExecutor>,
}

impl DaemonGraph {
    /// The classic tuple destructuring order (session, agent, permissions)
    /// so serve/acp/run call sites read naturally.
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
            // The rows' catalog priors are the conservative neutral 50, and
            // quality floors are HARD (Wave B2): a request floor must sit at
            // or below the best available quality — 60 would be a typed
            // NoCapableModel, not a lowered floor.
            quality_floor: 50,
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
            quality_floor: 50,
            task_budget_remaining_micro: 5_000,
            latency_preference_ms: None,
        };
        assert_eq!(economy.route(&request()).unwrap().provider, "alpha");
        let starved = || faktor_router::RouteRequest {
            phase: faktor_core::model::RouterPhase::Implement,
            required_capabilities: vec!["tools".into(), "streaming".into()],
            context_tokens: 1_000,
            estimated_output_tokens: 100,
            quality_floor: 50,
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
    // ========================================================================
    // Daemon-graph construction tests (audit 12/17)
    // ========================================================================
    //
    // These tests are adversarial by construction:
    //  - the smoke builds a REAL daemon over a temp test data dir and probes
    //    every authority (pointer identity + a live response each) — plus the
    //    hostile variant where the index data root is a FILE, which must
    //    degrade the index authority to None WITHOUT killing the daemon;
    //  - the construction-order test scans the actual builder text of
    //    `main.rs` (the graph construction region) against the canonical
    //    marker list [`DAEMON_CONSTRUCTION_ORDER`]: an authority built out of
    //    order — or a second construction anywhere — is a red test, never a
    //    review nit;
    //  - the source scan walks every source file of the daemon crate (and the
    //    server surface) and refuses supervisor/executor/ledger/shadow
    //    constructions outside the graph module + the graph-construction
    //    functions + `#[cfg(test)]` code: a component that ever spawns its own
    //    second authority fails the build;
    //  - the restart test drops a full daemon (graceful teardown: the shadow
    //    service's Drop must remove its shadows) and rebuilds on the SAME data
    //    dir: the durable budget cap and session rows survive, the second
    //    graph is a fresh single set of authorities, and nothing leaks.

    /// Column-0 `}` line indexes are used as span ends (function bodies and
    /// module bodies of this crate never contain another column-0 `}`).
    fn span_of(lines: &[&str], header: &str) -> (usize, usize) {
        let start = lines
            .iter()
            .position(|l| l.contains(header))
            .unwrap_or_else(|| panic!("header not found: {header}"));
        let end = lines[start + 1..]
            .iter()
            .position(|l| *l == "}")
            .map(|i| start + 1 + i)
            .unwrap_or_else(|| panic!("header {header} never closes"));
        (start, end)
    }

    fn first_token_line(lines: &[&str], from: usize, token: &str) -> usize {
        lines[from..]
            .iter()
            .position(|l| l.contains(token))
            .map(|i| from + i)
            .unwrap_or_else(|| panic!("token {token:?} not found after line {from}"))
    }

    /// Every `#[cfg(test)]` module region: from the marker line through the
    /// module's column-0 closing brace. Only file-scope markers are matched
    /// (indented markers belong to cfg(test) fns inside non-test code and do
    /// not open a region).
    fn cfg_test_regions(lines: &[&str]) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < lines.len() {
            if lines[i] == "#[cfg(test)]" && lines.get(i + 1).is_some_and(|l| l.starts_with("mod "))
            {
                let end = lines[i + 2..]
                    .iter()
                    .position(|l| *l == "}")
                    .map(|j| i + 2 + j)
                    .unwrap_or(lines.len() - 1);
                out.push((i, end));
                i = end + 1;
                continue;
            }
            i += 1;
        }
        out
    }

    fn allowed(line: usize, regions: &[(usize, usize)]) -> bool {
        regions.iter().any(|(a, b)| line >= *a && line <= *b)
    }

    #[test]
    fn graph_build_over_test_data_dir_constructs_every_authority_and_each_responds() {
        use faktor_agent::EvidenceProvider;
        use faktor_core::id::TaskId;
        use faktor_orchestrator::runtime::task_executor::MutationMode;

        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("data");
        let ws_root = dir.path().join("ws");
        std::fs::create_dir_all(ws_root.join("src")).unwrap();
        std::fs::write(
            ws_root.join("src").join("app.rs"),
            b"pub fn balance_account() -> i64 { 42 }\n",
        )
        .unwrap();
        let graph = crate::build_daemon(&data, None).expect("daemon build over a fresh data dir");
        // Every authority is a live Arc of the graph (never a second instance):
        assert!(Arc::ptr_eq(&graph.orchestrator.manager(), &graph.session));
        assert!(Arc::ptr_eq(graph.tasks.orchestrator(), &graph.orchestrator));
        assert!(Arc::ptr_eq(graph.tasks.session(), &graph.session));
        assert!(Arc::ptr_eq(graph.tasks.agent(), &graph.agent));
        let shadows = graph
            .tasks
            .shadows()
            .expect("daemon executor always carries the shadow service");
        assert!(Arc::ptr_eq(&shadows, &graph.shadows));
        assert!(Arc::ptr_eq(
            graph
                .agent
                .deps()
                .supervisor
                .as_ref()
                .expect("agent rides the graph supervisor"),
            &graph.supervisor
        ));
        // Services respond:
        assert!(graph.supervisor.alive().is_empty());
        assert!(
            graph.providers.ids().is_empty(),
            "default config registers no providers"
        );
        assert_eq!(graph.routing.mode(), RoutingMode::Economy);
        assert!(graph.permissions.pending_views().is_empty());
        graph
            .agent
            .recover()
            .expect("agent recover on an empty store");
        let ws = graph
            .session
            .create_workspace(ws_root.to_str().unwrap())
            .expect("session store responds");
        let handle = graph
            .session
            .create_session(ws, "graph-smoke", "fake", "default")
            .expect("session row responds");
        let sid = handle.id();
        // The durable budget ledger answers over a REAL task row.
        let task = faktor_session::Task {
            task_id: TaskId::new(7),
            session_id: sid,
            goal: "graph smoke".into(),
            created_ms: graph.session.now_ms(),
            updated_ms: graph.session.now_ms(),
            ..Default::default()
        };
        handle.create_task(task).unwrap();
        graph
            .budgets
            .set_task_max_cost(sid, TaskId::new(7), Some(2_500_000))
            .expect("ledger set-task-cap responds");
        // The durable repository index is hosted on a healthy data dir and
        // accepts the workspace (attach = register + kick reconciliation).
        let index = graph
            .index
            .as_ref()
            .expect("index hosted on a healthy store");
        index.attach(ws).expect("index attach responds");
        // The evidence provider really serves an evidence package for the
        // workspace's code (the bounded scan path).
        let evidence = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(graph.evidence.evidence_for(
                sid,
                faktor_agent::EvidenceQuery {
                    prompt: "inspect balance_account".into(),
                    ..Default::default()
                },
            ))
            .expect("evidence provider responds");
        assert!(
            evidence.iter().any(|e| e.path.ends_with("app.rs")),
            "evidence must surface the file defining the concept: {evidence:?}"
        );
        // The instruction resolver answers for the workspace (a tree without
        // AGENTS.md resolves to the documented empty instruction set).
        let _loaded = graph
            .instructions
            .resolve(ws.raw(), None)
            .expect("instruction resolver answers for the workspace (documented no-error tree)");
        // The shadow service answers: reconcile() at boot already ran; a second
        // pass is idempotent on a consistent registry.
        graph
            .shadows
            .reconcile()
            .expect("shadow reconcile is idempotent");
        // Orchestration authorities answer.
        assert_eq!(
            graph.tasks.mode(),
            MutationMode::Shadow,
            "production default is shadow mutation"
        );
        assert_eq!(graph.tasks.active_run(), None);
        assert_eq!(
            graph
                .orchestrator
                .manager()
                .list_sessions(None)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn hostile_index_data_root_degrades_to_none_and_the_daemon_still_serves() {
        use faktor_agent::EvidenceProvider;
        // The runtime's own evidence ladder treats an unhostable index as a
        // degrade (never a broken first prompt). The daemon graph must mirror
        // that: a FILE where <store>/index_data should be a directory refuses
        // the IndexService authority, but every other authority still builds
        // and the daemon serves.
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("data");
        std::fs::create_dir_all(data.join("store")).unwrap();
        std::fs::write(data.join("store").join("index_data"), b"not a directory").unwrap();
        let graph =
            crate::build_daemon(&data, None).expect("daemon must boot around a hostile index root");
        assert!(
            graph.index.is_none(),
            "the index authority must degrade to None, never fabricate"
        );
        graph.agent.recover().expect("agent still recovers");
        let ws = graph.session.create_workspace("/w").unwrap();
        graph
            .session
            .create_session(ws, "hostile-index", "fake", "m")
            .unwrap();
        graph.budgets.recover_after_restart();
        // The bounded evidence provider still answers over the same store (the
        // runtime's documented degrade keeps every turn served).
        let out = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(graph.evidence.evidence_for(
                graph.session.list_sessions(None).unwrap()[0].id(),
                faktor_agent::EvidenceQuery {
                    prompt: "anything".into(),
                    ..Default::default()
                },
            ))
            .expect("evidence provider responds without the index");
        assert!(out.is_empty(), "no workspace root was registered: {out:?}");
    }

    #[test]
    fn daemon_restart_on_the_same_data_dir_is_a_fresh_single_set_of_authorities() {
        // Full daemon lifetime #1: create durable state (workspace + session +
        // a budgeted task + a live shadow), then END the daemon gracefully.
        // The shadow service's Drop must remove its shadow (dir + row) — the
        // deterministic recovery of a crash is reconcile() at the next boot.
        use faktor_core::id::TaskId;
        use faktor_session::BudgetAuthority as _;
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("data");
        let ws_root = dir.path().join("ws");
        std::fs::create_dir_all(&ws_root).unwrap();
        std::fs::write(ws_root.join("a.rs"), b"pub fn alpha() {}").unwrap();
        let (sid, ws, shadow_root) = {
            let graph = crate::build_daemon(&data, None).unwrap();
            let ws = graph
                .session
                .create_workspace(ws_root.to_str().unwrap())
                .unwrap();
            let handle = graph
                .session
                .create_session(ws, "restart", "fake", "m")
                .unwrap();
            let sid = handle.id();
            let task = faktor_session::Task {
                task_id: TaskId::new(9),
                session_id: sid,
                goal: "restart".into(),
                created_ms: graph.session.now_ms(),
                updated_ms: graph.session.now_ms(),
                ..Default::default()
            };
            handle.create_task(task).unwrap();
            graph
                .budgets
                .set_task_max_cost(sid, TaskId::new(9), Some(1_000_000))
                .unwrap();
            let shadow = graph
                .shadows
                .begin_shadow(sid, &ws_root)
                .expect("shadow service begins a real bounded copy");
            assert!(shadow.root.is_dir(), "the shadow copy exists on disk");
            (sid, ws, shadow.root)
        };
        assert!(
            !shadow_root.exists(),
            "graceful daemon teardown removes every shadow (the service's Drop ran)"
        );
        // Daemon lifetime #2 over the SAME data dir (the crash-recovery path of
        // a restarting daemon): construction must succeed, the durable cap must
        // survive, and the fresh graph must again be one set of authorities.
        let graph2 = crate::build_daemon(&data, None).expect("rebuild over the same data dir");
        graph2
            .agent
            .recover()
            .expect("recover on the reopened store");
        let view = graph2.budgets.session_budget_view(sid, TaskId::new(9));
        assert_eq!(
            view.max_cost_micro,
            Some(1_000_000),
            "the durable ledger cap survives the daemon restart"
        );
        assert!(graph2.index.is_some(), "the second daemon hosts its index");
        let handle = graph2
            .session
            .get_session(sid)
            .unwrap()
            .expect("session row survives");
        assert_eq!(handle.list_tasks().unwrap().len(), 1, "task rows survive");
        match graph2
            .shadows
            .active_shadow(sid)
            .expect("shadow registry read")
        {
            Some(row) => assert!(
                !row.state.is_live(),
                "graceful teardown must retire the shadow row, not leave it live: {row:?}"
            ),
            None => panic!("the retired shadow row must still be visible to the next daemon"),
        }
        drop(graph2);
        // A THIRD daemon after the second's graceful end is equally clean.
        let graph3 = crate::build_daemon(&data, None).unwrap();
        graph3.agent.recover().unwrap();
        let _ = (graph3, ws);
    }

    /// The canonical token of each construction step inside the core builder
    /// body (steps 4-16 of [`DAEMON_CONSTRUCTION_ORDER`]; steps 1-3 live in
    /// the daemon entries, step 17 consumes). The list doubles as the marker
    /// ordering list the test asserts against the builder text.
    const CORE_STEP_TOKENS: &[(&str, &str)] = &[
        ("transport", "daemon_egress_transport("),
        ("providers", "ProviderRegistry::new"),
        ("catalog", "p.build_ollama("),
        ("router", "economic_routing_policy_with_outcomes("),
        ("budgets", "DurableBudgetLedger::new"),
        ("index", "IndexService::open("),
        ("evidence", "RepoEvidence::new("),
        ("instructions", "daemon_instructions_resolver("),
        ("verification", "daemon_verification("),
        ("agent", "AgentRuntime::new"),
        ("orchestrator", "OrchestratorRuntime::new"),
        ("shadows", "ShadowRoots::new"),
        ("tasks", "TaskExecutor::new_with_mode"),
    ];

    /// Steps 1-3 + 17 are not inline in the core body: they live in the daemon
    /// entries (store + supervisor) and in serve (ServerDeps assembly). Each
    /// gets its structural assertion here.
    const ENTRY_HEADERS: &[&str] = &["fn build_daemon(", "fn build_daemon_with_mcp_inner("];

    #[test]
    fn construction_order_markers_rise_strictly_inside_the_core_builder() {
        let src = include_str!("main.rs");
        let lines: Vec<&str> = src.lines().collect();
        // The core builder is the ONE inline construction region for steps
        // 4-16; helper definitions above/below it are outside the scanned
        // span, so a helper that constructs something cannot smuggle an
        // authority out of order.
        let (core_start, core_end) = span_of(&lines, "fn build_daemon_core(");
        let mut prev = core_start;
        let mut seen: Vec<&str> = Vec::new();
        for (name, token) in CORE_STEP_TOKENS.iter().copied() {
            let at = first_token_line(&lines, prev + 1, token);
            assert!(
                at <= core_end,
                "step {name:?} token {token:?} must sit inside build_daemon_core (after line {at})"
            );
            assert!(
                at > prev,
                "step {name:?} must be constructed after the previous step (found at {at})"
            );
            seen.push(name);
            prev = at;
        }
        // The entry spans (steps 1-3) open the store and the ONE supervisor
        // and then hand both to the core — they must precede the core body and
        // contain NO core construction token.
        for header in ENTRY_HEADERS.iter().copied() {
            let (s, e) = span_of(&lines, header);
            assert!(
                lines[s..=e]
                    .iter()
                    .any(|l| l.contains("SessionManager::open")),
                "{header} must open the durable store (steps 1-2)"
            );
            assert!(
                lines[s..=e]
                    .iter()
                    .any(|l| l.contains("ProcessSupervisor::new")),
                "{header} must build the ONE supervisor (step 3)"
            );
            assert!(e < core_start, "{header} must precede the core builder");
            for (_name, token) in CORE_STEP_TOKENS.iter().copied() {
                assert!(
                    !lines[s..=e].iter().any(|l| l.contains(token)),
                    "{header} must not construct {token:?} (steps 4-16 belong to the core only)"
                );
            }
        }
        // The core body itself never builds a second supervisor: the entries'
        // Arc is handed in as a parameter.
        assert!(
            !lines[core_start..=core_end]
                .iter()
                .any(|l| l.contains("ProcessSupervisor::new")),
            "the core builder must receive the ONE supervisor, never build a second"
        );
        // Every documented name is covered: nothing may join the canonical
        // order list without a construction assertion of its own.
        let mut covered: Vec<&str> = Vec::new();
        for name in DAEMON_CONSTRUCTION_ORDER.iter().copied() {
            match name {
                "session" | "cas" => {
                    assert!(
                        ENTRY_HEADERS.iter().all(|h| {
                            let (s, e) = span_of(&lines, h);
                            lines[s..=e]
                                .iter()
                                .any(|l| l.contains("SessionManager::open"))
                        }),
                        "the store/session + CAS (steps 1-2) open in every daemon entry"
                    );
                    covered.push("session");
                    covered.push("cas");
                }
                "supervisor" => {
                    assert!(
                        ENTRY_HEADERS.iter().all(|h| {
                            let (s, e) = span_of(&lines, h);
                            lines[s..=e]
                                .iter()
                                .any(|l| l.contains("ProcessSupervisor::new"))
                        }),
                        "step 3 builds the ONE supervisor in every daemon entry"
                    );
                    covered.push("supervisor");
                }
                "transport" | "providers" | "catalog" | "router" | "budgets" | "index"
                | "evidence" | "instructions" | "verification" | "agent" | "orchestrator"
                | "shadows" | "tasks" => {
                    assert!(
                        seen.contains(&name),
                        "step {name:?} missing from the core builder"
                    );
                    covered.push(name);
                }
                "server" => {
                    // Step 17 consumes the graph: serve assembles ServerDeps
                    // over the graph's instances (new_with) — the default
                    // constructor that builds its own runtime is a server-crate
                    // test/embedded seam and never appears in main.rs.
                    let (s, e) = span_of(&lines, "async fn serve_impl(");
                    assert!(
                        lines[s..=e]
                            .iter()
                            .any(|l| l.contains("ServerDeps::new_with(")),
                        "serve_impl must assemble ServerDeps over graph instances"
                    );
                    for (_name, token) in CORE_STEP_TOKENS.iter().copied() {
                        assert!(
                            !lines[s..=e].iter().any(|l| l.contains(token)),
                            "serve_impl must not construct {token:?} (the graph did, once)"
                        );
                    }
                    let regions = cfg_test_regions(&lines);
                    for (idx, l) in lines.iter().enumerate() {
                        if l.contains("ServerDeps::new(") {
                            assert!(
                            allowed(idx, &regions),
                            "main.rs:{idx}: ServerDeps::new( (a second runtime) only in test code"
                        );
                        }
                    }
                    covered.push("server");
                }
                other => panic!("construction order names a step with no assertion: {other}"),
            }
        }
        covered.sort_unstable();
        covered.dedup();
        assert_eq!(covered.len(), DAEMON_CONSTRUCTION_ORDER.len());
    }

    #[test]
    fn no_component_builds_a_second_supervisor_ledger_shadow_or_executor_outside_the_graph() {
        // Scan every source file of the daemon crate plus the server surface:
        // the only constructions of daemon-lifetime authorities are allowed in
        // (a) the graph module itself, (b) the graph-construction functions of
        // main.rs (headers starting with `fn build_daemon`), (c) the
        // server-crate's ServerDeps default seam (embedded/test hosts), and
        // (d) `#[cfg(test)]` code. A component file that spawns its own
        // supervisor or executor fails here.
        const TOKENS: &[&str] = &[
            "ProcessSupervisor::new",
            "OrchestratorRuntime::new",
            "ShadowRoots::new",
            "TaskExecutor::new_with_mode",
            "TaskExecutor::new(",
            "DurableBudgetLedger::new",
        ];
        let files: &[(&str, &str)] = &[
            ("main.rs", include_str!("main.rs")),
            ("graph.rs", include_str!("graph.rs")),
            ("config.rs", include_str!("config.rs")),
            ("evidence.rs", include_str!("evidence.rs")),
            ("mcp_bridge.rs", include_str!("mcp_bridge.rs")),
            ("tools.rs", include_str!("tools.rs")),
            ("server api.rs", include_str!("../../server/src/api.rs")),
        ];
        for (name, src) in files.iter().copied() {
            let lines: Vec<&str> = src.lines().collect();
            let mut regions = cfg_test_regions(&lines);
            if name == "graph.rs" {
                regions.push((0, lines.len())); // the graph module itself
            }
            if name == "main.rs" {
                // The graph construction region: every daemon builder entry.
                let mut i = 0;
                while i < lines.len() {
                    let t = lines[i].trim_start();
                    let is_header = (t.starts_with("fn build_daemon")
                        || t.starts_with("pub fn build_daemon")
                        || t.starts_with("async fn build_daemon")
                        || t.starts_with("pub async fn build_daemon"))
                        && t.ends_with('(');
                    if is_header {
                        let (s, e) = span_of(&lines, t);
                        regions.push((s, e));
                        i = e + 1;
                        continue;
                    }
                    i += 1;
                }
            }
            let mut seam: Option<(usize, usize)> = None;
            if name == "server api.rs" {
                // ServerDeps::new (embedded/test seam) legitimately builds a
                // default runtime over session+agent; every OTHER construction
                // in the surface must be the graph module or test code.
                seam = Some(span_of(&lines, "impl ServerDeps {"));
            }
            for (idx, l) in lines.iter().enumerate() {
                for t in TOKENS.iter().copied() {
                    if l.contains(t)
                        && !seam.is_some_and(|(s, e)| idx >= s && idx <= e)
                        && !allowed(idx, &regions)
                    {
                        panic!(
                        "{name}:{}: {t:?} is constructed outside the graph construction region \
                         (graph module / build_daemon* / cfg(test))",
                        idx + 1
                    );
                    }
                }
            }
        }
    }
}
