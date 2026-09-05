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
use faktor_core::model::{
    ModelCapabilities, ModelDescriptor, ModelEconomics, ModelSource, RoutingMode,
};
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

/// A router candidate descriptor for one registered provider model
/// (capabilities from the live provider — never guessed; economics default
/// to the zero-cost local profile: real per-model pricing tables are a
/// config-surface item of a later wave, and the policy's effective-floor
/// rule never lets an unpriced candidate set spuriously deny).
fn descriptor_for(provider_id: &str, model: &str, caps: &ModelCapabilities) -> ModelDescriptor {
    ModelDescriptor {
        provider: provider_id.to_string(),
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

/// The daemon's router candidate set: every known model of every registered
/// provider (bounded by the registry and the providers' own
/// `known_models()`). In Pinned mode the candidate set collapses to the pin
/// itself: the RouterService then VALIDATES the pin's capability/fit/
/// budget/health axes and the pin always wins when feasible — the router's
/// free choice can never silently substitute the configured pin (fail
/// closed). A pin whose (provider, model) is not among the registered
/// models is a graph-build error (loud, at boot — never a silent Economy).
pub fn build_router_service(
    providers: &ProviderRegistry,
    mode: &RoutingMode,
) -> Result<Arc<faktor_router::RouterService>, String> {
    let mut candidates: Vec<ModelDescriptor> = Vec::new();
    match mode {
        RoutingMode::Economy => {
            for id in providers.ids() {
                let Some(p) = providers.get(&id) else {
                    continue;
                };
                for model in p.known_models() {
                    let caps = p.capabilities(&model);
                    candidates.push(descriptor_for(&id, &model, &caps));
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
            let caps = p.capabilities(model);
            candidates.push(descriptor_for(provider, model, &caps));
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
    use faktor_provider::FakeProvider;

    #[test]
    fn economy_candidates_cover_every_registered_model_with_live_capabilities() {
        let mut registry = ProviderRegistry::new();
        registry.register(Arc::new(FakeProvider::with_script(
            "local-a",
            ModelCapabilities {
                tools: true,
                context: 64_000,
                ..Default::default()
            },
            vec![],
        )));
        // FakeProvider.known_models defaults to ["default"].
        let svc = build_router_service(&registry, &RoutingMode::Economy).unwrap();
        assert_eq!(svc.router.candidates.len(), 1);
        let c = &svc.router.candidates[0];
        assert_eq!(c.provider, "local-a");
        assert_eq!(c.model, "default");
        assert!(c.tools, "capabilities come from the live provider");
        assert_eq!(c.context, 64_000);
        // Empty registry -> empty candidates (every route then fails typed;
        // nothing silently falls back).
        let empty = ProviderRegistry::new();
        assert!(build_router_service(&empty, &RoutingMode::Economy)
            .unwrap()
            .router
            .candidates
            .is_empty());
    }

    fn fake_registry(two: bool) -> ProviderRegistry {
        let mut registry = ProviderRegistry::new();
        registry.register(Arc::new(FakeProvider::with_script(
            "alpha",
            ModelCapabilities {
                tools: true,
                streaming: true,
                context: 128_000,
                ..Default::default()
            },
            vec![],
        )));
        if two {
            registry.register(Arc::new(FakeProvider::with_script(
                "beta",
                ModelCapabilities {
                    tools: true,
                    streaming: true,
                    context: 128_000,
                    ..Default::default()
                },
                vec![],
            )));
        }
        registry
    }

    #[test]
    fn economy_policy_routes_deterministically_and_pinned_keeps_the_pin() {
        // (a) graph-level: an economy policy over two registered models
        // makes a decision (deterministic: zero-cost models tie-break on
        // the provider/model key), and a pinned policy returns the PIN even
        // when the free economy evaluation would have picked the other one.
        let registry = fake_registry(true);
        let economy = economic_routing_policy(&registry, RoutingMode::Economy).unwrap();
        assert_eq!(economy.mode(), RoutingMode::Economy);
        let decision = economy
            .route(&faktor_router::RouteRequest {
                phase: faktor_core::model::RouterPhase::Implement,
                required_capabilities: vec!["tools".into(), "streaming".into()],
                context_tokens: 8_000,
                estimated_output_tokens: 512,
                quality_floor: 60,
                task_budget_remaining_micro: 0,
                latency_preference_ms: None,
            })
            .expect("economy route over two capable models must decide");
        assert!(
            decision.provider == "alpha" || decision.provider == "beta",
            "the decision names a registered provider: {}",
            decision.provider
        );
        assert!(
            !decision.reasoning.is_empty(),
            "audit string rides the decision"
        );
        // Pinned to beta: the policy validates and returns beta — the free
        // evaluation may prefer alpha (lexicographic), the pin never loses.
        let pinned = economic_routing_policy(
            &registry,
            RoutingMode::Pinned {
                provider: "beta".into(),
                model: "default".into(),
            },
        )
        .unwrap();
        let decision = pinned
            .route(&faktor_router::RouteRequest {
                phase: faktor_core::model::RouterPhase::Implement,
                required_capabilities: vec!["tools".into(), "streaming".into()],
                context_tokens: 8_000,
                estimated_output_tokens: 512,
                quality_floor: 60,
                task_budget_remaining_micro: 0,
                latency_preference_ms: None,
            })
            .expect("pinned validation must pass for the registered pin");
        assert_eq!(decision.provider, "beta");
        assert_eq!(decision.model, "default");
        // (Budget-denial semantics for priced candidates are exercised at
        // the agent level where real ModelEconomics ride the router; the
        // daemon's unpriced default economics cost zero, so no request can
        // exceed them.)
    }

    #[test]
    fn pinned_candidates_collapse_to_the_pin_and_missing_pins_fail_the_build() {
        let mut registry = ProviderRegistry::new();
        registry.register(Arc::new(FakeProvider::with_script(
            "paid",
            ModelCapabilities {
                tools: true,
                ..Default::default()
            },
            vec![],
        )));
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
}
