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
//! 11. `verification` — the daemon's VerificationService (P0-5/6/9/10) over
//!     the ONE supervisor: the SAME instance the runtime gates completion
//!     with;
//! 12. `semantic` — the ONE semantic-provider registry (audits
//!     48-54/58/59/83) built from the strict `[semantic]` section over the
//!     daemon's supervisor + checked transport; the SAME `Arc` the agent and
//!     the server introspection endpoints hold;
//! 13. `learning` — the durable failure-learning prior handle (audit 68;
//!     `None` unless `[efficiency] failure_learning` is on) the runtime
//!     consults through `AgentDeps::context_prior`;
//! 14. `memory` — the daemon's ONE project-memory authority ([`DaemonMemory`])
//!     over the session store: per-session repositories are views of it;
//! 15. `tokenizers` — the ONE tokenizer registry (exact local backends with
//!     honest conservative fallback) shared by the daemon;
//! 16. `agent` — the reasoning runtime (drives sessions with commands);
//! 17. `orchestrator` — the OrchestratorRuntime (audits P0-20/21/23/61),
//!     the AUTHORITATIVE executor of multi-agent tasks and the durable
//!     child control surface;
//! 18. `shadows` — the daemon's shadow-mutation roots (P0-48 + wave-24);
//! 19. `tasks` — the TaskExecutor over the SAME orchestrator: the ONE
//!     native task-start authority of the daemon.
//!
//! Every authority is built EXACTLY ONCE per daemon lifetime, in the order
//! of [`DAEMON_CONSTRUCTION_ORDER`]; the construction happens in the
//! graph-construction region of `main.rs` (steps 1-2 in the daemon
//! entries, steps 3-20 inline in `build_daemon_core`). Serve/ACP/commands
//! never construct a supervisor, ledger, index or executor of their own —
//! they take references from this graph. The `cost_reservation`
//! route_decision_json column and the task row's max_cost_micro column
//! wait for the config surface that sets per-task money caps (provider-level
//! pricing tables and the per-session cap plumbing).

use std::sync::Arc;

use crate::evidence::RepoEvidence;
use faktor_agent::AgentRuntime;
use faktor_core::model::{unix_now_ms, ModelDescriptor, ModelSource, RoutingMode};
use faktor_index::IndexService;
use faktor_orchestrator::runtime::shadow::ShadowRoots;
use faktor_orchestrator::runtime::task_executor::TaskExecutor;
use faktor_orchestrator::runtime::OrchestratorRuntime;
use faktor_provider::catalog::{admissible_effective, ModelCatalogEntry, Provenance};
use faktor_provider::egress::HttpTransport;
use faktor_provider::ProviderRegistry;
use faktor_server::permission::ChannelPermissionRequester;
use faktor_session::{DurableBudgetLedger, SessionManager};
use faktor_terminal::ProcessSupervisor;

/// The ONE construction order of the daemon (audit 12/17): the canonical
/// marker list. Steps 1-2 run in the daemon entries of `main.rs`
/// (`build_daemon` / `build_daemon_with_mcp_inner`); steps 3-20 are inline
/// in `main.rs::build_daemon_core` in exactly this order; step 21 consumes
/// the graph (serve/ACP/commands) and constructs nothing of its own. The
/// tests verify the builder text against this list — a component inserted
/// out of order, or a second construction of any authority anywhere else,
/// is a compile-time-red test, never a review nit.
#[allow(dead_code)] // wave B8: consumed by the construction-order certification tests
pub(crate) const DAEMON_CONSTRUCTION_ORDER: [&str; 21] = [
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
    "semantic",     // 13. semantic-provider registry
    "learning",     // 14. failure-learning prior
    "memory",       // 15. project-memory authority
    "tokenizers",   // 16. tokenizer registry
    "agent",        // 17. AgentRuntime
    "orchestrator", // 18. OrchestratorRuntime
    "shadows",      // 19. ShadowRoots
    "tasks",        // 20. TaskExecutor
    "server",       // 21. ServerDeps/ACP/commands (consume only)
];

/// The daemon's ONE project-memory authority (audits round: graph
/// absorption): the session store every typed memory view is built over.
/// Per-session repositories ([`faktor_memory::StoreRepository`],
/// [`faktor_memory::SessionMemory`]) are VIEWS of this handle; the daemon
/// opens no second store for memory.
pub struct DaemonMemory {
    store: Arc<faktor_store::Store>,
}

impl DaemonMemory {
    pub fn new(store: Arc<faktor_store::Store>) -> Arc<Self> {
        Arc::new(Self { store })
    }

    /// The authority's store (the SAME store the session manager serves).
    pub fn store(&self) -> &Arc<faktor_store::Store> {
        &self.store
    }

    /// The per-session typed repository view over this authority.
    pub fn repository_for(
        &self,
        session: faktor_core::id::SessionId,
    ) -> faktor_memory::StoreRepository {
        faktor_memory::StoreRepository::new(self.store.clone(), session)
    }
}

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
    pub repo_evidence: Arc<RepoEvidence>,
    /// 11. The per-workspace instruction resolver (P0-32).
    pub instructions: Arc<faktor_instructions::InstructionResolver>,
    /// 12. The daemon's VerificationService (P0-5/6/9/10): the SAME
    ///     instance the runtime gates completion with — the graph holds it so
    ///     no consumer can build a second verifier over a second executor.
    pub verification: Arc<faktor_agent::VerificationService>,
    /// 13. The ONE semantic-provider registry (audits 48-54/58/59/83): built
    ///     once from the strict `[semantic]` section over the daemon's
    ///     supervisor + checked transport; handed to the agent and the
    ///     server surface as the SAME `Arc`.
    pub semantic: Arc<faktor_semantic::SemanticProviderRegistry>,
    /// 14. The failure-learning prior handle (audit 68): `Some` ONLY when
    ///     `[efficiency] failure_learning` is on; the SAME `Arc` the agent
    ///     consults through `AgentDeps::context_prior`.
    pub learning: Option<Arc<dyn faktor_context::information::FailurePrior + Send + Sync>>,
    /// 15. The daemon's ONE project-memory authority (see [`DaemonMemory`]).
    pub memory: Arc<DaemonMemory>,
    /// 16. The ONE tokenizer registry (exact local backends, honest
    ///     conservative fallback for unregistered identities) shared by the
    ///     daemon's context planning surface.
    pub tokenizers: Arc<faktor_context::TokenizerRegistry>,
    /// 17. The reasoning runtime (drives sessions with commands).
    pub agent: Arc<AgentRuntime>,
    /// THE durable evidence authority (schema v21): the ONE evidence store
    /// of record. It is the SAME allocation the runtime's ContextCompiler
    /// and archiver hold (`agent.evidence_authority()`), and the native
    /// server receives `graph.evidence.clone()` — no second authority is
    /// ever constructed, so the runtime and the server can never disagree
    /// about ids, scope or backing.
    pub evidence: Arc<faktor_context::compiler::DurableEvidenceAuthority>,
    /// 18. The orchestration runtime (audits P0-20/21/23/61): the
    ///     AUTHORITATIVE executor of multi-agent tasks and the durable control
    ///     surface the native `/agents/{child}/...` endpoints drive.
    pub orchestrator: Arc<OrchestratorRuntime>,
    /// 19. The shadow-mutation roots (P0-48 + wave-24): the executor's
    ///     shadow service; its Drop removes every shadow on graceful daemon
    ///     teardown, reconcile() at boot is the deterministic crash recovery.
    pub shadows: Arc<ShadowRoots>,
    /// 20. The TaskExecutor over [`DaemonGraph::orchestrator`]: the ONE
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

    /// Test-only clone of the graph's ONE durable evidence authority. The
    /// production wiring reads [`DaemonGraph::evidence`] directly; there is
    /// deliberately no constructing accessor (a per-call constructor would
    /// mint a parallel authority over the same rows).
    #[cfg(test)]
    pub(crate) fn evidence_authority(
        &self,
    ) -> Arc<faktor_context::compiler::DurableEvidenceAuthority> {
        self.evidence.clone()
    }
}

/// The additive `[semantic]` configuration section (audits 48-54/58/79):
/// strictly additive and EMPTY by default. It configures the Faktor-side
/// semantic-provider registry surface — an absent/empty section builds the
/// fallback-only registry, so ordinary operation NEVER requires a provider
/// (every runtime consult is optional and provider absence is byte-identical
/// parity).
///
/// A host may configure EXTERNAL providers as a bounded, strictly parsed
/// list: every entry is either a supervised `Process` child (typed
/// Content-Length framing on stdin/stdout, sanitized environment, explicit
/// deadline) or an `Http` endpoint reached through the daemon's checked
/// egress transport. Provider entries are preference-ordered; every response
/// is schema/identity/workspace/snapshot/payload validated by the semantic
/// crate, and absence or failure degrades to the generic fallback unless the
/// call carries `require_provider`.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct SemanticCfg {
    /// Bound on one provider response payload (bytes); absent = the semantic
    /// crate's default.
    pub max_payload_bytes: Option<usize>,
    /// Bound on provider-reported entity refs per response; absent = the
    /// semantic crate's default.
    pub max_entity_refs: Option<usize>,
    /// Strictly configured external provider entries, in preference order.
    /// Empty = fallback-only (ordinary operation never needs a provider).
    pub providers: Vec<faktor_semantic::SemanticProviderConfig>,
}

impl SemanticCfg {
    /// Strict validation of the whole additive section: positive response
    /// caps, unique provider ids, and every external provider entry bounded
    /// and well-formed (command/args/endpoint/timeout/auth-env). A hostile
    /// section is refused before any process or socket exists.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_payload_bytes == Some(0) {
            return Err("max_payload_bytes must be positive".to_string());
        }
        let mut seen = std::collections::HashSet::new();
        for provider in &self.providers {
            provider
                .validate()
                .map_err(|e| format!("semantic provider {}: {e}", provider.id()))?;
            if !seen.insert(provider.id().clone()) {
                return Err(format!("duplicate semantic provider id {}", provider.id()));
            }
        }
        Ok(())
    }
}

/// Build the daemon's ONE semantic-provider registry from the additive
/// `[semantic]` section: fallback-only by default, with the section's
/// response caps applied and every configured external provider built over
/// the daemon's OWN authorities — the ONE [`ProcessSupervisor`] for process
/// children and the checked egress [`HttpTransport`] for endpoints. A
/// malformed section or an unbuildable provider refuses the daemon at boot
/// (never a silent half-configured registry); a valid section with no
/// providers stays fallback-only, so ordinary operation never requires one.
pub fn semantic_registry(
    cfg: &SemanticCfg,
    supervisor: &Arc<ProcessSupervisor>,
    transport: &Arc<dyn HttpTransport>,
) -> Result<Arc<faktor_semantic::SemanticProviderRegistry>, String> {
    cfg.validate()?;
    let mut caps = faktor_semantic::SemanticResponseCaps::default();
    if let Some(max_payload_bytes) = cfg.max_payload_bytes {
        caps.max_payload_bytes = max_payload_bytes;
    }
    if let Some(max_entity_refs) = cfg.max_entity_refs {
        caps.max_entity_refs = max_entity_refs;
    }
    let env = faktor_semantic::SemanticClientEnv {
        supervisor: supervisor.clone(),
        transport: transport.clone(),
        caps,
    };
    let mut registry = faktor_semantic::SemanticProviderRegistry::new(
        faktor_semantic::GenericSemanticFallback::default(),
    )
    .with_response_caps(caps);
    for provider in &cfg.providers {
        let built = provider
            .build(&env)
            .map_err(|e| format!("semantic provider {}: {e}", provider.id()))?;
        registry.register(built);
    }
    Ok(Arc::new(registry))
}

/// One registered catalog row as the router's PRICED unit: its
/// non-monetary descriptor plus the row's catalog-resolved [`faktor_core::model::PricingState`].
/// This is the ONLY unit the daemon router consumes; qualification, scoring
/// and budget admission all evaluate the candidate's exact cost estimate
/// (per-million quote math), never a per-token projection.
fn route_candidate_for(
    provider_id: &str,
    entry: &ModelCatalogEntry,
) -> faktor_router::RouteCandidate {
    faktor_router::RouteCandidate::new(descriptor_for(provider_id, entry), entry.pricing.clone())
}

/// A router candidate descriptor for one registered provider model, built
/// from the provider's REAL catalog row (audit P0-1 / wave-B item B +
/// pricing-path audit):
///
/// - the descriptor carries only the NON-MONETARY performance surface the
///   router's qualification/scoring reads: reliability priors and the
///   latency estimate from the row's `quality_prior` (documented built-in
///   Faktor routing priors when the endpoint has them, adapter-declared
///   priors when the adapter knows better);
/// - it carries NO price projection: every per-token price field stays
///   zero and is never consulted. Money reaches the router exclusively
///   through the [`faktor_router::RouteCandidate`]'s [`faktor_core::model::PricingState`] (an
///   exact per-million quote, a conservative ceiling, an authoritative
///   local zero, or the non-numeric Unknown), evaluated by the router's
///   exact quote math — never through a lossy per-token field;
/// - `source` records the row's provenance.
fn descriptor_for(provider_id: &str, entry: &ModelCatalogEntry) -> ModelDescriptor {
    let caps = &entry.capabilities;
    let qp = entry.quality_prior;
    let economics = faktor_core::model::ModelEconomics {
        tool_reliability: qp.tool_reliability,
        reasoning_reliability: qp.reasoning_reliability,
        coding_reliability: qp.coding_reliability,
        context_reliability: qp.context_reliability,
        availability: qp.availability,
        estimated_latency_ms: qp.estimated_latency_ms,
        ..Default::default()
    };
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

/// The exclusion/admission policy for candidate sets (audit P0-1/wave-B
/// item C — the admission matrix). At graph build there is NO hard cost cap
/// (caps are per-task and arrive at route time), so:
///
/// - Economy excludes every Unknown-priced row (cost minimization cannot
///   price it);
/// - Balanced excludes Unknown-priced rows by default (no
///   allow-unknown-in-balanced knob exists yet);
/// - MaximumQuality admits Unknown-priced rows (quality decides; spend
///   settles as a documented Unknown amount, and the router's priced
///   qualification excludes them again the moment a hard cap appears);
/// - Pinned admits the pin regardless (the pin, not economics, decides) —
///   the router's pinned qualification still fails an Unknown pin closed
///   under a hard cap.
///
/// A configured `pricing_ceiling_micro_usd_per_million_tokens` never
/// reaches this function as Unknown: the provider wrapper turns Unknown
/// rows into [`faktor_core::model::PricingState::ConservativeCeiling`] at exactly the ceiling
/// with provenance [`Provenance::Composite`].
///
/// Admission is judged on the EFFECTIVE price state at build time: a
/// `Known` row whose validity window already elapsed (and any `Stale` row)
/// is Unknown here — its old exact price never enters a candidate set as if
/// current — while a documented conservative ceiling keeps a row
/// admissible as a bound.
fn candidate_entry_ok(mode: &RoutingMode, entry: &ModelCatalogEntry) -> bool {
    let effective = entry.pricing.effective_at(unix_now_ms(), false);
    admissible_effective(mode, &effective, false, false)
}

/// Quality-authority guard (audit item: performance profiles): a Balanced
/// configuration must be able to route at its default band. Balanced never
/// routes below [`faktor_agent::EconomicRoutingPolicy::BALANCED_QUALITY_FLOOR`],
/// so a candidate set where no admitted candidate clears that floor can
/// only serve typed refusals — the graph fails the daemon build with an
/// explanatory error instead of booting a configuration whose every
/// Balanced route is dead. The check mirrors the routing floor metric: a
/// candidate clears when its coding-relevant mean (heavy phases) OR its
/// context reliability (cheap phases) sits at/above the floor.
fn ensure_balanced_candidate(candidates: &[faktor_router::RouteCandidate]) -> Result<(), String> {
    let floor = faktor_agent::EconomicRoutingPolicy::BALANCED_QUALITY_FLOOR;
    let clears = |c: &faktor_router::RouteCandidate| {
        let p = c.descriptor.performance();
        p.coding_reliability >= floor || p.context_reliability >= floor
    };
    if candidates.iter().any(clears) {
        return Ok(());
    }
    Err(format!(
        "routing_mode balanced requires at least one admitted candidate at/above the balanced \
         quality floor {floor} (coding-quality mean or context reliability); none of the {} \
         admitted candidate(s) clears it. Configure an official endpoint whose model has a \
         documented Faktor routing prior (or supplier pricing so a known model enters the set), \
         or switch routing_mode to economy",
        candidates.len()
    ))
}

/// The daemon's router candidate set: every admitted model of every
/// registered provider (bounded by the registry and the providers' own
/// `known_models()`), each built as a [`faktor_router::RouteCandidate`]
/// from its real catalog row.
///
/// Candidate-set policy by mode (audit P0-1/wave-B C — unknown price !=
/// zero, authority never inferred):
///
/// | pricing state | Economy / Balanced | MaximumQuality | Pinned |
/// |---|---|---|---|
/// | Known (exact prices) | included at its real price | included | validation through [`faktor_router::qualify_specific`] |
/// | ConservativeCeiling | included at the ceiling | included | as today |
/// | LocalZero (Ollama) | always included, zero cost | included | as today |
/// | Unknown, no price knowledge | **EXCLUDED** (never a fabricated 0 / 1-micro fallback) | **included** (no hard cost cap at build; quality decides, spend settles as documented Unknown) | pin included (the pin decides; its snapshot stays Unknown — never LocalZero) |
/// | Unknown + configured ceiling | included as ConservativeCeiling (applied by the provider wrapper) | as Economy | as today |
///
/// Every candidate carries its catalog-resolved [`faktor_core::model::PricingState`] directly,
/// so route decisions freeze the real authority and the exact per-million
/// quote; there is no pricing map and no per-token projection.
///
/// In Pinned mode the candidate set collapses to the pin itself: the
/// RouterService then VALIDATES the pin's capability/fit/budget/health
/// axes through [`faktor_router::qualify_specific`] and the pin always
/// wins when feasible — the router's free choice can never silently
/// substitute the configured pin (fail closed). A pin whose (provider,
/// model) is not among the registered models is a graph-build error (loud,
/// at boot — never a silent Economy).
///
/// The daemon wiring twin [`build_router_service_with_outcomes`] builds the
/// SAME candidates through
/// [`faktor_router::RouterService::with_route_candidates`] (or
/// [`faktor_router::RouterService::with_pinned_route_candidates`] in
/// Pinned mode) so the service carries the durable verified-outcome
/// registry; this plain variant is the test/embedded shape (no registry —
/// the default empty store keeps decisions byte-identical to a
/// registry-less service).
pub fn build_router_service_with_outcomes(
    providers: &ProviderRegistry,
    mode: &RoutingMode,
    outcomes: Arc<dyn faktor_router::OutcomeStore>,
) -> Result<Arc<faktor_router::RouterService>, String> {
    let candidates = router_candidates(providers, mode)?;
    let service = match mode {
        RoutingMode::Pinned { provider, model } => {
            faktor_router::RouterService::with_pinned_route_candidates(
                candidates,
                provider.clone(),
                model.clone(),
                outcomes,
            )
        }
        _ => faktor_router::RouterService::with_route_candidates(candidates, outcomes),
    };
    Ok(Arc::new(service))
}

/// The candidate set the router constructors consume (see
/// [`build_router_service_with_outcomes`] for the admission policy).
type RouterCandidates = Vec<faktor_router::RouteCandidate>;

/// Candidate build shared by both service constructors (see
/// [`build_router_service_with_outcomes`] for the admission policy).
fn router_candidates(
    providers: &ProviderRegistry,
    mode: &RoutingMode,
) -> Result<RouterCandidates, String> {
    let mut candidates: Vec<faktor_router::RouteCandidate> = Vec::new();
    match mode {
        // MaximumQuality and Balanced route over the SAME full registered
        // candidate set as Economy — the mode is policy-level semantics
        // (top-quality tier / balanced quality band), not a candidate
        // filter at build time. Economy and Balanced exclude Unknown-priced
        // rows (an unknown price cannot be costed or budgeted);
        // MaximumQuality admits them because no hard cost cap exists at
        // build and quality decides (the admission matrix, exactly).
        RoutingMode::Economy | RoutingMode::MaximumQuality | RoutingMode::Balanced => {
            for id in providers.ids() {
                let Some(p) = providers.get(&id) else {
                    continue;
                };
                for model in p.known_models() {
                    let entry = p.catalog_entry(&model);
                    if candidate_entry_ok(mode, &entry) {
                        candidates.push(route_candidate_for(&id, &entry));
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
            // capability/fit/budget axes through the router's pinned
            // qualification; its decision snapshot stays the honest
            // Unknown (never a fabricated LocalZero).
            let entry = p.catalog_entry(model);
            candidates.push(route_candidate_for(provider, &entry));
        }
    }
    if matches!(mode, RoutingMode::Balanced) {
        ensure_balanced_candidate(&candidates)?;
    }
    Ok(candidates)
}

/// The daemon's economic routing policy over the candidates of
/// [`build_router_service_with_outcomes`]: the policy's RouterService is
/// built via the PRICED `with_route_candidates` /
/// `with_pinned_route_candidates` constructors, so `record_call_outcome`
/// verified samples (runtime deterministic-gate sites) land in the OUTCOME
/// STORE this call wires — the same registry every route consult reads. The
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
        MicroUsdPerMillionTokens, ModelCapabilities, ModelEconomics, PriceAuthority, PriceQuote,
        PricingSnapshot,
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
        quality_prior: QualityPrior,
    }

    impl PricedTestProvider {
        /// A Known row at `input`/`output` WHOLE DOLLARS per million tokens,
        /// carried as the exact per-million quote (never a per-token
        /// projection).
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

        /// A provider declaring an explicit performance prior (quality
        /// authority: adapter-declared `ProviderCatalog` knowledge).
        fn with_prior(
            id: &str,
            model: &str,
            caps: ModelCapabilities,
            pricing: PricingState,
            quality_prior: QualityPrior,
        ) -> Arc<dyn Provider> {
            Arc::new(Self {
                id: id.into(),
                caps,
                models: vec![model.into()],
                pricing,
                quality_prior,
            })
        }

        fn with_pricing(
            id: &str,
            model: &str,
            caps: ModelCapabilities,
            pricing: PricingState,
        ) -> Arc<dyn Provider> {
            Self::with_prior(id, model, caps, pricing, QualityPrior::default())
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
                quality_prior: self.quality_prior,
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
        assert_eq!(svc.priced.len(), 1, "one priced candidate");
        let pc = &svc.priced[0];
        assert_eq!(pc.descriptor.provider, "local-a");
        assert_eq!(pc.descriptor.model, "model-x");
        assert!(
            pc.descriptor.tools,
            "capabilities come from the live provider"
        );
        assert_eq!(pc.descriptor.context, 64_000);
        match &pc.pricing {
            PricingState::Known(snap) => {
                assert_eq!(snap.authority, PriceAuthority::Exact);
                let q = snap.quote.expect("Known quotes");
                assert_eq!(q.input, MicroUsdPerMillionTokens(15_000_000));
                assert_eq!(q.output, MicroUsdPerMillionTokens(60_000_000));
            }
            other => panic!("Known row must keep its exact state, got {other:?}"),
        }
        assert!(
            pc.descriptor.economics.input_price_per_mtok.is_zero(),
            "the descriptor carries NO per-token price projection"
        );
        assert_eq!(pc.descriptor.source, ModelSource::ProviderCatalog);
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
        let economy = empty_store_service(&registry, &RoutingMode::Economy).unwrap();
        assert!(
            economy.router.candidates.is_empty(),
            "Economy must exclude every Unknown-priced entry"
        );
        // Balanced excludes Unknown rows too, and with NO admitted candidate
        // at its default band the graph build fails with the explanatory
        // quality-authority error (never a booted always-refusing config).
        let err = match empty_store_service(&registry, &RoutingMode::Balanced) {
            Ok(_) => panic!("Balanced over Unknown-only rows must fail at build"),
            Err(e) => e,
        };
        assert!(err.contains("balanced"), "{err}");
        assert!(err.contains("88"), "{err}");
        // MaximumQuality admits Unknown-priced entries while no hard cost
        // cap exists (the admission matrix): quality decides and the spend
        // settles as a documented Unknown amount. The moment a hard cap
        // appears at route time, the priced qualification excludes them.
        let mq = empty_store_service(&registry, &RoutingMode::MaximumQuality).unwrap();
        assert_eq!(mq.priced.len(), 2, "MaximumQuality admits Unknown rows");
        for pc in &mq.priced {
            assert_eq!(pc.pricing, PricingState::Unknown);
            assert_eq!(pc.pricing.authority(), PriceAuthority::Unknown);
        }
        let hard_capped = faktor_router::RouteRequest {
            quality_floor: 50,
            task_budget_remaining_micro: 1_000_000,
            ..Default::default()
        };
        assert!(
            mq.route(&hard_capped, &[]).is_err(),
            "Unknown under a hard cap fails closed through the priced path"
        );
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
    fn expired_exact_rows_leave_economy_candidate_sets_and_fail_closed_under_caps() {
        // A Known row whose validity window already elapsed must not enter
        // Economy at its old price: at build it is effectively Unknown
        // (excluded from Economy), admitted by MaximumQuality without a cap
        // as a documented Unknown spend, and refused the moment a hard cap
        // appears.
        let expired = PricedTestProvider::with_pricing(
            "corp",
            "aged",
            ModelCapabilities {
                tools: true,
                streaming: true,
                context: 128_000,
                ..Default::default()
            },
            PricingState::Known(
                PricingSnapshot::exact(
                    PriceQuote {
                        input: MicroUsdPerMillionTokens::from_dollars_per_million(2),
                        output: MicroUsdPerMillionTokens::from_dollars_per_million(8),
                        ..PriceQuote::ZERO
                    },
                    CATALOG_FIRST_EPOCH,
                    "row".into(),
                )
                .with_valid_until(1),
            ),
        );
        let mut registry = ProviderRegistry::new();
        registry.try_register(expired).unwrap();
        let economy = empty_store_service(&registry, &RoutingMode::Economy).unwrap();
        assert!(
            economy.priced.is_empty(),
            "expired exact must not enter Economy at its old price"
        );
        let mq = empty_store_service(&registry, &RoutingMode::MaximumQuality).unwrap();
        assert_eq!(mq.priced.len(), 1);
        let req = faktor_router::RouteRequest {
            required_capabilities: vec!["tools".into()],
            context_tokens: 1_000,
            estimated_output_tokens: 100,
            quality_floor: 50,
            ..Default::default()
        };
        let d = mq.route(&req, &[]).unwrap();
        assert_eq!(
            d.pricing_snapshot.expect("snapshot").authority,
            PriceAuthority::Unknown,
            "the decision must freeze the downgraded Unknown snapshot"
        );
        let hard = faktor_router::RouteRequest {
            task_budget_remaining_micro: 10_000_000,
            ..req
        };
        assert!(
            mq.route(&hard, &[]).is_err(),
            "expired exact fails closed under a hard cap"
        );
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
        // ...and the economy candidate set includes it at the ceiling as a
        // ConservativeCeiling PRICED candidate (42_000_000 microUSD/M on
        // every line — a bound, never free, never projected per token).
        let svc = empty_store_service(&registry, &RoutingMode::Economy).unwrap();
        assert_eq!(svc.priced.len(), 1);
        let pc = &svc.priced[0];
        match &pc.pricing {
            PricingState::ConservativeCeiling(snap) => {
                assert_eq!(snap.authority, PriceAuthority::ConservativeCeiling);
                let q = snap.quote.expect("ceiling quotes");
                for line in [q.input, q.output, q.cache_read, q.cache_write] {
                    assert_eq!(line, MicroUsdPerMillionTokens(42_000_000));
                }
            }
            other => panic!("ceiling candidate must carry its bound, got {other:?}"),
        }
        assert!(
            pc.descriptor.economics.input_price_per_mtok.is_zero(),
            "no per-token projection exists on the routing path"
        );
        assert_eq!(
            pc.descriptor.source,
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
        assert_eq!(svc.priced.len(), 2);
        let local = svc
            .priced
            .iter()
            .find(|c| c.descriptor.provider == "ollama")
            .unwrap();
        assert!(
            local.pricing.is_local_zero(),
            "LocalZero stays the router's explicit zero-cost state"
        );
        assert_eq!(
            local.pricing.authority(),
            PriceAuthority::LocalZero,
            "LocalZero is authority, never inferred from zeros"
        );
        assert_eq!(local.descriptor.economics.estimated_latency_ms, 1000);
        let paid = svc
            .priced
            .iter()
            .find(|c| c.descriptor.provider == "openai")
            .unwrap();
        assert_eq!(paid.pricing.authority(), PriceAuthority::Exact);
        assert!(
            paid.descriptor.economics.input_price_per_mtok.is_zero(),
            "the descriptor never carries the per-token projection"
        );
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
                    ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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
                // Pinned validation only: the descriptor carries the
                // conservative non-monetary performance defaults and NO
                // price projection (money rides the candidate's state).
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
        assert_eq!(svc.priced.len(), 1);
        assert!(
            svc.priced[0].pricing.is_local_zero(),
            "the priced unit keeps the authoritative LocalZero state"
        );
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

    /// The high-quality cap shape used by the MaximumQuality/Balanced
    /// tests: tools+streaming, 256k context.
    fn full_caps() -> ModelCapabilities {
        ModelCapabilities {
            tools: true,
            streaming: true,
            context: 256_000,
            ..Default::default()
        }
    }

    fn prior(coding: u8, context_rel: u8, latency_ms: u64) -> QualityPrior {
        QualityPrior {
            tool_reliability: coding,
            reasoning_reliability: coding,
            coding_reliability: coding,
            context_reliability: context_rel,
            availability: 100,
            estimated_latency_ms: latency_ms,
        }
    }

    #[test]
    fn maximum_quality_routes_to_an_unknown_high_quality_model_without_a_cap() {
        // MaximumQuality admits Unknown-priced entries at build (no hard
        // cap): the 95-quality unpriced frontier wins its top tier; under a
        // hard cap it is excluded (never treated as free) and the priced
        // affordable tier serves.
        let mut registry = ProviderRegistry::new();
        registry
            .try_register(PricedTestProvider::with_prior(
                "frontier",
                "fm",
                full_caps(),
                PricingState::Unknown,
                prior(95, 95, 900),
            ))
            .unwrap();
        registry
            .try_register(PricedTestProvider::with_prior(
                "budget",
                "bm",
                full_caps(),
                PricingState::Known(PricingSnapshot::exact(
                    PriceQuote {
                        input: MicroUsdPerMillionTokens::from_dollars_per_million(1),
                        output: MicroUsdPerMillionTokens::from_dollars_per_million(3),
                        ..PriceQuote::ZERO
                    },
                    CATALOG_FIRST_EPOCH,
                    "row".into(),
                )),
                prior(80, 80, 500),
            ))
            .unwrap();
        let policy = empty_store_policy(&registry, RoutingMode::MaximumQuality).unwrap();
        let req = faktor_router::RouteRequest {
            phase: faktor_core::model::RouterPhase::Implement,
            required_capabilities: vec!["tools".into(), "streaming".into()],
            context_tokens: 4_000,
            estimated_output_tokens: 500,
            quality_floor: 60,
            task_budget_remaining_micro: 0,
            latency_preference_ms: None,
            ..Default::default()
        };
        let d = policy
            .route(&req)
            .expect("the unpriced top-quality model must not be excluded without a cap");
        assert_eq!((d.provider.as_str(), d.model.as_str()), ("frontier", "fm"));
        let snap = d.pricing_snapshot.expect("decisions carry snapshots");
        assert_eq!(snap.authority, PriceAuthority::Unknown);
        assert_eq!(snap.settle_cost(1_000, 0, 0, 100), None);

        // Hard cap: the unknown frontier fails closed; the affordable
        // priced tier clears the requested floor and serves.
        let mut capped = req;
        capped.task_budget_remaining_micro = 5_000_000;
        let d = policy
            .route(&capped)
            .expect("an affordable priced tier serves");
        assert_eq!(
            (d.provider.as_str(), d.model.as_str()),
            ("budget", "bm"),
            "a hard cap must exclude the unpriced model: {}",
            d.reasoning
        );
        assert_eq!(
            d.pricing_snapshot.expect("snapshot").authority,
            PriceAuthority::Exact
        );
    }

    #[test]
    fn balanced_config_without_a_candidate_above_the_default_floor_fails_with_an_explanation() {
        // A custom endpoint with only the conservative generic 50 prior
        // cannot serve Balanced's 88 band: the graph build fails LOUDLY
        // with an explanatory error instead of booting a configuration
        // whose every Balanced route is a typed refusal.
        let mut registry = ProviderRegistry::new();
        registry
            .try_register(PricedTestProvider::known(
                "corp-proxy",
                "m",
                full_caps(),
                1,
                3,
            ))
            .unwrap();
        let err = match empty_store_service(&registry, &RoutingMode::Balanced) {
            Ok(_) => panic!("Balanced over a 50-prior-only set must fail at build"),
            Err(e) => e,
        };
        assert!(err.contains("balanced"), "{err}");
        assert!(err.contains("88"), "the error names the floor: {err}");

        // The same shape WITH a documented high prior builds and routes at
        // the band (the built-in Faktor routing priors make a fresh default
        // Balanced config viable).
        let mut viable = ProviderRegistry::new();
        viable
            .try_register(PricedTestProvider::with_prior(
                "officialish",
                "m",
                full_caps(),
                PricingState::Known(PricingSnapshot::exact(
                    PriceQuote {
                        input: MicroUsdPerMillionTokens::from_dollars_per_million(1),
                        output: MicroUsdPerMillionTokens::from_dollars_per_million(3),
                        ..PriceQuote::ZERO
                    },
                    CATALOG_FIRST_EPOCH,
                    "row".into(),
                )),
                prior(92, 92, 700),
            ))
            .unwrap();
        let policy = empty_store_policy(&viable, RoutingMode::Balanced).unwrap();
        let d = policy
            .route(&faktor_router::RouteRequest {
                phase: faktor_core::model::RouterPhase::Implement,
                required_capabilities: vec!["tools".into(), "streaming".into()],
                context_tokens: 1_000,
                estimated_output_tokens: 100,
                quality_floor: 10,
                ..Default::default()
            })
            .expect("a documented prior above the band must serve Balanced");
        assert_eq!(d.provider, "officialish");
    }

    #[test]
    fn pin_lacks_tools_is_no_capable_model() {
        // The pinned policy runs ONLY the router's pinned qualification: a
        // pin missing a required capability is a typed NoCapableModel
        // refusal — never a silent substitution, never a lowered ask.
        let mut registry = ProviderRegistry::new();
        registry
            .try_register(Arc::new(FakeProvider::with_script(
                "plain",
                ModelCapabilities {
                    tools: false,
                    streaming: true,
                    context: 128_000,
                    ..Default::default()
                },
                vec![],
            )))
            .unwrap();
        let policy = empty_store_policy(
            &registry,
            RoutingMode::Pinned {
                provider: "plain".into(),
                model: "default".into(),
            },
        )
        .unwrap();
        let req = faktor_router::RouteRequest {
            required_capabilities: vec!["tools".into()],
            context_tokens: 100,
            estimated_output_tokens: 10,
            quality_floor: 50,
            ..Default::default()
        };
        assert_eq!(
            policy.route(&req),
            Err(faktor_agent::RouteFailure::NoCapableModel)
        );
    }

    #[test]
    fn daemon_graph_attaches_the_store_backed_outcome_registry_to_routing() {
        // Spy assertion (audit items 13/14/L wiring): the graph-built
        // policy's RouterService is constructed via the priced
        // `with_route_candidates` path over a StoreOutcomeStore on the
        // DAEMON store — a verified sample recorded through the policy's
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
        // Graph absorption: the semantic registry, verification service,
        // learning handle, memory authority and tokenizer registry are the
        // SAME instances the agent (and server) consumers see.
        assert!(
            Arc::ptr_eq(graph.agent.semantic_registry(), &graph.semantic),
            "the agent must consult the graph's ONE semantic registry"
        );
        assert!(Arc::ptr_eq(
            &graph.agent.deps().verification,
            &graph.verification
        ));
        assert!(
            graph.learning.is_some(),
            "failure_learning defaults ON in production: the prior handle is installed"
        );
        assert!(
            graph.agent.deps().context_prior.is_some(),
            "the agent's prior handle is exactly the graph's (Some here)"
        );
        assert!(
            Arc::ptr_eq(graph.memory.store(), &graph.session.store()),
            "the memory authority is a view of the daemon's ONE store"
        );
        assert!(
            graph
                .tokenizers
                .count(faktor_provider::TokenizerId::O200K_BASE, "let x = 1;")
                .count
                >= 1,
            "the tokenizer registry answers with an exact or bounded count"
        );
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
            .block_on(graph.repo_evidence.evidence_for(
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
            .block_on(graph.repo_evidence.evidence_for(
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
        let view = graph2
            .budgets
            .session_budget_view(sid, TaskId::new(9))
            .expect("durable budget view");
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

    /// The graph's durable evidence authority is the v21 evidence store of
    /// record over the graph's OWN store root: an id inserted before a
    /// daemon restart still resolves after it, a known backing id never
    /// crosses a session boundary, and the runtime's authority shares the
    /// same backing root.
    #[test]
    fn graph_evidence_authority_is_the_durable_store_of_record() {
        use faktor_context::compiler::{
            DurableEvidenceAuthority, EvidenceAccessContext, EvidenceKind, ProvenanceSource,
        };

        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("data");
        let graph = crate::build_daemon(&data, None).unwrap();
        let ws = graph.session.create_workspace("/w").unwrap();
        let owner = graph
            .session
            .create_session(ws, "owner", "p", "m")
            .unwrap()
            .id();
        let intruder = graph
            .session
            .create_session(ws, "intruder", "p", "m")
            .unwrap()
            .id();

        let authority = graph.evidence_authority();
        // ONE authority, pointer-identical everywhere: the graph field, the
        // runtime's compiler/archiver authority and a boxed native-server
        // handle (exactly what `with_evidence_store` wraps) are the SAME
        // allocation. A per-call constructor would mint a parallel authority
        // over the same rows and fail here.
        assert!(
            Arc::ptr_eq(&graph.evidence, &authority),
            "the graph must own the ONE evidence authority"
        );
        assert!(
            Arc::ptr_eq(graph.agent.evidence_authority(), &graph.evidence),
            "the runtime's authority must be the graph's ONE allocation"
        );
        let handle: Box<dyn faktor_evidence::store::EvidenceStore + Send + Sync> =
            Box::new(graph.evidence.clone());
        assert_eq!(
            handle.authority_ptr(),
            Some(Arc::as_ptr(&graph.evidence) as usize),
            "the boxed server handle must wrap the graph's ONE allocation"
        );
        // The runtime's authority roots at the SAME backing directory.
        assert_eq!(
            graph.agent.evidence_authority().backing_root(),
            authority.backing_root(),
            "the runtime and the graph must share one evidence identity space"
        );
        let stored = authority
            .archive_text(
                owner,
                ws,
                None,
                EvidenceKind::GenericText,
                Some("src/x.rs"),
                ProvenanceSource::Repository,
                "durable graph evidence body",
                faktor_context::compiler::MAX_COMPILED_BODY_BYTES,
            )
            .unwrap();
        // Dedupe: re-archiving identical bytes returns the SAME envelope.
        let again = authority
            .archive_text(
                owner,
                ws,
                None,
                EvidenceKind::GenericText,
                Some("src/x.rs"),
                ProvenanceSource::Repository,
                "durable graph evidence body",
                faktor_context::compiler::MAX_COMPILED_BODY_BYTES,
            )
            .unwrap();
        assert_eq!(again.id, stored.id, "identical output is one evidence item");
        // Scope: the intruder's session never sees it, by id.
        let owner_ctx = EvidenceAccessContext::new(owner.raw(), ws.raw(), None);
        let intruder_ctx = EvidenceAccessContext::new(intruder.raw(), ws.raw(), None);
        assert_eq!(
            authority
                .get_scoped(stored.id, &owner_ctx)
                .unwrap()
                .envelope
                .compact
                .body,
            "durable graph evidence body"
        );
        assert!(
            authority.get_scoped(stored.id, &intruder_ctx).is_err(),
            "a known evidence id must never cross sessions"
        );
        drop(authority);
        drop(graph);

        // Daemon restart over the same data dir: ids and bodies resolve.
        let graph2 = crate::build_daemon(&data, None).unwrap();
        let authority2 = graph2.evidence_authority();
        assert_eq!(
            authority2
                .get_scoped(stored.id, &owner_ctx)
                .unwrap()
                .envelope
                .compact
                .body,
            "durable graph evidence body",
            "the evidence id must survive a daemon restart"
        );
        let fresh = authority2
            .archive_text(
                owner,
                ws,
                None,
                EvidenceKind::GenericText,
                Some("src/y.rs"),
                ProvenanceSource::Repository,
                "a newer body",
                faktor_context::compiler::MAX_COMPILED_BODY_BYTES,
            )
            .unwrap();
        assert!(
            fresh.id > stored.id,
            "a restarted daemon must never reissue {}",
            stored.id
        );
        let _ = DurableEvidenceAuthority::for_store(graph2.session.store(), 1024);
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
        ("semantic", "graph::semantic_registry("),
        ("learning", "daemon_context_prior("),
        ("memory", "DaemonMemory::new("),
        ("tokenizers", "TokenizerRegistry::with_builtin_backends("),
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
                | "evidence" | "instructions" | "verification" | "semantic" | "learning"
                | "memory" | "tokenizers" | "agent" | "orchestrator" | "shadows" | "tasks" => {
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
            "SemanticProviderRegistry::new",
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
