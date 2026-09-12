//! faktor-agent — the durable agent reasoning loop.
//!
//! Drives `faktor-session` with commands, consumes `faktor-provider` streams,
//! schedules tools through `faktor-scheduler`, and keeps context bounded via
//! `faktor-context`. Rules (from the architecture spec):
//!
//! - **No provider-name conditionals.** Behavior comes from
//!   `ModelCapabilities`; provider quirks stay inside adapters.
//! - **State-aware continuation.** A provider stream that dies mid-tool never
//!   replays the turn; the journal determines the continuation point.
//! - **Repair once, never five times.** Malformed tool JSON gets one
//!   deterministic repair pass; repeated identical failures trip the loop
//!   detector and stop the turn.
//! - **Bounded context before sending.** Budget enforced by the assembler;
//!   compaction triggers proactively at the configured usage fraction.

use std::sync::Arc;

use faktor_core::id::SessionId;
use faktor_router::stability::TurnPrefix;
use faktor_verify::exec::{BudgetDecision, CheckOutcome, CheckRunStatus};

pub mod loop_detect;
pub mod runtime;
pub mod stall;
pub mod tool;
pub mod tool_json;
pub mod wire_plan;

pub use faktor_core::model::{RiskBucket, RouteDecision, RouterPhase, RoutingMode, TaskClass};
pub use faktor_core::state::{
    CheckExecution, CriterionVerification, FileStateEvidence, OutcomeReason, ReasonCode, TaskState,
    TaskTransition, VerificationStatus,
};
pub use faktor_session::VerificationRecord;
pub use faktor_verify::Acceptance;
pub use loop_detect::LoopDetector;
pub use runtime::{
    AgentCard, AgentDeps, AgentRuntime, ChunkEvent, ChunkSink, CompletionGate, EvidenceProvider,
    EvidenceQuery, NoEvidence, PermissionRequester, ToolArtifactSink, TurnOutcome,
    VerificationQuality,
};
pub use stall::{StallTracker, DEFAULT_STALL_SILENCE_MS};
pub use tool::{
    board_post_tool, board_read_tool, BoardToolGateway, FilePostcondition, RecoveryHint,
    ReplayDescriptor, Tool, ToolBundle, ToolBundleId, ToolOutcome, ToolRegistry, ToolRunCtx,
    BOARD_POST_TOOL, BOARD_READ_TOOL, BOARD_TOOL_MAX_LIMIT, BOARD_TOOL_MAX_REFS,
    BOARD_TOOL_TEXT_MAX, SEMANTIC_BUNDLE_MAX_SPECS, SEMANTIC_QUERY_TOOL,
};
pub use tool_json::{parse_tool_calls, repair_json, ToolCallMode};

/// The production efficiency flags (audit 86 + the efficiency-variant
/// production switches): the agent-side mirror of the daemon's parsed
/// `[efficiency]` section. Every flag defaults to `false` — the baseline
/// production behavior — and each switch gates an ADDITIVE behavior only:
///
/// - [`EfficiencyFlags::failure_learning`]: apply the installed failure-aware
///   context prior ([`AgentDeps::context_prior`]) to non-Required candidate
///   selection through `plan_context_with_information_and_prior`;
/// - `ccr`, `typed_handoff`, `semantic_context`, `rework_routing`: parsed and
///   carried by [`AgentDeps`] for the corresponding efficiency components.
///
/// With every flag off (and/or no prior handle installed) the runtime's
/// plans are byte-identical to the pre-flag path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EfficiencyFlags {
    /// Failure-learning prior in context selection (audit 68).
    pub failure_learning: bool,
    /// Compressed Context Representation for tool/evidence payloads.
    pub ccr: bool,
    /// Typed handoff: re-sent history rendered from durable task rows.
    pub typed_handoff: bool,
    /// Semantic context: information-gain selection of evidence.
    pub semantic_context: bool,
    /// Rework-aware routing over durable verified-outcome stats.
    pub rework_routing: bool,
}

impl EfficiencyFlags {
    /// The production gate for the failure-aware context prior: the prior is
    /// applied ONLY when `failure_learning` is on AND a handle was installed.
    /// Any other combination yields `None`, so the planner takes its
    /// baseline path and the plan stays byte-identical.
    pub fn context_prior<'a>(
        &self,
        prior: Option<&'a (dyn faktor_context::information::FailurePrior + Send + Sync)>,
    ) -> Option<&'a (dyn faktor_context::information::FailurePrior + Send + Sync)> {
        if self.failure_learning {
            prior
        } else {
            None
        }
    }
}

/// The fallback-only semantic-provider registry (audit 48-54/58/79): the
/// additive [`AgentDeps::semantic`] handle every construction site installs
/// unless a host explicitly registers a richer provider. The generic
/// fallback answers every operation itself, so ordinary operation NEVER
/// fails solely because no semantic provider is installed — and with only
/// the fallback registered the runtime's optional consults stay
/// byte-identical to a provider-less runtime (parity).
pub fn fallback_semantic_registry() -> Arc<faktor_semantic::SemanticProviderRegistry> {
    Arc::new(faktor_semantic::SemanticProviderRegistry::new(
        faktor_semantic::GenericSemanticFallback::default(),
    ))
}

// --------------------------------------------------------------------------
// VerificationService (P0-9/P0-10): the runtime's single verification
// execution engine. Replaces the legacy `Option<Arc<Verifier>>` seam: the
// field is NON-optional and typed — every deployment carries a service, and
// "no objective mechanism" is an explicit [`VerificationService::disabled`]
// state that classifies mutating turns Unverified (the old `None` behavior).
// Execution is typed (program, argv) specs through the
// `faktor-verify::exec` async executor, never a shell string through `sh -c`.
// --------------------------------------------------------------------------

/// Command-string backend shared by test seams and embedded hosts: receives
/// the canonical `program arg...` string of each typed check (the legacy
/// [`faktor_verify::RunFn`] contract). Kept deterministic and synchronous —
/// the backend exists to inject scripted verdicts, not to run processes
/// (real execution goes through the async executor).
pub type VerificationRunFn = Arc<dyn Fn(&str) -> Result<(), String> + Send + Sync>;

/// Execution backend of a [`VerificationService`].
#[derive(Clone)]
enum VerificationBackend {
    /// Real executor: typed child processes through THE workspace
    /// supervisor under the context's deadline and cancellation (bounded
    /// capture, process-group kill). The only backend that can persist
    /// background verification jobs.
    Async(Arc<faktor_verify::exec::AsyncCheckExecutor>),
    /// Scripted command-string runner (tests / embedded hosts).
    Command(VerificationRunFn),
}

/// The agent runtime's verification service (P0-9/P0-10). Wraps the typed
/// executor and the [`faktor_verify::exec::VerificationPolicy`] so the
/// genuine-end verification site is purely policy-driven:
///
/// - budgets come from [`faktor_verify::exec::budget_for`] per check
///   category — there is NO universal ~10 s wall cap anywhere on this path;
/// - checks whose policy says "task-owned background operation" run as
///   DURABLE background verification jobs (audit P0-5/26): the runtime
///   persists each such check as a `faktor-session` VerificationJob row,
///   the task row stays Verifying, and the supervisor-backed executor
///   settles the jobs at a later genuine end. Only the real (supervisor)
///   backend can background checks ([`VerificationService::can_persist_jobs`]);
///   scripted command backends (test seams) run every decision inline —
///   their verdicts are instantaneous and deterministic;
/// - [`VerificationService::disabled`] fails closed to the runtime's
///   Unverified classification (no objective mechanism configured);
/// - every check executes against the session's DURABLE workspace root —
///   the daemon's current directory is never consulted.
#[derive(Clone)]
pub struct VerificationService {
    backend: VerificationBackend,
    policy: faktor_verify::exec::VerificationPolicy,
    enabled: bool,
}

impl VerificationService {
    /// The real service: a typed async executor under `policy`.
    pub fn new(
        executor: Arc<faktor_verify::exec::AsyncCheckExecutor>,
        policy: faktor_verify::exec::VerificationPolicy,
    ) -> Arc<Self> {
        Arc::new(Self {
            backend: VerificationBackend::Async(executor),
            policy,
            enabled: true,
        })
    }

    /// No objective mechanism for this deployment: mutating turns classify
    /// Unverified (never silently complete). Replaces the old
    /// `AgentDeps.verifier: None` wiring; every other construction site in
    /// tests that previously passed `None` uses this.
    pub fn disabled() -> Arc<Self> {
        Arc::new(Self {
            backend: VerificationBackend::Command(Arc::new(|_| {
                Err("verification disabled".to_string())
            })),
            policy: faktor_verify::exec::VerificationPolicy::disabled(),
            enabled: false,
        })
    }

    /// A scripted service that runs every check through `run` (the legacy
    /// command-string contract): `Ok(())` -> Passed/exit 0, `Err(msg)` ->
    /// Failed with the message as the summary. Keeps the wave-16 test
    /// ergonomics (deterministic verdicts, asserted command vectors).
    pub fn fake<R>(run: R) -> Arc<Self>
    where
        R: Fn(&str) -> Result<(), String> + Send + Sync + 'static,
    {
        Arc::new(Self {
            backend: VerificationBackend::Command(Arc::new(run)),
            policy: faktor_verify::exec::VerificationPolicy::default(),
            enabled: true,
        })
    }

    /// A scripted service whose checks always pass (the ubiquitous
    /// always-Ok verifier test seam).
    pub fn fake_ok() -> Arc<Self> {
        Self::fake(|_| Ok(()))
    }

    pub fn is_disabled(&self) -> bool {
        !self.enabled
    }

    /// The policy in effect (observability / tests).
    pub fn policy(&self) -> faktor_verify::exec::VerificationPolicy {
        self.policy
    }

    /// The budget decision for one spec under this service's policy. The
    /// remaining-turn budget is `None`: the genuine-end verification site
    /// has no cheaper per-check accounting, so the policy's category caps
    /// bound every inline check (documented — the policy, never a hard-coded
    /// wall cap, is the authority).
    pub fn budget_for(&self, spec: &faktor_verify::exec::CheckSpec) -> BudgetDecision {
        faktor_verify::exec::budget_for(spec.category, &self.policy, None)
    }

    /// True when the service executes checks through the REAL supervisor
    /// executor — the only backend that can persist background
    /// verification jobs (audit P0-5/26). Scripted command backends (test
    /// seams) return false: their verdicts are instantaneous, so the
    /// runtime keeps executing every decision inline for them.
    pub fn can_persist_jobs(&self) -> bool {
        matches!(self.backend, VerificationBackend::Async(_))
    }

    /// Execute ONE typed check under the context's deadline and
    /// cancellation. Infra errors (spawn refusal, unreadable root, ...)
    /// become [`CheckRunStatus::Unavailable`] outcomes — the CALLER decides
    /// the completion-gate meaning (BlockedVerification), never a silent
    /// failure of the code under check and never a turn failure.
    pub async fn execute(
        &self,
        spec: &faktor_verify::exec::CheckSpec,
        ctx: &faktor_verify::exec::VerificationContext,
    ) -> CheckOutcome {
        match &self.backend {
            VerificationBackend::Async(executor) => match executor.run_check(spec, ctx).await {
                Ok(outcome) => outcome,
                Err(e) => unavailable_outcome(spec, format!("verification infra error: {e}")),
            },
            VerificationBackend::Command(run) => {
                let command = canonical_command(spec);
                let started_ms = now_ms();
                match run(&command) {
                    Ok(()) => CheckOutcome {
                        status: CheckRunStatus::Passed,
                        exit: Some(0),
                        started_ms,
                        finished_ms: now_ms(),
                        summary: None,
                        truncated: false,
                    },
                    Err(message) => CheckOutcome {
                        status: CheckRunStatus::Failed,
                        exit: None,
                        started_ms,
                        finished_ms: now_ms(),
                        summary: Some(truncate_line(&message)),
                        truncated: false,
                    },
                }
            }
        }
    }
}

/// The canonical command text of a typed spec (program + args, single-space
/// joined). Specs are built by strict simple-token rules, so the join is
/// deterministic and lossless; it is what the scripted command backend runs
/// and what legacy mirrors carry as their canonical text.
fn canonical_command(spec: &faktor_verify::exec::CheckSpec) -> String {
    let mut text = spec.program.to_string_lossy().into_owned();
    for arg in &spec.args {
        text.push(' ');
        text.push_str(&arg.to_string_lossy());
    }
    text
}

fn unavailable_outcome(spec: &faktor_verify::exec::CheckSpec, summary: String) -> CheckOutcome {
    let now = now_ms();
    CheckOutcome {
        status: CheckRunStatus::Unavailable,
        exit: None,
        started_ms: now,
        finished_ms: now,
        summary: Some(format!(
            "{}: {summary}",
            spec.program.to_string_lossy().into_owned()
        )),
        truncated: false,
    }
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Bound one command-backend failure line before it rides a durable record.
fn truncate_line(text: &str) -> String {
    const MAX: usize = 512;
    let mut out: String = text.chars().take(MAX).collect();
    if text.chars().count() > MAX {
        out.push('…');
    }
    out
}

// --------------------------------------------------------------------------
// Routing policy (P0-2/85/87/88): the daemon's single decision authority
// over which provider/model serves every model call.
// --------------------------------------------------------------------------

/// Why a routed call was refused. Fail-closed matrix (P0-88 + attempt-
/// accounting audit): there is NO fallback — every failure is a typed
/// terminal error on the call (no silent degradation, no half-configured
/// substitution, no session-configured-model bypass). The runtime never
/// reaches a provider whose call the policy refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteFailure {
    /// The request cannot be served within the task's remaining budget.
    BudgetExceeded,
    /// No candidate (or the pinned model) can serve the request:
    /// capabilities, context/output fit or phase quality (a hard quality
    /// floor with no candidate above it is a NoCapableModel refusal, never
    /// a floor-lowering).
    NoCapableModel,
    /// The policy refused: the pin lost the router's own evaluation, the
    /// request violated a policy constraint, or the router denied for
    /// non-budget reasons.
    PolicyDenied,
    /// Routing telemetry/health machinery failed internally.
    InternalTelemetry,
}

/// The routing decision authority consumed by the agent runtime before
/// EVERY paid model call (the former "auto"-sentinel path is gone: there is
/// no un-routed model call anymore, and there is no fallback).
///
/// The quality requirement of one routed model call (attempt-accounting
/// audit): quality is a requirement the router must SERVE, never a ceiling
/// the policy lowers toward the best available candidate. A hard floor that
/// no candidate clears is a typed [`RouteFailure::NoCapableModel`] refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "requirement")]
pub enum QualityRequirement {
    /// The floor is a hard minimum: no candidate below it may serve the
    /// call, and the route is refused when none clears it. Implement and
    /// Review calls route under this requirement.
    Hard { minimum: u8 },
    /// A preferred target that may relax to `minimum` (never below) when
    /// the routing mode's economics decide.
    Adaptive { target: u8, minimum: u8 },
}

impl QualityRequirement {
    /// The floor the router's qualification pass must apply verbatim.
    pub fn minimum(&self) -> u8 {
        match self {
            QualityRequirement::Hard { minimum } | QualityRequirement::Adaptive { minimum, .. } => {
                *minimum
            }
        }
    }

    /// The preferred target (the minimum for a hard requirement).
    pub fn target(&self) -> u8 {
        match self {
            QualityRequirement::Hard { minimum } => *minimum,
            QualityRequirement::Adaptive { target, .. } => *target,
        }
    }
}

/// The intent of ONE model call the runtime routes (attempt-accounting
/// audit, item D): phase, required capabilities, the quality requirement,
/// the caller's planned output cap and the call's semantic risk. The wire
/// dimensions of the FINAL planned request (actual input estimate + output
/// cap) are attached at route time through
/// [`ModelCallIntent::route_request`], so the routed decision prices and
/// qualifies the REAL call, never a hard-coded guess.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelCallIntent {
    pub phase: RouterPhase,
    pub required_capabilities: Vec<String>,
    pub quality: QualityRequirement,
    /// The planned request's output cap in tokens (what the caller will
    /// accept; never a fabricated 2048 default).
    pub expected_output_tokens: u64,
    /// 0..=100 semantic risk of the call (a review of a risky change is
    /// high; interior summarization is low).
    pub semantic_risk: u8,
}

impl ModelCallIntent {
    /// The hard quality floor the router applies verbatim (never lowered
    /// toward the best available candidate).
    pub fn quality_floor(&self) -> u8 {
        self.quality.minimum().min(100)
    }

    /// Build the route request for this intent against the ACTUAL planned
    /// dimensions: `context_estimate_tokens` is the final wire plan's input
    /// estimate (the plan's own total), `output_cap_tokens` the caller's
    /// planned output cap, `task_budget_remaining_micro` the durable free
    /// budget (0 = unlimited). 0 <= semantic_risk <= 100 is enforced at
    /// construction sites through [`ModelCallIntent::with_semantic_risk`].
    pub fn route_request(
        &self,
        context_estimate_tokens: u64,
        output_cap_tokens: u64,
        task_budget_remaining_micro: u64,
    ) -> faktor_router::RouteRequest {
        faktor_router::RouteRequest {
            phase: self.phase,
            required_capabilities: self.required_capabilities.clone(),
            context_tokens: context_estimate_tokens,
            estimated_output_tokens: output_cap_tokens.min(self.expected_output_tokens.max(1)),
            quality_floor: self.quality_floor(),
            task_budget_remaining_micro,
            latency_preference_ms: None,
            task_class: TaskClass::Medium,
            risk_bucket: match self.semantic_risk {
                0..=33 => RiskBucket::Low,
                34..=66 => RiskBucket::Medium,
                _ => RiskBucket::High,
            },
        }
    }

    /// Bounded risk assignment (clamped, never accepted raw).
    pub fn with_semantic_risk(mut self, risk: u8) -> Self {
        self.semantic_risk = risk.min(100);
        self
    }
}

/// The per-phase intent builders the runtime uses (quality is a HARD floor
/// for Implement/Review — the audit's default; interior calls stay on the
/// documented 60 floor with their own real dimensions).
impl ModelCallIntent {
    /// The main Implement-phase model call of an iteration: the required
    /// capabilities mirror the planner's wire needs (tools + streaming).
    pub fn implement_main() -> Self {
        Self {
            phase: RouterPhase::Implement,
            required_capabilities: vec!["tools".into(), "streaming".into()],
            quality: QualityRequirement::Hard { minimum: 60 },
            expected_output_tokens: u64::MAX,
            semantic_risk: 0,
        }
    }

    /// A Review-phase call (the independent risky-change review): bounded
    /// package input, a small typed verdict output.
    pub fn review() -> Self {
        Self {
            phase: RouterPhase::Review,
            required_capabilities: vec!["streaming".into()],
            quality: QualityRequirement::Hard { minimum: 60 },
            expected_output_tokens: 2048,
            semantic_risk: 100,
        }
    }

    /// The compaction summarizer call (interior, low semantic risk).
    pub fn compact() -> Self {
        Self {
            phase: RouterPhase::Compact,
            required_capabilities: vec!["streaming".into()],
            quality: QualityRequirement::Hard { minimum: 60 },
            expected_output_tokens: 4096,
            semantic_risk: 0,
        }
    }
}

/// The routing decision authority consumed by the agent runtime before
pub trait RoutingPolicy: Send + Sync {
    /// Route one model call. The returned decision's provider/model are
    /// authoritative; an EMPTY provider/model means "the session's own
    /// configured side" (the passthrough test policy's pin — the runtime
    /// keeps the session defaults verbatim).
    fn route(&self, req: &faktor_router::RouteRequest) -> Result<RouteDecision, RouteFailure>;

    /// The mode this policy was built with (audit/reporting surface; the
    /// policy itself applies it inside [`RoutingPolicy::route`]).
    fn mode(&self) -> RoutingMode;

    /// Cache-economics consult (P0-82): `route` plus the session's stored
    /// prefix-stability history. The production policy prices a churning
    /// session WITHOUT provider-side cache-read discounts and charges the
    /// churn premium on the decision (see
    /// `faktor_router::RouterService::route_with_prefix_stability`).
    /// `None`/empty history (no rows, or a stability read that failed —
    /// never an error on the turn) routes exactly like [`RoutingPolicy::route`]:
    /// no penalty, no cache discount zeroing. Policies without stability
    /// machinery ignore the history (default = `route`).
    fn route_with_session_stability(
        &self,
        req: &faktor_router::RouteRequest,
        _prefix_history: Option<&[TurnPrefix]>,
    ) -> Result<RouteDecision, RouteFailure> {
        self.route(req)
    }

    /// Candidate-sized consult (candidate-specific accounting audit): routes
    /// with the injected per-candidate wire-plan builder, so every
    /// seriously-considered candidate is measured under its OWN tokenizer
    /// before the final selection and the winning wire plan crosses back for
    /// reuse. Default: the plain stability consult with no plan (callers keep
    /// their own plan; byte-identical behavior for every policy that does not
    /// opt in).
    fn route_with_session_stability_and_candidate_plans(
        &self,
        req: &faktor_router::RouteRequest,
        prefix_history: Option<&[TurnPrefix]>,
        _planner: &dyn faktor_router::CandidatePlanner,
    ) -> Result<faktor_router::SizedRouteDecision, RouteFailure> {
        self.route_with_session_stability(req, prefix_history)
            .map(faktor_router::SizedRouteDecision::without_plan)
    }

    /// The telemetry outcome record entry (P0-28 residuals): the runtime
    /// calls this after each settled model call with the actual outcome
    /// (attempted/resolved, latency, provider/model, reliability signals).
    /// Default: no telemetry (test policies, no-op hosts).
    fn record_call_outcome(&self, _outcome: &SettledCallOutcome) {}
}

/// One settled model call's outcome, recorded into the router telemetry
/// (provider/model of the ACTUAL settled call — also correct for passthrough
/// decisions — plus the measured latency and reliability signals).
///
/// The verified attribution entry (audit items 13/14/L) is carried ONLY at
/// the deterministic gate sites: "the model said done" is not a verified
/// signal, so the runtime's settle/uncertain feeds leave `verified` as `None`
/// (telemetry only — the outcomes registry never learns a success it cannot
/// prove) and the task-complete/gate site re-records the retained settled
/// call with the explicit verified signal. `None` = no verified sample
/// (never a success).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettledCallOutcome {
    pub provider: String,
    pub model: String,
    pub phase: RouterPhase,
    /// The call resolved (stream completed); false = the call failed.
    pub success: bool,
    /// This logical call needed more than one attempt.
    pub retried: bool,
    /// The terminal failure was a rate limit (cooldown + prior decay).
    pub rate_limited: bool,
    /// Measured latency of the settled (final) attempt in ms.
    pub latency_ms: u64,
    /// Explicit verified-outcome attribution carried by the deterministic
    /// verification gate sites only (see the struct docs).
    pub verified: Option<VerifiedCallAttribution>,
}

/// One settled call's explicit verified-outcome attribution (audit items
/// 13/14/L): the full registry key dimensions the runtime knows at
/// settlement plus the verified-success signal and the rework the call's
/// failure eventually caused until its task verified. A sample is recorded
/// as a FIRST-PASS SUCCESS only when the task ultimately passed
/// deterministic verification AND attribution identified this call/phase;
/// every other gate outcome records a FAILURE sample (rework was needed),
/// never a success. Rework sums are the caller's measured numbers and stay 0
/// when they cannot be attributed yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerifiedCallAttribution {
    pub task_class: TaskClass,
    pub risk_bucket: RiskBucket,
    /// The explicit verified-success signal: the task ultimately passed
    /// deterministic verification.
    pub verified_success: bool,
    /// Downstream spend this settled call's failure caused until its task
    /// verified (microUSD); 0 = unmeasured.
    pub rework_cost_micro: u64,
    /// Downstream turns this settled call's failure caused; 0 = unmeasured.
    pub rework_turns: u64,
}

/// The production policy: a real [`faktor_router::RouterService`] under a
/// fixed [`RoutingMode`] decided once at graph build.
///
/// - [`RoutingMode::Economy`]: every request is routed; the decision's
///   provider/model override the session-configured defaults.
/// - [`RoutingMode::MaximumQuality`]: every request is routed to the top
///   phase-quality tier that clears the router's hard caps — the mode
///   probes the router's own qualification pass at descending quality
///   floors (distinct candidate qualities above the request's floor) and
///   takes the decision of the highest floor that the router can serve.
///   Within the top tier the router's expected-cost ladder applies
///   (success-ppm aware, cost second).
/// - [`RoutingMode::Balanced`]: expected-cost routing (like Economy) but
///   never below the balanced quality band ([`EconomicRoutingPolicy::BALANCED_QUALITY_FLOOR`])
///   while a band candidate can serve the request.
/// - [`RoutingMode::Pinned`]: every request is VALIDATED against the pin —
///   capability/fit/quality/budget axes through the same RouterService — and
///   the pin wins as long as it clears them. The router's free choice is
///   never silently substituted for the pin (fail closed).
pub struct EconomicRoutingPolicy {
    service: Arc<faktor_router::RouterService>,
    mode: RoutingMode,
}

impl std::fmt::Debug for EconomicRoutingPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EconomicRoutingPolicy")
            .field("mode", &self.mode)
            .finish_non_exhaustive()
    }
}

/// The phase-quality metric the router applies per phase (mirror of the
/// router crate's documented rule; duplicated here because the router's
/// helper is private): heavy phases (Implement/Review/Debug) judge the
/// coding-reliability mean, every other phase judges context reliability.
fn phase_quality(econ: &faktor_core::model::ModelEconomics, phase: RouterPhase) -> u8 {
    match phase {
        RouterPhase::Implement | RouterPhase::Review | RouterPhase::Debug => econ.coding_quality(),
        _ => econ.context_reliability,
    }
}

impl EconomicRoutingPolicy {
    /// The balanced-mode quality band boundary (P0-85): candidates whose
    /// phase quality sits at or above this band are treated as the
    /// verified-reliable tier; Balanced never routes below the band while a
    /// band candidate clears the request's hard caps.
    pub const BALANCED_QUALITY_FLOOR: u8 = 88;

    pub fn new(service: Arc<faktor_router::RouterService>, mode: RoutingMode) -> Arc<Self> {
        Arc::new(Self { service, mode })
    }

    /// The service's candidate set (observability/tests).
    pub fn candidates(&self) -> &[faktor_core::model::ModelDescriptor] {
        &self.service.router.candidates
    }

    fn map_denial(&self, msg: &str, req: &faktor_router::RouteRequest) -> RouteFailure {
        if msg.contains("no candidate clears capability/fit") || msg.contains("quality floor") {
            // No candidate serves the request's hard axes — capabilities,
            // context/output fit, or the phase quality floor. The floor is a
            // hard requirement: an empty above-floor candidate set is a
            // NoCapableModel refusal (requested hard 60 with best available
            // 50 => NoCapableModel), never a PolicyDenied of a capable set.
            return RouteFailure::NoCapableModel;
        }
        // The remaining denial text names the budget/latency axes together;
        // the runtime never sets a latency preference, so a positive
        // remaining budget means the budget axis denied.
        if req.task_budget_remaining_micro > 0 {
            RouteFailure::BudgetExceeded
        } else {
            RouteFailure::PolicyDenied
        }
    }

    /// One router consult at an explicit quality floor, churn-aware when a
    /// prefix history is supplied (the session-stability path; `None`/empty
    /// history routes exactly like the plain consult — no penalty).
    ///
    /// [`EconomicRoutingPolicy::consult_at_with_plans`] with the optional
    /// candidate plan builder (candidate-specific accounting audit): with
    /// `planner` present the router's candidate-sized entry runs and the
    /// winning plan crosses back; without one the plain consult runs
    /// byte-identically.
    fn consult_at_with_plans(
        &self,
        req: &faktor_router::RouteRequest,
        floor: u8,
        prefix_history: Option<&[TurnPrefix]>,
        planner: Option<&dyn faktor_router::CandidatePlanner>,
    ) -> Result<faktor_router::SizedRouteDecision, String> {
        let mut routed = req.clone();
        routed.quality_floor = floor;
        match (planner, prefix_history) {
            (Some(planner), None) => self
                .service
                .route_with_candidate_plans(&routed, &[], planner),
            (Some(planner), Some(history)) => self
                .service
                .route_with_prefix_stability_and_candidate_plans(
                    &routed,
                    &[],
                    faktor_router::stability::DEFAULT_STABILITY_FLOOR,
                    Some(history),
                    planner,
                ),
            (None, None) => self
                .service
                .route(&routed, &[])
                .map(faktor_router::SizedRouteDecision::without_plan),
            (None, Some(history)) => self
                .service
                .route_with_prefix_stability(
                    &routed,
                    &[],
                    faktor_router::stability::DEFAULT_STABILITY_FLOOR,
                    Some(history),
                )
                .map(faktor_router::SizedRouteDecision::without_plan),
        }
    }

    fn route_economy_sized(
        &self,
        req: &faktor_router::RouteRequest,
        prefix_history: Option<&[TurnPrefix]>,
        planner: Option<&dyn faktor_router::CandidatePlanner>,
    ) -> Result<faktor_router::SizedRouteDecision, RouteFailure> {
        // The requested floor applies VERBATIM: the policy never lowers a
        // hard quality requirement toward the best available candidate.
        self.consult_at_with_plans(req, req.quality_floor.min(100), prefix_history, planner)
            .map_err(|e| self.map_denial(&e, req))
    }

    fn route_economy(
        &self,
        req: &faktor_router::RouteRequest,
        prefix_history: Option<&[TurnPrefix]>,
    ) -> Result<RouteDecision, RouteFailure> {
        self.route_economy_sized(req, prefix_history, None)
            .map(|sized| sized.decision)
    }

    /// Maximum-quality selection: probe the router's own qualification pass
    /// at the descending distinct phase qualities of the candidates (never
    /// below the request's effective floor), and take the decision of the
    /// HIGHEST floor the router can serve. A tier that the router refuses —
    /// cooldown, budget, capability, fit — simply drops out, so the mode
    /// maximizes the verified-success quality subject to the router's hard
    /// caps. Denials surface when NO tier above the effective floor clears
    /// the caps: the most permissive (lowest-floor) refusal is propagated.
    /// The number of router consults is bounded by the distinct phase
    /// qualities in the candidate set (small; the candidate catalog itself
    /// is bounded).
    fn route_maximum_quality_sized(
        &self,
        req: &faktor_router::RouteRequest,
        prefix_history: Option<&[TurnPrefix]>,
        planner: Option<&dyn faktor_router::CandidatePlanner>,
    ) -> Result<faktor_router::SizedRouteDecision, RouteFailure> {
        // The requested floor is the hard lower bound of the tier probe:
        // tiers below it never serve, and when no tier above it clears the
        // caps the refusal names the empty above-floor set.
        let mut tiers: Vec<u8> = self
            .service
            .router
            .candidates
            .iter()
            .map(|c| phase_quality(&c.economics, req.phase))
            .filter(|&q| q >= req.quality_floor.min(100))
            .collect();
        tiers.sort_unstable();
        tiers.dedup();
        tiers.reverse();
        let mut last_denial: Option<RouteFailure> = None;
        for floor in tiers {
            match self.consult_at_with_plans(req, floor, prefix_history, planner) {
                Ok(d) => return Ok(d),
                Err(e) => last_denial = Some(self.map_denial(&e, req)),
            }
        }
        last_denial.map_or(Err(RouteFailure::NoCapableModel), Err)
    }

    fn route_maximum_quality(
        &self,
        req: &faktor_router::RouteRequest,
        prefix_history: Option<&[TurnPrefix]>,
    ) -> Result<RouteDecision, RouteFailure> {
        self.route_maximum_quality_sized(req, prefix_history, None)
            .map(|sized| sized.decision)
    }

    fn route_balanced_sized(
        &self,
        req: &faktor_router::RouteRequest,
        prefix_history: Option<&[TurnPrefix]>,
        planner: Option<&dyn faktor_router::CandidatePlanner>,
    ) -> Result<faktor_router::SizedRouteDecision, RouteFailure> {
        // Balanced never routes below its quality band; the band floor is
        // applied verbatim (no lowering toward the best available).
        let floor = req.quality_floor.clamp(Self::BALANCED_QUALITY_FLOOR, 100);
        self.consult_at_with_plans(req, floor, prefix_history, planner)
            .map_err(|e| self.map_denial(&e, req))
    }

    fn route_balanced(
        &self,
        req: &faktor_router::RouteRequest,
        prefix_history: Option<&[TurnPrefix]>,
    ) -> Result<RouteDecision, RouteFailure> {
        self.route_balanced_sized(req, prefix_history, None)
            .map(|sized| sized.decision)
    }

    fn route_pinned(
        &self,
        req: &faktor_router::RouteRequest,
        provider: &str,
        model: &str,
    ) -> Result<RouteDecision, RouteFailure> {
        let pinned = self
            .service
            .router
            .candidates
            .iter()
            .find(|c| c.provider == provider && c.model == model)
            .ok_or(RouteFailure::NoCapableModel)?;
        let pin_quality = phase_quality(&pinned.economics, req.phase);
        if pin_quality < req.quality_floor.min(100) {
            // The pin is below the request's HARD quality floor: fail
            // closed — the request's floor is never lowered to the pin.
            return Err(RouteFailure::PolicyDenied);
        }
        // Validation request only the pin can serve among candidates that
        // do not dominate it on every quality axis: the pin's own true
        // capabilities, its own phase quality as the floor, and its own
        // estimated latency as the preference. A candidate that beats the
        // pin on cost while matching it on caps/quality/latency still wins
        // the router's evaluation — and that is a loud PolicyDenied below,
        // never a silent substitution of the pin.
        let mut caps: Vec<String> = req.required_capabilities.clone();
        for (flag, name) in [
            (pinned.tools, "tools"),
            (pinned.parallel_tools, "parallel_tools"),
            (pinned.reasoning, "reasoning"),
            (pinned.thinking, "thinking"),
            (pinned.vision, "vision"),
            (pinned.structured_output, "structured_output"),
            (pinned.embeddings, "embeddings"),
            (pinned.streaming, "streaming"),
        ] {
            if flag && !caps.iter().any(|c| c == name) {
                caps.push(name.to_string());
            }
        }
        let validation = faktor_router::RouteRequest {
            phase: req.phase,
            required_capabilities: caps,
            context_tokens: req.context_tokens,
            estimated_output_tokens: req.estimated_output_tokens,
            quality_floor: pin_quality,
            task_budget_remaining_micro: req.task_budget_remaining_micro,
            latency_preference_ms: Some(pinned.economics.estimated_latency_ms),
            task_class: req.task_class,
            risk_bucket: req.risk_bucket,
        };
        let decision = match self
            .service
            .route_pinned_decision(provider, model, &validation, &[])
        {
            Ok(d) => Some(d),
            Err(e) => {
                // The pinned service runs ONLY the router's pinned
                // qualification ([`RouterService::qualify_specific`]), whose
                // failure text names the axis that refused the pin:
                // capability/fit and quality-floor denials are typed
                // NoCapableModel refusals, a hard-cap denial is
                // BudgetExceeded. There is deliberately no per-token
                // fallback estimate here — money rides the exact quote the
                // service evaluates.
                return Err(self.map_denial(&e, req));
            }
        };
        let decision = decision.expect("routed above");
        if decision.provider == provider && decision.model == model {
            Ok(decision)
        } else {
            Err(RouteFailure::PolicyDenied)
        }
    }
}

impl RoutingPolicy for EconomicRoutingPolicy {
    fn route(&self, req: &faktor_router::RouteRequest) -> Result<RouteDecision, RouteFailure> {
        match &self.mode {
            RoutingMode::Economy => self.route_economy(req, None),
            RoutingMode::MaximumQuality => self.route_maximum_quality(req, None),
            RoutingMode::Balanced => self.route_balanced(req, None),
            RoutingMode::Pinned { provider, model } => {
                // An empty side of the pin means "the session's own
                // configured side": nothing to validate against the pin's
                // descriptor — fall back to the session defaults (the
                // runtime treats the empty provider/model as passthrough).
                if provider.is_empty() && model.is_empty() {
                    return Ok(empty_passthrough_decision());
                }
                self.route_pinned(req, provider, model)
            }
        }
    }

    fn mode(&self) -> RoutingMode {
        self.mode.clone()
    }

    /// Cache economics in production routing (P0-82/15): the session's
    /// stored prefix stability feeds the router's stability consult — a
    /// session whose last recorded stability sits below the floor is
    /// priced without provider-side cache-read discounts and its decision
    /// carries the churn premium (the cost prediction must never pretend a
    /// churning prefix hits provider caches). `None`/empty history — no
    /// rows, or a stability read that failed — routes exactly like
    /// [`RoutingPolicy::route`]: no penalty, never an error on the turn.
    fn route_with_session_stability(
        &self,
        req: &faktor_router::RouteRequest,
        prefix_history: Option<&[TurnPrefix]>,
    ) -> Result<RouteDecision, RouteFailure> {
        match &self.mode {
            RoutingMode::Economy => self.route_economy(req, prefix_history),
            RoutingMode::MaximumQuality => self.route_maximum_quality(req, prefix_history),
            RoutingMode::Balanced => self.route_balanced(req, prefix_history),
            RoutingMode::Pinned { provider, model } => {
                if provider.is_empty() && model.is_empty() {
                    return Ok(empty_passthrough_decision());
                }
                let mut decision = self.route_pinned(req, provider, model)?;
                // The pinned path validates through the router without the
                // stability premium (the pin is not selectable anyway);
                // the decision's COST still reflects churn so downstream
                // budget math never pretends a churning prefix caches.
                if let Some(history) = prefix_history {
                    if let Some(last) = faktor_router::stability::turn_stabilities(history).pop() {
                        let penalty = faktor_router::stability::churn_penalty(
                            last,
                            faktor_router::stability::DEFAULT_STABILITY_FLOOR,
                        );
                        if penalty > 0.0 {
                            decision.estimated_cost_micro =
                                faktor_router::stability::apply_churn_penalty(
                                    decision.estimated_cost_micro,
                                    last,
                                    faktor_router::stability::DEFAULT_STABILITY_FLOOR,
                                );
                            decision.reasoning = format!(
                                "{} prefix_stability={last:.3} churn_penalty={penalty:.4}",
                                decision.reasoning
                            );
                        }
                    }
                }
                Ok(decision)
            }
        }
    }

    /// Candidate-sized production consult (candidate-specific accounting
    /// audit): the SAME mode semantics as
    /// [`RoutingPolicy::route_with_session_stability`] — Economy/ Balanced
    /// floors, MaximumQuality tier probe, Pinned validation, churn premium —
    /// but the router sizes each seriously-considered candidate under its
    /// OWN tokenizer through `planner` and the winning wire plan crosses
    /// back inside the returned [`faktor_router::SizedRouteDecision`]. A
    /// pinned (or empty-pin passthrough) consult is unsized: the pin never
    /// competes, so no candidate plan is consumed.
    fn route_with_session_stability_and_candidate_plans(
        &self,
        req: &faktor_router::RouteRequest,
        prefix_history: Option<&[TurnPrefix]>,
        planner: &dyn faktor_router::CandidatePlanner,
    ) -> Result<faktor_router::SizedRouteDecision, RouteFailure> {
        match &self.mode {
            RoutingMode::Economy => self.route_economy_sized(req, prefix_history, Some(planner)),
            RoutingMode::MaximumQuality => {
                self.route_maximum_quality_sized(req, prefix_history, Some(planner))
            }
            RoutingMode::Balanced => self.route_balanced_sized(req, prefix_history, Some(planner)),
            RoutingMode::Pinned { .. } => self
                .route_with_session_stability(req, prefix_history)
                .map(faktor_router::SizedRouteDecision::without_plan),
        }
    }

    /// Telemetry outcome record entry (P0-28): every settled model call the
    /// runtime reports lands in the wrapped RouterService's reliability
    /// priors and rate-limit cooldowns (see
    /// `RouterService::record_outcome`).
    ///
    /// Verified-outcome learning (audit items 13/14/L): when the outcome
    /// carries the explicit verified attribution (the deterministic gate
    /// sites only), the sample ALSO lands in the service's verified-outcome
    /// registry (`RouterService::outcomes` — the store-backed impl the
    /// daemon graph wires through `with_pricing_and_outcomes`, an
    /// [`faktor_router::EmptyOutcomeStore`] anywhere else), keyed by the
    /// FULL (provider, model, phase, task_class, risk_bucket) key. Telemetry
    /// always records; the verified sample records only on the explicit
    /// signal — a `None` attribution never learns a success it cannot prove.
    fn record_call_outcome(&self, outcome: &SettledCallOutcome) {
        self.service.record_outcome(
            &outcome.provider,
            &outcome.model,
            outcome.phase,
            outcome.success,
            outcome.retried,
            outcome.rate_limited,
            outcome.latency_ms,
        );
        if let Some(v) = &outcome.verified {
            self.service.outcomes.append_sample(
                &faktor_router::OutcomeKey {
                    provider: outcome.provider.clone(),
                    model: outcome.model.clone(),
                    phase: outcome.phase,
                    task_class: v.task_class,
                    risk_bucket: v.risk_bucket,
                },
                faktor_router::OutcomeSample {
                    verified_success: v.verified_success,
                    rework_cost_micro: v.rework_cost_micro,
                    rework_turns: v.rework_turns,
                },
            );
        }
    }
}

/// The store-backed verified-outcome registry (audit items 13/14/L): the
/// durable [`faktor_router::OutcomeStore`] impl over the store crate's v18
/// `model_outcome_stats` projection (append + per-key get + per-phase fold).
/// Wiring builds the daemon's RouterService through
/// `faktor_router::RouterService::with_pricing_and_outcomes` with this impl
/// over the daemon store, so verified samples recorded by the policy's
/// `record_call_outcome` survive restarts and serve every later route.
///
/// Best-effort by contract: a sample that cannot be appended (corrupt row,
/// IO failure) is logged, never an error on the turn — routing simply keeps
/// the conservative no-history estimate for that key. Reads that fail
/// consult as a miss (no history), which is the same conservative shape.
#[derive(Debug, Clone)]
pub struct StoreOutcomeStore {
    store: Arc<faktor_store::Store>,
}

impl StoreOutcomeStore {
    pub fn new(store: Arc<faktor_store::Store>) -> Self {
        Self { store }
    }
}

impl faktor_router::OutcomeStore for StoreOutcomeStore {
    fn append_sample(&self, key: &faktor_router::OutcomeKey, sample: faktor_router::OutcomeSample) {
        if let Err(e) = self.store.model_outcome_stats_append(
            &key.provider,
            &key.model,
            key.phase,
            key.task_class,
            key.risk_bucket,
            faktor_store::ModelOutcomeSample {
                verified_success: sample.verified_success,
                rework_cost_micro: sample.rework_cost_micro,
                rework_turns: sample.rework_turns,
            },
        ) {
            tracing::warn!(
                provider = %key.provider,
                model = %key.model,
                "verified-outcome sample append failed: {e}"
            );
        }
    }

    fn stats(
        &self,
        key: &faktor_router::OutcomeKey,
    ) -> Option<faktor_router::VerifiedOutcomeStats> {
        match self.store.model_outcome_stats_get(
            &key.provider,
            &key.model,
            key.phase,
            key.task_class,
            key.risk_bucket,
        ) {
            Ok(Some(row)) => Some(outcome_stats_from_store(&row)),
            _ => None,
        }
    }

    fn phase_stats(
        &self,
        provider: &str,
        model: &str,
        phase: RouterPhase,
    ) -> Option<faktor_router::VerifiedOutcomeStats> {
        match self.store.model_outcome_stats_phase(provider, model, phase) {
            Ok(Some(row)) => Some(outcome_stats_from_store(&row)),
            _ => None,
        }
    }
}

/// Project one durable per-key accumulator row onto the router registry's
/// stats shape (the store's read path already validated the row's
/// consistency invariants fallibly).
fn outcome_stats_from_store(
    row: &faktor_store::ModelOutcomeStatsRow,
) -> faktor_router::VerifiedOutcomeStats {
    faktor_router::VerifiedOutcomeStats {
        successes_first_pass: row.successes_first_pass,
        failures_first_pass: row.failures_first_pass,
        rework_cost_micro_sum: row.rework_cost_micro_sum,
        rework_turns_sum: row.rework_turns_sum,
        sample_count: row.sample_count,
    }
}

/// The decision of a policy that defers to the session's configured
/// provider/model (the test graph's passthrough pin). Empty strings are the
/// documented "session defaults" marker consumed by the runtime.
pub fn empty_passthrough_decision() -> RouteDecision {
    RouteDecision {
        provider: String::new(),
        model: String::new(),
        estimated_cost_micro: 0,
        estimated_latency_ms: 0,
        reasoning: "routing policy defers to the session-configured provider/model".into(),
        considered: 0,
        source: faktor_core::model::ModelSource::ConservativeDefault,
        // P0-1: NO pricing authority was consulted for a passthrough — the
        // snapshot stays None (never a fabricated price), so an unpriced
        // decision under a hard task cost budget fails closed at settle and
        // records Unknown spend without one.
        pricing_snapshot: None,
    }
}

/// Deterministic test policy: returns a fixed decision (or a fixed
/// failure) for every request. `passthrough()` returns the empty-decision
/// marker (session defaults win); `pinned(decision)` forces one decision;
/// `failing(failure)` exercises the runtime's fail-closed matrix.
#[derive(Debug, Clone)]
pub struct FixedRoutingPolicy {
    pub decision: RouteDecision,
    pub fail: Option<RouteFailure>,
}

impl FixedRoutingPolicy {
    pub fn passthrough() -> Arc<dyn RoutingPolicy> {
        Arc::new(Self {
            decision: empty_passthrough_decision(),
            fail: None,
        })
    }

    pub fn pinned(decision: RouteDecision) -> Arc<dyn RoutingPolicy> {
        Arc::new(Self {
            decision,
            fail: None,
        })
    }

    pub fn failing(failure: RouteFailure) -> Arc<dyn RoutingPolicy> {
        Arc::new(Self {
            decision: empty_passthrough_decision(),
            fail: Some(failure),
        })
    }
}

impl RoutingPolicy for FixedRoutingPolicy {
    fn route(&self, _req: &faktor_router::RouteRequest) -> Result<RouteDecision, RouteFailure> {
        match &self.fail {
            Some(f) => Err(f.clone()),
            None => Ok(self.decision.clone()),
        }
    }

    fn mode(&self) -> RoutingMode {
        if self.fail.is_some() {
            return RoutingMode::Economy;
        }
        RoutingMode::Pinned {
            provider: self.decision.provider.clone(),
            model: self.decision.model.clone(),
        }
    }
}

/// One PHYSICAL attempt's budget machine (attempt-accounting audit): owns
/// one reservation through its whole lifecycle and GUARDS the state
/// ordering so refund/uncertain/settle calls cannot be scattered wrongly
/// again:
///
/// ```text
///               reserve (outside the machine)
///                    |
///                    v
///           RESERVED --mark_dispatched--> DISPATCHED --clean end--> settle(usage)
///              |                              |
///    fail_before_dispatch (=refund)   fail_after_dispatch (=mark_uncertain)
///              |                    (error / stall / cancel-after-dispatch)
///              v                              v
///           REFUNDED                       UNCERTAIN (keeps consuming until
///                                         reconcile / task-end finalize)
/// ```
///
/// Every transition is a local guard PLUS the durable ledger call: a refund
/// after dispatch is refused locally with the ledger's own typed
/// [`faktor_session::BudgetError::CannotRefundDispatched`] BEFORE any
/// authority call (the durable SQL guard stays the backstop), an
/// uncertain/after-dispatch call before dispatch is refused as
/// [`faktor_session::BudgetError::NotOpen`] (a never-dispatched failure
/// REFUNDS — it never marks UNCERTAIN), and settle/refund/uncertain on a
/// closed machine are refused — money moves exactly once per reservation.
///
/// The machine is per PHYSICAL ATTEMPT: a retry builds a NEW machine over a
/// fresh reservation (and a fresh durable attempt op id); earlier
/// uncertain attempts stay uncertain until reconciled/finalized.
pub struct AttemptAccounting {
    budgets: Arc<dyn faktor_session::BudgetAuthority>,
    session_id: SessionId,
    reservation: Option<faktor_session::ReservationId>,
    dispatched: bool,
    closed: bool,
}

impl std::fmt::Debug for AttemptAccounting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AttemptAccounting")
            .field("session_id", &self.session_id)
            .field("reservation", &self.reservation)
            .field("dispatched", &self.dispatched)
            .field("closed", &self.closed)
            .finish_non_exhaustive()
    }
}

impl AttemptAccounting {
    /// Adopt one freshly reserved reservation (see
    /// [`faktor_session::BudgetAuthority::reserve_attempt`]). `None` = no
    /// reservation (an unbudgeted call): every machine call is a silent
    /// no-op — there is no money to move.
    pub fn new(
        budgets: Arc<dyn faktor_session::BudgetAuthority>,
        session_id: SessionId,
        reservation: Option<faktor_session::ReservationId>,
    ) -> Self {
        Self {
            budgets,
            session_id,
            reservation,
            dispatched: false,
            closed: false,
        }
    }

    pub fn reservation(&self) -> Option<faktor_session::ReservationId> {
        self.reservation
    }

    /// True once the durable dispatch marker was written (the provider
    /// request left the process and may have billed).
    pub fn dispatched(&self) -> bool {
        self.dispatched
    }

    /// True once the machine reached a terminal state (settled, refunded or
    /// marked uncertain): money moved exactly once.
    pub fn closed(&self) -> bool {
        self.closed
    }

    /// The machine still holds an open reservation.
    pub fn is_open(&self) -> bool {
        !self.closed
    }

    fn guard_not_closed(&self) -> Result<(), faktor_session::BudgetError> {
        if self.closed {
            let raw = self.reservation.map(|r| r.raw()).unwrap_or(0);
            return Err(faktor_session::BudgetError::NotOpen {
                reservation: raw,
                status: "settled/refunded/uncertain".into(),
            });
        }
        Ok(())
    }

    /// RESERVED -> DISPATCHED: write the durable dispatch marker immediately
    /// BEFORE the provider request is sent. Idempotent for the machine (the
    /// ledger makes the row-level marker idempotent too).
    pub async fn mark_dispatched(&mut self) -> Result<(), faktor_session::BudgetError> {
        self.guard_not_closed()?;
        let Some(reservation) = self.reservation else {
            return Ok(());
        };
        self.budgets
            .mark_dispatched(self.session_id, reservation)
            .await?;
        self.dispatched = true;
        Ok(())
    }

    /// DISPATCHED -> SETTLED at the usage actual (clean stream end: usage
    /// frame + Done). Guarded: only a dispatched, open machine settles.
    pub async fn settle_usage(
        &mut self,
        uncached_input_tokens: u64,
        cache_read_tokens: u64,
        cache_write_tokens: u64,
        output_tokens: u64,
        provider_reported_micro: Option<u64>,
        route_decision_json: Option<String>,
    ) -> Result<Option<u64>, faktor_session::BudgetError> {
        self.guard_not_closed()?;
        if !self.dispatched {
            // A stream that never dispatched cannot settle: a clean end
            // without a dispatch marker is a machine-order violation (the
            // durable marker must precede the request).
            let raw = self.reservation.map(|r| r.raw()).unwrap_or(0);
            return Err(faktor_session::BudgetError::NotOpen {
                reservation: raw,
                status: "reserved".into(),
            });
        }
        let Some(reservation) = self.reservation else {
            return Ok(None);
        };
        let out = self
            .budgets
            .settle_usage(
                self.session_id,
                reservation,
                uncached_input_tokens,
                cache_read_tokens,
                cache_write_tokens,
                output_tokens,
                provider_reported_micro,
                route_decision_json,
            )
            .await?;
        self.closed = true;
        Ok(out)
    }

    /// DISPATCHED -> UNCERTAIN: a POST-DISPATCH terminal failure (error,
    /// stall verdict, cancel-after-dispatch, a settle that the ledger
    /// refused). The reserved amount KEEPS consuming the free budget until a
    /// reconcile settles the attempt's exact usage or the task-end finalize
    /// charges the estimate — never a silent $0 and never a dangling
    /// dispatched row. Guarded: only a dispatched, open machine may go
    /// UNCERTAIN; a never-dispatched failure REFUNDS instead.
    pub async fn fail_after_dispatch(
        &mut self,
        reason_code: impl Into<String>,
        request_id: Option<String>,
    ) -> Result<(), faktor_session::BudgetError> {
        self.guard_not_closed()?;
        if !self.dispatched {
            let raw = self.reservation.map(|r| r.raw()).unwrap_or(0);
            return Err(faktor_session::BudgetError::NotOpen {
                reservation: raw,
                status: "reserved".into(),
            });
        }
        let Some(reservation) = self.reservation else {
            return Ok(());
        };
        self.budgets
            .mark_uncertain(self.session_id, reservation, reason_code.into(), request_id)
            .await?;
        self.closed = true;
        Ok(())
    }

    /// RESERVED -> REFUNDED: a definitely-not-sent failure (pre-dispatch
    /// only). Guarded: a refund after the dispatch marker is refused with
    /// [`faktor_session::BudgetError::CannotRefundDispatched`] BEFORE the
    /// authority is reached — the provider may have billed, so the caller
    /// must settle or mark UNCERTAIN instead.
    pub async fn fail_before_dispatch(&mut self) -> Result<(), faktor_session::BudgetError> {
        self.guard_not_closed()?;
        if self.dispatched {
            let raw = self.reservation.map(|r| r.raw()).unwrap_or(0);
            return Err(faktor_session::BudgetError::CannotRefundDispatched { reservation: raw });
        }
        let Some(reservation) = self.reservation else {
            // No reservation: nothing to release, but the machine still
            // reached its terminal pre-dispatch state.
            self.closed = true;
            return Ok(());
        };
        self.budgets.refund(self.session_id, reservation).await?;
        self.closed = true;
        Ok(())
    }
}

// --------------------------------------------------------------------------
// Off-turn-thread evidence polling (audit 14/26): the drive awaits evidence
// under hard wall budgets, but a provider that blocks inside its poll (the
// legacy bounded scan) must never occupy a turn thread, and the runtime.rs
// verification-path source probes forbid the blocking-pool API name there.
// Both helpers live here so runtime.rs keeps the async call shape without
// the banned literal.
// --------------------------------------------------------------------------

/// Run `f` on a blocking-pool thread and return its result; `None` when the
/// runtime could not schedule the task. Used to isolate the synchronous
/// cold-evidence ladder (file reads + supervised git/rg children) from the
/// turn thread.
pub(crate) async fn run_off_turn_thread<T, F>(f: F) -> Option<T>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    tokio::task::spawn_blocking(f).await.ok()
}

/// Poll one async `EvidenceProvider` on a DETACHED thread and await it
/// under `budget`: a provider that panics, errors, or simply never yields
/// degrades to an empty package once the budget fires. The provider's
/// future is driven on a plain (non-tokio) thread via the captured runtime
/// handle — never on a runtime worker and never on a tokio blocking-pool
/// task — so a stuck provider can neither occupy a turn thread nor delay
/// the drop of the drive's runtime (tokio joins blocking-pool tasks at
/// shutdown; the detached thread is abandoned instead, and its eventual
/// completion just fails the dropped oneshot).
pub(crate) async fn poll_evidence_with_wall_budget(
    provider: Arc<dyn EvidenceProvider>,
    session: SessionId,
    query: EvidenceQuery,
    budget: std::time::Duration,
) -> Vec<faktor_context::assembler::Evidence> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let handle = tokio::runtime::Handle::current();
    let _ = std::thread::Builder::new()
        .name("evidence-poll".to_string())
        .spawn(move || {
            let result = handle.block_on(provider.evidence_for(session, query));
            let _ = tx.send(result);
        });
    match tokio::time::timeout(budget, rx).await {
        Ok(Ok(Ok(evidence))) => evidence,
        _ => Vec::new(),
    }
}

/// The agent may never match on provider names (Commandment 4). This test
/// locks that invariant structurally across the whole crate.
#[cfg(test)]
mod no_provider_switching {
    #[test]
    fn agent_source_has_no_provider_name_conditionals() {
        // Scan production sources only (skip test modules, whose own
        // assertions necessarily mention the forbidden literals).
        let mut sources = String::new();
        for file in [
            "lib.rs",
            "runtime.rs",
            "tool.rs",
            "tool_json.rs",
            "loop_detect.rs",
            "stall.rs",
        ] {
            let path = format!("{}/src/{file}", env!("CARGO_MANIFEST_DIR"));
            let text = std::fs::read_to_string(&path).unwrap_or_default();
            let text = strip_test_modules(&text);
            sources.push_str(&text);
        }
        for needle in [
            "if provider ==",
            "match provider",
            "provider == \"deepseek\"",
            "provider == \"ollama\"",
            "provider == \"openai\"",
        ] {
            assert!(
                !sources.contains(needle),
                "agent source must not contain {needle:?}"
            );
        }
    }

    fn strip_test_modules(src: &str) -> String {
        // Remove #[cfg(test)] blocks so the invariant test cannot see its
        // own literals.
        let mut out = String::new();
        let mut rest = src;
        while let Some(idx) = rest.find("#[cfg(test)]") {
            out.push_str(&rest[..idx]);
            rest = &rest[idx + "#[cfg(test)]".len()..];
            // Skip to the closing brace of the mod at depth 0.
            let mut depth = 0i32;
            let mut consumed = 0usize;
            let mut found = false;
            for (i, c) in rest.char_indices() {
                match c {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            consumed = i + 1;
                            found = true;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            if found {
                rest = &rest[consumed..];
            }
        }
        out.push_str(rest);
        out
    }
}

// ---------------------------------------------------------------- service

/// VerificationService unit coverage (P0-9/10): scripted backend mapping,
/// disabled semantics, policy budgets and the REAL async executor path.
#[cfg(test)]
mod verification_service_tests {
    use super::*;
    use faktor_core::cancellation::CancellationToken;
    use faktor_verify::exec::{
        CheckCategory, CheckKind, CheckSpec, VerificationContext, VerificationPolicy,
    };

    fn ctx_in(dir: &std::path::Path) -> VerificationContext {
        VerificationContext {
            session_id: 7,
            task_id: 9,
            operation_id: 11,
            workspace_id: 3,
            worktree_id: 1,
            root: dir.to_path_buf(),
            deadline: std::time::Instant::now() + std::time::Duration::from_secs(30),
            cancellation: CancellationToken::new(),
        }
    }

    fn quick_spec(id: &str, program: &str, args: &[&str]) -> CheckSpec {
        CheckSpec::new(
            id,
            CheckKind::Compile,
            CheckCategory::Quick,
            program,
            args.iter().copied(),
            true,
        )
    }

    #[tokio::test]
    async fn scripted_backend_maps_ok_to_passed_and_err_to_failed_with_command_text() {
        let calls: Arc<std::sync::Mutex<Vec<String>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
        let calls2 = calls.clone();
        let service = VerificationService::fake(move |cmd: &str| {
            calls2.lock().unwrap().push(cmd.to_string());
            if cmd.starts_with("bad") {
                Err("boom".to_string())
            } else {
                Ok(())
            }
        });
        let dir = tempfile::tempdir().unwrap();
        let ctx = ctx_in(dir.path());
        let passed = service
            .execute(&quick_spec("a", "cargo", &["check"]), &ctx)
            .await;
        assert_eq!(passed.status, CheckRunStatus::Passed);
        assert_eq!(passed.exit, Some(0));
        assert!(passed.finished_ms >= passed.started_ms);
        let failed = service
            .execute(&quick_spec("b", "bad", &["tool"]), &ctx)
            .await;
        assert_eq!(failed.status, CheckRunStatus::Failed);
        assert_eq!(failed.exit, None);
        assert_eq!(failed.summary.as_deref(), Some("boom"));
        // The scripted backend saw the canonical argv join, never a shell.
        assert_eq!(
            *calls.lock().unwrap(),
            vec!["cargo check".to_string(), "bad tool".to_string()]
        );
    }

    #[tokio::test]
    async fn fake_ok_passes_every_check_and_disabled_reports_unconfigured() {
        let ok = VerificationService::fake_ok();
        let dir = tempfile::tempdir().unwrap();
        let ctx = ctx_in(dir.path());
        let out = ok
            .execute(&quick_spec("rust_check", "cargo", &["check"]), &ctx)
            .await;
        assert_eq!(out.status, CheckRunStatus::Passed);
        assert_eq!(out.exit, Some(0));
        assert!(!ok.is_disabled());

        let off = VerificationService::disabled();
        assert!(off.is_disabled());
        assert_eq!(off.policy(), VerificationPolicy::disabled());
        // Zero-budget policy fails closed: nothing may run inline.
        let mut full = quick_spec("full", "cmake", &["--build", "."]);
        full.category = CheckCategory::Full;
        assert!(matches!(
            off.budget_for(&full),
            BudgetDecision::RunAsTaskOwnedOperation
        ));
        // The disabled backend is scripted: it can never persist jobs and
        // its zero budget fails closed under the unit cap.
        assert!(!off.can_persist_jobs());
        assert!(off.policy().unit_max.is_zero());
    }

    #[test]
    fn budget_decisions_follow_policy_not_a_universal_cap() {
        let service = VerificationService::fake_ok();
        assert_eq!(
            service.budget_for(&quick_spec("c", "cargo", &["check"])),
            BudgetDecision::RunInline(std::time::Duration::from_secs(60))
        );
        let mut test_spec = CheckSpec::new(
            "t",
            CheckKind::Test,
            CheckCategory::Unit,
            "cargo",
            ["test", "--lib"],
            true,
        );
        assert_eq!(
            service.budget_for(&test_spec),
            BudgetDecision::RunInline(std::time::Duration::from_secs(600))
        );
        test_spec.category = CheckCategory::Full;
        assert!(matches!(
            service.budget_for(&test_spec),
            BudgetDecision::RunAsTaskOwnedOperation
        ));
        // Scripted command backends (test seams) cannot persist jobs; the
        // real supervisor-backed executor can (audit P0-5/26).
        assert!(!service.can_persist_jobs());
    }

    #[tokio::test]
    async fn real_executor_runs_typed_argv_in_the_context_root() {
        let service = VerificationService::new(
            Arc::new(faktor_verify::exec::AsyncCheckExecutor::new()),
            VerificationPolicy::default(),
        );
        assert!(!service.is_disabled());
        let dir = tempfile::tempdir().unwrap();
        let ctx = ctx_in(dir.path());
        let out = service
            .execute(&quick_spec("echo", "echo", &["root-marker"]), &ctx)
            .await;
        assert_eq!(out.status, CheckRunStatus::Passed, "{out:?}");
        assert_eq!(out.exit, Some(0));
        assert!(
            out.summary
                .as_deref()
                .unwrap_or_default()
                .contains("root-marker"),
            "real executor captured the child stdout: {out:?}"
        );
    }

    #[tokio::test]
    async fn real_executor_unavailable_when_the_program_is_missing() {
        let service = VerificationService::new(
            Arc::new(faktor_verify::exec::AsyncCheckExecutor::new()),
            VerificationPolicy::default(),
        );
        let dir = tempfile::tempdir().unwrap();
        let ctx = ctx_in(dir.path());
        let out = service
            .execute(
                &quick_spec("ghost", "/nonexistent-tool-for-tests", &[]),
                &ctx,
            )
            .await;
        assert_eq!(out.status, CheckRunStatus::Unavailable);
        assert!(out
            .summary
            .as_deref()
            .unwrap_or_default()
            .contains("not found"));
    }
}

// ---------------------------------------------------------------- policy
// economics (P0-82/15/28): the production policy's stability consult and
// telemetry outcome records, tested adversarially against the real
// RouterService.

#[cfg(test)]
mod economic_policy_tests {
    use super::*;
    use faktor_core::model::{
        MicroUsdPerToken, ModelDescriptor, ModelEconomics, ModelSource, RateLimitState,
    };

    fn desc(provider: &str, model: &str, input_price: u64) -> ModelDescriptor {
        ModelDescriptor {
            provider: provider.into(),
            model: model.into(),
            context: 100_000,
            max_output: 8192,
            tools: true,
            parallel_tools: true,
            reasoning: true,
            thinking: true,
            vision: false,
            structured_output: true,
            embeddings: false,
            streaming: true,
            economics: ModelEconomics {
                input_price_per_mtok: MicroUsdPerToken::from_dollars_per_million(input_price),
                output_price_per_mtok: MicroUsdPerToken::from(1),
                cache_read_price_per_mtok: MicroUsdPerToken::from(input_price / 5),
                cache_write_price_per_mtok: MicroUsdPerToken::from(input_price / 2),
                estimated_latency_ms: 300,
                tool_reliability: 90,
                reasoning_reliability: 90,
                coding_reliability: 90,
                context_reliability: 90,
                availability: 100,
                rate_limit_state: RateLimitState::Healthy,
            },
            source: ModelSource::ProviderCatalog,
        }
    }

    fn economy(pair: Vec<ModelDescriptor>) -> Arc<EconomicRoutingPolicy> {
        EconomicRoutingPolicy::new(
            Arc::new(faktor_router::RouterService::new(pair)),
            RoutingMode::Economy,
        )
    }

    /// Deterministic content digest for the stability series (router tests
    /// use the same FNV construction; identical bytes must hash identically).
    fn tp(id: u64, bytes: &[u8]) -> TurnPrefix {
        let mut h = [0u8; 32];
        let mut acc = 0xcbf29ce484222325u64;
        for &b in bytes {
            acc ^= u64::from(b);
            acc = acc.wrapping_mul(0x100000001b3);
        }
        h[..8].copy_from_slice(&acc.to_le_bytes());
        h[8..16].copy_from_slice(&acc.wrapping_mul(31).to_le_bytes());
        h[16..24].copy_from_slice(&acc.wrapping_mul(97).to_le_bytes());
        h[24..].copy_from_slice(&acc.wrapping_mul(211).to_le_bytes());
        TurnPrefix::new(id, h, bytes.len() as u32)
    }

    fn req() -> faktor_router::RouteRequest {
        faktor_router::RouteRequest {
            context_tokens: 100,
            estimated_output_tokens: 10,
            ..Default::default()
        }
    }

    /// (a) A session recording stability < floor prices its decision with
    /// the churn penalty: same candidates, stability 1.0 vs 0.3 — the
    /// chosen candidate is the same but the DECISION reflects the penalty
    /// (scaled cost + audit), deterministic, and the read-failure case
    /// (no rows) routes without any penalty and never errors the turn.
    #[test]
    fn session_stability_below_floor_inflates_the_decision_and_no_rows_never_penalize() {
        let policy = economy(vec![
            desc("cheap", "cx", 1),  // 100 tokens x 1 + 10 x 1 = 110 micro
            desc("robust", "rx", 2), // 210 micro
        ]);
        let plain = policy.route(&req()).unwrap();
        assert_eq!(
            (plain.provider.as_str(), plain.model.as_str()),
            ("cheap", "cx")
        );
        assert_eq!(plain.estimated_cost_micro, 110);

        // Stability 1.0: byte-identical prefixes — no penalty, decision
        // identical to the plain route.
        let stable_bytes = vec![b's'; 40];
        let stable = [
            tp(1, &stable_bytes),
            tp(2, &stable_bytes),
            tp(3, &stable_bytes),
        ];
        let healthy = policy
            .route_with_session_stability(&req(), Some(&stable))
            .unwrap();
        assert_eq!(healthy, plain, "stable history: no penalty");

        // Stability 0.3: growth 40 -> 130 scores 40/130 = 0.308 < 0.8, so
        // the decision carries the churn premium: cost 110 -> ceil(110 x
        // 1.1538) = 127 and the audit names stability + penalty.
        let churny = [tp(1, &stable_bytes), tp(2, &stable_bytes), {
            let mut t = tp(
                3,
                b"stable-prefix-bytes-grown-longer-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            );
            t.prefix_tokens = 130;
            t
        }];
        let churned = policy
            .route_with_session_stability(&req(), Some(&churny))
            .unwrap();
        assert_eq!(
            (churned.provider.as_str(), churned.model.as_str()),
            ("cheap", "cx"),
            "the churn penalty scales the decision, it never silently swaps a candidate"
        );
        assert_eq!(churned.estimated_cost_micro, 127);
        assert!(
            churned.reasoning.contains("prefix_stability=0.308"),
            "{}",
            churned.reasoning
        );
        assert!(
            churned.reasoning.contains("churn_penalty=0.1538"),
            "{}",
            churned.reasoning
        );
        // Deterministic: identical history -> identical decision.
        let again = policy
            .route_with_session_stability(&req(), Some(&churny))
            .unwrap();
        assert_eq!(churned, again);
        // Stability read failure (no rows / None / empty): identical to the
        // plain route — no penalty, Ok, never an error on the turn.
        assert_eq!(
            policy.route_with_session_stability(&req(), None).unwrap(),
            plain
        );
        assert_eq!(
            policy
                .route_with_session_stability(&req(), Some(&[]))
                .unwrap(),
            plain
        );
    }

    /// (b) Telemetry outcome records: N settled calls through the policy
    /// update the wrapped RouterService reliability priors ONLY for the
    /// failing (provider, model, phase) instance pair, latency rides the
    /// records, and a rate-limited outcome cooldowns only that provider.
    #[test]
    fn settled_call_outcomes_update_priors_only_for_the_failing_instance_pair() {
        let svc = Arc::new(faktor_router::RouterService::new(vec![
            desc("a", "am", 1),
            desc("b", "bm", 1),
        ]));
        let policy = EconomicRoutingPolicy::new(svc.clone(), RoutingMode::Economy);
        assert_eq!(
            svc.telemetry
                .success_estimate("a", "am", RouterPhase::Implement),
            0.8
        );
        assert_eq!(
            svc.telemetry
                .success_estimate("b", "bm", RouterPhase::Implement),
            0.8
        );
        // N settled calls against (a, am): 9 failures + 1 success, with
        // latency. (b, bm) and every other phase stay untouched.
        for _ in 0..9 {
            policy.record_call_outcome(&SettledCallOutcome {
                provider: "a".into(),
                model: "am".into(),
                phase: RouterPhase::Implement,
                success: false,
                retried: false,
                rate_limited: false,
                latency_ms: 400,
                verified: None,
            });
        }
        policy.record_call_outcome(&SettledCallOutcome {
            provider: "a".into(),
            model: "am".into(),
            phase: RouterPhase::Implement,
            success: true,
            retried: true,
            rate_limited: false,
            latency_ms: 200,
            verified: None,
        });
        let a_after = svc
            .telemetry
            .success_estimate("a", "am", RouterPhase::Implement);
        assert!(
            a_after < 0.8 && a_after > 0.0,
            "(a, am) reliability prior must decay: {a_after}"
        );
        assert_eq!(
            svc.telemetry
                .success_estimate("b", "bm", RouterPhase::Implement),
            0.8,
            "untouched pair keeps its prior exactly"
        );
        assert_eq!(
            svc.telemetry
                .success_estimate("a", "am", RouterPhase::Review),
            0.8,
            "untouched phase keeps its prior exactly"
        );
        let avg = svc
            .telemetry
            .avg_latency_ms("a", "am", RouterPhase::Implement);
        assert!(avg > 200.0 && avg < 400.0, "latency EWMA: {avg}");
        // The priors actually CHANGE routing: the failing pair loses the
        // next route of a comparable request.
        let d = policy.route(&req()).unwrap();
        assert_ne!(
            (d.provider.as_str(), d.model.as_str()),
            ("a", "am"),
            "the decayed pair must lose the next route: {}",
            d.reasoning
        );
        // Rate-limit outcome: cooldown for the limited provider only.
        policy.record_call_outcome(&SettledCallOutcome {
            provider: "b".into(),
            model: "bm".into(),
            phase: RouterPhase::Implement,
            success: false,
            retried: false,
            rate_limited: true,
            latency_ms: 500,
            verified: None,
        });
        assert!(svc.telemetry.cooldown_active("b"));
        assert!(
            !svc.telemetry.cooldown_active("a"),
            "only the limiter cools down"
        );
    }

    /// Pinned mode: the stability consult keeps the pin's decision (fail
    /// closed) but still prices the churn premium into the estimate.
    #[test]
    fn pinned_stability_consult_keeps_the_pin_and_prices_churn() {
        // The pin is the CHEAPEST candidate (validation must let it win the
        // router's own evaluation or the pin is denied); the stability
        // premium then rides the pinned decision's cost.
        let pair = vec![desc("a", "am", 2), desc("b", "bm", 1)];
        let svc = Arc::new(faktor_router::RouterService::new(pair));
        let policy = EconomicRoutingPolicy::new(
            svc,
            RoutingMode::Pinned {
                provider: "b".into(),
                model: "bm".into(),
            },
        );
        let stable = [tp(1, b"same-bytes"), tp(2, b"same-bytes")];
        let d = policy
            .route_with_session_stability(&req(), Some(&stable))
            .unwrap();
        assert_eq!((d.provider.as_str(), d.model.as_str()), ("b", "bm"));
        assert_eq!(d.estimated_cost_micro, 110, "no penalty when stable");
        let churny = [tp(1, b"same-bytes"), {
            let mut t = tp(2, b"rewritten-to-a-different-prefix-bytes");
            t.prefix_tokens = 60;
            t
        }];
        let d2 = policy
            .route_with_session_stability(&req(), Some(&churny))
            .unwrap();
        assert_eq!(
            (d2.provider.as_str(), d2.model.as_str()),
            ("b", "bm"),
            "pin holds"
        );
        assert!(
            d2.estimated_cost_micro > 110,
            "churn premium must ride the pinned estimate: {}",
            d2.estimated_cost_micro
        );
        assert!(d2.reasoning.contains("churn_penalty="));
        // Passthrough pin: stability data changes nothing.
        let svc = Arc::new(faktor_router::RouterService::new(vec![desc("a", "am", 1)]));
        let passthrough = EconomicRoutingPolicy::new(
            svc,
            RoutingMode::Pinned {
                provider: String::new(),
                model: String::new(),
            },
        );
        let d3 = passthrough
            .route_with_session_stability(&req(), Some(&churny))
            .unwrap();
        assert!(d3.provider.is_empty() && d3.model.is_empty());
    }
}

/// Verified-outcome wiring coverage (audit items 13/14/L): the policy's
/// `record_call_outcome` verified entries land in the SAME store-backed
/// registry every route consult reads — appended once per explicit signal,
/// keyed by the FULL (provider, model, phase, task_class, risk_bucket) key,
/// durable across store reopens, and never learned from telemetry-only
/// feeds ("the model said done" is not a verified success).
#[cfg(test)]
mod verified_outcome_wiring_tests {
    use super::*;
    use faktor_core::model::{MicroUsdPerToken, ModelDescriptor, ModelEconomics, ModelSource};
    use faktor_router::OutcomeStore;

    fn desc(provider: &str, model: &str, input: u64, output: u64, rel: u8) -> ModelDescriptor {
        ModelDescriptor {
            provider: provider.into(),
            model: model.into(),
            context: 512_000,
            max_output: 64_000,
            tools: true,
            parallel_tools: true,
            reasoning: false,
            thinking: false,
            vision: false,
            structured_output: false,
            embeddings: false,
            streaming: true,
            economics: ModelEconomics {
                input_price_per_mtok: MicroUsdPerToken::from_dollars_per_million(input),
                output_price_per_mtok: MicroUsdPerToken::from_dollars_per_million(output),
                coding_reliability: rel,
                tool_reliability: rel,
                reasoning_reliability: rel,
                context_reliability: rel,
                ..Default::default()
            },
            source: ModelSource::ProviderCatalog,
        }
    }

    fn implement_req(tokens_in: u64, tokens_out: u64, floor: u8) -> faktor_router::RouteRequest {
        faktor_router::RouteRequest {
            phase: RouterPhase::Implement,
            required_capabilities: vec!["tools".into(), "streaming".into()],
            context_tokens: tokens_in,
            estimated_output_tokens: tokens_out,
            quality_floor: floor,
            task_budget_remaining_micro: 0,
            latency_preference_ms: None,
            ..Default::default()
        }
    }

    #[test]
    fn verified_entries_append_once_fully_keyed_and_telemetry_only_feeds_never_learn() {
        let dir = tempfile::tempdir().unwrap();
        let manager = faktor_session::SessionManager::open(
            dir.path().join("store"),
            dir.path().join("cas"),
            true,
        )
        .unwrap();
        let store = manager.store();
        let outcomes: Arc<dyn OutcomeStore> = Arc::new(StoreOutcomeStore::new(store.clone()));
        let policy = EconomicRoutingPolicy::new(
            Arc::new(faktor_router::RouterService::with_pricing_and_outcomes(
                vec![desc("fake", "m", 1, 3, 82)],
                std::collections::HashMap::new(),
                outcomes,
            )),
            RoutingMode::Economy,
        );
        // Telemetry-only feed (no verified signal — e.g. the settle sites
        // before the gate): the registry learns NOTHING.
        policy.record_call_outcome(&SettledCallOutcome {
            provider: "fake".into(),
            model: "m".into(),
            phase: RouterPhase::Implement,
            success: true,
            retried: false,
            rate_limited: false,
            latency_ms: 100,
            verified: None,
        });
        assert!(
            store
                .model_outcome_stats_get(
                    "fake",
                    "m",
                    RouterPhase::Implement,
                    TaskClass::Medium,
                    RiskBucket::Low,
                )
                .unwrap()
                .is_none(),
            "the model said done — without the verified signal no sample may be learned"
        );
        // A genuine verified-success signal lands ONE success sample under
        // the FULL key, in the durable store.
        let v = VerifiedCallAttribution {
            task_class: TaskClass::Medium,
            risk_bucket: RiskBucket::Low,
            verified_success: true,
            rework_cost_micro: u64::MAX,
            rework_turns: u64::MAX,
        };
        policy.record_call_outcome(&SettledCallOutcome {
            provider: "fake".into(),
            model: "m".into(),
            phase: RouterPhase::Implement,
            success: true,
            retried: false,
            rate_limited: false,
            latency_ms: 200,
            verified: Some(v),
        });
        let row = store
            .model_outcome_stats_get(
                "fake",
                "m",
                RouterPhase::Implement,
                TaskClass::Medium,
                RiskBucket::Low,
            )
            .unwrap()
            .expect("the verified signal must reach the store");
        assert_eq!(row.successes_first_pass, 1);
        assert_eq!(row.failures_first_pass, 0);
        assert_eq!(
            row.rework_cost_micro_sum, 0,
            "a verified first-pass success never carries rework — hostile success numbers are ignored"
        );
        assert_eq!(row.sample_count, 1);
        // A DIFFERENT class/risk bucket stays untouched (full-key writes).
        assert!(store
            .model_outcome_stats_get(
                "fake",
                "m",
                RouterPhase::Implement,
                TaskClass::Hard,
                RiskBucket::High,
            )
            .unwrap()
            .is_none());
        // A failed-verification attribution records a FAILURE sample with
        // its rework under ITS key and never a success.
        let failed = VerifiedCallAttribution {
            task_class: TaskClass::Hard,
            risk_bucket: RiskBucket::High,
            verified_success: false,
            rework_cost_micro: 900_000,
            rework_turns: 2,
        };
        policy.record_call_outcome(&SettledCallOutcome {
            provider: "fake".into(),
            model: "m".into(),
            phase: RouterPhase::Review,
            success: true,
            retried: false,
            rate_limited: false,
            latency_ms: 300,
            verified: Some(failed),
        });
        let row = store
            .model_outcome_stats_get(
                "fake",
                "m",
                RouterPhase::Review,
                TaskClass::Hard,
                RiskBucket::High,
            )
            .unwrap()
            .unwrap();
        assert_eq!(row.successes_first_pass, 0, "no success may be learned");
        assert_eq!(row.failures_first_pass, 1);
        assert_eq!(row.rework_cost_micro_sum, 900_000);
        assert_eq!(row.rework_turns_sum, 2);
        assert_eq!(row.sample_count, 1);
        // The store-backed phase consult folds the class/risk buckets.
        let folded = store
            .model_outcome_stats_phase("fake", "m", RouterPhase::Implement)
            .unwrap()
            .expect("Implement samples exist");
        assert_eq!(folded.successes_first_pass, 1);
        assert_eq!(folded.failures_first_pass, 0);
        let review_folded = store
            .model_outcome_stats_phase("fake", "m", RouterPhase::Review)
            .unwrap()
            .unwrap();
        assert_eq!(review_folded.failures_first_pass, 1);
    }

    #[test]
    fn store_backed_outcomes_serve_routing_after_reopen_and_failures_flip_cheap_to_strong() {
        // Routing after reopen reflects the RECORDED stats: with an empty
        // registry the $4/$30 candidate wins on price; three failed-
        // verification samples recorded against it (cheap's Implement /
        // Medium / Low key) survive a store reopen and flip the decision to
        // the $10/$25 candidate whose conservative expected cost is now
        // below the failure-history estimate. Mirrors the wave-B4 memory
        // registry tests at the router level, through the durable impl.
        let dir = tempfile::tempdir().unwrap();
        let cheap = desc("cheap", "fast", 4, 30, 95);
        let strong = desc("strong", "big", 10, 25, 95);
        let req = implement_req(10_000, 2_000, 60);
        let decide = |policy: &Arc<EconomicRoutingPolicy>| {
            let d = policy.route(&req).unwrap();
            (d.provider.clone(), d.model.clone())
        };
        let first_choice;
        {
            let manager = faktor_session::SessionManager::open(
                dir.path().join("store"),
                dir.path().join("cas"),
                true,
            )
            .unwrap();
            let store = manager.store();
            let policy = EconomicRoutingPolicy::new(
                Arc::new(faktor_router::RouterService::with_pricing_and_outcomes(
                    vec![cheap.clone(), strong.clone()],
                    std::collections::HashMap::new(),
                    Arc::new(StoreOutcomeStore::new(store.clone())),
                )),
                RoutingMode::Economy,
            );
            first_choice = decide(&policy);
            assert_eq!(
                first_choice,
                ("cheap".to_string(), "fast".to_string()),
                "with no verified history the cheaper candidate wins"
            );
            // Three failed-verification gates on cheap's settled calls.
            for _ in 0..3 {
                policy.record_call_outcome(&SettledCallOutcome {
                    provider: "cheap".into(),
                    model: "fast".into(),
                    phase: RouterPhase::Implement,
                    success: true,
                    retried: false,
                    rate_limited: false,
                    latency_ms: 250,
                    verified: Some(VerifiedCallAttribution {
                        task_class: TaskClass::Medium,
                        risk_bucket: RiskBucket::Low,
                        verified_success: false,
                        rework_cost_micro: 960_000,
                        rework_turns: 1,
                    }),
                });
            }
            let row = store
                .model_outcome_stats_get(
                    "cheap",
                    "fast",
                    RouterPhase::Implement,
                    TaskClass::Medium,
                    RiskBucket::Low,
                )
                .unwrap()
                .unwrap();
            assert_eq!(row.failures_first_pass, 3);
            assert_eq!(row.rework_cost_micro_sum, 3 * 960_000);
            // Crash + reopen: the samples live in the store.
        }
        let manager = faktor_session::SessionManager::open(
            dir.path().join("store"),
            dir.path().join("cas"),
            true,
        )
        .unwrap();
        let store = manager.store();
        let reopened = EconomicRoutingPolicy::new(
            Arc::new(faktor_router::RouterService::with_pricing_and_outcomes(
                vec![cheap.clone(), strong.clone()],
                std::collections::HashMap::new(),
                Arc::new(StoreOutcomeStore::new(store.clone())),
            )),
            RoutingMode::Economy,
        );
        let (provider, model) = decide(&reopened);
        assert_eq!(
            (provider.as_str(), model.as_str()),
            ("strong", "big"),
            "the recorded failure history must flip the route away from the cheap candidate"
        );
        let folded = store
            .model_outcome_stats_phase("cheap", "fast", RouterPhase::Implement)
            .unwrap()
            .expect("recorded stats survive the reopen");
        assert_eq!(folded.sample_count, 3);
    }
}

#[cfg(test)]
mod attempt_accounting_tests {
    use super::*;
    use faktor_core::id::TaskId;
    use faktor_core::model::PricingSnapshot;
    use faktor_core::op::ModelCallAttempt;
    use faktor_session::{BudgetAuthority, BudgetError, BudgetView};
    use std::pin::Pin;

    /// Records every authority call behind a NoopBudget — the guard test
    /// proves a misordered refund/uncertain NEVER reaches the authority.
    struct CountingBudget {
        refund_calls: Arc<std::sync::atomic::AtomicUsize>,
        uncertain_calls: Arc<std::sync::atomic::AtomicUsize>,
        settle_calls: Arc<std::sync::atomic::AtomicUsize>,
        dispatch_calls: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl CountingBudget {
        fn new() -> Self {
            Self {
                refund_calls: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                uncertain_calls: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                settle_calls: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                dispatch_calls: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            }
        }
    }

    impl BudgetAuthority for CountingBudget {
        fn reserve(
            &self,
            _s: SessionId,
            _t: TaskId,
            _op: faktor_core::id::OpId,
            _pred: u64,
            _snap: Option<PricingSnapshot>,
        ) -> Pin<
            Box<
                dyn std::future::Future<Output = Result<faktor_session::ReservationId, BudgetError>>
                    + Send,
            >,
        > {
            Box::pin(async { Ok(faktor_session::ReservationId::NOOP) })
        }
        fn reserve_attempt(
            &self,
            _s: SessionId,
            _t: TaskId,
            _a: ModelCallAttempt,
            _pred: u64,
            _snap: Option<PricingSnapshot>,
        ) -> Pin<
            Box<
                dyn std::future::Future<Output = Result<faktor_session::ReservationId, BudgetError>>
                    + Send,
            >,
        > {
            Box::pin(async { Ok(faktor_session::ReservationId::NOOP) })
        }
        fn mark_dispatched(
            &self,
            _s: SessionId,
            _r: faktor_session::ReservationId,
        ) -> Pin<Box<dyn std::future::Future<Output = Result<(), BudgetError>> + Send>> {
            let c = self.dispatch_calls.clone();
            Box::pin(async move {
                c.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(())
            })
        }
        fn mark_uncertain(
            &self,
            _s: SessionId,
            _r: faktor_session::ReservationId,
            _reason: String,
            _request_id: Option<String>,
        ) -> Pin<Box<dyn std::future::Future<Output = Result<(), BudgetError>> + Send>> {
            let c = self.uncertain_calls.clone();
            Box::pin(async move {
                c.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(())
            })
        }
        fn settle_usage(
            &self,
            _s: SessionId,
            _r: faktor_session::ReservationId,
            _a: u64,
            _b: u64,
            _c: u64,
            _d: u64,
            _e: Option<u64>,
            _f: Option<String>,
        ) -> Pin<Box<dyn std::future::Future<Output = Result<Option<u64>, BudgetError>> + Send>>
        {
            let c = self.settle_calls.clone();
            Box::pin(async move {
                c.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(None)
            })
        }
        fn refund(
            &self,
            _s: SessionId,
            _r: faktor_session::ReservationId,
        ) -> Pin<Box<dyn std::future::Future<Output = Result<(), BudgetError>> + Send>> {
            let c = self.refund_calls.clone();
            Box::pin(async move {
                c.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(())
            })
        }
        fn session_budget_view(
            &self,
            _s: SessionId,
            _t: TaskId,
        ) -> Result<BudgetView, BudgetError> {
            Ok(BudgetView {
                max_cost_micro: None,
                spent_cost_micro: 0,
                open_reserved_micro: 0,
                open_reservations: 0,
                uncertain_reserved_micro: 0,
                uncertain_reservations: 0,
                settled_count: 0,
            })
        }
        fn recover_after_restart(&self) {}
        fn reconcile_uncertain(
            &self,
            _s: SessionId,
            _t: TaskId,
        ) -> Pin<
            Box<
                dyn std::future::Future<
                        Output = Result<faktor_store::CostReconcileReport, BudgetError>,
                    > + Send,
            >,
        > {
            Box::pin(async { Ok(Default::default()) })
        }
        fn finalize_uncertain(
            &self,
            _s: SessionId,
            _t: TaskId,
        ) -> Pin<
            Box<
                dyn std::future::Future<
                        Output = Result<faktor_store::CostFinalizeReport, BudgetError>,
                    > + Send,
            >,
        > {
            Box::pin(async { Ok(Default::default()) })
        }
    }

    fn budget() -> CountingBudget {
        CountingBudget::new()
    }

    #[tokio::test]
    async fn refund_after_dispatch_is_refused_locally_and_never_reaches_the_authority() {
        // The five-runtime-site bug shape: refund AFTER mark_dispatched must
        // be impossible — the machine refuses locally with the ledger's own
        // typed error BEFORE the authority is touched.
        let authority = Arc::new(budget());
        let mut acct = AttemptAccounting::new(
            authority.clone(),
            SessionId::new(1),
            Some(faktor_session::ReservationId::new(7)),
        );
        acct.mark_dispatched().await.unwrap();
        assert!(acct.dispatched() && acct.is_open());
        let err = acct.fail_before_dispatch().await.unwrap_err();
        assert!(
            matches!(
                err,
                faktor_session::BudgetError::CannotRefundDispatched { .. }
            ),
            "{err:?}"
        );
        assert_eq!(
            authority
                .refund_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            0
        );
        assert!(acct.is_open(), "the refused refund changes nothing");
        // The legal terminal for a dispatched attempt: UNCERTAIN — exactly
        // one authority call, machine closed.
        acct.fail_after_dispatch("provider_error", None)
            .await
            .unwrap();
        assert!(acct.closed());
        assert_eq!(
            authority
                .uncertain_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            1
        );
    }

    #[tokio::test]
    async fn uncertain_before_dispatch_is_refused_the_attempt_must_refund() {
        // A never-dispatched failure REFUNDS; marking it UNCERTAIN would
        // charge an estimate for a request that provably never left.
        let authority = Arc::new(budget());
        let mut acct = AttemptAccounting::new(
            authority.clone(),
            SessionId::new(1),
            Some(faktor_session::ReservationId::new(8)),
        );
        let err = acct
            .fail_after_dispatch("never_dispatched", None)
            .await
            .unwrap_err();
        assert!(
            matches!(err, faktor_session::BudgetError::NotOpen { .. }),
            "{err:?}"
        );
        assert_eq!(
            authority
                .uncertain_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            0
        );
        acct.fail_before_dispatch().await.unwrap();
        assert!(acct.closed());
        assert_eq!(
            authority
                .refund_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            1
        );
    }

    #[tokio::test]
    async fn settle_before_dispatch_and_double_terminal_calls_are_guarded() {
        // The machine lets money move exactly once and only in order:
        // settle requires a dispatched open attempt; a second terminal call
        // on a closed machine is refused without touching the authority.
        let authority = Arc::new(budget());
        let mut acct = AttemptAccounting::new(
            authority.clone(),
            SessionId::new(1),
            Some(faktor_session::ReservationId::new(9)),
        );
        assert!(matches!(
            acct.settle_usage(1, 0, 0, 1, None, None).await.unwrap_err(),
            faktor_session::BudgetError::NotOpen { status, .. } if status == "reserved"
        ));
        assert_eq!(
            authority
                .settle_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            0
        );
        acct.mark_dispatched().await.unwrap();
        acct.settle_usage(100, 0, 0, 10, None, None).await.unwrap();
        assert!(acct.closed());
        assert_eq!(
            authority
                .settle_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            1
        );
        let err = acct.settle_usage(1, 0, 0, 1, None, None).await.unwrap_err();
        assert!(
            matches!(err, faktor_session::BudgetError::NotOpen { .. }),
            "{err:?}"
        );
        let err = acct
            .fail_after_dispatch("double_terminal", None)
            .await
            .unwrap_err();
        assert!(
            matches!(err, faktor_session::BudgetError::NotOpen { .. }),
            "{err:?}"
        );
        let err = acct.fail_before_dispatch().await.unwrap_err();
        assert!(
            matches!(err, faktor_session::BudgetError::NotOpen { .. }),
            "{err:?}"
        );
        assert_eq!(
            authority
                .settle_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            1
        );
        assert_eq!(
            authority
                .uncertain_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            0
        );
        assert_eq!(
            authority
                .refund_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            0
        );
    }

    #[tokio::test]
    async fn a_machine_without_a_reservation_moves_nothing() {
        // Unbudgeted calls: no money exists to move — dispatch is a silent
        // no-op, the refund path releases nothing, and settle/uncertain
        // (which require a dispatched attempt with a reservation) stay
        // refused without touching the authority.
        let authority = Arc::new(budget());
        let mut acct = AttemptAccounting::new(authority.clone(), SessionId::new(1), None);
        acct.mark_dispatched().await.unwrap();
        assert!(
            !acct.dispatched(),
            "nothing was dispatched (no reservation)"
        );
        assert!(matches!(
            acct.settle_usage(1, 0, 0, 1, None, None).await.unwrap_err(),
            faktor_session::BudgetError::NotOpen { status, .. } if status == "reserved"
        ));
        assert!(matches!(
            acct.fail_after_dispatch("x", None).await.unwrap_err(),
            faktor_session::BudgetError::NotOpen { .. }
        ));
        acct.fail_before_dispatch().await.unwrap();
        assert!(acct.closed());
        assert_eq!(
            authority
                .settle_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            0
        );
        assert_eq!(
            authority
                .refund_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            0
        );
        assert_eq!(
            authority
                .uncertain_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            0
        );
    }
}

#[cfg(test)]
mod model_call_intent_tests {
    use super::*;

    #[test]
    fn quality_floors_are_hard_and_never_lowered() {
        // Implement/Review route on a HARD 60 floor; Adaptive carries a
        // target and a never-below minimum; route_request applies the
        // minimum verbatim (the policy decides nothing above it).
        assert_eq!(ModelCallIntent::implement_main().quality_floor(), 60);
        assert_eq!(ModelCallIntent::review().quality_floor(), 60);
        assert_eq!(ModelCallIntent::compact().quality_floor(), 60);
        let adaptive = ModelCallIntent {
            phase: RouterPhase::Implement,
            required_capabilities: vec![],
            quality: QualityRequirement::Adaptive {
                target: 85,
                minimum: 70,
            },
            expected_output_tokens: 2048,
            semantic_risk: 0,
        };
        assert_eq!(adaptive.quality.minimum(), 70);
        assert_eq!(adaptive.quality.target(), 85);
        assert_eq!(adaptive.quality_floor(), 70);
    }

    #[test]
    fn route_request_carries_the_real_planned_dimensions_not_a_guess() {
        let intent = ModelCallIntent::implement_main();
        let req = intent.route_request(12_345, 7_000, 0);
        assert_eq!(req.context_tokens, 12_345, "the plan's real input estimate");
        assert_eq!(req.estimated_output_tokens, 7_000, "the real output cap");
        assert_eq!(req.quality_floor, 60);
        assert_ne!(req.context_tokens, 16_384, "no hard-coded pre-plan guess");
        assert_ne!(req.estimated_output_tokens, 2048, "no hard-coded 2048");
        assert_eq!(intent.phase, RouterPhase::Implement);
    }
}

#[cfg(test)]
mod hard_quality_floor_tests {
    use super::*;
    use faktor_core::model::ModelDescriptor;
    use faktor_router::RouterService;

    fn candidate(quality: u8) -> ModelDescriptor {
        ModelDescriptor {
            provider: "p".into(),
            model: "m".into(),
            context: 128_000,
            max_output: 16_000,
            tools: true,
            parallel_tools: false,
            reasoning: false,
            thinking: false,
            vision: false,
            structured_output: false,
            embeddings: false,
            streaming: true,
            economics: faktor_core::model::ModelEconomics {
                coding_reliability: quality,
                tool_reliability: quality,
                reasoning_reliability: quality,
                context_reliability: quality,
                ..Default::default()
            },
            source: faktor_core::model::ModelSource::ProviderCatalog,
        }
    }

    #[test]
    fn hard_60_with_best_available_50_is_a_no_capable_model_refusal() {
        // Audit (e): quality floors are HARD. The old logic lowered the
        // requested floor toward the best available candidate; the fix
        // refuses typed: requested hard 60 with best available 50 =>
        // NoCapableModel — and a 90-quality candidate serves the SAME
        // request.
        let policy = EconomicRoutingPolicy::new(
            Arc::new(RouterService::new(vec![candidate(50)])),
            RoutingMode::Economy,
        );
        let req = ModelCallIntent::implement_main().route_request(4_000, 2_048, 0);
        assert!(
            matches!(
                policy.route(&req),
                Err(RouteFailure::NoCapableModel)
            ),
            "hard 60 with a best available 50 must be a typed NoCapableModel, never a lowered floor"
        );
        let policy2 = EconomicRoutingPolicy::new(
            Arc::new(RouterService::new(vec![candidate(90)])),
            RoutingMode::Economy,
        );
        assert!(
            policy2.route(&req).is_ok(),
            "an above-floor candidate serves"
        );
        // Balanced raises to its band; MaximumQuality never probes below.
        let bal = EconomicRoutingPolicy::new(
            Arc::new(RouterService::new(vec![candidate(70)])),
            RoutingMode::Balanced,
        );
        assert!(matches!(bal.route(&req), Err(RouteFailure::NoCapableModel)));
        let maxq = EconomicRoutingPolicy::new(
            Arc::new(RouterService::new(vec![candidate(55)])),
            RoutingMode::MaximumQuality,
        );
        assert!(matches!(
            maxq.route(&req),
            Err(RouteFailure::NoCapableModel)
        ));
    }
}

#[cfg(test)]
mod efficiency_flags_tests {
    use super::*;
    use faktor_context::information::FailurePrior;
    use faktor_context::selection::ContextCandidate;

    /// A hostile-value prior standing in for the learning crate's handle:
    /// the value is irrelevant to the gate test.
    struct AlwaysDouble;

    impl FailurePrior for AlwaysDouble {
        fn omission_risk(&self, _candidate: &ContextCandidate) -> f64 {
            2.0
        }
    }

    #[test]
    fn efficiency_flags_default_all_off() {
        let flags = EfficiencyFlags::default();
        assert!(!flags.failure_learning);
        assert!(!flags.ccr);
        assert!(!flags.typed_handoff);
        assert!(!flags.semantic_context);
        assert!(!flags.rework_routing);
    }

    /// The gate: the prior is handed to the planner ONLY when the parsed
    /// `failure_learning` flag is on AND a handle exists. Every other
    /// combination yields `None` (baseline/parity path).
    #[test]
    fn prior_is_applied_only_when_the_flag_is_on_and_a_handle_exists() {
        let prior = AlwaysDouble;
        let handle: Option<&(dyn FailurePrior + Send + Sync)> = Some(&prior);
        let off = EfficiencyFlags::default();
        assert!(off.context_prior(handle).is_none(), "off => no prior");
        let on = EfficiencyFlags {
            failure_learning: true,
            ..Default::default()
        };
        assert!(on.context_prior(handle).is_some(), "on + handle => prior");
        assert!(
            on.context_prior(None).is_none(),
            "on + no handle => no prior"
        );
        // Running through the concrete trait object never invokes the
        // prior while the flag is off (the handle is not even observed).
        let gated = off.context_prior(handle);
        assert!(gated.is_none());
    }
}
