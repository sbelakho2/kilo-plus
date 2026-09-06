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

use faktor_router::stability::TurnPrefix;
use faktor_verify::exec::{BudgetDecision, CheckOutcome, CheckRunStatus};

pub mod loop_detect;
pub mod runtime;
pub mod stall;
pub mod tool;
pub mod tool_json;
pub mod wire_plan;

pub use faktor_core::model::{RouteDecision, RouterPhase, RoutingMode};
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
    FilePostcondition, RecoveryHint, ReplayDescriptor, Tool, ToolOutcome, ToolRegistry, ToolRunCtx,
};
pub use tool_json::{parse_tool_calls, repair_json, ToolCallMode};

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
    /// Real executor: typed tokio child processes under the context's
    /// deadline and cancellation (bounded capture, process-group kill).
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
/// - checks whose policy says "task-owned background operation" have no
///   background machinery on the genuine-end path yet (that lands with
///   task-owned operations in a later wave): they run inline under the
///   unit cap and the runtime records the documented override note in the
///   check summary;
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

    /// The inline cap used when [`BudgetDecision::RunAsTaskOwnedOperation`]
    /// is decided but no task-owned background machinery exists on the
    /// calling path yet (P0-10 inline fallback). Zero fails closed: the
    /// caller must NOT run the check.
    pub fn inline_override_budget(&self) -> std::time::Duration {
        self.policy.unit_max
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

/// Why a routed call was refused. Fail-closed matrix (P0-88): only
/// [`RouteFailure::RouterUnavailable`] may fall back to the session's
/// configured model; every other failure is a typed terminal error on the
/// turn — no silent degradation, no half-configured substitution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteFailure {
    /// The routing service is not reachable (no candidates registered, the
    /// policy has no router to consult). The ONE failure that may fall back
    /// to the session's configured provider/model (documented + warned).
    RouterUnavailable,
    /// The request cannot be served within the task's remaining budget.
    BudgetExceeded,
    /// No candidate (or the pinned model) can serve the request:
    /// capabilities, context/output fit or phase quality.
    NoCapableModel,
    /// The policy refused: the pin lost the router's own evaluation, the
    /// request violated a policy constraint, or the router denied for
    /// non-budget reasons.
    PolicyDenied,
    /// Routing telemetry/health machinery failed internally.
    InternalTelemetry,
}

impl RouteFailure {
    /// Conservative fallback is allowed ONLY for [`RouteFailure::RouterUnavailable`]
    /// (P0-88: fail closed on budget/capability/policy denials).
    pub fn may_fallback(&self) -> bool {
        matches!(self, RouteFailure::RouterUnavailable)
    }
}

/// The routing decision authority consumed by the agent runtime before
/// EVERY paid model call (the former "auto"-sentinel path is gone: there is
/// no un-routed model call anymore, and there is no fallback except the one
/// [`RouteFailure::may_fallback`] names).
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

    /// The telemetry outcome record entry (P0-28 residuals): the runtime
    /// calls this after each settled model call with the actual outcome
    /// (attempted/resolved, latency, provider/model, reliability signals).
    /// Default: no telemetry (test policies, no-op hosts).
    fn record_call_outcome(&self, _outcome: &SettledCallOutcome) {}
}

/// One settled model call's outcome, recorded into the router telemetry
/// (provider/model of the ACTUAL settled call — also correct for passthrough
/// decisions — plus the measured latency and reliability signals).
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

/// The quality floor that never denies a whole candidate set spuriously:
/// `min(requested floor, best available phase quality over the candidates)`.
/// With default (unpriced) economics every candidate reports the 50/50/50
/// default profile and the floor collapses to 50; once real per-model
/// economics exist, the requested floor governs.
fn effective_floor(
    requested: u8,
    phase: RouterPhase,
    candidates: &[faktor_core::model::ModelDescriptor],
) -> u8 {
    let best = candidates
        .iter()
        .map(|c| phase_quality(&c.economics, phase))
        .max()
        .unwrap_or(0);
    requested.min(best)
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
        if msg.contains("no candidate clears capability/fit") {
            return RouteFailure::NoCapableModel;
        }
        if msg.contains("quality floor") {
            return RouteFailure::PolicyDenied;
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
    fn consult_at(
        &self,
        req: &faktor_router::RouteRequest,
        floor: u8,
        prefix_history: Option<&[TurnPrefix]>,
    ) -> Result<RouteDecision, String> {
        let mut routed = req.clone();
        routed.quality_floor = floor;
        match prefix_history {
            None => self.service.route(&routed, &[]),
            Some(history) => self.service.route_with_prefix_stability(
                &routed,
                &[],
                faktor_router::stability::DEFAULT_STABILITY_FLOOR,
                Some(history),
            ),
        }
    }

    fn route_economy(
        &self,
        req: &faktor_router::RouteRequest,
        prefix_history: Option<&[TurnPrefix]>,
    ) -> Result<RouteDecision, RouteFailure> {
        let floor = effective_floor(
            req.quality_floor,
            req.phase,
            &self.service.router.candidates,
        );
        self.consult_at(req, floor, prefix_history)
            .map_err(|e| self.map_denial(&e, req))
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
    fn route_maximum_quality(
        &self,
        req: &faktor_router::RouteRequest,
        prefix_history: Option<&[TurnPrefix]>,
    ) -> Result<RouteDecision, RouteFailure> {
        let eff = effective_floor(
            req.quality_floor,
            req.phase,
            &self.service.router.candidates,
        );
        let mut tiers: Vec<u8> = self
            .service
            .router
            .candidates
            .iter()
            .map(|c| phase_quality(&c.economics, req.phase))
            .filter(|&q| q >= eff)
            .collect();
        tiers.sort_unstable();
        tiers.dedup();
        tiers.reverse();
        let mut last_denial: Option<RouteFailure> = None;
        for floor in tiers {
            match self.consult_at(req, floor, prefix_history) {
                Ok(d) => return Ok(d),
                Err(e) => last_denial = Some(self.map_denial(&e, req)),
            }
        }
        last_denial.map_or(Err(RouteFailure::NoCapableModel), Err)
    }

    fn route_balanced(
        &self,
        req: &faktor_router::RouteRequest,
        prefix_history: Option<&[TurnPrefix]>,
    ) -> Result<RouteDecision, RouteFailure> {
        let floor = effective_floor(
            req.quality_floor.max(Self::BALANCED_QUALITY_FLOOR),
            req.phase,
            &self.service.router.candidates,
        );
        self.consult_at(req, floor, prefix_history)
            .map_err(|e| self.map_denial(&e, req))
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
        let floor_eff = effective_floor(
            req.quality_floor,
            req.phase,
            &self.service.router.candidates,
        );
        let pin_quality = phase_quality(&pinned.economics, req.phase);
        if pin_quality < floor_eff {
            // Real per-model quality data exists and the pin is below the
            // best-available bar the request asked for: fail closed, never
            // silently override the request's floor with the pin.
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
        };
        let decision = match self.service.route(&validation, &[]) {
            Ok(d) => Some(d),
            Err(e) => {
                // The pin clears capability/fit/quality/latency by
                // construction; the remaining axis is the budget (or the
                // router is empty — which also denies here).
                let base = faktor_router::estimated_call_cost(
                    &pinned.economics,
                    req.context_tokens,
                    req.estimated_output_tokens,
                    0,
                    0,
                );
                if req.task_budget_remaining_micro > 0 && base > req.task_budget_remaining_micro {
                    return Err(RouteFailure::BudgetExceeded);
                }
                let _ = e;
                return Err(RouteFailure::PolicyDenied);
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

    /// Telemetry outcome record entry (P0-28): every settled model call the
    /// runtime reports lands in the wrapped RouterService's reliability
    /// priors and rate-limit cooldowns (see
    /// `RouterService::record_outcome`).
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
        assert!(off.inline_override_budget().is_zero());
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
        // The P0-10 inline fallback for "background required but no
        // background machinery yet" runs under the unit cap.
        assert_eq!(
            service.inline_override_budget(),
            std::time::Duration::from_secs(600)
        );
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
