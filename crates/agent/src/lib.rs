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

pub mod loop_detect;
pub mod runtime;
pub mod stall;
pub mod tool;
pub mod tool_json;

pub use faktor_core::model::{RouteDecision, RouterPhase, RoutingMode};
pub use faktor_core::state::{
    CheckExecution, CriterionVerification, FileStateEvidence, OutcomeReason, ReasonCode, TaskState,
    TaskTransition, VerificationStatus,
};
pub use faktor_session::VerificationRecord;
pub use faktor_verify::{Acceptance, Verifier};
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
}

/// The production policy: a real [`faktor_router::RouterService`] under a
/// fixed [`RoutingMode`] decided once at graph build.
///
/// - [`RoutingMode::Economy`]: every request is routed; the decision's
///   provider/model override the session-configured defaults.
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

    fn route_economy(
        &self,
        req: &faktor_router::RouteRequest,
    ) -> Result<RouteDecision, RouteFailure> {
        let floor = effective_floor(
            req.quality_floor,
            req.phase,
            &self.service.router.candidates,
        );
        let mut routed = req.clone();
        routed.quality_floor = floor;
        self.service
            .route(&routed, &[])
            .map_err(|e| self.map_denial(&e, &routed))
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
            RoutingMode::Economy => self.route_economy(req),
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
