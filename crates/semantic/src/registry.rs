//! Provider registry, capability-driven selection and guarded dispatch
//! (audit 48-54/58/59).
//!
//! Selection is by [`SemanticCapabilities`] only — never by provider name,
//! never by a hardcoded language. When no registered provider covers the
//! required operation, the generic fallback serves it, so ordinary operation
//! never fails solely because no semantic provider is installed.
//!
//! Every provider call is wrapped in [`guard_call`]: cancellation and
//! deadline are checked before polling the provider at all, panics are caught
//! and converted to typed errors, and typed provider failures degrade to the
//! fallback (caller-cancellation is propagated, never swallowed).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use faktor_core::{CancellationToken, Clock, Deadline, SystemClock, WorkspaceId};

use crate::fallback::GenericSemanticFallback;
use crate::types::{
    AffectedRequest, AffectedSet, BoxFuture, SemanticCall, SemanticCapabilities,
    SemanticContextPack, SemanticContextRequest, SemanticDelta, SemanticDeltaRequest,
    SemanticEnvelope, SemanticError, SemanticExpectation, SemanticExplainRequest,
    SemanticExplanation, SemanticPayload, SemanticProvider, SemanticProviderId,
    SemanticResponseCaps, SemanticSnapshot, SemanticSnapshotId, SemanticSnapshotRequest,
    SemanticVerification, SemanticVerifyRequest,
};

/// The provider chosen for one operation.
pub enum SemanticSelection<'a> {
    /// A registered provider whose capabilities cover the requirement.
    Provider(&'a dyn SemanticProvider),
    /// The generic in-crate fallback.
    Fallback(&'a GenericSemanticFallback),
}

/// A provider call guarded for cancellation, deadline and panics.
pub struct GuardedCall<'a, T> {
    provider: SemanticProviderId,
    cancellation: CancellationToken,
    deadline: Option<Deadline>,
    clock: Arc<dyn Clock>,
    future: BoxFuture<'a, Result<T, SemanticError>>,
}

/// Wrap one provider call. Cancellation and deadline are observed before
/// every poll of the inner future, so a cancelled or expired call never
/// touches the provider.
pub fn guard_call<'a, T>(
    provider: SemanticProviderId,
    call: SemanticCall,
    clock: Arc<dyn Clock>,
    future: BoxFuture<'a, Result<T, SemanticError>>,
) -> GuardedCall<'a, T> {
    GuardedCall {
        provider,
        cancellation: call.cancellation,
        deadline: call.deadline,
        clock,
        future,
    }
}

impl<T> Future for GuardedCall<'_, T> {
    type Output = Result<T, SemanticError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        if this.cancellation.is_cancelled() {
            return Poll::Ready(Err(SemanticError::Cancelled {
                provider: this.provider.to_string(),
            }));
        }
        if let Some(deadline) = this.deadline {
            if deadline.is_expired(this.clock.now_ms()) {
                return Poll::Ready(Err(SemanticError::DeadlineExceeded {
                    provider: this.provider.to_string(),
                }));
            }
        }
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            this.future.as_mut().poll(cx)
        }));
        match outcome {
            Ok(Poll::Ready(result)) => Poll::Ready(result),
            Ok(Poll::Pending) => Poll::Pending,
            Err(_) => Poll::Ready(Err(SemanticError::ProviderCrashed {
                provider: this.provider.to_string(),
            })),
        }
    }
}

/// Registry of semantic providers plus the always-available generic
/// fallback.
pub struct SemanticProviderRegistry {
    providers: Vec<Arc<dyn SemanticProvider>>,
    fallback: GenericSemanticFallback,
    clock: Arc<dyn Clock>,
    response_caps: SemanticResponseCaps,
}

impl SemanticProviderRegistry {
    pub fn new(fallback: GenericSemanticFallback) -> Self {
        Self {
            providers: Vec::new(),
            fallback,
            clock: Arc::new(SystemClock),
            response_caps: SemanticResponseCaps::default(),
        }
    }

    /// Register a provider. Order is preference order: the first registered
    /// provider covering the requirement wins.
    pub fn register(&mut self, provider: Arc<dyn SemanticProvider>) -> &mut Self {
        self.providers.push(provider);
        self
    }

    pub fn with_clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = clock;
        self
    }

    pub fn with_response_caps(mut self, caps: SemanticResponseCaps) -> Self {
        self.response_caps = caps;
        self
    }

    pub fn providers(&self) -> &[Arc<dyn SemanticProvider>] {
        &self.providers
    }

    pub fn fallback(&self) -> &GenericSemanticFallback {
        &self.fallback
    }

    /// Select by capabilities only. No provider name or language is ever
    /// inspected.
    pub fn select(&self, required: &SemanticCapabilities) -> SemanticSelection<'_> {
        for provider in &self.providers {
            if provider.capabilities().covers(*required) {
                return SemanticSelection::Provider(provider.as_ref());
            }
        }
        SemanticSelection::Fallback(&self.fallback)
    }

    fn validate_response<T: SemanticPayload>(
        &self,
        provider: &SemanticProviderId,
        workspace: WorkspaceId,
        expected_snapshot: SemanticSnapshotId,
        envelope: &SemanticEnvelope<T>,
    ) -> Result<(), SemanticError> {
        envelope.validate_provider_id(provider)?;
        envelope.validate(
            &SemanticExpectation::new(workspace, expected_snapshot),
            &self.response_caps,
        )
    }

    /// The typed refusal of a provider-required call that no provider served
    /// (absent, or a provider failure that would otherwise fall back).
    fn provider_required(&self, op: crate::types::SemanticOp) -> SemanticError {
        SemanticError::ProviderRequired {
            op: op.as_str().to_string(),
        }
    }

    pub fn snapshot(
        &self,
        request: SemanticSnapshotRequest,
    ) -> BoxFuture<'_, Result<SemanticEnvelope<SemanticSnapshot>, SemanticError>> {
        Box::pin(async move {
            match self.select(&SemanticCapabilities::SNAPSHOT) {
                SemanticSelection::Provider(provider) => {
                    let provider_id = provider.id();
                    let attempt = guard_call(
                        provider_id.clone(),
                        request.call.clone(),
                        self.clock.clone(),
                        provider.snapshot(request.clone()),
                    );
                    match attempt.await {
                        Ok(envelope) => match self.validate_response(
                            &provider_id,
                            request.workspace,
                            envelope.snapshot_id,
                            &envelope,
                        ) {
                            Ok(()) => Ok(envelope),
                            Err(err) if request.call.require_provider => Err(err),
                            Err(_) => {
                                let workspace = request.workspace;
                                let envelope = self.fallback.snapshot(request).await?;
                                self.validate_response(
                                    &self.fallback.id(),
                                    workspace,
                                    envelope.snapshot_id,
                                    &envelope,
                                )?;
                                Ok(envelope)
                            }
                        },
                        Err(err) if err.caller_terminal() => Err(err),
                        Err(err) if request.call.require_provider => Err(err),
                        Err(_) => {
                            let workspace = request.workspace;
                            let envelope = self.fallback.snapshot(request).await?;
                            self.validate_response(
                                &self.fallback.id(),
                                workspace,
                                envelope.snapshot_id,
                                &envelope,
                            )?;
                            Ok(envelope)
                        }
                    }
                }
                SemanticSelection::Fallback(fallback) => {
                    if request.call.require_provider {
                        return Err(self.provider_required(crate::types::SemanticOp::Snapshot));
                    }
                    let envelope = guard_call(
                        fallback.id(),
                        request.call.clone(),
                        self.clock.clone(),
                        fallback.snapshot(request.clone()),
                    )
                    .await?;
                    self.validate_response(
                        &self.fallback.id(),
                        request.workspace,
                        envelope.snapshot_id,
                        &envelope,
                    )?;
                    Ok(envelope)
                }
            }
        })
    }

    pub fn context(
        &self,
        request: SemanticContextRequest,
    ) -> BoxFuture<'_, Result<SemanticEnvelope<SemanticContextPack>, SemanticError>> {
        Box::pin(async move {
            request.validate()?;
            match self.select(&SemanticCapabilities::CONTEXT) {
                SemanticSelection::Provider(provider) => {
                    let provider_id = provider.id();
                    let attempt = guard_call(
                        provider_id.clone(),
                        request.call.clone(),
                        self.clock.clone(),
                        provider.context(request.clone()),
                    );
                    match attempt.await {
                        Ok(envelope) => match self.validate_response(
                            &provider_id,
                            request.workspace,
                            request.snapshot_id,
                            &envelope,
                        ) {
                            Ok(()) => Ok(envelope),
                            Err(err) if request.call.require_provider => Err(err),
                            Err(_) => {
                                let workspace = request.workspace;
                                let snapshot_id = request.snapshot_id;
                                let envelope = self.fallback.context(request).await?;
                                self.validate_response(
                                    &self.fallback.id(),
                                    workspace,
                                    snapshot_id,
                                    &envelope,
                                )?;
                                Ok(envelope)
                            }
                        },
                        Err(err) if err.caller_terminal() => Err(err),
                        Err(err) if request.call.require_provider => Err(err),
                        Err(_) => {
                            let workspace = request.workspace;
                            let snapshot_id = request.snapshot_id;
                            let envelope = self.fallback.context(request).await?;
                            self.validate_response(
                                &self.fallback.id(),
                                workspace,
                                snapshot_id,
                                &envelope,
                            )?;
                            Ok(envelope)
                        }
                    }
                }
                SemanticSelection::Fallback(fallback) => {
                    if request.call.require_provider {
                        return Err(self.provider_required(crate::types::SemanticOp::Context));
                    }
                    let envelope = guard_call(
                        fallback.id(),
                        request.call.clone(),
                        self.clock.clone(),
                        fallback.context(request.clone()),
                    )
                    .await?;
                    self.validate_response(
                        &self.fallback.id(),
                        request.workspace,
                        request.snapshot_id,
                        &envelope,
                    )?;
                    Ok(envelope)
                }
            }
        })
    }

    pub fn delta(
        &self,
        request: SemanticDeltaRequest,
    ) -> BoxFuture<'_, Result<SemanticEnvelope<SemanticDelta>, SemanticError>> {
        Box::pin(async move {
            match self.select(&SemanticCapabilities::DELTA) {
                SemanticSelection::Provider(provider) => {
                    let provider_id = provider.id();
                    let attempt = guard_call(
                        provider_id.clone(),
                        request.call.clone(),
                        self.clock.clone(),
                        provider.delta(request.clone()),
                    );
                    match attempt.await {
                        Ok(envelope) => match self.validate_response(
                            &provider_id,
                            request.workspace,
                            envelope.snapshot_id,
                            &envelope,
                        ) {
                            Ok(()) => Ok(envelope),
                            Err(err) if request.call.require_provider => Err(err),
                            Err(_) => {
                                let workspace = request.workspace;
                                let envelope = self.fallback.delta(request).await?;
                                self.validate_response(
                                    &self.fallback.id(),
                                    workspace,
                                    envelope.snapshot_id,
                                    &envelope,
                                )?;
                                Ok(envelope)
                            }
                        },
                        Err(err) if err.caller_terminal() => Err(err),
                        Err(err) if request.call.require_provider => Err(err),
                        Err(_) => {
                            let workspace = request.workspace;
                            let envelope = self.fallback.delta(request).await?;
                            self.validate_response(
                                &self.fallback.id(),
                                workspace,
                                envelope.snapshot_id,
                                &envelope,
                            )?;
                            Ok(envelope)
                        }
                    }
                }
                SemanticSelection::Fallback(fallback) => {
                    if request.call.require_provider {
                        return Err(self.provider_required(crate::types::SemanticOp::Delta));
                    }
                    let envelope = guard_call(
                        fallback.id(),
                        request.call.clone(),
                        self.clock.clone(),
                        fallback.delta(request.clone()),
                    )
                    .await?;
                    self.validate_response(
                        &self.fallback.id(),
                        request.workspace,
                        envelope.snapshot_id,
                        &envelope,
                    )?;
                    Ok(envelope)
                }
            }
        })
    }

    pub fn affected(
        &self,
        request: AffectedRequest,
    ) -> BoxFuture<'_, Result<SemanticEnvelope<AffectedSet>, SemanticError>> {
        Box::pin(async move {
            match self.select(&SemanticCapabilities::AFFECTED) {
                SemanticSelection::Provider(provider) => {
                    let provider_id = provider.id();
                    let attempt = guard_call(
                        provider_id.clone(),
                        request.call.clone(),
                        self.clock.clone(),
                        provider.affected(request.clone()),
                    );
                    match attempt.await {
                        Ok(envelope) => match self.validate_response(
                            &provider_id,
                            request.workspace,
                            request.snapshot_id,
                            &envelope,
                        ) {
                            Ok(()) => Ok(envelope),
                            Err(err) if request.call.require_provider => Err(err),
                            Err(_) => {
                                let workspace = request.workspace;
                                let snapshot_id = request.snapshot_id;
                                let envelope = self.fallback.affected(request).await?;
                                self.validate_response(
                                    &self.fallback.id(),
                                    workspace,
                                    snapshot_id,
                                    &envelope,
                                )?;
                                Ok(envelope)
                            }
                        },
                        Err(err) if err.caller_terminal() => Err(err),
                        Err(err) if request.call.require_provider => Err(err),
                        Err(_) => {
                            let workspace = request.workspace;
                            let snapshot_id = request.snapshot_id;
                            let envelope = self.fallback.affected(request).await?;
                            self.validate_response(
                                &self.fallback.id(),
                                workspace,
                                snapshot_id,
                                &envelope,
                            )?;
                            Ok(envelope)
                        }
                    }
                }
                SemanticSelection::Fallback(fallback) => {
                    if request.call.require_provider {
                        return Err(self.provider_required(crate::types::SemanticOp::Affected));
                    }
                    let envelope = guard_call(
                        fallback.id(),
                        request.call.clone(),
                        self.clock.clone(),
                        fallback.affected(request.clone()),
                    )
                    .await?;
                    self.validate_response(
                        &self.fallback.id(),
                        request.workspace,
                        request.snapshot_id,
                        &envelope,
                    )?;
                    Ok(envelope)
                }
            }
        })
    }

    pub fn verify(
        &self,
        request: SemanticVerifyRequest,
    ) -> BoxFuture<'_, Result<SemanticEnvelope<SemanticVerification>, SemanticError>> {
        Box::pin(async move {
            match self.select(&SemanticCapabilities::VERIFY) {
                SemanticSelection::Provider(provider) => {
                    let provider_id = provider.id();
                    let attempt = guard_call(
                        provider_id.clone(),
                        request.call.clone(),
                        self.clock.clone(),
                        provider.verify(request.clone()),
                    );
                    match attempt.await {
                        Ok(envelope) => match self.validate_response(
                            &provider_id,
                            request.workspace,
                            request.snapshot_id,
                            &envelope,
                        ) {
                            Ok(()) => Ok(envelope),
                            Err(err) if request.call.require_provider => Err(err),
                            Err(_) => {
                                let workspace = request.workspace;
                                let snapshot_id = request.snapshot_id;
                                let envelope = self.fallback.verify(request).await?;
                                self.validate_response(
                                    &self.fallback.id(),
                                    workspace,
                                    snapshot_id,
                                    &envelope,
                                )?;
                                Ok(envelope)
                            }
                        },
                        Err(err) if err.caller_terminal() => Err(err),
                        Err(err) if request.call.require_provider => Err(err),
                        Err(_) => {
                            let workspace = request.workspace;
                            let snapshot_id = request.snapshot_id;
                            let envelope = self.fallback.verify(request).await?;
                            self.validate_response(
                                &self.fallback.id(),
                                workspace,
                                snapshot_id,
                                &envelope,
                            )?;
                            Ok(envelope)
                        }
                    }
                }
                SemanticSelection::Fallback(fallback) => {
                    if request.call.require_provider {
                        return Err(self.provider_required(crate::types::SemanticOp::Verify));
                    }
                    let envelope = guard_call(
                        fallback.id(),
                        request.call.clone(),
                        self.clock.clone(),
                        fallback.verify(request.clone()),
                    )
                    .await?;
                    self.validate_response(
                        &self.fallback.id(),
                        request.workspace,
                        request.snapshot_id,
                        &envelope,
                    )?;
                    Ok(envelope)
                }
            }
        })
    }

    pub fn explain(
        &self,
        request: SemanticExplainRequest,
    ) -> BoxFuture<'_, Result<SemanticEnvelope<SemanticExplanation>, SemanticError>> {
        Box::pin(async move {
            match self.select(&SemanticCapabilities::EXPLAIN) {
                SemanticSelection::Provider(provider) => {
                    let provider_id = provider.id();
                    let attempt = guard_call(
                        provider_id.clone(),
                        request.call.clone(),
                        self.clock.clone(),
                        provider.explain(request.clone()),
                    );
                    match attempt.await {
                        Ok(envelope) => match self.validate_response(
                            &provider_id,
                            request.workspace,
                            request.snapshot_id,
                            &envelope,
                        ) {
                            Ok(()) => Ok(envelope),
                            Err(err) if request.call.require_provider => Err(err),
                            Err(_) => {
                                let workspace = request.workspace;
                                let snapshot_id = request.snapshot_id;
                                let envelope = self.fallback.explain(request).await?;
                                self.validate_response(
                                    &self.fallback.id(),
                                    workspace,
                                    snapshot_id,
                                    &envelope,
                                )?;
                                Ok(envelope)
                            }
                        },
                        Err(err) if err.caller_terminal() => Err(err),
                        Err(err) if request.call.require_provider => Err(err),
                        Err(_) => {
                            let workspace = request.workspace;
                            let snapshot_id = request.snapshot_id;
                            let envelope = self.fallback.explain(request).await?;
                            self.validate_response(
                                &self.fallback.id(),
                                workspace,
                                snapshot_id,
                                &envelope,
                            )?;
                            Ok(envelope)
                        }
                    }
                }
                SemanticSelection::Fallback(fallback) => {
                    if request.call.require_provider {
                        return Err(self.provider_required(crate::types::SemanticOp::Explain));
                    }
                    let envelope = guard_call(
                        fallback.id(),
                        request.call.clone(),
                        self.clock.clone(),
                        fallback.explain(request.clone()),
                    )
                    .await?;
                    self.validate_response(
                        &self.fallback.id(),
                        request.workspace,
                        request.snapshot_id,
                        &envelope,
                    )?;
                    Ok(envelope)
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{block_on, call, entity, provider_id, snapshot};
    use crate::types::{SemanticContextPack, SemanticOp};
    use faktor_core::{TestClock, WorkspaceId};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::task::Waker;

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum ProbeMode {
        Ready,
        Panic,
        Park,
    }

    struct ProbeProvider {
        id: SemanticProviderId,
        caps: SemanticCapabilities,
        mode: ProbeMode,
        polled: Arc<AtomicBool>,
    }

    impl ProbeProvider {
        fn new(id: &str, caps: SemanticCapabilities, mode: ProbeMode) -> Self {
            Self {
                id: provider_id(id),
                caps,
                mode,
                polled: Arc::new(AtomicBool::new(false)),
            }
        }

        fn context_envelope(
            &self,
            request: &SemanticContextRequest,
        ) -> SemanticEnvelope<SemanticContextPack> {
            SemanticEnvelope::new(
                self.id.clone(),
                1,
                request.workspace,
                request.snapshot_id,
                7,
                SemanticContextPack {
                    items: Vec::new(),
                    truncated: false,
                    total_bytes: 0,
                    degraded: false,
                },
            )
        }
    }

    impl SemanticProvider for ProbeProvider {
        fn id(&self) -> SemanticProviderId {
            self.id.clone()
        }

        fn version(&self) -> u32 {
            1
        }

        fn capabilities(&self) -> SemanticCapabilities {
            self.caps
        }

        fn context(
            &self,
            request: SemanticContextRequest,
        ) -> BoxFuture<'_, Result<SemanticEnvelope<SemanticContextPack>, SemanticError>> {
            match self.mode {
                ProbeMode::Ready => {
                    let envelope = self.context_envelope(&request);
                    Box::pin(async move { Ok(envelope) })
                }
                ProbeMode::Panic => Box::pin(async move { panic!("probe provider crash") }),
                ProbeMode::Park => {
                    let polled = self.polled.clone();
                    Box::pin(async move {
                        polled.store(true, Ordering::SeqCst);
                        std::future::pending::<
                            Result<SemanticEnvelope<SemanticContextPack>, SemanticError>,
                        >()
                        .await
                    })
                }
            }
        }
    }

    fn context_request() -> SemanticContextRequest {
        SemanticContextRequest {
            call: call(),
            workspace: WorkspaceId::new(1),
            source_revision: "rev-1".to_string(),
            snapshot_id: snapshot(WorkspaceId::new(1), "rev-1"),
            query: "where is the scheduler".to_string(),
            max_items: 8,
            max_bytes: 4096,
        }
    }

    #[test]
    fn selection_is_capability_driven_not_name_driven() {
        let mut registry = SemanticProviderRegistry::new(GenericSemanticFallback::default());
        registry.register(Arc::new(ProbeProvider::new(
            "alpha",
            SemanticCapabilities::CONTEXT,
            ProbeMode::Ready,
        )));
        registry.register(Arc::new(ProbeProvider::new(
            "beta",
            SemanticCapabilities::EXPLAIN,
            ProbeMode::Ready,
        )));
        assert!(matches!(
            registry.select(&SemanticCapabilities::CONTEXT),
            SemanticSelection::Provider(provider) if provider.id().as_str() == "alpha"
        ));
        assert!(matches!(
            registry.select(&SemanticCapabilities::EXPLAIN),
            SemanticSelection::Provider(provider) if provider.id().as_str() == "beta"
        ));
        // Uncovered operation falls to the generic fallback.
        assert!(matches!(
            registry.select(&SemanticCapabilities::VERIFY),
            SemanticSelection::Fallback(_)
        ));
    }

    #[test]
    fn provider_absence_falls_back_and_never_fails_ordinary_operation() {
        let registry = SemanticProviderRegistry::new(GenericSemanticFallback::default());
        let envelope = block_on(registry.context(context_request())).unwrap();
        assert!(envelope.payload.degraded);
        assert_eq!(envelope.provider_id.as_str(), "generic-fallback");

        let snapshot_env = block_on(registry.snapshot(SemanticSnapshotRequest {
            call: call(),
            workspace: WorkspaceId::new(1),
            source_revision: "rev-1".to_string(),
        }))
        .unwrap();
        assert_eq!(snapshot_env.workspace, WorkspaceId::new(1));
    }

    #[test]
    fn provider_crash_is_typed_and_degrades_to_fallback() {
        let direct = ProbeProvider::new("crasher", SemanticCapabilities::CONTEXT, ProbeMode::Panic);
        let result = block_on(guard_call(
            direct.id(),
            call(),
            Arc::new(TestClock::new(0)),
            direct.context(context_request()),
        ));
        match result {
            Err(SemanticError::ProviderCrashed { provider }) => assert_eq!(provider, "crasher"),
            other => panic!("expected ProviderCrashed, got {other:?}"),
        }

        // Registry: ordinary operation still succeeds through the fallback.
        let mut registry = SemanticProviderRegistry::new(GenericSemanticFallback::default());
        registry.register(Arc::new(ProbeProvider::new(
            "crasher",
            SemanticCapabilities::CONTEXT,
            ProbeMode::Panic,
        )));
        let envelope = block_on(registry.context(context_request())).unwrap();
        assert!(envelope.payload.degraded);
    }

    #[test]
    fn slow_provider_cancel_is_typed_and_stops_before_polling() {
        // Pre-cancelled: the provider is never polled at all.
        let provider = ProbeProvider::new("slow", SemanticCapabilities::CONTEXT, ProbeMode::Park);
        let polled = provider.polled.clone();
        let mut request = context_request();
        let token = CancellationToken::new();
        token.cancel();
        request.call.cancellation = token;
        let result = block_on(guard_call(
            provider.id(),
            request.call.clone(),
            Arc::new(TestClock::new(0)),
            provider.context(request),
        ));
        assert!(matches!(result, Err(SemanticError::Cancelled { .. })));
        assert!(
            !polled.load(Ordering::SeqCst),
            "a cancelled call must never poll the provider"
        );

        // Parked call: first poll pends, cancel then resolves typed.
        let provider = ProbeProvider::new("slow2", SemanticCapabilities::CONTEXT, ProbeMode::Park);
        let polled = provider.polled.clone();
        let token = CancellationToken::new();
        let mut request = context_request();
        request.call.cancellation = token.clone();
        let mut future = Box::pin(guard_call(
            provider.id(),
            request.call.clone(),
            Arc::new(TestClock::new(0)),
            provider.context(request),
        ));
        let mut cx = Context::from_waker(Waker::noop());
        assert!(future.as_mut().poll(&mut cx).is_pending());
        assert!(polled.load(Ordering::SeqCst), "the parked call ran once");
        token.cancel();
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(Err(SemanticError::Cancelled { .. })) => {}
            other => panic!("expected typed Cancelled, got {other:?}"),
        }
    }

    #[test]
    fn expired_deadline_is_typed_before_any_provider_poll() {
        let provider = ProbeProvider::new("slow", SemanticCapabilities::CONTEXT, ProbeMode::Park);
        let polled = provider.polled.clone();
        let mut registry = SemanticProviderRegistry::new(GenericSemanticFallback::default())
            .with_clock(Arc::new(TestClock::new(1_000)));
        registry.register(Arc::new(provider));
        let mut request = context_request();
        request.call.deadline = Some(Deadline::at(999));
        let result = block_on(registry.context(request));
        match result {
            Err(SemanticError::DeadlineExceeded { provider }) => assert_eq!(provider, "slow"),
            other => panic!("expected DeadlineExceeded, got {other:?}"),
        }
        assert!(
            !polled.load(Ordering::SeqCst),
            "an expired call must never poll the provider"
        );
    }

    #[test]
    fn fallback_serves_every_operation_when_no_provider_is_registered() {
        let registry = SemanticProviderRegistry::new(GenericSemanticFallback::default());
        let workspace = WorkspaceId::new(1);
        let snapshot_id = snapshot(workspace, "rev-1");
        let entity = entity("src/lib.rs", "lib");

        assert!(block_on(registry.snapshot(SemanticSnapshotRequest {
            call: call(),
            workspace,
            source_revision: "rev-1".to_string(),
        }))
        .is_ok());
        assert!(block_on(registry.context(context_request())).is_ok());
        assert!(block_on(registry.delta(SemanticDeltaRequest {
            call: call(),
            workspace,
            from_snapshot: snapshot_id,
            from_source_revision: "rev-0".to_string(),
            to_source_revision: "rev-1".to_string(),
        }))
        .is_ok());
        assert!(block_on(registry.affected(AffectedRequest {
            call: call(),
            workspace,
            snapshot_id,
            changed: vec![entity.clone()],
            max_depth: 1,
        }))
        .is_ok());
        assert!(block_on(registry.verify(SemanticVerifyRequest {
            call: call(),
            workspace,
            snapshot_id,
            entity: entity.clone(),
            claim: "scheduler starts".to_string(),
        }))
        .is_ok());
        assert!(block_on(registry.explain(SemanticExplainRequest {
            call: call(),
            workspace,
            snapshot_id,
            entity,
            question: "why".to_string(),
        }))
        .is_ok());
        for op in SemanticOp::ALL {
            assert!(registry.fallback().capabilities().supports(op));
        }
    }
}
