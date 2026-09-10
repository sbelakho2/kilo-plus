//! Native semantic introspection (audit 83): provider registry status and
//! capability coverage.
//!
//! Two read-only endpoints:
//!
//! - `GET /native/semantic/status` — which semantic providers are
//!   registered (id, version, capabilities) plus the always-available
//!   fallback. When no registry is wired, the fallback-only shape is
//!   reported (`configured: false`) — never a 500.
//! - `GET /native/semantic/capabilities` — the capability matrix plus the
//!   union coverage; selection on this surface is capability-driven only.
//!
//! Only provider identity, version and capability booleans are exposed:
//! no endpoints, credentials, model names or other implementation data.

use faktor_semantic::fallback::GenericSemanticFallback;
use faktor_semantic::registry::SemanticProviderRegistry;
use faktor_semantic::types::{SemanticOp, SemanticProvider};

use super::*;

/// One provider's public introspection shape. Deliberately narrow: id +
/// version + capabilities, nothing else.
fn provider_json(provider: &dyn SemanticProvider) -> serde_json::Value {
    let caps = provider.capabilities();
    let mut operations = serde_json::Map::new();
    for op in SemanticOp::ALL {
        operations.insert(
            op.as_str().to_string(),
            serde_json::Value::Bool(caps.supports(op)),
        );
    }
    serde_json::json!({
        "id": provider.id().as_str(),
        "version": provider.version(),
        "capabilities": {
            "operations": operations,
            "composeDelta": caps.compose_delta,
            "constrainedEdit": caps.constrained_edit,
        },
    })
}

/// The authenticated registry view: the configured registry when one is
/// wired, else a freshly built fallback-only registry. The fallback is
/// always present by construction, so the shape never depends on whether a
/// provider was configured.
fn registry<'a>(
    state: &'a AppState,
    fallback_registry: &'a mut Option<SemanticProviderRegistry>,
) -> &'a SemanticProviderRegistry {
    match state.deps.semantic.as_deref() {
        Some(registry) => registry,
        None => fallback_registry
            .get_or_insert_with(|| SemanticProviderRegistry::new(GenericSemanticFallback::new())),
    }
}

/// `GET /native/semantic/status` — registry status without leaking
/// implementation data. `configured: false` + fallback is a 200, not an
/// error: an unconfigured daemon still has working fallback semantics.
pub(crate) async fn native_semantic_status(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let configured = state.deps.semantic.is_some();
    let mut fallback_registry = None;
    let registry = registry(&state, &mut fallback_registry);
    let providers = registry
        .providers()
        .iter()
        .map(|p| provider_json(p.as_ref()))
        .collect::<Vec<_>>();
    let snapshot_providers = survey_snapshot_ids(registry);
    Json(serde_json::json!({
        "configured": configured,
        "providerCount": providers.len(),
        "providers": providers,
        "fallback": provider_json(registry.fallback()),
        "snapshotState": {
            "providers": snapshot_providers,
            "fallback": registry.fallback().capabilities().supports(SemanticOp::Snapshot),
        },
    }))
    .into_response()
}

/// `GET /native/semantic/capabilities` — per-provider capability matrix plus
/// the union across providers and the fallback.
pub(crate) async fn native_semantic_capabilities(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let configured = state.deps.semantic.is_some();
    let mut fallback_registry = None;
    let registry = registry(&state, &mut fallback_registry);
    let providers = registry
        .providers()
        .iter()
        .map(|p| provider_json(p.as_ref()))
        .collect::<Vec<_>>();
    let mut union = serde_json::Map::new();
    for op in SemanticOp::ALL {
        let supported = registry
            .providers()
            .iter()
            .any(|p| p.capabilities().supports(op))
            || registry.fallback().capabilities().supports(op);
        union.insert(op.as_str().to_string(), serde_json::Value::Bool(supported));
    }
    let compose_delta = registry
        .providers()
        .iter()
        .any(|p| p.capabilities().compose_delta)
        || registry.fallback().capabilities().compose_delta;
    let constrained_edit = registry
        .providers()
        .iter()
        .any(|p| p.capabilities().constrained_edit)
        || registry.fallback().capabilities().constrained_edit;
    Json(serde_json::json!({
        "configured": configured,
        "providers": providers,
        "fallback": provider_json(registry.fallback()),
        "union": {
            "operations": union,
            "composeDelta": compose_delta,
            "constrainedEdit": constrained_edit,
        },
    }))
    .into_response()
}

fn survey_snapshot_ids(registry: &SemanticProviderRegistry) -> Vec<String> {
    registry
        .providers()
        .iter()
        .filter(|p| p.capabilities().supports(SemanticOp::Snapshot))
        .map(|p| p.id().as_str().to_string())
        .collect()
}
