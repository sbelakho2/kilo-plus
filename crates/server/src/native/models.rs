//! Model catalog / capability and provider registry projections.

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use faktor_core::model::ModelCapabilities;

use super::*;
use crate::api::AppState;

/// Provenance of one native catalog entry (docs/native-protocol.md):
/// `liveProbe` when the provider reports a LIVE runtime context limit
/// for the model (e.g. an Ollama `/api/ps` allocation); otherwise
/// `providerCatalog` for entries carrying a non-default capability
/// profile (configured or probed), and `conservativeDefault` for entries
/// still at the fail-safe default profile (unprobed).
pub(crate) fn catalog_source(
    p: &dyn faktor_provider::Provider,
    model: &str,
    caps: &ModelCapabilities,
) -> &'static str {
    if p.runtime_context_limit(model).is_some() {
        "liveProbe"
    } else if *caps == ModelCapabilities::default() {
        "conservativeDefault"
    } else {
        "providerCatalog"
    }
}

/// `GET /models` — the flat native model catalog: every registered
/// provider instance × its known models × capabilities
/// (docs/native-protocol.md).
pub(crate) async fn native_models(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let mut out = Vec::new();
    for p in state.deps.agent.deps().providers.all() {
        // Registry key (instance id) — the same id session rows store.
        let instance = p.identity().instance_id.clone();
        for model in p.known_models() {
            let caps = p.capabilities(&model);
            let source = catalog_source(p.as_ref(), &model, &caps);
            out.push(serde_json::json!({
                "provider": instance,
                "model": model,
                "context": caps.context,
                "maxOutput": caps.max_output,
                "tools": caps.tools,
                "parallelTools": caps.parallel_tools,
                "reasoning": caps.reasoning,
                "thinking": caps.thinking,
                "vision": caps.vision,
                "structuredOutput": caps.json_schema,
                "embeddings": caps.embeddings,
                "streaming": caps.streaming,
                "source": source,
            }));
        }
    }
    out.sort_by(|a, b| {
        (a["provider"].as_str(), a["model"].as_str())
            .cmp(&(b["provider"].as_str(), b["model"].as_str()))
    });
    Json(out).into_response()
}

/// `GET /capabilities` — native introspection map:
/// `{ "<provider>": { models: [{id, capabilities}],
/// runtimeContextLimitSupported } }` (docs/native-protocol.md).
pub(crate) async fn native_capabilities(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let mut sorted = Vec::new();
    for p in state.deps.agent.deps().providers.all() {
        let instance = p.identity().instance_id.clone();
        let mut models = Vec::new();
        let mut live = false;
        for m in p.known_models() {
            if p.runtime_context_limit(&m).is_some() {
                live = true;
            }
            models.push(serde_json::json!({ "id": m, "capabilities": p.capabilities(&m) }));
        }
        models.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
        sorted.push((
            instance,
            serde_json::json!({
                "models": models,
                "runtimeContextLimitSupported": live,
            }),
        ));
    }
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    let mut map = serde_json::Map::new();
    for (instance, entry) in sorted {
        map.insert(instance, entry);
    }
    Json(serde_json::Value::Object(map)).into_response()
}

// ------------------------------------------------- native v1: audits 55-56
// Liveness/readiness and the durable session listings (all auth-gated like
// every daemon route). Response bodies are native JSON over durable state:
// session rows, turn records, the task ledger, checkpoint rows, memory
// facts and live PTYs — never v7.5.6 wire shapes. Hostile ids are 400
// (unparseable/0) or 404 (unknown); handlers never panic on them.

/// `GET /native/providers` — the registry view of every registered provider
/// (audit P0-64): instance/family identity, the models it can serve with
/// their capability profiles, the capability source, and a live health
/// snapshot. The daemon exposes no per-provider rate-limit/cooldown state
/// to the server layer (adapters track those privately), so `health` is the
/// honest static snapshot: registration + live runtime-context probe
/// support + configured context limit. Never emits secrets (auth/endpoint
/// metadata stays in the provider layer).
pub(crate) async fn native_providers(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Response {
    if let Err(e) = authed(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Json(e.to_json())).into_response();
    }
    let mut out: Vec<serde_json::Value> = Vec::new();
    for p in state.deps.agent.deps().providers.all() {
        let identity = p.identity();
        let mut models: Vec<serde_json::Value> = Vec::new();
        for model in p.known_models() {
            let caps = p.capabilities(&model);
            models.push(serde_json::json!({
                "model": model,
                "context": caps.context,
                "maxOutput": caps.max_output,
                "tools": caps.tools,
                "parallelTools": caps.parallel_tools,
                "reasoning": caps.reasoning,
                "thinking": caps.thinking,
                "vision": caps.vision,
                "structuredOutput": caps.json_schema,
                "embeddings": caps.embeddings,
                "streaming": caps.streaming,
                "source": catalog_source(p.as_ref(), &model, &caps),
            }));
        }
        models.sort_by(|a, b| a["model"].as_str().cmp(&b["model"].as_str()));
        out.push(serde_json::json!({
            "instanceId": identity.instance_id,
            "family": identity.family,
            "models": models,
            "runtimeContextLimitSupported": p
                .known_models()
                .iter()
                .any(|m| p.runtime_context_limit(m).is_some()),
            "health": {
                "status": "registered",
                "note": "rate-limit/cooldown state is adapter-private and not exposed to the server; this snapshot is the registry view",
            },
        }));
    }
    out.sort_by(|a, b| a["instanceId"].as_str().cmp(&b["instanceId"].as_str()));
    Json(out).into_response()
}
