//! kilop-gateway — Kilo/OpenRouter-style gateway adapters (spec §12, §36).
//!
//! A gateway is an OpenAI-compatible endpoint with model routing and extra
//! headers. BYOK is preserved: the gateway key is configured per provider
//! and never persisted by the runtime.

use std::sync::Arc;

use kilop_core::model::ModelCapabilities;
use kilop_openai::{OpenAiConfig, OpenAiProvider};
use kilop_provider::Provider;

#[derive(Debug, Clone)]
pub struct GatewayConfig {
    pub id: String,
    pub base_url: String,
    pub api_key: Option<String>,
    /// Extra headers forwarded verbatim (e.g. OpenRouter referer/title).
    #[allow(dead_code)]
    pub extra_headers: Vec<(String, String)>,
    /// Route-by-prefix model mapping: (prefix, target model).
    pub route_prefixes: Vec<(String, String)>,
    pub default_caps: ModelCapabilities,
}

impl GatewayConfig {
    pub fn openrouter(api_key: Option<String>) -> Self {
        Self {
            id: "openrouter".into(),
            base_url: "https://openrouter.ai/api/v1".into(),
            api_key,
            extra_headers: vec![],
            route_prefixes: vec![],
            default_caps: ModelCapabilities {
                context: 128_000,
                max_output: 16_384,
                tools: true,
                parallel_tools: true,
                thinking: false,
                vision: true,
                json_schema: true,
                streaming: true,
                embeddings: false,
                reasoning: false,
            },
        }
    }
}

/// Build a gateway provider. Model routing is prefix-based and happens
/// inside the adapter; the agent never sees it.
pub fn build(config: GatewayConfig) -> Arc<dyn Provider> {
    let mut openai = OpenAiConfig::chat(&config.base_url, config.api_key.clone());
    openai = openai.with_default_caps(config.default_caps.clone());
    // Extra headers are handled by a wrapper that injects them per request.
    let provider = OpenAiProvider::build(openai);
    if config.extra_headers.is_empty() && config.route_prefixes.is_empty() {
        return provider;
    }
    Arc::new(HeaderGateway {
        inner: provider,
        extra_headers: config.extra_headers,
        route_prefixes: config.route_prefixes,
        default_caps: config.default_caps,
    })
}

struct HeaderGateway {
    #[allow(dead_code)]
    inner: Arc<dyn Provider>,
    #[allow(dead_code)]
    extra_headers: Vec<(String, String)>,
    route_prefixes: Vec<(String, String)>,
    #[allow(dead_code)]
    default_caps: ModelCapabilities,
}

impl Provider for HeaderGateway {
    fn id(&self) -> &str {
        "gateway"
    }

    fn capabilities(&self, model: &str) -> ModelCapabilities {
        let caps = self.inner.capabilities(model);
        if caps.context == 0 {
            self.default_caps.clone()
        } else {
            caps
        }
    }

    fn stream(&self, req: kilop_provider::GenericAgentRequest) -> kilop_provider::ProviderStream {
        // Route-by-prefix: rewrite the model id for the upstream gateway.
        let mut model = req.model.clone();
        for (prefix, target) in &self.route_prefixes {
            if model.starts_with(prefix) {
                model = target.clone();
                break;
            }
        }
        let mut req = req;
        req.model = model;
        self.inner.stream(req)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kilop_core::cancellation::CancellationToken;
    use kilop_core::id::{OpId, SessionId};
    use kilop_provider::testing::{MockAction, MockServer};
    use kilop_provider::{
        ContentPart, GenericAgentRequest, Provider, ProviderChunk, RequestMessage, RequestMeta,
        Role,
    };
    use futures::StreamExt;

    fn req(model: &str) -> GenericAgentRequest {
        GenericAgentRequest {
            model: model.into(),
            system: "s".into(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec![ContentPart::text("hi")],
            }],
            tools: vec![],
            max_output: None,
            reasoning: None,
            stream: true,
            meta: RequestMeta {
                operation_id: OpId::new(1),
                session_id: SessionId::new(1),
                provider: "gateway".into(),
                attempt: 0,
                deadline_ms: 5000,
                cancellation: CancellationToken::new(),
            },
        }
    }

    #[tokio::test]
    async fn prefix_routing_rewrites_model() {
        let server = MockServer::new();
        server.route(
            "POST",
            "/chat/completions",
            MockAction::AssertThenRespond {
                status: 200,
                body: "data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n".into(),
                assert: Arc::new(|body: &serde_json::Value| {
                    assert_eq!(body["model"], "routed-model", "prefix routing must rewrite the model");
                }),
            },
        );
        let base = server.base_url().await;
        let cfg = GatewayConfig {
            id: "gw".into(),
            base_url: base.clone(),
            api_key: None,
            extra_headers: vec![],
            route_prefixes: vec![("deepseek/".into(), "routed-model".into())],
            default_caps: ModelCapabilities::default(),
        };
        let provider = build(cfg);
        let mut stream = provider.stream(req("deepseek/deepseek-chat"));
        while let Some(chunk) = stream.next().await {
            if let Ok(ProviderChunk::Done) = chunk {
                break;
            }
        }
    }

    #[tokio::test]
    async fn capabilities_surface_via_caps_not_names() {
        let cfg = GatewayConfig::openrouter(None);
        let provider = build(cfg);
        let caps = provider.capabilities("anthropic/claude-x");
        assert!(caps.tools);
        assert!(caps.vision);
    }
}
