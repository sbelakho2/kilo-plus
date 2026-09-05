//! faktor-gateway — Kilo/OpenRouter-style gateway adapters (spec §12, §36).
//!
//! A gateway is an OpenAI-compatible endpoint with model routing and extra
//! headers. BYOK is preserved: the gateway key is configured per provider
//! and never persisted by the runtime.

use std::sync::Arc;

use faktor_core::model::ModelCapabilities;
use faktor_openai::{OpenAiConfig, OpenAiProvider};
use faktor_provider::egress::{HttpTransport, PolicyCheckedHttpTransport};
use faktor_provider::Provider;

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
    build_with_transport(config, Arc::new(PolicyCheckedHttpTransport::permissive()))
}

/// Build a gateway provider with an injected egress transport
/// (policy-checked in production, mock in tests). The SAME transport is
/// shared by both paths (plain forwarding through the openai provider and
/// the extra-headers path through `HeaderGateway`), so one installed
/// destination policy governs the whole gateway.
pub fn build_with_transport(
    config: GatewayConfig,
    transport: Arc<dyn HttpTransport>,
) -> Arc<dyn Provider> {
    let mut openai = OpenAiConfig::chat(&config.base_url, config.api_key.clone());
    openai = openai.with_default_caps(config.default_caps.clone());
    // Extra headers are applied by the openai transport on the gateway path
    // only (every other adapter passes an empty list).
    let provider = OpenAiProvider::build_with_transport(openai, transport.clone());
    if config.extra_headers.is_empty() && config.route_prefixes.is_empty() {
        return provider;
    }
    Arc::new(HeaderGateway {
        inner: provider,
        extra_headers: config.extra_headers,
        route_prefixes: config.route_prefixes,
        default_caps: config.default_caps,
        transport,
        base_url: config.base_url,
        api_key: config.api_key,
    })
}

struct HeaderGateway {
    inner: Arc<dyn Provider>,
    extra_headers: Vec<(String, String)>,
    route_prefixes: Vec<(String, String)>,
    default_caps: ModelCapabilities,
    transport: Arc<dyn HttpTransport>,
    base_url: String,
    api_key: Option<String>,
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

    fn stream(&self, req: faktor_provider::GenericAgentRequest) -> faktor_provider::ProviderStream {
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
        if self.extra_headers.is_empty() {
            // No extra headers: plain forwarding keeps one transport path.
            return self.inner.stream(req);
        }
        // Extra headers ride the shared openai transport (applied to the
        // request before send) — the gateway path is the only one that
        // passes a non-empty list.
        let body =
            faktor_openai::chat_completions_body(&req, &faktor_openai::OpenAiQuirks::default());
        let url = format!("{}/chat/completions", self.base_url);
        let headers = faktor_openai::authorization_headers(self.api_key.as_deref());
        let transport = self.transport.clone();
        let deadlines = faktor_provider::transport::StreamDeadlines::default();
        let cancel = req.meta.cancellation.clone();
        Box::pin(faktor_openai::openai_stream(
            transport,
            url,
            headers,
            self.extra_headers.clone(),
            body,
            deadlines,
            Some(cancel),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faktor_core::cancellation::CancellationToken;
    use faktor_core::id::{OpId, SessionId};
    use faktor_provider::egress::MockHttpTransport;
    use faktor_provider::testing::{MockAction, MockServer};
    use faktor_provider::{
        ContentPart, GenericAgentRequest, ProviderChunk, RequestMessage, RequestMeta, Role,
    };
    use faktor_security::destination::DestinationPolicy;
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

    #[tokio::test]
    async fn extra_headers_arrive_on_the_chat_completions_request() {
        // P0: GatewayConfig.extra_headers were dead config — the stream
        // never applied them. They must land on the wire request verbatim.
        let server = MockServer::new();
        server.route(
            "POST",
            "/v1/chat/completions",
            MockAction::Respond {
                status: 200,
                body: "data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n".into(),
            },
        );
        let base = server.base_url().await;
        let cfg = GatewayConfig {
            id: "gw".into(),
            base_url: format!("{base}/v1"),
            api_key: Some("sk".into()),
            extra_headers: vec![
                ("X-Title".into(), "Faktor".into()),
                ("X-Referer".into(), "https://kilo.ai".into()),
                ("authorization".into(), "sk-extra-override".into()),
            ],
            route_prefixes: vec![],
            default_caps: ModelCapabilities::default(),
        };
        let provider = build(cfg);
        let mut stream = provider.stream(req("deepseek/deepseek-v4-flash"));
        while let Some(chunk) = stream.next().await {
            if let Ok(ProviderChunk::Done) = chunk {
                break;
            }
        }
        let (_, path, _) = server.last_request().unwrap();
        assert_eq!(path, "/v1/chat/completions");
        let headers = server.last_request_headers();
        let get = |name: &str| {
            headers
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, v)| v.clone())
        };
        assert_eq!(get("x-title").as_deref(), Some("Faktor"));
        assert_eq!(get("x-referer").as_deref(), Some("https://kilo.ai"));
        // Forwarded verbatim: an explicit authorization extra replaces the
        // gateway key (reqwest .headers() overwrites per name).
        assert_eq!(get("authorization").as_deref(), Some("sk-extra-override"));
    }

    #[tokio::test]
    async fn gateway_without_extra_headers_stays_header_clean() {
        // The openai/direct path (empty extra list) must not grow gateway
        // headers: extra headers apply ONLY on the gateway path.
        let server = MockServer::new();
        server.route(
            "POST",
            "/chat/completions",
            MockAction::Respond {
                status: 200,
                body: "data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n".into(),
            },
        );
        let base = server.base_url().await;
        let provider =
            faktor_openai::OpenAiProvider::build(faktor_openai::OpenAiConfig::chat(base, None));
        let mut stream = provider.stream(req("m"));
        while let Some(chunk) = stream.next().await {
            if let Ok(ProviderChunk::Done) = chunk {
                break;
            }
        }
        let headers = server.last_request_headers();
        assert!(
            headers.iter().all(|(n, _)| n != "x-title"),
            "no extra headers without the gateway path"
        );
    }

    // ------------------------------------------------------- egress (P0-36)

    fn allow_only(port: u16) -> Arc<dyn HttpTransport> {
        Arc::new(PolicyCheckedHttpTransport::with_policy(Some(
            DestinationPolicy::parse_lines([&format!("http://127.0.0.1:{port}")]).unwrap(),
        )))
    }

    #[tokio::test]
    async fn egress_allowlist_gates_forwarding_and_extra_header_paths() {
        // Forwarding path (no extras): allowed through the injected
        // transport, denied BEFORE any network byte on the deny instance.
        let server = MockServer::new();
        let ok_body = "data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
        server.route(
            "POST",
            "/chat/completions",
            MockAction::Respond {
                status: 200,
                body: ok_body.into(),
            },
        );
        let base = server.base_url().await;
        let port = reqwest::Url::parse(&base).unwrap().port().unwrap();
        let cfg = |extra: bool| GatewayConfig {
            id: "gw".into(),
            base_url: base.clone(),
            api_key: None,
            extra_headers: if extra {
                vec![("X-Title".into(), "Faktor".into())]
            } else {
                vec![]
            },
            route_prefixes: vec![],
            default_caps: ModelCapabilities::default(),
        };
        let drain = |p: Arc<dyn Provider>| async move {
            let mut stream = p.stream(req("deepseek/deepseek-v4-flash"));
            while let Some(chunk) = stream.next().await {
                if let Ok(ProviderChunk::Done) = chunk {
                    break;
                }
            }
        };

        let allowed = build_with_transport(cfg(false), allow_only(port));
        drain(allowed).await;
        assert_eq!(server.request_count(), 1);

        let denied = build_with_transport(cfg(false), allow_only(port.wrapping_add(1)));
        let mut stream = denied.stream(req("m"));
        let err = stream
            .next()
            .await
            .expect("an item")
            .expect_err("the first item must be the deny error");
        assert!(err.message.contains("denied"), "{}", err.message);
        assert!(!err.retryable, "denied destinations are never retried");
        assert_eq!(server.request_count(), 1, "deny happened before connect");

        // Extra-headers path (HeaderGateway's OWN openai_stream send): the
        // shared injected transport gates it identically.
        server.route(
            "POST",
            "/chat/completions",
            MockAction::Sequence {
                actions: vec![
                    MockAction::Respond {
                        status: 200,
                        body: ok_body.into(),
                    },
                    MockAction::Respond {
                        status: 200,
                        body: ok_body.into(),
                    },
                ],
            },
        );
        let allowed = build_with_transport(cfg(true), allow_only(port));
        drain(allowed).await;
        assert_eq!(server.request_count(), 2);

        let denied = build_with_transport(cfg(true), allow_only(port.wrapping_add(1)));
        let mut stream = denied.stream(req("m"));
        let err = stream
            .next()
            .await
            .expect("an item")
            .expect_err("the extra-headers path must be denied too");
        assert!(err.message.contains("denied"), "{}", err.message);
        assert_eq!(server.request_count(), 2, "no connect on the deny path");
    }

    #[tokio::test]
    async fn mock_transport_canned_sse_drives_the_parser_without_http() {
        let body =
            "data: {\"choices\":[{\"delta\":{\"content\":\"canned\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
        let mock = Arc::new(MockHttpTransport::new(200, body));
        let as_transport: Arc<dyn HttpTransport> = mock.clone();
        let cfg = GatewayConfig {
            id: "gw".into(),
            base_url: "http://mock.invalid".into(),
            api_key: None,
            extra_headers: vec![("X-Title".into(), "Faktor".into())],
            route_prefixes: vec![],
            default_caps: ModelCapabilities::default(),
        };
        let provider = build_with_transport(cfg, as_transport);
        let mut stream = provider.stream(req("m"));
        let mut text = String::new();
        while let Some(chunk) = stream.next().await {
            match chunk.unwrap() {
                ProviderChunk::Text { text: t } => text.push_str(&t),
                ProviderChunk::Done => break,
                _ => {}
            }
        }
        assert_eq!(text, "canned");
        assert_eq!(mock.request_count(), 1);
        assert_eq!(
            mock.requests(),
            vec![(
                "POST".to_string(),
                "http://mock.invalid/chat/completions".to_string()
            )]
        );
    }
}
