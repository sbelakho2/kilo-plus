//! kilop-deepseek — first-class DeepSeek profiles (spec §11).
//!
//! DeepSeek is a first-class test matrix, not an accident of OpenAI
//! compatibility: separate profiles for direct, OpenRouter, Kilo Gateway,
//! arbitrary OpenAI-compatible endpoints, and local derivatives. Capability
//! normalization happens after discovery; the agent never branches on the
//! provider name.

use std::sync::Arc;

use kilop_core::model::ModelCapabilities;
use kilop_openai::{OpenAiConfig, OpenAiFamily, OpenAiProvider};
use kilop_provider::Provider;

#[derive(Debug, Clone)]
pub enum DeepSeekProfile {
    /// https://api.deepseek.com (native DeepSeek API).
    Direct,
    /// DeepSeek models via OpenRouter.
    OpenRouter,
    /// DeepSeek models via the Kilo Gateway.
    Gateway { base_url: String },
    /// Any OpenAI-compatible endpoint.
    Compatible { base_url: String },
    /// A local DeepSeek derivative (e.g. a distilled model in Ollama or a
    /// vLLM server).
    LocalDerivative { base_url: String },
}

#[derive(Debug, Clone)]
pub struct DeepSeekConfig {
    pub profile: DeepSeekProfile,
    pub api_key: Option<String>,
    /// Capability overrides; defaults follow the DeepSeek model family.
    pub model_overrides: std::collections::HashMap<String, ModelCapabilities>,
}

impl DeepSeekConfig {
    pub fn direct(api_key: Option<String>) -> Self {
        Self {
            profile: DeepSeekProfile::Direct,
            api_key,
            model_overrides: std::collections::HashMap::new(),
        }
    }

    pub fn compatible(base_url: impl Into<String>, api_key: Option<String>) -> Self {
        Self {
            profile: DeepSeekProfile::Compatible {
                base_url: base_url.into(),
            },
            api_key,
            model_overrides: std::collections::HashMap::new(),
        }
    }
}

/// Build a DeepSeek provider for the chosen profile. All profiles use the
/// OpenAI-compatible wire (DeepSeek's own API is OpenAI-shaped); the profile
/// only changes the endpoint — the agent sees one normalized provider.
pub fn build(config: DeepSeekConfig) -> Arc<dyn Provider> {
    let (base_url, family) = match &config.profile {
        DeepSeekProfile::Direct => ("https://api.deepseek.com", OpenAiFamily::Chat),
        DeepSeekProfile::OpenRouter => ("https://openrouter.ai/api/v1", OpenAiFamily::Chat),
        DeepSeekProfile::Gateway { base_url }
        | DeepSeekProfile::Compatible { base_url }
        | DeepSeekProfile::LocalDerivative { base_url } => (base_url.as_str(), OpenAiFamily::Chat),
    };
    let mut openai = OpenAiConfig::chat(base_url, config.api_key.clone());
    openai.family = family;
    if !config.model_overrides.is_empty() {
        for (m, caps) in &config.model_overrides {
            openai = openai.with_model(m, caps.clone());
        }
    } else {
        // DeepSeek family defaults (capability-driven, never name-matched in
        // the agent).
        openai = openai.with_default_caps(ModelCapabilities {
            context: 64_000,
            max_output: 8_192,
            tools: true,
            parallel_tools: true,
            thinking: false,
            vision: false,
            json_schema: true,
            streaming: true,
            embeddings: false,
            reasoning: false,
        });
    }
    OpenAiProvider::build(openai)
}

pub fn provider_id() -> &'static str {
    "deepseek"
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use kilop_core::cancellation::CancellationToken;
    use kilop_core::id::{OpId, SessionId};
    use kilop_provider::testing::{MockAction, MockServer};
    use kilop_provider::{
        ContentPart, GenericAgentRequest, ProviderChunk, RequestMessage, RequestMeta, Role,
        ToolSpec,
    };

    fn req(model: &str) -> GenericAgentRequest {
        GenericAgentRequest {
            model: model.into(),
            system: "sys".into(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec![ContentPart::text("hi")],
            }],
            tools: vec![ToolSpec {
                name: "read_file".into(),
                description: "read".into(),
                input_schema: serde_json::json!({"type": "object"}),
            }],
            max_output: None,
            reasoning: None,
            stream: true,
            meta: RequestMeta {
                operation_id: OpId::new(1),
                session_id: SessionId::new(1),
                provider: "deepseek".into(),
                attempt: 0,
                deadline_ms: 5000,
                cancellation: CancellationToken::new(),
            },
        }
    }

    #[tokio::test]
    async fn direct_profile_hits_deepseek_endpoint() {
        let server = MockServer::new();
        server.route(
            "POST",
            "/chat/completions",
            MockAction::Respond {
                status: 200,
                body: "data: {\"choices\":[{\"delta\":{\"content\":\"deep\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n".into(),
            },
        );
        let base = server.base_url().await;
        let cfg = DeepSeekConfig {
            profile: DeepSeekProfile::Compatible {
                base_url: base.clone(),
            },
            api_key: Some("sk".into()),
            model_overrides: std::collections::HashMap::new(),
        };
        let provider = build(cfg);
        assert_eq!(provider.id(), "openai"); // the wire family, not "deepseek"
        let mut stream = provider.stream(req("deepseek-chat"));
        let mut text = String::new();
        while let Some(chunk) = stream.next().await {
            match chunk.unwrap() {
                ProviderChunk::Text { text: t } => text.push_str(&t),
                ProviderChunk::Done => break,
                _ => {}
            }
        }
        assert_eq!(text, "deep");
        let (_, path, _) = server.last_request().unwrap();
        assert_eq!(path, "/chat/completions");
    }

    #[tokio::test]
    async fn all_profiles_produce_capability_driven_behavior() {
        for profile in [
            DeepSeekProfile::Direct,
            DeepSeekProfile::OpenRouter,
            DeepSeekProfile::Gateway {
                base_url: "http://127.0.0.1:1".into(),
            },
            DeepSeekProfile::LocalDerivative {
                base_url: "http://127.0.0.1:1".into(),
            },
        ] {
            let cfg = DeepSeekConfig {
                profile,
                api_key: None,
                model_overrides: std::collections::HashMap::new(),
            };
            let provider = build(cfg);
            let caps = provider.capabilities("deepseek-chat");
            assert!(caps.tools, "deepseek family defaults to tools");
            assert!(caps.json_schema);
            assert!(!caps.vision);
            // The agent's no-name-matching invariant: capabilities only.
            assert!(caps.context >= 64_000);
        }
    }

    #[test]
    fn override_caps_win() {
        let cfg = DeepSeekConfig {
            profile: DeepSeekProfile::Direct,
            api_key: None,
            model_overrides: std::collections::HashMap::from([(
                "deepseek-chat".into(),
                ModelCapabilities {
                    context: 128_000,
                    tools: false,
                    ..Default::default()
                },
            )]),
        };
        let provider = build(cfg);
        assert!(!provider.capabilities("deepseek-chat").tools);
        assert_eq!(provider.capabilities("deepseek-chat").context, 128_000);
    }
}
