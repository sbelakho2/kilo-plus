//! faktor-deepseek — first-class DeepSeek profiles (spec §11).
//!
//! DeepSeek is a first-class test matrix, not an accident of OpenAI
//! compatibility: separate profiles for direct, OpenRouter, Kilo Gateway,
//! arbitrary OpenAI-compatible endpoints, and local derivatives. Capability
//! normalization happens after discovery; the agent never branches on the
//! provider name.
//!
//! The DeepSeek-V4 family is the default profile table: 1M context across
//! the family, model-specific max output, thinking on by default. Native
//! DeepSeek endpoints additionally require the assistant's prior
//! `reasoning_content` to be replayed on later tool iterations
//! (`OpenAiQuirks`), or the API 400s.

use std::sync::Arc;

use faktor_core::model::ModelCapabilities;
use faktor_openai::{OpenAiConfig, OpenAiFamily, OpenAiProvider, OpenAiQuirks};
use faktor_provider::Provider;

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

/// DeepSeek-V4 family context window (vendor model card: 1M tokens for the
/// whole family). Max output is model-specific: 64K for the flash tier
/// (incl. the vision-experimental model), 384K for pro — conservative
/// numbers within the documented per-model ranges.
pub const V4_CONTEXT: usize = 1_000_000;
pub const V4_MAX_OUTPUT_FLASH: usize = 64_000;
pub const V4_MAX_OUTPUT_PRO: usize = 384_000;

/// Capabilities for one V4 model. `vision` is exclusive to the
/// vision-experimental model; thinking is ON by default for the family.
pub fn v4_capabilities(max_output: usize, vision: bool) -> ModelCapabilities {
    ModelCapabilities {
        context: V4_CONTEXT,
        max_output,
        tools: true,
        parallel_tools: true,
        thinking: true,
        vision,
        json_schema: true,
        streaming: true,
        embeddings: false,
        reasoning: true,
    }
}

/// The V4 model table: exact per-model caps, plus a family-default wildcard
/// for unknown names (flash-tier profile, no vision).
pub fn v4_model_entries() -> Vec<(&'static str, ModelCapabilities)> {
    vec![
        (
            "deepseek-v4-flash",
            v4_capabilities(V4_MAX_OUTPUT_FLASH, false),
        ),
        ("deepseek-v4-pro", v4_capabilities(V4_MAX_OUTPUT_PRO, false)),
        (
            "deepseek-v4-flash-vision-exp",
            v4_capabilities(V4_MAX_OUTPUT_FLASH, true),
        ),
    ]
}

fn v4_family_defaults() -> ModelCapabilities {
    v4_capabilities(V4_MAX_OUTPUT_FLASH, false)
}

/// Wire quirks per profile. Native DeepSeek API shapes (direct, OpenRouter,
/// the Kilo Gateway, and DeepSeek-compatible endpoints) replay the prior
/// assistant reasoning (`reasoning_content`) and always send a content field
/// next to tool calls — without the replay the V4 API 400s on later tool
/// iterations. Local derivatives (vLLM serving foreign models) get none.
pub fn quirks_for(profile: &DeepSeekProfile) -> OpenAiQuirks {
    match profile {
        DeepSeekProfile::LocalDerivative { .. } => OpenAiQuirks::default(),
        _ => OpenAiQuirks {
            requires_reasoning_replay_with_tools: true,
            requires_assistant_content_with_tool_calls: true,
        },
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
    let quirks = quirks_for(&config.profile);
    let mut openai = OpenAiConfig::chat(base_url, config.api_key.clone());
    openai.family = family;
    if !config.model_overrides.is_empty() {
        for (m, caps) in &config.model_overrides {
            openai = openai.with_model(m, caps.clone());
        }
    } else {
        // DeepSeek family defaults (capability-driven, never name-matched in
        // the agent): the V4 table is per-model; unknown names fall back to
        // the family wildcard.
        for (m, caps) in v4_model_entries() {
            openai = openai.with_model(m, caps);
        }
        openai = openai.with_default_caps(v4_family_defaults());
    }
    OpenAiProvider::build_with_quirks(openai, quirks)
}

pub fn provider_id() -> &'static str {
    "deepseek"
}

#[cfg(test)]
mod tests {
    use super::*;
    use faktor_core::cancellation::CancellationToken;
    use faktor_core::id::{OpId, SessionId};
    use faktor_provider::testing::{MockAction, MockServer};
    use faktor_provider::{
        ContentPart, GenericAgentRequest, ProviderChunk, RequestMessage, RequestMeta, Role,
        ToolSpec,
    };
    use futures::StreamExt;

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
            assert_eq!(
                caps.context, V4_CONTEXT,
                "family wildcard is the V4 default"
            );
            assert!(caps.thinking, "thinking is on by default for V4");
            assert!(caps.reasoning);
        }
    }

    #[test]
    fn v4_models_yield_exact_capabilities() {
        let cfg = DeepSeekConfig::direct(None);
        let provider = build(cfg);
        for (model, max_output, vision) in [
            ("deepseek-v4-flash", V4_MAX_OUTPUT_FLASH, false),
            ("deepseek-v4-pro", V4_MAX_OUTPUT_PRO, false),
            ("deepseek-v4-flash-vision-exp", V4_MAX_OUTPUT_FLASH, true),
        ] {
            let caps = provider.capabilities(model);
            assert_eq!(caps.context, V4_CONTEXT, "{model} context");
            assert_eq!(caps.max_output, max_output, "{model} max output");
            assert!(caps.tools, "{model} tools");
            assert!(caps.parallel_tools, "{model} parallel tools");
            assert!(caps.thinking, "{model} thinking default enabled");
            assert!(caps.reasoning, "{model} reasoning");
            assert_eq!(caps.vision, vision, "{model} vision is -exp-exclusive");
            assert!(caps.json_schema);
            assert!(caps.streaming);
            assert!(!caps.embeddings);
        }
        // Unknown names fall back to the conservative family wildcard
        // (flash-tier output, no vision claims).
        let fallback = provider.capabilities("deepseek-v4-unknown");
        assert_eq!(fallback.context, V4_CONTEXT);
        assert_eq!(fallback.max_output, V4_MAX_OUTPUT_FLASH);
        assert!(!fallback.vision);
    }

    #[test]
    fn v4_quirks_flag_is_set_on_native_deepseek_profiles() {
        // requires_reasoning_replay_with_tools: DeepSeek V4 400s when the
        // prior reasoning_content is not replayed on later tool iterations.
        for profile in [
            DeepSeekProfile::Direct,
            DeepSeekProfile::OpenRouter,
            DeepSeekProfile::Gateway {
                base_url: "https://api.kilo.ai".into(),
            },
            DeepSeekProfile::Compatible {
                base_url: "http://127.0.0.1:8000".into(),
            },
        ] {
            let q = quirks_for(&profile);
            assert!(
                q.requires_reasoning_replay_with_tools,
                "V4 profile {profile:?} must replay reasoning"
            );
            assert!(q.requires_assistant_content_with_tool_calls);
        }
        let local = quirks_for(&DeepSeekProfile::LocalDerivative {
            base_url: "http://127.0.0.1:8000".into(),
        });
        assert!(!local.requires_reasoning_replay_with_tools);
        assert!(!local.requires_assistant_content_with_tool_calls);
    }

    #[tokio::test]
    async fn prior_reasoning_with_tools_is_replayed_as_reasoning_content() {
        // A tool-iteration request after a reasoning turn: the assistant
        // message must carry message-level `reasoning_content` (never a
        // {type:"reasoning"} content block next to tool_calls) and the tool
        // result must be a role:"tool" message.
        let server = MockServer::new();
        server.route(
            "POST",
            "/chat/completions",
            MockAction::AssertThenRespond {
                status: 200,
                body: String::new(),
                assert: Arc::new(|body: &serde_json::Value| {
                    let raw = serde_json::to_string(body).unwrap();
                    for banned in [
                        "\"type\":\"tool_call\"",
                        "\"type\":\"tool_result\"",
                        "\"type\":\"reasoning\"",
                    ] {
                        assert!(!raw.contains(banned), "banned block {banned} in {raw}");
                    }
                    let msgs = body["messages"].as_array().unwrap();
                    assert_eq!(msgs.len(), 2);
                    assert_eq!(msgs[0]["role"], "assistant");
                    assert_eq!(
                        msgs[0]["reasoning_content"], "think about it",
                        "reasoning must replay at message level"
                    );
                    assert_eq!(msgs[0]["content"], "", "no text → empty string content");
                    assert_eq!(msgs[0]["tool_calls"][0]["id"], "call_1");
                    assert_eq!(
                        msgs[0]["tool_calls"][0]["function"]["arguments"],
                        r#"{"x":1}"#
                    );
                    assert_eq!(msgs[1]["role"], "tool");
                    assert_eq!(msgs[1]["tool_call_id"], "call_1");
                }),
            },
        );
        let base = server.base_url().await;
        let cfg = DeepSeekConfig {
            profile: DeepSeekProfile::Direct,
            api_key: None,
            model_overrides: std::collections::HashMap::new(),
        };
        // Override the endpoint so the Direct profile hits the mock.
        let mut cfg = cfg;
        cfg.profile = DeepSeekProfile::Compatible {
            base_url: base.clone(),
        };
        let provider = build(cfg);
        let mut r = req("deepseek-v4-flash");
        r.messages = vec![
            RequestMessage {
                role: Role::Assistant,
                content: vec![
                    ContentPart::reasoning("think about it"),
                    ContentPart::tool_call("call_1", "echo", serde_json::json!({"x": 1})),
                ],
            },
            RequestMessage {
                role: Role::User,
                content: vec![ContentPart::tool_result("echo: {\"x\":1}", false, "call_1")],
            },
        ];
        let mut stream = provider.stream(r);
        while let Some(chunk) = stream.next().await {
            if let Ok(ProviderChunk::Done) = chunk {
                break;
            }
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
