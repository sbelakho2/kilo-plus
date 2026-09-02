//! Daemon configuration (kilo-plus.json). Provider keys are referenced by
//! environment variable name — the runtime never stores secrets.

use std::path::Path;
use std::sync::Arc;

use kilop_core::model::ModelCapabilities;
use kilop_provider::Provider;

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct Config {
    pub model: String,
    pub compaction_model: Option<String>,
    pub compact_at_usage: f64,
    pub instructions: String,
    pub providers: Vec<ProviderCfg>,
    /// MCP servers (spec §31): each entry spawns one supervised stdio
    /// server whose dynamic tools are surfaced into the agent registry.
    pub mcp: Vec<McpEntry>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct McpEntry {
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: "default".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions:
                "You are Kilo+.\nAct as a careful senior engineer inside the user's repository."
                    .into(),
            providers: vec![],
            mcp: vec![],
        }
    }
}
/// Production MCP bounds (hostile configs are rejected, never spawned).
pub const MAX_MCP_SERVERS: usize = 8;
pub const MAX_MCP_NAME_BYTES: usize = 128;
pub const MAX_MCP_COMMAND_BYTES: usize = 512;
pub const MAX_MCP_ARGS: usize = 32;
pub const MAX_MCP_ARG_BYTES: usize = 512;

impl Config {
    /// Validate the configured MCP surface: bounded entries with sane
    /// names/commands; empty names and oversized anything are malformed.
    pub fn mcp_servers(&self) -> Result<Vec<McpEntry>, String> {
        if self.mcp.len() > MAX_MCP_SERVERS {
            return Err(format!(
                "mcp: {} servers exceed the cap of {MAX_MCP_SERVERS}",
                self.mcp.len()
            ));
        }
        let mut names = std::collections::HashSet::new();
        for e in &self.mcp {
            if e.name.is_empty() || e.name.len() > MAX_MCP_NAME_BYTES {
                return Err(format!("mcp: name {:?} is empty or oversized", e.name));
            }
            if e.command.is_empty() || e.command.len() > MAX_MCP_COMMAND_BYTES {
                return Err(format!(
                    "mcp: command for {:?} is empty or oversized",
                    e.name
                ));
            }
            if e.args.len() > MAX_MCP_ARGS {
                return Err(format!("mcp: {:?} has too many args", e.name));
            }
            for a in &e.args {
                if a.len() > MAX_MCP_ARG_BYTES {
                    return Err(format!("mcp: {:?} has an oversized arg", e.name));
                }
            }
            if !names.insert(e.name.clone()) {
                return Err(format!("mcp: duplicate server name {:?}", e.name));
            }
        }
        Ok(self.mcp.clone())
    }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProviderCfg {
    Ollama {
        id: String,
        base_url: Option<String>,
    },
    OpenAi {
        id: String,
        base_url: String,
        api_key_env: Option<String>,
    },
    Anthropic {
        id: String,
        api_key_env: Option<String>,
    },
    Google {
        id: String,
        api_key_env: Option<String>,
    },
    DeepSeek {
        id: String,
        profile: String,
        base_url: Option<String>,
        api_key_env: Option<String>,
    },
    Gateway {
        id: String,
        base_url: String,
        api_key_env: Option<String>,
    },
}

impl ProviderCfg {
    pub fn id(&self) -> &str {
        match self {
            ProviderCfg::Ollama { id, .. }
            | ProviderCfg::OpenAi { id, .. }
            | ProviderCfg::Anthropic { id, .. }
            | ProviderCfg::Google { id, .. }
            | ProviderCfg::DeepSeek { id, .. }
            | ProviderCfg::Gateway { id, .. } => id,
        }
    }

    fn key(&self) -> Option<String> {
        let env = match self {
            ProviderCfg::Ollama { .. } => return None,
            ProviderCfg::OpenAi { api_key_env, .. }
            | ProviderCfg::Anthropic { api_key_env, .. }
            | ProviderCfg::Google { api_key_env, .. }
            | ProviderCfg::DeepSeek { api_key_env, .. }
            | ProviderCfg::Gateway { api_key_env, .. } => api_key_env,
        };
        env.as_ref().and_then(|name| std::env::var(name).ok())
    }

    /// Build the adapter for this config entry. Every provider is wrapped
    /// with its CONFIGURED instance id so the registry resolves by id (two
    /// OpenAI-compatible endpoints never overwrite each other; the adapter's
    /// family id stays for capability queries).
    pub fn build(&self) -> Result<Arc<dyn Provider>, String> {
        let instance = self.id();
        let provider = match self {
            ProviderCfg::Ollama { base_url, .. } => {
                let cfg = kilop_ollama::OllamaConfig::new(base_url.clone());
                kilop_ollama::OllamaProvider::build(cfg)
            }
            ProviderCfg::OpenAi { base_url, .. } => {
                let cfg = kilop_openai::OpenAiConfig::chat(base_url, self.key());
                kilop_openai::OpenAiProvider::build(cfg)
            }
            ProviderCfg::Anthropic { .. } => {
                let cfg = kilop_anthropic::AnthropicConfig::new(self.key());
                kilop_anthropic::AnthropicProvider::build(cfg)
            }
            ProviderCfg::Google { .. } => {
                let cfg = kilop_google::GoogleConfig::new(self.key());
                kilop_google::GoogleProvider::build(cfg)
            }
            ProviderCfg::DeepSeek {
                profile, base_url, ..
            } => {
                let cfg = match profile.as_str() {
                    // The direct profile honors a configured base_url (a
                    // DeepSeek-compatible local/proxy endpoint): default is
                    // the native api.deepseek.com.
                    "direct" => {
                        let mut c = kilop_deepseek::DeepSeekConfig::direct(self.key());
                        if let Some(b) = base_url.clone() {
                            c.profile = kilop_deepseek::DeepSeekProfile::Compatible { base_url: b };
                        }
                        c
                    }
                    "gateway" => kilop_deepseek::DeepSeekConfig {
                        profile: kilop_deepseek::DeepSeekProfile::Gateway {
                            base_url: base_url
                                .clone()
                                .unwrap_or_else(|| "https://api.kilo.ai".into()),
                        },
                        api_key: self.key(),
                        model_overrides: Default::default(),
                    },
                    "openrouter" => kilop_deepseek::DeepSeekConfig {
                        profile: kilop_deepseek::DeepSeekProfile::OpenRouter,
                        api_key: self.key(),
                        model_overrides: Default::default(),
                    },
                    "compatible" => kilop_deepseek::DeepSeekConfig::compatible(
                        base_url
                            .clone()
                            .unwrap_or_else(|| "http://127.0.0.1:8000".into()),
                        self.key(),
                    ),
                    "local" => kilop_deepseek::DeepSeekConfig {
                        profile: kilop_deepseek::DeepSeekProfile::LocalDerivative {
                            base_url: base_url
                                .clone()
                                .unwrap_or_else(|| "http://127.0.0.1:8000".into()),
                        },
                        api_key: self.key(),
                        model_overrides: Default::default(),
                    },
                    other => {
                        return Err(format!("unknown deepseek profile {other:?}"));
                    }
                };
                kilop_deepseek::build(cfg)
            }
            ProviderCfg::Gateway { base_url, .. } => {
                let cfg = kilop_gateway::GatewayConfig {
                    id: "gateway".into(),
                    base_url: base_url.clone(),
                    api_key: self.key(),
                    extra_headers: vec![],
                    route_prefixes: vec![],
                    default_caps: ModelCapabilities::default(),
                };
                kilop_gateway::build(cfg)
            }
        };
        Ok(kilop_provider::InstanceProvider::wrap(provider, instance))
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let cfg: Config = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        Ok(cfg)
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, text).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_roundtrip_and_defaults() {
        let cfg = Config::default();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("kilo-plus.json");
        cfg.save(&path).unwrap();
        let loaded = Config::load(&path).unwrap();
        assert_eq!(loaded.model, cfg.model);
        assert_eq!(loaded.compact_at_usage, 0.65);
        assert!(loaded.providers.is_empty());
    }

    #[test]
    fn hostile_config_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.json");
        std::fs::write(&path, "{not json").unwrap();
        assert!(Config::load(&path).is_err());
        std::fs::write(&path, r#"{"providers": [{"kind": "nonsense"}]}"#).unwrap();
        assert!(Config::load(&path).is_err());
    }

    #[test]
    fn keys_read_from_env_not_file() {
        std::env::set_var("KP_TEST_KEY", "secret-value");
        let cfg = ProviderCfg::OpenAi {
            id: "t".into(),
            base_url: "http://x".into(),
            api_key_env: Some("KP_TEST_KEY".into()),
        };
        assert_eq!(cfg.key().as_deref(), Some("secret-value"));
        std::env::remove_var("KP_TEST_KEY");
        assert_eq!(
            cfg.key(),
            None,
            "missing env = no key, never a stored secret"
        );
    }

    #[test]
    fn provider_ids_are_stable() {
        let cfg = ProviderCfg::Ollama {
            id: "ollama".into(),
            base_url: None,
        };
        assert_eq!(cfg.id(), "ollama");
    }

    #[test]
    fn built_providers_register_under_configured_instance_ids() {
        // Two OpenAI-compatible endpoints with distinct configured ids:
        // both must register and resolve by their ids (the old registry
        // keyed the adapter family id "openai", so the second overwrote
        // the first and custom ids never looked up).
        let mut registry = kilop_provider::ProviderRegistry::new();
        for id in ["corp-proxy", "dev-proxy"] {
            let cfg = ProviderCfg::OpenAi {
                id: id.into(),
                base_url: format!("https://{id}.example.com/v1"),
                api_key_env: None,
            };
            registry.register(cfg.build().unwrap());
        }
        assert_eq!(registry.ids(), vec!["corp-proxy", "dev-proxy"]);
        assert!(registry.get("corp-proxy").is_some());
        assert!(registry.get("dev-proxy").is_some());
        assert!(
            registry.get("openai").is_none(),
            "family id must not resolve"
        );
    }

    #[test]
    fn deepseek_profiles_build_including_gateway_and_direct_base() {
        // The DeepSeek matrix (spec §11): every profile string in the
        // config builds a provider — including "gateway" (previously an
        // unparseable arm) and "direct" with a custom base_url.
        let mut registry = kilop_provider::ProviderRegistry::new();
        for (profile, base) in [
            ("direct", None),
            ("direct", Some("http://127.0.0.1:9000")),
            ("gateway", Some("https://gw.example.com")),
            ("openrouter", None),
            ("compatible", Some("http://127.0.0.1:8000")),
            ("local", Some("http://127.0.0.1:8000")),
        ] {
            let cfg = ProviderCfg::DeepSeek {
                id: format!("ds-{profile}-{}", base.is_some()),
                profile: profile.into(),
                base_url: base.map(|b| b.to_string()),
                api_key_env: None,
            };
            let provider = cfg
                .build()
                .unwrap_or_else(|e| panic!("{profile:?} build: {e}"));
            registry.register(provider);
        }
        assert!(registry.get("ds-gateway-true").is_some());
        assert!(registry.get("ds-direct-true").is_some());
        assert!(registry.get("ds-direct-false").is_some());
        // Unknown profiles stay loud.
        let cfg = ProviderCfg::DeepSeek {
            id: "x".into(),
            profile: "bogus".into(),
            base_url: None,
            api_key_env: None,
        };
        assert!(cfg.build().is_err());
    }

    #[test]
    fn mcp_config_validation_bounds_and_duplicates() {
        // Spec §31 hostile configs are rejected, never spawned.
        let mut cfg = Config::default();
        assert!(cfg.mcp_servers().unwrap().is_empty());
        cfg.mcp.push(McpEntry {
            name: "server".into(),
            command: "python3".into(),
            args: vec!["-m".into(), "srv".into()],
        });
        assert_eq!(cfg.mcp_servers().unwrap().len(), 1);
        // Duplicate names.
        cfg.mcp.push(McpEntry {
            name: "server".into(),
            command: "python3".into(),
            args: vec![],
        });
        assert!(cfg.mcp_servers().is_err(), "duplicate names rejected");
        cfg.mcp.pop();
        // Empty names/commands and oversized entries.
        for bad in [
            McpEntry {
                name: String::new(),
                command: "x".into(),
                args: vec![],
            },
            McpEntry {
                name: "n".into(),
                command: String::new(),
                args: vec![],
            },
            McpEntry {
                name: "x".repeat(200),
                command: "c".into(),
                args: vec![],
            },
            McpEntry {
                name: "n".into(),
                command: "c".into(),
                args: vec!["a".repeat(600)],
            },
            McpEntry {
                name: "n".into(),
                command: "c".into(),
                args: vec!["a".into(); MAX_MCP_ARGS + 1],
            },
        ] {
            cfg.mcp.push(bad);
            assert!(cfg.mcp_servers().is_err(), "hostile entry rejected");
            cfg.mcp.pop();
        }
        // Too many servers.
        cfg.mcp = (0..MAX_MCP_SERVERS + 1)
            .map(|i| McpEntry {
                name: format!("s{i}"),
                command: "c".into(),
                args: vec![],
            })
            .collect();
        assert!(cfg.mcp_servers().is_err(), "server count capped");
        // Round-trips through the file config loader.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("kilo-plus.json");
        std::fs::write(
            &path,
            r#"{"mcp": [{"name": "fixture", "command": "python3", "args": ["mock.py"]}]}"#,
        )
        .unwrap();
        let loaded = Config::load(&path).unwrap();
        assert_eq!(loaded.mcp.len(), 1);
        assert_eq!(loaded.mcp[0].name, "fixture");
    }
}
