//! Daemon configuration (faktor-plus.json). Provider keys are referenced by
//! environment variable name — the runtime never stores secrets.

use std::path::Path;
use std::sync::Arc;

use faktor_core::model::{ModelCapabilities, RoutingMode};
use faktor_provider::egress::HttpTransport;
use faktor_provider::Provider;
use faktor_sandbox::{NetworkGate, SandboxGuarantee, SandboxPolicy};

#[derive(Debug, Clone, serde::Serialize)]
pub struct Config {
    pub model: String,
    pub compaction_model: Option<String>,
    pub compact_at_usage: f64,
    pub instructions: String,
    pub providers: Vec<ProviderCfg>,
    /// Economic routing mode of the daemon (P0-2): `None` = Economy — every
    /// model call routes through the RouterService built from the
    /// registered providers. `Pinned { provider, model }` validates every
    /// call against the pin. Strictly additive; the file shape accepts it
    /// since config_version 1 with `serde(default)`.
    pub routing_mode: Option<RoutingMode>,
    /// MCP servers (spec §31): each entry spawns one supervised stdio
    /// server whose dynamic tools are surfaced into the agent registry.
    pub mcp: Vec<McpEntry>,
    /// The additive `[verification]` section: per-category check budgets.
    pub verification: VerificationCfg,
    /// The additive `[sandbox]` section: the daemon's network destination
    /// allowlist and the OS-level network-isolation guarantee.
    pub sandbox: SandboxCfg,
}

/// The additive `[verification]` section (daemon verification policy).
/// Strictly additive with `serde(default)`: an absent section (or absent
/// keys inside it) keep the crate defaults (quick ≤ 60 s, unit ≤ 600 s
/// inline, full in background). `quick_max_s: 0` disables the verification
/// service entirely (fail closed — mutating turns classify Unverified).
/// Unknown keys inside the section are parse errors (strict both on the
/// lenient and the strict load path).
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct VerificationCfg {
    #[serde(default = "default_quick_max_s")]
    pub quick_max_s: u64,
    #[serde(default = "default_unit_max_s")]
    pub unit_max_s: u64,
    #[serde(default = "default_full_as_background")]
    pub full_as_background: bool,
}

fn default_quick_max_s() -> u64 {
    60
}
fn default_unit_max_s() -> u64 {
    600
}
fn default_full_as_background() -> bool {
    true
}

impl Default for VerificationCfg {
    fn default() -> Self {
        Self {
            quick_max_s: default_quick_max_s(),
            unit_max_s: default_unit_max_s(),
            full_as_background: default_full_as_background(),
        }
    }
}

/// The additive `[sandbox]` section (daemon sandbox policy overrides).
/// `network` rows are parsed destination-allowlist rules in the security
/// crate's rule syntax (e.g. `http://127.0.0.1:8080`); `None` keeps the
/// sandbox crate's frozen default provider-endpoint allowlist, while an
/// explicit list — even an empty one (deny-all) — replaces it. The
/// `network_guarantee` (`none` default, `best_effort`, `required`) declares
/// what the policy requires of OS-level network isolation for shell
/// commands. Unknown keys inside the section are parse errors.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize, Default)]
#[serde(deny_unknown_fields)]
pub struct SandboxCfg {
    #[serde(default)]
    pub network: Option<Vec<String>>,
    #[serde(default)]
    pub network_guarantee: SandboxGuarantee,
}

/// The config FILE shape: `Config` plus `config_version` (default 1 when
/// the key is absent). Deserialization is STRICT: unknown fields anywhere
/// are rejected (a typo'd key fails startup instead of silently changing
/// behavior), and any `config_version` other than 1 is a parse error.
impl<'de> serde::Deserialize<'de> for Config {
    fn deserialize<D>(de: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error as _;
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct File {
            #[serde(default = "default_config_version")]
            config_version: u32,
            #[serde(default)]
            model: String,
            #[serde(default)]
            compaction_model: Option<String>,
            #[serde(default)]
            compact_at_usage: f64,
            #[serde(default)]
            instructions: String,
            #[serde(default)]
            providers: Vec<ProviderCfg>,
            /// `"economy"` (the default when absent), or
            /// `{"pinned": {"provider": "…", "model": "…"}}`. Anything else
            /// is a parse error (hostile configs are rejected, never
            /// half-honored).
            #[serde(default)]
            routing_mode: Option<RoutingMode>,
            #[serde(default)]
            mcp: Vec<McpEntry>,
            #[serde(default)]
            verification: VerificationCfg,
            #[serde(default)]
            sandbox: SandboxCfg,
        }
        let file = File::deserialize(de)?;
        if file.config_version != 1 {
            return Err(D::Error::custom(format!(
                "unsupported config_version {}; this build accepts only config_version 1",
                file.config_version
            )));
        }
        Ok(Self {
            model: file.model,
            compaction_model: file.compaction_model,
            compact_at_usage: file.compact_at_usage,
            instructions: file.instructions,
            providers: file.providers,
            routing_mode: file.routing_mode,
            mcp: file.mcp,
            verification: file.verification,
            sandbox: file.sandbox,
        })
    }
}

fn default_config_version() -> u32 {
    1
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
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
                "You are Faktor.\nAct as a careful senior engineer inside the user's repository."
                    .into(),
            providers: vec![],
            routing_mode: None,
            mcp: vec![],
            verification: VerificationCfg::default(),
            sandbox: SandboxCfg::default(),
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

impl VerificationCfg {
    /// The verification policy this section resolves to. `None` when
    /// `quick_max_s` is 0: the verification service is DISABLED (fail
    /// closed — mutating turns classify Unverified, never silently
    /// complete). Sane values map onto the crate policy whose per-category
    /// budgets gate every check (`budget_for`).
    pub fn policy(&self) -> Option<faktor_verify::exec::VerificationPolicy> {
        if self.quick_max_s == 0 {
            return None;
        }
        Some(faktor_verify::exec::VerificationPolicy {
            quick_max: std::time::Duration::from_secs(self.quick_max_s),
            unit_max: std::time::Duration::from_secs(self.unit_max_s),
            full_as_background: self.full_as_background,
            min_inline: std::time::Duration::from_secs(5),
        })
    }
}

impl Config {
    /// The daemon sandbox policy this config resolves to (pure mapping,
    /// used by the daemon build and by strict config validation). The
    /// section overrides the sandbox crate's defaults: an explicit
    /// `network` row list replaces the network gate (parsed strictly — one
    /// unparseable rule fails the whole policy), and the configured
    /// guarantee rides into `SandboxPolicy::network_guarantee`.
    pub fn sandbox_policy(&self) -> Result<SandboxPolicy, String> {
        let mut policy = SandboxPolicy::default();
        if let Some(rows) = &self.sandbox.network {
            policy.network =
                NetworkGate::parse(rows).map_err(|e| format!("network rule error: {e}"))?;
        }
        policy.network_guarantee = self.sandbox.network_guarantee;
        Ok(policy)
    }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
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

    /// The configured key read from its env var (never stored in the file;
    /// the runtime never logs or persists the value). `None` when the entry
    /// carries no key env or the env var is unset. Exposed for the daemon's
    /// outbound secret registry, which registers the SAME values the
    /// adapter builds from.
    pub(crate) fn key(&self) -> Option<String> {
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

    /// Build the adapter for this config entry over an explicit egress
    /// transport (the daemon passes the policy-checked transport built from
    /// its SandboxPolicy network gate + outbound secret scan; tests pass a
    /// default-allow one). Every provider is wrapped with its CONFIGURED
    /// instance id so the registry resolves by id (two OpenAI-compatible
    /// endpoints never overwrite each other; the adapter's family id stays
    /// for capability queries).
    /// Concrete Ollama provider when this entry configures one (the daemon
    /// warm-up keeps the concrete Arc so live probing reaches the SAME
    /// instance the registry serves).
    pub fn build_ollama(
        &self,
        transport: Arc<dyn HttpTransport>,
    ) -> Option<Arc<faktor_ollama::OllamaProvider>> {
        match self {
            ProviderCfg::Ollama { base_url, .. } => {
                let cfg = faktor_ollama::OllamaConfig::new(base_url.clone());
                Some(faktor_ollama::OllamaProvider::new_with_transport(
                    cfg, transport,
                ))
            }
            _ => None,
        }
    }

    pub fn build(&self, transport: Arc<dyn HttpTransport>) -> Result<Arc<dyn Provider>, String> {
        let instance = self.id();
        let provider: Arc<dyn Provider> = match self {
            ProviderCfg::Ollama { base_url, .. } => {
                let cfg = faktor_ollama::OllamaConfig::new(base_url.clone());
                faktor_ollama::OllamaProvider::new_with_transport(cfg, transport.clone())
            }
            ProviderCfg::OpenAi { base_url, .. } => {
                let cfg = faktor_openai::OpenAiConfig::chat(base_url, self.key());
                faktor_openai::OpenAiProvider::build_with_transport(cfg, transport.clone())
            }
            ProviderCfg::Anthropic { .. } => {
                let cfg = faktor_anthropic::AnthropicConfig::new(self.key());
                faktor_anthropic::AnthropicProvider::build_with_transport(cfg, transport.clone())
            }
            ProviderCfg::Google { .. } => {
                let cfg = faktor_google::GoogleConfig::new(self.key());
                faktor_google::GoogleProvider::build_with_transport(cfg, transport.clone())
            }
            ProviderCfg::DeepSeek {
                profile, base_url, ..
            } => {
                let cfg = match profile.as_str() {
                    // The direct profile honors a configured base_url (a
                    // DeepSeek-compatible local/proxy endpoint): default is
                    // the native api.deepseek.com.
                    "direct" => {
                        let mut c = faktor_deepseek::DeepSeekConfig::direct(self.key());
                        if let Some(b) = base_url.clone() {
                            c.profile =
                                faktor_deepseek::DeepSeekProfile::Compatible { base_url: b };
                        }
                        c
                    }
                    "gateway" => faktor_deepseek::DeepSeekConfig {
                        profile: faktor_deepseek::DeepSeekProfile::Gateway {
                            base_url: base_url
                                .clone()
                                .unwrap_or_else(|| "https://api.kilo.ai".into()),
                        },
                        api_key: self.key(),
                        model_overrides: Default::default(),
                    },
                    "openrouter" => faktor_deepseek::DeepSeekConfig {
                        profile: faktor_deepseek::DeepSeekProfile::OpenRouter,
                        api_key: self.key(),
                        model_overrides: Default::default(),
                    },
                    "compatible" => faktor_deepseek::DeepSeekConfig::compatible(
                        base_url
                            .clone()
                            .unwrap_or_else(|| "http://127.0.0.1:8000".into()),
                        self.key(),
                    ),
                    "local" => faktor_deepseek::DeepSeekConfig {
                        profile: faktor_deepseek::DeepSeekProfile::LocalDerivative {
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
                faktor_deepseek::build_with_transport(cfg, transport.clone())
            }
            ProviderCfg::Gateway { base_url, .. } => {
                let cfg = faktor_gateway::GatewayConfig {
                    id: "gateway".into(),
                    base_url: base_url.clone(),
                    api_key: self.key(),
                    extra_headers: vec![],
                    route_prefixes: vec![],
                    default_caps: ModelCapabilities::default(),
                };
                faktor_gateway::build_with_transport(cfg, transport.clone())
            }
        };
        Ok(faktor_provider::InstanceProvider::wrap(provider, instance))
    }
}

impl Config {
    /// Parse a config file. Lenient in the sense that it only parses (the
    /// strict `deny_unknown_fields`/`config_version` layer is inside
    /// deserialization); it does NOT run semantic validation. Default-only
    /// paths never call this.
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let cfg: Config = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        Ok(cfg)
    }

    /// Strict load for EXPLICIT --config paths: any parse error, unknown
    /// field, unsupported `config_version`, or semantic validation failure
    /// (duplicate provider ids, hostile MCP bounds) fails startup — the
    /// daemon never boots on a config it cannot fully honor.
    pub fn load_strict(path: &Path) -> Result<Self, String> {
        let cfg = Self::load(path)?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Semantic validation: duplicate provider ids are rejected (the error
    /// lists every duplicate), the MCP surface must satisfy its own
    /// hostile-config bounds, and the sandbox section's destination rows
    /// must all parse (a rule that cannot parse is a config error, never
    /// silently permissive).
    pub fn validate(&self) -> Result<(), String> {
        self.mcp_servers()?;
        self.sandbox_policy()
            .map_err(|e| format!("sandbox config: {e}"))?;
        let mut seen = std::collections::HashSet::new();
        let mut dupes: Vec<String> = Vec::new();
        for p in &self.providers {
            let id = p.id().to_string();
            if !seen.insert(id.clone()) {
                dupes.push(id);
            }
        }
        dupes.sort();
        dupes.dedup();
        if !dupes.is_empty() {
            return Err(format!("duplicate provider id(s): {}", dupes.join(", ")));
        }
        Ok(())
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
        let path = dir.path().join("faktor-plus.json");
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
    fn unknown_fields_are_rejected_everywhere() {
        // Audit 39: strict configs. Unknown top-level keys and unknown keys
        // inside provider/mcp entries are parse errors, never silent noise.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("strict.json");
        for bad in [
            r#"{"model": "m", "surprise_field": true}"#,
            r#"{"providers": [{"kind": "ollama", "id": "o", "bogus": 1}]}"#,
            r#"{"providers": [{"kind": "open_ai", "id": "a", "base_url": "u", "api_key_env": null, "bogus": "x"}]}"#,
            r#"{"mcp": [{"name": "s", "command": "c", "args": [], "bogus": true}]}"#,
        ] {
            std::fs::write(&path, bad).unwrap();
            let e = Config::load(&path).expect_err("hostile config must fail");
            assert!(
                e.contains("unknown field"),
                "expected an unknown-field error, got: {e}"
            );
        }
    }

    #[test]
    fn config_version_defaults_to_one_and_rejects_others() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.json");
        // Absent -> default 1.
        std::fs::write(&path, r#"{"model": "m"}"#).unwrap();
        Config::load(&path).expect("absent config_version defaults to 1");
        // Explicit 1 -> fine.
        std::fs::write(&path, r#"{"config_version": 1, "model": "m"}"#).unwrap();
        Config::load(&path).expect("config_version 1 is accepted");
        // Anything else -> rejected at parse, on both load paths.
        for v in [0u32, 2, 7, 999] {
            std::fs::write(&path, format!(r#"{{"config_version": {v}}}"#)).unwrap();
            let e = Config::load(&path).expect_err("unsupported version must fail");
            assert!(e.contains("config_version"), "{e}");
            assert!(Config::load_strict(&path).is_err());
        }
    }

    #[test]
    fn validate_rejects_duplicate_provider_ids() {
        let cfg = Config {
            providers: vec![
                ProviderCfg::Ollama {
                    id: "dup".into(),
                    base_url: None,
                },
                ProviderCfg::Ollama {
                    id: "other".into(),
                    base_url: None,
                },
                ProviderCfg::OpenAi {
                    id: "dup".into(),
                    base_url: "http://x".into(),
                    api_key_env: None,
                },
            ],
            ..Default::default()
        };
        let e = cfg.validate().expect_err("duplicates must be rejected");
        assert!(
            e.contains("dup") && !e.contains("other"),
            "the error lists the duplicate id, got: {e}"
        );
        let cfg = Config {
            providers: vec![
                cfg.providers[0].clone(),
                ProviderCfg::OpenAi {
                    id: "distinct".into(),
                    base_url: "http://y".into(),
                    api_key_env: None,
                },
            ],
            ..Default::default()
        };
        cfg.validate().expect("distinct provider ids are fine");
    }

    #[test]
    fn load_strict_rejects_malformed_and_invalid_semantics() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("strict.json");
        // Malformed JSON.
        std::fs::write(&path, "{not json").unwrap();
        assert!(Config::load_strict(&path).is_err());
        // Unknown top-level field.
        std::fs::write(&path, r#"{"model": "m", "extra": 1}"#).unwrap();
        assert!(Config::load_strict(&path).is_err());
        // Unknown field inside a provider entry.
        std::fs::write(
            &path,
            r#"{"providers": [{"kind": "ollama", "id": "o", "zzz": 1}]}"#,
        )
        .unwrap();
        assert!(Config::load_strict(&path).is_err());
        // Unsupported config_version.
        std::fs::write(&path, r#"{"config_version": 3}"#).unwrap();
        assert!(Config::load_strict(&path).is_err());
        // Semantic failure: duplicate provider ids.
        std::fs::write(
            &path,
            r#"{"providers": [
                {"kind": "ollama", "id": "twice", "base_url": null},
                {"kind": "open_ai", "id": "twice", "base_url": "http://x"}
            ]}"#,
        )
        .unwrap();
        let e = Config::load_strict(&path).expect_err("duplicate ids must fail strict load");
        assert!(e.contains("twice"), "{e}");
        // A healthy explicit config still loads strictly.
        std::fs::write(
            &path,
            r#"{"config_version": 1, "model": "m", "providers": [
                {"kind": "ollama", "id": "a", "base_url": null},
                {"kind": "open_ai", "id": "b", "base_url": "http://x"}
            ]}"#,
        )
        .unwrap();
        let cfg = Config::load_strict(&path).unwrap();
        assert_eq!(cfg.model, "m");
        assert_eq!(cfg.providers.len(), 2);
    }

    #[test]
    fn routing_mode_parses_economy_default_and_pinned_and_rejects_hostile() {
        // Absent -> None (the daemon treats None as Economy).
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("r.json");
        std::fs::write(&path, r#"{"model": "m"}"#).unwrap();
        let cfg = Config::load(&path).unwrap();
        assert_eq!(cfg.routing_mode, None);
        // Economy string.
        std::fs::write(&path, r#"{"routing_mode": "economy"}"#).unwrap();
        assert_eq!(
            Config::load(&path).unwrap().routing_mode,
            Some(RoutingMode::Economy)
        );
        // Pinned object.
        std::fs::write(
            &path,
            r#"{"routing_mode": {"pinned": {"provider": "deepseek", "model": "deepseek-chat"}}}"#,
        )
        .unwrap();
        let cfg = Config::load(&path).unwrap();
        assert_eq!(
            cfg.routing_mode,
            Some(RoutingMode::Pinned {
                provider: "deepseek".into(),
                model: "deepseek-chat".into(),
            })
        );
        // Round-trip through save/load (the daemon's own default file).
        cfg.save(&path).unwrap();
        assert_eq!(Config::load(&path).unwrap().routing_mode, cfg.routing_mode);
        // Hostile shapes are rejected: the old "auto" sentinel, a pinned
        // object missing the model, and wrong-typed values.
        for bad in [
            r#"{"routing_mode": "auto"}"#,
            r#"{"routing_mode": {"pinned": {"provider": "p"}}}"#,
            r#"{"routing_mode": 42}"#,
            r#"{"routing_mode": {"mode": "economy"}}"#,
        ] {
            std::fs::write(&path, bad).unwrap();
            assert!(
                Config::load(&path).is_err(),
                "hostile routing_mode must be rejected: {bad}"
            );
        }
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
        let mut registry = faktor_provider::ProviderRegistry::new();
        for id in ["corp-proxy", "dev-proxy"] {
            let cfg = ProviderCfg::OpenAi {
                id: id.into(),
                base_url: format!("https://{id}.example.com/v1"),
                api_key_env: None,
            };
            registry.register(cfg.build(open_transport()).unwrap());
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
        let mut registry = faktor_provider::ProviderRegistry::new();
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
                .build(open_transport())
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
        assert!(cfg.build(open_transport()).is_err());
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
        let path = dir.path().join("faktor-plus.json");
        std::fs::write(
            &path,
            r#"{"mcp": [{"name": "fixture", "command": "python3", "args": ["mock.py"]}]}"#,
        )
        .unwrap();
        let loaded = Config::load(&path).unwrap();
        assert_eq!(loaded.mcp.len(), 1);
        assert_eq!(loaded.mcp[0].name, "fixture");
    }

    /// Explicit default-allow transport for construction-level unit tests
    /// (the daemon always passes the policy-checked transport built from
    /// its SandboxPolicy; these tests never exercise egress).
    fn open_transport() -> Arc<dyn HttpTransport> {
        Arc::new(faktor_provider::egress::PolicyCheckedHttpTransport::with_policy(None))
    }

    #[test]
    fn verification_section_defaults_partial_objects_and_zero_disables() {
        // Absent section -> crate defaults (60/600/background).
        let cfg = Config::default();
        assert_eq!(cfg.verification.quick_max_s, 60);
        assert_eq!(cfg.verification.unit_max_s, 600);
        assert!(cfg.verification.full_as_background);
        let policy = cfg.verification.policy().expect("defaults stay enabled");
        use faktor_verify::exec::{
            budget_for, BudgetDecision, CheckCategory, CheckKind, CheckSpec,
        };
        let spec = CheckSpec::new(
            "q",
            CheckKind::Compile,
            CheckCategory::Quick,
            "cargo",
            ["check"],
            true,
        );
        // Budget probe: the derived per-check budget is the configured cap.
        assert_eq!(
            budget_for(spec.category, &policy, None),
            BudgetDecision::RunInline(std::time::Duration::from_secs(60))
        );
        assert_eq!(
            budget_for(CheckCategory::Unit, &policy, None),
            BudgetDecision::RunInline(std::time::Duration::from_secs(600))
        );
        assert_eq!(
            budget_for(CheckCategory::Full, &policy, None),
            BudgetDecision::RunAsTaskOwnedOperation,
            "full checks go background by default"
        );
        // Partial objects fill per-key defaults (60/600/true), never 0.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.json");
        std::fs::write(
            &path,
            r#"{"verification": {"full_as_background": false, "unit_max_s": 120}}"#,
        )
        .unwrap();
        let cfg = Config::load(&path).unwrap();
        assert_eq!(
            cfg.verification.quick_max_s, 60,
            "partial keeps quick default"
        );
        assert_eq!(cfg.verification.unit_max_s, 120);
        assert!(!cfg.verification.full_as_background);
        let policy = cfg.verification.policy().unwrap();
        assert_eq!(
            budget_for(CheckCategory::Quick, &policy, None),
            BudgetDecision::RunInline(std::time::Duration::from_secs(60))
        );
        assert_eq!(
            budget_for(CheckCategory::Full, &policy, None),
            BudgetDecision::RunInline(std::time::Duration::from_secs(120)),
            "full_as_background: false keeps full checks inline under unit_max"
        );
        // quick_max_s = 0 disables the service: the mapping yields None
        // (fail closed), on the parse path AND the daemon mapping path.
        std::fs::write(
            &path,
            r#"{"verification": {"quick_max_s": 0, "unit_max_s": 0}}"#,
        )
        .unwrap();
        let cfg = Config::load(&path).unwrap();
        assert_eq!(cfg.verification.policy(), None, "quick 0 -> disabled");
    }

    #[test]
    fn verification_section_unknown_fields_fail_everywhere() {
        // Strictness stays for EXPLICIT configs: an unknown key inside
        // [verification] is a parse error on both load paths.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.json");
        for bad in [
            r#"{"verification": {"quick_max_s": 30, "bogus": 1}}"#,
            r#"{"verification": {"quick_max_s": "fast"}}"#,
        ] {
            std::fs::write(&path, bad).unwrap();
            let e = Config::load(&path).expect_err("hostile [verification] must fail");
            assert!(
                e.contains("unknown field") || e.contains("invalid type"),
                "{e}"
            );
            assert!(Config::load_strict(&path).is_err());
        }
        // The section itself deserializes strict too (a nested object under
        // the wrong name is still an unknown top-level key).
        std::fs::write(&path, r#"{"verif": {"quick_max_s": 30}}"#).unwrap();
        assert!(Config::load(&path).is_err());
    }

    #[test]
    fn sandbox_section_maps_guarantees_and_rows_strictly() {
        use faktor_sandbox::{SandboxGuarantee, SandboxPolicy};
        // Absent section: crate defaults (frozen gate, guarantee None).
        let cfg = Config::default();
        let policy = cfg.sandbox_policy().unwrap();
        assert_eq!(policy.network_guarantee, SandboxGuarantee::None);
        assert!(policy.network.installed().is_some(), "frozen allowlist");
        assert_eq!(
            policy,
            SandboxPolicy::default(),
            "absent sandbox section == sandbox defaults"
        );
        // Explicit rows replace the gate; an empty list denies everything.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.json");
        std::fs::write(
            &path,
            r#"{"sandbox": {"network": ["http://127.0.0.1:8765"], "network_guarantee": "required"}}"#,
        )
        .unwrap();
        let cfg = Config::load(&path).unwrap();
        let policy = cfg.sandbox_policy().unwrap();
        assert_eq!(
            policy.network_guarantee,
            SandboxGuarantee::Required,
            "required parses through the sandbox serde field"
        );
        assert!(policy.network.installed().is_some());
        // Defaults round-trip through the file shape.
        cfg.save(&path).unwrap();
        let loaded = Config::load(&path).unwrap();
        assert_eq!(loaded.sandbox_policy().unwrap(), policy);
        // best_effort parses; a hostile guarantee value is a parse error;
        // unknown keys inside [sandbox] are rejected.
        for (text, expect) in [
            (
                r#"{"sandbox": {"network_guarantee": "best_effort"}}"#,
                SandboxGuarantee::BestEffort,
            ),
            (
                r#"{"sandbox": {"network_guarantee": "none"}}"#,
                SandboxGuarantee::None,
            ),
        ] {
            std::fs::write(&path, text).unwrap();
            let cfg = Config::load(&path).unwrap();
            assert_eq!(cfg.sandbox_policy().unwrap().network_guarantee, expect);
        }
        for bad in [
            r#"{"sandbox": {"network_guarantee": "mandatory"}}"#,
            r#"{"sandbox": {"network": ["http://127.0.0.1:1"], "bogus": true}}"#,
        ] {
            std::fs::write(&path, bad).unwrap();
            assert!(
                Config::load(&path).is_err(),
                "hostile [sandbox] must be rejected: {bad}"
            );
        }
        // A rule that cannot parse is a semantic (validate/strict-load and
        // daemon-policy) error — never silently permissive.
        std::fs::write(&path, r#"{"sandbox": {"network": ["not a url"]}}"#).unwrap();
        let cfg = Config::load(&path).unwrap();
        let e = cfg.sandbox_policy().expect_err("unparseable rule fails");
        assert!(!e.is_empty());
        assert!(Config::load_strict(&path).is_err());
    }
}
