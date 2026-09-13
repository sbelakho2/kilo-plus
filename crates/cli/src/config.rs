//! Daemon configuration (faktor-plus.json). Provider keys are referenced by
//! environment variable name — the runtime never stores secrets.

use std::path::Path;
use std::sync::Arc;

use faktor_core::model::{
    BillingOrigin, MicroUsdPerMillionTokens, ModelCapabilities, PriceQuote, RoutingMode,
};
use faktor_orchestrator::runtime::task_executor::MutationMode;
use faktor_provider::catalog::{BillingOriginProvider, PricingOverrides};
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
    /// The additive `[tasks]` section: native task execution policy.
    pub tasks: TasksCfg,
    /// The additive `[completion]` section (P2 completion-step execution):
    /// the strict commit/push/PR execution policy.
    pub completion: CompletionCfg,
    /// The additive `[efficiency]` section (audit 86 + the efficiency-variant
    /// production flags): five boolean feature switches, ALL default `false`.
    pub efficiency: EfficiencyCfg,
}

/// The additive `[completion]` section (P2 follow-up): how a contracted
/// run's ordered commit/push/PR steps execute.
///
/// Strict by construction: unknown keys are parse errors (derived
/// `deny_unknown_fields`), non-string values are type errors, and the
/// resolved [`faktor_orchestrator::runtime::completion_steps::CompletionStepsConfig`]
/// is validated (bounded remote/base branch; a bounded single-line PR
/// command/argv with known placeholders only and `{branch}` required; the
/// legacy `pr_command` string keeps every historic security check, while
/// the additive `pr_program`/`pr_args` typed argv preserves spaces per
/// element). An absent section keeps the inert defaults: push targets
/// `origin`, the PR base is `main`, and an unconfigured PR records the
/// documented `Skipped` outcome — a requested PR step is never silently
/// invented.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, Default)]
pub struct CompletionCfg {
    /// The git remote the push step targets (default `origin`).
    pub remote: Option<String>,
    /// The base branch rendered into the PR template (default `main`).
    pub base_branch: Option<String>,
    /// DEPRECATED whitespace-split PR command template, e.g.
    /// `"gh pr create --head {branch} --base {base}"`. `None` (the default)
    /// means a requested PR step records `Skipped` (unless `pr_program` is
    /// configured). Mutually exclusive with `pr_program`.
    pub pr_command: Option<String>,
    /// The typed argv program (P3): executed directly through the supervisor
    /// with no shell, so paths/arguments with spaces are representable.
    /// Additive to the legacy `pr_command`.
    pub pr_program: Option<String>,
    /// The typed argv arguments; every element substitutes
    /// `{branch}`/`{base}`/`{remote}` independently. Requires `pr_program`.
    pub pr_args: Vec<String>,
}

/// Map-only strict parsing: a positional JSON array must never configure the
/// section by position, duplicates are refused and unknown keys are typed
/// errors — the same discipline the completion contract itself uses.
impl<'de> serde::Deserialize<'de> for CompletionCfg {
    fn deserialize<D>(de: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{Error as _, MapAccess, Visitor};

        struct SectionVisitor;

        impl<'de> Visitor<'de> for SectionVisitor {
            type Value = CompletionCfg;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("the [completion] section as a JSON object")
            }

            fn visit_map<A>(self, mut map: A) -> Result<CompletionCfg, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut out = CompletionCfg::default();
                let mut seen: u8 = 0;
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "remote" => {
                            if seen & 1 != 0 {
                                return Err(A::Error::duplicate_field("remote"));
                            }
                            seen |= 1;
                            out.remote = map.next_value()?;
                        }
                        "base_branch" => {
                            if seen & 2 != 0 {
                                return Err(A::Error::duplicate_field("base_branch"));
                            }
                            seen |= 2;
                            out.base_branch = map.next_value()?;
                        }
                        "pr_command" => {
                            if seen & 4 != 0 {
                                return Err(A::Error::duplicate_field("pr_command"));
                            }
                            seen |= 4;
                            out.pr_command = map.next_value()?;
                        }
                        "pr_program" => {
                            if seen & 8 != 0 {
                                return Err(A::Error::duplicate_field("pr_program"));
                            }
                            seen |= 8;
                            out.pr_program = map.next_value()?;
                        }
                        "pr_args" => {
                            if seen & 16 != 0 {
                                return Err(A::Error::duplicate_field("pr_args"));
                            }
                            seen |= 16;
                            out.pr_args = map.next_value()?;
                        }
                        other => {
                            return Err(A::Error::unknown_field(
                                other,
                                &[
                                    "remote",
                                    "base_branch",
                                    "pr_command",
                                    "pr_program",
                                    "pr_args",
                                ],
                            ))
                        }
                    }
                }
                Ok(out)
            }
        }

        de.deserialize_map(SectionVisitor)
    }
}

impl CompletionCfg {
    /// Resolve and STRICTLY validate the completion-step execution policy.
    /// Any invalid remote/base/template is an error: a typo never half-runs.
    pub fn steps_config(
        &self,
    ) -> Result<faktor_orchestrator::runtime::completion_steps::CompletionStepsConfig, String> {
        let config = faktor_orchestrator::runtime::completion_steps::CompletionStepsConfig {
            remote: self.remote.clone().unwrap_or_else(|| "origin".to_string()),
            base_branch: self
                .base_branch
                .clone()
                .unwrap_or_else(|| "main".to_string()),
            pr_command: self.pr_command.clone(),
            pr_program: self.pr_program.clone(),
            pr_args: self.pr_args.clone(),
        };
        config.validate().map_err(|e| format!("completion: {e}"))?;
        Ok(config)
    }
}

/// The additive `[tasks]` section (P0-48 shadow mutation roots, wave-24
/// mutation policy).
///
/// The section is strictly additive with `serde(default)` and an absent
/// section keeping the crate default `mutation_mode: Shadow` — shadow
/// mutation is the PRODUCTION DEFAULT: single-agent MUTATING tasks work in
/// daemon-owned shadow worktrees and are integrated back into the user
/// checkout with a conflict-aware CAS commit. `mutation_mode:
/// "direct_compat"` opts one daemon back into today's direct behavior
/// (byte-identical to every wave before shadow mutation was the default).
///
/// The pre-wave-24 boolean key `shadow_mutation` is still accepted as a
/// LEGACY alias (old config files keep their meaning: `true` = shadow
/// mutation on, `false` = direct), but it is an error to specify BOTH keys
/// — the file never says two different things. Unknown keys inside the
/// section are parse errors (strict on both load paths).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, Default)]
pub struct TasksCfg {
    #[serde(default)]
    pub mutation_mode: MutationMode,
}

impl<'de> serde::Deserialize<'de> for TasksCfg {
    fn deserialize<D>(de: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error as _;
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct File {
            #[serde(default)]
            mutation_mode: Option<MutationMode>,
            /// Legacy pre-wave-24 alias (config_version 1 files written
            /// before the mode existed). `true` = shadow mutation on;
            /// `false` = the direct behavior of that era.
            #[serde(default)]
            shadow_mutation: Option<bool>,
        }
        let file = File::deserialize(de)?;
        match (file.mutation_mode, file.shadow_mutation) {
            (Some(_), Some(_)) => Err(D::Error::custom(
                "conflicting [tasks] keys: mutation_mode and the legacy shadow_mutation alias cannot both be present",
            )),
            (Some(m), None) => Ok(Self { mutation_mode: m }),
            (None, Some(true)) => Ok(Self {
                mutation_mode: MutationMode::Shadow,
            }),
            (None, Some(false)) => Ok(Self {
                mutation_mode: MutationMode::DirectCompat,
            }),
            (None, None) => Ok(Self {
                mutation_mode: MutationMode::Shadow,
            }),
        }
    }
}

/// The additive `[efficiency]` section (audit 86 + the efficiency-variant
/// production flags): five independent boolean feature switches. In
/// PRODUCTION every flag defaults ON — the efficiency system is active —
/// and an explicit `false` is the documented OFF-SWITCH for that one
/// component (an absent section keeps the production defaults; a partial
/// section flips only the keys it names):
///
/// - `failure_learning`: feed the learning crate's failure prior into
///   context selection through `faktor_context`'s `FailurePrior` planner
///   seam (audit 68). OFF installs no prior (neutral omission risk);
/// - `ccr`: compressed-context representation for tool/evidence payloads.
///   OFF keeps raw tool output byte-identical (bounded excerpts only);
/// - `typed_handoff`: re-sent history rendered from durable task rows.
///   OFF falls back to the legacy transcript rendering;
/// - `semantic_context`: information-gain selection of evidence through the
///   ContextCompiler. OFF keeps the producer evidence list byte-identical
///   (neutral sparse fallback; unit contexts stay parity);
/// - `rework_routing`: rework-aware routing over durable verified-outcome
///   stats.
///
/// The section is strictly additive: unknown keys, non-boolean values,
/// duplicate keys and non-object shapes (a JSON array must never enable
/// flags by position) are parse errors on both load paths.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct EfficiencyCfg {
    /// Failure-learning prior in context selection (audit 68).
    pub failure_learning: bool,
    /// Compressed Context Representation for tool/evidence payloads.
    pub ccr: bool,
    /// Typed handoff: re-sent history rendered from durable task rows.
    pub typed_handoff: bool,
    /// Semantic context: information-gain selection of evidence through the
    /// ContextCompiler (the audit's `information_gain` flag).
    pub semantic_context: bool,
    /// Rework-aware routing over durable verified-outcome stats.
    pub rework_routing: bool,
}

impl Default for EfficiencyCfg {
    /// The additive type-level default: every flag OFF. Unit/embedded
    /// callers that build `EfficiencyCfg::default()` keep the pre-efficiency
    /// behavior byte-for-byte; the PRODUCTION config paths use
    /// [`EfficiencyCfg::production_defaults`] (see [`Config::default`] and
    /// the serde default), which is what turns the efficiency system on for
    /// a daemon with an absent `[efficiency]` section.
    fn default() -> Self {
        Self {
            failure_learning: false,
            ccr: false,
            typed_handoff: false,
            semantic_context: false,
            rework_routing: false,
        }
    }
}

impl EfficiencyCfg {
    /// The PRODUCTION defaults (audit: the efficiency system is on by
    /// default): every component enabled, with the per-key `false` explicit
    /// off-switch documented on the struct.
    pub const fn production_defaults() -> Self {
        Self {
            failure_learning: true,
            ccr: true,
            typed_handoff: true,
            semantic_context: true,
            rework_routing: true,
        }
    }
}

/// Serde default for an absent `[efficiency]` section: production defaults
/// (all ON). A partial section flips only the keys it names.
fn production_efficiency() -> EfficiencyCfg {
    EfficiencyCfg::production_defaults()
}

/// The `[efficiency]` keys, in stable order (unknown-field errors list them).
const EFFICIENCY_FIELDS: &[&str] = &[
    "failure_learning",
    "ccr",
    "typed_handoff",
    "semantic_context",
    "information_gain",
    "rework_routing",
];

/// Map-only strict parsing for `[efficiency]`: unlike a derived struct with
/// all-default fields, a JSON sequence is REFUSED (serde would otherwise
/// accept `[true]` as positional field values), duplicates are refused, and
/// unknown keys are refused. Absent keys keep the `false` default.
impl<'de> serde::Deserialize<'de> for EfficiencyCfg {
    fn deserialize<D>(de: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{Error as _, MapAccess, Visitor};

        struct SectionVisitor;

        impl<'de> Visitor<'de> for SectionVisitor {
            type Value = EfficiencyCfg;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("the [efficiency] section as a JSON object of booleans")
            }

            fn visit_map<A>(self, mut map: A) -> Result<EfficiencyCfg, A::Error>
            where
                A: MapAccess<'de>,
            {
                // A PRESENT partial section starts from the production
                // defaults and flips only the keys it names: an operator who
                // disables one component never silently disables the rest.
                let mut out = EfficiencyCfg::production_defaults();
                let mut seen: u8 = 0;
                while let Some(key) = map.next_key::<String>()? {
                    let (bit, name) = match key.as_str() {
                        "failure_learning" => (1u8, "failure_learning"),
                        "ccr" => (2, "ccr"),
                        "typed_handoff" => (4, "typed_handoff"),
                        // The audit calls this switch `information_gain`;
                        // `semantic_context` remains the canonical key. Both
                        // names flip the SAME bit, so naming both is a
                        // duplicate (refused), never a silent last-wins.
                        "semantic_context" | "information_gain" => (8, "semantic_context"),
                        "rework_routing" => (16, "rework_routing"),
                        other => return Err(A::Error::unknown_field(other, EFFICIENCY_FIELDS)),
                    };
                    if seen & bit != 0 {
                        return Err(A::Error::duplicate_field(name));
                    }
                    seen |= bit;
                    let value = map.next_value::<bool>()?;
                    match bit {
                        1 => out.failure_learning = value,
                        2 => out.ccr = value,
                        4 => out.typed_handoff = value,
                        8 => out.semantic_context = value,
                        _ => out.rework_routing = value,
                    }
                }
                Ok(out)
            }
        }

        de.deserialize_map(SectionVisitor)
    }
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
            #[serde(default)]
            tasks: TasksCfg,
            #[serde(default)]
            completion: CompletionCfg,
            #[serde(default = "production_efficiency")]
            efficiency: EfficiencyCfg,
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
            tasks: file.tasks,
            completion: file.completion,
            efficiency: file.efficiency,
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
            tasks: TasksCfg::default(),
            completion: CompletionCfg::default(),
            efficiency: EfficiencyCfg::production_defaults(),
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
        #[serde(default)]
        pricing: Option<ProviderPricingCfg>,
    },
    OpenAi {
        id: String,
        base_url: String,
        api_key_env: Option<String>,
        #[serde(default)]
        pricing: Option<ProviderPricingCfg>,
    },
    Anthropic {
        id: String,
        api_key_env: Option<String>,
        #[serde(default)]
        pricing: Option<ProviderPricingCfg>,
    },
    Google {
        id: String,
        api_key_env: Option<String>,
        #[serde(default)]
        pricing: Option<ProviderPricingCfg>,
    },
    DeepSeek {
        id: String,
        profile: String,
        base_url: Option<String>,
        api_key_env: Option<String>,
        #[serde(default)]
        pricing: Option<ProviderPricingCfg>,
    },
    Gateway {
        id: String,
        base_url: String,
        api_key_env: Option<String>,
        #[serde(default)]
        pricing: Option<ProviderPricingCfg>,
    },
}

/// The additive per-provider `pricing` override section (audit P0-1 /
/// wave-B item A): prices are microUSD PER MILLION TOKENS — the exact unit
/// providers publish — so sub-$1/M list prices ($0.50/M = 500_000 microUSD
/// per million tokens) are representable and can never truncate to a free
/// lie. Example: `{"pricing": {"input_micro_usd_per_million_tokens":
/// 2_500_000, "output_micro_usd_per_million_tokens": 10_000_000}}` ($2.50/M
/// in, $10/M out) inside ONE `providers[]` entry, scoped to that entry's
/// `id`. This is the money surface that makes REAL production economics
/// reach the routing graph — without it, remote (OpenAI-compatible/gateway/
/// deepseek/anthropic/google) models are catalog-priced
/// [`PricingState::Unknown`] (unless the built-in table documents them) and
/// the Economy candidate set EXCLUDES the Unknown ones (no fake zero
/// prices, no 1-microUSD fallback).
///
/// Two independent knobs:
///
/// - exact prices: `input_micro_usd_per_million_tokens` + `output_micro_usd_per_million_tokens`
///   (both REQUIRED together, each >= 1 microUSD per million tokens), plus optional
///   `cache_read_*`/`cache_write_*` (0 = the endpoint publishes no cache price). They price
///   EVERY model the endpoint serves at the declared per-million microUSD values
///   (state `Known`, provenance `UserOverride`). Intended for custom OpenAI-compatible
///   endpoints whose real prices the operator knows; overrides NEVER apply to a local
///   runtime (Ollama rows are `LocalZero` and stay zero).
/// - `pricing_ceiling_micro_usd_per_million_tokens`: a CONSERVATIVE budget bound that prices
///   ONLY models the adapter itself leaves Unknown, at the ceiling on all four price lines
///   (state `ConservativeCeiling`, provenance `Composite` — a ceiling, never a measured
///   price). Known-priced and LocalZero models are untouched.
///
/// Both knobs bump the catalog row's `source_epoch` so settlement can tell
/// the price generation changed. Unknown keys inside `pricing` are parse
/// errors; hostile values (0 input, absurd magnitudes, a table on a local
/// provider) are typed validation errors.
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields, default)]
pub struct ProviderPricingCfg {
    /// Exact override, microUSD per MILLION tokens.
    pub input_micro_usd_per_million_tokens: Option<u64>,
    pub output_micro_usd_per_million_tokens: Option<u64>,
    pub cache_read_micro_usd_per_million_tokens: Option<u64>,
    pub cache_write_micro_usd_per_million_tokens: Option<u64>,
    /// Conservative ceiling, microUSD per million tokens, applied to
    /// Unknown-priced models only.
    pub pricing_ceiling_micro_usd_per_million_tokens: Option<u64>,
}

/// Ceiling and exact-price magnitude cap (microUSD per million tokens).
/// 1_000_000_000_000 microUSD/million tokens = $1M per million tokens —
/// beyond any production model; anything larger is a hostile/absurd config
/// value.
pub const MAX_PRICING_MICRO_USD_PER_MILLION_TOKENS: u64 = 1_000_000_000_000;

impl ProviderPricingCfg {
    /// True when the section configures anything at all.
    pub fn is_empty(&self) -> bool {
        self == &ProviderPricingCfg::default()
    }

    /// Typed validation of the override surface. Rules:
    ///
    /// - a pricing section under a LOCAL provider (kind `ollama`) is
    ///   refused (Ollama rows are measured LocalZero; a config table must
    ///   never make a local runtime look paid, nor is it honored);
    /// - exact input/output prices are REQUIRED as a pair and each must
    ///   be >= 1 microUSD — `0` is the local-free marker, and a remote
    ///   endpoint must never silently read as free — and <= the magnitude
    ///   cap;
    /// - cache prices are optional, 0 allowed (= no cache price), capped;
    /// - the ceiling must be >= 1 microUSD and <= the cap (a zero ceiling
    ///   would price unknown models as free — the exact lie this audit
    ///   removes).
    pub fn validate(&self, kind: &str) -> Result<(), String> {
        if self.is_empty() {
            return Ok(());
        }
        if kind == "ollama" {
            return Err(
                "pricing overrides on a local (ollama) provider are refused: \
                 ollama rows are measured local-zero cost and a pricing table would only \
                 fabricate a paid profile (id-scoped override tables apply to custom \
                 OpenAI-compatible REMOTE endpoints only)"
                    .to_string(),
            );
        }
        let exact_present = self.input_micro_usd_per_million_tokens.is_some()
            || self.output_micro_usd_per_million_tokens.is_some()
            || self.cache_read_micro_usd_per_million_tokens.is_some()
            || self.cache_write_micro_usd_per_million_tokens.is_some();
        if exact_present {
            let (Some(input), Some(output)) = (
                self.input_micro_usd_per_million_tokens,
                self.output_micro_usd_per_million_tokens,
            ) else {
                return Err(
                    "pricing override table must set BOTH input_micro_usd_per_million_tokens \
                     and output_micro_usd_per_million_tokens (a partial table would silently \
                     price the missing side at 0 microUSD)"
                        .to_string(),
                );
            };
            if input == 0 || output == 0 {
                return Err(
                    "pricing override input/output prices of 0 are refused: 0 microUSD is the \
                     LOCAL-free marker and a remote endpoint must never silently read as free"
                        .to_string(),
                );
            }
            for (name, v) in [
                ("input_micro_usd_per_million_tokens", input),
                ("output_micro_usd_per_million_tokens", output),
                (
                    "cache_read_micro_usd_per_million_tokens",
                    self.cache_read_micro_usd_per_million_tokens.unwrap_or(0),
                ),
                (
                    "cache_write_micro_usd_per_million_tokens",
                    self.cache_write_micro_usd_per_million_tokens.unwrap_or(0),
                ),
            ] {
                if v > MAX_PRICING_MICRO_USD_PER_MILLION_TOKENS {
                    return Err(format!(
                        "{name} = {v} exceeds the magnitude cap of \
                         {MAX_PRICING_MICRO_USD_PER_MILLION_TOKENS} microUSD per million tokens"
                    ));
                }
            }
        }
        if let Some(c) = self.pricing_ceiling_micro_usd_per_million_tokens {
            if c == 0 {
                return Err(
                    "pricing_ceiling_micro_usd_per_million_tokens of 0 is refused: a zero \
                     ceiling would price unknown models as free"
                        .to_string(),
                );
            }
            if c > MAX_PRICING_MICRO_USD_PER_MILLION_TOKENS {
                return Err(format!(
                    "pricing_ceiling_micro_usd_per_million_tokens = {c} exceeds the magnitude \
                     cap of {MAX_PRICING_MICRO_USD_PER_MILLION_TOKENS} microUSD per million \
                     tokens"
                ));
            }
        }
        Ok(())
    }

    /// Map the parsed config onto the provider crate's override policy
    /// (per-million-token quotes; exact = a real [`PriceQuote`], ceiling =
    /// the conservative per-million bound).
    pub(crate) fn to_overrides(&self) -> PricingOverrides {
        let exact = match (
            self.input_micro_usd_per_million_tokens,
            self.output_micro_usd_per_million_tokens,
        ) {
            (Some(input), Some(output)) => Some(PriceQuote {
                input: MicroUsdPerMillionTokens(input),
                output: MicroUsdPerMillionTokens(output),
                cache_read: MicroUsdPerMillionTokens(
                    self.cache_read_micro_usd_per_million_tokens.unwrap_or(0),
                ),
                cache_write: MicroUsdPerMillionTokens(
                    self.cache_write_micro_usd_per_million_tokens.unwrap_or(0),
                ),
            }),
            _ => None,
        };
        PricingOverrides {
            exact,
            ceiling_micro_usd_per_million_tokens: self
                .pricing_ceiling_micro_usd_per_million_tokens
                .map(MicroUsdPerMillionTokens),
        }
    }
}

/// The canonical official OpenAI API base URL (the ONLY OpenAI `base_url`
/// that resolves to [`BillingOrigin::OfficialOpenAi`]).
pub const OPENAI_OFFICIAL_BASE_URL: &str = "https://api.openai.com/v1";

/// The canonical official DeepSeek API base URL (the ONLY DeepSeek
/// `base_url` — besides the absent default — that resolves to
/// [`BillingOrigin::OfficialDeepSeek`]).
pub const DEEPSEEK_OFFICIAL_BASE_URL: &str = "https://api.deepseek.com";

/// Strict endpoint identity: exact string apart from surrounding
/// whitespace and a trailing slash.
fn same_endpoint(configured: &str, canonical: &str) -> bool {
    configured.trim().trim_end_matches('/') == canonical.trim_end_matches('/')
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

    /// The transport family of this entry (used to gate the pricing
    /// override surface: local runtimes refuse override tables).
    fn kind(&self) -> &'static str {
        match self {
            ProviderCfg::Ollama { .. } => "ollama",
            ProviderCfg::OpenAi { .. } => "open_ai",
            ProviderCfg::Anthropic { .. } => "anthropic",
            ProviderCfg::Google { .. } => "google",
            ProviderCfg::DeepSeek { .. } => "deepseek",
            ProviderCfg::Gateway { .. } => "gateway",
        }
    }

    /// The configured endpoint's STRICT billing origin (billing-origin
    /// audit): official canonical endpoints resolve to their official
    /// origin, any other `base_url` is a [`BillingOrigin::CustomEndpoint`]
    /// (whatever transport family it speaks), `kind = gateway` and the
    /// deepseek gateway/openrouter profiles are [`BillingOrigin::Gateway`],
    /// and Ollama is [`BillingOrigin::Local`]. The origin is decided by the
    /// ENDPOINT CONFIG ONLY — the instance id (and the adapter's transport
    /// family) never changes it.
    pub fn billing_origin(&self) -> BillingOrigin {
        match self {
            ProviderCfg::Ollama { .. } => BillingOrigin::Local,
            ProviderCfg::OpenAi { base_url, .. } => {
                if same_endpoint(base_url, OPENAI_OFFICIAL_BASE_URL)
                    || same_endpoint(base_url, "https://api.openai.com")
                {
                    BillingOrigin::OfficialOpenAi
                } else {
                    BillingOrigin::CustomEndpoint
                }
            }
            ProviderCfg::Anthropic { .. } => BillingOrigin::OfficialAnthropic,
            ProviderCfg::Google { .. } => BillingOrigin::OfficialGoogle,
            ProviderCfg::DeepSeek {
                profile, base_url, ..
            } => match profile.as_str() {
                "direct" => match base_url.as_deref() {
                    None => BillingOrigin::OfficialDeepSeek,
                    Some(b)
                        if same_endpoint(b, DEEPSEEK_OFFICIAL_BASE_URL)
                            || same_endpoint(b, "https://api.deepseek.com/v1") =>
                    {
                        BillingOrigin::OfficialDeepSeek
                    }
                    Some(_) => BillingOrigin::CustomEndpoint,
                },
                // Both gateway-shaped profiles bill through an aggregator:
                // never the official DeepSeek list price.
                "gateway" | "openrouter" => BillingOrigin::Gateway,
                _ => BillingOrigin::CustomEndpoint,
            },
            ProviderCfg::Gateway { .. } => BillingOrigin::Gateway,
        }
    }

    /// The pricing override section of this entry, when configured.
    pub fn pricing(&self) -> Option<&ProviderPricingCfg> {
        match self {
            ProviderCfg::Ollama { pricing, .. }
            | ProviderCfg::OpenAi { pricing, .. }
            | ProviderCfg::Anthropic { pricing, .. }
            | ProviderCfg::Google { pricing, .. }
            | ProviderCfg::DeepSeek { pricing, .. }
            | ProviderCfg::Gateway { pricing, .. } => pricing.as_ref(),
        }
    }

    /// Validate this entry's pricing override surface (typed errors for
    /// hostile values: 0 input, absurd magnitudes, tables on local
    /// runtimes). Called by the strict config validation and by the
    /// adapter build (the runtime gate — a provider whose pricing config
    /// cannot be honored is never registered).
    pub fn validate_pricing(&self) -> Result<(), String> {
        match self.pricing() {
            Some(p) => p.validate(self.kind()),
            None => Ok(()),
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
        // Billing-origin audit: EVERY configured endpoint is wrapped with
        // its STRICTLY-resolved billing origin (official canonical endpoint
        // vs custom base_url vs gateway vs local), so a custom
        // OpenAI-compatible endpoint can never inherit official list prices
        // through its transport family id. A configured `pricing` section
        // then applies (exact prices -> UserOverride rows, ceiling ->
        // Composite rows for Unknown-priced models only; both bump the
        // pricing epoch). Hostile values are refused HERE so a provider
        // whose pricing cannot be honored never registers — and the
        // local-runtime (ollama) gate also holds on the raw `build` path
        // (the daemon's warm-up path builds ollama separately, where
        // `Config::validate`/`load_strict` refuse such a config loudly).
        let overrides = match self.pricing() {
            Some(pricing) => {
                pricing.validate(self.kind())?;
                pricing.to_overrides()
            }
            None => PricingOverrides::default(),
        };
        Ok(BillingOriginProvider::wrap(
            provider,
            instance,
            self.billing_origin(),
            overrides,
        ))
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
    /// hostile-config bounds, the sandbox section's destination rows must
    /// all parse (a rule that cannot parse is a config error, never
    /// silently permissive), and every provider's `pricing` override
    /// section must validate (typed errors: 0 prices, absurd magnitudes,
    /// override tables on local runtimes).
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
        for p in &self.providers {
            p.validate_pricing()
                .map_err(|e| format!("provider {}: {e}", p.id()))?;
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
    fn tasks_section_defaults_shadow_parses_modes_and_rejects_hostile() {
        // Wave-24: absent [tasks] keeps the PRODUCT default — shadow
        // mutation ON (`MutationMode::Shadow`); `direct_compat` opts the
        // daemon back into the byte-identical direct behavior; the legacy
        // boolean key still parses with its historical meaning on both load
        // paths; specifying BOTH keys (or unknown keys / bad values) is a
        // parse error.
        let cfg = Config::default();
        assert_eq!(
            cfg.tasks.mutation_mode,
            MutationMode::Shadow,
            "shadow mutation is the production default"
        );
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.json");
        // The default round-trips through the daemon's own file shape.
        cfg.save(&path).unwrap();
        let loaded = Config::load(&path).unwrap();
        assert_eq!(loaded.tasks, cfg.tasks);
        for (body, expected) in [
            (
                r#"{"tasks": {"mutation_mode": "shadow"}}"#,
                MutationMode::Shadow,
            ),
            (
                r#"{"tasks": {"mutation_mode": "direct_compat"}}"#,
                MutationMode::DirectCompat,
            ),
            // Legacy alias: the pre-wave-24 boolean key keeps its meaning.
            (
                r#"{"tasks": {"shadow_mutation": true}}"#,
                MutationMode::Shadow,
            ),
            (
                r#"{"tasks": {"shadow_mutation": false}}"#,
                MutationMode::DirectCompat,
            ),
        ] {
            std::fs::write(&path, body).unwrap();
            let cfg = Config::load(&path).unwrap();
            assert_eq!(cfg.tasks.mutation_mode, expected, "{body}");
            let strict = Config::load_strict(&path).unwrap();
            assert_eq!(strict.tasks.mutation_mode, expected, "{body}");
        }
        // Partial objects keep the per-key default (Shadow).
        std::fs::write(&path, r#"{"tasks": {}}"#).unwrap();
        assert_eq!(
            Config::load(&path).unwrap().tasks.mutation_mode,
            MutationMode::Shadow
        );
        for bad in [
            // A file never says two different things at once.
            r#"{"tasks": {"mutation_mode": "shadow", "shadow_mutation": true}}"#,
            r#"{"tasks": {"mutation_mode": "direct_compat", "shadow_mutation": false}}"#,
            r#"{"tasks": {"shadow_mutation": true, "bogus": 1}}"#,
            r#"{"tasks": {"shadow_mutation": "yes"}}"#,
            r#"{"tasks": {"mutation_mode": "nonsense"}}"#,
            r#"{"tasks": {"mutation_mode": "Shadow"}}"#,
            r#"{"task": {"mutation_mode": "shadow"}}"#,
        ] {
            std::fs::write(&path, bad).unwrap();
            let e = Config::load(&path).expect_err("hostile [tasks] must fail");
            assert!(
                e.contains("unknown field")
                    || e.contains("invalid type")
                    || e.contains("unknown variant")
                    || e.contains("cannot both be present"),
                "{e}"
            );
            assert!(Config::load_strict(&path).is_err());
        }
    }

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
                    pricing: None,
                },
                ProviderCfg::Ollama {
                    id: "other".into(),
                    base_url: None,
                    pricing: None,
                },
                ProviderCfg::OpenAi {
                    id: "dup".into(),
                    base_url: "http://x".into(),
                    api_key_env: None,
                    pricing: None,
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
                    pricing: None,
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
    fn pricing_section_parses_roundtrips_and_applies_to_the_custom_endpoint_only() {
        // The `pricing` section rides the provider entry it names: an
        // exact table prices EVERY model of THAT endpoint (UserOverride,
        // epoch bumped); a second endpoint without a section keeps its
        // Unknown adapter rows — overrides never leak across ids.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("p.json");
        std::fs::write(
            &path,
            r#"{"config_version": 1, "model": "m", "providers": [
                {"kind": "open_ai", "id": "corp-proxy", "base_url": "https://corp.example.com/v1",
                 "pricing": {"input_micro_usd_per_million_tokens": 2000000,
                             "output_micro_usd_per_million_tokens": 8000000}},
                {"kind": "open_ai", "id": "dev-proxy", "base_url": "https://dev.example.com/v1"}
            ]}"#,
        )
        .unwrap();
        let cfg = Config::load(&path).unwrap();
        assert_eq!(cfg.providers.len(), 2);
        let p = &cfg.providers[0];
        assert_eq!(p.id(), "corp-proxy");
        let pricing = p.pricing().expect("pricing section parsed");
        assert_eq!(pricing.input_micro_usd_per_million_tokens, Some(2_000_000));
        assert_eq!(pricing.output_micro_usd_per_million_tokens, Some(8_000_000));
        assert_eq!(pricing.cache_read_micro_usd_per_million_tokens, None);
        assert_eq!(pricing.pricing_ceiling_micro_usd_per_million_tokens, None);
        // Strict load accepts the healthy config (semantic validation too).
        let strict = Config::load_strict(&path).unwrap();
        assert_eq!(strict.providers[0].pricing(), p.pricing());
        // The config file round-trips through save/load.
        cfg.save(&path).unwrap();
        assert_eq!(
            Config::load(&path).unwrap().providers[0].pricing(),
            p.pricing()
        );
        // Apply: the configured endpoint's rows become Known/UserOverride
        // with the pricing epoch bumped; the other endpoint stays Unknown.
        let mut registry = faktor_provider::ProviderRegistry::new();
        for provider in &cfg.providers {
            registry
                .try_register(provider.build(open_transport()).unwrap())
                .unwrap();
        }
        let corp = registry.get("corp-proxy").unwrap().catalog_entry("default");
        match &corp.pricing {
            faktor_provider::catalog::PricingState::Known(snap) => {
                assert_eq!(snap.authority, faktor_core::model::PriceAuthority::Exact);
                let q = snap.quote.expect("exact override quotes");
                assert_eq!(
                    q.input,
                    faktor_core::model::MicroUsdPerMillionTokens(2_000_000)
                );
                assert_eq!(
                    q.output,
                    faktor_core::model::MicroUsdPerMillionTokens(8_000_000)
                );
                assert_eq!(
                    q.cache_read,
                    faktor_core::model::MicroUsdPerMillionTokens(0)
                );
                assert_eq!(
                    q.cache_write,
                    faktor_core::model::MicroUsdPerMillionTokens(0)
                );
                // The exact quote never truncates and never reads free.
                assert_eq!(snap.settle_cost(1_000_000, 0, 0, 0), Some(2_000_000));
            }
            other => panic!("override must price the row Known, got {other:?}"),
        }
        assert_eq!(
            corp.provenance,
            faktor_provider::catalog::Provenance::UserOverride
        );
        assert_eq!(
            corp.source_epoch,
            faktor_provider::catalog::CATALOG_FIRST_EPOCH + 1,
            "the override increments the pricing epoch"
        );
        let dev = registry.get("dev-proxy").unwrap().catalog_entry("default");
        assert_eq!(
            dev.pricing,
            faktor_provider::catalog::PricingState::Unknown,
            "an endpoint without a pricing section keeps its Unknown adapter rows"
        );
    }

    #[test]
    fn pricing_ceiling_parses_and_composites_unknown_rows_only() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.json");
        std::fs::write(
            &path,
            r#"{"providers": [
                {"kind": "open_ai", "id": "gw", "base_url": "https://gw.example.com/v1",
                 "pricing": {"pricing_ceiling_micro_usd_per_million_tokens": 42000000}}
            ]}"#,
        )
        .unwrap();
        let cfg = Config::load_strict(&path).unwrap();
        let provider = cfg.providers[0].build(open_transport()).unwrap();
        let entry = provider.catalog_entry("whatever-model");
        assert_eq!(
            entry.provenance,
            faktor_provider::catalog::Provenance::Composite
        );
        assert_eq!(entry.source_epoch, 2);
        match &entry.pricing {
            faktor_provider::catalog::PricingState::ConservativeCeiling(snap) => {
                assert_eq!(
                    snap.authority,
                    faktor_core::model::PriceAuthority::ConservativeCeiling
                );
                let q = snap.quote.expect("ceiling quotes");
                for line in [q.input, q.output, q.cache_read, q.cache_write] {
                    assert_eq!(
                        line,
                        faktor_core::model::MicroUsdPerMillionTokens(42_000_000)
                    );
                }
            }
            other => panic!("ceiling must produce ConservativeCeiling, got {other:?}"),
        }
        // Known rows of the SAME endpoint keep their price under a ceiling
        // (covered by the graph test) — here only the epoch bump is
        // asserted for the Unknown row above.
        let _ = provider.known_models();
    }

    #[test]
    fn hostile_pricing_override_values_are_typed_errors_everywhere() {
        // 0 input, absurd magnitudes, partial tables, zero/absurd ceilings,
        // tables on local runtimes, and unknown keys inside `pricing` are
        // all refused — on the parse/validate path AND at adapter build.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("h.json");
        let cases: Vec<(&str, &str)> = vec![
            // Zero input price = the local-free marker on a remote endpoint.
            (
                "zero input",
                r#""pricing": {"input_micro_usd_per_million_tokens": 0,
                               "output_micro_usd_per_million_tokens": 8000000}"#,
            ),
            // Partial tables would silently price the missing side at 0.
            (
                "partial table",
                r#""pricing": {"input_micro_usd_per_million_tokens": 2000000}"#,
            ),
            // Absurd magnitudes beyond the cap.
            (
                "absurd price",
                r#""pricing": {"input_micro_usd_per_million_tokens": 1000000000001,
                               "output_micro_usd_per_million_tokens": 8000000}"#,
            ),
            (
                "absurd ceiling",
                r#""pricing": {"pricing_ceiling_micro_usd_per_million_tokens":
                               18446744073709551615}"#,
            ),
            // A zero ceiling prices unknown models as free — refused.
            (
                "zero ceiling",
                r#""pricing": {"pricing_ceiling_micro_usd_per_million_tokens": 0}"#,
            ),
        ];
        for (label, pricing_json) in cases {
            let text = format!(
                r#"{{"providers": [{{"kind": "open_ai", "id": "p", "base_url": "http://x", {pricing_json}}}]}}"#
            );
            std::fs::write(&path, &text).unwrap();
            // Parse succeeds (lenient); semantic validation refuses.
            let cfg = Config::load(&path).unwrap_or_else(|e| panic!("{label}: parse: {e}"));
            let e = cfg
                .validate()
                .expect_err(&format!("{label}: validate must refuse"));
            assert!(!e.is_empty(), "{label}");
            // Adapter build refuses too (the runtime gate).
            let err = match cfg.providers[0].build(open_transport()) {
                Ok(_) => panic!("{label}: build must refuse"),
                Err(e) => e,
            };
            assert!(!err.is_empty(), "{label}");
            // And the strict file load path refuses.
            std::fs::write(&path, &text).unwrap();
            let strict_err =
                Config::load_strict(&path).expect_err(&format!("{label}: strict load must refuse"));
            assert!(!strict_err.is_empty(), "{label}");
        }
        // A pricing table under a LOCAL (ollama) provider is refused:
        // overrides apply to custom REMOTE endpoints only.
        let ollama = ProviderCfg::Ollama {
            id: "ollama".into(),
            base_url: None,
            pricing: Some(ProviderPricingCfg {
                input_micro_usd_per_million_tokens: Some(15_000_000),
                output_micro_usd_per_million_tokens: Some(60_000_000),
                ..Default::default()
            }),
        };
        let e = ollama
            .validate_pricing()
            .expect_err("ollama pricing refused");
        assert!(e.contains("local"), "{e}");
        // Unknown keys inside the pricing section are parse errors.
        std::fs::write(
            &path,
            r#"{"providers": [{"kind": "open_ai", "id": "p", "base_url": "http://x",
                 "pricing": {"input_micro_usd_per_million_tokens": 2000000, "bogus": 1}}]}"#,
        )
        .unwrap();
        let e = Config::load(&path).expect_err("unknown pricing key must fail");
        assert!(e.contains("unknown field"), "{e}");
        // A hostile section on an unknown kind is refused at parse like any
        // unknown provider kind.
        std::fs::write(
            &path,
            r#"{"providers": [{"kind": "open_ai", "id": "p", "base_url": "http://x",
                 "pricing": "expensive"}]}"#,
        )
        .unwrap();
        assert!(Config::load(&path).is_err());
    }

    #[test]
    fn ceiling_applies_to_gateway_instances_too() {
        // The gateway family is a custom endpoint: its Unknown rows are
        // composite-priced by a ceiling exactly like open_ai endpoints.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.json");
        std::fs::write(
            &path,
            r#"{"providers": [
                {"kind": "gateway", "id": "kilo-gw", "base_url": "https://api.kilo.ai",
                 "pricing": {"pricing_ceiling_micro_usd_per_million_tokens": 60000000}}
            ]}"#,
        )
        .unwrap();
        let cfg = Config::load_strict(&path).unwrap();
        let provider = cfg.providers[0].build(open_transport()).unwrap();
        let entry = provider.catalog_entry("default");
        assert_eq!(
            entry.provenance,
            faktor_provider::catalog::Provenance::Composite
        );
    }

    #[test]
    fn keys_read_from_env_not_file() {
        std::env::set_var("KP_TEST_KEY", "secret-value");
        let cfg = ProviderCfg::OpenAi {
            id: "t".into(),
            base_url: "http://x".into(),
            api_key_env: Some("KP_TEST_KEY".into()),
            pricing: None,
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
            pricing: None,
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
                pricing: None,
            };
            registry
                .try_register(cfg.build(open_transport()).unwrap())
                .unwrap();
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
                pricing: None,
            };
            let provider = cfg
                .build(open_transport())
                .unwrap_or_else(|e| panic!("{profile:?} build: {e}"));
            registry.try_register(provider).unwrap();
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
            pricing: None,
        };
        assert!(cfg.build(open_transport()).is_err());
    }

    #[test]
    fn billing_origin_matrix_is_strict_and_instance_scoped() {
        use faktor_core::model::PriceAuthority;
        use faktor_provider::catalog::PricingState;

        // Origin resolution reads the ENDPOINT CONFIG only: the canonical
        // official URLs resolve official, any other base_url is a custom
        // endpoint (whatever transport family it speaks), gateway kinds are
        // gateways, ollama is local.
        let official_openai = ProviderCfg::OpenAi {
            id: "a".into(),
            base_url: OPENAI_OFFICIAL_BASE_URL.into(),
            api_key_env: None,
            pricing: None,
        };
        let official_openai_slash = ProviderCfg::OpenAi {
            id: "a2".into(),
            base_url: format!("{OPENAI_OFFICIAL_BASE_URL}/"),
            api_key_env: None,
            pricing: None,
        };
        let custom_openai = ProviderCfg::OpenAi {
            id: "corp-proxy".into(),
            base_url: "https://corp.example.com/v1".into(),
            api_key_env: None,
            pricing: None,
        };
        assert_eq!(
            official_openai.billing_origin(),
            BillingOrigin::OfficialOpenAi
        );
        assert_eq!(
            official_openai_slash.billing_origin(),
            BillingOrigin::OfficialOpenAi,
            "a trailing slash is the same canonical endpoint"
        );
        assert_eq!(
            custom_openai.billing_origin(),
            BillingOrigin::CustomEndpoint
        );
        assert_eq!(
            ProviderCfg::Anthropic {
                id: "anthropic".into(),
                api_key_env: None,
                pricing: None,
            }
            .billing_origin(),
            BillingOrigin::OfficialAnthropic
        );
        assert_eq!(
            ProviderCfg::Google {
                id: "google".into(),
                api_key_env: None,
                pricing: None,
            }
            .billing_origin(),
            BillingOrigin::OfficialGoogle
        );
        let deepseek = |profile: &str, base: Option<&str>| ProviderCfg::DeepSeek {
            id: format!("ds-{profile}"),
            profile: profile.into(),
            base_url: base.map(str::to_string),
            api_key_env: None,
            pricing: None,
        };
        assert_eq!(
            deepseek("direct", None).billing_origin(),
            BillingOrigin::OfficialDeepSeek
        );
        assert_eq!(
            deepseek("direct", Some(DEEPSEEK_OFFICIAL_BASE_URL)).billing_origin(),
            BillingOrigin::OfficialDeepSeek
        );
        assert_eq!(
            deepseek("direct", Some("https://corp.example.com/v1")).billing_origin(),
            BillingOrigin::CustomEndpoint
        );
        assert_eq!(
            deepseek("gateway", None).billing_origin(),
            BillingOrigin::Gateway
        );
        assert_eq!(
            deepseek("openrouter", None).billing_origin(),
            BillingOrigin::Gateway
        );
        assert_eq!(
            ProviderCfg::Gateway {
                id: "gw".into(),
                base_url: "https://api.kilo.ai".into(),
                api_key_env: None,
                pricing: None,
            }
            .billing_origin(),
            BillingOrigin::Gateway
        );
        assert_eq!(
            ProviderCfg::Ollama {
                id: "ollama".into(),
                base_url: None,
                pricing: None,
            }
            .billing_origin(),
            BillingOrigin::Local
        );
        // The instance id NEVER changes the origin: two entries differing
        // only in id resolve identically.
        let same_custom_other_id = ProviderCfg::OpenAi {
            id: "b".into(),
            base_url: "https://corp.example.com/v1".into(),
            api_key_env: None,
            pricing: None,
        };
        assert_ne!(custom_openai.id(), same_custom_other_id.id());
        assert_eq!(
            custom_openai.billing_origin(),
            same_custom_other_id.billing_origin()
        );

        // Built rows follow the origin, not the wire family: official
        // OpenAI gpt-4o is Exact at $2.50/M; the custom OpenAI-compatible
        // endpoint is Unknown; DeepSeek mirrors both.
        let official = official_openai.build(open_transport()).unwrap();
        let e = official.catalog_entry("gpt-4o");
        assert_eq!(e.pricing.authority(), PriceAuthority::Exact);
        assert_eq!(
            e.pricing.quote().unwrap().input,
            MicroUsdPerMillionTokens(2_500_000)
        );
        let official_other_id = ProviderCfg::OpenAi {
            id: "b".into(),
            base_url: OPENAI_OFFICIAL_BASE_URL.into(),
            api_key_env: None,
            pricing: None,
        }
        .build(open_transport())
        .unwrap();
        let e2 = official_other_id.catalog_entry("gpt-4o");
        assert_ne!(e.provider, e2.provider, "instance ids differ");
        assert_eq!(
            e.pricing, e2.pricing,
            "the instance id must not change the resolved price"
        );
        let custom = custom_openai.build(open_transport()).unwrap();
        let e = custom.catalog_entry("gpt-4o");
        assert_eq!(e.provider, "corp-proxy");
        assert_eq!(e.pricing, PricingState::Unknown);
        assert_eq!(e.pricing_snapshot().settle_cost(1_000_000, 0, 0, 0), None);
        // DeepSeek V4-era facts are not documented as exact: the official
        // endpoint resolves a CONSERVATIVE CEILING (never read as a list
        // price), and changing the origin never lets a model inherit
        // another origin's catalog.
        let official_ds = deepseek("direct", None).build(open_transport()).unwrap();
        let e = official_ds.catalog_entry("deepseek-chat");
        assert_eq!(
            e.pricing.authority(),
            PriceAuthority::ConservativeCeiling,
            "the V4-era row is a bound, not an exact claim"
        );
        assert_eq!(
            e.pricing.quote().unwrap().input,
            MicroUsdPerMillionTokens(560_000)
        );
        assert_eq!(
            official_ds.catalog_entry("gpt-4o").pricing,
            PricingState::Unknown,
            "a DeepSeek endpoint never inherits the OpenAI catalog"
        );
        assert_eq!(
            official.catalog_entry("deepseek-chat").pricing,
            PricingState::Unknown,
            "and an OpenAI endpoint never inherits DeepSeek's row"
        );
        // Retired Anthropic IDs never resolve to their last-known price.
        let official_anthropic = ProviderCfg::Anthropic {
            id: "anthropic".into(),
            api_key_env: None,
            pricing: None,
        }
        .build(open_transport())
        .unwrap();
        assert_eq!(
            official_anthropic.catalog_entry("claude-haiku-3.5").pricing,
            PricingState::Unknown
        );
        let custom_ds = deepseek("direct", Some("https://corp.example.com/v1"))
            .build(open_transport())
            .unwrap();
        assert_eq!(
            custom_ds.catalog_entry("deepseek-chat").pricing,
            PricingState::Unknown
        );
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

    #[test]
    fn efficiency_section_defaults_on_parses_strictly_and_roundtrips() {
        // Absent section: every flag ON (the production efficiency system).
        // `EfficiencyCfg::default()` stays the additive all-off semantics for
        // unit/embedded callers; the Config paths use production_defaults().
        let cfg = Config::default();
        assert_eq!(cfg.efficiency, EfficiencyCfg::production_defaults());
        assert!(
            cfg.efficiency.failure_learning
                && cfg.efficiency.ccr
                && cfg.efficiency.typed_handoff
                && cfg.efficiency.semantic_context
                && cfg.efficiency.rework_routing,
            "every [efficiency] flag defaults ON in production"
        );
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("e.json");
        // The documented off-switches: an explicit `false` per component.
        std::fs::write(
            &path,
            r#"{"efficiency": {"failure_learning": false, "ccr": false, "typed_handoff": false,
                 "semantic_context": false, "rework_routing": false}}"#,
        )
        .unwrap();
        let all_off = Config::load_strict(&path).unwrap();
        assert_eq!(
            all_off.efficiency,
            EfficiencyCfg {
                failure_learning: false,
                ccr: false,
                typed_handoff: false,
                semantic_context: false,
                rework_routing: false,
            }
        );
        // Partial objects flip only the named key; the others keep ON.
        std::fs::write(&path, r#"{"efficiency": {"ccr": false}}"#).unwrap();
        let partial = Config::load(&path).unwrap();
        assert!(!partial.efficiency.ccr);
        assert!(partial.efficiency.failure_learning);
        assert!(partial.efficiency.typed_handoff);
        assert!(partial.efficiency.semantic_context);
        assert!(partial.efficiency.rework_routing);
        // Round-trip through the daemon's own file shape.
        all_off.save(&path).unwrap();
        assert_eq!(Config::load(&path).unwrap().efficiency, all_off.efficiency);
        // Hostile shapes: unknown keys, non-boolean values, duplicate keys,
        // and non-object containers (a positional array must never enable
        // flags) all fail on both load paths.
        for bad in [
            r#"{"efficiency": {"ccr": true, "bogus": 1}}"#,
            r#"{"efficiency": {"ccr": "yes"}}"#,
            r#"{"efficiency": {"ccr": 1}}"#,
            r#"{"efficiency": {"failure_learning": null}}"#,
            r#"{"efficiency": {"ccr": true, "ccr": false}}"#,
            r#"{"efficiency": []}"#,
            r#"{"efficiency": [true, true, true, true, true]}"#,
            r#"{"efficiency": true}"#,
            r#"{"efficency": {"ccr": true}}"#,
        ] {
            std::fs::write(&path, bad).unwrap();
            let e = Config::load(&path).expect_err("hostile [efficiency] must fail");
            assert!(
                e.contains("unknown field")
                    || e.contains("invalid type")
                    || e.contains("duplicate field"),
                "{bad}: {e}"
            );
            assert!(Config::load_strict(&path).is_err(), "{bad}");
        }
    }

    /// The reachable half of the `failure_learning` flag: the parsed flag is
    /// exactly what decides whether a failure prior is handed to the context
    /// planner, and the planner honors it. The production wiring now exists:
    /// `crates/cli/src/main.rs` `daemon_context_prior` builds the real
    /// `LearningService`-backed adapter only when the flag is on,
    /// `efficiency_flags` mirrors the whole section onto
    /// `faktor_agent::EfficiencyFlags`, and the runtime consumes the gate at
    /// the `plan_wire_turn_with_prior` call site.
    #[test]
    fn failure_learning_flag_gates_the_planner_prior_hook() {
        use faktor_context::planner::{plan_context, plan_context_with_prior, ContextPlanRequest};
        use faktor_context::{CandidateKind, ContextCandidate, FailurePrior};

        struct BoostA;
        impl FailurePrior for BoostA {
            fn omission_risk(&self, candidate: &ContextCandidate) -> f64 {
                if candidate.id == "a" {
                    2.0
                } else {
                    1.0
                }
            }
        }
        let candidate = |id: &str, utility: f64| ContextCandidate {
            id: id.into(),
            kind: CandidateKind::FileNote,
            bytes: 10,
            estimate_tokens: 10,
            utility,
            ..ContextCandidate::default()
        };
        let request = || ContextPlanRequest {
            index_evidence: vec![candidate("a", 0.5), candidate("b", 0.6)],
            token_budget: 10,
            ..ContextPlanRequest::default()
        };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("e.json");
        std::fs::write(&path, r#"{"efficiency": {"failure_learning": false}}"#).unwrap();
        let off = Config::load(&path).unwrap();
        std::fs::write(&path, r#"{"efficiency": {"failure_learning": true}}"#).unwrap();
        let on = Config::load(&path).unwrap();
        assert!(!off.efficiency.failure_learning);
        assert!(on.efficiency.failure_learning);

        let prior = BoostA;
        let planned = |enabled: bool| {
            if enabled {
                plan_context_with_prior(request(), Some(&prior))
            } else {
                plan_context(request())
            }
        };
        let baseline = planned(off.efficiency.failure_learning);
        let boosted = planned(on.efficiency.failure_learning);
        assert!(
            baseline.selected.iter().any(|c| c.id == "b"),
            "flag off: the baseline selector keeps b"
        );
        assert!(
            boosted.selected.iter().any(|c| c.id == "a"),
            "flag on: the prior boosts a into the window"
        );
        assert_ne!(
            baseline, boosted,
            "the parsed flag must decide planner construction"
        );
    }
}

#[cfg(test)]
mod completion_cfg_tests {
    //! Adversarial strictness covers for the additive `[completion]` section
    //! (P2 step execution): absent => inert defaults, unknown keys / wrong
    //! shapes / invalid templates are parse or validation errors, and a
    //! configured template resolves to the exact orchestrator policy.
    use super::*;

    fn parse(json: serde_json::Value) -> Result<Config, serde_json::Error> {
        serde_json::from_value(json)
    }

    #[test]
    fn absent_completion_section_keeps_the_inert_defaults() {
        let cfg = parse(serde_json::json!({"model": "m"})).unwrap();
        assert_eq!(cfg.completion, CompletionCfg::default());
        let steps = cfg.completion.steps_config().unwrap();
        assert_eq!(steps.remote, "origin");
        assert_eq!(steps.base_branch, "main");
        assert_eq!(steps.pr_command, None);
    }

    #[test]
    fn configured_completion_section_resolves_to_the_validated_policy() {
        let cfg = parse(serde_json::json!({
            "model": "m",
            "completion": {
                "remote": "upstream",
                "base_branch": "develop",
                "pr_command": "gh pr create --head {branch} --base {base}"
            }
        }))
        .unwrap();
        let steps = cfg.completion.steps_config().unwrap();
        assert_eq!(steps.remote, "upstream");
        assert_eq!(steps.base_branch, "develop");
        assert_eq!(
            steps.pr_command.as_deref(),
            Some("gh pr create --head {branch} --base {base}")
        );
    }

    #[test]
    fn completion_section_is_strict_and_hostile_values_are_refused() {
        // Unknown key anywhere in the section.
        assert!(
            parse(serde_json::json!({"completion": {"pr_command": "x {branch}", "extra": 1}}))
                .is_err()
        );
        // Non-string member.
        assert!(parse(serde_json::json!({"completion": {"pr_command": 7}})).is_err());
        // Positional array shape is not a section.
        assert!(parse(serde_json::json!({"completion": ["origin"]})).is_err());
        // A shell-metachar / unclosed / unknown-placeholder / branch-less
        // template is refused at resolution time (never half-run).
        for (name, command) in [
            ("metachar", "gh pr create --head {branch}; rm -rf /"),
            ("unknown", "gh pr create --head {nope}"),
            ("branchless", "gh pr create --base main"),
            ("unclosed", "gh pr create --head {branch"),
        ] {
            let cfg = parse(serde_json::json!({"completion": {"pr_command": command}})).unwrap();
            let err = cfg.completion.steps_config().unwrap_err();
            assert!(err.starts_with("completion: "), "{name}: {err}");
        }
        // Hostile remote / base names are refused.
        let cfg = parse(serde_json::json!({"completion": {"remote": "a b"}})).unwrap();
        assert!(cfg.completion.steps_config().is_err());
        let cfg = parse(serde_json::json!({"completion": {"base_branch": "a..b"}})).unwrap();
        assert!(cfg.completion.steps_config().is_err());
    }

    #[test]
    fn typed_argv_completion_section_resolves_and_is_strict() {
        // The additive typed argv: spaces in the program path and in one
        // argument stay per-element values.
        let cfg = parse(serde_json::json!({
            "model": "m",
            "completion": {
                "pr_program": "/opt/My Tools/gh",
                "pr_args": ["pr", "create", "--head", "{branch}", "--title", "my PR title"]
            }
        }))
        .unwrap();
        let steps = cfg.completion.steps_config().unwrap();
        assert_eq!(steps.pr_program.as_deref(), Some("/opt/My Tools/gh"));
        assert_eq!(steps.pr_command, None);
        assert_eq!(
            steps.pr_args,
            vec![
                "pr",
                "create",
                "--head",
                "{branch}",
                "--title",
                "my PR title"
            ]
        );

        // Strict shapes inside the section.
        assert!(parse(serde_json::json!({"completion": {"pr_program": 7}})).is_err());
        assert!(parse(serde_json::json!({"completion": {"pr_args": "nope"}})).is_err());
        assert!(parse(serde_json::json!({"completion": {"pr_extra": 1}})).is_err());
        assert!(parse(serde_json::json!({"completion": {"pr_args": [1]}})).is_err());

        // pr_args without a program, and BOTH shapes (ambiguous), refuse.
        let args_only =
            parse(serde_json::json!({"completion": {"pr_args": ["{branch}"]}})).unwrap();
        assert!(args_only.completion.steps_config().is_err());
        let both = parse(serde_json::json!({
            "completion": {
                "pr_command": "gh pr create --head {branch}",
                "pr_program": "/bin/gh",
                "pr_args": ["--head", "{branch}"]
            }
        }))
        .unwrap();
        assert!(both
            .completion
            .steps_config()
            .unwrap_err()
            .starts_with("completion: "));

        // Typed refusals mirror the orchestrator validator exactly.
        for (name, program, args) in [
            ("branchless", "/bin/gh", vec!["--base", "main"]),
            ("unknown", "/bin/gh", vec!["--head", "{nope}"]),
            ("control", "/bin/gh", vec!["--head", "{branch}\u{7}"]),
        ] {
            let cfg = parse(serde_json::json!({
                "completion": {"pr_program": program, "pr_args": args}
            }))
            .unwrap();
            assert!(cfg.completion.steps_config().is_err(), "{name}");
        }
    }
}
