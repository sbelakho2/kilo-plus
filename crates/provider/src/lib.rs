//! faktor-provider — the common LLM provider interface hub.
//!
//! The agent depends on this trait; the transport families (ollama, openai,
//! anthropic, google, deepseek, gateway) implement it. Requests pass through:
//!
//! ```text
//! Generic Agent Request
//!         ↓
//! Capability Validation
//!         ↓
//! Provider Normalizer
//!         ↓
//! Wire Serializer   (inside each adapter)
//!         ↓
//! HTTP Transport    (inside each adapter)
//! ```
//!
//! Provider quirks stay inside adapters. There is **no `if provider == "…"`**
//! in the agent — behavior is decided by `ModelCapabilities`.

use std::collections::HashMap;
use std::pin::Pin;

use faktor_core::cancellation::CancellationToken;
use faktor_core::error::{Error, ErrorKind};
use faktor_core::id::{OpId, SessionId};
use faktor_core::model::{ModelCapabilities, PricingSnapshot, ReasoningMode};
use futures::Stream;
#[cfg(test)]
use futures::StreamExt;

use crate::catalog::{ModelCatalogEntry, PricingState, Provenance, QualityPrior};

pub mod catalog;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
    System,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ContentPart {
    pub kind: ContentKind,
    /// For tool_result parts: which tool call this answers.
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentKind {
    Text {
        text: String,
    },
    Reasoning {
        text: String,
    },
    Image {
        url: String,
    },
    ToolCall {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        content: String,
        is_error: bool,
    },
}

impl ContentPart {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            kind: ContentKind::Text { text: text.into() },
            tool_call_id: None,
        }
    }

    pub fn reasoning(text: impl Into<String>) -> Self {
        Self {
            kind: ContentKind::Reasoning { text: text.into() },
            tool_call_id: None,
        }
    }

    pub fn tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        input: serde_json::Value,
    ) -> Self {
        Self {
            kind: ContentKind::ToolCall {
                id: id.into(),
                name: name.into(),
                input,
            },
            tool_call_id: None,
        }
    }

    pub fn tool_result(
        content: impl Into<String>,
        is_error: bool,
        tool_call_id: impl Into<String>,
    ) -> Self {
        Self {
            kind: ContentKind::ToolResult {
                content: content.into(),
                is_error,
            },
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RequestMessage {
    pub role: Role,
    pub content: Vec<ContentPart>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// Metadata attached to every request. The wire serializer never sees these
/// fields — they exist for retry state, deadlines, and circuit breakers.
#[derive(Debug, Clone)]
pub struct RequestMeta {
    pub operation_id: OpId,
    pub session_id: SessionId,
    pub provider: String,
    pub attempt: u32,
    /// The operation deadline in ms remaining when this request was built
    /// (audit round 15). Adapters honor it as the stream's OVERALL bound:
    /// `overall_ms = min(deadline_ms, transport::PROVIDER_CEILING_MS)`.
    /// `0` means "no operation-level overall bound" (the transport's
    /// first-byte/idle defaults still apply).
    pub deadline_ms: u64,
    pub cancellation: CancellationToken,
}

/// A normalized agent-level request. Capability validation happens on this
/// type; normalization turns it into wire shapes inside adapters.
#[derive(Debug, Clone)]
pub struct GenericAgentRequest {
    pub model: String,
    /// Cacheable prefix (system instructions, tools, project rules, task state).
    pub system: String,
    pub messages: Vec<RequestMessage>,
    pub tools: Vec<ToolSpec>,
    pub max_output: Option<usize>,
    pub reasoning: Option<ReasoningMode>,
    pub stream: bool,
    pub meta: RequestMeta,
}

/// The surface a provider-reported cost rode in on.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReportedCostSource {
    /// The usage envelope of the provider's own response payload.
    ProviderUsage,
    /// A provider billing header on the response.
    ProviderBillingHeader,
    /// A provider reconciliation/billing surface (usage endpoint, invoice).
    ProviderReconciliation,
}

/// The currency of a [`ReportedCost`] amount. Only USD-compatible values
/// may override route-snapshot estimation — adapters always label what they
/// forward and the runtime keeps the refusal rule for anything else.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReportedCurrency {
    Usd,
    Other { code: String },
}

/// A provider-reported cost with its currency and provenance. `micro_usd`
/// is denominated in `currency` (named `micro_usd` because the value is a
/// micro-unit amount; the field's meaning is "micro units of `currency`" —
/// in practice every adapter that reports a cost today reports USD).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ReportedCost {
    pub micro_usd: u64,
    pub currency: ReportedCurrency,
    pub source: ReportedCostSource,
    pub request_id: Option<String>,
}

impl ReportedCost {
    /// An authoritative USD cost reported on the wire.
    pub fn usd(micro_usd: u64, source: ReportedCostSource) -> Self {
        Self {
            micro_usd,
            currency: ReportedCurrency::Usd,
            source,
            request_id: None,
        }
    }

    /// True only for USD-compatible amounts: only these are authoritative
    /// overrides of route-snapshot estimation.
    pub fn is_usd(&self) -> bool {
        self.currency == ReportedCurrency::Usd
    }
}

/// Why a canonical usage split was refused: the wire row is impossible
/// (a cache line larger than the input total it must be a subset of, or an
/// informational reasoning subset larger than the output total). Adapters
/// surface this as a typed [`ProviderErrorKind::Malformed`] stream error —
/// never a silent zero and never a panic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsageSplitError {
    CacheReadsExceedInput {
        total_input_tokens: u64,
        cache_read_tokens: u64,
    },
    CacheWritesExceedInput {
        total_input_tokens: u64,
        cache_write_tokens: u64,
    },
    ReasoningExceedsOutput {
        output_tokens: u64,
        reasoning_tokens: u64,
    },
}

impl std::fmt::Display for UsageSplitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UsageSplitError::CacheReadsExceedInput {
                total_input_tokens,
                cache_read_tokens,
            } => write!(
                f,
                "cache read tokens {cache_read_tokens} exceed the reported input total {total_input_tokens}"
            ),
            UsageSplitError::CacheWritesExceedInput {
                total_input_tokens,
                cache_write_tokens,
            } => write!(
                f,
                "cache write tokens {cache_write_tokens} exceed the reported input total {total_input_tokens}"
            ),
            UsageSplitError::ReasoningExceedsOutput {
                output_tokens,
                reasoning_tokens,
            } => write!(
                f,
                "reasoning tokens {reasoning_tokens} exceed the reported output total {output_tokens}"
            ),
        }
    }
}

/// ONE canonical usage frame (audit Phase-1 item C): non-overlapping token
/// categories every provider wire is mapped into at the adapter boundary,
/// so the runtime never has to guess what `tokens_in` meant on a given
/// wire. The old `tokens_in`/`tokens_out` pair is gone — it meant different
/// things on different wires (one provider folds cached input into its
/// input total, another excludes it) and the runtime double-billed cache
/// reads.
///
/// Category contract:
/// - `uncached_input_tokens` NEVER contains cache reads or cache writes.
///   For wires whose input total INCLUDES the cached portion the adapter
///   must split it out ([`CanonicalUsage::from_total_including_cache`]);
///   wires that already report the uncached remainder map as-is.
/// - `cache_read_tokens` / `cache_write_tokens` are purely additive lines
///   priced at their own frozen route-time quote lines.
/// - `output_tokens` already includes reasoning tokens whenever the
///   provider's output charge does; `reasoning_tokens` is an INFORMATIONAL
///   subset that is never billed a second time (no adapter below has a
///   separately-priced reasoning line).
/// - `reported_cost`: the provider-reported authoritative cost when the
///   wire carries one, WITH its currency and source (only USD-compatible
///   values may override route-snapshot estimation; the runtime refuses
///   the rest).
/// - `request_id`: preserved from the wire frame when it carries one.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CanonicalUsage {
    #[serde(default)]
    pub uncached_input_tokens: u64,
    #[serde(default)]
    pub cache_read_tokens: u64,
    #[serde(default)]
    pub cache_write_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default)]
    pub reasoning_tokens: u64,
    #[serde(default)]
    pub reported_cost: Option<ReportedCost>,
    #[serde(default)]
    pub request_id: Option<String>,
}

impl Default for CanonicalUsage {
    fn default() -> Self {
        Self::ZERO
    }
}

impl CanonicalUsage {
    pub const ZERO: Self = Self {
        uncached_input_tokens: 0,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        output_tokens: 0,
        reasoning_tokens: 0,
        reported_cost: None,
        request_id: None,
    };

    /// The four priced token categories (reasoning folds into output at
    /// settlement — the informational subset stays zero here).
    pub fn new(
        uncached_input_tokens: u64,
        cache_read_tokens: u64,
        cache_write_tokens: u64,
        output_tokens: u64,
    ) -> Self {
        Self {
            uncached_input_tokens,
            cache_read_tokens,
            cache_write_tokens,
            output_tokens,
            ..Self::ZERO
        }
    }

    /// Canonicalize a wire whose input TOTAL already includes its cached
    /// portion (openai `prompt_tokens`, gemini `promptTokenCount`): the
    /// uncached remainder is `total - cache reads - cache writes`. A
    /// hostile row whose cache lines exceed the reported total — or whose
    /// informational reasoning subset exceeds the output total — is a
    /// typed [`UsageSplitError`], never a silent saturate.
    pub fn from_total_including_cache(
        total_input_tokens: u64,
        cache_read_tokens: u64,
        cache_write_tokens: u64,
        output_tokens: u64,
        reasoning_tokens: u64,
    ) -> Result<Self, UsageSplitError> {
        if cache_read_tokens > total_input_tokens {
            return Err(UsageSplitError::CacheReadsExceedInput {
                total_input_tokens,
                cache_read_tokens,
            });
        }
        let remainder = total_input_tokens - cache_read_tokens;
        if cache_write_tokens > remainder {
            return Err(UsageSplitError::CacheWritesExceedInput {
                total_input_tokens,
                cache_write_tokens,
            });
        }
        if reasoning_tokens > output_tokens {
            return Err(UsageSplitError::ReasoningExceedsOutput {
                output_tokens,
                reasoning_tokens,
            });
        }
        Ok(Self {
            uncached_input_tokens: remainder - cache_write_tokens,
            cache_read_tokens,
            cache_write_tokens,
            output_tokens,
            reasoning_tokens,
            ..Self::ZERO
        })
    }

    /// Validate the shared cross-wire invariants of an already-built frame
    /// (the informational reasoning subset never exceeds the output total;
    /// a wire that reports reasoning must have folded it into output).
    /// Adapters that assemble frames field-by-field (anthropic-style split
    /// wires, ollama counts) call this before emission.
    pub fn validate(&self) -> Result<(), UsageSplitError> {
        if self.reasoning_tokens > self.output_tokens {
            return Err(UsageSplitError::ReasoningExceedsOutput {
                output_tokens: self.output_tokens,
                reasoning_tokens: self.reasoning_tokens,
            });
        }
        Ok(())
    }

    /// Every priced category is zero (nothing to settle).
    pub fn is_zero(&self) -> bool {
        self.uncached_input_tokens == 0
            && self.cache_read_tokens == 0
            && self.cache_write_tokens == 0
            && self.output_tokens == 0
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProviderChunk {
    Text {
        text: String,
    },
    Reasoning {
        text: String,
    },
    ToolCall {
        id: String,
        name: String,
        /// Accumulated so far; `complete` toggles on the final delta.
        input: serde_json::Value,
        complete: bool,
    },
    /// Terminal usage settlement: one canonical usage frame per stream,
    /// usually the LAST one wins. Adapters map their WIRE usage to
    /// [`CanonicalUsage`] at their own boundary (read/write/uncached lines
    /// are already split, output already includes reasoning) — the agent
    /// consumes the canonical categories directly and never re-derives
    /// provider semantics.
    Usage(CanonicalUsage),
    Done,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderErrorKind {
    Network,
    Timeout,
    RateLimited,
    BadRequest,
    Auth,
    Server,
    Cancelled,
    Malformed,
}

impl ProviderErrorKind {
    pub fn retryable(&self) -> bool {
        matches!(
            self,
            ProviderErrorKind::Network
                | ProviderErrorKind::Timeout
                | ProviderErrorKind::RateLimited
                | ProviderErrorKind::Server
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderError {
    pub kind: ProviderErrorKind,
    pub message: String,
    pub retryable: bool,
    /// Provider-native code (http status, ollama error, ...).
    pub code: Option<String>,
}

impl ProviderError {
    pub fn new(kind: ProviderErrorKind, message: impl Into<String>) -> Self {
        let retryable = kind.retryable();
        Self {
            kind,
            message: message.into(),
            retryable,
            code: None,
        }
    }

    pub fn with_code(
        kind: ProviderErrorKind,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        let retryable = kind.retryable();
        Self {
            kind,
            message: message.into(),
            retryable,
            code: Some(code.into()),
        }
    }
}

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for ProviderError {}

impl From<UsageSplitError> for ProviderError {
    fn from(e: UsageSplitError) -> Self {
        ProviderError::new(
            ProviderErrorKind::Malformed,
            format!("hostile usage frame: {e}"),
        )
    }
}

pub type ProviderStream = Pin<Box<dyn Stream<Item = Result<ProviderChunk, ProviderError>> + Send>>;

/// Registry identity of one provider instance. Adapters report their
/// transport family (`id()` = "openai"), but a daemon can register several
/// OpenAI-compatible endpoints (two proxies, corp gateways, ...). The
/// registry keys by `instance_id` so every configured instance resolves;
/// the family id stays the capability/label face of the provider.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ProviderIdentity {
    pub instance_id: String,
    pub family: String,
}

impl ProviderIdentity {
    pub fn new(instance_id: impl Into<String>, family: impl Into<String>) -> Self {
        Self {
            instance_id: instance_id.into(),
            family: family.into(),
        }
    }

    /// Default identity: one instance per family (instance_id == family).
    pub fn from_family(family: impl Into<String>) -> Self {
        let family = family.into();
        Self {
            instance_id: family.clone(),
            family,
        }
    }
}

/// One transport family. Implementations are stateless except config.
pub trait Provider: Send + Sync {
    fn id(&self) -> &str;

    /// Capabilities for a model; discovered by probing, never hard-coded
    /// lists in the agent.
    fn capabilities(&self, model: &str) -> ModelCapabilities;

    /// A live runtime context bound for `model` in tokens, when the provider
    /// can report one (e.g. an Ollama `/api/ps` allocation clamped by the
    /// probed model maximum — the window actually loaded for the model can
    /// sit far below the advertised maximum). `None` means no override: the
    /// agent budgets from [`ModelCapabilities::context`] as usual (safe
    /// direction when no live data exists). Never exceeds the model's
    /// advertised maximum; the agent takes `min(caps.context, limit)`
    /// defensively anyway. Must be cheap: called synchronously on every turn
    /// plan, so adapters serve the CACHED last-refreshed value.
    fn runtime_context_limit(&self, _model: &str) -> Option<usize> {
        None
    }

    /// The models this provider can serve (configured + discovered +
    /// probed). Feeds the model-selector surface; never a fabricated list
    /// in the agent. Default: only the "default" entry.
    fn known_models(&self) -> Vec<String> {
        vec!["default".into()]
    }

    /// The real model-catalog row of one model (audit P0-1/wave-B item C):
    /// pricing state, quality priors, provenance and the pricing epoch the
    /// routing graph consumes. Adapters with real knowledge override this
    /// (Ollama rows are [`PricingState::LocalZero`]); the DEFAULT first
    /// consults the versioned built-in list-price table
    /// ([`catalog::builtin`]) by (provider-family, model): a documented
    /// model returns [`PricingState::Known`] with its EXACT
    /// per-million-token quote at epoch
    /// [`catalog::CATALOG_FIRST_EPOCH`] and source
    /// [`catalog::BUILTIN_SOURCE_ID`]. Everything else derives a
    /// conservative row with [`PricingState::Unknown`] — **never a zero or
    /// 1-microUSD fake price** — provenance [`Provenance::BuiltIn`] and
    /// epoch [`catalog::CATALOG_FIRST_EPOCH`]. Legacy adapters compile
    /// unchanged and their undocumented models read as Unknown until
    /// priced by config ([`catalog::PricingOverrides`]).
    fn catalog_entry(&self, model: &str) -> ModelCatalogEntry {
        let row = catalog::builtin::lookup(self.id(), model);
        match row {
            Some(row) => ModelCatalogEntry {
                provider: self.identity().instance_id,
                model: model.to_string(),
                capabilities: self.capabilities(model),
                pricing: PricingState::Known(PricingSnapshot::exact(
                    catalog::builtin::quote_of(&row),
                    catalog::CATALOG_FIRST_EPOCH,
                    catalog::BUILTIN_SOURCE_ID.to_string(),
                )),
                quality_prior: QualityPrior::default(),
                source_epoch: catalog::CATALOG_FIRST_EPOCH,
                provenance: Provenance::BuiltIn,
            },
            None => ModelCatalogEntry {
                provider: self.identity().instance_id,
                model: model.to_string(),
                capabilities: self.capabilities(model),
                pricing: PricingState::Unknown,
                quality_prior: QualityPrior::default(),
                source_epoch: catalog::CATALOG_FIRST_EPOCH,
                provenance: Provenance::BuiltIn,
            },
        }
    }

    fn stream(&self, req: GenericAgentRequest) -> ProviderStream;

    /// Registry identity. The default is one instance per family; daemon
    /// wiring overrides `instance_id` with the configured provider id so
    /// two OpenAI-compatible endpoints never overwrite each other.
    fn identity(&self) -> ProviderIdentity {
        ProviderIdentity::from_family(self.id())
    }
}

/// Wraps an adapter with an explicit registry instance id while keeping the
/// adapter's family `id()` for capability queries and wire metadata. The
/// CLI builds one wrapper per configured provider entry.
pub struct InstanceProvider {
    inner: Arc<dyn Provider>,
    instance_id: String,
}

impl InstanceProvider {
    pub fn wrap(inner: Arc<dyn Provider>, instance_id: impl Into<String>) -> Arc<dyn Provider> {
        Arc::new(Self {
            inner,
            instance_id: instance_id.into(),
        })
    }
}

impl Provider for InstanceProvider {
    fn id(&self) -> &str {
        self.inner.id()
    }

    fn identity(&self) -> ProviderIdentity {
        ProviderIdentity::new(self.instance_id.clone(), self.id())
    }

    fn capabilities(&self, model: &str) -> ModelCapabilities {
        self.inner.capabilities(model)
    }

    fn known_models(&self) -> Vec<String> {
        // Delegate: an instance-wrapped adapter reports its OWN model list
        // (the trait default of ["default"] would collapse every custom
        // endpoint's real catalog on the daemon's routing graph).
        self.inner.known_models()
    }

    fn runtime_context_limit(&self, model: &str) -> Option<usize> {
        // Delegate (audit round 11): an instance-wrapped Ollama provider
        // must still shrink the budget to the /api/ps allocation.
        self.inner.runtime_context_limit(model)
    }

    fn catalog_entry(&self, model: &str) -> ModelCatalogEntry {
        // Delegate the row and rewrite its provider to THIS instance id:
        // catalog rows must name the registry key the daemon resolves
        // (two OpenAI-compatible endpoints never share rows).
        let mut entry = self.inner.catalog_entry(model);
        entry.provider = self.instance_id.clone();
        entry
    }

    fn stream(&self, req: GenericAgentRequest) -> ProviderStream {
        self.inner.stream(req)
    }
}

// ------------------------------------------------------------------ pipeline

/// Step 1: validate a request against known capabilities *before* any wire
/// call. Violations are loud errors, never silent truncation.
pub struct CapabilityValidator;

impl CapabilityValidator {
    pub fn validate(
        req: &GenericAgentRequest,
        caps: &ModelCapabilities,
    ) -> Result<(), faktor_core::Error> {
        use faktor_core::error::{Error, ErrorKind};
        if !req.tools.is_empty() && !caps.tools {
            return Err(Error::new(
                ErrorKind::Malformed,
                format!(
                    "model {} does not support tools, but {} tool(s) requested",
                    req.model,
                    req.tools.len()
                ),
            ));
        }
        if req.reasoning.is_some() && !(caps.reasoning || caps.thinking) {
            return Err(Error::new(
                ErrorKind::Malformed,
                format!("model {} does not support reasoning", req.model),
            ));
        }
        if let Some(max_out) = req.max_output {
            if max_out > caps.max_output {
                return Err(Error::new(
                    ErrorKind::Oversized,
                    format!(
                        "requested max_output {max_out} exceeds model cap {}",
                        caps.max_output
                    ),
                ));
            }
        }
        Ok(())
    }
}

/// Step 2: ensure internal option names never leak onto wire APIs. The
/// normalizer strips anything not on the explicit whitelist and enforces
/// bound clamps. Adapters additionally translate to their wire vocabulary.
pub struct RequestNormalizer;

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct NormalizedRequest {
    pub model: String,
    pub system: String,
    pub messages: Vec<RequestMessage>,
    pub tools: Vec<ToolSpec>,
    pub max_output: Option<usize>,
    pub reasoning: Option<ReasoningMode>,
    pub stream: bool,
}

impl RequestNormalizer {
    /// Whitelisted internal fields (the frozen set). Anything else that ever
    /// sneaks into `GenericAgentRequest` will simply not exist on the wire —
    /// this is the structural fix for leaked compaction/option names.
    pub fn normalize(req: &GenericAgentRequest) -> NormalizedRequest {
        NormalizedRequest {
            model: req.model.clone(),
            system: req.system.clone(),
            messages: req.messages.clone(),
            tools: req.tools.clone(),
            max_output: req.max_output,
            reasoning: req.reasoning,
            stream: req.stream,
        }
    }
}

/// Hard bound on one provider instance id, in bytes (P0-41). The registry
/// refuses longer ids with a typed [`ErrorKind::Oversized`] error — hostile
/// or corrupt wiring never populates the map with unbounded keys. Mirrors
/// the 256-byte provider-name bound the session layer enforces.
pub const MAX_PROVIDER_INSTANCE_ID_BYTES: usize = 256;

/// Dynamic model registry: providers register their models; the agent asks
/// the registry, never the provider string.
#[derive(Default)]
pub struct ProviderRegistry {
    providers: HashMap<String, Arc<dyn Provider>>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// The ONE registration API (P0-41) — fallible and typed; there is no
    /// infallible duplicate-accepting shim anymore.
    ///
    /// Providers are keyed by their INSTANCE id (never the family id: two
    /// OpenAI-compatible endpoints with distinct configured ids must both
    /// register). The audited semantics:
    ///
    /// | registration | result |
    /// |---|---|
    /// | fresh id | `Ok`, inserted |
    /// | same id, SAME instance (`Arc::ptr_eq`) | `Ok`, no-op — idempotent |
    /// | same id, DIFFERENT instance | `Err(Conflict)`, FIRST entry kept, never replaced |
    /// | id differing only by case from an existing key | `Err(Conflict)`, FIRST entry kept |
    /// | empty id | `Err(Malformed)`, nothing inserted |
    /// | id over [`MAX_PROVIDER_INSTANCE_ID_BYTES`] | `Err(Oversized)`, nothing inserted |
    pub fn try_register(&mut self, p: Arc<dyn Provider>) -> Result<(), Error> {
        let id = p.identity().instance_id;
        if id.is_empty() {
            return Err(Error::new(
                ErrorKind::Malformed,
                "provider instance id is empty; refusing to register",
            ));
        }
        if id.len() > MAX_PROVIDER_INSTANCE_ID_BYTES {
            return Err(Error::new(
                ErrorKind::Oversized,
                format!(
                    "provider instance id is {} bytes, over the cap of {MAX_PROVIDER_INSTANCE_ID_BYTES}",
                    id.len()
                ),
            ));
        }
        if let Some(existing) = self.providers.get(&id) {
            if Arc::ptr_eq(existing, &p) {
                return Ok(());
            }
            return Err(Error::conflict(format!(
                "provider {id:?} already registered by a DIFFERENT instance; keeping the first entry"
            )));
        }
        if let Some(first) = self
            .providers
            .keys()
            .find(|k| k.to_lowercase() == id.to_lowercase())
        {
            return Err(Error::conflict(format!(
                "provider {id:?} is a case variant of already-registered {first:?}; keeping the first entry"
            )));
        }
        self.providers.insert(id, p);
        Ok(())
    }

    /// Every registered provider (daemon warm-up / diagnostics).
    pub fn all(&self) -> Vec<Arc<dyn Provider>> {
        self.providers.values().cloned().collect()
    }

    pub fn get(&self, id: &str) -> Option<Arc<dyn Provider>> {
        self.providers.get(id).cloned()
    }

    pub fn ids(&self) -> Vec<String> {
        let mut v: Vec<String> = self.providers.keys().cloned().collect();
        v.sort();
        v
    }

    pub fn capabilities(&self, provider: &str, model: &str) -> Option<ModelCapabilities> {
        self.get(provider).map(|p| p.capabilities(model))
    }

    pub fn len(&self) -> usize {
        self.providers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }
}

// ------------------------------------------------------------------ tokenizer identity

/// Tokenizer family of a provider-known model (P0-81). The family is the
/// *static* identity a model's tokenizer is known by (tiktoken's o200k_base
/// for the modern GPT family, Anthropic's own tokenizer, ...); a real local
/// tokenizer implementation may one day name itself by family + version.
/// `GenericEstimator` is the conservative fallback — no exact tokenizer
/// exists for it, only the bounded generic estimator.
///
/// Variant order is the [`Ord`] order (deterministic; never change it).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum TokenFamily {
    /// OpenAI o200k_base (gpt-4o / gpt-4.1 / o1 / o3 / gpt-5 families).
    O200kBase,
    /// OpenAI cl100k_base (gpt-3.5 / gpt-4 families).
    Cl100kBase,
    /// Anthropic's tokenizer (claude models).
    Anthropic,
    /// Google's tokenizer (gemini models).
    Gemini,
    /// Meta/Llama-family BPE (llama, qwen, and llama-hosted distills).
    Llama,
    /// No provider-known tokenizer: the conservative generic estimator.
    GenericEstimator,
}

impl TokenFamily {
    /// Machine-readable family name (stable, lowercase, snake_case).
    pub const fn as_str(self) -> &'static str {
        match self {
            TokenFamily::O200kBase => "o200k_base",
            TokenFamily::Cl100kBase => "cl100k_base",
            TokenFamily::Anthropic => "anthropic",
            TokenFamily::Gemini => "gemini",
            TokenFamily::Llama => "llama",
            TokenFamily::GenericEstimator => "generic_estimator",
        }
    }
}

impl std::fmt::Display for TokenFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Versioned identity of the tokenizer a model family maps to (P0-81).
/// `version` distinguishes tokenizer API generations of one family — it is
/// frozen at `1` for every family today and must bump if a provider's
/// tokenizer vocabulary/API changes (cache entries and exact counts are
/// keyed by the full identity, so a version bump invalidates stale counts
/// instead of silently reusing them).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct TokenizerId {
    pub family: TokenFamily,
    pub version: u32,
}

impl TokenizerId {
    /// o200k_base, generation 1 (the tiktoken vocabulary as frozen today).
    pub const O200K_BASE: TokenizerId = TokenizerId {
        family: TokenFamily::O200kBase,
        version: 1,
    };
    /// cl100k_base, generation 1.
    pub const CL100K_BASE: TokenizerId = TokenizerId {
        family: TokenFamily::Cl100kBase,
        version: 1,
    };
    /// Anthropic's tokenizer, generation 1.
    pub const ANTHROPIC: TokenizerId = TokenizerId {
        family: TokenFamily::Anthropic,
        version: 1,
    };
    /// Google's tokenizer, generation 1.
    pub const GEMINI: TokenizerId = TokenizerId {
        family: TokenFamily::Gemini,
        version: 1,
    };
    /// Llama-family BPE, generation 1.
    pub const LLAMA: TokenizerId = TokenizerId {
        family: TokenFamily::Llama,
        version: 1,
    };
    /// The conservative fallback: no exact tokenizer, only the generic
    /// estimator. This is what unknown models map to.
    pub const GENERIC_ESTIMATOR: TokenizerId = TokenizerId {
        family: TokenFamily::GenericEstimator,
        version: 1,
    };
}

impl Default for TokenizerId {
    fn default() -> Self {
        TokenizerId::GENERIC_ESTIMATOR
    }
}

impl std::fmt::Display for TokenizerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@v{}", self.family, self.version)
    }
}

/// Match `s` after the LAST `/` (routed model strings are often
/// `provider/model`), trimmed and lowercased.
fn tokenizer_model_base(model: &str) -> String {
    model
        .rsplit('/')
        .next()
        .unwrap_or(model)
        .trim()
        .to_ascii_lowercase()
}

fn starts_with_any(s: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|p| s.starts_with(p))
}

/// The PURE model → tokenizer mapping (P0-81): decides which
/// [`TokenizerId`] a model string names, statically, from model-name
/// prefixes. Provider behavior is never probed here and no remote
/// tokenizer API is ever consulted.
///
/// Ordering contract (longest/specific prefixes first — `gpt-4o` MUST be
/// tested before `gpt-4`, or every gpt-4o row would mis-map to cl100k):
///
/// | prefix (lowercased, after the last `/`) | family |
/// |---|---|
/// | `gpt-4o`, `gpt-4.1`, `gpt-5`, `o1`, `o3` | `O200kBase` |
/// | `gpt-3.5`, `gpt-4` | `Cl100kBase` |
/// | `claude` | `Anthropic` |
/// | `gemini` | `Gemini` |
/// | `llama`, `qwen` | `Llama` |
/// | `deepseek` (only under a llama-family deployment hint) | `Llama` |
/// | anything else | `GenericEstimator` |
///
/// `catalog_hint` is an optional deployment hint (typically the provider
/// family id, e.g. `"ollama"`). It never UPGRADES an unknown model to a
/// named family — an OpenAI-compatible endpoint is not an OpenAI tokenizer
/// — and is consulted only where a model string is genuinely ambiguous
/// (`deepseek-*` weights served by a llama-family local runtime map to the
/// Llama tokenizer; `deepseek-*` on the official API keeps DeepSeek's own
/// tokenizer, which is NOT llama's, so it conservatively maps to
/// `GenericEstimator`). Every row maps to `version: 1` today.
pub fn tokenizer_for(model: &str, catalog_hint: Option<&str>) -> TokenizerId {
    let base = tokenizer_model_base(model);
    let hint = catalog_hint.unwrap_or("").to_ascii_lowercase();
    if starts_with_any(&base, &["gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3"]) {
        TokenizerId::O200K_BASE
    } else if starts_with_any(&base, &["gpt-3.5", "gpt-4"]) {
        TokenizerId::CL100K_BASE
    } else if base.starts_with("claude") {
        TokenizerId::ANTHROPIC
    } else if base.starts_with("gemini") {
        TokenizerId::GEMINI
    } else if (base.starts_with("llama") || base.starts_with("qwen"))
        || (base.starts_with("deepseek") && hint.contains("llama"))
    {
        TokenizerId::LLAMA
    } else {
        TokenizerId::GENERIC_ESTIMATOR
    }
}

pub use std::sync::Arc;

pub mod transport;

/// Parsed destination gate + checked outbound HTTP client (audits 36-37):
/// every egress decision runs against a parsed (scheme, host, port) triple
/// before the connection is attempted.
pub mod egress;

/// Adversarial wire-testing harness (mock HTTP server).
pub mod testing;

/// Shared canonical-usage conformance support (audit Phase-1 item C). The
/// [`canonical_usage_conformance!`](crate::canonical_usage_conformance)
/// macro is the driver; this module carries the per-wire-family required
/// case tables and the case row type. Adapters instantiate the macro in a
/// `#[cfg(test)]` module with mock wire bodies that match their REAL wire
/// shapes; the driver asserts every required case of the adapter's family
/// is present and that each case's stream yields exactly the expected
/// canonical frame (or a typed `Malformed` error for hostile rows).
#[doc(hidden)]
pub mod usage_conformance {
    use super::{CanonicalUsage, ProviderChunk};

    /// Which wire semantics the adapter's usage envelope has. Every
    /// instantiation must carry a [`WireUsageCase`] per required name of
    /// its family (enforced by the driver macro), so the audit case list is
    /// exercised once per adapter against that adapter's true wire shape.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum WireFamily {
        /// The wire input total INCLUDES cache reads (openai `prompt_tokens`
        /// with `prompt_tokens_details.cached_tokens`, gemini
        /// `promptTokenCount` with `cachedContentTokenCount`):
        /// canonicalization subtracts the cached portion, and a cached>total
        /// row is Malformed.
        InclusiveTotal,
        /// The wire input counter EXCLUDES cache reads and reports cache
        /// creation/write lines separately (anthropic `input_tokens` /
        /// `cache_read_input_tokens` / `cache_creation_input_tokens`):
        /// canonical categories map as-is, modulo the creation lines that
        /// ride inside `input_tokens`.
        SplitCache,
        /// The wire reports a single input count with NO cache split at all
        /// (ollama `prompt_eval_count`): the conservative category is
        /// uncached = the reported count, cache lines zero.
        NoCacheDetail,
    }

    /// Required case names per wire family. These ARE the audit Phase-1
    /// item C cases: total-including-cache split vs. pre-split identity,
    /// cache-detail missing (uncached = total), hostile cache>total typed
    /// Malformed, reasoning subset never double-billed, hostile reasoning
    /// subset, unknown fields never panicking, request id preservation, and
    /// the wire-family-specific rows.
    pub fn required_cases(family: WireFamily) -> &'static [&'static str] {
        match family {
            WireFamily::InclusiveTotal => &[
                "total_incl_cached_split",
                "cache_detail_missing_uncached_total",
                "hostile_cache_over_total",
                "reasoning_subset_inside_output",
                "hostile_reasoning_over_output",
                "unknown_fields_never_panic",
                "request_id_preserved",
            ],
            WireFamily::SplitCache => &[
                "split_input_cache_identity",
                "cache_write_inside_input_total_split",
                "hostile_cache_write_over_input",
                "cache_detail_missing_uncached_total",
                "cache_only_frame_no_input",
                "unknown_fields_never_panic",
            ],
            WireFamily::NoCacheDetail => &[
                "counts_map_uncached_total",
                "thinking_included_in_output_never_double_billed",
                "hostile_junk_counts_never_panic",
                "zero_counts_no_usage_frame",
            ],
        }
    }

    /// What one conformance case must produce.
    #[derive(Debug, Clone, PartialEq)]
    pub enum WireUsageExpectation {
        /// The stream must surface exactly one canonical usage frame equal
        /// to this (any leading text/reasoning/tool chunks are allowed —
        /// e.g. ollama's thinking/content share the final frame), followed
        /// by `Done`, with no errors anywhere.
        Frame(CanonicalUsage),
        /// The stream must fail exactly once with a typed `Malformed`
        /// error and emit no usage frame.
        Malformed,
        /// The stream must complete cleanly with no usage frame at all
        /// (hostile junk rows an adapter ignores, all-zero counter rows).
        NoUsageFrame,
    }

    /// One conformance row: a name from the family's required table (the
    /// driver asserts presence), the full wire body bytes, and what the
    /// stream must yield.
    #[derive(Debug, Clone)]
    pub struct WireUsageCase {
        pub name: &'static str,
        pub body: String,
        pub expect: WireUsageExpectation,
    }

    impl WireUsageCase {
        pub fn frame(name: &'static str, body: impl Into<String>, usage: CanonicalUsage) -> Self {
            Self {
                name,
                body: body.into(),
                expect: WireUsageExpectation::Frame(usage),
            }
        }

        pub fn malformed(name: &'static str, body: impl Into<String>) -> Self {
            Self {
                name,
                body: body.into(),
                expect: WireUsageExpectation::Malformed,
            }
        }

        pub fn no_usage(name: &'static str, body: impl Into<String>) -> Self {
            Self {
                name,
                body: body.into(),
                expect: WireUsageExpectation::NoUsageFrame,
            }
        }
    }

    /// Reduce one driven stream for failure assertions.
    pub fn usage_index(items: &[Result<ProviderChunk, super::ProviderError>]) -> Option<usize> {
        items
            .iter()
            .position(|i| matches!(i, Ok(ProviderChunk::Usage(_))))
    }
}

/// Adversarial canonical-usage conformance driver (shared test harness).
///
/// Expand once per adapter inside a `#[cfg(test)]` module:
///
/// ```ignore
/// canonical_usage_conformance! {
///     driver: openai_chat_usage_conformance,
///     family: faktor_provider::usage_conformance::WireFamily::InclusiveTotal,
///     label: "openai chat completions",
///     request: || req("m1"),
///     provider: |base: String| OpenAiProvider::build(OpenAiConfig::chat(base, None)),
///     method: "POST",
///     path: "/chat/completions",
///     cases: vec![ /* one WireUsageCase per required case name */ ],
/// }
/// ```
///
/// The driver asserts (1) the adapter's case table covers EVERY required
/// case name of its wire family, and (2) each case driven over a real
/// provider stream against a mock HTTP server yields exactly the expected
/// canonical usage frame then `Done` — or fails exactly once with a typed
/// `Malformed` error for hostile rows, or completes cleanly without a
/// usage frame where the expectation says so. Unknown-field payloads and
/// absurd values can never panic: a panic fails the test loudly.
#[macro_export]
macro_rules! canonical_usage_conformance {
    (
        driver: $driver:ident,
        family: $family:expr,
        label: $label:expr,
        request: $request:expr,
        provider: $provider:expr,
        method: $method:expr,
        path: $path:expr,
        cases: $cases:expr
    ) => {
        #[::tokio::test]
        async fn $driver() {
            use ::futures::StreamExt as _;
            use $crate::usage_conformance::{
                required_cases, WireUsageExpectation, WireUsageCase,
            };
            let cases: Vec<WireUsageCase> = $cases;
            assert!(
                !cases.is_empty(),
                "{}: at least one conformance case is required",
                $label
            );
            for (i, c) in cases.iter().enumerate() {
                for (j, other) in cases.iter().enumerate() {
                    assert!(
                        i == j || c.name != other.name,
                        "{}: duplicate conformance case name {:?}",
                        $label,
                        c.name
                    );
                }
            }
            let have: Vec<&str> = cases.iter().map(|c| c.name).collect();
            let required = required_cases($family);
            for want in required {
                assert!(
                    have.contains(want),
                    "{} conformance is missing required case {want:?} (have {have:?})",
                    $label
                );
            }
            for case in &cases {
                let server = $crate::testing::MockServer::new();
                server.route(
                    $method,
                    $path,
                    $crate::testing::MockAction::Respond {
                        status: 200,
                        body: case.body.clone(),
                    },
                );
                let base = server.base_url().await;
                let provider = $provider(base);
                let mut stream = provider.stream($request());
                let mut items: Vec<Result<$crate::ProviderChunk, $crate::ProviderError>> =
                    Vec::new();
                while let Some(item) = stream.next().await {
                    items.push(item);
                }
                match &case.expect {
                    WireUsageExpectation::Frame(expected) => {
                        let usage_at = $crate::usage_conformance::usage_index(&items);
                        let usage_at = usage_at.unwrap_or_else(|| {
                            panic!(
                                "{} case {:?} must emit a canonical usage frame; got {items:?}",
                                $label, case.name
                            )
                        });
                        for (k, item) in items[..usage_at].iter().enumerate() {
                            assert!(
                                matches!(
                                    item,
                                    Ok($crate::ProviderChunk::Text { .. })
                                        | Ok($crate::ProviderChunk::Reasoning { .. })
                                        | Ok($crate::ProviderChunk::ToolCall { .. })
                                ),
                                "{} case {:?}: unexpected item before the usage frame at \
                                 index {k}: {item:?}",
                                $label,
                                case.name
                            );
                        }
                        assert_eq!(
                            items[usage_at],
                            Ok($crate::ProviderChunk::Usage(expected.clone())),
                            "{} case {:?}: canonical usage mismatch",
                            $label,
                            case.name
                        );
                        assert_eq!(
                            usage_at + 1,
                            items.len().saturating_sub(1),
                            "{} case {:?}: usage must be the last chunk before Done \
                             (exactly one usage frame per stream); got {items:?}",
                            $label,
                            case.name
                        );
                        assert_eq!(
                            items.last(),
                            Some(&Ok($crate::ProviderChunk::Done)),
                            "{} case {:?}: stream must end with Done",
                            $label,
                            case.name
                        );
                    }
                    WireUsageExpectation::Malformed => {
                        let errs: Vec<&$crate::ProviderError> =
                            items.iter().filter_map(|i| i.as_ref().err()).collect();
                        assert_eq!(
                            errs.len(),
                            1,
                            "{} case {:?}: a hostile row must fail exactly once; got {items:?}",
                            $label,
                            case.name
                        );
                        assert_eq!(
                            errs[0].kind,
                            $crate::ProviderErrorKind::Malformed,
                            "{} case {:?}: hostile usage rows are typed Malformed; got {:?}",
                            $label,
                            case.name,
                            errs[0]
                        );
                        assert!(
                            $crate::usage_conformance::usage_index(&items).is_none(),
                            "{} case {:?}: no usage frame may follow a Malformed error; got {items:?}",
                            $label,
                            case.name
                        );
                    }
                    WireUsageExpectation::NoUsageFrame => {
                        assert!(
                            items.iter().all(|i| i.is_ok()),
                            "{} case {:?}: hostile junk must never error or panic; got {items:?}",
                            $label,
                            case.name
                        );
                        assert!(
                            $crate::usage_conformance::usage_index(&items).is_none(),
                            "{} case {:?}: no usage frame expected; got {items:?}",
                            $label,
                            case.name
                        );
                        assert_eq!(
                            items.last(),
                            Some(&Ok($crate::ProviderChunk::Done)),
                            "{} case {:?}: stream must still end with Done",
                            $label,
                            case.name
                        );
                    }
                }
            }
        }
    };
}

// ------------------------------------------------------------------ fake provider for tests

/// Scripted multi-model catalog provider (registry-mirror helper for the
/// economy certification suite): every known model carries its OWN
/// capabilities, `known_models` reports catalog insertion order and
/// `capabilities` resolves per model. There is no streaming behavior —
/// catalog/registry-mirror tests never settle a paid call; an empty stream
/// is served if one is ever requested.
pub struct CatalogProvider {
    id: String,
    default_caps: ModelCapabilities,
    models: Vec<(String, ModelCapabilities)>,
}

impl CatalogProvider {
    /// A catalog with no known models yet (`default_caps` answers
    /// `capabilities` for models the catalog does not name).
    pub fn new(id: impl Into<String>, default_caps: ModelCapabilities) -> Self {
        Self {
            id: id.into(),
            default_caps,
            models: Vec::new(),
        }
    }

    /// Add one known model with its own capabilities. Appends after the
    /// models added before it: `known_models()` reports insertion order.
    pub fn add_model(&mut self, model: impl Into<String>, caps: ModelCapabilities) -> &mut Self {
        self.models.push((model.into(), caps));
        self
    }
}

impl Provider for CatalogProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn capabilities(&self, model: &str) -> ModelCapabilities {
        self.models
            .iter()
            .find(|(m, _)| m == model)
            .map(|(_, caps)| caps.clone())
            .unwrap_or_else(|| self.default_caps.clone())
    }

    fn known_models(&self) -> Vec<String> {
        self.models.iter().map(|(m, _)| m.clone()).collect()
    }

    fn stream(&self, _req: GenericAgentRequest) -> ProviderStream {
        Box::pin(futures::stream::empty())
    }
}

/// Scripted provider for adversarial agent/server tests. Responses are
/// user-controlled; streams can be made to die mid-flight, return malformed
/// tool calls, rate-limit, etc.
pub struct FakeProvider {
    pub id: String,
    pub caps: ModelCapabilities,
    pub script: std::sync::Mutex<Vec<ScriptedResponse>>,
    pub fail_after_chunks: Option<usize>,
    /// One-shot: the FIRST stream call errors before any chunk; later
    /// calls delegate to the script (state-aware-retry tests).
    fail_once_before: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// The `model` of the most recent request streamed through this
    /// provider (test hook: asserts what the agent actually sent).
    last_model: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    /// The cancellation token of the most recent request (test hook:
    /// asserts the provider request shares the turn's cancellation lineage).
    last_cancellation: std::sync::Arc<std::sync::Mutex<Option<CancellationToken>>>,
}

#[derive(Debug, Clone)]
pub enum ScriptedResponse {
    Text(String),
    Reasoning(String),
    ToolCall {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// Ends the stream cleanly.
    End,
    /// Simulates a network/stream death (error mid-flight).
    Die(ProviderError),
}

impl FakeProvider {
    pub fn new(id: &str, caps: ModelCapabilities) -> Self {
        Self {
            id: id.to_string(),
            caps,
            script: std::sync::Mutex::new(vec![ScriptedResponse::End]),
            fail_after_chunks: None,
            fail_once_before: None,
            last_model: std::sync::Arc::new(std::sync::Mutex::new(None)),
            last_cancellation: std::sync::Arc::new(std::sync::Mutex::new(None)),
        }
    }

    pub fn with_script(id: &str, caps: ModelCapabilities, script: Vec<ScriptedResponse>) -> Self {
        Self {
            id: id.to_string(),
            caps,
            script: std::sync::Mutex::new(script),
            fail_after_chunks: None,
            fail_once_before: None,
            last_model: std::sync::Arc::new(std::sync::Mutex::new(None)),
            last_cancellation: std::sync::Arc::new(std::sync::Mutex::new(None)),
        }
    }

    /// Fail the FIRST stream call with a retryable network error BEFORE
    /// any chunk (the state-aware-retry test's pre-accept failure); later
    /// streams serve the script normally.
    pub fn die_before_stream(
        id: &str,
        caps: ModelCapabilities,
        script: Vec<ScriptedResponse>,
    ) -> Self {
        Self {
            id: id.to_string(),
            caps,
            script: std::sync::Mutex::new(script),
            fail_after_chunks: None,
            fail_once_before: Some(std::sync::Arc::new(std::sync::atomic::AtomicBool::new(
                false,
            ))),
            last_model: std::sync::Arc::new(std::sync::Mutex::new(None)),
            last_cancellation: std::sync::Arc::new(std::sync::Mutex::new(None)),
        }
    }

    pub fn die_mid_stream(id: &str, caps: ModelCapabilities) -> Self {
        Self {
            id: id.to_string(),
            caps,
            script: std::sync::Mutex::new(vec![ScriptedResponse::Text("partial reply…".into())]),
            fail_after_chunks: Some(1),
            fail_once_before: None,
            last_model: std::sync::Arc::new(std::sync::Mutex::new(None)),
            last_cancellation: std::sync::Arc::new(std::sync::Mutex::new(None)),
        }
    }

    /// The model of the last request this provider was asked to stream
    /// (`None` when nothing was streamed yet).
    pub fn last_request_model(&self) -> Option<String> {
        self.last_model.lock().unwrap().clone()
    }

    /// The cancellation token of the last request this provider was asked to
    /// stream (`None` when nothing was streamed yet).
    pub fn last_request_cancellation(&self) -> Option<CancellationToken> {
        self.last_cancellation.lock().unwrap().clone()
    }

    /// If true, the next call fails with RateLimited (and the script is
    /// untouched) — used for retry tests.
    pub fn inject_rate_limit(&self) {
        self.script.lock().unwrap().insert(
            0,
            ScriptedResponse::Die(ProviderError::new(
                ProviderErrorKind::RateLimited,
                "429 too many",
            )),
        );
    }
}

impl Clone for FakeProvider {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            caps: self.caps.clone(),
            script: std::sync::Mutex::new(self.script.lock().unwrap().clone()),
            fail_after_chunks: self.fail_after_chunks,
            fail_once_before: self.fail_once_before.clone(),
            last_model: self.last_model.clone(),
            last_cancellation: self.last_cancellation.clone(),
        }
    }
}

impl Provider for FakeProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn capabilities(&self, _model: &str) -> ModelCapabilities {
        self.caps.clone()
    }

    fn stream(&self, req: GenericAgentRequest) -> ProviderStream {
        // One-shot pre-accept failure (state-aware retry tests).
        if let Some(flag) = &self.fail_once_before {
            if !flag.swap(true, std::sync::atomic::Ordering::SeqCst) {
                return Box::pin(futures::stream::iter(vec![Err(ProviderError::new(
                    ProviderErrorKind::Network,
                    "connection reset (injected once)",
                ))]));
            }
        }
        // Test hook: record exactly which model the agent sent.
        *self.last_model.lock().unwrap() = Some(req.model.clone());
        // Test hook: record the request's cancellation token so tests can
        // assert it shares the turn's cancellation lineage.
        *self.last_cancellation.lock().unwrap() = Some(req.meta.cancellation.clone());
        // Scripts are consumed exactly once (a replaying provider would let
        // the agent loop forever re-executing the same calls).
        let script = std::mem::take(&mut *self.script.lock().unwrap());
        let fail_after = self.fail_after_chunks;
        let stream = futures::stream::unfold(
            (script.into_iter(), 0usize, fail_after, false),
            move |(mut remaining, mut emitted, fail_after, ended)| async move {
                if ended {
                    return None; // exactly one terminal item, then end
                }
                if let Some(limit) = fail_after {
                    if emitted >= limit {
                        return Some((
                            Err(ProviderError::new(
                                ProviderErrorKind::Network,
                                "connection vanished mid-stream (injected)",
                            )),
                            (remaining, emitted, fail_after, true),
                        ));
                    }
                }
                match remaining.next() {
                    Some(ScriptedResponse::Text(t)) => {
                        emitted += 1;
                        Some((
                            Ok(ProviderChunk::Text { text: t }),
                            (remaining, emitted, fail_after, false),
                        ))
                    }
                    Some(ScriptedResponse::Reasoning(t)) => {
                        emitted += 1;
                        Some((
                            Ok(ProviderChunk::Reasoning { text: t }),
                            (remaining, emitted, fail_after, false),
                        ))
                    }
                    Some(ScriptedResponse::ToolCall { id, name, input }) => {
                        emitted += 1;
                        Some((
                            Ok(ProviderChunk::ToolCall {
                                id,
                                name,
                                input,
                                complete: true,
                            }),
                            (remaining, emitted, fail_after, false),
                        ))
                    }
                    Some(ScriptedResponse::Die(e)) => {
                        emitted += 1;
                        Some((Err(e), (remaining, emitted, fail_after, true)))
                    }
                    Some(ScriptedResponse::End) | None => {
                        let _ = req.meta.deadline_ms;
                        Some((
                            Ok(ProviderChunk::Done),
                            (remaining, emitted, fail_after, true),
                        ))
                    }
                }
            },
        );
        Box::pin(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faktor_core::error::ErrorKind;

    fn req() -> GenericAgentRequest {
        GenericAgentRequest {
            model: "m".into(),
            system: "s".into(),
            messages: vec![],
            tools: vec![],
            max_output: None,
            reasoning: None,
            stream: true,
            meta: RequestMeta {
                operation_id: OpId::new(1),
                session_id: SessionId::new(1),
                provider: "fake".into(),
                attempt: 0,
                deadline_ms: 1000,
                cancellation: CancellationToken::new(),
            },
        }
    }

    #[test]
    fn canonical_usage_fields_round_trip_with_defaults() {
        // Audit Phase-1 item C: a usage frame is the canonical category
        // split — uncached input, cache lines and output. Older additive
        // shapes are gone: a legacy `tokens_in` envelope is NOT a usage
        // frame (it carries no canonical categories).
        let missing = serde_json::json!({"type": "usage"});
        let usage: ProviderChunk = serde_json::from_value(missing).unwrap();
        match usage {
            ProviderChunk::Usage(usage) => {
                assert!(usage.is_zero());
                assert_eq!(usage.reported_cost, None);
                assert_eq!(usage.request_id, None);
            }
            other => panic!("usage frame mis-parsed: {other:?}"),
        }
        // And the canonical fields round-trip.
        let rich = ProviderChunk::Usage(CanonicalUsage {
            uncached_input_tokens: 10,
            cache_read_tokens: 7,
            cache_write_tokens: 2,
            output_tokens: 5,
            reasoning_tokens: 3,
            reported_cost: Some(ReportedCost {
                micro_usd: 42,
                currency: ReportedCurrency::Usd,
                source: ReportedCostSource::ProviderUsage,
                request_id: Some("req_1".into()),
            }),
            request_id: Some("req_1".into()),
        });
        let back: ProviderChunk =
            serde_json::from_value(serde_json::to_value(&rich).unwrap()).unwrap();
        assert_eq!(back, rich);
    }

    #[test]
    fn total_including_cache_splits_and_refuses_hostile_rows() {
        // Wire total INCLUDING cached input: 1000 total / 600 cached ->
        // uncached 400 + cache_read 600, output untouched.
        let u = CanonicalUsage::from_total_including_cache(1000, 600, 0, 50, 0).unwrap();
        assert_eq!(
            u,
            CanonicalUsage {
                uncached_input_tokens: 400,
                cache_read_tokens: 600,
                ..CanonicalUsage::new(0, 0, 0, 50)
            }
        );
        // A full cache hit is legal (uncached 0)...
        let hit = CanonicalUsage::from_total_including_cache(600, 600, 0, 50, 0).unwrap();
        assert_eq!(hit.uncached_input_tokens, 0);
        // ...but cache > total is hostile and typed, never saturated.
        assert_eq!(
            CanonicalUsage::from_total_including_cache(100, 600, 0, 50, 0).unwrap_err(),
            UsageSplitError::CacheReadsExceedInput {
                total_input_tokens: 100,
                cache_read_tokens: 600,
            }
        );
        // Reasoning must be an informational subset of output.
        assert_eq!(
            CanonicalUsage::from_total_including_cache(1000, 0, 0, 20, 30).unwrap_err(),
            UsageSplitError::ReasoningExceedsOutput {
                output_tokens: 20,
                reasoning_tokens: 30,
            }
        );
        // Cache writes must fit the remainder after reads.
        assert_eq!(
            CanonicalUsage::from_total_including_cache(500, 400, 200, 50, 0).unwrap_err(),
            UsageSplitError::CacheWritesExceedInput {
                total_input_tokens: 500,
                cache_write_tokens: 200,
            }
        );
        // A wire that reports reasoning must have folded it into output.
        let split = CanonicalUsage {
            output_tokens: 0,
            reasoning_tokens: 3,
            ..CanonicalUsage::ZERO
        };
        assert_eq!(
            split.validate().unwrap_err(),
            UsageSplitError::ReasoningExceedsOutput {
                output_tokens: 0,
                reasoning_tokens: 3,
            }
        );
    }

    #[test]
    fn capability_validation_rejects_tools_on_tool_less_model() {
        let caps = ModelCapabilities {
            tools: false,
            ..Default::default()
        };
        let mut r = req();
        r.tools.push(ToolSpec {
            name: "read_file".into(),
            description: "d".into(),
            input_schema: serde_json::json!({}),
        });
        let err = CapabilityValidator::validate(&r, &caps).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Malformed);
    }

    #[test]
    fn capability_validation_rejects_reasoning_and_oversized_output() {
        let caps = ModelCapabilities {
            reasoning: false,
            thinking: false,
            max_output: 1000,
            ..Default::default()
        };
        let mut r = req();
        r.reasoning = Some(ReasoningMode::High);
        assert!(CapabilityValidator::validate(&r, &caps).is_err());
        r.reasoning = None;
        r.max_output = Some(2000);
        let err = CapabilityValidator::validate(&r, &caps).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Oversized);
        r.max_output = Some(1000);
        assert!(CapabilityValidator::validate(&r, &caps).is_ok());
    }

    #[test]
    fn normalizer_never_carries_internal_meta() {
        let r = req();
        let n = RequestNormalizer::normalize(&r);
        // Internal fields (op/session/deadline/cancellation/attempt) simply
        // do not exist on the normalized request — they cannot leak.
        let json = serde_json::to_value(&n).unwrap();
        let obj = json.as_object().unwrap();
        for leaked in [
            "operation_id",
            "session_id",
            "attempt",
            "deadline_ms",
            "cancellation",
        ] {
            assert!(!obj.contains_key(leaked), "internal field {leaked} leaked");
        }
        assert_eq!(obj.len(), 7, "frozen normalized shape");
    }

    #[test]
    fn registry_dynamic_and_capability_source_of_truth() {
        let mut reg = ProviderRegistry::new();
        let fake = FakeProvider::with_script(
            "test",
            ModelCapabilities {
                context: 262144,
                tools: true,
                ..Default::default()
            },
            vec![ScriptedResponse::Text("hi".into()), ScriptedResponse::End],
        );
        reg.try_register(Arc::new(fake)).unwrap();
        assert_eq!(reg.ids(), vec!["test"]);
        let caps = reg.capabilities("test", "qwen3.8").unwrap();
        assert_eq!(caps.context, 262144);
        assert!(caps.tools);
        assert!(reg.capabilities("missing", "x").is_none());
    }

    #[test]
    fn two_openai_compatible_instances_both_register_and_resolve_by_id() {
        // Two OpenAI-compatible endpoints (family "openai") with distinct
        // configured instance ids: on the old registry both inserted under
        // the family id and the second silently overwrote the first.
        let mut reg = ProviderRegistry::new();
        let caps = ModelCapabilities::default();
        for id in ["a-proxy", "b-proxy"] {
            let fake = FakeProvider::with_script(
                "openai",
                caps.clone(),
                vec![ScriptedResponse::Text(id.into()), ScriptedResponse::End],
            );
            reg.try_register(InstanceProvider::wrap(Arc::new(fake), id))
                .unwrap();
        }
        assert_eq!(reg.ids(), vec!["a-proxy", "b-proxy"]);
        assert_eq!(reg.len(), 2, "both instances must survive registration");
        assert!(reg.get("a-proxy").is_some());
        assert!(reg.get("b-proxy").is_some());
        assert!(reg.capabilities("a-proxy", "gpt-5").is_some());
    }

    #[test]
    fn configured_instance_id_resolves_not_family_id() {
        // A provider configured with id "corp-proxy" (family "openai"):
        // sessions configured with "corp-proxy" resolve, and the family id
        // "openai" must NOT resolve to it (the old registry keyed the
        // adapter's family id, so custom ids never looked up).
        let fake = FakeProvider::with_script(
            "openai",
            ModelCapabilities::default(),
            vec![ScriptedResponse::Text("hi".into()), ScriptedResponse::End],
        );
        let wrapped = InstanceProvider::wrap(Arc::new(fake), "corp-proxy");
        assert_eq!(wrapped.identity().instance_id, "corp-proxy");
        assert_eq!(wrapped.identity().family, "openai");
        assert_eq!(
            wrapped.id(),
            "openai",
            "family id stays for capability queries"
        );

        let mut reg = ProviderRegistry::new();
        reg.try_register(wrapped).unwrap();
        assert_eq!(reg.ids(), vec!["corp-proxy"]);
        assert!(reg.get("corp-proxy").is_some());
        assert!(
            reg.get("openai").is_none(),
            "family id must not shadow the instance"
        );
    }

    #[test]
    fn default_identity_is_instance_per_family() {
        // An unwrapped adapter gets instance_id == family: existing single-
        // instance deployments keep resolving exactly as before.
        let fake = FakeProvider::new("ollama", ModelCapabilities::default());
        assert_eq!(fake.identity(), ProviderIdentity::new("ollama", "ollama"));
    }

    #[test]
    fn provider_error_retryability_matches_kind() {
        assert!(ProviderErrorKind::Network.retryable());
        assert!(ProviderErrorKind::Timeout.retryable());
        assert!(ProviderErrorKind::RateLimited.retryable());
        assert!(ProviderErrorKind::Server.retryable());
        assert!(!ProviderErrorKind::BadRequest.retryable());
        assert!(!ProviderErrorKind::Auth.retryable());
        assert!(!ProviderErrorKind::Cancelled.retryable());
        assert!(!ProviderErrorKind::Malformed.retryable());
    }

    #[tokio::test]
    async fn fake_provider_stream_contract() {
        let caps = ModelCapabilities::default();
        let fake = FakeProvider::with_script(
            "f",
            caps,
            vec![
                ScriptedResponse::Text("a".into()),
                ScriptedResponse::ToolCall {
                    id: "c1".into(),
                    name: "read_file".into(),
                    input: serde_json::json!({"path": "x"}),
                },
                ScriptedResponse::End,
            ],
        );
        let chunks: Vec<_> = fake.stream(req()).collect().await;
        assert_eq!(chunks.len(), 3);
        assert!(matches!(chunks[0], Ok(ProviderChunk::Text { .. })));
        assert!(matches!(chunks[1], Ok(ProviderChunk::ToolCall { .. })));
        assert_eq!(chunks[2], Ok(ProviderChunk::Done));
    }

    #[tokio::test]
    async fn fake_provider_dies_mid_stream() {
        let fake = FakeProvider::die_mid_stream("f", ModelCapabilities::default());
        let chunks: Vec<_> = fake.stream(req()).collect().await;
        assert!(chunks.len() == 2 && chunks[0].is_ok() && chunks[1].is_err());
        assert_eq!(
            chunks[1].as_ref().unwrap_err().kind,
            ProviderErrorKind::Network
        );
    }

    #[test]
    fn duplicate_instance_id_is_conflict_and_first_entry_wins() {
        let caps_a = ModelCapabilities {
            context: 1000,
            ..Default::default()
        };
        let caps_b = ModelCapabilities {
            context: 2000,
            ..Default::default()
        };
        let first = FakeProvider::with_script(
            "family",
            caps_a.clone(),
            vec![
                ScriptedResponse::Text("first".into()),
                ScriptedResponse::End,
            ],
        );
        let first: Arc<dyn Provider> = Arc::new(first);
        let second = FakeProvider::with_script(
            "family",
            caps_b,
            vec![
                ScriptedResponse::Text("second".into()),
                ScriptedResponse::End,
            ],
        );
        let mut reg = ProviderRegistry::new();
        assert!(reg.try_register(first.clone()).is_ok());
        // Same id, a DIFFERENT instance (fresh Arc over equal content) is a
        // typed Conflict and the FIRST registration stays untouched.
        let err = reg.try_register(Arc::new(second)).unwrap_err();
        assert_eq!(
            err.kind,
            faktor_core::error::ErrorKind::Conflict,
            "duplicate instance id is a Conflict"
        );
        assert_eq!(reg.len(), 1, "the first entry is never replaced");
        let caps = reg.capabilities("family", "m").unwrap();
        assert_eq!(
            caps.context, 1000,
            "capabilities come from the FIRST registration"
        );
        assert!(
            Arc::ptr_eq(&reg.get("family").unwrap(), &first),
            "the stored instance is exactly the first Arc, never a replacement"
        );
        // Same id, the SAME instance (a clone of the first Arc) is an
        // idempotent no-op: Ok, nothing changes, nothing is duplicated.
        assert!(reg.try_register(first.clone()).is_ok());
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.capabilities("family", "m").unwrap().context, 1000);
        assert!(
            Arc::ptr_eq(&reg.get("family").unwrap(), &first),
            "the idempotent re-registration kept the original instance"
        );
        // Distinct instance ids of the same family still coexist.
        let wrapped = InstanceProvider::wrap(
            Arc::new(FakeProvider::new("family", ModelCapabilities::default())),
            "second-instance",
        );
        assert!(reg.try_register(wrapped).is_ok());
        assert_eq!(reg.len(), 2);
    }

    #[test]
    fn hostile_and_case_variant_instance_ids_are_typed_refusals() {
        use faktor_core::error::ErrorKind;
        // Empty instance ids never enter the map.
        let mut reg = ProviderRegistry::new();
        let empty = Arc::new(FakeProvider::with_script(
            "",
            ModelCapabilities::default(),
            vec![ScriptedResponse::End],
        ));
        let err = reg.try_register(empty).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Malformed, "{err}");
        assert_eq!(reg.len(), 0, "the hostile registration never lands");
        // Oversized instance ids are refused with the typed Oversized kind.
        let huge = Arc::new(FakeProvider::with_script(
            &"x".repeat(MAX_PROVIDER_INSTANCE_ID_BYTES + 1),
            ModelCapabilities::default(),
            vec![ScriptedResponse::End],
        ));
        let err = reg.try_register(huge).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Oversized, "{err}");
        assert_eq!(reg.len(), 0);
        // The boundary is exact: MAX bytes is still a valid id.
        let ok = Arc::new(FakeProvider::with_script(
            &"y".repeat(MAX_PROVIDER_INSTANCE_ID_BYTES),
            ModelCapabilities::default(),
            vec![ScriptedResponse::End],
        ));
        assert!(reg.try_register(ok).is_ok());
        assert_eq!(reg.len(), 1);
        // An empty id must not be recoverable through the idempotent path
        // either: hostile ids are refused before any duplicate logic runs.
        assert!(reg
            .try_register(Arc::new(FakeProvider::with_script(
                "",
                ModelCapabilities::default(),
                vec![ScriptedResponse::End],
            )))
            .is_err());

        // Duplicate-key case variants: an id that differs from a registered
        // key ONLY by case is a typed Conflict; the FIRST registration is
        // kept and the case variant never lands.
        let mut reg = ProviderRegistry::new();
        let first: Arc<dyn Provider> = Arc::new(FakeProvider::with_script(
            "Corp-Proxy",
            ModelCapabilities {
                context: 3000,
                ..Default::default()
            },
            vec![ScriptedResponse::End],
        ));
        assert!(reg.try_register(first.clone()).is_ok());
        for hostile in ["corp-proxy", "CORP-PROXY", "cOrP-pRoXy"] {
            let variant = Arc::new(FakeProvider::with_script(
                hostile,
                ModelCapabilities {
                    context: 4000,
                    ..Default::default()
                },
                vec![ScriptedResponse::End],
            ));
            let err = reg.try_register(variant).unwrap_err();
            assert_eq!(err.kind, ErrorKind::Conflict, "{hostile:?}: {err}");
        }
        assert_eq!(reg.len(), 1, "no case variant ever registers");
        assert_eq!(reg.capabilities("Corp-Proxy", "m").unwrap().context, 3000);
        assert!(
            Arc::ptr_eq(&reg.get("Corp-Proxy").unwrap(), &first),
            "the FIRST registration survives every case-variant attack"
        );
        // A genuinely distinct id in the same family still registers.
        assert!(reg
            .try_register(InstanceProvider::wrap(
                Arc::new(FakeProvider::new(
                    "Corp-Proxy",
                    ModelCapabilities::default()
                )),
                "second-proxy",
            ))
            .is_ok());
        assert_eq!(reg.len(), 2);
        // Resolution stays exact-case: hostile lookups of the registered
        // canonical key are what the registry serves, and case-swapped
        // lookups miss (there is no silent canonicalization of lookups).
        assert!(reg.get("Corp-Proxy").is_some());
        assert!(reg.get("corp-proxy").is_none());
    }

    #[test]
    fn infallible_provider_registration_api_is_gone_from_the_crate_source() {
        // P0-41 compile proof: the infallible duplicate-accepting
        // `ProviderRegistry::register` shim (warn-on-conflict) no longer
        // exists anywhere in this crate's source. If a future wave re-adds
        // an infallible registration API, this test fails at compile/run
        // time by scanning the crate sources.
        let mut scanned = 0usize;
        let reg_sig = ["fn ", "register"].concat();
        let warn_marker = ["provider ", "registration ", "rejected"].concat();
        for entry in std::fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/src")).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            scanned += 1;
            let src = std::fs::read_to_string(&path).unwrap();
            for line in src.lines() {
                // A registration method for Arc<dyn Provider> that does NOT
                // return Result is the infallible footgun.
                if line.contains(&reg_sig) && line.contains("Arc<dyn Provider>") {
                    assert!(
                        line.contains("-> Result"),
                        "{}:{}: infallible provider registration API re-added: {line}",
                        path.display(),
                        line
                    );
                }
                // The warn-on-duplicate shim marker must never reappear.
                assert!(
                    !line.contains(&warn_marker),
                    "{}:{}: warn-on-duplicate registration shim re-added",
                    path.display(),
                    line
                );
            }
        }
        assert!(
            scanned >= 2,
            "the source scan must actually cover the provider crate files (found {scanned})"
        );
    }

    #[test]
    fn tokenizer_mapping_matrix_rows() {
        use TokenFamily as F;
        // o200k rows: gpt-4o/gpt-4.1/o1/o3/gpt-5 families. Specific
        // prefixes must win over the generic `gpt-4` cl100k row.
        for model in [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4o1",
            "gpt-4.1",
            "gpt-4.1-mini",
            "o1",
            "o1-mini",
            "o1-pro",
            "o3",
            "o3-mini",
            "gpt-5",
            "gpt-5-mini",
            "GPT-5",
            "gpt-5-codex",
        ] {
            let id = tokenizer_for(model, None);
            assert_eq!(
                id,
                TokenizerId::O200K_BASE,
                "{model} must map to o200k_base (got {id})"
            );
            assert_eq!(id.family, F::O200kBase);
            assert_eq!(id.version, 1, "all named rows are generation 1");
        }
        // cl100k rows: gpt-3.5 / gpt-4 (incl. turbo + 4.5 leftovers).
        for model in [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4-1106-preview",
            "gpt-4.5",
            "gpt-3.5",
            "gpt-3.5-turbo",
            "GPT-4",
        ] {
            assert_eq!(
                tokenizer_for(model, None),
                TokenizerId::CL100K_BASE,
                "{model} must map to cl100k_base"
            );
        }
        // Anthropic / Gemini / Llama rows.
        for model in [
            "claude-3-5-sonnet",
            "claude-3-7-sonnet",
            "claude-sonnet-4-5",
            "claude-opus-4-1",
            "Claude-Opus-4-1",
            "claude-haiku-4-5",
        ] {
            assert_eq!(
                tokenizer_for(model, None),
                TokenizerId::ANTHROPIC,
                "{model} must map to anthropic"
            );
        }
        for model in [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-3",
            "Gemini-2.5-Pro",
        ] {
            assert_eq!(
                tokenizer_for(model, None),
                TokenizerId::GEMINI,
                "{model} must map to gemini"
            );
        }
        for model in [
            "llama-3.3-70b",
            "Llama-3.1-8B",
            "qwen3.8",
            "qwen3-coder",
            "qwen-2.5-72b",
        ] {
            assert_eq!(
                tokenizer_for(model, None),
                TokenizerId::LLAMA,
                "{model} must map to llama"
            );
        }
        // Slash-qualified routed model strings resolve by the last segment.
        assert_eq!(
            tokenizer_for("anthropic/claude-sonnet-4-5", None),
            TokenizerId::ANTHROPIC
        );
        assert_eq!(tokenizer_for("openai/gpt-5", None), TokenizerId::O200K_BASE);
        assert_eq!(tokenizer_for("ollama/qwen3.8", None), TokenizerId::LLAMA);
    }

    #[test]
    fn tokenizer_mapping_unknown_models_are_generic_and_never_upgraded() {
        // Unknown/empty/hostile model strings conservatively map to the
        // GenericEstimator fallback — a hint NEVER upgrades them to a named
        // family (an OpenAI-compatible endpoint is not an OpenAI tokenizer).
        for model in [
            "",
            "/",
            "default",
            "my-custom-model",
            "gpt",
            "gpt-x",
            "o",
            "o0",
            "xqwen",
            "gemma-2-9b",
            "mistral-large",
            "deepseek-chat",
            "deepseek-reasoner",
            "gpt5",
            "gpt_5",
            "😀-model",
        ] {
            let hint = Some("openai");
            assert_eq!(
                tokenizer_for(model, hint),
                TokenizerId::GENERIC_ESTIMATOR,
                "{model:?} must conservatively map to generic_estimator"
            );
        }
    }

    #[test]
    fn tokenizer_mapping_deepseek_rows_depend_on_the_deployment_hint() {
        // deepseek weights are llama-family ONLY when a llama-family runtime
        // (ollama/llama.cpp) serves them; the official deepseek API keeps
        // its own non-llama tokenizer → conservative generic fallback.
        for model in ["deepseek-chat", "deepseek-reasoner", "deepseek-v3"] {
            assert_eq!(
                tokenizer_for(model, None),
                TokenizerId::GENERIC_ESTIMATOR,
                "{model} on the official API is NOT llama-family"
            );
            assert_eq!(
                tokenizer_for(model, Some("deepseek")),
                TokenizerId::GENERIC_ESTIMATOR
            );
            assert_eq!(
                tokenizer_for(model, Some("ollama")),
                TokenizerId::LLAMA,
                "{model} under a llama-family runtime maps to llama"
            );
            assert_eq!(
                tokenizer_for(model, Some("local-llama-cpp")),
                TokenizerId::LLAMA
            );
        }
    }

    #[test]
    fn tokenizer_id_is_total_order_display_and_serde_stable() {
        use TokenFamily as F;
        // Derived Ord: variant declaration order, then version. A sorted
        // vec is deterministic across processes (cache keys rely on it).
        let mut ids = vec![
            TokenizerId::GENERIC_ESTIMATOR,
            TokenizerId {
                family: F::O200kBase,
                version: 2,
            },
            TokenizerId::CL100K_BASE,
            TokenizerId::LLAMA,
            TokenizerId::GEMINI,
            TokenizerId::ANTHROPIC,
            TokenizerId::O200K_BASE,
        ];
        let expected = vec![
            TokenizerId::O200K_BASE,
            TokenizerId {
                family: F::O200kBase,
                version: 2,
            },
            TokenizerId::CL100K_BASE,
            TokenizerId::ANTHROPIC,
            TokenizerId::GEMINI,
            TokenizerId::LLAMA,
            TokenizerId::GENERIC_ESTIMATOR,
        ];
        ids.sort();
        assert_eq!(ids, expected, "deterministic total order");
        assert!(
            TokenizerId::O200K_BASE
                < TokenizerId {
                    family: F::O200kBase,
                    version: 2,
                }
        );
        // Display.
        assert_eq!(TokenizerId::O200K_BASE.to_string(), "o200k_base@v1");
        assert_eq!(
            TokenizerId::GENERIC_ESTIMATOR.to_string(),
            "generic_estimator@v1"
        );
        assert_eq!(TokenFamily::Anthropic.to_string(), "anthropic");
        // Serde round-trips with the frozen wire names (snake_case).
        let json = serde_json::to_value(TokenizerId::O200K_BASE).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"family": "o200k_base", "version": 1})
        );
        assert_eq!(
            serde_json::from_value::<TokenizerId>(json).unwrap(),
            TokenizerId::O200K_BASE
        );
        assert_eq!(
            serde_json::from_value::<TokenizerId>(serde_json::json!(
                {"family": "cl100k_base", "version": 1}
            ))
            .unwrap(),
            TokenizerId::CL100K_BASE
        );
        // Hostile json: unknown family is a serde error, never a silent map.
        assert!(serde_json::from_value::<TokenizerId>(serde_json::json!(
            {"family": "not_a_family", "version": 1}
        ))
        .is_err());
        // Default is the conservative fallback.
        assert_eq!(TokenizerId::default(), TokenizerId::GENERIC_ESTIMATOR);
    }

    #[test]
    fn tokenizer_mapping_case_and_whitespace_hostile_rows() {
        // Case folding and trim are part of the mapping contract; whitespace
        // INSIDE the name is not stripped (hostile rows stay generic).
        assert_eq!(tokenizer_for("  GPT-5  ", None), TokenizerId::O200K_BASE);
        assert_eq!(
            tokenizer_for("\tclaude-sonnet-4", None),
            TokenizerId::ANTHROPIC
        );
        assert_eq!(tokenizer_for("gpt-5\n", None), TokenizerId::O200K_BASE);
        assert_eq!(tokenizer_for("gpt-4o\n ", None), TokenizerId::O200K_BASE);
        assert_eq!(
            tokenizer_for("gpt-4o", None),
            tokenizer_for(" GPT-4O\n", None),
            "leading whitespace, case and trailing newlines never change the identity"
        );
        assert_eq!(
            tokenizer_for("qwen3.8", None),
            tokenizer_for("ollama/qwen3.8", None),
            "the provider prefix before the last '/' never changes the identity"
        );
    }
}
